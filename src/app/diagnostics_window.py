"""Accessible Repair and Diagnostics window for local Ollama models."""
from __future__ import annotations

import logging
from threading import Event

from PySide6.QtCore import QObject, QProcess, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices, QTextCursor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from ..models.diagnostics import ModelDiagnostic, OllamaDiagnostics
from ..utils.gpu import discover_render_adapters, should_prefer_high_performance_gpu
from ..utils.logging import PROJECT_ROOT
from ..utils.console_ui import strip_ansi

LOG = logging.getLogger(__name__)


def normalize_process_output(output: str) -> str:
    """Convert terminal-oriented process output into readable Qt text."""
    return strip_ansi(output).replace("\r\n", "\n").replace("\r", "\n")


class DiagnosticsWorker(QObject):
    """Run expensive backend validation away from the GUI thread."""

    completed = Signal(object)
    failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._cancelled = Event()

    def cancel(self) -> None:
        self._cancelled.set()

    @Slot()
    def run(self) -> None:
        try:
            self.completed.emit(OllamaDiagnostics().inspect(verify_backend=True, cancelled=self._cancelled))
        except OSError as exc:
            self.failed.emit(str(exc))


class DiagnosticsWindow(QDialog):
    """Expose model validation findings and user-initiated repair actions."""

    _DIAGNOSTIC_ROLE = Qt.ItemDataRole.UserRole
    inspection_finished = Signal(bool)

    def __init__(self, parent=None, *, auto_refresh: bool = True) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.setWindowTitle("AIBrain Repair and Diagnostics")
        self.setMinimumSize(780, 500)
        self._repair_process: QProcess | None = None
        self._diagnostics_thread: QThread | None = None
        self._diagnostics_worker: DiagnosticsWorker | None = None
        self._last_refresh_succeeded = False
        self._closing = False
        self._build()
        if auto_refresh:
            self.refresh()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        heading = QLabel("Repair and Diagnostics")
        heading.setObjectName("title")
        layout.addWidget(heading)
        help_text = QLabel(
            "Validation reports every local Ollama manifest. Repair re-downloads the selected model. "
            "Remove stale deletes only the selected invalid manifest, never shared blob data."
        )
        help_text.setWordWrap(True)
        help_text.setObjectName("muted")
        layout.addWidget(help_text)
        self.system_status = QLabel()
        self.system_status.setObjectName("muted")
        self.system_status.setWordWrap(True)
        self.system_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.system_status)

        self.table = QTreeWidget()
        self.table.setAccessibleName("Ollama model diagnostics")
        self.table.setHeaderLabels(["Model", "Status", "Details", "Manifest"])
        self.table.setRootIsDecorated(False)
        self.table.setAlternatingRowColors(True)
        self.table.itemSelectionChanged.connect(self._update_actions)
        layout.addWidget(self.table, 1)

        self.output = QPlainTextEdit()
        self.output.setAccessibleName("Repair command output")
        self.output.setReadOnly(True)
        self.output.setPlaceholderText("Repair command output appears here.")
        self.output.setMaximumBlockCount(500)
        self.output.setFixedHeight(110)
        layout.addWidget(self.output)

        actions = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh")
        self.repair_button = QPushButton("Repair selected")
        self.remove_button = QPushButton("Remove stale manifest")
        self.open_logs_button = QPushButton("Open logs")
        self.refresh_button.clicked.connect(self.refresh)
        self.repair_button.clicked.connect(self.repair_selected)
        self.remove_button.clicked.connect(self.remove_selected)
        self.open_logs_button.clicked.connect(self.open_logs)
        for button in (self.refresh_button, self.repair_button, self.remove_button, self.open_logs_button):
            actions.addWidget(button)
        actions.addStretch(1)
        layout.addLayout(actions)
        close_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close_box.rejected.connect(self.reject)
        layout.addWidget(close_box)
        self._update_actions()

    def refresh(self) -> None:
        if self._closing or self._diagnostics_thread is not None:
            return
        self.table.clear()
        self._last_refresh_succeeded = False
        adapters = discover_render_adapters()
        adapter_text = ", ".join(
            adapter.name for adapter in adapters) if adapters else "No display adapters could be queried"
        preference = "high-performance GPU requested" if should_prefer_high_performance_gpu() else "Windows system-default GPU requested"
        self.system_status.setText(f"Render diagnostics: {preference}. Detected adapters: {adapter_text}.")
        self.system_status.setText(self.system_status.text() + "\nChecking local GGUF compatibility in the background…")
        self._set_refreshing(True)
        thread = QThread(self)
        worker = DiagnosticsWorker()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.completed.connect(self._diagnostics_ready)
        worker.failed.connect(self._diagnostics_failed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._diagnostics_finished)
        self._diagnostics_thread = thread
        self._diagnostics_worker = worker
        thread.start()

    @Slot(object)
    def _diagnostics_ready(self, result: object) -> None:
        diagnostics = result if isinstance(result, list) else []
        if not diagnostics:
            item = QTreeWidgetItem(["No local manifests found", "Info", "Install a model with ollama pull", ""])
            self.table.addTopLevelItem(item)
        for diagnostic in diagnostics:
            status = "Ready" if diagnostic.available else "Needs repair"
            item = QTreeWidgetItem(
                [diagnostic.reference, status, diagnostic.detail, str(diagnostic.manifest_path)]
            )
            item.setData(0, self._DIAGNOSTIC_ROLE, diagnostic)
            self.table.addTopLevelItem(item)
        self._last_refresh_succeeded = True
        self.table.resizeColumnToContents(0)
        self.table.resizeColumnToContents(1)

    @Slot(str)
    def _diagnostics_failed(self, message: str) -> None:
        LOG.error("Unable to inspect Ollama models: %s", message)
        QMessageBox.critical(self, "Diagnostics error", message)

    @Slot()
    def _diagnostics_finished(self) -> None:
        self._diagnostics_thread = None
        self._diagnostics_worker = None
        self._set_refreshing(False)
        self._update_actions()
        if self._closing:
            QTimer.singleShot(0, self.close)
            return
        self.inspection_finished.emit(self._last_refresh_succeeded)

    def _set_refreshing(self, refreshing: bool) -> None:
        self.refresh_button.setEnabled(not refreshing)
        self.repair_button.setEnabled(not refreshing and self.selected_diagnostic() is not None)
        self.remove_button.setEnabled(not refreshing and bool(self.selected_diagnostic()))

    def selected_diagnostic(self) -> ModelDiagnostic | None:
        item = self.table.currentItem()
        if item is None:
            return None
        data = item.data(0, self._DIAGNOSTIC_ROLE)
        return data if isinstance(data, ModelDiagnostic) else None

    def _update_actions(self) -> None:
        diagnostic = self.selected_diagnostic()
        repair_running = self._repair_process is not None
        self.repair_button.setEnabled(diagnostic is not None and not repair_running)
        self.remove_button.setEnabled(bool(diagnostic and diagnostic.can_remove_manifest and not repair_running))
        self.refresh_button.setEnabled(not repair_running)

    def repair_selected(self) -> None:
        diagnostic = self.selected_diagnostic()
        if diagnostic is None:
            return
        process = QProcess(self)
        process.setProgram("ollama")
        process.setArguments(["pull", diagnostic.reference])
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._append_repair_output)
        process.errorOccurred.connect(self._repair_error)
        process.finished.connect(self._repair_finished)
        self._repair_process = process
        self.output.clear()
        self.output.appendPlainText(f"Starting: ollama pull {diagnostic.reference}")
        process.start()
        self._update_actions()

    def _append_repair_output(self) -> None:
        if self._repair_process is None:
            return
        output = normalize_process_output(bytes(self._repair_process.readAllStandardOutput()).decode("utf-8", errors="replace"))
        self.output.moveCursor(QTextCursor.MoveOperation.End)
        self.output.insertPlainText(output)

    def _repair_error(self, _error: QProcess.ProcessError) -> None:
        if self._repair_process is not None:
            self.output.appendPlainText(f"Repair command error: {self._repair_process.errorString()}")

    def _repair_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._append_repair_output()
        self.output.appendPlainText(f"Repair command finished with exit code {exit_code}.")
        self._repair_process = None
        self._update_actions()
        self.refresh()

    def remove_selected(self) -> None:
        diagnostic = self.selected_diagnostic()
        if diagnostic is None or not diagnostic.can_remove_manifest:
            return
        confirmation = QMessageBox(self)
        confirmation.setIcon(QMessageBox.Icon.Warning)
        confirmation.setWindowTitle("Remove stale manifest")
        confirmation.setText(
            f"Remove this invalid manifest only?\n\n{diagnostic.manifest_path}\n\nShared model blobs will not be deleted."
        )
        remove = confirmation.addButton("Remove stale manifest", QMessageBox.ButtonRole.DestructiveRole)
        confirmation.addButton(QMessageBox.StandardButton.Cancel)
        confirmation.exec()
        if confirmation.clickedButton() is not remove:
            return
        try:
            OllamaDiagnostics.remove_stale_manifest(diagnostic)
        except OSError as exc:
            LOG.exception("Could not remove stale manifest")
            QMessageBox.critical(self, "Removal failed", str(exc))
            return
        self.output.appendPlainText(f"Removed stale manifest: {diagnostic.manifest_path}")
        self.refresh()

    @staticmethod
    def open_logs() -> None:
        logs = PROJECT_ROOT / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(logs)))

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._repair_process is not None:
            self._repair_process.kill()
            self._repair_process.waitForFinished(2_000)
        if self._diagnostics_worker is not None:
            self._diagnostics_worker.cancel()
        if self._diagnostics_thread is not None:
            # Do not wait from closeEvent: a backend load can take longer than
            # an event-loop turn. Hide now and close for real after the worker
            # returns and its QThread has finished.
            self._closing = True
            self._diagnostics_thread.quit()
            self.hide()
            event.ignore()
            return
        super().closeEvent(event)
