"""Accessible Repair and Diagnostics window for local Ollama models."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from threading import Event

from PySide6.QtCore import (
    QObject,
    QProcess,
    QSettings,
    QThread,
    QTimer,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices, QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .dashboard import metric_card
from .loading_window import GpuProbe
from .theme import load_colours, stylesheet
from ..models.diagnostics import ModelDiagnostic, OllamaDiagnostics
from ..utils.console_ui import strip_ansi
from ..utils.gpu import discover_render_adapters, should_prefer_high_performance_gpu
from ..utils.logging import PROJECT_ROOT

LOG = logging.getLogger(__name__)


def normalize_process_output(output: str) -> str:
    """Convert terminal-oriented process output into readable Qt text."""
    return strip_ansi(output).replace("\r\n", "\n").replace("\r", "\n")


class LiveOutputBuffer:
    """Keep completed lines separate from a carriage-return progress line."""

    def __init__(self) -> None:
        self.completed_lines: list[str] = []
        self.current_line = ""

    def clear(self) -> None:
        self.completed_lines.clear()
        self.current_line = ""

    def feed(self, output: str) -> None:
        for character in strip_ansi(output):
            if character == "\r":
                self.current_line = ""
            elif character == "\n":
                self.completed_lines.append(self.current_line)
                self.current_line = ""
            else:
                self.current_line += character
        del self.completed_lines[:-499]

    def render(self) -> str:
        return "\n".join((*self.completed_lines, self.current_line)).rstrip("\n")


class DiagnosticsWorker(QObject):
    """Run expensive backend validation away from the GUI thread."""

    completed = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int, str)

    def __init__(self) -> None:
        super().__init__()
        self._cancelled = Event()

    def cancel(self) -> None:
        self._cancelled.set()

    @Slot()
    def run(self) -> None:
        try:
            LOG.info("Starting background Ollama model diagnostics")
            diagnostics = OllamaDiagnostics().inspect(
                verify_backend=True,
                cancelled=self._cancelled,
                progress=self.progress.emit,
            )
            LOG.info("Background Ollama model diagnostics completed (%s models)", len(diagnostics))
            self.completed.emit(diagnostics)
        except Exception as exc:
            LOG.exception("Background Ollama model diagnostics failed")
            self.failed.emit(str(exc))


class DiagnosticsWindow(QMainWindow):
    """Expose model validation findings and user-initiated repair actions."""

    _DIAGNOSTIC_ROLE = Qt.ItemDataRole.UserRole
    inspection_finished = Signal(bool)
    inspection_progress = Signal(int, int, str)

    def __init__(self, parent=None, *, auto_refresh: bool = True) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.setWindowTitle("AIBrain Repair and Diagnostics")
        self.resize(1280, 780)
        self.setMinimumSize(900, 620)
        self.setStyleSheet(stylesheet(load_colours()))
        self._repair_process: QProcess | None = None
        self._diagnostics_thread: QThread | None = None
        self._diagnostics_worker: DiagnosticsWorker | None = None
        self._last_refresh_succeeded = False
        self._closing = False
        self._live_output = LiveOutputBuffer()
        self._gpu_probe = GpuProbe(self)
        self._build()
        self._gpu_probe.completed.connect(self._opengl_ready)
        self._gpu_probe.failed.connect(self._opengl_failed)
        if auto_refresh:
            self.refresh()

    def _build(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 20, 24, 18)
        layout.setSpacing(12)
        self.setCentralWidget(page)
        heading = QLabel("Repair and Diagnostics")
        heading.setObjectName("title")
        layout.addWidget(heading)
        help_text = QLabel(
            "Live checks separate the human cause from the raw failure. Select a model to repair it, "
            "or double-click its manifest path to reveal it in Explorer."
        )
        help_text.setWordWrap(True)
        help_text.setObjectName("muted")
        layout.addWidget(help_text)
        self.subsystem_cards = {}
        cards = QGridLayout()
        cards.setHorizontalSpacing(10)
        cards.setVerticalSpacing(10)
        for index, title in enumerate(
            (
                "Models",
                "GPU / CUDA",
                "OpenGL rendering",
                "Python environment",
                "pip / libraries",
                "Native DLLs",
                ".cache",
            )
        ):
            card, value, detail = metric_card(title, "Checking…")
            cards.addWidget(card, index // 4, index % 4)
            self.subsystem_cards[title] = (value, detail)
        layout.addLayout(cards)
        self.system_status = QLabel()
        self.system_status.setObjectName("muted")
        self.system_status.setWordWrap(True)
        self.system_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.system_status)

        self.table = QTreeWidget()
        self.table.setAccessibleName("Ollama model diagnostics")
        self.table.setHeaderLabels(
            ["Model", "Status", "Human reason", "Raw error trace", "Manifest"]
        )
        self.table.setRootIsDecorated(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setUniformRowHeights(False)
        self.table.setWordWrap(True)
        self.table.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.table.itemSelectionChanged.connect(self._update_actions)
        self.table.itemClicked.connect(self._open_manifest_item)
        header = self.table.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)

        self.output = QPlainTextEdit()
        self.output.setAccessibleName("Repair command output")
        self.output.setReadOnly(True)
        self.output.setPlaceholderText("Live diagnostic and repair events appear here.")
        self.output.setMaximumBlockCount(500)
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.table)
        splitter.addWidget(self.output)
        splitter.setSizes([470, 180])
        layout.addWidget(splitter, 1)

        actions = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh")
        self.repair_button = QPushButton("Repair selected")
        self.remove_button = QPushButton("Remove stale manifest")
        self.cache_button = QPushButton("Repair validation cache")
        self.open_logs_button = QPushButton("Open logs")
        self.refresh_button.setToolTip("Re-run health checks without deleting any data")
        self.repair_button.setToolTip(
            "Re-download only the selected model through Ollama"
        )
        self.remove_button.setToolTip(
            "Permanently remove only the selected invalid manifest after confirmation"
        )
        self.cache_button.setToolTip(
            "Rebuild only disposable validation cache data after confirmation"
        )
        self.open_logs_button.setToolTip(
            "Open the feature-specific console and crash logs"
        )
        self.refresh_button.clicked.connect(self.refresh)
        self.repair_button.clicked.connect(self.repair_selected)
        self.remove_button.clicked.connect(self.remove_selected)
        self.cache_button.clicked.connect(self.repair_validation_cache)
        self.open_logs_button.clicked.connect(self.open_logs)
        for button in (
            self.refresh_button,
            self.repair_button,
            self.remove_button,
            self.cache_button,
            self.open_logs_button,
        ):
            actions.addWidget(button)
        actions.addStretch(1)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        actions.addWidget(close_button)
        layout.addLayout(actions)
        self._update_actions()

    def refresh(self) -> None:
        if self._closing or self._diagnostics_thread is not None:
            LOG.debug(
                "Ignoring diagnostics refresh request (closing=%s, inspection_active=%s)",
                self._closing,
                self._diagnostics_thread is not None,
            )
            return
        self.table.clear()
        self._log("SCAN", "Starting model, CUDA, OpenGL, runtime, and cache checks")
        self._last_refresh_succeeded = False
        adapters = discover_render_adapters()
        adapter_text = (
            ", ".join(adapter.name for adapter in adapters)
            if adapters
            else "No display adapters could be queried"
        )
        preference = (
            "high-performance GPU requested"
            if should_prefer_high_performance_gpu()
            else "Windows system-default GPU requested"
        )
        self.system_status.setText(
            f"Render diagnostics: {preference}. Detected adapters: {adapter_text}."
        )
        self.system_status.setText(
            self.system_status.text()
            + "\nChecking local GGUF compatibility in the background…"
        )
        self._set_refreshing(True)
        self._refresh_subsystem_cards(adapters)
        self._gpu_probe.run()
        thread = QThread(self)
        worker = DiagnosticsWorker()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.completed.connect(self._diagnostics_ready)
        worker.failed.connect(self._diagnostics_failed)
        worker.progress.connect(self._diagnostics_progress)
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
            item = QTreeWidgetItem(
                [
                    "No local manifests found",
                    "Info",
                    "No model installed",
                    "Install a model with ollama pull",
                    "",
                ]
            )
            self.table.addTopLevelItem(item)
        for diagnostic in diagnostics:
            status = "Ready" if diagnostic.available else "Needs repair"
            item = QTreeWidgetItem(
                [
                    diagnostic.reference,
                    status,
                    diagnostic.reason,
                    diagnostic.detail,
                    str(diagnostic.manifest_path),
                ]
            )
            item.setData(0, self._DIAGNOSTIC_ROLE, diagnostic)
            item.setToolTip(3, diagnostic.detail)
            item.setToolTip(4, "Click to reveal this manifest in Explorer")
            link_font = item.font(4)
            link_font.setUnderline(True)
            item.setFont(4, link_font)
            self.table.addTopLevelItem(item)
        self._last_refresh_succeeded = True
        broken = sum(not item.available for item in diagnostics)
        model_card = self.subsystem_cards["Models"]
        model_card[0].setText("Ready" if not broken else f"{broken} issue(s)")
        model_card[1].setText(
            f"{len(diagnostics)} manifest(s) inspected with llama.cpp compatibility checks."
        )
        self._log(
            "DONE",
            f"Model inspection finished: {len(diagnostics)} checked, {broken} need attention",
        )

    @Slot(int, int, str)
    def _diagnostics_progress(self, current: int, total: int, message: str) -> None:
        prefix = f"{current}/{total}" if total else "…"
        self.system_status.setText(f"Model compatibility check {prefix}: {message}")
        self._log("CHECK", f"{prefix} {message}")
        self.inspection_progress.emit(current, total, message)

    @Slot(str)
    def _diagnostics_failed(self, message: str) -> None:
        LOG.error("Unable to inspect Ollama models: %s", message)
        self._log("ERROR", message)
        QMessageBox.critical(self, "Diagnostics error", message)

    @Slot()
    def _diagnostics_finished(self) -> None:
        self._diagnostics_thread = None
        self._diagnostics_worker = None
        self._set_refreshing(False)
        self._update_actions()
        if self._closing:
            LOG.info("Diagnostics window closed while background inspection was stopping")
            QTimer.singleShot(0, self.close)
            return
        LOG.info("Diagnostics inspection cycle finished (healthy=%s)", self._last_refresh_succeeded)
        self.inspection_finished.emit(self._last_refresh_succeeded)

    def _set_refreshing(self, refreshing: bool) -> None:
        self.refresh_button.setEnabled(not refreshing)
        self.repair_button.setEnabled(
            not refreshing and self.selected_diagnostic() is not None
        )
        self.remove_button.setEnabled(
            not refreshing and bool(self.selected_diagnostic())
        )

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
        self.remove_button.setEnabled(
            bool(diagnostic and diagnostic.can_remove_manifest and not repair_running)
        )
        self.refresh_button.setEnabled(not repair_running)

    def repair_selected(self) -> None:
        diagnostic = self.selected_diagnostic()
        if diagnostic is None:
            return
        self._live_output.clear()
        self.output.clear()
        if diagnostic.issue_code == "Backend incompatibility":
            program = sys.executable
            arguments = [
                str(PROJECT_ROOT / "cli" / "installer.py"),
                "--repair",
                "-y",
            ]
            action = "Reinstalling the llama.cpp backend; Ollama model data is preserved"
        else:
            confirmation = QMessageBox.question(
                self,
                "Repair selected model",
                "AIBrain will move the exact invalid artifact into a recoverable quarantine, then ask Ollama "
                f"to download a verified replacement for {diagnostic.reference}. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if confirmation != QMessageBox.StandardButton.Yes:
                return
            try:
                quarantined = OllamaDiagnostics.quarantine_for_redownload(diagnostic)
            except OSError as exc:
                QMessageBox.critical(self, "Repair preparation failed", str(exc))
                return
            if quarantined is not None:
                self._log("BACKUP", f"Moved invalid artifact to {quarantined}")
            program = "ollama"
            arguments = ["pull", diagnostic.reference]
            action = f"Downloading a verified replacement for {diagnostic.reference}"
        process = QProcess(self)
        process.setProgram(program)
        process.setArguments(arguments)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._append_repair_output)
        process.errorOccurred.connect(self._repair_error)
        process.finished.connect(self._repair_finished)
        self._repair_process = process
        LOG.info("Starting diagnostics repair: %s", action)
        self._live_output.feed(f"Starting: {action}\n")
        self.output.setPlainText(self._live_output.render())
        process.start()
        self._update_actions()

    def _append_repair_output(self) -> None:
        if self._repair_process is None:
            return
        output = bytes(self._repair_process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        self._live_output.feed(output)
        self.output.setPlainText(self._live_output.render())
        self.output.moveCursor(QTextCursor.MoveOperation.End)

    def _log(self, level: str, message: str) -> None:
        LOG.log(
            {
                "ERROR": logging.ERROR,
                "WARN": logging.WARNING,
            }.get(level, logging.INFO),
            "%s: %s",
            level,
            message,
        )
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {level:<7} {message}"
        self._live_output.feed(line + "\n")
        self.output.setPlainText(self._live_output.render())
        self.output.moveCursor(QTextCursor.MoveOperation.End)

    def _repair_error(self, _error: QProcess.ProcessError) -> None:
        if self._repair_process is not None:
            LOG.error("Diagnostics repair command error: %s", self._repair_process.errorString())
            self._live_output.feed(
                f"Repair command error: {self._repair_process.errorString()}\n"
            )
            self.output.setPlainText(self._live_output.render())

    def _repair_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._append_repair_output()
        self._live_output.feed(f"Repair command finished with exit code {exit_code}.\n")
        self.output.setPlainText(self._live_output.render())
        self._repair_process = None
        LOG.info("Diagnostics repair command finished with exit code %s", exit_code)
        if exit_code == 0:
            OllamaDiagnostics.invalidate_validation_cache(PROJECT_ROOT)
        self._update_actions()
        self.refresh()

    def _open_manifest_item(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 4:
            return
        diagnostic = item.data(0, self._DIAGNOSTIC_ROLE)
        if not isinstance(diagnostic, ModelDiagnostic):
            return
        path = diagnostic.manifest_path
        if path.is_file() and QProcess.startDetached(
            "explorer.exe", ["/select,", str(path)]
        ):
            self._log("OPEN", f"Revealed manifest in Explorer: {path}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))

    @Slot(str, str)
    def _opengl_ready(self, vendor: str, renderer: str) -> None:
        labels = self.subsystem_cards["OpenGL rendering"]
        labels[0].setText("Ready")
        labels[1].setText(f"{vendor} · {renderer}")
        self._log("OPENGL", f"Ready: {vendor} · {renderer}")

    @Slot(str)
    def _opengl_failed(self, message: str) -> None:
        labels = self.subsystem_cards["OpenGL rendering"]
        labels[0].setText("Needs repair")
        labels[1].setText(message)
        self._log("OPENGL", message)

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
        remove = confirmation.addButton(
            "Remove stale manifest", QMessageBox.ButtonRole.DestructiveRole
        )
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
        self.output.appendPlainText(
            f"Removed stale manifest: {diagnostic.manifest_path}"
        )
        self.refresh()

    def repair_validation_cache(self) -> None:
        confirmation = QMessageBox.question(
            self,
            "Repair validation cache",
            "Rebuild disposable validation-cache entries only? Model blobs, DLLs, and other cache data are preserved.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if confirmation != QMessageBox.StandardButton.Yes:
            return
        try:
            target = OllamaDiagnostics.invalidate_validation_cache(PROJECT_ROOT)
        except (OSError, ValueError) as exc:
            LOG.exception("Could not rebuild validation cache")
            QMessageBox.critical(
                self, "Validation cache repair failed", f"REASON: {exc}"
            )
            return
        self.output.appendPlainText(f"Rebuilt validation cache: {target}")
        self.refresh()

    @staticmethod
    def open_logs() -> None:
        logs = PROJECT_ROOT / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(logs)))

    def _refresh_subsystem_cards(self, adapters: list[object]) -> None:
        checks = OllamaDiagnostics().subsystem_health()
        checks["GPU / CUDA"] = (
            "Ready" if adapters else "Info",
            "Adapter discovery is a preference signal; actual OpenGL is verified at launch.",
        )
        mismatch = str(QSettings().value("opengl_gpu_mismatch_reason", ""))
        checks["OpenGL rendering"] = (
            "Needs repair" if mismatch else "Checking",
            mismatch
            or "The main window records the actual OpenGL vendor and renderer.",
        )
        for title, (value, detail) in checks.items():
            labels = self.subsystem_cards.get(title)
            if labels is not None:
                labels[0].setText(value)
                labels[1].setText(detail)

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
