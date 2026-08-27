"""Accessible Repair and Diagnostics window for local Ollama models."""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QProcess, Qt
from PySide6.QtGui import QDesktopServices, QTextCursor
from PySide6.QtCore import QUrl
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
from ..utils.logging import PROJECT_ROOT


LOG = logging.getLogger(__name__)


class DiagnosticsWindow(QDialog):
    """Expose model validation findings and user-initiated repair actions."""

    _DIAGNOSTIC_ROLE = Qt.ItemDataRole.UserRole

    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.setWindowTitle("AIBrain Repair and Diagnostics")
        self.setMinimumSize(780, 500)
        self._repair_process: QProcess | None = None
        self._build()
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
        self.table.clear()
        try:
            diagnostics = OllamaDiagnostics().inspect()
        except OSError as exc:
            LOG.exception("Unable to inspect Ollama models")
            QMessageBox.critical(self, "Diagnostics error", str(exc))
            return
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
        self.table.resizeColumnToContents(0)
        self.table.resizeColumnToContents(1)
        self._update_actions()

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
        output = bytes(self._repair_process.readAllStandardOutput()).decode("utf-8", errors="replace")
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
        answer = QMessageBox.question(
            self,
            "Remove stale manifest",
            f"Remove this invalid manifest only?\n\n{diagnostic.manifest_path}\n\nShared model blobs will not be deleted.",
            QMessageBox.StandardButton.Remove | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer is not QMessageBox.StandardButton.Remove:
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
