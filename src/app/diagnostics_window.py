"""Accessible Repair and Diagnostics window for local Ollama models."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from threading import Event

from PySide6.QtCore import (
    QObject,
    QProcess,
    QProcessEnvironment,
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
    QApplication,
    QDialog,
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
    compute_completed = Signal(str, str)

    def __init__(self) -> None:
        super().__init__()
        self._cancelled = Event()

    def cancel(self) -> None:
        self._cancelled.set()

    @Slot()
    def run(self) -> None:
        try:
            if self._cancelled.is_set():
                self.completed.emit([])
                return
            self.compute_completed.emit(*OllamaDiagnostics.compute_health())
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
    _TRACE_ROLE = Qt.ItemDataRole.UserRole + 1
    inspection_finished = Signal(bool)
    inspection_progress = Signal(int, int, str)

    def __init__(self, parent=None, *, auto_refresh: bool = True) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.setWindowTitle("AIBrain Repair and Diagnostics")
        self.resize(1280, 780)
        self.setMinimumSize(900, 620)
        self.setStyleSheet(stylesheet(load_colours()))
        self._repair_process: QProcess | None = None
        self._last_repair_status = "No repair command has started"
        self._diagnostics_thread: QThread | None = None
        self._diagnostics_worker: DiagnosticsWorker | None = None
        self._last_refresh_succeeded = False
        self._closing = False
        self._live_output = LiveOutputBuffer()
        self._repair_heartbeat = QTimer(self)
        self._repair_heartbeat.setInterval(15_000)
        self._repair_heartbeat.timeout.connect(self._report_repair_heartbeat)
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
        self.table.itemClicked.connect(self._open_table_item)
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
        self._refresh_subsystem_cards()
        self._gpu_probe.run()
        thread = QThread(self)
        worker = DiagnosticsWorker()
        worker.moveToThread(thread)
        worker.compute_completed.connect(self._compute_ready)
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
                    "N/A",
                    "",
                ]
            )
            self.table.addTopLevelItem(item)
        for diagnostic in diagnostics:
            status = (
                "Ready"
                if diagnostic.available
                else "Unsupported"
                if diagnostic.issue_code == "Unsupported model type"
                else "Needs repair"
            )
            item = QTreeWidgetItem(
                [
                    diagnostic.reference,
                    status,
                    diagnostic.reason,
                    "N/A" if diagnostic.available else "Open",
                    str(diagnostic.manifest_path),
                ]
            )
            item.setData(0, self._DIAGNOSTIC_ROLE, diagnostic)
            item.setData(
                3,
                self._TRACE_ROLE,
                "" if diagnostic.available else diagnostic.detail,
            )
            item.setToolTip(
                3,
                "No error trace is available"
                if diagnostic.available
                else "Click Open to view and copy the error trace",
            )
            item.setToolTip(4, "Click to reveal this manifest in Explorer")
            if not diagnostic.available:
                trace_font = item.font(3)
                trace_font.setUnderline(True)
                item.setFont(3, trace_font)
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
        if message.startswith("Hashing "):
            if message.endswith("(start)") or message.endswith("(complete)"):
                self._log("CHECK", f"{prefix} {message}")
            else:
                self._set_live_progress("CHECK", f"{prefix} {message}")
        else:
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
            not refreshing
            and bool(
                self.selected_diagnostic() and self.selected_diagnostic().can_repair
            )
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
        self.repair_button.setEnabled(
            bool(diagnostic and diagnostic.can_repair and not repair_running)
        )
        self.repair_button.setToolTip(
            "This model is valid but unsupported by the current AIBrain code; no automatic repair can change it"
            if diagnostic and diagnostic.issue_code == "Unsupported model type"
            else "Re-download the selected model or repair its llama.cpp backend"
        )
        self.remove_button.setEnabled(
            bool(diagnostic and diagnostic.can_remove_manifest and not repair_running)
        )
        self.refresh_button.setEnabled(not repair_running)

    def repair_selected(self) -> None:
        diagnostic = self.selected_diagnostic()
        if diagnostic is None or not diagnostic.can_repair:
            return
        self._live_output.clear()
        self.output.clear()
        if diagnostic.issue_code == "Backend incompatibility":
            program = sys.executable
            arguments = [
                str(PROJECT_ROOT / "cli" / "installer.py"),
                "--repair",
                "--repair-subsystem",
                "backend",
                "-y",
            ]
            action = "Updating llama.cpp only; Ollama model data is preserved"
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
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("AIBRAIN_EMBEDDED_REPAIR", "1")
        environment.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(environment)
        process.started.connect(self._repair_started)
        process.readyReadStandardOutput.connect(self._append_repair_output)
        process.errorOccurred.connect(self._repair_error)
        process.finished.connect(self._repair_finished)
        self._repair_process = process
        LOG.info("Starting diagnostics repair: %s", action)
        self._last_repair_status = action
        self._log("START", action)
        process.start()
        self._update_actions()

    @Slot()
    def _repair_started(self) -> None:
        self._repair_heartbeat.start()
        self._last_repair_status = (
            "Repair command started; streaming package-manager output"
        )
        self._log("RUN", self._last_repair_status)

    @Slot()
    def _report_repair_heartbeat(self) -> None:
        if self._repair_process is not None:
            seconds = self._repair_heartbeat.interval() // 1000
            self._log(
                "WAIT",
                f"No new command output for {seconds} seconds; last status: "
                f"{self._last_repair_status}",
            )

    def _append_repair_output(self) -> None:
        if self._repair_process is None:
            return
        output = bytes(self._repair_process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        if not output:
            return
        visible_lines = [
            line.strip()
            for line in normalize_process_output(output).splitlines()
            if any(character.isalnum() for character in line)
        ]
        if visible_lines:
            self._last_repair_status = visible_lines[-1]
        self._repair_heartbeat.start()
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
        self._live_output.current_line = ""
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {level:<7} {message}"
        self._live_output.feed(line + "\n")
        self.output.setPlainText(self._live_output.render())
        self.output.moveCursor(QTextCursor.MoveOperation.End)

    def _set_live_progress(self, level: str, message: str) -> None:
        """Update one hashing-progress row without persisting each percentage."""
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {level:<7} {message}"
        self._live_output.current_line = line
        self.output.setPlainText(self._live_output.render())
        self.output.moveCursor(QTextCursor.MoveOperation.End)

    def _repair_error(self, process_error: QProcess.ProcessError) -> None:
        if self._repair_process is not None:
            message = self._repair_process.errorString()
            self._log("ERROR", f"Repair command error: {message}")
            if process_error == QProcess.ProcessError.FailedToStart:
                self._repair_heartbeat.stop()
                self._repair_process = None
                self._update_actions()

    def _repair_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._append_repair_output()
        self._repair_heartbeat.stop()
        self._log(
            "DONE" if exit_code == 0 else "ERROR",
            f"Repair command finished with exit code {exit_code}",
        )
        self._repair_process = None
        LOG.info("Diagnostics repair command finished with exit code %s", exit_code)
        if exit_code == 0:
            OllamaDiagnostics.invalidate_validation_cache(PROJECT_ROOT)
        self._update_actions()
        self.refresh()

    def _open_table_item(self, item: QTreeWidgetItem, column: int) -> None:
        if column == 3:
            self._open_trace(item)
            return
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

    def _open_trace(self, item: QTreeWidgetItem) -> None:
        trace = item.data(3, self._TRACE_ROLE)
        if not isinstance(trace, str) or not trace:
            return
        dialog = QDialog(self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setWindowTitle("Raw diagnostic error trace")
        dialog.resize(760, 440)
        layout = QVBoxLayout(dialog)
        viewer = QPlainTextEdit()
        viewer.setAccessibleName("Copyable raw diagnostic error trace")
        viewer.setPlainText(trace)
        viewer.setReadOnly(True)
        layout.addWidget(viewer)
        actions = QHBoxLayout()
        copy = QPushButton("Copy")
        copy.setObjectName("copyTraceButton")
        copy.clicked.connect(lambda: QApplication.clipboard().setText(trace))
        close = QPushButton("Close")
        close.clicked.connect(dialog.close)
        actions.addStretch(1)
        actions.addWidget(copy)
        actions.addWidget(close)
        layout.addLayout(actions)
        dialog.open()

    @Slot(str, str)
    def _compute_ready(self, state: str, detail: str) -> None:
        labels = self.subsystem_cards["GPU / CUDA"]
        labels[0].setText(state)
        labels[1].setText(detail)
        self._log("CUDA", f"{state}: {detail}")

    @Slot(str, str)
    def _opengl_ready(self, vendor: str, renderer: str) -> None:
        labels = self.subsystem_cards["OpenGL rendering"]
        labels[0].setText("Ready")
        labels[1].setText(f"{vendor} · {renderer}")
        self._log("OPENGL", f"Rendering: {vendor} · {renderer}")

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

    def _refresh_subsystem_cards(self) -> None:
        checks = OllamaDiagnostics().subsystem_health()
        checks["GPU / CUDA"] = (
            "Checking",
            "Testing CUDA compute and llama.cpp separately from OpenGL rendering.",
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
            self._repair_heartbeat.stop()
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
