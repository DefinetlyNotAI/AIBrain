"""Coordinate asynchronous AIBrain desktop startup."""

from __future__ import annotations

import logging

from PySide6.QtCore import QCoreApplication, QObject, QThread, QTimer, Slot
from PySide6.QtWidgets import QApplication

from src.app.loading_window import GpuProbe, LoadingWindow
from src.app.main_window import MainWindow
from src.models.model_info import ModelInfo
from src.models.model_validator import StartupWorker
from src.utils.gpu import GPU_RELAUNCH_EXIT_CODE, can_request_gpu_relaunch


LOG = logging.getLogger(__name__)


class StartupCoordinator(QObject):
    """Coordinate model validation, GPU probing, and main-window startup."""

    def __init__(
        self,
        app: QApplication,
        loading: LoadingWindow,
        startup_thread: QThread,
        startup_worker: StartupWorker,
        *,
        prefer_high_performance_gpu: bool,
    ) -> None:
        """Initialize the desktop startup coordinator."""
        super().__init__(app)

        self._loading = loading
        self._startup_thread = startup_thread
        self._startup_worker = startup_worker
        self._prefer_high_performance_gpu = prefer_high_performance_gpu

        self._models: list[ModelInfo] | None = None
        self._main_window: MainWindow | None = None

        self._models_finished = False
        self._gpu_checked = False
        self._gpu_relaunch_requested = False
        self._main_window_started = False
        self._stopping = False

        self._shutdown_exit_code = 0

    @property
    def main_window(self) -> MainWindow | None:
        """Return the active main window, if startup has completed."""
        return self._main_window

    @property
    def stopping(self) -> bool:
        """Return whether desktop startup is currently stopping."""
        return self._stopping

    def connect_signals(self, gpu_probe: GpuProbe) -> None:
        """Connect startup workers, probes, and loader signals."""
        self._startup_thread.started.connect(self._startup_worker.run)

        self._startup_worker.progress.connect(self._loading.set_progress)
        self._startup_worker.finished.connect(self.models_ready)
        self._startup_worker.finished.connect(self._startup_thread.quit)
        self._startup_worker.failed.connect(self.show_startup_error)
        self._startup_worker.failed.connect(self._startup_thread.quit)

        gpu_probe.completed.connect(self.gpu_ready)
        gpu_probe.failed.connect(self.gpu_probe_failed)

        self._startup_thread.finished.connect(
            self._startup_worker.deleteLater
        )
        self._startup_thread.finished.connect(
            self.startup_thread_finished
        )

        self._loading.cancelled.connect(self.cancel_startup)

    @Slot()
    def start(self) -> None:
        """Start model validation unless startup is already stopping."""
        if self._stopping or self._startup_thread.isRunning():
            return

        self._startup_thread.start()

    @Slot(object)
    def models_ready(self, models: object) -> None:
        """Store the validated model collection returned by the worker."""
        if isinstance(models, list):
            self._models = [
                model
                for model in models
                if isinstance(model, ModelInfo)
            ]
        else:
            self._models = []

    @Slot(str, str)
    def gpu_ready(self, vendor: str, renderer: str) -> None:
        """Handle successful OpenGL adapter detection."""
        adapter = f"{vendor} {renderer}"
        is_nvidia = "nvidia" in adapter.lower()

        self._loading.set_progress(
            0,
            1,
            f"OpenGL adapter: {vendor} - {renderer}",
        )

        if (
            self._prefer_high_performance_gpu
            and not is_nvidia
            and can_request_gpu_relaunch()
        ):
            LOG.warning(
                "Loader detected a non-NVIDIA OpenGL adapter: "
                "vendor=%s renderer=%s. Restarting before main UI.",
                vendor,
                renderer,
            )

            self._gpu_relaunch_requested = True
            self._stop_startup(exit_code=GPU_RELAUNCH_EXIT_CODE)
            return

        self._gpu_checked = True
        self._show_main_when_ready()

    @Slot(str)
    def gpu_probe_failed(self, message: str) -> None:
        """Continue startup when OpenGL adapter probing fails."""
        LOG.warning("%s", message)

        self._loading.set_progress(0, 1, message)
        self._gpu_checked = True

        self._show_main_when_ready()

    @Slot()
    def cancel_startup(self) -> None:
        """Cancel desktop startup and stop the validation worker."""
        LOG.info(
            "Cancelling desktop startup and waiting for validation to stop"
        )

        self._stop_startup()

    @Slot()
    def startup_thread_finished(self) -> None:
        """Handle completion of the model-validation thread."""
        if self._stopping:
            QCoreApplication.exit(self._shutdown_exit_code)
            return

        self._models_finished = True
        self._show_main_when_ready()

    @Slot(str)
    def show_startup_error(self, message: str) -> None:
        """Display model-validation failure and continue with no models."""
        self._loading.set_progress(1, 1, message)
        self._models = []

    def _show_main_when_ready(self) -> None:
        """Create the main window once every startup prerequisite is ready."""
        if not self._can_show_main_window():
            return

        assert self._models is not None

        self._main_window_started = True

        try:
            window = MainWindow(self._models)
        except Exception:
            LOG.exception("Could not construct the AIBrain main window")
            self._stop_startup(exit_code=1)
            return

        self._main_window = window

        window.showMaximized()
        self._loading.finish()

    def _can_show_main_window(self) -> bool:
        """Return whether all prerequisites for the main window are satisfied."""
        return (
            not self._stopping
            and not self._gpu_relaunch_requested
            and self._loading.isVisible()
            and self._models is not None
            and self._models_finished
            and self._gpu_checked
            and not self._main_window_started
        )

    def _stop_startup(self, *, exit_code: int = 0) -> None:
        """Cancel startup and leave Qt after the worker thread has stopped."""
        if self._stopping:
            return

        self._stopping = True
        self._shutdown_exit_code = exit_code

        self._startup_worker.cancel()
        self._loading.finish()

        if self._startup_thread.isRunning():
            self._startup_thread.quit()
            return

        QTimer.singleShot(0, self.startup_thread_finished)
