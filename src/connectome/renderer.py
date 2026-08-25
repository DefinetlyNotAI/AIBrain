from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from .activity import ActivityField
from .graph import ConnectomeGraph


class ConnectomeRenderer(QOpenGLWidget):
    nodeSelected = Signal(object)

    def __init__(self, graph: ConnectomeGraph, field: ActivityField) -> None:
        super().__init__()
        self.graph, self.field = graph, field
        self.yaw, self.pitch, self.zoom = .25, -.2, 1.0
        self._last_pos = None
        self.paused = False
        self.setMinimumSize(420, 350)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(16)

    def _tick(self) -> None:
        if not self.paused:
            self.field.decay()
            self.update()

    def _project(self) -> np.ndarray:
        p = self.graph.positions
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        x = p[:, 0] * cy - p[:, 2] * sy
        z = p[:, 0] * sy + p[:, 2] * cy
        y = p[:, 1] * cp - z * sp
        scale = min(self.width(), self.height()) / (26 * self.zoom)
        return np.column_stack((self.width()/2 + x * scale, self.height()/2 - y * scale, z))

    def paintGL(self) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#071018"))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        points = self._project()
        # Sampling edges preserves responsiveness at high quality while retaining structure.
        cap = min(len(self.graph.edges), 45000)
        step = max(1, len(self.graph.edges) // cap)
        edges = self.graph.edges[::step]
        values = self.field.values
        for a, b in edges:
            intensity = float(max(values[a], values[b]))
            alpha = int(12 + intensity * 110)
            painter.setPen(QPen(QColor(38, 91, 116, alpha), 1))
            painter.drawLine(QPointF(*points[a, :2]), QPointF(*points[b, :2]))
        order = np.argsort(points[:, 2])
        for index in order:
            activity = float(values[index])
            region = int(self.graph.regions[index])
            base = (55 + region * 14, 104 + region * 9, 138 + region * 6)
            glow = int(70 + activity * 185)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(min(130, base[0] + glow), min(245, base[1] + glow), min(255, base[2] + glow), 75 + int(activity*180)))
            radius = 1.0 + activity * 3.3
            painter.drawEllipse(QPointF(*points[index, :2]), radius, radius)
        painter.end()

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._last_pos = event.position()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._last_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.position() - self._last_pos
            self.yaw += delta.x() * .008
            self.pitch = float(np.clip(self.pitch + delta.y() * .006, -1.3, 1.3))
            self._last_pos = event.position()
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._last_pos is not None and (event.position() - self._last_pos).manhattanLength() < 5:
            pts = self._project()[:, :2]
            click = np.array([event.position().x(), event.position().y()])
            index = int(np.argmin(np.sum((pts - click) ** 2, axis=1)))
            self.nodeSelected.emit(index)
        self._last_pos = None

    def wheelEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.zoom = float(np.clip(self.zoom * (0.88 if event.angleDelta().y() > 0 else 1.14), .45, 3.5))
        self.update()

    def reset_camera(self) -> None:
        self.yaw, self.pitch, self.zoom = .25, -.2, 1.0
        self.update()
