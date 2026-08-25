from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from .activity import ActivityField
from .graph import ConnectomeGraph

LOG = logging.getLogger(__name__)

POINT_VERTEX_SHADER = """#version 330
in vec3 in_position; in float in_region; in float in_activity;
uniform mat4 u_mvp; out float region; out float activity;
void main() { gl_Position = u_mvp * vec4(in_position, 1.0); gl_PointSize = 1.6 + in_activity * 8.0; region = in_region; activity = in_activity; }
"""
POINT_FRAGMENT_SHADER = """#version 330
in float region; in float activity; out vec4 f_color;
void main() { vec2 p = gl_PointCoord * 2.0 - 1.0; float radius = dot(p, p); if (radius > 1.0) discard;
 float r = .18 + region*.025 + activity*.35; float g = .43 + region*.030 + activity*.45; float b = .58 + region*.025 + activity*.35;
 float glow = 1.0 - smoothstep(.20, 1.0, radius); f_color = vec4(min(vec3(r,g,b)*(.55+glow*.75),1.0), .34+activity*.66); }
"""
EDGE_VERTEX_SHADER = """#version 330
in vec3 in_position; in float in_activity; uniform mat4 u_mvp; out float activity;
void main() { gl_Position = u_mvp * vec4(in_position, 1.0); activity = in_activity; }
"""
EDGE_FRAGMENT_SHADER = """#version 330
in float activity; out vec4 f_color;
void main() { f_color = vec4(.08+activity*.22, .27+activity*.60, .36+activity*.62, .055+activity*.34); }
"""


class ConnectomeRenderer(QOpenGLWidget):
    """ModernGL renderer with static topology and small per-frame activity uploads."""

    nodeSelected = Signal(object)
    backendChanged = Signal(str)

    def __init__(self, graph: ConnectomeGraph, field: ActivityField) -> None:
        super().__init__()
        self.graph, self.field = graph, field
        self.yaw, self.pitch, self.zoom = .25, -.2, 1.0
        self._last_pos: QPointF | None = None
        self.paused = False
        self._ctx = self._moderngl = None
        self._point_vao = self._edge_vao = None
        self._node_activity_buffer = self._edge_activity_buffer = None
        self._point_program = self._edge_program = None
        self._render_edges = self._select_render_edges()
        self._gpu_error: str | None = None
        self.setMinimumSize(420, 350)
        self.timer = QTimer(self); self.timer.timeout.connect(self._tick); self.timer.start(16)

    def _select_render_edges(self) -> np.ndarray:
        target = 40_000 if len(self.graph.positions) <= 5_000 else 100_000 if len(self.graph.positions) <= 12_000 else 180_000
        return self.graph.edges[::max(1, len(self.graph.edges) // target)]

    def initializeGL(self) -> None:
        try:
            import moderngl
            self._moderngl = moderngl
            self._ctx = moderngl.create_context(require=330)
            self._ctx.enable(moderngl.BLEND | moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE)
            self._ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
            self._build_gpu_resources()
            label = f"ModernGL GPU · {self._ctx.info.get('GL_RENDERER', 'OpenGL 3.3')}"
            LOG.info("Connectome renderer initialized: %s", label)
            self.backendChanged.emit(label)
        except Exception as exc:
            self._gpu_error = str(exc)
            LOG.exception("ModernGL renderer initialization failed")
            self.backendChanged.emit("GPU unavailable — reduced fallback renderer")

    def _build_gpu_resources(self) -> None:
        assert self._ctx is not None
        positions = np.ascontiguousarray(self.graph.positions, dtype="f4")
        regions = np.ascontiguousarray(self.graph.regions.astype("f4"))
        edge_positions = np.ascontiguousarray(positions[self._render_edges].reshape(-1, 3))
        position_buffer = self._ctx.buffer(positions.tobytes())
        region_buffer = self._ctx.buffer(regions.tobytes())
        self._node_activity_buffer = self._ctx.buffer(reserve=len(positions) * 4, dynamic=True)
        edge_position_buffer = self._ctx.buffer(edge_positions.tobytes())
        self._edge_activity_buffer = self._ctx.buffer(reserve=len(edge_positions) * 4, dynamic=True)
        self._point_program = self._ctx.program(vertex_shader=POINT_VERTEX_SHADER, fragment_shader=POINT_FRAGMENT_SHADER)
        self._edge_program = self._ctx.program(vertex_shader=EDGE_VERTEX_SHADER, fragment_shader=EDGE_FRAGMENT_SHADER)
        self._point_vao = self._ctx.vertex_array(self._point_program, [(position_buffer, "3f", "in_position"), (region_buffer, "1f", "in_region"), (self._node_activity_buffer, "1f", "in_activity")])
        self._edge_vao = self._ctx.vertex_array(self._edge_program, [(edge_position_buffer, "3f", "in_position"), (self._edge_activity_buffer, "1f", "in_activity")])

    def resizeGL(self, width: int, height: int) -> None:
        if self._ctx is not None:
            ratio = self.devicePixelRatioF()
            self._ctx.viewport = (0, 0, max(1, int(width * ratio)), max(1, int(height * ratio)))

    def _tick(self) -> None:
        if not self.paused:
            self.field.decay(); self.update()

    def _mvp(self) -> np.ndarray:
        aspect = max(self.width(), 1) / max(self.height(), 1)
        scale = 1.0 / (16.0 * self.zoom)
        projection = np.array(((scale / aspect, 0, 0, 0), (0, scale, 0, 0), (0, 0, .035, 0), (0, 0, 0, 1)), dtype="f4")
        cy, sy, cp, sp = np.cos(self.yaw), np.sin(self.yaw), np.cos(self.pitch), np.sin(self.pitch)
        rotate_y = np.array(((cy, 0, sy, 0), (0, 1, 0, 0), (-sy, 0, cy, 0), (0, 0, 0, 1)), dtype="f4")
        rotate_x = np.array(((1, 0, 0, 0), (0, cp, -sp, 0), (0, sp, cp, 0), (0, 0, 0, 1)), dtype="f4")
        return np.ascontiguousarray((projection @ rotate_x @ rotate_y).T)

    def _upload_activity(self) -> None:
        assert self._node_activity_buffer is not None and self._edge_activity_buffer is not None
        values = np.ascontiguousarray(self.field.values, dtype="f4")
        edge_values = np.maximum(values[self._render_edges[:, 0]], values[self._render_edges[:, 1]])
        self._node_activity_buffer.write(values.tobytes())
        self._edge_activity_buffer.write(np.repeat(edge_values, 2).astype("f4", copy=False).tobytes())

    def paintGL(self) -> None:
        if self._ctx is None or self._gpu_error:
            self._paint_fallback(); return
        try:
            matrix = self._mvp().tobytes()
            self._upload_activity(); self._ctx.clear(.025, .063, .090, 1.0, depth=1.0)
            self._edge_program["u_mvp"].write(matrix); self._point_program["u_mvp"].write(matrix)
            self._edge_vao.render(self._moderngl.LINES); self._point_vao.render(self._moderngl.POINTS)
        except Exception as exc:
            self._gpu_error = str(exc); LOG.exception("ModernGL frame failed; using fallback")
            self.backendChanged.emit("GPU frame failed — reduced fallback renderer"); self._paint_fallback()

    def _paint_fallback(self) -> None:
        painter = QPainter(self); painter.fillRect(self.rect(), QColor("#071018"))
        points = self._project_for_pick(); stride = max(1, len(points) // 3000)
        for point, activity in zip(points[::stride], self.field.values[::stride]):
            painter.setPen(QColor(75, 144, 169, int(55 + float(activity) * 190))); painter.drawPoint(QPointF(*point))
        painter.end()

    def _project_for_pick(self) -> np.ndarray:
        p = self.graph.positions; cy, sy, cp, sp = np.cos(self.yaw), np.sin(self.yaw), np.cos(self.pitch), np.sin(self.pitch)
        x = p[:, 0] * cy - p[:, 2] * sy; z = p[:, 0] * sy + p[:, 2] * cy; y = p[:, 1] * cp - z * sp
        scale = min(self.width(), self.height()) / (32 * self.zoom)
        return np.column_stack((self.width() / 2 + x * scale, self.height() / 2 - y * scale))

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._last_pos = event.position()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._last_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.position() - self._last_pos; self.yaw += delta.x() * .008
            self.pitch = float(np.clip(self.pitch + delta.y() * .006, -1.3, 1.3)); self._last_pos = event.position(); self.update()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._last_pos is not None and (event.position() - self._last_pos).manhattanLength() < 5:
            points = self._project_for_pick(); click = np.array((event.position().x(), event.position().y()))
            self.nodeSelected.emit(int(np.argmin(np.sum((points - click) ** 2, axis=1))))
        self._last_pos = None

    def wheelEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.zoom = float(np.clip(self.zoom * (.88 if event.angleDelta().y() > 0 else 1.14), .45, 3.5)); self.update()

    def reset_camera(self) -> None:
        self.yaw, self.pitch, self.zoom = .25, -.2, 1.0; self.update()
