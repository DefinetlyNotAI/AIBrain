from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import QPointF, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from ..native.wrapper.connectome_kernels import native
from ..utils.array_api import to_numpy
from ..utils.gpu import can_request_gpu_relaunch, should_prefer_high_performance_gpu
from .activity import ActivityField
from .generator import CLUSTER_COLOR_MAP, cluster_colour_map_for_background
from .graph import ConnectomeGraph

LOG = logging.getLogger(__name__)

# noinspection LongLine
POINT_VERTEX_SHADER = """#version 330
in vec3 in_position; in float in_region; in float in_activity;
uniform mat4 u_mvp; uniform float u_region_filter; out float region; out float activity;
void main() { if (u_region_filter >= 0.0 && abs(in_region - u_region_filter) > .5) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
 gl_Position = u_mvp * vec4(in_position, 1.0); gl_PointSize = 1.6 + in_activity * 8.0; region = in_region; activity = in_activity; }
"""


def _shader_palette(colour_map: dict[str, str] = CLUSTER_COLOR_MAP) -> str:
    colours = tuple(colour_map.values())
    clauses = []
    for index, colour in enumerate(colours):
        red, green, blue = (
            int(colour[offset : offset + 2], 16) / 255 for offset in (1, 3, 5)
        )
        clauses.append(
            f"if (cluster == {index}) return vec3({red:.6f}, {green:.6f}, {blue:.6f});"
        )
    return "\n ".join((*clauses, "return vec3(.35, .65, .85);"))


def _point_fragment_shader(colour_map: dict[str, str]) -> str:
    return f"""#version 330
in float region; in float activity; uniform float u_border_width; uniform float u_idle_strength; out vec4 f_color;
vec3 cluster_colour(int cluster) {{ {_shader_palette(colour_map)} }}
void main() {{ vec2 p = gl_PointCoord * 2.0 - 1.0; float radius = dot(p, p); if (radius > 1.0) discard;
 vec3 base = cluster_colour(int(region + .5)); float glow = 1.0 - smoothstep(.20, 1.0, radius);
 vec3 fill = min(base * (u_idle_strength + activity * .68 + glow * .14), 1.0);
 vec3 color = fill; if (u_border_width > .0) {{ float rim = smoothstep(1.0 - u_border_width, 1.0, radius); color = mix(fill, vec3(.015, .045, .065), rim); }}
 f_color = vec4(color, .78+activity*.22); }}
"""


EDGE_VERTEX_SHADER = """#version 330
in vec3 in_position; in float in_region; in float in_activity; uniform mat4 u_mvp; uniform float u_region_filter; out float region; out float activity;
void main() { if (u_region_filter >= 0.0 && abs(in_region - u_region_filter) > .5) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
 gl_Position = u_mvp * vec4(in_position, 1.0); region = in_region; activity = in_activity; }
"""


def _edge_fragment_shader(colour_map: dict[str, str]) -> str:
    return f"""#version 330
in float region; in float activity; uniform float u_idle_strength; out vec4 f_color;
vec3 cluster_colour(int cluster) {{ {_shader_palette(colour_map)} }}
void main() {{ vec3 base = cluster_colour(int(region + .5));
 f_color = vec4(min(base * (u_idle_strength + activity * .62), 1.0), .10 + activity * .52); }}
"""


POINT_FRAGMENT_SHADER = _point_fragment_shader(CLUSTER_COLOR_MAP)
EDGE_FRAGMENT_SHADER = _edge_fragment_shader(CLUSTER_COLOR_MAP)


class ConnectomeRenderer(QOpenGLWidget):
    """ModernGL renderer with static topology and small per-frame activity uploads."""

    nodeSelected = Signal(object)
    backendChanged = Signal(str)
    gpuRestartRequested = Signal(str)

    def __init__(self, graph: ConnectomeGraph, field: ActivityField) -> None:
        super().__init__()
        self.graph, self.field = graph, field
        self.yaw, self.pitch, self.zoom = 0.25, -0.2, 1.0
        self.pan_x, self.pan_y = 0.0, 0.0
        self._last_pos: QPointF | None = None
        self.paused = False
        self._ctx = self._moderngl = None
        self._framebuffer = None
        self._point_vao = self._edge_vao = None
        self._node_activity_buffer = self._edge_activity_buffer = None
        self._point_program = self._edge_program = None
        self._render_edges = self._select_render_edges()
        self._edge_activity_values = np.zeros(len(self._render_edges) * 2, dtype="f4")
        self._gpu_error: str | None = None
        self.show_node_borders = True
        self.node_border_width = 0.12
        self.view_mode = "3d"
        self.region_filter: int | None = None
        self.background_colour = "#071018"
        self._cluster_colour_map = cluster_colour_map_for_background(
            self.background_colour
        )
        self._palette_resources_pending = False
        self.setMinimumSize(260, 240)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        # Thirty frames per second keeps interaction smooth while avoiding
        # redundant host-buffer preparation between generated tokens.
        self.timer.start(33)

    def _select_render_edges(self) -> np.ndarray:
        target = (
            20_000
            if len(self.graph.positions) <= 5_000
            else 50_000 if len(self.graph.positions) <= 12_000 else 110_000
        )
        selected = self.graph.edges[:: max(1, len(self.graph.edges) // target)]
        return np.ascontiguousarray(to_numpy(selected), dtype=np.int32)

    def initializeGL(self) -> None:
        try:
            import moderngl

            self._moderngl = moderngl
            self._ctx = moderngl.create_context(require=330)
            # QOpenGLWidget renders into a Qt-owned FBO, not OpenGL FBO 0.
            # Capturing this current FBO is essential: otherwise ModernGL draws
            # successfully but its output never reaches the widget.
            self._framebuffer = self._ctx.detect_framebuffer()
            self._ctx.enable(
                moderngl.BLEND | moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE
            )
            self._ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
            self._build_gpu_resources()
            renderer_name = str(self._ctx.info.get("GL_RENDERER", "OpenGL 3.3"))
            vendor = str(self._ctx.info.get("GL_VENDOR", "unknown vendor"))
            is_nvidia = "nvidia" in f"{vendor} {renderer_name}".lower()
            if not is_nvidia:
                requested = should_prefer_high_performance_gpu()
                if requested and can_request_gpu_relaunch():
                    message = f"GPU mismatch: OpenGL selected {vendor} - {renderer_name}, expected NVIDIA"
                    QSettings().setValue("opengl_gpu_mismatch_reason", message)
                    renderer_name += (
                        "; GPU mismatch; restarting through the startup loader"
                    )
                    LOG.warning(
                        "OpenGL context is not on NVIDIA: vendor=%s renderer=%s",
                        vendor,
                        renderer_name,
                    )
                    self.backendChanged.emit(
                        "GPU mismatch detected — restarting through the startup loader…"
                    )
                    self.gpuRestartRequested.emit(message)
                    return
                if requested:
                    renderer_name += "; GPU mismatch (NVIDIA retry already attempted)"
                    QSettings().setValue(
                        "opengl_gpu_mismatch_reason",
                        f"OpenGL selected {vendor} - {renderer_name}; Windows or the driver ignored the preference",
                    )
                    LOG.warning(
                        "OpenGL context is not on NVIDIA: vendor=%s renderer=%s",
                        vendor,
                        renderer_name,
                    )
                else:
                    renderer_name += " · Windows system-default GPU"
                    QSettings().remove("opengl_gpu_mismatch_reason")
            label = f"ModernGL GPU - {renderer_name}"
            LOG.info("Connectome renderer initialized: %s", label)
            self.backendChanged.emit(label)
        except Exception as exc:
            self._gpu_error = str(exc)
            LOG.exception("ModernGL renderer initialization failed")
            self.backendChanged.emit("GPU unavailable — reduced fallback renderer")

    def _build_gpu_resources(self) -> None:
        assert self._ctx is not None
        positions = np.ascontiguousarray(to_numpy(self.graph.positions), dtype="f4")
        graph_regions = to_numpy(self.graph.regions)
        regions = np.ascontiguousarray(graph_regions, dtype="f4")
        edge_positions = np.ascontiguousarray(
            positions[self._render_edges].reshape(-1, 3)
        )
        edge_regions = np.ascontiguousarray(
            graph_regions[self._render_edges].reshape(-1).astype("f4")
        )
        position_buffer = self._ctx.buffer(positions.tobytes())
        region_buffer = self._ctx.buffer(regions.tobytes())
        self._node_activity_buffer = self._ctx.buffer(
            reserve=len(positions) * 4, dynamic=True
        )
        edge_position_buffer = self._ctx.buffer(edge_positions.tobytes())
        edge_region_buffer = self._ctx.buffer(edge_regions.tobytes())
        self._edge_activity_buffer = self._ctx.buffer(
            reserve=len(edge_positions) * 4, dynamic=True
        )
        self._point_program = self._ctx.program(
            vertex_shader=POINT_VERTEX_SHADER,
            fragment_shader=_point_fragment_shader(self._cluster_colour_map),
        )
        self._edge_program = self._ctx.program(
            vertex_shader=EDGE_VERTEX_SHADER,
            fragment_shader=_edge_fragment_shader(self._cluster_colour_map),
        )
        self._point_vao = self._ctx.vertex_array(
            self._point_program,
            [
                (position_buffer, "3f", "in_position"),
                (region_buffer, "1f", "in_region"),
                (self._node_activity_buffer, "1f", "in_activity"),
            ],
        )
        self._edge_vao = self._ctx.vertex_array(
            self._edge_program,
            [
                (edge_position_buffer, "3f", "in_position"),
                (edge_region_buffer, "1f", "in_region"),
                (self._edge_activity_buffer, "1f", "in_activity"),
            ],
        )

    def resizeGL(self, width: int, height: int) -> None:
        if self._ctx is not None:
            ratio = self.devicePixelRatioF()
            self._ctx.viewport = (
                0,
                0,
                max(1, int(width * ratio)),
                max(1, int(height * ratio)),
            )

    def _tick(self) -> None:
        if not self.paused:
            self.field.decay()
            self.update()

    def _mvp(self) -> np.ndarray:
        aspect = max(self.width(), 1) / max(self.height(), 1)
        scale = 1.0 / ((16.0 if self.view_mode == "2d" else 16.0) * self.zoom)
        depth_scale = 0.0 if self.view_mode == "2d" else 0.035
        projection = np.array(
            (
                (scale / aspect, 0, 0, 0),
                (0, scale, 0, 0),
                (0, 0, depth_scale, 0),
                (0, 0, 0, 1),
            ),
            dtype="f4",
        )
        if self.view_mode == "2d":
            projection[0, 3] = getattr(self, "pan_x", 0.0)
            projection[1, 3] = getattr(self, "pan_y", 0.0)
            return np.ascontiguousarray(projection.T)
        cy, sy, cp, sp = (
            np.cos(self.yaw),
            np.sin(self.yaw),
            np.cos(self.pitch),
            np.sin(self.pitch),
        )
        rotate_y = np.array(
            ((cy, 0, sy, 0), (0, 1, 0, 0), (-sy, 0, cy, 0), (0, 0, 0, 1)), dtype="f4"
        )
        rotate_x = np.array(
            ((1, 0, 0, 0), (0, cp, -sp, 0), (0, sp, cp, 0), (0, 0, 0, 1)), dtype="f4"
        )
        return np.ascontiguousarray((projection @ rotate_x @ rotate_y).T)

    def _upload_activity(self) -> None:
        assert (
            self._node_activity_buffer is not None
            and self._edge_activity_buffer is not None
        )
        values = np.ascontiguousarray(to_numpy(self.field.values), dtype="f4")
        native.edges(values, self._render_edges, self._edge_activity_values)
        self._node_activity_buffer.write(values.tobytes())
        self._edge_activity_buffer.write(self._edge_activity_values.tobytes())

    def paintGL(self) -> None:
        if self._ctx is None or self._gpu_error:
            self._paint_fallback()
            return
        try:
            self._framebuffer = self._ctx.detect_framebuffer()
            self._framebuffer.use()
            if self._palette_resources_pending:
                self._build_gpu_resources()
                self._palette_resources_pending = False
            ratio = self.devicePixelRatioF()
            self._ctx.viewport = (
                0,
                0,
                max(1, int(self.width() * ratio)),
                max(1, int(self.height() * ratio)),
            )
            matrix = self._mvp().tobytes()
            self._upload_activity()
            colour = QColor(self.background_colour)
            self._ctx.clear(
                colour.redF(), colour.greenF(), colour.blueF(), 1.0, depth=1.0
            )
            self._edge_program["u_mvp"].write(matrix)
            self._point_program["u_mvp"].write(matrix)
            self._point_program["u_border_width"].value = (
                self.node_border_width if self.show_node_borders else 0.0
            )
            idle_strength = 0.58 if self.view_mode == "2d" else 0.42
            self._point_program["u_idle_strength"].value = idle_strength
            self._edge_program["u_idle_strength"].value = idle_strength
            region_filter = (
                float(self.region_filter) if self.region_filter is not None else -1.0
            )
            self._point_program["u_region_filter"].value = region_filter
            self._edge_program["u_region_filter"].value = region_filter
            self._edge_vao.render(self._moderngl.LINES)
            self._point_vao.render(self._moderngl.POINTS)
        except Exception as exc:
            self._gpu_error = str(exc)
            LOG.exception("ModernGL frame failed; using fallback")
            self.backendChanged.emit("GPU frame failed — reduced fallback renderer")
            self._paint_fallback()

    def _paint_fallback(self) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(self.background_colour))
        indices = self._visible_indices()
        points = self._project_for_pick()[indices]
        stride = max(1, len(points) // 3000)
        activities = to_numpy(self.field.values)[indices]
        regions = to_numpy(self.graph.regions)[indices]
        for point, activity, region in zip(
            points[::stride],
            activities[::stride],
            regions[::stride],
        ):
            colour = QColor(
                self._cluster_colour_map[self.graph.region_names[int(region)]]
            )
            colour.setAlpha(int(155 + float(activity) * 100))
            painter.setPen(colour)
            painter.drawPoint(QPointF(*point))
        painter.end()

    def _project_for_pick(self) -> np.ndarray:
        p = to_numpy(self.graph.positions)
        if self.view_mode == "2d":
            x, y = p[:, 0], p[:, 1]
            scale = min(self.width(), self.height()) / (32 * self.zoom)
            return np.column_stack(
                (
                    self.width() / 2 + self.pan_x * self.width() / 2 + x * scale,
                    self.height() / 2 - self.pan_y * self.height() / 2 - y * scale,
                )
            )
        cy, sy, cp, sp = (
            np.cos(self.yaw),
            np.sin(self.yaw),
            np.cos(self.pitch),
            np.sin(self.pitch),
        )
        x = p[:, 0] * cy - p[:, 2] * sy
        z = p[:, 0] * sy + p[:, 2] * cy
        y = p[:, 1] * cp - z * sp
        scale = min(self.width(), self.height()) / (32 * self.zoom)
        return np.column_stack(
            (self.width() / 2 + x * scale, self.height() / 2 - y * scale)
        )

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._last_pos = event.position()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if (
            self.view_mode == "3d"
            and self._last_pos is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            delta = event.position() - self._last_pos
            self.yaw += delta.x() * 0.008
            self.pitch = min(1.3, max(-1.3, self.pitch + delta.y() * 0.006))
            self._last_pos = event.position()
            self.update()
        elif (
            self.view_mode == "2d"
            and self._last_pos is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            delta = event.position() - self._last_pos
            self.pan_x += delta.x() * 2 / max(self.width(), 1)
            self.pan_y -= delta.y() * 2 / max(self.height(), 1)
            self._last_pos = event.position()
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if (
            self._last_pos is not None
            and (event.position() - self._last_pos).manhattanLength() < 5
        ):
            indices = self._visible_indices()
            points = self._project_for_pick()[indices]
            click = np.array((event.position().x(), event.position().y()))
            if len(indices):
                self.nodeSelected.emit(
                    int(indices[np.argmin(np.sum((points - click) ** 2, axis=1))])
                )
        self._last_pos = None

    def wheelEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        # Smaller zoom is closer. 0.025 gives ~40x closer inspection than
        # the default view while retaining a finite, numerically stable scale.
        self.zoom = min(
            6.0,
            max(
                0.025,
                self.zoom * (0.82 if event.angleDelta().y() > 0 else 1.22),
            ),
        )
        self.update()

    def reset_camera(self) -> None:
        self.yaw, self.pitch, self.zoom = 0.25, -0.2, 1.0
        self.pan_x, self.pan_y = 0.0, 0.0
        self.update()

    def set_node_borders(self, visible: bool, width: float) -> None:
        self.show_node_borders = visible
        self.node_border_width = min(1.0, max(0.0, width))
        self.update()

    def set_background_colour(self, colour: str) -> None:
        """Apply a validated renderer background immediately without a restart."""
        if QColor(colour).isValid():
            self.background_colour = QColor(colour).name(QColor.NameFormat.HexRgb)
            palette = cluster_colour_map_for_background(self.background_colour)
            if palette != self._cluster_colour_map:
                self._cluster_colour_map = palette
                self._palette_resources_pending = True
            self.update()

    def set_projection_mode(self, mode: str, region_filter: int | None = None) -> None:
        """Switch between depth-aware 3D and flat, optionally sector-filtered 2D."""
        if mode not in {"2d", "3d"}:
            raise ValueError(f"Unsupported connectome projection mode: {mode}")
        if region_filter is not None and not 0 <= region_filter < len(
            self.graph.region_names
        ):
            raise ValueError(
                "The selected sector is outside the connectome region list"
            )
        self.view_mode = mode
        self.region_filter = region_filter if mode == "2d" else None
        self.update()

    def _visible_indices(self) -> np.ndarray:
        if self.region_filter is None:
            return np.arange(len(self.graph.positions))
        return np.flatnonzero(to_numpy(self.graph.regions) == self.region_filter)
