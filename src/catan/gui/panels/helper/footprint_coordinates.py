"""Coordinates in the 3D footprints view (Footprints panel).

- A floor grid under the lowest session (like the axes of a 3D plot):
  lines at round values over the displayed area, tick labels along the two
  edges nearest to the viewer, axis names with the unit. It never covers a
  footprint.
- While a footprint is hovered: the cursor's point on that session's plane,
  a drop line onto the floor and a readout ``x · y · session`` at the
  bottom left.
- A direction marker (x / y / session arrows) in the bottom-left corner,
  turning with the view.

Coordinates are the aligned (common) frame of the view, shown in the
display unit (px or µm).
"""

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel
from vispy import scene
from vispy.scene import cameras, visuals

GRID_COLOR = (0.62, 0.66, 0.72, 0.28)
EDGE_COLOR = (0.62, 0.66, 0.72, 0.6)
LABEL_COLOR = (0.72, 0.75, 0.80, 1.0)
DROP_COLOR = (0.45, 0.75, 1.0, 0.9)
AXIS_COLORS = {"x": (0.95, 0.45, 0.42, 1), "y": (0.45, 0.85, 0.5, 1), "z": (0.5, 0.7, 1.0, 1)}
STEPS = (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000)
MAX_LINES = 7
MARKER_SIZE = 70  # px, direction marker


def grid_step(span, max_lines=MAX_LINES):
    """Round step (display unit) giving at most ``max_lines`` lines."""
    for step in STEPS:
        if span / step <= max_lines:
            return step
    return STEPS[-1]


def grid_values(low, high, step):
    first = np.ceil(low / step) * step
    return np.arange(first, high + 1e-9, step)


def ray_plane_point(transform, pos, z):
    """Scene point where the ray through canvas ``pos`` meets the plane at
    height ``z`` (None if parallel or behind)."""
    ends = transform.map(
        np.array([[pos[0], pos[1], -1, 1], [pos[0], pos[1], 0.5, 1]], dtype=float)
    )
    near, farther = ends[:, :3] / ends[:, 3:4]
    direction = farther - near
    if abs(direction[2]) < 1e-12:
        return None
    t = (z - near[2]) / direction[2]
    if t < 0:
        return None
    return near + t * direction


class FootprintCoordinates:
    """Floor grid, hover readout and direction marker of a footprints
    canvas (``canvas``: the vispy SceneCanvas, ``plot_root``: the node the
    footprint meshes live in)."""

    def __init__(self, canvas, view, plot_root, units):
        # ``units``: callable returning the current DisplayUnits.
        self.canvas = canvas
        self.view = view
        self.plot_root = plot_root
        self.units = units
        self.extent = None  # (x0, x1, y0, y1, z_floor) in px / scene units
        self._edges = None
        self._camera_key = None
        self.enabled = True  # the panel's "Coordinate grid" option

        self.grid = visuals.Line(
            np.zeros((2, 3), dtype=np.float32),
            color=GRID_COLOR,
            connect="segments",
            method="gl",
            parent=plot_root,
        )
        self.border = visuals.Line(
            np.zeros((2, 3), dtype=np.float32),
            color=EDGE_COLOR,
            connect="strip",
            method="gl",
            parent=plot_root,
        )
        self.labels = visuals.Text(
            "", color=LABEL_COLOR, font_size=7, parent=plot_root
        )
        self.titles = visuals.Text(
            "", color=LABEL_COLOR, font_size=8, bold=True, parent=plot_root
        )
        self.drop = visuals.Line(
            np.zeros((2, 3), dtype=np.float32),
            color=DROP_COLOR,
            width=1.5,
            method="gl",
            parent=plot_root,
        )
        self.points = visuals.Markers(parent=plot_root)
        for visual in (self.grid, self.border, self.labels, self.titles, self.drop, self.points):
            visual.interactive = False
            visual.visible = False
        for visual in (self.grid, self.border, self.drop):
            visual.set_gl_state(blend=True, depth_test=True)
            visual.order = 5

        # Direction marker: its own small view, camera follows the main one.
        self.marker_view = scene.widgets.ViewBox(parent=canvas.scene, border_color=None)
        self.marker_view.camera = cameras.TurntableCamera(fov=0, interactive=False)
        self.marker_view.interactive = False
        arrows = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]],
            dtype=np.float32,
        )
        colors = np.array(
            [AXIS_COLORS[a] for a in ("x", "x", "y", "y", "z", "z")], dtype=np.float32
        )
        visuals.Line(
            arrows, color=colors, connect="segments", width=2, method="gl",
            parent=self.marker_view.scene,
        )
        visuals.Text(
            ["x", "y", "S"],
            pos=np.array([[1.3, 0, 0], [0, 1.3, 0], [0, 0, 1.3]], dtype=np.float32),
            color=[AXIS_COLORS[a] for a in ("x", "y", "z")],
            font_size=8,
            bold=True,
            parent=self.marker_view.scene,
        )
        self.marker_view.camera.set_range(x=(-1.5, 1.5), y=(-1.5, 1.5), z=(-1.5, 1.5))

        self.readout = QLabel(canvas.native)
        self.readout.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.readout.setStyleSheet(
            "QLabel { color: #d8dde3; background: rgba(35, 39, 45, 200);"
            " border: 1px solid #59616c; border-radius: 4px; padding: 1px 6px;"
            " font-size: 11px; }"
        )
        self.readout.hide()

        canvas.events.draw.connect(self._on_draw, position="first")
        self.place()

    # ------------------------------------------------------------------

    def set_enabled(self, enabled):
        """Show or hide all of it (grid, hover readout, direction marker)."""
        self.enabled = bool(enabled)
        self.marker_view.visible = self.enabled
        extent = self.extent
        self.hide()
        if extent is not None:
            if self.enabled:
                self.set_extent(extent[0:2], extent[2:4], extent[4])
            else:
                self.extent = extent  # drawn again once switched on
        self.canvas.update()

    def place(self):
        """Corner positions (call on canvas resize)."""
        height = self.canvas.size[1]
        self.marker_view.pos = (6, height - MARKER_SIZE - 6)
        self.marker_view.size = (MARKER_SIZE, MARKER_SIZE)
        self.readout.adjustSize()
        self.readout.move(MARKER_SIZE + 12, self.canvas.native.height() - self.readout.height() - 10)

    def _factor(self):
        units = self.units()
        return units.factor("distance") if units is not None else 1.0

    def _unit(self):
        units = self.units()
        label = units.label("distance") if units is not None else "px"
        return label or "px"

    def set_extent(self, x_range, y_range, z_floor):
        """Floor over ``x_range`` × ``y_range`` (px) at height ``z_floor``."""
        extent = (
            float(x_range[0]), float(x_range[1]),
            float(y_range[0]), float(y_range[1]),
            float(z_floor),
            self._factor(),
        )
        if extent == self.extent:
            return
        self.extent = extent
        x0, x1, y0, y1, z, factor = extent
        if not self.enabled:
            # Remembered, drawn once switched on again.
            self.hide()
            self.extent = extent
            return
        if not (x1 > x0 and y1 > y0):
            self.hide()
            return

        span = max(x1 - x0, y1 - y0) * factor
        step = grid_step(span) / factor  # px
        xs = grid_values(x0, x1, step)
        ys = grid_values(y0, y1, step)
        segments = [((x, y0, z), (x, y1, z)) for x in xs]
        segments += [((x0, y, z), (x1, y, z)) for y in ys]
        self.grid.set_data(pos=np.asarray(segments, dtype=np.float32).reshape(-1, 3))
        self.border.set_data(
            pos=np.array(
                [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z], [x0, y0, z]],
                dtype=np.float32,
            )
        )
        self._ticks = (xs, ys)
        self._edges = None  # labels placed on the next draw
        for visual in (self.grid, self.border, self.labels, self.titles):
            visual.visible = True
        self.canvas.update()

    def hide(self):
        self.extent = None
        for visual in (self.grid, self.border, self.labels, self.titles):
            visual.visible = False
        self.clear_hover()

    def _format(self, value_px):
        value = value_px * self._factor()
        return f"{value:.0f}" if abs(value - round(value)) < 1e-6 else f"{value:.1f}"

    # Labels go to the edges nearest the viewer (lowest on screen); checked
    # whenever the camera moved.
    def _on_draw(self, event=None):
        camera = self.view.camera
        key = (
            round(camera.azimuth, 1),
            round(camera.elevation, 1),
            round(getattr(camera, "roll", 0.0), 1),
            tuple(np.round(camera.center, 1)),
            round(camera.scale_factor, 2),
        )
        if key == self._camera_key:
            return
        self._camera_key = key

        marker = self.marker_view.camera
        marker.azimuth = camera.azimuth
        marker.elevation = camera.elevation
        if hasattr(camera, "roll"):
            marker.roll = camera.roll

        if self.extent is not None and self.grid.visible:
            self._place_labels()

    def _place_labels(self):
        x0, x1, y0, y1, z, _ = self.extent
        transform = self.plot_root.node_transform(self.canvas.scene)

        def screen_y(points):
            mapped = transform.map(np.asarray(points, dtype=float))
            return mapped[:, 1] / mapped[:, 3]

        xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        # x labels along the y edge lower on screen, y labels along the x edge.
        lower = screen_y([[xm, y0, z, 1], [xm, y1, z, 1], [x0, ym, z, 1], [x1, ym, z, 1]])
        y_edge = y0 if lower[0] >= lower[1] else y1
        x_edge = x0 if lower[2] >= lower[3] else x1
        edges = (y_edge, x_edge)
        if edges == self._edges:
            return
        self._edges = edges

        xs, ys = self._ticks
        offset = 0.06 * max(x1 - x0, y1 - y0)
        out_y = -offset if y_edge == y0 else offset
        out_x = -offset if x_edge == x0 else offset
        texts = [self._format(x) for x in xs] + [self._format(y) for y in ys]
        positions = [(x, y_edge + out_y, z) for x in xs] + [(x_edge + out_x, y, z) for y in ys]
        if texts:
            self.labels.text = texts
            self.labels.pos = np.asarray(positions, dtype=np.float32)
        unit = self._unit()
        self.titles.text = [f"x ({unit}) →", f"y ({unit}) →"]
        self.titles.pos = np.array(
            [[xm, y_edge + 2.4 * out_y, z], [x_edge + 2.4 * out_x, ym, z]], dtype=np.float32
        )

    # ------------------------------------------------------------------

    def show_hover(self, pos, z_plane, session_text):
        """Cursor ``pos`` (canvas) on the plane at ``z_plane``: point, drop
        line to the floor and readout. Returns the (x, y) point or None."""
        if self.extent is None or not self.enabled:
            self.clear_hover()
            return None
        transform = self.canvas.scene.node_transform(self.plot_root)
        point = ray_plane_point(transform, pos, z_plane)
        if point is None:
            self.clear_hover()
            return None
        x, y = float(point[0]), float(point[1])
        floor = self.extent[4]
        self.drop.set_data(pos=np.array([[x, y, z_plane], [x, y, floor]], dtype=np.float32))
        self.points.set_data(
            np.array([[x, y, z_plane], [x, y, floor]], dtype=np.float32),
            face_color=DROP_COLOR,
            edge_width=0,
            size=5,
        )
        self.drop.visible = self.points.visible = True
        self.readout.setText(
            f"x {self._format(x)} · y {self._format(y)} {self._unit()} · {session_text}"
        )
        self.readout.show()
        self.place()
        self.readout.raise_()
        self.canvas.update()
        return x, y

    def clear_hover(self):
        if self.drop.visible or self.readout.isVisible():
            self.drop.visible = self.points.visible = False
            self.readout.hide()
            self.canvas.update()
