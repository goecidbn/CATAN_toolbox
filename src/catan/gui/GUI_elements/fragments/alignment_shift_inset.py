import numpy as np
from vispy import scene
from vispy.scene.visuals import Line, Markers, Text
from copy import deepcopy

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class AlignmentShiftInset(QWidget):
    chosen = Signal(str, int)
    shift_changed = Signal(float, float)

    hovered = Signal(object)
    interaction_started = Signal()
    focus_released = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.items = []

        self.editable = False
        self._press = None
        self._moved = False
        self._disposed = False
        self._hover_key = None
        self._arrow_scale = None

        self.locked_session_id = None
        self._overview_state = None
        self._empty_press = None
        self._empty_moved = False

        self.setFixedSize(260, 220)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        title = QLabel("Shifts · dx → / dy ↓ · pixels", self)
        layout.addWidget(title)

        self.canvas = scene.SceneCanvas(bgcolor="#20252b")
        layout.addWidget(self.canvas.native, 1)
        self.setAttribute(Qt.WidgetAttribute.WA_NoMousePropagation, True)
        self.canvas.native.setAttribute(Qt.WidgetAttribute.WA_NoMousePropagation, True)

        self.info = QLabel("Click a line; drag the active endpoint.", self)
        self.info.setWordWrap(True)
        self.info.setFixedHeight(34)
        layout.addWidget(self.info)

        self.setStyleSheet("QLabel { color: #e8eaed; background: #20252b; }")

        grid = self.canvas.central_widget.add_grid(spacing=0)

        self.y_axis = scene.AxisWidget(orientation="left")
        self.y_axis.width_min = self.y_axis.width_max = 46
        grid.add_widget(self.y_axis, row=0, col=0)

        self.view = grid.add_view(row=0, col=1)
        self.view.camera = scene.PanZoomCamera(aspect=1)
        self.view.camera.flip = (False, True, False)
        self.view.camera.interactive = True

        self.x_axis = scene.AxisWidget(orientation="bottom")
        self.x_axis.height_min = self.x_axis.height_max = 36
        grid.add_widget(self.x_axis, row=1, col=1)

        for widget, label in (
            (self.x_axis, "dx (px)"),
            (self.y_axis, "dy (px)"),
        ):
            axis = widget.axis
            axis.axis_label = label
            axis.axis_font_size = 8
            axis.tick_font_size = 7
            # axis.axis_label_margin = 20
            # axis.tick_label_margin = 12
            axis.major_tick_length = 4
            axis.minor_tick_length = 2
            axis.text_color = "#e8eaed"
            axis.tick_color = "#aab2bd"
            axis.axis_color = "#aab2bd"
            widget.link_view(self.view)

        self.x_axis.axis.axis_label_margin = 20
        self.x_axis.axis.tick_label_margin = 12

        self.y_axis.axis.axis_label_margin = 28
        self.y_axis.axis.tick_label_margin = 5

        self.axes = Line(
            pos=np.zeros((4, 2), dtype=np.float32),
            connect="segments",
            color="#68717c",
            parent=self.view.scene,
        )
        self.lines = Line(
            pos=np.zeros((6, 2), dtype=np.float32),
            connect="segments",
            width=1.25,
            parent=self.view.scene,
        )
        self.tips = Markers(parent=self.view.scene)
        self.tips.set_data(
            pos=np.zeros((1, 2), dtype=np.float32),
            size=0,
        )

        for order, visual in enumerate((self.axes, self.lines, self.tips)):
            visual.order = order
            visual.set_gl_state("translucent", depth_test=False)

        self.point_labels = Text(
            text=[""],
            pos=np.zeros((1, 2), dtype=np.float32),
            color="#f0f2f5",
            font_size=8,
            anchor_x="left",
            anchor_y="bottom",
            parent=self.view.scene,
        )
        self.point_labels.order = 3
        self.point_labels.set_gl_state("translucent", depth_test=False)
        self.point_labels.visible = False

        self._connections = (
            (self.canvas.events.mouse_press, self._on_press),
            (self.canvas.events.mouse_move, self._on_move),
            (self.canvas.events.mouse_release, self._on_release),
            (self.canvas.events.draw, self._update_arrow_sizes),
        )
        # Decide whether this is an arrow interaction before camera handling.
        for signal, slot in self._connections:
            signal.connect(slot, position="first")

        # Accept wheel events after VisPy has applied the inset zoom.
        self.canvas.events.mouse_wheel.connect(self._on_wheel)

    def set_items(self, items, *, fit=False):
        self.items = items
        self.lines.visible = self.tips.visible = bool(items)
        self.point_labels.visible = False

        if not items:
            return

        transform = self._transform()
        positions, colors, tips, tip_colors, sizes = [], [], [], [], []

        for item in items:
            tip = np.asarray(item["xy"], dtype=np.float32)
            active = item.get("active", False)
            origin = np.asarray(item.get("start", (0, 0)), dtype=np.float32)

            color = (
                (1.0, 0.82, 0.40, 1.0)
                if active
                else (
                    (0.35, 0.85, 1.0, 1.0)
                    if item["kind"] == "reference"
                    else (0.63, 0.68, 0.75, 0.65)
                )
            )

            # length = float(np.linalg.norm(tip))
            # direction = tip / length if length else np.zeros(2)

            if not item.get("focused", True):
                color = (*color[:3], 0.18)

            if (item["kind"], item["owner"], item["id"]) == self._hover_key:
                color = (1.0, 1.0, 1.0, 1.0)

            screen_origin = np.asarray(transform.map(origin)[:2])
            screen_tip = np.asarray(transform.map(tip)[:2])
            delta = screen_tip - screen_origin
            length = float(np.linalg.norm(delta))

            if length > 1e-6:
                direction = delta / length
                normal = np.array([-direction[1], direction[0]])

                # Fixed logical-pixel dimensions, independent of shift or zoom.
                base = screen_tip - 5.0 * direction
                wing_a = transform.imap(base + 2.5 * normal)[:2]
                wing_b = transform.imap(base - 2.5 * normal)[:2]
            else:
                wing_a = wing_b = tip

            positions.extend(
                [
                    origin,
                    tip,
                    tip,
                    wing_a,
                    tip,
                    wing_b,
                ]
            )
            colors.extend([color] * 6)
            tips.append(tip)
            tip_colors.append(color)
            sizes.append(7 if active else 4)

        self.lines.set_data(
            pos=np.asarray(positions, dtype=np.float32),
            color=np.asarray(colors, dtype=np.float32),
            connect="segments",
        )
        self.tips.set_data(
            pos=np.asarray(tips, dtype=np.float32),
            face_color=np.asarray(tip_colors, dtype=np.float32),
            edge_color="#20252b",
            size=np.asarray(sizes, dtype=np.float32),
        )

        if fit:
            points = np.vstack(([0, 0], tips))
            lower = points.min(axis=0) - 5
            upper = points.max(axis=0) + 5

            self.axes.set_data(
                pos=np.array(
                    [
                        [lower[0], 0],
                        [upper[0], 0],
                        [0, lower[1]],
                        [0, upper[1]],
                    ],
                    dtype=np.float32,
                )
            )

            self.view.camera.set_range(
                x=(lower[0], upper[0]),
                y=(lower[1], upper[1]),
                z=(-1, 1),
                margin=0.08,
            )

        labelled = [item for item in items if item.get("focused", True)]
        if labelled:
            self.point_labels.text = [f'S{item["id"]}' for item in labelled]
            self.point_labels.pos = np.asarray(
                [
                    transform.imap(np.asarray(transform.map(item["xy"])[:2]) + (6, -7))[
                        :2
                    ]
                    for item in labelled
                ],
                dtype=np.float32,
            )
            self.point_labels.visible = True

        self.canvas.update()

    def _transform(self):
        return self.view.scene.node_transform(self.canvas.scene)

    def _hit(self, position):
        if not self.items:
            return None

        transform = self._transform()
        origin = np.asarray(
            [transform.map(item.get("start", (0, 0)))[:2] for item in self.items]
        )
        tips = np.asarray([transform.map(item["xy"])[:2] for item in self.items])
        position = np.asarray(position)

        # Endpoints take precedence over intersecting shafts.
        distances = np.linalg.norm(tips - position, axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= 10:
            return self.items[nearest]

        directions = tips - origin
        lengths_squared = np.sum(directions**2, axis=1)
        fractions = np.sum((position - origin) * directions, axis=1) / np.maximum(
            lengths_squared, 1e-12
        )

        closest = origin + np.clip(fractions, 0, 1)[:, None] * directions
        distances = np.linalg.norm(closest - position, axis=1)

        # Avoid selecting all arrows at their shared origin.
        distances[fractions < 0.15] = np.inf
        nearest = int(np.argmin(distances))
        return self.items[nearest] if distances[nearest] <= 6 else None

    def _set_hover(self, item):
        key = None if item is None else (item["kind"], item["owner"], item["id"])
        if key == self._hover_key:
            return

        self._hover_key = key
        self.set_items(self.items)

        # Empty space inside the inset keeps the current preview,
        # so the pointer can travel from a total to its candidates.
        if item is not None:
            self.hovered.emit(item)

    def _update_arrow_sizes(self, *_):
        if self._disposed or not self.items:
            return

        transform = self._transform()
        basis = np.asarray(transform.map([[0, 0], [1, 0], [0, 1]]))[:, :2]
        scale = basis[1:] - basis[0]

        if self._arrow_scale is not None and np.allclose(scale, self._arrow_scale):
            return

        self._arrow_scale = scale.copy()
        self.set_items(self.items)

    def lock_session(self, session_id):
        if self.locked_session_id == session_id:
            return

        if self.locked_session_id is None:
            self._overview_state = deepcopy(self.view.camera.get_state())

        self.locked_session_id = session_id
        points = []

        for item in self.items:
            if item["kind"] == "session" and item["id"] == session_id:
                points.append(item["xy"])
            elif item["kind"] == "reference" and item["owner"] == session_id:
                points.extend((item["start"], item["xy"]))

        if not points:
            return

        points = np.asarray(points)
        span = np.ptp(points, axis=0)
        padding = np.maximum(1.0, span * 0.2)
        lower = points.min(axis=0) - padding
        upper = points.max(axis=0) + padding

        self.view.camera.set_range(
            x=(lower[0], upper[0]),
            y=(lower[1], upper[1]),
            z=(-1, 1),
            margin=0,
        )

    def release_focus(self):
        self.locked_session_id = None

        if self._overview_state is not None:
            self.view.camera.set_state(self._overview_state)
            self._overview_state = None

    def leaveEvent(self, event):
        if self._press is None:
            self._set_hover(None)
            self.hovered.emit(None)
        super().leaveEvent(event)

    def hideEvent(self, event):
        if not self._disposed:
            self._set_hover(None)
            self.hovered.emit(None)
        super().hideEvent(event)

    def _on_wheel(self, event):

        self.interaction_started.emit()

        if event.native is not None:
            event.native.accept()
        event.handled = True

    def _on_press(self, event):
        if event.button != 1:
            return

        self._empty_press = None
        self._empty_moved = False

        self.interaction_started.emit()

        self.view.camera.interactive = True

        item = self._hit(event.pos)

        self._press = None
        self._moved = False
        if item is None:
            self._empty_press = np.asarray(event.pos, dtype=float)
            return

        # Clicking an arrow selects/drags it; empty space pans the camera.
        self.view.camera.interactive = False
        event.handled = True

        transform = self._transform()
        tip = np.asarray(transform.map(item["xy"])[:2])
        draggable = (
            self.editable
            and item.get("active", False)
            and np.linalg.norm(tip - event.pos) <= 10
        )

        if draggable:
            self.hovered.emit(None)
        self._press = (
            item,
            np.asarray(event.pos, dtype=float),
            np.asarray(transform.imap(event.pos)[:2]),
            draggable,
        )

    def _on_move(self, event):

        if self._empty_press is not None:
            if np.linalg.norm(event.pos - self._empty_press) >= 3:
                self._empty_moved = True
            return

        if self._press is not None:
            event.handled = True
            item, start_pixel, start_point, draggable = self._press

            if draggable:
                if np.linalg.norm(event.pos - start_pixel) >= 3:
                    self._moved = True

                if self._moved:
                    point = np.asarray(self._transform().imap(event.pos)[:2])
                    xy = np.asarray(item["xy"]) + point - start_point
                    self.shift_changed.emit(float(xy[0]), float(xy[1]))
                    self.info.setText(f"Draft: dx={xy[0]:.2f}, dy={xy[1]:.2f}")
            return

        # Do not switch previews while panning.
        if event.buttons:
            return

        item = self._hit(event.pos)
        self._set_hover(item)

        if item is None:
            self.info.setText("Click a line; drag the active endpoint.")
        elif item["kind"] == "reference":
            delta = np.asarray(item["xy"]) - item["start"]
            self.info.setText(
                f'{item["label"]}\n' f"Δdx={delta[0]:.2f}, Δdy={delta[1]:.2f}"
            )
        else:
            self.info.setText(
                f'{item["label"]}\n' f'dx={item["xy"][0]:.2f}, dy={item["xy"][1]:.2f}'
            )

    def _on_release(self, event):
        if event.button != 1:
            return

        if self._empty_press is not None:
            moved = self._empty_moved
            self._empty_press = None
            self._empty_moved = False

            if not moved:
                self.focus_released.emit()
            return

        if self._press is None:
            return

        self.view.camera.interactive = True
        event.handled = True

        item, _, _, _ = self._press
        moved = self._moved
        self._press = None
        self._moved = False

        if not moved:
            self.chosen.emit(item["kind"], item["id"])

        if not self.underMouse():
            self._set_hover(None)
            self.hovered.emit(None)

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True

        for signal, slot in self._connections:
            signal.disconnect(slot)

        self.canvas.events.mouse_wheel.disconnect(self._on_wheel)
        self.canvas.close()
