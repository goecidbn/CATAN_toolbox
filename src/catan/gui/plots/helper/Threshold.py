from time import time
import numpy as np
from typing import Literal, Optional, List
from dataclasses import dataclass

from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QToolTip
from vispy.scene import visuals

from catan.gui.interaction import click_events

ThresholdAxis = Literal["x", "y"]
ThresholdDirection = Literal["greater", "less"]

# THRESHOLD_STYLE = {
#     "inactive_line": (0.15, 0.35, 0.55, 0.35),  # muted blue-gray
#     "active_line": (0.05, 0.45, 0.85, 0.95),  # clear blue
#     "hover_line": (0.00, 0.62, 1.00, 1.00),  # brighter blue
#     "region": (0.05, 0.45, 0.85, 0.10),  # transparent blue fill
# }

THRESHOLD_STYLE = {
    "inactive_line": (0.25, 0.25, 0.25, 0.35),
    "active_line": (0.10, 0.35, 0.55, 0.95),
    "hover_line": (0.00, 0.55, 0.90, 1.00),
    "region": (0.00, 0.00, 0.00, 0.20),
}


@dataclass
class ThresholdSpec:
    axis: ThresholdAxis
    value: float = 0.0
    direction: ThresholdDirection = "greater"
    active: bool = False

    def mask(self, values):
        if not self.active:
            return None

        if self.direction == "greater":
            return values >= self.value

        if self.direction == "less":
            return values <= self.value

        raise ValueError(self.direction)


class ThresholdOverlay:

    pick_px = 10
    background_order = -50.0
    region_order = 290.0
    line_order = 300.0

    def __init__(
        self,
        canvas,
        axis: ThresholdAxis = "x",
        on_changed=None,
        *,
        label=None,
        unit=None,
    ):
        self.canvas = canvas
        self.view = canvas.view
        self.on_changed = on_changed

        r = self.view.camera.rect
        self.spec = ThresholdSpec(
            axis=axis,
            value=(r.left + r.right) / 2 if axis == "x" else (r.bottom + r.top) / 2,
            direction="greater",
            active=False,
        )

        self.dragging = False
        self.hovered = False
        self.active_visual_state = False

        self.start_drag_value = None
        self.start_drag_pos = None

        self.label = label or axis
        self.unit = unit or ""
        self.tooltip_precision = 4

        self.visuals = {}

        self.visuals["region"] = visuals.Rectangle(
            center=(0, 0),
            width=1,
            height=1,
            color=THRESHOLD_STYLE["region"],
            border_color=None,
            parent=canvas.plot_root,
        )
        self.visuals["region"].set_gl_state(
            blend=True,
            depth_test=False,
            blend_func=("src_alpha", "one_minus_src_alpha"),
        )

        self.visuals["line"] = visuals.Line(
            pos=np.zeros((2, 2), dtype=np.float32),
            color=THRESHOLD_STYLE["inactive_line"],
            width=2,
            parent=canvas.plot_root,
        )
        self.visuals["line"].set_gl_state(
            blend=True,
            depth_test=False,
            blend_func=("src_alpha", "one_minus_src_alpha"),
        )

        self.set_active_visual_state(False, emit=False)
        self.update_visuals()

    def set_visible(self, visible: bool):
        self.visuals["line"].visible = bool(visible)

        # Region visibility depends on both global visibility and active state.
        self.visuals["region"].visible = bool(visible and self.active_visual_state)

    def set_active_visual_state(self, active: bool, *, emit=False):
        self.spec.active = bool(active)
        self.active_visual_state = bool(active)

        if self.active_visual_state:
            self.visuals["line"].order = self.line_order
            self.visuals["region"].order = self.region_order
            self.visuals["region"].visible = True
        else:
            self.visuals["line"].order = self.background_order
            self.visuals["region"].visible = False

        self._update_line_style()

        if emit:
            self._emit_changed()

    def activate_visuals(self):
        self.set_active_visual_state(True, emit=False)

    def deactivate_visuals(self):
        self.set_active_visual_state(False, emit=False)

    def reset_to_view_center(self, emit=False):
        xmin, xmax, ymin, ymax = self.view_bounds()

        if self.spec.axis == "x":
            self.spec.value = 0.5 * (xmin + xmax)
        elif self.spec.axis == "y":
            self.spec.value = 0.5 * (ymin + ymax)
        else:
            raise ValueError(self.spec.axis)

        self.update_visuals()

        if emit:
            self._emit_changed()

    def view_bounds(self):
        r = self.view.camera.rect
        return r.left, r.right, r.bottom, r.top

    def set_value(self, value, *, emit=True):
        self.spec.value = float(value)
        self.update_visuals()

        if emit:
            self._emit_changed()

    def toggle_direction(self, *, emit=True):
        self.spec.direction = "less" if self.spec.direction == "greater" else "greater"
        self.update_visuals()

        if emit:
            self._emit_changed()

    def _emit_changed(self):
        if self.on_changed is not None:
            self.on_changed(self.spec)

    def _line_color(self):
        if self.dragging or self.hovered:
            return THRESHOLD_STYLE["hover_line"]

        if self.active_visual_state:
            return THRESHOLD_STYLE["active_line"]

        return THRESHOLD_STYLE["inactive_line"]

    def _line_width(self):
        if self.dragging:
            return 4.0

        if self.hovered:
            return 3.0

        if self.active_visual_state:
            return 2.5

        return 1.0

    def _update_line_style(self):
        if not hasattr(self, "_last_line_pos"):
            return

        self.visuals["line"].set_data(
            pos=self._last_line_pos,
            color=self._line_color(),
            width=self._line_width(),
        )

    def update_visuals(self):
        xmin, xmax, ymin, ymax = self.view_bounds()
        v = self.spec.value

        if self.spec.axis == "x":
            line_pos = np.asarray(
                [[v, ymin], [v, ymax]],
                dtype=np.float32,
            )

            if self.spec.direction == "greater":
                x0, x1 = xmin, v
            else:
                x0, x1 = v, xmax

            y0, y1 = ymin, ymax

        elif self.spec.axis == "y":
            line_pos = np.asarray(
                [[xmin, v], [xmax, v]],
                dtype=np.float32,
            )

            x0, x1 = xmin, xmax

            if self.spec.direction == "greater":
                y0, y1 = ymin, v
            else:
                y0, y1 = v, ymax

        else:
            raise ValueError(self.spec.axis)

        self._last_line_pos = line_pos

        self.visuals["line"].set_data(
            pos=line_pos,
            color=self._line_color(),
            width=self._line_width(),
        )

        self.visuals["region"].center = (
            0.5 * (x0 + x1),
            0.5 * (y0 + y1),
        )
        self.visuals["region"].width = max(0.0, x1 - x0)
        self.visuals["region"].height = max(0.0, y1 - y0)
        self.visuals["region"].color = THRESHOLD_STYLE["region"]
        self.visuals["region"].visible = self.active_visual_state

        self.canvas.update()

    def line_distance_px(self, event_pos):
        """
        something seems to go wrong here: hover on y-threshold disappears quickly on hover
        """
        xmin, xmax, ymin, ymax = self.view_bounds()
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)

        if self.spec.axis == "x":
            pts = np.asarray([[self.spec.value, cy]], dtype=np.float32)
            pts_canvas = click_events.visual_to_canvas(self.visuals["line"], pts)
            return abs(pts_canvas[0, 0] - event_pos[0])

        if self.spec.axis == "y":
            pts = np.asarray([[cx, self.spec.value]], dtype=np.float32)
            pts_canvas = click_events.visual_to_canvas(self.visuals["line"], pts)
            return abs(pts_canvas[0, 1] - event_pos[1])

        raise ValueError(self.spec.axis)

    def is_near_line(self, event_pos) -> bool:
        return self.line_distance_px(event_pos) <= self.pick_px

    def update_hover(self, event_pos):
        old_hovered = self.hovered
        self.hovered = self.is_near_line(event_pos)

        if old_hovered != self.hovered:
            self.update_visuals()

        return self.hovered

    def show_tooltip(self):
        QToolTip.showText(
            QCursor.pos(),
            self.tooltip_text(),
            self.canvas.native,
        )

    def set_tooltip_parameter(
        self, label: str, unit: str = "", tooltip_precision: int = 4
    ):
        self.label = label
        self.unit = unit
        self.tooltip_precision = tooltip_precision

    def tooltip_text(self) -> str:
        value = f"{self.spec.value:.{self.tooltip_precision}g}"
        unit = f" {self.unit}" if self.unit else ""

        direction_symbol = "≥" if self.spec.direction == "greater" else "≤"

        return (
            f"<b>{self.label}</b> {direction_symbol} {value}{unit}<br>"
            f"<span style='color: gray;'>· Click: toggle direction<br>· Drag: move threshold</span>"
        )

    def handle_mouse_press(self, event) -> bool:
        if event.button != 1:
            return False

        if not self.is_near_line(event.pos):
            self.set_active_visual_state(False, emit=False)
            return False

        self.set_active_visual_state(True, emit=False)

        self.dragging = True
        self.hovered = True
        self.start_drag_value = self.spec.value
        self.start_drag_pos = np.asarray(event.pos, dtype=float)
        self.start_drag_time = time()

        self.update_visuals()

        event.handled = True
        return True

    def handle_mouse_move(self, event) -> bool:
        if not self.dragging:
            # self.update_hover(event.pos)
            return False

        xy = click_events.canvas_to_visual(self.visuals["line"], event.pos)

        if self.spec.axis == "x":
            value = xy[0]
        else:
            value = xy[1]

        self.set_value(value, emit=True)

        QToolTip.showText(
            QCursor.pos(),
            self.tooltip_text(),
            self.canvas.native,
        )

        event.handled = True
        return True

    def handle_mouse_release(self, event) -> bool:
        if not self.dragging:
            return False

        self.dragging = False

        moved_px = np.linalg.norm(
            np.asarray(event.pos, dtype=float) - self.start_drag_pos
        )
        spent_time = time() - self.start_drag_time

        if moved_px < self.pick_px and spent_time < 0.2:
            self.toggle_direction(emit=True)
        else:
            self._emit_changed()

        self.start_drag_value = None
        self.start_drag_pos = None

        self.hovered = self.is_near_line(event.pos)
        if self.hovered:
            QToolTip.showText(
                QCursor.pos(),
                self.tooltip_text(),
                self.canvas.native,
            )
        else:
            QToolTip.hideText()

        self.update_visuals()

        event.handled = True
        return True

    def destroy(self):
        self.dragging = False
        self.hovered = False

        for visual in self.visuals.values():
            try:
                visual.visible = False
                visual.parent = None
            except Exception:
                pass

        self.visuals.clear()
        self.canvas.update()
