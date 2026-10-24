from typing import Dict, Optional, Tuple
import textwrap
from pathlib import Path
from dataclasses import dataclass
from catan.gui.data.statistics.queries import ReductionSpec
import numpy as np
from scipy import sparse
from vispy import scene, color
from vispy.scene import visuals
from vispy.scene.visuals import Markers, Image, Rectangle, Text

from PySide6.QtCore import Qt, Signal, Slot, QEvent, QTimer
from shiboken6 import isValid
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QComboBox,
    QCheckBox,
    QLabel,
    QInputDialog,
    QToolTip,
    QVBoxLayout,
    QToolButton,
    QFormLayout,
    QDoubleSpinBox,
    QPushButton,
    QHBoxLayout,
    QSpinBox,
)

from catan.core.changes import (
    ChangeKind as C,
    DataChange,
    SESSION_STRUCTURE_CHANGES,
)
from catan.core.structures import NeuronComponent
from catan.gui.panels import BasePlot
from catan.gui.interaction import click_events

from catan.gui.data.statistics.dimensions import SESSION_DIMS, neuron_bound_dim
from catan.gui.panels.StatisticsData import StatisticQuerySelector
from catan.gui.panels.helper import ReviewStatusFilter, ControlPanel
from catan.gui.GUI_elements.utils.paths import display_path

STATISTIC_CMAPS = {
    "viridis": "viridis",
    "coolwarm": "coolwarm",
    "cubehelix": "cubehelix",
    "red-green": (
        "#b2182b",
        "#f2f2f2",
        "#1a9850",
    ),
    "blue-orange": (
        "#2166ac",
        "#f2f2f2",
        "#b35806",
    ),
}

# vispy colormap names offered for the background image.
BACKGROUND_CMAPS = (
    "grays",
    "hot",
    "fire",
    "ice",
    "viridis",
    "cubehelix",
    "blues",
    "greens",
    "reds",
)


@dataclass
class OverviewRecord:
    # visual: scene.Markers
    pos: np.ndarray
    ids: np.ndarray
    vals: np.ndarray
    n_rois: int
    color: list | tuple | str
    # Points grouped by neuron in ascending id order (union records):
    # a neuron's points are found by binary search.
    sorted_ids: bool = False


class Display(BasePlot.BaseCanvas):

    main_visual = Markers
    main_visual_name = "marker"

    overlays = ["selected", "focused", "highlighted", "hovered"]

    def __init__(self, parent, controls, config=None):
        super().__init__(parent, controls, config)

        self.unfreeze()

        self.central_widget.bgcolor = "#f5f6f7"
        self.central_widget.margin = 0
        self.central_widget.padding = 0
        self.central_widget.border_width = 0

        self.view = scene.widgets.ViewBox(
            parent=self.central_widget,
            bgcolor="#20252b",
            border_width=0,
        )
        self.view.margin = 0
        self.view.padding = 0

        self.central_widget.events.resize.connect(self._layout_overview_view)
        self._layout_overview_view()

        ## set camera
        self.view.camera = scene.PanZoomCamera(aspect=1)
        self.view.camera.interactive = False

        self.plot_root = scene.Node(parent=self.view.scene)

        self.common_boundary_overlay = Image(
            np.zeros((1, 1, 4), dtype=np.float32),
            interpolation="nearest",
            method="subdivide",
            parent=self.plot_root,
        )

        # Background: -1000; footprint visuals: 0 and above.
        self.common_boundary_overlay.order = -900
        self.common_boundary_overlay.interactive = False
        self.common_boundary_overlay.visible = False

        self.common_boundary_overlay.set_gl_state(
            depth_test=False,
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
        )

        self.changes_on_click = "selected"

        self.control_overlay = None
        self.events.resize.connect(self._on_canvas_resize)

        self.neuron_statistic_values = None
        self.statistic_title = None

        self.statistic_cmap_name = "viridis"
        self.statistic_cmap_reversed = False

        self.statistic_cmap = self._make_statistic_colormap(
            self.statistic_cmap_name,
            self.statistic_cmap_reversed,
        )

        self.statistic_data_clim = None
        self.statistic_manual_clim = {
            "low": None,
            "high": None,
        }
        self.build_colorbar()

        self.statistic_nan_color = np.asarray(
            color.Color("#686868").rgba,
            dtype=np.float32,
        )
        self.statistic_nan_color[3] = 0.55

        # Union points of unchanged neurons are reused on rebuilds.
        self._union_points = UnionPointCache()
        # Recoloring is applied once per event-loop pass.
        self._statistic_colors_pending = False

        self.freeze()

    def build_colorbar(self):

        self._cbar_size = (120, 10)
        self._cbar_center = (90, 20)
        cbar_size = self._cbar_size
        cbar_center = self._cbar_center
        cbar_hmargin = 15
        cbar_vmargin = 10

        self.statistic_colorbar_bg = Rectangle(
            center=(cbar_center[0], cbar_center[1] + 15),
            width=cbar_size[0] + 2 * cbar_hmargin,
            height=cbar_size[1] + 2 * cbar_vmargin + 30,
            color=(1.0, 1.0, 1.0, 0.78),
            border_color=None,
            parent=self.scene,
        )

        self.statistic_colorbar_bg.set_gl_state(
            depth_test=False,
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
        )
        self.statistic_colorbar_bg.order = 10000
        self.statistic_colorbar_bg.visible = False

        self.statistic_colorbar = visuals.ColorBar(
            cmap=self.statistic_cmap,
            orientation="bottom",
            size=self._cbar_size,
            pos=self._cbar_center,
            label="",
            clim=("", ""),
            border_width=1,
            border_color="black",
            parent=self.scene,
        )

        self.statistic_colorbar.order = 10001
        self.statistic_colorbar.visible = False

        label_y_offset = 25
        label_y = self._cbar_center[1] + label_y_offset

        left = self._cbar_center[0] - self._cbar_size[0] / 2
        right = self._cbar_center[0] + self._cbar_size[0] / 2

        self._cbar_low_label_pos = (left + 8, label_y)
        self._cbar_high_label_pos = (right - 8, label_y)

        # Slightly above the labels:
        self._cbar_low_reset_pos = (left + 22, label_y - 11)
        self._cbar_high_reset_pos = (right + 6, label_y - 11)

        self.statistic_colorbar_low = Text(
            "",
            pos=self._cbar_low_label_pos,
            anchor_x="center",
            anchor_y="top",
            font_size=9,
            color="black",
            parent=self.scene,
        )

        self.statistic_colorbar_high = Text(
            "",
            pos=self._cbar_high_label_pos,
            anchor_x="center",
            anchor_y="top",
            font_size=9,
            color="black",
            parent=self.scene,
        )

        self.statistic_colorbar_title = Text(
            "",
            pos=(cbar_center[0], cbar_center[1] + label_y_offset + 13),
            anchor_x="center",
            anchor_y="top",
            font_size=10,
            color="black",
            parent=self.scene,
        )

        self.statistic_colorbar_low_reset = Text(
            "×",
            pos=self._cbar_low_reset_pos,
            anchor_x="center",
            anchor_y="center",
            font_size=11,
            color="#707070",
            parent=self.scene,
        )

        self.statistic_colorbar_high_reset = Text(
            "×",
            pos=self._cbar_high_reset_pos,
            anchor_x="center",
            anchor_y="center",
            font_size=11,
            color="#707070",
            parent=self.scene,
        )

        for visual in (
            self.statistic_colorbar_low_reset,
            self.statistic_colorbar_high_reset,
        ):
            visual.order = 10003
            visual.visible = False

        for text in (
            self.statistic_colorbar_low,
            self.statistic_colorbar_high,
            self.statistic_colorbar_title,
        ):
            text.order = 10002
            text.visible = False

    def initialize_overlays(self):
        for style in self.overlays:
            if not self.plotting["overlays"].get(style):
                self.add_overlay(style)

    def _on_canvas_resize(self, event=None):
        self._position_overlay_controls()

    def attach_control_overlay(self):
        """
        Attach the Trace control widgets to the canvas after
        Controller.build_controls() has created them.
        """

        control_overlay = self.controls.get("panel")

        if control_overlay is None:
            return

        self.control_overlay = control_overlay

        self.control_overlay.setParent(self.native)
        self.control_overlay.show()
        self.control_overlay.raise_()

        self.control_overlay.overlay_layout_changed.connect(
            self._position_overlay_controls
        )

        self._position_overlay_controls()

    def _layout_overview_view(self, event=None):
        margin = 4.0
        width, height = self.central_widget.size

        self.view.pos = (margin, margin)
        self.view.size = (
            max(1.0, width - 2 * margin),
            max(1.0, height - 2 * margin),
        )

    def _position_overlay_controls(self):

        if self.control_overlay is None:
            return

        margin = 8
        spacing = 6

        palette = self.control_overlay
        palette.adjustSize()

        x = self.native.width() - palette.width() - margin

        palette.move(max(margin, x), margin)
        palette.raise_()

    def display_background(self):
        """Background image chosen in the options (aligned to the sessions'
        reference frame), or None."""
        panel = self.controls.get("panel")
        source = "current" if panel is None else panel.background_source.currentData()

        if source == "session":
            session_id = panel.background_session.value()
            if 0 <= session_id < len(self.data.sessions):
                return self.data.sessions[session_id].background
            return None

        if source == "file":
            background = getattr(self.data, "overview_background", None)
            return None if background is None else background.background

        session = self.data.current_session
        return None if session is None else session.background

    def background_colormap(self):
        panel = self.controls.get("panel")
        name = "grays" if panel is None else panel.background_cmap.currentText()
        reverse = panel is not None and panel.background_cmap_reverse.isChecked()

        if not reverse:
            return name

        x = np.linspace(1.0, 0.0, 256)[:, None]
        return color.Colormap(color.get_colormap(name).map(x))

    def update_background_colormap(self):
        image = self.plotting.get("background")

        if image is not None:
            image.cmap = self.background_colormap()
            self.update()

    def plot_background(self):

        if self.data.current_session is None:
            return

        ## remove previous background if exists
        if self.plotting.get("background") is not None:
            self.plotting["background"].parent = None
            del self.plotting["background"]

        background = self.display_background()
        if background is None:
            return

        self.plotting["background"] = Image(
            background.astype(np.float32),  # .T,
            cmap=self.background_colormap(),
            method="subdivide",
            parent=self.plot_root,
        )
        self.plotting["background"].order = -1000  # ensure it's in the back
        self.plotting["background"].set_gl_state(depth_test=False, blend=False)

        H, W = background.shape[:2]
        self.view.camera.set_range(
            x=(0, W),
            y=(0, H),
            margin=0,
        )
        self.update()

    def clean_highlight(self):
        for _, visual in self.plotting["overlays"].items():
            if visual is None:
                continue
            visual.visible = False

    @property
    def session_only(self) -> bool:
        """Show the current session's own footprints instead of the union."""
        panel = self.controls.get("panel")
        return bool(panel is not None and panel.session_only.isChecked())

    @property
    def footprint_threshold(self) -> float:
        """Applied display cutoff (fraction of each footprint's peak)."""
        panel = self.controls.get("panel")
        return DEFAULT_FOOTPRINT_THRESHOLD if panel is None else panel.applied_threshold

    def _tooltip_statistic_entity_mode(self) -> str:
        return "footprint" if self.session_only else "neuron"

    def build_neuron_visuals_union(self):

        if not self.data.is_available():
            return

        union = self.data.assignments.union

        cache = getattr(self, "_union_points", None)
        if cache is not None:
            roi_pos, neuron_ids, roi_vals = cache.points(
                union.footprints,
                union.dims,
                thr=self.footprint_threshold,
            )
        else:
            roi_pos, neuron_ids, roi_vals = sparse_A_to_points(
                union.footprints,
                union.dims,
                thr=self.footprint_threshold,
            )

        # Excluded neurons are not displayed.
        if union.included is not None:
            keep = np.asarray(union.included, dtype=bool)[neuron_ids]
            roi_pos, neuron_ids, roi_vals = roi_pos[keep], neuron_ids[keep], roi_vals[keep]

        self.plotting["data"]["union"] = OverviewRecord(
            pos=roi_pos,
            ids=neuron_ids,
            vals=roi_vals,
            n_rois=self.data.assignments.union.n_neurons,
            color=None,
            sorted_ids=True,
        )

    def build_neuron_visuals_session(self, session_id: Optional[int] = None, thr=None):

        if thr is None:
            thr = self.footprint_threshold

        if session_id is None:
            session_id = self.state.current_session_id
        if (
            session_id is None
            or session_id >= len(self.data.sessions)
            or self.data.sessions[session_id] is None
            or not self.data.sessions[session_id].status["spatial_loaded"]
            or not self.data.session_assigned(session_id)
            # Inactive sessions are not displayed.
            or not self.data.session_is_active(session_id)
        ):
            return

        if not self.session_only and "union" in self.plotting["data"]:
            present_neuron_ids = np.where(
                self.state.assignments[:, self.state.current_session_id] >= 0
            )[0]

            union = self.plotting["data"]["union"]

            mask = np.isin(union.ids, present_neuron_ids)

            roi_pos = union.pos[mask]
            roi_vals = union.vals[mask]
            neuron_ids = union.ids[mask]

            n_rois = len(present_neuron_ids)
        else:
            roi_pos, roi_ids, roi_vals = sparse_A_to_points(
                self.data.sessions[session_id].footprints,
                self.data.sessions[session_id].dims,
                thr=thr,
            )
            n_rois = self.data.sessions[session_id].footprints.get_shape()[1]

            footprint_to_component = np.array(
                [self.state.get_component_from_footprint(i) for i in range(n_rois)]
            )
            footprint_to_neuron_id = np.array(
                [c.neuron_id if c is not None else -1 for c in footprint_to_component]
            )
            neuron_ids = footprint_to_neuron_id[roi_ids]

            # Excluded footprints and footprints of excluded neurons are
            # not displayed.
            keep = np.ones(len(roi_ids), dtype=bool)
            session = self.data.sessions[session_id]

            if session.included is not None:
                keep &= np.asarray(session.included, dtype=bool)[roi_ids]

            union = self.data.assignments.union
            if union is not None and union.included is not None:
                assigned = neuron_ids >= 0
                keep[assigned] &= np.asarray(union.included, dtype=bool)[
                    neuron_ids[assigned]
                ]

            roi_pos, roi_vals, neuron_ids = roi_pos[keep], roi_vals[keep], neuron_ids[keep]

        self.plotting["data"][session_id] = OverviewRecord(
            pos=roi_pos,
            ids=neuron_ids,
            vals=roi_vals,
            n_rois=n_rois,
            # color=self.state.session_colors[session_id],
            color="white",
        )

    def reset_neuron_records(self):
        """Forget the neuron point records but keep their visuals: the next
        plot_neurons_visuals reuses them, and they show the previous points
        until the new ones are applied (removing them first showed an empty
        frame: footprints vanished and reappeared)."""
        self.plotting["data"] = {}

    def _reuse_visuals(self):
        """Visuals without a record are handed to records without a visual
        (e.g. the current session changed in session-only mode, or the mode
        was toggled), the rest are removed."""
        visuals = self.plotting["visuals"]
        data = self.plotting["data"]
        unused = [key for key in visuals if key not in data]

        for key in data:
            if key in visuals:
                continue
            donor = next(iter(unused), None)
            if donor is not None:
                visuals[key] = visuals.pop(donor)
                visuals[key].order = 0 if key == "union" else 1
                unused.remove(donor)

        for key in unused:
            visual = visuals.pop(key)
            if visual is not None:
                visual.visible = False
                visual.parent = None

    def plot_neurons_visuals(self):

        self._reuse_visuals()

        for key, record in self.plotting["data"].items():

            if key in self.plotting["visuals"]:
                continue

            footprints = Markers(
                parent=self.plot_root,
                scaling="scene",
                symbol="square",
            )
            footprints.set_gl_state(
                depth_test=False,
                blend=True,
                blend_func=(
                    "src_alpha",
                    "one_minus_src_alpha",
                ),  # Make overlaps visible
            )

            # Data (positions and colors) is set by _update_statistic_colors.

            if key == "union":
                footprints.order = 0
            else:
                footprints.order = 1
            self.plotting["visuals"][key] = footprints

        self._update_statistic_colors()

    def set_common_boundary(self, mask):
        visual = self.common_boundary_overlay
        visual.visible = False

        background = self.plotting.get("background")
        session = self.data.current_session

        if mask is None or background is None or session is None:
            self.update()
            return

        mask = np.asarray(mask, dtype=bool)

        if mask.shape != tuple(session.dims):
            self.update()
            return

        outside = ~mask

        if not outside.any():
            self.update()
            return

        rgba = np.zeros(mask.shape + (4,), dtype=np.float32)
        rgba[..., :3] = (1.0, 0.30, 0.30)
        rgba[..., 3] = outside * 0.16

        visual.set_data(rgba)

        # Follow any positioning applied to the background image.
        visual.transform = background.transform
        visual.visible = True
        self.update()

    def clean(self, with_union=True):

        for key, visual in self.plotting["visuals"].items():
            if visual is None or (not with_union and key == "union"):
                continue
            visual.visible = False
            visual.parent = None
        self.plotting["visuals"] = (
            {}
            if with_union
            else {k: v for k, v in self.plotting["visuals"].items() if k == "union"}
        )

        for key, data in self.plotting["data"].items():
            if not with_union and key == "union":
                continue
        self.plotting["data"] = (
            {}
            if with_union
            else {k: v for k, v in self.plotting["data"].items() if k == "union"}
        )
        # self.clear_overlays()

    # def clear_overlays(self):
    #     super().clear_overlays()
    #     self.initialize_overlays()

    ### ==================================================================== ###
    ### ======================= STATISTICS SECTION ========================= ###
    ### ==================================================================== ###

    def set_neuron_statistic_values(
        self,
        values,
        *,
        title=None,
    ):
        if not self.data.is_available():
            return

        self.neuron_statistic_values = (
            None if values is None else np.asarray(values, dtype=float)
        )

        self.statistic_title = title

        if self.neuron_statistic_values is None:
            self.statistic_data_clim = None

        else:
            finite = self.neuron_statistic_values[
                np.isfinite(self.neuron_statistic_values)
            ]

            self.statistic_data_clim = (
                None
                if finite.size == 0
                else (float(np.min(finite)), float(np.max(finite)))
            )

        self._update_statistic_colorbar()
        self._update_statistic_colors()

    def _effective_statistic_clim(self):

        if self.statistic_data_clim is None:
            return None

        data_lo, data_hi = self.statistic_data_clim

        manual_lo = self.statistic_manual_clim["low"]
        manual_hi = self.statistic_manual_clim["high"]

        lo = data_lo if manual_lo is None else manual_lo
        hi = data_hi if manual_hi is None else manual_hi

        # Possible if one bound is fixed and a newly displayed
        # dataset lies completely beyond that bound.
        if lo >= hi:

            if manual_lo is not None and manual_hi is None:
                hi = np.nextafter(lo, np.inf)

            elif manual_hi is not None and manual_lo is None:
                lo = np.nextafter(hi, -np.inf)

            # elif manual_lo is not None and manual_hi is not None:
            else:
                self.set_statistic_clim_limit("low", None)
                self.set_statistic_clim_limit("high", None)

            # else:
            # return None

        return float(lo), float(hi)

    def set_statistic_clim_limit(
        self,
        which: str,
        value: float | None,
    ):

        if which not in ("low", "high"):
            raise ValueError(which)

        if value is not None:
            value = float(value)

            if not np.isfinite(value):
                return

            other_key = "high" if which == "low" else "low"

            other = self.statistic_manual_clim[other_key]

            if other is not None:
                if which == "low" and value >= other:
                    return

                if which == "high" and value <= other:
                    return

        self.statistic_manual_clim[which] = value

        self._update_statistic_colorbar()
        self._update_statistic_colors()

    def _colors_for_record(
        self,
        key,
        record,
    ):
        # Statistic coloring takes precedence
        if self.neuron_statistic_values is not None:
            return (
                self._statistic_colors_for_record(record),
                "default",
            )

        # Tracked overview: union contains everything
        if key == "union":
            return (
                self._tracked_union_colors(record),
                "default",
            )

        # Session overview: ordinary session color
        return (
            record.color,
            "default",
        )

    def _statistic_colors_for_record(
        self,
        record: OverviewRecord,
    ):

        ids = np.asarray(record.ids, dtype=int)

        point_values = np.full(ids.shape, np.nan, dtype=float)

        if self.neuron_statistic_values is not None:

            valid_ids = (ids >= 0) & (ids < len(self.neuron_statistic_values))

            point_values[valid_ids] = self.neuron_statistic_values[ids[valid_ids]]

        opts = self.styles.overview

        nan_color = np.asarray(
            color.Color(opts["statistic_nan_color"]).rgba,
            dtype=np.float32,
        )

        nan_color[3] = opts["statistic_nan_alpha"]

        rgba = np.tile(nan_color, (len(ids), 1))

        if self.statistic_data_clim is None or self.neuron_statistic_values is None:
            return rgba

        finite = np.isfinite(point_values)

        if not np.any(finite):
            return rgba

        clim = self._effective_statistic_clim()

        lo, hi = clim
        # lo, hi = self.statistic_data_clim

        if hi > lo:
            normalized = (point_values[finite] - lo) / (hi - lo)
        else:
            normalized = np.full(np.count_nonzero(finite), 0.5)

        normalized = np.clip(normalized, 0.0, 1.0)

        rgba[finite] = self.statistic_cmap.map(normalized).astype(np.float32)

        if (
            not self.session_only
            and record is self.plotting["data"].get("union")
            and self.state.current_session_id is not None
        ):

            current = self.state.assignments[:, self.state.current_session_id] >= 0

            ids = np.asarray(record.ids, dtype=int)

            valid_ids = (ids >= 0) & (ids < len(current))

            point_in_current = np.zeros(len(ids), dtype=bool)

            point_in_current[valid_ids] = current[ids[valid_ids]]

            # Neurons not present in current session remain visible,
            # but substantially dimmer.
            rgba[~point_in_current, 3] *= opts["tracked_stat_other_alpha_scale"]

        return rgba

    def _point_near(
        self,
        pos,
        target,
        *,
        radius_x=12,
        radius_y=10,
    ):
        return (
            abs(pos[0] - target[0]) <= radius_x and abs(pos[1] - target[1]) <= radius_y
        )

    def _colorbar_hit_target(self, pos):

        if not self.statistic_colorbar.visible:
            return None

        # Reset controls have priority.
        if self.statistic_colorbar_low_reset.visible and self._point_near(
            pos, self._cbar_low_reset_pos
        ):
            return "reset_low"

        if self.statistic_colorbar_high_reset.visible and self._point_near(
            pos, self._cbar_high_reset_pos
        ):
            return "reset_high"

        if self._point_near(pos, self._cbar_low_label_pos, radius_x=25, radius_y=12):
            return "low"

        if self._point_near(pos, self._cbar_high_label_pos, radius_x=25, radius_y=12):
            return "high"

        return None

    def _update_colorbar_hover(self, target):

        low_hover = target == "reset_low"
        high_hover = target == "reset_high"

        self.statistic_colorbar_low_reset.color = "#c62828" if low_hover else "#707070"

        self.statistic_colorbar_high_reset.color = (
            "#c62828" if high_hover else "#707070"
        )

        self.statistic_colorbar_low_reset.font_size = 13 if low_hover else 11
        self.statistic_colorbar_high_reset.font_size = 13 if high_hover else 11

    ### =========================== COLORSTYLES ============================ ###

    def _make_statistic_colormap(
        self,
        name: str,
        reverse: bool,
    ):
        spec = STATISTIC_CMAPS[name]

        if isinstance(spec, str):
            base = color.get_colormap(spec)
        else:
            base = color.Colormap(spec)

        x = np.linspace(0.0, 1.0, 256)

        if reverse:
            x = x[::-1]

        # Always return a fresh deterministic colormap.
        return color.Colormap(base.map(x))

    def _rebuild_statistic_colorbar_gradient(self):

        was_visible = (
            self.statistic_colorbar.visible
            if self.statistic_colorbar is not None
            else False
        )

        if self.statistic_colorbar is not None:
            self.statistic_colorbar.parent = None

        self.statistic_colorbar = visuals.ColorBar(
            cmap=self.statistic_cmap,
            orientation="bottom",
            size=self._cbar_size,
            pos=self._cbar_center,
            label="",
            clim=("", ""),
            border_width=1,
            border_color="black",
            parent=self.scene,
        )

        self.statistic_colorbar.order = 10001
        self.statistic_colorbar.visible = was_visible

    def set_statistic_colormap(
        self,
        name: str,
        *,
        reverse: bool = False,
    ):
        self.statistic_cmap_name = name
        self.statistic_cmap_reversed = reverse

        self.statistic_cmap = self._make_statistic_colormap(
            name,
            reverse,
        )

        self._rebuild_statistic_colorbar_gradient()
        self._update_statistic_colorbar()
        self._update_statistic_colors()

    def _update_statistic_colorbar(self):

        visible = (
            self.neuron_statistic_values is not None
            and self.statistic_data_clim is not None
        )

        self.statistic_colorbar.visible = visible
        self.statistic_colorbar_bg.visible = visible

        for text in (
            self.statistic_colorbar_low,
            self.statistic_colorbar_high,
            self.statistic_colorbar_title,
        ):
            text.visible = visible

        if not visible:
            return

        clim = self._effective_statistic_clim()

        lo, hi = clim

        lo_text, hi_text = format_colorbar_limits(
            lo,
            hi,
        )

        self.statistic_colorbar_low.text = lo_text
        self.statistic_colorbar_high.text = hi_text

        self.statistic_colorbar_low_reset.visible = (
            visible and self.statistic_manual_clim["low"] is not None
        )

        self.statistic_colorbar_high_reset.visible = (
            visible and self.statistic_manual_clim["high"] is not None
        )

        self.statistic_colorbar_title.text = self._format_colorbar_title(
            self.statistic_title or ""
        )

    def _edit_statistic_clim(
        self,
        which: str,
    ):
        clim = self._effective_statistic_clim()

        current = clim[0] if which == "low" else clim[1]

        value, accepted = QInputDialog.getDouble(
            self.native,
            ("Lower color limit" if which == "low" else "Upper color limit"),
            "Value:",
            current,
            -1e100,
            1e100,
            decimals=6,
        )

        if accepted:
            self.set_statistic_clim_limit(
                which,
                value,
            )

    def _update_statistic_colors(self):
        """
        Recolor existing neuron visuals from the current statistic.

        Does not rebuild positions/footprints. Requests are applied once per
        event-loop pass (one manipulation requested it ~3×, ~70 ms each).
        """
        if not hasattr(self, "_statistic_colors_pending"):
            # Canvas created before a hot reload of this module.
            self._apply_statistic_colors()
            return

        if self._statistic_colors_pending:
            return

        self._statistic_colors_pending = True
        QTimer.singleShot(0, self._apply_statistic_colors)

    def _apply_statistic_colors(self):
        if getattr(self, "_statistic_colors_pending", False):
            self._statistic_colors_pending = False

        if not isValid(self.native):
            return

        for key, record in self.plotting["data"].items():

            visual = self.plotting["visuals"].get(key)

            if visual is None:
                continue
            if (
                self.neuron_statistic_values is not None
                and not self.session_only
                and key != "union"
            ):
                visual.visible = False
                continue

            mask = self._review_mask_for_record(record)

            pos = record.pos[mask]
            if pos.shape[0] == 0:
                visual.visible = False
                continue

            vals = record.vals[mask]

            # -------------------------------------------
            # No statistic selected:
            # restore ordinary overview colors
            # -------------------------------------------
            if self.neuron_statistic_values is None:

                if key == "union":
                    colors = self._tracked_union_colors(record)
                    style = "default"
                else:
                    # colors = self.state.session_colors[key]
                    colors = "white"

                    style = (
                        "default"
                        if key == self.state.current_session_id
                        else "background"
                    )

            # -------------------------------------------
            # Statistic selected
            # -------------------------------------------
            else:
                colors = self._statistic_colors_for_record(record)

                # For now all statistic-colored data use the
                # ordinary visual style. The colors themselves
                # encode the statistic.
                style = "default"

            if (
                isinstance(colors, np.ndarray)
                and colors.ndim == 2
                and len(colors) == len(record.ids)
            ):
                colors = colors[mask]

            plot_options = self.styles.get_plot_options(
                style=style,
                plot_type="marker",
                values=vals,
                colors=colors,
                edge_width=0,
            )

            visual.visible = True
            visual.set_data(pos, **plot_options)

        self.update()

    def _format_colorbar_title(
        self,
        title: str,
    ) -> str:
        return "\n".join(
            textwrap.wrap(
                title,
                width=24,
                break_long_words=False,
            )
        )

    def _tracked_union_colors(
        self,
        record: OverviewRecord,
    ):
        ids = np.asarray(
            record.ids,
            dtype=int,
        )

        opts = self.styles.overview

        # Start with faint neutral colour.
        other = np.asarray(
            color.Color(opts["tracked_other_color"]).rgba,
            dtype=np.float32,
        )
        other[3] = opts["tracked_other_alpha"]

        rgba = np.tile(
            other,
            (len(ids), 1),
        )

        current_session = self.state.current_session_id or 0
        # Current-session neurons get the actual session colour.
        current_color = np.asarray(
            # color.Color(self.state.session_colors[current_session]).rgba,
            color.Color("white").rgba,
            dtype=np.float32,
        )

        current_color[3] = opts["tracked_current_alpha"]

        if (
            self.state.assignments is not None
            and current_session < self.state.assignments.shape[1]
        ):
            current_neurons = self.state.assignments[:, current_session] >= 0

            valid = (ids >= 0) & (ids < len(current_neurons))

            in_current = np.zeros(
                len(ids),
                dtype=bool,
            )
            in_current[valid] = current_neurons[ids[valid]]

            rgba[in_current] = current_color

        return rgba

    ### ==================================================================== ###
    ### ======================= GENERAL STYLE HELPER ======================= ###
    ### ==================================================================== ###

    def update_session_styles(self, session_id):

        if session_id not in self.plotting["visuals"]:
            return

        if session_id == "union":
            self.plotting["visuals"][session_id].visible = True
        else:
            self.plotting["visuals"][session_id].visible = self.data.sessions[
                session_id
            ].active

        self._update_statistic_colors()

    def update_style(self, component, style="default"):

        visuals = self.plotting["overlays"].get(style, [])
        if component is None:
            for vis in visuals:
                vis.visible = False
            self.update()
            return

        if not visuals:
            self.add_overlay(style)

        if not isinstance(component, list):
            component = [component]

        if self.session_only:
            key = self.state.current_session_id
        else:
            key = "union"

        if key is None or key not in self.plotting["data"]:
            ## can happen on session unregistration
            self.plotting["overlays"][style][0].visible = False
            self.update()
            return

        neuron_ids = [c.neuron_id for c in component]

        record = self.plotting["data"][key]

        if getattr(record, "sorted_ids", False):
            # Union points are grouped by neuron: no scan of all points.
            ids = np.asarray(record.ids)
            wanted = np.unique(np.asarray(neuron_ids, dtype=ids.dtype))
            starts = np.searchsorted(ids, wanted, side="left")
            stops = np.searchsorted(ids, wanted, side="right")
            index = (
                np.concatenate([np.arange(a, b) for a, b in zip(starts, stops)])
                if len(wanted)
                else np.zeros(0, dtype=int)
            )
            index = index[self._review_mask_for_ids(ids[index])]
        else:
            mask = np.isin(record.ids, neuron_ids)
            mask &= self._review_mask_for_record(record)
            index = np.flatnonzero(mask)

        if not len(index):
            self.plotting["overlays"][style][0].visible = False
            self.update()
            return
        vals = record.vals[index]
        vals += 0.5

        plot_options = self.styles.get_plot_options(style, "marker", vals, edge_width=0)

        self.plotting["overlays"][style][0].set_data(
            record.pos[index].astype(np.float32),
            **plot_options,
        )
        self.plotting["overlays"][style][0].visible = True
        self.update()

    def find_closest_component(self, mouse_pos) -> Optional[NeuronComponent]:
        # print("neurons:", self.data.neurons)
        if (
            self.state.current_session_id is None
            or "background" not in self.plotting
            or self.data.assignments is None
            or self.data.assignments.union is None
            or self.data.assignments.union.centroids is None
        ):
            return None

        mouse_pos = click_events.canvas_to_visual(
            self.plotting["background"], mouse_pos
        )

        # union_centroids = np.nanmean(self.data.neurons.centroids, axis=1)
        union_centroids = self.data.assignments.union.centroids

        # find closest footprint
        distances = (union_centroids[:, 0] - mouse_pos[0]) ** 2 + (
            union_centroids[:, 1] - mouse_pos[1]
        ) ** 2

        ## filter results for display based on review status
        neuron_ids = np.arange(len(union_centroids), dtype=int)
        focused_neuron_id = (
            None
            if self.state.focused_component is None
            else self.state.focused_component.neuron_id
        )

        review_mask = ReviewStatusFilter.neuron_mask(
            neuron_ids,
            self.data.assignments.review_status,
            (self.controls["panel"].review_filter.visible_statuses),
            focused_neuron_id=focused_neuron_id,
            keep_focused=True,
        )
        distances[~review_mask] = np.inf

        # Excluded neurons are not displayed, so they cannot be hovered.
        union = self.data.assignments.union
        if union.included is not None:
            distances[~np.asarray(union.included, dtype=bool)] = np.inf

        if self.session_only:
            mask = self.state.assignments[:, self.state.current_session_id] >= 0
            distances[~mask] = np.inf

        distances[~np.isfinite(distances)] = np.inf
        if not np.any(np.isfinite(distances)):
            return None

        neuron_id = np.argmin(distances).astype(int)
        if np.sqrt(distances[neuron_id]) > 10.0:
            return None

        return NeuronComponent(neuron_id, self.state.current_session_id)

    def on_mouse_move(self, event):

        target = self._colorbar_hit_target(event.pos)

        self._update_colorbar_hover(target)

        if target is not None:
            # Don't hover a neuron underneath the controls.
            QToolTip.hideText()
            event.handled = True
            self.update()
            return

        super().on_mouse_move(event)

    def on_mouse_release(self, event):

        if event.button == 1:

            target = self._colorbar_hit_target(event.pos)

            if target == "reset_low":
                self.set_statistic_clim_limit("low", None)
                event.handled = True
                return

            if target == "reset_high":
                self.set_statistic_clim_limit("high", None)
                event.handled = True
                return

            if target == "low":
                self._edit_statistic_clim("low")
                event.handled = True
                return

            if target == "high":
                self._edit_statistic_clim("high")
                event.handled = True
                return

        super().on_mouse_release(event)

    def _review_mask_for_record(self, record: OverviewRecord) -> np.ndarray:
        return self._review_mask_for_ids(record.ids)

    def _review_mask_for_ids(self, ids) -> np.ndarray:
        """Review-filter visibility of each point (by its neuron id)."""
        ids = np.asarray(ids, dtype=int)

        mask = np.zeros(ids.shape, dtype=bool)

        if self.data.assignments is None:
            return mask

        review_status = self.data.assignments.review_status

        valid = (ids >= 0) & (ids < len(review_status))

        if not np.any(valid):
            return mask

        focused_neuron_id = (
            None
            if self.state.focused_component is None
            else self.state.focused_component.neuron_id
        )

        # Per neuron, then gathered per point (points >> neurons).
        visible = ReviewStatusFilter.neuron_mask(
            np.arange(len(review_status)),
            review_status,
            (self.controls["panel"].review_filter.visible_statuses),
            focused_neuron_id=focused_neuron_id,
            keep_focused=True,
        )
        mask[valid] = visible[ids[valid]]

        return mask


class Controller(BasePlot.CanvasController):

    canvas: Display  # type hint for better code completion

    def build_controls(self):

        self._boundary_task_id = None
        self._boundary_generation = 0
        # Inputs (_boundary_inputs key) of the shown / calculating border.
        self._boundary_key = None
        self._boundary_mask = None
        self._boundary_pending_key = None

        self.current_statistic_query = None
        self.statistic_table = None

        self.controls["panel"] = OverviewControlPanel(
            self.section,
            session_only=bool(self.config.get("session_only", False)),
            threshold=float(
                self.config.get("footprint_threshold", DEFAULT_FOOTPRINT_THRESHOLD)
            ),
            background_source=self.config.get("background_source", "current"),
            background_session=int(self.config.get("background_session", 0)),
            background_cmap=self.config.get("background_cmap", "grays"),
            background_cmap_reversed=bool(
                self.config.get("background_cmap_reversed", False)
            ),
            border_tolerance=getattr(self.data, "border_tolerance", 0),
        )
        self.canvas.attach_control_overlay()
        self.controls["panel"].display_settings_changed.connect(
            self._on_display_settings_changed
        )
        self.controls["panel"].background_changed.connect(self._on_background_changed)
        self.controls["panel"].background_load_requested.connect(
            self._load_overview_background
        )
        self.controls["panel"].background_style_changed.connect(
            self._on_background_style_changed
        )
        self.controls["panel"].border_changed.connect(self._on_border_changed)
        self._update_background_controls()

        self.controls["statistics"] = StatisticQuerySelector(
            engine=self.data.statistic_engine,
        )
        self.section.x_options_layout.addWidget(self.controls["statistics"])

        self.controls["statistics"].set_default_reduction_provider(
            self._overview_statistic_defaults
        )
        self.controls["statistics"].queryChanged.connect(
            self._on_statistic_query_changed
        )
        self.controls["statistics"].set_query_mode("neuron_bound")

        self.connect_signal(
            self.data.statistic_engine.values_changed,
            self._on_statistics_values_changed,
            controls=True,
        )

        self.controls["stat_cmap_label"] = QLabel("Colormap:")

        self.controls["stat_cmap"] = QComboBox()
        self.controls["stat_cmap"].addItems(STATISTIC_CMAPS.keys())

        self.controls["stat_reverse"] = QCheckBox("Reverse")

        self.section.x_options_layout.addWidget(self.controls["stat_cmap_label"])
        self.section.x_options_layout.addWidget(self.controls["stat_cmap"])
        self.section.x_options_layout.addWidget(self.controls["stat_reverse"])

        self.controls["stat_cmap"].currentTextChanged.connect(
            self._on_statistic_color_options_changed
        )

        self.controls["stat_reverse"].toggled.connect(
            self._on_statistic_color_options_changed
        )

        self.controls["panel"].review_filter.changed.connect(
            self._on_review_filter_changed
        )

    def _on_display_settings_changed(self):
        panel = self.controls["panel"]
        self.config["session_only"] = panel.session_only.isChecked()
        self.config["footprint_threshold"] = panel.applied_threshold

        # Rebuild the footprint points (union or session) with the settings.
        self.initialize_display()

    def _on_background_changed(self):
        panel = self.controls["panel"]
        self.config["background_source"] = panel.background_source.currentData()
        self.config["background_session"] = panel.background_session.value()
        self._update_background_controls()
        self.canvas.plot_background()

    def _on_border_changed(self):
        # Shared with the border statistic; the change notification
        # refreshes the overlay of every Overview.
        self.data.set_border_tolerance(
            self.controls["panel"].applied_border_tolerance
        )

    def _on_background_style_changed(self):
        panel = self.controls["panel"]
        self.config["background_cmap"] = panel.background_cmap.currentText()
        self.config["background_cmap_reversed"] = (
            panel.background_cmap_reverse.isChecked()
        )
        self.canvas.update_background_colormap()

    def _update_background_controls(self):
        panel = self.controls["panel"]
        panel.background_session.setMaximum(max(0, len(self.data.sessions) - 1))

        background = getattr(self.data, "overview_background", None)
        panel.background_file_label.setText(
            "" if background is None else Path(str(background.path)).name
        )
        panel.background_file_label.setToolTip(
            "" if background is None else display_path(background.path)
        )

    def _load_overview_background(self):
        from catan.gui.GUI_elements.fragments.dialog_load_field import (
            FieldSelectDialog,
        )
        from catan.gui.GUI_elements.fragments.path_selector import choose_path

        path = choose_path(
            self.section,
            init_path=str(getattr(self.data, "root", "") or ""),
            display_text="Select background image or file",
            only_existing=True,
            state=self.state,
        )
        if not path:
            return

        selection = FieldSelectDialog.get_field(
            path=str(path),
            key="background",
            title="Select background",
            context="Overview background",
            parent=self.section,
        )
        if selection is None:
            return

        self.data.queue_load_overview_background(path, selection)

    def _cancel_boundary_task(self):
        self._boundary_generation += 1

        task_id = self._boundary_task_id
        self._boundary_task_id = None

        if task_id is not None:
            self.state.tasks.cancel(task_id)

    def _boundary_inputs(self):
        """Sessions the shared border is computed from, and a key of all it
        depends on (alignments, tolerance), so unrelated data changes (e.g.
        manipulations) do not recompute it."""
        # Sessions with current alignments only: unaligned or outdated ones
        # (during loading/realignment) are left out instead of hiding the
        # overlay.
        usable = [
            (session_id, session)
            for session_id, session in enumerate(self.data.sessions)
            if session is not None
            and session.status.get("spatial_loaded", False)
            and session.status.get("aligned", False)
            and not self.data.alignment_is_stale(session_id)
        ]
        tolerance = getattr(self.data, "border_tolerance", 0)
        key = (
            id(self.data.assignments),
            tolerance,
            tuple(
                (id(session), id(session.remap), self.data.geometry_revision(session_id))
                for session_id, session in usable
            ),
        )
        return tuple(session for _, session in usable), tolerance, key

    def _refresh_common_boundary(self):
        if self._deactivated:
            self._cancel_boundary_task()
            return

        assignments = self.data.assignments

        if (
            assignments is None
            or self.state.current_session_id is None
            or self.data.current_session is None
        ):
            self._cancel_boundary_task()
            self.canvas.set_common_boundary(None)
            return

        sessions, tolerance, key = self._boundary_inputs()

        if not sessions:
            self._cancel_boundary_task()
            self.canvas.set_common_boundary(None)
            self._boundary_key = self._boundary_mask = None
            return

        # A display task may have been cancelled for processing (then no
        # result arrives).
        running = (
            self._boundary_task_id is not None
            and self.state.tasks.get_task(self._boundary_task_id) is not None
        )

        # Unchanged inputs: show the stored mask again (it follows the
        # background); nothing to calculate.
        if key == getattr(self, "_boundary_key", None) and not running:
            self.canvas.set_common_boundary(self._boundary_mask)
            return

        if key == getattr(self, "_boundary_pending_key", None) and running:
            return

        self._cancel_boundary_task()
        self.canvas.set_common_boundary(None)
        self._boundary_pending_key = key

        generation = self._boundary_generation
        # Lets Assignments.common_spatial_coverage reuse its cached result.
        revision = key

        def calculate():
            from catan.gui.background_tasks.runtime import (
                current_task_context,
            )

            ctx = current_task_context()
            if ctx is not None:
                ctx.check_cancelled()

            try:
                mask, _ = assignments.common_spatial_coverage(
                    sessions,
                    revision=revision,
                    tolerance=tolerance,
                )
            except ValueError:
                # Coverage can become unavailable while loading changes.
                # Do not turn an optional visual into a modal error.
                return None

            if ctx is not None:
                ctx.check_cancelled()

            return mask

        self._boundary_task_id = self.state.tasks.start(
            "calculating",
            "Calculate shared imaging boundary",
            calculate,
            on_result=lambda mask: self._on_common_boundary_ready(
                mask,
                generation,
                revision,
            ),
            background=True,
        )

    def _on_common_boundary_ready(self, mask, generation, revision):
        if self._deactivated or generation != self._boundary_generation:
            return

        self._boundary_task_id = None
        self._boundary_pending_key = None

        if revision != self._boundary_inputs()[2]:
            # The inputs changed during calculation. Request current data.
            self._refresh_common_boundary()
            return

        self._boundary_key = revision
        self._boundary_mask = mask
        self.canvas.set_common_boundary(mask)

    def _on_statistic_query_changed(self, query):
        self.current_statistic_query = query
        self._recalculate_statistic()

    def _on_statistics_values_changed(self, statistic_keys):
        query = self.current_statistic_query

        if query is not None and query.statistic_key in statistic_keys:
            self._recalculate_statistic()

    def _recalculate_statistic(self):
        if self._deactivated:
            return

        self._cancel_background_tasks()

        query = self.current_statistic_query

        if query is None:
            self.statistic_table = None
            self.canvas.set_neuron_statistic_values(
                None,
                title=None,
            )
            return

        revision = self.data.statistic_engine.query_revision(query)

        def evaluate():
            return self.data.statistic_engine.evaluate_table(query)

        self._statistic_task_id = self.state.tasks.start(
            "calculating",
            f"Calculate {query.statistic_key}",
            evaluate,
            on_result=lambda table, query=query, revision=revision: (
                self._on_statistic_ready(query, table, revision)
            ),
            background=True,
        )

    def _on_statistic_ready(self, query, table, revision):
        if self._deactivated:
            return

        if query != self.current_statistic_query:
            return

        if revision != self.data.statistic_engine.query_revision(query):
            return

        self.statistic_table = table

        if table is None:
            self.canvas.set_neuron_statistic_values(
                None,
                title=None,
            )
            return
        values = self._dense_neuron_values(table)

        definition = self.data.statistic_engine.registry[query.statistic_key]

        self.canvas.set_neuron_statistic_values(values, title=definition.title)

    def _dense_neuron_values(self, table):

        neuron_dim = neuron_bound_dim(table.dims)

        if neuron_dim is None:
            raise ValueError(
                "Neuron-bound query returned no neuron dimension: " f"{table.dims}"
            )

        return table.indexed_values(
            neuron_dim,
            self.state.assignments.shape[0],
        )

    def _overview_statistic_defaults(
        self,
        stat_def,
    ):
        reductions = {}

        session_dims = [dim for dim in stat_def.dims if dim in SESSION_DIMS]

        if self.canvas.session_only:
            session_id = self.state.current_session_id

            if session_id is not None:
                for dim in session_dims:
                    reductions[dim] = ReductionSpec(
                        "single",
                        index=session_id,
                    )

        else:
            for dim in session_dims:
                reductions[dim] = ReductionSpec("mean")

        return reductions

    def _on_statistic_color_options_changed(
        self,
        *_,
    ):
        self.canvas.set_statistic_colormap(
            self.controls["stat_cmap"].currentText(),
            reverse=self.controls["stat_reverse"].isChecked(),
        )

    def initialize_display(self):

        # Visuals are reused by the rebuild (no empty frame in between).
        self.canvas.reset_neuron_records()
        self.canvas.initialize_overlays()

        self._on_session_changed()

    def _on_data_changed(self, event: DataChange):
        rebuild = event.has(
            *SESSION_STRUCTURE_CHANGES,
            C.ASSIGNMENT_SET,
            C.FOOTPRINT_GEOMETRY,
            C.BACKGROUND_IMAGE,
            C.SESSION_ACTIVITY,
            C.SESSION_METADATA,
        ) or event.has_availability("spatial")

        # Assignment content (manipulations, union rebuilds, inclusion):
        # only the neuron points change.
        content = event.has(
            C.ASSIGNMENT_MAPPING,
            C.UNION_GEOMETRY,
            C.INCLUSION,
        )

        if event.has(C.OVERVIEW_BACKGROUND, *SESSION_STRUCTURE_CHANGES):
            self._update_background_controls()

        if rebuild:
            self.initialize_display()
        elif content:
            self._refresh_neuron_visuals()
            if event.has(C.OVERVIEW_BACKGROUND):
                self.canvas.plot_background()
            if event.has(C.PROCESSING_STATUS, C.BORDER_SETTINGS):
                self._refresh_common_boundary()
        elif event.has(C.OVERVIEW_BACKGROUND):
            self.canvas.plot_background()
        elif event.has(C.REVIEW_STATUS):
            self._on_review_filter_changed()
        elif event.has(C.PROCESSING_STATUS, C.BORDER_SETTINGS):
            if event.has(C.BORDER_SETTINGS):
                panel = self.controls["panel"]
                blocked = panel.border_tolerance.blockSignals(True)
                panel.border_tolerance.setValue(self.data.border_tolerance)
                panel.border_tolerance.blockSignals(blocked)
            self._refresh_common_boundary()

    def _on_session_changed(self):
        if (
            self.data.assignments is None
            or self.data.assignments.ids.shape[1] == 0
            or self.state.current_session_id is None
        ):
            return

        # A fixed background (session X, file) does not follow the session.
        source = self.controls["panel"].background_source.currentData()
        if source == "current" or self.canvas.plotting.get("background") is None:
            self.canvas.plot_background()
        self._refresh_common_boundary()
        if not self.canvas.session_only:
            if "union" not in self.canvas.plotting["data"]:
                self.canvas.build_neuron_visuals_union()
            # Only the union record stays; visuals are reused when plotting.
            self.canvas.plotting["data"] = {
                key: record
                for key, record in self.canvas.plotting["data"].items()
                if key == "union"
            }

        else:
            self.canvas.reset_neuron_records()

            self.canvas.build_neuron_visuals_session(
                session_id=self.state.current_session_id,
            )

        self.canvas.plot_neurons_visuals()

        super()._on_session_changed()

    def _refresh_neuron_visuals(self):
        """Rebuild the neuron points after assignment content changed;
        background, shared border and overlays' visuals are kept."""
        if (
            self.data.assignments is None
            or self.data.assignments.ids.shape[1] == 0
            or self.state.current_session_id is None
        ):
            return

        canvas = self.canvas
        # Visuals are reused (they keep the previous points until the new
        # ones are applied; no empty frame).
        canvas.reset_neuron_records()

        if not canvas.session_only:
            canvas.build_neuron_visuals_union()
        else:
            canvas.build_neuron_visuals_session(
                session_id=self.state.current_session_id,
            )

        canvas.plot_neurons_visuals()

        # Hover / selection / focus / highlight styles of the new points.
        super()._on_session_changed()

    def _on_session_style_changed(self, session_id):
        self.canvas.update_session_styles(session_id)

    def _on_selection_changed(self):
        # self.side_menu.update_display()
        super()._on_selection_changed()

    def _on_focus_changed(self):
        self.canvas._update_statistic_colors()
        super()._on_focus_changed()

    def update_neuron_selection(self):
        self.update_styles()

    def _on_review_filter_changed(self):
        self.canvas._update_statistic_colors()
        self.update_styles()

    def deactivate(self):

        if self._deactivated:
            return

        self._cancel_boundary_task()
        self.canvas.set_common_boundary(None)

        self.canvas.clean()
        super().deactivate()

    def _cancel_background_tasks(self):
        task_id = getattr(self, "_statistic_task_id", None)
        self._statistic_task_id = None

        if task_id is not None:
            self.state.tasks.cancel(task_id)


class OverviewControlPanel(ControlPanel.ControlPanel):

    # Union vs. session footprints, or the applied threshold changed.
    display_settings_changed = Signal()
    background_changed = Signal()
    background_load_requested = Signal()
    background_style_changed = Signal()
    border_changed = Signal()

    def __init__(
        self,
        parent,
        *,
        session_only=False,
        threshold=None,
        background_source="current",
        background_session=0,
        background_cmap="grays",
        background_cmap_reversed=False,
        border_tolerance=0,
    ):
        super().__init__(parent)

        review_selector = self._build_review_selector()
        self.form.addRow("Review status", review_selector)

        # Display background, independent of the alignment backgrounds.
        self.background_source = QComboBox()
        self.background_source.addItem("current session", "current")
        self.background_source.addItem("from session", "session")
        self.background_source.addItem("from file", "file")
        self.background_source.setToolTip(
            "Current session: follows the session selection. From session: "
            "fixed to one session's background. From file: an image loaded "
            "from another source, aligned to the sessions' reference frame "
            "(adjust it in Session alignment, target: overview background)."
        )

        self.background_session = QSpinBox()
        self.background_session.setMinimum(0)
        self.background_session.setValue(int(background_session))
        self.background_session.setToolTip("Session ID; numbering starts at 0.")

        self.background_load = QPushButton("Load…")
        self.background_file_label = QLabel()
        self.background_file_label.setStyleSheet("color: #9aa3ad;")

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.background_source)
        row.addWidget(self.background_session)
        row.addWidget(self.background_load)
        row.addWidget(self.background_file_label, 1)
        background_widget = QWidget()
        background_widget.setLayout(row)
        self.form.addRow("Background", background_widget)

        index = self.background_source.findData(background_source)
        self.background_source.setCurrentIndex(max(0, index))
        self._update_background_widgets()

        self.background_source.currentIndexChanged.connect(
            lambda _: (self._update_background_widgets(), self.background_changed.emit())
        )
        self.background_session.valueChanged.connect(
            lambda _: self.background_changed.emit()
        )
        self.background_load.clicked.connect(self.background_load_requested.emit)

        self.background_cmap = QComboBox()
        self.background_cmap.addItems(BACKGROUND_CMAPS)
        self.background_cmap.setCurrentIndex(
            max(0, self.background_cmap.findText(background_cmap))
        )
        self.background_cmap_reverse = QCheckBox("Reverse")
        self.background_cmap_reverse.setChecked(bool(background_cmap_reversed))

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.background_cmap)
        row.addWidget(self.background_cmap_reverse)
        row.addStretch()
        cmap_widget = QWidget()
        cmap_widget.setLayout(row)
        self.form.addRow("Background colormap", cmap_widget)

        self.background_cmap.currentTextChanged.connect(
            lambda _: self.background_style_changed.emit()
        )
        self.background_cmap_reverse.toggled.connect(
            lambda _: self.background_style_changed.emit()
        )

        # Shared imaging border (red overlay outside).
        self.border_tolerance = QSpinBox()
        self.border_tolerance.setRange(0, 999)
        self.border_tolerance.setValue(max(0, int(border_tolerance)))
        self.border_tolerance.setSuffix(" sessions")
        self.border_tolerance.setToolTip(
            "The red overlay marks pixels outside the area imaged in all "
            "sessions (after alignment). A pixel counts as inside if it is "
            "within the imaging window of all but this many sessions. Sessions "
            "without a current alignment are left out."
        )
        self.border_tolerance.valueChanged.connect(
            lambda _: self.border_changed.emit()
        )
        self.form.addRow("Border: tolerate outside", self.border_tolerance)

        self.session_only = QCheckBox("Show current session only")
        self.session_only.setToolTip(
            "Show the footprints detected in the current session (including "
            "unassigned ones) instead of the union footprints of all neurons."
        )
        self.session_only.setChecked(bool(session_only))
        self.session_only.toggled.connect(lambda _: self.display_settings_changed.emit())
        self.form.addRow(self.session_only)

        # Converting all footprints to points is costly: apply explicitly.
        self.applied_threshold = (
            DEFAULT_FOOTPRINT_THRESHOLD if threshold is None else float(threshold)
        )

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setRange(0.01, 0.99)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(self.applied_threshold)
        self.threshold_spin.setFixedWidth(72)
        self.threshold_spin.setToolTip(
            "Footprint pixels above this fraction of each footprint's peak "
            "are displayed."
        )

        self.threshold_apply = QPushButton("Apply")
        self.threshold_apply.setEnabled(False)
        self.threshold_apply.clicked.connect(self._apply_threshold)
        self.threshold_spin.valueChanged.connect(
            lambda value: self.threshold_apply.setEnabled(
                abs(value - self.applied_threshold) > 1e-9
            )
        )

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.threshold_spin)
        row.addWidget(self.threshold_apply)
        row.addStretch()
        threshold_widget = QWidget()
        threshold_widget.setLayout(row)
        self.form.addRow("Threshold", threshold_widget)

    @property
    def applied_border_tolerance(self):
        return int(self.border_tolerance.value())

    def _update_background_widgets(self):
        source = self.background_source.currentData()
        self.background_session.setVisible(source == "session")
        self.background_load.setVisible(source == "file")
        self.background_file_label.setVisible(source == "file")

    def _apply_threshold(self):
        self.applied_threshold = float(self.threshold_spin.value())
        self.threshold_apply.setEnabled(False)
        self.display_settings_changed.emit()


DEFAULT_FOOTPRINT_THRESHOLD = 0.1


def sparse_A_to_points(A_csc, dims, thr=0.2):
    """
    A_csc: scipy.sparse.csc_matrix, shape (d, n)
    dims: (H, W)
    Returns:
        pos: (Nnz, 2) float32, (x,y) pixel coords
        roi: (Nnz,) int32, neuron index for each point
        val: (Nnz,) float32, footprint weight
    """
    H, W = dims

    # max_vals = A_csc.max(axis=0).toarray()
    # print("###", max_vals.shape, A_csc.shape)

    # Normalize each column by its 99th percentile

    A = normalize_sparse_array(A_csc, relative_threshold=thr, format="coo")
    return _points_from_normalized(A.row, A.col, A.data, W)


def _points_from_normalized(pix, roi, val, W):
    pix = np.asarray(pix).astype(np.int64)
    roi = np.asarray(roi).astype(np.int32)
    val = np.asarray(val).astype(np.float32)
    if val.size:
        val /= np.percentile(val, 99) + 1e-8  # normalize to [0,1] for color mapping

    x = (pix // W).astype(np.float32)
    y = (pix % W).astype(np.float32)

    pos = np.column_stack([y, x]).astype(np.float32)  # (Nnz, 2)
    return pos, roi, val


class UnionPointCache:
    """``sparse_A_to_points`` for the union footprints, reusing the
    normalized points of unchanged columns: a manipulation changes only a
    few neurons, but normalizing all 2000+ columns takes ~80 ms.

    Columns are matched by content (entry count, first row index, sum and
    last value), so added / removed / reordered neurons are handled; the
    result equals ``sparse_A_to_points`` (same point order).
    """

    def __init__(self):
        self.setting = None
        self.columns = {}

    def points(self, A_csc, dims, thr):
        H, W = dims
        A = sparse.csc_matrix(A_csc)
        A.sum_duplicates()

        setting = (float(thr), int(H), int(W), A.shape[0])
        if setting != self.setting:
            self.setting = setting
            self.columns = {}

        n_columns = A.shape[1]
        counts = np.diff(A.indptr)
        filled = np.flatnonzero(counts)
        starts = A.indptr[:-1][filled]

        sums = np.zeros(n_columns)
        first = np.full(n_columns, -1, dtype=np.int64)
        last = np.zeros(n_columns)
        if len(filled):
            sums[filled] = np.add.reduceat(A.data, starts)
            first[filled] = A.indices[starts]
            last[filled] = A.data[A.indptr[filled + 1] - 1]

        keys = list(zip(counts.tolist(), first.tolist(), sums.tolist(), last.tolist()))

        missing = [j for j, key in enumerate(keys) if key not in self.columns]
        if missing:
            N = normalize_sparse_array(
                A[:, missing], relative_threshold=thr, format="csc"
            )
            N.sort_indices()
            for i, j in enumerate(missing):
                lo, hi = N.indptr[i], N.indptr[i + 1]
                self.columns[keys[j]] = (N.indices[lo:hi].copy(), N.data[lo:hi].copy())

        used = set(keys)
        if len(self.columns) > len(used):
            self.columns = {key: value for key, value in self.columns.items() if key in used}

        parts = [self.columns[key] for key in keys]
        lengths = np.fromiter((len(part[0]) for part in parts), dtype=np.int64, count=n_columns)

        if lengths.sum() == 0:
            empty = np.zeros(0)
            return _points_from_normalized(empty, empty, empty, W)

        pix = np.concatenate([part[0] for part in parts])
        val = np.concatenate([part[1] for part in parts])
        roi = np.repeat(np.arange(n_columns, dtype=np.int32), lengths)

        return _points_from_normalized(pix, roi, val, W)


def normalize_sparse_array(
    A, relative_threshold=0.001, minimum_nonzero_entries=10, format="csc"
):
    """
    normalizing sparse arrays with some thresholding
      - relative_threshold: float
          fraction of peak value, at below which entries are considered to be 0
      - minimum_nonzero_entries: int
          minimum number of nonzero entries of the footprint to be considered for further analyses

    Per column: entries above relative_threshold * peak, divided by the sum
    of entries above 0.001 * peak; columns with too few positive entries
    are empty. Vectorized over columns.
    """
    A = sparse.csc_matrix(A)
    A.sum_duplicates()
    n_columns = A.shape[1]

    counts = np.diff(A.indptr)
    column = np.repeat(np.arange(n_columns), counts)
    data = np.asarray(A.data)

    peak = np.full(n_columns, -np.inf)
    filled = counts > 0
    if filled.any():
        peak[filled] = np.maximum.reduceat(data, A.indptr[:-1][filled])
    # Implicit zeros take part in the column maximum.
    peak[counts < A.shape[0]] = np.maximum(peak[counts < A.shape[0]], 0)

    positive = np.bincount(column, weights=data > 0, minlength=n_columns)
    norm = np.bincount(
        column,
        weights=np.where(data > 0.001 * peak[column], data, 0),
        minlength=n_columns,
    )

    keep = (data > relative_threshold * peak[column]) & (
        positive[column] > minimum_nonzero_entries
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        values = data[keep] / norm[column[keep]]

    result = sparse.coo_matrix(
        (values, (A.indices[keep], column[keep])),
        shape=A.shape,
    )

    return result.asformat(format)


def format_colorbar_limits(
    lo: float,
    hi: float,
    significant_digits: int = 4,
) -> tuple[str, str]:

    max_abs = max(abs(lo), abs(hi))

    if max_abs == 0:
        return ("0", "0")

    order = int(np.floor(np.log10(max_abs)))

    if order >= 5 or order <= -4:
        return (
            f"{lo:.{significant_digits - 1}e}",
            f"{hi:.{significant_digits - 1}e}",
        )

    decimals = max(
        0,
        significant_digits - 1 - order,
    )

    return (
        f"{lo:.{decimals}f}",
        f"{hi:.{decimals}f}",
    )
