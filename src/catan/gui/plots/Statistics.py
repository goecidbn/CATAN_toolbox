from dataclasses import dataclass
from functools import partial
from unittest import result

import numpy as np
import numbers
from vispy import scene, color
from vispy.scene import visuals
from typing import Optional, List, Tuple, Callable
from catan.gui.interaction import click_events

from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtWidgets import (
    QLabel,
    QToolTip,
    QDoubleSpinBox,
)
from PySide6.QtGui import (
    QCursor,
)

import importlib
from catan.gui.data.statistics import (
    STATISTICS,
    StatisticEngine,
    plotdata_histogram,
)
from catan.gui.plots import StatisticsData
from catan.gui.plots.helper import HistogramMesh
from catan.gui.plots.helper import series_with_confidence
from catan.gui.plots.helper.cameras import (
    FixedPanZoomCamera,
)

from catan.gui.data import analysis

# , statistics
# from data import statistics as stats
from catan.gui.data.statistics import (
    plotdata_scatter,
)
from catan.gui.data.statistics.queries import StatisticQuery
from catan.gui.data.statistics import plotdata_series
from catan.gui.structures.state import NeuronComponent
from catan.gui.plots import BasePlot
from catan.gui.plots.helper import Threshold

# from GUI_elements.utils.menu_creation import get_colored_label, add_label_to_menu

importlib.reload(analysis)
# importlib.reload(stats)
importlib.reload(series_with_confidence)
# importlib.reload(plotdata_histogram)
importlib.reload(plotdata_series)
importlib.reload(StatisticsData)
importlib.reload(Threshold)


@dataclass
class RectangleData:

    center: Tuple[float, float]
    width: float
    height: float


class DisplaySignals(QObject):
    marker_hovered = Signal(object)
    marker_clicked = Signal(object, object)

    bin_hovered = Signal(object)
    bin_clicked = Signal(object, object)


class Display(BasePlot.BaseCanvas):

    pick_radius = 0.05  # for scatter

    def __init__(self, parent, controls, config=None):
        super().__init__(parent, controls, config)

        self.unfreeze()
        self.grid = self.central_widget.add_grid(spacing=0)
        self.view = self.grid.add_view(row=0, col=1)
        self.view.stretch = (1, 1)  # expand a lot
        self.view.camera = FixedPanZoomCamera(aspect=None)

        self.right_view = self.grid.add_view(row=0, col=1)
        self.right_view.camera = FixedPanZoomCamera(aspect=None)
        self.right_view.bgcolor = (0, 0, 0, 0)
        self.plot_root_right = scene.Node(parent=self.right_view.scene)

        self.right_view.visible = False

        self.initialize_axis()
        # self._ensure_twin_y_view()

        self.plot_root = scene.Node(parent=self.view.scene)

        self.signals = DisplaySignals()

        self.plot_type = None
        self.plot_data: Optional[
            plotdata_histogram.PlotData
            | plotdata_scatter.PlotData
            | plotdata_series.PlotData
        ] = None

        self.hist_base_layer = None
        self.hist_selected_layer = None

        self.build_overlays()

        self._on_threshold_changed: Optional[Callable] = None
        self.thresholds: dict[str, Threshold.ThresholdOverlay] = {}

        self.freeze()

    def initialize_axis(self):

        def create_axis(orientation="bottom"):
            if orientation in ["bottom", "top"]:
                tick_direction = (0, 1)
            elif orientation == "left":
                tick_direction = (-1, 0)
            elif orientation == "right":
                tick_direction = (1, 0)
            else:
                raise ValueError(f"Invalid orientation: {orientation}")

            axis = scene.AxisWidget(
                orientation=orientation,
                # axis_label="X value",
                tick_direction=tick_direction,
            )
            axis.axis.tick_color = "black"
            axis.axis.text_color = "black"
            axis.axis.axis_color = "black"
            axis.axis.axis_width = 2
            axis.axis.tick_width = 1

            if orientation in ["bottom", "top"]:
                axis.height_min = 20
                axis.height_max = 35
                axis.stretch = (1.0, 0.15)  # don't take extra vertical space
            elif orientation in ["left", "right"]:
                axis.width_min = 30
                axis.width_max = 50
                axis.stretch = (0.15, 1.0)  # don't take extra horizontal space
            axis.visible = False
            return axis

        self.axes = {}

        self.axes["x"] = create_axis("bottom")
        self.grid.add_widget(self.axes["x"], row=1, col=1)
        self.axes["x"].link_view(self.view)

        self.axes["y"] = create_axis("left")
        self.grid.add_widget(self.axes["y"], row=0, col=0)
        self.axes["y"].link_view(self.view)

        self.axes["y_right"] = create_axis("right")
        self.grid.add_widget(self.axes["y_right"], row=0, col=2)
        self.axes["y_right"].link_view(self.right_view)

        self.axes["y_right"].visible = False

    def build_overlays(self):

        self.clear_overlays()

        if isinstance(self.plot_data, plotdata_histogram.PlotData):
            visual = partial(
                visuals.Rectangle,
                center=(0, 0),
                width=1,
                height=1,
                parent=self.plot_root,
            )
            plot_options = partial(
                self.styles.get_plot_options,
                **{"plot_type": "bar", "values": 0.7, "border_color": "black"},
            )
        elif isinstance(self.plot_data, plotdata_scatter.PlotData):
            visual = partial(visuals.Markers, parent=self.plot_root)
            plot_options = partial(
                self.styles.get_plot_options,
                **{"plot_type": "marker", "values": 0.7},
            )
        else:
            return

        for key in self.plotting["overlays"]:

            self.plotting["overlays"][key] = visual(
                **plot_options(key),
            )
            self.plotting["overlays"][key].visible = False
            self.plotting["overlays"][key].set_gl_state(
                blend=True,
                depth_test=False,
                blend_func=("src_alpha", "one_minus_src_alpha"),
            )
        self.plotting["overlays"]["hovered"].order = 100
        self.plotting["overlays"]["focused"].order = 90
        self.plotting["overlays"]["selected"].order = 80

    def set_plot_data(self, plot_data):
        self.plot_data = plot_data

        if (
            self.plot_type == "histogram"
        ):  # and isinstance(plot_data, plotdata_histogram.PlotData):
            self._draw_histogram()
        elif (
            self.plot_type == "session_series"
        ):  # and isinstance(plot_data, plotdata_series.PlotData):
            self._draw_session_series()
        elif (
            self.plot_type == "scatter"
        ):  # and isinstance(plot_data, plotdata_scatter.PlotData):
            self._draw_scatter()
        else:
            return

        self.highlight_visuals_from_selection()

    def _draw_histogram(self):
        # print("drawing histogram")

        self.clear()
        self.build_overlays()

        if not isinstance(self.plot_data, plotdata_histogram.PlotData):
            raise ValueError(
                "plot_data must be an instance of HistogramPlotData for histogram plot."
            )

        self.axes["x"].visible = True
        self.axes["x"].axis.axis_label = self.plot_data.title["x"]

        bin_edges = np.asarray(self.plot_data.bin_edges, dtype=np.float32)
        bin_counts = np.asarray(self.plot_data.bin_counts, dtype=np.float32)
        n_bins = len(bin_counts)

        # build the rectangles for the histogram bars
        for k in range(n_bins):
            x0 = bin_edges[k]
            x1 = bin_edges[k + 1]
            h = bin_counts[k]

            self.plotting["data"][k] = RectangleData(
                center=(0.5 * (x0 + x1), 0.5 * h),
                width=x1 - x0,
                height=h,
            )

        default_color = self.styles.get_color_array("default")
        selected_overlay_color = self.styles.get_color_array("selected", alpha=0.65)

        # print(f"{n_bins=}")
        base_colors = np.tile(default_color, (n_bins, 1))
        selected_overlay_color = np.tile(selected_overlay_color, (n_bins, 1))

        # Optional: make zero-height bars fully transparent.
        zero_mask = bin_counts <= 0
        base_colors[zero_mask, 3] = 0.0

        self.hist_base_layer = HistogramMesh.HistogramMeshLayer(
            parent=self.plot_root,
            order=0.0,
            edge_order=1.0,
            edge_color=(0, 0, 0, 0.8),
            edge_width=1.0,
            draw_edges=True,
        )
        self.hist_base_layer.build_from_counts(
            bin_edges=bin_edges,
            counts=bin_counts,
            bin_colors=base_colors,
        )

        self.hist_selected_layer = HistogramMesh.HistogramMeshLayer(
            parent=self.plot_root,
            order=110.0,
            draw_edges=False,
        )
        self.hist_selected_layer.build_from_counts(
            bin_edges=bin_edges,
            counts=np.zeros_like(bin_counts),
            bin_colors=selected_overlay_color,
        )
        self.hist_selected_layer.set_visible(False)

        self.plotting["visuals"]["base"] = self.hist_base_layer.mesh
        self.plotting["visuals"]["selected"] = self.hist_selected_layer.mesh

        xmin, xmax = bin_edges[[0, -1]]
        ymax = max(1.0, float(np.nanmax(bin_counts)))

        self.view.camera.set_exact_range(
            x=(min(0.0, xmin) * 1.1, xmax * 1.1),
            y=(0.0, ymax * 1.1),
        )

        self.axes["x"]._view_changed()

        self._init_threshold_overlays(kind="histogram")
        self.update()

    def _draw_session_series(self):

        self.clear()
        self.build_overlays()
        # print("drawing session series")
        if not isinstance(self.plot_data, plotdata_series.PlotData):
            raise ValueError(
                "plot_data must be an instance of SessionSeriesPlotData for session series plot."
            )

        self.axes["x"].visible = True
        self.axes["x"].axis.axis_label = "Session index"

        first = self.plot_data.first_series
        second = self.plot_data.second_series

        is_dual = second is not None

        if is_dual:
            # self._ensure_twin_y_view()
            self.right_view.visible = True
            self.axes["y_right"].visible = True
        else:
            # if getattr(self, "right_view", None) is not None:
            self.right_view.visible = False
            # if "y_right" in self.axes:
            self.axes["y_right"].visible = False

        self.axes["y"].visible = True
        self.axes["y"].axis.axis_label = first.table.stat.name

        first_visual = series_with_confidence.SeriesVisual(
            parent=self.plot_root,
            line_color=self.styles.get_color_array("selected"),
            band_color=self.styles.get_color_array("default"),
            marker_size=7,
            order=0,
        )

        self.plotting["visuals"]["base"] = first_visual.line
        self.plotting["visuals"]["band"] = first_visual.band
        self.plotting["visuals"]["markers"] = first_visual.markers

        first_visual.set_data(
            session_ids=first.session_ids,
            values=first.values,
            errors_low=first.errors_low,
            errors_high=first.errors_high,
        )

        # ---------- optional right series ----------
        if is_dual:
            self.axes["y_right"].axis.axis_label = second.table.stat.name

            second_visual = series_with_confidence.SeriesVisual(
                parent=self.plot_root_right,
                line_color=self.styles.get_color_array("highlighted"),
                band_color=self.styles.get_color_array("default"),
                marker_size=7,
                order=0,
            )

            self.plotting["visuals"]["base_second"] = second_visual.line
            self.plotting["visuals"]["band_second"] = second_visual.band
            self.plotting["visuals"]["markers_second"] = second_visual.markers

            second_visual.set_data(
                session_ids=second.session_ids,
                values=second.values,
                errors_low=second.errors_low,
                errors_high=second.errors_high,
            )
        # x_min, x_max = np.inf, -np.inf
        # y_min, y_max = np.inf, -np.inf
        # for key, suffix in zip(["first_series", "second_series"], ["", "_2nd"]):

        #     data = getattr(self.plot_data, key)
        #     if data is None:
        #         continue
        #     # print(data)
        #     print(
        #         f"Drawing session series for {key}: session_ids={data.session_ids}, values={data.values}, errors_low={data.errors_low}, errors_high={data.errors_high}"
        #     )
        #     visual = series_with_confidence.SeriesVisual(
        #         parent=self.plot_root,
        #         line_color=self.styles.get_color_array("selected"),
        #         band_color=self.styles.get_color_array("default"),
        #         marker_size=7,
        #         order=0,
        #     )
        #     visual.set_data(
        #         session_ids=data.session_ids,
        #         values=data.values,
        #         errors_low=data.errors_low,
        #         errors_high=data.errors_high,
        #     )
        #     self.plotting["visuals"]["base" + suffix] = visual.line
        #     self.plotting["visuals"]["band" + suffix] = visual.band
        #     self.plotting["visuals"]["markers" + suffix] = visual.markers

        #     finite = np.isfinite(data.session_ids) & np.isfinite(data.values)

        #     _x_min, _x_max = data.session_ids[[0, -1]]
        #     _y_min, _y_max = np.min(data.values[finite]), np.max(data.values[finite])

        #     x_min = min(x_min, _x_min)
        #     x_max = max(x_max, _x_max)
        #     y_min = min(y_min, _y_min)
        #     y_max = max(y_max, _y_max)

        # ---------- shared x range ----------
        if is_dual:
            all_sessions = np.union1d(first.session_ids, second.session_ids)
        else:
            all_sessions = np.asarray(first.session_ids)

        x_min = float(np.nanmin(all_sessions))
        x_max = float(np.nanmax(all_sessions))

        x_range = (x_min - 0.5, x_max + 0.5)

        # ---------- independent y ranges ----------
        left_y_range = _series_y_range(first)

        self.view.camera.set_exact_range(
            x=x_range,
            y=left_y_range,
        )

        if is_dual:
            right_y_range = _series_y_range(second)

            self.right_view.camera.set_exact_range(
                x=x_range,
                y=right_y_range,
            )

        # ---------- update axes ----------
        self.axes["x"]._view_changed()
        self.axes["y"]._view_changed()

        if is_dual:
            self.axes["y_right"]._view_changed()

        # self.view.camera.set_exact_range(
        #     x=(x_min - 0.5, x_max + 0.5),
        #     y=(min(0.0, y_min) * 0.9, y_max * 1.1),
        # )

        # self.axes["x"]._view_changed()
        # self.axes["y"]._view_changed()
        self.update()

    def _draw_scatter(self):

        self.clear()
        self.build_overlays()

        if not isinstance(self.plot_data, plotdata_scatter.PlotData):
            raise ValueError(
                "plot_data must be an instance of ScatterPlotData for scatter plot."
            )

        if "error" in self.plot_data.title:
            self.axes["x"].visible = False
            self.axes["y"].visible = False

            # Display error text in center of canvas
            error_text = self.plot_data.title.get("error", "Error")
            text_visual = visuals.Text(
                text=error_text,
                pos=[0, 0],
                anchor_x="center",
                anchor_y="center",
                font_size=12,
                color="red",
                parent=self.plot_root,
            )
            self.plotting["visuals"]["error_text"] = text_visual

            self.view.camera.set_range(x=(-1, 1), y=(-1, 1), margin=0.0)
            self.update()
            return
        # data = self.plot_data
        # x_data = self.plot_data.x
        # y_data = self.plot_data.y

        # print(x_data.shape, y_data.shape)

        self.axes["x"].visible = True
        self.axes["x"].axis.axis_label = self.plot_data.title["x"]

        self.axes["y"].visible = True
        self.axes["y"].axis.axis_label = self.plot_data.title["y"]

        style = "default" if self.state.selected_components is None else "background"
        plot_options = self.styles.get_plot_options(style, "marker", 0.7, size=5.0)

        self.plotting["visuals"]["base"] = visuals.Markers(
            pos=np.column_stack((self.plot_data.x, self.plot_data.y)),
            **plot_options,
            parent=self.plot_root,
        )
        self.plotting["visuals"]["base"].set_gl_state(
            depth_test=False,
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
        )

        self.view.camera.set_range(
            x=(
                min(0, np.nanmin(self.plot_data.x)) * 1.1,
                np.nanmax(self.plot_data.x) * 1.1,
            ),
            y=(
                min(0, np.nanmin(self.plot_data.y)) * 1.1,
                np.nanmax(self.plot_data.y) * 1.1,
            ),
            margin=0.0,
        )

        self.axes["x"]._view_changed()
        self.axes["y"]._view_changed()

        self._init_threshold_overlays(kind="scatter")

        self.update()

    def _init_threshold_overlays(self, kind: str):
        self._destroy_threshold_overlays()

        if self.plot_data is None:
            return

        self.thresholds = {}
        self.thresholds["x"] = Threshold.ThresholdOverlay(
            canvas=self,
            axis="x",
            on_changed=self._on_threshold_changed,
        )
        self.thresholds["x"].reset_to_view_center(emit=False)
        self.thresholds["x"].set_visible(True)
        self.thresholds["x"].set_tooltip_parameter(self.plot_data.title["x"])

        if kind == "scatter":
            self.thresholds["y"] = Threshold.ThresholdOverlay(
                canvas=self,
                axis="y",
                on_changed=self._on_threshold_changed,
            )
            self.thresholds["y"].reset_to_view_center(emit=False)
            self.thresholds["y"].set_visible(True)
            self.thresholds["y"].set_tooltip_parameter(self.plot_data.title["y"])

    def update_threshold_visuals(self):
        for threshold in self.thresholds.values():
            if threshold is not None:
                threshold.update_visuals()

    ### ======================================================= ###
    ### ---------------- INTERACTION FUNCTIONS ---------------- ###
    ### ------------------------------------------------------- ###
    ### ------------------------ INPUT ------------------------ ###
    ### ======================================================= ###

    def on_mouse_move(self, event):

        if (
            event.pos is None
            or not self.plotting["visuals"]
            or "error_text" in self.plotting["visuals"]
            or self.plot_data is None
        ):
            return

        # 1. Threshold dragging has priority
        threshold_consumed = False
        for threshold in self.thresholds.values():
            threshold_consumed |= threshold.handle_mouse_move(event)

        if threshold_consumed:
            return

        # 2. Hover thresholds, but only one owns the tooltip
        hovered_overlay = None
        hovered_distance = np.inf

        for threshold in self.thresholds.values():

            if threshold.update_hover(event.pos):
                dist = threshold.line_distance_px(event.pos)
                if dist < hovered_distance:
                    hovered_overlay = threshold
                    hovered_distance = dist

        if hovered_overlay is not None:
            hovered_overlay.show_tooltip()
            event.handled = True
            return

        # for threshold in self.thresholds.values():
        #     if threshold.update_hover(event.pos):
        #         # event.handled = True
        #         return

        idx = self.find_closest_visual(event.pos, requires_transform=True)
        self.update_style(idx, "hovered")
        self.handle_tooltip(idx)

        self.update()

        if isinstance(self.plot_data, plotdata_histogram.PlotData):
            self.signals.bin_hovered.emit(idx)
        elif isinstance(self.plot_data, plotdata_scatter.PlotData):
            self.signals.marker_hovered.emit(idx)

    def on_mouse_release(self, event):

        if (
            event.button != 1
            or event.pos is None
            or not self.plotting["visuals"]
            or "error_text" in self.plotting["visuals"]
            or self.plot_data is None
        ):
            return

        for threshold in self.thresholds.values():
            if threshold.handle_mouse_release(event):
                return

        idx = self.find_closest_visual(event.pos, requires_transform=True)

        if isinstance(self.plot_data, plotdata_histogram.PlotData):
            self.signals.bin_clicked.emit(idx, event.modifiers)
        elif isinstance(self.plot_data, plotdata_scatter.PlotData):
            self.signals.marker_clicked.emit(idx, [])

    def on_mouse_press(self, event):

        if (
            event.button != 1
            or event.pos is None
            or not self.plotting["visuals"]
            or "error_text" in self.plotting["visuals"]
            or self.plot_data is None
        ):
            return

        threshold_consumed = False

        for threshold in self.thresholds.values():
            if threshold.handle_mouse_press(event):
                threshold_consumed = True
                return

        if threshold_consumed:
            return

        # Click was away from all threshold lines.
        for threshold in self.thresholds.values():
            threshold.deactivate_visuals()

        super().on_mouse_press(event)

    def find_closest_visual(self, data_pos, requires_transform=False) -> Optional[int]:

        if self.plot_data is None:
            return None

        if requires_transform:
            data_pos = click_events.canvas_to_visual(
                self.plotting["visuals"]["base"], data_pos
            )

        if isinstance(self.plot_data, plotdata_histogram.PlotData):
            return self.find_bin_from_data(data_pos)
        elif isinstance(self.plot_data, plotdata_scatter.PlotData):
            return self.find_marker_from_data(data_pos)

    def find_bin_from_data(self, data_pos) -> Optional[int]:
        """
        Given data coords x_data, y_data, return bin index (int) or None.
        We require x within bin range and y >= 0 and y <= bar height.
        """
        assert isinstance(self.plot_data, plotdata_histogram.PlotData)

        nbins = int(self.controls["bin_selector"].value())
        if self.plot_data is None or nbins == 0:
            return None

        # Find bin where x is inside [edge_k, edge_k+1)
        k = np.searchsorted(self.plot_data.bin_edges, data_pos[0], side="right") - 1

        # if k < 0 or k >= nbins:
        if not (k in self.plotting["data"]):
            return None

        # only pick bin, if mouse is on bin
        h = self.plot_data.bin_counts[k]
        if h != 0 and (data_pos[1] < 0 or data_pos[1] > h):
            return None

        return int(k)

    def find_marker_from_data(self, data_pos) -> Optional[int]:
        """
        Given data coords x_data, y_data, return marker index (int) or None.
        We require (x,y) to be within pick_radius around marker.
        """
        assert isinstance(self.plot_data, plotdata_scatter.PlotData)
        if self.plot_data is None:
            return None

        cam_bounds = self.view.camera.rect
        dx = (self.plot_data.x - data_pos[0]) / (cam_bounds.right - cam_bounds.left)
        dy = (self.plot_data.y - data_pos[1]) / (cam_bounds.top - cam_bounds.bottom)
        d2 = dx * dx + dy * dy

        idx = int(np.nanargmin(d2))

        dist_px = float(np.sqrt(d2[idx]))

        if dist_px > self.pick_radius:
            return None

        return idx

    ### ======================================================= ###
    ### ---------------- INTERACTION FUNCTIONS ---------------- ###
    ### ------------------------------------------------------- ###
    ### ------------------------ OUTPUT ----------------------- ###
    ### ======================================================= ###

    def highlight_visuals_from_selection(self):
        """
        Highlights the visuals corresponding to the currently selected components.
        """
        self.state.timeit()
        if isinstance(self.plot_data, plotdata_histogram.PlotData):
            self.highlight_bins_from_selection()
        elif isinstance(self.plot_data, plotdata_scatter.PlotData):
            self.highlight_markers_from_selection()
        else:
            return

        # self.state.timeit("highlight_visuals_from_selection")
        self.update()
        self.state.timeit("updated canvas after highlight_visuals_from_selection")

    def highlight_bins_from_selection(self):
        plot_data = self.plot_data

        if plot_data is None:
            return

        if not isinstance(plot_data, plotdata_histogram.PlotData):
            return

        if self.hist_base_layer is None or self.hist_selected_layer is None:
            return

        self.state.timeit()

        if self.state.selected_components is None:
            selected_counts = np.zeros_like(plot_data.bin_counts, dtype=int)
            selected_bins = np.asarray([], dtype=int)
        else:
            selected_counts, _, selected_bin_ids = (
                plot_data.selected_histogram_for_components(
                    self.state.selected_components
                )
            )

            selected_bins = np.unique(selected_bin_ids)

        self.state.timeit("calculated selected bins")

        base_colors = self._histogram_base_colors_for_selection(selected_bins)

        self.hist_base_layer.set_bin_colors(base_colors)

        self.hist_selected_layer.set_counts(selected_counts)
        self.hist_selected_layer.set_visible(np.any(selected_counts > 0))

        self.state.timeit("updated histogram mesh selection")

    def _histogram_base_colors_for_selection(self, selected_bins):
        """
        helper function to define colors for histogram
        """
        assert isinstance(self.plot_data, plotdata_histogram.PlotData)

        n_bins = len(self.plot_data.bin_counts)
        bin_counts = np.asarray(self.plot_data.bin_counts)

        selected_bins = np.asarray(selected_bins, dtype=int)

        highlighted = np.zeros(n_bins, dtype=bool)

        if selected_bins.size > 0:
            highlighted[selected_bins] = True

        has_selection = highlighted.any()
        base_style = "background" if has_selection else "default"

        selected_color = self.styles.get_color_array("selected", alpha=0.4)
        base_color = self.styles.get_color_array(base_style)

        colors = np.tile(base_color, (n_bins, 1))
        colors[highlighted] = selected_color

        # Optional: keep empty bins invisible.
        colors[bin_counts <= 0, 3] = 0.0

        return colors

    def highlight_markers_from_selection(self):

        if not isinstance(self.plot_data, plotdata_scatter.PlotData):
            return

        if self.state.selected_components is None:
            self.update_style(None, "selected")
            return

        markers = self.plot_data.markers_matching_components(
            self.state.selected_components
        )
        self.update_style(markers, "selected")

    def update_style(self, idx: Optional[int | np.ndarray] = None, style="default"):

        if isinstance(self.plot_data, plotdata_histogram.PlotData):
            if not isinstance(idx, (numbers.Integral, type(None))):
                raise ValueError(
                    f"Expected idx to be an int or None for HistogramPlotData, but got {type(idx)}"
                )
            self.update_bin_style(idx, style)
        elif isinstance(self.plot_data, plotdata_scatter.PlotData):
            self.update_marker_style(idx, style)
        else:
            return

    def update_bin_style(self, bin: Optional[int], style: str = "default"):
        """Apply base/hover/selected colors to rectangle visuals."""

        assert isinstance(self.plot_data, plotdata_histogram.PlotData)
        if style == "hovered":
            rect = self.plotting["overlays"][style]
            if rect is None:
                return

            if (
                bin is None
                or self.plotting["data"].get(bin, None) is None
                or self.plot_data.bin_counts[bin] <= 0
            ):
                rect.visible = False
                return
            data = self.plotting["data"][bin]

            rect.visible = True
            rect.center = data.center
            rect.width = data.width
            rect.height = data.height
            return

    def update_marker_style(
        self, marker: Optional[int | np.ndarray], style: str = "default", update=True
    ):
        """Show highlight marker on points idx (or hide if idx is None)."""

        assert isinstance(self.plot_data, plotdata_scatter.PlotData)

        # print(f"updating style '{style}' for marker: {marker}")

        self.plotting["overlays"][style].visible = False
        if marker is None:
            self.plotting["overlays"][style].set_data(
                np.zeros((0, 2), dtype=np.float32)
            )
        else:
            plot_options = self.styles.get_plot_options(
                style, "marker", 0.7, size=8.0, edge_width=0.0  # , edge_color=None
            )

            pos = self.plot_data.pos_for_marker(marker)
            self.plotting["overlays"][style].visible = True

            data = np.column_stack(tuple(pos))
            self.plotting["overlays"][style].set_data(
                data,
                **plot_options,
            )

    def handle_tooltip(self, idx: Optional[int] = None):

        if idx is None:
            QToolTip.hideText()
            return

        if isinstance(self.plot_data, plotdata_histogram.PlotData):
            tooltip_text = self.plot_data.tooltip_for_bin(idx)
        elif isinstance(self.plot_data, plotdata_scatter.PlotData):
            tooltip_text = self.plot_data.tooltip_for_marker(idx)
        else:
            tooltip_text = "Unknown selection"

        QToolTip.showText(QCursor.pos(), tooltip_text)

    # def handle_tooltip(self, component: Optional[NeuronComponent] = None):

    #     if isinstance(self.plot_data, plotdata_histogram.PlotData):
    #         super().handle_tooltip(component)
    #         return

    #     point = None if component is None else int(component)
    #     # if not isinstance(component, list):
    #     #     return
    #     if point is None:
    #         QToolTip.hideText()
    #         return

    #     x_key = self.plot_data.title["x"]
    #     y_key = self.plot_data.title["y"]

    #     if len(self.stat_data.shape["x"]) == 1:

    #         fp_id = point
    #         component = self.state.get_component_from_footprint(fp_id)

    #         x = self.plot_data.x[point]
    #         y = self.plot_data.y[point]

    #         tooltip_text = f"Neuron ID: {'unassigned' if component is None else component.neuron_id}\nFootprint ID: {fp_id}"
    #         tooltip_text += f"\n{x_key}: {x:.3f}\n{y_key}: {y:.3f}"

    #     elif len(self.stat_data.shape["x"]) == 2:
    #         fp_ids = np.unravel_index(point, self.stat_data.shape["x"])
    #         x = self.plot_data.x[point]
    #         y = self.plot_data.y[point]
    #         # idxes = np.unravel_index(component, self.stat_data.shape["x"])
    #         components = [
    #             self.state.get_component_from_footprint(fp_id) for fp_id in fp_ids
    #         ]
    #         neuron_ids = [
    #             "unassigned" if c is None else c.neuron_id for c in components
    #         ]
    #         tooltip_text = f"Neuron IDs: {','.join(map(str, neuron_ids))}\nFootprint IDs: {','.join(map(str, fp_ids))}"
    #         tooltip_text += f"\n{x_key}: {x:.3f}\n{y_key}: {y:.3f}"
    #     else:
    #         tooltip_text = "Unknown selection"

    #     QToolTip.showText(QCursor.pos(), tooltip_text)

    def update_selection(self):
        pass

    def clear(self):
        self._destroy_threshold_overlays()
        self._clear_histogram_mesh_layers()

        for _, visual in self.plotting["visuals"].items():
            self._destroy_visual(visual)

        self.plotting["visuals"] = {}
        self.plotting["data"] = {}

        self.clear_overlays()

    def _clear_histogram_mesh_layers(self):
        for attr in ("hist_base_layer", "hist_selected_layer"):
            layer = getattr(self, attr, None)

            if layer is not None:
                layer.destroy()
                setattr(self, attr, None)

    def clear_overlays(self):
        for _, overlay in self.plotting["overlays"].items():
            if overlay is None:
                continue
            overlay.visible = False

    def _destroy_threshold_overlays(self):

        for key, threshold in self.thresholds.items():
            if threshold is not None:
                threshold.destroy()
        self.thresholds = {}

    def _destroy_visual(self, visual):
        if visual is None:
            return
        if isinstance(visual, series_with_confidence.SeriesVisual):
            visual.destroy()
        else:
            visual.visible = False
            visual.parent = None


class Controller(BasePlot.CanvasController):

    canvas: Display

    def build_controls(self):
        # print("Building controls for PlotController (statistics display)")
        self.engine = StatisticEngine(
            registry=STATISTICS,
            data=self.data,
            state=self.state,
        )

        self.current_x_query = None
        self.current_y_query = None
        self.current_y_query_2nd = None

        self.current_plot_data = None

        bin_selector = QDoubleSpinBox()
        initial_nbin = 30
        bin_selector.setDecimals(0)
        bin_selector.setRange(2.0, 100.0)
        bin_selector.setSingleStep(1)
        bin_selector.setValue(initial_nbin)
        self.controls["bin_selector"] = bin_selector
        self.controls["bin_label"] = QLabel("Histogram bins:")

        self.section.x_options_layout.addWidget(self.controls["bin_label"])
        self.section.x_options_layout.addWidget(self.controls["bin_selector"])

        self.controls["x_selector"] = StatisticsData.StatisticQuerySelector(
            registry=STATISTICS, data=self.data, state=self.state, axis="x"
        )
        self.section.x_options_layout.addWidget(self.controls["x_selector"])

        self.controls["x_selector"].queryChanged.connect(
            lambda query: self._on_query_changed("x", query)
        )

        self.controls["y_selector"] = StatisticsData.StatisticQuerySelector(
            registry=STATISTICS, data=self.data, state=self.state, axis="y"
        )
        self.section.y_options_layout.addWidget(self.controls["y_selector"])

        self.controls["y_selector"].queryChanged.connect(
            lambda query: self._on_query_changed("y", query)
        )

        self.controls["y_selector_2nd"] = StatisticsData.StatisticQuerySelector(
            registry=STATISTICS, data=self.data, state=self.state, axis="y"
        )
        self.section.y_options_layout.addWidget(self.controls["y_selector_2nd"])

        self.controls["y_selector_2nd"].queryChanged.connect(
            lambda query: self._on_query_changed("y_2nd", query)
        )

        self.section.y_options_layout.addStretch()

        self.controls["bin_selector"].valueChanged.connect(self._on_plot_params_changed)
        self.canvas.signals.marker_clicked.connect(self._on_visual_clicked)
        self.canvas.signals.bin_clicked.connect(self._on_visual_clicked)

        self.canvas._on_threshold_changed = self._on_threshold_changed

    def _on_plot_params_changed(self):
        self.rebuild_plot_data()
        self.update_canvas()

    def _on_query_changed(self, which, query: StatisticsData.StatisticQuery):

        if which == "x":
            self.current_x_query = query
        elif which == "y":
            self.current_y_query = query
        elif which == "y_2nd":
            ## should only be possible if plot_type is session_series
            self.current_y_query_2nd = query

        self.identify_plot_type()

        for key in ["bin_label", "bin_selector", "y_selector_2nd"]:
            self.controls[key].setVisible(False)

        # print(f"Plot type identified as: {self.plot_type}")

        if self.plot_type == "histogram":
            self.controls["bin_selector"].setVisible(self.current_x_query is not None)
            self.controls["bin_label"].setVisible(self.current_x_query is not None)

            self.controls["y_selector"].set_query_mode("generic")
            self.controls["y_selector"].set_query_preparer(None)

            self.controls["x_selector"].set_query_mode("generic")
            self.controls["x_selector"].set_query_preparer(None)
        elif self.plot_type == "session_series":

            self.controls["y_selector_2nd"].setVisible(self.current_y_query is not None)

            # session-series mode
            self.controls["y_selector"].set_query_mode("session_series")
            self.controls["y_selector"].set_query_preparer(
                plotdata_series.prepare_query
            )

            self.controls["y_selector_2nd"].set_query_mode("session_series")
            self.controls["y_selector_2nd"].set_query_preparer(
                plotdata_series.prepare_query
            )

            if query is None:
                if which == "y":
                    self.current_y_query_2nd = None
                    self.controls["y_selector_2nd"].set_query(None)
                return

            prepared_query = plotdata_series.prepare_query(query, self.engine.registry)
            if which == "y":
                self.current_y_query = prepared_query
            elif which == "y_2nd":
                self.current_y_query_2nd = prepared_query

        elif self.plot_type == "scatter":
            self.controls["y_selector"].set_query_mode("generic")
            self.controls["y_selector"].set_query_preparer(None)

            self.controls["x_selector"].set_query_mode("generic")
            self.controls["x_selector"].set_query_preparer(None)

        self.canvas.clear()
        self.rebuild_plot_data()
        self.update_canvas()

    def identify_plot_type(self):
        x_query = self.current_x_query
        y_query = self.current_y_query

        if y_query is not None and y_query.statistic_key != "none" and x_query is None:
            plot_type = "session_series"
            ## disable reduction selector
        elif (
            x_query is not None and x_query.statistic_key != "none" and y_query is None
        ):
            plot_type = "histogram"
        elif x_query is not None and y_query is not None:
            plot_type = "scatter"
        else:
            plot_type = None

        self.plot_type = self.canvas.plot_type = plot_type

    def rebuild_plot_data(self):

        # self.current_plot_data = None

        x_table = self.engine.evaluate_table(self.current_x_query)
        y_table = self.engine.evaluate_table(self.current_y_query)

        if x_table is None and y_table is None:
            return
        if self.plot_type == "histogram" and x_table is not None:
            nbins = int(self.controls["bin_selector"].value())
            self.current_plot_data = plotdata_histogram.build_plot_data(
                x_table,
                bins=nbins,
            )
        elif self.plot_type == "session_series" and y_table is not None:

            y_table_2nd = self.engine.evaluate_table(self.current_y_query_2nd)
            self.current_plot_data = plotdata_series.build_plot_data(
                first_table=y_table,
                second_table=y_table_2nd,
            )
            # print(f"Built session series plot data: {self.current_plot_data}")
        elif (
            self.plot_type == "scatter" and x_table is not None and y_table is not None
        ):
            self.current_plot_data = plotdata_scatter.build_plot_data(
                x_table=x_table,
                y_table=y_table,
            )
        else:
            raise ValueError(
                f"Invalid combination of x_query and y_query for plot_type '{self.plot_type}': x_query={self.current_x_query}, y_query={self.current_y_query}"
            )

    def _on_selection_changed(self):
        if self.current_plot_data is None:
            return

        self.canvas.highlight_visuals_from_selection()

    def update_canvas(self):
        self.canvas.set_plot_data(self.current_plot_data)

    def _on_visual_clicked(self, idx: Optional[int], modifiers):

        if idx is None:
            self._handle_picked_refs(None, modifiers)
            return

        if isinstance(self.current_plot_data, plotdata_histogram.PlotData):
            rows = self.current_plot_data.rows_for_bin(idx)
        elif isinstance(self.current_plot_data, plotdata_scatter.PlotData):
            rows = self.current_plot_data.rows_for_markers(idx)
        else:
            return

        refs = self.current_plot_data.table.refs_for_rows(rows)

        self._handle_picked_refs(refs, modifiers)

    # def _on_bin_hovered(self, bin_index):
    #     # self.handle_tooltip_for_bin(bin_index)
    #     pass

    # def _on_marker_hovered(self, marker_index):
    #     pass

    def _on_threshold_changed(self, spec: Threshold.ThresholdSpec):
        self._select_from_threshold()

    def _select_from_threshold(self):
        plot_data = self.current_plot_data

        if plot_data is None:
            return

        thresholds = self.canvas.thresholds

        if self.plot_type == "histogram" and isinstance(
            plot_data, plotdata_histogram.PlotData
        ):
            rows = plot_data.rows_for_threshold(thresholds["x"].spec)
            refs = plot_data.table.refs_for_rows(rows)

            self._handle_picked_refs(refs, [])
            return

        if self.plot_type == "session_series" and isinstance(
            plot_data, plotdata_series.PlotData
        ):
            pass

        if self.plot_type == "scatter" and isinstance(
            plot_data, plotdata_scatter.PlotData
        ):
            markers = plot_data.markers_for_thresholds(
                x_spec=thresholds["x"].spec,
                y_spec=thresholds["y"].spec,
            )

            rows = plot_data.rows_for_markers(markers)
            refs = plot_data.table.refs_for_rows(rows)

            self._handle_picked_refs(refs, [])
            return

    def _handle_picked_refs(self, refs: dict[str, np.ndarray], modifiers):

        if refs is None:
            self.state.update_selected_components(None, modifiers)
            return

        components = []
        # print(f"Picked refs: {refs}")

        # one value per neuron
        if "neuron" in refs:
            for i, neuron_id in enumerate(refs["neuron"]):
                components.append(
                    NeuronComponent(
                        session_id=(
                            refs["session"][i]
                            if "session" in refs
                            else self.state.current_session_id
                        ),
                        neuron_id=int(neuron_id),
                    )
                )

        # one value per neuron pair
        if "neuron_i" in refs or "neuron_j" in refs:
            neuron_ids = []

            if "neuron_i" in refs:
                neuron_ids.append(refs["neuron_i"])

            if "neuron_j" in refs:
                neuron_ids.append(refs["neuron_j"])

            neuron_ids = np.unique(np.concatenate(neuron_ids))
            # print(f"Picked neuron IDs from pairs: {neuron_ids}")

            for i, neuron_id in enumerate(neuron_ids):
                components.append(
                    NeuronComponent(
                        session_id=(
                            refs["session"][i]
                            if "session" in refs
                            else self.state.current_session_id
                        ),
                        neuron_id=int(neuron_id),
                    )
                )

        if components:
            # set() removes duplicates, list() keeps API simple
            components = list({c for c in components if c is not None})
            self.state.update_selected_components(components, modifiers)
            return

        # # optional later: session-based selections
        # if "session" in refs:
        #     self.state.update_selected_sessions(np.unique(refs["session"]), modifiers)
        #     return

        # if "session_i" in refs or "session_j" in refs:
        #     session_ids = []

        #     if "session_i" in refs:
        #         session_ids.append(refs["session_i"])

        #     if "session_j" in refs:
        #         session_ids.append(refs["session_j"])

        #     self.state.update_selected_sessions(
        #         np.unique(np.concatenate(session_ids)), modifiers
        #     )

        # if self.changes_on_click == "selected":
        #     self.state.update_selected_components(component, event.modifiers)
        # elif self.changes_on_click == "focused":
        #     self.state.focused_component = component
        # elif self.changes_on_click == "highlighted":
        #     self.state.highlighted_component = component
        # else:
        #     raise ValueError(f"Invalid changes_on_click value: {self.changes_on_click}")

    def deactivate(self):
        for key in self.controls:
            self.controls[key].requires_update.disconnect()
            self.controls[key].changed_statistic.disconnect()

        super().deactivate()

    def _on_session_changed(self):
        self.rebuild_plot_data()
        super()._on_session_changed()

    def update_neuron_selection(self):
        self.canvas.highlight_visuals_from_selection()

    def update_styles(self):
        pass


def _series_y_range(series: plotdata_series.SessionSeries, *, include_zero=True):
    values = np.asarray(series.values, dtype=float)

    finite = np.isfinite(values)

    if series.has_errors:
        lower = values - np.asarray(series.errors_low, dtype=float)
        upper = values + np.asarray(series.errors_high, dtype=float)

        finite &= np.isfinite(lower) & np.isfinite(upper)
    else:
        lower = values
        upper = values

    if not np.any(finite):
        return 0.0, 1.0

    ymin = float(np.nanmin(lower[finite]))
    ymax = float(np.nanmax(upper[finite]))

    if include_zero:
        ymin = min(0.0, ymin)
        ymax = max(0.0, ymax)

    if ymin == ymax:
        pad = max(1.0, abs(ymin) * 0.1)
    else:
        pad = 0.1 * (ymax - ymin)

    return ymin - pad, ymax + pad
