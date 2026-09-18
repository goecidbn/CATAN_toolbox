from dataclasses import dataclass
from functools import partial
import traceback

import numpy as np
import numbers
from vispy import scene
from vispy.scene import visuals
from vispy.scene.visuals import Rectangle, Text, Markers
from typing import Optional, Tuple, Callable
from catan.gui.interaction import click_events

from PySide6.QtCore import Qt, QEvent, Signal, QObject
from PySide6.QtWidgets import (
    QLabel,
    QToolTip,
    QDoubleSpinBox,
    QFrame,
    QVBoxLayout,
    QPushButton,
    QToolButton,
    QStyle,
)
from PySide6.QtGui import (
    QCursor,
)

import importlib
from catan.gui.panels import StatisticsData
from catan.gui.panels.helper import HistogramMesh, series_with_confidence
from catan.gui.panels.helper.cameras import (
    FixedPanZoomCamera,
)

from catan.gui.data.statistics.engine import StatisticsTaskResult
from catan.gui.data.statistics.errors import StatisticsPlotError
from catan.gui.data.statistics import (
    plotdata_histogram,
    plotdata_scatter,
    plotdata_series,
    PickTable,
    calculations,
)
from catan.gui.structures.state import NeuronComponent
from catan.gui.panels import BasePlot
from catan.gui.panels.helper import Threshold
from catan.gui.background_tasks.runtime import TaskCancelled
import catan.gui.data.curation_filter as curation_filter
from catan.gui.panels.helper.Threshold import ThresholdSpec
from catan.gui.GUI_elements.fragments.ResetViewButton import ResetViewButton

# # importlib.reload(curation_filter)
# importlib.reload(calculations)
# # importlib.reload(stats)
# importlib.reload(series_with_confidence)
# # importlib.reload(plotdata_histogram)
# importlib.reload(plotdata_series)
# importlib.reload(StatisticsData)
# importlib.reload(Threshold)

STATUS_ROW_HEIGHT = 58
STATUS_CARD_IDLE_HEIGHT = 30
STATUS_CARD_BUSY_HEIGHT = 50
STATUS_PROGRESS_HEIGHT = 6

SeriesPoint = tuple[str, int]
VisualIndex = int | SeriesPoint


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

    pick_radius_scatter = 0.05
    pick_radius_series = 0.05

    main_visual: Optional[Callable] = None
    main_visual_name: Optional[str] = None
    overlays: list[BasePlot.SelectionType] = []

    def __init__(self, parent, controls, config=None):
        super().__init__(parent, controls, config)

        self.unfreeze()
        self.grid = self.central_widget.add_grid(spacing=0)

        self.reset_view_button = ResetViewButton(
            native=self.native,
            pos=(6, STATUS_ROW_HEIGHT),
            callback=self.reset_view_range,
        )
        # self._initialize_reset_view_button()

        self.initialize_status_bar()
        self._status_progress_left = None
        self._status_progress_width = None
        self._status_progress_y = None

        self.view = self.grid.add_view(row=1, col=1)
        self.view.stretch = (1, 1)  # expand a lot
        # self.view.camera = FixedPanZoomCamera(aspect=None)
        self.view.camera = scene.PanZoomCamera(aspect=None)

        self.right_view = self.grid.add_view(row=1, col=1)
        # self.right_view.camera = FixedPanZoomCamera(aspect=None)
        self.right_view.camera = scene.PanZoomCamera(aspect=None)
        self.right_view.bgcolor = (0, 0, 0, 0)
        self.plot_root_right = scene.Node(parent=self.right_view.scene)

        self.right_view.interactive = False
        self.view.camera.link(
            self.right_view.camera,
            axis="x",
        )

        self.right_view.visible = False

        self.initialize_axis()

        self.plot_root = scene.Node(parent=self.view.scene)

        self.signals = DisplaySignals()

        self.plot_type = None
        self.plot_data: Optional[
            plotdata_histogram.PlotData
            | plotdata_scatter.PlotData
            | plotdata_series.PlotData
        ] = None

        self._default_ranges = {
            "x": None,
            "y": None,
            "y_2nd": None,
        }

        self.hist_base_layer = None
        self.hist_selected_layer = None

        self._hovered_series_session_id = None

        # self.build_overlays()

        self._on_threshold_changed: Optional[Callable] = None
        self.thresholds: dict[str, Threshold.ThresholdOverlay] = {}

        ## build overlay for displaying messages
        self._build_error_overlay()

        self.events.resize.connect(self._on_canvas_resize)
        self.view.scene.transform.changed.connect(self._on_view_transform_changed)

        self.freeze()

    def change_plot_type(self, plot_type):

        if plot_type != self.plot_type:
            self.clear_overlays()
        self.plot_type = plot_type

        if plot_type == "histogram":
            self.main_visual = Rectangle
            self.main_visual_name = "bar"

            # Selection remains special because it changes
            # histogram COUNTS, rather than overlaying bars.
            self.overlays = ["hovered"]

        elif plot_type == "scatter":
            self.main_visual = Markers
            self.main_visual_name = "marker"

            self.overlays = ["selected", "focused", "highlighted", "hovered"]
        elif plot_type == "series":
            self.main_visual = None
            self.main_visual_name = None

            self.overlays = []
        else:
            self.main_visual = None
            self.main_visual_name = None

            self.overlays = []

    def _on_canvas_resize(self, event):

        self.reset_view_button._reposition()

        if self.error_overlay.isVisible():
            self._position_error_overlay()

    def _on_view_transform_changed(self, event=None):
        if self.thresholds:
            self.update_threshold_visuals()

        if self._hovered_series_session_id is not None:
            self._update_series_hover_line()

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
        self.grid.add_widget(self.axes["x"], row=2, col=1)
        self.axes["x"].link_view(self.view)

        self.axes["y"] = create_axis("left")
        self.grid.add_widget(self.axes["y"], row=1, col=0)
        self.axes["y"].link_view(self.view)

        self.axes["y_2nd"] = create_axis("right")
        self.grid.add_widget(self.axes["y_2nd"], row=1, col=2)
        self.axes["y_2nd"].link_view(self.right_view)

        self.axes["y_2nd"].visible = False

    def initialize_status_bar(self):

        self.status_widget = scene.Widget()

        self.status_widget.height_min = STATUS_ROW_HEIGHT
        self.status_widget.height_max = STATUS_ROW_HEIGHT

        self.grid.add_widget(self.status_widget, row=0, col=1)

        # Main card
        self.status_bg = Rectangle(
            center=(5, 5),
            width=10,
            height=10,
            radius=5,
            color="#f7f8fa",
            border_color="#aeb4bc",
            border_width=1,
            parent=self.status_widget,
        )

        self.status_text = Text(
            "",
            pos=(0, 0),
            anchor_x="left",
            anchor_y="center",
            font_size=9,
            color="#31363d",
            parent=self.status_widget,
        )

        self.status_pending_text = Text(
            "",
            pos=(0, 0),
            anchor_x="left",
            anchor_y="center",
            font_size=8,
            color="#606770",
            parent=self.status_widget,
        )

        # Progress track
        self.status_progress_bg = Rectangle(
            center=(50, 3),
            width=100,
            height=STATUS_PROGRESS_HEIGHT,
            radius=STATUS_PROGRESS_HEIGHT / 2,
            color="#dde1e6",
            parent=self.status_widget,
        )

        # Keep the fill square-ended to avoid radius/width problems
        # when progress is close to zero.
        self.status_progress = Rectangle(
            center=(0.5, 3),
            width=1,
            height=STATUS_PROGRESS_HEIGHT,
            radius=0,
            color="#4f86c6",
            parent=self.status_widget,
        )

        for visual in (
            self.status_bg,
            self.status_text,
            self.status_pending_text,
            self.status_progress_bg,
            self.status_progress,
        ):
            visual.visible = False

        self._status_progress_left = 0.0
        self._status_progress_width = 1.0
        self._status_progress_y = 0.0

        # Important: keep horizontal geometry correct when the window resizes.
        self.status_widget.events.resize.connect(self._on_status_widget_resize)

    def _on_status_widget_resize(self, event):
        self._update_status_geometry(
            busy=self.status_pending_text.visible,
        )

    def _update_status_geometry(
        self,
        *,
        busy: bool,
    ):

        rect = self.status_widget.rect

        width = rect.width
        height = rect.height

        margin_x = 6
        inner_x = 10

        card_left = margin_x
        card_right = width - margin_x
        card_width = max(10, card_right - card_left)

        card_height = STATUS_CARD_BUSY_HEIGHT if busy else STATUS_CARD_IDLE_HEIGHT

        card_top = (height - card_height) / 2

        card_bottom = card_top + card_height

        # Card
        self.status_bg.center = (width / 2, height / 2)
        self.status_bg.width = card_width
        self.status_bg.height = card_height

        text_x = card_left + inner_x

        if not busy:
            self.status_text.pos = (
                text_x,
                height / 2,
            )
            return

        # Busy card:
        # shown line
        # updating line
        # progress bar
        self.status_text.pos = (text_x, card_top + 13)

        self.status_pending_text.pos = (text_x, card_top + 29)

        self._status_progress_left = text_x

        self._status_progress_width = max(
            1,
            card_right - inner_x - self._status_progress_left,
        )

        self._status_progress_y = card_bottom - 6

        self.status_progress_bg.center = (
            self._status_progress_left + self._status_progress_width / 2,
            self._status_progress_y,
        )

        self.status_progress_bg.width = self._status_progress_width

    def _set_status_progress(
        self,
        progress: int | None,
        *,
        visible: bool,
    ):

        self.status_progress_bg.visible = visible

        if not visible:
            self.status_progress.visible = False
            return

        # Pending, but no determinate progress yet:
        # show the track but not a fake 1-pixel progress value.
        if progress is None:
            self.status_progress.visible = False
            return

        self.status_progress.visible = True

        fraction = np.clip(progress / 100.0, 0.0, 1.0)

        fill_width = max(1.0, self._status_progress_width * fraction)

        self.status_progress.center = (
            self._status_progress_left + fill_width / 2,
            self._status_progress_y,
        )

        self.status_progress.width = fill_width

    def set_statistics_status(
        self,
        *,
        displayed: str | None,
        pending: str | None,
        progress: int | None = None,
    ):

        has_status = bool(displayed or pending)

        busy = pending is not None

        if not has_status:

            for visual in (
                self.status_bg,
                self.status_text,
                self.status_pending_text,
                self.status_progress_bg,
                self.status_progress,
            ):
                visual.visible = False

            self.update()
            return

        self.status_bg.visible = True
        self.status_text.visible = True

        self.status_text.text = f"Shown: {displayed}" if displayed else "Shown: —"

        if busy:
            self.status_pending_text.text = f"Updating: {pending}"
            self.status_pending_text.visible = True
        else:
            self.status_pending_text.visible = False

        self._update_status_geometry(
            busy=busy,
        )

        self._set_status_progress(
            progress,
            visible=busy,
        )

        self.update()

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

    def _draw_histogram(self):
        # print("drawing histogram")

        self.clear()
        # self.build_overlays()

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

        self._set_camera_ranges(
            {"x": (min(0.0, xmin) * 1.1, xmax * 1.1), "y": (0.0, ymax * 1.1)}
        )

        # self.axes["x"]._view_changed()

        self._init_threshold_overlays(kind="histogram")
        self.update()

    def _draw_session_series(self):

        self.clear()
        # self.build_overlays()
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
            self.axes["y_2nd"].visible = True
        else:
            # if getattr(self, "right_view", None) is not None:
            self.right_view.visible = False
            # if "y_2nd" in self.axes:
            self.axes["y_2nd"].visible = False

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
            self.axes["y_2nd"].axis.axis_label = second.table.stat.name

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

        self._set_camera_ranges({"x": x_range, "y": left_y_range})

        if is_dual:
            right_y_range = _series_y_range(second)

            self._set_camera_ranges({"x": x_range, "y_2nd": right_y_range})

        # ---------- update axes ----------
        # self.axes["x"]._view_changed()
        # self.axes["y"]._view_changed()

        # if is_dual:
        #     self.axes["y_2nd"]._view_changed()

        # Vertical session guide
        session_line = visuals.Line(
            pos=np.zeros((2, 2), dtype=np.float32),
            width=1.5,
            color=(0.25, 0.25, 0.25, 0.65),
            parent=self.plot_root,
        )

        session_line.visible = False
        session_line.order = 100

        self.plotting["visuals"]["session_hover_line"] = session_line

        first_hover = Markers(
            parent=self.plot_root,
        )

        first_hover.set_data(
            np.zeros((0, 2), dtype=np.float32),
        )

        first_hover.visible = False
        first_hover.order = 110

        self.plotting["visuals"]["session_hover_first"] = first_hover

        second_hover = Markers(
            parent=self.plot_root_right,
        )

        second_hover.set_data(
            np.zeros((0, 2), dtype=np.float32),
        )

        second_hover.visible = False
        second_hover.order = 110

        self.plotting["visuals"]["session_hover_second"] = second_hover

        self.update()

    def _set_camera_ranges(self, ranges):

        if "y" in ranges:
            self.view.camera.set_range(margin=0.0, **ranges)
            # self.view.camera.set_exact_range(**ranges)

        if "y_2nd" in ranges:
            ranges_ = ranges.copy()
            ranges_["y"] = ranges_.pop("y_2nd")
            self.right_view.camera.set_range(margin=0.0, **ranges_)
            # self.right_view.camera.set_exact_range(**ranges_)

        for key in ranges:
            self._default_ranges[key] = ranges[key]
            self.axes[key]._view_changed()

    def reset_view_range(self):

        x = self._default_ranges["x"]
        y = self._default_ranges["y"]

        if x is None or y is None:
            return

        self._set_camera_ranges({"x": x, "y": y})

        if self.right_view.visible and self._default_ranges["y_2nd"] is not None:
            self._set_camera_ranges({"x": x, "y_2nd": self._default_ranges["y_2nd"]})

        self.axes["x"]._view_changed()
        self.axes["y"]._view_changed()
        if self.axes["y_2nd"].visible:
            self.axes["y_2nd"]._view_changed()

        self.update()

    def _draw_scatter(self):

        self.clear()
        # self.build_overlays()

        if not isinstance(self.plot_data, plotdata_scatter.PlotData):
            raise ValueError(
                "plot_data must be an instance of ScatterPlotData for scatter plot."
            )

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

        self._set_camera_ranges(
            {
                "x": (
                    min(0, np.nanmin(self.plot_data.x)) * 1.1,
                    np.nanmax(self.plot_data.x) * 1.1,
                ),
                "y": (
                    min(0, np.nanmin(self.plot_data.y)) * 1.1,
                    np.nanmax(self.plot_data.y) * 1.1,
                ),
            }
        )

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
    def _set_camera_interactive(self, interactive: bool):
        if self.view.camera is not None:
            self.view.camera.interactive = interactive

    def on_mouse_move(self, event):

        if (
            event.pos is None
            or not self.plotting["visuals"]
            # or "error_text" in self.plotting["visuals"]
            or self.plot_data is None
        ):
            return

        # 1. Threshold dragging has priority
        threshold_consumed = False
        for threshold in self.thresholds.values():
            threshold_consumed |= threshold.handle_mouse_move(event)

        if threshold_consumed:
            self._set_camera_interactive(False)
            event.handled = True
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
            self._set_camera_interactive(False)
            hovered_overlay.show_tooltip()
            event.handled = True
            return

        # for threshold in self.thresholds.values():
        #     if threshold.update_hover(event.pos):
        #         # event.handled = True
        #         return

        idx = self.find_closest_visual(event.pos, requires_transform=True)

        if isinstance(self.plot_data, plotdata_histogram.PlotData):
            self.update_style(idx, "hovered")

        elif isinstance(self.plot_data, plotdata_scatter.PlotData):
            self.signals.marker_hovered.emit(idx)

        elif isinstance(self.plot_data, plotdata_series.PlotData):
            self.update_session_series_hover(idx)

        self.handle_tooltip(idx)

        self._set_camera_interactive(True)

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
            # or "error_text" in self.plotting["visuals"]
            or self.plot_data is None
        ):
            return

        threshold_consumed = False
        for threshold in self.thresholds.values():
            if threshold.handle_mouse_release(event):
                threshold_consumed = True

        if threshold_consumed:
            self._set_camera_interactive(True)
            event.handled = True
            return

        idx = self.find_closest_visual(event.pos, requires_transform=True)

        if isinstance(self.plot_data, plotdata_histogram.PlotData):
            self.signals.bin_clicked.emit(idx, event.modifiers)
        elif isinstance(self.plot_data, plotdata_scatter.PlotData):
            self.signals.marker_clicked.emit(idx, event.modifiers)

    def on_mouse_press(self, event):

        if (
            event.button != 1
            or event.pos is None
            or not self.plotting["visuals"]
            # or "error_text" in self.plotting["visuals"]
            or self.plot_data is None
        ):
            return

        threshold_consumed = False

        for threshold in self.thresholds.values():
            if threshold.handle_mouse_press(event):
                threshold_consumed = True
                # return

        if threshold_consumed:
            self._set_camera_interactive(False)
            event.handled = True
            return

        # Click was away from all threshold lines.
        for threshold in self.thresholds.values():
            threshold.deactivate_visuals()

        super().on_mouse_press(event)

    def find_closest_visual(
        self, canvas_pos, requires_transform=False
    ) -> VisualIndex | None:

        if self.plot_data is None:
            return None

        if isinstance(
            self.plot_data,
            plotdata_series.PlotData,
        ):
            return self.find_series_session_from_canvas(canvas_pos)

        if requires_transform:
            data_pos = click_events.canvas_to_visual(
                self.plotting["visuals"]["base"], canvas_pos
            )
        else:
            data_pos = canvas_pos

        if isinstance(self.plot_data, plotdata_histogram.PlotData):
            return self.find_bin_from_data(data_pos)
        elif isinstance(self.plot_data, plotdata_scatter.PlotData):
            return self.find_marker_from_data(data_pos)

        return None

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
        We require (x,y) to be within pick_radius_scatter around marker.
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

        if dist_px > self.pick_radius_scatter:
            return None

        return idx

    def find_series_session_from_canvas(
        self,
        canvas_pos,
    ) -> int | None:

        assert isinstance(
            self.plot_data,
            plotdata_series.PlotData,
        )

        candidates = []

        # First / left series
        first = self.plot_data.first_series

        first_pos = click_events.canvas_to_visual(
            self.plotting["visuals"]["markers"],
            canvas_pos,
        )

        hit = self.find_series_point_from_data(
            first_pos,
            first,
            self.view,
        )

        if hit is not None:
            distance, idx = hit
            candidates.append(
                (
                    distance,
                    int(first.session_ids[idx]),
                )
            )

        # Optional second / right series
        second = self.plot_data.second_series

        if second is not None:

            second_pos = click_events.canvas_to_visual(
                self.plotting["visuals"]["markers_second"],
                canvas_pos,
            )

            hit = self.find_series_point_from_data(
                second_pos,
                second,
                self.right_view,
            )

            if hit is not None:
                distance, idx = hit
                candidates.append(
                    (
                        distance,
                        int(second.session_ids[idx]),
                    )
                )

        if not candidates:
            return None

        _, session_id = min(
            candidates,
            key=lambda item: item[0],
        )

        return session_id

    def find_series_point_from_data(
        self,
        data_pos,
        series,
        view,
    ) -> tuple[float, int] | None:

        x = np.asarray(series.session_ids, dtype=float)
        y = np.asarray(series.values, dtype=float)

        finite = np.isfinite(x) & np.isfinite(y)

        if not np.any(finite):
            return None

        cam_bounds = view.camera.rect

        dx = (x - data_pos[0]) / (cam_bounds.right - cam_bounds.left)
        dy = (y - data_pos[1]) / (cam_bounds.top - cam_bounds.bottom)

        d2 = dx * dx + dy * dy
        d2[~finite] = np.inf

        idx = int(np.argmin(d2))

        distance = float(np.sqrt(d2[idx]))

        if distance > self.pick_radius_series:
            return None

        return distance, idx

    ### ======================================================= ###
    ### ---------------- INTERACTION FUNCTIONS ---------------- ###
    ### ------------------------------------------------------- ###
    ### ------------------------ OUTPUT ----------------------- ###
    ### ======================================================= ###

    def style_records(self, selection, style: str):

        if self.plot_data is None or selection is None:
            return []

        # =========================================================
        # Scatter:
        # component selection -> matching marker IDs
        #
        # Keep all matching marker IDs together as ONE record,
        # so BasePlot creates one batched Markers overlay rather
        # than hundreds of individual Markers visuals.
        # =========================================================

        if isinstance(self.plot_data, plotdata_scatter.PlotData):

            if not isinstance(selection, list):
                selection = [selection]

            marker_ids = self.plot_data.markers_matching_components(selection)

            marker_ids = np.asarray(marker_ids, dtype=int)

            if marker_ids.size == 0:
                return []

            return [marker_ids]

        # =========================================================
        # Histogram:
        # local mouse hover supplies a bin index.
        # =========================================================

        if isinstance(self.plot_data, plotdata_histogram.PlotData):

            if style != "hovered" or not isinstance(selection, numbers.Integral):
                return []

            bin_id = int(selection)

            if (
                bin_id not in self.plotting["data"]
                or self.plot_data.bin_counts[bin_id] <= 0
            ):
                return []

            return [self.plotting["data"][bin_id]]

        return []

    def plot_data_from_rec(
        self,
        rec,
        style: str,
    ) -> dict[str, np.ndarray]:

        if isinstance(self.plot_data, plotdata_histogram.PlotData):

            return {
                "center": rec.center,
                "width": rec.width,
                "height": rec.height,
            }

        if isinstance(self.plot_data, plotdata_scatter.PlotData):

            marker_ids = np.atleast_1d(np.asarray(rec, dtype=int))

            pos = np.column_stack(
                (self.plot_data.x[marker_ids], self.plot_data.y[marker_ids])
            ).astype(np.float32)

            plot_options = self.styles.get_plot_options(
                style,
                "marker",
                0.7,
                size=8.0,
                edge_width=0.0,
            )

            return {"pos": pos, **plot_options}

        return {}

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

    # def highlight_markers_from_selection(self):

    #     if not isinstance(self.plot_data, plotdata_scatter.PlotData):
    #         return

    #     if self.state.selected_components is None:
    #         self.update_style(None, "selected")
    #         return

    #     markers = self.plot_data.markers_matching_components(
    #         self.state.selected_components
    #     )
    #     self.update_style(markers, "selected")

    # def update_style(self, idx: Optional[int | np.ndarray] = None, style="default"):

    #     if isinstance(self.plot_data, plotdata_histogram.PlotData):
    #         if not isinstance(idx, (numbers.Integral, type(None))):
    #             raise ValueError(
    #                 f"Expected idx to be an int or None for HistogramPlotData, but got {type(idx)}"
    #             )
    #         self.update_bin_style(idx, style)
    #     elif isinstance(self.plot_data, plotdata_scatter.PlotData):
    #         self.update_marker_style(idx, style)
    #     elif isinstance(self.plot_data, plotdata_series.PlotData):
    #         self.update_session_series_hover(idx)
    #     else:
    #         return

    # def update_bin_style(self, bin: Optional[int], style: str = "default"):
    #     """Apply base/hover/selected colors to rectangle visuals."""

    #     assert isinstance(self.plot_data, plotdata_histogram.PlotData)
    #     if style == "hovered":
    #         rect = self.plotting["overlays"][style]
    #         if rect is None:
    #             return

    #         if (
    #             bin is None
    #             or self.plotting["data"].get(bin, None) is None
    #             or self.plot_data.bin_counts[bin] <= 0
    #         ):
    #             rect.visible = False
    #             return
    #         data = self.plotting["data"][bin]

    #         rect.visible = True
    #         rect.center = data.center
    #         rect.width = data.width
    #         rect.height = data.height
    #         return

    # def update_marker_style(
    #     self, marker: Optional[int | np.ndarray], style: str = "default", update=True
    # ):
    #     """Show highlight marker on points idx (or hide if idx is None)."""

    #     assert isinstance(self.plot_data, plotdata_scatter.PlotData)

    #     # print(f"updating style '{style}' for marker: {marker}")

    #     self.plotting["overlays"][style].visible = False
    #     if marker is None:
    #         self.plotting["overlays"][style].set_data(
    #             np.zeros((0, 2), dtype=np.float32)
    #         )
    #     else:
    #         plot_options = self.styles.get_plot_options(
    #             style, "marker", 0.7, size=8.0, edge_width=0.0  # , edge_color=None
    #         )

    #         pos = self.plot_data.pos_for_marker(marker)
    #         self.plotting["overlays"][style].visible = True

    #         data = np.column_stack(tuple(pos))
    #         self.plotting["overlays"][style].set_data(
    #             data,
    #             **plot_options,
    #         )

    def _update_series_hover_line(self):

        session_id = self._hovered_series_session_id

        if session_id is None or not isinstance(
            self.plot_data, plotdata_series.PlotData
        ):
            return

        line = self.plotting["visuals"].get("session_hover_line")

        if line is None:
            return

        rect = self.view.camera.rect

        line.set_data(
            pos=np.asarray(
                [
                    [session_id, rect.bottom],
                    [session_id, rect.top],
                ],
                dtype=np.float32,
            )
        )

        line.visible = True

    def update_session_series_hover(
        self,
        session_id: int | None,
    ):
        self._hovered_series_session_id = session_id

        line = self.plotting["visuals"].get("session_hover_line")

        first_hover = self.plotting["visuals"].get("session_hover_first")
        second_hover = self.plotting["visuals"].get("session_hover_second")

        # Hide everything first.
        for visual in (line, first_hover, second_hover):
            if visual is not None:
                visual.visible = False

        if session_id is None or not isinstance(
            self.plot_data,
            plotdata_series.PlotData,
        ):
            return

        # --------------------------------------------
        # Vertical line
        # --------------------------------------------
        self._update_series_hover_line()

        # --------------------------------------------
        # First series
        # --------------------------------------------

        first = self.plot_data.first_series

        idx = _session_index(first, session_id)

        if idx is not None:

            first_hover.set_data(
                np.asarray(
                    [[session_id, first.values[idx]]],
                    dtype=np.float32,
                ),
                size=12,
                face_color=self.styles.get_color_array("selected"),
                edge_width=0,
            )

            first_hover.visible = True

        # --------------------------------------------
        # Second series
        # --------------------------------------------

        second = self.plot_data.second_series

        if second is not None and second_hover is not None:

            idx = _session_index(second, session_id)

            if idx is not None:

                second_hover.set_data(
                    np.asarray(
                        [[session_id, second.values[idx]]],
                        dtype=np.float32,
                    ),
                    size=12,
                    face_color=self.styles.get_color_array("highlighted"),
                    edge_width=0,
                )

                second_hover.visible = True

    def _session_series_tooltip(
        self,
        session_id: int,
    ) -> str:

        assert isinstance(
            self.plot_data,
            plotdata_series.PlotData,
        )

        lines = [self._session_label(session_id)]

        first = self.plot_data.first_series

        idx = _session_index(
            first,
            session_id,
        )

        if idx is not None:
            lines.append(first.tooltip_value_for_index(idx))

        second = self.plot_data.second_series

        if second is not None:

            idx = _session_index(
                second,
                session_id,
            )

            if idx is not None:
                lines.append(second.tooltip_value_for_index(idx))

        return "\n".join(lines)

    def handle_tooltip(self, idx: Optional[VisualIndex] = None):

        if idx is None:
            QToolTip.hideText()
            return

        if isinstance(self.plot_data, plotdata_histogram.PlotData):
            tooltip_text = self.plot_data.tooltip_for_bin(idx)
        elif isinstance(self.plot_data, plotdata_scatter.PlotData):
            tooltip_text = self.plot_data.tooltip_for_marker(idx)
        elif isinstance(self.plot_data, plotdata_series.PlotData):
            tooltip_text = self._session_series_tooltip(idx)
        else:
            tooltip_text = "Unknown selection"

        QToolTip.showText(
            QCursor.pos(),
            tooltip_text,
            self.native,
        )

    def _session_label(
        self,
        session_id: int,
    ) -> str:

        session_id = int(session_id)

        if 0 <= session_id < len(self.data.sessions):
            session = self.data.sessions[session_id]

            if session is not None and session.name:
                return session.name

        return f"Session {session_id}"

    def update_selection(self):
        pass

    def clear(self):
        self._destroy_threshold_overlays()
        self._clear_histogram_mesh_layers()

        for _, visual in self.plotting["visuals"].items():
            self._destroy_visual(visual)

        self.plotting["visuals"] = {}
        self.plotting["data"] = {}

        # self.clear_overlays()

        for axis in self.axes.values():
            axis.visible = False

        self.right_view.visible = False

    def _clear_histogram_mesh_layers(self):
        for attr in ("hist_base_layer", "hist_selected_layer"):
            layer = getattr(self, attr, None)

            if layer is not None:
                layer.destroy()
                setattr(self, attr, None)

    # def clear_overlays(self):
    #     for _, overlay in self.plotting["overlays"].items():
    #         if overlay is None:
    #             continue
    #         overlay.visible = False

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

    def _build_error_overlay(self):

        self.error_overlay = QFrame(self.native)
        self.error_overlay.setObjectName("statisticsErrorOverlay")

        layout = QVBoxLayout(self.error_overlay)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(8)

        self.error_title = QLabel()
        self.error_title.setObjectName("statisticsErrorTitle")

        self.error_message = QLabel()
        self.error_message.setObjectName("statisticsErrorMessage")
        self.error_message.setWordWrap(True)
        self.error_message.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        layout.addWidget(self.error_title)
        layout.addWidget(self.error_message)

        self.error_overlay.setMaximumWidth(500)

        self.error_overlay.setStyleSheet("""
            QFrame#statisticsErrorOverlay {
                background-color: rgba(255, 245, 245, 248);
                border: 1px solid #c94b4b;
                border-radius: 7px;
            }

            QLabel#statisticsErrorTitle {
                color: #9b2c2c;
                background: transparent;
                font-weight: 600;
                font-size: 14px;
            }

            QLabel#statisticsErrorMessage {
                color: #30343b;
                background: transparent;
                font-size: 12px;
            }
        """)

        self.error_overlay.hide()

    def show_plot_error(
        self,
        title: str,
        message: str,
    ):
        self.error_title.setText(title)
        self.error_message.setText(message)

        self.error_overlay.show()
        self.error_overlay.adjustSize()

        self._position_error_overlay()

        self.error_overlay.raise_()

    def _position_error_overlay(self):

        self.error_overlay.adjustSize()

        parent_size = self.native.size()
        size = self.error_overlay.size()

        x = (parent_size.width() - size.width()) // 2

        y = (parent_size.height() - size.height()) // 2

        self.error_overlay.move(max(0, x), max(0, y))

    def hide_plot_error(self):
        self.error_overlay.hide()


class Controller(BasePlot.CanvasController):

    canvas: Display

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Currently selected queries for each axis
        self.current_query: dict[str, StatisticsData.StatisticQuery | None] = {
            "x": None,
            "y": None,
            "y_2nd": None,
        }

        self.current_results: dict[str, StatisticsTaskResult | None] = {
            "x": None,
            "y": None,
            "y_2nd": None,
        }

        self.statistic_tasks: dict[str, str | None] = {
            "x": None,
            "y": None,
            "y_2nd": None,
        }

        self.statistic_progress: dict[str, int | None] = {
            "x": None,
            "y": None,
            "y_2nd": None,
        }

        self.displayed_results = {}
        self.current_plot_data = None

        self.state.tasks.task_progress.connect(self._on_task_progress)
        self.state.tasks.task_cancelled.connect(self._on_statistics_task_stopped)
        self.state.tasks.task_failed.connect(self._on_statistics_task_stopped)

    # def _on_test_button_clicked(self):

    #     if self.current_query["x"] is None:
    #         return

    #     # query = self.current_query["x"]  # Example: using the current x-axis query
    #     condition_a = curation_filter.CurationFilterCondition(
    #         query=self.current_query["x"],
    #         threshold=ThresholdSpec(
    #             value=2.5,
    #             direction="less",
    #             active=True,
    #         ),
    #     )
    #     condition_b = curation_filter.CurationFilterCondition(
    #         query=self.current_query["x"],
    #         threshold=ThresholdSpec(
    #             value=3.0,
    #             direction="greater",
    #             active=True,
    #         ),
    #     )

    #     root = curation_filter.CurationFilterGroup(
    #         operator="and",
    #         match_level="footprint",
    #         children=[
    #             condition_a,
    #             condition_b,
    #         ],
    #     )
    #     evaluator = curation_filter.CurationFilterEvaluator(self.data.statistic_engine)

    #     result_a = evaluator._evaluate_condition(condition_a)
    #     result_b = evaluator._evaluate_condition(condition_b)

    #     result = evaluator.evaluate(root)

    #     print(
    #         "footprint:",
    #         len(result.neurons),
    #         sorted(result.neurons)[:20],
    #     )

    #     print("A neurons:", len(result_a.neurons))
    #     print("B neurons:", len(result_b.neurons))

    #     print("A components:", len(result_a.components))
    #     print("B components:", len(result_b.components))

    #     print(
    #         "neuron intersection:",
    #         len(result_a.neurons & result_b.neurons),
    #     )

    #     print(
    #         "component intersection:",
    #         len(result_a.components & result_b.components),
    #     )

    #     print(result_a.table.dims)
    #     print(result_a.table.refs.keys())

    #     root = curation_filter.CurationFilterGroup(
    #         operator="and",
    #         match_level="neuron",
    #         children=[
    #             condition_a,
    #             condition_b,
    #         ],
    #     )

    #     # root = CurationFilterGroup(
    #     #     operator="and",
    #     #     children=[condition],
    #     # )

    #     evaluator = curation_filter.CurationFilterEvaluator(self.data.statistic_engine)

    #     result = evaluator.evaluate(root)

    #     print(
    #         "neuron:",
    #         len(result.neurons),
    #         sorted(result.neurons)[:20],
    #     )
    #     # print(result)

    def build_controls(self):
        super().build_controls()

        # print("Building controls for PlotController (statistics display)")
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

        # self.controls["test_button"] = QPushButton("Test Filter")
        # self.section.x_options_layout.addWidget(self.controls["test_button"])
        # self.controls["test_button"].clicked.connect(self._on_test_button_clicked)

        self.controls["x_selector"] = StatisticsData.StatisticQuerySelector(
            engine=self.data.statistic_engine, axis="x"
        )
        self.section.x_options_layout.addWidget(self.controls["x_selector"])

        self.controls["x_selector"].queryChanged.connect(
            lambda query: self._on_query_changed("x", query)
        )

        self.controls["y_selector"] = StatisticsData.StatisticQuerySelector(
            engine=self.data.statistic_engine, axis="y"
        )
        self.section.y_options_layout.addWidget(self.controls["y_selector"])

        self.controls["y_selector"].queryChanged.connect(
            lambda query: self._on_query_changed("y", query)
        )

        self.controls["y_selector_2nd"] = StatisticsData.StatisticQuerySelector(
            engine=self.data.statistic_engine, axis="y"
        )
        self.section.y_options_layout.addWidget(self.controls["y_selector_2nd"])

        self.controls["y_selector_2nd"].queryChanged.connect(
            lambda query: self._on_query_changed("y_2nd", query)
        )

        self.section.y_options_layout.addStretch()

        self.controls["bin_selector"].valueChanged.connect(self._on_plot_params_changed)
        self.canvas.signals.marker_hovered.connect(self._on_marker_hovered)
        self.canvas.signals.marker_clicked.connect(self._on_visual_clicked)
        self.canvas.signals.bin_clicked.connect(self._on_visual_clicked)

        self.data.statistic_engine.values_changed.connect(self.recalculate_statistics)

        self.canvas._on_threshold_changed = self._on_threshold_changed

    def _on_data_changed(self, input: Tuple[str, int]):
        # if input[0] == "assignments":
        # self.update_canvas()
        pass

    def _on_plot_params_changed(self):
        self.rebuild_plot()

    def _on_query_changed(self, which, query: StatisticsData.StatisticQuery):

        if which == "x":
            self.current_query["x"] = query
        elif which == "y":
            self.current_query["y"] = query
        elif which == "y_2nd":
            ## should only be possible if plot_type is session_series
            self.current_query["y_2nd"] = query

        self.identify_plot_type()

        for key in ["bin_label", "bin_selector", "y_selector_2nd"]:
            self.controls[key].setVisible(False)

        # print(f"Plot type identified as: {self.plot_type}")

        self._hovered_series_session_id = None

        if self.plot_type == "histogram":
            self.controls["bin_selector"].setVisible(
                self.current_query["x"] is not None
            )
            self.controls["bin_label"].setVisible(self.current_query["x"] is not None)

            self.controls["y_selector"].set_query_mode("generic")
            self.controls["y_selector"].set_query_preparer(None)

            self.controls["x_selector"].set_query_mode("generic")
            self.controls["x_selector"].set_query_preparer(None)
        elif self.plot_type == "session_series":

            self.controls["y_selector_2nd"].setVisible(
                self.current_query["y"] is not None
            )

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
                    self.current_query["y_2nd"] = None
                    self.controls["y_selector_2nd"].set_query(None)
                return

            prepared_query = plotdata_series.prepare_query(
                query, self.data.statistic_engine.registry
            )
            if which == "y":
                self.current_query["y"] = prepared_query
            elif which == "y_2nd":
                self.current_query["y_2nd"] = prepared_query

        elif self.plot_type == "scatter":
            self.controls["y_selector"].set_query_mode("generic")
            self.controls["y_selector"].set_query_preparer(None)

            self.controls["x_selector"].set_query_mode("generic")
            self.controls["x_selector"].set_query_preparer(None)

        self.recalculate_statistics(which)

    def recalculate_statistics(
        self,
        which: str | None = None,
        *,
        force: bool = False,
    ):
        """
        wrapper around _recalculate_statistic for multiple slots.
        """
        if which is None:
            slots = tuple(
                slot for slot, query in self.current_query.items() if query is not None
            )
        else:
            slots = (which,)

        for slot in slots:
            self._recalculate_statistic(
                slot,
                force=force,
            )

    def _recalculate_statistic(
        self,
        which: str,
        *,
        force: bool = False,
    ):
        """
        Recalculates the statistic for the specified slot
        and cancels any obsolete tasks.
        """

        query = self.current_query[which]

        # Cancel obsolete task for THIS slot.
        old_task = self.statistic_tasks[which]

        if old_task is not None:
            self.state.tasks.cancel(old_task)
            self.statistic_tasks[which] = None

        if query is None:
            self.current_results[which] = None
            self.rebuild_plot()
            return

        existing = self.current_results[which]

        if (
            not force
            and existing is not None
            and existing.query == query
            and existing.data_version == self.state.data_version
        ):
            self.rebuild_plot()
            return

        # Important: snapshot the query.
        requested_query = query
        requested_data_version = self.state.data_version

        def evaluate():

            try:
                table = self.data.statistic_engine.evaluate_table(requested_query)

                return StatisticsTaskResult(
                    slot=which,
                    query=requested_query,
                    data_version=requested_data_version,
                    table=table,
                )

            except TaskCancelled:
                # Important: let Worker handle cancellation.
                raise

            except Exception as exc:
                return StatisticsTaskResult(
                    slot=which,
                    query=requested_query,
                    data_version=requested_data_version,
                    error=(f"{type(exc).__name__}: {exc}"),
                    traceback=traceback.format_exc(),
                )

        task_id = self.state.tasks.start(
            "calculating",
            f"Calculate {which}: {requested_query.statistic_key}",
            evaluate,
            on_result=self._on_statistic_ready,
        )

        self.statistic_tasks[which] = task_id

        self.statistic_progress[which] = None
        self._update_statistics_status()

    def _slot_for_task(
        self,
        task_id: str,
    ) -> str | None:

        for slot, slot_task_id in self.statistic_tasks.items():
            if slot_task_id == task_id:
                return slot

        return None

    def _on_statistics_task_stopped(
        self,
        group,
        task_id,
    ):

        if group != "calculating":
            return

        slot = self._slot_for_task(task_id)

        if slot is None:
            return

        self.statistic_tasks[slot] = None
        self.statistic_progress[slot] = None

        self._update_statistics_status()

    def _on_statistic_ready(
        self,
        result: StatisticsTaskResult,
    ):

        which = result.slot

        # A newer query has replaced this one.
        if result.query != self.current_query[which]:
            return

        # Data changed while this calculation was running.
        if result.data_version != self.state.data_version:
            return

        self.statistic_tasks[which] = None
        self.statistic_progress[which] = None
        self.current_results[which] = result

        self._update_statistics_status()
        self.rebuild_plot()

    def _result_is_current(
        self,
        slot: str,
    ) -> bool:

        result = self.current_results[slot]
        query = self.current_query[slot]

        return (
            result is not None
            and result.query == query
            and result.data_version == self.state.data_version
        )

    def _required_statistic_slots(self):

        self.identify_plot_type()

        if self.plot_type == "histogram":
            return ("x",)

        if self.plot_type == "scatter":
            return ("x", "y")

        if self.plot_type == "session_series":
            slots = ["y"]

            if self.current_query["y_2nd"] is not None:
                slots.append("y_2nd")

            return tuple(slots)

        return ()

    def identify_plot_type(self):
        x_query = self.current_query["x"]
        y_query = self.current_query["y"]

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

        self.plot_type = plot_type

        self.canvas.change_plot_type(plot_type)

    def rebuild_plot(self):
        """
        Rebuild the plot once all statistics required for the
        current plot have completed.

        Expected calculation/plot compatibility errors are shown
        inside the Statistics display. Unexpected programming
        errors are allowed to propagate normally.
        """

        slots = self._required_statistic_slots()

        # ---------------------------------------------------------
        # Nothing requested.
        # ---------------------------------------------------------

        if not slots:

            self.current_plot_data = None
            self.displayed_results = {}

            self.canvas.clear()
            self.canvas.hide_plot_error()

            self._update_statistics_status()
            return

        # ---------------------------------------------------------
        # Wait until all required results correspond to the
        # currently requested queries/data version.
        # ---------------------------------------------------------

        for slot in slots:

            if self.current_query[slot] is None:
                return

            if not self._result_is_current(slot):
                return

        # ---------------------------------------------------------
        # We now have one completed result for every required slot.
        # ---------------------------------------------------------

        results = {slot: self.current_results[slot] for slot in slots}

        self.displayed_results = dict(results)

        try:

            # -----------------------------------------------------
            # Calculation errors.
            # -----------------------------------------------------

            errors = [
                result.error for result in results.values() if result.error is not None
            ]

            if errors:
                raise StatisticsPlotError("\n".join(errors))

            # -----------------------------------------------------
            # Internal consistency.
            # This should never happen for a successful result.
            # -----------------------------------------------------

            if any(result.table is None for result in results.values()):
                raise RuntimeError("Successful statistics result contains no table.")

            tables = {slot: result.table for slot, result in results.items()}

            # -----------------------------------------------------
            # Plot-data construction.
            # This may itself raise StatisticsPlotError if the
            # tables cannot sensibly be combined.
            # -----------------------------------------------------

            new_plot_data = self._build_plot_data(tables)

        except StatisticsPlotError as exc:

            self._show_statistics_error(str(exc))
            return

        # ---------------------------------------------------------
        # Success.
        #
        # Only replace the currently visible plot AFTER the new
        # plot data was successfully constructed.
        # ---------------------------------------------------------

        self.current_plot_data = new_plot_data

        self.canvas.clear()
        self.canvas.hide_plot_error()

        self.update_canvas()
        self._update_statistics_status()

    def _build_plot_data(
        self,
        tables: dict[str, PickTable],
    ):

        if self.plot_type == "histogram":

            nbins = int(self.controls["bin_selector"].value())

            return plotdata_histogram.build_plot_data(
                tables["x"],
                bins=nbins,
            )

        if self.plot_type == "session_series":

            return plotdata_series.build_plot_data(
                first_table=tables["y"],
                second_table=tables.get("y_2nd"),
            )

        if self.plot_type == "scatter":

            return plotdata_scatter.build_plot_data(
                x_table=tables["x"],
                y_table=tables["y"],
            )

        # This is an internal bug, NOT a user-facing plotting error.
        raise RuntimeError(f"Unknown plot type {self.plot_type!r}")

    def _show_statistics_error(
        self,
        message: str,
    ):

        self.current_plot_data = None

        self.canvas.clear()

        self.canvas.show_plot_error(
            "Cannot display requested statistics",
            message,
        )

        self._update_statistics_status()

    def update_canvas(self):
        self.canvas.set_plot_data(self.current_plot_data)

        self.update_neuron_selection()

    def update_neuron_selection(self):

        if isinstance(self.current_plot_data, plotdata_histogram.PlotData):
            self.canvas.highlight_bins_from_selection()

        elif isinstance(self.current_plot_data, plotdata_scatter.PlotData):
            self.update_styles()

    def _on_visual_clicked(self, idx: Optional[int], modifiers):

        if idx is None:
            self._handle_picked_ref_sets(None, modifiers)
            return

        if isinstance(
            self.current_plot_data,
            plotdata_histogram.PlotData,
        ):
            rows = self.current_plot_data.rows_for_bin(idx)

        elif isinstance(
            self.current_plot_data,
            plotdata_scatter.PlotData,
        ):
            rows = self.current_plot_data.rows_for_markers(idx)

        else:
            return

        ref_sets = self.current_plot_data.ref_sets_for_rows(rows)

        self._handle_picked_ref_sets(ref_sets, modifiers)

    def _on_marker_hovered(self, marker_id):

        plot_data = self.current_plot_data

        if marker_id is None or not isinstance(plot_data, plotdata_scatter.PlotData):
            self.state.update_hovered_components(None)
            return

        rows = plot_data.rows_for_markers(marker_id)

        ref_sets = plot_data.ref_sets_for_rows(rows)

        components = set()

        for refs in ref_sets:
            components.update(self._components_from_refs(refs))

        self.state.update_hovered_components(list(components) if components else None)

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

            ref_sets = plot_data.ref_sets_for_rows(rows)

            self._handle_picked_ref_sets(ref_sets, [])
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

            ref_sets = plot_data.ref_sets_for_rows(rows)

            self._handle_picked_ref_sets(ref_sets, [])
            return

    def _handle_picked_ref_sets(
        self,
        ref_sets: tuple[dict[str, np.ndarray], ...] | None,
        modifiers,
    ):

        if ref_sets is None:
            self.state.update_selected_components(
                None,
                modifiers,
            )
            return

        components = set()

        for refs in ref_sets:
            components.update(self._components_from_refs(refs))

        if components:
            self.state.update_selected_components(
                list(components),
                modifiers,
            )

    def _components_from_refs(
        self,
        refs: dict[str, np.ndarray],
    ) -> list[NeuronComponent]:
        """
        Convert one coherent PickTable reference set into concrete
        NeuronComponents.

        References within this dict are row-aligned. Do not combine refs
        from different PickTables before calling this method.
        """

        if not refs:
            return []

        refs = {key: np.atleast_1d(value) for key, value in refs.items()}

        component_keys: set[tuple[int, int]] = set()

        def add_components(neuron_key: str, session_key: str | None):
            neurons = np.asarray(refs[neuron_key]).reshape(-1)

            if session_key is None:
                sessions = np.full(
                    neurons.shape,
                    self.state.current_session_id,
                    dtype=int,
                )

            else:
                sessions = np.asarray(refs[session_key]).reshape(-1)

                if sessions.shape != neurons.shape:
                    raise ValueError(
                        "Neuron/session reference arrays are not aligned: "
                        f"{neuron_key} has shape {neurons.shape}, "
                        f"{session_key} has shape {sessions.shape}."
                    )

            component_keys.update(
                (int(session_id), int(neuron_id))
                for session_id, neuron_id in zip(sessions, neurons)
            )

        # ---------------------------------------------------------
        # Ordinary neuron dimension.
        # ---------------------------------------------------------

        if "neuron" in refs:

            if "session" in refs:
                # Ordinary neuron/session statistic.
                add_components("neuron", "session")

            else:
                # One neuron may be associated with a session pair,
                # e.g. the same tracked neuron compared across sessions.
                pair_session_keys = [
                    key for key in ("session_i", "session_j") if key in refs
                ]

                if pair_session_keys:
                    for session_key in pair_session_keys:
                        add_components("neuron", session_key)
                else:
                    add_components("neuron", None)

        # ---------------------------------------------------------
        # Pairwise neuron dimensions.
        # ---------------------------------------------------------

        for suffix in ("i", "j"):

            neuron_key = f"neuron_{suffix}"

            if neuron_key not in refs:
                continue

            session_key = f"session_{suffix}"

            if session_key in refs:
                # neuron_i -> session_i
                # neuron_j -> session_j
                add_components(neuron_key, session_key)

            elif "session" in refs:
                # Both neurons belong to one collapsed/shared session.
                add_components(neuron_key, "session")

            else:
                # No concrete session survives in the statistic.
                add_components(neuron_key, None)

        return [
            NeuronComponent(session_id=session_id, neuron_id=neuron_id)
            for session_id, neuron_id in component_keys
        ]

    ### ================================================================== ###
    ### ================= CONTROL STATUS & ERROR DISPLAYS ================ ###
    ### ================================================================== ###

    def _displayed_query_description(self):

        if not self.displayed_results:
            return None

        return self._describe_results(self.displayed_results)

    def _describe_results(self, results):
        parts = []

        for slot in ("x", "y", "y_2nd"):
            result = results.get(slot)

            if result is None:
                continue

            if result.table is not None:
                title = result.table.stat.display_title
            else:
                # Fallback for failed/incomplete results.
                stat_def = self.data.statistic_engine.registry[
                    result.query.statistic_key
                ]
                title = stat_def.title

            parts.append(f"{slot}: {title}")

        return " | ".join(parts)

    def _pending_slots(self):

        return [
            slot
            for slot, task_id in self.statistic_tasks.items()
            if task_id is not None
        ]

    def _update_statistics_status(self):

        displayed = self._displayed_query_description()

        pending_slots = self._pending_slots()

        if pending_slots:
            pending = ", ".join(
                self._query_short_name(self.current_query[slot])
                for slot in pending_slots
            )
        else:
            pending = None

        self.canvas.set_statistics_status(
            displayed=displayed,
            pending=pending,
            progress=self._current_progress(),
        )

    def _current_progress(self) -> int | None:
        """
        Return progress of the currently running statistics task.

        None means:
        - no statistics task is currently running, or
        - it has not reported determinate progress yet.
        """

        current_task = self.state.tasks.current_task("calculating")

        if current_task is None:
            return None

        task_id = current_task.id

        slot = self._slot_for_task(task_id)

        # Current calculating task belongs to something
        # other than this Statistics controller.
        if slot is None:
            return None

        # Extra guard against superseded tasks.
        if self.statistic_tasks.get(slot) != task_id:
            return None

        return self.statistic_progress.get(slot)

    def _query_short_name(
        self,
        query: StatisticsData.StatisticQuery | None,
    ) -> str:

        if query is None:
            return "—"

        stat_def = self.data.statistic_engine.registry.get(query.statistic_key)

        if stat_def is None:
            return query.statistic_key

        return stat_def.title

    def _on_task_progress(
        self,
        group,
        task_id,
        progress,
    ):

        if group != "calculating":
            return

        slot = self._slot_for_task(task_id)

        if slot is None:
            return

        # Ignore progress from a task that has already
        # been superseded in this slot.
        if self.statistic_tasks[slot] != task_id:
            return

        self.statistic_progress[slot] = progress

        self._update_statistics_status()

    ### ================================================================== ###
    # Deactivation and session handling
    ### ================================================================== ###
    def deactivate(self):

        super().deactivate()

    def _on_session_changed(self):
        super()._on_session_changed()

    def update_styles(self):

        if isinstance(self.current_plot_data, plotdata_scatter.PlotData):
            super().update_styles()


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


def _session_index(
    series,
    session_id: int,
) -> int | None:

    matches = np.flatnonzero(series.session_ids == session_id)

    if matches.size == 0:
        return None

    return int(matches[0])
