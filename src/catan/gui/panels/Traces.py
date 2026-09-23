from typing import Dict, Tuple

import numpy as np
import time
from dataclasses import dataclass

from vispy import scene, color
from vispy.scene import visuals
from vispy.scene.visuals import Line, Text
from PySide6.QtCore import Qt, Signal, QEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QLineEdit,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QWidget,
    QToolButton,
    QFormLayout,
    QButtonGroup,
)

from catan.core.structures import NeuronComponent
from catan.gui.panels import BasePlot

from catan.gui.interaction import click_events
from catan.gui.panels.helper import ReviewStatusFilter, ControlPanel


from catan.gui.panels.helper.cameras import (
    XOnlyLockedPanZoomCamera,
)


@dataclass
class TraceRecord:
    key: Tuple[int | None, int | None]
    pos: np.ndarray


class Display(BasePlot.BaseCanvas):

    main_visual = Line
    main_visual_name: str = "line"

    overlays = ["focused", "highlighted", "hovered"]

    def __init__(self, parent, controls, config=None):
        super().__init__(parent, controls, config)

        self.unfreeze()
        self.grid = self.central_widget.add_grid(spacing=0)
        self.view = self.grid.add_view(row=0, col=0)

        self.plot_root = scene.Node(parent=self.view.scene)

        self.initialize_axis()
        self.trace_distance = 1.3
        self.labels = {}

        self.changes_on_click = "highlighted"  # or "highlighted" or "selected"

        self.control_overlay = None
        self.events.resize.connect(self._on_canvas_resize)

        self.clear()
        self.freeze()

        self.view.stretch = (1, 1)  # expand a lot
        self.view.camera = XOnlyLockedPanZoomCamera(
            aspect=None, on_changed=self.refresh_axis
        )
        # self.connect_axis_to_camera()

    def initialize_axis(self):
        self.axes = scene.AxisWidget(
            orientation="bottom",
            # axis_label="X value",
            tick_direction=(0, 1),
        )
        # self.axes.axis.axis_label_color = "black"
        self.axes.axis.tick_color = "black"
        self.axes.axis.text_color = "black"
        self.axes.axis.axis_color = "black"
        self.axes.axis.axis_width = 2
        self.axes.axis.tick_width = 1
        self.grid.add_widget(self.axes, row=1, col=0)
        self.axes.link_view(self.view)

        # # 🔑 Control layout sizing
        self.axes.height_min = 20
        self.axes.height_max = 35

        # Tell the layout who should expand
        self.axes.stretch = (1, 0.05)  # don't take extra vertical space
        self.axes.visible = True
        self.axes.axis.axis_label = "Time (s)"

    def refresh_axis(self):
        if self.axes is not None:
            self.axes._view_changed()
            self.update()

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

    def update_labels(self, trace_options):
        self.clear_labels()

        offset = 0
        for key, opt in trace_options.items():
            if not opt.isChecked():
                continue
            self.labels[key] = Text(
                key,
                pos=[0, -offset + 0.5 * self.trace_distance],
                anchor_x="left",
                parent=self.plot_root,
                color="black",
                font_size=10,
            )
            offset += self.trace_distance

    def plot_single_trace(self, component: NeuronComponent, offset, height=1.0, f=15.0):

        if component is None:
            return

        footprint_id = self.state.get_footprint_from_component(component)
        session_id = component.session_id
        session = self.data.sessions[session_id]

        traces = session.traces
        if not session.status["traces_loaded"] or session.trace is None:
            return

        time_axis = (np.arange(session.trace.shape[1]) + session.time_offset) / f

        ## build one big line with NaN separators for better performance
        parts = []
        for i, key in enumerate(self.labels):
            if key not in traces:
                continue

            y_vals = traces[key][footprint_id, :].copy()
            y_vals *= height * 0.9 / y_vals.max()

            baseline = -i * self.trace_distance + offset
            y_vals += baseline

            if key in ["S", "S_dff"]:
                segments = np.empty((3 * len(time_axis), 2))

                # start of each vertical line: baseline
                segments[0::3, 0] = time_axis
                segments[0::3, 1] = baseline

                # end of each vertical line: signal value
                segments[1::3, 0] = time_axis
                segments[1::3, 1] = y_vals

                segments[2::3, 0] = time_axis
                segments[2::3, 1] = baseline
                xy = segments
            else:
                xy = np.column_stack([time_axis, y_vals]).astype(np.float32)

            parts.append(xy)
            parts.append(np.array([[np.nan, np.nan]], dtype=np.float32))

        parts = np.vstack(parts)

        # col = self.state.session_colors[self.state.current_session_id]
        plot_options = self.styles.get_plot_options(
            "default",
            "line",
            values=0.7,
            colors=self.state.session_colors[session_id],
        )

        line = Line(
            parts,
            **plot_options,
            parent=self.plot_root,
        )
        line.set_gl_state(
            blend=True,
            depth_test=False,
            blend_func=("src_alpha", "one_minus_src_alpha"),
        )

        self.plotting["data"][component.id] = TraceRecord(component.id, line.pos)
        self.plotting["visuals"][component.id] = line

    def plot_neurons(self, max_components=10):

        f = 15.0  # sampling frequency - could be specified in GUI
        self.state.logger.debug(f"Updating traces for current neurons")
        t_start = time.time()
        self.clear_traces()

        if (
            self.data is None
            or self.data.current_session is None
            or self.state.selected_components is None
            or len(self.labels) == 0
            or self.data.assignments is None
        ):
            return

        time_lim = [np.inf, -np.inf]

        this_neuron = self.state.focused_component.neuron_id

        display_scope = self.controls["panel"].display_scope
        if display_scope == "across":
            if self.state.focused_component is None:
                to_plot_components = None
            else:
                session_presence = np.where(
                    self.data.assignments.ids[this_neuron, :] >= 0
                )

                to_plot_components = [
                    NeuronComponent(neuron_id=this_neuron, session_id=s)
                    for s in session_presence[0]
                ]
        elif display_scope == "adjacent":

            ## find closeby neurons
            union_centroids = self.data.assignments.union.centroids
            distances = np.linalg.norm(
                union_centroids - union_centroids[this_neuron], axis=1
            )
            (to_plot_neurons,) = np.where(distances <= self.state.adjacency_radius)
            to_plot_components = [
                NeuronComponent(neuron_id=n, session_id=self.state.current_session_id)
                for n in to_plot_neurons
                if self.state.assignments[n, self.state.current_session_id] >= 0
            ]
        elif display_scope == "selection":
            to_plot_components = self.state.selected_components
        else:
            raise ValueError(f"Unknown display scope: {display_scope}")

        if not to_plot_components:
            return

        neuron_ids = np.asarray(
            [component.neuron_id for component in to_plot_components], dtype=int
        )

        focused_neuron_id = (
            None
            if self.state.focused_component is None
            else self.state.focused_component.neuron_id
        )

        mask = ReviewStatusFilter.neuron_mask(
            neuron_ids,
            self.data.assignments.review_status,
            self.controls["panel"].review_filter.visible_statuses,
            focused_neuron_id=focused_neuron_id,
            keep_focused=True,
        )

        to_plot_components = [
            component for component, keep in zip(to_plot_components, mask) if keep
        ]

        if not to_plot_components:
            return

        ## restrict number of traces to plot, to avoid performance drop
        to_plot_components = to_plot_components[:max_components]

        height = 1.0 / len(to_plot_components)
        for n, component in enumerate(to_plot_components):

            if component.session_id is None:
                continue

            if not self.data.sessions[component.session_id].traces:
                continue

            self.plot_single_trace(component, n * height, height=height)

            ## update time range based on this component's session
            time_range = (
                np.array(
                    [0, self.data.sessions[component.session_id].traces["C"].shape[1]]
                )
                + self.data.current_session.time_offset
            ) / f

            time_lim[0] = min(time_lim[0], time_range[0])
            time_lim[1] = max(time_lim[1], time_range[1])

        if np.any(np.isinf(time_lim)):
            time_lim = [0, 1]
        y_range = (
            -(len(self.labels) - 1) * self.trace_distance,
            1.0,
        )

        # print(f"Plotted traces in {time.time() - t_start:.2f} seconds")
        self.view.camera.set_range(x=time_lim, y=y_range)
        self.view.camera.set_y_lock_from_current()
        self.view.camera.set_x_locks(*time_lim)
        self.view.camera.set_rect()
        r = self.view.camera.rect
        self.view.camera.rect = r

        self.axes._view_changed()

    def find_closest_component(self, mouse_pos):

        mouse_pos = click_events.canvas_to_visual(
            list(self.plotting["visuals"].values())[0], mouse_pos
        )

        for key, line in self.plotting["visuals"].items():
            # print("key", key)
            # line = record.visual
            if line.pos is None:
                continue
            neuron = NeuronComponent(*key)
            dists = np.linalg.norm(line.pos - mouse_pos, axis=1)
            min_dist = np.nanmin(dists)
            if min_dist < 0.1:  # threshold for picking
                return neuron

        return None

    def plot_data_from_rec(self, rec, style: str) -> dict[str, np.ndarray]:
        plot_options = self.styles.get_plot_options(style, "line", values=0.7)
        return {"pos": rec.pos, **plot_options}

    def clear_labels(self):

        for child in list(self.plot_root.children):
            if isinstance(child, Text):
                child.parent = None
        self.labels = {}

    def clear_traces(self):
        for visual in self.plotting["visuals"].values():
            visual.parent = None
        self.plotting["visuals"] = {}
        self.plotting["data"] = {}

    def clear(self):
        # for child in list(self.plot_root.children):
        # child.parent = None
        self.clear_labels()
        self.clear_traces()
        self.clear_overlays()


class Controller(BasePlot.CanvasController):

    def build_controls(self):
        super().build_controls()

        self.controls["panel"] = TraceOptionsController(self.section)
        self.canvas.attach_control_overlay()

        self.controls["panel"].display_parameter_changed.connect(
            lambda: self.update_neuron_selection()
        )
        self.state.adjacency_radius_changed.connect(self.replot_neurons)
        self.initialize_display()

        self.controls["panel"].displayed_traces_changed.connect(self.replot_neurons)

    def _on_data_changed(self, input: Tuple[str, int]):
        if input[0] == "traces":
            self.build_controls()
        self.replot_neurons()

    def _on_selection_changed(self):
        super()._on_selection_changed()

    def _on_focus_changed(self):
        super()._on_focus_changed()

    def _on_session_changed(self):
        self.controls["panel"].build_trace_checkboxes()
        self.canvas.update_labels(self.controls["panel"].checkbox_traces_options)
        super()._on_session_changed()

    def replot_neurons(self):
        self.canvas.update_labels(self.controls["panel"].checkbox_traces_options)
        self.canvas.plot_neurons()
        self.update_styles()

    def update_neuron_selection(self):
        self.canvas.plot_neurons()
        self.update_styles()


class TraceOptionsController(ControlPanel.ControlPanel):

    data_parameter_changed = Signal()

    displayed_traces_changed = Signal()

    session_only_changed = Signal(bool)
    reset_camera_requested = Signal()

    def __init__(self, parent):
        super().__init__(parent)

        scope_selector = self._build_display_scope_selection(
            {
                "adjacent": {
                    "label": "Nearby",
                    "position": "left",
                },
                "selection": {
                    "label": "Selected",
                    "position": "center",
                },
                "across": {
                    "label": "Across",
                    "position": "right",
                },
            }
        )
        self.form.addRow("Show", scope_selector)

        review_selector = self._build_review_selector()
        self.form.addRow("Review status", review_selector)

        self.checkbox_trace_container = QWidget()
        self.checkbox_traces_layout = QVBoxLayout(self.checkbox_trace_container)
        self.checkbox_traces_options = {}
        self.form.addRow("Traces", self.checkbox_trace_container)

    def build_trace_checkboxes(self):

        current_session = self.data.current_session

        for key, chkbox in self.checkbox_traces_options.items():
            try:
                chkbox.toggled.disconnect()
            except TypeError:
                pass
            chkbox.deleteLater()

        self.checkbox_traces_options.clear()

        if self.data is None or self.data.current_session is None:
            return

        for key in current_session.traces.keys():
            self.checkbox_traces_options[key] = QCheckBox(key)
            self.checkbox_traces_options[key].setChecked(True)

            self.checkbox_traces_layout.addWidget(
                self.checkbox_traces_options[key], alignment=Qt.AlignmentFlag.AlignLeft
            )
            self.checkbox_traces_options[key].toggled.connect(
                lambda: self.displayed_traces_changed.emit()
            )
