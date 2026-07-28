from typing import Dict, Tuple

import numpy as np
import time
from dataclasses import dataclass

from vispy import scene, color
from vispy.scene import visuals
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QLineEdit,
    QVBoxLayout,
    QLabel,
    QWidget,
)

# import importlib
from catan.gui.plots import BasePlot

from catan.gui.structures.state import NeuronComponent
from catan.gui.interaction import click_events

from catan.gui.plots.helper.FootprintSlider import (
    FootprintSliderController,
)
from catan.gui.plots.helper.cameras import (
    XOnlyLockedPanZoomCamera,
)


@dataclass
class TraceMeta:
    #     session_id: int
    #     cluster_id: int
    color: Tuple[float, float, float, float]


@dataclass
class TraceVisualRecord:
    visual: object  # SurfacePlot / Mesh visual
    # pick_points: np.ndarray  # (N, 2) array of (x,y) points for picking
    metadata: TraceMeta


class Display(BasePlot.BaseCanvas):
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

        self.build_overlays()
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

    def update_labels(self, trace_options):
        self.clear_labels()

        # print("Updating trace labels with options:", trace_options)

        offset = 0
        for key, opt in trace_options.items():
            if not opt.isChecked():
                continue
            self.labels[key] = visuals.Text(
                key,
                pos=[0, -offset + 0.5 * self.trace_distance],
                anchor_x="left",
                parent=self.plot_root,
                color="black",
                font_size=10,
            )
            offset += self.trace_distance
            # self.labels[key].set_visible(False)

    def build_overlays(self):

        for style in ["hovered", "focused", "highlighted"]:

            plot_options = self.styles.get_plot_options(style, "line", values=0.7)
            # cmap = color.get_colormap(display_style[style]["cmap"])
            # color_array = cmap.map(np.array([0.7], dtype=np.float32))

            self.plotting["overlays"][style] = visuals.Line(
                **plot_options,
                # color=color_array,
                # width=display_style[style]["width"],
                parent=self.plot_root,
            )
            self.plotting["overlays"][style].visible = False
            self.plotting["overlays"][style].set_gl_state(
                blend=True,
                depth_test=False,
                blend_func=("src_alpha", "one_minus_src_alpha"),
            )
        self.plotting["overlays"]["hovered"].order = 100
        self.plotting["overlays"]["focused"].order = 80
        self.plotting["overlays"]["highlighted"].order = 90

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

        print("traces:", traces.keys())
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

        line = visuals.Line(
            parts,
            **plot_options,
            parent=self.plot_root,
        )
        line.set_gl_state(
            blend=True,
            depth_test=False,
            blend_func=("src_alpha", "one_minus_src_alpha"),
        )

        self.plotting["visuals"][component.id] = line

    def plot_neurons(self, single=False, max_components=10):

        f = 15.0  # sampling frequency - could be specified in GUI
        self.state.logger.debug(f"Updating traces for current neurons")

        t_start = time.time()
        self.clear_traces()

        if (
            self.data is None
            or self.data.current_session is None
            or self.state.selected_components is None
            or len(self.labels) == 0
        ):
            return

        time_lim = [np.inf, -np.inf]

        if single:
            to_plot_components = (
                [self.state.focused_component]
                if self.state.focused_component is not None
                else None
            )
        else:
            to_plot_components = self.state.selected_components

        # to_plot_components = [c for c in to_plot_components if c is not None]
        if not to_plot_components:
            return
        ## restrict number of traces to plot, to avoid performance drop
        to_plot_components = to_plot_components[:max_components]

        height = 1.0 / len(to_plot_components)
        for n, component in enumerate(to_plot_components):

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
            # line = record.visual
            if line.pos is None:
                continue
            neuron = NeuronComponent(key[0], key[1])
            dists = np.linalg.norm(line.pos - mouse_pos, axis=1)
            min_dist = np.nanmin(dists)
            if min_dist < 0.1:  # threshold for picking
                return neuron

        return None

    def update_style(self, component, style="default"):

        if style == "selected":
            return

        if component is None:
            self.plotting["overlays"][style].visible = False
            self.update()
            return

        if (trace := self.plotting["visuals"].get(component.id)) is None:
            return

        self.plotting["overlays"][style].set_data(pos=trace.pos)
        self.plotting["overlays"][style].visible = True

        self.update()

    def clear_labels(self):

        for child in list(self.plot_root.children):
            if isinstance(child, visuals.Text):
                child.parent = None
        self.labels = {}

    def clear_traces(self):
        for visual in self.plotting["visuals"].values():
            visual.parent = None
        for visual in self.plotting["overlays"].values():
            if visual is not None:
                visual.visible = False

        self.plotting["visuals"] = {}

        self._selected = None
        self._hovered = None

    def clear(self):
        # for child in list(self.plot_root.children):
        # child.parent = None
        self.clear_labels()
        self.clear_traces()


class Controller(BasePlot.CanvasController):

    # def __init__(self, display_section, config=None):

    #     super().__init__(display_section, config)
    #     # self.canvas = PlotCanvas(display_section, self.controls, config)

    def build_controls(self):
        super().build_controls()
        self.controls["slider"] = FootprintSliderController(self.section)
        self.section.x_options_layout.addWidget(self.controls["slider"])

        self.controls["trace_options"] = TraceOptionsController(self.section)
        self.section.y_options_layout.addWidget(self.controls["trace_options"])
        self.initialize_display()

        self.controls["trace_options"].options_changed.connect(self.replot_neurons)

    def _on_data_changed(self, input: Tuple[str, int]):
        if input[0] == "traces":
            self.build_controls()
        self.replot_neurons()

    def _on_selection_changed(self):
        self.controls["slider"].update_setup()
        super()._on_selection_changed()

    def _on_focus_changed(self):
        self.controls["slider"].adjust_id()
        super()._on_focus_changed()

    def _on_session_changed(self):
        self.controls["trace_options"].build_trace_checkboxes()
        self.canvas.update_labels(
            self.controls["trace_options"].checkbox_traces_options
        )
        super()._on_session_changed()

    def replot_neurons(self):
        self.canvas.update_labels(
            self.controls["trace_options"].checkbox_traces_options
        )
        self.canvas.plot_neurons(single=self.single_mode)
        self.update_styles()

    def update_neuron_selection(self):
        self.canvas.plot_neurons(single=self.single_mode)
        self.update_styles()


### Trace options ###
class TraceOptionsController(QWidget):
    options_changed = Signal()

    def __init__(self, parent):

        super().__init__(parent)

        self.data = parent.data

        # self.trace_options = QWidget()
        self.trace_options_layout = QVBoxLayout(self)

        # self.checkbox_trace_options = self.build_trace_checkboxes()
        self.checkbox_trace_container = QWidget()
        self.checkbox_traces_layout = QVBoxLayout(self.checkbox_trace_container)
        self.checkbox_traces_options = {}

        self.trace_options_layout.addWidget(self.checkbox_trace_container)

        offset_layout = QVBoxLayout()
        self.trace_offset = QLineEdit()
        self.trace_offset.setFixedWidth(60)
        self.trace_offset.setText("0.0")
        offset_layout.addWidget(
            QLabel("Trace offset:"), alignment=Qt.AlignmentFlag.AlignLeft
        )
        offset_layout.addWidget(self.trace_offset, alignment=Qt.AlignmentFlag.AlignLeft)
        self.trace_options_layout.addLayout(offset_layout)
        offset_layout.addStretch(0)

        self.trace_offset.editingFinished.connect(lambda: self.options_changed.emit())

    # def build_trace_checkboxes(self) -> QWidget:
    #     self.checkbox_traces = QWidget()
    #     self.checkbox_traces_layout = QVBoxLayout(self.checkbox_traces)
    #     self.checkbox_traces_options = {}
    #     return self.checkbox_traces

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
                lambda: self.options_changed.emit()
            )
