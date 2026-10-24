from copy import copy
from html import escape
from textwrap import fill
import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QComboBox,
    QDoubleSpinBox, QPushButton, QLabel, QScrollArea, QGridLayout, QToolButton, QMenu,
    QMessageBox,
)
from catan.gui.GUI_elements.utils.menu_creation import fit_message_box_buttons
from vispy import scene, color
from vispy.geometry import Rect
from vispy.visuals.axis import Ticker
from catan.gui.GUI_elements.fragments.toggle_option import ToggleOption
from vispy.visuals.transforms import STTransform
from catan.core.changes import ChangeKind as C
from catan.gui.panels import BasePlot
from catan.gui.background_tasks.runtime import (
    TaskCancelled, current_task_context, progress_step,
)

PARAMETERS = (
    ("p_same", "Same-neuron fraction", 0.000001, 0.999999, 0.01),
    ("h", "Hard-core distance", 0.000001, 10000, 0.1),
    ("sigma_eff", "Distance spread", 0.000001, 10000, 0.1),
    ("c_diff_mean", "Different: mean correlation", 0, 1, 0.01),
    ("c_diff_sd", "Different: correlation SD", 0.000001, 1, 0.01),
    ("c_same_mean", "Same: mean correlation", 0.000001, 0.999999, 0.01),
    ("c_same_sd", "Same: correlation SD", 0.000001, 1, 0.01),
)


from catan.gui.GUI_elements.fragments.ResetViewButton import ResetViewButton

def tooltip(text):
    return "<qt>" + escape(fill(text, width=48)).replace("\n", "<br>") + "</qt>"


PARAMETER_LABELS = {
    "p_same": ("p(same)", "Prior fraction of pairs belonging to the same neuron."),
    "h": ("h", "Hard-core distance for the different-neuron distribution."),
    "sigma_eff": ("σ_D", "Detection uncertainty. Currently the model uses a single effective distance spread (sigma_eff); detection uncertainty and centroid movement are not yet separate parameters."),
    "c_same_mean": ("μ_c (same)", "Mean footprint correlation for the same neuron."),
    "c_same_sd": ("σ_c (same)", "Standard deviation of footprint correlation for the same neuron."),
    "c_diff_mean": ("μ_c (diff)", "Mean footprint correlation for different neurons."),
    "c_diff_sd": ("σ_c (diff)", "Standard deviation of footprint correlation for different neurons."),
}


class CorrelationTicker(Ticker):
    """Six major ticks at full range; no dense minor-tick comb."""
    def _get_tick_frac_labels(self):
        start, end = self.axis.domain
        low, high = sorted((start, end))
        if high <= low:
            return np.empty(0), np.empty(0), []
        step = .1
        if high-low < .2:
            raw = (high-low)/4
            scale = 10**np.floor(np.log10(raw))
            step = next(v for v in (1, 2, 5, 10) if v*scale >= raw)*scale
        ticks = np.arange(np.ceil((low-1e-10)/step), np.floor((high+1e-10)/step)+1)*step
        return (ticks-start)/(end-start), np.empty(0), [f"{v:g}" for v in ticks]


class BoundedCamera(scene.PanZoomCamera):
    """Clamp every camera update, including linked-camera updates and resize."""
    def __init__(self, **kwargs):
        self.limits = None
        self.count_axis = None
        super().__init__(aspect=None, **kwargs)

    @property
    def rect(self):
        return scene.PanZoomCamera.rect.fget(self)

    @rect.setter
    def rect(self, value):
        r = Rect(*value) if isinstance(value, tuple) else Rect(value)
        if self.limits is not None:
            x, y = self.limits
            coords = []
            for axis, start, size, limits in ((0, r.left, r.width, x), (1, r.bottom, r.height, y)):
                low, high = limits
                span = high - low
                size = float(np.clip(size, max(span*1e-6, 1e-12), span))
                start = low if axis == self.count_axis else float(np.clip(start, low, high-size))
                coords.append((start, size))
            r = Rect(coords[0][0], coords[1][0], coords[0][1], coords[1][1])
        scene.PanZoomCamera.rect.fset(self, r)

    def set_bounds(self, x, y, count_axis=None, reset=False):
        self.limits = (tuple(map(float, x)), tuple(map(float, y)))
        self.count_axis = count_axis
        if reset:
            self.rect = (x[0], y[0], x[1]-x[0], y[1]-y[0])
        else:
            self.rect = self.rect


class Display(QWidget):
    """Linked marginal histograms and a joint evidence/model view."""
    cmap = color.Colormap(["#c52c32", "#fff0b0", "#159447"])

    def __init__(self, section):
        super().__init__(section)
        self.canvases = []
        self.views = []
        self.defaults = []
        self.draw_mode = None
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.host = QWidget()
        self.grid = QGridLayout(self.host)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(0)
        for i in range(2):
            self.grid.setRowStretch(i, 1)
            self.grid.setColumnStretch(i, 1)
        root.addWidget(self.host, 1)
        options = QWidget()
        form = QFormLayout(options)
        self.dimension = QComboBox()
        self.dimension.addItems(["2D", "3D"])
        self.height_mode = QComboBox()
        self.height_mode.addItems(["Counts", "p(same)"])
        self.gaps = QComboBox()
        self.gaps.addItem("1", 1)
        form.addRow("Session difference", self.gaps)
        self.evidence = QLabel()
        self.evidence.setWordWrap(True)
        form.addRow(self.evidence)
        self.calculate_evidence = QPushButton("Calculate missing evidence")
        self.calculate_evidence.setToolTip(tooltip("Calculate missing or stale count records for the selected session difference. This does not refit the model or remove footprints."))
        self.calculate_evidence.hide()
        form.addRow(self.calculate_evidence)
        # Keep the launcher in the empty upper-left quadrant.
        self.launcher = QWidget()
        launch = QVBoxLayout(self.launcher)
        row = QHBoxLayout()
        self.options_button = ToggleOption(options, icon_name="gear",
            tooltip="Model options", popup=True, popup_size=(310, 200), parent=self.launcher)
        row.addWidget(self.options_button)
        self.reset_button = ResetViewButton(self.launcher, (0, 28), self.reset_view)
        self.reset_button.setFixedWidth(44)
        self.reset_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        reset_menu = QMenu(self.reset_button)
        self.revert = reset_menu.addAction("Revert parameter preview")
        self.revert.setToolTip(tooltip("Restore the parameters currently applied to the tracking model. Camera positions are unchanged."))
        self.reset_button.setMenu(reset_menu)
        self.reset_button.setToolTip(tooltip("Reset plot views. Open the arrow menu to revert parameter edits."))
        row.addWidget(self.reset_button)
        info = QToolButton()
        info.setText("ⓘ")
        info.setToolTip(tooltip("Histograms: NN green, non-NN red, total gray. Fits: same green, different red, mixture black. Faded curves and contours show the applied model while editing. NN is evidence, not a ground-truth label. Evidence selection changes only the display."))
        row.addWidget(info)
        row.addStretch()
        launch.addLayout(row)
        parameter_box = QWidget()
        parameter_form = QGridLayout(parameter_box)
        parameter_form.setContentsMargins(0, 0, 0, 0)
        parameter_form.setVerticalSpacing(3)
        parameter_form.setHorizontalSpacing(4)
        parameter_form.setColumnMinimumWidth(2, 22)
        parameter_form.setColumnStretch(5, 1)
        positions = {"p_same": (0, 0), "h": (0, 3), "sigma_eff": (1, 0),
                     "c_same_mean": (2, 0), "c_diff_mean": (2, 3),
                     "c_same_sd": (3, 0), "c_diff_sd": (3, 3)}
        self.editors = {}
        for key, label, low, high, step in PARAMETERS:
            editor = QDoubleSpinBox()
            editor.setDecimals(3)
            editor.setRange(max(low, .001) if low > 0 else low,
                            min(high, .999) if high < 1 else high)
            editor.setSingleStep(step)
            editor.setKeyboardTracking(False)
            self.editors[key] = editor
            editor.setFixedWidth(76)
            editor.setStyleSheet("QDoubleSpinBox { padding: 2px 5px; }")
            short, description = PARAMETER_LABELS[key]
            short = short.replace("σ_D", "σ<sub>D</sub>").replace("μ_c", "μ<sub>c</sub>").replace("σ_c", "σ<sub>c</sub>")
            short = short.replace(" (same)", "").replace(" (diff)", "")
            caption = QWidget()
            caption_layout = QHBoxLayout(caption)
            caption_layout.setContentsMargins(0, 0, 0, 0)
            caption_layout.setSpacing(4)
            if key.startswith("c_"):
                swatch = QLabel()
                swatch.setFixedSize(9, 9)
                swatch.setStyleSheet("background: " + ("#159447" if "same" in key else "#c52c32") + "; border: none;")
                swatch.setToolTip(tooltip(description))
                caption_layout.addWidget(swatch)
            text = QLabel(short)
            text.setToolTip(tooltip(description))
            caption_layout.addWidget(text)

            caption.setToolTip(tooltip(description))
            editor.setToolTip(tooltip(description))
            r, c = positions[key]
            parameter_form.addWidget(caption, r, c, alignment=Qt.AlignmentFlag.AlignRight)
            parameter_form.addWidget(editor, r, c+1)
        self.apply = QPushButton("Apply to model")
        parameter_form.addWidget(self.apply, 4, 0, 1, 5)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setMaximumWidth(270)
        parameter_form.addWidget(self.status, 5, 0, 1, 5)
        launch.addWidget(parameter_box)
        launch.addStretch()
        self.grid.addWidget(self.launcher, 0, 0)
        self.cards = []
        self._built = False
        self._model = None
        self._counts = None
        self._split = True
        self._bounds_key = None
        self._hist_visuals = []
        self._fit_lines = {}
        self._reference_lines = {}
        self._reference = None
        self._reference_surface = None
        self._surface_visual = None
        self._surface_labels = []

    def set_session_count(self, count):
        previous = self.gaps.currentData()
        self.gaps.blockSignals(True)
        self.gaps.clear()
        self.gaps.addItem("0 (same session)", 0)
        for gap in range(1, max(2, count)):
            self.gaps.addItem(str(gap), gap)
        self.gaps.addItem("Any (sum over all)", None)
        index = self.gaps.findData(previous)
        self.gaps.setCurrentIndex(index if index >= 0 else self.gaps.findData(1))
        self.gaps.blockSignals(False)

    def dispose(self):
        for canvas in self.canvases:
            canvas.close()
        self.canvases.clear()

    def reset_view(self):
        for view in self.views:
            cam = view.camera
            if isinstance(cam, BoundedCamera) and cam.limits:
                cam.set_bounds(*cam.limits, count_axis=cam.count_axis, reset=True)
            elif not isinstance(cam, BoundedCamera):
                cam.set_state(self._surface_default)

    def _panel(self, row, col, xlabel, ylabel, help_text, three_d=False):
        card = QWidget()
        card.setObjectName("modelPlotCard")
        card.setStyleSheet("QWidget#modelPlotCard { border: none; background: white; }")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(0, 0, 0, 0)
        canvas = scene.SceneCanvas(show=False, bgcolor="white")
        canvas.native.setStyleSheet("border: none; margin: 0px; padding: 0px;")
        canvas.central_widget.border_color = None
        layout.addWidget(canvas.native)
        self.grid.addWidget(card, row, col)
        self.cards.append(card)
        self.canvases.append(canvas)
        grid = canvas.central_widget.add_grid(spacing=0, margin=0)
        view = grid.add_view(row=0, col=0 if three_d else 1, camera="turntable" if three_d else BoundedCamera())
        grid.border_color = None
        view.border_color = None
        if not three_d:
            view.camera.aspect = None
            for orientation, label, r, c in (("left", ylabel, 0, 0), ("bottom", xlabel, 1, 1)):
                axis = scene.AxisWidget(orientation=orientation, axis_label=label,
                    axis_font_size=9, tick_font_size=8, axis_label_margin=40,
                    tick_label_margin=12, text_color="black", axis_color="#555555",
                    tick_color="#555555")
                if orientation == "left":
                    axis.width_min = axis.width_max = 64
                else:
                    axis.height_min = axis.height_max = 56
                grid.add_widget(axis, row=r, col=c)
                if label == "Correlation":
                    axis.axis.ticker = CorrelationTicker(axis.axis)
                axis.link_view(view)
        button = QToolButton(canvas.native)
        button.setText("ⓘ")
        button.setToolTip(tooltip(help_text))
        button.setAutoRaise(True)
        button.move(66 if not three_d else 4, 4)
        button.show()
        self.views.append(view)
        return view

    def _build(self):
        self.top = self._panel(0, 1, "Correlation", "Counts", "Correlation marginal: NN (green), non-NN (red), total outline (gray); fitted same/different components and mixture.")
        self.left = self._panel(1, 0, "Counts", "Distance", "Distance marginal; counts increase to the left. Counts remain anchored at zero.")
        self.joint = self._panel(1, 1, "Correlation", "Distance", "NN fraction: red to green. Opacity encodes counts; black contours show fitted p(same).")
        self.joint_card = self.cards[-1]
        self.surface_view = self._panel(1, 1, "", "", "Height: counts or fitted p(same), selected on this canvas. Drag to rotate.", three_d=True)
        self.surface_card = self.cards[-1]
        self.surface_card.hide()
        self.joint_controls = QWidget(self.canvases[2].native)
        controls = QHBoxLayout(self.joint_controls)
        controls.setContentsMargins(2, 0, 2, 0)
        controls.setSpacing(4)
        for combo in (self.dimension, self.height_mode):
            combo.setFixedWidth(86)
            combo.setStyleSheet("QComboBox { padding: 2px 5px; }")
            controls.addWidget(combo)
        self.dimension.setToolTip(tooltip("Switch the joint view between 2D and 3D."))
        self.height_mode.setToolTip(tooltip("Counts: observed evidence. p(same): current model probability. Preview references appear as faded contours or a translucent surface."))
        self.joint_controls.adjustSize()
        self.joint_controls.move(98, 4)
        self.joint_controls.show()
        self.left.camera.flip = (True, False, False)
        self.image = scene.visuals.Image(np.zeros((2, 2, 4), dtype=np.float32), parent=self.joint.scene)
        self.image.set_gl_state(depth_test=False, blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'))
        self.image.order = 0
        self.contours = scene.visuals.Isocurve(np.zeros((2, 2)), levels=[.05, .5, .95], color_lev="black", parent=self.joint.scene)
        self.contours.set_gl_state(depth_test=False)
        self.contours.order = 10
        self.reference_contours = scene.visuals.Isocurve(np.zeros((2, 2)), levels=[.05, .5, .95], color_lev=[(0, 0, 0, .25)]*3, parent=self.joint.scene)
        self.reference_contours.set_gl_state(depth_test=False, blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'))
        self.reference_contours.order = 9
        self.reference_contours.visible = False
        for axis, view in (("distance", self.left), ("correlation", self.top)):
            self._fit_lines[axis] = [self._new_line(view, col, width, 20+i)
                for i, (col, width) in enumerate((("black", 3), ("#159447", 3.5), ("#c52c32", 3.5)))]
            self._reference_lines[axis] = [self._new_line(view, (*color.Color(col).rgb, .25), 3, 15+i)
                for i, col in enumerate(("black", "#159447", "#c52c32"))]
        # Never transfer count coordinates between cameras.
        self.top.camera.link(self.joint.camera, axis="x")
        self.left.camera.link(self.joint.camera, axis="y")
        self.surface_view.camera.set_range(x=(-.2, 1), y=(-.2, 1), z=(0, 1), margin=0)
        self._surface_default = self.surface_view.camera.get_state()
        self._built = True

    @staticmethod
    def _new_line(view, col, width, order):
        line = scene.visuals.Line(np.zeros((2, 2)), color=col, width=width,
                                  method="agg", parent=view.scene)
        line.set_gl_state(depth_test=False, blend=True,
                          blend_func=('src_alpha', 'one_minus_src_alpha'))
        line.order = order
        line.visible = False
        return line

    def draw(self, model, counts, split=True, same_session=False):
        """Evidence changed: update data, never replace a canvas or camera."""
        self._model = model
        if model is None:
            for card in self.cards:
                card.hide()
            return
        if not self._built:
            self._build()
        self.cards[0].show()
        self.cards[1].show()
        self._counts = counts
        self._split = split
        self._same_session = same_session
        d = model.arrays["distance_bounds"]
        c = model.arrays["correlation_bounds"]
        key = (tuple(c), tuple(d))
        reset = key != self._bounds_key
        self._bounds_key = key
        self.joint.camera.set_bounds((.5, 1.), (d[0], d[-1]), reset=reset)
        self._update_histograms(reset)
        available = split and bool(model.p_same.get("joint") is not None)
        self.height_mode.model().item(1).setEnabled(available)
        if not available:
            self.height_mode.blockSignals(True)
            self.height_mode.setCurrentIndex(0)
            self.height_mode.blockSignals(False)
        self._rgba = self._evidence_rgba(counts, split)
        self.image.set_data(self._rgba)
        self.image.transform = STTransform(scale=(np.diff(c)[0], np.diff(d)[0]), translate=(c[0], d[0]))
        self.update_model(model)
        self.show_dimension()

    def _evidence_rgba(self, counts, split):
        ratio = np.divide(counts[..., 1], counts[..., 0], out=np.zeros(counts.shape[:2], dtype=float), where=counts[..., 0] > 0)
        rgba = self.cmap.map(ratio.ravel()).reshape((*ratio.shape, 4))
        if not split:
            rgba[..., :3] = color.Color("#c52c32").rgb if self._same_session else .4
        rgba[..., 3] = np.minimum(counts[..., 0]/max(1., counts[..., 0].max()/5), 1)
        return rgba.astype(np.float32)

    def _update_histograms(self, reset):
        for visual in self._hist_visuals:
            visual.parent = None
        self._hist_visuals.clear()
        for axis, view in (("distance", self.left), ("correlation", self.top)):
            horizontal = axis == "distance"
            edges = self._model.arrays[axis+"_bounds"]
            hist = self._counts.sum(axis=1 if horizontal else 0)
            maximum = max(1., float(hist[:, 0].max()))*1.1
            xlim = (0, maximum) if horizontal else (.5, 1.)
            ylim = (edges[0], edges[-1]) if horizontal else (0, maximum)
            # Evidence updates always rescale counts, preserving only the evidence-axis zoom.
            old = Rect(view.camera.rect)
            view.camera.set_bounds(xlim, ylim, count_axis=0 if horizontal else 1, reset=reset)
            if not reset:
                view.camera.rect = ((0, old.bottom, maximum, old.height) if horizontal
                                    else (old.left, 0, old.width, maximum))
            # Filled components; total is an outline, so it cannot cover them.
            for channel, col in ([(2, "#c52c32"), (1, "#159447")] if self._split else [(0, "#c52c32" if self._same_session else "#777777")]):
                x = np.repeat(edges, 2)[1:-1]
                y = np.repeat(hist[:, channel], 2)
                vertices = np.zeros((2*len(x), 2), dtype=np.float32)
                vertices[::2, 0] = vertices[1::2, 0] = x
                vertices[1::2, 1] = y
                if horizontal:
                    vertices = vertices[:, ::-1].copy()
                mesh = scene.visuals.Mesh(vertices=vertices, mode="triangle_strip", color=(*color.Color(col).rgb, .25), parent=view.scene)
                mesh.set_gl_state(depth_test=False, blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'))
                mesh.order = 1
                self._hist_visuals.append(mesh)
            for channel, col in ([(0, "#777777"), (2, "#c52c32"), (1, "#159447")] if self._split else [(0, "#c52c32" if self._same_session else "#777777")]):
                x = np.repeat(edges, 2)[1:-1]
                y = np.repeat(hist[:, channel], 2)
                line = self._new_line(view, col, 1, 5)
                line.set_data(pos=np.column_stack((y, x) if horizontal else (x, y)))
                line.visible = True
                self._hist_visuals.append(line)

    def update_model(self, model):
        """Parameter-only path: update fit lines/contours, without aggregating counts."""
        self._model = model
        if not self._built or self._counts is None:
            return
        self._update_fit_lines(model, self._fit_lines)
        reference = self._reference if self._reference is not model else None
        if reference is not None:
            self._update_fit_lines(reference, self._reference_lines)
        else:
            for lines in self._reference_lines.values():
                for line in lines:
                    line.visible = False
        for source, visual in ((model, self.contours), (reference, self.reference_contours)):
            joint = source.p_same.get("joint") if source is not None and self._split else None
            visual.visible = joint is not None and np.isfinite(joint).any()
            if visual.visible:
                c = source.arrays["correlation_bounds"]; d = source.arrays["distance_bounds"]
                visual.set_data(np.asarray(joint))
                visual.transform = STTransform(scale=(np.diff(c)[0], np.diff(d)[0]), translate=(c[0]+np.diff(c)[0]/2, d[0]+np.diff(d)[0]/2))
        if self.height_mode.currentIndex() == 1:
            self._update_joint_image()
        if self.dimension.currentIndex() == 1 and self.height_mode.currentIndex() == 1:
            self._update_surface()

    def _update_fit_lines(self, model, target):
        from scipy.integrate import cumulative_trapezoid
        pdf = model.distributions.get("pdf", {})
        for axis, lines in target.items():
            for line in lines:
                line.visible = False
            if (not self._split and not self._same_session) or not pdf:
                continue
            edges = model.arrays[axis+"_bounds"]
            centers = (edges[:-1]+edges[1:])/2
            components = []
            for suffix, weight in (("same", 0 if self._same_session else model.parameters["p_same"]), ("diff", 1 if self._same_session else 1-model.parameters["p_same"])):
                density = np.asarray(pdf[axis+"_"+suffix])
                grid = np.linspace(edges[0], edges[-1], len(density))
                cdf = cumulative_trapezoid(density, grid, initial=0)
                if not np.isfinite(cdf[-1]) or cdf[-1] <= 0:
                    break
                components.append(np.diff(np.interp(edges, grid, cdf/cdf[-1]))*self._counts[..., 0].sum()*weight)
            if len(components) == 2:
                for line, values in zip(lines, [sum(components), *components]):
                    line.set_data(pos=np.column_stack((values, centers) if axis == "distance" else (centers, values)))
                    line.visible = True
                if self._same_session:
                    lines[0].visible = lines[1].visible = False
    def _update_joint_image(self):
        joint = self._model.p_same.get("joint") if self._model is not None and self._split else None
        if self.height_mode.currentIndex() == 1 and joint is not None:
            rgba = self.cmap.map(np.clip(joint, 0, 1).ravel()).reshape((*joint.shape, 4))
            self.image.set_data(rgba.astype(np.float32))
        elif self._counts is not None:
            self.image.set_data(self._rgba)

    def update_joint(self, *_):
        if not self._built or self._counts is None:
            return
        self._update_joint_image()
        if self.dimension.currentIndex() == 1:
            self._update_surface()

    def show_dimension(self, *_):
        is3d = self.dimension.currentIndex() == 1
        if not self._built or self._counts is None:
            return
        self.joint_card.setVisible(not is3d)
        self.surface_card.setVisible(is3d)
        native = self.canvases[3 if is3d else 2].native
        self.joint_controls.setParent(native)
        self.joint_controls.move(34 if is3d else 98, 4)
        self.joint_controls.show()
        self.joint_controls.raise_()
        if is3d:
            self._update_surface()

    def _update_surface(self, *_):
        if not self._built or self._counts is None:
            return
        probability = self.height_mode.currentIndex() == 1
        values = self._model.p_same.get("joint") if probability and self._split else None
        if values is None:
            probability = False
            values = self._counts[..., 0]
        c = self._model.arrays["correlation_bounds"]
        centers = (c[:-1]+c[1:])/2
        selected = np.flatnonzero((centers >= .5) & (centers <= 1.))
        values = values[:, selected]
        maximum = 1. if probability else max(1., float(np.nanmax(values)))
        z = np.nan_to_num(values.T/maximum)
        colors = self.cmap.map(z.ravel()) if probability else self._rgba[:, selected].transpose(1, 0, 2).reshape(-1, 4).copy()
        colors[:, 3] = 1
        if self._surface_visual is None:
            self._surface_visual = scene.visuals.SurfacePlot(shading=None, parent=self.surface_view.scene)
        self._surface_visual.set_data(x=np.linspace(0, 1, z.shape[0]), y=np.linspace(0, 1, z.shape[1]), z=z)
        self._surface_visual.set_data(colors=colors)
        reference = self._reference if probability and self._reference is not self._model else None
        ref_joint = reference.p_same.get("joint") if reference is not None else None
        if ref_joint is not None:
            if self._reference_surface is None:
                self._reference_surface = scene.visuals.SurfacePlot(shading=None, parent=self.surface_view.scene)
                self._reference_surface.set_gl_state(depth_test=False, blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'))
                self._reference_surface.order = 10
            ref_z = np.nan_to_num(ref_joint[:, selected].T)
            ref_colors = self.cmap.map(np.clip(ref_z, 0, 1).ravel())
            ref_colors[:, 3] = .18
            self._reference_surface.set_data(x=np.linspace(0, 1, ref_z.shape[0]), y=np.linspace(0, 1, ref_z.shape[1]), z=ref_z)
            self._reference_surface.set_data(colors=ref_colors)
            self._reference_surface.visible = True
        elif self._reference_surface is not None:
            self._reference_surface.visible = False
        # Tick labels only need updating when the units/ranges change.
        c = self._model.arrays["correlation_bounds"]; d = self._model.arrays["distance_bounds"]
        label_key = (.5, 1., float(d[0]), float(d[-1]), maximum, probability)
        if getattr(self, '_surface_label_key', None) == label_key:
            return
        self._surface_label_key = label_key
        for visual in self._surface_labels:
            visual.parent = None
        self._surface_labels.clear()
        for axis, limits, label in ((0, (.5, 1.), "Correlation"), (1, (d[0], d[-1]), "Distance"), (2, (0, maximum), "p(same)" if probability else "Counts")):
            end = np.zeros(3); end[axis] = 1
            self._surface_labels.append(scene.visuals.Line(np.array([np.zeros(3), end]), color="#555555", parent=self.surface_view.scene))
            for fraction in (0., .5, 1.):
                pos = np.zeros(3); pos[axis] = fraction; pos[(axis+1)%3] = -.09
                self._surface_labels.append(scene.visuals.Text(f"{limits[0]+fraction*(limits[1]-limits[0]):.3g}", pos=pos, color="black", font_size=13, parent=self.surface_view.scene))
            pos = np.zeros(3); pos[axis] = .5; pos[(axis+1)%3] = -.24
            self._surface_labels.append(scene.visuals.Text(label, pos=pos, color="black", font_size=15, parent=self.surface_view.scene))


class Controller(BasePlot.BaseDisplayController):
    def configure_display(self):
        self.display_widget = Display(self.section)
        self.section.set_display_widget(self.display_widget)
        self._serial = 0
        self._candidate = None
        self._source = None
        self._evidence_key = None
        self._counts_running = False
        self._missing_evidence = []
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(300)
        self._timer.timeout.connect(self._preview)

    def build_controls(self):
        w = self.display_widget
        for key, editor in w.editors.items():
            editor.valueChanged.connect(lambda value, key=key: self._edited(key))
        w.dimension.currentIndexChanged.connect(w.show_dimension)
        w.height_mode.currentIndexChanged.connect(w.update_joint)
        w.gaps.currentIndexChanged.connect(self._redraw)
        w.calculate_evidence.clicked.connect(self._calculate_evidence)
        w.revert.triggered.connect(self.initialize_display)
        w.apply.clicked.connect(self._apply)

    def initialize_display(self, *_):
        if self._deactivated:
            return
        self._timer.stop()
        self._serial += 1
        self._candidate = None
        self._source = self.data.model
        w = self.display_widget
        w._reference = self._source
        w.apply.setEnabled(False)
        w.set_session_count(len(self.data.sessions))
        self._edited_keys = set()
        for key, editor in w.editors.items():
            editor.blockSignals(True)
            editor.setValue(0 if self._source is None else self._source.parameters.get(key, 0))
            editor.setEnabled(self._source is not None and bool(self._source.parameters))
            editor.blockSignals(False)
        w.status.setText("No fitted model available." if self._source is None or not self._source.parameters
            else "Showing the current model. Edits are previews until Apply.")
        self._redraw()

    def _on_data_changed(self, event):
        if event.has(C.MODEL_COUNTS):
            self._evidence_key = None
        if event.has(C.MODEL_COUNTS, C.MODEL_PARAMETERS):
            self.initialize_display()

    def _edited(self, key):
        self._edited_keys.add(key)
        self._serial += 1
        self._candidate = None
        self.display_widget.apply.setEnabled(False)
        self.display_widget.status.setText("Preparing preview…")
        self._timer.start()

    def _preview(self):
        source = self.data.model
        if source is None:
            return
        serial = self._serial
        revision = self.state.data_version
        # Preserve full precision for fields the user has not edited.
        parameters = dict(source.parameters)
        parameters.update({k: self.display_widget.editors[k].value() for k in self._edited_keys})
        # Capture the model settings before starting the worker.
        snapshot = copy(source)
        snapshot.params = dict(source.params)
        def calculate():
            try:
                return snapshot.parameter_preview(parameters), None
            except Exception as exc:
                return None, str(exc)
        def ready(result):
            if self._deactivated or serial != self._serial or self.data.model is not source:
                return
            if self.state.data_version != revision:
                self._timer.start()
                return
            candidate, error = result
            self._candidate = candidate
            self._source = source
            self._revision = revision
            self.display_widget.apply.setEnabled(candidate is not None)
            self.display_widget.status.setText(error or "Preview only. Apply updates tracking probabilities.")
            if candidate is not None:
                self.display_widget.update_model(candidate)
        self.state.tasks.start("calculating", "Preview model parameters",
            calculate, on_result=ready, background=True)

    def _redraw(self, *_):
        model = self._candidate or self.data.model
        if model is None:
            self._evidence_key = None
            self.display_widget.draw(None, None)
            return
        try:
            # Active sessions only (inactive ones provide no evidence).
            order = self.data.active_session_order()
            gap = self.display_widget.gaps.currentData()
            evidence_key = (id(self.data.model), tuple(order), gap)
            if self._evidence_key == evidence_key:
                self.display_widget.update_model(model)
                return
            aggregate = model.aggregate_counts(session_order=order,
                session_paths=order,
                session_distances=(gap,) if gap is not None and gap > 0 else None)
            self._missing_evidence, total_records = self._evidence_records(gap)
            w = self.display_widget
            w.calculate_evidence.setVisible(bool(self._missing_evidence))
            w.calculate_evidence.setEnabled(not self._counts_running)
            if self._missing_evidence:
                w.status.setText(f"{len(self._missing_evidence)} evidence records missing or stale. Open options to calculate them.")
            else:
                w.status.setText("Preview only. Apply updates tracking probabilities." if self._candidate is not None else "Showing the current model. Edits are previews until Apply.")
            counts = aggregate["cross"].copy()
            split = gap != 0
            if gap == 0:
                counts[:] = 0
                counts[..., 0] = aggregate["same"]
            elif gap is None:
                # Same-session records have no NN/non-NN labels. Add their totals,
                # but do not invent classifications or compare mixed evidence to a fit.
                counts[..., 0] += aggregate["same"]
                split = not np.any(aggregate["same"])
            self.display_widget.evidence.setText(
                f"Displayed evidence: {counts[..., 0].sum():,.0f} pairs; {total_records-len(self._missing_evidence)}/{total_records} records available. "
                + (f"{len(self._missing_evidence)} records missing or stale. Use Calculate missing evidence." if self._missing_evidence else "")
                + (" Within-session pairs (self-pairs excluded); showing the different-neuron distribution." if gap == 0 else " Mixed evidence; showing totals only." if not split else "")
            )
            self.display_widget.draw(model, counts, split=split, same_session=gap == 0)
            self._evidence_key = evidence_key
        except Exception as exc:
            self.display_widget.status.setText(str(exc))

    def _evidence_records(self, gap):
        model = self.data.model
        sessions = self.data.sessions
        requests = []
        # Pairs between active sessions; the gap counts active sessions.
        active = [j for j in self.data.active_session_ids() if sessions[j].path is not None]
        for position, j in enumerate(active):
            session = sessions[j]
            if gap in (0, None):
                requests.append(("same", str(session.path), j, j))
            if gap != 0:
                for earlier, i in enumerate(active[:position]):
                    if gap is None or position - earlier == gap:
                        requests.append(("cross", (str(sessions[i].path), str(session.path)), i, j))
        missing = [r for r in requests if r[1] not in model.counts[r[0]] or r[1] in model.stale_counts[r[0]]]
        return missing, len(requests)

    def _calculate_evidence(self, *_, offer_alignment=True):
        if self._counts_running or self._deactivated:
            return
        tasks = self.state.tasks
        if tasks.processing_busy() or tasks.processing_requested:
            self.display_widget.status.setText("Wait for current processing before calculating evidence.")
            return
        source = self.data.model
        requests, _ = self._evidence_records(self.display_widget.gaps.currentData())
        if not requests:
            return
        # Moving / removing sessions marks later alignments outdated; the
        # evidence needs current ones. Offer to update them first.
        outdated = sorted({sid for r in requests for sid in r[2:]
                           if self.data.alignment_is_stale(sid)})
        if outdated:
            if offer_alignment and self._offer_alignment_update(outdated):
                return
            self.display_widget.status.setText(
                f"{len(outdated)} session alignment(s) are outdated; update them "
                "(Session alignment panel) before calculating evidence.")
            return
        try:
            for sid in {sid for r in requests for sid in r[2:]}:
                self.data.require_current_alignment(sid)
        except Exception as exc:
            self.display_widget.status.setText(str(exc))
            return
        revision = self.state.data_version
        sessions = tuple(self.data.sessions)
        geometry = {sid: self.data.geometry_revision(sid) for r in requests for sid in r[2:]}
        self._counts_running = True
        self.display_widget.calculate_evidence.setEnabled(False)
        self.display_widget.status.setText(f"Calculating {len(requests)} evidence records…")
        def calculate():
            try:
                results = []
                for index, (kind, key, i, j) in enumerate(requests):
                    label = (sessions[j].name if kind == "same"
                             else f"{sessions[i].name} → {sessions[j].name}")
                    with progress_step(index, len(requests),
                                       message=f"Evidence {index + 1}/{len(requests)}: {label}"):
                        result = self.data.calculate_model_counts(sessions[j],
                            ref_data=sessions[i] if kind == "cross" else None, mode=kind,
                            ctx=current_task_context())
                    results.append((kind, key, i, j, result.counts))
                return results, None
            except TaskCancelled:
                raise
            except Exception as exc:
                return None, str(exc)
        task = {}
        def release():
            if task.pop("connected", False):
                tasks.task_cancelled.disconnect(stopped)
                tasks.task_failed.disconnect(stopped)
            self._counts_running = False
        def stopped(group, task_id):
            if task_id != task.get("id"):
                return
            release()
            if not self._deactivated:
                self.display_widget.calculate_evidence.setEnabled(True)
                self.display_widget.status.setText("Evidence calculation stopped.")
        def ready(result):
            release()
            if self._deactivated:
                return
            self.display_widget.calculate_evidence.setEnabled(True)
            results, error = result
            if error:
                self.display_widget.status.setText(error)
                return
            if self.data.model is not source or self.state.data_version != revision:
                self.display_widget.status.setText("Data changed during calculation; results discarded. Retry for the current data.")
                return
            for kind, key, i, j, counts in results:
                if kind == "same":
                    source.set_same_counts(key, counts, source_revision=geometry[j])
                else:
                    source.set_cross_counts(*key, counts, source_revisions=(geometry[i], geometry[j]))
            self._evidence_key = None
            self.data.notify_change(C.MODEL_COUNTS, C.PROCESSING_STATUS)
            self._redraw()
        tasks.task_cancelled.connect(stopped)
        tasks.task_failed.connect(stopped)
        task["connected"] = True
        task["id"] = tasks.start("calculating", "Calculate model evidence", calculate,
                                 on_result=ready, background=True)

    def _offer_alignment_update(self, outdated):
        """Ask to update outdated alignments, then calculate the evidence.
        Returns True if the update was started."""
        # Realigning a session outdates all later ones (and their evidence):
        # update every outdated alignment, not only the requested sessions.
        outdated = [sid for sid in range(len(self.data.sessions))
                    if self.data.alignment_is_stale(sid) and self.data.session_is_active(sid)]
        names = [self.data.sessions[sid].name for sid in outdated]
        listed = ", ".join(names[:6]) + (f" and {len(names) - 6} more" if len(names) > 6 else "")
        box = QMessageBox(self.display_widget)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Outdated alignments")
        box.setText(f"{len(outdated)} session alignment(s) are outdated (sessions were moved or removed).")
        box.setInformativeText(
            f"Evidence needs current alignments: {listed}.\n\n"
            "Update these alignments now and calculate the evidence afterwards? "
            "Sessions that need a review stop the update; calculate again after reviewing.")
        update = box.addButton("Update alignments", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        fit_message_box_buttons(box)
        box.exec()
        if box.clickedButton() is not update:
            return False

        self.display_widget.status.setText(f"Updating {len(outdated)} alignments, then calculating evidence…")
        tasks = self.state.tasks
        # finished runs only when all outdated alignments were updated (not
        # after a stop for review); the evidence waits for other processing.
        self.data.queue_process_alignments(
            session_ids=outdated,
            finished=lambda: tasks.queue_after_processing(
                "Calculate model evidence (after alignment)",
                lambda: self._calculate_evidence(offer_alignment=False),
            ),
        )
        # A refused update reports its reason itself.
        return True

    def _apply(self):
        if self._candidate is None:
            return
        try:
            self.data.apply_model_preview(self._source, self._candidate, self._revision)
        except Exception as exc:
            self.display_widget.status.setText(str(exc))
            return
        self.initialize_display()

    def _cancel_background_tasks(self):
        self._serial += 1
        self._timer.stop()
