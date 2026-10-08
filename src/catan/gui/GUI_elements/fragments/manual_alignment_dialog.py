import numpy as np
from copy import deepcopy

from vispy import scene
from vispy.visuals.transforms import STTransform
from vispy.scene.visuals import Markers, Line, Image
from vispy.geometry import Rect

from PySide6.QtCore import Signal, QTimer, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QGridLayout,
    QToolButton,
    QStyle,
    QPlainTextEdit,
    QSizePolicy,
)

from catan.core.image_correlation import calculate_shift_score_map
from catan.core.structures.remap import Remapping
from catan.core.spatial_geometry import bounded_view

from catan.gui.structures.alignment_draft import AlignmentDraft
from catan.gui.panels.helper.ControlPanel import ControlPanel
from catan.gui.background_tasks.runtime import (
    current_task_context,
    TaskCancelled,
)
from catan.gui.GUI_elements.fragments.alignment_shift_inset import (
    AlignmentShiftInset,
)
from catan.core.alignment import calculate_residual_flow


class AlignmentCamera(scene.PanZoomCamera):
    def __init__(self):
        self._alignment_bounds = None
        super().__init__(aspect=1)

    def set_bounds(self, bounds):
        bounds = tuple(map(float, bounds))
        if bounds != self._alignment_bounds:
            self._alignment_bounds = bounds
            self.view_changed()

    def _update_transform(self):
        if self._resetting or self._viewbox is None:
            return

        bounds = self._alignment_bounds
        vw, vh = self._viewbox.size

        if bounds is not None and vw > 0 and vh > 0:
            rect = self.rect

            self._rect = Rect(
                *bounded_view(
                    bounds,
                    (
                        rect.left,
                        rect.bottom,
                        rect.width,
                        rect.height,
                    ),
                    (vw, vh),
                )
            )

        super()._update_transform()


class ManualAlignmentEditor(QWidget):
    apply_requested = Signal()
    popout_requested = Signal()
    review_closed = Signal()

    def __init__(self, data, draft, parent=None):
        super().__init__(parent)

        self.data = data
        self.state = data.state
        self.draft = draft
        self._disposed = False

        self._score_task_id = None
        self._score_generation = 0
        self._requested_score_key = None
        self._score_key = None
        self._score_result = None
        self._displayed_score_key = None
        self._estimate_task_id = None
        self._estimate_generation = 0

        self._pair_shift_task_id = None
        self._pair_shift_key = None
        self._pair_shift_generation = 0
        self._pair_shift_cache = {}

        self._fit_next_score = True

        self._hover_context = None
        self._hover_draft = None
        self._pending_hover = None
        self._inset_session_id = None
        self._score_cache = {}

        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(150)
        self._hover_timer.timeout.connect(self._show_hover)

        session_id = draft.session_id
        session = draft.session
        self.template = draft.template
        self.dims = draft.dims

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.canvas = scene.SceneCanvas(
            keys="interactive",
            bgcolor="#20252b",
        )
        layout.addWidget(self.canvas.native, 1)

        self.options = ControlPanel(self)
        self.options.setParent(self.canvas.native)
        self.options.toggle_button.setToolTip("Alignment settings")

        form = self.options.form
        self.deformation_status = QLabel(self.options)
        self.deformation_status.setWordWrap(True)
        form.addRow(self.deformation_status)

        self.transpose_background = QCheckBox("Transpose background", self.options)
        self.transpose_background.setToolTip(
            "Swap the background image's rows and columns before alignment. "
            "Footprints retain their existing orientation. "
            "This correction currently requires a square background."
        )
        self.transpose_background.toggled.connect(self._set_background_transposed)
        form.addRow(self.transpose_background)

        self.legend = QLabel(
            "Reference: green · This session: magenta\n"
            "Overlap: white · Shifts in pixels",
            self.options,
        )
        self.legend.setWordWrap(True)

        self.view_mode = QComboBox(self)
        self.view_mode.addItem("Background overlay", "overlay")
        self.view_mode.addItem("Correlation surface", "correlation")
        self.view_mode.addItem("Estimated flow", "flow")

        self.correlation_method = QComboBox(self)
        for label, method in (
            ("Cross-correlation", "correlation"),
            ("Cosine — whole-image norms", "cosine"),
            ("Cosine — support-weighted", "cosine_union"),
            ("Pearson — overlapping pixels", "pearson"),
        ):
            self.correlation_method.addItem(label, method)

        self.correlation_method.setEnabled(False)

        self.score_status = QLabel(self)
        self.score_status.setWordWrap(True)
        self.score_status.hide()
        form.addRow(self.score_status)

        self.reference = QComboBox(self)
        self._references = []
        self._reference_session_ids = []
        self._populate_references()

        self.reference.setToolTip(
            "Changes the visual comparison only; "
            "transform coordinates remain global."
        )

        form.addRow("Display", self.view_mode)
        self.flow_scale = QDoubleSpinBox(self.options)
        self.flow_scale.setRange(0.1, 1000.0)
        self.flow_scale.setDecimals(1)
        self.flow_scale.setSingleStep(5.0)
        self.flow_scale.setValue(20.0)
        self.flow_scale.setSuffix("×")
        self.flow_scale.setToolTip(
            "Magnifies arrows only. Reported displacements remain in pixels."
        )
        form.addRow("Flow arrow scale", self.flow_scale)

        self.flow_status = QLabel(self.options)
        self.flow_status.setWordWrap(True)
        form.addRow(self.flow_status)

        self.flow_scale.hide()
        form.labelForField(self.flow_scale).hide()
        self.flow_status.hide()

        self.flow_scale.valueChanged.connect(lambda _value: self._draw_flow())

        form.addRow("Method", self.correlation_method)

        self.correlation_method.hide()
        form.labelForField(self.correlation_method).hide()

        # Retain its selection bookkeeping, without displaying a dropdown.
        self.reference.hide()
        self.legend.hide()

        # Apply lives below the popup canvas, or in the panel's x layout.
        self.view_controls = QWidget(self)
        controls = QHBoxLayout(self.view_controls)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addStretch()
        layout.addWidget(self.view_controls)

        def spin(minimum, maximum, step):
            widget = QDoubleSpinBox(self)
            widget.setRange(minimum, maximum)
            widget.setDecimals(2)
            widget.setSingleStep(step)
            widget.setKeyboardTracking(False)
            return widget

        height, width = self.dims
        self.dx = spin(-2 * width, 2 * width, 0.25)
        self.dy = spin(-2 * height, 2 * height, 0.25)
        self.angle = spin(-180, 180, 0.1)

        self.shift_controls = QWidget(self.canvas.native)
        self.shift_controls.setObjectName("alignmentCoordinates")
        self.shift_controls.setAttribute(Qt.WidgetAttribute.WA_NoMousePropagation, True)

        coordinates = QGridLayout(self.shift_controls)
        coordinates.setContentsMargins(5, 5, 5, 5)
        coordinates.setSpacing(3)

        for widget in (self.dx, self.dy):
            widget.setPrefix("")
            widget.setFixedWidth(widget.fontMetrics().horizontalAdvance("-999.99") + 26)

        self.dx.setToolTip("Horizontal shift, dx (pixels)")
        self.dy.setToolTip("Vertical shift, dy (pixels)")

        crosshair = QLabel("⌖", self.shift_controls)
        crosshair.setAlignment(Qt.AlignmentFlag.AlignCenter)
        crosshair.setFixedWidth(22)
        crosshair.setToolTip("Global translation in pixels")

        coordinates.setContentsMargins(0, 0, 0, 0)
        coordinates.setSpacing(2)

        coordinates.addWidget(
            self.dy,
            0,
            0,
            1,
            2,
            alignment=Qt.AlignmentFlag.AlignRight,
        )
        coordinates.addWidget(self.dx, 1, 0)
        coordinates.addWidget(crosshair, 1, 1)

        self.shift_controls.setStyleSheet("""
            QWidget#alignmentCoordinates {
                background: transparent;
                border: none;
            }
            QLabel {
                color: #e8eaed;
                border: none;
                font-size: 20px;
            }
            QDoubleSpinBox {
                color: #e8eaed;
                background: #343941;
                border: 1px solid #69727f;
                border-radius: 4px;
                padding: 3px;
            }
        """)

        self.edit_rotation = QToolButton(self.shift_controls)
        self.edit_rotation.setText("↻")
        self.edit_rotation.setCheckable(True)
        self.edit_rotation.setChecked(True)
        self.edit_rotation.setFixedWidth(24)
        self.edit_rotation.setToolTip(
            "Enable rotation editing. Turning this off retains " "the current angle."
        )

        self.angle.setSuffix("°")
        self.angle.setFixedWidth(self.dx.width())
        self.angle.setToolTip("Session rotation")

        coordinates.addWidget(self.angle, 2, 0)
        coordinates.addWidget(self.edit_rotation, 2, 1)

        self.edit_rotation.toggled.connect(lambda _checked: self._update_availability())

        estimate_row = QWidget(self.options)
        estimate_layout = QHBoxLayout(estimate_row)
        estimate_layout.setContentsMargins(0, 0, 0, 0)

        self.estimate_rotation_button = QPushButton("Estimate rotation")
        self.estimate_shift_button = QPushButton("Estimate shift")

        self.estimate_rotation_button.setToolTip(
            "Estimate rotation near the current angle, keeping dx and dy fixed. "
            "Uses whole-image cosine similarity."
        )
        self.estimate_shift_button.setToolTip(
            "Estimate dx and dy, keeping the current rotation fixed. "
            "Uses whole-image cosine similarity."
        )

        self.estimate_rotation_button.setText("Auto ↻")
        self.estimate_rotation_button.setFixedWidth(58)
        coordinates.addWidget(self.estimate_rotation_button, 2, 2)
        estimate_layout.addWidget(self.estimate_shift_button)
        form.addRow(estimate_row)

        self.estimate_status = QLabel(self.options)
        self.estimate_status.setWordWrap(True)
        form.addRow(self.estimate_status)

        self.estimate_rotation_button.clicked.connect(
            lambda: self._estimate_alignment("rotation")
        )
        self.estimate_shift_button.clicked.connect(
            lambda: self._estimate_alignment("shift")
        )

        self.refine_flow_button = QPushButton("Refine flow", self.shift_controls)
        self.refine_flow_button.setToolTip(
            "Calculate optical-flow correction from the current shift "
            "and rotation against the selected reference session. "
            "Preview the result, then Apply changes."
        )
        coordinates.addWidget(self.refine_flow_button, 3, 0, 1, 3)

        self.refine_flow_button.clicked.connect(
            lambda: self._estimate_alignment("flow")
        )

        grid = self.canvas.central_widget.add_grid(spacing=0)

        self.y_axis = scene.AxisWidget(orientation="left")
        self.y_axis.width_min = self.y_axis.width_max = 65
        grid.add_widget(self.y_axis, row=0, col=0)

        self.view = grid.add_view(row=0, col=1)
        self.view.camera = AlignmentCamera()
        self.view.camera.flip = (False, False, False)

        self.x_axis = scene.AxisWidget(orientation="bottom")
        self.x_axis.height_min = self.x_axis.height_max = 50
        grid.add_widget(self.x_axis, row=1, col=1)

        for axis in (self.x_axis, self.y_axis):
            axis.axis.text_color = "#e8eaed"
            axis.axis.tick_color = "#aab2bd"
            axis.axis.axis_color = "#aab2bd"
            axis.link_view(self.view)

        self.x_axis.axis.axis_label = "x (px)"
        self.y_axis.axis.axis_label = "y (px)"

        self._flow_key = None
        self._flow_result = None
        self._flow_pending = None
        self._flow_task_id = None
        self._flow_generation = 0

        self._flow_timer = QTimer(self)
        self._flow_timer.setSingleShot(True)
        self._flow_timer.setInterval(200)
        self._flow_timer.timeout.connect(self._start_flow)

        self.image = Image(
            np.zeros(self.dims + (3,), dtype=np.float32),
            parent=self.view.scene,
        )
        self.image.order = 0
        self.score_image = Image(
            np.zeros((1, 1), dtype=np.float32),
            texture_format="r32f",
            cmap="viridis",
            clim=(0, 1),
            parent=self.view.scene,
        )
        self.score_image.order = 0
        self.score_image.visible = False
        self.score_image.set_gl_state("translucent", depth_test=False)

        self.flow_arrows = Line(
            pos=np.zeros((2, 2), dtype=np.float32),
            connect="segments",
            color="#ffe066",
            width=2,
            parent=self.view.scene,
        )
        self.flow_arrows.order = 5
        self.flow_arrows.visible = False
        self.flow_arrows.set_gl_state("translucent", depth_test=False)

        self._color_window = None
        self.canvas.events.draw.connect(self._update_visible_clim, position="first")

        self.shift_arrow = Line(
            pos=np.zeros((6, 2), dtype=np.float32),
            connect="segments",
            color="#15191f",
            width=3,
            parent=self.view.scene,
        )
        self.shift_arrow.order = 1
        self.shift_arrow.visible = False

        self.shift_tip = Markers(parent=self.view.scene)
        self.shift_tip.set_data(
            pos=np.zeros((1, 2), dtype=np.float32),
            face_color="#ffffff",
            edge_color="#15191f",
            edge_width=2,
            size=6,
        )
        self.shift_tip.order = 2
        self.shift_tip.visible = False

        # All three visuals share a plane. Use their draw order
        # rather than depth testing to compose the overlay.
        for visual in (self.image, self.shift_arrow, self.shift_tip):
            visual.set_gl_state("translucent", depth_test=False)

        self.pair_arrow = Line(
            pos=np.zeros((6, 2), dtype=np.float32),
            connect="segments",
            color="#ed64bc",
            width=1.5,
            parent=self.view.scene,
        )
        self.pair_tip = Markers(parent=self.view.scene)
        self.pair_tip.set_data(
            pos=np.zeros((1, 2), dtype=np.float32),
            face_color="#ed64bc",
            edge_color="#ffffff",
            edge_width=1,
            size=6,
        )

        self.pair_arrow.visible = False
        self.pair_tip.visible = False

        # Keep the draggable total-shift handle above the fixed estimate.
        for order, visual in enumerate(
            (
                self.pair_arrow,
                self.pair_tip,
                self.shift_arrow,
                self.shift_tip,
            ),
            start=1,
        ):
            visual.order = order
            visual.set_gl_state("translucent", depth_test=False)

        self.reset_button = QToolButton(self.options)
        self.reset_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload)
        )
        self.reset_button.setToolTip("Discard unapplied changes and reset both views")
        self.reset_button.clicked.connect(self._reset)

        self.popout_button = QToolButton(self.options)
        self.popout_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarNormalButton)
        )
        self.popout_button.setToolTip("Open alignment in a separate window")
        self.popout_button.clicked.connect(self.popout_requested.emit)

        header = QWidget(self.options)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(3)
        header_layout.addStretch()

        self.options.layout().removeWidget(self.options.toggle_button)
        for button in (
            self.reset_button,
            self.popout_button,
            self.options.toggle_button,
        ):
            button.setFixedSize(28, 28)
            header_layout.addWidget(button)

        self.options.layout().insertWidget(0, header)

        self.status_label = QLabel(self.canvas.native)
        self.status_label.setWordWrap(True)
        self.status_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self.status_label.setStyleSheet("""
            QLabel {
                color: #e8eaed;
                background: rgba(35, 39, 45, 220);
                border: 1px solid #59616c;
                border-radius: 4px;
                padding: 4px 7px;
            }
        """)

        self.apply_button = QPushButton("Apply", self)
        self.view_controls.layout().addWidget(self.apply_button)
        self.apply_button.clicked.connect(self._apply)

        self.review_message = QPlainTextEdit(self)
        self.review_message.setReadOnly(True)
        self.review_message.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.review_message.setFixedHeight(
            4 * self.review_message.fontMetrics().lineSpacing() + 16
        )
        self.review_message.setStyleSheet("""
            QPlainTextEdit {
                color: #ffe1a3;
                background: #292d33;
                border: 1px solid #b58b43;
                border-radius: 4px;
                padding: 4px;
            }
        """)
        self.review_message.hide()
        layout.insertWidget(0, self.review_message)

        self.remove_button = QPushButton("Remove session…", self)
        self.remove_button.setToolTip(
            "Remove this session from the project; keep its original files."
        )
        self.remove_button.clicked.connect(self._remove_session)
        self.view_controls.layout().insertWidget(0, self.remove_button)

        self.later_button = QPushButton("Review later", self)
        self.later_button.clicked.connect(self._review_later)
        self.view_controls.layout().insertWidget(0, self.later_button)

        self.blocker_button = QPushButton(self)
        self.blocker_button.hide()
        self.blocker_button.clicked.connect(self._open_blocking_alignment)
        self.view_controls.layout().insertWidget(0, self.blocker_button)

        self.view_mode.currentIndexChanged.connect(self._on_view_mode_changed)
        self.correlation_method.currentIndexChanged.connect(self._render_preview)
        self.reference.currentIndexChanged.connect(self._on_reference_changed)

        for index, widget in enumerate((self.dx, self.dy, self.angle)):
            widget.valueChanged.connect(
                lambda value, index=index: self._set_parameter(index, value)
            )

        self.shift_inset = AlignmentShiftInset(self.canvas.native)
        self.shift_inset.chosen.connect(self._choose_inset_item)
        self.shift_inset.shift_changed.connect(self._set_inset_shift)
        self.shift_inset.hovered.connect(self._hover_inset)
        self.shift_inset.interaction_started.connect(self._hover_timer.stop)
        self.shift_inset.focus_released.connect(self._release_inset_focus)

        show_shifts = QCheckBox("Show shift inset", self.options)
        show_shifts.setChecked(True)
        show_shifts.toggled.connect(self.shift_inset.setVisible)
        form.addRow(show_shifts)

        self.pair_shift_status = QLabel(self.options)
        self.pair_shift_status.setWordWrap(True)
        form.addRow(self.pair_shift_status)

        self.recalculate_pair_shifts = QPushButton(
            "Recalculate comparisons", self.options
        )
        self.recalculate_pair_shifts.clicked.connect(self._retry_pair_shifts)
        form.addRow(self.recalculate_pair_shifts)

        self._refresh_shift_inset(fit=True)
        self.shift_inset.show()

        self._correlation_drag = None
        self._correlation_mouse_connections = (
            (self.canvas.events.mouse_press, self._on_correlation_press),
            (self.canvas.events.mouse_move, self._on_correlation_move),
            (self.canvas.events.mouse_release, self._on_correlation_release),
        )

        for signal, slot in self._correlation_mouse_connections:
            signal.connect(slot, position="first")

        self._connections = (
            (self.state.alignment_draft_changed, self._sync_from_draft),
            (self.state.data_changed, self._update_availability),
            (self.state.tasks.queue_changed, self._update_availability),
            (self.state.tasks.task_cancelled, self._on_score_task_stopped),
            (self.state.tasks.task_failed, self._on_score_task_stopped),
            (
                self.state.alignment_review_changed,
                self._update_availability,
            ),
            (self.state.tasks.task_cancelled, self._on_flow_task_stopped),
            (self.state.tasks.task_failed, self._on_flow_task_stopped),
            (self.state.tasks.task_cancelled, self._on_estimate_stopped),
            (self.state.tasks.task_failed, self._on_estimate_stopped),
            (
                self.state.tasks.task_cancelled,
                self._on_pair_shift_stopped,
            ),
            (
                self.state.tasks.task_failed,
                self._on_pair_shift_stopped,
            ),
            (
                self.state.data_changed,
                self._on_pair_shift_data_changed,
            ),
        )
        for signal, slot in self._connections:
            signal.connect(slot)

        # Also release the canvas when a whole display section is deleted.
        canvas = self.canvas
        self.destroyed.connect(lambda *_: canvas.close())

        self._sync_from_draft()
        self._fit_view()

        self.options.setStyleSheet(self.options.styleSheet() + """
            QWidget#footprintParameterOverlay QPushButton,
            QWidget#footprintParameterOverlay QComboBox,
            QWidget#footprintParameterOverlay QDoubleSpinBox {
                color: #e8eaed;
                background-color: #343941;
                border: 1px solid #69727f;
                border-radius: 4px;
                padding: 3px 5px;
            }

            QWidget#footprintParameterOverlay QPushButton:hover {
                background-color: #414751;
            }

            QWidget#footprintParameterOverlay QPushButton:disabled,
            QWidget#footprintParameterOverlay QDoubleSpinBox:disabled {
                color: #8b929c;
            }
            """)

        self.options.overlay_layout_changed.connect(self._position_options)
        self.canvas.events.resize.connect(self._position_options)

        self.options.show()
        self._position_options()

    def _set_background_transposed(self, checked):
        if not self.can_apply() or not self.draft.can_transpose_background:
            return

        checked = bool(checked)
        if checked == self.draft.background_transposed:
            return

        self._cancel_score_task()
        self.draft.background_transposed = checked
        self.state.alignment_draft_changed.emit()

    def _position_options(self, *_):
        if self._disposed:
            return

        margin = 8
        available_width = max(1, self.canvas.native.width() - 2 * margin)

        self.options.setMaximumWidth(min(280, available_width))
        self.options.adjustSize()
        self.options.move(
            max(
                margin,
                self.canvas.native.width() - self.options.width() - margin,
            ),
            margin,
        )
        self.options.raise_()

        # Keep it inside the plotting area, clear of the left axis.
        self.shift_inset.move(75, 8)
        self.shift_inset.raise_()

        self.shift_controls.adjustSize()
        self.shift_controls.move(
            max(
                75,
                self.canvas.native.width() - self.shift_controls.width() - margin,
            ),
            max(
                margin,
                self.canvas.native.height()
                - 50
                - self.shift_controls.height()
                - margin,
            ),
        )
        self.shift_controls.raise_()

        self.status_label.setMaximumWidth(
            max(
                100,
                self.canvas.native.width() - self.shift_controls.width() - 100,
            )
        )
        self.status_label.adjustSize()
        self.status_label.move(
            75,
            max(
                margin,
                self.canvas.native.height() - 50 - self.status_label.height() - margin,
            ),
        )
        self.status_label.raise_()

    def _update_comparison_label(self, draft, reference_id):
        if reference_id is None:
            reference = "own background"
        else:
            reference = f"S{reference_id}"

        suffix = " · unapplied" if draft is self.draft and draft.dirty else ""
        deformation = " · ∿ flow" if draft.effective_flow is not None else " · rigid"

        self.status_label.setText(
            f"S{draft.session_id} ↔ {reference}" f"{deformation}{suffix}"
        )
        self._position_options()

    def _populate_references(self, preferred_id=-1):
        previous = self.reference.blockSignals(True)
        try:
            self.reference.clear()
            self._references.clear()
            self._reference_session_ids.clear()

            def add(label, image, session_id=None):
                if image is None or tuple(image.shape) != self.dims:
                    return

                self.reference.addItem(label)
                self._references.append(self._normalize(image))
                self._reference_session_ids.append(session_id)

            add(
                "Current geometry of this session",
                self.draft.session.background,
            )

            for index, other in enumerate(self.data.sessions[: self.draft.session_id]):
                if not other.status["aligned"]:
                    continue

                label = f"Session {index}: {other.name}"
                if self.data.alignment_is_stale(index):
                    label += " (outdated alignment)"

                add(label, other.background, index)

            if not self._references:
                add("Original background", self.template)

            if preferred_id in self._reference_session_ids:
                index = self._reference_session_ids.index(preferred_id)
            else:
                index = self.reference.count() - 1

            self.reference.setCurrentIndex(index)
        finally:
            self.reference.blockSignals(previous)

    def set_draft(self, draft):
        if self._disposed or draft is self.draft:
            return

        self._end_correlation_drag()
        self.shift_inset.release_focus()

        index = self.reference.currentIndex()
        preferred_id = self._reference_session_ids[index] if index >= 0 else -1

        self._clear_hover(render=False)
        self._score_cache.clear()
        self._hover_draft = None

        self._cancel_score_task()
        self._score_key = None
        self._score_result = None
        self._displayed_score_key = None

        self.draft = draft
        self._inset_session_id = None

        self.template = draft.template
        self.dims = draft.dims

        # Keep the current camera range when changing sessions.
        self._fit_next_score = False

        height, width = self.dims
        for widget, limit in (
            (self.dx, 2 * width),
            (self.dy, 2 * height),
        ):
            previous = widget.blockSignals(True)
            widget.setRange(-limit, limit)
            widget.blockSignals(previous)

        self._populate_references(preferred_id)
        self._sync_from_draft()

    def _comparison_context(self):
        if self._hover_context is not None:
            return self._hover_context

        index = self.reference.currentIndex()
        return (
            self.draft,
            self._references[index],
            self._reference_session_ids[index],
        )

    def _pair_shift_context(self, owner_id):
        if not 0 <= owner_id < len(self.data.sessions):
            return None

        session = self.data.sessions[owner_id]

        if not session.status["spatial_loaded"] or session.background_template is None:
            return None

        active = session is self.draft.session

        if active and not self.draft.is_current(self.data):
            return None

        dimensions = tuple(session.dims)

        reference_ids = tuple(
            reference_id
            for reference_id, reference in enumerate(self.data.sessions[:owner_id])
            if (
                reference.status["aligned"]
                and not self.data.alignment_is_stale(reference_id)
                and reference.path is not None
                and reference.background_template is not None
                and reference.background is not None
                and tuple(reference.background.shape) == dimensions
            )
        )

        background_transposed = (
            bool(self.draft.background_transposed) if active else False
        )
        transpose = (
            bool(self.draft.transpose)
            if active
            else bool(getattr(session.remap, "transpose", False))
        )

        key = (
            self.state.data_version,
            owner_id,
            background_transposed,
            transpose,
            float(session.params["max_session_shift"]),
            reference_ids,
        )

        return key, reference_ids

    def _cancel_pair_shift_task(self):
        self._pair_shift_generation += 1

        task_id = self._pair_shift_task_id
        self._pair_shift_task_id = None
        self._pair_shift_key = None

        if task_id is not None:
            self.state.tasks.cancel(task_id)

    def _ensure_pair_shift_estimates(self, owner_id):
        context = self._pair_shift_context(owner_id)
        if context is None:
            return None

        key, reference_ids = context

        if key in self._pair_shift_cache:
            result = self._pair_shift_cache[key]
            self.pair_shift_status.setText(result["message"])
            return result

        if self._pair_shift_key == key:
            return None

        self._cancel_pair_shift_task()

        if not reference_ids:
            result = {
                "estimates": {},
                "message": "No eligible earlier reference sessions.",
            }
            self._pair_shift_cache[key] = result
            self.pair_shift_status.setText(result["message"])
            return result

        session = self.data.sessions[owner_id]
        active = session is self.draft.session

        template = np.asarray(
            self.draft.template if active else session.background_template,
            dtype=np.float32,
        ).copy()

        # Account for legacy remapping transposition separately from
        # the editable background-template orientation.
        if key[3]:
            template = template.T.copy()

        if tuple(template.shape) != tuple(session.dims):
            result = {
                "estimates": {},
                "message": (
                    "The oriented background does not match " "the session dimensions."
                ),
            }
            self._pair_shift_cache[key] = result
            self.pair_shift_status.setText(result["message"])
            return result

        # Snapshot worker inputs. No live session objects are accessed
        # by the calculation below.
        references = []

        for reference_id in reference_ids:
            reference = self.data.sessions[reference_id]
            references.append(
                (
                    reference_id,
                    str(reference.path),
                    reference.background_template.copy(),
                    deepcopy(reference.remap),
                )
            )

        max_shift = key[4]
        generation = self._pair_shift_generation
        self._pair_shift_key = key

        self.pair_shift_status.setText(
            f"Calculating independent flow comparisons for S{owner_id}…"
        )

        def calculate():
            ctx = current_task_context()
            estimates = {}
            rejected = []

            def check_cancelled():
                if ctx is not None:
                    ctx.check_cancelled()

            for reference_id, path, reference_template, reference_remap in references:
                check_cancelled()

                if ctx is not None:
                    ctx.message(
                        f"Comparing S{owner_id} independently "
                        f"against S{reference_id}…"
                    )

                try:
                    if reference_remap is None:
                        aligned_reference = reference_template
                        reference_valid = np.ones(
                            reference_template.shape,
                            dtype=bool,
                        )
                        reference_matrix = np.eye(3)
                    else:
                        aligned_reference = reference_remap.apply_remap(
                            reference_template,
                            use_optical_flow=True,
                        )
                        reference_valid = reference_remap.valid_mask(
                            use_optical_flow=True
                        )
                        reference_matrix = reference_remap.matrix

                    candidate = Remapping(
                        template=template,
                        references={
                            path: {
                                "template": reference_template,
                                "matrix": reference_matrix,
                                "aligned_template": aligned_reference,
                                "valid_mask": reference_valid,
                                "flow_candidate": True,
                            }
                        },
                        use_optical_flow=True,
                        max_shift=max_shift,
                    )

                    check_cancelled()

                    if not candidate.success:
                        raise ValueError(
                            candidate.flow_info.get("reason")
                            or "Flow comparison was rejected."
                        )

                    source_x, source_y = candidate.sampling_maps()
                    height, width = candidate.dims
                    yy, xx = np.indices(
                        candidate.dims,
                        dtype=np.float32,
                    )

                    valid = (
                        np.isfinite(source_x)
                        & np.isfinite(source_y)
                        & (source_x >= 0)
                        & (source_x <= width - 1)
                        & (source_y >= 0)
                        & (source_y <= height - 1)
                    )

                    if not valid.any():
                        raise ValueError(
                            "The estimated transformation has no valid pixels."
                        )

                    # Full transformation, including its residual flow.
                    # Sampling maps point output → source, hence this sign.
                    estimates[reference_id] = np.array(
                        [
                            np.mean(
                                xx[valid] - source_x[valid],
                                dtype=np.float64,
                            ),
                            np.mean(
                                yy[valid] - source_y[valid],
                                dtype=np.float64,
                            ),
                        ],
                        dtype=float,
                    )

                except TaskCancelled:
                    raise
                except Exception as exc:
                    rejected.append(f"S{reference_id}: {exc}")

            check_cancelled()

            message = (
                f"{len(estimates)} of {len(references)} "
                "independent flow comparisons accepted."
            )
            if rejected:
                message += "\nRejected comparisons:\n" + "\n".join(rejected)

            return {
                "estimates": estimates,
                "message": message,
            }

        self._pair_shift_task_id = self.state.tasks.start(
            "calculating",
            f"Flow comparison shifts: S{owner_id}",
            calculate,
            on_result=lambda result: self._on_pair_shifts_ready(
                generation, key, result
            ),
            background=True,
        )

        return None

    def _on_pair_shifts_ready(self, generation, key, result):
        if self._disposed or generation != self._pair_shift_generation:
            return

        self._pair_shift_task_id = None
        self._pair_shift_key = None

        context = self._pair_shift_context(key[1])
        if context is None or context[0] != key:
            return

        self._pair_shift_cache[key] = result

        # Only small summaries are cached, never the candidate flow arrays.
        while len(self._pair_shift_cache) > 4:
            oldest = next(iter(self._pair_shift_cache))
            del self._pair_shift_cache[oldest]

        if self._inset_session_id == key[1]:
            self.pair_shift_status.setText(result["message"])
            # Add the calculated arrows without fitting the global overview.
            self._refresh_shift_inset(fit=False)

            # Only zoom if the user has explicitly selected this session.
            # Hovering alone must not move the camera.
            if self.shift_inset.locked_session_id == key[1]:
                self.shift_inset.lock_session(key[1], force=True)

    def _on_pair_shift_stopped(self, group, task_id, *_):
        if self._disposed or task_id != self._pair_shift_task_id:
            return

        key = self._pair_shift_key
        self._pair_shift_task_id = None
        self._pair_shift_key = None

        result = {
            "estimates": {},
            "message": (
                "Comparison calculation stopped. "
                "Use Recalculate comparisons to retry."
            ),
        }

        # Prevent subsequent redraws from immediately restarting
        # a calculation the user just cancelled.
        if key is not None:
            self._pair_shift_cache[key] = result

        self.pair_shift_status.setText(result["message"])

    def _retry_pair_shifts(self):
        if self._disposed or self._inset_session_id is None:
            return

        self._cancel_pair_shift_task()
        self._pair_shift_cache.clear()
        self._refresh_shift_inset()

    def _on_pair_shift_data_changed(self, *_):
        if self._disposed:
            return

        self._cancel_pair_shift_task()
        self._pair_shift_cache.clear()
        self._refresh_shift_inset()

    def _clear_hover(self, *, render=True):
        self._hover_timer.stop()
        self._pending_hover = None
        self._hover_context = None
        self._inset_session_id = self.shift_inset.locked_session_id

        if render and not self._disposed:
            self._refresh_shift_inset()
            self._render_preview()
            self._update_availability()

    def _hover_inset(self, item):
        if self._disposed:
            return

        if item is None:
            self._clear_hover()
            return

        locked = self.shift_inset.locked_session_id
        if locked is not None and (
            item["kind"] != "reference" or item["owner"] != locked
        ):
            return

        self._pending_hover = (item["kind"], item["owner"], item["id"])
        self._hover_timer.start()

    def _show_hover(self):
        target = self._pending_hover
        if self._disposed or target is None:
            return

        kind, session_id, reference_id = target
        if not 0 <= session_id < len(self.data.sessions):
            self._clear_hover()
            return

        session = self.data.sessions[session_id]
        if not session.status["spatial_loaded"] or session.background_template is None:
            self._clear_hover()
            return

        if session is self.draft.session:
            draft = self.draft
        else:
            draft = self._hover_draft
            if (
                draft is None
                or draft.session is not session
                or not draft.is_current(self.data)
            ):
                # A local snapshot only; never installed in AppState.
                draft = AlignmentDraft(self.data, session_id)
                self._hover_draft = draft

        available = [
            index
            for index, other in enumerate(self.data.sessions[:session_id])
            if (
                other.status["aligned"]
                and other.background is not None
                and tuple(other.background.shape) == draft.dims
            )
        ]

        if kind == "session":
            if draft is self.draft:
                index = self.reference.currentIndex()
                reference_id = (
                    self._reference_session_ids[index]
                    if 0 <= index < len(self._reference_session_ids)
                    else None
                )
            else:
                reference_id = available[-1] if available else None

        elif reference_id not in available:
            return

        if reference_id is None:
            background = session.background
            if background is None or tuple(background.shape) != draft.dims:
                background = draft.template
        else:
            background = self.data.sessions[reference_id].background

        self._hover_context = (draft, self._normalize(background), reference_id)
        self._inset_session_id = session_id
        self._fit_next_score = False

        self._refresh_shift_inset()
        self._render_preview()
        self._update_availability()

    def _release_inset_focus(self):
        self.shift_inset.release_focus()
        self._clear_hover()

    def _on_reference_changed(self, *_):
        self._clear_hover()

    @staticmethod
    def _normalize(image):
        values = np.asarray(image, dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = float(values.min()), float(values.max())

        if hi <= lo:
            return np.zeros_like(values)

        return np.clip((values - lo) / (hi - lo), 0, 1)

    def _make_remap(self):
        return self.draft.make_remap()

    def _on_view_mode_changed(self, *_):
        correlation = self.view_mode.currentData() == "correlation"
        self._fit_next_score = correlation

        self.correlation_method.setEnabled(correlation)
        self.correlation_method.setVisible(correlation)
        self.options.form.labelForField(self.correlation_method).setVisible(correlation)
        self.score_status.setVisible(correlation)

        self.x_axis.axis.axis_label = "dx (px)" if correlation else "x (px)"
        self.y_axis.axis.axis_label = "dy (px)" if correlation else "y (px)"

        self.options.toggle_button.setToolTip(
            "Black: draggable total shift; magenta: pairwise peak."
            if correlation
            else "Green: reference; magenta: current session; white: overlap."
        )

        flow_mode = self.view_mode.currentData() == "flow"
        self.flow_scale.setVisible(flow_mode)
        self.options.form.labelForField(self.flow_scale).setVisible(flow_mode)
        self.flow_status.setVisible(flow_mode)

        self._render_preview()
        self._fit_view()
        self._position_options()

    @staticmethod
    def _warp_template(template, dims, transpose, dx, dy, angle, flow=None):
        remap = Remapping.identity(dims)
        remap.transpose = transpose
        remap.shift = np.array([dy, dx], dtype=float)
        remap.rotation = float(angle)
        remap.matrix = Remapping._rigid_matrix(dims, remap.shift, remap.rotation)
        remap.flow = flow

        return remap.apply_remap(template, use_optical_flow=True)

    def _render_preview(self, *_):
        if self._disposed:
            return

        flow_mode = self.view_mode.currentData() == "flow"
        if not flow_mode:
            self._clear_flow()

        reference_index = self.reference.currentIndex()
        if reference_index < 0:
            return

        draft, reference, reference_id = self._comparison_context()
        applied_flow = draft.effective_flow

        self.deformation_status.setText(
            draft.flow_message
            + (
                "\nCorrelation shows residual translation after the "
                "complete correction; zero is the current alignment."
                if applied_flow is not None
                else "\nCorrelation shows translation at the current rotation."
            )
        )
        dx, dy, angle = map(float, draft.values)
        correlation = self.view_mode.currentData() == "correlation"

        self._update_comparison_label(draft, reference_id)

        height, width = draft.dims
        self.view.camera.set_bounds(
            (
                -(width - 1) - 0.5,
                -(height - 1) - 0.5,
                reference.shape[1] - 0.5,
                reference.shape[0] - 0.5,
            )
            if correlation
            else (0, 0, width, height)
        )

        if not correlation:
            self.score_image.visible = False

            self._cancel_score_task()
            self._displayed_score_key = None

            moving = self._normalize(
                self._warp_template(
                    draft.template,
                    draft.dims,
                    draft.transpose,
                    dx,
                    dy,
                    angle,
                    flow=draft.effective_flow,
                )
            )

            rgb = np.empty(draft.dims + (3,), dtype=np.float32)
            rgb[..., 0] = moving
            rgb[..., 1] = reference
            rgb[..., 2] = moving

            self.image.transform = STTransform()
            self.image.set_data(rgb)
            self.image.visible = True
            self.shift_arrow.visible = False
            self.shift_tip.visible = False
            self.pair_arrow.visible = False
            self.pair_tip.visible = False

            if flow_mode:
                self._queue_flow(draft, reference, reference_id)

            self.canvas.update()
            return

        self.image.visible = False
        key = (
            (draft, reference_id, self.state.data_version),
            self.correlation_method.currentData(),
            angle,
            bool(draft.transpose),
            bool(draft.background_transposed),
            (
                (
                    tuple(map(float, draft.values[:2])),
                    id(applied_flow),
                )
                if applied_flow is not None
                else None
            ),
        )

        if self._requested_score_key is not None and self._requested_score_key != key:
            self._cancel_score_task()

        if key != self._score_key and key in self._score_cache:
            self._cancel_score_task()
            self._score_key = key
            self._score_result = self._score_cache.pop(key)
            self._score_cache[key] = self._score_result
            self._displayed_score_key = None

        if key != self._score_key:
            self.score_image.visible = False
            self.shift_arrow.visible = False
            self.shift_tip.visible = False
            self.pair_arrow.visible = False
            self.pair_tip.visible = False
            self._request_score_map(key, draft, reference)
            return

        result = self._score_result
        valid = result["scores"] is not None

        self.score_image.visible = valid
        self.shift_arrow.visible = valid
        self.shift_tip.visible = valid
        self.pair_arrow.visible = valid
        self.pair_tip.visible = valid
        if not valid:
            self.score_status.setText(result["message"])

        if valid:
            fit_needed = self._displayed_score_key != key

            if fit_needed:
                height, width = result["moving_shape"]

                # Image pixel centres are index + 0.5.
                # A correlation-map index becomes a shift after subtracting
                # moving_shape - 1.
                self.score_image.transform = STTransform(
                    translate=(
                        -(width - 1) - 0.5,
                        -(height - 1) - 0.5,
                    )
                )
                self.score_image.set_data(result["scores"])
                self._displayed_score_key = key
                self._color_window = None

            self._update_visible_clim()

            self._update_shift_marker(
                0.0 if applied_flow is not None else dx,
                0.0 if applied_flow is not None else dy,
            )
            if fit_needed and self._fit_next_score:
                self._fit_view()
                self._fit_next_score = False

        self.canvas.update()

    def _update_visible_clim(self, *_):
        if self._disposed or not self.score_image.visible:
            return

        result = self._score_result
        if result is None or result["scores"] is None:
            return

        scores = result["scores"]
        height, width = result["moving_shape"]

        # Use the actual visible region, including aspect-ratio expansion.
        transform = self.view.scene.node_transform(self.view)
        corners = np.asarray(transform.imap([[0, 0], self.view.size]))[:, :2]

        if not np.isfinite(corners).all():
            return

        lower = corners.min(axis=0) + (width - 1, height - 1)
        upper = corners.max(axis=0) + (width - 1, height - 1)

        x0, y0 = np.maximum(0, np.floor(lower)).astype(int)
        x1, y1 = np.minimum(
            (scores.shape[1], scores.shape[0]),
            np.ceil(upper) + 1,
        ).astype(int)

        window = (self._displayed_score_key, x0, y0, x1, y1)
        if window == self._color_window:
            return
        self._color_window = window

        if x1 <= x0 or y1 <= y0:
            self.score_status.setText(result["message"] + "\nNo scores in view.")
            return

        # Bound the work during pan/zoom to approximately 65k samples.
        stride = max(
            1,
            int(np.ceil(np.sqrt((x1 - x0) * (y1 - y0) / 65536))),
        )
        values = scores[y0:y1:stride, x0:x1:stride]
        values = values[np.isfinite(values)]

        if not values.size:
            self.score_status.setText(result["message"] + "\nNo valid scores in view.")
            return

        low, high = map(float, np.percentile(values, (2, 99.5)))

        if high <= low:
            low, high = float(values.min()), float(values.max())
        if high <= low:
            padding = max(1e-6, abs(low) * 1e-6)
            low, high = low - padding, high + padding

        self.score_image.clim = (low, high)
        self.score_status.setText(
            result["message"] + f"\nVisible color range: {low:.5g} to {high:.5g}"
        )

    def _cancel_score_task(self):
        task_id = self._score_task_id

        self._score_generation += 1
        self._score_task_id = None
        self._requested_score_key = None

        if task_id is not None:
            self.state.tasks.cancel(task_id)

    def _request_score_map(self, key, draft, reference):
        if key == self._requested_score_key:
            return

        self._cancel_score_task()
        self._requested_score_key = key
        generation = self._score_generation

        _, method, angle, transpose, _background_transposed = key[:5]

        reference = reference.copy()
        template = draft.template.copy()
        dims = draft.dims

        flow = draft.effective_flow
        flow = None if flow is None else flow.copy()

        dx, dy = map(float, draft.values[:2])
        residual_surface = flow is not None

        self.score_status.setText("Calculating correlation surface…")

        def calculate():
            ctx = current_task_context()

            try:
                if ctx is not None:
                    ctx.check_cancelled()

                moving = ManualAlignmentEditor._normalize(
                    ManualAlignmentEditor._warp_template(
                        template,
                        dims,
                        transpose,
                        dx if residual_surface else 0.0,
                        dy if residual_surface else 0.0,
                        angle,
                        flow=flow,
                    )
                )

                scores = calculate_shift_score_map(
                    reference,
                    moving,
                    mode=method,
                    min_overlap=0.25,
                )

                if ctx is not None:
                    ctx.check_cancelled()

                finite = np.isfinite(scores)
                if not finite.any():
                    return {
                        "scores": None,
                        "message": (
                            "No valid scores for this comparison. "
                            "Check that both backgrounds contain structure."
                        ),
                    }

                low = float(scores[finite].min())
                high = float(scores[finite].max())

                iy, ix = np.unravel_index(
                    np.argmax(np.where(finite, scores, -np.inf)), scores.shape
                )
                peak_dy = iy - (moving.shape[0] - 1)
                peak_dx = ix - (moving.shape[1] - 1)

                if ctx is not None:
                    ctx.check_cancelled()

                return {
                    "scores": np.where(finite, scores, np.nan).astype(np.float32),
                    "moving_shape": moving.shape,
                    "peak": (float(peak_dx), float(peak_dy)),
                    "message": (
                        (
                            "Residual translation after flow correction; "
                            "(0, 0) is the current alignment.\n"
                            if residual_surface
                            else "Translation at the current rotation.\n"
                        )
                        + f"Global score range: {low:.3g} to {high:.3g}\n"
                        + f"Grid peak: dx={peak_dx}, dy={peak_dy} px"
                    ),
                }

            except TaskCancelled:
                raise

            except Exception as exc:
                return {
                    "scores": None,
                    "message": f"Comparison failed: {type(exc).__name__}: {exc}",
                }

        self._score_task_id = self.state.tasks.start(
            "calculating",
            f"Alignment correlation: session {draft.session_id}",
            calculate,
            on_result=lambda result: self._on_score_ready(generation, key, result),
            background=True,
        )

    def _on_score_ready(self, generation, key, result):
        if (
            self._disposed
            or generation != self._score_generation
            or key != self._requested_score_key
        ):
            return

        self._score_task_id = None
        self._requested_score_key = None

        # Bounded cache: at most two correlation surfaces.
        self._score_cache.pop(key, None)
        self._score_cache[key] = result
        while len(self._score_cache) > 2:
            self._score_cache.pop(next(iter(self._score_cache)))

        self._score_key = key
        self._score_result = result
        self._displayed_score_key = None

        self._render_preview()

    def _on_score_task_stopped(self, group, task_id, *_):
        if self._disposed or task_id != self._score_task_id:
            return

        self._score_task_id = None
        self._requested_score_key = None
        self.score_status.setText(
            "Calculation stopped. Change the method or display mode to retry."
        )

    def _clear_flow(self):
        self._flow_timer.stop()
        self._flow_generation += 1
        self._flow_pending = None
        self._flow_key = None
        self._flow_result = None
        self.flow_arrows.visible = False

        task_id = self._flow_task_id
        self._flow_task_id = None
        if task_id is not None:
            self.state.tasks.cancel(task_id)

    def _queue_flow(self, draft, reference, reference_id):
        key = (
            draft,
            reference_id,
            self.state.data_version,
            tuple(map(float, draft.values)),
            bool(draft.transpose),
            bool(draft.background_transposed),
            id(draft.effective_flow),
        )

        if key == self._flow_key:
            self._draw_flow()
            return

        self._clear_flow()
        self._flow_key = key

        # Snapshot inputs on the GUI thread.
        self._flow_pending = (
            reference.copy(),
            draft.template.copy(),
            tuple(draft.dims),
            bool(draft.transpose),
            tuple(map(float, draft.values)),
            None if draft.effective_flow is None else draft.effective_flow.copy(),
        )
        self.flow_status.setText("Waiting for alignment changes to settle…")
        self._flow_timer.start()

    def _start_flow(self):
        pending = self._flow_pending
        if self._disposed or pending is None:
            return

        self._flow_pending = None
        generation = self._flow_generation
        reference, template, dims, transpose, values, applied_flow = pending
        dx, dy, angle = values

        self.flow_status.setText("Estimating residual flow…")

        def calculate():
            ctx = current_task_context()

            try:
                if ctx is not None:
                    ctx.check_cancelled()

                moving = ManualAlignmentEditor._warp_template(
                    template,
                    dims,
                    transpose,
                    dx,
                    dy,
                    angle,
                    flow=applied_flow,
                )

                # Identify actual overlap, excluding warp padding.
                coverage = ManualAlignmentEditor._warp_template(
                    np.ones_like(template, dtype=np.float32),
                    dims,
                    transpose,
                    dx,
                    dy,
                    angle,
                    flow=applied_flow,
                )

                flow = calculate_residual_flow(reference, moving)

                if ctx is not None:
                    ctx.check_cancelled()

                # Sampling validity uses the real flow, never its display scale.
                yy, xx = np.indices(dims, dtype=np.float32)
                destination_x = xx + flow[..., 0]
                destination_y = yy + flow[..., 1]

                valid = (
                    (coverage > 0.999)
                    & np.isfinite(flow).all(axis=-1)
                    & (destination_x >= 0)
                    & (destination_x <= dims[1] - 1)
                    & (destination_y >= 0)
                    & (destination_y <= dims[0] - 1)
                )

                # Check coverage at the corresponding moving-image location.
                ix = np.clip(
                    np.rint(np.nan_to_num(destination_x)),
                    0,
                    dims[1] - 1,
                ).astype(np.intp)
                iy = np.clip(
                    np.rint(np.nan_to_num(destination_y)),
                    0,
                    dims[0] - 1,
                ).astype(np.intp)
                valid &= coverage[iy, ix] > 0.999

                flow = np.asarray(flow, dtype=np.float32)
                flow[~valid] = np.nan

                magnitude = np.linalg.norm(flow[valid], axis=-1)
                if magnitude.size == 0:
                    raise ValueError("No valid overlapping region for flow.")

                return {
                    "flow": flow,
                    "median": float(np.median(magnitude)),
                    "p95": float(np.percentile(magnitude, 95)),
                }

            except TaskCancelled:
                raise
            except Exception as exc:
                return {"error": f"{type(exc).__name__}: {exc}"}

        self._flow_task_id = self.state.tasks.start(
            "calculating",
            "Estimating comparison flow",
            calculate,
            on_result=lambda result: self._on_flow_ready(generation, result),
            background=True,
        )

    def _on_flow_ready(self, generation, result):
        if self._disposed or generation != self._flow_generation:
            return

        self._flow_task_id = None
        self._flow_result = result
        self._draw_flow()

    def _draw_flow(self):
        if self._disposed:
            return

        self.flow_arrows.visible = False
        result = self._flow_result

        if self.view_mode.currentData() != "flow" or result is None:
            return

        if "error" in result:
            self.flow_status.setText(
                f"Flow unavailable: {result['error']}\n"
                "Check the backgrounds or adjust the alignment. "
                "Switch display modes to retry."
            )
            self.canvas.update()
            return

        flow = result["flow"]
        height, width = flow.shape[:2]

        # Roughly 32 arrows across a 512-pixel image.
        spacing = max(8, int(round(min(height, width) / 32)))
        yy, xx = np.mgrid[
            spacing // 2 : height : spacing,
            spacing // 2 : width : spacing,
        ]

        vectors = flow[yy, xx].reshape(-1, 2)
        starts = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
        valid = np.isfinite(vectors).all(axis=1)
        vectors = vectors[valid]
        starts = starts[valid]

        delta = vectors * float(self.flow_scale.value())
        lengths = np.linalg.norm(delta, axis=1)

        # Avoid tiny, directionless arrowheads.
        visible = lengths >= 0.5
        starts = starts[visible]
        delta = delta[visible]
        lengths = lengths[visible]

        if lengths.size:
            ends = starts + delta
            direction = delta / lengths[:, None]
            perpendicular = np.column_stack((-direction[:, 1], direction[:, 0]))

            head_length = np.minimum(5.0, lengths * 0.35)
            back = ends - direction * head_length[:, None]
            wing = perpendicular * (head_length * 0.5)[:, None]

            segments = np.stack(
                (
                    starts,
                    ends,
                    ends,
                    back + wing,
                    ends,
                    back - wing,
                ),
                axis=1,
            ).reshape(-1, 2)

            self.flow_arrows.set_data(
                pos=segments.astype(np.float32),
                connect="segments",
            )
            self.flow_arrows.visible = True

        self.flow_status.setText(
            "Residual flow: reference → current session\n"
            f"Median: {result['median']:.3f} px · "
            f"95th percentile: {result['p95']:.3f} px\n"
            f"Arrows magnified {self.flow_scale.value():g}×; "
            "flow is not applied."
        )
        self.canvas.update()

    def _on_flow_task_stopped(self, group, task_id, *_):
        if self._disposed or task_id != self._flow_task_id:
            return

        self._flow_task_id = None
        self.flow_status.setText(
            "Flow calculation stopped. Switch display modes to retry."
        )

    @staticmethod
    def _shift_arrow_positions(dx, dy):
        tip = np.array([dx, dy], dtype=np.float32)
        length = float(np.linalg.norm(tip))

        if length == 0:
            return np.zeros((6, 2), dtype=np.float32)

        direction = tip / length
        normal = np.array([-direction[1], direction[0]])
        head = min(1.5, length * 0.2)
        base = tip - head * direction

        return np.asarray(
            [
                [0, 0],
                tip,
                tip,
                base + 0.5 * head * normal,
                tip,
                base - 0.5 * head * normal,
            ],
            dtype=np.float32,
        )

    def _update_shift_marker(self, dx, dy):
        self.shift_arrow.set_data(
            pos=self._shift_arrow_positions(dx, dy),
            connect="segments",
        )
        self.shift_tip.set_data(
            pos=np.array([[dx, dy]], dtype=np.float32),
            face_color="#ffffff",
            edge_color="#15191f",
            edge_width=1,
            size=6,
        )

        peak_dx, peak_dy = self._score_result["peak"]
        self.pair_arrow.set_data(
            pos=self._shift_arrow_positions(peak_dx, peak_dy),
            connect="segments",
        )
        self.pair_tip.set_data(
            pos=np.array([[peak_dx, peak_dy]], dtype=np.float32),
            face_color="#ed64bc",
            edge_color="#ffffff",
            edge_width=1,
            size=6,
        )

    def _fit_view(self):

        draft, _, _ = self._comparison_context()

        if self.view_mode.currentData() == "correlation":
            points = [(0.0, 0.0)]

            if draft.effective_flow is None:
                points.append(tuple(map(float, draft.values[:2])))

            result = self._score_result
            if (
                result is not None
                and result["scores"] is not None
                and self._displayed_score_key == self._score_key
            ):
                points.append(result["peak"])

            points = np.asarray(points, dtype=float)
            lower = points.min(axis=0) - 20.0
            upper = points.max(axis=0) + 20.0

            x_range = (lower[0], upper[0])
            y_range = (lower[1], upper[1])
        else:
            height, width = draft.dims
            x_range = (0, width)
            y_range = (0, height)

        self.view.camera.set_range(
            x=x_range,
            y=y_range,
            z=(-1, 1),
            margin=0,
        )

    def can_apply(self):
        tasks = self.state.tasks
        busy = tasks.processing_busy() or tasks.processing_requested
        return (
            not self._disposed
            and self.state.alignment_draft is self.draft
            and self.draft.is_current(self.data)
            and not busy
        )

    def _update_availability(self, *_):
        if self._disposed:
            return

        previewing_other = (
            self._hover_context is not None and self._hover_context[0] is not self.draft
        )

        self.shift_inset.editable = self.can_apply()
        available = self.can_apply() and not previewing_other

        can_estimate = available and self._estimate_task_id is None
        self.estimate_rotation_button.setEnabled(can_estimate)
        self.estimate_shift_button.setEnabled(can_estimate)

        _, _, reference_id = self._comparison_context()
        self.refine_flow_button.setEnabled(
            can_estimate
            and reference_id is not None
            and not self.data.alignment_is_stale(reference_id)
        )

        for widget in (
            self.dx,
            self.dy,
            self.edit_rotation,
            self.reset_button,
            self.apply_button,
        ):
            widget.setEnabled(available)

        self.transpose_background.setEnabled(
            available and self.draft.can_transpose_background
        )

        self.angle.setEnabled(available and self.edit_rotation.isChecked())

        if self.state.alignment_draft is not self.draft:
            message = "This draft is no longer active."
        elif not self.draft.is_current(self.data):
            message = "Data changed. Reset to start a fresh draft."
        elif previewing_other:
            message = (
                f"Previewing session {self._hover_context[0].session_id}. "
                f"Draft remains session {self.draft.session_id}."
            )
        elif not available:
            message = "Waiting for background tasks to finish."
        else:
            message = (
                f"Editing session {self.draft.session_id}: "
                f"{self.draft.session.name}"
            )

        displayed_draft, _, _ = self._comparison_context()

        self.status_label.setToolTip(
            message
            + (
                "\nOptical-flow deformation is included in the preview. "
                "Main session arrows show the rigid component; "
                "comparison arrows summarize relative mean displacement."
                if displayed_draft.effective_flow is not None
                else "\nThe preview currently uses rigid geometry."
            )
        )
        self.apply_button.setToolTip(message)

        tasks = self.state.tasks
        idle = not (tasks.processing_busy() or tasks.processing_requested)
        self.reset_button.setEnabled(idle and self.state.alignment_draft is self.draft)
        self.popout_button.setEnabled(idle)

        review = self.state.alignment_review
        reviewing = review is not None and review["session"] is self.draft.session

        needs_acceptance = reviewing or not self.draft.session.status["aligned"]
        self.apply_button.setText(
            "Use alignment" if needs_acceptance else "Apply changes"
        )

        blocker = self._blocking_alignment()
        messages = []

        if blocker is not None:
            session = self.data.sessions[blocker]
            messages.append(
                f"Cannot accept S{self.draft.session_id} yet: "
                f"S{blocker} ({session.name}) has an outdated alignment.\n"
                f"Review and apply S{blocker} first, then return here."
            )

        if reviewing:
            messages.append(review["message"])

        text = "\n\n".join(messages)
        if self.review_message.toPlainText() != text:
            self.review_message.setPlainText(text)
        self.review_message.setVisible(bool(text))

        self.blocker_button.setVisible(blocker is not None)
        self.blocker_button.setEnabled(idle)
        if blocker is not None:
            self.blocker_button.setText(f"Review blocking S{blocker}")

        # Editing remains possible; committing waits for upstream geometry.
        self.apply_button.setEnabled(available and blocker is None)
        if blocker is not None:
            self.apply_button.setToolTip(
                f"Review and apply the outdated alignment of S{blocker} first."
            )

        self.remove_button.setVisible(needs_acceptance)
        self.remove_button.setEnabled(available)
        self.later_button.setVisible(reviewing)
        self.later_button.setEnabled(idle)

    def _review_later(self):
        self.state.alignment_review_finished.emit(self.draft.session)
        self.review_closed.emit()

    def _remove_session(self):
        if not self.can_apply():
            return

        session = self.draft.session
        answer = QMessageBox.question(
            self,
            "Remove session?",
            f"Remove {session.name!r} from this CATAN project?\n\n"
            "Its original data files will remain unchanged.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        # A modal confirmation can process other queued GUI events.
        if not self.can_apply():
            return

        session_id = next(
            (index for index, item in enumerate(self.data.sessions) if item is session),
            None,
        )
        if session_id is None:
            return

        self._cancel_score_task()
        self.data.remove_session(session_id)
        self.review_closed.emit()

    def _set_parameter(self, index, value):
        if not self.can_apply():
            return

        self.draft.values[index] = value
        self.state.alignment_draft_changed.emit()

    def _sync_from_draft(self):
        if self._disposed:
            return

        if self.state.alignment_draft is self.draft:
            self._clear_hover(render=False)
            for widget, value in zip(
                (self.dx, self.dy, self.angle),
                self.draft.values,
            ):
                previous = widget.blockSignals(True)
                widget.setValue(float(value))
                widget.blockSignals(previous)

            self.template = self.draft.template

            previous = self.transpose_background.blockSignals(True)
            self.transpose_background.setChecked(self.draft.background_transposed)
            self.transpose_background.blockSignals(previous)

            self._render_preview()
            self._refresh_shift_inset()

        self._update_availability()

    def _reset(self):
        tasks = self.state.tasks
        if (
            self.state.alignment_draft is not self.draft
            or tasks.processing_busy()
            or tasks.processing_requested
        ):
            return

        self.draft.clear_refined_flow()

        self._end_correlation_drag()
        self.shift_inset.release_focus()
        self._clear_hover(render=False)

        if self.can_apply():
            self._cancel_score_task()
            self.draft.values[:] = self.draft.initial
            self.draft.background_transposed = False
            self.state.alignment_draft_changed.emit()

        if self.draft.is_current(self.data):
            self.draft.values[:] = self.draft.initial
        else:
            self.state.alignment_draft = AlignmentDraft(
                self.data, self.draft.session_id
            )

        self._fit_next_score = self.view_mode.currentData() == "correlation"
        self.state.alignment_draft_changed.emit()
        self._refresh_shift_inset(fit=True)
        self._fit_view()

    def _apply(self):
        if self.can_apply() and self._blocking_alignment() is None:
            self.apply_requested.emit()

    def dispose(self):
        if self._disposed:
            return

        self._disposed = True
        self._estimate_generation += 1
        task_id = self._estimate_task_id
        self._estimate_task_id = None
        if task_id is not None:
            self.state.tasks.cancel(task_id)
        self._clear_flow()

        self._hover_timer.stop()
        self._hover_context = self._hover_draft = None
        self._score_cache.clear()

        self._cancel_score_task()
        self._score_result = None
        for signal, slot in self._connections:
            signal.disconnect(slot)

        # The controls may have been reparented into the panel toolbar.
        self.view_controls.hide()
        self.view_controls.deleteLater()

        self._end_correlation_drag()
        for signal, slot in self._correlation_mouse_connections:
            signal.disconnect(slot)

        self.shift_inset.dispose()
        self._cancel_pair_shift_task()
        self._pair_shift_cache.clear()

        self.options.overlay_layout_changed.disconnect(self._position_options)
        self.canvas.events.resize.disconnect(self._position_options)
        self.canvas.events.draw.disconnect(self._update_visible_clim)

        self.canvas.close()

    def _session_mean_displacement(self, session_id):
        """Mean (dx, dy) of the complete stored or draft transformation."""
        session = self.data.sessions[session_id]
        active = session is self.draft.session

        revision = self.data.state.data_version
        if getattr(self, "_shift_mean_revision", None) != revision:
            self._shift_mean_revision = revision
            self._shift_mean_cache = {}

        key = (
            session_id,
            id(session),
            id(session.remap),
            tuple(session.dims),
            (
                (
                    id(self.draft),
                    tuple(float(v) for v in self.draft.values),
                    bool(self.draft.background_transposed),
                )
                if active
                else None
            ),
            id(self.draft.effective_flow) if active else None,
        )

        if key in self._shift_mean_cache:
            return self._shift_mean_cache[key]

        remap = self.draft.make_remap() if active else session.remap

        if remap is None or not remap.success:
            return None

        source_x, source_y = remap.sampling_maps()
        height, width = session.dims

        valid = (
            np.isfinite(source_x)
            & np.isfinite(source_y)
            & (source_x >= 0)
            & (source_x <= width - 1)
            & (source_y >= 0)
            & (source_y <= height - 1)
        )

        if not valid.any():
            return None

        yy, xx = np.indices(
            (height, width),
            dtype=np.float32,
        )

        # sampling_maps maps output coordinates back to source coordinates.
        # Reverse that displacement to retain the UI's correction direction.
        displacement = np.array(
            [
                np.mean(
                    xx[valid] - source_x[valid],
                    dtype=np.float64,
                ),
                np.mean(
                    yy[valid] - source_y[valid],
                    dtype=np.float64,
                ),
            ],
            dtype=float,
        )

        # Retain only the current summary for this session.
        # Never cache the full sampling maps here.
        self._shift_mean_cache = {
            old_key: value
            for old_key, value in self._shift_mean_cache.items()
            if old_key[0] != session_id
        }
        self._shift_mean_cache[key] = displacement

        return displacement

    def _refresh_shift_inset(self, *, fit=False):
        items = []

        for session_id, session in enumerate(self.data.sessions):
            if (
                not session.status["spatial_loaded"]
                or session.background_template is None
            ):
                continue

            active = session is self.draft.session

            if not active:
                remap = session.remap
                if remap is None or not remap.success:
                    continue

            xy = self._session_mean_displacement(session_id)

            if xy is None:
                continue

            if not np.isfinite(xy).all():
                continue

            if "session" in session.name.lower():
                label = session.name
            else:
                label = f"Session {session_id}: {session.name}"
            if self.data.alignment_is_stale(session_id):
                label += " (outdated)"

            items.append(
                {
                    "kind": "session",
                    "id": session_id,
                    "xy": xy,
                    "rigid_xy": (self.draft.values[:2].copy() if active else xy.copy()),
                    "label": label,
                    "active": active,
                    "owner": session_id,
                    "focused": (
                        self._inset_session_id is None
                        or session_id == self._inset_session_id
                    ),
                }
            )

        focus = next(
            (item for item in items if item["id"] == self._inset_session_id),
            None,
        )
        if focus is None:
            self.shift_inset.set_items(items, fit=fit)
            return

        owner_id = self._inset_session_id
        result = self._ensure_pair_shift_estimates(owner_id)

        if result is not None:
            for reference_id, candidate_xy in result["estimates"].items():
                candidate_xy = np.asarray(candidate_xy, dtype=float)

                if not np.isfinite(candidate_xy).all():
                    continue

                difference = candidate_xy - focus["xy"]

                items.append(
                    {
                        "kind": "reference",
                        "id": reference_id,
                        "owner": owner_id,
                        "start": focus["xy"].copy(),
                        "xy": candidate_xy.copy(),
                        "label": (
                            f"S{owner_id} aligned only against S{reference_id}: "
                            f"dx={candidate_xy[0]:+.2f}, "
                            f"dy={candidate_xy[1]:+.2f} px; "
                            f"difference from current mean "
                            f"({difference[0]:+.2f}, {difference[1]:+.2f}) px"
                        ),
                        "active": False,
                        "focused": True,
                    }
                )

        self.shift_inset.set_items(items, fit=fit)

    def _choose_inset_item(self, kind, session_id):

        if kind == "session" and self.shift_inset.locked_session_id == session_id:
            self._release_inset_focus()
            return

        target_id = session_id if kind == "session" else self._inset_session_id

        index = self.reference.currentIndex()
        selected_reference = (
            self._reference_session_ids[index]
            if 0 <= index < len(self._reference_session_ids)
            else None
        )

        reference_id = selected_reference

        if kind == "reference":
            reference_id = session_id

        elif target_id != self.draft.session_id and self._hover_context is not None:
            preview, _, preview_reference = self._hover_context
            if preview.session_id == target_id:
                reference_id = preview_reference

        self._clear_hover(render=False)

        draft = select_alignment_draft(self.data, target_id, self)
        if draft is None:
            self._clear_hover()
            return

        self.set_draft(draft)

        if reference_id in self._reference_session_ids:
            previous = self.reference.blockSignals(True)
            self.reference.setCurrentIndex(
                self._reference_session_ids.index(reference_id)
            )
            self.reference.blockSignals(previous)

        self.state.current_session_id = target_id
        self._inset_session_id = target_id
        self._refresh_shift_inset()
        self.shift_inset.lock_session(target_id)
        self._clear_hover()

    def _set_inset_shift(self, dx, dy):
        if not self.can_apply():
            return

        self.draft.values[:2] = (
            np.clip(dx, self.dx.minimum(), self.dx.maximum()),
            np.clip(dy, self.dy.minimum(), self.dy.maximum()),
        )
        self.state.alignment_draft_changed.emit()

    def _on_correlation_press(self, event):
        if (
            event.button != 1
            or self.view_mode.currentData() != "correlation"
            or not self.score_image.visible
            or not self.can_apply()
        ):
            return

        displayed_draft, _, _ = self._comparison_context()
        if displayed_draft is not self.draft:
            # Hover previews of other sessions remain read-only.
            return
        if displayed_draft.effective_flow is not None:
            # This surface describes residual displacement, not the
            # absolute rigid coordinates edited by this drag handler.
            return

        transform = self.view.scene.node_transform(self.canvas.scene)
        start = np.asarray(transform.map((0, 0))[:2])
        tip = np.asarray(transform.map(self.draft.values[:2])[:2])
        position = np.asarray(event.pos, dtype=float)

        # Accept either the endpoint or the shaft, away from the origin.
        hit = np.linalg.norm(position - tip) <= 10
        vector = tip - start
        length_squared = float(vector @ vector)

        if not hit and length_squared > 0:
            fraction = float((position - start) @ vector / length_squared)
            closest = start + np.clip(fraction, 0, 1) * vector
            hit = 0.15 <= fraction <= 1.0 and np.linalg.norm(position - closest) <= 6

        if not hit:
            return

        self._hover_timer.stop()
        self._correlation_drag = (
            self.draft,
            np.asarray(transform.imap(position)[:2]),
            self.draft.values[:2].copy(),
            self.view.camera.interactive,
        )
        self.view.camera.interactive = False
        event.handled = True

    def _on_correlation_move(self, event):
        if self._correlation_drag is None:
            return

        event.handled = True
        draft, start_point, start_shift, _ = self._correlation_drag

        if draft is not self.draft or not self.can_apply():
            self._end_correlation_drag()
            return

        transform = self.view.scene.node_transform(self.canvas.scene)
        point = np.asarray(transform.imap(event.pos)[:2])
        shift = start_shift + point - start_point

        # Existing shared setter: clamps values and updates both views.
        self._set_inset_shift(float(shift[0]), float(shift[1]))

    def _on_correlation_release(self, event):
        if event.button == 1 and self._correlation_drag is not None:
            event.handled = True
            self._end_correlation_drag()

    def _end_correlation_drag(self):
        if self._correlation_drag is None:
            return

        camera_interactive = self._correlation_drag[3]
        self._correlation_drag = None
        self.view.camera.interactive = camera_interactive

    def _blocking_alignment(self):
        return next(
            (
                sid
                for sid in range(self.draft.session_id)
                if self.data.alignment_is_stale(sid)
            ),
            None,
        )

    def _open_blocking_alignment(self):
        sid = self._blocking_alignment()
        if sid is None:
            return

        draft = select_alignment_draft(self.data, sid, self)
        if draft is not None:
            self.set_draft(draft)

    def _estimate_context(self):
        draft, _, reference_id = self._comparison_context()
        return (
            draft,
            reference_id,
            self.state.data_version,
            tuple(map(float, draft.values)),
            bool(draft.transpose),
            bool(draft.background_transposed),
        )

    def _estimate_alignment(self, kind):
        if not self.can_apply() or self._estimate_task_id is not None:
            return

        draft, reference, reference_id = self._comparison_context()
        if draft is not self.draft:
            return

        context = self._estimate_context()
        reference = np.asarray(reference, dtype=np.float64).copy()
        template = self._normalize(draft.template).copy()
        dims = tuple(draft.dims)
        transpose = bool(draft.transpose)
        dx, dy, angle = map(float, draft.values)

        candidate = None
        reference_remap = None
        reference_path = None

        if kind == "flow":
            if reference_id is None or self.data.alignment_is_stale(reference_id):
                return

            reference_session = self.data.sessions[reference_id]
            reference_path = str(reference_session.path)
            reference_remap = deepcopy(reference_session.remap)

            candidate = draft.make_remap()

            # Re-estimate the complete residual correction at the
            # current rigid geometry; do not stack it onto old flow.
            candidate.flow = None
            candidate.flow_info = {}

        radius = float(draft.session.params.get("max_session_rotation", 10.0))
        coarse_step = float(draft.session.params.get("rotation_step", 1.0))
        fine_step = float(draft.session.params.get("rotation_refine_step", 0.1))

        angle_limits = (self.angle.minimum(), self.angle.maximum())
        x_limits = (self.dx.minimum(), self.dx.maximum())
        y_limits = (self.dy.minimum(), self.dy.maximum())

        self._estimate_generation += 1
        generation = self._estimate_generation
        self.estimate_status.setText(f"Estimating {kind}…")

        def calculate():
            ctx = current_task_context()

            def check_cancelled():
                if ctx is not None:
                    ctx.check_cancelled()

            try:
                check_cancelled()

                if (
                    not np.isfinite(reference).all()
                    or not np.isfinite(template).all()
                    or np.ptp(reference) == 0
                    or np.ptp(template) == 0
                ):
                    raise ValueError(
                        "Both backgrounds need finite, nonconstant image data."
                    )

                if kind == "flow":
                    valid_mask = (
                        np.ones(dims, dtype=bool)
                        if reference_remap is None
                        else reference_remap.valid_mask()
                    )

                    candidate.estimate_flow(
                        template,
                        {
                            reference_path: {
                                "aligned_template": reference,
                                "valid_mask": valid_mask,
                                "flow_candidate": True,
                            }
                        },
                        residual=True,
                    )

                    check_cancelled()

                    if candidate.flow is None:
                        raise ValueError(
                            candidate.flow_info.get("reason")
                            or "No acceptable flow correction was found."
                        )

                    return {
                        "kind": "flow",
                        "remap": candidate,
                        "message": (
                            "Flow refinement is ready for inspection. "
                            "Apply changes to commit it."
                        ),
                    }

                if kind == "shift":
                    # Keep rotation; search translations from zero shift.
                    moving = ManualAlignmentEditor._warp_template(
                        template, dims, transpose, 0.0, 0.0, angle
                    )

                    scores = calculate_shift_score_map(
                        reference,
                        moving,
                        mode="cosine",
                        min_overlap=0.25,
                    )
                    check_cancelled()

                    xs = np.arange(scores.shape[1]) - (moving.shape[1] - 1)
                    ys = np.arange(scores.shape[0]) - (moving.shape[0] - 1)

                    valid = (
                        np.isfinite(scores)
                        & (xs[None, :] >= x_limits[0])
                        & (xs[None, :] <= x_limits[1])
                        & (ys[:, None] >= y_limits[0])
                        & (ys[:, None] <= y_limits[1])
                    )
                    if not valid.any():
                        raise ValueError("No valid translation could be estimated.")

                    best = float(np.max(scores[valid]))

                    # Resolve numerically equivalent peaks toward the current shift.
                    candidates = np.argwhere(
                        valid & np.isclose(scores, best, rtol=0, atol=1e-10)
                    )
                    distance = (xs[candidates[:, 1]] - dx) ** 2 + (
                        ys[candidates[:, 0]] - dy
                    ) ** 2
                    iy, ix = candidates[np.argmin(distance)]

                    return {
                        "kind": kind,
                        "values": (float(xs[ix]), float(ys[iy])),
                        "message": (
                            f"Estimated shift: dx={xs[ix]:g}, dy={ys[iy]:g} px. "
                            "Rotation retained; preview only."
                        ),
                    }

                if radius <= 0 or coarse_step <= 0 or fine_step <= 0:
                    raise ValueError(
                        "Rotation search range and steps must be positive."
                    )

                lo = max(angle_limits[0], angle - radius)
                hi = min(angle_limits[1], angle + radius)
                reference_norm = np.linalg.norm(reference)
                scores = {}

                def score_rotation(candidate):
                    candidate = float(candidate)
                    if candidate in scores:
                        return scores[candidate]

                    check_cancelled()

                    rotated = ManualAlignmentEditor._warp_template(
                        template, dims, transpose, 0.0, 0.0, candidate
                    )
                    moving = ManualAlignmentEditor._warp_template(
                        template, dims, transpose, dx, dy, candidate
                    )
                    coverage = ManualAlignmentEditor._warp_template(
                        np.ones_like(template),
                        dims,
                        transpose,
                        dx,
                        dy,
                        candidate,
                    )

                    # Reject candidates with too little usable overlap.
                    if np.count_nonzero(coverage > 0.999) < 0.25 * reference.size:
                        value = -np.inf
                    else:
                        # Use the pre-translation norm so losing image content
                        # at the boundary does not improve normalization.
                        denominator = reference_norm * np.linalg.norm(rotated)
                        value = (
                            float(np.sum(reference * moving) / denominator)
                            if denominator > 0
                            else -np.inf
                        )

                    scores[candidate] = value
                    return value

                def grid(start, stop, step):
                    count = max(1, int(np.ceil((stop - start) / step)))
                    return np.linspace(start, stop, count + 1)

                best_angle = angle
                best_score = score_rotation(angle)

                for candidate in grid(lo, hi, coarse_step):
                    value = score_rotation(candidate)
                    if value > best_score + 1e-10:
                        best_angle, best_score = float(candidate), value

                fine_lo = max(lo, best_angle - coarse_step)
                fine_hi = min(hi, best_angle + coarse_step)

                for candidate in grid(fine_lo, fine_hi, fine_step):
                    value = score_rotation(candidate)
                    if value > best_score + 1e-10:
                        best_angle, best_score = float(candidate), value

                if not np.isfinite(best_score):
                    raise ValueError(
                        "Insufficient overlap to estimate rotation at this shift."
                    )

                boundary = (
                    abs(best_angle - lo) <= fine_step
                    or abs(best_angle - hi) <= fine_step
                )
                return {
                    "kind": kind,
                    "values": (best_angle,),
                    "message": (
                        f"Estimated rotation: {best_angle:.2f}°. "
                        "Shift retained; preview only."
                        + (
                            " Best angle is near the search boundary; "
                            "another estimate can search farther."
                            if boundary
                            else ""
                        )
                    ),
                }

            except TaskCancelled:
                raise
            except Exception as exc:
                return {"error": f"{type(exc).__name__}: {exc}"}

        self._estimate_task_id = self.state.tasks.start(
            "calculating",
            f"Estimating alignment {kind}",
            calculate,
            on_result=lambda result: self._on_estimate_ready(
                generation, context, result
            ),
            background=True,
        )
        self._update_availability()

    def _on_estimate_ready(self, generation, context, result):
        if self._disposed or generation != self._estimate_generation:
            return

        self._estimate_task_id = None

        if not self.can_apply() or self._estimate_context() != context:
            self.estimate_status.setText(
                "Estimate discarded because the alignment or comparison changed. "
                "Click Estimate again."
            )
        elif "error" in result:
            self.estimate_status.setText(f"Estimation failed: {result['error']}")
            self.state.issue(
                "warning",
                "Alignment estimation",
                result["error"],
                parent=self.window(),
            )
        else:
            if result["kind"] == "flow":
                self.draft.accept_refined_flow(result["remap"])
            elif result["kind"] == "shift":
                self.draft.values[:2] = result["values"]
            else:
                self.draft.values[2] = result["values"][0]

            self.estimate_status.setText(result["message"])
            self.state.alignment_draft_changed.emit()

        self._update_availability()

    def _on_estimate_stopped(self, group, task_id, *_):
        if self._disposed or task_id != self._estimate_task_id:
            return

        self._estimate_task_id = None
        self.estimate_status.setText("Estimation stopped. Click Estimate to retry.")
        self._update_availability()


def select_alignment_draft(data, session_id, parent=None, *, proposal=None):
    state = data.state
    old = state.alignment_draft

    if (
        old is not None
        and old.session_id == session_id
        and old.is_current(data)
        and (proposal is None or old.source_remap is proposal)
    ):
        state.current_session_id = session_id
        return old

    tasks = state.tasks
    if tasks.processing_busy() or tasks.processing_requested:
        state.issue(
            "warning",
            "Alignment editor",
            "Finish or cancel existing tasks first.",
        )
        return None

    if session_id is None or not 0 <= session_id < len(data.sessions):
        return None

    session = data.sessions[session_id]
    if not session.status["spatial_loaded"] or session.background_template is None:
        state.issue(
            "warning",
            "Alignment editor",
            "Load this session's spatial data and background first.",
        )
        return None

    if old is not None and old.dirty:
        answer = QMessageBox.question(
            parent,
            "Replace alignment draft?",
            f"Discard the unapplied changes for session {old.session_id} "
            "and start a new draft?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return None

    draft = AlignmentDraft(data, session_id, proposal=proposal)
    state.alignment_draft = draft
    state.current_session_id = session_id
    state.alignment_draft_changed.emit()
    return draft


class ManualAlignmentDialog(QDialog):
    def __init__(self, data, draft, parent=None):
        super().__init__(parent)

        self.result_remap = None
        self.setWindowTitle(f"Manual alignment — {draft.session.name}")
        self.resize(900, 750)

        layout = QVBoxLayout(self)
        self.editor = ManualAlignmentEditor(data, draft, self)
        self.editor.popout_button.hide()
        layout.addWidget(self.editor)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Close,
            parent=self,
        )
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.editor.apply_requested.connect(self._apply)
        self.editor.review_closed.connect(self.reject)
        data.state.alignment_draft_changed.connect(self._sync_draft)

    def _apply(self):
        if self.editor.can_apply():
            self.result_remap = self.editor.draft.make_remap()
            self.accept()

    def _sync_draft(self):
        draft = self.editor.state.alignment_draft
        if draft is None:
            return

        self.editor.set_draft(draft)
        self.setWindowTitle(f"Manual alignment — {draft.session.name}")


def open_manual_alignment(data, session_id, parent=None, *, failure_reason=None):
    draft = select_alignment_draft(data, session_id, parent)
    if draft is None:
        return

    dialog = ManualAlignmentDialog(data, draft, parent)
    if failure_reason:
        message = QLabel(failure_reason, dialog)
        message.setWordWrap(True)
        dialog.layout().insertWidget(0, message)
        dialog.setWindowTitle(f"Alignment failed — {draft.session.name}")

    try:
        if dialog.exec() == QDialog.DialogCode.Accepted:
            draft = dialog.editor.draft
            data.queue_commit_session_realignment(
                draft.session_id,
                background_template=draft.template,
                remap=dialog.result_remap,
                background_spec=None,
                expected_version=draft.expected_version,
            )
    finally:
        data.state.alignment_draft_changed.disconnect(dialog._sync_draft)
        dialog.editor.dispose()
        dialog.deleteLater()
