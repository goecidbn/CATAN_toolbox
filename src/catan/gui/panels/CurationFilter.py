import importlib
from copy import deepcopy
from dataclasses import replace

from PySide6.QtCore import Qt, Signal, QTimer, QEvent
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QDialog,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QSizePolicy,
    QToolButton,
    QMenu,
    QInputDialog,
    QMessageBox,
    QFormLayout,
    QLineEdit,
    QDialogButtonBox,
    QFrame,
    QGraphicsOpacityEffect,
)

from catan.gui.data.curation_filter import (
    CurationFilterCondition,
    CurationFilterGroup,
    CurationFilterEvaluator,
    CurationFilterError,
    is_filter_condition,
    is_filter_group,
    empty_filter_group_ids,
)

from catan.core.structures import NeuronComponent
from catan.core.changes import ChangeKind, DataChange
from catan.gui.panels import StatisticsData
from catan.gui.panels.helper.Threshold import ThresholdSpec
from catan.gui.data.threshold_references import (
    MODEL_DISTANCE_TERMS,
    ModelDistanceReference,
    PercentileReference,
    ThresholdReferenceError,
    is_distance_query,
)
from catan.gui.data.curation_filter import CurationFilterCondition
from catan.gui.data.curation_results import RetainedResults
from catan.gui.GUI_elements.fragments.curation_filter_group import (
    CurationFilterGroupWidget,
)
from catan.gui.GUI_elements.fragments.curation_actions_widget import (
    CuratorActions,
)
from catan.gui.data.curation_filter_presets import (
    CurationFilterPresetStore,
    PRESET_CATEGORIES,
    PRESET_CATEGORY_DESCRIPTIONS,
    UNCATEGORIZED_LABEL,
    suggest_category,
)

# Preset selector entry that starts an empty filter.
NEW_PRESET_KEY = "__new__"

# Target selector entry inspecting all targets of a neuron at once.
ALL_TARGETS = "__all_targets__"
from catan.gui.panels.helper.ReviewStatusFilter import (
    ReviewStatusFilter,
)
from catan.gui.GUI_elements.fragments.curator_interaction_guard import (
    CuratorInteractionGuard,
)
from catan.gui.background_tasks.runtime import TaskCancelled
from catan.gui.data.statistics.sampling import (
    SUGGESTION_PERCENTILES,
    SUGGESTION_SAMPLES,
)

from catan.gui.panels import BasePlot

importlib.reload(StatisticsData)


class ThresholdSourceSection(QFrame):
    """Framed source of a condition's threshold values with a title and
    explanation. Inactive sections are faded and their controls disabled;
    a click anywhere on one (disabled controls pass it on) activates it."""

    activated = Signal()

    def __init__(self, title, explanation, parent=None):
        super().__init__(parent)
        self.setObjectName("thresholdSource")
        self.setStyleSheet("""
            QFrame#thresholdSource {
                border: 1px solid #454b54;
                border-radius: 6px;
                background: transparent;
            }
            QFrame#thresholdSource[active="true"] {
                border: 1px solid #5b8bd6;
                background: rgba(91, 139, 214, 0.10);
            }
            QLabel#thresholdSourceTitle { font-weight: bold; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        self.title = QLabel(title)
        self.title.setObjectName("thresholdSourceTitle")
        self.explanation = QLabel(explanation)
        self.explanation.setWordWrap(True)
        self.explanation.setObjectName("curationConditionDescription")
        layout.addWidget(self.title)
        layout.addWidget(self.explanation)

        self.content = QWidget()
        self.body = QVBoxLayout(self.content)
        self.body.setContentsMargins(0, 2, 0, 0)
        self.body.setSpacing(4)
        layout.addWidget(self.content)

        self._fade = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._fade)

        self.active = None
        self.set_active(False)

    def set_active(self, active):
        active = bool(active)
        if active == self.active:
            return

        self.active = active
        self.setProperty("active", active)
        self.style().unpolish(self)
        self.style().polish(self)

        self.content.setEnabled(active)
        self._fade.setOpacity(1.0 if active else 0.5)
        self.setCursor(
            Qt.CursorShape.ArrowCursor if active else Qt.CursorShape.PointingHandCursor
        )
        self.setToolTip("" if active else "Click to use this source.")

    def mousePressEvent(self, event):
        if not self.active and event.button() == Qt.MouseButton.LeftButton:
            self.activated.emit()
            event.accept()
            return
        super().mousePressEvent(event)


class CurationFilterConditionPopup(QDialog):

    conditionAccepted = Signal(object)

    def __init__(
        self,
        *,
        engine,
        condition: CurationFilterCondition | None = None,
        tasks=None,
        parent=None,
    ):
        super().__init__(
            parent,
            Qt.WindowType.Popup,
        )

        self.engine = engine
        self.condition = condition
        self.tasks = tasks

        self._suggestion_token = None
        self._suggestion_task_id = None
        self._closed = False

        # Percentiles (threshold_references) of the lower (or single) and
        # upper bound in value mode; None: the spin box value is used.
        self._percentiles = {"lower": None, "upper": None}
        self._setting_values = False
        self._fit_pending = False

        self.setWindowTitle(
            "Edit filter condition" if condition is not None else "Add filter condition"
        )

        self._build_ui()

        if condition is not None:
            self.set_condition(condition)

    def _build_ui(self):

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # ---------------------------------------------
        # Statistic query
        # ---------------------------------------------

        title = QLabel("Filter condition")
        title.setObjectName("curationConditionTitle")

        description = QLabel(
            "Choose a statistic and its reductions, then define "
            "which values should match."
        )
        description.setWordWrap(True)
        description.setObjectName("curationConditionDescription")

        layout.addWidget(title)
        layout.addWidget(description)

        statistic_group = QGroupBox("Statistic")
        statistic_group.setObjectName("curationStatisticGroup")

        statistic_layout = QVBoxLayout(statistic_group)
        statistic_layout.setContentsMargins(10, 12, 10, 10)

        review_row = QHBoxLayout()
        review_row.addWidget(QLabel("Target review status:"))

        self.review_filter = ReviewStatusFilter(self)
        self.review_filter.setToolTip(
            "Choose review states of the neurons being selected or edited. "
            "Reference sources may have any review state."
        )

        review_row.addWidget(self.review_filter)
        review_row.addStretch()
        layout.addLayout(review_row)

        self.query_selector = StatisticsData.StatisticQuerySelector(
            engine=self.engine,
            axis="x",
        )

        self.query_selector.set_query_mode("generic")
        self.query_selector.set_query_preparer(None)

        statistic_layout.addWidget(self.query_selector)

        layout.addWidget(statistic_group)

        # ---------------------------------------------
        # Threshold
        # ---------------------------------------------

        threshold_group = QGroupBox("Threshold")
        threshold_group.setObjectName("curationThresholdGroup")

        threshold_layout = QVBoxLayout(threshold_group)
        threshold_layout.setSpacing(6)

        direction_row = QHBoxLayout()
        direction_row.addWidget(QLabel("Match when:"))

        self.direction_selector = QComboBox()
        self.direction_selector.addItem("≥", "greater")
        self.direction_selector.addItem("≤", "less")
        self.direction_selector.addItem("Between", "between")

        self.direction_selector.setCurrentIndex(0)

        self.direction_selector.setMinimumWidth(100)
        self.direction_selector.setStyleSheet("""
            QComboBox {
                color: #edf0f4;
                background-color: #363c44;
                border: 1px solid #59616c;
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 24px;
            }

            QComboBox QAbstractItemView {
                color: #edf0f4;
                background-color: #30353c;
                selection-background-color: #46566b;
                selection-color: white;
            }
        """)
        direction_row.addWidget(self.direction_selector)
        direction_row.addStretch()
        threshold_layout.addLayout(direction_row)

        # Where the bound values come from: a number / sampled percentile,
        # or the fitted model (distance statistics). One is active.
        self.value_section = ThresholdSourceSection(
            "Number or distribution percentile",
            "Type a fixed number, or pick a percentile of the sampled "
            "values: a percentile is resampled on every evaluation.",
        )
        self.model_section = ThresholdSourceSection(
            "From the fitted model",
            "Distances of the two-noise model in px, following refits. "
            "Negative factors subtract.",
        )
        self.value_section.activated.connect(lambda: self._set_mode("value"))
        self.model_section.activated.connect(lambda: self._set_mode("model"))
        self._mode = "value"

        self.threshold_value = QDoubleSpinBox()
        self.threshold_value.setDecimals(3)
        self.threshold_value.setRange(-1e12, 1e12)
        self.threshold_value.setValue(0.0)
        self.threshold_value.setSingleStep(0.1)

        self.threshold_upper = QDoubleSpinBox()
        self.threshold_upper.setDecimals(self.threshold_value.decimals())
        self.threshold_upper.setRange(
            self.threshold_value.minimum(),
            self.threshold_value.maximum(),
        )
        self.threshold_upper.setSingleStep(self.threshold_value.singleStep())
        self.threshold_upper.setValue(1.0)
        self.threshold_upper.setToolTip("Inclusive upper bound")

        self.interval_separator = QLabel("and")

        value_row = QHBoxLayout()
        value_row.addWidget(self.threshold_value, stretch=1)
        value_row.addWidget(self.interval_separator)
        value_row.addWidget(self.threshold_upper, stretch=1)
        self.value_section.body.addLayout(value_row)

        percentile_row = QHBoxLayout()
        self.percentile_label = QLabel()
        self.percentile_label.setWordWrap(True)
        self.percentile_label.setObjectName("curationConditionDescription")
        self.percentile_clear = QToolButton()
        self.percentile_clear.setText("Use fixed number")
        self.percentile_clear.setAutoRaise(True)
        self.percentile_clear.setToolTip(
            "Keep the shown number instead of the percentile."
        )
        self.percentile_clear.clicked.connect(self._clear_percentiles)
        percentile_row.addWidget(self.percentile_label, stretch=1)
        percentile_row.addWidget(self.percentile_clear)
        self.value_section.body.addLayout(percentile_row)

        self._build_suggestions(self.value_section.body, value_row)
        self._build_model_recipe(self.model_section.body)

        threshold_layout.addWidget(self.value_section)
        threshold_layout.addWidget(self.model_section)
        layout.addWidget(threshold_group)

        self.threshold_error = QLabel()
        self.threshold_error.setWordWrap(True)
        self.threshold_error.setStyleSheet("color: #ef9a9a;")
        self.threshold_error.hide()
        layout.addWidget(self.threshold_error)

        # ---------------------------------------------
        # Buttons
        # ---------------------------------------------

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.cancel_button = QPushButton("Cancel")
        self.accept_button = QPushButton(
            "Apply" if self.condition is not None else "Add"
        )

        self.accept_button.setDefault(True)
        self.accept_button.setAutoDefault(True)

        self.cancel_button.setDefault(False)
        self.cancel_button.setAutoDefault(False)

        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.accept_button)

        layout.addLayout(button_layout)

        self.cancel_button.clicked.connect(self.reject)
        self.accept_button.clicked.connect(self._accept_condition)

        self.query_selector.queryChanged.connect(self._on_query_changed)

        self.direction_selector.currentIndexChanged.connect(
            self._update_threshold_controls
        )
        self.threshold_value.valueChanged.connect(
            lambda: self._on_value_edited("lower")
        )
        self.threshold_upper.valueChanged.connect(
            lambda: self._on_value_edited("upper")
        )

        self._set_mode("value")

    def _build_suggestions(self, layout, button_row):
        """The suggest button joins ``button_row`` (the value inputs);
        the sampling status and the percentile table follow below."""
        self.suggest_status_row = QWidget()
        row = QHBoxLayout(self.suggest_status_row)
        row.setContentsMargins(0, 0, 0, 0)

        self.suggest_button = QPushButton("Suggest from distribution percentile")
        self.suggest_button.setAutoDefault(False)
        self.suggest_button.setToolTip(
            "Sample this statistic's values (about 2 s) and show percentiles. "
            "Pair statistics show same-neuron and nearest-other-neuron pairs. "
            "Click a value to use its percentile as threshold: it is "
            "resampled whenever the filter is evaluated."
        )

        self.suggest_more_button = QPushButton("Sample more")
        self.suggest_more_button.setAutoDefault(False)
        self.suggest_more_button.hide()

        self.suggest_status = QLabel()
        self.suggest_status.setWordWrap(True)
        self.suggest_status.setObjectName("curationConditionDescription")

        button_row.addWidget(self.suggest_button)
        row.addWidget(self.suggest_status, stretch=1)
        row.addWidget(self.suggest_more_button)
        layout.addWidget(self.suggest_status_row)
        self.suggest_status_row.hide()

        self.suggestion_panel = QWidget()
        self.suggestion_grid = QGridLayout(self.suggestion_panel)
        self.suggestion_grid.setContentsMargins(0, 0, 0, 0)
        self.suggestion_grid.setHorizontalSpacing(4)
        self.suggestion_grid.setVerticalSpacing(2)
        self.suggestion_panel.hide()
        layout.addWidget(self.suggestion_panel)

        self.suggest_button.clicked.connect(lambda: self._request_suggestions())
        self.suggest_more_button.clicked.connect(self._request_more_suggestions)

        if self.tasks is not None:
            self.tasks.task_cancelled.connect(self._on_suggestion_stopped)
            self.tasks.task_failed.connect(self._on_suggestion_stopped)
        else:
            self.suggest_button.setEnabled(False)
            self.suggest_button.setToolTip("Suggestions need the task manager.")

    def _clear_suggestions(self, status=""):
        while self.suggestion_grid.count():
            item = self.suggestion_grid.takeAt(0)

            if item.widget() is not None:
                item.widget().deleteLater()

        self.suggestion_panel.hide()
        self.suggest_more_button.hide()
        self._set_suggest_status(status)
        self._schedule_fit()

    def _set_suggest_status(self, text):
        self.suggest_status.setText(text)
        self.suggest_status_row.setVisible(bool(text))

    def _cancel_suggestions(self):
        task_id = self._suggestion_task_id

        self._suggestion_token = None
        self._suggestion_task_id = None

        if task_id is not None and self.tasks is not None:
            self.tasks.cancel(task_id)

    def _request_more_suggestions(self):
        self._request_suggestions(**self._more_request)

    def _request_suggestions(self, *, budget=2.0, allow_full=False):
        query = self.query_selector.effective_query()

        if self.tasks is None or query is None or query.statistic_key == "none":
            return

        self._cancel_suggestions()
        self._clear_suggestions("Sampling values…")

        engine = self.engine
        token = object()
        self._suggestion_token = token

        def sample():
            try:
                return (
                    engine.sample_values(
                        query,
                        SUGGESTION_SAMPLES,
                        budget=budget,
                        allow_full=allow_full,
                    ),
                    None,
                )
            except TaskCancelled:
                raise
            except Exception as exc:
                return None, f"{type(exc).__name__}: {exc}"

        def accept(result):
            if self._suggestion_token is not token or self._closed:
                return

            self._suggestion_token = None
            self._suggestion_task_id = None

            if self.query_selector.effective_query() != query:
                self._clear_suggestions()
                return

            self._show_suggestions(*result, budget=budget, allow_full=allow_full)

        self._suggestion_task_id = self.tasks.start(
            "calculating",
            "Suggest thresholds",
            sample,
            on_result=accept,
        )

    def _on_suggestion_stopped(self, group, task_id):
        if self._closed or task_id != self._suggestion_task_id:
            return

        self._suggestion_token = None
        self._suggestion_task_id = None
        self._clear_suggestions("Sampling stopped; see the task details.")

    def _show_suggestions(self, result, error, *, budget, allow_full):
        self._clear_suggestions()

        if error is not None:
            self._set_suggest_status(error)
            return

        if result is None:
            self._set_suggest_status("No data loaded.")
            return

        if result.method == "unsupported":
            self._set_suggest_status(result.note)
            self._more_request = {"budget": budget, "allow_full": True}
            self.suggest_more_button.setText("Calculate all")
            self.suggest_more_button.setVisible(not allow_full)
            return

        if not result.samples:
            self._set_suggest_status(result.note or "No values to suggest from.")
            return

        grid = self.suggestion_grid

        for column, level in enumerate(SUGGESTION_PERCENTILES, start=2):
            header = QLabel("median" if level == 50 else f"p{level}")
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(header, 0, column)

        for row, sample in enumerate(result.samples, start=1):
            grid.addWidget(QLabel(sample.label), row, 0)

            count = QLabel(f"n={len(sample.values):,}")
            count.setToolTip(
                f"All {sample.available:,} {sample.unit}"
                if sample.exact
                else f"Sampled {sample.sampled:,}"
                + (
                    f" of {sample.available:,}"
                    if sample.available is not None
                    else ""
                )
                + f" {sample.unit}"
            )
            grid.addWidget(count, row, 1)

            percentiles = sample.percentiles()

            if percentiles is None:
                grid.addWidget(QLabel("no values"), row, 2, 1, -1)
                continue

            for column, (level, value) in enumerate(
                zip(SUGGESTION_PERCENTILES, percentiles), start=2
            ):
                button = QPushButton(f"{value:.4g}")
                button.setAutoDefault(False)
                button.setFlat(True)
                button.setToolTip(
                    f"{sample.label}, p{level}: {value:.6g}\n"
                    "Click: use this percentile as threshold. "
                    "Shift+click: as upper bound."
                )
                button.clicked.connect(
                    lambda _=False, value=float(value), reference=PercentileReference(
                        sample.label, float(level)
                    ): self._apply_percentile(reference, value)
                )
                grid.addWidget(button, row, column)

        self.suggestion_panel.show()
        self._schedule_fit()

        complete = all(sample.exact for sample in result.samples)
        self._set_suggest_status(
            ("All values" if complete else "Sampled")
            + f" in {result.elapsed:.1f} s."
            + (f" {result.note}" if result.note else "")
        )

        self._more_request = {"budget": 5 * budget, "allow_full": False}
        self.suggest_more_button.setText("Sample more")
        self.suggest_more_button.setVisible(
            result.method in ("neurons", "pairs")
            and budget < 10
            and any(
                not sample.exact and len(sample.values) < SUGGESTION_SAMPLES
                for sample in result.samples
            )
        )

    def _build_model_recipe(self, layout):
        """Distance statistics: bounds from the fitted model,
        a·h + b·σ_det + c·σ_move + d·σ_h (px), resolved on evaluation."""
        # Per bound: (row widget, bound label, {term: spin box}, preview).
        self.model_rows = {}
        self.model_term_labels = []

        for bound, defaults in (
            ("lower", {"h": 1.0}),
            ("upper", {"h": 1.0, "sigma_h": 2.0}),
        ):
            widget = QWidget()
            row = QHBoxLayout(widget)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(3)

            bound_label = QLabel("Lower:" if bound == "lower" else "Upper:")
            row.addWidget(bound_label)

            spins = {}

            for index, (name, label) in enumerate(MODEL_DISTANCE_TERMS.items()):
                if index:
                    row.addWidget(QLabel("+"))

                spin = QDoubleSpinBox()
                spin.setDecimals(2)
                spin.setRange(-20.0, 20.0)
                spin.setSingleStep(0.5)
                spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
                spin.setAlignment(Qt.AlignmentFlag.AlignRight)
                spin.setFixedWidth(46)
                spin.setToolTip("Factor (scroll or type; negative subtracts)")
                spin.setValue(defaults.get(name, 0.0))
                spin.valueChanged.connect(self._update_threshold_controls)
                spins[name] = spin

                term = QLabel(f"·{label}")
                term.setProperty("term", name)
                self.model_term_labels.append(term)

                row.addWidget(spin)
                row.addWidget(term)

            preview = QLabel()
            preview.setObjectName("curationConditionDescription")
            row.addWidget(preview, stretch=1)

            layout.addWidget(widget)
            self.model_rows[bound] = (widget, bound_label, spins, preview)

    def _model_reference(self, bound):
        _, _, spins, _ = self.model_rows[bound]
        return ModelDistanceReference.from_mapping(
            {name: spin.value() for name, spin in spins.items()}
        )

    def _resolve_model(self, bound):
        try:
            return self._model_reference(bound).resolve(self.engine.data)
        except ThresholdReferenceError:
            return None

    def _update_model_section(self):
        interval = self.direction_selector.currentData() == "between"

        model = getattr(self.engine.data, "model", None)
        fitted = model is not None and model.fitted and model.kind == "two_noise"
        scale = float(self.engine.data.params.get("pxtomu", 1.0))

        from catan.tracking.analytics.two_noise_model import PARAMETERS

        for term in self.model_term_labels:
            name = term.property("term")
            description = PARAMETERS[name][1]
            term.setToolTip(
                f"{MODEL_DISTANCE_TERMS[name]} = "
                f"{float(model.parameters[name]) / scale:.3g} px\n{description}"
                if fitted
                else f"{description}\nNo fitted two-noise model."
            )

        for bound, (widget, bound_label, _, preview) in self.model_rows.items():
            widget.setVisible(bound == "lower" or interval)
            bound_label.setVisible(interval)

            value = self._resolve_model(bound)
            preview.setText(
                "= ? (no fitted model)" if value is None else f"= {value:.3g} px"
            )

    def _set_mode(self, mode):
        if mode == "model" and self.model_section.isHidden():
            mode = "value"

        self._mode = mode
        self.value_section.set_active(mode == "value")
        self.model_section.set_active(mode == "model")
        self._update_threshold_controls()

    def _apply_percentile(self, reference, value):
        shift = bool(
            QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier
        )
        bound = "upper" if shift else "lower"

        self._setting_values = True
        try:
            if shift:
                index = self.direction_selector.findData("between")
                self.direction_selector.setCurrentIndex(index)
            (self.threshold_upper if shift else self.threshold_value).setValue(value)
        finally:
            self._setting_values = False

        self._percentiles[bound] = reference
        self._set_mode("value")

    def _on_value_edited(self, bound):
        # Typing a number replaces the percentile of that bound.
        if not self._setting_values:
            self._percentiles[bound] = None
        self._update_threshold_controls()

    def _clear_percentiles(self):
        self._percentiles = {"lower": None, "upper": None}
        self._update_threshold_controls()

    def _update_percentile_label(self):
        interval = self.direction_selector.currentData() == "between"
        lower, upper = self._percentiles["lower"], self._percentiles["upper"]

        parts = []
        if lower is not None:
            parts.append(("Lower bound: " if interval else "Threshold: ") + lower.describe())
        if interval and upper is not None:
            parts.append("Upper bound: " + upper.describe())

        self.percentile_label.setText("; ".join(parts))
        self.percentile_label.setVisible(bool(parts))
        self.percentile_clear.setVisible(bool(parts))

    def _schedule_fit(self):
        """Fit the height to the content once the layout has settled
        (sections, interval rows and suggestion tables come and go)."""
        if not self._fit_pending:
            self._fit_pending = True
            QTimer.singleShot(0, self._fit_height)

    def _fit_height(self):
        self._fit_pending = False

        if self._closed or not self.isVisible():
            return

        self.layout().activate()
        geometry = self.geometry()
        available = self.screen().availableGeometry()

        height = min(self.sizeHint().height(), available.height())
        width = max(geometry.width(), self.minimumSizeHint().width())
        width = min(width, available.width())
        top = min(geometry.top(), available.bottom() - height + 1)
        left = min(geometry.left(), available.right() - width + 1)

        self.setGeometry(
            max(available.left(), left), max(available.top(), top), width, height
        )

    def showEvent(self, event):
        super().showEvent(event)
        self._schedule_fit()

    def event(self, event):
        # Any content change (sections, rows, suggestion tables, statistic
        # parameters) requests a new layout.
        if event.type() == QEvent.Type.LayoutRequest:
            self._schedule_fit()
        return super().event(event)

    def done(self, result):
        self._closed = True
        self._cancel_suggestions()

        if self.tasks is not None:
            self.tasks.task_cancelled.disconnect(self._on_suggestion_stopped)
            self.tasks.task_failed.disconnect(self._on_suggestion_stopped)
            self.tasks = None

        super().done(result)

    def _on_query_changed(self, query):
        if getattr(self, "_shown_query", None) != query:
            self._shown_query = query
            self._cancel_suggestions()
            self._clear_suggestions()

        definition = (
            None if query is None else self.engine.registry.get(query.statistic_key)
        )
        distance = query is not None and is_distance_query(definition, query)
        if self.model_section.isHidden() == distance:
            self.model_section.setVisible(distance)
            self._schedule_fit()
        if distance:
            self._update_model_section()
        elif self._mode == "model":
            self._set_mode("value")
            return

        error = ""

        try:
            spec = self._threshold_spec()
            spec.validate()

            if not distance and any(
                isinstance(reference, ModelDistanceReference)
                for reference in (spec.reference, spec.upper_reference)
            ):
                raise ValueError(
                    "Model distance thresholds need a distance statistic "
                    "without sum reductions."
                )
        except ValueError as exc:
            error = str(exc)

        self.threshold_error.setText(error)
        self.threshold_error.setVisible(bool(error))

        self.accept_button.setEnabled(
            query is not None and query.statistic_key != "none" and not error
        )

    def set_condition(
        self,
        condition: CurationFilterCondition,
    ):

        self.condition = condition

        self.query_selector.set_query(
            condition.query,
            emit=False,
        )

        self.direction_selector.setCurrentIndex(
            self.direction_selector.findData(condition.threshold.direction)
        )

        self._setting_values = True
        try:
            self.threshold_value.setValue(condition.threshold.value)
            self.threshold_upper.setValue(
                condition.threshold.upper_value
                if condition.threshold.upper_value is not None
                else max(1.0, condition.threshold.value)
            )
        finally:
            self._setting_values = False

        threshold = condition.threshold
        references = {"lower": threshold.reference, "upper": threshold.upper_reference}

        self._percentiles = {
            bound: reference if isinstance(reference, PercentileReference) else None
            for bound, reference in references.items()
        }

        model = False
        for bound, reference in references.items():
            if isinstance(reference, ModelDistanceReference):
                model = True
                coefficients = dict(reference.coefficients)
                for name, spin in self.model_rows[bound][2].items():
                    spin.setValue(coefficients.get(name, 0.0))

        self._on_query_changed(self.query_selector.effective_query())
        self._set_mode("model" if model else "value")

        self._update_threshold_controls()

        self.accept_button.setText("Apply")

        self._on_query_changed(self.query_selector.effective_query())

    def _accept_condition(self):

        query = self.query_selector.effective_query()

        if query is None or query.statistic_key == "none":
            return

        threshold = self._threshold_spec()

        try:
            threshold.validate()
        except ValueError:
            self._update_threshold_controls()
            return

        if self.condition is None:

            condition = CurationFilterCondition(
                query=query,
                threshold=threshold,
            )

        else:

            # Preserve the stable ID when editing.
            condition = CurationFilterCondition(
                query=query,
                threshold=threshold,
                id=self.condition.id,
            )

        self.conditionAccepted.emit(condition)

        self.accept()

    def _threshold_spec(self):
        direction = self.direction_selector.currentData()

        interval = direction == "between"

        if self._mode == "model":
            references = {
                bound: self._model_reference(bound) for bound in ("lower", "upper")
            }
            # Unresolvable without a fitted model; resolved on evaluation.
            values = {
                bound: self._resolve_model(bound) or 0.0 for bound in ("lower", "upper")
            }
        else:
            references = self._percentiles
            values = {
                "lower": self.threshold_value.value(),
                "upper": self.threshold_upper.value(),
            }

        return ThresholdSpec(
            axis=(None if self.condition is None else self.condition.threshold.axis),
            value=values["lower"],
            direction=direction,
            active=True,
            upper_value=values["upper"] if interval else None,
            reference=references["lower"],
            upper_reference=references["upper"] if interval else None,
        )

    def _update_threshold_controls(self, *_):
        interval = self.direction_selector.currentData() == "between"

        self.interval_separator.setVisible(interval)
        self.threshold_upper.setVisible(interval)

        self.threshold_value.setToolTip(
            "Inclusive lower bound" if interval else "Threshold value"
        )
        self._update_percentile_label()
        if not self.model_section.isHidden():
            self._update_model_section()
        self._schedule_fit()

        self._on_query_changed(self.query_selector.effective_query())


class Display(QWidget):

    condition_edit_requested = Signal(str)
    condition_remove_requested = Signal(str)
    condition_add_requested = Signal(str)

    subgroup_add_requested = Signal(str)
    group_remove_requested = Signal(str)

    operator_changed = Signal(str, str)
    match_level_changed = Signal(str, str)
    count_changed = Signal(str, object)
    inspection_changed = Signal()

    comment_changed = Signal(str, str)

    evaluate_requested = Signal()
    select_requested = Signal()

    condition_evidence_requested = Signal(str)
    group_evidence_requested = Signal(str)

    node_move_requested = Signal(str, str, int)

    # saving / loading presets
    preset_selected = Signal(str)

    preset_save_requested = Signal()
    preset_save_as_requested = Signal()

    preset_revert_requested = Signal()
    preset_set_default_requested = Signal()
    preset_rename_requested = Signal()
    preset_delete_requested = Signal()
    preset_category_requested = Signal(object)

    binding_changed = Signal(str, str)
    summary_changed = Signal(str, object)

    def __init__(self, parent, controls=None, config=None):
        super().__init__(parent)

        self.state = parent.state
        self.data = parent.data

        self.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground,
            True,
        )

        format_condition = None
        self.format_condition = format_condition or self._default_format_condition

        self.root_widget = None

        self._build_ui()

        self.set_dirty(False)

    def _build_ui(self):

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.setObjectName("curationFilterDisplay")

        # ---------------------------------------------
        # Preset controls
        # ---------------------------------------------

        preset_layout = QHBoxLayout()

        preset_layout.addWidget(QLabel("Preset:"))

        self.preset_selector = QComboBox()
        self.preset_selector.setObjectName("curationPresetSelector")
        self.preset_selector.setMinimumWidth(180)

        preset_layout.addWidget(
            self.preset_selector,
            stretch=1,
        )

        self.preset_dirty_label = QLabel("*")
        self.preset_dirty_label.setToolTip("Preset has unsaved changes")
        self.preset_dirty_label.hide()

        preset_layout.addWidget(self.preset_dirty_label)

        self.preset_menu_button = QToolButton()
        self.preset_menu_button.setText("⋮")
        self.preset_menu_button.setPopupMode(
            QToolButton.ToolButtonPopupMode.InstantPopup
        )

        self.preset_menu = QMenu(self.preset_menu_button)

        self.preset_save_action = self.preset_menu.addAction("Save")
        self.preset_save_as_action = self.preset_menu.addAction("Save as…")

        self.preset_revert_action = self.preset_menu.addAction("Revert")
        self.preset_menu.addSeparator()
        self.preset_default_action = self.preset_menu.addAction("Set as default")
        self.preset_rename_action = self.preset_menu.addAction("Rename…")
        self.preset_category_menu = self.preset_menu.addMenu("Category")
        self.preset_category_actions = {}

        for category, label in [*PRESET_CATEGORIES.items(), (None, UNCATEGORIZED_LABEL)]:
            action = self.preset_category_menu.addAction(label)
            action.setCheckable(True)
            action.setToolTip(PRESET_CATEGORY_DESCRIPTIONS.get(category, ""))
            action.triggered.connect(
                lambda checked=False, category=category: (
                    self.preset_category_requested.emit(category)
                )
            )
            self.preset_category_actions[category] = action

        self.preset_delete_action = self.preset_menu.addAction("Delete")
        self.preset_menu_button.setMenu(self.preset_menu)

        preset_layout.addWidget(self.preset_menu_button)

        layout.addLayout(preset_layout)

        self.preset_selector.currentIndexChanged.connect(
            self._on_preset_selector_changed
        )
        self.preset_save_action.triggered.connect(self.preset_save_requested)
        self.preset_save_as_action.triggered.connect(self.preset_save_as_requested)
        self.preset_revert_action.triggered.connect(self.preset_revert_requested)
        self.preset_default_action.triggered.connect(self.preset_set_default_requested)
        self.preset_rename_action.triggered.connect(self.preset_rename_requested)
        self.preset_delete_action.triggered.connect(self.preset_delete_requested)

        review_row = QHBoxLayout()
        review_row.addWidget(QLabel("Target review status:"))

        self.review_filter = ReviewStatusFilter(self)
        self.review_filter.setToolTip(
            "Restrict matching targets by review status. "
            "Sources can have any review status."
        )
        review_row.addWidget(self.review_filter)
        review_row.addStretch()

        layout.addLayout(review_row)

        # ---------------------------------------------
        # Scrollable filter tree
        # ---------------------------------------------

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll.setObjectName("curationFilterScroll")

        # self.scroll.setStyleSheet("""
        #     QScrollArea#curationFilterScroll {
        #         background: transparent;
        #         border: none;
        #     }
        # """)

        # self.scroll.viewport().setStyleSheet("background: transparent;")

        self.filter_container = QWidget()
        self.filter_container.setObjectName("curationFilterContainer")
        # self.filter_container.setStyleSheet("background: transparent;")

        self.filter_layout = QVBoxLayout(self.filter_container)
        self.filter_layout.setContentsMargins(8, 8, 8, 8)
        self.filter_layout.setSpacing(6)

        self.filter_layout.addStretch()

        self.scroll.setWidget(self.filter_container)

        layout.addWidget(self.scroll, stretch=1)

        # ---------------------------------------------
        # Evaluation/status area
        # ---------------------------------------------

        status_layout = QHBoxLayout()

        self.status_label = QLabel()
        self.status_label.setObjectName("curationStatusLabel")
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumWidth(0)

        self.status_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )

        status_layout.addWidget(self.status_label, stretch=1)
        status_layout.addStretch()

        self.result_label = QLabel("Matching neurons: —")
        self.result_label.setObjectName("curationResultLabel")
        status_layout.addWidget(
            self.result_label, stretch=0, alignment=Qt.AlignmentFlag.AlignTop
        )

        layout.addLayout(status_layout)

        # ---------------------------------------------
        # Evidence Highlighting
        # ---------------------------------------------

        self.evidence_label = QLabel("Highlighted evidence: —")
        self.evidence_label.setObjectName("curationEvidenceLabel")

        layout.addWidget(self.evidence_label)

        self.inspection_row = QWidget()
        inspection_layout = QHBoxLayout(self.inspection_row)
        inspection_layout.setContentsMargins(0, 0, 0, 0)

        self.target_selector = QComboBox()
        self.target_selector.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.target_selector.setSizePolicy(
            QSizePolicy.Policy.Minimum,
            QSizePolicy.Policy.Fixed,
        )
        self.target_selector.setToolTip(
            "Choose the specific target footprint to inspect."
        )

        self.highlight_selector = QComboBox()
        self.highlight_selector.addItem("Highlight target", "target")
        self.highlight_selector.addItem(
            "Highlight sources",
            "sources",
        )
        self.highlight_selector.addItem("Highlight both", "both")
        self.highlight_selector.setToolTip(
            "When highlighting both: targets are orange, " "sources are light violet."
        )

        inspection_layout.addWidget(self.target_selector)
        inspection_layout.addWidget(self.highlight_selector)
        inspection_layout.addStretch()

        layout.addWidget(self.inspection_row)
        self.inspection_row.hide()

        # Explicit choice of "All targets" (kept across neurons).
        self._all_targets_chosen = False

        self.target_selector.currentIndexChanged.connect(self._on_target_selected)
        self.highlight_selector.currentIndexChanged.connect(
            lambda: self.inspection_changed.emit()
        )

        # ---------------------------------------------
        # Buttons
        # ---------------------------------------------

        button_layout = QHBoxLayout()

        self.evaluate_button = QPushButton("Evaluate")
        self.evaluate_button.setObjectName("curationEvaluateButton")

        self.select_button = QPushButton("Select matching neurons")
        self.select_button.setEnabled(False)
        self.select_button.setObjectName("curationSelectButton")

        button_layout.addStretch()
        button_layout.addWidget(self.evaluate_button)
        button_layout.addWidget(self.select_button)

        self.clear_results_button = QPushButton("Clear results")
        self.clear_results_button.setEnabled(False)
        button_layout.addWidget(self.clear_results_button)

        layout.addLayout(button_layout)

        self.evaluate_button.clicked.connect(self.evaluate_requested)

        self.select_button.clicked.connect(self.select_requested)

        self._apply_styles()

    def _apply_styles(self):

        self.setStyleSheet("""
            /* =====================================================
            Main curator surface
            ===================================================== */

            QWidget#curationFilterDisplay {
                background-color: #202328;
                color: #e7e9ed;
            }

            QScrollArea#curationFilterScroll {
                background-color: #202328;
                border: 1px solid #3d424a;
                border-radius: 6px;
            }

            QScrollArea#curationFilterScroll > QWidget > QWidget {
                background-color: #202328;
            }

            QWidget#curationFilterContainer {
                background-color: #202328;
            }


            /* =====================================================
            Text
            ===================================================== */

            QLabel {
                color: #e7e9ed;
                background-color: transparent;
            }

            QLabel#curationStatusLabel {
                color: #b9c0ca;
            }

            QLabel#curationResultLabel {
                color: #e7e9ed;
                font-weight: 500;
            }


            /* =====================================================
            Filter groups
            ===================================================== */

            QFrame#curationFilterGroup {
                background-color: #292d33;
                border: 1px solid #484e57;
                border-radius: 6px;
            }

            QFrame#curationFilterGroup:hover {
                border-color: #59616c;
            }


            /* =====================================================
            Group selectors
            ===================================================== */

            QComboBox#curationGroupSelector {
                color: #edf0f4;
                background-color: #363b43;
                border: 1px solid #555d68;
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 20px;
            }

            QComboBox#curationGroupSelector:hover {
                background-color: #40464f;
                border-color: #68727f;
            }

            QComboBox#curationGroupSelector:focus {
                border-color: #7aa2d6;
            }

            QComboBox#curationGroupSelector QAbstractItemView {
                color: #edf0f4;
                background-color: #30353c;
                border: 1px solid #555d68;
                selection-background-color: #46566b;
                selection-color: #ffffff;
            }


            /* =====================================================
            Filter chips
            ===================================================== */

            QToolButton#curationConditionButton {
                color: #edf0f4;
                background-color: #3a414a;
                border: 1px solid #606a76;
                border-radius: 5px;
                padding: 2px 6px;
            }

            QToolButton#curationConditionButton:hover {
                background-color: #48515c;
                border-color: #75808d;
            }

            QToolButton#curationConditionRemoveButton {
                color: #dfe3e8;
                background-color: #3a414a;
                border: 1px solid #606a76;
                border-left: 0px;
                border-radius: 4px;
                padding: 1px;
            }

            QToolButton#curationConditionRemoveButton:hover {
                color: #ffffff;
                background-color: #734747;
                border-color: #8b5656;
            }


            /* =====================================================
            Group controls
            ===================================================== */

            QToolButton#curationAddButton {
                color: #d7dce3;
                background-color: transparent;
                border: 1px solid #555d68;
                border-radius: 4px;
                padding: 2px 6px;
            }

            QToolButton#curationAddButton:hover {
                color: #ffffff;
                background-color: #383e46;
                border-color: #68727f;
            }

            QToolButton#curationRemoveGroupButton {
                color: #cdd2d9;
                background-color: transparent;
                border: none;
                padding: 3px 6px;
            }

            QToolButton#curationRemoveGroupButton:hover {
                color: #ffffff;
                background-color: #734747;
                border-radius: 4px;
            }


            /* =====================================================
            Main action buttons
            ===================================================== */

            QPushButton#curationEvaluateButton,
            QPushButton#curationSelectButton {
                color: #edf0f4;
                background-color: #343a42;
                border: 1px solid #5b6470;
                border-radius: 5px;
                padding: 6px 12px;
                min-height: 22px;
            }

            QPushButton#curationEvaluateButton:hover,
            QPushButton#curationSelectButton:hover {
                background-color: #424a54;
                border-color: #6e7986;
            }

            QPushButton#curationEvaluateButton:pressed,
            QPushButton#curationSelectButton:pressed {
                background-color: #2c3138;
            }

            QPushButton#curationEvaluateButton:disabled,
            QPushButton#curationSelectButton:disabled {
                color: #747b84;
                background-color: #292d32;
                border-color: #40464e;
            }


            /* =====================================================
            Scroll bar
            ===================================================== */

            QScrollBar:vertical {
                background-color: #202328;
                width: 10px;
                margin: 0px;
            }

            QScrollBar::handle:vertical {
                background-color: #4c535d;
                border-radius: 5px;
                min-height: 30px;
            }

            QScrollBar::handle:vertical:hover {
                background-color: #606975;
            }

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }

            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: transparent;
            }

            /* =====================================================
            Evidence Highlighting
            ===================================================== */
            QLabel#curationEvidenceLabel {
                color: #b9c0ca;
                background: transparent;
            }
        """)

    def set_filter(self, root: CurationFilterGroup):
        new_widget = CurationFilterGroupWidget(
            root,
            format_condition=self.format_condition,
            is_root=True,
        )
        self._connect_group_widget(new_widget)

        old_widget = self.root_widget
        self.root_widget = new_widget

        if old_widget is not None:
            self.filter_layout.removeWidget(old_widget)
            old_widget.hide()
            old_widget.deleteLater()

        # Insert before the stretch.
        self.filter_layout.insertWidget(0, new_widget)

    def _connect_group_widget(
        self,
        widget: CurationFilterGroupWidget,
    ):

        widget.condition_edit_requested.connect(self.condition_edit_requested)
        widget.condition_remove_requested.connect(self.condition_remove_requested)
        widget.condition_add_requested.connect(self.condition_add_requested)
        widget.subgroup_add_requested.connect(self.subgroup_add_requested)
        widget.group_remove_requested.connect(self.group_remove_requested)
        widget.operator_changed.connect(self.operator_changed)
        widget.match_level_changed.connect(self.match_level_changed)
        widget.condition_evidence_requested.connect(self.condition_evidence_requested)
        widget.group_evidence_requested.connect(self.group_evidence_requested)
        widget.node_move_requested.connect(self.node_move_requested)
        widget.count_changed.connect(self.count_changed)
        widget.comment_changed.connect(self.comment_changed)
        widget.binding_changed.connect(self.binding_changed)
        widget.summary_changed.connect(self.summary_changed)

    def _default_format_condition(
        self,
        condition: CurationFilterCondition,
    ):
        key = condition.query.statistic_key
        stat_def = self.data.statistic_engine.registry.get(key)

        if stat_def is None:
            text = condition.threshold.expression(key, precision=4)
            tooltip = (
                f"Statistic unavailable: {key}\n\n"
                "Load data providing this statistic, or edit the condition "
                "to choose another statistic.\n"
                "This condition remains stored in the preset."
            )
            return text, tooltip

        query_title = StatisticsData.format_query_expression(
            condition.query,
            stat_def,
        )

        threshold, notes = self._display_threshold(condition.threshold)
        text = threshold.expression(query_title, precision=4)

        lines = [
            query_title,
            "",
            f"Threshold: {threshold.expression(precision=6)}",
            *notes,
        ]

        if not threshold.active:
            lines.append("Threshold disabled")

        parameters = dict(condition.query.parameters)

        if stat_def.parameters:
            lines.extend(["", "Parameters:"])

            for parameter in stat_def.parameters:
                value = parameters.get(parameter.key, parameter.default)

                if parameter.kind is bool:
                    value_text = "Yes" if value else "No"
                elif parameter.kind is int:
                    value_text = str(int(value))
                elif parameter.kind is float:
                    value_text = f"{float(value):.6g}"
                else:
                    value_text = str(value)

                lines.append(f"  {parameter.label}: {value_text}")

        return text, "\n".join(lines)

    def _display_threshold(self, threshold):
        """A copy of ``threshold`` with model recipes resolved for the
        current model, and notes on its recipes."""
        threshold = replace(threshold)
        notes = []

        for reference, attribute in (
            (threshold.reference, "value"),
            (
                threshold.upper_reference if threshold.direction == "between" else None,
                "upper_value",
            ),
        ):
            if isinstance(reference, ModelDistanceReference):
                try:
                    setattr(threshold, attribute, reference.resolve(self.data))
                except ThresholdReferenceError as exc:
                    notes.append(str(exc))
            elif isinstance(reference, PercentileReference):
                notes.append(
                    f"{reference.describe()}: value from the last sample, "
                    "resampled on evaluation."
                )

        return threshold, notes

    def set_retained_results(self, count, note=""):
        self.result_label.setText(
            "Matching neurons: —" if count is None else f"Matching neurons: {count}"
        )

        self.select_button.setEnabled(count is not None and count > 0)
        self.clear_results_button.setEnabled(count is not None)

        stale = count is not None and bool(note)

        self.select_button.setText(
            "Select matching neurons *" if stale else "Select matching neurons"
        )
        self.select_button.setToolTip(
            note + "\nSelect uses the retained results. Evaluate replaces them."
            if stale
            else "Select the remaining evaluation results."
        )
        self.select_button.setStyleSheet(
            "QPushButton {" " color: #F2CA7A;" " border: 1px solid #B88736;" "}"
            if stale
            else ""
        )

    def set_dirty(
        self,
        dirty=True,
        dirty_text="Filter changed — not evaluated",
    ):
        self.status_label.setText(dirty_text if dirty else "")

    def set_evaluating(self):
        self.clear_evaluation_errors()
        self.status_label.setText("Evaluating… Previous results remain available.")
        self.evaluate_button.setEnabled(False)

    def set_result_count(self, count):
        self.clear_evaluation_errors()
        self.status_label.clear()
        self.evaluate_button.setEnabled(True)
        self.set_retained_results(count)

    def set_evaluation_error(
        self,
        message: str,
        *,
        node_type: str | None = None,
        node_id: str | None = None,
    ):

        self.clear_evaluation_errors()

        self.status_label.setText(f"Evaluation failed: {message}")

        self.evaluate_button.setEnabled(True)

        if (
            node_type is not None
            and node_id is not None
            and self.root_widget is not None
        ):
            self.root_widget.set_evaluation_error(node_type, node_id, message)

    def set_evidence_status(
        self,
        neuron_id: int | None,
        n_components: int = 0,
    ):

        if neuron_id is None:
            self.set_target_inspection(None)
            self.evidence_label.setText("Highlighted evidence: —")
            return

        self.evidence_label.setText(
            f"Neuron {neuron_id}: "
            f"{n_components} evidence footprint"
            f"{'' if n_components == 1 else 's'} highlighted"
        )

    def set_evidence_source(self, source_type: str, source_id: str):

        if self.root_widget is None:
            return

        self.root_widget.set_evidence_source(source_type, source_id)

    def _on_target_selected(self):
        # Only user changes arrive here (programmatic ones are blocked).
        self._all_targets_chosen = self.target_selector.currentData() == ALL_TARGETS
        self.inspection_changed.emit()

    def set_target_inspection(self, mapping):
        """Fill the target selector; returns the inspected target, ALL_TARGETS
        or None. "All targets" is preselected when the user chose it, or
        when all targets share the same sources (e.g. two footprint groups
        of a track mismatch), where single targets show nothing new."""
        previous = self.target_selector.currentData()

        targets = sorted(
            mapping or {},
            key=lambda c: (c.neuron_id, -1 if c.session_id is None else c.session_id),
        )

        identical = len(targets) > 1 and all(
            mapping[target] == mapping[targets[0]] for target in targets
        )

        blocked = self.target_selector.blockSignals(True)

        try:
            self.target_selector.clear()

            if len(targets) > 1:
                self.target_selector.addItem(f"All targets ({len(targets)})", ALL_TARGETS)

            for target in targets:
                self.target_selector.addItem(
                    f"Target: neuron {target.neuron_id}"
                    + (
                        ""
                        if target.session_id is None
                        else f", session {target.session_id}"
                    ),
                    target,
                )

            if targets:
                if previous in targets:
                    index = self.target_selector.findData(previous)
                elif len(targets) > 1 and (identical or self._all_targets_chosen):
                    index = 0
                else:
                    index = self.target_selector.findData(targets[0])

                self.target_selector.setCurrentIndex(index)

        finally:
            self.target_selector.blockSignals(blocked)

        # Recalculate after populating the target descriptions.
        self.target_selector.setMinimumWidth(0)
        self.target_selector.setMinimumWidth(self.target_selector.sizeHint().width())
        self.target_selector.updateGeometry()
        self.inspection_row.setVisible(bool(targets))

        return self.target_selector.currentData() if targets else None

    def set_invalid_groups(self, invalid_group_ids: set[str]):

        if self.root_widget is not None:
            self.root_widget.set_invalid_groups(invalid_group_ids)

        self.evaluate_button.setEnabled(not invalid_group_ids)

    def refresh_condition_availability(self):
        if self.root_widget is None:
            return set()

        return self.root_widget.refresh_condition_availability(
            self.data.statistic_engine.registry
        )

    def _on_preset_selector_changed(self, index: int):

        if index < 0:
            return

        key = self.preset_selector.itemData(index)

        if key is not None:
            self.preset_selected.emit(key)

    def _add_preset_header(self, text, tooltip=""):
        """Non-selectable section title in the preset selector."""
        self.preset_selector.addItem(text, None)
        index = self.preset_selector.count() - 1
        item = self.preset_selector.model().item(index)
        item.setEnabled(False)
        font = item.font()
        font.setBold(True)
        item.setFont(font)

        if tooltip:
            self.preset_selector.setItemData(index, tooltip, Qt.ToolTipRole)

    def set_preset_state(
        self,
        presets,
        *,
        current_key: str | None,
        default_key: str | None,
        dirty: bool,
        can_edit_category: bool = False,
    ):

        self.preset_selector.blockSignals(True)

        try:
            self.preset_selector.clear()

            selected_index = -1

            self.preset_selector.addItem("New preset…", NEW_PRESET_KEY)
            self.preset_selector.setItemData(
                0, "Start an empty filter.", Qt.ToolTipRole
            )

            if current_key is None:
                # The working filter is not saved as a preset.
                self.preset_selector.addItem("Unsaved preset", None)
                selected_index = self.preset_selector.count() - 1

            # Presets arrive sorted by category, then name.
            section = object()

            for preset in presets:

                category = preset.category if preset.category in PRESET_CATEGORIES else None

                if category != section:
                    section = category
                    self.preset_selector.insertSeparator(self.preset_selector.count())
                    self._add_preset_header(
                        PRESET_CATEGORIES.get(category, UNCATEGORIZED_LABEL),
                        PRESET_CATEGORY_DESCRIPTIONS.get(category, ""),
                    )

                text = "    " + preset.name
                tooltip = []
                if preset.source == "builtin":
                    text = f"{text} [CATAN]"
                    tooltip.append("Built-in CATAN preset")

                if preset.key == default_key:
                    text = f"{text} (default)"
                    # tooltip.append("Default preset")

                self.preset_selector.addItem(
                    text,
                    preset.key,
                )
                if tooltip:
                    self.preset_selector.setItemData(
                        self.preset_selector.count() - 1,
                        "\n".join(tooltip),
                        Qt.ToolTipRole,
                    )

                if preset.key == current_key:
                    selected_index = self.preset_selector.count() - 1

            if selected_index >= 0:
                self.preset_selector.setCurrentIndex(selected_index)
            else:
                self.preset_selector.setCurrentIndex(-1)

        finally:
            self.preset_selector.blockSignals(False)

        self.preset_dirty_label.setVisible(dirty)

        current_info = next(
            (preset for preset in presets if preset.key == current_key),
            None,
        )

        writable = current_info is not None and current_info.writable

        # Built-in presets: editable installations only (same as categories).
        self.preset_save_action.setEnabled(can_edit_category and dirty)
        self.preset_revert_action.setEnabled(current_info is not None and dirty)
        self.preset_default_action.setEnabled(current_info is not None)
        self.preset_rename_action.setEnabled(writable)
        # Built-in presets: editable installations only (same as categories).
        self.preset_delete_action.setEnabled(can_edit_category)

        self.preset_category_menu.setEnabled(can_edit_category)
        category = None if current_info is None else current_info.category

        for key, action in self.preset_category_actions.items():
            action.setChecked(current_info is not None and key == category)

    def clear_evaluation_errors(self):

        if self.root_widget is not None:
            self.root_widget.clear_evaluation_errors()


class Controller(BasePlot.ControlsController):

    menu: Display

    def __init__(self, display_section, config=None):
        super().__init__(display_section, config)

        if not hasattr(self.state, "curation_filter_store"):
            self.state.curation_filter_store = CurationFilterPresetStore(
                settings=self.state.settings,
            )
        self.filter_store = self.state.curation_filter_store

        self.root_filter = self.filter_store.working_filter

        self.evaluator = CurationFilterEvaluator(self.data.statistic_engine)

        self.current_result = None
        self._results_note = ""
        self._filter_generation = 0
        self._evaluation_task_id = None
        self._evaluation_token = None

        self._evidence_source = ("group", self.root_filter.id)

    def build_controls(self):

        self.menu.condition_remove_requested.connect(self._remove_condition)
        self.menu.subgroup_add_requested.connect(self._add_subgroup)
        self.menu.group_remove_requested.connect(self._remove_group)
        self.menu.operator_changed.connect(self._set_group_operator)
        self.menu.match_level_changed.connect(self._set_group_match_level)
        self.menu.count_changed.connect(self._set_group_count)
        self.menu.comment_changed.connect(self._set_group_comment)
        self.menu.evaluate_requested.connect(self._evaluate_filter)
        self.menu.select_requested.connect(self._select_matching_neurons)
        self.menu.inspection_changed.connect(self._update_evidence_highlight)

        self.menu.condition_add_requested.connect(self._add_condition_requested)
        self.menu.condition_edit_requested.connect(self._edit_condition_requested)
        self.menu.condition_evidence_requested.connect(self._show_condition_evidence)
        self.menu.group_evidence_requested.connect(self._show_group_evidence)

        self.menu.node_move_requested.connect(self._move_filter_node)
        self.menu.binding_changed.connect(self._set_group_binding)
        self.menu.summary_changed.connect(self._set_group_summary)

        self.menu.clear_results_button.clicked.connect(self._clear_results)
        self.menu.review_filter.changed.connect(self._on_review_filter_changed)

        # preset loading / saving
        self.menu.preset_selected.connect(self._select_filter_preset)
        self.menu.preset_save_requested.connect(self._save_filter_preset)
        self.menu.preset_save_as_requested.connect(self._save_filter_preset_as)
        self.menu.preset_revert_requested.connect(self._revert_filter_preset)
        self.menu.preset_set_default_requested.connect(self._set_filter_preset_default)
        self.menu.preset_rename_requested.connect(self._rename_filter_preset)
        self.menu.preset_delete_requested.connect(self._delete_filter_preset)
        self.menu.preset_category_requested.connect(self._set_filter_preset_category)

        self.state.focused_component_changed.connect(self._on_focused_component_changed)

        self.actions = CuratorActions(self)
        self._interaction_guard = CuratorInteractionGuard(self)
        self.menu.layout().addWidget(self.actions)

        if self.filter_store.can_save_builtin():
            self.menu.preset_menu.addSeparator()
            action = self.menu.preset_menu.addAction("Save as CATAN preset…")
            action.triggered.connect(self._save_builtin_filter_preset)

        self._rerender()

        self.state.data_changed.connect(self._on_statistic_availability_changed)

        self.connect_signal(
            self.state.tasks.task_cancelled,
            self._on_evaluation_cancelled,
            controls=True,
        )
        self.connect_signal(
            self.state.tasks.task_failed, self._on_evaluation_failed, controls=True
        )

    def _set_group_binding(self, group_id: str, binding: str):
        if binding not in (
            "same",
            "source",
            "target",
            "between_sources",
            "between_targets",
            "between_sources_targets",
        ):
            return
            # raise ValueError(binding)

        group = self._find_group(group_id)

        if group is None or group.apply_to == binding:
            return

        group.apply_to = binding
        self._mark_dirty()

    def _set_group_summary(self, group_id: str, summary):
        if summary not in (None, "median", "mean", "min", "max"):
            return

        group = self._find_group(group_id)

        if group is None or group.summary == summary:
            return

        group.summary = summary
        self._mark_dirty()

    def _set_group_comment(self, group_id: str, comment: str):
        group = self._find_group(group_id)

        if group is None or group.comment == comment:
            return

        group.comment = comment

        # Documentation changes require saving, but not reevaluation.
        self.filter_store.mark_working_dirty()
        self._refresh_preset_controls()

    def _find_group(
        self,
        group_id: str,
        group=None,
    ):

        if group is None:
            group = self.root_filter

        if group.id == group_id:
            return group

        for child in group.children:
            if is_filter_group(child):
                found = self._find_group(group_id, child)

                if found is not None:
                    return found

        return None

    def _find_parent_group(
        self,
        node_id: str,
        group=None,
    ):

        if group is None:
            group = self.root_filter

        for child in group.children:

            if child.id == node_id:
                return group

            if is_filter_group(child):

                found = self._find_parent_group(
                    node_id,
                    child,
                )

                if found is not None:
                    return found

        return None

    def _set_group_count(self, group_id, spec):
        group = self._find_group(group_id)

        if group is None:
            return

        group.count = spec
        self._mark_dirty()

    def _refresh_results(self):
        count = (
            None if self.current_result is None else len(self.current_result.neurons)
        )
        self.menu.set_retained_results(count, self._results_note)

    def _clear_results(self):
        if self.actions.queue is not None:
            self.actions._finish("Stopped: results cleared")

        # Also discard an evaluation that was already running.
        self._filter_generation += 1

        self.current_result = None
        self._results_note = ""

        self.state.update_highlighted_components(None)
        self.menu.set_target_inspection(None)
        self.menu.set_evidence_status(None)

        self._update_filter_validity()
        self.actions.refresh_candidates()

    def _mark_dirty(self):
        self._filter_generation += 1
        self.filter_store.mark_working_dirty()

        self._results_note = "Preset changed since evaluation."
        self.menu.set_dirty(True, self._results_note)

        self._refresh_results()
        self._refresh_preset_controls()
        self.actions.refresh_candidates()

    def _rerender(self):

        self.menu.set_filter(self.root_filter)
        self.menu.review_filter.set_visible_statuses(self.root_filter.review_statuses)
        self.actions.sync()
        self.menu.set_evidence_source(*self._evidence_source)

        self._update_filter_validity()
        self._refresh_preset_controls()

    def _set_group_operator(self, group_id: str, operator: str):

        group = self._find_group(group_id)

        if group is None:
            return

        group.operator = operator

        self._mark_dirty()

    def _set_group_match_level(self, group_id: str, match_level: str):

        group = self._find_group(group_id)

        if group is None:
            return

        group.match_level = match_level

        self._mark_dirty()

    def _add_subgroup(self, parent_group_id: str):

        parent = self._find_group(parent_group_id)

        if parent is None:
            return

        parent.children.append(
            CurationFilterGroup(
                operator="and",
                match_level=parent.match_level,
            )
        )

        self._mark_dirty()
        self._rerender()

    def _remove_group(self, group_id: str):

        parent = self._find_parent_group(group_id)

        if parent is None:
            return

        parent.children = [child for child in parent.children if child.id != group_id]

        self._mark_dirty()
        self._rerender()

    def _remove_condition(self, condition_id: str):

        parent = self._find_parent_group(condition_id)

        if parent is None:
            return

        parent.children = [
            child for child in parent.children if child.id != condition_id
        ]

        self._mark_dirty()
        self._rerender()

    def _on_review_filter_changed(self):
        selected = self.menu.review_filter.visible_statuses

        statuses = (
            None
            if len(selected) == len(self.menu.review_filter.actions)
            else tuple(sorted(int(status) for status in selected))
        )

        if statuses == self.root_filter.review_statuses:
            return

        self.root_filter.review_statuses = statuses
        self._mark_dirty()

    def _evaluate_filter(self):
        if self._evaluation_token is not None or self.actions.queue is not None:
            return

        if not self._update_filter_validity():
            return

        generation = self._filter_generation
        version = self.state.data_version
        snapshot = deepcopy(self.root_filter)

        token = object()
        self._evaluation_token = token
        self.menu.set_evaluating()

        def evaluate():
            try:
                result = CurationFilterEvaluator(self.data.statistic_engine).evaluate(
                    snapshot
                )

                return result, None

            except TaskCancelled:
                # Cancellation must reach Worker, not become a normal result.
                raise

            except CurationFilterError as exc:
                return None, (str(exc), exc.node_type, exc.node_id)

            except Exception as exc:
                return None, (
                    f"{type(exc).__name__}: {exc}",
                    None,
                    None,
                )

        def accept_result(result):
            if self._evaluation_token is not token:
                return

            self._evaluation_task_id = None
            self._evaluation_token = None

            if self._deactivated:
                return

            try:
                self._on_filter_evaluated(
                    result,
                    generation=generation,
                    version=version,
                )
            finally:
                self._update_filter_validity(preserve_status=True)

        try:
            self._evaluation_task_id = self.state.tasks.start(
                "calculating",
                "Evaluate curation filter",
                evaluate,
                on_result=accept_result,
            )
        except Exception:
            self._evaluation_task_id = None
            self._evaluation_token = None
            self._update_filter_validity()
            raise

    def _finish_stopped_evaluation(self, group, task_id, message):
        if group != "calculating" or task_id != self._evaluation_task_id:
            return

        self._evaluation_task_id = None
        self._evaluation_token = None

        if self._deactivated:
            return

        self._update_filter_validity(preserve_status=True)

        if self.current_result is not None:
            message += " Previous results remain available."

        self.menu.status_label.setText(message)

    def _on_evaluation_cancelled(self, group, task_id):
        self._finish_stopped_evaluation(
            group,
            task_id,
            "Evaluation cancelled.",
        )

    def _on_evaluation_failed(self, group, task_id):
        self._finish_stopped_evaluation(
            group,
            task_id,
            "Evaluation failed; see the task error details.",
        )

    def deactivate(self):
        task_id = self._evaluation_task_id
        self._evaluation_task_id = None
        self._evaluation_token = None

        if task_id is not None:
            self.state.tasks.cancel(task_id)

        super().deactivate()

    def _on_filter_evaluated(
        self,
        result_and_error,
        *,
        generation: int,
        version: int,
    ):

        result, error = result_and_error

        # Filter was edited while calculation ran.
        if generation != self._filter_generation or version != self.state.data_version:
            self.menu.set_dirty(
                True,
                "Data or filter changed — evaluate again",
            )
            self._update_filter_validity(preserve_status=True)
            return

        if error is not None:

            message, node_type, node_id = error

            self.menu.set_evaluation_error(
                message, node_type=node_type, node_id=node_id
            )

            return

        self.current_result = RetainedResults(result, self.data)
        self._results_note = ""

        # Show the values the threshold recipes resolved to.
        for condition_id, (value, upper_value) in result.resolved_thresholds.items():
            condition = self._find_condition(condition_id)
            if condition is not None:
                condition.threshold.value = value
                condition.threshold.upper_value = upper_value
        if result.resolved_thresholds:
            self.menu.refresh_condition_availability()

        self._evidence_source = (
            "group",
            self.current_result.root_id,
        )
        self.menu.set_evidence_source(*self._evidence_source)

        self.actions.result_ready()
        self.menu.set_result_count(len(self.current_result.neurons))
        self._update_evidence_highlight()

    def _select_matching_neurons(self):

        if self.current_result is None:
            return

        components = [
            NeuronComponent(
                neuron_id=int(neuron_id),
                session_id=None,
            )
            for neuron_id in sorted(self.current_result.neurons)
        ]

        self.state.update_selected_components(components, [])

    def _highlight_endpoints(self, sources, targets):
        mode = self.menu.highlight_selector.currentData()

        source_components = set(self._display_components(sources))
        target_components = set(self._display_components(targets))

        roles = {}

        if mode == "both":
            components = source_components | target_components
            roles = {component: "source" for component in source_components}
            roles.update({component: "target" for component in target_components})
            showing = (
                '<span style="color:#FFB36B">targets</span> + '
                '<span style="color:#C9A6FF">sources</span> highlighted'
            )

        elif mode == "target":
            components = target_components
            showing = "target highlighted"

        else:
            components = source_components
            showing = "sources highlighted"

        self.state.update_highlighted_components(
            list(components) or None,
            roles=roles,
        )
        return showing

    def _add_condition_requested(self, group_id: str):

        group = self._find_group(group_id)

        if group is None:
            return

        popup = CurationFilterConditionPopup(
            engine=self.data.statistic_engine,
            tasks=self.state.tasks,
            parent=self.menu,
        )

        popup.conditionAccepted.connect(
            lambda condition: (
                self._add_condition(
                    group_id,
                    condition,
                )
            )
        )

        popup.show()

    def _add_condition(self, group_id: str, condition: CurationFilterCondition):

        group = self._find_group(group_id)

        if group is None:
            return

        group.children.append(condition)

        self._mark_dirty()
        self._rerender()

    def _find_condition(self, condition_id: str, group=None):

        if group is None:
            group = self.root_filter

        for child in group.children:

            if is_filter_condition(child) and child.id == condition_id:
                return child

            if is_filter_group(child):

                found = self._find_condition(
                    condition_id,
                    child,
                )

                if found is not None:
                    return found

        return None

    def _edit_condition_requested(self, condition_id: str):

        condition = self._find_condition(condition_id)

        if condition is None:
            return

        popup = CurationFilterConditionPopup(
            engine=self.data.statistic_engine,
            condition=condition,
            tasks=self.state.tasks,
            parent=self.menu,
        )

        popup.conditionAccepted.connect(
            lambda replacement: (
                self._replace_condition(
                    condition_id,
                    replacement,
                )
            )
        )

        popup.show()

    def _replace_condition(
        self, condition_id: str, replacement: CurationFilterCondition
    ):

        parent = self._find_parent_group(condition_id)

        if parent is None:
            return

        for index, child in enumerate(parent.children):
            if child.id == condition_id:

                parent.children[index] = replacement

                self._mark_dirty()
                self._rerender()
                return

    def _on_focused_component_changed(self):
        focused = self.state.focused_component
        neuron = None if focused is None else int(focused.neuron_id)

        # A source/target view (possibly edited by Ctrl-clicks) stays
        # until another neuron is focused.
        if (
            neuron is not None
            and neuron == getattr(self, "_evidence_neuron", None)
            and self.state.highlight_roles
        ):
            return

        self._evidence_neuron = neuron
        self._update_evidence_highlight()

    def _display_components(self, endpoints):
        """Expand neuron endpoints for display only."""
        result = set()
        ids = self.state.assignments

        for endpoint in endpoints:
            if endpoint.session_id is not None:
                result.add(endpoint)
                continue

            neuron = int(endpoint.neuron_id)

            if not 0 <= neuron < ids.shape[0]:
                continue

            for sid, session in enumerate(self.data.sessions[: ids.shape[1]]):
                # Inactive sessions are not displayed.
                if session is None or not getattr(session, "active", True):
                    continue

                footprint = int(ids[neuron, sid])

                if not 0 <= footprint < session.n_neurons:
                    continue

                if session.included is not None and not session.included[footprint]:
                    continue

                result.add(NeuronComponent(neuron, sid))

        return sorted(
            result,
            key=lambda c: (c.neuron_id, c.session_id),
        )

    def _update_evidence_highlight(self):

        if hasattr(self, "actions") and self.actions.committing:
            return

        focused = self.state.focused_component

        if self.current_result is None or focused is None:
            self.state.update_highlighted_components(None)
            self.menu.set_evidence_status(None)
            return

        neuron_id = int(focused.neuron_id)

        if neuron_id not in self.current_result.neurons:
            self.state.update_highlighted_components(None)
            self.menu.set_target_inspection(None)
            self.menu.set_evidence_status(neuron_id, 0)
            return

        source_type, source_id = self._evidence_source

        mapping = self.current_result.inspection_targets(
            neuron_id,
            source_type,
            source_id,
        )

        if mapping is not None:
            target = self.menu.set_target_inspection(mapping)

            if target is None:
                self.state.update_highlighted_components(None)
                self.menu.evidence_label.setText("No matching target in this group.")
                return

            if target == ALL_TARGETS:
                targets = set(mapping)
                sources = set().union(*mapping.values())
                described = f"All {len(targets)} targets of n{neuron_id}"
            else:
                targets = {target}
                sources = mapping[target]
                described = f"Target n{target.neuron_id}" + (
                    ""
                    if target.session_id is None
                    else f", session {target.session_id}"
                )

            source_neurons = {c.neuron_id for c in sources}

            showing = self._highlight_endpoints(sources, targets)

            self.menu.evidence_label.setText(
                described
                + f": {showing}. Sources: "
                + f"{len(source_neurons)} neurons, "
                + f"{sum(c.session_id is not None for c in sources)} "
                + "concrete footprints."
            )
            return

        # Existing display behavior for the other matching modes.
        self.menu.set_target_inspection(None)

        if source_type == "condition":
            components = self.current_result.components_for_condition(
                neuron_id,
                source_id,
            )
        elif source_type == "group":
            components = self.current_result.components_for_group(
                neuron_id,
                source_id,
            )
        else:
            raise ValueError(source_type)

        self.state.update_highlighted_components(
            self._display_components(components) or None
        )

        self.menu.set_evidence_status(neuron_id, len(components))

    def _show_condition_evidence(self, condition_id: str):
        if self.current_result is None:
            return

        self._evidence_source = ("condition", condition_id)
        self.menu.set_evidence_source(*self._evidence_source)

        self._update_evidence_highlight()

    def _show_group_evidence(self, group_id: str):

        if self.current_result is None:
            return

        self._evidence_source = ("group", group_id)
        self.menu.set_evidence_source(*self._evidence_source)

        self._update_evidence_highlight()

    def _move_filter_node(self, node_id: str, target_group_id: str, target_index: int):

        source = self._find_node_parent(self.root_filter, node_id)

        target_group = self._find_group(target_group_id)

        if source is None or target_group is None:
            return

        source_group, source_index = source
        node = source_group.children[source_index]

        # A group cannot be put inside itself
        # or any of its descendants.
        if is_filter_group(node):

            if self._group_contains_group(node, target_group.id):
                return

        # Remove first.
        source_group.children.pop(source_index)

        # Moving downward inside the same group shifts
        # the requested insertion point by one.
        if source_group is target_group and source_index < target_index:
            target_index -= 1

        target_index = max(
            0,
            min(int(target_index), len(target_group.children)),
        )

        target_group.children.insert(target_index, node)

        self._mark_dirty()
        self._rerender()

    def _find_node_parent(self, group: CurationFilterGroup, node_id: str):

        for index, child in enumerate(group.children):

            if child.id == node_id:
                return group, index

            if is_filter_group(child):

                found = self._find_node_parent(child, node_id)

                if found is not None:
                    return found

        return None

    def _group_contains_group(self, group: CurationFilterGroup, group_id: str) -> bool:

        if group.id == group_id:
            return True

        for child in group.children:

            if is_filter_group(child) and self._group_contains_group(child, group_id):
                return True

        return False

    def _on_statistic_availability_changed(self, *args):
        event = args[0] if args else None

        # Results, inspection and pending actions refer to session indices
        # of the old order (appending sessions keeps them valid), or may
        # contain footprints of a now inactive session.
        if (
            isinstance(event, DataChange)
            and event.has(
                ChangeKind.SESSION_REMOVED,
                ChangeKind.SESSION_ORDER,
                ChangeKind.SESSION_ACTIVITY,
            )
            and (self.current_result is not None or self._evaluation_token is not None)
        ):
            self._clear_results()
            return

        # Model recipes now resolve to other values than evaluated.
        if (
            isinstance(event, DataChange)
            and event.has(ChangeKind.MODEL_PARAMETERS)
            and self.current_result is not None
            and self._uses_model_references(self.root_filter)
        ):
            self._results_note = "Model changed since evaluation."

        self._update_filter_validity()

    def _uses_model_references(self, node):
        if is_filter_group(node):
            return any(self._uses_model_references(child) for child in node.children)

        return any(
            isinstance(reference, ModelDistanceReference)
            for reference in (node.threshold.reference, node.threshold.upper_reference)
        )

    def _update_filter_validity(self, *, preserve_status=False):
        invalid_groups = empty_filter_group_ids(self.root_filter)
        self.menu.set_invalid_groups(invalid_groups)

        invalid_conditions = self.menu.refresh_condition_availability()

        problems = []

        if invalid_groups:
            problems.append(f"{len(invalid_groups)} empty group(s)")

        if invalid_conditions:
            problems.append(
                f"{len(invalid_conditions)} condition(s) " "with unavailable statistics"
            )

        valid = not problems
        evaluating = self._evaluation_token is not None

        if not preserve_status:
            if evaluating:
                self.menu.status_label.setText(
                    "Evaluating… Previous results remain available."
                )
            elif problems:
                self.menu.set_dirty(
                    True,
                    "Filter incomplete — " + "; ".join(problems),
                )
            elif self.current_result is not None:
                self.menu.set_dirty(
                    bool(self._results_note),
                    self._results_note,
                )
            else:
                self.menu.set_dirty(True, "Evaluate to obtain results.")

        self.menu.evaluate_button.setEnabled(
            valid and not evaluating and self.actions.queue is None
        )

        self._refresh_results()
        return valid

    def _load_filter_preset(self, key: str):

        self.filter_store.set_working_preset(key)

        self.root_filter = self.filter_store.working_filter

        self._clear_results()

        self._evidence_source = (
            "group",
            self.root_filter.id,
        )

        self.state.update_highlighted_components(None)

        self.menu.set_evidence_status(None)

        self._rerender()

    def _select_filter_preset(self, key: str):

        if key == self.filter_store.working_preset_key:
            return

        if not self._confirm_discard_filter_changes():
            self._refresh_preset_controls()
            return

        if key == NEW_PRESET_KEY:
            self._new_filter_preset()
            return

        self._load_filter_preset(key)

    def _new_filter_preset(self):
        """Empty, unsaved filter (save it with "Save as…")."""
        self.filter_store.new_working_preset()

        self.root_filter = self.filter_store.working_filter

        self._clear_results()

        self._evidence_source = (
            "group",
            self.root_filter.id,
        )

        self.state.update_highlighted_components(None)
        self.menu.set_evidence_status(None)

        self._rerender()
        self._refresh_preset_controls()

    def _set_filter_preset_category(self, category):
        key = self.filter_store.working_preset_key

        if key is None:
            return

        try:
            self.filter_store.set_preset_category(key, category)
        except Exception as exc:
            QMessageBox.warning(self.menu, "Could not change category", str(exc))

        self._refresh_preset_controls()

    def _ask_preset_name_and_category(self, title, name="", *, ignore_key=None):
        """(name, category) or None; the category defaults to the current
        preset's, else the one suggested by the filter's manipulations.
        Names of other presets (``ignore_key`` excepted) are refused."""
        info = self.filter_store.preset_info(self.filter_store.working_preset_key)
        category = (
            info.category
            if info is not None and info.category is not None
            else suggest_category(self.root_filter)
        )

        dialog = QDialog(self.menu)
        dialog.setWindowTitle(title)
        form = QFormLayout(dialog)

        name_edit = QLineEdit(name)
        form.addRow("Preset name:", name_edit)

        category_box = QComboBox()
        for key, label in [*PRESET_CATEGORIES.items(), (None, UNCATEGORIZED_LABEL)]:
            category_box.addItem(label, key)
            category_box.setItemData(
                category_box.count() - 1,
                PRESET_CATEGORY_DESCRIPTIONS.get(key, ""),
                Qt.ToolTipRole,
            )
        category_box.setCurrentIndex(max(0, category_box.findData(category)))
        form.addRow("Category:", category_box)

        error = QLabel("")
        error.setStyleSheet("color: #ffb36b;")
        error.setWordWrap(True)
        error.hide()
        form.addRow(error)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )

        def accept():
            text = name_edit.text().strip()

            if not text:
                message = "Enter a name."
            else:
                conflict = self.filter_store.name_conflict(text, ignore_key=ignore_key)
                message = (
                    None
                    if conflict is None
                    else f"The name is already used by {conflict.name!r}"
                    + (" (CATAN preset)" if conflict.source == "builtin" else "")
                    + ". Choose another name."
                )

            if message is None:
                dialog.accept()
            else:
                error.setText(message)
                error.show()

        buttons.accepted.connect(accept)
        buttons.rejected.connect(dialog.reject)
        name_edit.textChanged.connect(lambda _: error.hide())
        form.addRow(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None

        name = name_edit.text().strip()

        if not name:
            return None

        return name, category_box.currentData()

    def _refresh_preset_controls(self):

        self.menu.set_preset_state(
            self.filter_store.presets(),
            current_key=(self.filter_store.working_preset_key),
            default_key=(self.filter_store.default_preset_key),
            dirty=(self.filter_store.working_dirty),
            can_edit_category=self.filter_store.can_edit_category(
                self.filter_store.working_preset_key
            ),
        )

    def _confirm_discard_filter_changes(self) -> bool:

        if not self.filter_store.working_dirty:
            return True

        answer = QMessageBox.question(
            self.menu,
            "Discard changes?",
            "The current filter has unsaved " "changes. Discard them?",
            (QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel),
            QMessageBox.StandardButton.Cancel,
        )

        return answer == QMessageBox.StandardButton.Discard

    def _save_filter_preset(self):

        try:
            self.filter_store.save_working()

        except Exception as exc:
            QMessageBox.warning(
                self.menu,
                "Could not save preset",
                str(exc),
            )
            return

        self._refresh_preset_controls()

    def _save_filter_preset_as(self):

        current = self.filter_store.working_preset_key

        # Saving under the current user preset's own name equals "Save".
        answer = self._ask_preset_name_and_category(
            "Save filter preset",
            ignore_key=current if current is not None and current.startswith("user:") else None,
        )

        if answer is None:
            return

        name, category = answer

        try:
            self.filter_store.save_working_as(name, overwrite=True, category=category)

        except Exception as exc:

            QMessageBox.warning(self.menu, "Could not save preset", str(exc))

            return

        self._refresh_preset_controls()

    def _save_builtin_filter_preset(self):
        key = self.filter_store.working_preset_key
        info = self.filter_store.preset_info(key)

        # The working preset itself may be replaced (a user preset moves).
        answer = self._ask_preset_name_and_category(
            "Save as CATAN preset",
            info.name if info is not None else "",
            ignore_key=key,
        )

        if answer is None:
            return

        name, category = answer

        try:
            self.filter_store.save_working_as_builtin(
                name,
                overwrite=info is not None and self.filter_store._slug(name)
                == self.filter_store._split_key(key)[1],
                category=category,
            )

        except Exception as exc:
            QMessageBox.warning(
                self.menu,
                "Could not save CATAN preset",
                str(exc),
            )
            return

        self._refresh_preset_controls()

    def _revert_filter_preset(self):

        key = self.filter_store.working_preset_key

        if key is None:
            return

        if not self._confirm_discard_filter_changes():
            return

        self._load_filter_preset(key)

    def _set_filter_preset_default(self):

        key = self.filter_store.working_preset_key

        if key is None:
            return

        self.filter_store.set_default_preset(key)

        self._refresh_preset_controls()

    def _rename_filter_preset(self):

        key = self.filter_store.working_preset_key

        if key is None:
            return

        info = self.filter_store.preset_info(key)

        if info is None or not info.writable:
            return

        name, accepted = QInputDialog.getText(
            self.menu,
            "Rename filter preset",
            "Preset name:",
            text=info.name,
        )

        if not accepted:
            return

        name = name.strip()

        if not name or name == info.name:
            return

        try:
            self.filter_store.rename_user_preset(key, name)

        except Exception as exc:
            QMessageBox.warning(
                self.menu,
                "Could not rename preset",
                str(exc),
            )
            return

        self._refresh_preset_controls()

    def _delete_filter_preset(self):

        key = self.filter_store.working_preset_key

        if key is None:
            return

        info = self.filter_store.preset_info(key)

        if info is None or not self.filter_store.can_edit_category(key):
            return

        answer = QMessageBox.question(
            self.menu,
            "Delete filter preset",
            f"Delete preset {info.name!r}?"
            + (
                "\n\nThis is a built-in CATAN preset: its file in the "
                "package is deleted."
                if not info.writable
                else ""
            ),
            (QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel),
            QMessageBox.StandardButton.Cancel,
        )

        if answer != QMessageBox.StandardButton.Yes:
            return

        self.filter_store.delete_user_preset(key)

        self._refresh_preset_controls()
