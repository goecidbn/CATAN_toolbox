import importlib
from copy import deepcopy

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
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
from catan.gui.panels import StatisticsData
from catan.gui.panels.helper.Threshold import ThresholdSpec
from catan.gui.data.curation_filter import CurationFilterCondition
from catan.gui.data.curation_results import RetainedResults
from catan.gui.GUI_elements.fragments.curation_filter_group import (
    CurationFilterGroupWidget,
)
from catan.gui.GUI_elements.fragments.curation_actions_widget import (
    CuratorActions,
)
from catan.gui.data.curation_filter_presets import CurationFilterPresetStore
from catan.gui.panels.helper.ReviewStatusFilter import (
    ReviewStatusFilter,
)
from catan.gui.GUI_elements.fragments.curator_interaction_guard import (
    CuratorInteractionGuard,
)

from catan.gui.panels import BasePlot

importlib.reload(StatisticsData)


class CurationFilterConditionPopup(QDialog):

    conditionAccepted = Signal(object)

    def __init__(
        self,
        *,
        engine,
        condition: CurationFilterCondition | None = None,
        parent=None,
    ):
        super().__init__(
            parent,
            Qt.WindowType.Popup,
        )

        self.engine = engine
        self.condition = condition

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

        threshold_layout = QHBoxLayout(threshold_group)

        threshold_layout.addWidget(QLabel("Match when:"))

        # ... direction selector + spin box

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

        layout.addWidget(threshold_group)
        threshold_layout.addWidget(self.direction_selector)
        threshold_layout.addWidget(self.threshold_value, stretch=1)

        threshold_layout.addWidget(self.interval_separator)
        threshold_layout.addWidget(self.threshold_upper, stretch=1)

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
        self.threshold_value.valueChanged.connect(self._update_threshold_controls)
        self.threshold_upper.valueChanged.connect(self._update_threshold_controls)

        self._update_threshold_controls()

    def _on_query_changed(self, query):
        error = ""

        try:
            self._threshold_spec().validate()
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

        self.threshold_value.setValue(condition.threshold.value)
        self.threshold_upper.setValue(
            condition.threshold.upper_value
            if condition.threshold.upper_value is not None
            else max(1.0, condition.threshold.value)
        )

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

        return ThresholdSpec(
            axis=(None if self.condition is None else self.condition.threshold.axis),
            value=self.threshold_value.value(),
            direction=direction,
            active=True,
            upper_value=(
                self.threshold_upper.value() if direction == "between" else None
            ),
        )

    def _update_threshold_controls(self, *_):
        interval = self.direction_selector.currentData() == "between"

        self.interval_separator.setVisible(interval)
        self.threshold_upper.setVisible(interval)

        self.threshold_value.setToolTip(
            "Inclusive lower bound" if interval else "Threshold value"
        )

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

    binding_changed = Signal(str, str)

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

        self.target_selector.currentIndexChanged.connect(
            lambda: self.inspection_changed.emit()
        )
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

        threshold = condition.threshold
        text = threshold.expression(query_title, precision=4)

        lines = [
            query_title,
            "",
            f"Threshold: {threshold.expression(precision=6)}",
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
                else:
                    value_text = f"{float(value):.6g}"

                lines.append(f"  {parameter.label}: {value_text}")

        return text, "\n".join(lines)

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

    def set_target_inspection(self, mapping):
        previous = self.target_selector.currentData()

        targets = sorted(
            mapping or {},
            key=lambda c: (c.neuron_id, -1 if c.session_id is None else c.session_id),
        )

        blocked = self.target_selector.blockSignals(True)

        try:
            self.target_selector.clear()

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
                index = targets.index(previous) if previous in targets else 0
                self.target_selector.setCurrentIndex(index)

        finally:
            self.target_selector.blockSignals(blocked)

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

    def set_preset_state(
        self,
        presets,
        *,
        current_key: str | None,
        default_key: str | None,
        dirty: bool,
    ):

        self.preset_selector.blockSignals(True)

        try:
            self.preset_selector.clear()

            selected_index = -1

            for preset in presets:

                text = preset.name
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

        self.preset_save_action.setEnabled(writable and dirty)
        self.preset_revert_action.setEnabled(current_info is not None and dirty)
        self.preset_default_action.setEnabled(current_info is not None)
        self.preset_rename_action.setEnabled(writable)
        self.preset_delete_action.setEnabled(writable)

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

    def _set_group_binding(self, group_id: str, binding: str):
        if binding not in (
            "same",
            "source",
            "target",
            "between_sources",
            "between_targets",
        ):
            return
            # raise ValueError(binding)

        group = self._find_group(group_id)

        if group is None or group.apply_to == binding:
            return

        group.apply_to = binding
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

        generation = self._filter_generation
        version = self.state.data_version
        generation = self._filter_generation
        snapshot = deepcopy(self.root_filter)

        self.menu.set_evaluating()

        def evaluate():

            try:
                result = CurationFilterEvaluator(self.data.statistic_engine).evaluate(
                    snapshot
                )
                return result, None

            except CurationFilterError as exc:
                return None, (str(exc), exc.node_type, exc.node_id)

            except Exception as exc:
                return None, (f"{type(exc).__name__}: {exc}", None, None)

        self.state.tasks.start(
            "calculating",
            "Evaluate curation filter",
            evaluate,
            on_result=lambda result: (
                self._on_filter_evaluated(
                    result,
                    generation=generation,
                    version=version,
                )
            ),
        )

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
            self._update_filter_validity()
            return

        if error is not None:

            message, node_type, node_id = error

            self.menu.set_evaluation_error(
                message, node_type=node_type, node_id=node_id
            )

            return

        self.current_result = RetainedResults(result, self.data)
        self._results_note = ""

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
                if session is None:
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

            sources = mapping[target]
            source_neurons = {c.neuron_id for c in sources}

            showing = self._highlight_endpoints(sources, {target})

            self.menu.evidence_label.setText(
                f"Target n{target.neuron_id}"
                + (
                    ""
                    if target.session_id is None
                    else f", session {target.session_id}"
                )
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
        self._update_filter_validity()

    def _update_filter_validity(self):
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

        if problems:
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

        self.menu.evaluate_button.setEnabled(valid and self.actions.queue is None)

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

        self._load_filter_preset(key)

    def _refresh_preset_controls(self):

        self.menu.set_preset_state(
            self.filter_store.presets(),
            current_key=(self.filter_store.working_preset_key),
            default_key=(self.filter_store.default_preset_key),
            dirty=(self.filter_store.working_dirty),
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

        name, accepted = QInputDialog.getText(
            self.menu, "Save filter preset", "Preset name:"
        )

        if not accepted:
            return

        name = name.strip()

        if not name:
            return

        try:
            self.filter_store.save_working_as(name, overwrite=True)

        except FileExistsError:

            overwrite = QMessageBox.question(
                self.menu,
                "Preset already exists",
                f"A preset named {name!r} " "already exists. Overwrite it?",
                (QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No),
                QMessageBox.StandardButton.No,
            )

            if overwrite != QMessageBox.StandardButton.Yes:
                return

            # Existing helper already supports overwrite,
            # so use the underlying method here.
            key = self.filter_store.save_user_preset(
                name=name, root=self.root_filter, overwrite=True
            )

            self.filter_store._working_preset_key = key
            self.filter_store._working_dirty = False

        except Exception as exc:

            QMessageBox.warning(self.menu, "Could not save preset", str(exc))

            return

        self._refresh_preset_controls()

    def _save_builtin_filter_preset(self):
        info = self.filter_store.preset_info(self.filter_store.working_preset_key)
        name, accepted = QInputDialog.getText(
            self.menu,
            "Save as CATAN preset",
            "Preset name:",
            text=info.name if info is not None else "",
        )

        if not accepted or not name.strip():
            return

        name = name.strip()

        try:
            try:
                self.filter_store.save_working_as_builtin(name)
            except FileExistsError:
                answer = QMessageBox.question(
                    self.menu,
                    "Replace CATAN preset?",
                    f"A built-in preset with this filename already exists "
                    f"for {name!r}.\n\nReplace it?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return

                self.filter_store.save_working_as_builtin(
                    name,
                    overwrite=True,
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

        if info is None or not info.writable:
            return

        answer = QMessageBox.question(
            self.menu,
            "Delete filter preset",
            f"Delete preset {info.name!r}?",
            (QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel),
            QMessageBox.StandardButton.Cancel,
        )

        if answer != QMessageBox.StandardButton.Yes:
            return

        self.filter_store.delete_user_preset(key)

        self._refresh_preset_controls()
