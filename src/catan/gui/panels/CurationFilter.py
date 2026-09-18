import importlib

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

from catan.gui.structures import NeuronComponent
from catan.gui.panels import StatisticsData
from catan.gui.panels.helper.Threshold import ThresholdSpec
from catan.gui.data.curation_filter import CurationFilterCondition
from catan.gui.GUI_elements.fragments.curation_filter_group import (
    CurationFilterGroupWidget,
)

from catan.gui.data.curation_filter_presets import CurationFilterPresetStore

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

        self.direction_selector.setCurrentIndex(0)

        self.direction_selector.setMinimumWidth(50)
        self.direction_selector.setMaximumWidth(70)
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

        layout.addWidget(threshold_group)
        threshold_layout.addWidget(self.direction_selector)
        threshold_layout.addWidget(self.threshold_value, stretch=1)

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

        self._on_query_changed(self.query_selector.effective_query())

    def _on_query_changed(
        self,
        query,
    ):

        self.accept_button.setEnabled(
            query is not None and query.statistic_key != "none"
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

        self.accept_button.setText("Apply")

        self._on_query_changed(self.query_selector.effective_query())

    def _accept_condition(self):

        query = self.query_selector.effective_query()

        if query is None or query.statistic_key == "none":
            return

        threshold = ThresholdSpec(
            value=self.threshold_value.value(),
            direction=self.direction_selector.currentData(),
            active=True,
        )

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


class Display(QWidget):

    condition_edit_requested = Signal(str)
    condition_remove_requested = Signal(str)
    condition_add_requested = Signal(str)

    subgroup_add_requested = Signal(str)
    group_remove_requested = Signal(str)

    operator_changed = Signal(str, str)
    match_level_changed = Signal(str, str)

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
                padding: 4px 8px;
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
                padding: 3px;
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
                padding: 4px 7px;
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

    def set_filter(
        self,
        root: CurationFilterGroup,
    ):

        if self.root_widget is not None:
            self.filter_layout.removeWidget(self.root_widget)
            self.root_widget.deleteLater()

        self.root_widget = CurationFilterGroupWidget(
            root,
            format_condition=self.format_condition,
            is_root=True,
        )

        self._connect_group_widget(self.root_widget)

        # Insert before the stretch.
        self.filter_layout.insertWidget(
            0,
            self.root_widget,
        )

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

    def _default_format_condition(
        self,
        condition: CurationFilterCondition,
    ):

        # threshold = condition.threshold
        # symbol = "≥" if threshold.direction == "greater" else "≤"

        stat_def = self.data.statistic_engine.registry[condition.query.statistic_key]

        query_title = StatisticsData.format_query_expression(
            condition.query,
            stat_def,
        )

        symbol = "≥" if condition.threshold.direction == "greater" else "≤"
        text = f"{query_title} " f"{symbol} " f"{condition.threshold.value:.4g}"
        tooltip = (
            f"{query_title}\n\n"
            f"Threshold: {symbol} "
            f"{condition.threshold.value:.6g}"
        )

        return text, tooltip

    def set_dirty(
        self, dirty: bool = True, dirty_text: str = "Filter changed — not evaluated"
    ):

        if dirty:
            self.status_label.setText(dirty_text)
            self.result_label.setText("Matching neurons: —")
            self.select_button.setEnabled(False)
        else:
            self.status_label.clear()

    def set_evaluating(self):
        self.clear_evaluation_errors()
        self.status_label.setText("Evaluating...")
        self.result_label.setText("Matching neurons: —")

        self.evaluate_button.setEnabled(False)
        self.select_button.setEnabled(False)

    def set_result_count(self, count: int):
        self.clear_evaluation_errors()
        self.status_label.clear()
        self.result_label.setText(f"Matching neurons: {count}")

        self.evaluate_button.setEnabled(True)
        self.select_button.setEnabled(count > 0)

    def set_evaluation_error(
        self,
        message: str,
        *,
        node_type: str | None = None,
        node_id: str | None = None,
    ):

        self.clear_evaluation_errors()

        self.status_label.setText(f"Evaluation failed: {message}")

        self.result_label.setText("Matching neurons: —")

        self.evaluate_button.setEnabled(True)
        self.select_button.setEnabled(False)

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

    def set_invalid_groups(self, invalid_group_ids: set[str]):

        if self.root_widget is not None:
            self.root_widget.set_invalid_groups(invalid_group_ids)

        self.evaluate_button.setEnabled(not invalid_group_ids)

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
                    tooltip.append("Built-in preset")
                    # text = f"{text}"

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
        self._filter_generation = 0

        self._evidence_source = ("group", self.root_filter.id)

    def build_controls(self):

        self.menu.condition_remove_requested.connect(self._remove_condition)
        self.menu.subgroup_add_requested.connect(self._add_subgroup)
        self.menu.group_remove_requested.connect(self._remove_group)
        self.menu.operator_changed.connect(self._set_group_operator)
        self.menu.match_level_changed.connect(self._set_group_match_level)
        self.menu.evaluate_requested.connect(self._evaluate_filter)
        self.menu.select_requested.connect(self._select_matching_neurons)

        self.menu.condition_add_requested.connect(self._add_condition_requested)
        self.menu.condition_edit_requested.connect(self._edit_condition_requested)
        self.menu.condition_evidence_requested.connect(self._show_condition_evidence)
        self.menu.group_evidence_requested.connect(self._show_group_evidence)

        self.menu.node_move_requested.connect(self._move_filter_node)

        # preset loading / saving
        self.menu.preset_selected.connect(self._select_filter_preset)
        self.menu.preset_save_requested.connect(self._save_filter_preset)
        self.menu.preset_save_as_requested.connect(self._save_filter_preset_as)
        self.menu.preset_revert_requested.connect(self._revert_filter_preset)
        self.menu.preset_set_default_requested.connect(self._set_filter_preset_default)
        self.menu.preset_rename_requested.connect(self._rename_filter_preset)
        self.menu.preset_delete_requested.connect(self._delete_filter_preset)

        self.state.focused_component_changed.connect(self._on_focused_component_changed)
        self._rerender()

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

    def _mark_dirty(self):

        self._filter_generation += 1
        self.current_result = None

        self.filter_store.mark_working_dirty()

        self._evidence_source = ("group", self.root_filter.id)
        self.menu.set_evidence_source(*self._evidence_source)

        self.menu.set_dirty(True)
        self.state.update_highlighted_components(None)
        self.menu.set_evidence_status(None)
        self._refresh_preset_controls()

    def _rerender(self):

        self.menu.set_filter(self.root_filter)
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

    def _evaluate_filter(self):

        generation = self._filter_generation

        self.menu.set_evaluating()

        def evaluate():

            try:
                result = self.evaluator.evaluate(self.root_filter)
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
                )
            ),
        )

    def _on_filter_evaluated(
        self,
        result_and_error,
        *,
        generation: int,
    ):

        result, error = result_and_error

        # Filter was edited while calculation ran.
        if generation != self._filter_generation:
            return

        if error is not None:

            self.current_result = None

            message, node_type, node_id = error

            self.menu.set_evaluation_error(
                message, node_type=node_type, node_id=node_id
            )

            return

        self.current_result = result

        self.menu.set_result_count(len(result.neurons))
        self._update_evidence_highlight()

        # import json
        # from catan.gui.data.curation_filter import (
        #     curation_filter_to_dict,
        # )

        # print(
        #     json.dumps(
        #         curation_filter_to_dict(
        #             self.root_filter,
        #             name="CATAN default",
        #         ),
        #         indent=2,
        #     )
        # )

    def _select_matching_neurons(self):

        if self.current_result is None:
            return

        components = [
            NeuronComponent(
                session_id=None,
                neuron_id=int(neuron_id),
            )
            for neuron_id in sorted(self.current_result.neurons)
        ]

        self.state.update_selected_components(components, [])

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

    def _update_evidence_highlight(self):

        # No evaluated filter yet.
        if self.current_result is None:
            self.menu.set_evidence_status(None)
            return

        focused = self.state.focused_component
        if focused is None:
            self.state.update_highlighted_components(None)
            self.menu.set_evidence_status(None)
            return

        neuron_id = int(focused.neuron_id)

        # Focus may belong to some unrelated neuron.
        if neuron_id not in self.current_result.neurons:
            self.state.update_highlighted_components(None)
            self.menu.set_evidence_status(neuron_id, 0)
            return

        source_type, source_id = self._evidence_source

        if source_type == "condition":
            components = self.current_result.components_for_condition(
                neuron_id, source_id
            )

        elif source_type == "group":
            components = self.current_result.components_for_group(neuron_id, source_id)

        else:
            raise ValueError(source_type)

        condition_result = next(iter(self.current_result.condition_results.values()))

        self.state.update_highlighted_components(
            list(components) if components else None
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

    def _update_filter_validity(self):

        invalid_groups = empty_filter_group_ids(self.root_filter)

        self.menu.set_invalid_groups(invalid_groups)

        if invalid_groups:

            n = len(invalid_groups)

            text = "Filter incomplete — " f"{n} empty group" f"{'s' if n != 1 else ''}"

            self.menu.set_dirty(
                True,
                text,
            )

            return False

        if self.filter_store.working_dirty:
            self.menu.set_dirty(True)
        else:
            self.menu.set_dirty(False)

        return True

    def _load_filter_preset(self, key: str):

        self.filter_store.set_working_preset(key)

        self.root_filter = self.filter_store.working_filter

        self.current_result = None
        self._filter_generation += 1

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
