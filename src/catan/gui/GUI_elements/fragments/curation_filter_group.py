from typing import Callable
from html import escape

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QComboBox,
    QToolButton,
    QSizePolicy,
    QCheckBox,
    QSpinBox,
    QLabel,
    QPlainTextEdit,
)

from catan.gui.data.curation_filter import (
    CurationFilterCondition,
    CurationFilterGroup,
    is_filter_condition,
    is_filter_group,
    CountSpec,
)
from .curation_filter_chip import (
    CurationFilterChip,
    CurationFilterDragHandle,
    CURATION_NODE_MIME,
)


class CurationFilterGroupWidget(QFrame):

    condition_edit_requested = Signal(str)
    condition_remove_requested = Signal(str)

    condition_add_requested = Signal(str)
    subgroup_add_requested = Signal(str)

    operator_changed = Signal(str, str)
    match_level_changed = Signal(str, str)

    group_remove_requested = Signal(str)

    condition_evidence_requested = Signal(str)
    group_evidence_requested = Signal(str)

    node_move_requested = Signal(
        str,  # node_id
        str,  # target_group_id
        int,  # insertion index
    )

    count_changed = Signal(str, object)
    comment_changed = Signal(str, str)
    binding_changed = Signal(str, str)

    def __init__(
        self,
        group: CurationFilterGroup,
        *,
        format_condition: Callable[
            [CurationFilterCondition],
            tuple[str, str | None],
        ],
        is_root: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Maximum,
        )

        self.group = group
        self.format_condition = format_condition
        self.is_root = is_root

        self.setFrameShape(QFrame.Shape.StyledPanel)

        self._drop_indicator = QFrame(self)
        self._drop_indicator.setFixedHeight(2)
        self._drop_indicator.setStyleSheet("background-color: #d89a45;")
        self._drop_indicator.hide()

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(4, 4, 4, 4)
        self.layout.setSpacing(4)

        self._condition_widgets = {}
        self._group_widgets = {}

        self.setAcceptDrops(True)
        self._child_widgets = []

        self._build_header()
        self.comment_edit = QPlainTextEdit(self)
        self.comment_edit.setPlaceholderText(
            "Comment: explain what this group identifies…"
        )
        self.comment_edit.setPlainText(self.group.comment)
        self.comment_edit.setFixedHeight(64)
        self.comment_edit.setTabChangesFocus(True)
        self.layout.addWidget(self.comment_edit)
        self.comment_edit.hide()
        self.comment_button.toggled.connect(self.comment_edit.setVisible)

        self.comment_edit.textChanged.connect(self._on_comment_edited)

        self._build_footer()
        self._build_count_controls()
        self._build_children()

        self.setObjectName("curationFilterGroup")

        self._normal_tooltip = "Click group background to show evidence for this group"
        self.setToolTip(self._normal_tooltip)

        self.setStyleSheet("""
            QFrame#curationFilterGroup[evidenceActive="true"] {
                border: 2px solid #d89a45;
            }
            QFrame#curationFilterGroup[filterInvalid="true"] {
                border: 2px solid #d45b5b;
            }
            QFrame#curationFilterGroup[evaluationError="true"] {
                border: 2px solid #d45b5b;
            }

            QFrame#curationFilterGroupHeader {
                background-color: #343941;

                border: 1px solid #505862;
                border-radius: 5px;
            }

            QFrame#curationFilterGroupHeader
            QToolButton#curationGroupDragHandle {
                color: #cdd2d8;
                background-color: transparent;

                border: none;
                border-right: 1px solid #59616c;

                border-top-left-radius: 4px;
                border-bottom-left-radius: 4px;

                padding: 7px 4px;
            }

            QFrame#curationFilterGroupHeader
            QToolButton#curationGroupDragHandle:hover {
                background-color: #49515c;
            }


            QFrame#curationFilterGroup[filterInvalid="true"]
            QFrame#curationFilterGroupHeader {
                background-color: #3d3438;
            }
        """)
        # QFrame#curationFilterGroup[filterInvalid="true"]
        # QFrame#curationFilterGroupHeader {
        #     background-color: #493338;
        # }

    def _build_header(self):

        self.header_widget = QFrame()
        self.header_widget.setObjectName("curationFilterGroupHeader")

        header_layout = QHBoxLayout(self.header_widget)

        header_layout.setContentsMargins(0, 0, 6, 0)
        header_layout.setSpacing(6)

        if not self.is_root:
            self.drag_handle = CurationFilterDragHandle(
                self.group.id, self.header_widget
            )

            self.drag_handle.setObjectName("curationGroupDragHandle")
            self.drag_handle.setText("⠿")
            self.drag_handle.setFixedWidth(30)

            header_layout.addWidget(self.drag_handle)

        self.operator_selector = QComboBox()
        self.operator_selector.addItem(
            "AND",
            "and",
        )
        self.operator_selector.addItem(
            "OR",
            "or",
        )

        self.match_level_selector = QComboBox()
        self.match_level_selector.addItem(
            "by neuron",
            "neuron",
        )
        self.match_level_selector.addItem(
            "by footprint",
            "footprint",
        )
        self.match_level_selector.addItem(
            "by pair relationship",
            "relationship",
        )

        self.match_level_selector.setToolTip(
            "Relationship: conditions must match the same "
            "source and target.\n"
            "The statistic defines whether each endpoint "
            "is a neuron or a footprint.\n"
            "The Curator selects the target neuron and "
            "can highlight either endpoint."
        )

        self.operator_selector.setCurrentIndex(
            self.operator_selector.findData(self.group.operator)
        )
        self.operator_selector.setObjectName("curationGroupSelector")

        self.match_level_selector.setCurrentIndex(
            self.match_level_selector.findData(self.group.match_level)
        )
        self.match_level_selector.setObjectName("curationGroupSelector")

        header_layout.addWidget(self.operator_selector)
        header_layout.addWidget(self.match_level_selector)

        if not self.is_root:
            self.binding_selector = QComboBox()
            self.binding_selector.addItem("Apply to: parent matches", "same")
            self.binding_selector.addItem("Apply to: source", "source")
            self.binding_selector.addItem("Apply to: target", "target")
            self.binding_selector.addItem(
                "Apply to: between sources", "between_sources"
            )
            self.binding_selector.addItem(
                "Apply to: between targets", "between_targets"
            )

            self.binding_selector.setCurrentIndex(
                self.binding_selector.findData(self.group.apply_to)
            )

            self.binding_selector.setToolTip(
                "Parent matches: ordinary subgroup combination.\n"
                "Source: require the source of each parent "
                "relationship to pass this subgroup.\n"
                "Target: require the target to pass this subgroup.\n"
                "Source/target bindings require an AND "
                "relationship parent.\n"
                "Between sources: compare source neurons or "
                "footprints sharing a target. Use an AND "
                "relationship subgroup without counting. "
                "The parent counts one largest mutually "
                "compatible source set.\n"
                "Between targets: compare partners sharing a source. "
                "The parent count must use the opposite side as anchor. "
                "One largest mutually compatible partner set is retained."
            )
            header_layout.addWidget(self.binding_selector)

            self.binding_selector.currentIndexChanged.connect(
                lambda _: self.binding_changed.emit(
                    self.group.id,
                    self.binding_selector.currentData(),
                )
            )

        self.comment_button = QToolButton(self)
        self.comment_button.setText("ⓘ")
        self.comment_button.setCheckable(True)
        self.comment_button.setAutoRaise(True)
        self.comment_button.setToolTip(
            "<qt>"
            + escape(self.group.comment or "No comment yet.").replace("\n", "<br>")
            + "<br><br>Click to edit or close the comment editor.</qt>"
        )

        header_layout.addWidget(self.comment_button)
        header_layout.addStretch()

        if not self.is_root:

            self.remove_group_button = QToolButton()
            self.remove_group_button.setText("×")
            self.remove_group_button.setToolTip("Remove group")

            header_layout.addWidget(self.remove_group_button)

            self.remove_group_button.clicked.connect(
                lambda: self.group_remove_requested.emit(self.group.id)
            )
            self.remove_group_button.setObjectName("curationRemoveGroupButton")

        self.layout.addWidget(self.header_widget)

        self.operator_selector.currentIndexChanged.connect(
            lambda: self.operator_changed.emit(
                self.group.id,
                self.operator_selector.currentData(),
            )
        )

        self.match_level_selector.currentIndexChanged.connect(
            lambda: self.match_level_changed.emit(
                self.group.id,
                self.match_level_selector.currentData(),
            )
        )

    def _build_count_controls(self):
        layout = QHBoxLayout()

        self.count_enabled = QCheckBox("Count matches")
        self.count_enabled.setChecked(self.group.count is not None)

        spec = self.group.count or CountSpec(
            unit=("neuron" if self.group.match_level == "relationship" else "footprint")
        )

        self.count_anchor = QComboBox()
        self.count_anchor.addItem("per target", "target")
        self.count_anchor.addItem("per source", "source")
        self.count_anchor.setCurrentIndex(self.count_anchor.findData(spec.anchor))
        self.count_anchor.setToolTip(
            "Count opposite endpoints around each source or target. "
            "Source-anchored counts retain the edited target relationships."
        )

        self.count_target = QComboBox()
        self.count_target.addItem("per neuron", "neuron")
        self.count_target.addItem("per target footprint", "footprint")
        self.count_target.setCurrentIndex(self.count_target.findData(spec.target))
        self.count_target.setToolTip(
            "Choose whether the count anchor is one neuron or one footprint."
        )

        self.count_comparison = QComboBox()
        self.count_comparison.addItem("at least", "ge")
        self.count_comparison.addItem("exactly", "eq")
        self.count_comparison.addItem("at most", "le")
        self.count_comparison.setCurrentIndex(
            self.count_comparison.findData(spec.comparison)
        )

        self.count_value = QSpinBox()
        self.count_value.setRange(0, 1_000)
        self.count_value.setValue(spec.value)
        self.count_value.setKeyboardTracking(False)
        self.count_value.setMaximumWidth(50)

        layout.addWidget(self.count_enabled)
        layout.addWidget(self.count_anchor)
        layout.addWidget(self.count_target)
        layout.addWidget(self.count_comparison)
        layout.addWidget(self.count_value)
        layout.addStretch()

        self.layout.addLayout(layout)

        self.count_unit_row = QWidget()
        unit_layout = QHBoxLayout(self.count_unit_row)
        unit_layout.setContentsMargins(0, 0, 0, 0)

        self.count_unit = QComboBox()
        self.count_unit.addItem("source neurons", "neuron")
        self.count_unit.addItem("source footprints", "footprint")
        self.count_unit.setCurrentIndex(self.count_unit.findData(spec.unit))
        self.count_unit.setToolTip(
            "Source neurons: repeated observations of one tracked neuron "
            "count once.\n"
            "Source footprints: each session-specific footprint counts separately."
        )

        unit_layout.addWidget(QLabel("Count distinct"))
        unit_layout.addWidget(self.count_unit)

        self.count_partner_sessions = QCheckBox("Separate partner sessions")
        self.count_partner_sessions.setChecked(spec.separate_partner_sessions)
        self.count_partner_sessions.setToolTip(
            "Count separately for each partner session. "
            "For footprint merging, this prevents combining fragments "
            "from different sessions."
        )

        unit_layout.addWidget(self.count_partner_sessions)

        unit_layout.addStretch()

        self.layout.addWidget(self.count_unit_row)

        self._set_count_controls_enabled()

        self.count_enabled.toggled.connect(self._on_count_changed)
        self.count_target.currentIndexChanged.connect(self._on_count_changed)
        self.count_comparison.currentIndexChanged.connect(self._on_count_changed)
        self.count_value.valueChanged.connect(self._on_count_changed)
        self.count_anchor.currentIndexChanged.connect(self._on_count_changed)
        self.count_partner_sessions.toggled.connect(self._on_count_changed)

        self.count_unit.currentIndexChanged.connect(self._on_count_changed)
        self.match_level_selector.currentIndexChanged.connect(
            lambda: self._set_count_controls_enabled()
        )

    def _set_count_controls_enabled(self):
        enabled = self.count_enabled.isChecked()

        self.count_target.setEnabled(enabled)
        self.count_comparison.setEnabled(enabled)
        self.count_value.setEnabled(enabled)
        pair_mode = self.match_level_selector.currentData() == "relationship"

        self.count_unit_row.setVisible(pair_mode)
        self.count_unit.setEnabled(enabled and pair_mode)

        self.count_anchor.setVisible(pair_mode)
        self.count_anchor.setEnabled(enabled and pair_mode)
        self.count_partner_sessions.setEnabled(enabled and pair_mode)

        self.count_target.setItemText(0, "neuron" if pair_mode else "per neuron")
        self.count_target.setItemText(1, "footprint" if pair_mode else "per footprint")

        partner = "target" if self.count_anchor.currentData() == "source" else "source"

        self.count_unit.setItemText(0, f"{partner} neurons")
        self.count_unit.setItemText(1, f"{partner} footprints")
        self.count_unit.setToolTip(
            "Neurons: repeated observations of one neuron count once. "
            "Footprints: each session-specific observation counts separately."
        )

    def _on_count_changed(self, *_):
        self._set_count_controls_enabled()

        spec = None

        if self.count_enabled.isChecked():
            spec = CountSpec(
                target=self.count_target.currentData(),
                comparison=self.count_comparison.currentData(),
                value=self.count_value.value(),
                unit=self.count_unit.currentData(),
                anchor=(
                    self.count_anchor.currentData()
                    if self.match_level_selector.currentData() == "relationship"
                    else "target"
                ),
                separate_partner_sessions=(
                    self.count_partner_sessions.isChecked()
                    and self.match_level_selector.currentData() == "relationship"
                ),
            )

        self.count_changed.emit(self.group.id, spec)

    def _on_comment_edited(self):
        text = self.comment_edit.toPlainText()

        self.comment_button.setToolTip(
            "<qt>"
            + escape(text or "No comment yet.").replace("\n", "<br>")
            + "<br><br>Click to edit or close the comment editor.</qt>"
        )

        self.comment_changed.emit(self.group.id, text)

    def _build_children(self):

        for child in self.group.children:

            if is_filter_condition(child):

                text, tooltip = self.format_condition(child)

                chip = CurationFilterChip(
                    condition_id=child.id,
                    text=text,
                    tooltip=tooltip,
                )

                self._condition_widgets[child.id] = chip
                self._child_widgets.append(chip)

                chip.edit_requested.connect(self.condition_edit_requested)
                chip.remove_requested.connect(self.condition_remove_requested)
                chip.evidence_requested.connect(self.condition_evidence_requested)

                self.layout.addWidget(chip)

            elif is_filter_group(child):

                widget = CurationFilterGroupWidget(
                    child,
                    format_condition=self.format_condition,
                    is_root=False,
                )
                self._group_widgets[child.id] = widget
                self._child_widgets.append(widget)

                self._forward_group_signals(widget)

                self.layout.addWidget(widget)

            else:
                raise TypeError(f"Unknown filter node: " f"{type(child)!r}")

    def _forward_group_signals(self, widget):

        widget.condition_edit_requested.connect(self.condition_edit_requested)
        widget.condition_remove_requested.connect(self.condition_remove_requested)
        widget.condition_add_requested.connect(self.condition_add_requested)
        widget.subgroup_add_requested.connect(self.subgroup_add_requested)
        widget.operator_changed.connect(self.operator_changed)
        widget.match_level_changed.connect(self.match_level_changed)
        widget.group_remove_requested.connect(self.group_remove_requested)
        widget.condition_evidence_requested.connect(self.condition_evidence_requested)
        widget.group_evidence_requested.connect(self.group_evidence_requested)
        widget.node_move_requested.connect(self.node_move_requested)
        widget.count_changed.connect(self.count_changed)
        widget.comment_changed.connect(self.comment_changed)
        widget.binding_changed.connect(self.binding_changed)

    def _build_footer(self):

        layout = QHBoxLayout()

        add_condition = QToolButton()
        add_condition.setObjectName("curationAddButton")
        add_condition.setText("+ condition")

        add_group = QToolButton()
        add_group.setObjectName("curationAddButton")
        add_group.setText("+ group")

        layout.addWidget(add_condition)
        layout.addWidget(add_group)
        layout.addStretch()

        self.layout.addLayout(layout)

        add_condition.clicked.connect(
            lambda: self.condition_add_requested.emit(self.group.id)
        )

        add_group.clicked.connect(
            lambda: self.subgroup_add_requested.emit(self.group.id)
        )

    def set_evidence_source(self, source_type: str, source_id: str):

        group_active = source_type == "group" and source_id == self.group.id

        self.setProperty("evidenceActive", group_active)

        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

        for condition_id, chip in self._condition_widgets.items():
            chip.set_evidence_active(
                source_type == "condition" and source_id == condition_id
            )

        for subgroup in self._group_widgets.values():
            subgroup.set_evidence_source(source_type, source_id)

    def mouseReleaseEvent(self, event):

        if event.button() == Qt.MouseButton.LeftButton:
            self.group_evidence_requested.emit(self.group.id)
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def _drop_index(self, y: int) -> int:

        for index, widget in enumerate(self._child_widgets):

            if y < widget.geometry().center().y():
                return index

        return len(self._child_widgets)

    def _show_drop_indicator(self, index: int):

        margin = 8

        if self._child_widgets:

            if index < len(self._child_widgets):
                y = self._child_widgets[index].geometry().top() - 3
            else:
                y = self._child_widgets[-1].geometry().bottom() + 3

        else:
            # Just underneath the header controls.
            y = 38

        self._drop_indicator.setGeometry(
            margin, y, max(1, self.width() - 2 * margin), 2
        )

        self._drop_indicator.raise_()
        self._drop_indicator.show()

    def dragEnterEvent(self, event):

        if event.mimeData().hasFormat(CURATION_NODE_MIME):
            event.acceptProposedAction()
            return

        event.ignore()

    def dragMoveEvent(self, event):

        if not event.mimeData().hasFormat(CURATION_NODE_MIME):
            event.ignore()
            return

        index = self._drop_index(int(event.position().y()))

        self._show_drop_indicator(index)

        event.acceptProposedAction()

    def dragLeaveEvent(self, event):

        self._drop_indicator.hide()
        super().dragLeaveEvent(event)

    def dropEvent(self, event):

        self._drop_indicator.hide()

        mime = event.mimeData()

        if not mime.hasFormat(CURATION_NODE_MIME):
            event.ignore()
            return

        node_id = bytes(mime.data(CURATION_NODE_MIME)).decode("utf-8")

        index = self._drop_index(int(event.position().y()))

        self.node_move_requested.emit(node_id, self.group.id, index)

        event.acceptProposedAction()

    def set_invalid_groups(self, invalid_group_ids: set[str]):

        invalid = self.group.id in invalid_group_ids

        self.setProperty("filterInvalid", invalid)

        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

        for subgroup in self._group_widgets.values():
            subgroup.set_invalid_groups(invalid_group_ids)

    def refresh_condition_availability(self, registry):
        """Refresh labels and mark unavailable statistics without rebuilding."""
        invalid_ids = set()

        for child in self.group.children:
            if is_filter_condition(child):
                chip = self._condition_widgets.get(child.id)
                if chip is None:
                    continue

                invalid = child.query.statistic_key not in registry
                if invalid:
                    invalid_ids.add(child.id)

                text, tooltip = self.format_condition(child)
                chip.set_text(text, tooltip=tooltip)

                # Keep the normal tooltip current for evaluation-error handling.
                chip._normal_tooltip = chip.condition_button.toolTip()

                if bool(chip.property("filterInvalid")) != invalid:
                    chip.setProperty("filterInvalid", invalid)
                    chip.style().unpolish(chip)
                    chip.style().polish(chip)
                    chip.update()

            elif is_filter_group(child):
                widget = self._group_widgets.get(child.id)
                if widget is not None:
                    invalid_ids.update(widget.refresh_condition_availability(registry))

        return invalid_ids

    def clear_evaluation_errors(self):

        self.setProperty(
            "evaluationError",
            False,
        )

        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

        for chip in self._condition_widgets.values():
            chip.set_evaluation_error(None)

        for subgroup in self._group_widgets.values():
            subgroup.clear_evaluation_errors()

        self.setToolTip(self._normal_tooltip)

    def set_evaluation_error(
        self,
        node_type: str,
        node_id: str,
        message: str,
    ) -> bool:

        if node_type == "group" and node_id == self.group.id:
            self.setProperty("evaluationError", True)

            self.setToolTip("Evaluation error:\n" + message)

            self.style().unpolish(self)
            self.style().polish(self)
            self.update()

            return True

        if node_type == "condition":

            chip = self._condition_widgets.get(node_id)

            if chip is not None:
                chip.set_evaluation_error(message)
                return True

        for subgroup in self._group_widgets.values():

            if subgroup.set_evaluation_error(node_type, node_id, message):
                return True

        return False
