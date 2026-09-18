from typing import Callable

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QComboBox,
    QToolButton,
    QSizePolicy,
)

from catan.gui.data.curation_filter import (
    CurationFilterCondition,
    CurationFilterGroup,
    is_filter_condition,
    is_filter_group,
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
        self.layout.setContentsMargins(6, 6, 6, 6)
        self.layout.setSpacing(6)

        self._condition_widgets = {}
        self._group_widgets = {}

        self.setAcceptDrops(True)
        self._child_widgets = []

        self._build_header()
        self._build_children()
        self._build_footer()

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
