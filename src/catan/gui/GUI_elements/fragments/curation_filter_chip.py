from html import escape
from PySide6.QtCore import Qt, Signal, QMimeData, QPoint
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import (
    QWidget,
    QFrame,
    QHBoxLayout,
    QToolButton,
    QSizePolicy,
    QMenu,
    QApplication,
    QToolButton,
)

CURATION_NODE_MIME = "application/x-catan-curation-filter-node"


class CurationFilterDragHandle(QToolButton):

    def __init__(
        self,
        node_id: str,
        parent=None,
    ):
        super().__init__(parent)

        self.node_id = node_id
        self._press_pos = None

        # self.setText("⋮⋮")
        self.setText("⠿")
        self.setToolTip("Drag to move")
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mousePressEvent(self, event):

        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position().toPoint()

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):

        if self._press_pos is None or not (event.buttons() & Qt.MouseButton.LeftButton):
            return super().mouseMoveEvent(event)

        distance = (event.position().toPoint() - self._press_pos).manhattanLength()

        if distance < QApplication.startDragDistance():
            return

        mime = QMimeData()
        mime.setData(
            CURATION_NODE_MIME,
            self.node_id.encode("utf-8"),
        )

        drag = QDrag(self)
        drag.setMimeData(mime)

        self.setCursor(Qt.CursorShape.ClosedHandCursor)

        try:
            drag.exec(Qt.DropAction.MoveAction)
        finally:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self._press_pos = None


class CurationFilterChip(QFrame):

    edit_requested = Signal(str)
    remove_requested = Signal(str)
    evidence_requested = Signal(str)

    def __init__(
        self,
        condition_id: str,
        text: str,
        *,
        tooltip: str | None = None,
        parent=None,
    ):
        super().__init__(parent)

        self.setObjectName("curationConditionChip")

        self.condition_id = condition_id

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.drag_handle = CurationFilterDragHandle(condition_id, self)
        self.drag_handle.setObjectName("curationConditionDragHandle")
        self.drag_handle.setFixedWidth(20)
        layout.addWidget(self.drag_handle)

        self.condition_button = QToolButton()
        self.condition_button.setObjectName("curationConditionButton")
        self.condition_button.setText(text)

        self.remove_button = QToolButton()
        self.remove_button.setObjectName("curationConditionRemoveButton")
        self.remove_button.setText("×")
        self.remove_button.setAutoRaise(True)
        self.remove_button.setFixedWidth(20)
        self.remove_button.setToolTip("Remove condition")

        layout.addWidget(self.condition_button)
        layout.addWidget(self.remove_button)

        self.condition_button.clicked.connect(
            lambda: self.evidence_requested.emit(self.condition_id)
        )

        self.remove_button.clicked.connect(
            lambda: self.remove_requested.emit(self.condition_id)
        )

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._open_context_menu)

        self.setSizePolicy(
            QSizePolicy.Policy.Maximum,
            QSizePolicy.Policy.Fixed,
        )

        self.set_text(
            text,
            tooltip=tooltip,
        )

        self.setStyleSheet("""
            QFrame#curationConditionChip {
                background-color: #343941;
                border: 1px solid #59616c;
                border-radius: 5px;
            }

            QFrame#curationConditionChip[evidenceActive="true"] {
                border: 2px solid #d89a45;
            }

            QFrame#curationConditionChip[evaluationError="true"] {
                border: 2px solid #d45b5b;
                background-color: #463438;
            }

            QFrame#curationConditionChip[filterInvalid="true"] {
                border: 2px solid #d45b5b;
                background-color: #463438;
            }

            QFrame#curationConditionChip QToolButton {
                color: #e8eaed;
                background: transparent;
                border: none;
                border-radius: 0px;
                padding: 1px 6px;
            }

            QFrame#curationConditionChip
            QToolButton#curationConditionDragHandle {
                border-right: 1px solid #59616c;
                padding-left: 4px;
                padding-right: 4px;
            }

            QFrame#curationConditionChip
            QToolButton#curationConditionButton {
                text-align: left;
            }

            QFrame#curationConditionChip
            QToolButton#curationConditionRemoveButton {
                border-left: 1px solid #59616c;
                padding-left: 4px;
                padding-right: 4px;
            }

            QFrame#curationConditionChip
            QToolButton#curationConditionDragHandle:hover,

            QFrame#curationConditionChip
            QToolButton#curationConditionRemoveButton:hover {
                background-color: #444b55;
            }
        """)

    def set_text(
        self,
        text: str,
        *,
        tooltip: str | None = None,
    ):
        self.condition_button.setText(text)

        if tooltip is not None:
            self._normal_tooltip = (
                tooltip + "\n\nClick: show evidence" + "\nRight-click: edit/remove"
            )
            self._refresh_condition_tooltip()

    def _refresh_condition_tooltip(self):
        text = getattr(self, "_normal_tooltip", "")
        error = getattr(self, "_evaluation_error", None)

        if error:
            text += f"\n\nEvaluation error:\n{error}"

        body = escape(text).replace("\n", "<br>")

        self.condition_button.setToolTip(
            f'<qt><table width="440" cellspacing="0" cellpadding="4">'
            f"<tr><td>{body}</td></tr>"
            f"</table></qt>"
        )

    def _open_context_menu(self, pos):

        menu = QMenu(self)
        menu.addAction("Edit...", lambda: self.edit_requested.emit(self.condition_id))
        menu.addAction("Remove", lambda: self.remove_requested.emit(self.condition_id))

        menu.exec(self.mapToGlobal(pos))

    def set_evidence_active(self, active: bool):
        self.setProperty("evidenceActive", active)

        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def set_evaluation_error(
        self,
        message: str | None,
    ):
        self._evaluation_error = message
        self.setProperty("evaluationError", message is not None)

        self._refresh_condition_tooltip()

        self.style().unpolish(self)
        self.style().polish(self)
        self.update()
