from typing import Optional
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QToolButton,
    QSizePolicy,
    QInputDialog,
    QMenu,
)
from PySide6.QtCore import Qt, Signal
from catan.core.structures.load_config import FieldSpec


class FieldChip(QWidget):

    rename_requested = Signal(str)
    remove_requested = Signal(str)
    change_path_requested = Signal(str)

    name: str
    field_path: str

    def __init__(
        self,
        name: str,
        spec: Optional[FieldSpec] = None,
        parent=None,
    ):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.field_button = QToolButton()
        self.field_button.setText(name)
        self.field_button.setChecked(True)

        self.field_button.clicked.connect(lambda: self.rename_requested.emit(self.name))

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._open_context_menu)

        self.set(name=name, path=name if spec is None else spec.path)

        self.remove_button = QToolButton()
        self.remove_button.setText("×")
        self.remove_button.setAutoRaise(True)
        self.remove_button.setToolTip(f"Remove {self.name}")

        self.remove_button.setFixedWidth(18)

        layout.addWidget(self.field_button)
        layout.addWidget(self.remove_button)

        self.remove_button.clicked.connect(
            lambda: self.remove_requested.emit(self.name)
        )

        self.setSizePolicy(
            QSizePolicy.Policy.Maximum,
            QSizePolicy.Policy.Fixed,
        )

    def _open_context_menu(self, pos):

        menu = QMenu(self)

        menu.addSeparator()
        menu.addAction(
            "Edit field path...", lambda: self.change_path_requested.emit(self.name)
        )
        menu.exec(self.mapToGlobal(pos))

    def set(self, *, name: Optional[str] = None, path: Optional[str] = None):
        if name is not None:
            self.name = name
        if path is not None:
            self.field_path = path

        self.field_button.setText(self.name)
        if self.name != self.field_path:
            self.field_button.setToolTip(f"{self.name} ({self.field_path})")
        else:
            self.field_button.setToolTip(self.name)
