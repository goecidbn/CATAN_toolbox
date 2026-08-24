from typing import Optional
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QToolButton,
    QSizePolicy,
    QInputDialog,
)
from PySide6.QtCore import Signal
from catan.core.structures.load_config import FieldSpec

class FieldChip(QWidget):

    remove_requested = Signal(str)

    def __init__(
        self,
        name: str,
        spec: Optional[FieldSpec] = None,
        parent=None,
    ):
        super().__init__(parent)

        self.name = name
        if spec is None:
            self.field_path = name
        else:
            self.field_path = spec.path
        

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.field_button = QToolButton()
        self.field_button.setText(self.name)
        self.field_button.setChecked(True)

        self.set_name()

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

    def set_name(self):
        self.field_button.setText(self.name)
        if self.name != self.field_path:
            self.field_button.setToolTip(f"{self.name} ({self.field_path})")
        else:
            self.field_button.setToolTip(self.name)
