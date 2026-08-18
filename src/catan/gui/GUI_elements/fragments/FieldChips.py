from PySide6.QtWidgets import (
    QLabel,
    QWidget,
    QHBoxLayout,
    QToolButton,
    QSizePolicy,
    QInputDialog,
)
from PySide6.QtCore import Signal

class FieldChip(QWidget):

    remove_requested = Signal(str)

    def __init__(
        self,
        opt: str,
        parent=None,
    ):
        super().__init__(parent)
        print("FieldChip init", opt)

        if isinstance(opt, str):
            self.label = self.key = opt
        if isinstance(opt, tuple):
            self.label, self.key = opt

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.field_button = QToolButton()
        self.field_button.setText(self.label)
        self.field_button.setChecked(True)

        self.set_name()

        self.remove_button = QToolButton()
        self.remove_button.setText("×")
        self.remove_button.setAutoRaise(True)
        self.remove_button.setToolTip(f"Remove {self.label}")

        self.remove_button.setFixedWidth(18)

        layout.addWidget(self.field_button)
        layout.addWidget(self.remove_button)

        self.remove_button.clicked.connect(
            lambda: self.remove_requested.emit(self.label)
        )

        self.setSizePolicy(
            QSizePolicy.Policy.Maximum,
            QSizePolicy.Policy.Fixed,
        )

    def set_name(self):
        self.field_button.setText(self.label)
        if self.label != self.key:
            self.field_button.setToolTip(f"{self.label} ({self.key})")
        else:
            self.field_button.setToolTip(self.label)
