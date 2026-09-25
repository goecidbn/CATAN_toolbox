from PySide6.QtWidgets import QToolButton, QStyle
from typing import Tuple


class SaveImageButton(QToolButton):

    def __init__(self, native, pos: Tuple[int, int], callback):
        super().__init__(native)
        self.position = pos

        self.setIcon(
            native.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton)
        )

        self.setToolTip("Save image")

        self.setAutoRaise(True)
        self.setFixedSize(28, 28)

        self.clicked.connect(callback)

        self._reposition()

    def _reposition(self):

        x = self.position[0]
        y = (self.position[1] - self.height()) // 2

        self.move(x, y)
