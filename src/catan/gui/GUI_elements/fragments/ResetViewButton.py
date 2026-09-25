from PySide6.QtWidgets import QToolButton, QStyle
from typing import Tuple

OVERLAY_BUTTON_STYLE = """
QToolButton {
    background-color: rgba(43, 49, 57, 225);
    color: white;
    border: 1px solid rgba(190, 200, 212, 210);
    border-radius: 5px;
    padding: 3px;
}
QToolButton:hover {
    background-color: rgba(66, 80, 98, 245);
    border-color: white;
}
QToolButton:pressed {
    background-color: rgba(30, 39, 51, 255);
}
QToolButton:disabled {
    background-color: rgba(43, 49, 57, 140);
    border-color: rgba(190, 200, 212, 100);
}
"""


class ResetViewButton(QToolButton):

    def __init__(self, native, pos: Tuple[int, int], callback):
        super().__init__(native)
        self.position = pos

        self.setIcon(
            native.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload)
        )

        self.setToolTip("Reset view")

        self.setAutoRaise(True)
        self.setStyleSheet(OVERLAY_BUTTON_STYLE)
        self.setFixedSize(28, 28)

        self.clicked.connect(callback)

        self._reposition()

    def _reposition(self):

        x = self.position[0]
        y = (self.position[1] - self.height()) // 2

        self.move(x, y)
