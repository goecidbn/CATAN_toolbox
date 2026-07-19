from PySide6.QtWidgets import QToolButton, QStyleOptionToolButton, QStyle, QMenu
from PySide6.QtGui import QPainter
from PySide6.QtCore import QSize, Qt


class RotatedToolButton(QToolButton):
    def __init__(self, angle=-90, parent=None):
        super().__init__(parent)
        self.angle = angle

    def sizeHint(self):
        size = super().sizeHint()
        if self.angle in (90, -90, 270):
            return QSize(size.height(), size.width())
        return size

    def minimumSizeHint(self):
        size = super().minimumSizeHint()
        if self.angle in (90, -90, 270):
            return QSize(size.height(), size.width())
        return size

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        opt = QStyleOptionToolButton()
        self.initStyleOption(opt)

        if self.angle == -90 or self.angle == 270:
            painter.translate(0, self.height())
            painter.rotate(-90)
            opt.rect = self.rect().transposed()

        elif self.angle == 90:
            painter.translate(self.width(), 0)
            painter.rotate(90)
            opt.rect = self.rect().transposed()

        else:
            opt.rect = self.rect()

        self.style().drawComplexControl(
            QStyle.ComplexControl.CC_ToolButton,
            opt,
            painter,
            self,
        )
