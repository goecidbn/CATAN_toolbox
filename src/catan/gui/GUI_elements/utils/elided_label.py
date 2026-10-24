"""Fixed-height label for texts of varying length (no layout jumps)."""

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QLabel, QSizePolicy


class ElidedLabel(QLabel):
    """Plain-text label showing exactly ``lines`` lines; each line is elided
    to the available width instead of wrapping. The full text is added to the
    tooltip when something is cut off."""

    def __init__(self, text="", parent=None, *, lines=1):
        super().__init__(parent)
        self._full_text = ""
        self._base_tooltip = ""
        self._lines = max(1, int(lines))

        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setWordWrap(False)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(self._lines * self.fontMetrics().lineSpacing() + 2)
        self.setText(text)

    def setText(self, text):
        self._full_text = "" if text is None else str(text)
        self._refresh()

    def text(self):
        return self._full_text

    def clear(self):
        self.setText("")

    def setToolTip(self, tooltip):
        self._base_tooltip = tooltip or ""
        self._refresh()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == event.Type.FontChange:
            self.setFixedHeight(self._lines * self.fontMetrics().lineSpacing() + 2)
            self._refresh()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()

    def sizeHint(self):
        metrics = self.fontMetrics()
        width = max(
            (metrics.horizontalAdvance(line) for line in self._full_text.split("\n")),
            default=0,
        )
        return QSize(width + 4, self.height())

    def minimumSizeHint(self):
        return QSize(1, self.height())

    def _refresh(self):
        metrics = self.fontMetrics()
        width = max(1, self.contentsRect().width())

        lines = self._full_text.split("\n")
        shown = lines[: self._lines]
        if len(lines) > self._lines:
            shown[-1] = shown[-1] + " …"
        shown = [
            metrics.elidedText(line, Qt.TextElideMode.ElideRight, width)
            for line in shown
        ]

        displayed = "\n".join(shown)
        QLabel.setText(self, displayed)

        cut = displayed != self._full_text
        tooltip = self._base_tooltip
        if cut:
            tooltip = (tooltip + "\n\n" if tooltip else "") + self._full_text
        QLabel.setToolTip(self, tooltip)
