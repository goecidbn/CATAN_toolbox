"""Global maximum tooltip width.

Qt shows plain-text tooltips on a single line per text line, however long.
Every tooltip (widget, menu, item view or QToolTip.showText) is displayed by
Qt's tooltip label (QTipLabel, recreated for each tooltip); its lines
are wrapped here to TOOLTIP_MAX_WIDTH whenever the label is resized for new
text. Rich-text tooltips are word-wrapped by Qt within the same width.
"""

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QApplication, QLabel

# Logical pixels.
TOOLTIP_MAX_WIDTH = 520


def wrap_text(text, metrics, max_width=TOOLTIP_MAX_WIDTH):
    """Plain text with lines wider than ``max_width`` broken at spaces
    (or inside words longer than a line, e.g. paths)."""
    lines = []

    for line in text.split("\n"):
        if metrics.horizontalAdvance(line) <= max_width:
            lines.append(line)
            continue

        current = ""
        for word in line.split(" "):
            candidate = f"{current} {word}" if current else word
            if metrics.horizontalAdvance(candidate) <= max_width:
                current = candidate
                continue

            if current:
                lines.append(current)
            current = word

            # A single word wider than the line: break it by characters.
            while metrics.horizontalAdvance(current) > max_width:
                cut = len(current) - 1
                while cut > 1 and metrics.horizontalAdvance(current[:cut]) > max_width:
                    cut -= 1
                lines.append(current[:cut])
                current = current[cut:]

        lines.append(current)

    return "\n".join(lines)


class _TooltipLabelFilter(QObject):
    def eventFilter(self, label, event):
        if event.type() in (QEvent.Type.Show, QEvent.Type.Resize):
            self._limit_width(label)
        return False

    @staticmethod
    def _limit_width(label):
        if label.width() <= TOOLTIP_MAX_WIDTH + 2 * label.margin() + 16:
            return

        text = label.text()

        if Qt.mightBeRichText(text):
            label.setWordWrap(True)
            width = TOOLTIP_MAX_WIDTH
            label.resize(width, label.heightForWidth(width))
        else:
            wrapped = wrap_text(text, label.fontMetrics())
            if wrapped == text:
                return
            label.setText(wrapped)
            label.resize(label.sizeHint())

        # Stay on screen after shrinking/growing.
        screen = label.screen()
        if screen is not None:
            area = screen.availableGeometry()
            geometry = label.geometry()
            x = min(geometry.x(), area.right() - geometry.width())
            y = min(geometry.y(), area.bottom() - geometry.height())
            label.move(max(area.left(), x), max(area.top(), y))


class _TooltipWatcher(QObject):
    """Attaches the width filter to each new tooltip label."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.label_filter = _TooltipLabelFilter(self)

    def eventFilter(self, obj, event):
        # The object name is set only after polishing.
        if (
            event.type() == QEvent.Type.Polish
            and isinstance(obj, QLabel)
            and obj.metaObject().className() == "QTipLabel"
        ):
            obj.installEventFilter(self.label_filter)
        return False


def install_tooltip_wrapping(app=None):
    """Install once per QApplication."""
    app = app or QApplication.instance()
    if app is None or getattr(app, "_catan_tooltip_watcher", None) is not None:
        return

    watcher = _TooltipWatcher(app)
    app.installEventFilter(watcher)
    app._catan_tooltip_watcher = watcher
