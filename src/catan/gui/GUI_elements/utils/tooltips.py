"""Global maximum tooltip width.

Qt shows plain-text tooltips on a single line per text line, however long.
Every tooltip (widget, menu, item view or QToolTip.showText) is displayed by
Qt's tooltip label (QTipLabel, recreated for each tooltip); its lines
are wrapped here to TOOLTIP_MAX_WIDTH whenever the label is resized for new
text. Rich-text tooltips are word-wrapped by Qt within the same width.
"""

from PySide6.QtCore import QEvent, QObject, QPoint, Qt, QTimer
from shiboken6 import isValid
from PySide6.QtGui import QCursor, QFontMetrics, QGuiApplication
from PySide6.QtWidgets import QApplication, QLabel, QProxyStyle, QToolTip

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
    """Wraps a tooltip label before it is shown (Polish / Resize events of
    the still hidden label: text and stylesheet font are set, the window is
    not mapped yet). Changing a
    shown window from its Show / Resize events flickered (the long line
    showed first) and, on X11, left newly exposed lines unpainted.
    A visible label whose text changes (reused tooltip) is wrapped after
    Qt's own resize, then repainted."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pending = set()

    def eventFilter(self, label, event):
        kind = event.type()

        if kind == QEvent.Type.Polish:
            # Polished on showing, after Qt placed it.
            _remember_label(label)
            self._limit_width(label, placed=True)
        elif kind == QEvent.Type.Resize:
            # A new label is sized while hidden, before Qt places it: wrap
            # right away (Qt then places the wrapped size). A visible
            # (reused) label: after Qt's handling.
            if label.isVisible():
                self._schedule(label)
            else:
                self._limit_width(label, placed=False)
        return False

    def _schedule(self, label):
        key = id(label)
        if key in self._pending:
            return
        self._pending.add(key)

        def run():
            self._pending.discard(key)
            if isValid(label):
                self._limit_width(label, placed=True)

        QTimer.singleShot(0, run)

    @staticmethod
    def _limit_width(label, *, placed=True):
        if label.width() <= TOOLTIP_MAX_WIDTH + 2 * label.margin() + 16:
            return

        text = label.text()
        metrics = label.fontMetrics()

        # Keep the label's text unchanged: QToolTip.showText compares it with
        # the requested text, and a changed text re-created the tooltip on
        # every mouse move (flicker of the long line). Only words longer
        # than a line (e.g. paths) need inserted line breaks.
        if Qt.mightBeRichText(text) or all(
            metrics.horizontalAdvance(word) <= TOOLTIP_MAX_WIDTH for word in text.split()
        ):
            label.setWordWrap(True)
            width = TOOLTIP_MAX_WIDTH
            label.resize(width, label.heightForWidth(width))
        else:
            wrapped = wrap_text(text, label.fontMetrics())
            if wrapped == text:
                return
            label.setText(wrapped)
            label.resize(label.sizeHint())

        # Wrapping only narrows the label and makes it taller: keep its
        # bottom on the screen it is on. A label Qt has not placed yet is not
        # moved (its position is not the final one: moving it sent tooltips
        # to a screen corner; it also must not use another screen than the
        # one at its position, which moved tooltips to the other display).
        if placed:
            geometry = label.geometry()
            screen = (
                QGuiApplication.screenAt(geometry.topLeft())
                or QGuiApplication.screenAt(QCursor.pos())
                or label.screen()
            )
            if screen is not None:
                area = screen.availableGeometry()
                if geometry.bottom() > area.bottom():
                    label.move(
                        geometry.x(),
                        max(area.top(), area.bottom() - geometry.height() + 1),
                    )

        label.update()


# Further per-widget setup run once when the style polishes a widget
# (e.g. message_boxes.py); callables taking the widget.
POLISH_HOOKS = []


def _run_polish_hooks(widget):
    for hook in POLISH_HOOKS:
        hook(widget)


def _is_tip_label(obj):
    # The object name is set only after polishing.
    return isinstance(obj, QLabel) and obj.metaObject().className() == "QTipLabel"


class _TooltipStyle(QProxyStyle):
    """Attaches the width filter to each new tooltip label when the style
    polishes it (once per label), instead of filtering every event of the
    application. All drawing is delegated to the wrapped style."""

    def __init__(self, base, label_filter):
        super().__init__(base)
        self.label_filter = label_filter

    def polish(self, arg):
        result = super().polish(arg)
        if _is_tip_label(arg):
            arg.installEventFilter(self.label_filter)
            _remember_label(arg)
        elif POLISH_HOOKS and hasattr(arg, "metaObject"):
            _run_polish_hooks(arg)
        return result


class _TooltipWatcher(QObject):
    """Fallback: attaches the width filter via an application event filter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.label_filter = _TooltipLabelFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Polish:
            if _is_tip_label(obj):
                obj.installEventFilter(self.label_filter)
                _remember_label(obj)
                # The label's own filter misses this Polish event.
                self.label_filter._limit_width(obj, placed=True)
            elif POLISH_HOOKS:
                _run_polish_hooks(obj)
        return False


def _current_style(app):
    """The style the stylesheet is applied on: qdarktheme's proxy style if
    installed (QApplication.style() returns the stylesheet wrapper), else the
    style key of a plain application."""
    try:
        from qdarktheme import _main
    except ImportError:
        _main = None

    style = getattr(_main, "_proxy_style", None)
    if style is not None:
        return style

    if not app.styleSheet():
        return app.style().name() or None
    return None


def install_tooltip_wrapping(app=None):
    """Install once per QApplication (after the theme is set up)."""
    app = app or QApplication.instance()
    if app is None or getattr(app, "_catan_tooltip_watcher", None) is not None:
        return

    base = _current_style(app)
    if base is not None:
        style = _TooltipStyle(base, _TooltipLabelFilter(app))
        app.setStyle(style)
        app._catan_tooltip_watcher = style
        return

    watcher = _TooltipWatcher(app)
    app.installEventFilter(watcher)
    app._catan_tooltip_watcher = watcher


_current_label = None


def _remember_label(label):
    global _current_label
    _current_label = label


def _place_tip(label, pos):
    """Qt's tooltip placement (QTipLabel::placeTip): below-right of the
    cursor, flipped to the other side at the screen edges."""
    screen = QGuiApplication.screenAt(pos) or label.screen()
    point = QPoint(pos.x() + 2, pos.y() + 16)

    if screen is not None:
        area = screen.availableGeometry()
        if point.x() + label.width() > area.x() + area.width():
            point.setX(point.x() - 4 - label.width())
        if point.y() + label.height() > area.y() + area.height():
            point.setY(point.y() - 24 - label.height())
        point.setX(max(area.x(), point.x()))
        point.setY(max(area.y(), point.y()))

    label.move(point)


def show_tooltip(pos, text, widget=None):
    """QToolTip.showText for hover tooltips (called on every mouse move).

    Plain text is wrapped beforehand, so a new text is displayed as given
    (no unwrapped long line first). Qt keeps a visible tooltip with an
    unchanged text where it is; it is moved here to follow the cursor,
    without re-creating it (which flickered).
    """
    if text and not Qt.mightBeRichText(text):
        text = wrap_text(text, QFontMetrics(QToolTip.font()))

    label = _current_label
    if (
        label is not None
        and isValid(label)
        and label.isVisible()
        and QToolTip.isVisible()
        and QToolTip.text() == text
    ):
        QToolTip.showText(pos, text, widget)  # restarts Qt's hide timer
        _place_tip(label, pos)
        return

    QToolTip.showText(pos, text, widget)
