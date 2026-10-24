"""Every QMessageBox: buttons wide enough for their labels, measured when
the box is shown (on the screen it appears on) and again after it moves to
another screen (font scaling may differ). If the buttons do not fit next to
each other within the width QMessageBox allows on that screen, they are
stacked vertically."""

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QMessageBox

from catan.gui.GUI_elements.utils.menu_creation import fit_message_box_buttons
from catan.gui.GUI_elements.utils.tooltips import POLISH_HOOKS

# Icon column and margins of a message box next to its button row (approx.).
_BOX_DECORATION = 120


def _width_limit(box):
    """Maximum width QMessageBox gives itself on the box's screen
    (QMessageBoxPrivate::updateSize)."""
    screen = box.screen() or QApplication.primaryScreen()
    if screen is None:
        return 1000

    width = screen.availableGeometry().width()
    return width if width <= 1024 else min(width - 480, 1000)


def arrange_message_box_buttons(box):
    fit_message_box_buttons(box)

    button_box = box.findChild(QDialogButtonBox)
    if button_box is None:
        return

    buttons = button_box.buttons()
    spacing = max(0, button_box.layout().spacing()) if button_box.layout() else 6
    row = sum(button.minimumWidth() for button in buttons) + spacing * max(
        0, len(buttons) - 1
    )

    vertical = row + _BOX_DECORATION > _width_limit(box)
    orientation = Qt.Orientation.Vertical if vertical else Qt.Orientation.Horizontal
    if button_box.orientation() != orientation:
        button_box.setOrientation(orientation)


class _MessageBoxFitter(QObject):
    def eventFilter(self, box, event):
        # Show reaches the filter before QMessageBox.showEvent, which fixes
        # the box size from the buttons' minimum widths.
        if event.type() == QEvent.Type.Show and isinstance(box, QMessageBox):
            arrange_message_box_buttons(box)
            self._watch_screen(box)
        return False

    def _watch_screen(self, box):
        window = box.windowHandle()
        if window is None or box.property("_catan_screen_watch"):
            return
        box.setProperty("_catan_screen_watch", True)

        def refit(*_):
            arrange_message_box_buttons(box)
            # QMessageBox recomputes its fixed size on font changes.
            QApplication.sendEvent(box, QEvent(QEvent.Type.FontChange))

        window.screenChanged.connect(refit)


_fitter = None


def _polish_hook(widget):
    if isinstance(widget, QMessageBox) and not widget.property("_catan_fitted"):
        widget.setProperty("_catan_fitted", True)
        widget.installEventFilter(_fitter)


def install_message_box_fitting(app=None):
    """Install once (after install_tooltip_wrapping, whose style polishes)."""
    global _fitter
    app = app or QApplication.instance()
    if app is None or _fitter is not None:
        return

    _fitter = _MessageBoxFitter(app)
    POLISH_HOOKS.append(_polish_hook)
