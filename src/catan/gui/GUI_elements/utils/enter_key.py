"""Enter / Return as the confirm key of a button outside a dialog."""

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QLineEdit,
    QPlainTextEdit,
    QTextEdit,
    QWidget,
)

ENTER_KEYS = (Qt.Key.Key_Return, Qt.Key.Key_Enter)


def _edits_text(widget):
    return isinstance(
        widget, (QLineEdit, QAbstractSpinBox, QTextEdit, QPlainTextEdit)
    ) or (isinstance(widget, QComboBox) and widget.isEditable())


class EnterKeyButton(QObject):
    """Enter / Return clicks ``button`` while it is shown and enabled, if the
    key goes to a widget of the button's window, no popup or modal dialog is
    open and no text input has the focus. The application event filter is
    installed only while the button is shown."""

    def __init__(self, button):
        super().__init__(button)
        self.button = button
        self._filtering = False
        button.installEventFilter(self)
        self._set_filtering(button.isVisible())

    def _set_filtering(self, active):
        app = QApplication.instance()
        if app is None or active == self._filtering:
            return
        if active:
            app.installEventFilter(self)
        else:
            app.removeEventFilter(self)
        self._filtering = active

    def _accepts(self, watched):
        button = self.button
        return (
            isinstance(watched, QWidget)
            and button.isVisible()
            and button.isEnabled()
            and QApplication.activePopupWidget() is None
            and QApplication.activeModalWidget() is None
            and watched.window() is button.window()
            and not _edits_text(QApplication.focusWidget())
        )

    def eventFilter(self, watched, event):
        if watched is self.button:
            if event.type() == QEvent.Type.Show:
                self._set_filtering(True)
            elif event.type() == QEvent.Type.Hide:
                self._set_filtering(False)
            return False

        if (
            event.type() == QEvent.Type.KeyPress
            and event.key() in ENTER_KEYS
            and event.modifiers()
            in (Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.KeypadModifier)
            and self._accepts(watched)
        ):
            self.button.click()
            return True

        return False
