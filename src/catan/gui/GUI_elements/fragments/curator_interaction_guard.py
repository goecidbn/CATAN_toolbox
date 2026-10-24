from PySide6.QtCore import (
    QObject,
    QEvent,
    Qt,
    QCoreApplication,
)
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QComboBox,
    QAbstractSpinBox,
    QLineEdit,
    QTextEdit,
    QPlainTextEdit,
    QAbstractScrollArea,
)


class CuratorInteractionGuard(QObject):
    def __init__(self, controller):
        super().__init__(controller.menu)
        self.controller = controller

        # Wheel and shortcut handling only matter while the curator is
        # shown: the application-wide filter (which sees every event of the
        # GUI) is installed on Show and removed on Hide of the menu.
        self._app_filter_installed = False
        controller.menu.installEventFilter(self)
        self._set_app_filter(controller.menu.isVisible())

    def _set_app_filter(self, installed):
        app = QApplication.instance()
        if app is None or installed == self._app_filter_installed:
            return

        if installed:
            app.installEventFilter(self)
        else:
            app.removeEventFilter(self)
        self._app_filter_installed = installed

    def eventFilter(self, obj, event):
        menu = self.controller.menu
        kind = event.type()
        app = QApplication.instance()

        if obj is menu and kind in (QEvent.Type.Show, QEvent.Type.Hide):
            self._set_app_filter(kind == QEvent.Type.Show)
            return False

        if kind == QEvent.Type.Wheel and isinstance(obj, QWidget):
            if app.activePopupWidget() is not None:
                return False

            if obj is not menu and not menu.isAncestorOf(obj):
                return False

            control = obj

            while control is not None and control is not menu:
                if isinstance(control, (QComboBox, QAbstractSpinBox)):
                    parent = control.parentWidget()

                    while parent is not None:
                        if isinstance(parent, QAbstractScrollArea):
                            QCoreApplication.sendEvent(parent.viewport(), event)
                            return True

                        parent = parent.parentWidget()

                    # No surrounding scroll area: don't edit the value.
                    return True

                control = control.parentWidget()

            return False

        if kind not in (
            QEvent.Type.ShortcutOverride,
            QEvent.Type.KeyPress,
        ):
            return False

        actions = self.controller.actions

        if (
            actions.queue is None
            or not menu.isVisible()
            or app.activeWindow() is not menu.window()
            or app.activePopupWidget() is not None
        ):
            return False

        # Editing takes precedence over manipulation shortcuts.
        focus = app.focusWidget()

        while focus is not None:
            if isinstance(
                focus,
                (
                    QComboBox,
                    QAbstractSpinBox,
                    QLineEdit,
                    QTextEdit,
                    QPlainTextEdit,
                ),
            ):
                return False

            focus = focus.parentWidget()

        if event.modifiers() != Qt.KeyboardModifier.NoModifier:
            return False

        stop = event.key() == Qt.Key.Key_Escape

        apply = (
            event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
            and actions.review
            and actions.pending is not None
            and not actions.waiting
            and actions.next.isEnabled()
        )

        if not (stop or apply):
            return False

        event.accept()

        if kind == QEvent.Type.KeyPress and not event.isAutoRepeat():
            if stop:
                actions._finish("Stopped by Escape")
            else:
                actions._apply(actions.run_id)

        return True
