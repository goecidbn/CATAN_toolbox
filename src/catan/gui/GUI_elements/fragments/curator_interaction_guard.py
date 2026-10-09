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
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, obj, event):
        menu = self.controller.menu
        kind = event.type()
        app = QApplication.instance()

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
