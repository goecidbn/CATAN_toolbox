from __future__ import annotations

from PySide6.QtCore import QPoint, QSize, Qt
from PySide6.QtWidgets import (
    QFrame,
    QMenu,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QWidget,
    QWidgetAction,
)

from catan.gui.resources.get_icon import get_fa_icon


class ToggleOption(QToolButton):
    def __init__(
        self,
        container: QWidget | None = None,
        *,
        text: str = "",
        icon_name: str | None = None,
        tooltip: str | None = None,
        expanded: bool = False,
        icon_size: int = 15,
        popup: bool = False,
        popup_size: tuple[int, int] = (480, 600),
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        self.container = None
        self._base_text = text
        self._popup_mode = popup
        self._popup_size = QSize(*popup_size)
        self._popup = None
        self._scroll = None

        self.setCheckable(True)
        self.setAutoRaise(True)
        self.setSizePolicy(
            QSizePolicy.Policy.Maximum,
            QSizePolicy.Policy.Fixed,
        )

        if icon_name is not None:
            self.setIcon(get_fa_icon(icon_name))
            self.setIconSize(QSize(icon_size, icon_size))

        if tooltip:
            self.setToolTip(tooltip)

        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

        if popup:
            self._popup = QMenu(self)
            self._popup.setAttribute(Qt.WidgetAttribute.WA_NoMouseReplay, True)
            self._popup.aboutToHide.connect(self._popup_hidden)

            self._scroll = QScrollArea(self._popup)
            self._scroll.setWidgetResizable(True)
            self._scroll.setFrameShape(QFrame.Shape.NoFrame)

            action = QWidgetAction(self._popup)
            action.setDefaultWidget(self._scroll)
            self._popup.addAction(action)

        self.toggled.connect(self.toggle_visibility)
        self.set_container(container)
        self.set_expanded(expanded)

    def _update_caption(self, expanded):
        if self._popup_mode and not self._base_text and not self.icon().isNull():
            self.setText("")
            self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            return

        arrow = "▾" if expanded else "▸"
        self.setText(f"{self._base_text}  {arrow}" if self._base_text else arrow)

    def _show_popup(self):
        if self._popup.isVisible():
            return

        screen = self.screen()
        available = screen.availableGeometry()
        hint = self.container.sizeHint()

        width = min(
            self._popup_size.width(),
            max(120, available.width() - 24),
        )
        height = min(
            max(80, hint.height() + 24),
            self._popup_size.height(),
            max(80, available.height() - 24),
        )

        self._scroll.setFixedSize(width, height)
        self.container.show()

        # QMenu handles screen-edge positioning and outside-click/Escape
        # dismissal using the existing Qt application/backend.
        self._popup.popup(self.mapToGlobal(QPoint(0, self.height())))

    def _popup_hidden(self):
        self.setChecked(False)

    def toggle_visibility(self, expanded):
        expanded = bool(expanded and self.container is not None)
        self._update_caption(expanded)

        if self._popup_mode:
            if expanded:
                self._show_popup()
            else:
                self._popup.hide()
        elif self.container is not None:
            self.container.setVisible(expanded)

    def set_expanded(self, expanded):
        expanded = bool(expanded and self.container is not None)

        if self.isChecked() != expanded:
            self.setChecked(expanded)
        else:
            self.toggle_visibility(expanded)

    def set_container(self, container=None):
        if self.container is container:
            return

        self.set_expanded(False)

        previous = self.container
        if previous is not None:
            if self._popup_mode:
                self._scroll.takeWidget()
                previous.setParent(self)
            previous.hide()

        self.container = container

        if container is not None:
            if self._popup_mode:
                self._scroll.setWidget(container)
            else:
                container.hide()

        self._update_caption(False)
