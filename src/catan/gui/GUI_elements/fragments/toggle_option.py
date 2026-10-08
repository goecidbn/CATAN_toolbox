from __future__ import annotations

from PySide6.QtCore import QPoint, QSize, Qt, QEvent, QTimer, Signal
from PySide6.QtWidgets import (
    QFrame,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QWidget,
    QVBoxLayout,
    QLayout,
)

from catan.gui.resources.get_icon import get_fa_icon


class OptionsPopup(QFrame):
    hidden = Signal()

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Popup)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setAttribute(Qt.WidgetAttribute.WA_NoMouseReplay, True)

    def hideEvent(self, event):
        super().hideEvent(event)
        self.hidden.emit()

    def mousePressEvent(self, event):
        if self.rect().contains(event.position().toPoint()):
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self.rect().contains(event.position().toPoint()):
            event.accept()
        else:
            super().mouseReleaseEvent(event)


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

        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._resize_popup)
        self._resizing_popup = False

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
            self._popup = OptionsPopup(self)
            self._popup.hidden.connect(self._popup_hidden)

            popup_layout = QVBoxLayout(self._popup)
            popup_layout.setContentsMargins(4, 4, 4, 4)
            popup_layout.setSpacing(0)

            self._scroll = QScrollArea(self._popup)
            self._scroll.setWidgetResizable(True)
            self._scroll.setFrameShape(QFrame.Shape.NoFrame)
            self._scroll.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
            self._scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )

            popup_layout.addWidget(self._scroll)

            self._scroll.viewport().installEventFilter(self)

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
        if self.container is None:
            return

        self.container.show()
        self._resize_popup()
        self._popup.show()
        self._schedule_popup_resize()

    def _schedule_popup_resize(self):
        if (
            self._popup_mode
            and self._popup is not None
            and self._popup.isVisible()
            and not self._resize_timer.isActive()
        ):
            self._resize_timer.start(0)

    def eventFilter(self, watched, event):
        if self._popup_mode and self._scroll is not None:
            content_changed = watched is self.container and event.type() in (
                QEvent.Type.LayoutRequest,
                QEvent.Type.Resize,
                QEvent.Type.Show,
            )
            viewport_changed = (
                watched is self._scroll.viewport()
                and event.type() == QEvent.Type.Resize
            )

            if content_changed or viewport_changed:
                self._schedule_popup_resize()

        return super().eventFilter(watched, event)

    def _resize_popup(self):
        if self._resizing_popup or self._popup is None or self.container is None:
            return

        self._resizing_popup = True

        try:
            available = self.screen().availableGeometry().adjusted(8, 8, -8, -8)

            width = min(
                self._popup_size.width(),
                max(1, available.width() - 8),
            )

            # Reserve space for the scrollbar when calculating wrapping.
            content_width = max(1, width - 24)
            content_layout = self.container.layout()

            if content_layout is not None:
                content_layout.activate()

                if content_layout.hasHeightForWidth():
                    content_height = content_layout.totalHeightForWidth(content_width)
                else:
                    content_height = content_layout.sizeHint().height()
            else:
                content_height = self.container.sizeHint().height()

            content_height = max(1, content_height)

            # The content retains its full height even when the popup
            # is capped, allowing the scroll area to expose all of it.
            if self.container.minimumHeight() != content_height:
                self.container.setMinimumHeight(content_height)

            height = min(
                max(80, content_height + 24),
                self._popup_size.height(),
                max(1, available.height() - 8),
            )

            target = QSize(width, height)
            if self._scroll.size() != target:
                self._scroll.setFixedSize(target)

            self._popup.adjustSize()
            popup_size = self._popup.size()

            below = self.mapToGlobal(QPoint(0, self.height()))
            above = self.mapToGlobal(QPoint(0, 0))

            x = below.x()
            y = below.y()

            if y + popup_size.height() > available.bottom() + 1:
                y = above.y() - popup_size.height()

            x = max(
                available.left(),
                min(
                    x,
                    available.right() - popup_size.width() + 1,
                ),
            )
            y = max(
                available.top(),
                min(
                    y,
                    available.bottom() - popup_size.height() + 1,
                ),
            )

            self._popup.move(x, y)

        finally:
            self._resizing_popup = False

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
            previous.removeEventFilter(self)
            if self._popup_mode:
                self._scroll.takeWidget()
                previous.setParent(self)
            previous.hide()

        self.container = container

        if container is not None:
            if self._popup_mode:
                content_layout = container.layout()
                if content_layout is not None:
                    content_layout.setSizeConstraint(
                        QLayout.SizeConstraint.SetMinimumSize
                    )

                self._scroll.setWidget(container)
                container.installEventFilter(self)
            else:
                container.hide()

        self._update_caption(False)
