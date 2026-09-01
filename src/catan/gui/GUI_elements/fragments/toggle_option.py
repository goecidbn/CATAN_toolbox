from __future__ import annotations

from catan.gui.resources.get_icon import get_fa_icon
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QSizePolicy, QToolButton, QWidget


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
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        self._base_text = text

        self.setCheckable(True)
        self.set_expanded(False)

        self.setAutoRaise(True)

        # let the button remain only as wide as necessary.
        self.setSizePolicy(
            QSizePolicy.Policy.Maximum,
            QSizePolicy.Policy.Fixed,
        )

        icon = get_fa_icon(icon_name) if icon_name is not None else None
        if icon is not None:
            self.setIcon(icon)
            self.setIconSize(
                QSize(icon_size, icon_size)
            )

        if tooltip is not None:
            self.setToolTip(tooltip)

        # Icon + text are displayed as one button.
        self.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )

        # if container is not None:
        self.set_container(container)


        self.toggled.connect(
            self.toggle_visibility
        )


        # Explicit call is useful if expanded=False,
        # because setChecked(False) may emit nothing.
        self.toggle_visibility(expanded)

    def toggle_visibility(
        self,
        expanded: bool,
    ) -> None:

        if self.container is not None:
            self.container.setVisible(expanded)
        expanded = expanded and self.container is not None
        
        arrow = "▾" if expanded else "▸"

        if self._base_text:
            self.setText(
                f"{self._base_text}  {arrow}"
            )
        else:
            self.setText(arrow)

    def set_expanded(
        self,
        expanded: bool,
    ) -> None:
        
        expanded = expanded and self.container is not None
        self.setChecked(expanded)

    def set_container(
        self,
        container: QWidget | None = None,
    ) -> None:
        self.container = container
        if self.container is not None:
            self.container.setVisible(self.isChecked())
