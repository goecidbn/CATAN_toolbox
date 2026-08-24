from __future__ import annotations

from catan.gui.resources.get_icon import get_fa_icon
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QSizePolicy, QToolButton, QWidget


class ToggleOption(QToolButton):
    def __init__(
        self,
        container: QWidget,
        *,
        text: str = "",
        icon_name: str | None = None,
        tooltip: str | None = None,
        expanded: bool = False,
        icon_size: int = 15,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        self.container = container
        self._base_text = text

        self.setCheckable(True)
        self.setAutoRaise(True)

        # Do not force a fixed width:
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

        self.toggled.connect(
            self.toggle_visibility
        )

        self.setChecked(expanded)

        # Explicit call is useful if expanded=False,
        # because setChecked(False) may emit nothing.
        self.toggle_visibility(expanded)

    def toggle_visibility(
        self,
        expanded: bool,
    ) -> None:

        self.container.setVisible(expanded)

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
        self.setChecked(expanded)

    def set_container(
        self,
        container: QWidget,
    ) -> None:
        self.container = container
        self.container.setVisible(
            self.isChecked()
        )