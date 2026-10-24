from PySide6.QtCore import QSize
from PySide6.QtWidgets import QToolButton
from PySide6.QtGui import QIcon
from typing import Optional

from catan.gui.resources.get_icon import get_fa_icon
import qtawesome as qta


def make_icon_button(
    icon_name: Optional[str] = None,
    color: str = "white",
    *,
    tooltip: str = "",
    fallback_theme_icon: str | None = None,
    size: int = 22,
    icon_size: int = 16,
) -> QToolButton:
    button = QToolButton()
    button.setToolTip(tooltip)
    button.setFixedSize(size, size)
    button.setIconSize(QSize(icon_size, icon_size))
    button.setAutoRaise(True)

    if icon_name is None:
        return button

    icon = get_fa_icon(icon_name,color) if icon_name is not None else None

    if icon is None and fallback_theme_icon is not None:
        icon = QIcon.fromTheme(fallback_theme_icon)

    if icon is not None:
        button.setIcon(icon)

    return button


def set_button_icon(
    button: QToolButton,
    icon_name: str,
    color: str = "white",
    *,
    tooltip: Optional[str] = None,
    fallback_theme_icon: str | None = None,
):
    # Rows refresh their buttons on every data change; creating the icon
    # again is the expensive part. Skip it if this icon is still shown
    # (an icon set elsewhere, e.g. a spinner, has another cache key).
    key = (icon_name, color, fallback_theme_icon)
    shown = getattr(button, "_catan_icon", None)

    if shown is None or shown != (key, button.icon().cacheKey()):
        icon = get_fa_icon(icon_name, color) if icon_name is not None else None

        if icon is None and fallback_theme_icon is not None:
            icon = QIcon.fromTheme(fallback_theme_icon)

        if icon is not None:
            button.setIcon(icon)
            button._catan_icon = (key, button.icon().cacheKey())

    if tooltip is not None and button.toolTip() != tooltip:
        button.setToolTip(tooltip)
