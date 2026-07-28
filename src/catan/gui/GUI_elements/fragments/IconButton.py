from PySide6.QtCore import QSize
from PySide6.QtWidgets import QToolButton
from PySide6.QtGui import QIcon
from typing import Optional

import qtawesome as qta


def make_icon_button(
    icon_names: Optional[tuple[str, ...]] = None,
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

    if icon_names is None:
        return button

    icon = None

    # if qta is not None:
    for name in icon_names:
        icon = qta.icon(name, color=color)

    if icon is None and fallback_theme_icon is not None:
        icon = QIcon.fromTheme(fallback_theme_icon)

    if icon is not None:
        button.setIcon(icon)

    return button


def set_button_icon(
    button: QToolButton,
    icon_names: tuple[str, ...],
    color: str = "white",
    *,
    tooltip: str,
    fallback_theme_icon: str | None = None,
):
    icon = None

    if qta is not None:
        for name in icon_names:
            try:
                icon = qta.icon(name, color=color)
                break
            except Exception:
                pass

    if icon is None and fallback_theme_icon is not None:
        icon = QIcon.fromTheme(fallback_theme_icon)

    if icon is not None:
        button.setIcon(icon)

    button.setToolTip(tooltip)
