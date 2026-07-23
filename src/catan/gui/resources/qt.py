"""Qt-specific helpers for bundled CATAN resources."""

from __future__ import annotations

from PySide6.QtCore import QByteArray
from PySide6.QtGui import QIcon, QPixmap

from catan.gui.resources.loader import read_bytes


def load_pixmap(name: str) -> QPixmap:
    """Load an image from ``resources/icons`` into a QPixmap."""
    data = read_bytes(f"icons/{name}")

    pixmap = QPixmap()

    if not pixmap.loadFromData(QByteArray(data)):
        raise ValueError(
            f"Qt could not load bundled icon: {name!r}"
        )

    return pixmap


def load_icon(name: str) -> QIcon:
    """Load an icon from ``resources/icons``."""
    return QIcon(load_pixmap(name))