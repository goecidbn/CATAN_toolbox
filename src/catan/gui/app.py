import sys, os

from pathlib import Path

from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
)

from catan.gui.GUI_elements.main_window import CatanToolbox

import qdarktheme

os.environ["QT_QPA_PLATFORM"] = "xcb"


STYLE_PATH = Path("styles") / "main.qss"


def main():
    app = QApplication(sys.argv)
    qdarktheme.setup_theme("dark")

    font = QFont("Noto Sans", 10)
    app.setFont(font)

    win = CatanToolbox(*sys.argv[1:])
    # win.resize(2400, 1600)

    win.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
