from __future__ import annotations

import sys, os

from PySide6.QtGui import QFont

import qdarktheme

os.environ["QT_QPA_PLATFORM"] = "xcb"

def main() -> int:

    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            raise RuntimeError(
                "The CATAN GUI dependencies are not installed.\n"
                'Install them with:\n\n'
                '    python -m pip install "catan[gui]"\n'
            ) from exc
        raise

    """Start the CATAN graphical application."""
    from catan.gui.GUI_elements.main_window import MainWindow
    app = QApplication.instance()

    owns_application = app is None
    if app is None:
        app = QApplication(sys.argv)
    qdarktheme.setup_theme("dark")

    font = QFont("Noto Sans", 10)
    app.setFont(font)

    window = MainWindow(*sys.argv[1:])
    window.show()

    if owns_application:
        return app.exec()


    return 0


if __name__ == "__main__":
    raise SystemExit(main())
