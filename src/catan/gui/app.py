from __future__ import annotations

import os, sys, platform

from catan.gui.utils.graphics_backend import configure_graphics_backend


def main() -> int:

    configure_graphics_backend()

    try:
        from PySide6.QtGui import QFont
        from PySide6.QtWidgets import QApplication

    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            raise RuntimeError(
                "The CATAN GUI dependencies are not installed.\n"
                "Install them with:\n\n"
                '    python -m pip install "catan-toolbox[gui]"\n'
            ) from exc
        raise

    """Start the CATAN graphical application."""
    import qdarktheme

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
