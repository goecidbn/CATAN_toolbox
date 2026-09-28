from __future__ import annotations

import argparse
import atexit
import os
import sys
from tempfile import TemporaryDirectory

from catan.gui.utils.graphics_backend import configure_graphics_backend


def main() -> int:

    parser = argparse.ArgumentParser(
        description="Start the CATAN GUI.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--fresh",
        "-fresh",
        action="store_true",
        help="Use temporary, empty settings without changing saved preferences.",
    )
    options, qt_arguments = parser.parse_known_args(sys.argv[1:])

    configure_graphics_backend()

    try:
        from PySide6.QtGui import QFont
        from PySide6.QtWidgets import QApplication

        from PySide6.QtCore import QSettings

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
        app = QApplication([sys.argv[0], *qt_arguments])
    qdarktheme.setup_theme("dark")

    font = QFont("Noto Sans", 10)
    app.setFont(font)

    # app.aboutToQuit.connect(lambda: print("QApplication aboutToQuit"))
    # app.lastWindowClosed.connect(lambda: print("lastWindowClosed"))
    # app.aboutToQuit.connect(lambda: print("aboutToQuit"))

    settings = None

    if options.fresh:
        settings_directory = TemporaryDirectory(prefix="catan-fresh-")
        settings = QSettings(
            os.path.join(settings_directory.name, "settings.ini"),
            QSettings.Format.IniFormat,
        )
        settings.setFallbacksEnabled(False)

        def discard_fresh_settings():
            # Keep both objects alive until shutdown, including when an
            # existing QApplication owns the event loop.
            settings.sync()
            settings_directory.cleanup()

        atexit.register(discard_fresh_settings)

    window = MainWindow(settings=settings)

    if options.fresh:
        window.setWindowTitle(f"{window.windowTitle()} [fresh settings]")

    window.show()

    if owns_application:
        return app.exec()

    return 0


# import threading

# from PySide6.QtCore import QThreadPool

if __name__ == "__main__":
    raise SystemExit(main())
