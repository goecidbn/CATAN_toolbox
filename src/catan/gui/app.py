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

    # Before any worker thread exists: fewer per-thread heaps keep freed
    # memory returnable to the system (see catan.core.memory).
    from catan.core.memory import limit_malloc_arenas

    limit_malloc_arenas()

    if sys.argv[1:2] == ["--catan-read-worker"]:
        if len(sys.argv) != 3:
            raise SystemExit("Internal reader mode requires a request directory.")

        from catan.core.io.isolated_read import _child_main

        _child_main(sys.argv[2])
        return 0

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

    from catan.gui.GUI_elements.utils.tooltips import install_tooltip_wrapping

    install_tooltip_wrapping(app)

    from catan.gui.GUI_elements.utils.message_boxes import (
        install_message_box_fitting,
    )

    install_message_box_fitting(app)

    # app.aboutToQuit.connect(lambda: print("QApplication aboutToQuit"))
    # app.lastWindowClosed.connect(lambda: print("lastWindowClosed"))
    # app.aboutToQuit.connect(lambda: print("aboutToQuit"))

    settings = None
    # CATAN's own exit cleanup, also run before a hard exit (Python's atexit
    # handlers are skipped there: PySide's waits for the worker threads).
    exit_cleanup = []

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
        exit_cleanup.append(discard_fresh_settings)

    # # Temporary Linux freeze diagnostics.
    # import faulthandler
    # import signal
    # import tempfile
    # import threading
    # from pathlib import Path

    # if sys.platform.startswith("linux"):
    #     freeze_log_path = (
    #         Path(tempfile.gettempdir()) / f"catan-freeze-{os.getpid()}.log"
    #     )

    #     # Keep the file alive for the application's lifetime.
    #     app._freeze_log = freeze_log_path.open("a", buffering=1)

    #     app._freeze_log.write(
    #         f"CATAN PID: {os.getpid()}\n" f"GUI thread: {threading.get_ident():#x}\n\n"
    #     )
    #     app._freeze_log.flush()

    #     faulthandler.register(
    #         signal.SIGUSR1,
    #         file=app._freeze_log,
    #         all_threads=True,
    #         chain=False,
    #     )

    #     print(
    #         f"Freeze diagnostics ready: PID={os.getpid()}, " f"log={freeze_log_path}",
    #         flush=True,
    #     )

    window = MainWindow(settings=settings)

    if options.fresh:
        window.setWindowTitle(f"{window.windowTitle()} [fresh settings]")

    window.show()

    # if owns_application:
    #     try:
    #         return app.exec()
    #     finally:
    #         faulthandler.cancel_dump_traceback_later()
    #         traceback_log.close()

    if owns_application:
        code = app.exec()
        if not _finish_background_work(window, SHUTDOWN_GRACE_S):
            # A worker is stuck (e.g. a read from a hanging network mount):
            # Qt would wait for it before the process ends. Files are
            # written atomically, so ending now leaves no partial file.
            print(
                f"CATAN: background work did not stop within {SHUTDOWN_GRACE_S:g} s; "
                "exiting without waiting for it.",
                file=sys.stderr,
                flush=True,
            )
            for cleanup in exit_cleanup:
                try:
                    cleanup()
                except Exception:
                    pass
            sys.stdout.flush()
            os._exit(code)
        return code

    return 0


# Closing CATAN cancels running work and waits this long (s) for it to end.
SHUTDOWN_GRACE_S = 2.0


def _finish_background_work(window, timeout):
    """Cancel tasks and file inspections; True if every worker thread
    ended within ``timeout`` seconds."""
    import threading
    import time

    from PySide6.QtCore import QThreadPool

    from catan.gui.GUI_elements.fragments.file_inspection import FileInspection

    try:
        window.state.tasks.cancel_all()
    except Exception as exc:  # the window may be half torn down
        print(f"CATAN: cancelling tasks on exit failed: {exc}", file=sys.stderr)
    FileInspection.cancel_all()

    deadline = time.monotonic() + timeout

    def remaining_ms():
        return max(0, int((deadline - time.monotonic()) * 1000))

    done = all(
        pool.waitForDone(remaining_ms())
        for pool in (QThreadPool.globalInstance(), *FileInspection.pools())
    )

    # Plain Python threads Python would join at exit (daemon threads are not).
    for thread in threading.enumerate():
        if thread is threading.main_thread() or thread.daemon:
            continue
        thread.join(remaining_ms() / 1000)
        done = done and not thread.is_alive()

    return done


# import threading

# from PySide6.QtCore import QThreadPool

if __name__ == "__main__":
    raise SystemExit(main())
