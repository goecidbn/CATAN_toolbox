import threading
import time

import pytest

pytest.importorskip("PySide6")
pytestmark = pytest.mark.gui

from PySide6.QtWidgets import QApplication  # noqa: E402

from catan.gui.background_tasks.TaskManager import TaskManager  # noqa: E402


def _wait(app, condition, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not condition():
        app.processEvents()
        time.sleep(0.01)
    return condition()


def test_export_runs_next_to_processing_and_display_work():
    app = QApplication.instance() or QApplication([])
    tasks = TaskManager()
    release = threading.Event()
    done = []

    # A long export (seed files) is no processing ...
    export = tasks.start("export", "export", lambda: release.wait(5))
    assert _wait(app, lambda: tasks.current["export"] is not None)
    assert not tasks.processing_busy()

    # ... so display work (statistics) and processing still start.
    tasks.start("calculating", "statistic", lambda: 1, background=True,
                on_result=lambda _: done.append("statistic"))
    tasks.start("model update", "manipulation", lambda: 2,
                on_result=lambda _: done.append("manipulation"))
    assert _wait(app, lambda: len(done) == 2)
    assert export in tasks.tasks

    release.set()
    assert _wait(app, lambda: export not in tasks.tasks)

    # Deliver the workers' queued signals while the manager still exists.
    tasks.pool.waitForDone(5000)
    app.processEvents()
    tasks.stop_queue_timer()
    tasks.deleteLater()
    app.processEvents()
