import os, psutil, sys

from scipy import sparse
import numpy as np
from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QProgressBar, QMessageBox


class ResourceMonitor(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.data = parent.data
        self.state = parent.state

        self.label = QLabel()
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.hide()  # hide initially
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel()
        layout.addWidget(self.status_label)

        self._active_tasks = {}
        self._task_messages = {}
        self._error_dialogs = set()

        tasks = self.state.tasks

        tasks.task_started.connect(self._on_task_started)
        tasks.task_progress.connect(self._on_task_progress)
        tasks.task_message.connect(self._on_task_message)

        tasks.task_finished.connect(self._on_task_ended)
        tasks.task_failed.connect(self._on_task_ended)
        tasks.task_cancelled.connect(self._on_task_ended)

        tasks.task_error.connect(self._on_task_error)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.timer.start(2000)  # every 2 seconds

    @Slot(str, str)
    def _on_task_started(self, group, task_id):
        task = self.state.tasks.tasks.get(task_id)
        self._active_tasks[task_id] = task.name if task is not None else group

        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        self.status_label.setText(self._active_tasks[task_id])

    @Slot(str, str, int)
    def _on_task_progress(self, group, task_id, value):
        if task_id not in self._active_tasks:
            return

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(value)

    @Slot(str, str, str)
    def _on_task_message(self, group, task_id, message):
        self._task_messages[task_id] = message
        self.status_label.setText(message)
        self.status_label.setWordWrap(True)
        self.status_label.setToolTip(message)

    @Slot(str, str)
    def _on_task_ended(self, group, task_id):
        self._active_tasks.pop(task_id, None)
        self._task_messages.pop(task_id, None)

        if not self._active_tasks:
            self.progress_bar.hide()
            self.status_label.clear()
            self.status_label.setToolTip("")
            return

        remaining_id = next(reversed(self._active_tasks))
        self.status_label.setText(
            self._task_messages.get(
                remaining_id,
                self._active_tasks[remaining_id],
            )
        )

    @Slot(str, str, str)
    def _on_task_error(self, group, task_id, traceback_text):
        task = self.state.tasks.tasks.get(task_id)

        name = task.name if task is not None else group
        failure = (
            getattr(task.worker, "failure_info", None) if task is not None else None
        )

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("Task failed")
        box.setTextFormat(Qt.TextFormat.PlainText)

        if failure is not None:
            box.setText(failure["summary"])
            box.setInformativeText(failure["recovery"])
        else:
            box.setText(f"{name} failed.")
            box.setInformativeText(
                "Open Details for the underlying error.\n\n"
                "This task may have made partial changes before it failed. "
                "Automatic retry is therefore not offered."
            )

        box.setDetailedText(traceback_text)
        box.setStandardButtons(QMessageBox.StandardButton.Close)

        # Keep the dialog alive without blocking the task-completion handler.
        self._error_dialogs.add(box)

        def release_dialog(_result):
            self._error_dialogs.discard(box)
            box.deleteLater()

        box.finished.connect(release_dialog)
        box.open()

    def update_display(self):
        import time

        # print("resource monitor update")

        t0 = time.perf_counter()

        try:
            # existing method contents

            text = self.collect_resource_text()
            self.label.setText(text)

        finally:
            dt = time.perf_counter() - t0

            if dt > 0.05:
                print(f"ResourceMonitor.update_display: " f"{dt * 1000:.1f} ms")

    def collect_resource_text(self):
        items = {
            "sessions": self.data.sessions,
            "assignments": self.data.assignments,
            # "neurons": self.data.neurons,
            # "matching": self.data.matching,
            # "statistics": self.data.statistics,
        }

        lines = []

        total = 0
        for name, obj in items.items():
            try:
                size = estimate_size(obj)
                lines.append(f"{name}: {format_bytes(size)}")
            except Exception as e:
                size = 0
                lines.append(f"{name}: Error estimating size: {e}")
                continue
            # size = estimate_size(obj)
            # total += size

        lines.append(f"Total tracked: {format_bytes(total)}")
        lines.append(f"Process memory: {format_bytes(process_memory())}")
        # lines.append(f"Current job: {self.state.current_job or 'None'}")

        return "\n".join(lines)


def estimate_size(obj, seen=None):
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    if isinstance(obj, np.ndarray):
        return obj.nbytes

    if sparse.issparse(obj):
        return obj.data.nbytes + obj.indices.nbytes + obj.indptr.nbytes

    if isinstance(obj, dict):
        return sys.getsizeof(obj) + sum(
            estimate_size(k, seen) + estimate_size(v, seen) for k, v in obj.items()
        )

    if isinstance(obj, (list, tuple, set)):
        return sys.getsizeof(obj) + sum(estimate_size(v, seen) for v in obj)

    if hasattr(obj, "__dict__"):
        return sys.getsizeof(obj) + estimate_size(vars(obj), seen)

    return sys.getsizeof(obj)


def format_bytes(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


process = psutil.Process(os.getpid())


def process_memory():
    return process.memory_info().rss
