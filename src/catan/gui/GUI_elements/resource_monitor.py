import os, psutil, sys

from scipy import sparse
import numpy as np
from PySide6.QtCore import QTimer
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

        tasks = self.state.tasks

        tasks.task_progress.connect(self.progress_bar.setValue)

        tasks.task_message.connect(self.status_label.setText)

        tasks.task_started.connect(lambda n: self.progress_bar.show())

        tasks.task_finished.connect(
            ## should additionally connect hiding message!
            lambda n: (self.progress_bar.hide(), self.status_label.clear())
        )

        tasks.task_error.connect(
            lambda err: QMessageBox.critical(self, "Worker Error", err)
        )

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.timer.start(2000)  # every 2 seconds

    def update_display(self):
        text = self.collect_resource_text()
        self.label.setText(text)

    def collect_resource_text(self):
        items = {
            "sessions": self.data.sessions,
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
