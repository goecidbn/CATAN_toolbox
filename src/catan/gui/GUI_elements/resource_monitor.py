import os, psutil, sys

from scipy import sparse
import numpy as np
from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtGui import QPalette, QTextOption
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from catan.gui.GUI_elements.utils.paths import shorten_paths

# Task message area: fixed height (lines), scrolls beyond.
STATUS_LINES = 3
# Process memory is read every UPDATE_MS; walking the data for the size
# estimates (main thread, ~13 ms for 60 sessions) every DATA_UPDATE_MS,
# also while tasks run (memory use during long loading).
UPDATE_MS = 2000
DATA_UPDATE_MS = 5000


class ResourceMonitor(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.data = parent.data
        self.state = parent.state

        # Fixed height: progress and messages must not resize the menu
        # (the session list above takes the remaining height).
        self.label = QLabel()
        # sessions, assignments, total, process memory
        self.label.setFixedHeight(4 * self.label.fontMetrics().lineSpacing() + 2)
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.addWidget(self.label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        layout.addWidget(self.progress_bar)

        self.status_label = QPlainTextEdit()
        self.status_label.setReadOnly(True)
        self.status_label.setFrameShape(QFrame.Shape.NoFrame)
        self.status_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.status_label.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.status_label.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.status_label.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.status_label.setPlaceholderText("No running task")
        self.status_label.viewport().setAutoFillBackground(False)
        self.status_label.setStyleSheet("QPlainTextEdit { background: transparent; }")
        margins = self.status_label.contentsMargins()
        self.status_label.setFixedHeight(
            STATUS_LINES * self.status_label.fontMetrics().lineSpacing()
            + 2 * int(self.status_label.document().documentMargin())
            + margins.top()
            + margins.bottom()
        )
        layout.addWidget(self.status_label)

        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        self._active_tasks = {}
        self._task_messages = {}
        self._error_dialogs = set()

        # Session list loading block (inspection, failures, stop notes)
        # takes precedence over task messages.
        self._task_text = ""
        self._load_message = ""
        self.state.load_status_message.connect(self._on_load_message)

        tasks = self.state.tasks

        tasks.task_started.connect(self._on_task_started)
        tasks.task_progress.connect(self._on_task_progress)
        tasks.task_message.connect(self._on_task_message)

        tasks.task_finished.connect(self._on_task_ended)
        tasks.task_failed.connect(self._on_task_ended)
        tasks.task_cancelled.connect(self._on_task_ended)

        tasks.task_error.connect(self._on_task_error)

        self._data_lines = []
        self._data_updated = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.timer.start(UPDATE_MS)
        self.update_display()

    @Slot(str, str)
    def _on_task_started(self, group, task_id):
        task = self.state.tasks.get_task(task_id)

        if task is None or task.background:
            return

        self._active_tasks[task_id] = task.name

        self.progress_bar.setRange(0, 0)
        self._set_status(task.name)

    @Slot(str, str, int)
    def _on_task_progress(self, group, task_id, value):
        if task_id not in self._active_tasks:
            return

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(value)

    @Slot(str, str, str)
    def _on_task_message(self, group, task_id, message):
        if task_id not in self._active_tasks:
            return

        self._task_messages[task_id] = message
        self._set_status(message)

    @Slot(str, str)
    def _on_task_ended(self, group, task_id):
        if task_id not in self._active_tasks:
            return

        self._active_tasks.pop(task_id, None)
        self._task_messages.pop(task_id, None)

        if not self._active_tasks:
            # Keep the (empty) bar: no layout change between tasks.
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self._set_status("")
            return

        remaining_id = next(reversed(self._active_tasks))
        self._set_status(
            self._task_messages.get(
                remaining_id,
                self._active_tasks[remaining_id],
            )
        )

    def _set_status(self, text):
        self._task_text = text or ""
        self._show_status()

    @Slot(str)
    def _on_load_message(self, text):
        self._load_message = text or ""
        self._show_status()

    def _show_status(self):
        text = shorten_paths(self._load_message or self._task_text)
        if self.status_label.toPlainText() != text:
            self.status_label.setPlainText(text)

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

        now = time.monotonic()
        due = (
            self._data_updated is None
            or now - self._data_updated >= DATA_UPDATE_MS / 1000
        )

        if due:
            self._data_lines = self.collect_data_lines()
            self._data_updated = now

        lines = self._data_lines + [f"Process memory: {format_bytes(process_memory())}"]
        self.label.setText("\n".join(lines))

    def collect_data_lines(self):
        items = {
            "sessions": self.data.sessions,
            "assignments": self.data.assignments,
        }

        lines = []
        total = 0

        for name, obj in items.items():
            try:
                size = estimate_size(obj)
            except Exception as e:
                lines.append(f"{name}: Error estimating size: {e}")
                continue

            total += size
            lines.append(f"{name}: {format_bytes(size)}")

        lines.append(f"Total tracked: {format_bytes(total)}")
        return lines


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
