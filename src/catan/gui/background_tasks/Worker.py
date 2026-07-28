from PySide6.QtCore import QObject, Signal, QRunnable, Slot
import traceback

from dataclasses import dataclass
from typing import Callable


@dataclass
class TaskContext:
    cancel_check: Callable[[], bool]
    progress: Callable[[int], None]
    message: Callable[[str], None]

    def cancelled(self):
        return self.cancel_check()


class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(int)
    message = Signal(str)


class Worker(QRunnable):

    def __init__(self, fn, *args, **kwargs):
        super().__init__()

        self.fn = fn
        self.args = args
        self.kwargs = kwargs

        self.cancelled = False

        self.signals = WorkerSignals()

    def cancel(self):
        self.cancelled = True

    def is_cancelled(self):
        return self.cancelled

    @Slot()
    def run(self):

        ctx = TaskContext(
            cancel_check=self.is_cancelled,
            progress=self.signals.progress.emit,
            message=self.signals.message.emit,
        )

        try:

            result = self.fn(
                *self.args,
                ctx=ctx,
                **self.kwargs,
            )

            if not self.cancelled:
                self.signals.finished.emit(result)

        except Exception:

            self.signals.error.emit(traceback.format_exc())
