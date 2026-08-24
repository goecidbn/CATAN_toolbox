from __future__ import annotations

import traceback

from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


class TaskCancelled(Exception):
    """Raised when a worker notices that cancellation was requested."""


@dataclass
class TaskContext:
    cancel_check: Callable[[], bool]
    progress: Callable[[int], None]
    message: Callable[[str], None]

    def cancelled(self) -> bool:
        return self.cancel_check()

    def check_cancelled(self) -> None:
        """
        Raise TaskCancelled if cancellation was requested.

        Long-running jobs can call this periodically to stop cleanly.
        """
        if self.cancelled():
            raise TaskCancelled()


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

        self._cancelled = False
        self._failed = False

        self.signals = WorkerSignals()

    def cancel(self) -> None:
        """
        Request cooperative cancellation.

        The running function must periodically check ctx.cancelled()
        or call ctx.check_cancelled().
        """
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled

    def is_failed(self) -> bool:
        return self._failed

    @Slot()
    def run(self):

        ctx = TaskContext(
            cancel_check=self.is_cancelled,
            progress=self.signals.progress.emit,
            message=self.signals.message.emit,
        )

        result = None

        try:
            # Catch cancellation requested before execution actually starts.
            ctx.check_cancelled()

            result = self.fn(
                *self.args,
                ctx=ctx,
                **self.kwargs,
            )

        except TaskCancelled:
            self._cancelled = True

        except Exception:
            self._failed = True
            self.signals.error.emit(
                traceback.format_exc()
            )

        finally:
            # "finished" here means:
            # the runnable has stopped executing,
            # regardless of success/cancellation/failure.
            self.signals.finished.emit(result)