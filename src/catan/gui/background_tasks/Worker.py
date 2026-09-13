from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from .runtime import (
    TaskCancelled,
    TaskContext,
    bind_task_context,
    reset_task_context,
)


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
            progress_callback=self.signals.progress.emit,
            message_callback=self.signals.message.emit,
        )

        result = None
        token = bind_task_context(ctx)

        try:
            # Catch cancellation requested before execution actually starts.
            ctx.check_cancelled()

            result = self.fn(
                *self.args,
                **self.kwargs,
            )

            ctx.check_cancelled()

        except TaskCancelled:
            self._cancelled = True

        except Exception:
            self._failed = True
            self.signals.error.emit(traceback.format_exc())

        finally:
            # "finished" here means:
            # the runnable has stopped executing,
            # regardless of success/cancellation/failure.
            reset_task_context(token)
            self.signals.finished.emit(result)
