from PySide6.QtCore import QObject, Signal, Slot


class MethodWorker(QObject):
    finished = Signal(object)
    error = Signal(str)
    canceled = Signal()
    progress = Signal(int)
    message = Signal(str)

    def __init__(self, owner, method_name: str, *args, **kwargs):
        super().__init__()
        self.owner = owner
        self.method_name = method_name
        self._cancel = False
        self.args = args
        self.kwargs = kwargs

    def request_cancel(self):
        self._cancel = True

    def cancel_check(self):
        return self._cancel

    @Slot()
    def run(self):
        try:
            method = getattr(self.owner, self.method_name)
            # If your method supports cancel/progress callbacks, pass them:
            result = method(
                *self.args,
                cancel_check=self.cancel_check,
                progress_cb=self.progress.emit,
                message_cb=self.message.emit,
                **self.kwargs,
            )
            if self._cancel:
                self.canceled.emit()
            else:
                self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
