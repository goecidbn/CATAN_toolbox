from PySide6.QtCore import QObject, Signal, QThreadPool
from .Worker import Worker

class TaskManager(QObject):

    task_started = Signal(str)
    task_finished = Signal(str)
    task_progress = Signal(int)
    task_message = Signal(str)
    task_error = Signal(str)

    def __init__(self):

        super().__init__()

        self.pool = QThreadPool.globalInstance()

        self.tasks = {}

    def start(
        self,
        name,
        fn,
        *args,
        finished=None,
        **kwargs,
    ):

        if name in self.tasks:

            self.tasks[name].cancel()

        worker = Worker(fn, *args, **kwargs)

        self.tasks[name] = worker

        self.task_started.emit(name)

        worker.signals.progress.connect(self.task_progress)

        worker.signals.message.connect(self.task_message)

        worker.signals.error.connect(self.task_error)

        def done(result):

            self.tasks.pop(name, None)

            self.task_finished.emit(name)

            if finished is not None:
                finished(result)

        worker.signals.finished.connect(done)

        self.pool.start(worker)

        return worker

    def cancel(self, name):

        if name in self.tasks:

            self.tasks[name].cancel()

    def cancel_all(self):

        for worker in self.tasks.values():

            worker.cancel()