from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import count
from typing import Callable

from PySide6.QtCore import QObject, Signal, QThreadPool, QTimer

from .Worker import Worker


@dataclass
class QueuedTask:
    id: str
    name: str
    group: str
    worker: Worker
    finished: Callable | None = None
    ready: Callable[[], bool] | None = None

class TaskManager(QObject):
    task_added = Signal(str, str)              # group, task_id
    task_started = Signal(str, str)            # group, task_id
    task_finished = Signal(str, str)           # group, task_id
    task_cancelled = Signal(str, str)          # group, task_id
    task_progress = Signal(str, str, int)      # group, task_id, progress
    task_message = Signal(str, str, str)       # group, task_id, message
    task_error = Signal(str, str, str)         # group, task_id, error

    queue_changed = Signal(str)

    GROUPS = ("loading", "model update", "calculating")

    def __init__(self):
        super().__init__()

        self.pool = QThreadPool.globalInstance()

        self.queues = {
            group: deque()
            for group in self.GROUPS
        }

        self.current = {
            group: None
            for group in self.GROUPS
        }

        self.tasks: dict[str, QueuedTask] = {}

        self._id_counter = count(1)

        self._queue_timer = QTimer(self)
        self._queue_timer.setInterval(1000)  # milliseconds
        self._queue_timer.timeout.connect(self.process_queues)

    def start_queue_timer(self):
        if not self._queue_timer.isActive():
            self._queue_timer.start()

    def stop_queue_timer(self):
        self._queue_timer.stop()

    def _new_task_id(self) -> str:
        return f"task-{next(self._id_counter):06d}"

    def start(
        self,
        group: str,
        name: str,
        fn,
        *args,
        finished=None,
        ready=None,
        **kwargs,
    ):

        if group not in self.queues:
            raise ValueError(f"Unknown task group {group!r}")

        # print("Starting task", name, "in group", group, "with args", args, "and kwargs", kwargs)
        worker = Worker(fn, *args, **kwargs)

        task = QueuedTask(
            id=self._new_task_id(),
            name=name,
            group=group,
            worker=worker,
            finished=finished,
            ready=ready,
        )

        self.tasks[task.id] = task
        self.queues[group].append(task)

        self.task_added.emit(group, task.id)
        self.queue_changed.emit(group)

        self._start_next(group)

        return task.id

    def process_queues(self):
        for group in self.GROUPS:
            self._start_next(group)

    def _start_next(self, group: str):
        # if self.current[group] is not None:
        #     return

        # queue = self.queues[group]

        # while queue:
        #     task = queue.popleft()

        #     if task.id not in self.tasks:
        #         continue

        #     self.current[group] = task
        #     break
        # else:
        #     self.queue_changed.emit(group)
        #     return

        if self.current[group] is not None:
            return

        queue = self.queues[group]

        runnable_index = None

        for i, task in enumerate(queue):
            # Skip cancelled tasks
            if task.id not in self.tasks:
                continue

            # No condition = immediately runnable
            if task.ready is None or task.ready():
                runnable_index = i
                break

        if runnable_index is None:
            self.queue_changed.emit(group)
            return

        task = queue[runnable_index]
        del queue[runnable_index]

        self.current[group] = task
        
        worker = task.worker

        self.task_started.emit(group, task.id)
        self.queue_changed.emit(group)

        worker.signals.progress.connect(
            lambda progress, g=group, tid=task.id:
                self.task_progress.emit(g, tid, progress)
        )

        worker.signals.message.connect(
            lambda message, g=group, tid=task.id:
                self.task_message.emit(g, tid, message)
        )

        def on_error(error, g=group, tid=task.id):
            print(f"Task {tid} in group {g} raised an error: {error}")
            self.task_error.emit(g, tid, error)

        worker.signals.error.connect(
            on_error
            # lambda error, g=group, tid=task.id:
                # self.task_error.emit(g, tid, error)
        )

        def done():
            self.tasks.pop(task.id, None)

            if self.current[group] is task:
                self.current[group] = None

            self.task_finished.emit(group, task.id)
            self.queue_changed.emit(group)

            try:
                if task.finished is not None:
                    task.finished()
            finally:
                self._start_next(group)

        worker.signals.finished.connect(done)

        self.pool.start(worker)

    def cancel(self, task_id: str):
        task = self.tasks.pop(task_id, None)

        if task is None:
            return

        group = task.group

        if self.current[group] is task:
            task.worker.cancel()

        self.task_cancelled.emit(group, task.id)
        self.queue_changed.emit(group)

    def cancel_group(self, group: str):
        current = self.current[group]

        if current is not None:
            self.cancel(current.id)

        for task in list(self.queues[group]):
            self.cancel(task.id)

        self.queues[group].clear()

        self.queue_changed.emit(group)

    def queued_tasks(self, group: str) -> list[QueuedTask]:
        return [
            task
            for task in self.queues[group]
            if task.id in self.tasks
        ]

    def current_task(self, group: str) -> QueuedTask | None:
        return self.current[group]

    def get_task(self, task_id: str) -> QueuedTask | None:
        return self.tasks.get(task_id)

    def move_queued_task(
        self,
        group: str,
        old_index: int,
        new_index: int,
    ):
        """
        Reorder a queued task.

        Running task is not part of this indexing.
        """

        tasks = self.queued_tasks(group)

        if not (0 <= old_index < len(tasks)):
            return

        if not (0 <= new_index < len(tasks)):
            return

        task = tasks.pop(old_index)
        tasks.insert(new_index, task)

        self.queues[group] = deque(tasks)

        self.queue_changed.emit(group)

    def group_summary(self, group: str):
        current = self.current_task(group)
        queued = self.queued_tasks(group)

        return {
            "running": current is not None,
            "current_name": current.name if current else None,
            "current_id": current.id if current else None,
            "queued_count": len(queued),
        }