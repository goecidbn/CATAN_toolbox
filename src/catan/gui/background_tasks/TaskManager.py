from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import count
from typing import Callable

from PySide6.QtCore import QObject, QThreadPool, QTimer, Signal

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
    task_failed = Signal(str, str)             # group, task_id

    task_progress = Signal(str, str, int)      # group, task_id, progress
    task_message = Signal(str, str, str)       # group, task_id, message
    task_error = Signal(str, str, str)         # group, task_id, traceback

    queue_changed = Signal(str)

    GROUPS = (
        "loading",
        "model update",
        "calculating",
    )

    def __init__(self):
        super().__init__()

        self.pool = QThreadPool.globalInstance()

        self.queues: dict[str, deque[QueuedTask]] = {
            group: deque()
            for group in self.GROUPS
        }

        self.current: dict[str, QueuedTask | None] = {
            group: None
            for group in self.GROUPS
        }

        self.tasks: dict[str, QueuedTask] = {}

        self._id_counter = count(1)

        # Periodically re-evaluate ready conditions.
        self._queue_timer = QTimer(self)
        self._queue_timer.setInterval(1000)
        self._queue_timer.timeout.connect(
            self.process_queues
        )

    # ------------------------------------------------------------------
    # Timer
    # ------------------------------------------------------------------

    def start_queue_timer(self) -> None:
        if not self._queue_timer.isActive():
            self._queue_timer.start()

    def stop_queue_timer(self) -> None:
        self._queue_timer.stop()

    # ------------------------------------------------------------------
    # Task creation
    # ------------------------------------------------------------------

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
    ) -> str:

        if group not in self.queues:
            raise ValueError(
                f"Unknown task group {group!r}. "
                f"Expected one of {self.GROUPS}."
            )

        worker = Worker(
            fn,
            *args,
            **kwargs,
        )

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

        self.task_added.emit(
            group,
            task.id,
        )

        self.queue_changed.emit(group)

        self._start_next(group)

        return task.id

    # ------------------------------------------------------------------
    # Queue processing
    # ------------------------------------------------------------------

    def process_queues(self) -> None:
        for group in self.GROUPS:
            self._start_next(group)

    def _task_is_ready(
        self,
        task: QueuedTask,
    ) -> bool:

        if task.ready is None:
            return True

        try:
            return bool(task.ready())

        except Exception as exc:
            # A broken ready() callback should not crash the Qt event loop.
            # Treat it as "not ready" and report it.
            self.task_error.emit(
                task.group,
                task.id,
                (
                    f"Error evaluating readiness condition "
                    f"for task {task.name!r}: {exc}"
                ),
            )

            return False

    def _start_next(
        self,
        group: str,
    ) -> None:

        # Only one task per group at once.
        if self.current[group] is not None:
            return

        queue = self.queues[group]

        runnable_index = None

        for i, task in enumerate(queue):

            # Cancelled/removed tasks may still physically exist
            # in the deque.
            if task.id not in self.tasks:
                continue

            if self._task_is_ready(task):
                runnable_index = i
                break

        if runnable_index is None:
            self.queue_changed.emit(group)
            return

        task = queue[runnable_index]
        del queue[runnable_index]

        # The task may have been cancelled immediately before promotion.
        if task.id not in self.tasks:
            self._start_next(group)
            return

        self.current[group] = task

        worker = task.worker

        # --------------------------------------------------------------
        # Forward worker signals
        # --------------------------------------------------------------

        worker.signals.progress.connect(
            lambda progress,
            g=group,
            tid=task.id:
                self.task_progress.emit(
                    g,
                    tid,
                    progress,
                )
        )

        worker.signals.message.connect(
            lambda message,
            g=group,
            tid=task.id:
                self.task_message.emit(
                    g,
                    tid,
                    message,
                )
        )

        worker.signals.error.connect(
            lambda error,
            g=group,
            tid=task.id:
                self._on_worker_error(
                    g,
                    tid,
                    error,
                )
        )

        worker.signals.finished.connect(
            lambda result,
            t=task:
                self._on_worker_finished(
                    t,
                    result,
                )
        )

        self.task_started.emit(
            group,
            task.id,
        )

        self.queue_changed.emit(group)

        self.pool.start(worker)

    # ------------------------------------------------------------------
    # Worker completion/error handling
    # ------------------------------------------------------------------

    def _on_worker_error(
        self,
        group: str,
        task_id: str,
        error: str,
    ) -> None:

        print(
            f"Task {task_id} in group {group} "
            f"raised an error:\n{error}"
        )

        self.task_error.emit(
            group,
            task_id,
            error,
        )

    def _on_worker_finished(
        self,
        task: QueuedTask,
        result,
    ) -> None:

        group = task.group
        worker = task.worker

        # Remove from active task lookup.
        self.tasks.pop(
            task.id,
            None,
        )

        if self.current[group] is task:
            self.current[group] = None

        cancelled = worker.is_cancelled()
        failed = worker.is_failed()

        try:
            if cancelled:
                self.task_cancelled.emit(
                    group,
                    task.id,
                )

            elif failed:
                self.task_failed.emit(
                    group,
                    task.id,
                )

            else:
                self.task_finished.emit(
                    group,
                    task.id,
                )

                # Only successful tasks execute their completion callback.
                if task.finished is not None:
                    task.finished()

        finally:
            self.queue_changed.emit(group)

            # Release the queue and immediately look for another
            # runnable task.
            self._start_next(group)

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def cancel(
        self,
        task_id: str,
    ) -> None:

        task = self.tasks.get(task_id)

        if task is None:
            return

        group = task.group

        # Running task:
        # request cooperative cancellation, but keep it registered and
        # current until Worker.finished fires.
        if self.current[group] is task:
            task.worker.cancel()

            # Do NOT emit task_cancelled yet:
            # the worker is still running.
            self.queue_changed.emit(group)

            return

        # Queued task:
        # it can be removed immediately.
        self.tasks.pop(
            task.id,
            None,
        )

        self.task_cancelled.emit(
            group,
            task.id,
        )

        self.queue_changed.emit(group)

        # If this happened while the queue was idle,
        # another ready task may now be runnable.
        self._start_next(group)

    def cancel_group(
        self,
        group: str,
    ) -> None:

        if group not in self.queues:
            raise ValueError(
                f"Unknown task group {group!r}"
            )

        current = self.current[group]

        if current is not None:
            self.cancel(current.id)

        # Copy because cancel() mutates self.tasks.
        for task in list(
            self.queued_tasks(group)
        ):
            self.cancel(task.id)

        # Remove cancelled stale entries from physical queue.
        self.queues[group] = deque(
            task
            for task in self.queues[group]
            if task.id in self.tasks
        )

        self.queue_changed.emit(group)

    def cancel_all(self) -> None:
        for group in self.GROUPS:
            self.cancel_group(group)

    # ------------------------------------------------------------------
    # Queue inspection
    # ------------------------------------------------------------------

    def queued_tasks(
        self,
        group: str,
    ) -> list[QueuedTask]:

        return [
            task
            for task in self.queues[group]
            if task.id in self.tasks
        ]

    def current_task(
        self,
        group: str,
    ) -> QueuedTask | None:

        return self.current[group]

    def get_task(
        self,
        task_id: str,
    ) -> QueuedTask | None:

        return self.tasks.get(task_id)

    # ------------------------------------------------------------------
    # Reordering
    # ------------------------------------------------------------------

    def move_queued_task(
        self,
        group: str,
        old_index: int,
        new_index: int,
    ) -> None:
        """
        Reorder a queued task.

        The currently running task is not included in this indexing.
        """

        tasks = self.queued_tasks(group)

        if not (
            0 <= old_index < len(tasks)
        ):
            return

        if not (
            0 <= new_index < len(tasks)
        ):
            return

        task = tasks.pop(old_index)
        tasks.insert(
            new_index,
            task,
        )

        self.queues[group] = deque(tasks)

        self.queue_changed.emit(group)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def group_summary(
        self,
        group: str,
    ) -> dict:

        current = self.current_task(group)
        queued = self.queued_tasks(group)

        return {
            "running": current is not None,
            "current_name": (
                current.name
                if current is not None
                else None
            ),
            "current_id": (
                current.id
                if current is not None
                else None
            ),
            "current_cancelling": (
                current.worker.is_cancelled()
                if current is not None
                else False
            ),
            "queued_count": len(queued),
        }