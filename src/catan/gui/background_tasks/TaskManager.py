from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import count
from typing import Callable

from PySide6.QtCore import QObject, QThreadPool, QTimer, Signal

from .Worker import Worker


@dataclass(eq=False)
class TaskBatch:
    cancelled: bool = False


@dataclass
class QueuedTask:
    id: str
    name: str
    group: str
    worker: Worker
    finished: Callable | None = None
    on_result: Callable | None = None
    ready: Callable[[], bool] | None = None
    background: bool = False
    batch: TaskBatch | None = None


class TaskManager(QObject):

    task_added = Signal(str, str)  # group, task_id
    task_started = Signal(str, str)  # group, task_id

    task_finished = Signal(str, str)  # group, task_id
    task_cancelled = Signal(str, str)  # group, task_id
    task_failed = Signal(str, str)  # group, task_id

    task_progress = Signal(str, str, int)  # group, task_id, progress
    task_message = Signal(str, str, str)  # group, task_id, message
    task_error = Signal(str, str, str)  # group, task_id, traceback

    queue_changed = Signal(str)
    scheduling_settled = Signal()

    GROUPS = (
        "loading",
        "model update",
        "calculating",
        "saving",
    )

    def __init__(self):
        super().__init__()

        self.pool = QThreadPool.globalInstance()

        self.queues: dict[str, deque[QueuedTask]] = {
            group: deque() for group in self.GROUPS
        }

        self.current: dict[str, QueuedTask | None] = {
            group: None for group in self.GROUPS
        }

        self.tasks: dict[str, QueuedTask] = {}

        self._id_counter = count(1)

        # Periodically re-evaluate ready conditions.
        self._queue_timer = QTimer(self)
        self._queue_timer.setInterval(1000)
        self._queue_timer.timeout.connect(self.process_queues)

        self._pending_processing = None
        self._finishing_depth = 0
        self._processing_queues = False

        self._processing_holds = {}

    # ------------------------------------------------------------------
    # Timer
    # ------------------------------------------------------------------

    def start_queue_timer(self) -> None:
        if not self._queue_timer.isActive():
            self._queue_timer.start()

    def stop_queue_timer(self) -> None:
        self._queue_timer.stop()

    # ------------------------------------------------------------------
    # Status inquiries
    # ------------------------------------------------------------------

    @property
    def processing_requested(self):
        return self._pending_processing is not None

    @property
    def processing_paused(self):
        return bool(self._processing_holds)

    def processing_busy(self):
        """Processing tasks or an outstanding interactive loading check."""
        return bool(self._processing_holds) or any(
            not task.background for task in self.tasks.values()
        )

    def pause_processing(self, group, *, batch, on_cancel):
        """Prevent new workers from starting while a GUI check is pending."""
        if group not in self.queues:
            raise ValueError(f"Unknown task group {group!r}")

        if batch.cancelled:
            return None

        token = object()
        self._processing_holds[token] = (group, batch, on_cancel)
        self.queue_changed.emit(group)
        return token

    def resume_processing(self, token):
        hold = self._processing_holds.pop(token, None)
        if hold is None:
            return

        group, _, _ = hold
        self.queue_changed.emit(group)
        self.process_queues()
        self.scheduling_settled.emit()

    def _cancel_processing_holds(self, batch):
        for token, (group, held_batch, callback) in list(
            self._processing_holds.items()
        ):
            if held_batch is not batch:
                continue

            self._processing_holds.pop(token, None)
            callback()
            self.queue_changed.emit(group)

    def _background_running(self):
        return any(
            task is not None and task.background for task in self.current.values()
        )

    def defer_for_background(self, callback):
        # Keep the first explicit request while a display worker stops.
        if self._pending_processing is not None:
            return True

        # Existing processing conflicts are handled by the caller.
        if self.processing_busy() or not self._background_running():
            return False

        self._pending_processing = callback

        # Cancellation remains cooperative: do not clear current here.
        for task in list(self.current.values()):
            if task is not None and task.background:
                self.cancel(task.id)

        return True

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
        on_result=None,
        ready=None,
        unique=False,
        background=False,
        batch=None,
        prepend=False,
        **kwargs,
    ) -> str | None:

        if batch is not None and batch.cancelled:
            return None

        if group not in self.queues:
            raise ValueError(
                f"Unknown task group {group!r}. " f"Expected one of {self.GROUPS}."
            )

        if unique:
            for task in list(self.queued_tasks(group)):
                if task.name == name:
                    self.cancel(task.id)

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
            on_result=on_result,
            ready=ready,
            background=background,
            batch=batch,
        )

        self.tasks[task.id] = task

        if prepend:
            self.queues[group].appendleft(task)
        else:
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

    def process_queues(self):
        if self._finishing_depth or self._processing_queues:
            return

        self._processing_queues = True
        try:
            if (
                self._pending_processing is not None
                and not self.processing_busy()
                and not self._background_running()
            ):
                callback = self._pending_processing
                self._pending_processing = None
                callback()

            for group in self.GROUPS:
                self._start_next(group)
        finally:
            self._processing_queues = False

    def _task_is_ready(
        self,
        task: QueuedTask,
    ) -> bool:

        if task.batch is not None and task.batch.cancelled:
            return False

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

        # Completion callbacks may show a modal selection dialog.
        # Do not promote further work until those callbacks finish.
        if self._finishing_depth or self._processing_holds:
            return
        # Only one task per group at once.
        if self.current[group] is not None:
            return

        # A display worker reads live data. Let it stop before processing.
        if self._background_running():
            return

        queue = self.queues[group]

        runnable_index = None

        for i, task in enumerate(queue):

            # Cancelled/removed tasks may still physically exist
            # in the deque.
            if task.id not in self.tasks:
                continue

            if task.background and (
                self._finishing_depth
                or self._pending_processing is not None
                or self.processing_busy()
            ):
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
            lambda progress, g=group, tid=task.id: self.task_progress.emit(
                g,
                tid,
                progress,
            )
        )

        worker.signals.message.connect(
            lambda message, g=group, tid=task.id: self.task_message.emit(
                g,
                tid,
                message,
            )
        )

        worker.signals.error.connect(
            lambda error, g=group, tid=task.id: self._on_worker_error(
                g,
                tid,
                error,
            )
        )

        worker.signals.finished.connect(
            lambda result, t=task: self._on_worker_finished(
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

        print(f"Task {task_id} in group {group} " f"raised an error:\n{error}")

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

        self._finishing_depth += 1

        # Remove from active task lookup.
        self.tasks.pop(
            task.id,
            None,
        )

        if self.current[group] is task:
            self.current[group] = None

        cancelled = worker.is_cancelled() or (
            task.batch is not None and task.batch.cancelled
        )
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

                # Successful task returned a value.
                if task.on_result is not None:
                    task.on_result(result)

                # on_result may have opened a prompt and cancelled the batch.
                if task.finished is not None and not (
                    task.batch is not None and task.batch.cancelled
                ):
                    task.finished()

        finally:
            self.queue_changed.emit(group)

            # Publishing results and scheduling their continuation are
            # complete. Display calculations may now be reconsidered.
            self._finishing_depth -= 1
            self.process_queues()
            self.scheduling_settled.emit()

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
        self.scheduling_settled.emit()

    def cancel_group(self, group):
        if group not in self.queues:
            raise ValueError(f"Unknown task group {group!r}")

        self._finishing_depth += 1
        try:
            held_batches = {
                batch
                for held_group, batch, _ in self._processing_holds.values()
                if held_group == group
            }

            for batch in held_batches:
                self.cancel_batch(batch)

            for task in list(self.tasks.values()):
                if task.group == group:
                    self.cancel(task.id)

            self.queues[group] = deque(
                task for task in self.queues[group] if task.id in self.tasks
            )
        finally:
            self._finishing_depth -= 1

        self.queue_changed.emit(group)
        self.process_queues()
        self.scheduling_settled.emit()

    def cancel_all(self):
        self._finishing_depth += 1
        try:
            self._pending_processing = None
            for group in self.GROUPS:
                self.cancel_group(group)
        finally:
            self._finishing_depth -= 1

        self.process_queues()
        self.scheduling_settled.emit()

    def cancel_batch(self, batch: TaskBatch) -> None:
        """Cancel this batch without touching unrelated work."""
        if batch.cancelled:
            return

        # Set this before emitting signals or cancelling individual tasks:
        # callbacks must not be able to schedule more batch work.
        batch.cancelled = True

        self._finishing_depth += 1
        try:
            self._cancel_processing_holds(batch)

            for task in list(self.tasks.values()):
                if task.batch is batch:
                    self.cancel(task.id)

            for group, queue in self.queues.items():
                self.queues[group] = deque(
                    task for task in queue if task.id in self.tasks
                )
        finally:
            self._finishing_depth -= 1

        self.process_queues()
        self.scheduling_settled.emit()

    # ------------------------------------------------------------------
    # Queue inspection
    # ------------------------------------------------------------------

    def queued_tasks(
        self,
        group: str,
    ) -> list[QueuedTask]:

        return [task for task in self.queues[group] if task.id in self.tasks]

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

        if not (0 <= old_index < len(tasks)):
            return

        if not (0 <= new_index < len(tasks)):
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
            "current_name": (current.name if current is not None else None),
            "current_id": (current.id if current is not None else None),
            "current_cancelling": (
                current.worker.is_cancelled() if current is not None else False
            ),
            "queued_count": len(queued),
        }

    def describe_work(self):
        lines = []

        for group in self.GROUPS:
            current = self.current_task(group)
            if current is not None:
                lines.append(f"{group}: current {current.id} — {current.name}")

            for task in self.queued_tasks(group):
                lines.append(f"{group}: queued {task.id} — {task.name}")

        return "\n".join(lines) or "No current or queued tasks found."
