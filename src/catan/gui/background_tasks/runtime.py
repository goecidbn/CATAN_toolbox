from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from contextvars import ContextVar, Token
from contextlib import contextmanager
from functools import wraps
from typing import Protocol
from collections.abc import Callable


class TaskCancelled(Exception):
    pass


@dataclass
class TaskContext:
    cancel_check: Callable[[], bool]
    progress_callback: Callable[[int], None]
    message_callback: Callable[[str], None]

    _last_progress_value: int | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _last_progress_time: float = field(
        default=0.0,
        init=False,
        repr=False,
    )

    _last_message: str | None = field(default=None, init=False, repr=False)
    _pending_message: str | None = field(default=None, init=False, repr=False)
    _last_message_time: float = field(default=0.0, init=False, repr=False)

    def cancelled(self) -> bool:
        return self.cancel_check()

    def check_cancelled(self) -> None:
        if self.cancelled():
            raise TaskCancelled()

    def progress(self, value: int) -> None:
        self._flush_message(force=value >= 100)
        value = max(0, min(100, int(value)))
        previous = self._last_progress_value

        # Repeated integer percentages do not need another UI update.
        if value == previous:
            return

        now = monotonic()

        # Forward the first update, completion, and a new phase immediately.
        immediate = previous is None or value == 100 or value < previous

        if not immediate and now - self._last_progress_time < 0.5:
            return

        self._last_progress_value = value
        self._last_progress_time = now
        self.progress_callback(value)

    def message(self, text: str) -> None:
        self._pending_message = text
        self._flush_message()

    def _flush_message(self, *, force=False) -> None:
        text = self._pending_message
        if text is None or text == self._last_message:
            return
        now = monotonic()
        if not force and self._last_message is not None and now - self._last_message_time < 0.5:
            return
        self._last_message = text
        self._last_message_time = now
        self._pending_message = None
        self.message_callback(text)


class TaskRuntime(Protocol):

    def cancelled(self) -> bool: ...

    def check_cancelled(self) -> None: ...

    def progress(self, value: int) -> None: ...

    def message(self, value: str) -> None: ...


class NullTaskContext:
    def cancelled(self) -> bool:
        return False

    def check_cancelled(self) -> None:
        pass

    def progress(self, value: int) -> None:
        pass

    def message(self, text: str) -> None:
        pass


_null_context = NullTaskContext()

_current_task_context: ContextVar[TaskRuntime | None] = ContextVar(
    "current_task_context",
    default=None,
)


def bind_task_context(
    ctx: TaskRuntime,
) -> Token:
    return _current_task_context.set(ctx)


def reset_task_context(
    token: Token,
) -> None:
    _current_task_context.reset(token)


def current_task_context() -> TaskRuntime:
    return _current_task_context.get() or _null_context


# Percentages below describe work, not elapsed time. Each child reports in its
# own 0..100 range; only its successful exit completes the parent's allocation.
class _ProgressScope:
    def __init__(self, parent, start, stop):
        self.parent = parent
        self.start = start
        self.stop = stop
        self.high_water = 0.0

    def cancelled(self):
        return self.parent.cancelled()

    def check_cancelled(self):
        self.parent.check_cancelled()

    def progress(self, value):
        # Keep completion for __exit__, including validation/cancellation checks.
        self.high_water = max(self.high_water, min(99.999, max(0.0, float(value))))
        self.parent.progress(self.start + (self.stop - self.start) * self.high_water / 100)

    def message(self, text):
        self.parent.message(text)


@contextmanager
def progress_scope(start=0, stop=100, message=None):
    """Map nested reports into an allocation; never complete a failed stage."""
    parent = current_task_context()
    if isinstance(parent, NullTaskContext):
        yield parent
        return
    if not 0 <= start <= stop <= 100:
        raise ValueError("Progress bounds must satisfy 0 <= start <= stop <= 100")
    parent.check_cancelled()
    scope = _ProgressScope(parent, start, stop)
    token = _current_task_context.set(scope)
    try:
        if message:
            scope.message(message)
        scope.progress(0)
        yield scope
        scope.check_cancelled()
        parent.progress(stop)
    finally:
        _current_task_context.reset(token)


def reports_progress(message=None):
    """Give a calculation its own local range, preserving its public signature."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with progress_scope(message=message):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def progress_step(index, total, *, start=0, stop=100, message=None):
    """Allocate one item in a sized block/group loop."""
    width = (stop - start) / max(1, total)
    return progress_scope(start + width * index,
                          min(stop, start + width * (index + 1)), message)


def progress_iter(iterable, *, total=None, start=0, stop=100, every=128):
    """Report a loop without materializing it or binding context across yields.

    Use for leaf loops. For loops that call other reporting calculations, use
    an explicit progress_scope per block so the child's percentage is mapped.
    """
    ctx = current_task_context()
    if isinstance(ctx, NullTaskContext):
        yield from iterable
        return
    if total is None:
        total = len(iterable)
    if every < 1 or total < 0:
        raise ValueError("Invalid progress iteration parameters")
    ctx.check_cancelled()
    ctx.progress(start)
    for index, item in enumerate(iterable):
        if index % every == 0:
            ctx.check_cancelled()
            ctx.progress(start + (stop - start) * index / max(1, total))
        yield item
    ctx.check_cancelled()
    ctx.progress(stop)
