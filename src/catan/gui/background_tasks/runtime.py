from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from contextvars import ContextVar, Token
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

    def cancelled(self) -> bool:
        return self.cancel_check()

    def check_cancelled(self) -> None:
        if self.cancelled():
            raise TaskCancelled()

    def progress(self, value: int) -> None:
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
        self.message_callback(text)


class TaskRuntime(Protocol):

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

_current_task_context: ContextVar[TaskContext | None] = ContextVar(
    "current_task_context",
    default=None,
)


def bind_task_context(
    ctx: TaskContext,
) -> Token:
    return _current_task_context.set(ctx)


def reset_task_context(
    token: Token,
) -> None:
    _current_task_context.reset(token)


def current_task_context() -> TaskContext | NullTaskContext:
    return _current_task_context.get() or _null_context
