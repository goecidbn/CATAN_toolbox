from __future__ import annotations

from dataclasses import dataclass
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

    def cancelled(self) -> bool:
        return self.cancel_check()

    def check_cancelled(self) -> None:
        if self.cancelled():
            raise TaskCancelled()

    def progress(self, value: int) -> None:
        self.progress_callback(int(value))

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
