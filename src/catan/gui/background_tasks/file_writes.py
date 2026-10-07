from .runtime import TaskCancelled, current_task_context
from catan.core.io.isolated_read import (
    write_bytes,
    write_session_snapshots,
    write_prepared_file,
)


def save_bytes_task(state, path, payload):
    try:
        return write_bytes(
            path,
            payload,
            ctx=current_task_context(),
        )
    except TaskCancelled as exc:
        # Cancelled tasks normally suppress error dialogs. For writes,
        # the user still needs to know that completion is uncertain.
        info = getattr(exc, "failure_info", None)

        if info:
            state.issue(
                "warning",
                "Save cancelled",
                f"{info['summary']}\n\n{info['recovery']}",
            )
        raise


def save_session_snapshots_task(
    state,
    path,
    snapshots,
    *,
    mat_version="7.3",
):
    try:
        return write_session_snapshots(
            path,
            snapshots,
            mat_version=mat_version,
            ctx=current_task_context(),
        )
    except TaskCancelled as exc:
        info = getattr(exc, "failure_info", None)

        if info:
            state.issue(
                "warning",
                "Session save cancelled",
                f"{info['summary']}\n\n{info['recovery']}",
            )
        raise


def save_prepared_file_task(
    state,
    path,
    prepared,
    *,
    mat_version="7.3",
):
    try:
        return write_prepared_file(
            path,
            prepared,
            mat_version=mat_version,
            ctx=current_task_context(),
        )
    except TaskCancelled as exc:
        info = getattr(exc, "failure_info", None)

        if info:
            state.issue(
                "warning",
                "Save cancelled",
                f"{info['summary']}\n\n{info['recovery']}",
            )
        raise
