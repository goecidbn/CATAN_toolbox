"""Assignment-based co-occurrence without loading images or traces."""

import numpy as np

from catan.gui.background_tasks.runtime import current_task_context

from .blockwise import evaluate_blockwise
from .distance_statistics import pair_plan


def cooccurrence_query(
    data,
    state,
    query,
    *,
    select_values=None,
):
    ids = np.asarray(state.assignments)

    if ids.ndim != 2:
        raise ValueError(
            "Co-occurrence requires a neuron-by-session " "assignment table."
        )

    n, s = ids.shape

    if s == 0:
        raise ValueError("Co-occurrence requires at least one registered session.")

    if len(data.sessions) != s or any(session is None for session in data.sessions):
        raise ValueError(
            "Session metadata is incomplete; " "co-occurrence cannot be determined."
        )

    present = ids >= 0
    ctx = current_task_context()

    for sid, session in enumerate(data.sessions):
        if ctx is not None:
            ctx.check_cancelled()

        included = getattr(session, "included", None)

        if included is None:
            # Without an explicit mask, assignment presence is used.
            continue

        included = np.asarray(included, dtype=bool)

        if included.ndim != 1:
            raise ValueError(f"Session {sid} has an invalid inclusion mask.")

        rows = np.flatnonzero(present[:, sid])
        footprint_ids = ids[rows, sid]

        if np.any(footprint_ids != np.floor(footprint_ids)) or np.any(
            footprint_ids >= len(included)
        ):
            raise ValueError(
                f"Session {sid}: assignments reference " "invalid footprint IDs."
            )

        present[rows, sid] &= included[footprint_ids.astype(np.int64)]

    included = getattr(
        data.assignments.union,
        "included",
        None,
    )

    active = (
        np.ones(n, dtype=bool) if included is None else np.asarray(included, dtype=bool)
    )

    if active.shape != (n,):
        raise ValueError("Union inclusion mask does not match " "the assignment table.")

    for pair_filter in query.filters:
        if pair_filter.target == "session" and pair_filter.relation != "all":
            raise ValueError(
                "Co-occurrence has one shared session axis. "
                "Select or reduce that axis."
            )

    dimensions, aliases, references, coord, allowed = pair_plan(
        ("neuron_i", "neuron_j", "session"),
        n,
        s,
        query,
    )

    def read(selected):
        neuron_i = coord("neuron_i", selected)
        neuron_j = coord("neuron_j", selected)
        session = coord("session", selected)

        values = (
            present[neuron_i, session]
            & present[neuron_j, session]
            & (ids[neuron_i, session] != ids[neuron_j, session])
        )

        valid = allowed(selected) & active[neuron_i] & active[neuron_j]

        return np.where(
            valid,
            values.astype(np.float32),
            np.nan,
        )

    return evaluate_blockwise(
        dimensions,
        read,
        query,
        aliases=aliases,
        reference_aliases=references,
        applied_filters=tuple(pair_filter.target for pair_filter in query.filters),
        select_values=select_values,
    )


cooccurrence_query.supports_value_selection = True
