"""Assignment-based co-occurrence without loading images or traces."""

import numpy as np

from catan.gui.background_tasks.runtime import current_task_context

from .blockwise import evaluate_blockwise
from .dimensions import Dimension
from .distance_statistics import pair_plan
from .queries import ReductionSpec
from .types import StatisticArray


def _presence(data, state, query):
    """(ids, present (n, s): included footprint, active (n,): included neuron)."""
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

    return ids, present, active


def cooccurrence_query(
    data,
    state,
    query,
    *,
    select_values=None,
):
    ids, present, active = _presence(data, state, query)
    n, s = ids.shape

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


def cooccurrence_requested_pairs(data, state, query, *, requested_pairs):
    """Requested neuron pairs (sessions -1, curator progressive evaluation).

    The shared session axis is kept (one row per session), fixed or reduced
    as in the full evaluation; only the requested pairs are calculated.
    """
    from .requested_rows import evaluate_requested_rows

    reductions = query.reduction_dict()
    keep = ReductionSpec("keep")

    if any(
        reductions.get(name, keep).method not in ("keep", "single")
        for name in ("neuron_i", "neuron_j")
    ):
        raise ValueError("Requested pairs require unreduced neuron identities.")

    spec = reductions.get("session", keep)

    if spec.error_method != "none":
        raise ValueError("Requested pairs do not support error estimates.")

    pairs = np.asarray(requested_pairs, dtype=np.int64)

    if pairs.ndim != 2 or pairs.shape[1] != 4:
        raise ValueError("Requested pairs must have four coordinate columns.")

    ids, present, active = _presence(data, state, query)
    n, s = ids.shape

    pairs = pairs[(pairs[:, 2] == -1) & (pairs[:, 3] == -1)]
    reduced = spec.method not in ("keep", "single")
    names = ("neuron_i", "neuron_j") if reduced else ("neuron_i", "neuron_j", "session")
    dims, aliases, references, _, _ = pair_plan(names, n, s, query)

    def finish(values, ni, nj):
        values[~(active[ni] & active[nj])] = np.nan

        for pair_filter in query.filters:
            if pair_filter.target != "neuron":
                continue

            if pair_filter.relation == "same":
                values[ni != nj] = np.nan
            elif pair_filter.relation == "different":
                values[ni == nj] = np.nan
            elif pair_filter.relation != "all":
                raise ValueError(f"Unsupported pair filter: {pair_filter}")

        return values

    if reduced:

        def read_rows(refs):
            ni, nj = refs["neuron_i"], refs["neuron_j"]
            matrix = present[ni] & present[nj] & (ids[ni] != ids[nj])
            # Same reduction as StatisticArray.reduce_dimension.
            values = StatisticArray._compute_center(
                None, matrix.astype(np.float64), 1, spec.method
            )
            return finish(np.asarray(values, dtype=np.float64), ni, nj)

        refs = {"neuron_i": pairs[:, 0], "neuron_j": pairs[:, 1]}

    else:

        def read_rows(refs):
            ni, nj, sid = refs["neuron_i"], refs["neuron_j"], refs["session"]
            values = (
                present[ni, sid] & present[nj, sid] & (ids[ni, sid] != ids[nj, sid])
            ).astype(np.float64)
            return finish(values, ni, nj)

        sessions = (
            np.array([spec.index], dtype=np.int64)
            if spec.method == "single"
            else np.arange(s, dtype=np.int64)
        )
        refs = {
            "neuron_i": np.repeat(pairs[:, 0], len(sessions)),
            "neuron_j": np.repeat(pairs[:, 1], len(sessions)),
            "session": np.tile(sessions, len(pairs)),
        }

    result = evaluate_requested_rows(
        dims,
        read_rows,
        refs,
        reference_aliases=references,
        reduction_aliases=aliases,
        applied_filters=tuple(f.target for f in query.filters),
        block_rows=65_536,
    )

    if reduced:
        result.dimensions["session"] = Dimension(
            name="session",
            coords=np.arange(s),
            parameter={"method": spec.method, "error_method": "none"},
            mode="reduced",
        )
        result.validate()

    return result
