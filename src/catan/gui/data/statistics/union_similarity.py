"""Similarity of session footprints to the union footprints of neurons.

Compares a footprint of neuron_i in one session with the union footprint
of neuron_j (one comparison per neuron instead of one per stored
footprint). Same score as footprint similarity (cosine_union peak with
overlap penalty ``gamma``); the session footprint is the left patch.
"""

from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from catan.core.image_correlation import (
    compact_positive_footprints,
    footprint_similarity_peaks,
)
from catan.gui.background_tasks.runtime import current_task_context, reports_progress

from .queries import ReductionSpec
from .sparse_pairs import SparsePairCollector

NAMES = ("neuron_i", "neuron_j", "session")


def _positions_px(centroids, params):
    scale = float(params.get("pxtomu", 1.0))

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("pxtomu must be finite and positive.")

    return np.asarray(centroids, dtype=np.float64) / scale


def _union_inputs(data, n):
    union = data.assignments.union

    if (
        union is None
        or union.footprints is None
        or union.centroids is None
        or union.footprints.shape[1] != n
    ):
        raise ValueError(
            "Build the union before calculating footprint–union similarity."
        )

    positions = _positions_px(union.centroids, union.params)
    usable = np.all(np.isfinite(positions), axis=1)

    if union.included is not None:
        usable &= np.asarray(union.included, dtype=bool)

    return union, positions, usable


def union_similarity_values(
    data, state, ni, nj, sid, *, neighborhood_thr=10, gamma=0.1
):
    """Scores of rows (footprint of neuron ni in session sid, union of nj).

    NaN where the footprint is missing or excluded, the union neuron is
    excluded, or the centroids are not closer than ``neighborhood_thr`` px.
    """
    ni, nj, sid = (np.asarray(v, dtype=np.int64).reshape(-1) for v in (ni, nj, sid))
    values = np.full(len(ni), np.nan, dtype=np.float64)

    if not len(ni):
        return values

    ids = np.asarray(state.assignments)
    union, union_positions, usable = _union_inputs(data, ids.shape[0])

    ctx = current_task_context()
    lefts, rights, rows = [], [], []

    for session_id in np.unique(sid).tolist():
        if ctx is not None:
            ctx.check_cancelled()

        session = data.sessions[session_id]

        if session is None or session.footprints is None or session.centroids is None:
            continue

        if tuple(session.dims) != tuple(union.dims):
            raise ValueError(
                "Footprint–union similarity requires the same aligned pixel grid; "
                f"received {session.dims} and {union.dims}."
            )

        group = np.flatnonzero(sid == session_id)
        footprints = ids[ni[group], session_id]
        keep = (footprints >= 0) & usable[nj[group]]

        if session.included is not None:
            included = np.asarray(session.included, dtype=bool)
            keep[keep] = included[footprints[keep]]

        group, footprints = group[keep], footprints[keep]

        if not len(group):
            continue

        positions = _positions_px(session.centroids, session.params)[footprints]
        distance = np.linalg.norm(positions - union_positions[nj[group]], axis=1)
        close = np.isfinite(distance) & (distance < neighborhood_thr)
        group, footprints = group[close], footprints[close]

        if not len(group):
            continue

        unique = np.unique(footprints)
        patches = dict(
            zip(
                unique.tolist(),
                compact_positive_footprints(session.footprints, unique, session.dims),
            )
        )

        lefts.extend(patches[fp] for fp in footprints.tolist())
        rights.extend(nj[group].tolist())
        rows.extend(group.tolist())

    if not rows:
        return values

    neurons = np.unique(rights)
    union_patches = dict(
        zip(
            neurons.tolist(),
            compact_positive_footprints(union.footprints, neurons, union.dims),
        )
    )

    values[rows] = footprint_similarity_peaks(
        lefts, [union_patches[j] for j in rights], gamma=gamma
    )

    return values


@reports_progress("Calculating footprint–union similarity")
def calculate_footprint_union_similarity(
    data,
    state,
    indexers=None,
    filters=(),
    neighborhood_thr=10,
    gamma=0.1,
):
    """All footprint/union pairs closer than ``neighborhood_thr`` px."""
    return footprint_union_pairs(
        data,
        state,
        indexers,
        filters,
        lambda ni, nj, sid: union_similarity_values(
            data, state, ni, nj, sid, neighborhood_thr=neighborhood_thr, gamma=gamma
        ),
        neighborhood_thr=neighborhood_thr,
    )


def footprint_union_pairs(
    data, state, indexers, filters, values_of, *, neighborhood_thr=10
):
    """Footprint (neuron_i, session) / union (neuron_j) pairs closer than
    ``neighborhood_thr`` px, scored by ``values_of(ni, nj, session)``."""
    from .distance_statistics import pair_plan

    indexers = indexers or {}
    ids = np.asarray(state.assignments)
    n, s = ids.shape

    query = SimpleNamespace(
        reduction_dict=lambda: {
            dim: ReductionSpec("single", int(index)) for dim, index in indexers.items()
        },
        filters=tuple(filters),
    )
    dims, aliases, references, _, _ = pair_plan(NAMES, n, s, query)

    collector = SparsePairCollector(
        NAMES,
        dims,
        aliases,
        references,
        n,
        s,
        query,
        lambda values: np.ones(np.shape(values), dtype=bool),
    )
    relation = collector.relations.get("neuron", "all")

    _, union_positions, usable = _union_inputs(data, n)
    targets = collector.domain("neuron_j")
    targets = targets[usable[targets]]
    sources = collector.domain("neuron_i")

    ctx = current_task_context()
    tree = cKDTree(union_positions[targets]) if len(targets) else None

    for session_id in collector.domain("session").tolist():
        if ctx is not None:
            ctx.check_cancelled()

        session = data.sessions[session_id]

        if tree is None or session is None or session.centroids is None:
            continue

        present = sources[ids[sources, session_id] >= 0]

        if relation == "same":
            ni = nj = np.intersect1d(present, targets)
        else:
            positions = _positions_px(session.centroids, session.params)[
                ids[present, session_id]
            ]
            finite = np.all(np.isfinite(positions), axis=1)
            present, positions = present[finite], positions[finite]

            if not len(present):
                continue

            found = tree.query_ball_point(positions, neighborhood_thr)
            lengths = np.fromiter((len(f) for f in found), dtype=np.int64, count=len(found))

            if not lengths.sum():
                continue

            ni = np.repeat(present, lengths)
            nj = targets[np.concatenate([np.asarray(f, dtype=np.int64) for f in found if len(f)])]

            if relation == "different":
                distinct = ni != nj
                ni, nj = ni[distinct], nj[distinct]

        if not len(ni):
            continue

        sid = np.full(len(ni), session_id, dtype=np.int64)
        values = values_of(ni, nj, sid)

        collector.emit({"neuron_i": ni, "neuron_j": nj, "session": sid}, values)

    return collector.result()


@reports_progress("Calculating requested footprint–union similarity")
def footprint_union_similarity_requested_pairs(
    data,
    state,
    query,
    *,
    requested_pairs,
    neighborhood_thr=10,
    gamma=0.1,
):
    """Requested rows (source footprint, target union: session -1)."""
    return footprint_union_requested_pairs(
        state,
        query,
        requested_pairs,
        lambda ni, nj, sid: union_similarity_values(
            data, state, ni, nj, sid, neighborhood_thr=neighborhood_thr, gamma=gamma
        ),
    )


def footprint_union_requested_pairs(state, query, requested_pairs, values_of):
    """Requested footprint/union rows scored by ``values_of(ni, nj, session)``."""
    from .distance_statistics import pair_plan
    from .requested_rows import evaluate_requested_rows

    if any(
        spec.method not in ("keep", "single") for spec in query.reduction_dict().values()
    ):
        raise ValueError("Requested pairs require unreduced identities.")

    pairs = np.asarray(requested_pairs, dtype=np.int64)

    if pairs.ndim != 2 or pairs.shape[1] != 4:
        raise ValueError("Requested pairs must have four coordinate columns.")

    pairs = pairs[pairs[:, 3] == -1]
    n, s = np.asarray(state.assignments).shape
    dims, aliases, references, _, _ = pair_plan(NAMES, n, s, query)

    def read_rows(refs):
        ni, nj, sid = (refs[name] for name in NAMES)

        values = values_of(ni, nj, sid)

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

    return evaluate_requested_rows(
        dims,
        read_rows,
        {"neuron_i": pairs[:, 0], "neuron_j": pairs[:, 1], "session": pairs[:, 2]},
        reference_aliases=references,
        reduction_aliases=aliases,
        applied_filters=tuple(f.target for f in query.filters),
        block_rows=65_536,
    )


# ----------------------------------------------------------------------
# Union vs union
# ----------------------------------------------------------------------

UNION_NAMES = ("neuron_i", "neuron_j")


def union_union_similarity_values(data, state, ni, nj, *, neighborhood_thr=10, gamma=0.1):
    """Scores of rows (union of ni, union of nj).

    NaN for identical or excluded neurons and for union centroids not closer
    than ``neighborhood_thr`` px. The score is not bitwise symmetric: each
    unordered pair is scored once, lower neuron first.
    """
    ni, nj = (np.asarray(v, dtype=np.int64).reshape(-1) for v in (ni, nj))
    values = np.full(len(ni), np.nan, dtype=np.float64)

    if not len(ni):
        return values

    n = np.asarray(state.assignments).shape[0]
    union, positions, usable = _union_inputs(data, n)

    valid = (ni != nj) & usable[ni] & usable[nj]
    rows = np.flatnonzero(valid)

    distance = np.linalg.norm(positions[ni[rows]] - positions[nj[rows]], axis=1)
    rows = rows[np.isfinite(distance) & (distance < neighborhood_thr)]

    if not len(rows):
        return values

    left = np.minimum(ni[rows], nj[rows])
    right = np.maximum(ni[rows], nj[rows])
    keys, inverse = np.unique(np.column_stack((left, right)), axis=0, return_inverse=True)

    neurons = np.unique(keys)
    patches = dict(
        zip(
            neurons.tolist(),
            compact_positive_footprints(union.footprints, neurons, union.dims),
        )
    )

    scores = footprint_similarity_peaks(
        [patches[a] for a in keys[:, 0].tolist()],
        [patches[b] for b in keys[:, 1].tolist()],
        gamma=gamma,
    )
    values[rows] = scores[np.asarray(inverse).reshape(-1)]

    return values


@reports_progress("Calculating union similarity")
def calculate_union_similarity(
    data,
    state,
    indexers=None,
    filters=(),
    neighborhood_thr=10,
    gamma=0.1,
):
    """All pairs of distinct union neurons closer than ``neighborhood_thr`` px."""
    from .distance_statistics import pair_plan

    indexers = indexers or {}
    n, s = np.asarray(state.assignments).shape

    query = SimpleNamespace(
        reduction_dict=lambda: {
            dim: ReductionSpec("single", int(index)) for dim, index in indexers.items()
        },
        filters=tuple(filters),
    )
    dims, aliases, references, _, _ = pair_plan(UNION_NAMES, n, s, query)

    collector = SparsePairCollector(
        UNION_NAMES,
        dims,
        aliases,
        references,
        n,
        s,
        query,
        lambda values: np.ones(np.shape(values), dtype=bool),
    )

    _, positions, usable = _union_inputs(data, n)
    sources = collector.domain("neuron_i")
    sources = sources[usable[sources]]
    targets = collector.domain("neuron_j")
    targets = targets[usable[targets]]

    if not len(sources) or not len(targets):
        return collector.result()

    found = cKDTree(positions[targets]).query_ball_point(
        positions[sources], neighborhood_thr
    )
    lengths = np.fromiter((len(f) for f in found), dtype=np.int64, count=len(found))

    if lengths.sum():
        ni = np.repeat(sources, lengths)
        nj = targets[np.concatenate([np.asarray(f, dtype=np.int64) for f in found if len(f)])]

        values = union_union_similarity_values(
            data, state, ni, nj, neighborhood_thr=neighborhood_thr, gamma=gamma
        )
        collector.emit({"neuron_i": ni, "neuron_j": nj}, values)

    return collector.result()


@reports_progress("Calculating requested union similarity")
def union_similarity_requested_pairs(
    data,
    state,
    query,
    *,
    requested_pairs,
    neighborhood_thr=10,
    gamma=0.1,
):
    """Requested rows between two union endpoints (sessions -1)."""
    from .distance_statistics import pair_plan
    from .requested_rows import evaluate_requested_rows

    if any(
        spec.method not in ("keep", "single") for spec in query.reduction_dict().values()
    ):
        raise ValueError("Requested pairs require unreduced identities.")

    pairs = np.asarray(requested_pairs, dtype=np.int64)

    if pairs.ndim != 2 or pairs.shape[1] != 4:
        raise ValueError("Requested pairs must have four coordinate columns.")

    pairs = pairs[(pairs[:, 2] == -1) & (pairs[:, 3] == -1)]
    n, s = np.asarray(state.assignments).shape
    dims, aliases, references, _, _ = pair_plan(UNION_NAMES, n, s, query)

    def read_rows(refs):
        ni, nj = refs["neuron_i"], refs["neuron_j"]

        values = union_union_similarity_values(
            data, state, ni, nj, neighborhood_thr=neighborhood_thr, gamma=gamma
        )

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

    return evaluate_requested_rows(
        dims,
        read_rows,
        {"neuron_i": pairs[:, 0], "neuron_j": pairs[:, 1]},
        reference_aliases=references,
        reduction_aliases=aliases,
        applied_filters=tuple(f.target for f in query.filters),
        block_rows=65_536,
    )
