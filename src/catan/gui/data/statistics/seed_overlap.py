"""Overlap of the seeds a session would export for seeded redetection.

The seeds of a session (catan.core.redetection.build_seeds) are its
included footprints plus, for each included neuron without a footprint
there, its union footprint placed at the interpolated position (inferred
footprint). Two strongly overlapping seeds compete for the same pixels in
the redetection. Same measure as footprint_overlap: binary supports above
``support_threshold`` of each seed's peak, shared pixels normalized by the
union (IoU), the smaller seed, seed i or seed j.
"""

from types import SimpleNamespace

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from catan.core.redetection import infer_aligned_footprint
from catan.gui.background_tasks.runtime import current_task_context, reports_progress

from .overlap_statistics import check_normalization, overlap_denominator
from .queries import ReductionSpec
from .sparse_pairs import SparsePairCollector
from .union_similarity import NAMES, _union_inputs

SEED_KINDS = ("all", "inferred")


def _session_seeds(data, ids, session_id, neurons, support_threshold):
    """Binary supports (one column per neuron), their pixel counts and
    whether each seed is inferred. Missing seeds have no pixels."""
    session = data.sessions[session_id]
    union = data.assignments.union
    n_pixels = session.footprints.shape[0]

    columns = []
    inferred = np.zeros(len(neurons), dtype=bool)

    for k, neuron in enumerate(neurons.tolist()):
        footprint = int(ids[neuron, session_id])

        if footprint >= 0:
            column = (
                session.footprints[:, footprint]
                if session.included[footprint]
                else None
            )
        else:
            column = infer_aligned_footprint(
                data.sessions, ids, union, session_id, neuron
            )
            inferred[k] = column is not None

        if column is None:
            columns.append(sparse.csc_matrix((n_pixels, 1), dtype=np.float32))
            continue

        column = sparse.csc_matrix(column, dtype=np.float32)
        peak = column.max()
        support = column >= support_threshold * peak if peak > 0 else column > 0
        columns.append(sparse.csc_matrix(support, dtype=np.float32))

    supports = sparse.hstack(columns, format="csc")
    masses = np.asarray(supports.sum(axis=0)).ravel()
    return supports, masses, inferred


def seed_overlap_values(
    data,
    state,
    ni,
    nj,
    sid,
    *,
    normalize_by="smaller",
    support_threshold=0.1,
    seeds="all",
):
    """Overlap of the seeds of ni and nj in session sid; NaN where a seed
    is missing (excluded footprint, nothing to infer) or, with
    ``seeds="inferred"``, where both seeds are detected footprints."""
    if not 0 <= support_threshold < 1:
        raise ValueError("Support threshold must be in [0, 1).")
    if seeds not in SEED_KINDS:
        raise ValueError(f"Unknown seed selection {seeds!r}.")
    check_normalization(normalize_by)

    ni, nj, sid = (np.asarray(v, dtype=np.int64).reshape(-1) for v in (ni, nj, sid))
    values = np.full(len(ni), np.nan, dtype=np.float32)
    ids = np.asarray(state.assignments)
    ctx = current_task_context()

    for session_id in np.unique(sid).tolist():
        if ctx is not None:
            ctx.check_cancelled()

        rows = np.flatnonzero(sid == session_id)
        neurons = np.unique(np.concatenate((ni[rows], nj[rows])))
        supports, masses, inferred = _session_seeds(
            data, ids, session_id, neurons, support_threshold
        )

        fi = np.searchsorted(neurons, ni[rows])
        fj = np.searchsorted(neurons, nj[rows])

        intersection = np.asarray(
            supports[:, fi].multiply(supports[:, fj]).sum(axis=0)
        ).ravel()
        denominator = overlap_denominator(
            normalize_by, intersection, masses[fi], masses[fj]
        )

        valid = (masses[fi] > 0) & (masses[fj] > 0) & (denominator > 0)
        if seeds == "inferred":
            valid &= inferred[fi] | inferred[fj]

        result = np.full(len(rows), np.nan, dtype=np.float32)
        np.divide(intersection, denominator, out=result, where=valid)
        values[rows] = np.clip(result, 0, 1)

    return values


@reports_progress("Calculating seed overlap")
def calculate_seed_overlap(
    data,
    state,
    indexers=None,
    filters=(),
    neighborhood_thr=10,
    **parameters,
):
    """Neuron pairs with union centroids closer than ``neighborhood_thr`` px
    (each pair once, neuron_i < neuron_j), in every session."""
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

    if collector.relations.get("neuron", "all") == "same":
        return collector.result()

    _, positions, usable = _union_inputs(data, n)
    neurons = np.intersect1d(
        np.intersect1d(collector.domain("neuron_i"), collector.domain("neuron_j")),
        np.flatnonzero(usable),
    )

    if len(neurons) < 2:
        return collector.result()

    pairs = cKDTree(positions[neurons]).query_pairs(neighborhood_thr, output_type="ndarray")
    if not len(pairs):
        return collector.result()

    ni_all = neurons[pairs.min(axis=1)]
    nj_all = neurons[pairs.max(axis=1)]

    for session_id in collector.domain("session").tolist():
        session = data.sessions[session_id]
        if session is None or session.footprints is None:
            continue

        sid = np.full(len(ni_all), session_id, dtype=np.int64)
        values = seed_overlap_values(data, state, ni_all, nj_all, sid, **parameters)

        keep = np.isfinite(values)
        if not keep.any():
            continue

        collector.emit(
            {"neuron_i": ni_all[keep], "neuron_j": nj_all[keep], "session": sid[keep]},
            values[keep],
        )

    return collector.result()
