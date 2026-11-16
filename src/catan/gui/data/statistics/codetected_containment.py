"""Containment of a footprint in a co-detected neuron's other footprints.

For a session footprint of neuron_i and another neuron neuron_j that has
its own footprint in the same session: the median, over neuron_j's
footprints in the other sessions, of the share of the footprint's support
inside each of them (as footprint overlap normalized by i). High values: the
footprint lies inside where neuron_j is seen in most sessions while
neuron_j is detected in this session as well — an additional (often
partial) detection of the same cell.
"""

import warnings

import numpy as np
from scipy import sparse

from catan.gui.background_tasks.runtime import current_task_context, reports_progress

from .overlap_statistics import _prepare_session_supports
from .union_similarity import footprint_union_pairs, footprint_union_requested_pairs


def _supports(data, ids, keys, support_threshold):
    """Binary supports (pixels × len(keys)) and their sizes for
    (neuron, session) keys; columns in key order."""
    columns = []
    masses = []
    order = []

    for session_id in np.unique(keys[:, 1]).tolist():
        rows = np.flatnonzero(keys[:, 1] == session_id)
        session = data.sessions[session_id]
        a, _, mass, _ = _prepare_session_supports(
            session,
            session_id,
            keys[rows, 0],
            ids,
            support_threshold=support_threshold,
        )
        columns.append(a)
        masses.append(mass)
        order.append(rows)

    order = np.concatenate(order)
    position = np.empty(len(keys), dtype=np.int64)
    position[order] = np.arange(len(keys))
    matrix = sparse.hstack(columns, format="csc")[:, position]
    return matrix, np.concatenate(masses)[position]


def codetected_containment_values(
    data,
    state,
    ni,
    nj,
    sid,
    *,
    core_threshold=0.5,
    support_threshold=0.1,
    min_sessions=2,
):
    """Median share of the core of footprint (ni, sid) (pixels ≥
    ``core_threshold`` of its peak) inside the supports of nj's footprints
    in other sessions (≥ ``support_threshold`` of their peaks). NaN
    without footprint of ni or nj in sid, with fewer than ``min_sessions``
    other footprints of nj, or when nj has fewer footprints than ni (the
    established track is the reference, not the shorter one). ni = nj: the
    footprint inside its own neuron's other footprints (reference values)."""
    for threshold in (core_threshold, support_threshold):
        if not 0 <= threshold < 1:
            raise ValueError("Thresholds must be in [0, 1).")

    ni, nj, sid = (np.asarray(v, dtype=np.int64).reshape(-1) for v in (ni, nj, sid))
    values = np.full(len(ni), np.nan, dtype=np.float32)

    if not len(ni):
        return values

    ids = np.asarray(state.assignments)
    n_sessions = ids.shape[1]

    n_footprints = (ids >= 0).sum(axis=1)
    rows = np.flatnonzero(
        (ids[ni, sid] >= 0)
        & (ids[nj, sid] >= 0)
        & (n_footprints[nj] >= n_footprints[ni])
    )
    if not len(rows):
        return values

    # Footprints of nj in the other sessions, per row.
    present = ids[nj[rows]] >= 0
    present[np.arange(len(rows)), sid[rows]] = False
    counts = present.sum(axis=1)
    keep = counts >= max(1, int(min_sessions))
    rows, present, counts = rows[keep], present[keep], counts[keep]

    if not len(rows):
        return values

    ctx = current_task_context()
    if ctx is not None:
        ctx.check_cancelled()

    own_keys, own_index = np.unique(
        np.column_stack((ni[rows], sid[rows])), axis=0, return_inverse=True
    )
    row_of, session_of = np.nonzero(present)
    other_keys, other_index = np.unique(
        np.column_stack((nj[rows][row_of], session_of)), axis=0, return_inverse=True
    )

    own, own_mass = _supports(data, ids, own_keys, core_threshold)
    other, _ = _supports(data, ids, other_keys, support_threshold)

    if ctx is not None:
        ctx.check_cancelled()

    # Shared pixels of every needed (footprint, other footprint) pair.
    shared = (own.T @ other).tocsr()
    own_of_pair = own_index.reshape(-1)[row_of]
    inside = np.asarray(shared[own_of_pair, other_index.reshape(-1)]).ravel()
    mass = own_mass[own_of_pair]
    share = np.full(len(inside), np.nan)
    np.divide(inside, mass, out=share, where=mass > 0)

    # Median per row (rows padded to the most other sessions).
    table = np.full((len(rows), n_sessions), np.nan)
    table[row_of, session_of] = share
    with warnings.catch_warnings():
        # Rows whose footprint has no support (all NaN) stay NaN.
        warnings.simplefilter("ignore", RuntimeWarning)
        values[rows] = np.nanmedian(table, axis=1)

    return values


@reports_progress("Calculating containment in co-detected neurons")
def calculate_codetected_containment(
    data,
    state,
    indexers=None,
    filters=(),
    neighborhood_thr=10,
    **parameters,
):
    """Footprint/neuron pairs closer than ``neighborhood_thr`` px."""
    return footprint_union_pairs(
        data,
        state,
        indexers,
        filters,
        lambda ni, nj, sid: codetected_containment_values(
            data, state, ni, nj, sid, **parameters
        ),
        neighborhood_thr=neighborhood_thr,
    )


@reports_progress("Calculating requested containment in co-detected neurons")
def codetected_containment_requested_pairs(
    data,
    state,
    query,
    *,
    requested_pairs,
    neighborhood_thr=10,
    **parameters,
):
    return footprint_union_requested_pairs(
        state,
        query,
        requested_pairs,
        lambda ni, nj, sid: codetected_containment_values(
            data, state, ni, nj, sid, **parameters
        ),
    )
