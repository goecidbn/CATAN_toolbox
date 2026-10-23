"""Overlap of session footprints with the union footprints of neurons.

Footprint of neuron_i in one session vs. the union footprint of neuron_j
(one comparison per neuron instead of one per stored footprint). Same
measure as footprint_overlap: supports above ``support_threshold`` of each
footprint's peak (binary, or peak-normalized intensities when weighted),
intersection = summed pixelwise minimum, normalized by the smaller
footprint (containment), the source / target (directional) or the union.
"""

import numpy as np

from catan.gui.background_tasks.runtime import current_task_context, reports_progress

from .overlap_statistics import _prepare_session_supports
from .union_similarity import footprint_union_pairs, footprint_union_requested_pairs


def footprint_union_overlap_values(
    data,
    state,
    ni,
    nj,
    sid,
    *,
    weighted=False,
    containment=True,
    directional=False,
    normalize_by_target=False,
    support_threshold=0.1,
):
    """Overlap of footprint (ni, sid) with union nj; NaN if unavailable."""
    if not 0 <= support_threshold < 1:
        raise ValueError("Support threshold must be in [0, 1).")

    ni, nj, sid = (np.asarray(v, dtype=np.int64) for v in (ni, nj, sid))
    values = np.full(len(ni), np.nan, dtype=np.float32)

    if not len(ni):
        return values

    ids = np.asarray(state.assignments)
    union = data.assignments.union

    # The union behaves like a session with one column per neuron.
    targets = np.unique(nj)
    b, _, mb, shape_b = _prepare_session_supports(
        union,
        0,
        targets,
        np.arange(len(ids))[:, None],
        weighted=weighted,
        support_threshold=support_threshold,
    )
    fj_all = np.searchsorted(targets, nj)

    ctx = current_task_context()

    for session_id in np.unique(sid).tolist():
        if ctx is not None:
            ctx.check_cancelled()

        rows = np.flatnonzero((sid == session_id) & (ids[ni, session_id] >= 0))

        session = data.sessions[session_id]

        if not len(rows) or session is None or session.footprints is None:
            continue

        sources = np.unique(ni[rows])
        a, _, ma, shape_a = _prepare_session_supports(
            session,
            session_id,
            sources,
            ids,
            weighted=weighted,
            support_threshold=support_threshold,
        )

        if shape_a != shape_b:
            raise ValueError("Overlap requires the same registered pixel grid.")

        fi = np.searchsorted(sources, ni[rows])
        fj = fj_all[rows]

        # Corresponding columns only; binary and weighted supports use min().
        intersection = np.asarray(a[:, fi].minimum(b[:, fj]).sum(axis=0)).ravel()

        if directional:
            denominator = mb[fj] if normalize_by_target else ma[fi]
        elif containment:
            denominator = np.minimum(ma[fi], mb[fj])
        else:
            denominator = ma[fi] + mb[fj] - intersection

        valid = (ma[fi] > 0) & (mb[fj] > 0) & (denominator > 0)

        result = np.full(len(rows), np.nan, dtype=np.float32)
        np.divide(intersection, denominator, out=result, where=valid)
        values[rows] = np.clip(result, 0, 1)

    return values


@reports_progress("Calculating footprint–union overlap")
def calculate_footprint_union_overlap(
    data,
    state,
    indexers=None,
    filters=(),
    neighborhood_thr=10,
    **parameters,
):
    """All footprint/union pairs closer than ``neighborhood_thr`` px."""
    return footprint_union_pairs(
        data,
        state,
        indexers,
        filters,
        lambda ni, nj, sid: footprint_union_overlap_values(
            data, state, ni, nj, sid, **parameters
        ),
        neighborhood_thr=neighborhood_thr,
    )


@reports_progress("Calculating requested footprint–union overlap")
def footprint_union_overlap_requested_pairs(
    data,
    state,
    query,
    *,
    requested_pairs,
    neighborhood_thr=10,
    **parameters,
):
    """Requested rows (source footprint, target union: session -1)."""
    return footprint_union_requested_pairs(
        state,
        query,
        requested_pairs,
        lambda ni, nj, sid: footprint_union_overlap_values(
            data, state, ni, nj, sid, **parameters
        ),
    )
