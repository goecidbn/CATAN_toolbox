"""Differences between the current assignments and another assignment set."""

import numpy as np
from scipy.spatial import cKDTree

from .calculations import apply_indexers


def other_assignment_names(data):
    """Choices: loaded assignment sets other than the current one."""
    current = getattr(data, "current_assignments", None)
    return tuple(
        name for name in getattr(data, "_assignments", {}) if name != current
    )


def _session_columns(ids, n_sessions):
    """Assignment ids with exactly n_sessions columns (missing: -1)."""
    ids = np.asarray(ids)
    out = np.full((ids.shape[0], n_sessions), -1, dtype=np.int64)
    width = min(ids.shape[1], n_sessions)
    out[:, :width] = ids[:, :width]
    return out


def best_matches(reference, other, reference_centroids, other_centroids, max_distance):
    """Best-matching other neuron of each reference neuron; -1: none.

    The match shares most footprints (ties: larger overlap of the
    footprint sets, then nearest union centroid), as
    AssignmentComparison.partners(). Without shared footprints, the
    nearest other union centroid within ``max_distance`` px.
    """
    n_reference, n_sessions = reference.shape
    rows, sessions = np.nonzero(reference >= 0)

    # Owner of each (session, footprint) in the other set.
    owner = {}
    for neuron, session in zip(*np.nonzero(other >= 0)):
        owner[(int(session), int(other[neuron, session]))] = int(neuron)

    partners = np.asarray(
        [
            owner.get((int(session), int(reference[row, session])), -1)
            for row, session in zip(rows, sessions)
        ],
        dtype=np.int64,
    )

    shared = partners >= 0
    pairs, counts = np.unique(
        np.column_stack([rows[shared], partners[shared]]),
        axis=0,
        return_counts=True,
    ) if shared.any() else (np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=int))

    match = np.full(n_reference, -1, dtype=np.int64)

    if len(pairs):
        distance = np.zeros(len(pairs))
        if reference_centroids is not None and other_centroids is not None:
            distance = np.linalg.norm(
                reference_centroids[pairs[:, 0]] - other_centroids[pairs[:, 1]],
                axis=1,
            )

        sizes_reference = np.count_nonzero(reference >= 0, axis=1)
        sizes_other = np.count_nonzero(other >= 0, axis=1)
        jaccard = counts / (
            sizes_reference[pairs[:, 0]] + sizes_other[pairs[:, 1]] - counts
        )

        # Per reference neuron: most shared footprints, largest overlap,
        # then nearest.
        order = np.lexsort((distance, -jaccard, -counts, pairs[:, 0]))
        first = np.unique(pairs[order, 0], return_index=True)[1]
        best = pairs[order[first]]
        match[best[:, 0]] = best[:, 1]

    unmatched = np.flatnonzero((match < 0) & np.any(reference >= 0, axis=1))

    if (
        len(unmatched)
        and reference_centroids is not None
        and other_centroids is not None
    ):
        present = np.flatnonzero(np.any(other >= 0, axis=1))
        finite = present[np.isfinite(other_centroids[present]).all(axis=1)]

        if len(finite):
            query = reference_centroids[unmatched]
            valid = np.isfinite(query).all(axis=1)
            distance, index = cKDTree(other_centroids[finite]).query(
                query[valid], distance_upper_bound=max_distance
            )
            found = np.isfinite(distance)
            match[unmatched[valid][found]] = finite[index[found]]

    return match


def calculate_assignment_difference(
    data,
    state,
    indexers=None,
    filters=(),
    compare_with="",
    max_distance=10.0,
):
    """Sessions in which a neuron's footprints differ from its best match
    in the other assignment set (missing, additional or different
    footprints). Unmatched neurons count all their footprints."""
    sets = getattr(data, "_assignments", {})

    if not compare_with:
        others = other_assignment_names(data)

        if len(others) != 1:
            raise ValueError("Choose the assignment set to compare with.")

        # The only other loaded set.
        compare_with = others[0]

    if compare_with not in sets:
        raise ValueError(f"Assignment set {compare_with!r} is not loaded.")

    if compare_with == getattr(data, "current_assignments", None):
        raise ValueError("Choose an assignment set other than the current one.")

    # Each set with its own exclusions; synthetic components (specific to
    # a set) are not comparable (Data.comparison_ids).
    n_sessions = np.asarray(state.assignments).shape[1]
    other_set = sets[compare_with]
    if hasattr(data, "comparison_ids") and data.assignments is not None:
        reference = _session_columns(data.comparison_ids(data.assignments), n_sessions)
        other = _session_columns(data.comparison_ids(other_set), n_sessions)
    else:
        reference = np.asarray(state.assignments).copy()
        other = _session_columns(other_set.ids, n_sessions)

    for index, session in enumerate(data.sessions[:n_sessions]):
        # Inactive sessions are absent in both sets.
        if not getattr(session, "active", True):
            other[:, index] = -1
            reference[:, index] = -1

    # Excluded neurons of the other set are no partners.
    other_included = getattr(other_set.union, "included", None)
    if other_included is not None:
        other[~np.asarray(other_included, dtype=bool)[: len(other)]] = -1

    current = data.assignments
    reference_centroids = (
        None
        if current is None or current.union is None
        else np.asarray(current.union.centroids, dtype=float)
    )
    other_centroids = (
        None
        if other_set.union is None
        else np.asarray(other_set.union.centroids, dtype=float)
    )

    match = best_matches(
        reference, other, reference_centroids, other_centroids, max_distance
    )

    matched_rows = np.where(match[:, None] >= 0, other[np.maximum(match, 0)], -1)
    differences = np.sum(
        (reference != matched_rows) & ((reference >= 0) | (matched_rows >= 0)),
        axis=1,
    ).astype(float)

    # Empty rows (e.g. retired neurons) have no assignment to compare.
    differences[~np.any(reference >= 0, axis=1)] = np.nan

    return apply_indexers(differences, ("neuron",), indexers)
