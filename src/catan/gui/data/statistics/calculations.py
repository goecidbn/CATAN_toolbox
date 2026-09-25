"""
Module for calculating statistics from data
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Callable


import numpy as np
from scipy import sparse, spatial

from catan.core.image_correlation import calculate_img_correlation
from catan.gui.background_tasks.runtime import current_task_context

if TYPE_CHECKING:
    from catan.gui.structures.state import AppState
    from catan.gui.structures.data import Data


def apply_indexers(
    values: np.ndarray,
    dims: tuple[str, ...],
    indexers: dict[str, int] | None = None,
):
    indexers = indexers or {}

    selection = tuple(indexers.get(dim, slice(None)) for dim in dims)

    return values[selection]


def requested_indices(
    size: int,
    dim: str,
    indexers: dict[str, int],
) -> np.ndarray:
    if dim in indexers:
        return np.asarray([indexers[dim]], dtype=int)

    return np.arange(size, dtype=int)


def drop_indexed_axes(
    values: np.ndarray,
    dims: tuple[str, ...],
    indexers: dict[str, int],
):
    """
    Used when the calculation itself was already restricted to indexed
    dimensions, so those axes now have length 1.
    """
    axes = tuple(axis for axis, dim in enumerate(dims) if dim in indexers)

    if axes:
        values = np.squeeze(values, axis=axes)

    return values


def get_stat_from_session(
    data: Data,
    state: AppState,
    indexers: dict[str, int] | None,
    get_stat: Callable,
):
    indexers = indexers or {}

    N, S = state.assignments.shape

    neuron_ids = requested_indices(N, "neuron", indexers)
    session_ids = requested_indices(S, "session", indexers)

    values = np.full(
        (len(neuron_ids), len(session_ids)),
        np.nan,
    )

    for j, session_id in enumerate(session_ids):

        fp_ids = state.assignments[neuron_ids, session_id]
        present = fp_ids >= 0

        # Registered-but-untracked sessions deliberately
        # have an empty assignment column.
        if not np.any(present):
            continue

        session = data.sessions[session_id]
        stat_session = get_stat(session)

        if stat_session is None:
            continue

        values[present, j] = stat_session[fp_ids[present]]

    return drop_indexed_axes(
        values,
        ("neuron", "session"),
        indexers,
    )


def get_quality_metric(
    data: Data,
    state: AppState,
    indexers: dict[str, int] | None = None,
    filters=(),
    **kwargs,
) -> np.ndarray:
    """
    Get a quality metric from the data.

    Parameters:
    key (str): The key of the quality metric to retrieve.

    Returns:
    np.ndarray: The requested quality metric.
    """
    indexers = indexers or {}
    key = kwargs.get("key", None)  # Default to 'snr' if no key is provided
    if key is None:
        raise ValueError("A 'key' must be provided to retrieve a quality metric.")

    # print(f"Retrieving quality metric '{key}' with indexers: {indexers}")

    def get_stat(session):
        return session.quality.get(key)

    return get_stat_from_session(data, state, indexers, get_stat)


def get_match_metric(
    data,
    state,
    indexers=None,
    filters=(),
    *,
    key: str,
    dims: tuple[str, ...],
):
    values = np.asarray(data.assignments.stats[key])

    return apply_indexers(
        values,
        dims,
        indexers,
    )


def normalize_csc_columns_to_max(A):

    A = A.tocsc(copy=True)

    if A.shape[0] == 0 or A.shape[1] == 0:
        return A

    col_max = A.max(axis=0).toarray().ravel()
    col_max = np.asarray(col_max, dtype=A.dtype)

    scale = np.ones(A.shape[1], dtype=A.dtype)

    nonzero = col_max > 0
    scale[nonzero] = 1.0 / col_max[nonzero]

    for j in range(A.shape[1]):
        start, end = (A.indptr[j], A.indptr[j + 1])

        A.data[start:end] *= scale[j]

    return A


def calculate_footprint_size(
    data: Data,
    state: AppState,
    indexers: dict[str, int] | None = None,
    filters=(),
    thr=0.01,
) -> np.ndarray:
    """
    Calculate the size of each footprint.

    Returns:
    np.ndarray: 1D array of footprint sizes.
    """
    indexers = indexers or {}

    def get_size(session):
        footprints: sparse.csc_matrix = session.footprints
        footprints = normalize_csc_columns_to_max(footprints)

        return (footprints > thr).getnnz(axis=0)

    return get_stat_from_session(data, state, indexers, get_size)


def calculate_temporal_correlation(
    data,
    state,
    indexers=None,
    filters=(),
    key="C",
):
    indexers = indexers or {}

    N, S = state.assignments.shape

    neuron_i = requested_indices(N, "neuron_i", indexers)
    neuron_j = requested_indices(N, "neuron_j", indexers)
    sessions = requested_indices(S, "session", indexers)

    values = np.full(
        (
            len(neuron_i),
            len(neuron_j),
            len(sessions),
        ),
        np.nan,
    )

    for k, session_id in enumerate(sessions):
        session = data.sessions[session_id]

        if session is None or key not in session.traces:
            continue

        fp_i = state.assignments[neuron_i, session_id]
        fp_j = state.assignments[neuron_j, session_id]

        valid_i = fp_i >= 0
        valid_j = fp_j >= 0

        traces_i = session.traces[key][fp_i[valid_i]]
        traces_j = session.traces[key][fp_j[valid_j]]

        if not valid_i.any() or not valid_j.any():
            continue

        # pair-wise Pearson correlation
        ti = traces_i - traces_i.mean(axis=1, keepdims=True)
        tj = traces_j - traces_j.mean(axis=1, keepdims=True)

        numerator = ti @ tj.T

        denominator = np.sqrt(
            np.sum(ti**2, axis=1)[:, None] * np.sum(tj**2, axis=1)[None, :]
        )

        corr = numerator / denominator

        values[
            np.ix_(
                valid_i,
                valid_j,
                [k],
            )
        ] = corr[..., None]

    return drop_indexed_axes(
        values,
        ("neuron_i", "neuron_j", "session"),
        indexers,
    )


def calculate_distances(
    data: Data,
    state: AppState,
    indexers: dict[str, int] | None = None,
    filters=(),
):
    indexers = indexers or {}

    N, S = state.assignments.shape

    neuron_i = requested_indices(N, "neuron_i", indexers)
    neuron_j = requested_indices(N, "neuron_j", indexers)
    sessions = requested_indices(S, "session", indexers)

    values = np.full(
        (
            len(neuron_i),
            len(neuron_j),
            len(sessions),
        ),
        np.nan,
    )

    for k, session_id in enumerate(sessions):
        session = data.sessions[session_id]

        if session is None or session.centroids is None:
            continue

        fp_i = state.assignments[neuron_i, session_id]
        fp_j = state.assignments[neuron_j, session_id]

        valid_i = fp_i >= 0
        valid_j = fp_j >= 0

        if not valid_i.any() or not valid_j.any():
            continue

        ctr_i = session.centroids[fp_i[valid_i]]
        ctr_j = session.centroids[fp_j[valid_j]]

        dist = spatial.distance.cdist(
            ctr_i,
            ctr_j,
        )

        values[
            np.ix_(
                valid_i,
                valid_j,
                [k],
            )
        ] = dist[..., None]

    # self-pairs
    for ii, n_i in enumerate(neuron_i):
        matches = np.where(neuron_j == n_i)[0]

        values[ii, matches, :] = np.nan

    return drop_indexed_axes(
        values,
        ("neuron_i", "neuron_j", "session"),
        indexers,
    )


def calculate_border_proximity(
    data: Data, state: AppState, indexers: dict[str, int] | None = None, filters=()
) -> np.ndarray:
    """
    Calculate the proximity of each centroid to the borders of the field of view.

    Parameters:

    Returns:
    np.ndarray: 1D array of border proximities.
    """
    # print("dims:", data.sessions[0].dims)
    # print("centroids:", data.sessions[0].centroids)

    def get_distances(session):
        width, height = session.dims

        ctrs = session.centroids

        x_proximity = np.minimum(ctrs[:, 0], width - ctrs[:, 0])
        y_proximity = np.minimum(ctrs[:, 1], height - ctrs[:, 1])
        return np.minimum(x_proximity, y_proximity)

    return get_stat_from_session(data, state, indexers, get_distances)


def calculate_occurrence(
    data,
    state,
    indexers=None,
    filters=(),
):
    return apply_indexers(
        state.assignments >= 0,
        ("neuron", "session"),
        indexers,
    )


def calculate_centroid_shift(
    data: Data,
    state: AppState,
    indexers: dict[str, int] | None = None,
    filters=(),
) -> np.ndarray:

    indexers = indexers or {}

    N, S = state.assignments.shape

    neuron_ids = requested_indices(N, "neuron", indexers)
    session_i_ids = requested_indices(S, "session_i", indexers)
    session_j_ids = requested_indices(S, "session_j", indexers)

    coords_i = np.full(
        (len(neuron_ids), len(session_i_ids), 2),
        np.nan,
    )

    coords_j = np.full(
        (len(neuron_ids), len(session_j_ids), 2),
        np.nan,
    )

    def fill_coords(
        target: np.ndarray,
        session_ids: np.ndarray,
    ):
        for k, session_id in enumerate(session_ids):
            session = data.sessions[session_id]

            if session is None or session.centroids is None:
                continue

            # global neuron ID -> local footprint ID
            fp_ids = state.assignments[
                neuron_ids,
                session_id,
            ]

            present = fp_ids >= 0

            target[present, k, :] = session.centroids[fp_ids[present], :]

    fill_coords(
        coords_i,
        session_i_ids,
    )

    fill_coords(
        coords_j,
        session_j_ids,
    )

    # neuron × session_i × session_j
    values = np.linalg.norm(
        coords_i[:, :, None, :] - coords_j[:, None, :, :],
        axis=-1,
    )

    return drop_indexed_axes(
        values,
        (
            "neuron",
            "session_i",
            "session_j",
        ),
        indexers,
    )


def calculate_footprint_similarity(
    data: Data,
    state: AppState,
    indexers=None,
    filters=(),
    neighborhood_thr=10,
):
    indexers = indexers or {}

    ctx = current_task_context()

    N, S = state.assignments.shape

    neuron_i_ids = requested_indices(N, "neuron_i", indexers)
    neuron_j_ids = requested_indices(N, "neuron_j", indexers)

    session_i_ids = requested_indices(S, "session_i", indexers)
    session_j_ids = requested_indices(S, "session_j", indexers)

    values = np.full(
        (
            len(neuron_i_ids),
            len(neuron_j_ids),
            len(session_i_ids),
            len(session_j_ids),
        ),
        np.nan,
    )

    session_relation = get_pair_relation(
        filters,
        "session",
    )

    for si, session_i_id in enumerate(session_i_ids):
        session_i = data.sessions[session_i_id]

        if (
            session_i is None
            or session_i.centroids is None
            or session_i.footprints is None
        ):
            continue

        fp_i = state.assignments[
            neuron_i_ids,
            session_i_id,
        ]

        valid_i = fp_i >= 0
        pos_i = np.flatnonzero(valid_i)

        if not valid_i.any():
            continue

        for sj, session_j_id in enumerate(session_j_ids):
            ctx.progress(
                int(
                    (si * len(session_j_ids) + sj)
                    / (len(session_i_ids) * len(session_j_ids))
                    * 100
                )
            )
            ctx.check_cancelled()
            session_j = data.sessions[session_j_id]

            if (
                session_j is None
                or session_j.centroids is None
                or session_j.footprints is None
            ):
                continue

            if session_relation == "same" and session_i_id != session_j_id:
                continue

            if session_relation == "different" and session_i_id == session_j_id:
                continue

            if session_relation == "with previous" and session_j_id != session_i_id - 1:
                continue

            fp_j = state.assignments[
                neuron_j_ids,
                session_j_id,
            ]

            valid_j = fp_j >= 0
            pos_j = np.flatnonzero(valid_j)

            if not valid_j.any():
                continue

            ctr_i = session_i.centroids[fp_i[valid_i]]
            ctr_j = session_j.centroids[fp_j[valid_j]]

            distances = spatial.distance.cdist(
                ctr_i,
                ctr_j,
            )

            # Do not calculate similarity of a neuron with itself
            # within the same session.
            if session_i_id == session_j_id:
                valid_neuron_i = neuron_i_ids[valid_i]
                valid_neuron_j = neuron_j_ids[valid_j]

                self_pairs = valid_neuron_i[:, None] == valid_neuron_j[None, :]

                distances[self_pairs] = np.inf

            similarity = np.full(
                distances.shape,
                np.nan,
            )

            for ii, jj in zip(*np.where(distances < neighborhood_thr)):
                similarity[ii, jj], _, _ = calculate_img_correlation(
                    session_i.footprints[:, fp_i[valid_i][ii]],
                    session_j.footprints[:, fp_j[valid_j][jj]],
                    crop=True,
                    shift=True,
                    mode="cosine_union",
                    gamma=0.1,
                    shift_optimized=True,
                )

            values[
                np.ix_(
                    pos_i,
                    pos_j,
                    [si],
                    [sj],
                )
            ] = similarity[:, :, None, None]

    return drop_indexed_axes(
        values,
        (
            "neuron_i",
            "neuron_j",
            "session_i",
            "session_j",
        ),
        indexers,
    )


def get_pair_relation(
    filters,
    target: str,
) -> str:
    for f in filters:
        if f.target == target:
            return f.relation

    return "all"
