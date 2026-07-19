"""
Module for calculating statistics from data
"""

import numpy as np
from typing import Dict, List, Optional, Callable

from catan.gui.structures.state import AppState
from catan.gui.structures.data import Data
from scipy import sparse, spatial

from catan.core.image_correlation import calculate_img_correlation


def get_stat_from_session(data, state, indexers, get_stat: Callable):

    session_ids = (
        range(len(data.sessions))
        if "session" not in indexers
        else [indexers["session"]]
    )
    stat = np.full((state.assignments.shape[0], len(session_ids)), np.nan)

    for i, session_id in enumerate(session_ids):
        session = data.sessions[session_id]
        stat_session = get_stat(session)

        neuron_ids = np.where(state.assignments[:, session_id] >= 0)[0]
        fp_ids = state.assignments[neuron_ids, session_id]

        stat[neuron_ids, i] = stat_session[fp_ids]

    if "session" in indexers:
        return np.squeeze(stat)
    return stat


def get_quality_metric(
    data: Data, state: AppState, indexers: Optional[Dict[str, int]] = None, **kwargs
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
        if key not in session.quality:
            raise KeyError(f"Quality metric '{key}' not found in session.")
        return session.quality[key]

    return get_stat_from_session(data, state, indexers, get_stat)


def normalize_csc_columns_to_max(A):
    A = A.tocsc(copy=True)

    col_max = A.max(axis=0).toarray().ravel()
    col_max = np.asarray(col_max, dtype=A.dtype)

    scale = np.ones(A.shape[1], dtype=A.dtype)
    nonzero = col_max > 0
    scale[nonzero] = 1.0 / col_max[nonzero]

    for j in range(A.shape[1]):
        start, end = A.indptr[j], A.indptr[j + 1]
        A.data[start:end] *= scale[j]

    return A


def calculate_footprint_size(
    data: Data, state: AppState, indexers: Optional[Dict[str, int]] = None, thr=0.01
) -> np.ndarray:
    """
    Calculate the size of each footprint.

    Returns:
    np.ndarray: 1D array of footprint sizes.
    """
    indexers = indexers or {}

    def get_size(session):
        footprints: sparse.csc_matrix = session.A
        footprints = normalize_csc_columns_to_max(footprints)

        return (footprints > thr).getnnz(axis=0)

    return get_stat_from_session(data, state, indexers, get_size)


def calculate_temporal_correlation(
    data: Data, state: AppState, indexers: Optional[Dict[str, int]] = None, key="C"
) -> np.ndarray:
    """
    Calculate the temporal correlation matrix for given traces.

    Parameters:
    traces (np.ndarray): 2D array where each row corresponds to a neuron's trace over time.

    Returns:
    np.ndarray: 2D correlation matrix.
    """

    indexers = indexers or {}

    session_ids = (
        range(len(data.sessions))
        if "session" not in indexers
        else [indexers["session"]]
    )

    N, S = state.assignments.shape

    correlations = np.full((N, N, len(session_ids)), np.nan)
    for i, session_id in enumerate(session_ids):
        session = data.sessions[session_id]

        if session is None or key not in session.traces:
            # traces are not necessarily loaded
            continue
        neuron_ids = np.where(state.assignments[:, session_id] >= 0)[0]
        fp_ids = state.assignments[neuron_ids, session_id]

        traces_neurons = session.traces[key][fp_ids, :]

        corr = np.corrcoef(traces_neurons)
        correlations[np.ix_(neuron_ids, neuron_ids) + (i,)] = corr

    if "session" in indexers:
        return np.squeeze(correlations)
    return correlations


def calculate_distances(
    data: Data, state: AppState, indexers: Optional[Dict[str, int]] = None
) -> np.ndarray:
    """
    Calculate the pairwise Euclidean distances between centroids.

    Parameters:
    centroids (np.ndarray): 2D array where each row corresponds to a neuron's centroid (x, y).

    Returns:
    np.ndarray: 2D distance matrix.
    """
    from scipy.spatial.distance import pdist, squareform

    indexers = indexers or {}

    session_ids = (
        range(len(data.sessions))
        if "session" not in indexers
        else [indexers["session"]]
    )

    N, S = state.assignments.shape

    distances = np.full((N, N, len(session_ids)), np.nan)
    for i, session_id in enumerate(session_ids):
        session = data.sessions[session_id]
        if session is None or session.centroids is None:
            # centroids are not necessarily loaded
            continue
        neuron_ids = np.where(state.assignments[:, session_id] >= 0)[0]
        fp_ids = state.assignments[neuron_ids, session_id]

        centroids = session.centroids[fp_ids, :]

        dist_matrix = squareform(pdist(centroids, metric="euclidean"))
        np.fill_diagonal(dist_matrix, np.nan)  # Set self-distance to nan
        distances[np.ix_(neuron_ids, neuron_ids) + (i,)] = dist_matrix

    if "session" in indexers:
        return np.squeeze(distances)
    return distances

    # if centroids.ndim != 2 or centroids.shape[1] != 2:
    #     raise ValueError("Centroids should be a 2D array with shape (n_neurons, 2).")
    # dist_matrix = squareform(pdist(centroids, metric="euclidean"))
    # np.fill_diagonal(dist_matrix, np.nan)  # Set self-distance to nan
    # return dist_matrix


def calculate_border_proximity(
    data: Data, state: AppState, indexers: Optional[Dict[str, int]] = None
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


# def calculate_session_presence(assignments: np.ndarray) -> np.ndarray:
#     """
#     Calculate the presence of each neuron in each session.

#     Parameters:
#     assignments (np.ndarray): 2D array where each row corresponds to a neuron and each column corresponds to a session.
#                               The value is the footprint ID for that neuron in that session, or -1 if not present.

#     Returns:
#     np.ndarray: 2D boolean array indicating presence (True) or absence (False) of each neuron in each session.
#     """
#     # if assignments.ndim != 2:
#     #     raise ValueError(
#     #         "Assignments should be a 2D array with shape (n_neurons, n_sessions)."
#     #     )
#     # presence = np.sum(assignments >= 0, axis=0)  # Count non-negative footprint IDs
#     return assignments >= 0


def calculate_occurrence(
    data: Data, state: AppState, indexers: Optional[Dict[str, int]] = None
) -> np.ndarray:
    """
    Calculate the occurrence of each neuron across sessions.

    Parameters:

    Returns:
    np.ndarray: 1D array of occurrence counts for each neuron.
    """
    # if assignments.ndim != 2:
    #     raise ValueError(
    #         "Assignments should be a 2D array with shape (n_neurons, n_sessions)."
    #     )
    # occurrence = np.sum(assignments >= 0, axis=1)  # Count non-negative footprint IDs
    return state.assignments >= 0


def calculate_centroid_shift(
    data: Data, state: AppState, indexers: Optional[Dict[str, int]] = None
) -> np.ndarray:

    indexers = indexers or {}

    ctrs = np.array(data.neurons["centroids"], copy=True)
    ctrs_ref = ctrs[..., None, :]
    ctrs_target = ctrs[:, None, ...]

    if ref_idxed := ("session_j" in indexers):
        ctrs_ref = ctrs_ref[:, [indexers["session_j"]], ...]

    if target_idxed := ("session_i" in indexers):
        ctrs_target = ctrs_target[..., [indexers["session_i"]], :]

    shift = ctrs_ref - ctrs_target
    dshift = np.linalg.norm(shift, axis=-1)

    if ref_idxed:
        dshift = dshift[:, 0, :]
    if target_idxed:
        dshift = dshift[..., 0]

    return dshift


def calculate_footprint_similarity(
    data: Data,
    state: AppState,
    indexers: Optional[Dict[str, int]] = None,
    neighborhood_thr=10,
) -> np.ndarray:
    """
    Calculate the similarity between footprints across sessions.

    Parameters:

    Returns:
    np.ndarray: 2D array of similarity scores between footprints.
    """
    indexers = indexers or {}

    if "session_j" not in indexers or "session_i" not in indexers:
        raise ValueError(
            "Both 'session_j' and 'session_i' must be specified in indexers."
        )
    session_ref = data.sessions[indexers["session_j"]]
    session_target = data.sessions[indexers["session_i"]]

    centroid_distance = spatial.distance.cdist(
        session_ref.centroids, session_target.centroids
    )

    fp_similarity = np.full(centroid_distance.shape, np.nan)
    for j, i in zip(*np.where(centroid_distance < neighborhood_thr)):
        # print(
        #     f"Neuron {i} in session {indexers['session_j']} is close to neuron {j} in session {indexers['session_i']}. Distance: {centroid_distance[j,i]}"
        # )
        fp_similarity[j, i], _, shift = calculate_img_correlation(
            session_ref.A[:, j],
            session_target.A[:, i],
            crop=True,
            shift=True,
            mode="cosine_union",
            gamma=0.1,
            shift_optimized=True,
        )

    neuron_ids_ref = np.where(state.assignments[:, indexers["session_j"]] >= 0)[0]
    fp_ids_ref = state.assignments[neuron_ids_ref, indexers["session_j"]]

    neuron_ids_target = np.where(state.assignments[:, indexers["session_i"]] >= 0)[0]
    fp_ids_target = state.assignments[neuron_ids_target, indexers["session_i"]]

    fp_similarity_full = np.full(
        (state.assignments.shape[0], state.assignments.shape[0]), np.nan
    )
    fp_similarity_full[np.ix_(neuron_ids_ref, neuron_ids_target)] = fp_similarity[
        np.ix_(fp_ids_ref, fp_ids_target)
    ]
    return fp_similarity_full
