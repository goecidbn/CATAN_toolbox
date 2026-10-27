from tqdm.auto import tqdm
import numpy as np
from scipy import sparse, spatial

from catan.core.structures.session import SessionData
from catan.core.alignment import calculate_img_correlation
from catan.core.image_correlation import (
    compact_positive_footprints,
    footprint_pair_peaks,
)

from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# from threading import Lock


def calculate_statistics(
    this_session: SessionData,
    reference_session: SessionData,
    distance_threshold: float = 25.0,
    *,
    ctx=None,
    **kwargs,
):
    """
    function to calculate footprint correlations used for the matching procedure for
      - nearest neighbour (NN) - closest center of mass to reference footprint
      - non-nearest neighbour (nNN) - other neurons with center of mass distance below threshold

    footprint correlations are calculated as shifted and non-shifted (depending on model used)

    ``ctx`` (optional, see catan.core.progress) receives progress and is
    checked for cancellation.

    this method is used both for comparing statistics
      - referencing previous session (used for matching)
      - referencing itself (same-session evidence)

    Returns (shifts, distances, correlations), (reference, this) neurons.

    Pairs closer than ``distance_threshold`` (centroids) are compared:
    cross-session (and ``force_cross``) by the shift-optimized cosine-union
    similarity of all pairs at once (image_correlation.footprint_pair_peaks;
    the distance becomes the length of the optimal shift), same-session
    without shift from sparse products (_unshifted_cosine_union). Values
    equal calculate_img_correlation per pair.
    """

    if this_session.footprints is None or reference_session.footprints is None:
        raise ValueError(
            "Both this_session and reference_session must have A attribute set."
        )

    dims = tuple(int(size) for size in this_session.dims)

    if dims != tuple(int(size) for size in reference_session.dims):
        raise ValueError(
            "Footprint statistics require one common image grid; received "
            f"{this_session.dims} and {reference_session.dims}."
        )

    identical = this_session is reference_session or (
        this_session.footprints.shape == reference_session.footprints.shape
        and len(this_session.footprints.indices)
        == len(reference_session.footprints.indices)
        and np.all(
            this_session.footprints.indices == reference_session.footprints.indices
        )
        and np.all(
            this_session.footprints.indptr == reference_session.footprints.indptr
        )
        and np.allclose(this_session.footprints.data, reference_session.footprints.data)
    )

    # force_cross: measure same-session pairs like cross-session ones
    # (shift-optimized correlation), each unordered pair once.
    mode = "cross" if (not identical or kwargs.get("force_cross", False)) else "same"

    centroid_distance = spatial.distance.cdist(
        reference_session.centroids, this_session.centroids
    )
    n_reference, n_this = centroid_distance.shape

    centroid_shift = np.full((n_reference, n_this, 2), np.nan)
    footprint_correlation = np.full((n_reference, n_this), np.nan)

    with np.errstate(invalid="ignore"):
        candidates = (
            (centroid_distance < distance_threshold)
            & np.asarray(reference_session.included, dtype=bool)[:, None]
            & np.asarray(this_session.included, dtype=bool)[None, :]
        )
    if identical and mode == "cross":
        candidates &= np.arange(n_this)[None, :] > np.arange(n_reference)[:, None]

    I, J = np.nonzero(candidates)

    def report(fraction):
        if ctx is not None:
            ctx.check_cancelled()
            ctx.progress(100 * fraction)

    if mode == "same":
        report(0.0)
        scores = _unshifted_cosine_union(this_session.footprints, I, J, gamma=0.9)
        # A zero score (no common support) is no measurement (NaN), as per pair.
        measured = scores > 0
        footprint_correlation[I[measured], J[measured]] = scores[measured]
        centroid_shift[I[measured], J[measured]] = 0.0
        report(1.0)
        return centroid_shift, centroid_distance, footprint_correlation

    rows, row_origins = compact_positive_footprints(
        reference_session.footprints, np.unique(I), dims, origins=True
    )
    columns, column_origins = compact_positive_footprints(
        this_session.footprints, np.unique(J), dims, origins=True
    )
    row_index = {i: k for k, i in enumerate(np.unique(I))}
    column_index = {j: k for k, j in enumerate(np.unique(J))}
    a = [column_index[j] for j in J]
    b = [row_index[i] for i in I]

    report(0.0)
    peaks, shifts = footprint_pair_peaks(
        [columns[k] for k in a],
        [rows[k] for k in b],
        gamma=0.1,
        on_chunk=lambda done, total: report(done / total),
    )

    # Patch shifts -> image shifts: add the difference of patch positions.
    offsets = np.array(
        [
            (np.subtract(column_origins[ka], row_origins[kb]) if column_origins[ka] is not None and row_origins[kb] is not None else (np.nan, np.nan))
            for ka, kb in zip(a, b)
        ],
        dtype=float,
    ).reshape(-1, 2)
    shifts = shifts + offsets

    footprint_correlation[I, J] = peaks
    centroid_shift[I, J] = shifts
    centroid_distance[I, J] = np.hypot(shifts[:, 0], shifts[:, 1])

    return centroid_shift, centroid_distance, footprint_correlation


def _unshifted_cosine_union(footprints, I, J, *, gamma, eps=1e-12):
    """Cosine-union similarity without shift (calculate_img_correlation
    with mode="cosine_union", shift=True, shift_optimized=False) of
    footprints[:, J] (left) and footprints[:, I] (right) for all pairs at
    once: every term is a sum over pixels, i.e. a sparse product."""
    A = sparse.csc_matrix(footprints, dtype=np.float64, copy=True)
    A.sum_duplicates()
    A.data = np.where(A.data > 0, A.data, 0.0)
    A.eliminate_zeros()
    M = A.copy()
    M.data[:] = 1.0
    A2 = A.multiply(A).tocsc()

    def pair_sums(left, right):
        # sum over pixels of left[:, J] * right[:, I], pair by pair
        product = (right.T @ left).tocsr()
        return np.asarray(product[I, J]).ravel()

    N = pair_sums(A, A)
    S1 = pair_sums(A2, M)  # left squared over the right support
    S2 = pair_sums(M, A2)
    inter = pair_sums(M, M)
    support = np.asarray(M.sum(axis=0)).ravel()

    denominator = np.sqrt(S1 * S2)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.clip(N / denominator, 0.0, 1.0)
        overlap = np.clip(inter / np.maximum(np.minimum(support[J], support[I]), eps), 0.0, 1.0)
    score = cosine * np.power(overlap, gamma)
    # No common support or degenerate: 0.
    score[(inter == 0) | ~(denominator >= eps)] = 0.0
    return np.clip(score, 0.0, 1.0)


def calculate_p(d_ROIs, fp_corr, p_model, neighbor_distance=12):
    """
    evaluates the probability of neuron footprints belonging to the same neuron. It uses an interpolated version of p_same

    This function requires the successful building of a matching model first
    """

    ## evaluate probability-function for each of a neurons neighbors
    # print(neighbor_distance)
    neighbors = d_ROIs < neighbor_distance
    p_same = np.zeros_like(d_ROIs)
    p_same[neighbors] = p_model(d_ROIs[neighbors], fp_corr[neighbors])

    p_same[np.isnan(p_same)] = 0  # fill "bad" entries with zeros
    p_same = np.clip(
        p_same, 0, 1
    )  # function-shapes may allow for values exceeding [0,1]

    return sparse.csc_matrix(p_same)
