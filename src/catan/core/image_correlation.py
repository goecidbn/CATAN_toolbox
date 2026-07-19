from typing import Tuple
import numpy as np

from scipy import signal
from scipy.signal import fftconvolve

from .utils import crop_to_common_bbox


def calculate_img_correlation(
    A1: np.ndarray,
    A2: np.ndarray,
    dims: Tuple[int, int] = (512, 512),
    crop=False,
    cm_crop=None,
    binary=False,
    shift=True,
    mode="cosine_union",
    **kwargs,
):

    if shift:
        ## try with binary and continuous
        if binary:
            A1_nnz = A1[A1 > 0]
            A2_nnz = A2[A2 > 0]
            if binary == "half":
                A1 = A1 * (A1 > np.median(A1_nnz))
                A2 = A2 * (A2 > np.median(A2_nnz))
            else:
                A1 = A1 > np.median(A1_nnz)
                A2 = A2 > np.median(A2_nnz)

        # t_start = time.time()
        A1 = A1.reshape(dims) if not np.all(A1.shape == dims) else A1
        A2 = A2.reshape(dims) if not np.all(A2.shape == dims) else A2
        # t_end = time.time()
        # print('reshaping --- time taken: %5.3g'%(t_end-t_start))

        if crop:
            A1, A2 = crop_to_common_bbox(A1, A2)
        else:
            A1 = np.array(A1) if not type(A1) is np.ndarray else A1
            A2 = np.array(A2) if not type(A2) is np.ndarray else A2

        dims = A1.shape

        # t_start = time.time()
        if mode == "correlation":
            C_max, C_zscored, img_shift = _from_correlation(A1, A2)
        elif mode == "cosine_union":
            C_max, C_zscored, img_shift = _from_cosine_union(A1, A2, **kwargs)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        # print(
        #     f"mode: {mode}, C_max: {C_max}, C_zscored: {C_zscored}, img_shift: {img_shift}"
        # )
        # t_end = time.time()
        # print('corr-computation --- time taken: %5.3g'%(t_end-t_start))

        if np.isnan(C_max) | (C_max == 0):
            return np.nan, np.nan, np.ones(2) * np.nan

        return C_max, C_zscored, img_shift  # C[crop_half],
    else:

        if not (cm_crop is None):

            cr = 20
            extent = np.array([cm_crop - cr, cm_crop + cr + 1]).astype("int")
            extent = np.maximum(extent, 0)
            extent = np.minimum(extent, dims)
            A1 = A1.reshape(dims)[
                extent[0, 0] : extent[1, 0], extent[0, 1] : extent[1, 1]
            ]
            A2 = A2.reshape(dims)[
                extent[0, 0] : extent[1, 0], extent[0, 1] : extent[1, 1]
            ]
        return (
            (A1 * A2).sum() / np.sqrt((A1**2).sum() * (A2**2).sum()),
            None,
            None,
        )


def _from_correlation(A1, A2):
    C = signal.convolve(A1 - A1.mean(), A2[::-1, ::-1] - A2.mean(), mode="full") / (
        np.prod(A1.shape) * A1.std() * A2.std()
    )
    # allowed_mask = _shift_mask(A1.shape, A2.shape, expected_shift, max_shift_radius)

    return subpixel_shift_from_score_map(
        C, A2.shape, window_radius=3, threshold_rel=0.1
    )


def _from_cosine_union(
    A1, A2, thr1=0.0, thr2=0.0, gamma=0.1, eps=1e-12, shift_optimized=True
):

    M1 = (A1 > thr1).astype(float)
    M2 = (A2 > thr2).astype(float)

    A1m = A1 * M1
    A2m = A2 * M2

    if shift_optimized:

        # 2) Correlation map: Overlap-normalized cosine map (calculates union correlation)
        N = fftconvolve(A1m, A2m[::-1, ::-1], mode="full")

        S1 = fftconvolve(A1m**2, M2[::-1, ::-1], mode="full")
        S2 = fftconvolve(M1, (A2m**2)[::-1, ::-1], mode="full")

        # denominator terms restricted to active support
        denom_overlap_cos = np.sqrt(np.maximum(S1, 0.0) * np.maximum(S2, 0.0))
        overlap_cosine_map = np.zeros_like(N)
        valid = denom_overlap_cos > eps
        overlap_cosine_map[valid] = np.clip(
            N[valid] / denom_overlap_cos[valid], 0.0, 1.0
        )

        # 3) Weighting map: Overlap coefficient map on masks
        inter = fftconvolve(M1, M2[::-1, ::-1], mode="full")
        denom_overlap = max(min(M1.sum(), M2.sum()), eps)
        # more tolerant to contained subset instead of complete match
        overlap_coeff_map = np.clip(inter / denom_overlap, 0.0, 1.0)

        robust_map = overlap_cosine_map * np.power(overlap_coeff_map, gamma)

        return subpixel_shift_from_score_map(
            robust_map, A2.shape, window_radius=3, threshold_rel=0.1
        )
    else:
        overlap = M1.astype(bool) & M2.astype(bool)

        if not np.any(overlap):
            # print("No overlap between masks.")
            return 0.0, np.nan, (0.0, 0.0)

        N = np.sum(A1m[overlap] * A2m[overlap])

        # denominator terms
        S1 = np.sum(A1m[overlap] ** 2)
        S2 = np.sum(A2m[overlap] ** 2)

        denom = np.sqrt(S1 * S2)
        if denom < eps:
            # print("Denominator is too small.")
            return 0.0, np.nan, (0.0, 0.0)

        overlap_cosine = np.clip(N / denom, 0.0, 1.0)

        # 3) Weighting map: Overlap coefficient map on masks
        inter = np.count_nonzero(overlap)
        denom_overlap = max(min(M1.sum(), M2.sum()), eps)
        overlap_coeff = np.clip(inter / denom_overlap, 0.0, 1.0)

        robust_score = overlap_cosine * np.power(overlap_coeff, gamma)

        return (
            float(np.clip(robust_score, 0.0, 1.0)),
            np.nan,
            (0.0, 0.0),
        )  # no shift optimization in this branch


def subpixel_shift_from_score_map(
    score_map, shape2, window_radius=2, threshold_rel=0.5
):
    """
    Estimate subpixel shift from a local weighted centroid around the peak.

    Parameters
    ----------
    score_map : 2D array
        Robust score map on the full correlation grid.
    shape2 : tuple
        Shape of the second footprint, used to convert map index -> shift.
    window_radius : int
        Radius of local window around peak. 2 means a 5x5 window.
    threshold_rel : float
        Keep only values >= threshold_rel * local_max inside the window.

    Returns
    -------
    dict with:
        dy, dx          : subpixel shift
        dy_int, dx_int  : integer peak shift
        peak_score      : max score
    """
    score_map = np.asarray(score_map, dtype=float)

    iy0, ix0 = np.unravel_index(np.nanargmax(score_map), score_map.shape)
    score_max = float(score_map[iy0, ix0])

    y0 = max(0, iy0 - window_radius)
    y1 = min(score_map.shape[0], iy0 + window_radius + 1)
    x0 = max(0, ix0 - window_radius)
    x1 = min(score_map.shape[1], ix0 + window_radius + 1)

    patch = score_map[y0:y1, x0:x1]

    # local coordinates in full-map index space
    Y, X = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing="ij")

    # threshold relative to local peak
    w = patch.copy()
    w[w < threshold_rel * score_max] = 0.0

    # optional baseline subtraction
    positive = w > 0
    if np.any(positive):
        w[positive] = w[positive] - w[positive].min()

    if w.sum() <= 0:
        iy = float(iy0)
        ix = float(ix0)
    else:
        iy = float((Y * w).sum() / w.sum())
        ix = float((X * w).sum() / w.sum())

    # convert map indices to shifts
    dy = iy - (shape2[0] - 1)
    dx = ix - (shape2[1] - 1)

    shift = (dy, dx)

    score_zscored = float((score_max - np.nanmedian(score_map)) / np.nanstd(score_map))

    return score_max, score_zscored, shift
