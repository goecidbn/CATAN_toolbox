from typing import Tuple
import numpy as np

from scipy import signal, sparse
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
    A1 = _as_dense_image(A1, dims)
    A2 = _as_dense_image(A2, dims)

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
        # A1 = A1.reshape(dims) if not np.all(A1.shape == dims) else A1
        # A2 = A2.reshape(dims) if not np.all(A2.shape == dims) else A2
        # t_end = time.time()
        # print('reshaping --- time taken: %5.3g'%(t_end-t_start))

        if crop:
            A1, A2 = crop_to_common_bbox(A1, A2)
        else:
            A1 = np.array(A1) if not type(A1) is np.ndarray else A1
            A2 = np.array(A2) if not type(A2) is np.ndarray else A2

        dims = A1.shape

        if mode == "correlation":
            C_max, C_zscored, img_shift = _from_correlation(A1, A2)

        elif mode == "cosine_union":
            C_max, C_zscored, img_shift = _from_cosine_union(A1, A2, **kwargs)
        elif mode == "cosine_weighted":
            C_max, C_zscored, img_shift = _from_weighted_cosine(A1, A2, **kwargs)
        elif mode in ("cosine", "pearson"):
            score_map = calculate_shift_score_map(
                A1,
                A2,
                mode=mode,
                **kwargs,
            )
            C_max, C_zscored, img_shift = subpixel_shift_from_score_map(
                score_map,
                A2.shape,
                window_radius=3,
                threshold_rel=0.1,
            )

        else:
            raise ValueError(f"Invalid mode: {mode}")

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

        if A1.shape != A2.shape:
            raise ValueError(
                f"Images must have identical shapes; got {A1.shape} and {A2.shape}."
            )

        if A1.size == 0:
            return np.nan, None, None

        if not (np.all(np.isfinite(A1)) and np.all(np.isfinite(A2))):
            raise ValueError("Images contain NaN or Inf values.")

        if mode == "cosine_union":
            options = dict(kwargs)
            options["shift_optimized"] = False
            return _from_cosine_union(A1, A2, **options)

        elif mode == "cosine_weighted":
            return _from_weighted_cosine(A1, A2, **kwargs)

        elif mode in ("pearson", "correlation"):
            x = A1.ravel() - A1.mean()
            y = A2.ravel() - A2.mean()

        elif mode == "cosine":
            x = A1.ravel()
            y = A2.ravel()

        else:
            raise ValueError(f"Invalid mode: {mode}")

        nx = np.linalg.norm(x)
        ny = np.linalg.norm(y)

        if nx == 0.0 or ny == 0.0:
            return np.nan, None, None

        score = float(np.dot(x / nx, y / ny))
        return score, None, None
        # return float(np.clip(score, -1.0, 1.0)), None, None
        # return (
        #     (A1 * A2).sum() / np.sqrt((A1**2).sum() * (A2**2).sum()),
        #     None,
        #     None,
        # )


def _from_correlation(A1, A2, *, return_score_map=False):
    C = signal.convolve(A1 - A1.mean(), A2[::-1, ::-1] - A2.mean(), mode="full") / (
        np.prod(A1.shape) * A1.std() * A2.std()
    )
    # allowed_mask = _shift_mask(A1.shape, A2.shape, expected_shift, max_shift_radius)

    if return_score_map:
        return C
    return subpixel_shift_from_score_map(
        C, A2.shape, window_radius=3, threshold_rel=0.1
    )


def _from_cosine_union(
    A1,
    A2,
    thr1=0.0,
    thr2=0.0,
    gamma=0.1,
    eps=1e-12,
    shift_optimized=True,
    return_score_map=False,
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
        if return_score_map:
            return robust_map

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


def _from_weighted_cosine(
    A1,
    A2,
    *,
    outside_weight=0.1,
    reference_threshold=0.1,
    reference_percentile=99.0,
):
    if not np.isfinite(outside_weight) or not 0 <= outside_weight <= 1:
        raise ValueError("outside_weight must be between 0 and 1.")

    if not np.isfinite(reference_threshold) or not 0 <= reference_threshold <= 1:
        raise ValueError("reference_threshold must be between 0 and 1.")

    if not np.isfinite(reference_percentile) or not 0 < reference_percentile <= 100:
        raise ValueError("reference_percentile must be in (0, 100].")

    # Use positive pixels: a sparse projection can have a whole-image
    # 95th percentile of zero.
    positive = A1[A1 > 0]
    if positive.size == 0:
        return np.nan, np.nan, (0.0, 0.0)

    reference_scale = np.percentile(positive, reference_percentile)
    threshold = reference_threshold * reference_scale
    mask = (A1 > 0) & (A1 >= threshold)

    sqrt_w = np.sqrt(np.where(mask, 1.0, outside_weight))
    x = (A1 * sqrt_w).ravel()
    y = (A2 * sqrt_w).ravel()

    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)

    if nx == 0 or ny == 0:
        return np.nan, np.nan, (0.0, 0.0)

    score = float(np.dot(x / nx, y / ny))
    return float(np.clip(score, -1.0, 1.0)), np.nan, (0.0, 0.0)


def calculate_shift_score_map(
    A1,
    A2,
    *,
    mode="correlation",
    min_overlap=0.25,
    **kwargs,
):
    """Return the full translation score surface for two 2D images."""
    A = np.asarray(A1, dtype=np.float64)
    B = np.asarray(A2, dtype=np.float64)

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Expected two 2D images.")

    if not np.isfinite(A).all() or not np.isfinite(B).all():
        raise ValueError("Images must contain only finite values.")

    if mode == "correlation":
        if A.std() == 0 or B.std() == 0:
            shape = tuple(a + b - 1 for a, b in zip(A.shape, B.shape))
            return np.full(shape, np.nan)

        return _from_correlation(A, B, return_score_map=True)

    if mode == "cosine_union":
        return _from_cosine_union(
            A,
            B,
            return_score_map=True,
            **kwargs,
        )

    if mode in ("cosine", "pearson"):
        return _additional_score_map(
            A,
            B,
            mode=mode,
            min_overlap=min_overlap,
        )

    raise ValueError(f"Invalid correlation mode: {mode}")


def _additional_score_map(A, B, *, mode, min_overlap):
    """Ordinary cosine or overlap-specific Pearson correlation."""
    if not 0 < min_overlap <= 1:
        raise ValueError("min_overlap must be in (0, 1].")

    def corr(a, b):
        return fftconvolve(a, b[::-1, ::-1], mode="full")

    ones_a = np.ones_like(A)
    ones_b = np.ones_like(B)

    # Number of overlapping image pixels at each translation.
    n = np.rint(np.maximum(corr(ones_a, ones_b), 0))
    valid = n >= max(3, min_overlap * min(A.size, B.size))
    result = np.full(n.shape, np.nan)

    if mode == "cosine":
        # Whole-image norms: shifted images are interpreted as zero-padded.
        numerator = corr(A, B)
        denominator = np.sqrt(np.sum(A * A) * np.sum(B * B))

        if denominator > 0:
            result[valid] = np.clip(numerator[valid] / denominator, -1, 1)

        return result

    # Subtracting global means improves numerical conditioning.
    # Local overlap means are still removed below.
    A = A - A.mean()
    B = B - B.mean()

    sa = corr(A, ones_b)
    sb = corr(ones_a, B)
    safe_n = np.maximum(n, 1)

    va = np.maximum(corr(A * A, ones_b) - sa * sa / safe_n, 0)
    vb = np.maximum(corr(ones_a, B * B) - sb * sb / safe_n, 0)
    covariance = corr(A, B) - sa * sb / safe_n

    # Exclude constant overlaps and numerical cancellation near zero.
    tiny = np.finfo(float).tiny
    valid &= va > 1e-12 * max(float(np.sum(A * A)), tiny)
    valid &= vb > 1e-12 * max(float(np.sum(B * B)), tiny)

    result[valid] = np.clip(
        covariance[valid] / np.sqrt(va[valid] * vb[valid]),
        -1,
        1,
    )

    return result


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
    if not np.isfinite(score_map).any():
        return np.nan, np.nan, (np.nan, np.nan)

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
    w = np.nan_to_num(
        patch,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
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

    score_std = float(np.nanstd(score_map))
    score_zscored = (
        float((score_max - np.nanmedian(score_map)) / score_std)
        if score_std > 0
        else np.nan
    )

    return score_max, score_zscored, shift


def _as_dense_image(value, dims):
    if sparse.issparse(value):
        value = value.toarray()

    image = np.asarray(value, dtype=np.float64)

    # Individual footprints may arrive as flattened sparse columns.
    if image.ndim == 1 or (
        image.ndim == 2 and 1 in image.shape and image.shape != tuple(dims)
    ):
        if image.size != int(np.prod(dims)):
            raise ValueError(
                f"Flattened image has {image.size} pixels; "
                f"expected {int(np.prod(dims))} for dims={dims}."
            )
        image = image.reshape(dims)

    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got {image.shape}.")

    return image
