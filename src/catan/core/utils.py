import numpy as np
from typing import Tuple, Optional, List
from scipy import sparse
from scipy.ndimage import gaussian_filter


def normalize_array(A: np.ndarray, a_type="uint", a_bits=8, axis=None):
    A -= A.min()
    A = A / A.max()

    return (np.multiply(A, (A > A.mean(axis))) * (2**a_bits - 1)).astype(
        "%s%d" % (a_type, a_bits)
    )


def normalize_sparse_array(A, relative_threshold=0.001, minimum_nonzero_entries=50):
    """
    normalizing sparse arrays with some thresholding
      - relative_threshold: float
          fraction of peak value, at below which entries are considered to be 0
      - minimum_nonzero_entries: int
          minimum number of nonzero entries of the footprint to be considered for further analyses
    """
    return sparse.vstack(
        [
            # a.multiply(a>relative_threshold*a.max())/a.max()  # threshold footprint
            (
                a.multiply(a > relative_threshold * a.max())
                / a[a > 0.001 * a.max()].sum()  # threshold footprint
                if (a > 0).sum()
                > minimum_nonzero_entries  # require minimum non-zero entries ...
                else sparse.csr_matrix(a.shape)
            )  # ... otherwise return empty slice
            for a in A.T  # loop through all footprints
        ]
    ).T


def nangauss_filter(X, sigma=None, mode="nearest", truncate=2):
    if (sigma is None) or not np.any(np.array(sigma) > 0):
        return X
    else:
        V = X.copy()
        V[np.isnan(X)] = 0
        VV = gaussian_filter(V, sigma, truncate=truncate, mode=mode)

        W = 0 * X.copy() + 1
        W[np.isnan(X)] = 0
        WW = gaussian_filter(W, sigma, truncate=truncate, mode=mode)

    return VV / WW


def crop_to_common_bbox(A1, A2, m=0) -> tuple[np.ndarray, np.ndarray]:
    """
    Crop A1 and A2 to the common bounding box of their nonzero values, with optional padding m.
    Works for both dense and sparse (scipy.sparse) inputs.
    """

    A1 = A1.tocoo() if sparse.issparse(A1) else A1
    A2 = A2.tocoo() if sparse.issparse(A2) else A2

    ### get bounding box indices
    if sparse.issparse(A1):
        rows1, cols1 = A1.row, A1.col
    else:
        rows1, cols1 = np.where(A1 > 0.0)

    if sparse.issparse(A2):
        rows2, cols2 = A2.row, A2.col
    else:
        rows2, cols2 = np.where(A2 > 0.0)

    x_lims = (
        min(cols1.min(), cols2.min()) - m,
        max(cols1.max(), cols2.max()) + m + 1,
    )
    y_lims = (
        min(rows1.min(), rows2.min()) - m,
        max(rows1.max(), rows2.max()) + m + 1,
    )

    ### slice arrays to bounding box and convert to dense if needed
    A1 = A1.tocsc() if sparse.issparse(A1) else A1
    A2 = A2.tocsc() if sparse.issparse(A2) else A2

    A1 = A1[y_lims[0] : y_lims[1], x_lims[0] : x_lims[1]]
    A2 = A2[y_lims[0] : y_lims[1], x_lims[0] : x_lims[1]]

    A1 = A1.toarray() if sparse.issparse(A1) else A1
    A2 = A2.toarray() if sparse.issparse(A2) else A2
    return A1, A2


def pad_axis(
    a: np.ndarray,
    dim_pads: Optional[List[int] | Tuple[int, ...]] = None,
    value: float = np.nan,
) -> np.ndarray:

    if dim_pads is None:
        dim_pads = [1] * len(a.shape)
    assert len(a.shape) == len(dim_pads)

    pads = [(0, pad) for pad in dim_pads]

    # if inplace:
    #     a = np.pad(a, pads, mode="constant", constant_values=value)
    # else:
    return np.pad(a, pads, mode="constant", constant_values=value)
