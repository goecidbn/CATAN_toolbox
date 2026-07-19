import numpy as np
import scipy as sp


def center_of_mass(
    A, d1, d2, d3=None, d1_offset=0, d2_offset=0, d3_offset=0, convert=1.0
) -> np.ndarray:
    """
    calculate center of mass of footprints A in 2D or 3D

    TODO:
      - implement offset in each dimension
    """

    if "csc_matrix" not in str(type(A)):
        A = sp.sparse.csc_matrix(A)

    if d3 is None:
        Coor = np.matrix(
            [
                np.outer(np.ones(d2), np.arange(d1)).ravel(),
                np.outer(np.arange(d2), np.ones(d1)).ravel(),
            ],
            dtype=A.dtype,
        )
    else:
        Coor = np.matrix(
            [
                np.outer(
                    np.ones(d3), np.outer(np.ones(d2), np.arange(d1)).ravel()
                ).ravel(),
                np.outer(
                    np.ones(d3), np.outer(np.arange(d2), np.ones(d1)).ravel()
                ).ravel(),
                np.outer(
                    np.arange(d3), np.outer(np.ones(d2), np.ones(d1)).ravel()
                ).ravel(),
            ],
            dtype=A.dtype,
        )

    Anorm = normalize_sparse_array(
        A, relative_threshold=0.001, minimum_nonzero_entries=0
    )

    cm = (Coor * Anorm).T
    cm[np.squeeze(np.array((Anorm > 0).sum(0))) == 0, :] = np.nan

    return np.array(cm) * convert


def normalize_sparse_array(A, relative_threshold=0.001, minimum_nonzero_entries=50):
    """
    normalizing sparse arrays with some thresholding
      - relative_threshold: float
          fraction of peak value, at below which entries are considered to be 0
      - minimum_nonzero_entries: int
          minimum number of nonzero entries of the footprint to be considered for further analyses
    """
    return sp.sparse.vstack(
        [
            # a.multiply(a>relative_threshold*a.max())/a.max()  # threshold footprint
            (
                a.multiply(a > relative_threshold * a.max())
                / a[a > 0.001 * a.max()].sum()  # threshold footprint
                if (a > 0).sum()
                > minimum_nonzero_entries  # require minimum non-zero entries ...
                else sp.sparse.csr_matrix(a.shape)
            )  # ... otherwise return empty slice
            for a in A.T  # loop through all footprints
        ]
    ).T
