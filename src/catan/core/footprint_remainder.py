"""The rest of a footprint after removing known pieces of it.

Used by a split with a single source: the source footprint (from another
session) is one cell inside the target footprint; what is left of the
target becomes a new footprint.
"""

import numpy as np
from scipy import sparse

# Remainder pixels below this fraction of its peak are dropped.
REMAINDER_FLOOR = 0.05

# Remainders with less than this fraction of the whole's mass are empty.
MIN_REMAINDER_MASS = 0.05


def remainder_footprint(whole, pieces):
    """``whole`` minus the pieces scaled to it (one non-negative least-
    squares factor for their sum), clipped at 0 and floored.

    whole, pieces: sparse (n_pixels x 1) columns. Returns a csc column, or
    None when (almost) nothing remains.
    """
    whole = np.asarray(sparse.csc_matrix(whole).todense(), dtype=float).ravel()
    piece = np.zeros_like(whole)
    for p in pieces:
        piece += np.asarray(sparse.csc_matrix(p).todense(), dtype=float).ravel()

    norm = float(piece @ piece)
    scale = max(float(whole @ piece) / norm, 0.0) if norm > 0 else 0.0

    rest = np.clip(whole - scale * piece, 0.0, None)

    if rest.max(initial=0.0) <= 0:
        return None

    rest[rest < REMAINDER_FLOOR * rest.max()] = 0.0

    if rest.sum() < MIN_REMAINDER_MASS * max(whole.sum(), 1e-12):
        return None

    return sparse.csc_matrix(rest.reshape(-1, 1).astype(np.float32))
