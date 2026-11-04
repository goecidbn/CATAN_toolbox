import numpy as np
from scipy import sparse

from catan.core.footprint_lobes import lobe_count, lobe_counts, split_lobes

DIMS = (40, 60)


def _blob(y, x, sigma=2.5, amplitude=1.0):
    yy, xx = np.mgrid[: DIMS[0], : DIMS[1]]
    return amplitude * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma**2))


def _column(image, floor=0.02):
    image = np.where(image >= floor * image.max(), image, 0.0)
    return sparse.csc_matrix(image.reshape(-1, 1))


def test_two_cells_are_two_lobes_one_cell_is_one():
    pair = _column(_blob(20, 18) + _blob(20, 34))
    single = _column(_blob(20, 25, sigma=3.5))
    assert lobe_count(pair, DIMS) == 2
    assert lobe_count(single, DIMS) == 1
    counts = lobe_counts(sparse.hstack([pair, single]).tocsc(), DIMS, [1])
    assert np.isnan(counts[0]) and counts[1] == 1


def test_faint_side_blob_is_no_lobe():
    # Its own peak stays below half of the footprint's peak.
    assert lobe_count(_column(_blob(20, 18) + _blob(20, 34, amplitude=0.3)), DIMS) == 1


def test_split_lobes_cuts_between_the_cells():
    pair = _column(_blob(20, 18) + _blob(20, 34, amplitude=0.8))
    parts = split_lobes(pair, DIMS)
    assert parts is not None
    # The parts add up to the footprint, no pixel in both.
    assert abs(parts[0] + parts[1] - pair).sum() < 1e-9
    assert parts[0].multiply(parts[1]).nnz == 0
    # Largest core first; each part holds one cell's peak.
    x_first = np.argmax(parts[0].toarray().reshape(DIMS).max(axis=0))
    x_second = np.argmax(parts[1].toarray().reshape(DIMS).max(axis=0))
    assert x_first == 18 and x_second == 34
    assert split_lobes(_column(_blob(20, 25, sigma=3.5)), DIMS) is None
