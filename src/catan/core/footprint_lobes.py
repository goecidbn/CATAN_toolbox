"""Lobes of a footprint: bright parts that separate when it is cut at a
threshold (e.g. two cells detected as one footprint, joined by a thin or
dim bridge).

A lobe is a connected part (8-neighbourhood) of the pixels above some
threshold t (from ``base`` up to the peak, in steps of ``step``; all
relative to the footprint's peak) whose own peak reaches ``peak`` and whose
area is at least ``size`` of the footprint's area at ``base`` (and at least
``min_pixels``). The lobe count is the largest number of such parts at any
threshold; faint side blobs (peak below ``peak``) and small specks do not
count.
"""

import numpy as np
from scipy import ndimage, sparse

EIGHT = np.ones((3, 3), dtype=bool)

DEFAULTS = dict(peak=0.5, size=0.1, base=0.1, step=0.05, min_pixels=10)


def footprint_image(column, dims):
    """Cropped dense image of a sparse footprint column (C order) and its
    offset (y0, x0); (None, None) without positive pixels."""
    column = sparse.csc_matrix(column)
    if column.nnz == 0:
        return None, None
    rows = column.indices
    values = column.data
    positive = values > 0
    if not positive.any():
        return None, None
    rows, values = rows[positive], values[positive]
    y, x = np.divmod(rows, int(dims[1]))
    y0, x0 = int(y.min()), int(x.min())
    image = np.zeros((int(y.max()) - y0 + 1, int(x.max()) - x0 + 1))
    image[y - y0, x - x0] = values
    return image, (y0, x0)


def _lobes_at_best_threshold(image, peak, size, base, step, min_pixels):
    """(count, labels of the qualifying lobes at the threshold with the
    most of them, their label ids)."""
    image = image / image.max()
    need = max(float(min_pixels), size * float(np.count_nonzero(image >= base)))
    best = (1, None, ())

    for threshold in np.arange(base, 1.0, step):
        labels, n = ndimage.label(image >= threshold, EIGHT)
        if n < 2:
            continue
        index = np.arange(1, n + 1)
        areas = np.bincount(labels.ravel(), minlength=n + 1)[1:]
        peaks = ndimage.maximum(image, labels, index)
        lobes = index[(areas >= need) & (np.asarray(peaks) >= peak)]
        if len(lobes) > best[0]:
            best = (len(lobes), labels, tuple(int(i) for i in lobes))

    return best


def lobe_count(column, dims, **parameters):
    """Number of lobes of a footprint (1 for a single compact one, 0
    without pixels)."""
    options = {**DEFAULTS, **parameters}
    image, _ = footprint_image(column, dims)
    if image is None:
        return 0
    return _lobes_at_best_threshold(image, **options)[0]


def lobe_counts(footprints, dims, columns=None, **parameters):
    """Lobe counts of the footprint matrix's ``columns`` (all by default);
    an array over all columns, NaN for the ones not requested."""
    footprints = sparse.csc_matrix(footprints)
    counts = np.full(footprints.shape[1], np.nan)
    columns = range(footprints.shape[1]) if columns is None else columns
    for column in columns:
        counts[int(column)] = lobe_count(footprints[:, int(column)], dims, **parameters)
    return counts


def split_lobes(column, dims, **parameters):
    """Cut a footprint into its two largest lobes: each pixel goes to the
    nearer lobe core (the parts above the threshold where the lobes
    separate best). Returns two sparse columns that add up to the
    footprint, or None when it has fewer than two lobes."""
    options = {**DEFAULTS, **parameters}
    column = sparse.csc_matrix(column)
    image, offset = footprint_image(column, dims)
    if image is None:
        return None

    count, labels, lobes = _lobes_at_best_threshold(image, **options)
    if count < 2:
        return None

    # The two largest lobes are the cores; other lobes join the nearer one.
    areas = [(np.count_nonzero(labels == lobe), lobe) for lobe in lobes]
    first, second = (lobe for _, lobe in sorted(areas, reverse=True)[:2])

    # Every pixel goes to the nearer lobe core (the lobes where they
    # separate): the cut runs between the cores, through the bridge.
    cores = np.zeros(image.shape, dtype=np.int16)
    cores[labels == first] = 1
    cores[labels == second] = 2
    _, (iy, ix) = ndimage.distance_transform_edt(cores == 0, return_indices=True)
    basins = cores[iy, ix]

    y0, x0 = offset
    rows = column.indices
    values = column.data
    y, x = np.divmod(rows, int(dims[1]))
    basin = basins[y - y0, x - x0]

    parts = []
    for part in (1, 2):
        keep = basin == part
        parts.append(
            sparse.csc_matrix(
                (values[keep], (rows[keep], np.zeros(keep.sum(), dtype=int))),
                shape=column.shape,
                dtype=column.dtype,
            )
        )
    return tuple(parts)
