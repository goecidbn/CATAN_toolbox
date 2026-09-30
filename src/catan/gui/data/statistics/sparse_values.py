"""Exact, NaN-filled coordinate storage for calculated pair statistics.

Only exceptions to a scalar fill value are stored. In particular, an empty
sum is zero (NumPy's existing behaviour), including in later reductions.
No Cartesian array is constructed during calculation or reduction.
"""
from dataclasses import dataclass, field
from copy import deepcopy
from math import prod
import warnings

import numpy as np

from .types import StatisticArray


def _center(values, method):
    reducers = {
        "mean": np.nanmean, "median": np.nanmedian,
        "min": np.nanmin, "max": np.nanmax,
        "sum": np.nansum, "std": np.nanstd,
    }
    if method not in reducers:
        raise ValueError(f"Unsupported reduction {method!r}")
    if values.size == 0 and method in ("min", "max"):
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(reducers[method](values))


def _error(values, center, method):
    finite = values[np.isfinite(values)]
    count = len(finite)
    if method in ("std", "sem"):
        err = float(np.std(finite, ddof=1)) if count > 1 else 0.0
        if method == "sem":
            err /= np.sqrt(max(count, 1))
        return err, err, count
    if method == "iqr":
        if count == 0:
            return np.nan, np.nan, 0
        low, high = np.percentile(finite, [25, 75])
        return center - low, high - center, count
    raise NotImplementedError(f"Error method {method!r} is not implemented")


@dataclass
class SparseStatisticArray(StatisticArray):
    # Integer positions into each remaining dimension's coordinate array.
    positions: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.int64))
    fill_value: float = np.nan
    error_fill: tuple = (np.nan, np.nan, 0)
    # Original pair coordinate -> (compact coordinate, semantic offset).
    reference_aliases: dict = field(default_factory=dict)

    def validate(self):
        if self.values.ndim != 1 or self.positions.shape != (len(self.values), self.ndim):
            raise ValueError("Sparse values and coordinate positions disagree")
        for axis, name in enumerate(self.dims):
            coords = self.dimensions[name].coords
            if coords is None:
                raise ValueError(f"Missing coordinates for {name}")
            if len(self.values) and (
                np.any(self.positions[:, axis] < 0)
                or np.any(self.positions[:, axis] >= len(coords))
            ):
                raise ValueError(f"Coordinate outside {name}")
        for values in (self.errors_low, self.errors_high, self.n):
            if values is not None and values.shape != self.values.shape:
                raise ValueError("Sparse auxiliary values must match stored rows")

    def copy(self):
        return deepcopy(self)

    def select_dimension(self, dim_name, index):
        axis = self.axis(dim_name)
        dim = self.dimensions[dim_name]
        fixed = dim.coords[index]
        mask = self.positions[:, axis] == index
        self.positions = np.delete(self.positions[mask], axis, axis=1)
        self.values = self.values[mask]
        for name in ("errors_low", "errors_high", "n"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value[mask])
        dim.mode, dim.parameter = "fixed", fixed
        self.validate()
        return self

    def reduce_dimension(self, dim_name, spec):
        from catan.gui.background_tasks.runtime import current_task_context

        axis = self.axis(dim_name)
        size = len(self.dimensions[dim_name].coords)
        if spec.method in ("keep", "single"):
            raise ValueError("Use selection before numeric reduction")
        if self.has_errors and spec.error_method != "none":
            raise ValueError("Only one error-producing reduction is supported")

        # Exact fibers, including implicit NaNs or zeros, have only the
        # length of ONE axis. Never average medians or unweighted means.
        baseline = np.full(size, self.fill_value)
        new_fill = _center(baseline, spec.method)
        had_errors = self.has_errors
        if spec.error_method != "none":
            new_error_fill = _error(baseline, new_fill, spec.error_method)
        elif had_errors:
            new_error_fill = (
                _center(np.full(size, self.error_fill[0]), spec.method),
                _center(np.full(size, self.error_fill[1]), spec.method),
                float(size * self.error_fill[2]),
            )
        else:
            new_error_fill = (np.nan, np.nan, 0)

        remaining = np.delete(self.positions, axis, axis=1)
        if len(remaining):
            if remaining.shape[1]:
                order = np.lexsort(remaining.T[::-1])
                sorted_positions = remaining[order]
                starts = np.r_[0, 1 + np.flatnonzero(np.any(
                    sorted_positions[1:] != sorted_positions[:-1], axis=1))]
            else:
                order = np.arange(len(remaining))
                starts = np.array([0])
            stops = np.r_[starts[1:], len(order)]
        else:
            order = starts = stops = np.empty(0, dtype=int)

        out = np.empty(len(starts), dtype=float)
        positions = np.empty((len(starts), self.ndim - 1), dtype=np.int64)
        auxiliary = np.empty((3, len(starts))) if had_errors or spec.error_method != "none" else None
        ctx = current_task_context()
        for group, (start, stop) in enumerate(zip(starts, stops)):
            if group % 256 == 0 and ctx is not None:
                ctx.check_cancelled()
            rows = order[start:stop]
            offsets = self.positions[rows, axis]
            if np.isnan(self.fill_value):
                # All reducers ignore unstored NaNs: process only the
                # observed samples, rather than allocating a full fiber.
                fiber = self.values[rows]
            else:
                fiber = baseline.copy()
                fiber[offsets] = self.values[rows]
            out[group] = _center(fiber, spec.method)
            positions[group] = remaining[rows[0]]
            if spec.error_method != "none":
                auxiliary[:, group] = _error(fiber, out[group], spec.error_method)
            elif had_errors:
                for k, source in enumerate((self.errors_low, self.errors_high, self.n)):
                    aux = np.full(size, self.error_fill[k], dtype=float)
                    aux[offsets] = source[rows]
                    auxiliary[k, group] = _center(aux, "sum" if k == 2 else spec.method)

        self.values, self.positions, self.fill_value = out, positions, new_fill
        self.error_fill = new_error_fill
        self.errors_low = None if auxiliary is None else auxiliary[0]
        self.errors_high = None if auxiliary is None else auxiliary[1]
        self.n = None if auxiliary is None else auxiliary[2]
        dim = self.dimensions[dim_name]
        dim.mode = "reduced"
        dim.parameter = {"method": spec.method, "error_method": spec.error_method}
        self.validate()
        return self

    def to_pick_table(self):
        from .table import PickTable

        self.validate()
        positions, values = self.positions, self.values
        low, high, counts = self.errors_low, self.errors_high, self.n
        if np.isfinite(self.fill_value):
            # Empty sums are valid zero results, so they must be shown too.
            shape = tuple(len(self.dimensions[d].coords) for d in self.dims)
            rows = prod(shape)
            if rows > 2_000_000:
                raise ValueError(
                    f"The result contains {rows:,} finite rows (including empty-sum zeros). "
                    "Reduce another axis or select a session before displaying it."
                )
            linear = np.arange(rows, dtype=np.int64)
            positions = (np.column_stack(np.unravel_index(linear, shape))
                         if shape else np.empty((1, 0), dtype=np.int64))
            stored = (np.ravel_multi_index(self.positions.T, shape)
                      if shape else np.zeros(len(self.values), dtype=int))
            values = np.full(rows, self.fill_value)
            values[stored] = self.values
            if self.has_errors:
                arrays = []
                for source, fill in zip((low, high, counts), self.error_fill):
                    target = np.full(rows, fill, dtype=float)
                    target[stored] = source
                    arrays.append(target)
                low, high, counts = arrays
        # Unstored NaNs need no pickable rows; all coordinate domains remain
        # in stat.dimensions. Finite zeros, however, are retained.
        refs = {name: np.asarray(self.dimensions[name].coords)[positions[:, axis]]
                for axis, name in enumerate(self.dims)}
        return PickTable(stat=self, values=values, dims=self.dims, refs=refs,
                         errors_low=low, errors_high=high, n=counts)
