from dataclasses import dataclass
import math

import numpy as np

from .table import PickTable
from catan.gui.panels.helper.Threshold import ThresholdSpec


@dataclass
class PlotData:
    title: dict[str, str]
    table: PickTable

    bin_edges: np.ndarray
    bin_counts: np.ndarray

    # bin index -> PickTable rows
    bin_rows: list[np.ndarray]

    # PickTable row -> bin index, or -1 if invalid / not plotted
    row_to_bin: np.ndarray

    # Integer-valued data: integers per bin (bins [x - 0.5, x + k - 0.5));
    # None: ordinary bins.
    integer_step: int | None = None

    def rows_for_threshold(self, spec: ThresholdSpec) -> np.ndarray:
        if not spec.active:
            return np.asarray([], dtype=int)

        if spec.direction not in {"greater", "less"}:
            raise ValueError(spec.direction)

        values = self.table.values
        # spec.mask honours strict thresholds (> / <).
        return np.flatnonzero(np.isfinite(values) & spec.mask(values))

    def rows_for_bin(self, bin_index: int) -> np.ndarray:
        return self.bin_rows[bin_index]

    def bins_for_rows(self, rows) -> np.ndarray:
        rows = np.asarray(rows, dtype=int)

        if rows.size == 0:
            return np.asarray([], dtype=int)

        bin_ids = self.row_to_bin[rows]
        bin_ids = bin_ids[bin_ids >= 0]

        return np.unique(bin_ids)

    def refs_for_bin(self, bin_index: int) -> dict[str, np.ndarray]:
        rows = self.rows_for_bin(bin_index)
        return self.table.refs_for_rows(rows)

    def ref_sets_for_rows(
        self,
        rows: int | np.ndarray,
    ) -> tuple[dict[str, np.ndarray], ...]:

        rows = np.atleast_1d(rows).astype(int, copy=False)

        return (self.table.refs_for_rows(rows),)

    def rows_matching_components(self, components) -> np.ndarray:
        return self.table.rows_matching_components(components)

    def bins_matching_components(self, components) -> np.ndarray:
        rows = self.rows_matching_components(components)
        return self.bins_for_rows(rows)

    def values_matching_components(self, components) -> np.ndarray:
        rows = self.rows_matching_components(components)
        return self.table.values_for_rows(rows)

    def selected_histogram_for_components(
        self, components
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            selected_counts:
                count per bin for selected components

            selected_rows:
                PickTable rows belonging to selected components

            selected_bin_ids:
                bin id for every selected row, excluding invalid rows
        """
        selected_rows = self.rows_matching_components(components)

        if selected_rows.size == 0:
            return (
                np.zeros_like(self.bin_counts, dtype=int),
                selected_rows,
                np.asarray([], dtype=int),
            )

        selected_bin_ids = self.row_to_bin[selected_rows]
        valid = selected_bin_ids >= 0

        selected_rows = selected_rows[valid]
        selected_bin_ids = selected_bin_ids[valid]

        selected_counts = np.bincount(
            selected_bin_ids,
            minlength=len(self.bin_counts),
        )

        return selected_counts, selected_rows, selected_bin_ids

    def tooltip_for_bin(self, bin_index: int) -> str:
        rows = self.rows_for_bin(bin_index)

        left = self.bin_edges[bin_index]
        right = self.bin_edges[bin_index + 1]

        if self.integer_step is not None:
            values = integer_bin_label(left, right)
        else:
            values = f"values \u2208 [{left:.4g}, {right:.4g})"

        return (
            f"{self.title}\n"
            f"bin: {bin_index}\n"
            f"{values}\n"
            f"count: {len(rows)}"
        )


# Integers per bin for integer-valued data: the smallest "nice" step
# (1, 2, 5, 10, 20, 50, \u2026) that keeps the bin count within the limit.
_NICE_BASES = (1, 2, 5)


def integer_valued(values) -> bool:
    """All finite values are integers (counts, sums of counts, px areas)."""
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    return finite.size > 0 and bool(np.all(finite == np.round(finite)))


def integer_bin_step(lower: int, upper: int, max_bins: int) -> int:
    span = int(upper) - int(lower) + 1
    max_bins = max(1, int(max_bins))
    magnitude = 1
    while True:
        for base in _NICE_BASES:
            step = base * magnitude
            if math.ceil(span / step) <= max_bins:
                return step
        magnitude *= 10


def integer_bin_edges(lower, upper, max_bins) -> tuple[np.ndarray, int]:
    """Bins centred on integers: [x - 0.5, x + k - 0.5) with x a multiple
    of the step k, covering lower \u2026 upper."""
    lower, upper = math.floor(lower), math.ceil(upper)
    upper = max(upper, lower)
    step = integer_bin_step(lower, upper, max_bins)
    start = math.floor(lower / step) * step
    count = math.ceil((upper - start + 1) / step)
    edges = start - 0.5 + step * np.arange(count + 1, dtype=float)
    return edges, step


def integer_bin_label(left, right) -> str:
    first, last = int(round(left + 0.5)), int(round(right - 0.5))
    return f"value {first}" if first == last else f"values {first} \u2013 {last}"


def possible_value_range(definition, stat, n_sessions):
    """(lower, upper) the statistic can take, from the definition's
    ``value_bounds`` (one entry, before reductions; a side may be None) and
    the reductions: a sum over a dimension multiplies by its size (session
    axes: the active sessions); mean / median / min / max keep the range.
    None if unknown."""
    hint = getattr(definition, "value_bounds", None)
    if hint is None or stat is None:
        return None
    bounds = hint(n_sessions) if callable(hint) else hint
    if bounds is None:
        return None
    lower, upper = (None if value is None else float(value) for value in bounds)

    for name, dim in stat.dimensions.items():
        if dim.mode != "reduced":
            continue
        parameter = dim.parameter
        method = parameter.get("method") if isinstance(parameter, dict) else parameter
        if method in ("mean", "median", "min", "max"):
            continue
        if method != "sum":
            return None
        size = n_sessions if name.startswith("session") else len(dim.coords)
        lower = None if lower is None else lower * size
        upper = None if upper is None else upper * size

    return lower, upper


def histogram_edges(values, bins, *, bounds=None) -> tuple[np.ndarray, int | None]:
    """Bin edges and the integer step (None: ordinary bins). Integer-valued
    data get integer-centred bins over the possible range (``bounds``; a
    None side: the data range), at most ``bins`` of them; other data
    ``bins`` equal bins over the data range."""
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]

    if integer_valued(finite):
        lower, upper = float(finite.min()), float(finite.max())
        if bounds is not None:
            low, high = bounds
            if low is not None and np.isfinite(low):
                lower = min(lower, low)
            if high is not None and np.isfinite(high):
                upper = max(upper, high)
        return integer_bin_edges(lower, upper, bins)

    if finite.size == 0:
        return np.linspace(0.0, 1.0, int(bins) + 1), None
    return np.histogram_bin_edges(finite, bins=int(bins)), None


def build_plot_data(
    table: PickTable,
    *,
    bins: int | np.ndarray = 40,
    bounds=None,
    integer_step: int | None = None,
) -> PlotData:
    """``bins``: the bin count (the maximum for integer-valued data, see
    ``histogram_edges``) or explicit edges (with ``integer_step`` when they
    are integer-centred)."""
    values = table.values

    valid_mask = np.isfinite(values)
    valid_rows = np.flatnonzero(valid_mask)
    valid_values = values[valid_mask]

    if np.ndim(bins) == 0:
        bins, integer_step = histogram_edges(valid_values, bins, bounds=bounds)

    counts, edges = np.histogram(valid_values, bins=bins)

    row_to_bin = np.full(table.n_rows, -1, dtype=int)

    if valid_values.size > 0:
        bin_ids = np.searchsorted(edges, valid_values, side="right") - 1
        bin_ids = np.clip(bin_ids, 0, len(counts) - 1)

        row_to_bin[valid_rows] = bin_ids

        bin_rows = [valid_rows[bin_ids == b] for b in range(len(counts))]
    else:
        bin_rows = [np.asarray([], dtype=int) for _ in range(len(counts))]

    return PlotData(
        title={"x": table.stat.display_title},
        table=table,
        bin_edges=edges,
        bin_counts=counts,
        bin_rows=bin_rows,
        row_to_bin=row_to_bin,
        integer_step=integer_step,
    )
