from typing import Literal

from dataclasses import dataclass
import numpy as np

from .statistics import STATISTICS
from .prepared_data import PickTable
from catan.gui.plots.helper.Threshold import ThresholdSpec


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

    def rows_for_threshold(self, spec: ThresholdSpec) -> np.ndarray:
        if not spec.active:
            return np.asarray([], dtype=int)

        values = self.table.values
        finite = np.isfinite(values)

        if spec.direction == "greater":
            mask = finite & (values >= spec.value)
        elif spec.direction == "less":
            mask = finite & (values <= spec.value)
        else:
            raise ValueError(spec.direction)

        return np.flatnonzero(mask)

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

        return (
            f"{self.title}\n"
            f"bin: {bin_index}\n"
            f"values \u2208 [{left:.4g}, {right:.4g})\n"
            f"count: {len(rows)}"
        )


def build_plot_data(
    table: PickTable,
    *,
    bins: int | np.ndarray = 40,
) -> PlotData:
    values = table.values

    valid_mask = np.isfinite(values)
    valid_rows = np.flatnonzero(valid_mask)
    valid_values = values[valid_mask]

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
        title={"x": STATISTICS[table.stat.name].title},
        table=table,
        bin_edges=edges,
        bin_counts=counts,
        bin_rows=bin_rows,
        row_to_bin=row_to_bin,
    )
