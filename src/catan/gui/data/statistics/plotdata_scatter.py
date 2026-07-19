from typing import Literal

from dataclasses import dataclass
import numpy as np


from .statistics import STATISTICS
from .dimensions import canonical_dims
from .prepared_data import PickTable


@dataclass
class PlotData:
    title: dict[str, str]
    x_table: PickTable
    y_table: PickTable

    x: np.ndarray  # plotted x values, shape: n_markers
    y: np.ndarray  # plotted y values, shape: n_markers

    # marker index -> PickTable row
    marker_rows: np.ndarray

    # PickTable row -> marker index, or -1 if invalid / not plotted
    row_to_marker: np.ndarray

    @property
    def table(self) -> PickTable:
        """
        Shared semantic table.

        This is valid because the builder checks that x_table and y_table
        have identical dims and refs.
        """
        return self.x_table

    @property
    def n_markers(self) -> int:
        return self.x.size

    def markers_for_thresholds(self, x_spec=None, y_spec=None) -> np.ndarray:
        mask = np.isfinite(self.x) & np.isfinite(self.y)

        if x_spec is not None and x_spec.active:
            if x_spec.direction == "greater":
                mask &= self.x >= x_spec.value
            elif x_spec.direction == "less":
                mask &= self.x <= x_spec.value
            else:
                raise ValueError(x_spec.direction)

        if y_spec is not None and y_spec.active:
            if y_spec.direction == "greater":
                mask &= self.y >= y_spec.value
            elif y_spec.direction == "less":
                mask &= self.y <= y_spec.value
            else:
                raise ValueError(y_spec.direction)

        return np.flatnonzero(mask)

    # def xy_for_marker(self, marker_index: int) -> tuple[float, float]:
    #     return float(self.x[marker_index]), float(self.y[marker_index])

    def pos_for_marker(self, marker_index: int | np.ndarray) -> np.ndarray:
        return np.asarray(
            [self.x[marker_index], self.y[marker_index]],
            dtype=np.float32,
        )

    def rows_for_markers(self, marker_indices: int | np.ndarray) -> np.ndarray:
        marker_indices = np.asarray(marker_indices, dtype=int)
        return self.marker_rows[marker_indices]

    def refs_for_markers(
        self, marker_indices: int | np.ndarray
    ) -> dict[str, np.ndarray]:
        rows = self.rows_for_markers(marker_indices)
        return self.table.refs_for_rows(rows)

    def markers_for_rows(self, rows: int | np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=int)

        if rows.size == 0:
            return np.asarray([], dtype=int)

        marker_ids = self.row_to_marker[rows]
        marker_ids = marker_ids[marker_ids >= 0]

        return np.unique(marker_ids)

    def markers_matching_components(self, components) -> np.ndarray:
        rows = self.table.rows_matching_components(components)
        return self.markers_for_rows(rows)

    def tooltip_for_marker(self, marker_index: int) -> str:
        row = self.marker_rows[marker_index]

        x = self.x[marker_index]
        y = self.y[marker_index]

        base = self.table.tooltip_for_row(row)

        return f"{base}\n" f"{self.title['x']}: {x:.4g}\n" f"{self.title['y']}: {y:.4g}"


def build_plot_data(
    x_table: PickTable,
    y_table: PickTable,
) -> PlotData:

    if canonical_dims(x_table.dims) != canonical_dims(y_table.dims):
        # raise ValueError(
        #     f"Cannot scatter statistics with incompatible dims: "
        #     f"{x_table.dims} vs {y_table.dims}"
        # )
        return PlotData(
            title={
                "error": f"Cannot scatter statistics with incompatible dims: "
                f"{x_table.dims} vs {y_table.dims}"
            },
            x_table=x_table,
            y_table=y_table,
            x=np.array([]),
            y=np.array([]),
            marker_rows=np.array([], dtype=int),
            row_to_marker=np.array([], dtype=int),
        )

    if x_table.n_rows != y_table.n_rows:
        raise ValueError(
            f"Cannot scatter tables with different row counts: "
            f"{x_table.n_rows} vs {y_table.n_rows}"
        )

    for x_dim, y_dim in zip(x_table.dims, y_table.dims):
        x_refs = x_table.refs[x_dim]
        y_refs = y_table.refs[y_dim]

        if not np.array_equal(x_refs, y_refs):
            raise ValueError(
                f"Cannot scatter statistics with different coordinate order: "
                f"{x_dim!r} vs {y_dim!r}"
            )

    x_values = x_table.values
    y_values = y_table.values

    valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
    marker_rows = np.flatnonzero(valid_mask)

    x = x_values[marker_rows]
    y = y_values[marker_rows]

    row_to_marker = np.full(x_table.n_rows, -1, dtype=int)
    row_to_marker[marker_rows] = np.arange(marker_rows.size)

    title = {
        "x": STATISTICS[x_table.stat.name].title,
        "y": STATISTICS[y_table.stat.name].title,
    }

    return PlotData(
        title=title,
        x_table=x_table,
        y_table=y_table,
        x=x,
        y=y,
        marker_rows=marker_rows,
        row_to_marker=row_to_marker,
    )
