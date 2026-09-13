from dataclasses import dataclass, field
import numpy as np

from .dimensions import canonical_dims
from .table import PickTable
from .errors import StatisticsPlotError


@dataclass(slots=True)
class PlotData:
    x_table: PickTable
    y_table: PickTable

    title: dict[str, str] = field(init=False)

    # matched row -> source-table row
    x_rows: np.ndarray = field(init=False, repr=False)
    y_rows: np.ndarray = field(init=False, repr=False)

    # plotted values, indexed by marker
    x: np.ndarray = field(init=False, repr=False)
    y: np.ndarray = field(init=False, repr=False)

    # marker -> matched row
    marker_rows: np.ndarray = field(init=False, repr=False)

    # matched row -> marker, -1 if not plotted
    row_to_marker: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):

        self.title = {
            "x": self.x_table.stat.display_title,
            "y": self.y_table.stat.display_title,
        }

        self.x_rows, self.y_rows = _matching_rows(
            self.x_table,
            self.y_table,
        )

        x_values = self.x_table.values[self.x_rows]
        y_values = self.y_table.values[self.y_rows]

        valid = np.isfinite(x_values) & np.isfinite(y_values)

        self.marker_rows = np.flatnonzero(valid)

        self.x = x_values[valid]
        self.y = y_values[valid]

        self.row_to_marker = np.full(self.n_rows, -1, dtype=int)

        self.row_to_marker[self.marker_rows] = np.arange(self.marker_rows.size)

    @property
    def n_rows(self) -> int:
        return self.x_rows.size

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

    def markers_for_rows(
        self,
        rows: int | np.ndarray,
    ) -> np.ndarray:

        rows = np.asarray(rows, dtype=int)

        if rows.size == 0 or self.row_to_marker.size == 0:
            return np.asarray([], dtype=int)

        valid_rows = (rows >= 0) & (rows < self.row_to_marker.size)

        rows = rows[valid_rows]

        if rows.size == 0:
            return np.asarray([], dtype=int)

        marker_ids = self.row_to_marker[rows]
        marker_ids = marker_ids[marker_ids >= 0]

        return np.unique(marker_ids)

    # def markers_matching_components(self, components) -> np.ndarray:
    #     rows = self.table.rows_matching_components(components)
    #     return self.markers_for_rows(rows)

    def ref_sets_for_rows(
        self,
        rows: int | np.ndarray,
    ) -> tuple[dict[str, np.ndarray], ...]:
        """
        Return the semantic provenance of matched scatter rows.

        A scatter point may represent values from two different semantic
        contexts, e.g. neuron 42 in session 0 on x and the same neuron in
        session 1 on y. Keep those reference sets separate.
        """
        rows = np.atleast_1d(rows).astype(int, copy=False)

        return (
            self.x_table.refs_for_rows(self.x_rows[rows]),
            self.y_table.refs_for_rows(self.y_rows[rows]),
        )

    def ref_sets_for_markers(
        self,
        marker_indices: int | np.ndarray,
    ) -> tuple[dict[str, np.ndarray], ...]:

        rows = self.rows_for_markers(marker_indices)

        return self.ref_sets_for_rows(rows)

    def markers_matching_components(
        self,
        components,
    ) -> np.ndarray:

        matched = np.zeros(
            self.n_rows,
            dtype=bool,
        )

        # x-side semantics
        x_selected_rows = self.x_table.rows_matching_components(components)

        if x_selected_rows.size:
            selected = np.zeros(self.x_table.n_rows, dtype=bool)
            selected[x_selected_rows] = True
            matched |= selected[self.x_rows]

        # y-side semantics
        y_selected_rows = self.y_table.rows_matching_components(components)

        if y_selected_rows.size:
            selected = np.zeros(self.y_table.n_rows, dtype=bool)
            selected[y_selected_rows] = True
            matched |= selected[self.y_rows]

        return self.markers_for_rows(np.flatnonzero(matched))

    def tooltip_for_marker(
        self,
        marker_index: int,
    ) -> str:

        row = self.marker_rows[marker_index]

        x_refs, y_refs = self.ref_sets_for_rows([row])

        same_refs = x_refs.keys() == y_refs.keys() and all(
            np.array_equal(x_refs[key], y_refs[key]) for key in x_refs
        )

        if same_refs:
            lines = [f"{dim}: {values[0]}" for dim, values in x_refs.items()]
        else:
            x_ref_text = ", ".join(
                f"{dim}: {values[0]}" for dim, values in x_refs.items()
            )

            y_ref_text = ", ".join(
                f"{dim}: {values[0]}" for dim, values in y_refs.items()
            )

            lines = [
                f"x: {x_ref_text}",
                f"y: {y_ref_text}",
            ]

        lines.extend(
            [
                (f"{self.title['x']}: " f"{self.x[marker_index]:.4g}"),
                (f"{self.title['y']}: " f"{self.y[marker_index]:.4g}"),
            ]
        )

        return "\n".join(lines)


def build_plot_data(
    x_table: PickTable,
    y_table: PickTable,
) -> PlotData:

    return PlotData(
        x_table=x_table,
        y_table=y_table,
    )

    # x_rows, y_rows = _matching_rows(x_table, y_table)

    # # Semantic table corresponding exactly to the joined rows.
    # matched_table = x_table.subset_rows(x_rows)

    # x_values = x_table.values[x_rows]
    # y_values = y_table.values[y_rows]

    # # Not every matched coordinate necessarily has a finite
    # # value in both statistics.
    # valid = np.isfinite(x_values) & np.isfinite(y_values)

    # marker_rows = np.flatnonzero(valid)

    # x = x_values[valid]
    # y = y_values[valid]

    # row_to_marker = np.full(
    #     matched_table.n_rows,
    #     -1,
    #     dtype=int,
    # )

    # row_to_marker[marker_rows] = np.arange(marker_rows.size)

    # return PlotData(
    #     title={
    #         "x": x_table.stat.display_title,
    #         "y": y_table.stat.display_title,
    #     },
    #     x_table=x_table,
    #     y_table=y_table,
    #     matched_table=matched_table,
    #     x=x,
    #     y=y,
    #     marker_rows=marker_rows,
    #     row_to_marker=row_to_marker,
    # )


def _matching_rows(
    x_table: PickTable,
    y_table: PickTable,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find rows representing the same semantic coordinates in
    x_table and y_table.

    Returns
    -------
    x_rows, y_rows
        Corresponding row indices into the original tables.
    """

    # Dimensions must describe corresponding semantic axes.
    if canonical_dims(x_table.dims) != canonical_dims(y_table.dims):
        raise StatisticsPlotError(
            "Cannot scatter statistics with incompatible dimensions:\n"
            f"{x_table.dims} vs {y_table.dims}"
        )

    if len(x_table.dims) != len(y_table.dims):
        # Mostly redundant with canonical_dims(), but explicit.
        raise StatisticsPlotError(
            "Cannot scatter statistics with different numbers "
            "of remaining dimensions."
        )

    # Scalar statistics.
    if not x_table.dims:
        if x_table.n_rows != 1 or y_table.n_rows != 1:
            raise StatisticsPlotError(
                "Cannot align scalar statistics with multiple rows."
            )

        return (
            np.asarray([0], dtype=int),
            np.asarray([0], dtype=int),
        )

    # Build structured keys. Dimension correspondence is positional:
    #
    # x_table.dims[0] <-> y_table.dims[0]
    # x_table.dims[1] <-> y_table.dims[1]
    # ...
    #
    # This is important for pair dimensions: neuron_i/neuron_j must
    # remain separate even though both canonicalize to "neuron".

    dtypes = []

    for x_dim, y_dim in zip(x_table.dims, y_table.dims):
        dtypes.append(
            np.result_type(
                x_table.refs[x_dim].dtype,
                y_table.refs[y_dim].dtype,
            )
        )

    key_dtype = np.dtype([(f"d{i}", dtype) for i, dtype in enumerate(dtypes)])

    x_keys = np.empty(x_table.n_rows, dtype=key_dtype)
    y_keys = np.empty(y_table.n_rows, dtype=key_dtype)

    for i, (x_dim, y_dim, dtype) in enumerate(zip(x_table.dims, y_table.dims, dtypes)):
        field = f"d{i}"
        x_keys[field] = np.asarray(x_table.refs[x_dim], dtype=dtype)
        y_keys[field] = np.asarray(y_table.refs[y_dim], dtype=dtype)

    _, x_rows, y_rows = np.intersect1d(
        x_keys,
        y_keys,
        assume_unique=True,
        return_indices=True,
    )

    if x_rows.size == 0:
        raise StatisticsPlotError(
            "The selected statistics have compatible dimensions, "
            "but no matching coordinates."
        )

    # np.intersect1d sorts by coordinate. Restore x-table order
    # so the resulting table behaves predictably.
    order = np.argsort(x_rows)

    return (x_rows[order], y_rows[order])
