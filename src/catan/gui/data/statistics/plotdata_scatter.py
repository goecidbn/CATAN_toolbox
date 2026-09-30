from dataclasses import dataclass, field
import numpy as np
from scipy.spatial import cKDTree

from .dimensions import canonical_dims
from .table import PickTable, refs_equal, ROW_BLOCK_SIZE
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

    _pick_tree: cKDTree | None = field(
        init=False, default=None, repr=False
    )
    _pick_origin: np.ndarray = field(init=False, repr=False)
    _pick_scale: np.ndarray = field(init=False, repr=False)

    _neuron_index: dict = field(
        init=False, default_factory=dict, repr=False
    )

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
        # Release construction temporaries before building the indexes.
        del x_values, y_values, valid

        self._build_interaction_indices()


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

    def markers_matching_components(self, components):
        components = (
            [] if components is None else list(components)
        )

        if not components or not self.n_markers:
            return np.empty(0, dtype=np.intp)

        neuron_ids = {
            int(component.neuron_id)
            for component in components
        }

        matches = []

        for side, table, source_rows in (
            ("x", self.x_table, self.x_rows),
            ("y", self.y_table, self.y_rows),
        ):
            parts = []

            for order, ranges in self._neuron_index.get(
                side, ()
            ):
                for neuron_id in neuron_ids:
                    bounds = ranges.get(neuron_id)

                    if bounds is not None:
                        start, stop = bounds
                        parts.append(order[start:stop])

            if not parts:
                continue

            candidates = np.unique(
                np.concatenate(parts)
            )

            rows = source_rows[
                self.marker_rows[candidates]
            ]
            candidate_table = table.subset_rows(rows)

            # Keep the existing session, wildcard, and pair rules,
            # but evaluate them only on candidate markers.
            selected = (
                candidate_table.rows_matching_components(
                    components
                )
            )

            if selected.size:
                matches.append(candidates[selected])

        if not matches:
            return np.empty(0, dtype=np.intp)

        return np.unique(np.concatenate(matches))
    
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

    def _build_interaction_indices(self):
        self._neuron_index = {}
        self._pick_tree = None

        if not self.n_markers:
            return

        # Spatial index for mouse picking.
        self._pick_origin = np.array([
            self.x.min(),
            self.y.min(),
        ])

        self._pick_scale = np.array([
            np.ptp(self.x),
            np.ptp(self.y),
        ])
        self._pick_scale[self._pick_scale == 0] = 1.0

        points = np.empty(
            (self.n_markers, 2),
            dtype=np.float64,
        )
        points[:, 0] = (
            self.x - self._pick_origin[0]
        ) / self._pick_scale[0]
        points[:, 1] = (
            self.y - self._pick_origin[1]
        ) / self._pick_scale[1]

        self._pick_tree = cKDTree(points, copy_data=False)

        # Neuron -> candidate markers for linked highlighting.
        index_dtype = (
            np.int32
            if self.n_markers <= np.iinfo(np.int32).max
            else np.int64
        )

        for side, table, source_rows in (
            ("x", self.x_table, self.x_rows),
            ("y", self.y_table, self.y_rows),
        ):
            groups = []

            for column in table._available_neuron_arrays():
                neuron_ids = np.empty(
                    self.n_markers,
                    dtype=np.int64,
                )

                for start in range(
                    0, self.n_markers, ROW_BLOCK_SIZE
                ):
                    markers = slice(
                        start, start + ROW_BLOCK_SIZE
                    )
                    rows = source_rows[
                        self.marker_rows[markers]
                    ]
                    neuron_ids[markers] = column[rows]

                order = np.argsort(
                    neuron_ids,
                    kind="stable",
                )
                ordered_ids = neuron_ids[order]

                starts = np.r_[
                    0,
                    np.flatnonzero(
                        ordered_ids[1:] != ordered_ids[:-1]
                    ) + 1,
                ]
                stops = np.r_[
                    starts[1:],
                    self.n_markers,
                ]

                ranges = {
                    int(ordered_ids[start]): (
                        int(start), int(stop)
                    )
                    for start, stop in zip(starts, stops)
                }

                groups.append((
                    order.astype(index_dtype, copy=False),
                    ranges,
                ))

                del neuron_ids, ordered_ids, order

            self._neuron_index[side] = groups

    def nearest_marker(
        self,
        data_pos,
        width,
        height,
        radius,
    ):
        if self._pick_tree is None:
            return None

        width = abs(float(width))
        height = abs(float(height))
        pos = np.asarray(data_pos, dtype=float)[:2]

        if (
            not np.all(np.isfinite(pos))
            or not np.isfinite(width + height)
            or width == 0
            or height == 0
        ):
            return None

        query = (
            pos - self._pick_origin
        ) / self._pick_scale

        # A nearby point provides an upper bound on the search.
        _, seed = self._pick_tree.query(query)

        seed_distance = np.hypot(
            (self.x[seed] - pos[0]) / width,
            (self.y[seed] - pos[1]) / height,
        )

        limit = min(
            float(radius),
            float(seed_distance),
        )

        # This circle encloses every point that could be closer
        # under the current camera-normalized distance metric.
        search_radius = limit * max(
            width / self._pick_scale[0],
            height / self._pick_scale[1],
        )

        search_radius += (
            16 * np.finfo(float).eps
            * (1 + np.linalg.norm(query))
        )

        candidates = np.asarray(
            self._pick_tree.query_ball_point(
                query,
                search_radius,
            ),
            dtype=np.intp,
        )

        if not candidates.size:
            return None

        # Apply the original picking metric only to candidates.
        dx = (self.x[candidates] - pos[0]) / width
        dy = (self.y[candidates] - pos[1]) / height
        distances = dx * dx + dy * dy

        best = distances.min()

        if np.sqrt(best) > radius:
            return None

        # Preserve the original first-marker behaviour for ties.
        return int(
            candidates[distances == best].min()
        )


def build_plot_data(
    x_table: PickTable,
    y_table: PickTable,
) -> PlotData:

    return PlotData(
        x_table=x_table,
        y_table=y_table,
    )

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

    if x_table.n_rows == y_table.n_rows and all(
        refs_equal(x_table.refs[x_dim], y_table.refs[y_dim])
        for x_dim, y_dim in zip(x_table.dims, y_table.dims)
    ):
        if not x_table.n_rows:
            raise StatisticsPlotError(
                "The selected statistics have no matching coordinates."
            )

        rows = np.arange(x_table.n_rows, dtype=np.intp)
        return rows, rows

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
        for table, keys, dim in (
            (x_table, x_keys, x_dim),
            (y_table, y_keys, y_dim),
        ):
            for start in range(0, table.n_rows, ROW_BLOCK_SIZE):
                rows = slice(start, start + ROW_BLOCK_SIZE)
                keys[field][rows] = table.refs[dim][rows]

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
