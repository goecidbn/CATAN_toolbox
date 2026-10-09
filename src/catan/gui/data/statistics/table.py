from dataclasses import dataclass, replace
from math import prod

import numpy as np

from .dimensions import canonical_dim
from .queries import PairFilter
from .types import StatisticArray

ROW_BLOCK_SIZE = 65_536


@dataclass(eq=False)
class GridRef(np.lib.mixins.NDArrayOperatorsMixin):
    """A flattened coordinate column without allocating the full column."""

    coords: np.ndarray
    grid_shape: tuple[int, ...]
    axis: int
    row_ids: np.ndarray | None = None

    @property
    def size(self):
        if self.row_ids is None:
            return prod(self.grid_shape)
        return self.row_ids.size

    @property
    def shape(self):
        return (self.size,)

    @property
    def dtype(self):
        return self.coords.dtype

    @property
    def ndim(self):
        return 1

    def __len__(self):
        return self.size

    def __getitem__(self, rows):
        if self.row_ids is not None:
            rows = self.row_ids[rows]

        shape = [1] * len(self.grid_shape)
        shape[self.axis] = len(self.coords)

        grid = np.broadcast_to(
            self.coords.reshape(shape),
            self.grid_shape,
        )

        # Index the broadcast view directly. Do not reshape/ravel it,
        # because that could allocate the entire coordinate column.
        return grid.flat[rows]

    def __array__(self, dtype=None, copy=None):
        # Explicit conversion remains available for existing callers.
        if copy is False:
            raise ValueError(
                "GridRef cannot expose a contiguous array without copying."
            )
        return np.asarray(self[:], dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if kwargs.get("out") is not None:
            return NotImplemented

        inputs = tuple(
            np.asarray(value) if isinstance(value, GridRef) else value
            for value in inputs
        )

        return getattr(ufunc, method)(*inputs, **kwargs)


def refs_equal(left, right):
    """Compare coordinate columns without expanding both full columns."""
    if left is right:
        return True

    if left.shape != right.shape:
        return False

    if isinstance(left, GridRef) and isinstance(right, GridRef):
        same_rows = left.row_ids is right.row_ids

        if left.row_ids is not None and right.row_ids is not None:
            same_rows = same_rows or refs_equal(
                left.row_ids,
                right.row_ids,
            )

        if (
            same_rows
            and left.axis == right.axis
            and left.grid_shape == right.grid_shape
            and np.array_equal(left.coords, right.coords)
        ):
            return True

    return all(
        np.array_equal(
            left[start : start + ROW_BLOCK_SIZE],
            right[start : start + ROW_BLOCK_SIZE],
        )
        for start in range(0, left.size, ROW_BLOCK_SIZE)
    )


@dataclass
class PickTable:
    """
    Flattened, pickable representation of a StatisticArray.

    rows:
        integer indices into `values`

    refs:
        maps dimension name -> semantic coordinate per row.
        Example:
            refs["neuron"][row]  -> neuron_id
            refs["session"][row] -> session_id
    """

    stat: StatisticArray
    values: np.ndarray  # shape: (n_rows,)
    dims: tuple[str, ...]  # remaining dimensions
    refs: dict[str, np.ndarray | GridRef]  # dim -> shape: (n_rows,)

    errors_low: np.ndarray | None = None
    errors_high: np.ndarray | None = None
    n: np.ndarray | None = None

    @property
    def n_rows(self) -> int:
        return self.values.size

    @property
    def has_errors(self) -> bool:
        return self.errors_low is not None and self.errors_high is not None

    @classmethod
    def from_stat(cls, stat: StatisticArray) -> "PickTable":
        from .sparse_values import SparseStatisticArray

        if isinstance(stat, SparseStatisticArray):
            return stat.to_pick_table()
        stat.validate()

        values = np.asarray(stat.values).reshape(-1)
        dims = stat.dims

        refs = {}
        grid_shape = np.asarray(stat.values).shape

        for axis, dim_name in enumerate(dims):
            coords = stat.dimensions[dim_name].coords

            if coords is None:
                raise ValueError(f"Remaining dimension {dim_name!r} has no coords.")

            refs[dim_name] = GridRef(
                coords=np.asarray(coords),
                grid_shape=grid_shape,
                axis=axis,
            )

        errors_low = None
        errors_high = None
        n = None

        if stat.errors_low is not None:
            errors_low = np.asarray(stat.errors_low).reshape(-1)

        if stat.errors_high is not None:
            errors_high = np.asarray(stat.errors_high).reshape(-1)

        if stat.n is not None:
            n = np.asarray(stat.n).reshape(-1)

        return cls(
            stat=stat,
            values=values,
            dims=dims,
            refs=refs,
            errors_low=errors_low,
            errors_high=errors_high,
            n=n,
        )

    @property
    def canonical_dims(self) -> tuple[str, ...]:
        return tuple(canonical_dim(dim) for dim in self.dims)

    def canonical_refs(self) -> dict[str, np.ndarray]:
        result = {}

        for dim_name in self.dims:
            canonical = canonical_dim(dim_name)

            if canonical in result:
                raise ValueError(
                    f"Cannot canonicalize table with duplicate canonical "
                    f"dimension {canonical!r}. Original dims are {self.dims}."
                )

            result[canonical] = self.refs[dim_name]

        return result

    def filtered(self, filters: tuple[PairFilter, ...]) -> "PickTable":
        table = self

        for f in filters:
            table = table._apply_pair_filter(f)

        return table

    def _apply_pair_filter(self, pair_filter: PairFilter) -> "PickTable":
        if pair_filter.target == "neuron":
            dim_i, dim_j, collapsed_dim = "neuron_i", "neuron_j", "neuron"
        elif pair_filter.target == "session":
            dim_i, dim_j, collapsed_dim = "session_i", "session_j", "session"
        else:
            raise ValueError(pair_filter.target)

        if pair_filter.relation == "all":
            return self

        # Check whether both dimensions are available without expanding
        # their coordinates across the whole table.
        probe = self._row_block(slice(0, min(1, self.n_rows)))

        if probe._values_for_dim(dim_i) is None or probe._values_for_dim(dim_j) is None:
            return self

        if pair_filter.relation not in (
            "same",
            "different",
            "with previous",
        ):
            raise ValueError(pair_filter.relation)

        if pair_filter.relation == "with previous" and pair_filter.target != "session":
            raise ValueError("'with previous' is only meaningful for session pairs.")

        def select(block):
            ref_i = block._values_for_dim(dim_i)
            ref_j = block._values_for_dim(dim_j)

            if pair_filter.relation == "same":
                mask = ref_i == ref_j
            elif pair_filter.relation == "different":
                mask = ref_i != ref_j
            else:
                # session_i = current; session_j = previous/reference.
                mask = ref_j == ref_i - 1

            return np.flatnonzero(mask)

        rows = self._collect_rows(select)

        table = self if rows.size == self.n_rows else self.subset_rows(rows)

        if not pair_filter.collapse_same:
            return table

        if (
            pair_filter.relation == "same"
            and dim_i in table.refs
            and dim_j in table.refs
        ):
            return table.collapse_equal_pair_dims(
                dim_i=dim_i,
                dim_j=dim_j,
                new_dim=collapsed_dim,
            )

        if (
            pair_filter.relation == "with previous"
            and pair_filter.collapse_same
            and dim_i in table.refs
            and dim_j in table.refs
        ):
            table = table.collapse_pair_dims(
                dim_i=dim_i,
                dim_j=dim_j,
                new_dim=collapsed_dim,
                coords=table.refs[dim_i],
            )

        return table

    def iter_row_blocks(self):
        for start in range(0, self.n_rows, ROW_BLOCK_SIZE):
            rows = slice(start, start + ROW_BLOCK_SIZE)
            yield start, self._row_block(rows)

    def _row_block(self, rows):
        """Create a small table with ordinary coordinate arrays."""
        return replace(
            self,
            values=self.values[rows],
            refs={name: column[rows] for name, column in self.refs.items()},
            errors_low=(None if self.errors_low is None else self.errors_low[rows]),
            errors_high=(None if self.errors_high is None else self.errors_high[rows]),
            n=None if self.n is None else self.n[rows],
        )

    def _collect_rows(self, select):
        """Collect selected row indices from bounded table blocks."""
        parts = []

        for start, block in self.iter_row_blocks():
            rows = select(block)

            if rows.size:
                parts.append(rows + start)

        if not parts:
            return np.empty(0, dtype=np.intp)

        if len(parts) == 1:
            return parts[0]

        return np.concatenate(parts)

    def subset_rows(self, rows) -> "PickTable":
        rows = np.atleast_1d(np.asarray(rows, dtype=np.intp))

        refs = {}
        row_maps = {}

        for name, column in self.refs.items():
            if isinstance(column, GridRef):
                # Coordinate columns using the same original row mapping
                # share their new mapping too.
                key = id(column.row_ids)

                if key not in row_maps:
                    row_maps[key] = (
                        rows if column.row_ids is None else column.row_ids[rows]
                    )

                refs[name] = replace(
                    column,
                    row_ids=row_maps[key],
                )
            else:
                refs[name] = column[rows]

        return replace(
            self,
            values=self.values[rows],
            refs=refs,
            errors_low=(None if self.errors_low is None else self.errors_low[rows]),
            errors_high=(None if self.errors_high is None else self.errors_high[rows]),
            n=None if self.n is None else self.n[rows],
        )

    def collapse_pair_dims(
        self,
        *,
        dim_i: str,
        dim_j: str,
        new_dim: str,
        coords: np.ndarray | GridRef,
    ) -> "PickTable":

        if dim_i not in self.refs or dim_j not in self.refs:
            raise ValueError(f"Cannot collapse {dim_i!r}/{dim_j!r}; missing refs.")

        if not isinstance(coords, GridRef):
            coords = np.asarray(coords)

        if coords.shape != (self.n_rows,):
            raise ValueError(
                f"Collapsed coordinates must have shape "
                f"({self.n_rows},), got {coords.shape}."
            )

        new_refs = {
            name: values
            for name, values in self.refs.items()
            if name not in (dim_i, dim_j)
        }

        new_refs[new_dim] = coords

        new_dims = []
        inserted = False

        for dim in self.dims:
            if dim == dim_i:
                if not inserted:
                    new_dims.append(new_dim)
                    inserted = True
            elif dim == dim_j:
                continue
            else:
                new_dims.append(dim)

        return PickTable(
            stat=self.stat,
            values=self.values,
            dims=tuple(new_dims),
            refs=new_refs,
            errors_low=self.errors_low,
            errors_high=self.errors_high,
            n=self.n,
        )

    def collapse_equal_pair_dims(
        self,
        *,
        dim_i: str,
        dim_j: str,
        new_dim: str,
    ) -> "PickTable":

        if dim_i not in self.refs or dim_j not in self.refs:
            raise ValueError(f"Cannot collapse {dim_i!r}/{dim_j!r}; missing refs.")

        if not refs_equal(
            self.refs[dim_i],
            self.refs[dim_j],
        ):
            raise ValueError(
                f"Cannot collapse {dim_i!r}/{dim_j!r}; refs are not equal."
            )

        return self.collapse_pair_dims(
            dim_i=dim_i,
            dim_j=dim_j,
            new_dim=new_dim,
            coords=self.refs[dim_i],
        )

    def indexed_values(
        self,
        dim_name: str,
        size: int,
        *,
        fill_value=np.nan,
    ) -> np.ndarray:

        if self.dims != (dim_name,):
            raise ValueError(
                f"Expected exactly one remaining dimension "
                f"{dim_name!r}, got {self.dims}."
            )

        refs = np.asarray(
            self.refs[dim_name],
            dtype=int,
        )

        values = np.full(
            size,
            fill_value,
            dtype=float,
        )

        values[refs] = self.values

        return values

    def refs_for_rows(
        self,
        rows,
        *,
        include_fixed: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Convert flattened row indices into semantic references.

        Example:
            rows [10, 15, 20]
            -> {"neuron": array([3, 5, 9])}
        """
        rows = np.atleast_1d(np.asarray(rows, dtype=int))

        result = {
            dim_name: np.asarray(ref_values[rows]).reshape(-1)
            for dim_name, ref_values in self.refs.items()
        }

        if include_fixed:
            for dim_name, dim in self.stat.dimensions.items():
                if dim.mode != "fixed":
                    continue

                result[dim_name] = np.full(
                    rows.shape,
                    dim.parameter,
                )

        # Restore original identities from collapsed axes, including the
        # offset between current and previous sessions.
        for original, (compact, offset) in self.stat.reference_aliases.items():
            if original not in result and compact in result:
                result[original] = result[compact] + offset

        return result

    def values_for_rows(self, rows) -> np.ndarray:
        rows = np.asarray(rows, dtype=int)
        return self.values[rows]

    def rows_matching_ref_values(
        self,
        dim_name: str,
        values,
    ) -> np.ndarray:
        """
        Find rows where a remaining dimension has one of the given values.
        Useful for external selection/highlighting.
        """
        if dim_name not in self.refs:
            return np.asarray([], dtype=int)

        values = np.asarray(list(values))
        return self._collect_rows(
            lambda block: np.flatnonzero(np.isin(block.refs[dim_name], values))
        )

    def tooltip_for_row(self, row: int, *, value_name: str = "value") -> str:
        """
        Construct tooltip lazily, without storing labels for all rows.
        """
        refs = self.refs_for_rows([row], include_fixed=True)

        lines = []

        for dim_name in self.stat.dimensions:
            dim = self.stat.dimensions[dim_name]

            if dim.mode == "remaining":
                if dim_name in refs:
                    lines.append(f"{dim_name}: {refs[dim_name][0]}")

            elif dim.mode == "fixed":
                lines.append(f"{dim_name}: {dim.parameter}")

            elif dim.mode == "reduced":
                lines.append(f"{dim_name}: {dim.parameter}")

        lines.append(f"{value_name}: {self.values[row]:.4g}")

        return "\n".join(lines)

    ### ============ reverse mapping: row -> bin, marker, etc. ============
    def _values_for_dim(self, dim_name: str) -> np.ndarray | None:
        """
        Return one value per PickTable row for this dimension.

        Handles both original pair dimensions and collapsed dimensions:
            neuron_i/neuron_j -> neuron
            session_i/session_j -> session
        if the pair dimensions were collapsed.
        """

        # 1. Direct remaining dimension
        if dim_name in self.refs:
            return self.refs[dim_name]

        # A compact previous-session pair is labelled by its current
        # session; its reference coordinate is one session earlier.
        binding = getattr(self.stat, "reference_aliases", {}).get(dim_name)
        if binding is not None:
            compact, offset = binding
            if compact in self.refs:
                return self.refs[compact] + offset
            info = self.stat.dimensions.get(compact)
            if info is not None and info.mode == "fixed":
                return np.full(self.n_rows, info.parameter + offset)
            return None

        # 2. Collapsed alias, e.g. neuron_i -> neuron
        alias = canonical_dim(dim_name)
        if alias is not None and alias in self.refs:
            return self.refs[alias]

        # 3. Fixed dimension in StatisticArray metadata
        if dim_name in self.stat.dimensions:
            dim = self.stat.dimensions[dim_name]

            if dim.mode == "fixed":
                return np.full(self.n_rows, dim.parameter)

            if dim.mode == "remaining":
                # It is marked remaining but not in refs.
                # This should usually not happen.
                return None

            if dim.mode == "reduced":
                return None

        # 4. Fixed collapsed alias
        if alias is not None and alias in self.stat.dimensions:
            dim = self.stat.dimensions[alias]

            if dim.mode == "fixed":
                return np.full(self.n_rows, dim.parameter)

        return None

    def rows_matching_components(
        self,
        components,
        *,
        use_session_filter=True,
        include_self_pairs=True,
    ) -> np.ndarray:
        components = [] if components is None else list(components)

        if not components:
            return np.empty(0, dtype=np.intp)

        return self._collect_rows(
            lambda block: block._rows_matching_components(
                components,
                use_session_filter=use_session_filter,
                include_self_pairs=include_self_pairs,
            )
        )

    def _rows_matching_components(
        self,
        components,
        *,
        use_session_filter: bool = True,
        include_self_pairs: bool = True,
    ) -> np.ndarray:
        """
        Find PickTable rows corresponding to NeuronComponents.

        session_id=None means that the component represents the tracked
        neuron independent of session and therefore acts as a wildcard
        for session matching.

        Pair semantics:
        - one selected neuron:
            match pairs involving that neuron
        - multiple selected neurons:
            match only pairs whose members are both represented in the
            selection
        """

        if not components:
            return np.asarray([], dtype=int)

        components = list(components)

        selected_neurons = {int(component.neuron_id) for component in components}

        neuron_arrays = self._available_neuron_arrays()

        if not neuron_arrays:
            return np.asarray([], dtype=int)

        all_neuron_ids = np.asarray(sorted(selected_neurons), dtype=int)

        wildcard_neurons = set()
        neurons_by_session = {}

        for component in components:
            neuron_id = int(component.neuron_id)

            if component.session_id is None:
                wildcard_neurons.add(neuron_id)
            else:
                neurons_by_session.setdefault(int(component.session_id), set()).add(
                    neuron_id
                )

        wildcard_ids = np.asarray(sorted(wildcard_neurons), dtype=int)

        session_groups = [
            (
                session_id,
                np.asarray(sorted(neurons - wildcard_neurons), dtype=int),
            )
            for session_id, neurons in neurons_by_session.items()
            if neurons - wildcard_neurons
        ]

        def component_mask(neuron_values, session_arrays):
            neuron_values = np.asarray(neuron_values)

            if not use_session_filter or not session_arrays:
                return np.isin(neuron_values, all_neuron_ids)

            # Session-independent selections match this neuron everywhere.
            mask = np.isin(neuron_values, wildcard_ids)

            # Process each selected session once, rather than scanning
            # the whole table separately for every selected component.
            for session_id, neuron_ids in session_groups:
                session_mask = np.zeros(self.n_rows, dtype=bool)

                for session_values in session_arrays:
                    session_mask |= np.asarray(session_values) == session_id

                rows = np.flatnonzero(session_mask & ~mask)

                if rows.size:
                    mask[rows] |= np.isin(
                        neuron_values[rows],
                        neuron_ids,
                    )

            return mask

        # ============================================================
        # Ordinary neuron-wise statistic
        # ============================================================

        if len(neuron_arrays) == 1:

            session_arrays = []

            if use_session_filter:

                # An ordinary neuron may occur with either one session
                # dimension or a session pair.
                for dim_name in (
                    "session",
                    "session_i",
                    "session_j",
                ):
                    values = self._values_for_dim(dim_name)

                    if values is not None:
                        session_arrays.append(values)

            mask = component_mask(
                neuron_arrays[0],
                session_arrays,
            )

            return np.flatnonzero(mask)

        # ============================================================
        # Pairwise neuron statistic
        # ============================================================

        neuron_i, neuron_j = neuron_arrays[:2]

        if use_session_filter:

            session_i = self._values_for_dim("session_i")
            session_j = self._values_for_dim("session_j")

            # _values_for_dim() also resolves collapsed/shared
            # session dimensions where applicable.
            sessions_i = [session_i] if session_i is not None else []

            sessions_j = [session_j] if session_j is not None else []

        else:
            sessions_i = []
            sessions_j = []

        mask_i = component_mask(
            neuron_i,
            sessions_i,
        )

        mask_j = component_mask(
            neuron_j,
            sessions_j,
        )

        if len(selected_neurons) == 1:

            # A single selected neuron highlights all interactions
            # involving that neuron.
            mask = mask_i | mask_j

        else:

            # Multiple selected neurons:
            # both sides of the interaction must belong to the
            # selected component set.
            mask = mask_i & mask_j

            if not include_self_pairs:
                mask &= neuron_i != neuron_j

        return np.flatnonzero(mask)

    def _available_neuron_arrays(self) -> list[np.ndarray]:
        if "neuron_i" in self.refs and "neuron_j" in self.refs:
            return [self.refs["neuron_i"], self.refs["neuron_j"]]

        if "neuron" in self.refs:
            return [self.refs["neuron"]]

        return []

    def _available_session_arrays(self) -> list[np.ndarray]:
        if "session_i" in self.refs and "session_j" in self.refs:
            return [self.refs["session_i"], self.refs["session_j"]]

        if "session" in self.refs:
            return [self.refs["session"]]

        return []
