from typing import Optional

from dataclasses import dataclass
import numpy as np

from .dimensions import canonical_dim

# from .plotdata_series import validate_session_series_reductions

from .statistics import StatisticArray

# from .dimensions import NEURON_DIMS, SESSION_DIMS, canonical_dim_name, DIM_CANONICAL
from .queries import ReductionSpec, StatisticQuery, PairFilter


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
    refs: dict[str, np.ndarray]  # dim -> shape: (n_rows,)

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
        stat.validate()

        values = np.asarray(stat.values).reshape(-1)
        dims = stat.dims

        refs: dict[str, np.ndarray] = {}

        if dims:
            coord_arrays = []

            for dim_name in dims:
                dim = stat.dimensions[dim_name]
                coords = dim.coords

                if coords is None:
                    raise ValueError(f"Remaining dimension {dim_name!r} has no coords.")

                coord_arrays.append(np.asarray(coords))

            grids = np.meshgrid(*coord_arrays, indexing="ij")

            refs = {dim_name: grid.reshape(-1) for dim_name, grid in zip(dims, grids)}

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

        # If the pair dimensions are not present, this filter is irrelevant.
        if dim_i not in self.refs or dim_j not in self.refs:
            return self

        ref_i = self.refs[dim_i]
        ref_j = self.refs[dim_j]

        if pair_filter.relation == "all":
            mask = np.ones(self.n_rows, dtype=bool)

        elif pair_filter.relation == "same":
            mask = ref_i == ref_j

        elif pair_filter.relation == "different":
            mask = ref_i != ref_j
        elif pair_filter.relation == "with previous":
            # For each row, check if the target dimension matches the previous row's target.
            # This assumes that the rows are sorted by the target dimension.
            mask = np.zeros(self.n_rows, dtype=bool)
            if self.n_rows > 1:
                mask[1:] = ref_i[1:] == ref_i[:-1]
        else:
            raise ValueError(pair_filter.relation)

        table = self.subset_rows(np.flatnonzero(mask))

        if (
            pair_filter.relation == "same"
            and pair_filter.collapse_same
            and dim_i in table.refs
            and dim_j in table.refs
        ):
            table = table.collapse_equal_pair_dims(
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
            table = table.collapse_equal_pair_dims(
                dim_i=dim_i,
                dim_j=dim_j,
                new_dim=collapsed_dim,
            )

        return table

    def subset_rows(self, rows) -> "PickTable":
        rows = np.asarray(rows, dtype=int)

        return PickTable(
            stat=self.stat,
            values=self.values[rows],
            dims=self.dims,
            refs={
                dim_name: ref_values[rows] for dim_name, ref_values in self.refs.items()
            },
            errors_low=None if self.errors_low is None else self.errors_low[rows],
            errors_high=None if self.errors_high is None else self.errors_high[rows],
            n=None if self.n is None else self.n[rows],
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

        if not np.array_equal(self.refs[dim_i], self.refs[dim_j]):
            raise ValueError(
                f"Cannot collapse {dim_i!r}/{dim_j!r}; refs are not equal."
            )

        new_refs = {
            dim_name: ref_values
            for dim_name, ref_values in self.refs.items()
            if dim_name not in (dim_i, dim_j)
        }

        new_refs[new_dim] = self.refs[dim_i]

        new_dims = []
        inserted = False

        for dim_name in self.dims:
            if dim_name == dim_i:
                if not inserted:
                    new_dims.append(new_dim)
                    inserted = True
            elif dim_name == dim_j:
                continue
            else:
                new_dims.append(dim_name)

        return PickTable(
            stat=self.stat,
            values=self.values,
            dims=tuple(new_dims),
            refs=new_refs,
            errors_low=self.errors_low,
            errors_high=self.errors_high,
            n=self.n,
        )

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
        rows = np.asarray(rows, dtype=int)

        result = {
            dim_name: np.atleast_1d(ref_values[rows])
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
        mask = np.isin(self.refs[dim_name], values)
        return np.flatnonzero(mask)

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

    # def _combined_values_for_dims(self, dim_names: tuple[str, ...]) -> list[np.ndarray]:
    #     arrays = []
    #     seen_sources = set()

    #     for dim_name in dim_names:
    #         values = self._values_for_dim(dim_name)
    #         if values is None:
    #             continue

    #         # identify by object id / data pointer-ish enough for this use case
    #         source_id = id(values)

    #         if source_id in seen_sources:
    #             continue

    #         seen_sources.add(source_id)
    #         arrays.append(values)

    #     return arrays

    def rows_matching_components(
        self,
        components,
        *,
        use_session_filter: bool = True,
        include_self_pairs: bool = True,
    ) -> np.ndarray:
        """
        Find PickTable rows corresponding to a list of NeuronComponent objects.

        A row matches a component if:
        - a neuron dimension is present/fixed and matches component.neuron_id
        - and, if a session dimension is present/fixed, it matches component.session_id

        If the session dimension was reduced, the session is not used as a filter.
        """

        if not components:
            return np.asarray([], dtype=int)

        selected_neurons = np.asarray(
            sorted({int(c.neuron_id) for c in components}),
            dtype=int,
        )

        selected_sessions = np.asarray(
            sorted({int(c.session_id) for c in components}),
            dtype=int,
        )
        # print(f"table dims:", self.dims)
        # print(f"table values:", self.values)
        # print(
        #     f"rows_matching_components: selected_neurons={selected_neurons}, selected_sessions={selected_sessions}"
        # )

        neuron_arrays = self._available_neuron_arrays()
        session_arrays = self._available_session_arrays()

        # print(f"neuron_arrays: {neuron_arrays}")
        # print(f"session_arrays: {session_arrays}")

        has_neuron_context = len(neuron_arrays) > 0
        has_session_context = len(session_arrays) > 0

        if not has_neuron_context and not has_session_context:
            return np.asarray([], dtype=int)

        mask = np.zeros(self.n_rows, dtype=bool)

        is_pairwise = len(neuron_arrays) >= 2

        if not is_pairwise:
            # Ordinary neuron-wise statistic
            mask = np.isin(neuron_arrays[0], selected_neurons)

        else:
            if len(selected_neurons) == 1:
                # One selected neuron:
                # highlight all pairs involving this neuron.
                mask = np.zeros(self.n_rows, dtype=bool)

                for arr in neuron_arrays:
                    mask |= arr == selected_neurons[0]

            else:
                # Two or more selected neurons:
                # highlight only interactions among selected neurons.
                mask = np.ones(self.n_rows, dtype=bool)

                for arr in neuron_arrays:
                    mask &= np.isin(arr, selected_neurons)

                if not include_self_pairs and len(neuron_arrays) == 2:
                    mask &= neuron_arrays[0] != neuron_arrays[1]

        if use_session_filter:
            session_arrays = []

            for dim_name in ("session", "session_i", "session_j"):
                arr = self._values_for_dim(dim_name)
                if arr is not None:
                    session_arrays.append(arr)

            if session_arrays:
                session_mask = np.zeros(self.n_rows, dtype=bool)

                for arr in session_arrays:
                    session_mask |= np.isin(arr, selected_sessions)

                mask &= session_mask

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


class StatisticEngine:
    def __init__(self, registry, data, state):
        self.registry = registry  ## contains the dict of possible statistics
        self.data = data
        self.state = state
        self._cache = {}

    def data_version(self):
        # Increase/change this whenever tracking/data/statistics change.
        # Could be a counter on your global state.
        return getattr(self.state, "data_version", 0)

    def clear_cache(self):
        self._cache.clear()

    def evaluate(self, query: Optional[StatisticQuery]) -> Optional[StatisticArray]:

        if self.data is None or len(self.data.sessions) == 0 or query is None:
            return None

        key = (query, self.data_version())

        if self._cache and key in self._cache:
            return self._cache[key]

        result = self._evaluate_uncached(query)

        self._cache[key] = result
        return result

    def _evaluate_uncached(self, query: StatisticQuery) -> StatisticArray:
        stat_def = self.registry[query.statistic_key]
        reductions = query.reduction_dict()

        indexers = {
            dim: spec.index
            for dim, spec in reductions.items()
            if spec.method == "single"
        }

        stat = stat_def.get_values(
            data=self.data,
            state=self.state,
            indexers=indexers,
        )

        reduction_order = query.reduction_order

        if not reduction_order:
            reduction_order = tuple(
                dim
                for dim in stat_def.dims
                if reductions.get(dim, ReductionSpec("keep")).method
                not in ("keep", "single")
            )

        for dim in reduction_order:
            spec = reductions.get(dim)

            if spec is None:
                continue

            if spec.method in ("keep", "single"):
                continue

            if dim not in stat.dims:
                continue

            stat.reduce_dimension(dim, spec)

        return stat

    def evaluate_table(self, query: Optional[StatisticQuery]) -> Optional[PickTable]:
        if query is None:
            return None

        if query.statistic_key == "none":
            return None

        stat = self.evaluate(query)
        if stat is None:
            return None

        table = PickTable.from_stat(stat)

        if getattr(query, "filters", None):
            table = table.filtered(query.filters)

        return table

    def _validate_filters_possible(self, query: StatisticQuery, stat: StatisticArray):
        remaining = set(stat.dims)

        for f in getattr(query, "filters", ()):
            if f.target == "neuron":
                required = {"neuron_i", "neuron_j"}
            elif f.target == "session":
                required = {"session_i", "session_j"}
            else:
                continue

            missing = required - remaining

            if missing:
                raise ValueError(
                    f"Cannot apply {f.target} filter {f.relation!r}; "
                    f"required dimensions {required} are not available after reductions. "
                    f"Remaining dims are {stat.dims}. Missing: {missing}."
                )
