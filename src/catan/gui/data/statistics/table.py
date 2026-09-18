from dataclasses import dataclass

import numpy as np

from .dimensions import canonical_dim
from .queries import PairFilter
from .types import StatisticArray


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

        ref_i = self._values_for_dim(dim_i)
        ref_j = self._values_for_dim(dim_j)

        if ref_i is None or ref_j is None:
            return self

        if pair_filter.relation == "all":
            return self

        elif pair_filter.relation == "same":
            mask = ref_i == ref_j

        elif pair_filter.relation == "different":
            mask = ref_i != ref_j

        elif pair_filter.relation == "with previous":
            if pair_filter.target != "session":
                raise ValueError(
                    "'with previous' is only meaningful for session pairs."
                )

            # session_i = current session
            # session_j = previous/reference session
            mask = ref_j == ref_i - 1

        else:
            raise ValueError(pair_filter.relation)

        table = self.subset_rows(np.flatnonzero(mask))

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

    def subset_rows(self, rows) -> "PickTable":
        rows = np.atleast_1d(np.asarray(rows, dtype=int))

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

    def collapse_pair_dims(
        self,
        *,
        dim_i: str,
        dim_j: str,
        new_dim: str,
        coords: np.ndarray,
    ) -> "PickTable":

        if dim_i not in self.refs or dim_j not in self.refs:
            raise ValueError(f"Cannot collapse {dim_i!r}/{dim_j!r}; missing refs.")

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

        if not np.array_equal(
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

    def rows_matching_components(
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

        selected_neurons = {
            int(component.neuron_id)
            for component in components
        }

        neuron_arrays = self._available_neuron_arrays()

        if not neuron_arrays:
            return np.asarray([], dtype=int)

        def component_mask(
            neuron_values,
            session_arrays,
        ):
            """
            Rows on one semantic neuron axis matching at least one
            selected component.
            """

            neuron_values = np.asarray(neuron_values)

            mask = np.zeros(
                self.n_rows,
                dtype=bool,
            )

            for component in components:

                component_mask = (
                    neuron_values
                    == int(component.neuron_id)
                )

                # None = tracked-neuron identity,
                # independent of session.
                if (
                    use_session_filter
                    and component.session_id is not None
                    and session_arrays
                ):

                    session_mask = np.zeros(
                        self.n_rows,
                        dtype=bool,
                    )

                    for session_values in session_arrays:
                        session_mask |= (
                            np.asarray(session_values)
                            == int(component.session_id)
                        )

                    component_mask &= session_mask

                mask |= component_mask

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
                    values = self._values_for_dim(
                        dim_name
                    )

                    if values is not None:
                        session_arrays.append(
                            values
                        )

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

            session_i = self._values_for_dim(
                "session_i"
            )
            session_j = self._values_for_dim(
                "session_j"
            )

            # _values_for_dim() also resolves collapsed/shared
            # session dimensions where applicable.
            sessions_i = (
                [session_i]
                if session_i is not None
                else []
            )

            sessions_j = (
                [session_j]
                if session_j is not None
                else []
            )

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
