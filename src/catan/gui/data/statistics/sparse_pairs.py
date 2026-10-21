"""Sparse evaluation of selected pair statistics.

Several pair statistics are only selectable for spatially close footprints
(overlap > x, distance < r). For such selections, candidate pairs are
enumerated directly, instead of scanning the complete neuron × neuron ×
session × session grid in blocks. The collector below reproduces the
coordinates, filters and metadata of the blockwise path.
"""

from copy import deepcopy

import numpy as np


def only_kept_axes(query):
    return all(
        spec.method in ("keep", "single") for spec in query.reduction_dict().values()
    )


def selects(select_values, value):
    try:
        mask = np.asarray(
            select_values(np.array([value], dtype=np.float32)),
            dtype=bool,
        )
    except Exception:
        return None

    return bool(mask[0]) if mask.shape == (1,) else None


def selection_upper_bound(select_values, limit, samples=4097):
    """Conservative upper bound of an interval-shaped value selection.

    Returns None if values at or beyond ``limit`` may be selected, 0.0 if
    nothing in [0, limit] is selected, otherwise a radius strictly above
    the largest accepted value.
    """
    if not np.isfinite(limit) or limit <= 0:
        return None

    if selects(select_values, limit) is not False:
        return None

    grid = np.linspace(0, limit, samples, dtype=np.float32)

    try:
        accepted = np.asarray(select_values(grid), dtype=bool)
    except Exception:
        return None

    if accepted.shape != grid.shape:
        return None

    if not np.any(accepted):
        return 0.0

    last = int(np.flatnonzero(accepted)[-1])

    return float(grid[min(last + 1, samples - 1)])


class SparsePairCollector:
    """Collects selected values at physical pair coordinates."""

    def __init__(self, names, dims, aliases, references, n, s, query, select_values):
        self.names = names
        self.dims = dims
        self.aliases = aliases
        self.references = references
        self.query = query
        self.select_values = select_values

        self.output_dims = tuple(
            name for name, info in dims.items() if info.mode == "remaining"
        )
        self.lookups, self.fixed = {}, {}

        for name in names:
            logical, offset = references.get(name, (name, 0))
            info = dims[logical]
            size = n if name.startswith("neuron") else s

            if info.mode == "fixed":
                self.fixed[name] = int(info.parameter) + offset
                continue

            lookup = np.full(size, -1, dtype=np.int64)
            physical = np.asarray(info.coords, dtype=np.int64) + offset
            inside = (physical >= 0) & (physical < size)
            lookup[physical[inside]] = np.flatnonzero(inside)
            self.lookups[name] = (logical, lookup)

        self.sizes = {name: (n if name.startswith("neuron") else s) for name in names}

        self.relations = {}

        for f in query.filters:
            if f.target + "_i" not in names or f.target + "_j" not in names:
                continue

            if f.relation not in ("all", "same", "different") and not (
                f.relation == "with previous" and f.target == "session"
            ):
                raise ValueError(f"Unsupported {f.target} relation: {f.relation}")

            self.relations[f.target] = f.relation

        self.values, self.positions = [], []

    def domain(self, name):
        """Allowed physical coordinates for an original pair axis."""
        size = self.sizes[name]

        if name in self.fixed:
            value = self.fixed[name]
            return np.array([value] if 0 <= value < size else [], dtype=np.int64)

        return np.flatnonzero(self.lookups[name][1] >= 0)

    def mask(self, name):
        result = np.zeros(self.sizes[name], dtype=bool)
        result[self.domain(name)] = True
        return result

    def related(self, target, left, right):
        relation = self.relations.get(target, "all")

        if relation == "same":
            return left == right
        if relation == "different":
            return left != right
        if relation == "with previous":
            return right == left - 1
        return np.ones(np.broadcast(left, right).shape, dtype=bool)

    def emit(self, physical, values):
        """Select rows; ``physical`` maps every original axis to coordinates."""
        values = np.asarray(values, dtype=np.float32)
        keep = ~np.isnan(values)

        for target in self.relations:
            keep &= self.related(
                target,
                physical[target + "_i"],
                physical[target + "_j"],
            )

        positions = {}

        for name in self.names:
            coordinate = np.broadcast_to(physical[name], values.shape)

            if name in self.fixed:
                keep &= coordinate == self.fixed[name]
                continue

            logical, lookup = self.lookups[name]
            position = lookup[coordinate]
            keep &= position >= 0

            if logical in positions:
                # Linked coordinates must agree (same / with previous).
                keep &= positions[logical] == position
            else:
                positions[logical] = position

        if not np.any(keep):
            return

        selected = np.asarray(self.select_values(values), dtype=bool) & keep

        if not np.any(selected):
            return

        self.values.append(values[selected])
        self.positions.append(
            np.column_stack([positions[d][selected] for d in self.output_dims])
            if self.output_dims
            else np.empty((int(selected.sum()), 0), dtype=np.int64)
        )

    def result(self):
        from .sparse_values import SparseStatisticArray

        if self.values:
            values = np.concatenate(self.values)
            positions = np.concatenate(self.positions, axis=0).astype(np.int64)
        else:
            values = np.empty(0, dtype=np.float32)
            positions = np.empty((0, len(self.output_dims)), dtype=np.int64)

        if len(values) and self.output_dims:
            order = np.lexsort(positions.T[::-1])
            values, positions = values[order], positions[order]

        result = SparseStatisticArray(
            name="",
            title="",
            dimensions=deepcopy(self.dims),
            values=values,
            positions=positions,
            errors_low=None,
            errors_high=None,
            n=None,
            reduction_aliases=dict(self.aliases),
            reference_aliases=dict(self.references),
            applied_pair_filters=tuple(f.target for f in self.query.filters),
        )
        result.validate()

        return result