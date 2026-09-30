from dataclasses import dataclass, fields, is_dataclass

from typing import Optional
from collections.abc import Callable

from PySide6.QtCore import QObject, Qt, Signal
import numpy as np
from threading import RLock
from functools import partial

from catan.core.changes import Change, ChangeKind, DataChange
from .table import PickTable
from .types import StatisticArray
from .queries import ReductionSpec, StatisticQuery


def _retained_array_bytes(value):
    """Count unique NumPy backing allocations, including sparse coordinates."""
    seen = set()

    def visit(item):
        if isinstance(item, np.ndarray):
            while isinstance(item.base, np.ndarray):
                item = item.base

        identity = id(item)

        if identity in seen:
            return 0

        seen.add(identity)

        if isinstance(item, np.ndarray):
            return item.nbytes

        if is_dataclass(item) and not isinstance(item, type):
            return sum(
                visit(getattr(item, field.name))
                for field in fields(item)
            )

        if isinstance(item, dict):
            return sum(visit(value) for value in item.values())

        if isinstance(item, (tuple, list)):
            return sum(visit(value) for value in item)

        return 0

    return visit(value)

@dataclass(frozen=True, slots=True)
class StatisticsTaskResult:
    slot: str
    query: StatisticQuery
    statistic_revision: int

    table: PickTable | None = None
    error: str | None = None
    traceback: str | None = None

    @property
    def successful(self) -> bool:
        return self.error is None and self.table is not None


class StatisticEngine(QObject):

    registry_changed = Signal()
    values_changed = Signal(object)

    def __init__(
        self,
        data,
        state,
        registry_factory: Callable,
        parent=None,
    ):
        super().__init__(parent)

        self.data = data
        self.state = state
        self.registry_factory = registry_factory

        self._cache = {}
        self.cache_max_bytes = 128 * 1024**2
        self.cache_max_entries = 8

        self._revisions = {}
        self._cache_lock = RLock()

        self._registry_signatures = {}
        self.registry_changes = frozenset()

        self.refresh_registry()

        self.state.data_changed.connect(self._on_data_changed)
        self.state.statistics_sources_changed.connect(self.refresh_registry)


    def data_version(self):
        # Increase/change this whenever tracking/data/statistics change.
        return getattr(self.state, "data_version", 0)

    def _on_data_changed(self, event: DataChange):
        if not isinstance(event, DataChange):
            raise TypeError("data_changed must carry a DataChange.")

        check_registry = event.has(
            ChangeKind.SESSION_ADDED,
            ChangeKind.SESSION_REMOVED,
            ChangeKind.SESSION_ORDER,
            ChangeKind.ASSIGNMENT_SET,
            ChangeKind.ASSIGNMENT_MAPPING,
            ChangeKind.QUALITY_VALUES,
            ChangeKind.MATCH_VALUES,
            ChangeKind.DATA_AVAILABILITY,
        )

        registry_changes = (
            self.refresh_registry(emit=False)
            if check_registry
            else frozenset()
        )

        value_changes = self._invalidate(
            key
            for key, definition in self.registry.items()
            if key not in registry_changes
            and definition.is_affected_by(event)
        )

        if registry_changes:
            self.registry_changed.emit()

        if value_changes:
            self.values_changed.emit(value_changes)
            
    def _definition_signature(self, definition):
        # Registry factories recreate partials; compare their configuration,
        # not the identity of the newly created partial object.
        getter = definition.getter
        if isinstance(getter, partial):
            getter = (
                getter.func,
                getter.args,
                tuple(sorted((getter.keywords or {}).items())),
            )

        coordinates = None
        if (
            definition.dims
            and getattr(self.state, "assignments", None) is not None
        ):
            coords = definition.coord_getter(self.state)
            coordinates = tuple(
                (dim, tuple(np.asarray(coords[dim]).tolist()))
                for dim in definition.dims
            )

        # Renaming a loaded quality field can change which sessions supply
        # a statistic even when both names already exist in the registry.
        sources = None
        if definition.key.startswith("session:"):
            name = definition.key.split(":", 1)[1]
            sources = tuple(
                (str(session.path), name in session.quality)
                for session in self.data.sessions
                if session is not None
            )

        return (
            definition.title,
            definition.description,
            definition.category,
            tuple(definition.dims),
            getter,
            coordinates,
            (
                None
                if definition.allowed_reductions is None
                else tuple(
                    sorted(
                        (dim, tuple(methods))
                        for dim, methods in definition.allowed_reductions.items()
                    )
                )
            ),
            (
                None
                if definition.default_reductions is None
                else tuple(sorted(definition.default_reductions.items()))
            ),
            definition.dependencies,
            definition.availability_dependencies,
            sources,
        )

    def refresh_registry(self, *, emit=True):
        registry = self.registry_factory(self.data, self.state)
        signatures = {
            key: self._definition_signature(definition)
            for key, definition in registry.items()
        }

        with self._cache_lock:
            previous = self._registry_signatures

            changed = frozenset(
                key
                for key in previous.keys() | signatures.keys()
                if previous.get(key) != signatures.get(key)
            )

            self.registry = registry
            self._registry_signatures = signatures
            self.registry_changes = changed

            if changed:
                self._invalidate(changed)

        if changed and emit:
            self.registry_changed.emit()

        return changed


    def query_revision(self, query):
        if query is None:
            return 0

        with self._cache_lock:
            return self._revisions.get(query.statistic_key, 0)


    def _invalidate(self, keys):
        keys = frozenset(keys)

        with self._cache_lock:
            for key in keys:
                self._revisions[key] = self._revisions.get(key, 0) + 1

            self._cache = {
                cache_key: value
                for cache_key, value in self._cache.items()
                if cache_key[0].statistic_key not in keys
            }

        return keys

    def clear_cache(self):
        with self._cache_lock:
            self._cache.clear()

    def evaluate(
        self,
        query: Optional[StatisticQuery],
    ) -> Optional[StatisticArray]:
        if self.data is None or len(self.data.sessions) == 0 or query is None:
            return None

        with self._cache_lock:
            revision = self._revisions.get(query.statistic_key, 0)
            key = (query, revision)

            if key in self._cache:
                ## move entry to the end of the dict
                result = self._cache.pop(key)
                self._cache[key] = result
                return result

            definition = self.registry[query.statistic_key]

        # Calculation happens outside the cache lock.
        result = self._evaluate_uncached(query, definition)

        with self._cache_lock:
            # A relevant change during calculation must not repopulate
            # the cache with an obsolete result.
            if self._revisions.get(query.statistic_key, 0) == revision:
                if (
                    self.cache_max_entries > 0
                    and _retained_array_bytes(result) <= self.cache_max_bytes
                ):
                    self._cache[key] = result

                    while (
                        len(self._cache) > self.cache_max_entries
                        or _retained_array_bytes(list(self._cache.values()))
                        > self.cache_max_bytes
                    ):
                        oldest_key = next(iter(self._cache))
                        self._cache.pop(oldest_key)

        return result

    def _evaluate_uncached(
        self,
        query: StatisticQuery,
        stat_def=None,
    ) -> StatisticArray:
        if stat_def is None:
            stat_def = self.registry[query.statistic_key]
        reductions = query.reduction_dict()
        if stat_def.key == "footprint_similarity":
            from .queries import normalize_footprint_pair_reductions
            reductions = normalize_footprint_pair_reductions(
                stat_def, reductions, query.filters,
                session_series=query.context == "session_series",
            )

        indexers = {
            dim: spec.index
            for dim, spec in reductions.items()
            if spec.method == "single"
        }

        stat = stat_def.get_values(
            data=self.data,
            state=self.state,
            indexers=indexers,
            filters=query.filters,
        )

        requested_order = tuple(query.reduction_order or ())
        reduction_order = requested_order + tuple(
            dim for dim in stat_def.dims
            if dim not in requested_order
            and reductions.get(dim, ReductionSpec("keep")).method
            not in ("keep", "single")
        )

        planned_reductions = {}

        for dim in reduction_order:
            spec = reductions.get(dim)

            if spec is None or spec.method in ("keep", "single"):
                continue

            target = stat.reduction_aliases.get(dim, dim)

            if target not in stat.dims:
                continue

            previous = planned_reductions.get(target)
            if previous is not None and previous != spec:
                raise ValueError(
                    f"Linked dimensions map to {target!r}, but "
                    "have conflicting reductions. Choose the same "
                    "reduction for both, or keep one."
                )

            planned_reductions[target] = spec

        for target, spec in planned_reductions.items():
            stat.reduce_dimension(target, spec)

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

        remaining_filters = tuple(
            f
            for f in (query.filters or ())
            if f.target not in stat.applied_pair_filters
        )

        if remaining_filters:
            table = table.filtered(remaining_filters)

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
