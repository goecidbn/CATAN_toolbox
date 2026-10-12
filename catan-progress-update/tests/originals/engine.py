from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Optional
from collections.abc import Callable
from copy import copy
from types import SimpleNamespace

from PySide6.QtCore import QObject, Qt, Signal
import numpy as np
from threading import RLock
from functools import partial

from catan.core.changes import Change, ChangeKind, DataChange
from .table import PickTable
from .types import StatisticArray
from .queries import ReductionSpec, StatisticQuery


import cProfile
import io
import pstats
import traceback

from contextvars import ContextVar
from functools import wraps
from itertools import count
from time import perf_counter


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
            return sum(visit(getattr(item, field.name)) for field in fields(item))

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

    table: PickTable | dict[str, PickTable] | None = None
    assignment_names: tuple[str, ...] = ()
    data_version: int | None = None

    error: str | None = None
    traceback: str | None = None

    @property
    def successful(self) -> bool:
        return self.error is None and self.table is not None


TRACE_STATISTICS = False
PROFILE_SLOW_STATISTICS = False
PROFILE_THRESHOLD_MS = 100.0

_statistic_request = ContextVar("statistic_request", default=None)
_statistic_request_ids = count(1)


def trace_statistic_request(method):
    @wraps(method)
    def wrapped(self, query, *args, **kwargs):
        # Trace the outermost entry only. For example:
        # evaluate_table -> evaluate -> _evaluate_uncached.
        if not TRACE_STATISTICS or _statistic_request.get() is not None:
            return method(self, query, *args, **kwargs)

        request_id = next(_statistic_request_ids)
        token = _statistic_request.set(request_id)

        caller = " -> ".join(
            f"{frame.name}:{frame.lineno}"
            for frame in traceback.extract_stack(limit=6)[:-1]
        )

        print(
            f"\n[stat #{request_id}] START {method.__name__}\n"
            f"  caller: {caller}\n"
            f"  query: {query!r}",
            flush=True,
        )

        profiler = cProfile.Profile() if PROFILE_SLOW_STATISTICS else None

        started = perf_counter()
        outcome = "OK"

        try:
            if profiler is not None:
                profiler.enable()

            return method(self, query, *args, **kwargs)

        except BaseException as exc:
            outcome = type(exc).__name__
            raise

        finally:
            if profiler is not None:
                profiler.disable()

            elapsed_ms = 1000 * (perf_counter() - started)
            _statistic_request.reset(token)

            print(
                f"[stat #{request_id}] END {outcome}: " f"{elapsed_ms:.3f} ms",
                flush=True,
            )

            if profiler is not None and elapsed_ms >= PROFILE_THRESHOLD_MS:
                stream = io.StringIO()

                pstats.Stats(profiler, stream=stream).sort_stats(
                    "cumulative"
                ).print_stats(20)

                print(
                    f"[stat #{request_id}] PROFILE\n{stream.getvalue()}",
                    flush=True,
                )

    return wrapped


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
            self.refresh_registry(emit=False) if check_registry else frozenset()
        )

        value_changes = self._invalidate(
            key
            for key, definition in self.registry.items()
            if key not in registry_changes and definition.is_affected_by(event)
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
        if definition.dims and getattr(self.state, "assignments", None) is not None:
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
            definition.parameters,
            definition.independent_pairs,
            definition.query_getter,
            definition.compact_values,
            definition.component_axes,
            definition.curation_kind,
            definition.relationship_axes,
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

    @trace_statistic_request
    def evaluate(
        self,
        query: Optional[StatisticQuery],
    ) -> Optional[StatisticArray]:
        if self.data is None or len(self.data.sessions) == 0 or query is None:
            return None

        with self._cache_lock:
            definition = self.registry[query.statistic_key]

            parameters = definition.resolve_parameters(dict(query.parameters))
            reductions = query.reduction_dict()

            if definition.independent_pairs:
                from .queries import normalize_footprint_pair_reductions

                reductions = normalize_footprint_pair_reductions(
                    definition,
                    reductions,
                    query.filters,
                    session_series=query.context == "session_series",
                )

            query = replace(
                query,
                reductions=tuple(sorted(reductions.items())),
                parameters=tuple(sorted(parameters.items())),
            )

            revision = self._revisions.get(query.statistic_key, 0)
            key = (query, revision)

            if key in self._cache:

                if TRACE_STATISTICS:
                    print(
                        f"[stat #{_statistic_request.get()}] CACHE HIT",
                        flush=True,
                    )

                ## move entry to the end of the dict
                result = self._cache.pop(key)
                self._cache[key] = result
                return result

        # Calculation happens outside the cache lock.
        result = self._evaluate_uncached(query, definition)
        if definition.compact_values:
            _compact_statistic_values(result)

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

    @trace_statistic_request
    def _evaluate_uncached(
        self,
        query: StatisticQuery,
        stat_def=None,
        *,
        select_values=None,
        requested_pairs=None,
        data=None,
        state=None,
    ) -> StatisticArray:

        data = self.data if data is None else data
        state = self.state if state is None else state

        if stat_def is None:
            stat_def = self.registry[query.statistic_key]

        reductions = query.reduction_dict()
        if stat_def.independent_pairs:

            from .queries import normalize_footprint_pair_reductions

            reductions = normalize_footprint_pair_reductions(
                stat_def,
                reductions,
                query.filters,
                session_series=query.context == "session_series",
            )

        if stat_def.query_getter is not None:
            prepared_query = replace(
                query,
                reductions=tuple(reductions.items()),
            )

            parameters = stat_def.resolve_parameters(dict(query.parameters))

            execution_options = {}

            if select_values is not None:
                execution_options["select_values"] = select_values

            if requested_pairs is not None:
                execution_options["requested_pairs"] = requested_pairs

            stat = stat_def.query_getter(
                data,
                state,
                prepared_query,
                **parameters,
                **execution_options,
            )

            stat.name = stat_def.key
            stat.title = stat_def.title_with_parameters(parameters)
            stat.category = stat_def.category
            stat.component_axes = stat_def.component_axes
            stat.curation_kind = stat_def.curation_kind
            stat.relationship_axes = stat_def.relationship_axes
            stat.validate()

            return stat

        indexers = {
            dim: spec.index
            for dim, spec in reductions.items()
            if spec.method == "single"
        }

        stat = stat_def.get_values(
            data=data,
            state=state,
            indexers=indexers,
            filters=query.filters,
            parameters=dict(query.parameters),
        )

        requested_order = tuple(query.reduction_order or ())
        reduction_order = requested_order + tuple(
            dim
            for dim in stat_def.dims
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

    @trace_statistic_request
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

    def evaluate_assignment_table(
        self,
        query: StatisticQuery,
        assignment_name: str,
    ) -> PickTable | None:
        """Evaluate without changing the globally active assignments."""
        if query is None or query.statistic_key == "none":
            return None

        assignments = self.data._assignments.get(assignment_name)

        if assignments is None:
            raise ValueError(
                f"Assignments dataset {assignment_name!r} is no longer loaded."
            )

        if not self.data.sessions:
            raise ValueError("Load session data before calculating statistics.")

        # Copy the small identity/context objects. Large spatial and temporal
        # arrays remain shared with the loaded data.
        bound_assignments = copy(assignments)
        bound_assignments.ids = assignments.ids.copy()
        bound_assignments.stats = dict(assignments.stats)

        if assignments.union is not None:
            bound_assignments.union = copy(assignments.union)

        bound_state = SimpleNamespace(assignments=bound_assignments.ids)

        bound_data = copy(self.data)
        bound_data._current_assignments = assignment_name
        bound_data._assignments = {assignment_name: bound_assignments}
        bound_data.sessions = list(self.data.sessions)
        bound_data.state = bound_state

        registry = self.registry_factory(bound_data, bound_state)
        definition = registry.get(query.statistic_key)

        if definition is None:
            raise ValueError(
                f"{assignment_name}: statistic {query.statistic_key!r} "
                "is not available for this assignments dataset."
            )

        parameters = definition.resolve_parameters(dict(query.parameters))
        prepared_query = replace(
            query,
            parameters=tuple(sorted(parameters.items())),
        )

        stat = self._evaluate_uncached(
            prepared_query,
            definition,
            data=bound_data,
            state=bound_state,
        )

        if definition.compact_values:
            _compact_statistic_values(stat)

        table = PickTable.from_stat(stat)

        remaining_filters = tuple(
            item
            for item in (query.filters or ())
            if item.target not in stat.applied_pair_filters
        )

        if remaining_filters:
            table = table.filtered(remaining_filters)

        return table

    def evaluate_matching_table(
        self,
        query,
        select_values,
        *,
        requested_pairs=None,
    ):
        """Return selected rows without caching a threshold-specific result."""
        if query is None or query.statistic_key == "none":
            return None

        if self.data is None or not self.data.sessions:
            return None

        if requested_pairs is not None and not self.supports_requested_pairs(query):
            raise ValueError(
                "This statistic does not support evaluation of requested pairs."
            )

        with self._cache_lock:
            definition = self.registry[query.statistic_key]
            revision = self._revisions.get(query.statistic_key, 0)

        def select(values):
            mask = select_values(values)

            if mask is None:
                raise ValueError("The value selector returned no mask.")

            mask = np.asarray(mask, dtype=bool)

            if mask.shape != values.shape:
                raise ValueError("The value selector returned an invalid mask shape.")

            return mask & ~np.isnan(values)

        getter = definition.query_getter

        while isinstance(getter, partial):
            getter = getter.func

        if getattr(getter, "supports_value_selection", False):
            stat = self._evaluate_uncached(
                query,
                definition,
                select_values=select,
                requested_pairs=requested_pairs,
            )

            if definition.compact_values:
                _compact_statistic_values(stat)

            table = PickTable.from_stat(stat)

            remaining = tuple(
                f
                for f in (query.filters or ())
                if f.target not in stat.applied_pair_filters
            )

            if remaining:
                table = table.filtered(remaining)

        else:
            # Existing getters, including sparse footprint similarity,
            # retain their current calculation and reduction paths.
            table = self.evaluate_table(query)

            if table is not None:
                table = table.subset_rows(np.flatnonzero(select(table.values)))

        with self._cache_lock:
            if self._revisions.get(query.statistic_key, 0) != revision:
                raise RuntimeError("The data changed during filtering. Evaluate again.")

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

    def supports_requested_pairs(self, query):
        definition = self.registry[query.statistic_key]
        getter = definition.query_getter

        while isinstance(getter, partial):
            getter = getter.func

        return (
            getattr(getter, "supports_requested_pairs", False)
            and getattr(getter, "supports_value_selection", False)
            and definition.relationship_axes is not None
            and all(
                spec.method in ("keep", "single")
                for spec in query.reduction_dict().values()
            )
        )


def _compact_statistic_values(stat):
    if stat is None:
        return

    values = np.asarray(stat.values)
    flat = values.reshape(-1)
    peak = 0.0

    # Inspect in blocks, avoiding a full-size temporary array.
    for start in range(0, flat.size, 65_536):
        block = flat[start : start + 65_536]
        finite = block[np.isfinite(block)]

        if finite.size:
            peak = max(peak, float(np.max(np.abs(finite))))

    if peak <= np.finfo(np.float16).max:
        dtype = np.float16
    elif peak <= np.finfo(np.float32).max:
        dtype = np.float32
    else:
        dtype = np.float64

    stat.values = values.astype(dtype, copy=False)
