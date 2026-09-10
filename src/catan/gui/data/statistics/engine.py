from PySide6.QtCore import QObject, Qt, Signal

from typing import Optional
from collections.abc import Callable

from dataclasses import dataclass
import numpy as np

from .table import PickTable
from .types import StatisticArray
from .queries import ReductionSpec, StatisticQuery


class StatisticEngine(QObject):

    registry_changed = Signal()
    values_changed = Signal()

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

        self.refresh_registry()

        self.state.data_changed.connect(self._on_data_changed)
        self.state.statistics_sources_changed.connect(self.refresh_registry)

    def refresh_registry(self):
        self.registry = self.registry_factory(
            self.data,
            self.state,
        )

        self.clear_cache()
        self.registry_changed.emit()

    def _on_data_changed(self, _change):
        # Values changed, but the catalogue of available
        # statistics did not necessarily change.
        self.clear_cache()
        self.values_changed.emit()

    # def _on_data_changed(self, input):
    #     self.refresh_registry()

    def data_version(self):
        # Increase/change this whenever tracking/data/statistics change.
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
            filters=query.filters,
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
