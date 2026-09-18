from PySide6.QtCore import QObject, Signal, QSettings
import json
from catan.gui.data.statistics.queries import StatisticQuery
from catan.gui.data.statistics.engine import StatisticEngine


class StatisticDisplayConfig(QObject):

    changed = Signal(str, str)  # entity_mode, row_mode

    def __init__(self, *, settings: QSettings, engine: StatisticEngine):
        super().__init__()

        self.settings = settings
        self.engine = engine

        self._queries_by_mode: dict[
            tuple[str, str],
            list[StatisticQuery],
        ] = {}

        self._loaded_defaults_initialized = False

        self._load()

        self.engine.registry_changed.connect(self._on_registry_changed)

    def _on_registry_changed(self):

        has_loaded_statistics = any(
            definition.category == "session_loaded"
            for definition in self.engine.registry.values()
        )

        if not has_loaded_statistics:
            self._loaded_defaults_initialized = False

        self._initialize_loaded_defaults()
        self._remove_invalid_queries()

    def queries(
        self,
        entity_mode: str,
        row_mode: str,
    ) -> list[StatisticQuery]:

        return list(
            self._queries_by_mode.get(
                (entity_mode, row_mode),
                [],
            )
        )

    def set_queries(
        self,
        entity_mode: str,
        row_mode: str,
        queries: list[StatisticQuery],
    ):

        mode = (entity_mode, row_mode)

        self._queries_by_mode[mode] = list(queries)

        self._save_mode(mode)

        self.changed.emit(
            entity_mode,
            row_mode,
        )

    def _save_mode(
        self,
        mode: tuple[str, str],
    ):

        queries = self._queries_by_mode.get(
            mode,
            [],
        )

        registry = self.engine.registry

        persistent_queries = []

        for query in queries:

            stat_def = registry.get(query.statistic_key)

            # Dynamically loaded quality fields are NOT persisted.
            if stat_def is not None and stat_def.category == "session_loaded":
                continue

            persistent_queries.append(query.to_dict())

        key = self._settings_key(mode)

        self.settings.setValue(
            key,
            json.dumps(persistent_queries),
        )

    def _settings_key(
        self,
        mode: tuple[str, str],
    ) -> str:

        entity_mode, row_mode = mode

        return "StatisticDisplay/" f"{entity_mode}_{row_mode}"

    def _load(self):

        for entity_mode in ("footprint", "neuron"):
            for row_mode in ("single", "pair"):

                mode = (entity_mode, row_mode)

                raw = self.settings.value(self._settings_key(mode), "")

                if not raw:
                    self._queries_by_mode[mode] = []
                    continue

                try:
                    data = json.loads(raw)

                    self._queries_by_mode[mode] = [
                        StatisticQuery.from_dict(item) for item in data
                    ]

                except (ValueError, TypeError, KeyError):
                    self._queries_by_mode[mode] = []

    def _initialize_loaded_defaults(self):

        if self._loaded_defaults_initialized:
            return

        registry = self.engine.registry

        defaults = [
            StatisticQuery(statistic_key=key, reductions=())
            for key, definition in registry.items()
            if definition.category == "session_loaded"
        ]

        if not defaults:
            return

        mode = ("footprint", "single")

        saved = self._queries_by_mode.get(mode, [])

        default_keys = {query.statistic_key for query in defaults}

        self._queries_by_mode[mode] = [
            *defaults,
            *[query for query in saved if query.statistic_key not in default_keys],
        ]

        self._loaded_defaults_initialized = True

        self.changed.emit(*mode)

    def _remove_invalid_queries(self):

        registry = self.engine.registry

        for mode, queries in list(self._queries_by_mode.items()):

            valid_queries = [
                query for query in queries if query.statistic_key in registry
            ]

            if len(valid_queries) == len(queries):
                continue

            self._queries_by_mode[mode] = valid_queries

            self._save_mode(mode)

            self.changed.emit(*mode)
