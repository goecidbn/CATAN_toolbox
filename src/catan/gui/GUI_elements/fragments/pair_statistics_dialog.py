"""Pair statistics of two footprints (or neurons) (Footprints context menu).

Every pair statistic of the registry with its default parameters, for both
orientations (i = A, j = B and i = B, j = A; directional statistics differ),
plus the footprint overlap variants the presets use and a free statistic
chooser (any pair statistic, parameters as in the statistics panels).
Statistics comparing a footprint with a union or within one session use
the session of the i side. Similarities and footprint/union overlaps are
calculated for the pair directly: the statistics themselves only score
pairs closer than 10 px (``neighborhood_thr``); closer pairs give the same
values. Statistics without a value in either direction are not listed.
"""

from dataclasses import replace

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from catan.gui.data.curation_candidates import evaluate_at
from catan.gui.data.statistics.queries import StatisticQuery

PAIR_DIMS = {"neuron_i", "neuron_j"}

# Overlap variants (parameters) used by curator presets.
OVERLAP_VARIANTS = (
    ("intersection over union (containment off)", {"containment": False, "directional": False, "normalize_by_target": False}),
    ("i inside j (directional containment)", {"containment": True, "directional": True, "normalize_by_target": False}),
    ("j covered by i (normalized by target)", {"containment": True, "directional": True, "normalize_by_target": True}),
)


def component_label(component):
    if component.session_id is None:
        return f"neuron {component.neuron_id}"
    return f"N{component.neuron_id} S{component.session_id}"


def pair_queries(engine, footprints, same_neuron=False):
    """(label, query) of every applicable pair statistic."""
    queries = []

    for key, definition in engine.registry.items():
        dims = set(definition.dims)
        if not PAIR_DIMS <= dims:
            continue
        # Neurons without a session: only statistics of neuron pairs.
        if not footprints and dims != PAIR_DIMS:
            continue
        # Union with itself.
        if same_neuron and dims == PAIR_DIMS:
            continue

        reductions = tuple(sorted((definition.default_reductions or {}).items()))
        parameters = definition.resolve_parameters({})
        query = StatisticQuery(key, reductions, parameters=tuple(sorted(parameters.items())))
        queries.append((definition.title, query))

        if key in ("footprint_overlap", "footprint_union_overlap"):
            for label, variant in OVERLAP_VARIANTS:
                values = definition.resolve_parameters({**parameters, **variant})
                queries.append(
                    (
                        f"{definition.title}: {label}",
                        replace(query, parameters=tuple(sorted(values.items()))),
                    )
                )

    return queries


def direct_values(engine, query, first, second):
    """Values of statistics limited to nearby pairs, calculated for the
    pair itself (None: not one of them)."""
    from catan.gui.data.statistics.engine import active_session_view
    from catan.gui.data.statistics.footprint_pairs import footprint_similarity_for_pairs
    from catan.gui.data.statistics.union_overlap import footprint_union_overlap_values
    from catan.gui.data.statistics.union_similarity import (
        union_similarity_values,
        union_union_similarity_values,
    )

    key = query.statistic_key
    parameters = dict(query.parameters)
    footprints = first.session_id is not None and second.session_id is not None
    ni, nj = [first.neuron_id], [second.neuron_id]

    if key in ("footprint_similarity", "footprint_union_similarity", "footprint_union_overlap") and not footprints:
        return None
    if key not in (
        "footprint_similarity",
        "footprint_union_similarity",
        "footprint_union_overlap",
        "union_similarity",
    ):
        return None

    data, state = active_session_view(engine.data, engine.state)

    if key == "footprint_similarity":
        values = footprint_similarity_for_pairs(
            data,
            state,
            [[first.neuron_id, second.neuron_id, first.session_id, second.session_id]],
            neighborhood_thr=np.inf,
            gamma=parameters.get("gamma", 0.1),
            canonical=False,
        )
    elif key == "footprint_union_similarity":
        values = union_similarity_values(
            data, state, ni, nj, [first.session_id],
            neighborhood_thr=np.inf, gamma=parameters.get("gamma", 0.1),
        )
    elif key == "footprint_union_overlap":
        values = footprint_union_overlap_values(
            data, state, ni, nj, [first.session_id], **parameters
        )
    else:
        values = union_union_similarity_values(
            data, state, ni, nj, neighborhood_thr=np.inf,
            gamma=parameters.get("gamma", 0.1),
        )

    values = np.asarray(values, dtype=np.float64).ravel()
    return values[np.isfinite(values)]


def evaluate_pair(engine, query, first, second):
    """Value text of ``query`` with i = ``first``, j = ``second``."""
    try:
        values = direct_values(engine, query, first, second)
        if values is None:
            values = evaluate_at(engine, query, (first, second))
    except Exception as exc:
        return f"error: {exc}"

    if values.size == 0:
        return "–"
    if values.size == 1:
        return f"{values[0]:.4g}"
    return f"mean {np.mean(values):.4g} ({values.size} values)"


class PairStatisticsDialog(QDialog):

    def __init__(self, state, data, first, second, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.state = state
        self.data = data
        self.first = first
        self.second = second
        self._closed = False
        self._task_ids = set()

        a, b = component_label(first), component_label(second)
        self.setWindowTitle(f"Compare {a} and {b}")
        self.resize(720, 560)

        layout = QVBoxLayout(self)
        header = QLabel(
            f"<b>A = {a}</b>, <b>B = {b}</b>. Directional statistics differ "
            "between the columns; footprint ↔ union and within-session "
            "statistics use the session of the i side. Statistics without a "
            "value (e.g. traces not loaded, or no footprint of the neuron in "
            "that session) are not listed; – : no value in this direction.",
            self,
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        self.tree = QTreeWidget(self)
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Statistic", "i = A, j = B", "i = B, j = A"])
        self.tree.setUniformRowHeights(True)
        head = self.tree.header()
        head.setStretchLastSection(False)
        head.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        head.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        head.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.tree, 1)

        # Free choice of statistic and parameters.
        from catan.gui.panels.StatisticsData import StatisticQuerySelector

        custom = QHBoxLayout()
        custom.addWidget(QLabel("Other statistic / parameters:"))
        self.selector = StatisticQuerySelector(self.data.statistic_engine, axis="x", parent=self)
        self.selector.set_query_mode("generic")
        if "footprint_overlap" in self.data.statistic_engine.registry:
            definition = self.data.statistic_engine.registry["footprint_overlap"]
            self.selector.set_query(
                StatisticQuery(
                    "footprint_overlap",
                    tuple(sorted((definition.default_reductions or {}).items())),
                    parameters=tuple(sorted(definition.resolve_parameters({}).items())),
                ),
                emit=False,
            )
        custom.addWidget(self.selector, 1)
        layout.addLayout(custom)

        self.custom_label = QLabel(self)
        self.custom_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.custom_label)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        refresh = QPushButton("Recalculate", self)
        refresh.clicked.connect(self.calculate)
        buttons.addWidget(refresh)
        close = QPushButton("Close", self)
        close.clicked.connect(self.close)
        buttons.addWidget(close)
        layout.addLayout(buttons)

        self.selector.queryChanged.connect(lambda *_: self._calculate_custom())

        self.calculate()

    # ------------------------------------------------------------------

    def _start(self, name, fn, accept):
        task = {}

        def on_result(result):
            self._task_ids.discard(task.get("id"))
            if not self._closed:
                accept(result)

        task["id"] = self.state.tasks.start("calculating", name, fn, on_result=on_result)
        self._task_ids.add(task["id"])

    def _units(self, key):
        definition = self.data.statistic_engine.registry.get(key)
        label = "" if definition is None else self.data.units.label(definition.value_kind)
        return f" {label}" if label else ""

    def calculate(self):
        engine = self.data.statistic_engine
        first, second = self.first, self.second
        footprints = first.session_id is not None and second.session_id is not None

        self.tree.clear()
        self.tree.addTopLevelItem(QTreeWidgetItem(["Calculating …", "", ""]))

        def work():
            rows = []
            same_neuron = first.neuron_id == second.neuron_id
            for label, query in pair_queries(engine, footprints, same_neuron):
                rows.append(
                    (
                        label,
                        query.statistic_key,
                        evaluate_pair(engine, query, first, second),
                        evaluate_pair(engine, query, second, first),
                    )
                )
            return rows

        def accept(rows):
            self.tree.clear()
            for label, key, ab, ba in rows:
                if ab == "–" and ba == "–":
                    continue
                unit = self._units(key)
                item = QTreeWidgetItem(
                    [
                        label,
                        ab + (unit if ab[:1].isdigit() else ""),
                        ba + (unit if ba[:1].isdigit() else ""),
                    ]
                )
                item.setToolTip(0, key)
                self.tree.addTopLevelItem(item)

        self._start(
            f"Compare {component_label(first)} and {component_label(second)}",
            work,
            accept,
        )
        self._calculate_custom()

    def _calculate_custom(self):
        query = self.selector.current_query()
        if query is None:
            self.custom_label.setText("Choose a pair statistic above.")
            return

        definition = self.data.statistic_engine.registry.get(query.statistic_key)
        if definition is None or not PAIR_DIMS <= set(definition.dims):
            self.custom_label.setText("Choose a pair statistic (neuron i / neuron j).")
            return

        engine = self.data.statistic_engine
        first, second = self.first, self.second
        self.custom_label.setText("Calculating …")

        def work():
            return (
                evaluate_pair(engine, query, first, second),
                evaluate_pair(engine, query, second, first),
            )

        def accept(values):
            unit = self._units(query.statistic_key)
            parameters = ", ".join(f"{k} = {v}" for k, v in query.parameters)
            self.custom_label.setText(
                f"{definition.title}"
                + (f" ({parameters})" if parameters else "")
                + f":  i = A, j = B: {values[0]}{unit}   ·   i = B, j = A: {values[1]}{unit}"
            )

        self._start("Compare footprints (custom statistic)", work, accept)

    def closeEvent(self, event):
        self._closed = True
        for task_id in list(self._task_ids):
            if task_id is not None:
                self.state.tasks.cancel(task_id)
        self.selector.dispose()
        super().closeEvent(event)
