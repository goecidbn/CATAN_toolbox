"""Pair statistics of two footprints (or neurons) (Footprints context menu).

What is compared depends on the two sides:

- two footprints: the footprints themselves (distance, similarity, the
  overlap variants the presets use, temporal correlation within one
  session), each footprint against the other's neuron (footprint → union
  statistics in the footprint's session) and, for different neurons, the
  neurons;
- a footprint and a neuron: the footprint against the neuron's union and,
  where the neuron has a footprint in that session, that footprint against
  the first footprint's union; then the neurons;
- two neurons: their unions (distance, similarity), co-occurrence and seed
  overlap.

Each row has two columns, the second with the roles swapped (directional
statistics differ). Statistics that normally score only nearby pairs
(similarities, overlaps) are calculated for the requested pair directly,
without the distance limit (``neighborhood_thr``). Rows without a value in
either column are not listed. Any other pair statistic can be added as a
row (free statistic chooser).
"""

from dataclasses import dataclass, field, replace

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from catan.core.structures import NeuronComponent
from catan.gui.data.statistics.queries import ReductionSpec, StatisticQuery

PAIR_DIMS = {"neuron_i", "neuron_j"}

# Overlap variants (parameters) used by curator presets.
OVERLAP_VARIANTS = (
    ("IoU", {"normalize_by": "IoU"}),
    ("i inside j", {"normalize_by": "i"}),
    ("j covered by i", {"normalize_by": "j"}),
)

FOOTPRINTS_GROUP = "Footprints"
FOOTPRINT_NEURON_GROUP = "Footprint ↔ other neuron"
NEURONS_GROUP = "Neurons (unions)"
ADDED_GROUP = "Added"


def component_label(component):
    if component.session_id is None:
        return f"neuron {component.neuron_id}"
    return f"N{component.neuron_id} S{component.session_id}"


def as_compared(data, component):
    """A footprint only if the neuron has one in that session; else the
    neuron (e.g. an overview selection in a session without it)."""
    if component.session_id is None:
        return component
    ids = data.assignments.ids
    neuron, session = int(component.neuron_id), int(component.session_id)
    if 0 <= neuron < ids.shape[0] and 0 <= session < ids.shape[1] and ids[neuron, session] >= 0:
        return NeuronComponent(neuron, session)
    return NeuronComponent(neuron, None)


@dataclass
class PairRow:
    """One table row: ``key`` with ``parameters`` for (i, j) = ``forward``
    and, in the second column, ``backward`` (None: not applicable).
    ``combine`` reduces several values (e.g. over sessions)."""

    group: str
    label: str
    key: str
    parameters: dict = field(default_factory=dict)
    forward: tuple | None = None
    backward: tuple | None = None
    combine: str = "mean"
    # j is a neuron's union (in i's session), not a footprint.
    j_union: bool = False


def _overlap_rows(group, key, forward, backward):
    return [
        PairRow(group, f"Overlap: {label}", key, dict(variant), forward, backward)
        for label, variant in OVERLAP_VARIANTS
    ]


def _footprint_union_rows(group, forward, backward, same_neuron):
    rows = _footprint_union_list(group, forward, backward, same_neuron)
    for row in rows:
        row.j_union = True
    return rows


def _footprint_union_list(group, forward, backward, same_neuron):
    rows = [
        PairRow(group, "Distance (footprint → union)", "footprint_union_distance", {}, forward, backward),
        PairRow(group, "Similarity (footprint → union)", "footprint_union_similarity", {}, forward, backward),
        *_overlap_rows(group, "footprint_union_overlap", forward, backward),
    ]
    if same_neuron:
        rows.append(PairRow(group, "Match probability", "match_probability", {}, forward, backward))
    else:
        rows.append(
            PairRow(group, "Partner's own match", "partner_session_match", {}, forward, backward)
        )
    return rows


def _neuron_rows(a, b):
    forward = (NeuronComponent(a.neuron_id, None), NeuronComponent(b.neuron_id, None))
    backward = forward[::-1]
    return [
        PairRow(NEURONS_GROUP, "Union-centroid distance", "union_distances", {}, forward, backward),
        PairRow(NEURONS_GROUP, "Union similarity", "union_similarity", {}, forward, backward),
        PairRow(NEURONS_GROUP, "Co-occurrence (sessions with both)", "cooccurrence", {}, forward, None, "sum"),
        PairRow(NEURONS_GROUP, "Seed overlap (largest over sessions)", "seed_overlap", {}, forward, backward, "max"),
    ]


def comparison_rows(a, b):
    """The rows for sides ``a`` and ``b`` (see the module docstring)."""
    rows = []
    same_neuron = a.neuron_id == b.neuron_id
    a_footprint = a.session_id is not None
    b_footprint = b.session_id is not None

    if a_footprint and b_footprint:
        pair, swapped = (a, b), (b, a)
        if a.session_id == b.session_id:
            rows.append(PairRow(FOOTPRINTS_GROUP, "Distance (same session)", "distances", {}, pair, swapped))
        else:
            rows.append(PairRow(FOOTPRINTS_GROUP, "Distance", "cross_session_distances", {}, pair, swapped))
        rows.append(PairRow(FOOTPRINTS_GROUP, "Similarity", "footprint_similarity", {}, pair, swapped))
        rows += _overlap_rows(FOOTPRINTS_GROUP, "footprint_overlap", pair, swapped)
        if a.session_id == b.session_id:
            rows.append(PairRow(FOOTPRINTS_GROUP, "Temporal correlation", "temporal_corr", {}, pair, swapped))

        # Each footprint against the other's union (its own session).
        group = "Footprint ↔ own neuron" if same_neuron else FOOTPRINT_NEURON_GROUP
        rows += _footprint_union_rows(
            group,
            (a, NeuronComponent(b.neuron_id, a.session_id)),
            (b, NeuronComponent(a.neuron_id, b.session_id)),
            same_neuron,
        )
    elif a_footprint or b_footprint:
        footprint, neuron = (a, b) if a_footprint else (b, a)
        session = footprint.session_id
        group = "Footprint ↔ own neuron" if same_neuron else "Footprint ↔ neuron"
        # Second column: the neuron's footprint in that session against the
        # first footprint's union (none for the own neuron: same rows).
        backward = None if same_neuron else (
            NeuronComponent(neuron.neuron_id, session),
            NeuronComponent(footprint.neuron_id, session),
        )
        rows += _footprint_union_rows(
            group,
            (footprint, NeuronComponent(neuron.neuron_id, session)),
            backward,
            same_neuron,
        )

    if not same_neuron:
        rows += _neuron_rows(a, b)
    return rows


def _query(engine, key, parameters):
    definition = engine.registry[key]
    reductions = tuple(sorted((definition.default_reductions or {}).items()))
    values = definition.resolve_parameters(dict(parameters))
    return StatisticQuery(key, reductions, parameters=tuple(sorted(values.items())))


def _footprint_present(data, component):
    if component.session_id is None:
        return False
    ids = data.assignments.ids
    footprint = int(ids[component.neuron_id, component.session_id])
    if footprint < 0:
        return False
    included = getattr(data.sessions[component.session_id], "included", None)
    return included is None or bool(included[footprint])


def _direct_footprint_overlap(data, first, second, normalize_by, support_threshold):
    """Overlap of two footprints (any distance); None without both."""
    from catan.gui.data.statistics.overlap_statistics import overlap_denominator

    if not (_footprint_present(data, first) and _footprint_present(data, second)):
        return None
    supports = []
    for component in (first, second):
        session = data.sessions[component.session_id]
        column = session.footprints[:, int(data.assignments.ids[component.neuron_id, component.session_id])]
        column = np.asarray(column.todense()).ravel()
        peak = column.max()
        supports.append(column >= support_threshold * peak if peak > 0 else column > 0)
    if supports[0].shape != supports[1].shape:
        return None
    intersection = np.array([np.count_nonzero(supports[0] & supports[1])], dtype=np.float64)
    denominator = overlap_denominator(
        normalize_by,
        intersection,
        np.array([np.count_nonzero(supports[0])], dtype=np.float64),
        np.array([np.count_nonzero(supports[1])], dtype=np.float64),
    )
    if denominator[0] <= 0:
        return np.empty(0)
    return np.array([min(1.0, intersection[0] / denominator[0])])


def direct_values(engine, key, parameters, first, second):
    """Values for the pair itself, without the distance limit of
    statistics that score only nearby pairs (None: not one of them)."""
    from catan.gui.data.statistics.engine import active_session_view
    from catan.gui.data.statistics.footprint_pairs import footprint_similarity_for_pairs
    from catan.gui.data.statistics.seed_overlap import seed_overlap_values
    from catan.gui.data.statistics.union_overlap import footprint_union_overlap_values
    from catan.gui.data.statistics.union_similarity import (
        partner_session_match_values,
        union_similarity_values,
        union_union_similarity_values,
    )

    footprints = first.session_id is not None and second.session_id is not None
    ni, nj = [first.neuron_id], [second.neuron_id]
    gamma = parameters.get("gamma", 0.1)
    data, state = active_session_view(engine.data, engine.state)

    if key == "footprint_similarity" and footprints:
        values = footprint_similarity_for_pairs(
            data,
            state,
            [[first.neuron_id, second.neuron_id, first.session_id, second.session_id]],
            neighborhood_thr=np.inf,
            gamma=gamma,
            canonical=False,
        )
    elif key == "footprint_overlap" and footprints:
        values = _direct_footprint_overlap(
            data,
            first,
            second,
            parameters.get("normalize_by", "smaller"),
            parameters.get("support_threshold", 0.1),
        )
        if values is None:
            return np.empty(0)
    elif key == "footprint_union_similarity" and first.session_id is not None:
        values = union_similarity_values(
            data, state, ni, nj, [first.session_id], neighborhood_thr=np.inf, gamma=gamma
        )
    elif key == "footprint_union_overlap" and first.session_id is not None:
        values = footprint_union_overlap_values(
            data, state, ni, nj, [first.session_id],
            normalize_by=parameters.get("normalize_by", "smaller"),
            support_threshold=parameters.get("support_threshold", 0.1),
        )
    elif key == "partner_session_match" and first.session_id is not None:
        values = partner_session_match_values(
            data, state, ni, nj, [first.session_id], gamma=gamma
        )
    elif key == "union_similarity":
        values = union_union_similarity_values(
            data, state, ni, nj, neighborhood_thr=np.inf, gamma=gamma
        )
    elif key == "seed_overlap":
        sessions = np.arange(np.asarray(state.assignments).shape[1])
        values = seed_overlap_values(
            data,
            state,
            np.full(len(sessions), first.neuron_id),
            np.full(len(sessions), second.neuron_id),
            sessions,
            normalize_by=parameters.get("normalize_by", "smaller"),
            support_threshold=parameters.get("support_threshold", 0.1),
            seeds=parameters.get("seeds", "all"),
        )
    else:
        return None

    values = np.asarray(values, dtype=np.float64).ravel()
    return values[np.isfinite(values)]


def engine_values(engine, query, first, second):
    """Values of ``query`` with neuron_i / neuron_j (and their sessions)
    fixed to the pair; dimensions left free keep the query's reduction
    (e.g. co-occurrence summed over sessions)."""
    definition = engine.registry.get(query.statistic_key)
    if definition is None or not PAIR_DIMS <= set(definition.dims):
        return np.empty(0)

    dims = set(definition.dims)
    fixed = {"neuron_i": first.neuron_id, "neuron_j": second.neuron_id}
    if {"session_i", "session_j"} <= dims:
        if None in (first.session_id, second.session_id):
            return np.empty(0)
        fixed.update(session_i=first.session_id, session_j=second.session_id)
    elif "session" in dims and first.session_id is not None:
        fixed["session"] = first.session_id

    reductions = query.reduction_dict()
    reductions.update(
        {dim: ReductionSpec("single", index=int(index)) for dim, index in fixed.items()}
    )
    single = replace(query, reductions=tuple(sorted(reductions.items())), filters=())
    table = engine.evaluate_table(single, cache=False)
    if table is None:
        return np.empty(0)
    values = np.asarray(table.values, dtype=np.float64).ravel()
    return values[np.isfinite(values)]


def pair_values(engine, query, first, second):
    values = direct_values(engine, query.statistic_key, dict(query.parameters), first, second)
    if values is None:
        values = engine_values(engine, query, first, second)
    return values


def format_values(values, combine="mean"):
    if values is None or values.size == 0:
        return "–"
    if values.size == 1:
        return f"{values[0]:.4g}"
    if combine == "max":
        return f"{np.max(values):.4g}"
    if combine == "sum":
        return f"{np.sum(values):.4g}"
    return f"mean {np.mean(values):.4g} ({values.size} values)"


def evaluate_row(engine, row, sides):
    """Text of ``row`` for ``sides`` (an (i, j) pair or None)."""
    if sides is None:
        return ""
    try:
        query = _query(engine, row.key, row.parameters)
        return format_values(pair_values(engine, query, *sides), row.combine)
    except Exception as exc:
        return f"error: {exc}"


class PairStatisticsDialog(QDialog):

    def __init__(self, state, data, first, second, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.state = state
        self.data = data
        self.first = as_compared(data, first)
        self.second = as_compared(data, second)
        self._closed = False
        self._task_ids = set()
        self._added = []
        self._groups = {}

        a, b = component_label(self.first), component_label(self.second)
        self.setWindowTitle(f"Compare {a} and {b}")

        layout = QVBoxLayout(self)
        header = QLabel(
            f"<b>A = {a}</b>, <b>B = {b}</b>. The second column swaps the "
            "roles (i ↔ j). Footprint ↔ neuron rows compare a footprint "
            "with a neuron's union in the footprint's session. Calculated "
            "for this pair at any distance; rows without a value are not "
            "listed, – : no value.",
            self,
        )
        header.setWordWrap(True)
        layout.addWidget(header)
        self.header = header

        self.tree = QTreeWidget(self)
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Statistic", "A → B", "B → A"])
        self.tree.setUniformRowHeights(True)
        self.tree.setRootIsDecorated(False)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        head = self.tree.header()
        head.setStretchLastSection(False)
        head.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        head.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        head.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.tree, 1)

        # Free choice of statistic and parameters, added as a row.
        from catan.gui.panels.StatisticsData import StatisticQuerySelector

        custom = QHBoxLayout()
        custom.addWidget(QLabel("Other statistic:"))
        self.selector = StatisticQuerySelector(self.data.statistic_engine, axis="x", parent=self)
        self.selector.set_query_mode("generic")
        if "footprint_overlap" in self.data.statistic_engine.registry:
            self.selector.set_query(
                _query(self.data.statistic_engine, "footprint_overlap", {}), emit=False
            )
        custom.addWidget(self.selector, 1)
        self.add_button = QPushButton("Add row", self)
        self.add_button.setToolTip(
            "Add the chosen pair statistic (with its parameters) to the table"
        )
        self.add_button.clicked.connect(self._add_custom_row)
        custom.addWidget(self.add_button)
        layout.addLayout(custom)

        self.custom_label = QLabel(self)
        self.custom_label.setStyleSheet("color: #9aa4b2;")
        self.custom_label.hide()
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

    def _rows(self):
        return comparison_rows(self.first, self.second) + list(self._added)

    def _sides_text(self, row, sides):
        if sides is None:
            return ""
        first, second = sides
        target = (
            f"union {second.neuron_id} (S{second.session_id})"
            if row.j_union
            else component_label(second)
        )
        return f"{component_label(first)} → {target}"

    def calculate(self):
        engine = self.data.statistic_engine
        rows = self._rows()

        self.tree.clear()
        self._groups = {}
        self.tree.addTopLevelItem(QTreeWidgetItem(["Calculating …", "", ""]))
        self._fit_to_content()

        def work():
            return [
                (row, evaluate_row(engine, row, row.forward), evaluate_row(engine, row, row.backward))
                for row in rows
            ]

        self._start(
            f"Compare {component_label(self.first)} and {component_label(self.second)}",
            work,
            self._show_rows,
        )

    def _group_item(self, row):
        item = self._groups.get(row.group)
        if item is None:
            item = QTreeWidgetItem(
                [row.group, self._sides_text(row, row.forward), self._sides_text(row, row.backward)]
            )
            font = item.font(0)
            font.setBold(True)
            item.setFont(0, font)
            for column in (1, 2):
                item.setForeground(column, Qt.GlobalColor.gray)
            item.setFirstColumnSpanned(False)
            self.tree.addTopLevelItem(item)
            self._groups[row.group] = item
        return item

    def _value_text(self, row, text):
        unit = self._units(row.key)
        return text + (unit if text[:1].isdigit() or text[:1] == "-" else "")

    def _show_rows(self, rows):
        self.tree.clear()
        self._groups = {}
        shown = 0
        for row, forward, backward in rows:
            # Added rows stay (feedback); others only with a value.
            if forward in ("–", "") and backward in ("–", "") and row.group != ADDED_GROUP:
                continue
            group = self._group_item(row)
            parameters = ", ".join(
                f"{k} = {v}"
                for k, v in sorted(row.parameters.items())
                if row.group == ADDED_GROUP
            )
            child = QTreeWidgetItem(
                [
                    row.label + (f" ({parameters})" if parameters else ""),
                    self._value_text(row, forward),
                    self._value_text(row, backward),
                ]
            )
            child.setToolTip(0, row.key)
            group.addChild(child)
            shown += 1
        if not shown:
            self.tree.addTopLevelItem(QTreeWidgetItem(["No statistic has a value for this pair.", "", ""]))
        self.tree.expandAll()
        # No row with swapped roles (e.g. a footprint and its own neuron).
        self.tree.setColumnHidden(2, all(row.backward is None for row, *_ in rows))
        self._fit_to_content()

    def _add_custom_row(self):
        query = self.selector.current_query()
        definition = (
            None if query is None else self.data.statistic_engine.registry.get(query.statistic_key)
        )
        if definition is None or not PAIR_DIMS <= set(definition.dims):
            self.custom_label.setText("Choose a pair statistic (neuron i / neuron j).")
            self.custom_label.show()
            return
        self.custom_label.hide()
        self._added.append(
            PairRow(
                ADDED_GROUP,
                definition.title,
                query.statistic_key,
                dict(query.parameters),
                (self.first, self.second),
                (self.second, self.first),
            )
        )
        self.calculate()

    def _fit_to_content(self):
        """Size the window to the table (no empty space), within the
        screen."""
        tree = self.tree
        tree.resizeColumnToContents(0)
        width = sum(tree.sizeHintForColumn(c) for c in range(3)) + 48
        width = max(width, tree.header().length() // 2, 360)
        rows = 0
        for i in range(tree.topLevelItemCount()):
            rows += 1 + tree.topLevelItem(i).childCount()
        row_height = max(tree.sizeHintForRow(0), tree.fontMetrics().height() + 4)
        height = tree.header().sizeHint().height() + rows * row_height + 8
        screen = (self.screen() or QApplication.primaryScreen()).availableGeometry()
        width = min(max(width, self.selector.sizeHint().width() + 160), int(screen.width() * 0.8))
        # Sized once to the content; the user can still resize freely.
        tree.setMinimumHeight(min(height, int(screen.height() * 0.6)))
        self.header.setMaximumWidth(width)
        self.setMinimumWidth(0)
        self.adjustSize()
        self.resize(width, self.sizeHint().height())
        tree.setMinimumHeight(60)

    def closeEvent(self, event):
        self._closed = True
        for task_id in list(self._task_ids):
            if task_id is not None:
                self.state.tasks.cancel(task_id)
        self.selector.dispose()
        super().closeEvent(event)
