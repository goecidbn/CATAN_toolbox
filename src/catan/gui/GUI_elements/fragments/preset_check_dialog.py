"""Check one neuron against a curator preset (Footprints context menu).

Shows the preset tree with the neuron's verdict per group and condition,
and per condition the statistic values of the neuron's rows against the
resolved threshold (gui/data/curation_check.py). Selecting a row
highlights its footprints.
"""

from copy import deepcopy

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTabWidget,
    QTreeWidget,
    QWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from catan.gui.data.curation_filter_presets import (
    PRESET_CATEGORIES,
    UNCATEGORIZED_LABEL,
    shared_preset_store,
)

WORKING_FILTER_KEY = "__working__"

VERDICTS = {
    True: ("✓", "#6cc070"),
    False: ("✗", "#e06c6c"),
    None: ("–", "#9a9a9a"),
}

OPERATOR_LABELS = {"and": "All of", "or": "Any of"}

LEVEL_LABELS = {
    "neuron": "neurons",
    "footprint": "footprints",
    "relationship": "source → target pairs",
    "reassignment": "reassignments",
    "neuron_pair": "neuron pairs",
}

BINDING_LABELS = {
    "source": "for the sources",
    "target": "for the targets",
    "between_sources": "between the sources",
    "between_targets": "between the targets",
    "between_sources_targets": "between sources and targets",
}

COMPARISON_SYMBOLS = {"eq": "=", "ge": "≥", "le": "≤"}


def describe_group(group):
    text = f"{OPERATOR_LABELS.get(group.operator, group.operator)} " + (
        f"({LEVEL_LABELS.get(group.match_level, group.match_level)})"
    )

    if group.apply_to != "same":
        text += f", {BINDING_LABELS.get(group.apply_to, group.apply_to)}"
        if group.summary is not None:
            text += f" ({group.summary})"

    if group.count is not None:
        count = group.count
        text += (
            f", {COMPARISON_SYMBOLS[count.comparison]} {count.value} "
            f"{count.unit}{'s' if count.value != 1 else ''}"
        )

    if group.manipulation is not None:
        text += f" → {group.manipulation.kind.replace('_', ' ')}"

    return text


def describe_row(check, index, component=None):
    components = sorted(
        check.row_components(index), key=lambda c: (c.neuron_id, c.session_id)
    )

    if components:
        text = " ↔ ".join(f"N{c.neuron_id} S{c.session_id}" for c in components)
    else:
        text = ", ".join(
            f"{name} {int(values[index])}" for name, values in sorted(check.columns.items())
        )

    if component is not None and component in components:
        text = "◆ " + text

    return text, components


def _set_verdict(item, column, verdict, text=None):
    symbol, color = VERDICTS[verdict]
    item.setText(column, symbol if text is None else f"{symbol} {text}")
    item.setForeground(column, QBrush(QColor(color)))


class PresetCheckDialog(QDialog):

    def __init__(self, state, data, neuron_id, component=None, preset_key=None, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(f"Check neuron {neuron_id} against a preset")
        self.resize(860, 640)

        self.state = state
        self.data = data
        self.neuron_id = int(neuron_id)
        self.component = component
        self.store = shared_preset_store(state)

        self._token = None
        self._task_id = None
        self._closed = False

        layout = QVBoxLayout(self)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset"))
        self.preset_combo = QComboBox(self)
        self._fill_presets(preset_key)
        preset_row.addWidget(self.preset_combo, 1)
        self.recheck_button = QPushButton("Check again", self)
        self.recheck_button.setToolTip("Evaluate with the current data, model and preset.")
        preset_row.addWidget(self.recheck_button)
        layout.addLayout(preset_row)

        self.verdict_label = QLabel(self)
        self.verdict_label.setTextFormat(Qt.TextFormat.RichText)
        self.verdict_label.setWordWrap(True)
        layout.addWidget(self.verdict_label)

        self.tabs = QTabWidget(self)
        layout.addWidget(self.tabs, 1)

        self.candidate_tree = self._make_tree(["Candidate / step", "Value", "Result"])
        self.tabs.addTab(
            self._tab(
                "Each candidate follows the preset step by step; the summary "
                "shows where it drops out (✗ at …). Hover for its full path, "
                "expand for the steps — greyed steps come after the first "
                "failure (what would follow). Select a candidate to highlight "
                "its footprints.",
                self.candidate_tree,
            ),
            "Candidates",
        )

        self.tree = self._make_tree(["Preset / rows", "Values", "Result"])
        self.tabs.addTab(
            self._tab(
                "Each condition lists this neuron's values (in source / target "
                "subgroups: its partners') against the threshold; ✗ marks values "
                "outside it, passing rows come first. If the parent group did not "
                "match, its subgroups show the partners of the pairs passing its "
                "conditions (– : not evaluated by the curator). "
                "◆ marks the clicked footprint. Select a row to highlight its "
                "footprints.",
                self.tree,
            ),
            "Preset details",
        )

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_button = QPushButton("Close", self)
        close_button.clicked.connect(self.close)
        close_row.addWidget(close_button)
        layout.addLayout(close_row)

        self.preset_combo.currentIndexChanged.connect(lambda *_: self.run_check())
        self.recheck_button.clicked.connect(self.run_check)
        self.tree.itemSelectionChanged.connect(
            lambda: self._highlight_selected_row(self.tree)
        )
        self.candidate_tree.itemSelectionChanged.connect(
            lambda: self._highlight_selected_row(self.candidate_tree)
        )

        self.run_check()

    def _make_tree(self, labels):
        tree = QTreeWidget(self)
        tree.setColumnCount(3)
        tree.setHeaderLabels(labels)
        tree.setUniformRowHeights(True)
        header = tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        return tree

    def _tab(self, hint_text, tree):
        widget = QWidget(self)
        tab_layout = QVBoxLayout(widget)
        hint = QLabel(hint_text, widget)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(mid);")
        tab_layout.addWidget(hint)
        tab_layout.addWidget(tree, 1)
        return widget

    # ------------------------------------------------------------------

    def _fill_presets(self, selected_key):
        combo = self.preset_combo
        model = combo.model()

        working = "Curator's current filter"
        if self.store.working_dirty:
            working += " (unsaved changes)"
        combo.addItem(working, WORKING_FILTER_KEY)

        category = object()
        for preset in self.store.presets():
            if preset.category != category:
                category = preset.category
                combo.insertSeparator(combo.count())
                combo.addItem(PRESET_CATEGORIES.get(category, UNCATEGORIZED_LABEL), None)
                model.item(combo.count() - 1).setEnabled(False)
            combo.addItem(f"   {preset.name}", preset.key)

        index = combo.findData(selected_key)
        combo.setCurrentIndex(max(index, 0))

    def _preset(self):
        key = self.preset_combo.currentData()

        if key == WORKING_FILTER_KEY:
            return "the curator's current filter", deepcopy(self.store.working_filter)

        info = self.store.preset_info(key)
        return f"“{info.name if info else key}”", self.store.load_preset(key)

    # ------------------------------------------------------------------

    def run_check(self):
        if self.preset_combo.currentData() is None:
            return

        from catan.gui.data.curation_check import check_neuron

        try:
            name, root = self._preset()
        except Exception as exc:
            self._show_message(f"Could not load the preset: {exc}")
            return

        if self._task_id is not None:
            self.state.tasks.cancel(self._task_id)

        token = object()
        self._token = token
        self.tree.clear()
        self.candidate_tree.clear()
        self._show_message(f"Checking neuron {self.neuron_id} against {name} …")

        engine = self.data.statistic_engine
        neuron_id, component = self.neuron_id, self.component

        def check():
            return check_neuron(engine, root, neuron_id, component)

        def accept(result):
            if self._closed or self._token is not token:
                return
            self._task_id = None
            self._show_result(name, result)

        self._task_id = self.state.tasks.start(
            "calculating",
            f"Check neuron {neuron_id} against a preset",
            check,
            on_result=accept,
        )

    def _show_message(self, text):
        self.verdict_label.setText(text)

    def _show_result(self, name, check):
        neuron = f"Neuron {check.neuron_id}"

        if check.error:
            self._show_message(
                f"<span style='color:{VERDICTS[None][1]}'>{neuron}: the preset "
                f"could not be evaluated</span> — {check.error}"
            )
        else:
            symbol, color = VERDICTS[check.passed]
            verb = "matches" if check.passed else "does not match"
            text = f"<b style='color:{color}'>{symbol} {neuron} {verb} {name}.</b>"
            if check.component_passed is not None:
                s_symbol, s_color = VERDICTS[check.component_passed]
                text += (
                    f"<br><span style='color:{s_color}'>{s_symbol} Its footprint in "
                    f"session {check.component.session_id} "
                    f"{'is' if check.component_passed else 'is not'} among the "
                    "matching footprints.</span>"
                )
            self._show_message(text)

        self.tree.clear()
        if check.root is not None:
            item = self._add_group(None, check.root, error_node=check.error_node)
            self.tree.addTopLevelItem(item)
            item.setExpanded(True)

        self.candidate_tree.clear()
        for candidate in check.candidates:
            item = self._add_candidate(candidate)
            self.candidate_tree.addTopLevelItem(item)
            item.setExpanded(candidate.passed is not True or len(check.candidates) == 1)
        if not check.candidates:
            self.tabs.setCurrentIndex(1)

    def _add_candidate(self, candidate, top=True):
        item = QTreeWidgetItem([candidate.label, candidate.summary, ""])
        _set_verdict(item, 2, candidate.passed)
        for column in range(3):
            item.setToolTip(column, candidate.detail)
        item.setData(0, Qt.ItemDataRole.UserRole, list(candidate.components))
        if top and self.component is not None and self.component in candidate.components:
            item.setText(0, "◆ " + candidate.label)

        # Steps, greyed after the first failure.
        downstream = False
        dim = QBrush(QColor(VERDICTS[None][1]))
        for step in candidate.steps:
            step_item = QTreeWidgetItem([step.label, step.value, ""])
            _set_verdict(step_item, 2, step.passed)
            if downstream:
                step_item.setForeground(0, dim)
                step_item.setForeground(1, dim)
                step_item.setToolTip(0, "After the first failure: shown for information.")
            if step.detail:
                step_item.setToolTip(1, step.detail)
            step_item.setData(0, Qt.ItemDataRole.UserRole, list(candidate.components))
            item.addChild(step_item)
            downstream = downstream or step.passed is False

        for child in candidate.children:
            item.addChild(self._add_candidate(child, top=False))

        return item

    def _add_group(self, parent_verdicts, group_check, *, error_node=None):
        group = group_check.group
        values = group_check.count
        if not values and group_check.note.startswith("Partners:"):
            values = group_check.note.splitlines()[0]
        item = QTreeWidgetItem([describe_group(group), values, ""])
        _set_verdict(item, 2, group_check.passed)

        tooltip = [group.comment] if group.comment else []
        if group_check.note:
            tooltip.append(group_check.note)
        if group.id == error_node:
            tooltip.append("The evaluation failed here.")
            item.setForeground(0, QBrush(QColor(VERDICTS[False][1])))
        if tooltip:
            item.setToolTip(0, "\n\n".join(tooltip))
        if group.comment:
            item.setText(0, f"{item.text(0)} — {group.comment.splitlines()[0]}")

        for child in group_check.children:
            if hasattr(child, "group"):
                child_item = self._add_group(None, child, error_node=error_node)
                item.addChild(child_item)
                child_item.setExpanded(child.passed is not True)
            else:
                verdict = group_check.condition_verdicts.get(child.condition.id)
                child_item = self._add_condition(child, verdict, error_node)
                item.addChild(child_item)
                child_item.setExpanded(verdict is False or (
                    verdict is None and child.n_rows > child.n_passing
                ))

        return item

    def _add_condition(self, check, verdict, error_node):
        from catan.gui.panels.StatisticsData import format_query_expression

        condition = check.condition
        definition = self.data.statistic_engine.registry.get(condition.query.statistic_key)
        unit = "" if definition is None else self.data.units.label(definition.value_kind)
        unit = f" {unit}" if unit else ""
        # The statistic title carries the unit already.

        title = (
            condition.query.statistic_key
            if definition is None
            else format_query_expression(condition.query, definition)
        )
        try:
            text = condition.threshold.expression(title, precision=4)
        except ValueError:
            text = title

        if check.error:
            summary = "error"
        elif check.n_rows:
            summary = f"{check.n_passing} of {check.n_rows} pass"
        else:
            summary = "no values"

        item = QTreeWidgetItem([text, summary, ""])
        _set_verdict(item, 2, verdict)

        tooltip = [text]
        if check.summary is not None:
            low, median, high = check.summary
            tooltip.append(
                f"Values: min {low:.4g}, median {median:.4g}, max {high:.4g}{unit}"
            )
        for note in (check.note, check.error):
            if note:
                tooltip.append(note)
        if verdict is None and not check.error:
            tooltip.append(
                "No verdict for this neuron: not reached (an earlier condition "
                "of the group failed) or not evaluated for it."
            )
        if condition.id == error_node:
            tooltip.append("The evaluation failed here.")
            item.setForeground(0, QBrush(QColor(VERDICTS[False][1])))
        item.setToolTip(0, "\n".join(tooltip))
        item.setToolTip(1, "\n".join(tooltip[1:]) or text)

        for index in range(len(check.values)):
            label, components = describe_row(check, index, self.component)
            value = check.values[index]
            row = QTreeWidgetItem([label, f"{value:.4g}{unit}", ""])
            _set_verdict(row, 2, bool(check.passes[index]))
            row.setData(0, Qt.ItemDataRole.UserRole, components)
            item.addChild(row)

        if check.n_rows > len(check.values):
            more = QTreeWidgetItem(
                [f"… {check.n_rows - len(check.values)} more failing rows (farther from the threshold)", "", ""]
            )
            more.setFlags(Qt.ItemFlag.NoItemFlags)
            item.addChild(more)

        return item

    def _highlight_selected_row(self, tree):
        items = tree.selectedItems()
        if not items:
            return

        components = items[0].data(0, Qt.ItemDataRole.UserRole)
        if components:
            self.state.update_highlighted_components(list(components))

    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self._closed = True
        if self._task_id is not None:
            self.state.tasks.cancel(self._task_id)
            self._task_id = None
        super().closeEvent(event)
