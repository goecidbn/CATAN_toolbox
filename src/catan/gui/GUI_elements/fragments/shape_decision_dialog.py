"""Shape decisions: review one ("two cells" / "one cell" in every session)
and list all of them (gui/data/shape_decisions).

Both dialogs are tool windows: the footprint panels stay usable. Hovering a
row highlights its footprints (violet: replaced / blueprint, orange: what is
split or merged); a click focuses the neuron.
"""

import numpy as np
from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from catan.core.structures import NeuronComponent
from catan.gui.data.shape_decisions import (
    APPLY,
    CONFLICT,
    DONE,
    ShapeDecision,
    owner,
    summary,
)
from catan.gui.utils.session_labels import session_tag


def decision_from_plan(data, plan, kind):
    """A ShapeDecision ``kind`` from a curator split / merge candidate (its
    whole and its two pieces); ValueError if it has not exactly two."""
    sources, targets = plan.resolve(data)
    whole, pieces = (
        (targets, sources) if plan.kind == "split" else (sources, targets)
    )
    if len(whole) != 1 or len(pieces) != 2:
        raise ValueError("A shape decision needs one whole footprint and two pieces.")

    def ref(component):
        footprint = data.state.get_footprint_from_component(component)
        if footprint is None or footprint < 0:
            raise ValueError(f"{component} has no footprint.")
        return (int(component.session_id), int(footprint))

    return ShapeDecision(kind, ref(whole[0]), tuple(ref(piece) for piece in pieces))


def decision_neurons(data, decision):
    """Neurons now holding the blueprint pieces (None: not assigned)."""
    if data.assignments is None:
        return [None, None]
    ids = np.asarray(data.assignments.ids)
    return [owner(ids, *ref) for ref in decision.pieces]


def decision_title(data, decision):
    neurons = " and ".join(
        "?" if n is None else str(n) for n in decision_neurons(data, decision)
    )
    return f"{decision.label.capitalize()}: neurons {neurons}"


def _neurons(components):
    return ", ".join(dict.fromkeys(str(c.neuron_id) for c in components))


def describe_step(decision, step):
    if step.state == DONE:
        return f"already {decision.label}", ""
    if step.state == CONFLICT:
        return "not applied: " + step.note, _neurons(step.targets)
    if decision.kind == "split":
        return (
            f"split the footprint of neuron {step.targets[0].neuron_id}",
            _neurons(step.targets + step.sources),
        )
    return (
        f"merge the footprints of neurons {_neurons(step.targets)}",
        _neurons(step.targets),
    )


class _HighlightingTree:
    """Hover highlight of a tree's rows (components, roles from
    ``row_components(item)``), restored when the pointer leaves."""

    def _init_highlight(self, tree):
        self._saved_highlight = None
        self._hovered = None
        tree.setMouseTracking(True)
        tree.itemEntered.connect(self._on_item_entered)
        tree.itemClicked.connect(self._on_item_clicked)
        tree.viewport().installEventFilter(self)

    def _on_item_entered(self, item, _column):
        if item is self._hovered:
            return
        components, roles = self.row_components(item)
        if self._saved_highlight is None:
            self._saved_highlight = (
                list(self.state.highlighted_components or []),
                dict(self.state.highlight_roles or {}),
            )
        self._hovered = item
        self.state.update_highlighted_components(components or None, roles=roles or None)

    def _restore_highlight(self):
        if self._saved_highlight is None:
            return
        components, roles = self._saved_highlight
        self._saved_highlight = None
        self._hovered = None
        ids = np.asarray(self.data.assignments.ids)
        components = [
            c
            for c in components
            if c.neuron_id < len(ids)
            and (c.session_id is None or ids[c.neuron_id, c.session_id] >= 0)
        ]
        self.state.update_highlighted_components(
            components or None,
            roles={c: roles[c] for c in components if c in roles} or None,
        )

    def _on_item_clicked(self, item, _column):
        components, roles = self.row_components(item)
        targets = [c for c in components if roles.get(c) == "target"] or components
        if targets:
            self.state.focused_component = targets[0]

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Leave and watched is self.tree.viewport():
            self._restore_highlight()
        return False

    def closeEvent(self, event):
        self._restore_highlight()
        super().closeEvent(event)


class ShapeDecisionDialog(_HighlightingTree, QDialog):
    """What a decision does in every session; apply it (a new decision is
    saved with the assignments first)."""

    def __init__(self, state, data, decision, decision_id=None, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.state = state
        self.data = data
        self.decision = decision
        self.decision_id = decision_id
        self.steps = []

        self.setWindowTitle("Shape decision")

        layout = QVBoxLayout(self)
        self.summary = QLabel(self)
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.tree = QTreeWidget(self)
        self.tree.setHeaderLabels(["Session", "What happens", "Neurons"])
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        head = self.tree.header()
        head.setStretchLastSection(False)
        head.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        head.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        head.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._init_highlight(self.tree)
        layout.addWidget(self.tree, 1)

        buttons = QHBoxLayout()
        self.status = QLabel(self)
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #9aa4b2;")
        buttons.addWidget(self.status, 1)

        self.apply_button = QPushButton(self)
        self.apply_button.clicked.connect(self.apply)
        buttons.addWidget(self.apply_button)

        self.save_button = None
        if decision_id is None:
            self.save_button = QPushButton("Save decision only", self)
            self.save_button.setToolTip(
                "Remember the decision without changing anything now: the "
                "curator then proposes this configuration only as decided."
            )
            self.save_button.clicked.connect(self.save_only)
            buttons.addWidget(self.save_button)

        close = QPushButton("Cancel" if decision_id is None else "Close", self)
        close.clicked.connect(self.close)
        buttons.addWidget(close)
        layout.addLayout(buttons)

        self.state.data_changed.connect(self._on_data_changed)
        self.refresh()
        self.resize(620, 420)

    def _on_data_changed(self, *_):
        if self.isVisible():
            self.refresh()

    def refresh(self):
        self.tree.clear()
        try:
            if self.data.assignments is None:
                raise ValueError("No assignments are active.")
            self.steps = self.data.plan_shape_decision(self.decision)
        except ValueError as exc:
            self.steps = []
            self.status.setText(str(exc))

        to_apply, done, conflicts = summary(self.steps)
        decision = self.decision
        whole_session, _ = decision.whole
        self.summary.setText(
            f"<b>{decision_title(self.data, decision)}</b><br>"
            + (
                "Every footprint covering both cells is split with the two "
                "pieces as blueprint."
                if decision.kind == "split"
                else "Every pair of footprints covering one cell each is merged, "
                f"with the whole footprint of {session_tag(whole_session)} as "
                "blueprint."
            )
            + " The decision is saved with the assignments: the curator then "
            "proposes this configuration only as decided, also where it "
            "appears later (e.g. after redetection)."
        )

        for step in self.steps:
            what, neurons = describe_step(decision, step)
            item = QTreeWidgetItem([session_tag(step.session_id), what, neurons])
            item.setData(0, Qt.ItemDataRole.UserRole, step)
            if step.state != APPLY:
                for column in range(3):
                    item.setForeground(column, Qt.GlobalColor.gray)
            self.tree.addTopLevelItem(item)

        self.apply_button.setText(
            ("Save decision && apply" if self.decision_id is None else "Apply")
            + f" ({to_apply} session{'s' if to_apply != 1 else ''})"
        )
        self.apply_button.setEnabled(to_apply > 0 or self.decision_id is None)
        if self.steps:
            self.status.setText(
                f"{to_apply} to apply · {done} already {decision.label} · "
                f"{conflicts} not applicable"
            )

    def row_components(self, item):
        step = item.data(0, Qt.ItemDataRole.UserRole)
        if step is None:
            return [], {}
        components = [*step.targets, *step.sources]
        roles = {c: "target" for c in step.targets}
        roles.update({c: "source" for c in step.sources if c not in roles})
        return components, roles

    def _save(self):
        if self.decision_id is None:
            self.decision_id = self.data.add_shape_decision(self.decision)
            if self.save_button is not None:
                self.save_button.hide()

    def save_only(self):
        self._save()
        self.close()

    def apply(self):
        self._restore_highlight()
        self._save()
        try:
            applied, skipped = self.data.apply_shape_decision(self.decision)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Decision not applied",
                f"{exc}\n\nNo changes were made to the data. The decision is saved.",
            )
            self.refresh()
            return

        lines = [
            f"Applied in {len(applied)} session{'s' if len(applied) != 1 else ''}"
            + (f": {', '.join(session_tag(s) for s in applied)}." if applied else ".")
        ]
        if skipped:
            lines.append("Not applied:")
            lines += [f"  {session_tag(s)}: {reason}" for s, reason in skipped]
        lines.append(
            "\nThe sessions were changed together: undo them as one batch in "
            "the manipulations list."
        )
        QMessageBox.information(self, "Shape decision applied", "\n".join(lines))
        self.refresh()


class ShapeDecisionsDialog(_HighlightingTree, QDialog):
    """All shape decisions of the active assignments: review / apply again,
    remove."""

    def __init__(self, state, data, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle("Shape decisions")
        self.state = state
        self.data = data

        layout = QVBoxLayout(self)
        self.summary = QLabel(self)
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.tree = QTreeWidget(self)
        self.tree.setHeaderLabels(["#", "Decision", "To apply", "As decided", "Not applicable"])
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        head = self.tree.header()
        head.setStretchLastSection(False)
        for column in range(5):
            head.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        head.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.tree.itemSelectionChanged.connect(self._update_buttons)
        self.tree.itemDoubleClicked.connect(lambda *_: self.review())
        self._init_highlight(self.tree)
        layout.addWidget(self.tree, 1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.review_button = QPushButton("Review && apply…", self)
        self.review_button.clicked.connect(self.review)
        buttons.addWidget(self.review_button)
        self.remove_button = QPushButton("Remove", self)
        self.remove_button.setToolTip(
            "Forget the decision. What it applied stays (undo it in the "
            "manipulations list)."
        )
        self.remove_button.clicked.connect(self.remove)
        buttons.addWidget(self.remove_button)
        close = QPushButton("Close", self)
        close.clicked.connect(self.close)
        buttons.addWidget(close)
        layout.addLayout(buttons)

        self.state.data_changed.connect(self._on_data_changed)
        self.refresh()
        self.resize(600, 340)

    def _on_data_changed(self, *_):
        if self.isVisible():
            self.refresh()

    def refresh(self):
        self.tree.clear()
        decisions = {} if self.data.assignments is None else self.data.shape_decisions()
        self.summary.setText(
            f"{len(decisions)} decision{'s' if len(decisions) != 1 else ''} of the "
            "active assignments. Hover a row to see its blueprint (orange: the "
            "two cells, violet: the whole footprint); open one to see and apply "
            "what it does in every session."
            if decisions
            else "No shape decisions yet. Make one from a split or merge "
            "candidate in the curator (Always split here / Always merge here)."
        )
        for decision_id, decision in sorted(decisions.items()):
            try:
                counts = summary(self.data.plan_shape_decision(decision))
            except ValueError:
                counts = ("–", "–", "–")
            item = QTreeWidgetItem(
                [str(decision_id), decision_title(self.data, decision), *map(str, counts)]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, decision_id)
            self.tree.addTopLevelItem(item)
        self._update_buttons()

    def _selected(self):
        items = self.tree.selectedItems()
        return None if not items else items[0].data(0, Qt.ItemDataRole.UserRole)

    def _update_buttons(self):
        chosen = self._selected() is not None
        self.review_button.setEnabled(chosen)
        self.remove_button.setEnabled(chosen)

    def row_components(self, item):
        decision = self.data.shape_decisions().get(item.data(0, Qt.ItemDataRole.UserRole))
        if decision is None:
            return [], {}
        ids = np.asarray(self.data.assignments.ids)
        components, roles = [], {}
        for refs, role in ((decision.pieces, "target"), ((decision.whole,), "source")):
            for ref in refs:
                row = owner(ids, *ref)
                if row is not None:
                    component = NeuronComponent(row, ref[0])
                    components.append(component)
                    roles[component] = role
        return components, roles

    def review(self):
        decision_id = self._selected()
        decision = self.data.shape_decisions().get(decision_id)
        if decision is None:
            return
        self._restore_highlight()
        dialog = ShapeDecisionDialog(
            self.state, self.data, decision, decision_id, parent=self.parentWidget()
        )
        dialog.setWindowFlag(Qt.WindowType.Tool, True)
        dialog.show()

    def remove(self):
        decision_id = self._selected()
        if decision_id is None:
            return
        self._restore_highlight()
        self.data.remove_shape_decision(decision_id)
        self.refresh()


def open_decision_dialog(widget, state, data, decision, decision_id=None):
    dialog = ShapeDecisionDialog(state, data, decision, decision_id, parent=widget.window())
    dialog.setWindowFlag(Qt.WindowType.Tool, True)
    dialog.show()
    return dialog


def open_decisions_list(widget, state, data):
    dialog = ShapeDecisionsDialog(state, data, parent=widget.window())
    dialog.setWindowFlag(Qt.WindowType.Tool, True)
    dialog.show()
    return dialog
