"""The manipulations of the active assignments (main menu status line).

One row per manipulation, newest first: what it did, the neurons now
holding its footprints, the sessions and footprints involved. The
manipulations of one curator batch are grouped under one row (expand it
for the single ones). Hovering a row highlights its footprints in all
panels (sources violet: the original footprints a split / merge replaced;
targets orange: the footprints it moved or created); a click focuses the
neuron. Manipulations recorded with their footprints (manipulation_records
format 2) can be undone, several at once (selected rows, or a whole batch);
undoing one that later manipulations built on undoes those too, after
asking. Undone manipulations are not listed. Records from older data are
listed but cannot be undone.
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
from catan.tracking.structures import manipulation_records as records

TYPE_LABELS = {
    "split": "Split",
    "merge": "Merge",
    "neuron_merge": "Neuron merge",
    "reassign": "Reassign",
    "move_footprints": "Move footprints",
    "new_neuron": "New neuron",
    "replace_shape": "Union shape",
}


def _owner(ids, footprint):
    session_id, footprint_id = footprint
    if not 0 <= session_id < ids.shape[1]:
        return None
    rows = np.flatnonzero(ids[:, session_id] == footprint_id)
    return int(rows[0]) if rows.size else None


def _ref(item):
    return (int(item["session_id"]), int(item["footprint_id"]))


def record_footprints(record):
    """(sources, targets) as footprints (session_id, footprint_id)."""
    if records.is_new(record):
        targets = records.moved(record) | records.created(record)
        sources = {tuple(x) for x in record.get("excluded", [])} - targets
        if record["type"] == "replace_shape":
            targets = {_ref(r) for r in record.get("results", [])}
        return sorted(sources), sorted(targets)
    sources = {_ref(r) for r in record.get("origin", [])}
    targets = {_ref(r) for r in (*record.get("results", []), *record.get("remainders", []))}
    return sorted(sources - targets), sorted(targets)


def _involved(data, record_id, record):
    """(neurons now, sessions, footprints) of a record."""
    ids = np.asarray(data.assignments.ids)
    sources, targets = record_footprints(record)
    footprints = [*targets, *sources]
    neurons = {n for n in (_owner(ids, fp) for fp in targets or footprints) if n is not None}
    if not neurons:
        neurons = {
            n for n, m in data.assignments.manipulation_id.items() if m == record_id
        }
    return neurons, {fp[0] for fp in footprints}, set(footprints)


def _texts(label, neurons, sessions, footprints, status):
    neurons, sessions = sorted(neurons), sorted(sessions)
    return (
        label,
        ", ".join(map(str, neurons[:6])) + (" …" if len(neurons) > 6 else ""),
        ", ".join(f"S{s}" for s in sessions[:8]) + (" …" if len(sessions) > 8 else ""),
        str(len(footprints)),
        status,
    )


def describe(data, record_id, record):
    """(type, neurons now, sessions, footprint count, status) texts."""
    status = "older data" if not records.is_new(record) else ""
    label = TYPE_LABELS.get(record["type"], record["type"])
    return _texts(label, *_involved(data, record_id, record), status)


def describe_batch(data, members):
    """The row of a curator batch (``members``: [(record_id, record)])."""
    neurons, sessions, footprints = set(), set(), set()
    for record_id, record in members:
        n, s, f = _involved(data, record_id, record)
        neurons |= n
        sessions |= s
        footprints |= f
    kinds = sorted({TYPE_LABELS.get(r["type"], r["type"]) for _, r in members})
    label = f"Curator batch · {len(members)} × {' / '.join(kinds)}"
    return _texts(label, neurons, sessions, footprints, "")


class ManipulationsDialog(QDialog):

    def __init__(self, state, data, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle("Manipulations")
        self.state = state
        self.data = data
        self._saved_highlight = None
        self._hovered = None

        layout = QVBoxLayout(self)
        self.summary = QLabel(self)
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.tree = QTreeWidget(self)
        self.tree.setColumnCount(6)
        self.tree.setHeaderLabels(["#", "Manipulation", "Neurons now", "Sessions", "Footprints", ""])
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setMouseTracking(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        head = self.tree.header()
        head.setStretchLastSection(False)
        for column in range(6):
            head.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        head.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.tree.itemEntered.connect(self._on_item_entered)
        self.tree.itemClicked.connect(self._on_item_clicked)
        self.tree.itemSelectionChanged.connect(self._update_buttons)
        self.tree.viewport().installEventFilter(self)
        layout.addWidget(self.tree, 1)

        buttons = QHBoxLayout()
        self.status = QLabel(self)
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #9aa4b2;")
        buttons.addWidget(self.status, 1)
        self.undo_button = QPushButton("Undo", self)
        self.undo_button.setToolTip("Undo the selected manipulations")
        self.undo_button.clicked.connect(self.undo_selected)
        buttons.addWidget(self.undo_button)
        close = QPushButton("Close", self)
        close.clicked.connect(self.close)
        buttons.addWidget(close)
        layout.addLayout(buttons)

        self.state.data_changed.connect(self._on_data_changed)
        self.refresh()
        self.resize(640, 420)

    # ------------------------------------------------------------------

    def _on_data_changed(self, *_):
        if self.isVisible():
            self.refresh()

    def refresh(self):
        selected = self._selected_keys()
        expanded = {
            self.tree.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole)
            for i in range(self.tree.topLevelItemCount())
            if self.tree.topLevelItem(i).isExpanded()
        }
        self.tree.clear()
        assignments = self.data.assignments
        if assignments is None:
            self.summary.setText("No assignments are active.")
            self._update_buttons()
            return

        manipulations = assignments.manipulations
        listed = [
            (record_id, record)
            for record_id, record in manipulations.items()
            if records.listed(record)
        ]
        undone = sum(1 for _, r in listed if records.is_undone(r))
        shown = [(i, r) for i, r in listed if not records.is_undone(r)]
        old = sum(1 for _, r in shown if not records.is_new(r))
        active = len(shown) - old

        lines = [f"{active} manipulation{'s' if active != 1 else ''} that can be undone"]
        if old:
            lines.append(f"{old} from older data — listed, cannot be edited")
        if undone:
            lines.append(f"{undone} undone (not listed)")
        self.summary.setText(
            " · ".join(lines)
            + ". Hover a row to see its footprints (violet: replaced originals, "
            "orange: moved or created), click it to focus the neuron. Select "
            "several rows (Ctrl / Shift + click) or a curator batch to undo "
            "them together."
        )

        # One row per single manipulation, one (expandable) per batch.
        batches = {}
        for record_id, record in shown:
            batch = records.batch_of(record)
            if batch is not None:
                batches.setdefault(batch, []).append((record_id, record))
        rows = [(i, r) for i, r in shown if records.batch_of(r) is None]
        rows += [(min(i for i, _ in members), members) for members in batches.values()]

        for record_id, entry in sorted(rows, key=lambda item: -item[0]):
            if isinstance(entry, list):
                members = sorted(entry, key=lambda item: -item[0])
                ids = [i for i, _ in members]
                key = f"batch {records.batch_of(members[0][1])}"
                item = QTreeWidgetItem(
                    [f"{min(ids)}–{max(ids)}" if len(ids) > 1 else str(ids[0]),
                     *describe_batch(self.data, members)]
                )
                item.setData(0, Qt.ItemDataRole.UserRole, key)
                item.setData(0, Qt.ItemDataRole.UserRole + 1, ids)
                item.setToolTip(
                    1,
                    "Applied together by the curator. Undo undoes the whole "
                    "batch; expand it to undo single manipulations.",
                )
                for member_id, member in members:
                    child = self._record_item(member_id, member)
                    item.addChild(child)
                    if member_id in selected:
                        child.setSelected(True)
                self.tree.addTopLevelItem(item)
                item.setExpanded(key in expanded)
            else:
                key = record_id
                item = self._record_item(record_id, entry)
                self.tree.addTopLevelItem(item)
            if key in selected:
                item.setSelected(True)
        self._update_buttons()

    def _record_item(self, record_id, record):
        texts = describe(self.data, record_id, record)
        item = QTreeWidgetItem([str(record_id), *texts])
        item.setData(0, Qt.ItemDataRole.UserRole, record_id)
        item.setData(0, Qt.ItemDataRole.UserRole + 1, [record_id])
        if texts[-1]:
            for column in range(6):
                item.setForeground(column, Qt.GlobalColor.gray)
        return item

    def _selected_keys(self):
        return {item.data(0, Qt.ItemDataRole.UserRole) for item in self.tree.selectedItems()}

    def _selected_records(self):
        """Selected record IDs (a batch row: all its manipulations)."""
        ids = []
        for item in self.tree.selectedItems():
            ids += item.data(0, Qt.ItemDataRole.UserRole + 1) or []
        return sorted(set(ids))

    def _update_buttons(self):
        manipulations = (
            {} if self.data.assignments is None else self.data.assignments.manipulations
        )
        selected = [manipulations[i] for i in self._selected_records() if i in manipulations]
        undoable = [r for r in selected if records.undoable(r)]
        self.undo_button.setEnabled(bool(undoable))
        self.undo_button.setText(f"Undo {len(undoable)}" if len(undoable) > 1 else "Undo")
        if not selected:
            self.undo_button.setToolTip("Select manipulations (Ctrl / Shift + click for several)")
        elif not undoable:
            self.undo_button.setToolTip("Recorded by an older CATAN version: cannot be undone")
        else:
            self.undo_button.setToolTip("Undo the selected manipulations")

    # ------------------------------------------------------------------
    # Hover: highlight the footprints of a row

    def _components(self, record_ids):
        manipulations = self.data.assignments.manipulations
        ids = np.asarray(self.data.assignments.ids)
        sources, targets = set(), set()
        for record_id in record_ids:
            record = manipulations.get(record_id)
            if record is None:
                continue
            s, t = record_footprints(record)
            sources |= set(s)
            targets |= set(t)
        sources, targets = sorted(sources - targets), sorted(targets)
        components, roles = [], {}
        for footprints, role in ((sources, "source"), (targets, "target")):
            for footprint in footprints:
                owner = _owner(ids, footprint)
                if owner is None:
                    continue
                component = NeuronComponent(owner, footprint[0])
                if component not in roles:
                    components.append(component)
                    roles[component] = role
        return components, roles

    def _on_item_entered(self, item, _column):
        record_id = item.data(0, Qt.ItemDataRole.UserRole)
        if record_id == self._hovered:
            return
        components, roles = self._components(item.data(0, Qt.ItemDataRole.UserRole + 1) or [])
        if self._saved_highlight is None:
            self._saved_highlight = (
                list(self.state.highlighted_components or []),
                dict(self.state.highlight_roles or {}),
            )
        self._hovered = record_id
        self.state.update_highlighted_components(components or None, roles=roles or None)

    def _restore_highlight(self):
        if self._saved_highlight is None:
            return
        components, roles = self._saved_highlight
        self._saved_highlight = None
        self._hovered = None
        self.state.update_highlighted_components(components or None, roles=roles or None)

    def eventFilter(self, watched, event):
        if watched is self.tree.viewport() and event.type() == QEvent.Type.Leave:
            self._restore_highlight()
        return False

    def _on_item_clicked(self, item, _column):
        components, roles = self._components(item.data(0, Qt.ItemDataRole.UserRole + 1) or [])
        targets = [c for c in components if roles.get(c) == "target"] or components
        if targets:
            self.state.focused_component = targets[0]

    # ------------------------------------------------------------------
    # Undo

    def undo_selected(self):
        manipulations = self.data.assignments.manipulations
        requested = [
            i for i in self._selected_records()
            if i in manipulations and records.undoable(manipulations[i])
        ]
        if not requested:
            return
        try:
            chain = self.data.manipulation_undo_chain(requested)
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot undo", str(exc))
            return

        def name(i):
            return f"#{i} {TYPE_LABELS.get(manipulations[i]['type'], manipulations[i]['type'])}"

        if len(requested) == 1:
            what = name(requested[0])
        else:
            what = f"{len(requested)} manipulations"
        later = [name(i) for i in sorted(chain) if i not in requested]
        if later:
            text = (
                f"Later manipulations built on {what}: "
                + ", ".join(later[:12])
                + (f" and {len(later) - 12} more" if len(later) > 12 else "")
                + ".\n\nThey moved footprints these moved or created, or filled a "
                "place they emptied. Undo all of them (newest first)?"
            )
            yes_text = f"Undo all {len(chain)}"
        else:
            text = f"Undo {what}? Their footprints go back to their neurons."
            yes_text = "Undo" if len(chain) == 1 else f"Undo {len(chain)}"

        box = QMessageBox(QMessageBox.Icon.Question, "Undo manipulation", text, parent=self)
        undo = box.addButton(yes_text, QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        if box.clickedButton() is not undo:
            return

        self._restore_highlight()

        def finished(prepared):
            notes = getattr(prepared, "notes", [])
            self.status.setText(
                f"Undone: {len(chain)} manipulation{'s' if len(chain) != 1 else ''}."
                + (" " + " ".join(notes) if notes else "")
            )

        try:
            self.data.undo_manipulations(chain, finished=finished)
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot undo", str(exc))

    def closeEvent(self, event):
        self._restore_highlight()
        try:
            self.state.data_changed.disconnect(self._on_data_changed)
        except (RuntimeError, TypeError):
            pass
        super().closeEvent(event)
