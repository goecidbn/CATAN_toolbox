"""Curator manipulation controls and sequential candidate review."""

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import replace

from PySide6.QtCore import Qt, QTimer, QEventLoop
from PySide6.QtWidgets import (
    QApplication,
    QProgressBar,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QCheckBox,
    QPushButton,
    QLabel,
    QGroupBox,
    QMessageBox,
    QBoxLayout,
    QSizePolicy,
)

from catan.gui.data.curation_actions import (
    BATCH_ACTIONS,
    ManipulationBatch,
    ManipulationSpec,
    ManipulationPlan,
    OccupiedDestination,
    _validate as validate_endpoints,
)

# Candidate entries built from the user's source/target highlight.
HIGHLIGHT_PREFIX = "Highlighted"
HIGHLIGHT_LABEL = HIGHLIGHT_PREFIX + ": "
# What a highlight must contain per kind (sources violet, targets orange).
HIGHLIGHT_NEEDS = {
    "split": "1 target footprint (to split) and 2 source footprints "
    "(references from two different neurons)",
    "merge": "2 target footprints (to merge) and 1 source footprint (reference)",
    "reassign": "1 source footprint and target footprints of 1 other neuron",
    "neuron_merge": "footprints of 1 source neuron and 1 target neuron",
}

CONFLICT_HELP = {
    "split": (
        "A conflict: a source neuron already has a footprint in the target "
        "session. Apply anyway excludes that footprint (it stays in the data "
        "as an excluded single-footprint neuron) and matches the split "
        "result to the source neuron instead. Ask additionally offers "
        "keeping it as a separate, included neuron."
    ),
    "reassign": (
        "A conflict: the target neuron already has a footprint in the "
        "session of the reassigned footprint. Apply anyway excludes the "
        "target's existing footprint (it stays in the data as an excluded "
        "single-footprint neuron) and assigns the reassigned footprint "
        "instead. Ask additionally offers keeping it as a separate, "
        "included neuron."
    ),
    "neuron_merge": (
        "A conflict: both neurons have a footprint in the same session. "
        "Apply anyway excludes the source neuron's footprint in those "
        "sessions (it stays in the data as an excluded single-footprint "
        "neuron) and keeps the target's, then merges. Ask additionally "
        "offers keeping the source's footprint as a separate, included neuron."
    ),
}

# Ask dialog: what Exclude / Keep do with the conflicting footprint.
CONFLICT_CHOICES = {
    "neuron_merge": (
        "Exclude: the source neuron's footprint in that session is excluded "
        "(kept as an excluded single-footprint neuron); the target keeps its "
        "own, then the neurons are merged.\n"
        "Keep: the source's footprint moves to a separate, included neuron; "
        "nothing is excluded.\n"
    ),
}
DEFAULT_CONFLICT_CHOICES = (
    "Exclude: the existing footprint is excluded (kept as an excluded "
    "single-footprint neuron) and the result takes its place.\n"
    "Keep: the existing footprint moves to a separate, included "
    "neuron; nothing is excluded.\n"
)
from catan.tracking.structures import ReviewStatus
from catan.core.structures import NeuronComponent
from catan.gui.GUI_elements.utils.menu_creation import fit_message_box_buttons


class CuratorActions(QGroupBox):
    def __init__(self, controller):
        super().__init__("Actions on results", controller.menu)

        self.controller = controller
        self.data = controller.data
        self.state = controller.state

        self.result_version = None
        self.run_id = 0
        self.queue = None
        self.pending = None
        self.reports = []
        self.committing = False
        self.waiting = False

        # Background preparation of the pending manipulation.
        self._prepare_task = None
        self.state.tasks.task_cancelled.connect(self._on_prepare_stopped)
        self.state.tasks.task_failed.connect(self._on_prepare_stopped)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 16, 10, 10)

        self.selector_row = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self.selector_row.setSpacing(8)

        self.kind = QComboBox()
        for label, value in (
            ("No manipulation", "none"),
            ("Split footprint", "split"),
            ("Merge footprints", "merge"),
            ("Merge neurons", "neuron_merge"),
            ("Reassign footprint", "reassign"),
            ("Mark reviewed", "mark_reviewed"),
            ("Mark uncertain", "mark_uncertain"),
            ("Exclude neurons", "exclude_neuron"),
            ("Exclude footprints", "exclude_footprint"),
        ):
            self.kind.addItem(label, value)

        # Separate per-candidate manipulations from batch actions.
        self.kind.insertSeparator(self.kind.findData("mark_reviewed"))

        self.kind.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )

        self.candidate = QComboBox()
        self.candidate.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.candidate.setMinimumContentsLength(16)
        self.candidate.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

        self.selector_row.addWidget(self.kind)
        self.selector_row.addWidget(self.candidate, 1)

        layout.addLayout(self.selector_row)

        self.confirm = QCheckBox("Display and confirm each manipulation")
        layout.addWidget(self.confirm)

        conflict_row = QHBoxLayout()
        self.conflict_label = QLabel("On conflict:")
        self.conflict = QComboBox()

        for label, value, tooltip in (
            ("Ask", "ask", "Ask for each conflicting candidate."),
            ("Skip candidate", "skip", "Leave conflicting candidates unchanged."),
            (
                "Apply anyway",
                "resolve",
                "Exclude the footprint occupying the destination, then apply.",
            ),
        ):
            self.conflict.addItem(label, value)
            self.conflict.setItemData(
                self.conflict.count() - 1, tooltip, Qt.ItemDataRole.ToolTipRole
            )

        conflict_row.addWidget(self.conflict_label)
        conflict_row.addWidget(self.conflict)
        conflict_row.addStretch()
        layout.addLayout(conflict_row)

        self.conflict_help = QLabel()
        self.conflict_help.setWordWrap(True)
        self.conflict_help.setStyleSheet("color: #9aa3ad;")
        layout.addWidget(self.conflict_help)

        row = QHBoxLayout()

        self.current = QPushButton("Apply current")
        self.all = QPushButton("Apply all selected")
        self.next = QPushButton("Apply && next")
        self.skip = QPushButton("Skip")
        self.stop = QPushButton("Stop")

        for button in (
            self.current,
            self.all,
            self.next,
            self.skip,
            self.stop,
        ):
            row.addWidget(button)

        layout.addLayout(row)

        # Progress of "Apply all selected" (no modal dialog: the window and
        # conflict dialogs stay usable).
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.hide()
        layout.addWidget(self.progress)

        # Stop during a batch: stop after the current plan, commit applied.
        self._batch_running = False
        self._batch_stop = False

        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.kind.currentIndexChanged.connect(self._settings_changed)
        self.confirm.toggled.connect(self._settings_changed)
        self.conflict.currentIndexChanged.connect(self._settings_changed)
        self.candidate.currentIndexChanged.connect(self._preview_current)
        self.candidate.currentIndexChanged.connect(
            lambda *_: self._update_selector_layout()
        )

        self.current.clicked.connect(lambda: self._start(False))
        self.all.clicked.connect(lambda: self._start(True))
        self.next.clicked.connect(lambda: self._apply(self.run_id))
        self.skip.clicked.connect(self._skip)
        self.stop.clicked.connect(self._on_stop)

        self.state.focused_component_changed.connect(self.refresh_candidates)
        self.state.selected_components_changed.connect(self.refresh_candidates)
        self.state.data_changed.connect(self._data_changed)
        self.state.highlighted_components_changed.connect(self._on_highlight_edited)
        self._showing = False

        self.sync()

    def sync(self):
        spec = self.controller.root_filter.manipulation or ManipulationSpec()

        for widget in (self.kind, self.confirm, self.conflict):
            widget.blockSignals(True)

        self.kind.setCurrentIndex(self.kind.findData(spec.kind))
        self.confirm.setChecked(spec.confirm_each)
        self.conflict.setCurrentIndex(self.conflict.findData(spec.on_conflict))

        for widget in (self.kind, self.confirm, self.conflict):
            widget.blockSignals(False)

        self.refresh_candidates()

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if hasattr(self, "candidate"):
            self._update_selector_layout()

    def _update_selector_layout(self):
        margins = self.layout().contentsMargins()
        available = max(
            0,
            self.contentsRect().width() - margins.left() - margins.right(),
        )

        text = self.candidate.currentText()

        candidate_width = max(
            320,
            self.candidate.fontMetrics().horizontalAdvance(text) + 56,
        )

        required = (
            self.kind.sizeHint().width() + self.selector_row.spacing() + candidate_width
        )

        direction = (
            QBoxLayout.Direction.LeftToRight
            if available >= required
            else QBoxLayout.Direction.TopToBottom
        )

        if self.selector_row.direction() != direction:
            self.selector_row.setDirection(direction)

        self.candidate.setToolTip(text or "Operation for the currently focused neuron")

    def _settings_changed(self, *_):
        old = self.controller.root_filter.manipulation or ManipulationSpec()

        self.controller.root_filter.manipulation = replace(
            old,
            kind=self.kind.currentData(),
            group_id=None,
            confirm_each=self.confirm.isChecked(),
            on_conflict=self.conflict.currentData(),
        )

        self.controller.filter_store.mark_working_dirty()
        self.controller._refresh_preset_controls()

        self.refresh_candidates()

    def result_ready(self):
        self.result_version = self.state.data_version
        self.refresh_candidates()

    def _conflict_choice(self, slots, *, batch=False):
        """"exclude", "detach", "skip" or (batch only) "stop"."""
        policy = getattr(self, "_policy", None) or self.conflict.currentData()

        if policy == "skip":
            return "skip"

        if policy == "resolve":
            return "exclude"

        if policy == "detach":
            return "detach"

        dialog = QMessageBox(self.window())
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Destination already contains a footprint")
        kind = self.kind.currentData()
        dialog.setText(str(OccupiedDestination(slots, kind)))
        dialog.setInformativeText(
            CONFLICT_CHOICES.get(kind, DEFAULT_CONFLICT_CHOICES)
            + "Skip leaves this candidate unchanged."
            + (
                "\n\nThis run applies all changes together when it ends."
                if batch
                else ""
            )
        )

        exclude_button = dialog.addButton(
            "Exclude existing && apply", QMessageBox.ButtonRole.AcceptRole
        )
        detach_button = dialog.addButton(
            "Keep as separate neuron && apply", QMessageBox.ButtonRole.AcceptRole
        )
        skip_button = dialog.addButton("Skip", QMessageBox.ButtonRole.RejectRole)
        stop_button = (
            dialog.addButton("Stop run", QMessageBox.ButtonRole.DestructiveRole)
            if batch
            else None
        )
        dialog.setDefaultButton(skip_button)

        fit_message_box_buttons(dialog)

        remember = None
        if self.queue is not None:
            remember = QCheckBox("Use this choice for the remaining conflicts")
            dialog.setCheckBox(remember)

        self._run_dialog(dialog)
        clicked = dialog.clickedButton()

        choice = {
            exclude_button: "exclude",
            detach_button: "detach",
            stop_button: "stop",
        }.get(clicked, "skip")

        if remember is not None and remember.isChecked() and choice != "stop":
            self._policy = {"exclude": "resolve", "detach": "detach"}.get(choice, "skip")

        return choice

    def _run_dialog(self, dialog):
        """Show a conflict dialog without blocking the application: it can
        be moved and the panels (e.g. footprints) stay usable while the
        answer is pending. Closing it counts as no choice."""
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        dialog.setModal(False)

        loop = QEventLoop(self)
        dialog.finished.connect(loop.quit)
        dialog.show()
        dialog.raise_()
        loop.exec()

    def _plans(self):
        result = self.controller.current_result

        kind = self.kind.currentData()
        if result is None or kind == "none" or kind in BATCH_ACTIONS:
            return []

        mapping = result.root_targets()

        # for neuron in result.neurons:
        #     mapping.update(
        #         result.inspection_targets(
        #             neuron,
        #             "group",
        #             group_id,
        #         )
        #         or {}
        #     )

        def sort(component):
            return (
                component.neuron_id,
                (-1 if component.session_id is None else component.session_id),
            )

        if kind == "split":
            requests = [
                (
                    tuple(sorted(sources, key=sort)),
                    (target,),
                )
                for target, sources in mapping.items()
                if sources
            ]

        elif kind == "merge":
            inverse = defaultdict(set)

            for target, sources in mapping.items():
                for source in sources:
                    inverse[(source, target.session_id)].add(target)

            requests = [
                (
                    (source,),
                    tuple(sorted(targets, key=sort)),
                )
                for (source, _), targets in inverse.items()
            ]

        else:
            requests = [
                ((source,), (target,))
                for target, sources in mapping.items()
                for source in sources
            ]

        requests.sort(
            key=lambda request: (
                tuple(map(sort, request[1])),
                tuple(map(sort, request[0])),
            )
        )

        plans = []
        seen = set()

        for sources, targets in requests:
            try:
                plan = ManipulationPlan.capture(self.data, kind, sources, targets)
            except ValueError:
                # Historical evidence may outlive an endpoint.
                continue

            key = (plan.sources, plan.targets)

            if key not in seen:
                seen.add(key)
                plans.append(plan)

        return plans

    def refresh_candidates(self, *_):
        if self.queue is not None:
            return

        previous = self.candidate.currentData()

        self.candidate.blockSignals(True)
        self.candidate.clear()

        try:
            focused = self.state.focused_component

            for plan in self._plans():
                sources, targets = plan.resolve(self.data)

                if focused is not None and any(
                    target.neuron_id == focused.neuron_id for target in targets
                ):
                    self.candidate.addItem(
                        self._label(plan.kind, sources, targets),
                        plan,
                    )

            index = self.candidate.findData(previous)

            if index >= 0:
                self.candidate.setCurrentIndex(index)

        except ValueError as exc:
            self.status.setText(str(exc))

        finally:
            self.candidate.blockSignals(False)

        self._update_selector_layout()
        self._buttons()

        # A user-edited highlight stays the candidate after the list is
        # rebuilt (e.g. a Ctrl-click also changed the focused footprint).
        if getattr(self.state, "highlight_edited", False):
            self._on_highlight_edited()

    # ------------------------------------------------------------------
    # Candidate from the user's source/target highlight
    # ------------------------------------------------------------------

    def _highlight_endpoints_for(self, kind):
        """(sources, targets) of the current source/target highlight in the
        endpoint form of ``kind``; None without a role view. Highlighted
        components without a role count as targets."""
        roles = self.state.highlight_roles
        if not roles:
            return None

        components = list(self.state.highlighted_components or ())
        sources = [c for c in components if roles.get(c) == "source"]
        targets = [c for c in components if roles.get(c, "target") == "target"]

        def neurons(items):
            return list(
                dict.fromkeys(NeuronComponent(int(c.neuron_id), None) for c in items)
            )

        if kind == "neuron_merge":
            sources, targets = neurons(sources), neurons(targets)
        elif kind == "reassign":
            targets = neurons(targets)

        key = lambda c: (c.neuron_id, -1 if c.session_id is None else c.session_id)
        return tuple(sorted(sources, key=key)), tuple(sorted(targets, key=key))

    def _on_highlight_edited(self):
        """A user edit of the source/target highlight (Ctrl-click roles,
        Footprints highlight menu) selects the matching candidate, or adds
        it as "Highlighted: …", so Apply uses what is highlighted."""
        if (
            self._showing
            or self.queue is not None
            or not getattr(self.state, "highlight_edited", False)
        ):
            return

        kind = self.kind.currentData()
        if kind in (None, "none") or kind in BATCH_ACTIONS:
            return

        endpoints = self._highlight_endpoints_for(kind)
        if endpoints is None:
            return

        sources, targets = endpoints
        plan, problem = None, None
        try:
            validate_endpoints(self.data, kind, sources, targets)
            plan = ManipulationPlan.capture(self.data, kind, sources, targets)
        except (ValueError, AttributeError, IndexError) as exc:
            problem = str(exc)

        self.candidate.blockSignals(True)
        try:
            # Replace an earlier highlight entry.
            for index in reversed(range(self.candidate.count())):
                if self.candidate.itemText(index).startswith(HIGHLIGHT_PREFIX):
                    self.candidate.removeItem(index)

            if plan is not None:
                index = next(
                    (
                        i
                        for i in range(self.candidate.count())
                        if self.candidate.itemData(i) == plan
                    ),
                    -1,
                )
                if index < 0:
                    self.candidate.insertItem(
                        0,
                        HIGHLIGHT_LABEL + self._label(kind, sources, targets),
                        plan,
                    )
                    index = 0
                self.candidate.setCurrentIndex(index)
                self.status.setText(
                    "Using the highlighted footprints: "
                    + self._label(kind, sources, targets)
                )
            else:
                self.candidate.insertItem(0, "Highlighted (incomplete)", None)
                self.candidate.setCurrentIndex(0)
                if "expected source/target counts" in (problem or ""):
                    problem = (
                        f"highlight {HIGHLIGHT_NEEDS.get(kind, 'sources and targets')} "
                        f"(now {len(sources)} source, {len(targets)} target)."
                    )
                self.status.setText(f"Highlight not usable for {kind}: {problem}")
        finally:
            self.candidate.blockSignals(False)

        self._update_selector_layout()
        self._buttons()

    @staticmethod
    def _label(kind, sources, targets):
        def label(component):
            # return (
            #     f"({component.neuron_id}"
            #     + ("" if component.session_id is None else f",{component.session_id}")
            #     + ")"
            # )
            return f"n{component.neuron_id}" + (
                "" if component.session_id is None else f"/s{component.session_id}"
            )

        source = ", ".join(map(label, sources))
        target = ", ".join(map(label, targets))

        return {
            "split": (f"{target} → {source}"),
            "merge": f"{target} using {source}",
            "neuron_merge": f"{source} into {target}",
            "reassign": f"{source} → {target}",
        }[kind]

    def _show(self, plan, *, focus=False):
        sources, targets = plan.resolve(self.data)

        if focus and targets:
            self.state.focused_component = targets[0]

        self._showing = True
        try:
            self.controller._highlight_endpoints(set(sources), set(targets))
        finally:
            self._showing = False

        self.status.setText(self._label(plan.kind, sources, targets))

    def _preview_current(self, *_):
        plan = self.candidate.currentData()

        if plan is not None and self.queue is None:
            try:
                self._show(plan)
            except ValueError as exc:
                self.status.setText(str(exc))

    def _buttons(self):
        running = self.queue is not None

        batch = self.kind.currentData() in BATCH_ACTIONS

        # Batch actions have no candidate list and no per-item review.
        self.candidate.setVisible(not batch)
        self.confirm.setVisible(not batch)

        help_text = CONFLICT_HELP.get(self.kind.currentData())
        for widget in (self.conflict_label, self.conflict, self.conflict_help):
            widget.setVisible(help_text is not None)
        self.conflict_help.setText(help_text or "")

        if running:
            self.controller.menu.evaluate_button.setEnabled(False)

        for widget in (
            self.kind,
            self.confirm,
            self.conflict,
            self.candidate,
        ):
            widget.setEnabled(not running)

        self.current.setVisible(not running)
        self.all.setVisible(not running)

        self.current.setEnabled(
            not running
            and (
                self._focused_result_neuron() is not None
                if batch
                else self.candidate.currentData() is not None
            )
        )
        self.all.setEnabled(
            not running
            and self.controller.current_result is not None
            and bool(self.controller.current_result.neurons)
            and bool(self.state.selected_components)
            and self.kind.currentData() != "none"
        )

        self.next.setVisible(running and self.review)
        self.skip.setVisible(running and self.review)
        self.stop.setVisible(running)

        self.next.setEnabled(self.pending is not None and not self.waiting)
        self.skip.setEnabled(self.pending is not None and not self.waiting)

    @staticmethod
    def _unique_plans(plans):
        unique = {}

        for plan in plans:
            key = (
                frozenset(plan.sources + plan.targets)
                if plan.kind == "neuron_merge"
                else (plan.sources, plan.targets)
            )

            unique.setdefault(key, plan)

        return list(unique.values())

    def _focused_result_neuron(self):
        result = self.controller.current_result
        focused = self.state.focused_component

        if result is None or focused is None:
            return None

        return focused.neuron_id if focused.neuron_id in result.neurons else None

    def _batch_targets(self, all_selected):
        result = self.controller.current_result

        if result is None:
            raise ValueError("Evaluate the filter first.")

        if all_selected:
            neurons = {
                component.neuron_id
                for component in self.state.selected_components or []
            } & result.neurons
        else:
            focused = self._focused_result_neuron()
            neurons = set() if focused is None else {focused}

        # Results may contain neurons excluded after (or before) evaluation.
        neurons = {neuron for neuron in neurons if self.data.is_included(neuron)}

        if not neurons:
            raise ValueError(
                "No matching neurons selected. Use 'Select matching "
                "neurons' or focus a matching neuron."
            )

        if self.kind.currentData() != "exclude_footprint":
            return sorted(neurons)

        components = {
            component
            for component in result.target_components()
            if component.neuron_id in neurons
        }

        if any(component.session_id is None for component in components):
            raise ValueError(
                "This preset matches whole neurons. Use 'Exclude neurons', "
                "or evaluate footprints (by footprint or with footprint "
                "endpoints)."
            )

        return sorted(components, key=lambda c: (c.neuron_id, c.session_id))

    def _apply_batch(self, all_selected):
        kind = self.kind.currentData()

        try:
            if self.state.tasks.processing_busy():
                raise ValueError("Other processing is running. Try again afterwards.")

            targets = self._batch_targets(all_selected)

        except ValueError as exc:
            self.status.setText(str(exc))
            return

        if kind in ("exclude_neuron", "exclude_footprint"):
            noun = "neurons" if kind == "exclude_neuron" else "footprints"

            answer = QMessageBox.question(
                self.window(),
                f"Exclude {noun}",
                f"Exclude {len(targets)} {noun} matched by this preset?\n\n"
                "Excluded items remain in the data and can be re-included.",
            )

            if answer != QMessageBox.StandardButton.Yes:
                self.status.setText("Exclusion cancelled.")
                return

        try:
            if kind == "mark_reviewed":
                self.data.set_review_status(targets, ReviewStatus.REVIEWED)
                done = f"Marked {len(targets)} neurons as reviewed."

            elif kind == "mark_uncertain":
                self.data.set_review_status(targets, ReviewStatus.UNCERTAIN)
                done = f"Marked {len(targets)} neurons as uncertain."

            elif kind == "exclude_neuron":
                excluded = self.data.exclude_neurons(targets)
                done = f"Excluded {len(excluded)} neurons."

            else:
                retired = self.data.exclude_components(targets)
                done = f"Excluded {len(retired)} footprints."

        except Exception as exc:
            QMessageBox.warning(self.window(), "Action failed", str(exc))
            return

        self.status.setText(done)

    def _start(self, all_selected):
        if self.queue is not None:
            return

        if self.kind.currentData() in BATCH_ACTIONS:
            self._apply_batch(all_selected)
            return

        try:
            if all_selected:
                selected = {
                    component.neuron_id
                    for component in self.state.selected_components or []
                }

                plans = [
                    plan
                    for plan in self._plans()
                    if all(
                        target.neuron_id in selected
                        for target in plan.resolve(self.data)[1]
                    )
                ]

            else:
                plan = self.candidate.currentData()
                plans = [] if plan is None else [plan]

            plans = self._unique_plans(plans)

            if not plans:
                raise ValueError(
                    "No current candidates. Evaluate the filter "
                    "and select matches first."
                )

        except ValueError as exc:
            self.status.setText(str(exc))
            return

        self.run_id += 1
        self.queue = deque(plans)
        self.pending = None
        self.reports = []
        self._policy = None

        self.review = all_selected and self.confirm.isChecked()

        # Single candidates and reviewed runs report problems immediately;
        # automatic runs collect them for the summary.
        self.interactive = self.review or not all_selected

        self.generation = self.controller._filter_generation
        self.root = self.controller.root_filter

        self.spec = deepcopy(
            self.root.manipulation or ManipulationSpec(kind=self.kind.currentData())
        )

        if (
            all_selected
            and not self.review
            and self.kind.currentData() in ManipulationBatch.KINDS
        ):
            self._run_batch(plans)
            return

        self._advance(self.run_id)

    def _advance(self, token):
        if token != self.run_id or self.queue is None:
            return

        if (
            self.controller.root_filter is not self.root
            or self.controller._filter_generation != self.generation
        ):
            self._finish("Stopped: filter changed")
            return

        if not self.queue:
            self._finish("Finished")
            return

        self.pending = self.queue.popleft()

        try:
            self._show(self.pending, focus=True)

        except ValueError as exc:
            self.reports.append(("skipped", str(exc)))
            self.pending = None

            QTimer.singleShot(
                0,
                lambda: self._advance(token),
            )
            return

        self._buttons()

        if not self.review:
            QTimer.singleShot(
                0,
                lambda: self._apply(token),
            )

    def _apply(self, token):
        if token != self.run_id or self.pending is None:
            return

        if (
            self.controller.root_filter is not self.root
            or self.controller._filter_generation != self.generation
        ):
            self._finish("Stopped: filter changed")
            return

        tasks = self.state.tasks

        if tasks.processing_busy():
            self._finish("Stopped: other processing is running")
            return

        if tasks.defer_for_background(lambda: self._apply(token)):
            self.waiting = True
            self._buttons()
            return

        plan = self.pending

        try:
            sources, targets = plan.resolve(self.data)
        except ValueError as exc:
            self._report_skipped(exc)
            self._continue(token)
            return

        self._prepare_in_background(token, plan, sources, targets)

    def _prepare_in_background(self, token, plan, sources, targets, **resolution):
        """Prepare the manipulation in a worker; commit on the GUI thread.

        prepare() works on an isolated candidate; commit() refuses it if
        the data changed in the meantime.
        """
        from catan.gui.data.curation_actions import prepare

        data = self.data

        def work():
            try:
                return "prepared", prepare(data, plan, **resolution)
            except OccupiedDestination as exc:
                return "occupied", exc
            except ValueError as exc:
                return "invalid", exc

        self.waiting = True
        self._buttons()
        self.status.setText(
            "Preparing: " + self._label(plan.kind, sources, targets) + " …"
        )

        self._prepare_task = self.state.tasks.start(
            "model update",
            f"Curator: prepare {plan.kind}",
            work,
            on_result=lambda result: self._prepared(token, plan, sources, targets, result),
        )

    def _on_prepare_stopped(self, group, task_id):
        if task_id is None or task_id != self._prepare_task:
            return

        self._prepare_task = None
        self.waiting = False
        self._finish("Stopped: preparation was cancelled or failed")

    def _prepared(self, token, plan, sources, targets, result):
        self._prepare_task = None
        self.waiting = False

        if token != self.run_id or self.queue is None or self.pending is not plan:
            return

        outcome, value = result

        if outcome == "occupied":
            choice = self._conflict_choice(value.slots)

            # The run may have been stopped while the dialog was open.
            if token != self.run_id or self.queue is None:
                return

            if choice not in ("exclude", "detach"):
                self._skip()
                return

            # Resolve and validate again after the dialog.
            self._prepare_in_background(
                token,
                plan,
                sources,
                targets,
                detach_occupied=choice == "detach",
                exclude_occupied=choice == "exclude",
            )
            return

        if outcome == "invalid":
            self._report_skipped(value)
            self._continue(token)
            return

        prepared = value

        try:
            prepared.validation = self.data.validate_prepared_manipulation(
                prepared, self.spec
            )
            prepared.validation.require_pass()

            self.committing = True
            try:
                self.data.commit_manipulation(prepared)

                result = self.controller.current_result
                if result is not None:
                    result.applied(
                        plan.kind,
                        sources,
                        targets,
                        prepared,
                        self.data,
                    )
            finally:
                self.committing = False

            self.controller._results_note = (
                "Data changed since evaluation; "
                "completed candidates were removed."
                + (
                    " Preset also changed."
                    if self.controller._results_note.startswith("Preset")
                    else ""
                )
            )
            self.controller._refresh_results()

            self._highlight_outcome(prepared, [(sources, targets)])
            self.reports.append(("applied", plan.kind))

        except ValueError as exc:
            self._report_skipped(exc)

        except Exception as exc:
            self.reports.append(("failed", f"{type(exc).__name__}: {exc}"))
            self._finish("Stopped after unexpected error")
            QMessageBox.warning(
                self.window(),
                "Manipulation failed",
                str(exc),
            )
            return

        self._continue(token)

    def _report_skipped(self, exc):
        self.reports.append(("skipped", str(exc)))

        if self.interactive:
            QMessageBox.warning(
                self.window(),
                "Manipulation not applied",
                str(exc),
            )

    def _continue(self, token):
        self.pending = None
        self._buttons()
        QTimer.singleShot(0, lambda: self._advance(token))

    def _highlight_outcome(self, prepared, endpoints):
        """Highlight results (targets) and the plans' sources at their new
        identities, following the highlight selector (roles, colours)."""

        def mapped(component):
            key = (component.neuron_id, component.session_id)

            if key in prepared.component_map:
                current = prepared.component_map[key]
                return None if current is None else NeuronComponent(*current)

            neuron = prepared.row_map.get(component.neuron_id)
            return None if neuron is None else NeuronComponent(neuron, component.session_id)

        results = {
            component
            for values in prepared.replacements.values()
            for component in values
        }

        if not results:
            # Neuron merges have no result footprints.
            results = set(prepared.affected_after)

        sources = {
            current
            for plan_sources, _ in endpoints
            for component in plan_sources
            if (current := mapped(component)) is not None
        } - results

        self.controller._highlight_endpoints(sources, results)

    def _run_batch(self, plans):
        """Apply all plans to one candidate and commit them together."""
        kind = self.kind.currentData()
        token = self.run_id

        try:
            batch = self.data.start_manipulation_batch(kind)
        except ValueError as exc:
            self.reports.append(("skipped", str(exc)))
            self._finish("Not started")
            return

        self._buttons()

        self._batch_running = True
        self._batch_stop = False
        self.progress.setRange(0, len(plans) + 1)
        self.progress.show()

        stopped = False

        try:
            for index, plan in enumerate(plans):
                self._batch_progress(
                    index, f"Applying manipulations: %v / {len(plans)}"
                )

                if self._batch_stop or token != self.run_id:
                    stopped = True
                    break

                try:
                    slots = batch.conflicts(plan)
                    occupied = None

                    if slots:
                        if self.conflict.currentData() == "ask" and self._policy is None:
                            # Context for the decision: the committed data.
                            self._show(plan, focus=True)

                        choice = self._conflict_choice(slots, batch=True)

                        if choice == "stop":
                            stopped = True
                            break

                        if choice == "skip":
                            self.reports.append(
                                ("skipped", "Conflict: " + str(OccupiedDestination(slots, kind)))
                            )
                            continue

                        occupied = choice

                    batch.apply(plan, occupied=occupied)
                    self.reports.append(("applied", kind))

                except ValueError as exc:
                    self.reports.append(("skipped", str(exc)))

            if token != self.run_id:
                # Data changed outside this run; nothing was committed.
                return

            if not batch.applied:
                self._finish("Stopped" if stopped else "Finished")
                return

            self.stop.setEnabled(False)
            self._batch_progress(
                len(plans),
                f"Rebuilding union footprints and committing {len(batch.applied)} "
                "manipulations…",
            )

            prepared = batch.finish()
            prepared.validation = self.data.validate_prepared_manipulation(
                prepared, self.spec
            )
            prepared.validation.require_pass()

            self.committing = True
            try:
                self.data.commit_manipulation(prepared)

                result = self.controller.current_result
                if result is not None:
                    result.applied_batch(
                        [(plan.kind, sources, targets) for plan, sources, targets in batch.applied],
                        prepared,
                        self.data,
                    )
            finally:
                self.committing = False

        except Exception as exc:
            self.reports = [
                (outcome, text) for outcome, text in self.reports if outcome != "applied"
            ]
            self.reports.append(("failed", f"{type(exc).__name__}: {exc}"))
            self._finish("Stopped: nothing was applied")
            QMessageBox.warning(
                self.window(),
                "Manipulations not applied",
                f"{exc}\n\nNo changes were made to the data.",
            )
            return

        finally:
            self._batch_running = False
            self.progress.hide()
            self.stop.setEnabled(True)

        self.controller._results_note = (
            "Data changed since evaluation; completed candidates were removed."
        )
        self.controller._refresh_results()

        self._highlight_outcome(
            prepared, [(sources, targets) for _, sources, targets in batch.applied]
        )

        self._finish("Stopped" if stopped else "Finished")

    def _on_stop(self):
        if self._batch_running:
            self._batch_stop = True
            self.progress.setFormat("Stopping after the current manipulation…")
            return

        self._finish("Stopped")

    def _batch_progress(self, value, text):
        self.progress.setValue(value)
        self.progress.setFormat(text)
        # Keep the window responsive (Stop, conflict context) during the run.
        QApplication.processEvents()

    def _skip(self):
        if self.pending is not None and not self.waiting:
            self.reports.append(("skipped", "Skipped by user"))
            self.pending = None

            QTimer.singleShot(
                0,
                lambda token=self.run_id: self._advance(token),
            )

    def _finish(self, message):
        if self._prepare_task is not None:
            task, self._prepare_task = self._prepare_task, None
            self.state.tasks.cancel(task)

        self.run_id += 1
        self.queue = None
        self.pending = None
        self.waiting = False

        counts = {
            key: sum(kind == key for kind, _ in self.reports)
            for key in (
                "applied",
                "skipped",
                "failed",
            )
        }

        summary = (
            message
            + ": "
            + ", ".join(f"{value} {key}" for key, value in counts.items())
        )

        problems = [text for kind, text in self.reports if kind != "applied"]

        self.status.setText(
            summary + ("\nLast issue: " + problems[-1] if problems else "")
        )
        self.status.setToolTip(
            "\n".join(f"{kind}: {text}" for kind, text in self.reports)
        )

        self.refresh_candidates()
        self.controller._update_filter_validity()

    def _data_changed(self, *_):
        if self.committing:
            # _apply updates retained identities after commit completes.
            return

        result = self.controller.current_result
        if result is not None:
            result.data_changed(self.data)
            self.controller._results_note = "Data changed since evaluation."

        if self.queue is not None:
            self._finish("Stopped: data changed outside this batch")
        else:
            self.refresh_candidates()

        self.controller._refresh_results()
        self.controller._update_evidence_highlight()
