"""Curator manipulation controls and sequential candidate review."""

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import replace

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
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
    ManipulationSpec,
    ManipulationPlan,
    OccupiedDestination,
)


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
        ):
            self.kind.addItem(label, value)

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

        row = QHBoxLayout()

        self.current = QPushButton("Apply current")
        self.all = QPushButton("Apply all selected")
        self.next = QPushButton(r"Apply \& next")
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

        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.kind.currentIndexChanged.connect(self._settings_changed)
        self.confirm.toggled.connect(self._settings_changed)
        self.candidate.currentIndexChanged.connect(self._preview_current)
        self.candidate.currentIndexChanged.connect(
            lambda *_: self._update_selector_layout()
        )

        self.current.clicked.connect(lambda: self._start(False))
        self.all.clicked.connect(lambda: self._start(True))
        self.next.clicked.connect(lambda: self._apply(self.run_id))
        self.skip.clicked.connect(self._skip)
        self.stop.clicked.connect(lambda: self._finish("Stopped"))

        self.state.focused_component_changed.connect(self.refresh_candidates)
        self.state.selected_components_changed.connect(self.refresh_candidates)
        self.state.data_changed.connect(self._data_changed)

        self.sync()

    def sync(self):
        spec = self.controller.root_filter.manipulation or ManipulationSpec()

        for widget in (self.kind, self.confirm):
            widget.blockSignals(True)

        self.kind.setCurrentIndex(self.kind.findData(spec.kind))
        self.confirm.setChecked(spec.confirm_each)

        for widget in (self.kind, self.confirm):
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
        )

        self.controller.filter_store.mark_working_dirty()
        self.controller._refresh_preset_controls()

        self.refresh_candidates()

    def result_ready(self):
        self.result_version = self.state.data_version
        self.refresh_candidates()

    def _prepare_with_confirmation(self, plan):
        try:
            return self.data.prepare_manipulation(plan)

        except OccupiedDestination as exc:
            dialog = QMessageBox(self.window())
            dialog.setIcon(QMessageBox.Icon.Warning)
            dialog.setWindowTitle("Destination already contains a footprint")
            dialog.setText(str(exc))
            dialog.setInformativeText(
                "Apply anyway moves each existing footprint to a "
                "separate, included neuron, then applies this "
                "manipulation. No footprint is overwritten or "
                "excluded. Reject leaves this candidate unchanged."
            )

            apply_button = dialog.addButton(
                "Apply anyway",
                QMessageBox.ButtonRole.AcceptRole,
            )
            reject_button = dialog.addButton(
                "Reject",
                QMessageBox.ButtonRole.RejectRole,
            )
            dialog.setDefaultButton(reject_button)
            dialog.exec()

            if dialog.clickedButton() is not apply_button:
                return None

            # Resolve and validate again after the dialog.
            return self.data.prepare_manipulation(
                plan,
                detach_occupied=True,
            )

    def _plans(self):
        result = self.controller.current_result

        kind = self.kind.currentData()
        if result is None or kind == "none":
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

        self.controller._highlight_endpoints(set(sources), set(targets))

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

        if running:
            self.controller.menu.evaluate_button.setEnabled(False)

        for widget in (
            self.kind,
            self.confirm,
            self.candidate,
        ):
            widget.setEnabled(not running)

        self.current.setVisible(not running)
        self.all.setVisible(not running)

        self.current.setEnabled(
            not running and self.candidate.currentData() is not None
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

    def _start(self, all_selected):
        if self.queue is not None:
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

        self.review = all_selected and self.confirm.isChecked()

        self.generation = self.controller._filter_generation
        self.root = self.controller.root_filter

        self.spec = deepcopy(
            self.root.manipulation or ManipulationSpec(kind=self.kind.currentData())
        )

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

        self.waiting = False
        plan = self.pending

        try:
            sources, targets = plan.resolve(self.data)
            prepared = self._prepare_with_confirmation(plan)

            if prepared is None:
                self._skip()
                return

            if token != self.run_id or self.queue is None:
                return

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

            outcome = {
                component
                for values in prepared.replacements.values()
                for component in values
            }
            if not outcome:
                outcome = set(prepared.affected_after)

            self.state.update_highlighted_components(
                self.controller._display_components(outcome) or None
            )
            self.reports.append(("applied", plan.kind))

        except ValueError as exc:
            self.reports.append(("skipped", str(exc)))
            QMessageBox.warning(
                self.window(),
                "Manipulation not applied",
                str(exc),
            )

        except Exception as exc:
            self.reports.append(("failed", f"{type(exc).__name__}: {exc}"))
            self._finish("Stopped after unexpected error")
            QMessageBox.warning(
                self.window(),
                "Manipulation failed",
                str(exc),
            )
            return

        self.pending = None
        QTimer.singleShot(0, lambda: self._advance(token))

    def _skip(self):
        if self.pending is not None and not self.waiting:
            self.reports.append(("skipped", "Skipped by user"))
            self.pending = None

            QTimer.singleShot(
                0,
                lambda token=self.run_id: self._advance(token),
            )

    def _finish(self, message):
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
