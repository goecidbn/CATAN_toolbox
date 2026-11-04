import numpy as np
from typing import Optional
from catan.core.changes import ChangeKind
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSlider,
    QLabel,
    QWidget,
    QDoubleSpinBox,
    QComboBox,
    QMessageBox,
)

from catan.core.structures import NeuronComponent
from catan.tracking.structures import ReviewStatus
from catan.gui.data.assignment_colors import assignment_label
from catan.gui.panels.helper.ReviewStatusFilter import ReviewStatusFilter


class NeuronNavigationBar(QWidget):
    ### SLIDER CONTROLS ###

    def __init__(self, parent):

        super().__init__(parent)

        self.state = parent.state
        self.data = parent.data

        self.editable = False
        self.current_id = 0

        # The active assignments are chosen in the main menu's assignments
        # section (one selector for navigation, display and processing).
        selector_layout = QHBoxLayout(self)

        self.neuron_count_label = QLabel()
        selector_layout.addWidget(self.neuron_count_label)

        self.footprint_edit = QLineEdit()
        self.footprint_edit.setFixedWidth(60)
        # Only a click gives it the keyboard focus (not Tab, not a hidden
        # button passing its focus on): it takes the arrow keys, which
        # otherwise step through the neurons.
        self.footprint_edit.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        self.footprint_id_prev = QPushButton("<")
        self.footprint_id_prev.setFixedWidth(30)
        self.footprint_slider = QSlider(Qt.Orientation.Horizontal)
        self.footprint_slider.setSingleStep(1)
        self.footprint_id_next = QPushButton(">")
        self.footprint_id_next.setFixedWidth(30)

        selector_layout.addWidget(QLabel("Focused ID:"))
        selector_layout.addWidget(self.footprint_edit)
        selector_layout.addWidget(self.footprint_id_prev)
        selector_layout.addWidget(self.footprint_slider, stretch=1)
        selector_layout.addWidget(self.footprint_id_next)

        self.review_filter = ReviewStatusFilter(self)
        self.review_filter.setToolTip(
            "Review states the arrows step through (in the selection, or "
            "through all neurons without one)."
        )
        self.review_filter.changed.connect(self.update_neuron_count_label)
        selector_layout.addWidget(self.review_filter)

        self.footprint_slider.sliderMoved.connect(self._on_slider_moved)
        self.footprint_slider.sliderReleased.connect(
            lambda: self._on_slider_moved(self.footprint_slider.sliderPosition())
        )

        self.footprint_id_prev.clicked.connect(self.on_prev_footprint)
        self.footprint_id_next.clicked.connect(self.on_next_footprint)
        self.footprint_edit.editingFinished.connect(self._on_neuron_edit_return)

        ## add adjacency radius control
        initial_adj_radius = 15
        self.adj_radius_spin = QDoubleSpinBox()
        self.adj_radius_spin.setDecimals(1)
        self.adj_radius_spin.setRange(0.0, 50.0)
        self.adj_radius_spin.setSingleStep(1.0)
        self.adj_radius_spin.setValue(initial_adj_radius)
        selector_layout.addWidget(QLabel("Neighborhood radius:"))
        selector_layout.addWidget(self.adj_radius_spin)

        self.adj_radius_spin.valueChanged.connect(
            lambda value: self.update_adj_radius(value)
        )
        # Shown in the display unit; the state keeps px.
        self._show_adj_radius()
        self.state.data_changed.connect(
            lambda event: self._show_adj_radius()
            if event.has(ChangeKind.DISPLAY_UNITS)
            else None
        )

        self.state.data_changed.connect(self._on_data_changed)
        self.state.focused_component_changed.connect(self._on_focus_changed)
        self.state.selected_components_changed.connect(self._on_selection_changed)

        self.update_setup()
        # self.footprint_edit.editingFinished.connect(self.on_neuron_edit_return)

        # selector_layout = self.controls.build_footprint_selector()
        # self.section.x_options_layout.addLayout(selector_layout)
        # return selector_layout

    def update_adj_radius(self, value: float):
        self.state.adjacency_radius = self.data.units.from_display(value, "distance")

    def _show_adj_radius(self):
        units = self.data.units
        blocked = self.adj_radius_spin.blockSignals(True)
        self.adj_radius_spin.setRange(0.0, units.to_display(50.0, "distance"))
        self.adj_radius_spin.setSuffix(f" {units.label('distance')}")
        self.adj_radius_spin.setValue(
            units.to_display(self.state.adjacency_radius, "distance")
        )
        self.adj_radius_spin.blockSignals(blocked)

    def _on_selection_changed(self):
        self.update_neuron_count_label()
        self.update_setup()

    def _on_focus_changed(self):
        self.adjust_id()

    def _on_data_changed(self):
        self.update_setup()

    def update_neuron_count_label(self):
        assignments = self.data.assignments

        if assignments is None:
            total = 0
        else:
            included = getattr(assignments.union, "included", None)
            total = (
                assignments.ids.shape[0]
                if included is None
                else int(np.count_nonzero(included))
            )

        if not self._in_selection_mode():
            text = f"Iterating {total} neurons"
        else:
            text = f"Selected {len(self.navigation_neuron_ids())} " f"/ {total} neurons"

        self.neuron_count_label.setText(text)

    def update_setup(self):
        self.update_neuron_count_label()
        self.set_id_range()
        self.adjust_id()

    def on_prev_footprint(self):
        self.iterate_footprint(-1)

    def on_next_footprint(self):
        self.iterate_footprint(1)

    def iterate_footprint(self, step: int):

        neuron_ids = self.navigation_neuron_ids()
        if not neuron_ids:
            self._report_nothing_left()
            return

        focused = self.state.focused_component

        # Global neuron navigation
        if focused is None:
            current_idx = 0
            new_idx = np.mod(min(0, step), len(neuron_ids))

        else:
            current_neuron_id = focused.neuron_id

            if focused.neuron_id not in neuron_ids:
                next_idxs = np.where(np.array(neuron_ids) >= focused.neuron_id)[0]
                if len(next_idxs) > 0:
                    new_idx = next_idxs[0]
                else:
                    new_idx = 0
            else:
                current_idx = neuron_ids.index(focused.neuron_id) or 0
                new_idx = np.mod(current_idx + step, len(neuron_ids))

        self.state.focused_component = NeuronComponent(
            neuron_ids[new_idx], self.state.current_session_id
        )

    def navigation_neuron_ids(self) -> list[int]:
        assignments = self.data.assignments

        if assignments is None or assignments.union is None:
            return []

        included = np.asarray(
            assignments.union.included,
            dtype=bool,
        )

        # Use the current data object rather than a potentially older
        # state.assignments array during a GUI update.
        count = min(
            assignments.ids.shape[0],
            included.size,
        )

        if count == 0:
            return []

        if self._in_selection_mode():
            neuron_ids = np.asarray(
                [component.neuron_id for component in self._navigation_selection()],
                dtype=int,
            )
        else:
            neuron_ids = np.arange(count, dtype=int)

        neuron_ids = neuron_ids[(neuron_ids >= 0) & (neuron_ids < count)]

        neuron_ids = neuron_ids[included[neuron_ids]]
        # Rows without footprints (left by older exclusions) show nothing.
        neuron_ids = neuron_ids[np.any(assignments.ids[neuron_ids] >= 0, axis=1)]

        statuses = self.review_filter.visible_statuses
        if len(statuses) < len(ReviewStatus):
            status = np.asarray(assignments.review_status)
            neuron_ids = neuron_ids[neuron_ids < len(status)]
            neuron_ids = neuron_ids[
                np.isin(status[neuron_ids], [int(s) for s in statuses])
            ]

        return np.unique(neuron_ids).tolist()

    def _in_selection_mode(self) -> bool:
        """Step through the selection: a selected set (preset results,
        statistic picks; also when one neuron is left) or several clicked
        neurons. Otherwise through all neurons."""
        selected = self._navigation_selection()
        return bool(selected) and (self.state.selection_is_set or len(selected) > 1)

    def _report_nothing_left(self):
        statuses = self.review_filter.visible_statuses
        where = "selected neurons" if self._in_selection_mode() else "neurons"

        if len(statuses) < len(ReviewStatus):
            labels = ", ".join(
                status.label for status in ReviewStatus if status in statuses
            )
            detail = f" with review status: {labels or 'none'}"
        else:
            detail = ""

        QMessageBox.information(
            self.window(),
            "Nothing left to step through",
            f"No more {where}{detail} (excluded neurons are skipped).",
        )

    def _navigation_selection(self) -> list[NeuronComponent]:
        """One representative per selected neuron, preserving selection order."""
        by_neuron = {}
        for component in self.state.selected_components or ():
            by_neuron.setdefault(int(component.neuron_id), component)
        return list(by_neuron.values())

    def set_id_range(self):
        enabled = not (
            self.data is None
            or self.state.current_session_id is None
            or self.state.assignments.shape[0] == 0
        )

        self.footprint_slider.setEnabled(enabled)
        self.footprint_edit.setEnabled(enabled)
        self.footprint_id_next.setEnabled(enabled)
        self.footprint_id_prev.setEnabled(enabled)

        if not enabled:
            return

        selected = self._navigation_selection()

        if self._in_selection_mode():
            maximum = len(selected) - 1
        else:
            maximum = self.state.assignments.shape[0] - 1

        self.footprint_slider.setRange(0, maximum)

    def _on_slider_moved(
        self,
        value: int,
    ):
        # During a request, confirm only the final position on release.
        if (
            self.state.current_request is not None
            and self.footprint_slider.isSliderDown()
        ):
            return

        selected = self._navigation_selection()

        # --------------------------------------------------
        # Multiple selected neurons:
        # slider position = index into selection.
        # --------------------------------------------------
        if self._in_selection_mode():
            if not 0 <= int(value) < len(selected):
                self.update_setup()
                return

            component = self._resolve_selected_component(selected[int(value)])

        # --------------------------------------------------
        # Otherwise:
        # slider position = actual neuron ID.
        # --------------------------------------------------
        else:

            component = NeuronComponent(
                neuron_id=int(value), session_id=self.state.current_session_id
            )

        assignments = self.data.assignments

        if (
            component is None
            or assignments is None
            or not 0 <= component.neuron_id < assignments.ids.shape[0]
        ):
            self.update_setup()
            return

        self.state.focused_component = component
        self.adjust_id()

    def adjust_id(
        self,
    ):
        """
        Synchronize slider position and neuron-ID editor
        from the current application state.

        This method never changes application state.
        """

        focused = self.state.focused_component
        selected = self._navigation_selection()

        if focused is None:
            self.footprint_edit.clear()
            return

        # Text always displays actual neuron ID.
        self.footprint_edit.setText(str(int(focused.neuron_id)))

        # --------------------------------------------------
        # Multiple-selection mode:
        # slider represents position within selection.
        # --------------------------------------------------
        if self._in_selection_mode():
            index = next(
                (
                    i
                    for i, component in enumerate(selected)
                    if component.neuron_id == focused.neuron_id
                ),
                None,
            )
            if index is None:
                return

            slider_value = index

        # --------------------------------------------------
        # Global mode:
        # slider represents neuron ID.
        # --------------------------------------------------
        else:
            slider_value = int(focused.neuron_id)

        was_blocked = self.footprint_slider.blockSignals(True)
        try:
            self.footprint_slider.setSliderPosition(slider_value)
            self.footprint_slider.setValue(slider_value)
        finally:
            self.footprint_slider.blockSignals(was_blocked)

    def _resolve_selected_component(
        self,
        component: NeuronComponent,
    ) -> NeuronComponent:

        if component.session_id is not None:
            return component

        return NeuronComponent(
            neuron_id=component.neuron_id,
            session_id=self.state.current_session_id,
        )

    def _on_neuron_edit_return(
        self,
    ):
        try:
            self._focus_typed_neuron()
        finally:
            # Arrow keys step through neurons again right away.
            self.footprint_edit.clearFocus()

    def _focus_typed_neuron(self):

        try:
            neuron_id = int(self.footprint_edit.text())
        except ValueError:
            self.adjust_id()
            return

        if not (0 <= neuron_id < self.state.assignments.shape[0]):
            self.adjust_id()
            return

        component = NeuronComponent(
            neuron_id=neuron_id, session_id=self.state.current_session_id
        )

        self.state.focused_component = component
        self.adjust_id()
