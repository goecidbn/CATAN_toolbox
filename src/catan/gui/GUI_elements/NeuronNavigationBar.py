import numpy as np
from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSlider,
    QLabel,
    QWidget,
    QCheckBox,
    QDoubleSpinBox,
)

from catan.core.structures import NeuronComponent
from catan.tracking.structures import ReviewStatus


class NeuronNavigationBar(QWidget):
    ### SLIDER CONTROLS ###

    def __init__(self, parent):

        super().__init__(parent)

        self.state = parent.state
        self.data = parent.data

        self.editable = False
        self.current_id = 0

        selector_layout = QHBoxLayout(self)

        self.footprint_edit = QLineEdit()
        self.footprint_edit.setFixedWidth(60)

        self.footprint_id_prev = QPushButton("<")
        self.footprint_id_prev.setFixedWidth(30)
        self.footprint_slider = QSlider(Qt.Orientation.Horizontal)
        self.footprint_slider.setSingleStep(1)
        self.footprint_id_next = QPushButton(">")
        self.footprint_id_next.setFixedWidth(30)

        selector_layout.addWidget(QLabel("Neuron ID:"))
        selector_layout.addWidget(self.footprint_edit)
        selector_layout.addWidget(self.footprint_id_prev)
        selector_layout.addWidget(self.footprint_slider, stretch=1)
        selector_layout.addWidget(self.footprint_id_next)

        self.only_open_checkbox = QCheckBox("Only open")
        selector_layout.addWidget(self.only_open_checkbox)

        self.footprint_slider.sliderMoved.connect(self._on_slider_moved)
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

        self.state.focused_component_changed.connect(self._on_focus_changed)
        self.state.selected_components_changed.connect(self._on_selection_changed)

        self.update_setup()
        # self.footprint_edit.editingFinished.connect(self.on_neuron_edit_return)

        # selector_layout = self.controls.build_footprint_selector()
        # self.section.x_options_layout.addLayout(selector_layout)
        # return selector_layout

    def update_adj_radius(self, value: float):
        self.state.adjacency_radius = value

    def _on_selection_changed(self):
        self.update_setup()

    def _on_focus_changed(self):
        self.adjust_id()

    def _on_data_changed(self):
        self.update_setup()

    def update_setup(self):
        self.set_id_range()
        self.adjust_id()

    def on_prev_footprint(self):
        self.iterate_footprint(-1)

    def on_next_footprint(self):
        self.iterate_footprint(1)

    def iterate_footprint(self, step: int):

        neuron_ids = self.navigation_neuron_ids()

        # selected = self.state.selected_components
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

        if self.state.selected_components and len(self.state.selected_components) > 1:
            neuron_ids = np.asarray(
                [c.neuron_id for c in self.state.selected_components],
                dtype=int,
            )
        else:
            neuron_ids = np.arange(self.state.assignments.shape[0])

        # Never navigate excluded neurons normally.
        neuron_ids = neuron_ids[self.data.assignments.union.included[neuron_ids]]

        if self.only_open_checkbox.isChecked():
            neuron_ids = neuron_ids[
                self.data.assignments.review_status[neuron_ids] != ReviewStatus.REVIEWED
            ]

        return list(np.unique(neuron_ids))

    def set_id_range(
        self,
    ):

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

        selected = self.state.selected_components

        if selected and len(selected) > 1:
            maximum = len(selected) - 1
        else:
            maximum = self.state.assignments.shape[0] - 1

        self.footprint_slider.setRange(0, maximum)

    def _on_slider_moved(
        self,
        value: int,
    ):

        selected = self.state.selected_components

        # --------------------------------------------------
        # Multiple selected neurons:
        # slider position = index into selection.
        # --------------------------------------------------
        if selected and len(selected) > 1:
            component = self._resolve_selected_component(selected[int(value)])

        # --------------------------------------------------
        # Otherwise:
        # slider position = actual neuron ID.
        # --------------------------------------------------
        else:

            component = NeuronComponent(
                neuron_id=int(value), session_id=self.state.current_session_id
            )

        self.state.focused_component = component

    def adjust_id(
        self,
    ):
        """
        Synchronize slider position and neuron-ID editor
        from the current application state.

        This method never changes application state.
        """

        focused = self.state.focused_component
        selected = self.state.selected_components

        if focused is None:
            self.footprint_edit.clear()
            return

        # Text always displays actual neuron ID.
        self.footprint_edit.setText(str(int(focused.neuron_id)))

        # --------------------------------------------------
        # Multiple-selection mode:
        # slider represents position within selection.
        # --------------------------------------------------
        if selected and len(selected) > 1:
            index = self.state.selected_component_index(focused)
            if index is None:
                return

            slider_value = index

        # --------------------------------------------------
        # Global mode:
        # slider represents neuron ID.
        # --------------------------------------------------
        else:
            slider_value = int(focused.neuron_id)

        if self.footprint_slider.value() != slider_value:
            self.footprint_slider.setValue(slider_value)

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
