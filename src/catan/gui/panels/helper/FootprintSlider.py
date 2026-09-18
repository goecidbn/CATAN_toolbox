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
)

from catan.gui.structures import NeuronComponent


class FootprintSliderController(QWidget):
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

        self.footprint_slider.sliderMoved.connect(self._on_slider_moved)
        self.footprint_id_prev.clicked.connect(self.on_prev_footprint)
        self.footprint_id_next.clicked.connect(self.on_next_footprint)
        self.footprint_edit.editingFinished.connect(self._on_neuron_edit_return)

        self.state.focused_component_changed.connect(self.adjust_id)
        self.state.selected_components_changed.connect(self.update_setup)

        self.update_setup()
        # self.footprint_edit.editingFinished.connect(self.on_neuron_edit_return)

        # selector_layout = self.controls.build_footprint_selector()
        # self.section.x_options_layout.addLayout(selector_layout)
        # return selector_layout

    def update_setup(self):
        self.set_id_range()
        self.adjust_id()

    def on_prev_footprint(self):

        selected = self.state.selected_components

        # Global neuron navigation
        if not selected:

            component = NeuronComponent(
                self.state.current_session_id,
                self.state.assignments.shape[0] - 1,
            )

        elif len(selected) == 1:

            current_id = (
                self.state.focused_component.neuron_id
                if self.state.focused_component is not None
                else selected[0].neuron_id
            )

            neuron_id = np.mod(current_id - 1, self.state.assignments.shape[0])

            component = NeuronComponent(self.state.current_session_id, int(neuron_id))

        # Navigate within multi-selection
        else:

            current_index = self.state.selected_component_index(
                self.state.focused_component
            )

            if current_index is None:
                index = len(selected) - 1
            else:
                index = np.mod(current_index - 1, len(selected))

            component = self._resolve_selected_component(selected[int(index)])

        self.state.focused_component = component

    def on_next_footprint(self):

        selected = self.state.selected_components

        # Global neuron navigation
        if not selected:

            component = NeuronComponent(
                self.state.current_session_id,
                0,
            )

        elif len(selected) == 1:

            current_id = (
                self.state.focused_component.neuron_id
                if self.state.focused_component is not None
                else selected[0].neuron_id
            )

            neuron_id = np.mod(current_id + 1, self.state.assignments.shape[0])

            component = NeuronComponent(self.state.current_session_id, int(neuron_id))

        # Navigate within multi-selection
        else:

            current_index = self.state.selected_component_index(
                self.state.focused_component
            )

            if current_index is None:
                index = 0
            else:
                index = np.mod(current_index + 1, len(selected))

            component = self._resolve_selected_component(selected[int(index)])

        self.state.focused_component = component

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
                session_id=self.state.current_session_id, neuron_id=int(value)
            )

        self.state.focused_component = component

    # def on_neuron_slider_changed(self):

    #     value = int(self.footprint_slider.value())

    #     selected = self.state.selected_components
    #     if selected and len(selected) > 1:
    #         component = self._resolve_selected_component(selected[value])

    #     else:
    #         component = NeuronComponent(
    #             session_id=self.state.current_session_id, neuron_id=value
    #         )

    #     self.state.focused_component = component

    #     self.adjust_id()

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

    # def adjust_id(self):
    #     """
    #     Adjust visually the slider and entry to show the given cluster ID.
    #     """

    #     focused = self.state.focused_component
    #     selected = self.state.selected_components

    #     if focused is None:
    #         slider_id = None

    #     elif selected and len(selected) > 1:
    #         slider_id = self.state.selected_component_index(focused)

    #     else:
    #         # In global-navigation mode the slider value is
    #         # the neuron ID itself.
    #         slider_id = int(focused.neuron_id)

    #     self.footprint_slider.setEnabled(True)
    #     self.footprint_edit.setEnabled(True)
    #     self.footprint_id_next.setEnabled(True)
    #     self.footprint_id_prev.setEnabled(True)
    #     if slider_id is None:
    #         self.footprint_edit.setText("")
    #         self.footprint_slider.setValue(0)
    #         return

    #     if slider_id != int(self.footprint_slider.value()):
    #         self.footprint_slider.setValue(slider_id)

    #     self.footprint_edit.setText(str(focused.neuron_id))

    def _resolve_selected_component(
        self,
        component: NeuronComponent,
    ) -> NeuronComponent:

        if component.session_id is not None:
            return component

        return NeuronComponent(
            session_id=self.state.current_session_id,
            neuron_id=component.neuron_id,
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
            session_id=self.state.current_session_id, neuron_id=neuron_id
        )

        self.state.focused_component = component
