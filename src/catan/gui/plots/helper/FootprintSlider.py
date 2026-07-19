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

# from data.state import NeuronComponent


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

        self.footprint_slider.sliderMoved.connect(self.adjust_id)
        self.footprint_slider.sliderReleased.connect(self.on_neuron_slider_changed)
        self.footprint_id_prev.clicked.connect(self.on_prev_footprint)
        self.footprint_id_next.clicked.connect(self.on_next_footprint)

        self.update_setup()
        # self.footprint_edit.editingFinished.connect(self.on_neuron_edit_return)

        # selector_layout = self.controls.build_footprint_selector()
        # self.section.x_options_layout.addLayout(selector_layout)
        # return selector_layout

    def update_setup(self):
        self.set_id_range()
        self.adjust_id()

    def on_prev_footprint(self):
        if self.state.selected_components is None:
            return

        if self.state.focused_component is None:
            self.state.focused_component = self.state.selected_components[-1]
        else:
            idx = self.state.selected_components.index(self.state.focused_component)
            idx = np.mod(idx - 1, len(self.state.selected_components))
            self.state.focused_component = self.state.selected_components[idx]

        self.adjust_id()

    def on_next_footprint(self):
        # print(f"Current selected neuron index: {self.state.current_neuron_idx}")
        if self.state.selected_components is None:
            return

        if self.state.focused_component is None:
            self.state.focused_component = self.state.selected_components[0]
        else:
            idx = self.state.selected_components.index(self.state.focused_component)
            idx = np.mod(idx + 1, len(self.state.selected_components))
            self.state.focused_component = self.state.selected_components[idx]

        self.adjust_id()

    def set_id_range(self):
        """
        Configure the slider's min/max from the state's cluster IDs.
        """

        if (
            self.data is None
            or self.state.current_session_id is None
            or self.state.selected_components is None
        ):
            self.footprint_slider.setEnabled(False)
            self.footprint_edit.setEnabled(False)
            self.footprint_id_next.setEnabled(False)
            self.footprint_id_prev.setEnabled(False)
            return

        min_id = 0
        # max_id = self.data.current_session.nA
        # max_id = self.state.assignments.shape[0] - 1
        max_id = len(self.state.selected_components) - 1
        self.footprint_slider.setMinimum(min_id)
        self.footprint_slider.setMaximum(max_id)
        self.footprint_slider.setValue(0)

    def on_neuron_slider_changed(self):
        # if value in self.state.clusters:
        # If skip-processed is enabled, you might jump to next unprocessed here.
        # value = int(self.footprint_slider.value())
        self.state.focused_component = self.state.selected_components[
            int(self.footprint_slider.value())
        ]

    # def on_neuron_edit_return(self):
    #     try:
    #         value = int(self.footprint_edit.text())
    #     except ValueError:
    #         return
    #     self.state.current_neuron = value
    #     self.adjust_id(value)

    def adjust_id(self):
        """
        Adjust visually the slider and entry to show the given cluster ID.
        """

        if self.state.selected_components is None:
            self.footprint_slider.setEnabled(False)
            self.footprint_edit.setEnabled(False)
            self.footprint_id_next.setEnabled(False)
            self.footprint_id_prev.setEnabled(False)
            return
        # if id is None:
        if self.state.focused_component is None:
            id = None
        else:
            id = self.state.selected_components.index(self.state.focused_component)

        self.footprint_slider.setEnabled(True)
        self.footprint_edit.setEnabled(self.editable)
        self.footprint_id_next.setEnabled(True)
        self.footprint_id_prev.setEnabled(True)
        if id is None:
            self.footprint_edit.setText("")
            self.footprint_slider.setValue(0)
            return
        elif id != int(self.footprint_slider.value()):
            # self.footprint_slider.setValue(neuron.cluster_id)
            self.footprint_slider.setValue(id)

        self.footprint_edit.setText(str(self.state.focused_component.neuron_id))
