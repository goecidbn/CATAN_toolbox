from dataclasses import dataclass
from time import time
import numpy as np
from PySide6.QtCore import QObject, Signal
from typing import Literal, Tuple, Optional, List
import logging

from catan.gui.background_tasks import TaskManager


@dataclass(frozen=True, slots=True)
class NeuronComponent:
    session_id: int
    neuron_id: int

    @property
    def id(self) -> Tuple[int, int]:
        return self.session_id, self.neuron_id

    def __iter__(self):
        yield self.session_id
        yield self.neuron_id


class AppState(QObject):

    assignments: np.ndarray  # shape (n_clusters, n_sessions)
    tracking_loaded: bool = False

    # Signals for things that can change

    session_color_changed = Signal(int, object)  # session_id, color_value
    session_offset_changed = Signal(int, int)  # session_id, offset_value

    current_session_changed = Signal(int)

    selected_components_changed = Signal()
    focused_component_changed = Signal()
    highlighted_component_changed = Signal()

    # compare_mode_changed = Signal(str)
    plot_update_required = Signal()
    session_toggled = Signal(tuple)

    data_changed = Signal(object)
    data_version: int = 0

    def __init__(self):
        super().__init__()
        # self._current_display = 0
        self._current_session_id = None

        self._session_colors = []
        self._session_active = []
        self._session_offsets = []

        self._selected_components: Optional[List[NeuronComponent]] = None
        self._focused_component: Optional[NeuronComponent] = None
        self._highlighted_component: Optional[NeuronComponent] = None
        self.current_job = None

        self.tasks = TaskManager()

        self.logger = logging.getLogger("GUI")
        self.set_logging_level("ERROR")
        # logging.basicConfig(level=getattr(logging, self.logging_level))

        self.time_ref = None

    def set_logging_level(self, level: str):
        self.logging_level = level
        self.logger.setLevel(getattr(logging, self.logging_level))

    def timeit(self, msg=None):
        if msg is not None and self.time_ref is not None:
            self.logger.debug(
                "time for %s: %3.2f ms" % (msg, (time() - self.time_ref) * 10**3)
            )

        if msg is None:
            self.logger.debug("--- Timer reset ---")

        self.time_ref = time()

    @property
    def app_mode(self):
        pass

    @property
    def session_offsets(self):
        return self._session_offsets

    @property
    def session_offset(self):
        return (
            self._session_offsets[self.current_session_id]
            if self.current_session_id is not None
            else None
        )

    @session_offset.setter
    def session_offset(self, input: Tuple[int, int]):
        session_id, offset = input

        if session_id == len(self._session_offsets):
            self._session_offsets.append(offset)
        elif session_id < len(self._session_offsets):
            self._session_offsets[session_id] = offset
        else:
            raise IndexError(
                f"Session ID {session_id} is out of bounds for session_offsets list of length {len(self._session_offsets)}"
            )
        self.session_offset_changed.emit(session_id, offset)

    @property
    def session_colors(self):
        return self._session_colors

    @property
    def session_color(self):
        return (
            self._session_colors[self.current_session_id]
            if self.current_session_id is not None
            else None
        )

    @session_color.setter
    def session_color(self, input: Tuple[int, Tuple[float, float, float]]):
        session_id, color = input
        if session_id == len(self._session_colors):
            self._session_colors.append(color)
        elif session_id < len(self._session_colors):
            self._session_colors[session_id] = color
        else:
            raise IndexError(
                f"Session ID {session_id} is out of bounds for session_colors list of length {len(self._session_colors)}"
            )
        self.session_color_changed.emit(session_id, color)

    @property
    def session_active(self):
        return self._session_active

    @session_active.setter
    def session_active(self, input):
        session_id, active = input
        # print(f"Setting session {session_id} active state to {active}")
        current_active = self._session_active[session_id]
        if current_active == active:
            return  # No change, do nothing

        self._session_active[session_id] = active
        self.session_toggled.emit(input)

    @property
    def session_added(self):
        pass

    @session_added.setter
    def session_added(self, session_id: int):
        # print(f"[STATE] Adding new session {session_id} to state...")
        self._session_active.append(True)
        self.data_changed.emit(("session", session_id))

    # --- current session ---
    @property
    def current_session_id(self) -> Optional[int]:
        return self._current_session_id

    @current_session_id.setter
    def current_session_id(self, s: int):
        if s != self._current_session_id:
            self._current_session_id = s
            self.current_session_changed.emit(s)
            self.session_active = (s, True)

    # --- current neuron ---
    """ 
        Logic for subselection of neurons
    """

    @property
    def selected_components(self) -> Optional[List[NeuronComponent]]:
        if self._selected_components is None:
            return None

        return self._selected_components

    def update_selected_components(
        self,
        components: Optional[NeuronComponent | List[NeuronComponent]],
        modifiers=[],
    ):
        if isinstance(components, NeuronComponent):
            # print(f"Casting input component to list: {components}")
            components = [components]

        if "Control" in modifiers:
            if components is None:
                return

            ## mode for adding/removing single components
            if self._selected_components is None:
                self.update_selected_components(components)
                return

            selected_components = self._selected_components.copy()

            for component in components:
                try:
                    ## if component is already in current selection, remove it (toggle behavior)
                    idx = selected_components.index(component)
                    selected_components.pop(idx)
                except ValueError:
                    ## if component is not in current selection, add it
                    selected_components.append(component)

            if len(selected_components) == 0:
                selected_components = None

        else:
            ## mode for replacing the current selection with a new one
            selected_components = components

        if (
            selected_components is not None
            and self.focused_component not in selected_components
        ):
            self._focused_component = selected_components[-1]
        elif selected_components is None:
            self._focused_component = None

        self._selected_components = (
            sorted(selected_components, key=lambda c: c.neuron_id)
            if selected_components is not None
            else None
        )
        self.selected_components_changed.emit()

    @property
    def focused_component(self) -> Optional[NeuronComponent]:
        return self._focused_component

    @focused_component.setter
    def focused_component(self, component: Optional[NeuronComponent]):

        self.logger.debug(f"Setting focused component to {component}")
        self._focused_component = component

        if component is None:
            self.focused_component_changed.emit()
            return

        if (
            self.selected_components is not None
            and component in self.selected_components
        ):
            self.focused_component_changed.emit()
            return

        self.update_selected_components(component)

    @property
    def highlighted_component(self) -> Optional[NeuronComponent]:
        return self._highlighted_component

    @highlighted_component.setter
    def highlighted_component(self, component: Optional[NeuronComponent]):
        self._highlighted_component = component
        self.highlighted_component_changed.emit()

    def get_footprint_from_component(self, component: NeuronComponent) -> Optional[int]:

        # print(f"Translate component to footprint ID: {component}")
        if component is None:
            return None

        if self.assignments is None:
            print("Warning: trying to get footprint but assignments are not loaded")
            return None

        if component.session_id is None or component.neuron_id is None:
            print("Warning: trying to get footprint with invalid neuron_id:", component)
            return None

        return self.assignments[component.neuron_id, component.session_id]

    def get_component_from_footprint(
        self, footprint_id: int, session_id: Optional[int] = None
    ) -> Optional[NeuronComponent]:
        if self.assignments is None:
            # print("Warning: trying to get neuron_id but assignments are not loaded")
            return None

        if session_id is None:
            session_id = self.current_session_id
        if session_id is None:
            return None

        neuron_ids = np.where(self.assignments[:, session_id] == footprint_id)[0]
        if len(neuron_ids) == 0:
            return None
        elif len(neuron_ids) > 1:
            raise ValueError(
                f"Error: multiple clusters found for footprint ID {footprint_id} in session {session_id}. This shouldnt happen, check your assignments array."
            )
        neuron_id = int(neuron_ids[0])

        return NeuronComponent(session_id, neuron_id)


def equal_neurons(id1: NeuronComponent, id2: NeuronComponent):
    return id1.session_id == id2.session_id and id1.neuron_id == id2.neuron_id
