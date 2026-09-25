from dataclasses import dataclass
from time import time
import numpy as np
from PySide6.QtCore import QObject, Signal, QSettings
from typing import Literal, Tuple, Optional, List
import logging


from catan.core.structures import NeuronComponent
from .request_handler import RequestHandler

from catan.gui.background_tasks import TaskManager
from catan.core.structures.load_config import LoadConfigManager

from platformdirs import user_config_dir
from pathlib import Path


class AppState(QObject):

    assignments: np.ndarray  # shape (n_clusters, n_sessions)

    request_status_changed = Signal()

    # Signals for things that can change
    session_color_changed = Signal(int, object)  # session_id, color_value

    current_session_changed = Signal(int)

    hovered_components_changed = Signal()
    selected_components_changed = Signal()
    focused_component_changed = Signal()
    highlighted_components_changed = Signal()

    # compare_mode_changed = Signal(str)
    busy_changed = Signal(bool)
    plot_update_required = Signal()
    session_toggled = Signal(tuple)

    data_changed = Signal(object)
    data_version: int = 0

    statistics_sources_changed = Signal()

    adjacency_radius_changed = Signal()

    def __init__(self, settings: QSettings):
        super().__init__()

        self.settings = settings

        self._current_session_id = None

        self._busy = False

        self._session_colors = []

        self._hovered_components: Optional[NeuronComponent] = None
        self._selected_components: Optional[List[NeuronComponent]] = None
        self._focused_component: Optional[NeuronComponent] = None
        self._highlighted_components: Optional[List[NeuronComponent]] = None

        self.current_job = None
        self.tasks = TaskManager()

        self.config_manager = LoadConfigManager(
            user_dir=(Path(user_config_dir("CATAN")) / "load_configs")
        )

        self.logger = logging.getLogger("GUI")
        self.set_logging_level("ERROR")
        # logging.basicConfig(level=getattr(logging, self.logging_level))

        self.time_ref = None

        self._current_request: Optional[RequestHandler] = None

        ## global parameters
        self._adjacency_radius = 15.0

    def issue(self, level, title, message):
        from PySide6.QtWidgets import QMessageBox

        if level == "info":
            QMessageBox.information(None, title, message)
        elif level == "warning":
            QMessageBox.warning(None, title, message)
        elif level == "error":
            QMessageBox.critical(None, title, message)
        else:
            raise ValueError(f"Unknown issue level: {level}")

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
    def busy(self):
        return self._busy

    @busy.setter
    def busy(self, val: bool):
        self._busy = val
        self.busy_changed.emit(val)

    @property
    def current_request(
        self,
    ) -> Optional[RequestHandler]:
        return self._current_request

    @current_request.setter
    def current_request(
        self,
        request: Optional[RequestHandler],
    ):
        if request is self._current_request:
            return

        self._current_request = request
        self.request_status_changed.emit()

    def notify_request_changed(self):
        self.request_status_changed.emit()

    @property
    def adjacency_radius(self) -> float:
        return self._adjacency_radius

    @adjacency_radius.setter
    def adjacency_radius(self, value: float):
        value = float(value)

        if value == self._adjacency_radius:
            return

        self._adjacency_radius = value
        self.adjacency_radius_changed.emit()

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

    # --- current session ---
    @property
    def current_session_id(self) -> Optional[int]:
        return self._current_session_id

    @current_session_id.setter
    def current_session_id(self, s: Optional[int]):
        if s != self._current_session_id:
            self._current_session_id = s
            self.current_session_changed.emit(s)

    # --- current neuron ---
    """ 
        Logic for subselection of neurons
    """

    @property
    def hovered_components(self):
        return self._hovered_components

    def update_hovered_components(
        self,
        components,
    ):

        if components is None:
            new_components = None
        elif isinstance(components, NeuronComponent):
            new_components = [components]
        else:
            new_components = list(components)

        if new_components == self._hovered_components:
            return

        self._hovered_components = new_components

        self.hovered_components_changed.emit()

    @property
    def selected_components(self) -> Optional[List[NeuronComponent]]:
        if self._selected_components is None:
            return None

        return self._selected_components

    def update_selected_components(
        self,
        components: Optional[NeuronComponent | List[NeuronComponent]],
        modifiers=(),
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

        else:
            ## mode for replacing the current selection with a new one
            selected_components = components

        if isinstance(selected_components, list) and len(selected_components) == 0:
            selected_components = None

        self._selected_components = (
            sorted(selected_components, key=lambda c: c.neuron_id)
            if selected_components is not None
            else None
        )
        if selected_components is not None and not self._component_in_components(
            self.focused_component, selected_components
        ):
            self.focused_component = selected_components[-1]
        elif selected_components is None:
            self.focused_component = None
        else:
            self.focused_component = self.focused_component

        self.selected_components_changed.emit()

    @property
    def focused_component(self) -> Optional[NeuronComponent]:
        return self._focused_component

    @focused_component.setter
    def focused_component(self, component: Optional[NeuronComponent]):

        if component == self._focused_component:
            return

        self.logger.debug(f"Setting focused component to {component}")
        self._focused_component = component

        if component is None:
            self.focused_component_changed.emit()
            return

        if not self._component_in_components(component, self.selected_components):
            self.update_selected_components(component)

        self.focused_component_changed.emit()

    @property
    def highlighted_components(
        self,
    ) -> Optional[List[NeuronComponent]]:

        if self._highlighted_components is None:
            return None

        return self._highlighted_components

    def update_highlighted_components(
        self,
        components: Optional[NeuronComponent | List[NeuronComponent]],
        modifiers=(),
        *,
        max_components: int | None = None,
    ):

        if isinstance(components, NeuronComponent):
            components = [components]

        if "Control" in modifiers:

            # Ctrl-clicking empty space does nothing.
            if components is None:
                return

            if self._highlighted_components is None:
                highlighted_components = list(components)

            else:
                highlighted_components = self._highlighted_components.copy()

                for component in components:

                    try:
                        idx = highlighted_components.index(component)

                        highlighted_components.pop(idx)

                    except ValueError:
                        highlighted_components.append(component)

        else:
            # Ordinary click replaces the highlight.
            highlighted_components = components

        if (
            isinstance(highlighted_components, list)
            and len(highlighted_components) == 0
        ):
            highlighted_components = None

        if highlighted_components is not None:

            # Avoid accidental duplicates while preserving
            # click/insertion order.
            highlighted_components = list(dict.fromkeys(highlighted_components))

        if highlighted_components == self._highlighted_components:
            return

        if (
            highlighted_components is not None
            and max_components is not None
            and len(highlighted_components) > max_components
        ):
            highlighted_components = highlighted_components[-max_components:]

        self._highlighted_components = highlighted_components

        self.highlighted_components_changed.emit()

    @staticmethod
    def _components_match_selection(
        a: NeuronComponent | None,
        b: NeuronComponent | None,
    ) -> bool:
        """
        Whether two components refer to compatible selected identities.

        A session-independent neuron acts as a wildcard for session.
        """

        if a is None or b is None:
            return False

        if int(a.neuron_id) != int(b.neuron_id):
            return False

        return (
            a.session_id is None
            or b.session_id is None
            or int(a.session_id) == int(b.session_id)
        )

    @classmethod
    def _component_in_components(
        cls,
        component: NeuronComponent | None,
        components,
    ) -> bool:

        if component is None or not components:
            return False

        return any(
            cls._components_match_selection(
                component,
                selected,
            )
            for selected in components
        )

    def selected_component_index(
        self,
        component: NeuronComponent | None,
    ) -> int | None:

        if component is None or not self.selected_components:
            return None

        for index, selected in enumerate(self.selected_components):
            if self._components_match_selection(
                component,
                selected,
            ):
                return index

        return None

    def apply_assignment_rebuild(
        self,
        assignments: np.ndarray,
        neuron_id_map: dict[int, int],
        *,
        from_session_id: int,
    ):
        """Publish rebuilt assignments and remap surviving GUI selections."""

        def remap_component(component):
            if component is None:
                return None

            # A session-specific selection in the rebuilt suffix may now
            # belong to a different neuron. Do not silently redirect it.
            if (
                component.session_id is not None
                and component.session_id >= from_session_id
            ):
                return None

            new_id = neuron_id_map.get(component.neuron_id)
            if new_id is None:
                return None

            return NeuronComponent(
                neuron_id=new_id,
                session_id=component.session_id,
            )

        def remap_components(components):
            result = []
            for component in components or []:
                mapped = remap_component(component)
                if mapped is not None:
                    result.append(mapped)
            return result or None

        selected = remap_components(self._selected_components)
        focused = remap_component(self._focused_component)
        highlighted = remap_components(self._highlighted_components)

        # Update everything before emitting any signals. The ordinary
        # selection setters affect one another and emit immediately.
        self.assignments = assignments
        self._selected_components = selected
        self._focused_component = focused
        self._highlighted_components = highlighted
        self._hovered_components = None
        self._current_request = None

        self.selected_components_changed.emit()
        self.focused_component_changed.emit()
        self.highlighted_components_changed.emit()
        self.hovered_components_changed.emit()
        self.request_status_changed.emit()

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

        if component.neuron_id >= self.assignments.shape[0]:
            print("Warning: trying to get footprint with invalid neuron_id:", component)
            return None
        if component.session_id >= self.assignments.shape[1]:
            print(
                "Warning: trying to get footprint with invalid session_id:", component
            )
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

        return NeuronComponent(neuron_id, session_id)


def equal_neurons(id1: NeuronComponent, id2: NeuronComponent):
    return id1.session_id == id2.session_id and id1.neuron_id == id2.neuron_id
