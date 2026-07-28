from typing import Dict, Optional, Tuple, List

import numpy as np
from scipy import sparse

from catan.core.structures import SessionData
from pathlib import Path

# from catan.gui.background_tasks import TaskContext
from catan import Tracking

# from catan.gui.data.utils import move_index_along_axis

print("reloading data.py")
# importlib.reload(neuron_tracking)  # reload to pick up changes during development


class Neurons:

    n: int = 0
    centroids: np.ndarray
    union_footprints: Dict[int, sparse.csc_matrix]


class Data(Tracking):

    def __init__(self, state):

        super().__init__()
        self.state = state
        self.current_session: Optional[SessionData] = None

        self.state.current_session_changed.connect(self._on_current_session_changed)

    def _on_data_changed(self, change):
        self.state.data_version += 1
        self.state.data_changed.emit(change)

    def _on_current_session_changed(self, session_id: int):
        if session_id < 0 or session_id >= len(self.sessions):
            raise ValueError(f"Invalid session_id {session_id}")

        self.current_session = self.sessions[session_id]

        self.change_trace_presence(session_id, True)

    def change_trace_presence(self, session_id: int, to_present: Optional[bool] = None):
        """
        should be realized by session structure directly
        """
        session = self.sessions[session_id]
        ## default to "toggle" if nothing provided
        if to_present is None:
            to_present = not session.status["traces_loaded"]

        if session.status["traces_loaded"] == to_present:
            return

        if to_present:
            session.load_data(["temporal"])
        else:
            session.clean_traces()

        self._on_data_changed(("traces", session_id))

    def move_session(self, old_session_id: int, new_session_id: int):
        super().move_session(old_session_id, new_session_id)
        self.state.assignments = self.assignments

        ## adjust selected components to reflect the new session IDs
        for component in self.state.selected_components:
            if component.session_id == old_session_id:
                component.session_id = new_session_id
            elif component.session_id == new_session_id:
                component.session_id = old_session_id

        if self.current_session is not None:
            self.state.current_session_id = self.current_session.id

        self._on_data_changed(("assignments", -1))  # Notify that sessions have changed

    def register_session(
        self,
        from_file: Optional[str | Path] = None,
        name: Optional[str] = None,
        load_content: List[str] = ["spatial", "temporal", "quality"],
        align=True,
    ) -> int:
        """ """
        session_id = super().register_session(from_file, name, load_content, align)
        # Notify that sessions have changed
        self._on_data_changed(("sessions", session_id))
        return session_id

    def remove_session(self, session_id: int):

        super().remove_session(session_id)
        self.state.assignments = self.assignments

        self.adjust_selected_components_after_data_change(session_id, -1)

        if self.current_session is not None:
            self.state.current_session_id = self.current_session.id

        self._on_data_changed(("assignments", -1))  # Notify that sessions have changed

    def register_neurons(
        self,
        from_file: Optional[str | Path] = None,
        from_data: Optional[SessionData] = None,
        from_session_id: Optional[int] = None,
        align_to_reference: bool = True,
        clean_traces: bool = True,
        p_thr=[0.5, 0.3],
    ):
        """ """
        super().register_neurons(
            from_file=from_file,
            from_data=from_data,
            from_session_id=from_session_id,
            align_to_reference=align_to_reference,
            clean_traces=clean_traces,
            p_thr=p_thr,
        )
        self.state.assignments = self.assignments
        self._on_data_changed(("assignments", -1))  # Notify that neurons have changed

    def unregister_neurons(self, session_id: int):

        super().unregister_neurons(session_id)
        self.state.assignments = self.assignments
        if self.state.current_session_id == session_id:
            self.state.current_session_id = None

        self.adjust_selected_components_after_data_change(session_id, -1)

        self._on_data_changed(("assignments", -1))

    def adjust_selected_components_after_data_change(
        self, session_id: int, new_session_id: int
    ):

        ## adjust selected components to reflect the new session IDs
        for component in self.state.selected_components:
            if component.session_id == session_id:

                ## first, check if neuron is still present
                if component.neuron_id >= self.assignments.shape[0]:
                    component.session_id = -1  # No other sessions, mark as invalid
                    continue

                # if new_session_id >= 0:
                component.session_id = new_session_id
                ## dynamically find first first session presence, if session is removed
                if new_session_id == -1:
                    other_sessions = np.where(
                        self.state.assignments[component.neuron_id, :]
                    )[0]
                    if len(other_sessions):
                        component.session_id = other_sessions[0]

        self.state.update_selected_components(
            [c for c in self.state.selected_components if c.session_id != -1]
        )

    def load_registration(self, path_registration: str | Path):
        super().load_registration(path_registration)
        self.state.assignments = self.assignments
        self._on_data_changed(("assignments", -1))  # Notify that neurons have changed
