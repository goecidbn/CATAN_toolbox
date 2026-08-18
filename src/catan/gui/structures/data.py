from typing import Dict, Optional, Tuple, List
import numpy as np
from scipy import sparse

from . import AppState, NeuronComponent, ConfigData
from catan.core.structures import SessionData
from pathlib import Path
from catan.gui.plots.colors import CyclicColorMap

# from catan.gui.background_tasks import TaskContext
from catan import Tracking

# from catan.gui.data.utils import move_index_along_axis


class Neurons:

    n: int = 0
    centroids: np.ndarray
    union_footprints: Dict[int, sparse.csc_matrix]


class Data(Tracking):

    def __init__(self, state: AppState, config: ConfigData):

        super().__init__()
        self.state = state
        self.config = config
        self.current_session: Optional[SessionData] = None

        self.session_colors = CyclicColorMap(n_colors=20, cmap_name="twilight")

        self.state.current_session_changed.connect(self._on_current_session_changed)

    def _on_data_changed(self, change):
        self.state.data_version += 1
        self.state.data_changed.emit(change)

    def _on_current_session_changed(self, session_id: int):
        if len(self.sessions) == 0:
            self.current_session = None
            return

        if session_id < 0 or session_id >= len(self.sessions):
            raise ValueError(f"Invalid session_id {session_id}")

        self.current_session = self.sessions[session_id]

        # self.change_trace_presence(session_id, True)

    def change_spatial_presence(
        self, session_id: int, to_present: Optional[bool] = None, **kwargs
    ):
        """
        does it even make sense to unload
         - no use for session anymore
         - not much data volume in there)
        """
        session = self.sessions[session_id]
        if to_present is None:
            to_present = not session.status["spatial_loaded"]

        if to_present:
            fields_to_load = {"spatial": self.config.fields["spatial"]}
            session.load_data(
                fields_to_load,
                self.alignment_template,
                force_load=True,
                ctx=kwargs.get("ctx", None),
            )
            self._on_data_changed(("session", session_id))
        else:
            self.state.logger.WARNING("Unloading footprints doesnt make much sense!")

    def change_trace_presence(
        self, session_id: int, to_present: Optional[bool] = None, **kwargs
    ):
        """
        should be realized by session structure directly
        """
        session = self.sessions[session_id]
        if to_present is None:
            ## default to "toggle" if nothing provided
            to_present = not session.status["traces_loaded"]

        if session.status["traces_loaded"] == to_present:
            return

        if to_present:
            fields_to_load = {"traces": self.config.fields["traces"]}
            session.load_data(
                fields_to_load, force_load=True, ctx=kwargs.get("ctx", None)
            )
        else:
            session.clean_traces()
        self._on_data_changed(("traces", session_id))

    def change_quality_presence(
        self, session_id: int, to_present: Optional[bool] = None, **kwargs
    ):
        """
        should be realized by session structure directly
        """
        session = self.sessions[session_id]
        ## default to "toggle" if nothing provided
        if to_present is None:
            to_present = not session.status["quality_loaded"]

        if session.status["quality_loaded"] == to_present:
            return

        if to_present:
            fields_to_load = {"quality": self.config.fields["quality"]}
            session.load_data(
                fields_to_load, force_load=True, ctx=kwargs.get("ctx", None)
            )
            self._on_data_changed(("quality", session_id))
        else:
            session.clean_quality()

        self._on_data_changed(("quality", session_id))

    def load_data(self, session_id: int, fields_to_load: dict, **kwargs):

        session = self.sessions[session_id]
        session.load_data(fields_to_load, self.alignment_template, **kwargs)
        if self.state.current_session_id is None:
            self.state.current_session_id = session_id

    def update_model_with_data(
        self,
        from_file: Optional[str | Path] = None,
        from_data: Optional[SessionData] = None,
        from_session_index: Optional[int] = None,
        align_to_reference=True,
        **kwargs,
    ):
        super().update_model_with_data(
            from_file=from_file,
            from_data=from_data,
            from_session_index=from_session_index,
            align_to_reference=align_to_reference,
        )

    def fit_to_model(self, **kwargs):
        super().fit_to_model()

    def move_session(self, old_session_id: int, new_session_id: int):
        super().move_session(old_session_id, new_session_id)
        self.state.assignments = self.assignments

        order_translation = list(range(len(self.sessions)))
        id = order_translation.pop(new_session_id)
        order_translation.insert(old_session_id, id)

        ## adjust selected components to reflect the new session IDs
        if self.state.selected_components is not None:
            components = set()
            for component in self.state.selected_components:
                components.add(
                    NeuronComponent(
                        neuron_id=component.neuron_id,
                        session_id=order_translation[component.session_id],
                    )
                )

                # if component.session_id == old_session_id:
                #     components.add(
                #         NeuronComponent(
                #             neuron_id=component.neuron_id, session_id=new_session_id
                #         )
                #     )
                # elif component.session_id == new_session_id:
                #     components.add(
                #         NeuronComponent(
                #             neuron_id=component.neuron_id, session_id=old_session_id
                #         )
                #     )
                # else:
                #     components.add(component)
            self.state.update_selected_components(list(components))

        if self.current_session is not None:
            self.state.current_session_id = self.current_session.id

        self._on_data_changed(("assignments", -1))  # Notify that sessions have changed

    def register_session(
        self,
        fields_to_load: Optional[dict] = None,
        from_file: Optional[str | Path] = None,
        name: Optional[str] = None,
        align=True,
        **kwargs,
    ) -> int:

        session_id = super().register_session(
            fields_to_load, from_file, name, align, **kwargs
        )
        self.state.session_color = (session_id, self.session_colors.next())
        # Notify that sessions have changed
        self._on_data_changed(("sessions", session_id))
        return session_id

    def remove_session(self, session_id: int):

        super().move_session(session_id, -1)
        self.state.assignments = self.assignments

        self.adjust_selected_components_after_data_change(session_id, -1)

        if self.current_session is not None:
            self.state.current_session_id = (
                self.current_session.id if len(self.sessions) > 0 else None
            )

        self._on_data_changed(("assignments", -1))  # Notify that sessions have changed

    def register_neurons(
        self,
        from_file: Optional[str | Path] = None,
        from_data: Optional[SessionData] = None,
        from_session_index: Optional[int] = None,
        align_to_reference: bool = True,
        clean_traces: bool = True,
        p_thr=[0.5, 0.3],
        **kwargs,
    ):
        """ """
        super().register_neurons(
            from_file=from_file,
            from_data=from_data,
            from_session_index=from_session_index,
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
        if self.state.selected_components is None:
            return

        components = set()
        for component in self.state.selected_components:
            if component.session_id == session_id:

                ## first, check if neuron is still present
                if component.neuron_id >= self.assignments.shape[0]:
                    continue

                # if new_session_id >= 0:
                # component.session_id = new_session_id
                ## dynamically find first first session presence, if session is removed
                # if new_session_id == -1:
                other_sessions = np.where(
                    self.state.assignments[component.neuron_id, :]
                )[0]
                if len(other_sessions):
                    components.add(
                        NeuronComponent(
                            neuron_id=component.neuron_id, session_id=other_sessions[0]
                        )
                    )
        if len(components) > 0:
            self.state.update_selected_components(list(components))
        else:
            self.state.update_selected_components(None)

    def load_registration(self, path_registration: str | Path):
        super().load_registration(path_registration)
        self.state.assignments = self.assignments
        self._on_data_changed(("assignments", -1))  # Notify that neurons have changed
