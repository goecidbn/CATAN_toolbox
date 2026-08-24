from typing import Dict, Optional, Tuple, List, Union
from catan.tracking.structures.model import Model
import numpy as np
from scipy import sparse
from pathlib import Path

from . import AppState, NeuronComponent, ConfigData
from catan.core.structures import SessionData, sessiondata_type
from catan.core.structures.load_config_manager import LoadConfigManager
from catan.tracking.structures import Assignments
from catan.gui.plots.colors import CyclicColorMap

from catan import Tracking

from importlib.resources import files
from platformdirs import user_config_dir

class Neurons:

    n: int = 0
    centroids: np.ndarray
    union_footprints: Dict[int, sparse.csc_matrix]


class Data(Tracking):

    def __init__(self, state: AppState):

        self.state = state
        
        super().__init__()
        
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


    def toggle_session_data(self, session_id: int, which: Optional[sessiondata_type], to_present: Optional[bool] = None, **kwargs):
        
        if self.load_configs.current is None:
            raise ValueError("No load configuration is currently selected.")
        session = self.sessions[session_id]

        if which is None:
            fields_to_load = self.load_configs.current.get_fields_to_load()
        else:
            fields_to_load = self.load_configs.current.get_fields_to_load([which])
            
            if to_present is None:
                ## default to "toggle" if nothing provided
                to_present = not session.status[f"{which}_loaded"]
            
            if not to_present:
                session.clean_data(which)
            
        session.load_data(
            fields_to_load, self.alignment_template, ctx=kwargs.get("ctx", None)
        )
        self._on_data_changed((which, session_id))

    def queue_load_data(self, session_id: int):
        session = self.sessions[session_id]
        self.state.tasks.start(
            "loading",
            f"Loading data for {session.name}",
            self.load_data,
            session_id=session_id,
        )
    
    def load_data(self, session_id: int, **kwargs):

        if self.load_configs.current is None:
            raise ValueError("No load configuration is currently selected.")
        fields_to_load = self.load_configs.current.get_fields_to_load()

        session = self.sessions[session_id]
        session.load_data(fields_to_load, self.alignment_template, **kwargs)
        if self.state.current_session_id is None:
            self.state.current_session_id = session_id
        self._on_data_changed(("sessions", session_id))


    def queue_update_model(self, session_id: int, to_present: bool = True, callback=None):
        if self.model is not None and self.model.loaded:
            self.state.issue(
                "warning", 
                "Model registration not allowed",
                f"Model '{self.current_model_name}' was loaded from file and cannot be updated. Please create a new model to fit to data.",
            )
            return
        session = self.sessions[session_id]

        def when_finished():
            self.fit_after_loading()
            if callback is not None:
                callback()

        self.state.tasks.start(
            "model update",
            f"Update model for {session.name}",
            self.update_counts,
            session_id=session_id,
            finished=when_finished,
            ready=lambda session=session: session.status["aligned"],
        )
    
    def update_counts(
        self,
        session_id: Optional[int] = None,
        **kwargs,
    ):
        super().update_counts_with_data(
            from_session_index=session_id,
            align_to_reference=True,
        )
        self._on_data_changed(("model", session_id))


    def fit_after_loading(self, key: str = "model update"):
        """
            ensures the fit is only executed once all current
            processes of session loading have finished
        """
        
        if (
            len(self.sessions) < 2
            or self.state.tasks.current[key] is not None
            or len(self.state.tasks.queues[key]) > 0
        ):
            return

        self.state.tasks.start(
            "model update",
            "Fit model to data",
            self.fit_model,
        )

    def fit_model(self, **kwargs):
        if self.model is None:
            raise ValueError("No model to fit. Please add a model before fitting.")
        self.model.fit_model_to_counts(self.counts["cross"])
        self._on_data_changed(("assignments", -1))  # Notify that model has changed


    def register_session_data(
        self,
        fname: str | Path,
        fields_to_load: Optional[dict] = None,
        align=True,
        **kwargs,
    ):
        """
        Loads and registers session data from a file `fname`.
        
        """

        sessions = super().load_session_data(fname, fields_to_load)

        for session in sessions:
            try:
                session_id = super().register_session(
                    from_data=session,
                    align=align,
                    **kwargs
                )
            except Exception as e:
                self.state.issue(
                    "error",
                    "Failed to register session",
                    e,
                )
                continue

            if not self.sessions[session_id].name:
                self.sessions[session_id].name = Path(self.sessions[session_id].path).parent.name

            self.state.session_color = (session_id, self.session_colors.next())

            if session_id == 0 and self.sessions[session_id].status["aligned"]:
                self.assign_neurons(from_session_index=session_id)

            if self.state.current_session_id is None:
                self.state.current_session_id = session_id

            # Notify that sessions have changed
            self._on_data_changed(("sessions", session_id))


        # return session_id

    def remove_session(self, session_id: int):

        super().move_session(session_id, -1)
        self.state.assignments = self.assignments.ids

        self.adjust_selected_components_after_data_change(session_id, -1)

        if self.current_session is not None:
            self.state.current_session_id = (
                self.current_session.id if len(self.sessions) > 0 else None
            )

        self._on_data_changed(("assignments", -1))  # Notify that sessions have changed

    def move_session(self, session_id: int, new_session_id: int):
        """
            behavior should be off on session removing - check that!
        """

        super().move_session(session_id, new_session_id)
        self.state.assignments = self.assignments.ids

        order_translation = list(range(len(self.sessions)))
        id = order_translation.pop(new_session_id)
        order_translation.insert(session_id, id)

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

            self.state.update_selected_components(list(components))

        if self.current_session is not None:
            self.state.current_session_id = self.current_session.id

        self._on_data_changed(("assignments", -1))  # Notify that sessions have changed

    def add_model(self, name: str, model: Optional[str|Model] = None):
        super().add_model(name, model)
        self._on_data_changed(("model", -1))  # Notify that model has changed

    def change_model(self, name: str):
        super().change_model(name)
        self._on_data_changed(("model", -1))  # Notify that model has changed
    
    def add_assignments(self, name: str, assignments: Optional[str|Assignments] = None):
        try:
            super().add_assignments(name, assignments)
        except:
            self.state.issue(
                "error",
                "Failed to add assignments",
                f"Could not add assignments '{name}'. Most common reasons are that the loaded session data and assignments file are not compatible. This could be due to too few sessions being loaded, or the sessions containing an incompatible number of neurons. Please check the assignments file and the loaded session data.",
            )
            return
        if self.assignments is None:
            raise ValueError("No assignments file was added. Please provide valid assignments.")
        self.state.assignments = self.assignments.ids
        self._on_data_changed(("assignments", -1))  # Notify that assignments have changed

    def change_assignments(self, name: str):
        super().change_assignments(name)
        self.state.assignments = self.assignments.ids
        self._on_data_changed(("assignments", -1))  # Notify that assignments have changed


    def queue_assign_neurons(self, session_id: int, to_present=True, callback=None):
        session = self.sessions[session_id]

        if to_present: 
            fn = lambda ctx: self.assign_neurons(from_session_index=session_id, clean_traces=False,ctx=ctx)
        else:
            fn = lambda ctx: self.unassign_neurons(session_id,ctx=ctx)

        self.state.tasks.start(
            "calculating",
            f"Register neurons for {session.name}",
            fn,
            ready=lambda session=session, session_id=session_id: (
                (session_id == 0) or (self.model is not None and self.model.fitted)
            )
            and session.status["aligned"],
            finished=callback,
        )

    def assign_neurons(
        self,
        from_file: Optional[str | Path] = None,
        from_data: Optional[SessionData] = None,
        from_session_index: Optional[int] = None,
        align_to_reference: bool = False,
        clean_traces: bool = True,
        force_registration: bool = False,
        p_thr=[0.5, 0.3],
        **kwargs,
    ):
        """ """
        super().assign_neurons(
            from_file=from_file,
            from_data=from_data,
            from_session_index=from_session_index,
            align_to_reference=align_to_reference,
            clean_traces=clean_traces,
            force_registration=force_registration,
            p_thr=p_thr,
        )
        self.state.assignments = self.assignments.ids
        self._on_data_changed(("assignments", -1))  # Notify that neurons have changed

    def unassign_neurons(self, session_id: int, **kwargs):

        super().unassign_neurons(session_id)
        self.state.assignments = self.assignments.ids
        if self.state.current_session_id == session_id:
            self.state.current_session_id = None

        self.adjust_selected_components_after_data_change(session_id, -1)
        self._on_data_changed(("assignments", -1))


    def adjust_selected_components_after_data_change(
        self, session_id: int, new_session_id: int
    ):
        """
            this is (and should) only be called on session removal! 
        """

        ## adjust selected components to reflect the new session IDs
        if self.state.selected_components is None:
            return

        components = set()
        for component in self.state.selected_components:
            if component.session_id == session_id:

                ## first, check if neuron is still present
                if component.neuron_id >= self.state.assignments.shape[0]:
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
