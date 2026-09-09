from typing import Dict, Optional, Tuple, List, Union
from catan.core.io.inspection import check_file_compatibility
from catan.tracking.structures.model import Model
import numpy as np
from scipy import sparse
from pathlib import Path

from . import AppState, NeuronComponent
from catan.core.io import inspect_file
from catan.core.structures import SessionData, sessiondata_type
from catan.tracking.structures import Assignments
from catan.gui.plots.colors import CyclicColorMap

from catan import Tracking

# class Neurons:

#     n: int = 0
#     centroids: np.ndarray
#     union_footprints: Dict[int, sparse.csc_matrix]


class Data(Tracking):

    root: str

    def __init__(self, state: AppState):

        self.state = state

        super().__init__()

        self.current_session: Optional[SessionData] = None

        self.session_colors = CyclicColorMap(n_colors=20, cmap_name="twilight")

        self.state.current_session_changed.connect(self._on_current_session_changed)

    def notify_change(self, change):
        self.state.data_version += 1
        self.state.data_changed.emit(change)

    def _on_current_session_changed(self, session_id: int):
        if len(self.sessions) == 0:
            self.current_session = None
            return

        if session_id < 0 or session_id >= len(self.sessions):
            raise ValueError(f"Invalid session_id {session_id}")

        self.current_session = self.sessions[session_id]

    def toggle_session_data(
        self,
        session_id: int,
        which: Optional[sessiondata_type] = None,
        to_present: Optional[bool] = None,
        **kwargs,
    ):

        session = self.sessions[session_id]
        if session.source_config is None:
            raise ValueError("No load configuration selected.")

        if which is None:
            fields_to_load = session.source_config.get_fields_to_load()
        else:
            fields_to_load = session.source_config.get_fields_to_load([which])

            if to_present is None:
                ## default to "toggle" if nothing provided
                to_present = not session.status[f"{which}_loaded"]

            if not to_present:
                session.clean_data(which)
        session.load_data(
            fields_to_load,
            alignment_template=self.alignment_template,
            ctx=kwargs.get("ctx", None),
        )
        self.notify_change(("session", session_id))
        if "traces" in fields_to_load:
            self.notify_change(("traces", session_id))

    def queue_load_data(self, session_id: int):
        session = self.sessions[session_id]
        self.state.tasks.start(
            "loading",
            f"Loading data for {session.name}",
            self.toggle_session_data,
            session_id=session_id,
            to_present=True,
        )

    def queue_update_model(
        self, session_id: int, to_present: bool = True, callback=None
    ):
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
        self.notify_change(("model", session_id))

    def fit_after_loading(self, key: str = "model update"):
        """
        ensures the fit is only executed once all current
        processes of session loading have finished
        """

        sessions_loaded = [s.status["spatial_loaded"] for s in self.sessions]
        if (
            np.sum(sessions_loaded) < 2
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
        self.notify_change(("assignments", -1))  # Notify that model has changed

    def register_session(
        self,
        from_file: str | Path,
        **kwargs,
    ):
        """
        Loads and registers session data from a file `fname`.
        """
        structure = inspect_file(from_file)
        sessions = [
            key
            for key, val in structure.entries.items()
            if (Path(key).name.startswith("session") and val.kind == "group")
        ]
        if len(sessions):
            ## dirty way to check between single and multiple session files
            sessions_data = super().load_session_data(from_file, {})
        else:
            sessions_data = [
                {"metadata": {"path": from_file}}
            ]  # Wrap single session data in a list for uniform processing

        for session_data in sessions_data:
            session_id = super().register_session(
                from_data=SessionData._from_dict(session_data), align=True, **kwargs
            )
            self.sessions[session_id].source_config = (
                self.state.config_manager.suggest_config_for(
                    path=self.sessions[session_id].path, source_type="session"
                )
            )

            if not self.sessions[session_id].name:
                self.sessions[session_id].name = Path(
                    self.sessions[session_id].path
                ).parent.name

            self.state.session_color = (session_id, self.session_colors.next())

            if session_id == 0 and self.sessions[session_id].status["aligned"]:
                self.assign_neurons(from_session_index=session_id)

            if self.state.current_session_id is None:
                self.state.current_session_id = session_id

            # Notify that sessions have changed
            self.notify_change(("session_added", session_id))

    def remove_session(self, session_id: int):

        super().move_session(session_id, -1)
        self.state.assignments = self.assignments.ids

        self.adjust_selected_components_after_data_change(session_id, -1)

        if self.current_session is not None:
            self.state.current_session_id = (
                self.current_session.id if len(self.sessions) > 0 else None
            )

        self.notify_change(
            ("session_removed", session_id)
        )  # Notify that sessions have changed

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

        self.notify_change(("session_moved", -1))  # Notify that sessions have changed

    def add_model(self, name: str, model: Optional[str | Model] = None):
        super().add_model(name, model)
        self.notify_change(("model", -1))  # Notify that model has changed

    def change_model(self, name: str):
        super().change_model(name)
        self.notify_change(("model", -1))  # Notify that model has changed

    def register_assignments(self, path: str, name: str):

        # print("Registering assignments with path:", path, "and name:", name)
        assignments = Assignments()
        assignments.path = path
        assignments.source_config = self.state.config_manager.suggest_config_for(
            path=path, source_type="assignments"
        )
        assert (
            assignments.source_config is not None
        ), "Failed to determine source config for the assignments file."

        self._assignments[name] = assignments
        self._current_assignments = name

    def load_assignments(self):
        if self.assignments is None:
            return

        self.assignments.load()
        self.rebuild_union()

        self.state.assignments = self.assignments.ids
        self.notify_change(("assignments", -1))

    def add_assignments(
        self, name: str, assignments: Optional[str | Assignments] = None
    ):
        try:
            super().add_assignments(name, assignments)
            if self.assignments is None:
                raise ValueError(
                    "Failed to add assignments. The assignments object is None."
                )
        except Exception as e:
            self.state.issue(
                "error",
                "Failed to add assignments",
                f"{e}",
                # f"Could not add assignments '{name}'. Most common reasons are that the loaded session data and assignments file are not compatible. This could be due to too few sessions being loaded, or the sessions containing an incompatible number of neurons. Please check the assignments file and the loaded session data.",
            )
            return

        self.assignments.source_config = self.state.config_manager.suggest_config_for(
            path=self.assignments.path, source_type="assignments"
        )

        if self.assignments is None:
            raise ValueError(
                "No assignments file was added. Please provide valid assignments."
            )
        self.state.assignments = self.assignments.ids
        self.notify_change(("assignments", -1))  # Notify that assignments have changed

    def change_assignments(self, name: str):
        super().change_assignments(name)
        self.state.assignments = self.assignments.ids
        self.notify_change(("assignments", -1))  # Notify that assignments have changed

    def queue_assign_neurons(self, session_id: int, to_present=True, callback=None):
        session = self.sessions[session_id]

        if to_present:
            fn = lambda ctx: self.assign_neurons(
                from_session_index=session_id, clean_traces=False, ctx=ctx
            )
        else:
            fn = lambda ctx: self.unassign_neurons(session_id, ctx=ctx)

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
        clean_traces: bool = False,
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
        self.notify_change(("assignments", -1))  # Notify that neurons have changed

    def unassign_neurons(self, session_id: int, **kwargs):

        super().unassign_neurons(session_id)
        self.state.assignments = self.assignments.ids
        if self.state.current_session_id == session_id:
            self.state.current_session_id = None

        self.adjust_selected_components_after_data_change(session_id, -1)
        self.notify_change(("assignments", -1))

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
