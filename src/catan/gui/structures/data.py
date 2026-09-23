from typing import Dict, Optional, Tuple, List, Union
from catan.tracking.structures.model import Model
import numpy as np
from pathlib import Path

from catan import Tracking
from . import AppState, StatisticDisplayConfig
from .request_handler import RequestHandler
from catan.core.io import inspect_file, evaluate_file_compatibility
from catan.core.structures import NeuronComponent, SessionData, sessiondata_type
from catan.tracking.structures import Assignments, ReviewStatus
from catan.gui.panels.colors import CyclicColorMap

from catan.gui.data.statistics.engine import StatisticEngine
from catan.gui.data.statistics.registry import build_statistics_registry

from catan.gui.background_tasks.runtime import current_task_context


class Data(Tracking):

    root: str

    def __init__(self, state: AppState):

        self.state = state
        super().__init__()

        self.statistic_engine = StatisticEngine(
            data=self,
            state=state,
            registry_factory=build_statistics_registry,
        )
        self.statistic_display_config = StatisticDisplayConfig(
            settings=state.settings, engine=self.statistic_engine
        )

        self.current_session: Optional[SessionData] = None

        self.session_colors = CyclicColorMap(n_colors=20, cmap_name="twilight")

        self.state.current_session_changed.connect(self._on_current_session_changed)

    def is_available(self, what: List[str] | None = None) -> bool:

        available = True
        available &= len(self.sessions) > 0
        available &= not (self.assignments is None or self.assignments.union is None)
        if not available:
            return available
        available &= (
            len(self.assignments.union.included) == self.assignments.union.n_neurons
        )

        if what is None:
            return available

        if "current_session" in what:
            available &= self.current_session is not None

        return available

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

    def evaluate_queries(self, **kwargs):
        table = {}
        for key, query in kwargs.items():
            table[key] = self.statistic_engine.evaluate_table(query)
        return table

    def toggle_session_data(
        self,
        session_id: int,
        which: Optional[sessiondata_type] = None,
        to_present: Optional[bool] = None,
        **kwargs,
    ):

        ctx = current_task_context()

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
            ctx=ctx,
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
            session = self.sessions[session_id]
            assert session.path is not None, "Session path should not be None"
            session.source_config = self.state.config_manager.suggest_config_for(
                path=session.path, source_type="session"
            )

            if session.source_config is not None:
                fields_to_load = session.source_config.get_fields_to_load()
                loading_possible = evaluate_file_compatibility(
                    session.path,
                    fields_to_load,
                )
                if loading_possible and all(
                    not session.status[f"{field}_loaded"] for field in fields_to_load
                ):
                    self.toggle_session_data(session_id, to_present=True)

            if not session.name:
                session.name = Path(session.path).parent.name

            self.state.session_color = (session_id, self.session_colors.next())

            if session_id == 0 and session.status["aligned"]:
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
        self.restore_manipulations()

        self.rebuild_union()
        self.rebuild_union_included()

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
            ctx = current_task_context()
            fn = lambda: self.assign_neurons(
                from_session_index=session_id, clean_traces=False, ctx=ctx
            )
        else:
            fn = lambda: self.unassign_neurons(session_id)

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

    def is_included(self, component: NeuronComponent | int) -> bool:

        if self.assignments is None or self.assignments.union is None:
            return False

        if isinstance(component, int):
            return self.assignments.union.included[component]
        elif component.session_id is None:
            return self.assignments.union.included[component.neuron_id]
        else:
            fp_id = self.state.get_footprint_from_component(component)
            session_id = component.session_id
            if fp_id is None:
                return False
            return self.sessions[session_id].included[fp_id]

    def exclude_component(
        self, component: NeuronComponent | int, notify=True
    ) -> NeuronComponent:
        """
        Retire one concrete component.

        The footprint itself remains in SessionData, but:
        - session.included[footprint_id] becomes False
        - the footprint is detached from its current tracked neuron
        - it is assigned to a new singleton neuron row

        Returns the new singleton NeuronComponent.
        """

        if isinstance(component, int):
            ## redirects to retiring a neuron, if only an int is provided
            self.exclude_neuron(component)
            return NeuronComponent(neuron_id=component, session_id=None)

        if self.assignments is None:
            raise ValueError("No assignments are available.")

        if component.session_id is None:
            raise ValueError("Cannot exclude a session-independent component.")

        neuron_id, session_id = component.id

        if not (0 <= session_id < len(self.sessions)):
            raise ValueError(f"Invalid session ID {session_id}.")

        if not (0 <= neuron_id < self.assignments.ids.shape[0]):
            raise ValueError(f"Invalid neuron ID {neuron_id}.")

        footprint_id = int(self.assignments.ids[component.id])

        if footprint_id < 0:
            raise ValueError(
                f"Neuron {neuron_id} has no footprint " f"in session {session_id}."
            )

        session = self.sessions[session_id]

        if not (0 <= footprint_id < session.n_neurons):
            raise ValueError(
                f"Invalid footprint ID {footprint_id} " f"for session {session_id}."
            )

        # ---------------------------------------------
        # 1. Mark the actual footprint as excluded.
        # ---------------------------------------------
        session.included[footprint_id] = False

        # ---------------------------------------------
        # 2. Create an orphan/singleton neuron identity.
        # ---------------------------------------------
        self.assignments.pad_empty(n_neurons=1, n_sessions=0)
        retired_neuron_id = self.assignments.ids.shape[0] - 1

        # every footprint remains assigned to exactly one neuron row.
        self.assignments.ids[retired_neuron_id, session_id] = footprint_id
        self.assignments.ids[neuron_id, session_id] = -1

        for key, values in self.assignments.stats.items():

            # single neuron with no reference gets default statistics.
            values[retired_neuron_id, session_id, ...] = (
                self.assignments.stats_default_value[key]
            )

            # The now-empty original slot gets a clean state.
            values[retired_neuron_id, session_id, ...] = np.nan

        # ---------------------------------------------
        # 3. Rebuild union representation.
        # ---------------------------------------------
        self.rebuild_union_neurons([neuron_id, retired_neuron_id])

        # The new singleton union neuron is explicitly
        # excluded; this is NOT derived from the session
        # flags.
        self.assignments.union.included[retired_neuron_id] = False

        # ---------------------------------------------
        # 4. Synchronize GUI assignment mirror.
        # ---------------------------------------------

        self.state.assignments = self.assignments.ids
        if notify:
            self.notify_change(("assignments", -1))

        return NeuronComponent(neuron_id=retired_neuron_id, session_id=session_id)

    def exclude_neuron(self, neuron_id: int) -> NeuronComponent:

        if self.assignments is None:
            raise ValueError("No assignments are available.")

        self.assignments.union.included[neuron_id] = False
        for session_id in range(self.assignments.ids.shape[1]):
            fp_id = self.state.get_footprint_from_component(
                NeuronComponent(neuron_id=neuron_id, session_id=session_id)
            )
            if fp_id is None:
                continue
            self.sessions[session_id].included[fp_id] = False

        self.notify_change(("assignments", -1))
        return NeuronComponent(neuron_id=neuron_id, session_id=None)

    def reinclude_component(self, component: NeuronComponent | int) -> None:

        if isinstance(component, int):
            self.reinclude_neuron(component)
            return

        if self.assignments is None:
            raise ValueError("No assignments available.")

        if component.session_id is None:
            raise ValueError("A concrete session component is required.")

        session_id = int(component.session_id)
        neuron_id = int(component.neuron_id)

        footprint_id = int(self.assignments.ids[neuron_id, session_id])

        if footprint_id < 0:
            raise ValueError(f"{component} has no assigned footprint.")

        # Restore the actual footprint.
        self.sessions[session_id].included[footprint_id] = True

        # Restore the singleton neuron as an available
        # tracked-neuron identity.
        self.assignments.union.included[neuron_id] = True

        self.notify_change(("assignments", -1))

    def reinclude_neuron(self, neuron_id: int) -> None:

        self.assignments.union.included[neuron_id] = True
        for session_id in range(self.assignments.ids.shape[1]):
            self.reinclude_component(
                NeuronComponent(neuron_id=neuron_id, session_id=session_id)
            )

    def remove_component(self, component: NeuronComponent | int, notify=True) -> None:

        if isinstance(component, int):
            self.remove_neuron(component)
            return

        if self.assignments is None:
            raise ValueError("No assignments available.")

        if component.session_id is None:
            raise ValueError("A concrete session component is required.")

        session_id = int(component.session_id)
        neuron_id = int(component.neuron_id)

        fp_id = int(self.assignments.ids[component.id])

        if fp_id < 0:
            raise ValueError(f"{component} has no assigned footprint.")

        # No longer part of the usable session data.
        self.sessions[session_id].included[fp_id] = False

        # Remove from tracked-neuron structure entirely.
        self.assignments.ids[component.id] = -1

        # Clear tracking statistics for that slot.
        for key, values in self.assignments.stats.items():
            values[*component.id, ...] = np.nan

        self.state.assignments = self.assignments.ids
        if notify:
            self.rebuild_union_neurons([neuron_id])
            self.assignments.updating_neuron_presence()
            self.notify_change(("assignments", -1))

    def remove_neuron(self, neuron_id: int) -> None:

        if self.assignments is None:
            raise ValueError("No assignments available.")

        for session_id in range(self.assignments.ids.shape[1]):
            component = NeuronComponent(neuron_id=neuron_id, session_id=session_id)
            if self.assignments.ids[component.id] >= 0:
                self.remove_component(component, notify=False)

        self.assignments.updating_neuron_presence()
        self.state.assignments = self.assignments.ids
        self.rebuild_union_neurons([neuron_id])
        self.notify_change(("assignments", -1))

    def add_synthetic_component(
        self,
        source: NeuronComponent,
        target_session_id: int,
        target_neuron_id: int,
        *,
        notify: bool = True,
    ) -> NeuronComponent:

        if self.assignments is None:
            raise ValueError("No assignments available.")

        if source.session_id is None:
            raise ValueError(
                "Synthetic component requires " "a concrete source component."
            )

        source_session_id = int(source.session_id)
        source_neuron_id = int(source.neuron_id)

        source_fp_id = int(self.assignments.ids[source.id])

        if source_fp_id < 0:
            raise ValueError(f"{source} has no footprint.")

        target_neuron = NeuronComponent(target_neuron_id, target_session_id)
        if self.assignments.ids[target_neuron.id] >= 0:
            raise ValueError(
                f"Neuron {target_neuron_id} already "
                f"has a component in session "
                f"{target_session_id}."
            )

        source_session = self.sessions[source_session_id]
        target_session = self.sessions[target_session_id]

        if not source_session.included[source_fp_id]:
            raise ValueError(
                "An excluded component cannot be used "
                "as a synthetic-component source."
            )

        if source_session.dims != target_session.dims:
            raise ValueError(
                "Source and target sessions have " "different spatial dimensions."
            )

        footprint = source_session.footprints[:, source_fp_id].copy()

        new_fp_id = target_session.append_synthetic_component(footprint)

        # Register it under its intended neuron identity.
        self.assignments.ids[target_neuron.id] = new_fp_id

        # A synthetic/manual assignment gets the ordinary
        # default assignment statistics.
        for key, values in self.assignments.stats.items():
            values[*target_neuron.id, ...] = np.nan

        self.rebuild_union_neurons([target_neuron_id])

        self.assignments.union.included[target_neuron_id] = True

        self.state.assignments = self.assignments.ids

        if notify:
            self.notify_change(("assignments", -1))

        return target_neuron

    def change_review_status(self, neuron_id: int, status: ReviewStatus):

        self.assignments.review_status[neuron_id] = status
        self.notify_change(("review_status", -1))

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

    def process_component_request(self, request: RequestHandler):
        if self.assignments is None:
            raise ValueError("No assignments available.")

        if not request.is_complete:
            raise ValueError("Cannot process an incomplete request.")

        target_session_id = int(request.origin[0].session_id)
        origin_neurons = {int(component.neuron_id) for component in request.origin}

        # ============================================
        # Determine result components
        # ============================================

        if request.type == "merge":
            sources = [request.destination[0]]
            target_neurons = [request.origin[0].neuron_id]

        elif request.type == "split":
            sources = list(request.destination)
            target_neurons = [component.neuron_id for component in request.destination]

        else:
            raise ValueError(f"Unknown request type: " f"{request.type!r}")

        target_neurons = [int(neuron) for neuron in target_neurons]

        # Two split sources may not resolve onto the
        # same tracked neuron.
        if len(set(target_neurons)) != len(target_neurons):
            raise ValueError(
                "Result components must belong to " "different neuron identities."
            )

        # Any occupied destination slot must be one
        # of the origins which is about to be retired.
        for neuron_id in target_neurons:

            current_fp = int(self.assignments.ids[neuron_id, target_session_id])

            if current_fp >= 0 and neuron_id not in origin_neurons:
                raise ValueError(
                    f"Neuron {neuron_id} already has "
                    f"a component in session "
                    f"{target_session_id}."
                )

        def component_ref(
            component: NeuronComponent,
        ) -> dict:

            fp_id = self.state.get_footprint_from_component(component)

            if fp_id is None or fp_id < 0:
                raise ValueError(f"{component} has no assigned footprint.")

            return {
                "session_id": int(component.session_id),
                "neuron_id": int(component.neuron_id),
                "footprint_id": int(fp_id),
            }

        origin_refs = [component_ref(component) for component in request.origin]
        source_refs = [component_ref(component) for component in sources]

        # ============================================
        # Retire original detections
        # ============================================
        for component in request.origin:
            self.exclude_component(component, notify=False)

        # ============================================
        # Create replacement synthetic components
        # ============================================

        added_components = []
        for source, neuron_id in zip(sources, target_neurons):

            added_component = self.add_synthetic_component(
                source=source,
                target_session_id=target_session_id,
                target_neuron_id=neuron_id,
                notify=False,
            )
            added_components.append(added_component)

        # Only expose the finished transaction to GUI /
        # statistics.
        self.state.assignments = self.assignments.ids
        self.assignments.register_manipulation(
            manipulation_type=request.type,
            origin=origin_refs,
            sources=source_refs,
            results=[component_ref(comp) for comp in added_components],
            affected_neurons=target_neurons,
        )

        self.notify_change(("assignments", -1))

    def restore_manipulations(self) -> None:
        """
        Restore session-level effects of saved split/merge manipulations.

        assignments.ids is assumed to already contain the final assignment
        state. This method only reconstructs changes to SessionData:
        - origin footprints become excluded
        - synthetic result footprints are recreated
        """

        if self.assignments is None:
            return

        for manipulation_id in sorted(self.assignments.manipulations):
            manipulation = self.assignments.manipulations[manipulation_id]

            # -----------------------------------------
            # Restore excluded origin footprints
            # -----------------------------------------

            for ref in manipulation.get("origin", []):
                session_id = int(ref["session_id"])
                footprint_id = int(ref["footprint_id"])

                session = self.sessions[session_id]

                if footprint_id >= session.n_neurons:
                    raise ValueError(
                        f"Manipulation {manipulation_id} "
                        f"references missing origin footprint "
                        f"{footprint_id} in session {session_id}."
                    )

                session.included[footprint_id] = False

            # -----------------------------------------
            # Recreate synthetic result footprints
            # -----------------------------------------

            sources = manipulation.get("sources", [])
            results = manipulation.get("results", [])

            if len(sources) != len(results):
                raise ValueError(
                    f"Manipulation {manipulation_id} has "
                    f"{len(sources)} sources but "
                    f"{len(results)} results."
                )

            for source_ref, result_ref in zip(sources, results):
                self._restore_synthetic_component(
                    source_ref=source_ref,
                    result_ref=result_ref,
                    manipulation_id=manipulation_id,
                )

    def _restore_synthetic_component(
        self, *, source_ref: dict, result_ref: dict, manipulation_id: int
    ) -> None:

        source_session_id = int(source_ref["session_id"])
        source_fp_id = int(source_ref["footprint_id"])

        target_session_id = int(result_ref["session_id"])
        target_fp_id = int(result_ref["footprint_id"])

        source_session = self.sessions[source_session_id]
        target_session = self.sessions[target_session_id]

        if source_fp_id >= source_session.n_neurons:
            raise ValueError(
                f"Manipulation {manipulation_id} "
                f"references missing source footprint "
                f"{source_fp_id} in session "
                f"{source_session_id}."
            )

        # Already restored, e.g. restore_manipulations()
        # was called twice in the same process.
        if target_fp_id < target_session.n_neurons:

            if (
                target_fp_id < len(target_session.synthetic)
                and target_session.synthetic[target_fp_id]
            ):
                return

            raise ValueError(
                f"Manipulation {manipulation_id} expects "
                f"synthetic footprint {target_fp_id} in "
                f"session {target_session_id}, but that "
                f"footprint ID already exists and is not synthetic."
            )

        # Synthetic footprint IDs should be reproduced
        # exactly in original append order.
        if target_fp_id != target_session.n_neurons:
            raise ValueError(
                f"Cannot restore manipulation "
                f"{manipulation_id}: expected next "
                f"footprint ID {target_session.n_neurons}, "
                f"but saved result uses {target_fp_id}."
            )

        footprint = source_session.footprints[:, source_fp_id].copy()

        # Use the SAME low-level helper that your live
        # split/merge implementation uses here.
        new_fp_id = target_session.append_synthetic_component(footprint)

        if new_fp_id != target_fp_id:
            raise RuntimeError(
                f"Restored footprint ID {new_fp_id}, " f"expected {target_fp_id}."
            )

    def set_review_status(self, neuron_ids, status: ReviewStatus):
        if self.assignments is None:
            return
        neuron_ids = np.unique(np.atleast_1d(neuron_ids).astype(int))
        valid = self.assignments.union.included[neuron_ids]

        neuron_ids = neuron_ids[valid]

        self.assignments.review_status[neuron_ids] = int(status)
        self.notify_change(("review_status", -1))
