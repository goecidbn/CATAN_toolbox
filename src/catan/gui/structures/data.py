from typing import Dict, Optional, Tuple, List, Union
from catan.tracking.structures.model import Model
import numpy as np
from pathlib import Path

from catan import Tracking
from . import AppState, StatisticDisplayConfig
from .request_handler import RequestHandler
from catan.core.io import (
    inspect_file,
    evaluate_fields_compatibility,
    load_file,
    get_backend,
)
from catan.core.structures import NeuronComponent, SessionData, sessiondata_type
from catan.core.structures.load_config import FieldSpec
from catan.tracking.structures import Assignments, ReviewStatus
from catan.tracking.realignment import build_realignment_update
from catan.gui.panels.colors import CyclicColorMap

from catan.gui.data.statistics.engine import StatisticEngine
from catan.gui.data.statistics.registry import build_statistics_registry

from catan.gui.background_tasks.runtime import current_task_context


class Data(Tracking):

    root: str

    def __init__(self, state: AppState):

        self.state = state
        self.correct_rotation = state.settings.value(
            "alignment/correct_rotation", False, type=bool
        )

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

        self._model_fit_requested = False
        self.state.tasks.scheduling_settled.connect(self._try_fit_after_loading)

    def is_available(self, what: List[str] | None = None) -> bool:

        if len(self.sessions) == 0:
            return False

        assignments = self.assignments

        if assignments is None or assignments.union is None:
            return False

        union = assignments.union

        n_neurons = assignments.ids.shape[0]

        if n_neurons == 0:
            return False

        # All neuron-level structures must describe the
        # same neuron population.
        if union.n_neurons != n_neurons:
            return False

        if union.footprints.shape[1] != n_neurons:
            return False

        if len(union.included) != n_neurons:
            return False

        if len(assignments.review_status) != n_neurons:
            return False

        if union.synthetic is not None and len(union.synthetic) != n_neurons:
            return False

        if what is None:
            return True

        if "current_session" in what:

            if self.current_session is None:
                return False

        return True

    # def is_available(self, what: List[str] | None = None) -> bool:

    #     available = True
    #     available &= len(self.sessions) > 0
    #     available &= not (self.assignments is None or self.assignments.union is None)
    #     if not available:
    #         return available
    #     available &= (
    #         len(self.assignments.union.included) == self.assignments.union.n_neurons
    #     )

    #     if what is None:
    #         return available

    #     if "current_session" in what:
    #         available &= self.current_session is not None

    #     return available

    def notify_change(self, change):
        self.state.data_version += 1
        self.state.data_changed.emit(change)

    def mark_geometry_changed(self, session_id: int):
        affected_paths = super().mark_geometry_changed(session_id)

        self.notify_change(("session", -1))

        return affected_paths

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

    def session_field_available(
        self, session_id: int, group_name: str, field_name: str
    ) -> bool:

        session = self.sessions[session_id]

        if session.path is None or session.source_config is None:
            return False

        group = session.source_config.groups.get(group_name)

        if group is None:
            return False

        spec = group.fields.get(field_name)

        if spec is None:
            return False

        return evaluate_fields_compatibility(
            session.path, {group_name: {field_name: spec}}
        )

    @staticmethod
    def _missing_session_fields(session, fields_to_load):
        return {
            group: dict(fields)
            for group, fields in fields_to_load.items()
            if fields
            and not (
                group in {"spatial", "traces", "quality"}
                and session.status[f"{group}_loaded"]
            )
        }

    def _load_session_fields(self, session_id: int, fields_to_load):
        session = self.sessions[session_id]

        fields_to_load = self._missing_session_fields(session, fields_to_load)
        if not fields_to_load:
            return

        if "spatial" in fields_to_load:
            session.params["correct_rotation"] = self.correct_rotation

        session.load_data(
            fields_to_load,
            alignment_references=(self.alignment_references_for_session(session_id)),
            ctx=current_task_context(),
        )

        self.notify_change(("session", session_id))

        if "traces" in fields_to_load:
            self.notify_change(("traces", session_id))

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
                to_present = not session.status[f"{which}_loaded"]

            if not to_present:
                session.clean_data(which)
                self.notify_change(("session", session_id))

                if which == "traces":
                    self.notify_change(("traces", session_id))

                return

        self._load_session_fields(session_id, fields_to_load)

    def queue_load_data(self, session_id: int, *, fields_to_load=None, finished=None):

        session = self.sessions[session_id]
        if session.source_config is None:
            raise ValueError("No load configuration selected.")

        if fields_to_load is None:
            fields_to_load = session.source_config.get_fields_to_load()

        self.state.tasks.start(
            "loading",
            f"Loading data for {session.name}",
            self._load_session_fields,
            session_id=session_id,
            fields_to_load=fields_to_load,
            finished=finished,
        )

    def queue_registration_actions(
        self,
        session_id: int,
        actions: set[str],
        *,
        background_mode: str = "configured",
        on_alignment_error=None,
    ):
        session = self.sessions[session_id]

        actions = set(actions)

        load_all = "load_all" in actions
        load_requested = load_all or "load_data" in actions

        fields_to_load = None

        # ==================================================
        # Data loading
        # ==================================================
        if load_requested:
            if session.source_config is None:
                self.state.issue(
                    "warning",
                    "Cannot load session",
                    "No load configuration is available.",
                )
                return

            fields_to_load = session.source_config.get_fields_to_load(
                enabled_only=not load_all
            )

            # Make an independent mapping because the
            # registration-specific background choice must
            # not alter the load configuration itself.
            fields_to_load = {
                group_name: dict(fields)
                for group_name, fields in fields_to_load.items()
            }
            fields_to_load = self._missing_session_fields(session, fields_to_load)

            if not fields_to_load:
                self._continue_registration_actions(
                    session_id, actions, on_alignment_error=on_alignment_error
                )
                return

            if background_mode == "footprints":
                fields_to_load.get("spatial", {}).pop("background", None)

            if not evaluate_fields_compatibility(session.path, fields_to_load):
                self.state.issue(
                    "warning",
                    "Session data incompatible",
                    (
                        "The fields selected for loading "
                        "are not compatible with their "
                        "configured sources."
                    ),
                )
                return

            self.queue_load_data(
                session_id,
                fields_to_load=fields_to_load,
                finished=lambda: self._continue_registration_actions(
                    session_id, actions, on_alignment_error=on_alignment_error
                ),
            )
        else:
            self._continue_registration_actions(
                session_id, actions, on_alignment_error=on_alignment_error
            )

    def _continue_registration_actions(
        self,
        session_id: int,
        actions: set[str],
        *,
        on_alignment_error=None,
    ):

        session = self.sessions[session_id]

        register_model = "register_model" in actions
        track_neurons = "track_neurons" in actions

        if not (register_model or track_neurons):
            return

        if not session.status["spatial_loaded"]:
            return

        # --------------------------------------------------
        # Recoverable alignment error
        # --------------------------------------------------
        if not session.status["aligned"]:

            self.notify_change(("session", session_id))

            if on_alignment_error is not None:
                on_alignment_error(
                    session_id,
                    (None if session.remap is None else session.remap.report),
                )

            # Important:
            # only skip dependent processing for THIS session.
            return

        # --------------------------------------------------
        # Normal processing
        # --------------------------------------------------
        if register_model:

            callback = None

            if track_neurons:
                callback = lambda: self.queue_assign_neurons(session_id)

            self.queue_update_model(session_id, callback=callback)

            return

        if track_neurons:
            self.queue_assign_neurons(session_id)

    def queue_process_alignments(self, *, session_ids, finished=None):
        """Repair required alignments and stale predecessors in session order."""
        tasks = self.state.tasks

        if any(
            tasks.current_task(group) is not None or tasks.queued_tasks(group)
            for group in tasks.GROUPS
        ):
            self.state.issue(
                "warning",
                "Alignment update not started",
                "Finish or cancel existing tasks first.",
            )
            return

        try:
            required = sorted(set(session_ids))

            if any(index < 0 or index >= len(self.sessions) for index in required):
                raise IndexError("Invalid alignment session selection.")

            last = max(required, default=-1)

            pending = [
                index
                for index in range(last + 1)
                if self.alignment_is_stale(index)
                or (index in required and not self.sessions[index].status["aligned"])
            ]

            for index in pending:
                session = self.sessions[index]

                if not session.status["spatial_loaded"]:
                    raise ValueError(f"Load spatial data for session {index} first.")

                if session.background_template is None:
                    raise ValueError(
                        f"Session {index} has no original background template."
                    )

        except (ValueError, IndexError) as exc:
            self.state.issue("warning", "Alignment update not started", str(exc))
            return

        if not pending:
            if finished is not None:
                finished()
            return

        # This explicit processing chain owns subsequent model fitting.
        self._model_fit_requested = False

        session_id = pending[0]
        expected_version = self.state.data_version

        def run():
            session = self.sessions[session_id]
            template = session.background_template.copy()

            remap = self.propose_session_remapping(
                session_id,
                background_template=template,
            )

            if not remap.report.success:
                raise ValueError(
                    f"Alignment failed for session {session_id}: "
                    f"{remap.report.reason}. Dependent processing stopped."
                )

            return build_realignment_update(
                self,
                session_id,
                background_template=template,
                remap=remap,
                background_spec=None,
            )

        def publish(result):
            if self._publish_realignment_update(result, expected_version):
                # Recheck dependencies after publishing each alignment.
                self.queue_process_alignments(
                    session_ids=required,
                    finished=finished,
                )

        return tasks.start(
            "loading",
            f"Update alignment for {self.sessions[session_id].name}",
            run,
            on_result=publish,
        )

    def queue_process_model(
        self,
        *,
        session_ids=None,
        from_session_id=None,
        mode="pending",
        session_distances=(1,),
        finished=None,
    ):
        tasks = self.state.tasks

        # This explicit processing run owns count updates and its final fit.
        # Avoid overlapping an existing loading/count/registration operation.
        if any(
            tasks.current_task(group) is not None or tasks.queued_tasks(group)
            for group in tasks.GROUPS
        ):
            self.state.issue(
                "warning",
                "Model update not started",
                "Finish or cancel existing tasks first.",
            )
            return

        try:
            plan = self.plan_model_update(
                session_ids=session_ids,
                from_session_id=from_session_id,
                mode=mode,
                session_distances=session_distances,
            )

            if plan["load_sessions"]:
                raise ValueError(
                    f"Load spatial data for sessions {plan['load_sessions']} first."
                )

        except (ValueError, IndexError) as exc:
            self.state.issue(
                "warning",
                "Model update not started",
                str(exc),
            )
            return

        if plan["alignment_sessions"]:
            return self.queue_process_alignments(
                session_ids=plan["alignment_sessions"],
                finished=lambda: self.queue_process_model(
                    session_ids=plan["session_ids"],
                    mode=mode,
                    session_distances=plan["session_distances"],
                    finished=finished,
                ),
            )

        self._model_fit_requested = False

        if not (plan["same_sessions"] or plan["cross_pairs"] or plan["check_fit"]):
            if finished is not None:
                finished()
            return

        def run():
            try:
                # Rebuild the plan at execution time.
                return self.process_model_updates(
                    session_ids=plan["session_ids"],
                    mode=mode,
                    session_distances=plan["session_distances"],
                )
            finally:
                # Also expose any completed records if a later step fails.
                self.notify_change(("model", -1))

        return tasks.start(
            "model update",
            (
                "Process pending model evidence"
                if mode == "pending"
                else f"Process model evidence ({mode})"
            ),
            run,
            finished=finished,
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
        # Called by the model-count completion callback.
        # TaskManager checks the request after callbacks finish.
        self._model_fit_requested = True

    def _try_fit_after_loading(self):
        if not self._model_fit_requested:
            return

        tasks = self.state.tasks

        if any(
            tasks.current_task(group) is not None or tasks.queued_tasks(group)
            for group in ("loading", "model update")
        ):
            return

        self._model_fit_requested = False

        if self.model is None or self.model.loaded:
            return

        counts = self.model.aggregate_counts(
            session_order=[
                str(session.path)
                for session in self.sessions
                if session.path is not None
            ],
            session_distances=(1,),
        )

        # Match the current minimum in Model.fit_model_to_counts().
        if counts["cross"][..., 0].sum() < 20:
            return

        tasks.start(
            "model update",
            "Fit model to data",
            self.fit_model,
        )

    def fit_model(self, **kwargs):
        if self.model is None:
            raise ValueError("No model to fit. Please add a model before fitting.")

        counts = super().fit_model(**kwargs)

        self.notify_change(("model", -1))  # Notify that model has changed
        return counts

    def propose_session_remapping(self, session_id: int, *, background_template=None):

        session = self.sessions[session_id]
        if any(self.alignment_is_stale(i) for i in range(session_id)):
            raise ValueError("Update earlier stale alignments first.")

        references = self.alignment_references_for_session(session_id)

        return session.propose_remapping(
            references,
            background_template=(background_template),
            use_optical_flow=False,
            correct_rotation=self.correct_rotation,
        )

    def propose_background_remapping(
        self,
        session_id: int,
        *,
        source_path: str | Path,
        field_path: str,
        source="dataset",
        attribute=None,
    ):

        session = self.sessions[session_id]

        spec = FieldSpec(
            path=field_path, source=source, attribute=attribute, required=True
        )

        data = load_file(source_path, {"spatial": {"background": spec}})

        background = data["spatial"]["background"]

        candidate_template = session.prepare_background_template(background)

        candidate_remap = self.propose_session_remapping(
            session_id, background_template=(candidate_template)
        )

        return (candidate_template, candidate_remap)

    def queue_commit_session_realignment(
        self,
        session_id,
        *,
        background_template,
        remap,
        background_spec,
        expected_version,
    ):
        if self.state.data_version != expected_version:
            self.state.issue(
                "warning",
                "Alignment preview outdated",
                "The data changed. Generate a new alignment preview.",
            )
            return

        tasks = self.state.tasks
        if any(tasks.current.values()) or any(
            tasks.queued_tasks(group) for group in tasks.current
        ):
            self.state.issue(
                "warning",
                "Realignment not started",
                "Finish or cancel existing tasks first.",
            )
            return

        def publish(result):
            self._publish_realignment_update(result, expected_version)

        tasks.start(
            "loading",
            f"Commit alignment for {self.sessions[session_id].name}",
            build_realignment_update,
            self,
            session_id,
            background_template=background_template,
            remap=remap,
            background_spec=background_spec,
            on_result=publish,
        )

    def _publish_realignment_update(self, result, expected_version):
        if (
            self.state.data_version != expected_version
            or len(self.sessions) != len(result["sessions"])
            or any(a is not b for a, b in zip(self.sessions, result["sessions"]))
        ):
            self.state.issue(
                "warning",
                "Realignment discarded",
                "The data changed while preparing the new geometry. "
                "Generate a new preview.",
            )
            return False

        for sid, values in result["updates"].items():
            self.sessions[sid].__dict__.update(values)

        Tracking.mark_geometry_changed(self, result["session_id"])

        extra_ids = set(result["updates"]) - {result["session_id"]}
        extra_paths = {
            str(self.sessions[sid].path)
            for sid in extra_ids
            if self.sessions[sid].path is not None
        }

        for sid in extra_ids:
            self._processing_state(sid).geometry_revision += 1

        for model in self._model.values():
            model.invalidate_counts_for_paths(extra_paths)

        # Dependent synthetic copies can precede the realigned session.
        first = min(result["updates"])
        assignment_paths = {
            str(item.path) for item in self.sessions[first:] if item.path is not None
        }

        for name in self._assignments:
            self._stale_assignments.setdefault(name, set()).update(assignment_paths)

        self.notify_change(("session", -1))
        self.notify_change(("assignments", -1))
        return True

    def register_session(self, from_file: str | Path, **kwargs) -> list[int]:
        backend = get_backend(from_file)

        with backend.open_read(from_file) as ref:
            object_type = backend.get_attribute(ref, "/", "object_type")

        if object_type in {"SessionData", "SessionList"}:
            sessions = super().load_session_data(from_file)
        else:
            sessions = [SessionData(path=str(Path(from_file).expanduser().resolve()))]

        registered_ids = []

        for session in sessions:
            session_id = super().register_session(
                from_data=session,
                **kwargs,
            )
            registered_ids.append(session_id)

            saved_state = session.__dict__.pop("_restored_processing", None)
            if saved_state is not None:
                processing = self._processing_state(session_id)
                processing.geometry_revision = int(saved_state["geometry_revision"])
                processing.alignment_stale = bool(saved_state["alignment_stale"])

            # Preserve restored field mappings and source overrides.
            if session.source_config is None:
                session.source_config = self.state.config_manager.suggest_config_for(
                    path=session.path,
                    source_type="session",
                )

            if not session.name:
                session.name = Path(session.path).parent.name

            self.state.session_color = (
                session_id,
                self.session_colors.next(),
            )

            if self.state.current_session_id is None:
                self.state.current_session_id = session_id

            self.notify_change(("session_added", session_id))

        return registered_ids

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

    def queue_process_assignments(
        self,
        *,
        session_ids=None,
        from_session_id=None,
        mode="pending",
        _model_attempted=False,
    ):
        tasks = self.state.tasks

        if any(
            tasks.current_task(group) is not None or tasks.queued_tasks(group)
            for group in tasks.GROUPS
        ):
            self.state.issue(
                "warning",
                "Assignment update not started",
                "Wait for existing tasks to finish, or cancel "
                "queued tasks before starting this rebuild.",
            )
            return

        try:
            plan = self.plan_assignment_update(
                session_ids=session_ids,
                from_session_id=from_session_id,
                mode=mode,
            )

            if not plan["register_sessions"]:
                return

            if plan["load_sessions"]:
                raise ValueError(
                    f"Load spatial data for sessions {plan['load_sessions']} first."
                )

        except (ValueError, IndexError) as exc:
            self.state.issue(
                "warning",
                "Assignment update not started",
                str(exc),
            )
            return

        if plan["alignment_sessions"]:
            return self.queue_process_alignments(
                session_ids=plan["alignment_sessions"],
                finished=lambda: self.queue_process_assignments(
                    session_ids=plan["session_ids"],
                    mode=mode,
                    _model_attempted=_model_attempted,
                ),
            )

        if plan["model_state"] in ("missing", "stale"):
            if _model_attempted:
                self.state.issue(
                    "warning",
                    "Assignment update stopped",
                    "The model is still unavailable or outdated after updating "
                    "its counts. There may be insufficient cross-session evidence.",
                )
                return

            required = sorted(
                set(plan["preserved_sessions"] + plan["register_sessions"])
            )

            return self.queue_process_model(
                session_ids=required,
                mode="pending",
                finished=lambda: self.queue_process_assignments(
                    session_ids=plan["session_ids"],
                    mode=mode,
                    _model_attempted=True,
                ),
            )

        source = self.assignments
        assignment_name = self.current_assignments
        model = self.model
        data_version = self.state.data_version

        def publish(result):
            if result is None:
                return

            if (
                self.state.data_version != data_version
                or self.assignments is not source
                or self.current_assignments != assignment_name
                or self.model is not model
            ):
                self.state.issue(
                    "warning",
                    "Assignment update discarded",
                    "Data or the active model/assignments changed "
                    "during processing. Run the update again.",
                )
                return

            candidate = result["candidate"]
            completed_plan = result["plan"]

            # Preserve object identity for existing GUI/config references.
            source.__dict__.update(candidate.__dict__)

            completed_paths = {
                str(self.sessions[index].path)
                for index in completed_plan["register_sessions"]
                if self.sessions[index].path is not None
            }
            self._stale_assignments.setdefault(
                assignment_name, set()
            ).difference_update(completed_paths)

            self.state.apply_assignment_rebuild(
                source.ids,
                result["neuron_id_map"],
                from_session_id=completed_plan["from_session_id"],
            )
            self.notify_change(("assignments", -1))

        return tasks.start(
            "calculating",
            "Process pending neuron registrations",
            self.build_assignment_update,
            session_ids=plan["session_ids"],
            mode=mode,
            on_result=publish,
        )

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
                session.status["aligned"]
                and not self.alignment_is_stale(session_id)
                and (
                    session_id == 0
                    or (
                        self.model is not None
                        and self.model.fitted
                        and not self.model.fit_stale
                        and not self._model_fit_requested
                        and all(
                            self.state.tasks.current_task(group) is None
                            and not self.state.tasks.queued_tasks(group)
                            for group in ("loading", "model update")
                        )
                    )
                )
            ),
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
