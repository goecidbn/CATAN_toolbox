from typing import Dict, Optional, Tuple, List, Union
from catan.tracking.structures.model import Model
import numpy as np
from pathlib import Path
from functools import wraps
from copy import copy, deepcopy
import hashlib, json, os, re, tempfile

from catan import Tracking
from . import AppState, StatisticDisplayConfig
from .request_handler import RequestHandler
from catan.core.changes import (
    Change,
    ChangeKind as C,
    DataChange,
    ASSIGNMENT_CONTENT_CHANGES,
)
from catan.core.io import (
    inspect_file,
    evaluate_fields_compatibility,
    load_file,
    get_backend,
)
from catan.core.structures import NeuronComponent, SessionData, sessiondata_type
from catan.core.structures.load_config import FieldSpec, LoadConfig
from catan.core.structures.session_snapshot import (
    prepare_session_snapshot,
)
from catan.core.structures.inclusion import (
    has_exclusions,
    inclusion_mask,
    inclusion_to_restore,
)
from catan.core.io.isolated_read import read_fields, read_operation

from catan.tracking.structures import Assignments, ReviewStatus
from catan.tracking.realignment import build_realignment_update

from catan.gui.panels.colors import CyclicColorMap
from catan.gui.data.statistics.engine import StatisticEngine
from catan.gui.data.statistics.registry import build_statistics_registry
from catan.gui.background_tasks.runtime import TaskCancelled, current_task_context
from catan.gui.background_tasks.file_writes import (
    save_bytes_task,
    save_session_snapshots_task,
    save_prepared_file_task,
)

from catan.gui.data.assignment_comparison import AssignmentComparison


def read_assignments_provenance(path, *, ctx=None):
    """Sessions and model stored with saved assignments, or None.

    {"recipe": loading recipe document or None, "model_path": str or None,
    "sessions_file": CATAN session file or None}.
    Call from a background worker.
    """
    from catan.tracking.structures.assignments import PROVENANCE_PATHS

    fields = {
        "provenance": {
            key: FieldSpec(path=field_path, required=False, exposed=False)
            for key, field_path in PROVENANCE_PATHS.items()
        }
    }

    try:
        result = read_fields(path, fields, ctx=ctx, timeout=120.0)
        values = (result or {}).get("provenance") or {}
    except TaskCancelled:
        raise
    except Exception:
        # Older or foreign files carry no provenance.
        return None

    def text(value):
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, np.ndarray) and value.size == 1:
            value = value.item()
        return value if isinstance(value, str) and value else None

    recipe = text(values.get("loading_recipe"))
    model_path = text(values.get("model_path"))
    sessions_file = text(values.get("sessions_file"))

    try:
        recipe = None if recipe is None else json.loads(recipe)
    except ValueError:
        recipe = None

    if recipe is None and model_path is None and sessions_file is None:
        return None

    return {"recipe": recipe, "model_path": model_path, "sessions_file": sessions_file}


def read_overview_background(path, *, ctx=None):
    """Overview background stored with saved assignments
    (background-only SessionData), or None. Call from a background worker."""
    from catan.tracking.structures.assignments import OVERVIEW_BACKGROUND_PATH
    from catan.core.structures.session_snapshot import restore_background_snapshot

    def spec(key):
        return FieldSpec(
            path=f"{OVERVIEW_BACKGROUND_PATH}/{key}", required=False, exposed=False
        )

    try:
        result = read_fields(
            path, {"background": {"document": spec("document")}}, ctx=ctx
        )
        document = ((result or {}).get("background") or {}).get("document")
    except TaskCancelled:
        raise
    except Exception:
        return None

    if isinstance(document, np.ndarray) and document.size == 1:
        document = document.item()
    if isinstance(document, bytes):
        document = document.decode("utf-8")
    if not isinstance(document, str) or not document:
        return None

    # Array keys referenced by the document ({"__array__": "a0000"}).
    keys = sorted(set(re.findall(r'"__array__": "(a\d{4})"', document)))
    arrays = read_fields(
        path, {"arrays": {key: spec(key) for key in keys}}, ctx=ctx
    )["arrays"]

    return restore_background_snapshot(document, arrays)


def after_display_tasks(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if self.state.tasks.defer_for_background(lambda: method(self, *args, **kwargs)):
            return

        return method(self, *args, **kwargs)

    return wrapped


class Data(Tracking):

    @property
    def root(self) -> str:
        return getattr(self, "_root", "")

    @root.setter
    def root(self, value):
        from catan.gui.GUI_elements.utils.paths import set_display_root

        self._root = value
        set_display_root(value)

    def __init__(self, state: AppState):

        self.state = state
        self.correct_rotation = state.settings.value(
            "alignment/correct_rotation", False, type=bool
        )
        # Shared border: sessions a pixel may lie outside of (common_spatial_coverage).
        self.border_tolerance = max(
            0, state.settings.value("alignment/border_tolerance", 0, type=int)
        )

        super().__init__()

        # Display-only overview background (queue_load_overview_background).
        self.overview_background = None

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
        self._model_fit_batches = []
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

    def notify_change(
        self,
        *changes,
        session_id=None,
        session_paths=None,
        neuron_ids=None,
        fields=None,
    ):
        if session_id is not None:
            if session_paths is not None:
                raise ValueError("Use session_id or session_paths, not both.")

            path = self.sessions[session_id].path
            session_paths = None if path is None else {str(path)}

        entries = []

        for change in changes:
            if isinstance(change, Change):
                # Explicit entries retain their individual scopes.
                entries.append(change)

            elif isinstance(change, C):
                entries.append(
                    Change(
                        change,
                        session_paths=session_paths,
                        neuron_ids=neuron_ids,
                        fields=fields,
                    )
                )

            else:
                raise TypeError("notify_change expects ChangeKind or Change entries.")

        event = DataChange(tuple(entries))

        # Preserve the existing guards for asynchronous processing.
        self.state.data_version += 1
        self.state.data_changed.emit(event)

    def mark_geometry_changed(self, session_id: int):
        affected_paths = super().mark_geometry_changed(session_id)

        path = self.sessions[session_id].path

        self.notify_change(
            Change(
                C.FOOTPRINT_GEOMETRY,
                session_paths=None if path is None else {str(path)},
            ),
            Change(C.PROCESSING_STATUS),
        )

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

        recipe_options = getattr(session, "_recipe_options", {})

        orientation = recipe_options.get(
            "background_orientation",
            getattr(self, "background_orientation", "auto"),
        )
        correct_rotation = recipe_options.get(
            "correct_rotation", bool(self.correct_rotation)
        )

        if "spatial" in fields_to_load:
            session.params["correct_rotation"] = correct_rotation

        session.load_data(
            fields_to_load,
            alignment_references=self.alignment_references_for_session(session_id),
            background_orientation=orientation,
            expected_dims=next(
                (
                    tuple(other.dims)
                    for other in self.sessions
                    if other is not session and other.status["spatial_loaded"]
                ),
                None,
            ),
            ctx=current_task_context(),
        )

        if "spatial" in fields_to_load:
            self._report_flow_fallback(session)
            session._last_load_options = {
                "background_orientation": orientation,
                "correct_rotation": correct_rotation,
                "background_mode": getattr(
                    session, "_load_background_mode", "configured"
                ),
            }

        self.notify_change(
            C.DATA_AVAILABILITY,
            C.PROCESSING_STATUS,
            session_id=session_id,
            fields=set(fields_to_load),
        )

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
                self.notify_change(
                    C.DATA_AVAILABILITY,
                    C.PROCESSING_STATUS,
                    session_id=session_id,
                    fields={which},
                )

                return

        self._load_session_fields(session_id, fields_to_load)

    def queue_load_data(
        self, session_id: int, *, fields_to_load=None, finished=None, batch=None
    ):
        if batch is not None and batch.cancelled:
            return

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
            batch=batch,
            prepend=True,
        )

    def queue_registration_actions(
        self,
        session_id: int,
        actions: set[str],
        *,
        background_mode: str = "configured",
        on_alignment_error=None,
        batch=None,
    ):
        if batch is not None and batch.cancelled:
            return

        session = self.sessions[session_id]

        actions = set(actions)

        load_all = "load_all" in actions
        load_requested = load_all or "load_data" in actions

        fields_to_load = None

        # ==================================================
        # Data loading
        # ==================================================
        if load_requested:
            session._load_background_mode = background_mode
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
                    session_id,
                    actions,
                    on_alignment_error=on_alignment_error,
                    batch=batch,
                )
                return

            if background_mode == "footprints":
                fields_to_load.get("spatial", {}).pop("background", None)

            self.queue_load_data(
                session_id,
                fields_to_load=fields_to_load,
                finished=lambda: self._continue_registration_actions(
                    session_id,
                    actions,
                    on_alignment_error=on_alignment_error,
                    batch=batch,
                ),
                batch=batch,
            )
        else:
            self._continue_registration_actions(
                session_id,
                actions,
                on_alignment_error=on_alignment_error,
                batch=batch,
            )

    def _continue_registration_actions(
        self,
        session_id: int,
        actions: set[str],
        *,
        on_alignment_error=None,
        batch=None,
    ):
        if batch is not None and batch.cancelled:
            return

        session = self.sessions[session_id]

        register_model = "register_model" in actions
        track_neurons = "track_neurons" in actions

        if not session.status["spatial_loaded"]:
            return

        # --------------------------------------------------
        # Recoverable alignment error
        # --------------------------------------------------
        if not session.status["aligned"]:

            self.notify_change(
                C.PROCESSING_STATUS,
                session_id=session_id,
            )

            if on_alignment_error is not None:
                on_alignment_error(
                    session_id,
                    (None if session.remap is None else session.remap.report),
                )

            # Important:
            # only skip dependent processing for THIS session.
            return

        if not (register_model or track_neurons):
            return

        # --------------------------------------------------
        # Normal processing
        # --------------------------------------------------
        if register_model:
            callback = None
            if track_neurons:
                callback = lambda: self.queue_assign_neurons(session_id, batch=batch)

            self.queue_update_model(
                session_id,
                callback=callback,
                batch=batch,
            )
            return

        if track_neurons:
            self.queue_assign_neurons(session_id, batch=batch)

    @after_display_tasks
    def queue_realign_all(self, *, replace_manual=False):
        """Realign all sessions after the first (reference) in session order.
        ``replace_manual``: recompute manually set alignments too (otherwise
        they keep their transform; only dependent geometry is refreshed)."""
        if len(self.sessions) < 2:
            return

        if self.state.tasks.processing_busy():
            self.state.issue(
                "warning",
                "Realignment not started",
                "Finish or cancel existing tasks first.",
            )
            return

        # Marks all later sessions outdated (and notifies).
        self.mark_geometry_changed(0)

        return self.queue_process_alignments(
            session_ids=list(range(1, len(self.sessions))),
            replace_manual=replace_manual,
        )

    @after_display_tasks
    def queue_process_alignments(
        self, *, session_ids, finished=None, replace_manual=False
    ):
        """Repair required alignments and stale predecessors in session order."""
        tasks = self.state.tasks

        if tasks.processing_busy():
            details = tasks.describe_work()
            self.state.logger.warning("Alignment processing blocked:\n%s", details)
            self.state.issue(
                "warning",
                "Alignment update not started",
                "Finish or cancel existing tasks first.\n\n" + details,
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

            if (
                not replace_manual
                and session.remap is not None
                and session.remap.method == "manual"
            ):
                # Manual transforms are specified in the common/global frame.
                # Refresh dependent geometry without replacing the chosen transform.
                remap = session.remap
            else:
                remap = self.propose_session_remapping(
                    session_id,
                    background_template=template,
                )

            if not remap.report.success:
                message = (
                    f"Automatic alignment of {session.name!r} needs confirmation.\n"
                    f"{remap.confirmation}\n"
                    "Check the proposal, then use it, adjust it or remove the session."
                    if remap.confirmation
                    else f"Automatic alignment failed for {session.name!r}.\n"
                    f"Reason: {remap.report.reason or 'unknown'}."
                )
                return {
                    "alignment_failure": (
                        session,
                        message + "\nDependent processing stopped.",
                        remap,
                    )
                }

            return self._build_realignment_update(
                session_id,
                background_template=template,
                remap=remap,
                background_spec=None,
            )

        def publish(result):
            if "alignment_failure" in result:
                session, message, proposal = result["alignment_failure"]
                if self.state.data_version == expected_version and any(
                    item is session for item in self.sessions
                ):
                    self.state.alignment_failed.emit(session, message, proposal)
                return

            if self._publish_realignment_update(result, expected_version):
                # Recheck dependencies after publishing each alignment.
                self.queue_process_alignments(
                    session_ids=required,
                    finished=finished,
                    replace_manual=replace_manual,
                )

        return tasks.start(
            "loading",
            f"Update alignment for {self.sessions[session_id].name}",
            run,
            on_result=publish,
        )

    @after_display_tasks
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
        if tasks.processing_busy():
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
                    ctx=current_task_context(),
                )
            finally:
                # Also expose any completed records if a later step fails.
                self.notify_change(
                    C.MODEL_COUNTS, C.MODEL_PARAMETERS, C.PROCESSING_STATUS
                )

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
        self, session_id: int, to_present: bool = True, callback=None, *, batch=None
    ):
        if batch is not None and batch.cancelled:
            return

        if self.model is not None and self.model.loaded:
            self.state.issue(
                "warning",
                "Model registration not allowed",
                f"Model '{self.current_model_name}' was loaded from file and cannot be updated. Please create a new model to fit to data.",
            )
            return
        session = self.sessions[session_id]

        def when_finished():
            if batch is not None and batch.cancelled:
                return

            self.fit_after_loading(batch=batch)

            if callback is not None:
                callback()

        self.state.tasks.start(
            "model update",
            f"Update model for {session.name}",
            self.update_counts,
            session_id=session_id,
            finished=when_finished,
            ready=lambda session=session: session.status["aligned"],
            batch=batch,
        )

    def update_counts(
        self,
        session_id: Optional[int] = None,
        **kwargs,
    ):
        super().update_counts_with_data(
            from_session_index=session_id,
            align_to_reference=True,
            ctx=current_task_context(),
        )
        self.notify_change(
            C.MODEL_COUNTS,
            C.INCLUSION,
            C.PROCESSING_STATUS,
            session_id=session_id,
        )

    def fit_after_loading(self, key: str = "model update", *, batch=None):
        if batch is not None and batch.cancelled:
            return

        if not any(item is batch for item in self._model_fit_batches):
            self._model_fit_batches.append(batch)

        self._model_fit_requested = True

    def _try_fit_after_loading(self):

        if self._model_fit_batches:
            self._model_fit_batches = [
                batch
                for batch in self._model_fit_batches
                if batch is None or not batch.cancelled
            ]
            if not self._model_fit_batches:
                self._model_fit_requested = False
                return

        if not self._model_fit_requested:
            return

        tasks = self.state.tasks

        if any(
            tasks.current_task(group) is not None
            or tasks.queued_tasks(group, include_deferred=False)
            for group in ("loading", "model update")
        ):
            return

        requests = self._model_fit_batches
        fit_batch = requests[0] if len(requests) == 1 else None
        self._model_fit_batches = []
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
            batch=fit_batch,
        )

    def fit_model(self, **kwargs):
        if self.model is None:
            raise ValueError("No model to fit. Please add a model before fitting.")

        counts = super().fit_model(**kwargs)

        self.notify_change(
            C.MODEL_PARAMETERS,
            C.PROCESSING_STATUS,
        )
        return counts

    def apply_model_preview(
        self,
        source,
        candidate,
        expected_version,
    ):
        if self.model is not source or self.state.data_version != expected_version:
            raise ValueError(
                "The model or data changed. " "Generate a new preview first."
            )

        if self.state.tasks.processing_busy() or self.state.tasks.processing_requested:
            raise ValueError("Wait for current processing to finish before applying.")

        fields = (
            "parameters",
            "distributions",
            "p_same",
            "distance_cutoff",
            "fitted",
            "fit_stale",
            "fit_used_fallback",
            "use_cdf",
        )

        source.__dict__.update({key: getattr(candidate, key) for key in fields})

        # Bind the interpolator to the live model, not its preview copy.
        source.set_f_same("joint")

        paths = {
            str(session.path) for session in self.sessions if session.path is not None
        }

        # Conservatively invalidate all assignment sets.
        for name in self._assignments:
            self._stale_assignments.setdefault(name, set()).update(paths)

        self.notify_change(
            C.MODEL_PARAMETERS,
            C.PROCESSING_STATUS,
        )

    def _source_loading_ready(self):
        tasks = self.state.tasks

        return (
            not tasks.processing_busy()
            and not tasks.processing_requested
            and not tasks._background_running()
            and not self._model_fit_requested
            and self.state.current_request is None
        )

    def queue_import_source(
        self,
        key,
        path,
        name,
        *,
        use_as_working=False,
        replace_empty_model=False,
        on_loaded=None,
    ):
        if key not in {"model", "assignments"}:
            raise ValueError(f"Unsupported source type: {key}")

        return self.state.tasks.queue_after_processing(
            f"Import {key}: {Path(path).name}",
            lambda: self._queue_import_source_now(
                key,
                path,
                name,
                use_as_working=use_as_working,
                replace_empty_model=replace_empty_model,
                on_loaded=on_loaded,
            ),
            ready=lambda: (
                not self._model_fit_requested and self.state.current_request is None
            ),
        )

    def _queue_import_source_now(
        self,
        key,
        path,
        name,
        *,
        use_as_working=False,
        replace_empty_model=False,
        on_loaded=None,
    ):
        if key not in {"model", "assignments"}:
            raise ValueError(f"Unsupported source type: {key}")

        existing = self._model if key == "model" else self._assignments

        if replace_empty_model:
            if key != "model" or not self.can_replace_empty_model(name):
                self.state.issue(
                    "warning",
                    "Model was not replaced",
                    "The requested model is no longer empty. "
                    "Load the file under a new name to preserve the existing model.",
                )
                return

        elif name in existing:
            self.state.issue(
                "warning",
                "Source not loaded",
                "Choose a name that is not already in use.",
            )
            return

        replacement_target = self._model.get(name) if replace_empty_model else None

        if use_as_working and (
            key != "assignments" or not self.can_use_working_assignments()
        ):
            self.state.issue(
                "warning",
                "Working assignments are not empty",
                "Load this file as a separate assignments dataset.",
            )
            return

        tasks = self.state.tasks

        if not self._source_loading_ready():
            return self.queue_import_source(
                key,
                path,
                name,
                use_as_working=use_as_working,
                replace_empty_model=replace_empty_model,
                on_loaded=on_loaded,
            )

        path = os.path.abspath(os.path.expanduser(os.fspath(path)))
        params = deepcopy(self.params)
        data_version = self.state.data_version

        previous_source = getattr(self, key)
        previous_name = (
            self.current_model_name if key == "model" else self.current_assignments
        )

        def run():
            ctx = current_task_context()

            if ctx is not None:
                ctx.check_cancelled()

            if key == "model":
                data = read_operation(
                    "model",
                    path,
                    ctx=ctx,
                    timeout=300.0,
                )

                # Build and validate a separate candidate. Its settings
                # must not mutate the live Tracking.params dictionary.
                candidate = Model._from_dict(data, params=params)
                candidate.path = path

            else:
                config = self.state.config_manager.suggest_config_for(
                    path=path,
                    source_type="assignments",
                    ctx=ctx,
                )

                check = read_operation(
                    "assignments_preflight",
                    path,
                    fields_to_load=(
                        config.get_fields_to_load() if config is not None else {}
                    ),
                    ctx=ctx,
                )

                candidate = Assignments()
                candidate._use_as_working = bool(use_as_working)
                candidate.path = path
                candidate.source_config = config
                candidate._load_config_problems = check["problems"]
                candidate.provenance = read_assignments_provenance(path, ctx=ctx)

            if ctx is not None:
                ctx.check_cancelled()

            return candidate

        def publish(candidate):
            current_name = (
                self.current_model_name if key == "model" else self.current_assignments
            )

            if (
                self.state.data_version != data_version
                or getattr(self, key) is not previous_source
                or current_name != previous_name
            ):
                self.state.issue(
                    "warning",
                    "Loaded source was not applied",
                    "The active data changed while loading. "
                    "Select the file again to load it into the current state.",
                )
                return

            if replace_empty_model and (
                self._model.get(name) is not replacement_target
                or not self.can_replace_empty_model(name)
            ):
                self.state.issue(
                    "warning",
                    "Model was not replaced",
                    "The target model changed while loading. "
                    "Load this file under a new name instead.",
                )
                return

            if key == "model":
                # Passing an object avoids the synchronous file-loading path.
                self.add_model(name, candidate)
            else:
                self._assignments[name] = candidate
                self._current_assignments = name

            if on_loaded is not None:
                on_loaded()

        return tasks.start(
            "loading",
            f"Loading {key}: {Path(path).name}",
            run,
            on_result=publish,
        )

    def propose_session_remapping(self, session_id: int, *, background_template=None):

        session = self.sessions[session_id]
        if any(self.alignment_is_stale(i) for i in range(session_id)):
            raise ValueError("Update earlier stale alignments first.")

        references = self.alignment_references_for_session(session_id)

        return session.propose_remapping(
            references,
            background_template=(background_template),
            use_optical_flow=True,
            correct_rotation=self.correct_rotation,
        )

    # ------------------------------------------------------------------
    # Overview background from another source (display only)
    # ------------------------------------------------------------------

    def queue_load_overview_background(self, source_path, selection):
        """Load an image (file + field) as overview background, oriented
        like the sessions' backgrounds and aligned to their reference frame.

        The result is an unregistered SessionData holding only the
        background (template, remap, aligned background); it takes no part
        in tracking. Its alignment can be edited like a session's.
        """
        reference = next(
            (
                session
                for session in self.sessions
                if session.status["spatial_loaded"]
                and session.background_template is not None
            ),
            None,
        )

        if reference is None:
            self.state.issue(
                "info",
                "Overview background",
                "Load a session's spatial data first: the background is "
                "aligned to the sessions' reference frame.",
            )
            return None

        source_path = os.path.abspath(os.path.expanduser(os.fspath(source_path)))
        orientation = getattr(self, "background_orientation", "auto")
        spec = FieldSpec(
            path=selection.path,
            source=selection.source,
            attribute=selection.attribute,
            required=True,
        )

        def run():
            values = read_fields(
                source_path,
                {"spatial": {"background": spec}},
                ctx=current_task_context(),
            )
            image = values["spatial"]["background"]

            background = SessionData(
                name=f"Overview background ({Path(source_path).name})",
                path=source_path,
            )
            background.dims = tuple(reference.dims)
            background.params = deepcopy(reference.params)
            background.background_template = reference.prepare_background_template(
                image, background_orientation=orientation
            )

            # Any time point: align to the most similar sessions, not the
            # most recent ones.
            references = self.alignment_references_by_similarity(
                background.background_template
            )
            # No session order: deviations are only reported (warning).
            remap = background.propose_remapping(
                references, require_confirmation=False
            )
            self._set_overview_background_geometry(
                background, background.background_template, remap
            )
            return background

        def publish(background):
            self.overview_background = background
            self.notify_change(C.OVERVIEW_BACKGROUND)

            if not background.status["aligned"]:
                self.state.issue(
                    "warning",
                    "Overview background not aligned",
                    "Automatic alignment failed. Adjust it in Session alignment "
                    "(target: overview background).",
                )

        return self.state.tasks.start(
            "loading",
            f"Load overview background: {Path(source_path).name}",
            run,
            on_result=publish,
        )

    def _apply_stored_overview_background(self, background):
        """Use the overview background saved with loaded assignments."""
        if background is None:
            return

        if isinstance(background, Exception):
            self.state.issue(
                "warning",
                "Stored overview background not loaded",
                f"The assignments file contains an overview background that "
                f"cannot be read: {background}",
            )
            return

        if self.sessions and tuple(background.dims) != tuple(self.sessions[0].dims):
            self.state.issue(
                "warning",
                "Stored overview background not loaded",
                f"Its size {tuple(background.dims)} differs from the sessions' "
                f"{tuple(self.sessions[0].dims)}.",
            )
            return

        self.overview_background = background
        self.notify_change(C.OVERVIEW_BACKGROUND)

    @staticmethod
    def _set_overview_background_geometry(background, template, remap):
        background.background_template = np.asarray(template, dtype=np.float32)
        background.remap = remap
        background.background = remap.apply_remap(
            background.background_template, use_optical_flow=True
        )
        background.status["spatial_loaded"] = True
        background.status["aligned"] = bool(remap.report.success)

    def set_overview_background_alignment(self, template, remap):
        """Apply an edited alignment to the overview background."""
        background = getattr(self, "overview_background", None)

        if background is None:
            return

        self._set_overview_background_geometry(background, template, remap)
        # Accepted explicitly by the user.
        background.status["aligned"] = True
        self.notify_change(C.OVERVIEW_BACKGROUND)

    def set_border_tolerance(self, value):
        """Number of sessions a pixel may lie outside of and still count as
        within the shared imaging border (overlay and border statistic)."""
        value = max(0, int(value))
        if value == self.border_tolerance:
            return

        self.border_tolerance = value
        self.state.settings.setValue("alignment/border_tolerance", value)
        self.notify_change(C.BORDER_SETTINGS)

    def clear_overview_background(self):
        if getattr(self, "overview_background", None) is not None:
            self.overview_background = None
            self.notify_change(C.OVERVIEW_BACKGROUND)

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

        data = read_fields(
            source_path,
            {"spatial": {"background": spec}},
            ctx=current_task_context(),
        )

        background = data["spatial"]["background"]

        candidate_template = session.prepare_background_template(
            background,
            background_orientation=getattr(self, "background_orientation", "auto"),
        )
        candidate_remap = self.propose_session_remapping(
            session_id, background_template=(candidate_template)
        )

        return (candidate_template, candidate_remap)

    def _report_flow_fallback(self, session):
        remap = session.remap
        if remap is None:
            return

        info = remap.flow_info or {}
        if info.get("status") != "skipped":
            return

        if info.get("mode") == "flow_only":
            title = f"Alignment failed: {session.name}"
            recovery = (
                "Inspect the background and its orientation in Session "
                "Alignment. Correct them if needed, then rerun alignment "
                "or apply a manual alignment."
            )
        else:
            title = f"Flow correction skipped: {session.name}"
            recovery = (
                "The rigid alignment remains usable. Inspect it in "
                "Session Alignment; adjust the geometry or background "
                "if necessary, then rerun alignment."
            )

        self.state.issue(
            "warning",
            title,
            remap.flow_message + "\n" + recovery,
        )

    @after_display_tasks
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
        if tasks.processing_busy():
            self.state.issue(
                "warning",
                "Realignment not started",
                "Finish or cancel existing tasks first.",
            )
            return

        def publish(result):
            if not self._publish_realignment_update(result, expected_version):
                return

            draft = self.state.alignment_draft
            if (
                draft is not None
                and draft.session_id == session_id
                and draft.expected_version == expected_version
            ):
                self.state.alignment_draft = None
                self.state.alignment_draft_changed.emit()

        tasks.start(
            "loading",
            f"Commit alignment for {self.sessions[session_id].name}",
            self._build_realignment_update,
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

        for sid, records in result["comparison_updates"].items():
            self.sessions[sid].remap.remap_data = records

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

        changed_paths = {
            str(self.sessions[sid].path)
            for sid in result["updates"]
            if self.sessions[sid].path is not None
        }

        self.notify_change(
            Change(
                C.FOOTPRINT_GEOMETRY,
                session_paths=changed_paths or None,
            ),
            Change(
                C.BACKGROUND_IMAGE,
                session_paths=changed_paths or None,
            ),
            Change(C.PROCESSING_STATUS),
        )
        self._report_flow_fallback(self.sessions[result["session_id"]])
        self.state.alignment_review_finished.emit(self.sessions[result["session_id"]])
        return True

    def _build_realignment_update(self, session_id, **kwargs):
        return build_realignment_update(
            self,
            session_id,
            ctx=current_task_context(),
            **kwargs,
        )

    def register_session(
        self,
        from_file: str | Path | None = None,
        *,
        prepared_sessions=None,
        **kwargs,
    ) -> list[int]:
        if prepared_sessions is not None:
            sessions = list(prepared_sessions)
            restored_from_catan = False
        elif Path(from_file).suffix.lower() == ".json":
            sessions = self._read_loading_recipe(from_file)
            restored_from_catan = False
        else:
            source_path = os.path.abspath(os.path.expanduser(os.fspath(from_file)))

            result = read_operation(
                "registration",
                source_path,
                ctx=current_task_context(),
                timeout=300.0,
                retry_hint=(
                    "Registration could not complete. Restore access to "
                    "the source, then select this file again. If this was "
                    "part of a batch, check the session list before "
                    "resubmitting files already registered."
                ),
            )

            restored_from_catan = result["restored_from_catan"]

            if restored_from_catan:
                sessions = result["sessions"]

                # Remember the CATAN session file (see catan_sessions_file).
                for index, session in enumerate(sessions):
                    session._catan_file = source_path
                    session._catan_file_index = index
                    session._catan_file_revision = None
            else:
                sessions = [SessionData(path=source_path)]

        registered_ids = []
        added_ids = []

        try:
            for session in sessions:

                ctx = current_task_context()
                if ctx is not None:
                    ctx.check_cancelled()

                # Reopening a recipe can resume an existing session.
                # Preserve its current configuration and already-loaded data.
                if hasattr(session, "_recipe_options"):
                    key = self.session_source_key(session.path)
                    existing_id = next(
                        (
                            index
                            for index, existing in enumerate(self.sessions)
                            if self.session_source_key(existing.path) == key
                        ),
                        None,
                    )
                    if existing_id is not None:
                        registered_ids.append(existing_id)
                        continue

                # Runtime marker for GUI validation; not persisted.
                session._restored_from_catan = restored_from_catan
                session_id = super().register_session(
                    from_data=session,
                    **kwargs,
                )
                registered_ids.append(session_id)
                added_ids.append(session_id)

                saved_state = session.__dict__.pop("_restored_processing", None)
                if saved_state is not None:
                    processing = self._processing_state(session_id)
                    processing.geometry_revision = int(saved_state["geometry_revision"])
                    processing.alignment_stale = bool(saved_state["alignment_stale"])

                # Preserve restored field mappings and source overrides.
                if session.source_config is None:
                    session.source_config = (
                        self.state.config_manager.suggest_config_for(
                            path=session.path,
                            source_type="session",
                            ctx=current_task_context(),
                        )
                    )

                if not session.name:
                    session.name = Path(session.path).parent.name

                self.state.session_color = (
                    session_id,
                    self.session_colors.next(),
                )

        finally:
            # Publish once, including sessions successfully added before
            # an error interrupted registration.
            if added_ids:
                # Padding replaces the underlying array. Publish its latest
                # reference before notifying statistics and displays.
                self.state.assignments = (
                    None if self.assignments is None else self.assignments.ids
                )

                paths = {
                    str(self.sessions[index].path)
                    for index in added_ids
                    if self.sessions[index].path is not None
                }

                self.notify_change(
                    C.SESSION_ADDED,
                    session_paths=paths or None,
                )

                if self.state.current_session_id is None:
                    self.state.current_session_id = added_ids[0]

            # Restored sessions match their file at this geometry revision.
            for index in added_ids:
                session = self.sessions[index]
                if (
                    getattr(session, "_catan_file", None) is not None
                    and getattr(session, "_catan_file_revision", 0) is None
                ):
                    session._catan_file_revision = self.geometry_revision(index)

        return registered_ids

    @after_display_tasks
    def queue_save_sessions(self, path, *, mat_version="7.3"):
        tasks = self.state.tasks

        if tasks.processing_busy():
            self.state.issue(
                "info",
                "Session save is waiting",
                "Finish or cancel the current processing tasks, "
                "then click Save again.",
            )
            return

        if not self.sessions:
            self.state.issue(
                "info",
                "No sessions to save",
                "Register a session before saving.",
            )
            return

        saved_sessions = tuple(self.sessions)
        revisions = tuple(self.geometry_revision(i) for i in range(len(saved_sessions)))
        target = os.path.abspath(os.path.expanduser(os.fspath(path)))

        def saved():
            # These sessions can be restored from this file (as saved).
            for index, (session, revision) in enumerate(zip(saved_sessions, revisions)):
                session._catan_file = target
                session._catan_file_index = index
                session._catan_file_revision = revision

        try:
            snapshots = []

            for session_id, session in enumerate(self.sessions):
                processing = self._processing_state(session_id)

                snapshots.append(
                    prepare_session_snapshot(
                        session,
                        processing={
                            "geometry_revision": processing.geometry_revision,
                            "alignment_stale": processing.alignment_stale,
                        },
                        copy_arrays=True,
                    )
                )
        except Exception as exc:
            self.state.issue(
                "warning",
                "Could not prepare session save",
                str(exc),
            )
            return

        return tasks.start(
            "saving",
            f"Saving sessions: {Path(path).name}",
            save_session_snapshots_task,
            self.state,
            str(path),
            snapshots,
            mat_version=mat_version,
            finished=saved,
        )

    def assignments_fingerprint(self, assignments=None):
        """Hash of the saved assignment content (IDs, review status, union
        and session inclusion, manipulations)."""
        assignments = self.assignments if assignments is None else assignments
        digest = hashlib.blake2b(digest_size=16)

        arrays = [assignments.ids, assignments.review_status]

        union = assignments.union
        if union is not None and union.included is not None:
            arrays.append(union.included)

        arrays.extend(
            session.included for session in self.sessions if session.included is not None
        )

        for values in arrays:
            values = np.ascontiguousarray(values)
            digest.update(str(values.shape).encode())
            digest.update(values.tobytes())

        digest.update(str(len(assignments.manipulations)).encode())
        return digest.hexdigest()

    def catan_sessions_file(self):
        """CATAN session file holding exactly the current sessions (same
        order, unchanged geometry since restoring or saving), or None."""
        files = {getattr(session, "_catan_file", None) for session in self.sessions}

        if len(files) != 1 or None in files:
            return None

        for index, session in enumerate(self.sessions):
            if getattr(session, "_catan_file_index", None) != index or getattr(
                session, "_catan_file_revision", None
            ) != self.geometry_revision(index):
                return None

        return files.pop()

    def assignments_provenance(self):
        """Sessions (CATAN session file and/or loading recipe) and model file
        used with the current assignments, stored when saving them."""
        provenance = {}

        sessions_file = self.catan_sessions_file()
        if sessions_file:
            provenance["sessions_file"] = sessions_file

        try:
            provenance["loading_recipe"] = json.dumps(
                self.loading_recipe_snapshot(), ensure_ascii=False
            )
        except ValueError:
            # Sessions without a source path or configuration.
            pass

        model_path = getattr(self.model, "path", None)
        if model_path:
            provenance["model_path"] = str(model_path)

        return provenance

    def _overview_background_snapshot(self):
        """The loaded overview background, stored with saved assignments."""
        from catan.core.structures.session_snapshot import (
            prepare_background_snapshot,
        )

        background = getattr(self, "overview_background", None)
        return None if background is None else prepare_background_snapshot(background)

    def mark_assignments_saved(self, assignments=None, fingerprint=None):
        assignments = self.assignments if assignments is None else assignments

        if assignments is not None:
            assignments._saved_fingerprint = (
                self.assignments_fingerprint(assignments)
                if fingerprint is None
                else fingerprint
            )

    def unsaved_assignments(self):
        """Names of assignment sets with matches changed since loading or
        saving (never saved: any match)."""
        names = []

        for name, assignments in self._assignments.items():
            if assignments is None or not np.any(assignments.ids >= 0):
                continue

            saved = getattr(assignments, "_saved_fingerprint", None)

            if saved != self.assignments_fingerprint(assignments):
                names.append(name)

        return names

    @after_display_tasks
    def queue_save_result(self, key, path):
        if key not in {"model", "assignments"}:
            raise ValueError(f"Unsupported result type: {key}")

        tasks = self.state.tasks

        if tasks.processing_busy():
            self.state.issue(
                "info",
                "Cannot save while processing",
                "Finish or cancel the current processing tasks, "
                "then click Save again.",
            )
            return

        source = getattr(self, key)

        if source is None:
            self.state.issue(
                "info",
                "Nothing to save",
                (
                    f"No {key} are available."
                    if key == "assignments"
                    else "No model is available."
                ),
            )
            return

        try:
            prepared = (
                source.prepare_save(
                    sessions=tuple(self.sessions),
                    provenance=self.assignments_provenance(),
                    overview_background=self._overview_background_snapshot(),
                )
                if key == "assignments"
                else source.prepare_save()
            )
        except Exception as exc:
            self.state.issue(
                "warning",
                f"Could not prepare {key} save",
                str(exc),
            )
            return

        # The content written by this save; recorded once it succeeded.
        if key == "assignments":
            fingerprint = self.assignments_fingerprint(source)

            def saved():
                self.mark_assignments_saved(source, fingerprint)

        else:

            def saved():
                # Referenced by assignments saved later.
                source.path = os.path.abspath(os.path.expanduser(str(path)))

        return tasks.start(
            "saving",
            f"Saving {key}: {Path(path).name}",
            save_prepared_file_task,
            self.state,
            str(path),
            prepared,
            finished=saved,
        )

    @after_display_tasks
    def queue_export_redetection(
        self,
        jobs,
        *,
        include_missing=False,
        seed=0,
    ):
        from catan.core.redetection import build_seeds
        from catan.gui.background_tasks.runtime import TaskCancelled

        tasks = self.state.tasks

        if (
            tasks.processing_busy()
            or tasks.processing_requested
            or tasks.has_deferred_processing()
            or self.state.current_request is not None
        ):
            self.state.issue(
                "info",
                "Export not started",
                "Finish or cancel processing before exporting seeds.",
            )
            return

        version = self.state.data_version
        assignments_name = self.current_assignments
        source = self.assignments

        if source is None:
            self.state.issue(
                "warning",
                "Export not started",
                "No active assignments dataset is available.",
            )
            return

        if include_missing and self._stale_assignments.get(assignments_name):
            self.state.issue(
                "warning",
                "Export not started",
                "Update the stale assignments before exporting " "inferred footprints.",
            )
            return

        sessions = []

        for original in self.sessions:
            item = copy(original)
            item.status = dict(original.status)
            item.footprints = original.footprints.copy()

            count = item.footprints.shape[1]

            item.included = (
                np.ones(count, dtype=bool)
                if original.included is None
                else np.asarray(original.included, dtype=bool).copy()
            )

            synthetic = getattr(original, "synthetic", None)
            item.synthetic = (
                np.zeros(count, dtype=bool)
                if synthetic is None
                else np.asarray(synthetic, dtype=bool).copy()
            )

            if item.included.shape != (count,) or item.synthetic.shape != (count,):
                self.state.issue(
                    "warning",
                    "Export not started",
                    f"Invalid component flags in {item.name}.",
                )
                return

            item.remap = deepcopy(original.remap)
            item.source_config = deepcopy(original.source_config)

            # Hold references to existing trace arrays, without
            # duplicating every session's traces. Export never
            # modifies these arrays.
            item._traces = dict(original.traces)

            sessions.append(item)

        ids = source.ids.copy()

        union = copy(source.union)
        union.footprints = source.union.footprints.copy()
        union.included = np.asarray(source.union.included, dtype=bool).copy()

        jobs = [(int(session_id), str(path)) for session_id, path in jobs]

        destination_keys = [os.path.normcase(os.path.abspath(path)) for _, path in jobs]
        original_paths = {
            os.path.normcase(os.path.abspath(str(session.path)))
            for session in sessions
            if session.path
        }

        if len(set(destination_keys)) != len(destination_keys) or any(
            key in original_paths for key in destination_keys
        ):
            self.state.issue(
                "warning",
                "Export not started",
                "Export paths must be distinct and must not "
                "overwrite session source files.",
            )
            return

        def run():
            ctx = current_task_context()
            issues = []

            for session_id, path in jobs:
                ctx.check_cancelled()

                if self.state.data_version != version:
                    raise ValueError(
                        "Data changed during export. Completed files "
                        "remain available; retry the remaining sessions."
                    )

                session = sessions[session_id]
                values = prepared = A = None

                try:
                    ctx.message(f"Preparing redetection seeds: {session.name}")

                    values = build_seeds(
                        sessions,
                        ids,
                        union,
                        session_id,
                        include_missing=include_missing,
                        seed=seed,
                        ctx=ctx,
                        read_fields=read_fields,
                    )

                    A = values.pop("A")

                    # Explicit CSC storage keeps the format easy
                    # to consume without CATAN.
                    values.update(
                        A_data=A.data,
                        A_indices=A.indices,
                        A_indptr=A.indptr,
                        A_shape=np.asarray(A.shape, dtype=np.int64),
                    )

                    fields = {
                        key: FieldSpec(
                            path=(
                                "/A/" + key[2:] if key.startswith("A_") else "/" + key
                            ),
                            required=True,
                        )
                        for key in values
                    }

                    prepared = {
                        "data": {"seeds": values},
                        "fields": {"seeds": fields},
                        "attributes": {
                            "object_type": "RedetectionSeeds",
                            "format_version": 1,
                            "assignments": str(assignments_name),
                            "session_id": session_id,
                            "source_path": str(session.path),
                            "coordinate_frame": "original_session",
                            "pixel_order": "F",
                            "sparse_format": "csc",
                            "trace_axes": "seed,frame",
                            "rng_seed": seed,
                            "noise_method": (
                                "median baseline; MAD of first "
                                "differences / "
                                "(0.67448975 * sqrt(2))"
                            ),
                        },
                    }

                    if self.state.data_version != version:
                        raise ValueError("Data changed; retry export.")

                    ctx.check_cancelled()

                    save_prepared_file_task(
                        self.state,
                        path,
                        prepared,
                    )

                    omitted = len(values["omitted_ids"])

                    if omitted:
                        issues.append(
                            f"{session.name}: omitted {omitted} seeds "
                            "because no detected location was available "
                            "or the seed fell outside the imaging window. "
                            "Their IDs are stored in omitted_ids."
                        )

                    ctx.message(f"Exported {session.name}: " f"{A.shape[1]} seeds")

                except TaskCancelled:
                    raise

                except Exception as exc:
                    issues.append(f"{session.name}: {exc}")

                finally:
                    # Release this session's output before loading
                    # the next session's C.
                    values = prepared = A = None

            return issues

        def finished(issues):
            if issues:
                self.state.issue(
                    "warning",
                    "Redetection export needs attention",
                    "\n\n".join(issues),
                )

        return tasks.start(
            "saving",
            "Export redetection seeds",
            run,
            on_result=finished,
        )

    def queue_save_included(self, session, path):
        tasks = self.state.tasks

        if tasks.processing_busy() or tasks.processing_requested:
            self.state.issue(
                "info",
                "Cannot save inclusion while processing",
                "Finish or cancel processing, then try again.",
            )
            return None

        if not any(item is session for item in self.sessions):
            return None

        if not has_exclusions(session):
            return None

        mask = inclusion_mask(
            session.included,
            session.n_neurons,
            session.name,
        )

        prepared = {
            "data": {
                "spatial": {
                    "included": mask,
                }
            },
            "fields": {
                "spatial": {
                    "included": FieldSpec(
                        path="/included",
                        required=True,
                    )
                }
            },
            "attributes": {
                "object_type": "SessionIncludedData",
                "format_version": 1,
                "source_path": str(session.path or ""),
                "n_footprints": int(session.n_neurons),
            },
        }

        return tasks.start(
            "saving",
            f"Saving inclusion: {session.name}",
            save_prepared_file_task,
            self.state,
            str(path),
            prepared,
        )

    def remove_session(self, session_id: int):

        removed_session = self.sessions[session_id]
        previous_current = self.current_session

        removed_path = self.sessions[session_id].path
        super().move_session(session_id, -1)
        self.state.assignments = (
            None if self.assignments is None else self.assignments.ids
        )

        if self.state.assignments is None:
            self.state.update_selected_components(None)
        else:
            self.adjust_selected_components_after_data_change(session_id, -1)

        if previous_current is not None:
            current_id = next(
                (
                    index
                    for index, item in enumerate(self.sessions)
                    if item is previous_current
                ),
                None,
            )
            if current_id is None and self.sessions:
                current_id = min(session_id, len(self.sessions) - 1)

            self.state.current_session_id = current_id

        self.notify_change(
            C.SESSION_REMOVED,
            session_paths=(None if removed_path is None else {str(removed_path)}),
        )

        draft = self.state.alignment_draft
        if draft is not None and draft.session is removed_session:
            self.state.alignment_draft = None
            self.state.alignment_draft_changed.emit()

        self.state.alignment_review_finished.emit(removed_session)

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

        self.notify_change(C.SESSION_ORDER)  # Notify that sessions have changed

    def add_model(self, name: str, model: Optional[str | Model] = None):
        super().add_model(name, model)
        self.notify_change(
            C.MODEL_COUNTS, C.MODEL_PARAMETERS, C.PROCESSING_STATUS
        )  # Notify that model has changed

    def change_model(self, name: str):
        super().change_model(name)
        self.notify_change(
            C.MODEL_COUNTS, C.MODEL_PARAMETERS, C.PROCESSING_STATUS
        )  # Notify that model has changed

    @after_display_tasks
    def remove_loaded_source(
        self,
        key,
        name,
        *,
        expected_source,
        on_removed=None,
    ):
        if key not in {"model", "assignments"}:
            raise ValueError(f"Unknown source type: {key}")

        tasks = self.state.tasks

        if (
            tasks.processing_busy()
            or tasks.processing_requested
            or tasks.has_deferred_processing()
            or self.state.current_request is not None
        ):
            self.state.issue(
                "info",
                "Dataset not removed",
                "Finish or cancel the current operations before removing it.",
            )
            return False

        sources = self._model if key == "model" else self._assignments

        if sources.get(name) is not expected_source:
            self.state.issue(
                "warning",
                "Dataset not removed",
                "The selected dataset changed. Select it again and retry.",
            )
            return False

        if key == "model":
            if name == "local":
                # Keep the permanent Working slot, but empty its contents.
                sources[name] = Model(params=self.params)
            else:
                if self.current_model_name == name:
                    self.change_model("local")
                del sources[name]

            self.notify_change(
                C.MODEL_COUNTS,
                C.MODEL_PARAMETERS,
                C.PROCESSING_STATUS,
            )

        else:
            if name == "local":
                replacement = Assignments()
                replacement.source_config = deepcopy(expected_source.source_config)
                self._ensure_assignment_session_count(replacement)

                sources[name] = replacement
                self._stale_assignments[name] = set()

                if self.current_assignments == name:
                    self.state.apply_assignment_rebuild(
                        replacement.ids,
                        {},
                        from_session_id=0,
                        component_id_map={},
                    )
                    self.state.assignments = replacement.ids

            else:
                # Switching must happen BEFORE deleting the source:
                # correspondence mapping still needs the old assignments.
                if self.current_assignments == name:
                    if not self.change_assignments("local"):
                        return False

                del sources[name]
                self._stale_assignments.pop(name, None)

            self._comparison_cache = {}
            self.statistic_engine.clear_cache()

            self.notify_change(
                C.ASSIGNMENT_SET,
                *ASSIGNMENT_CONTENT_CHANGES,
            )

        if on_removed is not None:
            on_removed()

        return True

    def can_replace_empty_model(self, name=None):
        if name is None:
            name = self.current_model_name

        model = self._model.get(name)

        return (
            model is not None
            and not model.loaded
            and not model.fitted
            and not model.parameters
            and not any(model.counts.values())
        )

    def register_assignments(self, path: str, name: str):

        assignments = Assignments()
        assignments.path = path
        assignments.source_config = self.state.config_manager.suggest_config_for(
            path=path,
            source_type="assignments",
            ctx=current_task_context(),
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

        restored_masks = inclusion_to_restore(
            self.assignments.session_included,
            self.sessions,
        )

        if restored_masks is not None:
            for session, mask in zip(self.sessions, restored_masks):
                session.included = mask

        self.rebuild_union()

        self.state.assignments = self.assignments.ids
        self.mark_assignments_saved()

        self.notify_change(
            C.ASSIGNMENT_SET,
            C.FOOTPRINT_GEOMETRY,
            *ASSIGNMENT_CONTENT_CHANGES,
        )

        if self.assignments.path:
            try:
                background = read_overview_background(self.assignments.path)
            except Exception as exc:
                background = exc
            self._apply_stored_overview_background(background)

    def queue_load_assignments(self, *, on_config_required=None):
        source = self.assignments
        source_name = self.current_assignments

        if source is None:
            return

        # Continue a just-imported assignments file immediately when idle.
        # Its data-loading task must precede the next deferred file import.
        if self._source_loading_ready():
            return self._queue_load_assignments_now(
                on_config_required=on_config_required,
            )

        def dispatch():
            if (
                self.current_assignments != source_name
                or self.assignments is not source
            ):
                self.state.issue(
                    "warning",
                    "Queued assignments load was not applied",
                    "The selected assignments changed while this request "
                    "was waiting. Select the intended dataset and click "
                    "its open-folder button again.",
                )
                return

            self._queue_load_assignments_now(
                on_config_required=on_config_required,
            )

        return self.state.tasks.queue_after_processing(
            f"Load assignments: {source_name}",
            dispatch,
            ready=lambda: (
                not self._model_fit_requested and self.state.current_request is None
            ),
        )

    def _queue_load_assignments_now(self, *, on_config_required=None):
        tasks = self.state.tasks

        if not self._source_loading_ready():
            return self.queue_load_assignments(
                on_config_required=on_config_required,
            )

        source = self.assignments
        if source is None:
            return

        if source.source_config is None:
            if on_config_required is not None:
                on_config_required()
            self.state.issue(
                "warning",
                "Assignment loading not started",
                "Select a load configuration first.",
            )
            return

        assignment_name = self.current_assignments
        data_version = self.state.data_version
        sessions = tuple(self.sessions)
        source_path = source.path
        source_fields = deepcopy(source.source_config.get_fields_to_load())

        candidate = Assignments()
        candidate.path = source_path
        candidate.source_config = deepcopy(source.source_config)

        # Borrow the tracking methods, but isolate everything this operation
        # mutates. Large footprint/trace arrays are initially shared read-only.
        worker = copy(self)
        worker._assignments = {assignment_name: candidate}
        worker._current_assignments = assignment_name
        worker.sessions = []

        for session in sessions:
            staged = copy(session)

            # Restoration changes inclusion in place.
            staged.included = session.included.copy()

            # Appending synthetic components replaces arrays, but writes
            # their new references into these dictionaries.
            staged._traces = dict(session._traces)
            staged.quality = dict(session.quality)

            worker.sessions.append(staged)

        def run():
            ctx = current_task_context()
            if ctx is not None:
                ctx.check_cancelled()

            result = read_operation(
                "assignments_fields",
                source_path,
                fields_to_load=source_fields,
                ctx=ctx,
                timeout=300.0,
            )

            if result["problems"]:
                return {"config_problems": result["problems"]}

            candidate.register_data(**result["data"])

            # Check column count before padding or restoring manipulations.
            worker.check_assignments_compatibility(
                candidate,
                check_footprints=False,
                raise_on_error=True,
            )

            worker._ensure_assignment_session_count(candidate)

            if ctx is not None:
                ctx.check_cancelled()

            worker.restore_manipulations()
            restored_masks = inclusion_to_restore(
                candidate.session_included,
                worker.sessions,
            )

            if restored_masks is not None:
                for session, mask in zip(
                    worker.sessions,
                    restored_masks,
                ):
                    session.included = mask

            worker.check_assignments_compatibility(
                candidate,
                raise_on_error=True,
            )

            worker.rebuild_union(ctx=ctx)

            if ctx is not None:
                ctx.check_cancelled()

            try:
                candidate._overview_background = read_overview_background(
                    source_path, ctx=ctx
                )
            except TaskCancelled:
                raise
            except Exception as exc:
                candidate._overview_background = exc

            return candidate

        def publish(result):
            if result is None:
                return

            sessions_unchanged = len(self.sessions) == len(sessions) and all(
                current is original
                for current, original in zip(self.sessions, sessions)
            )
            config_unchanged = (
                source.source_config is not None
                and source.path == source_path
                and source.source_config.get_fields_to_load() == source_fields
            )

            if (
                self.state.data_version != data_version
                or self.assignments is not source
                or self.current_assignments != assignment_name
                or not sessions_unchanged
                or not config_unchanged
            ):
                self.state.issue(
                    "warning",
                    "Assignment loading discarded",
                    "Data or the load configuration changed during loading. "
                    "Please load the assignments again.",
                )
                return

            if isinstance(result, dict) and "config_problems" in result:
                source.status["loading_possible"] = False
                source._load_config_problems = list(result["config_problems"])

                self.state.issue(
                    "info",
                    "Repair the assignments load configuration",
                    "\n\n".join(result["config_problems"])
                    + "\n\nSelect a matching preset or correct the IDs field, "
                    "then click the assignments open-folder button to load again.",
                )

                if on_config_required is not None and self.assignments is source:
                    on_config_required()

                return

            use_as_working = bool(getattr(source, "_use_as_working", False))

            if use_as_working and not self.can_use_working_assignments():
                self.state.issue(
                    "warning",
                    "Working assignments were not replaced",
                    "The working assignments are no longer empty. "
                    "Load the file again as a separate dataset.",
                )
                return

            # Publish restored manipulation effects, preserving session identity.
            for original, staged in zip(sessions, worker.sessions):
                original.included = staged.included

                if staged.n_neurons != original.n_neurons:
                    for field in (
                        "footprints",
                        "centroids",
                        "n_neurons",
                        "synthetic",
                        "_traces",
                        "quality",
                    ):
                        setattr(original, field, getattr(staged, field))

            # Preserve references held by configuration widgets.
            result.source_config = source.source_config
            source.__dict__.update(result.__dict__)
            source.__dict__.pop("_use_as_working", None)

            if use_as_working:
                self._assignments["local"] = source
                del self._assignments[assignment_name]

                self._stale_assignments["local"] = self._stale_assignments.pop(
                    assignment_name, set()
                )
                self._current_assignments = "local"

            self.state.assignments = source.ids
            self.mark_assignments_saved(source)
            self.notify_change(
                C.ASSIGNMENT_SET,
                C.FOOTPRINT_GEOMETRY,
                *ASSIGNMENT_CONTENT_CHANGES,
            )

            self._apply_stored_overview_background(
                source.__dict__.pop("_overview_background", None)
            )

        return tasks.start(
            "loading",
            "Load assignments and rebuild union",
            run,
            on_result=publish,
        )

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
            path=self.assignments.path,
            source_type="assignments",
            ctx=current_task_context(),
        )

        if self.assignments is None:
            raise ValueError(
                "No assignments file was added. Please provide valid assignments."
            )
        self.state.assignments = self.assignments.ids
        self.notify_change(
            *ASSIGNMENT_CONTENT_CHANGES
        )  # Notify that assignments have changed

    def change_assignments(self, name: str):
        if name == self.current_assignments:
            return True

        tasks = self.state.tasks

        if (
            self.state.current_request is not None
            or tasks.processing_busy()
            or tasks.processing_requested
            or any(task is not None for task in tasks.current.values())
        ):
            self.state.issue(
                "warning",
                "Assignments not switched",
                "Finish or cancel the current operation before "
                "switching the active assignments.",
            )
            return False

        try:
            comparison = self.assignment_comparison(
                self.current_assignments,
                name,
            )

            neuron_map = {}

            for neuron in range(len(comparison.source_rows)):
                partners = comparison.partners(neuron)
                if partners:
                    neuron_map[neuron] = partners[0].neuron_id

            components = set(self.state.selected_components or [])
            components.update(self.state.highlighted_components or [])
            components.update(self.state.highlight_roles)

            if self.state.focused_component is not None:
                components.add(self.state.focused_component)

            component_map = {
                (component.neuron_id, component.session_id): comparison.translate(
                    component.neuron_id,
                    component.session_id,
                )
                for component in components
            }

        except (KeyError, ValueError) as exc:
            self.state.issue(
                "error",
                "Could not switch assignments",
                str(exc),
            )
            return False

        super().change_assignments(name)

        self.state.apply_assignment_rebuild(
            self.assignments.ids,
            neuron_map,
            from_session_id=0,
            component_id_map=component_map,
        )

        self.notify_change(*ASSIGNMENT_CONTENT_CHANGES)
        return True

    def can_use_working_assignments(self):
        working = self._assignments.get("local")

        return (
            working is not None
            and working.ids.shape[0] == 0
            and not working.path
            and not working.manipulations
        )

    def assignment_comparison(self, source_name, target_name):
        source = self._assignments[source_name]
        target = self._assignments[target_name]

        version = self.state.data_version

        if getattr(self, "_comparison_cache_version", None) != version:
            self._comparison_cache = {}
            self._comparison_cache_version = version

        key = (
            source_name,
            target_name,
            id(source),
            id(target),
        )

        comparison = self._comparison_cache.get(key)

        if comparison is None:
            comparison = AssignmentComparison(
                source,
                target,
                self.sessions,
            )
            self._comparison_cache[key] = comparison

        return comparison

    def map_assignment_components(
        self,
        components,
        source_name,
        target_name,
        *,
        whole_neurons=False,
    ):
        if components is None:
            return []

        if isinstance(components, NeuronComponent):
            components = [components]

        if source_name not in self._assignments or target_name not in self._assignments:
            return []

        if source_name == target_name:
            return list(dict.fromkeys(components))

        comparison = self.assignment_comparison(source_name, target_name)
        result = []

        for component in components:
            if whole_neurons:
                # Preserve all partners, including split/merge counterparts.
                result.extend(
                    NeuronComponent(partner.neuron_id, None)
                    for partner in comparison.partners(component.neuron_id)
                )
            else:
                mapped = comparison.translate(
                    component.neuron_id,
                    component.session_id,
                )
                if mapped is not None:
                    result.append(NeuronComponent(*mapped))

        return list(dict.fromkeys(result))

    @after_display_tasks
    def queue_process_assignments(
        self,
        *,
        session_ids=None,
        from_session_id=None,
        mode="pending",
        _model_attempted=False,
    ):
        tasks = self.state.tasks

        if tasks.processing_busy():
            # Run once current processing (e.g. count registration and the
            # final fit) has finished; the plan is rebuilt at that point.
            # If a model update is in progress, it counts as the attempt,
            # so a cancelled/insufficient update does not restart itself.
            model_busy = tasks.current_task("model update") is not None or bool(
                tasks.queued_tasks("model update")
            )

            return tasks.queue_after_processing(
                "Process neuron registrations (after current processing)",
                lambda: self.queue_process_assignments(
                    session_ids=session_ids,
                    from_session_id=from_session_id,
                    mode=mode,
                    _model_attempted=_model_attempted or model_busy,
                ),
                ready=lambda: (
                    not self._model_fit_requested
                    and self.state.current_request is None
                ),
            )

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
            self.notify_change(*ASSIGNMENT_CONTENT_CHANGES)

        def run():
            return self.build_assignment_update(
                session_ids=plan["session_ids"],
                mode=mode,
                ctx=current_task_context(),
            )

        return tasks.start(
            "calculating",
            "Process pending neuron registrations",
            run,
            on_result=publish,
        )

    def queue_copy_assignments(self, name, *, on_copied=None):
        name = str(name).strip()
        source_name = self.current_assignments
        source = self.assignments
        tasks = self.state.tasks

        if not name or name in self._assignments:
            self.state.issue(
                "warning",
                "Assignments not copied",
                "Choose a non-empty name that is not already in use.",
            )
            return

        if source is None or source.ids.shape[0] == 0:
            self.state.issue(
                "info",
                "Assignments not copied",
                "Load or register assignments before creating a copy.",
            )
            return

        if self.state.current_request is not None or tasks.processing_busy():
            self.state.issue(
                "info",
                "Assignments not copied",
                "Finish the current manipulation or processing operation first.",
            )
            return

        version = self.state.data_version
        sessions = tuple(self.sessions)
        stale = set(self._stale_assignments.get(source_name, ()))

        def run():
            ctx = current_task_context()
            ctx.check_cancelled()

            candidate = source.copy()
            included = np.asarray(source.union.included, dtype=bool).copy()

            worker = copy(self)
            worker._assignments = {name: candidate}
            worker._current_assignments = name
            worker.sessions = list(sessions)

            worker.rebuild_union(ctx=ctx)

            if candidate.union.included.shape != included.shape:
                raise ValueError("The rebuilt copy has a different number of neurons.")

            candidate.union.included = included
            ctx.check_cancelled()
            return candidate

        def publish(candidate):
            if (
                self.state.data_version != version
                or self._assignments.get(source_name) is not source
                or name in self._assignments
            ):
                self.state.issue(
                    "warning",
                    "Assignments copy discarded",
                    "The data changed while copying. Create the copy again.",
                )
                return

            self._assignments[name] = candidate
            self._stale_assignments[name] = stale

            # Do not change the active assignments or selection.
            self.notify_change(C.ASSIGNMENT_SET)

            if on_copied is not None:
                on_copied()

        return tasks.start(
            "calculating",
            f"Copy assignments: {name}",
            run,
            on_result=publish,
        )

    @staticmethod
    def session_source_key(path):
        """Lexical comparison: avoid resolving or probing remote paths."""
        if path is None:
            return None
        return os.path.normcase(os.path.abspath(os.path.expanduser(str(path))))

    def loading_recipe_snapshot(self):
        """Create a plain-data snapshot on the GUI thread."""
        entries = []

        for session in self.sessions:
            if not session.path or session.source_config is None:
                raise ValueError(
                    f"Session {session.name!r} has no source path or load configuration."
                )

            options = dict(
                getattr(
                    session,
                    "_last_load_options",
                    getattr(session, "_recipe_options", {}),
                )
            )
            options.setdefault(
                "background_orientation",
                getattr(self, "background_orientation", "auto"),
            )
            options.setdefault("correct_rotation", bool(self.correct_rotation))
            options.setdefault(
                "background_mode",
                getattr(session, "_load_background_mode", "configured"),
            )

            # The displayed background may have been constructed from footprints.
            if session.background_origin == "footprints":
                options["background_mode"] = "footprints"

            entries.append(
                {
                    "path": os.path.abspath(os.path.expanduser(str(session.path))),
                    "name": session.name,
                    "load_config": deepcopy(session.source_config.to_dict()),
                    "load_options": options,
                }
            )

        if not entries:
            raise ValueError("There are no sessions to save.")

        return {
            "type": "catan-loading-recipe",
            "version": 1,
            "sessions": entries,
        }

    def save_loading_recipe(self, path, document):
        payload = (json.dumps(document, indent=2, ensure_ascii=False) + "\n").encode(
            "utf-8"
        )

        return save_bytes_task(self.state, path, payload)

    @staticmethod
    def _read_loading_recipe(path):
        """Parse all entries before registering any session."""
        path = Path(os.path.abspath(os.path.expanduser(str(path))))

        document = read_operation(
            "json",
            path,
            ctx=current_task_context(),
            timeout=60.0,
            retry_hint=(
                "Restore access to the recipe file, then reopen it. "
                "You can select the remaining sessions in the recipe chooser."
            ),
        )

        return Data.parse_loading_recipe(document, path.parent)

    @staticmethod
    def parse_loading_recipe(document, base_directory):
        """Recipe document -> unregistered SessionData (relative session
        paths resolve against ``base_directory``)."""
        path = Path(base_directory) / "recipe.json"

        if (
            not isinstance(document, dict)
            or document.get("type") != "catan-loading-recipe"
            or document.get("version") != 1
        ):
            raise ValueError("This JSON is not a supported CATAN loading recipe.")

        entries = document.get("sessions")
        if not isinstance(entries, list) or not entries:
            raise ValueError("The loading recipe contains no sessions.")

        sessions = []
        ctx = current_task_context()

        for index, entry in enumerate(entries, start=1):
            if ctx is not None:
                ctx.check_cancelled()

            try:
                source = entry["path"]
                if not isinstance(source, str) or not source.strip():
                    raise ValueError("Missing session source path.")

                source = Path(source).expanduser()
                if not source.is_absolute():
                    source = path.parent / source

                # Lexical normalization, without inspecting remote sources.
                source = os.path.abspath(str(source))

                config = LoadConfig.from_dict(entry["load_config"])
                if config.source_type != "session":
                    raise ValueError("Expected a session load configuration.")

                options = dict(entry.get("load_options", {}))
                options.setdefault("background_orientation", "auto")
                options.setdefault("background_mode", "configured")
                options.setdefault("correct_rotation", False)

                if options["background_orientation"] not in {
                    "auto",
                    "as_stored",
                    "transpose",
                }:
                    raise ValueError("Invalid background orientation.")

                if options["background_mode"] not in {"configured", "footprints"}:
                    raise ValueError("Invalid background mode.")

                if not isinstance(options["correct_rotation"], bool):
                    raise ValueError("correct_rotation must be true or false.")

                session = SessionData(
                    path=source,
                    name=entry.get("name"),
                )
                session.source_config = config
                session._recipe_options = options
                sessions.append(session)

            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                raise ValueError(
                    f"Invalid loading recipe, session entry {index}: {exc}"
                ) from exc

        return sessions

    def queue_assign_neurons(
        self, session_id: int, to_present=True, callback=None, *, batch=None
    ):
        if batch is not None and batch.cancelled:
            return
        session = self.sessions[session_id]

        if to_present:
            # The task context exists only inside the running worker.
            fn = lambda: self.assign_neurons(
                from_session_index=session_id,
                clean_traces=False,
                ctx=current_task_context(),
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
                            and not self.state.tasks.queued_tasks(
                                group, include_deferred=False
                            )
                            for group in ("loading", "model update")
                        )
                    )
                )
            ),
            finished=callback,
            batch=batch,
        )

    def assign_neurons(
        self,
        from_file: Optional[str | Path] = None,
        from_data: Optional[SessionData] = None,
        from_session_index: Optional[int] = None,
        align_to_reference: bool = False,
        clean_traces: bool = False,
        force_registration: bool = False,
        p_thr=None,
        ctx=None,
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
            ctx=ctx,
        )
        self.state.assignments = self.assignments.ids
        self.notify_change(
            *ASSIGNMENT_CONTENT_CHANGES
        )  # Notify that neurons have changed

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
            values[neuron_id, session_id, ...] = np.nan

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
            self.notify_change(*ASSIGNMENT_CONTENT_CHANGES)

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

        self.notify_change(
            C.INCLUSION,
            neuron_ids={neuron_id},
        )
        return NeuronComponent(neuron_id=neuron_id, session_id=None)

    def exclude_neurons(self, neuron_ids) -> list[int]:
        """Exclude several neurons with a single change notification."""
        if self.assignments is None:
            raise ValueError("No assignments are available.")

        excluded = []

        for neuron_id in sorted({int(n) for n in neuron_ids}):
            if not self.assignments.union.included[neuron_id]:
                continue

            self.assignments.union.included[neuron_id] = False

            for session_id, fp_id in enumerate(self.assignments.ids[neuron_id]):
                if fp_id >= 0:
                    self.sessions[session_id].included[int(fp_id)] = False

            excluded.append(neuron_id)

        if excluded:
            self.notify_change(C.INCLUSION, neuron_ids=set(excluded))

        return excluded

    def exclude_components(self, components) -> list[NeuronComponent]:
        """Exclude several footprints with a single change notification.

        Each footprint is retired to its own excluded singleton neuron,
        as in exclude_component().
        """
        retired = []

        for component in sorted(
            {c for c in components if c.session_id is not None},
            key=lambda c: (c.neuron_id, c.session_id),
        ):
            # Skip missing and already excluded footprints.
            if not self.is_included(component):
                continue

            retired.append(self.exclude_component(component, notify=False))

        if retired:
            self.notify_change(*ASSIGNMENT_CONTENT_CHANGES)

        return retired

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

        self.notify_change(
            C.INCLUSION,
            neuron_ids={neuron_id},
        )

    def reinclude_neuron(self, neuron_id: int) -> None:

        self.assignments.union.included[neuron_id] = True
        for session_id in range(self.assignments.ids.shape[1]):
            self.reinclude_component(
                NeuronComponent(neuron_id=neuron_id, session_id=session_id)
            )

    def remove_component(self, component: NeuronComponent | int) -> None:
        if isinstance(component, (int, np.integer)):
            self.remove_neuron(int(component))
            return

        targets = self.curation_targets(
            [component],
            remove=True,
            whole_neurons=False,
        )

        self.apply_curation_targets(
            targets,
            remove=True,
            whole_neurons=False,
        )

    def remove_neuron(self, neuron_id: int) -> None:
        targets = self.curation_targets(
            [
                NeuronComponent(
                    neuron_id=int(neuron_id),
                    session_id=None,
                )
            ],
            remove=True,
            whole_neurons=True,
        )

        self.apply_curation_targets(
            targets,
            remove=True,
            whole_neurons=True,
        )

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
            self.notify_change(
                C.FOOTPRINT_GEOMETRY,
                *ASSIGNMENT_CONTENT_CHANGES,
            )

        return target_neuron

    def partial_union_footprint(self, neuron_id, sessions):
        """Union footprint of a neuron built from ``sessions`` only (as
        rebuilding the union would after removing the other footprints)."""
        ids = self.assignments.ids
        row = ids[neuron_id].copy()
        keep = np.zeros(len(row), dtype=bool)
        keep[list(sessions)] = True

        try:
            ids[neuron_id, ~keep] = -1
            return self._build_union_footprint_for_neuron(neuron_id)
        finally:
            ids[neuron_id] = row

    def neuron_merge_distance(self, source_id, target_id, without=()):
        """Union centroid distance (px); ``without``: source sessions left
        out of the source union (partial merge, resolved conflicts)."""
        assignments = self.assignments
        if assignments is None or assignments.union is None:
            return np.inf

        without = [int(s) for s in without]

        if without and 0 <= source_id < len(assignments.ids):
            from catan.core.data import center_of_mass

            ids = assignments.ids
            union = assignments.union
            row = ids[source_id].copy()
            centroid = union.centroids[source_id].copy()

            try:
                ids[source_id, without] = -1

                if not np.any(ids[source_id] >= 0):
                    return np.inf

                column = self._build_union_footprint_for_neuron(source_id)
                union.centroids[source_id] = center_of_mass(
                    column, *union.dims, convert=union.params.get("pxtomu", 1.0)
                )[0]
                return self.neuron_merge_distance(source_id, target_id)
            finally:
                ids[source_id] = row
                union.centroids[source_id] = centroid

        union = assignments.union
        if (
            source_id == target_id
            or union.centroids is None
            or not 0 <= source_id < len(union.centroids)
            or not 0 <= target_id < len(union.centroids)
        ):
            return np.inf

        # Union centroids use the pixel-to-micron conversion.
        scale = float(union.params.get("pxtomu", 1.0))
        if not np.isfinite(scale) or scale <= 0:
            return np.inf

        distance = (
            np.linalg.norm(union.centroids[source_id] - union.centroids[target_id])
            / scale
        )
        return float(distance) if np.isfinite(distance) else np.inf

    def prepare_manipulation(
        self, plan, *, detach_occupied=False, exclude_occupied=False
    ):
        from catan.gui.data.curation_actions import prepare

        if self.state.tasks.processing_busy() or self.state.tasks.processing_requested:
            raise ValueError("Wait for current processing to finish.")

        return prepare(
            self,
            plan,
            detach_occupied=detach_occupied,
            exclude_occupied=exclude_occupied,
        )

    def start_manipulation_batch(self, kind):
        """Isolated batch of split/merge plans, committed at once."""
        from catan.gui.data.curation_actions import ManipulationBatch

        if self.state.tasks.processing_busy() or self.state.tasks.processing_requested:
            raise ValueError("Wait for current processing to finish.")

        return ManipulationBatch(self, kind)

    def validate_prepared_manipulation(self, prepared, spec):
        from catan.gui.data.curation_actions import (
            validate_prepared,
        )

        return validate_prepared(prepared, spec)

    def commit_manipulation(self, prepared):
        from catan.gui.data.curation_actions import commit

        return commit(self, prepared)

    def _apply_manipulation(self, kind, sources, targets):
        from catan.gui.data.curation_actions import (
            ManipulationPlan,
            ManipulationSpec,
        )

        plan = ManipulationPlan.capture(
            self,
            kind,
            sources,
            targets,
        )

        def perform():
            prepared = self.prepare_manipulation(plan)

            prepared.validation = self.validate_prepared_manipulation(
                prepared,
                ManipulationSpec(),
            )

            return self.commit_manipulation(prepared)

        if self.state.tasks.processing_busy():
            raise ValueError("Wait for current processing to finish.")

        if self.state.tasks.defer_for_background(perform):
            return None

        return perform()

    def process_component_request(self, request):
        if request.stage != "destination" or not request.is_complete:
            raise ValueError("Complete both request stages before applying.")

        return self._apply_manipulation(
            request.type,
            tuple(request.destination),
            tuple(request.origin),
        )

    def merge_neuron_request(self, request):
        if request.assignments is not self.assignments:
            raise ValueError("The active assignments changed. " "Start a new request.")

        request.refresh()
        if not request.is_complete:
            raise ValueError(request.error)

        target = (
            () if request.target_id is None else (NeuronComponent(request.target_id, None),)
        )

        if target and request.moves_whole_neuron:
            return self._apply_manipulation(
                "neuron_merge",
                (NeuronComponent(request.source_id, None),),
                target,
            )

        # A part of the neuron, or into a new neuron.
        return self._apply_manipulation(
            "move_footprints",
            tuple(request.origin),
            target,
        )

    def reassign_footprint(self, source, target_neuron=None):
        return self._apply_manipulation(
            ("new_neuron" if target_neuron is None else "reassign"),
            (source,),
            (
                ()
                if target_neuron is None
                else (
                    NeuronComponent(
                        int(target_neuron),
                        None,
                    ),
                )
            ),
        )

    def _merge_neuron_request_inplace(self, request):
        assignments = self.assignments

        if request.assignments is not assignments:
            raise ValueError("The active assignments changed; start a new request.")

        if self.state.tasks.processing_busy() or self.state.tasks.processing_requested:
            raise ValueError("Wait for assignment processing to finish.")

        request.refresh()
        if not request.is_complete:
            raise ValueError(request.error)

        source = request.source_id
        target = request.target_id

        if self.neuron_merge_distance(source, target) > 10.0:
            raise ValueError("The union centroids are now more than 10 px apart.")

        # Prepare changes separately. Publish only after union rebuilding
        # and row compaction have succeeded.
        candidate = copy(assignments)
        candidate.ids = assignments.ids.copy()
        candidate.stats = {
            key: values.copy() for key, values in assignments.stats.items()
        }
        candidate.review_status = assignments.review_status.copy()
        candidate.manipulation_id = dict(assignments.manipulation_id)

        candidate.union = copy(assignments.union)
        union = candidate.union
        union.centroids = union.centroids.copy()
        union.included = union.included.copy()
        union.synthetic = union.synthetic.copy()
        if union.background is not None:
            union.background = union.background.copy()

        sessions = np.flatnonzero(candidate.ids[source] >= 0)
        candidate.ids[target, sessions] = candidate.ids[source, sessions]
        candidate.ids[source, :] = -1

        # Existing matching scores no longer describe this combined neuron.
        for values in candidate.stats.values():
            values[target, ...] = np.nan
            values[source, ...] = np.nan

        candidate.review_status[target] = ReviewStatus.PENDING
        union.synthetic[target] |= union.synthetic[source]

        worker = copy(self)
        worker._assignments = {self._current_assignments: candidate}
        worker._current_assignments = self._current_assignments

        worker.rebuild_union_neurons([source, target])
        worker.rebuild_union_included()

        # Remove the now-empty source row.
        keep = np.delete(np.arange(len(candidate.ids)), source)
        row_map = {int(old): new for new, old in enumerate(keep)}

        candidate.ids = candidate.ids[keep]
        candidate.review_status = candidate.review_status[keep]
        candidate.stats = {key: values[keep] for key, values in candidate.stats.items()}
        candidate.manipulation_id = {
            row_map[old]: manipulation_id
            for old, manipulation_id in candidate.manipulation_id.items()
            if old in row_map
        }
        candidate.matched_status = np.any(candidate.ids >= 0, axis=0)

        union.footprints = union.footprints[:, keep].tocsc()
        union.centroids = union.centroids[keep]
        union.included = union.included[keep]
        union.synthetic = union.synthetic[keep]
        union.n_neurons = len(keep)

        # Preserve the Assignments object's identity.
        assignments.__dict__.update(candidate.__dict__)

        # Selections of moved components follow the destination neuron.
        row_map[source] = row_map[target]
        self.state.apply_assignment_rebuild(
            assignments.ids,
            row_map,
            from_session_id=len(self.sessions),
        )
        self.notify_change(
            *ASSIGNMENT_CONTENT_CHANGES,
            C.REVIEW_STATUS,
        )

    def change_review_status(self, neuron_id: int, status: ReviewStatus):

        self.assignments.review_status[neuron_id] = status
        self.notify_change(C.REVIEW_STATUS)

    def unassign_neurons(self, session_id: int, **kwargs):

        super().unassign_neurons(session_id)
        self.state.assignments = self.assignments.ids
        if self.state.current_session_id == session_id:
            self.state.current_session_id = None

        self.adjust_selected_components_after_data_change(session_id, -1)
        self.notify_change(*ASSIGNMENT_CONTENT_CHANGES)

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

    def _process_component_request_inplace(
        self,
        request: RequestHandler,
    ):
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

        self.notify_change(
            C.FOOTPRINT_GEOMETRY,
            *ASSIGNMENT_CONTENT_CHANGES,
        )

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
        self.notify_change(C.REVIEW_STATUS)

    def curation_targets(self, components, *, whole_neurons=False, remove=False):
        if self.assignments is None or self.assignments.union is None:
            raise ValueError("Load or calculate assignments first.")

        ids = self.assignments.ids
        slots = set()

        for component in components:
            neuron_id = int(component.neuron_id)
            if not 0 <= neuron_id < ids.shape[0]:
                raise ValueError("The selection contains an outdated neuron.")

            if whole_neurons:
                slots.update(
                    (neuron_id, int(session_id))
                    for session_id in np.flatnonzero(ids[neuron_id] >= 0)
                )
            else:
                session_id = component.session_id
                if session_id is None:
                    raise ValueError(
                        "Select session-specific components for this operation."
                    )
                session_id = int(session_id)
                if not 0 <= session_id < ids.shape[1]:
                    raise ValueError("The selection contains an outdated session.")
                slots.add((neuron_id, session_id))

        targets = []
        for neuron_id, session_id in sorted(slots):
            footprint_id = int(ids[neuron_id, session_id])
            if footprint_id < 0:
                continue

            session = self.sessions[session_id]
            if not 0 <= footprint_id < len(session.included):
                raise ValueError("An assigned footprint is missing from its session.")

            # Exclusion is idempotent; removal also applies to excluded items.
            if remove or session.included[footprint_id]:
                targets.append((neuron_id, session_id, footprint_id))

        return tuple(targets)

    def apply_curation_targets(self, targets, *, remove=False, whole_neurons=False):
        if not targets:
            return

        assignments = self.assignments
        if assignments is None or assignments.union is None:
            raise ValueError("No assignments are available.")

        # Validate the entire target set before changing anything.
        for neuron_id, session_id, footprint_id in targets:
            if (
                not 0 <= neuron_id < assignments.ids.shape[0]
                or not 0 <= session_id < assignments.ids.shape[1]
                or assignments.ids[neuron_id, session_id] != footprint_id
                or not 0 <= footprint_id < len(self.sessions[session_id].included)
            ):
                raise ValueError("Assignments changed; select the components again.")

        old_count = assignments.ids.shape[0]
        affected = {neuron_id for neuron_id, _, _ in targets}
        component_map = {}

        # Component exclusion creates one excluded singleton per footprint.
        detach = not remove and not whole_neurons
        if detach:
            assignments.pad_empty(n_neurons=len(targets), n_sessions=0)

        for index, (neuron_id, session_id, footprint_id) in enumerate(targets):
            self.sessions[session_id].included[footprint_id] = False

            if remove or detach:
                assignments.ids[neuron_id, session_id] = -1
                for values in assignments.stats.values():
                    values[neuron_id, session_id, ...] = np.nan

            if detach:
                new_id = old_count + index
                assignments.ids[new_id, session_id] = footprint_id
                for key, values in assignments.stats.items():
                    values[new_id, session_id, ...] = (
                        assignments.stats_default_value.get(key, np.nan)
                    )
                affected.add(new_id)
                component_map[(neuron_id, session_id)] = (new_id, session_id)
            elif remove:
                component_map[(neuron_id, session_id)] = None

        structural = remove or detach

        if structural:
            self.rebuild_union_neurons(sorted(affected))

        self.rebuild_union_included()

        # Compact once, after every target has been processed.
        keep = np.flatnonzero(np.any(assignments.ids >= 0, axis=1))
        row_map = {int(old): new for new, old in enumerate(keep)}

        if structural and len(keep) != assignments.ids.shape[0]:
            assignments.ids = assignments.ids[keep]
            assignments.review_status = assignments.review_status[keep]
            assignments.stats = {
                key: values[keep] for key, values in assignments.stats.items()
            }
            assignments.manipulation_id = {
                row_map[old]: manipulation_id
                for old, manipulation_id in assignments.manipulation_id.items()
                if old in row_map
            }

            union = assignments.union
            union.footprints = union.footprints[:, keep].tocsc()
            if union.centroids is not None:
                union.centroids = union.centroids[keep]
            union.included = union.included[keep]
            union.synthetic = union.synthetic[keep]
            union.n_neurons = len(keep)
        elif not structural:
            row_map = {index: index for index in range(assignments.ids.shape[0])}

        assignments.matched_status = np.any(assignments.ids >= 0, axis=0)

        component_map = {
            old: (None if target is None else (row_map[target[0]], target[1]))
            for old, target in component_map.items()
        }

        self.state.apply_assignment_rebuild(
            assignments.ids,
            row_map,
            from_session_id=len(self.sessions),
            component_id_map=component_map,
        )

        if structural:
            self.notify_change(*ASSIGNMENT_CONTENT_CHANGES)
        else:
            self.notify_change(C.INCLUSION, neuron_ids=affected)
