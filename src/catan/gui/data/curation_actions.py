"""Validated, isolated preparation of Curator manipulations."""

from copy import copy, deepcopy
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from catan.core.structures import NeuronComponent
from catan.core.changes import (
    ChangeKind as C,
    ASSIGNMENT_CONTENT_CHANGES,
)
from catan.tracking.structures import ReviewStatus


@dataclass(frozen=True)
class ManipulationSpec:
    kind: str = "none"
    group_id: str | None = None
    confirm_each: bool = True

    # Reserved for serialized validation-preset snapshots.
    validation_presets: tuple = ()

    def __post_init__(self):
        if self.kind not in (
            "none",
            "split",
            "merge",
            "neuron_merge",
            "reassign",
        ):
            raise ValueError(f"Unknown manipulation: {self.kind}")

        object.__setattr__(
            self,
            "validation_presets",
            tuple(deepcopy(self.validation_presets)),
        )


@dataclass(frozen=True)
class ValidationReport:
    # passed / rejected / unavailable
    status: str
    reasons: tuple[str, ...] = ()

    def require_pass(self):
        if self.status != "passed":
            raise ValueError("; ".join(self.reasons) or "Validation did not pass.")


@dataclass(frozen=True)
class EndpointRef:
    """Neuron identity through its physical footprint membership."""

    members: tuple[tuple[int, int], ...]
    session_id: int | None

    @classmethod
    def capture(cls, ids, component):
        neuron = int(component.neuron_id)

        if not 0 <= neuron < len(ids):
            raise ValueError("Invalid endpoint neuron.")

        if (
            component.session_id is not None
            and not 0 <= component.session_id < ids.shape[1]
        ):
            raise ValueError("Invalid endpoint session.")

        row = ids[neuron]

        members = tuple((int(s), int(row[s])) for s in np.flatnonzero(row >= 0))

        if not members:
            raise ValueError("An endpoint neuron is empty.")

        if component.session_id is not None and row[component.session_id] < 0:
            raise ValueError("An endpoint footprint is missing.")

        return cls(members, component.session_id)

    def resolve(self, ids):
        sid, fp = (
            self.members[0]
            if self.session_id is None
            else (
                self.session_id,
                dict(self.members)[self.session_id],
            )
        )

        rows = np.flatnonzero(ids[:, sid] == fp)

        if len(rows) != 1:
            raise ValueError(
                "Candidate no longer has a unique owner. " "Evaluate again."
            )

        neuron = int(rows[0])
        row = ids[neuron]

        members = tuple((int(s), int(row[s])) for s in np.flatnonzero(row >= 0))

        if self.session_id is None and members != self.members:
            raise ValueError("Candidate membership changed. Evaluate again.")

        return NeuronComponent(neuron, self.session_id)


@dataclass(frozen=True)
class ManipulationPlan:
    kind: str
    sources: tuple[EndpointRef, ...]
    targets: tuple[EndpointRef, ...]

    assignments: object = field(
        compare=False,
        repr=False,
    )
    sessions: tuple = field(
        compare=False,
        repr=False,
    )

    @classmethod
    def capture(cls, data, kind, sources, targets):
        ids = data.assignments.ids

        return cls(
            kind=kind,
            sources=tuple(EndpointRef.capture(ids, component) for component in sources),
            targets=tuple(EndpointRef.capture(ids, component) for component in targets),
            assignments=data.assignments,
            sessions=tuple(data.sessions),
        )

    def resolve(self, data):
        if (
            data.assignments is not self.assignments
            or len(data.sessions) != len(self.sessions)
            or any(
                current is not original
                for current, original in zip(
                    data.sessions,
                    self.sessions,
                )
            )
        ):
            raise ValueError("Assignments or sessions changed. Evaluate again.")

        ids = data.assignments.ids

        return (
            tuple(ref.resolve(ids) for ref in self.sources),
            tuple(ref.resolve(ids) for ref in self.targets),
        )


@dataclass
class PreparedManipulation:
    plan: ManipulationPlan
    version: int

    candidate: object
    candidate_data: object
    changed_sessions: dict

    row_map: dict
    component_map: dict

    affected_before: tuple
    affected_after: tuple

    # Edited original footprint -> resulting footprint(s).
    replacements: dict

    validation: ValidationReport = field(
        default_factory=lambda: ValidationReport(
            "unavailable",
            ("Not checked.",),
        )
    )

    committed: bool = False


def clone_session(session, *, union=False):
    result = copy(session)

    array_fields = (
        "centroids",
        "included",
        "synthetic",
    )

    if union:
        array_fields += ("background",)

    for name in array_fields:
        value = getattr(session, name, None)

        if value is not None:
            setattr(result, name, value.copy())

    # append_synthetic_component replaces the arrays,
    # but modifies these dictionaries.
    for name in (
        "_traces",
        "quality",
        "status",
        "params",
    ):
        value = getattr(session, name, None)

        if isinstance(value, dict):
            setattr(result, name, value.copy())

    # The existing preparation methods read/replace footprint
    # matrices; they do not edit their shared buffers in place.
    return result


class CandidateState:
    """Minimal state for preparing changes without GUI signals."""

    def __init__(self, ids):
        self.assignments = ids
        self.row_map = {neuron: neuron for neuron in range(len(ids))}

        self.tasks = SimpleNamespace(
            processing_busy=lambda: False,
            processing_requested=False,
        )

    def get_footprint_from_component(self, component):
        value = int(self.assignments[component.id])
        return None if value < 0 else value

    def apply_assignment_rebuild(
        self,
        ids,
        row_map,
        **kwargs,
    ):
        self.assignments = ids
        self.row_map = dict(row_map)


class OccupiedDestination(ValueError):
    def __init__(self, slots):
        self.slots = tuple(slots)
        super().__init__(
            "Destination slots are already occupied:\n"
            + "\n".join(
                f"Neuron {n}, session {sid}: footprint {fp}"
                for n, sid, fp in self.slots
            )
        )


def occupied_destinations(data, kind, sources, targets):
    if kind == "split":
        sid = targets[0].session_id
        origin_neurons = {c.neuron_id for c in targets}
        neurons = {c.neuron_id for c in sources} - origin_neurons

    elif kind == "reassign":
        sid = sources[0].session_id
        neurons = {targets[0].neuron_id}

    else:
        return ()

    return tuple(
        (n, sid, int(data.assignments.ids[n, sid]))
        for n in sorted(neurons)
        if data.assignments.ids[n, sid] >= 0
    )


def _detach_occupied(worker, slots):
    affected = set()

    for neuron_id, sid, footprint_id in slots:
        candidate = worker.assignments
        new_neuron_id = len(candidate.ids)

        candidate.pad_empty(n_neurons=1, n_sessions=0)
        candidate.ids[new_neuron_id, sid] = footprint_id
        candidate.ids[neuron_id, sid] = -1

        worker.state.assignments = candidate.ids

        for values in candidate.stats.values():
            values[[neuron_id, new_neuron_id], ...] = np.nan

        worker.rebuild_union_neurons([neuron_id, new_neuron_id])

        candidate.register_manipulation(
            manipulation_type="new_neuron",
            origin=[],
            sources=[],
            results=[],
            affected_neurons=[neuron_id, new_neuron_id],
        )
        affected.update((neuron_id, new_neuron_id))

    return affected


def _validate(data, kind, sources, targets):
    from catan.gui.structures.request_handler import (
        RequestHandler,
        NeuronMergeRequest,
    )

    expected = {
        "split": (2, 1),
        "merge": (1, 2),
        "neuron_merge": (1, 1),
        "reassign": (1, 1),
        # Used by the existing manual "new neuron" action.
        "new_neuron": (1, 0),
    }

    actual = (len(sources), len(targets))

    if kind not in expected or actual != expected[kind]:
        raise ValueError(
            f"{kind}: expected source/target counts "
            f"{expected.get(kind)}; received {actual}."
        )

    endpoints = sources + targets

    if len(set(endpoints)) != len(endpoints):
        raise ValueError("Sources and targets must be distinct.")

    for component in endpoints:
        if not data.assignments.union.included[component.neuron_id]:
            raise ValueError("An endpoint neuron is excluded.")

        if component.session_id is None:
            continue

        session = data.sessions[component.session_id]
        footprint = int(data.assignments.ids[component.id])

        if (
            footprint < 0
            or footprint >= session.n_neurons
            or not session.included[footprint]
        ):
            raise ValueError("An endpoint footprint is missing or excluded.")

        if not session.status.get("aligned", False) or data.alignment_is_stale(
            component.session_id
        ):
            raise ValueError(
                "Align the endpoint sessions before " "manipulating footprints."
            )

    if kind in ("split", "merge"):
        request = RequestHandler(kind)

        # Existing request terminology:
        # origin = edited targets
        # destination = reference sources
        for component in targets:
            request.add_component(component)

        request.advance_stage()

        for component in sources:
            request.add_component(component)

        if not request.is_complete:
            raise ValueError("Incomplete footprint manipulation.")

        if kind == "split" and len({c.neuron_id for c in sources}) != 2:
            raise ValueError(
                "Split references must belong to " "two different neurons."
            )

        target_session = targets[0].session_id

        if any(
            data.sessions[c.session_id].dims != data.sessions[target_session].dims
            for c in sources
        ):
            raise ValueError(
                "Reference and edited sessions have " "different image dimensions."
            )

        return request

    if kind == "neuron_merge":
        if any(component.session_id is not None for component in endpoints):
            raise ValueError("Neuron merging requires neuron endpoints.")

        request = NeuronMergeRequest(
            data.assignments,
            sources[0].neuron_id,
            targets[0].neuron_id,
        )

        if not request.is_complete:
            raise ValueError(request.error)

        return request

    if sources[0].session_id is None or (targets and targets[0].session_id is not None):
        raise ValueError(
            "Reassignment requires a source footprint " "and a target neuron."
        )

    if targets and sources[0].neuron_id == targets[0].neuron_id:
        raise ValueError("This footprint already belongs to the target neuron.")

    return None


def prepare(data, plan, *, detach_occupied=False):
    sources, targets = plan.resolve(data)
    _validate(data, plan.kind, sources, targets)

    slots = occupied_destinations(data, plan.kind, sources, targets)
    if slots and not detach_occupied:
        raise OccupiedDestination(slots)

    version = data.state.data_version

    original = data.assignments

    candidate = copy(original)

    for name in (
        "ids",
        "review_status",
        "matched_status",
    ):
        setattr(
            candidate,
            name,
            getattr(original, name).copy(),
        )

    candidate.stats = {key: values.copy() for key, values in original.stats.items()}
    candidate.manipulations = deepcopy(original.manipulations)
    candidate.manipulation_id = dict(original.manipulation_id)
    candidate.union = clone_session(
        original.union,
        union=True,
    )

    worker = copy(data)
    worker._assignments = {data._current_assignments: candidate}
    worker.sessions = list(data.sessions)

    changed = {}

    if plan.kind in ("split", "merge"):
        session_id = targets[0].session_id
        changed[session_id] = clone_session(data.sessions[session_id])
        worker.sessions[session_id] = changed[session_id]

    worker.state = CandidateState(candidate.ids)
    worker.notify_change = lambda *args, **kwargs: None

    # A future guard evaluator must create its own engine.
    worker.statistic_engine = None

    affected = {c.neuron_id for c in sources + targets}
    affected.update(_detach_occupied(worker, slots))

    request = _validate(worker, plan.kind, sources, targets)
    replacements = {}

    if plan.kind in ("split", "merge"):
        worker._process_component_request_inplace(request)

        session_id = targets[0].session_id

        result_ids = (
            [component.neuron_id for component in sources]
            if plan.kind == "split"
            else [targets[0].neuron_id]
        )

        results = tuple(NeuronComponent(neuron, session_id) for neuron in result_ids)

        replacements = {component: results for component in targets}

    elif plan.kind == "neuron_merge":
        worker._merge_neuron_request_inplace(request)

    else:
        source = sources[0]

        target = targets[0].neuron_id if targets else len(candidate.ids)

        if not targets:
            candidate.pad_empty(
                n_neurons=1,
                n_sessions=0,
            )

        candidate.ids[target, source.session_id] = candidate.ids[source.id]
        candidate.ids[source.id] = -1

        affected.add(target)

        for values in candidate.stats.values():
            values[list(affected), ...] = np.nan

        worker.rebuild_union_neurons(list(affected))
        worker.rebuild_union_included()

        # Compact empty rows, retaining an explicit identity map.
        keep = np.flatnonzero(np.any(candidate.ids >= 0, axis=1))

        row_map = {int(old): new for new, old in enumerate(keep)}

        if source.neuron_id not in row_map:
            row_map[source.neuron_id] = row_map[target]

        candidate.ids = candidate.ids[keep]
        candidate.stats = {key: values[keep] for key, values in candidate.stats.items()}
        candidate.review_status = candidate.review_status[keep]

        kept = set(keep)
        candidate.manipulation_id = {
            row_map[neuron]: value
            for neuron, value in candidate.manipulation_id.items()
            if neuron in kept
        }

        union = candidate.union
        union.footprints = union.footprints[:, keep].tocsc()

        for name in (
            "centroids",
            "included",
            "synthetic",
        ):
            setattr(
                union,
                name,
                getattr(union, name)[keep],
            )

        union.n_neurons = len(keep)
        worker.state.row_map = row_map

        replacements[source] = (
            NeuronComponent(
                row_map[target],
                source.session_id,
            ),
        )

    candidate = worker.assignments
    row_map = worker.state.row_map

    after = {row_map.get(neuron, neuron) for neuron in affected}
    after = {
        neuron
        for neuron in after
        if (neuron < len(candidate.ids) and np.any(candidate.ids[neuron] >= 0))
    }

    candidate.review_status[list(after)] = ReviewStatus.PENDING
    candidate.matched_status = np.any(
        candidate.ids >= 0,
        axis=0,
    )

    worker.rebuild_union_included()

    # Preserve explicitly excluded, unrelated union neurons.
    for old, new in row_map.items():
        if old < len(original.union.included) and not original.union.included[old]:
            candidate.union.included[new] = False

    if plan.kind not in ("split", "merge"):
        candidate.register_manipulation(
            manipulation_type=plan.kind,
            origin=[],
            sources=[],
            results=[],
            affected_neurons=after,
        )

    # Follow physical footprints when remapping GUI selections.
    owners = {
        (session_id, int(footprint)): neuron
        for neuron, row in enumerate(candidate.ids)
        for session_id, footprint in enumerate(row)
        if footprint >= 0
    }

    component_map = {}

    for neuron, row in enumerate(original.ids):
        for session_id, footprint in enumerate(row):
            if footprint < 0:
                continue

            owner = owners.get((session_id, int(footprint)))
            included = worker.sessions[session_id].included[footprint]

            if owner != row_map.get(neuron) or not included:
                component_map[(neuron, session_id)] = (
                    None if owner is None or not included else (owner, session_id)
                )

    worker.state.assignments = candidate.ids

    return PreparedManipulation(
        plan=plan,
        version=version,
        candidate=candidate,
        candidate_data=worker,
        changed_sessions=changed,
        row_map=row_map,
        component_map=component_map,
        affected_before=tuple(
            NeuronComponent(neuron, None)
            for neuron in sorted(affected)
            if neuron < len(original.ids)
        ),
        affected_after=tuple(NeuronComponent(neuron, None) for neuron in sorted(after)),
        replacements=replacements,
    )


def validate_prepared(prepared, spec):
    if spec.validation_presets:
        return ValidationReport(
            "unavailable",
            (
                "This preset requests before/after validation "
                "presets. That evaluator is not implemented "
                "yet; no changes were applied.",
            ),
        )

    return ValidationReport("passed")


def commit(data, prepared):
    prepared.validation.require_pass()

    if prepared.committed:
        raise ValueError("This manipulation was already committed.")

    if data.state.data_version != prepared.version:
        raise ValueError("Data changed during preparation; prepare again.")

    prepared.plan.resolve(data)
    previous_focus = data.state.focused_component

    for session_id, candidate_session in prepared.changed_sessions.items():
        data.sessions[session_id].__dict__.update(candidate_session.__dict__)

    data.assignments.__dict__.update(prepared.candidate.__dict__)

    prepared.committed = True

    data.state.apply_assignment_rebuild(
        data.assignments.ids,
        prepared.row_map,
        from_session_id=len(data.sessions),
        component_id_map=prepared.component_map,
    )

    results = tuple(
        component for values in prepared.replacements.values() for component in values
    )

    focused = data.state.focused_component
    replacements = prepared.replacements.get(previous_focus, ())

    if replacements:
        focused = replacements[0]

    elif focused is None or not np.any(data.assignments.ids[focused.neuron_id] >= 0):
        focused = results[0] if results else None

    if focused is not None:
        # Add results before focusing. Otherwise State's focus setter
        # can replace the entire selection with the focused component.
        selected = set(data.state.selected_components or ()) | set(results) | {focused}
        data.state.update_selected_components(list(selected), [])
        data.state.focused_component = focused

    data.notify_change(
        *ASSIGNMENT_CONTENT_CHANGES,
        C.REVIEW_STATUS,
        *([C.FOOTPRINT_GEOMETRY] if prepared.changed_sessions else []),
    )

    return prepared
