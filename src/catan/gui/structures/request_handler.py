from typing import Optional
from typing_extensions import Literal

import numpy as np

from catan.gui.structures.state import NeuronComponent


class RequestHandler:

    type: Literal["split", "merge"]
    stage: Literal["origin", "destination"]

    origin: list[NeuronComponent]
    destination: list[NeuronComponent]

    def __init__(self, request_type: str, component: Optional[NeuronComponent] = None):
        if request_type not in ("split", "merge"):
            raise ValueError(f"Unknown request type: {request_type!r}")

        self.type = request_type
        self.stage = "origin"

        self.origin = []
        self.destination = []

        if component is not None:
            self.origin.append(component)

    @property
    def components(self) -> list[NeuronComponent]:
        return [
            *self.origin,
            *self.destination,
        ]

    @property
    def is_complete(self) -> bool:

        if self.stage == "origin":
            return len(self.origin) == self.n_components_origin

        if self.stage == "destination":
            return (
                len(self.origin) == self.n_components_origin
                and len(self.destination) == self.n_components_destination
            )

        return False

    @property
    def n_components_origin(self) -> int:
        if self.type == "split":
            return 1
        elif self.type == "merge":
            return 2
        return 0

    @property
    def n_components_destination(self) -> int:
        if self.type == "split":
            return 2
        elif self.type == "merge":
            return 1
        return 0

    def validate_component(
        self,
        component: NeuronComponent,
    ) -> bool:
        return self.validation_error(component) is None

    def validation_error(
        self,
        component: NeuronComponent,
    ) -> str | None:

        if component.session_id is None:
            return "A concrete session component must be selected."

        if component in self.components:
            return "This component is already part of the request."

        # --------------------------------------------------
        # Origin
        # --------------------------------------------------

        if self.stage == "origin":

            if len(self.origin) >= self.n_components_origin:
                return "All origin components are already selected."

            if self.origin:

                origin_session = self.origin[0].session_id

                if component.session_id != origin_session:
                    return "All origin components must belong " "to the same session."

                if any(
                    existing.neuron_id == component.neuron_id
                    for existing in self.origin
                ):
                    return "Origin components must be " "different neurons."

            return None

        # --------------------------------------------------
        # Destination
        # --------------------------------------------------

        if self.stage == "destination":

            if len(self.destination) >= self.n_components_destination:
                return "All destination components are already selected."

            if not self.origin:
                return "The request has no origin component."

            origin_session = self.origin[0].session_id

            if component.session_id == origin_session:
                return (
                    "Destination components must come from "
                    "a different session than the origin."
                )

            return None

        return f"Unknown request stage: {self.stage!r}"

    def add_component(self, component: NeuronComponent):

        error = self.validation_error(component)

        if error is not None:
            raise ValueError(error)

        if self.stage == "origin":
            self.origin.append(component)
            return

        if self.stage == "destination":
            self.destination.append(component)
            return

        raise ValueError(f"Unknown request stage: {self.stage!r}")

    def remove_component(
        self,
        component: NeuronComponent,
    ) -> bool:

        target = self.origin if self.stage == "origin" else self.destination

        try:
            target.remove(component)
        except ValueError:
            return False

        return True

    def advance_stage(self):

        if self.stage != "origin":
            raise ValueError(f"Cannot advance request from stage {self.stage!r}.")

        if len(self.origin) != self.n_components_origin:
            raise ValueError(
                "Cannot continue: "
                f"{len(self.origin)}/{self.n_components_origin} "
                "origin components selected."
            )

        self.stage = "destination"

    @property
    def stage_complete(self) -> bool:

        if self.stage == "origin":
            return len(self.origin) == self.n_components_origin

        if self.stage == "destination":
            return len(self.destination) == self.n_components_destination

        return False

    def origin_status_text(self):
        selected = ", ".join(f"({c.session_id}, {c.neuron_id})" for c in self.origin)

        return f"Origin: " f"{len(self.origin)}/{self.n_components_origin}" + (
            f" — {selected}" if selected else ""
        )

    def destination_status_text(self):
        selected = ", ".join(
            f"({c.session_id}, {c.neuron_id})" for c in self.destination
        )

        return (
            f"Destination: "
            f"{len(self.destination)}/"
            f"{self.n_components_destination}" + (f" — {selected}" if selected else "")
        )


# Part of the source neuron moved by a neuron merge, relative to the
# session of a source footprint (included).
MERGE_EXTENTS = ("all", "previous", "following")


def merge_extent_sessions(row, extent="all", session_id=None):
    """Sessions of an assignment row moved by a neuron merge."""
    if extent not in MERGE_EXTENTS:
        raise ValueError(f"Unknown merge extent: {extent!r}")

    sessions = np.flatnonzero(np.asarray(row) >= 0)

    if extent == "all":
        return sessions

    if session_id is None:
        raise ValueError("A partial neuron merge needs a source footprint.")

    if extent == "previous":
        return sessions[sessions <= session_id]

    return sessions[sessions >= session_id]


class NeuronMergeRequest:
    """Move footprints of one neuron into another (or a new) neuron.

    The moved set starts as all footprints, a part relative to a session
    (``extent``), or explicit ``sessions`` (e.g. highlighted footprints),
    and can be edited with toggle(). ``target_id`` None: a new neuron.
    """

    type = "neuron_merge"
    stage = "destination"

    def __init__(
        self,
        assignments,
        source_id,
        target_id,
        extent="all",
        session_id=None,
        sessions=None,
    ):
        self.assignments = assignments
        self.source_id = int(source_id)
        self.target_id = None if target_id is None else int(target_id)
        self.highlight_mode = "footprints"
        self.conflicts = ()
        self.origin = []
        self.destination = []
        self.error = None

        if sessions is None:
            row = assignments.ids[self.source_id] if 0 <= self.source_id < len(
                assignments.ids
            ) else ()
            sessions = merge_extent_sessions(row, extent, session_id)

        self.sessions = {int(s) for s in sessions}
        self.refresh()

        if self.conflicts:
            self.highlight_mode = "conflicts"

    @property
    def moves_whole_neuron(self):
        """All footprints of the source are moved."""
        row = self.assignments.ids[self.source_id]
        return set(np.flatnonzero(row >= 0).tolist()) <= self.sessions

    def refresh(self):
        ids = self.assignments.ids
        self.error = None
        self.origin = []
        self.destination = []
        self.conflicts = ()

        if (
            self.source_id == self.target_id
            or not 0 <= self.source_id < len(ids)
            or (self.target_id is not None and not 0 <= self.target_id < len(ids))
        ):
            self.error = "The source or destination neuron is no longer available."
            return

        source_sessions = np.array(
            sorted(s for s in self.sessions if ids[self.source_id, s] >= 0),
            dtype=int,
        )
        target_sessions = (
            np.zeros(0, dtype=int)
            if self.target_id is None
            else np.flatnonzero(ids[self.target_id] >= 0)
        )

        self.origin = [
            NeuronComponent(self.source_id, int(s))
            for s in source_sessions
        ]
        self.destination = [
            NeuronComponent(self.target_id, int(s))
            for s in target_sessions
        ]
        self.conflicts = tuple(
            int(s)
            for s in np.intersect1d(source_sessions, target_sessions)
        )

        if not self.origin:
            self.error = "No footprints selected to move."
        elif self.target_id is not None and not self.destination:
            self.error = "The destination neuron is empty."
        elif self.conflicts:
            self.error = (
                "The destination already has footprints in sessions "
                + ", ".join(map(str, self.conflicts))
                + ". Deselect these footprints or resolve the conflicts."
            )

        if not self.conflicts:
            self.highlight_mode = "footprints"

    def toggle(self, component):
        """Add or remove a source footprint; returns an error or None."""
        if component.neuron_id != self.source_id or component.session_id is None:
            return f"Only footprints of neuron {self.source_id} can be moved."

        session = int(component.session_id)
        if self.assignments.ids[self.source_id, session] < 0:
            return "This neuron has no footprint in that session."

        self.sessions ^= {session}
        self.refresh()
        return None

    @property
    def components(self):
        if self.highlight_mode == "conflicts" and self.conflicts:
            return [
                NeuronComponent(neuron_id, session_id)
                for session_id in self.conflicts
                for neuron_id in (self.source_id, self.target_id)
            ]
        return list(self.origin)

    @property
    def is_complete(self):
        return self.error is None

    @property
    def stage_complete(self):
        return self.is_complete

    def remap_rows(self, neuron_id_map):
        source = neuron_id_map.get(self.source_id)
        target = (
            None if self.target_id is None else neuron_id_map.get(self.target_id)
        )

        if (
            source is None
            or (self.target_id is not None and target is None)
            or source == target
        ):
            return None

        self.source_id = source
        self.target_id = target
        self.refresh()
        return self
