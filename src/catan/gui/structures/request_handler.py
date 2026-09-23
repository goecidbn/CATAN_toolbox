from typing import Optional
from typing_extensions import Literal

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
