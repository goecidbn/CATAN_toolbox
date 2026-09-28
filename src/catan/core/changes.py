from dataclasses import dataclass
from enum import Enum


class ChangeKind(str, Enum):
    SESSION_ADDED = "session_added"
    SESSION_REMOVED = "session_removed"
    SESSION_ORDER = "session_order"
    SESSION_METADATA = "session_metadata"
    SESSION_ACTIVITY = "session_activity"
    SESSION_TIMEBASE = "session_timebase"

    FOOTPRINT_GEOMETRY = "footprint_geometry"
    BACKGROUND_IMAGE = "background_image"
    UNION_GEOMETRY = "union_geometry"

    TRACE_VALUES = "trace_values"
    QUALITY_VALUES = "quality_values"
    DATA_AVAILABILITY = "data_availability"

    ASSIGNMENT_SET = "assignment_set"
    ASSIGNMENT_MAPPING = "assignment_mapping"
    MATCH_VALUES = "match_values"
    INCLUSION = "inclusion"
    REVIEW_STATUS = "review_status"

    MODEL_COUNTS = "model_counts"
    MODEL_PARAMETERS = "model_parameters"
    PROCESSING_STATUS = "processing_status"


@dataclass(frozen=True, slots=True)
class Change:
    kind: ChangeKind

    # None means unrestricted/all; an empty set means none.
    session_paths: frozenset[str] | None = None
    neuron_ids: frozenset[int] | None = None

    # DATA_AVAILABILITY: groups such as "spatial", "traces", "quality".
    # Value changes: field names within the corresponding category.
    fields: frozenset[str] | None = None

    # Names identify the affected stored assignment/model set.
    # None means unspecified: consumers must handle it conservatively.
    assignment_name: str | None = None
    model_name: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "kind", ChangeKind(self.kind))

        for name in ("session_paths", "neuron_ids", "fields"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, frozenset(value))


@dataclass(frozen=True, slots=True)
class DataChange:
    changes: tuple[Change, ...]

    def __post_init__(self):
        changes = tuple(self.changes)

        if not changes or not all(
            isinstance(item, Change) for item in changes
        ):
            raise ValueError(
                "DataChange requires one or more Change entries."
            )

        object.__setattr__(self, "changes", changes)

    @property
    def kinds(self) -> frozenset[ChangeKind]:
        return frozenset(item.kind for item in self.changes)

    def has(self, *kinds: ChangeKind) -> bool:
        return bool(self.kinds.intersection(kinds))

    def has_availability(self, *groups: str) -> bool:
        return any(
            change.kind == ChangeKind.DATA_AVAILABILITY
            and (
                not groups
                or change.fields is None
                or bool(change.fields.intersection(groups))
            )
            for change in self.changes
        )


# These changes can alter neuron/session identities, coordinate arrays,
# or the mapping from tracked neurons to source footprints.
TRACKED_STATISTIC_STRUCTURE = frozenset({
    ChangeKind.SESSION_ADDED,
    ChangeKind.SESSION_REMOVED,
    ChangeKind.SESSION_ORDER,
    ChangeKind.ASSIGNMENT_SET,
    ChangeKind.ASSIGNMENT_MAPPING,
})

SESSION_STRUCTURE_CHANGES = (
    ChangeKind.SESSION_ADDED,
    ChangeKind.SESSION_REMOVED,
    ChangeKind.SESSION_ORDER,
)

ASSIGNMENT_CONTENT_CHANGES = (
    ChangeKind.ASSIGNMENT_MAPPING,
    ChangeKind.UNION_GEOMETRY,
    ChangeKind.MATCH_VALUES,
    ChangeKind.INCLUSION,
    ChangeKind.PROCESSING_STATUS,
)