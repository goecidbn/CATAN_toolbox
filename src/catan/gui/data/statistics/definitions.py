from functools import partial

from .types import StatisticDefinition
from .queries import ReductionSpec
from .parameters import StatisticParameter
from .distance_statistics import distance_query
from .overlap_statistics import overlap_query
from .cooccurrence_statistics import cooccurrence_query
from . import calculations

from catan.core.changes import (
    ChangeKind,
    TRACKED_STATISTIC_STRUCTURE,
)

CALCULATED_STATISTICS = {
    "none": StatisticDefinition(
        key="none",
        title="None",
        description="No statistic selected.",
        dims=(),
        category=None,
        getter=lambda data, state, indexers, filters: None,  # lambda data: StatisticArray(np.array([]), dims=()),
    ),
    "footprint_size": StatisticDefinition(
        key="footprint_size",
        title="Footprint size",
        description="Size of the neuron footprint in pixels.",
        dims=("neuron", "session"),
        category="neuron",
        getter=calculations.calculate_footprint_size,
    ),
    "border_proximity": StatisticDefinition(
        key="border_proximity",
        title="Border distance (px)",
        description=(
            "Signed distance of aligned neuron centroids to the imaging "
            "boundary: positive inside, negative outside."
        ),
        dims=("neuron", "session"),
        category="neuron",
        getter=calculations.calculate_border_proximity,
        parameters=(
            StatisticParameter(
                key="common_border",
                label="Shared border",
                kind=bool,
                default=False,
                tooltip=(
                    "Use the area covered by all loaded sessions. "
                    "Otherwise use this session's own aligned imaging area."
                ),
            ),
        ),
    ),
    "occurence": StatisticDefinition(
        key="occurence",
        title="Occurrence",
        description="Number of sessions in which each neuron is present.",
        dims=("neuron", "session"),
        category="neuron",
        getter=calculations.calculate_occurrence,
        allowed_reductions={
            "neuron": ("keep", "single", "mean", "sum"),
            "session": ("keep", "single", "mean", "sum"),
        },
        default_reductions={
            "neuron": ReductionSpec("keep"),
            "session": ReductionSpec("sum"),
        },
    ),
    "centroid_shift": StatisticDefinition(
        key="centroid_shift",
        title="Centroid shift",
        description="Centroid distance between sessions.",
        dims=("neuron", "session_i", "session_j"),
        category="neuron",
        getter=calculations.calculate_centroid_shift,
        default_reductions={
            "neuron": ReductionSpec("keep"),
            "session_ref": ReductionSpec("max"),
            "session_target": ReductionSpec("max"),
        },
    ),
    "temporal_corr": StatisticDefinition(
        key="temporal_corr",
        title="Temporal correlation",
        description="Pairwise temporal trace correlation within sessions.",
        dims=("neuron_i", "neuron_j", "session"),
        category="pair",
        getter=calculations.calculate_temporal_correlation,
        default_reductions={
            "neuron_i": ReductionSpec("keep"),
            "neuron_j": ReductionSpec("keep"),
            "session": ReductionSpec("mean"),
        },
    ),
    "distances": StatisticDefinition(
        key="distances",
        title="Centroid distances",
        description="Pairwise Euclidean distances between neuron centroids.",
        dims=("neuron_i", "neuron_j", "session"),
        category="pair",
        getter=calculations.calculate_distances,
        default_reductions={
            "neuron_i": ReductionSpec("keep"),
            "neuron_j": ReductionSpec("keep"),
            "session": ReductionSpec("mean"),
        },
    ),
    "footprint_similarity": StatisticDefinition(
        key="footprint_similarity",
        title="Footprint similarity",
        description="Pairwise similarity between neuron footprints.",
        dims=("neuron_i", "neuron_j", "session_i", "session_j"),
        category="pair",
        getter=calculations.calculate_footprint_similarity,
        # All implemented numeric reductions work on sparse pair values.
        allowed_reductions=None,
        default_reductions={
            "neuron_i": ReductionSpec("keep"),
            "neuron_j": ReductionSpec("keep"),
            "session_i": ReductionSpec("single", 0),
            "session_j": ReductionSpec("single", 1),
        },
        parameters=(
            StatisticParameter(
                key="gamma",
                label="Gamma",
                kind=float,
                default=0.1,
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                decimals=3,
                tooltip=(
                    "Exponent of the overlap penalty in cosine_union. "
                    "Larger values penalize limited overlap more strongly."
                ),
            ),
        ),
        independent_pairs=True,
        compact_values=True,
    ),
}

# Preserve the existing key and dimensions for saved configurations.
definition = CALCULATED_STATISTICS["distances"]
definition.title = "Within-session distances (px)"
definition.query_getter = partial(distance_query, mode="within")
definition.independent_pairs = True
definition.compact_values = True

definition = CALCULATED_STATISTICS["centroid_shift"]
definition.title = "Within-neuron displacement (px)"
definition.query_getter = partial(
    distance_query,
    mode="displacement",
    exclude_self=False,
)
definition.independent_pairs = True
definition.compact_values = True

# These were previously named session_ref/session_target,
# which do not match the statistic's actual dimensions.
definition.default_reductions = {
    "neuron": ReductionSpec("keep"),
    "session_i": ReductionSpec("max"),
    "session_j": ReductionSpec("max"),
}


def query_only_getter(*args, **kwargs):
    raise RuntimeError("This statistic must be evaluated through StatisticEngine.")


CALCULATED_STATISTICS.update(
    {
        "cross_session_distances": StatisticDefinition(
            key="cross_session_distances",
            title="Cross-session distances (px)",
            description=(
                "Distances between aligned session footprints. "
                "The identical footprint is omitted; other footprints "
                "of the same neuron remain available."
            ),
            category="pair",
            dims=("neuron_i", "neuron_j", "session_i", "session_j"),
            getter=query_only_getter,
            query_getter=partial(distance_query, mode="cross"),
            independent_pairs=True,
            compact_values=True,
            default_reductions={
                "neuron_i": ReductionSpec("keep"),
                "neuron_j": ReductionSpec("keep"),
                "session_i": ReductionSpec("single", 0),
                "session_j": ReductionSpec("single", 0),
            },
        ),
        "footprint_union_distance": StatisticDefinition(
            key="footprint_union_distance",
            title="Footprint → union distance (px)",
            description=(
                "Distance from a session footprint of neuron_i "
                "to the union centroid of neuron_j."
            ),
            category="pair",
            dims=("neuron_i", "neuron_j", "session"),
            getter=query_only_getter,
            query_getter=partial(distance_query, mode="to_union"),
            independent_pairs=True,
            compact_values=True,
            default_reductions={
                "neuron_i": ReductionSpec("keep"),
                "neuron_j": ReductionSpec("keep"),
                "session": ReductionSpec("single", 0),
            },
        ),
        "union_distances": StatisticDefinition(
            key="union_distances",
            title="Union-centroid distances (px)",
            description=(
                "Distances between the existing union centroids. "
                "Identical-neuron comparisons are omitted."
            ),
            category="pair",
            dims=("neuron_i", "neuron_j"),
            getter=query_only_getter,
            query_getter=partial(distance_query, mode="union"),
            independent_pairs=True,
            compact_values=True,
            default_reductions={
                "neuron_i": ReductionSpec("keep"),
                "neuron_j": ReductionSpec("keep"),
            },
        ),
    }
)


definition = CALCULATED_STATISTICS["footprint_union_distance"]
definition.component_axes = (("neuron_i", "session"),)
definition.curation_kind = "reassignment"

definition = CALCULATED_STATISTICS["union_distances"]
definition.component_axes = ()
definition.curation_kind = "neuron_pair"

CALCULATED_STATISTICS["footprint_overlap"] = StatisticDefinition(
    key="footprint_overlap",
    title="Footprint overlap",
    description=(
        "Overlap of registered footprints without additional shifting. "
        "Returns a fraction between 0 and 1."
    ),
    dims=(
        "neuron_i",
        "neuron_j",
        "session_i",
        "session_j",
    ),
    category="pair",
    getter=query_only_getter,
    query_getter=overlap_query,
    independent_pairs=True,
    compact_values=True,
    allowed_reductions=None,
    default_reductions={
        "neuron_i": ReductionSpec("keep"),
        "neuron_j": ReductionSpec("keep"),
        "session_i": ReductionSpec("single", index=0),
        "session_j": ReductionSpec("single", index=0),
    },
    parameters=(
        StatisticParameter(
            key="weighted",
            label="Intensity-weighted",
            kind=bool,
            default=False,
            tooltip=(
                "Use peak-normalized intensities and their pixelwise "
                "minimum as the intersection. Otherwise count support pixels."
            ),
        ),
        StatisticParameter(
            key="containment",
            label="Normalize by smaller footprint",
            kind=bool,
            default=True,
            tooltip=(
                "Enabled: intersection divided by the smaller footprint area. "
                "Disabled: intersection divided by union area."
            ),
        ),
        StatisticParameter(
            key="support_threshold",
            label="Relative support threshold",
            kind=float,
            default=0.1,
            minimum=0.0,
            maximum=0.99,
            step=0.05,
            decimals=3,
            tooltip=(
                "Ignore pixels below this fraction of each footprint's "
                "own maximum. 0.1 means 10% of its peak."
            ),
        ),
        StatisticParameter(
            key="directional",
            label="Directional coverage",
            kind=bool,
            default=False,
            tooltip=(
                "Measure the fraction of one footprint covered by the other. "
                "Overrides 'Normalize by smaller footprint'. "
                "By default, normalize by the source footprint (i)."
            ),
        ),
        StatisticParameter(
            key="normalize_by_target",
            label="Normalize directional coverage by target",
            kind=bool,
            default=False,
            tooltip=(
                "When directional coverage is enabled, normalize by the "
                "target footprint (j), instead of the source footprint (i)."
            ),
        ),
    ),
)

_CALCULATED_DEPENDENCIES = {
    "footprint_size": (
        frozenset({ChangeKind.FOOTPRINT_GEOMETRY}),
        frozenset({"spatial"}),
    ),
    "border_proximity": (
        frozenset(
            {
                ChangeKind.FOOTPRINT_GEOMETRY,
                ChangeKind.BACKGROUND_IMAGE,
                ChangeKind.PROCESSING_STATUS,
                ChangeKind.SESSION_METADATA,
            }
        ),
        frozenset({"spatial"}),
    ),
    "occurence": (
        frozenset(),
        frozenset(),
    ),
    "centroid_shift": (
        frozenset({ChangeKind.FOOTPRINT_GEOMETRY}),
        frozenset({"spatial"}),
    ),
    "temporal_corr": (
        frozenset({ChangeKind.TRACE_VALUES}),
        frozenset({"traces"}),
    ),
    "distances": (
        frozenset({ChangeKind.FOOTPRINT_GEOMETRY}),
        frozenset({"spatial"}),
    ),
    "footprint_similarity": (
        frozenset({ChangeKind.FOOTPRINT_GEOMETRY}),
        frozenset({"spatial"}),
    ),
    "footprint_overlap": (
        frozenset(
            {
                ChangeKind.FOOTPRINT_GEOMETRY,
                ChangeKind.INCLUSION,
                ChangeKind.PROCESSING_STATUS,
                ChangeKind.SESSION_METADATA,
            }
        ),
        frozenset({"spatial"}),
    ),
}

CALCULATED_STATISTICS["none"].dependencies = frozenset()

for key, (dependencies, availability) in _CALCULATED_DEPENDENCIES.items():
    definition = CALCULATED_STATISTICS[key]
    definition.dependencies = TRACKED_STATISTIC_STRUCTURE | dependencies
    definition.availability_dependencies = availability

# Endpoint order is source, target.
for key in (
    "footprint_similarity",
    "footprint_overlap",
    "cross_session_distances",
):
    CALCULATED_STATISTICS[key].relationship_axes = (
        ("neuron_i", "session_i"),
        ("neuron_j", "session_j"),
    )

for key in ("distances", "temporal_corr"):
    CALCULATED_STATISTICS[key].relationship_axes = (
        ("neuron_i", "session"),
        ("neuron_j", "session"),
    )

CALCULATED_STATISTICS["centroid_shift"].relationship_axes = (
    ("neuron", "session_i"),
    ("neuron", "session_j"),
)

CALCULATED_STATISTICS["footprint_union_distance"].relationship_axes = (
    ("neuron_i", "session"),
    ("neuron_j", None),
)

CALCULATED_STATISTICS["union_distances"].relationship_axes = (
    ("neuron_i", None),
    ("neuron_j", None),
)

distance_keys = (
    "distances",
    "centroid_shift",
    "cross_session_distances",
    "footprint_union_distance",
    "union_distances",
)

for key in distance_keys:
    definition = CALCULATED_STATISTICS[key]
    definition.dependencies = TRACKED_STATISTIC_STRUCTURE | frozenset(
        {
            ChangeKind.FOOTPRINT_GEOMETRY,
            ChangeKind.UNION_GEOMETRY,
            ChangeKind.INCLUSION,
            ChangeKind.PROCESSING_STATUS,
            ChangeKind.SESSION_METADATA,
        }
    )
    definition.availability_dependencies = frozenset({"spatial"})

CALCULATED_STATISTICS["cooccurrence"] = StatisticDefinition(
    key="cooccurrence",
    title="Footprint co-occurrence",
    description=(
        "One when two neurons have distinct, included footprints "
        "in the same session; otherwise zero. "
        "Summing over sessions counts conflicting sessions."
    ),
    category="pair",
    dims=("neuron_i", "neuron_j", "session"),
    getter=query_only_getter,
    query_getter=cooccurrence_query,
    default_reductions={
        "neuron_i": ReductionSpec("keep"),
        "neuron_j": ReductionSpec("keep"),
        "session": ReductionSpec("sum"),
    },
    dependencies=(
        TRACKED_STATISTIC_STRUCTURE
        | frozenset(
            {
                ChangeKind.INCLUSION,
                ChangeKind.SESSION_METADATA,
            }
        )
    ),
    availability_dependencies=frozenset({"spatial"}),
    independent_pairs=True,
    compact_values=False,
    component_axes=(),
    relationship_axes=(
        ("neuron_i", None),
        ("neuron_j", None),
    ),
)
