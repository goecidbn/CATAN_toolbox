from functools import partial

from .types import StatisticDefinition
from .queries import ReductionSpec
from .parameters import StatisticParameter
from .distance_statistics import distance_query
from .overlap_statistics import (
    migrate_overlap_parameters,
    normalize_by_choices,
    overlap_query,
)
from .cooccurrence_statistics import cooccurrence_query, cooccurrence_requested_pairs
from .footprint_pairs import footprint_similarity_requested_pairs
from .union_similarity import (
    calculate_footprint_union_similarity,
    calculate_partner_session_match,
    partner_session_match_requested_pairs,
    calculate_union_similarity,
    footprint_union_similarity_requested_pairs,
    union_similarity_requested_pairs,
)
from .event_rate import calculate_event_rate, trace_keys
from .assignment_difference import (
    calculate_assignment_difference,
    other_assignment_names,
)
from .match_statistics import (
    calculate_match_probability,
    match_probability_requested_pairs,
)
from .track_split import calculate_track_split, measure_choices
from .union_overlap import (
    calculate_footprint_union_overlap,
    footprint_union_overlap_requested_pairs,
)
from . import calculations

from catan.core.changes import (
    ChangeKind,
    TRACKED_STATISTIC_STRUCTURE,
)

def FOOTPRINT_THRESHOLD(default):
    """Pixels counted for a footprint: above this fraction of its peak."""
    return StatisticParameter(
        key="thr",
        short="thr",
        label="Pixel threshold",
        kind=float,
        default=default,
        minimum=0.0,
        maximum=0.99,
        step=0.01,
        decimals=3,
        tooltip="Pixels above this fraction of the footprint's peak are counted.",
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
        description="Area of the neuron footprint (pixels above threshold).",
        dims=("neuron", "session"),
        category="neuron",
        getter=calculations.calculate_footprint_size,
        parameters=(FOOTPRINT_THRESHOLD(0.01),),
    ),
    "relative_footprint_size": StatisticDefinition(
        key="relative_footprint_size",
        title="Relative footprint size",
        description=(
            "Footprint size relative to the median size of the neuron's "
            "footprints (1 = typical for this neuron)."
        ),
        dims=("neuron", "session"),
        category="neuron",
        getter=calculations.calculate_relative_footprint_size,
        parameters=(FOOTPRINT_THRESHOLD(0.01),),
    ),
    "footprint_eccentricity": StatisticDefinition(
        key="footprint_eccentricity",
        title="Footprint elongation",
        description=(
            "Shape of the footprint pixels above a threshold, from their "
            "extent along the longest and the perpendicular direction: "
            "axis ratio (1 = round) or eccentricity (0 = round, 1 = line)."
        ),
        dims=("neuron", "session"),
        category="neuron",
        getter=calculations.calculate_footprint_eccentricity,
        parameters=(
            FOOTPRINT_THRESHOLD(0.2),
            StatisticParameter(
                key="measure",
                short="measure",
                label="Measure",
                kind=str,
                default="axis ratio",
                tooltip=(
                    "axis ratio: longest / perpendicular extent (≥ 1). "
                    "eccentricity: sqrt(1 − minor² / major²), 0 round … 1 line."
                ),
                choices=calculations.shape_measure_choices,
            ),
        ),
    ),
    "footprint_lobes": StatisticDefinition(
        key="footprint_lobes",
        title="Footprint lobes",
        description=(
            "Number of lobes of the footprint: bright parts that separate "
            "when it is cut at some threshold, e.g. two cells detected as "
            "one footprint. A lobe reaches the lobe peak and covers at "
            "least the lobe size of the footprint's area; 1 = one compact "
            "part."
        ),
        dims=("neuron", "session"),
        category="neuron",
        getter=calculations.calculate_footprint_lobes,
        parameters=(
            FOOTPRINT_THRESHOLD(0.1),
            StatisticParameter(
                key="peak",
                short="peak",
                label="Lobe peak",
                kind=float,
                default=0.5,
                minimum=0.05,
                maximum=1.0,
                step=0.05,
                decimals=2,
                tooltip="A lobe's own peak reaches this fraction of the footprint's peak.",
            ),
            StatisticParameter(
                key="size",
                short="size",
                label="Lobe size",
                kind=float,
                default=0.1,
                minimum=0.0,
                maximum=0.5,
                step=0.01,
                decimals=2,
                tooltip=(
                    "A lobe covers at least this fraction of the footprint's "
                    "pixels above the pixel threshold (and at least 10 pixels)."
                ),
            ),
        ),
    ),
    "border_proximity": StatisticDefinition(
        key="border_proximity",
        title="Border distance",
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
                short="shared border",
                label="Shared border",
                kind=bool,
                default=False,
                tooltip=(
                    "Use the area covered by all loaded sessions (all but the "
                    "number tolerated in the Overview option 'Border: tolerate "
                    "outside'). Otherwise use this session's own aligned "
                    "imaging area."
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
        foreground_safe=True,
    ),
    "assignment_difference": StatisticDefinition(
        key="assignment_difference",
        title="Assignment differences",
        description=(
            "Sessions in which a neuron's footprints differ from its best "
            "match in another assignment set (most shared footprints, then "
            "nearest union centroid)."
        ),
        dims=("neuron",),
        category="neuron",
        getter=calculate_assignment_difference,
        parameters=(
            StatisticParameter(
                key="compare_with",
                short="vs",
                label="Compare with",
                kind=str,
                default="",
                tooltip="Loaded assignment set compared with the current one.",
                choices=other_assignment_names,
            ),
            StatisticParameter(
                key="max_distance",
                short="d_max",
                label="Match distance (px)",
                kind=float,
                default=10.0,
                minimum=0.0,
                step=1.0,
                decimals=1,
                tooltip=(
                    "Neurons sharing no footprint match the nearest union "
                    "centroid within this distance; otherwise all their "
                    "footprints count as differences."
                ),
            ),
        ),
        foreground_safe=True,
    ),
    "event_rate": StatisticDefinition(
        key="event_rate",
        title="Event rate",
        description=(
            "Rate of inferred events (e.g., spikes) for each neuron within sessions."
        ),
        dims=("neuron", "session"),
        category="neuron",
        getter=calculate_event_rate,
        foreground_safe=False,
        parameters=(
            StatisticParameter(
                key="key",
                short="trace",
                label="Trace Key",
                kind=str,
                default="S_dff",
                tooltip=("Key of the trace to use for event rate calculation."),
                choices=trace_keys,
            ),
            StatisticParameter(
                key="prctile",
                short="pct",
                label="Percentile",
                kind=float,
                default=20,
                minimum=0.0,
                maximum=100.0,
                step=1.0,
                decimals=0,
                tooltip=(
                    "Percentile of raw deconvolved event amplitude to consider as baseline. "
                    "Lower values potentially yield more spikes per events."
                ),
            ),
            StatisticParameter(
                key="frequency",
                short="f",
                label="Sampling frequency (Hz)",
                kind=float,
                default=15.0,
                minimum=0.001,
                step=1.0,
                decimals=3,
                tooltip="Sampling frequency of the selected traces.",
            ),
            StatisticParameter(
                key="load_all",
                in_title=False,
                label="Load all traces",
                kind=bool,
                default=False,
                tooltip=(
                    "Load trace data from all configured sessions before "
                    "calculating. Otherwise load only sessions needed "
                    "for missing results."
                ),
            ),
            StatisticParameter(
                key="save_results",
                in_title=False,
                label="Save results",
                kind=bool,
                default=True,
                tooltip=(
                    "Retain calculated rates after unloading traces and "
                    "include them when saving the CATAN session. "
                    "Existing saved results are reused either way."
                ),
            ),
        ),
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
                short="γ",
                label="Gamma",
                kind=float,
                default=0.1,
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                decimals=3,
                tooltip=(
                    "Similarity = cosine of the overlapping parts × "
                    "(support overlap / smaller support)^γ, maximised over "
                    "shifts. 0 ignores how much the supports overlap; larger "
                    "values penalize partial overlap more strongly."
                ),
            ),
        ),
        independent_pairs=True,
        compact_values=True,
        pairs_getter=footprint_similarity_requested_pairs,
    ),
}

# Preserve the existing key and dimensions for saved configurations.
definition = CALCULATED_STATISTICS["distances"]
definition.title = "Within-session distances"
definition.query_getter = partial(distance_query, mode="within")
definition.independent_pairs = True
definition.compact_values = True

definition = CALCULATED_STATISTICS["centroid_shift"]
definition.title = "Within-neuron displacement"
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
            title="Cross-session distances",
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
            title="Footprint → union distance",
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
            title="Union-centroid distances",
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


# Values are centroid distances in px (model distance thresholds).
for key in (
    "distances",
    "centroid_shift",
    "cross_session_distances",
    "footprint_union_distance",
    "union_distances",
):
    CALCULATED_STATISTICS[key].value_kind = "distance"

CALCULATED_STATISTICS["border_proximity"].value_kind = "distance"
CALCULATED_STATISTICS["footprint_size"].value_kind = "area"

definition = CALCULATED_STATISTICS["footprint_union_distance"]
definition.component_axes = (("neuron_i", "session"),)
definition.curation_kind = "reassignment"

definition = CALCULATED_STATISTICS["union_distances"]
definition.component_axes = ()
definition.curation_kind = "neuron_pair"

# Footprint counterpart of footprint_union_distance: compare with the union
# of a neuron instead of each of its stored footprints.
CALCULATED_STATISTICS["footprint_union_similarity"] = StatisticDefinition(
    key="footprint_union_similarity",
    title="Footprint → union similarity",
    description=(
        "Similarity of a session footprint of neuron_i to the union "
        "footprint of neuron_j (same score as footprint similarity). "
        "Pairs with centroids 10 px or more apart are omitted; the own "
        "union (neuron_i = neuron_j) is included."
    ),
    dims=("neuron_i", "neuron_j", "session"),
    category="pair",
    getter=calculate_footprint_union_similarity,
    pairs_getter=footprint_union_similarity_requested_pairs,
    allowed_reductions=None,
    default_reductions={
        "neuron_i": ReductionSpec("keep"),
        "neuron_j": ReductionSpec("keep"),
        "session": ReductionSpec("single", 0),
    },
    parameters=CALCULATED_STATISTICS["footprint_similarity"].parameters,
    independent_pairs=True,
    compact_values=True,
    component_axes=(("neuron_i", "session"),),
    curation_kind="reassignment",
    relationship_axes=(("neuron_i", "session"), ("neuron_j", None)),
)

# Union counterpart of union_distances (neuron pairs, e.g. merge candidates).
CALCULATED_STATISTICS["partner_session_match"] = StatisticDefinition(
    key="partner_session_match",
    title="Partner's own match",
    description=(
        "For a session footprint of neuron_i and the neuron neuron_j: how "
        "well neuron_j's own footprint in that session matches its union "
        "(footprint → union similarity); 0 if neuron_j has no footprint "
        "there. Low values: neuron_j has no good footprint in this session, "
        "so the footprint of neuron_i could be its missing one. Pairs with "
        "centroids 10 px or more apart are omitted."
    ),
    dims=("neuron_i", "neuron_j", "session"),
    category="pair",
    getter=calculate_partner_session_match,
    pairs_getter=partner_session_match_requested_pairs,
    allowed_reductions=None,
    default_reductions={
        "neuron_i": ReductionSpec("keep"),
        "neuron_j": ReductionSpec("keep"),
        "session": ReductionSpec("single", 0),
    },
    parameters=CALCULATED_STATISTICS["footprint_similarity"].parameters,
    independent_pairs=True,
    compact_values=True,
    component_axes=(("neuron_i", "session"),),
    curation_kind="reassignment",
    relationship_axes=(("neuron_i", "session"), ("neuron_j", None)),
)

CALCULATED_STATISTICS["union_similarity"] = StatisticDefinition(
    key="union_similarity",
    title="Union similarity",
    description=(
        "Similarity between the union footprints of two neurons (same score "
        "as footprint similarity). Pairs with union centroids 10 px or more "
        "apart and identical neurons are omitted."
    ),
    dims=("neuron_i", "neuron_j"),
    category="pair",
    getter=calculate_union_similarity,
    pairs_getter=union_similarity_requested_pairs,
    allowed_reductions=None,
    default_reductions={
        "neuron_i": ReductionSpec("keep"),
        "neuron_j": ReductionSpec("keep"),
    },
    parameters=CALCULATED_STATISTICS["footprint_similarity"].parameters,
    independent_pairs=True,
    compact_values=True,
    component_axes=(),
    curation_kind="neuron_pair",
    relationship_axes=(("neuron_i", None), ("neuron_j", None)),
)

# Neurons assembled from two cells: best two-group partition of the footprints.
CALCULATED_STATISTICS["track_split"] = StatisticDefinition(
    key="track_split",
    title="Track split",
    description=(
        "Best partition of a neuron's footprints into two groups by centroid "
        "position, maximizing the gap (smallest distance between the "
        "groups); any grouping, including a single cut in time. Values "
        "exist only for footprint pairs across the groups, from the group "
        "with the first detection (s_i) to the other (s_j)."
    ),
    dims=("neuron", "session_i", "session_j"),
    category="neuron",
    getter=calculate_track_split,
    default_reductions={
        "neuron": ReductionSpec("keep"),
        "session_i": ReductionSpec("keep"),
        "session_j": ReductionSpec("keep"),
    },
    parameters=(
        StatisticParameter(
            key="measure",
            short="measure",
            label="Measure",
            kind=str,
            default="gap",
            tooltip=(
                "gap: smallest distance between the groups (insensitive to "
                "gradual drift within a group). separation: distance "
                "between the group medians."
            ),
            choices=measure_choices,
        ),
        StatisticParameter(
            key="min_group",
            short="n_min",
            label="Minimum group size",
            kind=int,
            default=2,
            minimum=1,
            step=1,
            decimals=0,
            tooltip="Footprints required in each group.",
        ),
    ),
    relationship_axes=(("neuron", "session_i"), ("neuron", "session_j")),
)

CALCULATED_STATISTICS["footprint_overlap"] = StatisticDefinition(
    key="footprint_overlap",
    title="Footprint overlap",
    description=(
        "Overlap of registered footprints without additional shifting: "
        "shared support pixels divided by the union (IoU), the smaller "
        "footprint, footprint i or footprint j. Between 0 and 1."
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
            key="normalize_by",
            short="norm",
            label="Normalize by",
            kind=str,
            default="smaller",
            tooltip=(
                "Denominator of the shared support pixels. IoU: union of both "
                "supports. smaller: the smaller support (symmetric containment). "
                "i: footprint i — fraction of i covered by j. "
                "j: footprint j (or union j) — fraction of j covered by i."
            ),
            choices=normalize_by_choices,
        ),
        StatisticParameter(
            key="support_threshold",
            short="thr",
            label="Relative support threshold",
            kind=float,
            default=0.1,
            minimum=0.0,
            maximum=0.99,
            step=0.05,
            decimals=3,
            tooltip=(
                "Support of a footprint: pixels at or above this fraction of "
                "its own maximum. 0.1 means 10% of its peak."
            ),
        ),
    ),
    migrate_parameters=migrate_overlap_parameters,
)

_CALCULATED_DEPENDENCIES = {
    "footprint_size": (
        frozenset({ChangeKind.FOOTPRINT_GEOMETRY}),
        frozenset({"spatial"}),
    ),
    "footprint_eccentricity": (
        frozenset({ChangeKind.FOOTPRINT_GEOMETRY}),
        frozenset({"spatial"}),
    ),
    "footprint_lobes": (
        frozenset({ChangeKind.FOOTPRINT_GEOMETRY}),
        frozenset({"spatial"}),
    ),
    "relative_footprint_size": (
        frozenset({ChangeKind.FOOTPRINT_GEOMETRY, ChangeKind.INCLUSION}),
        frozenset({"spatial"}),
    ),
    "border_proximity": (
        frozenset(
            {
                ChangeKind.FOOTPRINT_GEOMETRY,
                ChangeKind.BACKGROUND_IMAGE,
                ChangeKind.PROCESSING_STATUS,
                ChangeKind.SESSION_METADATA,
                ChangeKind.BORDER_SETTINGS,
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

# Footprint-derived values without persisted side effects; threshold
# suggestions may sample them on a subset of neurons.
for key in (
    "footprint_size",
    "footprint_eccentricity",
    "footprint_lobes",
    "border_proximity",
    "centroid_shift",
    "footprint_similarity",
    "footprint_overlap",
    "cross_session_distances",
    "track_split",
):
    CALCULATED_STATISTICS[key].neuron_local = True

distance_keys = (
    "distances",
    "centroid_shift",
    "cross_session_distances",
    "footprint_union_distance",
    "union_distances",
    # Depend on the same footprint/union geometry and inclusion.
    "footprint_union_similarity",
    "partner_session_match",
    "union_similarity",
    "track_split",
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

# Overlap counterpart of footprint_union_similarity (one union per neuron).
CALCULATED_STATISTICS["footprint_union_overlap"] = StatisticDefinition(
    key="footprint_union_overlap",
    title="Footprint → union overlap",
    description=(
        "Overlap of a session footprint of neuron_i with the union footprint "
        "of neuron_j (same measure as footprint overlap). Pairs with "
        "centroids 10 px or more apart are omitted; the own union "
        "(neuron_i = neuron_j) is included."
    ),
    dims=("neuron_i", "neuron_j", "session"),
    category="pair",
    getter=calculate_footprint_union_overlap,
    pairs_getter=footprint_union_overlap_requested_pairs,
    allowed_reductions=None,
    default_reductions={
        "neuron_i": ReductionSpec("keep"),
        "neuron_j": ReductionSpec("keep"),
        "session": ReductionSpec("single", 0),
    },
    parameters=CALCULATED_STATISTICS["footprint_overlap"].parameters,
    migrate_parameters=migrate_overlap_parameters,
    independent_pairs=True,
    compact_values=True,
    component_axes=(("neuron_i", "session"),),
    curation_kind="reassignment",
    relationship_axes=(("neuron_i", "session"), ("neuron_j", None)),
    # Same footprint/union geometry and inclusion as the similarity.
    dependencies=CALCULATED_STATISTICS["footprint_union_similarity"].dependencies,
    availability_dependencies=CALCULATED_STATISTICS[
        "footprint_union_similarity"
    ].availability_dependencies,
)

# Registration probability of a footprint for its own neuron's union.
CALCULATED_STATISTICS["match_probability"] = StatisticDefinition(
    key="match_probability",
    title="Match probability",
    description=(
        "Probability with which a session footprint of neuron_i was matched "
        "to the union of its neuron during registration (only "
        "neuron_i = neuron_j). Not available for manually assigned or "
        "moved footprints."
    ),
    dims=("neuron_i", "neuron_j", "session"),
    category="pair",
    getter=calculate_match_probability,
    pairs_getter=match_probability_requested_pairs,
    allowed_reductions=None,
    default_reductions={
        "neuron_i": ReductionSpec("keep"),
        "neuron_j": ReductionSpec("keep"),
        "session": ReductionSpec("single", 0),
    },
    independent_pairs=True,
    component_axes=(("neuron_i", "session"),),
    curation_kind="reassignment",
    relationship_axes=(("neuron_i", "session"), ("neuron_j", None)),
    dependencies=TRACKED_STATISTIC_STRUCTURE | frozenset({ChangeKind.MATCH_VALUES}),
    foreground_safe=True,
    neuron_local=True,
    group="footprint_neuron",
)

# Track statistics (a neuron's footprints across sessions) belong to neurons.
CALCULATED_STATISTICS["centroid_shift"].group = "neuron"
CALCULATED_STATISTICS["track_split"].group = "neuron"
CALCULATED_STATISTICS["track_split"].value_kind = "distance"

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
    pairs_getter=cooccurrence_requested_pairs,
    requested_reduced_axes=("session",),
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

# Possible values of counting statistics (integer-centred histogram range):
# presence per session is 0 or 1 (a session sum: 0 … active sessions),
# assignment differences count sessions.
CALCULATED_STATISTICS["occurence"].value_bounds = (0, 1)
CALCULATED_STATISTICS["cooccurrence"].value_bounds = (0, 1)
CALCULATED_STATISTICS["footprint_lobes"].value_bounds = (0, None)
CALCULATED_STATISTICS["assignment_difference"].value_bounds = lambda n_sessions: (
    0,
    n_sessions,
)

# Overlap of the seeds a session exports for seeded redetection (detected
# and inferred footprints).
from .seed_overlap import SEED_KINDS, calculate_seed_overlap  # noqa: E402

CALCULATED_STATISTICS["seed_overlap"] = StatisticDefinition(
    key="seed_overlap",
    title="Seed overlap",
    description=(
        "Overlap of two neurons' redetection seeds in a session: their "
        "included footprints or, where a neuron has none, its union "
        "footprint at the interpolated position (inferred footprint). Same "
        "measure as footprint overlap. Pairs with union centroids 10 px or "
        "more apart are omitted; each pair once (neuron_i < neuron_j)."
    ),
    dims=("neuron_i", "neuron_j", "session"),
    category="pair",
    getter=calculate_seed_overlap,
    allowed_reductions=None,
    default_reductions={
        "neuron_i": ReductionSpec("keep"),
        "neuron_j": ReductionSpec("keep"),
        "session": ReductionSpec("keep"),
    },
    parameters=(
        *CALCULATED_STATISTICS["footprint_overlap"].parameters,
        StatisticParameter(
            key="seeds",
            short="seeds",
            label="Seed pairs",
            kind=str,
            default="all",
            tooltip=(
                "all: any two seeds. inferred: at least one of the two is an "
                "inferred footprint (detected pairs: footprint overlap)."
            ),
            choices=lambda data=None: SEED_KINDS,
        ),
    ),
    independent_pairs=True,
    compact_values=True,
    relationship_axes=(("neuron_i", "session"), ("neuron_j", "session")),
    dependencies=CALCULATED_STATISTICS["footprint_union_similarity"].dependencies,
    availability_dependencies=CALCULATED_STATISTICS[
        "footprint_union_similarity"
    ].availability_dependencies,
)
