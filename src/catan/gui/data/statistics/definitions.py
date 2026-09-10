from .types import StatisticDefinition
from .queries import ReductionSpec
from . import calculations

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
        title="Border proximity",
        description="Proximity of neuron centroids to the borders of the field of view.",
        dims=("neuron", "session"),
        category="neuron",
        getter=calculations.calculate_border_proximity,
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
        allowed_reductions={
            "neuron_i": ("keep", "mean"),
            "neuron_j": ("keep", "mean"),
            "session_i": ("single", "mean", "median", "max", "min"),
            "session_j": ("single", "mean", "median", "max", "min"),
        },
        default_reductions={
            "neuron_i": ReductionSpec("keep"),
            "neuron_j": ReductionSpec("keep"),
            "session_i": ReductionSpec("single", 0),
            "session_j": ReductionSpec("single", 1),
        },
    ),
}
