from typing import Literal, Optional
from dataclasses import dataclass

from .dimensions import canonical_dim

Contexts = Literal["generic", "session_series"]

ReductionMethod = Literal[
    "keep",
    "single",
    "mean",
    "median",
    "max",
    "min",
    "sum",
    "std",
]

ErrorMethod = Literal[
    "none",
    "std",
    "sem",
    "iqr",
    "bootstrap",
]

DEFAULT_REDUCTIONS = ("keep", "single", "mean", "median", "max", "min", "sum", "std")

REDUCTION_METHODS = {
    "generic": {
        "session": DEFAULT_REDUCTIONS,
        "neuron": DEFAULT_REDUCTIONS,
        "default": DEFAULT_REDUCTIONS,
    },
    "session_series": {
        "session": ("keep", "single"),
        "neuron": ("mean", "median"),
        "default": ("single", "mean", "median", "max", "min"),
    },
}

ERROR_METHODS = {
    "generic": {
        "session": {
            "mean": ("none", "std", "sem", "iqr", "bootstrap"),
            "median": ("none", "iqr", "bootstrap"),
            "max": ("none", "bootstrap"),
            "min": ("none", "bootstrap"),
            "sum": ("none", "bootstrap"),
            "std": ("none", "bootstrap"),
        },
        "neuron": {
            "mean": ("none", "std", "sem", "iqr", "bootstrap"),
            "median": ("none", "iqr", "bootstrap"),
            "max": ("none", "bootstrap"),
            "min": ("none", "bootstrap"),
            "sum": ("none", "bootstrap"),
            "std": ("none", "bootstrap"),
        },
        "default": {
            "mean": ("none", "std", "sem", "iqr", "bootstrap"),
            "median": ("none", "iqr", "bootstrap"),
            "max": ("none", "bootstrap"),
            "min": ("none", "bootstrap"),
            "sum": ("none", "bootstrap"),
            "std": ("none", "bootstrap"),
        },
    },
    "session_series": {
        "session": {},
        "neuron": {
            "mean": ("std", "sem", "bootstrap"),
            "median": ("iqr", "bootstrap"),
        },
        "default": {
            "mean": ("none",),
            "median": ("none",),
            "max": ("none",),
            "min": ("none",),
            "sum": ("none",),
            "std": ("none",),
        },
    },
}


@dataclass(frozen=True, slots=True)
class ReductionSpec:
    ## defines reduction for each dimension
    method: ReductionMethod
    index: Optional[int] = None
    error_method: ErrorMethod = "none"


def allowed_reduction_methods(
    dim_name: str,
    *,
    context: Contexts = "generic",
) -> tuple[str, ...]:
    """
    gets allowed reduction methods for a given dimension name and context
    """
    kind = canonical_dim(dim_name)

    context_methods = REDUCTION_METHODS.get(
        context,
        REDUCTION_METHODS["generic"],
    )

    return context_methods.get(
        kind,
        context_methods.get("default", DEFAULT_REDUCTIONS),
    )


def allowed_error_methods(
    dim_name: str,
    reduction_method: str,
    *,
    context: Contexts = "generic",
) -> tuple[str, ...]:
    if reduction_method in ("keep", "single"):
        return ("none",)

    kind = canonical_dim(dim_name)

    context_methods = ERROR_METHODS.get(
        context,
        ERROR_METHODS["generic"],
    )

    dim_methods = context_methods.get(
        kind,
        context_methods.get("default", {}),
    )

    return dim_methods.get(reduction_method, ("none",))


def validate_error_reductions(reductions: dict[str, ReductionSpec]):
    error_dims = [
        dim for dim, spec in reductions.items() if spec.error_method != "none"
    ]

    if len(error_dims) > 1:
        raise ValueError(
            "Only one error-producing reduction is currently supported. "
            f"Got error reductions on {error_dims}."
        )


PairRelation = Literal["all", "same", "different", "with previous"]
PairTarget = Literal["neuron", "session"]


@dataclass(frozen=True, slots=True)
class PairFilter:
    target: PairTarget
    relation: PairRelation
    collapse_same: bool = True


@dataclass(frozen=True, slots=True)
class StatisticQuery:
    statistic_key: str
    reductions: tuple[tuple[str, ReductionSpec], ...]
    reduction_order: tuple[str, ...] = ()
    filters: tuple[PairFilter, ...] = ()
    context: Contexts = "generic"

    # @staticmethod
    # def from_dict(statistic_key: str, reductions: dict[str, ReductionSpec]):
    #     return StatisticQuery(
    #         statistic_key=statistic_key,
    #         reductions=tuple(sorted(reductions.items())),
    #     )

    def reduction_dict(self):
        return dict(self.reductions)
