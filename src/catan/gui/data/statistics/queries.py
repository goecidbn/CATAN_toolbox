from typing import Literal, Optional
from dataclasses import dataclass

from .dimensions import canonical_dim, SESSION_DIMS, NEURON_DIMS

Contexts = Literal["generic", "session_series", "neuron_bound"]

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


PairRelation = Literal["all", "same", "different", "with previous"]
PairTarget = Literal["neuron", "session"]


@dataclass(frozen=True, slots=True)
class PairFilter:
    target: PairTarget
    relation: PairRelation
    collapse_same: bool = True


@dataclass(frozen=True, slots=True)
class ReductionSpec:
    ## defines reduction for each dimension
    method: ReductionMethod
    index: Optional[int] = None
    error_method: ErrorMethod = "none"


@dataclass(frozen=True, slots=True)
class StatisticQuery:
    statistic_key: str
    reductions: tuple[tuple[str, ReductionSpec], ...]
    reduction_order: tuple[str, ...] = ()
    filters: tuple[PairFilter, ...] = ()
    context: Contexts = "generic"

    def reduction_dict(self):
        return dict(self.reductions)


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


def normalize_session_series_reductions(
    stat_def,
    reductions: dict[str, ReductionSpec],
    filters: tuple[PairFilter, ...] = (),
) -> dict[str, ReductionSpec]:

    out = dict(reductions)

    session_dims = [d for d in stat_def.dims if d in SESSION_DIMS]
    neuron_dims = [d for d in stat_def.dims if d in NEURON_DIMS]

    session_relation = next(
        (f.relation for f in filters if f.target == "session"),
        "all",
    )

    # ----- session dimensions -----

    if len(session_dims) == 1:
        # This IS the x-axis of a session series.
        out[session_dims[0]] = ReductionSpec("keep")

    elif len(session_dims) == 2:
        dim_i = "session_i" if "session_i" in session_dims else session_dims[0]
        dim_j = "session_j" if "session_j" in session_dims else session_dims[1]

        if session_relation in ("same", "with previous"):
            # Both are needed until the pair filter is applied.
            # Afterwards they collapse into one session dimension.
            out[dim_i] = ReductionSpec("keep")
            out[dim_j] = ReductionSpec("keep")

        else:
            # session_i is the series axis;
            # session_j is one selected comparison session.
            out[dim_i] = ReductionSpec("keep")

            old = out.get(
                dim_j,
                ReductionSpec("single", index=0),
            )

            out[dim_j] = ReductionSpec(
                "single",
                index=old.index or 0,
            )

    # ----- neuron dimensions -----

    # Only ONE neuron reduction may create errors.
    error_dim = neuron_dims[-1] if neuron_dims else None

    for dim in neuron_dims:
        old = out.get(
            dim,
            ReductionSpec("median"),
        )

        method = old.method if old.method in ("mean", "median") else "median"

        if dim != error_dim:
            error_method = "none"

        else:
            allowed_errors = allowed_error_methods(
                dim,
                method,
                context="session_series",
            )

            if old.error_method in allowed_errors:
                error_method = old.error_method
            else:
                error_method = "iqr" if method == "median" else "sem"

        out[dim] = ReductionSpec(
            method,
            error_method=error_method,
        )

    # ----- other dimensions -----

    for dim in stat_def.dims:
        if dim in session_dims or dim in neuron_dims:
            continue

        old = out.get(
            dim,
            ReductionSpec("single", index=0),
        )

        allowed = allowed_reduction_methods(
            dim,
            context="session_series",
        )

        if old.method in allowed:
            method = old.method
        elif "single" in allowed:
            method = "single"
        else:
            method = allowed[0]

        out[dim] = ReductionSpec(
            method,
            index=(
                old.index
                if method == "single" and old.index is not None
                else 0 if method == "single" else None
            ),
            error_method="none",
        )

    validate_error_reductions(out)

    return out


def normalize_generic_session_pair_reductions(
    stat_def,
    reductions,
    filters,
):
    out = dict(reductions)

    if not ("session_i" in stat_def.dims and "session_j" in stat_def.dims):
        return out

    relation = next(
        (f.relation for f in filters if f.target == "session"),
        "all",
    )

    if relation == "same":
        i_spec = out.get(
            "session_i",
            ReductionSpec("single", index=0),
        )

        index = i_spec.index if i_spec.index is not None else 0

        out["session_i"] = ReductionSpec(
            "single",
            index=index,
        )
        out["session_j"] = ReductionSpec(
            "single",
            index=index,
        )

    elif relation == "with previous":
        i_spec = out.get(
            "session_i",
            ReductionSpec("single", index=1),
        )

        index = i_spec.index if i_spec.index is not None else 1

        index = max(1, index)

        out["session_i"] = ReductionSpec(
            "single",
            index=index,
        )
        out["session_j"] = ReductionSpec(
            "single",
            index=index - 1,
        )

    elif relation == "different":
        for dim in (
            "session_i",
            "session_j",
        ):
            old = out.get(
                dim,
                ReductionSpec("single", index=0),
            )

            out[dim] = ReductionSpec(
                "single",
                index=old.index or 0,
            )

    return out
