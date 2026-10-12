from typing import Any, Literal, Optional
from dataclasses import dataclass
import numpy as np

DimensionMode = Literal["remaining", "fixed", "reduced"]

SESSION_DIMS = ("session", "session_i", "session_j")
NEURON_DIMS = ("neuron", "neuron_i", "neuron_j")


def canonical_dim(dim_name: str) -> str:
    if dim_name in SESSION_DIMS:
        return "session"

    if dim_name in NEURON_DIMS:
        return "neuron"

    return "default"


def canonical_dims(dims: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(canonical_dim(dim) for dim in dims)


@dataclass
class Dimension:

    name: str
    coords: np.ndarray  # coordinates along axis

    # For fixed: selected coordinate value
    # For reduced: reduction method, e.g. "mean", "max"
    parameter: Any = None

    mode: DimensionMode = "remaining"

    def is_remaining(self) -> bool:
        return self.mode == "remaining"

    def is_fixed(self) -> bool:
        return self.mode == "fixed"

    def is_reduced(self) -> bool:
        return self.mode == "reduced"


@dataclass
class DimensionInfo:
    size: int
    labels: Optional[list[str]] = None
    coords: Optional[np.ndarray] = None  # optional coordinates along axis


def get_default_coords(state) -> dict[str, np.ndarray]:

    N, S = state.assignments.shape
    return {
        "session": np.arange(S),
        "session_i": np.arange(S),
        "session_j": np.arange(S),
        "neuron": np.arange(N),
        "neuron_i": np.arange(N),
        "neuron_j": np.arange(N),
    }


def get_match_stat_dims(
    name: str,
    values: np.ndarray,
) -> tuple[str, ...]:

    if values.ndim < 2:
        raise ValueError(
            f"Assignment statistic {name!r} "
            "must have at least neuron and session axes."
        )

    extra_dims = tuple(f"{name}_dim_{i}" for i in range(values.ndim - 2))

    return (
        "neuron",
        "session",
        *extra_dims,
    )


def make_match_coord_getter(
    dims: tuple[str, ...],
    shape: tuple[int, ...],
):
    def coord_getter(state):
        coords = get_default_coords(state)

        for dim, size in zip(
            dims[2:],
            shape[2:],
        ):
            coords[dim] = np.arange(size)

        return coords

    return coord_getter


def neuron_bound_dim(
    dims: tuple[str, ...],
) -> str | None:

    if "neuron" in dims:
        return "neuron"

    if "neuron_i" in dims:
        return "neuron_i"

    if "neuron_j" in dims:
        return "neuron_j"

    return None


def component_bound_dims(
    dims: tuple[str, ...],
) -> tuple[str, str | None] | None:
    """
    Return the dimensions identifying one session-specific neuron/component.

    Preference:
        ("neuron", "session")
        ("neuron_i", "session_i")
        ("neuron_j", "session_j")

    The session dimension may be None if the statistic has a neuron
    dimension but no session dimension.
    """

    if "neuron" in dims:
        session_dim = "session" if "session" in dims else None

        return "neuron", session_dim

    if "neuron_i" in dims:
        session_dim = "session_i" if "session_i" in dims else None

        return "neuron_i", session_dim

    if "neuron_j" in dims:
        session_dim = "session_j" if "session_j" in dims else None

        return "neuron_j", session_dim

    return None


def neuron_pair_bound_dims(
    dims: tuple[str, ...],
) -> tuple[str, str] | None:
    """
    Return the two neuron dimensions identifying a neuron pair.

    Pair-bound output requires explicit neuron_i / neuron_j dimensions.
    """
    if "neuron_i" in dims and "neuron_j" in dims:
        return "neuron_i", "neuron_j"

    return None


def component_pair_bound_dims(
    dims: tuple[str, ...],
) -> tuple[tuple[str, str], tuple[str, ...]] | None:
    """
    Return neuron-pair dimensions plus any session dimensions
    describing the concrete component pair.

    Session dimensions are optional: a neuron-pair statistic without
    session dimensions is still usable in footprint-pair mode and will
    simply have the same value for all component pairs of those neurons.
    """
    neuron_dims = neuron_pair_bound_dims(dims)

    if neuron_dims is None:
        return None

    session_dims = tuple(
        dim for dim in ("session", "session_i", "session_j") if dim in dims
    )

    return neuron_dims, session_dims
