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
