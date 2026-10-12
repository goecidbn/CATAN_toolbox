from dataclasses import dataclass
from numbers import Integral, Real
import math


@dataclass(frozen=True, slots=True)
class StatisticParameter:
    key: str
    label: str
    kind: type
    default: float | int | bool
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float = 0.1
    decimals: int = 3
    tooltip: str = ""

    def validate(self, value):
        if self.kind is bool:
            if type(value) is not bool:
                raise ValueError(f"{self.label} must be true or false.")
            return value

        if self.kind not in (int, float):
            raise TypeError(f"Unsupported parameter type: {self.kind}")

        expected = Integral if self.kind is int else Real

        if isinstance(value, bool) or not isinstance(value, expected):
            raise ValueError(f"{self.label} must be a {self.kind.__name__}.")

        value = self.kind(value)

        if not math.isfinite(value):
            raise ValueError(f"{self.label} must be finite.")

        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"{self.label} must be at least {self.minimum}.")

        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"{self.label} must be at most {self.maximum}.")

        return value
