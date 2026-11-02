from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral, Real
import math


@dataclass(frozen=True, slots=True)
class StatisticParameter:
    key: str
    label: str
    kind: type
    default: float | int | bool | str
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float = 0.1
    decimals: int = 3
    tooltip: str = ""
    choices: Callable | None = None
    # Short name for compact displays ("thr=0.1"); the label otherwise.
    short: str = ""
    # False for options that do not change the values (loading, saving):
    # left out of titles.
    in_title: bool = True

    @property
    def short_label(self) -> str:
        return self.short or self.label

    def format_value(self, value) -> str:
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:g}"
        return str(value)

    def help_text(self) -> str:
        """Tooltip: label, short name, default and description."""
        name = self.label
        if self.short and self.short != self.label:
            name += f" ({self.short})"
        text = f"{name}, default {self.format_value(self.default)}"
        if self.tooltip:
            text += f"\n\n{self.tooltip}"
        return text

    def validate(self, value):

        if self.kind is str:
            if not isinstance(value, str):
                raise ValueError(f"{self.label} must be a string.")
            return value

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
