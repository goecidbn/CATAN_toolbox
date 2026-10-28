"""Threshold bounds stored as a calculation recipe instead of a number.

- PercentileReference: a percentile of a suggestion sample of the
  condition's statistic (engine.sample_values; e.g. "Same neuron").
- ModelDistanceReference: a linear combination of the fitted two-noise
  model's distance parameters, converted to px (distance statistics only).

Presets store the recipe; the value is resolved whenever the curation
filter is evaluated, so it follows the current data and model.
"""

from dataclasses import dataclass

import numpy as np

# Model parameters a distance threshold may combine (model units).
MODEL_DISTANCE_TERMS = {
    "h": "h",
    "sigma_det": "σ_det",
    "sigma_move": "σ_move",
    "sigma_h": "σ_h",
}

# Reductions that leave values in the statistic's unit (no "sum").
UNIT_PRESERVING_REDUCTIONS = ("keep", "single", "mean", "median", "max", "min", "std")


class ThresholdReferenceError(ValueError):
    pass


@dataclass(frozen=True)
class PercentileReference:
    sample: str
    level: float

    def describe(self):
        level = "median" if self.level == 50 else f"p{self.level:g}"
        return f"{level} of {self.sample.lower()}"

    def to_dict(self):
        return {"kind": "percentile", "sample": self.sample, "level": float(self.level)}


@dataclass(frozen=True)
class ModelDistanceReference:
    # ((parameter, coefficient), ...) in MODEL_DISTANCE_TERMS order.
    coefficients: tuple[tuple[str, float], ...]

    @classmethod
    def from_mapping(cls, coefficients):
        unknown = set(coefficients) - set(MODEL_DISTANCE_TERMS)
        if unknown:
            raise ThresholdReferenceError(
                f"Unknown model distance terms: {', '.join(sorted(unknown))}"
            )
        return cls(
            tuple(
                (name, float(coefficients[name]))
                for name in MODEL_DISTANCE_TERMS
                if coefficients.get(name, 0.0) != 0.0
            )
        )

    def describe(self):
        if not self.coefficients:
            return "0"

        parts = []
        for index, (name, coefficient) in enumerate(self.coefficients):
            factor = "" if abs(coefficient) == 1 else f"{abs(coefficient):g}·"
            sign = ("−" if coefficient < 0 else "") if index == 0 else (
                "− " if coefficient < 0 else "+ "
            )
            parts.append(f"{sign}{factor}{MODEL_DISTANCE_TERMS[name]}")
        return " ".join(parts)

    def resolve(self, data):
        """The combination in px for the current model of ``data``."""
        model = getattr(data, "model", None)

        if model is None or not model.fitted or model.kind != "two_noise":
            raise ThresholdReferenceError(
                f"{self.describe()}: needs a fitted two-noise model."
            )

        value = sum(
            coefficient * float(model.parameters[name])
            for name, coefficient in self.coefficients
        )

        # Model distances are in session units (µm with pxtomu µm/px).
        scale = float(data.params.get("pxtomu", 1.0))
        return value / scale

    def to_dict(self):
        return {"kind": "model_distance", "coefficients": dict(self.coefficients)}


def reference_from_dict(data):
    if data is None:
        return None

    kind = data.get("kind")

    if kind == "percentile":
        return PercentileReference(str(data["sample"]), float(data["level"]))

    if kind == "model_distance":
        return ModelDistanceReference.from_mapping(data["coefficients"])

    raise ValueError(f"Unknown threshold reference kind: {kind!r}")


def is_distance_query(definition, query):
    """Values of the query are distances in px (see
    StatisticDefinition.value_kind) and stay so after its reductions."""
    if definition is None or definition.value_kind != "distance":
        return False

    return all(
        spec.method in UNIT_PRESERVING_REDUCTIONS
        for _, spec in query.reductions
    )


def resolve_percentile(reference, engine, query):
    from .statistics.sampling import SUGGESTION_SAMPLES

    sample = engine.sample_values(query, SUGGESTION_SAMPLES)

    if sample is not None and sample.method == "unsupported":
        sample = engine.sample_values(query, SUGGESTION_SAMPLES, allow_full=True)

    if sample is None:
        raise ThresholdReferenceError("No data loaded.")

    for values in sample.samples:
        if values.label == reference.sample:
            if not len(values.values):
                break
            return float(
                np.percentile(np.asarray(values.values, dtype=np.float64), reference.level)
            )

    raise ThresholdReferenceError(
        f"{reference.describe()}: no such values for this statistic."
    )


def resolve_reference(reference, engine, query):
    if isinstance(reference, PercentileReference):
        return resolve_percentile(reference, engine, query)

    definition = engine.registry.get(query.statistic_key)

    if not is_distance_query(definition, query):
        raise ThresholdReferenceError(
            "Model distance thresholds need a distance statistic "
            "without sum reductions."
        )

    return reference.resolve(engine.data)
