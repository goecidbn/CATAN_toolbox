from typing import Any, Callable, Optional

from dataclasses import dataclass
import numpy as np

from .dimensions import (
    Dimension,
    DimensionInfo,
    get_default_coords,
)
from .queries import ReductionSpec, DEFAULT_REDUCTIONS
from catan.gui.data import analysis


@dataclass
class StatisticArray:
    name: str
    values: np.ndarray
    dimensions: dict[str, Dimension]

    errors_low: np.ndarray | None = None
    errors_high: np.ndarray | None = None
    n: np.ndarray | None = None

    @property
    def dims(self) -> tuple[str, ...]:
        return tuple(
            name for name, dim in self.dimensions.items() if dim.mode == "remaining"
        )

    @property
    def ndim(self) -> int:
        return len(self.dims)

    @property
    def has_errors(self) -> bool:
        return self.errors_low is not None and self.errors_high is not None

    def axis(self, dim_name: str) -> int:
        if dim_name not in self.dimensions:
            raise KeyError(f"Unknown dimension {dim_name!r}")

        if self.dimensions[dim_name].mode != "remaining":
            raise ValueError(
                f"Dimension {dim_name!r} is not remaining "
                f"but {self.dimensions[dim_name].mode!r}"
            )

        return self.dims.index(dim_name)

    def validate(self):
        values = np.asarray(self.values)

        if values.ndim != self.ndim:
            raise ValueError(
                f"values.ndim={values.ndim}, but remaining dims are "
                f"{self.dims} with ndim={self.ndim}"
            )

        for axis, dim_name in enumerate(self.dims):
            dim = self.dimensions[dim_name]

            if dim.coords is None:
                raise RuntimeError(
                    f"Remaining dimension {dim.name!r} has no coordinates."
                )

            if len(dim.coords) != values.shape[axis]:
                raise ValueError(
                    f"Dimension {dim_name!r}: len(coords)={len(dim.coords)}, "
                    f"but axis length={values.shape[axis]}"
                )

        for dim_name, dim in self.dimensions.items():
            if dim.mode in ("fixed", "reduced") and dim.parameter is None:
                raise RuntimeError(
                    f"{dim.mode} dimension {dim.name!r} has no parameter."
                )

        if self.errors_low is not None and self.errors_low.shape != values.shape:
            raise ValueError("errors_low must match values.shape")

        if self.errors_high is not None and self.errors_high.shape != values.shape:
            raise ValueError("errors_high must match values.shape")

        if self.n is not None and self.n.shape != values.shape:
            raise ValueError("n must match values.shape")

    def select_dimension(self, dim_name: str, index: int):
        """
        Select one index along a still-present dimension.

        This modifies both:
        - values
        - dimension metadata
        """
        axis = self.axis(dim_name)
        dim = self.dimensions[dim_name]

        if dim.coords is not None:
            fixed_value = dim.coords[index]
        else:
            fixed_value = index

        self.values = np.take(self.values, index, axis=axis)

        if self.errors_low is not None:
            self.errors_low = np.take(self.errors_low, index, axis=axis)

        if self.errors_high is not None:
            self.errors_high = np.take(self.errors_high, index, axis=axis)

        if self.n is not None:
            self.n = np.take(self.n, index, axis=axis)

        dim.mode = "fixed"
        dim.parameter = fixed_value
        # dim.coords = dim.coords[index]

        self.validate()
        return self

    def mark_fixed(self, dim_name: str, index: Any):
        """
        Mark a dimension as fixed when the getter has already removed
        that axis from values.
        """
        if dim_name not in self.dimensions:
            raise KeyError(f"Unknown dimension {dim_name!r}")

        dim = self.dimensions[dim_name]

        if dim.mode != "remaining":
            raise ValueError(
                f"Cannot mark {dim_name!r} as fixed; " f"it is already {dim.mode!r}"
            )

        dim.mode = "fixed"
        dim.parameter = index
        # dim.coords = dim.coords[index]

        # self.validate()
        return self

    def reduce_dimension(self, dim_name: str, spec: ReductionSpec):
        axis = self.axis(dim_name)

        if spec.method in ("keep", "single"):
            raise ValueError(
                f"Cannot apply reduction method {spec.method!r} in reduce_dimension()."
            )

        if self.has_errors and spec.error_method != "none":
            raise ValueError(
                "Cannot compute a second independent error after errors already exist."
            )

        values = np.asarray(self.values, dtype=float)

        center = self._compute_center(values, axis, spec.method)
        self.values = center

        if spec.error_method != "none":

            self.errors_low, self.errors_high, self.n = self._compute_error(
                values,
                axis=axis,
                center=center,
                method=spec.error_method,
            )
        else:
            # If errors already exist and we now reduce another dimension normally,
            # reduce the auxiliary arrays consistently.
            if self.has_errors:
                self._reduce_existing_errors(axis=axis, method=spec.method)
            else:
                self.errors_low = None
                self.errors_high = None
                self.n = None

        # if errors_low is not None:
        #     self.errors_low = errors_low
        #     self.errors_high = errors_high
        #     self.n = n

        # # If this reduction creates new errors, store them.
        # if spec.error_method != "none":
        #     self.errors_low = errors_low
        #     self.errors_high = errors_high
        #     self.n = n

        dim = self.dimensions[dim_name]
        dim.mode = "reduced"
        dim.parameter = {
            "method": spec.method,
            "error_method": spec.error_method,
        }

        self.validate()
        return self

    def _compute_center(self, values, axis: int, method: str):
        if method == "mean":
            return np.nanmean(values, axis=axis)

        if method == "median":
            return np.nanmedian(values, axis=axis)

        if method == "max":
            return np.nanmax(values, axis=axis)

        if method == "min":
            return np.nanmin(values, axis=axis)

        if method == "sum":
            return np.nansum(values, axis=axis)

        if method == "std":
            return np.nanstd(values, axis=axis)

        raise ValueError(f"Unknown reduction method: {method!r}")

    def _compute_error(self, values, *, axis: int, center, method: str):
        finite = np.isfinite(values)
        n = np.sum(finite, axis=axis)

        if method == "std":
            err = np.nanstd(values, axis=axis, ddof=1)
            err = np.where(n > 1, err, 0.0)
            return err, err, n

        if method == "sem":
            std = np.nanstd(values, axis=axis, ddof=1)
            sem = std / np.sqrt(np.maximum(n, 1))
            sem = np.where(n > 1, sem, 0.0)
            return sem, sem, n

        if method == "iqr":
            q25 = np.nanpercentile(values, 25, axis=axis)
            q75 = np.nanpercentile(values, 75, axis=axis)
            return center - q25, q75 - center, n

        if method == "bootstrap":
            raise NotImplementedError(
                "Bootstrap error estimation is not implemented yet."
            )
            # return self._bootstrap_error(values, axis=axis, center=center)

        raise ValueError(f"Unknown error method: {method!r}")

    def _reduce_existing_errors(self, *, axis: int, method: str):
        """
        Reduce existing error arrays if another dimension is reduced after
        error creation.

        In session-series mode this should usually not happen, because the
        error-producing neuron reduction should be last. This method exists
        mainly as a safety net.
        """
        if self.errors_low is None or self.errors_high is None:
            return

        if method == "mean":
            reducer = np.nanmean
        elif method == "median":
            reducer = np.nanmedian
        elif method == "max":
            reducer = np.nanmax
        elif method == "min":
            reducer = np.nanmin
        elif method == "sum":
            reducer = np.nansum
        else:
            raise ValueError(
                f"Do not know how to reduce existing errors with method {method!r}."
            )

        self.errors_low = reducer(self.errors_low, axis=axis)
        self.errors_high = reducer(self.errors_high, axis=axis)

        if self.n is not None:
            self.n = np.nansum(self.n, axis=axis)

    def apply_reductions(self, reductions: dict[str, ReductionSpec]):
        """
        Apply non-single reductions.

        Assumption:
        'single' indexers have already been handled by the getter and
        marked via mark_fixed(), or are handled elsewhere.
        """
        for dim_name in list(self.dimensions.keys()):
            spec = reductions.get(dim_name)

            if spec is None:
                continue

            if spec.method == "keep":
                continue

            if spec.method == "single":
                raise ValueError(
                    "'single' should be handled before post-reduction. "
                    "Use getter indexers + mark_fixed(), or select_dimension()."
                )

            self.reduce_dimension(dim_name, spec)

        return self

    def fixed_dimensions(self) -> dict[str, Any]:
        return {
            name: dim.parameter
            for name, dim in self.dimensions.items()
            if dim.mode == "fixed"
        }

    def reduced_dimensions(self) -> dict[str, str]:
        return {
            name: dim.parameter
            for name, dim in self.dimensions.items()
            if dim.mode == "reduced"
        }

    def copy(self) -> "StatisticArray":
        return StatisticArray(
            name=self.name,
            values=np.array(self.values, copy=True),
            dimensions={
                name: Dimension(
                    name=dim.name,
                    coords=np.array(dim.coords, copy=True),
                    mode=dim.mode,
                    parameter=dim.parameter,
                )
                for name, dim in self.dimensions.items()
            },
            errors_low=(
                None
                if self.errors_low is None
                else np.array(self.errors_low, copy=True)
            ),
            errors_high=(
                None
                if self.errors_high is None
                else np.array(self.errors_high, copy=True)
            ),
            n=None if self.n is None else np.array(self.n, copy=True),
        )


@dataclass
class StatisticDefinition:
    key: str
    title: str
    description: str

    # Conceptual full dimensions before indexing/reduction
    dims: tuple[str, ...]

    # Raw calculation function.
    #
    # Contract:
    # getter(data, state, indexers) must return values with all indexed
    # dimensions already removed.
    getter: Callable
    # Returns coordinates for conceptual dimensions.
    coord_getter: Callable[..., dict[str, np.ndarray]] = get_default_coords

    allowed_reductions: Optional[dict[str, tuple[str, ...]]] = None
    default_reductions: Optional[dict[str, ReductionSpec]] = None

    def get_values(
        self,
        data,
        state,
        indexers: dict[str, int] | None = None,
    ) -> StatisticArray:
        """
        Compute statistic values and construct a consistent StatisticArray.

        Indexer contract:
        If indexers={"session": 3}, then the getter must already return
        values without the 'session' axis.
        """
        indexers = indexers or {}

        for dim_name in indexers:
            if dim_name not in self.dims:
                raise ValueError(
                    f"Indexer for unknown dimension {dim_name!r}. "
                    f"Statistic {self.key!r} has dims {self.dims}."
                )

        values = self.getter(
            data,
            state,
            indexers=indexers,
        )

        coords = self.coord_getter(state)

        dimensions = {}

        for dim_name in self.dims:
            dim_coords = np.asarray(coords[dim_name])

            dimensions[dim_name] = Dimension(
                name=dim_name,
                coords=dim_coords,
                mode="remaining",
                parameter=None,
            )

        stat = StatisticArray(
            name=self.key,
            values=np.asarray(values),
            dimensions=dimensions,
        )

        # Mark dimensions already consumed by the getter.
        for dim_name, index in indexers.items():
            dim = stat.dimensions[dim_name]

            if dim.coords is not None:
                fixed_value = dim.coords[index]
            else:
                fixed_value = index
            stat.mark_fixed(dim_name, fixed_value)

        # print(
        #     f"Statistic {self.key!r}: values.shape={stat.values.shape}, dims={stat.dims}"
        # )
        stat.validate()
        return stat

    def get_default_reductions(self) -> dict[str, ReductionSpec]:
        if self.default_reductions is not None:
            return dict(self.default_reductions)

        # fallback: keep first dim, mean all others
        """
        should rather be "keep" for all?
        """
        return {
            # dim: ReductionSpec("keep" if i == 0 else "mean")
            dim: ReductionSpec("keep")
            for i, dim in enumerate(self.dims)
        }

    def get_allowed_reductions(self, dim_name: str) -> tuple[str, ...]:
        """
        returns allowed reduction methods for this statistic
        """
        if self.allowed_reductions is None:
            return DEFAULT_REDUCTIONS

        return self.allowed_reductions.get(
            dim_name,
            DEFAULT_REDUCTIONS,
        )

    def get_dimension_info(self, state) -> dict[str, DimensionInfo]:
        coords = self.coord_getter(state)

        info = {}

        for dim_name in self.dims:
            dim_coords = coords.get(dim_name)

            if dim_coords is None:
                raise ValueError(
                    f"No coordinates found for dimension {dim_name!r} "
                    f"of statistic {self.key!r}."
                )

            dim_coords = np.asarray(dim_coords)

            info[dim_name] = DimensionInfo(
                size=len(dim_coords),
                # labels=[str(v) for v in dim_coords],
                coords=dim_coords,
            )

        return info

    def has_pair_dimension(self, target: str) -> bool:
        if target == "neuron":
            return "neuron_i" in self.dims and "neuron_j" in self.dims

        if target == "session":
            return "session_i" in self.dims and "session_j" in self.dims

        return False


STATISTICS = {
    "none": StatisticDefinition(
        key="none",
        title="None",
        description="No statistic selected.",
        dims=(),
        getter=lambda data, state, indexers: None,  # lambda data: StatisticArray(np.array([]), dims=()),
    ),
    "snr": StatisticDefinition(
        key="snr",
        title="SNR",
        description="Signal-to-noise ratio of inferred activity traces.",
        dims=("neuron", "session"),
        getter=lambda data, state, indexers: analysis.get_quality_metric(
            data, state, indexers=indexers, key="SNR_comp"
        ),
    ),
    "rval": StatisticDefinition(
        key="rval",
        title="R-value",
        description="Correlation coefficients from the quality metrics.",
        dims=("neuron", "session"),
        getter=lambda data, state, indexers: analysis.get_quality_metric(
            data, state, indexers=indexers, key="r_values"
        ),
    ),
    "cnn": StatisticDefinition(
        key="cnn",
        title="CNN score",
        description="Predictions from the convolutional neural network.",
        dims=("neuron", "session"),
        getter=lambda data, state, indexers: analysis.get_quality_metric(
            data, state, indexers=indexers, key="cnn_preds"
        ),
    ),
    "footprint_size": StatisticDefinition(
        key="footprint_size",
        title="Footprint size",
        description="Size of the neuron footprint in pixels.",
        dims=("neuron", "session"),
        getter=lambda data, state, indexers: analysis.calculate_footprint_size(
            data, state, indexers
        ),
    ),
    "border_proximity": StatisticDefinition(
        key="border_proximity",
        title="Border proximity",
        description="Proximity of neuron centroids to the borders of the field of view.",
        dims=("neuron", "session"),
        getter=lambda data, state, indexers: analysis.calculate_border_proximity(
            data, state, indexers
        ),
    ),
    "occurence": StatisticDefinition(
        key="occurence",
        title="Occurrence",
        description="Number of sessions in which each neuron is present.",
        dims=("neuron", "session"),
        getter=lambda data, state, indexers: analysis.calculate_occurrence(
            data, state, indexers
        ),
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
        getter=lambda data, state, indexers: analysis.calculate_centroid_shift(
            data, state, indexers
        ),
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
        getter=lambda data, state, indexers: analysis.calculate_temporal_correlation(
            data, state, indexers
        ),
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
        getter=lambda data, state, indexers: analysis.calculate_distances(
            data, state, indexers
        ),
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
        getter=lambda data, state, indexers: analysis.calculate_footprint_similarity(
            data, state, indexers
        ),
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
