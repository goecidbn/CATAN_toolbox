from typing import Dict, List, Optional, Tuple, Any, Literal

import logging
import numpy as np
from scipy import sparse
from pathlib import Path

from .remap import Remapping

from catan.core.io import (
    load_file,
    load_fields_from_sources,
    save_file,
    NATIVE_SESSION_CONFIG,
)
from catan.core.structures.load_config import LoadConfig, FieldSpec
from catan.core.structures.inclusion import inclusion_mask
from catan.core.data import center_of_mass
from catan.core.io.isolated_read import read_fields
from catan.core import footprint_frame

from catan.core.spatial_geometry import (
    check_shapes,
    background_transpose,
)

sessiondata_type = Literal["spatial", "traces", "quality"]

component_quality_default = {
    "SNR_lowest": 1.0,
    "SNR_min": 2.5,
    "rval_lowest": -1,
    "rval_min": 0.8,
    "cnn_lowest": 0.1,
    "cnn_min": 0.9,
}


def empty_footprint_mask(footprints):
    """Columns of a footprint matrix without any positive pixel."""
    if footprints is None or footprints.shape[1] == 0:
        return np.zeros(0 if footprints is None else footprints.shape[1], dtype=bool)
    peaks = np.asarray(sparse.csc_matrix(footprints).max(axis=0).todense()).ravel()
    return ~(peaks > 0)


class SessionData:
    ## meta data
    name: Optional[str] = None  #
    path: Optional[str] = None  #
    id: int = -1  #

    source_type: str = "session"
    source_config: LoadConfig | None = None

    active: bool = True  #
    time_offset: float = 0.0  #
    session_color: Optional[str] = None
    use_kde: bool = False

    status: Dict[str, bool] = {}

    ## loaded fields (from input)
    # spatial
    dims: Tuple[int, int] = (512, 512)  #
    footprints: sparse.csc_matrix = sparse.csc_matrix((0, 0))  #
    background: Optional[np.ndarray] = None  #
    background_origin: Optional[str] = None  #
    background_template: Optional[np.ndarray] = None
    # Background vs. own footprints at loading (footprint_frame.
    # background_coherence); None without a loaded background.
    background_coherence: Optional[dict] = None
    included: np.ndarray = np.array([], dtype=bool)
    synthetic: np.ndarray = np.array([], dtype=bool)  #
    # traces
    _traces: dict[str, np.ndarray] = {}
    _default_trace: Optional[str] = None
    # other
    quality: dict[str, np.ndarray] = {}  #

    ## to be calculated (from input)
    remap: Optional[Remapping] = None  #
    n_neurons: int = 0  #
    centroids: Optional[np.ndarray] = None  #
    ## to be calculated (with additional information)
    # idx_kde: np.ndarray

    HDF5_VERSION = 1

    def __init__(
        self,
        name: Optional[str] = None,
        alignment_references: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """ """
        self.name = name
        self.path = kwargs.get("path", None)
        self._restored_from_catan: bool = False

        self.status = {
            # load status flags
            "spatial_loaded": False,
            "traces_loaded": False,
            "quality_loaded": False,
            # processing status flags
            "aligned": False,
            "registered_to_model": False,
            "matched": False,
        }

        self.alignment_issue: str | None = None

        self.alignment_metrics = {
            "shift": None,
            "correlation": None,
            "correlation_zscore": None,
        }

        self.derived_statistics = {}
        self.known_trace_keys = []
        self.trace_revision = 0

        self.set_parameters(**kwargs)

        ## if some elements are provided in kwargs, which fit
        ## the general fields to be loaded, register them
        self.register_data(alignment_references=alignment_references, **kwargs)

        if alignment_references is None and kwargs.get("remap") is not None:
            self.remap = kwargs["remap"]

        self.evaluate_alignment_status()

    def set_parameters(self, **input):

        self.params = {
            # alignment parameters
            "max_session_shift": 50.0,
            "min_session_correlation": 0.3,
            "min_session_correlation_zscore": 4.0,
            "max_session_rotation": 10.0,
            "correct_rotation": False,
            "rotation_step": 1.0,
            "rotation_refine_step": 0.1,
            # flow alignment departing from the default (older reference,
            # shift beyond max_session_shift, low correlation) needs the
            # user's confirmation; False accepts it
            "confirm_alignment_deviations": True,
            # kde parameters
            "use_kde": False,
            "qtl": [0.05, 0.95],
        }

        # update with input parameters if provided
        for key in self.params:
            if key in input:
                self.params[key] = input[key]

    # def _ensure_component_flags(self, *, included=None, synthetic=None):

    #     n = self.n_neurons

    #     def prepare(supplied, current, default, name):
    #         if supplied is not None:
    #             arr = np.asarray(supplied, dtype=bool).reshape(-1)

    #             if arr.shape != (n,):
    #                 raise ValueError(
    #                     f"{name} has shape {arr.shape}, " f"expected {(n,)}."
    #                 )

    #             return arr.copy()

    #         current = np.asarray(current, dtype=bool).reshape(-1)

    #         if current.shape == (n,):
    #             return current

    #         result = np.full(n, default, dtype=bool)

    #         n_copy = min(len(current), n)

    #         if n_copy:
    #             result[:n_copy] = current[:n_copy]

    #         return result

    #     self.included = prepare(included, self.included, True, "included")
    #     self.synthetic = prepare(synthetic, self.synthetic, False, "synthetic")

    ### ========================================================= ###
    ### ================= LOAD / SAVE METHODS =================== ###
    ### ========================================================= ###

    @staticmethod
    def _from_file(
        path: str | Path,
        fields_to_load: dict[str, dict[str, FieldSpec]] | None = None,
        alignment_references: Optional[np.ndarray] = None,
    ) -> "SessionData":
        if fields_to_load is None:
            fields_to_load = LoadConfig.fields_from_resource(
                NATIVE_SESSION_CONFIG, enabled_only=False
            )

        data = load_file(path, fields_to_load)
        return SessionData._from_dict(data, alignment_references=alignment_references)

    @staticmethod
    def _from_dict(
        data: dict, alignment_references: Optional[np.ndarray] = None
    ) -> "SessionData":
        session = SessionData(alignment_references=alignment_references, **data)
        # session.register_data(alignment_references=alignment_references, **data)
        return session

    def load_data(
        self,
        fields_to_load: dict[str, dict[str, FieldSpec]] | None = None,
        **kwargs,
    ) -> None:
        if not self.path:
            raise ValueError("No source path is configured for this session.")

        if fields_to_load is None:
            fields_to_load = LoadConfig.fields_from_resource(
                NATIVE_SESSION_CONFIG, enabled_only=False
            )

        ctx = kwargs.get("ctx")

        spatial = "footprints" in fields_to_load.get("spatial", {})
        config = getattr(self, "source_config", None)

        if config is not None:
            dimensions = config.dimensions
        elif fields_to_load.get("spatial", {}).get("background") is not None:
            dimensions = {"mode": "image", "field": None}
        else:
            dimensions = {
                "mode": "manual",
                "height": self.dims[0],
                "width": self.dims[1],
            }

        data = read_fields(
            self.path,
            fields_to_load,
            ctx=ctx,
            operation="session_fields" if spatial else "fields",
            parameters={
                "dimensions": dimensions,
                "orientation": kwargs.get("background_orientation", "auto"),
                "expected_dims": kwargs.get("expected_dims"),
            },
        )

        if ctx is not None:
            ctx.check_cancelled()
            ctx.message("Applying loaded session data…")

        self.register_data(
            alignment_references=kwargs.get("alignment_references"),
            background_orientation=kwargs.get("background_orientation", "auto"),
            **data,
        )

    def save(
        self,
        path: str | Path,
        fields_to_save: dict[str, dict[str, FieldSpec]] | None = None,
        *,
        mat_version: Literal["pre73", "7.3"] = "7.3",
    ) -> None:

        fields_to_save = fields_to_save or LoadConfig.fields_from_resource(
            NATIVE_SESSION_CONFIG,
            enabled_only=False,
        )

        save_file(
            path,
            self,
            fields_to_save,
            mat_version=mat_version,
            root_attributes={"object_type": "SessionData", "format_version": 1},
            root="/",
        )

    ### ========================================================= ###
    ### ================= REGISTRATION METHODS ================== ###
    ### ========================================================= ###

    def register_data(
        self,
        alignment_references: Optional[np.ndarray] = None,
        *,
        background_orientation="auto",
        **data,
    ):
        """
        Registers data from kwargs 'data' input to SessionData object. Requires 'data' to contain the keys 'spatial', 'traces', and 'quality' with the corresponding data keys to be registered.

        If alignment_references is provided, spatial data will be aligned to it.
        """

        self.register_spatial(
            alignment_references=alignment_references,
            background_orientation=background_orientation,
            **data.get("spatial", {}),
        )
        self.register_traces(**data.get("traces", {}))
        self.register_quality(**data.get("quality", {}))
        self.register_metadata(**data.get("metadata", {}))

        if "remap" in data:
            self.remap = data["remap"]

    def register_metadata(self, **data):
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(
                    f"Warning: SessionData has no attribute '{key}' to register metadata."
                )

    def clean_data(self, which_in: Optional[str] = None):
        """
        Cleans the specified data type(s) from the session. If which_in is None, cleans all data types.
        """
        if which_in is None:
            which = ["spatial", "traces", "quality"]
        else:
            which = [which_in]

        if "traces" in which:
            self._clean_traces()
        if "spatial" in which:
            self._clean_spatial()
        if "quality" in which:
            self._clean_quality()

    ### ========================================================== ###
    ### ==================== QUALITY METHODS ===================== ###
    ### ========================================================== ###

    def register_quality(self, **data):

        if not data:
            return

        self.quality = data if data else {}

        if self.quality:
            self.status["quality_loaded"] = True

        # self.component_evaluation_from_quality_params()

    def component_evaluation_from_quality_params(
        self, component_quality=None, reset=False
    ):
        """
        function to create included boolean array based on component quality thresholds
        defined in self.params

        requires:
            * self.params containing SNR, rval, cnn thresholds

        returns:
            * included boolean array
        """
        if not self.status["quality_loaded"] or self.quality is None:
            # print("no quality info provided, skipping quality-based filtering")
            return

        # if self.n_neurons is None:
        #     self.n_neurons = self.quality["SNR_comp"].shape[0]
        assert (
            isinstance(self.n_neurons, int) and self.n_neurons > 0
        ), "n_neurons must be a positive integer"

        if reset:
            self.included = np.ones(self.n_neurons, dtype=bool)

        assert (
            len(self.included) == self.n_neurons
        ), "Included array length must match number of neurons."

        ## provide dummy values if not provided
        SNR_comp = self.quality.get("SNR_comp", np.full(self.n_neurons, np.inf))
        r_values = self.quality.get("r_values", np.full(self.n_neurons, np.inf))
        cnn_preds = self.quality.get("cnn_preds", np.full(self.n_neurons, np.inf))

        component_quality = (
            component_quality or component_quality_default
        )  # if not provided, use default thresholds

        ## all components must pass 'lowest' threshold...
        self.included &= SNR_comp >= component_quality["SNR_lowest"]
        self.included &= r_values >= component_quality["rval_lowest"]
        self.included &= cnn_preds >= component_quality["cnn_lowest"]
        ## ... and at least pass one 'min' threshold
        self.included &= (
            (SNR_comp >= component_quality["SNR_min"])
            | (r_values >= component_quality["rval_min"])
            | (cnn_preds >= component_quality["cnn_min"])
        )

    def _clean_quality(self):
        self.quality = {}
        self.status["quality_loaded"] = False

    ### ========================================================== ###
    ### ===================== TRACE METHODS ====================== ###
    ### ========================================================== ###

    @property
    def traces(self):
        return self._traces

    @property
    def trace(self):
        if self._default_trace is None:
            return None
        return self._traces.get(self._default_trace, None)

    def register_traces(self, **data):

        if not data:
            return

        self.known_trace_keys = sorted(set(self.known_trace_keys) | set(data))

        # Replacing traces invalidates calculations based on the old data.
        self.trace_revision += 1
        self.derived_statistics = {}

        self._traces = data if data else {}

        self._default_trace = "F_dff_dec" if "F_dff_dec" in self._traces else "C"
        if self._traces:
            self.status["traces_loaded"] = True

    def _clean_traces(self):
        # print("Cleaning traces for session. Current traces:", self._traces.keys())
        self._traces = {}
        self._default_trace = None
        self.status["traces_loaded"] = False

    ### ========================================================== ###
    ### ===================== SPATIAL METHODS ==================== ###
    ### ========================================================== ###

    def register_spatial(
        self,
        alignment_references=None,
        *,
        background_orientation="auto",
        **data,
    ):
        if "footprints" not in data or data["footprints"] is None:
            return

        if background_orientation not in {
            "auto",
            "as_stored",
            "transpose",
        }:
            raise ValueError(
                f"Unknown background orientation: " f"{background_orientation!r}"
            )

        footprints = data["footprints"]

        if not sparse.issparse(footprints):
            raise ValueError(
                "Footprints must be sparse, with shape " "(height * width, neurons)."
            )

        footprints = sparse.csc_matrix(footprints)
        loaded_included = None

        if data.get("included") is not None:
            loaded_included = inclusion_mask(
                data["included"],
                footprints.shape[1],
                "Loaded session inclusion",
            )

        loaded_background = data.get("background")
        if loaded_background is not None:
            loaded_background = self._prepare_background(loaded_background)

        dims = data.get("dims")

        if dims is None and loaded_background is not None:
            dims = loaded_background.shape
            if background_orientation == "transpose":
                dims = dims[::-1]

        if dims is None:
            raise ValueError(
                "Specify image dimensions when loading "
                "footprints without a background."
            )

        dims = check_shapes(
            dims,
            footprints.shape,
            (None if loaded_background is None else loaded_background.shape),
            background_orientation,
        )

        for reference in (alignment_references or {}).values():
            shape = np.asarray(reference["template"]).shape
            if tuple(shape) != dims:
                raise ValueError(
                    f"Alignment reference shape {shape} " f"does not match {dims}."
                )

        footprints_proj = np.asarray(
            footprints.sum(axis=1),
            dtype=np.float32,
        ).reshape(dims)

        coherence = None
        if loaded_background is None:
            origin = "footprints"
            template = footprints_proj

        else:
            origin = "loaded"
            transpose = background_transpose(
                loaded_background.shape,
                dims,
                background_orientation,
            )
            template = loaded_background.T if transpose else loaded_background

            if background_orientation == "auto" and dims[0] == dims[1]:
                orientation = Remapping(evaluate=False)
                orientation.test_transpose(
                    footprints_proj,
                    template,
                )
                template = orientation.fix_transpose(template)

            coherence = self._correct_background(template, footprints_proj)

        # Validate geometry before modifying the session.
        self.dims = dims
        self.footprints = footprints
        self.background_origin = origin
        self.background_coherence = coherence

        # Never alter this copy through cross-session alignment.
        self.background_template = np.asarray(
            template,
            dtype=np.float32,
        ).copy()
        self._corrected_template_cache = None
        self.background = np.asarray(self.corrected_template(), dtype=np.float32).copy()

        self.status["spatial_loaded"] = True

        if alignment_references:
            self.align_to_reference(
                alignment_references,
                use_optical_flow=True,
                alignment_template=(
                    footprints_proj
                    if self.alignment_source == "footprints"
                    else None
                ),
            )
        else:
            self.remap = Remapping.identity(self.dims)
            self.postprocess_spatial_data()

        # The original-frame footprint projection (see footprint_template).
        self._footprint_template_cache = (
            (id(self.footprints), id(self.remap)),
            footprints_proj,
        )

        if loaded_included is not None:
            self.included = loaded_included

        # Footprints without any pixel (moved entirely out of the field of
        # view by the alignment, or empty in the source) never take part;
        # Data reports them (empty_excluded).
        empty = empty_footprint_mask(self.footprints) & np.asarray(self.included, dtype=bool)
        if empty.any():
            self.included = np.asarray(self.included, dtype=bool).copy()
            self.included[empty] = False
            self.empty_excluded = getattr(self, "empty_excluded", 0) + int(empty.sum())

        self.evaluate_alignment_status()

    @staticmethod
    def _prepare_background(background: np.ndarray) -> np.ndarray:
        background = SessionData._background_as_grayscale(background)

        lo = float(background.min())
        hi = float(background.max())

        if hi > lo:
            return (background - lo) / (hi - lo)

        return np.zeros_like(background, dtype=np.float32)

    @staticmethod
    def _background_as_grayscale(background: np.ndarray) -> np.ndarray:
        image = np.asarray(background, dtype=np.float32)

        if image.size == 0:
            raise ValueError("Spatial background is empty.")

        finite = np.isfinite(image)
        if not finite.all():
            if not finite.any():
                raise ValueError("Spatial background contains no finite values.")

            # E.g. CaImAn's local correlation image (Cn) is NaN where a pixel
            # is constant over the whole movie (zero-padded borders after
            # motion correction, dead regions): such pixels carry no signal
            # and are shown/aligned as the darkest finite value.
            invalid = int(np.count_nonzero(~finite))
            logging.getLogger(__name__).warning(
                "Spatial background: %d of %d pixels are NaN/Inf; filled with "
                "the minimum finite value.",
                invalid,
                image.size,
            )
            image = np.where(finite, image, image[finite].min()).astype(np.float32)

        if image.ndim == 2:
            return image

        if image.ndim == 3 and image.shape[-1] == 3:
            return image.mean(axis=-1, dtype=np.float32)

        raise ValueError(
            "Spatial background must be a 2D grayscale image "
            f"or an (H, W, 3) RGB image; got shape {image.shape}."
        )

    def update_footprints(
        self,
        footprints: sparse.csc_matrix,
        mode="replace",
        included_values: bool | np.ndarray = True,
        synthetic_values: bool | np.ndarray = False,
    ):

        if mode == "replace":
            n_new = footprints.get_shape()[1] - self.n_neurons

            self.footprints = footprints
            self.centroids = center_of_mass(
                self.footprints, *self.dims
            )
        elif mode == "append":
            n_new = footprints.get_shape()[1]
            self.footprints = sparse.hstack([self.footprints, footprints], format="csc")
            centroids = center_of_mass(
                footprints, *self.dims
            )
            self.centroids = np.vstack([self.centroids, centroids])
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        self.n_neurons = self.footprints.get_shape()[1]

        def update_status_arrays(current, n_new, new_values):
            ## set included values to values according to input:
            if isinstance(new_values, bool):
                # if scalar value is given, pad the existing included array with this value for the new neurons
                current = np.pad(
                    current,
                    (0, n_new),
                    mode="constant",
                    constant_values=new_values,
                )
            else:
                ## if an array is given, check its length
                if mode == "replace" and len(new_values) == self.n_neurons:
                    ## if the length matches the total number of neurons, use it directly
                    current = new_values
                elif mode == "append" and len(new_values) == n_new:
                    ## if the length matches the number of new neurons, append it to the existing included array
                    current = np.append(current, new_values)
                else:
                    raise ValueError(
                        "Length of included_value array must match the number of new neurons or the total number of neurons."
                    )
            return current

        self.included = update_status_arrays(self.included, n_new, included_values)
        self.synthetic = update_status_arrays(self.synthetic, n_new, synthetic_values)

    def postprocess_spatial_data(self):
        if self.footprints is None:
            raise ValueError(
                "Spatial data (footprints and background) must be loaded before postprocessing."
            )

        self.n_neurons = self.footprints.get_shape()[1]
        self.included = np.ones(self.n_neurons, dtype=bool)
        self.synthetic = np.zeros(self.n_neurons, dtype=bool)

        self.centroids = center_of_mass(
            self.footprints, *self.dims
        )
        # self.evaluate_components_from_footprints(reset=True)
        # self.get_idx_kde()

    def evaluate_components_from_footprints(self, footprints_thr=10, reset=False):
        """
        function to create included boolean array based on component size thresholds

        requires:
            * self.footprints containing spatial footprints

        returns:
            * included boolean array
        """

        if not self.status["spatial_loaded"] or self.footprints is None:
            return
        ## finding non-empty rows in sparse array (https://mike.place/2015/sparse/)
        # included = np.ones(nA, bool)
        # included = np.diff(footprints.indptr) != 0

        if reset:
            self.included = np.ones(self.n_neurons, dtype=bool)

        assert (
            len(self.included) == self.n_neurons
        ), "Included array length must match number of neurons."

        ## only footprints above a certain size should be considered for evaluation
        self.included &= self.footprints.getnnz(axis=0) > footprints_thr

    def _clean_spatial(self):
        self.footprints = sparse.csc_matrix((0, 0))
        self.background = None
        self.background_origin = None
        self.background_template = None
        self.background_coherence = None
        self._footprint_template_cache = None
        self._corrected_template_cache = None
        self.remap = None
        self.status["aligned"] = False
        self.included = np.array([], dtype=bool)
        self.status["spatial_loaded"] = False

    ### ========================================================= ###
    ### ================== ALTERATION METHODS =================== ###
    ### ========================================================= ###

    # Detected shapes of components whose shape an assignment set replaced
    # (replace_footprint); never mutated, only reassigned (sessions are
    # shallow-copied).
    replaced_footprints = {}

    def replace_footprint(self, fp_id: int, footprint) -> None:
        """Give component ``fp_id`` a new shape; its ID and traces stay.
        remove_synthetic_components() restores the detected shape."""
        footprint = sparse.csc_matrix(footprint, dtype=self.footprints.dtype)
        expected_shape = (self.footprints.shape[0], 1)
        if footprint.shape != expected_shape:
            raise ValueError(
                f"Replacement footprint has shape {footprint.shape}, "
                f"expected {expected_shape}."
            )

        current = sparse.csc_matrix(self.footprints)
        originals = dict(self.replaced_footprints)
        originals.setdefault(int(fp_id), current[:, fp_id].copy())
        self.replaced_footprints = originals

        self._set_footprint_column(int(fp_id), footprint)

    def _set_footprint_column(self, fp_id, footprint):
        current = sparse.csc_matrix(self.footprints)
        self.footprints = sparse.hstack(
            [current[:, :fp_id], footprint, current[:, fp_id + 1 :]], format="csc"
        )
        centroids = np.array(self.centroids, dtype=float, copy=True)
        centroids[fp_id] = center_of_mass(footprint, *self.dims)
        self.centroids = centroids

    def restore_replaced_footprint(self, fp_id: int) -> bool:
        """Detected shape back for one replaced component (undo)."""
        originals = dict(self.replaced_footprints)
        original = originals.pop(int(fp_id), None)
        if original is None:
            return False
        self.replaced_footprints = originals
        self._set_footprint_column(int(fp_id), original)
        return True

    def restore_replaced_footprints(self):
        """Detected shapes back for all replaced components."""
        originals = self.replaced_footprints
        self.replaced_footprints = {}
        for fp_id, footprint in sorted(originals.items()):
            if fp_id < self.footprints.shape[1]:
                self._set_footprint_column(fp_id, footprint)

    def remove_synthetic_components(self):
        """Drop all synthetic components (appended after the detected
        ones, see append_synthetic_component) and replaced shapes: the
        session as detected, before the manipulations of an assignment set."""
        if not self.status.get("spatial_loaded", False) or self.synthetic is None:
            return
        if self.replaced_footprints:
            self.restore_replaced_footprints()
        synthetic = np.asarray(self.synthetic, dtype=bool)
        if not synthetic.any():
            return

        count = int(np.argmax(synthetic))
        if not synthetic[count:].all():
            raise ValueError(
                f"Session {self.name}: synthetic components are not the last ones."
            )

        self.footprints = sparse.csc_matrix(self.footprints)[:, :count]
        self.centroids = np.asarray(self.centroids)[:count]
        self.included = np.asarray(self.included, dtype=bool)[:count].copy()
        self.synthetic = synthetic[:count].copy()
        self.n_neurons = count
        for key in list(self._traces):
            self._traces[key] = np.asarray(self._traces[key])[:count]
        for key in list(self.quality):
            values = self.quality[key]
            if values is not None and np.ndim(values) and len(values) > count:
                self.quality[key] = np.asarray(values)[:count]

    def session_inclusion(self):
        """Inclusion independent of any assignment set: all components
        (filtering happens at detection or in an assignment set's curation)."""
        return np.ones(self.n_neurons, dtype=bool)

    def append_synthetic_component(self, footprint: sparse.csc_matrix) -> int:
        """
        Append one synthetic component.

        Returns its new footprint/component ID.
        """

        footprint = footprint.tocsc()

        expected_shape = (self.footprints.shape[0], 1)

        if footprint.shape != expected_shape:
            raise ValueError(
                f"Synthetic footprint has shape "
                f"{footprint.shape}, expected "
                f"{expected_shape}."
            )

        fp_id = self.n_neurons

        # --------------------------------------------
        # Spatial footprint
        # --------------------------------------------
        self.footprints = sparse.hstack([self.footprints, footprint], format="csc")

        self.n_neurons += 1

        centroid = center_of_mass(
            footprint,
            *self.dims
        )

        if self.centroids is None:
            self.centroids = centroid
        else:
            self.centroids = np.vstack([self.centroids, centroid])

        # --------------------------------------------
        # Component status
        # --------------------------------------------

        self.included = np.append(self.included, True)
        self.synthetic = np.append(self.synthetic, True)

        # --------------------------------------------
        # Empty trace entries
        # --------------------------------------------

        def append_nan_row(values):
            values = np.asarray(values)
            if not np.issubdtype(values.dtype, np.floating):
                values = values.astype(float)

            empty = np.full((1,) + values.shape[1:], np.nan, dtype=values.dtype)

            return np.concatenate([values, empty], axis=0)

        for key in list(self._traces):
            self._traces[key] = append_nan_row(self._traces[key])

        # --------------------------------------------
        # Empty quality entries
        # --------------------------------------------

        for key in list(self.quality):
            self.quality[key] = append_nan_row(self.quality[key])

        return fp_id

    ### ========================================================= ###
    ### ==================== ALIGNMENT METHODS ================== ###
    ### ========================================================= ###

    # ------------------------------------------------------------------
    # Footprints relative to the background (core.footprint_frame)
    # ------------------------------------------------------------------

    @property
    def background_correction(self):
        """The correction bringing the background into the footprints'
        frame (footprint_frame.correction_of), or None."""
        return footprint_frame.correction_of(self.params)

    def _correct_background(self, template, footprint_projection):
        """Estimate (or keep a manual) background correction for the loaded
        ``template``; returns the corrected background's coherence."""
        coherence = footprint_frame.correct_background(
            self.params, template, footprint_projection
        )
        if coherence.get("coherent") is False:
            logging.getLogger(__name__).warning(
                "Session %s: %s.", self.name, coherence["reason"]
            )
        elif coherence.get("uncorrected_offset") is not None:
            logging.getLogger(__name__).info(
                "Session %s: background moved onto its footprints (%s).",
                self.name,
                footprint_frame.describe_correction(self.background_correction),
            )
        return coherence

    def corrected_template(self):
        """The background template in the footprints' frame (its correction
        applied); the background template itself without correction."""
        template = self.background_template
        correction = self.background_correction
        if template is None or footprint_frame.is_identity(correction):
            return template
        key = (id(template), footprint_frame.correction_key(correction))
        cached = getattr(self, "_corrected_template_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        image = footprint_frame.apply_correction(template, correction)
        self._corrected_template_cache = (key, image)
        return image

    @property
    def alignment_source(self):
        """"background" or "footprints": the image this session is aligned
        by. Set explicitly in params["alignment_source"]; otherwise the
        footprints when the background does not resemble them."""
        explicit = self.params.get("alignment_source")
        if explicit in footprint_frame.ALIGNMENT_SOURCES:
            return explicit
        coherence = self.background_coherence or {}
        # Only a background that does not resemble its footprints at all;
        # one offset from them is corrected (background_correction).
        return (
            "footprints"
            if coherence.get("coherent") is False and not footprint_frame.matched(coherence)
            else "background"
        )

    def footprint_template(self):
        """Projection of the footprints in the original frame of the
        background template; None without footprints."""
        footprints = getattr(self, "footprints", None)
        if footprints is None or footprints.shape[1] == 0:
            return None
        key = (id(footprints), id(self.remap))
        cached = getattr(self, "_footprint_template_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        image = footprint_frame.unaligned_projection(self)
        self._footprint_template_cache = (key, image)
        return image

    def alignment_template(self):
        """The original-frame image this session is aligned by: its
        (corrected) background template, or its footprint projection
        (alignment_source)."""
        if self.alignment_source == "footprints" and self.background_origin != "footprints":
            image = self.footprint_template()
            if image is not None and self.background_template is not None and (
                image.shape == self.background_template.shape
            ):
                return image
        return self.corrected_template()

    def prepare_background_template(
        self,
        background: np.ndarray,
        *,
        background_orientation="auto",
    ) -> np.ndarray:
        """Prepare a source background without modifying this session."""

        if background_orientation not in {"auto", "as_stored", "transpose"}:
            raise ValueError(
                f"Unknown background orientation: {background_orientation!r}"
            )

        template = self._background_as_grayscale(background)
        dims = tuple(self.dims)

        if background_orientation == "transpose":
            template = template.T

        orientation = Remapping(evaluate=False)
        orientation.test_transpose(self.background_template, template)

        if background_orientation == "auto":
            if template.shape != dims:
                if template.T.shape == dims:
                    template = template.T
                else:
                    raise ValueError(
                        f"Background has incompatible shape {template.shape}; "
                        f"expected {dims}."
                    )

            elif dims[0] == dims[1] and self.background_template is not None:
                # orientation = Remapping(evaluate=False)
                # orientation.test_transpose(self.background_template, template)
                template = orientation.fix_transpose(template)

        if template.shape != dims:
            raise ValueError(
                f"Background orientation {background_orientation!r} produces "
                f"shape {template.shape}, but the session requires {dims}. "
                "Choose another background orientation and load again."
            )

        return np.asarray(template, dtype=np.float32).copy()

    def propose_remapping(
        self,
        alignment_references,
        *,
        background_template=None,
        use_optical_flow=True,
        correct_rotation: bool | None = None,
        require_confirmation: bool | None = None,
        moving_centroids=None,
    ) -> Remapping:
        """
        Calculate a candidate remapping without
        modifying any SessionData state.

        If background_template is omitted, the
        session's current unaligned template is used.
        """

        template = (
            self.alignment_template()
            if background_template is None
            else background_template
        )

        if template is None:
            raise ValueError("No background template available for alignment.")

        template = np.asarray(template, dtype=np.float32)

        if template.shape != tuple(self.dims):
            raise ValueError(
                f"Background template has shape {template.shape}, expected {tuple(self.dims)}."
            )

        # First/reference session.
        if not alignment_references:
            return Remapping.identity(self.dims)

        if correct_rotation is None:
            correct_rotation = self.params.get("correct_rotation", False)

        if require_confirmation is None:
            require_confirmation = self.params.get(
                "confirm_alignment_deviations", True
            )

        return Remapping(
            template=template,
            references=alignment_references,
            use_optical_flow=(use_optical_flow),
            max_shift=self.params["max_session_shift"],
            max_rotation=(
                self.params["max_session_rotation"] if correct_rotation else 0.0
            ),
            min_corr=self.params["min_session_correlation"],
            min_zcorr=self.params["min_session_correlation_zscore"],
            rotation_step=self.params["rotation_step"],
            rotation_refine_step=self.params["rotation_refine_step"],
            require_confirmation=require_confirmation,
            # Positions of the imaging window accepted by the user (dx, dy).
            allowed_offsets=self.params.get("alignment_offsets", ()),
            min_flow_correlation=self.params.get("min_flow_correlation", 0.10),
            moving_centroids=(
                self.template_centroids() if moving_centroids is None else moving_centroids
            ),
        )

    def template_centroids(self):
        """Included neurons' centroids (x, y) in the unaligned template's
        pixel frame, or None; for footprints already warped by a successful
        alignment, by inverting it."""
        from catan.core import footprint_alignment

        if self.centroids is None or not len(self.centroids):
            return None
        points = np.asarray(self.centroids, dtype=np.float64)
        if self.included is not None and len(self.included) == len(points):
            points = points[np.asarray(self.included, dtype=bool)]
        points = points[np.isfinite(points).all(axis=1)]

        remap = self.remap
        if remap is None or not remap.success:
            return points
        if remap.flow is not None:
            points = points + footprint_alignment.sample_flow(remap.flow, points)
        inverse = np.linalg.inv(np.asarray(remap.matrix, dtype=np.float64))
        points = points @ inverse[:2, :2].T + inverse[:2, 2]
        return points[:, ::-1] if remap.transpose else points

    def refine_remap_with_footprints(self, remap, alignment_references, *, use_optical_flow=True):
        """``remap`` corrected by the rigid offset between this session's
        footprint centroids (raw footprints, mapped through ``remap``) and
        the references' aligned centroids (core.footprint_alignment).
        The outcome is recorded in flow_info["footprint_refinement"]."""
        from catan.core import footprint_alignment

        references = [
            np.asarray(entry["centroids"])
            for entry in (alignment_references or {}).values()
            if entry.get("centroids") is not None and len(entry["centroids"])
        ]
        if not references or self.footprints is None or self.footprints.shape[1] == 0:
            return remap

        raw = center_of_mass(self.footprints, *self.dims)
        if remap.transpose:
            raw = raw[:, ::-1]
        points = footprint_alignment.map_points(
            raw, remap.matrix, remap.flow if use_optical_flow else None
        )
        center = (self.dims[1] / 2, self.dims[0] / 2)
        result = footprint_alignment.estimate_correction(points, references, center)

        if result["accepted"]:
            remap = footprint_alignment.compose_correction(remap, result["matrix"])

        remap.flow_info = dict(remap.flow_info or {})
        remap.flow_info["footprint_refinement"] = {
            key: result.get(key)
            for key in ("parameters", "accepted", "reason", "references", "matches", "median_before", "median_after")
        }
        return remap

    def align_to_reference(
        self, alignment_references, use_optical_flow=True, *, alignment_template=None
    ):

        if (
            not self.status["spatial_loaded"]
            or self.footprints is None
            or self.background_template is None
        ):
            raise ValueError("Spatial data must be loaded before alignment.")

        self.remap = self.propose_remapping(
            alignment_references,
            background_template=alignment_template,
            use_optical_flow=use_optical_flow,
        )

        # Optional: correct the cell positions against the references'
        # footprints before the single warp.
        if self.params.get("refine_alignment_with_footprints", False) and self.remap.success:
            self.remap = self.refine_remap_with_footprints(
                self.remap, alignment_references, use_optical_flow=use_optical_flow
            )

        # During INITIAL loading these footprints are still raw, so
        # applying the transform once is correct.
        self.footprints = self.remap.apply_remap(
            self.footprints,
            use_optical_flow=(use_optical_flow),
        )

        self.background = self.remap.apply_remap(
            self.corrected_template(),
            use_optical_flow=(use_optical_flow),
        )

        self.postprocess_spatial_data()

    def evaluate_alignment_status(self):

        if not self.status["spatial_loaded"]:
            self.status["aligned"] = False
            return

        if self.remap is None:
            self.status["aligned"] = False
            return

        self.status["aligned"] = self.remap.report.success

    ### ================================================================== ###
    ### ===================== KERNEL DENSITY ESTIMATE ==================== ###
    ### ================================================================== ###

    # def get_idx_kde(self, params=None, qtl=[0.05, 0.95]):
    #     """
    #     function to calculate kernel density estimate of neuron density in session s
    #     this is optional, but can be used to exclude highly dense and highly sparse regions from statistics in order to not skew statistics

    #     """

    #     if not self.use_kde:
    #         self.idx_kde = np.ones(self.n_neurons, dtype=bool)
    #         return

    #     if self.centroids is None:
    #         raise ValueError(
    #             "Centroids must be calculated before calculating kernel density estimate."
    #         )

    #     from scipy import stats

    #     params = params or self.params
    #     # self.log.info("calculating kernel density estimates for session %d" % s)

    #     ## calculating kde from center of masses
    #     x_grid, y_grid = np.meshgrid(
    #         *[np.linspace(0, dim * params.get("pxtomu", 1.0), dim) for dim in self.dims]
    #     )

    #     positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
    #     kde = stats.gaussian_kde(self.centroids[self.included, :].T)
    #     kde_kernel = np.reshape(kde(positions), x_grid.shape)

    #     cm_px = (self.centroids[self.included, :] / params.get("pxtomu", 1.0)).astype(
    #         "int"
    #     )
    #     kde_at_com = np.zeros(self.n_neurons) * np.nan
    #     kde_at_com[self.included] = kde_kernel[cm_px[:, 1], cm_px[:, 0]]
    #     self.idx_kde = (kde_at_com > np.quantile(kde_kernel, qtl[0])) & (
    #         kde_at_com < np.quantile(kde_kernel, qtl[1])
    #     )
