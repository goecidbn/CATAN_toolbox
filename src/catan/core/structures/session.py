from typing import Dict, List, Optional, Tuple, Any, Literal

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
from catan.core.data import center_of_mass

sessiondata_type = Literal["spatial", "traces", "quality"]

component_quality_default = {
    "SNR_lowest": 1.0,
    "SNR_min": 2.5,
    "rval_lowest": -1,
    "rval_min": 0.8,
    "cnn_lowest": 0.1,
    "cnn_min": 0.9,
}


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

        self.set_parameters(**kwargs)

        ## if some elements are provided in kwargs, which fit
        ## the general fields to be loaded, register them
        self.register_data(alignment_references=alignment_references, **kwargs)

        if alignment_references is None and kwargs.get("remap") is not None:
            self.remap = kwargs["remap"]

        self.evaluate_alignment_status()

    def set_parameters(self, **input):

        self.params = {
            "pxtomu": 1.0,  # transformation from pixels to microns
            # alignment parameters
            "max_session_shift": 50.0,
            "min_session_correlation": 0.3,
            "min_session_correlation_zscore": 4.0,
            "max_session_rotation": 10.0,
            "correct_rotation": False,
            "rotation_step": 1.0,
            "rotation_refine_step": 0.1,
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
        if self.path is None or not Path(self.path).exists():
            raise ValueError("No (valid) path provided for session, cannot load data.")

        if fields_to_load is None:
            fields_to_load = LoadConfig.fields_from_resource(
                NATIVE_SESSION_CONFIG, enabled_only=False
            )

        data = load_fields_from_sources(self.path, fields_to_load)
        self.register_data(
            alignment_references=kwargs.get("alignment_references", None), **data
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

    def register_data(self, alignment_references: Optional[np.ndarray] = None, **data):
        """
        Registers data from kwargs 'data' input to SessionData object. Requires 'data' to contain the keys 'spatial', 'traces', and 'quality' with the corresponding data keys to be registered.

        If alignment_references is provided, spatial data will be aligned to it.
        """

        self.register_spatial(
            alignment_references=alignment_references, **data.get("spatial", {})
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

    def register_spatial(self, alignment_references=None, **data):

        if "footprints" not in data or data["footprints"] is None:
            return

        self.footprints = data["footprints"]

        loaded_background = data.get("background")

        self.status["spatial_loaded"] = True

        self.dims = (
            data.get("dims", self.dims)
            if loaded_background is None
            else loaded_background.shape
        )

        footprints_proj = self.footprints.sum(axis=1).reshape(self.dims)

        if loaded_background is None:
            self.background_origin = "footprints"

            template = np.asarray(footprints_proj, dtype=np.float32)

        else:
            self.background_origin = "loaded"

            template = self._prepare_background(loaded_background)

            # Source-orientation correction only.
            orientation = Remapping(evaluate=False)

            orientation.test_transpose(footprints_proj, template)

            template = orientation.fix_transpose(template)

        # THIS COPY MUST NEVER BE ALTERED BY CROSS-SESSION ALIGNMENT.
        self.background_template = np.asarray(template, dtype=np.float32).copy()

        self.background = self.background_template.copy()

        if alignment_references:
            self.align_to_reference(alignment_references, use_optical_flow=False)

        else:
            self.remap = Remapping.identity(self.dims)

            self.postprocess_spatial_data()

        self.evaluate_alignment_status()

    # def register_spatial(self, alignment_references: Optional[np.ndarray] = None, **data):

    #     if "footprints" not in data or data["footprints"] is None:
    #         # print("No footprints provided, skipping spatial registration.")
    #         return

    #     self.footprints = data.get("footprints", sparse.csc_matrix((0, 0)))
    #     background = data.get("background", None)

    #     if background is not None and background.ndim != 2:
    #         raise ValueError(
    #             "Spatial background must be a 2D array. "
    #             f"Loaded shape: {background.shape}. "
    #             "Select or preprocess a single image plane/channel."
    #         )

    #     self.background = (
    #         None if background is None else self._prepare_background(background)
    #     )

    #     if self.footprints is None:
    #         return

    #     self.status["spatial_loaded"] = True

    #     self.dims = (
    #         data.get("dims", self.dims)
    #         if self.background is None
    #         else self.background.shape
    #     )  # assert dims is not None, "Either background or dims must be provided to prepare_background"

    #     footprints_proj = self.footprints.sum(axis=1).reshape(self.dims)
    #     if self.background is None:
    #         ## return projection image if no background available
    #         self.background_origin = "footprints"
    #         # self.background = np.array(footprints_proj).astype(np.float32)
    #         self.background = self._prepare_background(np.asarray(footprints_proj))
    #     else:
    #         ## check if footprints and background are consistent (e.g. transposition) and adjust if needed
    #         # print("testing for transpose of background relative to footprints...")
    #         self.background_origin = "loaded"
    #         remap = Remapping(
    #             template=footprints_proj,
    #             template_reference=self.background,
    #             use_optical_flow=False,
    #             evaluate=False,
    #         )
    #         remap.test_transpose(footprints_proj, self.background)
    #         self.background = remap.fix_transpose(self.background)

    #     if alignment_references is not None:
    #         # print("align to reference template")
    #         self.align_to_reference(
    #             alignment_references, use_optical_flow=False
    #         )  # includes a call to postprocess_spatial_data()
    #     else:
    #         self.postprocess_spatial_data()

    #     # self._ensure_component_flags(
    #     #     included=data.get("included"),
    #     #     synthetic=data.get("synthetic"),
    #     # )

    #     self.evaluate_alignment_status()

    @staticmethod
    def _prepare_background(
        background: np.ndarray,
    ) -> np.ndarray:

        background = np.asarray(background, dtype=np.float32)

        if background.ndim != 2:
            raise ValueError(
                "Spatial background must be a 2D array; "
                f"got shape {background.shape}."
            )

        if not np.all(np.isfinite(background)):
            raise ValueError("Spatial background contains NaN or Inf values.")

        lo = float(background.min())
        hi = float(background.max())

        if hi > lo:
            background = (background - lo) / (hi - lo)

        else:
            # Constant image: valid array, but carries no
            # useful intensity information for alignment.
            background = np.zeros_like(background, dtype=np.float32)

        return background

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
                self.footprints, *self.dims, convert=self.params.get("pxtomu", 1.0)
            )
        elif mode == "append":
            n_new = footprints.get_shape()[1]
            self.footprints = sparse.hstack([self.footprints, footprints], format="csc")
            centroids = center_of_mass(
                footprints, *self.dims, convert=self.params.get("pxtomu", 1.0)
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
            self.footprints, *self.dims, convert=self.params.get("pxtomu", 1.0)
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
        self.dims = (512, 512)
        self.footprints = sparse.csc_matrix((0, 0))
        self.background = None
        self.background_origin = None
        self.background_template = None
        self.remap = None
        self.status["aligned"] = False
        self.included = np.array([], dtype=bool)
        self.status["spatial_loaded"] = False

    ### ========================================================= ###
    ### ================== ALTERATION METHODS =================== ###
    ### ========================================================= ###

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
            *self.dims,
            convert=self.params.get("pxtomu", 1.0),
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

    def prepare_background_template(self, background: np.ndarray) -> np.ndarray:
        """
        Prepare a newly loaded background for use as an
        unaligned session background template.

        This does not modify SessionData.
        """

        template = np.asarray(background, dtype=np.float32)
        if template.ndim != 2:
            raise ValueError(
                f"Background must be a 2D image, got shape {template.shape}."
            )

        dims = tuple(self.dims)

        # Non-square data allow us to determine the
        # orientation directly from the shape.
        if template.shape != dims:

            if template.T.shape == dims:
                template = template.T
            else:
                raise ValueError(
                    f"Background has incompatible shape {template.shape}; expected {dims}."
                )

        # For square images, shape cannot tell us whether
        # the image is transposed. The current unaligned
        # template provides the orientation reference.
        elif dims[0] == dims[1] and self.background_template is not None:
            orientation = Remapping(evaluate=False)
            orientation.test_transpose(self.background_template, template)

            template = orientation.fix_transpose(template)

        return template.copy()

    def propose_remapping(
        self,
        alignment_references,
        *,
        background_template=None,
        use_optical_flow=False,
        correct_rotation: bool | None = None,
    ) -> Remapping:
        """
        Calculate a candidate remapping without
        modifying any SessionData state.

        If background_template is omitted, the
        session's current unaligned template is used.
        """

        template = (
            self.background_template
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
        )

    def align_to_reference(self, alignment_references, use_optical_flow=False):

        if (
            not self.status["spatial_loaded"]
            or self.footprints is None
            or self.background_template is None
        ):
            raise ValueError("Spatial data must be loaded before alignment.")

        self.remap = self.propose_remapping(
            alignment_references,
            use_optical_flow=use_optical_flow,
        )

        # During INITIAL loading these footprints are still raw, so
        # applying the transform once is correct.
        self.footprints = self.remap.apply_remap(
            self.footprints,
            use_optical_flow=(use_optical_flow),
        )

        self.background = self.remap.apply_remap(
            self.background_template,
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
