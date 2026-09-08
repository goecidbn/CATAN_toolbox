from typing import Dict, List, Optional, Tuple, Any, Literal

import numpy as np
from scipy import sparse
from pathlib import Path

from .remap import Remapping

from catan.core.io import load_file, save_file, NATIVE_SESSION_CONFIG
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
    # traces
    _traces: dict[str, np.ndarray] = {}
    _default_trace: Optional[str] = None
    # other
    quality: dict[str, np.ndarray] = {}  #

    ## to be calculated (from input)
    remap: Optional[Remapping] = None  #
    n_neurons: int = -1  #
    centroids: Optional[np.ndarray] = None  #
    idx_eval: Optional[np.ndarray] = None  #
    ## to be calculated (with additional information)
    idx_kde: np.ndarray

    HDF5_VERSION = 1

    def __init__(
        self,
        name: Optional[str] = None,
        alignment_template: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """ """

        self.name = name
        # self.id = kwargs.get("id", -1)
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

        self.set_parameters(**kwargs)

        ## if some elements are provided in kwargs, which fit
        ## the general fields to be loaded, register them
        self.register_data(alignment_template=alignment_template, **kwargs)

        if alignment_template is None:
            self.remap = kwargs.get("remap", None)
        self.evaluate_alignment_status()

    def set_parameters(self, **input):

        self.params = {
            "pxtomu": 1.0,  # transformation from pixels to microns
            # alignment parameters
            "max_session_shift": 50.0,
            "min_session_correlation": 0.3,
            "min_session_correlation_zscore": 4.0,
            # kde parameters
            "use_kde": False,
            "qtl": [0.05, 0.95],
        }

        # update with input parameters if provided
        for key in self.params:
            if key in input:
                self.params[key] = input[key]

    ### ========================================================= ###
    ### ================= LOAD / SAVE METHODS =================== ###
    ### ========================================================= ###

    @staticmethod
    def _from_file(
        path: str | Path,
        fields_to_load: dict[str, dict[str, FieldSpec]] | None = None,
        alignment_template: Optional[np.ndarray] = None,
    ) -> "SessionData":
        if fields_to_load is None:
            fields_to_load = LoadConfig.fields_from_resource(
                NATIVE_SESSION_CONFIG, enabled_only=False
            )

        data = load_file(path, fields_to_load)
        return SessionData._from_dict(data, alignment_template=alignment_template)

    @staticmethod
    def _from_dict(
        data: dict, alignment_template: Optional[np.ndarray] = None
    ) -> "SessionData":
        session = SessionData(alignment_template=alignment_template, **data)
        # session.register_data(alignment_template=alignment_template, **data)
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

        data = load_file(self.path, fields_to_load)
        self.register_data(
            alignment_template=kwargs.get("alignment_template", None), **data
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

    def register_data(self, alignment_template: Optional[np.ndarray] = None, **data):
        """
        Registers data from kwargs 'data' input to SessionData object. Requires 'data' to contain the keys 'spatial', 'traces', and 'quality' with the corresponding data keys to be registered.

        If alignment_template is provided, spatial data will be aligned to it.
        """

        self.register_spatial(
            alignment_template=alignment_template, **data.get("spatial", {})
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

    # def copy(self) -> "SessionData":
    #     """
    #     Creates a copy of the SessionData object. Note that this is a shallow copy; mutable attributes will still reference the same objects.
    #     """
    #     new_session = SessionData(
    #         name=self.name,
    #         alignment_template=None,  # Do not align to any template in the copy
    #         path=self.path,
    #         id=self.id,
    #         source_type=self.source_type,
    #         source_config=self.source_config,
    #         active=self.active,
    #         time_offset=self.time_offset,
    #         session_color=self.session_color,
    #         use_kde=self.use_kde,
    #     )
    #     new_session.status = self.status.copy()
    #     new_session.dims = self.dims
    #     new_session.footprints = self.footprints.copy() if self.footprints is not None else None
    #     new_session.background = self.background.copy() if self.background is not None else None
    #     new_session._traces = {k: v.copy() for k, v in self._traces.items()}
    #     new_session._default_trace = self._default_trace
    #     new_session.quality = {k: v.copy() for k, v in self.quality.items()}
    #     new_session.remap = self.remap  # Assuming Remapping is immutable or handled elsewhere
    #     new_session.n_neurons = self.n_neurons
    #     new_session.centroids = self.centroids.copy() if self.centroids is not None else None
    #     new_session.idx_eval = self.idx_eval.copy() if self.idx_eval is not None else None
    #     new_session.idx_kde = self.idx_kde.copy() if hasattr(self, 'idx_kde') and self.idx_kde is not None else None

    #     return new_session

    ### ========================================================== ###
    ### ==================== QUALITY METHODS ===================== ###
    ### ========================================================== ###

    def register_quality(self, **data):

        if not data:
            return

        self.quality = data if data else {}

        if self.quality:
            self.status["quality_loaded"] = True
        self.get_idx_eval_from_quality_params()

    def get_idx_eval_from_quality_params(self, component_quality=None):
        """
        function to create idx_eval boolean array based on component quality thresholds
        defined in self.params

        requires:
            * self.params containing SNR, rval, cnn thresholds

        returns:
            * idx_eval boolean array
        """
        if not self.status["quality_loaded"] or self.quality is None:
            # print("no quality info provided, skipping quality-based filtering")
            return

        # if self.n_neurons is None:
        #     self.n_neurons = self.quality["SNR_comp"].shape[0]
        assert (
            isinstance(self.n_neurons, int) and self.n_neurons > 0
        ), "n_neurons must be a positive integer"

        if self.idx_eval is not None:
            ## dont overwrite if idx_eval already exists
            return

        # self.idx_eval = np.ones(self.n_neurons, dtype=bool)

        ## provide dummy values if not provided
        SNR_comp = self.quality.get("SNR_comp", np.full(self.n_neurons, np.inf))
        r_values = self.quality.get("r_values", np.full(self.n_neurons, np.inf))
        cnn_preds = self.quality.get("cnn_preds", np.full(self.n_neurons, np.inf))

        component_quality = (
            component_quality or component_quality_default
        )  # if not provided, use default thresholds

        idx_eval = np.ones(self.n_neurons, dtype=bool)
        ## all components must pass 'lowest' threshold...
        idx_eval &= SNR_comp >= component_quality["SNR_lowest"]
        idx_eval &= r_values >= component_quality["rval_lowest"]
        idx_eval &= cnn_preds >= component_quality["cnn_lowest"]
        ## ... and at least pass one 'min' threshold
        idx_eval &= (
            (SNR_comp >= component_quality["SNR_min"])
            | (r_values >= component_quality["rval_min"])
            | (cnn_preds >= component_quality["cnn_min"])
        )
        self.idx_eval &= idx_eval

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

    def register_spatial(self, alignment_template: Optional[np.ndarray] = None, **data):

        if "footprints" not in data or data["footprints"] is None:
            print("No footprints provided, skipping spatial registration.")
            return

        self.footprints = data.get("footprints", sparse.csc_matrix((0, 0)))
        self.background = data.get("background", None)

        if self.footprints is None:
            return

        self.status["spatial_loaded"] = True

        self.dims = (
            data.get("dims", self.dims)
            if self.background is None
            else self.background.shape
        )  # assert dims is not None, "Either background or dims must be provided to prepare_background"

        footprints_proj = self.footprints.sum(axis=1).reshape(self.dims)
        if self.background is None:
            ## return projection image if no background available
            self.background = np.array(footprints_proj).astype(np.float32)
        else:
            ## check if footprints and background are consistent (e.g. transposition) and adjust if needed
            # print("testing for transpose of background relative to footprints...")
            remap = Remapping(
                template=footprints_proj,
                template_reference=self.background,
                use_optical_flow=False,
                evaluate=False,
            )
            remap.test_transpose(footprints_proj, self.background)
            self.background = remap.fix_transpose(self.background)

        if alignment_template is not None:
            # print("align to reference template")
            self.align_to_reference(
                alignment_template, use_optical_flow=False
            )  # includes a call to postprocess_spatial_data()
        else:
            self.postprocess_spatial_data()

        self.evaluate_alignment_status()

    def postprocess_spatial_data(self):
        if self.footprints is None:
            raise ValueError(
                "Spatial data (footprints and background) must be loaded before postprocessing."
            )

        self.n_neurons = self.footprints.get_shape()[1]

        self.centroids = center_of_mass(
            self.footprints, *self.dims, convert=self.params.get("pxtomu", 1.0)
        )
        self.get_idx_eval_from_footprints()
        self.get_idx_kde()

    def get_idx_eval_from_footprints(self, footprints_thr=10):
        """
        function to create idx_eval boolean array based on component size thresholds

        requires:
            * self.footprints containing spatial footprints

        returns:
            * idx_eval boolean array
        """

        if not self.status["spatial_loaded"] or self.footprints is None:
            # print(
            #     "Spatial data must be loaded before calculating idx_eval from sizes."
            # )
            return
        ## finding non-empty rows in sparse array (https://mike.place/2015/sparse/)
        # idx_eval = np.ones(nA, bool)
        # idx_eval = np.diff(footprints.indptr) != 0

        if self.idx_eval is not None:
            ## dont overwrite if idx_eval already exists
            return

        self.idx_eval = np.ones(self.n_neurons, dtype=bool)

        ## only footprints above a certain size should be considered for evaluation
        idx_eval = self.footprints.getnnz(axis=0) > footprints_thr
        self.idx_eval &= idx_eval

    def _clean_spatial(self):
        self.dims = (512, 512)
        self.footprints = sparse.csc_matrix((0, 0))
        self.background = None
        self.idx_eval = None
        self.status["spatial_loaded"] = False

    ### ========================================================= ###
    ### ==================== ALIGNMENT METHODS ================== ###
    ### ========================================================= ###

    def align_to_reference(self, alignment_template, use_optical_flow=True):
        """
        function to align this session to a reference session based on centroids of footprints

        requires:
            * reference_data with centroids

        returns:
            * remap dict with keys 'shift' and 'idx_ref' for each neuron in this session
        """

        if (
            not self.status["spatial_loaded"]
            or self.footprints is None
            or self.background is None
        ):
            raise ValueError("Spatial data must be loaded before alignment.")

        ## first, calculate remap structure
        self.remap = Remapping(
            template=self.background,
            template_reference=alignment_template,
            use_optical_flow=use_optical_flow,
            # self.footprints.sum(axis=1).reshape(self.dims),
            # reference=alignment_template,
            # use_optical_flow=use_optical_flow,
        )
        # print("shift:", self.remap.shift)
        self.footprints = self.remap.apply_remap(
            self.footprints, use_optical_flow=use_optical_flow
        )
        self.background = self.remap.apply_remap(
            self.background, use_optical_flow=use_optical_flow
        )

        self.postprocess_spatial_data()

    def evaluate_alignment_status(self, params=None):
        """
        checks if session alignment passes certain criteria to
        be included in the further analysis
        """

        # print(f"Evaluating alignment status for session {self.name}...")
        self.status["aligned"] = False
        if not self.status["spatial_loaded"]:
            # assert self.status["spatial_loaded"], "Spatial data must be loaded before evaluating alignment status."
            return

        if self.remap is None:
            ## if no remapping was done, assume this is the first session (and include it!)
            self.status["aligned"] = True
            return

        params = params or self.params
        max_shift = params.get("max_session_shift", 50.0)
        min_corr = params.get("min_session_correlation", 0.1)
        min_zscore = params.get("min_session_correlation_zscore", 4.0)

        ## check if data can be loaded properly
        # print("Checking if session data can be loaded from path:", self.path)
        # if not Path(self.path).exists():
        #     return False

        ## check for coherence with other sessions (low shift, high correlation)
        if self.remap.shift is None:
            return
        abs_shift = np.sqrt(self.remap.shift[0] ** 2 + self.remap.shift[1] ** 2)
        if np.isnan(abs_shift) or (abs_shift > max_shift):
            return  ## huge shift

        if self.remap.c_max is None:
            return
        if (
            np.all(np.isnan(self.remap.c_max))
            or np.nanmedian(self.remap.c_max) < min_corr
        ):
            return

        if self.remap.c_zscored is None:
            return
        if (
            np.all(np.isnan(self.remap.c_zscored))
            or np.nanmedian(self.remap.c_zscored) < min_zscore
        ):
            return

        self.status["aligned"] = True

    ### ================================================================== ###
    ### ===================== KERNEL DENSITY ESTIMATE ==================== ###
    ### ================================================================== ###

    def get_idx_kde(self, params=None, qtl=[0.05, 0.95]):
        """
        function to calculate kernel density estimate of neuron density in session s
        this is optional, but can be used to exclude highly dense and highly sparse regions from statistics in order to not skew statistics

        """

        if not self.use_kde:
            self.idx_kde = np.ones(self.n_neurons, dtype=bool)
            return

        if self.centroids is None:
            raise ValueError(
                "Centroids must be calculated before calculating kernel density estimate."
            )

        from scipy import stats

        params = params or self.params
        # self.log.info("calculating kernel density estimates for session %d" % s)

        ## calculating kde from center of masses
        x_grid, y_grid = np.meshgrid(
            *[np.linspace(0, dim * params.get("pxtomu", 1.0), dim) for dim in self.dims]
        )

        positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
        kde = stats.gaussian_kde(self.centroids[self.idx_eval, :].T)
        kde_kernel = np.reshape(kde(positions), x_grid.shape)

        cm_px = (self.centroids[self.idx_eval, :] / params.get("pxtomu", 1.0)).astype(
            "int"
        )
        kde_at_com = np.zeros(self.n_neurons) * np.nan
        kde_at_com[self.idx_eval] = kde_kernel[cm_px[:, 1], cm_px[:, 0]]
        self.idx_kde = (kde_at_com > np.quantile(kde_kernel, qtl[0])) & (
            kde_at_com < np.quantile(kde_kernel, qtl[1])
        )
