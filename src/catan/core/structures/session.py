from typing import Dict, Optional, Tuple

import inspect
import h5py
import numpy as np
from scipy import sparse
from pathlib import Path

from .remap import Remapping

from catan.core.data import center_of_mass
from catan.core.io import load_data, write_sparse_matrix, write_optional_attr, write_optional_array, read_sparse_matrix, read_optional_attr, read_optional_array

component_quality_default = {
    "SNR_lowest": 1.0,
    "SNR_min": 2.5,
    "rval_lowest": -1,
    "rval_min": 0.8,
    "cnn_lowest": 0.1,
    "cnn_min": 0.9,
}


def has_property(class_instance, property_name: str) -> bool:
    return property_name in [
        key for (key, _) in inspect.getmembers_static(class_instance)
    ]


def cast_dict_to_class_attributes(class_instance, dict, exclude_keys=[]):
    for key, value in dict.items():
        if key in exclude_keys:
            continue

        if has_property(class_instance, key):
            setattr(class_instance, key, value)
        # else:
        #     raise ValueError(f"Unknown field {key} for session_data")


class SessionData:
    ## meta data
    name: Optional[str] = None  #
    path: Optional[str] = None  #
    id: int = -1    #

    active: bool = True #
    time_offset: float = 0.0    #
    session_color: Optional[str] = None
    use_kde: bool = False

    # scheduled_for_loading: bool = False
    status: Dict[str, bool] = {}

    ## loaded fields (from input)
    # spatial
    _spatial_fields = ["dims", "A", "Cn"]
    dims: Tuple[int, int] = (512, 512)  #
    A: sparse.csc_matrix  # = None  #
    Cn: Optional[np.ndarray] = None #
    # temporal
    _trace_fields = ["C", "F_dff", "F_dff_dec", "S", "S_dff"]
    _traces: dict[str, np.ndarray] = {}
    _default_trace: Optional[str] = None
    # other
    _quality_fields = ["SNR_comp", "r_values", "cnn_preds"]
    quality: dict[str, np.ndarray] = {} #

    ## to be calculated (from input)
    remap: Optional[Remapping] = None   #
    n_neurons: int = -1 #
    centroids: np.ndarray  #
    idx_eval: np.ndarray  #
    ## to be calculated (with additional information)
    idx_kde: np.ndarray

    HDF5_VERSION = 1

    def __init__(
        self,
        name="",
        alignment_template: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """ """
        self.fields_scheduled_for_loading = {}

        self.name = name
        self.id = kwargs.get("id", -1)
        self.path = kwargs.get("path", None)

        self.status = {
            # load status flags
            "scheduled_spatial": False,
            "scheduled_traces": False,
            "scheduled_quality": False,
            "spatial_loaded": False,
            "traces_loaded": False,
            "quality_loaded": False,
            # processing status flags
            "aligned": False,
            "matched": False,
        }

        self.set_parameters(**kwargs)

        if np.isin(self._spatial_fields, list(kwargs.keys())).any():
            self.schedule_spatial_load()
            self.register_spatial(alignment_template=alignment_template, **kwargs)

        if np.isin(self._trace_fields, list(kwargs.keys())).any():
            self.schedule_traces_load()
            self.register_traces(**kwargs)

        if np.isin(self._quality_fields, list(kwargs.keys())).any():
            self.schedule_quality_load()
            self.register_quality(**kwargs)

        self.remap = kwargs.get("remap", None)
        if self.remap is not None:
            self.status["aligned"] = True

    @staticmethod
    def from_dict(data_dict: dict, params={}, **kwargs) -> "SessionData":
        alignment_template = kwargs.get("alignment_template", None)

        out = SessionData(
            params=params,
            alignment_template=alignment_template,
        )
        out.register_traces(**data_dict)
        out.register_spatial(alignment_template=alignment_template, **data_dict)
        out.register_quality(**data_dict)
        return out

    @staticmethod
    def from_file(
        path: str, which=["quality", "spatial", "temporal"], params={}
    ) -> "SessionData":
        out = SessionData(params=params)
        out.path = path
        out.load_data(which=which)
        return out

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

    ### ============================================================================== ###
    ### ================================ LOAD METHODS ================================ ###
    ### ============================================================================== ###

    def load_data(
        self,
        which=["quality", "spatial", "temporal"],
        alignment_template: Optional[np.ndarray] = None,
    ):
        if "spatial" in which:
            self.schedule_spatial_load()
        if "temporal" in which:
            self.schedule_traces_load()
        if "quality" in which:
            self.schedule_quality_load()

        self.execute_load(alignment_template=alignment_template)

    def execute_load(self, alignment_template: Optional[np.ndarray] = None):

        assert (
            self.path is not None
        ), "No path provided for session, cannot reload traces."
        assert Path(
            self.path
        ).exists(), f"Path {self.path} does not exist, cannot reload traces."
        if len(self.fields_scheduled_for_loading) == 0:
            # print("No fields scheduled for loading.")
            return

        data = load_data(
            self.path,
            fields=[
                field
                for sublist in self.fields_scheduled_for_loading.values()
                for field in sublist
            ],
            subpath="/estimates",
        )

        if self.status["scheduled_spatial"]:
            self.register_spatial(alignment_template=alignment_template, **data)
            self.status["scheduled_spatial"] = False

        if self.status["scheduled_traces"]:
            self.register_traces(**data)
            self.status["scheduled_traces"] = False

        if self.status["scheduled_quality"]:
            self.register_quality(**data)
            self.status["scheduled_quality"] = False


    def schedule_spatial_load(self):
        if not self.status["scheduled_spatial"]:
            self.fields_scheduled_for_loading["spatial"] = ["A", "Cn"]
            self.status["scheduled_spatial"] = True

    def schedule_traces_load(self, fields=["C", "F_dff", "F_dff_dec", "S", "S_dff"]):
        if not self.status["scheduled_traces"]:
            self.fields_scheduled_for_loading["temporal"] = fields
            self.status["scheduled_traces"] = True

    def schedule_quality_load(self, fields=["SNR_comp", "r_values", "cnn_preds"]):
        if not self.status["scheduled_quality"]:
            self.fields_scheduled_for_loading["quality"] = fields
            self.status["scheduled_quality"] = True

    def clean_data(self):

        self.clean_traces()
        self.clean_spatial()
        self.clean_quality()

        self.idx_eval = np.bool([])

    ### ============================================================================= ###
    ### ============================= QUALITY METHODS =============================== ###
    ### ============================================================================= ###

    def register_quality(self, **data):

        self.quality = {
            key: data[key]
            for key in data
            if key in self.fields_scheduled_for_loading.get("quality", [])
        }

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
            print("no quality info provided, skipping quality-based filtering")
            return

        # if self.n_neurons is None:
        #     self.n_neurons = self.quality["SNR_comp"].shape[0]
        assert (
            isinstance(self.n_neurons, int) and self.n_neurons > 0
        ), "n_neurons must be a positive integer"

        # if self.idx_eval is None:
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

    def clean_quality(self):
        self.quality = {}
        self.status["quality_loaded"] = False
        self.status["scheduled_quality"] = False

    ### ============================================================================== ###
    ### ================================= TRACE METHODS ============================== ###
    ### ============================================================================== ###

    @property
    def traces(self):
        return self._traces

    @property
    def trace(self):
        if self._default_trace is None:
            return None
        return self._traces.get(self._default_trace, None)

    # @traces.setter
    # def traces(self, value):
    #     if not isinstance(value, dict):
    #         raise ValueError(
    #             "Traces must be provided as a dictionary with keys corresponding to trace types (e.g., 'F_dff', 'C')."
    #         )
    #     self._traces = value
    #     self._default_trace = "F_dff" if "F_dff" in self._traces else "C"

    def clean_traces(self):
        # print("Cleaning traces for session. Current traces:", self._traces.keys())
        self._traces = {}
        self._default_trace = None
        self.status["traces_loaded"] = False
        self.status["scheduled_traces"] = False

    def register_traces(self, **data):
        self._traces = {
            key: data[key]
            for key in data
            if key in self.fields_scheduled_for_loading.get("temporal", [])
        }
        self._default_trace = "F_dff" if "F_dff" in self._traces else "C"
        if self._traces:
            self.status["traces_loaded"] = True

    ### ============================================================================== ###
    ### ================================ SPATIAL METHODS ============================= ###
    ### ============================================================================== ###

    def register_spatial(self, alignment_template: Optional[np.ndarray] = None, **data):
        self.dims = data.get("dims", self.dims)
        self.A = data.get("A", sparse.csc_matrix((0, 0)))
        self.Cn = data.get("Cn", None)

        if self.A is not None:
            self.status["spatial_loaded"] = True

        # dims = Cn.shape if Cn is not None else dims
        # assert dims is not None, "Either Cn or dims must be provided to prepare_background"

        A_proj = self.A.sum(axis=1).reshape(self.dims)
        if self.Cn is None:
            ## return projection image if no Cn available
            self.Cn = np.array(A_proj).astype(np.float32)
        else:
            ## check if A and Cn are consistent (e.g. transposition) and adjust if needed
            # print("testing for transpose of Cn relative to A...")
            remap = Remapping(template=A_proj, template_reference=self.Cn, use_optical_flow=False, evaluate=False)
            remap.test_transpose(A_proj, self.Cn)
            self.Cn = remap.fix_transpose(self.Cn)

        if alignment_template is not None:
            # print("align to reference template")
            self.align_to_reference(
                alignment_template, use_optical_flow=False
            )  # includes a call to postprocess_spatial_data()
        else:
            self.postprocess_spatial_data()

        self.evaluate_alignment_status()

    def postprocess_spatial_data(self):
        if self.A is None:
            raise ValueError(
                "Spatial data (A and Cn) must be loaded before postprocessing."
            )

        self.n_neurons = self.A.get_shape()[1]

        self.centroids = center_of_mass(
            self.A, *self.dims, convert=self.params.get("pxtomu", 1.0)
        )
        self.get_idx_eval_from_footprints()
        self.get_idx_kde()

    def get_idx_eval_from_footprints(self, A_thr=10):
        """
        function to create idx_eval boolean array based on component size thresholds

        requires:
            * self.A containing spatial footprints

        returns:
            * idx_eval boolean array
        """

        if not self.status["spatial_loaded"] or self.A is None:
            raise ValueError(
                "Spatial data must be loaded before calculating idx_eval from sizes."
            )
        ## finding non-empty rows in sparse array (https://mike.place/2015/sparse/)
        # idx_eval = np.ones(nA, bool)
        # idx_eval = np.diff(A.indptr) != 0

        # if self.idx_eval is None:
        self.idx_eval = np.ones(self.n_neurons, dtype=bool)

        ## only footprints above a certain size should be considered for evaluation
        idx_eval = self.A.getnnz(axis=0) > A_thr
        self.idx_eval &= idx_eval

    def clean_spatial(self):
        self.dims = (512, 512)
        self.A = sparse.csc_matrix((0, 0))
        self.Cn = None
        self.status["spatial_loaded"] = False
        self.status["scheduled_spatial"] = False


    ### =============================================================================== ###
    ### ================================ SAVE METHODS ================================= ###
    ### =============================================================================== ###
    
    def to_hdf5(self, group: h5py.Group, exclude_fields=["traces"]) -> None:
        group.attrs["object_type"] = "SessionData"
        group.attrs["schema_version"] = self.HDF5_VERSION

        write_optional_attr(group, "name", self.name)
        write_optional_attr(group, "path", str(self.path) if self.path is not None else None)
        group.attrs["id"] = self.id

        group.attrs["active"] = self.active
        group.attrs["time_offset"] = self.time_offset

        ## spatial group
        group.attrs["dims"] = self.dims

        A_group = group.create_group("A")
        write_sparse_matrix(A_group, self.A)

        write_optional_array(group, "Cn", self.Cn, compression="gzip")

        ## trace group
        if "traces" not in exclude_fields:
            traces_group = group.create_group("traces")
            for key, value in self.traces.items():
                traces_group.create_dataset(key, data=value, compression="gzip")

        ## quality group
        quality_group = group.create_group("quality")
        for key, value in self.quality.items():
            write_optional_array(quality_group, key, value)

        ## remap substructure
        if self.remap is not None:
            remapping_group = group.create_group("remapping")
            self.remap.to_hdf5(remapping_group)

        write_optional_array(group, "idx_eval", self.idx_eval, compression="gzip")


    @classmethod
    def from_hdf5(cls, group: h5py.Group | h5py.File) -> "SessionData":

        version = int(group.attrs.get("schema_version", 1))
        if version != 1:
            raise ValueError(
                f"Unsupported SessionData schema version: {version}"
            )

        name = read_optional_attr(group, "name")
        path = read_optional_attr(group, "path")
        id = int(group.attrs.get("id", -1))

        active = bool(group.attrs.get("active", True))
        time_offset = float(group.attrs.get("time_offset", 0.0))

        ## spatial group
        dims = tuple(group.attrs.get("dims", (512, 512)))

        A = read_sparse_matrix(group["A"]) if "A" in group else sparse.csc_matrix((0, 0))
        Cn = read_optional_array(group, "Cn")

        ## trace group
        traces = {}
        if "traces" in group:
            traces_group = group["traces"]
            if isinstance(traces_group, h5py.Group):
                traces = {key: traces_group[key][()] for key in traces_group}

        ## quality group
        quality = {}
        if "quality" in group:
            quality_group = group["quality"]
            if isinstance(quality_group, h5py.Group):
                for key in quality_group:
                    value = read_optional_array(quality_group, key)
                    if value is not None:
                        quality[key] = value
                        
        ## remap substructure
        remap = None
        if "remapping" in group:
            remap_group = group["remapping"]
            if isinstance(remap_group, h5py.Group):
                remap = Remapping.from_hdf5(remap_group)

        idx_eval = read_optional_array(group, "idx_eval")

        out = cls(
            name=name,
            path=path,
            id=id,
            active=active,
            time_offset=time_offset,
            dims=dims,
            A=A,
            Cn=Cn,
            remap=remap,
            idx_eval=idx_eval,
            # **traces,
            **quality,
        )
        # if A is not None:
        #     out.postprocess_spatial_data()
        # if quality:
        #     out.get_idx_eval_from_quality_params()
        
        return out
            
    # def cast_to_dict(self, fields=None):

    #     if fields is None:
    #         # set default values
    #         fields = ["name","path","id","active","time_offset",*self._spatial_fields, "quality","remap","n_neurons","centroids","idx_eval"]

    #     out = {}
    #     for field in fields:
    #         assert hasattr(self,field), "SessionData object is missing field {field} for converting to dict"
    #         out[field] = getattr(self, field)
    #     return out

            
        



    ### ============================================================================== ###
    ### ================================ ALIGNMENT METHODS =========================== ###
    ### ============================================================================== ###

    def align_to_reference(self, alignment_template, use_optical_flow=True):
        """
        function to align this session to a reference session based on centroids of footprints

        requires:
            * reference_data with centroids

        returns:
            * remap dict with keys 'shift' and 'idx_ref' for each neuron in this session
        """

        if not self.status["spatial_loaded"] or self.A is None or self.Cn is None:
            raise ValueError("Spatial data must be loaded before alignment.")

        ## first, calculate remap structure
        self.remap = Remapping(
            template=self.Cn,
            template_reference=alignment_template,
            use_optical_flow=use_optical_flow,
            # self.A.sum(axis=1).reshape(self.dims),
            # reference=alignment_template,
            # use_optical_flow=use_optical_flow,
        )
        # print("shift:", self.remap.shift)
        self.A = self.remap.apply_remap(self.A, use_optical_flow=use_optical_flow)
        self.Cn = self.remap.apply_remap(self.Cn, use_optical_flow=use_optical_flow)

        self.postprocess_spatial_data()

    def evaluate_alignment_status(self, params=None):
        """
        checks if session alignment passes certain criteria to
        be included in the further analysis
        """

        # print(f"Evaluating alignment status for session {self.name}...")
        if self.remap is None:
            ## if no remapping was done, assume this is the first session (and include it!)
            self.aligned = True
            return
        params = params or self.params
        max_shift = params.get("max_session_shift", 50.0)
        min_corr = params.get("min_session_correlation", 0.3)
        min_zscore = params.get("min_session_correlation_zscore", 4.0)

        ## check if data can be loaded properly
        # print("Checking if session data can be loaded from path:", self.path)
        # if not Path(self.path).exists():
        #     return False

        ## check for coherence with other sessions (low shift, high correlation)
        if self.remap.shift is None:
            self.aligned = False
            return
        abs_shift = np.sqrt(self.remap.shift[0] ** 2 + self.remap.shift[1] ** 2)
        if np.isnan(abs_shift) or (abs_shift > max_shift):
            self.aligned = False
            return  ## huge shift

        if self.remap.c_max is None:
            self.aligned = False
            return
        if (
            np.all(np.isnan(self.remap.c_max))
            or np.nanmedian(self.remap.c_max) < min_corr
        ):
            self.aligned = False
            return

        if self.remap.c_zscored is None:
            self.aligned = False
            return
        if (
            np.all(np.isnan(self.remap.c_zscored))
            or np.nanmedian(self.remap.c_zscored) < min_zscore
        ):
            self.aligned = False
            return

        self.aligned = True

    ### ============================================================================== ###
    ### =========================== KERNEL DENSITY ESTIMATE ========================== ###
    ### ============================================================================== ###

    def get_idx_kde(self, params=None, qtl=[0.05, 0.95]):
        """
        function to calculate kernel density estimate of neuron density in session s
        this is optional, but can be used to exclude highly dense and highly sparse regions from statistics in order to not skew statistics

        """

        if not self.use_kde:
            self.idx_kde = np.ones(self.n_neurons, dtype=bool)
            return

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


# def prepare_background(
#     A, Cn=None, dims: Optional[Tuple[int, int]] = None
# ) -> np.ndarray:
#     """
#     Prepare background image from loaded data
#     - use A projection if Cn is not provided
#     - check for consistency of A and Cn (e.g. transposition) and adjust if needed
#     """
#     dims = Cn.shape if Cn is not None else dims
#     assert dims is not None, "Either Cn or dims must be provided to prepare_background"

#     import matplotlib.pyplot as plt

#     A_proj = A.sum(axis=1).reshape(
#         dims,
#     )
#     if Cn is None:
#         ## return projection image if no Cn available
#         return A_proj
#     Cn = Cn.astype(np.float32)

#     remap = Remapping(A_proj, Cn, use_optical_flow=False, evaluate=False)
#     remap.test_transpose(A_proj, Cn)
