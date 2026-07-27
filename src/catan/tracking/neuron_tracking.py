"""
function written by Alexander Schmidt, based on the paper "Sheintuch et al., ...", allowing for complete registration of neuron footprints across several sessions

TODO:
  * write plotting procedure for cluster footprints (3D), to allow manual corrections
  * save data-attribute / structure after model-building, not only after registration
  * change save structure, such that all that is needed for further analysis is readily accessible:
      - filePath of results file
      - no redundancy in SNR, r_values, cnn saving
      - 'remap' into 'alignment' structure in results
      - cm only needed once

last updated on January 28th, 2024
"""

import os
from typing import Dict, Optional, Tuple, List
from functools import partial
import sys, copy, logging, time

import h5py
from tqdm.auto import tqdm

from pathlib import Path

import numpy as np
from scipy import sparse, spatial, special
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment

from catan.core.structures import SessionData
from catan.core.utils import nangauss_filter, pad_axis
from catan.core.analysis import calculate_statistics, calculate_p
from catan.core.alignment import _shift_sparse_bilinear

from .analytics.fit_model_theoretical import (
    fit_histogram_params,
    match_model,
)

from catan.core.io import load_data, save_data

logging.basicConfig(level=logging.INFO)

class Tracking:

    union: SessionData

    HDF5_VERSION = 1

    def __init__(self, neighbor_distance=25.0, bins=64, n_threads=1, use_kde=False, pxtomu=1., L=512, logLevel=logging.ERROR):
        """
        Central class for neuron tracking via session registration, model building and neuron registration
        
        Parameters
        ----------

        neighbor_distance : float = 25.
            the distance (in mu m) up to where neuron similarities and distances are calculated and registered for model building (shouldn't be much smaller, as the distance model requires at least some further ranging data)
            
        bins : int = 64
            number of bins used for model building and count registration. Should be a number 2^n to allow scaling down.

        n_threads: int = 1
            the number of threads to use, when calculating footprint similarity - the costly part of the analysis
            
        use_kde: bool = False
            specifies, whether neurons from areas of lowest and highest neuron density are excluded when building the probabilistic model. Costly, but can avoid some weird behavior
            
        pxtomu: float = 1.
            the factor to transform pixels to micrometer (mu m) distance
        
        L: int = 512
            number of pixels along one dimension (actually - this should rather be the window length in mu m)
                    """
        self.log = logging.getLogger("matchinglogger")
        self.log.setLevel(logLevel)

        self.params = {}
        self.params["neighbor_distance"] = neighbor_distance
        self.params["bins"] = bins
        self.params["n_threads"] = n_threads
        self.params["use_kde"] = use_kde
        self.params["pxtomu"] = pxtomu
        self.params["L"] = L

        self._update_bins(bins)

        # self.kernel = {"idxes": {}, "kde": {}}
        self.reference_data = None

        self.reset_data()
        self.reset_model()
        self.reset_registration()

    def reset_data(self):
        self.sessions: List[SessionData] = []

    def reset_model(self):
        self.model = None
        self.counts = {
            "same": np.zeros((self.params["nbins"], self.params["nbins"]), int),
            "cross": np.zeros((self.params["nbins"], self.params["nbins"], 3), int),
        }

    def register_session(
        self,
        from_file: Optional[str | Path] = None,
        name: Optional[str] = None,
        load_content: List[str] = ["quality", "spatial", "temporal"],
        align=True,
    ):
        """
        Register a new session from a file, load the data and add it to the list of sessions. The session is aligned to the previous sessions if align=True.

        Input
        
        - from_file: str 
            
            Path to the session file

        - name: Optional[str]

            Optional name for this session
        
        - load_content: list[str] = ["spatial", "temporal", "quality"]

            Specifies which data should be loaded and can be either combination of the three above - but setting all is strongly encouraged.
            
            spatial: loads footprint and background data - necessary to do any kind of further processing
            traces: loads temporal traces - not entirely necessary, but is used to test for overlapping neurons with highly correlated activity 
            quality: loads quality parameters (SNR_comp, r_values, cnn_preds for CaImAn) which are used for thresholding
        
        - align: bool = True

            Flag for aligning the spatial components to prior registered sessions (using rigid and non-rigid correction). Highly encouraged to leave this enabled, unless sessions are already aligned. For further details see demo notebook `alignment.ipynb`

        """

        if isinstance(from_file, (str, Path)):
            this_data = SessionData(
                name=name if name is not None else str(from_file).split("/")[-2],
                path=str(from_file),
                id=len(self.sessions),
            )
            this_data.load_data(
                which=load_content,
                alignment_template=self.alignment_template if align else None,
            )
            self.sessions.append(this_data)

        else:
            raise ValueError(
                "from_file must be provided for session registration for now - registration from raw data to be implemented later"
            )

    @property
    def alignment_template(self):

        if len(self.sessions) == 0:
            return None

        alignment_window = min(len(self.sessions), 10)

        return np.stack(
            [
                this_data.Cn
                for this_data in self.sessions[-alignment_window:]
                if this_data.aligned
            ],
            axis=0,
        )

    def batch_update_model(
        self,
        root_path=".",
        path_glob="*/neuron_detection_*",
        s_specific=None,
        align_to_reference=True,
    ):
        paths = sorted(Path(root_path).glob(path_glob))
        self.progress = tqdm(enumerate(paths), total=len(paths))
        for s, path in self.progress:
            if s_specific is not None and s not in s_specific:
                continue
            self.progress.set_description(f"Processing session {s}, {path.name}")
            # print(f"Processing {path}...")
            self.update_model_with_data(
                from_file=path, align_to_reference=align_to_reference
            )
        self.fit_to_model()

    def batch_register_neurons(
        self,
        root_path=".",
        path_glob="*/neuron_detection_*",
        s_specific=None,
        align_to_reference=True,
    ):

        paths = sorted(Path(root_path).glob(path_glob))
        self.progress = tqdm(enumerate(paths), total=len(paths))
        for s, path in self.progress:
            if s_specific is not None and s not in s_specific:
                continue
            sz_union = self.union.n_neurons
            self.progress.set_description(
                f"Registering session {s} to {sz_union} neurons, {path.name}"
            )
            self.register_neurons(from_file=path, align_to_reference=align_to_reference)

    def update_model_with_data(
        self,
        from_file: Optional[str | Path] = None,
        from_data: Optional[SessionData] = None,
        from_session_id: Optional[int] = None,
        align_to_reference=True,
    ):
        """
        takes existing model and adds new data from footprints to it
        """
        this_data = self.from_data(
            from_file=from_file,
            from_data=from_data,
            from_session_id=from_session_id,
            align_to_reference=align_to_reference,
        )

        if not this_data.aligned:
            print(f"Session {from_file} did not pass quality criteria, skipping.")
            return

        # build both models: self and cross (nNN from self and NN from cross)
        self.update_model_counts(this_data, mode="same")
        # self.this_data = this_data

        if self.reference_data is not None:
            self.update_model_counts(this_data, mode="to_reference")

        # self.alignment_template = copy.deepcopy(this_data.Cn)
        self.reference_data = copy.deepcopy(this_data)

    def from_data(
        self,
        from_file: Optional[str | Path] = None,
        from_data: Optional[SessionData] = None,
        from_session_id: Optional[int] = None,
        align_to_reference=True,
    ) -> SessionData:
        if from_file is not None:
            assert isinstance(
                from_file, (str, Path)
            ), "from_file must be a string or Path"
            this_data = SessionData(
                path=str(from_file),
            )
            this_data.load_data(
                alignment_template=(
                    self.alignment_template if align_to_reference else None
                )
            )
        elif from_data is not None:
            assert isinstance(
                from_data, SessionData
            ), "from_data must be a SessionData instance"
            this_data = from_data
        elif from_session_id is not None:
            assert isinstance(
                from_session_id, int
            ), "from_session_id must be an integer"
            assert (
                0 <= from_session_id < len(self.sessions)
            ), "from_session_id is out of range"
            this_data = self.sessions[from_session_id]
        else:
            raise ValueError(
                "Either from_file, from_data, or from_session_id must be provided."
            )
        return this_data

    def update_model_counts(self, this_data: SessionData, mode="to_reference"):
        """
        Function to update counts in the joint model

        inputs:
        - s,s_ref: int / string
            key of current (s) and reference (s_ref) session
        - use_kde: bool
            defines, whether kde (kernel density estimation) is used to ...

            TODO:
            * might just change everything to require "total counts" only, removing NN-calculation
            * change how correlation is calculated: just apply centroid distance shift! (test performance/timing before that)
        """

        # print(this_data)
        if mode == "to_reference":
            ## compare to reference session
            ref_data = self.reference_data
        elif mode == "same":
            ## find and mark potential duplicates of neuron footprints in session
            ref_data = this_data
        else:
            raise ValueError("mode must be 'to_reference' or 'same'")
        assert isinstance(ref_data, SessionData), "Reference data not defined!"

        (
            footprint_shifts,
            footprint_distances,
            footprint_correlations,
            idx_remove,
            remove_info,
        ) = calculate_statistics(
            this_data,
            ref_data,
            distance_threshold=self.params.get("neighbor_distance", 25.0),
            nP=12,
            params=self.params,
        )

        # print(f"idx_eval before removal: {np.sum(this_data.idx_eval)}")
        if mode == "same" and len(idx_remove) > 0:
            # print(
            #     len(idx_remove),
            #     "components removed due to high intra-session correlation",
            # )
            this_data.idx_eval[idx_remove] = False

        idx_this = this_data.idx_eval
        idx_ref = ref_data.idx_eval

        ### ------------------------------------------------------------ ###
        ### --------------------- define neighbours -------------------- ###
        ### ------------------------------------------------------------ ###
        ## find all neuron pairs below a distance threshold
        neighbors = footprint_distances < self.params.get("neighbor_distance", 15.0)
        is_NN = np.zeros((ref_data.n_neurons, this_data.n_neurons), bool)
        if mode == "to_reference":
            min_distance = np.nanmin(footprint_distances, axis=1)
            idx_finite = ~np.isnan(min_distance)

            min_distance_idx = np.nanargmin(
                footprint_distances[idx_ref & idx_finite, :], axis=1
            )
            # min_distance_idx = np.nanargmin(footprint_distances, axis=1)
            is_NN[
                idx_ref & idx_finite,
                min_distance_idx,
            ] = True
        else:
            is_NN[idx_this, idx_this] = True

        # print(f"number of neighbor pairs: {np.sum(neighbors)} ({np.sum(is_NN)} NN)")
        t_start = time.time()
        histo_options = {
            "bins": self.params["nbins"],
            "range": [
                self.params["arrays"]["distance_bounds"][[0, -1]],
                self.params["arrays"]["correlation_bounds"][[0, -1]],
            ],
        }
        if mode == "same":
            idxes = neighbors & ~is_NN & ref_data.idx_kde[:, None]
            # print(idxes.sum(), "counts to add")

            self.counts["same"] += np.histogram2d(
                footprint_distances[idxes],
                # footprint_correlations["shifted"][idxes],
                footprint_correlations[idxes],
                **histo_options,
            )[0].astype(int)

        else:
            idxes = neighbors & ref_data.idx_kde[:, None]
            self.counts["cross"][..., 0] += np.histogram2d(
                footprint_distances[idxes],
                # footprint_correlations["shifted"][idxes],
                footprint_correlations[idxes],
                **histo_options,
            )[0].astype(int)

            idxes = neighbors & is_NN & ref_data.idx_kde[:, None]
            self.counts["cross"][..., 1] += np.histogram2d(
                footprint_distances[idxes],
                # footprint_correlations["shifted"][idxes],
                footprint_correlations[idxes],
                **histo_options,
            )[0].astype(int)

            idxes = neighbors & ~is_NN & ref_data.idx_kde[:, None]
            self.counts["cross"][..., 2] += np.histogram2d(
                footprint_distances[idxes],
                # footprint_correlations["shifted"][idxes],
                footprint_correlations[idxes],
                **histo_options,
            )[0].astype(int)

        t_end = time.time()
        # print(self.counts["same"].sum(), "total 'same' counts")
        # print(self.counts["cross"].sum(axis=(0, 1)), "total 'cross' counts")
        # print(f"Updating joint model took {t_end - t_start:.2f} seconds.")

    ### ------------------------------------------------------------ ###
    ### ------------------ Model fitting functions ----------------- ###
    ### ------------------------------------------------------------ ###

    # Get correlation bin edges and centers

    def fit_to_model(self, use_cdf=True):
        """
        Currently takes over h almost as provided - add weights to  improve fit, or fit to NN-distr specifically?
        """

        p_init, bounds = self.get_parameter_estimates()

        # print("Fitting model to data with initial parameters:", p_init)
        lambda_ = (
            300 / self.params["L"] ** 2
        )  # initial guess for neuron density - result should be kinda independent

        match_function = partial(
            match_model,
            lambda_=lambda_,
            R_cut=self.params["neighbor_distance"],
            L=self.params["L"],
            nbins=self.params["nbins"],
        )

        # probs_empirical = self.counts["cross"][..., 0] / self.counts["cross"][..., 0].sum()
        opts = dict(
            counts=self.counts["cross"][..., 0]
            / self.counts["cross"][..., 0].sum(),  # empirical counts
            theta0=list(p_init.values()),  # initial parameter guesses
            model_bin_probs=match_function,  # model function to compute probabilities
            bounds=list(bounds.values()),  # parameter bounds
            mask=self.counts["cross"][..., 0] > 0,  # mask for valid bins
        )

        try:
            #     res = fit_histogram_params(
            #         **opts,
            #         method="multinomial",
            #     )
            #     if not res.success:
            #         print(res)
            #         raise ValueError("Fitting matching model failed!")
            # except:
            res = fit_histogram_params(
                **opts,
                method="poisson",  #
            )
            if not res.success:
                # print(res)
                raise ValueError("Fitting matching model failed!")
        except Exception as e:
            print("Fitting matching model failed with error:", e)
            print("Using initial parameters as fallback.")
            res = type("Result", (object,), {"theta_hat": list(p_init.values())})()

        p_out = {}
        for i, (key, val) in enumerate(p_init.items()):
            p_out[key] = res.theta_hat[i]
            # print(f"Updated {key}: {val} -> {p_out[key]}")

        self.model = {}
        self.model["parameters"] = p_out

        self.model["pdf"] = match_function(p_in=list(p_out.values()), return_1D=True)

        # convert to cumulative for better numerical stability
        def get_cdf(pdf, reverse=False):
            if reverse:
                return np.nancumsum(pdf[::-1])[::-1] / np.nansum(pdf)
            else:
                return np.nancumsum(pdf) / np.nansum(pdf)

        self.model["cdf"] = {}
        for key in self.model["pdf"].keys():
            reverse = key in ["distance_same", "correlation_diff"]
            self.model["cdf"][key] = get_cdf(self.model["pdf"][key], reverse=reverse)

        key_model = "cdf" if use_cdf else "pdf"
        self.model["p_same"] = {}
        pdf_NN = self.model[key_model]["distance_same"] * p_out["p_same"]
        pdf_nNN = self.model[key_model]["distance_diff"] * (1 - p_out["p_same"])
        self.model["p_same"]["distance"] = nangauss_filter(
            pdf_NN / (pdf_NN + pdf_nNN), sigma=0.5
        )

        pdf_NN = self.model[key_model]["correlation_same"] * p_out["p_same"]
        pdf_nNN = self.model[key_model]["correlation_diff"] * (1 - p_out["p_same"])
        self.model["p_same"]["correlation"] = nangauss_filter(
            pdf_NN / (pdf_NN + pdf_nNN), sigma=0.5
        )

        # print("could be using cdfs here instead of pdfs for better performance")
        # pdf_NN = np.outer(f_c_same, f_r_same) * p_out["p_same"]
        pdf_NN = (
            self.model[key_model]["distance_same"][:, None]
            * self.model[key_model]["correlation_same"][None, :]
        ) * p_out["p_same"]
        pdf_nNN = (
            self.model[key_model]["distance_diff"][:, None]
            * self.model[key_model]["correlation_diff"][None, :]
        ) * (1 - p_out["p_same"])

        self.model["p_same"]["joint"] = nangauss_filter(
            pdf_NN / (pdf_NN + pdf_nNN), sigma=0.5
        )

        self.f_same = self.get_f_same("joint")

        p_same = self.f_same(self.params["arrays"]["distance_bounds"], 1.0)
        idx_cutoff = np.where(p_same < 0.05)[0][0]
        self.distance_cutoff = max(10, self.params["arrays"]["distance_bounds"][idx_cutoff] * 1.5)  ## make sure, also half-detected ones have a chance!

    def get_f_same(self, model="joint"):

        from scipy import interpolate

        if model == "joint":
            f_same = lambda distance, correlation: interpolate.interpn(
                (
                    self.params["arrays"]["distance"],
                    self.params["arrays"]["correlation"],
                ),
                self.model["p_same"]["joint"],
                (distance, correlation),
                bounds_error=False,
                fill_value=None,
            )
        else:

            f_same = interpolate.interp1d(
                self.params["arrays"][model],
                self.model["p_same"][model],
                # x,
                # bounds_error=False,
                fill_value="extrapolate",
            )
        return f_same

    def get_parameter_estimates(self):
        """
        Correlation values are obtained from according parts of the histogram
        Distance values are just hard-coded for now
        """
        H = self.counts["cross"][..., 0]
        p_init = {
            "p_same": 0.2,
            "h": 8.0,
            "sigma_eff": 1.0,
        }

        idx_min = np.argmin(gaussian_filter(H.sum(axis=1), sigma=2))
        # print("idx_min:", idx_min)
        p_init["h"] = self.params["arrays"]["distance_bounds"][idx_min] * 1.5
        # print(p_init["h"], "initial h estimate based on distance histogram")

        bounds = {
            "p_same": (1e-2, 0.5),
            "h": (4.0, 15.0),
            "sigma_eff": (1e-3, 5.0),
            "c_diff_mean": (0.0, 1.0),
            "c_diff_sd": (1e-3, 0.5),
            "c_same_mean": (0.0, 1.0),
            "c_same_sd": (1e-3, 0.5),
        }

        c_bounds = self.params["arrays"]["correlation_bounds"]
        c_centers = (c_bounds[:-1] + c_bounds[1:]) / 2

        # Get the midpoint row index (upper half of distance dimension)
        mid_row = H.shape[0] // 2
        # Sum counts across the upper half of H (lower distances)
        # upper_half_counts = H[mid_row:, :].sum(axis=0)

        # Calculate weighted mean and SD of correlation
        def weighted_stats(centers, counts):
            total_counts = counts.sum()
            if total_counts > 0:
                weighted_mean = np.sum(centers * counts) / total_counts
                weighted_variance = (
                    np.sum(counts * (centers - weighted_mean) ** 2) / total_counts
                )
                weighted_sd = np.sqrt(weighted_variance)
            else:
                weighted_mean = 0.0
                weighted_sd = 0.0
            return weighted_mean, weighted_sd

        p_init["c_diff_mean"], p_init["c_diff_sd"] = weighted_stats(
            c_centers, self.counts["cross"][mid_row:, :, 2]
        )

        low_dist_bin = np.where(self.params["arrays"]["distance_bounds"] > p_init["h"])[
            0
        ][0]
        p_init["c_same_mean"], p_init["c_same_sd"] = weighted_stats(
            c_centers, self.counts["cross"][:low_dist_bin, :, 1].sum(axis=0)
        )
        # print(f"Initial parameter estimates: {p_init}")

        return p_init, bounds

    def reset_registration(self):

        ## prepare and initialize assignment- and p_matched-arrays for storing results

        ## parameters needed for registration process
        # self.nS = 0
        # self.sessions = []
        self.union = SessionData(name="union", params=self.params)

        self.assignments = np.zeros((0, 0), int)  # nNeurons x nSessions

        self.tracking = {
            "p_matched": np.zeros((0, 0, 2), float),
            "shifts": np.zeros((0, 0, 2), float),  # nNeurons x nSessions x 2 (x,y)
        }

    ### ------------------------------------------------------------ ###
    ### --------------- Neuron registration function --------------- ###
    ### ------------------------------------------------------------ ###

    def register_neurons(
        self,
        from_file: Optional[str | Path] = None,
        from_data: Optional[SessionData] = None,
        from_session_id: Optional[int] = None,
        align_to_reference=True,
        clean_traces=True,
        p_thr=[0.5, 0.3],
    ):

        this_data = self.from_data(
            from_file=from_file,
            from_data=from_data,
            from_session_id=from_session_id,
            align_to_reference=align_to_reference,
        )

        if not this_data.aligned or this_data.A is None:

            print(f"Session {this_data.path} did not pass quality criteria, skipping.")
            self.assignments = pad_axis(self.assignments, (0, 1), -1)
            self.tracking["p_matched"] = pad_axis(
                self.tracking["p_matched"], (0, 1, 0), np.nan
            )
            self.tracking["shifts"] = pad_axis(
                self.tracking["shifts"], (0, 1, 0), np.nan
            )
            # self.handover_parameters(0, this_data)
            if clean_traces:
                this_data.clean_traces()
            return

        if not self.union.status["spatial_loaded"]:

            A = this_data.A[:, this_data.idx_eval]
            self.union.register_spatial(A=A, dims=this_data.dims)

            actually_good = np.where(this_data.idx_eval)[0]
            N_add = len(actually_good)
            # print("actually good:", actually_good)

            self.assignments = pad_axis(self.assignments, (N_add, 1), -1)
            self.assignments[:, this_data.id] = actually_good

            self.tracking["p_matched"] = pad_axis(
                self.tracking["p_matched"], (N_add, 1, 0), np.nan
            )
            # first occurence of neuron defined as p_match = 1
            self.tracking["p_matched"][:, this_data.id, 0] = 1.0

            self.tracking["shifts"] = pad_axis(
                self.tracking["shifts"], (N_add, 1, 0), np.nan
            )
            self.tracking["shifts"][:, this_data.id, :] = 0.0

            # self.handover_parameters(N_add, this_data)
            this_data.matched = True
            if clean_traces:
                this_data.clean_traces()
            # self.nS += 1
            return

        ### obtain matching probability from cross session statistics and model
        footprint_shifts, footprint_distances, footprint_correlations, _, _ = (
            calculate_statistics(
                this_data,
                self.union,
                distance_threshold=self.distance_cutoff,
                nP=12,
                # params=self.params,
            )
        )
        p_same = calculate_p(
            footprint_distances,
            footprint_correlations,
            self.f_same,
            self.params["neighbor_distance"],
        )

        ### ------------------------------------------------------------ ###
        ### -------------- Hungarian Algorithm (matching) -------------- ###
        ### ------------------------------------------------------------ ###
        ### run hungarian algorithm (HA) with (1-p_same) as score
        ### ------------------------------------------------------------ ###
        matches = linear_sum_assignment(1 - p_same.toarray())
        p_matched = p_same.toarray()[matches]
        # print("\n \t ## Matching results ##")

        ## thresholds for accepting matches and removing non-matches
        ## (HA matches all pairs, but we only want matches above p_thr)
        idx_TP = np.where(p_matched > p_thr[0])[0]
        # print(matches)
        if len(idx_TP) > 0:
            matched_ref = matches[0][idx_TP]  # matched neurons in s_ref
            matched = matches[1][idx_TP]  # matched neurons in s
        else:
            matched_ref = np.array([], "int")
            matched = np.array([], "int")

        ## find neurons which were not matched in current and reference session
        non_matched_ref = np.setdiff1d(
            list(range(self.union.n_neurons)), matched_ref
        )
        non_matched = np.setdiff1d(
            list(np.where(this_data.idx_eval)[0]), matches[1][idx_TP]
        )
        non_matched = non_matched[this_data.idx_eval[non_matched]]

        ## calculate number of matches found
        # TP = np.sum(p_matched > p_thr[0]).astype("float32")

        ## removing footprints from the data which were competing with another one
        ## to be matched and lost, but have significant probability to be the same
        ## this step ensures, that downstream session don't confuse this one and the
        ## 'winner', leading to arbitrary assignments between two clusters
        for nm in non_matched:
            p_all = p_same[:, nm].todense()
            if np.any(p_all > p_thr[1]):
                #    print(f'!! neuron {nm} is removed, as it is nonmatched and has high match probability:',p_all)[p_all>0])
                non_matched = non_matched[non_matched != nm]

        ### ------------------------------------------------------------ ###
        ### -------------- update reference data structure ------------- ###
        ### ------------------------------------------------------------ ###
        ### update footprint shapes of matched neurons with A_ref = (1-p/2)*A_ref + p/2*A
        ### to maintain part or all of original shape, depending on p_matched
        ### ------------------------------------------------------------ ###

        ## shift union footprints to "new" location of neuron to ensure proper union construction
        shifted = sparse.hstack(
            [
                (
                    _shift_sparse_bilinear(
                        self.union.A[:, m_ref],  # .reshape(512, 512),
                        self.union.dims,
                        -footprint_shifts[m_ref, m, 0],
                        -footprint_shifts[m_ref, m, 1],
                        order="C",
                        # output_format="csc",
                    )  # .reshape(-1, 1)
                    if footprint_distances[m_ref, m] > 0.5
                    else self.union.A[:, m_ref]
                ).multiply(1 - footprint_correlations[m_ref, m] / 2)
                + this_data.A[:, m].multiply(footprint_correlations[m_ref, m] / 2)
                for m_ref, m in zip(matched_ref, matched)
            ],
            format="csc",
        )
        # print("time taken (shift):", time.time() - t_start)

        # self.union.A[:, matched_ref] = self.union.A[:, matched_ref].multiply(
        #     1 - p_matched[idx_TP] / 2
        # ) + this_data.A[:, matched].multiply(p_matched[idx_TP] / 2)

        self.union.A.toarray()[:, matched_ref] = shifted.toarray()
        ## append new neuron footprints to union
        A_updated = sparse.hstack(
            [sparse.coo_matrix(self.union.A), this_data.A[:, non_matched]],
            format="csc",
        )
        # ## update union data
        self.union.register_spatial(A=A_updated)
        # print("union update (shift + merge + stack):", time.time() - t_start)

        # self.progress.set_description(
        #     f"Union now contains {self.union.n_neurons} neurons"
        # )

        ### ------------------------------------------------------------ ###
        ### ------------------ store matching results ------------------ ###
        ### ------------------------------------------------------------ ###

        # (self.assignments>=0).sum(axis=0)
        N_add = len(non_matched)  ## assuming there are never empty rows

        # print(f"Previous shape of assignments: {self.assignments.shape}")
        # print(
        #     f"Session {this_data.path} matched {len(matched)} neurons and added {N_add} new neurons to the union."
        # )

        ## prepare to hold new results by padding existing arrays
        if self.assignments.shape[1] <= this_data.id:
            ## either append to end
            self.assignments = pad_axis(self.assignments, (N_add, 1), -1)
            self.tracking["p_matched"] = pad_axis(
                self.tracking["p_matched"], (N_add, 1, 0), np.nan
            )
            self.tracking["shifts"] = pad_axis(
                self.tracking["shifts"], (N_add, 1, 0), np.nan
            )
        else:
            ## or just write into already existing rows, if possible
            assert np.all(
                self.assignments[:, this_data.id] == -1
            ), "Session already has assignments, cannot overwrite!"
            print(f"adding {N_add} new neurons to union for session {this_data.id}")
            self.assignments = pad_axis(self.assignments, (N_add, 0), -1)
            self.tracking["p_matched"] = pad_axis(
                self.tracking["p_matched"], (N_add, 0, 0), np.nan
            )
            self.tracking["shifts"] = pad_axis(
                self.tracking["shifts"], (N_add, 0, 0), np.nan
            )

        # ... matched neurons are added
        self.assignments[matched_ref, this_data.id] = matched

        self.tracking["p_matched"][matched_ref, this_data.id, 0] = p_matched[idx_TP]
        self.tracking["shifts"][matched_ref, this_data.id, :] = footprint_shifts[
            matched_ref, matched
        ]

        ## ... and non-matched (new) neurons are appended
        self.assignments[-N_add:, this_data.id] = non_matched
        self.tracking["p_matched"][-N_add:, this_data.id, 0] = 1.0

        ## write best non-matching probability
        p_all = p_same.toarray()
        self.tracking["p_matched"][matched_ref, this_data.id, 1] = [
            max(
                p_all[
                    c,
                    np.where(
                        p_all[c, :] != self.tracking["p_matched"][c, this_data.id, 0]
                    )[0],
                ]
            )
            for c in matched_ref
        ]

        ### --------------------------------------------------------------------------- ###
        ### ------------------------- hand over parameters ---------------------------- ###
        ### --------------------------------------------------------------------------- ###

        # self.handover_parameters(N_add, this_data)
        this_data.matched = True
        if clean_traces:
            this_data.clean_traces()

        # if np.any(np.all(self.tracking["p_matched"] > 0.9, axis=2)):
        #     print("double match!")
        #     return

    def unregister_neurons(self, session_id: int):
        """
        Removes a session's neurons from the union data and updates assignments and tracking accordingly.
        """
        if session_id < 0 or session_id >= self.assignments.shape[1]:
            raise ValueError(
                "Invalid session_id. It must be within the range of existing sessions."
            )
        # print(f"Session {session_id} has been unregistered. Updating union data...")

        # Mark assignments and tracking stats in this session as unassigned
        self.assignments[:, session_id] = -1
        self.tracking["p_matched"][:, session_id, :] = np.nan
        self.tracking["shifts"][:, session_id, :] = np.nan

        self.updating_neuron_presence()  # Update neuron presence and clean union data
        self.sessions[session_id].matched = False  # Mark the session as unmatched

    def remove_session(self, session_id: int):
        """
        Removes a session from the union data and updates assignments and tracking accordingly.
        """
        if session_id < 0 or session_id >= self.assignments.shape[1]:
            raise ValueError(
                "Invalid session_id. It must be within the range of existing sessions."
            )
        # print(f"Session {session_id} has been unregistered. Updating union data...")

        # Mark assignments and tracking stats in this session as unassigned
        self.assignments = np.delete(self.assignments, session_id, axis=1)
        self.tracking["p_matched"] = np.delete(
            self.tracking["p_matched"], session_id, axis=1
        )
        self.tracking["shifts"] = np.delete(self.tracking["shifts"], session_id, axis=1)

        self.updating_neuron_presence()  # Update neuron presence and clean union data
        self.sessions.pop(session_id)

    def updating_neuron_presence(self):

        ## remove neurons that are no longer present in any session
        neuron_presence = (self.assignments >= 0).sum(axis=1) > 0
        self.assignments = self.assignments[neuron_presence, :]
        self.tracking["p_matched"] = self.tracking["p_matched"][neuron_presence, :, :]
        self.tracking["shifts"] = self.tracking["shifts"][neuron_presence, :, :]

        print(
            f"shape union data before cleaning: {self.union.A.shape}, vs {self.union.n_neurons}"
        )
        ## could just rebuild it entirely from the remaining sessions, but for now just remove the columns of empty neurons
        A_cleaned = sparse.hstack(
            [
                self.union.A[:, i]
                for i in range(self.union.n_neurons)
                if neuron_presence[i]
            ],
            format="csc",
        )
        self.union.register_spatial(A=A_cleaned)

        print(f"Updated union data now contains {self.union.n_neurons} neurons.")

    # def get_variable_by_cluster(self, var, c=None):
    #     """
    #     builds cluster-wise arrays from data stored by sessions, using assignment

    #     should be possible for:
    #         * A
    #         * centroids
    #         * quality metrics (SNR, r-value, CNN-prediction)
    #         * (traces)
    #     """

    #     clusters = getattr(self, self.cluster_field)
    #     if c is None:
    #         c = np.arange(self.assignments.shape[0])
    #     if not isinstance(c, (list, np.ndarray)):
    #         c = [c]

    #     neuron_idx = 1 if var in ["A"] else 0
    #     is_sparse = var in ["A"]

    #     if var in ["A", "centroids"]:
    #         dummy_var = getattr(
    #             self.sessions[0], var
    #         )  # get variable from first session to obtain dims
    #     elif var in ["SNR_comp", "r_values", "cnn_preds"]:
    #         dummy_var = self.sessions[0].quality[
    #             var
    #         ]  # get variable from first session to obtain dims
    #     elif var in ["C", "F_dff", "F_dff_dec", "S", "S_dff"]:
    #         dummy_var = self.sessions[0].traces[
    #             var
    #         ]  # get variable from first session to obtain dims

    #     dummy_shape = list(dummy_var.shape)
    #     dummy_shape.pop(neuron_idx)

    #     dummy_type = dummy_var.dtype
    #     # print("dummy shape:", dummy_shape)
    #     # print("dummy type:", dummy_type)

    #     # dims = self.assignments.shape + tuple(dummy_shape)
    #     dims = (len(c), self.assignments.shape[1]) + tuple(dummy_shape)

    #     if is_sparse:
    #         # cluster_var = sparse.csc_matrix(dims, dtype=dummy_type)
    #         cluster_var = sparse.hstack(
    #             [
    #                 (
    #                     self.sessions[s].A[:, n]
    #                     if n >= 0
    #                     else sparse.csc_matrix((np.prod(dims), 1))
    #                 )
    #                 for s, n in enumerate(self.assignments[c, :])
    #             ],
    #             format="csc",
    #         )
    #     else:
    #         cluster_var = np.full(dims, np.nan)
    #         for idx, c_ in enumerate(c):
    #             cluster_var[idx, ...] = np.array(
    #                 [
    #                     (
    #                         np.take(getattr(self.sessions[s], var), n, axis=neuron_idx)
    #                         if n >= 0
    #                         else np.full(dummy_shape, np.nan)
    #                     )
    #                     for s, n in enumerate(self.assignments[c_, :])
    #                 ]
    #             )
    #     # print("dims:", dims)

    #     return cluster_var

    def classify_sessions(self, interval=None, **kwargs):
        # max_shift=50.0, min_zscore=4.0):
        """
        checks all sessions to pass certain criteria to
        be included in the further analysis
        """
        # max_shift = kwargs.get("max_shift", self.params["max_session_shift"])
        # min_zscore = kwargs.get(
        #     "min_zscore", self.params["min_session_correlation_zscore"]
        # )

        n_session = len(self.sessions)  # alignment["shift"].shape[0]
        status = np.zeros(n_session, bool)

        ## if 'sessions' is provided (tuple), it specifies range
        ## of sessions to be included
        if interval is None:
            sStart = 0  # np.where(~np.all(np.isnan(alignment["c_max"]), axis=1))[0][0]
            sEnd = n_session
        else:
            sStart = max(0, interval[0] - 1)
            sEnd = interval[-1]

        status[sStart:sEnd] = True
        for session in self.sessions:

            session.evaluate_alignment_status()
            status[session.id] = session.aligned
        ## check for coherence with other sessions (low shift, high correlation)
        # abs_shift = np.array([np.sqrt(x**2 + y**2) for (x, y) in alignment["shift"]])
        # status[abs_shift > max_shift] = False  ## huge shift
        # status[np.nanmedian(alignment["c_zscored"], axis=1) < min_zscore] = False
        # status[np.all(np.isnan(alignment["c_zscored"]), axis=1)] = False

        ## finally, check if data can be loaded properly
        # for s in np.where(status)[0]:

        #     if not alignment["file_paths"][s].exists():
        #         status[s] = False
        # alignment["alignment_status"] = status
        self.alignment_status = status
        return status

    def build_borders(self, margin=0.0):

        dims = self.sessions[0].dims  # dims of first should be the same in all sessions
        shifts = np.array(
            [
                this_data.remap.shift if this_data.remap else (0, 0)
                for this_data in self.sessions
            ]
        )
        thr_high = np.nanmin(dims + shifts, axis=0)
        thr_low = np.nanmax(shifts, axis=0)

        borders = np.vstack([thr_low + margin, thr_high - margin])
        return borders

    def classify_components(self, **kwargs):
        """
        checks all clusters to pass certain criteria to be considered in the analysis

        Each cluster is required to:
            * pass all "lowest" thresholds (SNR, r-value, CNN-prediction)
            * at least one of the "min" thresholds (SNR, r-value, CNN-prediction)
            * be present in at least 'min_cluster_count' sessions
            * have a center of mass within the borders of the imaging window, leaving some margin of 'border_margin'
        """
        # clusters = getattr(self, self.cluster_field)

        n_cluster, n_session = self.assignments.shape
        status = np.ones(n_cluster, bool)

        # status_session = alignment.get("alignment_status", None)
        if self.alignment_status is None:
            self.classify_sessions()
            # status_session = np.ones(n_session, bool)
        # else:
        # status_session = self.alignment_status

        ## check for neuron detection thresholds
        # if not hasattr(self, "thr"):
        # set_thresholds()
        thr = {
            # component quality
            "SNR_lowest": 1.0,
            "SNR_min": 2.5,
            "rval_lowest": -1.0,
            "rval_min": 0.6,
            "cnn_lowest": 0.1,
            "cnn_min": 0.9,
            # tracking quality
            "p_matched": 0.3,
            "min_cluster_count": 2.0,
        }

        status_detected = (
            # (
            #     ## minimum requirements for each neuron
            #     (clusters["SNR_comp"] > thr["SNR_lowest"])
            #     & (clusters["r_values"] > thr["rval_lowest"])
            #     & (clusters["cnn_preds"] > thr["cnn_lowest"])
            # )
            # & (
            #     ## each neuron needs to exceed at least one of the following thresholds
            #     (clusters["SNR_comp"] > thr["SNR_min"])
            #     | (clusters["r_values"] > thr["rval_min"])
            #     | (clusters["cnn_preds"] > thr["cnn_min"])
            # )
            # &
            self.tracking["p_matched"][..., 0]
            > thr["p_matched"]
        )

        ## remove components from sessions that are not included in the data
        status_detected[:, ~self.alignment_status] = False

        ## check for presence in at least 'min_cluster_count' sessions
        status[
            status_detected[:, self.alignment_status].sum(1) < thr["min_cluster_count"]
        ] = False

        # if borders is None:
        borders = self.build_borders(kwargs.get("border_margin", 2.0))

        ## check for distance from imaging window borders
        for i in range(2):
            idx_remove_low = self.union.centroids[:, i] < (borders[0, i])
            # status[np.any(idx_remove_low, 1)] = False
            status[idx_remove_low] = False

            idx_remove_high = self.union.centroids[:, i] > (borders[1, i])
            # status[np.any(idx_remove_high, 1)] = False
            status[idx_remove_high] = False
        # clusters["status"] = status
        return status

    ### ---- Model manipulation functions ---- ###

    def _update_bins(self, nbins):

        self.params["nbins"] = nbins
        # self.params["arrays"] = self.build_arrays(nbins)

        ## create value arrays for distance and footprint correlation
        arrays = {}
        arrays["distance_bounds"] = np.linspace(
            0, self.params["neighbor_distance"], nbins + 1
        )
        arrays["correlation_bounds"] = np.linspace(0, 1, nbins + 1)

        distance_step = self.params["neighbor_distance"] / nbins
        correlation_step = 1.0 / nbins

        arrays["distance"] = arrays["distance_bounds"][:-1] + distance_step / 2
        arrays["correlation"] = arrays["correlation_bounds"][:-1] + correlation_step / 2

        self.params["arrays"] = arrays

    def scale_counts(self, times=0, key="cross"):

        counts = scale_down_counts(self.counts[key], times)
        bins = counts.shape[0]

        self._update_bins(bins)
        return counts

    ### ------------------------------------------------------------ ###
    """
        ------------------------------------------------------------
        ----------------- refitting from tracking ------------------
        ------------------------------------------------------------
    """
    ### ------------------------------------------------------------ ###

    def get_footprints_at_session(
        self, s: int, ds=np.inf, s_cuts: list[int] = [], complete_new: bool = False
    ):

        s -= 1  # adjust for 0-indexing

        ## load data from session s to store already detected ones
        session_path = self.sessions[s].path

        print(f"loading data from session {s+1}: {session_path}")
        ld = load_data(session_path, subpath="/estimates")
        dims = ld["Cn"].shape
        print("...done")

        ## classify sessions and components from tracking, to use only good ones
        self.classify_sessions()
        borders = self.build_borders(margin=2.0)
        self.classify_components(borders=borders)

        clusters = getattr(self, self.cluster_field)
        n_cluster = clusters["status"].sum()

        ## prepare some dictionaries for storing in- and output data
        dataIn = {}
        dataOut = {}
        idxes = {
            "in": {
                "active": np.zeros(n_cluster, "bool"),
                "silent": np.zeros(n_cluster, "bool"),
                "match_to_c": None,  # indexing of matching cluster number
                "match_to_n": None,  # indexing to session neuron number
            },
            "out": {
                "active": None,
                "silent": None,
            },
        }

        ## initialize input data with random values
        T = ld["C"].shape[1]
        dataIn["Cn"] = ld["Cn"]
        dataIn["C"] = np.random.rand(n_cluster, T)

        if complete_new:

            ## if (for whatever reason) you just want to throw in n_cluster random footprints
            ## (unsure if this even works without specifying 'A')
            idxes["in"]["silent"] = np.ones(n_cluster, "bool")
            dataIn["b"] = np.random.rand(int(np.prod(dims)), 1)
            dataIn["f"] = np.random.rand(1, T)
        else:
            ## hand over data from session s
            ## find active and silent neurons in session s
            detected = self.assignments[:, s] >= 0
            isSilent = clusters["status"] & ~detected
            idxes["in"]["nSilent"] = isSilent.sum()
            isActive = clusters["status"] & detected
            idxes["in"]["nActive"] = isActive.sum()

            idxes["in"]["active"][: idxes["in"]["nActive"]] = True
            idxes["in"]["silent"][idxes["in"]["nActive"] :] = True

            c_idx = np.concatenate([np.where(isActive)[0], np.where(isSilent)[0]])
            n_idx = self.assignments[isActive, s]

            idxes["in"]["match_to_c"] = c_idx
            idxes["in"]["match_to_n"] = n_idx

            # dataIn["A"][:, : idxes["in"]["nActive"]]
            dataIn["A"] = ld["A"][:, n_idx]
            # dataIn["A"] = alignment["A"][str(s)][:, n_idx]

            ## load temporal components of active cells from session s
            T1 = ld["C"].shape[1]  # adjusted for a session, where T != T1
            dataIn["C"][: idxes["in"]["nActive"], :T1] = ld["C"][n_idx, :]

            ## load background components from session s
            if not (ld["b"].shape[0] == dataIn["A"].shape[0]):
                ld["b"] = ld["b"].transpose()
            dataIn["b"] = ld["b"]
            if not (ld["f"].shape[1] == dataIn["C"].shape[1]):
                ld["f"] = ld["f"].transpose()
            dataIn["f"] = ld["f"]

        ## if given, find closest measurement cuts (session batches)
        s_cuts = np.array(s_cuts)
        s_min = s_cuts[s_cuts < s].max() if np.any(s_cuts < s) else 0
        s_max = s_cuts[s_cuts > s].min() if np.any(s_cuts > s) else np.inf

        ## find footprint data from adjacent sessions for neurons which were not detected
        print(
            f"There are {idxes['in']['nActive']}/{n_cluster} active neurons in session {s+1} and {idxes['in']['nSilent']} silent neuron footprints are attempted to be reconstructed from adjacent sessions."
        )
        print(
            f"Building silent footprints for session {s+1} from matched footprints from sessions between {s_min} and {s_max}"
        )
        dataIn_silent = np.empty((np.prod(dims), idxes["in"]["nSilent"]))
        for i, c in enumerate(idxes["in"]["match_to_c"][idxes["in"]["silent"]]):

            A_tmp = np.zeros((np.prod(dims), 1))

            ## search closest previous session with valid footprint
            s_pre = np.where(self.assignments[c, :s] >= 0)[0]
            if len(s_pre) > 0 and (s - s_pre[-1]) <= ds and s_pre[-1] >= s_min:
                s_ref = s_pre[-1]
                # print(
                #     f"Found previous session {s_ref+1} for cluster {c} (silent in session {s+1})"
                # )
                n_ref = self.assignments[c, s_ref]
                A_tmp += (
                    1.0 / abs(s_ref - s) * alignment["A"][str(s_ref)][:, n_ref]
                ).toarray()

            ## search closest following session with valid footprint
            s_post = s + 1 + np.where(self.assignments[c, s + 1 :] >= 0)[0]
            if len(s_post) > 0 and (s_post[0] - s) <= ds and s_post[0] < s_max:
                s_ref = s_post[0]
                # print(
                #     f"Found following session {s_ref+1} for cluster {c} (silent in session {s+1})"
                # )
                n_ref = self.assignments[c, s_ref]
                A_tmp += (
                    1.0 / abs(s_ref - s) * alignment["A"][str(s_ref)][:, n_ref]
                ).toarray()
            # print(A_tmp.max())
            dataIn_silent[:, i] = A_tmp.ravel()

        ## remap silent neuron footprints to current session
        dataIn_silent = sparse.csc_matrix(dataIn_silent)
        if np.all(np.isfinite(alignment["shift"][s, :].max())):
            dataIn_silent = apply_remap(
                dataIn_silent,
                dims=(512, 512),
                shift=-alignment["shift"][s, :],
            )

        ## and store in data along with detected neurons
        dataIn["A"] = normalize_sparse_array(
            sparse.hstack([dataIn["A"], dataIn_silent], format="csc")
        )

        return dataIn, dataOut, idxes

    def refit_footprints_from_tracking(self, s: int, **kwargs):
        """
        refits the model using the footprints of neurons detected in session s as "same" distribution
        and all other pairs of neurons as "different" distribution

        kwargs:
            ds: maximum session distance to consider for finding matching footprints for silent neurons (default: np.inf)
            s_cuts: list of session numbers where there are cuts in the data, to limit search for matching footprints (default: [])
        """

        dataIn, dataOut, idxes = self.get_footprints_at_session(s, **kwargs)

        # print("dataIn keys:", dataIn.keys())
        # print("dataOut keys:", dataOut.keys())
        # print("idxes:", idxes)

        p_init, bounds = self.estimate_initial_parameters(dataIn, idxes)

        self.model = fit_model(
            dataIn,
            idxes,
            p_init,
            bounds,
            params=self.params,
        )

    ### ------------------------------------------------------------ ###
    """
        ------------------------------------------------------------
        ---------------- saving and loading methods ----------------
        ------------------------------------------------------------
    """

    ### ------------------------------------------------------------ ###
    def save_model(
        self,
        output_directory: Optional[str | Path] = None,
        suffix: str = "",
        ext: str = ".hdf5",
    ):

        output_directory = self.get_result_directory(output_directory)

        fix_suffix(suffix)

        data = {"counts": self.counts, "model": self.model}
        save_data(data, str(output_directory / f"match_model{suffix}{ext}"))

    def load_model(self, path_model: str):

        ld = load_data(path_model, subpath="/")
        assert (
            "model" in ld and "counts" in ld
        ), "Data does not contain necessary fields 'model' and 'counts'"

        self.model = ld["model"]
        self.counts = ld["counts"]
        self._update_bins(self.counts["same"].shape[0])

        self.f_same = self.get_f_same("joint")

    def save_registration(
        self,
        output_directory: Optional[str | Path] = None,
        suffix: str = "",
        ext: str = ".hdf5",
    ):
        output_directory = self.get_result_directory(output_directory)

        suffix = fix_suffix(suffix)

        with h5py.File(output_directory / f"neuron_registration{suffix}{ext}", "w") as f:
            self.to_hdf5(f)

        print(f"Saved neuron registration to {output_directory / f'neuron_registration{suffix}{ext}'}")


        # data = {
        #     "sessions": self.sessions,
        #     "assignments": self.assignments,
        #     "tracking": self.tracking,
        #     # "clusters": getattr(self, self.cluster_field),
        # }
        # save_data(data, str(output_directory / f"neuron_registration{suffix}{ext}"))

    def to_hdf5(self, group: h5py.Group | h5py.File) -> None:
        group.attrs["object_type"] = "TrackingResult"
        group.attrs["schema_version"] = self.HDF5_VERSION

        group.create_dataset(
            "assignments",
            data=self.assignments,
            compression="gzip",
        )

        tracking_group = group.create_group("tracking")
        tracking_group.create_dataset(
            "p_matched",
            data=self.tracking["p_matched"],
            compression="gzip",
        )
        tracking_group.create_dataset(
            "shifts",
            data=self.tracking["shifts"],
            compression="gzip",
        )

        sessions_group = group.create_group("sessions")
        sessions_group.attrs["n_sessions"] = len(self.sessions)

        for session in self.sessions:
            session_group = sessions_group.create_group(
                f"session_{session.id:03d}"
            )
            session.to_hdf5(session_group)

        union_group = sessions_group.create_group(
            f"union"
        )
        self.union.to_hdf5(union_group)
            
    def load_registration(self, path_registration: str):

        with h5py.File(path_registration, "r") as f:
            self.assignments, self.tracking, self.sessions, self.union = self.from_hdf5(f)
        
        # ld = load_data(path_registration, subpath="/")
        # print("path:",path_registration)
        # print(ld.keys())
        # assert (
        #     "sessions" in ld and "assignments" in ld and "tracking" in ld
        # ), "Data does not contain necessary fields 'sessions', 'assignments', and 'tracking'"

        # self.sessions = ld["sessions"]
        # print(self.sessions)

        # self.assignments = ld["assignments"]
        # self.tracking = ld["tracking"]

        ## ensure session keys are integers (as saving/loading casts them to strings...)
        # self.sessions = []
        # for session in sessions:
            # print("remap", this_data["remap"])
            # print("Session key:", s)
            # self.sessions.append(SessionData(**this_data))

        # sessions["file_paths"] = [
        #     str(fp.decode("utf-8")) for fp in sessions["file_paths"]
        # ]
        # if suffix is None:
        #     suffix = self.paths["suffix"]
        # suffix = fix_suffix(suffix)

        # ext = "mat" if (matlab is None and self.matlab) or matlab else "pkl"
        # pathLd = (
        #     Path(self.paths["data"]) / f"matching/neuron_registration{suffix}.{ext}"
        # )


    def from_hdf5(cls, group: h5py.Group | h5py.File):
        if group.attrs.get("object_type") != "TrackingResult":
            raise ValueError(
                "The provided HDF5 group does not contain a TrackingResult object."
            )

        if group.attrs.get("schema_version") != cls.HDF5_VERSION:
            raise ValueError(
                f"Schema version mismatch: expected {cls.HDF5_VERSION}, found {group.attrs.get('schema_version')}"
            )

        assignments = group["assignments"][()]
        tracking = {
            "p_matched": group["tracking"]["p_matched"][()],
            "shifts": group["tracking"]["shifts"][()],
        }

        sessions_group = group["sessions"]
        n_sessions = sessions_group.attrs["n_sessions"]
        sessions = []
        for i in range(n_sessions):
            session_group = sessions_group[f"session_{i:03d}"]
            session = SessionData.from_hdf5(session_group)
            sessions.append(session)

        union_group = sessions_group["union"]
        union = SessionData.from_hdf5(union_group)

        return assignments, tracking, sessions, union

    def get_result_directory(
        self,
        output_directory: Path | str | None = None,
    ) -> Path:
        if output_directory is None:
            output_directory = self._default_matching_directory()

        output_directory = Path(output_directory).expanduser().resolve()
        assert isinstance(output_directory, Path), f"output_directory {output_directory} should be a Path object"
        
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return output_directory

    def _default_matching_directory(
        self,
    ) -> Path:

        candidate_paths = [Path(session.path).parent for session in self.sessions if session.path is not None]
        common_path = os.path.commonpath([str(path) for path in candidate_paths])
        return Path(common_path) / "matching"

def mean_of_trunc_lognorm(mu, sigma, trunc_loc):

    alpha = (trunc_loc[0] - mu) / sigma
    beta = (trunc_loc[1] - mu) / sigma

    phi = lambda x: 1 / np.sqrt(2 * np.pi) * np.exp(-1 / 2 * x**2)
    psi = lambda x: 1 / 2 * (1 + special.erf(x / np.sqrt(2)))

    trunc_mean = mu + sigma * (phi(alpha) - phi(beta)) / (psi(beta) - psi(alpha))
    trunc_var = np.sqrt(
        sigma**2
        * (
            1
            + (alpha * phi(alpha) - beta * phi(beta)) / (psi(beta) - psi(alpha))
            - ((phi(alpha) - phi(beta)) / (psi(beta) - psi(alpha))) ** 2
        )
    )

    return trunc_mean, trunc_var


def norm_nrg(a_):

    a = a_.copy()
    dims = a.shape
    a = a.reshape(-1, order="F")
    indx = np.argsort(a, axis=None)[::-1]
    cumEn = np.cumsum(a.flatten()[indx] ** 2)
    cumEn /= cumEn[-1]
    a = np.zeros(np.prod(dims))
    a[indx] = cumEn
    return a.reshape(dims, order="F")


def scale_down_counts(counts, times=1):
    """
    scales down the whole matrix "counts" by a factor of 2^times
    """

    if times == 0:
        return counts

    assert counts.shape[0] > 8, "No further scaling down allowed"

    if len(counts.shape) > 2:
        cts = np.zeros(tuple((d // 2 for d in counts.shape[:2])) + (counts.shape[2],))
        for d in range(counts.shape[2]):
            for i in range(2):
                for j in range(2):
                    cts[..., d] += counts[i::2, j::2, d]
    else:
        cts = np.zeros(tuple((d // 2 for d in counts.shape[:2])))  # + (3,))
        for i in range(2):
            for j in range(2):
                cts += counts[i::2, j::2]

    return scale_down_counts(cts, times - 1)


def fix_suffix(suffix):
    if suffix:
        if not suffix.startswith("_"):
            suffix = "_" + suffix
    return suffix
