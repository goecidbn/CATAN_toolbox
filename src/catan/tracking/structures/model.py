from functools import partial

from typing import Any, Literal
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy import interpolate

from catan.core.io import load_file, save_file
from catan.core.structures.load_config import LoadConfig, FieldSpec
from catan.core.utils import nangauss_filter

from catan.tracking.analytics.fit_model_theoretical import (
    match_model,
    fit_histogram_params,
)
from catan.tracking.utils_new.counts import scale_down_counts

NATIVE_MODEL_CONFIG = "catan_model.json"


class Model:

    loaded = False  # tag, whether model was loaded from file
    HDF5_VERSION = "1.0"

    source_type: str = "model"
    source_config: LoadConfig | None = None

    def __init__(self, params=None):

        ## clean this up!
        self.params = (
            params
            if params
            else {
                "neighbor_distance": 25.0,
                "bins": 64,
                "n_threads": 1,
                "use_kde": False,
                "pxtomu": 1.0,
                "L": 512,
            }
        )
        self.counts = {
            "same": {},
            "cross": {},
        }

        self.count_revisions = {
            "same": {},
            "cross": {},
        }

        self.stale_counts = {
            "same": set(),
            "cross": set(),
        }

        self.fit_stale = False

        self.reset()

    def reset(self):

        self.parameters = {}
        self.distributions = {"pdf": {}, "cdf": {}}
        self.p_same = {}
        self.f_same = None
        self.distance_cutoff = 0.0

        self.fitted: bool = False
        self.fit_used_fallback = False
        self.build_arrays()

    def clear_counts(self):

        self.counts["same"].clear()
        self.counts["cross"].clear()

        self.count_revisions["same"].clear()
        self.count_revisions["cross"].clear()

        self.stale_counts["same"].clear()
        self.stale_counts["cross"].clear()

        self.fitted = False
        self.fit_stale = False

    def set_same_counts(
        self,
        session_path: str | Path,
        counts: np.ndarray,
        *,
        source_revision: int | None = None,
    ):

        path = str(session_path)

        counts = np.asarray(counts, dtype=int)

        expected_shape = (self.params["bins"], self.params["bins"])

        if counts.shape != expected_shape:
            raise ValueError(
                f"Same-session counts have shape {counts.shape}, expected {expected_shape}."
            )

        self.counts["same"][path] = counts.copy()
        self.count_revisions["same"][path] = source_revision

        self.stale_counts["same"].discard(path)

    def set_cross_counts(
        self,
        reference_path: str | Path,
        session_path: str | Path,
        counts: np.ndarray,
        *,
        source_revisions: tuple[int | None, int | None] | None = None,
    ):

        key = (str(reference_path), str(session_path))

        counts = np.asarray(counts, dtype=int)

        expected_shape = (self.params["bins"], self.params["bins"], 3)

        if counts.shape != expected_shape:
            raise ValueError(
                f"Cross-session counts have shape {counts.shape}, expected {expected_shape}."
            )

        self.counts["cross"][key] = counts.copy()
        self.count_revisions["cross"][key] = source_revisions
        self.stale_counts["cross"].discard(key)

        # Conservatively assume an existing fitted
        # model may have depended on this evidence.
        if self.fitted:
            self.fit_stale = True

    def invalidate_counts_for_paths(self, paths):

        paths = {str(path) for path in paths}

        stale_same = {path for path in self.counts["same"] if path in paths}

        stale_cross = {
            key for key in self.counts["cross"] if (key[0] in paths or key[1] in paths)
        }

        self.stale_counts["same"].update(stale_same)
        self.stale_counts["cross"].update(stale_cross)

        if self.fitted and stale_cross:
            self.fit_stale = True

        return {"same": stale_same, "cross": stale_cross}

    def counts_stale_for_path(self, path: str | Path) -> bool:

        path = str(path)

        if path in self.stale_counts["same"]:
            return True

        return any(
            (reference_path == path or session_path == path)
            for (reference_path, session_path) in self.stale_counts["cross"]
        )

    @property
    def has_stale_counts(self) -> bool:

        return bool(self.stale_counts["same"] or self.stale_counts["cross"])

    def aggregate_counts(
        self,
        *,
        session_order=None,
        session_paths=None,
        session_distances=None,
        include_stale=False,
    ):

        nbins = self.params["bins"]

        same = np.zeros((nbins, nbins), dtype=int)
        cross = np.zeros((nbins, nbins, 3), dtype=int)

        # ============================================
        # Explicit subset
        # ============================================
        allowed_paths = (
            None if session_paths is None else {str(path) for path in session_paths}
        )

        # ============================================
        # Current session order
        # ============================================
        order_index = None

        if session_order is not None:

            ordered_paths = [str(path) for path in session_order]

            if len(set(ordered_paths)) != len(ordered_paths):
                raise ValueError("session_order contains " "duplicate paths.")

            order_index = {path: index for index, path in enumerate(ordered_paths)}

        # ============================================
        # Allowed session distances
        # ============================================
        allowed_distances = None

        if session_distances is not None:

            if order_index is None:
                raise ValueError(
                    "session_order is required when session_distances is specified."
                )

            allowed_distances = {int(distance) for distance in session_distances}

            if any(distance <= 0 for distance in allowed_distances):
                raise ValueError("session_distances must contain positive integers.")

        # ============================================
        # Same-session counts
        # ============================================
        for path, counts in self.counts["same"].items():

            if not include_stale and path in self.stale_counts["same"]:
                continue

            if allowed_paths is not None and path not in allowed_paths:
                continue

            if order_index is not None and path not in order_index:
                continue

            same += counts

        # ============================================
        # Cross-session counts
        # ============================================
        for (reference_path, session_path), counts in self.counts["cross"].items():

            if (
                not include_stale
                and (reference_path, session_path) in self.stale_counts["cross"]
            ):
                continue

            if allowed_paths is not None:

                if (
                    reference_path not in allowed_paths
                    or session_path not in allowed_paths
                ):
                    continue

            if order_index is not None:

                if reference_path not in order_index or session_path not in order_index:
                    continue

                distance = order_index[session_path] - order_index[reference_path]

                # Important:
                # the stored count calculation is directional.
                # If the pair has changed order, do not silently
                # reinterpret the old result.
                if distance <= 0:
                    continue

                if allowed_distances is not None and distance not in allowed_distances:
                    continue

            cross += counts

        return {"same": same, "cross": cross}

    def build_arrays(self):

        nbins = self.params["bins"]
        self.arrays = {}
        self.arrays["distance_bounds"] = np.linspace(
            0, self.params["neighbor_distance"], nbins + 1
        )
        self.arrays["correlation_bounds"] = np.linspace(0, 1, nbins + 1)

        distance_step = self.params["neighbor_distance"] / nbins
        correlation_step = 1.0 / nbins

        self.arrays["distance"] = (
            self.arrays["distance_bounds"][:-1] + distance_step / 2
        )
        self.arrays["correlation"] = (
            self.arrays["correlation_bounds"][:-1] + correlation_step / 2
        )

    def scale_counts(self, counts, times=0):

        counts = scale_down_counts(counts, times)
        bins = counts.shape[0]

        # self._update_bins(bins)
        return counts

    def fit_model_to_counts(self, counts, use_cdf=True, min_counts=20):
        """
        Currently takes over h almost as provided - add weights to  improve fit, or fit to NN-distr specifically?
        """
        if self.loaded:
            print("Model was loaded from file - fitting to counts not allowed.")
            return
        bin_counts = counts[..., 0].sum()
        if bin_counts < min_counts:
            raise Exception(
                f"Not enough data to fit model - at least {min_counts} counts in cross histogram required (currently: {bin_counts})."
            )

        self.reset()
        p_init, bounds = self.get_parameter_estimates(counts)
        # print("Fitting model to data with initial parameters:", p_init)

        lambda_ = (
            300 / self.params["L"] ** 2
        )  # initial guess for neuron density - result should be kinda independent

        match_function = partial(
            match_model,
            lambda_=lambda_,
            R_cut=self.params["neighbor_distance"],
            nbins=self.params["bins"],
            L=self.params["L"],
        )

        opts = dict(
            counts=counts[..., 0] / counts[..., 0].sum(),  # empirical counts
            theta0=list(p_init.values()),  # initial parameter guesses
            model_bin_probs=match_function,  # model function to compute probabilities
            bounds=list(bounds.values()),  # parameter bounds
            mask=counts[..., 0] > 0,  # mask for valid bins
        )

        self.fit_used_fallback = False

        try:
            res = fit_histogram_params(
                **opts,
                method="poisson",
            )

            if (
                not res.success
                or not np.isfinite(res.nll_hat)
                or res.nll_hat >= 1e29
                or not np.all(np.isfinite(res.theta_hat))
            ):
                raise ValueError("Optimizer did not produce a valid matching model.")

            self.parameters = dict(zip(p_init, res.theta_hat))
            self.build_from_parameters(use_cdf=use_cdf)

        except Exception as error:
            print("Fitting matching model failed:", error)
            print("Using feasible initial parameters as fallback.")

            self.parameters = dict(p_init)
            self.fit_used_fallback = True
            self.build_from_parameters(use_cdf=use_cdf)

    def get_parameter_estimates(self, counts):
        """
        Correlation values are obtained from according parts of the histogram
        Distance values are just hard-coded for now
        """
        H = counts[..., 0]
        p_init = {
            "p_same": 0.2,
            "h": 8.0,
            "sigma_eff": 1.0,
        }

        idx_min = np.argmin(gaussian_filter(H.sum(axis=1), sigma=2))
        # print("idx_min:", idx_min)
        p_init["h"] = self.arrays["distance_bounds"][idx_min] * 1.5
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

        lambda_ = 300 / self.params["L"] ** 2

        # Same feasibility condition as check_matern_feasible():
        # lambda_ * pi * h**2 <= 1/e.
        h_upper = min(
            bounds["h"][1],
            self.params["neighbor_distance"] * (1 - 1e-6),
            np.sqrt(1 / (np.e * np.pi * lambda_)) * (1 - 1e-6),
        )

        bounds["h"] = (min(4.0, h_upper / 2), h_upper)
        p_init["h"] = float(np.clip(p_init["h"], *bounds["h"]))

        c_bounds = self.arrays["correlation_bounds"]
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
                weighted_mean = np.nan
                weighted_sd = np.nan
            return weighted_mean, weighted_sd

        p_init["c_diff_mean"], p_init["c_diff_sd"] = weighted_stats(
            c_centers, counts[mid_row:, :, 2]
        )

        low_dist_bin = np.where(self.arrays["distance_bounds"] > p_init["h"])[0][0]
        p_init["c_same_mean"], p_init["c_same_sd"] = weighted_stats(
            c_centers, counts[:low_dist_bin, :, 1].sum(axis=0)
        )
        # print(f"Initial parameter estimates: {p_init}")
        defaults = {
            "p_same": 0.2,
            "h": 8.0,
            "sigma_eff": 1.0,
            "c_diff_mean": 0.5,
            "c_diff_sd": 0.15,
            "c_same_mean": 0.8,
            "c_same_sd": 0.1,
        }

        for key, value in p_init.items():
            if not np.isfinite(value):
                value = defaults[key]

            p_init[key] = float(np.clip(value, *bounds[key]))
        return p_init, bounds

    def build_from_parameters(self, use_cdf=True):
        if not self.parameters:
            raise ValueError(
                "Model parameters must be calculated before building the model."
            )

        p_fit = self.parameters

        self.distributions["pdf"] = match_model(
            list(p_fit.values()),
            lambda_=300 / self.params["L"] ** 2,
            R_cut=self.params["neighbor_distance"],
            nbins=self.params["bins"],
            L=self.params["L"],
            return_1D=True,
        )

        # convert to cumulative for better numerical stability
        def get_cdf(pdf, reverse=False):
            if reverse:
                return np.nancumsum(pdf[::-1])[::-1] / np.nansum(pdf)
            else:
                return np.nancumsum(pdf) / np.nansum(pdf)

        self.distributions["cdf"] = {}
        for key in self.distributions["pdf"].keys():
            reverse = key in ["distance_same", "correlation_diff"]
            self.distributions["cdf"][key] = get_cdf(
                self.distributions["pdf"][key], reverse=reverse
            )

        key_model = "cdf" if use_cdf else "pdf"
        self.p_same = {}
        pdf_NN = self.distributions[key_model]["distance_same"] * p_fit["p_same"]
        pdf_nNN = self.distributions[key_model]["distance_diff"] * (1 - p_fit["p_same"])
        self.p_same["distance"] = nangauss_filter(
            pdf_NN / (pdf_NN + pdf_nNN), sigma=0.5
        )

        pdf_NN = self.distributions[key_model]["correlation_same"] * p_fit["p_same"]
        pdf_nNN = self.distributions[key_model]["correlation_diff"] * (
            1 - p_fit["p_same"]
        )
        self.p_same["correlation"] = nangauss_filter(
            pdf_NN / (pdf_NN + pdf_nNN), sigma=0.5
        )

        # print("could be using cdfs here instead of pdfs for better performance")
        # pdf_NN = np.outer(f_c_same, f_r_same) * p_out["p_same"]
        pdf_NN = (
            self.distributions[key_model]["distance_same"][:, None]
            * self.distributions[key_model]["correlation_same"][None, :]
        ) * p_fit["p_same"]
        pdf_nNN = (
            self.distributions[key_model]["distance_diff"][:, None]
            * self.distributions[key_model]["correlation_diff"][None, :]
        ) * (1 - p_fit["p_same"])

        self.p_same["joint"] = nangauss_filter(pdf_NN / (pdf_NN + pdf_nNN), sigma=0.5)

        self.set_f_same("joint")

        p_same = self.f_same(self.arrays["distance_bounds"], 1.0)

        p_thr = 0.05
        found = False
        while not found:
            idx_low_prob = np.where(p_same < p_thr)[0]
            if len(idx_low_prob) > 0:
                idx_cutoff = idx_low_prob[0]
                found = True

            p_thr += 0.05

        self.distance_cutoff = max(
            10, self.arrays["distance_bounds"][idx_cutoff] * 1.5
        )  ## make sure, also half-detected ones have a chance!
        self.fitted = True
        self.fit_stale = False

    def set_f_same(self, model: str = "joint"):

        if model == "joint":
            self.f_same = lambda distance, correlation: interpolate.interpn(
                (
                    self.arrays["distance"],
                    self.arrays["correlation"],
                ),
                self.p_same["joint"],
                (distance, correlation),
                bounds_error=False,
                fill_value=None,
            )
        else:

            self.f_same = interpolate.interp1d(
                self.arrays[model],
                self.p_same[model],
                # x,
                # bounds_error=False,
                fill_value="extrapolate",
            )

    @staticmethod
    def _from_file(
        path: str | Path,
        fields_to_load: dict[str, dict[str, FieldSpec]] | None = None,
        params: dict | None = None,
    ) -> "Model":
        data = load_file(
            path, fields_to_load, config_name=NATIVE_MODEL_CONFIG, root="/"
        )
        return Model._from_dict(data, params=params)

    @staticmethod
    def _from_dict(data: dict, params: dict | None = None) -> "Model":
        model = Model(params=params)
        model.register_data(**data)
        return model

    def register_data(self, **data):
        names = data["parameters"]["names"]
        values = data["parameters"]["values"]

        parameters = {
            (name.decode("utf-8") if isinstance(name, bytes) else str(name)): value
            for name, value in zip(names, values)
        }
        self.parameters = parameters
        self.loaded = True
        self.build_from_parameters(use_cdf=True)

    def save(
        self,
        path: str | Path,
        *,
        mat_version: Literal["pre73", "7.3"] = "7.3",
    ) -> None:

        fields_to_save = LoadConfig.fields_from_resource(
            NATIVE_MODEL_CONFIG,
            enabled_only=False,
        )

        parameter_names = list(self.parameters.keys())

        parameter_values = np.asarray(list(self.parameters.values()))

        save_data = {
            "parameters": {
                "names": parameter_names,
                "values": parameter_values,
            }
        }

        save_file(
            path,
            save_data,
            fields_to_save,
            mat_version=mat_version,
            root_attributes={"object_type": "ModelData", "format_version": 1},
            root="/",
        )
