import numpy as np
from scipy.special import lambertw, i0e
from scipy.stats import norm, lognorm
from scipy.optimize import minimize

import time

# from .distance_model import matern1_parent_intensity, matern1_g, disc_overlap_area
from .correlation_model import pdf_truncated_normal, pdf_reverse_lognormal
from .distance_model import pdf_same_distance, pdf_diff_distance

# ----------------------------
# Build joint pdf on a grid for given params, then bin it
# Condition on r <= R_cut by restricting bins and renormalizing.
# ----------------------------


def timeit(t_ref=None, msg=None, timing=False):
    if not timing:
        return None
    t_new = time.time()
    if msg is not None and t_ref is not None:  # and (self.time_ref):
        print_msg = f"time for {msg}: {(t_new - t_ref) * 10**3:.3f} ms"
        print(print_msg)

    return t_new


def match_model(
    p_in,
    ## other parameters
    lambda_=300 / 512**2,
    R_cut=25.0,
    nbins=128,
    L=512.0,
    grain_factor=1.0,
    return_1D=False,
    timing=False,
):
    names = [
        "p_same",
        "h",
        "sigma_eff",
        "c_diff_mean",
        "c_diff_sd",
        "c_same_mean",
        "c_same_sd",
    ]
    params = {name: val for name, val in zip(names, p_in)}

    def invalid_model(reason):
        if return_1D:
            raise ValueError(reason)

        # fit_histogram_params.objective() rejects nonfinite probabilities.
        return np.full((nbins, nbins), np.nan, dtype=float)

    t_ref = timeit(timing)

    # Only consider bins up to R_cut
    r_edges = np.linspace(0, R_cut, nbins + 1)
    c_edges = np.linspace(0, 1, nbins + 1)

    # grids for integration
    r_grid = np.linspace(0.0, R_cut, int(grain_factor * nbins))
    c_grid = np.linspace(0.0, 1.0, int(grain_factor * nbins))

    t_ref = timeit(t_ref, "match_model: setup time", timing)

    ### distance model
    pdf_r_same = pdf_same_distance(
        r_grid, sigma_eff=params["sigma_eff"], offset=params.get("r_same_offset", 0.0)
    )
    t_ref = timeit(t_ref, "match_model: same distance model time", timing)
    pdf_r_diff = pdf_diff_distance(
        r_grid,
        params["h"],
        lambda_=lambda_,
        sigma=params["sigma_eff"],
        L=L,
        extensions=["hard-core", "window", "blur"],
    )
    t_ref = timeit(t_ref, "match_model: diff distance model time", timing)
    if pdf_r_diff is None:
        return invalid_model("Different-neuron distance PDF is unavailable.")

    # t_ref = timeit(t_ref,"match_model: distance model time")
    ### correlation model
    # if diff_corr_gauss:
    #     pdf_c_diff = pdf_truncated_normal(c_grid, mean=diff_corr_mean, sd=diff_corr_sd)
    # else:

    # pdf_c_diff = pdf_reverse_lognormal(
    #     c_grid, mean=params["c_diff_mean"], sigma=params["c_diff_sd"]
    # )
    pdf_c_diff = pdf_truncated_normal(
        c_grid, mean=params["c_diff_mean"], sd=params["c_diff_sd"]
    )
    pdf_c_same = pdf_reverse_lognormal(
        c_grid, mean=params["c_same_mean"], sigma=params["c_same_sd"]
    )
    t_ref = timeit(t_ref, "match_model: correlation model time", timing)

    distributions = {
        "correlation_same": pdf_c_same,
        "correlation_diff": pdf_c_diff,
        "distance_same": pdf_r_same,
        "distance_diff": pdf_r_diff,
    }

    for name, pdf in distributions.items():
        values = np.asarray(pdf, dtype=float)
        if (
            values.ndim != 1
            or not np.all(np.isfinite(values))
            or np.any(values < 0)
            or values.sum() <= 0
        ):
            return invalid_model(f"Invalid PDF: {name}.")

    if return_1D:
        return distributions

    # integrate into bins (1D) - normalizes all to 1.
    Pr_s = bin_integral_1d(pdf_r_same, r_grid, r_edges)
    Pr_d = bin_integral_1d(pdf_r_diff, r_grid, r_edges)
    Pc_s = bin_integral_1d(pdf_c_same, c_grid, c_edges)
    Pc_d = bin_integral_1d(pdf_c_diff, c_grid, c_edges)

    t_ref = timeit(t_ref, "match_model: 1D bin integration time", timing)

    probs = params["p_same"] * np.outer(Pr_s, Pc_s) + (
        1.0 - params["p_same"]
    ) * np.outer(Pr_d, Pc_d)
    t_ref = timeit(t_ref, "match_model: outer product time", timing)
    return probs


def bin_integral_1d(pdf_on_grid, x_grid, edges):
    """
    Integrate a 1D pdf sampled on x_grid into bins defined by edges using trapezoid+cumsum.
    This is fast and accurate; no per-bin loops.
    """
    pdf = np.asarray(pdf_on_grid, float)
    x = np.asarray(x_grid, float)
    edges = np.asarray(edges, float)

    # cumulative integral via trapezoid on the grid
    dx = np.diff(x)
    area_seg = 0.5 * (pdf[:-1] + pdf[1:]) * dx
    F = np.concatenate([[0.0], np.cumsum(area_seg)])  # F[k] = ∫_{x0}^{x[k]} pdf

    # interpolate cumulative integral at bin edges
    F_edges = np.interp(edges, x, F)
    return np.diff(F_edges)  # per-bin mass


# ----------------------------
# Integrate a sampled 2D pdf over histogram bins
# pdf_rc is sampled on (r_grid, c_grid) as pdf_rc[i_r, i_c].
# ----------------------------


import numpy as np
from scipy.optimize import minimize

# ----------------------------
# Likelihoods for sparse histograms
# ----------------------------


def nll_multinomial(counts, probs, eps=1e-15):
    """
    Multinomial composite likelihood (conditioning on total N).
    counts: 2D nonnegative counts (can be floats if aggregated)
    probs:  2D model bin probabilities (should sum to 1; we'll renormalize)
    """
    counts = np.asarray(counts, dtype=float)
    probs = np.asarray(probs, dtype=float)

    if probs.shape != counts.shape:
        raise ValueError(
            f"Shape mismatch: probs {probs.shape} vs counts {counts.shape}"
        )

    p = np.clip(probs, eps, 1.0)
    p = p / p.sum()
    return -np.sum(counts * np.log(p))


def nll_poisson(counts, probs, alpha=None, eps=1e-15):
    """
    Independent Poisson likelihood per bin:
      N_ij ~ Poisson(mu_ij),  mu_ij = alpha * p_ij
    If alpha is None, we profile it by setting alpha = sum(counts).
    """
    counts = np.asarray(counts, dtype=float)
    probs = np.asarray(probs, dtype=float)

    if probs.shape != counts.shape:
        raise ValueError(
            f"Shape mismatch: probs {probs.shape} vs counts {counts.shape}"
        )

    p = np.clip(probs, eps, 1.0)
    p = p / p.sum()

    if alpha is None:
        alpha = counts.sum()

    mu = alpha * p
    # NLL up to constants: sum(mu - n log mu)
    return np.sum(mu - counts * np.log(np.clip(mu, eps, None)))


# ----------------------------
# Generic fitter
# ----------------------------


def fit_histogram_params(
    counts,
    theta0,
    model_bin_probs,
    *,
    method="multinomial",
    bounds=None,
    mask=None,
    eps=1e-15,
    optimizer="L-BFGS-B",
    options=None,
    return_pred=True,
):
    """
    Fit parameters theta to a 2D histogram via multinomial or Poisson NLL.

    Parameters
    ----------
    counts : (H,W) array
        Empirical histogram counts (can be aggregated across sessions). Zeros allowed.
    theta0 : 1D array-like
        Initial parameter guess in the parameterization expected by model_bin_probs(theta).
    model_bin_probs : callable
        Function: theta -> probs (H,W), already normalized over the bins you want to fit
        (or at least nonnegative; we renormalize inside likelihood).
    method : {"multinomial","poisson"}
        Which likelihood to use.
    bounds : list of (low, high) or None
        Bounds in theta space (for L-BFGS-B).
    mask : (H,W) bool array or None
        If provided, fit only these bins (e.g., r<=R_cut region). Others are ignored.
    eps : float
        Numerical floor for probabilities.
    return_pred : bool
        If True, attach best-fit predicted probs to result.

    Returns
    -------
    res : scipy OptimizeResult with fields:
        - theta_hat
        - nll_hat
        - probs_hat (optional)
    """
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 2:
        raise ValueError("counts must be a 2D array")

    if mask is None:
        mask = np.ones_like(counts, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != counts.shape:
            raise ValueError("mask must have same shape as counts")

    counts_use = counts[mask]
    if counts_use.sum() <= 0:
        raise ValueError("No counts in the masked region.")

    def objective(theta):
        probs = model_bin_probs(theta)
        probs = np.asarray(probs, dtype=float)

        if probs.shape != counts.shape:
            raise ValueError(
                f"model_bin_probs returned shape {probs.shape}, expected {counts.shape}"
            )

        probs_use = probs[mask]

        # invalid model -> large penalty
        if (
            np.any(probs_use < 0)
            or not np.all(np.isfinite(probs_use))
            or probs_use.sum() <= 0
        ):
            return 1e30

        if method == "multinomial":
            return nll_multinomial(counts_use, probs_use, eps=eps)
        elif method == "poisson":
            return nll_poisson(counts_use, probs_use, alpha=counts_use.sum(), eps=eps)
        else:
            raise ValueError("method must be 'multinomial' or 'poisson'")

    res = minimize(
        objective,
        x0=np.asarray(theta0, dtype=float),
        method=optimizer,
        bounds=bounds,
        options=options or {"maxiter": 500},
    )

    res.theta_hat = res.x
    res.nll_hat = float(res.fun)

    if return_pred:
        probs_hat = np.asarray(model_bin_probs(res.theta_hat), dtype=float)
        # Renormalize over mask (so it's directly comparable to counts_use / sum)
        probs_hat = np.clip(probs_hat, 0.0, None)
        probs_hat[~mask] = 0.0
        s = probs_hat.sum()
        if s > 0:
            probs_hat /= s
        res.probs_hat = probs_hat

    return res


import numpy as np


def poisson_deviance(y, mu, eps=1e-12):
    """Poisson deviance (2 * negative log-likelihood ratio), robust for zeros."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    mu = np.clip(mu, eps, None)
    term = np.zeros_like(y)
    nz = y > 0
    term[nz] = y[nz] * np.log(y[nz] / mu[nz])
    return 2.0 * np.sum(mu - y + term)


import numpy as np


def check_matern_feasible(lam, h):
    return lam * np.pi * h**2 <= (1 / np.e)
