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
        penalty = (lambda_ * np.pi * params["h"] ** 2 > 1 - np.e) * 1e6
        return np.full((nbins, nbins), penalty, dtype=float)

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

    if return_1D:

        return {
            "correlation_same": pdf_c_same,
            "correlation_diff": pdf_c_diff,
            "distance_same": pdf_r_same,
            "distance_diff": pdf_r_diff,
        }

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


# def lm_fit_hist_poisson(
#     counts2d,
#     theta0,
#     model_bin_probs,
#     mask2d=None,
#     max_iter=30,
#     fd_rel_step=1e-3,
#     damping0=1e-2,
#     damping_up=10.0,
#     damping_down=0.3,
#     tol_rel_improve=1e-6,
#     tol_step=1e-6,
#     eps=1e-12,
#     alpha_mode="fixed_total",  # or "fit"
# ):
#     """
#     Levenberg–Marquardt / Gauss–Newton style fit for Poisson binned counts.

#     counts2d: histogram counts (H,W)
#     model_bin_probs(theta): returns probs (H,W) normalized over fitted region
#     mask2d: bins included; others ignored
#     alpha_mode:
#         - "fixed_total": alpha = sum(counts in mask)
#         - "fit": treat alpha as an extra parameter by appending log(alpha) to theta
#     """
#     counts2d = np.asarray(counts2d, dtype=float)
#     H, W = counts2d.shape

#     if mask2d is None:
#         mask2d = np.ones_like(counts2d, dtype=bool)
#     else:
#         mask2d = np.asarray(mask2d, dtype=bool)
#         if mask2d.shape != counts2d.shape:
#             raise ValueError("mask2d shape mismatch")

#     y = counts2d[mask2d].astype(float)
#     N = y.sum()
#     if N <= 0:
#         raise ValueError("No counts in masked region.")

#     theta = np.asarray(theta0, dtype=float).copy()

#     # optionally fit alpha by extending parameter vector
#     if alpha_mode == "fit":
#         # initialize alpha at N, optimize log(alpha) to enforce positivity
#         theta = np.concatenate([theta, [np.log(max(N, 1.0))]])
#         fit_alpha = True
#     elif alpha_mode == "fixed_total":
#         fit_alpha = False
#     else:
#         raise ValueError("alpha_mode must be 'fixed_total' or 'fit'")

#     def eval_mu(th):
#         if fit_alpha:
#             th_core = th[:-1]
#             alpha = np.exp(th[-1])
#         else:
#             th_core = th
#             alpha = N

#         probs = np.asarray(model_bin_probs(th_core), dtype=float)
#         if probs.shape != counts2d.shape:
#             print(probs.shape)
#             raise ValueError("model_bin_probs returned wrong shape")
#         probs = np.clip(probs, 0.0, None)
#         probs[~mask2d] = 0.0
#         s = probs.sum()
#         if not np.isfinite(s) or s <= 0:
#             return None  # invalid
#         probs = probs / s
#         mu2d = alpha * probs
#         mu = mu2d[mask2d]
#         mu = np.clip(mu, eps, None)
#         return mu

#     mu = eval_mu(theta)
#     if mu is None:
#         raise ValueError(
#             "Initial theta produces invalid model probabilities in masked region."
#         )

#     dev = poisson_deviance(y, mu, eps=eps)
#     lam = damping0

#     for it in range(max_iter):
#         # Working residuals r = (y - mu)/sqrt(mu)
#         r = (y - mu) / np.sqrt(mu)

#         # Finite-difference Jacobian of mu -> then convert to Jacobian of r
#         p = len(theta)
#         Jmu = np.zeros((y.size, p), dtype=float)

#         for k in range(p):
#             step = fd_rel_step * (abs(theta[k]) + 1.0)
#             th2 = theta.copy()
#             th2[k] += step
#             mu2 = eval_mu(th2)
#             if mu2 is None or np.any(~np.isfinite(mu2)):
#                 # try negative step
#                 th2 = theta.copy()
#                 th2[k] -= step
#                 mu2 = eval_mu(th2)
#                 if mu2 is None:
#                     # if still invalid, set derivative ~0 (or raise)
#                     Jmu[:, k] = 0.0
#                     continue
#                 Jmu[:, k] = (mu - mu2) / step
#             else:
#                 Jmu[:, k] = (mu2 - mu) / step

#         # dr/dtheta = d/dtheta[(y-mu)/sqrt(mu)]
#         # r = y*mu^{-1/2} - mu^{1/2}
#         # dr = -(1/2) y mu^{-3/2} dmu - (1/2) mu^{-1/2} dmu
#         # => dr = -(1/2)( y/mu + 1 ) * dmu / sqrt(mu)
#         factor = -0.5 * (y / mu + 1.0) / np.sqrt(mu)
#         Jr = Jmu * factor[:, None]

#         # LM step: (J^T J + lam I) d = J^T r
#         A = Jr.T @ Jr + lam * np.eye(p)
#         b = Jr.T @ r
#         try:
#             d = np.linalg.solve(A, b)
#         except np.linalg.LinAlgError:
#             lam *= damping_up
#             continue

#         # Propose update
#         theta_new = theta + d
#         mu_new = eval_mu(theta_new)
#         if mu_new is None:
#             lam *= damping_up
#             continue

#         dev_new = poisson_deviance(y, mu_new, eps=eps)

#         if dev_new < dev:  # accept
#             rel_improve = (dev - dev_new) / max(dev, 1.0)
#             theta, mu, dev = theta_new, mu_new, dev_new
#             lam *= damping_down

#             if rel_improve < tol_rel_improve or np.linalg.norm(d) < tol_step:
#                 break
#         else:  # reject
#             lam *= damping_up

#     # unpack alpha if fitted
#     if fit_alpha:
#         theta_hat = theta[:-1]
#         alpha_hat = float(np.exp(theta[-1]))
#     else:
#         theta_hat = theta
#         alpha_hat = float(N)

#     return {
#         "theta_hat": theta_hat,
#         "alpha_hat": alpha_hat,
#         "deviance": float(dev),
#         "mu_hat": mu,  # means for masked bins
#         "n_iter": it + 1,
#         "damping": float(lam),
#     }


## debugging functions

import numpy as np


def check_matern_feasible(lam, h):
    return lam * np.pi * h**2 <= (1 / np.e)


def sanity_report_model_bin_probs(probs, name="probs"):
    rep = {}
    rep["finite"] = bool(np.isfinite(probs).all())
    rep["sum"] = float(np.nansum(probs))
    rep["min"] = float(np.nanmin(probs))
    rep["max"] = float(np.nanmax(probs))
    rep["num_zeros"] = int(np.sum(probs == 0))
    rep["shape"] = probs.shape
    return {name: rep}


def debug_objective_once(
    H,
    r_edges,
    c_edges,
    R_cut,
    L,
    lam,
    h,
    sigma_eff,
    p_same,
    model_bin_probs_conditioned,
    **kwargs,
):
    """
    Calls your model builder once and returns (ok, probs, diagnostics dict).
    """
    diag = {
        "params": {"lam": lam, "h": h, "sigma_eff": sigma_eff, "p_same": p_same},
        "matern_feasible": check_matern_feasible(lam, h),
    }

    try:
        probs = model_bin_probs_conditioned(
            r_edges,
            c_edges,
            R_cut,
            lam=lam,
            h=h,
            sigma_eff=sigma_eff,
            p_same=p_same,
            L=L,
            **kwargs,
        )
    except Exception as e:
        diag["exception"] = repr(e)
        return False, None, diag

    if probs is None:
        diag["probs_is_none"] = True
        return False, None, diag

    diag.update(sanity_report_model_bin_probs(probs, name="model_probs"))

    # Check if probs is effectively empty
    if not np.isfinite(probs).all():
        diag["reason"] = "non-finite probabilities"
        return False, probs, diag
    if probs.sum() <= 0:
        diag["reason"] = "probabilities sum to zero"
        return False, probs, diag

    # Empirical histogram sanity
    H = np.asarray(H, dtype=float)
    diag.update(sanity_report_model_bin_probs(H / np.sum(H), name="empirical_probs"))

    return True, probs, diag


## PLOTS


# import matplotlib.pyplot as plt


# def plot_empirical_vs_model_components(
#     H, r_edges, c_edges, P_same, P_diff, P_joint, title_prefix=""
# ):
#     """
#     Shows 4 panels:
#       empirical, model joint, model same, model diff
#     All are displayed as normalized probabilities per bin.
#     """
#     H = np.asarray(H, dtype=float)
#     Hn = H / H.sum() if H.sum() > 0 else H

#     extent = [c_edges[0], c_edges[-1], r_edges[0], r_edges[-1]]  # x=c, y=r

#     fig, axs = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

#     def show(ax, M, title):
#         im = ax.imshow(
#             M,
#             origin="lower",
#             aspect="auto",
#             extent=extent,
#             interpolation="nearest",
#         )
#         ax.set_xlabel("correlation")
#         ax.set_ylabel("distance (µm)")
#         ax.set_title(title)
#         plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

#     show(axs[0, 0], Hn, f"{title_prefix}Empirical (normalized)")
#     show(axs[0, 1], P_joint, f"{title_prefix}Model joint")
#     show(axs[1, 0], P_same, f"{title_prefix}Model SAME component")
#     show(axs[1, 1], P_diff, f"{title_prefix}Model DIFF component")

#     plt.show()


# def plot_residual_map(H, P_joint, r_edges, c_edges, eps=1e-15, title="Residuals"):
#     """
#     Plot log-ratio residuals: log((H+eps)/(P_joint+eps)) to see where model misfits.
#     """
#     H = np.asarray(H, dtype=float)
#     Hn = H / H.sum() if H.sum() > 0 else H
#     R = np.log((Hn + eps) / (P_joint + eps))

#     extent = [c_edges[0], c_edges[-1], r_edges[0], r_edges[-1]]
#     plt.figure(figsize=(7, 6), constrained_layout=True)
#     im = plt.imshow(
#         R, origin="lower", aspect="auto", extent=extent, interpolation="nearest"
#     )
#     plt.xlabel("correlation")
#     plt.ylabel("distance (µm)")
#     plt.title(title)
#     plt.colorbar(im, fraction=0.046, pad=0.04)
#     plt.show()


# # ----------------------------
# # Example usage
# # ----------------------------
# if __name__ == "__main__":
#     # Suppose you already built a 128x128 histogram H over (r,c)
#     nrb = 128
#     ncb = 128
#     R_cut = 25.0

#     # Example edges:
#     r_edges = np.linspace(0.0, R_cut, nrb + 1)
#     c_edges = np.linspace(0.0, 1.0, ncb + 1)

#     # Fake histogram for demo; replace with your accumulated counts
#     rng = np.random.default_rng(0)
#     H = rng.poisson(1.0, size=(nrb, ncb)).astype(float)

#     # You should pass the actual FOV size in microns
#     L = 500.0

#     res = fit_from_histogram(
#         H,
#         r_edges,
#         c_edges,
#         R_cut=R_cut,
#         L=L,
#         lam_init=0.002,
#         h_init=10.0,
#         sigma_init=3.0,
#         p_same_init=0.02,
#         n_r_grid=1800,
#         n_c_grid=500,
#         n_theta=1024,
#     )

#     print("success:", res.success, res.message)
#     if res.success:
#         print("lambda_hat:", res.lam_hat)
#         print("h_hat:", res.h_hat)
#         print("sigma_eff_hat:", res.sigma_hat)
#         print("p_same_hat:", res.p_same_hat)
#         print("counts used:", res.N_used)
