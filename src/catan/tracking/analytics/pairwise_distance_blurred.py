import numpy as np
from scipy.special import lambertw, i0e
from scipy.optimize import minimize

# ----------------------------
# Matérn hard-core I
# ----------------------------


def matern1_parent_intensity(lam, h):
    """
    lam = retained intensity
    h   = hard-core distance
    Matérn-I: lam = lam_p * exp(-lam_p * pi h^2)
    """
    x = -lam * np.pi * h**2
    wp = lambertw(x, k=0)
    lam_p = -(wp.real) / (np.pi * h**2)
    if not np.isfinite(lam_p) or lam_p <= 0:
        raise ValueError("Invalid lam_p. Check that lam*pi*h^2 is feasible (<= ~1/e).")
    return lam_p


def disc_overlap_area(r, h):
    r = np.asarray(r, dtype=float)
    A = np.zeros_like(r)
    mask = (r >= 0) & (r < 2 * h)
    rm = r[mask]
    arg = np.clip(rm / (2 * h), -1.0, 1.0)
    A[mask] = 2 * h**2 * np.arccos(arg) - 0.5 * rm * np.sqrt(
        np.clip(4 * h**2 - rm**2, 0.0, None)
    )
    return A


def matern1_g(r, lam_p, h):
    r = np.asarray(r, dtype=float)
    g = np.ones_like(r)
    g[r < h] = 0.0
    mid = (r >= h) & (r < 2 * h)
    if np.any(mid):
        A = disc_overlap_area(r[mid], h)
        g[mid] = np.exp(lam_p * A)
    g[r >= 2 * h] = 1.0
    return g


# ----------------------------
# Square boundary factor
# ----------------------------


def square_overlap_area(L, dx, dy):
    ax = np.maximum(0.0, L - np.abs(dx))
    ay = np.maximum(0.0, L - np.abs(dy))
    return ax * ay


def square_directional_overlap_average(r, L, n_theta=4096):
    """
    Abar(r) = (1/2pi) \int |W ∩ (W-u)| dθ, u=r(cosθ,sinθ)
    """
    r = np.asarray(r, dtype=float)
    theta = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    c = np.cos(theta)
    s = np.sin(theta)
    dx = r[..., None] * c[None, :]
    dy = r[..., None] * s[None, :]
    A = square_overlap_area(L, dx, dy)
    return A.mean(axis=-1)


# ----------------------------
# True different-neuron distance PDF in square
# ----------------------------


def true_pdf_square_matern1(d_grid, lam, h, L, n_theta=4096):
    """
    f_true(d) ∝ d * g(d) * Abar(d), normalized on [0, sqrt(2)L].
    """
    lam_p = matern1_parent_intensity(lam, h)
    d_grid = np.asarray(d_grid, dtype=float)
    r_max = np.sqrt(2) * L

    if d_grid[0] < 0 or d_grid[-1] > r_max + 1e-9:
        raise ValueError("d_grid should lie within [0, sqrt(2)*L].")

    g = matern1_g(d_grid, lam_p, h)
    Abar = square_directional_overlap_average(d_grid, L, n_theta=n_theta)
    shape = d_grid * g * Abar

    Z = np.trapezoid(shape, d_grid)
    if Z <= 0:
        raise ValueError("Normalization failed; check lam, h, L.")
    return shape / Z


# ----------------------------
# Rice blur with stable Bessel
# ----------------------------


def rice_kernel_stable(r, d, tau):
    """
    Rice kernel for magnitude when vector noise ~ N(0, tau^2 I2):
      p(r|d,tau) = (r/tau^2) exp(-(r^2+d^2)/(2tau^2)) I0(r d / tau^2)

    Uses i0e for stability:
      I0(x) = i0e(x) * exp(|x|)
    """
    r = np.asarray(r, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    rr = r[..., None]
    dd = d[None, :]

    if tau <= 0:
        raise ValueError("tau must be > 0")

    x = (rr * dd) / (tau**2)
    expo = -(rr**2 + dd**2) / (2.0 * tau**2) + np.abs(x)
    return (rr / (tau**2)) * np.exp(expo) * i0e(x)


def blur_true_pdf_to_observed(f_true_d, d_grid, r_grid, tau):
    """
    f_obs(r) = ∫ f_true(d) p(r|d,tau) dd
    """
    K = rice_kernel_stable(r_grid, d_grid, tau)  # (len(r), len(d))
    f_obs = np.trapezoid(K * f_true_d[None, :], d_grid, axis=1)
    # normalize on r_grid support
    Z = np.trapezoid(f_obs, r_grid)
    if Z > 0:
        f_obs /= Z
    return f_obs


# ----------------------------
# Predicted PDFs you asked for
# ----------------------------


def predict_pair_pdfs(d_grid, r_grid, lam, h, L, sigma_E, sigma_M, n_theta=4096):
    """
    Returns:
      f_true(d)          : true different-neuron distance pdf in the square
      f_within(r)        : observed within-session pair distance pdf (measurement only)
      f_across(r)        : observed across-session pair distance pdf (measurement + motion)
    """
    f_true = true_pdf_square_matern1(d_grid, lam, h, L, n_theta=n_theta)

    # tau values (per-axis std of the *pairwise displacement* noise)
    tau_within = np.sqrt(2.0) * sigma_E
    tau_across = np.sqrt(2.0 * (sigma_E**2 + sigma_M**2))

    f_within = blur_true_pdf_to_observed(f_true, d_grid, r_grid, tau_within)
    f_across = blur_true_pdf_to_observed(f_true, d_grid, r_grid, tau_across)
    return f_true, f_within, f_across


# ----------------------------
# Fitting sigma_E and sigma_M from measured data
# ----------------------------


def hist_probs_from_pdf(pdf_vals, grid, bin_edges):
    """
    Convert a sampled pdf on 'grid' into bin probabilities by integrating per bin.
    """
    # Ensure pdf normalized on grid
    Z = np.trapezoid(pdf_vals, grid)
    if Z <= 0:
        raise ValueError("pdf_vals has non-positive integral.")
    pdf = pdf_vals / Z

    # integrate pdf over bins by trapezoid on a fine grid that includes edges
    probs = np.zeros(len(bin_edges) - 1, dtype=float)
    for k in range(len(probs)):
        a, b = bin_edges[k], bin_edges[k + 1]
        # select grid segment within [a,b]
        mask = (grid >= a) & (grid <= b)
        if not np.any(mask):
            # approximate by interpolation endpoints
            # (rare if grid is dense)
            xs = np.array([a, b])
            ys = np.interp(xs, grid, pdf, left=0.0, right=0.0)
            probs[k] = np.trapezoid(ys, xs)
        else:
            xs = grid[mask]
            ys = pdf[mask]
            # ensure bin edges included
            if xs[0] > a:
                xs = np.r_[a, xs]
                ys = np.r_[np.interp(a, grid, pdf), ys]
            if xs[-1] < b:
                xs = np.r_[xs, b]
                ys = np.r_[ys, np.interp(b, grid, pdf)]
            probs[k] = np.trapezoid(ys, xs)

    # renormalize to guard numerical drift
    s = probs.sum()
    if s > 0:
        probs /= s
    return probs


def multinomial_nll(counts, probs, eps=1e-15):
    probs = np.clip(probs, eps, 1.0)
    probs /= probs.sum()
    return -np.sum(counts * np.log(probs))


def fit_sigmas_from_pair_distances(
    distances_within,
    distances_across,
    lam,
    h,
    L,
    sigmaE_init=2.0,
    sigmaM_init=2.0,
    n_theta=2048,
    grid_size=5000,
    n_bins=80,
):
    """
    Fit sigma_E and sigma_M by minimizing binned multinomial negative log-likelihood
    for BOTH within-session and across-session pair-distance samples.

    Note: pair distances are not independent; this is a "pseudo-likelihood" fit.
    It works well in practice if you subsample pairs (recommended).
    """
    distances_within = np.asarray(distances_within, dtype=float)
    distances_across = np.asarray(distances_across, dtype=float)

    r_max = np.sqrt(2) * L

    # restrict to feasible range
    dw = distances_within[(distances_within >= 0) & (distances_within <= r_max)]
    da = distances_across[(distances_across >= 0) & (distances_across <= r_max)]

    # hist bins shared
    bin_edges = np.linspace(0.0, r_max, n_bins + 1)
    counts_w, _ = np.histogram(dw, bins=bin_edges)
    counts_a, _ = np.histogram(da, bins=bin_edges)

    # grids for prediction (dense for stable integration)
    d_grid = np.linspace(0.0, r_max, grid_size)
    r_grid = np.linspace(0.0, r_max, grid_size)

    # cache f_true because it does not depend on sigmas
    f_true = true_pdf_square_matern1(d_grid, lam, h, L, n_theta=n_theta)

    def objective(params):
        sigma_E, sigma_M = params
        # enforce positivity
        if sigma_E <= 1e-9 or sigma_M < 0:
            return 1e30

        tau_within = np.sqrt(2.0) * sigma_E
        tau_across = np.sqrt(2.0 * (sigma_E**2 + sigma_M**2))

        f_w = blur_true_pdf_to_observed(f_true, d_grid, r_grid, tau_within)
        f_a = blur_true_pdf_to_observed(f_true, d_grid, r_grid, tau_across)

        p_w = hist_probs_from_pdf(f_w, r_grid, bin_edges)
        p_a = hist_probs_from_pdf(f_a, r_grid, bin_edges)

        return multinomial_nll(counts_w, p_w) + multinomial_nll(counts_a, p_a)

    x0 = np.array([sigmaE_init, sigmaM_init], dtype=float)
    bounds = [(1e-6, None), (0.0, None)]

    res = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B")
    return res


# ----------------------------
# Same-neuron cross-session distance PDF (optional)
# ----------------------------


def same_neuron_distance_pdf(r, sigma_E, sigma_M):
    """
    Same-neuron displacement between sessions:
      Delta = M + (E2 - E1) ~ N(0, (sigma_M^2 + 2 sigma_E^2) I2)
    => R = ||Delta|| Rayleigh(scale = sqrt(sigma_M^2 + 2 sigma_E^2))
    """
    r = np.asarray(r, dtype=float)
    s = np.sqrt(sigma_M**2 + 2.0 * sigma_E**2)
    pdf = np.zeros_like(r)
    m = r >= 0
    rr = r[m]
    pdf[m] = (rr / s**2) * np.exp(-(rr**2) / (2.0 * s**2))
    return pdf
