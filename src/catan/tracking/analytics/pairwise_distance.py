import numpy as np
from scipy.special import lambertw, i0

# ----------------------------
# Matérn hard-core I utilities
# ----------------------------


def matern1_parent_intensity(lam, h):
    """
    Given retained intensity lam (points/area) and hard-core distance h,
    compute parent Poisson intensity lam_p for Matérn hard-core I:
        lam = lam_p * exp(-lam_p * pi * h^2)

    Solution uses Lambert W:
        lam_p = -(1/(pi h^2)) * W(-lam pi h^2)
    """
    x = -lam * np.pi * h**2
    # principal branch typically works for realistic lam*pi*h^2 <= 1/e
    wp = lambertw(x, k=0)
    lam_p = -(wp.real) / (np.pi * h**2)
    if lam_p <= 0:
        raise ValueError("Computed parent intensity is non-positive; check lam and h.")
    return lam_p


def disc_overlap_area(r, h):
    """
    Overlap area of two discs of radius h whose centers are separated by distance r.
    Closed form for 0 <= r <= 2h; returns 0 for r >= 2h.
    """
    r = np.asarray(r, dtype=float)
    A = np.zeros_like(r)

    mask = (r >= 0) & (r < 2 * h)
    rm = r[mask]

    # Avoid tiny numerical issues at the endpoints
    # Clip argument of arccos into [-1, 1]
    arg = np.clip(rm / (2 * h), -1.0, 1.0)

    A[mask] = 2 * h**2 * np.arccos(arg) - 0.5 * rm * np.sqrt(
        np.clip(4 * h**2 - rm**2, 0.0, None)
    )
    return A


def matern1_g(r, lam_p, h):
    """
    Pair correlation function g(r) for Matérn hard-core I:
        g(r)=0 for r<h
        g(r)=exp(lam_p * A_overlap(r)) for h <= r < 2h
        g(r)=1 for r >= 2h
    """
    r = np.asarray(r, dtype=float)
    g = np.ones_like(r)

    g[r < h] = 0.0
    mid = (r >= h) & (r < 2 * h)
    if np.any(mid):
        A = disc_overlap_area(r[mid], h)
        g[mid] = np.exp(lam_p * A)

    g[r >= 2 * h] = 1.0
    return g


# ---------------------------------------
# Infinite-plane expected pair distance
# ---------------------------------------


def expected_pair_density_infinite(r, lam, h):
    """
    Infinite-plane expected (unnormalized) pair-distance density shape
    for unordered pairs per unit area:
        dη(r)/dr = π * lam^2 * r * g(r)

    Returns the density value for each r (same units: pairs/(area * distance)).
    """
    lam_p = matern1_parent_intensity(lam, h)
    g = matern1_g(r, lam_p, h)
    return np.pi * lam**2 * np.asarray(r, dtype=float) * g


def expected_pair_pdf_infinite_over_range(r, lam, h, r_max):
    """
    Since the infinite plane has no natural normalization over r∈[0,∞),
    this returns a *normalized* pdf on [0, r_max] by dividing by the integral.

    Useful if you want to compare to a histogram computed with a cutoff.
    """
    r = np.asarray(r, dtype=float)
    dens = expected_pair_density_infinite(r, lam, h)

    # Normalize over [0, r_max]
    rr = np.linspace(0.0, r_max, 5000)
    dd = expected_pair_density_infinite(rr, lam, h)
    Z = np.trapezoid(dd, rr)
    if Z <= 0:
        raise ValueError("Normalization integral is non-positive; check parameters.")
    return dens / Z


# ---------------------------------------
# Finite square window boundary correction
# ---------------------------------------


def square_overlap_area(L, dx, dy):
    """
    Overlap area |W ∩ (W - (dx,dy))| for a square window W of side L,
    axis-aligned, when shifted by (dx, dy).
    """
    ax = np.maximum(0.0, L - np.abs(dx))
    ay = np.maximum(0.0, L - np.abs(dy))
    return ax * ay


def square_directional_overlap_average(r, L, n_theta=4096):
    """
    Compute:
        Abar_W(r) = (1/2π) ∫_0^{2π} |W ∩ (W - u)| dθ,   u = r(cosθ, sinθ)

    Uses a high-resolution trapezoidal rule over θ, which is very accurate
    for this smooth 2π-periodic integrand.
    """
    r = np.asarray(r, dtype=float)

    theta = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    c = np.cos(theta)
    s = np.sin(theta)

    # Broadcast over r
    dx = r[..., None] * c[None, ...]
    dy = r[..., None] * s[None, ...]
    A = square_overlap_area(L, dx, dy)

    # Average over theta (uniform)
    return A.mean(axis=-1)


def expected_pair_density_square(r, lam, h, L, n_theta=4096):
    """
    Expected (unnormalized) pair-distance density inside a square window of side L:
        dN_W(r)/dr = π * lam^2 * r * g(r) * Abar_W(r)

    where Abar_W(r) is the directional average overlap area.
    Units: pairs / distance (because total pairs in the window).
    """
    lam_p = matern1_parent_intensity(lam, h)
    r = np.asarray(r, dtype=float)
    g = matern1_g(r, lam_p, h)
    Abar = square_directional_overlap_average(r, L, n_theta=n_theta)
    return np.pi * lam**2 * r * g * Abar


def expected_pair_pdf_square(r, lam, h, L, n_theta=4096):
    """
    Properly normalized pair-distance pdf for unordered pairs in a square window.
    Support is r in [0, sqrt(2)*L].
    """
    r = np.asarray(r, dtype=float)
    r_max = np.sqrt(2) * L

    dens = expected_pair_density_square(r, lam, h, L, n_theta=n_theta)

    rr = np.linspace(0.0, r_max, 6000)
    dd = expected_pair_density_square(rr, lam, h, L, n_theta=n_theta)
    Z = np.trapezoid(dd, rr)
    if Z <= 0:
        raise ValueError("Normalization integral is non-positive; check parameters.")
    return dens / Z


# ---- Gaussian localization blur: Rice-mixture transform ----
from scipy.special import i0e


def rice_kernel_stable(r, d, sigma):
    """
    Stable p(r|d) for 2D isotropic Gaussian noise with per-point std sigma.

    Uses i0e(x) = i0(x) * exp(-|x|) to avoid overflow:
      p(r|d) = (r/(2σ^2)) * exp(-(r^2+d^2)/(4σ^2) + |x|) * i0e(x),
      x = r d / (2σ^2)
    """
    r = np.asarray(r, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    rr = r[..., None]
    dd = d[None, ...]

    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    denom = 2.0 * sigma**2
    x = (rr * dd) / denom  # = r d / (2 σ^2)

    # exponent part: -(r^2+d^2)/(4σ^2) + |x|
    expo = -(rr**2 + dd**2) / (4.0 * sigma**2) + np.abs(x)

    return (rr / denom) * np.exp(expo) * i0e(x)


# def rice_kernel(r, d, sigma):
#     """
#     p(r | d) for 2D isotropic Gaussian noise with per-point std sigma.
#     Pairwise displacement noise std is sqrt(2)*sigma, which yields:
#         p(r|d) = (r/(2sigma^2)) exp(-(r^2+d^2)/(4sigma^2)) I0(rd/(2sigma^2))
#     """
#     r = np.asarray(r, dtype=float)
#     d = np.asarray(d, dtype=float)
#     # broadcasting: r[...,None] with d[None,...] typically
#     rr = r[..., None]
#     dd = d[None, ...]
#     denom = 2.0 * sigma**2
#     return (
#         (rr / denom)
#         * np.exp(-(rr**2 + dd**2) / (4.0 * sigma**2))
#         * i0((rr * dd) / (2.0 * sigma**2))
#     )


def blur_distance_pdf(f_true_d, d_grid, r_grid, sigma):
    """
    Given a true distance pdf f_true(d) sampled on d_grid,
    return observed distance pdf f_obs(r) sampled on r_grid:
        f_obs(r) = ∫ f_true(d) p(r|d) dd

    Uses trapezoidal integration over d_grid.
    """
    d_grid = np.asarray(d_grid, dtype=float)
    r_grid = np.asarray(r_grid, dtype=float)
    f_true_d = np.asarray(f_true_d, dtype=float)

    K = rice_kernel_stable(r_grid, d_grid, sigma)  # shape (len(r), len(d))
    integrand = K * f_true_d[None, :]
    f_obs = np.trapezoid(integrand, d_grid, axis=1)

    # numerical cleanup: normalize over r_grid range
    Z = np.trapezoid(f_obs, r_grid)
    if Z > 0:
        f_obs = f_obs / Z
    return f_obs


# import numpy as np


def same_neuron_distance_scale(sigma_E1=0.0, sigma_E2=0.0, sigma_M=0.0, sigma_eff=None):
    """
    Returns the Rayleigh scale parameter s (std per axis of the 2D displacement vector).

    If sigma_eff is provided, it overrides the components and is interpreted as:
      Delta ~ N(0, sigma_eff^2 I2)  ->  R = ||Delta|| is Rayleigh(scale=sigma_eff)
    """
    if sigma_eff is not None:
        if sigma_eff <= 0:
            raise ValueError("sigma_eff must be > 0")
        return float(sigma_eff)

    s2 = sigma_M**2 + sigma_E1**2 + sigma_E2**2
    if s2 <= 0:
        raise ValueError("Need at least one positive sigma component.")
    return float(np.sqrt(s2))


def rayleigh_pdf(r, scale):
    """
    Rayleigh pdf: f(r) = (r/scale^2) * exp(-r^2/(2 scale^2)), r>=0
    """
    r = np.asarray(r, dtype=float)
    f = np.zeros_like(r)
    mask = r >= 0
    rr = r[mask]
    s2 = scale**2
    f[mask] = (rr / s2) * np.exp(-(rr**2) / (2 * s2))
    return f


def rayleigh_cdf(r, scale):
    r = np.asarray(r, dtype=float)
    F = np.zeros_like(r)
    mask = r >= 0
    rr = r[mask]
    F[mask] = 1.0 - np.exp(-(rr**2) / (2 * scale**2))
    return F


def distance_threshold_for_probability(p, scale):
    """
    Invert CDF: find r such that P(R <= r) = p.
    """
    if not (0 < p < 1):
        raise ValueError("p must be in (0,1)")
    return scale * np.sqrt(2.0 * np.log(1.0 / (1.0 - p)))


if __name__ == "__main__":
    # Example: one effective sigma for the 2D displacement vector
    sigma_eff = 3.0  # um

    s = same_neuron_distance_scale(sigma_eff=sigma_eff)

    r = np.linspace(0, 30, 1000)
    pdf = rayleigh_pdf(r, s)

    print("Integral ~", np.trapezoid(pdf, r))
    print("Median distance:", distance_threshold_for_probability(0.5, s))
    print("95% within:", distance_threshold_for_probability(0.95, s))


# ----------------------------
# Example usage
# ----------------------------
# if __name__ == "__main__":
#     # Example parameters (choose your own):
#     lam = 0.0025  # points per µm^2 (example)
#     h = 10.0  # µm minimum distance
#     L = 500.0  # µm window side

#     r = np.linspace(0, np.sqrt(2) * L, 2000)

#     # Infinite-plane shape (normalize on [0, r_max] for plotting convenience)
#     pdf_inf = expected_pair_pdf_infinite_over_range(r, lam, h, r_max=np.sqrt(2) * L)

#     # Finite square window pdf (edge-corrected)
#     pdf_sq = expected_pair_pdf_square(r, lam, h, L)

#     # Now pdf_inf and pdf_sq can be compared to histograms
#     print("pdf_inf integral (approx):", np.trapezoid(pdf_inf, r))
#     print("pdf_sq  integral (approx):", np.trapezoid(pdf_sq, r))
