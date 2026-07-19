import numpy as np
from scipy.special import lambertw, i0e

# --------------------------------------------------------------
# ----------- Extension I: exclude closeby neighbors -----------
# --------------------------------------------------------------
# Matérn-I g(r): hard-core model
# --------------------------------------------------------------


def matern1_parent_intensity(lambda_, h):
    """
    calculates the parent intensity lambda_p necessary to obtain effective intensity lambda
    """
    # lam = lam_p * exp(-lam_p*pi*h^2)
    x = -lambda_ * np.pi * h**2
    wp = lambertw(x, k=0)
    return -(wp.real) / (np.pi * h**2)


def disc_overlap_area(r, h):
    """
    evaluates the overlap area of two discs of radius h separated by distance r ("lens" area)
    """
    r = np.asarray(r, dtype=float)
    A = np.zeros_like(r)
    m = (r >= 0) & (r < 2 * h)
    rm = r[m]
    arg = np.clip(rm / (2 * h), -1.0, 1.0)
    A[m] = 2 * h**2 * np.arccos(arg) - 0.5 * rm * np.sqrt(
        np.clip(4 * h**2 - rm**2, 0.0, None)
    )
    return A


def matern1_g(r, lambda_, h, parent_density=False):
    """
    evaluates the pair correlation function g(r) for Matérn hard-core I
    takes into account all three regimes:
        * r<h (exclusion)
        * h<=r<2h (transition/clustering)
        * r>=2h (independence)
    """

    if parent_density:
        lambda_p = lambda_
    else:
        lambda_p = matern1_parent_intensity(lambda_, h)

    r = np.asarray(r, dtype=float)
    g = np.ones_like(r)
    g[r < h] = 0.0
    mid = (r >= h) & (r < 2 * h)
    if np.any(mid):
        g[mid] = np.exp(lambda_p * disc_overlap_area(r[mid], h))
    g[r >= 2 * h] = 1.0
    return g


# --------------------------------------------------------------
# ---------- Extension II: Finite imaging window size ----------
# --------------------------------------------------------------
# Square window boundary factor Abar(r) (numerical solution)
# For small r << L, Abar(r) ~ L^2 (-4L/pi*r + r^2/pi) so you can often skip this.
# Still included for correctness.
# --------------------------------------------------------------


def Abar_squared_window(r, L: float, mode: str = "exact"):
    """
    evaluates the average overlap area Abar(r) in a squared imaging window of side L x L
    """

    if mode == "exact":

        r = np.asarray(r, dtype=float)
        Abar = np.zeros_like(r)

        # 0 <= r <= L
        m1 = (r >= 0) & (r <= L)
        Abar[m1] = L**2 - (4 * L / np.pi) * r[m1] + (1 / np.pi) * r[m1] ** 2

        # L < r < sqrt(2)L
        m2 = (r > L) & (r < np.sqrt(2) * L)
        rr = r[m2]
        a = L / rr
        Abar[m2] = (2 / np.pi) * (
            L**2 * (np.pi / 2 - 2 * np.arccos(a))
            - 2 * L * rr * (a - np.sqrt(1 - a**2))
            + (L**2 - rr**2 / 2)
        )

        # r >= sqrt(2)L -> 0 already
        return Abar

    elif mode == "numeric":
        n_theta: int = 2048

        r = np.asarray(r, dtype=float)
        theta = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
        c = np.cos(theta)
        s = np.sin(theta)
        dx = r[..., None] * c[None, :]
        dy = r[..., None] * s[None, :]
        ax = np.maximum(0.0, L - np.abs(dx))
        ay = np.maximum(0.0, L - np.abs(dy))
        Abar = (ax * ay).mean(axis=-1)
        return Abar
    else:
        raise ValueError("mode must be 'exact' or 'numeric'")


# --------------------------------------------------------------
# --------- Extension III: Centroid localization noise ---------
# --------------------------------------------------------------
# Rice blur (stable)
# sigma is per-axis std of the 2D noise added to the displacement vector
# --------------------------------------------------------------


def rice_kernel_stable(r, d, sigma):
    """
    r: observed distance(s), shape (n_r,)
    d: true distance(s), shape (n_d,)
    """
    r = np.asarray(r, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    rr = r[..., None]
    dd = d[None, :]

    x = (rr * dd) / (sigma**2)
    expo = -(rr**2 + dd**2) / (2.0 * sigma**2) + np.abs(x)
    return (rr / (sigma**2)) * np.exp(expo) * i0e(x)


def blur_distance_pdf(f_true_d, d_grid, r_grid, sigma):
    """
    r: observed distance(s), shape (n_r,)
    d: true distance(s), shape (n_d,)
    """
    K = rice_kernel_stable(r_grid, d_grid, sigma)  # (len(r), len(d))

    ## integrate over d to evaluate blurring
    f_obs = np.trapezoid(K * f_true_d[None, :], d_grid, axis=1)

    ## calculate and apply normalization factor
    Z = np.trapezoid(f_obs, r_grid)
    if Z > 0:
        f_obs /= Z
    return f_obs


# --------------------------------------------------------------
# ---- Same-neuron distance pdf: Rayleigh(scale=sigma_eff) ----
# --------------------------------------------------------------
# -------------------------------------------------------------
# Same-neuron distance pdf: Rayleigh(scale=sigma_eff)
# Here sigma_eff is per-axis std of the *2D displacement vector* between sessions.
# -------------------------------------------------------------


def pdf_same_distance(d_grid, sigma_eff, offset=0.0):
    """
    Rayleigh distribution
    already normalized on [0, inf)

    kinda arbitrary offset term (slope) added to allow for some larger distances between matching footprints:
    * linear slope decaying to 0 @ h=10 - should be a bit more rooted in empirics / theory?!
    """
    d = np.asarray(d_grid, dtype=float)
    out = np.zeros_like(d)
    m = d >= 0
    rr = d[m]
    out[m] = (rr / sigma_eff**2) * np.exp(-(rr**2) / (2.0 * sigma_eff**2))
    # out += offset
    out[m] += np.maximum(0, offset - offset / 10 * rr)
    return out / np.trapezoid(out, d_grid)


# ----------------------------
# True diff-neuron distance pdf in square: f(d) ∝ d g(d) Abar(d)
# normalized on [0, r_max_model]
# ----------------------------


def pdf_diff_distance(
    d_grid,
    h: float,
    lambda_: float,
    sigma: float = 1.0,
    L: float = 512.0,
    extensions=["hard-core", "window", "blur"],
    offset=0.0,
):
    """
    distribution of distances between different neurons implementing
    * Matérn-I process with intensity lambda_ and hard-core distance h
    * finite square imaging window with boundary factor Abar(d)
    """
    r_grid = d_grid
    if "blur" in extensions:
        ## ensure blur is not applied to upper limit
        d_grid = np.concatenate([d_grid, d_grid[-1] + d_grid[1 : int(len(d_grid) / 5)]])

    if "hard-core" in extensions:
        # feasibility for Matérn-I
        if lambda_ * np.pi * h**2 > 1 / np.e:
            return None
        
        g = matern1_g(d_grid, lambda_, h)
    else:
        g = 1.0

    if "window" in extensions:
        window_factor = Abar_squared_window(d_grid, L=L, mode="exact")
    else:
        window_factor = 1.0

    ## evaluate normalized pdf of different-neuron distances
    f = np.pi * lambda_**2 * g * d_grid * window_factor + offset

    if "blur" in extensions:
        # print(d_grid)
        # d_grid = np.concatenate([d_grid, d_grid[-1] + d_grid[1:]])
        f = blur_distance_pdf(f, d_grid, r_grid, sigma=sigma)
        return f / np.trapezoid(f, r_grid)
    else:
        return f / np.trapezoid(f, d_grid)
