"""Two-noise matching model: detection noise and neuron movement.

Evidence (all footprint pairs within the neighbour distance R):
  "same2"  delta s = 0, pairs of different footprints of one session:
           centroid distance, shift-optimized correlation.
  "cross2" delta s = 1: nearest neighbours by |optimal shift|, all other
           pairs by centroid distance; channels total / NN / non-NN.
  "vector" delta s = 1, pairs with shift distance below
           vector_max_distance: optimal shift minus centroid displacement
           (2D histogram), mostly the detection noise of same neurons.

Model (distances in px, correlation c in [c_min, 1], u = 1 - c):
  different neurons: Matérn-I hard core h (neuron sizes: h ~ N(h, sigma_h))
      in a slab of thickness T (10 µm, fixed), whose projection allows
      d < h with probability P(d^2 + dz^2 >= h^2), dz triangular on
      [0, T]; square-window edge correction; intensity lambda measured
      from neuron counts. Blurred by sqrt(2) sigma_det (delta s = 0) or
      sqrt(2 sigma_det^2 + sigma_move^2) (delta s = 1).
  same neurons (delta s = 1): 2D Student-t displacement (heterogeneous
      movement), scale sqrt(2 sigma_det^2 + sigma_move^2), nu_move.
  correlation: Gamma distributions on u per class; for same neurons the
      mean grows with the displacement, u_same_mean (1 + u_same_slope d^2).
  vector term (determines sigma_det): (1 - w) N(0, 2 sigma_det^2) + w N(0, s^2)
      per axis; w and s describe the remaining different neurons only and
      are not model parameters.

Fit in stages (fit()): sigma_det from the vector term; the different-neuron
correlation and geometry from delta s = 0 alone (only different neurons);
movement, p_same and the same-neuron correlation from delta s = 1; then a
short joint refinement of all parameters but sigma_det.
"""

from dataclasses import dataclass

import numpy as np
from scipy import special, stats
from scipy.optimize import minimize

from .distance_model import Abar_squared_window, matern1_g, rice_kernel_stable

DEFAULT_SETTINGS = {
    "distance_bins": 50,
    "correlation_bins": 50,
    "c_min": 0.5,
    "vector_bins": 40,
    "vector_range": 5.0,
    "vector_max_distance": 4.0,
    # Imaged slab: +/- 5 µm around the focal plane (model units: µm with
    # a calibrated pixel size, px with pxtomu = 1). Fixed, not fitted.
    "slab_thickness": 10.0,
}

# name: (label, description, low, high, start)
PARAMETERS = {
    "p_same": ("π_same", "Prior: fraction of delta s = 1 pairs that belong to the same neuron (not the p(same) map over distance and correlation).", 1e-3, 0.6, 0.1),
    "h": ("h", "Mean hard-core distance between different neurons (px).", 2.0, 12.0, 6.0),
    "sigma_h": ("σ_h", "Spread of the hard-core distance (neuron sizes, px).", 0.0, 5.0, 1.5),
    "sigma_det": ("σ_det", "Detection noise: per-axis centroid uncertainty of one footprint (px).", 0.05, 3.0, 0.5),
    "sigma_move": ("σ_move", "Movement between consecutive sessions: per-axis scale (px).", 0.0, 4.0, 0.6),
    "nu_move": ("ν_move", "Tail of the movement distribution (Student-t degrees of freedom; large: Gaussian).", 1.0, 200.0, 5.0),
    "u_same_mean": ("μ_u same", "Same neuron: mean of 1 - correlation at zero displacement.", 0.005, 0.3, 0.05),
    "u_same_shape": ("k_u same", "Same neuron: Gamma shape of 1 - correlation.", 1.0, 50.0, 4.0),
    "u_same_slope": ("β_u same", "Same neuron: growth of the mean of 1 - correlation with squared displacement (1/px²).", 0.0, 1.0, 0.02),
    "u_diff_mean": ("μ_u diff", "Different neurons: mean of 1 - correlation.", 0.01, 0.4, 0.12),
    "u_diff_shape": ("k_u diff", "Different neurons: Gamma shape of 1 - correlation.", 1.0, 50.0, 6.0),
}

PARAMETER_NAMES = tuple(PARAMETERS)

_NODES, _WEIGHTS = np.polynomial.hermite_e.hermegauss(5)
_WEIGHTS = _WEIGHTS / _WEIGHTS.sum()


@dataclass
class Context:
    """Fixed inputs of the model."""

    lam: float
    dims: tuple
    R: float
    settings: dict

    @property
    def d_edges(self):
        return np.linspace(0, self.R, self.settings["distance_bins"] + 1)

    @property
    def c_edges(self):
        return np.linspace(self.settings["c_min"], 1, self.settings["correlation_bins"] + 1)

    @property
    def v_edges(self):
        r = self.settings["vector_range"]
        return np.linspace(-r, r, self.settings["vector_bins"] + 1)

    @property
    def r_fine(self):
        # 0.1 px: five points per distance bin.
        return np.linspace(0, self.R, int(round(self.R * 10)) + 1)

    @property
    def d_true(self):
        return np.linspace(0, self.R + 8, int(round((self.R + 8) * 10)) + 1)


def settings_with_defaults(settings=None):
    out = dict(DEFAULT_SETTINGS)
    out.update(settings or {})
    # Fixed: not taken from older model files.
    out["slab_thickness"] = DEFAULT_SETTINGS["slab_thickness"]
    return out


# ------------------------------------------------------------------ evidence
def evidence_shapes(settings):
    nd, nc, nv = (
        settings["distance_bins"],
        settings["correlation_bins"],
        settings["vector_bins"],
    )
    return {"same2": (nd, nc), "cross2": (nd, nc, 3), "vector": (nv, nv)}


def same_session_histogram(centroid_distance, correlation, included, R, settings):
    """delta s = 0: unordered pairs of different included footprints."""
    n = centroid_distance.shape[0]
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    mask = (
        upper
        & included[:, None]
        & included[None, :]
        & np.isfinite(correlation)
        & (centroid_distance < R)
    )
    d_edges = np.linspace(0, R, settings["distance_bins"] + 1)
    c_edges = np.linspace(settings["c_min"], 1, settings["correlation_bins"] + 1)
    return np.histogram2d(
        centroid_distance[mask], correlation[mask], bins=[d_edges, c_edges]
    )[0].astype(int)


def cross_session_histograms(
    centroid_distance,
    shift_distance,
    correlation,
    shifts,
    centroid_delta,
    nearest,
    included_ref,
    included_this,
    R,
    settings,
):
    """delta s = 1 (reference rows, session columns): pair histogram
    (total / NN / non-NN) and the vector histogram."""
    valid = (
        included_ref[:, None]
        & included_this[None, :]
        & np.isfinite(correlation)
    )
    distance = np.where(nearest, shift_distance, centroid_distance)
    valid &= distance < R

    d_edges = np.linspace(0, R, settings["distance_bins"] + 1)
    c_edges = np.linspace(settings["c_min"], 1, settings["correlation_bins"] + 1)

    cross = np.zeros(
        (settings["distance_bins"], settings["correlation_bins"], 3), dtype=int
    )
    for channel, mask in enumerate((valid, valid & nearest, valid & ~nearest)):
        cross[..., channel] = np.histogram2d(
            distance[mask], correlation[mask], bins=[d_edges, c_edges]
        )[0]

    near = valid & (shift_distance < settings["vector_max_distance"])
    residual = shifts[near] - centroid_delta[near]
    v_edges = np.linspace(
        -settings["vector_range"], settings["vector_range"], settings["vector_bins"] + 1
    )
    vector = np.histogram2d(residual[:, 0], residual[:, 1], bins=[v_edges, v_edges])[0]

    return cross, vector.astype(int)


def neuron_intensity(centroids_by_session, dims, sigma_px=40.0, threshold=0.1):
    """Neurons per px² inside the area covered by neurons (KDE of all
    session centroids; the imaging window may extend beyond the region)."""
    from scipy.ndimage import gaussian_filter

    counts = [len(c) for c in centroids_by_session if len(c)]
    if not counts:
        raise ValueError("No neurons to estimate the neuron density from.")

    centroids = np.concatenate([c for c in centroids_by_session if len(c)])
    step = 4
    occupancy = np.histogram2d(
        centroids[:, 0],
        centroids[:, 1],
        bins=(max(1, int(dims[0]) // step), max(1, int(dims[1]) // step)),
        range=[[0, dims[0]], [0, dims[1]]],
    )[0]
    density = gaussian_filter(occupancy, sigma_px / step)
    positive = density[density > 0]
    coverage = (
        float(np.mean(density > threshold * np.percentile(positive, 90)))
        if positive.size
        else 1.0
    )
    area = float(np.prod(dims)) * max(coverage, 1e-3)
    return float(np.mean(counts)) / area, coverage


# --------------------------------------------------------------------- model
def _bin_mass(pdf, grid, edges):
    cdf = np.concatenate([[0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    return np.clip(np.diff(np.interp(edges, grid, cdf)), 0, None)


def _normalized(mass):
    total = mass.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Degenerate model distribution.")
    return mass / total


def slab_pair_correlation(d, lam, h, T):
    """Pair correlation of a Matérn-I hard core h, projected from a slab
    of thickness T."""
    g = matern1_g(d, lam, h)
    below = d < h
    if T > 0:
        need = np.sqrt(np.clip(h**2 - d[below] ** 2, 0, None))
        g[below] = np.where(need < T, (1 - need / T) ** 2, 0.0)
    return g


def pair_correlation(d, lam, h, sigma_h, T):
    """slab_pair_correlation averaged over h ~ Normal(h, sigma_h)."""
    if sigma_h <= 1e-3:
        return slab_pair_correlation(d, lam, h, T)
    g = np.zeros_like(d)
    for node, weight in zip(_NODES, _WEIGHTS):
        g += weight * slab_pair_correlation(d, lam, max(0.5, h + sigma_h * node), T)
    return g


def _blur(f_true, d_true, r_fine, sigma):
    kernel = rice_kernel_stable(r_fine, d_true, sigma)
    return np.trapezoid(kernel * f_true[None, :], d_true, axis=1)


def movement_density(r, scale, nu):
    """Radial density of a 2D Student-t displacement."""
    if not np.isfinite(nu) or nu > 1e3:
        return stats.rayleigh.pdf(r, scale=scale)
    return r / scale**2 * (1 + r**2 / (nu * scale**2)) ** (-(nu / 2 + 1))


def distance_densities(theta, ctx, parts=None):
    """Observed distance densities on ctx.r_fine (normalized on [0, R]):
    "diff0", "diff1", "same" (or only ``parts``)."""
    d_true, r_fine = ctx.d_true, ctx.r_fine
    window = Abar_squared_window(d_true, L=ctx.dims) / np.prod(ctx.dims)
    poisson = d_true * window

    g = pair_correlation(
        d_true, ctx.lam, theta["h"], theta["sigma_h"], ctx.settings["slab_thickness"]
    )
    f_diff = poisson * g

    s0 = np.sqrt(2) * theta["sigma_det"]
    s1 = np.sqrt(2 * theta["sigma_det"] ** 2 + theta["sigma_move"] ** 2)

    def unit(values):
        area = np.trapezoid(values, r_fine)
        if not np.isfinite(area) or area <= 0:
            raise ValueError("Degenerate model distribution.")
        return values / area

    out = {}
    if parts is None or "diff0" in parts:
        out["diff0"] = unit(_blur(f_diff, d_true, r_fine, s0))
    if parts is None or "diff1" in parts:
        out["diff1"] = unit(_blur(f_diff, d_true, r_fine, s1))
    if parts is None or "same" in parts:
        out["same"] = unit(movement_density(r_fine, s1, theta["nu_move"]))
    return out


def gamma_pdf(u, mean, shape):
    """Gamma density on u (mean, shape); broadcasts over means."""
    scale = np.asarray(mean, dtype=float) / shape
    x = np.asarray(u, dtype=float) / scale
    with np.errstate(divide="ignore", invalid="ignore"):
        log = (shape - 1) * np.log(x) - x - special.gammaln(shape) - np.log(scale)
    return np.where(x > 0, np.exp(log), 0.0)


def correlation_masses(mean, shape, c_edges):
    """Gamma on u = 1 - c over the correlation bins (increasing c).
    ``mean`` may be an array (one row of masses per mean)."""
    u_edges = 1 - c_edges[::-1]
    mean = np.asarray(mean, dtype=float)
    cdf = special.gammainc(shape, u_edges / (mean[..., None] / shape))
    mass = np.diff(cdf, axis=-1)[..., ::-1]
    total = mass.sum(axis=-1, keepdims=True)
    if np.any(~np.isfinite(total)) or np.any(total <= 0):
        raise ValueError("Degenerate model distribution.")
    return mass / total


def same_mean(theta, d):
    return theta["u_same_mean"] * (1 + theta["u_same_slope"] * np.asarray(d) ** 2)


def _distance_masses(theta, ctx, parts):
    densities = distance_densities(theta, ctx, parts)
    return {
        k: _normalized(_bin_mass(v, ctx.r_fine, ctx.d_edges)) for k, v in densities.items()
    }


def predict_same_session(theta, ctx):
    """delta s = 0 histogram: different neurons only."""
    D = _distance_masses(theta, ctx, ("diff0",))
    c_diff = correlation_masses(theta["u_diff_mean"], theta["u_diff_shape"], ctx.c_edges)
    return np.outer(D["diff0"], c_diff)


def predict_cross_session(theta, ctx):
    """delta s = 1 histogram: same and different neurons."""
    D = _distance_masses(theta, ctx, ("diff1", "same"))
    c_edges = ctx.c_edges
    c_diff = correlation_masses(theta["u_diff_mean"], theta["u_diff_shape"], c_edges)
    d_centres = 0.5 * (ctx.d_edges[1:] + ctx.d_edges[:-1])
    c_same = correlation_masses(same_mean(theta, d_centres), theta["u_same_shape"], c_edges)
    p = theta["p_same"]
    return p * D["same"][:, None] * c_same + (1 - p) * np.outer(D["diff1"], c_diff)


def predict(theta, ctx):
    """Bin probabilities of the delta s = 0 and delta s = 1 histograms."""
    return predict_same_session(theta, ctx), predict_cross_session(theta, ctx)


def _multinomial_nll(H, P):
    if H is None or H.sum() <= 0:
        return 0.0
    P = np.clip(P / P.sum(), 1e-15, None)
    return float(-np.sum(H * np.log(P)))


def _minimize(objective, names, start, theta_fixed, maxiter=500):
    """L-BFGS-B over ``names`` (bounds from PARAMETERS)."""
    x0 = [float(np.clip(start[n], PARAMETERS[n][2], PARAMETERS[n][3])) for n in names]
    bounds = [(PARAMETERS[n][2], PARAMETERS[n][3]) for n in names]

    def wrapped(x):
        theta = dict(theta_fixed)
        theta.update(zip(names, x))
        try:
            value = objective(theta)
        except (ValueError, FloatingPointError, ZeroDivisionError):
            return 1e30
        return value if np.isfinite(value) else 1e30

    result = minimize(wrapped, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter})
    theta = dict(theta_fixed)
    theta.update(zip(names, map(float, result.x)))
    return theta, result


def fit_detection_noise(vector, ctx, start=0.5):
    """sigma_det from the shift - centroid vector histogram: a narrow
    Gaussian (2 sigma_det^2 per axis) plus a broad one for the remaining
    different neurons (nuisance)."""
    if vector is None or vector.sum() <= 0:
        return float(start)

    v_edges = ctx.v_edges

    def masses(scale):
        return _normalized(np.diff(stats.norm.cdf(v_edges, scale=scale)))

    def objective(x):
        sigma, weight, broad = x
        narrow, wide = masses(np.sqrt(2) * sigma), masses(broad)
        P = (1 - weight) * np.outer(narrow, narrow) + weight * np.outer(wide, wide)
        return _multinomial_nll(vector, P)

    result = minimize(
        objective,
        [start, 0.2, 2.5],
        method="L-BFGS-B",
        bounds=[PARAMETERS["sigma_det"][2:4], (0.0, 0.95), (0.5, 8.0)],
    )
    return float(result.x[0])


def fit(evidence, ctx, start=None, fixed=None, polish=False):
    """Staged fit (see module docstring).

    evidence: {"same2": (nd, nc), "cross2": (nd, nc, 3), "vector": (nv, nv)}.
    ``fixed`` parameters are kept. ``polish``: a final joint refinement
    (845ad: likelihood better by 1e-4 relative, 4x the time). Returns
    (parameters, result of the last stage).
    """
    H0 = evidence.get("same2")
    H1 = evidence["cross2"][..., 0]

    theta = {n: v[4] for n, v in PARAMETERS.items()}
    theta.update(start or {})
    fixed = dict(fixed or {})
    theta.update(fixed)

    def free(names):
        return [n for n in names if n not in fixed]

    # 1. Detection noise from the vector term.
    if "sigma_det" not in fixed:
        theta["sigma_det"] = fit_detection_noise(evidence.get("vector"), ctx, theta["sigma_det"])

    result = None
    if H0 is not None and H0.sum() > 0:
        # 2. Different-neuron correlation: delta s = 0 correlation marginal.
        marginal = H0.sum(axis=0)
        names = free(("u_diff_mean", "u_diff_shape"))
        if names:
            theta, result = _minimize(
                lambda t: _multinomial_nll(
                    marginal, correlation_masses(t["u_diff_mean"], t["u_diff_shape"], ctx.c_edges)
                ),
                names, theta, theta,
            )

        # 3. Geometry: delta s = 0 distance marginal.
        marginal = H0.sum(axis=1)
        names = free(("h", "sigma_h"))
        if names:
            theta, result = _minimize(
                lambda t: _multinomial_nll(marginal, _distance_masses(t, ctx, ("diff0",))["diff0"]),
                names, theta, theta,
            )

    # 4. Movement, matches and same-neuron correlation: delta s = 1.
    names = free(("p_same", "sigma_move", "nu_move", "u_same_mean", "u_same_shape", "u_same_slope"))
    if names:
        theta, result = _minimize(
            lambda t: _multinomial_nll(H1, predict_cross_session(t, ctx)), names, theta, theta
        )

    # 5. Joint refinement (sigma_det stays from the vector term).
    names = free(n for n in PARAMETER_NAMES if n != "sigma_det")
    if polish and names:
        theta, result = _minimize(
            lambda t: _multinomial_nll(H0, predict_same_session(t, ctx))
            + _multinomial_nll(H1, predict_cross_session(t, ctx)),
            names, theta, theta, maxiter=30,
        )

    return {n: float(theta[n]) for n in PARAMETER_NAMES}, result


# ---------------------------------------------------------- posterior / display
def _cumulative(density, grid):
    cdf = np.concatenate([[0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))])
    return cdf / cdf[-1]


def posterior(theta, ctx, d_centres, c_centres, use_cdf=True):
    """p(same | distance, correlation) for delta s = 1 pairs.

    use_cdf (default, as the classic model): tail probabilities, i.e.
    P(D >= d) P(C <= c) for same neurons against P(D <= d) P(C >= c) for
    different neurons; the closer and the better matching a pair, the
    clearer it is the same neuron. The same-neuron correlation tail is
    averaged over the same-neuron distances: conditioned on d, its growing
    mean would make the tail approach 1 at large distances and break the
    monotonicity in distance. Otherwise densities.
    """
    densities = distance_densities(theta, ctx, ("diff1", "same"))
    r = ctx.r_fine
    d_centres = np.asarray(d_centres, dtype=float)
    u = np.clip(1 - np.asarray(c_centres, dtype=float), 1e-9, None)

    same_means = same_mean(theta, d_centres)[:, None]
    diff_scale = theta["u_diff_mean"] / theta["u_diff_shape"]
    same_shape = theta["u_same_shape"]

    if use_cdf:
        f_same = 1 - np.interp(d_centres, r, _cumulative(densities["same"], r))
        f_diff = np.interp(d_centres, r, _cumulative(densities["diff1"], r))
        # C <= c is U >= u; C >= c is U <= u.
        weights = densities["same"] / densities["same"].sum()
        pick = np.flatnonzero(weights > 1e-6 * weights.max())
        scales = same_mean(theta, r[pick])[:, None] / same_shape
        g_same = (
            weights[pick, None] * special.gammaincc(same_shape, u[None, :] / scales)
        ).sum(axis=0)[None, :] / weights[pick].sum()
        g_diff = special.gammainc(theta["u_diff_shape"], u / diff_scale)
    else:
        f_same = np.interp(d_centres, r, densities["same"])
        f_diff = np.interp(d_centres, r, densities["diff1"])
        g_same = gamma_pdf(u[None, :], same_means, same_shape)
        g_diff = gamma_pdf(u, theta["u_diff_mean"], theta["u_diff_shape"])

    same = theta["p_same"] * f_same[:, None] * g_same
    different = (1 - theta["p_same"]) * f_diff[:, None] * g_diff[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        p = same / (same + different)
    return np.nan_to_num(p, nan=0.0)


def display_densities(theta, ctx):
    """1D component densities for plotting (distance on ctx.r_fine
    resampled to the distance bins' grid, correlation on a c grid)."""
    densities = distance_densities(theta, ctx)
    r = ctx.r_fine
    c_grid = np.linspace(ctx.settings["c_min"], 1, 501)
    u = np.clip(1 - c_grid, 1e-9, None)

    # Correlation marginal of same neurons: average over their distances.
    weights = densities["same"] / densities["same"].sum()
    means = same_mean(theta, r)
    pick = np.linspace(0, len(r) - 1, 60).astype(int)
    g_same = (
        weights[pick, None] * gamma_pdf(u[None, :], means[pick, None], theta["u_same_shape"])
    ).sum(axis=0) / weights[pick].sum()

    return {
        "grid": {"distance": r, "correlation": c_grid},
        "pdf": {
            "distance_same": densities["same"],
            "distance_diff": densities["diff1"],
            "distance_diff_same_session": densities["diff0"],
            "correlation_same": g_same,
            "correlation_diff": gamma_pdf(u, theta["u_diff_mean"], theta["u_diff_shape"]),
        },
    }
