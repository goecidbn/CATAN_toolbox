import numpy as np
from scipy.stats import norm, lognorm
from scipy.stats import skewnorm

# ----------------------------
# Correlation PDFs on [0,1]
# ----------------------------


def pdf_truncated_normal(
    x: np.ndarray,
    mean: float = 0.5,
    sd: float = 0.12,
    lower: float = 0.0,
    upper: float = 1.0,
):
    # def pdf_diff_correlation(c, mean=0.5, sd=0.12, lower=0.0, upper=1.0):
    """
    truncated normal on [0,1] with given mean and sd
    """
    phi_a, phi_b = (lower - mean) / sd, (upper - mean) / sd
    Z = norm.cdf(phi_b) - norm.cdf(phi_a)
    f = np.zeros_like(x)
    m = (x >= lower) & (x <= upper)
    f[m] = norm.pdf((x[m] - mean) / sd) / (sd * Z)
    return f / np.trapezoid(f, x)


def lognormal_from_mean_variance(mean, var):
    """
    Convert mean/variance of a lognormal distribution into the
    parameters expected by scipy.stats.lognorm.

    Parameters
    ----------
    mean : float
        Mean of the lognormal variable.
    var : float
        Variance of the lognormal variable.

    Returns
    -------
    mu, sigma
        Such that
            log(X) ~ Normal(mu, sigma^2)

    scipy:
        s = sigma
        scale = exp(mu)
    """
    sigma2 = np.log1p(var / mean**2)
    s = np.sqrt(sigma2)
    mu = np.log(mean) - sigma2 / 2
    scale = np.exp(mu)

    return s, scale


def pdf_reverse_lognormal(
    x: np.ndarray,
    mean: float = 0.8,
    sigma: float = 0.1,
    lower: float = 0.0,
    upper: float = 1.0,
    eps: float = 1e-12,
):
    # def pdf_same_correlation(c: np.ndarray, mu_ln: float=-2.5, sigma_ln: float=0.6, lower: float=0.0, upper: float=1.0, eps: float=1e-12):
    """
    y = 1-c ~ LogNormal(mu_ln, sigma_ln), restricted to y in (0,1] (i.e. c in [0,1)).
    Renormalized to be a valid pdf on [0,1).
    """
    s, scale = lognormal_from_mean_variance(1 - mean, sigma)
    # x = np.asarray(x, dtype=float)
    f = np.zeros_like(x)
    m = (x >= lower) & (x < upper)
    y = np.clip(upper - x[m], eps, upper)
    base = lognorm.pdf(y, s=s, scale=scale)
    Z = lognorm.cdf(upper, s=s, scale=scale)  # P(y <= 1)
    f[m] = base / Z
    # return f
    return f / np.trapezoid(f, x)
