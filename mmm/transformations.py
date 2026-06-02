"""
mmm/transformations.py
-----------------------
Adstock and saturation transformations for the MMM pipeline.

These are implemented as plain NumPy functions so they can be used both:
  - Inside PyMC models (using PyTensor ops — pass PyTensor tensors directly)
  - Outside models for visualisation / sanity checks (pass numpy arrays)

References:
  - Adstock: Broadbent (1979) — geometric decay
  - Saturation: Hill (1910) — sigmoidal response curve
"""

import numpy as np


# ---------------------------------------------------------------------------
# Adstock
# ---------------------------------------------------------------------------

def geometric_adstock(x: np.ndarray, lam: float) -> np.ndarray:
    """Apply geometric (Koyck) adstock decay to a spend series.

    The carry-over effect of advertising decays geometrically:

        x*_t = x_t + λ · x*_(t-1)

    Parameters
    ----------
    x : array-like, shape (T,)
        Raw spend values over time.
    lam : float in [0, 1]
        Decay rate.  lam=0 means no carry-over (pure direct effect).
        lam=0.9 means strong carry-over (slow decay).

    Returns
    -------
    np.ndarray, shape (T,)
        Adstocked spend series.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = x[t] + lam * out[t - 1]
    return out


def delayed_adstock(
    x: np.ndarray,
    lam: float,
    theta: int = 0,
) -> np.ndarray:
    """Geometric adstock with a peak-delay parameter (theta).

    Spend at time t first peaks at t+theta before decaying geometrically.
    theta=0 reduces to standard geometric_adstock.

    Parameters
    ----------
    x : array-like, shape (T,)
        Raw spend values.
    lam : float in [0, 1]
        Decay rate after the peak.
    theta : int >= 0
        Number of days until the advertising effect peaks.
    """
    x = np.asarray(x, dtype=float)
    T = len(x)
    weights = np.array([lam ** abs(t - theta) for t in range(T)])
    weights /= weights.sum()
    # Convolve — only use causal (past) lags to avoid data leakage
    out = np.convolve(x, weights[::-1], mode="full")[:T]
    return out


# ---------------------------------------------------------------------------
# Saturation
# ---------------------------------------------------------------------------

def hill_saturation(
    x: np.ndarray,
    k: float,
    K: float,
) -> np.ndarray:
    """Hill (sigmoidal) saturation function.

    Models diminishing returns: each additional unit of spend produces less
    incremental effect.

        f(x) = x^k / (x^k + K^k)

    Parameters
    ----------
    x : array-like, shape (T,)
        Input values (typically adstocked spend).
    k : float > 0
        Shape parameter — controls the steepness of the curve.
        k < 1: concave (strong diminishing returns from the start).
        k > 1: S-shaped curve (slow start, then fast growth, then plateau).
    K : float > 0
        Half-saturation point — the value of x at which f(x) = 0.5.

    Returns
    -------
    np.ndarray, shape (T,) — values in [0, 1]
    """
    x = np.asarray(x, dtype=float)
    x_k = np.power(np.maximum(x, 0.0), k)
    K_k = np.power(K, k)
    return x_k / (x_k + K_k)


def logistic_saturation(x: np.ndarray, lam: float) -> np.ndarray:
    """Simpler logistic (S-curve) saturation — alternative to Hill.

        f(x) = (1 - exp(-λ·x)) / (1 + exp(-λ·x))  ≈  tanh(λ·x / 2)

    Parameters
    ----------
    x : array-like
    lam : float > 0
        Controls how quickly the curve reaches saturation.
    """
    x = np.asarray(x, dtype=float)
    return (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x))


# ---------------------------------------------------------------------------
# Combined transform (adstock → saturation)
# ---------------------------------------------------------------------------

def adstock_saturation(
    x: np.ndarray,
    lam: float,
    k: float,
    K: float,
) -> np.ndarray:
    """Apply geometric adstock then Hill saturation in one call.

    This is the full media transformation used inside the MMM model:

        f(x) = hill_saturation(geometric_adstock(x, lam), k, K)

    Used primarily for visualising response curves given fixed parameters.
    """
    x_adstocked = geometric_adstock(x, lam)
    return hill_saturation(x_adstocked, k, K)
