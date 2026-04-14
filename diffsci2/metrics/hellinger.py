"""Gaussian-Hellinger distance between two 1D empirical distributions.

Equation (9) of main.tex. Used to quantify the overlap between a
generated volume's sub-block distribution (porosity or permeability)
and the reference rock's sub-block distribution.
"""
from __future__ import annotations

import numpy as np


def hellinger_gaussian(p: np.ndarray, q: np.ndarray) -> float:
    """Closed-form Hellinger distance between Gaussians fit to two samples.

    Given two sets of scalar samples, fit a 1D Gaussian to each (mean and
    standard deviation from the sample) and return the Hellinger distance
    between the two fitted Gaussians:

        H_G² = 1 − √(2 σ_p σ_q / (σ_p² + σ_q²))
                 · exp( −(μ_p − μ_q)² / (4 (σ_p² + σ_q²)) )

    Returns a scalar in [0, 1].

    Parameters
    ----------
    p, q
        1D arrays of samples. NaNs are dropped.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p[np.isfinite(p)]
    q = q[np.isfinite(q)]
    if p.size < 2 or q.size < 2:
        raise ValueError("Need at least 2 finite samples in each of p and q.")

    mu_p, mu_q = p.mean(), q.mean()
    # Sample std with ddof=0 so a single-value degenerate group gives sigma=0
    # rather than NaN — we guard against sigma=0 below.
    sigma_p, sigma_q = p.std(ddof=0), q.std(ddof=0)

    # Guard against degenerate zero-variance groups. If both are zero, the
    # distance is 0 if means coincide and 1 otherwise.
    if sigma_p == 0 and sigma_q == 0:
        return 0.0 if mu_p == mu_q else 1.0
    # If only one is zero, the overlap is zero.
    if sigma_p == 0 or sigma_q == 0:
        return 1.0

    num = 2.0 * sigma_p * sigma_q
    denom = sigma_p ** 2 + sigma_q ** 2
    exponent = -((mu_p - mu_q) ** 2) / (4.0 * denom)
    h_sq = 1.0 - np.sqrt(num / denom) * np.exp(exponent)
    # Numerical clamp: h_sq should live in [0, 1] but floating point can
    # produce tiny negatives near 0.
    h_sq = max(0.0, min(1.0, float(h_sq)))
    return float(np.sqrt(h_sq))
