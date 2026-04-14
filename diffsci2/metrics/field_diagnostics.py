"""Field-conditioning diagnostics for §4.3 of main.tex.

- field_pearson    : voxel-wise Pearson correlation between the input
                     conditioning field φ and the porosity field φ̂
                     recomputed from the generated binary volume.
- logit_field_tpc  : radial two-point correlation of the logit-warped
                     field, used for the FC-129 / FC-257 diagnostics
                     that compare true vs input vs generated covariance.
"""
from __future__ import annotations

import numpy as np
import torch

from diffsci2.extra import two_point_correlation


def field_pearson(phi_input: np.ndarray, phi_generated: np.ndarray) -> float:
    """Pearson correlation coefficient between two aligned porosity fields."""
    a = np.asarray(phi_input, dtype=np.float64).ravel()
    b = np.asarray(phi_generated, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(
            f"Field shapes must match: input {a.shape} vs generated {b.shape}"
        )
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)


def logit_field_tpc(
    phi: np.ndarray,
    bins: int = 50,
    voxel_size: float = 1.0,
    device: str = "cpu",
) -> dict:
    """Radial two-point correlation of the logit-warped field.

    Returns a dict with keys 'r', 'correlation', 'mean_logit',
    'variance_logit' — the same structure produced by the pipeline's
    stage 1 (01_fit_gaussian_process.py). Use it to compare the three
    curves required for main.tex Fig. 7 (true, GP input, generated).
    """
    t = torch.tensor(phi, dtype=torch.float32)
    warped = torch.logit(t)
    mean = warped.mean().item()
    centered = warped - mean
    variance = centered.var().item()
    tpc = two_point_correlation.tpcf_fft(
        centered.to(device), bins=bins, voxel_size=voxel_size,
    )
    return {
        "r": tpc.r,
        "correlation": tpc.correlation,
        "mean_logit": mean,
        "variance_logit": variance,
    }
