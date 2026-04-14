"""Integrated absolute error for relative permeability and capillary pressure.

§4.2 of main.tex:

    IAE_{kr} = 1/2 Σ_{p ∈ {w, nw}} ∫₀¹ |kᵣ^gen,p(Sᵥ) − kᵣ^ref,p(Sᵥ)| dSᵥ
    IAE_{Pc} = ∫₀¹ |log₁₀ Pc^gen(Sᵥ) − log₁₀ Pc^ref(Sᵥ)| dSᵥ

Curves coming from pore-network simulations are not sampled at the same
Sᵥ points between generated and reference volumes, so we linearly
interpolate onto a common [0, 1] grid before integrating.
"""
from __future__ import annotations

import numpy as np

# NumPy 2.x renamed trapz to trapezoid.
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def _interp_to_common(
    s_src: np.ndarray, y_src: np.ndarray, s_target: np.ndarray,
) -> np.ndarray:
    """Monotone sort + linear interpolate onto s_target.

    Clamps ends: values outside [s_src.min(), s_src.max()] are held
    constant at the nearest endpoint.
    """
    order = np.argsort(s_src)
    s_sorted = np.asarray(s_src, dtype=np.float64)[order]
    y_sorted = np.asarray(y_src, dtype=np.float64)[order]
    # np.interp already clamps at the boundaries.
    return np.interp(s_target, s_sorted, y_sorted)


def iae_kr_curves(
    s_gen, kr_w_gen, kr_nw_gen,
    s_ref, kr_w_ref, kr_nw_ref,
    n_points: int = 201,
) -> float:
    """IAE between generated and reference relative-permeability curves.

    Each curve pair (generated, reference) is interpolated onto the same
    n_points grid over [0, 1], then we take the mean of the two wetting/
    non-wetting integrated absolute errors (per the 1/2 factor in main.tex).
    """
    s_target = np.linspace(0.0, 1.0, n_points)
    kw_g = _interp_to_common(np.asarray(s_gen), np.asarray(kr_w_gen), s_target)
    kn_g = _interp_to_common(np.asarray(s_gen), np.asarray(kr_nw_gen), s_target)
    kw_r = _interp_to_common(np.asarray(s_ref), np.asarray(kr_w_ref), s_target)
    kn_r = _interp_to_common(np.asarray(s_ref), np.asarray(kr_nw_ref), s_target)
    iae_w = _trapz(np.abs(kw_g - kw_r), s_target)
    iae_n = _trapz(np.abs(kn_g - kn_r), s_target)
    return float(0.5 * (iae_w + iae_n))


def iae_pc_curve(
    s_gen, pc_gen,
    s_ref, pc_ref,
    n_points: int = 201,
    eps: float = 1e-30,
) -> float:
    """IAE between log10 capillary-pressure curves.

    Pc values must be positive; eps guards against log10(0) on curves
    that extend below the displacement threshold.
    """
    s_target = np.linspace(0.0, 1.0, n_points)
    pc_g = np.maximum(eps, _interp_to_common(
        np.asarray(s_gen), np.asarray(pc_gen), s_target,
    ))
    pc_r = np.maximum(eps, _interp_to_common(
        np.asarray(s_ref), np.asarray(pc_ref), s_target,
    ))
    return float(_trapz(np.abs(np.log10(pc_g) - np.log10(pc_r)), s_target))
