#!/usr/bin/env python
"""Stage 1 — compute porosity field and fit warped-Matérn Gaussian process.

Implements §3.3 (Field Learning) of main.tex:
    1. Valid-convolution porosity field via FFT (Eq. 1).
    2. Logit-warped two-point radial covariance.
    3. Fit stationary Matérn kernel (ν fixed or free) with fixed empirical σ².

Configuration resolution order (highest precedence first):
    CLI > YAML config > environment defaults in pipelines/_common.py

Legacy counterpart: scripts/legacy/0002-porosity-field-estimator.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import scipy.signal
import torch

# Make the repo-level package importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from diffsci2.extra import two_point_correlation, matern_gaussian_process  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

from pipelines._common import (  # noqa: E402
    load_config, raw_volume_path, field_dir,
)


# ---------------------------------------------------------------------------
# Porosity field (FFT convolution)
# ---------------------------------------------------------------------------

def calculate_valid_porosity_field(data: np.ndarray, kernel_size: int) -> np.ndarray:
    """Eq. (1) of main.tex, valid-crop version used by Alg. 2 pairing."""
    kernel = np.ones((kernel_size, kernel_size, kernel_size)) / (kernel_size ** 3)
    pore = 1.0 - data.astype(np.float32)
    return scipy.signal.fftconvolve(pore, kernel, mode="valid")


def avg_pool_numpy(volume: np.ndarray, factor: int) -> np.ndarray:
    """Latent-resolution average pooling, used to match the VAE compression."""
    t = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
    t = torch.nn.functional.avg_pool3d(t, kernel_size=factor, stride=factor)
    return t.squeeze(0).squeeze(0).numpy()


# ---------------------------------------------------------------------------
# Logit-warped two-point correlation
# ---------------------------------------------------------------------------

def logit_tpc(field: np.ndarray, device: str, bins: int, voxel_size: float) -> dict:
    t = torch.tensor(field).float()
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


# ---------------------------------------------------------------------------
# Matérn fit
# ---------------------------------------------------------------------------

def fit_matern(
    r: np.ndarray,
    correlation: np.ndarray,
    sigma_sq: float,
    fixed_nu: float | None = None,
) -> dict:
    """Fit Matérn kernel with empirical σ² fixed.

    If `fixed_nu` is given, only length_scale is free; otherwise nu and
    length_scale are both free.
    """
    valid = np.isfinite(correlation) & (r > 1e-10)
    r_c = r[valid]
    c_c = correlation[valid]
    drop = np.abs(c_c - sigma_sq * 0.36).argmin()
    p0_l = r_c[drop] if drop < len(r_c) else r_c[-1] / 2
    if p0_l == 0:
        p0_l = 1.0

    try:
        if fixed_nu is not None:
            def cov(r, length_scale):
                return matern_gaussian_process.matern_covariance(
                    r, sigma_sq, fixed_nu, length_scale,
                )
            popt, pcov = curve_fit(
                cov, r_c, c_c, p0=[p0_l],
                bounds=([1e-6], [np.inf]), maxfev=5000,
            )
            return {
                "sigma_sq": sigma_sq, "nu": fixed_nu,
                "length_scale": popt[0], "fit_success": True,
            }
        else:
            def cov(r, nu, length_scale):
                return matern_gaussian_process.matern_covariance(
                    r, sigma_sq, nu, length_scale,
                )
            popt, pcov = curve_fit(
                cov, r_c, c_c, p0=[1.5, p0_l],
                bounds=([0.1, 1e-6], [30.0, np.inf]), maxfev=5000,
            )
            return {
                "sigma_sq": sigma_sq, "nu": popt[0],
                "length_scale": popt[1], "fit_success": True,
            }
    except (RuntimeError, ValueError) as e:
        print(f"  Matérn fit failed: {e}")
        return {
            "sigma_sq": sigma_sq, "nu": np.nan,
            "length_scale": np.nan, "fit_success": False,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip())
    p.add_argument("--config", required=True, help="Rock YAML config.")
    p.add_argument("--variant", choices=["fc257", "fc129"], default="fc129",
                   help="Which averaging window to fit. fc257 -> r=128, fc129 -> r=64.")
    p.add_argument("--latent-downsample", type=int, default=8,
                   help="Downsample factor before TPC (0 or 1 disables). "
                        "Matches the VAE spatial compression (F=8).")
    p.add_argument("--tpc-bins", type=int, default=50)
    p.add_argument("--fixed-nu", type=float, default=1.5,
                   help="If set, fix Matérn ν to this value (paper uses 3/2). "
                        "Pass 0 or negative to fit ν freely.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--save-field", action="store_true",
                   help="Save the full valid porosity field as .npy (large).")
    p.add_argument("--output-dir", default=None,
                   help="Override output directory. Default: gpdata4-<kernel>/<rock>/")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    rock = cfg["rock"]
    radius = cfg["field"][args.variant]["radius"]
    kernel_size = 2 * radius + 1
    voxel_size_um = cfg["data"]["voxel_size_um"]

    volume_path = raw_volume_path(cfg["data"]["volume_filename"])
    out_dir = Path(args.output_dir) if args.output_dir else field_dir(radius, rock)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Rock:          {rock}")
    print(f"Variant:       {args.variant} (r={radius}, kernel={kernel_size})")
    print(f"Volume:        {volume_path}")
    print(f"Output dir:    {out_dir}")
    print()

    print("Loading volume...")
    data = np.fromfile(volume_path, dtype=np.uint8).reshape(cfg["data"]["volume_shape"])
    print(f"  shape={data.shape}")

    print(f"\nComputing porosity field (valid FFT convolution, kernel={kernel_size})...")
    field = calculate_valid_porosity_field(data, kernel_size)
    print(f"  shape={field.shape}, range=[{field.min():.4f}, {field.max():.4f}], mean={field.mean():.4f}")

    if args.save_field:
        field_path = out_dir / f"{rock.lower()}_porosity_field.npy"
        print(f"  saving -> {field_path}")
        np.save(field_path, field)

    # Work in latent resolution for the TPC + Matérn fit, to match the
    # resolution at which the diffusion model sees the conditioning.
    if args.latent_downsample and args.latent_downsample > 1:
        print(f"\nAverage-pooling by {args.latent_downsample} before TPC...")
        field_latent = avg_pool_numpy(field, args.latent_downsample)
        print(f"  shape={field_latent.shape}")
        voxel_size_tpc = 1.0  # latent units
    else:
        field_latent = field
        voxel_size_tpc = voxel_size_um

    print(f"\nComputing logit TPC on {args.device}...")
    corr = logit_tpc(field_latent, args.device, args.tpc_bins, voxel_size_tpc)
    print(f"  mean_logit={corr['mean_logit']:.4f}, variance_logit={corr['variance_logit']:.4f}")

    fixed_nu = args.fixed_nu if args.fixed_nu and args.fixed_nu > 0 else None
    print(f"\nFitting Matérn (fixed σ²; ν={'fitted' if fixed_nu is None else fixed_nu})...")
    fit = fit_matern(corr["r"], corr["correlation"], corr["variance_logit"], fixed_nu=fixed_nu)
    if fit["fit_success"]:
        print(f"  sigma²={fit['sigma_sq']:.4f}, nu={fit['nu']:.4f}, length_scale={fit['length_scale']:.4f}")
    else:
        print("  fit did not converge — check the output .npz manually.")

    analysis_path = out_dir / f"{rock.lower()}_porosity_analysis.npz"
    print(f"\nSaving analysis -> {analysis_path}")
    np.savez(
        analysis_path,
        r=corr["r"], correlation=corr["correlation"],
        mean_logit=corr["mean_logit"], variance_logit=corr["variance_logit"],
        matern_sigma_sq=fit["sigma_sq"], matern_nu=fit["nu"],
        matern_length_scale=fit["length_scale"],
        matern_fit_success=fit["fit_success"],
        kernel_size=kernel_size,
        voxel_size=voxel_size_tpc,
        original_voxel_size=voxel_size_um,
        latent_downsample=args.latent_downsample,
        rock=rock,
        method="logit",
    )
    print("Done.")


if __name__ == "__main__":
    main()
