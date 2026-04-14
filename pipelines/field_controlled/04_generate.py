#!/usr/bin/env python
"""Stage 4 — generate volumes (single GPU).

Implements §3.4 Algorithm 3 of main.tex:
    sample GP field → integrate probability-flow ODE → chunk-decode →
    binarize → border-crop.

Supports three variants (selected with `--variant`):
    fc257      : field-controlled with the published fc257 checkpoint
    fc129      : field-controlled with the published fc129 checkpoint
    uncond     : null conditioning with the base scalar-trained checkpoint
                 (the implicit unconditional model via condition dropout)

The *alternative* cases from legacy 0004c (cross-conditioning with
mismatched checkpoints, scalar-porosity sampling from the empirical
distribution — the "Controlled" baseline, and so on) live in
pipelines/experiments/. Those are useful for ablations but not part of
the canonical generation step.

Legacy counterpart: scripts/legacy/0004c-porosity-field-generator.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import diffsci2.models  # noqa: E402
import diffsci2.nets  # noqa: E402
from diffsci2.extra import chunk_decode_2  # noqa: E402
from diffsci2.extra.matern_gaussian_process import (  # noqa: E402
    MaternFieldSampler, PeriodicMaternFieldSampler,
)

from pipelines._common import (  # noqa: E402
    load_config, field_dir, published_checkpoints,
)


LATENT_TO_PIXEL = 8   # F = 8 VAE compression factor (main.tex §3.2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_checkpoint(rock: str, variant: str) -> Path:
    paths = published_checkpoints(rock)
    return {
        "fc257": paths.fc257,
        "fc129": paths.fc129,
        "uncond": paths.base,   # uncond uses the base model with null cond
    }[variant]


def gp_analysis_path(rock: str, variant: str, cfg: dict) -> Path:
    radius = cfg["field"][variant]["radius"] if variant != "uncond" else cfg["field"]["fc129"]["radius"]
    return field_dir(radius, rock) / f"{rock.lower()}_porosity_analysis.npz"


def load_gp_sampler(npz_path: Path, periodic: bool):
    analysis = np.load(npz_path)
    sigma_sq = float(analysis["matern_sigma_sq"])
    nu = float(analysis["matern_nu"])
    length_scale = float(analysis["matern_length_scale"])
    mean_logit = float(analysis["mean_logit"])
    cls = PeriodicMaternFieldSampler if periodic else MaternFieldSampler
    return cls(
        mean=mean_logit,
        sigma_sq=sigma_sq,
        nu=nu,
        length_scale=length_scale,
    )


def load_flow_module(ckpt: Path, device: str) -> diffsci2.models.SIModule:
    module = diffsci2.models.SIModule.load_from_checkpoint(ckpt, map_location=device)
    module.eval().to(device)
    return module


# ---------------------------------------------------------------------------
# Sampling one volume.
# ---------------------------------------------------------------------------

def sample_porosity_field(
    sampler, pixel_shape: tuple[int, int, int], coarse_n: int,
    device: str, rng: torch.Generator,
) -> torch.Tensor:
    """Sample a logit-warped GP on a coarse grid, trilinear-upsample,
    then inverse-warp to (0, 1). Returns shape (1, 1, Lx/F, Ly/F, Lz/F).
    """
    Lx, Ly, Lz = pixel_shape
    latent_shape = (Lx // LATENT_TO_PIXEL, Ly // LATENT_TO_PIXEL, Lz // LATENT_TO_PIXEL)
    # Aspect-preserving coarse grid.
    nx, ny, nz = coarse_n, coarse_n, int(coarse_n * latent_shape[2] / latent_shape[0])
    coarse = sampler.sample(shape=(nx, ny, nz), generator=rng, device=device)
    coarse = coarse.unsqueeze(0).unsqueeze(0)  # (1, 1, nx, ny, nz) in logit space
    field = torch.nn.functional.interpolate(
        coarse, size=latent_shape, mode="trilinear", align_corners=False,
    )
    return torch.sigmoid(field)   # inverse of logit → back to [0,1]


@torch.no_grad()
def generate_one(
    flow_module,
    pixel_shape: tuple[int, int, int],
    porosity_field: torch.Tensor | None,
    sigma_max: float,
    sigma_min: float,
    nsteps: int,
    device: str,
) -> torch.Tensor:
    """One full generation: ODE integration + chunk-decode.

    `porosity_field` should be either the (1,1,L/F,…) conditioning tensor
    or None for unconditional (null) sampling. For fc257/fc129 always
    pass the sampled field; for `uncond` pass None.
    """
    Lx, Ly, Lz = pixel_shape
    z_shape = (1, 4, Lx // LATENT_TO_PIXEL, Ly // LATENT_TO_PIXEL, Lz // LATENT_TO_PIXEL)

    conditioning = None if porosity_field is None else {"porosity": porosity_field.to(device)}
    z = flow_module.sample(
        shape=z_shape,
        conditioning=conditioning,
        n_steps=nsteps,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        device=device,
    )

    # Chunk decode to pixel space (main.tex Appendix B).
    decoder = flow_module.autoencoder
    cfg = chunk_decode_2.ChunkConfig(tile_size=128)
    x = chunk_decode_2.chunk_decode_3d(decoder, z, config=cfg)
    return x


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip())
    p.add_argument("--config", required=True, help="Rock YAML config.")
    p.add_argument("--variant", choices=["fc257", "fc129", "uncond"], required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--size", type=int, default=None,
                   help="Usable volume side length after border crop. "
                        "Default comes from config.generation.target_size.")
    p.add_argument("--n-samples", type=int, default=6)
    p.add_argument("--nsteps", type=int, default=None,
                   help="ODE steps. Default from config.generation.ode_steps.")
    p.add_argument("--coarse-n", type=int, default=None,
                   help="GP coarse grid size. Default from config.generation.coarse_grid_size.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--periodic", action="store_true",
                   help="Use periodic GP sampling (and assumes a periodically-wired model).")
    p.add_argument("--no-binarize", action="store_true",
                   help="Save raw float volumes instead of binary.")
    p.add_argument("--save-porosity", action="store_true",
                   help="Save the input porosity field as sidecar .porosity.npy.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    rock = cfg["rock"]
    gen_cfg = cfg["generation"]

    usable = args.size or gen_cfg["target_size"]
    border = gen_cfg["border_crop"]
    nsteps = args.nsteps or gen_cfg["ode_steps"]
    coarse_n = args.coarse_n or gen_cfg["coarse_grid_size"]
    padded = usable + 2 * border
    pixel_shape = (padded, padded, padded)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = resolve_checkpoint(rock, args.variant)
    print(f"Rock:      {rock}")
    print(f"Variant:   {args.variant}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Pixel shape (padded): {pixel_shape}")
    print(f"Usable (after crop):  {(usable, usable, usable)}")
    print()

    flow_module = load_flow_module(ckpt_path, args.device)

    sampler = None
    if args.variant in ("fc257", "fc129"):
        gp_path = gp_analysis_path(rock, args.variant, cfg)
        print(f"Loading GP analysis: {gp_path}")
        sampler = load_gp_sampler(gp_path, args.periodic)

    torch.manual_seed(args.seed)
    rng = torch.Generator(device=args.device).manual_seed(args.seed)

    sigma_min = cfg["base_training"]["sigma_min"]
    sigma_max = cfg["base_training"]["sigma_max"]

    for i in range(args.n_samples):
        t0 = time.time()
        if sampler is not None:
            porosity_field = sample_porosity_field(
                sampler, pixel_shape, coarse_n, args.device, rng,
            )
        else:
            porosity_field = None

        x = generate_one(
            flow_module, pixel_shape, porosity_field,
            sigma_max, sigma_min, nsteps, args.device,
        )
        # Border crop.
        x = x[..., border:-border, border:-border, border:-border]
        x_np = x.squeeze(0).squeeze(0).cpu().numpy()

        # Binarize at the volume mean, per main.tex Algorithm 3 line 8.
        if not args.no_binarize:
            x_np = (x_np > x_np.mean())

        out = output_dir / f"{rock.lower()}_{args.variant}_{i:03d}.npy"
        np.save(out, x_np)
        dt = time.time() - t0
        print(f"[{i+1}/{args.n_samples}] {out.name}  shape={x_np.shape}  elapsed={dt:.1f}s")

        if args.save_porosity and porosity_field is not None:
            pout = out.with_suffix(".porosity.npy")
            np.save(pout, porosity_field.squeeze().cpu().numpy())

    # Write a small manifest.
    manifest = {
        "rock": rock,
        "variant": args.variant,
        "checkpoint": str(ckpt_path),
        "n_samples": args.n_samples,
        "usable_size": usable,
        "border_crop": border,
        "pixel_shape": pixel_shape,
        "nsteps": nsteps,
        "coarse_n": coarse_n,
        "seed": args.seed,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest -> {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
