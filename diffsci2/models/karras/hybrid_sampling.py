"""Hybrid EM+Heun sampling for SIModule (promoted from the
localattentionnet2 capstone protocol, 2026-07-11; see
aiplayground/localattentionnet2/docs/PROTOCOL.md).

Two findings baked in:
1. Deterministic ODE sampling variance-collapses imperfect scores (solid
   bias in porous-media latents); Euler--Maruyama SDE re-equilibrates —
   but EM noise at the smallest sigmas decodes as boundary speckle.
   HYBRID: EM for sigma > sigma_stop, Heun below (Euler on the last step).
2. EM discretization inflates realized porosity when under-converged;
   nsteps must SCALE WITH VOLUME (empirical: 128 steps @128^3,
   384 @256^3+; re-verify convergence at any new size).
"""
from __future__ import annotations

import torch


def karras_sigmas(nsteps: int, sigma_min: float = 0.002,
                  sigma_max: float = 80.0, rho: float = 7.0) -> torch.Tensor:
    i = torch.arange(nsteps)
    return (sigma_max ** (1 / rho) + i / (nsteps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho


def recommended_nsteps(volume_side: int) -> int:
    """Empirical EM-convergence rule (localattentionnet2, 2026-07-09)."""
    return 128 if volume_side <= 128 else 384


@torch.inference_mode()
def hybrid_sample(module, shape, y=None, guidance: float = 1.0,
                  nsteps: int = 128, sigma_stop: float = 0.05,
                  rho: float = 7.0, seed: int | None = None,
                  device: str = 'cuda') -> torch.Tensor:
    """Sample latents with the hybrid EM+Heun integrator.

    Args:
        module: SIModule (sigma-space EDM config).
        shape:  [B, C, D, H, W] latent shape.
        y:      conditioning (scalar tensor, node field, or None).
    Returns latents (call the autoencoder/decode path yourself)."""
    sig = karras_sigmas(nsteps).to(device)
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(*shape, device=device) * sig[0]
    n = len(sig)
    for i in range(n - 1):
        em = bool(sig[i] > sigma_stop)
        method = 'euler_maruyama' if em else (
            'euler' if i == n - 2 else 'heun')
        t_c = sig[i] * torch.ones(shape[0], device=device)
        t_n = sig[i + 1] * torch.ones(shape[0], device=device)
        x = module.integration_step(x, t_c, t_n, y, guidance,
                                    method=method, noise_injection=em)
    return x
