"""Latent-space adapters for VAE-latent diffusion (promoted from
aiplayground/localattentionnet2/nets/latent_squash.py, 2026-07-11).

Invertible tail-taming: u = a * asinh(z / a).

Why: the raw s4 latent is heavy-tailed (kurtosis ~6.6, |z| up to ~8 sigma)
and the tail values are the PORES (14.6x outlier enrichment in pore blocks,
e3 diag). MSE on raw z stalls (e2); huber "fixes" the stall by capping
exactly the gradients that carry pore structure -> under-trained high-sigma
score, ignored conditioning, missing scattered-pore population (e3 diag).
Training diffusion on u instead makes MSE (the theory-correct conditional-
mean denoiser) workable: kurtosis drops to ~2, max|u| ~ 4.6 std_u.

The map is smooth, strictly monotone, identity-like for |z| < a, and
exactly invertible: z = a * sinh(u / a). Applied per channel with
a_c = mult * std_c. Samples of u map to samples of z with no bias.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _bcast(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return a.to(x.device, x.dtype).view(1, -1, *([1] * (x.dim() - 2)))


def squash(z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    aa = _bcast(a, z)
    return aa * torch.asinh(z / aa)


def unsquash(u: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    aa = _bcast(a, u)
    return aa * torch.sinh(u / aa)


class _SquashedDecoder(nn.Module):
    """decoder'(u) = decoder(a * sinh(u / a)) — pointwise pre-map, so it is
    exact under chunked decode (chunk_decode_3d)."""

    def __init__(self, decoder: nn.Module, a: torch.Tensor):
        super().__init__()
        self.inner = decoder
        self.register_buffer('a', a.clone())

    def forward(self, u):
        return self.inner(unsquash(u, self.a))


class SquashedVAE(nn.Module):
    """Adapter so all downstream code (SIModule.sample -> .decode, and
    chunk_decode_3d -> .decoder) transparently operates in u-space.

    Only the decode path is guaranteed; .encode assumes the inner encode
    returns a plain tensor (training uses pre-squashed cached latents, so
    the encode path is not exercised).
    """

    def __init__(self, vae: nn.Module, a):
        super().__init__()
        self.vae = vae
        a = torch.as_tensor(a, dtype=torch.float32)
        self.register_buffer('a', a)
        self.decoder = _SquashedDecoder(vae.decoder, a)

    def encode(self, x, *args, **kw):
        return squash(self.vae.encode(x, *args, **kw), self.a)

    def decode(self, u, *args, **kw):
        return self.vae.decode(unsquash(u, self.a), *args, **kw)


class _AffineDecoder(nn.Module):
    """decoder'(u) = decoder(u * std + mean) — pointwise, chunk-safe."""

    def __init__(self, decoder: nn.Module, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.inner = decoder
        self.register_buffer('m', mean.clone())
        self.register_buffer('s', std.clone())

    def forward(self, u):
        return self.inner(u * _bcast(self.s, u) + _bcast(self.m, u))


class StandardizedVAE(nn.Module):
    """Adapter for diffusion trained on u = (z - mean) / std per channel
    (e.g. the groupnorm_sft_v2 latent with per-channel offsets)."""

    def __init__(self, vae: nn.Module, mean, std):
        super().__init__()
        self.vae = vae
        m = torch.as_tensor(mean, dtype=torch.float32)
        s = torch.as_tensor(std, dtype=torch.float32)
        self.register_buffer('m', m)
        self.register_buffer('s', s)
        self.decoder = _AffineDecoder(vae.decoder, m, s)

    def encode(self, x, *a, **kw):
        z = self.vae.encode(x, *a, **kw)
        z = z[0] if isinstance(z, tuple) else z
        return (z - _bcast(self.m, z)) / _bcast(self.s, z)

    def decode(self, u, *a, **kw):
        return self.vae.decode(u * _bcast(self.s, u) + _bcast(self.m, u),
                               *a, **kw)
