"""Scalar/field-conditioned 3D LAUNet wrappers (promoted from
aiplayground/localattentionnet2/nets/scalar_model_3d.py, 2026-07-11 —
the localattentionnet2 capstone protocol, see its docs/PROTOCOL.md).

Conditioning enters the adaLN-Zero pathway: a scalar phi (or a node FIELD,
via SpatialCond spatial adaLN) is Fourier-embedded and added to the sigma
embedding. Constant field == scalar identically.

Conditioning enters through the adaLN-Zero pathway (position-free — the
same class of operation as GroupNorm, so it satisfies the thesis-chapter
constraints: periodization-safe and size-extrapolating). phi is embedded
with a Gaussian-Fourier MLP and ADDED to the sigma embedding; every block
is then globally modulated at every noise level. No spatial concat — the
scalar never touches the voxel grid.

CFG null: phi = NULL_PHI (-1.0), applied with prob cond_drop in training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from diffsci2.nets.localattn import LAUNet3D, LAUNet3DConfig
from diffsci2.nets.localattn.la_unet_3d import TimeMLP3D
from diffsci2.nets.localattn.local_attention_3d import SpatialCond


class _CondInject(nn.Module):
    """Shadows LAUNet3D.time_mlp: cond = time_mlp(t) + phi_mlp(phi).
    phi is stashed per forward call by the owning wrapper.

    phi_norm (optional LayerNorm): bounds the phi contribution's scale.
    Diagnosis 2026-07-08: on the gn latent the unbounded phi pathway ran
    away (||dc_phi|| ~700 vs sigma-emb ~20) and the adaLN projections
    learned to null its subspace EXACTLY (output delta 4e-5) — learned
    deafness as defense against a screaming input. The hearing pixnorm
    model kept ||c_phi|| ~ 8. Normalizing pins the scale forever."""

    def __init__(self, time_mlp: nn.Module, phi_mlp: nn.Module,
                 phi_norm: nn.Module | None = None):
        super().__init__()
        self.time_mlp = time_mlp
        self.phi_mlp = phi_mlp
        self.phi_norm = phi_norm
        self._phi = None        # scalar [B], set right before the net call
        self._phi_field = None  # node map [B, fd, fh, fw] (spatial adaLN)

    def forward(self, t):
        c = self.time_mlp(t)
        if self._phi_field is not None:
            # scalar-to-field (article part 2): pointwise Fourier-embed the
            # node field; blocks resolve per-level via SpatialCond.at().
            e = self.phi_mlp(self._phi_field.to(t.dtype))  # [B,f,f,f,C]
            if self.phi_norm is not None:
                e = self.phi_norm(e)
            return SpatialCond(c, e)
        if self._phi is not None:
            e = self.phi_mlp(self._phi.to(t.dtype))
            if self.phi_norm is not None:
                e = self.phi_norm(e)
            c = c + e
        return c


class ScalarCondLAUNet3D(nn.Module):
    """[B,C,D,H,W] latent + scalar phi [B] -> patchify(p) -> LAUNet3D
    (adaLN cond = sigma-emb + phi-emb) -> unpatchify(p)."""

    NULL_PHI = -1.0

    def __init__(self, patch_size: int, config: LAUNet3DConfig,
                 cond_drop: float = 0.1, data_channels: int = 2,
                 phi_fourier_scale: float = 3.0, phi_layernorm: bool = False):
        super().__init__()
        self.patch_size = patch_size
        self.cond_drop = cond_drop
        self.data_channels = data_channels
        self.net = LAUNet3D(config)
        phi_mlp = TimeMLP3D(config.cond_dim, fourier_dim=config.fourier_dim,
                            scale=phi_fourier_scale)
        # zero-init the last linear so training starts as the uncond model
        last = phi_mlp.mlp[-1] if isinstance(phi_mlp.mlp[-1], nn.Linear) else None
        if last is not None:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        phi_norm = nn.LayerNorm(config.cond_dim) if phi_layernorm else None
        self.net.time_mlp = _CondInject(self.net.time_mlp, phi_mlp, phi_norm)

    def forward(self, x, t, y=None):
        field = None
        if y is not None and y.dim() >= 4 and y.numel() > x.shape[0]:
            # FIELD conditioning (spatial adaLN). y: [B,1,fd,fh,fw] (or with
            # an extra sample()-unsqueezed dim). Constant field == scalar.
            f = y.to(x.device)
            while f.dim() > 5:
                f = f.squeeze(0)
            field = f[:, 0] if f.dim() == 5 else f      # [B,fd,fh,fw]
            if field.shape[0] != x.shape[0]:
                field = field.expand(x.shape[0], *field.shape[1:]).clone()
            if self.training and self.cond_drop > 0:
                drop = torch.rand(x.shape[0], device=x.device) < self.cond_drop
                field = torch.where(drop[:, None, None, None],
                                    torch.full_like(field, self.NULL_PHI),
                                    field)
            phi = None
        elif y is None:
            phi = torch.full((x.shape[0],), self.NULL_PHI, device=x.device)
        else:
            phi = y.reshape(-1).to(x.device)
            if phi.shape[0] != x.shape[0]:      # SIModule.sample unsqueeze quirk
                phi = phi.expand(x.shape[0]).clone()
            if self.training and self.cond_drop > 0:
                drop = torch.rand(x.shape[0], device=x.device) < self.cond_drop
                phi = torch.where(drop, torch.full_like(phi, self.NULL_PHI), phi)
        self.net.time_mlp._phi_field = field
        self.net.time_mlp._phi = phi
        p = self.patch_size
        h = x
        if p > 1:
            h = rearrange(h, "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w",
                          p1=p, p2=p, p3=p)
        out = self.net(h, t, None)
        if p > 1:
            out = rearrange(out, "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)",
                            p1=p, p2=p, p3=p, c=self.data_channels)
        self.net.time_mlp._phi = None
        self.net.time_mlp._phi_field = None
        return out


def build_scalar_launet_3d(
    patch_size: int = 2,
    cond_drop: float = 0.1,
    data_channels: int = 2,
    base_channels: int = 64,
    ch_mult: tuple[int, ...] = (1, 2, 4),
    num_blocks_per_level: int = 2,
    num_heads_base: int = 4,
    kernel_size: int = 3,
    mlp_ratio: float = 4.0,
    cond_dim: int = 256,
    fourier_dim: int = 256,
    radial_pe: bool = True,
    phi_layernorm: bool = False,
    qk_norm: bool = False,
) -> tuple[ScalarCondLAUNet3D, LAUNet3DConfig]:
    token_channels = data_channels * patch_size ** 3     # NO field channel
    if base_channels < token_channels:
        raise ValueError('lossy tokenizer')
    config = LAUNet3DConfig(
        in_channels=token_channels, out_channels=token_channels,
        base_channels=base_channels, ch_mult=tuple(ch_mult),
        num_blocks_per_level=num_blocks_per_level,
        num_heads_base=num_heads_base, kernel_size=kernel_size,
        mlp_ratio=mlp_ratio, periodic=False, cond_dim=cond_dim,
        fourier_dim=fourier_dim, radial_pe=radial_pe, qk_norm=qk_norm)
    return ScalarCondLAUNet3D(patch_size, config, cond_drop,
                              data_channels=data_channels,
                              phi_layernorm=phi_layernorm), config
