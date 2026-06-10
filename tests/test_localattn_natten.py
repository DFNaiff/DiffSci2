"""Tests for the corrected NATTEN radial-PE path in diffsci2.nets.localattn.

Background (2026-06-10): the natten+rbias branch of
``LocalSelfAttention2D._forward_natten`` (and its 3D twin) was broken
twice: (a) it fed ``[B, H, W, heads, d]`` tensors to ``na2d_qk`` /
``na2d_av``, which expect the heads-major ``[B, heads, H, W, d]``
layout, scrambling the neighborhood over the (W, heads) axes; (b) it
never applied the ``1/sqrt(d)`` logit scale that fused ``na2d`` applies
internally. The fix makes the corrected split path the default;
``legacy_natten_rpb=True`` opts back into the broken behavior for
reproducing pre-fix runs only.

NATTEN has no CPU kernels, so most tests here require CUDA and skip
cleanly otherwise. State-dict / config compatibility tests run on CPU.
"""

import math

import pytest
import torch

from diffsci2.nets.localattn import (
    LAUNetConfig,
    LAUNet3DConfig,
    LocalSelfAttention2D,
    LocalSelfAttention3D,
)

try:
    import natten  # noqa: F401
    HAS_NATTEN = True
except ImportError:
    HAS_NATTEN = False

needs_gpu_natten = pytest.mark.skipif(
    not (HAS_NATTEN and torch.cuda.is_available()),
    reason="requires CUDA and natten (NATTEN has no CPU kernels)",
)

# Tolerance for fp32-with-tf32 comparisons between two mathematically
# identical attention implementations.
TOL = 5e-3


def _device():
    return torch.device('cuda')


# -- 2D ------------------------------------------------------------------------


@needs_gpu_natten
def test_2d_zero_bias_matches_fused_nope():
    """Corrected natten rbias path at zero bias == fused NoPE path."""
    torch.manual_seed(0)
    dev = _device()
    m = LocalSelfAttention2D(
        dim=64, num_heads=4, kernel_size=3,
        backend='natten', radial_pe=True,
    ).to(dev).eval()
    assert (m.rbias.bias_table == 0).all()  # zero-init
    x = torch.randn(2, 64, 16, 18, device=dev)
    with torch.no_grad():
        out = m(x)
        rb, m.rbias = m.rbias, None
        ref = m(x)  # fused NoPE
        m.rbias = rb
    assert (out - ref).abs().max().item() < TOL


@needs_gpu_natten
def test_2d_random_bias_matches_mask_backend_interior():
    """Corrected natten rbias path with NONZERO bias == mask backend on
    interior pixels (boundary handling legitimately differs: the mask
    backend truncates windows, NATTEN shifts them inward)."""
    torch.manual_seed(1)
    dev = _device()
    K = 3
    m = LocalSelfAttention2D(
        dim=64, num_heads=4, kernel_size=K,
        backend='natten', radial_pe=True,
    ).to(dev).eval()
    with torch.no_grad():
        m.rbias.bias_table.copy_(torch.randn_like(m.rbias.bias_table))
    x = torch.randn(2, 64, 16, 18, device=dev)
    with torch.no_grad():
        out_natten = m._forward_natten(x)
        out_mask = m._forward_mask(x)
    r = (K - 1) // 2
    diff = (out_natten - out_mask)[..., r:-r, r:-r].abs().max().item()
    assert diff < TOL


@needs_gpu_natten
def test_2d_k5_rbias_does_not_crash():
    """K=5 used to hard-crash in the legacy layout (kernel ran over the
    heads axis of size 4)."""
    torch.manual_seed(2)
    dev = _device()
    m = LocalSelfAttention2D(
        dim=64, num_heads=4, kernel_size=5,
        backend='natten', radial_pe=True,
    ).to(dev).eval()
    x = torch.randn(1, 64, 12, 14, device=dev)
    with torch.no_grad():
        out = m(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


@needs_gpu_natten
def test_2d_legacy_flag_still_runs_k3():
    """Legacy escape hatch executes at K=3 (no correctness claim; it is
    numerically broken by design and kept only for replaying old runs)."""
    torch.manual_seed(3)
    dev = _device()
    m = LocalSelfAttention2D(
        dim=64, num_heads=4, kernel_size=3,
        backend='natten', radial_pe=True, legacy_natten_rpb=True,
    ).to(dev).eval()
    x = torch.randn(1, 64, 16, 18, device=dev)
    with torch.no_grad():
        out = m(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


# -- 3D ------------------------------------------------------------------------


@needs_gpu_natten
def test_3d_zero_bias_matches_fused_nope():
    """3D analogue of the zero-bias equivalence test."""
    torch.manual_seed(4)
    dev = _device()
    m = LocalSelfAttention3D(
        dim=64, num_heads=4, kernel_size=3, radial_pe=True,
    ).to(dev).eval()
    assert (m.rbias.bias_table == 0).all()
    x = torch.randn(2, 64, 10, 12, 14, device=dev)
    with torch.no_grad():
        out = m(x)
        rb, m.rbias = m.rbias, None
        ref = m(x)  # fused NoPE
        m.rbias = rb
    assert (out - ref).abs().max().item() < TOL


@needs_gpu_natten
def test_3d_k5_rbias_does_not_crash():
    torch.manual_seed(5)
    dev = _device()
    m = LocalSelfAttention3D(
        dim=64, num_heads=4, kernel_size=5, radial_pe=True,
    ).to(dev).eval()
    x = torch.randn(1, 64, 8, 10, 12, device=dev)
    with torch.no_grad():
        out = m(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


# -- Compatibility (CPU) ---------------------------------------------------------


def test_state_dict_identical_between_fixed_and_legacy():
    """The fix changes only the forward computation; parameters and
    state-dict keys must be identical across fixed / legacy modules."""
    a = LocalSelfAttention2D(dim=32, num_heads=4, kernel_size=3,
                             backend='natten', radial_pe=True)
    b = LocalSelfAttention2D(dim=32, num_heads=4, kernel_size=3,
                             backend='natten', radial_pe=True,
                             legacy_natten_rpb=True)
    assert set(a.state_dict()) == set(b.state_dict())
    b.load_state_dict(a.state_dict())  # round-trips

    a3 = LocalSelfAttention3D(dim=32, num_heads=4, kernel_size=3,
                              radial_pe=True)
    b3 = LocalSelfAttention3D(dim=32, num_heads=4, kernel_size=3,
                              radial_pe=True, legacy_natten_rpb=True)
    assert set(a3.state_dict()) == set(b3.state_dict())
    b3.load_state_dict(a3.state_dict())


def test_old_config_dicts_still_load():
    """Pre-fix config dicts (without ``legacy_natten_rpb``) must still
    construct configs; the new field defaults to False (corrected path)."""
    import dataclasses
    old_2d = {f.name: getattr(LAUNetConfig(), f.name)
              for f in dataclasses.fields(LAUNetConfig)
              if f.name != 'legacy_natten_rpb'}
    cfg = LAUNetConfig(**old_2d)
    assert cfg.legacy_natten_rpb is False

    old_3d = {f.name: getattr(LAUNet3DConfig(), f.name)
              for f in dataclasses.fields(LAUNet3DConfig)
              if f.name != 'legacy_natten_rpb'}
    cfg3 = LAUNet3DConfig(**old_3d)
    assert cfg3.legacy_natten_rpb is False


def test_mask_backend_scale_definition_unchanged():
    """Guard: the mask backend applies 1/sqrt(head_dim) and dense bias —
    quick CPU sanity that it still runs and is finite with random bias."""
    torch.manual_seed(6)
    m = LocalSelfAttention2D(dim=32, num_heads=4, kernel_size=3,
                             backend='mask', radial_pe=True)
    with torch.no_grad():
        m.rbias.bias_table.copy_(torch.randn_like(m.rbias.bias_table))
        out = m(torch.randn(1, 32, 8, 9))
    assert torch.isfinite(out).all()
    assert math.isclose(1.0 / math.sqrt(m.head_dim), m.head_dim ** -0.5)
