"""
Spatial-parallel layer wrappers for PUNetG.

Each wrapper replaces a standard layer with one that handles
communication across GPUs when the spatial domain is split along one axis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .halo_exchange import exchange_halos, pad_with_halos


class SpatialParallelConv3d(nn.Module):
    """Drop-in replacement for Conv3d / CircularConv3d with halo exchange.

    The split dimension (D by default) is padded via halo exchange with
    neighboring GPUs. Other spatial dimensions (H, W) are padded locally
    using the specified mode (zero or circular).

    Parameters
    ----------
    conv : nn.Conv3d
        The underlying convolution with padding=0. Weights are preserved.
    kernel_size : int
        Cubic kernel size (must be odd).
    hw_pad_modes : tuple of str
        Padding mode for (H, W): each is 'zeros' or 'circular'.
    ctx : SpatialContext
        Distributed context.
    """

    def __init__(self, conv, kernel_size, hw_pad_modes, ctx):
        super().__init__()
        self.conv = conv  # Conv3d with padding=0, weights already set
        self.halo_width = kernel_size // 2
        self.hw_pad_modes = hw_pad_modes  # (H_mode, W_mode)
        self.ctx = ctx

    def forward(self, x):
        p = self.halo_width
        if p == 0:
            return self.conv(x)

        dim = self.ctx.split_dim  # 2 for D in [B, C, D, H, W]

        # 1. Halo exchange along split dim (D)
        recv_left, recv_right = exchange_halos(x, p, self.ctx)
        x = pad_with_halos(x, recv_left, recv_right, dim=dim)

        # 2. Pad H (dim 3) locally
        h_mode = self.hw_pad_modes[0]
        if h_mode == 'circular':
            x = F.pad(x, (0, 0, p, p, 0, 0), mode='circular')
        else:
            x = F.pad(x, (0, 0, p, p, 0, 0), mode='constant', value=0)

        # 3. Pad W (dim 4) locally
        w_mode = self.hw_pad_modes[1]
        if w_mode == 'circular':
            x = F.pad(x, (p, p, 0, 0, 0, 0), mode='circular')
        else:
            x = F.pad(x, (p, p, 0, 0, 0, 0), mode='constant', value=0)

        return self.conv(x)


class SpatialParallelGroupNorm(nn.Module):
    """GroupNorm (LN or RMS variant) that synchronizes statistics across spatial shards.

    Wraps GroupLNorm or GroupRMSNorm. Re-implements the forward pass with
    AllReduce for correct global statistics when the spatial domain is split.

    Parameters
    ----------
    original : nn.Module
        The original GroupLNorm or GroupRMSNorm instance.
    norm_type : str
        'GroupLN' or 'GroupRMS'.
    ctx : SpatialContext
        Distributed context.
    """

    def __init__(self, original, norm_type, ctx):
        super().__init__()
        self.num_groups = original.num_groups
        self.num_channels = original.num_channels
        self.eps = original.eps
        self.affine = original.affine
        self.norm_type = norm_type
        self.ctx = ctx

        if self.affine:
            # Share the same parameters (not a copy)
            self.weight = original.weight
            self.bias = original.bias

    def forward(self, x):
        B, C = x.shape[:2]
        G = self.num_groups

        # Reshape: [B, C, D, H, W] -> [B, G, C//G, D, H, W]
        x = x.view(B, G, C // G, *x.shape[2:])
        reduce_dims = tuple(range(2, x.dim()))  # (2, 3, 4, 5) for 5D+1

        # Count local elements per group
        local_count = 1
        for d in reduce_dims:
            local_count *= x.shape[d]
        local_count = torch.tensor(local_count, dtype=x.dtype, device=x.device)

        if self.norm_type in ('GroupLN', 'GroupNorm'):
            # Need global mean for centering (GroupLN and torch.nn.GroupNorm both center)
            local_sum = x.sum(dim=reduce_dims, keepdim=True)  # [B, G, 1, 1, 1, 1]
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM, group=self.ctx.process_group)
            total_count = local_count * self.ctx.world_size
            mean = local_sum / total_count
            x = x - mean

        # RMS normalization (used by all: GroupLN centers then RMS, GroupRMS just RMS,
        # GroupNorm centers then divides by std = same as center + RMS on centered data)
        local_sum_sq = x.pow(2).sum(dim=reduce_dims, keepdim=True)  # [B, G, 1, 1, 1, 1]
        dist.all_reduce(local_sum_sq, op=dist.ReduceOp.SUM, group=self.ctx.process_group)
        total_count = local_count * self.ctx.world_size
        rms = torch.sqrt(local_sum_sq / total_count + self.eps)
        x = x / rms

        # Reshape back: [B, G, C//G, D, H, W] -> [B, C, D, H, W]
        x = x.view(B, C, *x.shape[3:])

        if self.affine:
            w = self.weight.view(1, C, *([1] * (x.dim() - 2)))
            b = self.bias.view(1, C, *([1] * (x.dim() - 2)))
            x = x * w + b

        return x
