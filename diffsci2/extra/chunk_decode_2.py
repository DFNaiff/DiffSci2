# chunk_decode_2.py
# -----------------------------------------------------------------------------
# Dimension-agnostic chunked (tiled) decode for VAE decoders using Strategy B:
# multi-stage, halo-propagating streaming with CPU stage buffers + periodic BCs.
#
# This is a refactored version of chunk_decode.py that supports both 2D and 3D.
#
# WHAT THIS FILE DOES (HIGH LEVEL):
#   1) Decode a large latent tensor z_latent: [B, C, *spatial] through a decoder
#      in tiles to avoid OOM.
#   2) Split work into "stages" (S0..SN). Each stage is a contiguous chunk of
#      decoder layers.
#   3) For each stage, build stage output piece-by-piece on CPU: read minimal
#      input with required halo, run only this stage on GPU, crop valid center,
#      write to CPU buffer, free GPU memory.
#   4) Periodicity (optional): wrap at boundaries using periodic_getitem.
#
# TENSOR LAYOUT:
#   - 2D: [B, C, H, W]
#   - 3D: [B, C, H, W, D]
#   - Spatial dimensions are always the trailing dimensions after [B, C].
#   - Parameters like chunk_latent use the same axis order as the tensor.
#
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import List, Tuple, Optional, Union, Callable, Iterator
from dataclasses import dataclass
import itertools

import torch

from diffsci2.torchutils import periodic_getitem


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _device_of(module: torch.nn.Module) -> torch.device:
    """Return device of module's first parameter, or CUDA if available."""
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dtype_of(module: torch.nn.Module) -> torch.dtype:
    """Return dtype of module's first parameter, or float32."""
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32


def normalize_tuple(
    v: Union[int, Tuple, List],
    ndim: int,
    name: str
) -> Tuple[int, ...]:
    """
    Normalize an int or tuple/list to an ndim-tuple of ints.

    Examples:
        normalize_tuple(32, 2, "chunk") -> (32, 32)
        normalize_tuple([16, 32], 2, "chunk") -> (16, 32)
        normalize_tuple(32, 3, "chunk") -> (32, 32, 32)
    """
    if isinstance(v, int):
        return (v,) * ndim
    if isinstance(v, (tuple, list)) and len(v) == ndim:
        return tuple(int(x) for x in v)
    raise ValueError(f"{name} must be int or {ndim}-tuple/list. Got: {v!r}")


def normalize_bool_tuple(
    v: Union[bool, Tuple[bool, ...], List[bool]],
    ndim: int,
    name: str
) -> Tuple[bool, ...]:
    """Normalize a bool or tuple/list of bools to an ndim-tuple."""
    if isinstance(v, bool):
        return (v,) * ndim
    if isinstance(v, (tuple, list)) and len(v) == ndim:
        return tuple(bool(x) for x in v)
    raise ValueError(f"{name} must be bool or {ndim}-tuple/list. Got: {v!r}")


def make_center_spans_1d(L: int, chunk: int, radius: int) -> List[Tuple[int, int]]:
    """
    Decide center spans along one axis (in LATENT units) for Stage-0 tiling.

    Args:
        L: axis length in latent units
        chunk: desired Stage-0 tile extent including halos
        radius: Stage-0 halo radius (latent units)

    Returns:
        List of (start, end) pairs that partition [0, L).
    """
    if chunk >= L:
        return [(0, L)]

    valid = chunk - 2 * radius
    if valid <= 0:
        valid = 1

    spans = []
    pos = 0
    while pos < L:
        end = min(pos + valid, L)
        spans.append((pos, end))
        pos = end

    return spans


def iterate_nd_tiles(
    spans_per_axis: List[List[Tuple[int, int]]]
) -> Iterator[Tuple[Tuple[int, int], ...]]:
    """
    Iterate over all tile combinations across N dimensions.

    Args:
        spans_per_axis: List of span lists, one per spatial axis.
                       E.g., for 2D: [spans_H, spans_W]

    Yields:
        Tuples of (start, end) pairs, one per axis.
        E.g., for 2D: ((h0, h1), (w0, w1))
    """
    for combo in itertools.product(*spans_per_axis):
        yield combo


# ============================================================================
# RF AND SCALE COMPUTATION
# ============================================================================

def compute_stage_radii_and_scales(decoder) -> Tuple[List[int], List[int]]:
    """
    Compute per-stage cumulative RF radii and spatial scales.

    Returns:
        radii_latent: RF radius (in latent units) after each stage
        scales_after: Total spatial scale after each stage
    """
    cfg = decoder.config

    # Check for attention (makes exact chunking impossible)
    if cfg.has_mid_attn or len(cfg.attn_resolutions) > 0:
        raise NotImplementedError(
            "Chunked decoding requires NO attention in the decoder. "
            f"Found has_mid_attn={cfg.has_mid_attn}, "
            f"attn_resolutions={cfg.attn_resolutions}"
        )

    info = decoder.calculate_receptive_field()
    rf_per_block = int(info["rf_per_block"])
    rf_mid = int(info["rf_after_middle"])
    rf_final = int(info["rf_latent"])

    num_res = int(cfg.num_resolutions)
    num_res_blocks = int(cfg.num_res_blocks)
    per_up_rf = (num_res_blocks + 1) * rf_per_block

    # Cumulative radii after each stage
    radii: List[int] = [rf_mid // 2]  # After S0
    for s in range(1, num_res):
        rf_s = rf_mid + per_up_rf * s
        radii.append(rf_s // 2)
    radii.append(rf_final // 2)  # Final stage

    # Total scale after each stage
    scales: List[int] = [1]  # S0 stays at latent resolution
    for s in range(1, num_res):
        scales.append(2 ** s)
    scales.append(2 ** max(0, num_res - 1))  # Final doesn't upsample more

    return radii, scales


def compute_delta_radii(radii_latent: List[int]) -> List[int]:
    """Compute per-stage local radii (what each stage adds to RF)."""
    delta = []
    for s in range(len(radii_latent)):
        prev = radii_latent[s - 1] if s > 0 else 0
        delta.append(max(0, radii_latent[s] - prev))
    return delta


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ChunkConfig:
    """Configuration for chunked decoding."""
    device: torch.device
    dtype: torch.dtype
    ndim: int                           # 2 or 3 spatial dimensions
    batch_size: int
    z_channels: int
    spatial_shape: Tuple[int, ...]      # (H, W) or (H, W, D)
    chunk_latent: Tuple[int, ...]       # Chunk sizes per axis
    num_stages: int
    radii_latent: List[int]
    scales_after: List[int]
    delta_r_lat: List[int]
    spans_per_axis: List[List[Tuple[int, int]]]
    caps: Tuple[Optional[int], ...]     # Max output size per axis (or None)
    periodic: Tuple[bool, ...]          # Periodicity per axis
    debug: bool


@dataclass
class TileSpec:
    """Specification of a tile's position in latent coordinates."""
    ranges: Tuple[Tuple[int, int], ...]  # (start, end) per axis


@dataclass
class ReadWindow:
    """Read window for fetching input to a stage."""
    lat_ranges: Tuple[Tuple[int, int], ...]   # Ranges in latent units
    src_ranges: Tuple[Tuple[int, int], ...]   # Ranges in source (prev stage) units


@dataclass
class CropSpec:
    """How to crop stage output and where to write it."""
    tile_slices: Tuple[slice, ...]   # Slices into y_tile (stage output)
    dest_ranges: Tuple[Tuple[int, int], ...]  # Global coords in dest buffer


# ============================================================================
# CPU STAGE BUFFER
# ============================================================================

class CPUStageBuffer:
    """
    CPU tensor buffer for an entire stage's output.
    Shape: [B, C, *spatial]
    Supports periodic reads.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        ndim: int
    ):
        self.tensor = torch.zeros(shape, dtype=dtype, device="cpu")
        self.ndim = ndim  # Number of spatial dimensions

    def write_block(
        self,
        ranges: Tuple[Tuple[int, int], ...],
        tile: torch.Tensor
    ):
        """
        Write tile to destination coordinates.

        Args:
            ranges: (start, end) per spatial axis
            tile: Tensor to write, shape [B, C, *spatial_tile]
        """
        slices = [slice(None), slice(None)]  # B, C
        for (s, e) in ranges:
            slices.append(slice(s, e))
        self.tensor[tuple(slices)].copy_(tile.detach().cpu())

    def read_block_periodic(
        self,
        ranges: Tuple[Tuple[int, int], ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Read block with periodic wrapping.

        Args:
            ranges: (start, end) per spatial axis (may be out of bounds)
            device: Target device
            dtype: Target dtype

        Returns:
            Tensor [B, C, *spatial_tile] on target device
        """
        B, C = self.tensor.shape[:2]
        spatial_shape = self.tensor.shape[2:]

        # Flatten [B, C, *spatial] -> [B*C, *spatial] for periodic_getitem
        flat = self.tensor.reshape(B * C, *spatial_shape)

        # Build slices for periodic_getitem
        indices = [slice(None)]  # Keep B*C dimension
        for (s, e) in ranges:
            indices.append(slice(s, e, None))

        # Periodic fetch
        sub = periodic_getitem(flat, *indices)

        # Reshape back to [B, C, *spatial_tile]
        new_spatial = sub.shape[1:]
        result = sub.reshape(B, C, *new_spatial)

        return result.to(device=device, dtype=dtype, non_blocking=False).contiguous()


# ============================================================================
# GEOMETRY COMPUTATIONS
# ============================================================================

def compute_read_window(
    tile: TileSpec,
    stage_local_lat: int,
    src_scale: int,
    spatial_shape: Tuple[int, ...],
    periodic: Tuple[bool, ...]
) -> ReadWindow:
    """
    Compute read window for a tile at a given stage.

    Args:
        tile: The tile specification (center ranges in latent units)
        stage_local_lat: How many latent cells this stage adds to RF
        src_scale: Scale factor from latent to source (prev stage) units
        spatial_shape: Full spatial shape in latent units
        periodic: Periodicity flags per axis

    Returns:
        ReadWindow with latent and source coordinate ranges
    """
    lat_ranges = []
    src_ranges = []

    for i, ((s, e), L, per) in enumerate(zip(tile.ranges, spatial_shape, periodic)):
        if per:
            # Allow negative/over-range; periodic_getitem will wrap
            rs = s - stage_local_lat
            re = e + stage_local_lat
        else:
            # Clamp to valid range
            rs = max(0, s - stage_local_lat)
            re = min(L, e + stage_local_lat)

        lat_ranges.append((rs, re))
        src_ranges.append((rs * src_scale, re * src_scale))

    return ReadWindow(
        lat_ranges=tuple(lat_ranges),
        src_ranges=tuple(src_ranges)
    )


def compute_crop_spec(
    tile: TileSpec,
    read_win: ReadWindow,
    src_scale: int,
    dest_scale: int,
    up_factor: int,
    spatial_shape: Tuple[int, ...],
    periodic: Tuple[bool, ...]
) -> CropSpec:
    """
    Compute how to crop stage output and where to write it.

    Args:
        tile: Tile specification (center in latent units)
        read_win: Read window used to fetch input
        src_scale: Scale of source (prev stage) relative to latent
        dest_scale: Scale of destination (this stage) relative to latent
        up_factor: Upsampling factor of this stage (dest_scale / src_scale)
        spatial_shape: Full spatial shape in latent units
        periodic: Periodicity flags per axis

    Returns:
        CropSpec with tile slices and destination ranges
    """
    tile_slices = []
    dest_ranges = []

    for i, ((cs, ce), (rs, _), L, per) in enumerate(
        zip(tile.ranges, read_win.lat_ranges, spatial_shape, periodic)
    ):
        # Compute offset within the read window
        if per:
            left_lat = (cs - rs) % L
            right_lat = (ce - rs) % L
            # Handle wrap-around case
            if right_lat <= left_lat and ce != cs:
                right_lat = left_lat + (ce - cs)
        else:
            left_lat = cs - rs
            right_lat = ce - rs

        # Scale to source and then to output units
        left_src = left_lat * src_scale
        right_src = right_lat * src_scale
        y_start = left_src * up_factor
        y_end = right_src * up_factor

        # Global destination coords
        g0 = cs * dest_scale
        g1 = ce * dest_scale

        # Verify consistency
        assert (y_end - y_start) == (g1 - g0), \
            f"Length mismatch on axis {i}: tile={y_end-y_start}, dest={g1-g0}"

        tile_slices.append(slice(y_start, y_end))
        dest_ranges.append((g0, g1))

    return CropSpec(
        tile_slices=tuple(tile_slices),
        dest_ranges=tuple(dest_ranges)
    )


def generate_sub_tiles(
    tile: TileSpec,
    dest_scale: int,
    caps: Tuple[Optional[int], ...]
) -> List[TileSpec]:
    """
    Split a center tile into sub-tiles to respect output size caps.

    Args:
        tile: Original tile in latent units
        dest_scale: Scale of destination stage relative to latent
        caps: Maximum output size per axis (or None for no limit)

    Returns:
        List of sub-tile specifications
    """
    # Compute sub-tile sizes per axis
    sub_sizes = []
    for (s, e), cap in zip(tile.ranges, caps):
        length = e - s
        if cap is None:
            sub_sizes.append(length)
        else:
            # sub_len * dest_scale <= cap
            sub_lat = max(1, min(length, cap // max(dest_scale, 1)))
            sub_sizes.append(sub_lat)

    # Generate sub-tile ranges per axis
    ranges_per_axis = []
    for (s, e), sub_size in zip(tile.ranges, sub_sizes):
        axis_ranges = []
        pos = s
        while pos < e:
            end = min(pos + sub_size, e)
            axis_ranges.append((pos, end))
            pos = end
        ranges_per_axis.append(axis_ranges)

    # Combine all axes
    sub_tiles = []
    for combo in itertools.product(*ranges_per_axis):
        sub_tiles.append(TileSpec(ranges=combo))

    return sub_tiles


# ============================================================================
# STAGE RUNNERS FOR VAE DECODER
# ============================================================================

@torch.inference_mode()
def run_vae_stage0(
    decoder,
    z: torch.Tensor,
    temb: Optional[torch.Tensor]
) -> torch.Tensor:
    """Run Stage 0: post_quant_conv -> conv_in -> mid blocks."""
    h = decoder.post_quant_conv(z)
    h = decoder.conv_in(h)
    h = decoder.mid.block_1(h, temb)
    if hasattr(decoder.mid, "attn_1"):
        h = decoder.mid.attn_1(h)
    h = decoder.mid.block_2(h, temb)
    return h


@torch.inference_mode()
def run_vae_up_stage(
    decoder,
    x: torch.Tensor,
    level_index: int,
    temb: Optional[torch.Tensor]
) -> torch.Tensor:
    """Run an upsampling stage: up[level_index].blocks + upsample."""
    up = decoder.up[level_index]
    h = x
    for i in range(len(up.block)):
        h = up.block[i](h, temb)
        if len(up.attn) > i:
            h = up.attn[i](h)
    if level_index != 0:
        h = up.upsample(h)
    return h


@torch.inference_mode()
def run_vae_final_stage(
    decoder,
    x: torch.Tensor,
    temb: Optional[torch.Tensor]
) -> torch.Tensor:
    """Run final stage: up[0].blocks -> norm_out -> swish -> conv_out."""
    up0 = decoder.up[0]
    h = x
    for i in range(len(up0.block)):
        h = up0.block[i](h, temb)
        if len(up0.attn) > i:
            h = up0.attn[i](h)
    h = decoder.norm_out(h)
    h = h * torch.sigmoid(h)  # swish
    h = decoder.conv_out(h)
    if getattr(decoder.config, "tanh_out", False):
        h = torch.tanh(h)
    return h


def make_vae_stage_runner(
    decoder,
    stage_idx: int,
    num_stages: int,
    temb: Optional[torch.Tensor]
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create a stage runner function for a specific stage.

    Args:
        decoder: VAE decoder module
        stage_idx: Stage index (0 to num_stages-1)
        num_stages: Total number of stages
        temb: Time embedding (optional)

    Returns:
        Callable that takes input tensor and returns stage output
    """
    num_res = int(decoder.config.num_resolutions)

    if stage_idx == 0:
        return lambda x: run_vae_stage0(decoder, x, temb)
    elif 1 <= stage_idx <= (num_res - 1):
        level_index = num_res - stage_idx
        return lambda x: run_vae_up_stage(decoder, x, level_index, temb)
    else:
        return lambda x: run_vae_final_stage(decoder, x, temb)


# ============================================================================
# CONFIGURATION SETUP
# ============================================================================

def setup_chunk_config(
    decoder,
    z_latent: torch.Tensor,
    chunk_latent: Union[int, Tuple, List],
    device: Optional[torch.device],
    max_stage_out_chunk: Optional[Union[int, Tuple, List]],
    periodicity: Union[bool, Tuple[bool, ...], List[bool]],
    debug: bool
) -> ChunkConfig:
    """Build configuration for chunked decoding."""

    # Determine spatial dimensions
    ndim = z_latent.dim() - 2  # Subtract batch and channel dims
    if ndim not in (2, 3):
        raise ValueError(f"z_latent must be 4D (2D) or 5D (3D), got {z_latent.dim()}D")

    # Device and dtype
    if device is None:
        device = _device_of(decoder)
    else:
        device = torch.device(device)
    dtype = _dtype_of(decoder)

    # Extract shapes
    batch_size = z_latent.shape[0]
    z_channels = z_latent.shape[1]
    spatial_shape = z_latent.shape[2:]

    # Normalize parameters
    chunk = normalize_tuple(chunk_latent, ndim, "chunk_latent")
    periodic = normalize_bool_tuple(periodicity, ndim, "periodicity")

    if max_stage_out_chunk is None:
        caps = (None,) * ndim
    else:
        caps = normalize_tuple(max_stage_out_chunk, ndim, "max_stage_out_chunk")
        caps = tuple(int(c) if c is not None else None for c in caps)

    # Compute RF info
    radii_latent, scales_after = compute_stage_radii_and_scales(decoder)
    num_stages = int(decoder.config.num_resolutions) + 1
    delta_r_lat = compute_delta_radii(radii_latent)

    # Compute center spans per axis
    spans_per_axis = []
    for i, (L, ch) in enumerate(zip(spatial_shape, chunk)):
        spans = make_center_spans_1d(L, ch, radii_latent[0])
        spans_per_axis.append(spans)

    if debug:
        print(f"Spatial shape: {spatial_shape} ({ndim}D)")
        print(f"radii_latent = {radii_latent}")
        print(f"delta_r_lat = {delta_r_lat}")
        print(f"scales_after = {scales_after}")
        print(f"Spans per axis: {spans_per_axis}")
        print(f"Periodicity: {periodic}")
        print(f"Caps (dest units): {caps}")

    return ChunkConfig(
        device=device,
        dtype=dtype,
        ndim=ndim,
        batch_size=batch_size,
        z_channels=z_channels,
        spatial_shape=spatial_shape,
        chunk_latent=chunk,
        num_stages=num_stages,
        radii_latent=radii_latent,
        scales_after=scales_after,
        delta_r_lat=delta_r_lat,
        spans_per_axis=spans_per_axis,
        caps=caps,
        periodic=periodic,
        debug=debug
    )


# ============================================================================
# MAIN ENTRY POINTS
# ============================================================================

@torch.inference_mode()
def chunk_decode_strategy_b(
    decoder,
    z_latent: torch.Tensor,
    chunk_latent: Union[int, Tuple[int, ...], List[int]],
    *,
    device: Optional[Union[str, torch.device]] = None,
    time: Optional[torch.Tensor] = None,
    debug: bool = False,
    max_stage_out_chunk: Optional[Union[int, Tuple[int, ...], List[int]]] = 128,
    periodicity: Union[bool, Tuple[bool, ...], List[bool]] = False,
) -> torch.Tensor:
    """
    Dimension-agnostic chunked decode with Strategy B + optional periodic BCs.

    This function decodes a latent tensor through a VAE decoder in tiles,
    using CPU buffers for intermediate stage outputs to minimize GPU memory.

    Args:
        decoder: VAE decoder module (must have calculate_receptive_field method)
        z_latent: Latent tensor [B, C, H, W] (2D) or [B, C, H, W, D] (3D)
        chunk_latent: Tile size in latent units (int or tuple per axis)
        device: Compute device for stage tiles (default: decoder's device)
        time: Optional time embedding tensor
        debug: Enable debug prints
        max_stage_out_chunk: Cap per-stage output size (int or tuple per axis)
        periodicity: Enable periodic BCs (bool or tuple per axis)

    Returns:
        Decoded output tensor [B, C_out, *spatial_out] on CPU
    """
    # Setup configuration
    cfg = setup_chunk_config(
        decoder, z_latent, chunk_latent, device,
        max_stage_out_chunk, periodicity, debug
    )

    # Create stage runners
    stage_runners = [
        make_vae_stage_runner(decoder, s, cfg.num_stages, time)
        for s in range(cfg.num_stages)
    ]

    # CPU buffers for each stage
    stage_bufs: List[Optional[CPUStageBuffer]] = [None] * cfg.num_stages
    prev_scale = 1
    prev_buf: Optional[CPUStageBuffer] = None

    # Save training state
    was_training = decoder.training
    decoder.eval()

    # Process each stage
    for s in range(cfg.num_stages):
        stage_local_lat = cfg.delta_r_lat[s]
        dest_scale = cfg.scales_after[s]
        src_scale = prev_scale
        up_factor = dest_scale // src_scale

        if cfg.debug:
            print(f"\n=== Stage {s} ===")
            print(f"  stage_local_lat={stage_local_lat}, src_scale={src_scale}, "
                  f"dest_scale={dest_scale}, up_factor={up_factor}")

        dest_buf = stage_bufs[s]
        dest_created = dest_buf is not None

        # Iterate over all center tiles
        for center_ranges in iterate_nd_tiles(cfg.spans_per_axis):
            tile = TileSpec(ranges=center_ranges)

            # Split into sub-tiles if needed
            sub_tiles = generate_sub_tiles(tile, dest_scale, cfg.caps)

            for sub_tile in sub_tiles:
                # Compute read window
                read_win = compute_read_window(
                    sub_tile, stage_local_lat, src_scale,
                    cfg.spatial_shape, cfg.periodic
                )

                # Fetch input from source
                if s == 0:
                    # Source is z_latent
                    B, C = z_latent.shape[:2]
                    flat = z_latent.reshape(B * C, *cfg.spatial_shape)
                    indices = [slice(None)]  # Keep B*C
                    for (rs, re) in read_win.lat_ranges:
                        indices.append(slice(rs, re, None))
                    x_in_flat = periodic_getitem(flat, *indices)
                    new_spatial = x_in_flat.shape[1:]
                    x_in = x_in_flat.reshape(B, C, *new_spatial)
                    x_in = x_in.to(device=cfg.device, dtype=z_latent.dtype).contiguous()
                else:
                    # Source is previous stage buffer
                    x_in = prev_buf.read_block_periodic(
                        read_win.src_ranges, cfg.device, cfg.dtype
                    )

                # Run stage
                y_tile = stage_runners[s](x_in)

                # Allocate destination buffer if needed
                if not dest_created:
                    B_ = cfg.batch_size
                    C_out = int(y_tile.shape[1])
                    out_spatial = tuple(L * dest_scale for L in cfg.spatial_shape)
                    stage_bufs[s] = CPUStageBuffer(
                        shape=(B_, C_out, *out_spatial),
                        dtype=y_tile.dtype,
                        ndim=cfg.ndim
                    )
                    dest_buf = stage_bufs[s]
                    dest_created = True
                    if cfg.debug:
                        print(f"  Allocated dest buffer: {(B_, C_out, *out_spatial)}")

                # Compute crop coordinates
                crop = compute_crop_spec(
                    sub_tile, read_win, src_scale, dest_scale, up_factor,
                    cfg.spatial_shape, cfg.periodic
                )

                # Crop valid center
                slices = [slice(None), slice(None)]  # B, C
                slices.extend(crop.tile_slices)
                y_center = y_tile[tuple(slices)]

                # Write to destination buffer
                dest_buf.write_block(crop.dest_ranges, y_center)

                # Free GPU memory
                del y_tile, y_center, x_in
                if torch.cuda.is_available():
                    torch.cuda.synchronize(cfg.device)
                    torch.cuda.empty_cache()

        # Update for next stage
        prev_scale = dest_scale
        prev_buf = dest_buf

    # Restore training state
    decoder.train(was_training)

    return stage_bufs[-1].tensor


def chunk_decode_2d(
    decoder,
    z_latent: torch.Tensor,
    chunk_latent: Union[int, Tuple[int, int], List[int]],
    **kwargs
) -> torch.Tensor:
    """
    2D chunked decode convenience wrapper.

    Args:
        decoder: VAE decoder (2D)
        z_latent: Latent tensor [B, C, H, W]
        chunk_latent: Tile size (int or (H, W) tuple)
        **kwargs: Additional arguments for chunk_decode_strategy_b

    Returns:
        Decoded tensor [B, C_out, H_out, W_out] on CPU
    """
    assert z_latent.dim() == 4, f"Expected 4D tensor [B, C, H, W], got {z_latent.dim()}D"
    return chunk_decode_strategy_b(decoder, z_latent, chunk_latent, **kwargs)


def chunk_decode_3d(
    decoder,
    z_latent: torch.Tensor,
    chunk_latent: Union[int, Tuple[int, int, int], List[int]],
    **kwargs
) -> torch.Tensor:
    """
    3D chunked decode convenience wrapper.

    Args:
        decoder: VAE decoder (3D)
        z_latent: Latent tensor [B, C, H, W, D]
        chunk_latent: Tile size (int or (H, W, D) tuple)
        **kwargs: Additional arguments for chunk_decode_strategy_b

    Returns:
        Decoded tensor [B, C_out, H_out, W_out, D_out] on CPU
    """
    assert z_latent.dim() == 5, f"Expected 5D tensor [B, C, H, W, D], got {z_latent.dim()}D"
    return chunk_decode_strategy_b(decoder, z_latent, chunk_latent, **kwargs)


# ============================================================================
# BACKWARD COMPATIBILITY ALIAS
# ============================================================================

# Alias for backward compatibility with original chunk_decode.py
chunk_decode_strategy_b_3d = chunk_decode_3d


# ============================================================================
# EXAMPLE / TEST
# ============================================================================

if __name__ == "__main__":
    print("chunk_decode_2.py - Dimension-agnostic chunked decoder")
    print("Supports both 2D [B, C, H, W] and 3D [B, C, H, W, D] tensors")
    print()
    print("Usage:")
    print("  from diffsci2.extra.chunk_decode_2 import chunk_decode_2d, chunk_decode_3d")
    print()
    print("  # For 2D:")
    print("  result = chunk_decode_2d(decoder, z_latent_2d, chunk_latent=32)")
    print()
    print("  # For 3D:")
    print("  result = chunk_decode_3d(decoder, z_latent_3d, chunk_latent=(16, 32, 32))")
    print()
    print("  # With periodicity:")
    print("  result = chunk_decode_3d(decoder, z, chunk, periodicity=(True, True, True))")
