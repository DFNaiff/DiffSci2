# chunk_decode_strategy_b_3d.py
# -----------------------------------------------------------------------------
# General 3D chunked (tiled) decode for a VAEDecoder using Strategy B:
# multi-stage, halo-propagating streaming with CPU stage buffers.
#
# Key properties:
#   - Works along ALL THREE spatial axes (latent dims D, H, W), i.e., you can
#     chunk in any subset of directions. If a chunk is >= the axis length,
#     we effectively "don't chunk" that axis.
#   - Per-stage *sub-tiling* ensures that NO intermediate per-stage CUDA tensor
#     exceeds your chosen cap (`max_stage_out_chunk`) in that stage's output units.
#   - Exact results on tile centers (no approximations) for decoders with only
#     local ops (no attention).
#
# Tensor layout expected:
#   z_latent: [B, z_dim, H, W, D]
#      where D is the LAST dimension (consistent with PyTorch 3D convs).
#   The code accepts chunk sizes as (D, H, W) in LATENT units — i.e., you give
#   the chunk along the last, then the first, then the second spatial axis
#   (to align terminology "Z,Y,X" = (D,H,W)).
#
# Notation in this file:
#   - "latent units": coordinates at the decoder's input (H,W,D) grid.
#   - "source units": coordinates of the previous stage's output grid.
#   - "dest units": coordinates of the current stage's output grid.
#   - Scales are uniform across spatial axes: at stage s, total scale
#     (relative to latent) is `dest_scale = scales_after[s]` (1,2,4,...).
#
# Assumptions:
#   - Decoder has NO attention (attn_type='none', has_mid_attn=False).
#   - Patch-based convs are effectively disabled (patch_size=None).
#   - GroupNorm is used; with sufficient halos and center-cropping this is exact.
#
# -----------------------------------------------------------------------------


from __future__ import annotations
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

import torch


# ------------------------------- tiny helpers ------------------------------ #

def _device_of(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dtype_of(module: torch.nn.Module) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32


def _norm3(v: Union[int, Tuple[int, int, int], List[int]], *, name: str) -> Tuple[int, int, int]:
    """
    Normalize an int or 3-tuple/list to a 3-tuple of ints.
    We use the ordering (D, H, W) to match z_latent's last three dims [H, W, D].
    """
    if isinstance(v, int):
        return (v, v, v)
    if isinstance(v, (tuple, list)) and len(v) == 3:
        D, H, W = int(v[0]), int(v[1]), int(v[2])
        return (D, H, W)
    raise ValueError(f"{name} must be int or 3-tuple/list (D, H, W). Got: {v!r}")


def _make_center_spans_1d(L: int, chunk: int, radius0: int) -> List[Tuple[int, int]]:
    """
    Plan center spans along ONE axis in LATENT coords.

    Inputs:
      - L: axis length in latent units (e.g., D=64 or H=32)
      - chunk: desired chunk length including halos at Stage 0 (latent)
      - radius0: Stage-0 halo (latent units)

    Behavior:
      - If chunk >= L  -> single span (0, L) => no tiling on this axis.
      - Else           -> valid0 = max(1, chunk - 2*radius0); step by valid0 to cover [0, L).

    Returns list of [cs, ce) center segments that partition [0,L) (last may be shorter).
    """
    if chunk >= L:
        return [(0, L)]
    valid0 = chunk - 2 * radius0
    if valid0 <= 0:
        # Degenerate request: force at least 1 latent cell for the center step.
        valid0 = 1
    spans = []
    pos = 0
    while pos < L:
        end = min(pos + valid0, L)
        spans.append((pos, end))
        pos = end
    return spans


# ----------------------- RF radii and per-stage scales --------------------- #

def _compute_stage_radii_and_scales(decoder) -> Tuple[List[int], List[int]]:
    """
    Radii (latent) and total output scales (relative to latent) after each stage.

    Stages (S0..SN):
      - S0: post_quant_conv -> conv_in -> mid.block_1 -> (mid.attn?) -> mid.block_2
      - S1..S_{N-1}: each up stage with upsample (deepest to shallowest)
      - SN: final stage (up[0] blocks + norm_out + act + conv_out [+ tanh])

    Returns:
      radii_latent: floor(RF_after_stage / 2), length N+1
      scales_after: [1,2,4,...,2^(N-1), 2^(N-1)]
    """
    cfg = decoder.config
    # Guard: attention makes RF global → Strategy B requires approximation (not done here)
    if cfg.has_mid_attn or (len(cfg.attn_resolutions) > 0):
        raise NotImplementedError("This chunked decoder assumes NO attention in the decoder.")

    info = decoder.calculate_receptive_field()
    rf_per_block = int(info["rf_per_block"])   # 4 standard, 2 minimal
    rf_mid = int(info["rf_after_middle"])
    rf_final = int(info["rf_latent"])

    num_res = int(cfg.num_resolutions)
    num_res_blocks = int(cfg.num_res_blocks)
    per_up_rf = (num_res_blocks + 1) * rf_per_block

    # Radii after each stage (latent coords), floor(RF/2)
    radii: List[int] = [rf_mid // 2]
    for s in range(1, num_res):  # S1..S_{N-1}
        rf_s = rf_mid + per_up_rf * s
        radii.append(rf_s // 2)
    radii.append(rf_final // 2)  # SN final

    # Scales after each stage (total upsampling factor relative to latent)
    scales: List[int] = [1]
    for s in range(1, num_res):
        scales.append(2 ** s)
    scales.append(2 ** max(0, num_res - 1))

    return radii, scales


# ----------------------------- Stage runners ------------------------------- #

@torch.inference_mode()
def _run_stage0(decoder, z: torch.Tensor, temb: Optional[torch.Tensor]) -> torch.Tensor:
    h = decoder.post_quant_conv(z)
    h = decoder.conv_in(h)
    h = decoder.mid.block_1(h, temb)
    if hasattr(decoder.mid, "attn_1"):
        h = decoder.mid.attn_1(h)
    h = decoder.mid.block_2(h, temb)
    return h


@torch.inference_mode()
def _run_up_stage(decoder, x: torch.Tensor, level_index: int, temb: Optional[torch.Tensor]) -> torch.Tensor:
    up = decoder.up[level_index]
    h = x
    for i in range(len(up.block)):
        h = up.block[i](h, temb)
        if len(up.attn) > i:
            h = up.attn[i](h)
    if level_index != 0:
        h = up.upsample(h)  # ×2 in spatial dims (H, W, D)
    return h


@torch.inference_mode()
def _run_final_stage(decoder, x: torch.Tensor, temb: Optional[torch.Tensor]) -> torch.Tensor:
    up0 = decoder.up[0]
    h = x
    for i in range(len(up0.block)):
        h = up0.block[i](h, temb)
        if len(up0.attn) > i:
            h = up0.attn[i](h)
    # swish
    h = decoder.norm_out(h)
    h = h * torch.sigmoid(h)
    h = decoder.conv_out(h)
    if getattr(decoder.config, "tanh_out", False):
        h = torch.tanh(h)
    return h


# ------------------------------ CPU stage buf ------------------------------ #

class _CPUStageBuffer:
    """
    CPU tensor buffer for the WHOLE output of a stage.
    Shape: [B, C, Hs, Ws, Ds]   (stage coords; s = stage index)
    """
    def __init__(self, shape: Tuple[int, int, int, int, int], dtype: torch.dtype):
        self.tensor = torch.zeros(shape, dtype=dtype, device="cpu")

    def write_block(self,
                    z0: int, z1: int,
                    y0: int, y1: int,
                    x0: int, x1: int,
                    tile: torch.Tensor):
        """
        Write a block into [B,C,y0:y1, x0:x1, z0:z1] from 'tile' (any device).
        'tile' must already be the VALID CENTER for this block.
        """
        self.tensor[..., y0:y1, x0:x1, z0:z1].copy_(tile.detach().to("cpu"))

    def read_block(self,
                   z0: int, z1: int,
                   y0: int, y1: int,
                   x0: int, x1: int,
                   device: torch.device,
                   dtype: torch.dtype) -> torch.Tensor:
        """
        Return [B,C,y0:y1, x0:x1, z0:z1] in the requested device/dtype.
        """
        return self.tensor[..., y0:y1, x0:x1, z0:z1].to(
            device=device, dtype=dtype, non_blocking=False
        ).contiguous()


# -------------------------- Configuration & Data structures ---------------- #

@dataclass
class ChunkDecodeConfig:
    """Configuration for a chunked decode operation."""
    device: torch.device
    model_dtype: torch.dtype
    B: int
    zC: int
    H: int
    W: int
    D: int
    chD: int
    chH: int
    chW: int
    num_stages: int
    radii_latent: List[int]
    scales_after: List[int]
    delta_r_lat: List[int]
    spans_D: List[Tuple[int, int]]
    spans_H: List[Tuple[int, int]]
    spans_W: List[Tuple[int, int]]
    capD: Optional[int]
    capH: Optional[int]
    capW: Optional[int]
    debug: bool


@dataclass
class SubTileRange:
    """Represents a sub-tile range in latent coordinates."""
    z0: int
    z1: int
    y0: int
    y1: int
    x0: int
    x1: int


@dataclass
class ReadWindow:
    """Read window coordinates in both latent and source units."""
    # Latent units
    rsD_lat: int
    reD_lat: int
    rsH_lat: int
    reH_lat: int
    rsW_lat: int
    reW_lat: int
    # Source units
    rsD_src: int
    reD_src: int
    rsH_src: int
    reH_src: int
    rsW_src: int
    reW_src: int


@dataclass
class TileCropCoords:
    """Coordinates for cropping and writing a tile."""
    # Tile offsets (where to crop from y_tile)
    yD_start: int
    yD_end: int
    yH_start: int
    yH_end: int
    yW_start: int
    yW_end: int
    # Global write coords (where to write in dest buffer)
    gD0: int
    gD1: int
    gH0: int
    gH1: int
    gW0: int
    gW1: int


# -------------------------- Setup & Helper Functions ----------------------- #

def _setup_chunk_decode_config(
    decoder,
    z_latent: torch.Tensor,
    chunk_latent: Union[int, Tuple[int, int, int], List[int]],
    device: Optional[Union[str, torch.device]],
    max_stage_out_chunk: Optional[Union[int, Tuple[int, int, int], List[int]]],
    debug: bool
) -> ChunkDecodeConfig:
    """Extract and compute all configuration for chunked decode."""
    if device is None:
        device = _device_of(decoder)
    else:
        device = torch.device(device)

    model_dtype = _dtype_of(decoder)

    assert z_latent.dim() == 5, "z_latent must be [B, z_dim, H, W, D]"
    B, zC, H, W, D = z_latent.shape

    chD, chH, chW = _norm3(chunk_latent, name="chunk_latent")

    if debug:
        print(f"chunk_latent: {chD}, {chH}, {chW}")
        print(f"z_latent.shape: {z_latent.shape}")

    radii_latent, scales_after = _compute_stage_radii_and_scales(decoder)
    num_res = int(decoder.config.num_resolutions)
    num_stages = num_res + 1

    # Compute stage-local radii
    delta_r_lat = []
    for s in range(num_stages):
        prev = radii_latent[s - 1] if s > 0 else 0
        delta_r_lat.append(max(0, radii_latent[s] - prev))

    # Plan center spans
    spans_D = _make_center_spans_1d(D, chD, radii_latent[0])
    spans_H = _make_center_spans_1d(H, chH, radii_latent[0])
    spans_W = _make_center_spans_1d(W, chW, radii_latent[0])

    # Parse caps
    if max_stage_out_chunk is None:
        capD, capH, capW = None, None, None
    else:
        cD, cH, cW = _norm3(max_stage_out_chunk, name="max_stage_out_chunk")
        capD, capH, capW = int(cD), int(cH), int(cW)

    if debug:
        print(f"Latent HW D: H={H}, W={W}, D={D}")
        print("radii_latent =", radii_latent)
        print("delta_r_lat  =", delta_r_lat)
        print("scales_after =", scales_after)
        print("spans_D =", spans_D)
        print("spans_H =", spans_H)
        print("spans_W =", spans_W)
        print("caps (dest units):", (capD, capH, capW))

    return ChunkDecodeConfig(
        device=device, model_dtype=model_dtype,
        B=B, zC=zC, H=H, W=W, D=D,
        chD=chD, chH=chH, chW=chW,
        num_stages=num_stages,
        radii_latent=radii_latent,
        scales_after=scales_after,
        delta_r_lat=delta_r_lat,
        spans_D=spans_D, spans_H=spans_H, spans_W=spans_W,
        capD=capD, capH=capH, capW=capW,
        debug=debug
    )


def _compute_sub_tile_size(center_len_lat: int, cap_dest: Optional[int], dest_scale: int) -> int:
    """Compute sub-tile size in latent units given a cap in dest units."""
    if cap_dest is None:
        return center_len_lat
    return max(1, min(center_len_lat, cap_dest // max(dest_scale, 1)))


def _generate_sub_tiles(
    csD: int, ceD: int,
    csH: int, ceH: int,
    csW: int, ceW: int,
    dest_scale: int,
    capD: Optional[int],
    capH: Optional[int],
    capW: Optional[int]
) -> List[SubTileRange]:
    """Generate all sub-tile ranges for a given center block."""
    lenD_lat = ceD - csD
    lenH_lat = ceH - csH
    lenW_lat = ceW - csW

    subD = _compute_sub_tile_size(lenD_lat, capD, dest_scale)
    subH = _compute_sub_tile_size(lenH_lat, capH, dest_scale)
    subW = _compute_sub_tile_size(lenW_lat, capW, dest_scale)

    sub_tiles = []
    z0 = csD
    while z0 < ceD:
        z1 = min(z0 + subD, ceD)
        y0 = csH
        while y0 < ceH:
            y1 = min(y0 + subH, ceH)
            x0 = csW
            while x0 < ceW:
                x1 = min(x0 + subW, ceW)
                sub_tiles.append(SubTileRange(z0, z1, y0, y1, x0, x1))
                x0 = x1
            y0 = y1
        z0 = z1

    return sub_tiles


def _compute_read_window(
    sub_tile: SubTileRange,
    stage_local_lat: int,
    src_scale: int,
    H: int, W: int, D: int
) -> ReadWindow:
    """Compute read window for a sub-tile with stage-local halo."""
    rsD_lat = max(0, sub_tile.z0 - stage_local_lat)
    reD_lat = min(D, sub_tile.z1 + stage_local_lat)
    rsH_lat = max(0, sub_tile.y0 - stage_local_lat)
    reH_lat = min(H, sub_tile.y1 + stage_local_lat)
    rsW_lat = max(0, sub_tile.x0 - stage_local_lat)
    reW_lat = min(W, sub_tile.x1 + stage_local_lat)

    return ReadWindow(
        rsD_lat=rsD_lat, reD_lat=reD_lat,
        rsH_lat=rsH_lat, reH_lat=reH_lat,
        rsW_lat=rsW_lat, reW_lat=reW_lat,
        rsD_src=rsD_lat * src_scale,
        reD_src=reD_lat * src_scale,
        rsH_src=rsH_lat * src_scale,
        reH_src=reH_lat * src_scale,
        rsW_src=rsW_lat * src_scale,
        reW_src=reW_lat * src_scale,
    )


def _compute_tile_crop_coords(
    sub_tile: SubTileRange,
    read_win: ReadWindow,
    src_scale: int,
    dest_scale: int,
    up_factor: int
) -> TileCropCoords:
    """Compute where to crop from tile and where to write in destination."""
    # D axis
    leftD_lat = sub_tile.z0 - read_win.rsD_lat
    rightD_lat = sub_tile.z1 - read_win.rsD_lat
    leftD_src = leftD_lat * src_scale
    rightD_src = rightD_lat * src_scale
    yD_start = leftD_src * up_factor
    yD_end = rightD_src * up_factor
    gD0 = sub_tile.z0 * dest_scale
    gD1 = sub_tile.z1 * dest_scale

    # H axis
    leftH_lat = sub_tile.y0 - read_win.rsH_lat
    rightH_lat = sub_tile.y1 - read_win.rsH_lat
    leftH_src = leftH_lat * src_scale
    rightH_src = rightH_lat * src_scale
    yH_start = leftH_src * up_factor
    yH_end = rightH_src * up_factor
    gH0 = sub_tile.y0 * dest_scale
    gH1 = sub_tile.y1 * dest_scale

    # W axis
    leftW_lat = sub_tile.x0 - read_win.rsW_lat
    rightW_lat = sub_tile.x1 - read_win.rsW_lat
    leftW_src = leftW_lat * src_scale
    rightW_src = rightW_lat * src_scale
    yW_start = leftW_src * up_factor
    yW_end = rightW_src * up_factor
    gW0 = sub_tile.x0 * dest_scale
    gW1 = sub_tile.x1 * dest_scale

    # Sanity checks
    assert (yD_end - yD_start) == (gD1 - gD0), "D length mismatch"
    assert (yH_end - yH_start) == (gH1 - gH0), "H length mismatch"
    assert (yW_end - yW_start) == (gW1 - gW0), "W length mismatch"

    return TileCropCoords(
        yD_start=yD_start, yD_end=yD_end,
        yH_start=yH_start, yH_end=yH_end,
        yW_start=yW_start, yW_end=yW_end,
        gD0=gD0, gD1=gD1,
        gH0=gH0, gH1=gH1,
        gW0=gW0, gW1=gW1,
    )


def _process_sub_tile(
    sub_tile: SubTileRange,
    stage_idx: int,
    cfg: ChunkDecodeConfig,
    z_latent: torch.Tensor,
    prev_buf: Optional[_CPUStageBuffer],
    dest_buf: _CPUStageBuffer,
    run_stage_fn,
    src_scale: int,
    dest_scale: int,
    up_factor: int,
    stage_local_lat: int
) -> None:
    """Process a single sub-tile through one stage."""
    # Compute read window
    read_win = _compute_read_window(
        sub_tile, stage_local_lat, src_scale,
        cfg.H, cfg.W, cfg.D
    )

    # Fetch source block
    if stage_idx == 0:
        x_in = z_latent[...,
                        read_win.rsH_lat:read_win.reH_lat,
                        read_win.rsW_lat:read_win.reW_lat,
                        read_win.rsD_lat:read_win.reD_lat]
        x_in = x_in.to(device=cfg.device, dtype=z_latent.dtype, non_blocking=False).contiguous()
    else:
        x_in = prev_buf.read_block(
            z0=read_win.rsD_src, z1=read_win.reD_src,
            y0=read_win.rsH_src, y1=read_win.reH_src,
            x0=read_win.rsW_src, x1=read_win.reW_src,
            device=cfg.device, dtype=cfg.model_dtype
        )

    if cfg.debug:
        print(f"[S{stage_idx}] sub-tile D:{sub_tile.z0}:{sub_tile.z1} "
              f"H:{sub_tile.y0}:{sub_tile.y1} W:{sub_tile.x0}:{sub_tile.x1} | "
              f"x_in.shape={tuple(x_in.shape)}")

    # Run stage
    y_tile = run_stage_fn(stage_idx, x_in)

    # Compute crop coordinates
    crop_coords = _compute_tile_crop_coords(
        sub_tile, read_win, src_scale, dest_scale, up_factor
    )

    # Crop and write
    y_center = y_tile[...,
                      crop_coords.yH_start:crop_coords.yH_end,
                      crop_coords.yW_start:crop_coords.yW_end,
                      crop_coords.yD_start:crop_coords.yD_end]

    dest_buf.write_block(
        z0=crop_coords.gD0, z1=crop_coords.gD1,
        y0=crop_coords.gH0, y1=crop_coords.gH1,
        x0=crop_coords.gW0, x1=crop_coords.gW1,
        tile=y_center
    )

    # Cleanup
    del y_tile, y_center, x_in
    if torch.cuda.is_available():
        torch.cuda.synchronize(cfg.device)
        torch.cuda.empty_cache()


# ------------------------------ Main entry --------------------------------- #

@torch.inference_mode()
def chunk_decode_strategy_b_3d(  # noqa: C901
    decoder,                    # your VAEDecoder
    z_latent: torch.Tensor,     # [B, z_dim, H, W, D] (CPU or GPU)
    chunk_latent: Union[int, Tuple[int, int, int], List[int]],
    *,
    # Compute device for each per-stage tile:
    device: Optional[Union[str, torch.device]] = None,
    # Optional time embedding input (forwarded to each stage):
    time: Optional[torch.Tensor] = None,
    # Debug prints (heavy):
    debug: bool = False,
    # Cap any per-stage CUDA tensor spatial size (in that stage's OUTPUT units).
    # Accepts int or (D_out, H_out, W_out).
    max_stage_out_chunk: Optional[Union[int, Tuple[int, int, int], List[int]]] = 128,
) -> torch.Tensor:
    """
    General 3D chunk decoder using Strategy B (multi-stage halo streaming).

    Parameters
    ----------
    decoder : VAEDecoder
        Your decoder module (no attention).
    z_latent : torch.Tensor
        Latent tensor [B, z_dim, H, W, D], can be on CPU or GPU.
    chunk_latent : int | (D, H, W)
        Desired *Stage-0* chunk sizes in LATENT units along (D, H, W).
        - If an axis chunk >= axis length, that axis is not chunked (single span).
        - Values include the Stage-0 halo; the center width per axis is
          (chunk - 2*radius0), auto-clamped to >=1.
    device : str | torch.device | None
        Compute device for per-stage tiles. Defaults to decoder's device.
    time : torch.Tensor | None
        Optional time embedding input, forwarded into each stage.
    debug : bool
        Print detailed index math and shapes per tile/sub-tile.
    max_stage_out_chunk : int | (D_out, H_out, W_out) | None
        Upper bound for any single per-stage CUDA tensor’s spatial extents,
        measured in that stage's OUTPUT units (after that stage's upsampling).
        Use None to disable (not recommended for large volumes).

    Returns
    -------
    torch.Tensor
        Final decoded tensor on CPU: [B, out_C, H*scale, W*scale, D*scale].

    Notes
    -----
    Index mapping per stage s (identical on each axis):

      Let (cs, ce) be a *center span in LATENT units*.
      For a sub-center [l0, l1] (latent):

        - Stage-local halo (latent):    h_lat_s = delta_r_lat[s]
        - Source scale:                 src_scale = scales_after[s-1] (for s>0; =1 for s=0)
        - Dest scale:                   dest_scale = scales_after[s]
        - Up factor:                    up = dest_scale // src_scale (1 or 2)

      Read window in LATENT units:      [rs_lat, re_lat] = [max(l0-h_lat_s, 0), min(l1+h_lat_s, L)]
      Map to SOURCE coords:             [rs_src, re_src] = [rs_lat*src_scale, re_lat*src_scale]
      Offsets INSIDE read window (src): left_src  = (l0 - rs_lat)*src_scale
                                        right_src = (l1 - rs_lat)*src_scale
      Offsets in stage output (tile y): y_start = left_src  * up
                                        y_end   = right_src * up

      Global write range in DEST units: write0 = l0 * dest_scale
                                        write1 = l1 * dest_scale

      Lengths match by construction:
        (y_end - y_start) == (write1 - write0)
    """
    # Setup configuration
    cfg = _setup_chunk_decode_config(
        decoder, z_latent, chunk_latent, device, max_stage_out_chunk, debug
    )

    # Stage runner closure
    num_res = int(decoder.config.num_resolutions)

    def run_stage(s: int, x: torch.Tensor) -> torch.Tensor:
        if s == 0:
            return _run_stage0(decoder, x, time)
        elif 1 <= s <= (num_res - 1):
            level_index = (num_res - s)
            return _run_up_stage(decoder, x, level_index, time)
        else:
            return _run_final_stage(decoder, x, time)

    # Initialize stage buffers
    stage_bufs: List[Optional[_CPUStageBuffer]] = [None] * cfg.num_stages
    prev_scale = 1
    prev_buf: Optional[_CPUStageBuffer] = None

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
            print(f"radius_total_lat={cfg.radii_latent[s]}, stage_local_lat={stage_local_lat}, "
                  f"src_scale={src_scale}, dest_scale={dest_scale}, up_factor={up_factor}")

        dest_buf = stage_bufs[s]
        dest_created = dest_buf is not None

        # Iterate over center blocks
        for (csD, ceD) in cfg.spans_D:
            for (csH, ceH) in cfg.spans_H:
                for (csW, ceW) in cfg.spans_W:
                    # Generate sub-tiles for this center block
                    sub_tiles = _generate_sub_tiles(
                        csD, ceD, csH, ceH, csW, ceW,
                        dest_scale, cfg.capD, cfg.capH, cfg.capW
                    )

                    # Process each sub-tile
                    for sub_tile in sub_tiles:
                        # Allocate dest buffer on first tile
                        if not dest_created:
                            # Need to run one tile to get output shape
                            temp_read = _compute_read_window(
                                sub_tile, stage_local_lat, src_scale,
                                cfg.H, cfg.W, cfg.D
                            )
                            if s == 0:
                                temp_in = z_latent[...,
                                                   temp_read.rsH_lat:temp_read.reH_lat,
                                                   temp_read.rsW_lat:temp_read.reW_lat,
                                                   temp_read.rsD_lat:temp_read.reD_lat]
                                temp_in = temp_in.to(device=cfg.device, dtype=z_latent.dtype)
                            else:
                                temp_in = prev_buf.read_block(
                                    z0=temp_read.rsD_src, z1=temp_read.reD_src,
                                    y0=temp_read.rsH_src, y1=temp_read.reH_src,
                                    x0=temp_read.rsW_src, x1=temp_read.reW_src,
                                    device=cfg.device, dtype=cfg.model_dtype
                                )
                            temp_out = run_stage(s, temp_in)
                            C_out = int(temp_out.shape[1])
                            Hs = cfg.H * dest_scale
                            Ws = cfg.W * dest_scale
                            Ds = cfg.D * dest_scale
                            stage_bufs[s] = _CPUStageBuffer(
                                shape=(cfg.B, C_out, Hs, Ws, Ds),
                                dtype=temp_out.dtype
                            )
                            dest_buf = stage_bufs[s]
                            dest_created = True
                            del temp_in, temp_out

                            if cfg.debug:
                                print(f"[S{s}] Alloc dest buffer shape={(cfg.B, C_out, Hs, Ws, Ds)}")

                        _process_sub_tile(
                            sub_tile, s, cfg, z_latent, prev_buf, dest_buf,
                            run_stage, src_scale, dest_scale, up_factor, stage_local_lat
                        )

        prev_scale = dest_scale
        prev_buf = dest_buf

    decoder.train(was_training)
    return stage_bufs[-1].tensor


# ------------------------------ Example (off) ------------------------------ #
if __name__ == "__main__":
    # Example usage (disabled by default):
    #
    # from yourmodule import VAENetConfig, VAEDecoder
    # cfg = VAENetConfig(
    #     dimension=3, in_channels=1, out_channels=1, z_dim=4, z_channels=4,
    #     ch=32, ch_mult=[1,2,4,4], num_res_blocks=2,
    #     attn_type="none", has_mid_attn=False, patch_size=None
    # )
    # dec = VAEDecoder(cfg).eval().to("cuda:0")
    #
    # z = torch.randn(1, cfg.z_dim, 32, 32, 64)  # [B,zC,H,W,D]
    #
    # # Chunk along D only (like the old Z-only path), S0 chunk=16; cap deep-stage tiles to 64^3:
    # y = chunk_decode_strategy_b_3d(
    #     dec, z,
    #     chunk_latent=16,                 # int -> (16,16,16) but spans_H/W collapse to (0,H),(0,W)
    #     device="cuda:0",
    #     debug=True,
    #     max_stage_out_chunk=(64, 64, 64) # caps in stage output coords
    # )
    #
    # # Chunk along all axes with different sizes:
    # y2 = chunk_decode_strategy_b_3d(
    #     dec, z,
    #     chunk_latent=(32, 16, 16),       # (D,H,W) in latent units
    #     device="cuda:0",
    #     max_stage_out_chunk=(96, 96, 64)
    # )
    #
    pass
