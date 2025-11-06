from typing import Literal
from jaxtyping import Float
from torch import Tensor
import torch
import numpy as np


def _get_grid_generation_order(grid_map: list[int]) -> tuple[list[tuple[int, int, int]], int]:
    """
    Returns positions in generation order based on coordinate parity patterns.

    The generation order follows 8 patterns based on whether each coordinate is even or odd:
    1. (even, even, even): positions (i*2, j*2, k*2) - all coordinates even
    2. (even, even, odd): positions (i*2, j*2, k*2+1) - x,y even, z odd
    3. (even, odd, even): positions (i*2, j*2+1, k*2) - x,z even, y odd
    4. (even, odd, odd): positions (i*2, j*2+1, k*2+1) - x even, y,z odd
    5. (odd, even, even): positions (i*2+1, j*2, k*2) - x odd, y,z even
    6. (odd, even, odd): positions (i*2+1, j*2, k*2+1) - x,z odd, y even
    7. (odd, odd, even): positions (i*2+1, j*2+1, k*2) - x,y odd, z even
    8. (odd, odd, odd): positions (i*2+1, j*2+1, k*2+1) - all coordinates odd

    Within each pattern, positions are generated in lexicographical order.

    Args:
        grid_map: [nx, ny, nz] - number of grid steps in each direction

    Returns:
        Tuple of:
        - List of (i, j, k) tuples in generation order
        - Integer count of positions in the first pattern (all even) - this is corner_inds_limit
    """
    nx, ny, nz = grid_map
    positions = []

    # Pattern 1: (even, even, even) - (i*2, j*2, k*2)
    pattern1 = []
    for i in range((nx + 1) // 2):  # i*2 < nx, so i < (nx+1)//2
        for j in range((ny + 1) // 2):
            for k in range((nz + 1) // 2):
                pos = (i * 2, j * 2, k * 2)
                if pos[0] < nx and pos[1] < ny and pos[2] < nz:
                    pattern1.append(pos)
    pattern1.sort()  # Lexicographical order
    positions.extend(pattern1)
    corner_inds_limit = len(pattern1)

    # Pattern 2: (even, even, odd) - (i*2, j*2, k*2+1)
    pattern2 = []
    for i in range((nx + 1) // 2):
        for j in range((ny + 1) // 2):
            for k in range(nz // 2):  # k*2+1 < nz, so k < (nz-1+1)//2 = nz//2
                pos = (i * 2, j * 2, k * 2 + 1)
                if pos[0] < nx and pos[1] < ny and pos[2] < nz:
                    pattern2.append(pos)
    pattern2.sort()
    positions.extend(pattern2)

    # Pattern 3: (even, odd, even) - (i*2, j*2+1, k*2)
    pattern3 = []
    for i in range((nx + 1) // 2):
        for j in range(ny // 2):
            for k in range((nz + 1) // 2):
                pos = (i * 2, j * 2 + 1, k * 2)
                if pos[0] < nx and pos[1] < ny and pos[2] < nz:
                    pattern3.append(pos)
    pattern3.sort()
    positions.extend(pattern3)

    # Pattern 4: (even, odd, odd) - (i*2, j*2+1, k*2+1)
    pattern4 = []
    for i in range((nx + 1) // 2):
        for j in range(ny // 2):
            for k in range(nz // 2):
                pos = (i * 2, j * 2 + 1, k * 2 + 1)
                if pos[0] < nx and pos[1] < ny and pos[2] < nz:
                    pattern4.append(pos)
    pattern4.sort()
    positions.extend(pattern4)

    # Pattern 5: (odd, even, even) - (i*2+1, j*2, k*2)
    pattern5 = []
    for i in range(nx // 2):
        for j in range((ny + 1) // 2):
            for k in range((nz + 1) // 2):
                pos = (i * 2 + 1, j * 2, k * 2)
                if pos[0] < nx and pos[1] < ny and pos[2] < nz:
                    pattern5.append(pos)
    pattern5.sort()
    positions.extend(pattern5)

    # Pattern 6: (odd, even, odd) - (i*2+1, j*2, k*2+1)
    pattern6 = []
    for i in range(nx // 2):
        for j in range((ny + 1) // 2):
            for k in range(nz // 2):
                pos = (i * 2 + 1, j * 2, k * 2 + 1)
                if pos[0] < nx and pos[1] < ny and pos[2] < nz:
                    pattern6.append(pos)
    pattern6.sort()
    positions.extend(pattern6)

    # Pattern 7: (odd, odd, even) - (i*2+1, j*2+1, k*2)
    pattern7 = []
    for i in range(nx // 2):
        for j in range(ny // 2):
            for k in range((nz + 1) // 2):
                pos = (i * 2 + 1, j * 2 + 1, k * 2)
                if pos[0] < nx and pos[1] < ny and pos[2] < nz:
                    pattern7.append(pos)
    pattern7.sort()
    positions.extend(pattern7)

    # Pattern 8: (odd, odd, odd) - (i*2+1, j*2+1, k*2+1)
    pattern8 = []
    for i in range(nx // 2):
        for j in range(ny // 2):
            for k in range(nz // 2):
                pos = (i * 2 + 1, j * 2 + 1, k * 2 + 1)
                if pos[0] < nx and pos[1] < ny and pos[2] < nz:
                    pattern8.append(pos)
    pattern8.sort()
    positions.extend(pattern8)

    return positions, corner_inds_limit



def _get_cube_spatial_bounds(
    grid_pos: tuple[int, int, int],
    base_shape: list[int],
    overlap_size: int,
    final_shape: list[int]
) -> tuple[slice, slice, slice]:
    """
    Computes spatial bounds for a cube at grid position (i, j, k).

    Args:
        grid_pos: (i, j, k) grid position
        base_shape: [channels, dx, dy, dz] - base cube shape
        overlap_size: overlap between cubes
        final_shape: [channels, final_dx, final_dy, final_dz] - final volume shape

    Returns:
        Tuple of 3 slice objects for extracting/placing cubes
    """
    i, j, k = grid_pos
    base_size = base_shape[1:]  # [dx, dy, dz]
    final_size = final_shape[1:]  # [final_dx, final_dy, final_dz]
    overlap_half = overlap_size // 2

    # Compute spatial start positions (with overlap)
    start_x = i * base_size[0] - overlap_half
    start_y = j * base_size[1] - overlap_half
    start_z = k * base_size[2] - overlap_half

    # Compute extended size
    extended_size = [s + overlap_size for s in base_size]

    # Compute end positions
    end_x = start_x + extended_size[0]
    end_y = start_y + extended_size[1]
    end_z = start_z + extended_size[2]

    # Clamp to volume boundaries
    start_x = max(0, start_x)
    start_y = max(0, start_y)
    start_z = max(0, start_z)
    end_x = min(final_size[0], end_x)
    end_y = min(final_size[1], end_y)
    end_z = min(final_size[2], end_z)

    return (slice(start_x, end_x), slice(start_y, end_y), slice(start_z, end_z))


def _build_inpaint_mask(
    grid_pos: tuple[int, int, int],
    generated_positions: set[tuple[int, int, int]],
    base_shape: list[int],
    overlap_size: int,
    final_shape: list[int]
) -> torch.Tensor:
    """
    Creates mask for inpainting at current position.
    Mask is 1 where data exists from previously generated cubes.

    Args:
        grid_pos: (i, j, k) current grid position
        generated_positions: set of previously generated (i, j, k) positions
        base_shape: [channels, dx, dy, dz] - base cube shape
        overlap_size: overlap between cubes
        final_shape: [channels, final_dx, final_dy, final_dz] - final volume shape

    Returns:
        Mask tensor of shape [channels, extended_size, extended_size, extended_size]
        where 1 indicates known data, 0 indicates unknown
    """
    # Get spatial bounds for current cube
    current_bounds = _get_cube_spatial_bounds(grid_pos, base_shape, overlap_size, final_shape)
    sx, sy, sz = current_bounds

    # Compute extended size
    extended_size = [s.stop - s.start for s in current_bounds]

    # Initialize mask (all zeros = unknown)
    mask = torch.zeros((base_shape[0], extended_size[0], extended_size[1], extended_size[2]))

    # For each previously generated position, mark overlap regions
    for prev_pos in generated_positions:
        prev_bounds = _get_cube_spatial_bounds(prev_pos, base_shape, overlap_size, final_shape)
        psx, psy, psz = prev_bounds

        # Find intersection region
        intersect_x_start = max(sx.start, psx.start)
        intersect_x_end = min(sx.stop, psx.stop)
        intersect_y_start = max(sy.start, psy.start)
        intersect_y_end = min(sy.stop, psy.stop)
        intersect_z_start = max(sz.start, psz.start)
        intersect_z_end = min(sz.stop, psz.stop)

        # If there's an intersection
        if (intersect_x_start < intersect_x_end and
            intersect_y_start < intersect_y_end and
            intersect_z_start < intersect_z_end):  # noqa: E129

            # Convert to local coordinates in current cube
            local_x_start = intersect_x_start - sx.start
            local_x_end = intersect_x_end - sx.start
            local_y_start = intersect_y_start - sy.start
            local_y_end = intersect_y_end - sy.start
            local_z_start = intersect_z_start - sz.start
            local_z_end = intersect_z_end - sz.start

            # Set mask to 1 in intersection region
            mask[:, local_x_start:local_x_end, local_y_start:local_y_end, local_z_start:local_z_end] = 1.0

    return mask


def _extract_noise_slice(
    noise_cube: torch.Tensor,
    spatial_bounds: tuple[slice, slice, slice]
) -> torch.Tensor:
    """
    Extracts noise slice from big noise cube using spatial bounds.

    Args:
        noise_cube: Big noise tensor of shape [channels, final_dx, final_dy, final_dz]
        spatial_bounds: Tuple of 3 slice objects

    Returns:
        Noise slice tensor of appropriate shape
    """
    sx, sy, sz = spatial_bounds
    return noise_cube[:, sx, sy, sz]


def _combine_cube_into_volume(
    volume: torch.Tensor,
    cube: torch.Tensor,
    spatial_bounds: tuple[slice, slice, slice],
    blend_mode: str = 'latest'
) -> torch.Tensor:
    """
    Places generated cube into final volume.

    Args:
        volume: Final volume tensor to update
        cube: Generated cube to place
        spatial_bounds: Tuple of 3 slice objects for placement
        blend_mode: 'latest' to overwrite, 'cosine' to blend

    Returns:
        Updated volume tensor
    """
    sx, sy, sz = spatial_bounds

    if blend_mode == 'latest':
        # Simply overwrite
        volume[:, sx, sy, sz] = cube
    elif blend_mode == 'cosine':
        # Cosine blending in overlap regions
        existing = volume[:, sx, sy, sz]

        # Compute distance from cube center for blending weights
        cube_shape = cube.shape[1:]  # [dx, dy, dz]
        center = [s // 2 for s in cube_shape]

        # Create coordinate grids
        coords = torch.meshgrid(
            torch.arange(cube_shape[0], device=cube.device),
            torch.arange(cube_shape[1], device=cube.device),
            torch.arange(cube_shape[2], device=cube.device),
            indexing='ij'
        )

        # Compute distances from center
        dist_x = torch.abs(coords[0] - center[0]).float()
        dist_y = torch.abs(coords[1] - center[1]).float()
        dist_z = torch.abs(coords[2] - center[2]).float()

        # Normalize distances by half the cube size
        max_dist = max(cube_shape) / 2.0
        dist_norm = torch.sqrt(dist_x**2 + dist_y**2 + dist_z**2) / max_dist

        # Cosine weighting: w = 0.5 * (1 + cos(π * d)) where d is normalized distance
        # At center (d=0): w=1 (use new value)
        # At edge (d=1): w=0 (use old value)
        weight = 0.5 * (1 + torch.cos(np.pi * dist_norm))
        weight = weight.unsqueeze(0)  # Add channel dimension

        # Blend
        blended = weight * cube + (1 - weight) * existing
        volume[:, sx, sy, sz] = blended
    else:
        raise ValueError(f"Unknown blend_mode: {blend_mode}")

    return volume


def sample_grid_volume(
    flow_module,
    grid_map: list[int],
    base_shape: list[int],
    overlap_size: int,
    y: None | Float[Tensor, "*yshape"] = None,
    guidance: float = 1.0,
    nsteps: int = 30,
    integrate_on_sigma: bool = False,
    noise_injection: bool = False,
    blend_mode: Literal['latest', 'cosine'] = 'latest',
    **kwargs
) -> Float[Tensor, "batch *final_shape"]:
    """
    Generate large volumes by tiling smaller cubes in a grid pattern using inpainting.

    Args:
        flow_module: SIModule instance for generation
        grid_map: [nx, ny, nz] - number of grid steps in each direction
        base_shape: [channels, dx, dy, dz] - base cube shape (e.g., [4, 32, 32, 32])
        overlap_size: int - overlap between cubes (e.g., 16)
        y: Optional condition tensor
        guidance: Guidance scale for conditional generation
        nsteps: Number of integration steps
        integrate_on_sigma: Whether to integrate on sigma
        noise_injection: Whether to inject noise during integration
        blend_mode: How to handle overlaps ('latest' or 'cosine')
        **kwargs: Additional arguments passed to sample() and inpaint()

    Returns:
        Generated volume tensor of shape [1, channels, final_dx, final_dy, final_dz]
    """
    # Compute final volume shape
    final_shape = [
        base_shape[0],
        base_shape[1] * grid_map[0],
        base_shape[2] * grid_map[1],
        base_shape[3] * grid_map[2]
    ]

    # Generate big noise cube of final shape
    device = flow_module.device
    noise_cube = torch.randn(1, *final_shape).to(device)

    # Initialize empty volume tensor
    volume = torch.zeros(1, *final_shape, device=device)

    # Get generation order based on parity patterns
    generation_order, corner_inds_limit = _get_grid_generation_order(grid_map)

    print(generation_order)
    print(f"corner_inds_limit: {corner_inds_limit}")
    # Track generated positions
    generated_positions = set()

    # Generate cubes in order
    for grid_ind, grid_pos in enumerate(generation_order):
        # Compute spatial bounds for this cube
        spatial_bounds = _get_cube_spatial_bounds(grid_pos, base_shape, overlap_size, final_shape)
        sx, sy, sz = spatial_bounds

        # Extract noise slice from big noise cube
        noise_slice = _extract_noise_slice(noise_cube[0], spatial_bounds)  # Remove batch dim for extraction
        noise_slice = noise_slice.unsqueeze(0)  # Add batch dim back

        # Get extended cube shape
        extended_shape = list(noise_slice.shape[1:])
        cube_shape = [base_shape[0]] + extended_shape

        # Check if this is a "corner" (first pattern: all even coordinates)
        is_corner = grid_ind < corner_inds_limit

        if is_corner:
            # Use independent sampling for corners
            generated_cube = flow_module.sample(
                nsamples=1,
                shape=extended_shape,
                y=y,
                guidance=guidance,
                nsteps=nsteps,
                is_latent_shape=True,
                integrate_on_sigma=integrate_on_sigma,
                noise_injection=noise_injection,
                orig_noise=noise_slice,
                return_latents=True,
                **kwargs
            )
        else:
            # Use inpainting for edges/faces/centers
            # Build mask from previously generated cubes
            mask = _build_inpaint_mask(
                grid_pos, generated_positions, base_shape, overlap_size, final_shape
            )
            mask = mask.to(device)

            # Extract known regions from volume into x_orig
            x_orig = volume[0, :, sx, sy, sz].clone()  # Remove batch dim for extraction

            # Use inpainting
            generated_cube = flow_module.inpaint(
                x_orig=x_orig,
                mask=mask,
                nsamples=1,
                y=y,
                guidance=guidance,
                nsteps=nsteps,
                integrate_on_sigma=integrate_on_sigma,
                noise_injection=noise_injection,
                orig_noise=noise_slice,  # Remove batch dim for inpaint
                **kwargs
            )

        # inpaint returns [batch, *shape], remove batch dim if needed
        if generated_cube.dim() == len(cube_shape) + 1 and generated_cube.shape[0] == 1:
            generated_cube = generated_cube[0]

        print('-------------------------------')
        print(grid_ind)
        print(grid_pos)
        print(spatial_bounds)
        print(volume.shape)
        print('--------------------------------')
        # Combine generated cube into volume
        volume = _combine_cube_into_volume(volume[0], generated_cube, spatial_bounds, blend_mode)
        volume = volume.unsqueeze(0)  # Add batch dim back

        # Add position to generated_positions set
        generated_positions.add(grid_pos)

    return volume
