#!/usr/bin/env python
"""
New Metrics evaluator for generated porous media volumes.

Computes porosity and two-phase flow properties (absolute/relative permeability,
Pc-Sw curves) using pore network modeling via OpenPNM.

Key differences from 0005-porosity-field-metrics-evaluator.py:
- Uses PoreNetworkPermeability class for full two-phase flow analysis
- Saves porespy network (before openpnm conversion) alongside each volume
- Computes relative permeabilities and capillary pressure curves

Usage:
    # Simple usage with stone name (uses default pattern _pfield_gen_3)
    python 0005b-porosity-field-new-metrics-evaluator.py --stone Estaillades --volume-sizes 1280

    # Different generation pattern (e.g., _pfield_gen_2)
    python 0005b-porosity-field-new-metrics-evaluator.py --stone Estaillades --pattern _pfield_gen_2 --volume-sizes 1280

    # With border cropping (crops 128 voxels from each side)
    python 0005b-porosity-field-new-metrics-evaluator.py --stone Estaillades --volume-sizes 1280 --border-crop 128

    # Custom paths (overrides --stone and --pattern)
    python 0005b-porosity-field-new-metrics-evaluator.py \
        --generated-dir ./generated_data/data/ \
        --output metrics_results.npz \
        --voxel-length 3.3116e-6
"""

import argparse
import os
import time

import numpy as np

from poregen.features.snow2 import snow2
from diffsci2.extra.pore.permeability_from_pnm import PoreNetworkPermeability


# Voxel lengths for different stones (in meters)
VOXEL_LENGTHS = {
    'Bentheimer': 3.0035e-6,
    'Doddington': 2.6929e-6,
    'Estaillades': 3.31136e-6,
    'Ketton': 3.00006e-6,
}

# Reference volume paths
DATA_DIR = '/home/ubuntu/repos/PoreGen/saveddata/raw/imperial_college/'
REFERENCE_PATHS = {
    'Bentheimer': DATA_DIR + 'Bentheimer_1000c_3p0035um.raw',
    'Doddington': DATA_DIR + 'Doddington_1000c_2p6929um.raw',
    'Estaillades': DATA_DIR + 'Estaillades_1000c_3p31136um.raw',
    'Ketton': DATA_DIR + 'Ketton_1000c_3p00006um.raw',
}

# Generated data directories
GENERATED_DATA_DIR = '/home/ubuntu/repos/DiffSci2/notebooks/exploratory/dfn/data/'
DEFAULT_PATTERN = '_pfield_gen_3'

AVAILABLE_STONES = list(VOXEL_LENGTHS.keys())


def get_generated_path(stone, pattern=DEFAULT_PATTERN):
    """Construct generated data path from stone name and pattern."""
    return GENERATED_DATA_DIR + stone.lower() + pattern + '/data'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate two-phase flow metrics for generated porous media volumes'
    )
    parser.add_argument(
        '--stone', type=str, default=None, choices=AVAILABLE_STONES,
        help='Stone type (sets voxel length and generated dir automatically)'
    )
    parser.add_argument(
        '--pattern', type=str, default=DEFAULT_PATTERN,
        help=f'Generation pattern suffix (default: {DEFAULT_PATTERN}). '
             'E.g., "_pfield_gen_2" will look in estaillades_pfield_gen_2/data'
    )
    parser.add_argument(
        '--generated-dir', type=str, default=None,
        help='Directory containing generated volumes (overrides --stone and --pattern)'
    )
    parser.add_argument(
        '--reference-path', type=str, default=None,
        help='Path to reference .raw volume file (overrides --stone default)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output .npz file path for results (default: metrics_{stone}_twophase.npz)'
    )
    parser.add_argument(
        '--voxel-length', type=float, default=None,
        help='Voxel length in meters (overrides --stone)'
    )
    parser.add_argument(
        '--reference-shape', type=int, nargs=3, default=[1000, 1000, 1000],
        help='Shape of reference volume (default: 1000 1000 1000)'
    )
    parser.add_argument(
        '--border-crop', type=int, default=128,
        help='Crop this many voxels from each border before analysis (default: 128)'
    )
    parser.add_argument(
        '--volume-sizes', type=str, default='1280',
        help='Comma-separated list of volume sizes to process (default: 1280)'
    )
    parser.add_argument(
        '--skip-reference', action='store_true',
        help='Skip processing reference volume'
    )
    parser.add_argument(
        '--contact-angle', type=float, default=140.0,
        help='Contact angle for drainage simulation [degrees] (default: 140)'
    )
    parser.add_argument(
        '--surface-tension', type=float, default=0.48,
        help='Surface tension for drainage simulation [N/m] (default: 0.48)'
    )
    parser.add_argument(
        '--max-volumes', type=int, default=None,
        help='Maximum number of volumes to process per size (default: all)'
    )
    return parser.parse_args()


def get_generated_volume_paths(data_dir, size):
    """Get paths to all generated volumes of a given size."""
    files = os.listdir(data_dir)
    paths = []
    for f in sorted(files):
        if f.startswith(f'{size}_') and f.endswith('.npy') and f.count('.') == 1:
            paths.append(os.path.join(data_dir, f))
    return paths


def center_crop(volume, border):
    """Crop border voxels from all sides of the volume."""
    if border <= 0:
        return volume
    return volume[border:-border, border:-border, border:-border]


def extract_porespy_network(binary_volume):
    """
    Extract pore network from binary volume using SNOW algorithm.

    Returns the raw porespy network dictionary (before openpnm conversion).
    """
    # SNOW expects pore space = 1, solid = 0
    # Our convention is pore space = 0, solid = 1
    pore_space = 1 - binary_volume

    partitioning = snow2(pore_space, voxel_size=1.0)
    return partitioning.network


def compute_two_phase_metrics(binary_volume, porespy_network, voxel_size, contact_angle, surface_tension):
    """
    Compute porosity and two-phase flow properties from a binary volume
    and its already-extracted porespy network.

    Args:
        binary_volume: 3D binary array (0=pore, 1=solid)
        porespy_network: Pre-extracted porespy network dict (from SNOW2)
        voxel_size: Physical voxel size in meters
        contact_angle: Contact angle for drainage [degrees]
        surface_tension: Surface tension for drainage [N/m]

    Returns:
        dict with porosity, absolute permeability, relative permeabilities, Sw, Pc
    """
    # Porosity: fraction of void space (pore space = 0, solid = 1)
    porosity = (1 - binary_volume).mean()

    # Create pore network wrapper from the already-extracted network
    pn_wrapper = PoreNetworkPermeability.from_porespy_network(
        porespy_network,
        volume_length=binary_volume.shape[0],
        voxel_size=voxel_size,
    )

    # Calculate absolute permeability
    abs_perm = pn_wrapper.calculate_absolute_permeability()

    # Run drainage simulation
    _ = pn_wrapper.run_drainage_simulation(
        contact_angle=contact_angle,
        surface_tension=surface_tension,
    )

    # Calculate relative permeability curves
    rel_perm = pn_wrapper.calculate_relative_permeability_curves()

    return {
        'porosity': porosity,
        # Absolute permeability
        'K_abs_x': abs_perm.K_x,
        'K_abs_y': abs_perm.K_y,
        'K_abs_z': abs_perm.K_z,
        'K_abs_mean': abs_perm.K_mean,
        'K_abs_x_physical': abs_perm.K_x_physical,
        'K_abs_y_physical': abs_perm.K_y_physical,
        'K_abs_z_physical': abs_perm.K_z_physical,
        'K_abs_mean_physical': abs_perm.K_mean_physical,
        # Saturation and capillary pressure
        'Sw': rel_perm.Sw,
        'Snw': rel_perm.Snwp,
        'Pc': rel_perm.Pc,
        # Relative permeabilities (shape: n_saturations x 3 for x,y,z)
        'kr_wetting': rel_perm.kr_wetting,
        'kr_nonwetting': rel_perm.kr_nonwetting,
        'kr_wetting_mean': rel_perm.kr_wetting_mean,
        'kr_nonwetting_mean': rel_perm.kr_nonwetting_mean,
    }


def save_network(network_dict, save_path):
    """Save porespy network dictionary to npz file."""
    # Convert network dict to saveable format
    save_dict = {}
    for key, value in network_dict.items():
        if isinstance(value, np.ndarray):
            save_dict[key] = value
        elif isinstance(value, (int, float, str, bool)):
            save_dict[key] = np.array(value)
        elif isinstance(value, list):
            save_dict[key] = np.array(value)
        else:
            # Skip unsaveable types
            print(f"  Warning: skipping network key '{key}' of type {type(value)}")

    np.savez(save_path, **save_dict)


def main():
    args = parse_args()

    # Resolve paths from stone if not explicitly given
    if args.stone is not None:
        stone = args.stone
        generated_dir = args.generated_dir or get_generated_path(stone, args.pattern)
        reference_path = args.reference_path or REFERENCE_PATHS[stone]
        voxel_length = args.voxel_length or VOXEL_LENGTHS[stone]
        # Include pattern in output filename for clarity
        pattern_suffix = args.pattern.replace('_', '-').strip('-')
        output_path = args.output or f'metrics_{stone.lower()}_{pattern_suffix}_twophase.npz'
    else:
        # Require explicit paths if no stone specified
        if args.generated_dir is None:
            raise ValueError("Must specify --stone or --generated-dir")
        if args.voxel_length is None:
            raise ValueError("Must specify --stone or --voxel-length")
        stone = None
        generated_dir = args.generated_dir
        reference_path = args.reference_path
        voxel_length = args.voxel_length
        output_path = args.output or 'metrics_twophase.npz'

    print(f"Stone: {stone or 'custom'}")
    print(f"Pattern: {args.pattern}")
    print(f"Voxel length: {voxel_length:.6e} m")
    print(f"Generated dir: {generated_dir}")
    print(f"Reference path: {reference_path}")
    print(f"Output: {output_path}")
    print(f"Border crop: {args.border_crop}")
    print(f"Contact angle: {args.contact_angle} deg")
    print(f"Surface tension: {args.surface_tension} N/m")

    # Timing
    timing = {}

    # Results storage
    results = {
        'stone': stone or 'custom',
        'pattern': args.pattern,
        'voxel_length': voxel_length,
        'border_crop': args.border_crop,
        'contact_angle': args.contact_angle,
        'surface_tension': args.surface_tension,
    }

    # Process reference data if available
    if not args.skip_reference and reference_path is not None and os.path.exists(reference_path):
        print(f"\nProcessing reference data from {reference_path}...")
        t_start = time.time()

        reference_data = np.fromfile(reference_path, dtype=np.uint8).reshape(args.reference_shape)
        print(f"  Reference shape: {reference_data.shape}")
        # NOTE: No border crop for reference - it's already the correct full volume.
        # Border crop is only for generated volumes (to remove boundary artifacts).

        # Extract and save network
        print(f"  Extracting pore network...")
        network = extract_porespy_network(reference_data)
        network_path = reference_path.replace('.raw', '.network.npz')
        save_network(network, network_path)
        print(f"  Saved network to {network_path}")

        # Compute metrics (reusing the already-extracted network)
        print(f"  Computing two-phase flow metrics...")
        ref_metrics = compute_two_phase_metrics(
            reference_data, network, voxel_length,
            args.contact_angle, args.surface_tension
        )

        timing['reference'] = time.time() - t_start
        print(f"  Reference metrics computed in {timing['reference']:.2f}s")
        print(f"    Porosity: {ref_metrics['porosity']:.4f}")
        print(f"    K_abs (mean): {ref_metrics['K_abs_mean_physical'] * 1e15:.2f} mD")

        # Store reference results
        for key, value in ref_metrics.items():
            results[f'reference_{key}'] = value

    # Parse volume sizes
    volume_sizes = [int(x.strip()) for x in args.volume_sizes.split(',')]

    # Process generated data for each size
    for size in volume_sizes:
        print(f"\nProcessing generated {size}^3 volumes...")
        t_start = time.time()

        volume_paths = get_generated_volume_paths(generated_dir, size)
        if len(volume_paths) == 0:
            print(f"  No {size}^3 volumes found, skipping")
            continue

        if args.max_volumes is not None:
            volume_paths = volume_paths[:args.max_volumes]

        print(f"  Found {len(volume_paths)} volumes")

        # Compute effective size after crop
        effective_size = size - 2 * args.border_crop if args.border_crop > 0 else size
        if args.border_crop > 0:
            print(f"  Will apply border crop of {args.border_crop} -> {effective_size}^3")

        # Process each volume
        all_metrics = []

        for vol_idx, vol_path in enumerate(volume_paths):
            vol_name = os.path.basename(vol_path)
            print(f"  [{vol_idx + 1}/{len(volume_paths)}] {vol_name}")

            # Load volume
            volume = np.load(vol_path)

            # Apply border crop if specified
            if args.border_crop > 0:
                volume = center_crop(volume, args.border_crop)

            # Extract and save network (before openpnm conversion)
            network = extract_porespy_network(volume)
            network_path = vol_path.replace('.npy', '.network.npz')
            save_network(network, network_path)
            print(f"    Saved network to {os.path.basename(network_path)}")

            # Compute two-phase flow metrics (reusing the already-extracted network)
            try:
                metrics = compute_two_phase_metrics(
                    volume, network, voxel_length,
                    args.contact_angle, args.surface_tension
                )
                all_metrics.append(metrics)
                print(f"    Porosity: {metrics['porosity']:.4f}, "
                      f"K_abs: {metrics['K_abs_mean_physical'] * 1e15:.2f} mD")
            except Exception as e:
                print(f"    Error computing metrics: {e}")
                continue

            # Free memory
            del volume

        timing[f'generated_{size}'] = time.time() - t_start
        print(f"  Metrics computed in {timing[f'generated_{size}']:.2f}s")

        if len(all_metrics) == 0:
            print(f"  No valid metrics computed for size {size}")
            continue

        # Aggregate results
        results[f'generated_{size}_n_volumes'] = len(all_metrics)
        results[f'generated_{size}_porosity'] = np.array([m['porosity'] for m in all_metrics])
        results[f'generated_{size}_K_abs_mean'] = np.array([m['K_abs_mean'] for m in all_metrics])
        results[f'generated_{size}_K_abs_mean_physical'] = np.array([m['K_abs_mean_physical'] for m in all_metrics])
        results[f'generated_{size}_K_abs_x'] = np.array([m['K_abs_x'] for m in all_metrics])
        results[f'generated_{size}_K_abs_y'] = np.array([m['K_abs_y'] for m in all_metrics])
        results[f'generated_{size}_K_abs_z'] = np.array([m['K_abs_z'] for m in all_metrics])

        # Store Sw, Pc, and relative permeabilities for each volume
        # These are arrays of varying lengths, so we store as object arrays
        results[f'generated_{size}_Sw'] = np.array([m['Sw'] for m in all_metrics], dtype=object)
        results[f'generated_{size}_Snw'] = np.array([m['Snw'] for m in all_metrics], dtype=object)
        results[f'generated_{size}_Pc'] = np.array([m['Pc'] for m in all_metrics], dtype=object)
        results[f'generated_{size}_kr_wetting'] = np.array([m['kr_wetting'] for m in all_metrics], dtype=object)
        results[f'generated_{size}_kr_nonwetting'] = np.array([m['kr_nonwetting'] for m in all_metrics], dtype=object)
        results[f'generated_{size}_kr_wetting_mean'] = np.array([m['kr_wetting_mean'] for m in all_metrics], dtype=object)
        results[f'generated_{size}_kr_nonwetting_mean'] = np.array([m['kr_nonwetting_mean'] for m in all_metrics], dtype=object)

        # Print summary statistics
        porosities = results[f'generated_{size}_porosity']
        perms = results[f'generated_{size}_K_abs_mean_physical'] * 1e15  # Convert to mD
        print(f"  Summary:")
        print(f"    Porosity: mean={porosities.mean():.4f}, std={porosities.std():.4f}")
        print(f"    K_abs (mD): mean={perms.mean():.2f}, std={perms.std():.2f}")

    # Add timing and metadata
    results['timing'] = timing
    results['total_time'] = sum(timing.values())

    # Save results
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True) if os.path.dirname(output_path) else None
    print(f"\nSaving results to {output_path}...")

    # Prepare save dict (handle timing dict)
    save_dict = {}
    for k, v in results.items():
        if k == 'timing':
            save_dict['timing_keys'] = list(timing.keys())
            save_dict['timing_values'] = list(timing.values())
        elif v is not None:
            save_dict[k] = v

    np.savez(output_path, **save_dict)

    print(f"\n=== Summary ===")
    print(f"Total processing time: {results['total_time']:.2f}s")
    print(f"Results saved to: {output_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()
