#!/usr/bin/env python
"""
Porosity Field Estimator for 3D Binary Volumes.

This script calculates the mean porosity field from a 3D binary volume,
computes the logit-transformed two-point correlation function, and fits
a Matérn Gaussian Process model to capture the spatial correlation structure.

The output can be used to generate new porosity field realizations for
generative models.

Usage:
    # Basic usage (saves only analysis data)
    python scripts/0002-porosity-field-estimator.py \
        --data-path /path/to/volume.raw \
        --output-dir ./output

    # Save porosity field as well
    python scripts/0002-porosity-field-estimator.py \
        --data-path /path/to/volume.raw \
        --output-dir ./output \
        --save-field

    # Custom kernel size and GPU
    python scripts/0002-porosity-field-estimator.py \
        --data-path /path/to/volume.raw \
        --output-dir ./output \
        --kernel-size 128 \
        --device cuda:0
"""
import argparse
import os

import numpy as np
import scipy.signal
import torch

from diffsci2.extra import two_point_correlation, matern_gaussian_process


def parse_args():
    parser = argparse.ArgumentParser(
        description='Calculate porosity field and fit Matérn GP parameters'
    )
    parser.add_argument(
        '--data-path', type=str, required=True,
        help='Path to .raw binary volume file'
    )
    parser.add_argument(
        '--volume-shape', type=int, nargs=3, default=[1000, 1000, 1000],
        help='Shape of the volume (D H W)'
    )
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='Directory to save output files'
    )
    parser.add_argument(
        '--output-prefix', type=str, default=None,
        help='Prefix for output files (default: derived from input filename)'
    )
    parser.add_argument(
        '--kernel-size', type=int, default=256,
        help='Size of the averaging kernel for porosity field calculation'
    )
    parser.add_argument(
        '--save-field', action='store_true',
        help='Save the porosity field volume (can be large)'
    )
    parser.add_argument(
        '--device', type=str, default='cuda:0',
        help='Device for TPC calculation (e.g., cuda:0, cpu)'
    )
    parser.add_argument(
        '--tpc-bins', type=int, default=50,
        help='Number of bins for radial TPC calculation'
    )
    parser.add_argument(
        '--voxel-size', type=float, default=1.0,
        help='Physical voxel size for distance scaling'
    )
    return parser.parse_args()


def create_averaging_kernel(kernel_size):
    """Create a 3D box averaging kernel."""
    return np.ones((kernel_size, kernel_size, kernel_size)) / (kernel_size ** 3)


def calculate_porosity_field(data, kernel_size):
    """
    Calculate the local porosity field using FFT convolution.

    Parameters
    ----------
    data : ndarray
        Binary volume (0=pore, 1=solid or vice versa)
    kernel_size : int
        Size of the averaging kernel

    Returns
    -------
    porosity_field : ndarray
        Local mean porosity field
    """
    kernel = create_averaging_kernel(kernel_size)
    # 1 - data converts from solid=1 to pore=1 representation
    porosity_field = scipy.signal.fftconvolve(
        1 - data.astype(np.float32),
        kernel,
        mode='valid'
    )
    return porosity_field


def calculate_logit_correlation(porosity_field, device, bins, voxel_size):
    """
    Calculate the two-point correlation function of the logit-transformed field.

    Parameters
    ----------
    porosity_field : ndarray
        Local porosity field
    device : str
        Device for computation
    bins : int
        Number of radial bins
    voxel_size : float
        Physical voxel size

    Returns
    -------
    dict with keys:
        - r: radial distances
        - correlation: TPC values
        - mean_logit: mean of logit-transformed field
    """
    porosity_tensor = torch.tensor(porosity_field).float()

    # Logit transform
    porosity_logit = torch.logit(porosity_tensor)
    mean_logit = porosity_logit.mean().item()

    # Center the field and compute TPC
    centered_logit = porosity_logit - mean_logit
    tpc = two_point_correlation.tpcf_fft(
        centered_logit.to(device),
        bins=bins,
        voxel_size=voxel_size
    )

    return {
        'r': tpc.r,
        'correlation': tpc.correlation,
        'mean_logit': mean_logit
    }


def fit_matern_to_correlation(r, correlation):
    """
    Fit Matérn kernel parameters to the correlation data.

    Parameters
    ----------
    r : ndarray
        Radial distances
    correlation : ndarray
        Correlation values

    Returns
    -------
    dict with keys:
        - sigma_sq: amplitude parameter
        - nu: smoothness parameter
        - length_scale: length scale parameter
        - popt: raw fitted parameters
        - pcov: covariance matrix of fit
        - fit_success: whether the fit succeeded
    """
    popt, pcov = matern_gaussian_process.fit_matern_parameters(r, correlation)

    if popt is None:
        return {
            'sigma_sq': np.nan,
            'nu': np.nan,
            'length_scale': np.nan,
            'popt': None,
            'pcov': None,
            'fit_success': False
        }

    sigma_sq, nu, length_scale = popt
    return {
        'sigma_sq': sigma_sq,
        'nu': nu,
        'length_scale': length_scale,
        'popt': popt,
        'pcov': pcov,
        'fit_success': True
    }


def main():
    args = parse_args()

    # Determine output prefix
    if args.output_prefix is None:
        # Extract name from input path (e.g., 'Berea_2d25um_binary.raw' -> 'Berea_2d25um_binary')
        basename = os.path.basename(args.data_path)
        output_prefix = os.path.splitext(basename)[0]
    else:
        output_prefix = args.output_prefix

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"Loading volume from {args.data_path}")
    data = np.fromfile(args.data_path, dtype=np.uint8).reshape(args.volume_shape)
    print(f"Volume shape: {data.shape}")

    # Calculate porosity field
    print(f"Calculating porosity field with kernel size {args.kernel_size}...")
    porosity_field = calculate_porosity_field(data, args.kernel_size)
    print(f"Porosity field shape: {porosity_field.shape}")
    print(f"Porosity range: [{porosity_field.min():.4f}, {porosity_field.max():.4f}]")
    print(f"Mean porosity: {porosity_field.mean():.4f}")

    # Optionally save porosity field
    if args.save_field:
        field_path = os.path.join(args.output_dir, f'{output_prefix}_porosity_field.npy')
        print(f"Saving porosity field to {field_path}")
        np.save(field_path, porosity_field)

    # Calculate logit correlation
    print(f"Calculating logit-transformed TPC on {args.device}...")
    corr_data = calculate_logit_correlation(
        porosity_field,
        device=args.device,
        bins=args.tpc_bins,
        voxel_size=args.voxel_size
    )
    print(f"Mean logit: {corr_data['mean_logit']:.4f}")

    # Fit Matérn parameters
    print("Fitting Matérn kernel parameters...")
    matern_fit = fit_matern_to_correlation(
        corr_data['r'],
        corr_data['correlation']
    )

    if matern_fit['fit_success']:
        print(f"Matérn fit successful:")
        print(f"  sigma^2 (amplitude): {matern_fit['sigma_sq']:.4f}")
        print(f"  nu (smoothness):     {matern_fit['nu']:.4f}")
        print(f"  l (length scale):    {matern_fit['length_scale']:.4f}")
    else:
        print("Warning: Matérn fitting failed to converge")

    # Save analysis data to npz
    analysis_path = os.path.join(args.output_dir, f'{output_prefix}_porosity_analysis.npz')
    print(f"Saving analysis data to {analysis_path}")

    np.savez(
        analysis_path,
        # Correlation data
        r=corr_data['r'],
        correlation=corr_data['correlation'],
        mean_logit=corr_data['mean_logit'],
        # Matérn parameters
        matern_sigma_sq=matern_fit['sigma_sq'],
        matern_nu=matern_fit['nu'],
        matern_length_scale=matern_fit['length_scale'],
        matern_fit_success=matern_fit['fit_success'],
        # Metadata
        kernel_size=args.kernel_size,
        volume_shape=args.volume_shape,
        porosity_field_shape=porosity_field.shape,
        voxel_size=args.voxel_size,
        source_file=os.path.basename(args.data_path)
    )

    print("Done!")


if __name__ == '__main__':
    main()
