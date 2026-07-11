#!/usr/bin/env python
"""Warped-GP prior for the coarse porosity field, sampled ON THE TORUS.

As in the paper: approximate the w=129 node field by a Gaussian process —
here via FFT spectral synthesis (exact periodic sampling):
  1. gaussianize the rock's node field by quantile transform;
  2. estimate its power spectrum (3D periodogram, radially smoothed);
  3. sample: white noise -> FFT -> sqrt(spectrum) -> iFFT  (periodic GP);
  4. warp back through the empirical marginal (inverse quantile map).
Samples are periodic by construction -> feed the periodized generator
directly (endgame step iii, 2026-07-10).

Usage:
  python scripts/field_gp_prior.py --fit          # fit + validation figure
  python scripts/field_gp_prior.py --sample 8     # write gp fields npy
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

_THIS = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(_THIS), _THIS):
    if p not in sys.path:
        sys.path.insert(0, p)

# Promoted to diffsci2.extra.field_prior (2026-07-11). Paths via env for
# CLI compatibility; library users call fit()/sample_one() after setting
# OUT/FIELD or importing and overriding module globals.
OUT = os.environ.get('GP_OUT', 'runs/gp_prior')
FIELD = os.environ.get('GP_FIELD', 'runs/gen_large/rock_field_w129_s8.npy')
N = int(os.environ.get('GP_N', '124'))    # node grid (124 for 992^3)


def fit():
    os.makedirs(OUT, exist_ok=True)
    f = np.load(FIELD)[:N, :N, :N].astype(np.float64)
    # ---- marginal (empirical quantiles) ----
    vals = np.sort(f.ravel())
    np.save(f'{OUT}/marginal_sorted.npy', vals)
    # gaussianize: rank -> normal scores
    from scipy.stats import norm
    ranks = np.argsort(np.argsort(f.ravel()))
    g = norm.ppf((ranks + 0.5) / ranks.size).reshape(f.shape)
    # ---- spectrum of the gaussianized field ----
    F = np.fft.fftn(g)
    P = (np.abs(F) ** 2) / g.size
    # radial smoothing of the spectrum (isotropize + denoise)
    kx = np.fft.fftfreq(N)
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
    kr = np.sqrt(KX**2 + KY**2 + KZ**2)
    nb = 60
    bins = np.linspace(0, kr.max() + 1e-9, nb + 1)
    which = np.digitize(kr.ravel(), bins) - 1
    prof = np.zeros(nb)
    for b in range(nb):
        m = which == b
        if m.any():
            prof[b] = P.ravel()[m].mean()
    P_iso = prof[np.clip(which, 0, nb - 1)].reshape(P.shape)
    P_iso[0, 0, 0] = 0.0                       # mean handled by marginal
    np.save(f'{OUT}/spectrum_iso.npy', P_iso)
    print('fit done: marginal + isotropized spectrum saved')

    # ---- validation figure: marginal + radial autocorr, real vs 3 samples ----
    samples = [sample_one(s) for s in range(3)]

    def radial_ac(x, nlag=40):
        x = x - x.mean()
        ac = np.fft.ifftn(np.abs(np.fft.fftn(x))**2).real / x.size / x.var()
        prof = [ac[0, 0, 0]]
        for r in range(1, nlag):
            prof.append((ac[r, 0, 0] + ac[0, r, 0] + ac[0, 0, r]) / 3)
        return np.array(prof)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    ax[0].hist(f.ravel(), bins=60, alpha=.6, density=True, label='rock field')
    ax[0].hist(np.concatenate([s.ravel() for s in samples]), bins=60,
               alpha=.6, density=True, label='GP samples')
    ax[0].set_title('marginal'); ax[0].legend()
    ax[1].plot(radial_ac(f), 'k-', lw=2, label='rock field')
    for s in samples:
        ax[1].plot(radial_ac(s), 'C1-', lw=.9, alpha=.8)
    ax[1].set_title('radial autocorrelation (nodes)')
    ax[1].set_xlabel('lag (nodes, 8 vox each)'); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[2].imshow(samples[0][N // 2], cmap='viridis')
    ax[2].set_title('GP field sample (mid slice)')
    fig.tight_layout()
    fig.savefig(f'{OUT}/gp_validation.png', dpi=105)
    print('validation figure ->', f'{OUT}/gp_validation.png')


def sample_one(seed):
    P_iso = np.load(f'{OUT}/spectrum_iso.npy')
    vals = np.load(f'{OUT}/marginal_sorted.npy')
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(P_iso.shape)
    g = np.fft.ifftn(np.fft.fftn(w) * np.sqrt(P_iso)).real
    g = (g - g.mean()) / g.std()
    # warp: normal scores -> empirical marginal
    from scipy.stats import norm
    u = norm.cdf(g)
    idx = np.clip((u * (vals.size - 1)).astype(np.int64), 0, vals.size - 1)
    return vals[idx].astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fit', action='store_true')
    p.add_argument('--sample', type=int, default=0)
    args = p.parse_args()
    if args.fit:
        fit()
    for s in range(args.sample):
        f = sample_one(1000 + s)
        np.save(f'{OUT}/gp_field_{s}.npy', f)
        print(f'gp_field_{s}: mean {f.mean():.4f} std {f.std():.4f}')


if __name__ == '__main__':
    main()
