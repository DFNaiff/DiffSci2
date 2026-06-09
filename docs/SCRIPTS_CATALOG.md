# Scripts Catalog

Every numbered `0001`–`0012` script, with its one-line role, pipeline stage, and
**status**. Distilled from `claude/plan/thesis-reproduction/` reports 01–03 and 06.

> **The numbered prefix is a LINEAGE, not a stable API.** Several numbers have
> multiple superseded variants. Do not run a superseded variant by accident —
> always use the **canonical** script for its stage.

## Locations after the reorg

The mess was physically resolved (not just documented):

- **CANONICAL field-controlled pipeline → `pipelines/fieldcontrolled/`** — the 10
  scripts marked **CANONICAL** for the thesis (field-controlled) line:
  `0002-…-estimator`, `0003-…-training`, `0004e-…-generator`,
  `0005-…-metrics-evaluator`, `0005-…-large-pnm`, `0005b-…-new-metrics-evaluator`,
  `0005b-rerun-drainage`, `0005d-…-large-subvol`, `0010-diversity-calculation`,
  `0011-oil-water-flow`. (See `pipelines/fieldcontrolled/README.md`.)
- **SUPERSEDED / DEAD → `scripts/old/`** — a tracked archive (19 scripts: the
  `0004`/`0004c`/`0004d` lineage, drosophila/2D side-experiments, `0005c`,
  `0005d-buckley-leverett`, `0006`, `0007`, `0008`, `0012-*`, …).
- **Stays in `scripts/`** — the **unconditional** branch (`0009`,
  `0009b`-masked/lysm) and general utilities (`download_imperial_rocks.py`,
  `clean_corrupted_notebook.py`, `diagnose_conditioning.py`,
  `test_spatial_parallel.py`).

The `Script` column below keeps bare filenames; resolve the location via the
**Status** column (CANONICAL → `pipelines/fieldcontrolled/`; SUPERSEDED/DEAD →
`scripts/old/`; the two `0009*` + utilities → `scripts/`).

## Canonical thesis spine

```
0002  (GP porosity-field fit, input prep)
  → 0003  (field-conditioned diffusion TRAINING)
    → 0004e  (two-stage GENERATION: --mode latent | --mode decode)
      → { 0005, 0005b, 0005d-large-subvol, 0010 }   (EVALUATION / metrics)
        → 0005-large-pnm   (whole-slab network extraction)
          → 0011  (oil-water flow + Buckley-Leverett PHYSICS)
```

`0004` / `0004c` / `0004d` are a **superseded generator lineage culminating in
`0004e`**. The `0005` family has four live members plus two superseded ones
(`0005c`, `0005d-buckley-leverett`).

## Status legend

- **CANONICAL** — thesis-canonical; the script to use for its stage.
- **SUPERSEDED-BY-X** — older variant of a canonical script `X`.
- **DEAD / SIDE** — scratch, diagnostic, or unrelated side-experiment.

## Full table

| Script | Role (one line) | Stage | Status |
|---|---|---|---|
| `0001-drosophila-training.py` | Pre-thesis Drosophila 2D training | early (other data) | DEAD / side-experiment (not porous media) |
| `0001-drosophila-training-latent.py` | Drosophila latent-space training | early | DEAD / side-experiment |
| `0001-drosophila-autoencoder.py` | Drosophila autoencoder | early | DEAD / side-experiment |
| `0002-porosity-field-estimator.py` | Fits the GP (Matérn) porosity-field model | GP field fit (input prep) | **CANONICAL** (input prep) |
| `0002-porosity-field-estimator-copula.py` | Copula variant of the GP fit | GP field fit | SUPERSEDED-BY-0002 (copula lineage is dead) |
| `0003-porosity-field-training.py` | Field-conditioned latent-diffusion fine-tune (PUNetG flow, frozen VAE) | TRAINING | **CANONICAL** |
| `0003b-porosity-field-training-enhanced.py` | Enhanced training variant | training | SUPERSEDED-BY-0003 |
| `0003c-porosity-field-training-vol-correction.py` | Volume-correction training variant | training | SUPERSEDED-BY-0003 |
| `0004-porosity-field-generator.py` | Earliest porosity-field generator | generation | SUPERSEDED-BY-0004e |
| `0004c-porosity-field-generator.py` | Predecessor of 0004e (same case logic) | generation | SUPERSEDED-BY-0004e |
| `0004d-porosity-field-generator.py` | Later generator variant | generation | SUPERSEDED-BY-0004e |
| `0004d-porosity-field-generator-TEST.py` | Scratch test of 0004d | generation | DEAD (scratch) |
| `0004e-porosity-field-generator.py` | Two-stage `latent`/`decode` generator (spatial-parallel + chunked decode) | GENERATION | **CANONICAL** |
| `0004e-porosity-field-generator-unconditional.py` | Unconditional fork of 0004e | generation | SUPERSEDED-BY-0004e (variant) |
| `0005-porosity-field-metrics-evaluator.py` | Part I porosity/perm/TPC/PSD KDEs (`Figures/0005-figures/**`) | eval (Part I) | **CANONICAL** (Part I) |
| `0005-porosity-field-new-metrics-evaluator-large-pnm.py` | Extracts whole-large-slab `0.network.npz` (no flow) | eval (network extraction) | **CANONICAL** (producer for 0011) |
| `0005b-porosity-field-new-metrics-evaluator.py` | Main per-cube two-phase evaluator (porosity, K, drainage, kr/Pc) | EVAL | **CANONICAL** |
| `0005b-rerun-drainage.py` | Cheap θ/σ drainage re-sweep on cached networks | eval (utility) | **CANONICAL** (utility) |
| `0005c-porosity-field-new-metrics-evaluator-large.py` | Older 3D-grid large-volume path | eval (large) | SUPERSEDED-BY-0005d-large-subvol |
| `0005d-porosity-field-new-metrics-evaluator-large-subvol.py` | Large-slab z-stride two-phase evaluator | EVAL (large) | **CANONICAL** |
| `0005d-porosity-field-buckley-leverett.py` | Corey + BL from cached network (predecessor to 0011) | physics | SUPERSEDED-BY-0011 *(name collides with the other `0005d-*` — disambiguate by full filename)* |
| `0006-porosity-field-generator-from-training.py` | Generate straight from a training run | generation | SUPERSEDED / utility |
| `0007-porosity-field-evaluation.py` | Older evaluation entry | eval | SUPERSEDED-BY-0005b |
| `0008-unconditional-2d-multistone-training.py` | 2D multistone unconditional training | training (2D) | DEAD / side (not the 3D thesis line) |
| `0009-unconditional-training-3d.py` | Unconditional 3D training | training (uncond) | CANONICAL (unconditional branch) |
| `0009b-unconditional-training-3d-masked.py` | Masked unconditional 3D (lysm) | training | SIDE (lysm subproject) |
| `0010-diversity-calculation.py` | Strided sub-block diversity porosity/permeability fields | EVAL | **CANONICAL** |
| `0011-oil-water-flow.py` | Oil-water flow + REV sweeps + Buckley-Leverett (consumes a cached network) | PHYSICS | **CANONICAL** |
| `0012-unfolding.py` | Unfolding experiment | misc | DEAD / test |
| `0012-volume-testing-1.py` | Scratch volume test | misc | DEAD / test |
| `0012-volume-testing-2.py` | Scratch volume test | misc | DEAD / test |
| `clean_corrupted_notebook.py` | Dev helper (notebook repair) | utility | UTILITY |
| `diagnose_conditioning.py` | Conditioning diagnostic | utility | UTILITY / diagnostic |
| `test_spatial_parallel.py` | Manual distributed/spatial-parallel check | utility | UTILITY / test |

## Notes & gotchas

- **Two `0005d-*` scripts.** `0005d-…-large-subvol` (CANONICAL large-slab
  evaluator) vs `0005d-…-buckley-leverett` (SUPERSEDED by `0011`). Disambiguate
  by full filename.
- **`--stride` is overloaded.** A mode word (`full`/`half`) in `0010`; an integer
  voxel count in `0005d-large-subvol`. Same flag, different meaning.
- **`--voxel-size` units differ.** Micrometers in `0011` (e.g. `3.3116`); meters
  elsewhere (derived from a hardcoded stone table, e.g. `3.31136e-6`).
- **`0009` is canonical only for the *unconditional* branch** (and is the script
  CLAUDE.md historically centers on). The *thesis field-controlled* pipeline is
  `0003`/`0004e`/`0005b`/`0011` — not `0009`.
- Full per-stage CLI surfaces, corrected invocations, and the
  generation-case decoder are in `docs/REPRODUCE_THESIS.md`; the verified source
  studies are `claude/plan/thesis-reproduction/01-…` and `02-…`.
