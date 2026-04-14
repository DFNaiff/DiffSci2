# scripts/legacy/

**Status: frozen archive. Do not edit. Do not delete.**

This directory is the verbatim state of `scripts/` at the time of the paper
*Large-Scale Porous Media Generation Through Field-Controlled Latent
Diffusion Models* (source at
`notebooks/exploratory/dfn/tolatex/article/696e4a9178f8165e4a5a4578/main.tex`).

It is preserved so that:

1. the exact code that produced the published FC-257 and FC-129 checkpoints
   (see `savedmodels/experimental/20260324-…-gpdata4-257-porosity-field/`,
   `20260325-…`, and `20260328-…-gpdata4-129-porosity-field/`) remains
   inspectable in-tree;
2. anyone reading the paper and finding old commits can navigate to the
   exact script referenced by the run log;
3. the new `pipelines/` tree can be re-checked against the behavior the
   legacy code encoded, one stage at a time.

All **new work** goes under `pipelines/`. When a legacy script's behavior
has been fully captured by a new pipeline script, the pipeline script
references its legacy counterpart at the top of the file.

## Mapping to the new layout

| Legacy                                                             | New pipeline                                                         |
|--------------------------------------------------------------------|----------------------------------------------------------------------|
| `0001-drosophila-autoencoder.py`                                    | `pipelines/drosophila_demo/01_train_autoencoder.py`                   |
| `0001-drosophila-training.py`                                       | `pipelines/drosophila_demo/02_train_diffusion.py`                     |
| `0001-drosophila-training-latent.py`                                | `pipelines/drosophila_demo/03_train_latent_diffusion.py`              |
| `0002-porosity-field-estimator.py`                                  | `pipelines/field_controlled/01_fit_gaussian_process.py`               |
| `0002-porosity-field-estimator-copula.py`                           | `pipelines/experiments/copula_field_estimator.py`                     |
| `0003-porosity-field-training.py`                                   | *superseded by 0003c — not ported*                                    |
| `0003b-porosity-field-training-enhanced.py`                         | `pipelines/experiments/enhanced_conditioning.py`                      |
| `0003c-porosity-field-training-vol-correction.py`                   | `pipelines/field_controlled/03_train_field_conditioned.py`            |
| *— (not in legacy; base scalar training lived in the Naiff2026 repo)* | `pipelines/field_controlled/02_train_base_scalar.py` (new)          |
| `0004-porosity-field-generator.py`                                  | *superseded by 0004c — not ported*                                    |
| `0004c-porosity-field-generator.py`                                 | `pipelines/field_controlled/04_generate.py`                           |
| `0004d-porosity-field-generator.py`                                 | `pipelines/field_controlled/04_generate_parallel.py`                  |
| `0004d-porosity-field-generator-TEST.py`                            | *diagnostic only — not ported*                                        |
| `0004e-porosity-field-generator.py`                                 | `pipelines/field_controlled/04_generate_two_stage.py`                 |
| `0004e-porosity-field-generator-unconditional.py`                   | `pipelines/unconditional/02_generate.py`                              |
| `0005-porosity-field-metrics-evaluator.py`                          | *superseded by 0005b — not ported*                                    |
| `0005-porosity-field-new-metrics-evaluator-large-pnm.py`            | *network-extraction helper — folded into `05_evaluate_metrics_large`* |
| `0005b-porosity-field-new-metrics-evaluator.py`                     | `pipelines/field_controlled/05_evaluate_metrics.py`                   |
| `0005c-porosity-field-new-metrics-evaluator-large.py`               | `pipelines/field_controlled/05_evaluate_metrics_large.py`             |
| `0005d-porosity-field-new-metrics-evaluator-large-subvol.py`        | *subvol variant — folded into `05_evaluate_metrics_large`*            |
| `0005d-porosity-field-buckley-leverett.py`                          | `pipelines/experiments/two_phase_flow.py`                             |
| `0006-porosity-field-generator-from-training.py`                    | `pipelines/experiments/field_from_real.py`                            |
| `0007-porosity-field-evaluation.py`                                 | `pipelines/field_controlled/07_field_diagnostics.py`                  |
| `0008-unconditional-2d-multistone-training.py`                      | `pipelines/experiments/multistone_2d.py`                              |
| `0009-unconditional-training-3d.py`                                 | `pipelines/unconditional/01_train_unconditional.py`                   |
| `0009b-unconditional-training-3d-masked.py`                         | `pipelines/unconditional/01b_train_unconditional_masked.py`           |
| `0010-diversity-calculation.py`                                     | `pipelines/field_controlled/06_diversity.py`                          |
| `0011-oil-water-flow.py`                                            | `pipelines/experiments/two_phase_flow.py` (merged with 0005d)         |
| `0012-unfolding.py`                                                 | `pipelines/experiments/unfolding.py`                                  |
| `clean_corrupted_notebook.py`                                       | `tools/clean_corrupted_notebook.py`                                   |
| `diagnose_conditioning.py`                                          | `tools/diagnose_conditioning.py`                                      |
| `test_spatial_parallel.py`                                          | `tools/test_spatial_parallel.py`                                      |

## Hardcoded paths

Every script in this directory assumes:

- `/home/ubuntu/repos/PoreGen/saveddata/raw/imperial_college/` for raw data;
- `notebooks/exploratory/dfn/data/gpdata4-{129,257}/` for GP fits;
- `savedmodels/pore/production/` for base + VAE checkpoints.

The new `pipelines/` scripts resolve these through `DIFFSCI2_DATA_ROOT`,
`DIFFSCI2_MODEL_ROOT`, and YAML configs instead. If you re-run anything
from this directory, those assumptions still apply.
