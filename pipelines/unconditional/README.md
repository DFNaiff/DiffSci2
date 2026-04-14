# pipelines/unconditional/

Baseline study: unconditional 3D latent diffusion on single-rock subvolumes.

Used in main.tex as the Uncond baseline; separate from the `field_controlled`
pipeline because those experiments also include architectural variants
(masked crop loss, border-periodic wiring) that would clutter the main flow.

## Stages

| # | Script                                 | Legacy counterpart                                          |
|---|----------------------------------------|-------------------------------------------------------------|
| 1 | `01_train_unconditional.py`            | `scripts/legacy/0009-unconditional-training-3d.py`          |
|1b | `01b_train_unconditional_masked.py`    | `scripts/legacy/0009b-unconditional-training-3d-masked.py`  |
| 2 | `02_generate.py`                        | `scripts/legacy/0004e-porosity-field-generator-unconditional.py` |

Both files in this pipeline are **skeletons** — the paper's Uncond baseline
uses the base scalar checkpoint with null conditioning, which is already
covered by `pipelines/field_controlled/04_generate.py --variant uncond`.
These scripts exist for the from-scratch unconditional training and border-
masked variants that were explored but not reported in the main table.
