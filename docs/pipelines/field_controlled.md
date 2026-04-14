# Pipeline — field-controlled latent diffusion

A short runbook on top of the paper. Everything here is derived from
`main.tex` §2–§4; see the article for the scientific rationale.

## Stages

```
raw binary micro-CT (4 rocks)
   │
   ▼
 01_fit_gaussian_process.py                  ← porosity field (Eq. 1)
   │                                         ← warped GP fit (Matérn 3/2)
   ▼
 02_train_base_scalar.py      (Alg. 1)       ← <rock>_pcond.ckpt
   │                                           (scalar-conditioned LDM)
   ▼
 03_train_field_conditioned.py (Alg. 2)      ← fc257 or fc129 checkpoint
   │
   ▼
 04_generate.py              (Alg. 3)        ← generated binary volumes (.npy)
   │ (+ 04_generate_parallel / _two_stage)
   ▼
 05_evaluate_metrics.py                      ← porosity, K_abs, k_r, P_c
   │
   ├── 06_diversity.py                       ← sub-block Hellinger
   ├── 07_field_diagnostics.py               ← §4.3 mechanism
   └── 08_two_phase_flow.py                  ← §4.2 aggregation
```

## Variants

`--variant` selects among the three trained models per rock:

| Variant  | Conditioning                | Checkpoint                                                     |
|----------|-----------------------------|----------------------------------------------------------------|
| `fc257`  | porosity field, w=257       | `savedmodels/pore/field_controlled/fc257/<rock>.ckpt` (symlink)|
| `fc129`  | porosity field, w=129       | `savedmodels/pore/field_controlled/fc129/<rock>.ckpt` (symlink)|
| `uncond` | null (empty) conditioning   | `savedmodels/pore/production/<rock>_pcond.ckpt`                 |

The `Controlled` baseline (scalar conditioning sampled from the
empirical distribution) is not a canonical stage in the new pipeline —
the article shows it fails at scale. It is available via the legacy
scripts if you need to reproduce that specific baseline.

## Configs

Each rock has a YAML in `pipelines/field_controlled/configs/`. These
define everything that is rock-specific (voxel size, filename) and
pipeline-wide hyperparameters (subvolume size, optimizer settings, ODE
steps).

The hyperparameters in these YAMLs were transcribed from
`main.tex` Algorithms 1 & 2 and from the argparse defaults of the
shipping scripts. The shipping runs' `logs/version_0/hparams.yaml` are
empty (Lightning did not auto-serialize hparams), so if you need
bit-perfect parity you should confirm against the exact command line
that Danilo used for each of the four FC-257 and four FC-129 runs.

## Environment variables

Three knobs let a fresh clone find data/checkpoints wherever they live:

- `DIFFSCI2_DATA_ROOT`   — raw micro-CT. Default `<repo>/saveddata/raw/imperial_college`.
- `DIFFSCI2_MODEL_ROOT`  — checkpoints. Default `<repo>/savedmodels`.
- `DIFFSCI2_FIELDS_ROOT` — GP fits / field npys. Default `<repo>/notebooks/exploratory/dfn/data`.

## Relationship to `scripts/legacy/`

Every pipeline file in `pipelines/field_controlled/` lists its legacy
counterpart at the top. The legacy directory is frozen and never edited.
If you are unsure whether a new pipeline file matches the behavior of
the run that produced the paper, the legacy script is the authority.

## Known skeletons (not yet fully ported)

These currently raise `NotImplementedError` with a pointer back to the
legacy script — run the legacy version until someone ports:

- `04_generate_parallel.py`
- `04_generate_two_stage.py`
- `05_evaluate_metrics_large.py`
- `06_diversity.py`
- `07_field_diagnostics.py`
- `pipelines/unconditional/*` (use `04_generate.py --variant uncond`
  instead; the uncond-training skeletons are for new runs only)
