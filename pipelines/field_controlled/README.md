# pipelines/field_controlled/

The canonical pipeline of the paper *Large-Scale Porous Media Generation
Through Field-Controlled Latent Diffusion Models*. Each numbered script
implements one stage of the method described in the article
(`notebooks/exploratory/dfn/tolatex/article/696e4a9178f8165e4a5a4578/main.tex`).

## Stages

| # | Script                                   | Article reference      | Reads                                         | Writes                                                 |
|---|------------------------------------------|-------------------------|-----------------------------------------------|--------------------------------------------------------|
| 1 | `01_fit_gaussian_process.py`             | §3.3 Field Learning     | raw binary volume                              | `*_porosity_field.npy` + `*_porosity_analysis.npz`      |
| 2 | `02_train_base_scalar.py`                | §3.2 Alg. 1 (base)      | raw volume, VAE checkpoint                     | scalar-conditioned LDM checkpoint (`<rock>_pcond.ckpt`)|
| 3 | `03_train_field_conditioned.py`          | §3.2 Alg. 2 (fine-tune) | raw volume, porosity field, base LDM          | FC-257 or FC-129 checkpoint                             |
| 4 | `04_generate.py` (single GPU)            | §3.4 Inference Alg. 3   | FC checkpoint, VAE, GP analysis               | generated `.npy` volumes                                |
| 4 | `04_generate_parallel.py` (multi-GPU)    | §3.4 Inference          | same                                          | generated volumes at $>1024^3$                          |
| 4 | `04_generate_two_stage.py` (non-cubic)   | §3.4 (feasibility demo) | same                                          | $1024^2 \times 4096$ volumes                            |
| 5 | `05_evaluate_metrics.py`                 | §4.2 Full-Scale         | generated `.npy`                              | metrics JSON + `.network.npz`                           |
| 5 | `05_evaluate_metrics_large.py`           | §4.2 large-scale demo   | generated `.npy` at $1024^2 \times 4096$      | subvolume metrics JSON                                  |
| 6 | `06_diversity.py`                        | §4.1 Sub-Block Div.     | generated `.npy`                              | sub-block porosity/permeability grids                   |
| 7 | `07_field_diagnostics.py`                | §4.3 Mechanism Diag.    | generated `.npy`                              | generated porosity fields + field-correlation metrics   |
| 8 | `08_two_phase_flow.py`                   | §4.2 $k_r(S_w)$, $P_c$  | `.network.npz`                                | relative permeability + capillary pressure curves       |

Experimental extensions (Buckley-Leverett, Corey fits, oil-water displacement
reservoir simulation, multi-stone ergodicity study, etc.) live under
`pipelines/experiments/`.

## Configs

Each of the four rocks has a YAML config under `configs/`:

- `configs/bentheimer.yaml`
- `configs/doddington.yaml`
- `configs/estaillades.yaml`
- `configs/ketton.yaml`

The hyperparameters in those configs reflect **what actually shipped for
the paper** (see §0 of `claude/report/CODE_ORGANIZATION.md` for the dated
checkpoint folders). If you change a config and re-run stage 3, you will
produce a new, non-paper-equivalent checkpoint; that is fine — just save
the new run somewhere other than the published names.

Global knobs come from environment:

- `DIFFSCI2_DATA_ROOT` — where to find raw `.raw` micro-CT volumes and
  their `.mhd` sidecars. Default `./saveddata/raw/imperial_college`.
- `DIFFSCI2_MODEL_ROOT` — where to find published checkpoints. Default
  `./savedmodels`. The pipeline reads the base VAE from
  `${DIFFSCI2_MODEL_ROOT}/pore/production/converted_vaenet.ckpt` and
  published FC checkpoints from
  `${DIFFSCI2_MODEL_ROOT}/pore/field_controlled/fc{257,129}/<rock>.ckpt`.
- `DIFFSCI2_FIELDS_ROOT` — where GP fits live. Default
  `./notebooks/exploratory/dfn/data`. Each stage 1 run writes a
  subdirectory `gpdata4-{kernel_size}/<rock>/` under this root.

## Running the whole pipeline end-to-end

```bash
# 0) Fetch data and checkpoints (once)
python tools/download_data.py
python tools/download_checkpoints.py --set all

# 1) GP fit (skip if using pre-shipped analyses from the checkpoint bundle)
python pipelines/field_controlled/01_fit_gaussian_process.py \
    --config pipelines/field_controlled/configs/bentheimer.yaml

# 2) Base scalar-conditioned training (skip if using published pcond.ckpt)
python pipelines/field_controlled/02_train_base_scalar.py \
    --config pipelines/field_controlled/configs/bentheimer.yaml \
    --devices 0,1,2,3

# 3) Field-conditioned fine-tune
python pipelines/field_controlled/03_train_field_conditioned.py \
    --config pipelines/field_controlled/configs/bentheimer.yaml \
    --variant fc257 \
    --devices 0,1,2,3

# 4) Generate volumes (single GPU)
python pipelines/field_controlled/04_generate.py \
    --config pipelines/field_controlled/configs/bentheimer.yaml \
    --variant fc129 \
    --size 1024 --n-samples 6 \
    --output-dir ./outputs/bentheimer_fc129

# 5) Evaluate
python pipelines/field_controlled/05_evaluate_metrics.py \
    --input-dir ./outputs/bentheimer_fc129 \
    --voxel-size 3.0035e-6
```

## Relationship to legacy

Each file in this directory lists its legacy counterpart at the top,
so the diff between "what shipped" and "what runs now" is always
inspectable. See `scripts/legacy/README.md` for the full mapping.
