# DiffSci2 — documentation

This repository is the reference implementation for the paper

> *Large-Scale Porous Media Generation Through Field-Controlled Latent
> Diffusion Models* (Naiff, Ramos, Wang, submitted 2026).

LaTeX source of the paper:
`notebooks/exploratory/dfn/tolatex/article/696e4a9178f8165e4a5a4578/main.tex`.

## Getting around

- **[Pipelines](pipelines/field_controlled.md)** — numbered scripts that
  implement each stage of the article.
- **[Checkpoints](checkpoints.md)** — what the published artifacts are
  and how to fetch them.
- **[Data](data.md)** — where the Imperial College four-rock dataset
  comes from and how to place it.
- **[Environment](environment.md)** — conda env, CUDA, dev setup.

## Code organization

Three top-level Python homes, separated by intent:

- `diffsci2/` — the **library**. VAE + diffusion models, networks,
  distributed helpers, GP / chunk-decode / pore-network tooling, metrics.
  Import this, do not edit casually.
- `pipelines/` — **consumer-facing pipelines**. Each sub-directory is a
  numbered, documented pipeline (see its `README.md`).
- `scripts/legacy/` — the **frozen archive** of the scripts that
  produced the paper. Nothing new goes here; nothing is ever deleted
  from here.

Adjacent:

- `tools/` — downloaders and diagnostics.
- `savedmodels/` — checkpoints (gitignored).
- `saveddata/` — raw micro-CT data (gitignored).
- `notebooks/exploratory/` — researcher notebooks (gitignored).

## Running the paper

The minimum to reproduce Table 4 of the paper from a clean clone:

```bash
# 0. Env (see docs/environment.md).
conda activate ddpm_env

# 1. Data (once).
python tools/download_data.py

# 2. Checkpoints (once — uses the Zenodo bundle).
python tools/download_checkpoints.py --set all

# 3. Generate 6 × 1024^3 volumes per rock per variant.
for rock in bentheimer doddington estaillades ketton; do
  for variant in fc257 fc129 uncond; do
    python pipelines/field_controlled/04_generate.py \
        --config pipelines/field_controlled/configs/${rock}.yaml \
        --variant ${variant} \
        --size 1024 --n-samples 6 \
        --output-dir outputs/${rock}_${variant}
  done
done

# 4. Evaluate.
for d in outputs/*; do
  rock=$(basename "$d" | cut -d_ -f1)
  case "$rock" in
    bentheimer)  vox=3.0035e-6;;
    doddington)  vox=2.6929e-6;;
    estaillades) vox=3.31136e-6;;
    ketton)      vox=3.00006e-6;;
  esac
  python pipelines/field_controlled/05_evaluate_metrics.py \
      --input-dir "$d" --voxel-size "$vox"
done
```

The `04_generate_parallel.py` and `04_generate_two_stage.py` variants,
used for the $1024^2 \times 4096$ feasibility demo, are currently
skeletons — run the legacy equivalents for now (they are unchanged
inside `scripts/legacy/`).
