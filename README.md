# DiffSci2

DiffSci2 is the research codebase behind the DSc thesis **"Large-Scale Porous
Media Reconstruction Through Generative Diffusion Models"** (Danilo Naiff). It
implements a 3D **latent diffusion** pipeline that reconstructs binarized
micro-CT rock volumes, conditions generation on a spatially varying **Gaussian-
process porosity field**, and scales inference (via chunked VAE decode and
spatial/domain parallelism) to volumes far larger than any training subvolume
(1024³ slabs and 1024²×4096 "large" slabs). Generated rocks are validated
physically through pore-network modeling (PNM): porosity, absolute and relative
permeability, capillary pressure, two-point correlation / pore-size
distribution, and an oil-water Buckley-Leverett flow analysis.

The installable Python package is **`diffsci2`** (`setup.py`, `name='diffsci2'`).
The four reference rocks — **Bentheimer** & **Doddington** (sandstones),
**Estaillades** & **Ketton** (carbonates) — are the public **Imperial College
pore-scale modelling micro-CT dataset** (2015 release, 1000³ at
3.0035 / 2.6929 / 3.31136 / 3.00006 µm), available from
[Imperial College's pore-scale modelling group](https://www.imperial.ac.uk/earth-science/research/research-groups/pore-scale-modelling/micro-ct-images-and-networks/).
The helper `scripts/download_imperial_rocks.py` fetches them into
`saveddata/raw/imperial_college/`.

> **Note on this README.** It replaces the old cookiecutter boilerplate (which
> referenced a nonexistent `diffsci` package and `make data`/`make train`
> targets). For the authoritative, verified pipeline study see
> `claude/plan/thesis-reproduction/` and the distilled docs in `docs/`.

---

## DiffSci2 as a general diffusion framework

Although this repository was built for — and validated on — porous rock, the
`diffsci2` package is a **general-purpose diffusion-model library**; the rock
application is one instantiation of a domain-agnostic core. Grounded in the code:

- **Diffusion formulations** — EDM / Karras sigma-space diffusion and a
  stochastic-interpolant / flow-matching core with pluggable interpolants
  (linear, cosine, EDM), plus VAE latent diffusion (`diffsci2/models/karras/`,
  `diffsci2/models/vae/`), all wired as PyTorch-Lightning modules for multi-GPU
  training.
- **Backbones** — a configurable 1D/2D/3D U-Net (`PUNetG`), ADM, a VAE
  encoder/decoder (`VAENet`, incl. PixelNorm / magnitude-preserving variants), a
  diffusion transformer, and a local-attention (NATTEN) backbone with optional
  circular/periodic boundaries for seamless tiling (`diffsci2/nets/`).
- **Samplers & guidance** — Heun / probability-flow ODE integration,
  classifier-free guidance, MCMC posterior samplers (ULA, MALA, BAOAB/HMC), and
  inpainting / sequential-inpainting (DPS-style) conditional generation
  (`diffsci2/models/karras/`, `diffsci2/extra/`).
- **Conditioning** — scalar, vector, field/spatial (FiLM) and function embedders
  with composite multi-modal conditioning (`diffsci2/nets/embedder.py`,
  `enhanced_conditioning.py`) — none of it rock-specific.
- **Generic data** — `VolumeSubvolumeDataset` loads arbitrary 2D/3D arrays with a
  pluggable conditioning `extractor` and discrete-symmetry augmentation; toy
  analytical datasets for algorithm validation (`diffsci2/data/`). The package has
  already been exercised on non-rock data (MNIST; the `0001-drosophila-*` scripts).
- **Scaling** — spatial/domain parallelism (`diffsci2/distributed/`) plus chunked
  VAE decode (`diffsci2/extra/chunk_decode*.py`) let a model trained on small
  subvolumes generate volumes far larger than fit in memory (here 1024³ slabs and
  1024²×4096).

**Domain-specific** (not reusable as-is): the pore-network physics and rock
metrics in `diffsci2/extra/pore/` (SNOW2, PNM permeability, Corey/Brooks-Corey,
Buckley-Leverett — vendored from the former `poregen` dependency), porosity-field
conditioning (`diffsci2/extra/porosity_map.py`, Matérn-GP fields), and the VAE
morphology fine-tune reward in `diffsci2/vaesft/`.

---

## End-to-end pipeline

```
RAW rock volumes (Imperial College: Bentheimer, Doddington, Estaillades, Ketton)
   |  (Imperial College micro-CT dataset → saveddata/raw/imperial_college/*.raw, uint8 1000^3)
   v
GP porosity-field training data  ──  gpdata4-129 (window 129) / gpdata4-257 (window 257)
   |     (per-stone *_porosity_field_full.npy, 1000^3 float64; + *_porosity_analysis.npz Matern params)
   v
[TRAIN]  pipelines/fieldcontrolled/0003-porosity-field-training.py   →  checkpoint
   |        (per stone × {129 crop64, 257 crop128}; frozen VAE = savedmodels/pore/production/converted_vaenet.ckpt)
   v
[GENERATE]  pipelines/fieldcontrolled/0004e-porosity-field-generator.py   (two stages)
   |   Stage 1  --mode latent   (torchrun, multi-GPU, spatial-parallel) → latents/*.latent.pt
   |   Stage 2  --mode decode   (1 GPU, chunked, optional disk-offload)  → data/*.npy (binarized)
   v
synthetic volumes  (per case dir; 1280^3 cubes → crop to 1024^3, or large 1280×1280×4352 slabs)
   |
   ├─[DIVERSITY] pipelines/fieldcontrolled/0010-diversity-calculation.py        → strided per-block porosity/permeability fields
   ├─[METRICS]   pipelines/fieldcontrolled/0005b-…-new-metrics-evaluator.py  → metrics_<stone>_<pattern>_twophase.npz
   ├─[LARGE]     pipelines/fieldcontrolled/0005d-…-large-subvol.py           → metrics_<stone>_largescale_stride1024_*.npz
   ├─[PNM-LARGE] pipelines/fieldcontrolled/0005-…-large-pnm.py               → whole-slab 0.network.npz (consumed by 0011)
   └─[FLOW]      pipelines/fieldcontrolled/0011-oil-water-flow.py               → oilwater_results.npz (Corey + Buckley-Leverett)
   |
   v
[FIGURES]  notebooks/exploratory/dfn/tolatex/scripts/generate_*.py (+ a few notebooks)  →  thesis Figures/
```

**Canonical thesis spine:** `0002 → 0003 → 0004e → {0005, 0005b, 0005d-large-subvol,
0010} → 0005-large-pnm → 0011`. These canonical scripts now live in
**`pipelines/fieldcontrolled/`** (see its `README.md` for the annotated run order);
superseded / dead variants (the `0004`/`0004c`/`0004d` generator lineage, the
drosophila/2D side-experiments, scratch tests, …) were archived to
**`scripts/old/`**. See `docs/SCRIPTS_CATALOG.md` for the full role + status table.

---

## Repository map

| Zone | Tracked | What it is |
|---|---|---|
| `diffsci2/` | yes | The installable library: `nets/` (PUNetG U-Net, VAENet, local-attention NATTEN backbone), `models/karras/` (EDM-sigma-space flow `SIModule`/`EDMModule` diffusion core), `models/vae/` (`VAEModule`), `vaesft/` (VAE supervised fine-tune through a frozen reward), `data/` (subvolume datasets + cube-symmetry aug), `distributed/` (spatial/domain parallelism for PUNetG inference), `extra/` (chunked decode, Matérn GP fields, `extra/pore/` PNM physics). |
| `pipelines/fieldcontrolled/` | yes | **The canonical thesis pipeline** (GP-field fit → training → generation → evaluation → oil-water physics). See its `README.md`. |
| `pipelines/vae/` | yes | Productionized VAE-SFT pipeline (train/eval/smoke). Depends INTO the AI-research playground (see below). |
| `scripts/` | yes | The **unconditional** training branch (`0009`), general utilities (`download_imperial_rocks.py`, diagnostics), and `scripts/old/` — the tracked archive of superseded/dead numbered scripts. See `docs/SCRIPTS_CATALOG.md`. |
| `tests/` | yes | Unit tests + `tests/chunk_decode_encode/` correctness suite. |
| `docs/` | yes | The distilled docs (this `.md` set) alongside a legacy Sphinx stub. |
| `claude/` | no (gitignored) | Agent-facing plans/reports, including `claude/plan/thesis-reproduction/` (the verified source-of-truth for the thesis pipeline). |
| `notebooks/exploratory/dfn/` | no | Thesis analysis notebooks + `tolatex/` figure scripts. **Pre-migration / unmaintained** — see `notebooks/exploratory/dfn/README_STALE.md`. |
| `notebooks/exploratory/dfnai/` | no | "AI-as-researchers" subprojects (poreregressor, vaeporesft, …) — being promoted to a top-level gitignored `aiplayground/` (report 05). |
| `notebooks/exploratory/lysm/` | no | Léo's multi-front workspace (self-documenting `context/` + `docs/`). |
| `saveddata/` | no (gitignored) | Data symlink farm (`raw/imperial_college/` the downloaded rocks; the new canonical `gp_training/`, `generated/`, `metrics/`, `archive/` layout — see `docs/ARTIFACTS_AND_DATA.md`). |
| `savedmodels/` | no (gitignored) | Checkpoint store: `experimental/` (historical dated runs), `pore/production/` (frozen VAEs + per-stone scalar ckpts), `pore/field_controlled/` (production field-controlled ckpts + preserved GP fields). |

---

## Environment

- **Conda env:** `ddpm_env` (Linux / Ubuntu; remote dev via VSCode Remote SSH is typical).
- **Install:** the package is `diffsci2`; install editable from repo root (`pip install -e .`) so `import diffsci2` resolves regardless of CWD.
- **GPUs:** scripts take literal physical GPU indices (no `CUDA_VISIBLE_DEVICES`); e.g. `--devices 1,2,3,4,5,6` binds GPUs 1–6. Multi-GPU generation uses `torchrun` for spatial-parallel latent sampling. **Avoid GPU 7 unless explicitly told otherwise.** Long jobs should run inside `screen`.
- **Raw data:** the Imperial College micro-CT rocks (see top of this README); `scripts/download_imperial_rocks.py` places them under `saveddata/raw/imperial_college/`. NOTE: several legacy scripts still hardcode an old `~/repos/PoreGen/saveddata/raw/imperial_college/` `DATA_DIR` — repoint it to `saveddata/raw/imperial_college/` (a known cleanup item).
- **PNM engine:** pore-network extraction (SNOW2) and PNM permeability are now **vendored into `diffsci2.extra.pore`** (ported from the former `poregen` dependency — see `claude/plan/poregen-port/`), so `poregen` is no longer required on `PYTHONPATH`. **OpenPNM** remains the underlying flow solver.
- **No seeding.** None of the pipeline scripts seed torch/numpy/Lightning, so every run is non-deterministic.

---

## Where to look next

- `docs/SCRIPTS_CATALOG.md` — every numbered script with status (canonical / superseded / dead).
- `docs/REPRODUCE_THESIS.md` — the corrected, runnable train → generate → evaluate → physics runbook.
- `docs/ARTIFACTS_AND_DATA.md` — where checkpoints and data live (`savedmodels/`, `saveddata/`, the `/opt/persistence2/stonedata` symlink farm).
- `claude/plan/thesis-reproduction/00-overview.md` … `06-…` — the seven verified study reports these docs distill (the agent-facing source of truth).
