# Reproduce the Thesis — Runbook

The corrected, runnable train → generate → evaluate → physics runbook for
*"Large-Scale Porous Media Reconstruction Through Generative Diffusion Models"*.
Distilled from `claude/plan/thesis-reproduction/` reports 01–03, with the
reproducibility corrections **baked in**. Run all commands from the repo root
`/opt/persistence/repos/DiffSci2` unless noted.

## Corrections you must know (read first)

1. **`models/` is a typo for `savedmodels/`.** Checkpoints live under
   `savedmodels/`, never a top-level `models/` (which does not exist). The
   training script even *prints* the broken `models/...` default path — ignore it.
2. **Checkpoint naming (historical experimental store):**
   `savedmodels/experimental/<YYYYMMDD>-dfn-<stone>-<datasource>-porosity-field/checkpoints/last.ckpt`.
   The date prefix is the *training* date and is **not derivable from the CLI** —
   look it up. (e.g. `gpdata4-129` → `20260328`, `gpdata4-257` → `20260325`.)
3. **Production field-controlled checkpoints** also live (cleaner, no date) under
   `savedmodels/pore/field_controlled/{fc129,fc257}/<stone>.ckpt`. Use these as
   the canonical, stable field-controlled checkpoints; the dated `experimental/`
   dirs are the historical originals.
4. **No seeding anywhere** — subvolume sampling, cube-symmetry augmentation, the
   GP porosity field, and diffusion noise are all unseeded. **Every run is
   non-deterministic.** Re-running will not reproduce a specific volume bit-for-bit.
5. **`--generation-case N` is NOT the `_case_N` directory label.** The dir suffix
   is a free-form manual label. Real mapping (see the table below).
6. **`0005b` writes its `.npz` to the current working directory** when `--output`
   is omitted (the thesis runs were launched from `scripts/`, so the npz landed
   there and was manually copied into a `metrics/` dir). **Always pass `--output`.**
7. **The paper keeps generation CASES 9, 10, 2, 3 only** (the `_case_9`,
   `_case_10`, `_3_case_2`, `_3_case_3` dirs). Cases 4 and 6 are excluded from the
   final paper.

### Generation-case decoder (`--generation-case` ↔ dir label ↔ regime)

| `--generation-case` | Conditioning regime | Checkpoint | GP source | Dir label (manual) | In paper? |
|---|---|---|---|---|---|
| 2 | None (y = None) | `<stone>_pcond.ckpt` (scalar) | — | `..._pfield_gen_3_case_2` | **yes** |
| 3 | Scalar porosity (random from real vol) | `<stone>_pcond.ckpt` (scalar) | gpdata2 real volume | `..._pfield_gen_3_case_3` | **yes** |
| 5 | Field porosity, **129-trained** field model | `--checkpoint` (129) | gpdata4-129 | `..._pfield_gen_case_10[_large/_xlarge]` | **yes** |
| 8 | Field porosity, **257-trained** field model | `--checkpoint` (257) | gpdata4-257 | `..._pfield_gen_case_9` | **yes** |
| 1 | Field porosity, post-trained | `--checkpoint` (field) | gpdata3c | — | no |
| 4 | Field porosity, original scalar model | `<stone>_pcond.ckpt` | gpdata3c | — | excluded |
| 6 | Field porosity, original model | `<stone>_pcond.ckpt` | gpdata4-129 | — | excluded |
| 7 | Unconditional | `--checkpoint` | — | — | no |

> Note the non-obvious mapping: dir `_gen_case_10` ⇒ `--generation-case 5`;
> dir `_gen_case_9` ⇒ `--generation-case 8`. The directory suffix is *not* the
> case number.

---

## Stage 0 — GP porosity-field fit (`0002`)

`pipelines/fieldcontrolled/0002-porosity-field-estimator.py` fits the Matérn GP porosity-field
model, producing the per-stone `*_porosity_analysis.npz` (Matérn params:
`mean_logit`, `matern_sigma_sq`, `matern_nu`, `matern_length_scale`) and the
`*_porosity_field_full.npy` (1000³ float64) used downstream. The thesis uses two
windows: **gpdata4-129** (window 129) and **gpdata4-257** (window 257). For
reproduction these GP fields are preserved under
`savedmodels/pore/field_controlled/{gpdata4-129,gpdata4-257}/<stone>/`.

---

## Stage 1 — TRAINING (`0003`)

Fine-tunes the per-stone scalar-porosity PUNetG flow checkpoint into a model
conditioned on a local 3D porosity field, in the frozen VAE latent space
(`savedmodels/pore/production/converted_vaenet.ckpt`).

**Training matrix** — per stone (Bentheimer / Doddington / Estaillades / Ketton)
× two GP datasources:

| Datasource | `--center-crop` | `--train-split` | Validation | Cosine decay |
|---|---|---|---|---|
| `gpdata4-129` | 64 | 550 | `--no-validation` | `--cosine-decay-epochs 5` |
| `gpdata4-257` | 128 | 480 | with validation | — |

**Variant A — `gpdata4-257`, crop 128, with validation:**
```bash
cd /opt/persistence/repos/DiffSci2
python pipelines/fieldcontrolled/0003-porosity-field-training.py \
    --stone Estaillades \
    --data-source gpdata4-257 \
    --center-crop 128 \
    --max-epochs 20 \
    --devices 1,2,3,4,5,6 \
    --warmup-epochs 2 \
    --train-split 480 \
    --ema-decay 0.99
# inherits: --batch-size 1, --accumulate-grad-batches 8, --subvolume-size 256,
#           --downsample-factor 8, --lr 2e-5
# Output (REAL): savedmodels/experimental/<date>-dfn-estaillades-gpdata4-257-porosity-field/
#                checkpoints/{last.ckpt, porosity-field-NNN-val_loss.ckpt}
```

**Variant B — `gpdata4-129`, crop 64, no validation + cosine decay:**
```bash
cd /opt/persistence/repos/DiffSci2
python pipelines/fieldcontrolled/0003-porosity-field-training.py \
    --stone Estaillades \
    --data-source gpdata4-129 \
    --center-crop 64 \
    --max-epochs 20 \
    --devices 1,2,3,4,5,6 \
    --warmup-epochs 2 \
    --train-split 550 \
    --ema-decay 0.99 \
    --no-validation \
    --cosine-decay-epochs 5
# Output (REAL): savedmodels/experimental/<date>-dfn-estaillades-gpdata4-129-porosity-field/
#                checkpoints/last.ckpt   (save_last only)
```
Swap `--stone` among Bentheimer / Doddington / Estaillades / Ketton.

> `--devices 1,2,3,4,5,6` are **literal physical GPUs** (no `CUDA_VISIBLE_DEVICES`).
> The script's printed default dir says `models/experimental/...`; the real
> artifacts are under `savedmodels/`.

---

## Stage 2 — GENERATION (`0004e`, two stages)

### Stage 2a — latent (multi-GPU, spatial-parallel)
```bash
cd /opt/persistence/repos/DiffSci2
torchrun --nproc_per_node=8 pipelines/fieldcontrolled/0004e-porosity-field-generator.py \
    --mode latent \
    --checkpoint savedmodels/pore/field_controlled/fc129/estaillades.ckpt \
    --stone Estaillades \
    --output-dir saveddata/generated/estaillades/pfield_gen_case_10 \
    --generation-case 5 \
    --volume-shape 1280x1280x1280 \
    --volume-samples 1 \
    --save-porosity
# Writes: <output-dir>/latents/<i>.latent.pt, <i>.porosity.npy, latent_config.json
# --volume-shape dims must each be multiples of 128; each latent axis (pixel/8)
#   must be divisible by nproc_per_node. For a 1280x1280x4352 "large" slab use
#   latent 160x160x544 with --nproc_per_node=4.
# Use field_controlled/fc257/<stone>.ckpt + --generation-case 8 for the 257 (case_9) regime.
```
> The dated `experimental/.../checkpoints/last.ckpt` is the historical original
> of `fc129/<stone>.ckpt` (e.g. `20260328-dfn-estaillades-gpdata4-129-porosity-field`);
> either works.

### Stage 2b — decode (1 GPU, chunked, optional disk-offload)
```bash
cd /opt/persistence/repos/DiffSci2
python pipelines/fieldcontrolled/0004e-porosity-field-generator.py \
    --mode decode \
    --output-dir saveddata/generated/estaillades/pfield_gen_case_10 \
    --device cuda:0 \
    --disk-offload \
    --disk-offload-dir /opt/persistence3/chunk_decode_mmap
# Reads: <output-dir>/latent_config.json + latents/*.latent.pt
# Writes: <output-dir>/data/<i>.npy  (bool; threshold x > x.mean()), timing.json
# Binarization threshold is PER-VOLUME x.mean(). --disk-offload-dir must exist with
#   ample free space; the default /tmp/... can fill /tmp on large decodes.
```

For the production paper cases without a `--checkpoint` (scalar regimes), use
`--generation-case 2` (null) or `--generation-case 3` (scalar) — these
auto-load `savedmodels/pore/production/<stone>_pcond.ckpt` and need no
`--checkpoint`.

---

## Stage 3 — EVALUATION

All evaluators use **pore = 0, solid = 1** in the stored `.npy` (porosity =
`(1 - volume).mean()`); they flip to pore-space before SNOW2. SNOW2 is now
**vendored in `diffsci2.extra.pore`** (ported from `poregen`), so no external
`poregen` is required. Reference `.raw` volumes are the Imperial College rocks;
legacy scripts still hardcode `DATA_DIR = ~/repos/PoreGen/saveddata/raw/imperial_college/`
— repoint it to `saveddata/raw/imperial_college/` (downloaded via
`scripts/download_imperial_rocks.py`). The `--use-cached-network` path skips
SNOW2 entirely.

### 3a — Per-cube two-phase metrics (`0005b`, main evaluator)
```bash
python pipelines/fieldcontrolled/0005b-porosity-field-new-metrics-evaluator.py \
  --stone Estaillades --pattern _pfield_gen_case_10 \
  --volume-sizes 1280 --use-cached-network \
  --output saveddata/metrics/metrics_estaillades_pfield-gen-case-10_twophase.npz
# WITHOUT --output the npz is written to the CURRENT WORKING DIRECTORY. Always pass --output.
# Side-effect: writes <vol>.network.npz next to each input cube (the SNOW2 cache).
```
Repeat over the four paper patterns: `_pfield_gen_case_10` (FC-129),
`_pfield_gen_case_9` (FC-257), `_pfield_gen_3_case_2` (Uncond),
`_pfield_gen_3_case_3` (Controlled), for each stone.

### 3b — Diversity (`0010`)
```bash
python pipelines/fieldcontrolled/0010-diversity-calculation.py \
  --path saveddata/generated/ketton/pfield_gen_case_10/data \
  --stone Ketton --divisions 4 --stride full --recalculate
# --stride is a MODE word here: full = non-overlap, half = 50% overlap.
# Repeat with --stride half; over the 4 paper cases × 4 stones.
# Writes *.calculated_{porosity,permeability}_strides_<div>_<mode>.npy into --path.
```

### 3c — Large-slab subvolume sweep (`0005d-large-subvol`)
```bash
python pipelines/fieldcontrolled/0005d-porosity-field-new-metrics-evaluator-large-subvol.py \
  --stone Ketton \
  --volume-path saveddata/archive/large_volumes/ketton_pfield_gen_case_10_large/data/0.npy \
  --stride 1024 --use-cached-network \
  --output-dir saveddata/metrics
# Here --stride is an INTEGER voxel count along z (1024 = 4 non-overlap subvols).
# Writes metrics_<stone>_largescale_stride1024_twophase.npz.
```

---

## Stage 4 — PHYSICS (`0005-large-pnm` → `0011`)

### 4a — Whole-slab network extraction (`0005-large-pnm`)
`pipelines/fieldcontrolled/0005-porosity-field-new-metrics-evaluator-large-pnm.py` extracts the
SNOW2 network from the whole large slab and saves `0.network.npz` (no flow).
This is the producer of the network `0011` consumes.

### 4b — Oil-water flow + Buckley-Leverett (`0011`)
```bash
python pipelines/fieldcontrolled/0011-oil-water-flow.py \
  --network saveddata/archive/large_volumes/estaillades_pfield_gen_case_10_large/data/0.network.npz \
  --voxel-size 3.3116 --n-steps 20
# --voxel-size is in MICROMETERS here (≠ the meters used by 0005b/0005d/0010).
# Consumes a PRE-EXTRACTED network; never runs SNOW2 itself.
# Output: <network_dir>/oilwater_results.npz (Corey fit + Buckley-Leverett solution).
```

---

## Stage 5 — FIGURES

Thesis figures are produced by
`notebooks/exploratory/dfn/tolatex/scripts/generate_*.py` (Part II + most
illustrative figures) and a couple of notebooks, reading from the metrics
`.npz` and the generated volumes. The exact figure → generator → input mapping
is in `claude/plan/thesis-reproduction/03-thesis-figure-provenance.md`. **Those
notebooks are unmaintained and reference pre-migration paths** — see
`notebooks/exploratory/dfn/README_STALE.md` and `docs/ARTIFACTS_AND_DATA.md`.

---

## Cached-artifact ordering (what feeds what)

```
generation case dirs                              large slab dirs (_large)
  1280_*.npy                                         0.npy (1280x1280x4352)
     |                                                  |
 0005b (--stone --pattern)                          0005-large-pnm  -> 0.network.npz
   -> 1280_*.network.npz   (SNOW2 cache)                |        \
   -> metrics_<stone>_<pattern>_twophase.npz       0005d (--stride 1024)   0011 (--network 0.network.npz)
                                                    -> 0.subvol_NN.network.npz   -> oilwater_results.npz
 0010 (--divisions --stride full/half)             -> metrics_<stone>_largescale_stride1024_twophase.npz
   -> *.subblock_*.network.npz
   -> *.calculated_{porosity,permeability}_strides_*.npy
```
