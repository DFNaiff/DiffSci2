# `notebooks/tutorials/fieldcontrolling/` — field-controlled generation, in 2D

A five-notebook walkthrough of the **field-controlling** method of Part II of the thesis,
done entirely in 2D on Berea sandstone slices so that every step is cheap to run, quick to
re-run, and legible in a figure.

The 3D production pipeline lives in [`pipelines/fieldcontrolled/`](../../../pipelines/fieldcontrolled/).
These notebooks are its 2D mirror, stage for stage — same modules, same configuration
style, same algorithms, one dimension down.

## Run order

Each notebook consumes the artifacts of the previous ones and writes its own into
`fcdata/` (gitignored). Run them in order the first time.

| notebook | what it does | 3D counterpart |
|---|---|---|
| `00-vae-training-2d.ipynb` | trains the frozen $F=8$, $C_\ell=4$ autoencoder; shows it generalises past its training size | the production VAE of Part I |
| `01-porosity-field-and-gaussian-process.ipynb` | **what a porosity field is**, why logit-Gaussian, Matérn fit, GP sampling, cost of the coarse-to-fine shortcut | `0002-porosity-field-estimator.py` |
| `02-scalar-porosity-pretraining.ipynb` | latent diffusion conditioned on one scalar porosity; shows the embedder is shape-agnostic | Algorithm 1 (base training) |
| `03-field-controlled-training.ipynb` | fine-tune on paired (patch, field) data | Algorithm 2 / `0003-porosity-field-training.py` |
| `04-field-controlled-inference.ipynb` | GP field → ODE → decode → binarise → crop; FC vs Uncond vs Controlled; sub-block diversity | Algorithm 3 / `0004e-porosity-field-generator.py` |
| `05-chunk-decoding.ipynb` | the same inference with tiled decoding; receptive-field identity, norm seams, memory scaling | Appendix B (chunk decoding) |

Plus one investigation notebook, off the presentation path:

| notebook | what it does |
|---|---|
| `06-what-the-loss-cannot-see.ipynb` | trains **both** phases with behavioural probes running alongside the loss, to test whether the denoising loss can see conditioning adherence at all — and therefore whether early stopping on `val_loss` is steering by the wrong instrument |

```
raw Berea volume
  ├─→ 00  VAE  ─────────────────────────────────────┐
  └─→ 01  slices + porosity fields + Matérn params ─┤
                    │                               │
                    ├─→ 02  scalar-conditioned model│
                    │        │                      │
                    └───────→ 03  field-conditioned model
                                   │
                                   ├─→ 04  inference (one-shot decode)
                                   └─→ 05  inference (chunk decode, larger)
```

## Configuration

`fcconfig.py` holds **only** paths, constants, model factories and checkpoint IO. All the
scientific mechanism — field computation, GP fitting, the paired dataset, the conditioning
at inference — is written out inline in the notebook that teaches it, and downstream
notebooks consume the saved artifact rather than re-deriving it. This is the same handoff
structure the real pipeline uses.

Key constants, and the 3D values they mirror:

| | 2D tutorial | 3D thesis |
|---|---|---|
| training patch | $256^2$ | $256^3$ |
| symmetries | 8 ($D_4$) | 48 (cube) |
| averaging window $w$ | 129 ($R=64$) | 129 / 257 ($R=64/128$) |
| VAE | $F=8$, $C_\ell=4$ | $F=8$, $C_\ell=4$ |
| conditioning field | $32^2$ | $32^3$ |
| GP coarse grid | $32^2$ | $32^3$ |
| border crop $b$ | 64 | 128 |
| ODE | 21 Heun steps, $\sigma\in[0.002, 80]$ | same |
| largest generated | $2048^2$ | $1024^3$, $1024^2\times4096$ |

Change `fc.DEVICE` to pick a GPU. **Never `cuda:7`** unless explicitly authorised.

## `run_all.sh` vs. working by hand

`run_all.sh` executes the notebooks with `jupyter nbconvert --inplace`, which **overwrites
the notebook files, including any outputs you produced interactively**. Use it for a clean
end-to-end rebuild, not while you have notebooks open in Jupyter.

`apply_fixes.py` is the opposite: surgical, idempotent, source-matched patches that leave
everything else in the notebook alone (including cells you added yourself). Run
`python apply_fixes.py` for a dry run, `--write` to apply. Patched cells have their stale
outputs cleared so it is obvious they need re-executing. Each fix is documented in the
module docstring.

## Checking notebooks without running them

`python scopecheck.py 0*.ipynb` parses every code cell in order, tracks which names are
bound, and reports any name used before it exists — module-level immediately, and names
referenced inside function bodies against the whole notebook. It catches the `NameError`
class of breakage (and syntax errors) in under a second, which matters when a notebook
takes an hour to execute or when it is being edited by something that cannot run it.

## Investigation logs (notebook 06)

Notebook 06 writes machine-readable logs *as it runs*, flushed and `fsync`ed after every
probe point, so the run can be read while it is still in progress:

```
fcdata/investigation/logs/
  manifest.json          every resolved knob, git commit + dirty flag, GPU, torch version
  probes_<phase>.jsonl   one JSON object per probe point, incl. per-sigma arrays
  progress_<phase>.log   the running commentary
  summary_<phase>.csv    flat table, scalars only
  records_all.json       everything, consolidated at the end
```

`load_records(tag)` in §7 reads a phase back from its JSONL, so the entire analysis can be
re-run in a fresh kernel — or mid-run — without retraining. Phases are `A-pretrain`,
`B-finetune`, `C-control`.

## Environment

```bash
conda activate ddpm_env
cd /home/ubuntu/repos/DiffSci2/notebooks/tutorials/fieldcontrolling
jupyter lab
```

`fcconfig` is imported by filename, so notebooks must be run with this directory as the
working directory.

## Artifacts

Everything lands in `fcdata/` (gitignored, ~1 GB when fully populated):

```
fcdata/
  berea_slices.npy              uint8   [160, 1000, 1000]   binary slices (solid = 1)
  berea_field_w129.npy          float16 [160, 1000, 1000]   porosity fields, aligned
  berea_gp_w129_latent.npz              fitted Matérn parameters (m, θ², ν, ρ) + the empirical C(r)
  checkpoints/vae/last.ckpt             frozen autoencoder
  checkpoints/scalar/last.ckpt          scalar-conditioned denoiser
  checkpoints/field/last.ckpt           field-conditioned denoiser
  figures/*.pdf, *.png                  every figure, saved for slides and LaTeX
```

Each notebook skips its training step if the checkpoint already exists; set `RETRAIN = True`
(or `FORCE_RECOMPUTE = True` in notebook 01) to force a fresh run.

## Figures

`fc.savefig(fig, name)` writes both a PDF and a PNG into `fcdata/figures/`. The names are
stable, so a figure can be pulled straight into a talk or into
`notebooks/exploratory/dfn/tolatex/thesis/`.

### Map to the Part II presentation frames

Every method frame in `presentation.tex` §*Part II: Field-Controlled Large-Scale
Generation* is currently text and algorithms only. These are the figures built for them:

| presentation frame | candidate figures |
|---|---|
| *Field Controlling* | `01_what_is_a_porosity_field`, `01_field_overlay` |
| *Field Model* | `01_gaussianity_vs_window`, `01_qq_logit`, `01_matern_fit`, `01_gp_vs_real_fields`, `01_coarse_to_fine_cost` |
| *Conditional model* | `02_scalar_training_pairs`, `02_denoising_process` |
| *Pre-Training: Scalar-Conditioned Base Model* | `02_scalar_control_gallery`, `02_scalar_control_curve`, `02_unconditional_samples` |
| *Post-Training: Field-Conditioned Fine-Tuning* | `03_training_pairs`, `03_symmetry_alignment`, `03_conditioning_on_real_field`, `03_synthetic_ramp_control`, **`03_pretrained_vs_posttrained`**, **`03_pretrained_vs_posttrained_tracking`**, **`03_conditioning_loss_sensitivity`** |
| *Inference: Field-Controlled Volume Generation* | `04_pipeline_stages`, `04_border_effect`, `04_zoom_detail` |
| *Chunk Decoding* | `05_tiling_schematic`, `05_normalisation_seams`, `05_seam_profile`, `05_memory_scaling` |
| *Experimental Setup* / results | `04_method_comparison`, `04_subblock_diversity`, `04_field_correlation`, `04_field_tpc_and_stats`, `05_large_generation` |

The window-size figure `01_window_size_comparison` is the 2D version of the FC-129 vs
FC-257 argument, and `04_decode_memory_scaling` motivates chunk decoding before it is
introduced.

## One definition of "the porosity field"

The conditioning field is a **window-$w$ local average, then pooled by $F$**. When you
measure the field a *generated* image actually has, it must be computed the same way, or
the comparison is meaningless. `fcconfig.realised_field()` is the single canonical
implementation, mirroring `scripts/old/0007-porosity-field-evaluation.py`:

```python
field = calculate_porosity_field_full(volume, kernel_size)   # window w
field = average_volume(field, F)                              # then pool by F
```

Pooling the binary image straight down by $F$ instead gives the pore fraction of each
$F\times F$ block — a *window-8* field. This is not a subtle difference. Self-test on real
rock, where the realised field of the real image must reproduce its own conditioning field:

| definition | ρ vs conditioning | std |
|---|---|---|
| `avg_pool2d(1 - binary, F)` | **+0.18** | 0.333 |
| `realised_field(binary)` | **+1.0000** | 0.0586 (= conditioning) |

The wrong version is dominated by block noise and would cap the measured field tracking at
ρ ≈ 0.18 for *any* generator, however good. Notebook 03 runs this self-test explicitly
before reporting any correlation. Comparisons also use `fc.field_interior()` to drop the
belt where the `'same'`-mode field averaged over fewer than $w^2$ real voxels.

## Notes on fidelity to the pipeline

- The `PUNetG` configuration is copied verbatim from `0004e-porosity-field-generator.py`
  (`dimension=3` → `2`), including `cond_dropout=0.1`. Notebook 02 flags one
  documentation/code divergence: what the thesis calls "conditional dropout" is, in the
  code, element-wise dropout on the conditioning embedding — not classifier-free-guidance
  null-token dropout. The unconditional baseline is obtained the way `0004e` does it
  (generation case 7): rebuild the same weights without the embedder and pass `y=None`.
- Chunk decoding uses the same `diffsci2.extra.chunk_decode_2` entry points as the
  pipeline, via the 2D wrapper `chunk_decode_2d`.
- `prepare_decoder_for_cached_decode` must be called while the decoder is on **CPU**,
  before `.to(device)`; otherwise the cached-norm buffers are created on the wrong device.
