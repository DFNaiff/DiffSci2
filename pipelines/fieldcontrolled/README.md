# `pipelines/fieldcontrolled/` — the field-controlled rock-reconstruction pipeline

This is the **canonical thesis pipeline** (field-controlled 3D latent diffusion of
porous rock), promoted out of the old numbered `scripts/` lineage so the real
pipeline is unambiguous. Superseded / dead variants were archived to
`scripts/old/`. Run order:

```
0002-porosity-field-estimator.py                         GP (Matérn) porosity-field FIT (input prep)
  → 0003-porosity-field-training.py                      field-conditioned latent-diffusion TRAINING (frozen VAE)
    → 0004e-porosity-field-generator.py                  two-stage GENERATION  (--mode latent | --mode decode)
      → evaluation / metrics:
          0005b-porosity-field-new-metrics-evaluator.py             main per-cube two-phase metrics (porosity, K, drainage, kr/Pc)
          0005d-porosity-field-new-metrics-evaluator-large-subvol.py  large-slab z-stride two-phase metrics
          0010-diversity-calculation.py                            strided sub-block diversity (porosity/permeability fields)
          0005-porosity-field-metrics-evaluator.py                 Part I porosity/perm/TPC/PSD KDE figures
          0005b-rerun-drainage.py                                  cheap θ/σ drainage re-sweep on cached networks (utility)
      → 0005-porosity-field-new-metrics-evaluator-large-pnm.py     whole-slab `0.network.npz` extraction (producer for ↓)
        → 0011-oil-water-flow.py                          oil-water flow + REV sweeps + Buckley-Leverett PHYSICS
```

The corrected, runnable commands for each stage are in
[`docs/REPRODUCE_THESIS.md`](../../docs/REPRODUCE_THESIS.md); the per-script roles
and the full superseded-vs-canonical verdict are in
[`docs/SCRIPTS_CATALOG.md`](../../docs/SCRIPTS_CATALOG.md).

## Notes
- Run from the **repo root**, e.g. `python pipelines/fieldcontrolled/0003-porosity-field-training.py …`
  (paths inside the scripts are absolute or resolved relative to the repo root).
- Raw rocks come from `saveddata/raw/imperial_college/`
  (`scripts/download_imperial_rocks.py`); SNOW2/PNM is vendored in
  `diffsci2.extra.pore` (no external `poregen`).
- `0003` and `0004e` still add `notebooks/exploratory/dfn/aux/` to `sys.path` for
  `model_loaders` — a remaining coupling to the (otherwise disposable) exploratory
  tree; that helper should eventually be ported into `diffsci2`.
- The **unconditional** training branch (`scripts/0009-unconditional-training-3d.py`)
  is a separate pipeline and is intentionally not here.
