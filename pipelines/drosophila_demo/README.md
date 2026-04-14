# pipelines/drosophila_demo/

A small 2D latent-diffusion example on drosophila wing tissue slices.

The purpose of this pipeline is **illustrative**: it is the smallest
complete example of the latent-diffusion workflow (VAE + conditional
diffusion in latent space) on a tractable 2D dataset, separate from the
heavyweight 3D porous-media pipeline. Use it when teaching or smoke-
testing the library.

## Stages

| # | Script                             | Legacy counterpart                                 |
|---|------------------------------------|----------------------------------------------------|
| 1 | `01_train_autoencoder.py`          | `scripts/legacy/0001-drosophila-autoencoder.py`    |
| 2 | `02_train_diffusion.py`            | `scripts/legacy/0001-drosophila-training.py`       |
| 3 | `03_train_latent_diffusion.py`     | `scripts/legacy/0001-drosophila-training-latent.py`|

Each file in this directory is currently a **thin delegator** to the
legacy script — the dataset paths and checkpoint locations are already
self-contained (they don't depend on the Imperial College data root), so
there was no strong reason to rewrite them. If you want these to use
`DIFFSCI2_DATA_ROOT` like the porous-media pipeline, port them the same
way as `pipelines/field_controlled/01_…`.
