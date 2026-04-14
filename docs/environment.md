# Environment

## Conda

```bash
conda create -n ddpm_env python=3.11
conda activate ddpm_env
pip install -e .
```

`-e .` installs `diffsci2` from the local tree via `setup.py`, so edits
to the library are picked up without reinstall. The top-level
`requirements.txt` lists runtime dependencies.

Python ≥ 3.10 is required for the PEP-604 union syntax used in the
pipeline scripts (`int | None`, etc.).

## GPU

- CUDA 11.8 or 12.x.
- Single A100 is enough for $1024^3$ generation (~50 min per volume
  per the paper).
- Multi-GPU is only needed for $2304^3$ or non-cubic
  $1024^2 \times 4096$ targets; see
  `pipelines/field_controlled/04_generate_parallel.py` (currently a
  skeleton — use `scripts/legacy/0004d-…` until ported).

## Environment variables

Set these only if you want data/checkpoints/GP fits outside the repo:

```bash
export DIFFSCI2_DATA_ROOT=/some/path/to/micro-ct-raw
export DIFFSCI2_MODEL_ROOT=/some/path/to/checkpoints
export DIFFSCI2_FIELDS_ROOT=/some/path/to/gp-fits
```

The default values, rooted inside the repo, work out of the box on the
development machine.

## Remote development

The repo is commonly worked on through VSCode Remote SSH. Nothing in
the pipelines is coupled to that workflow — pure CLI runs work
identically.
