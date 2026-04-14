# Checkpoints

This document lists the published artifacts needed to reproduce the
paper and how to fetch them.

## The bundle

| Role                                    | Canonical path (published view)                                        | Size   |
|-----------------------------------------|------------------------------------------------------------------------|--------|
| Shared VAE (encoder/decoder)            | `savedmodels/pore/production/converted_vaenet.ckpt`                    | ~59 MB |
| Scalar-conditioned base × 4 rocks       | `savedmodels/pore/production/{rock}_pcond.ckpt`                         | 4 × 400 MB |
| Field-controlled w=257 × 4 rocks        | `savedmodels/pore/field_controlled/fc257/{rock}.ckpt`                   | 4 × 400 MB |
| Field-controlled w=129 × 4 rocks        | `savedmodels/pore/field_controlled/fc129/{rock}.ckpt`                   | 4 × 400 MB |

Total: ~5 GB across 13 files.

The field-controlled entries under `savedmodels/pore/field_controlled/`
are **symlinks** into `savedmodels/experimental/` (the dated research
log), so the on-disk source of truth remains the dated folders:

- `savedmodels/experimental/20260324-dfn-{bentheimer,doddington}-gpdata4-257-porosity-field/`
- `savedmodels/experimental/20260325-dfn-{estaillades,ketton}-gpdata4-257-porosity-field/`
- `savedmodels/experimental/20260328-dfn-{rock}-gpdata4-129-porosity-field/`

Nothing under `savedmodels/experimental/` is ever renamed, moved, or
deleted by this repository's tooling — including other dated runs
(e.g. the later `20260327-…-gpdata4-257-…` set, or the early
`20260130*-129-*` set). Those are part of the research log.

## Download

Published on Zenodo (DOI: **TBD — awaiting upload**; see action checklist
in `claude/report/CODE_ORGANIZATION.md` §U.2). Once the record is
public, the manifest URL in `tools/download_checkpoints.py` points at
it. A fresh clone then runs:

```bash
python tools/download_checkpoints.py --set all
```

This downloads into `$DIFFSCI2_MODEL_ROOT` (default `./savedmodels`) and
writes the canonical paths above. It verifies SHA-256 on each file and
is idempotent (a second run only re-fetches anything missing or corrupt).

### If Zenodo is not yet set up

You can still use the script with a local manifest:

```bash
python tools/download_checkpoints.py --manifest-file path/to/MANIFEST.json
```

`MANIFEST.json` format:

```json
{
  "files": [
    {
      "tag": "vae",
      "url": "https://.../converted_vaenet.ckpt",
      "target_path": "pore/production/converted_vaenet.ckpt",
      "sha256": "...",
      "size_bytes": 61440000
    },
    { "tag": "base",  "url": "...", "target_path": "pore/production/bentheimer_pcond.ckpt",            "sha256": "..." },
    { "tag": "fc257", "url": "...", "target_path": "pore/field_controlled/fc257/bentheimer.ckpt",      "sha256": "..." },
    { "tag": "fc129", "url": "...", "target_path": "pore/field_controlled/fc129/bentheimer.ckpt",      "sha256": "..." }
  ]
}
```

`tag` values: `vae`, `base`, `fc257`, `fc129`, `uncond`. The `--set`
flag filters by tag (`base` means `vae` + `base`; `fc` means both FC;
`all` means everything).

## Alternate disk layout

If you keep checkpoints outside the repo:

```bash
export DIFFSCI2_MODEL_ROOT=/opt/persistence/diffsci2/savedmodels
python tools/download_checkpoints.py --set all
```

All pipelines resolve through `DIFFSCI2_MODEL_ROOT`, so no code changes
are needed.

## Provenance

- VAE and `<rock>_pcond.ckpt` come from the Naiff2026 repo; this
  repository ships them as convenience copies in the Zenodo bundle.
- FC-257 and FC-129 were produced by
  `scripts/legacy/0003c-porosity-field-training-vol-correction.py`
  (now also reachable as
  `pipelines/field_controlled/03_train_field_conditioned.py`) on the
  dated runs listed above.
