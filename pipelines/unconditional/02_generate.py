#!/usr/bin/env python
"""Generate with a standalone unconditional checkpoint.

For the paper's Uncond baseline (which uses the base scalar checkpoint
with null conditioning), use
    pipelines/field_controlled/04_generate.py --variant uncond
instead. This script is for standalone unconditional checkpoints
produced by 01_train_unconditional.py.

Legacy counterpart: scripts/legacy/0004e-porosity-field-generator-unconditional.py

TODO (Danilo): port when there's a standalone unconditional checkpoint
to serve.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    raise NotImplementedError(
        "Use scripts/legacy/0004e-porosity-field-generator-unconditional.py "
        "until ported."
    )


if __name__ == "__main__":
    main()
