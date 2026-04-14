#!/usr/bin/env python
"""Stage 4 (non-cubic) — two-stage generation for elongated volumes.

Splits generation into `--mode latent` (writes the latent tensor) and
`--mode decode` (chunk-decodes from disk-backed latent), needed for
1024^2 × 4096 and similar non-cubic targets described in the feasibility
demo of main.tex.

Legacy counterpart: scripts/legacy/0004e-porosity-field-generator.py

TODO (Danilo): port the two-stage logic from the legacy script, keeping
the new CLI shape (config + variant + --shape Lx Ly Lz).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    raise NotImplementedError(
        "04_generate_two_stage.py is a skeleton. For the paper's "
        "1024^2 × 4096 runs, use scripts/legacy/0004e-porosity-field-generator.py."
    )


if __name__ == "__main__":
    main()
