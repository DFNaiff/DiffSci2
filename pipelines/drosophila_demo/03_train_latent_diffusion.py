#!/usr/bin/env python
"""Train a 2D latent diffusion model on drosophila slices (using the VAE).

Legacy counterpart: scripts/legacy/0001-drosophila-training-latent.py
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEGACY = REPO / "scripts" / "legacy" / "0001-drosophila-training-latent.py"


def main() -> None:
    sys.argv[0] = str(LEGACY)
    runpy.run_path(str(LEGACY), run_name="__main__")


if __name__ == "__main__":
    main()
