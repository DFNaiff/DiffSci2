#!/usr/bin/env python
"""Train a 2D VAE on drosophila wing slices.

Delegates to the legacy script. If the hardcoded paths inside no longer
match your environment, edit scripts/legacy/0001-drosophila-autoencoder.py
directly or port the CLI here.

Legacy counterpart: scripts/legacy/0001-drosophila-autoencoder.py
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEGACY = REPO / "scripts" / "legacy" / "0001-drosophila-autoencoder.py"


def main() -> None:
    sys.argv[0] = str(LEGACY)
    runpy.run_path(str(LEGACY), run_name="__main__")


if __name__ == "__main__":
    main()
