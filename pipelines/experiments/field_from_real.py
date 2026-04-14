#!/usr/bin/env python
"""Generate conditioned on the real porosity field (not GP-sampled).

Useful for diagnostics: how closely can the diffusion model follow a
known porosity pattern?

Legacy: scripts/legacy/0006-porosity-field-generator-from-training.py
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEGACY = REPO / "scripts" / "legacy" / "0006-porosity-field-generator-from-training.py"


def main() -> None:
    sys.argv[0] = str(LEGACY)
    runpy.run_path(str(LEGACY), run_name="__main__")


if __name__ == "__main__":
    main()
