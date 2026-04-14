#!/usr/bin/env python
"""Unconditional 2D training on a mixture of stones (ergodicity study).

Legacy: scripts/legacy/0008-unconditional-2d-multistone-training.py
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEGACY = REPO / "scripts" / "legacy" / "0008-unconditional-2d-multistone-training.py"


def main() -> None:
    sys.argv[0] = str(LEGACY)
    runpy.run_path(str(LEGACY), run_name="__main__")


if __name__ == "__main__":
    main()
