#!/usr/bin/env python
"""FiLM + multi-scale enhanced conditioning post-training.

Legacy: scripts/legacy/0003b-porosity-field-training-enhanced.py
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEGACY = REPO / "scripts" / "legacy" / "0003b-porosity-field-training-enhanced.py"


def main() -> None:
    sys.argv[0] = str(LEGACY)
    runpy.run_path(str(LEGACY), run_name="__main__")


if __name__ == "__main__":
    main()
