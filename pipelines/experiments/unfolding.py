#!/usr/bin/env python
"""Mirror/unfold a generated volume along two axes (2x expansion) and
extract its pore network, to study border behavior under periodic extension.

Legacy: scripts/legacy/0012-unfolding.py
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEGACY = REPO / "scripts" / "legacy" / "0012-unfolding.py"


def main() -> None:
    sys.argv[0] = str(LEGACY)
    runpy.run_path(str(LEGACY), run_name="__main__")


if __name__ == "__main__":
    main()
