#!/usr/bin/env python
"""Train 3D unconditional LDM with crop-mask loss (border-only).

Legacy counterpart: scripts/legacy/0009b-unconditional-training-3d-masked.py

TODO (Danilo): port when revisiting border-masked training.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    raise NotImplementedError(
        "Use scripts/legacy/0009b-unconditional-training-3d-masked.py until ported."
    )


if __name__ == "__main__":
    main()
