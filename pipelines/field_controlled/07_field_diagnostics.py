#!/usr/bin/env python
"""Stage 7 — mechanism diagnostics.

For each generated volume, compute the generated porosity field (FFT
convolution, same kernel as stage 1), scatter vs the input conditioning
field, log-field mean/variance, and logit-warped two-point correlation.
Used for §4.3 of main.tex.

Legacy counterpart: scripts/legacy/0007-porosity-field-evaluation.py

TODO (Danilo): port the per-volume loop. The core ops (FFT convolution
+ TPC) are already in diffsci2.extra.two_point_correlation.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    raise NotImplementedError(
        "07_field_diagnostics.py is a skeleton. Use "
        "scripts/legacy/0007-porosity-field-evaluation.py until ported."
    )


if __name__ == "__main__":
    main()
