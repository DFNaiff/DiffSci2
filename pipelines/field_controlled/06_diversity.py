#!/usr/bin/env python
"""Stage 6 — sub-block diversity.

For each generated 1024^3 volume, partition into a 4^3 = 64 grid of
non-overlapping 256^3 sub-blocks, compute porosity and absolute
permeability per sub-block, and write a per-volume array. The Hellinger
distance against the reference rock's sub-block distribution is
computed with `diffsci2.metrics.hellinger_gaussian`.

Legacy counterpart: scripts/legacy/0010-diversity-calculation.py

TODO (Danilo): port the stride/grid loop from the legacy script. The
OpenPNM call per subcube is identical to stage 5.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    raise NotImplementedError(
        "06_diversity.py is a skeleton. Use "
        "scripts/legacy/0010-diversity-calculation.py for now; port when "
        "you're ready to consolidate."
    )


if __name__ == "__main__":
    main()
