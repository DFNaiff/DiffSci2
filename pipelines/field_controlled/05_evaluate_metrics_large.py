#!/usr/bin/env python
"""Stage 5 (large) — evaluate metrics on a single large volume by
splitting it into 1024^3 subvolumes.

For the 2304^3 or 1024^2 × 4096 feasibility demo runs in main.tex §4.2:
loads one large `.npy`, crops the border, partitions along the long
axis (or all axes for cubic 2304^3), and applies the standard 1024^3
evaluation pipeline to each subcube. Writes per-subcube JSON and a
summary.

Legacy counterpart: scripts/legacy/0005c-porosity-field-new-metrics-evaluator-large.py
                    (and -large-subvol.py variant).

TODO (Danilo): port the subcube loop from the legacy script. The per-
subcube body can reuse pipelines/field_controlled/05_evaluate_metrics.py
directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    raise NotImplementedError(
        "05_evaluate_metrics_large.py is a skeleton. For the paper's "
        "2304^3 and 1024^2 × 4096 runs, use "
        "scripts/legacy/0005c-porosity-field-new-metrics-evaluator-large.py."
    )


if __name__ == "__main__":
    main()
