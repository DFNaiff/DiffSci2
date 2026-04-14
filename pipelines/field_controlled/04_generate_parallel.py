#!/usr/bin/env python
"""Stage 4 (multi-GPU) — spatial-parallel volume generation.

Same method as 04_generate.py but uses diffsci2.distributed to split the
activation volume along the depth axis across N GPUs, enabling target
sizes up to ~2304^3 on 4×A100 hardware.

Legacy counterpart: scripts/legacy/0004d-porosity-field-generator.py

Usage:
    torchrun --nproc_per_node=4 \\
        pipelines/field_controlled/04_generate_parallel.py \\
        --config pipelines/field_controlled/configs/bentheimer.yaml \\
        --variant fc129 --size 2304 --n-samples 1 \\
        --output-dir ./outputs/bentheimer_fc129_2304

TODO (Danilo): this is a thin skeleton. The 0004d logic wrapping the
model in `convert_to_spatial_parallel` is unchanged — only the path /
config resolution is refactored. Port the body from 0004d when needed.
Until then, the legacy script in scripts/legacy/ still runs as it did
for the paper.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    raise NotImplementedError(
        "04_generate_parallel.py is a skeleton. For the paper runs, use "
        "scripts/legacy/0004d-porosity-field-generator.py. "
        "Port the body here when you need multi-GPU generation in the "
        "new pipeline."
    )


if __name__ == "__main__":
    main()
