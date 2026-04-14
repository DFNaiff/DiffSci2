#!/usr/bin/env python
"""Train a 3D unconditional latent diffusion model from scratch.

Legacy counterpart: scripts/legacy/0009-unconditional-training-3d.py

TODO (Danilo): port the body. The flow is nearly identical to
pipelines/field_controlled/02_train_base_scalar.py minus the porosity
embedder and condition dropout.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    raise NotImplementedError(
        "Use scripts/legacy/0009-unconditional-training-3d.py until ported."
    )


if __name__ == "__main__":
    main()
