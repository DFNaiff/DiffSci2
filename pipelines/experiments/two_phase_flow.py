#!/usr/bin/env python
"""Buckley-Leverett + Corey fit + oil-water drainage simulation.

Thesis-adjacent work layered on top of the paper's k_r / P_c drainage
(§4.2 and `pipelines/field_controlled/08_two_phase_flow.py`). Separated
per the decision recorded in
`claude/report/CODE_ORGANIZATION.md §13`: the paper keeps stage 8 thin
(k_r and P_c only); this richer analysis lives here.

Delegates to the two legacy scripts, which together cover:
  - 0005d-porosity-field-buckley-leverett.py: BL solver + Corey fit per
    pre-extracted network (.network.npz);
  - 0011-oil-water-flow.py: full drainage + relative-permeability +
    Corey + BL pipeline including REV sweeps.

Invoke the desired legacy script explicitly via --variant; we keep both
in-tree so nothing is lost.
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEGACY_DIR = REPO / "scripts" / "legacy"

VARIANTS = {
    "buckley_leverett": LEGACY_DIR / "0005d-porosity-field-buckley-leverett.py",
    "oil_water": LEGACY_DIR / "0011-oil-water-flow.py",
}


def main() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--variant", choices=VARIANTS.keys(), required=True,
                   help="Which legacy pipeline to run. Further CLI args "
                        "are forwarded to the legacy script verbatim.")
    args, rest = p.parse_known_args()
    target = VARIANTS[args.variant]
    # Replace our argv with the legacy script's argv.
    sys.argv = [str(target), *rest]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
