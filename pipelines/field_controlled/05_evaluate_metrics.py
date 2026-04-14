#!/usr/bin/env python
"""Stage 5 — evaluate flow metrics on generated 1024^3 volumes.

Porosity, absolute permeability (SNOW2 + OpenPNM, all three directions
averaged), drainage Pc(Sw), relative permeability k_r(Sw). Writes one
JSON per input volume plus a combined summary.

Legacy counterpart: scripts/legacy/0005b-porosity-field-new-metrics-evaluator.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from diffsci2.extra.pore import PoreNetworkPermeability  # noqa: E402
# poregen is an external helper; the legacy scripts rely on it. Port
# directly to keep behavior identical with 0005b.
try:
    from poregen.features import snow2  # type: ignore
    HAVE_POREGEN = True
except ImportError:
    HAVE_POREGEN = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip())
    p.add_argument("--input-dir", required=True,
                   help="Directory containing generated *.npy volumes.")
    p.add_argument("--voxel-size", type=float, required=True,
                   help="Physical voxel size in metres (e.g. 3.0035e-6 for Bentheimer).")
    p.add_argument("--contact-angle-deg", type=float, default=140.0)
    p.add_argument("--surface-tension", type=float, default=0.48)
    p.add_argument("--drainage-steps", type=int, default=40)
    p.add_argument("--glob", default="*.npy",
                   help="Glob pattern under --input-dir. Default: '*.npy'.")
    return p.parse_args()


def evaluate_volume(
    volume_path: Path,
    voxel_size: float,
    contact_angle_deg: float,
    surface_tension: float,
    drainage_steps: int,
) -> dict:
    if not HAVE_POREGEN:
        raise RuntimeError(
            "poregen.features.snow2 is required for network extraction. "
            "Install it in the environment or run from scripts/legacy/ where "
            "the import path is identical."
        )
    vol = np.load(volume_path)
    if vol.dtype != np.bool_:
        vol = vol.astype(bool)

    porosity = float((1 - vol).mean())

    # SNOW2 network extraction + OpenPNM flow.
    network = snow2.extract_pore_network(~vol, voxel_size=voxel_size)
    pnm = PoreNetworkPermeability(network=network, voxel_size=voxel_size)

    k_abs = pnm.absolute_permeability_mean_directions()
    drainage = pnm.drainage(
        contact_angle_deg=contact_angle_deg,
        surface_tension=surface_tension,
        n_steps=drainage_steps,
    )
    kr = pnm.relative_permeability_from_drainage(drainage)

    return {
        "file": volume_path.name,
        "porosity": porosity,
        "k_abs_m2": k_abs,
        "drainage_sw": drainage.saturation.tolist(),
        "drainage_pc_pa": drainage.capillary_pressure.tolist(),
        "kr_w": kr.kr_w.tolist(),
        "kr_nw": kr.kr_nw.tolist(),
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    volumes = sorted(input_dir.glob(args.glob))
    # Skip porosity sidecars.
    volumes = [v for v in volumes if not v.name.endswith(".porosity.npy")]

    print(f"Found {len(volumes)} volumes under {input_dir}")
    results = []
    for i, vp in enumerate(volumes):
        print(f"[{i+1}/{len(volumes)}] {vp.name}")
        r = evaluate_volume(
            vp, args.voxel_size, args.contact_angle_deg,
            args.surface_tension, args.drainage_steps,
        )
        results.append(r)
        # Write per-volume JSON alongside the .npy.
        with open(vp.with_suffix(".metrics.json"), "w") as f:
            json.dump(r, f, indent=2)

    summary = input_dir / "metrics_summary.json"
    with open(summary, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary -> {summary}")


if __name__ == "__main__":
    main()
