#!/usr/bin/env python
"""Stage 8 — two-phase flow: relative permeability and capillary pressure.

Per the paper decision in CODE_ORGANIZATION.md §13, this file is kept
*thin*: only k_r(S_w) and P_c(S_w) from pore-network drainage, the
same metrics that feed Table 4 of main.tex. The richer Buckley-Leverett
/ Corey / oil-water displacement analyses live in
`pipelines/experiments/two_phase_flow.py`.

In practice the computation here is already done by stage 5
(05_evaluate_metrics.py writes drainage_sw / drainage_pc / kr_w / kr_nw
into each *.metrics.json). This file aggregates those into the paper's
reporting format (case means ± std, IAE against reference, etc.).

TODO (Danilo): port the aggregation from scripts/legacy/0005b (the
downstream IAE computation) using `diffsci2.metrics.iae_kr` and
`diffsci2.metrics.iae_pc`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip())
    p.add_argument("--metrics-dir", required=True,
                   help="Directory containing *.metrics.json from stage 5.")
    p.add_argument("--reference-json", required=False,
                   help="Optional reference rock metrics JSON (same format). "
                        "If provided, IAE errors against reference are computed.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    files = sorted(metrics_dir.glob("*.metrics.json"))
    if not files:
        raise SystemExit(f"No *.metrics.json found in {metrics_dir}")

    summaries = [json.loads(f.read_text()) for f in files]
    ks = np.array([s["k_abs_m2"] for s in summaries])
    pors = np.array([s["porosity"] for s in summaries])

    out = {
        "n_samples": len(summaries),
        "porosity_mean": float(pors.mean()),
        "porosity_std": float(pors.std(ddof=1)) if len(pors) > 1 else 0.0,
        "k_abs_mean_m2": float(ks.mean()),
        "k_abs_std_m2": float(ks.std(ddof=1)) if len(ks) > 1 else 0.0,
    }

    if args.reference_json:
        from diffsci2.metrics import iae_kr_curves, iae_pc_curve
        ref = json.loads(Path(args.reference_json).read_text())
        iae_krs, iae_pcs = [], []
        for s in summaries:
            iae_krs.append(iae_kr_curves(
                s["drainage_sw"], s["kr_w"], s["kr_nw"],
                ref["drainage_sw"], ref["kr_w"], ref["kr_nw"],
            ))
            iae_pcs.append(iae_pc_curve(
                s["drainage_sw"], s["drainage_pc_pa"],
                ref["drainage_sw"], ref["drainage_pc_pa"],
            ))
        out["iae_kr_mean"] = float(np.mean(iae_krs))
        out["iae_kr_std"] = float(np.std(iae_krs, ddof=1)) if len(iae_krs) > 1 else 0.0
        out["iae_pc_mean"] = float(np.mean(iae_pcs))
        out["iae_pc_std"] = float(np.std(iae_pcs, ddof=1)) if len(iae_pcs) > 1 else 0.0

    out_path = metrics_dir / "two_phase_flow_summary.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
