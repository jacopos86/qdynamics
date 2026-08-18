#!/usr/bin/env python3
"""Emit the Paper III accuracy-versus-compiled-2Q frontier figure.

Single-plot figure (one plot per figure, repo plotting convention): first
excitation error against cumulative compiled two-qubit cost at the stored
weak-coupling dimer point, for the five selection arms of the frontier
evidence plus the fixed-class points and the Krylov best-per-cost envelope
from the comparator evidence. Deterministic and regenerable from the
committed evidence JSONs. Reporting only; never feeds controller decisions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DIAG = REPO_ROOT / "output/diagnostics/paper_iii_cost_frontier_arms_20260818_v1"
FRONTIER_JSON = DIAG / "frontier_arms_summary.json"
COMPARATORS_JSON = DIAG / "comparator_arms_summary.json"
EXCHANGE_JSON = DIAG / "exchange_maintenance_evidence.json"
DEFAULT_OUTPUT = REPO_ROOT / "MATH/paper_details/generated/paper_iii_frontier_figure.pdf"

_ARM_STYLE = {
    "geometry_alpha1": ("geometry-selected ($\\alpha=1$)", "#2673b8", "-", 2.2),
    "geometry_selected": ("geometry-selected (no cost discount)", "#7fb3d8", "--", 1.4),
    "compiled_cost": ("cheapest-first (compiled cost)", "#999999", ":", 1.2),
    "cost_proxy": ("cheapest-first (proxy cost)", "#bbbbbb", ":", 1.2),
    "input_order": ("input order", "#555555", "-.", 1.2),
}
_ERR_FLOOR = 1.0e-16


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frontier = json.loads(FRONTIER_JSON.read_text(encoding="utf-8"))
    comparators = json.loads(COMPARATORS_JSON.read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
    for arm_key, (label, color, linestyle, width) in _ARM_STYLE.items():
        rows: list[dict[str, Any]] = frontier["arms"].get(arm_key, [])
        rows = [row for row in rows if row.get("abs_err") is not None]
        if not rows:
            continue
        ax.plot(
            [max(float(row["cum_2q"]), 0.5) for row in rows],
            [max(float(row["abs_err"]), _ERR_FLOOR) for row in rows],
            linestyle,
            color=color,
            linewidth=width,
            label=label,
        )

    for name, arm in comparators["fixed_class_arms"].items():
        if arm.get("abs_err_vs_reference") is None:
            continue
        ax.plot(
            float(arm["total_2q_graph_span"]),
            max(float(arm["abs_err_vs_reference"]), _ERR_FLOOR),
            "s",
            color="#222222",
            markersize=7,
        )
        ax.annotate(
            f"fixed: {name}",
            (float(arm["total_2q_graph_span"]), float(arm["abs_err_vs_reference"])),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            color="#222222",
        )

    exchange = json.loads(EXCHANGE_JSON.read_text(encoding="utf-8"))
    reference = float(exchange["reference_root0"])
    run = exchange["runs"]["from_geometry_alpha1"]
    ax.plot(
        float(run["final"]["total_compiled_cost"]),
        max(abs(float(run["final"]["root0_energy"]) - reference), _ERR_FLOOR),
        "*",
        color="#b8262c",
        markersize=14,
        label="geometry + certified exchange",
        zorder=5,
    )

    envelope = comparators["krylov_arm"]["best_per_cost_envelope"]
    if envelope:
        ax.plot(
            [max(float(row["cum_2q_graph_span"]), 0.5) for row in envelope],
            [max(float(row["abs_err_vs_reference"]), _ERR_FLOOR) for row in envelope],
            "o-",
            color="#c05020",
            linewidth=1.4,
            markersize=4,
            label="real-time Krylov (best-per-cost envelope)",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cumulative compiled two-qubit gates (two_qubit_only_v1, graph-span oracle)")
    ax.set_ylabel(r"$|\Delta E_1|$ vs full-basis reference")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8, loc="lower left")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    png = args.output.with_suffix(".png")
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"wrote {args.output}")
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
