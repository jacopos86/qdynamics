#!/usr/bin/env python3
"""Build the Paper-I HH rho-sensitivity figure used by the manuscript."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


DEFAULT_CSV = Path("output/pdf/paper_i_fixed_settings_rho_sweep_deltae_vs_qiskit_costs_20260608_no_optuna.csv")
DEFAULT_OVERLAY = Path("MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_weak_strong_rho_partial_appendix_table_20260611.json")
DEFAULT_OUTPUT_STEM = Path("output/pdf/paper_i_fixed_settings_rho_sweep_hh_deltae_vs_rho_20260608_no_optuna")
RHO_ORDER = [0.05, 0.10, 0.25, 0.50, 1.00]
CASE_ORDER = ["HH weak-weak", "HH strong-weak", "HH weak-strong", "HH strong-strong"]
COLORS = {
    "HH weak-weak": "#d62728",
    "HH strong-weak": "#ff7f0e",
    "HH weak-strong": "#e377c2",
    "HH strong-strong": "#8c564b",
}
MARKERS = {
    "HH weak-weak": "o",
    "HH strong-weak": "o",
    "HH weak-strong": "s",
    "HH strong-strong": "o",
}


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_points(csv_path: Path) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            label = row.get("case_label", "")
            if label not in CASE_ORDER:
                continue
            rho = _float_or_none(row.get("rho"))
            error = _float_or_none(row.get("primary_error"))
            if rho is None or error is None:
                continue
            points.append(
                {
                    "case_label": label,
                    "rho": rho,
                    "primary_error": error,
                    "source": "fixed_settings_csv",
                    "source_record_dir": row.get("source_record_dir"),
                    "source_summary_json": row.get("source_summary_json"),
                }
            )
    return points


def _apply_weak_strong_overlay(points: list[dict[str, Any]], overlay_path: Path) -> list[dict[str, Any]]:
    if not overlay_path.exists():
        return points
    overlay = json.loads(overlay_path.read_text())
    rows = overlay.get("rows") if isinstance(overlay, dict) else None
    if not isinstance(rows, list):
        return points
    # Replace all weak-strong points with the repair/redo overlay so rho=1 and
    # the other weak-strong values share one provenance family.
    points = [p for p in points if p["case_label"] != "HH weak-strong"]
    for row in rows:
        if not isinstance(row, dict):
            continue
        rho = _float_or_none(row.get("rho"))
        error = _float_or_none(row.get("primary_error")) or _float_or_none(row.get("abs_delta_e"))
        if rho is None or error is None:
            continue
        points.append(
            {
                "case_label": "HH weak-strong",
                "rho": rho,
                "primary_error": error,
                "same_cutoff_error": _float_or_none(row.get("abs_delta_e")),
                "ansatz_depth": row.get("ansatz_depth"),
                "source": "weak_strong_repair_redo_overlay",
                "source_json": row.get("source_json"),
                "source_json_remote": row.get("source_json_remote"),
                "source_sha256": row.get("source_sha256"),
                "status": row.get("status"),
            }
        )
    return points


def build_plot(args: argparse.Namespace) -> None:
    points = _apply_weak_strong_overlay(_load_points(args.csv), args.overlay_json)
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for label in CASE_ORDER:
        case_points = sorted((p for p in points if p["case_label"] == label), key=lambda p: p["rho"])
        if not case_points:
            continue
        ax.plot(
            [p["rho"] for p in case_points],
            [p["primary_error"] for p in case_points],
            marker=MARKERS[label],
            markersize=7.5,
            linewidth=2.5,
            color=COLORS[label],
            markeredgecolor="black",
            markeredgewidth=0.8,
            label=label.replace("HH ", ""),
        )

    ax.axhline(args.threshold, color="0.45", linestyle="--", linewidth=1.6)
    ax.text(0.035, args.threshold * 1.12, r"$2\times10^{-4}$", color="0.25", fontsize=11)
    ax.set_yscale("log")
    ax.set_title("Hubbard-Holstein", fontsize=20, pad=8)
    ax.set_xlabel(r"$\rho$", fontsize=16)
    ax.set_ylabel(r"$|\Delta E|$", fontsize=16)
    ax.set_xticks(RHO_ORDER)
    ax.set_xticklabels(["0.05", "0.1", "0.25", "0.5", "1"], rotation=30, ha="right")
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=11)
    ax.set_ylim(args.ymin, args.ymax)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
    fig.tight_layout()
    fig.savefig(args.output_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(args.output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    provenance = {
        "schema": "paper_i_hh_rho_sensitivity_plot_provenance_v1",
        "metric": "primary_error",
        "csv_source": str(args.csv),
        "weak_strong_overlay_json": str(args.overlay_json),
        "output_png": str(args.output_stem.with_suffix(".png")),
        "output_pdf": str(args.output_stem.with_suffix(".pdf")),
        "threshold": args.threshold,
        "notes": [
            "Direct-command CHTC batch paper_i_direct_command_rho_sweep_20260610_v1 completed 35/40 rows.",
            "HH weak-strong points are filled from repair/redo overlay evidence, not direct-command completions.",
        ],
        "points": sorted(points, key=lambda p: (CASE_ORDER.index(p["case_label"]), RHO_ORDER.index(p["rho"]) if p["rho"] in RHO_ORDER else 99)),
    }
    args.output_stem.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(args.output_stem.with_suffix(".png"))
    print(args.output_stem.with_suffix(".pdf"))
    print(args.output_stem.with_suffix(".provenance.json"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--overlay-json", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--threshold", type=float, default=2e-4)
    parser.add_argument("--ymin", type=float, default=1.2e-4)
    parser.add_argument("--ymax", type=float, default=2.0e-1)
    return parser.parse_args()


def main() -> int:
    build_plot(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
