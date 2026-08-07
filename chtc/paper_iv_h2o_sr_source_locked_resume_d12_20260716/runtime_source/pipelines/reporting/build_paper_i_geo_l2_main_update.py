#!/usr/bin/env python3
"""Build the completed-Geo update for the Paper-I L=2 HH main figure.

The builder changes no manuscript source.  It replaces only the six Geo-ADAPT
curves/rows in the current main-result comparison, preserves the visible SNAKE
and Append-ADAPT evidence, and writes plot plus machine-readable provenance
assets for a subsequent evidence-to-manuscript transfer.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
INVENTORY = REPO_ROOT / (
    "output/pdf/paper_i_geo_evidence_inventory_20260711/"
    "paper_i_geo_evidence_inventory_20260711.json"
)
PAPER_TEX = REPO_ROOT / "MATH/paper_details/Paper_I.tex"
OLD_PLOT_PROVENANCE = REPO_ROOT / (
    "MATH/paper_details/figures/paper_i_physical_lane_snake_duplicate_20260708/"
    "paper_i_physical_lane_snake_duplicate_20260708_append_parent_only_provenance.json"
)
OUTPUT_DIR = REPO_ROOT / "output/pdf/paper_i_geo_l2_completed_update_20260711"
FIGURE_DIR = REPO_ROOT / "MATH/paper_details/figures/paper_i_geo_l2_completed_update_20260711"
STEM = "paper_i_geo_l2_completed_update_20260711"
SCHEMA = "paper_i_geo_l2_completed_main_update_v1"
DISPLAY_CROP = 30
GROUPED_EXACT_MAX_ACTIVE_QUBITS = 8

REGIME_ORDER = (
    "weak-weak",
    "intermediate-weak",
    "strong-weak",
    "weak-strong",
    "intermediate-strong",
    "strong-strong",
)
REGIME_DISPLAY = {
    "weak-weak": "Weak--weak",
    "intermediate-weak": "Intermediate--weak",
    "strong-weak": "Strong--weak",
    "weak-strong": "Weak--strong",
    "intermediate-strong": "Intermediate--strong",
    "strong-strong": "Strong--strong",
}
METHOD_STYLE = {
    "snake": {"label": "SNAKE", "color": "#E45756", "marker": "*", "width": 2.1},
    "geo": {"label": "Geo-ADAPT", "color": "#54A24B", "marker": "^", "width": 1.6},
    "append": {"label": "Append-ADAPT", "color": "#4C78A8", "marker": "o", "width": 1.6},
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def _active_append_cells() -> dict[str, dict[str, int | float]]:
    source = PAPER_TEX.read_text(encoding="utf-8")
    label_index = source.index(r"\label{fig:hh_main_results_composite}")
    start_index = source.rfind(r"\onecolumngrid", 0, label_index)
    block = source[start_index:label_index]
    pattern = re.compile(
        r"^Append\s*&\s*(\d+)\s*&\s*([0-9.eE+\-]+)\s*&\s*([0-9,]+)\s*&\s*"
        r"([0-9,]+)\s*&\s*([0-9,]+)\s*&\s*([0-9,]+)\s*\\\\\s*$",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(block))
    if len(matches) != len(REGIME_ORDER):
        raise ValueError(f"Expected six active Append rows, found {len(matches)}")
    return {
        regime: {
            "k_pl": int(match.group(1)),
            "abs_delta_e": float(match.group(2)),
            "N2q": int(match.group(3).replace(",", "")),
            "D2q": int(match.group(4).replace(",", "")),
            "Dc": int(match.group(5).replace(",", "")),
            "S_alg": int(match.group(6).replace(",", "")),
        }
        for regime, match in zip(REGIME_ORDER, matches, strict=True)
    }


def _support_csv() -> Path:
    provenance = _read_json(OLD_PLOT_PROVENANCE)
    path = Path(str(provenance["support_csv"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = str(provenance["support_csv_sha256"])
    if _sha256(path) != expected:
        raise ValueError("Current comparator support CSV hash mismatch")
    return path


def _append_sources() -> dict[str, dict[str, Any]]:
    cells = _active_append_cells()
    with _support_csv().open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out: dict[str, dict[str, Any]] = {}
    for regime in REGIME_ORDER:
        support = next(
            row
            for row in rows
            if row.get("regime") == regime and row.get("role_key") == "append_macro_c"
        )
        points = [
            {"k": int(round(float(k))), "abs_delta_e": float(error)}
            for k, error in json.loads(support["trajectory_points_json"])
        ]
        curve = {int(point["k"]): float(point["abs_delta_e"]) for point in points}
        k_pl = int(cells[regime]["k_pl"])
        if k_pl not in curve:
            raise ValueError(f"Append curve lacks visible marker k={k_pl} for {regime}")
        out[regime] = {
            **cells[regime],
            "method": "append",
            "trajectory_points": points,
            "marker_abs_delta_e": float(curve[k_pl]),
            "source_json": support["source_json"],
            "source_sha256": support["source_sha256"],
            "source_status": "preserved_visible_pending_completed_iter50_replacement",
            "marker_table_abs_diff": abs(
                float(curve[k_pl]) - float(cells[regime]["abs_delta_e"])
            ),
        }
    return out


def _geo_sources() -> dict[str, Path]:
    inventory = _read_json(INVENTORY)
    rows = [
        row
        for row in inventory["rows"]
        if row.get("paper_placement") == "main_results_hubbard_holstein_L2"
    ]
    if len(rows) != len(REGIME_ORDER):
        raise ValueError(f"Expected six main-result Geo rows, found {len(rows)}")
    out: dict[str, Path] = {}
    for row in rows:
        regime = str(row["display_regime"])
        path = REPO_ROOT / str(row["artifacts"]["result_json"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != str(row["artifacts"]["result_sha256"]):
            raise ValueError(f"Geo inventory source hash mismatch for {regime}")
        out[regime] = path
    if tuple(out) != REGIME_ORDER:
        out = {regime: out[regime] for regime in REGIME_ORDER}
    return out


def _snake_rows() -> dict[str, dict[str, Any]]:
    from pipelines.reporting.build_paper_i_hh_corrected_parent_comparator_page13_pdf import (
        build_snake_rows,
    )

    return {row.regime: row.as_dict() for row in build_snake_rows()}


def _geo_rows() -> dict[str, dict[str, Any]]:
    from pipelines.reporting.build_paper_i_hh_corrected_parent_comparator_page13_pdf import (
        build_corrected_row,
    )

    rows: dict[str, dict[str, Any]] = {}
    for regime, path in _geo_sources().items():
        row = build_corrected_row(
            regime,
            "geo",
            path,
            expected_iterations=50,
            plateau_horizon=DISPLAY_CROP,
            grouped_exact_max_active_qubits=GROUPED_EXACT_MAX_ACTIVE_QUBITS,
        )
        payload = row.as_dict()
        payload.update(
            completed_horizon=50,
            display_crop=DISPLAY_CROP,
            plateau_stable_under_full_horizon=True,
            strict_replay_status=(
                "blocked_prefix_theta_not_serialized; terminal runtime_seed envelope available"
            ),
        )
        rows[regime] = payload
    return rows


def _marker_y(row: Mapping[str, Any]) -> float:
    curve = {
        int(point["k"]): float(
            point["abs_delta_e"] if "abs_delta_e" in point else point["error"]
        )
        for point in row["trajectory_points"]
    }
    return float(curve[int(row["k_pl"])])


def _render_plot(
    regime: str,
    snake: Mapping[str, Any],
    geo: Mapping[str, Any],
    append: Mapping[str, Any],
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.15, 1.82))
    for method, row in (("snake", snake), ("geo", geo), ("append", append)):
        style = METHOD_STYLE[method]
        points = [
            point
            for point in row["trajectory_points"]
            if int(point["k"]) <= DISPLAY_CROP
        ]
        xs = [int(point["k"]) for point in points]
        ys = [
            max(
                float(
                    point["abs_delta_e"]
                    if "abs_delta_e" in point
                    else point["error"]
                ),
                1.0e-16,
            )
            for point in points
        ]
        ax.plot(xs, ys, color=style["color"], linewidth=style["width"], linestyle="-")
        ax.scatter(
            [int(row["k_pl"])],
            [_marker_y(row)],
            color=style["color"],
            marker=style["marker"],
            s=48 if method == "snake" else 27,
            edgecolor="black",
            linewidth=0.35,
            zorder=4,
        )
    ax.set_yscale("log")
    ax.set_xlim(0, DISPLAY_CROP)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    ax.set_xlabel("ADAPT outer iteration $k$", fontsize=7.5)
    ax.set_ylabel(r"$|\Delta E|$", fontsize=7.5)
    ax.set_title(REGIME_DISPLAY[regime], fontsize=8.5)
    ax.tick_params(axis="both", labelsize=6.5)
    ax.grid(True, which="major", alpha=0.24, linewidth=0.45)
    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_STYLE[method]["color"],
            linewidth=METHOD_STYLE[method]["width"],
            marker=METHOD_STYLE[method]["marker"],
            markersize=5,
            markeredgecolor="black",
            markeredgewidth=0.35,
            label=METHOD_STYLE[method]["label"],
        )
        for method in ("snake", "geo", "append")
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=5.8, frameon=True)
    fig.tight_layout(pad=0.55)
    stem = FIGURE_DIR / f"{STEM}__{regime.replace('-', '_')}"
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    fig.savefig(pdf, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    return {
        "png": _rel(png),
        "png_sha256": _sha256(png),
        "pdf": _rel(pdf),
        "pdf_sha256": _sha256(pdf),
    }


def _write_csv(path: Path, geo_rows: Mapping[str, Mapping[str, Any]]) -> None:
    fields = (
        "regime",
        "k_pl",
        "abs_delta_e",
        "N2q",
        "D2q",
        "Dc",
        "S_alg",
        "trajectory_point_count",
        "source_json",
        "source_sha256",
        "strict_replay_status",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for regime in REGIME_ORDER:
            row = geo_rows[regime]
            writer.writerow(
                {
                    "regime": regime,
                    "k_pl": row["k_pl"],
                    "abs_delta_e": row["abs_delta_e"],
                    "N2q": row["N2q"],
                    "D2q": row["D2q"],
                    "Dc": row["Dc"],
                    "S_alg": row["S_alg"],
                    "trajectory_point_count": len(row["trajectory_points"]),
                    "source_json": row["source_json"],
                    "source_sha256": row["source_sha256"],
                    "strict_replay_status": row["strict_replay_status"],
                }
            )


def build() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    snakes = _snake_rows()
    geos = _geo_rows()
    appends = _append_sources()
    plots: dict[str, dict[str, str]] = {}
    for regime in REGIME_ORDER:
        plots[regime] = _render_plot(regime, snakes[regime], geos[regime], appends[regime])
    payload = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "replace only completed Geo-ADAPT evidence in Paper-I L=2 HH main figure",
        "source_inventory": _rel(INVENTORY),
        "source_inventory_sha256": _sha256(INVENTORY),
        "paper_source": _rel(PAPER_TEX),
        "paper_source_sha256_before_transfer": _sha256(PAPER_TEX),
        "optimizer": "POWELL",
        "optimizer_maxiter": 200,
        "pool_policy": "parent macro generators; unfiltered full_meta; no Pauli-child split",
        "geo_immediate_repeat": "disabled",
        "completed_geo_horizon": 50,
        "main_display_crop": DISPLAY_CROP,
        "plateau_policy": "first prefix within 10 percent of best error over first 30 iterations",
        "plateau_stability": "all six selected prefixes unchanged when evaluated over all 50 iterations",
        "qiskit_compile": {
            "convention": "table_i_basis_gate_transpile_v1",
            "grouped_exact_synthesis": "commuting_pauli_or_active_support_unitary_exact_v1",
            "grouped_exact_max_active_qubits": GROUPED_EXACT_MAX_ACTIVE_QUBITS,
        },
        "method_update_status": {
            "SNAKE": "preserved",
            "Geo-ADAPT": "completed_iter50_replacement",
            "Append-ADAPT": "preserved_pending_completed_iter50_replacement",
        },
        "geo_rows": geos,
        "preserved_snake_rows": snakes,
        "preserved_append_rows": appends,
        "plots": plots,
        "known_blockers": [
            "plateau-prefix theta/state envelopes are not serialized, so normalized strict replay is blocked",
            "preserved Append marker/table rows retain known prefix-index mismatches until completed replacements arrive",
        ],
    }
    json_path = OUTPUT_DIR / f"{STEM}.json"
    csv_path = OUTPUT_DIR / f"{STEM}.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(csv_path, geos)
    payload["outputs"] = {
        "json": _rel(json_path),
        "json_sha256": _sha256(json_path),
        "csv": _rel(csv_path),
        "csv_sha256": _sha256(csv_path),
    }
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    payload = build()
    print(json.dumps(payload["outputs"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
