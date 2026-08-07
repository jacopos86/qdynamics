#!/usr/bin/env python3
"""Build the Paper-I Hubbard--Holstein fixed-accuracy result support.

Target hits use the first accepted prefix satisfying ``abs(E - E_ED) <= 2e-4``.
An explicit non-hit retains that classification and reports its terminal
round-50 error and resource costs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACKER = REPO_ROOT / (
    "output/pdf/paper_i_hh_sr_snake_no_prune_no_beam_no_ordinary_novelty_"
    "tracking_20260715/paper_i_hh_sr_snake_no_prune_no_beam_no_ordinary_"
    "novelty_tracking_20260715.json"
)
DEFAULT_TARGETS = DEFAULT_TRACKER.with_name("target_energy_prefix_costs.json")
DEFAULT_S_ALG_AUDIT = REPO_ROOT / (
    "raw_outputs/chtc_fetch_paper_i_hh_sr_20260721/"
    "paper_i_no_overlap_runtime_postrun_s_alg_audit.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / (
    "MATH/paper_details/figures/paper_i_hh_first_hit_20260721"
)
TARGET = 2.0e-4

REGIMES = (
    ("weak_weak", "weak--weak", 3),
    ("intermediate_weak", "intermediate--weak", 3),
    ("strong_weak_u8", "strong--weak", 3),
    ("weak_strong", "weak--strong", 7),
    ("intermediate_strong", "intermediate--strong", 7),
    ("strong_strong_u8", "strong--strong", 7),
)

METHODS = (
    {
        "route_id": "no_overlap_trust_projected_phase3_nph3_7",
        "short": "SNAKE",
        "label": "SNAKE (support-projected Phase III; no-overlap trust)",
        "pool_exposure": "canonical SNAKE staged macro/lane-to-child route",
        "color": "#1f4e79",
        "marker": "o",
    },
    {
        "route_id": "geo_adapt_macro_nph3_7",
        "short": "Geo-M",
        "label": "Geo-ADAPT (intact macro pool)",
        "pool_exposure": "intact macro generators",
        "color": "#a23b72",
        "marker": "s",
    },
    {
        "route_id": "append_adapt_projected_singleton_nph3_7",
        "short": "Append-S",
        "label": "Append-ADAPT (symmetry-guarded projected singletons)",
        "pool_exposure": "symmetry-guarded projected-singleton children",
        "color": "#2a9d8f",
        "marker": "D",
    },
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _route_map(tracker: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    routes = tracker.get("routes")
    if not isinstance(routes, list):
        raise ValueError("tracker lacks routes")
    return {
        str(route.get("id")): route
        for route in routes
        if isinstance(route, Mapping)
    }


def _target_maps(
    payload: Mapping[str, Any],
) -> tuple[dict[tuple[str, str], Mapping[str, Any]], dict[tuple[str, str], Mapping[str, Any]]]:
    complete = {
        (str(row.get("route_id")), str(row.get("regime"))): row
        for row in payload.get("rows", [])
        if isinstance(row, Mapping) and row.get("status") == "complete"
    }
    unresolved = {
        (str(row.get("route_id")), str(row.get("regime"))): row
        for row in payload.get("unresolved", [])
        if isinstance(row, Mapping)
    }
    return complete, unresolved


def _trajectory(route: Mapping[str, Any], regime: str) -> list[dict[str, float]]:
    results = route.get("results")
    if not isinstance(results, Mapping):
        raise ValueError(f"route {route.get('id')} lacks results")
    result = results.get(regime)
    if not isinstance(result, Mapping) or result.get("status") != "complete":
        raise ValueError(f"route {route.get('id')} lacks complete {regime} result")
    points = result.get("trajectory")
    if not isinstance(points, list) or len(points) != 50:
        raise ValueError(f"route {route.get('id')} {regime} lacks 50-round trajectory")
    normalized: list[dict[str, float]] = []
    for expected_round, point in enumerate(points, start=1):
        if not isinstance(point, Mapping) or int(point.get("round") or 0) != expected_round:
            raise ValueError(f"trajectory ordering drift: {route.get('id')}/{regime}")
        normalized.append(
            {"round": float(expected_round), "error": abs(float(point["error"]))}
        )
    return normalized


def _public_row(
    method: Mapping[str, Any],
    regime: str,
    complete: Mapping[tuple[str, str], Mapping[str, Any]],
    unresolved: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    route_id = str(method["route_id"])
    key = (route_id, regime)
    if key in complete:
        row = complete[key]
        qiskit = row.get("qiskit")
        if not isinstance(qiskit, Mapping):
            raise ValueError(f"complete target row lacks Qiskit costs: {key}")
        return {
            "status": "hit",
            "k": int(row["k_target"]),
            "error": float(row["error"]),
            "S_alg": int(row["S_alg"]),
            "N2q": int(qiskit["N2q"]),
            "D2q": int(qiskit["D2q"]),
            "Dc": int(qiskit["Dc"]),
            "prefix_source": row.get("prefix_source"),
            "source": row.get("source"),
        }
    if key not in unresolved:
        raise ValueError(f"target result missing for {key}")
    row = unresolved[key]
    if row.get("status") != "threshold_not_reached":
        raise ValueError(f"target result is not objectively classified for {key}")
    terminal = row.get("terminal")
    if not isinstance(terminal, Mapping):
        raise ValueError(f"non-hit target row lacks terminal costs: {key}")
    qiskit = terminal.get("qiskit")
    if not isinstance(qiskit, Mapping):
        raise ValueError(f"non-hit terminal row lacks Qiskit costs: {key}")
    if int(terminal.get("k_target") or 0) != int(row["horizon"]):
        raise ValueError(f"non-hit terminal prefix does not match horizon: {key}")
    return {
        "status": "terminal_nonhit",
        "k": int(terminal["k_target"]),
        "error": float(terminal["error"]),
        "S_alg": int(terminal["S_alg"]),
        "N2q": int(qiskit["N2q"]),
        "D2q": int(qiskit["D2q"]),
        "Dc": int(qiskit["Dc"]),
        "best_observed_error": float(row["best_observed_error"]),
        "horizon": int(row["horizon"]),
        "prefix_source": row.get("source"),
        "source": row.get("source"),
    }


def _aggregate(rows: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    coverage = {
        method["short"]: sum(
            rows[method["short"]][regime]["status"] == "hit"
            for regime, _label, _cutoff in REGIMES
        )
        for method in METHODS
    }
    common = [
        regime
        for regime, _label, _cutoff in REGIMES
        if rows["SNAKE"][regime]["status"] == "hit"
        and rows["Append-S"][regime]["status"] == "hit"
    ]
    sums: dict[str, dict[str, int]] = {}
    for short in ("SNAKE", "Append-S"):
        sums[short] = {
            metric: sum(int(rows[short][regime][metric]) for regime in common)
            for metric in ("N2q", "D2q", "Dc", "S_alg")
        }
    reductions = {
        metric: 100.0 * (1.0 - sums["SNAKE"][metric] / sums["Append-S"][metric])
        for metric in sums["SNAKE"]
    }
    return {
        "target_hit_count_of_six": coverage,
        "snake_vs_append_projected_singleton": {
            "common_hit_regimes": common,
            "aggregate_sums": sums,
            "snake_reduction_percent": reductions,
            "aggregation": "sum over the five regimes reached by both methods",
        },
    }


def _validate_s_alg_audit(
    payload: Mapping[str, Any],
    *,
    rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Mapping[str, Any]]:
    if (
        payload.get("schema") != "paper_i_runtime_postrun_s_alg_closure_v1"
        or payload.get("status") != "pass"
        or payload.get("route_id")
        != "no_overlap_trust_projected_phase3_nph3_7"
    ):
        raise ValueError("SNAKE runtime/post-run S_alg audit is not closed")
    audit_rows = payload.get("rows")
    if not isinstance(audit_rows, list):
        raise ValueError("SNAKE runtime/post-run S_alg audit lacks rows")
    by_regime = {
        str(row.get("regime")): row
        for row in audit_rows
        if isinstance(row, Mapping)
    }
    expected_regimes = {regime for regime, _label, _cutoff in REGIMES}
    if set(by_regime) != expected_regimes:
        raise ValueError("SNAKE runtime/post-run S_alg audit regime drift")
    for regime in expected_regimes:
        audit_row = by_regime[regime]
        public_row = rows["SNAKE"][regime]
        closure = audit_row.get("closure")
        if (
            audit_row.get("status") != "pass"
            or int(audit_row.get("S_alg") or -1)
            != int(public_row["S_alg"])
            or not isinstance(closure, Mapping)
            or closure.get("runtime_equals_postrun") is not True
            or closure.get("postrun_equals_paper_target_prefix") is not True
            or closure.get("componentwise_equal") is not True
            or closure.get("source_archive_hash_matched") is not True
        ):
            raise ValueError(
                f"SNAKE runtime/post-run S_alg audit mismatch for {regime}"
            )
    return by_regime


def build(
    tracker_path: Path,
    target_path: Path,
    output_dir: Path,
    *,
    s_alg_audit_path: Path = DEFAULT_S_ALG_AUDIT,
) -> dict[str, Path]:
    tracker_path = tracker_path.resolve()
    target_path = target_path.resolve()
    s_alg_audit_path = s_alg_audit_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = _read_json(tracker_path)
    target = _read_json(target_path)
    s_alg_audit = _read_json(s_alg_audit_path)
    route_map = _route_map(tracker)
    complete, unresolved = _target_maps(target)

    rows: dict[str, dict[str, dict[str, Any]]] = {}
    trajectories: dict[str, dict[str, list[dict[str, float]]]] = {}
    for method in METHODS:
        short = str(method["short"])
        route_id = str(method["route_id"])
        route = route_map.get(route_id)
        if route is None:
            raise ValueError(f"tracker lacks route {route_id}")
        rows[short] = {}
        trajectories[short] = {}
        for regime, _label, _cutoff in REGIMES:
            rows[short][regime] = _public_row(method, regime, complete, unresolved)
            trajectories[short][regime] = _trajectory(route, regime)
    audited_s_alg_rows = _validate_s_alg_audit(s_alg_audit, rows=rows)
    for regime, audit_row in audited_s_alg_rows.items():
        rows["SNAKE"][regime]["S_alg_runtime_postrun_audit"] = {
            "status": "pass",
            "runtime_receipt_sha256": audit_row["runtime_receipt_sha256"],
            "postrun_summary_sha256": audit_row["postrun_summary_sha256"],
            "primitive_set_sha256": audit_row["primitive_set_sha256"],
        }

    fig, axes = plt.subplots(2, 3, figsize=(10.4, 5.9), sharex=True, sharey=True)
    for axis, (regime, label, cutoff) in zip(axes.flat, REGIMES, strict=True):
        for method in METHODS:
            short = str(method["short"])
            points = trajectories[short][regime]
            x = [point["round"] for point in points]
            y = [max(point["error"], 1.0e-14) for point in points]
            axis.plot(
                x,
                y,
                color=method["color"],
                linewidth=1.35,
                label=short,
            )
            result = rows[short][regime]
            if result["status"] == "hit":
                axis.scatter(
                    [result["k"]],
                    [result["error"]],
                    color=method["color"],
                    marker=method["marker"],
                    s=26,
                    zorder=4,
                    edgecolors="white",
                    linewidths=0.45,
                )
            else:
                axis.scatter(
                    [50],
                    [y[-1]],
                    facecolors="none",
                    edgecolors=method["color"],
                    marker=method["marker"],
                    s=24,
                    zorder=4,
                    linewidths=0.8,
                )
        axis.axhline(TARGET, color="#555555", linestyle="--", linewidth=0.9)
        axis.set_yscale("log")
        axis.set_xlim(1, 50)
        axis.set_ylim(1.0e-10, 2.0)
        axis.grid(True, which="both", alpha=0.18, linewidth=0.5)
        axis.set_title(f"{label}; $M={cutoff}$", fontsize=9)
    for axis in axes[1, :]:
        axis.set_xlabel("accepted adaptive iteration $k$", fontsize=8.5)
    for axis in axes[:, 0]:
        axis.set_ylabel(r"same-cutoff $|\Delta E_k|$", fontsize=8.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(METHODS),
        frameon=False,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=0.7, h_pad=0.9)
    figure_pdf = output_dir / "paper_i_hh_first_hit_trajectories_20260721.pdf"
    figure_png = output_dir / "paper_i_hh_first_hit_trajectories_20260721.png"
    fig.savefig(figure_pdf, bbox_inches="tight")
    fig.savefig(figure_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    support = {
        "schema": "paper_i_hh_first_hit_results_support_v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reporting_rule": {
            "target_abs_error": TARGET,
            "prefix": "first accepted adaptive iteration at or below target",
            "non_hit_policy": (
                "retain non-hit classification and report terminal k=50 error "
                "and resource costs"
            ),
        },
        "methods": [
            {
                key: method[key]
                for key in ("route_id", "short", "label", "pool_exposure")
            }
            for method in METHODS
        ],
        "regimes": [
            {"id": regime, "label": label, "n_ph_max": cutoff}
            for regime, label, cutoff in REGIMES
        ],
        "rows": rows,
        "aggregates": _aggregate(rows),
        "sources": {
            "tracker": {"path": str(tracker_path), "sha256": _sha256(tracker_path)},
            "target_prefixes": {"path": str(target_path), "sha256": _sha256(target_path)},
            "snake_runtime_postrun_s_alg_audit": {
                "path": str(s_alg_audit_path),
                "sha256": _sha256(s_alg_audit_path),
                "schema": s_alg_audit["schema"],
                "status": s_alg_audit["status"],
            },
        },
        "figure": {
            "pdf": {"path": str(figure_pdf), "sha256": _sha256(figure_pdf)},
            "png": {"path": str(figure_png), "sha256": _sha256(figure_png)},
        },
    }
    support_json = output_dir / "paper_i_hh_first_hit_results_support_20260721.json"
    support_json.write_text(json.dumps(support, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"support_json": support_json, "figure_pdf": figure_pdf, "figure_png": figure_png}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker-json", type=Path, default=DEFAULT_TRACKER)
    parser.add_argument("--target-json", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--s-alg-audit-json",
        type=Path,
        default=DEFAULT_S_ALG_AUDIT,
    )
    args = parser.parse_args()
    outputs = build(
        args.tracker_json,
        args.target_json,
        args.output_dir,
        s_alg_audit_path=args.s_alg_audit_json,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
