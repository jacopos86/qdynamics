#!/usr/bin/env python3
"""Compare validated Test-1 material-window rows with the no-overlap parent.

This is a read-only post-run audit.  It reads the preserved transfer archives
and compact fail-closed validation receipts, then emits one JSON comparison
receipt.  It never mutates scientific results or report/PDF artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pipelines.reporting.build_paper_i_hh_sr_no_prune_no_beam_tracking_pdf import (
    _extract_trajectory,
    _tar_json,
)


TARGET = 2.0e-4
OUT = (
    REPO
    / "raw_outputs/chtc_fetch_paper_i_hh_sr_material_window_20260722"
    / "test1_vs_full_geometry_no_overlap_comparison.json"
)

REGIMES = (
    "weak_weak",
    "intermediate_weak",
    "strong_weak_u8",
    "weak_strong",
    "intermediate_strong",
    "strong_strong_u8",
)

ANCHOR_ARCHIVES = {
    "weak_weak": "raw_outputs/chtc_fetch_paper_i_hh_sr_20260721/priority_20260721T0037Z/8958273.0__weak_weak_transfer.tar.gz",
    "intermediate_weak": "raw_outputs/chtc_fetch_paper_i_hh_sr_20260721/priority_20260721T0037Z/8958273.1__intermediate_weak_transfer.tar.gz",
    "strong_weak_u8": "raw_outputs/chtc_fetch_paper_i_hh_sr_20260721/priority_20260721T0037Z/8958273.2__strong_weak_u8_transfer.tar.gz",
    "weak_strong": "raw_outputs/chtc_fetch_paper_i_hh_sr_20260721/status_20260721T0732Z/8958273.3__weak_strong_transfer.tar.gz",
    "intermediate_strong": "raw_outputs/chtc_fetch_paper_i_hh_sr_20260721/status_20260721T0502Z/8958273.4__intermediate_strong_transfer.tar.gz",
    "strong_strong_u8": "raw_outputs/chtc_fetch_paper_i_hh_sr_20260721/status_20260721T0149Z/8958273.5__strong_strong_u8_transfer.tar.gz",
}

TEST1_ARCHIVES = {
    "weak_weak": "raw_outputs/chtc_fetch_paper_i_hh_sr_material_window_20260722/test1_9308516_weak_cutoff/9308516.0__weak_weak_transfer.tar.gz",
    "intermediate_weak": "raw_outputs/chtc_fetch_paper_i_hh_sr_material_window_20260722/test1_9308516_weak_cutoff/9308516.1__intermediate_weak_transfer.tar.gz",
    "strong_weak_u8": "raw_outputs/chtc_fetch_paper_i_hh_sr_material_window_20260722/test1_9308516_weak_cutoff/9308516.2__strong_weak_u8_transfer.tar.gz",
    "weak_strong": "raw_outputs/chtc_fetch_paper_i_hh_sr_material_window_20260722/test1_9308516_strong_cutoff/9308516.3__weak_strong_transfer.tar.gz",
    "intermediate_strong": "raw_outputs/chtc_fetch_paper_i_hh_sr_material_window_20260722/test1_9308516_strong_cutoff/9308516.4__intermediate_strong_transfer.tar.gz",
    "strong_strong_u8": "raw_outputs/chtc_fetch_paper_i_hh_sr_material_window_20260722/test1_9308516_strong_cutoff/9308516.5__strong_strong_u8_transfer.tar.gz",
}

ANCHOR_VALIDATIONS = {
    regime: (
        "raw_outputs/chtc_fetch_paper_i_hh_sr_20260721/"
        "no_overlap_8958273_weak_cutoff_reporting_recovery/compact_artifacts/"
        f"{regime}/validation.json"
    )
    for regime in REGIMES
}

TEST1_VALIDATIONS = {
    "weak_weak": "raw_outputs/chtc_fetch_paper_i_hh_sr_material_window_20260722/test1_9308516_weak_cutoff_reporting_recovery/compact_artifacts/weak_weak/validation.json",
    "intermediate_weak": "raw_outputs/chtc_fetch_paper_i_hh_sr_material_window_20260722/test1_9308516_weak_cutoff_reporting_recovery/compact_artifacts/intermediate_weak/validation.json",
    "strong_weak_u8": "raw_outputs/chtc_fetch_paper_i_hh_sr_material_window_20260722/test1_9308516_weak_cutoff_reporting_recovery/compact_artifacts/strong_weak_u8/validation.json",
    "weak_strong": "raw_outputs/chtc_fetch_paper_i_hh_sr_material_window_20260722/test1_9308516_strong_cutoff_reporting_recovery_ws/compact_artifacts/weak_strong/validation.json",
    "intermediate_strong": "raw_outputs/chtc_fetch_paper_i_hh_sr_material_window_20260722/test1_9308516_strong_cutoff_reporting_recovery_is/compact_artifacts/intermediate_strong/validation.json",
    "strong_strong_u8": "raw_outputs/chtc_fetch_paper_i_hh_sr_material_window_20260722/test1_9308516_strong_cutoff_reporting_recovery/compact_artifacts/strong_strong_u8/validation.json",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _first_hit(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    for point in trajectory:
        error = float(point["error"])
        if error <= TARGET:
            return {
                "hit": True,
                "round": int(point["round"]),
                "active_depth": int(point["active_depth"]),
                "error": error,
            }
    return {"hit": False, "round": None, "active_depth": None, "error": None}


def _accounting(payload: Mapping[str, Any]) -> dict[str, Any]:
    adapt = payload.get("adapt_vqe")
    if not isinstance(adapt, Mapping):
        raise RuntimeError("missing adapt_vqe")
    raw = adapt.get("estimator_call_accounting")
    if not isinstance(raw, Mapping):
        raise RuntimeError("missing estimator_call_accounting")
    keep = {
        key: value
        for key, value in raw.items()
        if key
        in {
            "all_branch_search_work",
            "winning_lineage_search_work",
            "discarded_branch_search_work",
            "shared_source_search_work",
            "component_order",
            "aggregation_policy",
        }
    }
    return json.loads(json.dumps(keep, sort_keys=True))


def _row(
    archive_rel: str,
    validation_rel: str,
    regime: str,
) -> dict[str, Any]:
    archive = REPO / archive_rel
    validation_path = REPO / validation_rel
    validation = _load(validation_path)
    if validation.get("status") != "pass":
        raise RuntimeError(f"validation did not pass: {validation_path}")
    payload, member = _tar_json(archive, f"/{regime}/json/result.json")
    trajectory = _extract_trajectory(payload)
    if trajectory["rounds"] != 50 or len(trajectory["trajectory"]) != 50:
        raise RuntimeError(f"non-50-round trajectory: {archive}")
    receipt = validation.get("post_run_projector_fidelity_receipt")
    if not isinstance(receipt, Mapping) or receipt.get("status") != "pass":
        raise RuntimeError(f"missing fidelity receipt: {validation_path}")
    qiskit = validation.get("current_fake_marrakesh_metrics")
    if not isinstance(qiskit, Mapping):
        raise RuntimeError(f"missing Qiskit metrics: {validation_path}")
    scientific = validation.get("scientific_evidence_validation")
    if not isinstance(scientific, Mapping):
        raise RuntimeError(f"missing scientific validation: {validation_path}")
    ledger = scientific.get("active_prefix_estimator_ledger_receipts")
    if not isinstance(ledger, Mapping) or ledger.get("closure_passed") is not True:
        raise RuntimeError(f"ledger closure failed: {validation_path}")
    s_alg = int(ledger["S_alg"])
    if int(trajectory["s_alg"]) != s_alg:
        raise RuntimeError(f"S_alg disagreement: {archive}")
    return {
        "archive": {
            "path": archive_rel,
            "member": member,
            "sha256": _sha(archive),
            "size_bytes": archive.stat().st_size,
        },
        "validation": {
            "path": validation_rel,
            "sha256": _sha(validation_path),
        },
        "rounds": 50,
        "active_depth": int(trajectory["active_depth"]),
        "first_hit": _first_hit(trajectory["trajectory"]),
        "terminal_error": float(trajectory["terminal_error"]),
        "fidelity": float(receipt["fidelity"]),
        "S_alg": s_alg,
        "S_alg_accounting": _accounting(payload),
        "qiskit": {
            "N2q": int(qiskit["N2q"]),
            "D2q": int(qiskit["D2q"]),
            "Dc": int(qiskit["Dc"]),
        },
        "trajectory": trajectory["trajectory"],
    }


def _delta(candidate: float, parent: float) -> dict[str, float]:
    return {
        "absolute": candidate - parent,
        "relative": (candidate - parent) / parent,
        "percent": 100.0 * (candidate - parent) / parent,
    }


def main() -> int:
    rows: dict[str, Any] = {}
    for regime in REGIMES:
        parent = _row(ANCHOR_ARCHIVES[regime], ANCHOR_VALIDATIONS[regime], regime)
        candidate = _row(TEST1_ARCHIVES[regime], TEST1_VALIDATIONS[regime], regime)
        window = _load(REPO / TEST1_VALIDATIONS[regime]).get(
            "material_window_validation"
        )
        if not isinstance(window, Mapping) or window.get("status") != "pass":
            raise RuntimeError(f"material-window closure failed: {regime}")
        rows[regime] = {
            "parent_full_geometry_no_overlap": parent,
            "candidate_material_window_no_overlap": candidate,
            "candidate_window_telemetry": dict(window),
            "comparison": {
                "first_hit_round_delta": (
                    int(candidate["first_hit"]["round"])
                    - int(parent["first_hit"]["round"])
                ),
                "terminal_error_delta": _delta(
                    candidate["terminal_error"], parent["terminal_error"]
                ),
                "fidelity_delta": candidate["fidelity"] - parent["fidelity"],
                "S_alg_delta": _delta(candidate["S_alg"], parent["S_alg"]),
                "qiskit_delta": {
                    key: int(candidate["qiskit"][key]) - int(parent["qiskit"][key])
                    for key in ("N2q", "D2q", "Dc")
                },
            },
        }
    reductions = [
        -float(rows[regime]["comparison"]["S_alg_delta"]["percent"])
        for regime in REGIMES
    ]
    payload = {
        "schema": "paper_i_hh_sr_material_window_test1_comparison_v1",
        "status": "pass",
        "target_abs_error": TARGET,
        "parent_route": "fd5ec3fa2c98b2a9d1cbcc304241d723f57dbd6210f4ea2daf30753603a146c2",
        "candidate_route": "9d6cfae3fda84eb6a24232358c45a8f25b42c6147bfa4250905910eff687a417",
        "rows": rows,
        "summary": {
            "all_six_reach_target": all(
                rows[r]["candidate_material_window_no_overlap"]["first_hit"]["hit"]
                for r in REGIMES
            ),
            "all_six_reduce_S_alg": all(
                rows[r]["comparison"]["S_alg_delta"]["absolute"] < 0
                for r in REGIMES
            ),
            "min_S_alg_reduction_percent": min(reductions),
            "max_S_alg_reduction_percent": max(reductions),
            "geometric_mean_S_alg_reduction_percent": 100.0
            * (
                1.0
                - math.prod(
                    rows[r]["candidate_material_window_no_overlap"]["S_alg"]
                    / rows[r]["parent_full_geometry_no_overlap"]["S_alg"]
                    for r in REGIMES
                )
                ** (1.0 / len(REGIMES))
            ),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(OUT.relative_to(REPO))
    print(json.dumps(payload["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
