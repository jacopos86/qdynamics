#!/usr/bin/env python3
"""Build the six-regime joint-response/Paper-I diagnostic overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCHEMA = "paper_i_hh_joint_response_six_regime_overlay_v8"
JR_CHTC_LIVE_SNAPSHOT_SCHEMA = "jr_snake_chtc_live_snapshot_bundle_v1"
JR_CHTC_LIVE_STATUSES = frozenset(
    (
        "running_snapshot",
        "held_snapshot",
        "stopped_snapshot",
        "recovery_queued_snapshot",
        "completed_snapshot_pending_qiskit",
    )
)
FM_LIVE_SNAPSHOT_SCHEMA = "formal_manifold_live_snapshot_bundle_v1"
FM_LIVE_STATUS_SCHEMA = "formal_manifold_lightweight_status_snapshot_v1"
FM_STOPPED_SNAPSHOT_SCHEMA = "formal_manifold_stop_retrieval_manifest_v1"
FM_COMPLETED_RESOURCE_RECOVERY_SCHEMA = (
    "formal_manifold_completed_weak_resource_recovery_manifest_v1"
)
FM_LIVE_ROUTE_ID = "formal_manifold_warm_start_v1"
PAPER_I_ROUTE4_LIVE_SNAPSHOT_SCHEMA = "paper_i_route4_live_snapshot_bundle_v1"
PAPER_I_SR_RECOVERY_SNAPSHOT_SCHEMA = "paper_i_sr_snake_recovery_snapshot_bundle_v2"
PAPER_I_ROUTE4_LIVE_STATUS = "running_checkpoint_not_terminal"
PAPER_I_ROUTE4_STOPPED_STATUS = "stopped_checkpoint_not_terminal"
PAPER_I_ROUTE4_NONTERMINAL_STATUSES = frozenset(
    (PAPER_I_ROUTE4_LIVE_STATUS, PAPER_I_ROUTE4_STOPPED_STATUS)
)
PAPER_I_SR_TERMINAL_STATUS = "validated_terminal_recovery"
SR_EXPANDED_CHART_WHITENING_VALIDATION_SCHEMA = (
    "paper_i_hh_sr_expanded_chart_whitening_validation_v1"
)
SR_EXPANDED_CHART_WHITENING_PAGE_KEY = "sr_expanded_chart_whitening"
SR_COMPLETED_RUN_VALIDATION_SCHEMA = "paper_i_hh_sr_completed_run_validation_v1"
SR_EXPANDED_CHART_WHITENING_IW_PAGE_KEY = (
    "sr_expanded_chart_whitening_intermediate_weak"
)
SR_EXPANDED_CHART_WHITENING_IW_ROLE = (
    "sr_expanded_whitened_intermediate_weak_validated"
)
FM_LIVE_STATUSES = frozenset(
    (
        "running_snapshot",
        "failed_partial",
        "running_status_endpoint",
        "science_complete_packaging_failed",
        "stopped_snapshot",
    )
)
PENDING_RESOURCE_STATUSES = frozenset(
    (*FM_LIVE_STATUSES, *JR_CHTC_LIVE_STATUSES, *PAPER_I_ROUTE4_NONTERMINAL_STATUSES)
)
PARTIAL_RESOURCE_STATUSES = frozenset(
    (*PENDING_RESOURCE_STATUSES, "failed_partial_round21")
)
STEM = "paper_i_hh_joint_response_six_regime_overlay_20260711"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output/pdf/paper_i_hh_joint_response_six_regime_overlay_20260711"
CAMPAIGN_ROOT = REPO_ROOT / "raw_outputs/paper_i_hh_joint_response_six_regime_20260711"
COMPARISON_JSON = CAMPAIGN_ROOT / "comparison/six_regime_comparison.json"
QISKIT_TABLE_JSON = REPO_ROOT / (
    "output/pdf/paper_i_hh_joint_response_qiskit_tables_six_regime_20260712/"
    "paper_i_hh_joint_response_qiskit_tables_six_regime_20260712.json"
)
WAVE11_WEAK_WEAK_JSON = REPO_ROOT / (
    "raw_outputs/paper_i_hh_joint_selector_pareto_20260711/"
    "wave11_small_m64_m48_c128_c64_b2_l25_r15/current.json"
)
JR_L10_CAMPAIGN_ROOT = REPO_ROOT / (
    "raw_outputs/paper_i_hh_jr_snake_whitened_pareto_goal_20260711"
)
JR_L10_WEAK_WEAK_SEGMENTS = (
    JR_L10_CAMPAIGN_ROOT
    / "weak-weak/canonical_m32_m24_c32_c25_l10_b2_r7/current.json",
    JR_L10_CAMPAIGN_ROOT
    / (
        "l10_long_horizon_queue_r30/"
        "preserved_interrupted_weak_weak_r17_20260712/current.json"
    ),
    JR_L10_CAMPAIGN_ROOT
    / "weak-weak/l10_b2_continuation_resume1_to_r30/current.json",
)
JR_L10_QUEUE_STATUS = (
    JR_L10_CAMPAIGN_ROOT
    / "l10_long_horizon_queue_r30_resume1/queue_status.json"
)
JR_L10_SEGMENTS_BY_REGIME = {
    "weak-weak": (
        JR_L10_CAMPAIGN_ROOT
        / "weak-weak/canonical_m32_m24_c32_c25_l10_b2_r7/result.json",
        JR_L10_CAMPAIGN_ROOT
        / (
            "l10_long_horizon_queue_r30/"
            "preserved_interrupted_weak_weak_r17_20260712/current.json"
        ),
        JR_L10_CAMPAIGN_ROOT
        / "weak-weak/l10_b2_continuation_resume1_to_r30/result.json",
    ),
    "intermediate-weak": (
        JR_L10_CAMPAIGN_ROOT
        / "intermediate-weak/canonical_m32_m24_c32_c25_l10_b2_r7/result.json",
        JR_L10_CAMPAIGN_ROOT
        / "intermediate-weak/l10_b2_continuation_to_r30/result.json",
    ),
    "strong-weak": (
        JR_L10_CAMPAIGN_ROOT
        / "strong-weak-u8/canonical_m32_m24_c32_c25_l10_b2_r9/result.json",
        JR_L10_CAMPAIGN_ROOT
        / "strong-weak-u8/l10_b2_continuation_to_r30/result.json",
    ),
    "weak-strong": (
        JR_L10_CAMPAIGN_ROOT
        / "weak-strong/canonical_m32_m24_c32_c25_l10_b2_r9/result.json",
    ),
    "intermediate-strong": (
        JR_L10_CAMPAIGN_ROOT
        / "intermediate-strong/canonical_m32_m24_c32_c25_l10_b2_r9/result.json",
    ),
    "strong-strong": (
        JR_L10_CAMPAIGN_ROOT
        / "strong-strong-u8/canonical_m32_m24_c32_c25_l10_b2_r9/result.json",
    ),
}
PAPER_I_REFERENCE_JSON = REPO_ROOT / (
    "output/pdf/paper_i_hh_corrected_vs_current_20260710/"
    "paper_i_hh_corrected_vs_current_onepage_20260710.json"
)
FM_CAMPAIGN_ROOT = Path(
    "/Users/jakestrobel/local_repos/Holstein_test_fullclone_3_fm_snake/"
    "raw_outputs/paper_i_hh_fm_snake_pareto_goal_20260712"
)
ROUTE4_WEAK_WEAK_ROOT = REPO_ROOT / (
    "raw_outputs/paper_i_hh_weak_weak_route4_whitened_adaptive_"
    "geometry_expansion_repair_20260712"
)
ROUTE4_MATRIX_ROOT = REPO_ROOT / (
    "raw_outputs/paper_i_hh_new_paper_i_route4_two_stage_20260712"
)
ROUTE4_SOURCE_LOCK_DIFF = (
    ROUTE4_MATRIX_ROOT / "comparison/source_lock_and_settings_diff.json"
)

REGIME_ORDER = (
    "weak-weak",
    "intermediate-weak",
    "strong-weak",
    "weak-strong",
    "intermediate-strong",
    "strong-strong",
)
REGIME_DISPLAY = {
    "weak-weak": "Weak Hubbard / weak Holstein",
    "intermediate-weak": "Intermediate Hubbard / weak Holstein",
    "strong-weak": "Strong Hubbard / weak Holstein",
    "weak-strong": "Weak Hubbard / strong Holstein",
    "intermediate-strong": "Intermediate Hubbard / strong Holstein",
    "strong-strong": "Strong Hubbard / strong Holstein",
}
CAMPAIGN_DIR = {
    "weak-weak": "weak-weak",
    "intermediate-weak": "intermediate-weak",
    "strong-weak": "strong-weak-u8",
    "weak-strong": "weak-strong",
    "intermediate-strong": "intermediate-strong",
    "strong-strong": "strong-strong-u8",
}
FM_CELL_ID = {
    "weak-weak": "weak_weak__inverse_rbfgs_qbroyd_off__r7",
    "intermediate-weak": "intermediate_weak__inverse_rbfgs_qbroyd_off__r7",
    "strong-weak": "strong_weak_u8__inverse_rbfgs_qbroyd_off__r9",
    "weak-strong": "weak_strong__inverse_rbfgs_qbroyd_off__r9",
    "intermediate-strong": "intermediate_strong__inverse_rbfgs_qbroyd_off__r9",
    "strong-strong": "strong_strong_u8__inverse_rbfgs_qbroyd_off__r9",
}
PAPER_METHODS = ("snake", "geo", "append")
RESOURCE_METHODS = ("joint_response_snake", "snake", "geo", "append")
RESOURCE_METHOD_DISPLAY = {
    "jr_snake_whitened_l10": "JR-L10",
    "jr_snake_whitened_l10_live": "JR-L10 live",
    "jr_snake_chtc_live": "JR CHTC",
    "joint_response_snake": "JR-SNAKE",
    "prior_ledger_snake": "Prior ledger",
    "weak_weak_wave11_l25": "Early L25",
    "snake": "Paper-I SNAKE",
    "geo": "Geo-ADAPT",
    "append": "Append-ADAPT",
    "formal_manifold_snake": "FM-SNAKE",
    "fm_qbroyd_default": "FM qB on",
    "fm_qbroyd_off": "FM qB off",
    "fm_qbroyd_on_prior": "FM qB on prior terminal",
    "fm_qbroyd_off_prior": "FM qB off prior terminal",
    "repaired_l25_snake": "JR-L25 repaired",
    "paper_i_route4_snake": "SR-SNAKE",
    "paper_i_route4_live_checkpoint_snake": "SR recovery",
    SR_EXPANDED_CHART_WHITENING_IW_ROLE: "SR expanded IW diag.",
}
RESOURCE_METHOD_STYLE = {
    "jr_snake_whitened_l10_live": "l10_live",
    "jr_snake_chtc_live": "jr_chtc_live",
    "joint_response_snake": "current",
    "prior_ledger_snake": "prior",
    "weak_weak_wave11_l25": "early_l25",
    "snake": "snake",
    "geo": "geo",
    "append": "append",
    "paper_i_route4_snake": "paper_i_route4",
    "paper_i_route4_live_checkpoint_snake": "paper_i_route4_live",
    SR_EXPANDED_CHART_WHITENING_IW_ROLE: SR_EXPANDED_CHART_WHITENING_IW_ROLE,
    "formal_manifold_snake": "manifold",
}
STYLE = {
    "paper_i_route4": {
        "label": "SR-SNAKE",
        "color": "#9C3A6A",
        "marker": "h",
        "width": 1.95,
    },
    "paper_i_route4_live": {
        "label": "SR-SNAKE recovery/snapshot",
        "color": "#9C3A6A",
        "marker": "H",
        "width": 1.25,
    },
    SR_EXPANDED_CHART_WHITENING_IW_ROLE: {
        "label": "SR expanded-chart IW diagnostic (validated)",
        "color": "#5E1742",
        "marker": "8",
        "width": 2.05,
        "linestyle": "--",
    },
    "jr_selected": {"label": "JR-SNAKE selected L10", "color": "#008C95", "marker": "s", "width": 1.9},
    "jr_chtc_live": {
        "label": "JR-SNAKE rollback-free CHTC snapshot",
        "color": "#005F66",
        "marker": "o",
        "width": 1.75,
        "linestyle": "--",
    },
    "jr_baseline": {"label": "JR-SNAKE L15 baseline", "color": "#111111", "marker": "P", "width": 1.65},
    "jr_prior": {"label": "Prior-ledger SNAKE", "color": "#7A7A7A", "marker": "X", "width": 1.3},
    "jr_l25": {"label": "Repaired L25 fixed policy", "color": "#8F63A8", "marker": "v", "width": 1.75},
    "jr_early_l25": {"label": "Early L25 diagnostic", "color": "#D55E91", "marker": "d", "width": 1.3},
    "fm_qbroyd_default": {"label": "FM qB on", "color": "#ECA82C", "marker": "o", "width": 1.55},
    "fm_qbroyd_off": {"label": "FM qB off", "color": "#F58518", "marker": "D", "width": 1.75},
    "fm_qbroyd_on_prior": {"label": "FM qB on prior terminal", "color": "#F2C86D", "marker": "o", "width": 1.05},
    "fm_qbroyd_off_prior": {"label": "FM qB off prior terminal", "color": "#F9AE72", "marker": "D", "width": 1.05},
    "l10_live": {"label": "Whitened L10 JR-SNAKE (live)", "color": "#008C95", "marker": "s", "width": 1.9},
    "current": {"label": "Current joint-response SNAKE", "color": "#111111", "marker": "P", "width": 1.8},
    "prior": {"label": "Prior ledger SNAKE", "color": "#7A7A7A", "marker": "X", "width": 1.25},
    "snake": {"label": "Paper-I SNAKE", "color": "#E45756", "marker": "*", "width": 1.45},
    "geo": {"label": "Paper-I Geo-ADAPT", "color": "#54A24B", "marker": "^", "width": 1.2},
    "append": {"label": "Paper-I Append-ADAPT", "color": "#4C78A8", "marker": "o", "width": 1.2},
    "manifold": {"label": "Formal-manifold SNAKE", "color": "#F58518", "marker": "D", "width": 1.55},
    "early_l25": {"label": "Early L25 SNAKE (weak-weak)", "color": "#B279A2", "marker": "v", "width": 1.45},
}

RUN_SETTING_LEDGER = (
    {
        "curve": "Plum: SR-SNAKE (singleton-response SNAKE)",
        "selection": "physical lanes P1/P2, then symmetry/padding-guarded singleton Pauli children; B3 beam; no batching",
        "rho": "displacement-calibrated adaptive, zero lower bound; no fixed upper bound",
        "linear_solve": "supported-metric whitened eigensolve",
        "warm_start": "exact-Hessian Schur seed; bounded all-infeasible geometry expansion",
        "optimizer": "Powell 200; horizon 30 in preserved rows; recovery target 45 when a live bundle is supplied",
        "caveat": "Completed weak-weak/intermediate-weak/strong-weak rows are fixed preserved endpoints. Weak-strong and intermediate-strong retain their reconstructed round-21 rows; recovery evidence is additive. Supplied recovery endpoints retain per-entry running/stopped/validated status and explicit S_alg scope. No admission rollback, Phase 0, or batching.",
        "evidence_status": "preserved_endpoints_plus_hash_validated_mixed_recovery_evidence",
    },
    {
        "curve": "Teal: selected L10 JR-SNAKE",
        "selection": "Child P1/P2; joint B2/L10",
        "rho": "adaptive, zero lower bound; no fixed upper bound",
        "linear_solve": "supported-metric whitening",
        "warm_start": "exact guarded joint Schur seed",
        "optimizer": "Powell 50; maxfev 200",
        "caveat": "Validated rounds by regime are 30/30/11/9/9/9. Weak-weak crosses a recorded source boundary after round 17; strong-weak stopped by selector exhaustion. Historical segments exposed structural rollback but recorded no rollback event.",
        "evidence_status": "validated_terminal_or_stitched_endpoints_with_explicit_boundaries",
    },
    {
        "curve": "Dark teal dashed: rollback-free CHTC L10 JR-SNAKE",
        "selection": "Child P1/P2; joint B2/L10",
        "rho": "final selector adaptive unbounded-v2; Phase 1/2 static rho=0.25",
        "linear_solve": "supported-metric whitening",
        "warm_start": "exact guarded joint Schur seed",
        "optimizer": "Powell maxfev 200; recovery target round 30",
        "caveat": "Hash-validated clusters 8776170/8778421/8779013. Weak-strong and strong-strong are stopped round-29 fixed prefixes; intermediate-strong completed round 30. Validated fixed-prefix Qiskit and stitched winning-lineage S are shown where supplied. These archived runs predate unified-rho plumbing: only the final selector used adaptive rho; Phase 1/2 retained static rho=0.25. Structural rollback is disabled.",
        "evidence_status": "mixed_terminal_and_stopped_prefix_evidence_with_validated_resource_sidecars",
    },
    {
        "curve": "Black: L15 JR baseline",
        "selection": "Child P1/P2; joint B2/L15",
        "rho": "fixed rho=0.25",
        "linear_solve": "pre-whitening legacy",
        "warm_start": "off",
        "optimizer": "Powell 50; maxfev 200",
        "caveat": "Phase-II joint response with projected children; predates whitening and adaptive rho.",
        "evidence_status": "legacy_default_reconstructed_from_plan_and_absent_update_seed_telemetry",
    },
    {
        "curve": "Orange: FM qB-off",
        "selection": "Paper-I SNAKE selection; separate FM reoptimization",
        "rho": "trust-radius controller",
        "linear_solve": "supported-metric whitening; inverse-RBFGS",
        "warm_start": "manifold transport; qB off",
        "optimizer": "Formal-manifold route; Powell-compatible guards",
        "caveat": "Completed qB-off rows are plotted and unfinished regimes remain explicit. FM transactional state handling is route-local and is not JR structural rollback.",
        "evidence_status": "validated_formal_manifold_terminal_sidecars_or_explicit_pending",
    },
    {
        "curve": "Gray: prior-ledger SNAKE",
        "selection": "Mostly B2/L15; one singleton/all",
        "rho": "fixed rho=0.25",
        "linear_solve": "pre-whitening legacy",
        "warm_start": "off",
        "optimizer": "Powell 50; maxfev 200",
        "caveat": "Regime-dependent baseline; weak-strong is an interrupted r4 checkpoint; no intermediate-strong row.",
        "evidence_status": "legacy_default_reconstructed_from_plan_and_absent_update_seed_telemetry",
    },
    {
        "curve": "Purple: repaired L25 JR-SNAKE",
        "selection": "M64/M48; C128/C64; B2/L25",
        "rho": "campaign-locked fixed policy",
        "linear_solve": "joint-response campaign route",
        "warm_start": "off",
        "optimizer": "Powell 50; maxfev uncapped",
        "caveat": "Six validated R15-capped rows from repaired cluster 8775444; three regimes exhausted earlier. This is distinct from removed cluster 8775666.",
        "evidence_status": "validated_repaired_l25_six_regime_campaign",
    },
    {
        "curve": "Pink: early weak-weak L25",
        "selection": "Wide M/C funnel; joint B2/L25",
        "rho": "fixed rho=0.25",
        "linear_solve": "pre-projection legacy",
        "warm_start": "off",
        "optimizer": "Powell 50; maxfev uncapped",
        "caveat": "Weak-weak-only r13 diagnostic; predates projected child-padding repair and has no validated winning-lineage S.",
        "evidence_status": "historical_diagnostic",
    },
    {
        "curve": "Red: Paper-I SNAKE",
        "selection": "Physical lanes; batching off",
        "rho": "fixed rho=0.25",
        "linear_solve": "reduced geometry; no joint batch solve",
        "warm_start": "n/a",
        "optimizer": "Powell maxiter 200; maxfev uncapped",
        "caveat": "Locked visible plateau/source trajectory; different architecture from JR-SNAKE.",
        "evidence_status": "source_locked_result_and_reference_bundle",
    },
    {
        "curve": "Green: Geo-ADAPT",
        "selection": "Geo-ADAPT comparator",
        "rho": "n/a to JR policy",
        "linear_solve": "no JR joint solve",
        "warm_start": "n/a",
        "optimizer": "Powell maxiter 200",
        "caveat": "Matched comparator with a different operator-selection algorithm.",
        "evidence_status": "source_locked_comparator_bundle",
    },
    {
        "curve": "Blue: Append-ADAPT",
        "selection": "Append-only comparator",
        "rho": "n/a to JR policy",
        "linear_solve": "no JR joint solve",
        "warm_start": "n/a",
        "optimizer": "Powell maxiter 200",
        "caveat": "Matched comparator; no insertion-position or joint-response selection.",
        "evidence_status": "source_locked_comparator_bundle",
    },
)


@dataclass(frozen=True)
class Curve:
    role: str
    points: tuple[tuple[int, float], ...]
    marker_k: int
    marker_error: float
    source_json: str
    source_sha256: str
    source_segments: tuple[Mapping[str, Any], ...] = ()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _positive(value: Any) -> float:
    result = abs(float(value))
    return max(result, 1.0e-16)


def _complete_history(payload: Mapping[str, Any], *, path: Path) -> list[Mapping[str, Any]]:
    adapt = payload["adapt_vqe"]
    raw_history = adapt.get("history")
    history = (
        list(raw_history)
        if isinstance(raw_history, Sequence) and not isinstance(raw_history, (str, bytes))
        else []
    )
    if not history:
        tail = adapt.get("history_tail")
        try:
            history_count = int(adapt.get("history_count"))
            tail_count = int(adapt.get("history_tail_count"))
        except (TypeError, ValueError):
            history_count = -1
            tail_count = -1
        if (
            isinstance(tail, Sequence)
            and not isinstance(tail, (str, bytes))
            and history_count == tail_count == len(tail)
            and history_count > 0
        ):
            history = list(tail)
    if not history or not all(isinstance(row, Mapping) for row in history):
        raise ValueError(f"Missing complete ADAPT history: {path}")
    return history


def _history_curve(
    path: Path,
    *,
    role: str,
    marker_k: int | None = None,
    marker_error: float | None = None,
) -> Curve:
    payload = _read_json(path)
    adapt = payload["adapt_vqe"]
    history = _complete_history(payload, path=path)
    exact = float(adapt["exact_gs_energy"])
    initial_error = _positive(float(history[0]["energy_before_opt"]) - exact)
    points = [(0, initial_error)]
    points.extend(
        (index, _positive(row["delta_abs_current"]))
        for index, row in enumerate(history, start=1)
    )
    if adapt.get("abs_delta_e") is not None:
        points[-1] = (points[-1][0], _positive(adapt["abs_delta_e"]))
    if marker_k is None:
        marker_k = points[-1][0]
    if marker_error is None:
        marker_error = dict(points).get(marker_k)
    if marker_error is None:
        raise ValueError(f"Marker k={marker_k} is absent from curve: {path}")
    return Curve(
        role=role,
        points=tuple(points),
        marker_k=marker_k,
        marker_error=marker_error,
        source_json=_rel(path),
        source_sha256=_sha256(path),
    )


def _require_hash_link(
    path_value: Any,
    sha256_value: Any,
    *,
    context: str,
) -> Path:
    """Resolve and verify one immutable path/hash pair from an evidence bundle."""

    raw_path = str(path_value or "").strip()
    expected_sha256 = str(sha256_value or "").strip().lower()
    if not raw_path or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError(f"{context} is missing a complete path/SHA-256 link")
    path = _repo_path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{context} SHA-256 mismatch: expected {expected_sha256}, "
            f"found {actual_sha256}"
        )
    return path


def _sr_whitening_checkpoint_curve(
    path: Path,
    *,
    label: str,
    finalized_error: float,
) -> dict[str, Any]:
    """Load raw controller checkpoints without replacing the last point by terminal work."""

    payload = _read_json(path)
    adapt = payload.get("adapt_vqe")
    if not isinstance(adapt, Mapping):
        raise ValueError(f"SR whitening comparison has no adapt_vqe block: {path}")
    history = _complete_history(payload, path=path)
    exact = float(adapt["exact_gs_energy"])
    points = [
        {
            "k": 0,
            "error": _positive(float(history[0]["energy_before_opt"]) - exact),
        }
    ]
    points.extend(
        {
            "k": index,
            "error": _positive(row["delta_abs_current"]),
        }
        for index, row in enumerate(history, start=1)
    )
    observed_finalized = _positive(adapt["abs_delta_e"])
    if not math.isclose(
        observed_finalized,
        finalized_error,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError(
            f"SR whitening comparison finalized-error mismatch for {label}: "
            f"bundle={finalized_error!r}, result={observed_finalized!r}"
        )
    return {
        "label": label,
        "points": points,
        "rounds": len(history),
        "preterminal_checkpoint_error": float(points[-1]["error"]),
        "finalized_error": observed_finalized,
        "result_json": _rel(path),
        "result_sha256": _sha256(path),
    }


def _load_sr_expanded_chart_whitening_validation(
    path: Path,
) -> dict[str, Any]:
    """Load the hash-closed weak--weak expanded-chart whitening validation."""

    path = path.resolve()
    payload = _read_json(path)
    if payload.get("schema") != SR_EXPANDED_CHART_WHITENING_VALIDATION_SCHEMA:
        raise ValueError(
            "Unexpected SR expanded-chart whitening validation schema: "
            f"{payload.get('schema')!r}"
        )
    if payload.get("regime") != "weak-weak":
        raise ValueError("SR expanded-chart whitening page accepts weak-weak evidence only")
    if payload.get("status") != "validated_with_terminal_action_disclosure":
        raise ValueError("SR expanded-chart whitening validation is not terminally disclosed")

    route = payload.get("route")
    if not isinstance(route, Mapping):
        raise ValueError("SR expanded-chart whitening validation has no route block")
    expected_route = {
        "family": "singleton_response_snake",
        "profile": "supported_whitened_adaptive_trust_v1",
        "powell_base_chart": "expanded_runtime_projected_logical_v1",
        "accepted_refit_coordinate_chart": "supported_fs_whitened_fixed_v1",
    }
    for key, expected in expected_route.items():
        if route.get(key) != expected:
            raise ValueError(
                f"SR expanded-chart whitening route mismatch for {key}: "
                f"expected {expected!r}, found {route.get(key)!r}"
            )
    if route.get("all_history_rows_used_expected_base_chart") is not True:
        raise ValueError("SR expanded-chart whitening base-chart audit did not pass")
    if route.get("all_history_rows_used_expected_whitening") is not True:
        raise ValueError("SR expanded-chart whitening policy audit did not pass")

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("SR expanded-chart whitening validation has no artifact ledger")
    verified_artifacts: dict[str, dict[str, str]] = {}
    for key, raw_value in artifacts.items():
        if not key.endswith("_path"):
            continue
        stem = key.removesuffix("_path")
        linked_path = _require_hash_link(
            raw_value,
            artifacts.get(f"{stem}_sha256"),
            context=f"SR expanded-chart whitening artifact {stem}",
        )
        verified_artifacts[stem] = {
            "path": _rel(linked_path),
            "sha256": _sha256(linked_path),
        }

    raw_comparisons = payload.get("comparisons")
    if not isinstance(raw_comparisons, Sequence) or isinstance(
        raw_comparisons, (str, bytes)
    ):
        raise ValueError("SR expanded-chart whitening comparison rows are absent")
    expected_labels = (
        "historical_high_accuracy_sr_baseline",
        "wrong_reduced_chart_whitened_r22",
        "good_expanded_chart_whitened_r22",
        "good_expanded_chart_whitened_r30",
    )
    by_label = {
        str(row.get("label")): row
        for row in raw_comparisons
        if isinstance(row, Mapping)
    }
    if set(by_label) != set(expected_labels):
        raise ValueError(
            "SR expanded-chart whitening comparison labels differ from the "
            f"four-row contract: {sorted(by_label)!r}"
        )
    comparisons: list[dict[str, Any]] = []
    result_payloads: dict[str, Mapping[str, Any]] = {}
    for label in expected_labels:
        row = by_label[label]
        result_path = _require_hash_link(
            row.get("result_path"),
            row.get("result_sha256"),
            context=f"SR expanded-chart whitening comparison {label}",
        )
        finalized_error = float(row["finalized_abs_error"])
        curve = _sr_whitening_checkpoint_curve(
            result_path,
            label=label,
            finalized_error=finalized_error,
        )
        if int(row["rounds"]) != int(curve["rounds"]):
            raise ValueError(f"SR whitening round-count mismatch for {label}")
        if row.get("pre_terminal_checkpoint_abs_error") is not None and not math.isclose(
            float(row["pre_terminal_checkpoint_abs_error"]),
            float(curve["preterminal_checkpoint_error"]),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ValueError(f"SR whitening preterminal-error mismatch for {label}")
        expected_chart = (
            "logical_shared_reduced_v1"
            if label == "wrong_reduced_chart_whitened_r22"
            else "expanded_runtime_projected_logical_v1"
        )
        if row.get("powell_base_chart") != expected_chart:
            raise ValueError(f"SR whitening Powell-chart mismatch for {label}")
        curve["powell_base_chart"] = expected_chart
        comparisons.append(curve)
        result_payloads[label] = _read_json(result_path)

    main_result_artifact = verified_artifacts.get("result")
    if main_result_artifact is None:
        raise ValueError("SR expanded-chart whitening artifact ledger omits result")
    if main_result_artifact["sha256"] != comparisons[-1]["result_sha256"]:
        raise ValueError(
            "SR expanded-chart whitening main result does not match the r30 comparison"
        )

    r30_result = result_payloads["good_expanded_chart_whitened_r30"]
    r30_history = _complete_history(
        r30_result,
        path=_repo_path(comparisons[-1]["result_json"]),
    )
    support_ranks: list[int] = []
    logical_dimensions: list[int] = []
    runtime_dimensions: list[int] = []
    for index, row in enumerate(r30_history, start=1):
        accepted_refit = row.get("accepted_refit")
        if not isinstance(accepted_refit, Mapping):
            raise ValueError(f"SR whitening round {index} lacks accepted-refit telemetry")
        if accepted_refit.get("base_chart_policy") != expected_route["powell_base_chart"]:
            raise ValueError(f"SR whitening round {index} used the wrong base chart")
        if accepted_refit.get("policy") != expected_route["accepted_refit_coordinate_chart"]:
            raise ValueError(f"SR whitening round {index} used the wrong refit chart")
        if accepted_refit.get("chart_fixed_within_powell_invocation") is not True:
            raise ValueError(f"SR whitening round {index} did not fix the Powell chart")
        if accepted_refit.get("chart_recomputed_after_next_admission") is not True:
            raise ValueError(f"SR whitening round {index} did not record chart rebuild")
        support_ranks.append(int(accepted_refit["metric_support_rank"]))
        logical_dimensions.append(int(accepted_refit["logical_parameter_count"]))
        runtime_dimensions.append(int(accepted_refit["base_parameter_count"]))

    source_lock = payload.get("source_lock")
    if not isinstance(source_lock, Mapping):
        raise ValueError("SR expanded-chart whitening validation has no source lock")
    verified_source_lock: dict[str, Any] = {}
    for stem in ("archive", "manifest"):
        linked_path = _require_hash_link(
            source_lock.get(f"{stem}_path"),
            source_lock.get(f"{stem}_sha256"),
            context=f"SR expanded-chart whitening source lock {stem}",
        )
        verified_source_lock[f"{stem}_path"] = _rel(linked_path)
        verified_source_lock[f"{stem}_sha256"] = _sha256(linked_path)
    if source_lock.get("non_swept_settings_diff") != []:
        raise ValueError("SR expanded-chart whitening source lock has unapproved drift")
    verified_source_lock["non_swept_settings_diff"] = []
    verified_source_lock["approved_executable_diff"] = dict(
        source_lock.get("approved_executable_diff") or {}
    )

    final_round_refit = r30_history[-1]["accepted_refit"]
    final_round_query = final_round_refit["accepted_refit_invocation"][
        "metric_query_accounting"
    ]
    r30_adapt = r30_result["adapt_vqe"]
    terminal_full_refit = (r30_adapt.get("final_full_refit") or {}).get(
        "accepted_refit"
    )
    if not isinstance(terminal_full_refit, Mapping):
        raise ValueError("SR expanded-chart whitening terminal refit chart is absent")
    terminal_query = terminal_full_refit["accepted_refit_invocation"][
        "metric_query_accounting"
    ]
    round30_metric_accounting = {
        "logical_dimension": int(final_round_refit["logical_parameter_count"]),
        "expanded_runtime_dimension": int(final_round_refit["base_parameter_count"]),
        "retained_support_rank": int(final_round_refit["metric_support_rank"]),
        "symmetric_metric_element_occurrences": int(
            final_round_query["symmetric_metric_element_occurrences"]
        ),
        "new_unique_metric_elements_charged": int(
            final_round_query["new_unique_metric_elements_charged"]
        ),
        "deduplicated_metric_elements": int(
            final_round_query["deduplicated_or_ledger_disabled_count"]
        ),
    }
    terminal_refit_metric_accounting = {
        "fresh_chart": True,
        "logical_dimension": int(terminal_full_refit["logical_parameter_count"]),
        "expanded_runtime_dimension": int(terminal_full_refit["base_parameter_count"]),
        "retained_support_rank": int(terminal_full_refit["metric_support_rank"]),
        "symmetric_metric_element_occurrences": int(
            terminal_query["symmetric_metric_element_occurrences"]
        ),
        "new_unique_metric_elements_charged": int(
            terminal_query["new_unique_metric_elements_charged"]
        ),
        "deduplicated_metric_elements": int(
            terminal_query["deduplicated_or_ledger_disabled_count"]
        ),
    }

    validation = payload.get("validation")
    result = payload.get("result")
    reference = payload.get("reference")
    if not all(isinstance(item, Mapping) for item in (validation, result, reference)):
        raise ValueError("SR expanded-chart whitening validation summary is incomplete")
    if validation.get("estimator_ledger_complete") is not True:
        raise ValueError("SR expanded-chart whitening estimator ledger is incomplete")
    if validation.get("strict_terminal_replay_passed") is not True:
        raise ValueError("SR expanded-chart whitening strict replay did not pass")
    terminal_actions = result.get("terminal_actions")
    if not isinstance(terminal_actions, Mapping):
        raise ValueError("SR expanded-chart whitening terminal actions are undisclosed")
    if terminal_actions.get("final_full_refit_executed") is not True:
        raise ValueError("SR expanded-chart whitening terminal full refit is undisclosed")
    if int(terminal_actions.get("post_refit_phase1_prune_accepted_count_from_log", -1)) != 1:
        raise ValueError("SR expanded-chart whitening terminal prune disclosure changed")

    return {
        "schema": SR_EXPANDED_CHART_WHITENING_VALIDATION_SCHEMA,
        "status": str(payload["status"]),
        "validation_json": _rel(path),
        "validation_sha256": _sha256(path),
        "regime": "weak-weak",
        "route": dict(route),
        "reference": dict(reference),
        "result": dict(result),
        "validation": dict(validation),
        "source_lock": verified_source_lock,
        "verified_artifacts": verified_artifacts,
        "comparisons": comparisons,
        "support_rank_sequence": support_ranks,
        "logical_dimension_sequence": logical_dimensions,
        "expanded_runtime_dimension_sequence": runtime_dimensions,
        "round30_metric_accounting": round30_metric_accounting,
        "terminal_refit_metric_accounting": terminal_refit_metric_accounting,
        "gram_refresh_contract": {
            "after_each_admission": True,
            "fixed_within_each_powell_invocation": True,
            "classical_factorization_quantum_query_charge": 0,
            "metric_measurements_charged_to": "N_metric",
        },
    }


def _load_sr_expanded_chart_whitening_intermediate_weak_validation(
    path: Path,
    *,
    qiskit_sidecar_path: Path | None = None,
) -> dict[str, Any]:
    """Load one hash-closed completed intermediate--weak SR-SNAKE validation."""

    path = path.resolve()
    payload = _read_json(path)
    if payload.get("schema") != SR_COMPLETED_RUN_VALIDATION_SCHEMA:
        raise ValueError(
            "Unexpected SR intermediate-weak completed-run validation schema: "
            f"{payload.get('schema')!r}"
        )
    if payload.get("regime") != "intermediate-weak":
        raise ValueError(
            "SR intermediate-weak page accepts intermediate-weak evidence only"
        )
    if payload.get("status") != "validated":
        raise ValueError("SR intermediate-weak completed run is not validated")
    if payload.get("blockers") != []:
        raise ValueError("SR intermediate-weak validation contains blockers")

    route = payload.get("route")
    if not isinstance(route, Mapping):
        raise ValueError("SR intermediate-weak validation has no route block")
    expected_route = {
        "family": "singleton_response_snake",
        "profile": "supported_whitened_adaptive_trust_v1",
        "powell_base_chart": "expanded_runtime_projected_logical_v1",
        "accepted_refit_coordinate_chart": "supported_fs_whitened_fixed_v1",
        "accepted_refit_whitening_policy": "supported_metric_whitened_eigh_v1",
        "adaptive_trust_policy": "displacement_calibrated_unbounded_v2",
    }
    for key, expected in expected_route.items():
        if route.get(key) != expected:
            raise ValueError(
                f"SR intermediate-weak route mismatch for {key}: "
                f"expected {expected!r}, found {route.get(key)!r}"
            )
    if route.get("all_history_rows_used_expected_base_chart") is not True:
        raise ValueError("SR intermediate-weak base-chart audit did not pass")
    if route.get("all_history_rows_used_expected_whitening") is not True:
        raise ValueError("SR intermediate-weak whitening audit did not pass")

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("SR intermediate-weak validation has no artifact ledger")
    verified_artifacts: dict[str, dict[str, str]] = {}
    for label, link in artifacts.items():
        if not isinstance(link, Mapping):
            raise ValueError(
                f"SR intermediate-weak artifact {label} is not a path/hash link"
            )
        linked_path = _require_hash_link(
            link.get("path"),
            link.get("sha256"),
            context=f"SR intermediate-weak artifact {label}",
        )
        verified_artifacts[str(label)] = {
            "path": _rel(linked_path),
            "sha256": _sha256(linked_path),
        }
    if "result" not in verified_artifacts:
        raise ValueError("SR intermediate-weak artifact ledger omits result")

    source_lock = payload.get("source_lock")
    if not isinstance(source_lock, Mapping):
        raise ValueError("SR intermediate-weak validation has no source lock")
    verified_source_lock: dict[str, Any] = {
        "runtime_tree": str(source_lock.get("runtime_tree") or ""),
        "verified_file_count": int(source_lock.get("verified_file_count") or 0),
    }
    for stem in ("archive", "manifest"):
        linked_path = _require_hash_link(
            source_lock.get(f"{stem}_path"),
            source_lock.get(f"{stem}_sha256"),
            context=f"SR intermediate-weak source lock {stem}",
        )
        verified_source_lock[f"{stem}_path"] = _rel(linked_path)
        verified_source_lock[f"{stem}_sha256"] = _sha256(linked_path)

    reference = payload.get("reference")
    result = payload.get("result")
    checkpoint = payload.get("checkpoint_replay")
    accounting = payload.get("estimator_accounting")
    if not all(
        isinstance(item, Mapping)
        for item in (reference, result, checkpoint, accounting)
    ):
        raise ValueError("SR intermediate-weak validation summary is incomplete")
    n_ph_work = int(reference["n_ph_work"])
    n_ph_ref = int(reference["n_ph_ref"])
    if n_ph_work != n_ph_ref:
        raise ValueError("SR intermediate-weak validation is not same-cutoff")
    reference_energy = float(reference["same_cutoff_energy"])
    if not math.isclose(
        reference_energy,
        float(result["same_cutoff_reference_energy"]),
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError("SR intermediate-weak same-cutoff reference mismatch")

    result_path = _repo_path(verified_artifacts["result"]["path"])
    result_payload = _read_json(result_path)
    adapt = result_payload.get("adapt_vqe")
    if not isinstance(adapt, Mapping):
        raise ValueError("SR intermediate-weak result has no adapt_vqe block")
    history = _complete_history(result_payload, path=result_path)
    expected_rounds = int(route["outer_round_horizon"])
    if expected_rounds != 30 or len(history) != expected_rounds:
        raise ValueError("SR intermediate-weak completed run is not the r30 horizon")
    if int(checkpoint["active_checkpoint_count"]) != expected_rounds:
        raise ValueError("SR intermediate-weak checkpoint count differs from r30")

    checkpoint_rows = checkpoint.get("rows")
    if not isinstance(checkpoint_rows, Sequence) or isinstance(
        checkpoint_rows, (str, bytes)
    ):
        raise ValueError("SR intermediate-weak checkpoint ledger is absent")
    if len(checkpoint_rows) != expected_rounds:
        raise ValueError("SR intermediate-weak checkpoint ledger is incomplete")
    leakage_tolerance = float(checkpoint["leakage_tolerance"])
    maximum_sector_leakage = float(
        checkpoint["maximum_fixed_sector_illegal_probability"]
    )
    maximum_padding_leakage = float(
        checkpoint["maximum_binary_padding_illegal_probability"]
    )
    if max(maximum_sector_leakage, maximum_padding_leakage) > leakage_tolerance:
        raise ValueError("SR intermediate-weak leakage exceeds tolerance")
    if not math.isclose(
        float(checkpoint["terminal_state_fidelity_to_serialized_final_state"]),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("SR intermediate-weak terminal replay fidelity failed")

    support_ranks: list[int] = []
    logical_dimensions: list[int] = []
    runtime_dimensions: list[int] = []
    trajectory_points = [
        {
            "k": 0,
            "error": _positive(float(history[0]["energy_before_opt"]) - reference_energy),
        }
    ]
    depth_sequence: list[int] = []
    for index, row in enumerate(history, start=1):
        accepted_refit = row.get("accepted_refit")
        if not isinstance(accepted_refit, Mapping):
            raise ValueError(
                f"SR intermediate-weak round {index} lacks accepted-refit telemetry"
            )
        if accepted_refit.get("base_chart_policy") != expected_route["powell_base_chart"]:
            raise ValueError(f"SR intermediate-weak round {index} used the wrong chart")
        if (
            accepted_refit.get("policy")
            != expected_route["accepted_refit_coordinate_chart"]
        ):
            raise ValueError(
                f"SR intermediate-weak round {index} used the wrong refit policy"
            )
        if accepted_refit.get("supported_metric_whitening_policy") not in (
            None,
            expected_route["accepted_refit_whitening_policy"],
        ):
            raise ValueError(
                f"SR intermediate-weak round {index} used the wrong whitening policy"
            )
        if accepted_refit.get("chart_fixed_within_powell_invocation") is not True:
            raise ValueError(
                f"SR intermediate-weak round {index} did not fix the Powell chart"
            )
        if accepted_refit.get("chart_recomputed_after_next_admission") is not True:
            raise ValueError(
                f"SR intermediate-weak round {index} did not record chart rebuild"
            )
        support_ranks.append(int(accepted_refit["metric_support_rank"]))
        logical_dimensions.append(int(accepted_refit["logical_parameter_count"]))
        runtime_dimensions.append(int(accepted_refit["base_parameter_count"]))
        depth_sequence.append(int(row["depth_cumulative"]))
        trajectory_points.append(
            {
                "k": index,
                "error": _positive(float(row["energy_after_opt"]) - reference_energy),
            }
        )

    preterminal_energy = float(result["pre_terminal_checkpoint_replayed_energy"])
    preterminal_error = float(
        result["pre_terminal_checkpoint_replayed_absolute_error"]
    )
    if not math.isclose(
        preterminal_energy,
        float(history[-1]["energy_after_opt"]),
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ) or not math.isclose(
        preterminal_error,
        float(trajectory_points[-1]["error"]),
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError("SR intermediate-weak preterminal checkpoint mismatch")
    finalized_energy = float(result["displayed_energy"])
    finalized_error = float(result["displayed_absolute_error"])
    if not math.isclose(
        finalized_energy,
        float(adapt["energy"]),
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ) or not math.isclose(
        finalized_error,
        _positive(finalized_energy - reference_energy),
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError("SR intermediate-weak finalized result mismatch")
    if not math.isclose(
        float(result["replayed_energy"]),
        finalized_energy,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ) or float(result["displayed_replayed_energy_discrepancy"]) > 1.0e-15:
        raise ValueError("SR intermediate-weak independent replay mismatch")

    terminal_refit = (adapt.get("final_full_refit") or {}).get("accepted_refit")
    if not isinstance(terminal_refit, Mapping):
        raise ValueError("SR intermediate-weak terminal accepted-refit chart is absent")
    if terminal_refit.get("base_chart_policy") != expected_route["powell_base_chart"]:
        raise ValueError("SR intermediate-weak terminal refit used the wrong chart")
    if terminal_refit.get("policy") != expected_route["accepted_refit_coordinate_chart"]:
        raise ValueError("SR intermediate-weak terminal refit used the wrong whitening")
    terminal_prune = adapt.get("prune_summary")
    post_prune_refit = adapt.get("post_prune_refit")
    if not isinstance(terminal_prune, Mapping) or not isinstance(
        post_prune_refit, Mapping
    ):
        raise ValueError("SR intermediate-weak terminal prune disclosure is absent")
    if int(terminal_prune.get("candidate_count", -1)) != 1:
        raise ValueError("SR intermediate-weak terminal prune nomination count changed")
    if int(terminal_prune.get("accepted_count", -1)) != 0:
        raise ValueError("SR intermediate-weak terminal prune accepted a deletion")
    if post_prune_refit.get("executed") is not False:
        raise ValueError("SR intermediate-weak unexpected post-prune refit")

    if accounting.get("complete") is not True:
        raise ValueError("SR intermediate-weak estimator accounting is incomplete")
    winning = accounting.get("winning_lineage")
    all_branch = accounting.get("all_branch_search_work")
    if not isinstance(winning, Mapping) or not isinstance(all_branch, Mapping):
        raise ValueError("SR intermediate-weak S_alg blocks are absent")
    for label, block in (("winning", winning), ("all branch", all_branch)):
        reconstructed = sum(
            int(block[name]) for name in ("N_H_outer", "N_H_refit", "N_grad", "N_metric")
        )
        if reconstructed != int(block["S_alg"]):
            raise ValueError(f"SR intermediate-weak {label} S_alg does not close")
    discarded = int(accounting["discarded_branch_unique_work"])
    if int(all_branch["S_alg"]) - int(winning["S_alg"]) != discarded:
        raise ValueError("SR intermediate-weak discarded-branch work does not close")

    qiskit: dict[str, Any] | None = None
    if qiskit_sidecar_path is not None:
        from pipelines.exact_bench.generic_static_metric_enrichment import (
            _sha256_json_without_snake_sidecars,
        )

        qiskit_sidecar_path = qiskit_sidecar_path.resolve()
        if not qiskit_sidecar_path.is_file():
            raise FileNotFoundError(qiskit_sidecar_path)
        sidecar = _read_json(qiskit_sidecar_path)
        if sidecar.get("schema") != "paper_i_selected_prefix_qiskit_cost_sidecar_v1":
            raise ValueError("SR intermediate-weak Qiskit sidecar schema mismatch")
        if sidecar.get("compile_convention") != "table_i_basis_gate_transpile_v1":
            raise ValueError("SR intermediate-weak Qiskit compile convention mismatch")
        if sidecar.get("compiled_resource_qiskit_validated") is not True:
            raise ValueError("SR intermediate-weak Qiskit sidecar is not validated")
        if sidecar.get("compiled_circuit_stats_status") != "ok":
            raise ValueError("SR intermediate-weak Qiskit compilation did not close")
        source_result_path = _repo_path(sidecar.get("source_result_path"))
        if source_result_path.resolve() != result_path.resolve():
            raise ValueError("SR intermediate-weak Qiskit source-result path mismatch")
        if (
            sidecar.get("source_result_hash_convention")
            != "canonical_json_without_snake_sidecars_v1"
        ):
            raise ValueError("SR intermediate-weak Qiskit source hash convention mismatch")
        canonical_result_hash = _sha256_json_without_snake_sidecars(result_path)
        if canonical_result_hash != str(sidecar.get("source_result_sha256") or ""):
            raise ValueError("SR intermediate-weak Qiskit source-result hash mismatch")
        if int(sidecar["history_position"]) != expected_rounds:
            raise ValueError("SR intermediate-weak Qiskit prefix is not r30")
        if int(sidecar["logical_operator_count"]) != int(
            result["finalized_active_depth"]
        ):
            raise ValueError("SR intermediate-weak Qiskit logical depth mismatch")
        if int(sidecar["runtime_rotation_count"]) != int(runtime_dimensions[-1]):
            raise ValueError("SR intermediate-weak Qiskit runtime rotation mismatch")
        if not math.isclose(
            float(sidecar["energy_after_opt_at_prefix"]),
            finalized_energy,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ValueError("SR intermediate-weak Qiskit endpoint energy mismatch")
        qiskit = {
            "path": _rel(qiskit_sidecar_path),
            "sha256": _sha256(qiskit_sidecar_path),
            "schema": str(sidecar["schema"]),
            "compile_convention": str(sidecar["compile_convention"]),
            "N2q": int(sidecar["compiled_count_2q_total"]),
            "D2q": int(sidecar["compiled_depth_2q_total"]),
            "Dc": int(sidecar["compiled_depth_total"]),
            "logical_operator_count": int(sidecar["logical_operator_count"]),
            "runtime_rotation_count": int(sidecar["runtime_rotation_count"]),
            "source_result_path": _rel(source_result_path),
            "source_result_sha256": str(sidecar["source_result_sha256"]),
            "source_result_hash_convention": str(
                sidecar["source_result_hash_convention"]
            ),
            "primary_error_at_prefix_ignored": True,
            "accuracy_source": "validation_json_locked_same_cutoff_reference",
        }

    return {
        "schema": SR_COMPLETED_RUN_VALIDATION_SCHEMA,
        "status": "validated",
        "regime": "intermediate-weak",
        "validation_json": _rel(path),
        "validation_sha256": _sha256(path),
        "route": dict(route),
        "reference": dict(reference),
        "result": dict(result),
        "checkpoint_replay": dict(checkpoint),
        "estimator_accounting": dict(accounting),
        "source_lock": verified_source_lock,
        "verified_artifacts": verified_artifacts,
        "trajectory_points": trajectory_points,
        "depth_sequence": depth_sequence,
        "support_rank_sequence": support_ranks,
        "logical_dimension_sequence": logical_dimensions,
        "expanded_runtime_dimension_sequence": runtime_dimensions,
        "terminal_full_refit_nfev": int((adapt.get("final_full_refit") or {})["nfev"]),
        "terminal_actions": {
            "final_full_refit_nfev": int(
                (adapt.get("final_full_refit") or {})["nfev"]
            ),
            "terminal_prune_candidate_count": int(
                terminal_prune["candidate_count"]
            ),
            "terminal_prune_accepted_count": int(terminal_prune["accepted_count"]),
            "post_prune_refit_executed": bool(post_prune_refit["executed"]),
        },
        "qiskit": qiskit,
    }


def _normalize_fm_live_regime(raw: Any) -> str:
    regime = str(raw).strip().lower().replace("_", "-")
    if regime == "strong-weak-u8":
        regime = "strong-weak"
    elif regime == "strong-strong-u8":
        regime = "strong-strong"
    if regime not in REGIME_ORDER:
        raise ValueError(f"Unknown FM live-snapshot regime: {raw!r}")
    return regime


def _fm_live_policy_role(policy: str) -> str:
    roles = {
        "qbroyd_on": "fm_qbroyd_default",
        "qbroyd_off": "fm_qbroyd_off",
    }
    try:
        return roles[policy]
    except KeyError as exc:
        raise ValueError(f"Unknown FM live-snapshot policy: {policy!r}") from exc


def _fm_progress_log_curve(path: Path, *, role: str) -> Curve:
    """Read a compact JSON failure ledger when no full current.json survived."""

    payload = _read_json(path)
    raw_rows = next(
        (
            payload.get(key)
            for key in ("progress", "history", "events", "rows")
            if isinstance(payload.get(key), Sequence)
            and not isinstance(payload.get(key), (str, bytes))
        ),
        None,
    )
    if raw_rows is None:
        raise ValueError(f"Failure progress log has no JSON progress rows: {path}")
    points: list[tuple[int, float]] = []
    for index, raw_row in enumerate(raw_rows, start=1):
        if not isinstance(raw_row, Mapping):
            continue
        error = next(
            (
                raw_row.get(key)
                for key in ("delta_abs_current", "abs_delta_e", "error")
                if raw_row.get(key) is not None
            ),
            None,
        )
        if error is None:
            continue
        k_value = next(
            (
                raw_row.get(key)
                for key in ("controller_round", "round", "iteration", "k")
                if raw_row.get(key) is not None
            ),
            index,
        )
        points.append((int(k_value), _positive(error)))
    if not points:
        raise ValueError(f"Failure progress log has no usable error points: {path}")
    points.sort(key=lambda item: item[0])
    return Curve(
        role=role,
        points=tuple(points),
        marker_k=points[-1][0],
        marker_error=points[-1][1],
        source_json=_rel(path),
        source_sha256=_sha256(path),
    )


def _fm_live_payload_route(payload: Mapping[str, Any], *, path: Path) -> str:
    settings = payload.get("settings")
    adapt = payload.get("adapt_vqe")
    candidates = []
    if isinstance(settings, Mapping):
        candidates.append(settings.get("adapt_reoptimization_route"))
    if isinstance(adapt, Mapping):
        candidates.append(adapt.get("adapt_reoptimization_route"))
        warm = adapt.get("formal_manifold_warm_start")
        if isinstance(warm, Mapping):
            candidates.append(warm.get("route"))
    candidates.append(payload.get("route_id"))
    routes = {str(value) for value in candidates if value not in (None, "")}
    if routes != {FM_LIVE_ROUTE_ID}:
        raise ValueError(f"FM live snapshot route mismatch at {path}: {sorted(routes)}")
    return FM_LIVE_ROUTE_ID


def _fm_live_payload_qbroyd_epsilon(
    payload: Mapping[str, Any], *, path: Path
) -> float:
    settings = payload.get("settings")
    value: Any = None
    if isinstance(settings, Mapping):
        config = settings.get("adapt_formal_manifold_config")
        if isinstance(config, Mapping):
            value = config.get("qbroyd_epsilon0")
    if value is None:
        value = payload.get("qbroyd_epsilon0")
    if value is None:
        raise ValueError(f"FM live snapshot lacks qB policy telemetry: {path}")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"Invalid FM live qB epsilon at {path}: {value!r}")
    return result


def _validate_fm_live_no_structural_rollback(
    payload: Mapping[str, Any], *, path: Path
) -> None:
    settings = payload.get("settings")
    rollback_mode = (
        settings.get("adapt_rollback_mode")
        if isinstance(settings, Mapping)
        else None
    )
    if str(rollback_mode or "").strip().lower() == "structural":
        raise ValueError(f"FM live snapshot enables structural rollback: {path}")
    flags = [payload.get("outer_structural_rollback_active")]
    if isinstance(settings, Mapping):
        flags.append(settings.get("outer_structural_rollback_active"))
    if any(value is True for value in flags):
        raise ValueError(f"FM live snapshot reports active structural rollback: {path}")


def _load_fm_live_snapshot_manifest(
    manifest_path: Path,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    """Validate and load a nonterminal FM snapshot bundle for report-only use."""

    manifest_path = manifest_path.resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != FM_LIVE_SNAPSHOT_SCHEMA:
        raise ValueError(f"Unexpected FM live-snapshot schema: {manifest.get('schema')!r}")
    batch_id = str(manifest.get("batch_id") or "").strip()
    cluster_id = str(manifest.get("cluster_id") or "").strip()
    captured_at = str(manifest.get("captured_at") or "").strip()
    bundle_status = str(manifest.get("status") or "").strip()
    if not all((batch_id, cluster_id, captured_at, bundle_status)):
        raise ValueError("FM live-snapshot manifest lacks batch/cluster/time/status")
    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes)):
        raise ValueError("FM live-snapshot manifest entries must be a JSON array")

    rows: dict[str, dict[str, dict[str, Any]]] = {
        regime: {} for regime in REGIME_ORDER
    }
    seen_procs: set[int] = set()
    seen_rows: set[str] = set()
    provenance_entries: list[dict[str, Any]] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            raise TypeError("FM live-snapshot entry is not an object")
        proc_raw = raw_entry.get("proc_id")
        if isinstance(proc_raw, bool):
            raise ValueError(f"Invalid FM live proc id: {proc_raw!r}")
        proc_id = int(proc_raw)
        if proc_id < 0 or proc_id in seen_procs:
            raise ValueError(f"Duplicate or invalid FM live proc id: {proc_id}")
        seen_procs.add(proc_id)
        row_id = str(raw_entry.get("row_id") or "").strip()
        if not row_id or row_id in seen_rows:
            raise ValueError(f"Missing or duplicate FM live row id: {row_id!r}")
        seen_rows.add(row_id)
        if not row_id.startswith(batch_id + "__"):
            raise ValueError(f"FM live row does not belong to batch {batch_id}: {row_id}")

        regime = _normalize_fm_live_regime(raw_entry.get("regime"))
        policy = str(raw_entry.get("policy") or "").strip()
        role = _fm_live_policy_role(policy)
        expected_proc = 2 * REGIME_ORDER.index(regime) + (0 if policy == "qbroyd_on" else 1)
        if proc_id != expected_proc:
            raise ValueError(
                f"FM live proc/policy mapping mismatch for {row_id}: "
                f"proc={proc_id}, expected={expected_proc}"
            )
        regime_token = CAMPAIGN_DIR[regime].replace("-", "_")
        if f"__{regime_token}__" not in row_id or f"__{policy}__" not in row_id:
            raise ValueError(f"FM live row identity mismatch: {row_id}")
        if role in rows[regime]:
            raise ValueError(f"Duplicate FM live policy for {regime}/{policy}")

        scheduler_state = str(raw_entry.get("scheduler_state") or "").strip()
        source_kind = str(raw_entry.get("source_kind") or "").strip()
        expected_source_kind = {
            "running_snapshot": "live_current_json",
            "failed_partial": "failure_progress_log",
        }.get(scheduler_state)
        if expected_source_kind is None or source_kind != expected_source_kind:
            raise ValueError(
                f"FM live status/source mismatch for {row_id}: "
                f"{scheduler_state}/{source_kind}"
            )
        if str(raw_entry.get("route_id") or "") != FM_LIVE_ROUTE_ID:
            raise ValueError(f"FM live route id mismatch for {row_id}")
        if raw_entry.get("structural_rollback_enabled") is not False:
            raise ValueError(f"FM live entry does not prove structural rollback disabled: {row_id}")

        raw_snapshot_path = Path(str(raw_entry.get("snapshot_json") or ""))
        snapshot_path = (
            raw_snapshot_path
            if raw_snapshot_path.is_absolute()
            else manifest_path.parent / raw_snapshot_path
        ).resolve()
        if not snapshot_path.is_file():
            raise FileNotFoundError(snapshot_path)
        actual_hash = _sha256(snapshot_path)
        expected_hash = str(raw_entry.get("source_sha256") or "").lower()
        if actual_hash != expected_hash:
            raise ValueError(f"FM live snapshot hash mismatch: {snapshot_path}")
        payload = _read_json(snapshot_path)
        _fm_live_payload_route(payload, path=snapshot_path)
        _validate_fm_live_no_structural_rollback(payload, path=snapshot_path)
        epsilon = _fm_live_payload_qbroyd_epsilon(payload, path=snapshot_path)
        if policy == "qbroyd_on" and not epsilon > 0.0:
            raise ValueError(f"FM qB-on snapshot has disabled qB update: {snapshot_path}")
        if policy == "qbroyd_off" and not math.isclose(
            epsilon, 0.0, rel_tol=0.0, abs_tol=1.0e-15
        ):
            raise ValueError(f"FM qB-off snapshot has nonzero qB update: {snapshot_path}")
        for payload_batch in (
            payload.get("batch_id"),
            (payload.get("settings") or {}).get("batch_id")
            if isinstance(payload.get("settings"), Mapping)
            else None,
        ):
            if payload_batch not in (None, "") and str(payload_batch) != batch_id:
                raise ValueError(f"FM live payload batch mismatch: {snapshot_path}")

        if source_kind == "live_current_json":
            curve = _history_curve(snapshot_path, role=role)
        elif isinstance(payload.get("adapt_vqe"), Mapping):
            try:
                curve = _history_curve(snapshot_path, role=role)
            except (KeyError, TypeError, ValueError):
                curve = _fm_progress_log_curve(snapshot_path, role=role)
        else:
            curve = _fm_progress_log_curve(snapshot_path, role=role)
        adapt = payload.get("adapt_vqe")
        ansatz_depth = (
            int(adapt["ansatz_depth"])
            if isinstance(adapt, Mapping) and adapt.get("ansatz_depth") is not None
            else None
        )
        if ansatz_depth is None and source_kind == "failure_progress_log":
            raw_progress = next(
                (
                    payload.get(key)
                    for key in ("progress", "history", "events", "rows")
                    if isinstance(payload.get(key), Sequence)
                    and not isinstance(payload.get(key), (str, bytes))
                ),
                (),
            )
            ansatz_depth = next(
                (
                    int(row["ansatz_depth"])
                    for row in reversed(raw_progress)
                    if isinstance(row, Mapping) and row.get("ansatz_depth") is not None
                ),
                None,
            )
        resource = {
            "regime": regime,
            "method": role,
            "method_display": RESOURCE_METHOD_DISPLAY[role],
            "role": "formal_manifold_live_diagnostic",
            "status": scheduler_state,
            "k_pl": int(curve.marker_k),
            "ansatz_depth": ansatz_depth,
            "abs_delta_e": float(curve.marker_error),
            "N2q": None,
            "D2q": None,
            "Dc": None,
            "S": None,
            "resource_status": "pending_nonterminal_snapshot",
            "prefix_semantics": "nonterminal_current_prefix",
            "source_json": _rel(snapshot_path),
            "source_sha256": actual_hash,
            "source_kind": source_kind,
            "scheduler_state": scheduler_state,
            "terminal": False,
        }
        entry_provenance = {
            "proc_id": proc_id,
            "row_id": row_id,
            "regime": regime,
            "policy": policy,
            "scheduler_state": scheduler_state,
            "source_kind": source_kind,
            "snapshot_json": _rel(snapshot_path),
            "source_sha256": actual_hash,
            "route_id": FM_LIVE_ROUTE_ID,
            "qbroyd_epsilon0": epsilon,
            "structural_rollback_enabled": False,
            "controller_round": int(curve.marker_k),
            "ansatz_depth": ansatz_depth,
            "abs_delta_e": float(curve.marker_error),
        }
        rows[regime][role] = {
            "curve": curve,
            "resource": resource,
            "provenance": entry_provenance,
        }
        provenance_entries.append(entry_provenance)

    return rows, {
        "schema": FM_LIVE_SNAPSHOT_SCHEMA,
        "manifest_json": _rel(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "batch_id": batch_id,
        "cluster_id": cluster_id,
        "captured_at": captured_at,
        "status": bundle_status,
        "entry_count": len(provenance_entries),
        "entries": provenance_entries,
        "evidence_class": "matched_within_batch_diagnostic",
        "source_value_anchor": "absent_not_claimed",
        "source_locked_sensitivity": False,
        "terminal_evidence": False,
        "resource_fields": "pending_until_terminal_sidecars",
    }


def _load_fm_live_status_snapshot(
    status_path: Path,
    *,
    live_rows: dict[str, dict[str, dict[str, Any]]],
    live_campaign: Mapping[str, Any],
) -> dict[str, Any]:
    """Overlay hash-locked status endpoints without fabricating trajectories."""

    status_path = status_path.resolve()
    payload = _read_json(status_path)
    if payload.get("schema") != FM_LIVE_STATUS_SCHEMA:
        raise ValueError(f"Unexpected FM live-status schema: {payload.get('schema')!r}")
    captured_at = str(payload.get("captured_at") or "").strip()
    captured_at_local = str(payload.get("captured_at_local") or "").strip()
    if not captured_at or not captured_at_local:
        raise ValueError("FM live-status snapshot lacks capture time")
    captured_clock_label = " ".join(captured_at_local.split()[-2:])
    if str(payload.get("route_id") or "") != FM_LIVE_ROUTE_ID:
        raise ValueError("FM live-status route mismatch")
    if payload.get("structural_rollback_enabled") is not False:
        raise ValueError("FM live-status snapshot does not prove rollback disabled")

    prior = payload.get("prior_live_snapshot")
    if not isinstance(prior, Mapping):
        raise ValueError("FM live-status snapshot lacks prior-live provenance")
    if str(prior.get("manifest_sha256") or "").lower() != str(
        live_campaign.get("manifest_sha256") or ""
    ).lower():
        raise ValueError("FM live-status prior-manifest hash mismatch")

    repair = payload.get("replacement_repair")
    if not isinstance(repair, Mapping):
        raise ValueError("FM live-status snapshot lacks replacement provenance")
    repair_path = _repo_path(str(repair.get("manifest") or "")).resolve()
    if not repair_path.is_file():
        raise FileNotFoundError(repair_path)
    repair_hash = _sha256(repair_path)
    if repair_hash != str(repair.get("manifest_sha256") or "").lower():
        raise ValueError("FM live-status repair-manifest hash mismatch")

    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes)):
        raise ValueError("FM live-status entries must be a JSON array")
    if len(raw_entries) != 12:
        raise ValueError("FM live-status snapshot must cover the full 2x6 matrix")

    seen: set[tuple[str, str]] = set()
    normalized_entries: list[dict[str, Any]] = []
    for raw in raw_entries:
        if not isinstance(raw, Mapping):
            raise TypeError("FM live-status entry is not an object")
        regime = _normalize_fm_live_regime(raw.get("regime"))
        policy = str(raw.get("policy") or "").strip()
        role = _fm_live_policy_role(policy)
        key = (regime, policy)
        if key in seen:
            raise ValueError(f"Duplicate FM live-status entry: {regime}/{policy}")
        seen.add(key)
        if role not in live_rows[regime]:
            raise ValueError(f"FM live-status has no prior trajectory: {regime}/{policy}")
        state = str(raw.get("state") or "").strip()
        if state not in {"running_status_endpoint", "science_complete_packaging_failed"}:
            raise ValueError(f"Unsupported FM live-status state: {state}")
        round_index = int(raw["controller_round"])
        ansatz_depth = int(raw["ansatz_depth"])
        error = _positive(raw["abs_delta_e"])
        if round_index < 0 or ansatz_depth < 0:
            raise ValueError("FM live-status rounds/depths must be nonnegative")
        relation = str(raw.get("trajectory_relation") or "").strip()
        cluster_id = str(raw.get("cluster_id") or "").strip()
        if regime == "intermediate-weak":
            if relation != "replacement_restart_after_parent_failure":
                raise ValueError("IW replacement endpoint lacks restart boundary")
            if cluster_id != str(repair.get("cluster_id") or ""):
                raise ValueError("IW replacement endpoint cluster mismatch")
        elif relation != "same_row_later_observation":
            raise ValueError(f"Unexpected trajectory relation for {regime}/{policy}")
        if state == "science_complete_packaging_failed":
            if raw.get("terminal_metric_validated") is not False:
                raise ValueError("Packaging-failed row cannot claim a terminal metric")
            if str(raw.get("metric_source") or "") != "last_verified_prior_checkpoint":
                raise ValueError("Packaging-failed row must use the prior checkpoint")
        elif str(raw.get("metric_source") or "") != "live_checkpoint_status_observation":
            raise ValueError("Running endpoint has an invalid metric source")

        entry = live_rows[regime][role]
        prior_curve: Curve = entry["curve"]
        endpoint = {
            "k": round_index,
            "error": error,
            "ansatz_depth": ansatz_depth,
            "state": state,
            "cluster_id": cluster_id,
            "trajectory_relation": relation,
            "marker_only": True,
            "terminal_metric_validated": False,
            "source_status_json": _rel(status_path),
            "source_status_sha256": _sha256(status_path),
        }
        entry["curve"] = Curve(
            role=prior_curve.role,
            points=prior_curve.points,
            marker_k=round_index,
            marker_error=error,
            source_json=prior_curve.source_json,
            source_sha256=prior_curve.source_sha256,
            source_segments=(*prior_curve.source_segments, endpoint),
        )
        resource = entry["resource"]
        resource.update(
            status=state,
            k_pl=round_index,
            ansatz_depth=ansatz_depth,
            abs_delta_e=error,
            N2q=None,
            D2q=None,
            Dc=None,
            S=None,
            resource_status="pending_terminal_packaging_and_sidecars",
            terminal=False,
            terminal_metric_validated=False,
            status_endpoint_only=True,
            checkpoint_asterisk=(state == "science_complete_packaging_failed"),
            trajectory_relation=relation,
            status_cluster_id=cluster_id,
            source_status_json=_rel(status_path),
            source_status_sha256=_sha256(status_path),
            status_capture_label=captured_clock_label,
        )
        entry["provenance"]["status_endpoint"] = endpoint
        normalized_entries.append(
            {
                "regime": regime,
                "policy": policy,
                "state": state,
                "cluster_id": cluster_id,
                "controller_round": round_index,
                "ansatz_depth": ansatz_depth,
                "abs_delta_e": error,
                "trajectory_relation": relation,
                "terminal_metric_validated": False,
            }
        )

    return {
        "schema": FM_LIVE_STATUS_SCHEMA,
        "status_json": _rel(status_path),
        "status_sha256": _sha256(status_path),
        "captured_at": captured_at,
        "captured_at_local": captured_at_local,
        "entry_count": len(normalized_entries),
        "entries": normalized_entries,
        "prior_live_manifest_sha256": str(prior["manifest_sha256"]),
        "replacement_repair_manifest": _rel(repair_path),
        "replacement_repair_manifest_sha256": repair_hash,
        "endpoint_semantics": "marker_only_no_trajectory_interpolation",
        "intermediate_weak_semantics": "replacement_restart_not_continuous_parent_prefix",
        "terminal_resources": "pending_for_running_and_packaging_failed_rows",
    }


def _fm_stopped_policy(policy_id: str) -> tuple[str, str]:
    policies = {
        "inverse_rbfgs_qbroyd_on_v1": ("qbroyd_on", "fm_qbroyd_default"),
        "inverse_rbfgs_qbroyd_off_v1": ("qbroyd_off", "fm_qbroyd_off"),
    }
    try:
        return policies[policy_id]
    except KeyError as exc:
        raise ValueError(f"Unknown FM stopped-snapshot policy: {policy_id!r}") from exc


def _load_fm_stopped_snapshot_manifest(
    manifest_path: Path,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    """Load user-stopped FM checkpoints without treating them as terminal rows."""

    manifest_path = manifest_path.resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != FM_STOPPED_SNAPSHOT_SCHEMA:
        raise ValueError(
            f"Unexpected FM stopped-snapshot schema: {manifest.get('schema')!r}"
        )
    batch_id = str(manifest.get("batch_id") or "").strip()
    cluster_id = str(manifest.get("cluster_id") or "").strip()
    captured_at = str(manifest.get("snapshot_stage_created_utc") or "").strip()
    stopped_at = str(manifest.get("stop_requested_utc") or "").strip()
    if not all((batch_id, cluster_id, captured_at, stopped_at)):
        raise ValueError("FM stopped-snapshot manifest lacks batch/cluster/time fields")

    expected_procs = set(range(6, 12))
    expected_stop_scope = {f"{cluster_id}.{proc_id}" for proc_id in expected_procs}
    raw_stop_scope = manifest.get("stop_scope")
    if not isinstance(raw_stop_scope, Sequence) or isinstance(
        raw_stop_scope, (str, bytes)
    ):
        raise ValueError("FM stopped-snapshot stop scope must be a JSON array")
    if {str(value) for value in raw_stop_scope} != expected_stop_scope:
        raise ValueError("FM stopped-snapshot stop scope is not proc6-11")

    scheduler = manifest.get("scheduler_status_after_stop")
    if not isinstance(scheduler, Mapping):
        raise ValueError("FM stopped-snapshot scheduler status is missing")
    if (
        int(scheduler.get("job_status", -1)) != 3
        or str(scheduler.get("meaning") or "").lower() != "removed"
        or scheduler.get("unrelated_jobs_touched") is not False
    ):
        raise ValueError("FM stopped-snapshot scheduler removal evidence is invalid")

    validation = manifest.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError("FM stopped-snapshot validation block is missing")
    required_true = (
        "archive_sha256_matches_access_point",
        "gzip_streams_valid",
        "all_json_parse",
        "runner_expected_settings_hash_matches_plan",
        "all_current_checkpoints_are_completed_beam_rounds",
        "all_formal_warm_state_checkpoints_present",
        "all_query_closure_checkpoints_present",
        "all_route_trust_states_present",
    )
    failed_validation = [
        key for key in required_true if validation.get(key) is not True
    ]
    if validation.get("credentials_serialized") is not False:
        failed_validation.append("credentials_serialized")
    if failed_validation:
        raise ValueError(
            "FM stopped-snapshot validation failed: "
            + ", ".join(failed_validation)
        )

    raw_rows = manifest.get("rows")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise ValueError("FM stopped-snapshot rows must be a JSON array")
    rows: dict[str, dict[str, dict[str, Any]]] = {
        regime: {} for regime in REGIME_ORDER
    }
    seen_procs: set[int] = set()
    provenance_entries: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, Mapping):
            raise TypeError("FM stopped-snapshot row is not an object")
        proc_raw = raw_row.get("proc_id")
        if isinstance(proc_raw, bool):
            raise ValueError(f"Invalid FM stopped proc id: {proc_raw!r}")
        proc_id = int(proc_raw)
        if proc_id in seen_procs:
            raise ValueError(f"Duplicate FM stopped proc id: {proc_id}")
        seen_procs.add(proc_id)
        regime = _normalize_fm_live_regime(raw_row.get("regime"))
        policy_id = str(raw_row.get("policy_id") or "").strip()
        policy, role = _fm_stopped_policy(policy_id)
        expected_proc = 2 * REGIME_ORDER.index(regime) + (
            0 if policy == "qbroyd_on" else 1
        )
        if proc_id != expected_proc or proc_id not in expected_procs:
            raise ValueError(
                f"FM stopped proc/policy mapping mismatch: proc={proc_id}, "
                f"expected={expected_proc}"
            )

        raw_current_path = Path(str(raw_row.get("current_json") or ""))
        current_path = (
            raw_current_path
            if raw_current_path.is_absolute()
            else manifest_path.parent / raw_current_path
        ).resolve()
        if not current_path.is_file():
            raise FileNotFoundError(current_path)
        current_hash = _sha256(current_path)
        if current_hash != str(raw_row.get("current_sha256") or "").lower():
            raise ValueError(f"FM stopped current checkpoint hash mismatch: {current_path}")

        plan_path = current_path.with_name("plan.json")
        runner_path = current_path.with_name("runner_manifest.json")
        for path in (plan_path, runner_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        plan = _read_json(plan_path)
        runner = _read_json(runner_path)
        if str(runner.get("remote_dry_run_plan_sha256") or "").lower() != _sha256(
            plan_path
        ):
            raise ValueError(f"FM stopped runner/plan hash mismatch: {runner_path}")
        if str(runner.get("expected_scientific_settings_hash") or "") != str(
            plan.get("scientific_settings_hash") or ""
        ):
            raise ValueError(f"FM stopped scientific-settings hash mismatch: {runner_path}")
        runner_regime = _normalize_fm_live_regime(runner.get("regime"))
        row_id = str(runner.get("row_id") or "").strip()
        regime_token = CAMPAIGN_DIR[regime].replace("-", "_")
        if (
            str(runner.get("batch_id") or "") != batch_id
            or runner_regime != regime
            or str(runner.get("policy_id") or "") != policy_id
            or not row_id.startswith(batch_id + "__")
            or f"__{regime_token}__" not in row_id
            or f"__{policy}__" not in row_id
        ):
            raise ValueError(f"FM stopped runner identity mismatch: {runner_path}")

        payload = _read_json(current_path)
        _fm_live_payload_route(payload, path=current_path)
        _validate_fm_live_no_structural_rollback(payload, path=current_path)
        epsilon = _fm_live_payload_qbroyd_epsilon(payload, path=current_path)
        if policy == "qbroyd_on" and not epsilon > 0.0:
            raise ValueError(f"FM stopped qB-on checkpoint has disabled qB: {current_path}")
        if policy == "qbroyd_off" and not math.isclose(
            epsilon, 0.0, rel_tol=0.0, abs_tol=1.0e-15
        ):
            raise ValueError(f"FM stopped qB-off checkpoint has nonzero qB: {current_path}")
        adapt = payload.get("adapt_vqe")
        checkpoint = payload.get("checkpoint")
        if not isinstance(adapt, Mapping) or not isinstance(checkpoint, Mapping):
            raise ValueError(f"FM stopped checkpoint payload is incomplete: {current_path}")
        warm_checkpoint = adapt.get("formal_manifold_warm_state_checkpoint")
        query_checkpoint = adapt.get("formal_manifold_query_closure_checkpoint")
        query_closure = adapt.get("formal_manifold_query_closure")
        route_trust = adapt.get("route_a_trust_region_state")
        checkpoint_checks = {
            "history_checkpoint_complete": adapt.get("history_checkpoint_complete")
            is True,
            "partial_checkpoint": adapt.get("partial_checkpoint") is True,
            "beam_round_done": str(checkpoint.get("reason") or "")
            == "beam_round_done",
            "nonterminal_checkpoint": checkpoint.get("complete") is False,
            "warm_state_checkpoint": isinstance(warm_checkpoint, Mapping)
            and bool(warm_checkpoint),
            "query_closure_checkpoint": isinstance(query_checkpoint, Mapping)
            and query_checkpoint.get("current_round_finalized") is True,
            "route_trust_state": isinstance(route_trust, Mapping) and bool(route_trust),
            "query_closure_route": isinstance(query_closure, Mapping)
            and query_closure.get("route") == FM_LIVE_ROUTE_ID,
            "jr_selector_not_invoked": isinstance(query_closure, Mapping)
            and query_closure.get("joint_response_selector_invoked") is False,
        }
        if not all(checkpoint_checks.values()):
            failed = [name for name, passed in checkpoint_checks.items() if not passed]
            raise ValueError(
                f"FM stopped checkpoint is not a completed accepted round: {failed}"
            )
        _complete_history(payload, path=current_path)
        if payload.get("no_credentials_serialized") is not True:
            raise ValueError(f"FM stopped checkpoint credential audit failed: {current_path}")

        ansatz_depth = int(adapt["ansatz_depth"])
        error = _positive(adapt["abs_delta_e"])
        branch_id = int(adapt["branch_id"])
        assert isinstance(warm_checkpoint, Mapping)
        manifest_matches = {
            "ansatz_depth": int(raw_row["ansatz_depth"]) == ansatz_depth,
            "abs_delta_e": math.isclose(
                float(raw_row["abs_delta_e"]), error, rel_tol=1.0e-12, abs_tol=1.0e-15
            ),
            "branch_id": int(raw_row["branch_id"]) == branch_id,
            "metric_rank": int(raw_row["metric_rank"])
            == int(warm_checkpoint["rank"]),
            "formal_trust_radius": math.isclose(
                float(raw_row["formal_trust_radius"]),
                float(warm_checkpoint["trust_radius"]),
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            ),
            "curvature_branch": str(raw_row["curvature_branch"])
            == str(warm_checkpoint["curvature_branch"]),
        }
        if not all(manifest_matches.values()):
            failed = [name for name, passed in manifest_matches.items() if not passed]
            raise ValueError(f"FM stopped row/checkpoint mismatch: {failed}")

        curve = _history_curve(current_path, role=role)
        segment = {
            "schema": FM_STOPPED_SNAPSHOT_SCHEMA,
            "live_snapshot": True,
            "scheduler_state": "stopped_snapshot",
            "terminal": False,
            "retrieval_manifest": _rel(manifest_path),
            "retrieval_manifest_sha256": _sha256(manifest_path),
            "proc_id": proc_id,
            "row_id": row_id,
        }
        curve = Curve(
            role=curve.role,
            points=curve.points,
            marker_k=curve.marker_k,
            marker_error=curve.marker_error,
            source_json=curve.source_json,
            source_sha256=curve.source_sha256,
            source_segments=(segment,),
        )
        resource = {
            "regime": regime,
            "method": role,
            "method_display": RESOURCE_METHOD_DISPLAY[role],
            "role": "formal_manifold_stopped_diagnostic",
            "status": "stopped_snapshot",
            "k_pl": int(curve.marker_k),
            "ansatz_depth": ansatz_depth,
            "abs_delta_e": float(curve.marker_error),
            "N2q": None,
            "D2q": None,
            "Dc": None,
            "S": None,
            "resource_status": "pending_stopped_checkpoint_sidecars",
            "prefix_semantics": "user_stopped_completed_round_checkpoint",
            "source_json": _rel(current_path),
            "source_sha256": current_hash,
            "scheduler_state": "stopped_snapshot",
            "terminal": False,
        }
        provenance = {
            "proc_id": proc_id,
            "row_id": row_id,
            "regime": regime,
            "policy": policy,
            "policy_id": policy_id,
            "current_json": _rel(current_path),
            "current_sha256": current_hash,
            "runner_manifest": _rel(runner_path),
            "runner_manifest_sha256": _sha256(runner_path),
            "plan_json": _rel(plan_path),
            "plan_sha256": _sha256(plan_path),
            "route_id": FM_LIVE_ROUTE_ID,
            "qbroyd_epsilon0": epsilon,
            "structural_rollback_enabled": False,
            "controller_round": int(curve.marker_k),
            "ansatz_depth": ansatz_depth,
            "abs_delta_e": float(curve.marker_error),
            "terminal": False,
        }
        rows[regime][role] = {
            "curve": curve,
            "resource": resource,
            "provenance": provenance,
        }
        provenance_entries.append(provenance)

    if seen_procs != expected_procs:
        raise ValueError(
            f"FM stopped-snapshot manifest must contain proc6-11; got {sorted(seen_procs)}"
        )
    return rows, {
        "schema": FM_STOPPED_SNAPSHOT_SCHEMA,
        "manifest_json": _rel(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "batch_id": batch_id,
        "cluster_id": cluster_id,
        "captured_at": captured_at,
        "stopped_at": stopped_at,
        "entry_count": len(provenance_entries),
        "entries": provenance_entries,
        "evidence_class": str(validation.get("evidence_class") or ""),
        "source_value_anchor": str(validation.get("source_value_anchor") or ""),
        "source_locked_sensitivity": validation.get("source_locked_sensitivity"),
        "terminal_evidence": False,
        "resource_fields": "pending_until_validated_checkpoint_sidecars",
        "overlay_scope": "proc6-11_replaces_prior_live_status_rows_only",
    }


def _overlay_fm_stopped_snapshot_rows(
    live_rows: dict[str, dict[str, dict[str, Any]]],
    stopped_rows: Mapping[str, Mapping[str, dict[str, Any]]],
) -> None:
    """Replace only the supplied stopped rows, preserving all other FM evidence."""

    for regime in REGIME_ORDER:
        for role, entry in stopped_rows.get(regime, {}).items():
            live_rows[regime][role] = entry


def _compile_fm_stopped_snapshot_resources(
    stopped_rows: Mapping[str, Mapping[str, dict[str, Any]]],
    *,
    supplemental_dir: Path,
) -> None:
    """Compile stopped fixed prefixes under the report-only Paper-I convention."""

    for regime in REGIME_ORDER:
        for role, entry in stopped_rows.get(regime, {}).items():
            curve = entry["curve"]
            if not isinstance(curve, Curve):
                raise TypeError(f"Invalid FM stopped curve for {regime}/{role}")
            source_json = _repo_path(curve.source_json)
            compiled = _supplemental_resource_row(
                regime=regime,
                method=role,
                source_json=source_json,
                history_position=int(curve.marker_k),
                expected_error=float(curve.marker_error),
                sidecar_json=(
                    supplemental_dir / f"{role}-{regime}-stopped.json"
                ),
            )
            prior = entry["resource"]
            compiled.update(
                role="formal_manifold_stopped_fixed_prefix_diagnostic",
                status="stopped_snapshot",
                ansatz_depth=prior["ansatz_depth"],
                S=None,
                S_source="unavailable_for_stopped_checkpoint",
                resource_status="validated_report_qiskit_query_work_unavailable",
                prefix_semantics="user_stopped_completed_round_fixed_prefix",
                scheduler_state="stopped_snapshot",
                terminal=False,
            )
            entry["resource"] = compiled


def _fm_recovery_path(manifest_path: Path, raw: Any) -> Path:
    path = Path(str(raw or ""))
    if not str(path):
        raise ValueError("FM resource recovery manifest has an empty path")
    return (path if path.is_absolute() else manifest_path.parent / path).resolve()


def _overlay_fm_completed_resource_recovery(
    manifest_path: Path,
    *,
    live_rows: dict[str, dict[str, dict[str, Any]]],
    live_campaign: Mapping[str, Any],
) -> dict[str, Any]:
    """Overlay exact terminal query work and validated report-only Qiskit costs.

    The completed weak-Holstein jobs wrote successful terminal summaries but the
    original transfer intentionally omitted multi-gigabyte result artifacts.
    This loader accepts only the additive, hash-linked compact recovery bundle.
    """

    manifest_path = manifest_path.resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != FM_COMPLETED_RESOURCE_RECOVERY_SCHEMA:
        raise ValueError(
            "Unexpected FM completed-resource recovery schema: "
            f"{manifest.get('schema')!r}"
        )
    batch_id = str(manifest.get("batch_id") or "").strip()
    if not batch_id or batch_id != str(live_campaign.get("batch_id") or ""):
        raise ValueError("FM completed-resource recovery batch mismatch")
    source_manifest = _fm_recovery_path(
        manifest_path, manifest.get("source_retrieval_manifest")
    )
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)
    if _sha256(source_manifest) != str(
        manifest.get("source_retrieval_manifest_sha256") or ""
    ).lower():
        raise ValueError("FM completed-resource source retrieval hash mismatch")

    raw_rows = manifest.get("rows")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise ValueError("FM completed-resource rows must be a JSON array")
    expected = {
        (regime, policy)
        for regime in ("weak-weak", "intermediate-weak", "strong-weak")
        for policy in ("qbroyd_on", "qbroyd_off")
    }
    seen: set[tuple[str, str]] = set()
    recovered_entries: list[dict[str, Any]] = []
    qiskit_count = 0
    query_count = 0
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise TypeError("FM completed-resource row is not an object")
        regime = _normalize_fm_live_regime(raw.get("regime"))
        policy = str(raw.get("policy") or "").strip()
        key = (regime, policy)
        if key not in expected or key in seen:
            raise ValueError(f"Unexpected or duplicate FM recovery row: {key}")
        seen.add(key)
        role = _fm_live_policy_role(policy)
        entry = live_rows.get(regime, {}).get(role)
        if entry is None:
            raise ValueError(f"FM recovery lacks prior live row: {regime}/{policy}")
        provenance = entry.get("provenance")
        row_id = str(raw.get("row_id") or "").strip()
        if not isinstance(provenance, Mapping) or row_id != str(
            provenance.get("row_id") or ""
        ):
            raise ValueError(f"FM recovery row identity mismatch: {row_id}")

        query_path = _fm_recovery_path(
            manifest_path, raw.get("query_work_sidecar")
        )
        if not query_path.is_file():
            raise FileNotFoundError(query_path)
        if _sha256(query_path) != str(
            raw.get("query_work_sidecar_sha256") or ""
        ).lower():
            raise ValueError(f"FM recovered query sidecar hash mismatch: {row_id}")
        query = _read_json(query_path)
        query_total = float(raw["query_work_total"])
        query_checks = {
            "schema": query.get("schema")
            == "formal_manifold_terminal_query_work_stdout_recovery_v1",
            "status": query.get("query_work_status") == "ok",
            "scope": query.get("query_work_scope")
            == "accepted_terminal_lineage",
            "total": math.isclose(
                float(query.get("query_work_total")),
                query_total,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ),
            "winner": math.isclose(
                float((query.get("winning_branch") or {}).get("expanded_query_work")),
                query_total,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ),
            "source": str(query.get("source_full_result_sha256") or "")
            == str(raw.get("omitted_full_result_sha256") or ""),
        }
        if not all(query_checks.values()):
            failed = [name for name, passed in query_checks.items() if not passed]
            raise ValueError(
                f"FM recovered query validation failed for {row_id}: {failed}"
            )
        query_count += 1

        resource = entry["resource"]
        resource.update(
            S=int(round(query_total)),
            S_source=(
                "formal_manifold_terminal_query_work_stdout_recovery."
                "winning_branch.expanded_query_work"
            ),
            query_work_sidecar=_rel(query_path),
            query_work_sidecar_sha256=_sha256(query_path),
        )
        endpoint = raw.get("endpoint")
        qiskit_status = str(raw.get("qiskit_status") or "")
        if endpoint is None:
            if qiskit_status != "unavailable_terminal_operator_sequence_omitted":
                raise ValueError(f"FM recovery endpoint/Qiskit mismatch: {row_id}")
            resource.update(
                resource_status=(
                    "validated_terminal_query_work_qiskit_operator_sequence_unavailable"
                ),
                terminal_query_work_recovered=True,
                terminal_qiskit_unavailable=True,
                S_scope_mismatch_warning=True,
            )
            recovered_entries.append(
                {
                    "row_id": row_id,
                    "regime": regime,
                    "policy": policy,
                    "qiskit_status": qiskit_status,
                    "query_work_total": query_total,
                    "query_work_sidecar": _rel(query_path),
                    "query_work_sidecar_sha256": _sha256(query_path),
                }
            )
            continue
        if not isinstance(endpoint, Mapping):
            raise TypeError(f"FM recovered endpoint is not an object: {row_id}")
        if qiskit_status != "validated_report_qiskit":
            raise ValueError(f"FM recovered Qiskit status mismatch: {row_id}")
        qiskit_path = _fm_recovery_path(manifest_path, raw.get("qiskit_sidecar"))
        if not qiskit_path.is_file():
            raise FileNotFoundError(qiskit_path)
        if _sha256(qiskit_path) != str(raw.get("qiskit_sidecar_sha256") or "").lower():
            raise ValueError(f"FM recovered Qiskit sidecar hash mismatch: {row_id}")
        qiskit = _read_json(qiskit_path)
        error = _positive(endpoint.get("abs_delta_e"))
        controller_round = int(endpoint["controller_round"])
        ansatz_depth = int(endpoint["ansatz_depth"])
        qiskit_checks = {
            "validated": qiskit.get("compiled_resource_qiskit_validated") is True,
            "status": qiskit.get("compiled_circuit_stats_status") == "ok",
            "convention": qiskit.get("compile_convention")
            == "table_i_basis_gate_transpile_v1",
            "source": str(qiskit.get("source_full_result_sha256") or "")
            == str(raw.get("omitted_full_result_sha256") or ""),
            "error": math.isclose(
                float(qiskit.get("primary_error_at_prefix")),
                error,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            ),
            "query_total": math.isclose(
                float(qiskit.get("instrumented_runtime_S")),
                query_total,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ),
            "operator_count": int(
                ((qiskit.get("replay") or {}).get("replayed_operator_count"))
            )
            == ansatz_depth,
        }
        if not all(qiskit_checks.values()):
            failed = [name for name, passed in qiskit_checks.items() if not passed]
            raise ValueError(
                f"FM recovered Qiskit validation failed for {row_id}: {failed}"
            )
        prior_curve: Curve = entry["curve"]
        if controller_round < int(prior_curve.marker_k):
            raise ValueError(f"FM recovered endpoint regresses prior marker: {row_id}")
        recovery_endpoint = {
            "k": controller_round,
            "error": error,
            "ansatz_depth": ansatz_depth,
            "state": "science_complete_packaging_failed",
            "trajectory_relation": "same_row_terminal_stdout_recovery",
            "marker_only": True,
            "terminal_metric_validated": True,
            "terminal_resource_recovery": True,
            "source_recovery_manifest": _rel(manifest_path),
            "source_recovery_manifest_sha256": _sha256(manifest_path),
            "source_full_result_sha256": raw.get("omitted_full_result_sha256"),
        }
        entry["curve"] = Curve(
            role=prior_curve.role,
            points=prior_curve.points,
            marker_k=controller_round,
            marker_error=error,
            source_json=prior_curve.source_json,
            source_sha256=prior_curve.source_sha256,
            source_segments=(*prior_curve.source_segments, recovery_endpoint),
        )
        resource.update(
            status="science_complete_packaging_failed",
            k_pl=controller_round,
            ansatz_depth=ansatz_depth,
            abs_delta_e=error,
            N2q=int(qiskit["compiled_count_2q_total"]),
            D2q=int(qiskit["compiled_depth_2q_total"]),
            Dc=int(qiskit["compiled_depth_total"]),
            resource_status="validated_recovered_terminal_qiskit_and_query_work",
            prefix_semantics="terminal_selected_endpoint_recovered_from_stdout",
            terminal=True,
            terminal_metric_validated=True,
            checkpoint_asterisk=False,
            qiskit_sidecar=_rel(qiskit_path),
            qiskit_sidecar_sha256=_sha256(qiskit_path),
            qiskit_compile_convention="table_i_basis_gate_transpile_v1",
        )
        entry["provenance"]["terminal_resource_recovery"] = recovery_endpoint
        qiskit_count += 1
        recovered_entries.append(
            {
                "row_id": row_id,
                "regime": regime,
                "policy": policy,
                "controller_round": controller_round,
                "ansatz_depth": ansatz_depth,
                "abs_delta_e": error,
                "N2q": resource["N2q"],
                "D2q": resource["D2q"],
                "Dc": resource["Dc"],
                "S": resource["S"],
                "qiskit_sidecar": _rel(qiskit_path),
                "qiskit_sidecar_sha256": _sha256(qiskit_path),
                "query_work_sidecar": _rel(query_path),
                "query_work_sidecar_sha256": _sha256(query_path),
            }
        )

    if seen != expected:
        raise ValueError(f"FM completed-resource coverage mismatch: {sorted(seen)}")
    validation = manifest.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError("FM completed-resource validation block is missing")
    if (
        int(validation.get("exact_terminal_query_work_rows", -1)) != query_count
        or int(validation.get("validated_qiskit_rows", -1)) != qiskit_count
        or str(validation.get("qiskit_compile_convention") or "")
        != "table_i_basis_gate_transpile_v1"
    ):
        raise ValueError("FM completed-resource validation totals mismatch")
    return {
        "schema": FM_COMPLETED_RESOURCE_RECOVERY_SCHEMA,
        "manifest_json": _rel(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "source_retrieval_manifest": _rel(source_manifest),
        "source_retrieval_manifest_sha256": _sha256(source_manifest),
        "batch_id": batch_id,
        "created_utc": str(manifest.get("created_utc") or ""),
        "entry_count": len(recovered_entries),
        "exact_terminal_query_work_rows": query_count,
        "validated_qiskit_rows": qiskit_count,
        "terminal_operator_sequence_unavailable_rows": list(
            validation.get("terminal_operator_sequence_unavailable_rows") or ()
        ),
        "entries": recovered_entries,
        "resource_fields": (
            "five exact terminal Paper-I Qiskit rows and six exact terminal "
            "winning-lineage query-work totals"
        ),
    }


def _stitched_history_curve(paths: Sequence[Path], *, role: str) -> Curve:
    if not paths:
        raise ValueError("At least one history segment is required")
    payloads = [_read_json(path) for path in paths]
    histories = [
        _complete_history(payload, path=path)
        for payload, path in zip(payloads, paths)
    ]
    exact_values = [float(payload["adapt_vqe"]["exact_gs_energy"]) for payload in payloads]
    if not all(math.isclose(value, exact_values[0], rel_tol=0.0, abs_tol=1.0e-13) for value in exact_values):
        raise ValueError(f"Stitched histories use different exact energies: {exact_values}")
    initial_error = _positive(float(histories[0][0]["energy_before_opt"]) - exact_values[0])
    points: list[tuple[int, float]] = [(0, initial_error)]
    source_segments: list[Mapping[str, Any]] = []
    offset = 0
    for path, history in zip(paths, histories):
        source_segments.append(
            {
                "source_json": _rel(path),
                "source_sha256": _sha256(path),
                "controller_round_offset": offset,
                "history_count": len(history),
            }
        )
        for row in history:
            points.append((len(points), _positive(row["delta_abs_current"])))
        offset += len(history)
    terminal_adapt = payloads[-1]["adapt_vqe"]
    if terminal_adapt.get("abs_delta_e") is not None:
        points[-1] = (points[-1][0], _positive(terminal_adapt["abs_delta_e"]))
    return Curve(
        role=role,
        points=tuple(points),
        marker_k=points[-1][0],
        marker_error=points[-1][1],
        source_json=_rel(paths[-1]),
        source_sha256=_sha256(paths[-1]),
        source_segments=tuple(source_segments),
    )


def _verified_segment_estimator_ledger(
    *,
    path: Path,
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Load one segment's exact winning/all primitive sets when available.

    Terminal ledgers retain authenticated winning-branch IDs, from which the
    winning-lineage view is reconstructed and compared exactly with the stored
    summary. Completed-round checkpoints retain the full ledger but not that
    terminal summary, so their winning branch set is reconstructed from the
    completed history. A declared sidecar that is malformed or
    hash-inconsistent fails closed; an incomplete terminal sidecar may fall
    back only to an authenticated finalized-round checkpoint and records the
    terminal blocker explicitly.
    """

    from pipelines.static_adapt.estimator_call_ledger import EstimatorCallLedger

    adapt = payload.get("adapt_vqe")
    if not isinstance(adapt, Mapping):
        return None, "segment_missing_adapt_vqe"

    terminal_accounting = adapt.get("estimator_call_accounting")
    terminal_pointer = (
        terminal_accounting.get("sidecar")
        if isinstance(terminal_accounting, Mapping)
        else None
    )
    terminal_candidate: tuple[Path, str | None, str] | None = None
    if isinstance(terminal_pointer, Mapping):
        raw_pointer_path = terminal_pointer.get("path")
        if raw_pointer_path:
            pointer_path = Path(str(raw_pointer_path))
            if not pointer_path.is_absolute():
                relative_candidates = (
                    path.parent / pointer_path,
                    REPO_ROOT / pointer_path,
                )
                pointer_path = next(
                    (
                        candidate
                        for candidate in relative_candidates
                        if candidate.is_file()
                    ),
                    relative_candidates[0],
                )
            terminal_candidate = (
                pointer_path.resolve(),
                str(terminal_pointer.get("sha256") or "").lower() or None,
                "result_estimator_call_ledger_pointer",
            )
    adjacent_path = path.with_name("estimator_call_ledger.json")
    if terminal_candidate is None and adjacent_path.is_file():
        terminal_candidate = (
            adjacent_path.resolve(),
            None,
            "adjacent_estimator_call_ledger",
        )

    def _checkpoint_candidate() -> tuple[
        tuple[Path, str, str] | None, str | None
    ]:
        checkpoint_pointer = adapt.get("estimator_call_ledger_checkpoint")
        if not isinstance(checkpoint_pointer, Mapping):
            return None, "full_estimator_call_ledger_unavailable"
        if checkpoint_pointer.get("schema") not in {
            "paper_i_estimator_call_ledger_checkpoint_pointer_v1",
            "paper_i_estimator_call_ledger_checkpoint_pointer_v2",
        }:
            raise ValueError(
                f"Invalid estimator-ledger checkpoint pointer schema for {path}"
            )
        if checkpoint_pointer.get("enabled") is not True:
            return None, "checkpoint_estimator_ledger_disabled"
        if checkpoint_pointer.get("status") != "complete":
            return None, "checkpoint_estimator_ledger_incomplete"
        if checkpoint_pointer.get("current_round_finalized") is not True:
            return None, "checkpoint_estimator_round_not_finalized"
        raw_name = str(
            checkpoint_pointer.get("path")
            or checkpoint_pointer.get("sidecar_path")
            or ""
        )
        pointer_path = Path(raw_name)
        if (
            not raw_name
            or pointer_path.is_absolute()
            or len(pointer_path.parts) != 1
            or pointer_path.name != raw_name
        ):
            raise ValueError(
                f"Invalid estimator-ledger checkpoint path for {path}: {raw_name!r}"
            )
        expected_hash = str(
            checkpoint_pointer.get("sha256")
            or checkpoint_pointer.get("sidecar_sha256")
            or ""
        ).lower()
        if re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
            raise ValueError(
                f"Estimator-ledger checkpoint lacks an authenticated hash: {path}"
            )
        return (
            (
                (path.parent / pointer_path).resolve(),
                expected_hash,
                "completed_round_estimator_call_ledger_checkpoint",
            ),
            None,
        )

    def _read_candidate(
        candidate: tuple[Path, str | None, str],
    ) -> tuple[Path, str, str, Mapping[str, Any]]:
        ledger_path, expected_hash, source_kind = candidate
        if not ledger_path.is_file():
            raise FileNotFoundError(ledger_path)
        actual_hash = _sha256(ledger_path)
        if expected_hash is not None and actual_hash != expected_hash:
            raise ValueError(
                f"Estimator-ledger sidecar hash mismatch: {ledger_path}"
            )
        return ledger_path, actual_hash, source_kind, _read_json(ledger_path)

    terminal_fallback_blocker: str | None = None
    candidate: tuple[Path, str | None, str] | None = terminal_candidate
    checkpoint_mode = False
    if candidate is not None:
        ledger_path, actual_hash, source_kind, sidecar = _read_candidate(candidate)
        terminal_sidecar_schema = sidecar.get("schema")
        if terminal_sidecar_schema not in {
            "paper_i_estimator_call_ledger_sidecar_v1",
            "paper_i_estimator_call_ledger_sidecar_v2",
        }:
            raise ValueError(
                f"Unsupported estimator-ledger sidecar schema: {ledger_path}"
            )
        ledger_payload = sidecar.get("ledger")
        accounting = sidecar.get("accounting")
        if not isinstance(ledger_payload, Mapping):
            terminal_fallback_blocker = "terminal_sidecar_missing_full_ledger"
        elif (
            not isinstance(accounting, Mapping)
            or accounting.get("complete") is not True
        ):
            terminal_fallback_blocker = "terminal_estimator_accounting_not_complete"
        else:
            unique_winning_key = (
                "winning_lineage_unique_primitive_diagnostic"
                if terminal_sidecar_schema
                == "paper_i_estimator_call_ledger_sidecar_v2"
                else "winning_lineage"
            )
            unique_all_key = (
                "all_branch_unique_primitive_diagnostic"
                if terminal_sidecar_schema
                == "paper_i_estimator_call_ledger_sidecar_v2"
                else "all_branch_search_work"
            )
        if (
            terminal_fallback_blocker is None
            and (
                not isinstance(accounting, Mapping)
                or not isinstance(accounting.get(unique_winning_key), Mapping)
                or not isinstance(accounting.get(unique_all_key), Mapping)
            )
        ):
            terminal_fallback_blocker = (
                "terminal_sidecar_missing_unique_primitive_diagnostics"
            )
        elif (
            terminal_fallback_blocker is None
            and accounting.get("winning_branch_ids") is None
        ):
            terminal_fallback_blocker = (
                "terminal_sidecar_missing_winning_branch_ids"
            )
        elif (
            terminal_fallback_blocker is None
            and not isinstance(accounting.get("winning_branch_ids"), list)
        ):
            raise ValueError(
                f"Terminal winning_branch_ids must be a list: {ledger_path}"
            )
    else:
        ledger_path = Path()
        actual_hash = ""
        source_kind = ""
        sidecar = {}
        ledger_payload = None
        accounting = None

    if candidate is None or terminal_fallback_blocker is not None:
        checkpoint_candidate, checkpoint_blocker = _checkpoint_candidate()
        if checkpoint_candidate is None:
            if terminal_fallback_blocker is not None:
                return None, terminal_fallback_blocker
            return None, str(
                checkpoint_blocker or "full_estimator_call_ledger_unavailable"
            )
        ledger_path, actual_hash, source_kind, sidecar = _read_candidate(
            checkpoint_candidate
        )
        if sidecar.get("schema") not in {
            "paper_i_estimator_call_ledger_checkpoint_sidecar_v1",
            "paper_i_estimator_call_ledger_checkpoint_sidecar_v2",
        }:
            raise ValueError(
                f"Unsupported estimator-ledger sidecar schema: {ledger_path}"
            )
        checkpoint = sidecar.get("checkpoint")
        if (
            not isinstance(checkpoint, Mapping)
            or checkpoint.get("current_round_finalized") is not True
        ):
            blocker = "checkpoint_sidecar_round_not_finalized"
            if terminal_fallback_blocker is not None:
                blocker = f"{terminal_fallback_blocker};{blocker}"
            return None, blocker
        ledger_payload = sidecar.get("ledger")
        if not isinstance(ledger_payload, Mapping):
            blocker = "checkpoint_sidecar_missing_full_ledger"
            if terminal_fallback_blocker is not None:
                blocker = f"{terminal_fallback_blocker};{blocker}"
            return None, blocker
        accounting = None
        checkpoint_mode = True

    if not isinstance(ledger_payload, Mapping):
        raise AssertionError("validated estimator sidecar lost its ledger payload")
    ledger = EstimatorCallLedger.from_payload(ledger_payload)
    all_summary = ledger.summary()
    all_ids = set(str(value) for value in all_summary.get("primitive_ids", []))

    if not checkpoint_mode:
        if not isinstance(accounting, Mapping):
            raise AssertionError("validated terminal sidecar lost its accounting")
        raw_winning_branch_ids = accounting.get("winning_branch_ids")
        if not isinstance(raw_winning_branch_ids, list):
            raise AssertionError("validated terminal winning_branch_ids changed type")
        winning_branch_ids = [str(value) for value in raw_winning_branch_ids]
        if (
            any(not value for value in winning_branch_ids)
            or len(winning_branch_ids) != len(set(winning_branch_ids))
        ):
            raise ValueError(
                f"Terminal winning_branch_ids are invalid: {ledger_path}"
            )
        expected_winning_summary = (
            ledger.summary(
                branch_ids=winning_branch_ids,
                include_unbranched=True,
            )
            if winning_branch_ids
            else dict(all_summary)
        )
        sidecar_schema = sidecar.get("schema")
        if sidecar_schema == "paper_i_estimator_call_ledger_sidecar_v2":
            stored_winning_summary = accounting.get(
                "winning_lineage_unique_primitive_diagnostic"
            )
            all_stored_summary = accounting.get(
                "all_branch_unique_primitive_diagnostic"
            )
        else:
            stored_winning_summary = accounting.get("winning_lineage")
            all_stored_summary = accounting.get("all_branch_search_work")
        if dict(all_stored_summary) != all_summary:
            raise ValueError(
                "Stored all-branch estimator summary does not match ledger: "
                f"{ledger_path}"
            )
        if dict(stored_winning_summary) != expected_winning_summary:
            raise ValueError(
                "Stored winning-lineage estimator summary does not match "
                f"authenticated winning_branch_ids: {ledger_path}"
            )
        winning_summary = dict(expected_winning_summary)
        winning_ids = set(
            str(value) for value in winning_summary.get("primitive_ids", [])
        )
        discarded_stored = accounting.get(
            "discarded_branch_only_by_unique_set_difference"
        )
    else:
        history = _complete_history(payload, path=path)
        winning_branch_ids = {
            str(row[key])
            for row in history
            if isinstance(row, Mapping)
            for key in ("branch_id", "parent_branch_id")
            if row.get(key) is not None
        }
        for raw_branch in (
            adapt.get("branch_id"),
            (payload.get("checkpoint") or {}).get("branch_id")
            if isinstance(payload.get("checkpoint"), Mapping)
            else None,
        ):
            if raw_branch is not None:
                winning_branch_ids.add(str(raw_branch))
        all_branch_labels = {
            str(value)
            for value in dict(
                ledger.occurrence_summary().get(
                    "occurrence_count_by_consumer_branch", {}
                )
            )
            if str(value) != "__unbranched__"
        }
        settings = payload.get("settings")
        formal_state = adapt.get("formal_manifold_warm_start")
        route_values = {
            str(value)
            for value in (
                payload.get("route_id"),
                adapt.get("adapt_reoptimization_route"),
                settings.get("adapt_reoptimization_route")
                if isinstance(settings, Mapping)
                else None,
                formal_state.get("route")
                if isinstance(formal_state, Mapping)
                else None,
            )
            if value is not None
        }
        formal_manifold_lineage = bool(
            FM_LIVE_ROUTE_ID in route_values
            or any(
                label == "single_frontier:0"
                or label.startswith("beam_branch:")
                for label in all_branch_labels
            )
        )
        if formal_manifold_lineage:
            winning_branch_ids.update(
                f"beam_branch:{branch_id}"
                for branch_id in tuple(winning_branch_ids)
                if str(branch_id).lstrip("-").isdigit()
            )
            if "single_frontier:0" in all_branch_labels:
                winning_branch_ids.add("single_frontier:0")
        if all_branch_labels and not winning_branch_ids:
            return None, "checkpoint_winning_branch_lineage_unavailable"
        winning_summary = (
            ledger.summary(
                branch_ids=sorted(winning_branch_ids),
                include_unbranched=True,
            )
            if all_branch_labels
            else dict(all_summary)
        )
        winning_ids = set(
            str(value) for value in winning_summary.get("primitive_ids", [])
        )
        discarded_stored = None

    discarded_ids = all_ids.difference(winning_ids)
    if isinstance(discarded_stored, Mapping):
        stored_discarded_ids = set(
            str(value) for value in discarded_stored.get("primitive_ids", [])
        )
        if stored_discarded_ids != discarded_ids:
            raise ValueError(
                f"Discarded estimator set difference does not close: {ledger_path}"
            )

    return (
        {
            "source_kind": source_kind,
            "ledger_path": ledger_path,
            "ledger_sha256": actual_hash,
            "ledger_fingerprint": str(ledger_payload.get("ledger_fingerprint") or ""),
            "all_summary": dict(all_summary),
            "winning_summary": dict(winning_summary),
            "all_ids": all_ids,
            "winning_ids": winning_ids,
            "discarded_ids": discarded_ids,
            "terminal_fallback_blocker": terminal_fallback_blocker,
            "terminal_fallback_used": bool(
                checkpoint_mode and terminal_fallback_blocker is not None
            ),
        },
        None,
    )


def _legacy_segment_query_work(
    *,
    path: Path,
    payload: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the old per-segment quantity, explicitly as a non-exact proxy."""

    from pipelines.exact_bench.snake_table_i_measurement_work import (
        snake_algorithmic_work_from_payload,
    )

    component_keys = (
        "S_alg_N_H_outer_eval",
        "S_alg_N_H_refit_eval",
        "S_alg_N_grad_probe",
        "S_alg_N_metric_probe",
        "S_alg_N_other_quantum",
    )
    sidecar_path = path.with_name("query_work_sidecar.json")
    sidecar = _read_json(sidecar_path) if sidecar_path.is_file() else None
    if sidecar is not None:
        checks = {
            "status": sidecar.get("status") == "complete",
            "query_work_status": sidecar.get("query_work_status") == "ok",
            "query_work_scope": sidecar.get("query_work_scope")
            == "winner_lineage_terminal",
            "query_work_total": sidecar.get("query_work_total") is not None,
            "components": isinstance(sidecar.get("query_work_components"), Mapping),
        }
        if not all(checks.values()):
            raise ValueError(
                f"Invalid winning-lineage query sidecar for {path}: {checks}"
            )
        raw_components = sidecar["query_work_components"]
        components = {
            "S_alg_N_H_outer_eval": float(raw_components["N_H_outer"]),
            "S_alg_N_H_refit_eval": float(raw_components["N_H_refit"]),
            "S_alg_N_grad_probe": float(raw_components["N_grad"]),
            "S_alg_N_metric_probe": float(raw_components["N_metric"]),
            "S_alg_N_other_quantum": float(raw_components.get("N_other_quantum", 0.0)),
        }
        total = float(sidecar["query_work_total"])
        audit_status = "validated_query_work_sidecar_but_no_primitive_ids"
        source_kind = "adjacent_query_work_sidecar"
    else:
        work, audit = snake_algorithmic_work_from_payload(
            payload,
            scope="display_prefix",
            history_position=len(history),
            source_label=str(path),
        )
        if work.get("S_alg_status") != "ok" or work.get("S_alg") is None:
            return {
                "status": "unavailable",
                "S_alg": None,
                "components": None,
                "audit_status": str(audit.get("status")),
                "source_kind": "payload_formula_fallback_no_sidecar",
                "query_work_sidecar": None,
            }
        components = {}
        for key in component_keys:
            value = work.get(key)
            if value is None:
                return {
                    "status": "unavailable",
                    "S_alg": None,
                    "components": None,
                    "audit_status": f"missing_{key}",
                    "source_kind": "payload_formula_fallback_no_sidecar",
                    "query_work_sidecar": None,
                }
            components[key] = float(value)
        total = float(work["S_alg"])
        audit_status = str(audit.get("status"))
        source_kind = "payload_formula_fallback_no_sidecar"
    if not math.isclose(total, sum(components.values()), rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"Legacy query-work component closure failed: {path}")
    return {
        "status": "legacy_proxy_not_exact",
        "S_alg": total,
        "components": components,
        "audit_status": audit_status,
        "source_kind": source_kind,
        "query_work_sidecar": _rel(sidecar_path) if sidecar_path.is_file() else None,
    }


def _stitched_winning_lineage_query_work(paths: Sequence[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("At least one query-work segment is required")
    component_keys = (
        "S_alg_N_H_outer_eval",
        "S_alg_N_H_refit_eval",
        "S_alg_N_grad_probe",
        "S_alg_N_metric_probe",
        "S_alg_N_other_quantum",
    )
    report_key_by_ledger_component = {
        "N_H_outer": "S_alg_N_H_outer_eval",
        "N_H_refit": "S_alg_N_H_refit_eval",
        "N_grad": "S_alg_N_grad_probe",
        "N_metric": "S_alg_N_metric_probe",
    }
    segments: list[dict[str, Any]] = []
    exact_segments: list[dict[str, Any]] = []
    exact_blockers: list[str] = []
    legacy_components = {key: 0.0 for key in component_keys}
    legacy_total_available = True
    legacy_total = 0.0
    for path in paths:
        payload = _read_json(path)
        history = _complete_history(payload, path=path)
        exact, exact_blocker = _verified_segment_estimator_ledger(
            path=path,
            payload=payload,
        )
        if exact is not None:
            exact_segments.append(exact)
            winning_summary = exact["winning_summary"]
            segment_components = {
                report_key: float(winning_summary.get(ledger_key, 0))
                for ledger_key, report_key in report_key_by_ledger_component.items()
            }
            segment_components["S_alg_N_other_quantum"] = 0.0
            segment_total = float(winning_summary["S_unique"])
            segment_row = {
                "source_json": _rel(path),
                "source_sha256": _sha256(path),
                "history_count": len(history),
                "S_alg": None,
                "S_alg_status": "unavailable_raw_occurrence_stitching",
                "S_unique": int(segment_total),
                "S_unique_status": "validated_primitive_identity_ledger",
                "audit_status": "validated_unique_primitive_diagnostic_only",
                "source_kind": exact["source_kind"],
                "estimator_call_ledger": _rel(exact["ledger_path"]),
                "estimator_call_ledger_sha256": exact["ledger_sha256"],
                "estimator_call_ledger_fingerprint": exact["ledger_fingerprint"],
                "winning_unique_primitive_count": len(exact["winning_ids"]),
                "all_unique_primitive_count": len(exact["all_ids"]),
                "discarded_only_unique_primitive_count": len(
                    exact["discarded_ids"]
                ),
            }
            if exact.get("terminal_fallback_used") is True:
                segment_row.update(
                    {
                        "terminal_fallback_used": True,
                        "terminal_fallback_blocker": str(
                            exact["terminal_fallback_blocker"]
                        ),
                    }
                )
        else:
            blocker = str(exact_blocker or "exact_estimator_ledger_unavailable")
            exact_blockers.append(f"{_rel(path)}:{blocker}")
            legacy = _legacy_segment_query_work(
                path=path, payload=payload, history=history
            )
            segment_components = legacy.get("components")
            segment_total = legacy.get("S_alg")
            if segment_total is None or not isinstance(segment_components, Mapping):
                legacy_total_available = False
            segment_row = {
                "source_json": _rel(path),
                "source_sha256": _sha256(path),
                "history_count": len(history),
                "S_alg": segment_total,
                "S_alg_status": str(legacy["status"]),
                "audit_status": legacy["audit_status"],
                "source_kind": legacy["source_kind"],
                "query_work_sidecar": legacy["query_work_sidecar"],
                "exact_blocker": blocker,
            }
        if segment_total is not None and isinstance(segment_components, Mapping):
            legacy_total += float(segment_total)
            for key in component_keys:
                legacy_components[key] += float(segment_components.get(key, 0.0))
        segments.append(segment_row)

    if exact_blockers:
        return {
            "schema": "jr_snake_stitched_query_accounting_v3",
            "status": "unavailable_raw_occurrence_stitching",
            "S_alg_work_scope": (
                "unavailable_without_authenticated_all_execution_occurrence_stitching"
            ),
            "S_alg": None,
            "components": None,
            "S_unique": None,
            "S_unique_components": None,
            "segments": segments,
            "exact_blockers": exact_blockers,
            "legacy_proxy": {
                "status": (
                    "legacy_segment_sum_not_primitive_deduplicated"
                    if legacy_total_available
                    else "unavailable"
                ),
                "S_alg": legacy_total if legacy_total_available else None,
                "components": (
                    legacy_components if legacy_total_available else None
                ),
                "warning": (
                    "Segment totals are not a scientific stitched S_alg because "
                    "continuation-boundary primitive identities cannot be unioned."
                ),
            },
            "unique_primitive_union_validated": False,
            "branch_is_consumer_metadata_not_primitive_identity": True,
            "discarded_branch_search_work_included": False,
            "promotion_blocker": (
                "winning-lineage or unique-primitive segment quantities cannot "
                "replace all executed logical estimator occurrences"
            ),
        }

    winning_ids: set[str] = set()
    all_ids: set[str] = set()
    winning_component_by_id: dict[str, str] = {}
    all_component_by_id: dict[str, str] = {}
    winning_cross_component_ids: set[str] = set()
    all_cross_component_ids: set[str] = set()
    segment_winning_unique_sum = 0
    segment_all_unique_sum = 0
    for exact in exact_segments:
        winning_summary = exact["winning_summary"]
        all_summary = exact["all_summary"]
        segment_winning_unique_sum += len(exact["winning_ids"])
        segment_all_unique_sum += len(exact["all_ids"])
        for primitive_id, component in dict(
            winning_summary.get("component_by_primitive_id", {})
        ).items():
            primitive_id = str(primitive_id)
            component = str(component)
            prior = winning_component_by_id.setdefault(primitive_id, component)
            if prior != component:
                winning_cross_component_ids.add(primitive_id)
        for primitive_id, component in dict(
            all_summary.get("component_by_primitive_id", {})
        ).items():
            primitive_id = str(primitive_id)
            component = str(component)
            prior = all_component_by_id.setdefault(primitive_id, component)
            if prior != component:
                all_cross_component_ids.add(primitive_id)
        winning_ids.update(exact["winning_ids"])
        all_ids.update(exact["all_ids"])

    if not winning_ids.issubset(all_ids):
        raise AssertionError("Stitched winning primitive set is not a subset of all work")
    discarded_ids = all_ids.difference(winning_ids)
    missing_winning_components = winning_ids.difference(winning_component_by_id)
    missing_all_components = all_ids.difference(all_component_by_id)
    if missing_winning_components or missing_all_components:
        raise ValueError("Stitched estimator ledgers omit primitive component metadata")

    components = {key: 0 for key in component_keys}
    for primitive_id in winning_ids:
        ledger_component = winning_component_by_id[primitive_id]
        try:
            report_component = report_key_by_ledger_component[ledger_component]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported stitched estimator component: {ledger_component!r}"
            ) from exc
        components[report_component] += 1
    discarded_components = {key: 0 for key in component_keys}
    for primitive_id in discarded_ids:
        ledger_component = all_component_by_id[primitive_id]
        try:
            report_component = report_key_by_ledger_component[ledger_component]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported discarded estimator component: {ledger_component!r}"
            ) from exc
        discarded_components[report_component] += 1

    total = int(sum(components.values()))
    discarded_total = int(sum(discarded_components.values()))
    if total != len(winning_ids) or discarded_total != len(discarded_ids):
        raise AssertionError("Stitched estimator primitive partition did not close")
    return {
        "schema": "jr_snake_stitched_query_accounting_v3",
        "status": "unavailable_raw_occurrence_stitching",
        "S_alg_work_scope": (
            "unavailable_without_authenticated_all_execution_occurrence_stitching"
        ),
        "S_alg": None,
        "components": None,
        "S_unique": total,
        "S_unique_components": components,
        "segments": segments,
        "unique_primitive_union_validated": True,
        "winning_primitive_ids": sorted(winning_ids),
        "all_executed_primitive_ids": sorted(all_ids),
        "winning_component_by_primitive_id": {
            primitive_id: winning_component_by_id[primitive_id]
            for primitive_id in sorted(winning_ids)
        },
        "all_component_by_primitive_id": {
            primitive_id: all_component_by_id[primitive_id]
            for primitive_id in sorted(all_ids)
        },
        "discarded_branch_operational_overhead": {
            "definition": "all_executed_unique_ids_minus_winning_lineage_unique_ids",
            "S_unique": discarded_total,
            "S_unique_components": discarded_components,
            "primitive_ids": sorted(discarded_ids),
        },
        "continuation_boundary_deduplication": {
            "segment_winning_unique_sum": int(segment_winning_unique_sum),
            "stitched_winning_unique_union": int(len(winning_ids)),
            "deduplicated_boundary_primitive_count": int(
                segment_winning_unique_sum - len(winning_ids)
            ),
            "segment_all_unique_sum": int(segment_all_unique_sum),
            "stitched_all_unique_union": int(len(all_ids)),
            "deduplicated_all_boundary_primitive_count": int(
                segment_all_unique_sum - len(all_ids)
            ),
            "winning_cross_component_reuse_primitive_ids": sorted(
                winning_cross_component_ids
            ),
            "all_cross_component_reuse_primitive_ids": sorted(
                all_cross_component_ids
            ),
            "component_assignment_policy": "earliest_segment_consumer_v1",
        },
        "branch_is_consumer_metadata_not_primitive_identity": True,
        "discarded_branch_search_work_included": False,
        "promotion_blocker": (
            "the unique-primitive union omits repeated evaluations and discarded "
            "all-execution work required by clean S_alg"
        ),
    }


def _validate_exact_stitched_query_work_v2(
    query_work: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and quarantine a historical unique-primitive receipt.

    The v2 artifact called its primitive-ID union ``S_alg``.  That quantity is
    only ``S_unique``: it omits repeated estimator invocations and excludes
    discarded all-execution work.  Validation is retained for provenance, but
    callers must never promote the historical field to scientific ``S_alg``.
    """

    report_key_by_ledger_component = {
        "N_H_outer": "S_alg_N_H_outer_eval",
        "N_H_refit": "S_alg_N_H_refit_eval",
        "N_grad": "S_alg_N_grad_probe",
        "N_metric": "S_alg_N_metric_probe",
    }
    component_keys = tuple(report_key_by_ledger_component.values()) + (
        "S_alg_N_other_quantum",
    )

    checks = {
        "schema": query_work.get("schema")
        == "jr_snake_stitched_winning_lineage_query_work_v2",
        "status": query_work.get("status") == "ok",
        "scope": query_work.get("S_alg_work_scope")
        == "winning_lineage_unique_primitive_union",
        "primitive_union": query_work.get("primitive_union_validated") is True,
        "discarded_excluded": query_work.get(
            "discarded_branch_search_work_included"
        )
        is False,
    }
    if not all(checks.values()):
        raise ValueError(f"Invalid exact stitched query-work contract: {checks}")

    def _primitive_ids(field: str) -> list[str]:
        raw = query_work.get(field)
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError(f"Exact stitched query work requires {field}.")
        values = [str(value) for value in raw]
        if any(not value for value in values) or len(values) != len(set(values)):
            raise ValueError(
                f"Exact stitched query work has invalid or duplicate {field}."
            )
        return values

    def _nonnegative_integral(value: Any, *, field: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{field} must be a nonnegative integer.")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be a nonnegative integer.") from exc
        if not math.isfinite(numeric) or numeric < 0.0 or not numeric.is_integer():
            raise ValueError(f"{field} must be a nonnegative integer.")
        return int(numeric)

    def _component_map(field: str, ids: set[str]) -> dict[str, str]:
        raw = query_work.get(field)
        if not isinstance(raw, Mapping):
            raise ValueError(f"Exact stitched query work requires {field}.")
        mapping = {str(key): str(value) for key, value in raw.items()}
        if set(mapping) != ids:
            raise ValueError(f"{field} keys do not close over its primitive IDs.")
        unsupported = sorted(
            set(mapping.values()).difference(report_key_by_ledger_component)
        )
        if unsupported:
            raise ValueError(f"{field} has unsupported components: {unsupported}")
        return mapping

    def _declared_components(raw: Any, *, field: str) -> dict[str, int]:
        if not isinstance(raw, Mapping) or set(raw) != set(component_keys):
            raise ValueError(f"{field} must contain the canonical component keys.")
        return {
            key: _nonnegative_integral(raw[key], field=f"{field}.{key}")
            for key in component_keys
        }

    winning_list = _primitive_ids("winning_primitive_ids")
    all_list = _primitive_ids("all_executed_primitive_ids")
    winning_ids = set(winning_list)
    all_ids = set(all_list)
    if not winning_ids.issubset(all_ids):
        raise ValueError("Winning stitched primitives are not a subset of all work.")
    winning_map = _component_map(
        "winning_component_by_primitive_id", winning_ids
    )
    all_map = _component_map("all_component_by_primitive_id", all_ids)

    reconstructed_components = {key: 0 for key in component_keys}
    for primitive_id in winning_ids:
        reconstructed_components[
            report_key_by_ledger_component[winning_map[primitive_id]]
        ] += 1
    declared_components = _declared_components(
        query_work.get("components"), field="components"
    )
    if declared_components != reconstructed_components:
        raise ValueError("Winning stitched component totals do not close.")
    s_alg = _nonnegative_integral(query_work.get("S_alg"), field="S_alg")
    if s_alg != len(winning_ids) or s_alg != sum(declared_components.values()):
        raise ValueError("Winning stitched S_alg does not close over primitive IDs.")

    discarded_ids = all_ids.difference(winning_ids)
    discarded = query_work.get("discarded_branch_operational_overhead")
    if not isinstance(discarded, Mapping):
        raise ValueError("Exact stitched query work requires discarded overhead.")
    if discarded.get("definition") != (
        "all_executed_unique_ids_minus_winning_lineage_unique_ids"
    ):
        raise ValueError("Discarded stitched work has the wrong set definition.")
    discarded_list_raw = discarded.get("primitive_ids")
    if not isinstance(discarded_list_raw, Sequence) or isinstance(
        discarded_list_raw, (str, bytes)
    ):
        raise ValueError("Discarded stitched work requires primitive IDs.")
    discarded_list = [str(value) for value in discarded_list_raw]
    if len(discarded_list) != len(set(discarded_list)) or set(discarded_list) != discarded_ids:
        raise ValueError("Discarded stitched primitive set difference does not close.")
    reconstructed_discarded_components = {key: 0 for key in component_keys}
    for primitive_id in discarded_ids:
        reconstructed_discarded_components[
            report_key_by_ledger_component[all_map[primitive_id]]
        ] += 1
    declared_discarded_components = _declared_components(
        discarded.get("components"),
        field="discarded_branch_operational_overhead.components",
    )
    discarded_s_alg = _nonnegative_integral(
        discarded.get("S_alg"),
        field="discarded_branch_operational_overhead.S_alg",
    )
    if (
        declared_discarded_components != reconstructed_discarded_components
        or discarded_s_alg != len(discarded_ids)
        or discarded_s_alg != sum(declared_discarded_components.values())
    ):
        raise ValueError("Discarded stitched work does not close.")

    return {
        "S_unique": int(s_alg),
        "historical_mislabeled_field": "S_alg",
        "diagnostic_scope": str(query_work["S_alg_work_scope"]),
        "winning_primitive_count": int(len(winning_ids)),
        "all_executed_primitive_count": int(len(all_ids)),
        "discarded_primitive_count": int(len(discarded_ids)),
    }


def _stitched_query_resource_override(
    query_work: Mapping[str, Any],
) -> tuple[float | None, str, float | None]:
    """Fail closed until authenticated raw-occurrence stitching exists."""

    schema = query_work.get("schema")
    if schema == "jr_snake_stitched_query_accounting_v3":
        return (
            None,
            "unavailable_raw_occurrence_stitching",
            None,
        )
    if (
        schema == "jr_snake_stitched_winning_lineage_query_work_v2"
        and query_work.get("status") == "ok"
    ):
        _validate_exact_stitched_query_work_v2(query_work)
        return None, "withdrawn_unique_primitive_union_is_not_S_alg", None
    if query_work.get("status") != "legacy_proxy_not_exact":
        raise ValueError("Unsupported stitched query-work status")
    legacy = query_work.get("legacy_proxy")
    legacy_value = (
        float(legacy["S_alg"])
        if isinstance(legacy, Mapping) and legacy.get("S_alg") is not None
        else None
    )
    return None, "legacy_proxy_not_exact_missing_primitive_ledgers", legacy_value


def _paper_reference_rows(payload: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    rows: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in payload["corrected_and_snake_rows"]:
        key = (str(row["regime"]), str(row["method"]))
        rows[key] = row
    for row in payload["current_paper_i_comparator_rows"]:
        key = (str(row["regime"]), str(row["method"]))
        rows.setdefault(key, row)
    missing = [
        (regime, method)
        for regime in REGIME_ORDER
        for method in PAPER_METHODS
        if (regime, method) not in rows
    ]
    if missing:
        raise ValueError(f"Missing Paper-I reference rows: {missing}")
    return rows


def _paper_curve(row: Mapping[str, Any]) -> Curve:
    points = tuple(
        (int(point["k"]), _positive(point["error"]))
        for point in row["trajectory_points"]
    )
    marker_k = int(row["k_pl"])
    marker_error = _positive(
        row.get("abs_delta_e")
        or row.get("plot_marker_abs_delta_e")
        or points[min(marker_k, len(points) - 1)][1]
    )
    return Curve(
        role=str(row["method"]),
        points=points,
        marker_k=marker_k,
        marker_error=marker_error,
        source_json=str(row["source_json"]),
        source_sha256=str(row["source_sha256"]),
    )


def _resource_rows_by_regime(
    payload: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    validation = payload.get("validation")
    if not isinstance(validation, Mapping) or not validation or not all(validation.values()):
        raise ValueError("Selected-prefix Qiskit table validation is missing or failed")
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise ValueError("Selected-prefix Qiskit report is missing rows")
    grouped: dict[str, list[dict[str, Any]]] = {regime: [] for regime in REGIME_ORDER}
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise TypeError("Selected-prefix Qiskit row is not an object")
        regime = str(raw.get("regime"))
        if regime not in grouped:
            continue
        row = dict(raw)
        for key in ("k_pl", "N2q", "D2q", "Dc", "S"):
            value = int(row[key])
            if value < 0:
                raise ValueError(f"Negative {key} in {regime} resource row")
            row[key] = value
        error = float(row["abs_delta_e"])
        if not math.isfinite(error) or error < 0.0:
            raise ValueError(f"Invalid error in {regime} resource row")
        row["abs_delta_e"] = error
        grouped[regime].append(row)
    for regime, rows in grouped.items():
        methods = [str(row.get("method")) for row in rows]
        missing = [method for method in RESOURCE_METHODS if method not in methods]
        if missing:
            raise ValueError(f"Missing resource rows for {regime}: {missing}")
        required = [next(row for row in rows if row.get("method") == method) for method in RESOURCE_METHODS]
        optional = [row for row in rows if row.get("method") not in RESOURCE_METHODS]
        grouped[regime] = required + optional
    return grouped


def _supplemental_resource_row(
    *,
    regime: str,
    method: str,
    source_json: Path,
    history_position: int,
    expected_error: float,
    sidecar_json: Path,
    display_k_pl: int | None = None,
    s_override: float | None = None,
    s_source_override: str | None = None,
    source_segments: Sequence[Mapping[str, Any]] = (),
    query_work_sidecar: Path | None = None,
) -> dict[str, Any]:
    from pipelines.reporting.build_paper_i_selected_prefix_qiskit_sidecar import (
        build_sidecar,
    )

    sidecar = build_sidecar(
        result_json=source_json,
        history_position=int(history_position),
        output_json=sidecar_json,
        threshold=float(expected_error),
    )
    checks = {
        "qiskit_validated": sidecar.get("compiled_resource_qiskit_validated") is True,
        "compiled_status_ok": sidecar.get("compiled_circuit_stats_status") == "ok",
        "compile_convention_ok": sidecar.get("compile_convention")
        == "table_i_basis_gate_transpile_v1",
        "history_position_ok": int(sidecar.get("history_position", -1))
        == int(history_position),
        "error_ok": math.isclose(
            float(sidecar.get("primary_error_at_prefix", math.nan)),
            float(expected_error),
            rel_tol=1.0e-11,
            abs_tol=1.0e-15,
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(
            f"Supplemental selected-prefix validation failed for {regime}/{method}: {failed}"
        )
    s_value = None
    s_source = "unavailable"
    for value_key, status_key in (
        ("instrumented_runtime_S", "instrumented_runtime_status"),
        ("mechanism_formula_S", "mechanism_formula_status"),
    ):
        value = sidecar.get(value_key)
        status = str(sidecar.get(status_key) or "").lower()
        if value is not None and status.startswith("ok"):
            s_value = int(round(float(value)))
            s_source = value_key
            break
    if s_override is not None:
        if not math.isfinite(float(s_override)) or float(s_override) < 0.0:
            raise ValueError(f"Invalid S override for {regime}/{method}: {s_override}")
        s_value = int(round(float(s_override)))
        s_source = str(s_source_override or "explicit_stitched_winning_lineage_S_alg")
    row = {
        "regime": regime,
        "method": method,
        "method_display": RESOURCE_METHOD_DISPLAY.get(method, method),
        "role": "supplemental_comparison",
        "k_pl": int(display_k_pl if display_k_pl is not None else history_position),
        "source_history_position": int(history_position),
        "ansatz_depth": int(sidecar["replay"]["replayed_operator_count"]),
        "abs_delta_e": float(expected_error),
        "N2q": int(sidecar["compiled_count_2q_total"]),
        "D2q": int(sidecar["compiled_depth_2q_total"]),
        "Dc": int(sidecar["compiled_depth_total"]),
        "S": s_value,
        "S_source": s_source,
        "source_json": _rel(source_json),
        "source_sha256": _sha256(source_json),
        "qiskit_sidecar": _rel(sidecar_json),
        "qiskit_sidecar_sha256": _sha256(sidecar_json),
        "selected_prefix_validation": checks,
    }
    if source_segments:
        row["source_segments"] = [dict(segment) for segment in source_segments]
    if query_work_sidecar is not None:
        row["query_work_sidecar"] = _rel(query_work_sidecar)
        row["query_work_sidecar_sha256"] = _sha256(query_work_sidecar)
    return row


def _fm_rows_by_regime(
    *,
    campaign_root: Path,
    supplemental_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load FM evidence through a read-only, route-separated campaign adapter."""

    ledger_path = campaign_root / "pareto_ledger.json"
    ledger = _read_json(ledger_path)
    cells = {
        str(row.get("cell_id")): row
        for row in ledger.get("cells", [])
        if isinstance(row, Mapping)
    }
    missing_cells = sorted(set(FM_CELL_ID.values()) - set(cells))
    if missing_cells:
        raise ValueError(f"Missing planned FM cells: {missing_cells}")

    rows: dict[str, dict[str, Any]] = {}
    pending: list[str] = []
    for regime in REGIME_ORDER:
        cell = cells[FM_CELL_ID[regime]]
        status = str(cell.get("status") or "pending")
        if status != "complete":
            pending.append(regime)
            rows[regime] = {
                "status": status,
                "curve": None,
                "resource": {
                    "regime": regime,
                    "method": "formal_manifold_snake",
                    "method_display": "FM-SNAKE",
                    "role": "formal_manifold_transfer_candidate",
                    "status": status,
                    "k_pl": None,
                    "abs_delta_e": None,
                    "N2q": None,
                    "D2q": None,
                    "Dc": None,
                    "S": None,
                    "prefix_semantics": "pending_terminal_endpoint",
                },
                "sources": {
                    "cell_id": str(cell["cell_id"]),
                    "scientific_settings_sha256": str(
                        cell["scientific_settings_sha256"]
                    ),
                },
                "blocker": f"fm_{status}_terminal_sidecars_unavailable",
            }
            continue

        evidence = cell.get("evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError(f"Completed FM cell lacks ingested evidence: {cell['cell_id']}")
        paths = cell.get("paths")
        if not isinstance(paths, Mapping):
            raise ValueError(f"Completed FM cell lacks paths: {cell['cell_id']}")
        result_path = Path(str(paths["result_json"]))
        campaign_qiskit_path = Path(str(paths["qiskit_sidecar"]))
        query_path = Path(str(paths["query_work_sidecar"]))
        for path in (result_path, campaign_qiskit_path, query_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        result_hash = _sha256(result_path)
        if str(evidence.get("result_sha256")) != result_hash:
            raise ValueError(f"FM ledger/result hash mismatch: {cell['cell_id']}")
        result = _read_json(result_path)
        adapt = result.get("adapt_vqe")
        if not isinstance(adapt, Mapping) or adapt.get("success") is not True:
            raise ValueError(f"FM result is not successful: {result_path}")
        if str(adapt.get("adapt_reoptimization_route")) != "formal_manifold_warm_start_v1":
            raise ValueError(f"FM route identity mismatch: {result_path}")
        history = adapt.get("history")
        if not isinstance(history, Sequence) or isinstance(history, (str, bytes)) or not history:
            raise ValueError(f"FM terminal history is unavailable: {result_path}")

        campaign_qiskit = _read_json(campaign_qiskit_path)
        query = _read_json(query_path)
        if campaign_qiskit.get("compiled_resource_qiskit_validated") is not True:
            raise ValueError(f"FM campaign Qiskit sidecar is not validated: {campaign_qiskit_path}")
        if str(campaign_qiskit.get("source_result_sha256")) != result_hash:
            raise ValueError(f"FM campaign Qiskit source hash mismatch: {campaign_qiskit_path}")
        if query.get("science_valid") is not True:
            raise ValueError(f"FM exact query closure failed: {query_path}")
        if query.get("joint_response_selector_invoked") is not False:
            raise ValueError(f"FM/JR route separation failed: {query_path}")
        if str(query.get("source_result_sha256")) != result_hash:
            raise ValueError(f"FM query-work source hash mismatch: {query_path}")

        whitening = evidence.get("whitening_provenance")
        if not isinstance(whitening, Mapping) or not all(whitening.values()):
            raise ValueError(f"FM whitening/frame provenance is incomplete: {cell['cell_id']}")
        provenance_checks = {
            "curvature_whitening": whitening.get("whitening_id")
            == whitening.get("curvature_whitening_id"),
            "curvature_frame": whitening.get("frame_id")
            == whitening.get("curvature_frame_id"),
            "qbroyd_whitening": whitening.get("whitening_id")
            == whitening.get("qbroyd_whitening_id"),
            "qbroyd_logical_range": whitening.get("logical_range_id")
            == whitening.get("qbroyd_logical_range_id"),
        }
        if not all(provenance_checks.values()):
            failed = [name for name, passed in provenance_checks.items() if not passed]
            raise ValueError(
                f"FM coordinate provenance mismatch for {cell['cell_id']}: {failed}"
            )

        curve = _history_curve(result_path, role="manifold")
        report_qiskit_path = supplemental_dir / f"fm-{regime}.json"
        resource = _supplemental_resource_row(
            regime=regime,
            method="formal_manifold_snake",
            source_json=result_path,
            history_position=len(history),
            expected_error=float(adapt["abs_delta_e"]),
            sidecar_json=report_qiskit_path,
        )
        resource.update(
            status="complete",
            role="formal_manifold_transfer_candidate",
            prefix_semantics="terminal_selected_endpoint",
            S=int(query["winning_branch"]["expanded_query_work"]),
            S_source="formal_manifold_query_work_sidecar.winning_branch.expanded_query_work",
            query_work_sidecar=_rel(query_path),
            query_work_sidecar_sha256=_sha256(query_path),
            campaign_qiskit_sidecar=_rel(campaign_qiskit_path),
            campaign_qiskit_sidecar_sha256=_sha256(campaign_qiskit_path),
            coordinate_provenance_checks=provenance_checks,
        )
        rows[regime] = {
            "status": "complete",
            "curve": curve,
            "resource": resource,
            "sources": {
                "cell_id": str(cell["cell_id"]),
                "scientific_settings_sha256": str(cell["scientific_settings_sha256"]),
                "result_json": _rel(result_path),
                "result_sha256": result_hash,
                "query_work_sidecar": _rel(query_path),
                "query_work_sidecar_sha256": _sha256(query_path),
                "campaign_qiskit_sidecar": _rel(campaign_qiskit_path),
                "campaign_qiskit_sidecar_sha256": _sha256(campaign_qiskit_path),
                "report_qiskit_sidecar": _rel(report_qiskit_path),
                "report_qiskit_sidecar_sha256": _sha256(report_qiskit_path),
            },
        }

    return rows, {
        "campaign_root": str(campaign_root.resolve()),
        "ledger_json": str(ledger_path.resolve()),
        "ledger_sha256": _sha256(ledger_path),
        "selected_policy": "inverse_rbfgs_qbroyd_off_v1",
        "pending_regimes": pending,
        "report_qiskit_compile_convention": "table_i_basis_gate_transpile_v1",
        "campaign_qiskit_compile_convention": (
            "fake_marrakesh_local_fake_backend_transpile_opt1_seed7_v1"
        ),
    }


def _comparison_rows(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = {}
    for row in payload["rows"]:
        regime = str(row["regime"]).removesuffix("-u8")
        rows[regime] = row
    if set(rows) != set(REGIME_ORDER):
        raise ValueError(f"Unexpected comparison regimes: {sorted(rows)}")
    return rows


def _live_l10_weak_weak_evidence(output_dir: Path) -> dict[str, Any] | None:
    if not all(path.is_file() for path in JR_L10_WEAK_WEAK_SEGMENTS):
        return None
    supplemental_dir = output_dir / "supplemental_selected_prefix_qiskit"
    supplemental_dir.mkdir(parents=True, exist_ok=True)
    live_source = JR_L10_WEAK_WEAK_SEGMENTS[-1]
    live_bytes = live_source.read_bytes()
    live_payload = json.loads(live_bytes)
    live_adapt = live_payload.get("adapt_vqe")
    if not isinstance(live_adapt, Mapping) or live_adapt.get("history_checkpoint_complete") is not True:
        raise ValueError(f"Live JR-L10 checkpoint is incomplete: {live_source}")
    live_snapshot = supplemental_dir / "jr-l10-live-weak-weak-current-snapshot.json"
    live_snapshot.write_bytes(live_bytes)
    segment_paths = (*JR_L10_WEAK_WEAK_SEGMENTS[:-1], live_snapshot)
    curve = _stitched_history_curve(segment_paths, role="l10_live")
    query_work = _stitched_winning_lineage_query_work(segment_paths)
    query_path = supplemental_dir / "jr-l10-live-weak-weak-query-work.json"
    query_path.write_text(
        json.dumps(query_work, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    exact_s, s_source, legacy_proxy_s = _stitched_query_resource_override(
        query_work
    )
    live_history = _complete_history(live_payload, path=live_source)
    resource = _supplemental_resource_row(
        regime="weak-weak",
        method="jr_snake_whitened_l10_live",
        source_json=live_snapshot,
        history_position=len(live_history),
        display_k_pl=curve.marker_k,
        expected_error=curve.marker_error,
        sidecar_json=supplemental_dir / "jr-l10-live-weak-weak-qiskit.json",
        s_override=exact_s,
        s_source_override=s_source,
        source_segments=curve.source_segments,
        query_work_sidecar=query_path,
    )
    if exact_s is None:
        resource.update(
            S=None,
            S_source=s_source,
            S_status=(
                "legacy_proxy_not_exact"
                if legacy_proxy_s is not None
                else "unavailable_raw_occurrence_stitching"
            ),
            legacy_proxy_S=legacy_proxy_s,
        )
    else:
        resource["S_status"] = "ok_exact_primitive_union"
    queue_status = _read_json(JR_L10_QUEUE_STATUS) if JR_L10_QUEUE_STATUS.is_file() else {}
    return {
        "status": str(queue_status.get("status") or "checkpoint_available"),
        "curve": curve,
        "resource": resource,
        "total_controller_rounds": curve.marker_k,
        "terminal_ansatz_depth": int(resource["ansatz_depth"]),
        "live_source_json": _rel(live_source),
        "live_source_sha256": hashlib.sha256(live_bytes).hexdigest(),
        "report_snapshot_json": _rel(live_snapshot),
        "report_snapshot_sha256": _sha256(live_snapshot),
        "queue_status_json": _rel(JR_L10_QUEUE_STATUS) if JR_L10_QUEUE_STATUS.is_file() else None,
        "queue_status_sha256": _sha256(JR_L10_QUEUE_STATUS) if JR_L10_QUEUE_STATUS.is_file() else None,
        "mixed_source_boundary_after_round": 17,
        "scientific_settings_changed_at_boundary": False,
        "query_work": query_work,
    }


def _curve_payload(curve: Curve) -> dict[str, Any]:
    result = {
        "role": curve.role,
        "points": [{"k": k, "error": error} for k, error in curve.points],
        "point_count": len(curve.points),
        "marker_k": curve.marker_k,
        "marker_error": curve.marker_error,
        "source_json": curve.source_json,
        "source_sha256": curve.source_sha256,
        "source_segments": [dict(segment) for segment in curve.source_segments],
    }
    if curve.source_segments and curve.source_segments[-1].get("marker_only") is True:
        endpoint = dict(curve.source_segments[-1])
        result["status_endpoint_only"] = True
        result["status_endpoint"] = endpoint
        result["trajectory_relation"] = endpoint.get("trajectory_relation")
    if curve.source_segments and curve.source_segments[-1].get("live_snapshot") is True:
        result["live_snapshot"] = True
        result["live_snapshot_status"] = curve.source_segments[-1].get(
            "scheduler_state"
        )
    return result


def _load_jr_chtc_live_snapshot_manifest(
    manifest_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load immutable JR-SNAKE CHTC checkpoints without treating them as final costs."""

    manifest_path = manifest_path.resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != JR_CHTC_LIVE_SNAPSHOT_SCHEMA:
        raise ValueError(
            "Unexpected JR CHTC live-snapshot schema: "
            f"{manifest.get('schema')!r}"
        )
    batch_id = str(manifest.get("batch_id") or "").strip()
    cluster_id = str(manifest.get("cluster_id") or "").strip()
    captured_at = str(manifest.get("captured_at") or "").strip()
    bundle_status = str(manifest.get("status") or "").strip()
    if not all((batch_id, cluster_id, captured_at, bundle_status)):
        raise ValueError("JR CHTC live manifest lacks batch/cluster/time/status")
    expected_policy = {
        "route": "route_a",
        "batch_search_pool_size": 10,
        "batch_size_cap": 2,
        "inner_optimizer": "POWELL",
        "powell_maxfev": 200,
        "joint_linear_solve": "supported_metric_whitened_eigh_v1",
        "trust_region_update": "displacement_calibrated_unbounded_v2",
        "structural_rollback_enabled": False,
    }
    if manifest.get("policy") != expected_policy:
        raise ValueError("JR CHTC live policy mismatch")
    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes)):
        raise ValueError("JR CHTC live entries must be a JSON array")

    rows: dict[str, dict[str, Any]] = {}
    seen_procs: set[int] = set()
    provenance_entries: list[dict[str, Any]] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            raise TypeError("JR CHTC live entry is not an object")
        proc_raw = raw_entry.get("proc_id")
        if isinstance(proc_raw, bool):
            raise ValueError(f"Invalid JR CHTC proc id: {proc_raw!r}")
        proc_id = int(proc_raw)
        if proc_id in seen_procs or proc_id < 0:
            raise ValueError(f"Duplicate or invalid JR CHTC proc id: {proc_id}")
        seen_procs.add(proc_id)
        regime = _normalize_fm_live_regime(raw_entry.get("regime"))
        expected_proc = REGIME_ORDER.index(regime)
        if proc_id != expected_proc:
            raise ValueError(
                f"JR CHTC proc/regime mismatch: proc={proc_id}, regime={regime}"
            )
        if regime in rows:
            raise ValueError(f"Duplicate JR CHTC regime: {regime}")
        row_id = str(raw_entry.get("row_id") or "").strip()
        if not row_id or not row_id.startswith(batch_id + "__"):
            raise ValueError(f"JR CHTC row identity mismatch: {row_id!r}")

        scheduler_state = str(raw_entry.get("scheduler_state") or "").strip()
        if scheduler_state not in JR_CHTC_LIVE_STATUSES:
            raise ValueError(f"Unexpected JR CHTC scheduler state: {scheduler_state}")
        source_kind = str(raw_entry.get("source_kind") or "").strip()
        expected_source_kind = {
            "running_snapshot": "live_current_json",
            "held_snapshot": "live_current_json",
            "stopped_snapshot": "live_current_json",
            "recovery_queued_snapshot": "live_current_json",
            "completed_snapshot_pending_qiskit": "completed_result_json",
        }[scheduler_state]
        if source_kind != expected_source_kind:
            raise ValueError(
                f"JR CHTC status/source mismatch: {scheduler_state}/{source_kind}"
            )
        terminal = raw_entry.get("terminal")
        expected_terminal = scheduler_state == "completed_snapshot_pending_qiskit"
        if terminal is not expected_terminal:
            raise ValueError(f"JR CHTC terminal-state mismatch: {regime}")
        if raw_entry.get("structural_rollback_enabled") is not False:
            raise ValueError(
                f"JR CHTC entry does not prove structural rollback disabled: {regime}"
            )

        raw_snapshot_path = Path(str(raw_entry.get("snapshot_json") or ""))
        full_snapshot_path = (
            raw_snapshot_path
            if raw_snapshot_path.is_absolute()
            else manifest_path.parent / raw_snapshot_path
        ).resolve()
        if not full_snapshot_path.is_file():
            raise FileNotFoundError(full_snapshot_path)
        full_snapshot_hash = _sha256(full_snapshot_path)
        if full_snapshot_hash != str(raw_entry.get("source_sha256") or "").lower():
            raise ValueError(f"JR CHTC snapshot hash mismatch: {full_snapshot_path}")
        raw_projection_path = raw_entry.get("report_projection_json")
        if raw_projection_path:
            projection_path = Path(str(raw_projection_path))
            snapshot_path = (
                projection_path
                if projection_path.is_absolute()
                else manifest_path.parent / projection_path
            ).resolve()
            if not snapshot_path.is_file():
                raise FileNotFoundError(snapshot_path)
            snapshot_hash = _sha256(snapshot_path)
            if snapshot_hash != str(
                raw_entry.get("report_projection_sha256") or ""
            ).lower():
                raise ValueError(
                    f"JR CHTC report-projection hash mismatch: {snapshot_path}"
                )
        else:
            snapshot_path = full_snapshot_path
            snapshot_hash = full_snapshot_hash
        payload = _read_json(snapshot_path)
        _validate_fm_live_no_structural_rollback(payload, path=snapshot_path)
        settings = payload.get("settings")
        if not isinstance(settings, Mapping):
            scientific_settings = payload.get("scientific_settings")
            settings = (
                scientific_settings.get("run_kwargs")
                if isinstance(scientific_settings, Mapping)
                else None
            )
        if not isinstance(settings, Mapping) or settings.get("static_route_id") != "route_a":
            raise ValueError(f"JR CHTC route mismatch: {snapshot_path}")
        adapt = payload.get("adapt_vqe")
        if not isinstance(adapt, Mapping):
            raise ValueError(f"JR CHTC snapshot lacks adapt_vqe: {snapshot_path}")
        trust_state = settings.get("route_a_trust_region_state")
        if not isinstance(trust_state, Mapping):
            trust_state = adapt.get("route_a_trust_region_state")
        if isinstance(trust_state, Mapping):
            last_update = trust_state.get("last_update")
            if isinstance(last_update, Mapping) and last_update.get("policy") != (
                expected_policy["trust_region_update"]
            ):
                raise ValueError(f"JR CHTC trust-policy mismatch: {snapshot_path}")
        if adapt.get("history_checkpoint_complete") is not True:
            raise ValueError(f"JR CHTC history checkpoint is incomplete: {snapshot_path}")
        if expected_terminal and not str(adapt.get("stop_reason") or "").strip():
            raise ValueError(f"JR CHTC completed snapshot lacks a stop reason: {regime}")

        base_curve = _history_curve(snapshot_path, role="jr_chtc_live")
        controller_round = int(raw_entry["controller_round"])
        ansatz_depth = int(raw_entry["ansatz_depth"])
        energy = float(raw_entry["energy"])
        exact = float(raw_entry["exact_same_cutoff_energy"])
        error = _positive(raw_entry["abs_delta_e"])
        if controller_round != base_curve.marker_k or int(
            adapt.get("history_count", -1)
        ) != controller_round:
            raise ValueError(f"JR CHTC controller-round mismatch: {regime}")
        if int(adapt.get("ansatz_depth", -1)) != ansatz_depth:
            raise ValueError(f"JR CHTC ansatz-depth mismatch: {regime}")
        numeric_pairs = (
            (energy, float(adapt["energy"]), "energy"),
            (exact, float(adapt["exact_gs_energy"]), "exact same-cutoff energy"),
            (error, _positive(adapt["abs_delta_e"]), "absolute error"),
            (error, abs(energy - exact), "same-cutoff error identity"),
        )
        for expected, actual, label in numeric_pairs:
            if not math.isclose(expected, actual, rel_tol=1.0e-12, abs_tol=1.0e-12):
                raise ValueError(f"JR CHTC {label} mismatch: {regime}")

        segment = {
            "live_snapshot": True,
            "scheduler_state": scheduler_state,
            "terminal": expected_terminal,
            "controller_round": controller_round,
            "ansatz_depth": ansatz_depth,
            "cluster_id": cluster_id,
            "proc_id": proc_id,
        }
        curve = Curve(
            role="jr_chtc_live",
            points=base_curve.points,
            marker_k=base_curve.marker_k,
            marker_error=base_curve.marker_error,
            source_json=base_curve.source_json,
            source_sha256=base_curve.source_sha256,
            source_segments=(segment,),
        )
        resource = {
            "regime": regime,
            "method": "jr_snake_chtc_live",
            "method_display": RESOURCE_METHOD_DISPLAY["jr_snake_chtc_live"],
            "role": "jr_chtc_live",
            "status": scheduler_state,
            "k_pl": controller_round,
            "controller_rounds": controller_round,
            "ansatz_depth": ansatz_depth,
            "energy": energy,
            "exact_same_cutoff_energy": exact,
            "abs_delta_e": error,
            "N2q": None,
            "D2q": None,
            "Dc": None,
            "S": None,
            "resource_status": "pending_snapshot_sidecars",
            "prefix_semantics": (
                "terminal_algorithm_endpoint_pending_qiskit"
                if expected_terminal
                else "immutable_stopped_best_branch_checkpoint_pending_sidecars"
                if scheduler_state == "stopped_snapshot"
                else "immutable_nonterminal_best_branch_checkpoint"
            ),
            "source_json": _rel(snapshot_path),
            "source_sha256": snapshot_hash,
            "source_kind": source_kind,
            "terminal": expected_terminal,
        }
        if scheduler_state == "stopped_snapshot":
            resource["status_label"] = "stopped snapshot"
        qiskit_sidecar_path: Path | None = None
        query_sidecar_path: Path | None = None
        raw_qiskit_sidecar = raw_entry.get("qiskit_sidecar_json")
        raw_query_sidecar = raw_entry.get("query_work_sidecar_json")
        if bool(raw_qiskit_sidecar) != bool(raw_query_sidecar):
            raise ValueError(
                f"JR CHTC snapshot sidecars must be supplied together: {regime}"
            )
        if raw_qiskit_sidecar:
            supports_fixed_prefix_sidecars = (
                expected_terminal or scheduler_state == "stopped_snapshot"
            )
            if not supports_fixed_prefix_sidecars:
                raise ValueError(
                    "JR CHTC sidecars require a terminal or explicitly stopped "
                    f"fixed-prefix snapshot: {regime}"
                )

            def _resolved_sidecar(raw_path: Any) -> Path:
                path = Path(str(raw_path))
                return (
                    path if path.is_absolute() else manifest_path.parent / path
                ).resolve()

            qiskit_sidecar_path = _resolved_sidecar(raw_qiskit_sidecar)
            query_sidecar_path = _resolved_sidecar(raw_query_sidecar)
            for sidecar_path, hash_key, label in (
                (qiskit_sidecar_path, "qiskit_sidecar_sha256", "Qiskit"),
                (query_sidecar_path, "query_work_sidecar_sha256", "query-work"),
            ):
                if not sidecar_path.is_file():
                    raise FileNotFoundError(sidecar_path)
                if _sha256(sidecar_path) != str(raw_entry.get(hash_key) or "").lower():
                    raise ValueError(
                        f"JR CHTC {label} sidecar hash mismatch: {regime}"
                    )

            qiskit = _read_json(qiskit_sidecar_path)
            qiskit_checks = {
                "validated": qiskit.get("compiled_resource_qiskit_validated") is True,
                "status": qiskit.get("compiled_circuit_stats_status") == "ok",
                "convention": qiskit.get("compile_convention")
                == "table_i_basis_gate_transpile_v1",
                "error": math.isclose(
                    float(qiskit.get("primary_error_at_prefix", math.nan)),
                    error,
                    rel_tol=1.0e-11,
                    abs_tol=1.0e-15,
                ),
                "depth": int(qiskit.get("replay", {}).get("replayed_operator_count", -1))
                == ansatz_depth,
            }
            if not all(qiskit_checks.values()):
                failed = [key for key, passed in qiskit_checks.items() if not passed]
                raise ValueError(
                    f"JR CHTC Qiskit sidecar validation failed for {regime}: {failed}"
                )

            query = _read_json(query_sidecar_path)
            query_schema = query.get("schema")
            legacy_stitched_proxy = False
            query_s_unavailable = False
            legacy_query_total = None
            if query_schema == "jr_snake_stitched_query_accounting_v3":
                query_ok = (
                    query.get("status") == "unavailable_raw_occurrence_stitching"
                    and query.get("S_alg") is None
                )
                query_total = None
                query_scope = query.get("S_alg_work_scope")
                query_s_unavailable = True
            elif query_schema == "jr_snake_stitched_winning_lineage_query_work_v2":
                if query.get("status") == "legacy_proxy_not_exact":
                    query_ok = query.get("primitive_union_validated") is False
                    query_total = None
                    query_scope = "unresolved_without_full_estimator_ledgers"
                    legacy = query.get("legacy_proxy")
                    legacy_query_total = (
                        legacy.get("S_alg") if isinstance(legacy, Mapping) else None
                    )
                    legacy_stitched_proxy = True
                else:
                    _validate_exact_stitched_query_work_v2(query)
                    query_ok = True
                    query_total = None
                    query_scope = (
                        "withdrawn_unique_primitive_union_is_not_S_alg"
                    )
                    query_s_unavailable = True
            elif query_schema == "jr_snake_stitched_winning_lineage_query_work_v1":
                query_ok = (
                    query.get("status") == "ok"
                    and query.get("S_alg_work_scope")
                    == "winning_lineage_stitched_segments"
                    and query.get("discarded_branch_search_work_included") is False
                )
                query_total = None
                query_scope = "legacy_stitched_segment_sum_not_exact"
                legacy_query_total = query.get("S_alg")
                legacy_stitched_proxy = True
            else:
                query_ok = (
                    query.get("status") == "complete"
                    and query.get("query_work_status") == "ok"
                    and query.get("query_work_scope") == "winner_lineage_terminal"
                )
                query_total = query.get("query_work_total")
                query_scope = query.get("query_work_scope")
            if (
                not query_ok
                or (
                    not legacy_stitched_proxy
                    and not query_s_unavailable
                    and (query_total is None or float(query_total) < 0.0)
                )
                or (
                    legacy_query_total is not None
                    and float(legacy_query_total) < 0.0
                )
            ):
                raise ValueError(
                    f"JR CHTC winning-lineage query sidecar validation failed: {regime}"
                )
            resource.update(
                N2q=int(qiskit["compiled_count_2q_total"]),
                D2q=int(qiskit["compiled_depth_2q_total"]),
                Dc=int(qiskit["compiled_depth_total"]),
                S=(
                    None
                    if legacy_stitched_proxy or query_s_unavailable
                    else int(round(float(query_total)))
                ),
                S_source=(
                    "legacy_proxy_not_exact_missing_primitive_ledgers"
                    if legacy_stitched_proxy
                    else (
                        "unavailable_raw_occurrence_stitching"
                        if query_s_unavailable
                        else "validated_winning_lineage_query_sidecar"
                    )
                ),
                S_scope=str(query_scope),
                S_status=(
                    "legacy_proxy_not_exact"
                    if legacy_stitched_proxy
                    else (
                        "unavailable_raw_occurrence_stitching"
                        if query_s_unavailable
                        else "ok_exact"
                    )
                ),
                legacy_proxy_S=(
                    None
                    if legacy_query_total is None
                    else int(round(float(legacy_query_total)))
                ),
                resource_status=(
                    "validated_qiskit_query_work_legacy_proxy_only"
                    if legacy_stitched_proxy
                    else (
                        "validated_qiskit_but_query_work_unavailable"
                        if query_s_unavailable
                        else "validated_snapshot_sidecars"
                    )
                ),
                prefix_semantics=(
                    "terminal_algorithm_endpoint_with_validated_sidecars"
                    if expected_terminal
                    else "immutable_stopped_checkpoint_with_validated_snapshot_sidecars"
                ),
                qiskit_sidecar=_rel(qiskit_sidecar_path),
                qiskit_sidecar_sha256=_sha256(qiskit_sidecar_path),
                query_work_sidecar=_rel(query_sidecar_path),
                query_work_sidecar_sha256=_sha256(query_sidecar_path),
            )
        provenance = {
            **segment,
            "row_id": row_id,
            "regime": regime,
            "source_kind": source_kind,
            "snapshot_json": _rel(snapshot_path),
            "snapshot_sha256": snapshot_hash,
            "full_snapshot_json": _rel(full_snapshot_path),
            "full_snapshot_sha256": full_snapshot_hash,
            "energy": energy,
            "exact_same_cutoff_energy": exact,
            "abs_delta_e": error,
            "structural_rollback_enabled": False,
            "resource_fields": "pending_until_qiskit_and_query_sidecars",
        }
        if qiskit_sidecar_path is not None and query_sidecar_path is not None:
            provenance.update(
                resource_fields=(
                    "validated_qiskit_query_work_legacy_proxy_only"
                    if resource.get("S_status") == "legacy_proxy_not_exact"
                    else "validated_qiskit_and_winning_lineage_query_sidecars"
                ),
                qiskit_sidecar=_rel(qiskit_sidecar_path),
                qiskit_sidecar_sha256=_sha256(qiskit_sidecar_path),
                query_work_sidecar=_rel(query_sidecar_path),
                query_work_sidecar_sha256=_sha256(query_sidecar_path),
            )
        rows[regime] = {
            "curve": curve,
            "resource": resource,
            "provenance": provenance,
        }
        provenance_entries.append(provenance)

    return rows, {
        "schema": JR_CHTC_LIVE_SNAPSHOT_SCHEMA,
        "manifest_json": _rel(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "batch_id": batch_id,
        "cluster_id": cluster_id,
        "captured_at": captured_at,
        "status": bundle_status,
        "policy": expected_policy,
        "entry_count": len(provenance_entries),
        "entries": provenance_entries,
        "terminal_regimes": [
            row["regime"] for row in provenance_entries if row["terminal"]
        ],
        "stopped_regimes": [
            row["regime"]
            for row in provenance_entries
            if row["scheduler_state"] == "stopped_snapshot"
        ],
        "running_regimes": [
            row["regime"]
            for row in provenance_entries
            if not row["terminal"]
            and row["scheduler_state"] != "stopped_snapshot"
        ],
        "resource_fields": "per_entry_pending_or_validated_sidecars",
        "structural_rollback_enabled": False,
    }


def _route4_source_records() -> dict[str, dict[str, Any]]:
    matrix = ROUTE4_MATRIX_ROOT / "route4_exact_hessian_schur"
    return {
        "weak-weak": {
            "status": "complete",
            "source_json": ROUTE4_WEAK_WEAK_ROOT / "full/json/result.json",
            "ledger_json": ROUTE4_WEAK_WEAK_ROOT
            / "full/json/estimator_call_ledger.json",
            "command_json": ROUTE4_WEAK_WEAK_ROOT / "full/command.json",
            "validation_json": None,
        },
        "intermediate-weak": {
            "status": "complete",
            "source_json": matrix / "intermediate_weak/full/json/result.json",
            "ledger_json": matrix
            / "intermediate_weak/full/json/estimator_call_ledger.json",
            "command_json": matrix / "intermediate_weak/full/command.json",
            "normalized_manifest_json": matrix
            / "intermediate_weak/full/normalized_manifest.json",
            "validation_json": matrix / "intermediate_weak/full/validation.json",
        },
        "strong-weak": {
            "status": "complete",
            "source_json": matrix / "strong_weak/full/json/result.json",
            "ledger_json": matrix
            / "strong_weak/full/json/estimator_call_ledger.json",
            "command_json": matrix / "strong_weak/full/command.json",
            "normalized_manifest_json": matrix
            / "strong_weak/full/normalized_manifest.json",
            "validation_json": matrix / "strong_weak/full/validation.json",
        },
        "weak-strong": {
            "status": "failed_partial_round21",
            "source_json": matrix / "weak_strong/full/json/current.json",
            "ledger_json": matrix
            / "weak_strong/full/json/estimator_call_ledger.json",
            "command_json": matrix / "weak_strong/full/command.json",
            "normalized_manifest_json": matrix
            / "weak_strong/full/normalized_manifest.json",
            "validation_json": None,
        },
        "intermediate-strong": {
            "status": "failed_partial_round21",
            "source_json": matrix / "intermediate_strong/full/json/current.json",
            "ledger_json": matrix
            / "intermediate_strong/full/json/estimator_call_ledger.json",
            "command_json": matrix / "intermediate_strong/full/command.json",
            "normalized_manifest_json": matrix
            / "intermediate_strong/full/normalized_manifest.json",
            "validation_json": None,
        },
        "strong-strong": {
            "status": "not_run",
            "source_json": None,
            "ledger_json": None,
            "command_json": None,
            "normalized_manifest_json": matrix
            / "strong_strong/full/normalized_manifest.json",
            "validation_json": None,
        },
    }


def _route4_accounting_summary(ledger_path: Path) -> dict[str, Any]:
    payload = _read_json(ledger_path)
    accounting = payload.get("accounting")
    if not isinstance(accounting, Mapping):
        raise ValueError(f"Route-4 ledger lacks accounting metadata: {ledger_path}")
    component_keys = ("N_H_outer", "N_H_refit", "N_grad", "N_metric", "S_alg")

    def compact(raw: Any) -> dict[str, int | None] | None:
        if not isinstance(raw, Mapping):
            return None
        return {
            key: None if raw.get(key) is None else int(raw[key])
            for key in component_keys
        }

    return {
        "schema": "paper_i_route4_compact_estimator_accounting_v1",
        "definition": accounting.get("definition"),
        "status": accounting.get("status"),
        "complete": accounting.get("complete") is True,
        "exact_blockers": list(accounting.get("exact_blockers") or []),
        "adapt_success": payload.get("adapt_success") is True,
        "adapt_error": payload.get("adapt_error"),
        "winning_lineage": compact(accounting.get("winning_lineage")),
        "discarded_branch_only": compact(
            accounting.get("discarded_branch_only_by_unique_set_difference")
        ),
        "all_branch_search_work": compact(accounting.get("all_branch_search_work")),
        "source_ledger_json": _rel(ledger_path),
        "source_ledger_sha256": _sha256(ledger_path),
    }


def _route4_rows_by_regime(
    output_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    supplemental_dir = output_dir / "supplemental_selected_prefix_qiskit"
    supplemental_dir.mkdir(parents=True, exist_ok=True)
    records = _route4_source_records()
    rows: dict[str, dict[str, Any]] = {}
    source_records: dict[str, dict[str, Any]] = {}
    for regime in REGIME_ORDER:
        record = records[regime]
        status = str(record["status"])
        artifact_refs: dict[str, Any] = {"status": status}
        for key in (
            "source_json",
            "ledger_json",
            "command_json",
            "normalized_manifest_json",
            "validation_json",
        ):
            path = record.get(key)
            if path is None:
                artifact_refs[key] = None
                artifact_refs[f"{key.removesuffix('_json')}_sha256"] = None
                continue
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(path)
            artifact_refs[key] = _rel(path)
            artifact_refs[f"{key.removesuffix('_json')}_sha256"] = _sha256(path)

        source_path = record.get("source_json")
        if source_path is None:
            resource = {
                "regime": regime,
                "method": "paper_i_route4_snake",
                "method_display": RESOURCE_METHOD_DISPLAY["paper_i_route4_snake"],
                "role": "paper_i_route4",
                "status": status,
                "exact_blockers": ["scientific_run_not_launched"],
            }
            rows[regime] = {"status": status, "curve": None, "resource": resource}
            source_records[regime] = artifact_refs
            continue

        source_path = Path(source_path)
        curve = _history_curve(source_path, role="paper_i_route4")
        ledger_path = Path(record["ledger_json"])
        accounting = _route4_accounting_summary(ledger_path)
        accounting_path = supplemental_dir / f"paper-i-route4-{regime}-accounting.json"
        accounting_path.write_text(
            json.dumps(accounting, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        winning = accounting.get("winning_lineage")
        winning_s = (
            None
            if not isinstance(winning, Mapping) or winning.get("S_alg") is None
            else float(winning["S_alg"])
        )
        resource = _supplemental_resource_row(
            regime=regime,
            method="paper_i_route4_snake",
            source_json=source_path,
            history_position=int(curve.marker_k),
            display_k_pl=int(curve.marker_k),
            expected_error=float(curve.marker_error),
            sidecar_json=supplemental_dir / f"paper-i-route4-{regime}-qiskit.json",
            s_override=winning_s,
            s_source_override=(
                "state_keyed_estimator_ledger.winning_lineage.S_alg"
                if winning_s is not None
                else None
            ),
        )
        resource.update(
            {
                "role": "paper_i_route4",
                "status": status,
                "controller_rounds": int(curve.marker_k),
                "accounting_status": accounting.get("status"),
                "accounting_complete": accounting.get("complete") is True,
                "accounting_summary_json": _rel(accounting_path),
                "accounting_summary_sha256": _sha256(accounting_path),
                "discarded_branch_S_alg": (
                    (accounting.get("discarded_branch_only") or {}).get("S_alg")
                ),
                "all_branch_search_S_alg": (
                    (accounting.get("all_branch_search_work") or {}).get("S_alg")
                ),
                "exact_blockers": list(accounting.get("exact_blockers") or []),
            }
        )
        if winning_s is None:
            # The two failed rows never finalized their state-keyed ledgers.  A
            # formula estimate from the Qiskit replay is not a substitute.
            resource["S"] = None
            resource["S_source"] = "unresolved_failed_before_ledger_finalization"
        artifact_refs.update(
            {
                "accounting_summary_json": _rel(accounting_path),
                "accounting_summary_sha256": _sha256(accounting_path),
                "qiskit_sidecar": resource["qiskit_sidecar"],
                "qiskit_sidecar_sha256": resource["qiskit_sidecar_sha256"],
            }
        )
        rows[regime] = {
            "status": status,
            "curve": curve,
            "resource": resource,
            "accounting": accounting,
        }
        source_records[regime] = artifact_refs

    if not ROUTE4_SOURCE_LOCK_DIFF.is_file():
        raise FileNotFoundError(ROUTE4_SOURCE_LOCK_DIFF)
    campaign = {
        "schema": "paper_i_route4_six_regime_report_adapter_v1",
        "route": "route_4_whitened_adaptive",
        "whitening": "supported_metric_whitened_eigh_v1",
        "adaptive_trust": "displacement_calibrated_unbounded_v2",
        "phase0": "off",
        "batching": "off",
        "prune_policy": "recoverability_ladder_v1",
        "schur_nomination_route": "hessian_coupling_v1",
        "controller_horizon_preserved": 30,
        "completed_regimes": [
            regime for regime in REGIME_ORDER if rows[regime]["status"] == "complete"
        ],
        "partial_regimes": [
            regime
            for regime in REGIME_ORDER
            if rows[regime]["status"] == "failed_partial_round21"
        ],
        "not_run_regimes": [
            regime for regime in REGIME_ORDER if rows[regime]["status"] == "not_run"
        ],
        "source_lock_and_settings_diff_json": _rel(ROUTE4_SOURCE_LOCK_DIFF),
        "source_lock_and_settings_diff_sha256": _sha256(ROUTE4_SOURCE_LOCK_DIFF),
        "source_records": source_records,
    }
    return rows, campaign


def _load_paper_i_route4_live_snapshot_manifest(
    manifest_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load immutable SR-SNAKE recovery evidence as additive endpoints.

    The legacy v1 schema remains strictly nonterminal.  The v2 schema may mix
    immutable running checkpoints, stopped checkpoints with separately validated
    fixed-prefix sidecars, and terminal recovery endpoints.  Hash-validated
    stopped checkpoints retain their full serialized history.  Terminal entries
    may additionally provide a compact, hash-linked controller trajectory while
    retaining the separately validated final-refit endpoint.
    """

    manifest_path = manifest_path.resolve()
    manifest = _read_json(manifest_path)
    schema = str(manifest.get("schema") or "")
    if schema not in {
        PAPER_I_ROUTE4_LIVE_SNAPSHOT_SCHEMA,
        PAPER_I_SR_RECOVERY_SNAPSHOT_SCHEMA,
    }:
        raise ValueError(
            "Unexpected Paper-I SR-SNAKE recovery-snapshot schema: "
            f"{schema!r}"
        )
    mixed_recovery = schema == PAPER_I_SR_RECOVERY_SNAPSHOT_SCHEMA
    captured_at = str(manifest.get("captured_at_utc") or "").strip()
    if not captured_at:
        raise ValueError("Paper-I SR-SNAKE recovery snapshot lacks capture time")
    if str(manifest.get("route") or "") != "route_4_whitened_adaptive":
        raise ValueError("Paper-I SR-SNAKE recovery snapshot route mismatch")
    if mixed_recovery:
        if manifest.get("stable_family_id") != "singleton_response_snake":
            raise ValueError("Paper-I SR-SNAKE stable-family id mismatch")
        if manifest.get("evidence_status") != "mixed_terminal_and_nonterminal_recovery":
            raise ValueError("Paper-I SR-SNAKE mixed recovery status mismatch")
        if manifest.get("terminal") is not None:
            raise ValueError("Paper-I SR-SNAKE mixed recovery terminal field must be null")
    else:
        if manifest.get("evidence_status") != PAPER_I_ROUTE4_LIVE_STATUS:
            raise ValueError("Paper-I Route-4 live snapshot status mismatch")
        if manifest.get("terminal") is not False:
            raise ValueError("Paper-I Route-4 live snapshot must be nonterminal")
    expected_policy = {
        "coordinate_solve": "supported_metric_whitened_eigh_v1",
        "trust_region_update": "displacement_calibrated_unbounded_v2",
        "phase0": "off",
        "phase2_batching": "off",
        "phase3_batching": "off",
        "prune_policy": "recoverability_ladder_v1",
    }
    if manifest.get("policy") != expected_policy:
        raise ValueError("Paper-I Route-4 live snapshot policy mismatch")

    def manifest_path_for(raw: Any) -> Path:
        path = Path(str(raw or ""))
        if not str(path):
            raise ValueError("Paper-I Route-4 live snapshot has an empty artifact path")
        return (path if path.is_absolute() else manifest_path.parent / path).resolve()

    def repo_path_for(raw: Any) -> Path:
        path = Path(str(raw or ""))
        if not str(path):
            raise ValueError("Paper-I Route-4 live snapshot has an empty source path")
        return (path if path.is_absolute() else REPO_ROOT / path).resolve()

    def require_hash(path: Path, expected: Any, *, label: str) -> str:
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != str(expected or "").lower():
            raise ValueError(f"Paper-I Route-4 {label} hash mismatch: {path}")
        return actual

    def validate_route_contract(
        raw_entry: Mapping[str, Any], *, regime: str
    ) -> tuple[Path, str, Path, str, dict[str, bool], dict[str, bool]]:
        command_path = repo_path_for(raw_entry.get("command_json"))
        command_hash = require_hash(
            command_path,
            raw_entry.get("command_sha256"),
            label=f"{regime} command",
        )
        normalized_manifest_path = repo_path_for(
            raw_entry.get("normalized_manifest_json")
        )
        normalized_manifest_hash = require_hash(
            normalized_manifest_path,
            raw_entry.get("normalized_manifest_sha256"),
            label=f"{regime} normalized manifest",
        )

        command_payload = _read_json(command_path)
        raw_argv = command_payload.get("argv")
        if not isinstance(raw_argv, Sequence) or isinstance(raw_argv, (str, bytes)):
            raise ValueError(f"Paper-I Route-4 command lacks argv: {regime}")
        argv = [str(value) for value in raw_argv]

        def argv_value(flag: str) -> str | None:
            try:
                return argv[argv.index(flag) + 1]
            except (ValueError, IndexError):
                return None

        command_checks = {
            "phase0_disabled": "--phase0-no-pilot" in argv,
            "phase2_batching_disabled": "--phase2-no-batching" in argv,
            "phase3_batching_disabled": "--phase3-no-batching" in argv,
            "hard_symmetry_guard": argv_value(
                "--phase3-runtime-split-child-set-symmetry-policy"
            )
            == "hard_guard",
            "padding_projection": argv_value(
                "--phase3-runtime-split-child-padding-policy"
            )
            == "exact_projected_grouped_v1",
            "whitened_coordinate_solve": argv_value(
                "--historical-singleton-coordinate-solve-policy"
            )
            == "supported_metric_whitened_eigh_v1",
            "adaptive_trust": argv_value(
                "--historical-singleton-trust-region-update-policy"
            )
            == "displacement_calibrated_unbounded_v2",
        }
        if not all(command_checks.values()):
            failed = [name for name, passed in command_checks.items() if not passed]
            raise ValueError(
                f"Paper-I Route-4 command policy mismatch for {regime}: {failed}"
            )

        normalized_payload = _read_json(normalized_manifest_path)
        scientific_contract = normalized_payload.get("scientific_contract")
        normalized_checks = {
            "phase0_disabled": isinstance(scientific_contract, Mapping)
            and scientific_contract.get("phase0_enabled") is False,
            "phase2_batching_disabled": isinstance(scientific_contract, Mapping)
            and scientific_contract.get("phase2_batching_enabled") is False,
            "phase3_batching_disabled": isinstance(scientific_contract, Mapping)
            and scientific_contract.get("phase3_batching_enabled") is False,
            "hard_symmetry_guard": isinstance(scientific_contract, Mapping)
            and scientific_contract.get("symmetry_policy") == "hard_guard",
            "padding_projection": isinstance(scientific_contract, Mapping)
            and scientific_contract.get("padding_policy")
            == "exact_projected_grouped_v1",
            "singleton_subset": isinstance(scientific_contract, Mapping)
            and scientific_contract.get("singleton_subset_size") == 1,
            "whitened_coordinate_solve": isinstance(scientific_contract, Mapping)
            and scientific_contract.get("coordinate_solve_policy")
            == "supported_metric_whitened_eigh_v1",
            "adaptive_trust": isinstance(scientific_contract, Mapping)
            and scientific_contract.get("trust_region_update_policy")
            == "displacement_calibrated_unbounded_v2",
        }
        if not all(normalized_checks.values()):
            failed = [name for name, passed in normalized_checks.items() if not passed]
            raise ValueError(
                f"Paper-I Route-4 normalized policy mismatch for {regime}: {failed}"
            )
        return (
            command_path,
            command_hash,
            normalized_manifest_path,
            normalized_manifest_hash,
            command_checks,
            normalized_checks,
        )

    queue_path = manifest_path_for(manifest.get("queue_state_snapshot"))
    queue_hash = require_hash(
        queue_path,
        manifest.get("queue_state_snapshot_sha256"),
        label="queue-state snapshot",
    )
    queue_source_hash = str(
        manifest.get("queue_state_source_sha256_at_capture") or queue_hash
    ).lower()
    if queue_source_hash != queue_hash:
        raise ValueError("Paper-I Route-4 queue-state capture hash mismatch")
    source_lock_path = repo_path_for(
        manifest.get("source_lock_and_settings_diff_json")
    )
    source_lock_hash = require_hash(
        source_lock_path,
        manifest.get("source_lock_and_settings_diff_sha256"),
        label="source-lock",
    )

    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes)):
        raise ValueError("Paper-I Route-4 live snapshot entries must be a JSON array")
    rows: dict[str, dict[str, Any]] = {}
    provenance_entries: list[dict[str, Any]] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            raise TypeError("Paper-I Route-4 live snapshot entry is not an object")
        regime = _normalize_fm_live_regime(raw_entry.get("regime"))
        if regime in rows:
            raise ValueError(f"Duplicate Paper-I Route-4 live snapshot regime: {regime}")
        entry_status = str(raw_entry.get("status") or "")
        if entry_status == PAPER_I_SR_TERMINAL_STATUS:
            if not mixed_recovery:
                raise ValueError(
                    f"Legacy Paper-I Route-4 bundle cannot contain terminal recovery: {regime}"
                )
            if raw_entry.get("terminal") is not True:
                raise ValueError(f"Paper-I SR-SNAKE terminal flag mismatch: {regime}")
            if raw_entry.get("source_kind") != "validated_recovery_endpoint":
                raise ValueError(f"Paper-I SR-SNAKE terminal source-kind mismatch: {regime}")

            (
                command_path,
                command_hash,
                normalized_manifest_path,
                normalized_manifest_hash,
                command_checks,
                normalized_checks,
            ) = validate_route_contract(raw_entry, regime=regime)
            validation_path = repo_path_for(raw_entry.get("validation_json"))
            validation_hash = require_hash(
                validation_path,
                raw_entry.get("validation_sha256"),
                label=f"{regime} validation",
            )
            qiskit_path = repo_path_for(raw_entry.get("qiskit_json"))
            qiskit_hash = require_hash(
                qiskit_path,
                raw_entry.get("qiskit_sha256"),
                label=f"{regime} Qiskit sidecar",
            )
            result_path = repo_path_for(raw_entry.get("result_json"))
            result_hash = require_hash(
                result_path,
                raw_entry.get("result_sha256"),
                label=f"{regime} terminal result",
            )

            validation = _read_json(validation_path)
            if validation.get("schema") != "paper_i_hh_route4_round45_validation_v1":
                raise ValueError(f"Paper-I SR-SNAKE validation schema mismatch: {regime}")
            if validation.get("status") != "validated":
                raise ValueError(f"Paper-I SR-SNAKE endpoint is not validated: {regime}")
            validation_result = Path(str(validation.get("result_json") or "")).resolve()
            if validation_result != result_path:
                raise ValueError(f"Paper-I SR-SNAKE validation result path mismatch: {regime}")
            if str(validation.get("result_sha256") or "").lower() != result_hash:
                raise ValueError(f"Paper-I SR-SNAKE validation result hash mismatch: {regime}")

            qiskit = _read_json(qiskit_path)
            validation_qiskit = validation.get("qiskit")
            if not isinstance(validation_qiskit, Mapping):
                raise ValueError(f"Paper-I SR-SNAKE validation lacks Qiskit evidence: {regime}")
            if Path(str(validation_qiskit.get("path") or "")).resolve() != qiskit_path:
                raise ValueError(f"Paper-I SR-SNAKE Qiskit path mismatch: {regime}")
            if str(validation_qiskit.get("sha256") or "").lower() != qiskit_hash:
                raise ValueError(f"Paper-I SR-SNAKE Qiskit hash mismatch: {regime}")
            if qiskit.get("compile_convention") != "table_i_basis_gate_transpile_v1":
                raise ValueError(f"Paper-I SR-SNAKE Qiskit convention mismatch: {regime}")
            if qiskit.get("compiled_resource_qiskit_validated") is not True:
                raise ValueError(f"Paper-I SR-SNAKE Qiskit sidecar is not validated: {regime}")

            controller_round = int(validation["controller_rounds"])
            ansatz_depth = int(validation["active_ansatz_depth"])
            energy = float(validation["energy"])
            exact = float(validation["exact_same_cutoff_energy"])
            error = _positive(validation["absolute_error"])
            numeric_pairs = (
                (controller_round, int(raw_entry["controller_round"]), "controller round"),
                (ansatz_depth, int(raw_entry["ansatz_depth"]), "ansatz depth"),
                (energy, float(raw_entry["energy"]), "energy"),
                (exact, float(raw_entry["exact_same_cutoff_energy"]), "exact energy"),
                (error, _positive(raw_entry["abs_delta_e"]), "absolute error"),
                (error, abs(energy - exact), "same-cutoff error identity"),
            )
            for expected, actual, label in numeric_pairs:
                if not math.isclose(
                    float(expected), float(actual), rel_tol=1.0e-12, abs_tol=1.0e-12
                ):
                    raise ValueError(f"Paper-I SR-SNAKE {label} mismatch: {regime}")

            leakage = validation.get("leakage")
            if not isinstance(leakage, Mapping):
                raise ValueError(f"Paper-I SR-SNAKE validation lacks leakage: {regime}")
            leakage_tolerance = float(leakage["tolerance"])
            max_sector_leakage = float(leakage["maximum_sector_leakage"])
            max_padding_leakage = float(leakage["maximum_padding_leakage"])
            if max(max_sector_leakage, max_padding_leakage) > leakage_tolerance:
                raise ValueError(f"Paper-I SR-SNAKE leakage exceeds tolerance: {regime}")

            accounting = validation.get("estimator_accounting")
            if not isinstance(accounting, Mapping) or accounting.get("complete") is not True:
                raise ValueError(f"Paper-I SR-SNAKE accounting is incomplete: {regime}")
            winning_s = int(accounting["winning_lineage_S_alg"])
            if winning_s != int(raw_entry["winning_lineage_S_alg"]):
                raise ValueError(f"Paper-I SR-SNAKE S_alg mismatch: {regime}")
            exact_blockers = [str(value) for value in accounting.get("exact_blockers") or ()]
            accounting_scope = str(accounting.get("scope") or "")
            if accounting_scope == "continuation_segment_only":
                expected_blockers = [
                    "source_rounds_1_21_state_keyed_ledger_missing_after_failed_run"
                ]
                s_scope = "continuation_segment_rounds_22_to_45_only"
                cumulative_s: int | None = None
                status_label = "validated; S r22--45"
                replay = validation.get("resume")
                replay_label = "resume audit"
            elif accounting_scope == "full_horizon":
                expected_blockers = []
                s_scope = "full_horizon_rounds_1_to_45"
                cumulative_s = int(accounting["cumulative_rounds_1_to_final_S_alg"])
                if cumulative_s != winning_s:
                    raise ValueError(
                        f"Paper-I SR-SNAKE full-horizon cumulative S_alg mismatch: {regime}"
                    )
                status_label = "validated; full S r1--45"
                replay = validation.get("fixed_prefix_reconstruction")
                replay_label = "fixed-prefix reconstruction audit"
            else:
                raise ValueError(f"Paper-I SR-SNAKE accounting scope mismatch: {regime}")
            if exact_blockers != expected_blockers:
                raise ValueError(f"Paper-I SR-SNAKE accounting blocker mismatch: {regime}")
            if raw_entry.get("S_alg_scope") != s_scope:
                raise ValueError(f"Paper-I SR-SNAKE manifest S_alg scope mismatch: {regime}")
            raw_cumulative_s = raw_entry.get("cumulative_rounds_1_to_45_S_alg")
            if (
                raw_cumulative_s is not None
                and int(raw_cumulative_s) != cumulative_s
            ) or (raw_cumulative_s is None) != (cumulative_s is None):
                raise ValueError(f"Paper-I SR-SNAKE manifest cumulative S_alg mismatch: {regime}")
            if [str(value) for value in raw_entry.get("exact_blockers") or ()] != exact_blockers:
                raise ValueError(f"Paper-I SR-SNAKE manifest blocker mismatch: {regime}")

            if not isinstance(replay, Mapping):
                raise ValueError(
                    f"Paper-I SR-SNAKE validation lacks {replay_label}: {regime}"
                )
            replay_discrepancy = float(replay["prefix_replay_abs_discrepancy"])
            if replay_discrepancy > leakage_tolerance:
                raise ValueError(f"Paper-I SR-SNAKE prefix replay mismatch: {regime}")

            n2q = int(validation_qiskit["N2q"])
            d2q = int(validation_qiskit["D2q"])
            dc = int(validation_qiskit["circuit_depth"])
            qiskit_pairs = (
                (n2q, int(qiskit["compiled_count_2q_total"]), "N2q"),
                (d2q, int(qiskit["compiled_depth_2q_total"]), "D2q"),
                (dc, int(qiskit["compiled_depth_total"]), "total depth"),
                (controller_round, int(qiskit["history_position"]), "history position"),
                (ansatz_depth, int(qiskit["logical_operator_count"]), "logical depth"),
                (n2q, int(raw_entry["N2q"]), "manifest N2q"),
                (d2q, int(raw_entry["D2q"]), "manifest D2q"),
                (dc, int(raw_entry["Dc"]), "manifest total depth"),
            )
            for expected, actual, label in qiskit_pairs:
                if expected != actual:
                    raise ValueError(f"Paper-I SR-SNAKE Qiskit {label} mismatch: {regime}")

            trajectory_curve: Curve | None = None
            trajectory_path: Path | None = None
            trajectory_hash: str | None = None
            raw_trajectory_json = str(raw_entry.get("trajectory_json") or "").strip()
            raw_trajectory_hash = str(raw_entry.get("trajectory_sha256") or "").strip()
            if bool(raw_trajectory_json) != bool(raw_trajectory_hash):
                raise ValueError(
                    f"Paper-I SR-SNAKE terminal trajectory fields are incomplete: {regime}"
                )
            if raw_trajectory_json:
                trajectory_path = repo_path_for(raw_trajectory_json)
                trajectory_hash = require_hash(
                    trajectory_path,
                    raw_trajectory_hash,
                    label=f"{regime} terminal trajectory",
                )
                trajectory = _read_json(trajectory_path)
                if trajectory.get("schema") != "paper_i_sr_snake_terminal_trajectory_v1":
                    raise ValueError(
                        f"Paper-I SR-SNAKE terminal trajectory schema mismatch: {regime}"
                    )
                if _normalize_fm_live_regime(trajectory.get("regime")) != regime:
                    raise ValueError(
                        f"Paper-I SR-SNAKE terminal trajectory regime mismatch: {regime}"
                    )

                source_current_path = repo_path_for(raw_entry.get("source_current_json"))
                source_current_hash = require_hash(
                    source_current_path,
                    raw_entry.get("source_current_sha256"),
                    label=f"{regime} terminal trajectory source current",
                )
                linked_paths = (
                    (source_current_path, trajectory.get("source_current_json"), "current"),
                    (result_path, trajectory.get("source_result_json"), "result"),
                    (validation_path, trajectory.get("validation_json"), "validation"),
                )
                for expected_path, raw_path, label in linked_paths:
                    if repo_path_for(raw_path) != expected_path:
                        raise ValueError(
                            "Paper-I SR-SNAKE terminal trajectory "
                            f"{label} path mismatch: {regime}"
                        )
                linked_hashes = (
                    (source_current_hash, trajectory.get("source_current_sha256"), "current"),
                    (result_hash, trajectory.get("source_result_sha256"), "result"),
                    (validation_hash, trajectory.get("validation_sha256"), "validation"),
                )
                for expected_hash, raw_hash, label in linked_hashes:
                    if expected_hash != str(raw_hash or "").lower():
                        raise ValueError(
                            "Paper-I SR-SNAKE terminal trajectory "
                            f"{label} hash mismatch: {regime}"
                        )

                trajectory_rounds = int(trajectory.get("controller_rounds", -1))
                trajectory_point_count = int(trajectory.get("trajectory_point_count", -1))
                raw_point_count = int(raw_entry.get("trajectory_point_count", -1))
                if trajectory_rounds != controller_round:
                    raise ValueError(
                        f"Paper-I SR-SNAKE terminal trajectory round mismatch: {regime}"
                    )
                if trajectory_point_count != controller_round + 1:
                    raise ValueError(
                        f"Paper-I SR-SNAKE terminal trajectory point-count mismatch: {regime}"
                    )
                if raw_point_count != trajectory_point_count:
                    raise ValueError(
                        f"Paper-I SR-SNAKE manifest trajectory point-count mismatch: {regime}"
                    )
                if trajectory.get("trajectory_semantics") != raw_entry.get(
                    "trajectory_semantics"
                ):
                    raise ValueError(
                        f"Paper-I SR-SNAKE terminal trajectory semantics mismatch: {regime}"
                    )
                trajectory_adapt = trajectory.get("adapt_vqe")
                if not isinstance(trajectory_adapt, Mapping):
                    raise ValueError(
                        f"Paper-I SR-SNAKE terminal trajectory lacks ADAPT history: {regime}"
                    )
                if trajectory_adapt.get("history_checkpoint_complete") is not True:
                    raise ValueError(
                        f"Paper-I SR-SNAKE terminal trajectory history is incomplete: {regime}"
                    )
                if int(trajectory_adapt.get("history_count", -1)) != controller_round:
                    raise ValueError(
                        f"Paper-I SR-SNAKE terminal trajectory history count mismatch: {regime}"
                    )
                numeric_trajectory_pairs = (
                    (exact, float(trajectory_adapt["exact_gs_energy"]), "exact energy"),
                    (energy, float(trajectory["validated_terminal_energy"]), "terminal energy"),
                    (
                        error,
                        _positive(trajectory["validated_terminal_abs_delta_e"]),
                        "terminal error",
                    ),
                    (error, _positive(trajectory_adapt["abs_delta_e"]), "curve endpoint"),
                )
                for expected, actual, label in numeric_trajectory_pairs:
                    if not math.isclose(
                        float(expected), float(actual), rel_tol=1.0e-12, abs_tol=1.0e-12
                    ):
                        raise ValueError(
                            "Paper-I SR-SNAKE terminal trajectory "
                            f"{label} mismatch: {regime}"
                        )
                trajectory_curve = _history_curve(
                    trajectory_path,
                    role="paper_i_route4_live",
                    marker_k=controller_round,
                    marker_error=error,
                )
                if len(trajectory_curve.points) != trajectory_point_count:
                    raise ValueError(
                        f"Paper-I SR-SNAKE terminal trajectory curve-size mismatch: {regime}"
                    )

            endpoint = {
                "trajectory_relation": (
                    "same_route_complete_controller_trajectory_with_validated_terminal_endpoint"
                    if trajectory_curve is not None
                    else "same_route_later_validated_terminal_endpoint"
                ),
                "status": PAPER_I_SR_TERMINAL_STATUS,
                "controller_round": controller_round,
                "ansatz_depth": ansatz_depth,
                "terminal": True,
                "validation_json": _rel(validation_path),
                "validation_sha256": validation_hash,
                "trajectory_point_count": (
                    len(trajectory_curve.points) if trajectory_curve is not None else 0
                ),
            }
            if trajectory_curve is None:
                curve = Curve(
                    role="paper_i_route4_live",
                    points=(),
                    marker_k=controller_round,
                    marker_error=error,
                    source_json=_rel(validation_path),
                    source_sha256=validation_hash,
                    source_segments=(endpoint,),
                )
            else:
                curve = Curve(
                    role=trajectory_curve.role,
                    points=trajectory_curve.points,
                    marker_k=trajectory_curve.marker_k,
                    marker_error=trajectory_curve.marker_error,
                    source_json=trajectory_curve.source_json,
                    source_sha256=trajectory_curve.source_sha256,
                    source_segments=(endpoint,),
                )
            resource = {
                "regime": regime,
                "method": "paper_i_route4_live_checkpoint_snake",
                "method_display": RESOURCE_METHOD_DISPLAY[
                    "paper_i_route4_live_checkpoint_snake"
                ],
                "role": "paper_i_route4_live",
                "status": PAPER_I_SR_TERMINAL_STATUS,
                "status_label": status_label,
                "k_pl": controller_round,
                "controller_rounds": controller_round,
                "ansatz_depth": ansatz_depth,
                "energy": energy,
                "exact_same_cutoff_energy": exact,
                "abs_delta_e": error,
                "N2q": n2q,
                "D2q": d2q,
                "Dc": dc,
                "S": winning_s,
                "S_definition": "S_alg = N_H_outer + N_H_refit + N_grad + N_metric",
                "S_scope": s_scope,
                "cumulative_rounds_1_to_45_S_alg": cumulative_s,
                "resource_status": "validated_terminal_fixed_prefix",
                "prefix_semantics": "validated_best_branch_recovery_endpoint",
                "source_json": _rel(validation_path),
                "source_sha256": validation_hash,
                "trajectory_json": (
                    None if trajectory_path is None else _rel(trajectory_path)
                ),
                "trajectory_sha256": trajectory_hash,
                "trajectory_point_count": len(curve.points),
                "result_json": _rel(result_path),
                "result_sha256": result_hash,
                "qiskit_json": _rel(qiskit_path),
                "qiskit_sha256": qiskit_hash,
                "terminal": True,
                "stop_reason": str(validation.get("stop_reason") or ""),
                "maximum_sector_leakage": max_sector_leakage,
                "maximum_padding_leakage": max_padding_leakage,
                "leakage_tolerance": leakage_tolerance,
                "prefix_replay_abs_discrepancy": replay_discrepancy,
                "exact_blockers": exact_blockers,
                "configured_symmetry_policy": "hard_guard",
                "symmetry_evidence_status": "validated_with_checkpoint_leakage",
                "configured_padding_policy": "exact_projected_grouped_v1",
                "padding_evidence_status": "validated_with_checkpoint_leakage",
            }
            provenance = {
                "regime": regime,
                "status": PAPER_I_SR_TERMINAL_STATUS,
                "terminal": True,
                "validation_json": _rel(validation_path),
                "validation_sha256": validation_hash,
                "result_json": _rel(result_path),
                "result_sha256": result_hash,
                "qiskit_json": _rel(qiskit_path),
                "qiskit_sha256": qiskit_hash,
                "command_json": _rel(command_path),
                "command_sha256": command_hash,
                "normalized_manifest_json": _rel(normalized_manifest_path),
                "normalized_manifest_sha256": normalized_manifest_hash,
                "controller_round": controller_round,
                "ansatz_depth": ansatz_depth,
                "energy": energy,
                "exact_same_cutoff_energy": exact,
                "abs_delta_e": error,
                "N2q": n2q,
                "D2q": d2q,
                "Dc": dc,
                "winning_lineage_S_alg": winning_s,
                "S_scope": s_scope,
                "cumulative_rounds_1_to_45_S_alg": cumulative_s,
                "exact_blockers": exact_blockers,
                "maximum_sector_leakage": max_sector_leakage,
                "maximum_padding_leakage": max_padding_leakage,
                "leakage_tolerance": leakage_tolerance,
                "prefix_replay_abs_discrepancy": replay_discrepancy,
                "trajectory_json": (
                    None if trajectory_path is None else _rel(trajectory_path)
                ),
                "trajectory_sha256": trajectory_hash,
                "trajectory_point_count": len(curve.points),
                "marker_semantics": (
                    "complete_controller_trajectory_with_filled_validated_terminal_endpoint"
                    if trajectory_curve is not None
                    else "filled_validated_terminal_endpoint"
                ),
                "command_policy_checks": command_checks,
                "normalized_policy_checks": normalized_checks,
            }
            rows[regime] = {
                "curve": curve,
                "resource": resource,
                "provenance": provenance,
            }
            provenance_entries.append(provenance)
            continue

        if entry_status not in PAPER_I_ROUTE4_NONTERMINAL_STATUSES:
            raise ValueError(f"Paper-I Route-4 live snapshot status mismatch: {regime}")
        if raw_entry.get("terminal") is not False:
            raise ValueError(f"Paper-I Route-4 live snapshot is terminal: {regime}")
        if raw_entry.get("source_kind") != "immutable_current_json_snapshot":
            raise ValueError(f"Paper-I Route-4 live snapshot source-kind mismatch: {regime}")
        if raw_entry.get("history_checkpoint_complete") is not True:
            raise ValueError(f"Paper-I Route-4 checkpoint is incomplete: {regime}")
        pending_fields = set(raw_entry.get("pending_fields") or ())
        expected_pending_fields = {"N2q", "D2q", "Dc", "S_alg"}
        fixed_prefix_evidence = not pending_fields
        if pending_fields not in (set(), expected_pending_fields):
            raise ValueError(f"Paper-I Route-4 pending-resource contract mismatch: {regime}")
        if fixed_prefix_evidence and entry_status != PAPER_I_ROUTE4_STOPPED_STATUS:
            raise ValueError(
                "Paper-I Route-4 fixed-prefix resources are only valid for a "
                f"stopped immutable checkpoint: {regime}"
            )

        snapshot_path = manifest_path_for(raw_entry.get("snapshot_json"))
        snapshot_hash = require_hash(
            snapshot_path,
            raw_entry.get("snapshot_sha256"),
            label=f"{regime} snapshot",
        )
        source_current_hash = str(
            raw_entry.get("source_current_sha256")
            or raw_entry.get("source_current_sha256_at_capture")
            or ""
        ).lower()
        if source_current_hash != snapshot_hash:
            raise ValueError(f"Paper-I Route-4 source-current capture hash mismatch: {regime}")
        source_current_json = str(raw_entry.get("source_current_json") or "").strip()
        if not source_current_json:
            raise ValueError(f"Paper-I Route-4 source-current path is missing: {regime}")

        (
            command_path,
            command_hash,
            normalized_manifest_path,
            normalized_manifest_hash,
            command_checks,
            normalized_checks,
        ) = validate_route_contract(raw_entry, regime=regime)

        payload = _read_json(snapshot_path)
        adapt = payload.get("adapt_vqe")
        checkpoint = payload.get("checkpoint")
        if not isinstance(adapt, Mapping) or not isinstance(checkpoint, Mapping):
            raise ValueError(f"Paper-I Route-4 snapshot lacks checkpoint metadata: {regime}")
        if adapt.get("history_checkpoint_complete") is not True:
            raise ValueError(f"Paper-I Route-4 payload history is incomplete: {regime}")
        if adapt.get("partial_checkpoint") is not True or checkpoint.get("complete") is not False:
            raise ValueError(f"Paper-I Route-4 payload is not a live partial checkpoint: {regime}")
        settings = payload.get("settings")
        overlay = (
            settings.get("historical_singleton_coordinate_trust_overlay")
            if isinstance(settings, Mapping)
            else None
        )
        trust_update = (
            overlay.get("trust_region_update")
            if isinstance(overlay, Mapping)
            else None
        )
        padding_contract = (
            overlay.get("child_padding_contract")
            if isinstance(overlay, Mapping)
            else None
        )
        overlay_checks = {
            "active": isinstance(overlay, Mapping) and overlay.get("active") is True,
            "whitening_active": isinstance(overlay, Mapping)
            and overlay.get("whitening_active") is True,
            "adaptive_trust_active": isinstance(overlay, Mapping)
            and overlay.get("adaptive_trust_active") is True,
            "coordinate_solve": isinstance(overlay, Mapping)
            and overlay.get("coordinate_solve_policy")
            == "supported_metric_whitened_eigh_v1",
            "trust_policy": isinstance(trust_update, Mapping)
            and trust_update.get("policy")
            == "displacement_calibrated_unbounded_v2",
            "phase0_disabled": isinstance(overlay, Mapping)
            and overlay.get("phase0_pilot_enabled") is False,
            "phase2_batching_disabled": isinstance(overlay, Mapping)
            and overlay.get("phase2_batching_enabled") is False,
            "phase3_batching_disabled": isinstance(overlay, Mapping)
            and overlay.get("phase3_batching_enabled") is False,
            "route_a_funnel_disabled": isinstance(overlay, Mapping)
            and overlay.get("route_a_funnel_active") is False,
            "padding_projection_active": isinstance(padding_contract, Mapping)
            and padding_contract.get("projection_active") is True,
            "padding_projection_satisfied": isinstance(padding_contract, Mapping)
            and padding_contract.get("satisfied") is True,
            "padding_projection_source": isinstance(padding_contract, Mapping)
            and padding_contract.get("source") == "exact_projected_grouped_v1",
        }
        if not all(overlay_checks.values()):
            failed = [name for name, passed in overlay_checks.items() if not passed]
            raise ValueError(
                f"Paper-I Route-4 snapshot overlay mismatch for {regime}: {failed}"
            )
        history = _complete_history(payload, path=snapshot_path)
        controller_round = int(raw_entry["controller_round"])
        history_count = int(adapt.get("history_count", -1))
        if controller_round != len(history) or history_count != controller_round:
            raise ValueError(f"Paper-I Route-4 controller-round mismatch: {regime}")
        if int(checkpoint.get("depth", -1)) != controller_round:
            raise ValueError(f"Paper-I Route-4 checkpoint-depth mismatch: {regime}")
        ansatz_depth = int(raw_entry["ansatz_depth"])
        if int(adapt.get("ansatz_depth", -1)) != ansatz_depth or int(
            checkpoint.get("ansatz_depth", -1)
        ) != ansatz_depth:
            raise ValueError(f"Paper-I Route-4 ansatz-depth mismatch: {regime}")
        branch_id = int(raw_entry["branch_id"])
        parent_branch_id = int(raw_entry["parent_branch_id"])
        if int(adapt.get("branch_id", -1)) != branch_id or int(
            checkpoint.get("branch_id", -1)
        ) != branch_id:
            raise ValueError(f"Paper-I Route-4 branch mismatch: {regime}")
        if int(adapt.get("parent_branch_id", -1)) != parent_branch_id or int(
            checkpoint.get("parent_branch_id", -1)
        ) != parent_branch_id:
            raise ValueError(f"Paper-I Route-4 parent-branch mismatch: {regime}")

        energy = float(raw_entry["energy"])
        exact = float(raw_entry["exact_same_cutoff_energy"])
        error = _positive(raw_entry["abs_delta_e"])
        numeric_pairs = (
            (energy, float(adapt["energy"]), "energy"),
            (exact, float(adapt["exact_gs_energy"]), "exact same-cutoff energy"),
            (error, _positive(adapt["abs_delta_e"]), "absolute error"),
            (error, abs(energy - exact), "same-cutoff error identity"),
        )
        for expected, actual, label in numeric_pairs:
            if not math.isclose(expected, actual, rel_tol=1.0e-12, abs_tol=1.0e-12):
                raise ValueError(f"Paper-I Route-4 {label} mismatch: {regime}")

        if fixed_prefix_evidence:
            qiskit_path = repo_path_for(raw_entry.get("qiskit_json"))
            qiskit_hash = require_hash(
                qiskit_path,
                raw_entry.get("qiskit_sha256"),
                label=f"{regime} stopped-prefix Qiskit sidecar",
            )
            replay_path = repo_path_for(raw_entry.get("fixed_prefix_replay_json"))
            replay_hash = require_hash(
                replay_path,
                raw_entry.get("fixed_prefix_replay_sha256"),
                label=f"{regime} fixed-prefix replay",
            )
            accounting_path = repo_path_for(
                raw_entry.get("accounting_and_trajectory_json")
            )
            accounting_hash = require_hash(
                accounting_path,
                raw_entry.get("accounting_and_trajectory_sha256"),
                label=f"{regime} accounting/trajectory sidecar",
            )

            qiskit = _read_json(qiskit_path)
            if qiskit.get("schema") != "paper_i_selected_prefix_qiskit_cost_sidecar_v1":
                raise ValueError(
                    f"Paper-I Route-4 stopped-prefix Qiskit schema mismatch: {regime}"
                )
            if qiskit.get("compile_convention") != "table_i_basis_gate_transpile_v1":
                raise ValueError(
                    f"Paper-I Route-4 stopped-prefix Qiskit convention mismatch: {regime}"
                )
            if qiskit.get("compiled_resource_qiskit_validated") is not True:
                raise ValueError(
                    f"Paper-I Route-4 stopped-prefix Qiskit sidecar is not validated: {regime}"
                )
            n2q = int(qiskit["compiled_count_2q_total"])
            d2q = int(qiskit["compiled_depth_2q_total"])
            dc = int(qiskit["compiled_depth_total"])
            if (
                int(qiskit["history_position"]) != controller_round
                or int(qiskit["logical_operator_count"]) != ansatz_depth
            ):
                raise ValueError(
                    f"Paper-I Route-4 stopped-prefix Qiskit prefix mismatch: {regime}"
                )
            qiskit_numeric_pairs = (
                (energy, float(qiskit["energy_after_opt_at_prefix"]), "Qiskit energy"),
                (error, _positive(qiskit["primary_error_at_prefix"]), "Qiskit error"),
                (n2q, int(raw_entry["N2q"]), "manifest N2q"),
                (d2q, int(raw_entry["D2q"]), "manifest D2q"),
                (dc, int(raw_entry["Dc"]), "manifest total depth"),
            )
            for expected, actual, label in qiskit_numeric_pairs:
                if not math.isclose(
                    float(expected), float(actual), rel_tol=1.0e-12, abs_tol=1.0e-12
                ):
                    raise ValueError(
                        f"Paper-I Route-4 stopped-prefix {label} mismatch: {regime}"
                    )

            replay = _read_json(replay_path)
            if replay.get("schema") != "paper_i_hh_fixed_prefix_reconstruction_v1":
                raise ValueError(
                    f"Paper-I Route-4 fixed-prefix replay schema mismatch: {regime}"
                )
            if replay.get("status") != "validated":
                raise ValueError(
                    f"Paper-I Route-4 fixed-prefix replay is not validated: {regime}"
                )
            if str(replay.get("source_result_sha256") or "").lower() != snapshot_hash:
                raise ValueError(
                    f"Paper-I Route-4 fixed-prefix replay source hash mismatch: {regime}"
                )
            if (
                int(replay["controller_round"]) != controller_round
                or int(replay["operator_count"]) != ansatz_depth
                or replay.get("ordered_labels_exact_match") is not True
                or replay.get("logical_parameters_exact_match") is not True
                or replay.get("runtime_parameters_exact_match") is not True
            ):
                raise ValueError(
                    f"Paper-I Route-4 fixed-prefix replay ordering mismatch: {regime}"
                )
            replay_tolerance = float(replay["energy_tolerance"])
            replay_discrepancy = float(replay["prefix_replay_abs_discrepancy"])
            if replay_discrepancy > replay_tolerance:
                raise ValueError(
                    f"Paper-I Route-4 fixed-prefix replay energy mismatch: {regime}"
                )
            replay_numeric_pairs = (
                (energy, float(replay["saved_energy"]), "saved replay energy"),
                (error, _positive(replay["saved_absolute_error"]), "saved replay error"),
                (
                    float(replay["replayed_energy"]),
                    energy - replay_discrepancy
                    if float(replay["replayed_energy"]) <= energy
                    else energy + replay_discrepancy,
                    "replayed energy discrepancy identity",
                ),
            )
            for expected, actual, label in replay_numeric_pairs:
                if not math.isclose(
                    float(expected), float(actual), rel_tol=1.0e-12, abs_tol=1.0e-12
                ):
                    raise ValueError(
                        f"Paper-I Route-4 fixed-prefix {label} mismatch: {regime}"
                    )

            replay_qiskit = replay.get("qiskit")
            if not isinstance(replay_qiskit, Mapping):
                raise ValueError(
                    f"Paper-I Route-4 fixed-prefix replay lacks Qiskit evidence: {regime}"
                )
            if repo_path_for(replay_qiskit.get("path")) != qiskit_path:
                raise ValueError(
                    f"Paper-I Route-4 fixed-prefix Qiskit path mismatch: {regime}"
                )
            replay_qiskit_pairs = (
                (qiskit_hash, str(replay_qiskit.get("sha256") or "").lower(), "hash"),
                (n2q, int(replay_qiskit["N2q"]), "N2q"),
                (d2q, int(replay_qiskit["D2q"]), "D2q"),
                (dc, int(replay_qiskit["circuit_depth"]), "total depth"),
            )
            for expected, actual, label in replay_qiskit_pairs:
                if expected != actual:
                    raise ValueError(
                        f"Paper-I Route-4 fixed-prefix replay Qiskit {label} mismatch: {regime}"
                    )

            leakage = replay.get("leakage")
            if not isinstance(leakage, Mapping):
                raise ValueError(
                    f"Paper-I Route-4 fixed-prefix replay lacks leakage evidence: {regime}"
                )
            leakage_tolerance = float(leakage["tolerance"])
            max_sector_leakage = float(leakage["maximum_sector_leakage"])
            max_padding_leakage = float(leakage["maximum_padding_leakage"])
            if max(max_sector_leakage, max_padding_leakage) > leakage_tolerance:
                raise ValueError(
                    f"Paper-I Route-4 fixed-prefix replay leakage exceeds tolerance: {regime}"
                )

            estimator_accounting = replay.get("estimator_accounting")
            if (
                not isinstance(estimator_accounting, Mapping)
                or estimator_accounting.get("complete") is not True
            ):
                raise ValueError(
                    f"Paper-I Route-4 stopped-prefix accounting is incomplete: {regime}"
                )
            s_scope = str(estimator_accounting.get("scope") or "")
            expected_s_scope = (
                f"display_prefix_rounds_1_to_{controller_round}_"
                "retained_history_reconstruction"
            )
            if s_scope != expected_s_scope or raw_entry.get("S_alg_scope") != s_scope:
                raise ValueError(
                    f"Paper-I Route-4 stopped-prefix S_alg scope mismatch: {regime}"
                )
            s_components_raw = estimator_accounting.get("components")
            if not isinstance(s_components_raw, Mapping):
                raise ValueError(
                    f"Paper-I Route-4 stopped-prefix accounting lacks components: {regime}"
                )
            s_components = {
                key: int(s_components_raw[key])
                for key in ("N_H_outer", "N_H_refit", "N_grad", "N_metric")
            }
            winning_s = int(estimator_accounting["winning_lineage_S_alg"])
            if sum(s_components.values()) != winning_s:
                raise ValueError(
                    f"Paper-I Route-4 stopped-prefix S_alg does not close: {regime}"
                )
            if winning_s != int(raw_entry["winning_lineage_S_alg"]):
                raise ValueError(
                    f"Paper-I Route-4 stopped-prefix manifest S_alg mismatch: {regime}"
                )
            if int(qiskit["instrumented_runtime_S"]) != winning_s:
                raise ValueError(
                    f"Paper-I Route-4 stopped-prefix Qiskit S_alg mismatch: {regime}"
                )
            exact_blockers = [
                str(value)
                for value in estimator_accounting.get("exact_blockers") or ()
            ]
            if [str(value) for value in raw_entry.get("exact_blockers") or ()] != exact_blockers:
                raise ValueError(
                    f"Paper-I Route-4 stopped-prefix accounting blocker mismatch: {regime}"
                )

            accounting_sidecar = _read_json(accounting_path)
            if (
                accounting_sidecar.get("schema")
                != "paper_i_sr_snake_stopped_checkpoint_accounting_trajectory_v1"
            ):
                raise ValueError(
                    f"Paper-I Route-4 accounting/trajectory schema mismatch: {regime}"
                )
            retained = accounting_sidecar.get("retained_history_reconstruction")
            retained_components = (
                retained.get("components") if isinstance(retained, Mapping) else None
            )
            if not isinstance(retained_components, Mapping):
                raise ValueError(
                    f"Paper-I Route-4 accounting/trajectory lacks retained S_alg: {regime}"
                )
            retained_pairs = (
                (winning_s, int(retained_components["S_alg"]), "S_alg"),
                (s_components["N_H_outer"], int(retained_components["N_H_outer"]), "N_H_outer"),
                (s_components["N_H_refit"], int(retained_components["N_H_refit"]), "N_H_refit"),
                (s_components["N_grad"], int(retained_components["N_grad"]), "N_grad"),
                (s_components["N_metric"], int(retained_components["N_metric"]), "N_metric"),
            )
            if any(expected != actual for expected, actual, _ in retained_pairs):
                failed = [
                    label
                    for expected, actual, label in retained_pairs
                    if expected != actual
                ]
                raise ValueError(
                    "Paper-I Route-4 accounting/trajectory S_alg mismatch for "
                    f"{regime}: {failed}"
                )

            history_curve = _history_curve(
                snapshot_path,
                role="paper_i_route4_live",
                marker_k=controller_round,
                marker_error=error,
            )
            endpoint = {
                "trajectory_relation": "same_route_stopped_checkpoint_complete_history",
                "status": entry_status,
                "controller_round": controller_round,
                "ansatz_depth": ansatz_depth,
                "branch_id": branch_id,
                "parent_branch_id": parent_branch_id,
                "terminal": False,
                "fixed_prefix_resources_validated": True,
                "trajectory_point_count": len(history_curve.points),
            }
            curve = Curve(
                role=history_curve.role,
                points=history_curve.points,
                marker_k=history_curve.marker_k,
                marker_error=history_curve.marker_error,
                source_json=history_curve.source_json,
                source_sha256=history_curve.source_sha256,
                source_segments=(endpoint,),
            )
            resource = {
                "regime": regime,
                "method": "paper_i_route4_live_checkpoint_snake",
                "method_display": RESOURCE_METHOD_DISPLAY[
                    "paper_i_route4_live_checkpoint_snake"
                ],
                "role": "paper_i_route4_live",
                "status": entry_status,
                "status_label": f"stopped r{controller_round}; fixed prefix",
                "k_pl": controller_round,
                "controller_rounds": controller_round,
                "ansatz_depth": ansatz_depth,
                "branch_id": branch_id,
                "parent_branch_id": parent_branch_id,
                "energy": energy,
                "exact_same_cutoff_energy": exact,
                "abs_delta_e": error,
                "N2q": n2q,
                "D2q": d2q,
                "Dc": dc,
                "S": winning_s,
                "S_definition": "S_alg = N_H_outer + N_H_refit + N_grad + N_metric",
                "S_components": s_components,
                "S_scope": s_scope,
                "S_source": "retained_history_reconstruction",
                "resource_status": "validated_stopped_fixed_prefix",
                "prefix_semantics": "immutable_stopped_best_branch_fixed_prefix",
                "source_json": _rel(snapshot_path),
                "source_sha256": snapshot_hash,
                "qiskit_json": _rel(qiskit_path),
                "qiskit_sha256": qiskit_hash,
                "fixed_prefix_replay_json": _rel(replay_path),
                "fixed_prefix_replay_sha256": replay_hash,
                "accounting_and_trajectory_json": _rel(accounting_path),
                "accounting_and_trajectory_sha256": accounting_hash,
                "terminal": False,
                "trajectory_status": "complete_retained_history_rounds_1_to_43",
                "prefix_replay_abs_discrepancy": replay_discrepancy,
                "maximum_sector_leakage": max_sector_leakage,
                "maximum_padding_leakage": max_padding_leakage,
                "leakage_tolerance": leakage_tolerance,
                "exact_blockers": exact_blockers,
                "raw_state_keyed_accounting_status": "unresolved_not_preserved",
                "configured_symmetry_policy": "hard_guard",
                "symmetry_evidence_status": "validated_at_replayed_fixed_prefix",
                "configured_padding_policy": "exact_projected_grouped_v1",
                "padding_evidence_status": "validated_at_replayed_fixed_prefix",
                "final_full_refit_executed": False,
            }
            provenance = {
                "regime": regime,
                "status": entry_status,
                "terminal": False,
                "snapshot_json": _rel(snapshot_path),
                "snapshot_sha256": snapshot_hash,
                "source_current_json": source_current_json,
                "source_current_sha256_at_capture": source_current_hash,
                "command_json": _rel(command_path),
                "command_sha256": command_hash,
                "normalized_manifest_json": _rel(normalized_manifest_path),
                "normalized_manifest_sha256": normalized_manifest_hash,
                "qiskit_json": _rel(qiskit_path),
                "qiskit_sha256": qiskit_hash,
                "fixed_prefix_replay_json": _rel(replay_path),
                "fixed_prefix_replay_sha256": replay_hash,
                "accounting_and_trajectory_json": _rel(accounting_path),
                "accounting_and_trajectory_sha256": accounting_hash,
                "controller_round": controller_round,
                "ansatz_depth": ansatz_depth,
                "branch_id": branch_id,
                "parent_branch_id": parent_branch_id,
                "energy": energy,
                "exact_same_cutoff_energy": exact,
                "abs_delta_e": error,
                "N2q": n2q,
                "D2q": d2q,
                "Dc": dc,
                "winning_lineage_S_alg": winning_s,
                "S_scope": s_scope,
                "S_components": s_components,
                "exact_blockers": exact_blockers,
                "maximum_sector_leakage": max_sector_leakage,
                "maximum_padding_leakage": max_padding_leakage,
                "leakage_tolerance": leakage_tolerance,
                "prefix_replay_abs_discrepancy": replay_discrepancy,
                "marker_semantics": "complete_checkpoint_trajectory_with_fixed_prefix_resources",
                "resource_fields": "validated_fixed_prefix_with_reconstructed_S_alg",
                "command_policy_checks": command_checks,
                "normalized_policy_checks": normalized_checks,
                "snapshot_overlay_checks": overlay_checks,
                "symmetry_evidence_status": "fixed_prefix_leakage_validated",
                "padding_evidence_status": "fixed_prefix_leakage_validated",
            }
            rows[regime] = {
                "curve": curve,
                "resource": resource,
                "provenance": provenance,
            }
            provenance_entries.append(provenance)
            continue

        endpoint = {
            "marker_only": True,
            "trajectory_relation": "same_route_later_checkpoint",
            "status": entry_status,
            "controller_round": controller_round,
            "ansatz_depth": ansatz_depth,
            "branch_id": branch_id,
            "parent_branch_id": parent_branch_id,
            "terminal": False,
        }
        curve = Curve(
            role="paper_i_route4_live",
            points=(),
            marker_k=controller_round,
            marker_error=error,
            source_json=_rel(snapshot_path),
            source_sha256=snapshot_hash,
            source_segments=(endpoint,),
        )
        resource = {
            "regime": regime,
            "method": "paper_i_route4_live_checkpoint_snake",
            "method_display": RESOURCE_METHOD_DISPLAY[
                "paper_i_route4_live_checkpoint_snake"
            ],
            "role": "paper_i_route4_live",
            "status": entry_status,
            "status_label": (
                "stopped nonterminal"
                if entry_status == PAPER_I_ROUTE4_STOPPED_STATUS
                else "nonterminal"
            ),
            "k_pl": controller_round,
            "controller_rounds": controller_round,
            "ansatz_depth": ansatz_depth,
            "branch_id": branch_id,
            "parent_branch_id": parent_branch_id,
            "energy": energy,
            "exact_same_cutoff_energy": exact,
            "abs_delta_e": error,
            "N2q": None,
            "D2q": None,
            "Dc": None,
            "S": None,
            "resource_status": "pending_nonterminal_checkpoint",
            "prefix_semantics": "immutable_nonterminal_best_branch_checkpoint",
            "source_json": _rel(snapshot_path),
            "source_sha256": snapshot_hash,
            "terminal": False,
            "status_endpoint_only": True,
            "exact_blockers": [
                "qiskit_compile_not_run_for_live_checkpoint",
                "state_keyed_S_alg_not_finalized_for_live_checkpoint",
                "fixed_sector_leakage_not_serialized_in_live_checkpoint",
                "binary_padding_leakage_not_serialized_in_live_checkpoint",
            ],
            "configured_symmetry_policy": "hard_guard",
            "symmetry_evidence_status": (
                "command_and_normalized_policy_validated_checkpoint_leakage_unresolved"
            ),
            "configured_padding_policy": "exact_projected_grouped_v1",
            "padding_evidence_status": (
                "snapshot_contract_satisfied_checkpoint_leakage_unresolved"
            ),
        }
        provenance = {
            "regime": regime,
            "status": entry_status,
            "terminal": False,
            "snapshot_json": _rel(snapshot_path),
            "snapshot_sha256": snapshot_hash,
            "source_current_json": source_current_json,
            "source_current_sha256_at_capture": source_current_hash,
            "command_json": _rel(command_path),
            "command_sha256": command_hash,
            "normalized_manifest_json": _rel(normalized_manifest_path),
            "normalized_manifest_sha256": normalized_manifest_hash,
            "controller_round": controller_round,
            "ansatz_depth": ansatz_depth,
            "branch_id": branch_id,
            "parent_branch_id": parent_branch_id,
            "energy": energy,
            "exact_same_cutoff_energy": exact,
            "abs_delta_e": error,
            "marker_semantics": "marker_only_no_trajectory_interpolation",
            "resource_fields": "pending_until_fixed_prefix_sidecars",
            "command_policy_checks": command_checks,
            "normalized_policy_checks": normalized_checks,
            "snapshot_overlay_checks": overlay_checks,
            "symmetry_evidence_status": (
                "hard_guard_configured_checkpoint_leakage_not_serialized"
            ),
            "padding_evidence_status": (
                "exact_projection_contract_satisfied_checkpoint_leakage_not_serialized"
            ),
        }
        rows[regime] = {
            "curve": curve,
            "resource": resource,
            "provenance": provenance,
        }
        provenance_entries.append(provenance)

    strong_submission: dict[str, Any] | None = None
    if mixed_recovery:
        raw_submission = manifest.get("strong_strong_submission")
        if not isinstance(raw_submission, Mapping):
            raise ValueError("Paper-I SR-SNAKE mixed recovery lacks strong-strong submission")
        submission_status = str(raw_submission.get("status") or "")
        if submission_status not in {"submitted_pending_chtc", "completed_fetched"}:
            raise ValueError("Paper-I SR-SNAKE strong-strong submission status mismatch")
        cluster_id = int(raw_submission.get("cluster_id", -1))
        if cluster_id <= 0:
            raise ValueError("Paper-I SR-SNAKE strong-strong cluster id is invalid")
        receipt_path = repo_path_for(raw_submission.get("submission_receipt"))
        receipt_hash = require_hash(
            receipt_path,
            raw_submission.get("submission_receipt_sha256"),
            label="strong-strong submission receipt",
        )
        strong_submission = {
            "status": submission_status,
            "cluster_id": cluster_id,
            "batch_name": str(raw_submission.get("batch_name") or ""),
            "submission_receipt": _rel(receipt_path),
            "submission_receipt_sha256": receipt_hash,
        }

    terminal_regimes = [
        entry["regime"]
        for entry in provenance_entries
        if entry.get("status") == PAPER_I_SR_TERMINAL_STATUS
    ]
    running_regimes = [
        entry["regime"]
        for entry in provenance_entries
        if entry.get("status") == PAPER_I_ROUTE4_LIVE_STATUS
    ]
    stopped_regimes = [
        entry["regime"]
        for entry in provenance_entries
        if entry.get("status") == PAPER_I_ROUTE4_STOPPED_STATUS
    ]
    fixed_prefix_checkpoint_regimes = [
        entry["regime"]
        for entry in provenance_entries
        if entry.get("marker_semantics")
        == "complete_checkpoint_trajectory_with_fixed_prefix_resources"
    ]
    complete_terminal_trajectory_regimes = [
        entry["regime"]
        for entry in provenance_entries
        if entry.get("marker_semantics")
        == "complete_controller_trajectory_with_filled_validated_terminal_endpoint"
    ]
    return rows, {
        "schema": schema,
        "manifest_json": _rel(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "captured_at_utc": captured_at,
        "campaign_root": str(manifest.get("campaign_root") or ""),
        "route": "route_4_whitened_adaptive",
        "stable_family_id": str(
            manifest.get("stable_family_id") or "singleton_response_snake"
        ),
        "evidence_status": str(manifest.get("evidence_status") or ""),
        "terminal": manifest.get("terminal"),
        "policy": expected_policy,
        "queue_state_snapshot": _rel(queue_path),
        "queue_state_snapshot_sha256": queue_hash,
        "queue_state_source": str(manifest.get("queue_state_source") or ""),
        "queue_state_source_sha256_at_capture": queue_source_hash,
        "source_lock_and_settings_diff_json": _rel(source_lock_path),
        "source_lock_and_settings_diff_sha256": source_lock_hash,
        "entry_count": len(provenance_entries),
        "entries": provenance_entries,
        "running_regimes": running_regimes,
        "stopped_regimes": stopped_regimes,
        "fixed_prefix_checkpoint_regimes": fixed_prefix_checkpoint_regimes,
        "complete_terminal_trajectory_regimes": complete_terminal_trajectory_regimes,
        "terminal_regimes": terminal_regimes,
        "strong_strong_submission": strong_submission,
        "endpoint_semantics": (
            "mixed_complete_checkpoint_and_terminal_trajectories_with_validated_endpoints"
            if mixed_recovery and complete_terminal_trajectory_regimes
            else "mixed_complete_checkpoint_trajectories_and_filled_validated_terminal_endpoints"
            if mixed_recovery and fixed_prefix_checkpoint_regimes
            else "mixed_open_nonterminal_and_filled_validated_terminal_endpoints"
            if mixed_recovery
            else "marker_only_no_trajectory_interpolation"
        ),
        "preserved_row_relation": "additive_to_preserved_round21_rows",
        "resource_fields": (
            "per_entry_pending_or_fixed_prefix_or_terminal_with_explicit_S_scope"
            if mixed_recovery and fixed_prefix_checkpoint_regimes
            else "per_entry_pending_or_validated_with_explicit_S_scope"
            if mixed_recovery
            else "pending_until_fixed_prefix_sidecars"
        ),
    }


def _sr_recovery_summary(campaign: Mapping[str, Any] | None) -> str:
    if campaign is None:
        return "recovery evidence not supplied"
    endpoint_summaries: list[str] = []
    for entry in campaign.get("entries") or ():
        if not isinstance(entry, Mapping):
            continue
        regime = str(entry.get("regime") or "unknown")
        display = REGIME_DISPLAY.get(regime, regime)
        controller_round = entry.get("controller_round", "?")
        status = str(entry.get("status") or "")
        if status == PAPER_I_SR_TERMINAL_STATUS:
            s_scope = str(entry.get("S_scope") or "")
            scope_label = (
                "full S r1--45"
                if s_scope == "full_horizon_rounds_1_to_45"
                else "S r22--45 only"
                if s_scope == "continuation_segment_rounds_22_to_45_only"
                else "explicit S scope"
            )
            endpoint_summaries.append(
                f"{display} r{controller_round} validated ({scope_label})"
            )
        elif status == PAPER_I_ROUTE4_STOPPED_STATUS:
            if (
                entry.get("marker_semantics")
                == "complete_checkpoint_trajectory_with_fixed_prefix_resources"
            ):
                endpoint_summaries.append(
                    f"{display} r{controller_round} stopped fixed prefix "
                    "(full trajectory; reconstructed S)"
                )
            else:
                endpoint_summaries.append(
                    f"{display} r{controller_round} stopped nonterminal"
                )
        elif status == PAPER_I_ROUTE4_LIVE_STATUS:
            endpoint_summaries.append(
                f"{display} r{controller_round} running nonterminal"
            )
    submission = campaign.get("strong_strong_submission")
    if isinstance(submission, Mapping) and not any(
        entry.get("regime") == "strong-strong"
        and entry.get("status") == PAPER_I_SR_TERMINAL_STATUS
        for entry in campaign.get("entries") or ()
        if isinstance(entry, Mapping)
    ):
        endpoint_summaries.append(
            "strong-strong CHTC " + str(submission.get("status") or "unknown")
        )
    return "recovery overlay: " + "; ".join(endpoint_summaries)


def _jr_l10_rows_by_regime(output_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    supplemental_dir = output_dir / "supplemental_selected_prefix_qiskit"
    supplemental_dir.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict[str, Any]] = {}
    source_boundaries = {"weak-weak": 17}
    for regime in REGIME_ORDER:
        paths = JR_L10_SEGMENTS_BY_REGIME[regime]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(missing[0])
        curve = (
            _stitched_history_curve(paths, role="jr_selected")
            if len(paths) > 1
            else _history_curve(paths[0], role="jr_selected")
        )
        query_work = _stitched_winning_lineage_query_work(paths)
        terminal_path = paths[-1]
        terminal_payload = _read_json(terminal_path)
        terminal_history = _complete_history(terminal_payload, path=terminal_path)
        query_path = supplemental_dir / f"jr-l10-{regime}-stitched-query-work.json"
        query_path.write_text(
            json.dumps(query_work, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        exact_s, s_source, legacy_proxy_s = _stitched_query_resource_override(
            query_work
        )
        resource = _supplemental_resource_row(
            regime=regime,
            method="jr_snake_whitened_l10",
            source_json=terminal_path,
            history_position=len(terminal_history),
            display_k_pl=curve.marker_k,
            expected_error=curve.marker_error,
            sidecar_json=supplemental_dir / f"jr-l10-{regime}-qiskit.json",
            s_override=exact_s,
            s_source_override=s_source,
            source_segments=curve.source_segments,
            query_work_sidecar=query_path,
        )
        if exact_s is None:
            resource.update(
                S=None,
                S_source=s_source,
                S_status=(
                    "legacy_proxy_not_exact"
                    if legacy_proxy_s is not None
                    else "unavailable_raw_occurrence_stitching"
                ),
                legacy_proxy_S=legacy_proxy_s,
            )
        else:
            resource["S_status"] = "ok_exact_primitive_union"
        summary = terminal_payload.get("summary")
        if not isinstance(summary, Mapping):
            summary = terminal_payload.get("adapt_vqe") or {}
        rows[regime] = {
            "status": "complete",
            "curve": curve,
            "resource": resource,
            "stop_reason": str(summary.get("stop_reason") or "unknown"),
            "total_controller_rounds": curve.marker_k,
            "mixed_source_boundary_after_round": source_boundaries.get(regime),
            "query_work": query_work,
        }
    return rows, {
        "policy": "M32_M24_C32_C25_L10_B2_lambda0_whitened_adaptive_unbounded_v2",
        "source_boundaries": source_boundaries,
        "regime_rounds": {
            regime: int(rows[regime]["total_controller_rounds"])
            for regime in REGIME_ORDER
        },
        "selector_exhausted_regimes": [
            regime
            for regime in REGIME_ORDER
            if rows[regime]["stop_reason"] == "joint_geometry_selector_exhausted"
        ],
    }


def _fm_ablation_rows(
    *,
    campaign_root: Path,
    supplemental_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    ledger_path = campaign_root / "pareto_ledger.json"
    ledger = _read_json(ledger_path)
    rows = {
        regime: {"curves": {}, "resources": []}
        for regime in REGIME_ORDER
    }
    regime_prefixes = {
        "weak_weak": "weak-weak",
        "intermediate_weak": "intermediate-weak",
        "strong_weak_u8": "strong-weak",
        "weak_strong": "weak-strong",
        "intermediate_strong": "intermediate-strong",
        "strong_strong_u8": "strong-strong",
    }
    completed = 0
    for cell in ledger.get("cells", []):
        if not isinstance(cell, Mapping):
            continue
        cell_id = str(cell.get("cell_id") or "")
        prefix = next((key for key in regime_prefixes if cell_id.startswith(key + "__")), None)
        if prefix is None:
            continue
        regime = regime_prefixes[prefix]
        if "__inverse_rbfgs_qbroyd_defaults__" in cell_id:
            role = "fm_qbroyd_default"
        elif "__inverse_rbfgs_qbroyd_off__" in cell_id:
            role = "fm_qbroyd_off"
        else:
            continue
        status = str(cell.get("status") or "pending")
        if status != "complete":
            rows[regime]["resources"].append(
                {
                    "regime": regime,
                    "method": role,
                    "method_display": RESOURCE_METHOD_DISPLAY[role],
                    "status": status,
                }
            )
            continue
        paths = cell.get("paths")
        evidence = cell.get("evidence")
        if not isinstance(paths, Mapping) or not isinstance(evidence, Mapping):
            raise ValueError(f"Completed FM ablation cell lacks evidence: {cell_id}")
        result_path = Path(str(paths["result_json"]))
        query_path = Path(str(paths["query_work_sidecar"]))
        campaign_qiskit_path = Path(str(paths["qiskit_sidecar"]))
        result_hash = _sha256(result_path)
        if str(evidence.get("result_sha256")) != result_hash:
            raise ValueError(f"FM ablation result hash mismatch: {cell_id}")
        result = _read_json(result_path)
        adapt = result.get("adapt_vqe")
        if not isinstance(adapt, Mapping) or adapt.get("success") is not True:
            raise ValueError(f"FM ablation result is not successful: {cell_id}")
        if str(adapt.get("adapt_reoptimization_route")) != "formal_manifold_warm_start_v1":
            raise ValueError(f"FM ablation route identity mismatch: {cell_id}")
        history = _complete_history(result, path=result_path)
        campaign_qiskit = _read_json(campaign_qiskit_path)
        query = _read_json(query_path)
        if campaign_qiskit.get("compiled_resource_qiskit_validated") is not True:
            raise ValueError(f"FM ablation Qiskit validation failed: {cell_id}")
        if str(campaign_qiskit.get("source_result_sha256")) != result_hash:
            raise ValueError(f"FM ablation Qiskit hash mismatch: {cell_id}")
        if query.get("science_valid") is not True:
            raise ValueError(f"FM ablation query closure failed: {cell_id}")
        curve = _history_curve(result_path, role=role)
        resource = _supplemental_resource_row(
            regime=regime,
            method=role,
            source_json=result_path,
            history_position=len(history),
            expected_error=float(adapt["abs_delta_e"]),
            sidecar_json=supplemental_dir / f"{role}-{regime}.json",
            s_override=float(query["winning_branch"]["expanded_query_work"]),
            s_source_override="formal_manifold_query_work_sidecar.winning_branch.expanded_query_work",
            query_work_sidecar=query_path,
        )
        resource.update(status="complete", cell_id=cell_id)
        rows[regime]["curves"][role] = curve
        rows[regime]["resources"].append(resource)
        completed += 1
    return rows, {
        "ledger_json": str(ledger_path.resolve()),
        "ledger_sha256": _sha256(ledger_path),
        "completed_variant_cells": completed,
    }


def _repaired_l25_evidence(
    output_dir: Path,
    prior_report_json: Path | None = None,
) -> dict[str, Any]:
    from pipelines.reporting import (
        build_paper_i_hh_joint_response_six_regime_overlay_l25_repaired as repaired,
    )

    try:
        return repaired._load_evidence(output_dir)
    except FileNotFoundError:
        # Storage cleanup may remove the unpacked CHTC retrieval after this
        # report has already locked its curves, resources, and source hashes.
        # Preserve those prior report rows rather than fabricating replacement
        # raw artifacts or silently dropping the comparison.
        prior_report = (
            prior_report_json.resolve()
            if prior_report_json is not None
            else output_dir / f"{STEM}.json"
        )
        prior = _read_json(prior_report)
        campaign = dict(prior.get("repaired_l25_campaign") or {})
        jr_page = dict((prior.get("pages") or {}).get("jr_policies") or {})
        prior_rows = {
            str(row.get("regime")): row
            for row in jr_page.get("regimes") or []
        }
        regimes: list[dict[str, Any]] = []
        for regime in REGIME_ORDER:
            row = prior_rows.get(regime)
            if row is None:
                raise FileNotFoundError(
                    f"Missing preserved repaired-L25 row for {regime}: {prior_report}"
                )
            curve = dict((row.get("curves") or {}).get("jr_l25") or {})
            resource = next(
                (
                    dict(item)
                    for item in row.get("resource_table_rows") or []
                    if item.get("method") == "repaired_l25_snake"
                ),
                None,
            )
            if not curve.get("points") or resource is None:
                raise ValueError(
                    f"Incomplete preserved repaired-L25 evidence for {regime}"
                )
            regimes.append(
                {
                    "regime": regime,
                    "display": REGIME_DISPLAY[regime],
                    "curves": {"repaired_l25": curve},
                    "resource_table_rows": [resource],
                    "source_recovery": "prior_report_provenance_v1",
                }
            )
        campaign.update(
            {
                "regimes": regimes,
                "source_recovery": {
                    "mode": "prior_report_provenance_v1",
                    "report_json": _rel(prior_report),
                    "report_sha256": _sha256(prior_report),
                },
            }
        )
        return campaign


def _fm_prior_role(role: str) -> str:
    return {
        "fm_qbroyd_default": "fm_qbroyd_on_prior",
        "fm_qbroyd_off": "fm_qbroyd_off_prior",
    }[role]


def _fm_prior_resource(resource: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    prior_role = _fm_prior_role(role)
    result = dict(resource)
    result["method"] = prior_role
    result["method_display"] = RESOURCE_METHOD_DISPLAY[prior_role]
    result["evidence_relation"] = "prior_terminal_preserved_beside_live_snapshot"
    return result


def _merge_fm_model_live_evidence(
    fm: Mapping[str, Any],
    live_off: Mapping[str, Any] | None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Merge live qB-off prefix while retaining prior terminal evidence."""

    curves: dict[str, dict[str, Any]] = {}
    prior_resource = dict(fm["resource"])
    prior_resource["method"] = "fm_qbroyd_off"
    prior_resource["method_display"] = RESOURCE_METHOD_DISPLAY["fm_qbroyd_off"]
    if live_off is None:
        if isinstance(fm.get("curve"), Curve):
            curves["fm_qbroyd_off"] = _curve_payload(fm["curve"])
        return curves, [prior_resource]

    if isinstance(fm.get("curve"), Curve):
        prior_curve = _curve_payload(fm["curve"])
        prior_curve["role"] = "fm_qbroyd_off_prior"
        curves["fm_qbroyd_off_prior"] = prior_curve
    curves["fm_qbroyd_off"] = _curve_payload(live_off["curve"])
    resources = [dict(live_off["resource"])]
    if str(prior_resource.get("status") or "complete") == "complete":
        resources.append(
            _fm_prior_resource(prior_resource, role="fm_qbroyd_off")
        )
    return curves, resources


def _merge_fm_policy_live_evidence(
    fm_variant: Mapping[str, Any],
    live_entries: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Add live qB prefixes and retain only source-backed prior terminal rows."""

    curves = {
        role: _curve_payload(curve)
        for role, curve in fm_variant["curves"].items()
    }
    resources = [dict(row) for row in fm_variant["resources"]]
    for role, live_entry in live_entries.items():
        if role in curves:
            prior_role = _fm_prior_role(role)
            prior_curve = dict(curves.pop(role))
            prior_curve["role"] = prior_role
            curves[prior_role] = prior_curve
        preserved_resources: list[dict[str, Any]] = []
        for resource in resources:
            if str(resource.get("method")) != role:
                preserved_resources.append(resource)
            elif str(resource.get("status") or "complete") == "complete":
                preserved_resources.append(_fm_prior_resource(resource, role=role))
        resources = preserved_resources
        curves[role] = _curve_payload(live_entry["curve"])
        resources.append(dict(live_entry["resource"]))
    return curves, resources


def build_evidence(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    fm_campaign_root: Path = FM_CAMPAIGN_ROOT,
    fm_live_snapshot_manifest: Path | None = None,
    fm_live_status_snapshot: Path | None = None,
    fm_completed_resource_recovery_manifest: Path | None = None,
    fm_stopped_snapshot_manifest: Path | None = None,
    paper_i_route4_live_snapshot_manifest: Path | None = None,
    jr_chtc_live_snapshot_manifest: Path | None = None,
    prior_report_json: Path | None = None,
    sr_expanded_chart_whitening_validation_json: Path | None = None,
    sr_expanded_chart_whitening_intermediate_weak_validation_json: Path | None = None,
    sr_expanded_chart_whitening_intermediate_weak_qiskit_json: Path | None = None,
) -> dict[str, Any]:
    comparison_path = COMPARISON_JSON
    reference_path = PAPER_I_REFERENCE_JSON
    resource_path = QISKIT_TABLE_JSON
    comparison = _read_json(comparison_path)
    references = _paper_reference_rows(_read_json(reference_path))
    resource_payload = _read_json(resource_path)
    resource_rows = _resource_rows_by_regime(resource_payload)
    comparison_rows = _comparison_rows(comparison)
    supplemental_dir = output_dir / "supplemental_selected_prefix_qiskit"
    fm_rows, fm_campaign = _fm_rows_by_regime(
        campaign_root=fm_campaign_root.resolve(),
        supplemental_dir=supplemental_dir,
    )
    jr_l10_rows, jr_l10_campaign = _jr_l10_rows_by_regime(output_dir)
    if jr_chtc_live_snapshot_manifest is None:
        jr_chtc_live_rows: dict[str, dict[str, Any]] = {}
        jr_chtc_live_campaign = None
    else:
        jr_chtc_live_rows, jr_chtc_live_campaign = (
            _load_jr_chtc_live_snapshot_manifest(
                jr_chtc_live_snapshot_manifest
            )
        )
    fm_ablation_rows, fm_ablation_campaign = _fm_ablation_rows(
        campaign_root=fm_campaign_root.resolve(),
        supplemental_dir=supplemental_dir,
    )
    route4_rows, route4_campaign = _route4_rows_by_regime(output_dir)
    if paper_i_route4_live_snapshot_manifest is None:
        route4_live_rows: dict[str, dict[str, Any]] = {}
        route4_live_campaign = None
    else:
        route4_live_rows, route4_live_campaign = (
            _load_paper_i_route4_live_snapshot_manifest(
                paper_i_route4_live_snapshot_manifest
            )
        )
    sr_expanded_chart_whitening_campaign = (
        None
        if sr_expanded_chart_whitening_validation_json is None
        else _load_sr_expanded_chart_whitening_validation(
            sr_expanded_chart_whitening_validation_json
        )
    )
    sr_expanded_chart_whitening_intermediate_weak_campaign = (
        None
        if sr_expanded_chart_whitening_intermediate_weak_validation_json is None
        else _load_sr_expanded_chart_whitening_intermediate_weak_validation(
            sr_expanded_chart_whitening_intermediate_weak_validation_json,
            qiskit_sidecar_path=(
                sr_expanded_chart_whitening_intermediate_weak_qiskit_json
            ),
        )
    )
    if (
        sr_expanded_chart_whitening_intermediate_weak_validation_json is None
        and sr_expanded_chart_whitening_intermediate_weak_qiskit_json is not None
    ):
        raise ValueError(
            "SR intermediate-weak Qiskit sidecar requires its validation JSON"
        )
    if fm_live_snapshot_manifest is None:
        fm_live_rows = {regime: {} for regime in REGIME_ORDER}
        fm_live_campaign = None
    else:
        fm_live_rows, fm_live_campaign = _load_fm_live_snapshot_manifest(
            fm_live_snapshot_manifest
        )
    if fm_live_status_snapshot is None:
        fm_live_status_campaign = None
    else:
        if fm_live_campaign is None:
            raise ValueError("FM live status requires the prior live-snapshot manifest")
        fm_live_status_campaign = _load_fm_live_status_snapshot(
            fm_live_status_snapshot,
            live_rows=fm_live_rows,
            live_campaign=fm_live_campaign,
        )
    if fm_completed_resource_recovery_manifest is None:
        fm_completed_resource_recovery_campaign = None
    else:
        if fm_live_campaign is None:
            raise ValueError(
                "FM completed-resource recovery requires the prior live manifest"
            )
        fm_completed_resource_recovery_campaign = (
            _overlay_fm_completed_resource_recovery(
                fm_completed_resource_recovery_manifest,
                live_rows=fm_live_rows,
                live_campaign=fm_live_campaign,
            )
        )
    if fm_stopped_snapshot_manifest is None:
        fm_stopped_snapshot_campaign = None
    else:
        fm_stopped_rows, fm_stopped_snapshot_campaign = (
            _load_fm_stopped_snapshot_manifest(fm_stopped_snapshot_manifest)
        )
        _compile_fm_stopped_snapshot_resources(
            fm_stopped_rows,
            supplemental_dir=supplemental_dir,
        )
        _overlay_fm_stopped_snapshot_rows(fm_live_rows, fm_stopped_rows)
        fm_stopped_snapshot_campaign["resource_fields"] = (
            "validated_report_qiskit_N2q_D2q_Dc; S_unavailable"
        )
    repaired_l25 = _repaired_l25_evidence(
        output_dir,
        prior_report_json=prior_report_json,
    )
    repaired_l25_rows = {
        str(row["regime"]): row
        for row in repaired_l25["regimes"]
    }
    model_regimes: list[dict[str, Any]] = []
    jr_policy_regimes: list[dict[str, Any]] = []
    fm_policy_regimes: list[dict[str, Any]] = []
    route4_regimes: list[dict[str, Any]] = []
    original_route_regimes: list[dict[str, Any]] = []
    for regime in REGIME_ORDER:
        campaign_dir = CAMPAIGN_DIR[regime]
        current_result = CAMPAIGN_ROOT / campaign_dir / "result.json"
        current_resource = next(
            row for row in resource_rows[regime] if row["method"] == "joint_response_snake"
        )
        paper_curves = {
            method: _paper_curve(references[(regime, method)])
            for method in PAPER_METHODS
        }
        paper_resources = [
            next(row for row in resource_rows[regime] if row["method"] == method)
            for method in PAPER_METHODS
        ]
        jr_selected = jr_l10_rows[regime]
        jr_chtc_live = jr_chtc_live_rows.get(regime)
        fm = fm_rows[regime]
        route4 = route4_rows[regime]
        route4_live = route4_live_rows.get(regime)
        model_curves = {
            "jr_selected": _curve_payload(jr_selected["curve"]),
            **{
                method: _curve_payload(curve)
                for method, curve in paper_curves.items()
            },
        }
        if route4["curve"] is not None:
            model_curves["paper_i_route4"] = _curve_payload(route4["curve"])
        if route4_live is not None:
            model_curves["paper_i_route4_live"] = _curve_payload(
                route4_live["curve"]
            )
        if jr_chtc_live is not None:
            model_curves["jr_chtc_live"] = _curve_payload(
                jr_chtc_live["curve"]
            )
        intermediate_weak_validated_resource: dict[str, Any] | None = None
        if (
            regime == "intermediate-weak"
            and sr_expanded_chart_whitening_intermediate_weak_campaign is not None
        ):
            iw_campaign = sr_expanded_chart_whitening_intermediate_weak_campaign
            model_curves[SR_EXPANDED_CHART_WHITENING_IW_ROLE] = {
                "role": SR_EXPANDED_CHART_WHITENING_IW_ROLE,
                "points": list(iw_campaign["trajectory_points"]),
                "marker_k": int(iw_campaign["route"]["outer_round_horizon"]),
                "marker_error": float(
                    iw_campaign["result"]["displayed_absolute_error"]
                ),
                "source_json": iw_campaign["validation_json"],
                "source_sha256": iw_campaign["validation_sha256"],
                "validated_terminal_marker": True,
                "preterminal_checkpoint_error": float(
                    iw_campaign["result"][
                        "pre_terminal_checkpoint_replayed_absolute_error"
                    ]
                ),
            }
            qiskit = iw_campaign.get("qiskit")
            if isinstance(qiskit, Mapping):
                intermediate_weak_validated_resource = {
                    "method": SR_EXPANDED_CHART_WHITENING_IW_ROLE,
                    "method_display": RESOURCE_METHOD_DISPLAY[
                        SR_EXPANDED_CHART_WHITENING_IW_ROLE
                    ],
                    "role": SR_EXPANDED_CHART_WHITENING_IW_ROLE,
                    "status": "complete",
                    "resource_status": "validated_terminal_diagnostic",
                    "k_pl": int(iw_campaign["route"]["outer_round_horizon"]),
                    "active_ansatz_depth": int(
                        iw_campaign["result"]["finalized_active_depth"]
                    ),
                    "abs_delta_e": float(
                        iw_campaign["result"]["displayed_absolute_error"]
                    ),
                    "N2q": int(qiskit["N2q"]),
                    "D2q": int(qiskit["D2q"]),
                    "Dc": int(qiskit["Dc"]),
                    "S": int(
                        iw_campaign["estimator_accounting"]["winning_lineage"][
                            "S_alg"
                        ]
                    ),
                    "source_json": iw_campaign["validation_json"],
                    "source_sha256": iw_campaign["validation_sha256"],
                    "qiskit_json": qiskit["path"],
                    "qiskit_sha256": qiskit["sha256"],
                    "diagnostic_only": True,
                }
        live_off = fm_live_rows[regime].get("fm_qbroyd_off")
        fm_model_curves, fm_model_resources = _merge_fm_model_live_evidence(
            fm, live_off
        )
        model_curves.update(fm_model_curves)
        model_regimes.append(
            {
                "regime": regime,
                "display": REGIME_DISPLAY[regime],
                "curves": model_curves,
                "resource_table_rows": [
                    *(
                        []
                        if intermediate_weak_validated_resource is None
                        else [intermediate_weak_validated_resource]
                    ),
                    *(
                        []
                        if route4_live is None
                        else [dict(route4_live["resource"])]
                    ),
                    dict(route4["resource"]),
                    *(
                        []
                        if jr_chtc_live is None
                        else [dict(jr_chtc_live["resource"])]
                    ),
                    dict(jr_selected["resource"]),
                    *fm_model_resources,
                    *paper_resources,
                ],
            }
        )

        jr_curves = {
            "jr_selected": _curve_payload(jr_selected["curve"]),
            "jr_baseline": _curve_payload(
                _history_curve(
                    current_result,
                    role="jr_baseline",
                    marker_k=int(current_resource["k_pl"]),
                    marker_error=float(current_resource["abs_delta_e"]),
                )
            ),
        }
        jr_resources = [dict(jr_selected["resource"]), dict(current_resource)]
        jr_resources[-1]["method_display"] = "JR-L15 baseline"
        if jr_chtc_live is not None:
            jr_curves["jr_chtc_live"] = _curve_payload(
                jr_chtc_live["curve"]
            )
            jr_resources.insert(0, dict(jr_chtc_live["resource"]))
        comparison_row = comparison_rows[regime]
        prior = comparison_row.get("prior_ledger_row") or {}
        prior_source = str(prior.get("source_json") or "")
        if prior_source:
            prior_path = _repo_path(prior_source)
            jr_curves["jr_prior"] = _curve_payload(
                _history_curve(prior_path, role="jr_prior")
            )
            jr_resources.append(
                _supplemental_resource_row(
                    regime=regime,
                    method="prior_ledger_snake",
                    source_json=prior_path,
                    history_position=int(prior["controller_round_count"]),
                    expected_error=float(prior["abs_delta_e"]),
                    sidecar_json=supplemental_dir / f"prior-{regime}.json",
                )
            )
        repaired_row = repaired_l25_rows[regime]
        jr_curves["jr_l25"] = dict(repaired_row["curves"]["repaired_l25"])
        jr_curves["jr_l25"]["role"] = "jr_l25"
        jr_resources.append(dict(repaired_row["resource_table_rows"][0]))
        if regime == "weak-weak":
            wave11_error = 1.1883371849874536e-6
            wave11_position = 13
            jr_curves["jr_early_l25"] = _curve_payload(
                _history_curve(
                    WAVE11_WEAK_WEAK_JSON,
                    role="jr_early_l25",
                    marker_k=wave11_position,
                    marker_error=wave11_error,
                )
            )
            jr_resources.append(
                _supplemental_resource_row(
                    regime=regime,
                    method="weak_weak_wave11_l25",
                    source_json=WAVE11_WEAK_WEAK_JSON,
                    history_position=wave11_position,
                    expected_error=wave11_error,
                    sidecar_json=supplemental_dir / "weak-weak-wave11-l25-r13.json",
                )
            )
        jr_policy_regimes.append(
            {
                "regime": regime,
                "display": REGIME_DISPLAY[regime],
                "curves": jr_curves,
                "resource_table_rows": jr_resources,
            }
        )

        fm_variant = fm_ablation_rows[regime]
        fm_variant_curves, fm_variant_resources = _merge_fm_policy_live_evidence(
            fm_variant, fm_live_rows[regime]
        )
        fm_policy_regimes.append(
            {
                "regime": regime,
                "display": REGIME_DISPLAY[regime],
                "curves": fm_variant_curves,
                "resource_table_rows": fm_variant_resources,
            }
        )
        route4_regimes.append(
            {
                "regime": regime,
                "display": REGIME_DISPLAY[regime],
                "curves": {
                    **(
                        {}
                        if route4_live is None
                        else {
                            "paper_i_route4_live": _curve_payload(
                                route4_live["curve"]
                            )
                        }
                    ),
                    **(
                        {}
                        if route4["curve"] is None
                        else {"paper_i_route4": _curve_payload(route4["curve"])}
                    ),
                    "snake": _curve_payload(paper_curves["snake"]),
                },
                "resource_table_rows": [
                    *(
                        []
                        if route4_live is None
                        else [dict(route4_live["resource"])]
                    ),
                    dict(route4["resource"]),
                    dict(next(row for row in paper_resources if row["method"] == "snake")),
                ],
            }
        )
        original_route_regimes.append(
            {
                "regime": regime,
                "display": REGIME_DISPLAY[regime],
                "curves": {
                    method: _curve_payload(curve)
                    for method, curve in paper_curves.items()
                },
                "resource_table_rows": paper_resources,
            }
        )
    route4_recovery_note = _sr_recovery_summary(route4_live_campaign)
    pages = {
        "model": {
            "title": "Model comparison: SR-SNAKE, selected JR/FM, and historical baselines",
            "subtitle": (
                "SR-SNAKE preserves 3 complete, 2 round-21 partial, and 1 not-run baseline row; "
                + route4_recovery_note
                + "; "
                "JR uses transferable L10/B2; FM lines preserve the atomic snapshot and "
                "open markers show the latest status endpoints; starred checkpoints are science-complete/package-failed"
                if route4_live_campaign is not None and fm_live_status_campaign is not None
                else "SR-SNAKE preserves 3 complete, 2 round-21 partial, and 1 not-run baseline row; "
                + route4_recovery_note
                + "; "
                "JR uses transferable L10/B2; FM uses qB-off nonterminal snapshots where supplied"
                if route4_live_campaign is not None
                else "SR-SNAKE has 3 complete, 2 round-21 partial, 1 not run; "
                "JR uses transferable L10/B2; FM lines preserve the atomic snapshot and "
                "open markers show the latest status endpoints; starred checkpoints are science-complete/package-failed"
                if fm_live_status_campaign is not None
                else "SR-SNAKE has 3 complete, 2 round-21 partial, 1 not run; "
                "JR uses transferable L10/B2; FM uses qB-off nonterminal snapshots "
                "where supplied, with prior terminal resource evidence retained separately"
                if fm_live_campaign is not None
                else "SR-SNAKE has 3 complete, 2 round-21 partial, 1 not run; JR uses transferable L10/B2; FM pending cells remain explicit"
            ),
            "roles": (
                ["paper_i_route4_live", "paper_i_route4", "jr_selected", "fm_qbroyd_off", "fm_qbroyd_off_prior", "snake", "geo", "append"]
                if route4_live_campaign is not None and fm_live_campaign is not None
                else ["paper_i_route4_live", "paper_i_route4", "jr_selected", "fm_qbroyd_off", "snake", "geo", "append"]
                if route4_live_campaign is not None
                else ["paper_i_route4", "jr_selected", "fm_qbroyd_off", "fm_qbroyd_off_prior", "snake", "geo", "append"]
                if fm_live_campaign is not None
                else ["paper_i_route4", "jr_selected", "fm_qbroyd_off", "snake", "geo", "append"]
            ),
            "regimes": model_regimes,
        },
        "jr_policies": {
            "title": "JR-SNAKE policy comparison",
            "subtitle": "Selected L10 versus L15 baseline, prior ledger, repaired L25, and the early weak-weak L25 diagnostic",
            "roles": ["jr_selected", "jr_baseline", "jr_prior", "jr_l25", "jr_early_l25"],
            "regimes": jr_policy_regimes,
        },
        "fm_policies": {
            "title": "FM-SNAKE policy comparison",
            "subtitle": (
                "Matched qB on/off diagnostic; lines are the atomic snapshot, open markers are "
                f"{' '.join(str(fm_live_status_campaign['captured_at_local']).split()[-2:])} endpoints; "
                "IW markers restart in cluster 8776378; starred checkpoints are science-complete/package-failed; no source-value anchor"
                if fm_live_status_campaign is not None
                else "Matched within-batch qB on/off nonterminal diagnostic; no source-value anchor; "
                "prior terminal rows are retained separately"
                if fm_live_campaign is not None
                else "Inverse-RBFGS formal-manifold route with qB on versus qB disabled; unfinished regimes are labeled"
            ),
            "roles": (
                ["fm_qbroyd_default", "fm_qbroyd_off", "fm_qbroyd_on_prior", "fm_qbroyd_off_prior"]
                if fm_live_campaign is not None
                else ["fm_qbroyd_default", "fm_qbroyd_off"]
            ),
            "regimes": fm_policy_regimes,
        },
        "paper_i_route4": {
            "title": "SR-SNAKE: supported-metric whitening plus adaptive trust",
            "subtitle": (
                "Exact-Hessian Schur nomination, recoverability-ladder prune, singleton children, "
                "no Phase 0, no batching; historical Paper-I SNAKE is retained for context. "
                "Preserved SR baseline: 3 complete, 2 round-21 partial, 1 not run; "
                + route4_recovery_note
                if route4_live_campaign is not None
                else "Exact-Hessian Schur nomination, recoverability-ladder prune, singleton children, "
                "no Phase 0, no batching; historical Paper-I SNAKE is retained for context: "
                "3 complete; 2 preserved round-21 partial; 1 not run"
            ),
            "roles": (
                ["paper_i_route4_live", "paper_i_route4", "snake"]
                if route4_live_campaign is not None
                else ["paper_i_route4", "snake"]
            ),
            "regimes": route4_regimes,
        },
        "original_route": {
            "title": "Original Paper-I Phase 1-3 SNAKE route",
            "subtitle": "Locked no-batching SNAKE trajectory with beam/prune and matched Geo/Append comparators; no separate internal-ablation bundle is ingested yet",
            "roles": ["snake", "geo", "append"],
            "regimes": original_route_regimes,
        },
    }
    if sr_expanded_chart_whitening_campaign is not None:
        pages[SR_EXPANDED_CHART_WHITENING_PAGE_KEY] = {
            "page_type": SR_EXPANDED_CHART_WHITENING_PAGE_KEY,
            "title": "Weak--weak SR-SNAKE: expanded-chart accepted-refit whitening",
            "subtitle": (
                "Fixed controller checkpoints versus separately disclosed terminal "
                "full-refit/prune work; no manuscript result is replaced"
            ),
            "campaign": sr_expanded_chart_whitening_campaign,
        }
    if sr_expanded_chart_whitening_intermediate_weak_campaign is not None:
        pages[SR_EXPANDED_CHART_WHITENING_IW_PAGE_KEY] = {
            "page_type": SR_EXPANDED_CHART_WHITENING_IW_PAGE_KEY,
            "title": (
                "Intermediate--weak SR-SNAKE: expanded-chart accepted-refit whitening"
            ),
            "subtitle": (
                "Validated r30 diagnostic; checkpoint and finalized errors remain "
                "separate, and no selected model row or Qiskit resource is replaced"
            ),
            "campaign": sr_expanded_chart_whitening_intermediate_weak_campaign,
        }
        pages["model"]["roles"].insert(
            0, SR_EXPANDED_CHART_WHITENING_IW_ROLE
        )
        pages["model"]["subtitle"] += (
            "; dark-plum dashed intermediate-weak curve is an additive validated "
            "expanded-chart diagnostic; the selected SR row and resources are unchanged"
        )
    if fm_completed_resource_recovery_campaign is not None:
        pages["model"]["subtitle"] += (
            "; FM weak-Holstein qB-off rows use recovered exact terminal "
            "Paper-I Qiskit costs and winning-lineage S"
        )
        pages["fm_policies"]["subtitle"] += (
            "; weak-Holstein terminal summaries recover six S totals and five "
            "validated Qiskit rows; strong-weak qB-on lacks its terminal operator "
            "list, so * marks its last verified k/error and dagger marks terminal S"
        )
    if fm_stopped_snapshot_campaign is not None:
        pages["model"]["subtitle"] += (
            "; FM strong-Holstein proc6--11 lines are full user-stopped "
            "completed-round checkpoints with Paper-I Qiskit prefix costs and S unavailable"
        )
        pages["fm_policies"]["subtitle"] += (
            "; proc6--11 use full stopped-checkpoint trajectories, not stale status markers"
        )
    if jr_chtc_live_campaign is not None:
        model_roles = pages["model"]["roles"]
        model_roles.insert(model_roles.index("jr_selected"), "jr_chtc_live")
        pages["model"]["subtitle"] += (
            "; dark-teal dashed JR lines are immutable CHTC snapshots with "
            "per-entry fixed-prefix Qiskit/S where hash-validated sidecars are supplied"
        )
        pages["jr_policies"]["roles"].insert(0, "jr_chtc_live")
        pages["jr_policies"]["subtitle"] = (
            "Rollback-free CHTC L10 prefixes versus selected L10, L15 baseline, "
            "prior ledger, repaired L25, and the early weak-weak L25 diagnostic"
        )
    return {
        "schema": SCHEMA,
        "comparison_json": _rel(comparison_path),
        "comparison_sha256": _sha256(comparison_path),
        "paper_i_reference_json": _rel(reference_path),
        "paper_i_reference_sha256": _sha256(reference_path),
        "selected_prefix_resource_json": _rel(resource_path),
        "selected_prefix_resource_sha256": _sha256(resource_path),
        "supplemental_resource_dir": _rel(supplemental_dir),
        "qiskit_contract": {
            "compile_convention": resource_payload.get("qiskit_compile_convention"),
            "S_convention": resource_payload.get("S_convention"),
            "metrics": ["N2q", "D2q", "Dc", "S"],
            "validation": dict(resource_payload.get("validation") or {}),
        },
        "formal_manifold_campaign": fm_campaign,
        "formal_manifold_ablation_campaign": fm_ablation_campaign,
        "formal_manifold_live_snapshot_campaign": fm_live_campaign,
        "formal_manifold_live_status_campaign": fm_live_status_campaign,
        "formal_manifold_completed_resource_recovery_campaign": (
            fm_completed_resource_recovery_campaign
        ),
        "formal_manifold_stopped_snapshot_campaign": fm_stopped_snapshot_campaign,
        "paper_i_route4_campaign": route4_campaign,
        "paper_i_route4_live_snapshot_campaign": route4_live_campaign,
        "sr_expanded_chart_whitening_campaign": (
            sr_expanded_chart_whitening_campaign
        ),
        "sr_expanded_chart_whitening_intermediate_weak_campaign": (
            sr_expanded_chart_whitening_intermediate_weak_campaign
        ),
        "jr_l10_campaign": jr_l10_campaign,
        "jr_chtc_live_snapshot_campaign": jr_chtc_live_campaign,
        "repaired_l25_campaign": {
            key: repaired_l25[key]
            for key in (
                "schema",
                "status",
                "cluster",
                "scientific_contract_hash",
                "execution_profile",
                "policy",
            )
        }
        | (
            {"source_recovery": repaired_l25["source_recovery"]}
            if repaired_l25.get("source_recovery") is not None
            else {}
        ),
        "status": (
            "partial_new_paper_i_route4_with_live_jr_chtc_snapshots"
            if jr_chtc_live_campaign is not None
            else
            "partial_new_paper_i_route4_with_live_nonterminal_checkpoints"
            if route4_live_campaign is not None
            else
            "partial_new_paper_i_route4_and_live_formal_manifold_nonterminal"
            if fm_live_campaign is not None
            else (
                "partial_new_paper_i_route4"
                if not fm_campaign["pending_regimes"]
                else "partial_new_paper_i_route4_and_pending_formal_manifold"
            )
        ),
        "run_setting_caveat_ledger": [dict(row) for row in RUN_SETTING_LEDGER],
        "error_contract": "same-cutoff absolute energy error versus ADAPT controller round",
        "pages": pages,
        "regimes": model_regimes,
    }


def _resource_status_label(status: str, row: Mapping[str, Any] | None = None) -> str:
    explicit_label = str((row or {}).get("status_label") or "").strip()
    if explicit_label:
        return explicit_label
    if status == "running_status_endpoint":
        capture_label = str((row or {}).get("status_capture_label") or "latest")
        return f"run@{capture_label}"
    if status == PAPER_I_SR_TERMINAL_STATUS:
        return "validated recovery"
    return {
        "running_snapshot": "running",
        "held_snapshot": "held",
        "recovery_queued_snapshot": "recovery queued",
        "completed_snapshot_pending_qiskit": "done",
        PAPER_I_ROUTE4_LIVE_STATUS: "nonterminal",
        PAPER_I_ROUTE4_STOPPED_STATUS: "stopped nonterminal",
        "failed_partial": "failed partial",
        "science_complete_packaging_failed": "done/pkg fail*",
        "stopped_snapshot": "stopped fixed-prefix",
        "failed_partial_round21": "partial r21",
        "not_run": "not run",
    }.get(status, status.replace("_", " "))


def _has_compiled_resource_values(row: Mapping[str, Any]) -> bool:
    return all(row.get(key) is not None for key in ("k_pl", "abs_delta_e", "N2q", "D2q", "Dc"))


def _plot_resource_table(ax: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    headers = ("Method", r"$k_{\rm pl}$", r"$|\Delta E|$", r"$N_{2q}$", r"$D_{2q}$", r"$D_c$", r"$S$")
    cell_text = []
    for row in rows:
        method = str(row["method"])
        status = str(row.get("status") or "complete")
        display = RESOURCE_METHOD_DISPLAY.get(
            method, str(row.get("method_display") or method)
        )
        if status != "complete" and _has_compiled_resource_values(row):
            s_value = row.get("S")
            cell_text.append(
                [
                    f"{display} [{_resource_status_label(status, row)}]",
                    f"{int(row['k_pl'])}",
                    f"{float(row['abs_delta_e']):.2e}",
                    f"{int(row['N2q']):,}",
                    f"{int(row['D2q']):,}",
                    f"{int(row['Dc']):,}",
                    "n/a" if s_value is None else f"{int(s_value):,}",
                ]
            )
            continue
        if (
            row.get("resource_status")
            == "validated_terminal_query_work_qiskit_operator_sequence_unavailable"
        ):
            cell_text.append(
                [
                    f"{display} [S recovered; Qiskit unavailable]",
                    f"{int(row['k_pl'])}*",
                    f"{float(row['abs_delta_e']):.2e}*",
                    "unavail.",
                    "unavail.",
                    "unavail.",
                    f"{int(row['S']):,}†",
                ]
            )
            continue
        if status in PENDING_RESOURCE_STATUSES:
            asterisk = "*" if row.get("checkpoint_asterisk") is True else ""
            cell_text.append(
                [
                    f"{display} [{_resource_status_label(status, row)}]",
                    f"{int(row['k_pl'])}{asterisk}",
                    f"{float(row['abs_delta_e']):.2e}{asterisk}",
                    "pending",
                    "pending",
                    "pending",
                    "pending",
                ]
            )
            continue
        if status != "complete":
            cell_text.append(
                [
                    display,
                    status,
                    "--",
                    "--",
                    "--",
                    "--",
                    "--",
                ]
            )
            continue
        s_value = row.get("S")
        cell_text.append(
            [
                display,
                f"{int(row['k_pl'])}",
                f"{float(row['abs_delta_e']):.2e}",
                f"{int(row['N2q']):,}",
                f"{int(row['D2q']):,}",
                f"{int(row['Dc']):,}",
                "n/a" if s_value is None else f"{int(s_value):,}",
            ]
        )
    ax.set_axis_off()
    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        cellLoc="right",
        colLoc="right",
        colWidths=(0.29, 0.08, 0.15, 0.11, 0.11, 0.11, 0.15),
        bbox=(0.0, 0.01, 1.0, 0.97),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(3.9 if len(rows) >= 8 else 4.15 if len(rows) >= 7 else 4.4)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.visible_edges = "horizontal"
        cell.set_edgecolor("#A8A8A8")
        cell.set_linewidth(0.28)
        cell.PAD = 0.025
        if row_index == 0:
            cell.set_text_props(weight="semibold", color="#222222")
            cell.set_facecolor("#F7F7F7")
        elif column_index == 0:
            method = str(rows[row_index - 1]["method"])
            role = RESOURCE_METHOD_STYLE.get(method)
            color = STYLE.get(role or "", {}).get("color", "#222222")
            cell.set_text_props(ha="left", color=color, weight="medium")
        else:
            cell.set_text_props(ha="right", color="#222222")


def _plot(evidence: Mapping[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(13.6, 8.45), constrained_layout=False)
    outer = fig.add_gridspec(2, 3, left=0.055, right=0.992, bottom=0.055, top=0.91, wspace=0.19, hspace=0.28)
    for index, regime_row in enumerate(evidence["regimes"]):
        inner = outer[index // 3, index % 3].subgridspec(2, 1, height_ratios=(2.35, 1.25), hspace=0.035)
        error_ax = fig.add_subplot(inner[0])
        table_ax = fig.add_subplot(inner[1])
        curves = regime_row["curves"]
        for role in ("append", "geo", "snake", "early_l25", "prior", "manifold", "current", "l10_live"):
            if role not in curves:
                continue
            curve = curves[role]
            style = STYLE[role]
            error_ax.plot(
                [point["k"] for point in curve["points"]],
                [point["error"] for point in curve["points"]],
                color=style["color"],
                linewidth=style["width"],
                linestyle="-",
                alpha=0.96,
                zorder=4 if role == "l10_live" else 2 if role not in {"current", "prior"} else 3,
            )
            error_ax.scatter(
                [curve["marker_k"]],
                [curve["marker_error"]],
                color=style["color"],
                marker=style["marker"],
                s=34 if role != "snake" else 52,
                edgecolor="white",
                linewidth=0.35,
                zorder=5,
            )
        error_ax.set_yscale("log")
        error_ax.set_xlim(left=0)
        error_ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=6))
        error_ax.grid(True, which="major", alpha=0.2, linewidth=0.45)
        error_ax.tick_params(axis="both", labelsize=6.4)
        error_ax.set_ylabel(r"$|\Delta E|$", fontsize=7.2)
        error_ax.set_title(str(regime_row["display"]), fontsize=8.4, pad=2.5)
        _plot_resource_table(table_ax, regime_row["resource_table_rows"])

    legend_order = ("l10_live", "current", "prior", "early_l25", "manifold", "snake", "geo", "append")
    present_roles = {
        role
        for regime_row in evidence["regimes"]
        for role in regime_row["curves"]
    }
    handles = [
        Line2D(
            [0], [0], color=STYLE[role]["color"], linewidth=STYLE[role]["width"],
            marker=STYLE[role]["marker"], markersize=5.2, label=STYLE[role]["label"]
        )
        for role in legend_order
        if role in present_roles
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        fontsize=6.25,
        bbox_to_anchor=(0.52, 0.985),
    )
    fig.text(
        0.5,
        0.018,
        "ADAPT controller round; tables report Paper-I-convention compiled selected prefixes/endpoints and winning-branch S; FM pending rows are explicit",
        ha="center",
        fontsize=6.7,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_regime_panel(
    regime_row: Mapping[str, Any],
    *,
    roles: Sequence[str],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.42, 1.92), constrained_layout=True)
    curves = regime_row["curves"]
    plotted = False
    for role in roles:
        curve = curves.get(role)
        if not isinstance(curve, Mapping):
            continue
        style = STYLE[role]
        ax.plot(
            [point["k"] for point in curve["points"]],
            [point["error"] for point in curve["points"]],
            color=style["color"],
            linewidth=style["width"],
            linestyle=str(style.get("linestyle", "-")),
            alpha=0.97,
        )
        open_marker = bool(
            curve.get("status_endpoint_only") or curve.get("live_snapshot")
        )
        ax.scatter(
            [curve["marker_k"]],
            [curve["marker_error"]],
            color=("white" if open_marker else style["color"]),
            marker=style["marker"],
            s=(43 if open_marker else 34) if role != "snake" else 48,
            edgecolor=(style["color"] if open_marker else "white"),
            linewidth=1.15 if open_marker else 0.35,
            zorder=5,
        )
        plotted = True
    if plotted:
        ax.set_yscale("log")
        ax.set_xlim(left=0)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=6))
        ax.grid(True, which="major", alpha=0.2, linewidth=0.45)
    else:
        ax.text(
            0.5,
            0.5,
            "Pending evidence",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="#666666",
        )
        ax.set_xticks([])
        ax.set_yticks([])
    ax.tick_params(axis="both", labelsize=6.2)
    ax.set_xlabel("controller round", fontsize=6.6, labelpad=1)
    ax.set_ylabel(r"$|\Delta E|$", fontsize=6.8, labelpad=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, facecolor="white")
    plt.close(fig)


def _plot_page_panels(
    page_key: str,
    page: Mapping[str, Any],
    *,
    output_dir: Path,
    stem: str,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for regime_row in page["regimes"]:
        regime = str(regime_row["regime"])
        path = output_dir / f"{stem}-{page_key}-{regime}.png"
        _plot_regime_panel(regime_row, roles=page["roles"], output_path=path)
        paths[regime] = path
    return paths


def _plot_sr_expanded_chart_whitening_page(
    page: Mapping[str, Any],
    *,
    output_path: Path,
) -> None:
    """Plot the weak--weak chart comparison and per-admission metric support."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    campaign = page["campaign"]
    comparisons = campaign["comparisons"]
    styles = {
        "historical_high_accuracy_sr_baseline": {
            "label": "historical SR baseline",
            "color": "#505050",
            "linestyle": "--",
            "marker": "o",
        },
        "wrong_reduced_chart_whitened_r22": {
            "label": "reduced logical chart (r22)",
            "color": "#D97706",
            "linestyle": "-.",
            "marker": "s",
        },
        "good_expanded_chart_whitened_r22": {
            "label": "expanded/projected chart (r22)",
            "color": "#7C3AED",
            "linestyle": ":",
            "marker": "D",
        },
        "good_expanded_chart_whitened_r30": {
            "label": "expanded/projected chart (r30)",
            "color": "#0369A1",
            "linestyle": "-",
            "marker": "*",
        },
    }
    fig, (error_ax, rank_ax) = plt.subplots(
        1,
        2,
        figsize=(10.4, 3.25),
        gridspec_kw={"width_ratios": (1.72, 1.0)},
        constrained_layout=True,
    )
    for comparison in comparisons:
        style = styles[str(comparison["label"])]
        rounds = [int(point["k"]) for point in comparison["points"]]
        errors = [float(point["error"]) for point in comparison["points"]]
        error_ax.plot(
            rounds,
            errors,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.65,
            alpha=0.95,
            label=style["label"],
        )
        error_ax.scatter(
            [int(comparison["rounds"])],
            [float(comparison["preterminal_checkpoint_error"])],
            color="white",
            edgecolor=style["color"],
            marker=style["marker"],
            linewidth=1.15,
            s=47,
            zorder=5,
        )
        error_ax.scatter(
            [int(comparison["rounds"])],
            [float(comparison["finalized_error"])],
            color=style["color"],
            edgecolor="white",
            marker="P",
            linewidth=0.45,
            s=58,
            zorder=6,
        )
    error_ax.set_yscale("log")
    error_ax.set_xlim(left=0)
    error_ax.grid(True, which="major", linewidth=0.45, alpha=0.22)
    error_ax.set_xlabel("ADAPT controller round", fontsize=8)
    error_ax.set_ylabel(r"same-cutoff $|\Delta E|$", fontsize=8)
    error_ax.set_title(
        "Open marker: last controller checkpoint; filled plus: finalized result",
        fontsize=8.1,
        pad=3,
    )
    error_ax.tick_params(axis="both", labelsize=7)
    error_ax.legend(frameon=False, fontsize=6.7, loc="lower left")

    rounds = list(range(1, len(campaign["support_rank_sequence"]) + 1))
    rank_ax.plot(
        rounds,
        campaign["expanded_runtime_dimension_sequence"],
        color="#9CA3AF",
        linewidth=1.2,
        linestyle=":",
        label="expanded runtime dimension",
    )
    rank_ax.plot(
        rounds,
        campaign["logical_dimension_sequence"],
        color="#7C3AED",
        linewidth=1.35,
        linestyle="--",
        label="projected logical dimension",
    )
    rank_ax.plot(
        rounds,
        campaign["support_rank_sequence"],
        color="#0369A1",
        linewidth=1.8,
        marker="o",
        markersize=2.7,
        label="retained FS support rank",
    )
    rank_ax.set_xlim(left=1)
    rank_ax.set_ylim(bottom=0)
    rank_ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=7))
    rank_ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=7))
    rank_ax.grid(True, which="major", linewidth=0.45, alpha=0.22)
    rank_ax.set_xlabel("accepted admission / refit", fontsize=8)
    rank_ax.set_ylabel("coordinate count", fontsize=8)
    rank_ax.set_title(
        "Gram support is rebuilt after each admission",
        fontsize=8.1,
        pad=3,
    )
    rank_ax.tick_params(axis="both", labelsize=7)
    rank_ax.legend(frameon=False, fontsize=6.7, loc="upper left")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, facecolor="white")
    plt.close(fig)


def _plot_sr_expanded_chart_whitening_intermediate_weak_page(
    page: Mapping[str, Any],
    *,
    output_path: Path,
) -> None:
    """Plot the validated IW trajectory and its accepted-refit dimensions."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    campaign = page["campaign"]
    result = campaign["result"]
    trajectory = campaign["trajectory_points"]
    rounds = [int(point["k"]) for point in trajectory]
    errors = [float(point["error"]) for point in trajectory]
    horizon = int(campaign["route"]["outer_round_horizon"])
    color = STYLE[SR_EXPANDED_CHART_WHITENING_IW_ROLE]["color"]
    fig, (error_ax, dimension_ax) = plt.subplots(
        1,
        2,
        figsize=(10.4, 3.25),
        gridspec_kw={"width_ratios": (1.72, 1.0)},
        constrained_layout=True,
    )
    error_ax.plot(
        rounds,
        errors,
        color=color,
        linewidth=1.9,
        linestyle="-",
        label="r30 accepted-controller trajectory",
    )
    error_ax.scatter(
        [horizon],
        [float(result["pre_terminal_checkpoint_replayed_absolute_error"])],
        color="white",
        edgecolor=color,
        marker="8",
        linewidth=1.25,
        s=54,
        zorder=5,
        label="round-30 checkpoint",
    )
    error_ax.scatter(
        [horizon],
        [float(result["displayed_absolute_error"])],
        color=color,
        edgecolor="white",
        marker="P",
        linewidth=0.45,
        s=62,
        zorder=6,
        label="finalized validated result",
    )
    error_ax.set_yscale("log")
    error_ax.set_xlim(left=0)
    error_ax.grid(True, which="major", linewidth=0.45, alpha=0.22)
    error_ax.set_xlabel("ADAPT controller round", fontsize=8)
    error_ax.set_ylabel(r"same-cutoff $|\Delta E|$", fontsize=8)
    error_ax.set_title(
        "Checkpoint trajectory and separately replayed finalized endpoint",
        fontsize=8.1,
        pad=3,
    )
    error_ax.tick_params(axis="both", labelsize=7)
    error_ax.legend(frameon=False, fontsize=6.7, loc="lower left")

    accepted_rounds = list(range(1, horizon + 1))
    dimension_ax.plot(
        accepted_rounds,
        campaign["expanded_runtime_dimension_sequence"],
        color="#9CA3AF",
        linewidth=1.2,
        linestyle=":",
        label="expanded runtime dimension",
    )
    dimension_ax.plot(
        accepted_rounds,
        campaign["logical_dimension_sequence"],
        color="#7C3AED",
        linewidth=1.35,
        linestyle="--",
        label="logical dimension",
    )
    dimension_ax.plot(
        accepted_rounds,
        campaign["support_rank_sequence"],
        color="#0369A1",
        linewidth=1.65,
        marker="o",
        markersize=2.6,
        label="retained FS support rank",
    )
    dimension_ax.plot(
        accepted_rounds,
        campaign["depth_sequence"],
        color=color,
        linewidth=1.35,
        linestyle="-.",
        label="active ansatz depth",
    )
    dimension_ax.set_xlim(left=1)
    dimension_ax.set_ylim(bottom=0)
    dimension_ax.xaxis.set_major_locator(
        matplotlib.ticker.MaxNLocator(integer=True, nbins=7)
    )
    dimension_ax.yaxis.set_major_locator(
        matplotlib.ticker.MaxNLocator(integer=True, nbins=7)
    )
    dimension_ax.grid(True, which="major", linewidth=0.45, alpha=0.22)
    dimension_ax.set_xlabel("accepted controller round", fontsize=8)
    dimension_ax.set_ylabel("coordinate count / depth", fontsize=8)
    dimension_ax.set_title(
        "Expanded chart, projected logical span, and retained support",
        fontsize=8.1,
        pad=3,
    )
    dimension_ax.tick_params(axis="both", labelsize=7)
    dimension_ax.legend(frameon=False, fontsize=6.4, loc="upper left")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, facecolor="white")
    plt.close(fig)


def _plot_report_page(
    page_key: str,
    page: Mapping[str, Any],
    *,
    output_dir: Path,
    stem: str,
) -> dict[str, Path]:
    if page_key == SR_EXPANDED_CHART_WHITENING_PAGE_KEY:
        path = output_dir / f"{stem}-{page_key}-weak-weak.png"
        _plot_sr_expanded_chart_whitening_page(page, output_path=path)
        return {"wide": path}
    if page_key == SR_EXPANDED_CHART_WHITENING_IW_PAGE_KEY:
        path = output_dir / f"{stem}-{page_key}.png"
        _plot_sr_expanded_chart_whitening_intermediate_weak_page(
            page, output_path=path
        )
        return {"wide": path}
    return _plot_page_panels(
        page_key,
        page,
        output_dir=output_dir,
        stem=stem,
    )


def _latex_escape(value: str) -> str:
    replacements = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}
    return "".join(replacements.get(char, char) for char in value)


def _latex_escape_breakable_identifier(value: str) -> str:
    """Escape a path/identifier while allowing TeX line breaks at stable separators."""

    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_\allowbreak{}",
        "#": r"\#",
        "/": r"/\allowbreak{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def _latex_breakable_digest(value: str) -> str:
    """Render a hex digest exactly while allowing line breaks every 16 digits."""

    return r"\allowbreak{}".join(
        value[index : index + 16] for index in range(0, len(value), 16)
    )


def _write_tex(
    path: Path,
    *,
    image_path: Path,
    provenance_path: Path,
    evidence: Mapping[str, Any],
) -> None:
    image = image_path.resolve().as_posix()
    provenance = _rel(provenance_path)
    setting_ledger = [dict(row) for row in evidence["run_setting_caveat_ledger"]]
    source_comment = json.dumps(
        {
            "schema": SCHEMA,
            "provenance_json": provenance,
            "manuscript_edited": False,
            "repo_provenance": {
                "builder": _rel(Path(__file__)),
                "builder_sha256": _sha256(Path(__file__)),
                "comparison_json": evidence.get("comparison_json"),
                "comparison_sha256": evidence.get("comparison_sha256"),
                "selected_prefix_resource_json": evidence.get(
                    "selected_prefix_resource_json"
                ),
                "selected_prefix_resource_sha256": evidence.get(
                    "selected_prefix_resource_sha256"
                ),
                "paper_i_reference_json": evidence.get("paper_i_reference_json"),
                "paper_i_reference_sha256": evidence.get(
                    "paper_i_reference_sha256"
                ),
                "supplemental_resource_dir": evidence.get(
                    "supplemental_resource_dir"
                ),
                "formal_manifold_campaign": evidence.get(
                    "formal_manifold_campaign"
                ),
                "formal_manifold_live_snapshot_campaign": evidence.get(
                    "formal_manifold_live_snapshot_campaign"
                ),
                "formal_manifold_live_status_campaign": evidence.get(
                    "formal_manifold_live_status_campaign"
                ),
                "formal_manifold_completed_resource_recovery_campaign": evidence.get(
                    "formal_manifold_completed_resource_recovery_campaign"
                ),
                "formal_manifold_stopped_snapshot_campaign": evidence.get(
                    "formal_manifold_stopped_snapshot_campaign"
                ),
                "live_jr_l10_weak_weak": evidence.get(
                    "live_jr_l10_weak_weak"
                ),
            },
            "algorithm_structures": {
                "current_jr_snake": [
                    "macro_phase1",
                    "macro_phase2",
                    "singleton_child_expansion_and_global_dedup",
                    "child_phase1",
                    "child_phase2",
                    "joint_ansatz_plus_batch_response",
                    "Powell_refit_and_prune",
                ],
                "live_l10_jr_snake": [
                    "same_macro_to_global_child_funnel",
                    "joint_B2_L10",
                    "supported_metric_whitened_joint_solve",
                    "exact_guarded_joint_step_warm_start",
                    "adaptive_unbounded_v2_trust_radius",
                    "Powell_refit_and_prune",
                ],
                "prior_ledger_snake": "pre_phase2_joint_response_regime_dependent_route",
                "early_l25_snake": "pre_projection_wide_funnel_diagnostic",
                "paper_i_snake": "physical_operator_lanes_no_batching_visible_plateau_route",
                "formal_manifold_snake": [
                    "separate_formal_manifold_warm_start_v1_route",
                    "supported_metric_whitened_coordinates",
                    "transported_inverse_rbfgs_curvature",
                    "transactional_commit_rollback",
                    "qbroyd_shadow_off_for_transfer_candidate",
                ],
            },
            "algorithm_setting_ledger": setting_ledger,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    appendix_manifest = (
        ("Schema", str(evidence["schema"])),
        ("Campaign evidence", f"six-regime comparison; sha256={str(evidence['comparison_sha256'])[:12]}..."),
        (
            "Resource evidence",
            f"selected-prefix Qiskit table; sha256={str(evidence['selected_prefix_resource_sha256'])[:12]}...",
        ),
        ("Paper-I reference", f"locked comparison bundle; sha256={str(evidence['paper_i_reference_sha256'])[:12]}..."),
        (
            "FM campaign",
            "qB-off transfer candidate; ledger sha256="
            f"{str(evidence['formal_manifold_campaign']['ledger_sha256'])[:12]}...; "
            "pending="
            + (
                ", ".join(evidence["formal_manifold_campaign"]["pending_regimes"])
                or "none"
            ),
        ),
        (
            "Live JR L10",
            (
                "unavailable"
                if not evidence.get("live_jr_l10_weak_weak")
                else "weak-weak stitched through round "
                f"{int(evidence['live_jr_l10_weak_weak']['total_controller_rounds'])}; "
                f"status={evidence['live_jr_l10_weak_weak']['status']}; "
                "source boundary after round 17"
            ),
        ),
        (
            "FM live status",
            (
                "not supplied"
                if not evidence.get("formal_manifold_live_status_campaign")
                else "marker-only endpoints captured "
                f"{evidence['formal_manifold_live_status_campaign']['captured_at_local']}; "
                "IW replacement boundary is explicit; terminal resources pending"
            ),
        ),
        ("Machine provenance", f"{Path(provenance).name}; full paths and hashes are recorded there"),
        ("Manuscript", "Paper_I.tex not edited"),
    )
    appendix_manifest_tex = "\n".join(
        f"\\textbf{{{_latex_escape(label)}}} & {_latex_escape(value)} \\\\"
        for label, value in appendix_manifest
    )
    prefix_rows = []
    for regime_row in evidence["regimes"]:
        route_methods = [
            ("JR-SNAKE", "joint_response_snake"),
            ("FM-SNAKE", "formal_manifold_snake"),
        ]
        if any(
            row.get("method") == "jr_snake_whitened_l10_live"
            for row in regime_row["resource_table_rows"]
        ):
            route_methods.insert(0, ("JR-L10 live", "jr_snake_whitened_l10_live"))
        for route_label, method in route_methods:
            resource = next(
                row
                for row in regime_row["resource_table_rows"]
                if row["method"] == method
            )
            status = str(resource.get("status") or "complete")
            if status != "complete":
                prefix_rows.append(
                    f"{route_label} & {_latex_escape(str(regime_row['regime']))} & "
                    f"{_latex_escape(status)} & -- & -- & -- & -- & -- \\\\"
                )
                continue
            prefix_rows.append(
                f"{route_label} & {_latex_escape(str(regime_row['regime']))} & "
                f"{int(resource['k_pl'])} & {float(resource['abs_delta_e']):.3e} & "
                f"{int(resource['N2q']):,} & {int(resource['D2q']):,} & "
                f"{int(resource['Dc']):,} & {int(resource['S']):,} \\\\"
            )
    setting_rows = "\n".join(
        " & ".join(
            _latex_escape(str(row[key]))
            for key in (
                "curve",
                "selection",
                "rho",
                "linear_solve",
                "warm_start",
                "optimizer",
                "caveat",
            )
        )
        + r" \\"
        for row in setting_ledger
    )
    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[letterpaper,landscape,margin=0.18in]{{geometry}}
\usepackage{{booktabs,graphicx,microtype}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
% BEGIN_MACHINE_READABLE_JOINT_RESPONSE_OVERLAY
% {source_comment}
% END_MACHINE_READABLE_JOINT_RESPONSE_OVERLAY
\begin{{document}}
\begin{{center}}
{{\large\bfseries Paper-I Hubbard--Holstein SNAKE: FM/JR six-regime trajectories and compiled resources}}\\[-0.15ex]
{{\fontsize{{5.35}}{{5.95}}\selectfont \textbf{{Settings:}} live weak-weak JR uses supported-metric whitening, adaptive unbounded-v2 $\rho$, guarded Schur seed, $B_{{\max}}=2$, $L_{{\rm search}}=10$; six-regime JR baseline uses $L_{{\rm search}}=15$; selected-prefix/endpoint Paper-I basis-gate Qiskit transpilation.}}\\[-0.1ex]
\includegraphics[width=0.998\linewidth,height=7.55in,keepaspectratio]{{{image}}}
\end{{center}}

\newpage
\section*{{Provenance appendix}}
{{\small Detailed machine-readable provenance remains in the JSON sidecar; this appendix keeps it off the visual comparison page.}}

\vspace{{0.5em}}
{{\small
\renewcommand{{\arraystretch}}{{1.12}}
\begin{{tabular*}}{{0.98\linewidth}}{{@{{}}p{{0.20\linewidth}}@{{\extracolsep{{\fill}}}}p{{0.75\linewidth}}@{{}}}}
\toprule
Field & Value\\
\midrule
{appendix_manifest_tex}
\bottomrule
\end{{tabular*}}}}

\subsection*{{FM/JR selected-prefix and endpoint audit}}
{{\small
\begin{{tabular*}}{{0.98\linewidth}}{{@{{}}ll@{{\extracolsep{{\fill}}}}rrrrrr@{{}}}}
\toprule
Route & Regime & $k_{{\rm pl}}$ & $|\Delta E|$ & $N_{{2q}}$ & $D_{{2q}}$ & $D_c$ & $S$\\
\midrule
{chr(10).join(prefix_rows)}
\bottomrule
\end{{tabular*}}}}

\vspace{{0.6em}}
\subsection*{{Run-setting caveats and drift}}
{{\fontsize{{6.1}}{{7.1}}\selectfont
This ledger is the quick human comparison. Exact structured settings, source
paths, and hashes are embedded in the \texttt{{.tex}} machine-readable comment
and the JSON sidecar.

\vspace{{0.35em}}
\renewcommand{{\arraystretch}}{{1.18}}
\begin{{tabular}}{{@{{}}p{{1.15in}}p{{1.55in}}p{{1.05in}}p{{1.25in}}p{{0.75in}}p{{1.15in}}p{{2.55in}}@{{}}}}
\toprule
Curve & Selection & Trust radius & Linear solve & Schur seed & Optimizer & Caveat\\
\midrule
{setting_rows}
\bottomrule
\end{{tabular}}}}
\end{{document}}
"""
    path.write_text(tex, encoding="utf-8")


def _resource_table_tex(rows: Sequence[Mapping[str, Any]]) -> str:
    body: list[str] = []
    for row in rows:
        method = str(row.get("method") or "unknown")
        display = RESOURCE_METHOD_DISPLAY.get(
            method,
            str(row.get("method_display") or method),
        )
        status = str(row.get("status") or "complete")
        if status != "complete" and _has_compiled_resource_values(row):
            s_value = row.get("S")
            body.append(
                f"{_latex_escape(display + ' [' + _resource_status_label(status, row) + ']')} & "
                f"{int(row['k_pl'])} & {float(row['abs_delta_e']):.2e} & "
                f"{int(row['N2q']):,} & {int(row['D2q']):,} & "
                f"{int(row['Dc']):,} & "
                f"{'n/a' if s_value is None else f'{int(s_value):,}'} \\\\"
            )
            continue
        if (
            row.get("resource_status")
            == "validated_terminal_query_work_qiskit_operator_sequence_unavailable"
        ):
            body.append(
                f"{_latex_escape(display + ' [S recovered; Qiskit unavailable]')} & "
                f"{int(row['k_pl'])}* & {float(row['abs_delta_e']):.2e}* & "
                r"unavail. & unavail. & unavail. & "
                + f"{int(row['S']):,}\\textsuperscript{{\\dag}} \\\\"
            )
            continue
        if status in PENDING_RESOURCE_STATUSES:
            asterisk = "*" if row.get("checkpoint_asterisk") is True else ""
            body.append(
                f"{_latex_escape(display + ' [' + _resource_status_label(status, row) + ']')} & "
                f"{int(row['k_pl'])}{asterisk} & {float(row['abs_delta_e']):.2e}{asterisk} & "
                r"pending & pending & pending & pending \\"
            )
            continue
        if status != "complete":
            body.append(
                f"{_latex_escape(display)} & {_latex_escape(status)} & -- & -- & -- & -- & -- \\\\"
            )
            continue
        s_value = row.get("S")
        body.append(
            f"{_latex_escape(display)} & {int(row['k_pl'])} & "
            f"{float(row['abs_delta_e']):.2e} & {int(row['N2q']):,} & "
            f"{int(row['D2q']):,} & {int(row['Dc']):,} & "
            f"{'n/a' if s_value is None else f'{int(s_value):,}'} \\\\"
        )
    if not body:
        body.append(r"No completed row & pending & -- & -- & -- & -- & -- \\")
    return "\n".join(body)


def _page_legend_tex(roles: Sequence[str]) -> str:
    entries = []
    for role in roles:
        style = STYLE[role]
        color = str(style["color"]).removeprefix("#")
        entries.append(
            rf"\textcolor[HTML]{{{color}}}{{\rule{{0.13in}}{{1.4pt}}}}\," 
            + _latex_escape(str(style["label"]))
        )
    return r"\quad ".join(entries)


def _comparison_page_tex(
    page: Mapping[str, Any],
    *,
    panel_paths: Mapping[str, Path],
) -> str:
    panels: list[str] = []
    for index, regime_row in enumerate(page["regimes"]):
        regime = str(regime_row["regime"])
        panel = rf"""\begin{{minipage}}[t][3.42in][t]{{0.326\linewidth}}
\centering
{{\fontsize{{7.4}}{{8.0}}\selectfont\bfseries {_latex_escape(str(regime_row['display']))}}}\par
\includegraphics[width=0.985\linewidth,height=1.82in,keepaspectratio]{{{panel_paths[regime].resolve().as_posix()}}}\par
\vspace{{-0.45em}}
{{\fontsize{{3.90}}{{4.40}}\selectfont
\renewcommand{{\arraystretch}}{{0.98}}
\begin{{tabular*}}{{0.985\linewidth}}{{@{{}}p{{0.70in}}@{{\extracolsep{{\fill}}}}rrrrrr@{{}}}}
\toprule
Method & $k$ & $|\Delta E|$ & $N_{{2q}}$ & $D_{{2q}}$ & $D_c$ & $S$\\
\midrule
{_resource_table_tex(regime_row['resource_table_rows'])}
\bottomrule
\end{{tabular*}}}}
\end{{minipage}}"""
        panels.append(panel)
        if index % 3 == 2 and index != len(page["regimes"]) - 1:
            panels.append(r"\par\vspace{0.02in}")
        elif index % 3 != 2:
            panels.append(r"\hfill")
    return rf"""\pagecolor{{white}}\color{{black}}
\begin{{center}}
{{\large\bfseries {_latex_escape(str(page['title']))}}}\\[-0.15ex]
{{\fontsize{{5.65}}{{6.2}}\selectfont {_latex_escape(str(page['subtitle']))}}}\\[-0.15ex]
{{\fontsize{{5.2}}{{5.8}}\selectfont {_page_legend_tex(page['roles'])}}}
\end{{center}}
\vspace{{-0.08in}}
{''.join(panels)}
"""


def _sr_expanded_chart_whitening_page_tex(
    page: Mapping[str, Any],
    *,
    image_path: Path,
) -> str:
    campaign = page["campaign"]
    comparison_rows = {
        str(row["label"]): row for row in campaign["comparisons"]
    }
    display_rows = (
        ("Historical high-accuracy SR baseline", "historical_high_accuracy_sr_baseline"),
        ("Reduced-logical chart whitened", "wrong_reduced_chart_whitened_r22"),
        ("Expanded/projected chart whitened", "good_expanded_chart_whitened_r22"),
        ("Expanded/projected chart whitened", "good_expanded_chart_whitened_r30"),
    )
    result_rows: list[str] = []
    for display, label in display_rows:
        row = comparison_rows[label]
        checkpoint = (
            "--"
            if label == "historical_high_accuracy_sr_baseline"
            else f"{float(row['preterminal_checkpoint_error']):.6e}"
        )
        result_rows.append(
            f"{_latex_escape(display)} & {int(row['rounds'])} & "
            f"{_latex_escape_breakable_identifier(str(row['powell_base_chart']))} & "
            f"{checkpoint} & "
            f"{float(row['finalized_error']):.6e} \\\\"
        )
    validation = campaign["validation"]
    result = campaign["result"]
    reference = campaign["reference"]
    source_lock = campaign["source_lock"]
    s_alg = validation["winning_lineage_s_alg"]
    support_ranks = campaign["support_rank_sequence"]
    source_lock_digest = str(source_lock["archive_sha256"])
    validation_digest = str(campaign["validation_sha256"])
    terminal_actions = result["terminal_actions"]
    round30_metric = campaign["round30_metric_accounting"]
    terminal_metric = campaign["terminal_refit_metric_accounting"]
    support_rank_split = max(1, len(support_ranks) // 2)
    support_rank_rows = (
        (
            f"Support rank, rounds 1--{support_rank_split}",
            ",".join(str(value) for value in support_ranks[:support_rank_split]),
        ),
        (
            f"Support rank, rounds {support_rank_split + 1}--{len(support_ranks)}",
            ",".join(str(value) for value in support_ranks[support_rank_split:]),
        ),
    )
    support_rank_tex = "\n".join(
        f"{_latex_escape(label)} & {_latex_escape(values)} \\\\"
        for label, values in support_rank_rows
        if values
    )
    return rf"""\pagecolor{{white}}\color{{black}}
\begin{{center}}
{{\large\bfseries {_latex_escape(str(page['title']))}}}\\[-0.15ex]
{{\fontsize{{5.8}}{{6.4}}\selectfont {_latex_escape(str(page['subtitle']))}}}
\end{{center}}
\vspace{{-0.08in}}
\begin{{center}}
\includegraphics[width=0.985\linewidth,height=3.48in,keepaspectratio]{{{image_path.resolve().as_posix()}}}
\end{{center}}
\vspace{{-0.10in}}
\begin{{minipage}}[t]{{0.61\linewidth}}
{{\fontsize{{6.0}}{{6.7}}\selectfont
\renewcommand{{\arraystretch}}{{1.08}}
\begin{{tabular*}}{{0.985\linewidth}}{{@{{}}p{{1.55in}}r p{{1.54in}}rr@{{}}}}
\toprule
Evidence & rounds & Powell base chart & checkpoint $|\Delta E|$ & finalized $|\Delta E|$\\
\midrule
{chr(10).join(result_rows)}
\bottomrule
\end{{tabular*}}}}
\end{{minipage}}\hfill
\begin{{minipage}}[t]{{0.375\linewidth}}
{{\fontsize{{5.8}}{{6.6}}\selectfont
\renewcommand{{\arraystretch}}{{1.06}}
\begin{{tabular*}}{{0.985\linewidth}}{{@{{}}p{{1.48in}}@{{\extracolsep{{\fill}}}}p{{1.88in}}@{{}}}}
\toprule
Validation field & Value\\
\midrule
Same-cutoff reference & $n_{{\rm ph}}^{{\rm work}}=n_{{\rm ph}}^{{\rm ref}}={int(reference['n_ph_work'])}$; $E_0={float(reference['same_cutoff_energy']):.12f}$\\
Maximum sector leakage & ${float(validation['maximum_fixed_sector_illegal_probability']):.3e}$\\
Maximum padding leakage & ${float(validation['maximum_binary_padding_illegal_probability']):.3e}$\\
Strict terminal replay & pass; fidelity ${float(validation['strict_terminal_replay_fidelity']):.15f}$\\
Winning-lineage $S_{{\rm alg}}$ & {int(s_alg['S_alg']):,}\\
Discarded-branch unique work & {int(validation['discarded_branch_unique_work']):,}\\
All-branch $S_{{\rm alg}}$ & {int(validation['all_branch_search_s_alg']):,}\\
{support_rank_tex}
Round-30 Gram accounting & $L={int(round30_metric['logical_dimension'])}$, $B={int(round30_metric['expanded_runtime_dimension'])}$, rank={int(round30_metric['retained_support_rank'])}; {int(round30_metric['symmetric_metric_element_occurrences'])} occurrences, {int(round30_metric['new_unique_metric_elements_charged'])} new, {int(round30_metric['deduplicated_metric_elements'])} deduplicated\\
Terminal-refit Gram accounting & fresh chart; {int(terminal_metric['symmetric_metric_element_occurrences'])} occurrences, {int(terminal_metric['new_unique_metric_elements_charged'])} new, {int(terminal_metric['deduplicated_metric_elements'])} deduplicated\\
\bottomrule
\end{{tabular*}}}}
\end{{minipage}}

\vspace{{0.10in}}
{{\fontsize{{6.2}}{{7.1}}\selectfont
\textbf{{What the whitening does.}}
For each newly accepted ansatz, logical tangent states first define the Fubini--Study
Gram matrix $G_L$.  The expanded/projected-runtime base is then formed as
$G_B=P^{{\mathsf T}}G_LP$, where $P$ is the base-to-logical projection recorded by
the chart.  After eigendecomposition, only numerically supported modes are retained;
the accepted Powell map $W_B$ rescales them by the raw $\lambda^{{-1/2}}$ factors.
The recorded ridge is a diagnostic regularization and does not enter this accepted
$W_B$.  A Euclidean Powell displacement therefore represents a geometry-normalized
state-space displacement.  The Gram matrix and whitening map are rebuilt after every
admission, because the state and tangent span changed; they are fixed during that
individual Powell invocation and are \emph{{not}} recomputed at each Powell objective call.
Metric measurements enter $N_{{\rm metric}}$; eigendecomposition and map construction
are classical and have zero quantum-query charge.

\vspace{{0.05in}}
\textbf{{Terminal-action disclosure.}}
The round-30 controller checkpoint has $|\Delta E|={float(result['pre_terminal_checkpoint_abs_error']):.6e}$
at active depth {int(result['pre_terminal_checkpoint_active_depth'])}.  The separately
finalized value $|\Delta E|={float(result['finalized_abs_error']):.6e}$ at depth
{int(result['finalized_active_depth'])} includes a final full refit
({int(terminal_actions['final_full_refit_nfev'])} evaluations) and one accepted
post-refit Phase-I prune/refit recorded in the run log.  These two values are not
silently treated as the same endpoint.

\vspace{{0.05in}}
\textbf{{Immutable provenance.}}
Validation JSON SHA-256 {_latex_escape(validation_digest)}; source archive SHA-256
{_latex_escape(source_lock_digest)}.  Full paths and all consumed hashes are retained
    in the report JSON sidecar.}}
"""


def _sr_expanded_chart_whitening_intermediate_weak_page_tex(
    page: Mapping[str, Any],
    *,
    image_path: Path,
) -> str:
    campaign = page["campaign"]
    route = campaign["route"]
    result = campaign["result"]
    reference = campaign["reference"]
    replay = campaign["checkpoint_replay"]
    accounting = campaign["estimator_accounting"]
    winning = accounting["winning_lineage"]
    all_branch = accounting["all_branch_search_work"]
    source_lock = campaign["source_lock"]
    result_artifact = campaign["verified_artifacts"]["result"]
    terminal_actions = campaign["terminal_actions"]
    qiskit = campaign.get("qiskit")
    qiskit_rows = (
        r"Qiskit resource status & not supplied; no proxy substituted\\"
        if not isinstance(qiskit, Mapping)
        else (
            f"Qiskit resources & $N_{{2q}}={int(qiskit['N2q'])}$; "
            f"$D_{{2q}}={int(qiskit['D2q'])}$; $D_c={int(qiskit['Dc'])}$; "
            f"{int(qiskit['logical_operator_count'])} logical operators / "
            f"{int(qiskit['runtime_rotation_count'])} runtime rotations\\\\\n"
            f"Compile convention & "
            f"{_latex_escape_breakable_identifier(str(qiskit['compile_convention']))}\\\\\n"
            f"Qiskit sidecar SHA-256 & "
            f"{_latex_breakable_digest(str(qiskit['sha256']))}\\\\"
        )
    )
    return rf"""\pagecolor{{white}}\color{{black}}
\begin{{center}}
{{\large\bfseries {_latex_escape(str(page['title']))}}}\\[-0.15ex]
{{\fontsize{{5.8}}{{6.4}}\selectfont {_latex_escape(str(page['subtitle']))}}}
\end{{center}}
\vspace{{-0.08in}}
\begin{{center}}
\includegraphics[width=0.985\linewidth,height=3.52in,keepaspectratio]{{{image_path.resolve().as_posix()}}}
\end{{center}}
\vspace{{-0.09in}}
\begin{{minipage}}[t]{{0.485\linewidth}}
{{\fontsize{{5.85}}{{6.65}}\selectfont
\renewcommand{{\arraystretch}}{{1.05}}
\begin{{tabular*}}{{0.985\linewidth}}{{@{{}}p{{1.72in}}@{{\extracolsep{{\fill}}}}p{{3.35in}}@{{}}}}
\toprule
Route / endpoint field & Value\\
\midrule
Regime and same-cutoff target & intermediate--weak; $n_{{\rm ph}}^{{\rm work}}=n_{{\rm ph}}^{{\rm ref}}={int(reference['n_ph_work'])}$; $E_0={float(reference['same_cutoff_energy']):.14f}$\\
Route family / profile & {_latex_escape_breakable_identifier(str(route['family']))}; {_latex_escape_breakable_identifier(str(route['profile']))}\\
Powell base chart & {_latex_escape_breakable_identifier(str(route['powell_base_chart']))}\\
Accepted-refit chart & {_latex_escape_breakable_identifier(str(route['accepted_refit_coordinate_chart']))}\\
Whitening policy & {_latex_escape_breakable_identifier(str(route['accepted_refit_whitening_policy']))}\\
Adaptive trust & {_latex_escape_breakable_identifier(str(route['adaptive_trust_policy']))}\\
Controller checkpoint & round {int(route['outer_round_horizon'])}; depth {int(result['pre_terminal_checkpoint_active_depth'])}; $|\Delta E|={float(result['pre_terminal_checkpoint_replayed_absolute_error']):.12e}$\\
Finalized result & depth {int(result['finalized_active_depth'])}; $E={float(result['displayed_energy']):.14f}$; $|\Delta E|={float(result['displayed_absolute_error']):.12e}$\\
Stop / terminal action & {_latex_escape(str(result['stop_reason']))}; final full refit {int(terminal_actions['final_full_refit_nfev'])} nfev; terminal prune nominated {int(terminal_actions['terminal_prune_candidate_count'])} and accepted {int(terminal_actions['terminal_prune_accepted_count'])}\\
\bottomrule
\end{{tabular*}}}}
\end{{minipage}}\hfill
\begin{{minipage}}[t]{{0.495\linewidth}}
{{\fontsize{{5.85}}{{6.65}}\selectfont
\renewcommand{{\arraystretch}}{{1.05}}
\begin{{tabular*}}{{0.985\linewidth}}{{@{{}}p{{1.88in}}@{{\extracolsep{{\fill}}}}p{{3.26in}}@{{}}}}
\toprule
Validation / accounting field & Value\\
\midrule
Ordered checkpoints & {int(replay['active_checkpoint_count'])} active checkpoints; terminal hash {_latex_escape(str(replay['terminal_checkpoint_sha256'])[:16])}\ldots\\
Independent finalized replay & discrepancy ${float(result['displayed_replayed_energy_discrepancy']):.3e}$; terminal-state fidelity ${float(replay['terminal_state_fidelity_to_serialized_final_state']):.15f}$\\
Maximum sector leakage & ${float(replay['maximum_fixed_sector_illegal_probability']):.3e}$ (tolerance ${float(replay['leakage_tolerance']):.1e}$)\\
Maximum padding leakage & ${float(replay['maximum_binary_padding_illegal_probability']):.3e}$ (tolerance ${float(replay['leakage_tolerance']):.1e}$)\\
Winning-lineage $S_{{\rm alg}}$ & {int(winning['S_alg']):,} = {int(winning['N_H_outer']):,} + {int(winning['N_H_refit']):,} + {int(winning['N_grad']):,} + {int(winning['N_metric']):,}\\
Discarded / all-branch work & {int(accounting['discarded_branch_unique_work']):,} / {int(all_branch['S_alg']):,}\\
{qiskit_rows}
Validation JSON SHA-256 & {_latex_breakable_digest(str(campaign['validation_sha256']))}\\
Result JSON SHA-256 & {_latex_breakable_digest(str(result_artifact['sha256']))}\\
Source archive SHA-256 & {_latex_breakable_digest(str(source_lock['archive_sha256']))}\\
\bottomrule
\end{{tabular*}}}}
\end{{minipage}}

\vspace{{0.10in}}
{{\fontsize{{6.2}}{{7.0}}\selectfont
\textbf{{Interpretation boundary.}}
This page adds one validated intermediate--weak diagnostic for the same expanded-runtime,
projected-logical Powell chart audited on the weak--weak page.  Its trajectory and
finalized endpoint are shown on the model page as a dark-plum dashed diagnostic,
while the older selected SR-SNAKE row remains present.  The finalized error follows
the configured terminal full refit.  The terminal prune evaluated one nomination,
rejected it, accepted no deletion, and executed no post-prune refit.  Qiskit costs use
the explicitly recorded Paper-I basis-gate convention; the sidecar's own primary-error
field uses a different internal reference and is ignored in favor of this validation
bundle's locked same-cutoff error.  Full paths and SHA-256 hashes for every consumed
artifact are retained in the report JSON sidecar.}}
"""


def _write_model_tex(
    path: Path,
    *,
    panel_paths: Mapping[str, Mapping[str, Path]],
    provenance_path: Path,
    evidence: Mapping[str, Any],
) -> None:
    setting_ledger = [dict(row) for row in evidence["run_setting_caveat_ledger"]]
    sr_whitening_campaign = evidence.get("sr_expanded_chart_whitening_campaign")
    sr_whitening_iw_campaign = evidence.get(
        "sr_expanded_chart_whitening_intermediate_weak_campaign"
    )
    page_contract = [
        "parameter_manifest_human_caveats_and_machine_provenance",
        "selected_model_comparison",
        "jr_policy_comparison",
        "fm_policy_comparison",
        "new_paper_i_route4_comparison",
    ]
    if sr_whitening_campaign is not None:
        page_contract.append("sr_expanded_chart_whitening_weak_weak_diagnostic")
    if sr_whitening_iw_campaign is not None:
        page_contract.append(
            "sr_expanded_chart_whitening_intermediate_weak_validated_diagnostic"
        )
    page_contract.append("original_paper_i_route_context")
    source_comment = json.dumps(
        {
            "schema": SCHEMA,
            "page_contract": page_contract,
            "provenance_json": _rel(provenance_path),
            "builder": _rel(Path(__file__)),
            "builder_sha256": _sha256(Path(__file__)),
            "paper_i_reference_json": evidence["paper_i_reference_json"],
            "paper_i_reference_sha256": evidence["paper_i_reference_sha256"],
            "jr_l10_campaign": evidence["jr_l10_campaign"],
            "jr_chtc_live_snapshot_campaign": evidence.get(
                "jr_chtc_live_snapshot_campaign"
            ),
            "repaired_l25_campaign": evidence["repaired_l25_campaign"],
            "formal_manifold_campaign": evidence["formal_manifold_campaign"],
            "formal_manifold_ablation_campaign": evidence[
                "formal_manifold_ablation_campaign"
            ],
            "formal_manifold_live_snapshot_campaign": evidence.get(
                "formal_manifold_live_snapshot_campaign"
            ),
            "formal_manifold_live_status_campaign": evidence.get(
                "formal_manifold_live_status_campaign"
            ),
            "formal_manifold_completed_resource_recovery_campaign": evidence.get(
                "formal_manifold_completed_resource_recovery_campaign"
            ),
            "formal_manifold_stopped_snapshot_campaign": evidence.get(
                "formal_manifold_stopped_snapshot_campaign"
            ),
            "paper_i_route4_campaign": evidence["paper_i_route4_campaign"],
            "paper_i_route4_live_snapshot_campaign": evidence.get(
                "paper_i_route4_live_snapshot_campaign"
            ),
            "sr_expanded_chart_whitening_campaign": sr_whitening_campaign,
            "sr_expanded_chart_whitening_intermediate_weak_campaign": (
                sr_whitening_iw_campaign
            ),
            "algorithm_setting_ledger": setting_ledger,
            "manuscript_edited": False,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    page_order = [
        "model",
        "jr_policies",
        "fm_policies",
        "paper_i_route4",
    ]
    if sr_whitening_campaign is not None:
        page_order.append(SR_EXPANDED_CHART_WHITENING_PAGE_KEY)
    if sr_whitening_iw_campaign is not None:
        page_order.append(SR_EXPANDED_CHART_WHITENING_IW_PAGE_KEY)
    page_order.append("original_route")
    page_tex: list[str] = []
    for index, page_key in enumerate(page_order):
        if index:
            page_tex.append(r"\newpage")
        if page_key == SR_EXPANDED_CHART_WHITENING_PAGE_KEY:
            page_tex.append(
                _sr_expanded_chart_whitening_page_tex(
                    evidence["pages"][page_key],
                    image_path=panel_paths[page_key]["wide"],
                )
            )
        elif page_key == SR_EXPANDED_CHART_WHITENING_IW_PAGE_KEY:
            page_tex.append(
                _sr_expanded_chart_whitening_intermediate_weak_page_tex(
                    evidence["pages"][page_key],
                    image_path=panel_paths[page_key]["wide"],
                )
            )
        else:
            page_tex.append(
                _comparison_page_tex(
                    evidence["pages"][page_key],
                    panel_paths=panel_paths[page_key],
                )
            )

    manifest_rows = (
        ("Schema", str(evidence["schema"])),
        (
            "HH model parameters",
            "source-derived contract: L=2; t=1; omega0=1; "
            "g_ep=0.790569415042; U/t in {0.25,1.25,8}; "
            "(N_up,N_down)=(1,1)",
        ),
        (
            "Phonon cutoff and target",
            "n_ph_max=2 for weak-Holstein regimes and 4 for strong-Holstein "
            "regimes; exact_gs_energy and abs_delta_e are same-cutoff quantities",
        ),
        (
            "Sector and padding guards",
            "source-derived route contract configures hard fixed-sector guard at "
            "(N_up,N_down)=(1,1) and binary-padding projection before child scoring; "
            "live checkpoint leakage is not claimed without a checkpoint sidecar",
        ),
        (
            "JR selected policy",
            str(evidence["jr_l10_campaign"]["policy"]),
        ),
        (
            "JR completed rounds",
            json.dumps(evidence["jr_l10_campaign"]["regime_rounds"], sort_keys=True),
        ),
        (
            "JR selector exhaustion",
            ", ".join(evidence["jr_l10_campaign"]["selector_exhausted_regimes"])
            or "none",
        ),
        (
            "JR CHTC live snapshots",
            (
                "not supplied"
                if evidence.get("jr_chtc_live_snapshot_campaign") is None
                else "cluster "
                f"{evidence['jr_chtc_live_snapshot_campaign']['cluster_id']}; "
                f"captured={evidence['jr_chtc_live_snapshot_campaign']['captured_at']}; "
                "rollback-free L10/B2; per-entry running/stopped/terminal status; "
                "hash-validated fixed-prefix Qiskit and winning-lineage S where supplied; "
                "archived route used adaptive rho only in the final selector"
            ),
        ),
        (
            "Repaired L25",
            f"cluster {evidence['repaired_l25_campaign']['cluster']}; "
            f"status={evidence['repaired_l25_campaign']['status']}",
        ),
        (
            "SR-SNAKE preserved baseline",
            "complete="
            + ",".join(evidence["paper_i_route4_campaign"]["completed_regimes"])
            + "; partial="
            + ",".join(evidence["paper_i_route4_campaign"]["partial_regimes"])
            + "; not-run="
            + ",".join(evidence["paper_i_route4_campaign"]["not_run_regimes"]),
        ),
        (
            "SR-SNAKE recovery evidence",
            (
                "not supplied"
                if evidence.get("paper_i_route4_live_snapshot_campaign") is None
                else f"captured={evidence['paper_i_route4_live_snapshot_campaign']['captured_at_utc']}; "
                + _sr_recovery_summary(
                    evidence["paper_i_route4_live_snapshot_campaign"]
                )
                + "; preserved round-21 rows retained"
            ),
        ),
        *(
            ()
            if sr_whitening_campaign is None
            else (
                (
                    "SR expanded-chart whitening",
                    "weak-weak validated diagnostic; validation="
                    f"{sr_whitening_campaign['validation_json']}; sha256="
                    f"{sr_whitening_campaign['validation_sha256']}; Gram rebuilt "
                    "after each admission and fixed within each Powell invocation",
                ),
            )
        ),
        *(
            ()
            if sr_whitening_iw_campaign is None
            else (
                (
                    "SR expanded-chart IW diagnostic",
                    "validated intermediate-weak r30; validation="
                    f"{sr_whitening_iw_campaign['validation_json']}; sha256="
                    f"{sr_whitening_iw_campaign['validation_sha256']}; qiskit="
                    + (
                        "not supplied"
                        if sr_whitening_iw_campaign.get("qiskit") is None
                        else f"{sr_whitening_iw_campaign['qiskit']['path']}; sha256="
                        f"{sr_whitening_iw_campaign['qiskit']['sha256']}"
                    ),
                ),
            )
        ),
        (
            "FM",
            "pending="
            + (
                ", ".join(evidence["formal_manifold_campaign"]["pending_regimes"])
                or "none"
            ),
        ),
        (
            "FM live snapshots",
            (
                "not supplied"
                if evidence.get("formal_manifold_live_snapshot_campaign") is None
                else "cluster "
                f"{evidence['formal_manifold_live_snapshot_campaign']['cluster_id']}; "
                f"captured={evidence['formal_manifold_live_snapshot_campaign']['captured_at']}; "
                "nonterminal matched within-batch diagnostic; no source-value anchor; "
                + (
                    "weak-Holstein terminal resources overlaid by the compact recovery bundle"
                    if evidence.get(
                        "formal_manifold_completed_resource_recovery_campaign"
                    )
                    is not None
                    else "Qiskit/query resources pending"
                )
            ),
        ),
        (
            "FM status endpoints",
            (
                "not supplied"
                if evidence.get("formal_manifold_live_status_campaign") is None
                else f"captured={evidence['formal_manifold_live_status_campaign']['captured_at_local']}; "
                "marker-only; IW replacement restart kept discontinuous; "
                "starred rows are last verified checkpoints, not terminal metrics; "
                + (
                    "superseded for five recovered weak-Holstein terminal endpoints"
                    if evidence.get(
                        "formal_manifold_completed_resource_recovery_campaign"
                    )
                    is not None
                    else "Qiskit/query resources pending"
                )
            ),
        ),
        (
            "FM completed weak resources",
            (
                "not supplied"
                if evidence.get(
                    "formal_manifold_completed_resource_recovery_campaign"
                )
                is None
                else "exact terminal winning-lineage S: 6/6; validated Paper-I "
                "Qiskit: 5/6; strong-weak qB-on terminal operator sequence "
                "was omitted, so its Qiskit cell remains unavailable"
            ),
        ),
        (
            "FM stopped checkpoints",
            (
                "not supplied"
                if evidence.get("formal_manifold_stopped_snapshot_campaign") is None
                else "cluster "
                f"{evidence['formal_manifold_stopped_snapshot_campaign']['cluster_id']}; "
                f"stopped={evidence['formal_manifold_stopped_snapshot_campaign']['stopped_at']}; "
                "proc6--11 completed-round fixed prefixes; Paper-I Qiskit costs validated; "
                "S unavailable"
            ),
        ),
        ("Machine provenance", _rel(provenance_path)),
        ("Manuscript", "Paper_I.tex not edited"),
    )
    manifest_tex = "\n".join(
        f"\\textbf{{{_latex_escape(label)}}} & "
        + (
            _latex_escape_breakable_identifier(value)
            if label in {
                "SR expanded-chart whitening",
                "SR expanded-chart IW diagnostic",
            }
            else _latex_escape(value)
        )
        + r" \\"
        for label, value in manifest_rows
    )
    audit_rows: list[str] = []
    audit_methods = {
        "paper_i_route4_live_checkpoint_snake",
        "paper_i_route4_snake",
        "jr_snake_whitened_l10",
        "jr_snake_chtc_live",
    }
    for regime_row in evidence["pages"]["model"]["regimes"]:
        for resource in regime_row["resource_table_rows"]:
            method = str(resource.get("method") or "unknown")
            if method not in audit_methods:
                continue
            display = RESOURCE_METHOD_DISPLAY.get(
                method,
                str(resource.get("method_display") or method),
            )
            status = str(resource.get("status") or "complete")
            if status != "complete" and _has_compiled_resource_values(resource):
                s_value = resource.get("S")
                audit_rows.append(
                    f"{_latex_escape(display + ' [' + _resource_status_label(status, resource) + ']')} & "
                    f"{_latex_escape(str(regime_row['regime']))} & "
                    f"{int(resource['k_pl'])} & {float(resource['abs_delta_e']):.3e} & "
                    f"{int(resource['N2q']):,} & {int(resource['D2q']):,} & "
                    f"{int(resource['Dc']):,} & "
                    f"{'n/a' if s_value is None else f'{int(s_value):,}'} \\\\"
                )
                continue
            if status in PENDING_RESOURCE_STATUSES:
                audit_rows.append(
                    f"{_latex_escape(display + ' [' + _resource_status_label(status, resource) + ']')} & "
                    f"{_latex_escape(str(regime_row['regime']))} & "
                    f"{int(resource['k_pl'])} & {float(resource['abs_delta_e']):.3e} & "
                    r"pending & pending & pending & pending \\"
                )
                continue
            if status != "complete":
                audit_rows.append(
                    f"{_latex_escape(display)} & {_latex_escape(str(regime_row['regime']))} & "
                    f"{_latex_escape(status)} & -- & -- & -- & -- & -- \\\\"
                )
                continue
            audit_rows.append(
                f"{_latex_escape(display)} & {_latex_escape(str(regime_row['regime']))} & "
                f"{int(resource['k_pl'])} & {float(resource['abs_delta_e']):.3e} & "
                f"{int(resource['N2q']):,} & {int(resource['D2q']):,} & "
                f"{int(resource['Dc']):,} & "
                f"{'n/a' if resource.get('S') is None else f'{int(resource['S']):,}'} \\\\"
            )
    setting_rows = "\n".join(
        " & ".join(
            _latex_escape(str(row[key]))
            for key in (
                "curve",
                "selection",
                "rho",
                "linear_solve",
                "warm_start",
                "optimizer",
                "caveat",
            )
        )
        + r" \\"
        for row in setting_ledger
    )
    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[letterpaper,landscape,margin=0.18in]{{geometry}}
\usepackage{{booktabs,graphicx,microtype,xcolor}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
% BEGIN_MACHINE_READABLE_JOINT_RESPONSE_OVERLAY
% {source_comment}
% END_MACHINE_READABLE_JOINT_RESPONSE_OVERLAY
\begin{{document}}
\pagecolor{{white}}\color{{black}}
\section*{{Parameter manifest: Human caveats and machine provenance}}
{{\fontsize{{6.1}}{{6.8}}\selectfont
\begin{{tabular*}}{{0.985\linewidth}}{{@{{}}p{{0.19\linewidth}}@{{\extracolsep{{\fill}}}}p{{0.77\linewidth}}@{{}}}}
\toprule
Field & Value\\
\midrule
{manifest_tex}
\bottomrule
\end{{tabular*}}}}

\vspace{{0.12in}}
{{\fontsize{{5.25}}{{5.8}}\selectfont
\begin{{tabular*}}{{0.985\linewidth}}{{@{{}}ll@{{\extracolsep{{\fill}}}}rrrrrr@{{}}}}
\toprule
Selected model & Regime & $k$ & $|\Delta E|$ & $N_{{2q}}$ & $D_{{2q}}$ & $D_c$ & $S$\\
\midrule
{chr(10).join(audit_rows)}
\bottomrule
\end{{tabular*}}}}

\vspace{{0.12in}}
{{\fontsize{{4.7}}{{5.35}}\selectfont
\renewcommand{{\arraystretch}}{{1.05}}
\begin{{tabular}}{{@{{}}p{{1.08in}}p{{1.42in}}p{{0.92in}}p{{1.12in}}p{{0.72in}}p{{1.02in}}p{{2.72in}}@{{}}}}
\toprule
Curve & Selection & Trust radius & Linear solve & Schur seed & Optimizer & Caveat\\
\midrule
{setting_rows}
\bottomrule
\end{{tabular}}}}

\newpage
{chr(10).join(page_tex)}
\end{{document}}
"""
    path.write_text(tex, encoding="utf-8")


def _compile_latex(tex_path: Path) -> Path:
    executable = shutil.which("latexmk")
    if executable:
        command = [executable, "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    else:
        executable = shutil.which("tectonic")
        if not executable:
            raise RuntimeError("latexmk or tectonic is required")
        command = [executable, "--keep-logs", "--reruns", "2", tex_path.name]
    completed = subprocess.run(command, cwd=tex_path.parent, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout + completed.stderr)
    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    return pdf_path


def _page_count(path: Path) -> int:
    executable = shutil.which("pdfinfo")
    if not executable:
        raise RuntimeError("pdfinfo is required")
    completed = subprocess.run([executable, str(path)], text=True, capture_output=True, check=False)
    match = re.search(r"^Pages:\s+(\d+)\s*$", completed.stdout, re.MULTILINE)
    if completed.returncode != 0 or match is None:
        raise RuntimeError(completed.stderr or completed.stdout)
    return int(match.group(1))


def build(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    stem: str = STEM,
    fm_campaign_root: Path = FM_CAMPAIGN_ROOT,
    fm_live_snapshot_manifest: Path | None = None,
    fm_live_status_snapshot: Path | None = None,
    fm_completed_resource_recovery_manifest: Path | None = None,
    fm_stopped_snapshot_manifest: Path | None = None,
    paper_i_route4_live_snapshot_manifest: Path | None = None,
    jr_chtc_live_snapshot_manifest: Path | None = None,
    prior_report_json: Path | None = None,
    sr_expanded_chart_whitening_validation_json: Path | None = None,
    sr_expanded_chart_whitening_intermediate_weak_validation_json: Path | None = None,
    sr_expanded_chart_whitening_intermediate_weak_qiskit_json: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = build_evidence(
        output_dir,
        fm_campaign_root=fm_campaign_root,
        fm_live_snapshot_manifest=fm_live_snapshot_manifest,
        fm_live_status_snapshot=fm_live_status_snapshot,
        fm_completed_resource_recovery_manifest=(
            fm_completed_resource_recovery_manifest
        ),
        fm_stopped_snapshot_manifest=fm_stopped_snapshot_manifest,
        paper_i_route4_live_snapshot_manifest=(
            paper_i_route4_live_snapshot_manifest
        ),
        jr_chtc_live_snapshot_manifest=jr_chtc_live_snapshot_manifest,
        prior_report_json=prior_report_json,
        sr_expanded_chart_whitening_validation_json=(
            sr_expanded_chart_whitening_validation_json
        ),
        sr_expanded_chart_whitening_intermediate_weak_validation_json=(
            sr_expanded_chart_whitening_intermediate_weak_validation_json
        ),
        sr_expanded_chart_whitening_intermediate_weak_qiskit_json=(
            sr_expanded_chart_whitening_intermediate_weak_qiskit_json
        ),
    )
    provenance_path = output_dir / f"{stem}.json"
    tex_path = output_dir / f"{stem}.tex"
    panel_paths = {
        page_key: _plot_report_page(
            page_key,
            page,
            output_dir=output_dir,
            stem=stem,
        )
        for page_key, page in evidence["pages"].items()
    }
    evidence["artifacts"] = {
        "tex": _rel(tex_path),
        "pdf": _rel(tex_path.with_suffix(".pdf")),
        "panel_images": {
            page_key: {
                regime: _rel(path)
                for regime, path in paths.items()
            }
            for page_key, paths in panel_paths.items()
        },
    }
    provenance_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_model_tex(
        tex_path,
        panel_paths=panel_paths,
        provenance_path=provenance_path,
        evidence=evidence,
    )
    pdf_path = _compile_latex(tex_path)
    expected_page_count = (
        6
        + int(evidence.get("sr_expanded_chart_whitening_campaign") is not None)
        + int(
            evidence.get("sr_expanded_chart_whitening_intermediate_weak_campaign")
            is not None
        )
    )
    if _page_count(pdf_path) != expected_page_count:
        raise ValueError(
            f"{expected_page_count}-page model-comparison PDF contract failed"
        )
    return {
        "pdf": str(pdf_path),
        "provenance": str(provenance_path),
        "tex": str(tex_path),
        "panel_images": {
            page_key: {regime: str(path) for regime, path in paths.items()}
            for page_key, paths in panel_paths.items()
        },
        "status": evidence["status"],
        "pending_fm_regimes": evidence["formal_manifold_campaign"][
            "pending_regimes"
        ],
        "fm_live_snapshot_manifest": (
            None
            if evidence.get("formal_manifold_live_snapshot_campaign") is None
            else evidence["formal_manifold_live_snapshot_campaign"]["manifest_json"]
        ),
        "fm_live_status_snapshot": (
            None
            if evidence.get("formal_manifold_live_status_campaign") is None
            else evidence["formal_manifold_live_status_campaign"]["status_json"]
        ),
        "fm_live_status_snapshot_sha256": (
            None
            if evidence.get("formal_manifold_live_status_campaign") is None
            else evidence["formal_manifold_live_status_campaign"]["status_sha256"]
        ),
        "fm_completed_resource_recovery_manifest": (
            None
            if evidence.get("formal_manifold_completed_resource_recovery_campaign")
            is None
            else evidence["formal_manifold_completed_resource_recovery_campaign"][
                "manifest_json"
            ]
        ),
        "fm_completed_resource_recovery_manifest_sha256": (
            None
            if evidence.get("formal_manifold_completed_resource_recovery_campaign")
            is None
            else evidence["formal_manifold_completed_resource_recovery_campaign"][
                "manifest_sha256"
            ]
        ),
        "fm_stopped_snapshot_manifest": (
            None
            if evidence.get("formal_manifold_stopped_snapshot_campaign") is None
            else evidence["formal_manifold_stopped_snapshot_campaign"]["manifest_json"]
        ),
        "fm_stopped_snapshot_manifest_sha256": (
            None
            if evidence.get("formal_manifold_stopped_snapshot_campaign") is None
            else evidence["formal_manifold_stopped_snapshot_campaign"]["manifest_sha256"]
        ),
        "paper_i_route4_live_snapshot_manifest": (
            None
            if evidence.get("paper_i_route4_live_snapshot_campaign") is None
            else evidence["paper_i_route4_live_snapshot_campaign"]["manifest_json"]
        ),
        "paper_i_route4_live_snapshot_manifest_sha256": (
            None
            if evidence.get("paper_i_route4_live_snapshot_campaign") is None
            else evidence["paper_i_route4_live_snapshot_campaign"]["manifest_sha256"]
        ),
        "jr_chtc_live_snapshot_manifest": (
            None
            if evidence.get("jr_chtc_live_snapshot_campaign") is None
            else evidence["jr_chtc_live_snapshot_campaign"]["manifest_json"]
        ),
        "jr_chtc_live_snapshot_manifest_sha256": (
            None
            if evidence.get("jr_chtc_live_snapshot_campaign") is None
            else evidence["jr_chtc_live_snapshot_campaign"]["manifest_sha256"]
        ),
        "sr_expanded_chart_whitening_validation_json": (
            None
            if evidence.get("sr_expanded_chart_whitening_campaign") is None
            else evidence["sr_expanded_chart_whitening_campaign"]["validation_json"]
        ),
        "sr_expanded_chart_whitening_validation_sha256": (
            None
            if evidence.get("sr_expanded_chart_whitening_campaign") is None
            else evidence["sr_expanded_chart_whitening_campaign"][
                "validation_sha256"
            ]
        ),
        "sr_expanded_chart_whitening_intermediate_weak_validation_json": (
            None
            if evidence.get(
                "sr_expanded_chart_whitening_intermediate_weak_campaign"
            )
            is None
            else evidence[
                "sr_expanded_chart_whitening_intermediate_weak_campaign"
            ]["validation_json"]
        ),
        "sr_expanded_chart_whitening_intermediate_weak_validation_sha256": (
            None
            if evidence.get(
                "sr_expanded_chart_whitening_intermediate_weak_campaign"
            )
            is None
            else evidence[
                "sr_expanded_chart_whitening_intermediate_weak_campaign"
            ]["validation_sha256"]
        ),
        "sr_expanded_chart_whitening_intermediate_weak_qiskit_json": (
            None
            if evidence.get(
                "sr_expanded_chart_whitening_intermediate_weak_campaign"
            )
            is None
            or evidence[
                "sr_expanded_chart_whitening_intermediate_weak_campaign"
            ].get("qiskit")
            is None
            else evidence[
                "sr_expanded_chart_whitening_intermediate_weak_campaign"
            ]["qiskit"]["path"]
        ),
        "sr_expanded_chart_whitening_intermediate_weak_qiskit_sha256": (
            None
            if evidence.get(
                "sr_expanded_chart_whitening_intermediate_weak_campaign"
            )
            is None
            or evidence[
                "sr_expanded_chart_whitening_intermediate_weak_campaign"
            ].get("qiskit")
            is None
            else evidence[
                "sr_expanded_chart_whitening_intermediate_weak_campaign"
            ]["qiskit"]["sha256"]
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default=STEM)
    parser.add_argument("--fm-campaign-root", type=Path, default=FM_CAMPAIGN_ROOT)
    parser.add_argument(
        "--fm-live-snapshot-manifest",
        type=Path,
        help=(
            "Optional formal_manifold_live_snapshot_bundle_v1 manifest. "
            "Snapshots remain nonterminal and resource fields remain pending."
        ),
    )
    parser.add_argument(
        "--fm-live-status-snapshot",
        type=Path,
        help=(
            "Optional formal_manifold_lightweight_status_snapshot_v1. "
            "It updates endpoint markers only; prior trajectory points remain unchanged."
        ),
    )
    parser.add_argument(
        "--fm-completed-resource-recovery-manifest",
        type=Path,
        help=(
            "Optional formal_manifold_completed_weak_resource_recovery_manifest_v1. "
            "It overlays hash-linked terminal winning-lineage S and validated "
            "Paper-I Qiskit costs recovered after CHTC packaging omitted full results."
        ),
    )
    parser.add_argument(
        "--fm-stopped-snapshot-manifest",
        type=Path,
        help=(
            "Optional formal_manifold_stop_retrieval_manifest_v1 for proc6-11. "
            "It replaces stale live/status rows with completed-round stopped "
            "trajectories and report-only Qiskit fixed-prefix costs."
        ),
    )
    parser.add_argument(
        "--paper-i-route4-live-snapshot-manifest",
        type=Path,
        help=(
            "Optional Paper-I SR recovery manifest. Immutable checkpoints remain "
            "additive to preserved Route-4 rows; pending checkpoints are marker-only, "
            "while hash-validated stopped-prefix sidecars may supply the complete "
            "retained trajectory, Qiskit costs, replay, leakage, and reconstructed S_alg."
        ),
    )
    parser.add_argument(
        "--jr-chtc-live-snapshot-manifest",
        type=Path,
        help=(
            "Optional jr_snake_chtc_live_snapshot_bundle_v1 manifest. "
            "Its immutable trajectories remain snapshot evidence and resource "
            "fields stay pending until fixed sidecars are available."
        ),
    )
    parser.add_argument(
        "--prior-report-json",
        type=Path,
        help=(
            "Optional immutable prior overlay JSON used only to recover preserved "
            "repaired-L25 rows when their unpacked source bundle is unavailable."
        ),
    )
    parser.add_argument(
        "--sr-expanded-chart-whitening-validation-json",
        type=Path,
        help=(
            "Optional paper_i_hh_sr_expanded_chart_whitening_validation_v1 "
            "bundle. Adds one weak-weak page with hash-validated trajectories, "
            "support ranks, replay/leakage/accounting evidence, and terminal-action "
            "disclosure."
        ),
    )
    parser.add_argument(
        "--sr-expanded-chart-whitening-intermediate-weak-validation-json",
        type=Path,
        help=(
            "Optional paper_i_hh_sr_completed_run_validation_v1 bundle for the "
            "validated intermediate-weak expanded-chart r30 diagnostic. Adds an "
            "additive page and model-page diagnostic marker without replacing the "
            "selected SR row."
        ),
    )
    parser.add_argument(
        "--sr-expanded-chart-whitening-intermediate-weak-qiskit-json",
        type=Path,
        help=(
            "Optional hash-closed Paper-I basis-gate Qiskit sidecar for the "
            "intermediate-weak diagnostic. Requires the matching completed-run "
            "validation JSON."
        ),
    )
    args = parser.parse_args(argv)
    optional_build_kwargs: dict[str, Any] = {}
    if args.prior_report_json is not None:
        optional_build_kwargs["prior_report_json"] = args.prior_report_json
    if args.sr_expanded_chart_whitening_validation_json is not None:
        optional_build_kwargs["sr_expanded_chart_whitening_validation_json"] = (
            args.sr_expanded_chart_whitening_validation_json
        )
    if (
        args.sr_expanded_chart_whitening_intermediate_weak_validation_json
        is not None
    ):
        optional_build_kwargs[
            "sr_expanded_chart_whitening_intermediate_weak_validation_json"
        ] = args.sr_expanded_chart_whitening_intermediate_weak_validation_json
    if args.sr_expanded_chart_whitening_intermediate_weak_qiskit_json is not None:
        optional_build_kwargs[
            "sr_expanded_chart_whitening_intermediate_weak_qiskit_json"
        ] = args.sr_expanded_chart_whitening_intermediate_weak_qiskit_json
    print(
        json.dumps(
            build(
                args.output_dir,
                args.stem,
                args.fm_campaign_root,
                args.fm_live_snapshot_manifest,
                args.fm_live_status_snapshot,
                args.fm_completed_resource_recovery_manifest,
                args.fm_stopped_snapshot_manifest,
                args.paper_i_route4_live_snapshot_manifest,
                args.jr_chtc_live_snapshot_manifest,
                **optional_build_kwargs,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
