"""Run the frozen-SR anchor before the FM accepted-reoptimizer backtrack.

The comparison changes one mechanism family: accepted-ansatz
reoptimization.  Candidate selection remains the recovered Paper-I
singleton-response controller.  The first two stages reproduce the immutable
historical weak--weak anchor and then prove current-code SR parity.  Only after
both pass may the qB-off FM variant run.  The intermediate--weak cell is a
cross-regime policy transfer and is deliberately not labeled a source-locked
same-regime sensitivity.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

from pipelines.static_adapt.cli_config import _build_adapt_arg_parser
from pipelines.static_adapt.formal_manifold_route_profile import (
    FM_ROUTE_PROFILE_SR_SOURCE_LOCKED_SUPPORTED_WHITENED_ADAPTIVE_TRUST_V1,
    FORMAL_MANIFOLD_REOPTIMIZATION_ROUTE,
)
from pipelines.static_adapt.formal_manifold_warm_start import FormalManifoldConfig
from pipelines.static_adapt.selector_query_closure import QueryPrimitiveLedger


SCHEMA = "paper_i_hh_fm_sr_source_locked_backtrack_campaign_v1"
AUDIT_SCHEMA = "source_locked_sensitivity_audit_v1"
CAMPAIGN_ID = "paper_i_hh_fm_sr_source_locked_qbroyd_off_backtrack_v2_20260715"
DISK_LAUNCH_FLOOR_GIB = 10.0

SOURCE_COMMAND = Path(
    "raw_outputs/"
    "paper_i_hh_weak_weak_route4_whitened_adaptive_geometry_expansion_repair_20260712/"
    "full/command.json"
)
SOURCE_RESULT = Path(
    "raw_outputs/"
    "paper_i_hh_weak_weak_route4_whitened_adaptive_geometry_expansion_repair_20260712/"
    "full/json/result.json"
)
ARCHIVE_REVISION = Path(
    "raw_outputs/paper_i_hh_sr_snake_historical_source_recovery_20260714/"
    "source_lock_revision_v2_self_contained_20260715"
)
ARCHIVE_LAUNCHER = ARCHIVE_REVISION / "launch_exact_0caf_replay_v2.py"
ARCHIVE_TARBALL = ARCHIVE_REVISION / (
    "source_lock/"
    "paper_i_hh_sr_snake_0caf2834_self_contained_source_tree_v2.tar.gz"
)

EXPECTED_SOURCE_COMMAND_SHA256 = (
    "37751de2805875337cb8a0034a7394b02344c893e1b0a583439b1954c7c8061e"
)
EXPECTED_ARCHIVE_SHA256 = (
    "c290d9ee1b31cd211e41faad174cd2e311ca65cf351c46bbb84fbaaea9504c6c"
)
EXPECTED_ARCHIVE_LAUNCHER_SHA256 = (
    "7e1efb128cd65203ae8598404e908d7beee1aa9cceb15b267a111ef99dd8ccf2"
)
EXPECTED_SOURCE_RESULT_SHA256 = (
    "f8d2bb9756d395d7806bb2f365d95a5fcb4c5aa6de55e96f89ecfc35295b10da"
)
EXPECTED_SOURCE_ABS_DELTA_E = 4.472864776339236e-7
EXPECTED_CONTROLLER_HORIZON = 30
LIVE_IMPLEMENTATION_LOCK_SCHEMA = "fm_sr_live_implementation_lock_v1"
LIVE_IMPLEMENTATION_ROOTS = (
    Path("src"),
    Path("pipelines/static_adapt"),
    Path("pipelines/scaffold"),
    Path("pipelines/contracts"),
    Path("docs/reports"),
)

WEAK_WEAK = {
    "id": "weak-weak",
    "u": 0.25,
    "g_ep": 0.353553390593,
    "n_ph_max": 2,
    "exact_energy": -0.9183531194991743,
}
INTERMEDIATE_WEAK = {
    "id": "intermediate-weak",
    "u": 1.25,
    "g_ep": 0.353553390593,
    "n_ph_max": 2,
    "exact_energy": -0.49499563910866023,
}

_OPERATIONAL_DESTINATIONS = frozenset(
    {
        "adapt_current_json",
        "adapt_estimator_call_ledger_json",
        "output_json",
    }
)
_FM_MECHANISM_DESTINATIONS = frozenset(
    {
        "adapt_reoptimization_route",
        "adapt_formal_manifold_route_profile",
        "adapt_formal_manifold_config_json",
    }
)
_INTERMEDIATE_TRANSFER_DESTINATIONS = frozenset(
    _OPERATIONAL_DESTINATIONS
    | _FM_MECHANISM_DESTINATIONS
    | {
        "adapt_exact_gs_override",
        "u",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Expected a JSON object: {path}")
    return dict(raw)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _required_authorities(repo_root: Path) -> dict[str, Path]:
    return {
        "source_command": repo_root / SOURCE_COMMAND,
        "source_result": repo_root / SOURCE_RESULT,
        "archive_launcher": repo_root / ARCHIVE_LAUNCHER,
        "archive_tarball": repo_root / ARCHIVE_TARBALL,
    }


def _validate_source_authorities(repo_root: Path) -> dict[str, Path]:
    required = _required_authorities(repo_root)
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    expected_hashes = {
        "source_command": EXPECTED_SOURCE_COMMAND_SHA256,
        "source_result": EXPECTED_SOURCE_RESULT_SHA256,
        "archive_launcher": EXPECTED_ARCHIVE_LAUNCHER_SHA256,
        "archive_tarball": EXPECTED_ARCHIVE_SHA256,
    }
    for key, expected in expected_hashes.items():
        actual = _sha256(required[key])
        if actual != expected:
            raise ValueError(
                f"Source-lock authority hash mismatch for {key}: "
                f"{actual} != {expected}"
            )
    source_result = _read_json(required["source_result"])
    source_error = float(source_result["adapt_vqe"]["abs_delta_e"])
    if not math.isfinite(source_error) or not math.isclose(
        source_error,
        EXPECTED_SOURCE_ABS_DELTA_E,
        rel_tol=0.0,
        abs_tol=1.0e-16,
    ):
        raise ValueError("Historical source result error drifted")
    return required


def _live_implementation_lock(repo_root: Path) -> dict[str, Any]:
    """Hash the live scientific import surface used by current-code stages."""

    files: dict[str, str] = {}
    for relative_root in LIVE_IMPLEMENTATION_ROOTS:
        root = repo_root / relative_root
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(root.rglob("*.py")):
            if not path.is_file():
                continue
            relative = str(path.relative_to(repo_root))
            files[relative] = _sha256(path)
    if not files:
        raise ValueError("Live implementation lock resolved no Python files")
    return {
        "schema": LIVE_IMPLEMENTATION_LOCK_SCHEMA,
        "roots": [str(value) for value in LIVE_IMPLEMENTATION_ROOTS],
        "file_count": len(files),
        "files": files,
        "tree_sha256": _json_sha256(files),
    }


def _set_value_flag(argv: Sequence[str], flag: str, value: str) -> list[str]:
    tokens = [str(token) for token in argv]
    if flag in tokens:
        if tokens.count(flag) != 1:
            raise ValueError(f"Expected at most one {flag}")
        index = tokens.index(flag)
        if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
            raise ValueError(f"Flag {flag} has no value")
        tokens[index + 1] = str(value)
        return tokens
    return [*tokens, str(flag), str(value)]


def _normalized_namespace(argv: Sequence[str]) -> dict[str, Any]:
    tokens = [str(value) for value in argv]
    if len(tokens) < 3 or tokens[1:3] != [
        "-m",
        "pipelines.static_adapt.adapt_pipeline",
    ]:
        raise ValueError("Expected a direct adapt_pipeline module invocation")
    parser = _build_adapt_arg_parser(adapt_gradient_parity_rtol=1.0e-8)
    parsed = parser.parse_args(tokens[3:])
    normalized: dict[str, Any] = {}
    for key, value in vars(parsed).items():
        if isinstance(value, Path):
            normalized[str(key)] = str(value)
        elif isinstance(value, tuple):
            normalized[str(key)] = list(value)
        else:
            normalized[str(key)] = value
    return normalized


def audit_semantic_diff(
    source_argv: Sequence[str],
    effective_argv: Sequence[str],
    *,
    allowed_destinations: frozenset[str],
) -> dict[str, Any]:
    source = _normalized_namespace(source_argv)
    effective = _normalized_namespace(effective_argv)
    changed = {
        key: {"source": source.get(key), "effective": effective.get(key)}
        for key in sorted(set(source) | set(effective))
        if source.get(key) != effective.get(key)
    }
    unexpected = sorted(set(changed) - set(allowed_destinations))
    payload = {
        "schema": AUDIT_SCHEMA,
        "status": "pass" if not unexpected else "blocked",
        "source_argv_sha256": _json_sha256(list(source_argv)),
        "effective_argv_sha256": _json_sha256(list(effective_argv)),
        "allowed_destinations": sorted(allowed_destinations),
        "changed_destinations": changed,
        "unexpected_changed_destinations": unexpected,
        "non_swept_settings_diff_empty": not unexpected,
    }
    if unexpected:
        raise ValueError(
            "Source-lock semantic argv audit failed: " + ", ".join(unexpected)
        )
    return payload


def _direct_paths(campaign_dir: Path, stage_id: str) -> dict[str, Path]:
    root = campaign_dir / stage_id
    return {
        "root": root,
        "result": root / "result.json",
        "current": root / "current.json",
        "estimator_ledger": root / "estimator_call_ledger.json",
        "command": root / "command.json",
        "validation": root / "validation.json",
        "normalized_manifest": root / "normalized_manifest.json",
        "stdout": root / "stdout.log",
        "stderr": root / "stderr.log",
    }


def _archive_paths(campaign_dir: Path) -> dict[str, Path]:
    root = campaign_dir / "archived_sr_anchor"
    return {
        "root": root,
        "result": root / "json" / "result.json",
        "current": root / "json" / "current.json",
        "estimator_ledger": root / "json" / "estimator_call_ledger.json",
        "command": root / "command.json",
        "validation": root / "validation.json",
        "normalized_manifest": root / "normalized_manifest.json",
        "stdout": campaign_dir / "archived_sr_anchor.stdout.log",
        "stderr": campaign_dir / "archived_sr_anchor.stderr.log",
    }


def _source_argv(repo_root: Path) -> list[str]:
    source_command = repo_root / SOURCE_COMMAND
    if _sha256(source_command) != EXPECTED_SOURCE_COMMAND_SHA256:
        raise ValueError("Historical weak--weak command hash drifted")
    payload = _read_json(source_command)
    argv = [str(value) for value in payload.get("argv", [])]
    if argv[:3] != ["python3", "-m", "pipelines.static_adapt.adapt_pipeline"]:
        raise ValueError("Historical weak--weak command entry point drifted")
    argv[0] = sys.executable
    return argv


def build_live_anchor_argv(*, repo_root: Path, campaign_dir: Path) -> list[str]:
    paths = _direct_paths(campaign_dir, "current_sr_parity_anchor")
    argv = _source_argv(repo_root)
    argv = _set_value_flag(argv, "--adapt-current-json", str(paths["current"]))
    argv = _set_value_flag(
        argv,
        "--adapt-estimator-call-ledger-json",
        str(paths["estimator_ledger"]),
    )
    return _set_value_flag(argv, "--output-json", str(paths["result"]))


def build_fm_argv(
    *,
    repo_root: Path,
    campaign_dir: Path,
    stage_id: str,
    regime: Mapping[str, Any],
    formal_config_json: Path,
) -> list[str]:
    paths = _direct_paths(campaign_dir, stage_id)
    argv = _source_argv(repo_root)
    argv = _set_value_flag(
        argv,
        "--adapt-reoptimization-route",
        FORMAL_MANIFOLD_REOPTIMIZATION_ROUTE,
    )
    argv = _set_value_flag(
        argv,
        "--adapt-formal-manifold-route-profile",
        FM_ROUTE_PROFILE_SR_SOURCE_LOCKED_SUPPORTED_WHITENED_ADAPTIVE_TRUST_V1,
    )
    argv = _set_value_flag(
        argv,
        "--adapt-formal-manifold-config-json",
        str(formal_config_json),
    )
    argv = _set_value_flag(argv, "--u", repr(float(regime["u"])))
    argv = _set_value_flag(argv, "--g-ep", repr(float(regime["g_ep"])))
    argv = _set_value_flag(argv, "--n-ph-max", str(int(regime["n_ph_max"])))
    if str(regime["id"]) != "weak-weak":
        argv = _set_value_flag(
            argv,
            "--adapt-exact-gs-override",
            repr(float(regime["exact_energy"])),
        )
    argv = _set_value_flag(argv, "--adapt-current-json", str(paths["current"]))
    argv = _set_value_flag(
        argv,
        "--adapt-estimator-call-ledger-json",
        str(paths["estimator_ledger"]),
    )
    return _set_value_flag(argv, "--output-json", str(paths["result"]))


_EPHEMERAL_SIGNATURE_KEYS = frozenset(
    {
        "controller_snapshot",
        "gradient_eval_elapsed_s",
        "iter_elapsed_s",
        "measurement_cache_stats",
        "optimizer_elapsed_s",
        "selector_measurement_cache_stats",
    }
)


def _strip_ephemeral_signature_fields(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _strip_ephemeral_signature_fields(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _EPHEMERAL_SIGNATURE_KEYS
            and not str(key).endswith("_elapsed_s")
        }
    if isinstance(value, list):
        return [_strip_ephemeral_signature_fields(item) for item in value]
    return value


def _scientific_payload_sha256(value: Any) -> str:
    return _json_sha256(_strip_ephemeral_signature_fields(value))


_SELECTED_FEATURE_SIGNATURE_KEYS = (
    "candidate_family",
    "candidate_label",
    "candidate_pool_index",
    "generator_id",
    "parent_generator_id",
    "position_id",
    "append_position",
    "active_post_refit_indices",
    "refit_window_indices",
    "schur_window_indices",
    "actual_fallback_mode",
    "curvature_mode",
    "F_metric",
    "F_raw",
    "F_red",
    "H_window",
    "b_hat",
    "h_eff",
    "h_hat",
    "ridge_used",
    "novelty",
    "novelty_mode",
    "phase2_novelty_mode",
    "phase2_raw_novelty",
    "phase2_raw_trust_gain",
    "phase2_raw_score",
    "phase2_raw_score_formula",
    "phase3_reduced_novelty",
    "phase3_reduced_trust_gain",
    "phase3_primary_score",
    "phase3_canonical_score_formula",
    "selector_score",
    "full_v2_score",
    "compile_cost_total",
    "c_hat_1q",
    "c_hat_2q",
    "c_hat_d",
    "c_hat_shot",
    "c_hat_theta",
    "new_group_cost",
    "new_shot_cost",
    "reuse_count_cost",
    "phase2_burden_total",
    "phase3_burden_total",
    "runtime_split_mode",
    "runtime_split_chosen_representation",
    "runtime_split_child_indices",
    "runtime_split_child_generator_ids",
    "route_a_coordinate_model_infeasible_reason",
    "route_a_geometry_expansion_mode",
    "route_a_geometry_expansion_reason",
    "route_a_geometry_expansion_score",
)


def _feature_signature(records: Any) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        return []
    return [
        {key: record.get(key) for key in _SELECTED_FEATURE_SIGNATURE_KEYS}
        for record in records
        if isinstance(record, Mapping)
    ]


def _row_query_work_signature(proxy: Any) -> dict[str, Any]:
    proxy = dict(proxy) if isinstance(proxy, Mapping) else {}
    return {
        key: proxy.get(key)
        for key in (
            "actual_evaluated_candidate_count_total",
            "actual_operator_probe_count",
            "actual_operator_probe_count_total",
            "candidate_count_total",
            "controller_group_proxy",
            "controller_shot_proxy",
            "expanded_measurement_group_probe_count",
            "expanded_measurement_group_probe_count_total",
            "groups_new",
            "groups_reused",
            "groups_total",
            "records_evaluated",
            "shots_new",
            "shots_reused",
            "shots_total",
            "work_scope_count",
        )
    }


def _query_work_signature(adapt: Mapping[str, Any]) -> dict[str, Any]:
    accounting = adapt.get("estimator_call_accounting", {})
    accounting = dict(accounting) if isinstance(accounting, Mapping) else {}

    def _scope_totals(name: str) -> dict[str, Any]:
        scope = accounting.get(name, {})
        scope = dict(scope) if isinstance(scope, Mapping) else {}
        return {
            key: scope.get(key)
            for key in (
                "N_H_outer",
                "N_H_refit",
                "N_grad",
                "N_metric",
                "S_alg",
                "unique_primitive_count",
            )
        }

    controller = adapt.get("controller_measurement_work_summary", {})
    controller = dict(controller) if isinstance(controller, Mapping) else {}
    return {
        "nfev_total": adapt.get("nfev_total"),
        "winning_branch": _scope_totals("winning_branch"),
        "all_branch_search_work": _scope_totals("all_branch_search_work"),
        "controller": {
            key: controller.get(key)
            for key in (
                "actual_operator_probe_count",
                "candidate_count_total",
                "controller_group_proxy",
                "controller_shot_proxy",
                "expanded_measurement_group_probe_count",
                "groups_new",
                "groups_reused",
                "history_row_count",
                "records_evaluated",
                "shots_new",
                "shots_reused",
                "work_scope_count",
            )
        },
    }


def _history_signature(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    adapt = result.get("adapt_vqe", {})
    rows = adapt.get("history", []) if isinstance(adapt, Mapping) else []
    signature: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        signature.append(
            {
                "depth": int(row.get("depth", len(signature) + 1)),
                "selected_ops": [str(value) for value in row.get("selected_ops", [])],
                "selected_positions": [
                    int(value) for value in row.get("selected_positions", [])
                ],
                "reopt_policy": str(row.get("reopt_policy", "")),
                "periodic_full": bool(
                    row.get("reopt_periodic_full_refit_triggered", False)
                ),
                "reopt_policy_effective": str(
                    row.get("reopt_policy_effective", "")
                ),
                "reopt_active_indices": [
                    int(value) for value in row.get("reopt_active_indices", [])
                ],
                "refit_window_indices": [
                    int(value) for value in row.get("refit_window_indices", [])
                ],
                "stage_name": str(row.get("stage_name", "")),
                "selection_mode": str(row.get("selection_mode", "")),
                "actual_fallback_mode": str(
                    row.get("actual_fallback_mode", "")
                ),
                "fallback_scan_size": int(row.get("fallback_scan_size", 0) or 0),
                "fallback_best_probe_theta": row.get("fallback_best_probe_theta"),
                "fallback_best_probe_theta_values": row.get(
                    "fallback_best_probe_theta_values"
                ),
                "fallback_best_probe_delta_e": row.get(
                    "fallback_best_probe_delta_e"
                ),
                "phase3_batching_enabled": bool(
                    row.get("phase3_batching_enabled", False)
                ),
                "batch_size": int(row.get("batch_size", 1) or 1),
                "runtime_split_mode": str(row.get("runtime_split_mode", "")),
                "runtime_split_child_count": int(
                    row.get("runtime_split_child_count", 0) or 0
                ),
                "novelty_mode": str(row.get("novelty_mode", "")),
                "selector_score": row.get("selector_score"),
                "phase2_raw_score": row.get("phase2_raw_score"),
                "full_v2_score": row.get("full_v2_score"),
                "nfev_opt": int(row.get("nfev_opt", 0) or 0),
                "nfev_seed_probe": int(row.get("nfev_seed_probe", 0) or 0),
                "nfev_step_total_delta": int(
                    row.get("nfev_step_total_delta", 0) or 0
                ),
                "nfev_total_after_step": int(
                    row.get("nfev_total_after_step", 0) or 0
                ),
                "controller_measurement_work_proxy": _row_query_work_signature(
                    row.get("controller_measurement_work_proxy")
                ),
                "selected_feature_rows": _feature_signature(
                    row.get("selected_feature_rows", [])
                ),
                "admitted_records": _feature_signature(
                    row.get("admitted_records", [])
                ),
            }
        )
    return signature


def _controller_signature(result: Mapping[str, Any]) -> dict[str, Any]:
    adapt = result.get("adapt_vqe", {})
    adapt = dict(adapt) if isinstance(adapt, Mapping) else {}
    history = _history_signature(result)
    return {
        "success": adapt.get("success"),
        "stop_reason": adapt.get("stop_reason"),
        "ansatz_depth": adapt.get("ansatz_depth"),
        "operators": [str(value) for value in adapt.get("operators", [])],
        "history": history,
        "query_work": _query_work_signature(adapt),
        "final_full_refit": {
            key: dict(adapt.get("final_full_refit", {})).get(key)
            for key in (
                "performed",
                "success",
                "energy",
                "fun",
                "nfev",
                "nit",
                "optimizer",
                "active_indices",
            )
        }
        if isinstance(adapt.get("final_full_refit"), Mapping)
        else {},
    }


def _source_selector_contract_blockers(
    result: Mapping[str, Any],
) -> list[str]:
    adapt = result.get("adapt_vqe", {})
    adapt = dict(adapt) if isinstance(adapt, Mapping) else {}
    settings = result.get("settings", {})
    settings = dict(settings) if isinstance(settings, Mapping) else {}
    blockers: list[str] = []
    required_settings = {
        "adapt_reopt_policy": "windowed",
        "adapt_window_size": 3,
        "adapt_window_topk": 0,
        "adapt_full_refit_every": 8,
        "adapt_final_refit_maxiter": 200,
        "adapt_finite_angle_fallback": True,
        "adapt_finite_angle": 0.1,
        "adapt_finite_angle_min_improvement": 1.0e-12,
        "phase2_novelty_mode": "collective_span_v1",
        "phase2_enable_batching": False,
        "phase3_runtime_split_mode": "shortlist_pauli_children_v1",
        "phase3_runtime_split_selection_mode": "archival_child_set_forward_v1",
        "phase3_runtime_split_max_subset_size": 1,
        "phase3_runtime_split_child_set_symmetry_policy": "hard_guard",
        "phase1_shortlist_size": 24,
        "phase2_shortlist_size": 12,
        "phase2_shortlist_fraction": 0.25,
        "phase1_probe_max_positions": 6,
        "phase1_prune_policy": "recoverability_ladder_v1",
        "phase1_prune_mode": "both",
    }
    for key, expected in required_settings.items():
        if settings.get(key) != expected:
            blockers.append(f"setting:{key}")
    if str(settings.get("adapt_final_full_refit")).strip().lower() not in {
        "true",
        "1",
    }:
        blockers.append("setting:adapt_final_full_refit")
    if settings.get("phase3_geometry_window_size") not in {None, 0}:
        blockers.append("setting:phase3_geometry_window_size")

    overlay = adapt.get("historical_singleton_coordinate_trust_overlay", {})
    overlay = dict(overlay) if isinstance(overlay, Mapping) else {}
    required_overlay = {
        "active": True,
        "adaptive_trust_active": True,
        "admission_cardinality": 1,
        "coordinate_solve_policy": "supported_metric_whitened_eigh_v1",
        "phase0_pilot_enabled": False,
        "phase2_batching_enabled": False,
        "phase3_batching_enabled": False,
        "route_a_funnel_active": False,
        "whitening_active": True,
    }
    for key, expected in required_overlay.items():
        if overlay.get(key) != expected:
            blockers.append(f"coordinate_trust_overlay:{key}")
    trust_update = overlay.get("trust_region_update", {})
    if not isinstance(trust_update, Mapping) or trust_update.get("policy") != (
        "displacement_calibrated_unbounded_v2"
    ):
        blockers.append("coordinate_trust_overlay:trust_region_update")

    history = adapt.get("history", [])
    if not isinstance(history, list) or len(history) != EXPECTED_CONTROLLER_HORIZON:
        blockers.append("controller_horizon")
        history = []
    finite_angle_depths: list[int] = []
    finite_angle_candidate_scans = 0
    for row in history:
        if not isinstance(row, Mapping):
            blockers.append("history:row_type")
            continue
        if bool(row.get("phase3_batching_enabled", False)):
            blockers.append("history:phase3_batching_enabled")
        if int(row.get("batch_size", 1) or 1) != 1:
            blockers.append("history:non_singleton_batch")
        if len(row.get("selected_ops", [])) != 1:
            blockers.append("history:non_singleton_admission")
        scan_size = int(row.get("fallback_scan_size", 0) or 0)
        if scan_size:
            finite_angle_depths.append(int(row.get("depth", 0) or 0))
            finite_angle_candidate_scans += scan_size
        for feature in row.get("selected_feature_rows", []):
            if not isinstance(feature, Mapping):
                blockers.append("history:selected_feature_type")
                continue
            if feature.get("phase2_novelty_mode") != "collective_span_v1":
                blockers.append("history:phase2_novelty_mode")
            if feature.get("phase2_raw_score_formula") != (
                "DeltaE_TR_raw * N2 / (1 + K2)"
            ):
                blockers.append("history:phase2_n2_score_formula")
            if feature.get("phase3_canonical_score_formula") != (
                "DeltaE_TR * N3 / (1 + K3)"
            ):
                blockers.append("history:phase3_n3_score_formula")
    if finite_angle_depths != [23, 24, 25, 30]:
        blockers.append("finite_angle_fallback_depths")
    if finite_angle_candidate_scans != 388:
        blockers.append("finite_angle_fallback_scan_count")
    return sorted(set(blockers))


def _validate_anchor_against(
    observed_path: Path,
    reference_path: Path,
    *,
    role: str,
) -> dict[str, Any]:
    observed = _read_json(observed_path)
    reference = _read_json(reference_path)
    observed_adapt = dict(observed.get("adapt_vqe", {}))
    reference_adapt = dict(reference.get("adapt_vqe", {}))
    observed_energy = float(observed_adapt.get("energy"))
    reference_energy = float(reference_adapt.get("energy"))
    observed_error = float(observed_adapt.get("abs_delta_e"))
    reference_error = float(reference_adapt.get("abs_delta_e"))
    energy_delta = abs(observed_energy - reference_energy)
    error_delta = abs(observed_error - reference_error)
    observed_signature = _history_signature(observed)
    reference_signature = _history_signature(reference)
    observed_controller = _controller_signature(observed)
    reference_controller = _controller_signature(reference)
    blockers: list[str] = []
    if not all(
        math.isfinite(value)
        for value in (
            observed_energy,
            reference_energy,
            observed_error,
            reference_error,
            energy_delta,
            error_delta,
        )
    ):
        blockers.append("nonfinite_energy_or_error")
    if energy_delta > 1.0e-10:
        blockers.append(f"energy_delta={energy_delta:.3e}")
    if error_delta > 1.0e-10:
        blockers.append(f"abs_delta_e_delta={error_delta:.3e}")
    if observed_adapt.get("operators") != reference_adapt.get("operators"):
        blockers.append("terminal_operator_sequence_mismatch")
    if observed_signature != reference_signature:
        blockers.append("controller_history_signature_mismatch")
    if observed_controller != reference_controller:
        blockers.append("controller_or_query_work_signature_mismatch")
    blockers.extend(_source_selector_contract_blockers(observed))
    blockers = sorted(set(blockers))
    payload = {
        "schema": "paper_i_hh_sr_anchor_parity_validation_v1",
        "status": "pass" if not blockers else "blocked",
        "role": str(role),
        "observed_result": str(observed_path),
        "observed_sha256": _sha256(observed_path),
        "reference_result": str(reference_path),
        "reference_sha256": _sha256(reference_path),
        "energy_delta": float(energy_delta),
        "abs_delta_e_delta": float(error_delta),
        "history_rounds": len(observed_signature),
        "history_signature_sha256": _json_sha256(observed_signature),
        "controller_signature_sha256": _json_sha256(observed_controller),
        "query_work_signature": _query_work_signature(observed_adapt),
        "finite_angle_fallback_candidate_scan_count": int(
            sum(
                int(row.get("fallback_scan_size", 0) or 0)
                for row in observed_adapt.get("history", [])
                if isinstance(row, Mapping)
            )
        ),
        "finite_angle_fallback_objective_probe_count": int(
            2
            * sum(
                int(row.get("fallback_scan_size", 0) or 0)
                for row in observed_adapt.get("history", [])
                if isinstance(row, Mapping)
            )
        ),
        "blockers": blockers,
    }
    if blockers:
        raise ValueError(f"{role} parity blocked: {blockers}")
    return payload


def _validate_fm_result(
    result_path: Path,
    current_path: Path,
    *,
    regime: Mapping[str, Any],
    expected_horizon: int = EXPECTED_CONTROLLER_HORIZON,
) -> dict[str, Any]:
    result = _read_json(result_path)
    current = _read_json(current_path)
    adapt = dict(result.get("adapt_vqe", {}))
    identity = dict(adapt.get("static_route_identity", {}))
    closure = dict(adapt.get("formal_manifold_query_closure", {}))
    settings = dict(result.get("settings", {}))
    current_adapt = dict(current.get("adapt_vqe", {}))
    current_settings = dict(current.get("settings", {}))
    blockers: list[str] = []
    history = adapt.get("history", [])
    if adapt.get("success") is not True:
        blockers.append("result:success")
    if not isinstance(history, list) or not history:
        blockers.append("result:nonempty_history")
        history = []
    if len(history) != int(expected_horizon):
        blockers.append("result:terminal_horizon")
    if str(adapt.get("stop_reason")) != "max_depth":
        blockers.append("result:stop_reason")
    if int(adapt.get("ansatz_depth", 0) or 0) < 1:
        blockers.append("result:ansatz_depth")
    if not adapt.get("operators"):
        blockers.append("result:operators")
    expected_identity = {
        "route_family": "formal_manifold_snake",
        "route_profile": (
            FM_ROUTE_PROFILE_SR_SOURCE_LOCKED_SUPPORTED_WHITENED_ADAPTIVE_TRUST_V1
        ),
        "candidate_selector_family": "singleton_response_snake",
        "candidate_selector_profile": "supported_whitened_adaptive_trust_v1",
        "coordinate_solve_scope": "phase3_only_v1",
        "phase2_whitening_active": False,
        "phase3_whitening_active": True,
        "phase2_gram_novelty_policy": "ordinary_multiplier_v1",
        "phase3_gram_novelty_policy": "ordinary_multiplier_v1",
        "structural_rollback_enabled": False,
    }
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            blockers.append(f"identity:{key}")
    for key, expected in {
        "joint_response_selector_invoked": False,
        "formal_combinatorial_selector_invoked": False,
        "singleton_response_selector_invoked": True,
    }.items():
        if closure.get(key) is not expected:
            blockers.append(f"query_closure:{key}")
    if closure.get("candidate_scoring_policy") != (
        "singleton_response_snake_supported_whitened_adaptive_trust_v1"
    ):
        blockers.append("query_closure:candidate_scoring_policy")
    if closure.get("schema") != "formal_manifold_query_closure_telemetry_v1":
        blockers.append("query_closure:schema")
    if closure.get("route") != FORMAL_MANIFOLD_REOPTIMIZATION_ROUTE:
        blockers.append("query_closure:route")

    closure_checkpoint = adapt.get("formal_manifold_query_closure_checkpoint")
    if not isinstance(closure_checkpoint, Mapping):
        blockers.append("query_closure:checkpoint_missing")
    else:
        if closure_checkpoint.get("schema") != (
            "formal_manifold_query_closure_checkpoint_v1"
        ):
            blockers.append("query_closure:checkpoint_schema")
        if closure_checkpoint.get("current_round_finalized") is not True:
            blockers.append("query_closure:round_not_finalized")
        for raw_key, telemetry_key in (
            ("winning_branch_ledger", "winning_branch"),
            (
                "discarded_branch_operational_ledger",
                "discarded_branch_operational_overhead",
            ),
        ):
            raw_ledger = closure_checkpoint.get(raw_key)
            expected_telemetry = closure.get(telemetry_key)
            if not isinstance(raw_ledger, Mapping) or not isinstance(
                expected_telemetry, Mapping
            ):
                blockers.append(f"query_closure:{raw_key}")
                continue
            try:
                actual_telemetry = QueryPrimitiveLedger.from_checkpoint_payload(
                    raw_ledger
                ).telemetry()
            except (TypeError, ValueError) as exc:
                blockers.append(
                    f"query_closure:{raw_key}:{type(exc).__name__}"
                )
                continue
            if actual_telemetry != dict(expected_telemetry):
                blockers.append(f"query_closure:{raw_key}_telemetry_mismatch")
            reconciliation = actual_telemetry.get(
                "primitive_count_reconciliation", {}
            )
            if not isinstance(reconciliation, Mapping) or (
                reconciliation.get("count_equal") is not True
            ):
                blockers.append(f"query_closure:{raw_key}_count_reconciliation")
        winning = closure.get("winning_branch", {})
        if not isinstance(winning, Mapping) or int(
            winning.get("actual_operator_probe_count", 0) or 0
        ) < 1:
            blockers.append("query_closure:winning_branch_empty")

    config = settings.get("adapt_formal_manifold_config", {})
    if not isinstance(config, Mapping):
        blockers.append("formal_config_missing")
        config = {}
    else:
        if float(config.get("qbroyd_epsilon0", float("nan"))) != 0.0:
            blockers.append("formal_config:qbroyd_epsilon0")
        if int(config.get("line_search_max_steps", -1)) != 15:
            blockers.append("formal_config:line_search_max_steps")
        supported = config.get("supported_metric", {})
        if not isinstance(supported, Mapping) or supported.get("policy") != (
            "supported_metric_whitened_eigh_v1"
        ):
            blockers.append("formal_config:supported_metric_policy")
    config_sha256 = _json_sha256(dict(config)) if config else None

    selector = identity.get("formal_manifold_route_composition", {})
    expected_selector = {
        "additional_n3_multiplier_applied": True,
        "admission_cardinality": 1,
        "historical_singleton_coordinate_solve_policy": (
            "supported_metric_whitened_eigh_v1"
        ),
        "historical_singleton_coordinate_solve_scope": "phase3_only_v1",
        "historical_singleton_trust_region_update_policy": (
            "displacement_calibrated_unbounded_v2"
        ),
        "measured_n2_retained": True,
        "phase0_pilot_enabled": False,
        "phase2_enable_batching": False,
        "phase2_gram_novelty_policy": "ordinary_multiplier_v1",
        "phase2_novelty_mode": "collective_span_v1",
        "phase2_novelty_multiplier_policy": "legacy_ablation_mode_v1",
        "phase3_enable_batching": False,
        "phase3_gram_novelty_policy": "ordinary_multiplier_v1",
        "phase3_novelty_ablation_mode": "off",
        "phase3_novelty_multiplier_policy": "legacy_ablation_mode_v1",
        "phase3_runtime_split_subset_sizes": [1],
        "route_a_funnel_active": False,
        "structural_rollback_enabled": False,
    }
    if not isinstance(selector, Mapping):
        blockers.append("identity:singleton_response_selector")
    else:
        for key, expected in expected_selector.items():
            if selector.get(key) != expected:
                blockers.append(f"identity:selector:{key}")

    for row in history:
        if not isinstance(row, Mapping):
            blockers.append("history:row_type")
            continue
        step = row.get("formal_manifold_warm_start")
        if not isinstance(step, Mapping):
            blockers.append("history:formal_step_missing")
            continue
        if step.get("selector_reopt_policy") != "windowed":
            blockers.append("history:selector_reopt_policy")
        if step.get("accepted_reoptimizer_scope") != (
            "full_supported_manifold_v1"
        ):
            blockers.append("history:accepted_reoptimizer_scope")
        if step.get("schema") != "formal_manifold_reoptimization_result_v1":
            blockers.append("history:formal_step_schema")
        if step.get("qbang_momentum_active") is not False:
            blockers.append("history:qbang_momentum_active")
        if step.get("qbroyd_mode") != "shadow_predictor_exact_refresh_v1":
            blockers.append("history:qbroyd_mode")
        if step.get("supported_metric_whitening_policy") != (
            "supported_metric_whitened_eigh_v1"
        ):
            blockers.append("history:supported_metric_whitening_policy")
        if step.get("optimizer_coordinate_system") != (
            "supported_raw_fs_orthonormal_frame_v1"
        ):
            blockers.append("history:optimizer_coordinate_system")
        if step.get("shared_solver_coordinate_system") != (
            "supported_regularized_metric_v1"
        ):
            blockers.append("history:shared_solver_coordinate_system")
        for left, right, label in (
            ("whitening_id", "curvature_whitening_id", "curvature_whitening"),
            ("whitening_id", "qbroyd_whitening_id", "qbroyd_whitening"),
            ("frame_id", "curvature_frame_id", "curvature_frame"),
            (
                "logical_range_id",
                "qbroyd_logical_range_id",
                "qbroyd_logical_range",
            ),
        ):
            if not step.get(left) or step.get(left) != step.get(right):
                blockers.append(f"history:provenance:{label}")
        adapter = step.get("adapter", {})
        expected_count = int(step.get("accepted_reoptimizer_coordinate_count", 0) or 0)
        if (
            not isinstance(adapter, Mapping)
            or expected_count < 1
            or int(adapter.get("coordinate_count", -1)) != expected_count
            or len(adapter.get("coordinate_registry", [])) != expected_count
        ):
            blockers.append("history:accepted_coordinate_registry")
        receipt = step.get("query_closure_growth_receipt", {})
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("schema")
            != "formal_manifold_growth_geometry_receipt_v1"
            or receipt.get("zero_new_coordinates") is not True
            or receipt.get("metric_convention")
            != "raw_fubini_study_supported_metric_whitened_v1"
        ):
            blockers.append("history:zero_growth_receipt")
        for inner in step.get("steps", []):
            if not isinstance(inner, Mapping):
                blockers.append("history:inner_step_type")
                continue
            shadow = inner.get("qbroyd_shadow")
            if isinstance(shadow, Mapping) and float(
                shadow.get("epsilon", float("nan"))
            ) != 0.0:
                blockers.append("history:qbroyd_not_disabled")

    warm = adapt.get("formal_manifold_warm_start")
    warm_state = adapt.get("formal_manifold_warm_state_checkpoint")
    if not isinstance(warm, Mapping):
        blockers.append("formal_state:summary_missing")
    else:
        if warm.get("schema") != "formal_manifold_session_summary_v1":
            blockers.append("formal_state:summary_schema")
        if warm.get("qbang_momentum_active") is not False:
            blockers.append("formal_state:qbang_momentum_active")
        if warm.get("supported_metric_whitening") != (
            "supported_metric_whitened_eigh_v1"
        ):
            blockers.append("formal_state:supported_metric_whitening")
        if config_sha256 and warm.get("formal_manifold_config_sha256") != config_sha256:
            blockers.append("formal_state:config_sha256")
        state = warm.get("state", {})
        if not isinstance(state, Mapping) or state.get("active") is not True:
            blockers.append("formal_state:inactive")
        elif (
            state.get("valid_metric") is not True
            or state.get("valid_curvature") is not True
            or not state.get("whitening_id")
            or state.get("whitening_id") != state.get("curvature_whitening_id")
            or state.get("whitening_id") != state.get("qbroyd_whitening_id")
            or state.get("frame_id") != state.get("curvature_frame_id")
            or state.get("logical_range_id") != state.get("qbroyd_logical_range_id")
        ):
            blockers.append("formal_state:provenance")
    if not isinstance(warm_state, Mapping):
        blockers.append("formal_state:checkpoint_missing")
    else:
        if warm_state.get("schema") != "formal_manifold_warm_state_checkpoint_v1":
            blockers.append("formal_state:checkpoint_schema")
        if config_sha256 and warm_state.get("formal_manifold_config_sha256") != config_sha256:
            blockers.append("formal_state:checkpoint_config_sha256")
        if (
            not warm_state.get("whitening_id")
            or warm_state.get("whitening_id")
            != warm_state.get("curvature_whitening_id")
            or warm_state.get("whitening_id")
            != warm_state.get("qbroyd_whitening_id")
            or warm_state.get("frame_id") != warm_state.get("curvature_frame_id")
            or warm_state.get("logical_range_id")
            != warm_state.get("qbroyd_logical_range_id")
        ):
            blockers.append("formal_state:checkpoint_provenance")

    if current_settings.get("route_family") != "formal_manifold_snake":
        blockers.append("current:route_family")
    if current_settings.get("formal_manifold_route_profile") != (
        FM_ROUTE_PROFILE_SR_SOURCE_LOCKED_SUPPORTED_WHITENED_ADAPTIVE_TRUST_V1
    ):
        blockers.append("current:route_profile")
    current_config = current_settings.get("adapt_formal_manifold_config")
    if not isinstance(current_config, Mapping) or dict(current_config) != dict(config):
        blockers.append("current:formal_config")
    if int(current_adapt.get("history_count", 0) or 0) != int(expected_horizon):
        blockers.append("current:history_horizon")
    runtime_checkpoint = current_adapt.get("formal_manifold_runtime_checkpoint")
    if not isinstance(runtime_checkpoint, Mapping):
        blockers.append("current:runtime_checkpoint_missing")
    else:
        if runtime_checkpoint.get("schema") != (
            "formal_manifold_beam_branch_runtime_checkpoint_v1"
        ):
            blockers.append("current:runtime_checkpoint_schema")
        if runtime_checkpoint.get("structural_rollback_supported") is not False:
            blockers.append("current:structural_rollback")
        if runtime_checkpoint.get("rollback_scope") != "pending_proposal_only":
            blockers.append("current:rollback_scope")
        if config_sha256 and runtime_checkpoint.get(
            "formal_manifold_config_sha256"
        ) != config_sha256:
            blockers.append("current:config_sha256")
        checkpoint_warm = runtime_checkpoint.get("warm_state")
        if not isinstance(checkpoint_warm, Mapping):
            blockers.append("current:warm_state_missing")
        elif (
            not checkpoint_warm.get("whitening_id")
            or checkpoint_warm.get("whitening_id")
            != checkpoint_warm.get("curvature_whitening_id")
            or checkpoint_warm.get("whitening_id")
            != checkpoint_warm.get("qbroyd_whitening_id")
            or checkpoint_warm.get("frame_id")
            != checkpoint_warm.get("curvature_frame_id")
            or checkpoint_warm.get("logical_range_id")
            != checkpoint_warm.get("qbroyd_logical_range_id")
        ):
            blockers.append("current:warm_state_provenance")

    energy = float(adapt.get("energy"))
    reported_error = float(adapt.get("abs_delta_e"))
    expected_error = abs(energy - float(regime["exact_energy"]))
    if not all(math.isfinite(value) for value in (energy, reported_error, expected_error)):
        blockers.append("nonfinite_energy_or_error")
    if not math.isclose(expected_error, reported_error, rel_tol=0.0, abs_tol=1.0e-10):
        blockers.append("same_cutoff_abs_delta_e_mismatch")
    blockers = sorted(set(blockers))
    payload = {
        "schema": "paper_i_hh_fm_sr_source_locked_result_validation_v1",
        "status": "pass" if not blockers else "blocked",
        "run_class": "diagnostic",
        "regime": str(regime["id"]),
        "result": str(result_path),
        "result_sha256": _sha256(result_path),
        "current": str(current_path),
        "current_sha256": _sha256(current_path),
        "energy": float(energy),
        "abs_delta_e": float(reported_error),
        "controller_round_count": len(history),
        "expected_controller_horizon": int(expected_horizon),
        "ansatz_depth": int(adapt.get("ansatz_depth", len(adapt.get("operators", [])))),
        "formal_manifold_config_sha256": config_sha256,
        "winning_branch_query_work": closure.get("winning_branch"),
        "blockers": blockers,
    }
    if blockers:
        raise ValueError(f"FM result validation blocked: {blockers}")
    return payload


def _disk_preflight(repo_root: Path) -> dict[str, Any]:
    free_gib = float(shutil.disk_usage(repo_root).free / (1024**3))
    return {
        "free_gib": free_gib,
        "launch_floor_gib": DISK_LAUNCH_FLOOR_GIB,
        "status": "ok" if free_gib >= DISK_LAUNCH_FLOOR_GIB else "blocked",
    }


def _formal_config_payload() -> dict[str, Any]:
    return FormalManifoldConfig(
        qbroyd_epsilon0=0.0,
        line_search_max_steps=15,
    ).as_dict()


def _expected_stage_contracts(
    *,
    repo_root: Path,
    campaign_dir: Path,
    formal_config_path: Path,
) -> dict[str, dict[str, Any]]:
    source_argv = _source_argv(repo_root)
    archive_paths = _archive_paths(campaign_dir)
    live_paths = _direct_paths(campaign_dir, "current_sr_parity_anchor")
    weak_paths = _direct_paths(campaign_dir, "fm_qbroyd_off_weak_weak")
    intermediate_paths = _direct_paths(
        campaign_dir,
        "fm_qbroyd_off_intermediate_weak_transfer",
    )
    live_argv = build_live_anchor_argv(
        repo_root=repo_root,
        campaign_dir=campaign_dir,
    )
    weak_argv = build_fm_argv(
        repo_root=repo_root,
        campaign_dir=campaign_dir,
        stage_id="fm_qbroyd_off_weak_weak",
        regime=WEAK_WEAK,
        formal_config_json=formal_config_path,
    )
    intermediate_argv = build_fm_argv(
        repo_root=repo_root,
        campaign_dir=campaign_dir,
        stage_id="fm_qbroyd_off_intermediate_weak_transfer",
        regime=INTERMEDIATE_WEAK,
        formal_config_json=formal_config_path,
    )
    required = _required_authorities(repo_root)
    return {
        "archived_sr_anchor": {
            "order": 0,
            "kind": "archived_sr_anchor",
            "regime": dict(WEAK_WEAK),
            "argv": [
                sys.executable,
                str(required["archive_launcher"]),
                "--execute",
                "--output-root",
                str(archive_paths["root"]),
            ],
            "paths": {key: str(value) for key, value in archive_paths.items()},
            "semantic_diff_audit": None,
        },
        "current_sr_parity_anchor": {
            "order": 1,
            "kind": "current_sr_parity_anchor",
            "regime": dict(WEAK_WEAK),
            "argv": live_argv,
            "paths": {key: str(value) for key, value in live_paths.items()},
            "semantic_diff_audit": audit_semantic_diff(
                source_argv,
                live_argv,
                allowed_destinations=_OPERATIONAL_DESTINATIONS,
            ),
        },
        "fm_qbroyd_off_weak_weak": {
            "order": 2,
            "kind": "fm_source_locked_variant",
            "regime": dict(WEAK_WEAK),
            "argv": weak_argv,
            "paths": {key: str(value) for key, value in weak_paths.items()},
            "semantic_diff_audit": audit_semantic_diff(
                source_argv,
                weak_argv,
                allowed_destinations=(
                    _OPERATIONAL_DESTINATIONS | _FM_MECHANISM_DESTINATIONS
                ),
            ),
        },
        "fm_qbroyd_off_intermediate_weak_transfer": {
            "order": 3,
            "kind": "fm_cross_regime_policy_transfer",
            "regime": dict(INTERMEDIATE_WEAK),
            "argv": intermediate_argv,
            "paths": {
                key: str(value) for key, value in intermediate_paths.items()
            },
            "semantic_diff_audit": audit_semantic_diff(
                source_argv,
                intermediate_argv,
                allowed_destinations=_INTERMEDIATE_TRANSFER_DESTINATIONS,
            ),
        },
    }


def _revalidate_campaign_manifest(
    *,
    repo_root: Path,
    campaign_dir: Path,
    manifest: Mapping[str, Any],
) -> None:
    """Fail closed if a planned campaign's authorities or semantics drift."""

    if manifest.get("schema") != SCHEMA or manifest.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("Campaign identity drifted")
    if Path(str(manifest.get("repo_root", ""))).resolve() != repo_root:
        raise ValueError("Campaign repo_root drifted")
    if Path(str(manifest.get("campaign_dir", ""))).resolve() != campaign_dir:
        raise ValueError("Campaign directory drifted")
    if manifest.get("scientific_concurrency") != 1:
        raise ValueError("Campaign scientific concurrency drifted")

    required = _validate_source_authorities(repo_root)
    live_lock = _live_implementation_lock(repo_root)
    if manifest.get("live_implementation_lock") != live_lock:
        raise ValueError("Campaign live implementation lock drifted")
    recorded_lock = manifest.get("source_lock")
    if not isinstance(recorded_lock, Mapping):
        raise ValueError("Campaign source_lock missing")
    for key, path in required.items():
        record = recorded_lock.get(key)
        if not isinstance(record, Mapping):
            raise ValueError(f"Campaign source_lock missing {key}")
        if Path(str(record.get("path", ""))).resolve() != path.resolve():
            raise ValueError(f"Campaign source_lock path drifted for {key}")
        if record.get("sha256") != _sha256(path):
            raise ValueError(f"Campaign source_lock hash drifted for {key}")

    formal_record = manifest.get("formal_config")
    if not isinstance(formal_record, Mapping):
        raise ValueError("Campaign formal config record missing")
    formal_path = (campaign_dir / "formal_manifold_config.qbroyd_off.json").resolve()
    if Path(str(formal_record.get("path", ""))).resolve() != formal_path:
        raise ValueError("Campaign formal config path drifted")
    if not formal_path.is_file():
        raise FileNotFoundError(formal_path)
    if _read_json(formal_path) != _formal_config_payload():
        raise ValueError("Campaign formal config payload drifted")
    if formal_record.get("sha256") != _sha256(formal_path):
        raise ValueError("Campaign formal config hash drifted")
    if (
        formal_record.get("qbroyd_epsilon0") != 0.0
        or formal_record.get("line_search_max_steps") != 15
        or formal_record.get("qbang_momentum_active") is not False
    ):
        raise ValueError("Campaign formal config summary drifted")

    expected = _expected_stage_contracts(
        repo_root=repo_root,
        campaign_dir=campaign_dir,
        formal_config_path=formal_path,
    )
    stages = manifest.get("stages")
    if not isinstance(stages, list) or [stage.get("id") for stage in stages] != list(
        expected
    ):
        raise ValueError("Campaign stage ids or order drifted")
    for stage in stages:
        stage_id = str(stage["id"])
        contract = expected[stage_id]
        for key in ("order", "kind", "regime", "argv", "paths"):
            if stage.get(key) != contract[key]:
                raise ValueError(f"Campaign stage {stage_id} {key} drifted")
        expected_audit = contract["semantic_diff_audit"]
        if expected_audit is not None and stage.get("semantic_diff_audit") != (
            expected_audit
        ):
            raise ValueError(f"Campaign stage {stage_id} semantic audit drifted")
    transfer = stages[-1]
    if transfer.get("source_lock_claim") is not False or transfer.get(
        "transfer_policy_source"
    ) != "fm_qbroyd_off_weak_weak":
        raise ValueError("Intermediate transfer labeling drifted")


def initialize_campaign(*, repo_root: Path, campaign_dir: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    campaign_dir = campaign_dir.resolve()
    manifest_path = campaign_dir / "campaign_manifest.json"
    if manifest_path.exists():
        existing = _read_json(manifest_path)
        _revalidate_campaign_manifest(
            repo_root=repo_root,
            campaign_dir=campaign_dir,
            manifest=existing,
        )
        return existing
    if campaign_dir.exists() and any(campaign_dir.iterdir()):
        raise FileExistsError(f"Refusing to clobber nonempty campaign: {campaign_dir}")

    required = _validate_source_authorities(repo_root)
    source_result = _read_json(required["source_result"])
    source_error = float(source_result["adapt_vqe"]["abs_delta_e"])

    campaign_dir.mkdir(parents=True, exist_ok=True)
    formal_config_path = campaign_dir / "formal_manifold_config.qbroyd_off.json"
    _write_json(formal_config_path, _formal_config_payload())
    contracts = _expected_stage_contracts(
        repo_root=repo_root,
        campaign_dir=campaign_dir,
        formal_config_path=formal_config_path,
    )
    stages: list[dict[str, Any]] = []
    for stage_id, contract in contracts.items():
        stage = {"id": stage_id, "status": "queued", **contract}
        if contract["semantic_diff_audit"] is None:
            stage.pop("semantic_diff_audit")
        if stage_id == "fm_qbroyd_off_intermediate_weak_transfer":
            stage["source_lock_claim"] = False
            stage["transfer_policy_source"] = "fm_qbroyd_off_weak_weak"
        stages.append(stage)
    payload = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "status": "queued",
        "run_class": "diagnostic",
        "scientific_concurrency": 1,
        "created_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "repo_root": str(repo_root),
        "campaign_dir": str(campaign_dir),
        "source_lock": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in required.items()
        },
        "live_implementation_lock": _live_implementation_lock(repo_root),
        "source_result_abs_delta_e": source_error,
        "formal_config": {
            "path": str(formal_config_path),
            "sha256": _sha256(formal_config_path),
            "qbroyd_epsilon0": 0.0,
            "line_search_max_steps": 15,
            "qbang_momentum_active": False,
        },
        "mechanism_family_changed": "accepted_ansatz_reoptimization",
        "selector_policy_held": (
            "paper_i_sr_source_locked_supported_whitened_adaptive_trust_v1"
        ),
        "stages": stages,
    }
    _write_json(manifest_path, payload)
    _revalidate_campaign_manifest(
        repo_root=repo_root,
        campaign_dir=campaign_dir,
        manifest=payload,
    )
    return payload


def _write_stage_command(stage: Mapping[str, Any]) -> None:
    paths = {key: Path(value) for key, value in dict(stage["paths"]).items()}
    if str(stage["kind"]) == "archived_sr_anchor":
        return
    _write_json(
        paths["command"],
        {
            "schema": "paper_i_hh_fm_sr_source_locked_stage_command_v1",
            "stage_id": str(stage["id"]),
            "kind": str(stage["kind"]),
            "regime": dict(stage["regime"]),
            "argv": [str(value) for value in stage["argv"]],
            "working_directory": str(Path(stage.get("repo_root", "."))),
            "semantic_diff_audit": stage.get("semantic_diff_audit"),
        },
    )


def _normalized_manifest(
    stage: Mapping[str, Any],
    validation: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    return {
        "schema": "normalized_scientific_run_manifest_v1",
        "stage_id": str(stage["id"]),
        "run_class": "diagnostic",
        "route_family": (
            "formal_manifold_snake"
            if str(stage["kind"]).startswith("fm_")
            else "singleton_response_snake"
        ),
        "regime": dict(stage["regime"]),
        "result": str(paths["result"]),
        "result_sha256": _sha256(paths["result"]),
        "current": str(paths["current"]),
        "current_sha256": _sha256(paths["current"]),
        "estimator_ledger": str(paths["estimator_ledger"]),
        "estimator_ledger_sha256": _sha256(paths["estimator_ledger"]),
        "validation": dict(validation),
    }


def run_campaign(*, repo_root: Path, campaign_dir: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    campaign_dir = campaign_dir.resolve()
    manifest_path = campaign_dir / "campaign_manifest.json"
    manifest = initialize_campaign(repo_root=repo_root, campaign_dir=campaign_dir)
    _revalidate_campaign_manifest(
        repo_root=repo_root,
        campaign_dir=campaign_dir,
        manifest=manifest,
    )
    lock_path = campaign_dir / ".campaign.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Campaign lock already exists: {lock_path}") from exc
    with os.fdopen(lock_fd, "w", encoding="utf-8") as handle:
        handle.write(f"pid={os.getpid()}\nstarted_utc={_utc_now()}\n")

    try:
        manifest["status"] = "running"
        manifest["updated_utc"] = _utc_now()
        _write_json(manifest_path, manifest)
        for stage in manifest["stages"]:
            _revalidate_campaign_manifest(
                repo_root=repo_root,
                campaign_dir=campaign_dir,
                manifest=manifest,
            )
            if stage.get("status") == "complete":
                continue
            paths = {
                key: Path(value) for key, value in dict(stage["paths"]).items()
            }
            if paths["result"].exists():
                raise RuntimeError(
                    f"Refusing implicit restart for nonterminal stage {stage['id']}"
                )
            disk = _disk_preflight(repo_root)
            if disk["status"] != "ok":
                raise RuntimeError(f"Disk preflight blocked launch: {disk}")
            if str(stage["kind"]) != "archived_sr_anchor":
                paths["root"].mkdir(parents=True, exist_ok=True)
                _write_stage_command({**stage, "repo_root": str(repo_root)})
            stage["status"] = "running"
            stage["started_utc"] = _utc_now()
            stage["disk_preflight"] = disk
            manifest["active_stage"] = str(stage["id"])
            manifest["updated_utc"] = _utc_now()
            _write_json(manifest_path, manifest)

            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            environment["STATIC_ADAPT_CANDIDATE_RECORD_CACHE_DIR"] = str(
                paths["root"] / "cache" / "candidate_records"
            )
            with paths["stdout"].open("ab") as stdout_handle, paths[
                "stderr"
            ].open("ab") as stderr_handle:
                completed = subprocess.run(
                    [str(value) for value in stage["argv"]],
                    cwd=repo_root,
                    env=environment,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    check=False,
                )
            stage["returncode"] = int(completed.returncode)
            stage["finished_utc"] = _utc_now()
            if int(completed.returncode) != 0:
                raise RuntimeError(
                    f"Stage {stage['id']} exited {completed.returncode}"
                )
            _revalidate_campaign_manifest(
                repo_root=repo_root,
                campaign_dir=campaign_dir,
                manifest=manifest,
            )

            if str(stage["kind"]) == "archived_sr_anchor":
                validation = _validate_anchor_against(
                    paths["result"],
                    repo_root / SOURCE_RESULT,
                    role="immutable_archive_anchor",
                )
            elif str(stage["kind"]) == "current_sr_parity_anchor":
                archive_result = Path(
                    manifest["stages"][0]["paths"]["result"]
                )
                validation = _validate_anchor_against(
                    paths["result"],
                    archive_result,
                    role="current_code_sr_parity_anchor",
                )
            else:
                validation = _validate_fm_result(
                    paths["result"],
                    paths["current"],
                    regime=dict(stage["regime"]),
                )
            _write_json(paths["validation"], validation)
            normalized = _normalized_manifest(stage, validation, paths)
            _write_json(paths["normalized_manifest"], normalized)
            stage["validation"] = validation
            stage["status"] = "complete"
            manifest["updated_utc"] = _utc_now()
            _write_json(manifest_path, manifest)

        manifest["status"] = "complete"
        manifest["active_stage"] = None
        manifest["finished_utc"] = _utc_now()
        manifest["updated_utc"] = _utc_now()
        _write_json(manifest_path, manifest)
        return manifest
    except Exception as exc:
        active_id = manifest.get("active_stage")
        for stage in manifest.get("stages", []):
            if str(stage.get("id")) == str(active_id):
                stage["status"] = "failed"
                stage["error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
        manifest["status"] = "blocked"
        manifest["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        manifest["updated_utc"] = _utc_now()
        _write_json(manifest_path, manifest)
        raise
    finally:
        lock_path.unlink(missing_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    campaign_dir = args.campaign_dir.expanduser().resolve()
    payload = (
        initialize_campaign(repo_root=repo_root, campaign_dir=campaign_dir)
        if args.command == "plan"
        else run_campaign(repo_root=repo_root, campaign_dir=campaign_dir)
    )
    print(
        json.dumps(
            {
                "campaign_id": payload["campaign_id"],
                "status": payload["status"],
                "manifest": str(campaign_dir / "campaign_manifest.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
