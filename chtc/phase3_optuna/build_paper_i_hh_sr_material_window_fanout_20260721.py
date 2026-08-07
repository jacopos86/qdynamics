#!/usr/bin/env python3
"""Build the six-regime Phase-III material-window candidate fanout.

This builder is deliberately unusable without a fetched, passing parent anchor.
It verifies the anchor's exact operator sequence, controller energies, checkpoint
sequence, terminal values, and normalized settings against the locked no-overlap
weak-weak source before producing any fanout artifact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import build_paper_i_hh_sr_material_window_anchor_20260721 as anchor


ROOT = anchor.ROOT
BASE = anchor.BASE
BASE_ID = anchor.BASE_ID
BASE_BATCH = anchor.BASE_BATCH
OUTPUT_ID = (
    "paper_i_hh_sr_snake_no_overlap_trust_material_window_all_six_"
    "r50_20260721_v1_chtc"
)
OUTPUT_BATCH = "paper-i-hh-sr-material-window-six-r50-20260721-v1"
OUTPUT = anchor.INPUT / OUTPUT_ID
MATERIAL_SCOPE = anchor.CHILD_SCOPE
FULL_SCOPE = anchor.PARENT_SCOPE
ANCHOR_SOURCE_ARCHIVE_SHA256 = (
    "ced6b10d6bfbe4ae6a54495ff2ef4747a90036fa2027b0386555d016d5869a05"
)
SUPPORT_CHANGE_POLICY = (
    "full_geometry_refresh_on_unexpected_supported_nullity_drift_v1"
)
EXPECTED_REGIME_MATRIX: dict[str, dict[str, Any]] = {
    "weak_weak": {
        "u_over_t": 0.25, "lambda": 0.25, "g_ep": 0.353553390593, "n_ph": 3,
    },
    "intermediate_weak": {
        "u_over_t": 1.25, "lambda": 0.25, "g_ep": 0.353553390593, "n_ph": 3,
    },
    "strong_weak_u8": {
        "u_over_t": 8.0, "lambda": 0.25, "g_ep": 0.353553390593, "n_ph": 3,
    },
    "weak_strong": {
        "u_over_t": 0.25, "lambda": 1.25, "g_ep": 0.790569415042, "n_ph": 7,
    },
    "intermediate_strong": {
        "u_over_t": 1.25, "lambda": 1.25, "g_ep": 0.790569415042, "n_ph": 7,
    },
    "strong_strong_u8": {
        "u_over_t": 8.0, "lambda": 1.25, "g_ep": 0.790569415042, "n_ph": 7,
    },
}


def _validate_regime_contract(job: Mapping[str, Any]) -> None:
    """Fail closed on any regime/cutoff/horizon drift in a fanout row."""

    slug = str(job.get("regime_slug") or "")
    if slug not in EXPECTED_REGIME_MATRIX:
        raise ValueError(f"unexpected Paper-I regime: {slug!r}")
    expected = EXPECTED_REGIME_MATRIX[slug]
    physics = job.get("physics")
    segment = job.get("segment")
    if not isinstance(physics, Mapping) or not isinstance(segment, Mapping):
        raise TypeError(f"{slug}: physics/segment contract is not an object")
    for key in ("u_over_t", "lambda", "g_ep"):
        if not math.isclose(
            float(physics.get(key, float("nan"))), float(expected[key]),
            rel_tol=0.0, abs_tol=1.0e-12,
        ):
            raise ValueError(f"{slug}: exact {key} drift")
    n_ph = int(expected["n_ph"])
    if (
        physics.get("same_cutoff_reference") is not True
        or int(physics.get("n_ph_work", -1)) != n_ph
        or int(physics.get("n_ph_reference", -1)) != n_ph
    ):
        raise ValueError(f"{slug}: same-cutoff n_ph={n_ph} contract drift")
    if any(
        int(segment.get(key, -1)) != 50
        for key in (
            "target_controller_round", "target_depth", "max_new_admissions",
        )
    ):
        raise ValueError(f"{slug}: exact 50-round/admission horizon drift")
    argv = list(job.get("command", {}).get("argv", []))
    expected_argv = {
        "--u": expected["u_over_t"],
        "--g-ep": expected["g_ep"],
        "--n-ph-max": n_ph,
        "--adapt-max-depth": 50,
        "--adapt-segment-target-controller-round": 50,
        "--adapt-segment-target-depth": 50,
        "--adapt-segment-max-new-admissions": 50,
    }
    for flag, value in expected_argv.items():
        if flag not in argv:
            raise ValueError(f"{slug}: command is missing {flag}")
        actual = argv[argv.index(flag) + 1]
        if isinstance(value, int):
            matches = int(actual) == value
        else:
            matches = math.isclose(
                float(actual), float(value), rel_tol=0.0, abs_tol=1.0e-12
            )
        if not matches:
            raise ValueError(f"{slug}: command {flag} drift")


def _read_tar_member_bytes(archive: Path, member: str) -> bytes:
    with tarfile.open(archive, "r:gz") as handle:
        matches = [item for item in handle.getmembers() if item.name == member]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one anchor member {member!r}; got {len(matches)}"
            )
        extracted = handle.extractfile(matches[0])
        if extracted is None:
            raise ValueError(f"anchor member is not a regular file: {member}")
        return extracted.read()


def _locked_anchor_source(
    temp: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Materialize only the exact tested anchor-v2 source archive.

    Live-tree source construction is intentionally forbidden: another study may
    be editing the working tree while this builder runs.
    """

    archive = anchor.ANCHOR / "source_locked.tar.gz"
    bundle_receipt = anchor.load(anchor.ANCHOR / "anchor_bundle_receipt.json")
    archive_manifest = anchor.load(anchor.ANCHOR / "source_archive_manifest.json")
    if (
        bundle_receipt.get("source_archive_sha256")
        != ANCHOR_SOURCE_ARCHIVE_SHA256
        or bundle_receipt.get("parent_route_contract_sha256")
        != anchor.PARENT_DIGEST
        or bundle_receipt.get("candidate_route_contract_sha256")
        != anchor.CHILD_DIGEST
        or bundle_receipt.get("candidate_not_executed") is not True
        or archive_manifest.get("archive_sha256")
        != ANCHOR_SOURCE_ARCHIVE_SHA256
        or not archive.is_file()
        or anchor.sha256(archive) != ANCHOR_SOURCE_ARCHIVE_SHA256
    ):
        raise ValueError("tested material-window anchor source authority drift")
    files = archive_manifest.get("files")
    overlay = archive_manifest.get("material_window_source_overlay", {})
    overlays = overlay.get("overlay_files") if isinstance(overlay, Mapping) else None
    if not isinstance(files, dict) or not files or not isinstance(overlays, dict):
        raise ValueError("anchor archive inventory/overlay authority is incomplete")
    source = temp / "source"
    source.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(source, filter="data")
    for relative, record in files.items():
        path = source / relative
        if (
            not path.is_file()
            or anchor.sha256(path) != record.get("sha256")
            or path.stat().st_size != int(record.get("size_bytes", -1))
        ):
            raise ValueError(f"anchor source inventory drift: {relative}")
    contracts = anchor.isolated_contracts(source)
    return archive, copy.deepcopy(files), copy.deepcopy(overlays), contracts


def _is_sha256_text(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _validate_anchor_validation_receipt(
    *, validation: Mapping[str, Any], result_sha256: str,
) -> dict[str, Any]:
    """Validate the exact fetched-anchor receipt schema by substance.

    The fetched validation has a top-level ``status=pass``.  Its nested
    ``scientific_evidence_validation`` is a metrics/closure object and
    intentionally has no status field, so this gate validates its scientific
    invariants and estimator-ledger closure directly.
    """
    validation_schema = validation.get("schema")
    allowed_schemas = {
        "paper_i_hh_sr_symcost_noprune_validation_v1",
        "paper_i_hh_sr_symcost_noprune_fetched_validation_v1",
    }
    if (
        validation_schema not in allowed_schemas
        or validation.get("status") != "pass"
        or validation.get("result_sha256") != result_sha256
    ):
        raise ValueError("anchor top-level validation receipt drift")
    if validation_schema == "paper_i_hh_sr_symcost_noprune_fetched_validation_v1" and (
        validation.get("ledger_schema")
        != "paper_i_estimator_call_ledger_sidecar_v1"
        or validation.get("profile_contract_sha256") != anchor.PARENT_DIGEST
        or int(validation.get("target_controller_round", -1)) != 50
    ):
        raise ValueError("anchor fetched-validation provenance drift")
    scientific = validation.get("scientific_evidence_validation")
    projected = validation.get("projected_generalized_phase3_validation")
    no_overlap = validation.get("no_overlap_trust_validation")
    if not all(isinstance(item, Mapping) for item in (
        scientific, projected, no_overlap,
    )):
        raise ValueError("anchor validation subreceipt schema drift")

    prefix = scientific.get("active_prefix_estimator_ledger_receipts")
    ledger = scientific.get("ledger")
    if not isinstance(prefix, Mapping) or not isinstance(ledger, Mapping):
        raise ValueError("anchor estimator-ledger closure receipts are missing")
    s_alg = int(prefix.get("S_alg", -1))
    raw_occurrences = int(prefix.get("raw_occurrence_count", -1))
    if (
        prefix.get("schema")
        != "paper_i_active_prefix_estimator_ledger_closure_v1"
        or prefix.get("closure_passed") is not True
        or int(prefix.get("round_receipt_count", -1)) != 50
        or int(prefix.get("terminal_receipt_count", -1)) != 1
        or int(prefix.get("receipt_count", -1)) != 51
        or s_alg <= 0
        or raw_occurrences < s_alg
        or int(ledger.get("all_branch_s_alg", -1)) != s_alg
        or int(ledger.get("winning_lineage_s_alg", -1)) != s_alg
        or int(ledger.get("raw_entry_count", -1)) != s_alg
        or int(ledger.get("raw_occurrence_count", -1)) != raw_occurrences
        or int(ledger.get("finite_angle_guard_occurrence_count", -1)) != 0
        or not _is_sha256_text(ledger.get("ledger_fingerprint"))
    ):
        raise ValueError("anchor estimator-ledger closure drift")

    fallback_rounds = scientific.get(
        "infeasible_model_fallback_controller_rounds"
    )
    if not isinstance(fallback_rounds, list):
        raise ValueError("anchor fallback controller-round receipt is missing")
    fallback_rounds = [int(value) for value in fallback_rounds]
    fallback_count = int(
        scientific.get("infeasible_model_fallback_activation_count", -1)
    )
    leakage_values = (
        float(scientific.get("max_binary_padding_leakage", math.inf)),
        float(scientific.get("max_fixed_sector_leakage", math.inf)),
    )
    curvature_count = int(scientific.get("phase2_full_candidate_occurrences", -1))
    if (
        int(scientific.get("controller_rounds", -1)) != 50
        or int(scientific.get("new_admissions", -1)) != 50
        or int(scientific.get("final_active_depth", -1)) != 50
        or int(scientific.get("adaptive_trust_updates", -1)) != 50
        or scientific.get("phase3_response_scope")
        != "full_active_plus_singleton_v1"
        or scientific.get("supported_rank_recorded_each_round") is not True
        or scientific.get("terminal_state_unchanged_from_last_ordinary_round")
        is not True
        or not _is_sha256_text(scientific.get("terminal_checkpoint_sha256"))
        or int(scientific.get("prune_rounds_executed", -1)) != 0
        or scientific.get("ordinary_phase2_novelty_multiplier_active") is not False
        or scientific.get("ordinary_phase3_novelty_multiplier_active") is not False
        or int(scientific.get("phase1_lambda_f_proxy_occurrences", -1)) != 0
        or int(scientific.get("phase2_lambda_f_proxy_occurrences", -1)) != 0
        or int(scientific.get(
            "phase2_missing_curvature_fallback_occurrences", -1
        )) != 0
        or curvature_count <= 0
        or int(scientific.get(
            "validated_phase2_curvature_receipt_occurrences", -1
        )) != curvature_count
        or any(not math.isfinite(value) or value > 1.0e-10 for value in leakage_values)
        or fallback_count != len(fallback_rounds)
        or fallback_rounds != sorted(set(fallback_rounds))
        or any(not 1 <= value <= 50 for value in fallback_rounds)
        or scientific.get("infeasible_model_fallback_enabled") is not True
        or scientific.get("infeasible_model_fallback_fired")
        is not bool(fallback_count)
    ):
        raise ValueError("anchor substantive scientific validation drift")

    projected_feasible = int(projected.get("feasible_solver_receipt_count", -1))
    projected_infeasible = int(
        projected.get("infeasible_solver_receipt_count", -1)
    )
    if (
        projected.get("schema")
        != "paper_i_sr_projected_generalized_phase3_evidence_v1"
        or projected.get("status") != "pass"
        or int(projected.get("controller_rounds", -1)) != 50
        or int(projected.get("projected_solver_receipt_count", -1)) != 50
        or projected_feasible < 0
        or projected_infeasible != fallback_count
        or projected_feasible + projected_infeasible != 50
        or int(projected.get("projection_provenance_count", -1))
        != projected_feasible
        or projected.get("supported_metric_whitening_active") is not False
        or projected.get("accepted_powell_refit_whitening_active") is not True
        or int(projected.get("classical_quantum_query_charge", -1)) != 0
    ):
        raise ValueError("anchor projected Phase-III validation drift")

    expansion_count = int(no_overlap.get("expansion_count", -1))
    contraction_count = int(no_overlap.get("contraction_count", -1))
    hold_count = int(no_overlap.get("hold_count", -1))
    if (
        no_overlap.get("schema")
        != "paper_i_sr_source_metric_no_overlap_trust_evidence_v1"
        or no_overlap.get("status") != "pass"
        or int(no_overlap.get("controller_rounds", -1)) != 50
        or expansion_count < 0
        or contraction_count < 0
        or hold_count < 0
        or expansion_count + contraction_count + hold_count != 50
        or int(no_overlap.get("geometry_expansion_no_overlap_hold_count", -1))
        != fallback_count
        or int(no_overlap.get("initial_zero_active_no_overlap_hold_count", -1))
        != 1
        or int(no_overlap.get("source_metric_receipt_count", -1))
        + fallback_count + 1 != 50
        or int(no_overlap.get(
            "source_metric_displacement_unresolved_hold_count", -1
        )) != 0
        or int(no_overlap.get(
            "source_metric_transaction_failure_hold_count", -1
        )) != 0
        or int(no_overlap.get("endpoint_overlap_measurement_count", -1)) != 0
        or int(no_overlap.get("endpoint_overlap_query_charge", -1)) != 0
        or no_overlap.get("accepted_powell_refit_whitening_active") is not True
    ):
        raise ValueError("anchor no-overlap trust validation drift")
    return {
        "schema": "paper_i_material_window_anchor_validation_contract_v1",
        "status": "pass",
        "validation_schema": validation_schema,
        "controller_rounds": 50,
        "S_alg": s_alg,
        "raw_occurrence_count": raw_occurrences,
        "fallback_count": fallback_count,
        "projected_feasible_count": projected_feasible,
        "endpoint_overlap_query_charge": 0,
    }


def _anchor_evidence(
    *, result_path: Path, validation_path: Path, transfer_path: Path,
) -> dict[str, Any]:
    for path in (result_path, validation_path, transfer_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_result, source_result_sha = anchor.read_json_member(
        anchor.SOURCE_TRANSFER, anchor.SOURCE_RESULT_MEMBER
    )
    prefix = f"raw_outputs/{anchor.ANCHOR_ID}/weak_weak"
    members = {
        "result": f"{prefix}/json/result.json",
        "validation": f"{prefix}/validation.json",
        "normalized": f"{prefix}/normalized_run_manifest.json",
        "execution": f"{prefix}/execution.json",
    }
    raw = {
        name: _read_tar_member_bytes(transfer_path, member)
        for name, member in members.items()
    }
    result = json.loads(raw["result"])
    validation = json.loads(raw["validation"])
    normalized = json.loads(raw["normalized"])
    execution = json.loads(raw["execution"])
    if not all(isinstance(item, dict) for item in (
        result, validation, normalized, execution,
    )):
        raise TypeError("anchor transfer provenance members must be JSON objects")
    result_sha = anchor.sha256_bytes(raw["result"])
    validation_sha = anchor.sha256_bytes(raw["validation"])
    if anchor.sha256(result_path) != result_sha:
        raise ValueError("supplied anchor result is not the transfer member")
    if anchor.sha256(validation_path) != validation_sha:
        raise ValueError("supplied anchor validation is not the transfer member")
    anchor_validation_contract = _validate_anchor_validation_receipt(
        validation=validation, result_sha256=result_sha,
    )
    route = normalized.get("route_identity", {})
    source_lock = normalized.get("source_lock", {})
    locked_parent_profile = anchor.load(
        anchor.ANCHOR / "anchor_bundle_receipt.json"
    ).get("parent_route_profile")
    if (
        normalized.get("job_manifest") != "jobs/weak_weak.json"
        and not str(normalized.get("job_manifest") or "").endswith(
            "/jobs/weak_weak.json"
        )
    ):
        raise ValueError("anchor normalized manifest job identity drift")
    if (
        route.get("profile_request") != anchor.PARENT_ALIAS
        or route.get("profile_resolved")
        != locked_parent_profile
        or route.get("profile_contract_sha256") != anchor.PARENT_DIGEST
        or source_lock.get("source_archive_sha256")
        != ANCHOR_SOURCE_ARCHIVE_SHA256
        or normalized.get("physics", {}).get("same_cutoff_reference") is not True
        or int(normalized.get("physics", {}).get("n_ph_work", -1)) != 3
        or int(normalized.get("physics", {}).get("n_ph_reference", -1)) != 3
        or int(normalized.get("segment", {}).get("target_controller_round", -1))
        != 50
        or int(normalized.get("segment", {}).get("max_new_admissions", -1))
        != 50
    ):
        raise ValueError("anchor normalized manifest route/source/regime drift")
    artifacts = execution.get("artifacts", {})
    if (
        execution.get("status") != "completed"
        or int(execution.get("exit_code", -1)) != 0
        or artifacts.get("result_json", {}).get("sha256") != result_sha
        or artifacts.get("validation_json", {}).get("sha256") != validation_sha
        or execution.get("route_identity") != normalized.get("route_identity")
        or execution.get("physics") != normalized.get("physics")
        or execution.get("segment") != normalized.get("segment")
    ):
        raise ValueError("anchor execution provenance is not terminal/closed")
    source_signature = anchor.result_signature(source_result)
    result_signature = anchor.result_signature(result)
    comparisons = {
        "operator_sequence_match": (
            result_signature["operators"] == source_signature["operators"]
        ),
        "controller_energy_history_exact_match": (
            result_signature["controller_energies"]
            == source_signature["controller_energies"]
        ),
        "checkpoint_sequence_match": (
            result_signature["checkpoint_sha256_sequence"]
            == source_signature["checkpoint_sha256_sequence"]
        ),
        "terminal_metric_exact_match": (
            result_signature["terminal_energy"] == source_signature["terminal_energy"]
            and result_signature["terminal_abs_delta_e"]
            == source_signature["terminal_abs_delta_e"]
        ),
        "settings_exact_match": (
            result_signature["settings"] == source_signature["settings"]
        ),
    }
    if not all(comparisons.values()):
        raise ValueError(f"parent anchor does not reproduce source: {comparisons}")
    return {
        "anchor_reproduces_source": True,
        **comparisons,
        "source_result_sha256": source_result_sha,
        "anchor_result_json": str(result_path.resolve()),
        "anchor_result_sha256": result_sha,
        "anchor_validation_receipt": str(validation_path.resolve()),
        "anchor_validation_receipt_sha256": validation_sha,
        "anchor_transfer_archive": str(transfer_path.resolve()),
        "anchor_transfer_archive_sha256": anchor.sha256(transfer_path),
        "anchor_transfer_exact_members": {
            name: {"path": members[name], "sha256": anchor.sha256_bytes(raw[name])}
            for name in sorted(members)
        },
        "anchor_validation_contract": anchor_validation_contract,
        "anchor_source_archive_sha256": ANCHOR_SOURCE_ARCHIVE_SHA256,
        "operator_sequence_sha256": result_signature["operator_sequence_sha256"],
        "controller_energies_sha256": result_signature["controller_energies_sha256"],
        "checkpoint_sequence_sha256": result_signature["checkpoint_sequence_sha256"],
        "settings_sha256": result_signature["settings_sha256"],
        "terminal_energy": result_signature["terminal_energy"],
        "terminal_abs_delta_e": result_signature["terminal_abs_delta_e"],
        "non_swept_settings_diff": [],
    }


def _material_validator_extension() -> str:
    """Return strict material-window validation added to the recovered validator."""

    return r'''

import copy

MATERIAL_WINDOW_SCOPE = "candidate_material_coupling_window_v1"
MATERIAL_WINDOW_POLICY = {
    "policy_version": "phase3_material_window_policy_v1",
    "gram_entry_threshold": 4.0e-3,
    "hessian_entry_threshold": 2.0e-22,
    "gram_omitted_l2_tolerance": 1.0,
    "hessian_omitted_l2_tolerance": 1.0,
    "gram_cross_block_tolerance": 1.0e-1,
    "hessian_cross_block_tolerance": 1.0e-1,
    "epsilon": 1.0e-12,
}


def material_parent_validation_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Make a validation-only full-scope view for inherited non-window gates.

    The returned copy is used only by ``validate_parent_evidence`` for its
    checkpoint, leakage, Phase-I/II, pruning-off, and estimator-ledger gates.
    The untouched result is independently and fail-closed validated below for
    every material-window field; no runtime artifact is modified.
    """
    copied = copy.deepcopy(payload)
    adapt = copied.get("adapt_vqe", {})
    settings = copied.get("settings", {})
    if isinstance(settings, dict):
        settings["phase3_response_coordinate_scope"] = FULL_RESPONSE_SCOPE
    if isinstance(adapt, dict):
        adapt["phase3_response_coordinate_scope"] = FULL_RESPONSE_SCOPE
        for previous_depth, row in enumerate(adapt.get("history", [])):
            if not isinstance(row, dict):
                continue
            row["phase3_response_coordinate_scope"] = FULL_RESPONSE_SCOPE
            row["phase3_active_logical_coordinate_count"] = previous_depth
            row["phase3_response_pre_support_count"] = previous_depth + 1
            row["phase3_response_coordinate_indices"] = list(range(previous_depth + 1))
    return copied


def material_no_overlap_validation_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize only the coordinate-invariant trust metric authority label.

    Material-window Phase III still calibrates the no-overlap update from the
    complete accepted refit.  The inherited validator's older label describes
    the same source-Gram displacement; no other runtime field is changed here.
    """
    copied = copy.deepcopy(payload)
    adapt = copied.get("adapt_vqe", {})
    if not isinstance(adapt, dict):
        return copied
    for row in adapt.get("history", []):
        if not isinstance(row, dict):
            continue
        trust = row.get("route_a_trust_region_update")
        if (
            isinstance(trust, dict)
            and trust.get("geometry_expansion_active") is not True
            and trust.get("displacement_ratio_metric")
            == "full_accepted_refit_supported_source_gram_coordinates_v1"
        ):
            trust["displacement_ratio_metric"] = (
                "supported_source_gram_parameter_displacement_v1"
            )
    return copied


def _material_int_sequence(value: Any, *, field: str) -> list[int]:
    return [int(item) for item in _sequence(value, field=field)]


def _material_pairs(value: Any, *, field: str) -> list[list[int]]:
    pairs = []
    for raw in _sequence(value, field=field):
        pair = _material_int_sequence(raw, field=field + " pair")
        if len(pair) != 2 or pair[0] > pair[1]:
            raise ValueError(f"{field} contains a noncanonical pair")
        pairs.append(pair)
    if len({tuple(pair) for pair in pairs}) != len(pairs):
        raise ValueError(f"{field} contains duplicate pairs")
    return pairs


MATERIAL_SUPPORT_CHANGE_POLICY = (
    "full_geometry_refresh_on_unexpected_supported_nullity_drift_v1"
)


def _material_close(actual: Any, expected: float, *, field: str) -> None:
    value = float(actual)
    if not math.isfinite(value) or not math.isclose(
        value, float(expected), rel_tol=1.0e-12, abs_tol=1.0e-12
    ):
        raise ValueError(f"{field} numerical closure drift: {value} != {expected}")


def _material_receipt_digest(receipt: Mapping[str, Any]) -> str:
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    return hashlib.sha256(json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _material_tail_ratio(
    scores: Sequence[Any], retained_mask: Sequence[bool], *, epsilon: float,
) -> float | None:
    if any(value is None for value in scores):
        return None
    values = [float(value) for value in scores]
    full = math.sqrt(sum(value * value for value in values))
    omitted = math.sqrt(sum(
        value * value for value, retained in zip(values, retained_mask, strict=True)
        if not retained
    ))
    return 0.0 if full <= float(epsilon) else float(omitted / full)


def _material_optional_close(actual: Any, expected: float | None, *, field: str) -> None:
    if expected is None:
        if actual is not None:
            raise ValueError(f"{field} must be null")
        return
    _material_close(actual, expected, field=field)


def _validate_material_receipt(
    receipt: Mapping[str, Any], *, active: list[int], field: str,
) -> dict[str, Any]:
    count = len(active)
    policy = _mapping(receipt.get("policy"), field=field + " policy")
    if (
        receipt.get("receipt_version") != "phase3_material_window_receipt_v1"
        or dict(policy) != MATERIAL_WINDOW_POLICY
        or receipt.get("active_indices") != active
        or receipt.get("receipt_sha256") != _material_receipt_digest(receipt)
    ):
        raise ValueError(f"{field} identity/digest drift")
    for key in (
        "gram_entry_threshold", "hessian_entry_threshold",
        "gram_omitted_l2_tolerance", "hessian_omitted_l2_tolerance",
    ):
        _material_close(receipt.get(key), policy[key], field=f"{field} {key}")
    gram_scores = list(_sequence(
        receipt.get("gram_normalized_scores"), field=field + " Gram scores"
    ))
    hessian_scores = list(_sequence(
        receipt.get("hessian_normalized_scores"), field=field + " Hessian scores"
    ))
    mask_fields = (
        "initial_gram_mask", "initial_hessian_mask", "initial_union_mask",
        "final_retained_mask",
    )
    masks: dict[str, list[bool]] = {}
    for key in mask_fields:
        raw = list(_sequence(receipt.get(key), field=f"{field} {key}"))
        if len(raw) != count or any(type(value) is not bool for value in raw):
            raise ValueError(f"{field} {key} is not a typed active-length mask")
        masks[key] = [bool(value) for value in raw]
    if len(gram_scores) != count or len(hessian_scores) != count:
        raise ValueError(f"{field} normalized-score length drift")
    finite = all(value is not None for value in (*gram_scores, *hessian_scores))
    if receipt.get("inputs_finite") is not finite:
        raise ValueError(f"{field} finite-input classification drift")
    expected_gram = [
        value is not None and float(value) >= float(policy["gram_entry_threshold"])
        for value in gram_scores
    ]
    expected_hessian = [
        value is not None and float(value) >= float(policy["hessian_entry_threshold"])
        for value in hessian_scores
    ]
    union = [left or right for left, right in zip(
        expected_gram, expected_hessian, strict=True
    )]
    if (
        masks["initial_gram_mask"] != expected_gram
        or masks["initial_hessian_mask"] != expected_hessian
        or masks["initial_union_mask"] != union
    ):
        raise ValueError(f"{field} threshold-union decision drift")
    final = list(union)
    initial_gram_tail = _material_tail_ratio(
        gram_scores, final, epsilon=float(policy["epsilon"])
    )
    initial_hessian_tail = _material_tail_ratio(
        hessian_scores, final, epsilon=float(policy["epsilon"])
    )
    closure_added: list[int] = []
    if finite:
        omitted_positions = [index for index, keep in enumerate(final) if not keep]
        omitted_positions.sort(key=lambda index: (
            -max(float(gram_scores[index] or 0.0), float(hessian_scores[index] or 0.0)),
            active[index], index,
        ))
        for index in omitted_positions:
            gram_tail = _material_tail_ratio(
                gram_scores, final, epsilon=float(policy["epsilon"])
            )
            hessian_tail = _material_tail_ratio(
                hessian_scores, final, epsilon=float(policy["epsilon"])
            )
            if (
                gram_tail is not None
                and gram_tail <= float(policy["gram_omitted_l2_tolerance"])
                and hessian_tail is not None
                and hessian_tail <= float(policy["hessian_omitted_l2_tolerance"])
            ):
                break
            final[index] = True
            closure_added.append(active[index])
    final_gram_tail = _material_tail_ratio(
        gram_scores, final, epsilon=float(policy["epsilon"])
    )
    final_hessian_tail = _material_tail_ratio(
        hessian_scores, final, epsilon=float(policy["epsilon"])
    )
    closure_satisfied = bool(
        finite
        and final_gram_tail is not None
        and final_gram_tail <= float(policy["gram_omitted_l2_tolerance"])
        and final_hessian_tail is not None
        and final_hessian_tail <= float(policy["hessian_omitted_l2_tolerance"])
    )
    if count == 0 and finite:
        closure_reason = "candidate_only"
    elif not finite:
        closure_reason = "nonfinite_input"
    elif closure_satisfied and closure_added:
        closure_reason = "satisfied_after_greedy_expansion"
    elif closure_satisfied:
        closure_reason = "satisfied_by_threshold_union"
    else:
        closure_reason = "omitted_tail_closure_failed"
    retained = [value for value, keep in zip(active, final, strict=True) if keep]
    omitted = [value for value, keep in zip(active, final, strict=True) if not keep]
    if (
        masks["final_retained_mask"] != final
        or receipt.get("closure_added_indices") != closure_added
        or receipt.get("retained_indices") != retained
        or receipt.get("omitted_indices") != omitted
        or receipt.get("closure_satisfied") is not closure_satisfied
        or receipt.get("closure_reason") != closure_reason
    ):
        raise ValueError(f"{field} closure/partition semantics drift")
    for key, expected in (
        ("initial_gram_omitted_l2_ratio", initial_gram_tail),
        ("initial_hessian_omitted_l2_ratio", initial_hessian_tail),
        ("final_gram_omitted_l2_ratio", final_gram_tail),
        ("final_hessian_omitted_l2_ratio", final_hessian_tail),
    ):
        _material_optional_close(receipt.get(key), expected, field=f"{field} {key}")
    active_rank = int(receipt.get("measured_active_supported_rank", -1))
    joint_rank = int(receipt.get("measured_joint_supported_rank", -1))
    active_dimension = len(retained)
    joint_dimension = active_dimension + 1
    active_valid = 0 <= active_rank <= active_dimension
    joint_valid = 0 <= joint_rank <= joint_dimension
    active_nullity = active_dimension - active_rank if active_valid else None
    joint_nullity = joint_dimension - joint_rank if joint_valid else None
    rank_gain = joint_rank - active_rank if active_valid and joint_valid else None
    drift = bool(
        active_nullity is not None
        and receipt.get("prior_active_nullity") is not None
        and active_nullity != int(receipt["prior_active_nullity"])
    ) or bool(
        joint_nullity is not None
        and receipt.get("prior_joint_nullity") is not None
        and joint_nullity != int(receipt["prior_joint_nullity"])
    )
    if (
        receipt.get("measured_active_nullity") != active_nullity
        or receipt.get("measured_joint_nullity") != joint_nullity
        or receipt.get("measured_rank_gain") != rank_gain
        or receipt.get("support_nullity_drift") is not drift
    ):
        raise ValueError(f"{field} supported-rank/nullity arithmetic drift")
    return {
        "retained": retained, "omitted": omitted,
        "active_rank": active_rank, "joint_rank": joint_rank,
        "active_valid": active_valid, "joint_valid": joint_valid,
        "active_nullity": active_nullity, "joint_nullity": joint_nullity,
        "rank_gain": rank_gain, "drift": drift,
        "closure_satisfied": closure_satisfied,
    }


def _expected_material_pairs(
    active: list[int], retained: list[int], omitted: list[int],
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    local = {value: index for index, value in enumerate(active)}
    def ordered(left: int, right: int) -> list[int]:
        return [left, right] if local[left] <= local[right] else [right, left]
    retained_retained = [
        ordered(retained[left], retained[right])
        for left in range(len(retained)) for right in range(left, len(retained))
    ]
    retained_omitted = [ordered(left, right) for left in retained for right in omitted]
    omitted_omitted = [
        ordered(omitted[left], omitted[right])
        for left in range(len(omitted)) for right in range(left, len(omitted))
    ]
    return retained_retained, retained_omitted, omitted_omitted


def _validate_material_block_closure(
    closure: Mapping[str, Any], *, retained: list[int], omitted: list[int], field: str,
) -> list[str]:
    if (
        closure.get("schema") != "phase3_material_window_block_closure_v1"
        or closure.get("retained_indices") != retained
        or closure.get("omitted_indices") != omitted
    ):
        raise ValueError(f"{field} identity drift")
    numeric = {
        key: float(closure.get(key, float("nan"))) for key in (
            "gram_retained_fro_norm", "hessian_retained_fro_norm",
            "gram_retained_omitted_fro_norm",
            "hessian_retained_omitted_fro_norm",
            "gram_retained_omitted_ratio", "hessian_retained_omitted_ratio",
            "gram_tolerance", "hessian_tolerance",
        )
    }
    finite = all(math.isfinite(value) for value in numeric.values())
    if finite:
        gram_ratio = 0.0 if not retained or not omitted else (
            numeric["gram_retained_omitted_fro_norm"]
            / max(numeric["gram_retained_fro_norm"], MATERIAL_WINDOW_POLICY["epsilon"])
        )
        hessian_ratio = 0.0 if not retained or not omitted else (
            numeric["hessian_retained_omitted_fro_norm"]
            / max(numeric["hessian_retained_fro_norm"], MATERIAL_WINDOW_POLICY["epsilon"])
        )
        _material_close(
            numeric["gram_retained_omitted_ratio"], gram_ratio,
            field=field + " Gram ratio",
        )
        _material_close(
            numeric["hessian_retained_omitted_ratio"], hessian_ratio,
            field=field + " Hessian ratio",
        )
    _material_close(
        numeric["gram_tolerance"], MATERIAL_WINDOW_POLICY["gram_cross_block_tolerance"],
        field=field + " Gram tolerance",
    )
    _material_close(
        numeric["hessian_tolerance"],
        MATERIAL_WINDOW_POLICY["hessian_cross_block_tolerance"],
        field=field + " Hessian tolerance",
    )
    gram_satisfied = bool(
        finite and numeric["gram_retained_omitted_ratio"] <= numeric["gram_tolerance"]
    )
    hessian_satisfied = bool(
        finite
        and numeric["hessian_retained_omitted_ratio"] <= numeric["hessian_tolerance"]
    )
    reasons: list[str] = []
    if not finite:
        reasons.append("nonfinite_retained_omitted_block_closure")
    if finite and not gram_satisfied:
        reasons.append("gram_cross_block_closure_failed")
    if finite and not hessian_satisfied:
        reasons.append("hessian_cross_block_closure_failed")
    if (
        closure.get("inputs_finite") is not finite
        or closure.get("gram_satisfied") is not gram_satisfied
        or closure.get("hessian_satisfied") is not hessian_satisfied
        or closure.get("closure_satisfied") is not (gram_satisfied and hessian_satisfied)
        or closure.get("refresh_reasons") != reasons
    ):
        raise ValueError(f"{field} numerical decision/reason drift")
    return reasons


def _validate_material_plan(
    plan: Mapping[str, Any], *, active: list[int], retained: list[int],
    omitted: list[int], refresh: bool, field: str,
) -> None:
    rr, ro, oo = _expected_material_pairs(active, retained, omitted)
    refresh_pairs = oo if refresh else []
    initial = [*rr, *ro]
    old_old = [*initial, *refresh_pairs]
    gradients = [*retained, *omitted] if refresh else list(retained)
    exact_sequences = {
        "active_indices": active,
        "screen_gram_diagonal_indices": active,
        "candidate_cross_gram_active_indices": active,
        "candidate_cross_hessian_active_indices": active,
        "retained_indices": retained,
        "omitted_indices": omitted,
        "retained_retained_pairs": rr,
        "retained_omitted_closure_pairs": ro,
        "omitted_omitted_refresh_pairs": refresh_pairs,
        "old_old_metric_pairs_acquired": old_old,
        "old_old_hessian_pairs_acquired": old_old,
        "active_gradient_indices_acquired": gradients,
        "screen_gram_diagonal_indices_reused_in_old_old_pairs": retained,
    }
    if plan.get("schema") != "phase3_material_window_estimator_acquisition_plan_v1":
        raise ValueError(f"{field} schema drift")
    for key, expected in exact_sequences.items():
        if plan.get(key) != expected:
            raise ValueError(f"{field} exact acquisition partition drift: {key}")
    for key in (
        "candidate_self_gram_acquired", "candidate_self_hessian_acquired",
        "candidate_gradient_acquired",
    ):
        if plan.get(key) is not True:
            raise ValueError(f"{field} omitted {key}")
    if (
        plan.get("retained_omitted_closure_acquired") is not bool(ro)
        or plan.get("full_geometry_refresh_performed") is not refresh
        or int(plan.get("full_geometry_refresh_count", -1)) != int(refresh)
    ):
        raise ValueError(f"{field} refresh/acquisition flags drift")
    exact_counts = {
        "screen_gram_diagonal_count": len(active),
        "candidate_cross_gram_count": len(active),
        "candidate_cross_hessian_count": len(active),
        "old_old_metric_pair_count": len(old_old),
        "old_old_hessian_pair_count": len(old_old),
        "retained_retained_pair_count": len(rr),
        "retained_omitted_closure_pair_count": len(ro),
        "omitted_omitted_refresh_pair_count": len(refresh_pairs),
        "active_gradient_count": len(gradients),
    }
    for key, expected in exact_counts.items():
        if int(plan.get(key, -1)) != expected:
            raise ValueError(f"{field} count drift: {key}")
    for fingerprint in (
        "candidate_coordinate_fingerprint", "hamiltonian_fingerprint",
        "ordered_scaffold_fingerprint", "state_fingerprint", "theta_fingerprint",
    ):
        value = str(plan.get(fingerprint) or "")
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"{field} unresolved {fingerprint}")


def _validate_material_terminal_ledger_link(
    *, adapt: Mapping[str, Any], ledger_sidecar: Mapping[str, Any],
    terminal_plan: Mapping[str, Any], terminal_identity: tuple[int, str, int],
) -> dict[str, Any]:
    continuation = _mapping(adapt.get("continuation"), field="material continuation")
    runtime = _mapping(
        continuation.get("runtime_split_summary"), field="material runtime split"
    )
    overlay = _mapping(
        runtime.get("historical_singleton_coordinate_overlay_last_round"),
        field="terminal material coordinate overlay",
    )
    population = _mapping(
        overlay.get("active_gradient_query_accounting"),
        field="terminal material population accounting",
    )
    records = list(_sequence(
        population.get("records"), field="terminal material accounting records"
    ))
    if (
        population.get("schema")
        != "phase3_material_window_population_estimator_accounting_v1"
        or population.get("identity_deduplication")
        != "estimator_call_ledger_global_v1"
        or int(population.get("record_count", -1)) != len(records)
        or not records
    ):
        raise ValueError("terminal material population accounting is open")
    raw_ledger = _mapping(
        ledger_sidecar.get("ledger"), field="material raw estimator ledger"
    )
    entry_ids = {
        str(row.get("primitive_id")) for row in _sequence(
            raw_ledger.get("entries"), field="material ledger entries"
        ) if isinstance(row, Mapping)
    }
    occurrence_ids = {
        str(row.get("primitive_id")) for row in _sequence(
            raw_ledger.get("occurrences"), field="material ledger occurrences"
        ) if isinstance(row, Mapping)
    }
    matched: list[Mapping[str, Any]] = []
    for index, raw_record in enumerate(records):
        record = _mapping(raw_record, field=f"material accounting record {index}")
        source_plan = _mapping(
            record.get("source_plan"), field=f"material accounting plan {index}"
        )
        active = _material_int_sequence(
            source_plan.get("active_indices"), field="accounting active indices"
        )
        retained = _material_int_sequence(
            source_plan.get("retained_indices"), field="accounting retained indices"
        )
        omitted = _material_int_sequence(
            source_plan.get("omitted_indices"), field="accounting omitted indices"
        )
        _validate_material_plan(
            source_plan, active=active, retained=retained, omitted=omitted,
            refresh=source_plan.get("full_geometry_refresh_performed") is True,
            field=f"material accounting plan {index}",
        )
        identity = (
            int(record.get("candidate_pool_index", -1)),
            str(record.get("candidate_label") or ""),
            int(record.get("candidate_position_id", -1)),
        )
        if identity != (
            int(source_plan.get("candidate_pool_index", -2)),
            str(source_plan.get("candidate_label") or ""),
            int(source_plan.get("candidate_position_id", -2)),
        ):
            raise ValueError(f"material accounting candidate identity drift: {index}")
        ids = list(_sequence(
            record.get("primitive_ids"), field=f"material primitive ids {index}"
        ))
        if (
            record.get("schema") != "phase3_material_window_estimator_accounting_v1"
            or record.get("identity_deduplication")
            != "estimator_call_ledger_global_v1"
            or ids != sorted(set(str(value) for value in ids))
            or any(
                len(str(value)) != 64
                or any(char not in "0123456789abcdef" for char in str(value))
                for value in ids
            )
            or int(record.get("unique_primitive_id_count", -1)) != len(ids)
            or not set(ids).issubset(entry_ids)
            or not set(ids).issubset(occurrence_ids)
        ):
            raise ValueError(f"material accounting primitive identity drift: {index}")
        expected_occurrences = (
            len(active)
            + int(source_plan["old_old_metric_pair_count"])
            + int(source_plan["old_old_hessian_pair_count"])
            + int(source_plan["active_gradient_count"])
            + 2 * len(active) + 3
        )
        if int(record.get("primitive_occurrence_id_count", -1)) != expected_occurrences:
            raise ValueError(f"material accounting occurrence closure drift: {index}")
        gradient = _mapping(
            record.get("active_gradient_accounting"),
            field=f"material active-gradient accounting {index}",
        )
        gradient_ids = list(_sequence(
            gradient.get("primitive_ids"), field="material active-gradient ids"
        ))
        if (
            gradient.get("schema") != "phase3_active_gradient_query_accounting_v1"
            or int(gradient.get("active_coordinate_count", -1))
            != int(source_plan["active_gradient_count"])
            or len(gradient_ids) != int(source_plan["active_gradient_count"])
            or not set(gradient_ids).issubset(ids)
        ):
            raise ValueError(f"material active-gradient identity drift: {index}")
        if dict(source_plan) == dict(terminal_plan) and identity == terminal_identity:
            matched.append(record)
    if len(matched) != 1:
        raise ValueError(
            "terminal selected material plan does not map to exactly one ledger record"
        )
    return {
        "scope": "terminal_population_only_v1",
        "population_record_count": len(records),
        "selected_record_match_count": 1,
        "selected_unique_primitive_count": int(
            matched[0]["unique_primitive_id_count"]
        ),
        "selected_primitive_occurrence_count": int(
            matched[0]["primitive_occurrence_id_count"]
        ),
    }


def _validate_material_projected_solver_receipt(
    *, row: Mapping[str, Any], summary: Mapping[str, Any], expected_round: int,
) -> bool:
    """Validate either a projected solve or its exact zero-query fallback."""
    fallback_fired = row.get(
        "all_energy_models_infeasible_novelty_fallback_fired"
    ) is True
    if fallback_fired:
        if (
            row.get("all_energy_models_infeasible_novelty_fallback_enabled")
            is not True
            or int(row.get(
                "all_energy_models_infeasible_novelty_fallback_query_charge", -1
            )) != 0
        ):
            raise ValueError(
                f"round {expected_round} zero-query fallback receipt drift"
            )
        return True
    if (
        summary.get("joint_solve_policy")
        != "supported_metric_projected_generalized_trust_v1"
        or summary.get("joint_linear_solve_policy_requested")
        != "supported_metric_projected_generalized_trust_v1"
        or summary.get("joint_linear_solve_policy_effective")
        != "supported_metric_projected_generalized_trust_v1"
        or summary.get("supported_metric_projection_active") is not True
        or summary.get("supported_metric_whitening_active") is not False
        or summary.get("supported_metric_inverse_sqrt_constructed") is not False
        or summary.get("supported_metric_inverse_constructed") is not False
        or summary.get("metric_regularization_applied") is not False
        or int(summary.get("classical_quantum_query_charge", -1)) != 0
    ):
        raise ValueError(f"round {expected_round} projected non-whitened solve drift")
    return False


def validate_material_window_evidence(
    *, result: dict[str, Any], target_round: int,
    ledger_sidecar: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate Phase-III W+c, closure/refresh, trust reuse, and refit scope."""
    settings = _mapping(result.get("settings"), field="material settings")
    adapt = _mapping(result.get("adapt_vqe"), field="material adapt_vqe")
    route_contract = _mapping(
        settings.get("sr_route_profile_contract"),
        field="material route contract",
    )
    semantic_invariants = _mapping(
        route_contract.get("semantic_invariants"),
        field="material route semantic invariants",
    )
    if (
        settings.get("phase3_response_coordinate_scope") != MATERIAL_WINDOW_SCOPE
        or semantic_invariants.get("phase3_material_window_support_change_policy")
        != MATERIAL_SUPPORT_CHANGE_POLICY
    ):
        raise ValueError("material-window route scope/semantics missing")
    history = list(_sequence(adapt.get("history"), field="material history"))
    if len(history) != int(target_round):
        raise ValueError("material-window history does not cover every round")
    refresh_count = 0
    retained_coordinate_total = 0
    omitted_coordinate_total = 0
    projected_receipts = 0
    exact_fallback_holds = 0
    terminal_plan: dict[str, Any] | None = None
    terminal_identity: tuple[int, str, int] | None = None
    for expected_round, raw_row in enumerate(history, start=1):
        row = _mapping(raw_row, field=f"round {expected_round} material row")
        active_count = expected_round - 1
        if int(row.get("phase3_active_logical_coordinate_count", -1)) != active_count:
            raise ValueError(f"round {expected_round} active-coordinate count drift")
        if row.get("phase3_response_coordinate_scope") != MATERIAL_WINDOW_SCOPE:
            raise ValueError(f"round {expected_round} material response scope drift")
        admitted = list(_sequence(
            row.get("admitted_records"), field=f"round {expected_round} admissions"
        ))
        if len(admitted) != 1:
            raise ValueError(f"round {expected_round} is not one admission")
        admission = _mapping(admitted[0], field=f"round {expected_round} admission")
        summary = _mapping(
            admission.get(
                "phase2_joint_geometry_reuse"
            ),
            field=f"round {expected_round} Phase-III receipt",
        )
        receipt = _mapping(
            summary.get("material_window_receipt"),
            field=f"round {expected_round} material-window receipt",
        )
        refresh = _mapping(
            summary.get("material_window_refresh"),
            field=f"round {expected_round} material refresh",
        )
        plan = _mapping(
            summary.get("estimator_acquisition_plan"),
            field=f"round {expected_round} estimator acquisition plan",
        )
        active = _material_int_sequence(
            receipt.get("active_indices"), field=f"round {expected_round} active indices"
        )
        if active != list(range(active_count)):
            raise ValueError(f"round {expected_round} active index registry drift")
        receipt_validation = _validate_material_receipt(
            receipt, active=active, field=f"round {expected_round} material receipt"
        )
        retained = receipt_validation["retained"]
        omitted = receipt_validation["omitted"]
        requires_refresh = receipt.get("requires_full_geometry_refresh") is True
        refresh_performed = refresh.get("performed") is True
        if requires_refresh != refresh_performed:
            raise ValueError(f"round {expected_round} refresh decision/execution mismatch")
        closure = _mapping(
            plan.get("retained_omitted_block_closure"),
            field=f"round {expected_round} block closure",
        )
        external_refresh_reasons = _validate_material_block_closure(
            closure, retained=retained, omitted=omitted,
            field=f"round {expected_round} block closure",
        )
        expected_refresh_reasons: list[str] = []
        if receipt.get("inputs_finite") is not True:
            expected_refresh_reasons.append("nonfinite_input")
        if receipt.get("closure_satisfied") is not True:
            expected_refresh_reasons.append("closure_failed")
        if not receipt_validation["active_valid"]:
            expected_refresh_reasons.append("invalid_active_supported_rank")
        if not receipt_validation["joint_valid"]:
            expected_refresh_reasons.append("invalid_joint_supported_rank")
        if (
            receipt_validation["rank_gain"] is not None
            and receipt_validation["rank_gain"] not in (0, 1)
        ):
            expected_refresh_reasons.append("invalid_rank_gain")
        if (
            receipt_validation["active_nullity"] is not None
            and receipt.get("prior_active_nullity") is not None
            and receipt_validation["active_nullity"]
            != int(receipt["prior_active_nullity"])
        ):
            expected_refresh_reasons.append("active_support_nullity_drift")
        if (
            receipt_validation["joint_nullity"] is not None
            and receipt.get("prior_joint_nullity") is not None
            and receipt_validation["joint_nullity"]
            != int(receipt["prior_joint_nullity"])
        ):
            expected_refresh_reasons.append("joint_support_nullity_drift")
        for reason in external_refresh_reasons:
            if reason not in expected_refresh_reasons:
                expected_refresh_reasons.append(reason)
        if (
            receipt.get("refresh_reasons") != expected_refresh_reasons
            or requires_refresh is not bool(expected_refresh_reasons)
        ):
            raise ValueError(f"round {expected_round} exact refresh-reason drift")
        final_active = _material_int_sequence(
            refresh.get("final_active_indices"),
            field=f"round {expected_round} final material indices",
        )
        if (
            refresh.get("reasons") != expected_refresh_reasons
            or int(refresh.get("count", -1)) != int(requires_refresh)
            or int(refresh.get("retained_supported_rank", -1))
            != receipt_validation["active_rank"]
            or int(refresh.get("retained_joint_supported_rank", -1))
            != receipt_validation["joint_rank"]
        ):
            raise ValueError(f"round {expected_round} refresh receipt drift")
        if requires_refresh:
            if final_active != active or not expected_refresh_reasons:
                raise ValueError(f"round {expected_round} full refresh is incomplete")
            refresh_count += 1
        else:
            if (
                final_active != retained
                or receipt.get("closure_satisfied") is not True
                or refresh.get("refresh_sparse_acquisition") is not None
            ):
                raise ValueError(f"round {expected_round} closed window drift")
        response_indices = _material_int_sequence(
            row.get("phase3_response_coordinate_indices"),
            field=f"round {expected_round} response indices",
        )
        expected_response = final_active + [active_count]
        if (
            response_indices != expected_response
            or int(row.get("phase3_response_pre_support_count", -1))
            != len(expected_response)
        ):
            raise ValueError(f"round {expected_round} W+c response identity drift")
        supported_rank = row.get("phase3_response_supported_rank")
        fallback_fired = row.get(
            "all_energy_models_infeasible_novelty_fallback_fired"
        ) is True
        if supported_rank is None and not fallback_fired:
            raise ValueError(f"round {expected_round} supported rank missing")
        if supported_rank is not None and not 1 <= int(supported_rank) <= len(expected_response):
            raise ValueError(f"round {expected_round} supported rank out of bounds")
        if _validate_material_projected_solver_receipt(
            row=row, summary=summary, expected_round=expected_round,
        ) != fallback_fired:
            raise ValueError(f"round {expected_round} fallback classification drift")
        projected_receipts += 1
        _validate_material_plan(
            plan, active=active, retained=retained, omitted=omitted,
            refresh=requires_refresh, field=f"round {expected_round} acquisition plan",
        )
        candidate_identity = (
            int(plan.get("candidate_pool_index", -1)),
            str(plan.get("candidate_label") or ""),
            int(plan.get("candidate_position_id", -1)),
        )
        if (
            candidate_identity[2] != active_count
            or candidate_identity != (
                int(summary.get("candidate_pool_index", -2)),
                str(summary.get("candidate_label") or ""),
                int(summary.get("position_id", -2)),
            )
            or int(admission.get("candidate_pool_index", candidate_identity[0]))
            != candidate_identity[0]
        ):
            raise ValueError(f"round {expected_round} candidate identity drift")
        terminal_plan = dict(plan)
        terminal_identity = candidate_identity
        accepted = _mapping(
            row.get("accepted_refit"), field=f"round {expected_round} accepted refit"
        )
        config = _mapping(
            _mapping(
                accepted.get("accepted_refit_invocation"), field="accepted invocation"
            ).get("config"), field="accepted config",
        )
        if (
            int(row.get("phase3_accepted_refit_coordinate_count", -1))
            != active_count + 1
            or config.get("scope") != "full_ansatz_v1"
            or config.get("full_ansatz") is not True
            or config.get("coordinate_chart") != "supported_fs_whitened_fixed_v1"
            or config.get("supported_fs_whitened") is not True
        ):
            raise ValueError(f"round {expected_round} full accepted refit drift")
        trust = _mapping(
            row.get("route_a_trust_region_update"),
            field=f"round {expected_round} trust update",
        )
        overlap = _mapping(
            trust.get("endpoint_overlap_query_accounting"), field="overlap accounting"
        )
        if (
            trust.get("policy") != "source_metric_inverse_sqrt_no_overlap_v1"
            or trust.get("endpoint_overlap_measurement_required") is not False
            or trust.get("endpoint_overlap_measurement_performed") is not False
            or int(trust.get("endpoint_overlap_query_charge", -1)) != 0
            or overlap.get("performed") is not False
            or overlap.get("charged") is not False
            or int(overlap.get("added_query_count", -1)) != 0
        ):
            raise ValueError(f"round {expected_round} no-overlap contract drift")
        transaction = trust.get("source_metric_trust_transaction")
        if trust.get("geometry_expansion_active") is True:
            if (
                transaction is not None
                or trust.get("update_reason")
                != "geometry_expansion_no_coordinate_prediction_no_overlap_hold"
            ):
                raise ValueError(f"round {expected_round} fallback hold drift")
            exact_fallback_holds += 1
        else:
            transaction = _mapping(transaction, field=f"round {expected_round} trust transaction")
            if (
                transaction.get("schema")
                != "sr_material_window_full_source_metric_accepted_path_transaction_v1"
                or transaction.get("transaction_complete") is not True
                or transaction.get("phase3_prediction_coordinate_scope")
                != "candidate_material_W_plus_singleton_v1"
                or transaction.get("trust_calibration_metric_scope")
                != "full_accepted_refit_source_gram_v1"
                or transaction.get("source_metric_reused_from_accepted_refit") is not True
                or transaction.get("accepted_refit_supported_fs_whitening_active") is not True
                or transaction.get("supported_metric_whitening_active_in_phase3") is not False
                or transaction.get("supported_metric_inverse_sqrt_constructed_in_phase3") is not False
                or int(transaction.get("incremental_quantum_query_charge", -1)) != 0
                or transaction.get("material_window_receipt") != receipt
                or trust.get("displacement_ratio_metric")
                != "full_accepted_refit_supported_source_gram_coordinates_v1"
                or float(trust.get("trust_radius_update_exponent")) != -0.5
            ):
                raise ValueError(f"round {expected_round} full-source trust reuse drift")
        retained_coordinate_total += len(retained)
        omitted_coordinate_total += len(omitted)
    if ledger_sidecar is None or terminal_plan is None or terminal_identity is None:
        raise ValueError("material-window validation requires terminal ledger identity")
    ledger_link = _validate_material_terminal_ledger_link(
        adapt=adapt,
        ledger_sidecar=_mapping(ledger_sidecar, field="material estimator ledger"),
        terminal_plan=terminal_plan,
        terminal_identity=terminal_identity,
    )
    return {
        "schema": "paper_i_sr_material_window_evidence_v1",
        "status": "pass",
        "controller_rounds": int(target_round),
        "phase3_response_coordinate_scope": MATERIAL_WINDOW_SCOPE,
        "projected_solver_receipt_count": projected_receipts,
        "supported_metric_whitening_active_in_phase3": False,
        "accepted_powell_refit_whitening_active": True,
        "endpoint_overlap_measurement_count": 0,
        "endpoint_overlap_query_charge": 0,
        "material_window_refresh_count": refresh_count,
        "retained_coordinate_occurrences": retained_coordinate_total,
        "omitted_coordinate_occurrences": omitted_coordinate_total,
        "exact_geometry_expansion_hold_count": exact_fallback_holds,
        "incremental_trust_metric_query_charge": 0,
        "support_change_policy": MATERIAL_SUPPORT_CHANGE_POLICY,
        "valid_rank_gain_values": [0, 1],
        "ordinary_plus_one_rank_gain_requires_refresh": False,
        "terminal_estimator_ledger_linkage": ledger_link,
    }
'''


def _patch_run_job(
    path: Path, *, parent_profile: str, child_profile: str,
) -> None:
    text = path.read_text(encoding="utf-8")
    start = text.index("PROFILE = (")
    end = text.index("DIGEST =", start)
    text = text[:start] + f"PROFILE = {child_profile!r}\n" + text[end:]
    text = text.replace(anchor.PARENT_ALIAS, anchor.CHILD_ALIAS)
    text = text.replace(anchor.PARENT_DIGEST, anchor.CHILD_DIGEST)
    old_import = "    validate_no_overlap_trust_evidence,\n)"
    new_import = (
        "    validate_no_overlap_trust_evidence,\n"
        "    MATERIAL_WINDOW_SCOPE,\n"
        "    MATERIAL_SUPPORT_CHANGE_POLICY,\n"
        "    material_parent_validation_view,\n"
        "    material_no_overlap_validation_view,\n"
        "    validate_material_window_evidence,\n"
        ")"
    )
    if old_import not in text:
        raise ValueError("run_job evidence import seam missing")
    text = text.replace(old_import, new_import, 1)
    old_gate = '        "full_active_plus_singleton_response_each_round_required",'
    new_gate = '        "material_window_response_each_round_required",'
    if old_gate not in text:
        raise ValueError("run_job evidence gate seam missing")
    text = text.replace(old_gate, new_gate, 1)
    old_scope = (
        '        "phase3_response_coordinate_scope": '
        '"full_active_plus_singleton_v1",'
    )
    new_scope = (
        '        "phase3_response_coordinate_scope": MATERIAL_WINDOW_SCOPE,'
    )
    if old_scope not in text:
        raise ValueError("run_job executable response-scope seam missing")
    text = text.replace(old_scope, new_scope, 1)
    old_semantic_gate = '        "terminal_prune_active": False,\n    }'
    new_semantic_gate = (
        '        "terminal_prune_active": False,\n'
        '        "phase3_material_window_support_change_policy": '
        'MATERIAL_SUPPORT_CHANGE_POLICY,\n'
        '    }'
    )
    if old_semantic_gate not in text:
        raise ValueError("run_job material semantic-invariant seam missing")
    text = text.replace(old_semantic_gate, new_semantic_gate, 1)
    old_segment_id = (
        'f"{slug}-sr-no-overlap-trust-r0-r{target}-20260720-v2"'
    )
    new_segment_id = (
        'f"{slug}-sr-material-window-r0-r{target}-20260721-v1"'
    )
    if old_segment_id not in text:
        raise ValueError("run_job material segment-identity seam missing")
    text = text.replace(old_segment_id, new_segment_id, 1)
    old_calls = '''    evidence = validate_parent_evidence(
        result=result,
        current=current,
        ledger_sidecar=ledger,
        profile=PROFILE,
        digest=DIGEST,
        target_round=target_round,
        target_new_admissions=target_admissions,
        require_supported_rank=True,
    )
    projected_evidence = validate_projected_generalized_phase3_evidence(
        result=result, target_round=target_round
    )
    no_overlap_evidence = validate_no_overlap_trust_evidence(
        result=result, target_round=target_round
    )
'''
    new_calls = '''    material_evidence = validate_material_window_evidence(
        result=result, target_round=target_round, ledger_sidecar=ledger
    )
    evidence = validate_parent_evidence(
        result=material_parent_validation_view(result),
        current=material_parent_validation_view(current),
        ledger_sidecar=ledger,
        profile=PROFILE,
        digest=DIGEST,
        target_round=target_round,
        target_new_admissions=target_admissions,
        require_supported_rank=True,
    )
    evidence["phase3_response_scope"] = MATERIAL_WINDOW_SCOPE
    projected_evidence = validate_projected_generalized_phase3_evidence(
        result=material_parent_validation_view(result), target_round=target_round
    )
    no_overlap_evidence = validate_no_overlap_trust_evidence(
        result=material_no_overlap_validation_view(result), target_round=target_round
    )
'''
    if old_calls not in text:
        raise ValueError("run_job runtime validator seam missing")
    text = text.replace(old_calls, new_calls, 1)
    old_return = '        "no_overlap_trust_validation": no_overlap_evidence,\n'
    new_return = old_return + '        "material_window_validation": material_evidence,\n'
    if old_return not in text:
        raise ValueError("run_job validation-return seam missing")
    path.write_text(text.replace(old_return, new_return, 1), encoding="utf-8")


def _patch_validator(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "def validate_material_window_evidence(" in text:
        raise ValueError("validator already contains material-window extension")
    path.write_text(text.rstrip() + "\n" + _material_validator_extension(), encoding="utf-8")


def _patch_validate_fetched(path: Path, *, child_profile: str) -> None:
    """Patch the fetched validator to recompute every material-route receipt."""

    text = path.read_text(encoding="utf-8")
    start = text.index("PROFILE = (")
    end = text.index("PHASE1_ENERGY_MODEL", start)
    text = text[:start] + f"PROFILE = {child_profile!r}\n" + text[end:]
    text = text.replace(anchor.PARENT_ALIAS, anchor.CHILD_ALIAS)
    text = text.replace(anchor.PARENT_DIGEST, anchor.CHILD_DIGEST)
    old_import = "    validate_no_overlap_trust_evidence,\n)"
    new_import = (
        "    validate_no_overlap_trust_evidence,\n"
        "    MATERIAL_WINDOW_SCOPE,\n"
        "    MATERIAL_SUPPORT_CHANGE_POLICY,\n"
        "    material_parent_validation_view,\n"
        "    material_no_overlap_validation_view,\n"
        "    validate_material_window_evidence,\n"
        ")"
    )
    if old_import not in text:
        raise ValueError("fetched validator evidence import seam missing")
    text = text.replace(old_import, new_import, 1)
    old_calls = '''    evidence = validate_parent_evidence(
        result=result,
        current=current,
        ledger_sidecar=ledger,
        profile=PROFILE,
        digest=digest,
        target_round=target_round,
        target_new_admissions=target_new_admissions,
        require_supported_rank=True,
    )
    projected_evidence = validate_projected_generalized_phase3_evidence(
        result=result, target_round=target_round
    )
    no_overlap_evidence = validate_no_overlap_trust_evidence(
        result=result, target_round=target_round
    )
'''
    new_calls = '''    material_evidence = validate_material_window_evidence(
        result=result, target_round=target_round, ledger_sidecar=ledger
    )
    evidence = validate_parent_evidence(
        result=material_parent_validation_view(result),
        current=material_parent_validation_view(current),
        ledger_sidecar=ledger,
        profile=PROFILE,
        digest=digest,
        target_round=target_round,
        target_new_admissions=target_new_admissions,
        require_supported_rank=True,
    )
    evidence["phase3_response_scope"] = MATERIAL_WINDOW_SCOPE
    projected_evidence = validate_projected_generalized_phase3_evidence(
        result=material_parent_validation_view(result), target_round=target_round
    )
    no_overlap_evidence = validate_no_overlap_trust_evidence(
        result=material_no_overlap_validation_view(result), target_round=target_round
    )
'''
    if old_calls not in text:
        raise ValueError("fetched validator runtime evidence seam missing")
    text = text.replace(old_calls, new_calls, 1)
    old_runtime = '''    runtime_evidence = validation.get("scientific_evidence_validation")
    if runtime_evidence != evidence:
        raise ValueError("runtime/fetched scientific-evidence validation mismatch")
'''
    new_runtime = old_runtime + '''    if validation.get(
        "projected_generalized_phase3_validation"
    ) != projected_evidence:
        raise ValueError("runtime/fetched projected Phase-III validation mismatch")
    if validation.get("no_overlap_trust_validation") != no_overlap_evidence:
        raise ValueError("runtime/fetched no-overlap validation mismatch")
    if validation.get("material_window_validation") != material_evidence:
        raise ValueError("runtime/fetched material-window validation mismatch")
'''
    if old_runtime not in text:
        raise ValueError("fetched validator runtime receipt comparison seam missing")
    text = text.replace(old_runtime, new_runtime, 1)
    old_return = '        "no_overlap_trust_validation": no_overlap_evidence,\n'
    if old_return not in text:
        raise ValueError("fetched validator return seam missing")
    text = text.replace(
        old_return,
        old_return + '        "material_window_validation": material_evidence,\n',
        1,
    )
    physics_seam = '    physics = normalized.get("physics", {})\n'
    regime_gate = f'''    expected_regimes = {EXPECTED_REGIME_MATRIX!r}
    regime = output.name
    if regime not in expected_regimes:
        raise ValueError(f"unexpected fetched regime {{regime!r}}")
    expected_regime = expected_regimes[regime]
    for key in ("u_over_t", "lambda", "g_ep"):
        if abs(float(physics.get(key, float("nan"))) - float(expected_regime[key])) > 1.0e-12:
            raise ValueError(f"fetched regime {{regime}} {{key}} drift")
    expected_n_ph = int(expected_regime["n_ph"])
    if (
        physics.get("same_cutoff_reference") is not True
        or int(physics.get("n_ph_work", -1)) != expected_n_ph
        or int(physics.get("n_ph_reference", -1)) != expected_n_ph
    ):
        raise ValueError(f"fetched regime {{regime}} same-cutoff drift")
'''
    if physics_seam not in text:
        raise ValueError("fetched validator physics seam missing")
    text = text.replace(physics_seam, physics_seam + regime_gate, 1)
    path.write_text(text, encoding="utf-8")


def _route_job(
    base_job: Mapping[str, Any], *, contracts: Mapping[str, Any],
    replacements: Mapping[str, str], archive_sha: str,
    archive_manifest_sha: str, revision_sha: str, physics_sha: str,
    overlay_receipt: Mapping[str, Any], anchor_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    job = anchor.replace_tree(copy.deepcopy(base_job), replacements)
    job["bundle_id"] = OUTPUT_ID
    job["batch_name"] = OUTPUT_BATCH
    argv = list(job["command"]["argv"])
    route_index = argv.index("--sr-route-profile") + 1
    argv[route_index] = anchor.CHILD_ALIAS
    job["command"]["argv"] = argv
    contract = copy.deepcopy(contracts[anchor.CHILD_ALIAS]["contract"])
    route = job["route_identity"]
    route.update({
        "profile_request": anchor.CHILD_ALIAS,
        "profile_resolved": contract["route_profile"],
        "profile_contract": contract,
        "profile_contract_sha256": anchor.CHILD_DIGEST,
    })
    requirements = job["evidence_requirements"]
    requirements.pop("full_active_plus_singleton_response_each_round_required", None)
    requirements.update({
        "material_window_response_each_round_required": True,
        "material_window_closure_rank_refresh_telemetry_required": True,
        "material_window_acquisition_identity_closure_required": True,
        "full_accepted_refit_each_round_required": True,
    })
    job["source_lock"].update({
        "source_archive": f"chtc/phase3_optuna/input/{OUTPUT_ID}/source_locked.tar.gz",
        "source_archive_sha256": archive_sha,
        "source_archive_manifest": f"chtc/phase3_optuna/input/{OUTPUT_ID}/source_archive_manifest.json",
        "source_archive_manifest_sha256": archive_manifest_sha,
        "source_revision_manifest": f"chtc/phase3_optuna/input/{OUTPUT_ID}/source_revision_manifest.json",
        "source_revision_manifest_sha256": revision_sha,
        "physics_reference_lock": f"chtc/phase3_optuna/input/{OUTPUT_ID}/physics_and_exact_reference_lock.json",
        "physics_reference_lock_sha256": physics_sha,
        "material_window_source_overlay": copy.deepcopy(overlay_receipt),
        "material_window_anchor_authorization": copy.deepcopy(anchor_receipt),
    })
    job["source_locked_sensitivity"] = {
        "schema": "source_locked_sensitivity_candidate_row_v1",
        "swept_field": "phase3_response_coordinate_scope",
        "source_value": FULL_SCOPE,
        "candidate_value": MATERIAL_SCOPE,
        "only_intended_execution_field_change": {
            "phase3_response_coordinate_scope": MATERIAL_SCOPE,
        },
        "non_swept_settings_diff": [],
        "anchor_reproduces_source": True,
    }
    return job


def build_fanout(
    *, anchor_result: Path, anchor_validation: Path, anchor_transfer: Path,
) -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"immutable fanout already exists: {OUTPUT}")
    authorization = _anchor_evidence(
        result_path=anchor_result,
        validation_path=anchor_validation,
        transfer_path=anchor_transfer,
    )
    old_archive_sha = anchor.sha256(BASE / "source_locked.tar.gz")
    with tempfile.TemporaryDirectory(prefix="paper-i-material-window-fanout-") as raw:
        temp = Path(raw)
        archive_path, files, overlays, contracts = _locked_anchor_source(temp)
        archive_sha = anchor.sha256(archive_path)
        if archive_sha != ANCHOR_SOURCE_ARCHIVE_SHA256:
            raise ValueError("fanout source differs from the exact tested anchor archive")
        shutil.copytree(BASE, OUTPUT, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        anchor.clean_inherited_bundle_state(OUTPUT)
        shutil.copy2(archive_path, OUTPUT / "source_locked.tar.gz")

    replacements: dict[str, str] = {
        BASE_ID: OUTPUT_ID,
        BASE_BATCH: OUTPUT_BATCH,
        anchor.ANCHOR_ID: OUTPUT_ID,
        anchor.ANCHOR_BATCH: OUTPUT_BATCH,
        old_archive_sha: archive_sha,
        "sr-no-overlap-trust-r0-r50-20260720-v2": (
            "sr-material-window-r0-r50-20260721-v1"
        ),
    }
    overlay_receipt = copy.deepcopy(
        anchor.load(anchor.ANCHOR / "source_archive_manifest.json").get(
            "material_window_source_overlay", {}
        )
    )
    if overlay_receipt.get("overlay_files") != overlays:
        raise ValueError("tested anchor material-window overlay inventory drift")
    overlay_receipt["immutable_anchor_source_archive_sha256"] = archive_sha
    overlay_receipt["fanout_bundle"] = OUTPUT_ID
    archive_manifest = anchor.replace_tree(
        anchor.load(anchor.ANCHOR / "source_archive_manifest.json"), replacements
    )
    archive_manifest.update({
        "archive": f"chtc/phase3_optuna/input/{OUTPUT_ID}/source_locked.tar.gz",
        "archive_sha256": archive_sha,
        "archive_size_bytes": (OUTPUT / "source_locked.tar.gz").stat().st_size,
        "file_count": len(files),
        "files": files,
        "material_window_source_overlay": overlay_receipt,
    })
    anchor.dump(OUTPUT / "source_archive_manifest.json", archive_manifest)
    archive_manifest_sha = anchor.sha256(OUTPUT / "source_archive_manifest.json")
    revision = anchor.replace_tree(
        anchor.load(anchor.ANCHOR / "source_revision_manifest.json"), replacements
    )
    revision["material_window_source_overlay"] = overlay_receipt
    child_contract = copy.deepcopy(contracts[anchor.CHILD_ALIAS]["contract"])
    revision.update({
        "profile_request": anchor.CHILD_ALIAS,
        "profile_resolved": child_contract["route_profile"],
        "profile_contract_sha256": anchor.CHILD_DIGEST,
        "source_locked_route_transition": {
            "schema": "paper_i_sr_material_window_route_transition_v1",
            "parent_profile_request": anchor.PARENT_ALIAS,
            "parent_profile_resolved": contracts[anchor.PARENT_ALIAS]["contract"][
                "route_profile"
            ],
            "parent_profile_contract_sha256": anchor.PARENT_DIGEST,
            "candidate_profile_request": anchor.CHILD_ALIAS,
            "candidate_profile_resolved": child_contract["route_profile"],
            "candidate_profile_contract_sha256": anchor.CHILD_DIGEST,
            "changed_execution_field": "phase3_response_coordinate_scope",
            "source_value": FULL_SCOPE,
            "candidate_value": MATERIAL_SCOPE,
            "immutable_anchor_source_archive_sha256": archive_sha,
            "non_swept_settings_diff": [],
        },
    })
    anchor.dump(OUTPUT / "source_revision_manifest.json", revision)
    revision_sha = anchor.sha256(OUTPUT / "source_revision_manifest.json")
    physics = anchor.replace_tree(
        anchor.load(anchor.ANCHOR / "physics_and_exact_reference_lock.json"), replacements
    )
    anchor.dump(OUTPUT / "physics_and_exact_reference_lock.json", physics)
    physics_sha = anchor.sha256(OUTPUT / "physics_and_exact_reference_lock.json")

    jobs = []
    normalized_paths = []
    for path in sorted((BASE / "jobs").glob("*.json")):
        _validate_regime_contract(anchor.load(path))
        job = _route_job(
            anchor.load(path), contracts=contracts, replacements=replacements,
            archive_sha=archive_sha, archive_manifest_sha=archive_manifest_sha,
            revision_sha=revision_sha, physics_sha=physics_sha,
            overlay_receipt=overlay_receipt, anchor_receipt=authorization,
        )
        _validate_regime_contract(job)
        out = OUTPUT / "jobs" / path.name
        anchor.dump(out, job)
        jobs.append(str(out.relative_to(ROOT)))
        source_normalized = BASE / "normalized_manifests" / path.name
        normalized = anchor.replace_tree(anchor.load(source_normalized), replacements)
        normalized.update({
            "bundle_id": OUTPUT_ID,
            "batch_name": OUTPUT_BATCH,
            "command_argv": copy.deepcopy(job["command"]["argv"]),
            "route_identity": copy.deepcopy(job["route_identity"]),
            "evidence_requirements": copy.deepcopy(job["evidence_requirements"]),
            "source_lock": copy.deepcopy(job["source_lock"]),
            "source_locked_sensitivity": copy.deepcopy(job["source_locked_sensitivity"]),
        })
        normalized_out = OUTPUT / "normalized_manifests" / path.name
        anchor.dump(normalized_out, normalized)
        normalized_paths.append(str(normalized_out.relative_to(ROOT)))

    shutil.copy2(anchor.RECOVERED_VALIDATOR, OUTPUT / "evidence_validation.py")
    for relative in ("run_job.py", "evidence_validation.py", "validate_fetched.py", "execute_source_locked_job.sh"):
        anchor.patch_text(OUTPUT / relative, replacements)
    _patch_validator(OUTPUT / "evidence_validation.py")
    _patch_run_job(
        OUTPUT / "run_job.py",
        parent_profile=contracts[anchor.PARENT_ALIAS]["contract"]["route_profile"],
        child_profile=contracts[anchor.CHILD_ALIAS]["contract"]["route_profile"],
    )
    _patch_validate_fetched(
        OUTPUT / "validate_fetched.py",
        child_profile=contracts[anchor.CHILD_ALIAS]["contract"]["route_profile"],
    )
    queue = anchor.replace_tree(
        (BASE / "queue.tsv").read_text(encoding="utf-8"), replacements
    )
    (OUTPUT / "queue.tsv").write_text(queue, encoding="utf-8")
    queue_rel = f"chtc/phase3_optuna/input/{OUTPUT_ID}/queue.tsv"
    (OUTPUT / "submit.sub").write_text(
        anchor.submit_text(OUTPUT_ID, OUTPUT_BATCH, archive_sha, queue_rel),
        encoding="utf-8",
    )
    threshold_audit = anchor.threshold_source_audit()
    anchor.dump(OUTPUT / "material_window_threshold_source_audit.json", threshold_audit)
    sensitivity = {
        "schema": "source_locked_sensitivity_audit_v1",
        "source": {
            "method": "SR-SNAKE no-overlap full geometry",
            "regime_or_case": "weak_weak",
            "source_transfer_archive": str(anchor.SOURCE_TRANSFER.relative_to(ROOT)),
            "source_transfer_archive_sha256": anchor.sha256(anchor.SOURCE_TRANSFER),
            "source_result_archive_member": anchor.SOURCE_RESULT_MEMBER,
            "route_or_profile_id": anchor.PARENT_ALIAS,
            "route_contract_sha256": anchor.PARENT_DIGEST,
            "source_variable_value": FULL_SCOPE,
        },
        "sweep": {
            "run_class": "candidate",
            "variable": "phase3_response_coordinate_scope",
            "grid": [FULL_SCOPE, MATERIAL_SCOPE],
            "runner_mode": "direct_source_locked_replay",
            "baseline_materialization_status": "complete",
            "unresolved_source_fields": [],
            "fields_added_by_current_defaults": [],
        },
        "planned_rows": [
            {
                "bundle": OUTPUT_ID,
                "regime_or_case": anchor.load(path)["regime_slug"],
                "value": MATERIAL_SCOPE,
                "changed_fields_vs_source": ["phase3_response_coordinate_scope"],
                "non_swept_settings_diff": [],
            }
            for path in sorted((BASE / "jobs").glob("*.json"))
        ],
        "anchor": authorization,
        "fanout_authorized": True,
        "fanout_bundle": OUTPUT_ID,
        "fanout_route_contract_sha256": anchor.CHILD_DIGEST,
        "fanout_source_archive_sha256": archive_sha,
        "status": "anchor_pass_fanout_authorized",
    }
    anchor.dump(OUTPUT / "source_locked_sensitivity_audit.json", sensitivity)
    receipt = {
        "schema": "paper_i_sr_material_window_fanout_bundle_v1",
        "bundle_id": OUTPUT_ID,
        "batch_name": OUTPUT_BATCH,
        "source_archive_sha256": archive_sha,
        "source_archive_manifest_sha256": archive_manifest_sha,
        "source_revision_manifest_sha256": revision_sha,
        "physics_reference_lock_sha256": physics_sha,
        "parent_route_contract_sha256": anchor.PARENT_DIGEST,
        "material_window_route_contract_sha256": anchor.CHILD_DIGEST,
        "anchor_result_sha256": authorization["anchor_result_sha256"],
        "job_count": 6,
        "jobs": jobs,
        "normalized_manifests": normalized_paths,
        "fanout_authorized": True,
        "submission_performed": False,
    }
    anchor.dump(OUTPUT / "fanout_bundle_receipt.json", receipt)
    anchor.dump(OUTPUT / "bundle_manifest.json", receipt)
    anchor.dump(OUTPUT / "route_parity.json", {
        "schema": "paper_i_sr_material_window_fanout_route_parity_v1",
        "status": "pass",
        "changed_execution_fields_vs_parent": ["phase3_response_coordinate_scope"],
        "parent_value": FULL_SCOPE,
        "candidate_value": MATERIAL_SCOPE,
        "non_swept_settings_diff": [],
        "parent_route_contract_sha256": anchor.PARENT_DIGEST,
        "candidate_route_contract_sha256": anchor.CHILD_DIGEST,
    })
    anchor.dump(OUTPUT / "scientific_settings_audit.json", {
        "schema": "paper_i_sr_material_window_fanout_scientific_audit_v1",
        "status": "pass",
        "anchor_reproduces_source": True,
        "changed_scientific_execution_fields_vs_parent": [
            "phase3_response_coordinate_scope"
        ],
        "non_swept_settings_diff": [],
        "phase3_supported_whitening_active": False,
        "accepted_powell_refit_whitening_active": True,
        "endpoint_overlap_measurement_active": False,
        "endpoint_overlap_query_charge": 0,
        "pruning_active": False,
        "beam_active": False,
    })
    preflight = {
        "schema": "paper_i_sr_material_window_fanout_preflight_v1",
        "status": "local_archive_preflight_pending",
        "checks": {
            "anchor_reproduces_source": True,
            "six_job_records": len(jobs) == 6,
            "six_normalized_records": len(normalized_paths) == 6,
            "single_scientific_execution_field_changed": True,
            "same_cutoff_all_rows": True,
            "weak_holstein_n_ph_3": True,
            "strong_holstein_n_ph_7": True,
            "all_rows_exact_round_50": True,
            "material_window_receipt_required": True,
            "closure_rank_refresh_telemetry_required": True,
            "acquisition_identity_closure_required": True,
            "phase3_raw_supported_projection": True,
            "phase3_whitening_disabled": True,
            "accepted_powell_refit_whitening_preserved": True,
            "endpoint_overlap_measurement_disabled": True,
            "endpoint_overlap_query_charge_zero_required": True,
            "archive_only_worker_validation": False,
            "archive_focused_tests": False,
            "deterministic_result_evidence_validation": False,
            "submission_not_performed": True,
        },
    }
    anchor.dump(OUTPUT / "preflight.json", preflight)
    anchor.dump(OUTPUT / "archive_only_preflight.json", preflight)
    anchor.dump(OUTPUT / "remote_execution_gate.json", {
        "schema": "paper_i_sr_material_window_fanout_remote_gate_v1",
        "status": "pending_authenticated_remote_preflight",
        "image_sha256": anchor.IMAGE_SHA256,
        "source_archive_sha256": archive_sha,
        "exact_remote_image_validate_rows_passed": 0,
        "submission_performed": False,
    })
    (OUTPUT / "README.md").write_text(
        "# Phase-III independent material-window fanout\n\n"
        "Six source-locked no-overlap SR-SNAKE rows. Only the Phase-III response "
        "coordinate scope changes from full active-plus-singleton to the "
        "candidate-material W-plus-singleton window. Powell remains full and "
        "supported-FS whitened; pruning and beam remain off.\n",
        encoding="utf-8",
    )
    verifier = f'''#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
 r=json.loads((B/"fanout_bundle_receipt.json").read_text())
 assert h(B/"source_locked.tar.gz")==r["source_archive_sha256"]
 assert r["source_archive_sha256"]=={ANCHOR_SOURCE_ARCHIVE_SHA256!r}
 revision=json.loads((B/"source_revision_manifest.json").read_text())
 assert revision["profile_request"]=={anchor.CHILD_ALIAS!r}
 assert revision["profile_contract_sha256"]=={anchor.CHILD_DIGEST!r}
 transition=revision["source_locked_route_transition"]
 assert transition["parent_profile_contract_sha256"]=={anchor.PARENT_DIGEST!r}
 assert transition["candidate_profile_contract_sha256"]=={anchor.CHILD_DIGEST!r}
 expected={EXPECTED_REGIME_MATRIX!r}
 jobs=sorted((B/"jobs").glob("*.json")); assert len(jobs)==6
 for p in jobs:
  j=json.loads(p.read_text())
  e=expected[j["regime_slug"]]; physics=j["physics"]
  assert abs(float(physics["u_over_t"])-float(e["u_over_t"]))<=1e-12
  assert abs(float(physics["lambda"])-float(e["lambda"]))<=1e-12
  assert abs(float(physics["g_ep"])-float(e["g_ep"]))<=1e-12
  assert int(physics["n_ph_work"])==int(e["n_ph"])
  assert int(physics["n_ph_reference"])==int(e["n_ph"])
  assert physics["same_cutoff_reference"] is True
  assert j["route_identity"]["profile_request"]=={anchor.CHILD_ALIAS!r}
  assert j["route_identity"]["profile_contract_sha256"]=={anchor.CHILD_DIGEST!r}
  contract=j["route_identity"]["profile_contract"]
  settings=contract["execution_settings"]
  invariants=contract["semantic_invariants"]
  assert settings["phase3_response_coordinate_scope"]=={MATERIAL_SCOPE!r}
  assert invariants["phase3_material_window_support_change_policy"]=={SUPPORT_CHANGE_POLICY!r}
  assert settings["phase1_prune_enabled"] is False
  assert settings["adapt_beam_live_branches"]==1
  assert int(j["segment"]["target_controller_round"])==50
  assert j["physics"]["same_cutoff_reference"] is True
 assert json.loads((B/"source_locked_sensitivity_audit.json").read_text())["fanout_authorized"] is True
 assert "requirements = False" not in (B/"submit.sub").read_text()
 return True
if __name__=="__main__": verify(); print("material-window fanout verified")
'''
    (OUTPUT / "build_bundle.py").write_text(verifier, encoding="utf-8")
    (OUTPUT / "test_bundle.py").write_text(
        "import build_bundle\ndef test_bundle(): assert build_bundle.verify()\n",
        encoding="utf-8",
    )
    return receipt


def _fixture_digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _deterministic_material_evidence_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert the locked WW parent result into a deterministic validator fixture.

    This is validation-only evidence.  It gives every round a closed full-W
    material receipt so the generated archive proves its material, projected,
    and no-overlap validators execute against a result rather than merely import.
    """

    with tarfile.open(anchor.SOURCE_TRANSFER, "r:gz") as handle:
        result_members = [
            item for item in handle.getmembers()
            if item.name == anchor.SOURCE_RESULT_MEMBER
        ]
        ledger_members = [
            item for item in handle.getmembers()
            if item.name.endswith("/json/estimator_call_ledger.json")
        ]
        if len(result_members) != 1 or len(ledger_members) != 1:
            raise ValueError("locked WW source lacks exact result/ledger fixture members")
        result_file = handle.extractfile(result_members[0])
        ledger_file = handle.extractfile(ledger_members[0])
        if result_file is None or ledger_file is None:
            raise ValueError("locked WW fixture members are not regular files")
        result = json.load(result_file)
        ledger = json.load(ledger_file)
    result = copy.deepcopy(result)
    ledger = copy.deepcopy(ledger)
    settings = result["settings"]
    adapt = result["adapt_vqe"]
    settings["phase3_response_coordinate_scope"] = MATERIAL_SCOPE
    settings["sr_route_profile_contract"]["semantic_invariants"][
        "phase3_material_window_support_change_policy"
    ] = SUPPORT_CHANGE_POLICY
    adapt["phase3_response_coordinate_scope"] = MATERIAL_SCOPE
    terminal_plan: dict[str, Any] | None = None
    terminal_identity: tuple[int, str, int] | None = None
    for outer_iteration, row in enumerate(adapt["history"], start=1):
        active = list(range(outer_iteration - 1))
        admission = row["admitted_records"][0]
        summary = admission["phase2_joint_geometry_reuse"]
        active_rank = int(summary.get("active_metric_support_rank", 0))
        joint_rank = int(summary.get("joint_metric_support_rank", 0))
        policy = copy.deepcopy({
            "policy_version": "phase3_material_window_policy_v1",
            "gram_entry_threshold": 4.0e-3,
            "hessian_entry_threshold": 2.0e-22,
            "gram_omitted_l2_tolerance": 1.0,
            "hessian_omitted_l2_tolerance": 1.0,
            "gram_cross_block_tolerance": 1.0e-1,
            "hessian_cross_block_tolerance": 1.0e-1,
            "epsilon": 1.0e-12,
        })
        receipt: dict[str, Any] = {
            "receipt_version": "phase3_material_window_receipt_v1",
            "policy": policy,
            "active_indices": active,
            "prior_active_nullity": None,
            "prior_joint_nullity": None,
            "gram_normalized_scores": [1.0 for _ in active],
            "hessian_normalized_scores": [1.0 for _ in active],
            "initial_gram_mask": [True for _ in active],
            "initial_hessian_mask": [True for _ in active],
            "initial_union_mask": [True for _ in active],
            "final_retained_mask": [True for _ in active],
            "closure_added_indices": [],
            "retained_indices": active,
            "omitted_indices": [],
            "initial_gram_omitted_l2_ratio": 0.0,
            "initial_hessian_omitted_l2_ratio": 0.0,
            "final_gram_omitted_l2_ratio": 0.0,
            "final_hessian_omitted_l2_ratio": 0.0,
            "gram_entry_threshold": 4.0e-3,
            "hessian_entry_threshold": 2.0e-22,
            "gram_omitted_l2_tolerance": 1.0,
            "hessian_omitted_l2_tolerance": 1.0,
            "inputs_finite": True,
            "closure_satisfied": True,
            "closure_reason": "candidate_only" if not active else "satisfied_by_threshold_union",
            "measured_active_supported_rank": active_rank,
            "measured_joint_supported_rank": joint_rank,
            "measured_active_nullity": len(active) - active_rank,
            "measured_joint_nullity": len(active) + 1 - joint_rank,
            "measured_rank_gain": joint_rank - active_rank,
            "support_nullity_drift": False,
            "requires_full_geometry_refresh": False,
            "refresh_reasons": [],
        }
        receipt["receipt_sha256"] = hashlib.sha256(json.dumps(
            receipt, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()
        rr = [[left, right] for left in active for right in range(left, len(active))]
        candidate_pool_index = int(admission["candidate_pool_index"])
        candidate_label = str(summary["candidate_label"])
        candidate_position = len(active)
        closure = {
            "schema": "phase3_material_window_block_closure_v1",
            "retained_indices": active,
            "omitted_indices": [],
            "gram_retained_fro_norm": 0.0,
            "hessian_retained_fro_norm": 0.0,
            "gram_retained_omitted_fro_norm": 0.0,
            "hessian_retained_omitted_fro_norm": 0.0,
            "gram_retained_omitted_ratio": 0.0,
            "hessian_retained_omitted_ratio": 0.0,
            "gram_tolerance": 1.0e-1,
            "hessian_tolerance": 1.0e-1,
            "inputs_finite": True,
            "gram_satisfied": True,
            "hessian_satisfied": True,
            "closure_satisfied": True,
            "refresh_reasons": [],
        }
        plan = {
            "schema": "phase3_material_window_estimator_acquisition_plan_v1",
            **{
                key: _fixture_digest(f"{outer_iteration}:{key}") for key in (
                    "state_fingerprint", "ordered_scaffold_fingerprint",
                    "theta_fingerprint", "hamiltonian_fingerprint",
                    "candidate_coordinate_fingerprint",
                )
            },
            "candidate_pool_index": candidate_pool_index,
            "candidate_label": candidate_label,
            "candidate_position_id": candidate_position,
            "active_indices": active,
            "screen_gram_diagonal_indices": active,
            "candidate_cross_gram_active_indices": active,
            "candidate_cross_hessian_active_indices": active,
            "candidate_self_gram_acquired": True,
            "candidate_self_hessian_acquired": True,
            "candidate_gradient_acquired": True,
            "retained_indices": active,
            "omitted_indices": [],
            "retained_retained_pairs": rr,
            "retained_omitted_closure_pairs": [],
            "omitted_omitted_refresh_pairs": [],
            "old_old_metric_pairs_acquired": rr,
            "old_old_hessian_pairs_acquired": rr,
            "active_gradient_indices_acquired": active,
            "retained_omitted_closure_acquired": False,
            "retained_omitted_block_closure": closure,
            "screen_gram_diagonal_indices_reused_in_old_old_pairs": active,
            "full_geometry_refresh_performed": False,
            "full_geometry_refresh_count": 0,
            "screen_gram_diagonal_count": len(active),
            "candidate_cross_gram_count": len(active),
            "candidate_cross_hessian_count": len(active),
            "old_old_metric_pair_count": len(rr),
            "old_old_hessian_pair_count": len(rr),
            "retained_retained_pair_count": len(rr),
            "retained_omitted_closure_pair_count": 0,
            "omitted_omitted_refresh_pair_count": 0,
            "active_gradient_count": len(active),
        }
        summary.update({
            "material_window_receipt": receipt,
            "prior_nullity_comparison_scope": "same_retained_W_and_W_plus_candidate_v1",
            "material_window_refresh": {
                "performed": False, "count": 0, "reasons": [],
                "retained_supported_rank": active_rank,
                "retained_joint_supported_rank": joint_rank,
                "final_active_indices": active,
                "refresh_sparse_acquisition": None,
            },
            "estimator_acquisition_plan": plan,
        })
        admission["material_window_receipt"] = receipt
        admission["estimator_acquisition_plan"] = plan
        row["phase3_response_coordinate_scope"] = MATERIAL_SCOPE
        row["phase3_response_coordinate_indices"] = [*active, candidate_position]
        row["phase3_response_pre_support_count"] = len(active) + 1
        trust = row["route_a_trust_region_update"]
        trust.update({
            "geometry_expansion_active": True,
            "update_reason": "geometry_expansion_no_coordinate_prediction_no_overlap_hold",
            "source_metric_trust_transaction": None,
            "source_metric_trust_transaction_failure": None,
            "radius_after": trust["radius_before"],
        })
        if outer_iteration < len(adapt["history"]):
            adapt["history"][outer_iteration]["route_a_trust_region_update"][
                "radius_before"
            ] = trust["radius_after"]
        terminal_plan = plan
        terminal_identity = (
            candidate_pool_index, candidate_label, candidate_position,
        )
    if terminal_plan is None or terminal_identity is None:
        raise ValueError("locked WW fixture has no terminal material plan")
    raw_ledger = ledger["ledger"]
    occurrence_ids = {
        str(row["primitive_id"]) for row in raw_ledger["occurrences"]
    }
    available_ids = [
        str(row["primitive_id"]) for row in raw_ledger["entries"]
        if str(row["primitive_id"]) in occurrence_ids
    ]
    gradient_count = int(terminal_plan["active_gradient_count"])
    if len(available_ids) < max(gradient_count, 64):
        raise ValueError("locked WW ledger lacks fixture primitive identities")
    primitive_ids = sorted(set(available_ids[:64]))
    gradient_ids = available_ids[:gradient_count]
    active_count = len(terminal_plan["active_indices"])
    occurrence_count = (
        active_count
        + int(terminal_plan["old_old_metric_pair_count"])
        + int(terminal_plan["old_old_hessian_pair_count"])
        + gradient_count
        + 2 * active_count + 3
    )
    record = {
        "schema": "phase3_material_window_estimator_accounting_v1",
        "candidate_pool_index": terminal_identity[0],
        "candidate_label": terminal_identity[1],
        "candidate_position_id": terminal_identity[2],
        "full_geometry_refresh_performed": False,
        "source_plan": terminal_plan,
        "active_gradient_accounting": {
            "schema": "phase3_active_gradient_query_accounting_v1",
            "active_coordinate_count": gradient_count,
            "new_unique_gradients_charged": 0,
            "deduplicated_or_ledger_disabled_count": gradient_count,
            "primitive_ids": gradient_ids,
            "component": "N_grad",
            "consumer_scope": "fixture:active_gradient",
        },
        "primitive_occurrence_id_count": occurrence_count,
        "unique_primitive_id_count": len(primitive_ids),
        "primitive_ids": primitive_ids,
        "identity_deduplication": "estimator_call_ledger_global_v1",
    }
    overlay = adapt["continuation"]["runtime_split_summary"][
        "historical_singleton_coordinate_overlay_last_round"
    ]
    overlay["active_gradient_query_accounting"] = {
        "schema": "phase3_material_window_population_estimator_accounting_v1",
        "record_count": 1,
        "records": [record],
        "identity_deduplication": "estimator_call_ledger_global_v1",
    }
    return result, ledger


def archive_preflight() -> None:
    with tempfile.TemporaryDirectory(prefix="paper-i-material-window-fanout-preflight-") as raw:
        root = Path(raw)
        with tarfile.open(OUTPUT / "source_locked.tar.gz", "r:gz") as archive_file:
            archive_file.extractall(root, filter="data")
        target = root / "chtc/phase3_optuna/input" / OUTPUT_ID
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(OUTPUT, target)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)
        env.pop("PYTHONNOUSERSITE", None)
        for job in sorted((target / "jobs").glob("*.json")):
            subprocess.run(
                [sys.executable, str(target / "run_job.py"), "--validate-only", str(job)],
                cwd=root, env=env, check=True,
            )
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *anchor.FOCUSED_TEST_OVERLAYS],
            cwd=root, env=env, check=True,
        )
        fixture_result, fixture_ledger = _deterministic_material_evidence_fixture()
        fixture_result_path = root / "material_window_result_fixture.json"
        fixture_ledger_path = root / "material_window_ledger_fixture.json"
        fixture_result_path.write_text(
            json.dumps(
                fixture_result, indent=2, sort_keys=True, allow_nan=True,
            ) + "\n",
            encoding="utf-8",
        )
        anchor.dump(fixture_ledger_path, fixture_ledger)
        fixture_code = r'''import json,sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from evidence_validation import (
    material_no_overlap_validation_view, material_parent_validation_view,
    validate_material_window_evidence, validate_no_overlap_trust_evidence,
    validate_projected_generalized_phase3_evidence,
)
result=json.loads(Path(sys.argv[2]).read_text())
ledger=json.loads(Path(sys.argv[3]).read_text())
material=validate_material_window_evidence(
    result=result, target_round=50, ledger_sidecar=ledger,
)
projected=validate_projected_generalized_phase3_evidence(
    result=material_parent_validation_view(result), target_round=50,
)
no_overlap=validate_no_overlap_trust_evidence(
    result=material_no_overlap_validation_view(result), target_round=50,
)
assert material["status"]==projected["status"]==no_overlap["status"]=="pass"
print("deterministic material-window result evidence: pass")
'''
        subprocess.run(
            [
                sys.executable, "-c", fixture_code, str(target),
                str(fixture_result_path), str(fixture_ledger_path),
            ],
            cwd=root, env=env, check=True,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-result-json", type=Path, required=True)
    parser.add_argument("--anchor-validation-json", type=Path, required=True)
    parser.add_argument("--anchor-transfer-archive", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = build_fanout(
        anchor_result=args.anchor_result_json,
        anchor_validation=args.anchor_validation_json,
        anchor_transfer=args.anchor_transfer_archive,
    )
    subprocess.run([sys.executable, str(OUTPUT / "build_bundle.py")], check=True)
    subprocess.run([sys.executable, "-m", "pytest", "-q", str(OUTPUT / "test_bundle.py")], check=True)
    archive_preflight()
    preflight = anchor.load(OUTPUT / "preflight.json")
    preflight["status"] = "pass"
    preflight["checks"]["archive_only_worker_validation"] = True
    preflight["checks"]["archive_focused_tests"] = True
    preflight["checks"]["deterministic_result_evidence_validation"] = True
    anchor.dump(OUTPUT / "preflight.json", preflight)
    anchor.dump(OUTPUT / "archive_only_preflight.json", preflight)
    anchor.dump(OUTPUT / "submission_artifact_hashes.json", {
        "schema": "paper_i_sr_material_window_fanout_submission_artifacts_v1",
        "files": {
            path.relative_to(OUTPUT).as_posix(): {
                "sha256": anchor.sha256(path), "size_bytes": path.stat().st_size,
            }
            for path in sorted(OUTPUT.rglob("*"))
            if path.is_file() and path.name != "submission_artifact_hashes.json"
        },
    })
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
