#!/usr/bin/env python3
"""Opt-in promoted-ansatz McLachlan plumbing smoke for Paper III P6a.

This module intentionally does not import or modify the default realtime
controller routes.  It accepts only a validated
``qse_runtime_promoted_ansatz_v1`` artifact, exposes only its
``runtime_payload`` to the scaffold runtime loader, and runs a tiny fixed-step
ideal-observable McLachlan Euler trajectory on the locked promoted scaffold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.scaffold.runtime_loader import load_scaffold_runtime_input_from_payload
from pipelines.time_dynamics.legacy.checkpoint_types import (
    DECISION_DATA_FLOW_IDEAL_OBSERVABLE,
    decision_data_flow_fields,
    strict_qpu_faithful_decision_contract,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix

MODULE_NAME = "pipelines.time_dynamics.legacy.experiments.qse_promoted_mclachlan_smoke"
DEFAULT_PROMOTED_ARTIFACT_JSON = Path(
    "artifacts/agent_runs/paper_iii_p5b_validated_runtime_promotion_smoke/"
    "qse_runtime_promoted_ansatz.json"
)
DEFAULT_OUTPUT_JSON = Path(
    "artifacts/agent_runs/paper_iii_p6a_promoted_mclachlan_smoke/"
    "qse_promoted_mclachlan_run.json"
)

PROMOTED_SCHEMA_VERSION = "qse_runtime_promoted_ansatz_v1"
RUNTIME_PAYLOAD_PIPELINE = "promoted_ansatz_runtime_payload_v1"
OUTPUT_SCHEMA_VERSION = "qse_promoted_mclachlan_run_v1"

_ALLOWED_EXACT_DECISION_KEYS = {
    "controller_exact_input_mode",
    "diagnostic_exact_reference_mode",
    "uses_future_exact_forecast_for_decision",
}
_FORBIDDEN_KEY_TOKENS = (
    "qse",
    "basis_coefficients",
    "target_state",
    "fit_summary",
    "exact_target",
    "exact_step_forecast",
    "state_at",
    "fidelity_exact",
)
_FORBIDDEN_VALUE_TOKENS = (
    "qse_",
    "basis_coefficients",
    "target_state",
    "fit_summary",
    "exact_target",
    "exact_step_forecast",
    "state_at(",
    "fidelity_exact",
)


class QSEPromotedMclachlanSmokeError(ValueError):
    """Raised when a P6a promoted-ansatz smoke precondition fails."""


@dataclass(frozen=True)
class QSEPromotedMclachlanSmokeConfig:
    promoted_artifact_json: Path = DEFAULT_PROMOTED_ARTIFACT_JSON
    output_json: Path = DEFAULT_OUTPUT_JSON
    t_final: float = 0.02
    num_steps: int = 2
    regularization_lambda: float = 1.0e-8
    pinv_relative_cutoff: float = 1.0e-10


@dataclass(frozen=True)
class _LoadedPromotedArtifact:
    path: Path
    artifact: dict[str, Any]
    runtime_payload: dict[str, Any]
    file_sha256: str
    runtime_payload_sha256: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = Path(path).read_bytes()
    except FileNotFoundError as exc:
        raise QSEPromotedMclachlanSmokeError(f"Input artifact not found: {path}") from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise QSEPromotedMclachlanSmokeError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise QSEPromotedMclachlanSmokeError(f"Expected JSON object at {path}.")
    return dict(payload)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QSEPromotedMclachlanSmokeError(message)


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise QSEPromotedMclachlanSmokeError(f"{label} must be a JSON object.")
    return dict(value)


def _validate_promoted_artifact(path: Path) -> _LoadedPromotedArtifact:
    artifact_path = Path(path)
    artifact = _read_json_object(artifact_path)
    schema = str(artifact.get("schema_version", ""))
    _require(
        schema == PROMOTED_SCHEMA_VERSION,
        "P6a accepts only validated qse_runtime_promoted_ansatz_v1 artifacts; "
        f"got schema_version={schema!r}.",
    )
    _require(
        str(artifact.get("pipeline", "")) == "qse_runtime_promotion",
        "Promoted artifact pipeline must be 'qse_runtime_promotion'.",
    )
    _require(
        artifact.get("uses_qiskit", None) is False,
        "Promoted artifact must have uses_qiskit=false for this repo-native smoke.",
    )

    runtime_contract = _require_mapping(artifact.get("runtime_contract"), "runtime_contract")
    _require(
        str(runtime_contract.get("status", "")) == "validated",
        "Promoted artifact runtime_contract.status must be 'validated'.",
    )
    _require(
        runtime_contract.get("controller_usable", None) is True,
        "Promoted artifact runtime_contract.controller_usable must be true.",
    )

    controller_boundary = _require_mapping(
        artifact.get("controller_boundary"), "controller_boundary"
    )
    _require(
        controller_boundary.get("controller_usable", None) is True,
        "Promoted artifact controller_boundary.controller_usable must be true.",
    )
    _require(
        controller_boundary.get("matches_scaffold_runtime_contract", None) is True,
        "Promoted artifact must match the scaffold runtime contract.",
    )

    visibility = _require_mapping(artifact.get("visibility"), "visibility")
    controller_refs = visibility.get("controller_visible_payload_refs", None)
    _require(
        controller_refs == ["runtime_payload"],
        "Promoted artifact visibility.controller_visible_payload_refs must equal "
        "['runtime_payload'].",
    )

    runtime_payload = _require_mapping(artifact.get("runtime_payload"), "runtime_payload")
    _require(
        str(runtime_payload.get("pipeline", "")) == RUNTIME_PAYLOAD_PIPELINE,
        "runtime_payload.pipeline must be 'promoted_ansatz_runtime_payload_v1'.",
    )

    raw = artifact_path.read_bytes()
    return _LoadedPromotedArtifact(
        path=artifact_path,
        artifact=artifact,
        runtime_payload=runtime_payload,
        file_sha256=_sha256_bytes(raw),
        runtime_payload_sha256=_sha256_bytes(_canonical_json_bytes(runtime_payload)),
    )


def _validate_config(config: QSEPromotedMclachlanSmokeConfig) -> None:
    _require(float(config.t_final) > 0.0, "--t-final must be positive.")
    _require(int(config.num_steps) >= 1, "--num-steps must be at least 1.")
    _require(
        float(config.regularization_lambda) >= 0.0,
        "--regularization-lambda must be non-negative.",
    )
    _require(
        float(config.pinv_relative_cutoff) >= 0.0,
        "--pinv-relative-cutoff must be non-negative.",
    )


def _normalize_state(psi: np.ndarray) -> np.ndarray:
    arr = np.asarray(psi, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 0.0:
        raise QSEPromotedMclachlanSmokeError("Prepared ansatz state has non-positive norm.")
    return np.asarray(arr / norm, dtype=complex)


def _energy_observable_row(psi: np.ndarray, hmat: np.ndarray) -> dict[str, float]:
    state = _normalize_state(psi)
    energy = complex(np.vdot(state, np.asarray(hmat, dtype=complex) @ state))
    return {
        "energy_total_observable_re": float(np.real(energy)),
        "energy_total_observable_im_abs": float(abs(np.imag(energy))),
        "state_norm": float(np.linalg.norm(state)),
    }


def _build_executor(runtime_input: Any) -> CompiledAnsatzExecutor:
    layout = runtime_input.base_layout
    return CompiledAnsatzExecutor(
        list(runtime_input.selected_terms),
        coefficient_tolerance=float(layout.coefficient_tolerance),
        ignore_identity=bool(layout.ignore_identity),
        sort_terms=(str(layout.term_order).strip().lower() == "sorted"),
        parameterization_mode="per_pauli_term",
        parameterization_layout=layout,
    )


def _runtime_contract_assertions(runtime_input: Any, hmat: np.ndarray) -> dict[str, Any]:
    layout = runtime_input.base_layout
    theta = np.asarray(runtime_input.theta_runtime, dtype=float).reshape(-1)
    psi_ref = np.asarray(runtime_input.psi_ref, dtype=complex).reshape(-1)
    psi_initial = np.asarray(runtime_input.psi_initial, dtype=complex).reshape(-1)

    _require(bool(runtime_input.structure_locked) is True, "runtime_input.structure_locked must be true.")
    _require(
        bool(runtime_input.can_structural_edit) is False,
        "runtime_input.can_structural_edit must be false for P6a.",
    )
    _require(runtime_input.exact_energy is None, "runtime_input.exact_energy must be None.")
    _require(len(tuple(runtime_input.selected_terms)) > 0, "runtime_input.selected_terms must be non-empty.")
    _require(
        int(layout.runtime_parameter_count) == int(theta.size),
        "runtime theta length must match base_layout.runtime_parameter_count.",
    )
    _require(
        int(hmat.shape[0]) == int(hmat.shape[1]) == int(psi_initial.size) == int(psi_ref.size),
        "Hamiltonian dimension must match runtime prepared/reference state dimensions.",
    )

    return {
        "input_runtime_contract_status": "validated",
        "input_controller_usable": True,
        "loader_boundary": "runtime_payload_only",
        "structure_locked": True,
        "can_structural_edit": False,
        "reference_energy_absent": True,
        "problem_key": str(runtime_input.resolved_problem.family_key),
        "runtime_parameter_count": int(layout.runtime_parameter_count),
        "logical_operator_count": int(layout.logical_parameter_count),
        "selected_term_count": int(len(tuple(runtime_input.selected_terms))),
        "candidate_pool_source_kind": str(runtime_input.candidate_pool_source.source_kind),
        "candidate_pool_completeness": str(runtime_input.candidate_pool_source.completeness),
    }


def _base_decision_fields() -> dict[str, Any]:
    flow = decision_data_flow_fields(
        controller_mode="observable_v1",
        controller_exact_input_mode="off",
        decision_backend="ideal_observable",
        decision_noise_mode="ideal",
        strict_qpu_faithful=True,
        uses_reference_for_decision=False,
        uses_future_exact_forecast_for_decision=False,
    )
    return {
        "decision_backend": "ideal_observable",
        "decision_noise_mode": "ideal",
        "diagnostic_exact_reference_mode": "off",
        **flow,
        "append_attempted": False,
        "prune_attempted": False,
        "structure_edit_attempted": False,
    }


def _trajectory_row(
    *,
    time_index: int,
    time_value: float,
    state: np.ndarray,
    hmat: np.ndarray,
    runtime_parameter_count: int,
    logical_block_count: int,
    mclachlan_step_index: int | None,
) -> dict[str, Any]:
    return {
        "trajectory_sample_kind": "state_sample",
        "time_index": int(time_index),
        "time": float(time_value),
        "mclachlan_step_index": (
            None if mclachlan_step_index is None else int(mclachlan_step_index)
        ),
        "measurement_sample_source": "prepared_ansatz_state_ideal_observable_estimator",
        "runtime_parameter_count": int(runtime_parameter_count),
        "logical_block_count": int(logical_block_count),
        **_energy_observable_row(state, hmat),
        **_base_decision_fields(),
    }


def _solve_mclachlan_tangent_step(
    *,
    executor: CompiledAnsatzExecutor,
    psi_ref: np.ndarray,
    theta_start: np.ndarray,
    hmat: np.ndarray,
    dt: float,
    regularization_lambda: float,
    pinv_relative_cutoff: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    theta_vec = np.asarray(theta_start, dtype=float).reshape(-1)
    psi, tangents = executor.prepare_state_with_runtime_tangents(theta_vec, psi_ref)
    psi = _normalize_state(psi)
    rhs_state = -1.0j * (np.asarray(hmat, dtype=complex) @ psi)
    rhs_norm = float(np.linalg.norm(rhs_state))
    param_count = int(theta_vec.size)
    if param_count == 0:
        return theta_vec, psi, {
            "theta_dot": [],
            "rhs_norm": float(rhs_norm),
            "rhs_residual_norm": float(rhs_norm),
            "rhs_residual_ratio": 0.0 if rhs_norm <= 1.0e-15 else 1.0,
            "projected_rhs_norm": 0.0,
            "delta_norm": 0.0,
            "linear_solve_status": "no_parameters",
            "linear_solve_count": 0,
            "regularization_lambda": float(regularization_lambda),
            "pinv_relative_cutoff": float(pinv_relative_cutoff),
            "retained_rank": 0,
            "parameter_count": 0,
            "tangent_condition_estimate": None,
            "state_prep_count": 1,
            "success": True,
            "message": "no runtime parameters available",
        }

    tangent_matrix = np.column_stack(
        [np.asarray(tangents[idx], dtype=complex).reshape(-1) for idx in range(param_count)]
    )
    metric = np.real(tangent_matrix.conj().T @ tangent_matrix)
    force = np.real(tangent_matrix.conj().T @ rhs_state)
    reg = float(max(0.0, regularization_lambda))
    solve_matrix = metric + reg * np.eye(param_count)
    singular_values = np.linalg.svd(solve_matrix, compute_uv=False)
    cutoff = float(max(0.0, pinv_relative_cutoff))
    max_sv = float(np.max(singular_values)) if singular_values.size else 0.0
    retained_rank = int(sum(float(sv) > cutoff * max_sv for sv in singular_values)) if max_sv > 0.0 else 0
    positive = [float(sv) for sv in singular_values if float(sv) > 1.0e-15]
    condition = float(max(positive) / min(positive)) if positive else None
    theta_dot = np.asarray(np.linalg.pinv(solve_matrix, rcond=cutoff) @ force, dtype=float).reshape(-1)
    projected = tangent_matrix @ theta_dot
    residual = rhs_state - projected
    delta = float(dt) * theta_dot
    theta_next = theta_vec + delta
    final_state = _normalize_state(executor.prepare_state(theta_next, psi_ref))
    residual_norm = float(np.linalg.norm(residual))
    projected_norm = float(np.linalg.norm(projected))
    ratio = 0.0 if rhs_norm <= 1.0e-15 else float(residual_norm / rhs_norm)
    return theta_next, final_state, {
        "theta_dot": [float(x) for x in theta_dot.tolist()],
        "rhs_norm": float(rhs_norm),
        "rhs_residual_norm": float(residual_norm),
        "rhs_residual_ratio": float(ratio),
        "projected_rhs_norm": float(projected_norm),
        "delta_norm": float(np.linalg.norm(delta)),
        "linear_solve_status": "ok",
        "linear_solve_count": 1,
        "regularization_lambda": float(reg),
        "pinv_relative_cutoff": float(cutoff),
        "retained_rank": int(retained_rank),
        "parameter_count": int(param_count),
        "tangent_condition_estimate": condition,
        "state_prep_count": 1,
        "success": True,
        "message": "repo-native regularized tangent solve",
    }


def _run_fixed_step_trajectory(
    *,
    runtime_input: Any,
    hmat: np.ndarray,
    t_final: float,
    num_steps: int,
    regularization_lambda: float,
    pinv_relative_cutoff: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    layout = runtime_input.base_layout
    executor = _build_executor(runtime_input)
    psi_ref = _normalize_state(np.asarray(runtime_input.psi_ref, dtype=complex).reshape(-1))
    theta_current = np.asarray(runtime_input.theta_runtime, dtype=float).reshape(-1)
    current_state = _normalize_state(executor.prepare_state(theta_current, psi_ref))
    initial_state = _normalize_state(np.asarray(runtime_input.psi_initial, dtype=complex).reshape(-1))
    prepared_state_delta_norm = float(np.linalg.norm(current_state - initial_state))

    times = np.linspace(0.0, float(t_final), int(num_steps) + 1)
    trajectory: list[dict[str, Any]] = [
        _trajectory_row(
            time_index=0,
            time_value=float(times[0]),
            state=current_state,
            hmat=hmat,
            runtime_parameter_count=int(theta_current.size),
            logical_block_count=int(layout.logical_parameter_count),
            mclachlan_step_index=None,
        )
    ]
    steps: list[dict[str, Any]] = []

    for interval_index, (left, right) in enumerate(zip(times[:-1], times[1:])):
        dt = float(right - left)
        theta_before = np.asarray(theta_current, dtype=float).reshape(-1)
        theta_next, next_state, fit = _solve_mclachlan_tangent_step(
            executor=executor,
            psi_ref=psi_ref,
            theta_start=theta_before,
            hmat=hmat,
            dt=dt,
            regularization_lambda=float(regularization_lambda),
            pinv_relative_cutoff=float(pinv_relative_cutoff),
        )
        step_payload = {
            "interval_index": int(interval_index),
            "time_start": float(left),
            "time_stop": float(right),
            "dt": float(dt),
            "controller_action": "advance_fixed_step",
            "theta_runtime_before": [float(x) for x in theta_before.tolist()],
            "theta_runtime_after": [float(x) for x in np.asarray(theta_next, dtype=float).reshape(-1).tolist()],
            "runtime_parameter_count": int(theta_next.size),
            "logical_block_count": int(layout.logical_parameter_count),
            **fit,
            **_base_decision_fields(),
        }
        steps.append(step_payload)
        theta_current = np.asarray(theta_next, dtype=float).reshape(-1)
        current_state = np.asarray(next_state, dtype=complex).reshape(-1)
        trajectory.append(
            _trajectory_row(
                time_index=int(interval_index) + 1,
                time_value=float(right),
                state=current_state,
                hmat=hmat,
                runtime_parameter_count=int(theta_current.size),
                logical_block_count=int(layout.logical_parameter_count),
                mclachlan_step_index=int(interval_index),
            )
        )

    residual_ratios = [float(step["rhs_residual_ratio"]) for step in steps]
    state_norm_errors = [abs(float(row["state_norm"]) - 1.0) for row in trajectory]
    initial_energy = float(trajectory[0]["energy_total_observable_re"])
    final_energy = float(trajectory[-1]["energy_total_observable_re"])
    summary = {
        "scope_label": "P6a contract/plumbing smoke only",
        "paper_iii_science_benchmark": False,
        "benchmarks_or_reporting_generated": False,
        "t_final": float(t_final),
        "num_steps": int(num_steps),
        "step_count": int(len(steps)),
        "trajectory_row_count": int(len(trajectory)),
        "final_time": float(times[-1]),
        "initial_energy_total_observable_re": float(initial_energy),
        "final_energy_total_observable_re": float(final_energy),
        "energy_total_observable_delta_re": float(final_energy - initial_energy),
        "max_rhs_residual_ratio": (max(residual_ratios) if residual_ratios else None),
        "max_state_norm_error": float(max(state_norm_errors) if state_norm_errors else 0.0),
        "prepared_state_delta_norm": float(prepared_state_delta_norm),
        "append_count": 0,
        "prune_count": 0,
        "structure_edit_count": 0,
        "qpu_faithful_contract_smoke": True,
    }
    return trajectory, steps, summary


def _key_is_forbidden(key: str) -> str | None:
    key_lower = str(key).lower()
    if "exact" in key_lower and key_lower not in _ALLOWED_EXACT_DECISION_KEYS:
        return "exact_key"
    if key_lower == "ed" or key_lower.startswith("ed_") or key_lower.endswith("_ed"):
        return "ed_key"
    for token in _FORBIDDEN_KEY_TOKENS:
        if token in key_lower:
            return f"key_token:{token}"
    return None


def _value_is_forbidden(value: str) -> str | None:
    value_lower = str(value).lower()
    for token in _FORBIDDEN_VALUE_TOKENS:
        if token in value_lower:
            return f"value_token:{token}"
    return None


def _audit_forbidden_markers(sections: Mapping[str, Any]) -> dict[str, Any]:
    hits: list[dict[str, Any]] = []

    def _walk(value: Any, path: str, *, key_name: str | None = None) -> None:
        if key_name is not None:
            reason = _key_is_forbidden(key_name)
            if reason is not None:
                hits.append({"path": path, "marker": reason, "key": str(key_name)})
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                _walk(child_value, f"{path}.{child_key}", key_name=str(child_key))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for idx, item in enumerate(value):
                _walk(item, f"{path}[{idx}]", key_name=None)
        elif isinstance(value, str):
            reason = _value_is_forbidden(value)
            if reason is not None:
                hits.append({"path": path, "marker": reason, "value": value[:120]})

    for section_name, section_payload in sections.items():
        _walk(section_payload, str(section_name), key_name=None)
    return {
        "passed": not hits,
        "hit_count": int(len(hits)),
        "hits": hits,
        "scanned_sections": [str(key) for key in sections.keys()],
        "allowlisted_exact_fields": sorted(_ALLOWED_EXACT_DECISION_KEYS),
    }


def _json_safe_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run_promoted_mclachlan_smoke(
    config: QSEPromotedMclachlanSmokeConfig,
    *,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the P6a promoted-ansatz McLachlan plumbing smoke and write JSON."""

    _validate_config(config)
    loaded = _validate_promoted_artifact(Path(config.promoted_artifact_json))

    # Critical boundary: only the extracted runtime_payload is passed to the
    # scaffold runtime loader.  The full promoted artifact remains diagnostic
    # input metadata and is never handed to controller/tangent logic.
    runtime_input = load_scaffold_runtime_input_from_payload(
        loaded.runtime_payload,
        artifact_json=loaded.path,
    )
    hmat = np.asarray(hamiltonian_matrix(runtime_input.h_poly), dtype=complex)
    runtime_contract = _runtime_contract_assertions(runtime_input, hmat)

    trajectory, mclachlan_steps, summary = _run_fixed_step_trajectory(
        runtime_input=runtime_input,
        hmat=hmat,
        t_final=float(config.t_final),
        num_steps=int(config.num_steps),
        regularization_lambda=float(config.regularization_lambda),
        pinv_relative_cutoff=float(config.pinv_relative_cutoff),
    )
    decision_flow = {
        "controller_mode": "observable_v1",
        "decision_backend": "ideal_observable",
        "decision_noise_mode": "ideal",
        "decision_data_flow": DECISION_DATA_FLOW_IDEAL_OBSERVABLE,
        "diagnostic_exact_reference_mode": "off",
        **decision_data_flow_fields(
            controller_mode="observable_v1",
            controller_exact_input_mode="off",
            decision_backend="ideal_observable",
            decision_noise_mode="ideal",
            strict_qpu_faithful=True,
            uses_reference_for_decision=False,
            uses_future_exact_forecast_for_decision=False,
        ),
    }
    strict_contract = strict_qpu_faithful_decision_contract(
        summary=decision_flow,
        reference={"reference_mode": "off", "controller_exact_input_mode": "off"},
        decision_rows=[*trajectory, *mclachlan_steps],
    )
    if not bool(strict_contract.get("passed", False)):
        raise QSEPromotedMclachlanSmokeError(
            "Strict QPU-faithful decision contract failed: "
            + ", ".join(str(x) for x in strict_contract.get("violations", []))
        )
    summary["strict_decision_contract_passed"] = True
    summary["strict_decision_contract_violations"] = []

    forbidden_marker_audit = _audit_forbidden_markers(
        {"trajectory": trajectory, "mclachlan_steps": mclachlan_steps}
    )
    if not bool(forbidden_marker_audit.get("passed", False)):
        raise QSEPromotedMclachlanSmokeError(
            "Forbidden marker audit failed for decision rows: "
            + json.dumps(forbidden_marker_audit.get("hits", []), sort_keys=True)
        )

    command_list = list(command) if command is not None else ["python", "-m", MODULE_NAME]
    artifact = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "pipeline": "qse_promoted_mclachlan_smoke",
        "generated_utc": _utc_now_iso(),
        "backend": "repo_native_statevector_ideal_observable_estimator",
        "uses_qiskit": False,
        "source": {
            "promoted_artifact_json": str(loaded.path),
            "promoted_artifact_sha256": str(loaded.file_sha256),
            "runtime_payload_sha256": str(loaded.runtime_payload_sha256),
            "controller_visible_payload_refs_used": ["runtime_payload"],
            "loader_boundary": "runtime_payload_only",
        },
        "controller_boundary": {
            "controller_usable": True,
            "matches_scaffold_runtime_contract": True,
            "runtime_payload_feeds_controller_decisions": True,
            "top_level_diagnostic_metadata_feeds_controller_decisions": False,
            "qse_diagnostics_forbidden_to_controller": True,
            "source_payload_loaded": "runtime_payload_only",
            "append_allowed": False,
            "prune_allowed": False,
            "structural_editing_allowed": False,
            "exact_target_inputs_allowed": False,
        },
        "decision_data_flow": decision_flow,
        "runtime_contract": runtime_contract,
        "trajectory": trajectory,
        "mclachlan_steps": mclachlan_steps,
        "summary": summary,
        "forbidden_marker_audit": forbidden_marker_audit,
        "strict_decision_contract_audit": strict_contract,
        "command": command_list,
        "command_rendered": shlex.join(str(part) for part in command_list),
    }
    _json_safe_dump(Path(config.output_json), artifact)
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the opt-in P6a promoted-ansatz McLachlan plumbing smoke from "
            "a validated qse_runtime_promoted_ansatz_v1 artifact."
        )
    )
    parser.add_argument(
        "--promoted-artifact-json",
        type=Path,
        default=DEFAULT_PROMOTED_ARTIFACT_JSON,
        help=f"Validated promoted ansatz artifact (default: {DEFAULT_PROMOTED_ARTIFACT_JSON}).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=f"Output qse_promoted_mclachlan_run_v1 JSON (default: {DEFAULT_OUTPUT_JSON}).",
    )
    parser.add_argument("--t-final", type=float, default=0.02)
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--regularization-lambda", type=float, default=1.0e-8)
    parser.add_argument("--pinv-relative-cutoff", type=float, default=1.0e-10)
    return parser


def _config_from_args(args: argparse.Namespace) -> QSEPromotedMclachlanSmokeConfig:
    return QSEPromotedMclachlanSmokeConfig(
        promoted_artifact_json=Path(args.promoted_artifact_json),
        output_json=Path(args.output_json),
        t_final=float(args.t_final),
        num_steps=int(args.num_steps),
        regularization_lambda=float(args.regularization_lambda),
        pinv_relative_cutoff=float(args.pinv_relative_cutoff),
    )


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv_list)
    command = ["python", "-m", MODULE_NAME, *argv_list]
    try:
        run_promoted_mclachlan_smoke(_config_from_args(args), command=command)
    except QSEPromotedMclachlanSmokeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


__all__ = [
    "DEFAULT_OUTPUT_JSON",
    "DEFAULT_PROMOTED_ARTIFACT_JSON",
    "OUTPUT_SCHEMA_VERSION",
    "QSEPromotedMclachlanSmokeConfig",
    "QSEPromotedMclachlanSmokeError",
    "build_parser",
    "main",
    "run_promoted_mclachlan_smoke",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
