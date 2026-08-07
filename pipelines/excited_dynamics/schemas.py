from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


QSE_RESULT_SCHEMA_VERSION = "qse_spectra_v1"
EXCITED_STATE_SEED_SCHEMA_VERSION = "excited_state_seed_v1"
EXCITED_STATE_SEED_PIPELINE = "excited_dynamics"
QSE_RITZ_STATEVECTOR_MODE = "qse_ritz_statevector_diagnostic"
LEGACY_QSE_BASIS_VECTOR_NORMALIZATION = "qse_core_normalized_B_i_psi"
DIAGNOSTIC_QSE_PAYLOAD_REFS = (
    "static_record_selection",
    "transition_observables",
    "spectral_functions",
    "spectral_window_metrics",
    "cutoff_boundary_diagnostics",
)


class ValidationError(ValueError):
    """Raised when a QSE result or excited-state seed manifest is invalid."""


@dataclass(frozen=True)
class QSEManifestSummary:
    schema_version: str
    pipeline: str
    backend: str
    uses_qiskit: bool
    num_qubits: int
    basis_size: int
    retained_rank: int
    eigenvalue_count: int


@dataclass(frozen=True)
class ExcitedStateSeedSummary:
    schema_version: str
    pipeline: str
    state_preparation_mode: str
    state_index: int
    controller_usable: bool
    qpu_faithful_preparation: bool
    diagnostic_exact_assisted: bool
    promotion_status: str


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValidationError(f"{name} must be a sequence")
    return value


def _strict_int(value: Any, *, name: str, min_value: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{name} must be an integer")
    if min_value is not None and value < min_value:
        raise ValidationError(f"{name} must be >= {min_value}")
    return value


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a finite number")
    out = float(value)
    if out != out or out in (float("inf"), float("-inf")):
        raise ValidationError(f"{name} must be a finite number")
    return out


def _bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{name} must be boolean")
    return value


def _optional_basis_vector_policy(qse_payload: Mapping[str, Any]) -> dict[str, Any] | None:
    diagnostics = qse_payload.get("diagnostics")
    settings = qse_payload.get("settings")
    for container in (diagnostics, settings):
        if isinstance(container, Mapping) and isinstance(container.get("basis_vector_policy"), Mapping):
            raw = container["basis_vector_policy"]
            return {
                "reference_projection": str(raw.get("reference_projection", "none")),
                "basis_vector_normalization": str(raw.get("basis_vector_normalization", "normalized")),
                "sector_projection": str(raw.get("sector_projection", "identity")),
                "sector_label": raw.get("sector_label"),
            }
    return None


def _root_index_zero_role(qse_payload: Mapping[str, Any]) -> str:
    """Resolve whether root zero is a ground Ritz state or a Q0 excitation."""

    policy = _optional_basis_vector_policy(qse_payload)
    if policy is not None and str(policy.get("reference_projection", "none")) == "q0":
        return "lowest_orthogonal_ritz_root"
    return "lowest_qse_ritz_root"


def _coefficients(eigenvalue: Mapping[str, Any], *, basis_size: int, index: int) -> Sequence[Any]:
    coeffs = _sequence(eigenvalue.get("basis_coefficients"), name=f"eigenvalues[{index}].basis_coefficients")
    if len(coeffs) != basis_size:
        raise ValidationError(
            f"eigenvalues[{index}].basis_coefficients length {len(coeffs)} does not match basis_size {basis_size}"
        )
    seen: set[int] = set()
    for coeff_index, coeff in enumerate(coeffs):
        record = _mapping(coeff, name=f"eigenvalues[{index}].basis_coefficients[{coeff_index}]")
        basis_index = _strict_int(
            record.get("basis_index"),
            name=f"eigenvalues[{index}].basis_coefficients[{coeff_index}].basis_index",
            min_value=0,
        )
        if basis_index >= basis_size:
            raise ValidationError(f"basis coefficient index {basis_index} exceeds basis_size {basis_size}")
        seen.add(basis_index)
        _finite_float(record.get("re"), name=f"eigenvalues[{index}].basis_coefficients[{coeff_index}].re")
        _finite_float(record.get("im"), name=f"eigenvalues[{index}].basis_coefficients[{coeff_index}].im")
    if len(seen) != basis_size:
        raise ValidationError(f"eigenvalues[{index}].basis_coefficients must cover each basis index exactly once")
    return coeffs


def validate_qse_result_manifest(payload: Mapping[str, Any]) -> QSEManifestSummary:
    payload = _mapping(payload, name="qse_result")
    schema_version = payload.get("schema_version")
    if schema_version != QSE_RESULT_SCHEMA_VERSION:
        raise ValidationError(f"schema_version must be {QSE_RESULT_SCHEMA_VERSION!r}")
    pipeline = payload.get("pipeline")
    if pipeline != "qse_spectra":
        raise ValidationError("pipeline must be 'qse_spectra'")
    backend = payload.get("backend")
    if backend != "ideal_statevector":
        raise ValidationError("backend must be 'ideal_statevector'")
    uses_qiskit = _bool(payload.get("uses_qiskit"), name="uses_qiskit")
    if uses_qiskit:
        raise ValidationError("QSE result must not use qiskit")

    diagnostics = _mapping(payload.get("diagnostics"), name="diagnostics")
    num_qubits = _strict_int(diagnostics.get("num_qubits"), name="diagnostics.num_qubits", min_value=1)
    basis_size = _strict_int(diagnostics.get("basis_size"), name="diagnostics.basis_size", min_value=1)
    retained_rank = _strict_int(diagnostics.get("retained_rank"), name="diagnostics.retained_rank", min_value=0)
    if retained_rank > basis_size:
        raise ValidationError("diagnostics.retained_rank cannot exceed diagnostics.basis_size")

    operator_basis = _sequence(payload.get("operator_basis"), name="operator_basis")
    if len(operator_basis) != basis_size:
        raise ValidationError(f"operator_basis length {len(operator_basis)} does not match basis_size {basis_size}")
    for idx, element in enumerate(operator_basis):
        record = _mapping(element, name=f"operator_basis[{idx}]")
        basis_index = _strict_int(record.get("basis_index"), name=f"operator_basis[{idx}].basis_index", min_value=0)
        if basis_index != idx:
            raise ValidationError(f"operator_basis[{idx}].basis_index must equal {idx}")

    eigenvalues = _sequence(payload.get("eigenvalues"), name="eigenvalues")
    if len(eigenvalues) == 0:
        raise ValidationError("eigenvalues must be non-empty")
    for idx, eigenvalue in enumerate(eigenvalues):
        record = _mapping(eigenvalue, name=f"eigenvalues[{idx}]")
        state_index = _strict_int(record.get("state_index"), name=f"eigenvalues[{idx}].state_index", min_value=0)
        if state_index != idx:
            raise ValidationError(f"eigenvalues[{idx}].state_index must equal {idx}")
        _finite_float(record.get("energy"), name=f"eigenvalues[{idx}].energy")
        if "generalized_residual_norm" in record and record.get("generalized_residual_norm") is not None:
            _finite_float(record.get("generalized_residual_norm"), name=f"eigenvalues[{idx}].generalized_residual_norm")
        _coefficients(record, basis_size=basis_size, index=idx)

    return QSEManifestSummary(
        schema_version=schema_version,
        pipeline=pipeline,
        backend=backend,
        uses_qiskit=uses_qiskit,
        num_qubits=num_qubits,
        basis_size=basis_size,
        retained_rank=retained_rank,
        eigenvalue_count=len(eigenvalues),
    )


def _select_eigenvalue(payload: Mapping[str, Any], *, state_index: int, allow_ground_state: bool) -> Mapping[str, Any]:
    if state_index == 0 and _root_index_zero_role(payload) == "lowest_qse_ritz_root" and not allow_ground_state:
        raise ValidationError("state_index=0 is the QSE ground Ritz state; pass allow_ground_state=True to build it")
    eigenvalues = _sequence(payload.get("eigenvalues"), name="eigenvalues")
    if state_index < 0 or state_index >= len(eigenvalues):
        raise ValidationError(f"state_index {state_index} out of range for {len(eigenvalues)} eigenvalues")
    record = _mapping(eigenvalues[state_index], name=f"eigenvalues[{state_index}]")
    actual = _strict_int(record.get("state_index"), name=f"eigenvalues[{state_index}].state_index", min_value=0)
    if actual != state_index:
        raise ValidationError(f"selected eigenvalue state_index {actual} does not equal requested {state_index}")
    return record


def build_excited_state_seed_manifest(
    qse_payload: Mapping[str, Any],
    *,
    state_index: int,
    source_qse_path: str | Path | None = None,
    source_qse_sha256: str | None = None,
    allow_ground_state: bool = False,
) -> dict[str, Any]:
    summary = validate_qse_result_manifest(qse_payload)
    selected = _select_eigenvalue(qse_payload, state_index=state_index, allow_ground_state=allow_ground_state)
    diagnostics = _mapping(qse_payload.get("diagnostics"), name="diagnostics")
    operator_basis = _sequence(qse_payload.get("operator_basis"), name="operator_basis")

    energy = _finite_float(selected.get("energy"), name=f"eigenvalues[{state_index}].energy")
    gap = selected.get("energy_relative_to_lowest_qse")
    gap_value = _finite_float(gap, name=f"eigenvalues[{state_index}].energy_relative_to_lowest_qse") if gap is not None else None
    reference_gap = selected.get("energy_relative_to_reference")
    reference_gap_value = (
        _finite_float(
            reference_gap,
            name=f"eigenvalues[{state_index}].energy_relative_to_reference",
        )
        if reference_gap is not None
        else None
    )
    residual = selected.get("generalized_residual_norm")
    residual_value = _finite_float(residual, name=f"eigenvalues[{state_index}].generalized_residual_norm") if residual is not None else None
    basis_vector_policy = _optional_basis_vector_policy(qse_payload)
    root_role = _root_index_zero_role(qse_payload) if state_index == 0 else "qse_ritz_root"
    excitation_energy = (
        reference_gap_value
        if root_role == "lowest_orthogonal_ritz_root"
        else gap_value
    )
    excitation_energy_reference = (
        "prepared_reference_state"
        if root_role == "lowest_orthogonal_ritz_root"
        else "lowest_qse_ritz_root"
    )
    basis_block = {
        "basis_size": summary.basis_size,
        "basis_vector_normalization": (
            LEGACY_QSE_BASIS_VECTOR_NORMALIZATION
            if basis_vector_policy is None
            else str(basis_vector_policy.get("basis_vector_normalization", LEGACY_QSE_BASIS_VECTOR_NORMALIZATION))
        ),
        "operator_basis_hash_source": "qse_manifest.operator_basis",
        "operator_basis": list(operator_basis),
    }
    if basis_vector_policy is not None:
        basis_block["basis_vector_policy"] = basis_vector_policy

    return {
        "schema_version": EXCITED_STATE_SEED_SCHEMA_VERSION,
        "pipeline": EXCITED_STATE_SEED_PIPELINE,
        "generated_utc": datetime.now(UTC).isoformat(),
        "seed_kind": "qse_ritz_state_seed",
        "state_preparation_mode": QSE_RITZ_STATEVECTOR_MODE,
        "promotion_status": "diagnostic",
        "qpu_faithful_preparation": False,
        "diagnostic_exact_assisted": True,
        "controller_exact_input_mode": "off",
        "diagnostic_exact_reference_mode": "benchmark_exact",
        "controller_boundary": {
            "controller_usable": False,
            "requires_scaffold_fit": True,
            "qpu_faithful_state_prep_eligible": False,
            "decision_path_allowed": False,
            "post_run_diagnostic_only": True,
            "feeds_controller_decisions": False,
        },
        "source": {
            "qse_schema_version": summary.schema_version,
            "qse_pipeline": summary.pipeline,
            "qse_backend": summary.backend,
            "source_qse_path": str(source_qse_path) if source_qse_path is not None else None,
            "source_qse_sha256": source_qse_sha256,
            "qse_generated_utc": qse_payload.get("generated_utc"),
        },
        "model": {
            "settings": qse_payload.get("settings", {}),
            "num_qubits": summary.num_qubits,
        },
        "basis": basis_block,
        "qse_ritz": {
            "state_index": state_index,
            "root_role": root_role,
            "energy": energy,
            "energy_relative_to_reference": reference_gap_value,
            "energy_relative_to_lowest_qse": gap_value,
            "excitation_energy": excitation_energy,
            "excitation_energy_reference": excitation_energy_reference,
            "generalized_residual_norm": residual_value,
            "basis_coefficients": list(_coefficients(selected, basis_size=summary.basis_size, index=state_index)),
            "retained_rank": summary.retained_rank,
            "discarded_rank": diagnostics.get("discarded_rank"),
            "overlap_condition_estimate": diagnostics.get("overlap_condition_estimate"),
            "overlap_pruning_threshold": diagnostics.get("overlap_pruning_threshold"),
        },
        "visibility": {
            "controller_visible_payload_refs": [],
            "diagnostic_only_payload_refs": [
                "source",
                "basis",
                "qse_ritz",
                "benchmark_diagnostics",
                *DIAGNOSTIC_QSE_PAYLOAD_REFS,
            ],
            "forbidden_to_controller_refs": [
                "qse_ritz.energy",
                "qse_ritz.energy_relative_to_reference",
                "qse_ritz.energy_relative_to_lowest_qse",
                "qse_ritz.excitation_energy",
                "qse_ritz.basis_coefficients",
                "qse_ritz.generalized_residual_norm",
                "benchmark_diagnostics",
                "exact_statevectors",
                "exact_target_trajectories",
                *DIAGNOSTIC_QSE_PAYLOAD_REFS,
            ],
        },
        "benchmark_diagnostics": {
            "available": False,
            "feeds_controller_decisions": False,
            "records": [],
        },
        "warnings": [
            "qse_ritz_state_seed_is_diagnostic_not_controller_artifact",
            "qse_sorted_ritz_index_is_not_a_physical_quantum_number",
            "requires_scaffold_fit_before_realtime_controller_use",
            *(
                ["q0_root_zero_is_lowest_orthogonal_ritz_root_not_ground_state"]
                if root_role == "lowest_orthogonal_ritz_root"
                else []
            ),
        ],
    }


def validate_excited_state_seed_manifest(payload: Mapping[str, Any]) -> ExcitedStateSeedSummary:
    payload = _mapping(payload, name="excited_state_seed")
    if payload.get("schema_version") != EXCITED_STATE_SEED_SCHEMA_VERSION:
        raise ValidationError(f"schema_version must be {EXCITED_STATE_SEED_SCHEMA_VERSION!r}")
    if payload.get("pipeline") != EXCITED_STATE_SEED_PIPELINE:
        raise ValidationError(f"pipeline must be {EXCITED_STATE_SEED_PIPELINE!r}")
    mode = payload.get("state_preparation_mode")
    if mode != QSE_RITZ_STATEVECTOR_MODE:
        raise ValidationError(f"state_preparation_mode must be {QSE_RITZ_STATEVECTOR_MODE!r}")
    qpu_faithful = _bool(payload.get("qpu_faithful_preparation"), name="qpu_faithful_preparation")
    diagnostic_exact_assisted = _bool(payload.get("diagnostic_exact_assisted"), name="diagnostic_exact_assisted")
    if qpu_faithful:
        raise ValidationError("qse_ritz_statevector_diagnostic seeds cannot be marked qpu_faithful_preparation=true")
    if not diagnostic_exact_assisted:
        raise ValidationError("qse_ritz_statevector_diagnostic seeds must be diagnostic_exact_assisted=true")
    if payload.get("controller_exact_input_mode") != "off":
        raise ValidationError("controller_exact_input_mode must be 'off'")
    promotion_status = payload.get("promotion_status")
    if promotion_status != "diagnostic":
        raise ValidationError("promotion_status must be 'diagnostic'")

    boundary = _mapping(payload.get("controller_boundary"), name="controller_boundary")
    controller_usable = _bool(boundary.get("controller_usable"), name="controller_boundary.controller_usable")
    if controller_usable:
        raise ValidationError("qse_ritz_statevector_diagnostic seeds cannot be controller_usable")
    if not _bool(boundary.get("requires_scaffold_fit"), name="controller_boundary.requires_scaffold_fit"):
        raise ValidationError("requires_scaffold_fit must be true")
    if _bool(boundary.get("decision_path_allowed"), name="controller_boundary.decision_path_allowed"):
        raise ValidationError("decision_path_allowed must be false")
    if _bool(boundary.get("feeds_controller_decisions"), name="controller_boundary.feeds_controller_decisions"):
        raise ValidationError("feeds_controller_decisions must be false")

    qse_ritz = _mapping(payload.get("qse_ritz"), name="qse_ritz")
    state_index = _strict_int(qse_ritz.get("state_index"), name="qse_ritz.state_index", min_value=0)
    _finite_float(qse_ritz.get("energy"), name="qse_ritz.energy")
    coeffs = _sequence(qse_ritz.get("basis_coefficients"), name="qse_ritz.basis_coefficients")
    basis = _mapping(payload.get("basis"), name="basis")
    basis_size = _strict_int(basis.get("basis_size"), name="basis.basis_size", min_value=1)
    if len(coeffs) != basis_size:
        raise ValidationError("qse_ritz.basis_coefficients length must match basis.basis_size")

    return ExcitedStateSeedSummary(
        schema_version=payload["schema_version"],
        pipeline=payload["pipeline"],
        state_preparation_mode=mode,
        state_index=state_index,
        controller_usable=controller_usable,
        qpu_faithful_preparation=qpu_faithful,
        diagnostic_exact_assisted=diagnostic_exact_assisted,
        promotion_status=promotion_status,
    )
