"""Diagnostic-only frozen-QSE coefficient propagation sidecar.

This module consumes a ``qse_spectra_v1`` manifest with matrices included,
projects the selected QSE Ritz coefficient vector into the retained Löwdin
support, propagates it with the static projected QSE Hamiltonian, and emits a
``frozen_qse_propagation_v1`` diagnostic artifact.

The artifact is never controller-usable: it contains QSE-basis coefficients and
scalar diagnostics only, emits no raw physical statevectors, and does not import
or call realtime/controller/ED/reference routes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.excited_dynamics.io import load_json, sha256_file
from pipelines.excited_dynamics.schemas import QSE_RESULT_SCHEMA_VERSION, ValidationError, validate_qse_result_manifest


FROZEN_QSE_PROPAGATION_SCHEMA_VERSION = "frozen_qse_propagation_v1"
FROZEN_QSE_PROPAGATION_PIPELINE = "frozen_qse_propagation"
DEFAULT_SUPPORT_CUTOFF = 1.0e-12
DEFAULT_HERMITIAN_ABSOLUTE_TOLERANCE = 1.0e-10
DEFAULT_HERMITIAN_RELATIVE_TOLERANCE = 1.0e-8
DEFAULT_OVERLAP_NEGATIVE_ABSOLUTE_TOLERANCE = 1.0e-12
DEFAULT_OVERLAP_NEGATIVE_RELATIVE_TOLERANCE = 1.0e-9


class FrozenQSEPropagationError(ValueError):
    """Raised when frozen-QSE propagation input or configuration is invalid."""


@dataclass(frozen=True)
class FrozenQSEPropagationConfig:
    qse_manifest_json: Path
    initial_root_index: int
    t_final: float
    num_steps: int
    output_json: Path | None = None
    support_cutoff: float = DEFAULT_SUPPORT_CUTOFF


@dataclass(frozen=True)
class FrozenQSESupport:
    overlap: np.ndarray
    hamiltonian: np.ndarray
    overlap_eigenvalues_raw: np.ndarray
    overlap_eigenvalues_clamped: np.ndarray
    retained_overlap_indices: tuple[int, ...]
    x_map: np.ndarray
    hamiltonian_orth: np.ndarray
    overlap_condition_estimate: float | None
    overlap_hermitian_residual_max_abs: float
    hamiltonian_hermitian_residual_max_abs: float
    overlap_negative_tolerance: float
    hermitian_allowed_overlap: float
    hermitian_allowed_hamiltonian: float


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FrozenQSEPropagationError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise FrozenQSEPropagationError(f"{name} must be a sequence")
    return value


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FrozenQSEPropagationError(f"{name} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise FrozenQSEPropagationError(f"{name} must be a finite number")
    return out


def _positive_float(value: Any, *, name: str) -> float:
    out = _finite_float(value, name=name)
    if out <= 0.0:
        raise FrozenQSEPropagationError(f"{name} must be positive")
    return out


def _strict_int(value: Any, *, name: str, min_value: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FrozenQSEPropagationError(f"{name} must be an integer")
    if min_value is not None and value < min_value:
        raise FrozenQSEPropagationError(f"{name} must be >= {min_value}")
    return int(value)


def _settings_float(settings: Mapping[str, Any], key: str, default: float) -> float:
    if key not in settings or settings.get(key) is None:
        return float(default)
    return _positive_float(settings.get(key), name=f"settings.{key}")


def _complex_from_json(value: Any, *, name: str) -> complex:
    record = _mapping(value, name=name)
    return complex(
        _finite_float(record.get("re"), name=f"{name}.re"),
        _finite_float(record.get("im"), name=f"{name}.im"),
    )


def _matrix_from_json(value: Any, *, name: str, expected_size: int) -> np.ndarray:
    rows = _sequence(value, name=name)
    if len(rows) != expected_size:
        raise FrozenQSEPropagationError(f"{name} row count {len(rows)} does not match basis_size {expected_size}")
    matrix = np.zeros((expected_size, expected_size), dtype=complex)
    for row_idx, row in enumerate(rows):
        cells = _sequence(row, name=f"{name}[{row_idx}]")
        if len(cells) != expected_size:
            raise FrozenQSEPropagationError(
                f"{name}[{row_idx}] column count {len(cells)} does not match basis_size {expected_size}"
            )
        for col_idx, cell in enumerate(cells):
            matrix[row_idx, col_idx] = _complex_from_json(cell, name=f"{name}[{row_idx}][{col_idx}]")
    return matrix


def _complex_to_json(value: complex) -> dict[str, float]:
    z = complex(value)
    if not (math.isfinite(float(z.real)) and math.isfinite(float(z.imag))):
        raise FrozenQSEPropagationError("attempted to serialize a non-finite complex value")
    return {"re": float(z.real), "im": float(z.imag)}


def _vector_to_json(vector: np.ndarray, *, index_key: str) -> list[dict[str, float | int]]:
    out: list[dict[str, float | int]] = []
    flat = np.asarray(vector, dtype=complex).reshape(-1)
    for idx, value in enumerate(flat):
        out.append({index_key: int(idx), **_complex_to_json(complex(value))})
    return out


def _matrix_residual_max_abs(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0
    return float(np.max(np.abs(matrix - matrix.conj().T)))


def _max_abs_entry(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0
    return float(np.max(np.abs(matrix)))


def _hermitian_allowed(matrix: np.ndarray, *, abs_tol: float, rel_tol: float) -> float:
    return max(float(abs_tol), float(rel_tol) * max(1.0, _max_abs_entry(matrix)))


def _real_quadratic(vector: np.ndarray, matrix: np.ndarray, *, name: str) -> float:
    value = complex(vector.conj().T @ matrix @ vector)
    allowed_imag = 1.0e-10 * max(1.0, abs(value.real), abs(value.imag))
    if abs(value.imag) > allowed_imag:
        raise FrozenQSEPropagationError(f"{name} has non-negligible imaginary part {value.imag}")
    if not math.isfinite(float(value.real)):
        raise FrozenQSEPropagationError(f"{name} is not finite")
    return float(value.real)


def _coefficient_vector(eigenvalue: Mapping[str, Any], *, basis_size: int, root_index: int) -> np.ndarray:
    coeffs = _sequence(eigenvalue.get("basis_coefficients"), name=f"eigenvalues[{root_index}].basis_coefficients")
    if len(coeffs) != basis_size:
        raise FrozenQSEPropagationError(
            f"eigenvalues[{root_index}].basis_coefficients length {len(coeffs)} does not match basis_size {basis_size}"
        )
    out = np.zeros(basis_size, dtype=complex)
    seen: set[int] = set()
    for coeff_idx, coeff in enumerate(coeffs):
        record = _mapping(coeff, name=f"eigenvalues[{root_index}].basis_coefficients[{coeff_idx}]")
        basis_index = _strict_int(
            record.get("basis_index"),
            name=f"eigenvalues[{root_index}].basis_coefficients[{coeff_idx}].basis_index",
            min_value=0,
        )
        if basis_index >= basis_size:
            raise FrozenQSEPropagationError(f"basis coefficient index {basis_index} exceeds basis_size {basis_size}")
        if basis_index in seen:
            raise FrozenQSEPropagationError(f"basis coefficient index {basis_index} appears more than once")
        seen.add(basis_index)
        out[basis_index] = complex(
            _finite_float(record.get("re"), name=f"eigenvalues[{root_index}].basis_coefficients[{coeff_idx}].re"),
            _finite_float(record.get("im"), name=f"eigenvalues[{root_index}].basis_coefficients[{coeff_idx}].im"),
        )
    if len(seen) != basis_size:
        raise FrozenQSEPropagationError(f"eigenvalues[{root_index}].basis_coefficients must cover each basis index exactly once")
    return out


def _load_qse_manifest(path: Path) -> tuple[dict[str, Any], str]:
    payload = load_json(path)
    try:
        validate_qse_result_manifest(payload)
    except ValidationError as exc:
        raise FrozenQSEPropagationError(str(exc)) from exc
    return dict(payload), sha256_file(path)


def _lowdin_support(
    *,
    overlap: np.ndarray,
    hamiltonian: np.ndarray,
    support_cutoff: float,
    settings: Mapping[str, Any],
) -> FrozenQSESupport:
    if overlap.ndim != 2 or hamiltonian.ndim != 2:
        raise FrozenQSEPropagationError("Hamiltonian and overlap matrices must be 2D arrays")
    if overlap.shape[0] != overlap.shape[1]:
        raise FrozenQSEPropagationError("Overlap matrix must be square")
    if hamiltonian.shape != overlap.shape:
        raise FrozenQSEPropagationError("Hamiltonian and overlap matrix shapes must match")
    if overlap.shape[0] == 0:
        raise FrozenQSEPropagationError("Cannot propagate an empty QSE generalized eigenproblem")

    hermitian_abs_tol = _settings_float(settings, "hermitian_absolute_tolerance", DEFAULT_HERMITIAN_ABSOLUTE_TOLERANCE)
    hermitian_rel_tol = _settings_float(settings, "hermitian_relative_tolerance", DEFAULT_HERMITIAN_RELATIVE_TOLERANCE)
    negative_abs_tol = _settings_float(
        settings,
        "overlap_negative_absolute_tolerance",
        DEFAULT_OVERLAP_NEGATIVE_ABSOLUTE_TOLERANCE,
    )
    negative_rel_tol = _settings_float(
        settings,
        "overlap_negative_relative_tolerance",
        DEFAULT_OVERLAP_NEGATIVE_RELATIVE_TOLERANCE,
    )

    overlap_residual = _matrix_residual_max_abs(overlap)
    hamiltonian_residual = _matrix_residual_max_abs(hamiltonian)
    overlap_allowed = _hermitian_allowed(overlap, abs_tol=hermitian_abs_tol, rel_tol=hermitian_rel_tol)
    hamiltonian_allowed = _hermitian_allowed(hamiltonian, abs_tol=hermitian_abs_tol, rel_tol=hermitian_rel_tol)
    if overlap_residual > overlap_allowed:
        raise FrozenQSEPropagationError("Overlap matrix is non-Hermitian beyond configured tolerance")
    if hamiltonian_residual > hamiltonian_allowed:
        raise FrozenQSEPropagationError("Hamiltonian matrix is non-Hermitian beyond configured tolerance")

    overlap_h = 0.5 * (overlap + overlap.conj().T)
    hamiltonian_h = 0.5 * (hamiltonian + hamiltonian.conj().T)

    s_raw, u = np.linalg.eigh(overlap_h)
    s_raw = np.asarray(s_raw, dtype=float)
    if not np.all(np.isfinite(s_raw)):
        raise FrozenQSEPropagationError("Overlap eigenvalues must be finite")
    max_abs_s = float(np.max(np.abs(s_raw))) if s_raw.size else 0.0
    negative_tol = max(float(negative_abs_tol), float(negative_rel_tol) * max_abs_s)
    min_raw = float(np.min(s_raw))
    if min_raw < -negative_tol:
        raise FrozenQSEPropagationError(f"Overlap matrix has negative eigenvalue {min_raw} below tolerance {-negative_tol}")

    s_clamped = np.where(s_raw < 0.0, 0.0, s_raw)
    retained_mask = s_clamped >= float(support_cutoff)
    retained_indices = tuple(int(i) for i in np.nonzero(retained_mask)[0])
    if not retained_indices:
        max_clamped = float(np.max(s_clamped)) if s_clamped.size else 0.0
        raise FrozenQSEPropagationError(
            f"QSE overlap retained rank is zero; max overlap eigenvalue is {max_clamped}, support_cutoff is {support_cutoff}"
        )

    s_retained = s_clamped[list(retained_indices)]
    u_retained = u[:, list(retained_indices)]
    x_map = u_retained @ np.diag(1.0 / np.sqrt(s_retained))
    h_orth = x_map.conj().T @ hamiltonian_h @ x_map
    h_orth = 0.5 * (h_orth + h_orth.conj().T)
    condition = float(np.max(s_retained) / np.min(s_retained)) if s_retained.size else None

    return FrozenQSESupport(
        overlap=overlap_h,
        hamiltonian=hamiltonian_h,
        overlap_eigenvalues_raw=s_raw,
        overlap_eigenvalues_clamped=np.asarray(s_clamped, dtype=float),
        retained_overlap_indices=retained_indices,
        x_map=np.asarray(x_map, dtype=complex),
        hamiltonian_orth=np.asarray(h_orth, dtype=complex),
        overlap_condition_estimate=condition,
        overlap_hermitian_residual_max_abs=overlap_residual,
        hamiltonian_hermitian_residual_max_abs=hamiltonian_residual,
        overlap_negative_tolerance=float(negative_tol),
        hermitian_allowed_overlap=float(overlap_allowed),
        hermitian_allowed_hamiltonian=float(hamiltonian_allowed),
    )


def _unitary_step(y: np.ndarray, h_orth: np.ndarray, dt: float) -> np.ndarray:
    evals, evecs = np.linalg.eigh(h_orth)
    phase = np.exp(-1.0j * float(dt) * np.asarray(evals, dtype=float))
    return evecs @ (phase * (evecs.conj().T @ y))


def _trajectory_row(
    *,
    step_index: int,
    time_value: float,
    y: np.ndarray,
    c: np.ndarray,
    support: FrozenQSESupport,
    initial_projected_norm: float,
) -> dict[str, Any]:
    qse_norm = _real_quadratic(c, support.overlap, name=f"trajectory[{step_index}].qse_norm")
    retained_norm = float(np.vdot(y, y).real)
    energy = _real_quadratic(c, support.hamiltonian, name=f"trajectory[{step_index}].qse_energy_expectation")
    return {
        "step_index": int(step_index),
        "time": float(time_value),
        "qse_norm": float(qse_norm),
        "retained_support_norm": float(retained_norm),
        "qse_norm_error_vs_initial": float(abs(qse_norm - initial_projected_norm)),
        "qse_energy_expectation": float(energy),
        "retained_support_coefficients": _vector_to_json(y, index_key="retained_index"),
        "qse_basis_coefficients": _vector_to_json(c, index_key="basis_index"),
    }


def run_frozen_qse_propagation(
    config: FrozenQSEPropagationConfig,
    *,
    command: Sequence[str] | str | None = None,
) -> dict[str, Any]:
    qse_path = Path(config.qse_manifest_json)
    output_path = None if config.output_json is None else Path(config.output_json)
    initial_root_index = _strict_int(config.initial_root_index, name="initial_root_index", min_value=0)
    t_final = _finite_float(config.t_final, name="t_final")
    if t_final < 0.0:
        raise FrozenQSEPropagationError("t_final must be non-negative")
    num_steps = _strict_int(config.num_steps, name="num_steps", min_value=1)
    support_cutoff = _positive_float(config.support_cutoff, name="support_cutoff")

    payload, qse_sha256 = _load_qse_manifest(qse_path)
    summary = validate_qse_result_manifest(payload)
    settings = _mapping(payload.get("settings", {}), name="settings")
    matrices = _mapping(payload.get("matrices"), name="matrices")
    if matrices.get("included") is not True:
        raise FrozenQSEPropagationError("qse_spectra_v1 manifest must include matrices for frozen-QSE propagation")

    basis_size = int(summary.basis_size)
    overlap = _matrix_from_json(matrices.get("overlap"), name="matrices.overlap", expected_size=basis_size)
    hamiltonian = _matrix_from_json(matrices.get("hamiltonian"), name="matrices.hamiltonian", expected_size=basis_size)
    support = _lowdin_support(
        overlap=overlap,
        hamiltonian=hamiltonian,
        support_cutoff=support_cutoff,
        settings=settings,
    )

    eigenvalues = _sequence(payload.get("eigenvalues"), name="eigenvalues")
    if initial_root_index >= len(eigenvalues):
        raise FrozenQSEPropagationError(
            f"initial_root_index {initial_root_index} out of range for {len(eigenvalues)} QSE roots"
        )
    selected = _mapping(eigenvalues[initial_root_index], name=f"eigenvalues[{initial_root_index}]")
    selected_state_index = _strict_int(
        selected.get("state_index"),
        name=f"eigenvalues[{initial_root_index}].state_index",
        min_value=0,
    )
    if selected_state_index != initial_root_index:
        raise FrozenQSEPropagationError("selected QSE root state_index does not match initial_root_index")
    selected_energy = _finite_float(selected.get("energy"), name=f"eigenvalues[{initial_root_index}].energy")
    c0 = _coefficient_vector(selected, basis_size=basis_size, root_index=initial_root_index)

    y = support.x_map.conj().T @ support.overlap @ c0
    c_projected = support.x_map @ y
    initial_qse_norm = _real_quadratic(c0, support.overlap, name="initial_qse_norm")
    initial_projected_norm = _real_quadratic(c_projected, support.overlap, name="initial_projected_qse_norm")
    if initial_projected_norm <= 0.0:
        raise FrozenQSEPropagationError("selected root has non-positive retained-support norm")
    projection_residual_norm = float(np.linalg.norm(c0 - c_projected))

    dt = float(t_final) / float(num_steps)
    rows: list[dict[str, Any]] = []
    times = [float(idx) * dt for idx in range(num_steps + 1)]
    current_y = np.asarray(y, dtype=complex)
    for step_index, time_value in enumerate(times):
        current_c = support.x_map @ current_y
        rows.append(
            _trajectory_row(
                step_index=step_index,
                time_value=time_value,
                y=current_y,
                c=current_c,
                support=support,
                initial_projected_norm=initial_projected_norm,
            )
        )
        if step_index != num_steps:
            current_y = _unitary_step(current_y, support.hamiltonian_orth, dt)

    norm_errors = [float(row["qse_norm_error_vs_initial"]) for row in rows]
    retained_norm_errors = [abs(float(row["retained_support_norm"]) - initial_projected_norm) for row in rows]
    energy_values = [float(row["qse_energy_expectation"]) for row in rows]
    h_orth_evals = np.linalg.eigvalsh(support.hamiltonian_orth)

    artifact: dict[str, Any] = {
        "schema_version": FROZEN_QSE_PROPAGATION_SCHEMA_VERSION,
        "pipeline": FROZEN_QSE_PROPAGATION_PIPELINE,
        "generated_utc": _utc_now(),
        "propagation_kind": "fixed_retained_qse_support_static_hamiltonian",
        "controller_usable": False,
        "feeds_controller_decisions": False,
        "exact_or_ed_reference_used": False,
        "uses_qiskit": False,
        "controller_boundary": {
            "controller_usable": False,
            "feeds_controller_decisions": False,
            "decision_path_allowed": False,
            "post_run_diagnostic_only": True,
            "requires_scaffold_fit": True,
            "controller_exact_input_mode": "off",
            "diagnostic_exact_reference_mode": "off",
            "realtime_route_integrated": False,
        },
        "source": {
            "qse_schema_version": QSE_RESULT_SCHEMA_VERSION,
            "qse_pipeline": "qse_spectra",
            "qse_generated_utc": payload.get("generated_utc"),
            "source_qse_path": str(qse_path),
            "source_qse_sha256": qse_sha256,
            "initial_root_index": int(initial_root_index),
            "initial_root_energy": float(selected_energy),
            "qse_basis_size": int(basis_size),
            "qse_manifest_retained_rank": int(summary.retained_rank),
        },
        "command": None if command is None else (list(command) if not isinstance(command, str) else command),
        "config": {
            "t_final": float(t_final),
            "num_steps": int(num_steps),
            "dt": float(dt),
            "support_cutoff": float(support_cutoff),
        },
        "qse_support": {
            "basis_size": int(basis_size),
            "retained_rank": int(len(support.retained_overlap_indices)),
            "discarded_rank": int(basis_size - len(support.retained_overlap_indices)),
            "retained_overlap_indices": [int(i) for i in support.retained_overlap_indices],
            "overlap_eigenvalues_raw": [float(x) for x in support.overlap_eigenvalues_raw],
            "overlap_eigenvalues_clamped": [float(x) for x in support.overlap_eigenvalues_clamped],
            "overlap_condition_estimate": support.overlap_condition_estimate,
            "support_cutoff": float(support_cutoff),
            "lowdin_map_definition": "X = U_ret diag(1/sqrt(s_ret)); map is not emitted",
            "lowdin_map_emitted": False,
            "overlap_negative_tolerance": float(support.overlap_negative_tolerance),
            "overlap_hermitian_residual_max_abs": float(support.overlap_hermitian_residual_max_abs),
            "hamiltonian_hermitian_residual_max_abs": float(support.hamiltonian_hermitian_residual_max_abs),
            "overlap_hermitian_allowed_max_abs": float(support.hermitian_allowed_overlap),
            "hamiltonian_hermitian_allowed_max_abs": float(support.hermitian_allowed_hamiltonian),
        },
        "projected_hamiltonian": {
            "dimension": int(support.hamiltonian_orth.shape[0]),
            "eigenvalues": [float(x) for x in h_orth_evals],
            "hermitian_residual_max_abs": _matrix_residual_max_abs(support.hamiltonian_orth),
            "matrix_emitted": False,
        },
        "initial_condition": {
            "root_index": int(initial_root_index),
            "qse_energy": float(selected_energy),
            "qse_norm_before_retained_projection": float(initial_qse_norm),
            "qse_norm_after_retained_projection": float(initial_projected_norm),
            "retained_support_norm": float(np.vdot(y, y).real),
            "basis_projection_residual_l2": float(projection_residual_norm),
        },
        "trajectory": rows,
        "metrics": {
            "trajectory_rows": int(len(rows)),
            "retained_rank": int(len(support.retained_overlap_indices)),
            "max_qse_norm_error": float(max(norm_errors) if norm_errors else 0.0),
            "max_retained_support_norm_error": float(max(retained_norm_errors) if retained_norm_errors else 0.0),
            "initial_energy_expectation": float(energy_values[0]) if energy_values else None,
            "final_energy_expectation": float(energy_values[-1]) if energy_values else None,
            "max_energy_drift_abs": float(max(abs(value - energy_values[0]) for value in energy_values)) if energy_values else 0.0,
        },
        "visibility": {
            "controller_visible_payload_refs": [],
            "diagnostic_only_payload_refs": [
                "source",
                "qse_support",
                "projected_hamiltonian",
                "initial_condition",
                "trajectory",
                "metrics",
            ],
            "forbidden_to_controller_refs": [
                "trajectory",
                "trajectory.qse_basis_coefficients",
                "trajectory.retained_support_coefficients",
                "initial_condition.qse_energy",
                "source.initial_root_energy",
                "qse_support.overlap_eigenvalues_raw",
                "projected_hamiltonian.eigenvalues",
            ],
        },
        "warnings": [
            "frozen_qse_propagation_is_diagnostic_only",
            "qse_basis_coefficients_are_for_offline_analysis_not_controller_decisions",
            "no_ed_reference_or_raw_physical_vectors_are_emitted",
            "adaptive_qse_basis_growth_is_out_of_scope_for_this_artifact",
        ],
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Propagate QSE-basis coefficients in a fixed retained QSE support.")
    parser.add_argument("--qse-manifest-json", required=True, help="Input qse_spectra_v1 manifest with matrices included.")
    parser.add_argument("--initial-root-index", required=True, type=int, help="QSE Ritz root index used as c(0).")
    parser.add_argument("--t-final", required=True, type=float, help="Final propagation time.")
    parser.add_argument("--num-steps", required=True, type=int, help="Number of uniform unitary propagation steps.")
    parser.add_argument("--output-json", required=True, help="Output frozen_qse_propagation_v1 JSON path.")
    parser.add_argument(
        "--support-cutoff",
        type=float,
        default=DEFAULT_SUPPORT_CUTOFF,
        help="Absolute Löwdin overlap eigenvalue cutoff for the retained support.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cli_args = list(sys.argv[1:] if argv is None else argv)
    command = ["python", "-m", "pipelines.excited_dynamics.frozen_qse_propagation", *cli_args]
    run_frozen_qse_propagation(
        FrozenQSEPropagationConfig(
            qse_manifest_json=Path(args.qse_manifest_json),
            initial_root_index=args.initial_root_index,
            t_final=args.t_final,
            num_steps=args.num_steps,
            output_json=Path(args.output_json),
            support_cutoff=args.support_cutoff,
        ),
        command=command,
    )
    return 0


__all__ = [
    "DEFAULT_SUPPORT_CUTOFF",
    "FROZEN_QSE_PROPAGATION_PIPELINE",
    "FROZEN_QSE_PROPAGATION_SCHEMA_VERSION",
    "FrozenQSEPropagationConfig",
    "FrozenQSEPropagationError",
    "build_parser",
    "main",
    "run_frozen_qse_propagation",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
