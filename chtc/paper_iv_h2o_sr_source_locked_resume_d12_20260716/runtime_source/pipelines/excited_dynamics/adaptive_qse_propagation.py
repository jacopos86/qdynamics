"""Diagnostic-only adaptive-QSE coefficient propagation sidecar.

This module consumes a ``qse_spectra_v1`` manifest with full-pool matrices
included, propagates coefficients in an initially active retained QSE support,
and grows that active support at checkpoints using residual-coupling escape
scores for inactive QSE records.

The artifact is never controller-usable: it contains QSE-basis coefficients,
adaptation diagnostics, and scalar telemetry only. It emits no raw physical
statevectors and does not import or call realtime/controller/ED/reference
routes.
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


ADAPTIVE_QSE_PROPAGATION_SCHEMA_VERSION = "adaptive_qse_propagation_v1"
ADAPTIVE_QSE_PROPAGATION_PIPELINE = "adaptive_qse_propagation"
DEFAULT_SUPPORT_CUTOFF = 1.0e-12
DEFAULT_SCORE_FLOOR = 1.0e-12
DEFAULT_HERMITIAN_ABSOLUTE_TOLERANCE = 1.0e-10
DEFAULT_HERMITIAN_RELATIVE_TOLERANCE = 1.0e-8
DEFAULT_OVERLAP_NEGATIVE_ABSOLUTE_TOLERANCE = 1.0e-12
DEFAULT_OVERLAP_NEGATIVE_RELATIVE_TOLERANCE = 1.0e-9


class AdaptiveQSEPropagationError(ValueError):
    """Raised when adaptive-QSE propagation input or configuration is invalid."""


@dataclass(frozen=True)
class AdaptiveQSEPropagationConfig:
    qse_manifest_json: Path
    initial_active_indices: Sequence[int] | str
    initial_root_index: int
    t_final: float
    num_steps: int
    checkpoint_every_steps: int
    support_cutoff: float
    escape_threshold: float
    max_add_per_checkpoint: int
    max_active_records: int
    output_json: Path | None = None


@dataclass(frozen=True)
class QSESupport:
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
        raise AdaptiveQSEPropagationError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise AdaptiveQSEPropagationError(f"{name} must be a sequence")
    return value


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdaptiveQSEPropagationError(f"{name} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise AdaptiveQSEPropagationError(f"{name} must be a finite number")
    return out


def _positive_float(value: Any, *, name: str) -> float:
    out = _finite_float(value, name=name)
    if out <= 0.0:
        raise AdaptiveQSEPropagationError(f"{name} must be positive")
    return out


def _nonnegative_float(value: Any, *, name: str) -> float:
    out = _finite_float(value, name=name)
    if out < 0.0:
        raise AdaptiveQSEPropagationError(f"{name} must be non-negative")
    return out


def _strict_int(value: Any, *, name: str, min_value: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AdaptiveQSEPropagationError(f"{name} must be an integer")
    if min_value is not None and value < min_value:
        raise AdaptiveQSEPropagationError(f"{name} must be >= {min_value}")
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
        raise AdaptiveQSEPropagationError(f"{name} row count {len(rows)} does not match basis_size {expected_size}")
    matrix = np.zeros((expected_size, expected_size), dtype=complex)
    for row_idx, row in enumerate(rows):
        cells = _sequence(row, name=f"{name}[{row_idx}]")
        if len(cells) != expected_size:
            raise AdaptiveQSEPropagationError(
                f"{name}[{row_idx}] column count {len(cells)} does not match basis_size {expected_size}"
            )
        for col_idx, cell in enumerate(cells):
            matrix[row_idx, col_idx] = _complex_from_json(cell, name=f"{name}[{row_idx}][{col_idx}]")
    return matrix


def _complex_to_json(value: complex) -> dict[str, float]:
    z = complex(value)
    if not (math.isfinite(float(z.real)) and math.isfinite(float(z.imag))):
        raise AdaptiveQSEPropagationError("attempted to serialize a non-finite complex value")
    return {"re": float(z.real), "im": float(z.imag)}


def _vector_to_json(vector: np.ndarray, *, index_key: str) -> list[dict[str, float | int]]:
    out: list[dict[str, float | int]] = []
    flat = np.asarray(vector, dtype=complex).reshape(-1)
    for idx, value in enumerate(flat):
        out.append({index_key: int(idx), **_complex_to_json(complex(value))})
    return out


def _active_vector_to_json(vector: np.ndarray, active_indices: Sequence[int]) -> list[dict[str, float | int]]:
    out: list[dict[str, float | int]] = []
    flat = np.asarray(vector, dtype=complex).reshape(-1)
    if len(flat) != len(active_indices):
        raise AdaptiveQSEPropagationError("active coefficient vector length does not match active_indices")
    for local_idx, value in enumerate(flat):
        out.append(
            {
                "active_index": int(local_idx),
                "basis_index": int(active_indices[local_idx]),
                **_complex_to_json(complex(value)),
            }
        )
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
        raise AdaptiveQSEPropagationError(f"{name} has non-negligible imaginary part {value.imag}")
    if not math.isfinite(float(value.real)):
        raise AdaptiveQSEPropagationError(f"{name} is not finite")
    return float(value.real)


def _real_scalar(value: complex, *, name: str) -> float:
    z = complex(value)
    allowed_imag = 1.0e-10 * max(1.0, abs(z.real), abs(z.imag))
    if abs(z.imag) > allowed_imag:
        raise AdaptiveQSEPropagationError(f"{name} has non-negligible imaginary part {z.imag}")
    if not math.isfinite(float(z.real)):
        raise AdaptiveQSEPropagationError(f"{name} is not finite")
    return float(z.real)


def _coefficient_vector(eigenvalue: Mapping[str, Any], *, basis_size: int, root_index: int) -> np.ndarray:
    coeffs = _sequence(eigenvalue.get("basis_coefficients"), name=f"eigenvalues[{root_index}].basis_coefficients")
    if len(coeffs) != basis_size:
        raise AdaptiveQSEPropagationError(
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
            raise AdaptiveQSEPropagationError(f"basis coefficient index {basis_index} exceeds basis_size {basis_size}")
        if basis_index in seen:
            raise AdaptiveQSEPropagationError(f"basis coefficient index {basis_index} appears more than once")
        seen.add(basis_index)
        out[basis_index] = complex(
            _finite_float(record.get("re"), name=f"eigenvalues[{root_index}].basis_coefficients[{coeff_idx}].re"),
            _finite_float(record.get("im"), name=f"eigenvalues[{root_index}].basis_coefficients[{coeff_idx}].im"),
        )
    if len(seen) != basis_size:
        raise AdaptiveQSEPropagationError(f"eigenvalues[{root_index}].basis_coefficients must cover each basis index exactly once")
    return out


def _load_qse_manifest(path: Path) -> tuple[dict[str, Any], str, Any]:
    payload = load_json(path)
    try:
        summary = validate_qse_result_manifest(payload)
    except ValidationError as exc:
        raise AdaptiveQSEPropagationError(str(exc)) from exc
    return dict(payload), sha256_file(path), summary


def _lowdin_support(
    *,
    overlap: np.ndarray,
    hamiltonian: np.ndarray,
    support_cutoff: float,
    settings: Mapping[str, Any],
) -> QSESupport:
    if overlap.ndim != 2 or hamiltonian.ndim != 2:
        raise AdaptiveQSEPropagationError("Hamiltonian and overlap matrices must be 2D arrays")
    if overlap.shape[0] != overlap.shape[1]:
        raise AdaptiveQSEPropagationError("Overlap matrix must be square")
    if hamiltonian.shape != overlap.shape:
        raise AdaptiveQSEPropagationError("Hamiltonian and overlap matrix shapes must match")
    if overlap.shape[0] == 0:
        raise AdaptiveQSEPropagationError("Cannot propagate an empty QSE generalized eigenproblem")

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
        raise AdaptiveQSEPropagationError("Overlap matrix is non-Hermitian beyond configured tolerance")
    if hamiltonian_residual > hamiltonian_allowed:
        raise AdaptiveQSEPropagationError("Hamiltonian matrix is non-Hermitian beyond configured tolerance")

    overlap_h = 0.5 * (overlap + overlap.conj().T)
    hamiltonian_h = 0.5 * (hamiltonian + hamiltonian.conj().T)

    s_raw, u = np.linalg.eigh(overlap_h)
    s_raw = np.asarray(s_raw, dtype=float)
    if not np.all(np.isfinite(s_raw)):
        raise AdaptiveQSEPropagationError("Overlap eigenvalues must be finite")
    max_abs_s = float(np.max(np.abs(s_raw))) if s_raw.size else 0.0
    negative_tol = max(float(negative_abs_tol), float(negative_rel_tol) * max_abs_s)
    min_raw = float(np.min(s_raw))
    if min_raw < -negative_tol:
        raise AdaptiveQSEPropagationError(f"Overlap matrix has negative eigenvalue {min_raw} below tolerance {-negative_tol}")

    s_clamped = np.where(s_raw < 0.0, 0.0, s_raw)
    retained_mask = s_clamped >= float(support_cutoff)
    retained_indices = tuple(int(i) for i in np.nonzero(retained_mask)[0])
    if not retained_indices:
        max_clamped = float(np.max(s_clamped)) if s_clamped.size else 0.0
        raise AdaptiveQSEPropagationError(
            f"QSE overlap retained rank is zero; max overlap eigenvalue is {max_clamped}, support_cutoff is {support_cutoff}"
        )

    s_retained = s_clamped[list(retained_indices)]
    u_retained = u[:, list(retained_indices)]
    x_map = u_retained @ np.diag(1.0 / np.sqrt(s_retained))
    h_orth = x_map.conj().T @ hamiltonian_h @ x_map
    h_orth = 0.5 * (h_orth + h_orth.conj().T)
    condition = float(np.max(s_retained) / np.min(s_retained)) if s_retained.size else None

    return QSESupport(
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


def _parse_initial_active_indices(value: Sequence[int] | str) -> tuple[int, ...]:
    if isinstance(value, str):
        if value.strip() == "":
            raise AdaptiveQSEPropagationError("initial_active_indices must be non-empty")
        parts = [part.strip() for part in value.split(",")]
        if any(part == "" for part in parts):
            raise AdaptiveQSEPropagationError("initial_active_indices must be a comma-separated list of integers")
        raw_values: list[int] = []
        for part in parts:
            try:
                raw_values.append(int(part, 10))
            except ValueError as exc:
                raise AdaptiveQSEPropagationError("initial_active_indices must be a comma-separated list of integers") from exc
    else:
        raw_values = list(_sequence(value, name="initial_active_indices"))

    if len(raw_values) == 0:
        raise AdaptiveQSEPropagationError("initial_active_indices must be non-empty")
    parsed: list[int] = []
    seen: set[int] = set()
    for idx, raw in enumerate(raw_values):
        active_index = _strict_int(raw, name=f"initial_active_indices[{idx}]", min_value=0)
        if active_index in seen:
            raise AdaptiveQSEPropagationError(f"initial_active_indices contains duplicate index {active_index}")
        seen.add(active_index)
        parsed.append(active_index)
    return tuple(parsed)


def _validate_active_indices(indices: Sequence[int], *, basis_size: int) -> tuple[int, ...]:
    out = tuple(int(index) for index in indices)
    for index in out:
        if index >= basis_size:
            raise AdaptiveQSEPropagationError(f"initial_active_indices contains out-of-range index {index} for basis_size {basis_size}")
    return out


def _submatrix(matrix: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    idx = list(indices)
    return np.asarray(matrix[np.ix_(idx, idx)], dtype=complex)


def _support_for_indices(
    *,
    overlap: np.ndarray,
    hamiltonian: np.ndarray,
    active_indices: Sequence[int],
    support_cutoff: float,
    settings: Mapping[str, Any],
) -> QSESupport:
    return _lowdin_support(
        overlap=_submatrix(overlap, active_indices),
        hamiltonian=_submatrix(hamiltonian, active_indices),
        support_cutoff=support_cutoff,
        settings=settings,
    )


def _project_onto_support(
    *,
    c_seed: np.ndarray,
    support: QSESupport,
    target_norm: float,
    name: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    y = support.x_map.conj().T @ support.overlap @ c_seed
    c_projected = support.x_map @ y
    projected_norm = _real_quadratic(c_projected, support.overlap, name=f"{name}.projected_qse_norm")
    if projected_norm <= 0.0:
        raise AdaptiveQSEPropagationError(f"{name} has non-positive retained-support norm")
    target = _positive_float(target_norm, name=f"{name}.target_qse_norm")
    scale = math.sqrt(target / projected_norm)
    y_scaled = np.asarray(y * scale, dtype=complex)
    c_scaled = np.asarray(support.x_map @ y_scaled, dtype=complex)
    residual = float(np.linalg.norm(c_seed - c_projected))
    return y_scaled, c_scaled, float(projected_norm), residual


def _support_diagnostics(
    *,
    stage: str,
    step_index: int,
    time_value: float,
    active_indices: Sequence[int],
    support: QSESupport,
    support_cutoff: float,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "step_index": int(step_index),
        "time": float(time_value),
        "active_indices": [int(index) for index in active_indices],
        "active_record_count": int(len(active_indices)),
        "retained_rank": int(len(support.retained_overlap_indices)),
        "discarded_rank": int(len(active_indices) - len(support.retained_overlap_indices)),
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
    }


def _trajectory_row(
    *,
    step_index: int,
    time_value: float,
    y: np.ndarray,
    c: np.ndarray,
    active_indices: Sequence[int],
    support: QSESupport,
) -> dict[str, Any]:
    qse_norm = _real_quadratic(c, support.overlap, name=f"trajectory[{step_index}].qse_norm")
    retained_norm = float(np.vdot(y, y).real)
    energy = _real_quadratic(c, support.hamiltonian, name=f"trajectory[{step_index}].qse_energy_expectation")
    return {
        "step_index": int(step_index),
        "time": float(time_value),
        "active_indices": [int(index) for index in active_indices],
        "active_record_count": int(len(active_indices)),
        "retained_rank": int(len(support.retained_overlap_indices)),
        "qse_norm": float(qse_norm),
        "retained_support_norm": float(retained_norm),
        "qse_norm_error_vs_unit": float(abs(qse_norm - 1.0)),
        "qse_energy_expectation": float(energy),
        "retained_support_coefficients": _vector_to_json(y, index_key="retained_index"),
        "qse_basis_coefficients": _active_vector_to_json(c, active_indices),
    }


def _inactive_candidate_scores(
    *,
    full_overlap: np.ndarray,
    full_hamiltonian: np.ndarray,
    active_indices: Sequence[int],
    c_active: np.ndarray,
    score_floor: float,
) -> tuple[float, list[dict[str, Any]]]:
    active = list(active_indices)
    active_set = set(active)
    active_overlap = _submatrix(full_overlap, active)
    active_hamiltonian = _submatrix(full_hamiltonian, active)
    active_norm = _real_quadratic(c_active, active_overlap, name="checkpoint.active_qse_norm")
    if active_norm <= 0.0:
        raise AdaptiveQSEPropagationError("checkpoint active QSE norm must be positive")
    active_energy = _real_quadratic(c_active, active_hamiltonian, name="checkpoint.active_energy_numerator") / active_norm

    records: list[dict[str, Any]] = []
    for basis_index in range(int(full_overlap.shape[0])):
        if basis_index in active_set:
            continue
        h_row = np.asarray(full_hamiltonian[basis_index, active], dtype=complex).reshape(-1)
        s_row = np.asarray(full_overlap[basis_index, active], dtype=complex).reshape(-1)
        defect = complex(h_row @ c_active - active_energy * (s_row @ c_active))
        s_diag = _real_scalar(full_overlap[basis_index, basis_index], name=f"overlap[{basis_index},{basis_index}]")
        if s_diag < 0.0:
            raise AdaptiveQSEPropagationError(f"overlap diagonal {basis_index} is negative")
        denominator = max(math.sqrt(max(s_diag, 0.0)), float(score_floor))
        score = abs(defect) / denominator
        records.append(
            {
                "basis_index": int(basis_index),
                "score": float(score),
                "defect_abs": float(abs(defect)),
                "overlap_diagonal": float(s_diag),
                "denominator": float(denominator),
            }
        )
    records.sort(key=lambda record: (-float(record["score"]), int(record["basis_index"])))
    return float(active_energy), records


def _selected_eigenvalue(
    payload: Mapping[str, Any],
    *,
    initial_root_index: int,
    basis_size: int,
) -> tuple[Mapping[str, Any], float, np.ndarray]:
    eigenvalues = _sequence(payload.get("eigenvalues"), name="eigenvalues")
    if initial_root_index >= len(eigenvalues):
        raise AdaptiveQSEPropagationError(
            f"initial_root_index {initial_root_index} out of range for {len(eigenvalues)} QSE roots"
        )
    selected = _mapping(eigenvalues[initial_root_index], name=f"eigenvalues[{initial_root_index}]")
    selected_state_index = _strict_int(
        selected.get("state_index"),
        name=f"eigenvalues[{initial_root_index}].state_index",
        min_value=0,
    )
    if selected_state_index != initial_root_index:
        raise AdaptiveQSEPropagationError("selected QSE root state_index does not match initial_root_index")
    selected_energy = _finite_float(selected.get("energy"), name=f"eigenvalues[{initial_root_index}].energy")
    c0 = _coefficient_vector(selected, basis_size=basis_size, root_index=initial_root_index)
    return selected, float(selected_energy), c0


def run_adaptive_qse_propagation(
    config: AdaptiveQSEPropagationConfig,
    *,
    command: Sequence[str] | str | None = None,
) -> dict[str, Any]:
    qse_path = Path(config.qse_manifest_json)
    output_path = None if config.output_json is None else Path(config.output_json)
    initial_root_index = _strict_int(config.initial_root_index, name="initial_root_index", min_value=0)
    t_final = _finite_float(config.t_final, name="t_final")
    if t_final < 0.0:
        raise AdaptiveQSEPropagationError("t_final must be non-negative")
    num_steps = _strict_int(config.num_steps, name="num_steps", min_value=1)
    checkpoint_every_steps = _strict_int(config.checkpoint_every_steps, name="checkpoint_every_steps", min_value=1)
    support_cutoff = _positive_float(config.support_cutoff, name="support_cutoff")
    escape_threshold = _nonnegative_float(config.escape_threshold, name="escape_threshold")
    max_add_per_checkpoint = _strict_int(config.max_add_per_checkpoint, name="max_add_per_checkpoint", min_value=0)
    max_active_records = _strict_int(config.max_active_records, name="max_active_records", min_value=1)
    score_floor = _positive_float(DEFAULT_SCORE_FLOOR, name="score_floor")

    payload, qse_sha256, summary = _load_qse_manifest(qse_path)
    settings = _mapping(payload.get("settings", {}), name="settings")
    matrices = _mapping(payload.get("matrices"), name="matrices")
    if matrices.get("included") is not True:
        raise AdaptiveQSEPropagationError("qse_spectra_v1 manifest must include matrices for adaptive-QSE propagation")

    basis_size = int(summary.basis_size)
    active_indices = _validate_active_indices(_parse_initial_active_indices(config.initial_active_indices), basis_size=basis_size)
    if max_active_records < len(active_indices):
        raise AdaptiveQSEPropagationError("max_active_records must be >= len(initial_active_indices)")
    if max_active_records > basis_size:
        raise AdaptiveQSEPropagationError("max_active_records cannot exceed qse basis_size")

    overlap_raw = _matrix_from_json(matrices.get("overlap"), name="matrices.overlap", expected_size=basis_size)
    hamiltonian_raw = _matrix_from_json(matrices.get("hamiltonian"), name="matrices.hamiltonian", expected_size=basis_size)
    full_support = _lowdin_support(
        overlap=overlap_raw,
        hamiltonian=hamiltonian_raw,
        support_cutoff=support_cutoff,
        settings=settings,
    )
    overlap = full_support.overlap
    hamiltonian = full_support.hamiltonian
    selected, selected_energy, c0_full = _selected_eigenvalue(
        payload,
        initial_root_index=initial_root_index,
        basis_size=basis_size,
    )
    del selected

    initial_support = _support_for_indices(
        overlap=overlap,
        hamiltonian=hamiltonian,
        active_indices=active_indices,
        support_cutoff=support_cutoff,
        settings=settings,
    )
    c0_active_seed = np.asarray(c0_full[list(active_indices)], dtype=complex)
    initial_full_qse_norm = _real_quadratic(c0_full, full_support.overlap, name="initial_full_qse_norm")
    initial_active_seed_norm = _real_quadratic(
        c0_active_seed,
        initial_support.overlap,
        name="initial_active_restricted_qse_norm",
    )
    if initial_active_seed_norm <= 0.0:
        raise AdaptiveQSEPropagationError("initial active restriction has non-positive QSE norm")
    current_y, current_c, initial_projected_norm, initial_projection_residual = _project_onto_support(
        c_seed=c0_active_seed,
        support=initial_support,
        target_norm=1.0,
        name="initial_active_projection",
    )

    dt = float(t_final) / float(num_steps)
    times = [float(idx) * dt for idx in range(num_steps + 1)]
    support = initial_support
    trajectory: list[dict[str, Any]] = []
    adaptation_events: list[dict[str, Any]] = []
    active_support_history: list[dict[str, Any]] = [
        _support_diagnostics(
            stage="initial",
            step_index=0,
            time_value=0.0,
            active_indices=active_indices,
            support=support,
            support_cutoff=support_cutoff,
        )
    ]
    max_escape_scores: list[float] = []
    remap_norm_errors: list[float] = []

    for step_index, time_value in enumerate(times):
        current_c = np.asarray(support.x_map @ current_y, dtype=complex)
        row = _trajectory_row(
            step_index=step_index,
            time_value=time_value,
            y=current_y,
            c=current_c,
            active_indices=active_indices,
            support=support,
        )

        if step_index > 0 and step_index < num_steps and step_index % checkpoint_every_steps == 0:
            checkpoint_energy, candidate_scores = _inactive_candidate_scores(
                full_overlap=overlap,
                full_hamiltonian=hamiltonian,
                active_indices=active_indices,
                c_active=current_c,
                score_floor=score_floor,
            )
            max_score = float(candidate_scores[0]["score"]) if candidate_scores else 0.0
            max_escape_scores.append(max_score)
            remaining_capacity = int(max_active_records - len(active_indices))
            eligible = [record for record in candidate_scores if float(record["score"]) > escape_threshold]
            add_count = min(int(max_add_per_checkpoint), remaining_capacity, len(eligible))
            added_indices = [int(record["basis_index"]) for record in eligible[:add_count]]
            row["checkpoint_escape_scan"] = {
                "performed": True,
                "active_energy_expectation": float(checkpoint_energy),
                "max_escape_score": float(max_score),
                "escape_threshold": float(escape_threshold),
                "inactive_candidate_count": int(len(candidate_scores)),
                "growth_triggered": bool(len(added_indices) > 0),
            }

            if added_indices:
                before_indices = tuple(active_indices)
                after_indices = tuple([*active_indices, *added_indices])
                target_norm = _real_quadratic(current_c, support.overlap, name=f"adaptation_events[{len(adaptation_events)}].target_norm")
                seed_new = np.zeros(len(after_indices), dtype=complex)
                old_positions = {int(index): pos for pos, index in enumerate(before_indices)}
                for pos, basis_index in enumerate(after_indices):
                    if int(basis_index) in old_positions:
                        seed_new[pos] = current_c[old_positions[int(basis_index)]]
                new_support = _support_for_indices(
                    overlap=overlap,
                    hamiltonian=hamiltonian,
                    active_indices=after_indices,
                    support_cutoff=support_cutoff,
                    settings=settings,
                )
                new_y, new_c, remap_projected_norm, remap_residual = _project_onto_support(
                    c_seed=seed_new,
                    support=new_support,
                    target_norm=target_norm,
                    name=f"adaptation_events[{len(adaptation_events)}].remap",
                )
                remap_after_norm = _real_quadratic(
                    new_c,
                    new_support.overlap,
                    name=f"adaptation_events[{len(adaptation_events)}].remap_after_norm",
                )
                remap_norm_errors.append(abs(remap_after_norm - target_norm))
                selected_for_addition = set(added_indices)
                event = {
                    "event_index": int(len(adaptation_events)),
                    "checkpoint_step_index": int(step_index),
                    "time": float(time_value),
                    "active_indices_before": [int(index) for index in before_indices],
                    "active_indices_after": [int(index) for index in after_indices],
                    "added_indices": [int(index) for index in added_indices],
                    "active_record_count_before": int(len(before_indices)),
                    "active_record_count_after": int(len(after_indices)),
                    "max_escape_score": float(max_score),
                    "escape_threshold": float(escape_threshold),
                    "max_add_per_checkpoint": int(max_add_per_checkpoint),
                    "max_active_records": int(max_active_records),
                    "remaining_capacity_before": int(remaining_capacity),
                    "candidate_score_summary": [
                        {
                            **record,
                            "selected_for_addition": bool(int(record["basis_index"]) in selected_for_addition),
                        }
                        for record in candidate_scores
                    ],
                    "remap": {
                        "target_qse_norm": float(target_norm),
                        "projected_qse_norm_before_rescale": float(remap_projected_norm),
                        "qse_norm_after_rescale": float(remap_after_norm),
                        "qse_norm_error_after_rescale": float(abs(remap_after_norm - target_norm)),
                        "basis_projection_residual_l2": float(remap_residual),
                    },
                    "new_retained_rank": int(len(new_support.retained_overlap_indices)),
                }
                adaptation_events.append(event)
                active_indices = after_indices
                support = new_support
                current_y = new_y
                active_support_history.append(
                    _support_diagnostics(
                        stage=f"after_adaptation_event_{event['event_index']}",
                        step_index=step_index,
                        time_value=time_value,
                        active_indices=active_indices,
                        support=support,
                        support_cutoff=support_cutoff,
                    )
                )
        else:
            row["checkpoint_escape_scan"] = {"performed": False}

        trajectory.append(row)
        if step_index != num_steps:
            current_y = _unitary_step(current_y, support.hamiltonian_orth, dt)

    row_norm_errors = [float(row["qse_norm_error_vs_unit"]) for row in trajectory]
    retained_norm_errors = [abs(float(row["retained_support_norm"]) - 1.0) for row in trajectory]
    energy_values = [float(row["qse_energy_expectation"]) for row in trajectory]

    artifact: dict[str, Any] = {
        "schema_version": ADAPTIVE_QSE_PROPAGATION_SCHEMA_VERSION,
        "pipeline": ADAPTIVE_QSE_PROPAGATION_PIPELINE,
        "generated_utc": _utc_now(),
        "propagation_kind": "adaptive_retained_qse_support_static_hamiltonian",
        "controller_usable": False,
        "feeds_controller_decisions": False,
        "exact_or_ed_reference_used": False,
        "raw_physical_statevectors_emitted": False,
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
            "initial_active_indices": [int(index) for index in active_support_history[0]["active_indices"]],
        },
        "command": None if command is None else (list(command) if not isinstance(command, str) else command),
        "config": {
            "t_final": float(t_final),
            "num_steps": int(num_steps),
            "dt": float(dt),
            "checkpoint_every_steps": int(checkpoint_every_steps),
            "support_cutoff": float(support_cutoff),
            "escape_threshold": float(escape_threshold),
            "max_add_per_checkpoint": int(max_add_per_checkpoint),
            "max_active_records": int(max_active_records),
            "score_floor": float(score_floor),
        },
        "full_qse_support": {
            "basis_size": int(basis_size),
            "retained_rank": int(len(full_support.retained_overlap_indices)),
            "discarded_rank": int(basis_size - len(full_support.retained_overlap_indices)),
            "retained_overlap_indices": [int(i) for i in full_support.retained_overlap_indices],
            "overlap_condition_estimate": full_support.overlap_condition_estimate,
            "support_cutoff": float(support_cutoff),
            "full_matrices_emitted": False,
            "lowdin_map_emitted": False,
            "overlap_hermitian_residual_max_abs": float(full_support.overlap_hermitian_residual_max_abs),
            "hamiltonian_hermitian_residual_max_abs": float(full_support.hamiltonian_hermitian_residual_max_abs),
        },
        "active_support_history": active_support_history,
        "initial_condition": {
            "root_index": int(initial_root_index),
            "qse_energy": float(selected_energy),
            "full_pool_qse_norm": float(initial_full_qse_norm),
            "active_restricted_qse_norm_before_retained_projection": float(initial_active_seed_norm),
            "projected_qse_norm_before_unit_rescale": float(initial_projected_norm),
            "qse_norm_after_retained_projection": 1.0,
            "retained_support_norm": float(np.vdot(current_y, current_y).real) if not trajectory else float(trajectory[0]["retained_support_norm"]),
            "basis_projection_residual_l2": float(initial_projection_residual),
            "normalization_policy": "active retained projection rescaled to unit QSE norm",
        },
        "trajectory": trajectory,
        "adaptation_events": adaptation_events,
        "metrics": {
            "trajectory_rows": int(len(trajectory)),
            "initial_active_record_count": int(active_support_history[0]["active_record_count"]),
            "final_active_record_count": int(len(active_indices)),
            "adaptation_event_count": int(len(adaptation_events)),
            "max_escape_score": float(max(max_escape_scores) if max_escape_scores else 0.0),
            "max_qse_norm_error": float(max([*row_norm_errors, *remap_norm_errors]) if row_norm_errors or remap_norm_errors else 0.0),
            "max_retained_support_norm_error": float(max(retained_norm_errors) if retained_norm_errors else 0.0),
            "initial_energy_expectation": float(energy_values[0]) if energy_values else None,
            "final_energy_expectation": float(energy_values[-1]) if energy_values else None,
            "max_energy_drift_abs": float(max(abs(value - energy_values[0]) for value in energy_values)) if energy_values else 0.0,
            "controller_usable": False,
            "feeds_controller_decisions": False,
            "exact_or_ed_reference_used": False,
            "raw_physical_statevectors_emitted": False,
            "uses_qiskit": False,
        },
        "visibility": {
            "controller_visible_payload_refs": [],
            "diagnostic_only_payload_refs": [
                "source",
                "full_qse_support",
                "active_support_history",
                "initial_condition",
                "trajectory",
                "adaptation_events",
                "metrics",
            ],
            "forbidden_to_controller_refs": [
                "trajectory",
                "trajectory.qse_basis_coefficients",
                "trajectory.retained_support_coefficients",
                "adaptation_events",
                "adaptation_events.candidate_score_summary",
                "initial_condition.qse_energy",
                "source.initial_root_energy",
                "active_support_history.overlap_eigenvalues_raw",
            ],
        },
        "warnings": [
            "adaptive_qse_propagation_is_diagnostic_only",
            "qse_basis_coefficients_are_for_offline_analysis_not_controller_decisions",
            "adaptive_candidate_scores_are_for_offline_analysis_not_controller_decisions",
            "no_ed_reference_or_raw_physical_vectors_are_emitted",
            "no_realtime_route_or_controller_integration_is_performed",
        ],
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Propagate QSE coefficients with diagnostic adaptive support growth.")
    parser.add_argument("--qse-manifest-json", required=True, help="Input qse_spectra_v1 manifest with matrices included.")
    parser.add_argument("--initial-active-indices", required=True, help="Comma-separated full-pool QSE basis indices, e.g. 0,1.")
    parser.add_argument("--initial-root-index", required=True, type=int, help="QSE Ritz root index used as c(0).")
    parser.add_argument("--t-final", required=True, type=float, help="Final propagation time.")
    parser.add_argument("--num-steps", required=True, type=int, help="Number of uniform unitary propagation steps.")
    parser.add_argument("--checkpoint-every-steps", required=True, type=int, help="Score/grow active support every N steps.")
    parser.add_argument(
        "--support-cutoff",
        required=True,
        type=float,
        help="Absolute Löwdin overlap eigenvalue cutoff for retained active support.",
    )
    parser.add_argument("--escape-threshold", required=True, type=float, help="Residual-coupling score threshold for growth.")
    parser.add_argument("--max-add-per-checkpoint", required=True, type=int, help="Maximum inactive records to add per checkpoint.")
    parser.add_argument("--max-active-records", required=True, type=int, help="Maximum active QSE records after growth.")
    parser.add_argument("--output-json", required=True, help="Output adaptive_qse_propagation_v1 JSON path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cli_args = list(sys.argv[1:] if argv is None else argv)
    command = ["python", "-m", "pipelines.excited_dynamics.adaptive_qse_propagation", *cli_args]
    run_adaptive_qse_propagation(
        AdaptiveQSEPropagationConfig(
            qse_manifest_json=Path(args.qse_manifest_json),
            initial_active_indices=args.initial_active_indices,
            initial_root_index=args.initial_root_index,
            t_final=args.t_final,
            num_steps=args.num_steps,
            checkpoint_every_steps=args.checkpoint_every_steps,
            support_cutoff=args.support_cutoff,
            escape_threshold=args.escape_threshold,
            max_add_per_checkpoint=args.max_add_per_checkpoint,
            max_active_records=args.max_active_records,
            output_json=Path(args.output_json),
        ),
        command=command,
    )
    return 0


__all__ = [
    "ADAPTIVE_QSE_PROPAGATION_PIPELINE",
    "ADAPTIVE_QSE_PROPAGATION_SCHEMA_VERSION",
    "DEFAULT_SCORE_FLOOR",
    "DEFAULT_SUPPORT_CUTOFF",
    "AdaptiveQSEPropagationConfig",
    "AdaptiveQSEPropagationError",
    "build_parser",
    "main",
    "run_adaptive_qse_propagation",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
