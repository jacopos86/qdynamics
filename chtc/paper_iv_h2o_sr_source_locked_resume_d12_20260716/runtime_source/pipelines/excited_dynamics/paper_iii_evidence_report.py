"""Paper-III local evidence/reporting harness.

This report-only sidecar aggregates existing P3 frozen-QSE, P4 adaptive-QSE,
and P6a promoted-McLachlan smoke artifacts. It validates their data-flow
boundaries, computes coefficient-only post-run reference comparisons from the
included QSE matrices, and emits a diagnostic ``paper_iii_evidence_report_v1``
JSON/markdown report.

The report is not a controller artifact. It does not modify source artifacts,
does not emit raw physical vectors, and does not import realtime/controller,
Qiskit, CHTC, Optuna, or exact-benchmark surfaces.
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
from pipelines.excited_dynamics.schemas import ValidationError, validate_qse_result_manifest


PAPER_III_EVIDENCE_REPORT_SCHEMA_VERSION = "paper_iii_evidence_report_v1"
PAPER_III_EVIDENCE_REPORT_PIPELINE = "paper_iii_evidence_report"
MODULE_NAME = "pipelines.excited_dynamics.paper_iii_evidence_report"
DEFAULT_SUPPORT_CUTOFF = 1.0e-12
DEFAULT_HERMITIAN_ABSOLUTE_TOLERANCE = 1.0e-10
DEFAULT_HERMITIAN_RELATIVE_TOLERANCE = 1.0e-8
DEFAULT_OVERLAP_NEGATIVE_ABSOLUTE_TOLERANCE = 1.0e-12
DEFAULT_OVERLAP_NEGATIVE_RELATIVE_TOLERANCE = 1.0e-9
DEFAULT_DISTANCE_NEGATIVE_ABSOLUTE_TOLERANCE = 1.0e-12
DEFAULT_TIME_ALIGNMENT_ABSOLUTE_TOLERANCE = 1.0e-9


class PaperIIIEvidenceReportError(ValueError):
    """Raised when P7a evidence report inputs fail closed."""


@dataclass(frozen=True)
class PaperIIIEvidenceReportConfig:
    frozen_run_manifest: Path
    adaptive_run_manifest: Path
    promoted_mclachlan_run_manifest: Path
    output_json: Path
    output_md: Path
    support_cutoff: float = DEFAULT_SUPPORT_CUTOFF


@dataclass(frozen=True)
class _ReferenceProblem:
    overlap: np.ndarray
    hamiltonian: np.ndarray
    x_map: np.ndarray
    hamiltonian_orth: np.ndarray
    y0: np.ndarray
    c0_projected: np.ndarray
    support_cutoff: float
    retained_rank: int
    discarded_rank: int
    overlap_condition_estimate: float | None
    initial_qse_norm: float
    initial_projected_qse_norm: float
    projection_residual_l2: float
    qse_basis_size: int
    initial_root_index: int


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PaperIIIEvidenceReportError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise PaperIIIEvidenceReportError(f"{name} must be a sequence")
    return value


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PaperIIIEvidenceReportError(f"{name} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise PaperIIIEvidenceReportError(f"{name} must be a finite number")
    return out


def _positive_float(value: Any, *, name: str) -> float:
    out = _finite_float(value, name=name)
    if out <= 0.0:
        raise PaperIIIEvidenceReportError(f"{name} must be positive")
    return out


def _strict_int(value: Any, *, name: str, min_value: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PaperIIIEvidenceReportError(f"{name} must be an integer")
    if min_value is not None and value < min_value:
        raise PaperIIIEvidenceReportError(f"{name} must be >= {min_value}")
    return int(value)


def _bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise PaperIIIEvidenceReportError(f"{name} must be boolean")
    return bool(value)


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
        raise PaperIIIEvidenceReportError(f"{name} row count {len(rows)} does not match basis_size {expected_size}")
    matrix = np.zeros((expected_size, expected_size), dtype=complex)
    for row_idx, row in enumerate(rows):
        cells = _sequence(row, name=f"{name}[{row_idx}]")
        if len(cells) != expected_size:
            raise PaperIIIEvidenceReportError(
                f"{name}[{row_idx}] column count {len(cells)} does not match basis_size {expected_size}"
            )
        for col_idx, cell in enumerate(cells):
            matrix[row_idx, col_idx] = _complex_from_json(cell, name=f"{name}[{row_idx}][{col_idx}]")
    return matrix


def _max_abs_entry(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0
    return float(np.max(np.abs(matrix)))


def _matrix_residual_max_abs(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0
    return float(np.max(np.abs(matrix - matrix.conj().T)))


def _hermitian_allowed(matrix: np.ndarray, *, abs_tol: float, rel_tol: float) -> float:
    return max(float(abs_tol), float(rel_tol) * max(1.0, _max_abs_entry(matrix)))


def _real_quadratic(vector: np.ndarray, matrix: np.ndarray, *, name: str) -> float:
    value = complex(vector.conj().T @ matrix @ vector)
    allowed_imag = 1.0e-10 * max(1.0, abs(value.real), abs(value.imag))
    if abs(value.imag) > allowed_imag:
        raise PaperIIIEvidenceReportError(f"{name} has non-negligible imaginary part {value.imag}")
    if not math.isfinite(float(value.real)):
        raise PaperIIIEvidenceReportError(f"{name} is not finite")
    return float(value.real)


def _coefficient_vector(eigenvalue: Mapping[str, Any], *, basis_size: int, name: str) -> np.ndarray:
    coeffs = _sequence(eigenvalue.get("basis_coefficients"), name=f"{name}.basis_coefficients")
    if len(coeffs) != basis_size:
        raise PaperIIIEvidenceReportError(f"{name}.basis_coefficients length {len(coeffs)} does not match basis_size {basis_size}")
    out = np.zeros(basis_size, dtype=complex)
    seen: set[int] = set()
    for coeff_idx, coeff in enumerate(coeffs):
        record = _mapping(coeff, name=f"{name}.basis_coefficients[{coeff_idx}]")
        basis_index = _strict_int(record.get("basis_index"), name=f"{name}.basis_coefficients[{coeff_idx}].basis_index", min_value=0)
        if basis_index >= basis_size:
            raise PaperIIIEvidenceReportError(f"{name}.basis_coefficients[{coeff_idx}].basis_index exceeds basis_size")
        if basis_index in seen:
            raise PaperIIIEvidenceReportError(f"{name}.basis_coefficients duplicate basis_index {basis_index}")
        seen.add(basis_index)
        out[basis_index] = complex(
            _finite_float(record.get("re"), name=f"{name}.basis_coefficients[{coeff_idx}].re"),
            _finite_float(record.get("im"), name=f"{name}.basis_coefficients[{coeff_idx}].im"),
        )
    if len(seen) != basis_size:
        raise PaperIIIEvidenceReportError(f"{name}.basis_coefficients must cover each basis index exactly once")
    return out


def _trajectory_coefficients_full(row: Mapping[str, Any], *, basis_size: int, name: str) -> np.ndarray:
    coeffs = _sequence(row.get("qse_basis_coefficients"), name=f"{name}.qse_basis_coefficients")
    if len(coeffs) != basis_size:
        raise PaperIIIEvidenceReportError(f"{name}.qse_basis_coefficients length {len(coeffs)} does not match basis_size {basis_size}")
    out = np.zeros(basis_size, dtype=complex)
    seen: set[int] = set()
    for coeff_idx, coeff in enumerate(coeffs):
        record = _mapping(coeff, name=f"{name}.qse_basis_coefficients[{coeff_idx}]")
        basis_index = _strict_int(record.get("basis_index"), name=f"{name}.qse_basis_coefficients[{coeff_idx}].basis_index", min_value=0)
        if basis_index >= basis_size:
            raise PaperIIIEvidenceReportError(f"{name}.qse_basis_coefficients[{coeff_idx}].basis_index exceeds basis_size")
        if basis_index in seen:
            raise PaperIIIEvidenceReportError(f"{name}.qse_basis_coefficients duplicate basis_index {basis_index}")
        seen.add(basis_index)
        out[basis_index] = complex(
            _finite_float(record.get("re"), name=f"{name}.qse_basis_coefficients[{coeff_idx}].re"),
            _finite_float(record.get("im"), name=f"{name}.qse_basis_coefficients[{coeff_idx}].im"),
        )
    if len(seen) != basis_size:
        raise PaperIIIEvidenceReportError(f"{name}.qse_basis_coefficients must cover each basis index exactly once")
    return out


def _trajectory_coefficients_active(row: Mapping[str, Any], *, basis_size: int, name: str) -> np.ndarray:
    coeffs = _sequence(row.get("qse_basis_coefficients"), name=f"{name}.qse_basis_coefficients")
    out = np.zeros(basis_size, dtype=complex)
    seen: set[int] = set()
    active_indices = row.get("active_indices")
    active_set = None
    if active_indices is not None:
        active_set = {int(_strict_int(value, name=f"{name}.active_indices[{idx}]", min_value=0)) for idx, value in enumerate(_sequence(active_indices, name=f"{name}.active_indices"))}
    for coeff_idx, coeff in enumerate(coeffs):
        record = _mapping(coeff, name=f"{name}.qse_basis_coefficients[{coeff_idx}]")
        basis_index = _strict_int(record.get("basis_index"), name=f"{name}.qse_basis_coefficients[{coeff_idx}].basis_index", min_value=0)
        if basis_index >= basis_size:
            raise PaperIIIEvidenceReportError(f"{name}.qse_basis_coefficients[{coeff_idx}].basis_index exceeds basis_size")
        if basis_index in seen:
            raise PaperIIIEvidenceReportError(f"{name}.qse_basis_coefficients duplicate basis_index {basis_index}")
        if active_set is not None and basis_index not in active_set:
            raise PaperIIIEvidenceReportError(f"{name}.qse_basis_coefficients basis_index {basis_index} is not listed in active_indices")
        seen.add(basis_index)
        out[basis_index] = complex(
            _finite_float(record.get("re"), name=f"{name}.qse_basis_coefficients[{coeff_idx}].re"),
            _finite_float(record.get("im"), name=f"{name}.qse_basis_coefficients[{coeff_idx}].im"),
        )
    return out


def _display_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _resolve_existing_path(value: Any, *, manifest_path: Path, field: str) -> Path:
    if not isinstance(value, str) or value.strip() == "":
        raise PaperIIIEvidenceReportError(f"{field} must be a non-empty path string")
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, manifest_path.parent / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise PaperIIIEvidenceReportError(f"{field} does not exist; tried {rendered}")


def _load_json_file(path: Path, *, name: str) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise PaperIIIEvidenceReportError(f"{name} must contain a JSON object")
    return payload


def _find_true_key(value: Any, *, key: str, prefix: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, raw_value in value.items():
            path = f"{prefix}.{raw_key}" if prefix else str(raw_key)
            if raw_key == key and raw_value is True:
                hits.append(path)
            hits.extend(_find_true_key(raw_value, key=key, prefix=path))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            path = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            hits.extend(_find_true_key(item, key=key, prefix=path))
    return hits


def _assert_no_true_key(value: Any, *, key: str, context: str) -> None:
    hits = _find_true_key(value, key=key)
    if hits:
        raise PaperIIIEvidenceReportError(f"{context} marks {key}=true at {hits}")


def _expect_false(value: Mapping[str, Any], key: str, *, context: str, required: bool = True) -> None:
    if key not in value:
        if required:
            raise PaperIIIEvidenceReportError(f"{context}.{key} is required")
        return
    if _bool(value.get(key), name=f"{context}.{key}") is not False:
        raise PaperIIIEvidenceReportError(f"{context}.{key} must be false")


def _expect_true(value: Mapping[str, Any], key: str, *, context: str, required: bool = True) -> None:
    if key not in value:
        if required:
            raise PaperIIIEvidenceReportError(f"{context}.{key} is required")
        return
    if _bool(value.get(key), name=f"{context}.{key}") is not True:
        raise PaperIIIEvidenceReportError(f"{context}.{key} must be true")


def _load_qse_manifest(path: Path) -> tuple[dict[str, Any], str, Any]:
    payload = _load_json_file(path, name="qse_manifest")
    try:
        summary = validate_qse_result_manifest(payload)
    except ValidationError as exc:
        raise PaperIIIEvidenceReportError(str(exc)) from exc
    return payload, sha256_file(path), summary


def _lowdin_reference_problem(
    *,
    qse_payload: Mapping[str, Any],
    initial_root_index: int,
    support_cutoff: float,
) -> _ReferenceProblem:
    summary = validate_qse_result_manifest(qse_payload)
    settings = _mapping(qse_payload.get("settings", {}), name="qse_manifest.settings")
    matrices = _mapping(qse_payload.get("matrices"), name="qse_manifest.matrices")
    if matrices.get("included") is not True:
        raise PaperIIIEvidenceReportError("qse_spectra_v1 manifest must include matrices for P7a reference comparison")

    basis_size = int(summary.basis_size)
    overlap_raw = _matrix_from_json(matrices.get("overlap"), name="qse_manifest.matrices.overlap", expected_size=basis_size)
    hamiltonian_raw = _matrix_from_json(matrices.get("hamiltonian"), name="qse_manifest.matrices.hamiltonian", expected_size=basis_size)
    if overlap_raw.ndim != 2 or hamiltonian_raw.ndim != 2:
        raise PaperIIIEvidenceReportError("QSE matrices must be 2D arrays")
    if overlap_raw.shape[0] != overlap_raw.shape[1] or hamiltonian_raw.shape != overlap_raw.shape:
        raise PaperIIIEvidenceReportError("QSE Hamiltonian and overlap matrix shapes must match and be square")

    hermitian_abs_tol = _settings_float(settings, "hermitian_absolute_tolerance", DEFAULT_HERMITIAN_ABSOLUTE_TOLERANCE)
    hermitian_rel_tol = _settings_float(settings, "hermitian_relative_tolerance", DEFAULT_HERMITIAN_RELATIVE_TOLERANCE)
    negative_abs_tol = _settings_float(settings, "overlap_negative_absolute_tolerance", DEFAULT_OVERLAP_NEGATIVE_ABSOLUTE_TOLERANCE)
    negative_rel_tol = _settings_float(settings, "overlap_negative_relative_tolerance", DEFAULT_OVERLAP_NEGATIVE_RELATIVE_TOLERANCE)

    overlap_residual = _matrix_residual_max_abs(overlap_raw)
    hamiltonian_residual = _matrix_residual_max_abs(hamiltonian_raw)
    if overlap_residual > _hermitian_allowed(overlap_raw, abs_tol=hermitian_abs_tol, rel_tol=hermitian_rel_tol):
        raise PaperIIIEvidenceReportError("QSE overlap matrix is non-Hermitian beyond configured tolerance")
    if hamiltonian_residual > _hermitian_allowed(hamiltonian_raw, abs_tol=hermitian_abs_tol, rel_tol=hermitian_rel_tol):
        raise PaperIIIEvidenceReportError("QSE Hamiltonian matrix is non-Hermitian beyond configured tolerance")

    overlap = 0.5 * (overlap_raw + overlap_raw.conj().T)
    hamiltonian = 0.5 * (hamiltonian_raw + hamiltonian_raw.conj().T)
    s_raw, u = np.linalg.eigh(overlap)
    s_raw = np.asarray(s_raw, dtype=float)
    if not np.all(np.isfinite(s_raw)):
        raise PaperIIIEvidenceReportError("QSE overlap eigenvalues must be finite")
    max_abs_s = float(np.max(np.abs(s_raw))) if s_raw.size else 0.0
    negative_tol = max(float(negative_abs_tol), float(negative_rel_tol) * max_abs_s)
    min_raw = float(np.min(s_raw)) if s_raw.size else 0.0
    if min_raw < -negative_tol:
        raise PaperIIIEvidenceReportError(f"QSE overlap matrix has negative eigenvalue {min_raw} below tolerance {-negative_tol}")
    s_clamped = np.where(s_raw < 0.0, 0.0, s_raw)
    retained_mask = s_clamped >= support_cutoff
    retained_indices = tuple(int(idx) for idx in np.nonzero(retained_mask)[0])
    if not retained_indices:
        raise PaperIIIEvidenceReportError("QSE reference retained rank is zero")
    s_retained = s_clamped[list(retained_indices)]
    u_retained = u[:, list(retained_indices)]
    x_map = u_retained @ np.diag(1.0 / np.sqrt(s_retained))
    h_orth = x_map.conj().T @ hamiltonian @ x_map
    h_orth = 0.5 * (h_orth + h_orth.conj().T)
    condition = float(np.max(s_retained) / np.min(s_retained)) if s_retained.size else None

    eigenvalues = _sequence(qse_payload.get("eigenvalues"), name="qse_manifest.eigenvalues")
    if initial_root_index >= len(eigenvalues):
        raise PaperIIIEvidenceReportError(f"initial_root_index {initial_root_index} out of range for QSE eigenvalues")
    selected = _mapping(eigenvalues[initial_root_index], name=f"qse_manifest.eigenvalues[{initial_root_index}]")
    state_index = _strict_int(selected.get("state_index"), name=f"qse_manifest.eigenvalues[{initial_root_index}].state_index", min_value=0)
    if state_index != initial_root_index:
        raise PaperIIIEvidenceReportError("selected QSE root state_index does not match initial_root_index")
    c0 = _coefficient_vector(selected, basis_size=basis_size, name=f"qse_manifest.eigenvalues[{initial_root_index}]")
    y0 = x_map.conj().T @ overlap @ c0
    c0_projected = x_map @ y0
    initial_qse_norm = _real_quadratic(c0, overlap, name="reference.initial_qse_norm")
    projected_norm = _real_quadratic(c0_projected, overlap, name="reference.initial_projected_qse_norm")
    if projected_norm <= 0.0:
        raise PaperIIIEvidenceReportError("QSE reference initial projected norm must be positive")

    return _ReferenceProblem(
        overlap=np.asarray(overlap, dtype=complex),
        hamiltonian=np.asarray(hamiltonian, dtype=complex),
        x_map=np.asarray(x_map, dtype=complex),
        hamiltonian_orth=np.asarray(h_orth, dtype=complex),
        y0=np.asarray(y0, dtype=complex),
        c0_projected=np.asarray(c0_projected, dtype=complex),
        support_cutoff=float(support_cutoff),
        retained_rank=int(len(retained_indices)),
        discarded_rank=int(basis_size - len(retained_indices)),
        overlap_condition_estimate=condition,
        initial_qse_norm=float(initial_qse_norm),
        initial_projected_qse_norm=float(projected_norm),
        projection_residual_l2=float(np.linalg.norm(c0 - c0_projected)),
        qse_basis_size=int(basis_size),
        initial_root_index=int(initial_root_index),
    )


def _reference_coefficients_at(problem: _ReferenceProblem, *, time_value: float) -> np.ndarray:
    evals, evecs = np.linalg.eigh(problem.hamiltonian_orth)
    phase = np.exp(-1.0j * float(time_value) * np.asarray(evals, dtype=float))
    y = evecs @ (phase * (evecs.conj().T @ problem.y0))
    return np.asarray(problem.x_map @ y, dtype=complex)


def _phase_distance(candidate: np.ndarray, reference: np.ndarray, overlap: np.ndarray, *, name: str) -> dict[str, Any]:
    c = np.asarray(candidate, dtype=complex).reshape(-1)
    r = np.asarray(reference, dtype=complex).reshape(-1)
    if c.shape != r.shape:
        raise PaperIIIEvidenceReportError(f"{name} candidate/reference coefficient shapes do not match")
    inner = complex(c.conj().T @ overlap @ r)
    inner_abs = float(abs(inner))
    # Align the reference by the phase that minimizes the S-overlap distance.
    # With inner=<c|S|r>, the minimizing factor is exp(-i arg(inner)).
    phase_angle = math.atan2(inner.imag, inner.real) if inner_abs > 0.0 else 0.0
    phase_factor = np.exp(-1.0j * phase_angle) if inner_abs > 0.0 else 1.0 + 0.0j
    delta = c - phase_factor * r
    raw_distance_squared = complex(delta.conj().T @ overlap @ delta)
    allowed_imag = 1.0e-10 * max(1.0, abs(raw_distance_squared.real), abs(raw_distance_squared.imag))
    if abs(raw_distance_squared.imag) > allowed_imag:
        raise PaperIIIEvidenceReportError(f"{name} overlap distance has non-negligible imaginary part")
    real_value = float(raw_distance_squared.real)
    clamped = False
    if real_value < 0.0:
        if real_value < -DEFAULT_DISTANCE_NEGATIVE_ABSOLUTE_TOLERANCE:
            raise PaperIIIEvidenceReportError(f"{name} overlap distance squared is negative beyond tolerance")
        real_value = 0.0
        clamped = True
    candidate_norm = _real_quadratic(c, overlap, name=f"{name}.candidate_norm")
    reference_norm = _real_quadratic(r, overlap, name=f"{name}.reference_norm")
    return {
        "overlap_phase_distance": float(math.sqrt(real_value)),
        "overlap_phase_distance_squared": float(real_value),
        "candidate_qse_norm": float(candidate_norm),
        "reference_qse_norm": float(reference_norm),
        "phase_alignment_inner_abs": float(inner_abs),
        "phase_alignment_angle_rad": float(-phase_angle),
        "negative_distance_squared_clamped": bool(clamped),
        "feeds_controller_decisions": False,
    }


def _validate_report_only_artifact(payload: Mapping[str, Any], *, context: str, expected_schema: str, expected_pipeline: str) -> None:
    if payload.get("schema_version") != expected_schema:
        raise PaperIIIEvidenceReportError(f"{context}.schema_version must be {expected_schema!r}")
    if payload.get("pipeline") != expected_pipeline:
        raise PaperIIIEvidenceReportError(f"{context}.pipeline must be {expected_pipeline!r}")
    _expect_false(payload, "controller_usable", context=context)
    _expect_false(payload, "feeds_controller_decisions", context=context)
    _expect_false(payload, "exact_or_ed_reference_used", context=context, required=False)
    _expect_false(payload, "uses_qiskit", context=context, required=False)
    _expect_false(payload, "raw_physical_statevectors_emitted", context=context, required=False)
    boundary = _mapping(payload.get("controller_boundary"), name=f"{context}.controller_boundary")
    _expect_false(boundary, "controller_usable", context=f"{context}.controller_boundary")
    _expect_false(boundary, "feeds_controller_decisions", context=f"{context}.controller_boundary")
    _expect_false(boundary, "decision_path_allowed", context=f"{context}.controller_boundary")
    _assert_no_true_key(payload, key="feeds_controller_decisions", context=context)


def _load_p3_or_p4_source(
    *,
    run_manifest_path: Path,
    expected_schema: str,
    expected_pipeline: str,
    context: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, Path, Path]:
    run_manifest = _load_json_file(run_manifest_path, name=f"{context}_run_manifest")
    _assert_no_true_key(run_manifest, key="feeds_controller_decisions", context=f"{context}_run_manifest")
    artifacts = _mapping(run_manifest.get("artifacts"), name=f"{context}_run_manifest.artifacts")
    output_path = _resolve_existing_path(artifacts.get("output_json"), manifest_path=run_manifest_path, field=f"{context}.artifacts.output_json")
    qse_path = _resolve_existing_path(
        artifacts.get("input_qse_manifest_json"),
        manifest_path=run_manifest_path,
        field=f"{context}.artifacts.input_qse_manifest_json",
    )
    output = _load_json_file(output_path, name=f"{context}_output")
    _validate_report_only_artifact(output, context=f"{context}_output", expected_schema=expected_schema, expected_pipeline=expected_pipeline)
    qse_payload, qse_sha256, _summary = _load_qse_manifest(qse_path)
    source = _mapping(output.get("source"), name=f"{context}_output.source")
    if source.get("source_qse_sha256") != qse_sha256:
        raise PaperIIIEvidenceReportError(f"{context} source_qse_sha256 does not match input QSE manifest")
    return run_manifest, output, qse_payload, output_path, qse_path, run_manifest_path


def _validate_p6_decision_row(row: Mapping[str, Any], *, context: str) -> None:
    for key in ("uses_reference_for_decision", "uses_future_exact_forecast_for_decision", "append_attempted", "prune_attempted", "structure_edit_attempted"):
        _expect_false(row, key, context=context)
    _expect_true(row, "strict_measurement_oracle_certified", context=context)
    if row.get("controller_exact_input_mode") != "off":
        raise PaperIIIEvidenceReportError(f"{context}.controller_exact_input_mode must be 'off'")
    if row.get("diagnostic_exact_reference_mode") != "off":
        raise PaperIIIEvidenceReportError(f"{context}.diagnostic_exact_reference_mode must be 'off'")
    if row.get("decision_backend") == "exact":
        raise PaperIIIEvidenceReportError(f"{context}.decision_backend must not be exact")
    if row.get("decision_data_flow") != "ideal_observable_estimator":
        raise PaperIIIEvidenceReportError(f"{context}.decision_data_flow must be ideal_observable_estimator")


def _load_p6_source(
    run_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    run_manifest = _load_json_file(run_manifest_path, name="promoted_mclachlan_run_manifest")
    _assert_no_true_key(run_manifest, key="feeds_controller_decisions", context="promoted_mclachlan_run_manifest")
    output_path = _resolve_existing_path(run_manifest.get("output_json"), manifest_path=run_manifest_path, field="promoted_mclachlan.output_json")
    output = _load_json_file(output_path, name="promoted_mclachlan_output")
    _assert_no_true_key(output, key="feeds_controller_decisions", context="promoted_mclachlan_output")
    if output.get("schema_version") != "qse_promoted_mclachlan_run_v1":
        raise PaperIIIEvidenceReportError("promoted_mclachlan_output.schema_version must be 'qse_promoted_mclachlan_run_v1'")
    if output.get("pipeline") != "qse_promoted_mclachlan_smoke":
        raise PaperIIIEvidenceReportError("promoted_mclachlan_output.pipeline must be 'qse_promoted_mclachlan_smoke'")
    _expect_false(output, "uses_qiskit", context="promoted_mclachlan_output")

    source = _mapping(output.get("source"), name="promoted_mclachlan_output.source")
    if source.get("loader_boundary") != "runtime_payload_only":
        raise PaperIIIEvidenceReportError("promoted_mclachlan_output.source.loader_boundary must be runtime_payload_only")
    if list(source.get("controller_visible_payload_refs_used", [])) != ["runtime_payload"]:
        raise PaperIIIEvidenceReportError("promoted_mclachlan_output.source.controller_visible_payload_refs_used must be ['runtime_payload']")

    runtime_contract = _mapping(output.get("runtime_contract"), name="promoted_mclachlan_output.runtime_contract")
    if runtime_contract.get("loader_boundary") != "runtime_payload_only":
        raise PaperIIIEvidenceReportError("promoted_mclachlan_output.runtime_contract.loader_boundary must be runtime_payload_only")
    _expect_true(runtime_contract, "structure_locked", context="promoted_mclachlan_output.runtime_contract")
    _expect_false(runtime_contract, "can_structural_edit", context="promoted_mclachlan_output.runtime_contract")
    _expect_true(runtime_contract, "reference_energy_absent", context="promoted_mclachlan_output.runtime_contract")
    if runtime_contract.get("input_runtime_contract_status") != "validated":
        raise PaperIIIEvidenceReportError("promoted_mclachlan_output.runtime_contract.input_runtime_contract_status must be validated")

    boundary = _mapping(output.get("controller_boundary"), name="promoted_mclachlan_output.controller_boundary")
    if boundary.get("source_payload_loaded") != "runtime_payload_only":
        raise PaperIIIEvidenceReportError("promoted_mclachlan_output.controller_boundary.source_payload_loaded must be runtime_payload_only")
    _expect_false(boundary, "top_level_diagnostic_metadata_feeds_controller_decisions", context="promoted_mclachlan_output.controller_boundary")
    _expect_false(boundary, "structural_editing_allowed", context="promoted_mclachlan_output.controller_boundary")
    _expect_false(boundary, "append_allowed", context="promoted_mclachlan_output.controller_boundary")
    _expect_false(boundary, "prune_allowed", context="promoted_mclachlan_output.controller_boundary")
    _expect_true(boundary, "qse_diagnostics_forbidden_to_controller", context="promoted_mclachlan_output.controller_boundary")

    forbidden_audit = _mapping(output.get("forbidden_marker_audit"), name="promoted_mclachlan_output.forbidden_marker_audit")
    _expect_true(forbidden_audit, "passed", context="promoted_mclachlan_output.forbidden_marker_audit")
    if int(forbidden_audit.get("hit_count", 0)) != 0 or list(forbidden_audit.get("hits", [])) != []:
        raise PaperIIIEvidenceReportError("promoted_mclachlan_output forbidden marker audit must have zero hits")

    strict_audit = _mapping(output.get("strict_decision_contract_audit"), name="promoted_mclachlan_output.strict_decision_contract_audit")
    _expect_true(strict_audit, "passed", context="promoted_mclachlan_output.strict_decision_contract_audit")
    _expect_false(strict_audit, "uses_reference_for_decision", context="promoted_mclachlan_output.strict_decision_contract_audit")
    _expect_false(strict_audit, "uses_future_exact_forecast_for_decision", context="promoted_mclachlan_output.strict_decision_contract_audit")
    if int(strict_audit.get("violation_count", 0)) != 0:
        raise PaperIIIEvidenceReportError("promoted_mclachlan_output strict decision contract has violations")

    summary = _mapping(output.get("summary"), name="promoted_mclachlan_output.summary")
    _expect_false(summary, "paper_iii_science_benchmark", context="promoted_mclachlan_output.summary")
    for key in ("append_count", "prune_count", "structure_edit_count"):
        if int(summary.get(key, 0)) != 0:
            raise PaperIIIEvidenceReportError(f"promoted_mclachlan_output.summary.{key} must be zero")
    if summary.get("strict_decision_contract_passed") is not True:
        raise PaperIIIEvidenceReportError("promoted_mclachlan_output.summary.strict_decision_contract_passed must be true")

    decision_data_flow = _mapping(output.get("decision_data_flow"), name="promoted_mclachlan_output.decision_data_flow")
    if decision_data_flow.get("decision_data_flow") != "ideal_observable_estimator":
        raise PaperIIIEvidenceReportError("promoted_mclachlan_output.decision_data_flow must be ideal_observable_estimator")
    if decision_data_flow.get("controller_exact_input_mode") != "off":
        raise PaperIIIEvidenceReportError("promoted_mclachlan_output decision controller_exact_input_mode must be off")
    if decision_data_flow.get("diagnostic_exact_reference_mode") != "off":
        raise PaperIIIEvidenceReportError("promoted_mclachlan_output diagnostic_exact_reference_mode must be off")
    _expect_false(decision_data_flow, "uses_reference_for_decision", context="promoted_mclachlan_output.decision_data_flow")
    _expect_false(decision_data_flow, "uses_future_exact_forecast_for_decision", context="promoted_mclachlan_output.decision_data_flow")

    for idx, row in enumerate(_sequence(output.get("trajectory"), name="promoted_mclachlan_output.trajectory")):
        _validate_p6_decision_row(_mapping(row, name=f"promoted_mclachlan_output.trajectory[{idx}]"), context=f"promoted_mclachlan_output.trajectory[{idx}]")
    for idx, row in enumerate(_sequence(output.get("mclachlan_steps"), name="promoted_mclachlan_output.mclachlan_steps")):
        _validate_p6_decision_row(_mapping(row, name=f"promoted_mclachlan_output.mclachlan_steps[{idx}]"), context=f"promoted_mclachlan_output.mclachlan_steps[{idx}]")

    if run_manifest.get("paper_iii_science_benchmark") is True:
        raise PaperIIIEvidenceReportError("promoted_mclachlan_run_manifest.paper_iii_science_benchmark must be false")
    if run_manifest.get("forbidden_marker_hit_count") not in (None, 0):
        raise PaperIIIEvidenceReportError("promoted_mclachlan_run_manifest.forbidden_marker_hit_count must be zero")
    if run_manifest.get("loader_boundary") not in (None, "runtime_payload_only"):
        raise PaperIIIEvidenceReportError("promoted_mclachlan_run_manifest.loader_boundary must be runtime_payload_only")
    if run_manifest.get("loader_structure_locked") is False:
        raise PaperIIIEvidenceReportError("promoted_mclachlan_run_manifest.loader_structure_locked must be true")
    if run_manifest.get("loader_can_structural_edit") is True:
        raise PaperIIIEvidenceReportError("promoted_mclachlan_run_manifest.loader_can_structural_edit must be false")

    return run_manifest, output, output_path, run_manifest_path


def _compare_trajectory_to_reference(
    *,
    label: str,
    artifact: Mapping[str, Any],
    qse_payload: Mapping[str, Any],
    active_coefficients: bool,
    support_cutoff: float,
) -> dict[str, Any]:
    source = _mapping(artifact.get("source"), name=f"{label}.source")
    initial_root_index = _strict_int(source.get("initial_root_index"), name=f"{label}.source.initial_root_index", min_value=0)
    problem = _lowdin_reference_problem(qse_payload=qse_payload, initial_root_index=initial_root_index, support_cutoff=support_cutoff)
    trajectory = _sequence(artifact.get("trajectory"), name=f"{label}.trajectory")
    if len(trajectory) == 0:
        raise PaperIIIEvidenceReportError(f"{label}.trajectory must be non-empty")

    rows: list[dict[str, Any]] = []
    max_distance = 0.0
    max_norm_delta = 0.0
    for idx, raw_row in enumerate(trajectory):
        row = _mapping(raw_row, name=f"{label}.trajectory[{idx}]")
        step_index = _strict_int(row.get("step_index"), name=f"{label}.trajectory[{idx}].step_index", min_value=0)
        time_value = _finite_float(row.get("time"), name=f"{label}.trajectory[{idx}].time")
        if idx > 0 and time_value + DEFAULT_TIME_ALIGNMENT_ABSOLUTE_TOLERANCE < float(rows[-1]["time"]):
            raise PaperIIIEvidenceReportError(f"{label}.trajectory times must be nondecreasing")
        candidate = (
            _trajectory_coefficients_active(row, basis_size=problem.qse_basis_size, name=f"{label}.trajectory[{idx}]")
            if active_coefficients
            else _trajectory_coefficients_full(row, basis_size=problem.qse_basis_size, name=f"{label}.trajectory[{idx}]")
        )
        reference = _reference_coefficients_at(problem, time_value=time_value)
        distance = _phase_distance(candidate, reference, problem.overlap, name=f"{label}.trajectory[{idx}]")
        max_distance = max(max_distance, float(distance["overlap_phase_distance"]))
        max_norm_delta = max(max_norm_delta, abs(float(distance["candidate_qse_norm"]) - float(distance["reference_qse_norm"])))
        rows.append(
            {
                "step_index": int(step_index),
                "time": float(time_value),
                **distance,
            }
        )

    final_row = rows[-1]
    comparison = {
        "comparison_kind": "post_run_full_qse_matrix_reference",
        "post_run_only": True,
        "feeds_controller_decisions": False,
        "coefficient_payload_emitted": False,
        "qse_basis_size": int(problem.qse_basis_size),
        "initial_root_index": int(problem.initial_root_index),
        "reference_support": {
            "retained_rank": int(problem.retained_rank),
            "discarded_rank": int(problem.discarded_rank),
            "support_cutoff": float(problem.support_cutoff),
            "overlap_condition_estimate": problem.overlap_condition_estimate,
            "initial_qse_norm": float(problem.initial_qse_norm),
            "initial_projected_qse_norm": float(problem.initial_projected_qse_norm),
            "projection_residual_l2": float(problem.projection_residual_l2),
        },
        "summary": {
            "row_count": int(len(rows)),
            "max_overlap_phase_distance": float(max_distance),
            "final_overlap_phase_distance": float(final_row["overlap_phase_distance"]),
            "max_candidate_reference_qse_norm_delta": float(max_norm_delta),
            "distance_negative_clamp_count": int(sum(1 for row in rows if row["negative_distance_squared_clamped"])),
            "feeds_controller_decisions": False,
        },
        "rows": rows,
    }
    return comparison


def _source_summary(
    *,
    label: str,
    run_manifest: Mapping[str, Any],
    artifact: Mapping[str, Any],
    run_manifest_path: Path,
    output_path: Path,
    qse_path: Path | None = None,
) -> dict[str, Any]:
    metrics = _mapping(artifact.get("metrics", artifact.get("summary", {})), name=f"{label}.metrics_or_summary")
    summary: dict[str, Any] = {
        "label": label,
        "run_manifest_path": _display_path(run_manifest_path),
        "output_json": _display_path(output_path),
        "output_sha256": sha256_file(output_path),
        "schema_version": artifact.get("schema_version"),
        "pipeline": artifact.get("pipeline"),
        "feeds_controller_decisions": False,
        "uses_qiskit": bool(artifact.get("uses_qiskit", False)),
        "metrics": {},
    }
    if qse_path is not None:
        summary["input_qse_manifest_json"] = _display_path(qse_path)
        summary["input_qse_manifest_sha256"] = sha256_file(qse_path)
    for key in (
        "trajectory_rows",
        "retained_rank",
        "max_qse_norm_error",
        "max_energy_drift_abs",
        "initial_active_record_count",
        "final_active_record_count",
        "adaptation_event_count",
        "max_escape_score",
        "trajectory_row_count",
        "step_count",
        "max_rhs_residual_ratio",
        "max_state_norm_error",
        "paper_iii_science_benchmark",
        "strict_decision_contract_passed",
    ):
        if key in metrics:
            summary["metrics"][key] = metrics[key]
        elif key in run_manifest:
            summary["metrics"][key] = run_manifest[key]
    return summary


def _promoted_summary(run_manifest: Mapping[str, Any], artifact: Mapping[str, Any], *, run_manifest_path: Path, output_path: Path) -> dict[str, Any]:
    runtime_contract = _mapping(artifact.get("runtime_contract"), name="promoted.runtime_contract")
    strict_audit = _mapping(artifact.get("strict_decision_contract_audit"), name="promoted.strict_decision_contract_audit")
    forbidden_audit = _mapping(artifact.get("forbidden_marker_audit"), name="promoted.forbidden_marker_audit")
    summary = _mapping(artifact.get("summary"), name="promoted.summary")
    return {
        "label": "promoted_mclachlan",
        "evidence_classification": "contract_plumbing",
        "paper_iii_science_benchmark": False,
        "run_manifest_path": _display_path(run_manifest_path),
        "output_json": _display_path(output_path),
        "output_sha256": sha256_file(output_path),
        "schema_version": artifact.get("schema_version"),
        "pipeline": artifact.get("pipeline"),
        "source_payload_loaded": "runtime_payload_only",
        "feeds_controller_decisions": False,
        "runtime_payload_used_by_p6a_decisions": True,
        "structure_locked": bool(runtime_contract.get("structure_locked")),
        "can_structural_edit": bool(runtime_contract.get("can_structural_edit")),
        "reference_energy_absent": bool(runtime_contract.get("reference_energy_absent")),
        "strict_decision_contract_passed": bool(strict_audit.get("passed")),
        "forbidden_marker_hit_count": int(forbidden_audit.get("hit_count", 0)),
        "append_prune_structure_edits": {
            "append_count": int(summary.get("append_count", run_manifest.get("append_attempted", 0) or 0)),
            "prune_count": int(summary.get("prune_count", run_manifest.get("prune_attempted", 0) or 0)),
            "structure_edit_count": int(summary.get("structure_edit_count", run_manifest.get("structure_edit_attempted", 0) or 0)),
        },
        "metrics": {
            "trajectory_row_count": int(summary.get("trajectory_row_count", run_manifest.get("trajectory_row_count", 0) or 0)),
            "mclachlan_step_count": int(summary.get("step_count", run_manifest.get("mclachlan_step_count", 0) or 0)),
            "max_rhs_residual_ratio": float(summary.get("max_rhs_residual_ratio", run_manifest.get("max_rhs_residual_ratio", 0.0) or 0.0)),
            "max_state_norm_error": float(summary.get("max_state_norm_error", run_manifest.get("max_state_norm_error", 0.0) or 0.0)),
        },
    }


def _build_markdown(report: Mapping[str, Any]) -> str:
    metrics = _mapping(report.get("metrics"), name="report.metrics")
    frozen_summary = _mapping(_mapping(report.get("reference_comparisons"), name="report.reference_comparisons").get("frozen_qse"), name="frozen_qse_comparison")["summary"]
    adaptive_summary = _mapping(_mapping(report.get("reference_comparisons"), name="report.reference_comparisons").get("adaptive_qse"), name="adaptive_qse_comparison")["summary"]
    promoted = _mapping(_mapping(report.get("source_artifacts"), name="report.source_artifacts").get("promoted_mclachlan"), name="promoted_mclachlan_summary")
    lines = [
        "# Paper III P7a Local Evidence Report",
        "",
        f"Generated UTC: `{report.get('generated_utc')}`",
        "",
        "## Report-only boundary",
        "",
        f"- controller_usable: `{str(report.get('controller_usable')).lower()}`",
        f"- feeds_controller_decisions: `{str(report.get('feeds_controller_decisions')).lower()}`",
        f"- reference_comparisons_feed_controller_decisions: `{str(report.get('reference_comparisons_feed_controller_decisions')).lower()}`",
        f"- raw_physical_vectors_emitted: `{str(report.get('raw_physical_vectors_emitted')).lower()}`",
        "",
        "## Coefficient reference comparisons",
        "",
        f"- Frozen QSE rows: `{frozen_summary['row_count']}`, max distance: `{frozen_summary['max_overlap_phase_distance']}`",
        f"- Adaptive QSE rows: `{adaptive_summary['row_count']}`, max distance: `{adaptive_summary['max_overlap_phase_distance']}`",
        f"- Adaptive final active records: `{metrics.get('adaptive_final_active_record_count')}`, adaptation events: `{metrics.get('adaptive_adaptation_event_count')}`",
        "",
        "## Promoted McLachlan smoke classification",
        "",
        f"- Evidence classification: `{promoted.get('evidence_classification')}`",
        f"- Paper III science benchmark: `{str(promoted.get('paper_iii_science_benchmark')).lower()}`",
        f"- Loader boundary: `{promoted.get('source_payload_loaded')}`",
        f"- Strict decision contract passed: `{str(promoted.get('strict_decision_contract_passed')).lower()}`",
        f"- Forbidden marker hit count: `{promoted.get('forbidden_marker_hit_count')}`",
        "",
        "## Source artifacts",
        "",
    ]
    for label, source in _mapping(report.get("source_artifacts"), name="report.source_artifacts").items():
        source_map = _mapping(source, name=f"source_artifacts.{label}")
        lines.append(f"- `{label}`: `{source_map.get('output_json')}`")
    lines.extend(
        [
            "",
            "## Scope guardrails",
            "",
        ]
    )
    for key, value in _mapping(report.get("scope_guardrails"), name="report.scope_guardrails").items():
        lines.append(f"- {key}: `{str(value).lower() if isinstance(value, bool) else value}`")
    return "\n".join(lines) + "\n"


def _build_command_log(*, command: Sequence[str] | str | None, report: Mapping[str, Any]) -> str:
    command_rendered = " ".join(command) if isinstance(command, Sequence) and not isinstance(command, str) else str(command or "not provided")
    metrics = _mapping(report.get("metrics"), name="report.metrics")
    return (
        "# P7a Local Evidence Report Command Log\n\n"
        f"Generated UTC: `{report.get('generated_utc')}`\n\n"
        "## Report command\n\n"
        f"`{command_rendered}`\n\n"
        "Exit code: `0`\n\n"
        "## Result\n\n"
        "Generated report-only JSON and markdown artifacts.\n\n"
        "## Key metrics\n\n"
        f"- frozen_reference_max_overlap_phase_distance: `{metrics.get('frozen_reference_max_overlap_phase_distance')}`\n"
        f"- adaptive_reference_max_overlap_phase_distance: `{metrics.get('adaptive_reference_max_overlap_phase_distance')}`\n"
        f"- promoted_mclachlan_evidence_classification: `{metrics.get('promoted_mclachlan_evidence_classification')}`\n"
    )


def _build_run_manifest(
    *,
    config: PaperIIIEvidenceReportConfig,
    report: Mapping[str, Any],
    command: Sequence[str] | str | None,
    command_log_path: Path,
) -> dict[str, Any]:
    command_rendered = " ".join(command) if isinstance(command, Sequence) and not isinstance(command, str) else str(command or "not provided")
    metrics = _mapping(report.get("metrics"), name="report.metrics")
    return {
        "schema_version": "agent_run_manifest_v1",
        "slice": "paper_iii_p7a_local_evidence_report",
        "generated_utc": report.get("generated_utc"),
        "diagnostic_only": True,
        "pipeline": PAPER_III_EVIDENCE_REPORT_PIPELINE,
        "commands": {
            "smoke": {
                "command": command_rendered,
                "exit_code": 0,
                "result": "paper_iii_evidence_report generated",
            }
        },
        "artifacts": {
            "output_json": _display_path(config.output_json),
            "output_md": _display_path(config.output_md),
            "command_log_md": _display_path(command_log_path),
            "run_manifest_json": _display_path(config.output_json.parent / "run_manifest.json"),
            "frozen_run_manifest_json": _display_path(config.frozen_run_manifest),
            "adaptive_run_manifest_json": _display_path(config.adaptive_run_manifest),
            "promoted_mclachlan_run_manifest_json": _display_path(config.promoted_mclachlan_run_manifest),
        },
        "controller_boundary": {
            "controller_usable": False,
            "feeds_controller_decisions": False,
            "reference_comparisons_feed_controller_decisions": False,
            "post_run_diagnostic_only": True,
            "decision_path_allowed": False,
            "controller_exact_input_mode": "off",
            "diagnostic_exact_reference_mode": "benchmark_post_run_only",
        },
        "output_summary": {
            "schema_version": report.get("schema_version"),
            "pipeline": report.get("pipeline"),
            "source_artifact_count": metrics.get("source_artifact_count"),
            "frozen_reference_max_overlap_phase_distance": metrics.get("frozen_reference_max_overlap_phase_distance"),
            "adaptive_reference_max_overlap_phase_distance": metrics.get("adaptive_reference_max_overlap_phase_distance"),
            "adaptive_final_active_record_count": metrics.get("adaptive_final_active_record_count"),
            "adaptive_adaptation_event_count": metrics.get("adaptive_adaptation_event_count"),
            "promoted_mclachlan_evidence_classification": metrics.get("promoted_mclachlan_evidence_classification"),
            "paper_iii_science_benchmark": False,
            "controller_usable": False,
            "feeds_controller_decisions": False,
            "reference_comparisons_feed_controller_decisions": False,
            "raw_physical_vectors_emitted": False,
        },
        "scope_guardrails": dict(_mapping(report.get("scope_guardrails"), name="report.scope_guardrails")),
    }


def _assert_report_has_no_raw_vector_payload(report: Mapping[str, Any]) -> None:
    serialized = json.dumps(report, sort_keys=True, allow_nan=False)
    forbidden_markers = (
        "amplitudes_qn_to_q0",
        "raw_physical_state",
        "basis_matrix_vectors",
        "exact_target_trajectories",
        "exact_step_forecast",
        "state_at(",
    )
    hits = [marker for marker in forbidden_markers if marker in serialized]
    if hits:
        raise PaperIIIEvidenceReportError(f"P7a report would emit forbidden raw/reference payload markers: {hits}")


def run_paper_iii_evidence_report(
    config: PaperIIIEvidenceReportConfig,
    *,
    command: Sequence[str] | str | None = None,
) -> dict[str, Any]:
    support_cutoff = _positive_float(config.support_cutoff, name="support_cutoff")
    frozen_run_manifest_path = Path(config.frozen_run_manifest).resolve()
    adaptive_run_manifest_path = Path(config.adaptive_run_manifest).resolve()
    promoted_run_manifest_path = Path(config.promoted_mclachlan_run_manifest).resolve()
    output_json = Path(config.output_json)
    output_md = Path(config.output_md)

    frozen_run_manifest, frozen_output, frozen_qse, frozen_output_path, frozen_qse_path, _ = _load_p3_or_p4_source(
        run_manifest_path=frozen_run_manifest_path,
        expected_schema="frozen_qse_propagation_v1",
        expected_pipeline="frozen_qse_propagation",
        context="frozen_qse",
    )
    adaptive_run_manifest, adaptive_output, adaptive_qse, adaptive_output_path, adaptive_qse_path, _ = _load_p3_or_p4_source(
        run_manifest_path=adaptive_run_manifest_path,
        expected_schema="adaptive_qse_propagation_v1",
        expected_pipeline="adaptive_qse_propagation",
        context="adaptive_qse",
    )
    promoted_run_manifest, promoted_output, promoted_output_path, _ = _load_p6_source(promoted_run_manifest_path)

    frozen_comparison = _compare_trajectory_to_reference(
        label="frozen_qse",
        artifact=frozen_output,
        qse_payload=frozen_qse,
        active_coefficients=False,
        support_cutoff=support_cutoff,
    )
    adaptive_comparison = _compare_trajectory_to_reference(
        label="adaptive_qse",
        artifact=adaptive_output,
        qse_payload=adaptive_qse,
        active_coefficients=True,
        support_cutoff=support_cutoff,
    )
    _assert_no_true_key(frozen_comparison, key="feeds_controller_decisions", context="frozen_reference_comparison")
    _assert_no_true_key(adaptive_comparison, key="feeds_controller_decisions", context="adaptive_reference_comparison")

    adaptive_metrics = _mapping(adaptive_output.get("metrics"), name="adaptive_qse.metrics")
    report: dict[str, Any] = {
        "schema_version": PAPER_III_EVIDENCE_REPORT_SCHEMA_VERSION,
        "pipeline": PAPER_III_EVIDENCE_REPORT_PIPELINE,
        "generated_utc": _utc_now(),
        "report_kind": "paper_iii_p7a_local_evidence_report",
        "controller_usable": False,
        "feeds_controller_decisions": False,
        "reference_comparisons_feed_controller_decisions": False,
        "exact_or_ed_reference_values_feed_controller_decisions": False,
        "raw_physical_vectors_emitted": False,
        "uses_qiskit": False,
        "source_artifacts_modified": False,
        "comparison_method": {
            "coefficient_only": True,
            "post_run_only": True,
            "full_qse_matrix_reference_from_included_matrices": True,
            "physical_vector_payloads_emitted": False,
            "distance_definition": "S-overlap phase-aligned coefficient distance; reference coefficients are not emitted",
            "phase_alignment_convention": "candidate - exp(-i arg(candidate^dagger S reference)) reference",
            "negative_distance_squared_clamp_abs_tolerance": DEFAULT_DISTANCE_NEGATIVE_ABSOLUTE_TOLERANCE,
            "feeds_controller_decisions": False,
        },
        "source_artifacts": {
            "frozen_qse": _source_summary(
                label="frozen_qse",
                run_manifest=frozen_run_manifest,
                artifact=frozen_output,
                run_manifest_path=frozen_run_manifest_path,
                output_path=frozen_output_path,
                qse_path=frozen_qse_path,
            ),
            "adaptive_qse": _source_summary(
                label="adaptive_qse",
                run_manifest=adaptive_run_manifest,
                artifact=adaptive_output,
                run_manifest_path=adaptive_run_manifest_path,
                output_path=adaptive_output_path,
                qse_path=adaptive_qse_path,
            ),
            "promoted_mclachlan": _promoted_summary(
                promoted_run_manifest,
                promoted_output,
                run_manifest_path=promoted_run_manifest_path,
                output_path=promoted_output_path,
            ),
        },
        "reference_comparisons": {
            "post_run_only": True,
            "feeds_controller_decisions": False,
            "frozen_qse": frozen_comparison,
            "adaptive_qse": {
                **adaptive_comparison,
                "active_support_summary": {
                    "initial_active_record_count": int(adaptive_metrics.get("initial_active_record_count", 0)),
                    "final_active_record_count": int(adaptive_metrics.get("final_active_record_count", 0)),
                    "adaptation_event_count": int(adaptive_metrics.get("adaptation_event_count", 0)),
                    "active_support_history_count": int(len(_sequence(adaptive_output.get("active_support_history", []), name="adaptive_qse.active_support_history"))),
                    "adaptation_events_emitted_in_source_only": True,
                },
            },
        },
        "metrics": {
            "source_artifact_count": 3,
            "frozen_reference_row_count": frozen_comparison["summary"]["row_count"],
            "frozen_reference_max_overlap_phase_distance": frozen_comparison["summary"]["max_overlap_phase_distance"],
            "frozen_reference_final_overlap_phase_distance": frozen_comparison["summary"]["final_overlap_phase_distance"],
            "adaptive_reference_row_count": adaptive_comparison["summary"]["row_count"],
            "adaptive_reference_max_overlap_phase_distance": adaptive_comparison["summary"]["max_overlap_phase_distance"],
            "adaptive_reference_final_overlap_phase_distance": adaptive_comparison["summary"]["final_overlap_phase_distance"],
            "adaptive_final_active_record_count": int(adaptive_metrics.get("final_active_record_count", 0)),
            "adaptive_adaptation_event_count": int(adaptive_metrics.get("adaptation_event_count", 0)),
            "promoted_mclachlan_evidence_classification": "contract_plumbing",
            "promoted_mclachlan_paper_iii_science_benchmark": False,
            "controller_usable": False,
            "feeds_controller_decisions": False,
            "reference_comparisons_feed_controller_decisions": False,
            "raw_physical_vectors_emitted": False,
        },
        "controller_boundary": {
            "controller_usable": False,
            "feeds_controller_decisions": False,
            "reference_comparisons_feed_controller_decisions": False,
            "decision_path_allowed": False,
            "post_run_diagnostic_only": True,
            "controller_exact_input_mode": "off",
            "diagnostic_exact_reference_mode": "benchmark_post_run_only",
        },
        "visibility": {
            "controller_visible_payload_refs": [],
            "diagnostic_only_payload_refs": [
                "source_artifacts",
                "reference_comparisons",
                "metrics",
            ],
            "forbidden_to_controller_refs": [
                "reference_comparisons",
                "source_artifacts.frozen_qse",
                "source_artifacts.adaptive_qse",
                "source_artifacts.promoted_mclachlan",
            ],
        },
        "scope_guardrails": {
            "report_only": True,
            "source_artifacts_modified": False,
            "reference_comparisons_post_run_only": True,
            "reference_comparisons_feed_controller_decisions": False,
            "chtc_used": False,
            "optuna_used": False,
            "hh_science_run_executed": False,
            "realtime_or_controller_route_changed": False,
            "adapt_static_defaults_changed": False,
            "mclachlan_realtime_defaults_changed": False,
            "raw_physical_vectors_emitted": False,
        },
        "warnings": [
            "p7a_report_is_diagnostic_only_and_not_controller_usable",
            "reference_comparisons_are_post_run_only",
            "source_p3_p4_p6a_artifacts_are_read_only_inputs",
            "promoted_mclachlan_row_is_contract_plumbing_not_science_benchmark",
        ],
    }
    _assert_no_true_key(report, key="feeds_controller_decisions", context="paper_iii_evidence_report")
    _assert_report_has_no_raw_vector_payload(report)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    output_md.write_text(_build_markdown(report), encoding="utf-8")

    command_log_path = output_json.parent / "command_log.md"
    command_log_path.write_text(_build_command_log(command=command, report=report), encoding="utf-8")
    run_manifest = _build_run_manifest(config=config, report=report, command=command, command_log_path=command_log_path)
    _assert_report_has_no_raw_vector_payload(run_manifest)
    (output_json.parent / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a local Paper-III P7a evidence report from existing smoke artifacts.")
    parser.add_argument("--frozen-run-manifest", required=True, help="P3 frozen-QSE run_manifest.json")
    parser.add_argument("--adaptive-run-manifest", required=True, help="P4 adaptive-QSE run_manifest.json")
    parser.add_argument("--promoted-mclachlan-run-manifest", required=True, help="P6a promoted-McLachlan run_manifest.json")
    parser.add_argument("--output-json", required=True, help="Output paper_iii_evidence_report_v1 JSON path")
    parser.add_argument("--output-md", required=True, help="Output markdown summary path")
    parser.add_argument(
        "--support-cutoff",
        type=float,
        default=DEFAULT_SUPPORT_CUTOFF,
        help="Absolute QSE overlap eigenvalue cutoff for post-run reference comparisons.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cli_args = list(sys.argv[1:] if argv is None else argv)
    command = ["python", "-m", MODULE_NAME, *cli_args]
    run_paper_iii_evidence_report(
        PaperIIIEvidenceReportConfig(
            frozen_run_manifest=Path(args.frozen_run_manifest),
            adaptive_run_manifest=Path(args.adaptive_run_manifest),
            promoted_mclachlan_run_manifest=Path(args.promoted_mclachlan_run_manifest),
            output_json=Path(args.output_json),
            output_md=Path(args.output_md),
            support_cutoff=args.support_cutoff,
        ),
        command=command,
    )
    return 0


__all__ = [
    "MODULE_NAME",
    "PAPER_III_EVIDENCE_REPORT_PIPELINE",
    "PAPER_III_EVIDENCE_REPORT_SCHEMA_VERSION",
    "PaperIIIEvidenceReportConfig",
    "PaperIIIEvidenceReportError",
    "build_parser",
    "main",
    "run_paper_iii_evidence_report",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
