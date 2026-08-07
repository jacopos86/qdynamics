"""Offline QSE Ritz-root refit into a fixed Pauli-rotation ansatz artifact.

This module intentionally stays outside realtime/controller routes.  It rebuilds a
QSE Ritz root from a ``qse_spectra_v1`` manifest, fits that diagnostic target with
a small Pauli-rotation ansatz seeded by the selected QSE operator basis, and emits
a ``qse_root_refit_v1`` sidecar whose ansatz payload is only potentially
promotable after separate runtime-contract validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.qse_spectra.core import (
    QSEBasisElement,
    QSEBasisVectorPolicy,
    QSEPruningConfig,
    _prepare_basis_vectors,
    computational_basis_state,
    normalize_statevector,
)
from pipelines.qse_spectra.io import (
    load_operator_basis_json,
    load_polynomial_json,
    load_state_json,
    polynomial_from_serialized_terms,
    statevector_from_manifest,
    write_manifest_json,
)
from pipelines.scaffold.handoff_state_bundle import build_statevector_manifest
from src.quantum.ansatz_parameterization import (
    build_parameter_layout,
    deserialize_layout,
    iter_runtime_rotation_terms,
    project_runtime_theta_block_mean,
    serialize_layout,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import compile_polynomial_action, energy_via_one_apply
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


QSE_ROOT_REFIT_SCHEMA_VERSION = "qse_root_refit_v1"
QSE_ROOT_REFIT_PIPELINE = "qse_root_refit"
QSE_RESULT_SCHEMA_VERSION = "qse_spectra_v1"


class QSERootRefitError(ValueError):
    """Raised when the QSE root-refit input or fit is invalid."""


@dataclass(frozen=True)
class QSERootRefitConfig:
    qse_result_json: Path
    state_index: int
    output_json: Path
    allow_ground_state: bool = False
    prepared_state_json: Path | None = None
    prepared_state_json_key: str = "auto"
    prepared_state_bitstring: str | None = None
    hamiltonian_json: Path | None = None
    max_infidelity: float = 1.0e-8
    max_energy_error: float | None = None
    maxiter: int = 200
    amplitude_cutoff: float = 1.0e-12


@dataclass(frozen=True)
class QSERootTarget:
    state: np.ndarray
    norm_before_normalization: float
    coefficients: np.ndarray
    selected_eigenvalue: Mapping[str, Any]
    basis_vector_policy: QSEBasisVectorPolicy
    qse_energy: float
    nearest_energy_gap: float | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class PauliRotationRefitResult:
    terms: tuple[AnsatzTerm, ...]
    layout: Any
    theta_runtime: np.ndarray
    theta_logical: np.ndarray
    fitted_state: np.ndarray
    fidelity: float
    infidelity: float
    optimizer_summary: dict[str, Any]


@dataclass(frozen=True)
class PreparedStateResolution:
    state: np.ndarray
    provenance: dict[str, Any]
    override_used: bool


@dataclass(frozen=True)
class HamiltonianResolution:
    polynomial: PauliPolynomial | None
    provenance: dict[str, Any] | None
    explicit_override_used: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QSERootRefitError(f"{name} must be a mapping.")
    return value


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise QSERootRefitError(f"{name} must be a sequence.")
    return value


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QSERootRefitError(f"{name} must be a finite number.")
    out = float(value)
    if not math.isfinite(out):
        raise QSERootRefitError(f"{name} must be a finite number.")
    return out


def _strict_int(value: Any, *, name: str, min_value: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise QSERootRefitError(f"{name} must be an integer.")
    if min_value is not None and int(value) < int(min_value):
        raise QSERootRefitError(f"{name} must be >= {min_value}.")
    return int(value)


def _bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise QSERootRefitError(f"{name} must be boolean.")
    return bool(value)


def _complex_from_record(record: Mapping[str, Any], *, name: str) -> complex:
    return complex(
        _finite_float(record.get("re"), name=f"{name}.re"),
        _finite_float(record.get("im"), name=f"{name}.im"),
    )


def _complex_to_json(value: complex) -> dict[str, float]:
    value_c = complex(value)
    if not math.isfinite(float(value_c.real)) or not math.isfinite(float(value_c.imag)):
        raise QSERootRefitError("Cannot serialize non-finite complex value.")
    return {"re": float(value_c.real), "im": float(value_c.imag)}


def _resolve_path(raw_path: str | Path, *, relative_to: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    return Path(relative_to).parent / candidate


def _validate_threshold(value: float | None, *, name: str, required: bool = False) -> float | None:
    if value is None:
        if required:
            raise QSERootRefitError(f"{name} is required.")
        return None
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise QSERootRefitError(f"{name} must be finite and non-negative.")
    return out


def validate_qse_root_refit_config(config: QSERootRefitConfig) -> None:
    _validate_threshold(config.max_infidelity, name="max_infidelity", required=True)
    _validate_threshold(config.max_energy_error, name="max_energy_error")
    if int(config.maxiter) < 0:
        raise QSERootRefitError("maxiter must be non-negative.")
    if not math.isfinite(float(config.amplitude_cutoff)) or float(config.amplitude_cutoff) < 0.0:
        raise QSERootRefitError("amplitude_cutoff must be finite and non-negative.")
    if config.prepared_state_json is not None and config.prepared_state_bitstring is not None:
        raise QSERootRefitError("Use only one of prepared_state_json or prepared_state_bitstring.")
    if str(config.prepared_state_json_key) not in {"auto", "initial_state", "ansatz_input_state"}:
        raise QSERootRefitError("prepared_state_json_key must be auto, initial_state, or ansatz_input_state.")


def _validate_qse_manifest(payload: Mapping[str, Any], *, state_index: int, allow_ground_state: bool) -> tuple[int, int, Mapping[str, Any]]:
    payload = _mapping(payload, name="qse_result")
    if payload.get("schema_version") != QSE_RESULT_SCHEMA_VERSION:
        raise QSERootRefitError(f"schema_version must be {QSE_RESULT_SCHEMA_VERSION!r}.")
    if payload.get("pipeline") != "qse_spectra":
        raise QSERootRefitError("pipeline must be 'qse_spectra'.")
    if payload.get("backend") != "ideal_statevector":
        raise QSERootRefitError("backend must be 'ideal_statevector'.")
    if _bool(payload.get("uses_qiskit"), name="uses_qiskit"):
        raise QSERootRefitError("qse_spectra_v1 input must not use qiskit.")

    diagnostics = _mapping(payload.get("diagnostics"), name="diagnostics")
    nq = _strict_int(diagnostics.get("num_qubits"), name="diagnostics.num_qubits", min_value=1)
    hilbert_dim = _strict_int(diagnostics.get("hilbert_dim"), name="diagnostics.hilbert_dim", min_value=1)
    if hilbert_dim != (1 << int(nq)):
        raise QSERootRefitError(f"diagnostics.hilbert_dim={hilbert_dim} does not match num_qubits={nq}.")
    basis_size = _strict_int(diagnostics.get("basis_size"), name="diagnostics.basis_size", min_value=1)
    retained_rank = _strict_int(diagnostics.get("retained_rank"), name="diagnostics.retained_rank", min_value=0)
    if retained_rank > basis_size:
        raise QSERootRefitError("diagnostics.retained_rank cannot exceed diagnostics.basis_size.")

    operator_basis = _sequence(payload.get("operator_basis"), name="operator_basis")
    if len(operator_basis) != int(basis_size):
        raise QSERootRefitError(f"operator_basis length {len(operator_basis)} does not match basis_size {basis_size}.")
    for idx, raw in enumerate(operator_basis):
        record = _mapping(raw, name=f"operator_basis[{idx}]")
        basis_index = _strict_int(record.get("basis_index"), name=f"operator_basis[{idx}].basis_index", min_value=0)
        if basis_index != idx:
            raise QSERootRefitError(f"operator_basis[{idx}].basis_index must equal {idx}.")

    eigenvalues = _sequence(payload.get("eigenvalues"), name="eigenvalues")
    if len(eigenvalues) == 0:
        raise QSERootRefitError("eigenvalues must be non-empty.")
    for idx, raw in enumerate(eigenvalues):
        record = _mapping(raw, name=f"eigenvalues[{idx}]")
        actual_index = _strict_int(record.get("state_index"), name=f"eigenvalues[{idx}].state_index", min_value=0)
        if actual_index != idx:
            raise QSERootRefitError(f"eigenvalues[{idx}].state_index must equal {idx}.")
        _finite_float(record.get("energy"), name=f"eigenvalues[{idx}].energy")
        if record.get("generalized_residual_norm") is not None:
            _finite_float(record.get("generalized_residual_norm"), name=f"eigenvalues[{idx}].generalized_residual_norm")
        _basis_coefficients(record, basis_size=basis_size, eigenvalue_index=idx)

    if int(state_index) == 0 and not bool(allow_ground_state):
        raise QSERootRefitError("state_index=0 is the QSE ground Ritz state; pass --allow-ground-state to refit it.")
    if int(state_index) < 0 or int(state_index) >= len(eigenvalues):
        raise QSERootRefitError(f"state_index {state_index} out of range for {len(eigenvalues)} eigenvalues.")
    selected = _mapping(eigenvalues[int(state_index)], name=f"eigenvalues[{state_index}]")
    actual_selected = _strict_int(selected.get("state_index"), name=f"eigenvalues[{state_index}].state_index", min_value=0)
    if actual_selected != int(state_index):
        raise QSERootRefitError(f"selected eigenvalue state_index {actual_selected} does not equal requested {state_index}.")
    return int(nq), int(basis_size), selected


def _basis_coefficients(
    eigenvalue: Mapping[str, Any],
    *,
    basis_size: int,
    eigenvalue_index: int,
) -> np.ndarray:
    raw_coeffs = _sequence(
        eigenvalue.get("basis_coefficients"),
        name=f"eigenvalues[{eigenvalue_index}].basis_coefficients",
    )
    if len(raw_coeffs) != int(basis_size):
        raise QSERootRefitError(
            f"eigenvalues[{eigenvalue_index}].basis_coefficients length {len(raw_coeffs)} does not match basis_size {basis_size}."
        )
    coeffs = np.zeros(int(basis_size), dtype=complex)
    seen: set[int] = set()
    for coeff_idx, raw in enumerate(raw_coeffs):
        record = _mapping(raw, name=f"eigenvalues[{eigenvalue_index}].basis_coefficients[{coeff_idx}]")
        basis_index = _strict_int(
            record.get("basis_index"),
            name=f"eigenvalues[{eigenvalue_index}].basis_coefficients[{coeff_idx}].basis_index",
            min_value=0,
        )
        if basis_index >= int(basis_size):
            raise QSERootRefitError(f"basis coefficient index {basis_index} exceeds basis_size {basis_size}.")
        if basis_index in seen:
            raise QSERootRefitError(f"basis coefficient index {basis_index} appears more than once.")
        seen.add(basis_index)
        coeffs[int(basis_index)] = _complex_from_record(
            record,
            name=f"eigenvalues[{eigenvalue_index}].basis_coefficients[{coeff_idx}]",
        )
    if len(seen) != int(basis_size):
        raise QSERootRefitError(f"eigenvalues[{eigenvalue_index}].basis_coefficients must cover each basis index exactly once.")
    return coeffs


def _basis_vector_policy_from_manifest(payload: Mapping[str, Any]) -> QSEBasisVectorPolicy:
    for container_name in ("settings", "diagnostics"):
        container = payload.get(container_name)
        if isinstance(container, Mapping) and isinstance(container.get("basis_vector_policy"), Mapping):
            raw = container["basis_vector_policy"]
            return QSEBasisVectorPolicy(
                reference_projection=str(raw.get("reference_projection", "none")),
                basis_vector_normalization=str(raw.get("basis_vector_normalization", "normalized")),
                sector_projection=str(raw.get("sector_projection", "identity")),
                sector_label=raw.get("sector_label"),
            )
    raise QSERootRefitError("qse_spectra_v1 manifest is missing settings/diagnostics.basis_vector_policy.")


def _qse_pruning_config_from_manifest(payload: Mapping[str, Any]) -> QSEPruningConfig:
    settings = payload.get("settings")
    cfg = QSEPruningConfig()
    if not isinstance(settings, Mapping):
        return cfg
    values = {}
    for name in QSEPruningConfig.__dataclass_fields__:
        if name in settings:
            values[name] = _finite_float(settings[name], name=f"settings.{name}")
        else:
            values[name] = getattr(cfg, name)
    return QSEPruningConfig(**values)


def _resolve_prepared_state(
    payload: Mapping[str, Any],
    *,
    qse_result_json: Path,
    nq: int,
    prepared_state_json: Path | None,
    prepared_state_json_key: str,
    prepared_state_bitstring: str | None,
) -> PreparedStateResolution:
    if prepared_state_json is not None:
        state, provenance = load_state_json(
            Path(prepared_state_json),
            expected_nq=int(nq),
            state_key=str(prepared_state_json_key),
        )
        provenance = {**provenance, "override_source": "--prepared-state-json"}
        return PreparedStateResolution(state=state, provenance=provenance, override_used=True)
    if prepared_state_bitstring is not None:
        state = computational_basis_state(int(nq), str(prepared_state_bitstring))
        provenance = {
            "source_schema": "override_computational_basis_state",
            "override_source": "--prepared-state-bitstring",
            "state_bitstring": str(prepared_state_bitstring),
            "nq_total": int(nq),
        }
        return PreparedStateResolution(state=state, provenance=provenance, override_used=True)

    input_block = payload.get("input")
    if isinstance(input_block, Mapping):
        state_block = input_block.get("state")
    else:
        state_block = None
    if isinstance(state_block, Mapping):
        source_schema = str(state_block.get("source_schema", ""))
        if source_schema == "computational_basis_state":
            bitstring = state_block.get("state_bitstring")
            if bitstring is None:
                raise QSERootRefitError("input.state computational_basis_state is missing state_bitstring.")
            raw_nq = state_block.get("nq_total", nq)
            state_nq = _strict_int(raw_nq, name="input.state.nq_total", min_value=0)
            if int(state_nq) != int(nq):
                raise QSERootRefitError(f"input.state.nq_total={state_nq} does not match diagnostics.num_qubits={nq}.")
            state = computational_basis_state(int(nq), str(bitstring))
            return PreparedStateResolution(state=state, provenance=dict(state_block), override_used=False)
        if "path" in state_block:
            state_path = _resolve_path(str(state_block["path"]), relative_to=Path(qse_result_json))
            state_key = str(state_block.get("selected_state_key", "auto"))
            if state_key not in {"auto", "initial_state", "ansatz_input_state", "top_level"}:
                state_key = "auto"
            if state_key == "top_level":
                state_key = "auto"
            state, provenance = load_state_json(state_path, expected_nq=int(nq), state_key=state_key)
            provenance = {**dict(state_block), **provenance, "resolved_path": str(state_path)}
            return PreparedStateResolution(state=state, provenance=provenance, override_used=False)

    raise QSERootRefitError(
        "prepared state provenance is insufficient; pass --prepared-state-json or --prepared-state-bitstring."
    )


def _resolve_hamiltonian(
    payload: Mapping[str, Any],
    *,
    qse_result_json: Path,
    hamiltonian_json: Path | None,
    require_energy: bool,
) -> HamiltonianResolution:
    path: Path | None = None
    explicit = False
    if hamiltonian_json is not None:
        path = Path(hamiltonian_json)
        explicit = True
    else:
        input_block = payload.get("input")
        if isinstance(input_block, Mapping):
            ham_block = input_block.get("hamiltonian")
            if isinstance(ham_block, Mapping) and "path" in ham_block:
                path = _resolve_path(str(ham_block["path"]), relative_to=Path(qse_result_json))

    if path is None:
        if require_energy:
            raise QSERootRefitError("--max-energy-error requires --hamiltonian-json or loadable input.hamiltonian.path.")
        return HamiltonianResolution(polynomial=None, provenance=None, explicit_override_used=False)
    if not path.exists():
        if require_energy or explicit:
            raise QSERootRefitError(f"Hamiltonian JSON not found: {path}.")
        return HamiltonianResolution(
            polynomial=None,
            provenance={"path": str(path), "available": False},
            explicit_override_used=False,
        )
    polynomial, provenance = load_polynomial_json(path)
    provenance = {**provenance, "resolved_path": str(path), "available": True}
    return HamiltonianResolution(polynomial=polynomial, provenance=provenance, explicit_override_used=explicit)


def _phase_fix_state(state: np.ndarray, *, cutoff: float) -> tuple[np.ndarray, dict[str, Any]]:
    psi = np.asarray(state, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(psi))
    if norm <= 0.0:
        raise QSERootRefitError("Reconstructed QSE target has zero norm.")
    psi = psi / norm
    anchor_index = None
    anchor_value = 0.0 + 0.0j
    for idx, amp in enumerate(psi):
        if abs(amp) > float(cutoff):
            anchor_index = int(idx)
            anchor_value = complex(amp)
            phase = anchor_value / abs(anchor_value)
            psi = psi / phase
            break
    if anchor_index is None:
        raise QSERootRefitError("Reconstructed QSE target has no amplitudes above phase cutoff.")
    psi[np.abs(psi) <= float(cutoff) * 0.1] = 0.0
    phase_info = {
        "phase_convention": "first_nonzero_amplitude_real_positive",
        "anchor_index": int(anchor_index),
        "anchor_amplitude_before_phase_fix": _complex_to_json(anchor_value),
    }
    return np.asarray(psi, dtype=complex), phase_info


def _nearest_energy_gap(payload: Mapping[str, Any], *, state_index: int, energy: float) -> float | None:
    gaps: list[float] = []
    for raw in _sequence(payload.get("eigenvalues"), name="eigenvalues"):
        record = _mapping(raw, name="eigenvalue")
        other_index = _strict_int(record.get("state_index"), name="eigenvalue.state_index", min_value=0)
        if other_index == int(state_index):
            continue
        other_energy = _finite_float(record.get("energy"), name=f"eigenvalues[{other_index}].energy")
        gaps.append(abs(float(other_energy) - float(energy)))
    return min(gaps) if gaps else None


def reconstruct_qse_root_target(
    qse_payload: Mapping[str, Any],
    *,
    qse_result_json: Path,
    state_index: int,
    allow_ground_state: bool = False,
    prepared_state_json: Path | None = None,
    prepared_state_json_key: str = "auto",
    prepared_state_bitstring: str | None = None,
    amplitude_cutoff: float = 1.0e-12,
) -> tuple[QSERootTarget, PreparedStateResolution, tuple[QSEBasisElement, ...], int]:
    nq, basis_size, selected = _validate_qse_manifest(
        qse_payload,
        state_index=int(state_index),
        allow_ground_state=bool(allow_ground_state),
    )
    prepared = _resolve_prepared_state(
        qse_payload,
        qse_result_json=Path(qse_result_json),
        nq=int(nq),
        prepared_state_json=prepared_state_json,
        prepared_state_json_key=str(prepared_state_json_key),
        prepared_state_bitstring=prepared_state_bitstring,
    )
    psi_ref, _, inferred_nq = normalize_statevector(prepared.state)
    if int(inferred_nq) != int(nq):
        raise QSERootRefitError(f"Prepared state inferred nq={inferred_nq}; QSE manifest has nq={nq}.")

    basis, _basis_provenance = load_operator_basis_json(Path(qse_result_json), nq=int(nq))
    if len(basis) != int(basis_size):
        raise QSERootRefitError(f"Loaded operator_basis length {len(basis)} does not match diagnostics.basis_size={basis_size}.")

    policy = _basis_vector_policy_from_manifest(qse_payload)
    cfg = _qse_pruning_config_from_manifest(qse_payload)
    prepared_vectors = _prepare_basis_vectors(
        basis,
        psi_ref,
        nq=int(nq),
        config=cfg,
        policy=policy,
        pauli_action_cache={},
    )
    if len(prepared_vectors.matrix_vectors) != int(basis_size):
        raise QSERootRefitError("Internal QSE basis-vector reconstruction produced the wrong basis size.")

    coeffs = _basis_coefficients(selected, basis_size=int(basis_size), eigenvalue_index=int(state_index))
    target = np.zeros_like(psi_ref, dtype=complex)
    for idx, coeff in enumerate(coeffs):
        target += complex(coeff) * np.asarray(prepared_vectors.matrix_vectors[idx], dtype=complex).reshape(-1)
    target_norm = float(np.linalg.norm(target))
    target_fixed, phase_info = _phase_fix_state(target, cutoff=float(amplitude_cutoff))

    energy = _finite_float(selected.get("energy"), name=f"eigenvalues[{state_index}].energy")
    gap = _nearest_energy_gap(qse_payload, state_index=int(state_index), energy=float(energy))
    warnings: list[str] = []
    if gap is not None and gap <= 1.0e-8:
        warnings.append("selected_qse_root_nearly_degenerate")
    if int(state_index) == 0:
        warnings.append("ground_qse_root_refit_allowed_by_explicit_flag")
    phase_info["target_norm_before_normalization"] = float(target_norm)

    return (
        QSERootTarget(
            state=target_fixed,
            norm_before_normalization=float(target_norm),
            coefficients=coeffs,
            selected_eigenvalue=selected,
            basis_vector_policy=policy,
            qse_energy=float(energy),
            nearest_energy_gap=gap,
            warnings=tuple(warnings),
        ),
        prepared,
        tuple(basis),
        int(nq),
    )


def _basis_element_terms_for_ansatz(element: QSEBasisElement, *, nq: int) -> PauliPolynomial | None:
    if element.kind == "pauli_string":
        label = str(element.pauli_label_exyz)
        if len(label) != int(nq):
            raise QSERootRefitError(f"Basis element {element.name!r} label length does not match nq={nq}.")
        poly = PauliPolynomial("JW", [PauliTerm(int(nq), ps=label, pc=1.0)])
    elif element.kind == "pauli_polynomial" and element.polynomial is not None:
        poly = element.polynomial
    else:
        raise QSERootRefitError(f"Unsupported QSE basis element kind {element.kind!r} for ansatz refit.")
    specs = iter_runtime_rotation_terms(poly, ignore_identity=True, coefficient_tolerance=1.0e-12, sort_terms=True)
    if not specs:
        return None
    return poly


def build_pauli_rotation_terms_from_qse_basis(
    basis: Sequence[QSEBasisElement],
    *,
    nq: int,
) -> tuple[AnsatzTerm, ...]:
    terms: list[AnsatzTerm] = []
    for idx, element in enumerate(basis):
        poly = _basis_element_terms_for_ansatz(element, nq=int(nq))
        if poly is None:
            continue
        terms.append(
            AnsatzTerm(
                label=f"qse_basis_{idx}:{element.name}",
                polynomial=poly,
                execution_mode="termwise_product",
            )
        )
    if not terms:
        raise QSERootRefitError("QSE operator_basis contains no non-identity Pauli rotations for refit.")
    return tuple(terms)


def _candidate_thetas(count: int) -> list[np.ndarray]:
    n = int(count)
    if n < 0:
        raise QSERootRefitError("Runtime parameter count cannot be negative.")
    zeros = np.zeros(n, dtype=float)
    candidates = [zeros]
    for value in (math.pi / 2.0, -math.pi / 2.0, math.pi / 4.0, -math.pi / 4.0, math.pi, -math.pi):
        for idx in range(n):
            theta = np.zeros(n, dtype=float)
            theta[idx] = float(value)
            candidates.append(theta)
    if n > 0:
        for value in (math.pi / 4.0, -math.pi / 4.0):
            candidates.append(np.full(n, float(value), dtype=float))
    return candidates


def _state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm, _, _ = normalize_statevector(np.asarray(a, dtype=complex).reshape(-1))
    b_norm, _, _ = normalize_statevector(np.asarray(b, dtype=complex).reshape(-1))
    fid = float(abs(np.vdot(a_norm, b_norm)) ** 2)
    return max(0.0, min(1.0, fid))


def fit_pauli_rotation_ansatz(
    *,
    target_state: np.ndarray,
    prepared_state: np.ndarray,
    basis: Sequence[QSEBasisElement],
    nq: int,
    maxiter: int = 200,
) -> PauliRotationRefitResult:
    terms = build_pauli_rotation_terms_from_qse_basis(basis, nq=int(nq))
    layout = build_parameter_layout(
        terms,
        ignore_identity=True,
        coefficient_tolerance=1.0e-12,
        sort_terms=True,
    )
    executor = CompiledAnsatzExecutor(
        terms,
        coefficient_tolerance=1.0e-12,
        ignore_identity=True,
        sort_terms=True,
        parameterization_mode="per_pauli_term",
        parameterization_layout=layout,
    )
    runtime_count = int(layout.runtime_parameter_count)
    if runtime_count <= 0:
        raise QSERootRefitError("Pauli-rotation ansatz has zero runtime parameters after identity filtering.")

    target, _, target_nq = normalize_statevector(target_state)
    reference, _, ref_nq = normalize_statevector(prepared_state)
    if int(target_nq) != int(nq) or int(ref_nq) != int(nq):
        raise QSERootRefitError("Target/reference state qubit count mismatch in ansatz fit.")

    evaluations = 0

    def objective(theta: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        psi = executor.prepare_state(np.asarray(theta, dtype=float), reference)
        return float(1.0 - _state_fidelity(target, psi))

    best_theta = np.zeros(runtime_count, dtype=float)
    best_value = float("inf")
    candidate_count = 0
    for theta0 in _candidate_thetas(runtime_count):
        candidate_count += 1
        value = objective(theta0)
        if value < best_value:
            best_value = float(value)
            best_theta = np.asarray(theta0, dtype=float).copy()

    scipy_summary: dict[str, Any] = {"attempted": False, "available": False}
    if int(maxiter) > 0 and best_value > 1.0e-14:
        try:
            from scipy.optimize import minimize

            scipy_summary["attempted"] = True
            scipy_summary["available"] = True
            result = minimize(
                objective,
                best_theta,
                method="BFGS",
                options={"maxiter": int(maxiter), "gtol": 1.0e-10},
            )
            scipy_summary.update(
                {
                    "success": bool(result.success),
                    "message": str(result.message),
                    "nit": int(getattr(result, "nit", 0)),
                    "nfev": int(getattr(result, "nfev", 0)),
                    "fun": float(result.fun),
                }
            )
            if math.isfinite(float(result.fun)) and float(result.fun) < best_value:
                best_value = float(result.fun)
                best_theta = np.asarray(result.x, dtype=float).reshape(-1)
        except Exception as exc:  # pragma: no cover - exercised only when SciPy is absent/broken.
            scipy_summary.update({"attempted": True, "available": False, "error": str(exc)})

    fitted = executor.prepare_state(best_theta, reference)
    fidelity = _state_fidelity(target, fitted)
    infidelity = float(max(0.0, 1.0 - float(fidelity)))
    theta_logical = project_runtime_theta_block_mean(best_theta, layout)
    optimizer_summary = {
        "method": "deterministic_candidates_then_optional_scipy_bfgs",
        "candidate_count": int(candidate_count),
        "objective_evaluations": int(evaluations),
        "scipy": scipy_summary,
        "best_objective": float(infidelity),
    }
    return PauliRotationRefitResult(
        terms=terms,
        layout=layout,
        theta_runtime=np.asarray(best_theta, dtype=float),
        theta_logical=np.asarray(theta_logical, dtype=float),
        fitted_state=np.asarray(fitted, dtype=complex),
        fidelity=float(fidelity),
        infidelity=float(infidelity),
        optimizer_summary=optimizer_summary,
    )


def _pauli_polynomial_to_terms(poly: PauliPolynomial) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for term in poly.return_polynomial():
        records.append(
            {
                "pauli_exyz": str(term.pw2strng()),
                "coeff": _complex_to_json(complex(term.p_coeff)),
                "nq": int(term.nqubit()),
            }
        )
    return records


def _ansatz_terms_to_payload(terms: Sequence[AnsatzTerm]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, term in enumerate(terms):
        out.append(
            {
                "logical_index": int(idx),
                "label": str(term.label),
                "execution_mode": str(term.execution_mode),
                "terms": _pauli_polynomial_to_terms(term.polynomial),
            }
        )
    return out


def _sparse_amplitudes(psi: np.ndarray, *, cutoff: float) -> dict[str, dict[str, float]]:
    vec, _, nq = normalize_statevector(np.asarray(psi, dtype=complex).reshape(-1))
    out: dict[str, dict[str, float]] = {}
    for idx, amp in enumerate(vec):
        if abs(amp) <= float(cutoff):
            continue
        out[format(idx, f"0{int(nq)}b")] = _complex_to_json(complex(amp))
    return out


def _energy_diagnostics(
    *,
    fitted_state: np.ndarray,
    target_state: np.ndarray,
    hamiltonian: PauliPolynomial | None,
    qse_energy: float,
) -> dict[str, Any]:
    if hamiltonian is None:
        return {
            "available": False,
            "fitted_energy": None,
            "target_energy": None,
            "qse_energy": float(qse_energy),
            "abs_energy_error_vs_qse": None,
        }
    compiled = compile_polynomial_action(hamiltonian)
    fitted_energy, _ = energy_via_one_apply(fitted_state, compiled)
    target_energy, _ = energy_via_one_apply(target_state, compiled)
    return {
        "available": True,
        "fitted_energy": float(fitted_energy),
        "target_energy": float(target_energy),
        "qse_energy": float(qse_energy),
        "abs_energy_error_vs_qse": float(abs(float(fitted_energy) - float(qse_energy))),
        "abs_target_energy_error_vs_qse": float(abs(float(target_energy) - float(qse_energy))),
    }


def _basis_policy_manifest(policy: QSEBasisVectorPolicy) -> dict[str, Any]:
    return {
        "reference_projection": str(policy.reference_projection),
        "basis_vector_normalization": str(policy.basis_vector_normalization),
        "sector_projection": str(policy.sector_projection),
        "sector_label": None if policy.sector_label is None else str(policy.sector_label),
    }


def _selected_operator_labels(terms: Sequence[AnsatzTerm], layout_payload: Mapping[str, Any]) -> list[str]:
    labels: list[str] = []
    for block in layout_payload.get("blocks", []):
        if not isinstance(block, Mapping):
            continue
        for term in block.get("runtime_terms_exyz", []):
            if isinstance(term, Mapping) and "pauli_exyz" in term:
                labels.append(str(term["pauli_exyz"]))
    if labels:
        return labels
    out: list[str] = []
    for term in terms:
        for pauli_term in term.polynomial.return_polynomial():
            out.append(str(pauli_term.pw2strng()))
    return out


def build_qse_root_refit_artifact(
    *,
    config: QSERootRefitConfig,
    qse_payload: Mapping[str, Any],
    target: QSERootTarget,
    prepared: PreparedStateResolution,
    basis: Sequence[QSEBasisElement],
    nq: int,
    fit: PauliRotationRefitResult,
    hamiltonian: HamiltonianResolution,
) -> dict[str, Any]:
    max_infidelity = _validate_threshold(config.max_infidelity, name="max_infidelity", required=True)
    max_energy_error = _validate_threshold(config.max_energy_error, name="max_energy_error")
    energy = _energy_diagnostics(
        fitted_state=fit.fitted_state,
        target_state=target.state,
        hamiltonian=hamiltonian.polynomial,
        qse_energy=target.qse_energy,
    )
    if max_energy_error is not None and not bool(energy["available"]):
        raise QSERootRefitError("Energy threshold requested but Hamiltonian diagnostics are unavailable.")
    fidelity_pass = bool(float(fit.infidelity) <= float(max_infidelity))
    energy_error = energy.get("abs_energy_error_vs_qse")
    energy_pass = True if max_energy_error is None else bool(float(energy_error) <= float(max_energy_error))
    thresholds_pass = bool(fidelity_pass and energy_pass)

    layout_payload = serialize_layout(fit.layout)
    theta_runtime = [float(x) for x in fit.theta_runtime.reshape(-1)]
    theta_logical = [float(x) for x in fit.theta_logical.reshape(-1)]
    reference_state_manifest = build_statevector_manifest(
        psi_state=prepared.state,
        source="qse_root_refit.prepared_state_reference",
        handoff_state_kind="reference_state",
        amplitude_cutoff=float(config.amplitude_cutoff),
    )
    prepared_state_manifest = build_statevector_manifest(
        psi_state=prepared.state,
        source="qse_root_refit.prepared_state",
        handoff_state_kind="prepared_state",
        amplitude_cutoff=float(config.amplitude_cutoff),
    )

    selected = target.selected_eigenvalue
    basis_coeffs_json = list(_sequence(selected.get("basis_coefficients"), name="selected.basis_coefficients"))
    ansatz_payload = {
        "ansatz_schema": "pauli_rotation_ansatz_v1",
        "parameterization_mode": "per_pauli_term",
        "operator_basis_source": "qse_manifest.operator_basis",
        "selected_operator_labels": _selected_operator_labels(fit.terms, layout_payload),
        "generator_terms": _ansatz_terms_to_payload(fit.terms),
        "parameterization": layout_payload,
        "theta_runtime": theta_runtime,
        "theta_logical": theta_logical,
        "reference_state": reference_state_manifest,
        "prepared_state": prepared_state_manifest,
        "qpu_preparable_in_principle": True,
        "matches_scaffold_runtime_contract": False,
        "promotion_status": "candidate_passed_thresholds" if thresholds_pass else "candidate_failed_thresholds",
    }

    artifact = {
        "schema_version": QSE_ROOT_REFIT_SCHEMA_VERSION,
        "pipeline": QSE_ROOT_REFIT_PIPELINE,
        "generated_utc": _utc_now(),
        "backend": "offline_statevector",
        "uses_qiskit": False,
        "controller_boundary": {
            "controller_usable": False,
            "feeds_controller_decisions": False,
            "decision_path_allowed": False,
            "realtime_wiring": False,
            "ansatz_payload_potentially_promotable": bool(thresholds_pass),
            "promotion_requires_runtime_contract_validation": True,
            "matches_scaffold_runtime_contract": False,
            "qse_coefficients_forbidden_to_controller": True,
            "target_state_diagnostics_forbidden_to_controller": True,
        },
        "source": {
            "qse_schema_version": qse_payload.get("schema_version"),
            "qse_pipeline": qse_payload.get("pipeline"),
            "qse_backend": qse_payload.get("backend"),
            "qse_generated_utc": qse_payload.get("generated_utc"),
            "qse_result_json": str(Path(config.qse_result_json)),
            "qse_result_sha256": _sha256_file(Path(config.qse_result_json)),
            "state_index": int(config.state_index),
            "prepared_state_provenance": dict(prepared.provenance),
            "prepared_state_override_used": bool(prepared.override_used),
            "hamiltonian_provenance": hamiltonian.provenance,
            "hamiltonian_override_used": bool(hamiltonian.explicit_override_used),
            "selected_operator_basis": {
                "source": "qse_manifest.operator_basis",
                "basis_size": int(len(basis)),
            },
            "basis_vector_policy": _basis_policy_manifest(target.basis_vector_policy),
        },
        "qse_ritz_diagnostics": {
            "state_index": int(config.state_index),
            "energy": float(target.qse_energy),
            "energy_relative_to_lowest_qse": selected.get("energy_relative_to_lowest_qse"),
            "generalized_residual_norm": selected.get("generalized_residual_norm"),
            "basis_coefficients": basis_coeffs_json,
            "retained_rank": qse_payload.get("diagnostics", {}).get("retained_rank") if isinstance(qse_payload.get("diagnostics"), Mapping) else None,
            "discarded_rank": qse_payload.get("diagnostics", {}).get("discarded_rank") if isinstance(qse_payload.get("diagnostics"), Mapping) else None,
            "overlap_condition_estimate": qse_payload.get("diagnostics", {}).get("overlap_condition_estimate") if isinstance(qse_payload.get("diagnostics"), Mapping) else None,
            "nearest_energy_gap": target.nearest_energy_gap,
            "forbidden_to_controller": True,
        },
        "target_state_diagnostics": {
            "norm_before_normalization": float(target.norm_before_normalization),
            "norm_after_normalization": float(np.linalg.norm(target.state)),
            "nonzero_amplitude_count": int(len(_sparse_amplitudes(target.state, cutoff=float(config.amplitude_cutoff)))),
            "amplitudes_qn_to_q0": _sparse_amplitudes(target.state, cutoff=float(config.amplitude_cutoff)),
            "amplitude_cutoff": float(config.amplitude_cutoff),
            "phase_convention": "first_nonzero_amplitude_real_positive",
            "forbidden_to_controller": True,
        },
        "ansatz_payload": ansatz_payload,
        "fit_summary": {
            "fidelity": float(fit.fidelity),
            "infidelity": float(fit.infidelity),
            "energy_diagnostics": energy,
            "thresholds": {
                "max_infidelity": float(max_infidelity),
                "max_energy_error": None if max_energy_error is None else float(max_energy_error),
            },
            "passes": {
                "fidelity": bool(fidelity_pass),
                "energy_error": bool(energy_pass),
                "all_thresholds": bool(thresholds_pass),
            },
            "optimizer": dict(fit.optimizer_summary),
        },
        "visibility": {
            "controller_visible_payload_refs": [],
            "potentially_promotable_payload_refs": ["ansatz_payload"],
            "diagnostic_only_payload_refs": [
                "source",
                "qse_ritz_diagnostics",
                "target_state_diagnostics",
                "fit_summary",
            ],
            "forbidden_to_controller_refs": [
                "qse_ritz_diagnostics.energy",
                "qse_ritz_diagnostics.energy_relative_to_lowest_qse",
                "qse_ritz_diagnostics.generalized_residual_norm",
                "qse_ritz_diagnostics.basis_coefficients",
                "target_state_diagnostics",
                "target_state_diagnostics.amplitudes_qn_to_q0",
            ],
        },
        "warnings": [
            "offline_qse_root_refit_not_controller_artifact",
            "qse_ritz_coefficients_and_target_state_are_diagnostic_only",
            "ansatz_payload_requires_separate_runtime_contract_validation_before_controller_use",
            *target.warnings,
        ],
    }
    return artifact


def run_qse_root_refit(config: QSERootRefitConfig) -> dict[str, Any]:
    validate_qse_root_refit_config(config)
    qse_path = Path(config.qse_result_json)
    qse_payload = _mapping(_read_json(qse_path), name="qse_result")
    target, prepared, basis, nq = reconstruct_qse_root_target(
        qse_payload,
        qse_result_json=qse_path,
        state_index=int(config.state_index),
        allow_ground_state=bool(config.allow_ground_state),
        prepared_state_json=config.prepared_state_json,
        prepared_state_json_key=str(config.prepared_state_json_key),
        prepared_state_bitstring=config.prepared_state_bitstring,
        amplitude_cutoff=float(config.amplitude_cutoff),
    )
    hamiltonian = _resolve_hamiltonian(
        qse_payload,
        qse_result_json=qse_path,
        hamiltonian_json=config.hamiltonian_json,
        require_energy=config.max_energy_error is not None,
    )
    fit = fit_pauli_rotation_ansatz(
        target_state=target.state,
        prepared_state=prepared.state,
        basis=basis,
        nq=int(nq),
        maxiter=int(config.maxiter),
    )
    artifact = build_qse_root_refit_artifact(
        config=config,
        qse_payload=qse_payload,
        target=target,
        prepared=prepared,
        basis=basis,
        nq=int(nq),
        fit=fit,
        hamiltonian=hamiltonian,
    )
    write_manifest_json(Path(config.output_json), artifact)
    return artifact


def _terms_from_ansatz_payload(ansatz_payload: Mapping[str, Any]) -> tuple[AnsatzTerm, ...]:
    records = _sequence(ansatz_payload.get("generator_terms"), name="ansatz_payload.generator_terms")
    terms: list[AnsatzTerm] = []
    for idx, raw in enumerate(records):
        record = _mapping(raw, name=f"ansatz_payload.generator_terms[{idx}]")
        term_records = _sequence(record.get("terms"), name=f"ansatz_payload.generator_terms[{idx}].terms")
        poly = polynomial_from_serialized_terms(
            term_records,
            require_real_coefficients=True,
            allow_empty_after_pruning=False,
        )
        terms.append(
            AnsatzTerm(
                label=str(record.get("label", f"term_{idx}")),
                polynomial=poly,
                execution_mode=str(record.get("execution_mode", "termwise_product")),
            )
        )
    return tuple(terms)


def reconstruct_ansatz_state_from_payload(payload: Mapping[str, Any]) -> np.ndarray:
    """Replay the emitted minimal ``pauli_rotation_ansatz_v1`` payload."""

    root = _mapping(payload, name="payload")
    ansatz_payload = root.get("ansatz_payload", root)
    ansatz_payload = _mapping(ansatz_payload, name="ansatz_payload")
    if ansatz_payload.get("ansatz_schema") != "pauli_rotation_ansatz_v1":
        raise QSERootRefitError("ansatz_payload.ansatz_schema must be 'pauli_rotation_ansatz_v1'.")
    layout_payload = _mapping(ansatz_payload.get("parameterization"), name="ansatz_payload.parameterization")
    layout = deserialize_layout(layout_payload)
    terms = _terms_from_ansatz_payload(ansatz_payload)
    theta_runtime = np.asarray(
        _sequence(ansatz_payload.get("theta_runtime"), name="ansatz_payload.theta_runtime"),
        dtype=float,
    ).reshape(-1)
    if int(theta_runtime.size) != int(layout.runtime_parameter_count):
        raise QSERootRefitError(
            f"theta_runtime length {theta_runtime.size} does not match runtime_parameter_count {layout.runtime_parameter_count}."
        )
    reference_block = _mapping(ansatz_payload.get("reference_state"), name="ansatz_payload.reference_state")
    reference_state, _provenance = statevector_from_manifest(reference_block, state_key="auto")
    executor = CompiledAnsatzExecutor(
        terms,
        coefficient_tolerance=1.0e-12,
        ignore_identity=True,
        sort_terms=True,
        parameterization_mode="per_pauli_term",
        parameterization_layout=layout,
    )
    return executor.prepare_state(theta_runtime, reference_state)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline refit of a qse_spectra_v1 Ritz root into a Pauli-rotation ansatz artifact."
    )
    parser.add_argument("--qse-result-json", type=Path, required=True)
    parser.add_argument("--state-index", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--allow-ground-state", action="store_true")
    override = parser.add_mutually_exclusive_group()
    override.add_argument("--prepared-state-json", type=Path, default=None)
    override.add_argument("--prepared-state-bitstring", type=str, default=None)
    parser.add_argument(
        "--prepared-state-json-key",
        choices=["auto", "initial_state", "ansatz_input_state"],
        default="auto",
    )
    parser.add_argument("--hamiltonian-json", type=Path, default=None)
    parser.add_argument("--max-infidelity", type=float, default=1.0e-8)
    parser.add_argument("--max-energy-error", type=float, default=None)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--amplitude-cutoff", type=float, default=1.0e-12)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = QSERootRefitConfig(
        qse_result_json=args.qse_result_json,
        state_index=int(args.state_index),
        output_json=args.output_json,
        allow_ground_state=bool(args.allow_ground_state),
        prepared_state_json=args.prepared_state_json,
        prepared_state_json_key=str(args.prepared_state_json_key),
        prepared_state_bitstring=args.prepared_state_bitstring,
        hamiltonian_json=args.hamiltonian_json,
        max_infidelity=float(args.max_infidelity),
        max_energy_error=args.max_energy_error,
        maxiter=int(args.maxiter),
        amplitude_cutoff=float(args.amplitude_cutoff),
    )
    try:
        artifact = run_qse_root_refit(config)
    except QSERootRefitError as exc:
        parser.error(str(exc))
    print(f"output_json: {args.output_json}")
    print(f"state_index: {int(args.state_index)}")
    print(f"fidelity: {artifact['fit_summary']['fidelity']}")
    print(f"infidelity: {artifact['fit_summary']['infidelity']}")
    print("controller_usable: false")
    return 0


__all__ = [
    "QSE_ROOT_REFIT_SCHEMA_VERSION",
    "QSERootRefitConfig",
    "QSERootRefitError",
    "QSERootTarget",
    "PauliRotationRefitResult",
    "build_pauli_rotation_terms_from_qse_basis",
    "build_qse_root_refit_artifact",
    "fit_pauli_rotation_ansatz",
    "reconstruct_ansatz_state_from_payload",
    "reconstruct_qse_root_target",
    "run_qse_root_refit",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
