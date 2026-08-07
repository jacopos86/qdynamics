#!/usr/bin/env python3
"""Calibrate scalar SNAKE value-noise levels to estimator-variance shot proxies.

This is a diagnostic bridge between the repo's scalar post-expectation noise
knob

    E_tilde = E + Normal(0, sigma_E^2),  sigma_E = sigma0_abs/sqrt(N_eff),

and a state-dependent Hamiltonian estimator variance.  It does *not* change the
Paper-I Table-I/II deterministic shot proxy; it emits a separate JSON artifact
that can be used to reason about how large a measurement budget would be needed
to make estimator noise comparable to the scalar value-noise level used in a
noise diagnostic run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.scaffold.runtime_loader import (  # noqa: E402
    _reconstruct_prepared_state_from_runtime_input,
    load_scaffold_runtime_input_from_payload,
)
from src.quantum.vqe_latex_python_pairs import (  # noqa: E402
    apply_pauli_string,
    expval_pauli_polynomial_one_apply,
)


SCHEMA_VERSION = "snake_noise_shot_equivalent_calibration_v1"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object at {path}")
    return dict(payload)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_float(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _normalize_gate_tuple(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return tuple()
    if isinstance(raw, str):
        if raw == "":
            return tuple()
        text = raw.strip()
        if text == "":
            return tuple()
        return tuple(part for part in text.replace(",", " ").split() if part)
    if isinstance(raw, (list, tuple)):
        out: list[str] = []
        for item in raw:
            if isinstance(item, str):
                out.extend(part for part in item.replace(",", " ").split() if part)
            elif item is not None:
                out.append(str(item))
        return tuple(out)
    return (str(raw),)


def resolve_scalar_value_noise_contract(
    payload: Mapping[str, Any],
    *,
    sigma_e_override: float | None = None,
) -> dict[str, Any]:
    """Resolve the scalar value-noise standard deviation from artifact settings."""

    settings = payload.get("settings", {}) if isinstance(payload, Mapping) else {}
    if not isinstance(settings, Mapping):
        settings = {}
    model = str(settings.get("phase3_oracle_value_noise_model", "missing"))
    sigma0_abs = _safe_float(settings.get("phase3_oracle_value_noise_sigma0_abs"))
    n_eff = _safe_float(settings.get("phase3_oracle_value_noise_n_eff"))
    recorded_std = _safe_float(settings.get("phase3_oracle_value_noise_std"))

    if sigma_e_override is not None:
        sigma_e = float(sigma_e_override)
        source = "cli_override"
    elif recorded_std is not None:
        sigma_e = float(recorded_std)
        source = "settings.phase3_oracle_value_noise_std"
    elif sigma0_abs is not None and n_eff is not None and n_eff > 0.0:
        sigma_e = float(sigma0_abs / math.sqrt(n_eff))
        source = "settings.sigma0_abs_over_sqrt_n_eff"
    else:
        raise ValueError(
            "Could not resolve scalar value-noise sigma_E; pass --sigma-e or use an artifact "
            "with phase3_oracle_value_noise_std / sigma0_abs+n_eff."
        )
    if sigma_e <= 0.0 or not math.isfinite(sigma_e):
        raise ValueError(f"sigma_E must be positive and finite, got {sigma_e!r}")

    derived_from_n_eff = None
    if sigma0_abs is not None and n_eff is not None and n_eff > 0.0:
        derived_from_n_eff = float(sigma0_abs / math.sqrt(n_eff))

    return {
        "model": model,
        "sigma_E": float(sigma_e),
        "sigma_E_source": source,
        "sigma0_abs": None if sigma0_abs is None else float(sigma0_abs),
        "N_eff": None if n_eff is None else float(n_eff),
        "std_recorded": None if recorded_std is None else float(recorded_std),
        "std_derived_from_N_eff": derived_from_n_eff,
        "seed": settings.get("phase3_oracle_value_noise_seed"),
        "physical_shots_interpretation": "not_physical_shot_count_without_estimator_variance_calibration",
    }


def resolve_synthetic_depolarizing_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    settings = payload.get("settings", {}) if isinstance(payload, Mapping) else {}
    if not isinstance(settings, Mapping):
        settings = {}
    p1q = _safe_float(settings.get("phase3_oracle_synthetic_depolarizing_1q_error"))
    p2q = _safe_float(settings.get("phase3_oracle_synthetic_depolarizing_2q_error"))
    return {
        "gradient_mode": settings.get("phase3_oracle_gradient_mode"),
        "execution_surface": settings.get("phase3_oracle_execution_surface"),
        "inner_objective_mode": settings.get("phase3_oracle_inner_objective_mode"),
        "p1q": p1q,
        "p2q": p2q,
        "one_qubit_gates": list(
            _normalize_gate_tuple(settings.get("phase3_oracle_synthetic_depolarizing_1q_gates"))
        ),
        "two_qubit_gates": list(
            _normalize_gate_tuple(settings.get("phase3_oracle_synthetic_depolarizing_2q_gates"))
        ),
        "calibration_note": "density-matrix estimator calibration uses this synthetic depolarizing contract when enabled",
    }


def _collect_active_pauli_terms(polynomial: Any, *, coeff_tol: float) -> tuple[int, list[tuple[str, float]]]:
    terms_by_label: dict[str, complex] = {}
    width: int | None = None
    for term in polynomial.return_polynomial():
        label = str(term.pw2strng()).lower()
        coeff = complex(term.p_coeff)
        if any(ch not in {"e", "x", "y", "z"} for ch in label):
            raise ValueError(f"Unsupported repo Pauli label {label!r}")
        if width is None:
            width = int(len(label))
        elif int(len(label)) != int(width):
            raise ValueError("Hamiltonian Pauli labels have inconsistent widths")
        if abs(coeff) <= float(coeff_tol):
            continue
        terms_by_label[label] = terms_by_label.get(label, 0.0j) + coeff

    if width is None:
        width = 0
    identity = "e" * int(width)
    active: list[tuple[str, float]] = []
    for label, coeff in terms_by_label.items():
        if label == identity or abs(coeff) <= float(coeff_tol):
            continue
        if abs(coeff.imag) > max(1e-10, 100.0 * float(coeff_tol)):
            raise ValueError(
                "Hamiltonian term has non-negligible imaginary coefficient; "
                f"label={label}, coeff={coeff}"
            )
        active.append((label, float(coeff.real)))
    active.sort(key=lambda item: (item[0], item[1]))
    return int(width), active


def ungrouped_equal_allocation_calibration(
    polynomial: Any,
    statevector: Any,
    *,
    target_sigma: float,
    coeff_tol: float = 1e-12,
) -> dict[str, Any]:
    r"""Compute state-dependent ungrouped Pauli estimator variance.

    For H = sum_alpha c_alpha P_alpha and state psi, this reports

        C_state = sum_alpha c_alpha^2 (1 - <P_alpha>_psi^2).

    If each active Pauli term receives M shots, then Var[H_hat] = C_state/M,
    so M = C_state/sigma_E^2 shots per active Pauli term.  The corresponding
    total term-shot budget is active_term_count * M.
    """

    sigma = float(target_sigma)
    if sigma <= 0.0 or not math.isfinite(sigma):
        raise ValueError("target_sigma must be positive and finite")
    psi = np.asarray(statevector, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(psi))
    if norm <= 0.0:
        raise ValueError("statevector has zero norm")
    psi = psi / norm
    width, active = _collect_active_pauli_terms(polynomial, coeff_tol=float(coeff_tol))
    if width and psi.size != (1 << int(width)):
        raise ValueError(
            f"Statevector length mismatch: got {psi.size}, expected {1 << int(width)} for nq={width}."
        )

    term_rows: list[dict[str, Any]] = []
    c_state = 0.0
    for label, coeff in active:
        p_psi = apply_pauli_string(psi, label)
        expectation = complex(np.vdot(psi, p_psi))
        if abs(expectation.imag) > max(1e-10, 100.0 * float(coeff_tol)):
            raise ValueError(f"Pauli expectation has non-negligible imaginary part for {label}: {expectation}")
        exp_real = float(expectation.real)
        exp_clamped = max(-1.0, min(1.0, exp_real))
        variance = float(max(0.0, 1.0 - exp_clamped * exp_clamped))
        contribution = float((float(coeff) ** 2) * variance)
        c_state += contribution
        term_rows.append(
            {
                "pauli_exyz": label,
                "coeff": float(coeff),
                "expectation": exp_real,
                "variance": variance,
                "c_squared_variance_contribution": contribution,
            }
        )

    shots_per_term = float(c_state / (sigma * sigma))
    return {
        "model": "statevector_ungrouped_equal_allocation_pauli_variance_v1",
        "formula": "C_state=sum_alpha c_alpha^2*(1-<P_alpha>_psi^2); shots_per_term=C_state/sigma_E^2",
        "target_sigma": sigma,
        "active_pauli_term_count": int(len(active)),
        "hamiltonian_width_qubits": int(width),
        "C_state_variance": float(c_state),
        "shots_per_active_pauli_term_for_sigma_E": shots_per_term,
        "total_term_shots_equal_allocation_for_sigma_E": float(int(len(active)) * shots_per_term),
        "term_rows": term_rows,
    }


def grouped_qwc_statevector_calibration(
    polynomial: Any,
    statevector: Any,
    *,
    target_sigma: float,
    coeff_tol: float = 1e-12,
) -> dict[str, Any]:
    """Reuse the repo's deterministic QWC grouped statevector variance proxy."""

    from pipelines.exact_bench.generic_static_metric_enrichment import (  # noqa: WPS433
        _pauli_polynomial_grouped_statevector_variance_proxy,
    )

    grouped = _pauli_polynomial_grouped_statevector_variance_proxy(
        polynomial,
        statevector,
        observable_kind="hamiltonian_energy",
        target_sigma=float(target_sigma),
        coeff_tol=float(coeff_tol),
    )
    out = dict(grouped)
    out["interpretation"] = (
        "C_var is the total grouped-measurement shot proxy under optimal allocation "
        "across deterministic greedy QWC groups for this state and sigma_E."
    )
    out["total_group_shots_optimal_allocation_for_sigma_E"] = float(out.get("C_var", 0.0))
    return out


def _to_qiskit_ixyz(label_exyz: str) -> str:
    return (
        str(label_exyz)
        .lower()
        .replace("e", "I")
        .replace("x", "X")
        .replace("y", "Y")
        .replace("z", "Z")
    )


def _sparse_pauli_op_from_terms(terms: list[tuple[str, complex]], *, n_qubits: int, coeff_tol: float):
    from qiskit.quantum_info import SparsePauliOp

    cleaned: list[tuple[str, complex]] = []
    for label, coeff in terms:
        coeff_c = complex(coeff)
        if abs(coeff_c) <= float(coeff_tol):
            continue
        cleaned.append((_to_qiskit_ixyz(label), coeff_c))
    if not cleaned:
        cleaned = [("I" * int(n_qubits), 0.0 + 0.0j)]
    return SparsePauliOp.from_list(cleaned).simplify(atol=float(coeff_tol))


def _density_matrix_data(density_matrix: Any) -> np.ndarray:
    raw = getattr(density_matrix, "data", density_matrix)
    rho = np.asarray(raw, dtype=complex)
    if rho.ndim == 1:
        rho = np.diag(rho)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("density matrix must be a square matrix or qiskit DensityMatrix")
    tr = complex(np.trace(rho))
    if abs(tr) <= 0.0:
        raise ValueError("density matrix has zero trace")
    return np.asarray(rho / tr, dtype=complex)


def _real_trace_expectation(rho: np.ndarray, matrix: np.ndarray, *, label: str, coeff_tol: float) -> float:
    value = complex(np.trace(rho @ matrix))
    if abs(value.imag) > max(1e-10, 100.0 * float(coeff_tol)):
        raise ValueError(f"density-matrix expectation {label} has non-negligible imaginary part: {value}")
    return float(value.real)


def ungrouped_density_matrix_calibration(
    polynomial: Any,
    density_matrix: Any,
    *,
    target_sigma: float,
    coeff_tol: float = 1e-12,
) -> dict[str, Any]:
    """Compute ungrouped estimator variance for a possibly mixed density matrix."""

    sigma = float(target_sigma)
    if sigma <= 0.0 or not math.isfinite(sigma):
        raise ValueError("target_sigma must be positive and finite")
    rho = _density_matrix_data(density_matrix)
    width, active = _collect_active_pauli_terms(polynomial, coeff_tol=float(coeff_tol))
    if width and rho.shape[0] != (1 << int(width)):
        raise ValueError(
            f"Density-matrix dimension mismatch: got {rho.shape[0]}, expected {1 << int(width)} for nq={width}."
        )

    term_rows: list[dict[str, Any]] = []
    c_state = 0.0
    for label, coeff in active:
        op = _sparse_pauli_op_from_terms([(label, 1.0)], n_qubits=int(width), coeff_tol=float(coeff_tol))
        mat = np.asarray(op.to_matrix(), dtype=complex)
        exp_real = _real_trace_expectation(rho, mat, label=label, coeff_tol=float(coeff_tol))
        exp_clamped = max(-1.0, min(1.0, exp_real))
        variance = float(max(0.0, 1.0 - exp_clamped * exp_clamped))
        contribution = float((float(coeff) ** 2) * variance)
        c_state += contribution
        term_rows.append(
            {
                "pauli_exyz": label,
                "coeff": float(coeff),
                "expectation": exp_real,
                "variance": variance,
                "c_squared_variance_contribution": contribution,
            }
        )
    shots_per_term = float(c_state / (sigma * sigma))
    return {
        "model": "density_matrix_ungrouped_equal_allocation_pauli_variance_v1",
        "formula": "C_rho=sum_alpha c_alpha^2*(1-Tr(rho P_alpha)^2); shots_per_term=C_rho/sigma_E^2",
        "target_sigma": sigma,
        "active_pauli_term_count": int(len(active)),
        "hamiltonian_width_qubits": int(width),
        "C_density_matrix_variance": float(c_state),
        "shots_per_active_pauli_term_for_sigma_E": shots_per_term,
        "total_term_shots_equal_allocation_for_sigma_E": float(int(len(active)) * shots_per_term),
        "term_rows": term_rows,
    }


def grouped_qwc_density_matrix_calibration(
    polynomial: Any,
    density_matrix: Any,
    *,
    target_sigma: float,
    coeff_tol: float = 1e-12,
) -> dict[str, Any]:
    """Compute grouped QWC estimator variance for a possibly mixed density matrix."""

    from pipelines.exact_bench.generic_static_metric_enrichment import (  # noqa: WPS433
        _pauli_polynomial_qwc_groups,
    )

    sigma = float(target_sigma)
    if sigma <= 0.0 or not math.isfinite(sigma):
        raise ValueError("target_sigma must be positive and finite")
    rho = _density_matrix_data(density_matrix)
    groups, width, active_terms = _pauli_polynomial_qwc_groups(polynomial, coeff_tol=float(coeff_tol))
    if width is not None and rho.shape[0] != (1 << int(width)):
        raise ValueError(
            f"Density-matrix dimension mismatch: got {rho.shape[0]}, expected {1 << int(width)} for nq={width}."
        )
    if width is None:
        width = int(round(math.log2(rho.shape[0]))) if rho.shape[0] > 0 else 0

    group_variances: list[float] = []
    group_sqrt_variances: list[float] = []
    group_expectations: list[float] = []
    group_second_moments: list[float] = []
    for group in groups:
        terms = [(str(label), complex(coeff)) for label, coeff in group.get("terms", [])]
        op = _sparse_pauli_op_from_terms(terms, n_qubits=int(width), coeff_tol=float(coeff_tol))
        mat = np.asarray(op.to_matrix(), dtype=complex)
        exp_real = _real_trace_expectation(rho, mat, label=str(group.get("basis_key", "group")), coeff_tol=float(coeff_tol))
        second = _real_trace_expectation(
            rho,
            np.asarray(mat @ mat, dtype=complex),
            label=f"{group.get('basis_key', 'group')}^2",
            coeff_tol=float(coeff_tol),
        )
        variance = float(max(0.0, second - exp_real * exp_real))
        group_variances.append(variance)
        group_sqrt_variances.append(float(math.sqrt(variance)))
        group_expectations.append(exp_real)
        group_second_moments.append(second)
    c_var = float((sum(group_sqrt_variances) ** 2) / (sigma * sigma))
    return {
        "model": "deterministic_greedy_qwc_grouped_density_matrix_variance_proxy_v1",
        "observable_kind": "hamiltonian_energy",
        "target_sigma": sigma,
        "term_count": int(len(active_terms)),
        "group_count": int(len(groups)),
        "group_basis_keys": [str(group["basis_key"]) for group in groups],
        "group_variances": group_variances,
        "group_sqrt_variances": group_sqrt_variances,
        "group_expectations": group_expectations,
        "group_second_moments": group_second_moments,
        "C_var": c_var,
        "total_group_shots_optimal_allocation_for_sigma_E": c_var,
        "density_matrix_dimension": int(rho.shape[0]),
        "variance_formula": "Tr(rho O_b^2)_minus_Tr(rho O_b)^2",
        "interpretation": (
            "C_var is the total grouped-measurement shot proxy under optimal allocation "
            "across deterministic greedy QWC groups for the gate-noisy density matrix and sigma_E."
        ),
    }


def _density_matrix_from_synthetic_depolarizing_circuit(
    payload: Mapping[str, Any],
    *,
    artifact_json: Path,
) -> tuple[Any, dict[str, Any]]:
    """Run the reconstructed final circuit under the artifact's synthetic depolarizing model."""

    settings = payload.get("settings", {}) if isinstance(payload, Mapping) else {}
    if not isinstance(settings, Mapping):
        settings = {}
    contract = resolve_synthetic_depolarizing_contract(payload)
    p1q = contract.get("p1q")
    p2q = contract.get("p2q")
    if p1q is None and p2q is None:
        raise ValueError("Artifact does not record synthetic depolarizing p1q/p2q settings.")

    from pipelines.exact_bench.noise_oracle_defaults import (  # noqa: WPS433
        SYNTHETIC_DEPOLARIZING_1Q_GATES_DEFAULT,
        SYNTHETIC_DEPOLARIZING_2Q_GATES_DEFAULT,
    )
    from pipelines.exact_bench.noise_oracle_runtime import (  # noqa: WPS433
        OracleConfig,
        _build_synthetic_depolarizing_noise_model,
        _compile_circuit_for_aer_density_matrix,
    )
    from pipelines.scaffold.adapt_circuit_cost import reconstruct_imported_adapt_circuit  # noqa: WPS433
    from qiskit.quantum_info import DensityMatrix
    from qiskit_aer import AerSimulator

    oneq_gates = tuple(contract.get("one_qubit_gates") or SYNTHETIC_DEPOLARIZING_1Q_GATES_DEFAULT)
    twoq_gates = tuple(contract.get("two_qubit_gates") or SYNTHETIC_DEPOLARIZING_2Q_GATES_DEFAULT)
    seed = int(_safe_float(settings.get("phase3_oracle_seed")) or _safe_float(settings.get("seed")) or 7)
    seed_transpiler_raw = _safe_float(settings.get("phase3_oracle_seed_transpiler"))
    transpile_optimization_level = int(
        _safe_float(settings.get("phase3_oracle_transpile_optimization_level")) or 1
    )
    cfg = OracleConfig(
        noise_mode="aer_density_matrix_synthetic_depolarizing",
        seed=int(seed),
        seed_transpiler=(None if seed_transpiler_raw is None else int(seed_transpiler_raw)),
        transpile_optimization_level=int(transpile_optimization_level),
        execution_surface="expectation_v1",
        synthetic_depolarizing_1q_error=float(p1q or 0.0),
        synthetic_depolarizing_2q_error=float(p2q or 0.0),
        synthetic_depolarizing_1q_gates=tuple(str(g) for g in oneq_gates),
        synthetic_depolarizing_2q_gates=tuple(str(g) for g in twoq_gates),
    )
    noise_model = _build_synthetic_depolarizing_noise_model(cfg)
    simulator_kwargs: dict[str, Any] = {"method": "density_matrix", "noise_model": noise_model}
    simulator_kwargs["seed_simulator"] = int(seed)
    simulator = AerSimulator(**simulator_kwargs)
    bundle = reconstruct_imported_adapt_circuit(payload)
    circuit = bundle["circuit"]
    base = _compile_circuit_for_aer_density_matrix(
        circuit,
        simulator=simulator,
        noise_model=noise_model,
        seed_transpiler=int(seed if seed_transpiler_raw is None else seed_transpiler_raw),
        optimization_level=int(transpile_optimization_level),
    )
    rho_circuit = base["compiled"].copy()
    rho_circuit.save_density_matrix(label="snake_noise_calibration_rho")
    result = simulator.run(rho_circuit, shots=1).result()
    rho = DensityMatrix(result.data(0)["snake_noise_calibration_rho"])
    meta = {
        "state_model": "aer_density_matrix_synthetic_depolarizing_final_circuit",
        "artifact_json": str(Path(artifact_json).resolve()),
        "p1q": float(p1q or 0.0),
        "p2q": float(p2q or 0.0),
        "one_qubit_gates": [str(g) for g in oneq_gates],
        "two_qubit_gates": [str(g) for g in twoq_gates],
        "seed_simulator": int(seed),
        "seed_transpiler": int(seed if seed_transpiler_raw is None else seed_transpiler_raw),
        "transpile_optimization_level": int(transpile_optimization_level),
        "compiled_num_qubits": int(base.get("compiled_num_qubits", getattr(base.get("compiled"), "num_qubits", 0))),
        "compile_signature": dict(base.get("compile_signature", {})),
        "noise_model_basis_gates": [str(g) for g in getattr(noise_model, "basis_gates", ()) or ()],
        "density_matrix_trace": float(np.real(np.trace(np.asarray(rho.data, dtype=complex)))),
    }
    return rho, meta


def replay_final_state_and_hamiltonian(
    payload: Mapping[str, Any],
    *,
    artifact_json: Path,
) -> tuple[np.ndarray, Any, dict[str, Any]]:
    """Replay the artifact's final ansatz state and return (psi, H, metadata)."""

    runtime_input = load_scaffold_runtime_input_from_payload(payload, artifact_json=artifact_json)
    psi = _reconstruct_prepared_state_from_runtime_input(
        selected_terms=runtime_input.selected_terms,
        layout=runtime_input.base_layout,
        theta_runtime=np.asarray(runtime_input.theta_runtime, dtype=float).reshape(-1),
        psi_ref=np.asarray(runtime_input.psi_ref, dtype=complex).reshape(-1),
    )
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(psi))
    if norm <= 0.0:
        raise ValueError("replayed final state has zero norm")
    psi = psi / norm
    meta = {
        "state_model": "replayed_noiseless_ansatz_state_from_artifact_terms_and_theta",
        "problem_family": str(runtime_input.resolved_problem.family_key),
        "num_qubits": int(runtime_input.resolved_problem.layout.total_qubits),
        "selected_term_count": int(len(runtime_input.selected_terms)),
        "runtime_parameter_count": int(runtime_input.base_layout.runtime_parameter_count),
        "logical_parameter_count": int(runtime_input.base_layout.logical_parameter_count),
        "loader_provenance": dict(runtime_input.provenance),
        "state_norm_after_replay": float(norm),
        "note": (
            "This calibration uses the replayed pure ansatz state.  Synthetic depolarizing gate-noise "
            "density-matrix variance calibration is a separate extension."
        ),
    }
    return psi, runtime_input.h_poly, meta


def build_calibration_payload(
    artifact_json: Path,
    *,
    sigma_e_override: float | None = None,
    coeff_tol: float = 1e-12,
    include_term_rows: bool = True,
    include_grouped_qwc: bool = True,
    include_density_matrix: bool = True,
) -> dict[str, Any]:
    source_path = Path(artifact_json).resolve()
    payload = _read_json(source_path)
    settings = payload.get("settings", {}) if isinstance(payload.get("settings", {}), Mapping) else {}
    adapt_vqe = payload.get("adapt_vqe", {}) if isinstance(payload.get("adapt_vqe", {}), Mapping) else {}
    noise = resolve_scalar_value_noise_contract(payload, sigma_e_override=sigma_e_override)
    sigma_e = float(noise["sigma_E"])

    psi, h_poly, replay_meta = replay_final_state_and_hamiltonian(payload, artifact_json=source_path)
    exact_energy = float(expval_pauli_polynomial_one_apply(psi, h_poly, tol=float(coeff_tol)))
    exact_gs_energy = _safe_float(adapt_vqe.get("exact_gs_energy"))
    exact_abs_delta = None
    if exact_gs_energy is not None:
        exact_abs_delta = float(abs(exact_energy - exact_gs_energy))

    ungrouped = ungrouped_equal_allocation_calibration(
        h_poly,
        psi,
        target_sigma=sigma_e,
        coeff_tol=float(coeff_tol),
    )
    if not include_term_rows:
        ungrouped = dict(ungrouped)
        ungrouped["term_rows_omitted"] = int(len(ungrouped.get("term_rows", [])))
        ungrouped.pop("term_rows", None)

    variance_calibration: dict[str, Any] = {
        "target_scalar_sigma_E": sigma_e,
        "pure_state_replay": {
            "ungrouped_equal_allocation": ungrouped,
        },
        "ungrouped_equal_allocation": ungrouped,
    }
    if include_grouped_qwc:
        grouped_statevector = grouped_qwc_statevector_calibration(
            h_poly,
            psi,
            target_sigma=sigma_e,
            coeff_tol=float(coeff_tol),
        )
        variance_calibration["pure_state_replay"]["grouped_qwc_optimal_allocation"] = grouped_statevector
        variance_calibration["grouped_qwc_optimal_allocation"] = grouped_statevector

    density_meta: dict[str, Any] | None = None
    if include_density_matrix:
        try:
            rho_gate, density_meta = _density_matrix_from_synthetic_depolarizing_circuit(
                payload,
                artifact_json=source_path,
            )
            density_ungrouped = ungrouped_density_matrix_calibration(
                h_poly,
                rho_gate,
                target_sigma=sigma_e,
                coeff_tol=float(coeff_tol),
            )
            if not include_term_rows:
                density_ungrouped = dict(density_ungrouped)
                density_ungrouped["term_rows_omitted"] = int(len(density_ungrouped.get("term_rows", [])))
                density_ungrouped.pop("term_rows", None)
            density_payload: dict[str, Any] = {
                "metadata": density_meta,
                "ungrouped_equal_allocation": density_ungrouped,
            }
            if include_grouped_qwc:
                density_payload["grouped_qwc_optimal_allocation"] = grouped_qwc_density_matrix_calibration(
                    h_poly,
                    rho_gate,
                    target_sigma=sigma_e,
                    coeff_tol=float(coeff_tol),
                )
            variance_calibration["synthetic_depolarizing_density_matrix"] = density_payload
        except Exception as exc:
            variance_calibration["synthetic_depolarizing_density_matrix"] = {
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifact": {
            "path": str(source_path),
            "sha256": _sha256_file(source_path),
        },
        "problem": {
            "family": str(settings.get("problem", replay_meta.get("problem_family", "unknown"))),
            "L": settings.get("L"),
            "t": settings.get("t"),
            "u": settings.get("u"),
            "boundary": settings.get("boundary"),
        },
        "noise_contract": {
            "scalar_value_noise": noise,
            "synthetic_depolarizing": resolve_synthetic_depolarizing_contract(payload),
        },
        "artifact_result_summary": {
            "ansatz_depth": adapt_vqe.get("ansatz_depth"),
            "stop_reason": adapt_vqe.get("stop_reason"),
            "reported_energy": adapt_vqe.get("energy"),
            "reported_abs_delta_e": adapt_vqe.get("abs_delta_e"),
            "reported_exact_gs_energy": adapt_vqe.get("exact_gs_energy"),
            "benchmark_target_hit_success": adapt_vqe.get("benchmark_target_hit_success"),
        },
        "replay_summary": {
            **replay_meta,
            "exact_hamiltonian_energy_on_replayed_state": exact_energy,
            "exact_abs_delta_e_on_replayed_state": exact_abs_delta,
            "reported_minus_replayed_exact_energy": (
                None
                if _safe_float(adapt_vqe.get("energy")) is None
                else float(float(adapt_vqe.get("energy")) - exact_energy)
            ),
        },
        "variance_calibration": variance_calibration,
        "claim_boundaries": [
            "N_eff is the scalar post-expectation value-noise scale used by the noisy objective, not a physical shot count.",
            "Equivalent-shot quantities here are estimator-variance diagnostics at the replayed ansatz state.",
            "This artifact does not modify Paper-I Table-I/II deterministic shot-proxy semantics.",
            "Synthetic depolarizing density-matrix calibration is included when the source artifact records p1q/p2q and Aer supports the circuit.",
        ],
    }
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True, type=Path, help="Completed/stopped SNAKE JSON artifact")
    parser.add_argument("--output-json", type=Path, default=None, help="Write calibration JSON here; stdout if omitted")
    parser.add_argument("--sigma-e", type=float, default=None, help="Override scalar target sigma_E")
    parser.add_argument("--coeff-tol", type=float, default=1e-12)
    parser.add_argument("--omit-term-rows", action="store_true", help="Drop per-Pauli rows from output")
    parser.add_argument("--no-grouped-qwc", action="store_true", help="Skip grouped QWC variance proxy")
    parser.add_argument("--no-density-matrix", action="store_true", help="Skip synthetic depolarizing density-matrix calibration")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    calibration = build_calibration_payload(
        args.input_json,
        sigma_e_override=args.sigma_e,
        coeff_tol=float(args.coeff_tol),
        include_term_rows=not bool(args.omit_term_rows),
        include_grouped_qwc=not bool(args.no_grouped_qwc),
        include_density_matrix=not bool(args.no_density_matrix),
    )
    text = json.dumps(calibration, indent=2, sort_keys=True)
    if args.output_json is None:
        print(text)
    else:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
        print(str(args.output_json))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
