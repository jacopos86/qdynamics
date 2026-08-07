#!/usr/bin/env python3
"""Audit the production molecular-vibronic H2O linear-FD fixture.

The audit is diagnostic: it does not launch ADAPT, alter a checkpoint, or
promote evidence.  It independently reconstructs tensor finite differences,
checks the electronic and vibronic fixed-sector matrices, validates the exact
eigenstate, and records the intended RHF-referenced model boundary.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.contracts.problem import ProblemRequest
from pipelines.static_adapt.builders.problem_registry import resolve_problem_context
from pipelines.static_adapt.builders.problem_setup import (
    resolve_exact_reference_state_for_problem,
)
from pipelines.static_adapt.engine_support import _apply_pauli_polynomial_uncached
from pipelines.static_adapt.sector_invariants import (
    FixedCountSectorStateAuditor,
    audit_candidate_pool_sector_contract,
)
from src.quantum.chemistry.generate_h2o_linear_fd_fixture import (
    _sector_sparse_matrix,
)
from src.quantum.chemistry.vibronic_h2o_linear_fd import (
    ProductionVibronicH2OFixture,
    RegisterLayout,
    build_production_vibronic_h2o_linear_fd_runtime_model,
    load_cached_production_vibronic_h2o_linear_fd_fixture,
    load_production_vibronic_h2o_fixture,
    validate_paper_iv_h2o_linear_fd_evidence_fixture,
)
from src.quantum.hartree_fock_reference_state import hartree_fock_occupied_qubits
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


AUDIT_SCHEMA = "molecular_vibronic_h2o_linear_fd_correctness_audit_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_hermiticity_residual(matrix: Any) -> float:
    delta = matrix - matrix.getH()
    delta_norm = float(np.sqrt(np.sum(np.abs(delta.data) ** 2)))
    matrix_norm = float(np.sqrt(np.sum(np.abs(matrix.data) ** 2)))
    return float(delta_norm / max(matrix_norm, 1.0e-30))


def _one_body_hermiticity_residual(matrix: np.ndarray) -> float:
    arr = np.asarray(matrix)
    return float(
        np.linalg.norm(arr - arr.T.conj())
        / max(float(np.linalg.norm(arr)), 1.0e-30)
    )


def _chemist_eri_symmetry_residual(tensor: np.ndarray) -> float:
    arr = np.asarray(tensor)
    residuals = (
        np.linalg.norm(arr - np.swapaxes(arr, 0, 1)),
        np.linalg.norm(arr - np.swapaxes(arr, 2, 3)),
        np.linalg.norm(arr - np.transpose(arr, (2, 3, 0, 1))),
    )
    return float(max(residuals) / max(float(np.linalg.norm(arr)), 1.0e-30))


def _closed_shell_energy_from_tensors(
    scalar: float,
    one_body: np.ndarray,
    two_body: np.ndarray,
    *,
    n_occupied: int,
) -> float:
    h = np.asarray(one_body, dtype=float)
    g = np.asarray(two_body, dtype=float)
    occupied = range(int(n_occupied))
    return float(
        float(scalar)
        + 2.0 * sum(float(h[i, i]) for i in occupied)
        + sum(
            2.0 * float(g[i, i, j, j]) - float(g[i, j, j, i])
            for i in occupied
            for j in occupied
        )
    )


def _lowest_eigenpair(matrix: Any) -> tuple[float, np.ndarray]:
    dimension = int(matrix.shape[0])
    if dimension <= 512:
        dense = np.asarray(matrix.toarray(), dtype=complex)
        values, vectors = np.linalg.eigh(dense)
        return float(values[0]), np.asarray(vectors[:, 0], dtype=complex)
    from scipy.sparse.linalg import eigsh

    values, vectors = eigsh(matrix, k=1, which="SA", tol=1.0e-12)
    index = int(np.argmin(values))
    return float(values[index]), np.asarray(vectors[:, index], dtype=complex)


def _electronic_layout(fixture: ProductionVibronicH2OFixture) -> RegisterLayout:
    n_fermion = int(fixture.layout.n_fermion_qubits)
    return RegisterLayout(
        n_fermion_qubits=n_fermion,
        fermion_qubits=tuple(range(n_fermion)),
        boson_modes=(),
        spin_orbital_ordering="blocked",
    )


def _finite_difference_audit(fixture: ProductionVibronicH2OFixture) -> dict[str, Any]:
    aligned = {row.aligned_tensor_id: row for row in fixture.aligned_tensors}
    rows: list[dict[str, Any]] = []
    for derivative in fixture.first_derivatives:
        plus = aligned[derivative.plus_aligned_tensor_id]
        minus = aligned[derivative.minus_aligned_tensor_id]
        denominator = 2.0 * float(derivative.q_step_au)
        scalar = (float(plus.scalar_energy_hartree) - float(minus.scalar_energy_hartree)) / denominator
        one_body = (
            np.asarray(plus.one_body_integrals, dtype=float)
            - np.asarray(minus.one_body_integrals, dtype=float)
        ) / denominator
        two_body = (
            np.asarray(plus.two_body_integrals, dtype=float)
            - np.asarray(minus.two_body_integrals, dtype=float)
        ) / denominator
        diagnostics = dict(derivative.finite_difference_diagnostics)
        row = {
            "mode_index": int(derivative.mode_index),
            "mode_label": str(derivative.mode_label),
            "q_step_au": float(derivative.q_step_au),
            "scalar_reconstruction_abs_residual": float(
                abs(scalar - float(derivative.scalar_derivative_hartree_per_q))
            ),
            "one_body_reconstruction_max_abs_residual": float(
                np.max(np.abs(one_body - np.asarray(derivative.one_body_derivative)))
            ),
            "two_body_reconstruction_max_abs_residual": float(
                np.max(np.abs(two_body - np.asarray(derivative.two_body_derivative)))
            ),
            "one_body_hermiticity_residual": _one_body_hermiticity_residual(
                derivative.one_body_derivative
            ),
            "two_body_chemist_symmetry_residual": _chemist_eri_symmetry_residual(
                derivative.two_body_derivative
            ),
            "alternate_step_validation_passed": bool(diagnostics.get("passed", False)),
            "alternate_step_max_scaled_residual": diagnostics.get("max_scaled_residual"),
            "alternate_step_legacy_relative_drift": diagnostics.get("legacy_relative_drift"),
        }
        row["passed"] = bool(
            row["scalar_reconstruction_abs_residual"] <= 1.0e-12
            and row["one_body_reconstruction_max_abs_residual"] <= 1.0e-12
            and row["two_body_reconstruction_max_abs_residual"] <= 1.0e-12
            and row["one_body_hermiticity_residual"] <= 1.0e-10
            and row["two_body_chemist_symmetry_residual"] <= 1.0e-8
            and row["alternate_step_validation_passed"]
        )
        rows.append(row)
    return {
        "passed": bool(rows and all(bool(row["passed"]) for row in rows)),
        "modes": rows,
    }


def _tensor_audit(fixture: ProductionVibronicH2OFixture) -> dict[str, Any]:
    active = fixture.active_space
    aligned_one = [
        _one_body_hermiticity_residual(row.one_body_integrals)
        for row in fixture.aligned_tensors
    ]
    aligned_two = [
        _chemist_eri_symmetry_residual(row.two_body_integrals)
        for row in fixture.aligned_tensors
    ]
    payload = {
        "active_one_body_hermiticity_residual": _one_body_hermiticity_residual(
            active.one_body_integrals
        ),
        "active_two_body_chemist_symmetry_residual": _chemist_eri_symmetry_residual(
            active.two_body_integrals
        ),
        "max_aligned_one_body_hermiticity_residual": float(max(aligned_one, default=0.0)),
        "max_aligned_two_body_chemist_symmetry_residual": float(max(aligned_two, default=0.0)),
        "alignment_record_count": int(len(fixture.alignment_diagnostics)),
        "all_alignment_records_passed": bool(
            fixture.alignment_diagnostics
            and all(row.passed for row in fixture.alignment_diagnostics)
        ),
    }
    payload["passed"] = bool(
        payload["active_one_body_hermiticity_residual"] <= 1.0e-10
        and payload["active_two_body_chemist_symmetry_residual"] <= 1.0e-8
        and payload["max_aligned_one_body_hermiticity_residual"] <= 1.0e-10
        and payload["max_aligned_two_body_chemist_symmetry_residual"] <= 1.0e-8
        and payload["all_alignment_records_passed"]
    )
    return payload


def _normal_mode_audit(fixture: ProductionVibronicH2OFixture) -> dict[str, Any]:
    vectors = np.asarray(
        [np.asarray(mode.mass_weighted_eigenvector, dtype=float).reshape(-1) for mode in fixture.normal_modes]
    )
    gram = vectors @ vectors.T
    orthonormality_residual = float(np.max(np.abs(gram - np.eye(len(vectors)))))
    diagnostics = fixture.report_summary.get("normal_mode_diagnostics", ())
    overlaps = [
        float(row["trans_rot_overlap"])
        for row in diagnostics
        if isinstance(row, Mapping) and row.get("trans_rot_overlap") is not None
    ]
    return {
        "passed": bool(orthonormality_residual <= 1.0e-10 and max(overlaps, default=0.0) <= 1.0e-3),
        "orthonormality_max_abs_residual": orthonormality_residual,
        "max_translation_rotation_overlap": float(max(overlaps, default=0.0)),
        "mode_labels": [str(mode.label) for mode in fixture.normal_modes],
        "frequencies_hartree": [float(mode.frequency_hartree) for mode in fixture.normal_modes],
        "frequencies_cm1": [
            None if mode.frequency_cm1 is None else float(mode.frequency_cm1)
            for mode in fixture.normal_modes
        ],
    }


def _electronic_reference_audit(
    fixture: ProductionVibronicH2OFixture,
    model: Any,
) -> dict[str, Any]:
    active = fixture.active_space
    layout = _electronic_layout(fixture)
    matrix, basis = _sector_sparse_matrix(
        model.h_electronic,
        layout=layout,
        n_spatial_orbitals=int(active.n_spatial_orbitals),
        num_particles=tuple(int(value) for value in active.num_particles),
        coeff_tol=0.0,
    )
    correlated_energy, correlated_state = _lowest_eigenpair(matrix)
    occupied = hartree_fock_occupied_qubits(
        int(active.n_spatial_orbitals),
        tuple(int(value) for value in active.num_particles),
        indexing="blocked",
    )
    hf_full_index = sum(1 << int(qubit) for qubit in occupied)
    hf_state = np.zeros(len(basis), dtype=complex)
    hf_state[basis.index(hf_full_index)] = 1.0
    hf_jw_energy = float(np.vdot(hf_state, matrix @ hf_state).real)
    hf_tensor_energy = _closed_shell_energy_from_tensors(
        active.scalar_energy_hartree,
        active.one_body_integrals,
        active.two_body_integrals,
        n_occupied=int(active.num_particles[0]),
    )

    center_gradient = fixture.report_summary.get("center_gradient_diagnostics", {})
    gradient = None
    if isinstance(center_gradient, Mapping) and center_gradient.get("available"):
        gradient = np.asarray(center_gradient.get("gradient_hartree_per_bohr"), dtype=float)
    mode_rows: list[dict[str, Any]] = []
    for mode, derivative, polynomial in zip(
        fixture.normal_modes,
        fixture.first_derivatives,
        model.dH_dQ_by_mode,
    ):
        derivative_matrix, derivative_basis = _sector_sparse_matrix(
            polynomial,
            layout=layout,
            n_spatial_orbitals=int(active.n_spatial_orbitals),
            num_particles=tuple(int(value) for value in active.num_particles),
            coeff_tol=0.0,
        )
        if derivative_basis != basis:
            raise ValueError("Electronic derivative sector basis differs from the electronic Hamiltonian basis.")
        hf_jw = float(np.vdot(hf_state, derivative_matrix @ hf_state).real)
        hf_tensor = _closed_shell_energy_from_tensors(
            derivative.scalar_derivative_hartree_per_q,
            derivative.one_body_derivative,
            derivative.two_body_derivative,
            n_occupied=int(active.num_particles[0]),
        )
        correlated = float(
            np.vdot(correlated_state, derivative_matrix @ correlated_state).real
        )
        projected_gradient = None
        if gradient is not None:
            projected_gradient = float(
                np.sum(
                    gradient
                    * np.asarray(mode.mass_weighted_eigenvector, dtype=float)
                    / np.sqrt(np.asarray(fixture.geometry.masses_me, dtype=float))[:, None]
                )
            )
        mode_rows.append(
            {
                "mode_label": str(mode.label),
                "rhf_derivative_from_jw_hartree_per_q": hf_jw,
                "rhf_derivative_from_tensors_hartree_per_q": hf_tensor,
                "psi4_gradient_projection_hartree_per_q": projected_gradient,
                "correlated_cas_ground_derivative_hartree_per_q": correlated,
                "jw_vs_tensor_abs_residual": float(abs(hf_jw - hf_tensor)),
                "jw_vs_psi4_gradient_abs_residual": (
                    None
                    if projected_gradient is None
                    else float(abs(hf_jw - projected_gradient))
                ),
            }
        )
    max_gradient_residual = max(
        (
            float(row["jw_vs_psi4_gradient_abs_residual"])
            for row in mode_rows
            if row["jw_vs_psi4_gradient_abs_residual"] is not None
        ),
        default=0.0,
    )
    max_correlated_force = max(
        abs(float(row["correlated_cas_ground_derivative_hartree_per_q"]))
        for row in mode_rows
    )
    return {
        "passed": bool(
            _relative_hermiticity_residual(matrix) <= 1.0e-10
            and abs(hf_jw_energy - hf_tensor_energy) <= 1.0e-10
            and max(float(row["jw_vs_tensor_abs_residual"]) for row in mode_rows) <= 1.0e-10
            and max_gradient_residual <= 1.0e-7
        ),
        "sector_dimension": int(matrix.shape[0]),
        "matrix_hermiticity_residual": _relative_hermiticity_residual(matrix),
        "rhf_energy_from_jw_hartree": hf_jw_energy,
        "rhf_energy_from_tensors_hartree": hf_tensor_energy,
        "rhf_energy_mapping_abs_residual": float(abs(hf_jw_energy - hf_tensor_energy)),
        "correlated_cas_electronic_ground_energy_hartree": correlated_energy,
        "mode_derivatives": mode_rows,
        "max_rhf_vs_psi4_gradient_abs_residual_hartree_per_q": max_gradient_residual,
        "reference_surface": "RHF_optimized_geometry_and_RHF_orbital_frame",
        "correlated_cas_surface_stationary_at_reference": bool(max_correlated_force <= 1.0e-5),
        "max_correlated_cas_ground_derivative_abs_hartree_per_q": float(max_correlated_force),
    }


def _sector_summary(audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: audit.get(key)
        for key in (
            "checked",
            "passed",
            "execution_checked",
            "execution_passed",
            "generator_count",
            "grouped_violation_count",
            "grouped_violation_labels",
            "execution_violation_count",
            "execution_violation_labels",
            "logical_shared_required_count",
            "requires_logical_shared_parameterization",
            "tolerance",
        )
    }


def _vibronic_reference_audit(
    fixture: ProductionVibronicH2OFixture,
    model: Any,
    resolved_problem: Any,
) -> dict[str, Any]:
    sector_matrix, _basis = _sector_sparse_matrix(
        model.h_vibronic,
        layout=fixture.layout,
        n_spatial_orbitals=int(fixture.active_space.n_spatial_orbitals),
        num_particles=tuple(int(value) for value in fixture.active_space.num_particles),
        coeff_tol=0.0,
    )
    exact_resolution = resolve_exact_reference_state_for_problem(
        model.h_vibronic,
        resolved_problem=resolved_problem,
    )
    if not exact_resolution.available or exact_resolution.state is None:
        raise ValueError(
            "Production H2O fixture exact state is unavailable: "
            f"{exact_resolution.skip_reason!r}."
        )
    state = np.asarray(exact_resolution.state, dtype=complex).reshape(-1)
    h_state = _apply_pauli_polynomial_uncached(state, model.h_vibronic)
    energy_complex = complex(np.vdot(state, h_state))
    declared_energy = float(fixture.exact_reference.ground_energy_hartree)
    residual = float(np.linalg.norm(h_state - declared_energy * state))
    state_audit = FixedCountSectorStateAuditor(resolved_problem).audit(
        state,
        source="fixture_exact_ground_state",
    )
    hamiltonian_sector = audit_candidate_pool_sector_contract(
        [
            AnsatzTerm(
                label="H2O linear-FD Hamiltonian",
                polynomial=model.h_vibronic,
                execution_mode="grouped_exact",
            )
        ],
        resolved_problem=resolved_problem,
        tolerance=1.0e-10,
    )
    pool_sector = audit_candidate_pool_sector_contract(
        model.pool,
        resolved_problem=resolved_problem,
        tolerance=1.0e-10,
    )
    return {
        "passed": bool(
            _relative_hermiticity_residual(sector_matrix) <= 1.0e-10
            and abs(float(energy_complex.real) - declared_energy) <= 1.0e-10
            and abs(float(energy_complex.imag)) <= 1.0e-10
            and residual <= 1.0e-9
            and bool(state_audit["passed"])
            and bool(hamiltonian_sector["passed"])
            and bool(pool_sector["passed"])
            and bool(pool_sector["execution_passed"])
        ),
        "sector_dimension": int(sector_matrix.shape[0]),
        "full_register_dimension": int(state.size),
        "matrix_hermiticity_residual": _relative_hermiticity_residual(sector_matrix),
        "declared_ground_energy_hartree": declared_energy,
        "state_expectation_energy_hartree": float(energy_complex.real),
        "state_expectation_imag_abs": float(abs(energy_complex.imag)),
        "eigenstate_residual_l2": residual,
        "state_sector_contract": state_audit,
        "hamiltonian_sector_contract": _sector_summary(hamiltonian_sector),
        "pool_sector_contract": _sector_summary(pool_sector),
    }


def _polynomial_coefficients(polynomial: Any) -> dict[str, complex]:
    coefficients: dict[str, complex] = {}
    for term in polynomial.return_polynomial():
        word = str(term.pw2strng())
        coefficients[word] = coefficients.get(word, 0.0j) + complex(term.p_coeff)
    return coefficients


def _historical_comparison(
    historical_path: Path,
    *,
    current_fixture: ProductionVibronicH2OFixture,
    current_model: Any,
) -> dict[str, Any]:
    historical_fixture = load_production_vibronic_h2o_fixture(historical_path)
    historical_model = build_production_vibronic_h2o_linear_fd_runtime_model(
        historical_fixture,
        require_paper_iv_evidence=False,
    )
    current_coeffs = _polynomial_coefficients(current_model.h_vibronic)
    historical_coeffs = _polynomial_coefficients(historical_model.h_vibronic)
    words = set(current_coeffs) | set(historical_coeffs)
    coefficient_delta = {
        word: current_coeffs.get(word, 0.0j) - historical_coeffs.get(word, 0.0j)
        for word in words
    }
    old_energy = float(historical_fixture.exact_reference.ground_energy_hartree)
    new_energy = float(current_fixture.exact_reference.ground_energy_hartree)
    unsafe_modes = [
        str(row.get("label"))
        for row in historical_fixture.pool
        if row.get("generator_family") == "linear_vibronic_derivative_momentum"
        and str(row.get("execution_mode", "termwise_product")) != "grouped_exact"
    ]
    strict_contract_error = None
    try:
        validate_paper_iv_h2o_linear_fd_evidence_fixture(
            historical_fixture,
            require_exact_state=True,
            require_reference_cutoff=True,
            require_cutoff_converged=False,
        )
        strict_contract_passed = True
    except ValueError as exc:
        strict_contract_passed = False
        strict_contract_error = str(exc)
    return {
        "fixture_path": str(historical_path),
        "fixture_sha256": _sha256_file(historical_path),
        "hamiltonian_term_count": int(len(historical_coeffs)),
        "corrected_hamiltonian_term_count": int(len(current_coeffs)),
        "coefficient_delta_l1": float(sum(abs(value) for value in coefficient_delta.values())),
        "coefficient_delta_max_abs": float(max((abs(value) for value in coefficient_delta.values()), default=0.0)),
        "same_cutoff_ground_energy_hartree": old_energy,
        "corrected_same_cutoff_ground_energy_hartree": new_energy,
        "ground_energy_correction_hartree": float(new_energy - old_energy),
        "unsafe_derivative_momentum_execution_labels": unsafe_modes,
        "historical_fixture_satisfies_current_strict_contract": strict_contract_passed,
        "historical_fixture_strict_contract_error": strict_contract_error,
    }


def _checkpoint_summary(checkpoint_path: Path) -> dict[str, Any]:
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    adapt = payload.get("adapt_vqe", {})
    checkpoint = payload.get("checkpoint", {})
    operators = [str(value) for value in adapt.get("operators", ())]
    family_counts = Counter(
        "coupled_vibronic"
        if label.startswith("coupled::")
        else "boson_momentum"
        if label.startswith("boson::")
        else "electronic"
        if label.startswith("el::")
        else "other"
        for label in operators
    )
    failure_error = None
    stdout_path = checkpoint_path.with_name("stdout.log")
    if stdout_path.exists():
        for line in stdout_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("AI_LOG "):
                continue
            try:
                event = json.loads(line[len("AI_LOG ") :])
            except json.JSONDecodeError:
                continue
            if event.get("event") == "hardcoded_adapt_vqe_failed":
                failure_error = event.get("error")
    energy = adapt.get("energy")
    exact = adapt.get("exact_gs_energy")
    return {
        "path": str(checkpoint_path),
        "sha256": _sha256_file(checkpoint_path),
        "success": bool(adapt.get("success", False)),
        "partial_checkpoint": bool(adapt.get("partial_checkpoint", False)),
        "checkpoint_complete": bool(checkpoint.get("complete", False)),
        "checkpoint_reason": checkpoint.get("reason"),
        "segment_depth": checkpoint.get("depth"),
        "total_ansatz_depth": adapt.get("ansatz_depth"),
        "energy_hartree": energy,
        "same_cutoff_reference_energy_hartree": exact,
        "delta_energy_hartree": (
            None if energy is None or exact is None else float(energy) - float(exact)
        ),
        "operator_count": int(len(operators)),
        "operator_family_counts": dict(family_counts),
        "contains_bosonic_or_coupled_generator": bool(
            family_counts.get("boson_momentum", 0) + family_counts.get("coupled_vibronic", 0)
        ),
        "operators": operators,
        "failure_error": failure_error,
        "reader_facing_status": "partial_failed_checkpoint_not_completed_result",
    }


def audit_h2o_linear_fd_fixture(
    fixture_path: str | Path,
    *,
    historical_fixture_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(fixture_path).resolve()
    cached = load_cached_production_vibronic_h2o_linear_fd_fixture(
        path,
        require_exact_state=True,
        require_reference_cutoff=True,
        require_cutoff_converged=False,
    )
    fixture = cached.fixture
    model = cached.model
    validate_paper_iv_h2o_linear_fd_evidence_fixture(
        fixture,
        require_exact_state=True,
        require_reference_cutoff=True,
        require_cutoff_converged=False,
    )
    request = ProblemRequest(
        problem_key="molecular_vibronic_h2o_linear_fd",
        num_sites=int(model.n_spatial_orbitals),
        t=1.0,
        u=0.0,
        dv=0.0,
        omega0=1.0,
        g_ep=1.0,
        n_ph_max=max(int(value) for value in model.mode_cutoffs),
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        include_zero_point=True,
        molecular_vibronic_h2o_linear_fd_fixture_json=str(path),
    )
    resolved_problem = resolve_problem_context(request)
    tensors = _tensor_audit(fixture)
    finite_difference = _finite_difference_audit(fixture)
    normal_modes = _normal_mode_audit(fixture)
    electronic = _electronic_reference_audit(fixture, model)
    vibronic = _vibronic_reference_audit(fixture, model, resolved_problem)
    cutoff = fixture.cutoff_diagnostics
    cutoff_payload = {
        "work_cutoffs": list(cutoff.work_cutoffs),
        "reference_cutoffs": (
            None if cutoff.reference_cutoffs is None else list(cutoff.reference_cutoffs)
        ),
        "work_ground_energy_hartree": cutoff.work_ground_energy_hartree,
        "reference_ground_energy_hartree": cutoff.reference_ground_energy_hartree,
        "delta_energy_hartree": cutoff.delta_energy_hartree,
        "total_boundary_weight": (
            None
            if cutoff.work_boundary_weight is None
            else float(cutoff.work_boundary_weight.total_boundary_weight)
        ),
        "energy_tolerance_hartree": cutoff.energy_tolerance_hartree,
        "boundary_weight_tolerance": cutoff.boundary_weight_tolerance,
        "energy_passed": cutoff.energy_passed,
        "boundary_passed": cutoff.boundary_passed,
        "vibrational_cutoff_converged": bool(cutoff.passed),
        "diagnostic_truthfully_encoded": True,
    }
    implementation_passed = bool(
        tensors["passed"]
        and finite_difference["passed"]
        and normal_modes["passed"]
        and electronic["passed"]
        and vibronic["passed"]
    )
    payload: dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "implementation_checks_passed": implementation_passed,
        "scientific_status": {
            "model_identity": "RHF-referenced STO-3G CAS(8e,6o) three-mode linear vibronic-coupling benchmark",
            "same_model_implementation_consistent": implementation_passed,
            "vibrational_cutoff_converged": bool(cutoff.passed),
            "correlated_reference_surface_stationary": bool(
                electronic["correlated_cas_surface_stationary_at_reference"]
            ),
            "fresh_independent_psi4_recomputation_performed": False,
            "fresh_psi4_unavailable_reason": (
                None
                if importlib.util.find_spec("psi4") is not None
                else "psi4_not_importable_in_audit_python_environment"
            ),
        },
        "input_manifest": {
            "fixture_path": str(path),
            "fixture_sha256": _sha256_file(path),
            "fixture_schema": fixture.manifest.schema,
            "generator_version": fixture.manifest.generator_version,
            "backend": dict(fixture.provenance.get("backend", {})),
            "backend_record_sha256": fixture.provenance.get("backend_record_sha256"),
        },
        "model_dimensions": {
            "basis": fixture.geometry.basis,
            "electronic_reference": fixture.geometry.reference,
            "active_space_kind": fixture.active_space.active_space_kind,
            "n_spatial_orbitals": int(fixture.active_space.n_spatial_orbitals),
            "num_particles": list(fixture.active_space.num_particles),
            "n_fermion_qubits": int(model.n_fermion_qubits),
            "n_boson_qubits": int(model.n_boson_qubits),
            "n_total_qubits": int(model.n_total_qubits),
            "hamiltonian_pauli_term_count": int(len(model.h_vibronic.return_polynomial())),
            "generator_count": int(len(model.pool)),
            "generator_execution_mode_counts": dict(
                Counter(str(term.execution_mode) for term in model.pool)
            ),
        },
        "tensor_checks": tensors,
        "finite_difference_checks": finite_difference,
        "normal_mode_checks": normal_modes,
        "electronic_reference_checks": electronic,
        "vibronic_reference_checks": vibronic,
        "cutoff_diagnostics": cutoff_payload,
        "model_caveats": [
            "The Hamiltonian is linear in mass-weighted normal coordinates.",
            "The reference geometry and orbital frame are RHF/STO-3G, not a correlated CAS-optimized surface.",
            "The same-model (1,1,1) cutoff fails the configured energy and boundary-weight convergence checks.",
            "The exact energy is exact only for this active-space, linearized, encoded, and truncated Hamiltonian.",
        ],
    }
    if historical_fixture_path is not None:
        payload["historical_fixture_comparison"] = _historical_comparison(
            Path(historical_fixture_path).resolve(),
            current_fixture=fixture,
            current_model=model,
        )
    if checkpoint_path is not None:
        payload["checkpoint"] = _checkpoint_summary(Path(checkpoint_path).resolve())
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--historical-fixture", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = audit_h2o_linear_fd_fixture(
        args.fixture,
        historical_fixture_path=args.historical_fixture,
        checkpoint_path=args.checkpoint,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output_json}")
    print(f"implementation_checks_passed={payload['implementation_checks_passed']}")
    return 0 if bool(payload["implementation_checks_passed"]) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
