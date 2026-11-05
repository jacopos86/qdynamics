"""A ``qdynamics_qubit_hamiltonian_v1`` file (Doron Wu's qubit Hamiltonian, or ours in his schema) through this
repository's solvers, with a report.

    PYTHONPATH=. python -m pipelines.chemistry_bridge.qubit_hamiltonian --hamiltonian h2o.json --out-dir out \
        [--methods exact,qse,apm] [--qse-basis uccsd_boson_ladders] [--apm-times 0,0.02,...] [--declared-origin simulated]

The file is the whole input: Pauli words over the fermion register (blocked spin order, qubits 0..2n-1) and one
boson register per mode above it (binary or unary), particle numbers and mode cutoffs. Nothing is rebuilt from
integrals and no fixture is needed. Every method starts from the Hartree-Fock determinant (x) boson vacuum.

exact  sparse physical-sector eigensolver (fixed (n_alpha, n_beta), n_I <= cutoff_I): low roots, the reference
       energy <HF x vac|H|HF x vac>, the ground state's boson occupations and reference weight, and a one-apply
       sector-leakage witness.
qse    the repository QSE core (pipelines.qse_spectra.core.compute_qse_spectra) with a mixed-register basis:
       identity, UCCSD generators lifted to the full register, boson raising operators and their products with
       the singles ("uccsd_boson_ladders"), or every UCCSD generator times every mode coordinate Q_I
       ("uccsd_times_displacements"). Ritz values are upper bounds; the exact roots are reported next to them.
apm    the repository AP-McLachlan exchange controller (pipelines.time_dynamics.ap_mclachlan) from HF (x) vacuum
       with lifted Hermitian excitations, the mode coordinates Q_I and Q_I times the singles as candidates, every
       candidate audited by the repository fixed-count sector contract; energy drift, sector leakage and the
       fidelity against exact sector propagation are diagnostics, never controller inputs.
RA-ADAPT is not here: the Paper IV lane runs it from a fixture (see agent_guidance/shared/paper-iv-worklog.md).

Exit codes: 0 every method verified; 1 a method failed or raised; 2 usage error or a refused input (no report).
The report ``<out-dir>/qubit_hamiltonian_report.json`` carries the file's sha256, the operator digest, the
repository commit, the environment, the declared origin and every method's requested and executed settings.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_SCHEMA = "qubit_hamiltonian_report_v1"
RECEIPT_SCHEMA = "qubit_hamiltonian_method_receipt_v1"
FAMILY_KEY = "molecular_vibronic_qdynamics"
METHODS = ("exact", "qse", "apm")
QSE_BASIS_KINDS = ("uccsd", "uccsd_boson_ladders", "uccsd_times_displacements")
APM_POOL_KINDS = ("hermitian_excitations_lifted", "hermitian_excitations_and_displacements")
ORIGINS = ("unknown", "synthetic", "simulated", "attested")
RUN_CLASSES = ("smoke", "diagnostic", "candidate", "paper_facing")
STATUS_VERIFIED, STATUS_FAILED, STATUS_ERROR = "verified", "failed", "error"


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, complex):
        return {"re": float(value.real), "im": float(value.imag)}
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass
class MethodReceipt:
    method: str
    lane: str
    route: str
    status: str
    requested: dict[str, Any]
    executed: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    checks: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    schema: str = RECEIPT_SCHEMA
    created_utc: str = field(default_factory=_now)

    def to_json_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


# ------------------------------------------------------------------ problem


@dataclass(frozen=True)
class QubitHamiltonianProblem:
    path: Path
    h: Any                       # QdynamicsQubitHamiltonian
    raw_sha256: str
    operator_sha256: str
    n_spatial: int
    nq: int
    particles: tuple[int, int]
    cutoffs: tuple[int, ...]
    context: Any                 # ResolvedProblemContext (sector constraints + layout for the audits)
    reference_index: int

    @property
    def n_boson_qubits(self) -> int:
        return int(self.nq) - 2 * int(self.n_spatial)

    def reference_state(self) -> np.ndarray:
        psi = np.zeros(1 << int(self.nq), dtype=complex)
        psi[int(self.reference_index)] = 1.0
        return psi

    def identity(self) -> dict[str, Any]:
        return {
            "path": str(self.path), "raw_sha256": self.raw_sha256, "operator_sha256": self.operator_sha256,
            "n_qubits": int(self.nq), "n_fermion_qubits": 2 * int(self.n_spatial), "n_spatial_orbitals": int(self.n_spatial),
            "num_particles": list(self.particles), "mode_cutoffs": list(self.cutoffs), "boson_encoding": str(self.h.encoding),
            "omega_hartree": [float(v) for v in self.h.frequencies_hartree], "coupling_model": self.h.coupling_model,
            "source_files": [str(s) for s in self.h.source_files], "schema": "qdynamics_qubit_hamiltonian_v1",
        }


def operator_digest(operator: Any) -> str:
    from src.quantum.chemistry.qdynamics_interop import pauli_term_table

    table = pauli_term_table(operator)
    payload = json.dumps([[w, table[w].real, table[w].imag] for w in sorted(table)], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _problem_context(h: Any, path: Path, *, exact_tol: float) -> Any:
    """A ResolvedProblemContext for the audits and the AP-McLachlan scaffold (sector, layout, reference, pools)."""
    from pipelines.contracts.problem import (
        ExactTargetSpec, FixedCountConstraint, ProblemRequest, ReferenceStateSpec, RegisterBlockSpec,
        RegisterLayoutSpec, ResolvedProblemContext, SectorSelection, TruncationConstraint,
    )
    from src.quantum.chemistry.qdynamics_interop import hartree_fock_vacuum_index, physical_sector_energies

    nf, n = int(h.n_fermion_qubits), int(h.n_qubits)
    n_spatial = nf // 2
    particles = (int(h.num_particles[0]), int(h.num_particles[1]))
    cutoffs = tuple(int(c) for c in h.mode_cutoffs)
    blocks = [RegisterBlockSpec(name="fermion", kind="fermion", start_qubit=0, stop_qubit=nf)]
    if n > nf:
        blocks.append(RegisterBlockSpec(name="boson", kind="boson", start_qubit=nf, stop_qubit=n))
    for k, block in enumerate(h.layout.boson_modes):
        blocks.append(RegisterBlockSpec(name=f"boson_mode_{k}", kind="boson_mode", start_qubit=int(block.qubit_start),
                                        stop_qubit=int(block.qubit_start) + int(block.n_qubits)))
    layout = RegisterLayoutSpec(total_qubits=n, fermion_qubits=nf, boson_qubits=n - nf, ordering="blocked",
                                boson_encoding=(str(h.encoding) if n > nf else None), blocks=tuple(blocks))
    request = ProblemRequest(problem_key=FAMILY_KEY, num_sites=n_spatial, t=0.0, u=0.0, dv=0.0, omega0=0.0, g_ep=0.0,
                             n_ph_max=(max(cutoffs) if cutoffs else 0), boson_encoding=str(h.encoding), ordering="blocked",
                             boundary="open", include_zero_point=True, molecular_problem_json=str(path))
    index = hartree_fock_vacuum_index(h.layout, n_spatial_orbitals=n_spatial, num_particles=particles)

    def _build_state() -> np.ndarray:
        psi = np.zeros(1 << n, dtype=complex)
        psi[index] = 1.0
        return psi

    cache: dict[str, float] = {}

    def _resolve_energy(**_: Any) -> float:
        if "ground" not in cache:
            cache["ground"] = float(physical_sector_energies(h, n_low=1, tol=exact_tol)[0])
        return cache["ground"]

    constraints: list[Any] = [FixedCountConstraint(quantity="n_up", value=particles[0], scope="fermion_register"),
                              FixedCountConstraint(quantity="n_dn", value=particles[1], scope="fermion_register")]
    constraints.extend(TruncationConstraint(quantity="vibrational_occupancy", max_local_occupancy=c, scope=f"boson_mode:{k}")
                       for k, c in enumerate(cutoffs))
    return ResolvedProblemContext(
        family_key=FAMILY_KEY, request=request, layout=layout, hamiltonian=h.operator,
        sector=SectorSelection(label="closed_shell_fermions_with_truncated_vibrations",
                               comparison_space_label="spin_orbital_register_with_truncated_vibrations",
                               constraints=tuple(constraints), num_particles=particles),
        reference_state=ReferenceStateSpec(kind="restricted_hf_times_boson_vacuum", source_label="qdynamics_json",
                                           state_kind="reference_state", build_state=_build_state),
        exact_target=ExactTargetSpec(kind="exact_ground_energy_physical_sector_sparse_eigsh",
                                     comparison_space_label="spin_orbital_register_with_truncated_vibrations",
                                     resolve_energy=_resolve_energy, exact_state_policy="sparse_sector_eigsh",
                                     build_fallback_anchor_state=_build_state, fallback_policy="reference_state_anchor"),
        default_controller_profile="phase3_v1", default_continuation_mode="phase3_v1",
        admissible_pool_keys=("full_meta", "full_hamiltonian"), default_pool_key="full_meta", default_pool_resolution_scope="problem",
        default_sector_label="closed_shell_fermions_with_truncated_vibrations", default_reference_label="restricted_hf_times_boson_vacuum",
        exact_target_label="exact_ground_energy_physical_sector_sparse_eigsh",
        exact_comparison_space_label="spin_orbital_register_with_truncated_vibrations",
        default_num_particles=particles,
        runtime_data={"qdynamics_hamiltonian": h, "qdynamics_path": str(path)},
    )


def load_problem(path: str | Path, *, exact_tol: float = 1.0e-10) -> QubitHamiltonianProblem:
    from src.quantum.chemistry.qdynamics_interop import hartree_fock_vacuum_index, load_qdynamics_qubit_hamiltonian

    path = Path(path)
    h = load_qdynamics_qubit_hamiltonian(path)
    nf = int(h.n_fermion_qubits)
    if nf % 2:
        raise ValueError(f"{nf} fermion qubits cannot be a blocked spin-orbital register")
    n_spatial = nf // 2
    particles = (int(h.num_particles[0]), int(h.num_particles[1]))
    if not (0 < particles[0] <= n_spatial and 0 < particles[1] <= n_spatial):
        raise ValueError(f"num_particles {particles} do not fit {n_spatial} spatial orbitals")
    if int(h.n_qubits) > 30:
        raise ValueError(f"{h.n_qubits} qubits: statevector methods here stop at 30")
    index = hartree_fock_vacuum_index(h.layout, n_spatial_orbitals=n_spatial, num_particles=particles)
    return QubitHamiltonianProblem(
        path=path, h=h, raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), operator_sha256=operator_digest(h.operator),
        n_spatial=n_spatial, nq=int(h.n_qubits), particles=particles, cutoffs=tuple(int(c) for c in h.mode_cutoffs),
        context=_problem_context(h, path, exact_tol=exact_tol), reference_index=int(index),
    )


# --------------------------------------------------------------- operators


def lift_fermion_polynomial(poly: Any, *, boson_qubits: int, tol: float = 1.0e-12) -> Any:
    """A fermion-register polynomial on the full register: identity on the boson qubits (the left end of the word)."""
    from src.quantum.pauli_polynomial_class import PauliPolynomial
    from src.quantum.qubitization_module import PauliTerm

    terms = list(poly.return_polynomial())
    if not terms:
        return PauliPolynomial("JW")
    nf = int(terms[0].nqubit())
    out = PauliPolynomial("JW")
    for term in terms:
        c = complex(term.p_coeff)
        if abs(c) <= tol:
            continue
        out.add_term(PauliTerm(nf + int(boson_qubits), ps=("e" * int(boson_qubits)) + str(term.pw2strng()), pc=c))
    out._reduce()
    return out


def _real_clean(poly: Any, *, nq: int, tol: float = 1.0e-12) -> Any:
    from src.quantum.pauli_polynomial_class import PauliPolynomial
    from src.quantum.qubitization_module import PauliTerm

    out = PauliPolynomial("JW")
    for term in poly.return_polynomial():
        c = complex(term.p_coeff)
        if abs(c.imag) > 1.0e-9 * max(1.0, abs(c)):
            raise ValueError(f"expected a real polynomial; {term.pw2strng()} has coefficient {c}")
        if abs(c.real) > tol:
            out.add_term(PauliTerm(int(nq), ps=str(term.pw2strng()), pc=complex(c.real, 0.0)))
    out._reduce()
    return out


def mode_ladders(problem: QubitHamiltonianProblem) -> list[tuple[int, Any, Any]]:
    """(mode index, b_I, b_I+) on the full register, truncated to the mode's cutoff as the file's assembly was."""
    from src.quantum.hubbard_latex_python_pairs import boson_operator

    out = []
    for k, block in enumerate(problem.h.layout.boson_modes):
        qubits = list(range(int(block.qubit_start), int(block.qubit_start) + int(block.n_qubits)))
        b = boson_operator("JW", int(problem.nq), qubits, which="b", n_ph_max=int(block.n_ph_max), encoding=str(block.encoding))
        bdag = boson_operator("JW", int(problem.nq), qubits, which="bdag", n_ph_max=int(block.n_ph_max), encoding=str(block.encoding))
        out.append((k, b, bdag))
    return out


def mode_coordinates(problem: QubitHamiltonianProblem) -> list[tuple[int, Any]]:
    """Q_I = (b_I + b_I+) / sqrt(2 omega_I) on the full register (Hermitian, real)."""
    out = []
    for k, b, bdag in mode_ladders(problem):
        omega = float(problem.h.frequencies_hartree[k])
        out.append((k, _real_clean((1.0 / np.sqrt(2.0 * omega)) * (b + bdag), nq=problem.nq)))
    return out


def lifted_uccsd_terms(problem: QubitHamiltonianProblem) -> list[Any]:
    from src.quantum.chemistry.molecular_uccsd import build_molecular_uccsd_pool
    from src.quantum.vqe_latex_python_pairs import AnsatzTerm

    pool = build_molecular_uccsd_pool(n_spatial_orbitals=problem.n_spatial, num_particles=problem.particles, ordering="blocked")
    return [AnsatzTerm(label=str(t.label), polynomial=lift_fermion_polynomial(t.polynomial, boson_qubits=problem.n_boson_qubits)) for t in pool]


def lifted_hermitian_terms(problem: QubitHamiltonianProblem) -> list[Any]:
    from pipelines.chemistry_bridge.molecular_pools import build_hermitian_excitation_pool
    from src.quantum.vqe_latex_python_pairs import AnsatzTerm

    pool = build_hermitian_excitation_pool(n_spatial_orbitals=problem.n_spatial, num_particles=problem.particles, ordering="blocked")
    return [AnsatzTerm(label=str(t.label), polynomial=lift_fermion_polynomial(t.polynomial, boson_qubits=problem.n_boson_qubits)) for t in pool]


def compiled_hamiltonian(problem: QubitHamiltonianProblem) -> Any:
    from src.quantum.compiled_polynomial import compile_polynomial_action

    return compile_polynomial_action(problem.h.operator)


def apply_hamiltonian(compiled: Any, psi: np.ndarray) -> np.ndarray:
    from src.quantum.compiled_polynomial import apply_compiled_polynomial

    return apply_compiled_polynomial(psi, compiled)


def sector_basis(problem: QubitHamiltonianProblem) -> np.ndarray:
    from src.quantum.chemistry.molecular_vibronic import physical_sector_basis

    return np.asarray(physical_sector_basis(problem.h.layout, n_spatial_orbitals=problem.n_spatial, num_particles=problem.particles), dtype=np.int64)


def out_of_sector_norm(psi: np.ndarray, basis: np.ndarray) -> float:
    vector = np.asarray(psi, dtype=complex).reshape(-1)
    inside = np.zeros(vector.shape, dtype=bool)
    inside[np.asarray(basis, dtype=np.int64)] = True
    return float(np.linalg.norm(vector[~inside]))


# ------------------------------------------------------------------- exact


def run_exact(problem: QubitHamiltonianProblem, *, n_low: int = 2, seed: int = 20261104) -> MethodReceipt:
    from src.quantum.chemistry.qdynamics_interop import ground_state_summary

    started = time.perf_counter()
    requested = {"n_low": int(n_low), "eigensolver": "scipy.sparse.linalg.eigsh on the repository sparse sector matrix"}
    summary = ground_state_summary(problem.h, n_low=int(n_low))
    basis = sector_basis(problem)
    rng = np.random.default_rng(int(seed))
    probe = np.zeros(1 << problem.nq, dtype=complex)
    probe[basis] = rng.normal(size=basis.size) + 1j * rng.normal(size=basis.size)
    probe /= np.linalg.norm(probe)
    leakage = out_of_sector_norm(apply_hamiltonian(compiled_hamiltonian(problem), probe), basis)
    psi_ref = problem.reference_state()
    hpsi = apply_hamiltonian(compiled_hamiltonian(problem), psi_ref)
    reference_energy_apply = float(np.real(np.vdot(psi_ref, hpsi)))
    checks = {
        "sector_leakage_one_apply": float(leakage),
        "reference_energy_apply_minus_diagonal_abs": abs(reference_energy_apply - float(summary["hartree_fock_vacuum_energy"])),
        "ground_at_or_below_reference": bool(summary["ground_energy"] <= summary["hartree_fock_vacuum_energy"] + 1.0e-12),
    }
    passed = leakage <= 1.0e-10 and checks["reference_energy_apply_minus_diagonal_abs"] <= 1.0e-10 and checks["ground_at_or_below_reference"]
    results = dict(summary)
    results["reference_energy_definition"] = "<HF x vacuum| H |HF x vacuum>: E_RHF + sum_I omega_I/2 for a mean-field vertex without counter-term; not a ground state"
    results["vibronic_lowering_below_reference"] = float(summary["hartree_fock_vacuum_energy"] - summary["ground_energy"])
    return MethodReceipt(method="exact", lane="shared_chemistry_reference", route="qdynamics_interop.ground_state_summary",
                         status=(STATUS_VERIFIED if passed else STATUS_FAILED), requested=requested, executed=requested,
                         results=results, checks=checks, elapsed_s=time.perf_counter() - started,
                         notes=["reporting input only; never an online controller input",
                                "boson occupations of the ground state say whether the cutoffs are converged"])


# --------------------------------------------------------------------- QSE


def qse_basis_elements(problem: QubitHamiltonianProblem, kind: str) -> tuple[list[Any], dict[str, int]]:
    from pipelines.qse_spectra.core import pauli_string_basis_element, polynomial_basis_element

    if kind not in QSE_BASIS_KINDS:
        raise ValueError(f"qse basis must be one of {QSE_BASIS_KINDS}")
    elements = [pauli_string_basis_element("e" * problem.nq, nq=problem.nq, name="identity")]
    counts = {"identity": 1, "dropped_zero_products": 0}

    def add(poly: Any, name: str, klass: str) -> None:
        # A product can vanish on a truncated register (b+ b+ at cutoff 1); such elements are dropped and counted.
        if not list(poly.return_polynomial()):
            counts["dropped_zero_products"] += 1
            return
        elements.append(polynomial_basis_element(poly, name=name))
        counts[klass] = counts.get(klass, 0) + 1

    uccsd = lifted_uccsd_terms(problem)
    for term in uccsd:
        add(term.polynomial, str(term.label), "uccsd_lifted")
    if kind == "uccsd_boson_ladders":
        ladders = mode_ladders(problem)
        singles = [t for t in uccsd if str(t.label).startswith("uccsd_sing(")]
        for k, _b, bdag in ladders:
            add(bdag, f"bdag({k})", "boson_raising")
        for k, _b, bdag in ladders:
            for t in singles:
                add(bdag * t.polynomial, f"bdag({k})*{t.label}", "boson_raising_times_singles")
        for i, (k, _b, bk) in enumerate(ladders):
            for (m, _b2, bm) in ladders[i:]:
                add(bk * bm, f"bdag({k})*bdag({m})", "boson_raising_pairs")
    elif kind == "uccsd_times_displacements":
        coordinates = mode_coordinates(problem)
        for k, q in coordinates:
            add(q, f"Q({k})", "mode_coordinates")
        for k, q in coordinates:
            for t in uccsd:
                add(q * t.polynomial, f"Q({k})*{t.label}", "mode_coordinates_times_uccsd")
    return elements, counts


def run_qse(problem: QubitHamiltonianProblem, *, basis_kind: str = "uccsd_boson_ladders", n_roots: int = 4,
            exact_roots: Sequence[float] | None = None) -> MethodReceipt:
    from pipelines.qse_spectra.core import QSEBasisVectorPolicy, compute_qse_spectra

    started = time.perf_counter()
    requested = {"basis_kind": str(basis_kind), "n_roots": int(n_roots), "reference_projection": "none", "basis_vector_normalization": "normalized"}
    elements, counts = qse_basis_elements(problem, str(basis_kind))
    policy = QSEBasisVectorPolicy(reference_projection="none", basis_vector_normalization="normalized")
    result = compute_qse_spectra(problem.h.operator, problem.reference_state(), elements, basis_vector_policy=policy)
    ritz = [float(v) for v in np.asarray(result.eigenvalues, dtype=float)[: int(n_roots)]]
    basis_dim = int(sector_basis(problem).size)
    results: dict[str, Any] = {
        "ritz_roots": ritz, "reference_energy": float(result.matrices.reference_energy), "solver_status": str(result.solver_status),
        "basis_size": int(len(elements)), "basis_counts": counts, "retained_rank": int(result.retained_rank),
        "discarded_rank": int(result.discarded_rank), "sector_dimension": basis_dim,
        "basis_spans_sector": bool(int(result.retained_rank) == basis_dim),
    }
    checks: dict[str, Any] = {
        "core_overlap_hermitian_residual_raw": float(result.matrices.overlap_hermitian_residual_max_abs_raw),
        "core_hamiltonian_hermitian_residual_raw": float(result.matrices.hamiltonian_hermitian_residual_max_abs_raw),
    }
    passed = str(result.solver_status).lower() in {"ok", "solved", "converged", "success"} or bool(ritz)
    if exact_roots is not None and ritz:
        exact = np.asarray(list(exact_roots), dtype=float)
        count = min(exact.size, len(ritz))
        results["exact_roots_compared"] = [float(v) for v in exact[:count]]
        checks["lowest_ritz_minus_exact_ground"] = float(ritz[0] - exact[0])
        checks["ritz_values_are_upper_bounds"] = bool(all(r >= e - 1.0e-8 for r, e in zip(sorted(ritz)[:count], sorted(exact)[:count])))
        passed = passed and checks["ritz_values_are_upper_bounds"]
    return MethodReceipt(method="qse", lane="paper_iii", route="pipelines.qse_spectra.core.compute_qse_spectra",
                         status=(STATUS_VERIFIED if passed else STATUS_FAILED), requested=requested,
                         executed={**requested, "basis_size": int(len(elements))}, results=results, checks=checks,
                         elapsed_s=time.perf_counter() - started,
                         notes=["Ritz values from the HF (x) vacuum are upper bounds; the gap to the exact ground state measures the basis, not the file",
                                "the basis is a bridge choice: UCCSD generators lifted to the full register plus boson operators; it is not Doron data"])


# ------------------------------------------------------------- AP-McLachlan


def apm_candidate_pool(problem: QubitHamiltonianProblem, kind: str) -> tuple[list[Any], dict[str, int]]:
    from src.quantum.vqe_latex_python_pairs import AnsatzTerm

    if kind not in APM_POOL_KINDS:
        raise ValueError(f"apm pool must be one of {APM_POOL_KINDS}")
    pool = lifted_hermitian_terms(problem)
    counts = {"hermitian_excitations_lifted": len(pool)}
    if kind == "hermitian_excitations_and_displacements":
        coordinates = mode_coordinates(problem)
        singles = [t for t in pool if str(t.label).startswith("herm_sing(")]
        for k, q in coordinates:
            pool.append(AnsatzTerm(label=f"Q({k})", polynomial=q))
        for k, q in coordinates:
            for t in singles:
                product = _real_clean(q * t.polynomial, nq=problem.nq)
                if list(product.return_polynomial()):
                    pool.append(AnsatzTerm(label=f"Q({k})*{t.label}", polynomial=product))
        counts["mode_coordinates"] = len(coordinates)
        counts["mode_coordinates_times_singles"] = len(coordinates) * len(singles)
    return pool, counts


def _audit_candidates(problem: QubitHamiltonianProblem, pool: Sequence[Any], *, parameterization_mode: str) -> tuple[dict[str, Any], list[Any]]:
    from pipelines.static_adapt.sector_invariants import audit_generator_sector_contract, resolve_fixed_count_qubit_groups

    groups, unsupported = resolve_fixed_count_qubit_groups(problem.context)
    if unsupported or not groups:
        raise ValueError(f"fixed-count sector groups unavailable: unsupported={list(unsupported)}")
    criterion = "execution_preserves_fixed_counts" if parameterization_mode == "logical_shared" else "components_individually_preserve_fixed_counts"
    eligible, failing = [], {}
    for term in pool:
        row = audit_generator_sector_contract(term, groups=groups, total_qubits=problem.nq)
        klass = str(term.label).split("(")[0]
        if bool(row[criterion]):
            eligible.append(term)
        else:
            failing[klass] = failing.get(klass, 0) + 1
    summary = {"audit": "pipelines.static_adapt.sector_invariants.audit_generator_sector_contract", "criterion": criterion,
               "fixed_count_quantities": [str(g.quantity) for g in groups], "generator_count": int(len(pool)),
               "eligible_count": int(len(eligible)), "audit_failing_by_class": failing}
    return summary, eligible


class CompiledStaticHamiltonian:
    """``TimeDependentHamiltonian`` with the static operator applied through compiled Pauli actions (no dense matrix).

    Built lazily as a subclass so the AP-McLachlan package is imported only when the method runs."""

    @staticmethod
    def build(static_poly: Any, metadata: Mapping[str, Any]) -> Any:
        from dataclasses import dataclass as _dataclass

        from pipelines.time_dynamics.ap_mclachlan.hamiltonian import TimeDependentHamiltonian
        from src.quantum.compiled_polynomial import apply_compiled_polynomial, compile_polynomial_action

        @_dataclass(frozen=True)
        class _Compiled(TimeDependentHamiltonian):
            def apply_at(self, time: float, psi: np.ndarray) -> np.ndarray:  # noqa: ARG002 - static operator
                if self.drive_model is not None:
                    raise ValueError("CompiledStaticHamiltonian has no drive")
                compiled = self.__dict__.get("_compiled_static")
                if compiled is None:
                    compiled = compile_polynomial_action(self.static_poly)
                    object.__setattr__(self, "_compiled_static", compiled)
                return apply_compiled_polynomial(np.asarray(psi, dtype=complex).reshape(-1), compiled)

        return _Compiled(static_poly=static_poly, drive_model=None, metadata=dict(metadata))


def _exact_sector_propagation(problem: QubitHamiltonianProblem, basis: np.ndarray, psi0: np.ndarray, elapsed: Sequence[float]) -> list[np.ndarray]:
    from scipy.sparse.linalg import expm_multiply  # type: ignore

    from src.quantum.chemistry.molecular_vibronic import sector_sparse_matrix

    matrix = sector_sparse_matrix(problem.h.operator, basis).tocsc()
    start = np.asarray(psi0, dtype=complex)[basis]
    out = []
    for t in elapsed:
        sector = start if float(t) == 0.0 else expm_multiply(-1j * float(t) * matrix, start)
        full = np.zeros(1 << problem.nq, dtype=complex)
        full[basis] = sector
        out.append(full)
    return out


def run_apm(
    problem: QubitHamiltonianProblem,
    *,
    times: Sequence[float] = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10),
    pool_kind: str = "hermitian_excitations_and_displacements",
    parameterization_mode: str = "logical_shared",
    integrator_method: str = "rk4",
    support_config_overrides: Mapping[str, Any] | None = None,
    compare_exact: bool = True,
    exact_ground: float | None = None,
) -> MethodReceipt:
    from pipelines.contracts.scaffold import CandidatePoolSource, ScaffoldRuntimeInput
    from pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory import SupportPatchControllerConfig, run_append_mclachlan_trajectory
    from pipelines.time_dynamics.ap_mclachlan.fixed_step import SolveRepairConfig
    from pipelines.time_dynamics.ap_mclachlan.hamiltonian import assert_static_hamiltonian_parity
    from pipelines.time_dynamics.ap_mclachlan.inverse import McLachlanInversePolicy
    from pipelines.time_dynamics.ap_mclachlan.state import state_from_scaffold_runtime_input
    from src.quantum.ansatz_parameterization import build_parameter_layout
    from src.quantum.compiled_polynomial import compile_polynomial_action

    started = time.perf_counter()
    grid = [float(t) for t in times]
    if len(grid) < 2 or any(b <= a for a, b in zip(grid, grid[1:])):
        raise ValueError("apm times must be strictly increasing with at least two points")
    requested = {"times": grid, "pool_kind": str(pool_kind), "parameterization_mode": str(parameterization_mode),
                 "integrator_method": str(integrator_method), "support_config_overrides": dict(support_config_overrides or {}),
                 "compare_exact": bool(compare_exact)}
    for block in problem.h.layout.boson_modes:
        if str(block.encoding) == "binary" and (int(block.n_ph_max) + 1) != (1 << int(block.n_qubits)):
            raise ValueError(f"mode cutoff {block.n_ph_max} on {block.n_qubits} qubits leaves padded boson codes; "
                             "termwise boson generators leak there, so this runner refuses (use cutoff 2^k - 1)")
    pool, pool_counts = apm_candidate_pool(problem, str(pool_kind))
    audit, eligible = _audit_candidates(problem, pool, parameterization_mode=str(parameterization_mode))
    if not eligible:
        raise ValueError(f"no generator passes the sector audit: {audit}")
    psi_ref = problem.reference_state()
    compiled_h = compile_polynomial_action(problem.h.operator)
    hpsi = apply_hamiltonian(compiled_h, psi_ref)
    energy0 = float(np.real(np.vdot(psi_ref, hpsi)))
    residual = hpsi - energy0 * psi_ref
    velocities = []
    for term in eligible:
        gpsi = apply_hamiltonian(compile_polynomial_action(term.polynomial), psi_ref)
        velocities.append(float(abs(np.real(np.vdot(gpsi, residual)))))
    order = sorted(range(len(eligible)), key=lambda k: (-velocities[k], k))
    nonzero = int(sum(1 for v in velocities if v > 1.0e-12))
    seed = eligible[order[0]]
    seed_selection = "auto_max_initial_velocity_v1" if nonzero else "auto_first_eligible_all_initial_velocities_zero"
    candidates = tuple(eligible[k] for k in order if str(eligible[k].label) != str(seed.label))
    selected = (seed,)
    layout = build_parameter_layout(selected)
    basis = sector_basis(problem)
    if exact_ground is None:
        exact_ground = float(problem.context.exact_target.resolve_energy())
    runtime_input = ScaffoldRuntimeInput(
        resolved_problem=problem.context, psi_ref=psi_ref, psi_initial=psi_ref.copy(), base_layout=layout,
        theta_runtime=np.zeros(int(layout.runtime_parameter_count), dtype=float),
        theta_logical=np.zeros(int(layout.logical_parameter_count), dtype=float),
        structure_locked=False, exact_energy=float(exact_ground), selected_terms=selected, candidate_pool_terms=candidates,
        candidate_pool_source=CandidatePoolSource(source_kind="resolved_pool", pool_key=str(pool_kind), completeness="complete"),
        provenance={"schema": REPORT_SCHEMA, "input_raw_sha256": problem.raw_sha256, "input_operator_sha256": problem.operator_sha256, "seed": "hf_x_vacuum_theta_zero"},
    )
    state = state_from_scaffold_runtime_input(runtime_input, parameterization_mode=str(parameterization_mode))
    parity = assert_static_hamiltonian_parity(runtime_input, static_poly=problem.h.operator, atol=1.0e-10)
    hamiltonian = CompiledStaticHamiltonian.build(problem.h.operator, {"runtime_provenance": dict(runtime_input.provenance),
                                                                        "static_hamiltonian_parity": parity, "drive_source": "none",
                                                                        "action": "compiled_pauli_actions_no_dense_matrix"})
    config = SupportPatchControllerConfig(**dict(support_config_overrides or {}))
    trajectory = run_append_mclachlan_trajectory(
        state=state, hamiltonian=hamiltonian, times=grid, inverse_policy=McLachlanInversePolicy(),
        integrator_method=str(integrator_method), support_patch_config=config, solve_repair_config=SolveRepairConfig(),
        metadata={"entry": REPORT_SCHEMA},
    )
    points = list(trajectory.points)
    energies = [float(p.energy_expectation) for p in points]
    recorded = [float(p.time) for p in points]
    leak = [out_of_sector_norm(np.asarray(p.geometry.psi), basis) for p in points]
    supports = [int(p.runtime_parameter_count) for p in points]
    patch_kinds = [str(p.patch_decision.patch_kind) for p in points]
    theta_max = max((float(np.max(np.abs(np.asarray(p.theta_runtime, dtype=float)))) if np.asarray(p.theta_runtime).size else 0.0) for p in points) if points else 0.0
    checks: dict[str, Any] = {
        "energy_drift_max_abs": float(max(energies) - min(energies)) if energies else 0.0,
        "sector_leakage_max_norm": float(max(leak)) if leak else 0.0,
        "initial_energy_minus_reference_abs": abs(energies[0] - energy0) if energies else None,
        "point_count": int(len(points)),
    }
    results: dict[str, Any] = {
        "times": recorded, "energies": energies, "runtime_parameter_counts": supports, "patch_kinds": patch_kinds,
        "pool_counts": pool_counts, "sector_audit": audit, "seed_term": str(seed.label), "seed_selection": seed_selection,
        "seed_abs_initial_velocity": float(velocities[order[0]]), "initial_velocity_nonzero_count": nonzero,
        "initial_velocity_top": [{"label": str(eligible[k].label), "abs_initial_velocity": velocities[k]} for k in order[:8]],
        "candidate_count_offered": int(len(candidates)), "parameterization_mode": str(state.parameterization_mode),
        "integrator_method": str(trajectory.integrator_method), "reference_energy": energy0, "exact_ground_energy": float(exact_ground),
        "final_runtime_parameter_count": int(trajectory.final_state.runtime_parameter_count),
        "trajectory_moved": bool(theta_max > 0.0 or any(k != "no_edit" for k in patch_kinds)), "theta_max_abs": float(theta_max),
        "support_patch_config": _json_safe(config.to_json_dict()) if hasattr(config, "to_json_dict") else {},
    }
    if compare_exact and points:
        elapsed = [t - recorded[0] for t in recorded]
        exact_states = _exact_sector_propagation(problem, basis, psi_ref, elapsed)
        fidelities = [float(abs(np.vdot(e, np.asarray(p.geometry.psi, dtype=complex).reshape(-1))) ** 2) for e, p in zip(exact_states, points)]
        static = [float(abs(np.vdot(e, psi_ref)) ** 2) for e in exact_states]
        results["fidelity_vs_exact_sector_propagation"] = fidelities
        results["fidelity_static_reference_vs_exact"] = static
        checks["min_fidelity_vs_exact"] = float(min(fidelities))
        checks["min_fidelity_static_reference"] = float(min(static))
    passed = checks["sector_leakage_max_norm"] <= 1.0e-8 and (checks["initial_energy_minus_reference_abs"] or 0.0) <= 1.0e-10
    return MethodReceipt(method="apm", lane="paper_ii", route="pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory.run_append_mclachlan_trajectory",
                         status=(STATUS_VERIFIED if passed else STATUS_FAILED), requested=requested,
                         executed={**requested, "pool_size": int(len(pool)), "eligible_count": int(len(eligible))},
                         results=results, checks=checks, elapsed_s=time.perf_counter() - started,
                         notes=["CPU, statevector: the Hamiltonian acts through compiled Pauli actions; no dense matrix is formed",
                                "fidelity against exact sector propagation is a diagnostic of the variational trajectory, not a certification",
                                "the pool (lifted Hermitian excitations, mode coordinates Q_I, Q_I times singles) is a bridge choice, not Doron data"])


# ------------------------------------------------------------------ report


def repository_record() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30, check=False).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    status = git("status", "--porcelain")
    return {"root": str(REPO_ROOT), "commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_working_tree": bool(status)}


def environment_record() -> dict[str, Any]:
    out: dict[str, Any] = {"python": sys.version.split()[0], "executable": sys.executable, "platform": platform.platform(),
                           "machine": platform.machine(), "hostname": platform.node()}
    for name in ("numpy", "scipy", "pyscf", "h5py", "torch"):
        try:
            module = __import__(name)
            out[name] = str(getattr(module, "__version__", "?"))
        except Exception:  # noqa: BLE001 - optional packages
            out[name] = None
    return out


def write_report(out_dir: Path, problem: QubitHamiltonianProblem, receipts: Sequence[MethodReceipt], *, settings: Mapping[str, Any],
                 started_utc: str, elapsed_s: float) -> tuple[Path, int]:
    statuses = {r.method: r.status for r in receipts}
    exit_code = 0 if all(s == STATUS_VERIFIED for s in statuses.values()) else 1
    report = {
        "schema": REPORT_SCHEMA, "created_utc": _now(), "started_utc": started_utc, "elapsed_s": float(elapsed_s),
        "inputs": problem.identity(), "settings": dict(settings), "repository": repository_record(), "environment": environment_record(),
        "origin": {"declared_origin": settings.get("declared_origin"), "claimed_producer": settings.get("claimed_producer"),
                   "producer_origin_established": settings.get("declared_origin") == "attested",
                   "schema_reference_producer": "github.com/jacopos86/qdynamics doron-chemistry src/chemistry_qubitization at 38f40805"},
        "run_class": settings.get("run_class"),
        "methods": [r.to_json_dict() for r in receipts], "status_by_method": statuses,
        "exit_code": exit_code, "exit_policy": "0 every method verified; 1 a method failed or raised; 2 refused input (no report)",
        "exact_energies_used_online": False,
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "qubit_hamiltonian_report.json"
    path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path, exit_code


def run_methods(problem: QubitHamiltonianProblem, methods: Sequence[str], *, options: Mapping[str, Any]) -> list[MethodReceipt]:
    receipts: list[MethodReceipt] = []
    exact_roots: list[float] | None = None
    for method in methods:
        started = time.perf_counter()
        try:
            if method == "exact":
                receipt = run_exact(problem, n_low=int(options.get("exact_roots", 2)))
                exact_roots = list(receipt.results["sector_energies"])
            elif method == "qse":
                receipt = run_qse(problem, basis_kind=str(options.get("qse_basis", "uccsd_boson_ladders")), n_roots=int(options.get("qse_roots", 4)), exact_roots=exact_roots)
            elif method == "apm":
                receipt = run_apm(problem, times=options.get("apm_times", (0.0, 0.02, 0.04, 0.06, 0.08, 0.10)), pool_kind=str(options.get("apm_pool", "hermitian_excitations_and_displacements")),
                                  parameterization_mode=str(options.get("apm_parameterization", "logical_shared")),
                                  support_config_overrides=options.get("apm_support_config") or {}, compare_exact=bool(options.get("apm_compare_exact", True)),
                                  exact_ground=(exact_roots[0] if exact_roots else None))
            else:
                raise ValueError(f"unknown method {method!r}")
        except Exception as exc:  # noqa: BLE001 - the report records the failure
            receipt = MethodReceipt(method=str(method), lane="", route="", status=STATUS_ERROR, requested=dict(options),
                                    checks={"exception": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=12)},
                                    elapsed_s=time.perf_counter() - started)
        receipts.append(receipt)
        print(json.dumps({"method": receipt.method, "status": receipt.status, "elapsed_s": round(receipt.elapsed_s, 2),
                          **{k: v for k, v in receipt.results.items() if k in ("ground_energy", "hartree_fock_vacuum_energy", "ritz_roots", "energies", "runtime_parameter_counts")}},
                         default=str), flush=True)
    return receipts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m pipelines.chemistry_bridge.qubit_hamiltonian", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hamiltonian", required=True, help="qdynamics_qubit_hamiltonian_v1 JSON")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--methods", default="exact,qse,apm", help=f"comma-separated subset of {','.join(METHODS)}, run in this order")
    parser.add_argument("--exact-roots", type=int, default=2)
    parser.add_argument("--qse-basis", default="uccsd_boson_ladders", choices=QSE_BASIS_KINDS)
    parser.add_argument("--qse-roots", type=int, default=4)
    parser.add_argument("--apm-times", default="0,0.02,0.04,0.06,0.08,0.10")
    parser.add_argument("--apm-pool", default="hermitian_excitations_and_displacements", choices=APM_POOL_KINDS)
    parser.add_argument("--apm-parameterization", default="logical_shared", choices=("logical_shared", "per_pauli_term"))
    parser.add_argument("--apm-support-config", default="{}", help="JSON object of SupportPatchControllerConfig overrides")
    parser.add_argument("--no-apm-exact-comparison", action="store_true")
    parser.add_argument("--declared-origin", default="unknown", choices=ORIGINS, help="what you declare about the file; never inferred from its contents")
    parser.add_argument("--claimed-producer", default=None)
    parser.add_argument("--run-class", default="diagnostic", choices=RUN_CLASSES)
    parser.add_argument("--pauli-cache-nq", type=int, default=None, help="cache compiled Pauli permutation tables up to this register width (memory: 1024 x 9 x 2^nq bytes)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.pauli_cache_nq is not None:
        os.environ["STATIC_ADAPT_PAULI_SIGN_CACHE_MAX_NQ"] = str(int(args.pauli_cache_nq))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    methods = tuple(m.strip() for m in str(args.methods).split(",") if m.strip())
    if not methods or any(m not in METHODS for m in methods):
        print(f"--methods must be a non-empty subset of {METHODS}", file=sys.stderr)
        return 2
    try:
        times = tuple(float(t) for t in str(args.apm_times).split(",") if t.strip())
        support = json.loads(str(args.apm_support_config))
        if not isinstance(support, dict):
            raise ValueError("--apm-support-config must be a JSON object")
    except ValueError as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        return 2
    started_utc, started = _now(), time.perf_counter()
    try:
        problem = load_problem(args.hamiltonian)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"refused input: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    options = {"exact_roots": args.exact_roots, "qse_basis": args.qse_basis, "qse_roots": args.qse_roots, "apm_times": times,
               "apm_pool": args.apm_pool, "apm_parameterization": args.apm_parameterization, "apm_support_config": support,
               "apm_compare_exact": not args.no_apm_exact_comparison, "declared_origin": args.declared_origin,
               "claimed_producer": args.claimed_producer, "run_class": args.run_class, "methods": list(methods),
               "pauli_cache_nq": args.pauli_cache_nq}
    print(json.dumps({"input": problem.identity()}, default=str), flush=True)
    receipts = run_methods(problem, methods, options=options)
    path, exit_code = write_report(Path(args.out_dir), problem, receipts, settings=options, started_utc=started_utc, elapsed_s=time.perf_counter() - started)
    print(json.dumps({"report": str(path), "status_by_method": {r.method: r.status for r in receipts}, "exit_code": exit_code}), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
