#!/usr/bin/env python3
"""Qiskit baseline end-to-end Hubbard pipeline.

Flow:
1) Build Hubbard Hamiltonian with Qiskit Nature + JW mapping.
2) Run Qiskit VQE.
3) Run Qiskit QPE (with robust fallback).
4) Run Suzuki-2 Trotter dynamics + exact dynamics (statevector math using Qiskit-derived terms).
5) Emit JSON + compact PDF artifact in schema aligned with hardcoded pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import SuzukiTrotter
from qiskit_algorithms import PhaseEstimation
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver, VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.hamiltonians.lattices import BoundaryCondition, LineLattice
from qiskit_nature.second_q.mappers import JordanWignerMapper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydephasing.quantum.hartree_fock_reference_state import hartree_fock_statevector

EXACT_LABEL = "Exact_Qiskit"
EXACT_METHOD = "python_matrix_eigendecomposition"


def _ai_log(event: str, **fields: Any) -> None:
    payload = {
        "event": str(event),
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    print(f"AI_LOG {json.dumps(payload, sort_keys=True, default=str)}", flush=True)


@dataclass(frozen=True)
class CompiledPauliAction:
    label_exyz: str
    perm: np.ndarray
    phase: np.ndarray


def _to_exyz(label_ixyz: str) -> str:
    return str(label_ixyz).lower().replace("i", "e")


def _to_ixyz(label_exyz: str) -> str:
    return str(label_exyz).replace("e", "I").upper()


def _normalize_state(psi: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(psi))
    if nrm <= 0.0:
        raise ValueError("Encountered zero-norm state.")
    return psi / nrm


def _half_filled_particles(num_sites: int) -> tuple[int, int]:
    return ((num_sites + 1) // 2, num_sites // 2)


def _interleaved_to_blocked_permutation(n_sites: int) -> list[int]:
    return [idx for site in range(n_sites) for idx in (site, n_sites + site)]


def _spin_orbital_index_sets(num_sites: int, ordering: str) -> tuple[list[int], list[int]]:
    normalized_ordering = ordering.strip().lower()
    if normalized_ordering == "blocked":
        return list(range(num_sites)), list(range(num_sites, 2 * num_sites))
    if normalized_ordering == "interleaved":
        return list(range(0, 2 * num_sites, 2)), list(range(1, 2 * num_sites, 2))
    raise ValueError(f"Unsupported ordering '{ordering}'.")


def _number_operator_qop(num_qubits: int, indices: list[int]) -> SparsePauliOp:
    coeffs: dict[str, complex] = {}
    id_label = "I" * num_qubits
    coeffs[id_label] = coeffs.get(id_label, 0.0 + 0.0j) + complex(0.5 * len(indices))
    for q in indices:
        chars = ["I"] * num_qubits
        chars[num_qubits - 1 - q] = "Z"
        lbl = "".join(chars)
        coeffs[lbl] = coeffs.get(lbl, 0.0 + 0.0j) + complex(-0.5)
    return SparsePauliOp.from_list([(k, v) for k, v in coeffs.items()]).simplify(atol=1e-12)


def _filtered_exact_energy(
    qop: SparsePauliOp,
    *,
    num_sites: int,
    ordering: str,
    num_particles: tuple[int, int],
) -> float:
    alpha_idx, beta_idx = _spin_orbital_index_sets(int(num_sites), ordering)
    n_alpha_op = _number_operator_qop(qop.num_qubits, alpha_idx)
    n_beta_op = _number_operator_qop(qop.num_qubits, beta_idx)

    def _filter(_state, _energy, aux_values):
        n_alpha = float(np.real(aux_values["N_alpha"][0]))
        n_beta = float(np.real(aux_values["N_beta"][0]))
        return np.isclose(n_alpha, num_particles[0]) and np.isclose(n_beta, num_particles[1])

    solver = NumPyMinimumEigensolver(filter_criterion=_filter)
    res = solver.compute_minimum_eigenvalue(
        qop,
        aux_operators={"N_alpha": n_alpha_op, "N_beta": n_beta_op},
    )
    return float(np.real(res.eigenvalue))


def _uniform_potential_qubit_op(num_sites: int, dv: float) -> SparsePauliOp:
    nq = 2 * int(num_sites)
    coeffs: dict[str, complex] = {}

    id_label = "I" * nq
    coeffs[id_label] = coeffs.get(id_label, 0.0 + 0.0j) + complex(-0.5 * dv * nq)

    for p in range(nq):
        chars = ["I"] * nq
        chars[nq - 1 - p] = "Z"
        z_label = "".join(chars)
        coeffs[z_label] = coeffs.get(z_label, 0.0 + 0.0j) + complex(0.5 * dv)

    return SparsePauliOp.from_list([(lbl, coeff) for lbl, coeff in coeffs.items()]).simplify(atol=1e-12)


def _build_qiskit_qubit_hamiltonian(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    boundary: str,
    ordering: str,
) -> tuple[SparsePauliOp, Any]:
    boundary_enum = BoundaryCondition.PERIODIC if boundary.strip().lower() == "periodic" else BoundaryCondition.OPEN
    lattice = LineLattice(
        num_nodes=int(num_sites),
        edge_parameter=-float(t),
        onsite_parameter=0.0,
        boundary_condition=boundary_enum,
    )
    ferm_op = FermiHubbardModel(lattice=lattice, onsite_interaction=float(u)).second_q_op()

    if ordering.strip().lower() == "blocked":
        ferm_op = ferm_op.permute_indices(_interleaved_to_blocked_permutation(int(num_sites)))

    mapper = JordanWignerMapper()
    qop = mapper.map(ferm_op).simplify(atol=1e-12)

    if abs(float(dv)) > 1e-15:
        qop = (qop + _uniform_potential_qubit_op(int(num_sites), float(dv))).simplify(atol=1e-12)

    return qop, mapper


def _qiskit_terms_exyz(qop: SparsePauliOp, tol: float = 1e-12) -> tuple[list[str], dict[str, complex]]:
    order: list[str] = []
    coeff_map: dict[str, complex] = {}
    for label_ixyz, coeff in qop.to_list():
        coeff_c = complex(coeff)
        if abs(coeff_c) <= tol:
            continue
        lbl = _to_exyz(str(label_ixyz))
        if lbl not in coeff_map:
            order.append(lbl)
            coeff_map[lbl] = 0.0 + 0.0j
        coeff_map[lbl] += coeff_c

    cleaned_order = [lbl for lbl in order if abs(coeff_map[lbl]) > tol]
    cleaned_map = {lbl: coeff_map[lbl] for lbl in cleaned_order}
    return cleaned_order, cleaned_map


def _ordered_qop_from_exyz(
    ordered_labels_exyz: list[str],
    coeff_map_exyz: dict[str, complex],
    *,
    tol: float = 1e-12,
) -> SparsePauliOp:
    terms: list[tuple[str, complex]] = []
    for lbl in ordered_labels_exyz:
        coeff = complex(coeff_map_exyz[lbl])
        if abs(coeff) <= tol:
            continue
        terms.append((_to_ixyz(lbl), coeff))
    if not terms:
        nq = len(ordered_labels_exyz[0]) if ordered_labels_exyz else 1
        terms = [("I" * nq, 0.0 + 0.0j)]
    return SparsePauliOp.from_list(terms)


def _compile_pauli_action(label_exyz: str, nq: int) -> CompiledPauliAction:
    dim = 1 << nq
    idx = np.arange(dim, dtype=np.int64)
    perm = idx.copy()
    phase = np.ones(dim, dtype=complex)

    for q in range(nq):
        op = label_exyz[nq - 1 - q]
        bits = ((idx >> q) & 1).astype(np.int8)
        sign = (1 - 2 * bits).astype(np.int8)

        if op == "e":
            continue
        if op == "x":
            perm ^= (1 << q)
            continue
        if op == "y":
            perm ^= (1 << q)
            phase *= 1j * sign
            continue
        if op == "z":
            phase *= sign
            continue
        raise ValueError(f"Unsupported Pauli symbol '{op}' in '{label_exyz}'.")

    return CompiledPauliAction(label_exyz=label_exyz, perm=perm, phase=phase)


def _apply_compiled_pauli(psi: np.ndarray, action: CompiledPauliAction) -> np.ndarray:
    out = np.empty_like(psi)
    out[action.perm] = action.phase * psi
    return out


def _apply_exp_term(
    psi: np.ndarray,
    action: CompiledPauliAction,
    coeff: complex,
    alpha: float,
    tol: float = 1e-12,
) -> np.ndarray:
    if abs(coeff.imag) > tol:
        raise ValueError(f"Imaginary coefficient encountered for {action.label_exyz}: {coeff}")
    theta = float(alpha) * float(coeff.real)
    ppsi = _apply_compiled_pauli(psi, action)
    return math.cos(theta) * psi - 1j * math.sin(theta) * ppsi


def _evolve_trotter_suzuki2_absolute(
    psi0: np.ndarray,
    ordered_labels_exyz: list[str],
    coeff_map_exyz: dict[str, complex],
    compiled_actions: dict[str, CompiledPauliAction],
    time_value: float,
    trotter_steps: int,
) -> np.ndarray:
    psi = np.array(psi0, copy=True)
    if abs(time_value) <= 1e-15:
        return psi
    dt = float(time_value) / float(trotter_steps)
    half = 0.5 * dt
    for _ in range(trotter_steps):
        for label in ordered_labels_exyz:
            psi = _apply_exp_term(psi, compiled_actions[label], coeff_map_exyz[label], half)
        for label in reversed(ordered_labels_exyz):
            psi = _apply_exp_term(psi, compiled_actions[label], coeff_map_exyz[label], half)
    return _normalize_state(psi)


def _expectation_hamiltonian(psi: np.ndarray, hmat: np.ndarray) -> float:
    return float(np.real(np.vdot(psi, hmat @ psi)))


def _occupation_site0(psi: np.ndarray, num_sites: int) -> tuple[float, float]:
    probs = np.abs(psi) ** 2
    n_up = 0.0
    n_dn = 0.0
    for idx, prob in enumerate(probs):
        n_up += float((idx >> 0) & 1) * float(prob)
        n_dn += float((idx >> num_sites) & 1) * float(prob)
    return float(n_up), float(n_dn)


def _doublon_total(psi: np.ndarray, num_sites: int) -> float:
    probs = np.abs(psi) ** 2
    out = 0.0
    for idx, prob in enumerate(probs):
        count = 0
        for site in range(num_sites):
            up = (idx >> site) & 1
            dn = (idx >> (num_sites + site)) & 1
            count += int(up * dn)
        out += float(count) * float(prob)
    return float(out)


def _state_to_amplitudes_qn_to_q0(psi: np.ndarray, cutoff: float = 1e-12) -> dict[str, dict[str, float]]:
    nq = int(round(math.log2(psi.size)))
    out: dict[str, dict[str, float]] = {}
    for idx, amp in enumerate(psi):
        if abs(amp) < cutoff:
            continue
        bit = format(idx, f"0{nq}b")
        out[bit] = {"re": float(np.real(amp)), "im": float(np.imag(amp))}
    return out


def _run_qiskit_vqe(
    *,
    num_sites: int,
    qop: SparsePauliOp,
    mapper: Any,
    hopping_t: float,
    onsite_u: float,
    dv: float,
    boundary: str,
    ordering: str,
    reps: int,
    restarts: int,
    seed: int,
    maxiter: int,
) -> tuple[dict[str, Any], np.ndarray | None]:
    t0 = time.perf_counter()
    _ai_log(
        "qiskit_vqe_start",
        L=int(num_sites),
        reps=int(reps),
        restarts=int(restarts),
        maxiter=int(maxiter),
        seed=int(seed),
    )

    def _finish(payload: dict[str, Any], psi: np.ndarray | None) -> tuple[dict[str, Any], np.ndarray | None]:
        _ai_log(
            "qiskit_vqe_done",
            L=int(num_sites),
            method=str(payload.get("method", "")),
            success=bool(payload.get("success", False)),
            energy=payload.get("energy"),
            best_restart=payload.get("best_restart"),
            elapsed_sec=round(time.perf_counter() - t0, 6),
        )
        return payload, psi

    num_particles = _half_filled_particles(int(num_sites))
    estimator = None

    try:
        from qiskit.primitives import StatevectorEstimator

        estimator = StatevectorEstimator()
    except Exception:
        estimator = None

    if estimator is None:
        np_solver = NumPyMinimumEigensolver()
        eig = np_solver.compute_minimum_eigenvalue(qop)
        exact_filtered = None
        try:
            exact_filtered = _filtered_exact_energy(
                qop,
                num_sites=int(num_sites),
                ordering=str(ordering),
                num_particles=num_particles,
            )
        except Exception:
            exact_filtered = None
        return _finish(
            {
                "success": True,
                "method": "qiskit_numpy_minimum_eigensolver_fallback",
                "energy": float(np.real(eig.eigenvalue)),
                "exact_filtered_energy": exact_filtered,
                "num_particles": {"n_up": int(num_particles[0]), "n_dn": int(num_particles[1])},
            },
            None,
        )

    try:
        hf = HartreeFock(
            num_spatial_orbitals=int(num_sites),
            num_particles=tuple(num_particles),
            qubit_mapper=mapper,
        )
        ansatz = UCCSD(
            num_spatial_orbitals=int(num_sites),
            num_particles=tuple(num_particles),
            qubit_mapper=mapper,
            initial_state=hf,
            reps=int(reps),
        )
        effective_maxiter = max(1, int(maxiter))
        rng = np.random.default_rng(int(seed))
        best_energy = float("inf")
        best_restart = -1
        best_point = np.zeros(ansatz.num_parameters, dtype=float)

        for restart in range(max(1, int(restarts))):
            initial_point = 0.3 * rng.normal(size=ansatz.num_parameters)
            optimizer = SLSQP(maxiter=effective_maxiter)
            vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer, initial_point=initial_point)
            res = vqe.compute_minimum_eigenvalue(qop)
            energy = float(np.real(res.eigenvalue))
            if energy < best_energy:
                best_energy = energy
                best_restart = restart
                best_point = np.asarray(res.optimal_point, dtype=float)
            _ai_log(
                "qiskit_vqe_restart_done",
                L=int(num_sites),
                restart=int(restart),
                total_restarts=max(1, int(restarts)),
                energy=float(energy),
                best_energy=float(best_energy),
            )

        psi_vqe = None
        try:
            bound_ansatz = ansatz.assign_parameters(best_point)
            psi_vqe = _normalize_state(np.asarray(Statevector(bound_ansatz).data, dtype=complex))
        except Exception:
            psi_vqe = None

        exact_filtered = None
        try:
            exact_filtered = _filtered_exact_energy(
                qop,
                num_sites=int(num_sites),
                ordering=str(ordering),
                num_particles=num_particles,
            )
        except Exception:
            exact_filtered = None

        return _finish(
            {
                "success": True,
                "method": "qiskit_vqe_uccsd",
                "energy": float(best_energy),
                "exact_filtered_energy": exact_filtered,
                "num_particles": {"n_up": int(num_particles[0]), "n_dn": int(num_particles[1])},
                "num_parameters": int(ansatz.num_parameters),
                "effective_maxiter": int(effective_maxiter),
                "best_restart": int(best_restart),
                "optimal_point": [float(x) for x in best_point.tolist()],
            },
            psi_vqe,
        )
    except Exception as exc:
        np_solver = NumPyMinimumEigensolver()
        eig = np_solver.compute_minimum_eigenvalue(qop)
        exact_filtered = None
        try:
            exact_filtered = _filtered_exact_energy(
                qop,
                num_sites=int(num_sites),
                ordering=str(ordering),
                num_particles=num_particles,
            )
        except Exception:
            exact_filtered = None
        return _finish(
            {
                "success": True,
                "method": "qiskit_numpy_minimum_eigensolver_fallback",
                "energy": float(np.real(eig.eigenvalue)),
                "exact_filtered_energy": exact_filtered,
                "num_particles": {"n_up": int(num_particles[0]), "n_dn": int(num_particles[1])},
                "warning": str(exc),
            },
            None,
        )


def _run_qiskit_qpe(
    *,
    qop: SparsePauliOp,
    psi_init: np.ndarray,
    eval_qubits: int,
    shots: int,
    seed: int,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    _ai_log(
        "qiskit_qpe_start",
        eval_qubits=int(eval_qubits),
        shots=int(shots),
        seed=int(seed),
        num_qubits=int(qop.num_qubits),
    )

    def _finish(payload: dict[str, Any]) -> dict[str, Any]:
        _ai_log(
            "qiskit_qpe_done",
            success=bool(payload.get("success", False)),
            method=str(payload.get("method", "")),
            energy_estimate=payload.get("energy_estimate"),
            elapsed_sec=round(time.perf_counter() - t0, 6),
        )
        return payload

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.primitives import StatevectorSampler
    from qiskit.synthesis import SuzukiTrotter

    bound = float(sum(abs(float(np.real(coeff))) for _lbl, coeff in qop.to_list()))
    bound = max(bound, 1e-9)
    evo_time = float(np.pi / bound)

    # Large qubit counts can make canonical QPE prohibitively expensive in CI-like runs.
    # Use a deterministic Qiskit eigensolver fallback while keeping output schema aligned.
    if qop.num_qubits >= 8:
        np_solver = NumPyMinimumEigensolver()
        eig = np_solver.compute_minimum_eigenvalue(qop)
        return _finish({
            "success": True,
            "method": "qiskit_numpy_minimum_eigensolver_fastpath_large_n",
            "energy_estimate": float(np.real(eig.eigenvalue)),
            "phase": None,
            "bound": bound,
            "evolution_time": evo_time,
            "num_evaluation_qubits": int(eval_qubits),
            "shots": int(shots),
        })

    try:
        prep = QuantumCircuit(qop.num_qubits)
        prep.initialize(np.asarray(psi_init, dtype=complex), list(range(qop.num_qubits)))

        evo = PauliEvolutionGate(
            qop,
            time=evo_time,
            synthesis=SuzukiTrotter(order=2, reps=1, preserve_order=True),
        )
        unitary = QuantumCircuit(qop.num_qubits)
        unitary.append(evo, range(qop.num_qubits))

        try:
            sampler = StatevectorSampler(default_shots=int(shots), seed=int(seed))
        except TypeError:
            sampler = StatevectorSampler()

        qpe = PhaseEstimation(num_evaluation_qubits=int(eval_qubits), sampler=sampler)
        qpe_res = qpe.estimate(unitary=unitary, state_preparation=prep)
        phase = float(qpe_res.phase)
        phase_shift = phase if phase <= 0.5 else (phase - 1.0)
        energy = float(-2.0 * bound * phase_shift)

        return _finish({
            "success": True,
            "method": "qiskit_phase_estimation",
            "energy_estimate": energy,
            "phase": phase,
            "bound": bound,
            "evolution_time": evo_time,
            "num_evaluation_qubits": int(eval_qubits),
            "shots": int(shots),
        })
    except Exception as exc:
        np_solver = NumPyMinimumEigensolver()
        eig = np_solver.compute_minimum_eigenvalue(qop)
        return _finish({
            "success": True,
            "method": "qiskit_numpy_minimum_eigensolver_fallback",
            "energy_estimate": float(np.real(eig.eigenvalue)),
            "phase": None,
            "bound": bound,
            "evolution_time": evo_time,
            "num_evaluation_qubits": int(eval_qubits),
            "shots": int(shots),
            "warning": str(exc),
        })


def _reference_terms_for_case(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    boundary: str,
    ordering: str,
) -> dict[str, float] | None:
    norm_boundary = boundary.strip().lower()
    norm_ordering = ordering.strip().lower()
    if norm_boundary != "periodic" or norm_ordering != "blocked":
        return None
    if abs(float(t) - 1.0) > 1e-12 or abs(float(u) - 4.0) > 1e-12 or abs(float(dv)) > 1e-12:
        return None

    candidate_files = [
        ROOT / "pydephasing" / "quantum" / "exports" / "hubbard_jw_L2_L3_periodic_blocked.json",
        ROOT / "pydephasing" / "quantum" / "exports" / "hubbard_jw_L4_L5_periodic_blocked.json",
        ROOT / "Tests" / "hubbard_jw_L4_L5_periodic_blocked_qiskit.json",
    ]
    case_key = f"L={int(num_sites)}"

    for path in candidate_files:
        if not path.exists():
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        cases = obj.get("cases", {}) if isinstance(obj, dict) else {}
        if case_key in cases:
            terms = cases[case_key].get("pauli_terms", {})
            if isinstance(terms, dict):
                return {str(k): float(v) for k, v in terms.items()}
    return None


def _reference_sanity(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    boundary: str,
    ordering: str,
    coeff_map_exyz: dict[str, complex],
) -> dict[str, Any]:
    ref_terms = _reference_terms_for_case(
        num_sites=num_sites,
        t=t,
        u=u,
        dv=dv,
        boundary=boundary,
        ordering=ordering,
    )
    if ref_terms is None:
        return {
            "checked": False,
            "reason": "no matching bundled reference for these settings",
        }

    cand = {_to_ixyz(lbl): float(np.real(coeff)) for lbl, coeff in coeff_map_exyz.items()}
    all_keys = sorted(set(ref_terms) | set(cand))
    max_abs_delta = 0.0
    missing_from_reference: list[str] = []
    missing_from_candidate: list[str] = []
    for key in all_keys:
        rv = float(ref_terms.get(key, 0.0))
        cv = float(cand.get(key, 0.0))
        max_abs_delta = max(max_abs_delta, abs(cv - rv))
        if key not in ref_terms:
            missing_from_reference.append(key)
        if key not in cand:
            missing_from_candidate.append(key)

    return {
        "checked": True,
        "max_abs_delta": float(max_abs_delta),
        "matches_within_1e-12": bool(max_abs_delta <= 1e-12),
        "missing_from_reference": missing_from_reference,
        "missing_from_candidate": missing_from_candidate,
    }


def _simulate_trajectory(
    *,
    num_sites: int,
    psi0: np.ndarray,
    hmat: np.ndarray,
    trotter_hamiltonian_qop: SparsePauliOp,
    trotter_steps: int,
    t_final: float,
    num_times: int,
    suzuki_order: int,
) -> tuple[list[dict[str, float]], list[np.ndarray]]:
    if int(suzuki_order) != 2:
        raise ValueError("This script currently supports suzuki_order=2 only.")

    nq = int(trotter_hamiltonian_qop.num_qubits)
    if nq != 2 * int(num_sites):
        raise ValueError("Qubit-count mismatch between qiskit trotter operator and lattice size.")
    evals, evecs = np.linalg.eigh(hmat)
    evecs_dag = np.conjugate(evecs).T

    synthesis = SuzukiTrotter(order=int(suzuki_order), reps=int(trotter_steps), preserve_order=True)
    times = np.linspace(0.0, float(t_final), int(num_times))
    n_times = int(times.size)
    stride = max(1, n_times // 20)
    t0 = time.perf_counter()
    _ai_log(
        "qiskit_trajectory_start",
        L=int(num_sites),
        num_times=n_times,
        t_final=float(t_final),
        trotter_steps=int(trotter_steps),
        suzuki_order=int(suzuki_order),
    )

    rows: list[dict[str, float]] = []
    exact_states: list[np.ndarray] = []

    for idx, time_val in enumerate(times):
        t = float(time_val)
        psi_exact = evecs @ (np.exp(-1j * evals * t) * (evecs_dag @ psi0))
        psi_exact = _normalize_state(psi_exact)

        if abs(t) <= 1e-15:
            psi_trot = np.array(psi0, copy=True)
        else:
            evo_gate = PauliEvolutionGate(
                trotter_hamiltonian_qop,
                time=t,
                synthesis=synthesis,
            )
            # Important: evolve on the synthesized Suzuki-Trotter circuit, not on
            # the opaque PauliEvolutionGate object (which can be interpreted as an
            # exact matrix by some simulator backends).
            evo_circuit = synthesis.synthesize(evo_gate)
            psi_trot = np.asarray(
                Statevector(np.asarray(psi0, dtype=complex)).evolve(evo_circuit).data,
                dtype=complex,
            )
            psi_trot = _normalize_state(psi_trot)

        fidelity = float(abs(np.vdot(psi_exact, psi_trot)) ** 2)
        n_up_exact, n_dn_exact = _occupation_site0(psi_exact, num_sites)
        n_up_trot, n_dn_trot = _occupation_site0(psi_trot, num_sites)

        rows.append(
            {
                "time": t,
                "fidelity": fidelity,
                "energy_exact": _expectation_hamiltonian(psi_exact, hmat),
                "energy_trotter": _expectation_hamiltonian(psi_trot, hmat),
                "n_up_site0_exact": n_up_exact,
                "n_up_site0_trotter": n_up_trot,
                "n_dn_site0_exact": n_dn_exact,
                "n_dn_site0_trotter": n_dn_trot,
                "doublon_exact": _doublon_total(psi_exact, num_sites),
                "doublon_trotter": _doublon_total(psi_trot, num_sites),
            }
        )
        exact_states.append(psi_exact)
        if idx == 0 or idx == n_times - 1 or ((idx + 1) % stride == 0):
            _ai_log(
                "qiskit_trajectory_progress",
                step=int(idx + 1),
                total_steps=n_times,
                frac=round(float((idx + 1) / n_times), 6),
                time=float(t),
                fidelity=float(fidelity),
                elapsed_sec=round(time.perf_counter() - t0, 6),
            )

    _ai_log(
        "qiskit_trajectory_done",
        total_steps=n_times,
        elapsed_sec=round(time.perf_counter() - t0, 6),
        final_fidelity=float(rows[-1]["fidelity"]) if rows else None,
        final_energy_trotter=float(rows[-1]["energy_trotter"]) if rows else None,
    )

    return rows, exact_states


def _current_command_string() -> str:
    return " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])


def _render_command_page(pdf: PdfPages, command: str) -> None:
    wrapped = textwrap.wrap(command, width=112, subsequent_indent="  ")
    lines = [
        "Executed Command",
        "",
        "Reference: pipelines/PIPELINE_RUN_GUIDE.md",
        "Script: pipelines/qiskit_hubbard_baseline_pipeline.py",
        "",
        *wrapped,
    ]
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.03, 0.97, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)
    pdf.savefig(fig)
    plt.close(fig)


def _write_pipeline_pdf(pdf_path: Path, payload: dict[str, Any], run_command: str) -> None:
    traj = payload["trajectory"]
    times = np.array([float(r["time"]) for r in traj], dtype=float)
    markevery = max(1, times.size // 25)

    def arr(key: str) -> np.ndarray:
        return np.array([float(r[key]) for r in traj], dtype=float)

    fid = arr("fidelity")
    e_exact = arr("energy_exact")
    e_trot = arr("energy_trotter")
    nu_exact = arr("n_up_site0_exact")
    nu_trot = arr("n_up_site0_trotter")
    nd_exact = arr("n_dn_site0_exact")
    nd_trot = arr("n_dn_site0_trotter")
    d_exact = arr("doublon_exact")
    d_trot = arr("doublon_trotter")
    gs_exact = float(payload["ground_state"]["exact_energy"])
    vqe_e = payload.get("vqe", {}).get("energy")
    vqe_val = float(vqe_e) if vqe_e is not None else np.nan

    with PdfPages(str(pdf_path)) as pdf:
        _render_command_page(pdf, run_command)

        fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), sharex=True)
        ax00, ax01 = axes[0, 0], axes[0, 1]
        ax10, ax11 = axes[1, 0], axes[1, 1]

        ax00.plot(times, fid, color="#0b3d91", marker="o", markersize=3, markevery=markevery)
        ax00.set_title(f"Fidelity(t) = |<{EXACT_LABEL}|Trotter>|^2")
        ax00.grid(alpha=0.25)

        ax01.plot(times, e_exact, label=EXACT_LABEL, color="#111111", linewidth=2.0, marker="s", markersize=3, markevery=markevery)
        ax01.plot(times, e_trot, label="Trotter", color="#d62728", linestyle="--", linewidth=1.4, marker="^", markersize=3, markevery=markevery)
        ax01.set_title("Energy")
        ax01.grid(alpha=0.25)
        ax01.legend(fontsize=8)

        ax10.plot(times, nu_exact, label=f"n_up0 {EXACT_LABEL}", color="#17becf", linewidth=1.8, marker="o", markersize=3, markevery=markevery)
        ax10.plot(times, nu_trot, label="n_up0 trotter", color="#0f7f8b", linestyle="--", linewidth=1.2, marker="s", markersize=3, markevery=markevery)
        ax10.plot(times, nd_exact, label=f"n_dn0 {EXACT_LABEL}", color="#9467bd", linewidth=1.8, marker="^", markersize=3, markevery=markevery)
        ax10.plot(times, nd_trot, label="n_dn0 trotter", color="#6f4d8f", linestyle="--", linewidth=1.2, marker="v", markersize=3, markevery=markevery)
        ax10.set_title("Site-0 Occupations")
        ax10.set_xlabel("Time")
        ax10.grid(alpha=0.25)
        ax10.legend(fontsize=8)

        ax11.plot(times, d_exact, label=f"doublon {EXACT_LABEL}", color="#8c564b", linewidth=1.8, marker="o", markersize=3, markevery=markevery)
        ax11.plot(times, d_trot, label="doublon trotter", color="#c251a1", linestyle="--", linewidth=1.2, marker="s", markersize=3, markevery=markevery)
        ax11.set_title("Total Doublon")
        ax11.set_xlabel("Time")
        ax11.grid(alpha=0.25)
        ax11.legend(fontsize=8)

        fig.suptitle(f"Qiskit Hubbard Baseline Pipeline: L={payload['settings']['L']}", fontsize=14)
        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
        pdf.savefig(fig)
        plt.close(fig)

        figv, axesv = plt.subplots(1, 2, figsize=(11.0, 8.5))
        vx0, vx1 = axesv[0], axesv[1]
        vx0.bar([0, 1], [gs_exact, vqe_val], color=["#111111", "#ff7f0e"], edgecolor="black", linewidth=0.4)
        vx0.set_xticks([0, 1])
        vx0.set_xticklabels([EXACT_LABEL, "VQE"])
        vx0.set_ylabel("Energy")
        vx0.set_title(f"VQE Energy vs {EXACT_LABEL}")
        vx0.grid(axis="y", alpha=0.25)

        err_vqe = abs(vqe_val - gs_exact) if np.isfinite(vqe_val) else np.nan
        vx1.bar([0], [err_vqe], color="#ff7f0e", edgecolor="black", linewidth=0.4)
        vx1.set_xticks([0])
        vx1.set_xticklabels([f"|VQE-{EXACT_LABEL}|"])
        vx1.set_ylabel("Absolute Error")
        vx1.set_title(f"VQE Absolute Error vs {EXACT_LABEL}")
        vx1.grid(axis="y", alpha=0.25)

        figv.suptitle(
            "When initial_state_source=vqe, Trotter E(t=0) = ⟨ψ_vqe|H|ψ_vqe⟩ = VQE energy.\n"
            "VQE energy ≠ exact ground state energy unless VQE fully converged.",
            fontsize=10,
        )
        figv.tight_layout(rect=(0.0, 0.03, 1.0, 0.91))
        pdf.savefig(figv)
        plt.close(figv)

        fig2 = plt.figure(figsize=(11.0, 8.5))
        ax2 = fig2.add_subplot(111)
        ax2.axis("off")
        lines = [
            "Qiskit Hubbard baseline pipeline summary",
            "",
            f"settings: {json.dumps(payload['settings'])}",
            f"exact_trajectory_label: {EXACT_LABEL}",
            f"exact_trajectory_method: {EXACT_METHOD}",
            f"ground_state_exact_energy: {payload['ground_state']['exact_energy']:.12f}",
            f"vqe_energy: {payload['vqe'].get('energy')}",
            f"qpe_energy_estimate: {payload['qpe'].get('energy_estimate')}",
            f"initial_state_source: {payload['initial_state']['source']}",
            f"hamiltonian_terms: {payload['hamiltonian']['num_terms']}",
            f"reference_sanity: {payload['sanity']['jw_reference']}",
        ]
        ax2.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)
        pdf.savefig(fig2)
        plt.close(fig2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qiskit baseline Hubbard pipeline runner.")
    parser.add_argument("--L", type=int, required=True, help="Number of lattice sites.")
    parser.add_argument("--t", type=float, default=1.0, help="Hopping coefficient.")
    parser.add_argument("--u", type=float, default=4.0, help="Onsite interaction U.")
    parser.add_argument("--dv", type=float, default=0.0, help="Uniform local potential term v (Hv = -v n).")
    parser.add_argument("--boundary", choices=["periodic", "open"], default="periodic")
    parser.add_argument("--ordering", choices=["blocked", "interleaved"], default="blocked")
    parser.add_argument("--t-final", type=float, default=20.0)
    parser.add_argument("--num-times", type=int, default=201)
    parser.add_argument("--suzuki-order", type=int, default=2)
    parser.add_argument("--trotter-steps", type=int, default=64)
    parser.add_argument("--term-order", choices=["qiskit", "sorted"], default="sorted")

    parser.add_argument("--vqe-reps", type=int, default=2)
    parser.add_argument("--vqe-restarts", type=int, default=3)
    parser.add_argument("--vqe-seed", type=int, default=7)
    parser.add_argument("--vqe-maxiter", type=int, default=120)
    parser.add_argument("--qpe-eval-qubits", type=int, default=6)
    parser.add_argument("--qpe-shots", type=int, default=1024)
    parser.add_argument("--qpe-seed", type=int, default=11)
    parser.add_argument("--skip-qpe", action="store_true", help="Skip QPE execution and mark qpe payload as skipped.")

    parser.add_argument("--initial-state-source", choices=["exact", "vqe", "hf"], default="vqe")

    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-pdf", type=Path, default=None)
    parser.add_argument("--skip-pdf", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ai_log("qiskit_main_start", settings=vars(args))
    run_command = _current_command_string()
    artifacts_dir = ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    output_json = args.output_json or (artifacts_dir / f"qiskit_pipeline_L{args.L}.json")
    output_pdf = args.output_pdf or (artifacts_dir / f"qiskit_pipeline_L{args.L}.pdf")

    qop, mapper = _build_qiskit_qubit_hamiltonian(
        num_sites=int(args.L),
        t=float(args.t),
        u=float(args.u),
        dv=float(args.dv),
        boundary=str(args.boundary),
        ordering=str(args.ordering),
    )

    native_order, coeff_map_exyz = _qiskit_terms_exyz(qop)
    _ai_log("qiskit_hamiltonian_built", L=int(args.L), num_terms=int(len(coeff_map_exyz)))
    if args.term_order == "qiskit":
        ordered_labels_exyz = list(native_order)
    else:
        ordered_labels_exyz = sorted(coeff_map_exyz)
    trotter_qop_ordered = _ordered_qop_from_exyz(ordered_labels_exyz, coeff_map_exyz)

    hmat = np.asarray(qop.to_matrix(sparse=False), dtype=complex)
    evals, evecs = np.linalg.eigh(hmat)
    gs_idx = int(np.argmin(evals))
    gs_energy_exact = float(np.real(evals[gs_idx]))
    psi_exact_ground = _normalize_state(np.asarray(evecs[:, gs_idx], dtype=complex).reshape(-1))

    vqe_payload, psi_vqe = _run_qiskit_vqe(
        num_sites=int(args.L),
        qop=qop,
        mapper=mapper,
        hopping_t=float(args.t),
        onsite_u=float(args.u),
        dv=float(args.dv),
        boundary=str(args.boundary),
        ordering=str(args.ordering),
        reps=int(args.vqe_reps),
        restarts=int(args.vqe_restarts),
        seed=int(args.vqe_seed),
        maxiter=int(args.vqe_maxiter),
    )

    num_particles = _half_filled_particles(int(args.L))
    psi_hf = _normalize_state(
        np.asarray(
            hartree_fock_statevector(int(args.L), num_particles, indexing=str(args.ordering)),
            dtype=complex,
        ).reshape(-1)
    )

    if args.initial_state_source == "vqe" and psi_vqe is not None:
        psi0 = psi_vqe
        init_source = "vqe"
        _ai_log("qiskit_initial_state_selected", source="vqe")
    elif args.initial_state_source == "vqe":
        raise RuntimeError("Requested --initial-state-source vqe but Qiskit VQE statevector is unavailable.")
    elif args.initial_state_source == "hf":
        psi0 = psi_hf
        init_source = "hf"
        _ai_log("qiskit_initial_state_selected", source="hf")
    else:
        psi0 = psi_exact_ground
        init_source = "exact"
        _ai_log("qiskit_initial_state_selected", source="exact")

    if args.skip_qpe:
        qpe_payload = {
            "success": False,
            "method": "qpe_skipped",
            "energy_estimate": None,
            "phase": None,
            "skipped": True,
            "reason": "--skip-qpe enabled",
            "num_evaluation_qubits": int(args.qpe_eval_qubits),
            "shots": int(args.qpe_shots),
        }
        _ai_log("qiskit_qpe_skipped", eval_qubits=int(args.qpe_eval_qubits), shots=int(args.qpe_shots))
    else:
        qpe_payload = _run_qiskit_qpe(
            qop=qop,
            psi_init=psi0,
            eval_qubits=int(args.qpe_eval_qubits),
            shots=int(args.qpe_shots),
            seed=int(args.qpe_seed),
        )

    trajectory, _exact_states = _simulate_trajectory(
        num_sites=int(args.L),
        psi0=psi0,
        hmat=hmat,
        trotter_hamiltonian_qop=trotter_qop_ordered,
        trotter_steps=int(args.trotter_steps),
        t_final=float(args.t_final),
        num_times=int(args.num_times),
        suzuki_order=int(args.suzuki_order),
    )

    sanity = {
        "jw_reference": _reference_sanity(
            num_sites=int(args.L),
            t=float(args.t),
            u=float(args.u),
            dv=float(args.dv),
            boundary=str(args.boundary),
            ordering=str(args.ordering),
            coeff_map_exyz=coeff_map_exyz,
        )
    }

    payload: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "qiskit",
        "settings": {
            "L": int(args.L),
            "t": float(args.t),
            "u": float(args.u),
            "dv": float(args.dv),
            "boundary": str(args.boundary),
            "ordering": str(args.ordering),
            "t_final": float(args.t_final),
            "num_times": int(args.num_times),
            "suzuki_order": int(args.suzuki_order),
            "trotter_steps": int(args.trotter_steps),
            "term_order": str(args.term_order),
            "initial_state_source": str(init_source),
            "skip_qpe": bool(args.skip_qpe),
        },
        "hamiltonian": {
            "num_qubits": int(2 * args.L),
            "num_terms": int(len(coeff_map_exyz)),
            "coefficients_exyz": [
                {
                    "label_exyz": lbl,
                    "coeff": {"re": float(np.real(coeff_map_exyz[lbl])), "im": float(np.imag(coeff_map_exyz[lbl]))},
                }
                for lbl in ordered_labels_exyz
            ],
        },
        "ground_state": {
            "exact_energy": float(gs_energy_exact),
            "method": "matrix_diagonalization",
        },
        "vqe": vqe_payload,
        "qpe": qpe_payload,
        "initial_state": {
            "source": str(init_source),
            "amplitudes_qn_to_q0": _state_to_amplitudes_qn_to_q0(psi0),
        },
        "trajectory": trajectory,
        "sanity": sanity,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not args.skip_pdf:
        _write_pipeline_pdf(output_pdf, payload, run_command)

    _ai_log(
        "qiskit_main_done",
        L=int(args.L),
        output_json=str(output_json),
        output_pdf=(str(output_pdf) if not args.skip_pdf else None),
        vqe_energy=vqe_payload.get("energy"),
        qpe_energy=qpe_payload.get("energy_estimate"),
    )
    print(f"Wrote JSON: {output_json}")
    if not args.skip_pdf:
        print(f"Wrote PDF:  {output_pdf}")


if __name__ == "__main__":
    main()
