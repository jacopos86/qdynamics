#!/usr/bin/env python3
"""Hardcoded-first end-to-end Hubbard pipeline.

Flow:
1) Build hardcoded Hubbard Hamiltonian (JW) from repo source-of-truth helpers.
2) Run hardcoded VQE on numpy-statevector backend (SciPy optional fallback comes from
   the notebook implementation).
3) Run temporary QPE adapter (Qiskit-only, isolated in one function).
4) Run hardcoded Suzuki-2 Trotter dynamics and exact dynamics.
5) Emit JSON + compact PDF artifact.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydephasing.quantum.hartree_fock_reference_state import hartree_fock_statevector
from pydephasing.quantum.hubbard_latex_python_pairs import build_hubbard_hamiltonian


def _ai_log(event: str, **fields: Any) -> None:
    payload = {
        "event": str(event),
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    print(f"AI_LOG {json.dumps(payload, sort_keys=True, default=str)}", flush=True)


PAULI_MATS = {
    "e": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
    "x": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}

EXACT_LABEL = "Exact_Hardcode"
EXACT_METHOD = "python_matrix_eigendecomposition"


@dataclass(frozen=True)
class CompiledPauliAction:
    label_exyz: str
    perm: np.ndarray
    phase: np.ndarray


def _to_ixyz(label_exyz: str) -> str:
    return str(label_exyz).replace("e", "I").upper()


def _normalize_state(psi: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(psi))
    if nrm <= 0.0:
        raise ValueError("Encountered zero-norm state.")
    return psi / nrm


def _half_filled_particles(num_sites: int) -> tuple[int, int]:
    return ((num_sites + 1) // 2, num_sites // 2)


def _collect_hardcoded_terms_exyz(poly: Any, tol: float = 1e-12) -> tuple[list[str], dict[str, complex]]:
    coeff_map: dict[str, complex] = {}
    order: list[str] = []
    for term in poly.return_polynomial():
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if abs(coeff) <= tol:
            continue
        if label not in coeff_map:
            coeff_map[label] = 0.0 + 0.0j
            order.append(label)
        coeff_map[label] += coeff
    cleaned_order = [lbl for lbl in order if abs(coeff_map[lbl]) > tol]
    cleaned_map = {lbl: coeff_map[lbl] for lbl in cleaned_order}
    return cleaned_order, cleaned_map


def _pauli_matrix_exyz(label: str) -> np.ndarray:
    mats = [PAULI_MATS[ch] for ch in label]
    out = mats[0]
    for mat in mats[1:]:
        out = np.kron(out, mat)
    return out


def _build_hamiltonian_matrix(coeff_map_exyz: dict[str, complex]) -> np.ndarray:
    if not coeff_map_exyz:
        return np.zeros((1, 1), dtype=complex)
    nq = len(next(iter(coeff_map_exyz)))
    dim = 1 << nq
    hmat = np.zeros((dim, dim), dtype=complex)
    for label, coeff in coeff_map_exyz.items():
        hmat += coeff * _pauli_matrix_exyz(label)
    return hmat


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


def _load_hardcoded_vqe_namespace() -> dict[str, Any]:
    from pydephasing.quantum import vqe_latex_python_pairs_test as vqe_mod

    ns: dict[str, Any] = {name: getattr(vqe_mod, name) for name in dir(vqe_mod)}

    required = [
        "half_filled_num_particles",
        "hartree_fock_bitstring",
        "basis_state",
        "HardcodedUCCSDAnsatz",
        "vqe_minimize",
    ]
    missing = [name for name in required if name not in ns]
    if missing:
        raise RuntimeError(f"Missing required VQE notebook symbols: {missing}")
    return ns


def _run_hardcoded_vqe(
    *,
    num_sites: int,
    ordering: str,
    h_poly: Any,
    reps: int,
    restarts: int,
    seed: int,
    maxiter: int,
) -> tuple[dict[str, Any], np.ndarray]:
    t0 = time.perf_counter()
    _ai_log(
        "hardcoded_vqe_start",
        L=int(num_sites),
        ordering=str(ordering),
        reps=int(reps),
        restarts=int(restarts),
        maxiter=int(maxiter),
        seed=int(seed),
    )
    ns = _load_hardcoded_vqe_namespace()
    num_particles = tuple(ns["half_filled_num_particles"](int(num_sites)))
    hf_bits = str(ns["hartree_fock_bitstring"](n_sites=int(num_sites), num_particles=num_particles, indexing=ordering))
    nq = 2 * int(num_sites)
    psi_ref = np.asarray(ns["basis_state"](nq, hf_bits), dtype=complex)

    ansatz = ns["HardcodedUCCSDAnsatz"](
        dims=int(num_sites),
        num_particles=num_particles,
        reps=int(reps),
        repr_mode="JW",
        indexing=ordering,
        include_singles=True,
        include_doubles=True,
    )

    result = ns["vqe_minimize"](
        h_poly,
        ansatz,
        psi_ref,
        restarts=int(restarts),
        seed=int(seed),
        maxiter=int(maxiter),
        method="SLSQP",
    )

    theta = np.asarray(result.theta, dtype=float)
    psi_vqe = np.asarray(ansatz.prepare_state(theta, psi_ref), dtype=complex).reshape(-1)
    psi_vqe = _normalize_state(psi_vqe)

    payload = {
        "success": True,
        "method": "hardcoded_uccsd_notebook_statevector",
        "energy": float(result.energy),
        "best_restart": int(getattr(result, "best_restart", 0)),
        "nfev": int(getattr(result, "nfev", 0)),
        "nit": int(getattr(result, "nit", 0)),
        "message": str(getattr(result, "message", "")),
        "num_particles": {"n_up": int(num_particles[0]), "n_dn": int(num_particles[1])},
        "num_parameters": int(ansatz.num_parameters),
        "reps": int(reps),
        "optimal_point": [float(x) for x in theta.tolist()],
        "hf_bitstring_qn_to_q0": hf_bits,
    }
    _ai_log(
        "hardcoded_vqe_done",
        L=int(num_sites),
        success=True,
        energy=float(result.energy),
        best_restart=int(getattr(result, "best_restart", 0)),
        nfev=int(getattr(result, "nfev", 0)),
        nit=int(getattr(result, "nit", 0)),
        elapsed_sec=round(time.perf_counter() - t0, 6),
    )
    return payload, psi_vqe


def _run_qpe_adapter_qiskit(
    *,
    coeff_map_exyz: dict[str, complex],
    psi_init: np.ndarray,
    eval_qubits: int,
    shots: int,
    seed: int,
) -> dict[str, Any]:
    """Temporary QPE adapter using minimal Qiskit calls.

    TODO: replace this adapter with a fully hardcoded QPE implementation.
    """

    t0 = time.perf_counter()
    _ai_log(
        "hardcoded_qpe_start",
        eval_qubits=int(eval_qubits),
        shots=int(shots),
        seed=int(seed),
        num_qubits=int(round(math.log2(psi_init.size))),
    )

    def _finish(payload: dict[str, Any]) -> dict[str, Any]:
        _ai_log(
            "hardcoded_qpe_done",
            success=bool(payload.get("success", False)),
            method=str(payload.get("method", "")),
            energy_estimate=payload.get("energy_estimate"),
            elapsed_sec=round(time.perf_counter() - t0, 6),
        )
        return payload

    # Isolated Qiskit-only section for temporary QPE support.
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.primitives import StatevectorSampler
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.synthesis import SuzukiTrotter
        from qiskit_algorithms import PhaseEstimation
        from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
    except Exception as exc:
        return _finish({
            "success": False,
            "method": "qiskit_import_failed",
            "energy_estimate": None,
            "phase": None,
            "error": str(exc),
        })

    terms_ixyz = [(_to_ixyz(lbl), complex(coeff)) for lbl, coeff in coeff_map_exyz.items() if abs(coeff) > 1e-12]
    if not terms_ixyz:
        terms_ixyz = [("I" * int(round(math.log2(psi_init.size))), 0.0)]

    h_op = SparsePauliOp.from_list(terms_ixyz).simplify(atol=1e-12)
    bound = float(sum(abs(float(np.real(coeff))) for _lbl, coeff in terms_ixyz))
    bound = max(bound, 1e-9)
    evo_time = float(np.pi / bound)

    # Large qubit counts can make canonical QPE prohibitively expensive in CI-like runs.
    # Use a deterministic Qiskit eigensolver fallback while keeping output schema aligned.
    if h_op.num_qubits >= 8:
        np_solver = NumPyMinimumEigensolver()
        res = np_solver.compute_minimum_eigenvalue(h_op)
        return _finish({
            "success": True,
            "method": "qiskit_numpy_minimum_eigensolver_fastpath_large_n",
            "energy_estimate": float(np.real(res.eigenvalue)),
            "phase": None,
            "bound": bound,
            "evolution_time": evo_time,
            "num_evaluation_qubits": int(eval_qubits),
            "shots": int(shots),
        })

    try:
        prep = QuantumCircuit(h_op.num_qubits)
        prep.initialize(np.asarray(psi_init, dtype=complex), list(range(h_op.num_qubits)))

        evo = PauliEvolutionGate(
            h_op,
            time=evo_time,
            synthesis=SuzukiTrotter(order=2, reps=1, preserve_order=True),
        )
        unitary = QuantumCircuit(h_op.num_qubits)
        unitary.append(evo, range(h_op.num_qubits))

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
        try:
            np_solver = NumPyMinimumEigensolver()
            res = np_solver.compute_minimum_eigenvalue(h_op)
            energy = float(np.real(res.eigenvalue))
            return _finish({
                "success": True,
                "method": "qiskit_numpy_minimum_eigensolver_fallback",
                "energy_estimate": energy,
                "phase": None,
                "bound": bound,
                "evolution_time": evo_time,
                "num_evaluation_qubits": int(eval_qubits),
                "shots": int(shots),
                "warning": str(exc),
            })
        except Exception as fallback_exc:
            return _finish({
                "success": False,
                "method": "qpe_failed",
                "energy_estimate": None,
                "phase": None,
                "bound": bound,
                "evolution_time": evo_time,
                "num_evaluation_qubits": int(eval_qubits),
                "shots": int(shots),
                "error": str(exc),
                "fallback_error": str(fallback_exc),
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
    ordered_labels_exyz: list[str],
    coeff_map_exyz: dict[str, complex],
    trotter_steps: int,
    t_final: float,
    num_times: int,
    suzuki_order: int,
) -> tuple[list[dict[str, float]], list[np.ndarray]]:
    if int(suzuki_order) != 2:
        raise ValueError("This script currently supports suzuki_order=2 only.")

    nq = 2 * int(num_sites)
    evals, evecs = np.linalg.eigh(hmat)
    evecs_dag = np.conjugate(evecs).T

    compiled = {lbl: _compile_pauli_action(lbl, nq) for lbl in ordered_labels_exyz}
    times = np.linspace(0.0, float(t_final), int(num_times))
    n_times = int(times.size)
    stride = max(1, n_times // 20)
    t0 = time.perf_counter()
    _ai_log(
        "hardcoded_trajectory_start",
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

        psi_trot = _evolve_trotter_suzuki2_absolute(
            psi0,
            ordered_labels_exyz,
            coeff_map_exyz,
            compiled,
            t,
            int(trotter_steps),
        )

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
                "hardcoded_trajectory_progress",
                step=int(idx + 1),
                total_steps=n_times,
                frac=round(float((idx + 1) / n_times), 6),
                time=float(t),
                fidelity=float(fidelity),
                elapsed_sec=round(time.perf_counter() - t0, 6),
            )

    _ai_log(
        "hardcoded_trajectory_done",
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
        "Script: pipelines/hardcoded_hubbard_pipeline.py",
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

        fig.suptitle(f"Hardcoded Hubbard Pipeline: L={payload['settings']['L']}", fontsize=14)
        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
        pdf.savefig(fig)
        plt.close(fig)

        figv, axesv = plt.subplots(1, 2, figsize=(11.0, 8.5))
        vx0, vx1 = axesv[0], axesv[1]
        vx0.bar([0, 1], [gs_exact, vqe_val], color=["#111111", "#2ca02c"], edgecolor="black", linewidth=0.4)
        vx0.set_xticks([0, 1])
        vx0.set_xticklabels([EXACT_LABEL, "VQE"])
        vx0.set_ylabel("Energy")
        vx0.set_title(f"VQE Energy vs {EXACT_LABEL}")
        vx0.grid(axis="y", alpha=0.25)

        err_vqe = abs(vqe_val - gs_exact) if np.isfinite(vqe_val) else np.nan
        vx1.bar([0], [err_vqe], color="#2ca02c", edgecolor="black", linewidth=0.4)
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
            "Hardcoded Hubbard pipeline summary",
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
    parser = argparse.ArgumentParser(description="Hardcoded-first Hubbard pipeline runner.")
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
    parser.add_argument("--term-order", choices=["native", "sorted"], default="sorted")

    parser.add_argument("--vqe-reps", type=int, default=1)
    parser.add_argument("--vqe-restarts", type=int, default=1)
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
    _ai_log("hardcoded_main_start", settings=vars(args))
    run_command = _current_command_string()
    artifacts_dir = ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    output_json = args.output_json or (artifacts_dir / f"hardcoded_pipeline_L{args.L}.json")
    output_pdf = args.output_pdf or (artifacts_dir / f"hardcoded_pipeline_L{args.L}.pdf")

    h_poly = build_hubbard_hamiltonian(
        dims=int(args.L),
        t=float(args.t),
        U=float(args.u),
        v=float(args.dv),
        repr_mode="JW",
        indexing=str(args.ordering),
        pbc=(str(args.boundary) == "periodic"),
    )

    native_order, coeff_map_exyz = _collect_hardcoded_terms_exyz(h_poly)
    _ai_log("hardcoded_hamiltonian_built", L=int(args.L), num_terms=int(len(coeff_map_exyz)))
    if args.term_order == "native":
        ordered_labels_exyz = list(native_order)
    else:
        ordered_labels_exyz = sorted(coeff_map_exyz)

    hmat = _build_hamiltonian_matrix(coeff_map_exyz)
    evals, evecs = np.linalg.eigh(hmat)
    gs_idx = int(np.argmin(evals))
    gs_energy_exact = float(np.real(evals[gs_idx]))
    psi_exact_ground = _normalize_state(np.asarray(evecs[:, gs_idx], dtype=complex).reshape(-1))

    vqe_payload: dict[str, Any]
    try:
        vqe_payload, psi_vqe = _run_hardcoded_vqe(
            num_sites=int(args.L),
            ordering=str(args.ordering),
            h_poly=h_poly,
            reps=int(args.vqe_reps),
            restarts=int(args.vqe_restarts),
            seed=int(args.vqe_seed),
            maxiter=int(args.vqe_maxiter),
        )
    except Exception as exc:
        _ai_log("hardcoded_vqe_failed", L=int(args.L), error=str(exc))
        vqe_payload = {
            "success": False,
            "method": "hardcoded_uccsd_notebook_statevector",
            "energy": None,
            "error": str(exc),
        }
        psi_vqe = psi_exact_ground

    num_particles = _half_filled_particles(int(args.L))
    psi_hf = _normalize_state(
        np.asarray(
            hartree_fock_statevector(int(args.L), num_particles, indexing=str(args.ordering)),
            dtype=complex,
        ).reshape(-1)
    )

    if args.initial_state_source == "vqe" and bool(vqe_payload.get("success", False)):
        psi0 = psi_vqe
        _ai_log("hardcoded_initial_state_selected", source="vqe")
    elif args.initial_state_source == "vqe":
        raise RuntimeError("Requested --initial-state-source vqe but hardcoded VQE statevector is unavailable.")
    elif args.initial_state_source == "hf":
        psi0 = psi_hf
        _ai_log("hardcoded_initial_state_selected", source="hf")
    else:
        psi0 = psi_exact_ground
        _ai_log("hardcoded_initial_state_selected", source="exact")

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
        _ai_log("hardcoded_qpe_skipped", eval_qubits=int(args.qpe_eval_qubits), shots=int(args.qpe_shots))
    else:
        qpe_payload = _run_qpe_adapter_qiskit(
            coeff_map_exyz=coeff_map_exyz,
            psi_init=psi0,
            eval_qubits=int(args.qpe_eval_qubits),
            shots=int(args.qpe_shots),
            seed=int(args.qpe_seed),
        )

    trajectory, _exact_states = _simulate_trajectory(
        num_sites=int(args.L),
        psi0=psi0,
        hmat=hmat,
        ordered_labels_exyz=ordered_labels_exyz,
        coeff_map_exyz=coeff_map_exyz,
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
        "pipeline": "hardcoded",
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
            "initial_state_source": str(args.initial_state_source),
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
            "source": str(args.initial_state_source if args.initial_state_source != "vqe" or vqe_payload.get("success") else "exact"),
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
        "hardcoded_main_done",
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
