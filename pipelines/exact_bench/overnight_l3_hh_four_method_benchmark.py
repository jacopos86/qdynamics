#!/usr/bin/env python3
"""Overnight L=3 HH benchmark runner for four methods across sectors/configs.

Methods:
  1) Conventional VQE + HH layerwise HVA
  2) Conventional VQE + HH termwise fixed pool parameterization (non-ADAPT)
  3) ADAPT-VQE with PAOP std pool
  4) ADAPT-VQE with PAOP LF std pool

This script does not modify model physics definitions. It only orchestrates
benchmark runs, collects per-run metrics, and writes summary reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Path setup
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.hubbard_latex_python_pairs import build_hubbard_holstein_hamiltonian
from src.quantum.operator_pools import make_pool as make_paop_pool
from src.quantum.vqe_latex_python_pairs import (
    AnsatzTerm,
    HubbardHolsteinLayerwiseAnsatz,
    HubbardHolsteinTermwiseAnsatz,
    apply_exp_pauli_polynomial,
    apply_pauli_string,
    exact_ground_energy_sector_hh,
    expval_pauli_polynomial,
    hubbard_holstein_reference_state,
)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ai_log(event: str, **fields: Any) -> None:
    payload = {"event": str(event), "ts_utc": _now_utc_iso(), **fields}
    print(f"AI_LOG {json.dumps(payload, sort_keys=True, default=str)}", flush=True)


METHOD_SPECS: dict[str, dict[str, str]] = {
    "m1_hh_hva": {"kind": "conventional", "ansatz": "hh_hva"},
    "m2_hh_hva_tw": {"kind": "conventional", "ansatz": "hh_hva_tw"},
    "m3_adapt_paop_std": {"kind": "adapt", "pool": "paop_std"},
    "m4_adapt_paop_lf_std": {"kind": "adapt", "pool": "paop_lf_std"},
}


SUMMARY_FIELDS: list[str] = [
    "run_id",
    "status",
    "error_message",
    "started_utc",
    "finished_utc",
    "runtime_s",
    "smoke_test",
    "method_id",
    "method_kind",
    "sector",
    "seed",
    "attempt_idx",
    "L",
    "problem",
    "t",
    "U",
    "dv",
    "omega0",
    "g_ep",
    "n_ph_max",
    "boson_encoding",
    "ordering",
    "boundary",
    "ansatz_name",
    "pool_name",
    "vqe_reps",
    "vqe_restarts",
    "vqe_maxiter",
    "vqe_method",
    "adapt_max_depth",
    "adapt_eps_grad",
    "adapt_eps_energy",
    "adapt_maxiter",
    "adapt_depth_reached",
    "adapt_stop_reason",
    "wallclock_cap_s",
    "E_exact_sector",
    "E_best",
    "E_last",
    "delta_E_abs",
    "N_up_target",
    "N_dn_target",
    "N_up_expect",
    "N_dn_expect",
    "N_up_abs_err",
    "N_dn_abs_err",
    "N_sector_abs_err_sum",
    "sector_leak_flag",
    "num_parameters",
    "pool_size",
    "nfev",
    "nit",
]


@dataclass(frozen=True)
class AttemptConfig:
    run_id: str
    method_id: str
    sector_n_up: int
    sector_n_dn: int
    seed: int
    attempt_idx: int
    smoke_test: bool
    # Physics
    L: int
    t: float
    U: float
    dv: float
    omega0: float
    g_ep: float
    n_ph_max: int
    boson_encoding: str
    ordering: str
    boundary: str
    # Conventional VQE knobs
    vqe_reps: int
    vqe_restarts: int
    vqe_maxiter: int
    vqe_method: str
    # ADAPT knobs
    adapt_max_depth: int
    adapt_eps_grad: float
    adapt_eps_energy: float
    adapt_maxiter: int
    paop_r: int
    paop_split_paulis: bool
    paop_prune_eps: float
    paop_normalization: str
    adapt_allow_repeats: bool
    wallclock_cap_s: int


def _sector_tuple_str(n_up: int, n_dn: int) -> str:
    return f"({int(n_up)},{int(n_dn)})"


# Built-in math:
#   H_HH = H_t + H_U + H_ph + H_ep
def _build_hh_hamiltonian(cfg: AttemptConfig):
    return build_hubbard_holstein_hamiltonian(
        dims=int(cfg.L),
        J=float(cfg.t),
        U=float(cfg.U),
        omega0=float(cfg.omega0),
        g=float(cfg.g_ep),
        n_ph_max=int(cfg.n_ph_max),
        boson_encoding=str(cfg.boson_encoding),
        repr_mode="JW",
        indexing=str(cfg.ordering),
        pbc=(str(cfg.boundary).strip().lower() == "periodic"),
        include_zero_point=True,
    )


# Built-in math:
#   E_exact^(sector) = min_{|psi> in sector} <psi|H|psi>
def _exact_sector_energy(h_poly: Any, cfg: AttemptConfig) -> float:
    return float(
        exact_ground_energy_sector_hh(
            h_poly,
            num_sites=int(cfg.L),
            num_particles=(int(cfg.sector_n_up), int(cfg.sector_n_dn)),
            n_ph_max=int(cfg.n_ph_max),
            boson_encoding=str(cfg.boson_encoding),
            indexing=str(cfg.ordering),
        )
    )


# Built-in math:
#   |psi_ref> = |HF_(n_up,n_dn)> ⊗ |phonon_vac>
def _reference_state(cfg: AttemptConfig) -> np.ndarray:
    psi = np.asarray(
        hubbard_holstein_reference_state(
            dims=int(cfg.L),
            num_particles=(int(cfg.sector_n_up), int(cfg.sector_n_dn)),
            n_ph_max=int(cfg.n_ph_max),
            boson_encoding=str(cfg.boson_encoding),
            indexing=str(cfg.ordering),
        ),
        dtype=complex,
    )
    return _normalize_state(psi)


def _normalize_state(psi: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(psi))
    if nrm <= 0.0:
        raise ValueError("Encountered zero-norm state.")
    return psi / nrm


def _fermion_spin_qubit_index_sets(num_sites: int, ordering: str) -> tuple[list[int], list[int]]:
    """Return (up_qubits, down_qubits) among fermion qubits q=0..(2L-1)."""
    n_sites_i = int(num_sites)
    ordering_s = str(ordering).strip().lower()
    if ordering_s == "blocked":
        up = list(range(n_sites_i))
        dn = list(range(n_sites_i, 2 * n_sites_i))
        return up, dn
    if ordering_s == "interleaved":
        up = list(range(0, 2 * n_sites_i, 2))
        dn = list(range(1, 2 * n_sites_i, 2))
        return up, dn
    raise ValueError(f"Unsupported ordering '{ordering}'")


def _particle_number_expectation(psi: np.ndarray, qubits: list[int]) -> float:
    """Expectation of Σ_{q in qubits} n_q for a statevector in computational basis."""
    psi_vec = np.asarray(psi, dtype=complex).reshape(-1)
    probs = np.abs(psi_vec) ** 2
    norm = float(np.sum(probs))
    if norm <= 0.0:
        raise ValueError("Invalid state norm in particle-number diagnostic.")
    probs = probs / norm
    basis_idx = np.arange(probs.size, dtype=np.uint64)
    count = 0.0
    for q in qubits:
        bits = ((basis_idx >> np.uint64(int(q))) & np.uint64(1)).astype(np.float64)
        count += float(np.dot(probs, bits))
    return float(count)


# Built-in math:
#   N_up = <psi| Σ_{i in up} n_i |psi>, N_dn = <psi| Σ_{i in dn} n_i |psi>
def _sector_particle_diagnostics(psi: np.ndarray, cfg: AttemptConfig) -> dict[str, Any]:
    up_qubits, dn_qubits = _fermion_spin_qubit_index_sets(int(cfg.L), str(cfg.ordering))
    n_up_target = float(cfg.sector_n_up)
    n_dn_target = float(cfg.sector_n_dn)
    n_up_expect = _particle_number_expectation(psi, up_qubits)
    n_dn_expect = _particle_number_expectation(psi, dn_qubits)
    n_up_abs_err = abs(n_up_expect - n_up_target)
    n_dn_abs_err = abs(n_dn_expect - n_dn_target)
    n_sector_abs_err_sum = n_up_abs_err + n_dn_abs_err
    return {
        "N_up_target": int(cfg.sector_n_up),
        "N_dn_target": int(cfg.sector_n_dn),
        "N_up_expect": float(n_up_expect),
        "N_dn_expect": float(n_dn_expect),
        "N_up_abs_err": float(n_up_abs_err),
        "N_dn_abs_err": float(n_dn_abs_err),
        "N_sector_abs_err_sum": float(n_sector_abs_err_sum),
        "sector_leak_flag": bool(n_sector_abs_err_sum > 1e-6),
    }


def _try_import_scipy_minimize():
    try:
        from scipy.optimize import minimize  # type: ignore
    except Exception:
        minimize = None
    return minimize


# Built-in math:
#   E(theta) = <psi_ref| U(theta)^dagger H U(theta) |psi_ref>
#   E_best = min_r E_r, E_last = E_{r=last}
def _run_vqe_with_restarts(
    h_poly: Any,
    ansatz: Any,
    psi_ref: np.ndarray,
    *,
    restarts: int,
    seed: int,
    maxiter: int,
    method: str,
    initial_point_stddev: float = 0.3,
) -> dict[str, Any]:
    minimize = _try_import_scipy_minimize()
    rng = np.random.default_rng(int(seed))
    npar = int(ansatz.num_parameters)
    if npar <= 0:
        raise ValueError("ansatz has no parameters")

    def energy_fn(x: np.ndarray) -> float:
        theta = np.asarray(x, dtype=float)
        psi = ansatz.prepare_state(theta, psi_ref)
        return float(expval_pauli_polynomial(psi, h_poly))

    best_energy = float("inf")
    best_theta: np.ndarray | None = None
    best_restart = -1
    best_nfev = 0
    best_nit = 0
    nfev_total = 0
    nit_total = 0
    restart_energies: list[float] = []

    for r in range(int(restarts)):
        x0 = initial_point_stddev * rng.normal(size=npar)
        if minimize is not None:
            res = minimize(
                energy_fn,
                x0,
                method=str(method),
                options={"maxiter": int(maxiter)},
            )
            energy = float(res.fun)
            theta_opt = np.asarray(res.x, dtype=float)
            nfev = int(getattr(res, "nfev", 0))
            nit = int(getattr(res, "nit", 0))
        else:
            # Fallback coordinate search
            theta_opt = np.asarray(x0, dtype=float)
            step = 0.2
            nfev = 0
            nit = 0
            energy = energy_fn(theta_opt)
            nfev += 1
            for it in range(int(maxiter)):
                improved = False
                for k in range(npar):
                    for sgn in (+1.0, -1.0):
                        trial = theta_opt.copy()
                        trial[k] += sgn * step
                        e_trial = energy_fn(trial)
                        nfev += 1
                        if e_trial < energy:
                            energy = e_trial
                            theta_opt = trial
                            improved = True
                nit = it + 1
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break

        restart_energies.append(float(energy))
        nfev_total += int(nfev)
        nit_total += int(nit)
        if float(energy) < best_energy:
            best_energy = float(energy)
            best_theta = np.asarray(theta_opt, dtype=float)
            best_restart = int(r)
            best_nfev = int(nfev)
            best_nit = int(nit)

    if best_theta is None:
        raise RuntimeError("No valid VQE restart completed")

    psi_best = np.asarray(ansatz.prepare_state(best_theta, psi_ref), dtype=complex).reshape(-1)
    psi_best = _normalize_state(psi_best)
    return {
        "E_best": float(best_energy),
        "E_last": float(restart_energies[-1]),
        "best_restart": int(best_restart),
        "nfev": int(best_nfev),
        "nit": int(best_nit),
        "nfev_total": int(nfev_total),
        "nit_total": int(nit_total),
        "num_parameters": int(ansatz.num_parameters),
        "theta_best": best_theta.tolist(),
        "psi_best": psi_best.tolist(),  # retained for parity/debug; not used in summary
    }


def _apply_pauli_polynomial(state: np.ndarray, poly: Any) -> np.ndarray:
    terms = poly.return_polynomial()
    if not terms:
        return np.zeros_like(state)
    nq = int(terms[0].nqubit())
    result = np.zeros_like(state)
    id_str = "e" * nq
    for term in terms:
        ps = term.pw2strng()
        coeff = complex(term.p_coeff)
        if abs(coeff) < 1e-15:
            continue
        if ps == id_str:
            result += coeff * state
        else:
            result += coeff * apply_pauli_string(state, ps)
    return result


# Built-in math:
#   dE/dtheta_j|_{0} = i <psi| [H, G_j] |psi> = 2 Im(<H psi | G_j psi>)
def _commutator_gradient(h_poly: Any, op: AnsatzTerm, psi_current: np.ndarray) -> float:
    g_psi = _apply_pauli_polynomial(psi_current, op.polynomial)
    h_psi = _apply_pauli_polynomial(psi_current, h_poly)
    return float(2.0 * np.vdot(h_psi, g_psi).imag)


def _prepare_adapt_state(psi_ref: np.ndarray, selected_ops: list[AnsatzTerm], theta: np.ndarray) -> np.ndarray:
    psi = np.array(psi_ref, copy=True)
    for k, op in enumerate(selected_ops):
        psi = apply_exp_pauli_polynomial(psi, op.polynomial, float(theta[k]))
    return psi


def _adapt_energy(h_poly: Any, psi_ref: np.ndarray, selected_ops: list[AnsatzTerm], theta: np.ndarray) -> float:
    psi = _prepare_adapt_state(psi_ref, selected_ops, theta)
    return float(expval_pauli_polynomial(psi, h_poly))


# Built-in math:
#   ADAPT grows U(theta) = Π_k exp(-i theta_k G_{j_k}) by greedy max |gradient|
def _run_adapt_once(h_poly: Any, psi_ref: np.ndarray, pool: list[AnsatzTerm], cfg: AttemptConfig) -> dict[str, Any]:
    minimize = _try_import_scipy_minimize()
    if minimize is None:
        raise RuntimeError("SciPy minimize is required for ADAPT re-optimization in this runner.")

    selected_ops: list[AnsatzTerm] = []
    theta = np.zeros(0, dtype=float)
    available = set(range(len(pool)))
    allow_repeats = bool(cfg.adapt_allow_repeats)

    energy_current = float(expval_pauli_polynomial(psi_ref, h_poly))
    history: list[float] = [float(energy_current)]
    nfev_total = 1
    stop_reason = "max_depth"

    for _depth in range(int(cfg.adapt_max_depth)):
        psi_current = _prepare_adapt_state(psi_ref, selected_ops, theta)
        gradients = np.zeros(len(pool), dtype=float)
        if allow_repeats:
            indices = range(len(pool))
        else:
            indices = sorted(available)
        for idx in indices:
            gradients[idx] = _commutator_gradient(h_poly, pool[idx], psi_current)

        grad_mag = np.abs(gradients)
        if not allow_repeats:
            mask = np.zeros(len(pool), dtype=bool)
            for idx in available:
                mask[idx] = True
            grad_mag[~mask] = 0.0
        best_idx = int(np.argmax(grad_mag))
        max_grad = float(grad_mag[best_idx])
        if max_grad < float(cfg.adapt_eps_grad):
            stop_reason = "eps_grad"
            break

        selected_ops.append(pool[best_idx])
        theta = np.append(theta, 0.0)
        if not allow_repeats:
            available.discard(best_idx)

        energy_prev = float(energy_current)

        def objective(x: np.ndarray) -> float:
            return _adapt_energy(h_poly, psi_ref, selected_ops, x)

        res = minimize(
            objective,
            theta,
            method="COBYLA",
            options={"maxiter": int(cfg.adapt_maxiter), "rhobeg": 0.3},
        )
        theta = np.asarray(res.x, dtype=float)
        energy_current = float(res.fun)
        history.append(float(energy_current))
        nfev_total += int(getattr(res, "nfev", 0))

        if abs(energy_current - energy_prev) < float(cfg.adapt_eps_energy):
            stop_reason = "eps_energy"
            break
        if (not allow_repeats) and len(available) == 0:
            stop_reason = "pool_exhausted"
            break

    psi_best = _prepare_adapt_state(psi_ref, selected_ops, theta)
    psi_best = _normalize_state(np.asarray(psi_best, dtype=complex).reshape(-1))
    return {
        "E_best": float(min(history)),
        "E_last": float(history[-1]),
        "adapt_depth_reached": int(len(selected_ops)),
        "adapt_stop_reason": str(stop_reason),
        "num_parameters": int(theta.size),
        "nfev": int(nfev_total),
        "nit": int(len(history) - 1),
        "history": [float(x) for x in history],
        "pool_size": int(len(pool)),
        "psi_best": psi_best.tolist(),  # retained for diagnostics parity with conventional branch
    }


def _build_conventional_ansatz(cfg: AttemptConfig, ansatz_name: str):
    if ansatz_name == "hh_hva":
        return HubbardHolsteinLayerwiseAnsatz(
            dims=int(cfg.L),
            J=float(cfg.t),
            U=float(cfg.U),
            omega0=float(cfg.omega0),
            g=float(cfg.g_ep),
            n_ph_max=int(cfg.n_ph_max),
            boson_encoding=str(cfg.boson_encoding),
            reps=int(cfg.vqe_reps),
            repr_mode="JW",
            indexing=str(cfg.ordering),
            pbc=(str(cfg.boundary).strip().lower() == "periodic"),
        )
    if ansatz_name == "hh_hva_tw":
        return HubbardHolsteinTermwiseAnsatz(
            dims=int(cfg.L),
            J=float(cfg.t),
            U=float(cfg.U),
            omega0=float(cfg.omega0),
            g=float(cfg.g_ep),
            n_ph_max=int(cfg.n_ph_max),
            boson_encoding=str(cfg.boson_encoding),
            reps=int(cfg.vqe_reps),
            repr_mode="JW",
            indexing=str(cfg.ordering),
            pbc=(str(cfg.boundary).strip().lower() == "periodic"),
        )
    raise ValueError(f"Unsupported conventional ansatz '{ansatz_name}'")


def _run_attempt_inner(cfg: AttemptConfig) -> dict[str, Any]:
    method_spec = METHOD_SPECS[str(cfg.method_id)]
    h_poly = _build_hh_hamiltonian(cfg)
    e_exact = _exact_sector_energy(h_poly, cfg)
    psi_ref = _reference_state(cfg)

    out: dict[str, Any] = {
        "E_exact_sector": float(e_exact),
        "ansatz_name": "",
        "pool_name": "",
        "N_up_target": int(cfg.sector_n_up),
        "N_dn_target": int(cfg.sector_n_dn),
        "N_up_expect": None,
        "N_dn_expect": None,
        "N_up_abs_err": None,
        "N_dn_abs_err": None,
        "N_sector_abs_err_sum": None,
        "sector_leak_flag": None,
        "adapt_depth_reached": None,
        "adapt_stop_reason": "",
        "num_parameters": None,
        "pool_size": None,
        "nfev": None,
        "nit": None,
    }
    psi_final: np.ndarray | None = None

    if method_spec["kind"] == "conventional":
        ansatz_name = str(method_spec["ansatz"])
        ansatz = _build_conventional_ansatz(cfg, ansatz_name)
        vqe = _run_vqe_with_restarts(
            h_poly,
            ansatz,
            psi_ref,
            restarts=int(cfg.vqe_restarts),
            seed=int(cfg.seed),
            maxiter=int(cfg.vqe_maxiter),
            method=str(cfg.vqe_method),
        )
        out.update(
            {
                "ansatz_name": ansatz_name,
                "E_best": float(vqe["E_best"]),
                "E_last": float(vqe["E_last"]),
                "num_parameters": int(vqe["num_parameters"]),
                "nfev": int(vqe["nfev_total"]),
                "nit": int(vqe["nit_total"]),
            }
        )
        psi_final = _normalize_state(np.asarray(vqe["psi_best"], dtype=complex).reshape(-1))
    elif method_spec["kind"] == "adapt":
        pool_name = str(method_spec["pool"])
        pool_specs = make_paop_pool(
            pool_name,
            num_sites=int(cfg.L),
            num_particles=(int(cfg.sector_n_up), int(cfg.sector_n_dn)),
            n_ph_max=int(cfg.n_ph_max),
            boson_encoding=str(cfg.boson_encoding),
            ordering=str(cfg.ordering),
            boundary=str(cfg.boundary),
            paop_r=int(cfg.paop_r),
            paop_split_paulis=bool(cfg.paop_split_paulis),
            paop_prune_eps=float(cfg.paop_prune_eps),
            paop_normalization=str(cfg.paop_normalization),
        )
        pool = [AnsatzTerm(label=str(lbl), polynomial=poly) for lbl, poly in pool_specs]
        if len(pool) == 0:
            raise RuntimeError(f"Empty ADAPT pool for {pool_name}")
        adapt = _run_adapt_once(h_poly, psi_ref, pool, cfg)
        out.update(
            {
                "pool_name": pool_name,
                "E_best": float(adapt["E_best"]),
                "E_last": float(adapt["E_last"]),
                "adapt_depth_reached": int(adapt["adapt_depth_reached"]),
                "adapt_stop_reason": str(adapt["adapt_stop_reason"]),
                "num_parameters": int(adapt["num_parameters"]),
                "pool_size": int(adapt["pool_size"]),
                "nfev": int(adapt["nfev"]),
                "nit": int(adapt["nit"]),
            }
        )
        psi_final = _normalize_state(np.asarray(adapt["psi_best"], dtype=complex).reshape(-1))
    else:
        raise ValueError(f"Unsupported method kind for {cfg.method_id}")

    if psi_final is None:
        raise RuntimeError("Internal error: missing final state for sector diagnostics.")
    out.update(_sector_particle_diagnostics(psi_final, cfg))
    out["delta_E_abs"] = float(abs(float(out["E_best"]) - float(out["E_exact_sector"])))
    return out


def _attempt_worker(cfg: AttemptConfig, queue: mp.Queue) -> None:
    started = _now_utc_iso()
    try:
        payload = _run_attempt_inner(cfg)
        queue.put({"ok": True, "started_utc": started, "payload": payload})
    except Exception:
        queue.put(
            {
                "ok": False,
                "started_utc": started,
                "error_message": traceback.format_exc(limit=8),
            }
        )


def _execute_with_timeout(cfg: AttemptConfig) -> dict[str, Any]:
    queue: mp.Queue = mp.Queue()
    process = mp.Process(target=_attempt_worker, args=(cfg, queue))
    started_wall = time.perf_counter()
    process.start()
    process.join(timeout=float(cfg.wallclock_cap_s))
    runtime_s = float(time.perf_counter() - started_wall)

    base_row: dict[str, Any] = {
        "run_id": str(cfg.run_id),
        "status": "error",
        "error_message": "",
        "started_utc": _now_utc_iso(),
        "finished_utc": _now_utc_iso(),
        "runtime_s": runtime_s,
        "smoke_test": bool(cfg.smoke_test),
        "method_id": str(cfg.method_id),
        "method_kind": str(METHOD_SPECS[str(cfg.method_id)]["kind"]),
        "sector": _sector_tuple_str(cfg.sector_n_up, cfg.sector_n_dn),
        "seed": int(cfg.seed),
        "attempt_idx": int(cfg.attempt_idx),
        "L": int(cfg.L),
        "problem": "hh",
        "t": float(cfg.t),
        "U": float(cfg.U),
        "dv": float(cfg.dv),
        "omega0": float(cfg.omega0),
        "g_ep": float(cfg.g_ep),
        "n_ph_max": int(cfg.n_ph_max),
        "boson_encoding": str(cfg.boson_encoding),
        "ordering": str(cfg.ordering),
        "boundary": str(cfg.boundary),
        "ansatz_name": "",
        "pool_name": "",
        "vqe_reps": int(cfg.vqe_reps),
        "vqe_restarts": int(cfg.vqe_restarts),
        "vqe_maxiter": int(cfg.vqe_maxiter),
        "vqe_method": str(cfg.vqe_method),
        "adapt_max_depth": int(cfg.adapt_max_depth),
        "adapt_eps_grad": float(cfg.adapt_eps_grad),
        "adapt_eps_energy": float(cfg.adapt_eps_energy),
        "adapt_maxiter": int(cfg.adapt_maxiter),
        "adapt_depth_reached": None,
        "adapt_stop_reason": "",
        "wallclock_cap_s": int(cfg.wallclock_cap_s),
        "E_exact_sector": None,
        "E_best": None,
        "E_last": None,
        "delta_E_abs": None,
        "N_up_target": int(cfg.sector_n_up),
        "N_dn_target": int(cfg.sector_n_dn),
        "N_up_expect": None,
        "N_dn_expect": None,
        "N_up_abs_err": None,
        "N_dn_abs_err": None,
        "N_sector_abs_err_sum": None,
        "sector_leak_flag": None,
        "num_parameters": None,
        "pool_size": None,
        "nfev": None,
        "nit": None,
    }

    if process.is_alive():
        process.terminate()
        process.join(timeout=2.0)
        base_row["status"] = "timeout"
        base_row["error_message"] = f"wallclock cap exceeded ({cfg.wallclock_cap_s}s)"
        base_row["finished_utc"] = _now_utc_iso()
        return base_row

    message: dict[str, Any] | None = None
    try:
        if not queue.empty():
            message = queue.get_nowait()
    except Exception:
        message = None

    if message is None:
        base_row["status"] = "error"
        base_row["error_message"] = "worker exited without result payload"
        base_row["finished_utc"] = _now_utc_iso()
        return base_row

    base_row["started_utc"] = str(message.get("started_utc", base_row["started_utc"]))
    base_row["finished_utc"] = _now_utc_iso()
    if bool(message.get("ok", False)):
        payload = message["payload"]
        base_row.update(payload)
        base_row["status"] = "ok"
        base_row["error_message"] = ""
    else:
        base_row["status"] = "error"
        base_row["error_message"] = str(message.get("error_message", "unknown error"))
    return base_row


def _parse_sector_list(raw: list[str]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for item in raw:
        s = str(item).strip().replace("(", "").replace(")", "")
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError(f"Invalid sector '{item}'. Use n_up,n_dn.")
        out.append((int(parts[0]), int(parts[1])))
    return out


def _build_ordering_boundary_pairs(raw: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for item in raw:
        parts = [p.strip().lower() for p in str(item).split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError(f"Invalid ordering/boundary pair '{item}'. Use ordering,boundary.")
        ordering, boundary = parts
        if ordering not in {"blocked", "interleaved"}:
            raise ValueError(f"Unsupported ordering '{ordering}'.")
        if boundary not in {"open", "periodic"}:
            raise ValueError(f"Unsupported boundary '{boundary}'.")
        out.append((ordering, boundary))
    return out


def _make_attempts(
    *,
    smoke_test: bool,
    methods: list[str],
    sectors: list[tuple[int, int]],
    order_boundary_pairs: list[tuple[str, str]],
    encodings: list[str],
    seeds: list[int],
    args: argparse.Namespace,
) -> list[AttemptConfig]:
    attempts: list[AttemptConfig] = []
    idx = 0
    if smoke_test:
        # SMOKE TEST — intentionally weak settings
        smoke_methods = methods
        smoke_seeds = [seeds[0]]
        smoke_pairs = [order_boundary_pairs[0]]
        smoke_encs = [encodings[0]]
        smoke_sectors = [sectors[0]]
        for method_id in smoke_methods:
            sector = smoke_sectors[0]
            ordering, boundary = smoke_pairs[0]
            boson_encoding = smoke_encs[0]
            seed = smoke_seeds[0]
            run_id = (
                f"smoke|{method_id}|S{sector[0]}_{sector[1]}|"
                f"{ordering}_{boundary}|{boson_encoding}|seed{seed}"
            )
            attempts.append(
                AttemptConfig(
                    run_id=run_id,
                    method_id=method_id,
                    sector_n_up=int(sector[0]),
                    sector_n_dn=int(sector[1]),
                    seed=int(seed),
                    attempt_idx=idx,
                    smoke_test=True,
                    L=int(args.L),
                    t=float(args.t),
                    U=float(args.U),
                    dv=float(args.dv),
                    omega0=float(args.omega0),
                    g_ep=float(args.g_ep),
                    n_ph_max=int(args.n_ph_max),
                    boson_encoding=str(boson_encoding),
                    ordering=str(ordering),
                    boundary=str(boundary),
                    vqe_reps=1,
                    vqe_restarts=1,
                    vqe_maxiter=min(200, int(args.vqe_maxiter)),
                    vqe_method=str(args.vqe_method),
                    adapt_max_depth=min(10, int(args.adapt_max_depth)),
                    adapt_eps_grad=float(args.adapt_eps_grad),
                    adapt_eps_energy=float(args.adapt_eps_energy),
                    adapt_maxiter=min(200, int(args.adapt_maxiter)),
                    paop_r=int(args.paop_r),
                    paop_split_paulis=bool(args.paop_split_paulis),
                    paop_prune_eps=float(args.paop_prune_eps),
                    paop_normalization=str(args.paop_normalization),
                    adapt_allow_repeats=bool(args.adapt_allow_repeats),
                    wallclock_cap_s=max(300, int(args.smoke_cap_s)),
                )
            )
            idx += 1
        return attempts

    for sector in sectors:
        for ordering, boundary in order_boundary_pairs:
            for boson_encoding in encodings:
                for method_id in methods:
                    for seed in seeds:
                        run_id = (
                            f"{method_id}|S{sector[0]}_{sector[1]}|"
                            f"{ordering}_{boundary}|{boson_encoding}|seed{seed}"
                        )
                        kind = METHOD_SPECS[method_id]["kind"]
                        cap_s = int(args.adapt_cap_s if kind == "adapt" else args.vqe_cap_s)
                        attempts.append(
                            AttemptConfig(
                                run_id=run_id,
                                method_id=method_id,
                                sector_n_up=int(sector[0]),
                                sector_n_dn=int(sector[1]),
                                seed=int(seed),
                                attempt_idx=idx,
                                smoke_test=False,
                                L=int(args.L),
                                t=float(args.t),
                                U=float(args.U),
                                dv=float(args.dv),
                                omega0=float(args.omega0),
                                g_ep=float(args.g_ep),
                                n_ph_max=int(args.n_ph_max),
                                boson_encoding=str(boson_encoding),
                                ordering=str(ordering),
                                boundary=str(boundary),
                                vqe_reps=int(args.vqe_reps),
                                vqe_restarts=int(args.vqe_restarts),
                                vqe_maxiter=int(args.vqe_maxiter),
                                vqe_method=str(args.vqe_method),
                                adapt_max_depth=int(args.adapt_max_depth),
                                adapt_eps_grad=float(args.adapt_eps_grad),
                                adapt_eps_energy=float(args.adapt_eps_energy),
                                adapt_maxiter=int(args.adapt_maxiter),
                                paop_r=int(args.paop_r),
                                paop_split_paulis=bool(args.paop_split_paulis),
                                paop_prune_eps=float(args.paop_prune_eps),
                                paop_normalization=str(args.paop_normalization),
                                adapt_allow_repeats=bool(args.adapt_allow_repeats),
                                wallclock_cap_s=max(60, cap_s),
                            )
                        )
                        idx += 1
    return attempts


def _write_smoke_sector_check(path: Path, args: argparse.Namespace, ordering: str, boundary: str, boson_encoding: str) -> None:
    cfg_base = {
        "L": int(args.L),
        "t": float(args.t),
        "U": float(args.U),
        "dv": float(args.dv),
        "omega0": float(args.omega0),
        "g_ep": float(args.g_ep),
        "n_ph_max": int(args.n_ph_max),
        "ordering": str(ordering),
        "boundary": str(boundary),
        "boson_encoding": str(boson_encoding),
    }
    h_poly = build_hubbard_holstein_hamiltonian(
        dims=cfg_base["L"],
        J=cfg_base["t"],
        U=cfg_base["U"],
        omega0=cfg_base["omega0"],
        g=cfg_base["g_ep"],
        n_ph_max=cfg_base["n_ph_max"],
        boson_encoding=cfg_base["boson_encoding"],
        repr_mode="JW",
        indexing=cfg_base["ordering"],
        pbc=(cfg_base["boundary"] == "periodic"),
        include_zero_point=True,
    )
    e_21 = exact_ground_energy_sector_hh(
        h_poly,
        num_sites=cfg_base["L"],
        num_particles=(2, 1),
        n_ph_max=cfg_base["n_ph_max"],
        boson_encoding=cfg_base["boson_encoding"],
        indexing=cfg_base["ordering"],
    )
    e_11 = exact_ground_energy_sector_hh(
        h_poly,
        num_sites=cfg_base["L"],
        num_particles=(1, 1),
        n_ph_max=cfg_base["n_ph_max"],
        boson_encoding=cfg_base["boson_encoding"],
        indexing=cfg_base["ordering"],
    )
    payload = {
        "checked_at_utc": _now_utc_iso(),
        "config": cfg_base,
        "E_exact_sector_(2,1)": float(e_21),
        "E_exact_sector_(1,1)": float(e_11),
        "different": bool(abs(float(e_21) - float(e_11)) > 1e-12),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_markdown_reports(rows: list[dict[str, Any]], out_dir: Path, top_n: int = 15) -> None:
    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    ok_rows = [r for r in rows if str(r.get("status")) == "ok" and r.get("delta_E_abs") is not None]
    ok_rows.sort(key=lambda r: float(r["delta_E_abs"]))

    ranked_path = summary_dir / "ranked_best_runs.md"
    lines: list[str] = []
    lines.append("# Ranked Best Runs")
    lines.append("")
    lines.append("| rank | method | sector | seed | ordering | boundary | encoding | E_best | E_exact | |ΔE| | runtime_s |")
    lines.append("|---:|---|---|---:|---|---|---|---:|---:|---:|---:|")
    for i, row in enumerate(ok_rows[: int(top_n)], start=1):
        lines.append(
            f"| {i} | {row['method_id']} | {row['sector']} | {row['seed']} | "
            f"{row['ordering']} | {row['boundary']} | {row['boson_encoding']} | "
            f"{float(row['E_best']):.12f} | {float(row['E_exact_sector']):.12f} | "
            f"{float(row['delta_E_abs']):.3e} | {float(row['runtime_s']):.1f} |"
        )
    ranked_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in ok_rows:
        by_method.setdefault(str(row["method_id"]), []).append(row)

    best_lines: list[str] = []
    best_lines.append("# Best Per Method")
    best_lines.append("")
    best_lines.append("| method_id | sector | seed | key_knobs | E_best | E_exact_sector | |ΔE| | runtime_s |")
    best_lines.append("|---|---|---:|---|---:|---:|---:|---:|")
    for method_id in METHOD_SPECS.keys():
        candidates = by_method.get(method_id, [])
        if len(candidates) == 0:
            best_lines.append(f"| {method_id} | - | - | - | - | - | - | - |")
            continue
        best = min(candidates, key=lambda r: float(r["delta_E_abs"]))
        if METHOD_SPECS[method_id]["kind"] == "conventional":
            key_knobs = (
                f"ansatz={best['ansatz_name']}, reps={best['vqe_reps']}, "
                f"restarts={best['vqe_restarts']}, maxiter={best['vqe_maxiter']}"
            )
        else:
            key_knobs = (
                f"pool={best['pool_name']}, depth={best['adapt_depth_reached']}, "
                f"max_depth={best['adapt_max_depth']}, maxiter={best['adapt_maxiter']}"
            )
        best_lines.append(
            f"| {method_id} | {best['sector']} | {best['seed']} | {key_knobs} | "
            f"{float(best['E_best']):.12f} | {float(best['E_exact_sector']):.12f} | "
            f"{float(best['delta_E_abs']):.3e} | {float(best['runtime_s']):.1f} |"
        )
    (summary_dir / "best_per_method.md").write_text("\n".join(best_lines) + "\n", encoding="utf-8")


def _run_attempt_batch(attempts: list[AttemptConfig], out_dir: Path) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = summary_dir / "runs_summary.csv"
    summary_jsonl = summary_dir / "runs_summary.jsonl"

    rows: list[dict[str, Any]] = []
    with summary_csv.open("w", encoding="utf-8", newline="") as f_csv, summary_jsonl.open("w", encoding="utf-8") as f_jsonl:
        writer = csv.DictWriter(f_csv, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        f_csv.flush()
        for i, cfg in enumerate(attempts, start=1):
            _ai_log(
                "benchmark_attempt_start",
                idx=i,
                total=len(attempts),
                run_id=cfg.run_id,
                method_id=cfg.method_id,
                sector=_sector_tuple_str(cfg.sector_n_up, cfg.sector_n_dn),
                seed=cfg.seed,
                ordering=cfg.ordering,
                boundary=cfg.boundary,
                boson_encoding=cfg.boson_encoding,
                smoke=cfg.smoke_test,
            )
            row = _execute_with_timeout(cfg)
            rows.append(row)
            writer.writerow({k: row.get(k, None) for k in SUMMARY_FIELDS})
            f_csv.flush()
            f_jsonl.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            f_jsonl.flush()
            _ai_log(
                "benchmark_attempt_done",
                idx=i,
                total=len(attempts),
                run_id=cfg.run_id,
                status=row.get("status"),
                delta_E_abs=row.get("delta_E_abs"),
                runtime_s=row.get("runtime_s"),
            )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Overnight L=3 HH benchmark: 4 methods × sectors × config grid with per-run timeout."
    )
    p.add_argument("--mode", choices=["smoke", "full", "smoke_then_full"], default="smoke_then_full")

    # Locked physics defaults from recent HH profile
    p.add_argument("--L", type=int, default=3)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--U", type=float, default=2.0)
    p.add_argument("--dv", type=float, default=0.0)
    p.add_argument("--omega0", type=float, default=1.0)
    p.add_argument("--g-ep", type=float, default=1.0)
    p.add_argument("--n-ph-max", type=int, default=1)

    p.add_argument(
        "--sectors",
        nargs="+",
        default=["2,1", "1,1"],
        help="Sector list as n_up,n_dn entries (default: 2,1 1,1).",
    )
    p.add_argument(
        "--ordering-boundary",
        nargs="+",
        default=["blocked,open", "interleaved,periodic", "interleaved,open"],
        help="Pairs ordering,boundary (default requested matrix).",
    )
    p.add_argument(
        "--boson-encodings",
        nargs="+",
        default=["binary", "unary"],
        help="Boson encodings to run if available.",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=list(METHOD_SPECS.keys()),
        help=f"Method ids (default all): {list(METHOD_SPECS.keys())}",
    )

    p.add_argument("--num-seeds", type=int, default=5)
    p.add_argument("--seed-start", type=int, default=101)

    # Full-run knobs (respecting HH L=3 minimums; rounded up)
    p.add_argument("--vqe-reps", type=int, default=2)
    p.add_argument("--vqe-restarts", type=int, default=5)
    p.add_argument("--vqe-maxiter", type=int, default=3000)
    p.add_argument("--vqe-method", choices=["COBYLA", "SLSQP", "L-BFGS-B"], default="COBYLA")

    p.add_argument("--adapt-max-depth", type=int, default=80)
    p.add_argument("--adapt-eps-grad", type=float, default=1e-6)
    p.add_argument("--adapt-eps-energy", type=float, default=1e-8)
    p.add_argument("--adapt-maxiter", type=int, default=3000)
    p.add_argument("--adapt-allow-repeats", action="store_true")
    p.add_argument("--adapt-no-repeats", dest="adapt_allow_repeats", action="store_false")
    p.set_defaults(adapt_allow_repeats=True)

    # PAOP build knobs
    p.add_argument("--paop-r", type=int, default=1)
    p.add_argument("--paop-split-paulis", action="store_true")
    p.add_argument("--paop-prune-eps", type=float, default=0.0)
    p.add_argument("--paop-normalization", choices=["none", "fro", "maxcoeff"], default="none")

    p.add_argument("--vqe-cap-s", type=int, default=1200, help="Per-attempt cap for conventional VQE runs.")
    p.add_argument("--adapt-cap-s", type=int, default=1800, help="Per-attempt cap for ADAPT runs.")
    p.add_argument("--smoke-cap-s", type=int, default=420)
    p.add_argument("--max-attempts", type=int, default=0, help="Optional truncation for debugging (0 means full).")

    p.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "overnight_l3_hh_4method",
        help="Root directory where timestamped run directory is created.",
    )
    p.add_argument("--tag", type=str, default="")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if int(args.L) != 3:
        raise ValueError("This runner is intended for L=3.")
    if int(args.vqe_reps) < 2:
        raise ValueError("HH L=3 minimum requires vqe_reps >= 2.")
    if int(args.vqe_restarts) < 4:
        raise ValueError("HH L=3 minimum requires vqe_restarts >= 4.")
    if int(args.vqe_maxiter) < 2400:
        raise ValueError("HH L=3 minimum requires vqe_maxiter >= 2400.")
    if str(args.vqe_method).upper() != "COBYLA":
        raise ValueError("HH L=3 minimum table specifies optimizer COBYLA.")

    methods = [m for m in args.methods if m in METHOD_SPECS]
    if len(methods) != 4:
        raise ValueError(f"Expected exactly four valid methods. Got: {methods}")

    sectors = _parse_sector_list(list(args.sectors))
    pairs = _build_ordering_boundary_pairs(list(args.ordering_boundary))
    encodings = [str(e).strip().lower() for e in args.boson_encodings]
    for enc in encodings:
        if enc not in {"binary", "unary"}:
            raise ValueError(f"Unsupported boson encoding '{enc}'.")

    seeds = [int(args.seed_start) + i for i in range(int(args.num_seeds))]
    tag = str(args.tag).strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"L3_hh_4method_{stamp}" if tag == "" else f"L3_hh_4method_{tag}_{stamp}"
    out_dir = Path(args.output_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "created_utc": _now_utc_iso(),
        "script": str(Path(__file__).resolve()),
        "mode": str(args.mode),
        "methods": methods,
        "sectors": sectors,
        "ordering_boundary": pairs,
        "boson_encodings": encodings,
        "seeds": seeds,
        "physics": {
            "L": int(args.L),
            "problem": "hh",
            "t": float(args.t),
            "U": float(args.U),
            "dv": float(args.dv),
            "omega0": float(args.omega0),
            "g_ep": float(args.g_ep),
            "n_ph_max": int(args.n_ph_max),
        },
        "knobs": {
            "vqe_reps": int(args.vqe_reps),
            "vqe_restarts": int(args.vqe_restarts),
            "vqe_maxiter": int(args.vqe_maxiter),
            "vqe_method": str(args.vqe_method),
            "adapt_max_depth": int(args.adapt_max_depth),
            "adapt_eps_grad": float(args.adapt_eps_grad),
            "adapt_eps_energy": float(args.adapt_eps_energy),
            "adapt_maxiter": int(args.adapt_maxiter),
            "vqe_cap_s": int(args.vqe_cap_s),
            "adapt_cap_s": int(args.adapt_cap_s),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _ai_log("overnight_runner_start", out_dir=str(out_dir), mode=str(args.mode))
    all_rows: list[dict[str, Any]] = []

    # Smoke-phase exact-sector sanity check
    _write_smoke_sector_check(
        out_dir / "summary" / "smoke_sector_filter_check.json",
        args,
        ordering=pairs[0][0],
        boundary=pairs[0][1],
        boson_encoding=encodings[0],
    )

    if args.mode in {"smoke", "smoke_then_full"}:
        smoke_attempts = _make_attempts(
            smoke_test=True,
            methods=methods,
            sectors=sectors,
            order_boundary_pairs=pairs,
            encodings=encodings,
            seeds=seeds,
            args=args,
        )
        _ai_log("smoke_phase_start", attempts=len(smoke_attempts))
        smoke_rows = _run_attempt_batch(smoke_attempts, out_dir / "smoke")
        all_rows.extend(smoke_rows)
        _write_markdown_reports(smoke_rows, out_dir / "smoke")
        _ai_log("smoke_phase_done", attempts=len(smoke_rows))
        if args.mode == "smoke":
            _ai_log("overnight_runner_done", out_dir=str(out_dir), total_rows=len(all_rows))
            return

    full_attempts = _make_attempts(
        smoke_test=False,
        methods=methods,
        sectors=sectors,
        order_boundary_pairs=pairs,
        encodings=encodings,
        seeds=seeds,
        args=args,
    )
    if int(args.max_attempts) > 0:
        full_attempts = full_attempts[: int(args.max_attempts)]
    _ai_log("full_phase_start", attempts=len(full_attempts))
    full_rows = _run_attempt_batch(full_attempts, out_dir / "full")
    all_rows.extend(full_rows)
    _write_markdown_reports(full_rows, out_dir / "full")

    # Morning deliverables as single combined summary + reports
    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    combined_csv = summary_dir / "runs_summary.csv"
    combined_jsonl = summary_dir / "runs_summary.jsonl"
    with combined_csv.open("w", encoding="utf-8", newline="") as f_csv, combined_jsonl.open("w", encoding="utf-8") as f_jsonl:
        writer = csv.DictWriter(f_csv, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, None) for k in SUMMARY_FIELDS})
            f_jsonl.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    _write_markdown_reports(all_rows, out_dir)

    _ai_log(
        "overnight_runner_done",
        out_dir=str(out_dir),
        total_rows=len(all_rows),
        ok_rows=sum(1 for r in all_rows if str(r.get("status")) == "ok"),
        error_rows=sum(1 for r in all_rows if str(r.get("status")) == "error"),
        timeout_rows=sum(1 for r in all_rows if str(r.get("status")) == "timeout"),
    )


if __name__ == "__main__":
    main()
