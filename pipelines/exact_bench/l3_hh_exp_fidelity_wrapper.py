#!/usr/bin/env python3
"""Wrapper-only L=3 HH fidelity gate benchmark.

DIAGNOSTIC ONLY (served role):
  - Evaluated on 2026-03-02 for L=3 HH plateau debugging.
  - Result: no meaningful convergence improvement toward near-zero DeltaE.
  - Keep for reproducibility/reference; do not treat as production benchmark path.

Purpose:
  - Keep existing repo files untouched.
  - Evaluate whether improved exp(-i theta * H_op) fidelity helps convergence.
  - Run a targeted gate matrix with early stall-stop heuristics.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
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
    apply_pauli_rotation,
    apply_pauli_string,
    exact_ground_energy_sector_hh,
    expval_pauli_polynomial,
    hubbard_holstein_reference_state,
)


METHOD_SPECS: dict[str, dict[str, str]] = {
    "m1_hh_hva": {"kind": "conventional", "ansatz": "hh_hva"},
    "m3_adapt_paop_std": {"kind": "adapt", "pool": "paop_std"},
    "m4_adapt_paop_lf_std": {"kind": "adapt", "pool": "paop_lf_std"},
}

DEFAULT_CONFIGS = ["interleaved,periodic,binary", "blocked,open,binary"]

RUN_FIELDS: list[str] = [
    "run_id",
    "status",
    "error_message",
    "started_utc",
    "finished_utc",
    "runtime_s",
    "phase",
    "decision",
    "method_id",
    "method_kind",
    "sector",
    "seed",
    "L",
    "problem",
    "t",
    "U",
    "dv",
    "omega0",
    "g_ep",
    "n_ph_max",
    "ordering",
    "boundary",
    "boson_encoding",
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
    "adapt_allow_repeats",
    "adapt_depth_reached",
    "adapt_stop_reason",
    "paop_r",
    "paop_prune_eps",
    "paop_normalization",
    "fidelity_grouping_mode",
    "fidelity_residual_trotter_steps",
    "fidelity_coeff_tolerance",
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
    phase: str
    decision: str
    method_id: str
    seed: int
    sector_n_up: int
    sector_n_dn: int
    L: int
    t: float
    U: float
    dv: float
    omega0: float
    g_ep: float
    n_ph_max: int
    ordering: str
    boundary: str
    boson_encoding: str
    vqe_reps: int
    vqe_restarts: int
    vqe_maxiter: int
    vqe_method: str
    adapt_max_depth: int
    adapt_eps_grad: float
    adapt_eps_energy: float
    adapt_maxiter: int
    adapt_allow_repeats: bool
    paop_r: int
    paop_prune_eps: float
    paop_normalization: str
    wallclock_cap_s: int


@dataclass(frozen=True)
class FidelityConfig:
    grouping_mode: str
    residual_trotter_steps: int
    coeff_tolerance: float


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ai_log(event: str, **fields: Any) -> None:
    payload = {"event": str(event), "ts_utc": _now_utc_iso(), **fields}
    print(f"AI_LOG {json.dumps(payload, sort_keys=True, default=str)}", flush=True)


def _parse_sector(raw: str) -> tuple[int, int]:
    s = str(raw).strip().replace("(", "").replace(")", "")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Invalid sector '{raw}'. Use n_up,n_dn.")
    return int(parts[0]), int(parts[1])


def _parse_config(raw: str) -> tuple[str, str, str]:
    parts = [p.strip().lower() for p in str(raw).split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(
            f"Invalid config '{raw}'. Use ordering,boundary,boson_encoding."
        )
    ordering, boundary, encoding = parts
    if ordering not in {"blocked", "interleaved"}:
        raise ValueError(f"Unsupported ordering '{ordering}'.")
    if boundary not in {"open", "periodic"}:
        raise ValueError(f"Unsupported boundary '{boundary}'.")
    if encoding not in {"binary", "unary"}:
        raise ValueError(f"Unsupported boson encoding '{encoding}'.")
    return ordering, boundary, encoding


def _normalize_state(psi: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(psi))
    if nrm <= 0.0:
        raise ValueError("Encountered zero-norm state.")
    return psi / nrm


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


def _try_import_scipy_minimize():
    try:
        from scipy.optimize import minimize  # type: ignore
    except Exception:
        minimize = None
    return minimize


def _poly_terms_real(
    poly: Any, *, coeff_tolerance: float, ignore_identity: bool = True
) -> tuple[str, list[tuple[str, float]]]:
    terms = list(poly.return_polynomial())
    if len(terms) == 0:
        return ("", [])
    nq = int(terms[0].nqubit())
    id_str = "e" * nq
    out: list[tuple[str, float]] = []
    for term in terms:
        ps = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(coeff_tolerance):
            continue
        if ignore_identity and ps == id_str:
            continue
        if abs(coeff.imag) > max(float(coeff_tolerance), 1e-10):
            raise ValueError(f"Non-negligible imaginary coefficient for term {ps}: {coeff}")
        out.append((ps, float(coeff.real)))
    return (id_str, out)


def _qwc_commute(ps_a: str, ps_b: str) -> bool:
    for a, b in zip(ps_a, ps_b):
        if a == "e" or b == "e" or a == b:
            continue
        return False
    return True


def _group_terms_qwc(terms: list[tuple[str, float]]) -> list[list[tuple[str, float]]]:
    ordered = sorted(
        terms, key=lambda x: sum(1 for ch in x[0] if ch != "e"), reverse=True
    )
    groups: list[list[tuple[str, float]]] = []
    for ps, coeff in ordered:
        placed = False
        for g in groups:
            if all(_qwc_commute(ps, other_ps) for other_ps, _ in g):
                g.append((ps, coeff))
                placed = True
                break
        if not placed:
            groups.append([(ps, coeff)])
    return groups


def _apply_group_exact(
    psi: np.ndarray,
    group: list[tuple[str, float]],
    theta: float,
) -> np.ndarray:
    out = np.array(psi, copy=True)
    for ps, coeff in group:
        angle = 2.0 * float(theta) * float(coeff)
        out = apply_pauli_rotation(out, ps, angle)
    return out


# Built-in math:
#   U(theta) ≈ [ Π_g exp(-i * (theta/2n) * G_g) Π_g^rev exp(-i * (theta/2n) * G_g) ]^n
def apply_exp_grouped_residual_trotter(
    psi: np.ndarray,
    poly: Any,
    theta: float,
    fidelity: FidelityConfig,
) -> np.ndarray:
    id_str, terms = _poly_terms_real(
        poly, coeff_tolerance=float(fidelity.coeff_tolerance), ignore_identity=True
    )
    if id_str == "" or len(terms) == 0:
        return np.array(psi, copy=True)
    if str(fidelity.grouping_mode) != "qubitwise_commute":
        raise ValueError(f"Unsupported grouping mode '{fidelity.grouping_mode}'.")
    groups = _group_terms_qwc(terms)
    steps = max(1, int(fidelity.residual_trotter_steps))
    delta = float(theta) / float(steps)
    out = np.array(psi, copy=True)
    for _ in range(steps):
        half = 0.5 * delta
        for group in groups:
            out = _apply_group_exact(out, group, half)
        for group in reversed(groups):
            out = _apply_group_exact(out, group, half)
    return out


def _prepare_layerwise_state_fidelity(
    ansatz: HubbardHolsteinLayerwiseAnsatz,
    theta: np.ndarray,
    psi_ref: np.ndarray,
    fidelity: FidelityConfig,
) -> np.ndarray:
    if int(theta.size) != int(ansatz.num_parameters):
        raise ValueError("theta has wrong length for HubbardHolsteinLayerwiseAnsatz.")
    psi = np.array(psi_ref, copy=True)
    k = 0
    for _ in range(int(ansatz.reps)):
        for op in ansatz.base_terms:
            psi = apply_exp_grouped_residual_trotter(
                psi,
                op.polynomial,
                float(theta[k]),
                fidelity,
            )
            k += 1
    return psi


def _apply_pauli_polynomial(state: np.ndarray, poly: Any) -> np.ndarray:
    terms = list(poly.return_polynomial())
    if len(terms) == 0:
        return np.zeros_like(state)
    nq = int(terms[0].nqubit())
    id_str = "e" * nq
    out = np.zeros_like(state)
    for term in terms:
        ps = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if abs(coeff) < 1e-15:
            continue
        if ps == id_str:
            out += coeff * state
        else:
            out += coeff * apply_pauli_string(state, ps)
    return out


def _commutator_gradient(h_poly: Any, op: AnsatzTerm, psi_current: np.ndarray) -> float:
    g_psi = _apply_pauli_polynomial(psi_current, op.polynomial)
    h_psi = _apply_pauli_polynomial(psi_current, h_poly)
    return float(2.0 * np.vdot(h_psi, g_psi).imag)


def _prepare_adapt_state_fidelity(
    psi_ref: np.ndarray,
    selected_ops: list[AnsatzTerm],
    theta: np.ndarray,
    fidelity: FidelityConfig,
) -> np.ndarray:
    psi = np.array(psi_ref, copy=True)
    for k, op in enumerate(selected_ops):
        psi = apply_exp_grouped_residual_trotter(
            psi,
            op.polynomial,
            float(theta[k]),
            fidelity,
        )
    return psi


def _try_minimize(
    objective: Any,
    x0: np.ndarray,
    method: str,
    maxiter: int,
) -> tuple[np.ndarray, float, int, int]:
    minimize = _try_import_scipy_minimize()
    if minimize is None:
        # Fallback coordinate search
        theta = np.asarray(x0, dtype=float)
        step = 0.2
        energy = float(objective(theta))
        nfev = 1
        nit = 0
        for it in range(int(maxiter)):
            improved = False
            for j in range(theta.size):
                for sgn in (+1.0, -1.0):
                    trial = theta.copy()
                    trial[j] += sgn * step
                    e_trial = float(objective(trial))
                    nfev += 1
                    if e_trial < energy:
                        energy = e_trial
                        theta = trial
                        improved = True
            nit = it + 1
            if not improved:
                step *= 0.5
                if step < 1e-6:
                    break
        return theta, float(energy), int(nfev), int(nit)

    res = minimize(
        objective,
        np.asarray(x0, dtype=float),
        method=str(method),
        options={"maxiter": int(maxiter)},
    )
    theta = np.asarray(res.x, dtype=float)
    return theta, float(res.fun), int(getattr(res, "nfev", 0)), int(getattr(res, "nit", 0))


def _run_vqe_layerwise_fidelity(
    h_poly: Any,
    ansatz: HubbardHolsteinLayerwiseAnsatz,
    psi_ref: np.ndarray,
    cfg: AttemptConfig,
    fidelity: FidelityConfig,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(cfg.seed))
    npar = int(ansatz.num_parameters)
    if npar <= 0:
        raise ValueError("ansatz has no parameters")

    def objective(x: np.ndarray) -> float:
        psi = _prepare_layerwise_state_fidelity(
            ansatz, np.asarray(x, dtype=float), psi_ref, fidelity
        )
        return float(expval_pauli_polynomial(psi, h_poly))

    best_energy = float("inf")
    best_theta: np.ndarray | None = None
    nfev_total = 0
    nit_total = 0
    restart_energies: list[float] = []

    for _r in range(int(cfg.vqe_restarts)):
        x0 = 0.3 * rng.normal(size=npar)
        theta_opt, energy, nfev, nit = _try_minimize(
            objective, x0, str(cfg.vqe_method), int(cfg.vqe_maxiter)
        )
        restart_energies.append(float(energy))
        nfev_total += int(nfev)
        nit_total += int(nit)
        if float(energy) < best_energy:
            best_energy = float(energy)
            best_theta = np.asarray(theta_opt, dtype=float)

    if best_theta is None:
        raise RuntimeError("No VQE restart completed.")
    psi_best = _prepare_layerwise_state_fidelity(ansatz, best_theta, psi_ref, fidelity)
    psi_best = _normalize_state(np.asarray(psi_best, dtype=complex).reshape(-1))
    return {
        "E_best": float(best_energy),
        "E_last": float(restart_energies[-1]),
        "num_parameters": int(npar),
        "nfev": int(nfev_total),
        "nit": int(nit_total),
        "psi_best": psi_best.tolist(),
    }


def _run_adapt_fidelity(
    h_poly: Any,
    psi_ref: np.ndarray,
    pool: list[AnsatzTerm],
    cfg: AttemptConfig,
    fidelity: FidelityConfig,
) -> dict[str, Any]:
    minimize = _try_import_scipy_minimize()
    if minimize is None:
        raise RuntimeError("SciPy minimize is required for ADAPT re-optimization in fidelity wrapper.")
    if len(pool) == 0:
        raise RuntimeError("Empty ADAPT pool.")

    selected_ops: list[AnsatzTerm] = []
    theta = np.zeros(0, dtype=float)
    available = set(range(len(pool)))
    allow_repeats = bool(cfg.adapt_allow_repeats)

    energy_current = float(expval_pauli_polynomial(psi_ref, h_poly))
    history: list[float] = [float(energy_current)]
    nfev_total = 1
    stop_reason = "max_depth"

    for _depth in range(int(cfg.adapt_max_depth)):
        psi_current = _prepare_adapt_state_fidelity(psi_ref, selected_ops, theta, fidelity)
        gradients = np.zeros(len(pool), dtype=float)
        indices = range(len(pool)) if allow_repeats else sorted(available)
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
            psi = _prepare_adapt_state_fidelity(
                psi_ref, selected_ops, np.asarray(x, dtype=float), fidelity
            )
            return float(expval_pauli_polynomial(psi, h_poly))

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

    psi_best = _prepare_adapt_state_fidelity(psi_ref, selected_ops, theta, fidelity)
    psi_best = _normalize_state(np.asarray(psi_best, dtype=complex).reshape(-1))
    return {
        "E_best": float(min(history)),
        "E_last": float(history[-1]),
        "adapt_depth_reached": int(len(selected_ops)),
        "adapt_stop_reason": str(stop_reason),
        "num_parameters": int(theta.size),
        "pool_size": int(len(pool)),
        "nfev": int(nfev_total),
        "nit": int(max(0, len(history) - 1)),
        "psi_best": psi_best.tolist(),
    }


def _fermion_spin_qubit_index_sets(num_sites: int, ordering: str) -> tuple[list[int], list[int]]:
    n_sites_i = int(num_sites)
    order = str(ordering).strip().lower()
    if order == "blocked":
        return list(range(n_sites_i)), list(range(n_sites_i, 2 * n_sites_i))
    if order == "interleaved":
        return list(range(0, 2 * n_sites_i, 2)), list(range(1, 2 * n_sites_i, 2))
    raise ValueError(f"Unsupported ordering '{ordering}'.")


def _particle_number_expectation(psi: np.ndarray, qubits: list[int]) -> float:
    vec = np.asarray(psi, dtype=complex).reshape(-1)
    probs = np.abs(vec) ** 2
    norm = float(np.sum(probs))
    if norm <= 0.0:
        raise ValueError("Invalid state norm in particle-number diagnostic.")
    probs = probs / norm
    basis = np.arange(probs.size, dtype=np.uint64)
    out = 0.0
    for q in qubits:
        bits = ((basis >> np.uint64(int(q))) & np.uint64(1)).astype(np.float64)
        out += float(np.dot(probs, bits))
    return float(out)


def _sector_particle_diagnostics(psi: np.ndarray, cfg: AttemptConfig) -> dict[str, Any]:
    up_q, dn_q = _fermion_spin_qubit_index_sets(int(cfg.L), str(cfg.ordering))
    n_up_t = float(cfg.sector_n_up)
    n_dn_t = float(cfg.sector_n_dn)
    n_up_e = _particle_number_expectation(psi, up_q)
    n_dn_e = _particle_number_expectation(psi, dn_q)
    n_up_err = abs(n_up_e - n_up_t)
    n_dn_err = abs(n_dn_e - n_dn_t)
    total_err = n_up_err + n_dn_err
    return {
        "N_up_target": int(cfg.sector_n_up),
        "N_dn_target": int(cfg.sector_n_dn),
        "N_up_expect": float(n_up_e),
        "N_dn_expect": float(n_dn_e),
        "N_up_abs_err": float(n_up_err),
        "N_dn_abs_err": float(n_dn_err),
        "N_sector_abs_err_sum": float(total_err),
        "sector_leak_flag": bool(total_err > 1e-6),
    }


def _run_attempt_inner(cfg: AttemptConfig, fidelity: FidelityConfig) -> dict[str, Any]:
    h_poly = _build_hh_hamiltonian(cfg)
    e_exact = _exact_sector_energy(h_poly, cfg)
    psi_ref = _reference_state(cfg)

    out: dict[str, Any] = {
        "ansatz_name": "",
        "pool_name": "",
        "adapt_depth_reached": None,
        "adapt_stop_reason": "",
        "num_parameters": None,
        "pool_size": None,
        "nfev": None,
        "nit": None,
    }

    spec = METHOD_SPECS[str(cfg.method_id)]
    psi_final: np.ndarray | None = None

    if spec["kind"] == "conventional":
        if spec.get("ansatz") != "hh_hva":
            raise ValueError(f"Unsupported conventional ansatz mapping for {cfg.method_id}")
        ansatz = HubbardHolsteinLayerwiseAnsatz(
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
        vqe = _run_vqe_layerwise_fidelity(h_poly, ansatz, psi_ref, cfg, fidelity)
        out.update(
            {
                "ansatz_name": "hh_hva",
                "E_best": float(vqe["E_best"]),
                "E_last": float(vqe["E_last"]),
                "num_parameters": int(vqe["num_parameters"]),
                "nfev": int(vqe["nfev"]),
                "nit": int(vqe["nit"]),
            }
        )
        psi_final = _normalize_state(np.asarray(vqe["psi_best"], dtype=complex).reshape(-1))
    elif spec["kind"] == "adapt":
        pool_name = str(spec["pool"])
        pool_specs = make_paop_pool(
            pool_name,
            num_sites=int(cfg.L),
            num_particles=(int(cfg.sector_n_up), int(cfg.sector_n_dn)),
            n_ph_max=int(cfg.n_ph_max),
            boson_encoding=str(cfg.boson_encoding),
            ordering=str(cfg.ordering),
            boundary=str(cfg.boundary),
            paop_r=int(cfg.paop_r),
            paop_split_paulis=False,
            paop_prune_eps=float(cfg.paop_prune_eps),
            paop_normalization=str(cfg.paop_normalization),
        )
        pool = [AnsatzTerm(label=str(lbl), polynomial=poly) for lbl, poly in pool_specs]
        adapt = _run_adapt_fidelity(h_poly, psi_ref, pool, cfg, fidelity)
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
        raise RuntimeError("Internal error: missing final state.")
    out["E_exact_sector"] = float(e_exact)
    out["delta_E_abs"] = float(abs(float(out["E_best"]) - float(e_exact)))
    out.update(_sector_particle_diagnostics(psi_final, cfg))
    return out


def _attempt_worker(cfg: AttemptConfig, fidelity: FidelityConfig, queue: mp.Queue) -> None:
    started = _now_utc_iso()
    try:
        payload = _run_attempt_inner(cfg, fidelity)
        queue.put({"ok": True, "started_utc": started, "payload": payload})
    except Exception:
        queue.put(
            {
                "ok": False,
                "started_utc": started,
                "error_message": traceback.format_exc(limit=8),
            }
        )


def _execute_with_timeout(cfg: AttemptConfig, fidelity: FidelityConfig) -> dict[str, Any]:
    queue: mp.Queue = mp.Queue()
    process = mp.Process(target=_attempt_worker, args=(cfg, fidelity, queue))
    started_wall = time.perf_counter()
    process.start()
    process.join(timeout=float(cfg.wallclock_cap_s))
    runtime_s = float(time.perf_counter() - started_wall)

    row: dict[str, Any] = {
        "run_id": str(cfg.run_id),
        "status": "error",
        "error_message": "",
        "started_utc": _now_utc_iso(),
        "finished_utc": _now_utc_iso(),
        "runtime_s": runtime_s,
        "phase": str(cfg.phase),
        "decision": str(cfg.decision),
        "method_id": str(cfg.method_id),
        "method_kind": str(METHOD_SPECS[str(cfg.method_id)]["kind"]),
        "sector": f"({cfg.sector_n_up},{cfg.sector_n_dn})",
        "seed": int(cfg.seed),
        "L": int(cfg.L),
        "problem": "hh",
        "t": float(cfg.t),
        "U": float(cfg.U),
        "dv": float(cfg.dv),
        "omega0": float(cfg.omega0),
        "g_ep": float(cfg.g_ep),
        "n_ph_max": int(cfg.n_ph_max),
        "ordering": str(cfg.ordering),
        "boundary": str(cfg.boundary),
        "boson_encoding": str(cfg.boson_encoding),
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
        "adapt_allow_repeats": bool(cfg.adapt_allow_repeats),
        "adapt_depth_reached": None,
        "adapt_stop_reason": "",
        "paop_r": int(cfg.paop_r),
        "paop_prune_eps": float(cfg.paop_prune_eps),
        "paop_normalization": str(cfg.paop_normalization),
        "fidelity_grouping_mode": str(fidelity.grouping_mode),
        "fidelity_residual_trotter_steps": int(fidelity.residual_trotter_steps),
        "fidelity_coeff_tolerance": float(fidelity.coeff_tolerance),
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
        row["status"] = "timeout"
        row["error_message"] = f"wallclock cap exceeded ({cfg.wallclock_cap_s}s)"
        row["finished_utc"] = _now_utc_iso()
        return row

    message: dict[str, Any] | None = None
    try:
        if not queue.empty():
            message = queue.get_nowait()
    except Exception:
        message = None

    if message is None:
        row["status"] = "error"
        row["error_message"] = "worker exited without payload"
        row["finished_utc"] = _now_utc_iso()
        return row

    row["started_utc"] = str(message.get("started_utc", row["started_utc"]))
    row["finished_utc"] = _now_utc_iso()
    if bool(message.get("ok", False)):
        payload = message["payload"]
        row.update(payload)
        row["status"] = "ok"
        row["error_message"] = ""
    else:
        row["status"] = "error"
        row["error_message"] = str(message.get("error_message", "unknown error"))
    return row


def _safe_float(x: Any) -> float | None:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any) -> int | None:
    try:
        if x is None or x == "":
            return None
        return int(x)
    except Exception:
        return None


def _small_improvement_stall(best_by_rung: list[float], eps: float) -> bool:
    if len(best_by_rung) < 3:
        return False
    d1 = best_by_rung[-3] - best_by_rung[-2]
    d2 = best_by_rung[-2] - best_by_rung[-1]
    return (d1 < float(eps)) and (d2 < float(eps))


def _adapt_plateau(last_ok_rows: list[dict[str, Any]], eps_e: float = 1e-5) -> bool:
    if len(last_ok_rows) < 3:
        return False
    tail = last_ok_rows[-3:]
    depths = [_safe_int(r.get("adapt_depth_reached")) for r in tail]
    if any(d is None for d in depths):
        return False
    if len(set(depths)) != 1:
        return False
    energies = [_safe_float(r.get("E_best")) for r in tail]
    if any(e is None for e in energies):
        return False
    return (max(energies) - min(energies)) < float(eps_e)


def _write_run_summary_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    by_method: dict[str, list[dict[str, Any]]] = {}
    ok_rows = [
        r
        for r in rows
        if str(r.get("status")) == "ok" and _safe_float(r.get("delta_E_abs")) is not None
    ]
    for row in ok_rows:
        by_method.setdefault(str(row["method_id"]), []).append(row)

    lines: list[str] = []
    lines.append("# Fidelity Gate Summary")
    lines.append("")
    lines.append("| method | config | seed | trotter | E_best | E_exact | |ΔE| | runtime_s |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for method in METHOD_SPECS.keys():
        candidates = by_method.get(method, [])
        if len(candidates) == 0:
            lines.append(f"| {method} | - | - | - | - | - | - | - |")
            continue
        best = min(candidates, key=lambda r: float(r["delta_E_abs"]))
        cfg = f"{best['ordering']},{best['boundary']},{best['boson_encoding']}"
        lines.append(
            f"| {method} | {cfg} | {best['seed']} | {best['fidelity_residual_trotter_steps']} | "
            f"{float(best['E_best']):.12f} | {float(best['E_exact_sector']):.12f} | "
            f"{float(best['delta_E_abs']):.3e} | {float(best['runtime_s']):.1f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wrapper-only L=3 HH exp fidelity gate.")
    p.add_argument("--L", type=int, default=3)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--U", type=float, default=2.0)
    p.add_argument("--dv", type=float, default=0.0)
    p.add_argument("--omega0", type=float, default=1.0)
    p.add_argument("--g-ep", type=float, default=1.0)
    p.add_argument("--n-ph-max", type=int, default=1)
    p.add_argument("--sector", type=str, default="2,1")
    p.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    p.add_argument("--methods", nargs="+", default=list(METHOD_SPECS.keys()))

    p.add_argument("--seed-start", type=int, default=101)
    p.add_argument("--base-seeds", type=int, default=2)
    p.add_argument("--extra-seeds", type=int, default=1)

    p.add_argument("--fidelity-rungs", nargs="+", type=int, default=[1, 2, 4])
    p.add_argument("--grouping-mode", choices=["qubitwise_commute"], default="qubitwise_commute")
    p.add_argument("--fidelity-coeff-tolerance", type=float, default=1e-12)

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

    p.add_argument("--paop-r", type=int, default=1)
    p.add_argument("--paop-prune-eps", type=float, default=0.0)
    p.add_argument("--paop-normalization", choices=["none", "fro", "maxcoeff"], default="none")

    p.add_argument("--vqe-cap-s", type=int, default=600)
    p.add_argument("--adapt-cap-s", type=int, default=720)
    p.add_argument("--max-total-wallclock-s", type=int, default=3600)
    p.add_argument("--stall-improve-eps", type=float, default=1e-2)
    p.add_argument("--stall-time-gain-eps", type=float, default=5e-3)
    p.add_argument("--success-delta", type=float, default=1e-3)
    p.add_argument("--seed-escalation-gain-threshold", type=float, default=5e-3)

    p.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "l3_hh_exp_fidelity_gate",
    )
    p.add_argument("--tag", type=str, default="run")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if int(args.L) != 3:
        raise ValueError("This gate is intended for L=3.")
    if str(args.vqe_method).upper() != "COBYLA":
        raise ValueError("Use COBYLA for L=3 HH consistency.")
    if int(args.vqe_reps) < 2:
        raise ValueError("Require vqe_reps >= 2 for HH L=3.")
    if int(args.vqe_restarts) < 4:
        raise ValueError("Require vqe_restarts >= 4 for HH L=3.")
    if int(args.vqe_maxiter) < 2400:
        raise ValueError("Require vqe_maxiter >= 2400 for HH L=3.")
    if int(args.adapt_maxiter) < 2400:
        raise ValueError("Require adapt_maxiter >= 2400 for HH L=3.")

    methods = [m for m in args.methods if m in METHOD_SPECS]
    if sorted(methods) != sorted(list(METHOD_SPECS.keys())):
        raise ValueError(f"methods must be exactly {list(METHOD_SPECS.keys())}. Got {methods}")
    configs = [_parse_config(c) for c in args.configs]
    n_up, n_dn = _parse_sector(str(args.sector))

    base_seed_list = [int(args.seed_start) + i for i in range(int(args.base_seeds))]
    extra_seed_list = [
        int(args.seed_start) + int(args.base_seeds) + i for i in range(int(args.extra_seeds))
    ]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_root) / f"L3_hh_exp_fidelity_{args.tag}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "created_utc": _now_utc_iso(),
        "script": str(Path(__file__).resolve()),
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
        "sector": [int(n_up), int(n_dn)],
        "methods": methods,
        "configs": [",".join(c) for c in configs],
        "base_seeds": base_seed_list,
        "extra_seeds": extra_seed_list,
        "fidelity_rungs": [int(x) for x in args.fidelity_rungs],
        "fidelity_grouping_mode": str(args.grouping_mode),
        "fidelity_coeff_tolerance": float(args.fidelity_coeff_tolerance),
        "knobs": {
            "vqe_reps": int(args.vqe_reps),
            "vqe_restarts": int(args.vqe_restarts),
            "vqe_maxiter": int(args.vqe_maxiter),
            "adapt_max_depth": int(args.adapt_max_depth),
            "adapt_maxiter": int(args.adapt_maxiter),
            "adapt_allow_repeats": bool(args.adapt_allow_repeats),
            "paop_r": int(args.paop_r),
            "paop_prune_eps": float(args.paop_prune_eps),
            "paop_normalization": str(args.paop_normalization),
            "vqe_cap_s": int(args.vqe_cap_s),
            "adapt_cap_s": int(args.adapt_cap_s),
            "max_total_wallclock_s": int(args.max_total_wallclock_s),
            "stall_improve_eps": float(args.stall_improve_eps),
            "stall_time_gain_eps": float(args.stall_time_gain_eps),
            "success_delta": float(args.success_delta),
            "seed_escalation_gain_threshold": float(args.seed_escalation_gain_threshold),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _ai_log("exp_fidelity_gate_start", out_dir=str(out_dir))
    t_start = time.perf_counter()

    rows: list[dict[str, Any]] = []
    csv_path = summary_dir / "fidelity_gate_runs.csv"
    jsonl_path = summary_dir / "fidelity_gate_runs.jsonl"
    branch_summary: list[dict[str, Any]] = []

    with csv_path.open("w", encoding="utf-8", newline="") as f_csv, jsonl_path.open(
        "w", encoding="utf-8"
    ) as f_jsonl:
        writer = csv.DictWriter(f_csv, fieldnames=RUN_FIELDS)
        writer.writeheader()
        f_csv.flush()

        for ordering, boundary, boson_encoding in configs:
            for method_id in methods:
                branch_key = f"{method_id}|{ordering},{boundary},{boson_encoding}"
                best_by_rung: list[float] = []
                runtime_by_rung: list[float] = []
                recent_ok: list[dict[str, Any]] = []
                prev_best: float | None = None
                branch_stop = "max_rungs_reached"
                escalate_used = False

                for rung_steps in args.fidelity_rungs:
                    elapsed = float(time.perf_counter() - t_start)
                    if elapsed > float(args.max_total_wallclock_s):
                        branch_stop = "global_wallclock_cap"
                        break

                    rung_name = f"trotter_{int(rung_steps)}"
                    rung_rows: list[dict[str, Any]] = []
                    seed_list = list(base_seed_list)
                    decision = "continue"

                    for seed in seed_list:
                        elapsed = float(time.perf_counter() - t_start)
                        if elapsed > float(args.max_total_wallclock_s):
                            decision = "global_wallclock_cap"
                            break

                        cfg = AttemptConfig(
                            run_id=(
                                f"{method_id}|{rung_name}|S{n_up}_{n_dn}|"
                                f"{ordering}_{boundary}|{boson_encoding}|seed{seed}"
                            ),
                            phase="fidelity_gate",
                            decision=decision,
                            method_id=str(method_id),
                            seed=int(seed),
                            sector_n_up=int(n_up),
                            sector_n_dn=int(n_dn),
                            L=int(args.L),
                            t=float(args.t),
                            U=float(args.U),
                            dv=float(args.dv),
                            omega0=float(args.omega0),
                            g_ep=float(args.g_ep),
                            n_ph_max=int(args.n_ph_max),
                            ordering=str(ordering),
                            boundary=str(boundary),
                            boson_encoding=str(boson_encoding),
                            vqe_reps=int(args.vqe_reps),
                            vqe_restarts=int(args.vqe_restarts),
                            vqe_maxiter=int(args.vqe_maxiter),
                            vqe_method=str(args.vqe_method),
                            adapt_max_depth=int(args.adapt_max_depth),
                            adapt_eps_grad=float(args.adapt_eps_grad),
                            adapt_eps_energy=float(args.adapt_eps_energy),
                            adapt_maxiter=int(args.adapt_maxiter),
                            adapt_allow_repeats=bool(args.adapt_allow_repeats),
                            paop_r=int(args.paop_r),
                            paop_prune_eps=float(args.paop_prune_eps),
                            paop_normalization=str(args.paop_normalization),
                            wallclock_cap_s=int(
                                args.adapt_cap_s
                                if METHOD_SPECS[method_id]["kind"] == "adapt"
                                else args.vqe_cap_s
                            ),
                        )
                        fidelity = FidelityConfig(
                            grouping_mode=str(args.grouping_mode),
                            residual_trotter_steps=int(rung_steps),
                            coeff_tolerance=float(args.fidelity_coeff_tolerance),
                        )
                        _ai_log(
                            "fidelity_attempt_start",
                            branch=branch_key,
                            method_id=method_id,
                            rung=rung_name,
                            seed=seed,
                        )
                        row = _execute_with_timeout(cfg, fidelity)
                        rows.append(row)
                        rung_rows.append(row)
                        writer.writerow({k: row.get(k, None) for k in RUN_FIELDS})
                        f_csv.flush()
                        f_jsonl.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                        f_jsonl.flush()
                        _ai_log(
                            "fidelity_attempt_done",
                            branch=branch_key,
                            method_id=method_id,
                            rung=rung_name,
                            seed=seed,
                            status=row.get("status"),
                            delta_E_abs=row.get("delta_E_abs"),
                            runtime_s=row.get("runtime_s"),
                        )

                    ok_rows = [
                        r
                        for r in rung_rows
                        if str(r.get("status")) == "ok"
                        and _safe_float(r.get("delta_E_abs")) is not None
                    ]
                    best_delta = (
                        min(float(r["delta_E_abs"]) for r in ok_rows)
                        if len(ok_rows) > 0
                        else None
                    )
                    med_runtime = (
                        float(statistics.median(float(r["runtime_s"]) for r in ok_rows))
                        if len(ok_rows) > 0
                        else None
                    )

                    if best_delta is not None:
                        best_by_rung.append(float(best_delta))
                        recent_ok.extend(ok_rows)
                        if med_runtime is not None:
                            runtime_by_rung.append(float(med_runtime))

                    # Optional one-time seed escalation when trend is improving.
                    if (
                        not escalate_used
                        and len(extra_seed_list) > 0
                        and prev_best is not None
                        and best_delta is not None
                        and (prev_best - best_delta) >= float(args.seed_escalation_gain_threshold)
                    ):
                        escalate_used = True
                        for seed in extra_seed_list:
                            cfg = AttemptConfig(
                                run_id=(
                                    f"{method_id}|{rung_name}|S{n_up}_{n_dn}|"
                                    f"{ordering}_{boundary}|{boson_encoding}|seed{seed}"
                                ),
                                phase="fidelity_gate_extra_seed",
                                decision="extra_seed",
                                method_id=str(method_id),
                                seed=int(seed),
                                sector_n_up=int(n_up),
                                sector_n_dn=int(n_dn),
                                L=int(args.L),
                                t=float(args.t),
                                U=float(args.U),
                                dv=float(args.dv),
                                omega0=float(args.omega0),
                                g_ep=float(args.g_ep),
                                n_ph_max=int(args.n_ph_max),
                                ordering=str(ordering),
                                boundary=str(boundary),
                                boson_encoding=str(boson_encoding),
                                vqe_reps=int(args.vqe_reps),
                                vqe_restarts=int(args.vqe_restarts),
                                vqe_maxiter=int(args.vqe_maxiter),
                                vqe_method=str(args.vqe_method),
                                adapt_max_depth=int(args.adapt_max_depth),
                                adapt_eps_grad=float(args.adapt_eps_grad),
                                adapt_eps_energy=float(args.adapt_eps_energy),
                                adapt_maxiter=int(args.adapt_maxiter),
                                adapt_allow_repeats=bool(args.adapt_allow_repeats),
                                paop_r=int(args.paop_r),
                                paop_prune_eps=float(args.paop_prune_eps),
                                paop_normalization=str(args.paop_normalization),
                                wallclock_cap_s=int(
                                    args.adapt_cap_s
                                    if METHOD_SPECS[method_id]["kind"] == "adapt"
                                    else args.vqe_cap_s
                                ),
                            )
                            fidelity = FidelityConfig(
                                grouping_mode=str(args.grouping_mode),
                                residual_trotter_steps=int(rung_steps),
                                coeff_tolerance=float(args.fidelity_coeff_tolerance),
                            )
                            row = _execute_with_timeout(cfg, fidelity)
                            rows.append(row)
                            writer.writerow({k: row.get(k, None) for k in RUN_FIELDS})
                            f_csv.flush()
                            f_jsonl.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                            f_jsonl.flush()
                            if (
                                str(row.get("status")) == "ok"
                                and _safe_float(row.get("delta_E_abs")) is not None
                            ):
                                recent_ok.append(row)
                                d = float(row["delta_E_abs"])
                                if best_delta is None or d < best_delta:
                                    best_delta = d

                    if best_delta is not None and best_delta <= float(args.success_delta):
                        decision = "stop_success_gate"
                    elif len(best_by_rung) >= 3 and _small_improvement_stall(
                        best_by_rung, float(args.stall_improve_eps)
                    ):
                        decision = "stop_small_improvement_twice"
                    elif (
                        len(best_by_rung) >= 2
                        and len(runtime_by_rung) >= 2
                        and runtime_by_rung[-1] > 2.0 * runtime_by_rung[-2]
                        and (best_by_rung[-2] - best_by_rung[-1]) < float(args.stall_time_gain_eps)
                    ):
                        decision = "stop_runtime_inefficient"
                    elif method_id.startswith("m3") or method_id.startswith("m4"):
                        if _adapt_plateau(recent_ok):
                            decision = "stop_adapt_plateau"

                    if best_delta is not None:
                        prev_best = float(best_delta)

                    if decision.startswith("stop_") or decision == "global_wallclock_cap":
                        branch_stop = decision
                        break

                branch_summary.append(
                    {
                        "branch": branch_key,
                        "method_id": method_id,
                        "config": f"{ordering},{boundary},{boson_encoding}",
                        "stop_reason": branch_stop,
                        "best_delta_seen": (
                            min(best_by_rung) if len(best_by_rung) > 0 else None
                        ),
                        "rungs_completed": len(best_by_rung),
                    }
                )

            elapsed = float(time.perf_counter() - t_start)
            if elapsed > float(args.max_total_wallclock_s):
                _ai_log("exp_fidelity_gate_global_stop", elapsed_s=elapsed)
                break

    summary_json = {
        "created_utc": _now_utc_iso(),
        "out_dir": str(out_dir),
        "total_rows": len(rows),
        "branch_summary": branch_summary,
    }
    (summary_dir / "fidelity_gate_summary.json").write_text(
        json.dumps(summary_json, indent=2), encoding="utf-8"
    )
    _write_run_summary_markdown(summary_dir / "fidelity_gate_summary.md", rows)
    _ai_log("exp_fidelity_gate_done", out_dir=str(out_dir), total_rows=len(rows))


if __name__ == "__main__":
    main()
