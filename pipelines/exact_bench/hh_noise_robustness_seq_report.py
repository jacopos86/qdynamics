#!/usr/bin/env python3
"""HH noise-robustness sequential report (stage-transition + Magnus/Trotter).

Pipeline sequence:
1) HVA warm-start VQE
2) ADAPT-VQE with Pool B strict union: UCCSD_lifted + HVA + PAOP_FULL
3) Conventional VQE seeded from ADAPT final state

Then audits noiseless dynamics (Suzuki-2, midpoint Magnus-2, CFQM4/CFQM6, exact)
and noisy dynamics for selected methods (default: cfqm4,suzuki2) under
ideal/shots/aer_noise for static + drive profiles.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter

from reports.pdf_utils import (
    HAS_MATPLOTLIB,
    current_command_string,
    get_PdfPages,
    get_plt,
    render_command_page,
    render_compact_table,
    render_parameter_manifest,
    render_text_page,
    require_matplotlib,
)
from src.quantum.drives_time_potential import (
    build_gaussian_sinusoid_density_drive,
    evaluate_drive_waveform,
    reference_method_name,
)
from src.quantum.hartree_fock_reference_state import hubbard_holstein_reference_state
from src.quantum.hubbard_latex_python_pairs import (
    SPIN_DN,
    SPIN_UP,
    boson_qubits_per_site,
    build_hubbard_holstein_hamiltonian,
    mode_index,
)
from src.quantum.operator_pools import make_pool as make_paop_pool
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.time_propagation.cfqm_schemes import get_cfqm_scheme
from src.quantum.vqe_latex_python_pairs import (
    AnsatzTerm,
    HardcodedUCCSDAnsatz,
    HubbardHolsteinPhysicalTermwiseAnsatz,
    apply_exp_pauli_polynomial,
    apply_pauli_string,
    exact_ground_energy_sector_hh,
    expval_pauli_polynomial,
    hamiltonian_matrix,
    vqe_minimize,
)

from pipelines.exact_bench.hh_seq_transition_utils import (
    TransitionConfig,
    TransitionState,
    build_pool_b_strict_union,
    build_time_dependent_sparse_qop,
    flatten_coeff_map_real_imag,
    summarize_transition,
    update_transition_state,
)
from pipelines.exact_bench.noise_oracle_runtime import (
    ExpectationOracle,
    OracleConfig,
    _append_reference_state,
    _doublon_site_qop,
    _number_operator_qop,
)
from pipelines.hardcoded import hubbard_pipeline as hc_pipeline


def _ai_log(event: str, **fields: Any) -> None:
    payload = {
        "event": str(event),
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    print(f"AI_LOG {json.dumps(payload, sort_keys=True, default=str)}", flush=True)


_HH_MINIMUMS: dict[tuple[int, int], dict[str, Any]] = {
    (2, 1): {"trotter_steps": 64, "reps": 2, "restarts": 3, "maxiter": 800, "method": "COBYLA"},
    (2, 2): {"trotter_steps": 128, "reps": 3, "restarts": 4, "maxiter": 1500, "method": "COBYLA"},
    (3, 1): {"trotter_steps": 192, "reps": 2, "restarts": 4, "maxiter": 2400, "method": "COBYLA"},
}

_NOISY_METHODS_ALLOWED = {"suzuki2", "cfqm4", "cfqm6"}


def _half_filled_particles(num_sites: int) -> tuple[int, int]:
    return ((int(num_sites) + 1) // 2, int(num_sites) // 2)


def _collect_hardcoded_terms_exyz(poly: Any, tol: float = 1e-12) -> tuple[list[str], dict[str, complex]]:
    coeff_map: dict[str, complex] = {}
    order: list[str] = []
    for term in poly.return_polynomial():
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        if label not in coeff_map:
            coeff_map[label] = 0.0 + 0.0j
            order.append(label)
        coeff_map[label] += coeff
    cleaned_order = [lbl for lbl in order if abs(coeff_map[lbl]) > float(tol)]
    cleaned_map = {lbl: coeff_map[lbl] for lbl in cleaned_order}
    return cleaned_order, cleaned_map


def _parse_noisy_methods_csv(raw: str) -> list[str]:
    vals: list[str] = []
    for tok in str(raw).split(","):
        t = str(tok).strip().lower()
        if not t:
            continue
        if t not in _NOISY_METHODS_ALLOWED:
            raise ValueError(
                f"Unsupported noisy method {t!r}. Allowed: {sorted(_NOISY_METHODS_ALLOWED)}"
            )
        if t not in vals:
            vals.append(t)
    if not vals:
        raise ValueError("Expected at least one noisy method in --noisy-methods.")
    return vals


def _normalize_state(psi: np.ndarray) -> np.ndarray:
    arr = np.asarray(psi, dtype=complex).reshape(-1)
    nrm = float(np.linalg.norm(arr))
    if nrm <= 0.0:
        raise ValueError("Encountered zero norm state.")
    return arr / nrm


def _build_hh_hamiltonian(args: argparse.Namespace) -> Any:
    return build_hubbard_holstein_hamiltonian(
        dims=int(args.L),
        J=float(args.t),
        U=float(args.u),
        omega0=float(args.omega0),
        g=float(args.g_ep),
        n_ph_max=int(args.n_ph_max),
        boson_encoding=str(args.boson_encoding),
        v_t=None,
        v0=float(args.dv),
        t_eval=None,
        repr_mode="JW",
        indexing=str(args.ordering),
        pbc=(str(args.boundary).strip().lower() == "periodic"),
        include_zero_point=True,
    )


def _build_reference_state(args: argparse.Namespace, num_particles: tuple[int, int]) -> np.ndarray:
    return _normalize_state(
        np.asarray(
            hubbard_holstein_reference_state(
                dims=int(args.L),
                num_particles=num_particles,
                n_ph_max=int(args.n_ph_max),
                boson_encoding=str(args.boson_encoding),
                indexing=str(args.ordering),
            ),
            dtype=complex,
        )
    )


def _build_hh_hva_ansatz(args: argparse.Namespace, *, reps: int) -> Any:
    return HubbardHolsteinPhysicalTermwiseAnsatz(
        dims=int(args.L),
        J=float(args.t),
        U=float(args.u),
        omega0=float(args.omega0),
        g=float(args.g_ep),
        n_ph_max=int(args.n_ph_max),
        boson_encoding=str(args.boson_encoding),
        reps=int(reps),
        repr_mode="JW",
        indexing=str(args.ordering),
        pbc=(str(args.boundary).strip().lower() == "periodic"),
    )


def _apply_pauli_polynomial(state: np.ndarray, poly: Any) -> np.ndarray:
    terms = poly.return_polynomial()
    if not terms:
        return np.zeros_like(state)
    nq = int(terms[0].nqubit())
    id_str = "e" * nq
    out = np.zeros_like(state)
    for term in terms:
        coeff = complex(term.p_coeff)
        if abs(coeff) <= 1e-15:
            continue
        ps = str(term.pw2strng())
        if ps == id_str:
            out += coeff * state
        else:
            out += coeff * apply_pauli_string(state, ps)
    return out


def _prepare_state(psi_ref: np.ndarray, selected_ops: list[AnsatzTerm], theta: np.ndarray) -> np.ndarray:
    psi = np.array(psi_ref, copy=True)
    for k, op in enumerate(selected_ops):
        psi = apply_exp_pauli_polynomial(psi, op.polynomial, float(theta[k]))
    return _normalize_state(psi)


def _energy(h_poly: Any, psi_ref: np.ndarray, selected_ops: list[AnsatzTerm], theta: np.ndarray) -> float:
    psi = _prepare_state(psi_ref, selected_ops, theta)
    return float(expval_pauli_polynomial(psi, h_poly))


def _commutator_gradient(h_poly: Any, op: AnsatzTerm, psi_current: np.ndarray) -> float:
    # dE/dtheta|_0 = i<psi|[H,G]|psi> = 2 Im(<Hpsi|Gpsi>)
    g_psi = _apply_pauli_polynomial(psi_current, op.polynomial)
    h_psi = _apply_pauli_polynomial(psi_current, h_poly)
    return float(2.0 * np.vdot(h_psi, g_psi).imag)


def _build_uccsd_fermion_lifted_pool(
    *,
    num_sites: int,
    num_particles: tuple[int, int],
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
) -> list[AnsatzTerm]:
    base = HardcodedUCCSDAnsatz(
        dims=int(num_sites),
        num_particles=(int(num_particles[0]), int(num_particles[1])),
        reps=1,
        repr_mode="JW",
        indexing=str(ordering),
        include_singles=True,
        include_doubles=True,
    )
    ferm_nq = 2 * int(num_sites)
    bps = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    boson_bits = int(num_sites) * bps
    nq_total = ferm_nq + boson_bits

    lifted_pool: list[AnsatzTerm] = []
    for op in base.base_terms:
        lifted = PauliPolynomial("JW")
        for term in op.polynomial.return_polynomial():
            coeff = complex(term.p_coeff)
            if abs(coeff) <= 1e-15:
                continue
            if abs(coeff.imag) > 1e-10:
                raise ValueError(f"UCCSD coeff has non-negligible imaginary part: {coeff}")
            ferm_ps = str(term.pw2strng())
            if len(ferm_ps) != int(ferm_nq):
                raise ValueError(
                    f"Unexpected UCCSD Pauli length {len(ferm_ps)} != ferm_nq {ferm_nq}."
                )
            full_ps = ("e" * int(boson_bits)) + ferm_ps
            lifted.add_term(PauliTerm(int(nq_total), ps=full_ps, pc=float(coeff.real)))
        lifted_pool.append(AnsatzTerm(label=f"uccsd_ferm_lifted::{op.label}", polynomial=lifted))

    return lifted_pool


def _build_hva_pool(args: argparse.Namespace) -> list[AnsatzTerm]:
    return list(_build_hh_hva_ansatz(args, reps=1).base_terms)


def _build_paop_full_pool(
    *,
    args: argparse.Namespace,
    num_particles: tuple[int, int],
) -> list[AnsatzTerm]:
    specs = make_paop_pool(
        "paop_full",
        num_sites=int(args.L),
        num_particles=(int(num_particles[0]), int(num_particles[1])),
        n_ph_max=int(args.n_ph_max),
        boson_encoding=str(args.boson_encoding),
        ordering=str(args.ordering),
        boundary=str(args.boundary),
        paop_r=int(args.paop_r),
        paop_split_paulis=bool(args.paop_split_paulis),
        paop_prune_eps=float(args.paop_prune_eps),
        paop_normalization=str(args.paop_normalization),
    )
    return [AnsatzTerm(label=str(lbl), polynomial=poly) for lbl, poly in specs]


def _run_vqe_stage_with_transition(
    *,
    stage_name: str,
    h_poly: Any,
    ansatz: Any,
    psi_ref: np.ndarray,
    exact_energy: float,
    restarts: int,
    seed: int,
    maxiter: int,
    method: str,
    transition_cfg: TransitionConfig,
) -> tuple[dict[str, Any], np.ndarray]:
    t0 = time.perf_counter()
    transition_state = TransitionState(cfg=transition_cfg)
    progress_events: list[dict[str, Any]] = []

    def _early_stop_checker(payload: dict[str, Any]) -> bool:
        energy_cur = payload.get("energy_current", None)
        if energy_cur is None:
            return False
        rec = update_transition_state(
            transition_state,
            delta_abs=abs(float(energy_cur) - float(exact_energy)),
        )
        progress_events.append(
            {
                "event": "objective_step",
                "restart_index": int(payload.get("restart_index", 0)),
                "nfev_so_far": int(payload.get("nfev_so_far", 0)),
                "energy_current": float(energy_cur),
                "delta_abs": float(rec["delta_abs"]),
                "slope": rec["slope"],
                "plateau_hits": int(rec["plateau_hits"]),
                "switch_triggered": bool(rec["switch_triggered"]),
            }
        )
        return bool(rec["switch_triggered"])

    def _progress_logger(ev: dict[str, Any]) -> None:
        evt = dict(ev)
        if "energy_current" in evt and evt["energy_current"] is not None:
            evt["delta_abs"] = float(abs(float(evt["energy_current"]) - float(exact_energy)))
        progress_events.append(evt)

    res = vqe_minimize(
        h_poly,
        ansatz,
        np.asarray(psi_ref, dtype=complex),
        restarts=int(restarts),
        seed=int(seed),
        maxiter=int(maxiter),
        method=str(method),
        progress_logger=_progress_logger,
        progress_every_s=0.0,
        progress_label=str(stage_name),
        track_history=False,
        emit_theta_in_progress=False,
        return_best_on_keyboard_interrupt=True,
        early_stop_checker=_early_stop_checker,
    )

    theta = np.asarray(res.theta, dtype=float)
    psi_best = _normalize_state(np.asarray(ansatz.prepare_state(theta, psi_ref), dtype=complex).reshape(-1))
    final_energy = float(res.energy)

    if bool(transition_state.switch_triggered):
        stop_reason = "slope_plateau"
    elif bool(res.success):
        stop_reason = "optimizer_success"
    else:
        stop_reason = str(res.message)

    payload = {
        "stage": str(stage_name),
        "success": bool(res.success),
        "stop_reason": str(stop_reason),
        "optimizer_message": str(res.message),
        "energy": float(final_energy),
        "exact_energy": float(exact_energy),
        "delta_abs": float(abs(final_energy - float(exact_energy))),
        "num_parameters": int(getattr(ansatz, "num_parameters", theta.size)),
        "best_restart": int(res.best_restart + 1),
        "nfev": int(res.nfev),
        "nit": int(res.nit),
        "restarts": int(restarts),
        "maxiter": int(maxiter),
        "method": str(method),
        "theta": [float(x) for x in theta.tolist()],
        "transition": summarize_transition(transition_state),
        "progress_events": progress_events,
        "elapsed_s": float(time.perf_counter() - t0),
    }
    return payload, psi_best


def _run_adapt_stage_with_transition(
    *,
    h_poly: Any,
    psi_start: np.ndarray,
    pool: list[AnsatzTerm],
    exact_energy: float,
    allow_repeats: bool,
    max_depth: int,
    maxiter: int,
    eps_grad: float,
    eps_energy: float,
    seed: int,
    transition_cfg: TransitionConfig,
) -> tuple[dict[str, Any], np.ndarray]:
    t0 = time.perf_counter()
    if not pool:
        raise ValueError("Pool B is empty.")

    try:
        from scipy.optimize import minimize as scipy_minimize
    except Exception as exc:
        raise RuntimeError("SciPy is required for ADAPT stage in this report.") from exc

    rng = np.random.default_rng(int(seed))
    selected_ops: list[AnsatzTerm] = []
    theta = np.zeros(0, dtype=float)
    available_indices = set(range(len(pool)))
    history: list[dict[str, Any]] = []
    nfev_total = 1
    nit_total = 0

    energy_current = float(expval_pauli_polynomial(np.asarray(psi_start, dtype=complex), h_poly))
    transition_state = TransitionState(cfg=transition_cfg)
    update_transition_state(transition_state, delta_abs=abs(float(energy_current) - float(exact_energy)))

    stop_reason = "max_depth"

    for depth in range(int(max_depth)):
        candidate_indices = list(range(len(pool))) if bool(allow_repeats) else sorted(available_indices)
        if not candidate_indices:
            stop_reason = "pool_exhausted"
            break

        psi_current = _prepare_state(np.asarray(psi_start, dtype=complex), selected_ops, theta)
        gradients = {
            idx: float(_commutator_gradient(h_poly, pool[idx], psi_current)) for idx in candidate_indices
        }
        grad_abs = {idx: abs(v) for idx, v in gradients.items()}
        best_idx = max(candidate_indices, key=lambda i: (grad_abs[i], -i))
        best_grad_abs = float(grad_abs[best_idx])

        if best_grad_abs < float(eps_grad):
            stop_reason = "eps_grad"
            break

        selected_ops.append(pool[best_idx])
        theta = np.append(theta, 0.0)
        if not bool(allow_repeats):
            available_indices.discard(best_idx)

        energy_prev = float(energy_current)

        def _obj(x: np.ndarray) -> float:
            return _energy(h_poly, np.asarray(psi_start, dtype=complex), selected_ops, np.asarray(x, dtype=float))

        x0 = np.asarray(theta, dtype=float) + 0.02 * rng.normal(size=theta.size)
        res = scipy_minimize(
            _obj,
            x0,
            method="COBYLA",
            options={"maxiter": int(maxiter), "rhobeg": 0.3},
        )
        theta = np.asarray(res.x, dtype=float)
        energy_current = float(res.fun)
        nfev_total += int(getattr(res, "nfev", 0))
        nit_total += int(getattr(res, "nit", 0))

        trans_rec = update_transition_state(
            transition_state,
            delta_abs=abs(float(energy_current) - float(exact_energy)),
        )

        history.append(
            {
                "depth": int(depth + 1),
                "selected_pool_index": int(best_idx),
                "selected_label": str(pool[best_idx].label),
                "max_gradient_abs": float(best_grad_abs),
                "energy_before_opt": float(energy_prev),
                "energy_after_opt": float(energy_current),
                "delta_energy_abs": float(abs(energy_current - energy_prev)),
                "delta_abs_vs_exact": float(abs(energy_current - float(exact_energy))),
                "slope": trans_rec["slope"],
                "plateau_hits": int(trans_rec["plateau_hits"]),
                "switch_triggered": bool(trans_rec["switch_triggered"]),
                "opt_nfev": int(getattr(res, "nfev", 0)),
                "opt_nit": int(getattr(res, "nit", 0)),
                "opt_success": bool(getattr(res, "success", False)),
                "opt_message": str(getattr(res, "message", "")),
            }
        )

        if abs(float(energy_current) - float(energy_prev)) < float(eps_energy):
            stop_reason = "eps_energy"
            break

        if bool(trans_rec["switch_triggered"]):
            stop_reason = "slope_plateau"
            break

    psi_best = _prepare_state(np.asarray(psi_start, dtype=complex), selected_ops, theta)

    payload = {
        "stage": "adapt_pool_b",
        "success": True,
        "stop_reason": str(stop_reason),
        "energy": float(energy_current),
        "exact_energy": float(exact_energy),
        "delta_abs": float(abs(float(energy_current) - float(exact_energy))),
        "ansatz_depth": int(len(selected_ops)),
        "num_parameters": int(theta.size),
        "allow_repeats": bool(allow_repeats),
        "max_depth": int(max_depth),
        "maxiter": int(maxiter),
        "eps_grad": float(eps_grad),
        "eps_energy": float(eps_energy),
        "nfev_total": int(nfev_total),
        "nit_total": int(nit_total),
        "operators": [str(op.label) for op in selected_ops],
        "optimal_point": [float(x) for x in theta.tolist()],
        "history": history,
        "transition": summarize_transition(transition_state),
        "elapsed_s": float(time.perf_counter() - t0),
    }
    return payload, psi_best


def _parse_custom_s(raw: str | None, *, L: int) -> list[float] | None:
    if raw is None:
        return None
    txt = str(raw).strip()
    if not txt:
        return None
    if txt.startswith("["):
        data = json.loads(txt)
        arr = [float(x) for x in data]
    else:
        arr = [float(x.strip()) for x in txt.split(",") if x.strip()]
    if len(arr) != int(L):
        raise ValueError(f"drive-custom-s length {len(arr)} does not match L={L}")
    return arr


def _build_drive_profile(args: argparse.Namespace, *, enabled: bool) -> dict[str, Any] | None:
    if not bool(enabled):
        return None
    custom_weights = _parse_custom_s(args.drive_custom_s, L=int(args.L)) if str(args.drive_pattern) == "custom" else None
    return {
        "enabled": True,
        "A": float(args.drive_A),
        "omega": float(args.drive_omega),
        "tbar": float(args.drive_tbar),
        "phi": float(args.drive_phi),
        "pattern": str(args.drive_pattern),
        "custom_s": custom_weights,
        "include_identity": bool(args.drive_include_identity),
        "t0": float(args.drive_t0),
        "time_sampling": str(args.drive_time_sampling),
        "reference_steps_multiplier": int(args.exact_steps_multiplier),
        "reference_method": reference_method_name(str(args.drive_time_sampling)),
    }


def _drive_provider_from_profile(
    *,
    profile: dict[str, Any] | None,
    num_sites: int,
    nq_total: int,
    ordering: str,
) -> tuple[Any | None, dict[str, Any] | None]:
    if profile is None:
        return None, None
    drive = build_gaussian_sinusoid_density_drive(
        n_sites=int(num_sites),
        nq_total=int(nq_total),
        indexing=str(ordering),
        A=float(profile["A"]),
        omega=float(profile["omega"]),
        tbar=float(profile["tbar"]),
        phi=float(profile["phi"]),
        pattern_mode=str(profile["pattern"]),
        custom_weights=profile.get("custom_s"),
        include_identity=bool(profile.get("include_identity", False)),
        electron_qubit_offset=0,
        coeff_tol=0.0,
    )
    return drive.coeff_map_exyz, {
        "labels": int(len(drive.template.labels_exyz(include_identity=bool(drive.include_identity)))),
        "identity_included": bool(drive.include_identity),
    }


def _spin_orbital_bit_index(site: int, spin: int, num_sites: int, ordering: str) -> int:
    ord_norm = str(ordering).strip().lower()
    if ord_norm == "blocked":
        return int(site) if int(spin) == 0 else int(num_sites) + int(site)
    if ord_norm == "interleaved":
        return (2 * int(site)) + int(spin)
    raise ValueError(f"Unsupported ordering {ordering!r}")


def _staggered_qop(num_qubits: int, num_sites: int, ordering: str) -> SparsePauliOp:
    op = SparsePauliOp.from_list([("I" * int(num_qubits), 0.0)])
    L = int(num_sites)
    for site in range(L):
        sign = 1.0 if (site % 2 == 0) else -1.0
        up_idx = _spin_orbital_bit_index(site, 0, L, ordering)
        dn_idx = _spin_orbital_bit_index(site, 1, L, ordering)
        op = op + (sign / float(L)) * _number_operator_qop(int(num_qubits), int(up_idx))
        op = op + (sign / float(L)) * _number_operator_qop(int(num_qubits), int(dn_idx))
    return op.simplify(atol=1e-12)


def _time_sample(step_idx: int, dt: float, sampling: str) -> float:
    mode = str(sampling).strip().lower()
    if mode == "midpoint":
        return float((float(step_idx) + 0.5) * float(dt))
    if mode == "left":
        return float(float(step_idx) * float(dt))
    if mode == "right":
        return float((float(step_idx) + 1.0) * float(dt))
    raise ValueError(f"Unsupported time sampling {sampling!r}")


def _build_suzuki2_time_dependent_circuit(
    *,
    initial_circuit: QuantumCircuit,
    ordered_labels_exyz: list[str],
    static_coeff_map_exyz: dict[str, complex],
    drive_provider_exyz: Any | None,
    time_value: float,
    trotter_steps: int,
    drive_t0: float,
    drive_time_sampling: str,
) -> QuantumCircuit:
    qc = initial_circuit.copy()
    if abs(float(time_value)) <= 1e-15:
        return qc

    dt = float(time_value) / float(trotter_steps)
    synthesis = SuzukiTrotter(order=2, reps=1, preserve_order=True)

    for step_idx in range(int(trotter_steps)):
        t_sample = _time_sample(step_idx, dt, drive_time_sampling)
        drive_map = {}
        if drive_provider_exyz is not None:
            drive_map = dict(drive_provider_exyz(float(drive_t0) + float(t_sample)))
        qop = build_time_dependent_sparse_qop(
            ordered_labels_exyz=ordered_labels_exyz,
            static_coeff_map_exyz=static_coeff_map_exyz,
            drive_coeff_map_exyz=drive_map,
        )
        qc.append(
            PauliEvolutionGate(qop, time=float(dt), synthesis=synthesis),
            list(range(int(initial_circuit.num_qubits))),
        )
    return qc


def _build_cfqm_stage_map_exyz(
    *,
    ordered_labels_exyz: list[str],
    static_coeff_map_exyz: dict[str, complex],
    drive_maps_exyz: list[dict[str, complex]],
    a_row: list[float],
    s_static: float,
    coeff_drop_abs_tol: float,
) -> dict[str, complex]:
    ordered_set = set(ordered_labels_exyz)
    stage_map: dict[str, complex] = {}

    for lbl in ordered_labels_exyz:
        coeff0 = static_coeff_map_exyz.get(lbl, 0.0 + 0.0j)
        scaled = complex(float(s_static)) * complex(coeff0)
        if scaled != 0.0:
            stage_map[lbl] = scaled

    for j, drive_map in enumerate(drive_maps_exyz):
        w = float(a_row[j])
        if w == 0.0:
            continue
        for lbl, coeff_drive in drive_map.items():
            if lbl not in ordered_set:
                continue
            inc = complex(w) * complex(coeff_drive)
            if inc == 0.0 and lbl not in stage_map:
                continue
            stage_map[lbl] = stage_map.get(lbl, 0.0 + 0.0j) + inc

    drop = float(max(0.0, coeff_drop_abs_tol))
    if drop > 0.0:
        for lbl in list(stage_map):
            if abs(stage_map[lbl]) < drop:
                del stage_map[lbl]
    return stage_map


def _build_cfqm_time_dependent_circuit(
    *,
    method: str,
    initial_circuit: QuantumCircuit,
    ordered_labels_exyz: list[str],
    static_coeff_map_exyz: dict[str, complex],
    drive_provider_exyz: Any | None,
    time_value: float,
    trotter_steps: int,
    drive_t0: float,
    coeff_drop_abs_tol: float,
) -> QuantumCircuit:
    qc = initial_circuit.copy()
    if abs(float(time_value)) <= 1e-15:
        return qc

    scheme = get_cfqm_scheme(str(method))
    c_nodes = [float(x) for x in scheme["c"]]
    a_rows = [[float(v) for v in row] for row in scheme["a"]]
    s_static = [float(v) for v in scheme["s_static"]]

    dt = float(time_value) / float(trotter_steps)
    synthesis = SuzukiTrotter(order=2, reps=1, preserve_order=True)
    qubits = list(range(int(initial_circuit.num_qubits)))

    for step_idx in range(int(trotter_steps)):
        t_abs = float(drive_t0) + float(step_idx) * float(dt)
        drive_maps_exyz: list[dict[str, complex]] = []
        for c_j in c_nodes:
            t_node = float(t_abs) + float(c_j) * float(dt)
            raw = {} if drive_provider_exyz is None else dict(drive_provider_exyz(float(t_node)))
            drive_maps_exyz.append({str(k): complex(v) for k, v in raw.items()})

        for k, a_row in enumerate(a_rows):
            stage_map = _build_cfqm_stage_map_exyz(
                ordered_labels_exyz=list(ordered_labels_exyz),
                static_coeff_map_exyz=dict(static_coeff_map_exyz),
                drive_maps_exyz=drive_maps_exyz,
                a_row=[float(v) for v in a_row],
                s_static=float(s_static[k]),
                coeff_drop_abs_tol=float(coeff_drop_abs_tol),
            )
            qop = build_time_dependent_sparse_qop(
                ordered_labels_exyz=ordered_labels_exyz,
                static_coeff_map_exyz=static_coeff_map_exyz,
                drive_coeff_map_exyz=stage_map,
            )
            qc.append(PauliEvolutionGate(qop, time=float(dt), synthesis=synthesis), qubits)
    return qc


def _pauli_weight(label_exyz: str) -> int:
    return int(sum(1 for ch in str(label_exyz) if ch in {"x", "y", "z"}))


def _pauli_xy_count(label_exyz: str) -> int:
    return int(sum(1 for ch in str(label_exyz) if ch in {"x", "y"}))


def _cx_proxy_term(label_exyz: str) -> int:
    return int(2 * max(_pauli_weight(label_exyz) - 1, 0))


def _sq_proxy_term(label_exyz: str) -> int:
    return int(2 * _pauli_xy_count(label_exyz) + 1)


def _active_labels_exyz(
    coeff_map_exyz: dict[str, complex],
    ordered_labels_exyz: list[str],
    tol: float,
) -> list[str]:
    thr = float(max(0.0, tol))
    out: list[str] = []
    for lbl in ordered_labels_exyz:
        if abs(complex(coeff_map_exyz.get(lbl, 0.0 + 0.0j))) > thr:
            out.append(lbl)
    return out


def _compute_sweep_proxy_cost(active_labels_exyz: list[str]) -> dict[str, int]:
    term_exp_count = int(2 * len(active_labels_exyz))
    cx_proxy = int(2 * sum(_cx_proxy_term(lbl) for lbl in active_labels_exyz))
    sq_proxy = int(2 * sum(_sq_proxy_term(lbl) for lbl in active_labels_exyz))
    return {
        "term_exp_count": int(term_exp_count),
        "cx_proxy": int(cx_proxy),
        "sq_proxy": int(sq_proxy),
    }


def _compute_time_dynamics_proxy_cost(
    *,
    method: str,
    t_final: float,
    trotter_steps: int,
    drive_t0: float,
    drive_time_sampling: str,
    ordered_labels_exyz: list[str],
    static_coeff_map_exyz: dict[str, complex],
    drive_provider_exyz: Any | None,
    active_coeff_tol: float,
    coeff_drop_abs_tol: float,
) -> dict[str, int]:
    method_norm = str(method).strip().lower()
    if method_norm not in _NOISY_METHODS_ALLOWED:
        raise ValueError(f"Unsupported noisy method {method!r}.")
    if int(trotter_steps) < 1:
        raise ValueError("trotter_steps must be >= 1")
    if float(t_final) < 0.0:
        raise ValueError("t_final must be >= 0")

    total_term = 0
    total_cx = 0
    total_sq = 0
    dt = float(t_final) / float(trotter_steps)

    if method_norm == "suzuki2":
        for step_idx in range(int(trotter_steps)):
            t_sample = _time_sample(step_idx, dt, str(drive_time_sampling))
            raw = {} if drive_provider_exyz is None else dict(drive_provider_exyz(float(drive_t0) + float(t_sample)))
            merged: dict[str, complex] = {}
            for lbl in ordered_labels_exyz:
                merged[lbl] = complex(static_coeff_map_exyz.get(lbl, 0.0 + 0.0j)) + complex(raw.get(lbl, 0.0))
            active = _active_labels_exyz(merged, ordered_labels_exyz, float(active_coeff_tol))
            sweep = _compute_sweep_proxy_cost(active)
            total_term += int(sweep["term_exp_count"])
            total_cx += int(sweep["cx_proxy"])
            total_sq += int(sweep["sq_proxy"])
    else:
        scheme = get_cfqm_scheme(str(method_norm))
        c_nodes = [float(x) for x in scheme["c"]]
        a_rows = [[float(v) for v in row] for row in scheme["a"]]
        s_static = [float(v) for v in scheme["s_static"]]

        for step_idx in range(int(trotter_steps)):
            t_abs = float(drive_t0) + float(step_idx) * float(dt)
            drive_maps_exyz: list[dict[str, complex]] = []
            for c_j in c_nodes:
                t_node = float(t_abs) + float(c_j) * float(dt)
                raw = {} if drive_provider_exyz is None else dict(drive_provider_exyz(float(t_node)))
                drive_maps_exyz.append({str(k): complex(v) for k, v in raw.items()})

            for k, a_row in enumerate(a_rows):
                stage_map = _build_cfqm_stage_map_exyz(
                    ordered_labels_exyz=list(ordered_labels_exyz),
                    static_coeff_map_exyz=dict(static_coeff_map_exyz),
                    drive_maps_exyz=drive_maps_exyz,
                    a_row=[float(v) for v in a_row],
                    s_static=float(s_static[k]),
                    coeff_drop_abs_tol=float(coeff_drop_abs_tol),
                )
                active = _active_labels_exyz(stage_map, ordered_labels_exyz, float(active_coeff_tol))
                sweep = _compute_sweep_proxy_cost(active)
                total_term += int(sweep["term_exp_count"])
                total_cx += int(sweep["cx_proxy"])
                total_sq += int(sweep["sq_proxy"])

    return {
        "term_exp_count_total": int(total_term),
        "pauli_rot_count_total": int(total_term),
        "cx_proxy_total": int(total_cx),
        "sq_proxy_total": int(total_sq),
        "depth_proxy_total": int(total_term),
    }


def _run_noisy_suzuki_trajectory(
    *,
    L: int,
    ordering: str,
    psi_seed: np.ndarray,
    ordered_labels_exyz: list[str],
    static_coeff_map_exyz: dict[str, complex],
    t_final: float,
    num_times: int,
    trotter_steps: int,
    drive_profile: dict[str, Any] | None,
    noise_mode: str,
    shots: int,
    seed: int,
    oracle_repeats: int,
    oracle_aggregate: str,
    backend_name: str | None,
    use_fake_backend: bool,
    allow_aer_fallback: bool,
    omp_shm_workaround: bool,
) -> dict[str, Any]:
    nq = int(round(math.log2(int(np.asarray(psi_seed).size))))
    drive_provider_exyz, drive_meta = _drive_provider_from_profile(
        profile=drive_profile,
        num_sites=int(L),
        nq_total=int(nq),
        ordering=str(ordering),
    )

    initial_circuit = QuantumCircuit(int(nq))
    _append_reference_state(initial_circuit, np.asarray(psi_seed, dtype=complex))

    static_qop = build_time_dependent_sparse_qop(
        ordered_labels_exyz=ordered_labels_exyz,
        static_coeff_map_exyz=static_coeff_map_exyz,
        drive_coeff_map_exyz=None,
    )

    up0_idx = _spin_orbital_bit_index(0, 0, int(L), ordering)
    dn0_idx = _spin_orbital_bit_index(0, 1, int(L), ordering)
    obs_up0 = _number_operator_qop(int(nq), int(up0_idx))
    obs_dn0 = _number_operator_qop(int(nq), int(dn0_idx))
    obs_doublon0 = _doublon_site_qop(int(nq), int(up0_idx), int(dn0_idx))
    obs_staggered = _staggered_qop(int(nq), int(L), str(ordering))

    noisy_cfg = OracleConfig(
        noise_mode=str(noise_mode),
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        backend_name=(None if backend_name is None else str(backend_name)),
        use_fake_backend=bool(use_fake_backend),
        allow_aer_fallback=bool(allow_aer_fallback),
        aer_fallback_mode="sampler_shots",
        omp_shm_workaround=bool(omp_shm_workaround),
    )
    ideal_cfg = OracleConfig(
        noise_mode="ideal",
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        backend_name=None,
        use_fake_backend=False,
    )

    times = np.linspace(0.0, float(t_final), int(num_times))
    rows: list[dict[str, Any]] = []

    with ExpectationOracle(noisy_cfg) as noisy_oracle, ExpectationOracle(ideal_cfg) as ideal_oracle:
        for t_val in times:
            qc_t = _build_suzuki2_time_dependent_circuit(
                initial_circuit=initial_circuit,
                ordered_labels_exyz=ordered_labels_exyz,
                static_coeff_map_exyz=static_coeff_map_exyz,
                drive_provider_exyz=drive_provider_exyz,
                time_value=float(t_val),
                trotter_steps=int(trotter_steps),
                drive_t0=float(0.0 if drive_profile is None else drive_profile.get("t0", 0.0)),
                drive_time_sampling=str(
                    "midpoint" if drive_profile is None else drive_profile.get("time_sampling", "midpoint")
                ),
            )

            if drive_provider_exyz is None:
                total_qop = static_qop
            else:
                drive_obs_map = dict(
                    drive_provider_exyz(
                        float(drive_profile.get("t0", 0.0)) + float(t_val)
                    )
                )
                total_qop = build_time_dependent_sparse_qop(
                    ordered_labels_exyz=ordered_labels_exyz,
                    static_coeff_map_exyz=static_coeff_map_exyz,
                    drive_coeff_map_exyz=drive_obs_map,
                )

            def _pair(obs: SparsePauliOp) -> tuple[float, float, float, float]:
                n_est = noisy_oracle.evaluate(qc_t, obs)
                i_est = ideal_oracle.evaluate(qc_t, obs)
                return (
                    float(n_est.mean),
                    float(n_est.std),
                    float(i_est.mean),
                    float(n_est.mean - i_est.mean),
                )

            e_s = _pair(static_qop)
            e_t = _pair(total_qop)
            up0 = _pair(obs_up0)
            dn0 = _pair(obs_dn0)
            dbl = _pair(obs_doublon0)
            stg = _pair(obs_staggered)

            rows.append(
                {
                    "time": float(t_val),
                    "energy_static_noisy": e_s[0],
                    "energy_static_noisy_std": e_s[1],
                    "energy_static_ideal": e_s[2],
                    "energy_static_delta_noisy_minus_ideal": e_s[3],
                    "energy_total_noisy": e_t[0],
                    "energy_total_noisy_std": e_t[1],
                    "energy_total_ideal": e_t[2],
                    "energy_total_delta_noisy_minus_ideal": e_t[3],
                    "n_up_site0_noisy": up0[0],
                    "n_up_site0_noisy_std": up0[1],
                    "n_up_site0_ideal": up0[2],
                    "n_up_site0_delta_noisy_minus_ideal": up0[3],
                    "n_dn_site0_noisy": dn0[0],
                    "n_dn_site0_noisy_std": dn0[1],
                    "n_dn_site0_ideal": dn0[2],
                    "n_dn_site0_delta_noisy_minus_ideal": dn0[3],
                    "doublon_noisy": dbl[0],
                    "doublon_noisy_std": dbl[1],
                    "doublon_ideal": dbl[2],
                    "doublon_delta_noisy_minus_ideal": dbl[3],
                    "staggered_noisy": stg[0],
                    "staggered_noisy_std": stg[1],
                    "staggered_ideal": stg[2],
                    "staggered_delta_noisy_minus_ideal": stg[3],
                }
            )

        backend_details = {
            "backend_info": {
                "noise_mode": str(noisy_oracle.backend_info.noise_mode),
                "estimator_kind": str(noisy_oracle.backend_info.estimator_kind),
                "backend_name": noisy_oracle.backend_info.backend_name,
                "using_fake_backend": bool(noisy_oracle.backend_info.using_fake_backend),
                "details": dict(noisy_oracle.backend_info.details),
            }
        }

    return {
        "success": True,
        "noise_mode": str(noise_mode),
        "drive_enabled": bool(drive_profile is not None),
        "drive_meta": drive_meta,
        "trajectory": rows,
        **backend_details,
    }


def _run_noisy_method_trajectory(
    *,
    L: int,
    ordering: str,
    psi_seed: np.ndarray,
    ordered_labels_exyz: list[str],
    static_coeff_map_exyz: dict[str, complex],
    t_final: float,
    num_times: int,
    trotter_steps: int,
    drive_profile: dict[str, Any] | None,
    noise_mode: str,
    shots: int,
    seed: int,
    oracle_repeats: int,
    oracle_aggregate: str,
    backend_name: str | None,
    use_fake_backend: bool,
    allow_aer_fallback: bool,
    omp_shm_workaround: bool,
    method: str,
    benchmark_active_coeff_tol: float,
    cfqm_coeff_drop_abs_tol: float,
) -> dict[str, Any]:
    t_wall_start = float(time.perf_counter())
    method_norm = str(method).strip().lower()
    if method_norm not in _NOISY_METHODS_ALLOWED:
        raise ValueError(f"Unsupported noisy method {method!r}.")

    nq = int(round(math.log2(int(np.asarray(psi_seed).size))))
    drive_provider_exyz, drive_meta = _drive_provider_from_profile(
        profile=drive_profile,
        num_sites=int(L),
        nq_total=int(nq),
        ordering=str(ordering),
    )

    initial_circuit = QuantumCircuit(int(nq))
    _append_reference_state(initial_circuit, np.asarray(psi_seed, dtype=complex))

    static_qop = build_time_dependent_sparse_qop(
        ordered_labels_exyz=ordered_labels_exyz,
        static_coeff_map_exyz=static_coeff_map_exyz,
        drive_coeff_map_exyz=None,
    )

    up0_idx = _spin_orbital_bit_index(0, 0, int(L), ordering)
    dn0_idx = _spin_orbital_bit_index(0, 1, int(L), ordering)
    obs_up0 = _number_operator_qop(int(nq), int(up0_idx))
    obs_dn0 = _number_operator_qop(int(nq), int(dn0_idx))
    obs_doublon0 = _doublon_site_qop(int(nq), int(up0_idx), int(dn0_idx))
    obs_staggered = _staggered_qop(int(nq), int(L), str(ordering))

    noisy_cfg = OracleConfig(
        noise_mode=str(noise_mode),
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        backend_name=(None if backend_name is None else str(backend_name)),
        use_fake_backend=bool(use_fake_backend),
        allow_aer_fallback=bool(allow_aer_fallback),
        aer_fallback_mode="sampler_shots",
        omp_shm_workaround=bool(omp_shm_workaround),
    )
    ideal_cfg = OracleConfig(
        noise_mode="ideal",
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        backend_name=None,
        use_fake_backend=False,
    )

    times = np.linspace(0.0, float(t_final), int(num_times))
    rows: list[dict[str, Any]] = []
    circuit_build_s_total = 0.0
    oracle_eval_s_total = 0.0
    oracle_calls_total = 0

    with ExpectationOracle(noisy_cfg) as noisy_oracle, ExpectationOracle(ideal_cfg) as ideal_oracle:
        for t_val in times:
            t_circ0 = float(time.perf_counter())
            if method_norm == "suzuki2":
                qc_t = _build_suzuki2_time_dependent_circuit(
                    initial_circuit=initial_circuit,
                    ordered_labels_exyz=ordered_labels_exyz,
                    static_coeff_map_exyz=static_coeff_map_exyz,
                    drive_provider_exyz=drive_provider_exyz,
                    time_value=float(t_val),
                    trotter_steps=int(trotter_steps),
                    drive_t0=float(0.0 if drive_profile is None else drive_profile.get("t0", 0.0)),
                    drive_time_sampling=str(
                        "midpoint" if drive_profile is None else drive_profile.get("time_sampling", "midpoint")
                    ),
                )
            else:
                qc_t = _build_cfqm_time_dependent_circuit(
                    method=str(method_norm),
                    initial_circuit=initial_circuit,
                    ordered_labels_exyz=ordered_labels_exyz,
                    static_coeff_map_exyz=static_coeff_map_exyz,
                    drive_provider_exyz=drive_provider_exyz,
                    time_value=float(t_val),
                    trotter_steps=int(trotter_steps),
                    drive_t0=float(0.0 if drive_profile is None else drive_profile.get("t0", 0.0)),
                    coeff_drop_abs_tol=float(cfqm_coeff_drop_abs_tol),
                )
            circuit_build_s_total += float(time.perf_counter() - t_circ0)

            if drive_provider_exyz is None:
                total_qop = static_qop
            else:
                drive_obs_map = dict(
                    drive_provider_exyz(
                        float(drive_profile.get("t0", 0.0)) + float(t_val)
                    )
                )
                total_qop = build_time_dependent_sparse_qop(
                    ordered_labels_exyz=ordered_labels_exyz,
                    static_coeff_map_exyz=static_coeff_map_exyz,
                    drive_coeff_map_exyz=drive_obs_map,
                )

            def _pair(obs: SparsePauliOp) -> tuple[float, float, float, float]:
                nonlocal oracle_eval_s_total, oracle_calls_total
                t_eval0 = float(time.perf_counter())
                n_est = noisy_oracle.evaluate(qc_t, obs)
                i_est = ideal_oracle.evaluate(qc_t, obs)
                oracle_eval_s_total += float(time.perf_counter() - t_eval0)
                oracle_calls_total += 2
                return (
                    float(n_est.mean),
                    float(n_est.std),
                    float(i_est.mean),
                    float(n_est.mean - i_est.mean),
                )

            e_s = _pair(static_qop)
            e_t = _pair(total_qop)
            up0 = _pair(obs_up0)
            dn0 = _pair(obs_dn0)
            dbl = _pair(obs_doublon0)
            stg = _pair(obs_staggered)

            rows.append(
                {
                    "time": float(t_val),
                    "energy_static_noisy": e_s[0],
                    "energy_static_noisy_std": e_s[1],
                    "energy_static_ideal": e_s[2],
                    "energy_static_delta_noisy_minus_ideal": e_s[3],
                    "energy_total_noisy": e_t[0],
                    "energy_total_noisy_std": e_t[1],
                    "energy_total_ideal": e_t[2],
                    "energy_total_delta_noisy_minus_ideal": e_t[3],
                    "n_up_site0_noisy": up0[0],
                    "n_up_site0_noisy_std": up0[1],
                    "n_up_site0_ideal": up0[2],
                    "n_up_site0_delta_noisy_minus_ideal": up0[3],
                    "n_dn_site0_noisy": dn0[0],
                    "n_dn_site0_noisy_std": dn0[1],
                    "n_dn_site0_ideal": dn0[2],
                    "n_dn_site0_delta_noisy_minus_ideal": dn0[3],
                    "doublon_noisy": dbl[0],
                    "doublon_noisy_std": dbl[1],
                    "doublon_ideal": dbl[2],
                    "doublon_delta_noisy_minus_ideal": dbl[3],
                    "staggered_noisy": stg[0],
                    "staggered_noisy_std": stg[1],
                    "staggered_ideal": stg[2],
                    "staggered_delta_noisy_minus_ideal": stg[3],
                }
            )

        backend_details = {
            "backend_info": {
                "noise_mode": str(noisy_oracle.backend_info.noise_mode),
                "estimator_kind": str(noisy_oracle.backend_info.estimator_kind),
                "backend_name": noisy_oracle.backend_info.backend_name,
                "using_fake_backend": bool(noisy_oracle.backend_info.using_fake_backend),
                "details": dict(noisy_oracle.backend_info.details),
            }
        }

    benchmark_cost = _compute_time_dynamics_proxy_cost(
        method=str(method_norm),
        t_final=float(t_final),
        trotter_steps=int(trotter_steps),
        drive_t0=float(0.0 if drive_profile is None else drive_profile.get("t0", 0.0)),
        drive_time_sampling=str(
            "midpoint" if drive_profile is None else drive_profile.get("time_sampling", "midpoint")
        ),
        ordered_labels_exyz=list(ordered_labels_exyz),
        static_coeff_map_exyz=dict(static_coeff_map_exyz),
        drive_provider_exyz=drive_provider_exyz,
        active_coeff_tol=float(benchmark_active_coeff_tol),
        coeff_drop_abs_tol=float(cfqm_coeff_drop_abs_tol),
    )
    benchmark_runtime = {
        "wall_total_s": float(time.perf_counter() - t_wall_start),
        "circuit_build_s_total": float(circuit_build_s_total),
        "oracle_eval_s_total": float(oracle_eval_s_total),
        "oracle_calls_total": int(oracle_calls_total),
        "trajectory_points": int(len(rows)),
    }

    return {
        "success": True,
        "method": str(method_norm),
        "noise_mode": str(noise_mode),
        "drive_enabled": bool(drive_profile is not None),
        "drive_meta": drive_meta,
        "trajectory": rows,
        "benchmark_cost": benchmark_cost,
        "benchmark_runtime": benchmark_runtime,
        **backend_details,
    }


def _noisy_worker_entry(queue: Any, kwargs: dict[str, Any]) -> None:
    try:
        payload = _run_noisy_method_trajectory(**kwargs)
        queue.put({"ok": True, "payload": payload})
    except Exception as exc:  # pragma: no cover - subprocess fault path
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _run_noisy_mode_isolated(
    *,
    kwargs: dict[str, Any],
    timeout_s: int,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_noisy_worker_entry, args=(queue, kwargs), daemon=False)
    proc.start()
    proc.join(timeout=float(timeout_s))

    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        return {
            "success": False,
            "env_blocked": True,
            "reason": f"timeout_after_{int(timeout_s)}s",
            "exitcode": proc.exitcode,
        }

    if int(proc.exitcode or 0) != 0:
        return {
            "success": False,
            "env_blocked": True,
            "reason": "subprocess_nonzero_exit",
            "exitcode": int(proc.exitcode or 0),
        }

    if queue.empty():
        return {
            "success": False,
            "env_blocked": True,
            "reason": "subprocess_completed_without_payload",
            "exitcode": int(proc.exitcode or 0),
        }

    msg = queue.get()
    if not bool(msg.get("ok", False)):
        return {
            "success": False,
            "env_blocked": True,
            "reason": "worker_exception",
            "error": str(msg.get("error", "unknown")),
            "exitcode": int(proc.exitcode or 0),
        }

    return dict(msg.get("payload", {}))


def _extract_series(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _compute_exact_reference_for_hh(
    *,
    hmat: np.ndarray,
    num_sites: int,
    ordering: str,
    num_particles: tuple[int, int],
    energy_tol: float = 0.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (E0, psi_exact_gs, basis_gs_manifold) for HH sector-filtered reference."""
    gs_energy, basis = hc_pipeline._ground_manifold_basis_sector_filtered_hh(
        hmat=np.asarray(hmat, dtype=complex),
        num_sites=int(num_sites),
        num_particles=(int(num_particles[0]), int(num_particles[1])),
        ordering=str(ordering),
        nq_total=int(round(math.log2(int(np.asarray(hmat).shape[0])))),
        energy_tol=float(energy_tol),
    )
    psi_exact = hc_pipeline._normalize_state(
        np.asarray(basis[:, 0], dtype=complex).reshape(-1)
    )
    basis_orth = hc_pipeline._orthonormalize_basis_columns(
        np.asarray(basis, dtype=complex)
    )
    return float(gs_energy), np.asarray(psi_exact, dtype=complex), np.asarray(basis_orth, dtype=complex)


def _run_hardcoded_suzuki_profile(
    *,
    args: argparse.Namespace,
    hmat: np.ndarray,
    ordered_labels_exyz: list[str],
    coeff_map_exyz: dict[str, complex],
    psi_warm: np.ndarray,
    psi_adapt: np.ndarray,
    psi_final: np.ndarray,
    psi_exact_ref: np.ndarray,
    fidelity_basis_v0: np.ndarray,
    fidelity_subspace_energy_tol: float,
    drive_profile: dict[str, Any] | None,
) -> dict[str, Any]:
    """Run hardcoded-style suzuki2 trajectory with three mapped branches."""
    nq = int(round(math.log2(int(np.asarray(psi_final).size))))
    drive_provider_exyz, drive_meta = _drive_provider_from_profile(
        profile=drive_profile,
        num_sites=int(args.L),
        nq_total=int(nq),
        ordering=str(args.ordering),
    )

    rows, _ = hc_pipeline._simulate_trajectory(
        num_sites=int(args.L),
        ordering=str(args.ordering),
        psi0_legacy_trot=np.asarray(psi_warm, dtype=complex),
        psi0_paop_trot=np.asarray(psi_adapt, dtype=complex),
        psi0_hva_trot=np.asarray(psi_final, dtype=complex),
        legacy_branch_label="warm_start",
        psi0_exact_ref=np.asarray(psi_exact_ref, dtype=complex),
        fidelity_subspace_basis_v0=np.asarray(fidelity_basis_v0, dtype=complex),
        fidelity_subspace_energy_tol=float(fidelity_subspace_energy_tol),
        hmat=np.asarray(hmat, dtype=complex),
        ordered_labels_exyz=list(ordered_labels_exyz),
        coeff_map_exyz=dict(coeff_map_exyz),
        trotter_steps=int(args.trotter_steps),
        t_final=float(args.t_final),
        num_times=int(args.num_times),
        suzuki_order=2,
        drive_coeff_provider_exyz=drive_provider_exyz,
        drive_t0=float(0.0 if drive_profile is None else drive_profile.get("t0", 0.0)),
        drive_time_sampling=str(
            "midpoint" if drive_profile is None else drive_profile.get("time_sampling", "midpoint")
        ),
        exact_steps_multiplier=(
            int(args.exact_steps_multiplier) if drive_profile is not None else 1
        ),
        propagator="suzuki2",
        cfqm_stage_exp=str(args.cfqm_stage_exp),
        cfqm_coeff_drop_abs_tol=float(args.cfqm_coeff_drop_abs_tol),
        cfqm_normalize=bool(args.cfqm_normalize),
    )

    return {
        "drive_enabled": bool(drive_profile is not None),
        "drive_profile": drive_profile,
        "drive_meta": drive_meta,
        "branch_semantics": {
            "legacy": "warm_start_hva",
            "paop": "adapt_pool_b",
            "hva": "final_seeded_conventional_vqe",
        },
        "trajectory": rows,
        "fidelity_subspace_energy_tol": float(fidelity_subspace_energy_tol),
        "final": {
            "fidelity_legacy": float(rows[-1]["fidelity"]),
            "fidelity_paop": float(rows[-1]["fidelity_paop_trotter"]),
            "fidelity_hva": float(rows[-1]["fidelity_hva_trotter"]),
            "energy_total_trotter_legacy": float(rows[-1]["energy_total_trotter"]),
            "energy_total_trotter_paop": float(rows[-1]["energy_total_trotter_paop"]),
            "energy_total_trotter_hva": float(rows[-1]["energy_total_trotter_hva"]),
        },
    }


def _run_noiseless_profile(
    *,
    args: argparse.Namespace,
    psi_seed: np.ndarray,
    hmat: np.ndarray,
    ordered_labels_exyz: list[str],
    coeff_map_exyz: dict[str, complex],
    drive_profile: dict[str, Any] | None,
) -> dict[str, Any]:
    nq = int(round(math.log2(int(np.asarray(psi_seed).size))))
    drive_provider_exyz, drive_meta = _drive_provider_from_profile(
        profile=drive_profile,
        num_sites=int(args.L),
        nq_total=int(nq),
        ordering=str(args.ordering),
    )

    methods = [
        ("suzuki2", "suzuki2"),
        ("magnus2", "piecewise_exact"),
        ("cfqm4", "cfqm4"),
        ("cfqm6", "cfqm6"),
    ]

    method_payloads: dict[str, Any] = {}
    reference_rows: list[dict[str, Any]] | None = None

    for method_name, propagator_key in methods:
        rows, _ = hc_pipeline._simulate_trajectory(
            num_sites=int(args.L),
            ordering=str(args.ordering),
            psi0_legacy_trot=np.asarray(psi_seed, dtype=complex),
            psi0_paop_trot=np.asarray(psi_seed, dtype=complex),
            psi0_hva_trot=np.asarray(psi_seed, dtype=complex),
            legacy_branch_label="seq",
            psi0_exact_ref=np.asarray(psi_seed, dtype=complex),
            fidelity_subspace_basis_v0=np.asarray(psi_seed, dtype=complex).reshape(-1, 1),
            fidelity_subspace_energy_tol=0.0,
            hmat=np.asarray(hmat, dtype=complex),
            ordered_labels_exyz=list(ordered_labels_exyz),
            coeff_map_exyz=dict(coeff_map_exyz),
            trotter_steps=int(args.trotter_steps),
            t_final=float(args.t_final),
            num_times=int(args.num_times),
            suzuki_order=2,
            drive_coeff_provider_exyz=drive_provider_exyz,
            drive_t0=float(0.0 if drive_profile is None else drive_profile.get("t0", 0.0)),
            drive_time_sampling=str(
                "midpoint" if drive_profile is None else drive_profile.get("time_sampling", "midpoint")
            ),
            exact_steps_multiplier=(
                int(args.exact_steps_multiplier) if drive_profile is not None else 1
            ),
            propagator=str(propagator_key),
            cfqm_stage_exp=str(args.cfqm_stage_exp),
            cfqm_coeff_drop_abs_tol=float(args.cfqm_coeff_drop_abs_tol),
            cfqm_normalize=bool(args.cfqm_normalize),
        )
        if reference_rows is None:
            reference_rows = rows

        method_payloads[method_name] = {
            "propagator": str(propagator_key),
            "trajectory": rows,
            "final": {
                "energy_total_trotter": float(rows[-1]["energy_total_trotter"]),
                "energy_total_exact": float(rows[-1]["energy_total_exact"]),
                "abs_energy_total_error": float(
                    abs(float(rows[-1]["energy_total_trotter"]) - float(rows[-1]["energy_total_exact"]))
                ),
                "fidelity": float(rows[-1]["fidelity"]),
                "doublon_trotter": float(rows[-1]["doublon_trotter"]),
                "doublon_exact": float(rows[-1]["doublon_exact"]),
            },
        }

    assert reference_rows is not None

    return {
        "drive_enabled": bool(drive_profile is not None),
        "drive_profile": drive_profile,
        "drive_meta": drive_meta,
        "times": [float(r["time"]) for r in reference_rows],
        "reference": {
            "energy_static_exact": [float(r["energy_static_exact"]) for r in reference_rows],
            "energy_total_exact": [float(r["energy_total_exact"]) for r in reference_rows],
            "n_up_site0_exact": [float(r["n_up_site0_exact"]) for r in reference_rows],
            "n_dn_site0_exact": [float(r["n_dn_site0_exact"]) for r in reference_rows],
            "doublon_exact": [float(r["doublon_exact"]) for r in reference_rows],
            "staggered_exact": [float(r["staggered_exact"]) for r in reference_rows],
            "method": (
                "eigendecomposition"
                if drive_profile is None
                else str(reference_method_name(str(drive_profile.get("time_sampling", "midpoint"))) )
            ),
        },
        "methods": method_payloads,
    }


def _compute_comparisons(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "noiseless_vs_exact": {},
        "noise_vs_noiseless": {},
        "noise_vs_noiseless_methods": {},
    }

    noiseless = payload.get("dynamics_noiseless", {})
    for profile_name, profile_data in noiseless.get("profiles", {}).items():
        methods = profile_data.get("methods", {})
        profile_cmp: dict[str, Any] = {}
        for method_name, method_data in methods.items():
            final = method_data.get("final", {})
            profile_cmp[method_name] = {
                "final_abs_energy_total_error": float(final.get("abs_energy_total_error", float("nan"))),
                "final_fidelity": float(final.get("fidelity", float("nan"))),
            }
        out["noiseless_vs_exact"][str(profile_name)] = profile_cmp

    noisy = payload.get("dynamics_noisy", {})
    for profile_name, profile_data in noisy.get("profiles", {}).items():
        method_payloads = profile_data.get("methods", {})
        if not isinstance(method_payloads, dict) or not method_payloads:
            method_payloads = {"suzuki2": {"modes": profile_data.get("modes", {})}}

        method_cmp: dict[str, Any] = {}
        for method_name, method_data in method_payloads.items():
            modes = method_data.get("modes", {}) if isinstance(method_data, dict) else {}
            comp_profile: dict[str, Any] = {}

            noisl_ref = (
                payload.get("dynamics_noiseless", {})
                .get("profiles", {})
                .get(profile_name, {})
                .get("methods", {})
                .get(str(method_name), {})
                .get("trajectory", [])
            )
            if not noisl_ref:
                noisl_ref = (
                    payload.get("dynamics_noiseless", {})
                    .get("profiles", {})
                    .get(profile_name, {})
                    .get("methods", {})
                    .get("suzuki2", {})
                    .get("trajectory", [])
                )

            e_ref_final = None
            d_ref_final = None
            if noisl_ref:
                e_ref_final = float(noisl_ref[-1]["energy_total_trotter"])
                d_ref_final = float(noisl_ref[-1]["doublon_trotter"])

            for mode_name, mode_data in modes.items():
                if not bool(mode_data.get("success", False)):
                    comp_profile[str(mode_name)] = {
                        "available": False,
                        "reason": str(mode_data.get("reason", mode_data.get("error", "unknown"))),
                    }
                    continue
                traj = mode_data.get("trajectory", [])
                if not traj:
                    comp_profile[str(mode_name)] = {"available": False, "reason": "empty_trajectory"}
                    continue

                e_noisy_final = float(traj[-1]["energy_total_noisy"])
                d_noisy_final = float(traj[-1]["doublon_noisy"])
                rec = {
                    "available": True,
                    "final_energy_total_noisy": e_noisy_final,
                    "final_doublon_noisy": d_noisy_final,
                    "final_energy_total_delta_noisy_minus_ideal": float(
                        traj[-1]["energy_total_delta_noisy_minus_ideal"]
                    ),
                    "final_doublon_delta_noisy_minus_ideal": float(
                        traj[-1]["doublon_delta_noisy_minus_ideal"]
                    ),
                }
                if e_ref_final is not None:
                    rec[f"final_energy_total_delta_noisy_minus_noiseless_{method_name}"] = float(e_noisy_final - e_ref_final)
                if d_ref_final is not None:
                    rec[f"final_doublon_delta_noisy_minus_noiseless_{method_name}"] = float(d_noisy_final - d_ref_final)
                comp_profile[str(mode_name)] = rec
            method_cmp[str(method_name)] = comp_profile

        out["noise_vs_noiseless_methods"][str(profile_name)] = method_cmp
        out["noise_vs_noiseless"][str(profile_name)] = dict(method_cmp.get("suzuki2", {}))

    return out


def _build_summary(payload: dict[str, Any]) -> dict[str, Any]:
    stage_pipeline = payload.get("stage_pipeline", {})
    warm = stage_pipeline.get("warm_start", {})
    adapt = stage_pipeline.get("adapt_pool_b", {})
    final = stage_pipeline.get("conventional_vqe", {})

    noisy = payload.get("dynamics_noisy", {}).get("profiles", {})
    noisy_completed = 0
    noisy_total = 0
    noisy_method_modes_completed = 0
    noisy_method_modes_total = 0
    for prof in noisy.values():
        modes = prof.get("modes", {}) if isinstance(prof, dict) else {}
        for mode_data in modes.values():
            noisy_total += 1
            if bool(mode_data.get("success", False)):
                noisy_completed += 1
        methods = prof.get("methods", {}) if isinstance(prof, dict) else {}
        if isinstance(methods, dict):
            for method_data in methods.values():
                method_modes = method_data.get("modes", {}) if isinstance(method_data, dict) else {}
                for mode_data in method_modes.values():
                    noisy_method_modes_total += 1
                    if bool(mode_data.get("success", False)):
                        noisy_method_modes_completed += 1

    dyn_bench = payload.get("dynamics_benchmarks", {})
    dyn_rows = dyn_bench.get("rows", []) if isinstance(dyn_bench, dict) else []

    return {
        "warm_delta_abs": float(warm.get("delta_abs", float("nan"))),
        "adapt_delta_abs": float(adapt.get("delta_abs", float("nan"))),
        "final_delta_abs": float(final.get("delta_abs", float("nan"))),
        "warm_stop_reason": str(warm.get("stop_reason", "")),
        "adapt_stop_reason": str(adapt.get("stop_reason", "")),
        "final_stop_reason": str(final.get("stop_reason", "")),
        "noisy_modes_completed": int(noisy_completed),
        "noisy_modes_total": int(noisy_total),
        "noisy_method_modes_completed": int(noisy_method_modes_completed),
        "noisy_method_modes_total": int(noisy_method_modes_total),
        "dynamics_benchmark_rows": int(len(dyn_rows) if isinstance(dyn_rows, list) else 0),
    }


def _build_equation_registry_and_contracts(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build equation registry and plot contracts for equation-first audit pages."""
    registry: dict[str, Any] = {}

    def _add(
        eq_id: str,
        *,
        latex: str,
        plain: str,
        symbols: dict[str, str],
        units: str,
        source_keys: list[str],
    ) -> None:
        registry[str(eq_id)] = {
            "latex": str(latex),
            "plain": str(plain),
            "symbols": dict(symbols),
            "units": str(units),
            "source_keys": [str(x) for x in source_keys],
        }

    _add(
        "eq_h_total",
        latex=r"H(t)=H_0 + H_{\mathrm{drive}}(t)",
        plain="Total Hamiltonian splits into static + drive terms.",
        symbols={"H0": "static HH Hamiltonian", "H_drive": "time-dependent onsite-density drive"},
        units="energy",
        source_keys=["settings.t", "settings.u", "settings.g_ep", "settings.drive_profile"],
    )
    _add(
        "eq_h_drive_density",
        latex=r"H_{\mathrm{drive}}(t)=\sum_{i,\sigma} v_i(t)\,n_{i\sigma}",
        plain="Drive couples to onsite electron number operators.",
        symbols={"i": "site index", "sigma": "spin channel"},
        units="energy",
        source_keys=["settings.drive_profile", "diagnostics.metric_definitions.energy_total"],
    )
    _add(
        "eq_drive_waveform",
        latex=r"v_i(t)=s_i A\sin[\omega(t+t_0)+\phi]\exp\!\left(-\frac{(t+t_0)^2}{2\bar t^2}\right)",
        plain="Gaussian-envelope sinusoidal drive with spatial weights s_i.",
        symbols={"A": "drive amplitude", "omega": "carrier angular frequency", "tbar": "Gaussian width", "phi": "phase", "s_i": "site weight"},
        units="energy",
        source_keys=["settings.drive_profile.A", "settings.drive_profile.omega", "settings.drive_profile.tbar"],
    )
    _add(
        "eq_number_operator",
        latex=r"n_{i\sigma}=\frac{I-Z_{q(i,\sigma)}}{2}",
        plain="JW number-operator identity used for occupancy observables.",
        symbols={"q(i,sigma)": "qubit index for spin-orbital (i,sigma)"},
        units="dimensionless",
        source_keys=["diagnostics.metric_definitions"],
    )
    _add(
        "eq_total_density",
        latex=r"n_i=\langle n_{i\uparrow}+n_{i\downarrow}\rangle",
        plain="Site-resolved total occupancy.",
        symbols={"n_i": "site occupancy expectation"},
        units="particles",
        source_keys=["dynamics_noiseless.profiles.*.methods.suzuki2.trajectory"],
    )
    _add(
        "eq_doublon",
        latex=r"D_i=\langle n_{i\uparrow}n_{i\downarrow}\rangle",
        plain="Onsite doublon expectation.",
        symbols={"D_i": "double occupancy at site i"},
        units="dimensionless",
        source_keys=["dynamics_noiseless.profiles.*.methods.suzuki2.trajectory"],
    )
    _add(
        "eq_staggered",
        latex=r"S=\frac{1}{L}\sum_{i=0}^{L-1}(-1)^i\langle n_i\rangle",
        plain="Staggered charge-density order parameter.",
        symbols={"L": "number of sites"},
        units="dimensionless",
        source_keys=["dynamics_noiseless.profiles.*.methods.suzuki2.trajectory"],
    )
    _add(
        "eq_energy_static",
        latex=r"E_{\mathrm{static}}^b(t)=\langle\psi_b(t)|H_0|\psi_b(t)\rangle",
        plain="Static-Hamiltonian energy channel for branch b.",
        symbols={"b": "branch (warm/adapt/final)"},
        units="energy",
        source_keys=["dynamics_noiseless.profiles.*.methods.suzuki2.trajectory"],
    )
    _add(
        "eq_energy_total",
        latex=r"E_{\mathrm{total}}^b(t)=\langle\psi_b(t)|H_0+H_{\mathrm{drive}}(t)|\psi_b(t)\rangle",
        plain="Instantaneous total energy under drive.",
        symbols={"b": "branch (warm/adapt/final)"},
        units="energy",
        source_keys=["dynamics_noiseless.profiles.*.methods.suzuki2.trajectory"],
    )
    _add(
        "eq_subspace_fidelity",
        latex=r"F_{\mathrm{sub}}^b(t)=\langle\psi_b^{\mathrm{trot}}(t)|P_{\mathrm{GS}}(t)|\psi_b^{\mathrm{trot}}(t)\rangle",
        plain="Projected fidelity against sector-filtered exact GS manifold.",
        symbols={"P_GS": "projector onto exact GS manifold"},
        units="probability",
        source_keys=["dynamics_noiseless.profiles.*.methods.suzuki2.trajectory"],
    )
    _add(
        "eq_delta_e",
        latex=r"\Delta E_k=E_k-E_{\mathrm{exact,sector}},\quad \delta_k=|\Delta E_k|",
        plain="Stage transition error metrics.",
        symbols={"k": "checkpoint index"},
        units="energy",
        source_keys=["stage_pipeline.*.transition.delta_abs_trace"],
    )
    _add(
        "eq_slope_switch",
        latex=r"\mathrm{switch}\iff |\mathrm{slope}(\delta_{k-w+1:k})|\le \varepsilon\ \text{for}\ p\ \text{consecutive checkpoints}",
        plain="Windowed abs-slope plateau rule for stage transitions.",
        symbols={"w": "window_k", "epsilon": "slope threshold", "p": "patience"},
        units="energy/checkpoint",
        source_keys=["settings.transition_policy"],
    )
    _add(
        "eq_noisy_estimator",
        latex=r"\mu_O=\mathrm{agg}(\{\langle O\rangle_r\}_{r=1}^{R}),\quad \sigma_O=\mathrm{std}(\{\langle O\rangle_r\}_{r=1}^{R})",
        plain="Noisy oracle aggregate and empirical spread.",
        symbols={"R": "oracle repeats", "agg": "mean/median aggregate"},
        units="observable units",
        source_keys=["settings.oracle_repeats", "settings.oracle_aggregate"],
    )
    _add(
        "eq_noisy_delta",
        latex=r"\Delta_O^{\mathrm{noise-ideal}}(t)=\mu_O^{\mathrm{noise}}(t)-\mu_O^{\mathrm{ideal}}(t)",
        plain="Noisy-vs-ideal observable delta.",
        symbols={"O": "observable channel"},
        units="observable units",
        source_keys=["dynamics_noisy.profiles.*.modes.*.trajectory.*.*_delta_noisy_minus_ideal"],
    )
    _add(
        "eq_proxy_cost",
        latex=r"\mathrm{cx\_proxy\_term}(p)=2\max(w(p)-1,0),\quad \mathrm{sq\_proxy\_term}(p)=2\,xy(p)+1",
        plain="Per-term hardware proxy definitions for dynamics benchmark totals.",
        symbols={"w(p)": "Pauli weight", "xy(p)": "count of X/Y letters"},
        units="proxy-count",
        source_keys=["dynamics_noisy.profiles.*.methods.*.modes.*.benchmark_cost"],
    )
    _add(
        "eq_runtime_bench",
        latex=r"t_{\mathrm{wall}}=t_{\mathrm{build}}+t_{\mathrm{oracle}}+\cdots",
        plain="Runtime decomposition for noisy dynamics evaluation.",
        symbols={"t_wall": "wall-clock total", "t_oracle": "oracle evaluation subtotal"},
        units="seconds",
        source_keys=["dynamics_noisy.profiles.*.methods.*.modes.*.benchmark_runtime"],
    )
    _add(
        "eq_suzuki2",
        latex=r"U_{\mathrm{S2}}(\Delta t)\approx\prod_j e^{-i\frac{\Delta t}{2}H_j}\prod_{j}^{\mathrm{rev}}e^{-i\frac{\Delta t}{2}H_j}",
        plain="Second-order symmetric Suzuki product step.",
        symbols={"H_j": "Hamiltonian term blocks"},
        units="unitary",
        source_keys=["dynamics_noiseless.profiles.*.methods.suzuki2"],
    )
    _add(
        "eq_magnus2",
        latex=r"U_{\mathrm{Magnus2}}(\Delta t)=\exp\!\left[-i\Delta t\,H\!\left(t+\frac{\Delta t}{2}\right)\right]",
        plain="Exponential midpoint (Magnus-2) reference approximation.",
        symbols={"Delta t": "macro step size"},
        units="unitary",
        source_keys=["dynamics_noiseless.profiles.*.methods.magnus2"],
    )
    _add(
        "eq_cfqm",
        latex=r"U_{\mathrm{CFQM}}(\Delta t)\approx\prod_{m} \exp[-i\,a_m\Delta t\,H(t+c_m\Delta t)]",
        plain="Commutator-free Magnus stage product (CFQM4/CFQM6).",
        symbols={"a_m,c_m": "scheme coefficients"},
        units="unitary",
        source_keys=["dynamics_noiseless.profiles.*.methods.cfqm4", "dynamics_noiseless.profiles.*.methods.cfqm6"],
    )
    _add(
        "eq_error_abs",
        latex=r"\epsilon_X(t)=|X_{\mathrm{approx}}(t)-X_{\mathrm{exact}}(t)|",
        plain="Absolute trajectory error metric.",
        symbols={"X": "observable channel"},
        units="observable units",
        source_keys=["comparisons.noiseless_vs_exact", "comparisons.noise_vs_noiseless"],
    )

    profiles = ["static", "drive"]
    observables = ["energy_total", "energy_static", "doublon", "staggered", "n_up_site0", "n_dn_site0"]
    for profile in profiles:
        for obs in observables:
            eq_id = f"eq_{obs}_{profile}_final_seed"
            _add(
                eq_id,
                latex=rf"{obs}^{{\mathrm{{final\_seed}}}}_{{{profile}}}(t)",
                plain=f"{obs} channel for {profile} profile, propagated from final-seed state only.",
                symbols={"profile": profile, "seed": "final VQE state"},
                units="observable units",
                source_keys=[f"dynamics_noiseless.profiles.{profile}.methods.suzuki2.trajectory"],
            )

    modes = ["ideal", "shots", "aer_noise"]
    noisy_obs = ["energy_total", "energy_static", "doublon", "staggered", "n_up_site0", "n_dn_site0"]
    for profile in profiles:
        for mode in modes:
            for obs in noisy_obs:
                eq_id = f"eq_noisy_{obs}_{profile}_{mode}"
                _add(
                    eq_id,
                    latex=rf"\mu_{{{obs}}}^{{{mode},{profile}}}(t)",
                    plain=f"Noisy estimate for {obs} in {profile}/{mode}.",
                    symbols={"mode": mode, "profile": profile},
                    units="observable units",
                    source_keys=[f"dynamics_noisy.profiles.{profile}.modes.{mode}.trajectory"],
                )

    contracts: dict[str, Any] = {}
    contracts["plot_stage_transition"] = {
        "x": "checkpoint index",
        "y": ["delta_abs_trace", "slope_trace"],
        "source": ["stage_pipeline.warm_start.transition", "stage_pipeline.adapt_pool_b.transition"],
        "notes": "Windowed abs-slope switching traces.",
    }
    for profile in profiles:
        contracts[f"plot_{profile}_magnus_cfqm_overlay"] = {
            "x": "time grid",
            "y": ["suzuki2.energy_total_trotter", "magnus2.energy_total_trotter", "cfqm4.energy_total_trotter", "cfqm6.energy_total_trotter", "exact.energy_total_exact"],
            "source": [f"dynamics_noiseless.profiles.{profile}.methods.*.trajectory"],
            "notes": "Noiseless method comparison.",
        }
        contracts[f"plot_{profile}_magnus_cfqm_error"] = {
            "x": "time grid",
            "y": ["|E_method-E_exact|", "|F_method-1|"],
            "source": [f"dynamics_noiseless.profiles.{profile}.methods.*.trajectory"],
            "notes": "Noiseless error channels vs exact.",
        }
        contracts[f"plot_{profile}_overlay_energy"] = {
            "x": "time grid",
            "y": ["noiseless+suzuki2+exact+noisy energy overlays"],
            "source": [f"dynamics_noisy.profiles.{profile}.modes.*.trajectory", f"dynamics_noiseless.profiles.{profile}.methods.suzuki2.trajectory"],
            "notes": "Overlay page for exact/noiseless/noisy energies.",
        }
        contracts[f"plot_{profile}_overlay_occupancy"] = {
            "x": "time grid",
            "y": ["n_up(0)", "n_dn(0) overlays"],
            "source": [f"dynamics_noisy.profiles.{profile}.modes.*.trajectory", f"dynamics_noiseless.profiles.{profile}.methods.suzuki2.trajectory"],
            "notes": "Overlay page for site-0 occupancies.",
        }
        contracts[f"plot_{profile}_overlay_doublon_staggered"] = {
            "x": "time grid",
            "y": ["doublon", "staggered overlays"],
            "source": [f"dynamics_noisy.profiles.{profile}.modes.*.trajectory", f"dynamics_noiseless.profiles.{profile}.methods.suzuki2.trajectory"],
            "notes": "Overlay page for doublon and staggered order.",
        }
        for mode in modes:
            contracts[f"plot_{profile}_{mode}_energy"] = {
                "x": "time grid",
                "y": ["energy_static_noisy", "energy_static_ideal", "energy_total_noisy", "energy_total_ideal"],
                "source": [f"dynamics_noisy.profiles.{profile}.modes.{mode}.trajectory"],
                "notes": "Noisy energy channels and noisy-ideal deltas.",
            }
            contracts[f"plot_{profile}_{mode}_occupancy"] = {
                "x": "time grid",
                "y": ["n_up_site0_noisy", "n_up_site0_ideal", "n_dn_site0_noisy", "n_dn_site0_ideal"],
                "source": [f"dynamics_noisy.profiles.{profile}.modes.{mode}.trajectory"],
                "notes": "Noisy site-0 spin channels and deltas.",
            }
            contracts[f"plot_{profile}_{mode}_doublon_staggered"] = {
                "x": "time grid",
                "y": ["doublon_noisy", "doublon_ideal", "staggered_noisy", "staggered_ideal"],
                "source": [f"dynamics_noisy.profiles.{profile}.modes.{mode}.trajectory"],
                "notes": "Noisy doublon/staggered channels and deltas.",
            }

    return registry, contracts


def _apply_report_theme(plt: Any) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.5,
            "figure.dpi": 220,
            "savefig.dpi": 220,
            "axes.grid": False,
        }
    )


def _render_page_header_footer(fig: Any, *, header: str, footer: str) -> None:
    fig.text(0.5, 0.985, str(header), ha="center", va="top", fontsize=9, color="#222222")
    fig.text(0.5, 0.012, str(footer), ha="center", va="bottom", fontsize=8, color="#333333")


_ACTIVE_CAPTION_OVERRIDES: dict[str, list[str]] = {}


def _build_caption_overrides(shots: int) -> dict[str, list[str]]:
    shots_text = f"{int(shots)} shots" if int(shots) > 0 else "configured shots"
    mode_shot_text = f"mode=ideal, {shots_text}"
    mapping: dict[str, list[str]] = {
        "plot_stage_transition": [
            "Stage switching: $|\\Delta E|$ vs checkpoint for warm start and ADAPT.",
            "Dashed trace is the windowed slope used by the plateau rule ($w=5$, $\\epsilon=5\\times10^{-5}$, patience=3).",
        ],
        "plot_static_density_total_heatmaps": [
            "Static: total occupancy n_i(t) heatmaps (sites i=0..L-1).",
            "Panels: exact GS (left), exact evolution (middle), noiseless ADAPT-HVA + Suzuki-2 (right).",
        ],
        "plot_static_density_up_heatmaps": [
            "Static: spin-up occupancy n_up_i(t) heatmaps.",
            "Panels: exact GS (left), exact evolution (middle), noiseless ADAPT-HVA + Suzuki-2 (right).",
        ],
        "plot_static_density_dn_heatmaps": [
            "Static: spin-down occupancy n_dn_i(t) heatmaps.",
            "Panels: exact GS (left), exact evolution (middle), noiseless ADAPT-HVA + Suzuki-2 (right).",
        ],
        "plot_static_energy_fidelity_final": [
            "Static (final-seed trajectory): top F_sub, middle E_total, bottom doublon vs time.",
            "Exact reference vs noiseless ADAPT-HVA + Suzuki-2.",
        ],
        "plot_drive_density_total_heatmaps": [
            "Drive: total occupancy n_i(t) heatmaps (sites i=0..L-1).",
            "Panels: exact GS (left), exact evolution (middle), noiseless ADAPT-HVA + Suzuki-2 (right).",
        ],
        "plot_drive_density_up_heatmaps": [
            "Drive: spin-up occupancy n_up_i(t) heatmaps.",
            "Panels: exact GS (left), exact evolution (middle), noiseless ADAPT-HVA + Suzuki-2 (right).",
        ],
        "plot_drive_density_dn_heatmaps": [
            "Drive: spin-down occupancy n_dn_i(t) heatmaps.",
            "Panels: exact GS (left), exact evolution (middle), noiseless ADAPT-HVA + Suzuki-2 (right).",
        ],
        "plot_drive_energy_fidelity_final": [
            "Drive (final-seed trajectory): top F_sub, middle E_total, bottom doublon vs time.",
            "Exact reference vs noiseless ADAPT-HVA + Suzuki-2.",
        ],
        "plot_static_magnus_cfqm_overlay": [
            "Static: noiseless integrator comparison - top E_total, bottom F_sub.",
            "Methods: Suzuki-2 / Magnus-2 / CFQM4 / CFQM6; exact reference in black.",
        ],
        "plot_static_magnus_cfqm_error": [
            "Static: integrator error vs exact - top |E-E_exact|, bottom (1-F_sub).",
            "Smaller is better for ranking Suzuki-2 vs Magnus/CFQM baselines.",
        ],
        "plot_drive_magnus_cfqm_overlay": [
            "Drive: noiseless integrator comparison - top E_total, bottom F_sub.",
            "Methods: Suzuki-2 / Magnus-2 / CFQM4 / CFQM6; exact reference in black.",
        ],
        "plot_drive_magnus_cfqm_error": [
            "Drive: integrator error vs exact - top |E-E_exact|, bottom (1-F_sub).",
            "Smaller is better for ranking Suzuki-2 vs Magnus/CFQM baselines.",
        ],
        "plot_static_overlay_energy": [
            "Static: energy vs time - top E_total, bottom E_static.",
            f"Exact vs noiseless (ADAPT-HVA+Suzuki-2) vs noisy ({mode_shot_text}).",
        ],
        "plot_static_overlay_occupancy": [
            "Static: site-0 occupations vs time - top $n_{\\uparrow}(0)$, bottom $n_{\\downarrow}(0)$.",
            f"Noiseless vs noisy ({mode_shot_text}).",
        ],
        "plot_static_overlay_doublon_staggered": [
            "Static: correlators vs time - top doublon, bottom staggered order S.",
            f"Noiseless vs noisy ({mode_shot_text}).",
        ],
        "plot_drive_overlay_energy": [
            "Drive: energy vs time - top E_total, bottom E_static.",
            f"Exact vs noiseless (ADAPT-HVA+Suzuki-2) vs noisy ({mode_shot_text}).",
        ],
        "plot_drive_overlay_occupancy": [
            "Drive: site-0 occupations vs time - top $n_{\\uparrow}(0)$, bottom $n_{\\downarrow}(0)$.",
            f"Noiseless vs noisy ({mode_shot_text}).",
        ],
        "plot_drive_overlay_doublon_staggered": [
            "Drive: correlators vs time - top doublon, bottom staggered order S.",
            f"Noiseless vs noisy ({mode_shot_text}).",
        ],
        "plot_static_scalar_error_heatmap": [
            "Static: absolute scalar error channels heatmap (rows are |E|, |D|, and |S| error metrics).",
            "Columns run over time; brighter color means larger deviation from the matched exact channel.",
        ],
        "plot_drive_scalar_error_heatmap": [
            "Drive: absolute scalar error channels heatmap (rows are |E|, |D|, and |S| error metrics).",
            "Columns run over time; brighter color means larger deviation from the matched exact channel.",
        ],
        "plot_drive_waveform": [
            "Drive waveform v(t) evaluated on the simulation time grid.",
            "Gaussian-envelope sinusoid used in H_drive(t)=sum_i,sigma v_i(t) n_i,sigma.",
        ],
        "plot_noisy_benchmark_table": [
            "Noisy dynamics benchmark table: profile/method/mode rows.",
            "Reports proxy costs (term-exp, cx, sq, depth) and runtime totals (wall/oracle).",
        ],
    }

    for profile in ("static", "drive"):
        profile_label = "Static" if profile == "static" else "Drive"
        for mode in ("ideal", "shots", "aer_noise"):
            mode_label = mode.replace("_", " ")
            mapping[f"plot_{profile}_{mode}_energy"] = [
                f"{profile_label} / mode={mode_label}: top E_total, middle E_static, bottom $\\Delta$(measured-ideal-ref).",
                "Shows estimator deviation from its internal ideal reference (not the ED exact curve).",
            ]
            mapping[f"plot_{profile}_{mode}_occupancy"] = [
                f"{profile_label} / mode={mode_label}: top $n_{{\\uparrow}}(0)$, middle $n_{{\\downarrow}}(0)$, bottom $\\Delta$(measured-ideal-ref).",
                "Use the $\\Delta$ panel to isolate estimator/noise bias over time.",
            ]
            mapping[f"plot_{profile}_{mode}_doublon_staggered"] = [
                f"{profile_label} / mode={mode_label}: top doublon, middle staggered S, bottom $\\Delta$(measured-ideal-ref).",
                "Use the $\\Delta$ panel to isolate estimator/noise bias over time.",
            ]
    return mapping


def _fallback_caption_line(plot_id: str, plot_contracts: dict[str, Any]) -> str:
    pid = str(plot_id)
    if pid.startswith("plot_static_"):
        section = "Static"
    elif pid.startswith("plot_drive_"):
        section = "Drive"
    elif pid.startswith("plot_stage_"):
        section = "Stage"
    else:
        section = "Report"
    rec = plot_contracts.get(pid, {})
    notes = str(rec.get("notes", "")).strip()
    if notes:
        title = notes.rstrip(".")
    else:
        title = pid.replace("plot_", "").replace("_", " ").strip()
    return f"{section}: {title}."


def _extract_numeric_caption_line(style_legend_lines: list[str]) -> str:
    for raw in style_legend_lines:
        line = str(raw).strip()
        if ("max|Δ|" in line) or ("RMS|Δ|" in line) or ("max|Delta|" in line) or ("RMS|Delta|" in line):
            return line.rstrip(".") + "."
    return ""


def _disabled_hardcoded_superset_meta() -> dict[str, Any]:
    return {
        "profiles": {},
        "disabled": True,
        "reason": "branch propagation deactivated; final-only dynamics",
    }


def _noise_style_legend_lines() -> list[str]:
    return [
        "noise ideal   : #ff7f0e",
        "noise shots   : #2ca02c",
        "noise aer     : #d62728",
        "noiseless (final-seed Suzuki-2) : #1f77b4",
        "exact reference                    : #111111",
        "solid mode color = noisy measured; dashed mode color = ideal reference for mode",
        "dotted black horizontal = zero baseline for Δ(noisy-ideal)",
    ]


def _latex_to_unicode_math(expr: str) -> str:
    text = str(expr)
    if not text:
        return ""
    # Reduce common wrappers first.
    text = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    # Flatten simple fractions repeatedly.
    frac_pat = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    while True:
        new_text = frac_pat.sub(r"(\1)/(\2)", text)
        if new_text == text:
            break
        text = new_text
    repl = {
        r"\Delta": "Δ",
        r"\delta": "δ",
        r"\epsilon": "ε",
        r"\omega": "ω",
        r"\phi": "φ",
        r"\bar t": "t̄",
        r"\sum": "Σ",
        r"\prod": "∏",
        r"\approx": "≈",
        r"\le": "≤",
        r"\ge": "≥",
        r"\iff": "⇔",
        r"\cdot": "·",
        r"\times": "×",
        r"\langle": "⟨",
        r"\rangle": "⟩",
        r"\left": "",
        r"\right": "",
    }
    for src, dst in repl.items():
        text = text.replace(src, dst)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("\\", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _latex_to_mathtext(expr: str) -> str:
    text = str(expr)
    if not text:
        return ""
    text = re.sub(r"\\text\{([^{}]*)\}", r"\\mathrm{\1}", text)
    text = text.replace(r"\iff", r"\Leftrightarrow")
    text = re.sub(r"\\le(?![A-Za-z])", r"\\leq", text)
    text = re.sub(r"\\ge(?![A-Za-z])", r"\\geq", text)
    text = text.replace(r"\quad", r"\;")
    text = text.replace(r"\!", "")
    return text


def _annotate_plot_with_equations(
    fig: Any,
    *,
    eq_ids: list[str],
    equation_registry: dict[str, Any],
    plot_id: str,
    plot_contracts: dict[str, Any],
    style_legend_lines: list[str],
) -> None:
    """Attach short human-readable caption strip below plots."""
    _ = eq_ids
    _ = equation_registry
    override = _ACTIVE_CAPTION_OVERRIDES.get(str(plot_id), [])
    if override:
        caption_lines = [str(x).strip() for x in override if str(x).strip()]
    else:
        caption_lines = [_fallback_caption_line(str(plot_id), plot_contracts)]

    numeric_line = _extract_numeric_caption_line(style_legend_lines)
    if numeric_line and len(caption_lines) < 3:
        caption_lines.append(numeric_line)

    caption_lines = caption_lines[:3]
    if caption_lines:
        caption_lines[0] = f"[PLOT_CAPTION] {caption_lines[0]}"
    else:
        caption_lines = ["[PLOT_CAPTION]"]

    caption = "\n".join(caption_lines[:3])
    fig.subplots_adjust(bottom=0.20, top=0.93)
    fig.text(
        0.02,
        0.04,
        caption,
        ha="left",
        va="bottom",
        fontsize=7.7,
        family="DejaVu Sans",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f7f7f7", "edgecolor": "#303030", "alpha": 0.98},
    )
    _render_page_header_footer(
        fig,
        header=f"Hubbard-Holstein report | plot={plot_id}",
        footer=f"generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}",
    )


def _matrix_from_rows(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([[float(v) for v in row[key]] for row in rows], dtype=float)


def _render_formula_atlas(
    pdf: Any,
    *,
    equation_registry: dict[str, Any],
    title: str = "SECTION: APPENDIX FORMULA ATLAS",
    keys: list[str] | None = None,
    include_sources: bool = True,
) -> None:
    require_matplotlib()
    plt = get_plt()
    selected = sorted(list(keys if keys is not None else equation_registry.keys()))
    if not selected:
        return
    per_page = 6 if include_sources else 8
    for start in range(0, len(selected), per_page):
        chunk_keys = selected[start : start + per_page]
        fig = plt.figure(figsize=(8.5, 11.0))
        fig.patch.set_facecolor("white")
        fig.text(0.06, 0.965, title, ha="left", va="top", fontsize=11, fontweight="bold")
        fig.text(0.06, 0.942, "Formula index for this report.", ha="left", va="top", fontsize=9)

        y = 0.91
        for eq_id in chunk_keys:
            rec = equation_registry.get(eq_id, {})
            fig.text(0.06, y, f"[{eq_id}]", ha="left", va="top", fontsize=8.5, fontweight="bold")
            y -= 0.020

            latex_expr = _latex_to_mathtext(rec.get("latex", ""))
            rendered = False
            if latex_expr:
                try:
                    fig.text(0.08, y, f"${latex_expr}$", ha="left", va="top", fontsize=10)
                    rendered = True
                except Exception:
                    rendered = False
            if not rendered:
                fig.text(0.08, y, _latex_to_unicode_math(rec.get("latex", "")), ha="left", va="top", fontsize=9)
            y -= 0.026

            plain = str(rec.get("plain", "")).strip()
            units = str(rec.get("units", "")).strip()
            if plain:
                fig.text(0.08, y, f"Meaning: {plain}", ha="left", va="top", fontsize=8)
                y -= 0.018
            if units:
                fig.text(0.08, y, f"Units: {units}", ha="left", va="top", fontsize=8)
                y -= 0.018

            if include_sources:
                src = rec.get("source_keys", [])
                src_list = [str(x) for x in src] if isinstance(src, list) else [str(src)]
                if len(src_list) > 2:
                    src_text = ", ".join(src_list[:2]) + f", ... (+{len(src_list)-2} more)"
                else:
                    src_text = ", ".join(src_list)
                fig.text(0.08, y, f"Source: {src_text}", ha="left", va="top", fontsize=7)
                y -= 0.018
            y -= 0.010

        _render_page_header_footer(
            fig,
            header="Hubbard-Holstein report | formula atlas",
            footer=f"generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}",
        )
        pdf.savefig(fig)
        plt.close(fig)


def _write_pdf(pdf_path: Path, payload: dict[str, Any]) -> None:
    require_matplotlib()
    plt = get_plt()
    PdfPages = get_PdfPages()
    _apply_report_theme(plt)

    settings = payload.get("settings", {})
    stage_pipeline = payload.get("stage_pipeline", {})
    pool_B = payload.get("pool_B", {})
    noiseless = payload.get("dynamics_noiseless", {}).get("profiles", {})
    noisy = payload.get("dynamics_noisy", {}).get("profiles", {})
    hardcoded_superset = payload.get("hardcoded_superset", {})
    comparisons = payload.get("comparisons", {})
    diagnostics = payload.get("diagnostics", {})
    equation_registry = payload.get("equation_registry", {})
    plot_contracts = payload.get("plot_contracts", {})
    global _ACTIVE_CAPTION_OVERRIDES
    _ACTIVE_CAPTION_OVERRIDES = _build_caption_overrides(int(settings.get("shots", 0)))

    if not isinstance(equation_registry, dict) or len(equation_registry) == 0:
        equation_registry, plot_contracts = _build_equation_registry_and_contracts(payload)

    branch_style_lines = [
        "color warm (legacy)  : #1f77b4",
        "color adapt (paop)   : #2ca02c",
        "color final (hva)    : #d62728",
        "color exact reference: #111111",
        "Warm/ADAPT/final are optimization checkpoints only.",
        "Dynamics/noise pages are seeded from final VQE only.",
    ]
    method_style_lines = [
        "method suzuki2: #1f77b4",
        "method magnus2: #ff7f0e",
        "method cfqm4  : #2ca02c",
        "method cfqm6  : #d62728",
        "exact ref     : #111111",
    ]
    noise_style_lines = _noise_style_legend_lines()

    with PdfPages(str(pdf_path)) as pdf:
        render_parameter_manifest(
            pdf,
            model="Hubbard-Holstein",
            ansatz="warm:hh_hva_ptw -> adapt:PoolB(UCCSD+HVA+PAOP_FULL) -> final:hh_hva_ptw",
            drive_enabled=True,
            t=float(settings.get("t", 0.0)),
            U=float(settings.get("u", 0.0)),
            dv=float(settings.get("dv", 0.0)),
            extra={
                "L": settings.get("L"),
                "n_ph_max": settings.get("n_ph_max"),
                "omega0": settings.get("omega0"),
                "g_ep": settings.get("g_ep"),
                "noise_modes": settings.get("noise_modes"),
                "noisy_methods": settings.get("noisy_methods"),
                "trotter_steps": settings.get("trotter_steps"),
                "num_times": settings.get("num_times"),
                "magnus_variants": "midpoint_magnus2,cfqm4,cfqm6",
                "hardcoded_family": "deactivated(final-only dynamics)",
            },
            command=str(payload.get("run_command", "")),
        )

        summary = payload.get("summary", {})
        used_eq_ids: set[str] = set()
        lines = [
            "SECTION: RESULTS SUMMARY",
            "",
            "HH Noise Robustness Report Summary",
            "",
            f"final delta_abs: {summary.get('final_delta_abs')}",
            f"warm stop: {summary.get('warm_stop_reason')}",
            f"adapt stop: {summary.get('adapt_stop_reason')}",
            f"final stop: {summary.get('final_stop_reason')}",
            f"noisy modes completed: {summary.get('noisy_modes_completed')} / {summary.get('noisy_modes_total')}",
            f"noisy method-modes completed: {summary.get('noisy_method_modes_completed')} / {summary.get('noisy_method_modes_total')}",
            "",
            "Transition policies:",
            f"  warm->adapt: {stage_pipeline.get('warm_start', {}).get('transition', {}).get('policy', {})}",
            f"  adapt->vqe: {stage_pipeline.get('adapt_pool_b', {}).get('transition', {}).get('policy', {})}",
            "",
            "Definitions:",
            "  DeltaE_k = E_k - E_exact_sector",
            "  switch when abs(slope( |DeltaE| window )) <= epsilon for patience checkpoints",
            "",
            "Noisy-vs-noiseless policy:",
            f"  noisy methods: {settings.get('noisy_methods', ['suzuki2'])} with modes {settings.get('noise_modes', ['ideal','shots','aer_noise'])}.",
            "  Magnus/CFQM remain in noiseless matrix; noisy matrix includes selected methods above.",
            "",
            "SECTION: HARDCODED FAMILY REMAP",
            "  legacy lane -> warm-start HVA seed",
            "  paop lane   -> ADAPT Pool-B seed",
            "  hva lane    -> final seeded conventional VQE",
        ]
        render_text_page(pdf, lines, fontsize=9)
        render_text_page(
            pdf,
            [
                "SECTION: RESULTS STAGE TRANSITIONS",
                "",
                "Transition traces and Pool-B composition.",
            ],
            fontsize=10,
        )

        # Stage transition slopes
        fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=False)
        fig.subplots_adjust(right=0.68)
        for ax, stage_key, title in [
            (axes[0], "warm_start", "Warm-start transition trace"),
            (axes[1], "adapt_pool_b", "ADAPT transition trace"),
        ]:
            trans = stage_pipeline.get(stage_key, {}).get("transition", {})
            d = np.asarray(trans.get("delta_abs_trace", []), dtype=float)
            s = np.asarray(trans.get("slope_trace", []), dtype=float)
            ax.set_title(title)
            if d.size > 0:
                ax.plot(np.arange(d.size), d, color="#1f77b4", linewidth=1.4, label="|DeltaE|")
                ax.set_ylabel("|DeltaE|")
            if s.size > 0:
                ax2 = ax.twinx()
                ax2.plot(np.arange(max(0, d.size - s.size), d.size), s, color="#d62728", linewidth=1.2, linestyle="--", label="slope")
                ax2.set_ylabel("slope")
            ax.grid(alpha=0.25)
        axes[1].set_xlabel("checkpoint index")
        _annotate_plot_with_equations(
            fig,
            eq_ids=["eq_delta_e", "eq_slope_switch"],
            equation_registry=equation_registry,
            plot_id="plot_stage_transition",
            plot_contracts=plot_contracts,
            style_legend_lines=branch_style_lines,
        )
        used_eq_ids.update(["eq_delta_e", "eq_slope_switch"])
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Pool B accounting table
        fig_tbl, ax_tbl = plt.subplots(figsize=(9.5, 5.0))
        fig_tbl.subplots_adjust(right=0.68)
        rows = [
            ["raw uccsd", str(pool_B.get("raw_sizes", {}).get("uccsd", ""))],
            ["raw hva", str(pool_B.get("raw_sizes", {}).get("hva", ""))],
            ["raw paop_full", str(pool_B.get("raw_sizes", {}).get("paop_full", ""))],
            ["dedup total", str(pool_B.get("dedup_total", ""))],
            ["presence uccsd", str(pool_B.get("dedup_source_presence_counts", {}).get("uccsd", ""))],
            ["presence hva", str(pool_B.get("dedup_source_presence_counts", {}).get("hva", ""))],
            ["presence paop_full", str(pool_B.get("dedup_source_presence_counts", {}).get("paop_full", ""))],
            ["overlap count", str(pool_B.get("overlap_count", ""))],
        ]
        render_compact_table(
            ax_tbl,
            title="Pool B Composition (strict union + dedup)",
            col_labels=["Metric", "Value"],
            rows=rows,
        )
        _annotate_plot_with_equations(
            fig_tbl,
            eq_ids=["eq_delta_e"],
            equation_registry=equation_registry,
            plot_id="plot_stage_transition",
            plot_contracts=plot_contracts,
            style_legend_lines=branch_style_lines,
        )
        used_eq_ids.update(["eq_delta_e"])
        fig_tbl.tight_layout()
        pdf.savefig(fig_tbl)
        plt.close(fig_tbl)

        warm_stage = stage_pipeline.get("warm_start", {}) if isinstance(stage_pipeline, dict) else {}
        adapt_stage = stage_pipeline.get("adapt_pool_b", {}) if isinstance(stage_pipeline, dict) else {}
        final_stage = stage_pipeline.get("conventional_vqe", {}) if isinstance(stage_pipeline, dict) else {}
        render_text_page(
            pdf,
            [
                "SECTION: RESULTS STAGE OPTIMIZATION SUMMARY",
                "",
                "Warm/ADAPT/final are optimization checkpoints only.",
                "They are not independently time-evolved in this report path.",
                "Dynamics/noise comparisons below are seeded from final VQE only.",
                "",
                f"warm_start: energy={warm_stage.get('energy')} delta_abs={warm_stage.get('delta_abs')} stop={warm_stage.get('stop_reason')}",
                f"adapt_pool_b: energy={adapt_stage.get('energy')} delta_abs={adapt_stage.get('delta_abs')} stop={adapt_stage.get('stop_reason')}",
                f"conventional_vqe: energy={final_stage.get('energy')} delta_abs={final_stage.get('delta_abs')} stop={final_stage.get('stop_reason')}",
                "",
                f"hardcoded_superset disabled: {bool(hardcoded_superset.get('disabled', False))}",
                f"reason: {hardcoded_superset.get('reason', '')}",
            ],
            fontsize=10,
        )

        render_text_page(
            pdf,
            [
                "SECTION: RESULTS METHOD COMPARISON",
                "",
                "Suzuki / Magnus / CFQM comparison overlays versus exact reference.",
                "",
                "Legend key (applies to this section):",
                "  Exact: exact reference evolution.",
                "  Noiseless: ADAPT-HVA pipeline on noiseless simulator.",
                "  Integrators: Suzuki-2, Magnus-2, CFQM4, CFQM6.",
            ],
            fontsize=10,
        )

        # Noiseless overlays for static + drive (Magnus/CFQM extension)
        for profile_name in ("static", "drive"):
            pdata = noiseless.get(profile_name)
            if not isinstance(pdata, dict):
                continue
            times = np.asarray(pdata.get("times", []), dtype=float)
            if times.size == 0:
                continue
            ref = pdata.get("reference", {})
            e_ref = np.asarray(ref.get("energy_total_exact", []), dtype=float)

            fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)
            fig.subplots_adjust(right=0.68)
            axes[0].plot(times, e_ref, color="#111111", linewidth=2.0, label="exact reference")

            for name, color in [
                ("suzuki2", "#1f77b4"),
                ("magnus2", "#ff7f0e"),
                ("cfqm4", "#2ca02c"),
                ("cfqm6", "#d62728"),
            ]:
                m = pdata.get("methods", {}).get(name, {})
                traj = m.get("trajectory", [])
                if not traj:
                    continue
                e = np.asarray([float(r["energy_total_trotter"]) for r in traj], dtype=float)
                f = np.asarray([float(r["fidelity"]) for r in traj], dtype=float)
                axes[0].plot(times, e, linewidth=1.2, label=name, color=color)
                axes[1].plot(times, f, linewidth=1.2, label=name, color=color)

            axes[0].set_ylabel("Energy total")
            axes[0].set_title(f"Noiseless methods overlay ({profile_name})")
            axes[0].grid(alpha=0.25)
            axes[0].legend(fontsize=8, ncol=3)
            axes[1].set_ylabel("Subspace fidelity")
            axes[1].set_xlabel("Time")
            axes[1].grid(alpha=0.25)
            axes[1].legend(fontsize=8, ncol=3)
            _annotate_plot_with_equations(
                fig,
                eq_ids=["eq_energy_total", "eq_subspace_fidelity", "eq_suzuki2", "eq_magnus2", "eq_cfqm"],
                equation_registry=equation_registry,
                plot_id=f"plot_{profile_name}_magnus_cfqm_overlay",
                plot_contracts=plot_contracts,
                style_legend_lines=method_style_lines,
            )
            used_eq_ids.update(["eq_energy_total", "eq_subspace_fidelity", "eq_suzuki2", "eq_magnus2", "eq_cfqm"])
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # Method error page vs exact
            fig_err, axes_err = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)
            fig_err.subplots_adjust(right=0.68)
            for name, color in [
                ("suzuki2", "#1f77b4"),
                ("magnus2", "#ff7f0e"),
                ("cfqm4", "#2ca02c"),
                ("cfqm6", "#d62728"),
            ]:
                traj = pdata.get("methods", {}).get(name, {}).get("trajectory", [])
                if not traj:
                    continue
                e = np.asarray([float(r["energy_total_trotter"]) for r in traj], dtype=float)
                f = np.asarray([float(r["fidelity"]) for r in traj], dtype=float)
                axes_err[0].plot(times, np.abs(e - e_ref), color=color, linewidth=1.3, label=name)
                axes_err[1].plot(times, np.abs(1.0 - f), color=color, linewidth=1.3, label=name)
            axes_err[0].set_ylabel("|E_method-E_exact|")
            axes_err[0].grid(alpha=0.25)
            axes_err[0].legend(fontsize=8, ncol=3)
            axes_err[1].set_ylabel("|1-F_sub|")
            axes_err[1].set_xlabel("Time")
            axes_err[1].grid(alpha=0.25)
            axes_err[1].legend(fontsize=8, ncol=3)
            axes_err[0].set_title(f"{profile_name}: Magnus/CFQM error-to-exact")
            _annotate_plot_with_equations(
                fig_err,
                eq_ids=["eq_error_abs", "eq_subspace_fidelity", "eq_suzuki2", "eq_magnus2", "eq_cfqm"],
                equation_registry=equation_registry,
                plot_id=f"plot_{profile_name}_magnus_cfqm_error",
                plot_contracts=plot_contracts,
                style_legend_lines=method_style_lines,
            )
            used_eq_ids.update(["eq_error_abs", "eq_subspace_fidelity", "eq_suzuki2", "eq_magnus2", "eq_cfqm"])
            fig_err.tight_layout()
            pdf.savefig(fig_err)
            plt.close(fig_err)

        render_text_page(
            pdf,
            [
                "SECTION: RESULTS NOISE DETAILS",
                "",
                "Overlay + per-mode detail pages for noise robustness.",
                "",
                "Legend key (applies to this section):",
                "  Exact: exact reference evolution.",
                "  Noiseless: ADAPT-HVA pipeline on noiseless simulator.",
                f"  Noisy: shot-based estimator under selected mode ({int(settings.get('shots', 0))} shots).",
                "  Ideal-reference: baseline used for Delta(noisy-ideal) panels.",
            ],
            fontsize=10,
        )

        # Mandatory overlay pages: exact + noiseless + all available noisy modes.
        for profile_name in ("static", "drive"):
            nprof = noisy.get(profile_name)
            pprof = noiseless.get(profile_name)
            if not isinstance(nprof, dict) or not isinstance(pprof, dict):
                continue
            base_traj = pprof.get("methods", {}).get("suzuki2", {}).get("trajectory", [])
            if not base_traj:
                continue
            times = np.asarray([float(r["time"]) for r in base_traj], dtype=float)
            mode_order = list(settings.get("noise_modes", ["ideal", "shots", "aer_noise"]))
            available_modes = [
                m for m in mode_order if bool(nprof.get("modes", {}).get(str(m), {}).get("success", False))
            ]
            unavailable_modes = [m for m in mode_order if m not in available_modes]

            # Energy overlay
            fig_eov, axes_eov = plt.subplots(2, 1, figsize=(8.5, 11.0), sharex=True)
            axes_eov[0].plot(
                times,
                np.asarray([float(r["energy_total_exact"]) for r in base_traj], dtype=float),
                color="#111111",
                linewidth=2.2,
                label="exact reference",
            )
            axes_eov[0].plot(
                times,
                np.asarray([float(r["energy_total_trotter"]) for r in base_traj], dtype=float),
                color="#1f77b4",
                linewidth=1.6,
                label="noiseless (final-seed Suzuki-2)",
            )
            axes_eov[1].plot(
                times,
                np.asarray([float(r["energy_static_trotter"]) for r in base_traj], dtype=float),
                color="#1f77b4",
                linewidth=1.6,
                label="noiseless (final-seed Suzuki-2)",
            )
            for mode in available_modes:
                color = {"ideal": "#ff7f0e", "shots": "#2ca02c", "aer_noise": "#d62728"}.get(str(mode), "#9467bd")
                traj = nprof.get("modes", {}).get(str(mode), {}).get("trajectory", [])
                if not traj:
                    continue
                axes_eov[0].plot(
                    times,
                    np.asarray([float(r["energy_total_noisy"]) for r in traj], dtype=float),
                    color=color,
                    linewidth=1.3,
                    label=f"noise mode={mode} measured",
                )
                axes_eov[1].plot(
                    times,
                    np.asarray([float(r["energy_static_noisy"]) for r in traj], dtype=float),
                    color=color,
                    linewidth=1.3,
                    label=f"noise mode={mode} measured",
                )
            axes_eov[0].set_title(f"{profile_name}: energy overlays")
            axes_eov[0].set_ylabel("E_total")
            axes_eov[1].set_ylabel("E_static")
            axes_eov[1].set_xlabel("time")
            axes_eov[0].grid(alpha=0.25)
            axes_eov[1].grid(alpha=0.25)
            axes_eov[0].legend(fontsize=8, ncol=2)
            axes_eov[1].legend(fontsize=8, ncol=2)
            _annotate_plot_with_equations(
                fig_eov,
                eq_ids=["eq_energy_total", "eq_energy_static", "eq_noisy_estimator"],
                equation_registry=equation_registry,
                plot_id=f"plot_{profile_name}_overlay_energy",
                plot_contracts=plot_contracts,
                style_legend_lines=noise_style_lines
                + [f"unavailable modes: {', '.join(unavailable_modes) if unavailable_modes else 'none'}"],
            )
            used_eq_ids.update(["eq_energy_total", "eq_energy_static", "eq_noisy_estimator"])
            fig_eov.tight_layout()
            pdf.savefig(fig_eov)
            plt.close(fig_eov)

            # Site-0 occupation overlay
            fig_oov, axes_oov = plt.subplots(2, 1, figsize=(8.5, 11.0), sharex=True)
            axes_oov[0].plot(
                times,
                np.asarray([float(r["n_up_site0_trotter_hva"]) for r in base_traj], dtype=float),
                color="#1f77b4",
                linewidth=1.5,
                label="noiseless (final-seed) up0",
            )
            axes_oov[1].plot(
                times,
                np.asarray([float(r["n_dn_site0_trotter_hva"]) for r in base_traj], dtype=float),
                color="#1f77b4",
                linewidth=1.5,
                label="noiseless (final-seed) dn0",
            )
            for mode in available_modes:
                color = {"ideal": "#ff7f0e", "shots": "#2ca02c", "aer_noise": "#d62728"}.get(str(mode), "#9467bd")
                traj = nprof.get("modes", {}).get(str(mode), {}).get("trajectory", [])
                if not traj:
                    continue
                axes_oov[0].plot(
                    times,
                    np.asarray([float(r["n_up_site0_noisy"]) for r in traj], dtype=float),
                    color=color,
                    linewidth=1.3,
                    label=f"{mode} measured up0",
                )
                axes_oov[1].plot(
                    times,
                    np.asarray([float(r["n_dn_site0_noisy"]) for r in traj], dtype=float),
                    color=color,
                    linewidth=1.3,
                    label=f"{mode} measured dn0",
                )
            axes_oov[0].set_title(f"{profile_name}: site-0 occupation overlays")
            axes_oov[0].set_ylabel("n_up(0)")
            axes_oov[1].set_ylabel("n_dn(0)")
            axes_oov[1].set_xlabel("time")
            axes_oov[0].grid(alpha=0.25)
            axes_oov[1].grid(alpha=0.25)
            axes_oov[0].legend(fontsize=8, ncol=2)
            axes_oov[1].legend(fontsize=8, ncol=2)
            _annotate_plot_with_equations(
                fig_oov,
                eq_ids=["eq_number_operator", "eq_total_density", "eq_noisy_estimator"],
                equation_registry=equation_registry,
                plot_id=f"plot_{profile_name}_overlay_occupancy",
                plot_contracts=plot_contracts,
                style_legend_lines=noise_style_lines
                + [f"unavailable modes: {', '.join(unavailable_modes) if unavailable_modes else 'none'}"],
            )
            used_eq_ids.update(["eq_number_operator", "eq_total_density", "eq_noisy_estimator"])
            fig_oov.tight_layout()
            pdf.savefig(fig_oov)
            plt.close(fig_oov)

            # Doublon / staggered overlay
            fig_dov, axes_dov = plt.subplots(2, 1, figsize=(8.5, 11.0), sharex=True)
            axes_dov[0].plot(
                times,
                np.asarray([float(r["doublon_trotter_hva"]) for r in base_traj], dtype=float),
                color="#1f77b4",
                linewidth=1.5,
                label="noiseless (final-seed) doublon",
            )
            axes_dov[1].plot(
                times,
                np.asarray([float(r["staggered_trotter_hva"]) for r in base_traj], dtype=float),
                color="#1f77b4",
                linewidth=1.5,
                label="noiseless (final-seed) staggered",
            )
            for mode in available_modes:
                color = {"ideal": "#ff7f0e", "shots": "#2ca02c", "aer_noise": "#d62728"}.get(str(mode), "#9467bd")
                traj = nprof.get("modes", {}).get(str(mode), {}).get("trajectory", [])
                if not traj:
                    continue
                axes_dov[0].plot(
                    times,
                    np.asarray([float(r["doublon_noisy"]) for r in traj], dtype=float),
                    color=color,
                    linewidth=1.3,
                    label=f"{mode} measured doublon",
                )
                axes_dov[1].plot(
                    times,
                    np.asarray([float(r["staggered_noisy"]) for r in traj], dtype=float),
                    color=color,
                    linewidth=1.3,
                    label=f"{mode} measured staggered",
                )
            axes_dov[0].set_title(f"{profile_name}: doublon/staggered overlays")
            axes_dov[0].set_ylabel("doublon")
            axes_dov[1].set_ylabel("staggered")
            axes_dov[1].set_xlabel("time")
            axes_dov[0].grid(alpha=0.25)
            axes_dov[1].grid(alpha=0.25)
            axes_dov[0].legend(fontsize=8, ncol=2)
            axes_dov[1].legend(fontsize=8, ncol=2)
            _annotate_plot_with_equations(
                fig_dov,
                eq_ids=["eq_doublon", "eq_staggered", "eq_noisy_estimator"],
                equation_registry=equation_registry,
                plot_id=f"plot_{profile_name}_overlay_doublon_staggered",
                plot_contracts=plot_contracts,
                style_legend_lines=noise_style_lines
                + [f"unavailable modes: {', '.join(unavailable_modes) if unavailable_modes else 'none'}"],
            )
            used_eq_ids.update(["eq_doublon", "eq_staggered", "eq_noisy_estimator"])
            fig_dov.tight_layout()
            pdf.savefig(fig_dov)
            plt.close(fig_dov)

        # Noisy vs noiseless Suzuki full per-mode pages
        for profile_name in ("static", "drive"):
            nprof = noisy.get(profile_name)
            pprof = noiseless.get(profile_name)
            if not isinstance(nprof, dict) or not isinstance(pprof, dict):
                continue
            base_traj = pprof.get("methods", {}).get("suzuki2", {}).get("trajectory", [])
            if not base_traj:
                continue

            times = np.asarray([float(r["time"]) for r in base_traj], dtype=float)
            e_base = np.asarray([float(r["energy_total_trotter"]) for r in base_traj], dtype=float)
            s_base = np.asarray([float(r["energy_static_trotter"]) for r in base_traj], dtype=float)
            d_base = np.asarray([float(r["doublon_trotter_hva"]) for r in base_traj], dtype=float)
            stg_base = np.asarray([float(r["staggered_trotter_hva"]) for r in base_traj], dtype=float)
            e_ref = np.asarray([float(r["energy_total_exact"]) for r in base_traj], dtype=float)
            up_base = np.asarray([float(r["n_up_site0_trotter_hva"]) for r in base_traj], dtype=float)
            dn_base = np.asarray([float(r["n_dn_site0_trotter_hva"]) for r in base_traj], dtype=float)

            mode_order = list(settings.get("noise_modes", ["ideal", "shots", "aer_noise"]))
            for mode in mode_order:
                color = {"ideal": "#ff7f0e", "shots": "#2ca02c", "aer_noise": "#d62728"}.get(str(mode), "#9467bd")
                mdata = nprof.get("modes", {}).get(str(mode), {})
                if not bool(mdata.get("success", False)):
                    render_text_page(
                        pdf,
                        [
                            f"SECTION: NOISE MODE UNAVAILABLE ({profile_name}/{mode})",
                            "",
                            f"Reason: {mdata.get('reason', mdata.get('error', 'unknown'))}",
                            f"Diagnostics: {json.dumps(mdata, indent=2)}",
                        ],
                        fontsize=9,
                    )
                    continue
                traj = mdata.get("trajectory", [])
                if not traj:
                    continue
                e_tot_n = np.asarray([float(r["energy_total_noisy"]) for r in traj], dtype=float)
                e_tot_i = np.asarray([float(r["energy_total_ideal"]) for r in traj], dtype=float)
                e_sta_n = np.asarray([float(r["energy_static_noisy"]) for r in traj], dtype=float)
                e_sta_i = np.asarray([float(r["energy_static_ideal"]) for r in traj], dtype=float)
                e_tot_d = np.asarray([float(r["energy_total_delta_noisy_minus_ideal"]) for r in traj], dtype=float)
                e_sta_d = np.asarray([float(r["energy_static_delta_noisy_minus_ideal"]) for r in traj], dtype=float)
                up_n = np.asarray([float(r["n_up_site0_noisy"]) for r in traj], dtype=float)
                up_i = np.asarray([float(r["n_up_site0_ideal"]) for r in traj], dtype=float)
                up_d = np.asarray([float(r["n_up_site0_delta_noisy_minus_ideal"]) for r in traj], dtype=float)
                dn_n = np.asarray([float(r["n_dn_site0_noisy"]) for r in traj], dtype=float)
                dn_i = np.asarray([float(r["n_dn_site0_ideal"]) for r in traj], dtype=float)
                dn_d = np.asarray([float(r["n_dn_site0_delta_noisy_minus_ideal"]) for r in traj], dtype=float)
                dbl_n = np.asarray([float(r["doublon_noisy"]) for r in traj], dtype=float)
                dbl_i = np.asarray([float(r["doublon_ideal"]) for r in traj], dtype=float)
                dbl_d = np.asarray([float(r["doublon_delta_noisy_minus_ideal"]) for r in traj], dtype=float)
                stg_n = np.asarray([float(r["staggered_noisy"]) for r in traj], dtype=float)
                stg_i = np.asarray([float(r["staggered_ideal"]) for r in traj], dtype=float)
                stg_d = np.asarray([float(r["staggered_delta_noisy_minus_ideal"]) for r in traj], dtype=float)
                delta_stats = (
                    f"max|ΔE_total|={float(np.max(np.abs(e_tot_d))):.3e}; "
                    f"RMS|ΔE_total|={float(np.sqrt(np.mean(e_tot_d ** 2))):.3e}; "
                    f"max|ΔD|={float(np.max(np.abs(dbl_d))):.3e}"
                )

                # Energy page
                fig_e, axes_e = plt.subplots(3, 1, figsize=(10.8, 9.0), sharex=True)
                fig_e.subplots_adjust(hspace=0.28)
                axes_e[0].plot(times, e_ref, color="#111111", linewidth=1.8, label="exact total")
                axes_e[0].plot(times, e_base, color="#1f77b4", linewidth=1.4, label="noiseless (final-seed Suzuki-2) total")
                axes_e[0].plot(times, e_tot_n, color=color, linewidth=1.2, label=f"noisy measured (mode={mode}) total")
                axes_e[0].plot(
                    times,
                    e_tot_i,
                    color=color,
                    linewidth=1.0,
                    linestyle="--",
                    label=f"ideal reference for mode={mode} total",
                )
                axes_e[0].set_ylabel("E_total")
                axes_e[0].grid(alpha=0.25)
                axes_e[0].legend(fontsize=7, ncol=2)
                axes_e[1].plot(times, s_base, color="#1f77b4", linewidth=1.4, label="noiseless (final-seed Suzuki-2) static")
                axes_e[1].plot(times, e_sta_n, color=color, linewidth=1.2, label=f"noisy measured (mode={mode}) static")
                axes_e[1].plot(
                    times,
                    e_sta_i,
                    color=color,
                    linewidth=1.0,
                    linestyle="--",
                    label=f"ideal reference for mode={mode} static",
                )
                axes_e[1].set_ylabel("E_static")
                axes_e[1].grid(alpha=0.25)
                axes_e[1].legend(fontsize=7, ncol=2)
                axes_e[2].plot(times, e_tot_d, color=color, linewidth=1.2, label="Δ(noisy-ideal) total")
                axes_e[2].plot(times, e_sta_d, color=color, linewidth=1.2, linestyle="--", label="Δ(noisy-ideal) static")
                axes_e[2].axhline(0.0, color="#111111", linewidth=0.8, linestyle=":", label="zero baseline")
                axes_e[2].set_ylabel("Δ(noisy-ideal) E")
                axes_e[2].set_xlabel("time")
                axes_e[2].grid(alpha=0.25)
                axes_e[2].legend(fontsize=7, ncol=2)
                fig_e.suptitle(f"{profile_name}/{mode}: noisy energy audit")
                _annotate_plot_with_equations(
                    fig_e,
                    eq_ids=["eq_energy_static", "eq_energy_total", "eq_noisy_estimator", "eq_noisy_delta"],
                    equation_registry=equation_registry,
                    plot_id=f"plot_{profile_name}_{mode}_energy",
                    plot_contracts=plot_contracts,
                    style_legend_lines=noise_style_lines + [delta_stats],
                )
                used_eq_ids.update(["eq_energy_static", "eq_energy_total", "eq_noisy_estimator", "eq_noisy_delta"])
                fig_e.tight_layout()
                pdf.savefig(fig_e)
                plt.close(fig_e)

                # Occupation page
                fig_o, axes_o = plt.subplots(3, 1, figsize=(10.8, 9.0), sharex=True)
                fig_o.subplots_adjust(hspace=0.28)
                axes_o[0].plot(times, up_base, color="#1f77b4", linewidth=1.3, label="noiseless (final-seed) up0")
                axes_o[0].plot(times, up_n, color=color, linewidth=1.2, label=f"noisy measured (mode={mode}) up0")
                axes_o[0].plot(
                    times,
                    up_i,
                    color=color,
                    linewidth=1.0,
                    linestyle="--",
                    label=f"ideal reference for mode={mode} up0",
                )
                axes_o[0].set_ylabel("n_up(0)")
                axes_o[0].grid(alpha=0.25)
                axes_o[0].legend(fontsize=7, ncol=2)
                axes_o[1].plot(times, dn_base, color="#1f77b4", linewidth=1.3, label="noiseless (final-seed) dn0")
                axes_o[1].plot(times, dn_n, color=color, linewidth=1.2, label=f"noisy measured (mode={mode}) dn0")
                axes_o[1].plot(
                    times,
                    dn_i,
                    color=color,
                    linewidth=1.0,
                    linestyle="--",
                    label=f"ideal reference for mode={mode} dn0",
                )
                axes_o[1].set_ylabel("n_dn(0)")
                axes_o[1].grid(alpha=0.25)
                axes_o[1].legend(fontsize=7, ncol=2)
                axes_o[2].plot(times, up_d, color=color, linewidth=1.2, label="Δ(noisy-ideal) up0")
                axes_o[2].plot(times, dn_d, color=color, linewidth=1.2, linestyle="--", label="Δ(noisy-ideal) dn0")
                axes_o[2].axhline(0.0, color="#111111", linewidth=0.8, linestyle=":", label="zero baseline")
                axes_o[2].set_ylabel("Δ(noisy-ideal) n(0)")
                axes_o[2].set_xlabel("time")
                axes_o[2].grid(alpha=0.25)
                axes_o[2].legend(fontsize=7, ncol=2)
                fig_o.suptitle(f"{profile_name}/{mode}: site-0 occupation audit")
                _annotate_plot_with_equations(
                    fig_o,
                    eq_ids=["eq_number_operator", "eq_total_density", "eq_noisy_estimator", "eq_noisy_delta"],
                    equation_registry=equation_registry,
                    plot_id=f"plot_{profile_name}_{mode}_occupancy",
                    plot_contracts=plot_contracts,
                    style_legend_lines=noise_style_lines + [delta_stats],
                )
                used_eq_ids.update(["eq_number_operator", "eq_total_density", "eq_noisy_estimator", "eq_noisy_delta"])
                fig_o.tight_layout()
                pdf.savefig(fig_o)
                plt.close(fig_o)

                # Doublon/staggered page
                fig_d, axes_d = plt.subplots(3, 1, figsize=(10.8, 9.0), sharex=True)
                fig_d.subplots_adjust(hspace=0.28)
                axes_d[0].plot(times, d_base, color="#1f77b4", linewidth=1.3, label="noiseless (final-seed) doublon")
                axes_d[0].plot(times, dbl_n, color=color, linewidth=1.2, label=f"noisy measured (mode={mode}) doublon")
                axes_d[0].plot(
                    times,
                    dbl_i,
                    color=color,
                    linewidth=1.0,
                    linestyle="--",
                    label=f"ideal reference for mode={mode} doublon",
                )
                axes_d[0].set_ylabel("doublon")
                axes_d[0].grid(alpha=0.25)
                axes_d[0].legend(fontsize=7, ncol=2)
                axes_d[1].plot(times, stg_base, color="#1f77b4", linewidth=1.3, label="noiseless (final-seed) staggered")
                axes_d[1].plot(times, stg_n, color=color, linewidth=1.2, label=f"noisy measured (mode={mode}) staggered")
                axes_d[1].plot(
                    times,
                    stg_i,
                    color=color,
                    linewidth=1.0,
                    linestyle="--",
                    label=f"ideal reference for mode={mode} staggered",
                )
                axes_d[1].set_ylabel("staggered")
                axes_d[1].grid(alpha=0.25)
                axes_d[1].legend(fontsize=7, ncol=2)
                axes_d[2].plot(times, dbl_d, color=color, linewidth=1.2, label="Δ(noisy-ideal) doublon")
                axes_d[2].plot(times, stg_d, color=color, linewidth=1.2, linestyle="--", label="Δ(noisy-ideal) staggered")
                axes_d[2].axhline(0.0, color="#111111", linewidth=0.8, linestyle=":", label="zero baseline")
                axes_d[2].set_ylabel("Δ(noisy-ideal)")
                axes_d[2].set_xlabel("time")
                axes_d[2].grid(alpha=0.25)
                axes_d[2].legend(fontsize=7, ncol=2)
                fig_d.suptitle(f"{profile_name}/{mode}: doublon & staggered audit")
                _annotate_plot_with_equations(
                    fig_d,
                    eq_ids=["eq_doublon", "eq_staggered", "eq_noisy_estimator", "eq_noisy_delta"],
                    equation_registry=equation_registry,
                    plot_id=f"plot_{profile_name}_{mode}_doublon_staggered",
                    plot_contracts=plot_contracts,
                    style_legend_lines=noise_style_lines + [delta_stats],
                )
                used_eq_ids.update(["eq_doublon", "eq_staggered", "eq_noisy_estimator", "eq_noisy_delta"])
                fig_d.tight_layout()
                pdf.savefig(fig_d)
                plt.close(fig_d)

        # Drive waveform diagnostics
        drive_cfg = settings.get("drive_profile", {})
        if isinstance(drive_cfg, dict) and bool(drive_cfg.get("enabled", False)):
            times = np.linspace(0.0, float(settings.get("t_final", 0.0)), int(settings.get("num_times", 1)))
            waveform = evaluate_drive_waveform(times, drive_cfg, float(drive_cfg.get("A", 0.0)))
            fig, ax = plt.subplots(figsize=(10.5, 3.8))
            fig.subplots_adjust(right=0.68)
            ax.plot(times, waveform, color="#111111", linewidth=1.6)
            ax.set_title("Drive waveform A*sin(omega t + phi)*exp(-(t+t0)^2/(2 tbar^2))")
            ax.set_xlabel("Time")
            ax.set_ylabel("drive scalar")
            ax.grid(alpha=0.25)
            _annotate_plot_with_equations(
                fig,
                eq_ids=["eq_drive_waveform", "eq_h_drive_density"],
                equation_registry=equation_registry,
                plot_id="plot_drive_waveform",
                plot_contracts={"plot_drive_waveform": {"x": "time", "y": ["drive scalar"], "source": ["settings.drive_profile"], "notes": "Drive scalar evaluation grid."}},
                style_legend_lines=method_style_lines,
            )
            used_eq_ids.update(["eq_drive_waveform", "eq_h_drive_density"])
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        benchmark_rows = payload.get("dynamics_benchmarks", {}).get("rows", [])
        if isinstance(benchmark_rows, list) and benchmark_rows:
            fig = plt.figure(figsize=(12.0, 7.0))
            ax = fig.add_subplot(111)
            headers = ["profile", "method", "mode", "cx", "term_exp", "sq", "depth", "wall_s", "oracle_s"]
            rows: list[list[str]] = []
            for rec in benchmark_rows:
                if not isinstance(rec, dict):
                    continue
                rows.append(
                    [
                        str(rec.get("profile", "")),
                        str(rec.get("method", "")),
                        str(rec.get("mode", "")),
                        f"{int(rec.get('cx_proxy_total', 0))}",
                        f"{int(rec.get('term_exp_count_total', 0))}",
                        f"{int(rec.get('sq_proxy_total', 0))}",
                        f"{int(rec.get('depth_proxy_total', 0))}",
                        f"{float(rec.get('wall_total_s', float('nan'))):.3f}",
                        f"{float(rec.get('oracle_eval_s_total', float('nan'))):.3f}",
                    ]
                )
            if rows:
                render_compact_table(
                    ax,
                    title="Noisy dynamics benchmark (Trotter vs CFQM under noise)",
                    col_labels=headers,
                    rows=rows,
                    fontsize=7,
                )
                _annotate_plot_with_equations(
                    fig,
                    eq_ids=["eq_proxy_cost", "eq_runtime_bench"],
                    equation_registry=equation_registry,
                    plot_id="plot_noisy_benchmark_table",
                    plot_contracts=plot_contracts,
                    style_legend_lines=method_style_lines,
                )
                used_eq_ids.update(["eq_proxy_cost", "eq_runtime_bench"])
                fig.tight_layout()
                pdf.savefig(fig)
            plt.close(fig)

        diag_lines = [
            "Diagnostics and availability",
            "",
            f"Comparisons keys: {list(comparisons.keys())}",
            f"Noisy mode diagnostics: {json.dumps(diagnostics.get('noisy_mode_diagnostics', {}), indent=2)}",
            f"Noisy dynamics benchmark rows: {len(payload.get('dynamics_benchmarks', {}).get('rows', []))}",
            "",
            "Metric formulas:",
            "  energy_static(t) = <psi(t)|H_static|psi(t)>",
            "  energy_total(t)  = <psi(t)|H_static + H_drive(t)|psi(t)>",
            "  n_up_site0 = <n_{0,up}> , n_dn_site0 = <n_{0,dn}>",
            "  doublon(site0) = <n_{0,up} n_{0,dn}>",
            "  staggered = (1/L) * Sum_j (-1)^j <n_j>",
            "",
            f"equation_registry_count: {len(equation_registry)}",
            f"plot_contract_count: {len(plot_contracts)}",
        ]
        render_text_page(pdf, diag_lines, fontsize=9)

        render_text_page(
            pdf,
            [
                "SECTION: APPENDIX DEFINITIONS USED",
                "",
                "Short equation definitions used in result pages.",
            ],
            fontsize=10,
        )
        used_keys = sorted(used_eq_ids)
        if len(used_keys) < 8:
            used_keys = sorted(list(equation_registry.keys()))[:24]
        _render_formula_atlas(
            pdf,
            equation_registry=equation_registry,
            title="SECTION: APPENDIX DEFINITIONS USED",
            keys=used_keys,
            include_sources=False,
        )

        _render_formula_atlas(
            pdf,
            equation_registry=equation_registry,
            title="SECTION: APPENDIX FORMULA ATLAS",
            keys=None,
            include_sources=True,
        )

        render_text_page(
            pdf,
            [
                "SECTION: APPENDIX PLOT CONTRACTS",
                "",
                "Full machine-readable plot contract map.",
            ],
            fontsize=10,
        )
        contract_lines: list[str] = ["SECTION: APPENDIX PLOT CONTRACTS", ""]
        for idx, key in enumerate(sorted(plot_contracts.keys())):
            rec = plot_contracts.get(key, {})
            contract_lines.append(f"[{key}]")
            contract_lines.append(f"  x: {rec.get('x', '')}")
            contract_lines.append(f"  y: {rec.get('y', '')}")
            contract_lines.append(f"  source: {rec.get('source', '')}")
            contract_lines.append(f"  notes: {rec.get('notes', '')}")
            contract_lines.append("")
            if (idx + 1) % 8 == 0:
                render_text_page(pdf, contract_lines, fontsize=8, line_spacing=0.024, max_line_width=140)
                contract_lines = [f"SECTION: APPENDIX PLOT CONTRACTS (cont. {idx + 1})", ""]
        if contract_lines:
            render_text_page(pdf, contract_lines, fontsize=8, line_spacing=0.024, max_line_width=140)

        render_command_page(
            pdf,
            str(payload.get("run_command", "")),
            script_name="pipelines/exact_bench/hh_noise_robustness_seq_report.py",
        )


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def _enforce_defaults_and_minimums(args: argparse.Namespace) -> argparse.Namespace:
    key = (int(args.L), int(args.n_ph_max))
    minimum = dict(_HH_MINIMUMS.get(key, _HH_MINIMUMS[(2, 1)]))

    if args.warm_reps is None:
        args.warm_reps = int(minimum["reps"])
    if args.warm_restarts is None:
        args.warm_restarts = int(minimum["restarts"])
    if args.warm_maxiter is None:
        args.warm_maxiter = int(minimum["maxiter"])

    if args.final_reps is None:
        args.final_reps = int(minimum["reps"])
    if args.final_restarts is None:
        args.final_restarts = int(minimum["restarts"])
    if args.final_maxiter is None:
        args.final_maxiter = int(minimum["maxiter"])

    if args.trotter_steps is None:
        args.trotter_steps = int(minimum["trotter_steps"])

    if args.warm_method is None:
        args.warm_method = str(minimum["method"])
    if args.final_method is None:
        args.final_method = str(minimum["method"])

    if args.num_times is None:
        args.num_times = 101
    if args.t_final is None:
        args.t_final = 10.0

    if not bool(args.smoke_test_intentionally_weak):
        checks = {
            "trotter_steps": int(args.trotter_steps) >= int(minimum["trotter_steps"]),
            "warm_reps": int(args.warm_reps) >= int(minimum["reps"]),
            "warm_restarts": int(args.warm_restarts) >= int(minimum["restarts"]),
            "warm_maxiter": int(args.warm_maxiter) >= int(minimum["maxiter"]),
            "final_reps": int(args.final_reps) >= int(minimum["reps"]),
            "final_restarts": int(args.final_restarts) >= int(minimum["restarts"]),
            "final_maxiter": int(args.final_maxiter) >= int(minimum["maxiter"]),
        }
        failed = [k for k, ok in checks.items() if not bool(ok)]
        if failed:
            raise ValueError(
                "Under-parameterized report run rejected by AGENTS minimum table. "
                f"Failed fields: {failed}. Minimums: {minimum}. "
                "Pass --smoke-test-intentionally-weak only for explicit smoke tests."
            )

    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "HH noise robustness comprehensive report: warm-start -> ADAPT PoolB -> conventional VQE, "
            "with noiseless Magnus/Trotter matrix and noisy method matrix."
        )
    )

    p.add_argument("--L", type=int, default=2)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--u", type=float, default=2.0)
    p.add_argument("--dv", type=float, default=0.0)
    p.add_argument("--omega0", type=float, default=1.0)
    p.add_argument("--g-ep", type=float, default=1.0)
    p.add_argument("--n-ph-max", type=int, default=1)
    p.add_argument("--boson-encoding", choices=["binary"], default="binary")
    p.add_argument("--ordering", choices=["blocked", "interleaved"], default="blocked")
    p.add_argument("--boundary", choices=["periodic", "open"], default="periodic")

    p.add_argument("--warm-reps", type=int, default=None)
    p.add_argument("--warm-restarts", type=int, default=None)
    p.add_argument("--warm-maxiter", type=int, default=None)
    p.add_argument("--warm-method", choices=["COBYLA", "SLSQP", "L-BFGS-B", "Powell", "Nelder-Mead"], default=None)
    p.add_argument("--warm-seed", type=int, default=7)

    p.add_argument("--window-k", type=int, default=5)
    p.add_argument("--slope-epsilon", type=float, default=5e-5)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min-points-before-switch", type=int, default=8)

    p.set_defaults(adapt_allow_repeats=False)
    p.add_argument("--adapt-allow-repeats", dest="adapt_allow_repeats", action="store_true")
    p.add_argument("--adapt-no-repeats", dest="adapt_allow_repeats", action="store_false")
    p.add_argument("--adapt-max-depth", type=int, default=40)
    p.add_argument("--adapt-maxiter", type=int, default=600)
    p.add_argument("--adapt-eps-grad", type=float, default=1e-6)
    p.add_argument("--adapt-eps-energy", type=float, default=1e-8)
    p.add_argument("--adapt-seed", type=int, default=11)

    p.add_argument("--paop-r", type=int, default=1)
    p.add_argument("--paop-split-paulis", action="store_true")
    p.add_argument("--paop-prune-eps", type=float, default=0.0)
    p.add_argument("--paop-normalization", choices=["none", "fro", "maxcoeff"], default="none")

    p.add_argument("--final-reps", type=int, default=None)
    p.add_argument("--final-restarts", type=int, default=None)
    p.add_argument("--final-maxiter", type=int, default=None)
    p.add_argument("--final-method", choices=["COBYLA", "SLSQP", "L-BFGS-B", "Powell", "Nelder-Mead"], default=None)
    p.add_argument("--final-seed", type=int, default=19)

    p.add_argument("--t-final", type=float, default=None)
    p.add_argument("--num-times", type=int, default=None)
    p.add_argument("--trotter-steps", type=int, default=None)
    p.add_argument("--exact-steps-multiplier", type=int, default=2)
    p.add_argument("--fidelity-subspace-energy-tol", type=float, default=1e-9)
    p.add_argument("--cfqm-stage-exp", choices=["expm_multiply_sparse", "dense_expm", "pauli_suzuki2"], default="expm_multiply_sparse")
    p.add_argument("--cfqm-coeff-drop-abs-tol", type=float, default=0.0)
    p.add_argument("--cfqm-normalize", action="store_true")

    p.add_argument("--include-drive-profile", action="store_true", default=True)
    p.add_argument("--drive-A", type=float, default=0.6)
    p.add_argument("--drive-omega", type=float, default=2.0)
    p.add_argument("--drive-tbar", type=float, default=2.5)
    p.add_argument("--drive-phi", type=float, default=0.0)
    p.add_argument("--drive-pattern", choices=["staggered", "dimer_bias", "custom"], default="staggered")
    p.add_argument("--drive-custom-s", type=str, default=None)
    p.add_argument("--drive-include-identity", action="store_true")
    p.add_argument("--drive-time-sampling", choices=["midpoint", "left", "right"], default="midpoint")
    p.add_argument("--drive-t0", type=float, default=0.0)

    p.add_argument("--noise-modes", type=str, default="ideal,shots,aer_noise")
    p.add_argument("--noisy-methods", type=str, default="cfqm4,suzuki2")
    p.add_argument("--benchmark-active-coeff-tol", type=float, default=1e-12)
    p.add_argument("--shots", type=int, default=2048)
    p.add_argument("--oracle-repeats", type=int, default=4)
    p.add_argument("--oracle-aggregate", choices=["mean", "median"], default="mean")
    p.add_argument("--noise-seed", type=int, default=7)
    p.add_argument("--backend-name", type=str, default="FakeManilaV2")
    p.add_argument("--use-fake-backend", action="store_true")
    p.set_defaults(allow_aer_fallback=True)
    p.add_argument("--allow-aer-fallback", dest="allow_aer_fallback", action="store_true")
    p.add_argument("--no-allow-aer-fallback", dest="allow_aer_fallback", action="store_false")
    p.set_defaults(omp_shm_workaround=True)
    p.add_argument("--omp-shm-workaround", dest="omp_shm_workaround", action="store_true")
    p.add_argument("--no-omp-shm-workaround", dest="omp_shm_workaround", action="store_false")
    p.add_argument("--noisy-mode-timeout-s", type=int, default=1200)

    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-pdf", type=Path, default=None)
    p.add_argument("--skip-pdf", action="store_true")
    p.add_argument("--retain-stage-events", action="store_true")
    p.add_argument("--require-at-least-one-noisy", action="store_true", default=True)
    p.add_argument("--smoke-test-intentionally-weak", action="store_true", help="# SMOKE TEST - intentionally weak settings")

    return p.parse_args(argv)


def _validate_pool_b_strict_composition(pool_b_meta: dict[str, Any]) -> dict[str, Any]:
    required = {"uccsd", "hva", "paop_full"}
    raw = pool_b_meta.get("raw_sizes", {})
    if not isinstance(raw, dict):
        raise ValueError("Pool B metadata missing raw_sizes.")
    if set(raw.keys()) != required:
        raise ValueError(
            "Pool B composition mismatch: expected exactly raw_sizes keys "
            f"{sorted(required)}, got {sorted(raw.keys())}."
        )
    missing = [fam for fam in sorted(required) if int(raw.get(fam, 0)) <= 0]
    if missing:
        raise ValueError(
            "Pool B composition invalid: each family must be non-empty. "
            f"Missing/empty={missing}."
        )
    dedup_presence = pool_b_meta.get("dedup_source_presence_counts", {})
    if not isinstance(dedup_presence, dict):
        raise ValueError("Pool B metadata missing dedup_source_presence_counts.")
    if set(dedup_presence.keys()) != required:
        raise ValueError(
            "Pool B dedup presence mismatch: expected keys "
            f"{sorted(required)}, got {sorted(dedup_presence.keys())}."
        )
    return {
        "required_families": ["uccsd_lifted", "hva", "paop_full"],
        "raw_sizes": {k: int(raw[k]) for k in sorted(raw.keys())},
        "dedup_source_presence_counts": {
            k: int(dedup_presence[k]) for k in sorted(dedup_presence.keys())
        },
        "passed": True,
    }


def _collect_noisy_benchmark_rows(dynamics_noisy: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    profiles = dynamics_noisy.get("profiles", {}) if isinstance(dynamics_noisy, dict) else {}
    for profile_name, profile_data in profiles.items():
        methods = profile_data.get("methods", {}) if isinstance(profile_data, dict) else {}
        if not isinstance(methods, dict):
            continue
        for method_name, method_data in methods.items():
            modes = method_data.get("modes", {}) if isinstance(method_data, dict) else {}
            if not isinstance(modes, dict):
                continue
            for mode_name, mode_data in modes.items():
                if not bool(isinstance(mode_data, dict) and mode_data.get("success", False)):
                    continue
                cost = mode_data.get("benchmark_cost", {})
                runtime = mode_data.get("benchmark_runtime", {})
                rows.append(
                    {
                        "profile": str(profile_name),
                        "method": str(method_name),
                        "mode": str(mode_name),
                        "term_exp_count_total": int(cost.get("term_exp_count_total", 0)),
                        "pauli_rot_count_total": int(cost.get("pauli_rot_count_total", 0)),
                        "cx_proxy_total": int(cost.get("cx_proxy_total", 0)),
                        "sq_proxy_total": int(cost.get("sq_proxy_total", 0)),
                        "depth_proxy_total": int(cost.get("depth_proxy_total", 0)),
                        "wall_total_s": float(runtime.get("wall_total_s", float("nan"))),
                        "oracle_eval_s_total": float(runtime.get("oracle_eval_s_total", float("nan"))),
                        "oracle_calls_total": int(runtime.get("oracle_calls_total", 0)),
                    }
                )
    return rows


def main(argv: list[str] | None = None) -> None:
    args = _enforce_defaults_and_minimums(parse_args(argv))
    noisy_methods = _parse_noisy_methods_csv(str(args.noisy_methods))

    if int(args.L) != 2 and not bool(args.smoke_test_intentionally_weak):
        raise ValueError("This report is currently locked to L=2 for the full static+drive noise matrix.")

    transition_cfg = TransitionConfig(
        window_k=int(args.window_k),
        slope_epsilon=float(args.slope_epsilon),
        patience=int(args.patience),
        min_points_before_switch=int(args.min_points_before_switch),
    )

    artifacts_json_dir = REPO_ROOT / "artifacts" / "json"
    docs_dir = REPO_ROOT / "docs"
    artifacts_json_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    tag = (
        f"L{int(args.L)}_hh_u{float(args.u):g}_g{float(args.g_ep):g}_"
        f"S{int(args.trotter_steps)}_N{int(args.num_times)}"
    ).replace(".", "p")

    output_json = args.output_json or (artifacts_json_dir / f"hh_noise_robustness_L2_{tag}.json")
    output_pdf = args.output_pdf or (docs_dir / "HH noise robustness report.PDF")

    _ai_log("hh_noise_robustness_start", settings=vars(args))

    num_particles = _half_filled_particles(int(args.L))
    h_poly = _build_hh_hamiltonian(args)
    hmat = np.asarray(hamiltonian_matrix(h_poly), dtype=complex)
    native_order, coeff_map_exyz = _collect_hardcoded_terms_exyz(h_poly)
    ordered_labels_exyz = sorted(native_order)

    exact_sector_energy = float(
        exact_ground_energy_sector_hh(
            h_poly,
            num_sites=int(args.L),
            num_particles=(int(num_particles[0]), int(num_particles[1])),
            n_ph_max=int(args.n_ph_max),
            boson_encoding=str(args.boson_encoding),
            indexing=str(args.ordering),
        )
    )
    exact_basis_energy, psi_exact_ref, fidelity_basis_v0 = _compute_exact_reference_for_hh(
        hmat=np.asarray(hmat, dtype=complex),
        num_sites=int(args.L),
        ordering=str(args.ordering),
        num_particles=(int(num_particles[0]), int(num_particles[1])),
        energy_tol=float(args.fidelity_subspace_energy_tol),
    )
    exact_sector_energy = float(exact_basis_energy)

    psi_ref = _build_reference_state(args, num_particles)

    # Stage 1: warm-start HVA
    warm_ansatz = _build_hh_hva_ansatz(args, reps=int(args.warm_reps))
    warm_payload, psi_warm = _run_vqe_stage_with_transition(
        stage_name="hva_warm_start",
        h_poly=h_poly,
        ansatz=warm_ansatz,
        psi_ref=psi_ref,
        exact_energy=float(exact_sector_energy),
        restarts=int(args.warm_restarts),
        seed=int(args.warm_seed),
        maxiter=int(args.warm_maxiter),
        method=str(args.warm_method),
        transition_cfg=transition_cfg,
    )
    _ai_log(
        "hh_noise_robustness_stage_done",
        stage="warm_start",
        energy=float(warm_payload.get("energy", float("nan"))),
        delta_abs=float(warm_payload.get("delta_abs", float("nan"))),
        stop_reason=str(warm_payload.get("stop_reason", "")),
    )

    # Stage 2: ADAPT with strict Pool B union
    uccsd_lifted = _build_uccsd_fermion_lifted_pool(
        num_sites=int(args.L),
        num_particles=(int(num_particles[0]), int(num_particles[1])),
        n_ph_max=int(args.n_ph_max),
        boson_encoding=str(args.boson_encoding),
        ordering=str(args.ordering),
    )
    hva_pool = _build_hva_pool(args)
    paop_full_pool = _build_paop_full_pool(args=args, num_particles=num_particles)

    pool_B, pool_B_meta, pool_B_source_by_sig = build_pool_b_strict_union(
        uccsd_ops=uccsd_lifted,
        hva_ops=hva_pool,
        paop_full_ops=paop_full_pool,
    )
    pool_b_audit = _validate_pool_b_strict_composition(pool_B_meta)

    adapt_payload, psi_adapt = _run_adapt_stage_with_transition(
        h_poly=h_poly,
        psi_start=np.asarray(psi_warm, dtype=complex),
        pool=pool_B,
        exact_energy=float(exact_sector_energy),
        allow_repeats=bool(args.adapt_allow_repeats),
        max_depth=int(args.adapt_max_depth),
        maxiter=int(args.adapt_maxiter),
        eps_grad=float(args.adapt_eps_grad),
        eps_energy=float(args.adapt_eps_energy),
        seed=int(args.adapt_seed),
        transition_cfg=transition_cfg,
    )
    _ai_log(
        "hh_noise_robustness_stage_done",
        stage="adapt_pool_b",
        energy=float(adapt_payload.get("energy", float("nan"))),
        delta_abs=float(adapt_payload.get("delta_abs", float("nan"))),
        stop_reason=str(adapt_payload.get("stop_reason", "")),
    )

    # Stage 3: conventional VQE seeded from ADAPT state
    final_ansatz = _build_hh_hva_ansatz(args, reps=int(args.final_reps))
    final_payload, psi_final = _run_vqe_stage_with_transition(
        stage_name="conventional_vqe_seeded_from_adapt",
        h_poly=h_poly,
        ansatz=final_ansatz,
        psi_ref=np.asarray(psi_adapt, dtype=complex),
        exact_energy=float(exact_sector_energy),
        restarts=int(args.final_restarts),
        seed=int(args.final_seed),
        maxiter=int(args.final_maxiter),
        method=str(args.final_method),
        transition_cfg=transition_cfg,
    )
    _ai_log(
        "hh_noise_robustness_stage_done",
        stage="conventional_vqe",
        energy=float(final_payload.get("energy", float("nan"))),
        delta_abs=float(final_payload.get("delta_abs", float("nan"))),
        stop_reason=str(final_payload.get("stop_reason", "")),
    )

    if not bool(args.retain_stage_events):
        warm_payload.pop("progress_events", None)

    # Noiseless dynamics matrix (static + drive)
    static_profile = None
    drive_profile = _build_drive_profile(args, enabled=bool(args.include_drive_profile))

    dynamics_noiseless = {
        "profiles": {
            "static": _run_noiseless_profile(
                args=args,
                psi_seed=np.asarray(psi_final, dtype=complex),
                hmat=hmat,
                ordered_labels_exyz=ordered_labels_exyz,
                coeff_map_exyz=coeff_map_exyz,
                drive_profile=static_profile,
            )
        }
    }
    if drive_profile is not None:
        dynamics_noiseless["profiles"]["drive"] = _run_noiseless_profile(
            args=args,
            psi_seed=np.asarray(psi_final, dtype=complex),
            hmat=hmat,
            ordered_labels_exyz=ordered_labels_exyz,
            coeff_map_exyz=coeff_map_exyz,
            drive_profile=drive_profile,
        )

    hardcoded_superset: dict[str, Any] = _disabled_hardcoded_superset_meta()

    noise_modes = [x.strip() for x in str(args.noise_modes).split(",") if x.strip()]
    dynamics_noisy: dict[str, Any] = {"profiles": {}}
    noisy_mode_diagnostics: dict[str, Any] = {}

    for profile_name, profile_cfg in [("static", static_profile), ("drive", drive_profile)]:
        if profile_name == "drive" and profile_cfg is None:
            continue

        method_payloads: dict[str, Any] = {}
        for method in noisy_methods:
            profile_modes: dict[str, Any] = {}
            for mode in noise_modes:
                kwargs = {
                    "L": int(args.L),
                    "ordering": str(args.ordering),
                    "psi_seed": np.asarray(psi_final, dtype=complex),
                    "ordered_labels_exyz": list(ordered_labels_exyz),
                    "static_coeff_map_exyz": dict(coeff_map_exyz),
                    "t_final": float(args.t_final),
                    "num_times": int(args.num_times),
                    "trotter_steps": int(args.trotter_steps),
                    "drive_profile": profile_cfg,
                    "noise_mode": str(mode),
                    "shots": int(args.shots),
                    "seed": int(args.noise_seed),
                    "oracle_repeats": int(args.oracle_repeats),
                    "oracle_aggregate": str(args.oracle_aggregate),
                    "backend_name": (None if args.backend_name is None else str(args.backend_name)),
                    "use_fake_backend": bool(args.use_fake_backend),
                    "allow_aer_fallback": bool(args.allow_aer_fallback),
                    "omp_shm_workaround": bool(args.omp_shm_workaround),
                    "method": str(method),
                    "benchmark_active_coeff_tol": float(args.benchmark_active_coeff_tol),
                    "cfqm_coeff_drop_abs_tol": float(args.cfqm_coeff_drop_abs_tol),
                }
                mode_result = _run_noisy_mode_isolated(
                    kwargs=kwargs,
                    timeout_s=int(args.noisy_mode_timeout_s),
                )
                profile_modes[str(mode)] = mode_result
                noisy_mode_diagnostics[f"{profile_name}:{method}:{mode}"] = {
                    "success": bool(mode_result.get("success", False)),
                    "env_blocked": bool(mode_result.get("env_blocked", False)),
                    "reason": mode_result.get("reason", None),
                    "error": mode_result.get("error", None),
                }
            method_payloads[str(method)] = {"modes": profile_modes}

        suzuki_alias_modes = (
            method_payloads.get("suzuki2", {}).get("modes", {})
            if isinstance(method_payloads.get("suzuki2", {}), dict)
            else {}
        )
        dynamics_noisy["profiles"][profile_name] = {
            "drive_enabled": bool(profile_cfg is not None),
            "methods": method_payloads,
            "modes": suzuki_alias_modes,
        }

    dynamics_benchmark_rows = _collect_noisy_benchmark_rows(dynamics_noisy)

    run_command = current_command_string()

    payload: dict[str, Any] = {
        "settings": {
            "L": int(args.L),
            "problem": "hh",
            "t": float(args.t),
            "u": float(args.u),
            "dv": float(args.dv),
            "omega0": float(args.omega0),
            "g_ep": float(args.g_ep),
            "n_ph_max": int(args.n_ph_max),
            "boson_encoding": str(args.boson_encoding),
            "ordering": str(args.ordering),
            "boundary": str(args.boundary),
            "t_final": float(args.t_final),
            "num_times": int(args.num_times),
            "trotter_steps": int(args.trotter_steps),
            "exact_steps_multiplier": int(args.exact_steps_multiplier),
            "fidelity_subspace_energy_tol": float(args.fidelity_subspace_energy_tol),
            "include_drive_profile": bool(args.include_drive_profile),
            "smoke_test_intentionally_weak": bool(args.smoke_test_intentionally_weak),
            "noise_modes": noise_modes,
            "noisy_methods": [str(x) for x in noisy_methods],
            "shots": int(args.shots),
            "oracle_repeats": int(args.oracle_repeats),
            "oracle_aggregate": str(args.oracle_aggregate),
            "benchmark_active_coeff_tol": float(args.benchmark_active_coeff_tol),
            "drive_profile": drive_profile,
            "transition_policy": {
                "window_k": int(args.window_k),
                "slope_epsilon": float(args.slope_epsilon),
                "patience": int(args.patience),
                "min_points_before_switch": int(args.min_points_before_switch),
            },
        },
        "stage_pipeline": {
            "warm_start": warm_payload,
            "adapt_pool_b": adapt_payload,
            "conventional_vqe": final_payload,
        },
        "transitions": {
            "warm_to_adapt": dict(warm_payload.get("transition", {})),
            "adapt_to_vqe": dict(adapt_payload.get("transition", {})),
        },
        "pool_B": {
            **pool_B_meta,
            "strict_union_families": ["uccsd_lifted", "hva", "paop_full"],
        },
        "hardcoded_superset": hardcoded_superset,
        "dynamics_noiseless": dynamics_noiseless,
        "dynamics_noisy": dynamics_noisy,
        "dynamics_benchmarks": {
            "rows": dynamics_benchmark_rows,
            "metric_fields": [
                "term_exp_count_total",
                "pauli_rot_count_total",
                "cx_proxy_total",
                "sq_proxy_total",
                "depth_proxy_total",
                "wall_total_s",
                "oracle_eval_s_total",
                "oracle_calls_total",
            ],
        },
        "comparisons": {},
        "summary": {},
        "equation_registry": {},
        "plot_contracts": {},
        "diagnostics": {
            "noisy_mode_diagnostics": noisy_mode_diagnostics,
            "pool_signature_space": int(len(pool_B_source_by_sig)),
            "pool_b_audit": pool_b_audit,
            "coeff_map_exyz": flatten_coeff_map_real_imag(coeff_map_exyz),
            "ordered_labels_count": int(len(ordered_labels_exyz)),
            "exact_sector_energy": float(exact_sector_energy),
            "exact_basis_energy": float(exact_basis_energy),
            "metric_definitions": {
                "delta_abs": "abs(E - E_exact_sector)",
                "energy_static": "<psi|H_static|psi>",
                "energy_total": "<psi|H_static + H_drive(t)|psi>",
                "doublon": "<n_up(site0) n_dn(site0)>",
                "staggered": "(1/L) sum_j (-1)^j <n_j>",
            },
        },
        "run_command": str(run_command),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    payload["comparisons"] = _compute_comparisons(payload)
    payload["summary"] = _build_summary(payload)
    equation_registry, plot_contracts = _build_equation_registry_and_contracts(payload)
    payload["equation_registry"] = equation_registry
    payload["plot_contracts"] = plot_contracts

    _json_dump(output_json, payload)

    if not bool(args.skip_pdf):
        _write_pdf(output_pdf, payload)

    if bool(args.require_at_least_one_noisy):
        completed = int(payload["summary"].get("noisy_method_modes_completed", payload["summary"].get("noisy_modes_completed", 0)))
        if completed < 1:
            raise RuntimeError(
                "No noisy mode completed successfully. "
                "Set --require-at-least-one-noisy false to treat this as non-fatal."
            )

    _ai_log(
        "hh_noise_robustness_done",
        output_json=str(output_json),
        output_pdf=(None if bool(args.skip_pdf) else str(output_pdf)),
        noisy_completed=int(payload["summary"].get("noisy_method_modes_completed", payload["summary"].get("noisy_modes_completed", 0))),
        noisy_total=int(payload["summary"].get("noisy_method_modes_total", payload["summary"].get("noisy_modes_total", 0))),
    )


if __name__ == "__main__":
    main()
