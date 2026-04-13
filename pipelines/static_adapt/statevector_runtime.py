from __future__ import annotations

import math
import time
from typing import Any, Callable

import numpy as np

from pipelines.static_adapt.builders.problem_setup import _normalize_state
from src.quantum.compiled_polynomial import (
    CompiledPolynomialAction,
    apply_compiled_polynomial as _apply_compiled_polynomial_shared,
    compile_polynomial_action as _compile_polynomial_action_shared,
)
from src.quantum.pauli_actions import (
    CompiledPauliAction,
    apply_compiled_pauli as _apply_compiled_pauli_shared,
    apply_exp_term as _apply_exp_term_shared,
    compile_pauli_action_exyz as _compile_pauli_action_exyz_shared,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial


def _to_ixyz(label_exyz: str) -> str:
    return str(label_exyz).replace("e", "i")


def _compile_pauli_action(label_exyz: str, nq: int) -> CompiledPauliAction:
    return _compile_pauli_action_exyz_shared(label_exyz=label_exyz, nq=nq)


def _apply_compiled_pauli(psi: np.ndarray, action: CompiledPauliAction) -> np.ndarray:
    return _apply_compiled_pauli_shared(psi=psi, action=action)


def _compile_polynomial_action(
    poly: Any,
    *,
    ignore_identity: bool = False,
    coefficient_tolerance: float = 1e-15,
    sort_terms: bool = True,
) -> CompiledPolynomialAction:
    poly_obj = poly if isinstance(poly, PauliPolynomial) else PauliPolynomial.from_pauli_terms(poly.return_polynomial())
    return _compile_polynomial_action_shared(
        poly_obj,
        ignore_identity=bool(ignore_identity),
        coefficient_tolerance=float(coefficient_tolerance),
        sort_terms=bool(sort_terms),
    )


def _apply_compiled_polynomial(state: np.ndarray, compiled_poly: CompiledPolynomialAction) -> np.ndarray:
    return _apply_compiled_polynomial_shared(state, compiled_poly)


def _apply_exp_term(
    psi: np.ndarray,
    action: CompiledPauliAction,
    coeff: complex,
    alpha: float,
) -> np.ndarray:
    return _apply_exp_term_shared(
        psi=psi,
        action=action,
        coeff=complex(coeff),
        alpha=float(alpha),
    )


def _evolve_trotter_suzuki2_absolute(
    psi0,
    ordered_labels,
    coeff_map,
    compiled_actions,
    time_value,
    trotter_steps,
):
    psi = np.array(psi0, copy=True)
    dt = float(time_value) / float(trotter_steps)
    half = dt / 2.0
    for _ in range(trotter_steps):
        for label in ordered_labels:
            psi = _apply_exp_term(psi, compiled_actions[label], coeff_map[label], half)
        for label in reversed(ordered_labels):
            psi = _apply_exp_term(psi, compiled_actions[label], coeff_map[label], half)
    return psi


def _expectation_hamiltonian(psi: np.ndarray, hmat: np.ndarray) -> float:
    return float(np.real(np.vdot(psi, hmat @ psi)))


def _occupation_site0(psi: np.ndarray, num_sites: int) -> tuple[float, float]:
    nq = int(np.log2(psi.size))
    probs = np.abs(psi) ** 2
    n_up = 0.0
    n_dn = 0.0
    for idx, p in enumerate(probs):
        if ((idx >> 0) & 1) == 1:
            n_up += float(p)
        if ((idx >> num_sites) & 1) == 1:
            n_dn += float(p)
    return n_up, n_dn


def _doublon_total(psi: np.ndarray, num_sites: int) -> float:
    probs = np.abs(psi) ** 2
    total = 0.0
    for idx, p in enumerate(probs):
        subtotal = 0.0
        for site in range(num_sites):
            n_up = (idx >> site) & 1
            n_dn = (idx >> (num_sites + site)) & 1
            subtotal += float(n_up & n_dn)
        total += subtotal * float(p)
    return total


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
    ai_log: Callable[..., None] | None = None,
) -> tuple[list[dict[str, float]], list[np.ndarray]]:
    if int(suzuki_order) != 2:
        raise ValueError("This script currently supports suzuki_order=2 only.")

    nq = len(ordered_labels_exyz[0]) if ordered_labels_exyz else int(np.log2(max(1, psi0.size)))
    evals, evecs = np.linalg.eigh(hmat)
    evecs_dag = np.conjugate(evecs).T

    compiled = {lbl: _compile_pauli_action(lbl, nq) for lbl in ordered_labels_exyz}
    times = np.linspace(0.0, float(t_final), int(num_times))
    n_times = int(times.size)
    stride = max(1, n_times // 20)
    t0 = time.perf_counter()
    if ai_log is not None:
        ai_log(
            "hardcoded_adapt_trajectory_start",
            L=int(num_sites),
            num_times=n_times,
            t_final=float(t_final),
            trotter_steps=int(trotter_steps),
        )

    rows: list[dict[str, float]] = []
    exact_states: list[np.ndarray] = []

    for idx, time_val in enumerate(times):
        tv = float(time_val)
        psi_exact = evecs @ (np.exp(-1j * evals * tv) * (evecs_dag @ psi0))
        psi_exact = _normalize_state(psi_exact)

        psi_trot = _evolve_trotter_suzuki2_absolute(
            psi0, ordered_labels_exyz, coeff_map_exyz, compiled, tv, int(trotter_steps),
        )

        fidelity = float(abs(np.vdot(psi_exact, psi_trot)) ** 2)
        n_up_exact, n_dn_exact = _occupation_site0(psi_exact, num_sites)
        n_up_trot, n_dn_trot = _occupation_site0(psi_trot, num_sites)

        rows.append({
            "time": tv,
            "fidelity": fidelity,
            "energy_exact": _expectation_hamiltonian(psi_exact, hmat),
            "energy_trotter": _expectation_hamiltonian(psi_trot, hmat),
            "n_up_site0_exact": n_up_exact,
            "n_up_site0_trotter": n_up_trot,
            "n_dn_site0_exact": n_dn_exact,
            "n_dn_site0_trotter": n_dn_trot,
            "doublon_exact": _doublon_total(psi_exact, num_sites),
            "doublon_trotter": _doublon_total(psi_trot, num_sites),
        })
        exact_states.append(psi_exact)
        if ai_log is not None and (idx == 0 or idx == n_times - 1 or ((idx + 1) % stride == 0)):
            ai_log(
                "hardcoded_adapt_trajectory_progress",
                step=int(idx + 1),
                total_steps=n_times,
                frac=round(float((idx + 1) / n_times), 6),
                time=tv,
                fidelity=float(fidelity),
                elapsed_sec=round(time.perf_counter() - t0, 6),
            )

    if ai_log is not None:
        ai_log("hardcoded_adapt_trajectory_done", L=int(num_sites), num_times=n_times)
    return rows, exact_states
