#!/usr/bin/env python3
"""L=3 HH accessibility-fix sweep (tests-only).

Implements and compares three ADAPT fixes on the same PAOP pool:
  1) Warm-start reference (pre-optimized HH physical-termwise VQE state)
  2) Finite-angle selection (top-k probe by delta-E)
  3) Early diversity constraints (reduce repeated-family collapse)

Runs 3 heavy rungs per fix (9 runs total) and writes JSON/CSV/MD summaries.
This script is intentionally isolated under test/ and does not edit core code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.hubbard_latex_python_pairs import SPIN_DN, SPIN_UP, build_hubbard_holstein_hamiltonian, mode_index
from src.quantum.operator_pools import make_pool as make_paop_pool
from src.quantum.vqe_latex_python_pairs import (
    AnsatzTerm,
    HubbardHolsteinPhysicalTermwiseAnsatz,
    apply_exp_pauli_polynomial,
    apply_pauli_string,
    exact_ground_energy_sector_hh,
    expval_pauli_polynomial,
    hubbard_holstein_reference_state,
    jw_number_operator,
    vqe_minimize,
)

try:
    from scipy.optimize import minimize as scipy_minimize
except Exception as exc:
    raise RuntimeError("SciPy is required for this sweep.") from exc


@dataclass(frozen=True)
class CommonConfig:
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
    sector_n_up: int
    sector_n_dn: int
    pool_name: str
    paop_r: int
    paop_split_paulis: bool
    paop_prune_eps: float
    paop_normalization: str


@dataclass(frozen=True)
class RungSpec:
    rung_id: str
    adapt_max_depth: int
    adapt_maxiter: int
    eps_grad: float
    eps_energy: float


@dataclass(frozen=True)
class WarmStartSpec:
    reps: int
    restarts: int
    maxiter: int


@dataclass(frozen=True)
class FiniteAngleSpec:
    delta: float
    top_k_probe: int
    min_probe_improvement: float


@dataclass(frozen=True)
class DiversitySpec:
    diversity_steps: int
    max_consecutive_same_family: int
    must_include_drag_by_depth: int
    drag_min_grad: float


def _poly_signature(poly: Any, tol: float = 1e-12) -> tuple[tuple[str, float], ...]:
    items: list[tuple[str, float]] = []
    for term in poly.return_polynomial():
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        if abs(coeff.imag) > 1e-10:
            raise ValueError(f"Non-negligible imaginary coefficient: {coeff} ({label})")
        items.append((label, round(float(coeff.real), 12)))
    items.sort()
    return tuple(items)


def _apply_pauli_polynomial(state: np.ndarray, poly: Any) -> np.ndarray:
    terms = poly.return_polynomial()
    if not terms:
        return np.zeros_like(state)
    nq = int(terms[0].nqubit())
    out = np.zeros_like(state)
    id_str = "e" * nq
    for term in terms:
        coeff = complex(term.p_coeff)
        if abs(coeff) <= 1e-15:
            continue
        ps = str(term.pw2strng())
        out += coeff * state if ps == id_str else coeff * apply_pauli_string(state, ps)
    return out


# Built-in math:
#   dE/dtheta|_0 = i<psi|[H,G]|psi> = 2 Im(<H psi | G psi>)
def _commutator_gradient(h_poly: Any, op: AnsatzTerm, psi_current: np.ndarray) -> float:
    g_psi = _apply_pauli_polynomial(psi_current, op.polynomial)
    h_psi = _apply_pauli_polynomial(psi_current, h_poly)
    return float(2.0 * np.vdot(h_psi, g_psi).imag)


def _prepare_state(psi_ref: np.ndarray, selected_ops: list[AnsatzTerm], theta: np.ndarray) -> np.ndarray:
    psi = np.array(psi_ref, copy=True)
    for k, op in enumerate(selected_ops):
        psi = apply_exp_pauli_polynomial(psi, op.polynomial, float(theta[k]))
    return psi


def _energy(h_poly: Any, psi_ref: np.ndarray, selected_ops: list[AnsatzTerm], theta: np.ndarray) -> float:
    psi = _prepare_state(psi_ref, selected_ops, theta)
    return float(expval_pauli_polynomial(psi, h_poly))


def _operator_family(label: str) -> str:
    s = str(label)
    if "paop_disp(" in s:
        return "disp"
    if "paop_hopdrag(" in s:
        return "hopdrag"
    if "paop_curdrag(" in s:
        return "curdrag"
    if "hop(" in s:
        return "hop"
    if "onsite(" in s:
        return "onsite"
    return "other"


def _top_gradients(h_poly: Any, pool: list[AnsatzTerm], psi_ref: np.ndarray, topk: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, op in enumerate(pool):
        grad = _commutator_gradient(h_poly, op, psi_ref)
        rows.append(
            {
                "idx": int(idx),
                "label": str(op.label),
                "family": _operator_family(str(op.label)),
                "gradient": float(grad),
                "abs_gradient": float(abs(grad)),
            }
        )
    rows.sort(key=lambda r: float(r["abs_gradient"]), reverse=True)
    return rows[: int(topk)]


def _sector_diagnostics(psi: np.ndarray, cfg: CommonConfig) -> dict[str, float]:
    nq = int(round(math.log2(int(psi.size))))
    n_up = 0.0
    n_dn = 0.0
    for i in range(int(cfg.L)):
        p_up = mode_index(i, SPIN_UP, indexing=str(cfg.ordering), n_sites=int(cfg.L))
        p_dn = mode_index(i, SPIN_DN, indexing=str(cfg.ordering), n_sites=int(cfg.L))
        n_up += float(np.real(expval_pauli_polynomial(psi, jw_number_operator("JW", nq, int(p_up)))))
        n_dn += float(np.real(expval_pauli_polynomial(psi, jw_number_operator("JW", nq, int(p_dn)))))
    return {
        "N_up_target": float(cfg.sector_n_up),
        "N_dn_target": float(cfg.sector_n_dn),
        "N_up_expect": float(n_up),
        "N_dn_expect": float(n_dn),
        "N_up_abs_err": float(abs(n_up - float(cfg.sector_n_up))),
        "N_dn_abs_err": float(abs(n_dn - float(cfg.sector_n_dn))),
        "N_sector_abs_err_sum": float(abs(n_up - float(cfg.sector_n_up)) + abs(n_dn - float(cfg.sector_n_dn))),
    }


def _build_hamiltonian_and_ref(cfg: CommonConfig) -> tuple[Any, np.ndarray, float]:
    h_poly = build_hubbard_holstein_hamiltonian(
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
    psi_ref = np.asarray(
        hubbard_holstein_reference_state(
            dims=int(cfg.L),
            num_particles=(int(cfg.sector_n_up), int(cfg.sector_n_dn)),
            n_ph_max=int(cfg.n_ph_max),
            boson_encoding=str(cfg.boson_encoding),
            indexing=str(cfg.ordering),
        ),
        dtype=complex,
    )
    e_exact = float(
        exact_ground_energy_sector_hh(
            h_poly,
            num_sites=int(cfg.L),
            num_particles=(int(cfg.sector_n_up), int(cfg.sector_n_dn)),
            n_ph_max=int(cfg.n_ph_max),
            boson_encoding=str(cfg.boson_encoding),
            indexing=str(cfg.ordering),
        )
    )
    return h_poly, psi_ref, e_exact


def _build_paop_pool(cfg: CommonConfig) -> list[AnsatzTerm]:
    specs = make_paop_pool(
        str(cfg.pool_name),
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
    return [AnsatzTerm(label=str(lbl), polynomial=poly) for lbl, poly in specs]


def _warm_start_state(
    cfg: CommonConfig,
    h_poly: Any,
    psi_ref: np.ndarray,
    ws: WarmStartSpec,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    ansatz = HubbardHolsteinPhysicalTermwiseAnsatz(
        dims=int(cfg.L),
        J=float(cfg.t),
        U=float(cfg.U),
        omega0=float(cfg.omega0),
        g=float(cfg.g_ep),
        n_ph_max=int(cfg.n_ph_max),
        boson_encoding=str(cfg.boson_encoding),
        v=float(cfg.dv),
        reps=int(ws.reps),
        repr_mode="JW",
        indexing=str(cfg.ordering),
        pbc=(str(cfg.boundary).strip().lower() == "periodic"),
    )
    result = vqe_minimize(
        h_poly,
        ansatz,
        psi_ref,
        restarts=int(ws.restarts),
        seed=7,
        maxiter=int(ws.maxiter),
        method="COBYLA",
    )
    theta = np.asarray(result.theta, dtype=float)
    psi_warm = np.asarray(ansatz.prepare_state(theta, psi_ref), dtype=complex).reshape(-1)
    psi_warm = psi_warm / np.linalg.norm(psi_warm)
    return {
        "psi_start": psi_warm,
        "E_warm": float(result.energy),
        "warm_num_parameters": int(ansatz.num_parameters),
        "warm_nfev": int(result.nfev),
        "warm_nit": int(result.nit),
        "warm_runtime_s": float(time.perf_counter() - t0),
    }


def _select_by_finite_angle(
    h_poly: Any,
    psi_ref: np.ndarray,
    selected_ops: list[AnsatzTerm],
    theta: np.ndarray,
    candidates: list[int],
    pool: list[AnsatzTerm],
    top_k_probe: int,
    delta: float,
    min_probe_improvement: float,
    energy_current: float,
    nfev_counter: list[int],
) -> tuple[int | None, float | None, float, dict[str, Any]]:
    # Keep strongest gradient candidates; break ties by index for determinism.
    ranked = sorted(candidates, key=lambda i: float(i))
    # We need gradients for ranking.
    psi_current = _prepare_state(psi_ref, selected_ops, theta)
    grad_pairs = []
    for idx in ranked:
        g = abs(_commutator_gradient(h_poly, pool[idx], psi_current))
        grad_pairs.append((float(g), int(idx)))
    grad_pairs.sort(key=lambda x: x[0], reverse=True)

    probe_indices = [idx for _g, idx in grad_pairs[: int(max(1, top_k_probe))]]

    best_energy = float(energy_current)
    best_idx: int | None = None
    best_theta_init: float | None = None
    best_sign = "none"

    for idx in probe_indices:
        for sign, trial_theta in (("plus", float(delta)), ("minus", -float(delta))):
            trial_ops = selected_ops + [pool[idx]]
            trial_vec = np.append(theta, trial_theta)
            e_probe = _energy(h_poly, psi_ref, trial_ops, trial_vec)
            nfev_counter[0] += 1
            if e_probe < best_energy:
                best_energy = float(e_probe)
                best_idx = int(idx)
                best_theta_init = float(trial_theta)
                best_sign = sign

    improvement = float(energy_current - best_energy)
    if best_idx is None or improvement <= float(min_probe_improvement):
        return None, None, best_energy, {
            "probe_indices": probe_indices,
            "best_sign": best_sign,
            "probe_improvement": improvement,
        }
    return best_idx, best_theta_init, best_energy, {
        "probe_indices": probe_indices,
        "best_sign": best_sign,
        "probe_improvement": improvement,
    }


def _run_adapt_variant(
    *,
    variant_id: str,
    h_poly: Any,
    psi_start: np.ndarray,
    pool: list[AnsatzTerm],
    rung: RungSpec,
    wallclock_cap_s: int,
    finite: FiniteAngleSpec | None,
    diversity: DiversitySpec | None,
    default_allow_repeats: bool,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    selected_ops: list[AnsatzTerm] = []
    theta = np.zeros(0, dtype=float)
    history = []
    trace: list[dict[str, Any]] = []
    nfev_total = 0
    nit_total = 0
    stop_reason = "max_depth"

    energy_current = float(expval_pauli_polynomial(psi_start, h_poly))
    history.append(float(energy_current))
    nfev_total += 1

    available = set(range(len(pool)))
    last_family = ""
    consecutive_same_family = 0
    has_selected_drag = False

    for depth in range(int(rung.adapt_max_depth)):
        elapsed = time.perf_counter() - t0
        if elapsed >= float(wallclock_cap_s):
            stop_reason = "wallclock_cap"
            break

        # Candidate indices depend on repeat policy and diversity early phase.
        early_phase = bool(diversity is not None and depth < int(diversity.diversity_steps))
        if early_phase:
            candidate_indices = sorted(available)
        else:
            candidate_indices = list(range(len(pool))) if bool(default_allow_repeats) else sorted(available)

        if not candidate_indices:
            stop_reason = "pool_exhausted"
            break

        psi_current = _prepare_state(psi_start, selected_ops, theta)
        gradients = {idx: float(_commutator_gradient(h_poly, pool[idx], psi_current)) for idx in candidate_indices}
        grad_abs = {idx: abs(val) for idx, val in gradients.items()}

        # Diversity filtering (fix 3)
        if diversity is not None and early_phase:
            # Cap repeated family streak.
            if last_family:
                filtered = [
                    idx for idx in candidate_indices
                    if not (
                        _operator_family(str(pool[idx].label)) == last_family
                        and consecutive_same_family >= int(diversity.max_consecutive_same_family)
                    )
                ]
                if filtered:
                    candidate_indices = filtered

            # Encourage at least one drag operator by a target depth when signal exists.
            if (
                (depth + 1) >= int(diversity.must_include_drag_by_depth)
                and (not has_selected_drag)
            ):
                drag_candidates = [
                    idx for idx in candidate_indices
                    if _operator_family(str(pool[idx].label)) in {"hopdrag", "curdrag"}
                    and grad_abs.get(idx, 0.0) >= float(diversity.drag_min_grad)
                ]
                if drag_candidates:
                    candidate_indices = drag_candidates

        selection_mode = "gradient"
        selected_idx: int | None = None
        init_theta = 0.0
        probe_meta: dict[str, Any] = {}

        if finite is not None:
            selection_mode = "finite_angle"
            selected_idx, theta_probe, _best_probe_energy, probe_meta = _select_by_finite_angle(
                h_poly=h_poly,
                psi_ref=psi_start,
                selected_ops=selected_ops,
                theta=theta,
                candidates=candidate_indices,
                pool=pool,
                top_k_probe=int(finite.top_k_probe),
                delta=float(finite.delta),
                min_probe_improvement=float(finite.min_probe_improvement),
                energy_current=float(energy_current),
                nfev_counter=[nfev_total],
            )
            # nfev counter was local list; update total from returned meta pattern.
            # Recompute probe eval count deterministically:
            probe_count = int(2 * len(probe_meta.get("probe_indices", [])))
            nfev_total += probe_count
            if selected_idx is None:
                stop_reason = "finite_probe_stall"
                break
            init_theta = float(theta_probe)
        else:
            # Gradient-based selection
            best_pair = max(((grad_abs[idx], idx) for idx in candidate_indices), key=lambda x: (x[0], -x[1]))
            best_grad = float(best_pair[0])
            selected_idx = int(best_pair[1])
            if best_grad < float(rung.eps_grad):
                stop_reason = "eps_grad"
                break

        assert selected_idx is not None
        selected_ops.append(pool[selected_idx])
        theta = np.append(theta, float(init_theta))

        if early_phase or (not bool(default_allow_repeats)):
            available.discard(selected_idx)

        energy_prev = float(energy_current)

        def objective(x: np.ndarray) -> float:
            return _energy(h_poly, psi_start, selected_ops, x)

        res = scipy_minimize(
            objective,
            theta,
            method="COBYLA",
            options={"maxiter": int(rung.adapt_maxiter), "rhobeg": 0.3},
        )
        theta = np.asarray(res.x, dtype=float)
        energy_current = float(res.fun)
        history.append(float(energy_current))

        nfev_total += int(getattr(res, "nfev", 0))
        nit_total += int(getattr(res, "nit", 0))

        fam = _operator_family(str(pool[selected_idx].label))
        if fam in {"hopdrag", "curdrag"}:
            has_selected_drag = True

        if fam == last_family:
            consecutive_same_family += 1
        else:
            last_family = fam
            consecutive_same_family = 1

        trace.append(
            {
                "depth": int(depth + 1),
                "selection_mode": selection_mode,
                "selected_idx": int(selected_idx),
                "selected_label": str(pool[selected_idx].label),
                "selected_family": fam,
                "init_theta": float(init_theta),
                "max_abs_grad_among_candidates": float(max(grad_abs[idx] for idx in candidate_indices)),
                "E_before": float(energy_prev),
                "E_after": float(energy_current),
                "delta_E_step": float(energy_current - energy_prev),
                "probe_meta": probe_meta,
            }
        )

        if abs(energy_current - energy_prev) < float(rung.eps_energy):
            stop_reason = "eps_energy"
            break

    psi_best = _prepare_state(psi_start, selected_ops, theta)
    psi_best = np.asarray(psi_best, dtype=complex).reshape(-1)
    psi_best = psi_best / np.linalg.norm(psi_best)

    fam_counts: dict[str, int] = {}
    for row in trace:
        fam = str(row["selected_family"])
        fam_counts[fam] = fam_counts.get(fam, 0) + 1

    return {
        "variant_id": str(variant_id),
        "E_best": float(min(history)),
        "E_last": float(history[-1]),
        "adapt_depth_reached": int(len(selected_ops)),
        "adapt_stop_reason": str(stop_reason),
        "num_parameters": int(theta.size),
        "nfev_total": int(nfev_total),
        "nit_total": int(nit_total),
        "runtime_s": float(time.perf_counter() - t0),
        "history": [float(x) for x in history],
        "selected_trace": trace,
        "selected_family_counts": fam_counts,
        "psi_best": psi_best.tolist(),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L=3 HH accessibility-fix heavy sweep (tests-only).")

    p.add_argument("--L", type=int, default=3)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--U", type=float, default=4.0)
    p.add_argument("--dv", type=float, default=0.0)
    p.add_argument("--omega0", type=float, default=1.0)
    p.add_argument("--g-ep", type=float, default=0.5, dest="g_ep")
    p.add_argument("--n-ph-max", type=int, default=1, dest="n_ph_max")
    p.add_argument("--boson-encoding", type=str, default="binary", choices=["binary", "unary"], dest="boson_encoding")
    p.add_argument("--ordering", type=str, default="blocked", choices=["blocked", "interleaved"])
    p.add_argument("--boundary", type=str, default="open", choices=["open", "periodic"])
    p.add_argument("--sector", type=str, default="2,1")

    p.add_argument("--pool-name", type=str, default="paop_lf_std")
    p.add_argument("--paop-r", type=int, default=1)
    p.add_argument("--paop-split-paulis", action="store_true")
    p.add_argument("--paop-prune-eps", type=float, default=0.0)
    p.add_argument("--paop-normalization", type=str, default="none", choices=["none", "fro", "maxcoeff"])

    p.add_argument("--default-allow-repeats", action="store_true")
    p.add_argument("--no-default-allow-repeats", dest="default_allow_repeats", action="store_false")
    p.set_defaults(default_allow_repeats=True)

    p.add_argument("--wallclock-cap-s", type=int, default=1200)
    p.add_argument("--topk-grad0", type=int, default=15)

    p.add_argument(
        "--output-prefix",
        type=str,
        default=str(REPO_ROOT / "artifacts" / "tmp" / "l3_hh_accessibility_fixes"),
        help="Outputs: <prefix>_runs.json/.csv/.md",
    )
    return p.parse_args(argv)


def _build_common(args: argparse.Namespace) -> CommonConfig:
    parts = [x.strip() for x in str(args.sector).split(",")]
    if len(parts) != 2:
        raise ValueError("--sector must be n_up,n_dn")
    n_up, n_dn = int(parts[0]), int(parts[1])
    return CommonConfig(
        L=int(args.L),
        t=float(args.t),
        U=float(args.U),
        dv=float(args.dv),
        omega0=float(args.omega0),
        g_ep=float(args.g_ep),
        n_ph_max=int(args.n_ph_max),
        boson_encoding=str(args.boson_encoding),
        ordering=str(args.ordering),
        boundary=str(args.boundary),
        sector_n_up=int(n_up),
        sector_n_dn=int(n_dn),
        pool_name=str(args.pool_name),
        paop_r=int(args.paop_r),
        paop_split_paulis=bool(args.paop_split_paulis),
        paop_prune_eps=float(args.paop_prune_eps),
        paop_normalization=str(args.paop_normalization),
    )


def _rungs() -> list[RungSpec]:
    return [
        RungSpec("A", adapt_max_depth=80, adapt_maxiter=3000, eps_grad=1e-6, eps_energy=1e-8),
        RungSpec("B", adapt_max_depth=120, adapt_maxiter=5000, eps_grad=5e-7, eps_energy=1e-9),
        RungSpec("C", adapt_max_depth=160, adapt_maxiter=8000, eps_grad=1e-7, eps_energy=1e-9),
    ]


def _warm_specs() -> dict[str, WarmStartSpec]:
    return {
        "A": WarmStartSpec(reps=2, restarts=4, maxiter=2400),
        "B": WarmStartSpec(reps=3, restarts=5, maxiter=4000),
        "C": WarmStartSpec(reps=3, restarts=6, maxiter=6000),
    }


def _finite_specs() -> dict[str, FiniteAngleSpec]:
    return {
        "A": FiniteAngleSpec(delta=0.08, top_k_probe=6, min_probe_improvement=1e-12),
        "B": FiniteAngleSpec(delta=0.10, top_k_probe=8, min_probe_improvement=1e-12),
        "C": FiniteAngleSpec(delta=0.12, top_k_probe=10, min_probe_improvement=1e-12),
    }


def _diversity_specs() -> dict[str, DiversitySpec]:
    return {
        "A": DiversitySpec(diversity_steps=8, max_consecutive_same_family=2, must_include_drag_by_depth=6, drag_min_grad=1e-8),
        "B": DiversitySpec(diversity_steps=10, max_consecutive_same_family=2, must_include_drag_by_depth=6, drag_min_grad=1e-8),
        "C": DiversitySpec(diversity_steps=12, max_consecutive_same_family=2, must_include_drag_by_depth=6, drag_min_grad=1e-8),
    }


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = _build_common(args)

    if int(cfg.L) != 3:
        raise ValueError("This sweep is intended for L=3.")
    if str(cfg.boundary).strip().lower() != "open":
        raise ValueError("This sweep is configured for open boundary runs.")

    h_poly, psi_ref, e_exact = _build_hamiltonian_and_ref(cfg)
    pool = _build_paop_pool(cfg)
    if len(pool) == 0:
        raise RuntimeError("PAOP pool is empty.")

    grad0 = _top_gradients(h_poly, pool, psi_ref, topk=int(args.topk_grad0))

    out_prefix = Path(args.output_prefix)
    out_json = out_prefix.with_suffix("_runs.json") if out_prefix.suffix else Path(str(out_prefix) + "_runs.json")
    out_csv = out_prefix.with_suffix("_summary.csv") if out_prefix.suffix else Path(str(out_prefix) + "_summary.csv")
    out_md = out_prefix.with_suffix("_best.md") if out_prefix.suffix else Path(str(out_prefix) + "_best.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    rung_map = {r.rung_id: r for r in _rungs()}
    warm_map = _warm_specs()
    finite_map = _finite_specs()
    div_map = _diversity_specs()

    run_order = []
    for fix_id in ("fix1_warm_start", "fix2_finite_angle", "fix3_diversity"):
        for rung_id in ("A", "B", "C"):
            run_order.append((fix_id, rung_id))

    global_start = time.perf_counter()

    for run_idx, (fix_id, rung_id) in enumerate(run_order, start=1):
        rung = rung_map[rung_id]
        print(f"RUN {run_idx}/9 start: {fix_id} rung={rung_id}", flush=True)
        run_t0 = time.perf_counter()

        run_payload: dict[str, Any] = {
            "run_id": f"{fix_id}_{rung_id}",
            "fix_id": str(fix_id),
            "rung_id": str(rung_id),
            "status": "ok",
            "error": "",
            "config": {
                "L": int(cfg.L),
                "sector": [int(cfg.sector_n_up), int(cfg.sector_n_dn)],
                "t": float(cfg.t),
                "U": float(cfg.U),
                "dv": float(cfg.dv),
                "omega0": float(cfg.omega0),
                "g_ep": float(cfg.g_ep),
                "n_ph_max": int(cfg.n_ph_max),
                "boson_encoding": str(cfg.boson_encoding),
                "ordering": str(cfg.ordering),
                "boundary": str(cfg.boundary),
                "pool_name": str(cfg.pool_name),
                "pool_size": int(len(pool)),
                "paop_r": int(cfg.paop_r),
                "paop_split_paulis": bool(cfg.paop_split_paulis),
                "paop_prune_eps": float(cfg.paop_prune_eps),
                "paop_normalization": str(cfg.paop_normalization),
                "wallclock_cap_s": int(args.wallclock_cap_s),
                "default_allow_repeats": bool(args.default_allow_repeats),
            },
            "rung": {
                "adapt_max_depth": int(rung.adapt_max_depth),
                "adapt_maxiter": int(rung.adapt_maxiter),
                "eps_grad": float(rung.eps_grad),
                "eps_energy": float(rung.eps_energy),
            },
            "E_exact_sector": float(e_exact),
            "step0_top_gradients": grad0,
        }

        try:
            psi_start = np.asarray(psi_ref, dtype=complex)
            warm_meta: dict[str, Any] = {}
            finite: FiniteAngleSpec | None = None
            diversity: DiversitySpec | None = None

            if fix_id == "fix1_warm_start":
                ws = warm_map[rung_id]
                warm_meta = _warm_start_state(cfg, h_poly, psi_ref, ws)
                psi_start = np.asarray(warm_meta.pop("psi_start"), dtype=complex)
                run_payload["warm_start"] = {**{k: v for k, v in warm_meta.items()}, **{
                    "reps": int(ws.reps), "restarts": int(ws.restarts), "maxiter": int(ws.maxiter)
                }}
            elif fix_id == "fix2_finite_angle":
                finite = finite_map[rung_id]
                run_payload["finite_angle"] = {
                    "delta": float(finite.delta),
                    "top_k_probe": int(finite.top_k_probe),
                    "min_probe_improvement": float(finite.min_probe_improvement),
                }
            elif fix_id == "fix3_diversity":
                diversity = div_map[rung_id]
                run_payload["diversity"] = {
                    "diversity_steps": int(diversity.diversity_steps),
                    "max_consecutive_same_family": int(diversity.max_consecutive_same_family),
                    "must_include_drag_by_depth": int(diversity.must_include_drag_by_depth),
                    "drag_min_grad": float(diversity.drag_min_grad),
                }
            else:
                raise ValueError(f"Unknown fix_id: {fix_id}")

            adapt = _run_adapt_variant(
                variant_id=fix_id,
                h_poly=h_poly,
                psi_start=psi_start,
                pool=pool,
                rung=rung,
                wallclock_cap_s=int(args.wallclock_cap_s),
                finite=finite,
                diversity=diversity,
                default_allow_repeats=bool(args.default_allow_repeats),
            )
            psi_best = np.asarray(adapt.pop("psi_best"), dtype=complex)
            sector_diag = _sector_diagnostics(psi_best, cfg)

            delta_e = abs(float(adapt["E_best"]) - float(e_exact))
            rel_err = float(delta_e / max(abs(float(e_exact)), 1e-14))

            run_payload["result"] = {
                **adapt,
                "delta_E_abs": float(delta_e),
                "relative_error_abs": float(rel_err),
                "sector_diag": sector_diag,
                "runtime_total_s": float(time.perf_counter() - run_t0),
            }

            print(
                f"RUN {run_idx}/9 done: {fix_id} rung={rung_id} "
                f"E_best={adapt['E_best']:.12f} ΔE={delta_e:.6e} depth={adapt['adapt_depth_reached']} "
                f"stop={adapt['adapt_stop_reason']} runtime={time.perf_counter()-run_t0:.1f}s",
                flush=True,
            )
        except Exception as exc:
            run_payload["status"] = "error"
            run_payload["error"] = str(exc)
            run_payload["result"] = {
                "runtime_total_s": float(time.perf_counter() - run_t0),
            }
            print(f"RUN {run_idx}/9 error: {fix_id} rung={rung_id} err={exc}", flush=True)

        rows.append(run_payload)

        # Persist incrementally so long sweeps are crash-resilient.
        bundle = {
            "generated_at_unix": time.time(),
            "runtime_total_s": float(time.perf_counter() - global_start),
            "rows": rows,
        }
        out_json.write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    # Build summary table rows.
    summary_rows: list[dict[str, Any]] = []
    for row in rows:
        result = row.get("result", {})
        sdiag = result.get("sector_diag", {})
        summary_rows.append(
            {
                "run_id": row.get("run_id", ""),
                "fix_id": row.get("fix_id", ""),
                "rung_id": row.get("rung_id", ""),
                "status": row.get("status", ""),
                "E_exact_sector": row.get("E_exact_sector", ""),
                "E_best": result.get("E_best", ""),
                "E_last": result.get("E_last", ""),
                "delta_E_abs": result.get("delta_E_abs", ""),
                "relative_error_abs": result.get("relative_error_abs", ""),
                "adapt_depth_reached": result.get("adapt_depth_reached", ""),
                "adapt_stop_reason": result.get("adapt_stop_reason", ""),
                "num_parameters": result.get("num_parameters", ""),
                "nfev_total": result.get("nfev_total", ""),
                "nit_total": result.get("nit_total", ""),
                "runtime_total_s": result.get("runtime_total_s", ""),
                "N_sector_abs_err_sum": sdiag.get("N_sector_abs_err_sum", ""),
                "error": row.get("error", ""),
            }
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        if summary_rows:
            w.writeheader()
            w.writerows(summary_rows)

    # Best-per-fix markdown.
    md_lines = [
        "# L=3 HH Accessibility Fix Sweep (Tests-Only)",
        "",
        f"Runtime total: {time.perf_counter() - global_start:.2f} s",
        "",
        "| fix_id | best_rung | best_delta_E | best_rel_err | depth | stop_reason |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for fix_id in ("fix1_warm_start", "fix2_finite_angle", "fix3_diversity"):
        ok_rows = [r for r in summary_rows if r["fix_id"] == fix_id and r["status"] == "ok" and r["delta_E_abs"] != ""]
        if not ok_rows:
            md_lines.append(f"| {fix_id} | - | - | - | - | error |")
            continue
        best = min(ok_rows, key=lambda r: float(r["delta_E_abs"]))
        md_lines.append(
            f"| {fix_id} | {best['rung_id']} | {float(best['delta_E_abs']):.6e} | "
            f"{float(best['relative_error_abs']):.6e} | {best['adapt_depth_reached']} | {best['adapt_stop_reason']} |"
        )

    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"WROTE {out_json}")
    print(f"WROTE {out_csv}")
    print(f"WROTE {out_md}")


if __name__ == "__main__":
    main()
