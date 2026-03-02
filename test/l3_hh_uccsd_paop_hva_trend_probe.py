#!/usr/bin/env python3
"""L=3 HH convergence trend probe in test-only scope.

Compares two ADAPT pool constructions under identical budgets:
  A) UCCSD(fermion-only lifted) + PAOP
  B) UCCSD(fermion-only lifted) + PAOP + HVA

Runs both A/B at two budgets:
  - medium: max_depth=20, maxiter=1200
  - heavy : max_depth=36, maxiter=2400

Outputs JSON in test/artifacts with traces and a heuristic
"likely_convergent_with_more_budget" decision.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from src.quantum.vqe_latex_python_pairs import (
    AnsatzTerm,
    HardcodedUCCSDAnsatz,
    HubbardHolsteinPhysicalTermwiseAnsatz,
    apply_exp_pauli_polynomial,
    apply_pauli_string,
    exact_ground_energy_sector_hh,
    expval_pauli_polynomial,
    half_filled_num_particles,
    hubbard_holstein_reference_state,
    jw_number_operator,
)

try:
    from scipy.optimize import minimize as scipy_minimize
except Exception as exc:  # pragma: no cover
    raise RuntimeError("SciPy is required for this trend probe script.") from exc


@dataclass(frozen=True)
class ProbeConfig:
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
    eps_grad: float
    eps_energy: float
    allow_repeats: bool
    paop_key: str
    paop_r: int
    paop_split_paulis: bool
    paop_prune_eps: float
    paop_normalization: str
    seed: int
    medium_depth: int
    medium_maxiter: int
    heavy_depth: int
    heavy_maxiter: int
    improve_abs_threshold: float
    improve_rel_threshold: float
    output_json: Path


@dataclass(frozen=True)
class StageBudget:
    name: str
    max_depth: int
    maxiter: int


def _poly_signature(poly: Any, tol: float = 1e-12) -> tuple[tuple[str, float], ...]:
    """Canonical real-valued signature used for deduplication."""
    items: list[tuple[str, float]] = []
    for term in poly.return_polynomial():
        ps = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        if abs(coeff.imag) > 1e-10:
            raise ValueError(f"Non-negligible imaginary coefficient in pool term: {coeff} ({ps})")
        items.append((ps, round(float(coeff.real), 12)))
    items.sort()
    return tuple(items)


def _apply_pauli_polynomial(state: np.ndarray, poly: Any) -> np.ndarray:
    terms = poly.return_polynomial()
    if not terms:
        return np.zeros_like(state)
    nq = int(terms[0].nqubit())
    id_ps = "e" * nq
    out = np.zeros_like(state)
    for term in terms:
        coeff = complex(term.p_coeff)
        if abs(coeff) <= 1e-15:
            continue
        ps = str(term.pw2strng())
        if ps == id_ps:
            out += coeff * state
        else:
            out += coeff * apply_pauli_string(state, ps)
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


def _source_tag(flags: dict[str, bool]) -> str:
    on = [k for k in ("uccsd", "paop", "hva") if bool(flags.get(k))]
    if not on:
        return "unknown"
    return "+".join(on)


def _build_uccsd_ferm_only_lifted_pool(cfg: ProbeConfig) -> list[AnsatzTerm]:
    n_sites = int(cfg.L)
    num_particles = (int(cfg.sector_n_up), int(cfg.sector_n_dn))
    base = HardcodedUCCSDAnsatz(
        dims=n_sites,
        num_particles=num_particles,
        reps=1,
        repr_mode="JW",
        indexing=str(cfg.ordering),
        include_singles=True,
        include_doubles=True,
    )
    ferm_nq = 2 * n_sites
    boson_bits = n_sites * int(boson_qubits_per_site(int(cfg.n_ph_max), str(cfg.boson_encoding)))
    nq_total = ferm_nq + boson_bits

    lifted_pool: list[AnsatzTerm] = []
    for op in base.base_terms:
        lifted = PauliPolynomial("JW")
        for term in op.polynomial.return_polynomial():
            coeff = complex(term.p_coeff)
            if abs(coeff) <= 1e-15:
                continue
            if abs(coeff.imag) > 1e-10:
                raise ValueError(f"Imaginary UCCSD coeff in {op.label}: {coeff}")
            ferm_ps = str(term.pw2strng())
            if len(ferm_ps) != ferm_nq:
                raise ValueError(f"Unexpected fermion Pauli length {len(ferm_ps)} != {ferm_nq}")
            full_ps = ("e" * boson_bits) + ferm_ps
            lifted.add_term(PauliTerm(nq_total, ps=full_ps, pc=float(coeff.real)))
        lifted_pool.append(AnsatzTerm(label=f"uccsd_ferm_lifted::{op.label}", polynomial=lifted))
    return lifted_pool


def _assert_uccsd_fermion_only(pool: list[AnsatzTerm], boson_bits: int, nq_total: int) -> None:
    if boson_bits <= 0:
        return
    boson_identity = "e" * boson_bits
    for op in pool:
        nontrivial_ferm_support = False
        for term in op.polynomial.return_polynomial():
            coeff = complex(term.p_coeff)
            if abs(coeff) <= 1e-15:
                continue
            ps = str(term.pw2strng())
            if len(ps) != int(nq_total):
                raise ValueError(
                    f"UCCSD lift length mismatch for {op.label}: {len(ps)} vs nq_total={nq_total}"
                )
            if ps[:boson_bits] != boson_identity:
                raise ValueError(f"UCCSD term leaks into boson register: {op.label} -> {ps}")
            ferm_ps = ps[boson_bits:]
            if any(ch != "e" for ch in ferm_ps):
                nontrivial_ferm_support = True
        if not nontrivial_ferm_support:
            raise ValueError(f"UCCSD operator has no nontrivial fermion support after lift: {op.label}")


def _build_paop_pool(cfg: ProbeConfig) -> list[AnsatzTerm]:
    specs = make_paop_pool(
        str(cfg.paop_key),
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


def _build_hva_pool(cfg: ProbeConfig) -> list[AnsatzTerm]:
    ptw = HubbardHolsteinPhysicalTermwiseAnsatz(
        dims=int(cfg.L),
        J=float(cfg.t),
        U=float(cfg.U),
        omega0=float(cfg.omega0),
        g=float(cfg.g_ep),
        n_ph_max=int(cfg.n_ph_max),
        boson_encoding=str(cfg.boson_encoding),
        v=float(cfg.dv),
        reps=1,
        repr_mode="JW",
        indexing=str(cfg.ordering),
        pbc=(str(cfg.boundary).strip().lower() == "periodic"),
    )
    return list(ptw.base_terms)


def _make_dedup_pool(
    ordered_components: list[tuple[str, list[AnsatzTerm]]],
) -> tuple[list[AnsatzTerm], dict[tuple[tuple[str, float], ...], dict[str, bool]], dict[str, Any]]:
    source_by_sig: dict[tuple[tuple[str, float], ...], dict[str, bool]] = {}
    raw_sizes: dict[str, int] = {}
    for source, ops in ordered_components:
        raw_sizes[source] = int(len(ops))
        for op in ops:
            sig = _poly_signature(op.polynomial)
            source_by_sig.setdefault(sig, {"uccsd": False, "paop": False, "hva": False})[source] = True

    dedup: list[AnsatzTerm] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for _, ops in ordered_components:
        for op in ops:
            sig = _poly_signature(op.polynomial)
            if sig in seen:
                continue
            seen.add(sig)
            dedup.append(op)

    dedup_source_counts = {"uccsd": 0, "paop": 0, "hva": 0}
    overlap_count = 0
    for op in dedup:
        sig = _poly_signature(op.polynomial)
        flags = source_by_sig.get(sig, {"uccsd": False, "paop": False, "hva": False})
        active = 0
        for key in ("uccsd", "paop", "hva"):
            if bool(flags.get(key)):
                dedup_source_counts[key] += 1
                active += 1
        if active >= 2:
            overlap_count += 1

    meta = {
        "raw_sizes": raw_sizes,
        "dedup_total": int(len(dedup)),
        "dedup_source_presence_counts": dedup_source_counts,
        "overlap_count": int(overlap_count),
    }
    return dedup, source_by_sig, meta


def _run_adapt(
    h_poly: Any,
    psi_ref: np.ndarray,
    pool: list[AnsatzTerm],
    source_by_sig: dict[tuple[tuple[str, float], ...], dict[str, bool]],
    *,
    max_depth: int,
    maxiter: int,
    eps_grad: float,
    eps_energy: float,
    allow_repeats: bool,
) -> dict[str, Any]:
    selected_ops: list[AnsatzTerm] = []
    theta = np.zeros(0, dtype=float)
    energy_current = float(expval_pauli_polynomial(psi_ref, h_poly))
    energy_trace = [float(energy_current)]
    grad_max_trace: list[float] = []
    trace: list[dict[str, Any]] = []
    nfev_total = 1
    nit_total = 0
    stop_reason = "max_depth"
    available = set(range(len(pool)))

    for depth in range(int(max_depth)):
        psi_current = _prepare_state(psi_ref, selected_ops, theta)
        gradients = np.zeros(len(pool), dtype=float)
        idx_iter = range(len(pool)) if bool(allow_repeats) else sorted(available)
        for i in idx_iter:
            gradients[i] = _commutator_gradient(h_poly, pool[i], psi_current)

        grad_mag = np.abs(gradients)
        if not bool(allow_repeats):
            mask = np.zeros(len(pool), dtype=bool)
            for i in available:
                mask[i] = True
            grad_mag[~mask] = 0.0

        best_idx = int(np.argmax(grad_mag))
        max_grad = float(grad_mag[best_idx])
        grad_max_trace.append(float(max_grad))
        if max_grad < float(eps_grad):
            stop_reason = "eps_grad"
            break

        selected_ops.append(pool[best_idx])
        theta = np.append(theta, 0.0)
        if not bool(allow_repeats):
            available.discard(best_idx)

        e_before = float(energy_current)

        def objective(x: np.ndarray) -> float:
            return _energy(h_poly, psi_ref, selected_ops, x)

        res = scipy_minimize(
            objective,
            theta,
            method="COBYLA",
            options={"maxiter": int(maxiter), "rhobeg": 0.3},
        )
        theta = np.asarray(res.x, dtype=float)
        energy_current = float(res.fun)
        energy_trace.append(float(energy_current))
        nfev_total += int(getattr(res, "nfev", 0))
        nit_total += int(getattr(res, "nit", 0))

        sig = _poly_signature(pool[best_idx].polynomial)
        trace.append(
            {
                "depth": int(depth + 1),
                "selected_idx": int(best_idx),
                "selected_label": str(pool[best_idx].label),
                "source": _source_tag(source_by_sig.get(sig, {})),
                "max_grad": float(max_grad),
                "E_before": float(e_before),
                "E_after": float(energy_current),
                "delta_E_step": float(energy_current - e_before),
            }
        )

        if abs(energy_current - e_before) < float(eps_energy):
            stop_reason = "eps_energy"
            break
        if (not bool(allow_repeats)) and len(available) == 0:
            stop_reason = "pool_exhausted"
            break

    psi_best = _prepare_state(psi_ref, selected_ops, theta)
    return {
        "E_best": float(np.min(np.asarray(energy_trace, dtype=float))),
        "E_last": float(energy_current),
        "adapt_depth_reached": int(len(selected_ops)),
        "adapt_stop_reason": str(stop_reason),
        "num_parameters": int(theta.size),
        "nfev_total": int(nfev_total),
        "nit_total": int(nit_total),
        "trace": trace,
        "energy_trace": energy_trace,
        "grad_max_trace": grad_max_trace,
        "psi_best": psi_best,
    }


def _sector_diagnostics(psi: np.ndarray, cfg: ProbeConfig) -> dict[str, float | int | bool]:
    n_sites = int(cfg.L)
    nq = int(round(math.log2(int(psi.size))))
    n_up_e = 0.0
    n_dn_e = 0.0
    for site in range(n_sites):
        p_up = mode_index(site, SPIN_UP, indexing=str(cfg.ordering), n_sites=n_sites)
        p_dn = mode_index(site, SPIN_DN, indexing=str(cfg.ordering), n_sites=n_sites)
        n_up_e += float(np.real(expval_pauli_polynomial(psi, jw_number_operator("JW", nq, int(p_up)))))
        n_dn_e += float(np.real(expval_pauli_polynomial(psi, jw_number_operator("JW", nq, int(p_dn)))))
    up_err = abs(n_up_e - float(cfg.sector_n_up))
    dn_err = abs(n_dn_e - float(cfg.sector_n_dn))
    total = up_err + dn_err
    return {
        "N_up_target": int(cfg.sector_n_up),
        "N_dn_target": int(cfg.sector_n_dn),
        "N_up_expect": float(n_up_e),
        "N_dn_expect": float(n_dn_e),
        "N_up_abs_err": float(up_err),
        "N_dn_abs_err": float(dn_err),
        "N_sector_abs_err_sum": float(total),
        "sector_leak_flag": bool(total > 1e-6),
    }


def _safe_run_stage(
    *,
    label: str,
    budget: StageBudget,
    h_poly: Any,
    psi_ref: np.ndarray,
    pool: list[AnsatzTerm],
    source_by_sig: dict[tuple[tuple[str, float], ...], dict[str, bool]],
    e_exact: float,
    cfg: ProbeConfig,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        run = _run_adapt(
            h_poly,
            psi_ref,
            pool,
            source_by_sig,
            max_depth=int(budget.max_depth),
            maxiter=int(budget.maxiter),
            eps_grad=float(cfg.eps_grad),
            eps_energy=float(cfg.eps_energy),
            allow_repeats=bool(cfg.allow_repeats),
        )
        psi_best = np.asarray(run["psi_best"], dtype=complex)
        run.pop("psi_best", None)
        delta = abs(float(run["E_best"]) - float(e_exact))
        return {
            "ok": True,
            "label": str(label),
            "budget": {"name": str(budget.name), "max_depth": int(budget.max_depth), "maxiter": int(budget.maxiter)},
            "runtime_s": float(time.perf_counter() - t0),
            "E_best": float(run["E_best"]),
            "E_last": float(run["E_last"]),
            "delta_E_abs": float(delta),
            "adapt_depth_reached": int(run["adapt_depth_reached"]),
            "adapt_stop_reason": str(run["adapt_stop_reason"]),
            "num_parameters": int(run["num_parameters"]),
            "nfev_total": int(run["nfev_total"]),
            "nit_total": int(run["nit_total"]),
            "energy_trace": run["energy_trace"],
            "grad_max_trace": run["grad_max_trace"],
            "selected_trace": run["trace"],
            "sector_diag": _sector_diagnostics(psi_best, cfg),
        }
    except Exception:
        return {
            "ok": False,
            "label": str(label),
            "budget": {"name": str(budget.name), "max_depth": int(budget.max_depth), "maxiter": int(budget.maxiter)},
            "runtime_s": float(time.perf_counter() - t0),
            "error": traceback.format_exc(limit=12),
        }


def _trend_judgement(
    medium: dict[str, Any],
    heavy: dict[str, Any],
    cfg: ProbeConfig,
) -> dict[str, Any]:
    if (not bool(medium.get("ok"))) or (not bool(heavy.get("ok"))):
        return {
            "assessable": False,
            "likely_convergent_with_more_budget": False,
            "reason": "missing_successful_run",
        }

    med_delta = float(medium["delta_E_abs"])
    heavy_delta = float(heavy["delta_E_abs"])
    abs_improve = float(med_delta - heavy_delta)
    rel_improve = float(abs_improve / max(med_delta, 1e-15))

    med_grad = medium.get("grad_max_trace", [])
    heavy_grad = heavy.get("grad_max_trace", [])
    med_last_grad = float(med_grad[-1]) if med_grad else float("inf")
    heavy_last_grad = float(heavy_grad[-1]) if heavy_grad else float("inf")
    grad_down = bool(heavy_last_grad <= med_last_grad + 1e-14)

    material = bool(
        abs_improve >= max(float(cfg.improve_abs_threshold), float(cfg.improve_rel_threshold) * max(med_delta, 1e-15))
    )
    stop_support = str(heavy.get("adapt_stop_reason", "")) in {"eps_grad", "eps_energy"}
    likely = bool(material and (grad_down or stop_support))

    return {
        "assessable": True,
        "likely_convergent_with_more_budget": bool(likely),
        "medium_delta_E_abs": float(med_delta),
        "heavy_delta_E_abs": float(heavy_delta),
        "abs_improvement": float(abs_improve),
        "rel_improvement": float(rel_improve),
        "medium_last_grad_max": float(med_last_grad),
        "heavy_last_grad_max": float(heavy_last_grad),
        "grad_down": bool(grad_down),
        "material_improvement": bool(material),
        "heavy_stop_reason": str(heavy.get("adapt_stop_reason", "")),
    }


def _build_cfg(args: argparse.Namespace) -> ProbeConfig:
    sector_txt = str(args.sector).strip()
    if "," in sector_txt:
        a, b = sector_txt.split(",", 1)
    else:
        bits = sector_txt.split()
        if len(bits) != 2:
            raise ValueError("Sector must be 'n_up,n_dn' or 'n_up n_dn'.")
        a, b = bits[0], bits[1]
    n_up, n_dn = int(a), int(b)

    return ProbeConfig(
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
        eps_grad=float(args.eps_grad),
        eps_energy=float(args.eps_energy),
        allow_repeats=bool(args.allow_repeats),
        paop_key=str(args.paop_key),
        paop_r=int(args.paop_r),
        paop_split_paulis=bool(args.paop_split_paulis),
        paop_prune_eps=float(args.paop_prune_eps),
        paop_normalization=str(args.paop_normalization),
        seed=int(args.seed),
        medium_depth=int(args.medium_depth),
        medium_maxiter=int(args.medium_maxiter),
        heavy_depth=int(args.heavy_depth),
        heavy_maxiter=int(args.heavy_maxiter),
        improve_abs_threshold=float(args.improve_abs_threshold),
        improve_rel_threshold=float(args.improve_rel_threshold),
        output_json=Path(args.output_json),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="L=3 HH quick convergence trend probe: UCCSD+PAOP vs UCCSD+PAOP+HVA."
    )
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
    p.add_argument("--sector", type=str, default="2,1", help="Fermion sector as n_up,n_dn.")

    p.add_argument("--eps-grad", type=float, default=1e-6)
    p.add_argument("--eps-energy", type=float, default=1e-8)
    p.add_argument("--allow-repeats", action="store_true")
    p.add_argument("--no-allow-repeats", dest="allow_repeats", action="store_false")
    p.set_defaults(allow_repeats=True)

    p.add_argument("--paop-key", type=str, default="paop_lf_std")
    p.add_argument("--paop-r", type=int, default=1)
    p.add_argument("--paop-split-paulis", action="store_true")
    p.add_argument("--paop-prune-eps", type=float, default=0.0)
    p.add_argument("--paop-normalization", type=str, default="none", choices=["none", "fro", "maxcoeff"])

    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--medium-depth", type=int, default=20)
    p.add_argument("--medium-maxiter", type=int, default=1200)
    p.add_argument("--heavy-depth", type=int, default=36)
    p.add_argument("--heavy-maxiter", type=int, default=2400)
    p.add_argument("--improve-abs-threshold", type=float, default=1e-3)
    p.add_argument("--improve-rel-threshold", type=float, default=0.10)

    p.add_argument(
        "--output-json",
        type=str,
        default=str(REPO_ROOT / "test" / "artifacts" / "l3_uccsd_paop_hva_trend.json"),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    cfg = _build_cfg(parse_args(argv))
    np.random.seed(int(cfg.seed))
    t_start = time.perf_counter()

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

    boson_bits = int(cfg.L) * int(boson_qubits_per_site(int(cfg.n_ph_max), str(cfg.boson_encoding)))
    nq_total = 2 * int(cfg.L) + boson_bits

    uccsd_pool = _build_uccsd_ferm_only_lifted_pool(cfg)
    _assert_uccsd_fermion_only(uccsd_pool, boson_bits, nq_total)
    paop_pool = _build_paop_pool(cfg)
    hva_pool = _build_hva_pool(cfg)

    pool_A, source_A, meta_A = _make_dedup_pool(
        [("uccsd", uccsd_pool), ("paop", paop_pool)]
    )
    pool_B, source_B, meta_B = _make_dedup_pool(
        [("uccsd", uccsd_pool), ("paop", paop_pool), ("hva", hva_pool)]
    )
    if len(pool_A) == 0 or len(pool_B) == 0:
        raise ValueError("One of the probe pools is empty after deduplication.")

    medium = StageBudget(name="medium", max_depth=int(cfg.medium_depth), maxiter=int(cfg.medium_maxiter))
    heavy = StageBudget(name="heavy", max_depth=int(cfg.heavy_depth), maxiter=int(cfg.heavy_maxiter))

    run_A_medium = _safe_run_stage(
        label="A_uccsd_plus_paop_medium",
        budget=medium,
        h_poly=h_poly,
        psi_ref=psi_ref,
        pool=pool_A,
        source_by_sig=source_A,
        e_exact=e_exact,
        cfg=cfg,
    )
    run_A_heavy = _safe_run_stage(
        label="A_uccsd_plus_paop_heavy",
        budget=heavy,
        h_poly=h_poly,
        psi_ref=psi_ref,
        pool=pool_A,
        source_by_sig=source_A,
        e_exact=e_exact,
        cfg=cfg,
    )
    run_B_medium = _safe_run_stage(
        label="B_uccsd_plus_paop_plus_hva_medium",
        budget=medium,
        h_poly=h_poly,
        psi_ref=psi_ref,
        pool=pool_B,
        source_by_sig=source_B,
        e_exact=e_exact,
        cfg=cfg,
    )
    run_B_heavy = _safe_run_stage(
        label="B_uccsd_plus_paop_plus_hva_heavy",
        budget=heavy,
        h_poly=h_poly,
        psi_ref=psi_ref,
        pool=pool_B,
        source_by_sig=source_B,
        e_exact=e_exact,
        cfg=cfg,
    )

    trend_A = _trend_judgement(run_A_medium, run_A_heavy, cfg)
    trend_B = _trend_judgement(run_B_medium, run_B_heavy, cfg)
    overall_likely = bool(trend_A.get("likely_convergent_with_more_budget") or trend_B.get("likely_convergent_with_more_budget"))

    payload: dict[str, Any] = {
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
            "eps_grad": float(cfg.eps_grad),
            "eps_energy": float(cfg.eps_energy),
            "allow_repeats": bool(cfg.allow_repeats),
            "seed": int(cfg.seed),
            "medium_depth": int(cfg.medium_depth),
            "medium_maxiter": int(cfg.medium_maxiter),
            "heavy_depth": int(cfg.heavy_depth),
            "heavy_maxiter": int(cfg.heavy_maxiter),
            "paop_key": str(cfg.paop_key),
            "paop_r": int(cfg.paop_r),
            "paop_split_paulis": bool(cfg.paop_split_paulis),
            "paop_prune_eps": float(cfg.paop_prune_eps),
            "paop_normalization": str(cfg.paop_normalization),
        },
        "exact": {
            "E_exact_sector": float(e_exact),
            "nq_total": int(nq_total),
            "fermion_qubits": int(2 * cfg.L),
            "boson_qubits": int(boson_bits),
        },
        "pool_components_raw": {
            "uccsd_ferm_only_lifted": int(len(uccsd_pool)),
            "paop": int(len(paop_pool)),
            "hva": int(len(hva_pool)),
        },
        "pool_A_uccsd_plus_paop": meta_A,
        "pool_B_uccsd_plus_paop_plus_hva": meta_B,
        "runs": {
            "A_medium": run_A_medium,
            "A_heavy": run_A_heavy,
            "B_medium": run_B_medium,
            "B_heavy": run_B_heavy,
        },
        "trend": {
            "A_uccsd_plus_paop": trend_A,
            "B_uccsd_plus_paop_plus_hva": trend_B,
            "likely_convergent_with_more_budget": bool(overall_likely),
        },
        "runtime_total_s": float(time.perf_counter() - t_start),
    }

    cfg.output_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"WROTE {cfg.output_json}")
    print(
        "A (UCCSD+PAOP): "
        f"med_ok={run_A_medium.get('ok')} heavy_ok={run_A_heavy.get('ok')} "
        f"med_ΔE={run_A_medium.get('delta_E_abs', float('nan')):.6e} "
        f"heavy_ΔE={run_A_heavy.get('delta_E_abs', float('nan')):.6e} "
        f"likely={trend_A.get('likely_convergent_with_more_budget')}"
    )
    print(
        "B (UCCSD+PAOP+HVA): "
        f"med_ok={run_B_medium.get('ok')} heavy_ok={run_B_heavy.get('ok')} "
        f"med_ΔE={run_B_medium.get('delta_E_abs', float('nan')):.6e} "
        f"heavy_ΔE={run_B_heavy.get('delta_E_abs', float('nan')):.6e} "
        f"likely={trend_B.get('likely_convergent_with_more_budget')}"
    )
    print(f"Overall likely_convergent_with_more_budget={overall_likely}")


if __name__ == "__main__":
    main()
