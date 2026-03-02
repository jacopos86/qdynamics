#!/usr/bin/env python3
"""Quick L=3 HH diagnostic: compare PAOP-only vs HVA+PAOP meta-pool ADAPT.

This script is intentionally isolated in the test folder and does not modify
core pipeline/operator code. It builds:
  1) PAOP pool: paop_lf_std
  2) Meta pool: HH physical-termwise HVA generators + PAOP generators
Then runs a small ADAPT loop for both under identical budgets.
"""

from __future__ import annotations

import argparse
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
    half_filled_num_particles,
    hubbard_holstein_reference_state,
    jw_number_operator,
)

try:
    from scipy.optimize import minimize as scipy_minimize
except Exception as exc:
    raise RuntimeError("SciPy is required for this quick ADAPT diagnostic.") from exc


@dataclass(frozen=True)
class RunConfig:
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
    adapt_max_depth: int
    adapt_maxiter: int
    adapt_eps_grad: float
    adapt_eps_energy: float
    allow_repeats: bool
    paop_r: int
    paop_split_paulis: bool
    paop_prune_eps: float
    paop_normalization: str
    topk: int
    improve_threshold: float
    output_json: Path


def _poly_signature(poly: Any, tol: float = 1e-12) -> tuple[tuple[str, float], ...]:
    """Canonical real-valued polynomial signature for dedupe/source-tagging."""
    items: list[tuple[str, float]] = []
    for term in poly.return_polynomial():
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        if abs(coeff.imag) > 1e-10:
            raise ValueError(f"Non-negligible imaginary coefficient in pool term: {coeff} ({label})")
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


def _source_tag(flags: dict[str, bool]) -> str:
    if bool(flags.get("hva")) and bool(flags.get("paop")):
        return "hva+paop"
    if bool(flags.get("hva")):
        return "hva"
    if bool(flags.get("paop")):
        return "paop"
    return "unknown"


def _top_gradients(
    h_poly: Any,
    pool: list[AnsatzTerm],
    psi_ref: np.ndarray,
    source_by_sig: dict[tuple[tuple[str, float], ...], dict[str, bool]],
    topk: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, op in enumerate(pool):
        grad = _commutator_gradient(h_poly, op, psi_ref)
        sig = _poly_signature(op.polynomial)
        rows.append(
            {
                "idx": int(idx),
                "label": str(op.label),
                "source": _source_tag(source_by_sig.get(sig, {})),
                "gradient": float(grad),
                "abs_gradient": float(abs(grad)),
            }
        )
    rows.sort(key=lambda r: float(r["abs_gradient"]), reverse=True)
    return rows[: int(topk)]


def _run_adapt(
    h_poly: Any,
    psi_ref: np.ndarray,
    pool: list[AnsatzTerm],
    source_by_sig: dict[tuple[tuple[str, float], ...], dict[str, bool]],
    cfg: RunConfig,
) -> dict[str, Any]:
    selected_ops: list[AnsatzTerm] = []
    theta = np.zeros(0, dtype=float)
    energy_current = float(expval_pauli_polynomial(psi_ref, h_poly))
    nfev_total = 1
    nit_total = 0
    history = [float(energy_current)]
    trace: list[dict[str, Any]] = []
    stop_reason = "max_depth"
    available = set(range(len(pool)))

    for depth in range(int(cfg.adapt_max_depth)):
        psi_current = _prepare_state(psi_ref, selected_ops, theta)
        gradients = np.zeros(len(pool), dtype=float)
        idx_iter = range(len(pool)) if bool(cfg.allow_repeats) else sorted(available)
        for i in idx_iter:
            gradients[i] = _commutator_gradient(h_poly, pool[i], psi_current)

        grad_mag = np.abs(gradients)
        if not bool(cfg.allow_repeats):
            mask = np.zeros(len(pool), dtype=bool)
            for i in available:
                mask[i] = True
            grad_mag[~mask] = 0.0
        best_idx = int(np.argmax(grad_mag))
        max_grad = float(grad_mag[best_idx])
        if max_grad < float(cfg.adapt_eps_grad):
            stop_reason = "eps_grad"
            break

        selected_ops.append(pool[best_idx])
        theta = np.append(theta, 0.0)
        if not bool(cfg.allow_repeats):
            available.discard(best_idx)

        energy_prev = float(energy_current)

        def objective(x: np.ndarray) -> float:
            return _energy(h_poly, psi_ref, selected_ops, x)

        res = scipy_minimize(
            objective,
            theta,
            method="COBYLA",
            options={"maxiter": int(cfg.adapt_maxiter), "rhobeg": 0.3},
        )
        theta = np.asarray(res.x, dtype=float)
        energy_current = float(res.fun)
        history.append(float(energy_current))
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
                "E_before": float(energy_prev),
                "E_after": float(energy_current),
                "delta_E_step": float(energy_current - energy_prev),
            }
        )

        if abs(energy_current - energy_prev) < float(cfg.adapt_eps_energy):
            stop_reason = "eps_energy"
            break
        if (not bool(cfg.allow_repeats)) and len(available) == 0:
            stop_reason = "pool_exhausted"
            break

    psi_best = _prepare_state(psi_ref, selected_ops, theta)
    psi_best = np.asarray(psi_best, dtype=complex).reshape(-1)
    psi_best = psi_best / np.linalg.norm(psi_best)

    return {
        "E_best": float(min(history)),
        "E_last": float(history[-1]),
        "adapt_depth_reached": int(len(selected_ops)),
        "adapt_stop_reason": str(stop_reason),
        "num_parameters": int(theta.size),
        "nfev_total": int(nfev_total),
        "nit_total": int(nit_total),
        "history": [float(x) for x in history],
        "trace": trace,
        "psi_best": psi_best.tolist(),
    }


def _sector_diagnostics(psi: np.ndarray, cfg: RunConfig) -> dict[str, float]:
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
        "N_sector_abs_err_sum": float(
            abs(n_up - float(cfg.sector_n_up)) + abs(n_dn - float(cfg.sector_n_dn))
        ),
    }


def _build_cfg(args: argparse.Namespace) -> RunConfig:
    if args.sector is None:
        n_up, n_dn = half_filled_num_particles(int(args.L))
    else:
        parts = [p.strip() for p in str(args.sector).split(",")]
        if len(parts) != 2:
            raise ValueError("--sector must be 'n_up,n_dn'")
        n_up, n_dn = int(parts[0]), int(parts[1])
    return RunConfig(
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
        adapt_max_depth=int(args.adapt_max_depth),
        adapt_maxiter=int(args.adapt_maxiter),
        adapt_eps_grad=float(args.adapt_eps_grad),
        adapt_eps_energy=float(args.adapt_eps_energy),
        allow_repeats=bool(args.allow_repeats),
        paop_r=int(args.paop_r),
        paop_split_paulis=bool(args.paop_split_paulis),
        paop_prune_eps=float(args.paop_prune_eps),
        paop_normalization=str(args.paop_normalization),
        topk=int(args.topk),
        improve_threshold=float(args.improve_threshold),
        output_json=Path(args.output_json),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick L=3 HH ADAPT comparison: PAOP vs HVA+PAOP meta-pool.")
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

    p.add_argument("--adapt-max-depth", type=int, default=20)
    p.add_argument("--adapt-maxiter", type=int, default=800)
    p.add_argument("--adapt-eps-grad", type=float, default=1e-6)
    p.add_argument("--adapt-eps-energy", type=float, default=1e-8)
    p.add_argument("--allow-repeats", action="store_true")
    p.add_argument("--no-allow-repeats", dest="allow_repeats", action="store_false")
    p.set_defaults(allow_repeats=True)

    p.add_argument("--paop-r", type=int, default=1)
    p.add_argument("--paop-split-paulis", action="store_true")
    p.add_argument("--paop-prune-eps", type=float, default=0.0)
    p.add_argument("--paop-normalization", type=str, default="none", choices=["none", "fro", "maxcoeff"])

    p.add_argument("--topk", type=int, default=15)
    p.add_argument("--improve-threshold", type=float, default=1e-2)
    p.add_argument(
        "--output-json",
        type=str,
        default=str(REPO_ROOT / "artifacts" / "tmp" / "l3_meta_pool_quick.json"),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    cfg = _build_cfg(parse_args(argv))
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

    paop_specs = make_paop_pool(
        "paop_lf_std",
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
    paop_pool = [AnsatzTerm(label=str(lbl), polynomial=poly) for lbl, poly in paop_specs]

    hva_ptw = HubbardHolsteinPhysicalTermwiseAnsatz(
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
    hva_pool = list(hva_ptw.base_terms)

    source_by_sig: dict[tuple[tuple[str, float], ...], dict[str, bool]] = {}
    hva_sigs: set[tuple[tuple[str, float], ...]] = set()
    paop_sigs: set[tuple[tuple[str, float], ...]] = set()
    for op in hva_pool:
        sig = _poly_signature(op.polynomial)
        hva_sigs.add(sig)
        source_by_sig.setdefault(sig, {"hva": False, "paop": False})["hva"] = True
    for op in paop_pool:
        sig = _poly_signature(op.polynomial)
        paop_sigs.add(sig)
        source_by_sig.setdefault(sig, {"hva": False, "paop": False})["paop"] = True

    meta_pool: list[AnsatzTerm] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for op in list(hva_pool) + list(paop_pool):
        sig = _poly_signature(op.polynomial)
        if sig in seen:
            continue
        seen.add(sig)
        meta_pool.append(op)

    grad0_paop = _top_gradients(h_poly, paop_pool, psi_ref, source_by_sig, topk=int(cfg.topk))
    grad0_meta = _top_gradients(h_poly, meta_pool, psi_ref, source_by_sig, topk=int(cfg.topk))

    paop_t0 = time.perf_counter()
    paop_run = _run_adapt(h_poly, psi_ref, paop_pool, source_by_sig, cfg)
    paop_runtime = time.perf_counter() - paop_t0
    paop_psi = np.asarray(paop_run["psi_best"], dtype=complex)
    paop_diag = _sector_diagnostics(paop_psi, cfg)

    meta_t0 = time.perf_counter()
    meta_run = _run_adapt(h_poly, psi_ref, meta_pool, source_by_sig, cfg)
    meta_runtime = time.perf_counter() - meta_t0
    meta_psi = np.asarray(meta_run["psi_best"], dtype=complex)
    meta_diag = _sector_diagnostics(meta_psi, cfg)

    paop_delta = abs(float(paop_run["E_best"]) - float(e_exact))
    meta_delta = abs(float(meta_run["E_best"]) - float(e_exact))
    improves = (paop_delta - meta_delta) >= float(cfg.improve_threshold)
    decision = "meta_better" if improves else "no_clear_meta_gain"

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
            "adapt_max_depth": int(cfg.adapt_max_depth),
            "adapt_maxiter": int(cfg.adapt_maxiter),
            "adapt_eps_grad": float(cfg.adapt_eps_grad),
            "adapt_eps_energy": float(cfg.adapt_eps_energy),
            "allow_repeats": bool(cfg.allow_repeats),
            "paop_r": int(cfg.paop_r),
            "paop_split_paulis": bool(cfg.paop_split_paulis),
            "paop_prune_eps": float(cfg.paop_prune_eps),
            "paop_normalization": str(cfg.paop_normalization),
            "improve_threshold": float(cfg.improve_threshold),
        },
        "E_exact_sector": float(e_exact),
        "pool_sizes": {
            "hva": int(len(hva_pool)),
            "paop_lf_std": int(len(paop_pool)),
            "meta_dedup": int(len(meta_pool)),
            "overlap_count": int(len(hva_sigs & paop_sigs)),
        },
        "step0_gradients": {
            "paop_topk": grad0_paop,
            "meta_topk": grad0_meta,
        },
        "runs": {
            "paop_only": {
                "E_best": float(paop_run["E_best"]),
                "E_last": float(paop_run["E_last"]),
                "delta_E_abs": float(paop_delta),
                "runtime_s": float(paop_runtime),
                "adapt_depth_reached": int(paop_run["adapt_depth_reached"]),
                "adapt_stop_reason": str(paop_run["adapt_stop_reason"]),
                "num_parameters": int(paop_run["num_parameters"]),
                "nfev_total": int(paop_run["nfev_total"]),
                "nit_total": int(paop_run["nit_total"]),
                "selected_trace": paop_run["trace"],
                "sector_diag": paop_diag,
            },
            "meta_hva_plus_paop": {
                "E_best": float(meta_run["E_best"]),
                "E_last": float(meta_run["E_last"]),
                "delta_E_abs": float(meta_delta),
                "runtime_s": float(meta_runtime),
                "adapt_depth_reached": int(meta_run["adapt_depth_reached"]),
                "adapt_stop_reason": str(meta_run["adapt_stop_reason"]),
                "num_parameters": int(meta_run["num_parameters"]),
                "nfev_total": int(meta_run["nfev_total"]),
                "nit_total": int(meta_run["nit_total"]),
                "selected_trace": meta_run["trace"],
                "sector_diag": meta_diag,
            },
        },
        "decision": {
            "label": decision,
            "delta_E_improvement": float(paop_delta - meta_delta),
            "criterion": f"meta better if ΔE improves by >= {cfg.improve_threshold:g}",
        },
        "runtime_total_s": float(time.perf_counter() - t_start),
    }

    cfg.output_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"WROTE {cfg.output_json}")
    print(
        "PAOP-only: "
        f"E_best={paop_run['E_best']:.12f} "
        f"ΔE={paop_delta:.6e} depth={paop_run['adapt_depth_reached']} "
        f"runtime_s={paop_runtime:.2f}"
    )
    print(
        "META(HVA+PAOP): "
        f"E_best={meta_run['E_best']:.12f} "
        f"ΔE={meta_delta:.6e} depth={meta_run['adapt_depth_reached']} "
        f"runtime_s={meta_runtime:.2f}"
    )
    print(
        "Decision: "
        f"{decision} "
        f"(ΔE improvement = {paop_delta - meta_delta:.6e}, threshold={cfg.improve_threshold:.6e})"
    )


if __name__ == "__main__":
    main()
