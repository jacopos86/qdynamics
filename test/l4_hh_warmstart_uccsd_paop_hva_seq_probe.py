#!/usr/bin/env python3
"""L=4 HH warm-start ADAPT sequential benchmark (A then B).

Pool arms:
  A) UCCSD (fermion-only lifted) + PAOP
  B) UCCSD (fermion-only lifted) + PAOP + HVA

Run order:
  1) Shared warm-start (HH-HVA PTW)
  2) A_probe
  3) A_medium
  4) B_probe
  5) B_medium

Primary online metric: |DeltaE| = |E - E_exact_sector|.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
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
    hubbard_holstein_reference_state,
    jw_number_operator,
    vqe_minimize,
)
from pipelines.exact_bench.benchmark_metrics_proxy import write_proxy_sidecars

try:
    from scipy.optimize import minimize as scipy_minimize
except Exception as exc:  # pragma: no cover
    raise RuntimeError("SciPy is required for this script.") from exc


class StageWallclockCap(RuntimeError):
    """Raised to abort a stage when wallclock cap is hit during optimization."""


@dataclass(frozen=True)
class StageBudget:
    name: str
    max_depth: int
    maxiter: int
    eps_grad: float
    eps_energy: float
    wallclock_cap_s: int


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
    allow_repeats: bool
    paop_key: str
    paop_r: int
    paop_split_paulis: bool
    paop_prune_eps: float
    paop_normalization: str
    seed: int
    warm_ansatz: str
    warm_reps: int
    warm_restarts: int
    warm_maxiter: int
    warm_method: str
    warm_seed: int
    probe_budget: StageBudget
    medium_budget: StageBudget
    progress_every_depth: int
    progress_every_seconds: int
    medium_delta_target: float
    probe_abs_drop_min: float
    probe_rel_drop_min: float
    output_json: Path
    output_csv: Path
    output_md: Path
    output_log: Path


class RunLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("", encoding="utf-8")

    def log(self, msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


# Built-in math:
#   dE/dtheta|_0 = i<psi|[H,G]|psi> = 2 Im(<H psi | G psi>)
def _commutator_gradient(h_poly: Any, op: AnsatzTerm, psi_current: np.ndarray) -> float:
    g_psi = _apply_pauli_polynomial(psi_current, op.polynomial)
    h_psi = _apply_pauli_polynomial(psi_current, h_poly)
    return float(2.0 * np.vdot(h_psi, g_psi).imag)


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


def _source_tag(flags: dict[str, bool]) -> str:
    on = [k for k in ("uccsd", "paop", "hva") if bool(flags.get(k))]
    if not on:
        return "unknown"
    return "+".join(on)


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


def _prepare_state(psi_ref: np.ndarray, selected_ops: list[AnsatzTerm], theta: np.ndarray) -> np.ndarray:
    psi = np.array(psi_ref, copy=True)
    for k, op in enumerate(selected_ops):
        psi = apply_exp_pauli_polynomial(psi, op.polynomial, float(theta[k]))
    return psi


def _energy(h_poly: Any, psi_ref: np.ndarray, selected_ops: list[AnsatzTerm], theta: np.ndarray) -> float:
    psi = _prepare_state(psi_ref, selected_ops, theta)
    return float(expval_pauli_polynomial(psi, h_poly))


def _sector_diagnostics(psi: np.ndarray, cfg: ProbeConfig) -> dict[str, float]:
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


def _serialize_stage_result(run: dict[str, Any]) -> dict[str, Any]:
    out = dict(run)
    out.pop("psi_best", None)
    return out


def _run_adapt_stage(
    *,
    label: str,
    h_poly: Any,
    psi_start: np.ndarray,
    pool: list[AnsatzTerm],
    source_by_sig: dict[tuple[tuple[str, float], ...], dict[str, bool]],
    e_exact: float,
    cfg: ProbeConfig,
    budget: StageBudget,
    logger: RunLogger,
    progress_events: list[dict[str, Any]],
) -> dict[str, Any]:
    stage_start = time.perf_counter()
    selected_ops: list[AnsatzTerm] = []
    theta = np.zeros(0, dtype=float)
    energy_current = float(expval_pauli_polynomial(psi_start, h_poly))
    initial_delta = abs(float(energy_current) - float(e_exact))
    best_energy = float(energy_current)

    energy_trace = [float(energy_current)]
    grad_max_trace: list[float] = []
    trace: list[dict[str, Any]] = []
    nfev_total = 1
    nit_total = 0
    stop_reason = "max_depth"
    available = set(range(len(pool)))

    logger.log(
        f"{label}: start | E0={energy_current:.12f} |DeltaE0|={initial_delta:.6e} "
        f"depth_cap={budget.max_depth} maxiter={budget.maxiter} wallclock_cap_s={budget.wallclock_cap_s}"
    )

    for depth in range(int(budget.max_depth)):
        elapsed = time.perf_counter() - stage_start
        if elapsed >= float(budget.wallclock_cap_s):
            stop_reason = "wallclock_cap"
            logger.log(f"{label}: stop wallclock before depth {depth + 1}")
            break

        if bool(cfg.allow_repeats):
            candidate_indices = list(range(len(pool)))
        else:
            candidate_indices = sorted(available)

        if not candidate_indices:
            stop_reason = "pool_exhausted"
            break

        psi_current = _prepare_state(psi_start, selected_ops, theta)
        gradients = {idx: float(_commutator_gradient(h_poly, pool[idx], psi_current)) for idx in candidate_indices}
        grad_abs = {idx: abs(val) for idx, val in gradients.items()}
        best_idx = max(candidate_indices, key=lambda i: (grad_abs[i], -i))
        max_grad = float(grad_abs[best_idx])
        grad_max_trace.append(float(max_grad))

        cur_delta = abs(float(energy_current) - float(e_exact))
        best_delta = abs(float(best_energy) - float(e_exact))

        if ((depth + 1) % max(int(cfg.progress_every_depth), 1)) == 0:
            logger.log(
                f"{label}: depth {depth + 1:03d} pre-opt |DeltaE|={cur_delta:.6e} "
                f"best|DeltaE|={best_delta:.6e} max|g|={max_grad:.6e}"
            )

        progress_events.append(
            {
                "run_id": str(label),
                "kind": "depth_preopt",
                "depth": int(depth + 1),
                "elapsed_s": float(elapsed),
                "E_current": float(energy_current),
                "delta_E_abs_current": float(cur_delta),
                "delta_E_abs_best": float(best_delta),
                "max_grad": float(max_grad),
            }
        )

        if max_grad < float(budget.eps_grad):
            stop_reason = "eps_grad"
            logger.log(f"{label}: stop eps_grad at depth {depth + 1}, max|g|={max_grad:.6e}")
            break

        selected_ops.append(pool[best_idx])
        theta = np.append(theta, 0.0)
        if not bool(cfg.allow_repeats):
            available.discard(best_idx)

        e_before = float(energy_current)
        hb_last = time.perf_counter()
        local_calls = 0
        local_best = [float("inf")]

        def objective(x: np.ndarray) -> float:
            nonlocal hb_last, local_calls
            now = time.perf_counter()
            if (now - stage_start) >= float(budget.wallclock_cap_s):
                raise StageWallclockCap("stage wallclock cap hit during local optimize")
            e_val = _energy(h_poly, psi_start, selected_ops, x)
            local_calls += 1
            if e_val < local_best[0]:
                local_best[0] = float(e_val)
            if now - hb_last >= float(max(1, int(cfg.progress_every_seconds))):
                hb_delta = abs(float(e_val) - float(e_exact))
                hb_best = abs(float(min(best_energy, local_best[0])) - float(e_exact))
                logger.log(
                    f"{label}: depth {depth + 1:03d} heartbeat calls={local_calls} "
                    f"|DeltaE|={hb_delta:.6e} best|DeltaE|={hb_best:.6e}"
                )
                progress_events.append(
                    {
                        "run_id": str(label),
                        "kind": "heartbeat",
                        "depth": int(depth + 1),
                        "elapsed_s": float(now - stage_start),
                        "delta_E_abs_current": float(hb_delta),
                        "delta_E_abs_best": float(hb_best),
                        "local_calls": int(local_calls),
                    }
                )
                hb_last = now
            return float(e_val)

        try:
            res = scipy_minimize(
                objective,
                theta,
                method="COBYLA",
                options={"maxiter": int(budget.maxiter), "rhobeg": 0.3},
            )
        except StageWallclockCap:
            selected_ops.pop()
            theta = theta[:-1]
            stop_reason = "wallclock_cap"
            logger.log(f"{label}: stop wallclock during local optimize at depth {depth + 1}")
            break

        theta = np.asarray(res.x, dtype=float)
        energy_current = float(res.fun)
        energy_trace.append(float(energy_current))
        nfev_total += int(getattr(res, "nfev", 0))
        nit_total += int(getattr(res, "nit", 0))
        best_energy = float(min(best_energy, energy_current))

        fam_source = _source_tag(source_by_sig.get(_poly_signature(pool[best_idx].polynomial), {}))
        step_delta = float(energy_current - e_before)

        trace.append(
            {
                "depth": int(depth + 1),
                "selected_idx": int(best_idx),
                "selected_label": str(pool[best_idx].label),
                "source": str(fam_source),
                "init_theta": 0.0,
                "max_abs_grad_among_candidates": float(max_grad),
                "E_before": float(e_before),
                "E_after": float(energy_current),
                "delta_E_step": float(step_delta),
            }
        )

        cur_delta = abs(float(energy_current) - float(e_exact))
        best_delta = abs(float(best_energy) - float(e_exact))

        logger.log(
            f"{label}: depth {depth + 1:03d} post-opt |DeltaE|={cur_delta:.6e} "
            f"best|DeltaE|={best_delta:.6e} stepDeltaE={step_delta:.6e} max|g|={max_grad:.6e}"
        )

        progress_events.append(
            {
                "run_id": str(label),
                "kind": "depth_postopt",
                "depth": int(depth + 1),
                "elapsed_s": float(time.perf_counter() - stage_start),
                "E_after": float(energy_current),
                "delta_E_abs_current": float(cur_delta),
                "delta_E_abs_best": float(best_delta),
                "delta_E_step": float(step_delta),
                "max_grad": float(max_grad),
                "selected_label": str(pool[best_idx].label),
            }
        )

        if abs(energy_current - e_before) < float(budget.eps_energy):
            stop_reason = "eps_energy"
            logger.log(
                f"{label}: stop eps_energy at depth {depth + 1}, "
                f"|E_after-E_before|={abs(energy_current-e_before):.6e}"
            )
            break

    psi_best = _prepare_state(psi_start, selected_ops, theta)
    psi_best = np.asarray(psi_best, dtype=complex).reshape(-1)
    psi_best = psi_best / np.linalg.norm(psi_best)

    result = {
        "ok": True,
        "label": str(label),
        "budget": {
            "name": str(budget.name),
            "max_depth": int(budget.max_depth),
            "maxiter": int(budget.maxiter),
            "eps_grad": float(budget.eps_grad),
            "eps_energy": float(budget.eps_energy),
            "wallclock_cap_s": int(budget.wallclock_cap_s),
        },
        "E_initial": float(energy_trace[0]),
        "E_best": float(np.min(np.asarray(energy_trace, dtype=float))),
        "E_last": float(energy_current),
        "delta_E_abs": float(abs(float(np.min(np.asarray(energy_trace, dtype=float))) - float(e_exact))),
        "relative_error_abs": float(abs(float(np.min(np.asarray(energy_trace, dtype=float))) - float(e_exact)) / max(abs(float(e_exact)), 1e-14)),
        "delta_E_abs_initial": float(initial_delta),
        "adapt_depth_reached": int(len(selected_ops)),
        "adapt_stop_reason": str(stop_reason),
        "num_parameters": int(theta.size),
        "nfev_total": int(nfev_total),
        "nit_total": int(nit_total),
        "runtime_s": float(time.perf_counter() - stage_start),
        "energy_trace": [float(x) for x in energy_trace],
        "grad_max_trace": [float(x) for x in grad_max_trace],
        "selected_trace": trace,
        "sector_diag": _sector_diagnostics(psi_best, cfg),
        "psi_best": psi_best.tolist(),
    }
    return result


def _assess_probe_signal(run: dict[str, Any], cfg: ProbeConfig) -> dict[str, Any]:
    if not bool(run.get("ok", False)):
        return {
            "assessable": False,
            "probe_encouraging": False,
            "reason": "run_failed",
        }
    d0 = float(run.get("delta_E_abs_initial", float("inf")))
    d1 = float(run.get("delta_E_abs", float("inf")))
    abs_drop = float(d0 - d1)
    rel_drop = float(abs_drop / max(d0, 1e-15))
    grad_trace = run.get("grad_max_trace", [])
    g_last = float(grad_trace[-1]) if grad_trace else float("inf")

    min_drop = max(float(cfg.probe_abs_drop_min), float(cfg.probe_rel_drop_min) * max(d0, 1e-15))
    encouraging = bool(abs_drop >= min_drop and g_last > 1e-8)

    return {
        "assessable": True,
        "probe_encouraging": bool(encouraging),
        "delta_E_abs_initial": float(d0),
        "delta_E_abs_final": float(d1),
        "abs_drop": float(abs_drop),
        "rel_drop": float(rel_drop),
        "final_max_grad": float(g_last),
        "min_required_drop": float(min_drop),
    }


def _safe_run_stage(
    *,
    label: str,
    h_poly: Any,
    psi_start: np.ndarray,
    pool: list[AnsatzTerm],
    source_by_sig: dict[tuple[tuple[str, float], ...], dict[str, bool]],
    e_exact: float,
    cfg: ProbeConfig,
    budget: StageBudget,
    logger: RunLogger,
    progress_events: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        return _run_adapt_stage(
            label=label,
            h_poly=h_poly,
            psi_start=psi_start,
            pool=pool,
            source_by_sig=source_by_sig,
            e_exact=e_exact,
            cfg=cfg,
            budget=budget,
            logger=logger,
            progress_events=progress_events,
        )
    except Exception as exc:
        logger.log(f"{label}: ERROR {exc}")
        return {
            "ok": False,
            "label": str(label),
            "budget": {
                "name": str(budget.name),
                "max_depth": int(budget.max_depth),
                "maxiter": int(budget.maxiter),
                "eps_grad": float(budget.eps_grad),
                "eps_energy": float(budget.eps_energy),
                "wallclock_cap_s": int(budget.wallclock_cap_s),
            },
            "error": traceback.format_exc(limit=20),
        }


def _parse_sector(text: str) -> tuple[int, int]:
    parts = [x.strip() for x in str(text).replace(" ", "").split(",") if x.strip()]
    if len(parts) != 2:
        raise ValueError("--sector must be n_up,n_dn")
    return int(parts[0]), int(parts[1])


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_summary_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    e_exact = payload.get("exact", {}).get("E_exact_sector", "")
    for run_id, run in payload.get("runs", {}).items():
        if not isinstance(run, dict):
            continue
        run_id_s = str(run_id)
        pool_name = ""
        if run_id_s.startswith("A_"):
            pool_name = "uccsd+paop"
        elif run_id_s.startswith("B_"):
            pool_name = "uccsd+paop+hva"
        budget = run.get("budget", {}) if isinstance(run.get("budget", {}), dict) else {}
        rows.append(
            {
                "run_id": run_id_s,
                "method_id": run_id_s,
                "method_kind": "adapt",
                "pool_name": pool_name,
                "pool_arm": "A" if run_id_s.startswith("A_") else ("B" if run_id_s.startswith("B_") else ""),
                "stage": str(run.get("budget", {}).get("name", "")),
                "ok": bool(run.get("ok", False)),
                "E_exact_sector": e_exact,
                "E_best": run.get("E_best", ""),
                "E_last": run.get("E_last", ""),
                "delta_E_abs": run.get("delta_E_abs", ""),
                "relative_error_abs": run.get("relative_error_abs", ""),
                "delta_E_abs_initial": run.get("delta_E_abs_initial", ""),
                "adapt_depth_reached": run.get("adapt_depth_reached", ""),
                "num_parameters": run.get("num_parameters", ""),
                "nfev_total": run.get("nfev_total", ""),
                "nit_total": run.get("nit_total", ""),
                "runtime_s": run.get("runtime_s", ""),
                "wallclock_cap_s": budget.get("wallclock_cap_s", ""),
                "maxiter": budget.get("maxiter", ""),
                "adapt_stop_reason": run.get("adapt_stop_reason", ""),
                "final_max_grad": (run.get("grad_max_trace", [])[-1] if run.get("grad_max_trace") else ""),
                "medium_gate_pass": run.get("medium_gate_pass", ""),
            }
        )
    return rows


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.6e}"
    return str(x)


def _write_summary_md(path: Path, payload: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = payload.get("config", {})
    warm = payload.get("warm_start", {})
    comp = payload.get("comparisons", {})
    lines: list[str] = []
    lines.append("# L4 HH Warm-Start Sequential Benchmark Summary")
    lines.append("")
    lines.append("## L3 to L4 Parameter Mapping")
    lines.append("")
    lines.append("| Parameter | L3 successful reference | L4 chosen in this run |")
    lines.append("|---|---|---|")
    lines.append("| boundary | open | open |")
    lines.append("| ordering | blocked | blocked |")
    lines.append("| boson encoding | binary | binary |")
    lines.append("| n_ph_max | 1 | 1 |")
    lines.append("| sector | (2,1) at L=3 | (2,2) at L=4 |")
    lines.append("| warm reps | 3 | " + str(cfg.get("warm_reps")) + " |")
    lines.append("| warm restarts | 5/6 | " + str(cfg.get("warm_restarts")) + " |")
    lines.append("| warm maxiter | 4000/6000 | " + str(cfg.get("warm_maxiter")) + " |")
    lines.append("| probe depth/maxiter | 20/1200 (trend style) | " + f"{cfg.get('probe_depth')}/{cfg.get('probe_maxiter')}" + " |")
    lines.append("| medium depth/maxiter | L3 B/C style heavier | " + f"{cfg.get('medium_depth')}/{cfg.get('medium_maxiter')}" + " |")
    lines.append("")
    lines.append("## Warm-Start Stage")
    lines.append("")
    lines.append("- Shared warm-start used for both pool arms.")
    lines.append("- E_warm: `" + _fmt(warm.get("E_warm", "")) + "`")
    lines.append("- |DeltaE|_warm: `" + _fmt(warm.get("delta_E_abs", "")) + "`")
    lines.append("- rel_err_warm: `" + _fmt(warm.get("relative_error_abs", "")) + "`")
    lines.append("- warm work: nfev=`" + _fmt(warm.get("warm_nfev", "")) + "`, nit=`" + _fmt(warm.get("warm_nit", "")) + "`")
    lines.append("")
    lines.append("## Run Table")
    lines.append("")
    lines.append("| run_id | ok | |DeltaE| | rel_err | depth | nfev | runtime_s | stop | medium_gate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---:|")
    for row in rows:
        lines.append(
            "| "
            + str(row["run_id"])
            + " | "
            + str(row["ok"])
            + " | "
            + _fmt(row["delta_E_abs"])
            + " | "
            + _fmt(row["relative_error_abs"])
            + " | "
            + _fmt(row["adapt_depth_reached"])
            + " | "
            + _fmt(row["nfev_total"])
            + " | "
            + _fmt(row["runtime_s"])
            + " | "
            + str(row["adapt_stop_reason"])
            + " | "
            + str(row["medium_gate_pass"])
            + " |"
        )

    lines.append("")
    lines.append("## Comparisons")
    lines.append("")
    lines.append("- medium target: |DeltaE| <= " + _fmt(cfg.get("medium_delta_target")))
    lines.append("- A_probe encouraging: `" + str(comp.get("probe_assessment", {}).get("A_probe", {}).get("probe_encouraging")) + "`")
    lines.append("- B_probe encouraging: `" + str(comp.get("probe_assessment", {}).get("B_probe", {}).get("probe_encouraging")) + "`")
    lines.append("- medium winner by |DeltaE|: `" + str(comp.get("medium_winner", {}).get("run_id", "none")) + "`")
    lines.append("- A_vs_B medium delta difference (A-B): `" + _fmt(comp.get("A_vs_B_medium", {}).get("delta_diff_A_minus_B", "")) + "`")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_cfg(args: argparse.Namespace) -> ProbeConfig:
    n_up, n_dn = _parse_sector(str(args.sector))
    probe_budget = StageBudget(
        name="probe",
        max_depth=int(args.probe_depth),
        maxiter=int(args.probe_maxiter),
        eps_grad=float(args.probe_eps_grad),
        eps_energy=float(args.probe_eps_energy),
        wallclock_cap_s=int(args.probe_wallclock_cap_s),
    )
    medium_budget = StageBudget(
        name="medium",
        max_depth=int(args.medium_depth),
        maxiter=int(args.medium_maxiter),
        eps_grad=float(args.medium_eps_grad),
        eps_energy=float(args.medium_eps_energy),
        wallclock_cap_s=int(args.medium_wallclock_cap_s),
    )
    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv) if args.output_csv else Path(str(out_json).replace(".json", "_summary.csv"))
    out_md = Path(args.output_md) if args.output_md else Path(str(out_json).replace(".json", "_summary.md"))
    out_log = Path(args.output_log) if args.output_log else Path(str(out_json).replace(".json", ".log"))

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
        allow_repeats=bool(args.allow_repeats),
        paop_key=str(args.paop_key),
        paop_r=int(args.paop_r),
        paop_split_paulis=bool(args.paop_split_paulis),
        paop_prune_eps=float(args.paop_prune_eps),
        paop_normalization=str(args.paop_normalization),
        seed=int(args.seed),
        warm_ansatz=str(args.warm_ansatz),
        warm_reps=int(args.warm_reps),
        warm_restarts=int(args.warm_restarts),
        warm_maxiter=int(args.warm_maxiter),
        warm_method=str(args.warm_method),
        warm_seed=int(args.warm_seed),
        probe_budget=probe_budget,
        medium_budget=medium_budget,
        progress_every_depth=int(args.progress_every_depth),
        progress_every_seconds=int(args.progress_every_seconds),
        medium_delta_target=float(args.medium_delta_target),
        probe_abs_drop_min=float(args.probe_abs_drop_min),
        probe_rel_drop_min=float(args.probe_rel_drop_min),
        output_json=out_json,
        output_csv=out_csv,
        output_md=out_md,
        output_log=out_log,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="L=4 HH warm-start ADAPT sequential benchmark: A(UCCSD+PAOP) then B(UCCSD+PAOP+HVA)."
    )
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--U", type=float, default=4.0)
    p.add_argument("--dv", type=float, default=0.0)
    p.add_argument("--omega0", type=float, default=1.0)
    p.add_argument("--g-ep", type=float, default=0.5, dest="g_ep")
    p.add_argument("--n-ph-max", type=int, default=1, dest="n_ph_max")
    p.add_argument("--boson-encoding", type=str, default="binary", choices=["binary", "unary"], dest="boson_encoding")
    p.add_argument("--ordering", type=str, default="blocked", choices=["blocked", "interleaved"])
    p.add_argument("--boundary", type=str, default="open", choices=["open", "periodic"])
    p.add_argument("--sector", type=str, default="2,2")

    p.add_argument("--allow-repeats", action="store_true")
    p.add_argument("--no-allow-repeats", dest="allow_repeats", action="store_false")
    p.set_defaults(allow_repeats=True)

    p.add_argument("--paop-key", type=str, default="paop_lf_std")
    p.add_argument("--paop-r", type=int, default=1)
    p.add_argument("--paop-split-paulis", action="store_true")
    p.add_argument("--paop-prune-eps", type=float, default=0.0)
    p.add_argument("--paop-normalization", type=str, default="none", choices=["none", "fro", "maxcoeff"])

    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--warm-ansatz", type=str, default="hh_hva_ptw", choices=["hh_hva", "hh_hva_tw", "hh_hva_ptw"])
    p.add_argument("--warm-reps", type=int, default=4)
    p.add_argument("--warm-restarts", type=int, default=6)
    p.add_argument("--warm-maxiter", type=int, default=6000)
    p.add_argument("--warm-method", type=str, default="COBYLA", choices=["COBYLA", "SLSQP", "L-BFGS-B", "Powell", "Nelder-Mead"])
    p.add_argument("--warm-seed", type=int, default=7)

    p.add_argument("--probe-depth", type=int, default=20)
    p.add_argument("--probe-maxiter", type=int, default=1200)
    p.add_argument("--probe-eps-grad", type=float, default=1e-6)
    p.add_argument("--probe-eps-energy", type=float, default=1e-8)
    p.add_argument("--probe-wallclock-cap-s", type=int, default=900)

    p.add_argument("--medium-depth", type=int, default=48)
    p.add_argument("--medium-maxiter", type=int, default=3600)
    p.add_argument("--medium-eps-grad", type=float, default=5e-7)
    p.add_argument("--medium-eps-energy", type=float, default=1e-9)
    p.add_argument("--medium-wallclock-cap-s", type=int, default=1800)
    p.add_argument("--medium-delta-target", type=float, default=1e-3)

    p.add_argument("--probe-abs-drop-min", type=float, default=1e-3)
    p.add_argument("--probe-rel-drop-min", type=float, default=0.10)

    p.add_argument("--progress-every-depth", type=int, default=1)
    p.add_argument("--progress-every-seconds", type=int, default=15)

    p.add_argument(
        "--output-json",
        type=str,
        default=str(REPO_ROOT / "artifacts" / "useful" / "L4" / "l4_hh_warmstart_uccsd_paop_hva_seq.json"),
    )
    p.add_argument("--output-csv", type=str, default="")
    p.add_argument("--output-md", type=str, default="")
    p.add_argument("--output-log", type=str, default="")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    cfg = _build_cfg(parse_args(argv))
    logger = RunLogger(cfg.output_log)

    np.random.seed(int(cfg.seed))
    t_global = time.perf_counter()

    if int(cfg.L) != 4:
        raise ValueError("This sequential benchmark is intended for L=4.")
    if str(cfg.boundary).strip().lower() != "open":
        raise ValueError("This benchmark is configured for open boundary conditions.")

    logger.log(
        "CONFIG "
        f"L={cfg.L} sector=({cfg.sector_n_up},{cfg.sector_n_dn}) "
        f"encoding={cfg.boson_encoding} ordering={cfg.ordering} boundary={cfg.boundary} "
        f"poolA=UCCSD+PAOP poolB=UCCSD+PAOP+HVA warm={cfg.warm_ansatz}"
    )

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
    logger.log(f"EXACT sector energy: {e_exact:.12f}")

    # Shared warm-start stage.
    t_warm = time.perf_counter()
    warm_ansatz = HubbardHolsteinPhysicalTermwiseAnsatz(
        dims=int(cfg.L),
        J=float(cfg.t),
        U=float(cfg.U),
        omega0=float(cfg.omega0),
        g=float(cfg.g_ep),
        n_ph_max=int(cfg.n_ph_max),
        boson_encoding=str(cfg.boson_encoding),
        v=float(cfg.dv),
        reps=int(cfg.warm_reps),
        repr_mode="JW",
        indexing=str(cfg.ordering),
        pbc=(str(cfg.boundary).strip().lower() == "periodic"),
    )
    warm_res = vqe_minimize(
        h_poly,
        warm_ansatz,
        psi_ref,
        restarts=int(cfg.warm_restarts),
        seed=int(cfg.warm_seed),
        maxiter=int(cfg.warm_maxiter),
        method=str(cfg.warm_method),
    )
    psi_warm = np.asarray(warm_ansatz.prepare_state(np.asarray(warm_res.theta, dtype=float), psi_ref), dtype=complex).reshape(-1)
    psi_warm = psi_warm / np.linalg.norm(psi_warm)
    e_warm = float(warm_res.energy)
    d_warm = abs(e_warm - e_exact)
    rel_warm = float(d_warm / max(abs(e_exact), 1e-14))
    logger.log(
        f"WARM done: E={e_warm:.12f} |DeltaE|={d_warm:.6e} rel={rel_warm:.6e} "
        f"reps={cfg.warm_reps} restarts={cfg.warm_restarts} maxiter={cfg.warm_maxiter} "
        f"nfev={warm_res.nfev} nit={warm_res.nit} runtime_s={time.perf_counter()-t_warm:.2f}"
    )

    # Build pools.
    boson_bits = int(cfg.L) * int(boson_qubits_per_site(int(cfg.n_ph_max), str(cfg.boson_encoding)))
    nq_total = 2 * int(cfg.L) + boson_bits

    uccsd_pool = _build_uccsd_ferm_only_lifted_pool(cfg)
    _assert_uccsd_fermion_only(uccsd_pool, boson_bits, nq_total)
    paop_pool = _build_paop_pool(cfg)
    hva_pool = _build_hva_pool(cfg)

    pool_A, source_A, meta_A = _make_dedup_pool([("uccsd", uccsd_pool), ("paop", paop_pool)])
    pool_B, source_B, meta_B = _make_dedup_pool([("uccsd", uccsd_pool), ("paop", paop_pool), ("hva", hva_pool)])

    logger.log(
        f"POOLS raw_sizes: uccsd={len(uccsd_pool)} paop={len(paop_pool)} hva={len(hva_pool)} | "
        f"A_dedup={len(pool_A)} B_dedup={len(pool_B)}"
    )

    progress_events: list[dict[str, Any]] = []

    payload: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
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
            "allow_repeats": bool(cfg.allow_repeats),
            "paop_key": str(cfg.paop_key),
            "paop_r": int(cfg.paop_r),
            "paop_split_paulis": bool(cfg.paop_split_paulis),
            "paop_prune_eps": float(cfg.paop_prune_eps),
            "paop_normalization": str(cfg.paop_normalization),
            "seed": int(cfg.seed),
            "warm_ansatz": str(cfg.warm_ansatz),
            "warm_reps": int(cfg.warm_reps),
            "warm_restarts": int(cfg.warm_restarts),
            "warm_maxiter": int(cfg.warm_maxiter),
            "warm_method": str(cfg.warm_method),
            "warm_seed": int(cfg.warm_seed),
            "probe_depth": int(cfg.probe_budget.max_depth),
            "probe_maxiter": int(cfg.probe_budget.maxiter),
            "probe_eps_grad": float(cfg.probe_budget.eps_grad),
            "probe_eps_energy": float(cfg.probe_budget.eps_energy),
            "probe_wallclock_cap_s": int(cfg.probe_budget.wallclock_cap_s),
            "medium_depth": int(cfg.medium_budget.max_depth),
            "medium_maxiter": int(cfg.medium_budget.maxiter),
            "medium_eps_grad": float(cfg.medium_budget.eps_grad),
            "medium_eps_energy": float(cfg.medium_budget.eps_energy),
            "medium_wallclock_cap_s": int(cfg.medium_budget.wallclock_cap_s),
            "medium_delta_target": float(cfg.medium_delta_target),
            "probe_abs_drop_min": float(cfg.probe_abs_drop_min),
            "probe_rel_drop_min": float(cfg.probe_rel_drop_min),
            "progress_every_depth": int(cfg.progress_every_depth),
            "progress_every_seconds": int(cfg.progress_every_seconds),
            "l3_reference_profile": {
                "L": 3,
                "boundary": "open",
                "ordering": "blocked",
                "boson_encoding": "binary",
                "n_ph_max": 1,
                "sector": [2, 1],
                "warm_reps": [3, 3],
                "warm_restarts": [5, 6],
                "warm_maxiter": [4000, 6000],
                "adapt_depth_caps": [120, 160],
                "adapt_maxiter": [5000, 8000],
                "adapt_wallclock_cap_s": 1200,
            },
        },
        "exact": {
            "E_exact_sector": float(e_exact),
            "nq_total": int(nq_total),
            "fermion_qubits": int(2 * cfg.L),
            "boson_qubits": int(boson_bits),
        },
        "warm_start": {
            "ansatz": str(cfg.warm_ansatz),
            "reps": int(cfg.warm_reps),
            "restarts": int(cfg.warm_restarts),
            "maxiter": int(cfg.warm_maxiter),
            "method": str(cfg.warm_method),
            "seed": int(cfg.warm_seed),
            "E_warm": float(e_warm),
            "delta_E_abs": float(d_warm),
            "relative_error_abs": float(rel_warm),
            "warm_nfev": int(warm_res.nfev),
            "warm_nit": int(warm_res.nit),
            "warm_num_parameters": int(warm_ansatz.num_parameters),
            "runtime_s": float(time.perf_counter() - t_warm),
        },
        "pool_components_raw": {
            "uccsd_ferm_only_lifted": int(len(uccsd_pool)),
            "paop": int(len(paop_pool)),
            "hva": int(len(hva_pool)),
        },
        "pool_A_meta": meta_A,
        "pool_B_meta": meta_B,
        "runs": {},
        "progress_events": progress_events,
    }

    def persist_checkpoint() -> None:
        payload["runtime_total_s"] = float(time.perf_counter() - t_global)
        _json_dump(cfg.output_json, payload)

    persist_checkpoint()

    # Sequential order requested: A first, then B.
    run_order = [
        ("A_probe", pool_A, source_A, cfg.probe_budget),
        ("A_medium", pool_A, source_A, cfg.medium_budget),
        ("B_probe", pool_B, source_B, cfg.probe_budget),
        ("B_medium", pool_B, source_B, cfg.medium_budget),
    ]

    for run_id, pool, src, budget in run_order:
        logger.log(f"RUN_START {run_id}")
        run = _safe_run_stage(
            label=str(run_id),
            h_poly=h_poly,
            psi_start=psi_warm,
            pool=pool,
            source_by_sig=src,
            e_exact=e_exact,
            cfg=cfg,
            budget=budget,
            logger=logger,
            progress_events=progress_events,
        )

        if bool(run.get("ok", False)) and str(budget.name) == "medium":
            run["medium_gate_pass"] = bool(float(run["delta_E_abs"]) <= float(cfg.medium_delta_target))
        else:
            run["medium_gate_pass"] = None

        payload["runs"][str(run_id)] = _serialize_stage_result(run)

        if bool(run.get("ok", False)):
            logger.log(
                f"RUN_DONE {run_id}: E_best={run['E_best']:.12f} |DeltaE|={run['delta_E_abs']:.6e} "
                f"rel={run['relative_error_abs']:.6e} depth={run['adapt_depth_reached']} "
                f"stop={run['adapt_stop_reason']} nfev={run['nfev_total']} runtime_s={run['runtime_s']:.2f}"
            )
        else:
            logger.log(f"RUN_FAIL {run_id}: {run.get('error', 'unknown error')}")

        persist_checkpoint()

    # Build comparisons.
    probe_assessment = {
        "A_probe": _assess_probe_signal(payload["runs"].get("A_probe", {}), cfg),
        "B_probe": _assess_probe_signal(payload["runs"].get("B_probe", {}), cfg),
    }

    def _delta(run_id: str) -> float | None:
        r = payload["runs"].get(run_id, {})
        if not isinstance(r, dict) or not bool(r.get("ok", False)):
            return None
        return float(r.get("delta_E_abs"))

    a_probe_d = _delta("A_probe")
    b_probe_d = _delta("B_probe")
    a_medium_d = _delta("A_medium")
    b_medium_d = _delta("B_medium")

    medium_candidates = []
    for rid in ("A_medium", "B_medium"):
        d = _delta(rid)
        if d is not None:
            medium_candidates.append((rid, d))
    medium_winner = min(medium_candidates, key=lambda x: x[1]) if medium_candidates else (None, None)

    payload["comparisons"] = {
        "probe_assessment": probe_assessment,
        "A_vs_B_probe": {
            "A_delta_E_abs": a_probe_d,
            "B_delta_E_abs": b_probe_d,
            "delta_diff_A_minus_B": (None if (a_probe_d is None or b_probe_d is None) else float(a_probe_d - b_probe_d)),
        },
        "A_vs_B_medium": {
            "A_delta_E_abs": a_medium_d,
            "B_delta_E_abs": b_medium_d,
            "delta_diff_A_minus_B": (None if (a_medium_d is None or b_medium_d is None) else float(a_medium_d - b_medium_d)),
        },
        "medium_winner": {
            "run_id": medium_winner[0],
            "delta_E_abs": medium_winner[1],
        },
        "medium_target_delta_abs": float(cfg.medium_delta_target),
    }

    payload["gates"] = {
        "medium_target_delta_abs": float(cfg.medium_delta_target),
        "A_medium_pass": bool(payload["runs"].get("A_medium", {}).get("medium_gate_pass") is True),
        "B_medium_pass": bool(payload["runs"].get("B_medium", {}).get("medium_gate_pass") is True),
    }

    payload["runtime_total_s"] = float(time.perf_counter() - t_global)
    _json_dump(cfg.output_json, payload)

    rows = _build_summary_rows(payload)
    _write_summary_csv(cfg.output_csv, rows)
    _write_summary_md(cfg.output_md, payload, rows)
    source_comp = {
        "A_uccsd_plus_paop": (
            payload.get("pool_A_meta", {}).get("dedup_source_presence_counts", {})
            if isinstance(payload.get("pool_A_meta", {}), dict)
            else {}
        ),
        "B_uccsd_plus_paop_plus_hva": (
            payload.get("pool_B_meta", {}).get("dedup_source_presence_counts", {})
            if isinstance(payload.get("pool_B_meta", {}), dict)
            else {}
        ),
    }
    sidecars = write_proxy_sidecars(
        rows,
        cfg.output_json.parent,
        csv_name=f"{cfg.output_json.stem}_metrics_proxy.csv",
        jsonl_name=f"{cfg.output_json.stem}_metrics_proxy.jsonl",
        summary_name=f"{cfg.output_json.stem}_metrics_proxy.json",
        summary_extras={"source_composition_proxy": source_comp},
        defaults={"problem": "hh", "L": int(cfg.L)},
    )

    logger.log(f"WROTE JSON {cfg.output_json}")
    logger.log(f"WROTE CSV  {cfg.output_csv}")
    logger.log(f"WROTE MD   {cfg.output_md}")
    logger.log(f"WROTE METRICS JSON {sidecars['summary_json']}")
    logger.log(f"WROTE METRICS CSV  {sidecars['csv']}")


if __name__ == "__main__":
    main()
