#!/usr/bin/env python3
"""Benchmark legacy vs compiled energy + ADAPT gradient scoring (HH case).

This is a non-pytest utility script. It benchmarks:
1) Legacy energy expectation: ``expval_pauli_polynomial(psi, H)``
2) Compiled one-apply energy: ``Re(<psi|Hpsi>)`` with one compiled apply
3) Legacy gradient scoring cost path (no Hpsi reuse)
4) New gradient scoring path (reuse Hpsi once across pool)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — this file lives at pipelines/hardcoded/bench_compiled_energy_and_grad.py
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.compiled_polynomial import (
    CompiledPolynomialAction,
    adapt_commutator_grad_from_hpsi,
    apply_compiled_polynomial,
    compile_polynomial_action,
    energy_via_one_apply,
)
from src.quantum.hubbard_latex_python_pairs import build_hubbard_holstein_hamiltonian
from src.quantum.operator_pools import make_pool
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, expval_pauli_polynomial, half_filled_num_particles

try:
    from pipelines.hardcoded.adapt_pipeline import _commutator_gradient as _legacy_commutator_gradient
except Exception:
    _legacy_commutator_gradient = None


def _ai_log(event: str, **fields: Any) -> None:
    payload = {
        "event": str(event),
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    print(f"AI_LOG {json.dumps(payload, sort_keys=True, default=str)}", flush=True)


@dataclass(frozen=True)
class BenchConfig:
    L: int
    n_ph_max: int
    t: float
    u: float
    omega0: float
    g_ep: float
    dv: float
    ordering: str
    boundary: str
    boson_encoding: str
    pool_key: str
    pool_cap: int
    paop_r: int
    paop_split_paulis: bool
    paop_prune_eps: float
    paop_normalization: str
    repeats: int
    seed: int


@dataclass(frozen=True)
class BenchCase:
    h_poly: Any
    psi: np.ndarray
    pool_terms: tuple[AnsatzTerm, ...]
    nq: int
    dim: int


@dataclass(frozen=True)
class TimingStat:
    mean_s: float
    min_s: float


_MATH_RANDOM_STATE = r"|\psi\rangle = \frac{r + i\,s}{\|r + i\,s\|_2}"


def _build_case(cfg: BenchConfig) -> BenchCase:
    h_poly = build_hubbard_holstein_hamiltonian(
        dims=int(cfg.L),
        J=float(cfg.t),
        U=float(cfg.u),
        omega0=float(cfg.omega0),
        g=float(cfg.g_ep),
        n_ph_max=int(cfg.n_ph_max),
        boson_encoding=str(cfg.boson_encoding),
        v_t=None,
        v0=float(cfg.dv),
        t_eval=None,
        repr_mode="JW",
        indexing=str(cfg.ordering),
        pbc=(str(cfg.boundary).strip().lower() == "periodic"),
        include_zero_point=True,
    )
    h_terms = h_poly.return_polynomial()
    if not h_terms:
        raise ValueError("HH Hamiltonian unexpectedly empty.")
    nq = int(h_terms[0].nqubit())
    dim = 1 << nq

    rng = np.random.default_rng(int(cfg.seed))
    psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    psi = psi / np.linalg.norm(psi)

    num_particles = tuple(half_filled_num_particles(int(cfg.L)))
    pool_specs = make_pool(
        str(cfg.pool_key),
        num_sites=int(cfg.L),
        num_particles=num_particles,
        n_ph_max=int(cfg.n_ph_max),
        boson_encoding=str(cfg.boson_encoding),
        ordering=str(cfg.ordering),
        boundary=str(cfg.boundary),
        paop_r=int(cfg.paop_r),
        paop_split_paulis=bool(cfg.paop_split_paulis),
        paop_prune_eps=float(cfg.paop_prune_eps),
        paop_normalization=str(cfg.paop_normalization),
    )
    if int(cfg.pool_cap) > 0 and len(pool_specs) > int(cfg.pool_cap):
        pool_specs = pool_specs[: int(cfg.pool_cap)]
    pool_terms = tuple(AnsatzTerm(label=str(lbl), polynomial=poly) for lbl, poly in pool_specs)
    if not pool_terms:
        raise ValueError("PAOP pool unexpectedly empty after trimming.")

    return BenchCase(
        h_poly=h_poly,
        psi=np.asarray(psi, dtype=complex),
        pool_terms=pool_terms,
        nq=nq,
        dim=dim,
    )


_MATH_ENERGY_LEGACY = r"E_{\mathrm{legacy}}=\sum_j h_j\langle\psi|P_j|\psi\rangle"


def _energy_legacy(psi: np.ndarray, h_poly: Any) -> float:
    return float(expval_pauli_polynomial(psi, h_poly))


_MATH_ENERGY_COMPILED = r"H|\psi\rangle\ \text{once},\ E=\operatorname{Re}\langle\psi|H|\psi\rangle"


def _energy_compiled(psi: np.ndarray, compiled_h: CompiledPolynomialAction) -> float:
    energy, _ = energy_via_one_apply(psi, compiled_h)
    return float(energy)


_MATH_GRAD_LEGACY = r"g_k=2\,\operatorname{Im}\langle H\psi_k|A_k\psi\rangle,\ H\psi_k\ \text{recomputed per}\ k"


def _gradients_legacy(
    psi: np.ndarray,
    h_poly: Any,
    compiled_h: CompiledPolynomialAction,
    pool_terms: tuple[AnsatzTerm, ...],
    compiled_pool: tuple[CompiledPolynomialAction, ...],
) -> np.ndarray:
    grads = np.zeros(len(pool_terms), dtype=float)
    if _legacy_commutator_gradient is not None:
        for idx, op in enumerate(pool_terms):
            grads[idx] = float(
                _legacy_commutator_gradient(
                    h_poly,
                    op,
                    psi,
                    h_compiled=compiled_h,
                    pool_compiled=compiled_pool[idx],
                    hpsi_precomputed=None,
                )
            )
        return grads

    for idx, comp_op in enumerate(compiled_pool):
        hpsi_each = apply_compiled_polynomial(psi, compiled_h)
        apsi = apply_compiled_polynomial(psi, comp_op)
        grads[idx] = adapt_commutator_grad_from_hpsi(hpsi_each, apsi)
    return grads


_MATH_GRAD_REUSE = r"g_k=2\,\operatorname{Im}\langle H\psi|A_k\psi\rangle,\ H\psi\ \text{computed once}"


def _gradients_hpsi_reuse(
    psi: np.ndarray,
    compiled_h: CompiledPolynomialAction,
    compiled_pool: tuple[CompiledPolynomialAction, ...],
) -> np.ndarray:
    hpsi = apply_compiled_polynomial(psi, compiled_h)
    grads = np.zeros(len(compiled_pool), dtype=float)
    for idx, comp_op in enumerate(compiled_pool):
        apsi = apply_compiled_polynomial(psi, comp_op)
        grads[idx] = adapt_commutator_grad_from_hpsi(hpsi, apsi)
    return grads


def _time_repeated(fn: Callable[[], Any], repeats: int) -> tuple[Any, TimingStat]:
    if int(repeats) < 1:
        raise ValueError("repeats must be >= 1")
    elapsed: list[float] = []
    out: Any = None
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        out = fn()
        elapsed.append(float(time.perf_counter() - t0))
    return out, TimingStat(mean_s=float(np.mean(elapsed)), min_s=float(np.min(elapsed)))


def _format_ms(value_s: float) -> str:
    return f"{1e3 * float(value_s):9.3f}"


def _speedup(a_s: float, b_s: float) -> float:
    denom = max(float(b_s), 1e-15)
    return float(a_s) / denom


def _parse_args() -> BenchConfig:
    p = argparse.ArgumentParser(description="Benchmark compiled HH energy + ADAPT gradients.")
    p.add_argument("--L", type=int, default=3)
    p.add_argument("--n-ph-max", type=int, default=1)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--u", type=float, default=4.0)
    p.add_argument("--omega0", type=float, default=1.0)
    p.add_argument("--g-ep", type=float, default=0.5)
    p.add_argument("--dv", type=float, default=0.0)
    p.add_argument("--ordering", type=str, default="blocked")
    p.add_argument("--boundary", choices=["open", "periodic"], default="periodic")
    p.add_argument("--boson-encoding", choices=["binary"], default="binary")
    p.add_argument("--pool-key", type=str, default="paop_lf_full")
    p.add_argument("--pool-cap", type=int, default=120)
    p.add_argument("--paop-r", type=int, default=0)
    p.add_argument("--paop-split-paulis", action="store_true", default=True)
    p.add_argument("--no-paop-split-paulis", dest="paop_split_paulis", action="store_false")
    p.add_argument("--paop-prune-eps", type=float, default=0.0)
    p.add_argument("--paop-normalization", choices=["none", "fro", "maxcoeff"], default="none")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()
    return BenchConfig(
        L=int(args.L),
        n_ph_max=int(args.n_ph_max),
        t=float(args.t),
        u=float(args.u),
        omega0=float(args.omega0),
        g_ep=float(args.g_ep),
        dv=float(args.dv),
        ordering=str(args.ordering),
        boundary=str(args.boundary),
        boson_encoding=str(args.boson_encoding),
        pool_key=str(args.pool_key),
        pool_cap=int(args.pool_cap),
        paop_r=int(args.paop_r),
        paop_split_paulis=bool(args.paop_split_paulis),
        paop_prune_eps=float(args.paop_prune_eps),
        paop_normalization=str(args.paop_normalization),
        repeats=int(args.repeats),
        seed=int(args.seed),
    )


def main() -> None:
    cfg = _parse_args()
    _ai_log("bench_compiled_energy_grad_start", config=vars(cfg))

    case = _build_case(cfg)
    pauli_action_cache: dict[str, Any] = {}
    compile_t0 = time.perf_counter()
    compiled_h = compile_polynomial_action(case.h_poly, pauli_action_cache=pauli_action_cache)
    compiled_pool = tuple(
        compile_polynomial_action(op.polynomial, pauli_action_cache=pauli_action_cache)
        for op in case.pool_terms
    )
    compile_elapsed_s = float(time.perf_counter() - compile_t0)

    energy_legacy, t_energy_legacy = _time_repeated(
        lambda: _energy_legacy(case.psi, case.h_poly),
        repeats=cfg.repeats,
    )
    energy_comp, t_energy_comp = _time_repeated(
        lambda: _energy_compiled(case.psi, compiled_h),
        repeats=cfg.repeats,
    )

    grads_legacy, t_grad_legacy = _time_repeated(
        lambda: _gradients_legacy(case.psi, case.h_poly, compiled_h, case.pool_terms, compiled_pool),
        repeats=cfg.repeats,
    )
    grads_reuse, t_grad_reuse = _time_repeated(
        lambda: _gradients_hpsi_reuse(case.psi, compiled_h, compiled_pool),
        repeats=cfg.repeats,
    )

    energy_delta = abs(float(energy_legacy) - float(energy_comp))
    grad_delta = float(np.max(np.abs(np.asarray(grads_legacy) - np.asarray(grads_reuse))))
    grad_mode = "legacy_helper" if _legacy_commutator_gradient is not None else "fallback_recompute_hpsi"

    print("\nBenchmark: compiled energy + ADAPT gradient reuse (HH)")
    print("-" * 80)
    print(
        f"L={cfg.L} n_ph_max={cfg.n_ph_max} nq={case.nq} dim={case.dim} "
        f"pool={len(case.pool_terms)} repeats={cfg.repeats} seed={cfg.seed}"
    )
    print(
        f"pool_key={cfg.pool_key} ordering={cfg.ordering} boundary={cfg.boundary} "
        f"split_paulis={cfg.paop_split_paulis} grad_legacy_mode={grad_mode}"
    )
    print(
        f"compile_shared_cache_ms={_format_ms(compile_elapsed_s)} "
        f"unique_pauli_actions={len(pauli_action_cache)}"
    )
    print("")
    print(f"{'Workload':35s} {'Legacy mean ms':>15s} {'Compiled mean ms':>17s} {'Speedup':>9s}")
    print("-" * 80)
    print(
        f"{'Energy expectation':35s} {_format_ms(t_energy_legacy.mean_s):>15s} "
        f"{_format_ms(t_energy_comp.mean_s):>17s} {(_speedup(t_energy_legacy.mean_s, t_energy_comp.mean_s)):>8.3f}x"
    )
    print(
        f"{'Gradient scoring (pool)':35s} {_format_ms(t_grad_legacy.mean_s):>15s} "
        f"{_format_ms(t_grad_reuse.mean_s):>17s} {(_speedup(t_grad_legacy.mean_s, t_grad_reuse.mean_s)):>8.3f}x"
    )
    print("-" * 80)
    print(
        f"parity: abs_delta_energy={energy_delta:.3e}, "
        f"max_abs_delta_grad={grad_delta:.3e}"
    )

    _ai_log(
        "bench_compiled_energy_grad_done",
        L=int(cfg.L),
        n_ph_max=int(cfg.n_ph_max),
        nq=int(case.nq),
        dim=int(case.dim),
        pool_size=int(len(case.pool_terms)),
        repeats=int(cfg.repeats),
        compile_elapsed_s=float(compile_elapsed_s),
        energy_legacy_mean_s=float(t_energy_legacy.mean_s),
        energy_compiled_mean_s=float(t_energy_comp.mean_s),
        energy_speedup=float(_speedup(t_energy_legacy.mean_s, t_energy_comp.mean_s)),
        grad_legacy_mean_s=float(t_grad_legacy.mean_s),
        grad_reuse_mean_s=float(t_grad_reuse.mean_s),
        grad_speedup=float(_speedup(t_grad_legacy.mean_s, t_grad_reuse.mean_s)),
        abs_delta_energy=float(energy_delta),
        max_abs_delta_grad=float(grad_delta),
        grad_legacy_mode=grad_mode,
        unique_pauli_actions=int(len(pauli_action_cache)),
    )


if __name__ == "__main__":
    main()
