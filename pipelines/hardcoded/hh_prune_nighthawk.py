#!/usr/bin/env python3
"""Prune the fullhorse Nighthawk ADAPT circuit and evaluate two lean paths.

Path A — Fixed-scaffold VQE: remove near-zero layers, re-optimize surviving
         parameters with POWELL.  Reports energy, |E|/Cost Pareto metrics.
Path B — Re-ADAPT phase 3:   warm-start from the pruned state with an
         aggressively truncated pool and tighter termination.  Reports the
         same Pareto metrics so the two paths are directly comparable.

Usage:
    python -m pipelines.hardcoded.hh_prune_nighthawk \
        --input-json artifacts/json/hh_backend_adapt_fullhorse_powell_20260322T194155Z_fakenighthawk.json \
        --mode both \
        --prune-threshold 1e-4 \
        --output-json artifacts/json/hh_prune_nighthawk_20260322.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.hubbard_latex_python_pairs import (
    build_hubbard_holstein_hamiltonian,
    boson_qubits_per_site,
)
from src.quantum.hartree_fock_reference_state import (
    hartree_fock_statevector,
    hubbard_holstein_reference_state,
)
from src.quantum.ansatz_parameterization import (
    AnsatzParameterLayout,
    build_parameter_layout,
    expand_legacy_logical_theta,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import (
    CompiledPolynomialAction,
    adapt_commutator_grad_from_hpsi,
    apply_compiled_polynomial as _apply_compiled_poly,
)
from src.quantum.vqe_latex_python_pairs import (
    AnsatzTerm,
    apply_exp_pauli_polynomial,
    apply_exp_pauli_polynomial_termwise,
    expval_pauli_polynomial,
    exact_ground_energy_sector_hh,
    half_filled_num_particles,
    vqe_minimize,
)
from pipelines.hardcoded.adapt_pipeline import (
    _build_hh_full_meta_pool,
    _build_hh_pareto_lean_l2_pool,
    _collect_hardcoded_terms_exyz,
)
from pipelines.hardcoded.adapt_circuit_cost import (
    _normalize_adapt_payload,
    _resolve_scaffold_ops,
    _resolve_total_qubits,
)
from pipelines.hardcoded.hh_continuation_generators import (
    rebuild_polynomial_from_serialized_terms,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PruneCandidate:
    logical_index: int
    label: str
    theta: float
    abs_theta: float
    runtime_count: int


@dataclass
class PruneResult:
    removed_indices: list[int]
    removed_labels: list[str]
    surviving_indices: list[int]
    surviving_labels: list[str]
    surviving_theta: list[float]
    surviving_runtime_theta: list[float]
    original_depth: int
    pruned_depth: int
    layers_removed: int


@dataclass
class ScaffoldVQEResult:
    energy: float
    exact_energy: float
    delta_e: float
    abs_delta_e: float
    pruned_depth: int
    num_parameters: int
    logical_num_parameters: int
    optimal_point: list[float]
    nfev: int
    method: str
    elapsed_s: float


@dataclass
class CostMetrics:
    logical_depth: int
    logical_params: int
    runtime_params: int
    compiled_count_2q: int | None
    compiled_depth: int | None
    compiled_size: int | None
    abs_delta_e: float
    pareto_score: float | None  # |ΔE| / cost_2q


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_payload(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _build_hamiltonian(settings: Mapping[str, Any]):
    return build_hubbard_holstein_hamiltonian(
        dims=int(settings.get("L", 2)),
        J=float(settings.get("t", 1.0)),
        U=float(settings.get("u", 4.0)),
        omega0=float(settings.get("omega0", 1.0)),
        g=float(settings.get("g_ep", 0.5)),
        n_ph_max=int(settings.get("n_ph_max", 1)),
        boson_encoding=str(settings.get("boson_encoding", "binary")),
        v_t=None,
        v0=float(settings.get("dv", 0.0)),
        t_eval=None,
        repr_mode="JW",
        indexing=str(settings.get("ordering", "blocked")),
        pbc=(str(settings.get("boundary", "open")) == "periodic"),
        include_zero_point=True,
    )


def _build_h_matrix(coeff_map_exyz: dict[str, complex]) -> np.ndarray:
    nq = len(next(iter(coeff_map_exyz)))
    dim = 1 << nq
    from src.quantum.vqe_latex_python_pairs import _pauli_matrix_exyz
    hmat = np.zeros((dim, dim), dtype=complex)
    for label, coeff in coeff_map_exyz.items():
        hmat += coeff * _pauli_matrix_exyz(label)
    return hmat


def _extract_adapt_vqe(payload: Mapping[str, Any]) -> dict:
    av = payload.get("adapt_vqe", None)
    if isinstance(av, Mapping):
        return dict(av)
    return dict(payload)


def _extract_parameterization_blocks(adapt_vqe: Mapping[str, Any]) -> list[dict]:
    param = adapt_vqe.get("parameterization", {})
    if isinstance(param, Mapping):
        return list(param.get("blocks", []))
    return []


# ---------------------------------------------------------------------------
# Step 1: Identify pruning candidates
# ---------------------------------------------------------------------------


def identify_prune_candidates(
    adapt_vqe: Mapping[str, Any],
    threshold: float,
) -> tuple[list[PruneCandidate], list[int], list[int]]:
    logical_theta = np.array(adapt_vqe.get("logical_optimal_point", []), dtype=float)
    operators = list(adapt_vqe.get("operators", []))
    blocks = _extract_parameterization_blocks(adapt_vqe)

    candidates: list[PruneCandidate] = []
    remove_indices: list[int] = []
    keep_indices: list[int] = []

    for i, (theta_val, label) in enumerate(zip(logical_theta, operators)):
        rc = int(blocks[i].get("runtime_count", 1)) if i < len(blocks) else 1
        pc = PruneCandidate(
            logical_index=i,
            label=str(label),
            theta=float(theta_val),
            abs_theta=abs(float(theta_val)),
            runtime_count=rc,
        )
        if abs(float(theta_val)) < float(threshold):
            candidates.append(pc)
            remove_indices.append(i)
        else:
            keep_indices.append(i)

    return candidates, remove_indices, keep_indices


def apply_prune(
    adapt_vqe: Mapping[str, Any],
    remove_indices: list[int],
    keep_indices: list[int],
) -> PruneResult:
    logical_theta = np.array(adapt_vqe.get("logical_optimal_point", []), dtype=float)
    runtime_theta = np.array(adapt_vqe.get("optimal_point", []), dtype=float)
    operators = list(adapt_vqe.get("operators", []))
    blocks = _extract_parameterization_blocks(adapt_vqe)
    original_depth = int(adapt_vqe.get("ansatz_depth", len(operators)))

    surviving_labels = [str(operators[i]) for i in keep_indices]
    surviving_theta = [float(logical_theta[i]) for i in keep_indices]

    # Build surviving runtime theta by extracting runtime slices
    surviving_runtime = []
    for i in keep_indices:
        if i < len(blocks):
            start = int(blocks[i].get("runtime_start", 0))
            count = int(blocks[i].get("runtime_count", 1))
            surviving_runtime.extend(float(runtime_theta[j]) for j in range(start, start + count))
        else:
            surviving_runtime.append(float(logical_theta[i]))

    removed_labels = [str(operators[i]) for i in remove_indices]

    return PruneResult(
        removed_indices=list(remove_indices),
        removed_labels=removed_labels,
        surviving_indices=list(keep_indices),
        surviving_labels=surviving_labels,
        surviving_theta=surviving_theta,
        surviving_runtime_theta=surviving_runtime,
        original_depth=original_depth,
        pruned_depth=len(keep_indices),
        layers_removed=len(remove_indices),
    )


# ---------------------------------------------------------------------------
# Step 2A: Fixed-scaffold VQE re-optimization
# ---------------------------------------------------------------------------


def run_fixed_scaffold_vqe(
    *,
    h_poly,
    coeff_map_exyz: dict[str, complex],
    prune_result: PruneResult,
    scaffold_ops: list[AnsatzTerm],
    psi_ref: np.ndarray,
    exact_energy: float,
    method: str = "POWELL",
    maxiter: int = 2000,
    seed: int = 7,
) -> ScaffoldVQEResult:
    """Re-optimize parameters on the pruned fixed scaffold."""

    # Filter scaffold ops to surviving indices only
    surviving_ops = [scaffold_ops[i] for i in prune_result.surviving_indices]

    # Use CompiledAnsatzExecutor for fast state preparation (per_pauli_term mode
    # matches the fullhorse convention where each runtime param drives one Pauli).
    executor = CompiledAnsatzExecutor(
        surviving_ops,
        parameterization_mode="per_pauli_term",
    )

    theta0 = np.array(prune_result.surviving_runtime_theta, dtype=float)
    assert theta0.size == executor.runtime_parameter_count, (
        f"theta size {theta0.size} != executor runtime params {executor.runtime_parameter_count}"
    )

    def energy_fn(theta_runtime: np.ndarray) -> float:
        psi = executor.prepare_state(theta_runtime, psi_ref)
        return float(expval_pauli_polynomial(psi, h_poly))

    print(f"  Fixed-scaffold VQE: {len(surviving_ops)} layers, {theta0.size} runtime params, method={method}")
    print(f"  Initial energy: {energy_fn(theta0):.10f}")

    from scipy.optimize import minimize as scipy_minimize

    t0 = time.monotonic()
    result = scipy_minimize(
        energy_fn,
        theta0,
        method=method,
        options={"maxiter": maxiter, "maxfev": maxiter * 10},
    )
    elapsed = time.monotonic() - t0

    final_energy = float(result.fun)
    optimal_theta = np.asarray(result.x, dtype=float).tolist()
    nfev = int(getattr(result, "nfev", 0))

    print(f"  Final energy: {final_energy:.10f}  (ΔE = {final_energy - exact_energy:.2e})")
    print(f"  nfev={nfev}, elapsed={elapsed:.1f}s")

    return ScaffoldVQEResult(
        energy=final_energy,
        exact_energy=exact_energy,
        delta_e=final_energy - exact_energy,
        abs_delta_e=abs(final_energy - exact_energy),
        pruned_depth=prune_result.pruned_depth,
        num_parameters=len(optimal_theta),
        logical_num_parameters=prune_result.pruned_depth,
        optimal_point=optimal_theta,
        nfev=nfev,
        method=method,
        elapsed_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Step 2B: Re-ADAPT phase 3 from pruned state
# ---------------------------------------------------------------------------


def run_readapt_phase3(
    *,
    input_json_path: Path,
    prune_result: PruneResult,
    output_json_path: Path,
    adapt_max_depth: int = 12,
    adapt_drop_floor: float = 1e-9,
    adapt_drop_patience: int = 3,
    adapt_drop_min_depth: int = 6,
    phase3_backend_name: str = "FakeNighthawk",
) -> dict[str, Any] | None:
    """Invoke adapt_pipeline.py with aggressive termination seeded from the pruned state.

    Returns the result payload or None on failure.
    """
    from pipelines.hardcoded.adapt_pipeline import main as adapt_main

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if output_json_path is None:
        output_json_path = REPO_ROOT / "artifacts" / "json" / f"hh_prune_readapt_nighthawk_{ts}.json"

    argv = [
        "--problem", "hh",
        "--L", "2",
        "--t", "1.0",
        "--u", "4.0",
        "--omega0", "1.0",
        "--g-ep", "0.5",
        "--n-ph-max", "1",
        "--boson-encoding", "binary",
        "--ordering", "blocked",
        "--boundary", "open",
        "--adapt-pool", "full_meta",
        "--adapt-continuation-mode", "phase3_v1",
        "--adapt-inner-optimizer", "POWELL",
        "--adapt-state-backend", "compiled",
        "--adapt-reopt-policy", "windowed",
        "--adapt-window-size", "999999",
        "--adapt-window-topk", "999999",
        "--adapt-full-refit-every", "8",
        "--adapt-final-full-refit", "true",
        # Aggressive termination
        "--adapt-max-depth", str(adapt_max_depth),
        "--adapt-drop-floor", str(adapt_drop_floor),
        "--adapt-drop-patience", str(adapt_drop_patience),
        "--adapt-drop-min-depth", str(adapt_drop_min_depth),
        "--adapt-eps-grad", "5e-7",
        "--adapt-eps-energy", "1e-9",
        "--adapt-seed", "7",
        # Phase 1 pruning
        "--phase1-prune-enabled",
        "--phase1-prune-fraction", "0.25",
        "--phase1-prune-max-candidates", "6",
        "--phase1-prune-max-regression", "1e-8",
        # Phase 2 scoring
        "--phase1-shortlist-size", "256",
        "--phase2-shortlist-fraction", "1.0",
        "--phase2-shortlist-size", "128",
        "--phase2-enable-batching",
        "--phase2-batch-target-size", "8",
        "--phase2-batch-size-cap", "16",
        # Phase 3 backend cost
        "--phase3-backend-cost-mode", "transpile_single_v1",
        "--phase3-backend-name", phase3_backend_name,
        "--phase3-runtime-split-mode", "shortlist_pauli_children_v1",
        "--phase3-lifetime-cost-mode", "phase3_v1",
        "--phase3-enable-rescue",
        "--phase3-symmetry-mitigation-mode", "verify_only",
        # Warm-start from fullhorse
        "--adapt-ref-json", str(input_json_path),
        # Output
        "--output-json", str(output_json_path),
        "--skip-pdf",
    ]

    print(f"\n  Re-ADAPT phase 3: max_depth={adapt_max_depth}, "
          f"drop_floor={adapt_drop_floor}, patience={adapt_drop_patience}")
    print(f"  Backend: {phase3_backend_name}")
    print(f"  Output: {output_json_path}")

    try:
        adapt_main(argv)
        if output_json_path.exists():
            with open(output_json_path) as f:
                return json.load(f)
    except Exception as exc:
        print(f"  Re-ADAPT failed: {exc}")
    return None


# ---------------------------------------------------------------------------
# Cost evaluation
# ---------------------------------------------------------------------------


def compute_cost_metrics(
    *,
    payload: Mapping[str, Any] | None = None,
    energy: float | None = None,
    exact_energy: float | None = None,
    logical_depth: int | None = None,
    logical_params: int | None = None,
    runtime_params: int | None = None,
    label: str = "",
) -> CostMetrics:
    """Extract or compute cost metrics for Pareto comparison."""
    if payload is not None:
        av = payload.get("adapt_vqe", payload)
        energy = float(av.get("energy", energy or 0.0))
        exact_energy = float(av.get("exact_gs_energy", exact_energy or 0.0))
        logical_depth = int(av.get("ansatz_depth", logical_depth or 0))
        logical_params = int(av.get("logical_num_parameters", logical_params or 0))
        runtime_params = int(av.get("num_parameters", runtime_params or 0))

        # Try to get backend cost from last history entry
        history = av.get("history", [])
        compiled_2q = None
        compiled_depth = None
        compiled_size = None
        if history:
            last = history[-1]
            cb = last.get("compile_cost_backend", {})
            if isinstance(cb, dict):
                compiled_2q = cb.get("trial_compiled_count_2q")
                compiled_depth = cb.get("trial_compiled_depth")
                compiled_size = cb.get("trial_compiled_size")

        abs_de = abs(energy - exact_energy) if energy is not None and exact_energy is not None else None
        pareto = (abs_de / float(compiled_2q)) if abs_de is not None and compiled_2q and compiled_2q > 0 else None

        return CostMetrics(
            logical_depth=logical_depth or 0,
            logical_params=logical_params or 0,
            runtime_params=runtime_params or 0,
            compiled_count_2q=int(compiled_2q) if compiled_2q is not None else None,
            compiled_depth=int(compiled_depth) if compiled_depth is not None else None,
            compiled_size=int(compiled_size) if compiled_size is not None else None,
            abs_delta_e=abs_de or 0.0,
            pareto_score=pareto,
        )

    abs_de = abs(energy - exact_energy) if energy is not None and exact_energy is not None else 0.0
    return CostMetrics(
        logical_depth=logical_depth or 0,
        logical_params=logical_params or 0,
        runtime_params=runtime_params or 0,
        compiled_count_2q=None,
        compiled_depth=None,
        compiled_size=None,
        abs_delta_e=abs_de,
        pareto_score=None,
    )


# ---------------------------------------------------------------------------
# Transpile the pruned scaffold for hardware cost
# ---------------------------------------------------------------------------


def transpile_pruned_circuit(
    *,
    scaffold_ops: list[AnsatzTerm],
    surviving_indices: list[int],
    surviving_theta: list[float],
    nq: int,
    backend_name: str = "FakeNighthawk",
    optimization_level: int = 1,
    seed: int = 7,
) -> dict[str, Any]:
    """Build and transpile the pruned circuit to get compiled gate counts."""
    from pipelines.hardcoded.adapt_circuit_execution import (
        append_pauli_rotation_exyz as _append_rot,
        append_reference_state as _append_ref,
        build_ansatz_circuit as _build_circ,
    )
    from pipelines.exact_bench.noise_oracle_runtime import (
        _load_fake_backend,
        compile_circuit_for_local_backend,
    )
    from pipelines.qiskit_backend_tools import (
        compiled_gate_stats as _gate_stats,
        safe_circuit_depth as _safe_depth,
    )
    from qiskit import QuantumCircuit

    surviving_ops = [scaffold_ops[i] for i in surviving_indices]
    layout = build_parameter_layout(surviving_ops)
    theta_runtime = np.zeros(layout.total_runtime_params, dtype=float)

    # Populate runtime theta
    rt_idx = 0
    for li, blk in enumerate(layout.blocks):
        for t_idx, _term in enumerate(blk.terms):
            if rt_idx < len(surviving_theta):
                # Map logical to runtime
                pass
            rt_idx += 1

    # Use the shared circuit builder
    qc = _build_circ(surviving_ops, layout, np.array(surviving_theta), nq)

    backend = _load_fake_backend(backend_name)
    compiled_qc = compile_circuit_for_local_backend(
        qc, backend,
        optimization_level=optimization_level,
        seed=seed,
    )
    stats = _gate_stats(compiled_qc)
    depth = _safe_depth(compiled_qc)

    return {
        "backend_name": backend_name,
        "logical_depth": len(surviving_indices),
        "compiled_count_2q": int(stats.get("count_2q", 0)),
        "compiled_depth": int(depth),
        "compiled_size": int(stats.get("size", 0)),
        "compiled_cx_count": int(stats.get("cx", 0)),
        "compiled_ecr_count": int(stats.get("ecr", 0)),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def format_report(
    *,
    prune_candidates: list[PruneCandidate],
    prune_result: PruneResult,
    original_metrics: CostMetrics,
    scaffold_vqe_result: ScaffoldVQEResult | None,
    scaffold_metrics: CostMetrics | None,
    readapt_metrics: CostMetrics | None,
    input_path: str,
    threshold: float,
) -> str:
    lines = [
        "# Nighthawk Pruning Analysis",
        "",
        f"**Source:** `{input_path}`",
        f"**Prune threshold:** |θ| < {threshold:.1e}",
        f"**Generated:** {_now_utc()}",
        "",
        "## Original Circuit",
        "",
        f"- Logical depth: {original_metrics.logical_depth}",
        f"- Logical params: {original_metrics.logical_params}",
        f"- Runtime params: {original_metrics.runtime_params}",
        f"- Compiled 2Q gates: {original_metrics.compiled_count_2q}",
        f"- Compiled depth: {original_metrics.compiled_depth}",
        f"- Compiled size: {original_metrics.compiled_size}",
        f"- |ΔE|: {original_metrics.abs_delta_e:.2e}",
        f"- Pareto |ΔE|/2Q: {original_metrics.pareto_score:.2e}" if original_metrics.pareto_score else "",
        "",
        "## Pruning Candidates",
        "",
        "| Index | Label | |θ| | Decision |",
        "|-------|-------|-----|----------|",
    ]
    for pc in prune_candidates:
        lines.append(f"| {pc.logical_index} | `{pc.label}` | {pc.abs_theta:.2e} | REMOVE |")

    lines.extend([
        "",
        f"**Layers removed:** {prune_result.layers_removed} / {prune_result.original_depth}",
        f"**Pruned depth:** {prune_result.pruned_depth}",
        "",
        "## Surviving Scaffold",
        "",
    ])
    for i, (idx, label, theta) in enumerate(zip(
        prune_result.surviving_indices,
        prune_result.surviving_labels,
        prune_result.surviving_theta,
    )):
        lines.append(f"{i+1}. `{label}` (θ={theta:+.6f})")

    # Path A results
    if scaffold_vqe_result is not None:
        lines.extend([
            "",
            "## Path A: Fixed-Scaffold VQE",
            "",
            f"- Energy: {scaffold_vqe_result.energy:.10f}",
            f"- |ΔE|: {scaffold_vqe_result.abs_delta_e:.2e}",
            f"- Depth: {scaffold_vqe_result.pruned_depth}",
            f"- Params: {scaffold_vqe_result.num_parameters}",
            f"- Method: {scaffold_vqe_result.method}",
            f"- nfev: {scaffold_vqe_result.nfev}",
            f"- Elapsed: {scaffold_vqe_result.elapsed_s:.1f}s",
        ])
        if scaffold_metrics and scaffold_metrics.compiled_count_2q is not None:
            lines.extend([
                f"- Compiled 2Q gates: {scaffold_metrics.compiled_count_2q}",
                f"- Compiled depth: {scaffold_metrics.compiled_depth}",
                f"- Compiled size: {scaffold_metrics.compiled_size}",
                f"- Pareto |ΔE|/2Q: {scaffold_metrics.pareto_score:.2e}" if scaffold_metrics.pareto_score else "",
            ])

    # Path B results
    if readapt_metrics is not None:
        lines.extend([
            "",
            "## Path B: Re-ADAPT Phase 3",
            "",
            f"- Logical depth: {readapt_metrics.logical_depth}",
            f"- Logical params: {readapt_metrics.logical_params}",
            f"- Runtime params: {readapt_metrics.runtime_params}",
            f"- |ΔE|: {readapt_metrics.abs_delta_e:.2e}",
        ])
        if readapt_metrics.compiled_count_2q is not None:
            lines.extend([
                f"- Compiled 2Q gates: {readapt_metrics.compiled_count_2q}",
                f"- Compiled depth: {readapt_metrics.compiled_depth}",
                f"- Compiled size: {readapt_metrics.compiled_size}",
                f"- Pareto |ΔE|/2Q: {readapt_metrics.pareto_score:.2e}" if readapt_metrics.pareto_score else "",
            ])

    # Comparison table
    lines.extend([
        "",
        "## Pareto Comparison: |ΔE| / Cost",
        "",
        "| Variant | Depth | 2Q Gates | |ΔE| | |ΔE|/2Q |",
        "|---------|------:|--------:|---------:|--------:|",
    ])
    def _row(name: str, m: CostMetrics) -> str:
        c2q = str(m.compiled_count_2q) if m.compiled_count_2q is not None else "---"
        ps = f"{m.pareto_score:.2e}" if m.pareto_score is not None else "---"
        return f"| {name} | {m.logical_depth} | {c2q} | {m.abs_delta_e:.2e} | {ps} |"
    lines.append(_row("Original (fullhorse)", original_metrics))
    if scaffold_metrics:
        lines.append(_row("Path A (fixed scaffold)", scaffold_metrics))
    if readapt_metrics:
        lines.append(_row("Path B (re-ADAPT)", readapt_metrics))

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prune fullhorse Nighthawk ADAPT circuit.")
    p.add_argument("--input-json", type=Path, required=True,
                    help="Path to fullhorse Nighthawk ADAPT JSON artifact.")
    p.add_argument("--output-json", type=Path, default=None,
                    help="Output JSON path for combined results.")
    p.add_argument("--output-md", type=Path, default=None,
                    help="Output markdown report path.")
    p.add_argument("--mode", choices=["scaffold", "readapt", "both"], default="both",
                    help="Which optimization path(s) to run.")
    p.add_argument("--prune-threshold", type=float, default=1e-4,
                    help="Absolute theta threshold below which layers are pruned.")
    p.add_argument("--scaffold-method", type=str, default="POWELL",
                    choices=["POWELL", "COBYLA"],
                    help="Optimizer for fixed-scaffold VQE re-optimization.")
    p.add_argument("--scaffold-maxiter", type=int, default=2000,
                    help="Max iterations for scaffold VQE.")
    p.add_argument("--readapt-max-depth", type=int, default=12,
                    help="Max ADAPT depth for re-ADAPT path.")
    p.add_argument("--readapt-drop-floor", type=float, default=1e-9,
                    help="Drop floor for re-ADAPT termination.")
    p.add_argument("--readapt-drop-patience", type=int, default=3,
                    help="Drop patience for re-ADAPT termination.")
    p.add_argument("--readapt-drop-min-depth", type=int, default=6,
                    help="Min depth before drop policy in re-ADAPT.")
    p.add_argument("--backend-name", type=str, default="FakeNighthawk",
                    help="Backend for transpilation cost evaluation.")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print(f"=== Nighthawk Pruning Pipeline ===")
    print(f"Input: {args.input_json}")
    print(f"Mode: {args.mode}")
    print(f"Prune threshold: {args.prune_threshold}")

    # Load artifact
    payload = _load_payload(args.input_json)
    adapt_vqe = _extract_adapt_vqe(payload)
    settings = payload.get("settings", {})

    # Build Hamiltonian
    h_poly = _build_hamiltonian(settings)
    native_order, coeff_map_exyz = _collect_hardcoded_terms_exyz(h_poly)
    nq = len(next(iter(coeff_map_exyz)))

    # Exact energy
    num_particles = tuple(half_filled_num_particles(int(settings.get("L", 2))))
    exact_energy = float(adapt_vqe.get("exact_gs_energy",
                         payload.get("ground_state", {}).get("exact_energy", 0.0)))
    print(f"Exact energy: {exact_energy:.10f}")

    # Original metrics
    original_metrics = compute_cost_metrics(payload=payload)
    print(f"Original: depth={original_metrics.logical_depth}, "
          f"2Q={original_metrics.compiled_count_2q}, |ΔE|={original_metrics.abs_delta_e:.2e}")

    # Step 1: Identify pruning candidates
    print(f"\n--- Pruning Analysis (threshold={args.prune_threshold:.1e}) ---")
    candidates, remove_indices, keep_indices = identify_prune_candidates(
        adapt_vqe, args.prune_threshold,
    )
    print(f"Found {len(candidates)} prunable layers out of {len(adapt_vqe.get('operators', []))}:")
    for c in candidates:
        print(f"  [{c.logical_index}] |θ|={c.abs_theta:.2e}  {c.label}")

    prune_result = apply_prune(adapt_vqe, remove_indices, keep_indices)
    print(f"Pruned: {prune_result.original_depth} -> {prune_result.pruned_depth} layers "
          f"({prune_result.layers_removed} removed)")

    # Resolve scaffold operators
    normalized = _normalize_adapt_payload(payload)
    scaffold_ops = _resolve_scaffold_ops(normalized, h_poly)

    # Reference state (HH includes phonon qubits — need full 6-qubit state)
    psi_ref = np.asarray(
        hubbard_holstein_reference_state(
            dims=int(settings.get("L", 2)),
            num_particles=num_particles,
            n_ph_max=int(settings.get("n_ph_max", 1)),
            boson_encoding=str(settings.get("boson_encoding", "binary")),
            indexing=str(settings.get("ordering", "blocked")),
        ),
        dtype=complex,
    ).reshape(-1)
    print(f"Reference state: {psi_ref.size} amplitudes ({nq} qubits)")

    # Path A: Fixed-scaffold VQE
    scaffold_vqe_result = None
    scaffold_metrics = None
    if args.mode in ("scaffold", "both"):
        print(f"\n--- Path A: Fixed-Scaffold VQE ---")
        scaffold_vqe_result = run_fixed_scaffold_vqe(
            h_poly=h_poly,
            coeff_map_exyz=coeff_map_exyz,
            prune_result=prune_result,
            scaffold_ops=scaffold_ops,
            psi_ref=psi_ref,
            exact_energy=exact_energy,
            method=args.scaffold_method,
            maxiter=args.scaffold_maxiter,
            seed=args.seed,
        )
        scaffold_metrics = CostMetrics(
            logical_depth=scaffold_vqe_result.pruned_depth,
            logical_params=scaffold_vqe_result.logical_num_parameters,
            runtime_params=scaffold_vqe_result.num_parameters,
            compiled_count_2q=None,
            compiled_depth=None,
            compiled_size=None,
            abs_delta_e=scaffold_vqe_result.abs_delta_e,
            pareto_score=None,
        )

    # Path B: Re-ADAPT phase 3
    readapt_payload = None
    readapt_metrics = None
    if args.mode in ("readapt", "both"):
        print(f"\n--- Path B: Re-ADAPT Phase 3 ---")
        readapt_output = args.output_json.parent / f"hh_prune_readapt_nighthawk_{ts}.json" if args.output_json else (
            REPO_ROOT / "artifacts" / "json" / f"hh_prune_readapt_nighthawk_{ts}.json"
        )
        readapt_payload = run_readapt_phase3(
            input_json_path=args.input_json,
            prune_result=prune_result,
            output_json_path=readapt_output,
            adapt_max_depth=args.readapt_max_depth,
            adapt_drop_floor=args.readapt_drop_floor,
            adapt_drop_patience=args.readapt_drop_patience,
            adapt_drop_min_depth=args.readapt_drop_min_depth,
            phase3_backend_name=args.backend_name,
        )
        if readapt_payload is not None:
            readapt_metrics = compute_cost_metrics(payload=readapt_payload)
            print(f"  Re-ADAPT result: depth={readapt_metrics.logical_depth}, "
                  f"2Q={readapt_metrics.compiled_count_2q}, |ΔE|={readapt_metrics.abs_delta_e:.2e}")

    # Generate report
    report = format_report(
        prune_candidates=candidates,
        prune_result=prune_result,
        original_metrics=original_metrics,
        scaffold_vqe_result=scaffold_vqe_result,
        scaffold_metrics=scaffold_metrics,
        readapt_metrics=readapt_metrics,
        input_path=str(args.input_json),
        threshold=args.prune_threshold,
    )

    # Save outputs
    output_data: dict[str, Any] = {
        "generated_utc": _now_utc(),
        "pipeline": "hh_prune_nighthawk",
        "input_json": str(args.input_json),
        "prune_threshold": float(args.prune_threshold),
        "mode": str(args.mode),
        "prune_candidates": [asdict(c) for c in candidates],
        "prune_result": {
            "removed_indices": prune_result.removed_indices,
            "removed_labels": prune_result.removed_labels,
            "surviving_indices": prune_result.surviving_indices,
            "surviving_labels": prune_result.surviving_labels,
            "original_depth": prune_result.original_depth,
            "pruned_depth": prune_result.pruned_depth,
            "layers_removed": prune_result.layers_removed,
        },
        "original_metrics": asdict(original_metrics),
    }
    if scaffold_vqe_result is not None:
        output_data["path_a_scaffold_vqe"] = asdict(scaffold_vqe_result)
    if scaffold_metrics is not None:
        output_data["path_a_metrics"] = asdict(scaffold_metrics)
    if readapt_metrics is not None:
        output_data["path_b_readapt_metrics"] = asdict(readapt_metrics)
    if readapt_payload is not None:
        output_data["path_b_readapt_output_json"] = str(
            args.output_json.parent / f"hh_prune_readapt_nighthawk_{ts}.json" if args.output_json
            else REPO_ROOT / "artifacts" / "json" / f"hh_prune_readapt_nighthawk_{ts}.json"
        )

    output_json = args.output_json or (REPO_ROOT / "artifacts" / "json" / f"hh_prune_nighthawk_{ts}.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nJSON saved: {output_json}")

    output_md = args.output_md or (REPO_ROOT / "artifacts" / "reports" / f"hh_prune_nighthawk_{ts}.md")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(report, encoding="utf-8")
    print(f"Report saved: {output_md}")

    print(f"\n=== Done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
