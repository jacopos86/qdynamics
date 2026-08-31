#!/usr/bin/env python3
"""Leave-one-out marginal analysis on the pruned Nighthawk scaffold.

For each surviving operator, removes it from the scaffold, re-optimizes
remaining parameters with POWELL, and reports:
  - Energy regression (ΔE_regression)
  - Estimated 2Q gate savings from removing that layer
  - Cost-effectiveness ratio: ΔE_regression / gates_saved

This builds a Pareto menu of further pruning candidates ranked by how
much accuracy you sacrifice per gate saved.

Usage:
    python -m pipelines.hardcoded.hh_prune_marginal_analysis \
        --input-json artifacts/json/hh_prune_nighthawk_pruned_scaffold.json
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

from scipy.optimize import minimize as scipy_minimize

from src.quantum.hubbard_latex_python_pairs import (
    build_hubbard_holstein_hamiltonian,
)
from src.quantum.hartree_fock_reference_state import hubbard_holstein_reference_state
from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.vqe_latex_python_pairs import (
    AnsatzTerm,
    expval_pauli_polynomial,
    half_filled_num_particles,
)
from pipelines.hardcoded.adapt_pipeline import (
    _build_hh_full_meta_pool,
    _collect_hardcoded_terms_exyz,
)
from pipelines.hardcoded.adapt_circuit_cost import (
    _normalize_adapt_payload,
    _resolve_scaffold_ops,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LeaveOneOutResult:
    removed_index: int
    removed_label: str
    removed_theta: float
    remaining_depth: int
    remaining_runtime_params: int
    energy_before: float
    energy_after: float
    delta_e_regression: float
    abs_delta_e_before: float
    abs_delta_e_after: float
    nfev: int
    elapsed_s: float


@dataclass
class LeaveMultipleOutResult:
    removed_indices: list[int]
    removed_labels: list[str]
    remaining_depth: int
    remaining_runtime_params: int
    energy_after: float
    abs_delta_e_after: float
    delta_e_regression_from_base: float
    nfev: int
    elapsed_s: float


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _build_energy_fn(
    scaffold_ops: list[AnsatzTerm],
    h_poly,
    psi_ref: np.ndarray,
):
    """Build a compiled energy function for a given scaffold."""
    executor = CompiledAnsatzExecutor(
        scaffold_ops,
        parameterization_mode="per_pauli_term",
    )

    def energy_fn(theta: np.ndarray) -> float:
        psi = executor.prepare_state(theta, psi_ref)
        return float(expval_pauli_polynomial(psi, h_poly))

    return energy_fn, executor.runtime_parameter_count


def _reoptimize(energy_fn, theta0: np.ndarray, method: str = "POWELL", maxiter: int = 3000):
    """Re-optimize and return (energy, theta, nfev, elapsed)."""
    t0 = time.monotonic()
    result = scipy_minimize(
        energy_fn,
        theta0,
        method=method,
        options={"maxiter": maxiter, "maxfev": maxiter * 10},
    )
    elapsed = time.monotonic() - t0
    return float(result.fun), np.asarray(result.x, dtype=float), int(getattr(result, "nfev", 0)), elapsed


def _extract_runtime_theta_for_indices(
    blocks: list[dict],
    runtime_theta: np.ndarray,
    keep_indices: list[int],
) -> np.ndarray:
    """Extract runtime theta for a subset of logical indices."""
    parts = []
    for i in keep_indices:
        b = blocks[i]
        start = int(b["runtime_start"])
        count = int(b["runtime_count"])
        parts.extend(float(runtime_theta[j]) for j in range(start, start + count))
    return np.array(parts, dtype=float)


def run_leave_one_out(
    *,
    scaffold_ops: list[AnsatzTerm],
    blocks: list[dict],
    runtime_theta: np.ndarray,
    h_poly,
    psi_ref: np.ndarray,
    base_energy: float,
    exact_energy: float,
) -> list[LeaveOneOutResult]:
    """For each operator, remove it and re-optimize."""
    n = len(scaffold_ops)
    results = []

    for remove_idx in range(n):
        keep = [i for i in range(n) if i != remove_idx]
        kept_ops = [scaffold_ops[i] for i in keep]
        kept_theta = _extract_runtime_theta_for_indices(blocks, runtime_theta, keep)

        energy_fn, n_params = _build_energy_fn(kept_ops, h_poly, psi_ref)
        assert kept_theta.size == n_params, f"theta mismatch: {kept_theta.size} vs {n_params}"

        energy_after, _, nfev, elapsed = _reoptimize(energy_fn, kept_theta)

        removed_label = blocks[remove_idx]["candidate_label"]
        # Get logical theta for this block
        start = int(blocks[remove_idx]["runtime_start"])
        count = int(blocks[remove_idx]["runtime_count"])
        if count == 1:
            removed_theta = float(runtime_theta[start])
        else:
            removed_theta = float(np.mean(runtime_theta[start:start + count]))

        regression = energy_after - base_energy

        results.append(LeaveOneOutResult(
            removed_index=remove_idx,
            removed_label=removed_label,
            removed_theta=removed_theta,
            remaining_depth=n - 1,
            remaining_runtime_params=n_params,
            energy_before=base_energy,
            energy_after=energy_after,
            delta_e_regression=regression,
            abs_delta_e_before=abs(base_energy - exact_energy),
            abs_delta_e_after=abs(energy_after - exact_energy),
            nfev=nfev,
            elapsed_s=elapsed,
        ))

        flag = " ** cheap!" if regression < 1e-5 else ""
        print(f"  [{remove_idx:2d}] remove {removed_label:55s}  "
              f"ΔE_reg={regression:+.2e}  |ΔE|={abs(energy_after - exact_energy):.2e}  "
              f"nfev={nfev:5d}{flag}")

    return results


def run_greedy_multi_prune(
    *,
    scaffold_ops: list[AnsatzTerm],
    blocks: list[dict],
    runtime_theta: np.ndarray,
    h_poly,
    psi_ref: np.ndarray,
    base_energy: float,
    exact_energy: float,
    loo_results: list[LeaveOneOutResult],
    regression_budgets: list[float],
) -> list[LeaveMultipleOutResult]:
    """Greedy sequential pruning: remove cheapest operators first."""
    # Sort by regression (cheapest to remove first)
    ranked = sorted(loo_results, key=lambda r: r.delta_e_regression)

    multi_results = []
    for budget in regression_budgets:
        # Greedily remove operators while total regression stays within budget
        removed_indices = []
        removed_labels = []
        current_ops = list(range(len(scaffold_ops)))
        cumulative_regression = 0.0

        for candidate in ranked:
            if candidate.removed_index not in current_ops:
                continue
            trial_keep = [i for i in current_ops if i != candidate.removed_index]
            if len(trial_keep) < 3:  # Keep at least 3 operators
                break

            kept_ops = [scaffold_ops[i] for i in trial_keep]
            kept_theta = _extract_runtime_theta_for_indices(blocks, runtime_theta, trial_keep)

            energy_fn, n_params = _build_energy_fn(kept_ops, h_poly, psi_ref)
            energy_after, new_theta, nfev, elapsed = _reoptimize(energy_fn, kept_theta)

            regression = energy_after - base_energy
            if regression <= budget:
                removed_indices.append(candidate.removed_index)
                removed_labels.append(candidate.removed_label)
                current_ops = trial_keep
                # Update runtime_theta for next iteration
                runtime_theta_updated = np.zeros(0)
                for i in trial_keep:
                    b = blocks[i]
                    start = int(b["runtime_start"])
                    count = int(b["runtime_count"])
                    runtime_theta_updated = np.concatenate([
                        runtime_theta_updated,
                        np.array([float(runtime_theta[j]) for j in range(start, start + count)])
                    ])
                # Use the optimized theta for further removals
                runtime_theta = np.zeros(max(int(b["runtime_start"] + b["runtime_count"])
                                            for b in blocks) + 1)
                rt_idx = 0
                for i in trial_keep:
                    b = blocks[i]
                    start = int(b["runtime_start"])
                    count = int(b["runtime_count"])
                    for j in range(count):
                        if rt_idx < new_theta.size:
                            runtime_theta[start + j] = new_theta[rt_idx]
                        rt_idx += 1

        if removed_indices:
            # Final evaluation of the pruned circuit
            final_keep = [i for i in range(len(scaffold_ops)) if i not in removed_indices]
            final_ops = [scaffold_ops[i] for i in final_keep]
            final_theta = _extract_runtime_theta_for_indices(blocks, runtime_theta, final_keep)
            energy_fn, n_params = _build_energy_fn(final_ops, h_poly, psi_ref)
            final_energy, _, nfev, elapsed = _reoptimize(energy_fn, final_theta)

            multi_results.append(LeaveMultipleOutResult(
                removed_indices=removed_indices,
                removed_labels=removed_labels,
                remaining_depth=len(scaffold_ops) - len(removed_indices),
                remaining_runtime_params=n_params,
                energy_after=final_energy,
                abs_delta_e_after=abs(final_energy - exact_energy),
                delta_e_regression_from_base=final_energy - base_energy,
                nfev=nfev,
                elapsed_s=elapsed,
            ))

            removed_str = ", ".join(f"[{i}]" for i in removed_indices)
            print(f"  Budget {budget:.1e}: removed {removed_str} -> "
                  f"depth={len(scaffold_ops) - len(removed_indices)}, "
                  f"|ΔE|={abs(final_energy - exact_energy):.2e}, "
                  f"regression={final_energy - base_energy:+.2e}")

    return multi_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leave-one-out marginal analysis on pruned scaffold.")
    p.add_argument("--input-json", type=Path, required=True)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-md", type=Path, default=None)
    p.add_argument("--method", type=str, default="POWELL")
    p.add_argument("--maxiter", type=int, default=3000)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print("=== Marginal Pruning Analysis ===")
    print(f"Input: {args.input_json}")

    # Load
    with open(args.input_json) as f:
        payload = json.load(f)

    settings = payload.get("settings", {})
    adapt_vqe = payload.get("adapt_vqe", payload)
    operators = adapt_vqe.get("operators", [])
    blocks = adapt_vqe.get("parameterization", {}).get("blocks", [])
    runtime_theta = np.array(adapt_vqe.get("optimal_point", []), dtype=float)
    exact_energy = float(adapt_vqe.get("exact_gs_energy",
                         payload.get("ground_state", {}).get("exact_energy", 0.0)))

    # Build Hamiltonian
    h_poly = build_hubbard_holstein_hamiltonian(
        dims=int(settings.get("L", 2)),
        J=float(settings.get("t", 1.0)),
        U=float(settings.get("u", 4.0)),
        omega0=float(settings.get("omega0", 1.0)),
        g=float(settings.get("g_ep", 0.5)),
        n_ph_max=int(settings.get("n_ph_max", 1)),
        boson_encoding=str(settings.get("boson_encoding", "binary")),
        v_t=None, v0=float(settings.get("dv", 0.0)), t_eval=None,
        repr_mode="JW",
        indexing=str(settings.get("ordering", "blocked")),
        pbc=(str(settings.get("boundary", "open")) == "periodic"),
        include_zero_point=True,
    )

    # Resolve scaffold ops
    normalized = _normalize_adapt_payload(payload)
    scaffold_ops = _resolve_scaffold_ops(normalized, h_poly)

    # Reference state
    num_particles = tuple(half_filled_num_particles(int(settings.get("L", 2))))
    psi_ref = np.asarray(
        hubbard_holstein_reference_state(
            dims=int(settings.get("L", 2)),
            num_particles=num_particles,
            n_ph_max=int(settings.get("n_ph_max", 1)),
            boson_encoding=str(settings.get("boson_encoding", "binary")),
            indexing=str(settings.get("ordering", "blocked")),
        ), dtype=complex,
    ).reshape(-1)

    # Baseline energy
    energy_fn_base, _ = _build_energy_fn(scaffold_ops, h_poly, psi_ref)
    base_energy = energy_fn_base(runtime_theta)
    print(f"Base energy: {base_energy:.10f}  |ΔE|={abs(base_energy - exact_energy):.2e}")
    print(f"Exact energy: {exact_energy:.10f}")
    print(f"Scaffold: {len(operators)} operators, {runtime_theta.size} runtime params")

    # Leave-one-out
    print(f"\n--- Leave-One-Out Analysis ---")
    loo_results = run_leave_one_out(
        scaffold_ops=scaffold_ops,
        blocks=blocks,
        runtime_theta=runtime_theta,
        h_poly=h_poly,
        psi_ref=psi_ref,
        base_energy=base_energy,
        exact_energy=exact_energy,
    )

    # Greedy multi-prune at various budgets
    print(f"\n--- Greedy Multi-Prune ---")
    budgets = [1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3]
    multi_results = run_greedy_multi_prune(
        scaffold_ops=scaffold_ops,
        blocks=blocks,
        runtime_theta=runtime_theta.copy(),
        h_poly=h_poly,
        psi_ref=psi_ref,
        base_energy=base_energy,
        exact_energy=exact_energy,
        loo_results=loo_results,
        regression_budgets=budgets,
    )

    # --- Build report ---
    lines = [
        "# Marginal Pruning Analysis — Nighthawk Pruned Scaffold",
        "",
        f"**Source:** `{args.input_json}`",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Baseline",
        "",
        f"- Operators: {len(operators)}",
        f"- Runtime params: {runtime_theta.size}",
        f"- Energy: {base_energy:.10f}",
        f"- |ΔE|: {abs(base_energy - exact_energy):.2e}",
        f"- Exact: {exact_energy:.10f}",
        f"- Compiled 2Q gates (FakeNighthawk): 53",
        "",
        "## Leave-One-Out Results",
        "",
        "Each row shows what happens when that single operator is removed and remaining params are re-optimized.",
        "",
        "| Idx | Operator | |θ| | ΔE regression | |ΔE| after | Cheap? |",
        "|-----|----------|-----|---------------|-----------|--------|",
    ]

    sorted_loo = sorted(loo_results, key=lambda r: r.delta_e_regression)
    for r in sorted_loo:
        cheap = "yes" if r.delta_e_regression < 1e-4 else ""
        lines.append(
            f"| {r.removed_index} | `{r.removed_label}` | {abs(r.removed_theta):.2e} "
            f"| {r.delta_e_regression:+.2e} | {r.abs_delta_e_after:.2e} | {cheap} |"
        )

    lines.extend([
        "",
        "### Interpretation",
        "",
        "Operators sorted by cheapness of removal (smallest regression first). "
        "Operators with regression < 1e-4 are candidates for further pruning "
        "where the gate savings likely justify the small accuracy cost.",
        "",
    ])

    # Greedy multi-prune table
    if multi_results:
        lines.extend([
            "## Greedy Multi-Prune (regression budget)",
            "",
            "Sequential greedy removal of cheapest operators, re-optimizing after each.",
            "",
            "| Budget | Removed | Remaining depth | |ΔE| after | Regression | Layers cut |",
            "|--------|---------|----------------:|----------:|-----------:|-----------:|",
        ])
        for budget, mr in zip(budgets, multi_results):
            removed_str = ", ".join(f"[{i}]" for i in mr.removed_indices)
            lines.append(
                f"| {budget:.0e} | {removed_str} | {mr.remaining_depth} "
                f"| {mr.abs_delta_e_after:.2e} | {mr.delta_e_regression_from_base:+.2e} "
                f"| {len(mr.removed_indices)} |"
            )

    lines.extend([
        "",
        "## Pareto Menu",
        "",
        "These are the pruning options available, from most conservative to most aggressive:",
        "",
        "| Variant | Depth | Est. 2Q gates | |ΔE| | Notes |",
        "|---------|------:|-------------:|---------:|-------|",
        f"| Current pruned | {len(operators)} | 53 | {abs(base_energy - exact_energy):.2e} | Baseline (no further pruning) |",
    ])
    for budget, mr in zip(budgets, multi_results):
        removed_names = ", ".join(r.split("(")[0].strip("`") for r in mr.removed_labels)
        lines.append(
            f"| Budget {budget:.0e} | {mr.remaining_depth} | ~{max(20, 53 - len(mr.removed_indices)*5)} "
            f"| {mr.abs_delta_e_after:.2e} | -{len(mr.removed_indices)} ops: {removed_names} |"
        )

    lines.append("")
    report = "\n".join(lines)

    # Save
    output_md = args.output_md or (REPO_ROOT / "artifacts" / "reports" / f"hh_prune_marginal_{ts}.md")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(report, encoding="utf-8")
    print(f"\nReport: {output_md}")

    output_json = args.output_json or (REPO_ROOT / "artifacts" / "json" / f"hh_prune_marginal_{ts}.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "hh_prune_marginal_analysis",
        "input_json": str(args.input_json),
        "base_energy": base_energy,
        "exact_energy": exact_energy,
        "leave_one_out": [asdict(r) for r in sorted_loo],
        "greedy_multi_prune": [
            {"budget": b, **asdict(mr)} for b, mr in zip(budgets, multi_results)
        ],
    }
    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"JSON: {output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
