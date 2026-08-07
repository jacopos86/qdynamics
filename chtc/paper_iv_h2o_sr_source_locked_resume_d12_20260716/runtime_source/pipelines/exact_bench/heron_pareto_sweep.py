#!/usr/bin/env python3
"""Transpile-sweep all fixed L=2 scaffolds to a Heron backend, then extract
the Pareto front over (|DeltaE|, compiled_count_2q).

Usage
-----
    python -m pipelines.exact_bench.heron_pareto_sweep                       # defaults: FakeMarrakesh, opt {1,2}, seed {0..9}
    python -m pipelines.exact_bench.heron_pareto_sweep --backend FakeMarrakesh --output-json artifacts/json/heron_pareto_front.json
    python -m pipelines.exact_bench.heron_pareto_sweep --seeds 0,1,2,3,4 --opt-levels 1,2,3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Scaffold catalog
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScaffoldEntry:
    label: str
    path: str
    coupling: str  # "strong" or "weak"


# Excluded scaffolds:
#   - circuit_optimized_7term: DeltaE=0.338, disp_first ordering broke VQE convergence.
#
# Operator-level scaffolds: full runtime terms, fidelity-preserving.
# Term-pruned scaffolds: runtime Pauli terms dropped from 7-term locked base,
#   then reoptimized. Valid for NISQ where noise >> exact regression.
SCAFFOLD_CATALOG: tuple[ScaffoldEntry, ...] = (
    ScaffoldEntry(
        label="pruned_scaffold_11op",
        path="artifacts/json/hh_prune_nighthawk_pruned_scaffold.json",
        coupling="strong",
    ),
    ScaffoldEntry(
        label="ultra_lean_6op",
        path="artifacts/json/hh_prune_nighthawk_ultra_lean_6op.json",
        coupling="strong",
    ),
    ScaffoldEntry(
        label="aggressive_5op",
        path="artifacts/json/hh_prune_nighthawk_aggressive_5op.json",
        coupling="strong",
    ),
    ScaffoldEntry(
        label="readapt_6op",
        path="artifacts/json/hh_prune_nighthawk_readapt_6op.json",
        coupling="strong",
    ),
    ScaffoldEntry(
        label="readapt_5op",
        path="artifacts/json/hh_prune_nighthawk_readapt_5op.json",
        coupling="strong",
    ),
    ScaffoldEntry(
        label="gate_pruned_7term",
        path="artifacts/json/hh_prune_nighthawk_gate_pruned_7term.json",
        coupling="strong",
    ),
    ScaffoldEntry(
        label="marrakesh_6term_drop_IZ",
        path="artifacts/json/hh_marrakesh_fixed_scaffold_6term_drop_eyezee_20260323T171528Z.json",
        coupling="strong",
    ),
    ScaffoldEntry(
        label="weak_lean4_locked",
        path="artifacts/json/useful/L2/hh_l2_u05_g02_full_meta_class_pruned_lean4_locked_scaffold_v1.json",
        coupling="weak",
    ),
)


# ---------------------------------------------------------------------------
# Reoptimized prune-menu: pre-computed term-pruned variants with Marrakesh
# compile costs already evaluated. These do NOT need a transpile sweep —
# we ingest them directly as Pareto candidates.
# ---------------------------------------------------------------------------

REOPT_PRUNE_MENU_PATH = "artifacts/json/marrakesh_gatepruned7_reoptimized_prune_menu_20260323.json"


def load_prune_menu_candidates(
    delta_e_threshold: float = 1e-2,
) -> list[dict[str, Any]]:
    """Load reoptimized term-pruned variants as Pareto candidate rows."""
    menu_path = REPO_ROOT / REOPT_PRUNE_MENU_PATH
    if not menu_path.exists():
        return []
    with open(menu_path) as f:
        menu = json.load(f)

    candidates: list[dict[str, Any]] = []
    for row in menu.get("rows", []):
        de = row.get("optimized_abs_delta_e")
        fid = row.get("optimized_exact_state_fidelity")
        mc = row.get("marrakesh_compile_best", {})
        q2 = mc.get("compiled_count_2q")
        dep = mc.get("compiled_depth")
        sz = mc.get("compiled_size")
        if de is None or q2 is None:
            continue
        if float(de) > float(delta_e_threshold):
            continue
        tc = int(row.get("runtime_term_count", 0))
        dropped = row.get("dropped_pauli_exyz", [])
        label = f"term_pruned_{tc}t_drop_{'_'.join(dropped)}" if dropped else f"term_pruned_{tc}t_baseline"
        candidates.append({
            "label": str(label),
            "artifact_path": str(REOPT_PRUNE_MENU_PATH),
            "coupling": "strong",
            "delta_e_abs": float(de),
            "fidelity": float(fid) if fid is not None else None,
            "sweep_candidates": 0,
            "sweep_ok": 0,
            "elapsed_s": 0.0,
            "best_compiled_count_2q": int(q2),
            "best_compiled_depth": int(dep) if dep is not None else None,
            "best_compiled_size": int(sz) if sz is not None else None,
            "best_optimization_level": None,
            "best_seed_transpiler": None,
            "best_transpile_backend": "FakeMarrakesh",
            "source": "reoptimized_prune_menu",
            "runtime_term_count": int(tc),
            "dropped_pauli_exyz": list(dropped),
        })
    return candidates


# ---------------------------------------------------------------------------
# Core transpile sweep
# ---------------------------------------------------------------------------

def _load_artifact(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _extract_delta_e(payload: Mapping[str, Any]) -> float | None:
    """Derive |DeltaE| from the artifact payload."""
    adapt_vqe = payload.get("adapt_vqe", {}) if isinstance(payload, Mapping) else {}
    if isinstance(adapt_vqe, Mapping):
        ade = adapt_vqe.get("abs_delta_e")
        if ade is not None:
            return float(ade)
        energy = adapt_vqe.get("energy")
        exact = adapt_vqe.get("exact_gs_energy")
        if energy is not None and exact is not None:
            return abs(float(energy) - float(exact))
    # Try top-level keys (locked scaffold exports)
    energy = payload.get("energy")
    exact = payload.get("exact_gs_energy")
    if energy is not None and exact is not None:
        return abs(float(energy) - float(exact))
    exact_block = payload.get("exact", {})
    if isinstance(exact_block, Mapping):
        exact = exact_block.get("E_exact_sector")
    if energy is not None and exact is not None:
        return abs(float(energy) - float(exact))
    return None


def sweep_single_scaffold(
    artifact_path: Path,
    backend_name: str,
    opt_levels: Sequence[int],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    """Run compile scout for all (opt_level, seed) combos. Returns row list."""
    from pipelines.hardcoded.adapt_circuit_cost import (
        reconstruct_imported_adapt_circuit,
        _compile_candidate_row,
    )

    bundle = reconstruct_imported_adapt_circuit(_load_artifact(artifact_path))
    qc = bundle["circuit"]
    rows: list[dict[str, Any]] = []
    for opt_level in opt_levels:
        for seed in seeds:
            row = _compile_candidate_row(
                backend_request=str(backend_name),
                qc=qc,
                seed_transpiler=int(seed),
                optimization_level=int(opt_level),
            )
            row["optimization_level"] = int(opt_level)
            row["seed_transpiler"] = int(seed)
            rows.append(row)
    return rows


def best_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Pick best row by (compiled_count_2q, compiled_depth, compiled_size)."""
    ok_rows = [r for r in rows if str(r.get("transpile_status", "")) == "ok"]
    if not ok_rows:
        return None
    ok_rows.sort(
        key=lambda r: (
            int(r.get("compiled_count_2q", 10**9)),
            int(r.get("compiled_depth", 10**9)),
            int(r.get("compiled_size", 10**9)),
        )
    )
    return dict(ok_rows[0])


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------

def pareto_front(
    points: Sequence[dict[str, Any]],
    x_key: str = "delta_e_abs",
    y_key: str = "best_compiled_count_2q",
) -> list[dict[str, Any]]:
    """2D Pareto front: minimize both x_key and y_key."""
    eligible = [p for p in points if p.get(x_key) is not None and p.get(y_key) is not None]
    frontier: list[dict[str, Any]] = []
    for i, cand in enumerate(eligible):
        dominated = False
        cx, cy = float(cand[x_key]), float(cand[y_key])
        for j, other in enumerate(eligible):
            if i == j:
                continue
            ox, oy = float(other[x_key]), float(other[y_key])
            if ox <= cx and oy <= cy and (ox < cx or oy < cy):
                dominated = True
                break
        if not dominated:
            frontier.append(dict(cand))
    frontier.sort(key=lambda p: (float(p[x_key]), float(p[y_key])))
    return frontier


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Transpile-sweep L=2 scaffolds to Heron and extract Pareto front over (|DeltaE|, 2Q gates)."
    )
    p.add_argument(
        "--backend", type=str, default="FakeMarrakesh",
        help="Fake backend target for transpilation (default: FakeMarrakesh).",
    )
    p.add_argument(
        "--opt-levels", type=str, default="1,2",
        help="Comma-separated optimization levels (default: 1,2).",
    )
    p.add_argument(
        "--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9",
        help="Comma-separated transpiler seeds (default: 0-9).",
    )
    p.add_argument(
        "--coupling", type=str, default="both", choices=["strong", "weak", "both"],
        help="Which coupling regime to include (default: both).",
    )
    p.add_argument(
        "--output-json", type=Path, default=None,
        help="Output JSON path (default: artifacts/json/heron_pareto_front_<timestamp>.json).",
    )
    p.add_argument(
        "--delta-e-threshold", type=float, default=1e-2,
        help="Max |DeltaE| for term-pruned variants from reoptimized prune menu (default: 1e-2).",
    )
    return p


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    backend_name = str(args.backend)
    opt_levels = [int(x.strip()) for x in str(args.opt_levels).split(",")]
    seeds = [int(x.strip()) for x in str(args.seeds).split(",")]
    coupling_filter = str(args.coupling)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_json = args.output_json or Path(REPO_ROOT / f"artifacts/json/heron_pareto_front_{stamp}.json")
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    catalog = SCAFFOLD_CATALOG
    if coupling_filter != "both":
        catalog = tuple(e for e in catalog if e.coupling == coupling_filter)

    summary_rows: list[dict[str, Any]] = []
    all_sweep_rows: list[dict[str, Any]] = []

    for entry in catalog:
        artifact_path = REPO_ROOT / entry.path
        if not artifact_path.exists():
            print(f"SKIP {entry.label}: artifact not found at {artifact_path}")
            continue

        print(f"Sweeping {entry.label} ({entry.coupling}) ...")
        t0 = time.monotonic()

        payload = _load_artifact(artifact_path)
        delta_e = _extract_delta_e(payload)

        rows = sweep_single_scaffold(artifact_path, backend_name, opt_levels, seeds)
        elapsed = time.monotonic() - t0
        ok_count = sum(1 for r in rows if str(r.get("transpile_status", "")) == "ok")
        best = best_from_rows(rows)

        scaffold_summary: dict[str, Any] = {
            "label": str(entry.label),
            "artifact_path": str(entry.path),
            "coupling": str(entry.coupling),
            "delta_e_abs": delta_e,
            "sweep_candidates": int(len(rows)),
            "sweep_ok": int(ok_count),
            "elapsed_s": round(elapsed, 2),
        }
        if best is not None:
            scaffold_summary.update({
                "best_compiled_count_2q": int(best.get("compiled_count_2q", -1)),
                "best_compiled_depth": int(best.get("compiled_depth", -1)),
                "best_compiled_size": int(best.get("compiled_size", -1)),
                "best_optimization_level": int(best.get("optimization_level", -1)),
                "best_seed_transpiler": int(best.get("seed_transpiler", -1)),
                "best_transpile_backend": str(best.get("transpile_backend", "")),
            })
        else:
            scaffold_summary["best_compiled_count_2q"] = None

        summary_rows.append(scaffold_summary)

        for r in rows:
            all_sweep_rows.append({
                "label": str(entry.label),
                "coupling": str(entry.coupling),
                "delta_e_abs": delta_e,
                "optimization_level": r.get("optimization_level"),
                "seed_transpiler": r.get("seed_transpiler"),
                "transpile_status": str(r.get("transpile_status", "")),
                "compiled_count_2q": r.get("compiled_count_2q"),
                "compiled_depth": r.get("compiled_depth"),
                "compiled_size": r.get("compiled_size"),
                "transpile_backend": str(r.get("transpile_backend", "")),
            })

        best_2q = scaffold_summary.get("best_compiled_count_2q")
        print(
            f"  done in {elapsed:.1f}s — |DeltaE|={delta_e:.2e}, "
            f"best 2Q={best_2q}, "
            f"ok={ok_count}/{len(rows)}"
        )

    # Ingest reoptimized prune-menu candidates (pre-computed Marrakesh compile costs)
    delta_e_threshold = float(getattr(args, "delta_e_threshold", 1e-2))
    prune_menu_candidates = load_prune_menu_candidates(delta_e_threshold=delta_e_threshold)
    if prune_menu_candidates:
        print(f"\nIngesting {len(prune_menu_candidates)} term-pruned variants from reoptimized prune menu (|DeltaE| < {delta_e_threshold:.0e})")
        for pc in prune_menu_candidates:
            print(
                f"  {pc['label']:<45} |DeltaE|={pc['delta_e_abs']:.2e}  "
                f"2Q={pc['best_compiled_count_2q']}  "
                f"depth={pc.get('best_compiled_depth')}  "
                f"fid={pc.get('fidelity', 'n/a')}"
            )
        summary_rows.extend(prune_menu_candidates)

    # Pareto front
    frontier = pareto_front(summary_rows, x_key="delta_e_abs", y_key="best_compiled_count_2q")

    # Print Pareto front table
    print("\n" + "=" * 90)
    print("Pareto front: (|DeltaE|, compiled_count_2q)")
    print("=" * 90)
    print(f"{'Label':<48} {'|DeltaE|':>12} {'2Q gates':>10} {'Depth':>8} {'Coupling':>10}")
    print("-" * 90)
    for pt in frontier:
        print(
            f"{pt['label']:<48} {pt['delta_e_abs']:>12.2e} "
            f"{pt['best_compiled_count_2q']:>10d} "
            f"{pt.get('best_compiled_depth', -1):>8d} "
            f"{pt['coupling']:>10}"
        )
    print("-" * 90)
    print(f"Total candidates: {len(summary_rows)}, Pareto-optimal: {len(frontier)}")

    # Dominated points
    dominated = [s for s in summary_rows if s not in frontier and s.get("best_compiled_count_2q") is not None]
    if dominated:
        print("\nDominated points:")
        for pt in sorted(dominated, key=lambda p: float(p.get("delta_e_abs") or 1e9)):
            print(
                f"  {pt['label']:<48} |DeltaE|={pt['delta_e_abs']:.2e}  "
                f"2Q={pt.get('best_compiled_count_2q')}  "
                f"depth={pt.get('best_compiled_depth')}"
            )

    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "heron_pareto_sweep_v1",
        "backend": str(backend_name),
        "opt_levels": list(opt_levels),
        "seeds": list(seeds),
        "coupling_filter": str(coupling_filter),
        "scaffold_count": int(len(summary_rows)),
        "frontier_count": int(len(frontier)),
        "pareto_frontier": frontier,
        "all_scaffolds": summary_rows,
        "sweep_detail": all_sweep_rows,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True, default=str)
    print(f"\nOutput: {output_json}")
    return result


if __name__ == "__main__":
    run()
