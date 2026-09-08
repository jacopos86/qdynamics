"""Matched local selector diagnostics, using ordinary full trajectories.

Planning, execution, and reporting surround the existing mathematical kernels.
Exact propagation is imported only by the post-run report.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
from itertools import product
import json
from pathlib import Path
import subprocess
import sys


POLICIES = ("greedy", "bounded_exhaustive", "block_exchange")
GAINS = ("joint_batch", "singleton_factorized")
ADMISSIONS = {"singleton": (1, 1), "batch": (4, None)}
SEED = "chtc/paper_ii_production_v2/input/seeds/hubbard_snake_nph1.json"
SEED_SHA256 = "637b694a8e6ccde1511b8a33a6b10678277cbedfab6c855a63c52efed07cb339"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def source_hashes(repo: Path) -> dict[str, str]:
    """Fingerprint executed source, including any local changes, without git status."""
    files = set()
    for directory in ("pipelines", "src"):
        files.update((repo / directory).rglob("*.py"))
    return {str(path.relative_to(repo)): file_hash(path) for path in sorted(files)}


def pilot_case(repo: Path, root: Path, *, policy: str, gain: str,
               admission: str, budget: int, t_final: float = 0.125,
               num_times: int = 4) -> dict:
    from pipelines.time_dynamics.paper_ii_runs import Horizon, build_selector_comparison_run
    from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import _build_parser

    if budget < 1:
        raise ValueError("budget must be positive")
    a, d = ADMISSIONS[admission]
    case_id = f"{policy}__{gain}__{admission}__b{budget}"
    output = root / case_id / "run.json"
    run = build_selector_comparison_run(
        search_policy=policy, gain_evaluation=gain,
        insertion_cardinality=a, deletion_cardinality=d,
        seed_path=str(repo / SEED), drive="fastweak", horizon="smoke",
        numerics="rk4_ridge1e-7", search_profile="diagnostic_fast_v1",
        output_json=str(output),
        extra_flags=(
            "--replay-candidate-pool-mode", "diagnostic_replay_family_pool",
            "--require-complete-candidate-pool", "--record-statevector",
            "--max-compiled-patches-per-checkpoint", str(budget),
            "--compile-workers", "1", "--support-patch-scoring-workers", "1",
        ),
    )
    run = replace(
        run, horizon=Horizon("selector_local_prefix", t_final, num_times),
        search_profile=replace(
            run.search_profile, profile_id=case_id,
            joint_patch_evaluations=budget, rounds_per_checkpoint=1,
        ),
    )
    argv = run.argv()
    effective = vars(_build_parser().parse_args(list(argv)))
    expected = {
        "search_policy": policy, "gain_evaluation": gain,
        "max_insertion_batch_size": a, "max_deletion_batch_size": d,
        "max_joint_patch_evaluations": budget,
        "max_compiled_patches_per_checkpoint": budget,
        "max_insertion_rounds_per_checkpoint": 1,
        "require_complete_candidate_pool": True,
        "exchange_route_bundle": "paper_ii_exchange_q_compiled_v1",
    }
    for key, value in expected.items():
        if effective[key] != value:
            raise ValueError(f"pilot setting changed: {key}")
    return {
        "case_id": case_id, "runner_argv": list(argv),
        "worker_count": 1, "blas_threads": 1,
        "artifact_json": str(repo / SEED), "output_json": str(output),
        "description": "Local full-trajectory diagnostic; exact references are post-run only.",
        "metadata": {
            "policy": policy, "gain": gain, "admission": admission, "budget": budget,
            "max_A": a, "max_D": d,
            "run_settings": asdict(run), "effective_arguments": effective,
        },
    }


def make_plan(repo: Path, root: Path, *, budgets: tuple[int, ...],
              calibration: bool = False, execution_authorized: bool = False) -> dict:
    seed = repo / SEED
    if file_hash(seed) != SEED_SHA256:
        raise ValueError("the diagnosed seed hash changed")
    if calibration:
        if len(budgets) != 1:
            raise ValueError("calibration takes one preliminary budget")
        axes = [("bounded_exhaustive", "joint_batch", "batch", budgets[0])]
    else:
        if len(budgets) != 2 or not 0 < budgets[0] < budgets[1]:
            raise ValueError("the pilot needs distinct positive low/high budgets")
        axes = product(POLICIES, GAINS, ADMISSIONS, budgets)
    return {
        "schema": "paper_ii_selector_local_diagnostic_v1",
        "classification": "diagnostic", "calibration": calibration,
        "execution_authorized": execution_authorized,
        "repo": str(repo), "output_root": str(root),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "source_hashes": source_hashes(repo), "seed_sha256": SEED_SHA256,
        "pool": {"complete": True, "deduplicated_operator_count_at_load": 22,
                 "fallback_used": False, "truncated": False},
        "scope": "Hubbard dimer, existing fastweak drive; t=0..0.125, 4 reporting points.",
        "budget_fields": ["max_joint_patch_evaluations", "max_compiled_patches_per_checkpoint"],
        "coverage_scope": "All recorded selector calls and passes at every checkpoint.",
        "round_limit": "Existing rounds_per_checkpoint=1 permits up to two selector calls; held fixed across cells.",
        "limits": {"timeout_seconds_per_case": 600, "ram_mb_per_case": 3000},
        "cases": [pilot_case(repo, root / "runs", policy=p, gain=g, admission=a, budget=b)
                  for p, g, a, b in axes],
    }


def execute_plan(plan: dict, *, python: str) -> None:
    from pipelines.time_dynamics.diagnostics.selector_local_executor import execute_case

    repo = Path(plan["repo"])
    if not plan["execution_authorized"]:
        raise ValueError("execution is not authorized")
    if source_hashes(repo) != plan["source_hashes"] or file_hash(repo / SEED) != plan["seed_sha256"]:
        raise ValueError("source or seed changed after the plan was frozen")
    for case in plan["cases"]:
        case_dir = Path(case["output_json"]).parent
        print(f"[pilot] start {case['case_id']}", flush=True)
        receipt = execute_case(
            case_dir=case_dir, case=case, python=python, repo_root=repo,
            timeout_seconds=plan["limits"]["timeout_seconds_per_case"],
            limit_mb=plan["limits"]["ram_mb_per_case"],
        )
        print(f"[pilot] done {case['case_id']} {receipt['status']}", flush=True)


def search_coverage(run: dict) -> dict:
    patches, permissions, reasons = Counter(), Counter(), Counter()
    decisions = []
    guards = []
    for point in run["trajectory"]["points"]:
        decision = point.get("patch_decision")
        if not decision:
            continue
        metadata = decision.get("metadata", {})
        for call in metadata.get("selection_round_work", []):
            decisions.append({"time": point["time"], "call": call["selector_call_index"],
                              "accepted": call["accepted"], "kind": call["patch_kind"],
                              "reason": call["reason"], "committed": call.get("committed")})
            guards.append({"time": point["time"], "call": call["selector_call_index"],
                           "work": call.get("work_guard"), "compile": call.get("compile_guard")})
            for search in call["search_receipts"]:
                reasons[search["stop_reason"]] += 1
                coverage = search["coverage"]
                for row in coverage["evaluated_patches_by_cardinality"]:
                    patches[row["deletion_size"], row["insertion_size"]] += row["count"]
                for row in coverage["visited_deletion_sets_by_cardinality"]:
                    for outcome in ("permitted", "refused"):
                        permissions[row["deletion_size"], outcome] += row[outcome]
    return {
        "scope": "all_recorded_selector_calls_and_passes",
        "evaluated_patches": [{"D": d, "A": a, "count": n} for (d, a), n in sorted(patches.items())],
        "visited_deletion_sets": [{"D": d, "outcome": outcome, "count": n}
                                  for (d, outcome), n in sorted(permissions.items())],
        "stop_reasons": dict(reasons), "decisions": decisions, "guards": guards,
    }


def report_plan(plan: dict) -> dict:
    from pipelines.time_dynamics.ap_mclachlan.exchange_config import CompileContract
    from pipelines.time_dynamics.diagnostics.ap_terminal_qiskit_cost import build_ap_terminal_cost_table
    from pipelines.time_dynamics.diagnostics.paper_ii_four_route_comparison import complete_estimator_work
    from pipelines.time_dynamics.diagnostics.physical_errors import physical_errors_from_artifacts

    root = Path(plan["output_root"])
    rows = []
    contract = CompileContract()
    for case in plan["cases"]:
        path = Path(case["output_json"])
        directory = path.parent
        receipt_path = directory / "execution.json"
        execution = (json.loads(receipt_path.read_text()) if receipt_path.exists() else
                     {"status": "missing_receipt" if (directory / "case.json").exists() else "not_started"})
        row = {k: case["metadata"][k] for k in ("policy", "gain", "admission", "budget", "max_A", "max_D")}
        row.update(case_id=case["case_id"], execution=execution)
        measurement_path = directory / "measurement.json"
        measurement = json.loads(measurement_path.read_text()) if measurement_path.exists() else {}
        row["measurement"] = {k: measurement.get(k) for k in ("ok", "error", "wall_seconds", "cpu_seconds", "peak_rss_bytes")}
        if path.exists() and measurement.get("ok"):
            run = json.loads(path.read_text())
            row.update(
                run_sha256=file_hash(path), summary=run["summary"],
                support=[p["runtime_parameter_count"] for p in run["plot_rows"]],
                coverage=search_coverage(run),
                estimator_work=complete_estimator_work([run], route="q_compiled"),
                errors=physical_errors_from_artifacts(
                    run_path=path, seed_path=case["artifact_json"],
                    reference_store_root=root / "exact_reference_store").to_json_dict(),
                terminal_resources=build_ap_terminal_cost_table(
                    trajectory_path=path, label=case["case_id"],
                    backend_name=contract.backend_name, seed_transpiler=contract.transpile_seed,
                    optimization_level=contract.optimization_level,
                    source_artifact_path=Path(case["artifact_json"]))["rows"][0],
            )
        write_json(directory / "metrics.json", row)
        rows.append(row)
    result = {"classification": "diagnostic", "plan_sha256": file_hash(root / "plan.json"), "rows": rows}
    write_json(root / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("plan", "execute", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--budgets", type=int, nargs="+", default=())
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--execution-authorized", action="store_true",
                        help="Record already obtained authorization when freezing the plan.")
    args = parser.parse_args()
    path = args.root.resolve() / "plan.json"
    if args.operation == "plan":
        write_json(path, make_plan(args.repo.resolve(), args.root.resolve(),
                   budgets=tuple(args.budgets), calibration=args.calibration,
                   execution_authorized=args.execution_authorized), exclusive=True)
    elif args.operation == "execute":
        execute_plan(json.loads(path.read_text()), python=sys.executable)
    else:
        report_plan(json.loads(path.read_text()))


if __name__ == "__main__":
    main()
