"""Run the predeclared early-pruning diagnostic with bounded parallelism."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUN_ARM = HERE / "run_arm.py"

VARIANTS = (
    {
        "label": "prune_protect0",
        "theta": 1.0e-3,
        "protect": 0,
        "max_regression": 1.0e-8,
        "retained_gain_ratio": 0.5,
    },
    {
        "label": "prune_theta1e-2_protect0",
        "theta": 1.0e-2,
        "protect": 0,
        "max_regression": 1.0e-8,
        "retained_gain_ratio": 0.5,
    },
    {
        "label": "prune_theta1e-1_protect0",
        "theta": 1.0e-1,
        "protect": 0,
        "max_regression": 1.0e-8,
        "retained_gain_ratio": 0.5,
    },
    {
        "label": "prune_theta1e-1_protect1",
        "theta": 1.0e-1,
        "protect": 1,
        "max_regression": 1.0e-8,
        "retained_gain_ratio": 0.5,
    },
    {
        "label": "prune_theta1e-1_eps1e-6",
        "theta": 1.0e-1,
        "protect": 0,
        "max_regression": 1.0e-6,
        "retained_gain_ratio": 0.5,
    },
    {
        "label": "prune_theta1e-1_zeta025",
        "theta": 1.0e-1,
        "protect": 0,
        "max_regression": 1.0e-8,
        "retained_gain_ratio": 0.25,
    },
)


def _apply_phase_reuse_accounting(
    summary: dict[str, object],
    summary_path: Path,
) -> dict[str, object]:
    ledger_path = summary_path.parent / "estimator_ledger.json"
    accounting = json.loads(ledger_path.read_text())["accounting"]
    # Every sweep arm prunes, so rejected delete--refit siblings remain real
    # work and must not be dropped with the losing lineage.
    occurrence_scope = accounting.get("all_branch_search_work") or accounting
    unique_scope = accounting.get("all_branch_unique_primitive_diagnostic") or {}
    raw = int(occurrence_scope.get("S_alg", accounting["S_alg"]))
    corrected = int(
        occurrence_scope.get("N_H_outer", 0)
        + occurrence_scope.get("N_H_refit", 0)
        + unique_scope.get("N_grad", occurrence_scope.get("N_grad", 0))
        + unique_scope.get("N_metric", occurrence_scope.get("N_metric", 0))
    )
    normalized = {
        **summary,
        "S_alg": corrected,
        "S_alg_raw": raw,
        "S_alg_corrected": corrected,
        "S_alg_removed_staged_reuse": raw - corrected,
        "S_alg_accounting_scope": "all_executed_delete_refit_work",
    }
    summary_path.write_text(json.dumps(normalized, indent=2))
    return normalized


def _run_one(
    variant: dict[str, float | int | str],
    output_root: Path,
    horizon: int,
) -> dict[str, object]:
    label = str(variant["label"])
    summary_path = output_root / f"{label}_k{horizon}" / "summary.json"
    if summary_path.exists():
        return _apply_phase_reuse_accounting(
            json.loads(summary_path.read_text()),
            summary_path,
        )
    command = [
        sys.executable,
        str(RUN_ARM),
        "metric_ablation_gated",
        str(output_root),
        "--horizon",
        str(horizon),
        "--variant-label",
        label,
        "--prune-theta",
        str(variant["theta"]),
        "--prune-protect",
        str(variant["protect"]),
        "--prune-max-regression",
        str(variant["max_regression"]),
        "--prune-retained-gain-ratio",
        str(variant["retained_gain_ratio"]),
    ]
    completed = subprocess.run(
        command,
        cwd=HERE.parents[2],
        check=True,
        capture_output=True,
        text=True,
    )
    if not summary_path.exists():
        raise RuntimeError(
            f"{label} completed without writing {summary_path}: "
            f"{completed.stdout}\n{completed.stderr}"
        )
    return _apply_phase_reuse_accounting(
        json.loads(summary_path.read_text()),
        summary_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=HERE / "results" / "prune_sweep",
    )
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(2, args.workers))) as pool:
        futures = {
            pool.submit(_run_one, variant, args.output_root, args.horizon): variant
            for variant in VARIANTS
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

    results.sort(
        key=lambda row: (
            row.get("first_accepted_deletion_round") is None,
            row.get("first_accepted_deletion_round") or 10**9,
            row.get("error") or float("inf"),
            row.get("Dc") or 10**9,
        )
    )
    (args.output_root / "sweep_summary.json").write_text(
        json.dumps(results, indent=2)
    )


if __name__ == "__main__":
    main()
