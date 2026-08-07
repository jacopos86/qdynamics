#!/usr/bin/env python3
"""Generate Paper-I HH ordered batch-beam SNAKE diagnostics.

This is a narrow diagnostic generator for the newly implemented ordered
batch-beam SNAKE path.  It reuses the current full-meta/HVA-included Phase-III
singleton hard-guard SNAKE row, then changes only the user-approved batch/beam
controls.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chtc.phase3_optuna import generate_paper_i_hh_fullmeta_phase3_singleton_rotosolve_records as base


DEFAULT_BATCH_ID = "paper_i_hh_fullmeta_singleton_symmetry_ordered_batch_beam_weakweak_powell_20260704_v3"
DEFAULT_SMOKE_BATCH_ID = (
    "paper_i_hh_fullmeta_singleton_symmetry_ordered_batch_beam_weakweak_powell_smoke_20260703_v1"
)
DEFAULT_LAMBDAS = (0.0, 0.01, 0.025, 0.10)
DEFAULT_BATCH_MODES = ("greedy_reduced_plane", "combinatorial_reduced_plane")
RUN_CLASS = "diagnostic"
DEFAULT_REGIMES = ("weak-weak",)
METHOD = "snake"
OPTIMIZER_OVERLAY_ID = "powell"
MATRIX_LABEL = "A_native_staged_singleton_hard_guard"
TARGET_SIZE = "5"
SIZE_CAP = "5"
BEAM_LIVE_BRANCHES = "3"
BEAM_CHILDREN_PER_PARENT = "3"
PHASE3_RUNTIME_SPLIT_MAX_SUBSET_SIZE = "3"
PARALLEL_GRADIENT_WORKERS = "4"
BEAM_PARENT_WORKERS = "3"


def _lambda_token(value: float) -> str:
    text = f"{float(value):.6g}".replace("-", "m").replace(".", "p")
    return text


def _mode_token(mode: str) -> str:
    return "greedy" if mode == "greedy_reduced_plane" else "combinatorial"


def _regime_token(regime: str) -> str:
    return str(regime).replace("-", "_")


def _record_id(
    batch_id: str,
    *,
    regime: str,
    mode: str,
    lambda_beam: float,
    budget: int,
    max_depth: int,
) -> str:
    return (
        f"{batch_id}__{_regime_token(regime)}__snake__ordered_batch_beam__{_mode_token(mode)}"
        f"__lambda_{_lambda_token(lambda_beam)}__powell{int(budget)}__depth{int(max_depth)}"
        "__fullmeta_hva_phase3_singleton"
    )


def _merge_ordered_batch_beam_overrides(
    row: Mapping[str, str],
    *,
    mode: str,
    lambda_beam: float,
) -> dict[str, Any]:
    payload = json.loads(str(row["snake_cli_overrides_json"]))
    set_flags = dict(payload.get("set_flags") or {})
    set_flags.update(
        {
            "--static-route-id": "unspecified",
            "--phase3-runtime-split-max-subset-size": PHASE3_RUNTIME_SPLIT_MAX_SUBSET_SIZE,
            "--phase2-batch-selection-mode": str(mode),
            "--phase3-batch-selection-mode": str(mode),
            "--phase2-batch-target-size": TARGET_SIZE,
            "--phase2-batch-size-cap": SIZE_CAP,
            "--adapt-parallel-gradient-workers": PARALLEL_GRADIENT_WORKERS,
            "--adapt-beam-parent-workers": BEAM_PARENT_WORKERS,
            "--adapt-beam-live-branches": BEAM_LIVE_BRANCHES,
            "--adapt-beam-children-per-parent": BEAM_CHILDREN_PER_PARENT,
            "--adapt-beam-lambda": f"{float(lambda_beam):.12g}",
        }
    )
    enable_flags = list(payload.get("enable_flags") or [])
    for flag in ("--phase2-enable-batching",):
        if flag not in enable_flags:
            enable_flags.append(flag)
    remove_bool_flags = list(payload.get("remove_bool_flags") or [])
    for flag in ("--phase2-no-batching",):
        if flag not in remove_bool_flags:
            remove_bool_flags.append(flag)
    payload["set_flags"] = set_flags
    payload["enable_flags"] = enable_flags
    payload["remove_bool_flags"] = remove_bool_flags
    return payload


def _expected_diagnostics(mode: str, lambda_beam: float) -> dict[str, Any]:
    return {
        "ordered_batch_beam_enabled": True,
        "phase2_batch_selection_mode": str(mode),
        "prune_key_version": "beam_energy_cost_pareto_lambda_v1",
        "survival_policy_version": "beam_pairwise_energy_cost_pareto_v1",
        "lambda_beam": float(lambda_beam),
        "history_beam_structural_mode": "ordered_batch_admission",
        "batch_size_gt_one_when_batching_fires": True,
        "phase3_batch_score_formula": "phase3_batch_delta_e3 / phase3_batch_denominator_1_plus_K3",
        "requires_beam_cost_K": True,
        "requires_survival_audits": True,
        "adapt_parallel_gradient_workers": int(PARALLEL_GRADIENT_WORKERS),
        "adapt_beam_parent_workers": int(BEAM_PARENT_WORKERS),
        "beam_parent_parallel_requested": True,
        "batch_warm_start_expected_reason": "batch_warm_start_not_available",
    }


def build_records(
    batch_id: str,
    *,
    regimes: Sequence[str] = DEFAULT_REGIMES,
    lambdas: Sequence[float] = DEFAULT_LAMBDAS,
    batch_modes: Sequence[str] = DEFAULT_BATCH_MODES,
    budget: int = 15,
    max_depth: int = 15,
    smoke_only: bool = False,
) -> list[dict[str, str]]:
    if any(float(value) < 0.0 for value in lambdas):
        raise ValueError("lambda values must be non-negative.")
    unknown_modes = [mode for mode in batch_modes if mode not in DEFAULT_BATCH_MODES]
    if unknown_modes:
        raise ValueError(f"Unknown batch modes: {unknown_modes}; expected {DEFAULT_BATCH_MODES}")
    selected_regimes = tuple(str(regime) for regime in regimes)
    unknown_regimes = [regime for regime in selected_regimes if regime not in set(base.REGIME_ORDER)]
    if unknown_regimes:
        raise ValueError(f"Unknown regimes: {unknown_regimes}; expected subset of {base.REGIME_ORDER}")

    sources = base.source_rows()
    missing_sources = [regime for regime in selected_regimes if (regime, METHOD) not in sources]
    if missing_sources:
        raise ValueError(f"Missing source rows for ordered batch-beam regimes: {missing_sources}")
    overlay = base.OPTIMIZER_OVERLAYS[OPTIMIZER_OVERLAY_ID]
    policy = base.MatrixPolicy(
        MATRIX_LABEL,
        "A1 Phase-III singleton hard-guard route with ordered batch-beam diagnostics",
        "native_phase3_singleton",
        "hard_guard",
    )

    planned = [(mode, float(lambda_beam)) for mode in batch_modes for lambda_beam in lambdas]
    if smoke_only:
        planned = planned[:1]

    rows: list[dict[str, str]] = []
    for regime in selected_regimes:
        source = sources[(regime, METHOD)]
        for mode, lambda_beam in planned:
            row = base.make_row(
                batch_id,
                source,
                policy=policy,
                budget=int(budget),
                max_depth=int(max_depth),
                overlay=overlay,
                strong_strong_snake_start_mode=base.STRONG_STRONG_SNAKE_START_MODE_DEPTH_ZERO,
            )
            record_id = _record_id(
                batch_id,
                regime=regime,
                mode=mode,
                lambda_beam=lambda_beam,
                budget=int(budget),
                max_depth=int(max_depth),
            )
            row["record_id"] = record_id
            row.update(base.output_paths(record_id, METHOD))
            row["run_class"] = RUN_CLASS
            row["static_route_id"] = "unspecified"
            row["matrix_role"] = policy.role
            row["child_subset_size"] = PHASE3_RUNTIME_SPLIT_MAX_SUBSET_SIZE
            row["snake_phase3_runtime_split_max_subset_size"] = PHASE3_RUNTIME_SPLIT_MAX_SUBSET_SIZE
            row["phase3_adapt_parallel_gradient_workers"] = PARALLEL_GRADIENT_WORKERS
            row["phase3_adapt_beam_parent_workers"] = BEAM_PARENT_WORKERS
            row["ordered_batch_beam_label"] = f"{_mode_token(mode)}_lambda_{_lambda_token(lambda_beam)}"
            row["ordered_batch_beam_enabled"] = "true"
            row["ordered_batch_beam_run_role"] = "smoke" if smoke_only else "lambda_mode_matrix"
            row["phase2_batch_selection_mode"] = str(mode)
            row["phase2_batch_target_size"] = TARGET_SIZE
            row["phase2_batch_size_cap"] = SIZE_CAP
            row["adapt_beam_live_branches"] = BEAM_LIVE_BRANCHES
            row["adapt_beam_children_per_parent"] = BEAM_CHILDREN_PER_PARENT
            row["adapt_beam_lambda"] = f"{float(lambda_beam):.12g}"
            row["ordered_batch_beam_expected_diagnostics_json"] = json.dumps(
                _expected_diagnostics(mode, lambda_beam),
                sort_keys=True,
                separators=(",", ":"),
            )
            row["snake_cli_overrides_json"] = json.dumps(
                _merge_ordered_batch_beam_overrides(row, mode=mode, lambda_beam=lambda_beam),
                sort_keys=True,
                separators=(",", ":"),
            )
            changed = [field for field in row["changed_fields_vs_anchor"].split(",") if field]
            changed.extend(
                [
                    "ordered_batch_beam_enabled",
                    "phase2_batch_selection_mode",
                    "phase2_batch_target_size",
                    "phase2_batch_size_cap",
                    "phase3_runtime_split_max_subset_size",
                    "adapt_parallel_gradient_workers",
                    "adapt_beam_parent_workers",
                    "adapt_beam_live_branches",
                    "adapt_beam_children_per_parent",
                    "adapt_beam_lambda",
                    "static_route_id",
                ]
            )
            row["changed_fields_vs_anchor"] = ",".join(dict.fromkeys(changed))
            row["source_settings_status"] = "ok_diagnostic_ordered_batch_beam_fullmeta_hva"
            row["schedule_source_policy"] = "diagnostic_ordered_batch_beam_powell_fullmeta_hva"
            row["source_contract_note"] = (
                "Diagnostic ordered batch-beam SNAKE row. Pool contract is full_meta_unfiltered "
                "with HVA included; Pauli children use the A1 Phase-III archival child-set hard-guard "
                "route. This row varies only the approved ordered batch-beam controls and "
                "declares static_route_id=unspecified because the ordered batch-beam selection "
                "mode is intentionally not canonical Route A reduced_plane."
            )
            rows.append(row)
    return rows


def write_records(
    batch_id: str,
    records: Sequence[dict[str, str]],
    *,
    budget: int,
    max_depth: int,
    request_cpus: int,
    request_memory_mb: int,
    request_disk_mb: int,
    max_runtime_s: int,
) -> dict[str, Any]:
    manifest = base.write_records(
        batch_id,
        records,
        budget=int(budget),
        max_depth=int(max_depth),
        request_cpus=int(request_cpus),
        request_memory_mb=int(request_memory_mb),
        request_disk_mb=int(request_disk_mb),
        max_runtime_s=int(max_runtime_s),
    )
    manifest["schema"] = "paper_i_hh_ordered_batch_beam_diagnostic_manifest_v1"
    manifest["run_class"] = RUN_CLASS
    manifest["ordered_batch_beam_diagnostic"] = {
        "schema": "paper_i_hh_ordered_batch_beam_diagnostic_contract_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "regimes": sorted({str(row["display_regime"]) for row in records}),
        "method": METHOD,
        "optimizer": "POWELL",
        "pool_contract": "full_meta_unfiltered",
        "hva_policy": "included_unfiltered_full_meta",
        "pauli_child_policy": "A1 Phase-III archival child-set hard_guard",
        "phase3_runtime_split_max_subset_size": int(PHASE3_RUNTIME_SPLIT_MAX_SUBSET_SIZE),
        "adapt_parallel_gradient_workers": int(PARALLEL_GRADIENT_WORKERS),
        "adapt_beam_parent_workers": int(BEAM_PARENT_WORKERS),
        "batch_modes": sorted({row["phase2_batch_selection_mode"] for row in records}),
        "lambda_grid": [float(row["adapt_beam_lambda"]) for row in records],
        "maxiter": int(budget),
        "max_depth": int(max_depth),
        "changed_settings": [
            "--static-route-id",
            "--phase2-enable-batching",
            "--phase2-batch-selection-mode",
            "--phase3-batch-selection-mode",
            "--phase2-batch-target-size",
            "--phase2-batch-size-cap",
            "--phase3-runtime-split-max-subset-size",
            "--adapt-parallel-gradient-workers",
            "--adapt-beam-parent-workers",
            "--adapt-beam-live-branches",
            "--adapt-beam-children-per-parent",
            "--adapt-beam-lambda",
        ],
        "expected_runtime_diagnostics": _expected_diagnostics("greedy_reduced_plane", 0.0),
        "paper_facing_status": "diagnostic_not_paper_facing",
    }
    manifest["source_contract"]["paper_facing_status"] = "diagnostic_not_paper_facing"
    manifest["source_contract"]["strong_strong_snake_start_mode"] = (
        base.STRONG_STRONG_SNAKE_START_MODE_DEPTH_ZERO
    )
    snake_child_policy = manifest["source_contract"].get("snake_child_policy")
    if isinstance(snake_child_policy, dict):
        snake_child_policy["phase3_runtime_split_max_subset_size"] = int(
            PHASE3_RUNTIME_SPLIT_MAX_SUBSET_SIZE
        )
        snake_child_policy["phase3_runtime_split_contract_note"] = (
            "Ordered batch-beam diagnostic uses the recovered Paper-I archival "
            "Phase-III child-set route with subset cap 3, not singleton cap 1."
        )
    for matrix_policy in manifest["source_contract"].get("matrix_policies", []) or []:
        if isinstance(matrix_policy, dict) and matrix_policy.get("matrix_label") == MATRIX_LABEL:
            matrix_policy["child_subset_size"] = int(PHASE3_RUNTIME_SPLIT_MAX_SUBSET_SIZE)
            matrix_policy["child_policy_note"] = (
                "Row field name remains native_phase3_singleton for compatibility; "
                "effective Phase-III runtime split max subset size is 3."
            )
    manifest["source_contract"]["note"] = (
        "SNAKE ordered batch-beam diagnostic matrix. Uses full_meta_unfiltered "
        "with HVA included and the A1 Phase-III archival child-set hard-guard route; varies "
        "only the user-approved ordered batch-beam controls."
    )
    manifest_path = ROOT / "chtc" / "phase3_optuna" / "input" / batch_id / "paper_i_hh_spsa_budget_ladder_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    parser.add_argument("--budget", type=int, default=15)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument(
        "--regime",
        action="append",
        choices=base.REGIME_ORDER,
        help="Regime to include. May be repeated. Defaults to weak-weak for smoke compatibility.",
    )
    parser.add_argument("--lambda-beam", type=float, action="append", default=[])
    parser.add_argument("--batch-mode", choices=DEFAULT_BATCH_MODES, action="append", default=[])
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--request-cpus", type=int, default=2)
    parser.add_argument("--request-memory-mb", type=int, default=32768)
    parser.add_argument("--request-disk-mb", type=int, default=61440)
    parser.add_argument("--max-runtime-s", type=int, default=172800)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    batch_id = str(args.batch_id)
    budget = int(args.budget)
    max_depth = int(args.max_depth)
    if budget < 1 or max_depth < 1:
        raise ValueError("--budget and --max-depth must be positive.")
    records = build_records(
        batch_id,
        regimes=tuple(args.regime or DEFAULT_REGIMES),
        lambdas=tuple(args.lambda_beam or DEFAULT_LAMBDAS),
        batch_modes=tuple(args.batch_mode or DEFAULT_BATCH_MODES),
        budget=budget,
        max_depth=max_depth,
        smoke_only=bool(args.smoke_only),
    )
    manifest = write_records(
        batch_id,
        records,
        budget=budget,
        max_depth=max_depth,
        request_cpus=int(args.request_cpus),
        request_memory_mb=int(args.request_memory_mb),
        request_disk_mb=int(args.request_disk_mb),
        max_runtime_s=int(args.max_runtime_s),
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "records"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
