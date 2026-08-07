from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


FIXTURE_SHA256 = "570690bd126787305b340bd2f7493499c0f3101e3e2820c2d355c55c16afa594"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--command-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    runtime_source = root / "runtime_source"
    if str(runtime_source) not in sys.path:
        sys.path.insert(0, str(runtime_source))
    from pipelines.static_adapt.cli_config import _build_adapt_arg_parser
    from pipelines.static_adapt.sr_snake_route_profile import (
        CANONICAL_SR_SNAKE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3_EXECUTION_SETTINGS,
        SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3,
        canonical_sr_snake_h2o_derivative_resolved_paper_i_v3_contract_sha256,
    )

    command = json.loads(args.command_json.read_text(encoding="utf-8"))
    argv = list(command["argv"])
    parsed = _build_adapt_arg_parser(
        adapt_gradient_parity_rtol=1.0e-8
    ).parse_args(argv[4:])
    expected = (
        CANONICAL_SR_SNAKE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3_EXECUTION_SETTINGS
    )
    mismatches = {
        field: {"expected": value, "actual": getattr(parsed, field, None)}
        for field, value in expected.items()
        if getattr(parsed, field, None) != value
    }
    fixture = root / "runtime_inputs/h2o_fixture.json"
    checks = {
        "fixture_hash": fixture.is_file() and sha256(fixture) == FIXTURE_SHA256,
        "route_profile_request": parsed.sr_route_profile_request
        == SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3,
        "route_profile_resolved": parsed.sr_route_profile_resolved
        == SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3,
        "route_contract_digest": parsed.sr_route_profile_contract_sha256
        == canonical_sr_snake_h2o_derivative_resolved_paper_i_v3_contract_sha256(),
        "route_contract_matches_submit_manifest": (
            parsed.sr_route_profile_contract_sha256
            == command.get("route_profile_contract_sha256")
        ),
        "profile_settings": not mismatches,
        "chemical_accuracy_stop": (
            parsed.adapt_benchmark_target_abs_delta_e == 0.0016
        ),
        "route_difference_audit": command.get("paper_i_route_difference_audit", {}).get(
            "status"
        )
        == "pass",
        "effective_beam_disabled": (
            parsed.adapt_beam_live_branches == 1
            and parsed.adapt_beam_children_per_parent == 1
            and parsed.adapt_beam_terminated_keep == 0
            and parsed.adapt_beam_terminal_archive_mode == "disabled"
        ),
        "pruning_disabled": parsed.phase1_prune_enabled is False,
        "ordinary_novelty_disabled_with_fallback_retained": (
            parsed.phase3_novelty_ablation_mode == "off"
            and parsed.phase2_gram_novelty_policy == "fallback_only_v1"
            and parsed.phase3_gram_novelty_policy == "fallback_only_v1"
        ),
        "batching_disabled": (
            parsed.phase2_enable_batching is False
            and parsed.phase3_enable_batching is False
        ),
    }
    payload = {
        "schema": (
            "paper_iv_h2o_sr_paper_i_noprune_nobeam_derivative_resolved_"
            "depth50_runtime_preflight_v1"
        ),
        "status": "pass" if all(checks.values()) else "blocked",
        "checks": checks,
        "profile_setting_mismatches": mismatches,
        "effective_settings": {
            "route_family": parsed.sr_route_profile_contract["route_family"],
            "route_profile": parsed.sr_route_profile_resolved,
            "route_contract_sha256": parsed.sr_route_profile_contract_sha256,
            "problem": parsed.problem,
            "adapt_pool": parsed.adapt_pool,
            "adapt_max_depth": parsed.adapt_max_depth,
            "adapt_benchmark_target_abs_delta_e": (
                parsed.adapt_benchmark_target_abs_delta_e
            ),
            "phase3_novelty_ablation_mode": parsed.phase3_novelty_ablation_mode,
            "phase2_gram_novelty_policy": parsed.phase2_gram_novelty_policy,
            "phase3_gram_novelty_policy": parsed.phase3_gram_novelty_policy,
            "phase2_enable_batching": parsed.phase2_enable_batching,
            "phase3_enable_batching": parsed.phase3_enable_batching,
            "phase3_runtime_split_subset_sizes": (
                parsed.phase3_runtime_split_subset_sizes
            ),
            "phase3_runtime_split_child_padding_policy": (
                parsed.phase3_runtime_split_child_padding_policy
            ),
            "adapt_beam_live_branches": parsed.adapt_beam_live_branches,
            "adapt_beam_children_per_parent": parsed.adapt_beam_children_per_parent,
            "phase1_prune_enabled": parsed.phase1_prune_enabled,
            "phase1_prune_schur_nomination_route": (
                parsed.phase1_prune_schur_nomination_route
            ),
            "phase1_prune_metric_schur_mu": parsed.phase1_prune_metric_schur_mu,
            "adapt_inner_optimizer": parsed.adapt_inner_optimizer,
            "phase3_backend_cost_mode": parsed.phase3_backend_cost_mode,
            "adapt_maxiter": parsed.adapt_maxiter,
            "adapt_accepted_refit_scope": parsed.adapt_accepted_refit_scope,
            "adapt_accepted_refit_coordinate_chart": (
                parsed.adapt_accepted_refit_coordinate_chart
            ),
        },
    }
    write_json(args.output_json.resolve(), payload)
    print(json.dumps(payload, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
