from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROFILE = (
    "supported_whitened_adaptive_trust_full_response_no_novelty_"
    "metric_prune_beam_v1"
)


def get(payload: dict[str, Any], dotted: str) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def validate_result_payload(
    *, command: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    contract_settings = "settings.sr_route_profile_contract.execution_settings"
    expected = {
        "settings.problem": "molecular_vibronic_h2o_linear_fd",
        "settings.route_family": "singleton_response_snake",
        "settings.route_profile": PROFILE,
        "settings.route_profile_conformance": "registered_profile",
        "settings.sr_route_profile_request": PROFILE,
        "settings.sr_route_profile_resolved": PROFILE,
        "settings.sr_route_profile_contract_sha256": command[
            "route_profile_contract_sha256"
        ],
        "settings.phase3_novelty_ablation_mode": "all",
        "settings.phase2_enable_batching": False,
        "settings.phase3_response_coordinate_scope": (
            "full_active_plus_singleton_v1"
        ),
        "settings.phase1_prune_enabled": True,
        "settings.phase1_prune_schur_nomination_route": "metric_regularized_v1",
        "settings.phase1_prune_metric_schur_mu": 0.01,
        "settings.phase3_backend_cost_mode": "proxy",
        f"{contract_settings}.problem": "molecular_vibronic_h2o_linear_fd",
        f"{contract_settings}.adapt_max_depth": 50,
        f"{contract_settings}.phase0_pilot_enabled": False,
        f"{contract_settings}.phase3_novelty_ablation_mode": "all",
        f"{contract_settings}.phase2_enable_batching": False,
        f"{contract_settings}.phase3_enable_batching": False,
        f"{contract_settings}.phase3_response_coordinate_scope": (
            "full_active_plus_singleton_v1"
        ),
        f"{contract_settings}.phase3_runtime_split_subset_sizes": "1",
        f"{contract_settings}.phase3_runtime_split_child_padding_policy": (
            "full_binary_code_space_v1"
        ),
        f"{contract_settings}.adapt_beam_live_branches": 3,
        f"{contract_settings}.adapt_beam_children_per_parent": 2,
        f"{contract_settings}.phase1_prune_enabled": True,
        f"{contract_settings}.phase1_prune_schur_nomination_route": (
            "metric_regularized_v1"
        ),
        f"{contract_settings}.phase1_prune_metric_schur_mu": 0.01,
        f"{contract_settings}.phase3_backend_cost_mode": "proxy",
        f"{contract_settings}.adapt_accepted_refit_scope": "full_ansatz_v1",
        f"{contract_settings}.adapt_accepted_refit_coordinate_chart": (
            "supported_fs_whitened_fixed_v1"
        ),
    }
    mismatches = {
        dotted: {"expected": value, "actual": get(result, dotted)}
        for dotted, value in expected.items()
        if get(result, dotted) != value
    }
    depth = get(result, "adapt_vqe.ansatz_depth")
    energy = get(result, "adapt_vqe.energy")
    exact = get(result, "adapt_vqe.exact_gs_energy")
    checks = {
        "settings": not mismatches,
        "depth_within_contract": isinstance(depth, int) and 0 <= depth <= 50,
        "energy_present": isinstance(energy, (int, float)),
        "same_cutoff_exact_present": isinstance(exact, (int, float)),
    }
    return {
        "schema": "paper_iv_h2o_sr_depth50_result_validation_v1",
        "status": "pass" if all(checks.values()) else "blocked",
        "checks": checks,
        "mismatches": mismatches,
        "ansatz_depth": depth,
        "energy": energy,
        "exact_gs_energy": exact,
        "abs_delta_e": (
            abs(float(energy) - float(exact))
            if isinstance(energy, (int, float)) and isinstance(exact, (int, float))
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command-json", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    command = json.loads(args.command_json.read_text(encoding="utf-8"))
    result = json.loads(args.result_json.read_text(encoding="utf-8"))
    payload = validate_result_payload(command=command, result=result)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
