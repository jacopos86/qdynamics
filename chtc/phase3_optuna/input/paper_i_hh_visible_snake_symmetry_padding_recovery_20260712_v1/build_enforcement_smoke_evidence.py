#!/usr/bin/env python3
"""Build the fail-closed corrected symmetry/padding depth-1 smoke evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


BUNDLE = Path(
    "chtc/phase3_optuna/input/"
    "paper_i_hh_visible_snake_symmetry_padding_recovery_20260712_v1"
)
EVIDENCE = BUNDLE / "preflight_evidence"
STEM = EVIDENCE / "weak_weak_depth1_current_enforcement_on"
OUTPUT = EVIDENCE / "weak_weak_depth1_corrected_enforcement_smoke_20260712.json"
LEAKAGE_ATOL = 1.0e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": path.as_posix(), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _components(payload: dict[str, Any]) -> dict[str, int]:
    result = {
        key: int(payload[key])
        for key in ("N_H_outer", "N_H_refit", "N_grad", "N_metric")
    }
    result["S_alg"] = int(payload.get("S_alg", sum(result.values())))
    return result


def main() -> int:
    paths = {
        "result_json": Path(f"{STEM}.result.json"),
        "current_json": Path(f"{STEM}.current.json"),
        "estimator_ledger_json": Path(f"{STEM}.estimator_ledger.json"),
        "stdout_log": Path(f"{STEM}.stdout.log"),
        "stderr_log": Path(f"{STEM}.stderr.log"),
    }
    artifacts = {name: _artifact(path) for name, path in paths.items()}
    result = json.loads(paths["result_json"].read_text(encoding="utf-8"))
    current = json.loads(paths["current_json"].read_text(encoding="utf-8"))
    ledger = json.loads(paths["estimator_ledger_json"].read_text(encoding="utf-8"))
    adapt = result["adapt_vqe"]
    continuation = adapt["continuation"]
    compatibility = continuation["historical_paper_i_route_compatibility"]
    controls = compatibility["resolved_behavior"]["shortlist_budget_contract"]
    runtime_split = continuation["runtime_split_summary"]
    post_checkpoint = adapt["active_prefix_checkpoints"][0]
    terminal_checkpoint = adapt["terminal_active_prefix_checkpoint"]
    if len(post_checkpoint["ordered_active_operators"]) != 1:
        raise ValueError("expected one active operator at depth 1")
    selected_operator = post_checkpoint["ordered_active_operators"][0]
    lineage = selected_operator["route_a_child_padding_lineage"]
    symmetry_gate = lineage["raw_symmetry_gate"]
    fixed_sector = symmetry_gate["fixed_count_sector"]
    accounting = ledger["accounting"]
    occurrence = accounting["executed_occurrence_accounting"]["all_execution"]
    sidecar_pointer = continuation["estimator_call_accounting"]["sidecar"]

    assertions = {
        "compatibility_active": bool(
            compatibility["active"]
            and compatibility["compatibility_id"]
            == "paper_i_july8_physical_singleton_route_v1"
        ),
        "historical_caps_restored": bool(
            controls["phase1_shortlist_size_effective"] == 8
            and controls["phase2_shortlist_size_effective"] == 4
            and continuation["physical_operator_lane_policy"]["shortlist_controls"][
                "phase3_shortlist_size_effective"
            ]
            == 4
            and abs(float(controls["phase2_shortlist_fraction_effective"]) - 1.0 / 12.0)
            <= 1.0e-15
        ),
        "hard_guard_policy_active": bool(
            result["settings"]["phase3_runtime_split_child_set_symmetry_policy"]
            == "hard_guard"
            and runtime_split["child_set_symmetry_policy"] == "hard_guard"
        ),
        "selected_child_symmetry_checked_and_passed": bool(
            symmetry_gate["checked"]
            and symmetry_gate["passed"]
            and symmetry_gate["hard_guard_present"]
            and symmetry_gate["hard_guard_required"]
            and not symmetry_gate["rejected"]
        ),
        "selected_child_fixed_sector_exact": bool(
            fixed_sector["fixed_num_particles"] == {"n_dn": 1, "n_up": 1}
            and fixed_sector["particle_sector_invariant"]
            and fixed_sector["spin_sector_invariant"]
            and float(fixed_sector["sector_leakage_max_abs"]) == 0.0
            and float(fixed_sector["particle_sector_leakage_l1"]) == 0.0
            and float(fixed_sector["spin_sector_leakage_l1"]) == 0.0
        ),
        "padding_policy_active": bool(
            runtime_split["child_padding"]["policy"] == "exact_projected_grouped_v1"
            and runtime_split["child_padding_projection_active"]
        ),
        "padding_counts_exact": bool(
            runtime_split["projected_child_count_padding"] == 36
            and runtime_split["rejected_child_count_padding_zero"] == 0
            and runtime_split["deduplicated_child_count_padding"] == 8
            and runtime_split["admissible_child_set_count"] == 36
            and runtime_split["child_set_parallel_scoring_count"] == 36
        ),
        "selected_execution_grouped_exact": bool(
            selected_operator["execution_mode"] == "grouped_exact"
            and lineage["projection"]["recommended_execution_mode"] == "grouped_exact"
            and lineage["projection"]["applied_before_child_phase1_evaluation"]
        ),
        "post_checkpoint_complete": bool(
            post_checkpoint["schema"] == "paper_i_signed_active_prefix_checkpoint_v1"
            and post_checkpoint["outer_iteration"] == 1
            and post_checkpoint["active_ansatz_depth"] == 1
            and post_checkpoint["checkpoint_sha256"]
            == "d7ac61ca8a51c128d5ea7f03961303ccfebfc9f23181511b69b338c4792187d1"
            and post_checkpoint["signed_unwrapped_logical_parameters"]
            == [0.8475756625181428]
            and post_checkpoint["signed_unwrapped_runtime_parameters"]
            == [0.8475756625181428]
        ),
        "terminal_checkpoint_complete": bool(
            terminal_checkpoint["schema"] == "paper_i_signed_active_prefix_checkpoint_v1"
            and terminal_checkpoint["outer_iteration"] == 1
            and terminal_checkpoint["active_ansatz_depth"] == 1
            and terminal_checkpoint["checkpoint_sha256"]
            == "2e357b591a6bbbe50b34b51544a0f69875e6839cfe797b8371853f169f366036"
        ),
        "checkpoint_leakage_within_tolerance": bool(
            float(post_checkpoint["fixed_spin_sector_illegal_probability"]) <= LEAKAGE_ATOL
            and float(post_checkpoint["boson_illegal_codeword_probability"]) <= LEAKAGE_ATOL
            and float(terminal_checkpoint["fixed_spin_sector_illegal_probability"])
            <= LEAKAGE_ATOL
            and float(terminal_checkpoint["boson_illegal_codeword_probability"])
            <= LEAKAGE_ATOL
        ),
        "estimator_ledger_complete": bool(
            ledger["schema"] == "paper_i_estimator_call_ledger_sidecar_v1"
            and accounting["complete"]
            and accounting["exact_blockers"] == []
            and accounting["status"] == "resolved_from_live_state_keyed_instrumentation"
        ),
        "estimator_ledger_sidecar_hash_roundtrip": bool(
            sidecar_pointer["path"] == paths["estimator_ledger_json"].as_posix()
            and sidecar_pointer["sha256"] == artifacts["estimator_ledger_json"]["sha256"]
            and sidecar_pointer["size_bytes"] == artifacts["estimator_ledger_json"]["size_bytes"]
            and {
                key: value
                for key, value in adapt["estimator_call_accounting"].items()
                if key != "sidecar"
            }
            == accounting
        ),
        "estimator_accounting_components_exact": bool(
            _components(accounting["winning_lineage"])
            == {"N_H_outer": 1, "N_H_refit": 46, "N_grad": 133, "N_metric": 177, "S_alg": 357}
            and _components(accounting["all_branch_search_work"])
            == {"N_H_outer": 1, "N_H_refit": 72, "N_grad": 133, "N_metric": 177, "S_alg": 383}
            and accounting["discarded_branch_only_by_unique_set_difference"]["S_alg"] == 26
            and _components(occurrence)
            == {"N_H_outer": 3, "N_H_refit": 84, "N_grad": 141, "N_metric": 229, "S_alg": 457}
        ),
        "runtime_child_gradient_occurrences_complete": bool(
            occurrence["occurrence_count_by_consumer_scope"][
                "runtime_split_child_gradient"
            ]
            == 36
            and occurrence["occurrence_count_by_consumer_scope"][
                "runtime_split_child_gradient"
            ]
            == runtime_split["child_set_parallel_scoring_count"]
        ),
        "current_checkpoint_artifact_present": bool(
            isinstance(current, dict) and artifacts["current_json"]["size_bytes"] > 0
        ),
        "stderr_empty": artifacts["stderr_log"]["sha256"]
        == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    }
    passed = all(assertions.values())
    output = {
        "schema": "paper_i_hh_weak_weak_depth1_corrected_enforcement_smoke_v1",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "leakage_tolerance": LEAKAGE_ATOL,
        "input_artifacts": artifacts,
        "assertions": assertions,
        "failed_assertions": sorted(key for key, value in assertions.items() if not value),
        "route_compatibility": compatibility,
        "symmetry": {
            "policy": runtime_split["child_set_symmetry_policy"],
            "selected_gate": symmetry_gate,
            "rejected_child_count": runtime_split["rejected_child_count_symmetry"],
        },
        "padding": {
            "policy": runtime_split["child_padding"],
            "projected": runtime_split["projected_child_count_padding"],
            "zero_rejected": runtime_split["rejected_child_count_padding_zero"],
            "deduplicated": runtime_split["deduplicated_child_count_padding"],
            "admissible": runtime_split["admissible_child_set_count"],
            "scored": runtime_split["child_set_parallel_scoring_count"],
            "selected_execution_mode": selected_operator["execution_mode"],
        },
        "prefix_checkpoints": {
            "post_admission": post_checkpoint,
            "terminal": terminal_checkpoint,
        },
        "estimator_accounting": {
            "winning_lineage": _components(accounting["winning_lineage"]),
            "all_branch_search_work": _components(accounting["all_branch_search_work"]),
            "discarded_branch_unique": accounting[
                "discarded_branch_only_by_unique_set_difference"
            ],
            "executed_occurrences": _components(occurrence),
            "runtime_split_child_gradient_occurrences": occurrence[
                "occurrence_count_by_consumer_scope"
            ]["runtime_split_child_gradient"],
            "complete": accounting["complete"],
            "exact_blockers": accounting["exact_blockers"],
            "sidecar_pointer": sidecar_pointer,
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": OUTPUT.as_posix(), "sha256": _sha256(OUTPUT), "passed": passed}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
