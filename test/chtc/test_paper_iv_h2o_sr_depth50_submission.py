from __future__ import annotations

import json
from pathlib import Path

from chtc.paper_iv_h2o_sr_no_novelty_metric_prune_beam_d50_20260717.validate_result import (
    PROFILE,
    validate_result_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = (
    REPO_ROOT
    / "chtc/paper_iv_h2o_sr_no_novelty_metric_prune_beam_d50_20260717"
)
INPUT = RUN_DIR / "input"


def test_command_uses_registered_sr_profile_from_depth_zero_to_50() -> None:
    command = json.loads((INPUT / "command.json").read_text(encoding="utf-8"))
    argv = command["argv"]

    assert command["starting_depth"] == 0
    assert command["target_depth"] == 50
    assert "--adapt-resume-scaffold-json" not in argv
    profile_index = argv.index("--sr-route-profile")
    assert argv[profile_index + 1] == "sr_snake_no_novelty_metric_prune_beam_v1"
    assert command["route_profile_contract"]["route_family"] == (
        "singleton_response_snake"
    )
    assert command["route_profile_contract"]["execution_settings"][
        "adapt_max_depth"
    ] == 50


def test_route_audit_allows_only_requested_h2o_differences() -> None:
    audit = json.loads(
        (INPUT / "route_settings_diff_audit.json").read_text(encoding="utf-8")
    )
    assert audit["status"] == "pass"
    assert audit["unexpected_difference_fields"] == []
    assert audit["missing_expected_difference_fields"] == []
    assert set(audit["differences"]) == {
        "problem",
        "adapt_max_depth",
        "phase3_novelty_ablation_mode",
        "phase3_runtime_split_child_padding_policy",
        "phase3_backend_cost_mode",
        "phase1_prune_schur_nomination_route",
        "phase1_prune_metric_schur_mu",
    }
    shared = audit["shared_controller_checks"]
    assert shared["phase0_pilot_enabled"] is False
    assert shared["phase2_enable_batching"] is False
    assert shared["phase3_enable_batching"] is False
    assert shared["phase3_runtime_split_subset_sizes"] == "1"
    assert shared["phase3_runtime_split_child_padding_policy"] == (
        "full_binary_code_space_v1"
    )
    assert shared["phase3_backend_cost_mode"] == "proxy"
    assert shared["adapt_beam_live_branches"] == 3
    assert shared["adapt_beam_children_per_parent"] == 2
    assert shared["adapt_accepted_refit_scope"] == "full_ansatz_v1"

    result_comparison = audit["paper_i_result_evidence_comparison"]
    assert result_comparison["status"] == "pass"
    assert result_comparison["paper_i_anchor_mismatches"] == {}
    assert result_comparison["shared_method_contract"]["admission"] == "singleton"
    assert result_comparison["shared_method_contract"]["accepted_refit_scope"] == (
        "full_ansatz_v1"
    )
    assert set(result_comparison["intentional_differences"]) == {
        "physics_problem",
        "depth_horizon",
        "novelty",
        "phase3_response_window",
        "beam",
        "pruning",
        "child_padding",
        "backend_cost",
    }
    assert result_comparison["intentional_differences"]["beam"]["h2o_target"] == {
        "live_branches": 3,
        "children_per_parent": 2,
    }
    assert result_comparison["intentional_differences"]["pruning"][
        "h2o_target"
    ]["nomination_route"] == "metric_regularized_v1"
    response_window = result_comparison["intentional_differences"][
        "phase3_response_window"
    ]
    assert response_window["paper_i_result"]["phase3_score_policy"] == (
        "reduced_window_geometry_v1"
    )
    assert response_window["h2o_target"]["scope"] == (
        "full_active_plus_singleton_v1"
    )


def test_submit_requests_production_resources_and_transfers_recoverable_output() -> None:
    submit = (RUN_DIR / "submit.sub").read_text(encoding="utf-8")

    assert "request_cpus = 8" in submit
    assert "request_memory = 64GB" in submit
    assert "request_disk = 80GB" in submit
    assert "+MaxRuntime = 1209600" in submit
    assert "kill_sig_timeout = 900" in submit
    assert "when_to_transfer_output = ON_EXIT_OR_EVICT" in submit
    assert "transfer_output_files = deliverables/" in submit
    assert (
        "paper_iv_h2o_sr_v3_no_novelty_metric_prune_beam_from_zero_"
        "d50_20260717_v1"
    ) in submit


def test_local_command_preflight_passes_with_locked_fixture() -> None:
    preflight = json.loads(
        (INPUT / "command_preflight.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (INPUT / "input_manifest.json").read_text(encoding="utf-8")
    )

    assert preflight["status"] == "pass"
    assert all(preflight["checks"].values())
    assert manifest["route_difference_audit_status"] == "pass"
    assert manifest["starting_depth"] == 0
    assert manifest["target_depth"] == 50


def test_result_validator_reads_profile_owned_fields_from_route_contract() -> None:
    command = json.loads((INPUT / "command.json").read_text(encoding="utf-8"))
    execution_settings = command["route_profile_contract"]["execution_settings"]
    result = {
        "settings": {
            "problem": "molecular_vibronic_h2o_linear_fd",
            "route_family": "singleton_response_snake",
            "route_profile": PROFILE,
            "route_profile_conformance": "registered_profile",
            "sr_route_profile_request": PROFILE,
            "sr_route_profile_resolved": PROFILE,
            "sr_route_profile_contract_sha256": command[
                "route_profile_contract_sha256"
            ],
            "sr_route_profile_contract": {
                "execution_settings": execution_settings,
            },
            "phase3_novelty_ablation_mode": "all",
            "phase2_enable_batching": False,
            "phase3_response_coordinate_scope": "full_active_plus_singleton_v1",
            "phase1_prune_enabled": True,
            "phase1_prune_schur_nomination_route": "metric_regularized_v1",
            "phase1_prune_metric_schur_mu": 0.01,
            "phase3_backend_cost_mode": "proxy",
        },
        "adapt_vqe": {
            "ansatz_depth": 50,
            "energy": -75.0,
            "exact_gs_energy": -75.01,
        },
    }

    validation = validate_result_payload(command=command, result=result)

    assert validation["status"] == "pass"
    assert validation["checks"]["settings"] is True
    assert validation["mismatches"] == {}
