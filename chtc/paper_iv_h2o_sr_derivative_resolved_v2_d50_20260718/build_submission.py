from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FIXTURE_SHA256 = "570690bd126787305b340bd2f7493499c0f3101e3e2820c2d355c55c16afa594"
PAPER_I_RESULT_EVIDENCE_RELATIVE_PATH = Path(
    "MATH/paper_facing/paper_I_static_scaffold/"
    "paper_i_no_ordinary_novelty_sr_snake_evidence_copy_20260717.json"
)
PAPER_I_RESULT_EVIDENCE_SHA256 = (
    "eb8286c4d8df9035f425caa5906d21395f66c2df37083b5d05a5561b5f8d7c98"
)
PAPER_I_RESULT_SOURCE_RELATIVE_PATH = Path(
    "raw_outputs/paper_i_hh_sr_snake_weak_weak_undamped_no_prune_no_beam_"
    "no_ordinary_novelty_fallback_on_20260715/json/result.json"
)
PAPER_I_RESULT_SOURCE_SHA256 = (
    "68fde0ab9de5ae69cee27ac0f54cb52f9e377882969daa0a1630d14f520ffdaa"
)
PROFILE_ALIAS = "sr_snake_h2o_derivative_resolved_v2"
RECORD_ID = (
    "paper_iv_h2o_sr_derivative_resolved_v2_from_zero_d50_20260718_v1"
)
ALLOWED_PROFILE_DIFFS = {"adapt_pool"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def archive_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(info.name).parts
    if "__pycache__" in parts or info.name.endswith((".pyc", ".pyo")):
        return None
    return info


def build_source_archive(repo: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for relative in ("src", "pipelines", "docs/reports"):
            archive.add(repo / relative, arcname=relative, filter=archive_filter)


def command_argv() -> list[str]:
    output_root = f"../raw_outputs/{RECORD_ID}"
    return [
        "python3",
        "-u",
        "-m",
        "pipelines.static_adapt.adapt_pipeline",
        "--sr-route-profile",
        PROFILE_ALIAS,
        "--problem",
        "molecular_vibronic_h2o_linear_fd",
        "--molecular-vibronic-h2o-linear-fd-fixture-json",
        "../runtime_inputs/h2o_fixture.json",
        "--L",
        "6",
        "--n-ph-max",
        "1",
        "--term-order",
        "sorted",
        "--ordering",
        "blocked",
        "--boson-encoding",
        "binary",
        "--boundary",
        "open",
        "--include-zero-point",
        "--skip-trajectory",
        "--skip-pdf",
        "--adapt-parallel-gradient-workers",
        "8",
        "--adapt-current-json",
        f"{output_root}/current.json",
        "--adapt-current-json-every-depth",
        "1",
        "--adapt-current-json-keep-history-tail",
        "1",
        "--adapt-benchmark-target-abs-delta-e",
        "0.0016",
        "--adapt-estimator-call-ledger-json",
        f"{output_root}/estimator_call_ledger.json",
        "--output-json",
        f"{output_root}/result.json",
    ]


def paper_i_result_evidence_comparison(
    *, repo: Path, target: dict[str, Any]
) -> dict[str, Any]:
    evidence_path = repo / PAPER_I_RESULT_EVIDENCE_RELATIVE_PATH
    if (
        not evidence_path.is_file()
        or sha256(evidence_path) != PAPER_I_RESULT_EVIDENCE_SHA256
    ):
        raise SystemExit("The locked Paper-I SR-SNAKE result evidence has drifted.")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    route = evidence.get("route_contract", {})
    paper_i_result_source = repo / PAPER_I_RESULT_SOURCE_RELATIVE_PATH
    if (
        not paper_i_result_source.is_file()
        or sha256(paper_i_result_source) != PAPER_I_RESULT_SOURCE_SHA256
    ):
        raise SystemExit("The locked Paper-I SR-SNAKE result source has drifted.")
    evidence_source = evidence.get("rows", [{}])[0].get("source", {})
    if (
        evidence_source.get("path") != str(PAPER_I_RESULT_SOURCE_RELATIVE_PATH)
        or evidence_source.get("sha256") != PAPER_I_RESULT_SOURCE_SHA256
    ):
        raise SystemExit("Paper-I evidence does not bind the expected result source.")
    expected_anchor = {
        "route_family": "singleton_response_snake",
        "phase0": "off",
        "admission": "singleton",
        "phase2_batching": "off",
        "phase3_batching": "off",
        "beam": "off_effective_1x1",
        "pruning": "off",
        "phase2_gram_novelty_policy": "fallback_only_v1",
        "phase3_gram_novelty_policy": "fallback_only_v1",
        "all_energy_models_infeasible_novelty_fallback": "retained",
        "phase3_coordinate_policy": "supported_metric_whitened_eigh_v1",
        "trust_policy": "displacement_calibrated_unbounded_v2",
        "accepted_refit": "full_accepted_ansatz_supported_fs_whitened",
        "powell_base_chart": "expanded_runtime_projected_logical_v1",
        "negative_curvature_escape": "off",
        "powell_maxiter": 200,
    }
    anchor_mismatches = {
        key: {"expected": value, "actual": route.get(key)}
        for key, value in expected_anchor.items()
        if route.get(key) != value
    }
    shared_method_contract = {
        "route_family": "singleton_response_snake",
        "phase0": "off",
        "admission": "singleton",
        "phase2_batching": "off",
        "phase3_batching": "off",
        "phase3_coordinate_policy": target[
            "historical_singleton_coordinate_solve_policy"
        ],
        "trust_policy": target[
            "historical_singleton_trust_region_update_policy"
        ],
        "accepted_refit_scope": target["adapt_accepted_refit_scope"],
        "accepted_refit_coordinate_chart": target[
            "adapt_accepted_refit_coordinate_chart"
        ],
        "powell_base_chart": target["adapt_accepted_refit_base_chart_policy"],
        "powell_maxiter": target["adapt_maxiter"],
        "negative_curvature_escape": target["sr_escape_mode"],
    }
    intentional_differences = {
        "physics_problem": {
            "paper_i_result": "hh",
            "h2o_target": target["problem"],
            "reason": "application_hamiltonian_change",
        },
        "depth_horizon": {
            "paper_i_result": evidence["display_semantics"]["snake_horizons"],
            "h2o_target": target["adapt_max_depth"],
            "reason": "user_requested_depth_50",
        },
        "operator_pool": {
            "paper_i_result": "full_meta",
            "h2o_target": target["adapt_pool"],
            "reason": "derivative_resolved_molecular_vibronic_pool_v2",
        },
        "novelty": {
            "paper_i_result": (
                "ordinary_novelty_off_with_infeasible_model_fallback_retained"
            ),
            "h2o_target": "all_phase2_phase3_novelty_paths_bypassed",
            "reason": "user_requested_no_novelty",
        },
        "phase3_response_window": {
            "paper_i_result": {
                "scope": "two_active_coordinates_plus_singleton_candidate",
                "phase3_score_policy": "reduced_window_geometry_v1",
                "w3_wopt_decoupled": False,
            },
            "h2o_target": {
                "scope": target["phase3_response_coordinate_scope"],
                "response_coordinates": "all_m_active_plus_singleton_candidate",
                "gram_null_projection_only": True,
            },
            "reason": "remove_candidate_ranking_dependence_on_reduced_window",
        },
        "beam": {
            "paper_i_result": route["beam"],
            "h2o_target": {
                "live_branches": target["adapt_beam_live_branches"],
                "children_per_parent": target["adapt_beam_children_per_parent"],
            },
            "reason": "user_requested_beam",
        },
        "pruning": {
            "paper_i_result": route["pruning"],
            "h2o_target": {
                "enabled": target["phase1_prune_enabled"],
                "nomination_route": target[
                    "phase1_prune_schur_nomination_route"
                ],
                "metric_mu": target["phase1_prune_metric_schur_mu"],
            },
            "reason": "user_requested_metric_pruning",
        },
        "child_padding": {
            "paper_i_result": "exact_projected_grouped_v1",
            "h2o_target": target[
                "phase3_runtime_split_child_padding_policy"
            ],
            "reason": "one_qubit_nph1_boson_register_has_no_illegal_codewords",
        },
        "backend_cost": {
            "paper_i_result": "marrakesh_graph_span_v1",
            "h2o_target": target["phase3_backend_cost_mode"],
            "reason": "paper_i_marrakesh_cost_path_is_hh_only",
        },
    }
    return {
        "schema": "paper_iv_h2o_vs_paper_i_result_route_audit_v1",
        "status": "pass" if not anchor_mismatches else "blocked",
        "source": {
            "path": str(PAPER_I_RESULT_EVIDENCE_RELATIVE_PATH),
            "sha256": PAPER_I_RESULT_EVIDENCE_SHA256,
            "evidence_status": evidence.get("status"),
        },
        "paper_i_result_response_scope_evidence": {
            "source_path": str(PAPER_I_RESULT_SOURCE_RELATIVE_PATH),
            "source_sha256": PAPER_I_RESULT_SOURCE_SHA256,
            "history_rows_checked": 30,
            "observed_phase3_score_policy": "reduced_window_geometry_v1",
            "observed_w3_wopt_decoupled": False,
            "observed_post_depth3_active_model": (
                "two_active_coordinates_plus_singleton_candidate"
            ),
            "accepted_refit_distinction": (
                "full_ansatz_refit_after_selection_does_not_repair_reduced_"
                "candidate_ranking"
            ),
        },
        "paper_i_anchor_mismatches": anchor_mismatches,
        "shared_method_contract": shared_method_contract,
        "intentional_differences": intentional_differences,
        "result_comparison_scope": (
            "route_semantics_only; energies and convergence rates are not "
            "cross-Hamiltonian comparators"
        ),
    }


def profile_diff_audit() -> tuple[dict[str, Any], dict[str, Any]]:
    from pipelines.static_adapt.sr_snake_route_profile import (
        CANONICAL_SR_SNAKE_H2O_DERIVATIVE_RESOLVED_V2_EXECUTION_SETTINGS,
        CANONICAL_SR_SNAKE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1_EXECUTION_SETTINGS,
        SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2,
        SR_ROUTE_PROFILE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1,
        canonical_sr_snake_h2o_derivative_resolved_v2_contract,
        canonical_sr_snake_h2o_derivative_resolved_v2_contract_sha256,
        canonical_sr_snake_no_novelty_metric_prune_beam_v1_contract,
        canonical_sr_snake_no_novelty_metric_prune_beam_v1_contract_sha256,
    )

    baseline = dict(
        CANONICAL_SR_SNAKE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1_EXECUTION_SETTINGS
    )
    target = dict(
        CANONICAL_SR_SNAKE_H2O_DERIVATIVE_RESOLVED_V2_EXECUTION_SETTINGS
    )
    differences = {
        field: {"parent_h2o_route": baseline.get(field), "h2o_target": target.get(field)}
        for field in sorted(set(baseline) | set(target))
        if baseline.get(field) != target.get(field)
    }
    unexpected = sorted(set(differences) - ALLOWED_PROFILE_DIFFS)
    missing = sorted(ALLOWED_PROFILE_DIFFS - set(differences))
    contract = canonical_sr_snake_h2o_derivative_resolved_v2_contract()
    parent_contract = canonical_sr_snake_no_novelty_metric_prune_beam_v1_contract()
    if contract.get("lineage_authority", {}).get("parent_contract_sha256") != (
        canonical_sr_snake_no_novelty_metric_prune_beam_v1_contract_sha256()
    ):
        raise SystemExit("Derivative-resolved route does not bind the expected parent.")
    audit = {
        "schema": "paper_iv_h2o_derivative_resolved_route_diff_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not unexpected and not missing else "blocked",
        "baseline": {
            "route_family": "singleton_response_snake",
            "route_profile": SR_ROUTE_PROFILE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1,
            "route_contract_sha256": (
                canonical_sr_snake_no_novelty_metric_prune_beam_v1_contract_sha256()
            ),
            "route_contract": parent_contract,
        },
        "target": {
            "route_family": "singleton_response_snake",
            "route_profile": SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2,
            "route_contract_sha256": (
                canonical_sr_snake_h2o_derivative_resolved_v2_contract_sha256()
            ),
            "route_contract": contract,
        },
        "allowed_difference_fields": sorted(ALLOWED_PROFILE_DIFFS),
        "differences": differences,
        "unexpected_difference_fields": unexpected,
        "missing_expected_difference_fields": missing,
        "shared_settings_count": sum(
            baseline.get(field) == target.get(field)
            for field in set(baseline) | set(target)
        ),
        "shared_controller_checks": {
            "phase0_pilot_enabled": target["phase0_pilot_enabled"],
            "phase2_enable_batching": target["phase2_enable_batching"],
            "phase3_enable_batching": target["phase3_enable_batching"],
            "phase3_runtime_split_subset_sizes": target[
                "phase3_runtime_split_subset_sizes"
            ],
            "phase3_runtime_split_child_padding_policy": target[
                "phase3_runtime_split_child_padding_policy"
            ],
            "adapt_beam_live_branches": target["adapt_beam_live_branches"],
            "adapt_beam_children_per_parent": target[
                "adapt_beam_children_per_parent"
            ],
            "adapt_inner_optimizer": target["adapt_inner_optimizer"],
            "adapt_maxiter": target["adapt_maxiter"],
            "adapt_accepted_refit_scope": target["adapt_accepted_refit_scope"],
            "adapt_accepted_refit_coordinate_chart": target[
                "adapt_accepted_refit_coordinate_chart"
            ],
            "phase3_response_coordinate_scope": target[
                "phase3_response_coordinate_scope"
            ],
            "phase3_backend_cost_mode": target["phase3_backend_cost_mode"],
        },
        "comparison_authority": (
            "registered_parent_route_contract_sha256_and_exact_setting_diff"
        ),
    }
    if audit["status"] != "pass":
        raise SystemExit(
            "H2O derivative-resolved route difference audit blocked: "
            f"unexpected={unexpected}, missing={missing}"
        )
    return audit, contract


def parser_preflight(repo: Path, argv: list[str]) -> dict[str, Any]:
    from pipelines.static_adapt.cli_config import _build_adapt_arg_parser
    from pipelines.static_adapt.sr_snake_route_profile import (
        CANONICAL_SR_SNAKE_H2O_DERIVATIVE_RESOLVED_V2_EXECUTION_SETTINGS,
        SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2,
        canonical_sr_snake_h2o_derivative_resolved_v2_contract_sha256,
    )

    parser = _build_adapt_arg_parser(adapt_gradient_parity_rtol=1.0e-8)
    args = parser.parse_args(argv[4:])
    expected = CANONICAL_SR_SNAKE_H2O_DERIVATIVE_RESOLVED_V2_EXECUTION_SETTINGS
    mismatches = {
        field: {"expected": value, "actual": getattr(args, field, None)}
        for field, value in expected.items()
        if getattr(args, field, None) != value
    }
    checks = {
        "profile_request": args.sr_route_profile_request
        == SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2,
        "profile_resolved": args.sr_route_profile_resolved
        == SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2,
        "contract_digest": args.sr_route_profile_contract_sha256
        == canonical_sr_snake_h2o_derivative_resolved_v2_contract_sha256(),
        "profile_settings": not mismatches,
        "fixture_exists": (repo / "tmp/h2o_linear_fd_valence_psi4_optimized/"
                           "h2o_linear_fd_sparse_fixture_nph1_ref2_reencoded_v2.json").is_file(),
    }
    if not all(checks.values()):
        raise SystemExit(
            "Normalized command preflight blocked: "
            + json.dumps({"checks": checks, "mismatches": mismatches}, default=str)
        )
    return {
        "schema": "paper_iv_h2o_sr_derivative_resolved_command_preflight_v1",
        "status": "pass",
        "checks": checks,
        "profile_setting_mismatches": mismatches,
        "route_contract_sha256": args.sr_route_profile_contract_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    stage = Path(__file__).resolve().parent
    input_dir = stage / "input"
    fixture_source = (
        repo
        / "tmp/h2o_linear_fd_valence_psi4_optimized/"
        "h2o_linear_fd_sparse_fixture_nph1_ref2_reencoded_v2.json"
    )
    if not fixture_source.is_file() or sha256(fixture_source) != FIXTURE_SHA256:
        raise SystemExit("The corrected H2O fixture is missing or has drifted.")

    input_dir.mkdir(parents=True, exist_ok=True)
    source_archive = input_dir / "source_tree.tar.gz"
    fixture_target = input_dir / "h2o_fixture.json"
    build_source_archive(repo, source_archive)
    shutil.copy2(fixture_source, fixture_target)

    argv = command_argv()
    audit, contract = profile_diff_audit()
    preflight = parser_preflight(repo, argv)
    command = {
        "schema": "paper_iv_h2o_sr_derivative_resolved_depth50_command_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "record_id": RECORD_ID,
        "run_class": "candidate_application_run",
        "starting_depth": 0,
        "target_depth": 50,
        "argv": argv,
        "route_profile_contract": contract,
        "route_profile_contract_sha256": audit["target"][
            "route_contract_sha256"
        ],
        "paper_i_route_difference_audit": audit,
    }
    write_json(input_dir / "command.json", command)
    write_json(input_dir / "route_settings_diff_audit.json", audit)
    write_json(input_dir / "command_preflight.json", preflight)

    files = {}
    for name in (
        "source_tree.tar.gz",
        "h2o_fixture.json",
        "command.json",
        "route_settings_diff_audit.json",
        "command_preflight.json",
    ):
        path = input_dir / name
        files[name] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    runtime_files = {}
    for name in (
        "run_payload.py",
        "runtime_preflight.py",
        "validate_result.py",
        "run_apptainer.sh",
    ):
        path = stage / name
        runtime_files[name] = {
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
    manifest = {
        "schema": "paper_iv_h2o_sr_derivative_resolved_depth50_input_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "record_id": RECORD_ID,
        "files": files,
        "runtime_files": runtime_files,
        "route_contract_sha256": audit["target"]["route_contract_sha256"],
        "route_difference_audit_status": audit["status"],
        "starting_depth": 0,
        "target_depth": 50,
    }
    write_json(input_dir / "input_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
