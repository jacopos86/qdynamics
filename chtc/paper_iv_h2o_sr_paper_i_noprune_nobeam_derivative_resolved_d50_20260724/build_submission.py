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
PAPER_I_MAIN_BUNDLE_RELATIVE_PATH = Path(
    "chtc/phase3_optuna/input/"
    "paper_i_hh_sr_snake_main_full_response_symmetric_cost_noprune_nobeam_"
    "no_ordinary_novelty_all_six_r50_20260718_v1_chtc/bundle_manifest.json"
)
PAPER_I_MAIN_BUNDLE_SHA256 = (
    "00eddce32f2e184422faed847ff5be129bfabc2c56d4a0b730adf8b71b62e8b1"
)
PAPER_I_MAIN_JOB_RELATIVE_PATH = Path(
    "chtc/phase3_optuna/input/"
    "paper_i_hh_sr_snake_main_full_response_symmetric_cost_noprune_nobeam_"
    "no_ordinary_novelty_all_six_r50_20260718_v1_chtc/jobs/weak_weak.json"
)
PAPER_I_MAIN_JOB_SHA256 = (
    "9806436a3b84f143a5c7db2be7eb2a337488b85c1dc0ceade58b67fb7d801918"
)
PAPER_I_MAIN_PROFILE_SHA256 = (
    "69dcf6d5711a05b54cc143562bc6f437511d781164319b2705857b22deda1538"
)
PAPER_I_MAIN_SOURCE_ARCHIVE_SHA256 = (
    "fa9014b9608ccb8a301df0429268482abf6dd10c91eb336111a15d00be256d35"
)
PROFILE_ALIAS = "sr_snake_h2o_derivative_resolved_paper_i_v3"
RECORD_ID = (
    "paper_iv_h2o_sr_paper_i_noprune_nobeam_derivative_resolved_"
    "from_zero_d50_20260724_v2"
)
ALLOWED_PROFILE_DIFFS = {
    "problem",
    "adapt_max_depth",
    "adapt_pool",
    "phase3_runtime_split_child_padding_policy",
    "phase3_backend_cost_mode",
}
EXPECTED_PAPER_I_ENFORCEMENT_REPAIR_PATHS = {
    "execution_settings.phase_live_hysteresis_enabled",
    "semantic_invariants.phase_live_hysteresis_enabled",
    "semantic_invariants.phase_retirement_policy",
}


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


def nested_difference_paths(
    current: Any, archived: Any, *, prefix: str = ""
) -> dict[str, dict[str, Any]]:
    if isinstance(current, dict) and isinstance(archived, dict):
        differences: dict[str, dict[str, Any]] = {}
        for key in sorted(set(current) | set(archived)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in current:
                differences[path] = {
                    "current": None,
                    "archived_paper_i": archived[key],
                }
            elif key not in archived:
                differences[path] = {
                    "current": current[key],
                    "archived_paper_i": None,
                }
            else:
                differences.update(
                    nested_difference_paths(
                        current[key], archived[key], prefix=path
                    )
                )
        return differences
    if current != archived:
        return {
            prefix: {
                "current": current,
                "archived_paper_i": archived,
            }
        }
    return {}


def paper_i_main_route_comparison(
    *,
    repo: Path,
    target: dict[str, Any],
    current_parent_contract: dict[str, Any],
) -> dict[str, Any]:
    bundle_path = repo / PAPER_I_MAIN_BUNDLE_RELATIVE_PATH
    job_path = repo / PAPER_I_MAIN_JOB_RELATIVE_PATH
    if (
        not bundle_path.is_file()
        or sha256(bundle_path) != PAPER_I_MAIN_BUNDLE_SHA256
    ):
        raise SystemExit("The locked Paper-I main bundle manifest has drifted.")
    if not job_path.is_file() or sha256(job_path) != PAPER_I_MAIN_JOB_SHA256:
        raise SystemExit("The locked Paper-I main route job has drifted.")

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    job = json.loads(job_path.read_text(encoding="utf-8"))
    route_identity = job.get("route_identity", {})
    archived_contract = route_identity.get("profile_contract", {})
    archived_digest = route_identity.get("profile_contract_sha256")
    archived_source_digest = (
        bundle.get("source_archive", {}).get("archive_sha256")
    )
    archived_identity_checks = {
        "profile_request": route_identity.get("profile_request")
        == "sr_snake_no_prune_symmetric_cost_v1",
        "profile_resolved": route_identity.get("profile_resolved")
        == (
            "supported_whitened_adaptive_trust_full_response_symmetric_cost_"
            "no_prune_v1"
        ),
        "profile_digest": archived_digest == PAPER_I_MAIN_PROFILE_SHA256,
        "source_archive_digest": (
            archived_source_digest == PAPER_I_MAIN_SOURCE_ARCHIVE_SHA256
        ),
        "bundle_route_parity": (
            bundle.get("route_parity", {}).get("status") == "pass"
        ),
        "bundle_scientific_settings": (
            bundle.get("scientific_settings_audit", {}).get("status") == "pass"
        ),
    }
    enforcement_differences = nested_difference_paths(
        current_parent_contract, archived_contract
    )
    enforcement_paths = set(enforcement_differences)
    enforcement_checks = {
        "only_expected_paths": (
            enforcement_paths == EXPECTED_PAPER_I_ENFORCEMENT_REPAIR_PATHS
        ),
        "phase_live_hysteresis_disabled": (
            enforcement_differences.get(
                "execution_settings.phase_live_hysteresis_enabled", {}
            ).get("current")
            is False
        ),
        "semantic_hysteresis_disabled": (
            enforcement_differences.get(
                "semantic_invariants.phase_live_hysteresis_enabled", {}
            ).get("current")
            is False
        ),
        "phase_retirement_disabled": (
            enforcement_differences.get(
                "semantic_invariants.phase_retirement_policy", {}
            ).get("current")
            == "disabled_v1"
        ),
    }
    shared_method_checks = {
        "effective_beam_1x1": (
            target["adapt_beam_live_branches"] == 1
            and target["adapt_beam_children_per_parent"] == 1
        ),
        "pruning_disabled": target["phase1_prune_enabled"] is False,
        "terminal_prune_disabled": (
            current_parent_contract["semantic_invariants"][
                "terminal_prune_active"
            ]
            is False
        ),
        "phase2_batching_disabled": target["phase2_enable_batching"] is False,
        "phase3_batching_disabled": target["phase3_enable_batching"] is False,
        "ordinary_novelty_disabled": (
            target["phase2_gram_novelty_policy"] == "fallback_only_v1"
            and target["phase3_gram_novelty_policy"] == "fallback_only_v1"
            and target["phase3_novelty_ablation_mode"] == "off"
        ),
        "full_active_plus_singleton": (
            target["phase3_response_coordinate_scope"]
            == "full_active_plus_singleton_v1"
        ),
        "supported_whitened_coordinates": (
            target["historical_singleton_coordinate_solve_policy"]
            == "supported_metric_whitened_eigh_v1"
        ),
        "powell_full_ansatz_refit": (
            target["adapt_inner_optimizer"] == "POWELL"
            and target["adapt_accepted_refit_scope"] == "full_ansatz_v1"
        ),
    }
    status = (
        "pass"
        if all(archived_identity_checks.values())
        and all(enforcement_checks.values())
        and all(shared_method_checks.values())
        else "blocked"
    )
    return {
        "schema": "paper_iv_h2o_vs_archived_paper_i_main_route_audit_v2",
        "status": status,
        "archived_paper_i_source": {
            "bundle_manifest_path": str(PAPER_I_MAIN_BUNDLE_RELATIVE_PATH),
            "bundle_manifest_sha256": PAPER_I_MAIN_BUNDLE_SHA256,
            "route_job_path": str(PAPER_I_MAIN_JOB_RELATIVE_PATH),
            "route_job_sha256": PAPER_I_MAIN_JOB_SHA256,
            "profile_contract_sha256": PAPER_I_MAIN_PROFILE_SHA256,
            "source_archive_sha256": PAPER_I_MAIN_SOURCE_ARCHIVE_SHA256,
        },
        "archived_identity_checks": archived_identity_checks,
        "post_bundle_enforcement_repair": {
            "classification": "phase3_service_enforcement_repair_v1",
            "difference_paths": enforcement_differences,
            "checks": enforcement_checks,
        },
        "shared_method_checks": shared_method_checks,
        "application_setting_differences": {
            "problem": target["problem"],
            "adapt_max_depth": target["adapt_max_depth"],
            "adapt_pool": target["adapt_pool"],
            "phase3_runtime_split_child_padding_policy": target[
                "phase3_runtime_split_child_padding_policy"
            ],
            "phase3_backend_cost_mode": target["phase3_backend_cost_mode"],
        },
        "result_comparison_scope": (
            "route semantics only; H2O and Hubbard-Holstein energies are not "
            "cross-Hamiltonian comparators"
        ),
    }


def profile_diff_audit(
    *, repo: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    from pipelines.static_adapt.sr_snake_route_profile import (
        CANONICAL_SR_SNAKE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3_EXECUTION_SETTINGS,
        CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_V1_EXECUTION_SETTINGS,
        SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3,
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
        canonical_sr_snake_h2o_derivative_resolved_paper_i_v3_contract,
        canonical_sr_snake_h2o_derivative_resolved_paper_i_v3_contract_sha256,
        canonical_sr_snake_no_prune_symmetric_cost_v1_contract,
        canonical_sr_snake_no_prune_symmetric_cost_v1_contract_sha256,
    )

    baseline = dict(
        CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_V1_EXECUTION_SETTINGS
    )
    target = dict(
        CANONICAL_SR_SNAKE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3_EXECUTION_SETTINGS
    )
    differences = {
        field: {"parent_h2o_route": baseline.get(field), "h2o_target": target.get(field)}
        for field in sorted(set(baseline) | set(target))
        if baseline.get(field) != target.get(field)
    }
    unexpected = sorted(set(differences) - ALLOWED_PROFILE_DIFFS)
    missing = sorted(ALLOWED_PROFILE_DIFFS - set(differences))
    contract = canonical_sr_snake_h2o_derivative_resolved_paper_i_v3_contract()
    parent_contract = canonical_sr_snake_no_prune_symmetric_cost_v1_contract()
    if contract.get("lineage_authority", {}).get("parent_contract_sha256") != (
        canonical_sr_snake_no_prune_symmetric_cost_v1_contract_sha256()
    ):
        raise SystemExit("Derivative-resolved route does not bind the expected parent.")
    paper_i_comparison = paper_i_main_route_comparison(
        repo=repo,
        target=target,
        current_parent_contract=parent_contract,
    )
    audit = {
        "schema": "paper_iv_h2o_paper_i_noprune_nobeam_route_diff_audit_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "pass"
            if not unexpected
            and not missing
            and paper_i_comparison["status"] == "pass"
            else "blocked"
        ),
        "baseline": {
            "route_family": "singleton_response_snake",
            "route_profile": SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
            "route_contract_sha256": (
                canonical_sr_snake_no_prune_symmetric_cost_v1_contract_sha256()
            ),
            "route_contract": parent_contract,
        },
        "target": {
            "route_family": "singleton_response_snake",
            "route_profile": (
                SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3
            ),
            "route_contract_sha256": (
                canonical_sr_snake_h2o_derivative_resolved_paper_i_v3_contract_sha256()
            ),
            "route_contract": contract,
        },
        "archived_paper_i_comparison": paper_i_comparison,
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
            "phase1_prune_enabled": target["phase1_prune_enabled"],
            "phase2_gram_novelty_policy": target[
                "phase2_gram_novelty_policy"
            ],
            "phase3_gram_novelty_policy": target[
                "phase3_gram_novelty_policy"
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
            "archived_paper_i_bundle_plus_registered_parent_contract_and_"
            "exact_application_setting_diff"
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
        CANONICAL_SR_SNAKE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3_EXECUTION_SETTINGS,
        SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3,
        canonical_sr_snake_h2o_derivative_resolved_paper_i_v3_contract_sha256,
    )

    parser = _build_adapt_arg_parser(adapt_gradient_parity_rtol=1.0e-8)
    args = parser.parse_args(argv[4:])
    expected = (
        CANONICAL_SR_SNAKE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3_EXECUTION_SETTINGS
    )
    mismatches = {
        field: {"expected": value, "actual": getattr(args, field, None)}
        for field, value in expected.items()
        if getattr(args, field, None) != value
    }
    checks = {
        "profile_request": args.sr_route_profile_request
        == SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3,
        "profile_resolved": args.sr_route_profile_resolved
        == SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3,
        "contract_digest": args.sr_route_profile_contract_sha256
        == canonical_sr_snake_h2o_derivative_resolved_paper_i_v3_contract_sha256(),
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
    audit, contract = profile_diff_audit(repo=repo)
    preflight = parser_preflight(repo, argv)
    command = {
        "schema": (
            "paper_iv_h2o_sr_paper_i_noprune_nobeam_derivative_resolved_"
            "depth50_command_v1"
        ),
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
        "schema": (
            "paper_iv_h2o_sr_paper_i_noprune_nobeam_derivative_resolved_"
            "depth50_input_manifest_v1"
        ),
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
