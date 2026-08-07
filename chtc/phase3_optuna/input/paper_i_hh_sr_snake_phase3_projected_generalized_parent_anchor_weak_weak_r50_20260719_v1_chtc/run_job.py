#!/usr/bin/env python3
"""Validate and execute one archive-only SR-SNAKE symmetric-cost job."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from evidence_validation import checkpoint_sha256, validate_parent_evidence


SCHEMA = "paper_i_hh_sr_main_all_six_r50_job_v1"
BUNDLE_ID = 'paper_i_hh_sr_snake_phase3_projected_generalized_parent_anchor_weak_weak_r50_20260719_v1_chtc'
PROFILE_REQUEST = "sr_snake_no_prune_symmetric_cost_v1"
PROFILE = (
    "supported_whitened_adaptive_trust_full_response_symmetric_cost_"
    "no_prune_v1"
)
DIGEST = "023bc7ac535ee4d88d78dd5336a59dd2fb0543c133fa0a60b009efab75422c91"
PHASE1_ENERGY_MODEL = "first_order_fs_trust_v1"
PHASE2_CURVATURE_POLICY = "measured_required_fail_closed_v1"
PHASE2_CHEAP_CURVATURE_PROXY_POLICY = "off"
NONSCIENTIFIC_ARCHIVE_OVERLAYS = {
    "pipelines/hardcoded/adapt_pipeline.py": (
        "93c0e91cd01981f5bfa1e9d1434b74296395ca0837f2901ca66ad18ac63dd42f"
    ),
    "pipelines/hardcoded/hh_continuation_scoring.py": (
        "f25b2ae3f4037758c5f1942e6e3e0c75df04f9c5c7008d8b8dacfa9f150aa492"
    ),
    "pipelines/hardcoded/hh_continuation_generators.py": (
        "8c6292c5c71f67312bdc32afc8d3908a83cb550b8f2d8871ed7f7183824e6570"
    ),
    "pipelines/hardcoded/hh_continuation_symmetry.py": (
        "5f61b9c43c253fb81bc354aace4e015f0c4f06a1e8aa8a48b24a43a11b341e01"
    ),
    "pipelines/hardcoded/hh_continuation_types.py": (
        "f24b1a670179ec17c05132b3b65f9541db54ffd888429951f7e17d6aaaf41f4c"
    ),
}
NONSCIENTIFIC_ARCHIVE_OVERLAY_SIZES = {
    "pipelines/hardcoded/adapt_pipeline.py": 1807,
    "pipelines/hardcoded/hh_continuation_scoring.py": 658,
    "pipelines/hardcoded/hh_continuation_generators.py": 664,
    "pipelines/hardcoded/hh_continuation_symmetry.py": 668,
    "pipelines/hardcoded/hh_continuation_types.py": 654,
}
REQUIRED_UNTRACKED_SOURCE_MODULES = {
    "pipelines/static_adapt/formal_manifold_outer_information.py": (
        "d0fbd924aba5b1630fce05c5701c75d2f20397ec08356d84a9d41e7794b2df91"
    ),
    "pipelines/static_adapt/formal_manifold_sr_v3_outer_bridge.py": (
        "fb8f18d159e19ce3b46fdabf7bcbab3a76611dadf65fbf027837ab7e551c2c5d"
    ),
}
REQUIRED_HASH_LOCKED_FIDELITY_FILES = {
    "pipelines/scaffold/ground_space_fidelity.py": (
        "b6a7cba65995f536faa1d9bdb7210aea69918c2cb84babd6abe34c35f7c66ae3"
    ),
    "agent_guidance/skills/paper-i-results/scripts/"
    "compute_paper_i_main_fidelities.py": (
        "5534333b6ad14a440a8b5f4e1104d388a11048c1a27b90e7f8466f048cbe1a42"
    ),
    "test/test_ground_space_fidelity.py": (
        "55ff094aca73f59362886cb5b951c9ab4b70eff2733f9e4061970be88836a8bf"
    ),
}
SIDECAR_SCHEMA = "paper_i_hh_recovery_prune_aware_qiskit_sidecar_v1"
CHECKPOINT_REPAIR_SCHEMA = "paper_i_checkpoint_execution_order_repair_v1"
FIDELITY_RECEIPT_SCHEMA = "paper_i_hh_sr_post_run_projector_fidelity_receipt_v1"
FIDELITY_POLICY_SCHEMA = "paper_i_hh_sr_post_run_projector_fidelity_policy_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _compute_post_run_projector_fidelity(
    *, manifest: dict[str, Any], result_path: Path, ledger_path: Path,
) -> dict[str, Any]:
    """Persist a reporting-only same-cutoff projector-fidelity receipt.

    The shared audit implementation replays the saved terminal ansatz and
    constructs the same-cutoff physical-sector projector.  Hashing the
    estimator ledger before and after proves this postprocessor adds no
    controller/optimizer query charge.
    """

    policy = manifest.get("segment", {}).get(
        "post_run_projector_fidelity_policy", {}
    )
    expected_policy = {
        "schema": FIDELITY_POLICY_SCHEMA,
        "reference": "same_cutoff_physical_sector_ground_space_projector",
        "usage_scope": "post_run_reporting_only",
        "controller_decision_eligible": False,
        "optimizer_input_eligible": False,
        "stopping_input_eligible": False,
        "s_alg_charged": False,
        "persistence_required": True,
    }
    if policy != expected_policy:
        raise ValueError("post-run projector-fidelity policy drift")
    physics = manifest["physics"]
    if (
        physics.get("same_cutoff_reference") is not True
        or int(physics["n_ph_work"]) != int(physics["n_ph_reference"])
    ):
        raise ValueError("projector fidelity requires the locked same cutoff")
    script = Path(
        "agent_guidance/skills/paper-i-results/scripts/"
        "compute_paper_i_main_fidelities.py"
    )
    if not script.is_file():
        raise ValueError("projector-fidelity audit implementation is missing")
    module_spec = importlib.util.spec_from_file_location(
        "paper_i_main_fidelity_audit", script
    )
    if module_spec is None or module_spec.loader is None:
        raise ValueError("could not load projector-fidelity audit implementation")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    result_sha256 = sha256(result_path)
    ledger_sha256_before = sha256(ledger_path)
    computed = module.compute_row_fidelity({
        "source_json": str(result_path.resolve()),
        "source_sha256": result_sha256,
        "prefix_operator_count": None,
    })
    ledger_sha256_after = sha256(ledger_path)
    if ledger_sha256_after != ledger_sha256_before:
        raise ValueError("reporting-only fidelity changed the estimator ledger")
    ground = computed.get("ground_space_fidelity", {})
    if (
        ground.get("status") != "ok"
        or ground.get("usage_scope") != "post_run_reporting_only"
        or ground.get("s_alg_charged") is not False
        or ground.get("same_cutoff_verified") is not True
        or int(ground.get("working_cutoff", -1)) != int(physics["n_ph_work"])
        or int(ground.get("reference_cutoff", -2))
        != int(physics["n_ph_reference"])
    ):
        raise ValueError("projector-fidelity receipt failed its reporting-only gate")
    return {
        "schema": FIDELITY_RECEIPT_SCHEMA,
        "status": "pass",
        "policy": expected_policy,
        "source_result": str(result_path),
        "source_result_sha256": result_sha256,
        "implementation": str(script),
        "implementation_sha256": sha256(script),
        "estimator_ledger_sha256_before": ledger_sha256_before,
        "estimator_ledger_sha256_after": ledger_sha256_after,
        "estimator_query_delta": 0,
        "fidelity": float(computed["fidelity"]),
        "infidelity": float(computed["one_minus_fidelity"]),
        "state_replay_source": computed["state_replay_source"],
        "ground_space_fidelity": ground,
    }


def argument_default_from_source(source: Path, option: str) -> Any:
    """Read one argparse default without importing scientific dependencies."""

    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    matches: list[Any] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        first = node.args[0]
        if not isinstance(first, ast.Constant) or first.value != option:
            continue
        defaults = [
            keyword.value for keyword in node.keywords if keyword.arg == "default"
        ]
        if len(defaults) != 1 or not isinstance(defaults[0], ast.Constant):
            raise ValueError(f"{option} default is not one constant")
        matches.append(defaults[0].value)
    if len(matches) != 1:
        raise ValueError(f"expected one argparse definition for {option}")
    return matches[0]


def expected_argv(manifest: dict[str, Any]) -> list[str]:
    physics = manifest["physics"]
    paths = manifest["paths"]
    slug = str(manifest["regime_slug"])
    target = int(manifest["segment"]["target_controller_round"])
    return [
        "python3", "-m", "pipelines.static_adapt.adapt_pipeline",
        "--problem", "hh", "--L", "2", "--ordering", "blocked",
        "--boundary", "open", "--t", "1.0", "--dv", "0.0",
        "--omega0", "1.0", "--boson-encoding", "binary",
        "--u", str(physics["u_over_t"]), "--g-ep", str(physics["g_ep"]),
        "--n-ph-max", str(physics["n_ph_work"]),
        "--sr-route-profile", PROFILE_REQUEST,
        "--phase-live-hysteresis-disabled",
        "--adapt-disable-hh-seed",
        "--adapt-max-depth", str(target),
        "--adapt-segment-id", (
            f"{slug}-sr-main-symcost-noprune-r0-r{target}-20260718-v1"
        ),
        "--adapt-segment-target-controller-round", str(target),
        "--adapt-segment-target-depth", str(target),
        "--adapt-segment-max-new-admissions", str(target),
        "--adapt-current-json-every-depth", "1",
        "--adapt-current-json", str(paths["current_json"]),
        "--adapt-estimator-call-ledger-json", str(paths["ledger_json"]),
        "--output-json", str(paths["result_json"]),
        "--skip-pdf",
    ]


def validate(manifest_path: Path, manifest: dict[str, Any]) -> list[str]:
    if manifest.get("schema") != SCHEMA or manifest.get("bundle_id") != BUNDLE_ID:
        raise ValueError("unexpected job schema/bundle")
    slug = str(manifest.get("regime_slug", ""))
    if not slug or "/" in slug or ".." in slug:
        raise ValueError("unsafe regime slug")
    route = manifest["route_identity"]
    if route.get("profile_request") != PROFILE_REQUEST:
        raise ValueError("profile request drift")
    if route.get("profile_resolved") != PROFILE or route.get("profile_contract_sha256") != DIGEST:
        raise ValueError("profile resolution/digest drift")
    sys.path.insert(0, str(Path.cwd()))
    from pipelines.static_adapt.sr_snake_route_profile import (
        canonical_sr_snake_contract,
        canonical_sr_snake_contract_sha256,
        normalize_sr_route_profile_request,
    )
    if normalize_sr_route_profile_request(PROFILE_REQUEST) != PROFILE:
        raise ValueError("runtime source resolves candidate profile incorrectly")
    if canonical_sr_snake_contract_sha256(PROFILE_REQUEST) != DIGEST:
        raise ValueError("runtime source candidate digest mismatch")
    contract = canonical_sr_snake_contract(PROFILE_REQUEST)
    if route.get("profile_contract") != contract:
        raise ValueError("manifest candidate profile contract drift")
    from pipelines.static_adapt.cli_config import _build_adapt_arg_parser
    parser = _build_adapt_arg_parser(adapt_gradient_parity_rtol=1.0e-7)
    parsed = parser.parse_args(manifest["command"]["argv"][3:])
    if parsed.sr_route_profile_contract_sha256 != DIGEST:
        raise ValueError("exact scientific argv resolved the wrong route digest")
    if parsed.phase_live_hysteresis_enabled is not False:
        raise ValueError("full-response route enabled phase-live hysteresis")
    if contract["execution_settings"].get("phase_live_hysteresis_enabled") is not False:
        raise ValueError("route contract did not disable phase-live hysteresis")
    if parsed.adapt_disable_hh_seed is not True:
        raise ValueError("exact scientific argv did not preserve disabled HH preseed")
    if contract["execution_settings"].get("adapt_finite_angle_fallback") is not False:
        raise ValueError("candidate finite-angle fallback must be disabled")
    if contract["semantic_invariants"].get("finite_angle_fallback_active") is not False:
        raise ValueError("candidate finite-angle semantic invariant must be false")
    if contract["execution_settings"].get("phase3_enable_rescue") is not False:
        raise ValueError("candidate Phase-III rescue must be disabled")
    phase12 = route.get("phase12_energy_model_contract", {})
    expected_phase12 = {
        "phase1_energy_model": PHASE1_ENERGY_MODEL,
        "phase2_curvature_policy": PHASE2_CURVATURE_POLICY,
        "phase2_cheap_curvature_proxy_policy": (
            PHASE2_CHEAP_CURVATURE_PROXY_POLICY
        ),
    }
    for key, expected in expected_phase12.items():
        if contract["execution_settings"].get(key) != expected:
            raise ValueError(f"candidate profile {key} drift")
        if contract["semantic_invariants"].get(key) != expected:
            raise ValueError(f"candidate semantic invariant {key} drift")
        if phase12.get(key) != expected:
            raise ValueError(f"candidate job Phase-I/II contract {key} drift")
    if contract["semantic_invariants"].get(
        "phase1_phase2_lambda_f_proxy_active"
    ) is not False:
        raise ValueError("candidate lambda-F proxy invariant must be inactive")
    if phase12.get("lambda_f_proxy_flags_forbidden") is not True:
        raise ValueError("candidate job does not forbid lambda-F proxy flags")
    if phase12.get("missing_curvature_failure_policy") != "abort_run_v1":
        raise ValueError("candidate Phase-II missing-curvature policy must abort")
    execution_settings = contract["execution_settings"]
    semantic_invariants = contract["semantic_invariants"]
    required_execution = {
        "phase2_gram_novelty_policy": "fallback_only_v1",
        "phase3_gram_novelty_policy": "fallback_only_v1",
        "phase1_prune_enabled": False,
        "adapt_beam_live_branches": 1,
        "adapt_beam_children_per_parent": 1,
        "phase2_enable_batching": False,
        "phase3_enable_batching": False,
        "phase3_shadow_damping_policy": "off",
        "phase3_response_coordinate_scope": "full_active_plus_singleton_v1",
        "historical_singleton_coordinate_solve_scope": "phase3_only_v1",
        "historical_singleton_coordinate_solve_policy": (
            "supported_metric_whitened_eigh_v1"
        ),
        "historical_singleton_trust_region_update_policy": (
            "displacement_calibrated_unbounded_v2"
        ),
        "adapt_accepted_refit_scope": "full_ansatz_v1",
        "adapt_accepted_refit_coordinate_chart": "supported_fs_whitened_fixed_v1",
        "adapt_accepted_refit_base_chart_policy": (
            "expanded_runtime_projected_logical_v1"
        ),
        "adapt_full_refit_every": 0,
        "adapt_final_full_refit": "false",
        "phase3_hardware_cost_normalization_mode": (
            "family_robust_symmetric_arctan_v1"
        ),
        "adapt_seed": 7,
        "adapt_disable_hh_seed": True,
    }
    for key, expected in required_execution.items():
        if execution_settings.get(key) != expected:
            raise ValueError(f"candidate executable contract drift: {key}")
    required_semantics = {
        "ordinary_phase2_novelty_multiplier_active": False,
        "ordinary_phase3_novelty_multiplier_active": False,
        "all_energy_models_infeasible_novelty_fallback_active": True,
        "all_energy_models_infeasible_novelty_fallback_telemetry_required": True,
        "pruning_active": False,
        "phase2_supported_whitening_active": False,
        "phase3_supported_whitening_active": True,
        "negative_curvature_escape_active": False,
        "periodic_full_refit_active": False,
        "terminal_full_refit_active": False,
        "terminal_prune_active": False,
    }
    for key, expected in required_semantics.items():
        if semantic_invariants.get(key) != expected:
            raise ValueError(f"candidate semantic invariant drift: {key}")
    physics = manifest["physics"]
    if physics.get("same_cutoff_reference") is not True:
        raise ValueError("same-cutoff contract disabled")
    if int(physics["n_ph_work"]) != int(physics["n_ph_reference"]):
        raise ValueError("working/reference cutoff mismatch")
    segment = manifest["segment"]
    target = int(segment["target_controller_round"])
    if target != 50:
        raise ValueError("main six-regime bundle requires exact round-50 horizon")
    expected_segment = {
        "source_controller_round": 0, "source_depth": 0,
        "target_controller_round": target, "target_depth": target,
        "max_new_admissions": target,
    }
    for key, expected in expected_segment.items():
        if int(segment[key]) != expected:
            raise ValueError(f"segment drift: {key}")
    argv = [str(token) for token in manifest["command"]["argv"]]
    if argv != expected_argv(manifest):
        raise ValueError("execution argv differs from exact normalized parent command")
    forbidden = {
        "--adapt-exact-gs-override",
        "--adapt-exact-gs-reference-json", "--phase1-no-prune",
        "--phase2-gram-novelty-policy", "--phase3-gram-novelty-policy",
        "--phase1-lambda-F", "--phase1-lambda-f",
        "--phase2-lambda-F", "--phase2-lambda-f",
        "--phase2-cheap-curvature-proxy-policy",
    }
    present = forbidden.intersection(argv)
    if present:
        raise ValueError(f"forbidden repeated/profile override flags: {sorted(present)}")
    oracle_default = argument_default_from_source(
        Path("pipelines/static_adapt/cli_config.py"),
        "--phase3-oracle-gradient-mode",
    )
    if str(oracle_default).strip().lower() != "off":
        raise ValueError("source default enabled Phase-III oracle gradients")
    source = manifest["source_lock"]
    archive = Path(source["source_archive"])
    if not archive.is_file() or sha256(archive) != source["source_archive_sha256"]:
        raise ValueError("source archive missing/hash mismatch")
    for path_key in (
        "physics_reference_lock", "source_revision_manifest", "source_archive_manifest"
    ):
        record = Path(source[path_key])
        if not record.is_file() or sha256(record) != source[f"{path_key}_sha256"]:
            raise ValueError(f"source-lock record missing/hash mismatch: {path_key}")
    revision = load(Path(source["source_revision_manifest"]))
    archive_manifest = load(Path(source["source_archive_manifest"]))
    source_commit = str(revision.get("git_commit") or "")
    source_tree = str(revision.get("git_tree") or "")
    source_archive_sha256 = str(archive_manifest.get("archive_sha256") or "")
    if (
        len(source_commit) != 40
        or len(source_tree) != 40
        or revision.get("profile_contract_sha256") != DIGEST
        or revision.get("phase1_energy_model") != PHASE1_ENERGY_MODEL
        or revision.get("phase2_curvature_policy") != PHASE2_CURVATURE_POLICY
        or revision.get("phase2_cheap_curvature_proxy_policy")
        != PHASE2_CHEAP_CURVATURE_PROXY_POLICY
        or revision.get("git_role") != "base_ancestry_metadata_only"
        or revision.get("dirty_live_source_lock") is not False
        or revision.get("executable_source_authority")
        != "immutable_parent_archive_plus_hash_locked_overlay_inventory_v1"
    ):
        raise ValueError("source-revision manifest drift")
    if (
        archive_manifest.get("git_commit") != source_commit
        or archive_manifest.get("git_tree") != source_tree
        or archive_manifest.get("git_role") != "base_ancestry_metadata_only"
        or archive_manifest.get("executable_source_authority")
        != "derived_archive_sha256_plus_complete_per_file_sha256_inventory_v1"
        or len(source_archive_sha256) != 64
    ):
        raise ValueError("source-archive manifest drift")
    expected_overlays = {
        relative: {
            "sha256": digest,
            "size_bytes": NONSCIENTIFIC_ARCHIVE_OVERLAY_SIZES[relative],
            "mode": "0644",
            "classification": "compatibility_import_shim_only",
            "tracked_in_frozen_commit": False,
        }
        for relative, digest in NONSCIENTIFIC_ARCHIVE_OVERLAYS.items()
    }
    expected_untracked_modules = {
        relative: {
            "sha256": digest,
            "classification": "required_executable_live_source_module",
            "tracked_in_base_commit": False,
        }
        for relative, digest in REQUIRED_UNTRACKED_SOURCE_MODULES.items()
    }
    revision_fidelity = revision.get("required_hash_locked_fidelity_files")
    archive_fidelity = archive_manifest.get(
        "required_hash_locked_fidelity_files"
    )
    source_fidelity = source.get("required_hash_locked_fidelity_files")
    revision_fidelity_overlays = revision.get("required_untracked_hash_overlays")
    archive_fidelity_overlays = archive_manifest.get(
        "required_untracked_hash_overlays"
    )
    source_fidelity_overlays = source.get("required_untracked_hash_overlays")
    if (
        archive_manifest.get("worker_source_mode")
        != "immutable_parent_archive_plus_hash_locked_overlay_v1"
        or revision.get("non_scientific_archive_overlays") != expected_overlays
        or archive_manifest.get("non_scientific_archive_overlays")
        != expected_overlays
        or source.get("worker_source_mode")
        != "immutable_parent_archive_plus_hash_locked_overlay_v1"
        or source.get("non_scientific_archive_overlays") != expected_overlays
        or revision.get("required_untracked_source_modules")
        != expected_untracked_modules
        or archive_manifest.get("required_untracked_source_modules")
        != expected_untracked_modules
        or source.get("required_untracked_source_modules")
        != expected_untracked_modules
        or revision_fidelity != archive_fidelity
        or archive_fidelity != source_fidelity
        or not isinstance(archive_fidelity, dict)
        or set(archive_fidelity) != set(REQUIRED_HASH_LOCKED_FIDELITY_FILES)
        or revision_fidelity_overlays != archive_fidelity_overlays
        or archive_fidelity_overlays != source_fidelity_overlays
        or not isinstance(archive_fidelity_overlays, dict)
        or not set(archive_fidelity_overlays).issubset(
            REQUIRED_HASH_LOCKED_FIDELITY_FILES
        )
    ):
        raise ValueError("compatibility-overlay provenance drift")
    archive_files = archive_manifest.get("files", {})
    for relative, expected_hash in NONSCIENTIFIC_ARCHIVE_OVERLAYS.items():
        path = Path(relative)
        if (
            not path.is_file()
            or sha256(path) != expected_hash
            or path.stat().st_size != NONSCIENTIFIC_ARCHIVE_OVERLAY_SIZES[relative]
            or archive_files.get(relative, {}).get("sha256") != expected_hash
            or archive_files.get(relative, {}).get("size_bytes")
            != NONSCIENTIFIC_ARCHIVE_OVERLAY_SIZES[relative]
        ):
            raise ValueError(f"compatibility overlay missing/drifted: {relative}")
    for relative, expected_hash in REQUIRED_UNTRACKED_SOURCE_MODULES.items():
        path = Path(relative)
        if (
            not path.is_file()
            or sha256(path) != expected_hash
            or archive_files.get(relative, {}).get("sha256") != expected_hash
        ):
            raise ValueError(
                f"required untracked source module missing/drifted: {relative}"
            )
    for relative, expected_hash in REQUIRED_HASH_LOCKED_FIDELITY_FILES.items():
        record = archive_fidelity.get(relative, {})
        path = Path(relative)
        if (
            record.get("sha256") != expected_hash
            or record.get("classification")
            != "reporting_only_fidelity_source_or_test"
            or not isinstance(record.get("tracked_in_parent_archive"), bool)
            or not path.is_file()
            or sha256(path) != expected_hash
            or archive_files.get(relative, {}).get("sha256") != expected_hash
        ):
            raise ValueError(
                f"required hash-locked fidelity source missing/drifted: {relative}"
            )
    for relative, record in archive_fidelity_overlays.items():
        if (
            record.get("sha256")
            != REQUIRED_HASH_LOCKED_FIDELITY_FILES[relative]
            or record.get("classification")
            != "reporting_only_fidelity_archive_overlay"
            or record.get("tracked_in_parent_archive") is not False
            or archive_fidelity[relative].get("tracked_in_parent_archive")
            is not False
        ):
            raise ValueError(
                f"required untracked fidelity overlay drift: {relative}"
            )
    if (
        source.get("git_commit") != source_commit
        or source.get("git_tree") != source_tree
        or source.get("source_archive_sha256") != source_archive_sha256
    ):
        raise ValueError("frozen source commit/tree/archive digest drift")
    environment = {str(k): str(v) for k, v in manifest["environment"].items()}
    if environment.get("PYTHONPATH") != "/work":
        raise ValueError("worker PYTHONPATH must resolve only the extracted archive")
    if environment.get("PYTHONNOUSERSITE") != "1":
        raise ValueError("PYTHONNOUSERSITE must be enabled")
    if manifest.get("cache_policy") != {
        "candidate_record_cache": "disk",
        "hh_pool_cache": "disk",
        "hh_pool_cache_scope": "exact",
        "cache_namespace": "job_local_empty_on_start_v1",
        "cache_semantics": "performance_only_no_scientific_fallback_v1",
    }:
        raise ValueError("job cache policy drift")
    required_evidence = manifest.get("evidence_requirements", {})
    for gate in (
        "exact_s_alg_ledger_closure_required",
        "active_prefix_estimator_receipt_each_round_required",
        "terminal_estimator_closure_receipt_required",
        "fallback_telemetry_required",
        "full_active_plus_singleton_response_each_round_required",
        "full_accepted_refit_each_round_required",
        "symmetry_and_padding_leakage_gate_required",
        "exact_round_50_horizon_required",
    ):
        if required_evidence.get(gate) is not True:
            raise ValueError(f"required evidence gate disabled: {gate}")
    return argv


def terminal_checkpoint(result: dict[str, Any], target_round: int) -> dict[str, Any]:
    checkpoints = result.get("adapt_vqe", {}).get("active_prefix_checkpoints", [])
    matches = [
        item for item in checkpoints
        if isinstance(item, dict)
        and int(item.get("outer_iteration", -1)) == target_round
        and str(item.get("checkpoint_kind")) == "post_admission_prune"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one round-{target_round} checkpoint; got {len(matches)}"
        )
    checkpoint = matches[0]
    digest = str(checkpoint.get("checkpoint_sha256") or "")
    if len(digest) != 64:
        raise ValueError("terminal checkpoint missing SHA-256")
    return checkpoint


def validate_result_and_compile(manifest: dict[str, Any]) -> dict[str, Any]:
    paths = {key: Path(value) for key, value in manifest["paths"].items()}
    result_path = paths["result_json"]
    result = load(result_path)
    current = load(paths["current_json"])
    ledger = load(paths["ledger_json"])
    exact = float(result["ground_state"]["exact_energy"])
    expected_exact = float(manifest["physics"]["expected_exact_energy"])
    tolerance = float(manifest["physics"]["exact_energy_tolerance"])
    if abs(exact - expected_exact) > tolerance:
        raise ValueError(f"runtime exact-energy mismatch: {exact} vs {expected_exact}")
    settings = result.get("settings", {})
    adapt = result.get("adapt_vqe", {})
    for payload in (settings, adapt):
        if payload.get("sr_route_profile_resolved") != PROFILE:
            raise ValueError("result profile resolution drift")
        if payload.get("sr_route_profile_contract_sha256") != DIGEST:
            raise ValueError("result profile digest drift")
        if payload.get("phase1_energy_model") != PHASE1_ENERGY_MODEL:
            raise ValueError("result Phase-I energy model drift")
        if payload.get("phase2_curvature_policy") != PHASE2_CURVATURE_POLICY:
            raise ValueError("result Phase-II curvature policy drift")
        if payload.get("phase2_cheap_curvature_proxy_policy") != (
            PHASE2_CHEAP_CURVATURE_PROXY_POLICY
        ):
            raise ValueError("result Phase-II cheap-proxy policy drift")
    target_round = int(manifest["segment"]["target_controller_round"])
    target_admissions = int(manifest["segment"]["max_new_admissions"])
    evidence = validate_parent_evidence(
        result=result,
        current=current,
        ledger_sidecar=ledger,
        profile=PROFILE,
        digest=DIGEST,
        target_round=target_round,
        target_new_admissions=target_admissions,
        require_supported_rank=True,
    )
    fidelity = _compute_post_run_projector_fidelity(
        manifest=manifest,
        result_path=result_path,
        ledger_path=paths["ledger_json"],
    )
    dump(paths["ground_space_fidelity_json"], fidelity)
    checkpoint = terminal_checkpoint(result, target_round)
    result_digest = sha256(result_path)
    checkpoint_digest = str(checkpoint["checkpoint_sha256"])
    command = [
        sys.executable,
        "pipelines/exact_bench/paper_i_hh_recovery_prefix_qiskit_sidecar.py",
        "--result-json", str(result_path),
        "--outer-iteration", str(target_round),
        "--checkpoint-kind", "post_admission_prune",
        "--expected-result-sha256", result_digest,
        "--expected-checkpoint-sha256", checkpoint_digest,
        "--repair-permutation-only-execution-order",
        "--repaired-checkpoint-json", str(paths["repaired_terminal_checkpoint_json"]),
        "--require-fixed-prefix-replay",
        "--output-json", str(paths["qiskit_cost_sidecar_json"]),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise ValueError(f"terminal Qiskit sidecar failed with exit {completed.returncode}")
    sidecar = load(paths["qiskit_cost_sidecar_json"])
    repaired = load(paths["repaired_terminal_checkpoint_json"])
    if sidecar.get("schema") != SIDECAR_SCHEMA or sidecar.get("status") != "ok":
        raise ValueError("unexpected Qiskit sidecar schema/status")
    if repaired.get("schema") != CHECKPOINT_REPAIR_SCHEMA:
        raise ValueError("unexpected repaired-checkpoint schema")
    repaired_source = repaired.get("source", {})
    repaired_summary = repaired.get("repair", {})
    repaired_checkpoint = repaired.get("repaired_checkpoint", {})
    if repaired_source.get("result_sha256") != result_digest:
        raise ValueError("repaired-checkpoint result hash drift")
    if repaired_source.get("checkpoint_sha256") != checkpoint_digest:
        raise ValueError("repaired-checkpoint source hash drift")
    if repaired_summary.get("substantive_term_changes") is not False:
        raise ValueError("checkpoint repair was not permutation-only")
    repaired_digest = checkpoint_sha256(repaired_checkpoint)
    if repaired_checkpoint.get("checkpoint_sha256") != repaired_digest:
        raise ValueError("repaired checkpoint SHA-256 mismatch")
    if repaired_summary.get("repaired_checkpoint_sha256") != repaired_digest:
        raise ValueError("repair summary/repaired checkpoint hash mismatch")
    replay = sidecar.get("fixed_prefix_replay", {})
    if replay.get("status") != "pass" or replay.get("prefix_reconstructed") is not True:
        raise ValueError("Qiskit sidecar fixed-prefix replay did not pass")
    if float(replay.get("energy_abs_discrepancy", float("inf"))) > 1.0e-12:
        raise ValueError("Qiskit sidecar fixed-prefix replay energy mismatch")
    repair = sidecar.get("source", {}).get("checkpoint_execution_order_repair", {})
    summary = repair.get("repair_summary", repair)
    if summary.get("substantive_term_changes") not in {False, None}:
        raise ValueError("checkpoint repair changed substantive operator terms")
    source = sidecar.get("source", {})
    if source.get("result_sha256") != result_digest:
        raise ValueError("Qiskit sidecar result hash drift")
    if source.get("source_checkpoint_sha256") != checkpoint_digest:
        raise ValueError("Qiskit sidecar checkpoint hash drift")
    if int(source.get("outer_iteration", -1)) != target_round:
        raise ValueError("Qiskit sidecar compiled the wrong controller round")
    if source.get("checkpoint_kind") != "post_admission_prune":
        raise ValueError("Qiskit sidecar compiled the wrong checkpoint kind")
    prefix = sidecar.get("prefix", {})
    if prefix.get("prune_aware") is not True:
        raise ValueError("Qiskit sidecar is not prune aware")
    if int(prefix.get("active_ansatz_depth", -1)) != int(
        checkpoint.get("active_ansatz_depth", -2)
    ):
        raise ValueError("Qiskit sidecar active depth drift")
    if prefix.get("ordered_active_operator_labels") != checkpoint.get(
        "ordered_active_operator_labels"
    ):
        raise ValueError("Qiskit sidecar operator ordering drift")
    current = sidecar.get("current_jr_fake_marrakesh_convention", {})
    historical = sidecar.get("historical_displayed_convention", {})
    if current.get("status") != "ok" or historical.get("status") != "ok":
        raise ValueError("one or both Qiskit compile conventions failed")
    if (
        historical.get("identity") != "table_i_basis_gate_transpile_v1"
        or historical.get("backend") is not None
        or int(historical.get("optimization_level", -1)) != 0
        or int(historical.get("seed_transpiler", -1)) != 7
    ):
        raise ValueError("historical displayed Qiskit convention drift")
    if (
        current.get("identity") != "jr_signed_runtime_fake_marrakesh_transpile_v1"
        or current.get("requested_backend") != "FakeMarrakesh"
        or int(current.get("optimization_level", -1)) != 1
        or int(current.get("seed_transpiler", -1)) != 7
    ):
        raise ValueError("current JR FakeMarrakesh convention drift")
    comparison = sidecar.get("convention_comparison", {})
    if comparison.get("same_convention") is not False:
        raise ValueError("historical/current Qiskit conventions were conflated")
    archive_manifest = load(Path(manifest["source_lock"]["source_archive_manifest"]))
    archive_files = archive_manifest.get("files", {})
    implementation_sources = sidecar.get("implementation_sources", {})
    expected_implementations = {
        "postprocessor": "pipelines/exact_bench/paper_i_hh_recovery_prefix_qiskit_sidecar.py",
        "historical_table_i_compiler": "pipelines/exact_bench/table_i_qiskit_resource_compile.py",
        "ansatz_circuit_builder": "pipelines/hardcoded/adapt_circuit_execution.py",
        "backend_transpile_tools": "pipelines/qiskit_backend_tools.py",
    }
    for key, relative in expected_implementations.items():
        record = implementation_sources.get(key, {})
        archived = archive_files.get(relative, {})
        if not str(record.get("path", "")).endswith(relative):
            raise ValueError(f"Qiskit implementation source path drift: {key}")
        if not archived or record.get("sha256") != archived.get("sha256"):
            raise ValueError(f"Qiskit implementation source hash drift: {key}")
    for name, convention in (("historical", historical), ("current", current)):
        metrics = convention.get("metrics", {})
        for key in ("N2q", "D2q", "Dc"):
            if int(metrics.get(key, -1)) < 0:
                raise ValueError(f"{name} Qiskit metric missing/invalid: {key}")
    return {
        "schema": "paper_i_hh_sr_symcost_noprune_validation_v1",
        "status": "pass",
        "result_sha256": result_digest,
        "same_cutoff_exact_energy": exact,
        "expected_exact_energy": expected_exact,
        "exact_energy_abs_discrepancy": abs(exact - expected_exact),
        "terminal_checkpoint_sha256": checkpoint_digest,
        "qiskit_sidecar": str(paths["qiskit_cost_sidecar_json"]),
        "qiskit_sidecar_sha256": sha256(paths["qiskit_cost_sidecar_json"]),
        "repaired_terminal_checkpoint": str(paths["repaired_terminal_checkpoint_json"]),
        "repaired_terminal_checkpoint_sha256": sha256(paths["repaired_terminal_checkpoint_json"]),
        "historical_metrics": historical.get("metrics"),
        "current_fake_marrakesh_metrics": current.get("metrics"),
        "fixed_prefix_replay": replay,
        "scientific_evidence_validation": evidence,
        "post_run_projector_fidelity": str(paths["ground_space_fidelity_json"]),
        "post_run_projector_fidelity_sha256": sha256(
            paths["ground_space_fidelity_json"]
        ),
        "post_run_projector_fidelity_receipt": fidelity,
        "qiskit_implementation_sources": implementation_sources,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    validate_only = bool(args and args[0] == "--validate-only")
    if validate_only:
        args = args[1:]
    if len(args) != 1:
        raise SystemExit("usage: run_job.py [--validate-only] JOB.json")
    manifest_path = Path(args[0])
    manifest = load(manifest_path)
    execution_argv = validate(manifest_path, manifest)
    if validate_only:
        print(json.dumps({
            "status": "pass",
            "job": f"jobs/{manifest_path.name}",
        }, sort_keys=True))
        return 0

    paths = {key: Path(value) for key, value in manifest["paths"].items()}
    for path in paths.values():
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
    environment = {str(k): str(v) for k, v in manifest["environment"].items()}
    for key, value in environment.items():
        if key.endswith("_CACHE_DIR"):
            cache = Path(value)
            if cache.exists():
                raise ValueError(f"job-local cache must start absent: {cache}")
            cache.mkdir(parents=True)
    normalized = {
        "schema": "paper_i_hh_sr_symcost_noprune_runtime_manifest_v1",
        "job_manifest": str(manifest_path),
        "job_manifest_sha256": sha256(manifest_path),
        "command_argv": execution_argv,
        "environment": environment,
        "route_identity": manifest["route_identity"],
        "physics": manifest["physics"],
        "segment": manifest["segment"],
        "cache_policy": manifest["cache_policy"],
        "evidence_requirements": manifest["evidence_requirements"],
        "source_lock": manifest["source_lock"],
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    dump(paths["normalized_runtime_manifest_json"], normalized)
    execution = {
        **normalized,
        "schema": "paper_i_hh_sr_symcost_noprune_execution_v1",
        "status": "running",
        "exit_code": None,
    }
    dump(paths["execution_json"], execution)
    env = os.environ.copy()
    env.update(environment)
    returncode = 70
    try:
        completed = subprocess.run(execution_argv, env=env, check=False)
        returncode = int(completed.returncode)
        if returncode == 0:
            validation = validate_result_and_compile(manifest)
            dump(paths["validation_json"], validation)
        execution["status"] = "completed" if returncode == 0 else "failed"
        execution["exit_code"] = returncode
        return returncode
    except Exception as exc:
        if returncode == 0:
            returncode = 70
        execution["status"] = "failed"
        execution["exit_code"] = returncode
        execution["validation_error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        execution["finished_utc"] = datetime.now(timezone.utc).isoformat()
        execution["artifacts"] = {
            key: {
                "path": str(path), "exists": path.is_file(),
                "sha256": sha256(path) if path.is_file() else None,
                "size_bytes": path.stat().st_size if path.is_file() else None,
            }
            for key, path in paths.items() if key != "output_root"
        }
        dump(paths["execution_json"], execution)


if __name__ == "__main__":
    raise SystemExit(main())
