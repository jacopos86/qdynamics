#!/usr/bin/env python3
"""Fail-closed validator for one fetched symmetric-cost no-prune archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from evidence_validation import checkpoint_sha256, validate_parent_evidence


PROFILE = (
    "supported_whitened_adaptive_trust_full_response_symmetric_cost_"
    "fs_prune_nodamping_v1"
)
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
BUNDLE = Path(__file__).resolve().parent
SIDECAR_SCHEMA = "paper_i_hh_recovery_prune_aware_qiskit_sidecar_v1"
CHECKPOINT_REPAIR_SCHEMA = "paper_i_checkpoint_execution_order_repair_v1"
FIDELITY_RECEIPT_SCHEMA = "paper_i_hh_sr_post_run_projector_fidelity_receipt_v1"
QISKIT_IMPLEMENTATION_PATHS = {
    "postprocessor": (
        "pipelines/exact_bench/paper_i_hh_recovery_prefix_qiskit_sidecar.py"
    ),
    "historical_table_i_compiler": (
        "pipelines/exact_bench/table_i_qiskit_resource_compile.py"
    ),
    "ansatz_circuit_builder": "pipelines/hardcoded/adapt_circuit_execution.py",
    "backend_transpile_tools": "pipelines/qiskit_backend_tools.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected object: {path}")
    return payload


def safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            name = PurePosixPath(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or member.issym()
                or member.islnk()
                or not (member.isfile() or member.isdir())
                or any(part in {".DS_Store", "__MACOSX"} or part.startswith("._") for part in name.parts)
            ):
                raise ValueError(f"unsafe transfer member: {member.name}")
        handle.extractall(destination, filter="data")


def find_output(root: Path) -> Path:
    matches = list(root.rglob("validation.json"))
    if len(matches) != 1:
        raise ValueError(f"expected one validation.json, found {len(matches)}")
    return matches[0].parent


def validate(root: Path) -> dict[str, Any]:
    source_revision = load(BUNDLE / "source_revision_manifest.json")
    source_archive_manifest = load(BUNDLE / "source_archive_manifest.json")
    digest = str(source_revision.get("profile_contract_sha256") or "")
    source_commit = str(source_revision.get("git_commit") or "")
    source_tree = str(source_revision.get("git_tree") or "")
    source_archive_sha256 = str(
        source_archive_manifest.get("archive_sha256") or ""
    )
    if (
        len(source_commit) != 40
        or len(source_tree) != 40
        or len(digest) != 64
        or len(source_archive_sha256) != 64
        or source_revision.get("phase1_energy_model") != PHASE1_ENERGY_MODEL
        or source_revision.get("phase2_curvature_policy")
        != PHASE2_CURVATURE_POLICY
        or source_revision.get("phase2_cheap_curvature_proxy_policy")
        != PHASE2_CHEAP_CURVATURE_PROXY_POLICY
        or source_revision.get("git_role") != "base_ancestry_metadata_only"
        or source_revision.get("dirty_live_source_lock") is not False
        or source_revision.get("executable_source_authority")
        != "immutable_parent_archive_plus_hash_locked_overlay_inventory_v1"
        or source_archive_manifest.get("git_commit") != source_commit
        or source_archive_manifest.get("git_tree") != source_tree
        or source_archive_manifest.get("git_role")
        != "base_ancestry_metadata_only"
        or source_archive_manifest.get("executable_source_authority")
        != "derived_archive_sha256_plus_complete_per_file_sha256_inventory_v1"
    ):
        raise ValueError("local bundle source authority is incomplete")
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
            "sha256": module_digest,
            "classification": "required_executable_live_source_module",
            "tracked_in_base_commit": False,
        }
        for relative, module_digest in REQUIRED_UNTRACKED_SOURCE_MODULES.items()
    }
    revision_fidelity = source_revision.get("required_hash_locked_fidelity_files")
    archive_fidelity = source_archive_manifest.get(
        "required_hash_locked_fidelity_files"
    )
    revision_fidelity_overlays = source_revision.get(
        "required_untracked_hash_overlays"
    )
    archive_fidelity_overlays = source_archive_manifest.get(
        "required_untracked_hash_overlays"
    )
    source_archive = BUNDLE / "source_locked.tar.gz"
    if (
        source_archive_manifest.get("worker_source_mode")
        != "immutable_parent_archive_plus_hash_locked_overlay_v1"
        or source_revision.get("non_scientific_archive_overlays")
        != expected_overlays
        or source_archive_manifest.get("non_scientific_archive_overlays")
        != expected_overlays
        or source_revision.get("required_untracked_source_modules")
        != expected_untracked_modules
        or source_archive_manifest.get("required_untracked_source_modules")
        != expected_untracked_modules
        or revision_fidelity != archive_fidelity
        or not isinstance(archive_fidelity, dict)
        or set(archive_fidelity) != set(REQUIRED_HASH_LOCKED_FIDELITY_FILES)
        or revision_fidelity_overlays != archive_fidelity_overlays
        or not isinstance(archive_fidelity_overlays, dict)
        or not set(archive_fidelity_overlays).issubset(
            REQUIRED_HASH_LOCKED_FIDELITY_FILES
        )
        or not source_archive.is_file()
        or sha256(source_archive) != source_archive_sha256
    ):
        raise ValueError("local compatibility-overlay source authority drift")
    for relative, expected_hash in NONSCIENTIFIC_ARCHIVE_OVERLAYS.items():
        record = source_archive_manifest.get("files", {}).get(relative, {})
        if (
            record.get("sha256") != expected_hash
            or record.get("size_bytes")
            != NONSCIENTIFIC_ARCHIVE_OVERLAY_SIZES[relative]
        ):
            raise ValueError(f"local compatibility overlay drift: {relative}")
    for relative, expected_hash in REQUIRED_UNTRACKED_SOURCE_MODULES.items():
        record = source_archive_manifest.get("files", {}).get(relative, {})
        if record.get("sha256") != expected_hash:
            raise ValueError(
                f"local required untracked source module drift: {relative}"
            )
    for relative, expected_hash in REQUIRED_HASH_LOCKED_FIDELITY_FILES.items():
        record = archive_fidelity.get(relative, {})
        archive_record = source_archive_manifest.get("files", {}).get(relative, {})
        if (
            record.get("sha256") != expected_hash
            or record.get("classification")
            != "reporting_only_fidelity_source_or_test"
            or not isinstance(record.get("tracked_in_parent_archive"), bool)
            or archive_record.get("sha256") != expected_hash
        ):
            raise ValueError(
                f"local required hash-locked fidelity source drift: {relative}"
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
            raise ValueError(f"local untracked fidelity overlay drift: {relative}")
    output = find_output(root)
    validation = load(output / "validation.json")
    result = load(output / "json/result.json")
    current = load(output / "json/current.json")
    ledger = load(output / "json/estimator_call_ledger.json")
    sidecar = load(output / "qiskit_cost_sidecar.json")
    repaired = load(output / "terminal_checkpoint.execution_order_repaired.json")
    fidelity = load(output / "ground_space_projector_fidelity.json")
    execution = load(output / "execution.json")
    normalized = load(output / "normalized_run_manifest.json")
    if validation.get("status") != "pass":
        raise ValueError("runtime validation did not pass")
    if execution.get("status") != "completed" or int(execution.get("exit_code", -1)) != 0:
        raise ValueError("execution record is not terminal-success")
    normalized_source = normalized.get("source_lock", {})
    if (
        normalized_source.get("git_commit") != source_commit
        or normalized_source.get("git_tree") != source_tree
        or normalized_source.get("source_archive_sha256")
        != source_archive_sha256
        or normalized_source.get("required_untracked_source_modules")
        != expected_untracked_modules
        or normalized_source.get("required_hash_locked_fidelity_files")
        != archive_fidelity
        or normalized_source.get("required_untracked_hash_overlays")
        != archive_fidelity_overlays
    ):
        raise ValueError("fetched normalized source lock drift")
    if (
        normalized.get("route_identity", {}).get("profile_resolved") != PROFILE
        or normalized.get("route_identity", {}).get("profile_contract_sha256")
        != digest
    ):
        raise ValueError("fetched normalized route identity drift")
    normalized_contract = normalized.get("route_identity", {}).get(
        "profile_contract", {}
    )
    if normalized_contract.get("execution_settings", {}).get(
        "adapt_finite_angle_fallback"
    ) is not False:
        raise ValueError("fetched finite-angle fallback contract drift")
    if normalized_contract.get("semantic_invariants", {}).get(
        "finite_angle_fallback_active"
    ) is not False:
        raise ValueError("fetched finite-angle semantic invariant drift")
    if normalized_contract.get("execution_settings", {}).get(
        "phase3_enable_rescue"
    ) is not False:
        raise ValueError("fetched Phase-III rescue contract drift")
    phase12 = normalized.get("route_identity", {}).get(
        "phase12_energy_model_contract", {}
    )
    for key, expected in {
        "phase1_energy_model": PHASE1_ENERGY_MODEL,
        "phase2_curvature_policy": PHASE2_CURVATURE_POLICY,
        "phase2_cheap_curvature_proxy_policy": (
            PHASE2_CHEAP_CURVATURE_PROXY_POLICY
        ),
    }.items():
        if normalized_contract.get("execution_settings", {}).get(key) != expected:
            raise ValueError(f"fetched profile {key} drift")
        if normalized_contract.get("semantic_invariants", {}).get(key) != expected:
            raise ValueError(f"fetched semantic invariant {key} drift")
        if phase12.get(key) != expected:
            raise ValueError(f"fetched job Phase-I/II contract {key} drift")
    if normalized_contract.get("semantic_invariants", {}).get(
        "phase1_phase2_lambda_f_proxy_active"
    ) is not False:
        raise ValueError("fetched lambda-F proxy invariant is active")
    artifact_paths = {
        "result_json": output / "json/result.json",
        "current_json": output / "json/current.json",
        "ledger_json": output / "json/estimator_call_ledger.json",
        "normalized_runtime_manifest_json": output / "normalized_run_manifest.json",
        "validation_json": output / "validation.json",
        "qiskit_cost_sidecar_json": output / "qiskit_cost_sidecar.json",
        "repaired_terminal_checkpoint_json": (
            output / "terminal_checkpoint.execution_order_repaired.json"
        ),
        "ground_space_fidelity_json": output / "ground_space_projector_fidelity.json",
    }
    execution_artifacts = execution.get("artifacts", {})
    for key, path in artifact_paths.items():
        record = execution_artifacts.get(key, {})
        if record.get("exists") is not True or record.get("sha256") != sha256(path):
            raise ValueError(f"execution artifact hash mismatch: {key}")
    for payload in (result.get("settings", {}), result.get("adapt_vqe", {})):
        if payload.get("sr_route_profile_resolved") != PROFILE:
            raise ValueError("profile drift")
        if payload.get("sr_route_profile_contract_sha256") != digest:
            raise ValueError("profile digest drift")
        if payload.get("phase1_energy_model") != PHASE1_ENERGY_MODEL:
            raise ValueError("fetched result Phase-I energy model drift")
        if payload.get("phase2_curvature_policy") != PHASE2_CURVATURE_POLICY:
            raise ValueError("fetched result Phase-II curvature policy drift")
        if payload.get("phase2_cheap_curvature_proxy_policy") != (
            PHASE2_CHEAP_CURVATURE_PROXY_POLICY
        ):
            raise ValueError("fetched result Phase-II cheap-proxy policy drift")
    segment = normalized.get("segment", {})
    target_round = int(segment.get("target_controller_round", -1))
    target_new_admissions = int(segment.get("max_new_admissions", -1))
    if target_round != 50 or target_new_admissions != 50:
        raise ValueError("prune appendix requires exact round-50 horizon")
    physics = normalized.get("physics", {})
    if (
        fidelity.get("schema") != FIDELITY_RECEIPT_SCHEMA
        or fidelity.get("status") != "pass"
        or fidelity.get("source_result_sha256")
        != sha256(output / "json/result.json")
        or fidelity.get("estimator_ledger_sha256_before")
        != sha256(output / "json/estimator_call_ledger.json")
        or fidelity.get("estimator_ledger_sha256_after")
        != fidelity.get("estimator_ledger_sha256_before")
        or int(fidelity.get("estimator_query_delta", -1)) != 0
    ):
        raise ValueError("post-run projector-fidelity provenance/ledger gate failed")
    ground = fidelity.get("ground_space_fidelity", {})
    if (
        ground.get("status") != "ok"
        or ground.get("usage_scope") != "post_run_reporting_only"
        or ground.get("s_alg_charged") is not False
        or ground.get("same_cutoff_verified") is not True
        or int(ground.get("working_cutoff", -1)) != int(physics.get("n_ph_work", -2))
        or int(ground.get("reference_cutoff", -1))
        != int(physics.get("n_ph_reference", -2))
    ):
        raise ValueError("post-run projector-fidelity scientific contract failed")
    evidence = validate_parent_evidence(
        result=result,
        current=current,
        ledger_sidecar=ledger,
        profile=PROFILE,
        digest=digest,
        target_round=target_round,
        target_new_admissions=target_new_admissions,
        require_supported_rank=True,
    )
    checkpoints = result.get("adapt_vqe", {}).get("active_prefix_checkpoints", [])
    terminal = [
        row for row in checkpoints
        if isinstance(row, dict)
        and int(row.get("outer_iteration", -1)) == target_round
        and row.get("checkpoint_kind") == "post_admission_prune"
    ]
    if len(terminal) != 1:
        raise ValueError(
            "exactly one target-round post-admission/prune checkpoint is required"
        )
    terminal_checkpoint = terminal[0]
    if sidecar.get("schema") != SIDECAR_SCHEMA or sidecar.get("status") != "ok":
        raise ValueError("Qiskit sidecar schema/status failed")
    replay = sidecar.get("fixed_prefix_replay", {})
    if replay.get("status") != "pass" or replay.get("prefix_reconstructed") is not True:
        raise ValueError("Qiskit sidecar/fixed-prefix replay failed")
    if float(replay.get("energy_abs_discrepancy", float("inf"))) > 1.0e-12:
        raise ValueError("Qiskit fixed-prefix replay energy mismatch")
    source = sidecar.get("source", {})
    if source.get("result_sha256") != sha256(output / "json/result.json"):
        raise ValueError("Qiskit sidecar result hash mismatch")
    if source.get("source_checkpoint_sha256") != terminal_checkpoint.get("checkpoint_sha256"):
        raise ValueError("Qiskit sidecar checkpoint hash mismatch")
    if (
        int(source.get("outer_iteration", -1)) != target_round
        or source.get("checkpoint_kind") != "post_admission_prune"
    ):
        raise ValueError("Qiskit sidecar selected the wrong checkpoint")
    if repaired.get("schema") != CHECKPOINT_REPAIR_SCHEMA:
        raise ValueError("repaired-checkpoint schema mismatch")
    repaired_source = repaired.get("source", {})
    repaired_summary = repaired.get("repair", {})
    repaired_checkpoint = repaired.get("repaired_checkpoint", {})
    result_digest = sha256(output / "json/result.json")
    if repaired_source.get("result_sha256") != result_digest:
        raise ValueError("repaired-checkpoint result hash mismatch")
    if repaired_source.get("checkpoint_sha256") != terminal_checkpoint.get("checkpoint_sha256"):
        raise ValueError("repaired-checkpoint source hash mismatch")
    repaired_digest = checkpoint_sha256(repaired_checkpoint)
    if repaired_checkpoint.get("checkpoint_sha256") != repaired_digest:
        raise ValueError("repaired checkpoint SHA-256 mismatch")
    if repaired_summary.get("repaired_checkpoint_sha256") != repaired_digest:
        raise ValueError("repair summary/repaired checkpoint hash mismatch")
    if repaired_summary.get("substantive_term_changes") is not False:
        raise ValueError("checkpoint repair was not permutation-only")
    repair = sidecar.get("source", {}).get("checkpoint_execution_order_repair", {})
    if repair.get("substantive_term_changes") not in {False, None}:
        raise ValueError("checkpoint execution-order repair changed substantive terms")
    for name, convention in (
        ("historical", sidecar.get("historical_displayed_convention", {})),
        ("current", sidecar.get("current_jr_fake_marrakesh_convention", {})),
    ):
        if convention.get("status") != "ok":
            raise ValueError(f"{name} Qiskit convention failed")
        for key in ("N2q", "D2q", "Dc"):
            if int(convention.get("metrics", {}).get(key, -1)) < 0:
                raise ValueError(f"{name} Qiskit metric missing/invalid: {key}")
    historical = sidecar.get("historical_displayed_convention", {})
    current_convention = sidecar.get("current_jr_fake_marrakesh_convention", {})
    if (
        historical.get("identity") != "table_i_basis_gate_transpile_v1"
        or historical.get("backend") is not None
        or int(historical.get("optimization_level", -1)) != 0
        or int(historical.get("seed_transpiler", -1)) != 7
    ):
        raise ValueError("historical displayed Qiskit convention drift")
    if (
        current_convention.get("identity")
        != "jr_signed_runtime_fake_marrakesh_transpile_v1"
        or current_convention.get("requested_backend") != "FakeMarrakesh"
        or int(current_convention.get("optimization_level", -1)) != 1
        or int(current_convention.get("seed_transpiler", -1)) != 7
    ):
        raise ValueError("current JR FakeMarrakesh convention drift")
    if sidecar.get("convention_comparison", {}).get("same_convention") is not False:
        raise ValueError("historical/current Qiskit conventions were conflated")
    prefix = sidecar.get("prefix", {})
    if prefix.get("prune_aware") is not True:
        raise ValueError("Qiskit sidecar is not prune aware")
    if int(prefix.get("active_ansatz_depth", -1)) != int(
        terminal_checkpoint.get("active_ansatz_depth", -2)
    ):
        raise ValueError("Qiskit sidecar active depth drift")
    if prefix.get("ordered_active_operator_labels") != terminal_checkpoint.get(
        "ordered_active_operator_labels"
    ):
        raise ValueError("Qiskit sidecar operator ordering drift")
    implementation_sources = sidecar.get("implementation_sources", {})
    archive_files = source_archive_manifest.get("files", {})
    for key, relative in QISKIT_IMPLEMENTATION_PATHS.items():
        expected_hash = archive_files.get(relative, {}).get("sha256")
        record = implementation_sources.get(key, {})
        if (
            not expected_hash
            or not str(record.get("path", "")).endswith(relative)
            or record.get("sha256") != expected_hash
        ):
            raise ValueError(f"Qiskit implementation source drift: {key}")
    runtime_evidence = validation.get("scientific_evidence_validation")
    if runtime_evidence != evidence:
        raise ValueError("runtime/fetched scientific-evidence validation mismatch")
    if validation.get("qiskit_implementation_sources") != sidecar.get(
        "implementation_sources"
    ):
        raise ValueError("runtime/fetched Qiskit implementation-source mismatch")
    if validation.get("result_sha256") != sha256(output / "json/result.json"):
        raise ValueError("runtime validation result hash mismatch")
    if validation.get("qiskit_sidecar_sha256") != sha256(
        output / "qiskit_cost_sidecar.json"
    ):
        raise ValueError("runtime validation Qiskit-sidecar hash mismatch")
    if validation.get("repaired_terminal_checkpoint_sha256") != sha256(
        output / "terminal_checkpoint.execution_order_repaired.json"
    ):
        raise ValueError("runtime validation repaired-checkpoint hash mismatch")
    if validation.get("post_run_projector_fidelity_sha256") != sha256(
        output / "ground_space_projector_fidelity.json"
    ):
        raise ValueError("runtime validation projector-fidelity hash mismatch")
    return {
        "schema": "paper_i_hh_sr_fs_prune_nodamping_fetched_validation_v1",
        "status": "pass",
        "output_root": str(output),
        "result_sha256": sha256(output / "json/result.json"),
        "current_sha256": sha256(output / "json/current.json"),
        "ledger_sha256": sha256(output / "json/estimator_call_ledger.json"),
        "ledger_schema": ledger.get("schema"),
        "current_schema": current.get("schema"),
        "target_controller_round": target_round,
        "profile_contract_sha256": digest,
        "qiskit_sidecar_sha256": sha256(output / "qiskit_cost_sidecar.json"),
        "repaired_checkpoint_sha256": sha256(output / "terminal_checkpoint.execution_order_repaired.json"),
        "post_run_projector_fidelity_sha256": sha256(
            output / "ground_space_projector_fidelity.json"
        ),
        "post_run_projector_fidelity": fidelity,
        "historical_metrics": sidecar.get("historical_displayed_convention", {}).get("metrics"),
        "current_fake_marrakesh_metrics": sidecar.get("current_jr_fake_marrakesh_convention", {}).get("metrics"),
        "scientific_evidence_validation": evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.path.is_file():
        with tempfile.TemporaryDirectory(prefix="sr_fsprune_nodamp_fetch_") as tmp:
            root = Path(tmp)
            safe_extract(args.path, root)
            report = validate(root)
    else:
        report = validate(args.path)
    if args.output_json:
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
