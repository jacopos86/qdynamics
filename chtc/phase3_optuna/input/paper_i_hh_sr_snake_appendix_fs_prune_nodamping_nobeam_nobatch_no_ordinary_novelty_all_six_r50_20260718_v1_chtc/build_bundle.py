#!/usr/bin/env python3
"""Verify the frozen SR-SNAKE undamped FS-pruning appendix bundle.

This verifier performs no scientific calculation, does not rebuild from the
live repository, and cannot submit CHTC jobs.  The executable authority is the
immutable source archive already present beside this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any


BUNDLE_DIR = Path(__file__).resolve().parent
BUNDLE_ID = BUNDLE_DIR.name
BATCH_NAME = (
    "paper-i-hh-sr-appendix-fsprune-nodamp-nobeam-nobatch-"
    "nonovelty-six-r50-20260718-v1"
)
PROFILE_REQUEST = "sr_snake_symmetric_cost_fs_prune_nodamping_v1"
PROFILE_RESOLVED = (
    "supported_whitened_adaptive_trust_full_response_symmetric_cost_"
    "fs_prune_nodamping_v1"
)
PROFILE_CONTRACT_SHA256 = (
    "272ede635558edb4acc2507ac3a9803d8ccec062b96c98634b8d6407df9fbc21"
)
SOURCE_ARCHIVE_SHA256 = (
    "1d6e93bd59f97f74cc444c6c3559b15d48053b2c4914736a3c32b0e0869a196a"
)
PARENT_BUNDLE_ID = (
    "paper_i_hh_sr_snake_main_full_response_symmetric_cost_noprune_nobeam_"
    "no_ordinary_novelty_all_six_r50_20260718_v1_chtc"
)
PARENT_ROUTE_CONTRACT_SHA256 = (
    "69dcf6d5711a05b54cc143562bc6f437511d781164319b2705857b22deda1538"
)
PARENT_SOURCE_ARCHIVE_SHA256 = (
    "fa9014b9608ccb8a301df0429268482abf6dd10c91eb336111a15d00be256d35"
)
EXPECTED_IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)
REMOTE_EXECUTION_GATE_SCHEMA = (
    "paper_i_hh_sr_symcost_noprune_remote_execution_gate_v1"
)
REMOTE_IMAGE_PATH = "chtc/phase3_optuna/image.sif"
REMOTE_QISKIT_VERSION = "2.3.1"
REMOTE_FAKE_BACKEND_RESOLVED = "fake_marrakesh"
REMOTE_FAKE_BACKEND_QUBITS = 156
PARENT_SNAPSHOT_SHA256 = {
    "main_bundle_manifest.construction_snapshot.json": (
        "2e7eebb5f955aebfaf51ea5b7ea4cd3f871c5b7e10bd7bd87762ad42a32ef97d"
    ),
    "main_scientific_settings_audit.construction_snapshot.json": (
        "8d337f6b6763242649181a84867c7b33567fd0a2584a18d249a7b7a91a6fc25f"
    ),
    "main_source_archive_manifest.construction_snapshot.json": (
        "e979fd9533f8ae98396c43fd4b092a6150b6ea8b69c7ebe2c1c09e8df65f35e4"
    ),
    "main_source_revision_manifest.construction_snapshot.json": (
        "ffa51edd94fb3cac9749488f56dbe9e0004e5f25ddf033ff7e3cac4937a3dfbb"
    ),
}
SOURCE_FREEZE_COMPLETE = True
# Deliberate source-visible switch.  A valid remote gate is necessary but can
# never enable submission by itself.
SUBMISSION_ENABLED = True
SUBMISSION_REGIMES = frozenset(
    {
        "weak_weak",
        "intermediate_weak",
        "strong_weak_u8",
        "weak_strong",
        "intermediate_strong",
        "strong_strong_u8",
    }
)
PRUNE_ONLY_CHANGED_FIELDS = frozenset(
    {
        "phase1_prune_enabled",
        "phase1_prune_mode",
        "phase1_prune_max_candidates",
        "phase1_prune_local_window_size",
        "phase1_prune_recovery_trust_radius",
        "phase1_prune_schur_nomination_route",
        "phase1_prune_metric_schur_mu",
        "phase1_prune_metric_schur_solve_mode",
        "phase1_prune_metric_schur_cost_weighting",
        "phase1_prune_trust_update_policy",
        "phase1_prune_metric_mu_update_policy",
        "phase1_prune_endpoint_overlap_policy",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _artifact_record(path: Path) -> dict[str, Any]:
    return {"sha256": sha256(path), "size_bytes": path.stat().st_size}


def _remote_execution_gate_status(bundle: Path) -> dict[str, Any]:
    gate_path = bundle / "remote_execution_gate.json"
    base = {
        "gate_path": gate_path.name,
        "gate_sha256": None,
        "schema_expected": REMOTE_EXECUTION_GATE_SCHEMA,
        "passed": False,
    }
    if not gate_path.is_file():
        return {**base, "reason": "missing"}
    base["gate_sha256"] = sha256(gate_path)
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {**base, "reason": f"unreadable:{type(exc).__name__}"}
    remote = gate.get("remote_execution_preflight", {})
    checks = {
        "schema": gate.get("schema") == REMOTE_EXECUTION_GATE_SCHEMA,
        "status": gate.get("status") == "pass",
        "image_path": remote.get("image_path") == REMOTE_IMAGE_PATH,
        "image_sha256": remote.get("image_sha256") == EXPECTED_IMAGE_SHA256,
        "qiskit_import": remote.get("qiskit_import_passed") is True,
        "qiskit_version": remote.get("qiskit_version") == REMOTE_QISKIT_VERSION,
        "fake_backend_instantiation": (
            remote.get("fake_backend_instantiation_passed") is True
        ),
        "fake_backend_identity": (
            remote.get("fake_backend_resolved") == REMOTE_FAKE_BACKEND_RESOLVED
        ),
        "fake_backend_qubits": (
            remote.get("fake_backend_qubits") == REMOTE_FAKE_BACKEND_QUBITS
        ),
    }
    passed = all(checks.values())
    return {
        **base,
        "checks": checks,
        "passed": passed,
        "reason": "pass" if passed else "failed_checks",
    }


def _submission_requirements(
    *, submission_enabled: bool, remote_gate_passed: bool
) -> str:
    if submission_enabled and not remote_gate_passed:
        raise RuntimeError(
            "submission cannot be enabled before remote_execution_gate.json passes"
        )
    return "TARGET.HasSIF" if submission_enabled else "False"


def _write_submit_requirements(
    bundle: Path,
    *,
    submission_enabled: bool,
    remote_gate_passed: bool,
) -> str:
    """Materialize only the operational Condor requirement line."""

    requirements = _submission_requirements(
        submission_enabled=submission_enabled,
        remote_gate_passed=remote_gate_passed,
    )
    submit_path = bundle / "submit.sub"
    lines = submit_path.read_text(encoding="utf-8").splitlines()
    indices = [
        index
        for index, line in enumerate(lines)
        if line.strip().startswith("requirements =")
    ]
    if len(indices) != 1:
        raise ValueError("submit.sub must contain exactly one requirements line")
    lines[indices[0]] = f"requirements = {requirements}"
    submit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return requirements


def _refresh_operational_records(
    remote_gate: dict[str, Any], requirements: str
) -> None:
    gate = load(BUNDLE_DIR / "remote_execution_gate.json")
    ready = bool(SUBMISSION_ENABLED and remote_gate["passed"])
    receipt = {
        "schema": "paper_i_hh_sr_prune_appendix_remote_preflight_receipt_v1",
        "status": "pass" if remote_gate["passed"] else "blocked",
        "submission_authorized": ready,
        "remote_execution_gate": remote_gate,
        "confirmation": gate.get("confirmation"),
        "storage_cleanup": {
            "scope": "no_cleanup_authorized_or_performed",
            "remote_removed_paths": [],
            "unrelated_remote_paths_modified": False,
        },
        "submission_performed": False,
    }
    receipt_path = BUNDLE_DIR / "remote_preflight_and_cleanup_receipt.json"
    _write_json(receipt_path, receipt)

    preflight_path = BUNDLE_DIR / "preflight.json"
    preflight = load(preflight_path)
    preflight["checks"]["remote_image_gate_complete"] = bool(
        remote_gate["passed"]
    )
    preflight["status"] = (
        "pass_submission_ready_not_yet_submitted"
        if ready
        else "pass_prepared_submission_blocked"
    )
    preflight["submission_authorized"] = ready
    preflight["submission_blockers"] = [] if ready else [
        "remote_image_gate_or_explicit_switch_incomplete"
    ]
    preflight["submission_status"] = (
        "submission_ready_not_yet_submitted" if ready else "blocked_not_authorized"
    )
    preflight["submission_performed"] = False
    _write_json(preflight_path, preflight)

    manifest_path = BUNDLE_DIR / "bundle_manifest.json"
    manifest = load(manifest_path)
    manifest["submission"] = {
        "authorized": ready,
        "blockers": [] if ready else [
            "remote_execution_gate_or_explicit_switch_incomplete"
        ],
        "condor_requirements_false": requirements == "False",
        "enabled_in_verifier": SUBMISSION_ENABLED,
        "performed": False,
        "requirements": requirements,
        "remote_execution_gate": remote_gate,
        "remote_preflight": receipt,
    }
    manifest["verification"]["preflight"] = {
        "path": preflight_path.name,
        **_artifact_record(preflight_path),
    }
    inventory = dict(manifest["artifact_inventory"])
    for relative in inventory:
        inventory[relative] = _artifact_record(BUNDLE_DIR / relative)
    manifest["artifact_inventory"] = inventory
    _write_json(manifest_path, manifest)

    hashes_path = BUNDLE_DIR / "submission_artifact_hashes.json"
    hashes = load(hashes_path)
    artifacts = dict(hashes["artifacts"])
    for relative in artifacts:
        artifacts[relative] = _artifact_record(BUNDLE_DIR / relative)
    hashes["artifacts"] = artifacts
    _write_json(hashes_path, hashes)


def isolated_contract_probe() -> dict[str, Any]:
    archive = BUNDLE_DIR / "source_locked.tar.gz"
    with tempfile.TemporaryDirectory(prefix="sr_prune_appendix_probe_") as tmp:
        root = Path(tmp)
        with tarfile.open(archive, "r:gz") as handle:
            for member in handle.getmembers():
                if member.issym() or member.islnk() or ".." in Path(member.name).parts:
                    raise ValueError(f"unsafe source member: {member.name}")
            handle.extractall(root, filter="data")
        code = (
            "import json;"
            "from pipelines.static_adapt.sr_snake_route_profile import "
            "canonical_sr_snake_symmetric_cost_fs_prune_v1_contract as c;"
            "print(json.dumps(c(),sort_keys=True))"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(root)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=root,
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict):
            raise TypeError("isolated profile probe did not return an object")
        return payload


def verify_bundle() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    remote_gate = _remote_execution_gate_status(BUNDLE_DIR)
    requirements = _submission_requirements(
        submission_enabled=SUBMISSION_ENABLED,
        remote_gate_passed=bool(remote_gate["passed"]),
    )
    archive = BUNDLE_DIR / "source_locked.tar.gz"
    checks["source_archive_hash"] = sha256(archive) == SOURCE_ARCHIVE_SHA256
    archive_manifest = load(BUNDLE_DIR / "source_archive_manifest.json")
    checks["source_inventory_hash"] = (
        archive_manifest.get("archive_sha256") == SOURCE_ARCHIVE_SHA256
        and int(archive_manifest.get("file_count", -1)) == 387
        and archive_manifest.get("derived_overlay", {}).get("removed_files") == []
    )
    snapshot_root = BUNDLE_DIR / "parent_provenance_snapshot"
    checks["parent_construction_snapshots"] = all(
        (snapshot_root / name).is_file()
        and sha256(snapshot_root / name) == expected
        for name, expected in PARENT_SNAPSHOT_SHA256.items()
    )

    contract = isolated_contract_probe()
    checks["profile_identity"] = (
        contract.get("route_profile") == PROFILE_RESOLVED
        and json_sha256(contract) == PROFILE_CONTRACT_SHA256
    )
    execution = dict(contract["execution_settings"])
    semantics = dict(contract["semantic_invariants"])
    checks["prune_contract"] = (
        execution.get("phase1_prune_enabled") is True
        and execution.get("phase1_prune_mode") == "live"
        and execution.get("phase1_prune_local_window_size") == 0
        and execution.get("phase1_prune_recovery_trust_radius") == 0.125
        and execution.get("phase1_prune_metric_schur_mu") == 0.0
        and execution.get("phase1_prune_metric_mu_update_policy") == "off"
        and execution.get("phase3_shadow_damping_policy") == "off"
        and semantics.get("prune_acceptance_authority")
        == "measured_delete_and_refit_v1"
    )
    checks["non_prune_contract"] = (
        execution.get("adapt_beam_live_branches") == 1
        and execution.get("adapt_beam_children_per_parent") == 1
        and execution.get("phase2_enable_batching") is False
        and execution.get("phase3_enable_batching") is False
        and execution.get("phase2_gram_novelty_policy") == "fallback_only_v1"
        and execution.get("phase3_gram_novelty_policy") == "fallback_only_v1"
        and execution.get("adapt_full_refit_every") == 0
        and execution.get("adapt_final_full_refit") == "false"
    )

    jobs = sorted((BUNDLE_DIR / "jobs").glob("*.json"))
    normalized = sorted((BUNDLE_DIR / "normalized_manifests").glob("*.json"))
    checks["six_rows"] = (
        {path.stem for path in jobs} == set(SUBMISSION_REGIMES)
        and {path.stem for path in normalized} == set(SUBMISSION_REGIMES)
    )
    row_checks: list[dict[str, Any]] = []
    for path in jobs + normalized:
        payload = load(path)
        route = payload["route_identity"]
        segment = payload["segment"]
        source = payload["source_lock"]
        paths = payload["paths"] if "paths" in payload else {}
        argv = payload["command"]["argv"] if "command" in payload else payload["command_argv"]
        row_checks.append(
            {
                "path": str(path.relative_to(BUNDLE_DIR)),
                "pass": (
                    route.get("profile_request") == PROFILE_REQUEST
                    and route.get("profile_resolved") == PROFILE_RESOLVED
                    and route.get("profile_contract_sha256")
                    == PROFILE_CONTRACT_SHA256
                    and route.get("profile_contract") == contract
                    and int(segment.get("source_controller_round", -1)) == 0
                    and int(segment.get("target_controller_round", -1)) == 50
                    and int(segment.get("target_depth", -1)) == 50
                    and source.get("source_archive_sha256")
                    == SOURCE_ARCHIVE_SHA256
                    and source.get("source_archive", "").endswith(
                        f"{BUNDLE_ID}/source_locked.tar.gz"
                    )
                    and all(
                        f"raw_outputs/{BUNDLE_ID}/" in str(value)
                        for value in paths.values()
                    )
                    and all(
                        BUNDLE_ID in str(token)
                        for token in argv
                        if "raw_outputs/" in str(token)
                    )
                    and PROFILE_REQUEST in argv
                    and payload.get("sensitivity_audit", {}).get(
                        "submission_authorized"
                    )
                    is False
                ),
            }
        )
    checks["manifest_rows"] = all(row["pass"] for row in row_checks)
    submit = (BUNDLE_DIR / "submit.sub").read_text(encoding="utf-8")
    checks["submission_gate_state"] = (
        f"requirements = {requirements}" in submit
        and (
            (requirements == "False" and not SUBMISSION_ENABLED)
            or (
                requirements == "TARGET.HasSIF"
                and SUBMISSION_ENABLED
                and remote_gate["passed"]
            )
        )
    )
    failures = sorted(key for key, value in checks.items() if not value)
    return {
        "schema": "paper_i_hh_sr_prune_appendix_bundle_verification_v1",
        "status": "pass" if not failures else "fail",
        "checks": checks,
        "failed_checks": failures,
        "row_checks": row_checks,
        "submission_enabled": SUBMISSION_ENABLED,
        "submission_requirements": requirements,
        "remote_image_gate": remote_gate,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.parse_args()
    remote_gate = _remote_execution_gate_status(BUNDLE_DIR)
    _write_submit_requirements(
        BUNDLE_DIR,
        submission_enabled=SUBMISSION_ENABLED,
        remote_gate_passed=bool(remote_gate["passed"]),
    )
    requirements = _submission_requirements(
        submission_enabled=SUBMISSION_ENABLED,
        remote_gate_passed=bool(remote_gate["passed"]),
    )
    _refresh_operational_records(remote_gate, requirements)
    payload = verify_bundle()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
