#!/usr/bin/env python3
"""Fail-closed validator for one fetched v4 transfer archive or output root."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from evidence_validation import checkpoint_sha256, validate_parent_evidence


PROFILE = "supported_whitened_adaptive_trust_full_response_symmetric_cost_fs_prune_v4"
DIGEST = "447b8fe3f4fef340fbb1cd5d221a0234826ba80c7e4e405937004e4ab25bec93"
SOURCE_COMMIT = "dfe8d8cad94167ebb1be6f919eeab3a64bb904d2"
SOURCE_TREE = "e49f80b371ed0236875b7fa317ce475adb8d5b50"
SOURCE_ARCHIVE_SHA256 = "954682affa511d3c73c127ec5c512475ecba953adc1c650a97ba90433139891a"
SIDECAR_SCHEMA = "paper_i_hh_recovery_prune_aware_qiskit_sidecar_v1"
CHECKPOINT_REPAIR_SCHEMA = "paper_i_checkpoint_execution_order_repair_v1"
QISKIT_IMPLEMENTATION_SHA256 = {
    "postprocessor": (
        "pipelines/exact_bench/paper_i_hh_recovery_prefix_qiskit_sidecar.py",
        "5486c285deffcb47fd0f5ef0314a9e3ab2fd1c83ebb7e0bb72d629d6a81dd044",
    ),
    "historical_table_i_compiler": (
        "pipelines/exact_bench/table_i_qiskit_resource_compile.py",
        "cdc182772288593de6087049470a8b6bb47a00c254cd6176276eda63320d19cd",
    ),
    "ansatz_circuit_builder": (
        "pipelines/hardcoded/adapt_circuit_execution.py",
        "1b569d31a45f98522b615fba0bb5645a6fba8af63ecc338f1059f14623364a0e",
    ),
    "backend_transpile_tools": (
        "pipelines/qiskit_backend_tools.py",
        "46fcfcce70479b5cad5346b456b689531d4f28fbc1200fe5ef22b5c68494c05b",
    ),
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
    output = find_output(root)
    validation = load(output / "validation.json")
    result = load(output / "json/result.json")
    current = load(output / "json/current.json")
    ledger = load(output / "json/estimator_call_ledger.json")
    sidecar = load(output / "qiskit_cost_sidecar.json")
    repaired = load(output / "terminal_checkpoint.execution_order_repaired.json")
    execution = load(output / "execution.json")
    normalized = load(output / "normalized_run_manifest.json")
    if validation.get("status") != "pass":
        raise ValueError("runtime validation did not pass")
    if execution.get("status") != "completed" or int(execution.get("exit_code", -1)) != 0:
        raise ValueError("execution record is not terminal-success")
    normalized_source = normalized.get("source_lock", {})
    if (
        normalized_source.get("git_commit") != SOURCE_COMMIT
        or normalized_source.get("git_tree") != SOURCE_TREE
        or normalized_source.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256
    ):
        raise ValueError("fetched normalized source lock drift")
    if (
        normalized.get("route_identity", {}).get("profile_resolved") != PROFILE
        or normalized.get("route_identity", {}).get("profile_contract_sha256")
        != DIGEST
    ):
        raise ValueError("fetched normalized route identity drift")
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
    }
    execution_artifacts = execution.get("artifacts", {})
    for key, path in artifact_paths.items():
        record = execution_artifacts.get(key, {})
        if record.get("exists") is not True or record.get("sha256") != sha256(path):
            raise ValueError(f"execution artifact hash mismatch: {key}")
    for payload in (result.get("settings", {}), result.get("adapt_vqe", {})):
        if payload.get("sr_route_profile_resolved") != PROFILE:
            raise ValueError("profile drift")
        if payload.get("sr_route_profile_contract_sha256") != DIGEST:
            raise ValueError("profile digest drift")
    evidence = validate_parent_evidence(
        result=result,
        current=current,
        ledger_sidecar=ledger,
        profile=PROFILE,
        digest=DIGEST,
        target_round=30,
        target_new_admissions=30,
        require_supported_rank=True,
    )
    checkpoints = result.get("adapt_vqe", {}).get("active_prefix_checkpoints", [])
    terminal = [
        row for row in checkpoints
        if isinstance(row, dict)
        and int(row.get("outer_iteration", -1)) == 30
        and row.get("checkpoint_kind") == "post_admission_prune"
    ]
    if len(terminal) != 1:
        raise ValueError("exactly one round-30 post-admission/prune checkpoint is required")
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
    if int(source.get("outer_iteration", -1)) != 30 or source.get("checkpoint_kind") != "post_admission_prune":
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
    for key, (relative, expected_hash) in QISKIT_IMPLEMENTATION_SHA256.items():
        record = implementation_sources.get(key, {})
        if (
            not str(record.get("path", "")).endswith(relative)
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
    return {
        "schema": "paper_i_hh_sr_snake_v4_fetched_validation_v1",
        "status": "pass",
        "output_root": str(output),
        "result_sha256": sha256(output / "json/result.json"),
        "current_sha256": sha256(output / "json/current.json"),
        "ledger_sha256": sha256(output / "json/estimator_call_ledger.json"),
        "ledger_schema": ledger.get("schema"),
        "current_schema": current.get("schema"),
        "qiskit_sidecar_sha256": sha256(output / "qiskit_cost_sidecar.json"),
        "repaired_checkpoint_sha256": sha256(output / "terminal_checkpoint.execution_order_repaired.json"),
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
        with tempfile.TemporaryDirectory(prefix="sr_v4_fetch_") as tmp:
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
