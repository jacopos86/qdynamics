#!/usr/bin/env python3
"""Non-destructively revalidate one completed v6 failure with the v9 validator."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

import run_job


BUNDLE = Path(__file__).resolve().parent
V9_BUNDLE_ID = run_job.BUNDLE_ID
V6_BUNDLE_ID = V9_BUNDLE_ID.replace("_v9_chtc", "_v6_chtc")
KNOWN_FAILURE = (
    "ValueError: normalized candidate setting drift: "
    "phase_live_hysteresis_enabled"
)
GENERATED_NAMES = (
    "validation.json",
    "qiskit_cost_sidecar.json",
    "terminal_checkpoint.execution_order_repaired.json",
    "ground_space_projector_fidelity.json",
)


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
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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
                or any(
                    part in {".DS_Store", "__MACOSX"}
                    or part.startswith("._")
                    for part in name.parts
                )
            ):
                raise ValueError(f"unsafe archive member: {member.name}")
        handle.extractall(destination, filter="data")


def find_output(root: Path) -> Path:
    matches = list(root.rglob("json/result.json"))
    if len(matches) != 1:
        raise ValueError(f"expected one result.json, found {len(matches)}")
    output = matches[0].parents[1]
    if output.parent.name != V6_BUNDLE_ID:
        raise ValueError("transfer archive belongs to the wrong v6 family")
    return output


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def rebase_manifest(output: Path, generated: Path) -> dict[str, Any]:
    regime = output.name
    manifest_path = BUNDLE / "jobs" / f"{regime}.json"
    if not manifest_path.is_file():
        raise ValueError(f"v9 job manifest missing for regime {regime}")
    manifest = copy.deepcopy(load(manifest_path))
    paths = manifest["paths"]
    paths.update(
        {
            "output_root": str(output),
            "result_json": str(output / "json/result.json"),
            "current_json": str(output / "json/current.json"),
            "ledger_json": str(output / "json/estimator_call_ledger.json"),
            "normalized_runtime_manifest_json": str(
                output / "normalized_run_manifest.json"
            ),
            "execution_json": str(output / "execution.json"),
            "validation_json": str(generated / "validation.json"),
            "qiskit_cost_sidecar_json": str(
                generated / "qiskit_cost_sidecar.json"
            ),
            "repaired_terminal_checkpoint_json": str(
                generated / "terminal_checkpoint.execution_order_repaired.json"
            ),
            "ground_space_fidelity_json": str(
                generated / "ground_space_projector_fidelity.json"
            ),
        }
    )
    source_lock = manifest["source_lock"]
    source_lock["source_archive"] = str(BUNDLE / "source_locked.tar.gz")
    source_lock["source_archive_manifest"] = str(
        BUNDLE / "source_archive_manifest.json"
    )
    source_lock["source_revision_manifest"] = str(
        BUNDLE / "source_revision_manifest.json"
    )
    source_lock["physics_reference_lock"] = str(
        BUNDLE / "physics_and_exact_reference_lock.json"
    )
    return manifest


def validate_original_failure(
    *, output: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    execution_path = output / "execution.json"
    normalized_path = output / "normalized_run_manifest.json"
    result_path = output / "json/result.json"
    current_path = output / "json/current.json"
    ledger_path = output / "json/estimator_call_ledger.json"
    for path in (
        execution_path,
        normalized_path,
        result_path,
        current_path,
        ledger_path,
    ):
        if not path.is_file():
            raise ValueError(f"completed scientific payload missing: {path.name}")
    execution = load(execution_path)
    normalized = load(normalized_path)
    if (
        execution.get("status") != "failed"
        or int(execution.get("exit_code", -1)) != 70
        or execution.get("validation_error") != KNOWN_FAILURE
    ):
        raise ValueError("archive did not stop at the exact known validator defect")
    if (
        normalized.get("route_identity") != manifest.get("route_identity")
        or normalized.get("physics") != manifest.get("physics")
        or normalized.get("segment") != manifest.get("segment")
    ):
        raise ValueError("frozen v6 normalized manifest differs from v9 science lock")
    artifacts = execution.get("artifacts", {})
    required = {
        "result_json": result_path,
        "current_json": current_path,
        "ledger_json": ledger_path,
        "normalized_runtime_manifest_json": normalized_path,
    }
    for key, path in required.items():
        record = artifacts.get(key, {})
        if (
            record.get("exists") is not True
            or record.get("sha256") != sha256(path)
            or int(record.get("size_bytes", -1)) != path.stat().st_size
        ):
            raise ValueError(f"frozen execution artifact mismatch: {key}")
    return {
        "execution_sha256": sha256(execution_path),
        "normalized_manifest_sha256": sha256(normalized_path),
        "result_sha256": sha256(result_path),
        "current_sha256": sha256(current_path),
        "ledger_sha256": sha256(ledger_path),
        "known_failure": KNOWN_FAILURE,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    archive = args.archive.resolve()
    output_dir = args.output_dir.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output directory must be absent or empty")
    archive_sha_before = sha256(archive)
    source_archive = BUNDLE / "source_locked.tar.gz"
    expected_source_sha = str(
        load(BUNDLE / "source_archive_manifest.json").get("archive_sha256") or ""
    )
    if not source_archive.is_file() or sha256(source_archive) != expected_source_sha:
        raise ValueError("v9 source archive lock is missing or changed")

    with tempfile.TemporaryDirectory(prefix="sr_macro_cost_v9_revalidate_") as tmp:
        workspace = Path(tmp)
        transfer_root = workspace / "transfer"
        source_root = workspace / "source"
        generated = workspace / "generated"
        safe_extract(archive, transfer_root)
        safe_extract(source_archive, source_root)
        output = find_output(transfer_root)
        manifest = rebase_manifest(output, generated)
        original = validate_original_failure(output=output, manifest=manifest)
        with working_directory(source_root):
            validation = run_job.validate_result_and_compile(manifest)
        dump(generated / "validation.json", validation)
        generated_hashes = {
            name: {
                "sha256": sha256(generated / name),
                "size_bytes": (generated / name).stat().st_size,
            }
            for name in GENERATED_NAMES
        }
        archive_sha_after = sha256(archive)
        if archive_sha_after != archive_sha_before:
            raise ValueError("raw transfer archive changed during revalidation")
        receipt = {
            "schema": "paper_i_sr_macro_beam_cost_v9_v6_archive_revalidation_v1",
            "status": "pass",
            "raw_transfer_archive": str(archive),
            "raw_transfer_archive_sha256_before": archive_sha_before,
            "raw_transfer_archive_sha256_after": archive_sha_after,
            "raw_transfer_archive_preserved": True,
            "source_archive_sha256": expected_source_sha,
            "v6_bundle_id": V6_BUNDLE_ID,
            "v9_validator_bundle_id": V9_BUNDLE_ID,
            "regime_slug": output.name,
            "profile_contract_sha256": run_job.DIGEST,
            "original_failure_receipt": original,
            "generated_reporting_artifacts": generated_hashes,
            "scientific_evidence_validation": validation.get(
                "scientific_evidence_validation"
            ),
            "source_only_runtime_settings_receipt": validation.get(
                "source_only_runtime_settings_receipt"
            ),
            "scientific_rerun_required": False,
        }
        dump(generated / "revalidation_receipt.json", receipt)
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in generated.iterdir():
            shutil.copy2(path, output_dir / path.name)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
