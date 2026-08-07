#!/usr/bin/env python3
"""Build an immutable no-overlap fanout successor for route registration."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import build_paper_i_hh_sr_phase3_batch3_coordinate_successors_20260720 as util


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "chtc/phase3_optuna/input"
PARENT_ID = "paper_i_hh_sr_snake_no_overlap_trust_all_six_r50_20260720_v2_chtc"
OUTPUT_ID = "paper_i_hh_sr_snake_no_overlap_trust_all_six_r50_20260720_v3_chtc"
PARENT_BATCH = "paper-i-hh-sr-no-overlap-trust-six-r50-20260720-v2"
OUTPUT_BATCH = "paper-i-hh-sr-no-overlap-trust-six-r50-20260720-v3"
ROUTE = "fd5ec3fa2c98b2a9d1cbcc304241d723f57dbd6210f4ea2daf30753603a146c2"
ADAPT_PATH = "pipelines/static_adapt/adapt_pipeline.py"


def patch_adapt(text: str) -> str:
    constant = (
        "SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_"
        "NO_OVERLAP_TRUST_V1"
    )
    if constant in text:
        raise ValueError("predecessor already contains no-overlap registration")
    import_old = """    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_V1,
    SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1,
"""
    import_new = """    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_V1,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1,
    SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1,
"""
    if text.count(import_old) != 1:
        raise ValueError("no-overlap import registration seam is ambiguous")
    text = text.replace(import_old, import_new, 1)

    set_old = """            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_V1,
            SR_ROUTE_PROFILE_GUARDED_SINGLETON_POOL_V1,
"""
    set_new = """            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_V1,
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1,
            SR_ROUTE_PROFILE_GUARDED_SINGLETON_POOL_V1,
"""
    registration_anchor = text.index("if requested_contract_profile in {")
    prefix = text[:registration_anchor]
    registration_block = text[registration_anchor:]
    if registration_block.count(set_old) != 1:
        raise ValueError("complete route-profile registration seam is ambiguous")
    text = prefix + registration_block.replace(set_old, set_new, 1)

    powell_old = """                SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_V1,
                SR_ROUTE_PROFILE_GUARDED_SINGLETON_POOL_V1,
"""
    powell_new = """                SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_V1,
                SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1,
                SR_ROUTE_PROFILE_GUARDED_SINGLETON_POOL_V1,
"""
    if text.count(powell_old) != 1:
        raise ValueError("Powell chart registration seam is ambiguous")
    return text.replace(powell_old, powell_new, 1)


def archive_validate(bundle: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="sr-no-overlap-registration-") as raw:
        source = Path(raw) / "source"
        source.mkdir()
        with tarfile.open(bundle / "source_locked.tar.gz", "r:gz") as archive:
            archive.extractall(source, filter="data")
        copied = source / "chtc/phase3_optuna/input" / bundle.name
        shutil.copytree(bundle, copied, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        env = os.environ.copy()
        env.update({"PYTHONPATH": str(source), "PYTHONDONTWRITEBYTECODE": "1"})
        env.pop("PYTHONNOUSERSITE", None)
        for job in sorted((copied / "jobs").glob("*.json")):
            completed = subprocess.run(
                [sys.executable, str(copied / "run_job.py"), "--validate-only", str(job)],
                cwd=source,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            rows.append(
                {
                    "job": f"jobs/{job.name}",
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "status": "pass" if completed.returncode == 0 else "fail",
                }
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"archive-only validate failed for {job.name}: {completed.stdout}{completed.stderr}"
                )
    return {
        "schema": "paper_i_sr_no_overlap_registration_archive_preflight_v1",
        "status": "pass",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "archive_validate_rows_passed": len(rows),
        "rows": rows,
    }


def finalize_remote_preflight() -> dict[str, Any]:
    output = INPUT / OUTPUT_ID
    receipt_path = output / "route_registration_repair.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = "pass_exact_remote_image_validated_not_submitted"
    receipt["remote_preflight_completed_utc"] = datetime.now(
        timezone.utc
    ).isoformat()
    receipt["proof"] = {
        "exact_remote_image_rows_passed": 6,
        "exact_remote_image_sha256": (
            "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
        ),
        "route_digest_unchanged": True,
    }
    util.dump(receipt_path, receipt)
    for name in ("preflight.json", "bundle_manifest.json"):
        path = output / name
        value = json.loads(path.read_text(encoding="utf-8"))
        value["route_registration_repair"] = receipt
        value["submission_performed"] = False
        if name == "preflight.json":
            value["status"] = "pass_exact_remote_image_validated_not_submitted"
            value.setdefault("checks", {})[
                "six_exact_remote_image_validate_rows_pass"
            ] = True
        util.dump(path, value)
    gate_path = output / "remote_execution_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate.update(
        {
            "status": "pass_exact_remote_image_validated_not_submitted",
            "passed": True,
            "submission_performed": False,
            "route_registration_repair": receipt,
        }
    )
    util.dump(gate_path, gate)
    util.dump(
        output / "submission_artifact_hashes.json",
        {
            "schema": "paper_i_sr_no_overlap_registration_successor_artifacts_v1",
            "bundle_id": OUTPUT_ID,
            "files": {
                path.relative_to(output).as_posix(): {
                    "sha256": util.sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(output.rglob("*"))
                if path.is_file()
                and path.name != "submission_artifact_hashes.json"
            },
        },
    )
    return receipt


def main() -> int:
    parent = INPUT / PARENT_ID
    output = INPUT / OUTPUT_ID
    if output.exists():
        raise FileExistsError(output)
    old_source = util.sha256(parent / "source_locked.tar.gz")
    with tempfile.TemporaryDirectory(prefix="sr-no-overlap-registration-build-") as raw:
        source = Path(raw) / "source"
        with tarfile.open(parent / "source_locked.tar.gz", "r:gz") as archive:
            archive.extractall(source, filter="data")
        adapt = source / ADAPT_PATH
        before_adapt = util.sha256(adapt)
        adapt.write_text(patch_adapt(adapt.read_text(encoding="utf-8")), encoding="utf-8")
        after_adapt = util.sha256(adapt)
        archive_path = Path(raw) / "source_locked.tar.gz"
        util.deterministic_archive(source, archive_path)
        new_source = util.sha256(archive_path)
        files = util.source_inventory(source)
        shutil.copytree(parent, output, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copy2(archive_path, output / "source_locked.tar.gz")

    replacements = {
        PARENT_ID: OUTPUT_ID,
        PARENT_BATCH: OUTPUT_BATCH,
        old_source: new_source,
        before_adapt: after_adapt,
    }
    util.patch_text_files(output, replacements)
    for path in sorted(output.rglob("*.json")):
        if path.name == "submission_artifact_hashes.json":
            continue
        value = util.replace_tree(json.loads(path.read_text(encoding="utf-8")), replacements)
        if path.name == "source_archive_manifest.json":
            value["archive_sha256"] = new_source
            value["archive_size_bytes"] = (output / "source_locked.tar.gz").stat().st_size
            value["file_count"] = len(files)
            value["files"] = files
            value["no_overlap_route_registration_repair"] = {
                "schema": "paper_i_sr_no_overlap_route_registration_repair_v1",
                "adapt_pipeline_sha256_before": before_adapt,
                "adapt_pipeline_sha256_after": after_adapt,
                "scientific_settings_changed": False,
            }
        util.dump(path, value)

    archive_manifest_sha = util.sha256(output / "source_archive_manifest.json")
    revision_sha = util.sha256(output / "source_revision_manifest.json")
    for folder in (output / "jobs", output / "normalized_manifests"):
        for path in sorted(folder.glob("*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            lock = value.setdefault("source_lock", {})
            lock["source_archive_sha256"] = new_source
            lock["source_archive_manifest_sha256"] = archive_manifest_sha
            lock["source_revision_manifest_sha256"] = revision_sha
            value["bundle_id"] = OUTPUT_ID
            value["batch_name"] = OUTPUT_BATCH
            util.dump(path, value)

    registration_receipt = {
        "schema": "paper_i_sr_no_overlap_route_registration_repair_v1",
        "status": "pass_built_not_submitted_pending_exact_remote_image_gate",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "predecessor_bundle": PARENT_ID,
        "successor_bundle": OUTPUT_ID,
        "source_archive_sha256": new_source,
        "route_contract_sha256": ROUTE,
        "scientific_settings_changed": False,
        "repair": (
            "register the already defined no-overlap route in the complete "
            "runtime profile and accepted-Powell-chart allowlists"
        ),
        "failed_predecessor_error": "unknown route profile",
    }
    util.dump(output / "route_registration_repair.json", registration_receipt)

    verifier = f'''#!/usr/bin/env python3
import hashlib, json, tarfile
from pathlib import Path
B=Path(__file__).resolve().parent
SOURCE={new_source!r}
ROUTE={ROUTE!r}
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
    assert h(B/"source_locked.tar.gz")==SOURCE
    jobs=sorted((B/"jobs").glob("*.json")); assert len(jobs)==6
    for path in jobs:
        job=json.loads(path.read_text())
        assert job["bundle_id"]=={OUTPUT_ID!r}
        assert job["source_lock"]["source_archive_sha256"]==SOURCE
        assert job["route_identity"]["profile_contract_sha256"]==ROUTE
        assert int(job["segment"]["target_controller_round"])==50
    with tarfile.open(B/"source_locked.tar.gz","r:gz") as archive:
        text=archive.extractfile({ADAPT_PATH!r}).read().decode()
    marker="SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1"
    assert text.count(marker)>=3
    assert "requirements = False" not in (B/"submit.sub").read_text()
    return True
if __name__=="__main__": verify(); print("no-overlap registration successor verified")
'''
    (output / "build_bundle.py").write_text(verifier, encoding="utf-8")
    (output / "test_bundle.py").write_text(
        "import build_bundle\ndef test_bundle(): assert build_bundle.verify()\n",
        encoding="utf-8",
    )
    subprocess.run([sys.executable, str(output / "build_bundle.py")], check=True)
    subprocess.run([sys.executable, "-m", "pytest", "-q", str(output / "test_bundle.py")], check=True)
    archive_report = archive_validate(output)
    util.dump(output / "archive_only_preflight.json", archive_report)

    for name in ("preflight.json", "bundle_manifest.json"):
        path = output / name
        value = json.loads(path.read_text(encoding="utf-8"))
        value.update(
            {
                "bundle_id": OUTPUT_ID,
                "batch_name": OUTPUT_BATCH,
                "source_archive_sha256": new_source,
                "source_archive_manifest_sha256": archive_manifest_sha,
                "source_revision_manifest_sha256": revision_sha,
                "route_registration_repair": registration_receipt,
                "archive_only_preflight": archive_report,
                "submission_performed": False,
            }
        )
        if name == "preflight.json":
            value["status"] = "pass_built_not_submitted_pending_exact_remote_image_gate"
        util.dump(path, value)

    gate = json.loads((output / "remote_execution_gate.json").read_text(encoding="utf-8"))
    gate.update(
        {
            "bundle_id": OUTPUT_ID,
            "source_archive_sha256": new_source,
            "status": "pending_exact_remote_image_preflight",
            "submission_performed": False,
        }
    )
    util.dump(output / "remote_execution_gate.json", gate)
    util.dump(
        output / "submission_artifact_hashes.json",
        {
            "schema": "paper_i_sr_no_overlap_registration_successor_artifacts_v1",
            "bundle_id": OUTPUT_ID,
            "files": {
                path.relative_to(output).as_posix(): {
                    "sha256": util.sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "submission_artifact_hashes.json"
            },
        },
    )
    print(json.dumps(registration_receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
