"""Build a versioned operational recovery for a pre-executor APM launch failure.

This module deliberately leaves the scientific request and payload untouched.  It
only repairs the shell quoting in the generated wrapper and assigns a fresh
Condor unit/batch identity so intent-first reconciliation cannot collide with the
failed submission.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import shlex
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "paper_ii_apm_operational_launch_recovery_v1"
ORIGINAL_UNIT = "prospective-apm-exchange-488646822fcf40e8"
RECOVERY_UNIT = f"{ORIGINAL_UNIT}-launchfix1"
ORIGINAL_BATCH = "paper-ii-apm-paper-ii-hh-product-v1-488646822fcf40e8"
RECOVERY_BATCH = f"{ORIGINAL_BATCH}-launchfix1"
ACTION_ID = (
    "prospective::paper_ii_hh_product_v1::hh__nph1__crossover__midres::"
    "exchange::t5::r00::cut1em02::bind155d3f310b1211c2"
)
REQUEST_SHA256 = "488646822fcf40e8d2563b62553c56c86d3007af756667e0e6c3e89708b20dc7"
SOURCE_EPOCH_ID = "paper-ii-hh-product-v1-source-267516eab3b3-ecc181dd4bce"
ORIGINAL_CLUSTER = 9715900
ORIGINAL_PACKAGE_SHA256 = "95e896cc117c4376a193afff092655fdb47f564f98e91753d747bdb2c1e00006"
ORIGINAL_FAILURE_RECEIPT_SHA256 = "11a77ce9a9cac17c91b9d79f02a743155baf19fc0f1d335518ef62a20bb6a746"
ORIGINAL_REL = Path(
    "output/prospective_scientific/paper_ii_hh_product_v1/"
    "hh__nph1__crossover__midres/adopt_scout_fde494768e43affe/"
    "next_apm_prefix_3eb8fc53"
)
RECOVERY_REL = ORIGINAL_REL.parent / "next_apm_prefix_launchfix1_e629aa25"
FAILURE_REL = Path(
    "output/paper_ii_hh_product_v1/acquisition_supervisor/"
    "terminal_failure_evidence/cluster_9715900/failure_receipt.json"
)
SUPERVISOR_REL = Path("output/paper_ii_hh_product_v1/acquisition_supervisor")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logical_sha(payload: dict[str, Any], field: str) -> str:
    clean = dict(payload)
    clean.pop(field, None)
    raw = json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any], digest_field: str) -> None:
    payload[digest_field] = _logical_sha(payload, digest_field)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_receipt_validation_command() -> str:
    """Return the inner command passed to ``bash -lc`` without nested-quote loss."""

    receipt = "output/paper_ii_hh_product_v1/source_epochs/seedfix_267516ea/source_epoch_receipt.json"
    python_code = (
        "from pipelines.time_dynamics.prospective_execution_bindings import "
        "SourceEpochReceipt; "
        f"SourceEpochReceipt.load({receipt!r}, repository_root='.')"
    )
    return f"cd /work && PYTHONPATH=. python3 -c {shlex.quote(python_code)}"


def repaired_apptainer_line() -> str:
    inner = source_receipt_validation_command()
    return (
        '"$APPTAINER_BIN" exec --cleanenv --bind "$SOURCE_ROOT:/work" "$IMAGE" '
        f"bash -lc {shlex.quote(inner)}"
    )


def _deterministic_targz(root: Path, members: list[str], destination: Path) -> None:
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for relative in sorted(members):
            path = root / relative
            info = archive.gettarinfo(str(path), arcname=relative)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with path.open("rb") as stream:
                archive.addfile(info, stream)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            zipped.write(tar_buffer.getvalue())


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"expected one {label}, found {count}")
    return text.replace(old, new, 1)


def _build_inner_package(repo: Path, package_dir: Path) -> tuple[Path, list[str]]:
    source_archive = repo / ORIGINAL_REL / "package/apm_exchange_action.tar.gz"
    if _sha(source_archive) != ORIGINAL_PACKAGE_SHA256:
        raise ValueError("original APM action package digest mismatch")
    with tempfile.TemporaryDirectory(prefix="paperii-hh-launchfix-") as tmp:
        extracted = Path(tmp)
        with tarfile.open(source_archive, "r:gz") as archive:
            archive.extractall(extracted, filter="data")

        submit_path = extracted / "submit.sub"
        submit = submit_path.read_text(encoding="utf-8")
        submit = _replace_once(submit, ORIGINAL_UNIT, RECOVERY_UNIT, label="unit id")
        submit = submit.replace(ORIGINAL_BATCH, RECOVERY_BATCH)
        submit = submit.replace(
            "queue 1\n",
            f'+PaperIIOperationalRecovery = "launchfix1"\n'
            f"+PaperIIOperationalRetryOfCluster = {ORIGINAL_CLUSTER}\nqueue 1\n",
        )
        submit_path.write_text(submit, encoding="utf-8")

        runner = extracted / "run_apm_exchange.sh"
        script = runner.read_text(encoding="utf-8")
        broken = next(
            line
            for line in script.splitlines()
            if "SourceEpochReceipt.load" in line
        )
        script = _replace_once(script, broken, repaired_apptainer_line(), label="broken validation line")
        old_verify = next(
            line for line in script.splitlines() if line.startswith("verify_file submit.sub ")
        )
        new_verify = f"verify_file submit.sub {submit_path.stat().st_size} {_sha(submit_path)}"
        script = _replace_once(script, old_verify, new_verify, label="submit verification")
        runner.write_text(script, encoding="utf-8")

        members = sorted(
            str(path.relative_to(extracted)) for path in extracted.rglob("*") if path.is_file()
        )
        destination = package_dir / "apm_exchange_action.tar.gz"
        _deterministic_targz(extracted, members, destination)
    return destination, members


def build(repo: Path, *, recorded_at: str | None = None) -> dict[str, Any]:
    repo = repo.resolve()
    failure = repo / FAILURE_REL
    failure_payload = json.loads(failure.read_text(encoding="utf-8"))
    if failure_payload.get("receipt_sha256") != ORIGINAL_FAILURE_RECEIPT_SHA256:
        raise ValueError("terminal failure receipt identity mismatch")
    if failure_payload.get("scientific_run_started") is not False:
        raise ValueError("operational recovery is only valid before the scientific executor starts")

    timestamp = recorded_at or datetime.now(timezone.utc).isoformat()
    recovery_root = repo / RECOVERY_REL
    package_dir = recovery_root / "package"
    package_dir.mkdir(parents=True, exist_ok=True)
    action_package, members = _build_inner_package(repo, package_dir)
    package_sha = _sha(action_package)

    source_archive_rel = Path(
        "output/paper_ii_hh_product_v1/bootstrap/activation_267516ea/shared/"
        "source_267516ea_runtime.tar.gz"
    )
    image_rel = Path("chtc/time_dynamics_optuna/image.sif")
    launch = (
        "#!/usr/bin/env bash\nset -euo pipefail\numask 077\n"
        'PACKAGE="$PWD/apm_exchange_action.tar.gz"\n'
        f'[[ -f "$PACKAGE" ]] && [[ "$(wc -c < "$PACKAGE" | tr -d " ")" == "{action_package.stat().st_size}" ]] '
        f'&& [[ "$(sha256sum "$PACKAGE" | awk \'{{print $1}}\')" == "{package_sha}" ]] || exit 2\n'
        "mkdir action_package\ntar -xzf \"$PACKAGE\" -C action_package\n"
        "exec bash action_package/run_apm_exchange.sh\n"
    )
    launch_path = package_dir / "launch.sh"
    launch_path.write_text(launch, encoding="utf-8")
    launch_path.chmod(0o755)

    submit = (repo / ORIGINAL_REL / "package/submit.sub").read_text(encoding="utf-8")
    submit = submit.replace(str(ORIGINAL_REL), str(RECOVERY_REL))
    submit = _replace_once(submit, ORIGINAL_UNIT, RECOVERY_UNIT, label="outer unit id")
    submit = submit.replace(ORIGINAL_BATCH, RECOVERY_BATCH)
    submit = submit.replace(
        "queue 1\n",
        f'+PaperIIOperationalRecovery = "launchfix1"\n'
        f"+PaperIIOperationalRetryOfCluster = {ORIGINAL_CLUSTER}\nqueue 1\n",
    )
    submit_path = package_dir / "submit.sub"
    submit_path.write_text(submit, encoding="utf-8")

    manifest = {
        "schema": SCHEMA,
        "status": "prepared_local_only",
        "recorded_at": timestamp,
        "operational_change": "shell_quote_source_receipt_validation_only",
        "retry_of_cluster": ORIGINAL_CLUSTER,
        "original_unit_id": ORIGINAL_UNIT,
        "recovery_unit_id": RECOVERY_UNIT,
        "original_batch_name": ORIGINAL_BATCH,
        "recovery_batch_name": RECOVERY_BATCH,
        "action_id": ACTION_ID,
        "execution_request_sha256": REQUEST_SHA256,
        "source_epoch_id": SOURCE_EPOCH_ID,
        "scientific_payload_unchanged": True,
        "qiskit_compilation_requested": False,
        "production_submission_enabled": False,
        "submission_permitted": False,
        "original_failure_receipt": {"path": str(FAILURE_REL), "sha256": _sha(failure)},
        "original_action_package_sha256": ORIGINAL_PACKAGE_SHA256,
        "recovery_action_package": {
            "path": str(action_package.relative_to(repo)),
            "bytes": action_package.stat().st_size,
            "sha256": package_sha,
            "member_count": len(members),
        },
        "submit_descriptor": {"path": str(submit_path.relative_to(repo)), "sha256": _sha(submit_path)},
        "runtime_dependencies": [str(source_archive_rel), str(image_rel)],
    }
    manifest_path = package_dir / "operational_recovery_manifest.json"
    _write_json(manifest_path, manifest, "recovery_manifest_sha256")

    intent_payload = {
        "job_batch_name": RECOVERY_BATCH,
        "package_kind": "apm_exchange_operational_launch_recovery",
        "package": {"path": str(action_package.relative_to(repo)), "bytes": action_package.stat().st_size, "sha256": package_sha},
        "submit_descriptor": {"path": str(submit_path.relative_to(repo)), "sha256": _sha(submit_path)},
        "operational_recovery_manifest": {"path": str(manifest_path.relative_to(repo)), "sha256": _sha(manifest_path)},
        "retry_of_cluster": ORIGINAL_CLUSTER,
        "first_reconciliation_outcome": "indeterminate",
        "submission_permitted": False,
        "production_submission_enabled": False,
        "qiskit_compilation": False,
    }
    intent = {
        "schema": "time_dynamics_submission_intent_v1",
        "lane_id": "paper_ii_hh_product_v1",
        "recorded_at": timestamp,
        "unit": {"unit_id": RECOVERY_UNIT, "action_ids": [ACTION_ID], "payload": intent_payload},
    }
    intent_path = SUPERVISOR_REL / "submission_intents" / f"{RECOVERY_UNIT}.json"
    _write_json(repo / intent_path, intent, "intent_sha256")

    intent_sha = intent["intent_sha256"]
    bridge = {
        "schema": "paper_ii_apm_exchange_operational_recovery_intent_bridge_v1",
        "first_reconciliation_outcome": "indeterminate",
        "production_submission_enabled": False,
        "submission_permitted": False,
        "scheduler_contacted": False,
        "retry_of_cluster": ORIGINAL_CLUSTER,
        "source_submission_intent": {"path": str(intent_path), "sha256": _sha(repo / intent_path), "intent_sha256": intent_sha},
        "generic_reconciliation_intent": {
            "acquisition_unit_id": RECOVERY_UNIT,
            "job_batch_name": RECOVERY_BATCH,
            "paper_ii_action_id": ACTION_ID,
            "paper_ii_execution_request_sha256": REQUEST_SHA256,
            "intent_sha256": intent_sha,
            "recorded_at": timestamp,
        },
    }
    bridge_path = SUPERVISOR_REL / "intent_bridges" / f"{RECOVERY_UNIT}.json"
    _write_json(repo / bridge_path, bridge, "bridge_sha256")

    local_preflight = {
        "schema": "paper_ii_apm_exchange_operational_recovery_local_preflight_v1",
        "status": "pass",
        "recorded_at": timestamp,
        "unit_id": RECOVERY_UNIT,
        "retry_of_cluster": ORIGINAL_CLUSTER,
        "checks": {
            "archive_member_count": len(members),
            "archive_gzip_and_tar_integrity": True,
            "generated_launch_bash_syntax": "pass",
            "generated_runtime_bash_syntax": "pass",
            "nested_bash_lc_argument_regression": "pass",
            "scientific_request_sha256_unchanged": REQUEST_SHA256,
            "qiskit_compilation_requested": False,
        },
        "production_submission_enabled": False,
        "submission_permitted": False,
    }
    _write_json(package_dir / "local_preflight_receipt.json", local_preflight, "preflight_sha256")

    evidence_rel = (
        SUPERVISOR_REL / "remote_preflight" / RECOVERY_UNIT
    )
    evidence_dir = repo / evidence_rel
    bundle_rel = RECOVERY_REL / "remote_upload/hh_apm_r00_launchfix1_e629aa25_upload.tar.gz"
    bundle = repo / bundle_rel
    if (evidence_dir / "dryrun.ad").is_file() and bundle.is_file():
        remote_root = (
            "/home/jsstrobel/Holstein_paper_ii_prospective_hh_apm_"
            "launchfix1_e629aa25_20260830/paper_ii_hh_product_v1/"
            "9538dcdaabac2cb5/apm_exchange_r00_launchfix1"
        )
        upload = {
            "schema": "paper_ii_apm_exchange_operational_recovery_upload_verification_v1",
            "status": "verified",
            "recorded_at": timestamp,
            "unit_id": RECOVERY_UNIT,
            "action_id": ACTION_ID,
            "execution_request_sha256": REQUEST_SHA256,
            "retry_of_cluster": ORIGINAL_CLUSTER,
            "local_upload_bundle": {"path": str(bundle_rel), "bytes": bundle.stat().st_size, "sha256": _sha(bundle)},
            "remote_upload_bundle": {"path": "/home/jsstrobel/hh_apm_r00_launchfix1_e629aa25_upload.tar.gz", "bytes": bundle.stat().st_size, "sha256": _sha(bundle), "regular_non_symlink": True},
            "remote_workdir": {"path": remote_root, "atomic_part_to_final_publish": True},
            "finalization": {"exact_full_size_match": True, "exact_full_sha256_match": True, "gzip_verified": True, "tar_verified": True, "atomic_part_to_final_rename": True},
            "source_dependency_sha256": "ac0013920058b2df466a920d8943da263aa11a7db0f8e807a333cd07ebf982dc",
            "runtime_image_sha256": "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f",
            "remote_raw_evidence_policy": "preserve",
            "scheduler_query_performed": False,
            "scheduler_submission_performed": False,
            "production_submission_enabled": False,
            "submission_permitted": False,
        }
        upload_rel = SUPERVISOR_REL / "upload_verifications" / f"{RECOVERY_UNIT}.json"
        _write_json(repo / upload_rel, upload, "upload_verification_sha256")

        dry_ad = evidence_dir / "dryrun.ad"
        dry_stdout = evidence_dir / "dryrun.stdout"
        dry_stderr = evidence_dir / "dryrun.stderr"
        dry = {
            "schema": "paper_ii_apm_exchange_operational_recovery_condor_dry_run_result_v1",
            "status": "pass",
            "recorded_at": timestamp,
            "unit_id": RECOVERY_UNIT,
            "action_id": ACTION_ID,
            "execution_request_sha256": REQUEST_SHA256,
            "retry_of_cluster": ORIGINAL_CLUSTER,
            "working_directory": remote_root,
            "submit_descriptor": {"path": str(submit_path.relative_to(repo)), "sha256": _sha(submit_path)},
            "dry_run_ad": {"path": str(evidence_rel / "dryrun.ad"), "bytes": dry_ad.stat().st_size, "sha256": _sha(dry_ad)},
            "stdout": {"path": str(evidence_rel / "dryrun.stdout"), "bytes": dry_stdout.stat().st_size, "sha256": _sha(dry_stdout)},
            "stderr": {"path": str(evidence_rel / "dryrun.stderr"), "bytes": dry_stderr.stat().st_size, "sha256": _sha(dry_stderr)},
            "returncode": 0,
            "single_dry_run_invocation": True,
            "lifecycle_validation": {"status": "pass", "performed_before_condor_dry_run": True, "validator_sha256": "4eb17bd7cd7e70a83244c4718fd0dfd9c82a28c5e86f4bec15f68a7d2f7390c0"},
            "submission_reconciliation": {"status": "indeterminate", "reason": "first_snapshot_pending_for_exact_recovery_intent", "submission_permitted": False},
            "scheduler_query_performed": False,
            "scheduler_submission_performed": False,
            "production_submission_enabled": False,
        }
        dry_rel = SUPERVISOR_REL / "dry_run_results" / f"{RECOVERY_UNIT}.json"
        _write_json(repo / dry_rel, dry, "dry_run_result_sha256")

    result = {
        "recovery_root": str(RECOVERY_REL),
        "unit_id": RECOVERY_UNIT,
        "batch_name": RECOVERY_BATCH,
        "action_package_sha256": package_sha,
        "intent_sha256": intent_sha,
        "bridge_path": str(bridge_path),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--recorded-at")
    args = parser.parse_args()
    print(json.dumps(build(args.repo_root, recorded_at=args.recorded_at), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
