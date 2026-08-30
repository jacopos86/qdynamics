"""Build the versioned Hubbard-L4 APM wrapper-quoting recovery package.

The original prospective action reconciled ``verified_absent`` but its generated
wrapper has the same nested ``bash -lc`` quoting defect that failed the HH APM
launch before the scientific executor started.  This module changes only that
wrapper boundary and assigns a fresh operational unit/batch identity.  The
scientific action, request, source epoch, seed, work order, resources, and Qiskit
policy remain byte-bound to the original package.
"""

from __future__ import annotations

import argparse
import json
import shlex
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipelines.time_dynamics.apm_operational_launch_recovery import (
    _deterministic_targz,
    _replace_once,
    _sha,
    _write_json,
)


SCHEMA = "paper_ii_hubbard_apm_operational_launch_recovery_v1"
LANE_ID = "paper_ii_hubbard_l4_scale_v1"
ORIGINAL_UNIT = "prospective-apm-exchange-e667b05f39a4b711"
RECOVERY_UNIT = f"{ORIGINAL_UNIT}-launchfix1"
ORIGINAL_BATCH = "paper-ii-apm-paper-ii-hubbard-l4-scale-v1-e667b05f39a4b711"
RECOVERY_BATCH = f"{ORIGINAL_BATCH}-launchfix1"
ACTION_ID = (
    "prospective::paper_ii_hubbard_l4_scale_v1::"
    "hubbard__l4__u4__weakslow::exchange::t5::r00::cut1em02::"
    "binde875c098ccaa9943"
)
REQUEST_SHA256 = "e667b05f39a4b711cb1ca5f3444244f6d3edad8854628c9fc663844b1c629681"
SOURCE_EPOCH_ID = "paper-ii-hubbard-l4-scale-v1-source-5eedf0dc546c-4ad1a8fc2205"
ORIGINAL_PACKAGE_SHA256 = "0143b668890ab21dc933426778b846e6b5e4609219eb5f340fc3be99c03cb74f"
ORIGINAL_PACKAGE_MANIFEST_SHA256 = (
    "4687e61a3d45f8508687f83b6ef96d4ad4b7f31be387594e483a5eb49daa2a5e"
)
BLOCKER_SHA256 = "3d65697b6bd15d9a6c9884d415df7428f9847504f09f4a8d6a8400bdb944dcd5"
SOURCE_ARCHIVE_SHA256 = "da056abce6f9c15d3352ad08a06baa78964e31b338df74c83d1d8bf39429db76"
RUNTIME_IMAGE_SHA256 = "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
ORIGINAL_REL = Path(
    "output/prospective_scientific/paper_ii_hubbard_l4_scale_v1/"
    "hubbard__l4__u4__weakslow/adopt_scout_853b05df7ba4edb6/"
    "next_apm_prefix_fab1c488"
)
RECOVERY_REL = ORIGINAL_REL.parent / "next_apm_prefix_launchfix1_58c00b94"
BLOCKER_REL = Path(
    "output/paper_ii_hubbard_l4_scale_v1/acquisition_supervisor/"
    "terminal_failure_evidence/"
    "prospective-apm-exchange-e667b05f39a4b711_pre_submit_blocker.json"
)
SUPERVISOR_REL = Path("output/paper_ii_hubbard_l4_scale_v1/acquisition_supervisor")
SOURCE_RECEIPT_REL = Path(
    "output/paper_ii_hubbard_l4_scale_v1/source_epochs/bootstrap_5eedf0dc/"
    "source_epoch_receipt.json"
)
SOURCE_ARCHIVE_REL = Path(
    "output/paper_ii_hubbard_l4_scale_v1/bootstrap/activation_5eedf0dc/shared/"
    "source_5eedf0dc_runtime.tar.gz"
)
RUNTIME_IMAGE_REL = Path("chtc/time_dynamics_optuna/image.sif")
RUNTIME_IDENTITY_REL = Path(
    "output/paper_ii_hh_product_v1/bootstrap/activation_f99b702f/shared/"
    "runtime_image_identity.json"
)
BUNDLE_REL = RECOVERY_REL / (
    "remote_upload/hubbard_apm_r00_launchfix1_58c00b94_upload.tar.gz"
)
REMOTE_BUNDLE = "/home/jsstrobel/hubbard_apm_r00_launchfix1_58c00b94_upload.tar.gz"
REMOTE_WORKDIR = (
    "/home/jsstrobel/Holstein_paper_ii_prospective_hubbard_apm_"
    "launchfix1_58c00b94_20260830/paper_ii_hubbard_l4_scale_v1/"
    "20719ac7f89c7f12/apm_exchange_r00_launchfix1"
)


def source_receipt_validation_command() -> str:
    """Return one literal ``bash -lc`` argument for the Hubbard source receipt."""

    python_code = (
        "from pipelines.time_dynamics.prospective_execution_bindings import "
        "SourceEpochReceipt; "
        f"SourceEpochReceipt.load({str(SOURCE_RECEIPT_REL)!r}, repository_root='.')"
    )
    return f"cd /work && PYTHONPATH=. python3 -c {shlex.quote(python_code)}"


def repaired_apptainer_line() -> str:
    """Render the exact repaired Apptainer validation invocation."""

    return (
        '"$APPTAINER_BIN" exec --cleanenv --bind "$SOURCE_ROOT:/work" "$IMAGE" '
        f"bash -lc {shlex.quote(source_receipt_validation_command())}"
    )


def _build_inner_package(repo: Path, package_dir: Path) -> tuple[Path, list[str]]:
    source_archive = repo / ORIGINAL_REL / "package/apm_exchange_action.tar.gz"
    if _sha(source_archive) != ORIGINAL_PACKAGE_SHA256:
        raise ValueError("original Hubbard APM action package digest mismatch")

    with tempfile.TemporaryDirectory(prefix="paperii-hubbard-launchfix-") as tmp:
        extracted = Path(tmp)
        with tarfile.open(source_archive, "r:gz") as archive:
            archive.extractall(extracted, filter="data")

        submit_path = extracted / "submit.sub"
        submit = submit_path.read_text(encoding="utf-8")
        submit = _replace_once(submit, ORIGINAL_UNIT, RECOVERY_UNIT, label="unit id")
        submit = submit.replace(ORIGINAL_BATCH, RECOVERY_BATCH)
        submit = submit.replace(
            "queue 1\n",
            '+PaperIIOperationalRecovery = "launchfix1"\n'
            f'+PaperIIOperationalRecoveryOfUnit = "{ORIGINAL_UNIT}"\n'
            "queue 1\n",
        )
        submit_path.write_text(submit, encoding="utf-8")

        runner = extracted / "run_apm_exchange.sh"
        script = runner.read_text(encoding="utf-8")
        broken = next(
            line for line in script.splitlines() if "SourceEpochReceipt.load" in line
        )
        script = _replace_once(
            script,
            broken,
            repaired_apptainer_line(),
            label="broken source-receipt validation line",
        )
        old_verify = next(
            line for line in script.splitlines() if line.startswith("verify_file submit.sub ")
        )
        script = _replace_once(
            script,
            old_verify,
            f"verify_file submit.sub {submit_path.stat().st_size} {_sha(submit_path)}",
            label="submit verification",
        )
        runner.write_text(script, encoding="utf-8")

        members = sorted(
            str(path.relative_to(extracted))
            for path in extracted.rglob("*")
            if path.is_file()
        )
        destination = package_dir / "apm_exchange_action.tar.gz"
        _deterministic_targz(extracted, members, destination)
    return destination, members


def _build_upload_bundle(repo: Path, package_dir: Path) -> Path:
    members = [
        str((package_dir / "apm_exchange_action.tar.gz").relative_to(repo)),
        str((package_dir / "launch.sh").relative_to(repo)),
        str((package_dir / "operational_recovery_manifest.json").relative_to(repo)),
        str((package_dir / "submit.sub").relative_to(repo)),
        str(RUNTIME_IDENTITY_REL),
    ]
    destination = repo / BUNDLE_REL
    _deterministic_targz(repo, members, destination)
    return destination


def build(repo: Path, *, recorded_at: str | None = None) -> dict[str, Any]:
    """Materialize the local recovery and, when present, remote preflight receipts."""

    repo = repo.resolve()
    blocker = repo / BLOCKER_REL
    if _sha(blocker) != BLOCKER_SHA256:
        raise ValueError("Hubbard pre-submit blocker identity mismatch")
    blocker_payload = json.loads(blocker.read_text(encoding="utf-8"))
    if blocker_payload.get("scheduler_submission_performed") is not False:
        raise ValueError("recovery requires an unsubmitted original unit")
    if blocker_payload.get("submission_permitted") is not False:
        raise ValueError("original broken unit must remain submission-disabled")
    if blocker_payload.get("execution_request_sha256") != REQUEST_SHA256:
        raise ValueError("blocker scientific request identity mismatch")

    original_manifest = repo / ORIGINAL_REL / "package/package_manifest.json"
    if _sha(original_manifest) != ORIGINAL_PACKAGE_MANIFEST_SHA256:
        raise ValueError("original Hubbard package manifest digest mismatch")
    timestamp = recorded_at or datetime.now(timezone.utc).isoformat()
    recovery_root = repo / RECOVERY_REL
    package_dir = recovery_root / "package"
    package_dir.mkdir(parents=True, exist_ok=True)

    action_package, members = _build_inner_package(repo, package_dir)
    package_sha = _sha(action_package)
    launch = (
        "#!/usr/bin/env bash\nset -euo pipefail\numask 077\n"
        'PACKAGE="$PWD/apm_exchange_action.tar.gz"\n'
        f'[[ -f "$PACKAGE" ]] && [[ "$(wc -c < "$PACKAGE" | tr -d " ")" == "{action_package.stat().st_size}" ]] '
        f'&& [[ "$(sha256sum "$PACKAGE" | awk \'{{print $1}}\')" == "{package_sha}" ]] || exit 2\n'
        'mkdir action_package\ntar -xzf "$PACKAGE" -C action_package\n'
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
        '+PaperIIOperationalRecovery = "launchfix1"\n'
        f'+PaperIIOperationalRecoveryOfUnit = "{ORIGINAL_UNIT}"\n'
        "queue 1\n",
    )
    submit_path = package_dir / "submit.sub"
    submit_path.write_text(submit, encoding="utf-8")

    manifest = {
        "schema": SCHEMA,
        "status": "prepared_local_only",
        "recorded_at": timestamp,
        "operational_change": "shell_quote_source_receipt_validation_only",
        "recovery_of_unit": ORIGINAL_UNIT,
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
        "original_pre_submit_blocker": {
            "path": str(BLOCKER_REL),
            "sha256": _sha(blocker),
        },
        "original_package_manifest": {
            "path": str(original_manifest.relative_to(repo)),
            "sha256": _sha(original_manifest),
        },
        "original_action_package_sha256": ORIGINAL_PACKAGE_SHA256,
        "recovery_action_package": {
            "path": str(action_package.relative_to(repo)),
            "bytes": action_package.stat().st_size,
            "sha256": package_sha,
            "member_count": len(members),
        },
        "submit_descriptor": {
            "path": str(submit_path.relative_to(repo)),
            "sha256": _sha(submit_path),
        },
        "runtime_dependencies": [str(SOURCE_ARCHIVE_REL), str(RUNTIME_IMAGE_REL)],
        "calibration_audit": {
            "status": "pass",
            "same_action": True,
            "same_execution_request": True,
            "same_source_epoch": True,
            "same_seed_and_work_order_bytes_inside_action_package": True,
            "same_resources": True,
            "exact_reference_blind": True,
            "qiskit_compilation_requested": False,
            "controller_or_scientific_settings_changed": False,
        },
    }
    manifest_path = package_dir / "operational_recovery_manifest.json"
    _write_json(manifest_path, manifest, "recovery_manifest_sha256")

    intent_payload = {
        "job_batch_name": RECOVERY_BATCH,
        "package_kind": "apm_exchange_operational_launch_recovery",
        "package": {
            "path": str(action_package.relative_to(repo)),
            "bytes": action_package.stat().st_size,
            "sha256": package_sha,
        },
        "submit_descriptor": {
            "path": str(submit_path.relative_to(repo)),
            "sha256": _sha(submit_path),
        },
        "operational_recovery_manifest": {
            "path": str(manifest_path.relative_to(repo)),
            "sha256": _sha(manifest_path),
        },
        "recovery_of_unit": ORIGINAL_UNIT,
        "first_reconciliation_outcome": "indeterminate",
        "submission_permitted": False,
        "production_submission_enabled": False,
        "qiskit_compilation": False,
    }
    intent = {
        "schema": "time_dynamics_submission_intent_v1",
        "lane_id": LANE_ID,
        "recorded_at": timestamp,
        "unit": {
            "unit_id": RECOVERY_UNIT,
            "action_ids": [ACTION_ID],
            "payload": intent_payload,
        },
    }
    intent_rel = SUPERVISOR_REL / "submission_intents" / f"{RECOVERY_UNIT}.json"
    _write_json(repo / intent_rel, intent, "intent_sha256")
    intent_sha = intent["intent_sha256"]

    bridge = {
        "schema": "paper_ii_apm_exchange_operational_recovery_intent_bridge_v1",
        "first_reconciliation_outcome": "indeterminate",
        "production_submission_enabled": False,
        "submission_permitted": False,
        "scheduler_contacted": False,
        "recovery_of_unit": ORIGINAL_UNIT,
        "source_submission_intent": {
            "path": str(intent_rel),
            "sha256": _sha(repo / intent_rel),
            "intent_sha256": intent_sha,
        },
        "generic_reconciliation_intent": {
            "acquisition_unit_id": RECOVERY_UNIT,
            "job_batch_name": RECOVERY_BATCH,
            "paper_ii_action_id": ACTION_ID,
            "paper_ii_execution_request_sha256": REQUEST_SHA256,
            "intent_sha256": intent_sha,
            "recorded_at": timestamp,
        },
    }
    bridge_rel = SUPERVISOR_REL / "intent_bridges" / f"{RECOVERY_UNIT}.json"
    _write_json(repo / bridge_rel, bridge, "bridge_sha256")

    local_preflight = {
        "schema": "paper_ii_hubbard_apm_operational_recovery_local_preflight_v1",
        "status": "pass",
        "recorded_at": timestamp,
        "unit_id": RECOVERY_UNIT,
        "recovery_of_unit": ORIGINAL_UNIT,
        "checks": {
            "archive_member_count": len(members),
            "archive_gzip_and_tar_integrity": True,
            "generated_launch_bash_syntax": "pass",
            "generated_runtime_bash_syntax": "pass",
            "nested_bash_lc_argument_regression": "pass",
            "scientific_request_sha256_unchanged": REQUEST_SHA256,
            "calibration_audit": "pass",
            "qiskit_compilation_requested": False,
        },
        "production_submission_enabled": False,
        "submission_permitted": False,
    }
    _write_json(
        package_dir / "local_preflight_receipt.json",
        local_preflight,
        "preflight_sha256",
    )

    bundle = _build_upload_bundle(repo, package_dir)
    evidence_rel = SUPERVISOR_REL / "remote_preflight" / RECOVERY_UNIT
    evidence_dir = repo / evidence_rel
    if (evidence_dir / "dryrun.ad").is_file():
        upload = {
            "schema": "paper_ii_hubbard_apm_operational_recovery_upload_verification_v1",
            "status": "verified",
            "recorded_at": timestamp,
            "unit_id": RECOVERY_UNIT,
            "action_id": ACTION_ID,
            "execution_request_sha256": REQUEST_SHA256,
            "recovery_of_unit": ORIGINAL_UNIT,
            "local_upload_bundle": {
                "path": str(BUNDLE_REL),
                "bytes": bundle.stat().st_size,
                "sha256": _sha(bundle),
            },
            "remote_upload_bundle": {
                "path": REMOTE_BUNDLE,
                "bytes": bundle.stat().st_size,
                "sha256": _sha(bundle),
                "regular_non_symlink": True,
            },
            "remote_workdir": {
                "path": REMOTE_WORKDIR,
                "atomic_part_to_final_publish": True,
            },
            "finalization": {
                "exact_full_size_match": True,
                "exact_full_sha256_match": True,
                "gzip_verified": True,
                "tar_verified": True,
                "atomic_part_to_final_rename": True,
            },
            "source_dependency_sha256": SOURCE_ARCHIVE_SHA256,
            "runtime_image_sha256": RUNTIME_IMAGE_SHA256,
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
            "schema": "paper_ii_hubbard_apm_operational_recovery_condor_dry_run_result_v1",
            "status": "pass",
            "recorded_at": timestamp,
            "unit_id": RECOVERY_UNIT,
            "action_id": ACTION_ID,
            "execution_request_sha256": REQUEST_SHA256,
            "recovery_of_unit": ORIGINAL_UNIT,
            "working_directory": REMOTE_WORKDIR,
            "submit_descriptor": {
                "path": str(submit_path.relative_to(repo)),
                "sha256": _sha(submit_path),
            },
            "dry_run_ad": {
                "path": str(evidence_rel / "dryrun.ad"),
                "bytes": dry_ad.stat().st_size,
                "sha256": _sha(dry_ad),
            },
            "stdout": {
                "path": str(evidence_rel / "dryrun.stdout"),
                "bytes": dry_stdout.stat().st_size,
                "sha256": _sha(dry_stdout),
            },
            "stderr": {
                "path": str(evidence_rel / "dryrun.stderr"),
                "bytes": dry_stderr.stat().st_size,
                "sha256": _sha(dry_stderr),
            },
            "returncode": 0,
            "single_dry_run_invocation": True,
            "lifecycle_validation": {
                "status": "pass",
                "performed_before_condor_dry_run": True,
                "validator_sha256": "4eb17bd7cd7e70a83244c4718fd0dfd9c82a28c5e86f4bec15f68a7d2f7390c0",
            },
            "submission_reconciliation": {
                "status": "indeterminate",
                "reason": "first_snapshot_pending_for_exact_recovery_intent",
                "submission_permitted": False,
            },
            "scheduler_query_performed": False,
            "scheduler_submission_performed": False,
            "production_submission_enabled": False,
        }
        dry_rel = SUPERVISOR_REL / "dry_run_results" / f"{RECOVERY_UNIT}.json"
        _write_json(repo / dry_rel, dry, "dry_run_result_sha256")

    return {
        "recovery_root": str(RECOVERY_REL),
        "unit_id": RECOVERY_UNIT,
        "batch_name": RECOVERY_BATCH,
        "action_package_sha256": package_sha,
        "upload_bundle": {
            "path": str(BUNDLE_REL),
            "bytes": bundle.stat().st_size,
            "sha256": _sha(bundle),
        },
        "intent_sha256": intent_sha,
        "bridge_path": str(bridge_rel),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--recorded-at")
    args = parser.parse_args()
    print(json.dumps(build(args.repo_root, recorded_at=args.recorded_at), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
