#!/usr/bin/env python3
"""Prepare and reconcile local-only prospective exact/AVQDS activation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from pipelines.time_dynamics.prospective_avqds_scout_chtc_package import (
    AVQDSCHTCResources,
)
from pipelines.time_dynamics.prospective_bootstrap_activation import (
    authorize_retry_after_rejected_submit,
    authorize_from_snapshot,
    load_submission_intent,
    prepare_bootstrap_activation,
    record_dry_run_result,
    record_provisional_submit_result,
    record_upload_verification,
    recover_claim_after_absence,
    recover_submission_receipt,
    validate_artifact,
)
from pipelines.time_dynamics.prospective_seed_chtc_package import (
    SeedCHTCResources,
)


def _add_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repository-root", "--repo-root", default=".")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    _add_root(prepare)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--seed-receipt", action="append", required=True)
    prepare.add_argument("--source-epoch-receipt", action="append", required=True)
    prepare.add_argument(
        "--exact-source-archive",
        action="append",
        required=True,
        metavar="LANE=PATH",
    )
    prepare.add_argument(
        "--avqds-source-archive",
        action="append",
        required=True,
        metavar="LANE=PATH",
    )
    prepare.add_argument("--runtime-image-identity", required=True)
    prepare.add_argument(
        "--remote-root",
        default="Holstein_phase3_optuna_chtc/paper_ii_bootstrap_v1",
    )
    prepare.add_argument("--recorded-at")
    prepare.add_argument("--free-bytes", type=int)
    prepare.add_argument("--exact-cpus", type=int, default=1)
    prepare.add_argument("--exact-memory-gib", type=int, default=16)
    prepare.add_argument("--exact-disk-gib", type=int, default=8)
    prepare.add_argument("--exact-runtime-seconds", type=int, default=43_200)
    prepare.add_argument("--scout-cpus", type=int, default=1)
    prepare.add_argument("--scout-memory-gib", type=int, default=16)
    prepare.add_argument("--scout-disk-gib", type=int, default=8)
    prepare.add_argument("--scout-runtime-seconds", type=int, default=43_200)

    upload = commands.add_parser("record-upload")
    _add_root(upload)
    upload.add_argument("--intent", required=True)
    upload.add_argument("--remote-bytes", type=int, required=True)
    upload.add_argument("--remote-sha256", required=True)
    upload.add_argument("--recorded-at")

    dry = commands.add_parser("record-dry-run")
    _add_root(dry)
    dry.add_argument("--intent", required=True)
    dry.add_argument("--returncode", type=int, required=True)
    dry.add_argument("--dry-run-ad", required=True)
    dry.add_argument("--remote-submit-descriptor", required=True)
    dry.add_argument("--stdout", default="")
    dry.add_argument("--stderr", default="")
    dry.add_argument("--recorded-at")

    authorize = commands.add_parser("authorize-from-snapshot")
    _add_root(authorize)
    authorize.add_argument("--intent", required=True)
    authorize.add_argument("--snapshot", required=True)
    authorize.add_argument("--queue-stdout", required=True)
    authorize.add_argument("--history-stdout", required=True)
    authorize.add_argument("--prior-snapshot")
    authorize.add_argument("--prior-queue-stdout")
    authorize.add_argument("--prior-history-stdout")

    recover_claim = commands.add_parser("recover-claim-after-absence")
    _add_root(recover_claim)
    recover_claim.add_argument("--intent", required=True)
    recover_claim.add_argument("--predecessor-claim", required=True)
    recover_claim.add_argument("--first-snapshot", required=True)
    recover_claim.add_argument("--first-queue-stdout", required=True)
    recover_claim.add_argument("--first-history-stdout", required=True)
    recover_claim.add_argument("--later-snapshot", required=True)
    recover_claim.add_argument("--later-queue-stdout", required=True)
    recover_claim.add_argument("--later-history-stdout", required=True)

    rejected_retry = commands.add_parser(
        "authorize-retry-after-rejected-submit"
    )
    _add_root(rejected_retry)
    rejected_retry.add_argument("--intent", required=True)
    rejected_retry.add_argument("--failed-result", required=True)
    rejected_retry.add_argument("--later-snapshot", required=True)
    rejected_retry.add_argument("--queue-stdout", required=True)
    rejected_retry.add_argument("--history-stdout", required=True)
    rejected_retry.add_argument("--prior-snapshot", required=True)
    rejected_retry.add_argument("--prior-queue-stdout", required=True)
    rejected_retry.add_argument("--prior-history-stdout", required=True)

    submit = commands.add_parser("record-submit")
    _add_root(submit)
    submit.add_argument("--intent", required=True)
    submit.add_argument("--authorization", required=True)
    submit.add_argument("--returncode", type=int, required=True)
    submit.add_argument("--stdout", required=True)
    submit.add_argument("--stderr", default="")
    submit.add_argument("--submitted-at")

    recover = commands.add_parser("recover")
    _add_root(recover)
    recover.add_argument("--intent", required=True)
    recover.add_argument("--provisional-result", required=True)
    recover.add_argument("--later-snapshot", required=True)
    recover.add_argument("--queue-stdout", required=True)
    recover.add_argument("--history-stdout", required=True)
    recover.add_argument("--prior-snapshot")
    recover.add_argument("--prior-queue-stdout")
    recover.add_argument("--prior-history-stdout")

    validate = commands.add_parser("validate")
    _add_root(validate)
    validate.add_argument("--path", required=True)
    return parser


def _source_archives(rows: Sequence[str], *, option: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in rows:
        lane, separator, path = str(row).partition("=")
        if not separator or not lane or not path or lane in result:
            raise ValueError(f"{option} must be unique LANE=PATH entries")
        result[lane] = path
    return result


def _intent_output(root: Path, intent_path: str, category: str) -> str:
    intent = load_submission_intent(intent_path, repository_root=root)
    return (
        Path("output")
        / intent.lane_id
        / "acquisition_supervisor"
        / category
        / f"{intent.unit.unit_id}.json"
    ).as_posix()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.repository_root).resolve(strict=True)
    if args.command == "prepare":
        payload = prepare_bootstrap_activation(
            repository_root=root,
            output_dir=args.output_dir,
            seed_receipt_paths=args.seed_receipt,
            source_epoch_receipt_paths=args.source_epoch_receipt,
            exact_source_archive_paths=_source_archives(
                args.exact_source_archive, option="--exact-source-archive"
            ),
            avqds_source_archive_paths=_source_archives(
                args.avqds_source_archive, option="--avqds-source-archive"
            ),
            runtime_image_identity_path=args.runtime_image_identity,
            remote_root=args.remote_root,
            exact_resources=SeedCHTCResources(
                args.exact_cpus,
                args.exact_memory_gib,
                args.exact_disk_gib,
                args.exact_runtime_seconds,
            ),
            scout_resources=AVQDSCHTCResources(
                args.scout_cpus,
                args.scout_memory_gib,
                args.scout_disk_gib,
                args.scout_runtime_seconds,
            ),
            free_bytes=args.free_bytes,
            recorded_at=args.recorded_at,
        )
        summary = {
            "status": payload["status"],
            "activation_manifest": (
                Path(args.output_dir) / "activation_manifest.json"
            ).as_posix(),
            "activation_manifest_sha256": payload[
                "activation_manifest_sha256"
            ],
            "pair_count": payload["pair_count"],
            "action_count": payload["action_count"],
            "exact_and_avqds_launch_independently": True,
            "siam_dependency": False,
            "scheduler_contacted": False,
            "submission_authorized": False,
        }
    elif args.command == "record-upload":
        payload = record_upload_verification(
            repository_root=root,
            intent_path=args.intent,
            remote_bytes=args.remote_bytes,
            remote_sha256=args.remote_sha256,
            recorded_at=args.recorded_at,
        )
        summary = {
            "status": payload["status"],
            "upload_verification": _intent_output(
                root, args.intent, "upload_verifications"
            ),
            "upload_verification_sha256": payload[
                "upload_verification_sha256"
            ],
        }
    elif args.command == "record-dry-run":
        payload = record_dry_run_result(
            repository_root=root,
            intent_path=args.intent,
            returncode=args.returncode,
            dry_run_ad_path=args.dry_run_ad,
            remote_submit_descriptor_path=args.remote_submit_descriptor,
            stdout=args.stdout,
            stderr=args.stderr,
            recorded_at=args.recorded_at,
        )
        summary = {
            "status": payload["status"],
            "dry_run_result": _intent_output(root, args.intent, "dry_run_results"),
            "dry_run_result_sha256": payload["dry_run_result_sha256"],
            "submission_authorized": False,
        }
    elif args.command == "authorize-from-snapshot":
        summary = authorize_from_snapshot(
            repository_root=root,
            intent_path=args.intent,
            snapshot_path=args.snapshot,
            queue_stdout_path=args.queue_stdout,
            history_stdout_path=args.history_stdout,
            prior_snapshot_path=args.prior_snapshot,
            prior_queue_stdout_path=args.prior_queue_stdout,
            prior_history_stdout_path=args.prior_history_stdout,
        )
        if summary.get("schema") is not None:
            summary = {
                "status": summary["status"],
                "submission_authorization": summary["authorization_path"],
                "submission_authorization_sha256": summary[
                    "submission_authorization_sha256"
                ],
                "claim_generation": summary["claim_generation"],
                "single_condor_submit_authorized": True,
            }
    elif args.command == "recover-claim-after-absence":
        payload = recover_claim_after_absence(
            repository_root=root,
            intent_path=args.intent,
            predecessor_claim_path=args.predecessor_claim,
            first_snapshot_path=args.first_snapshot,
            first_queue_stdout_path=args.first_queue_stdout,
            first_history_stdout_path=args.first_history_stdout,
            later_snapshot_path=args.later_snapshot,
            later_queue_stdout_path=args.later_queue_stdout,
            later_history_stdout_path=args.later_history_stdout,
        )
        summary = {
            "status": payload["status"],
            "submission_authorization": payload["authorization_path"],
            "submission_authorization_sha256": payload[
                "submission_authorization_sha256"
            ],
            "claim_generation": payload["claim_generation"],
            "predecessor_resolution": payload["claim_resolution"],
            "single_condor_submit_authorized": True,
        }
    elif args.command == "authorize-retry-after-rejected-submit":
        payload = authorize_retry_after_rejected_submit(
            repository_root=root,
            intent_path=args.intent,
            failed_result_path=args.failed_result,
            later_snapshot_path=args.later_snapshot,
            queue_stdout_path=args.queue_stdout,
            history_stdout_path=args.history_stdout,
            prior_snapshot_path=args.prior_snapshot,
            prior_queue_stdout_path=args.prior_queue_stdout,
            prior_history_stdout_path=args.prior_history_stdout,
        )
        summary = {
            "status": payload["status"],
            "submission_authorization": payload["authorization_path"],
            "submission_authorization_sha256": payload[
                "submission_authorization_sha256"
            ],
            "claim_generation": payload["claim_generation"],
            "predecessor_resolution": payload["claim_resolution"],
            "single_condor_submit_authorized": True,
        }
    elif args.command == "record-submit":
        payload = record_provisional_submit_result(
            repository_root=root,
            intent_path=args.intent,
            authorization_path=args.authorization,
            returncode=args.returncode,
            stdout=args.stdout,
            stderr=args.stderr,
            submitted_at=args.submitted_at,
        )
        authorization_name = Path(
            payload["submission_authorization"]["path"]
        ).name
        result_path = (
            Path("output")
            / payload["lane_id"]
            / "acquisition_supervisor"
            / "submission_results"
            / authorization_name
        ).as_posix()
        summary = {
            "status": payload["status"],
            "provisional_result": result_path,
            "submission_result_sha256": payload["submission_result_sha256"],
            "cluster_id": payload["cluster_id"],
            "scheduler_acceptance_query_performed": False,
            "automatic_retry_authorized": False,
        }
    elif args.command == "recover":
        payload = recover_submission_receipt(
            repository_root=root,
            intent_path=args.intent,
            provisional_result_path=args.provisional_result,
            later_snapshot_path=args.later_snapshot,
            queue_stdout_path=args.queue_stdout,
            history_stdout_path=args.history_stdout,
            prior_snapshot_path=args.prior_snapshot,
            prior_queue_stdout_path=args.prior_queue_stdout,
            prior_history_stdout_path=args.prior_history_stdout,
        )
        receipt_path = (
            Path("output")
            / str(payload["lane_id"])
            / "acquisition_supervisor"
            / "submission_receipts"
            / f"{payload['unit_id']}.json"
        ).as_posix()
        summary = {
            "status": "canonical_submission_receipt_recovered",
            "submission_receipt": receipt_path,
            "cluster_id": payload["remote"]["cluster_id"],
            "recovered_without_post_submit_query": True,
        }
    else:
        summary = validate_artifact(args.path, repository_root=root)
    print(
        json.dumps(
            summary,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through ``main``
    raise SystemExit(main())
