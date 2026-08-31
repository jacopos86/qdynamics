#!/usr/bin/env python3
"""Recover or validate the sole canonical prospective Hubbard-L4 submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from pipelines.time_dynamics.prospective_hubbard_seed_recovery import (
    HUBBARD_LANE_ID,
    load_hubbard_durable_submission_intent,
    recover_hubbard_submission_receipt,
)
from pipelines.time_dynamics.prospective_submission_recovery import (
    SubmissionRecoveryError,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        "--repo-root",
        dest="repository_root",
        default=".",
        help="repository root containing every bound recovery artifact",
    )
    parser.add_argument("--intent", required=True)
    parser.add_argument(
        "--provisional-result",
        "--submission-result",
        dest="provisional_result",
        required=True,
    )
    parser.add_argument(
        "--later-combined-snapshot",
        "--later-snapshot",
        dest="later_combined_snapshot",
        required=True,
    )
    parser.add_argument(
        "--output",
        help=(
            "optional assertion of the canonical receipt path; arbitrary "
            "recovery destinations are rejected"
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="require and revalidate an existing canonical receipt",
    )
    return parser


def _canonical_receipt_path(
    root: Path, *, intent_path: str | Path
) -> Path:
    intent = load_hubbard_durable_submission_intent(
        intent_path, repository_root=root
    )
    return (
        root
        / "output"
        / HUBBARD_LANE_ID
        / "acquisition_supervisor/submission_receipts"
        / f"{intent.base.unit.unit_id}.json"
    ).resolve(strict=False)


def _assert_output_path(root: Path, value: str | Path, canonical: Path) -> None:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise SubmissionRecoveryError(
            "Hubbard recovery output must remain inside repository root"
        ) from exc
    if candidate != canonical:
        raise SubmissionRecoveryError(
            "Hubbard recovery output must be the canonical Hubbard receipt path"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.repository_root).resolve(strict=True)
    canonical = _canonical_receipt_path(root, intent_path=args.intent)
    if args.output is not None:
        _assert_output_path(root, args.output, canonical)
    if args.validate_only and not canonical.is_file():
        raise SubmissionRecoveryError(
            "validate-only requires an existing canonical Hubbard receipt"
        )

    result = recover_hubbard_submission_receipt(
        repository_root=root,
        intent_path=args.intent,
        provisional_result_path=args.provisional_result,
        later_snapshot_path=args.later_combined_snapshot,
    )
    recovered_path = (root / result.path).resolve(strict=True)
    if recovered_path != canonical:
        raise SubmissionRecoveryError(
            "recovery API returned a noncanonical Hubbard receipt path"
        )

    summary = {
        "status": (
            "canonical_submission_receipt_validated"
            if args.validate_only
            else "canonical_submission_receipt_recovered"
        ),
        "mode": "validate" if args.validate_only else "recover",
        "receipt": result.path,
        "receipt_file_sha256": result.file_sha256,
        "lane_id": HUBBARD_LANE_ID,
        "unit_id": result.unit_id,
        "cluster_id": result.cluster_id,
        "process_ids": list(result.process_ids),
        "snapshot_sha256": result.snapshot_sha256,
        "scheduler_confirmation": True,
        "remote_raw_evidence_policy": "preserve",
        "submission_authorized": False,
    }
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
