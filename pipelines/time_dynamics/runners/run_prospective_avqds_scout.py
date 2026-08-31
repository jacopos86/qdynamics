"""Prepare or execute one exact-blind prospective AVQDS prefix action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from pipelines.time_dynamics.prospective_avqds_scout_worker import (
    build_work_order,
    load_work_order,
    materialize_scout,
    write_work_order,
)
from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    build_identity_registry,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    ExecutionRequest,
    SeedQualificationReceipt,
    SourceEpochReceipt,
)


def _repository_path(root: Path, value: str, *, must_exist: bool) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve(strict=must_exist)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path is outside repository root: {candidate}") from exc
    return candidate


def _request(path: Path) -> ExecutionRequest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("execution request JSON must contain one object")
    return ExecutionRequest.from_mapping(payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare or execute one digest-locked, exact-blind Paper-II "
            "AVQDS calibration prefix."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--repo-root", default=".")
    prepare.add_argument("--request-json", required=True)
    prepare.add_argument("--source-epoch-receipt", required=True)
    prepare.add_argument("--seed-receipt", required=True)
    prepare.add_argument("--output", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--repo-root", default=".")
    run.add_argument("--work-order", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--repo-root", default=".")
    validate.add_argument("--work-order", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.repo_root).resolve(strict=True)
    if args.command == "prepare":
        request = _request(
            _repository_path(root, args.request_json, must_exist=True)
        )
        source = SourceEpochReceipt.load(
            _repository_path(root, args.source_epoch_receipt, must_exist=True),
            repository_root=root,
        )
        registry = build_identity_registry()
        seed_identity = next(
            row
            for row in registry.seeds
            if row.identity_sha256 == request.inputs.get("seed_identity_sha256")
            and row.lane_id == request.lane_id
        )
        seed = SeedQualificationReceipt.load(
            _repository_path(root, args.seed_receipt, must_exist=True),
            repository_root=root,
            expected_seed=seed_identity,
            expected_source_epoch=source.source_epoch,
        )
        order = build_work_order(
            request,
            source_epoch_receipt=source,
            seed_receipt=seed,
            repository_root=root,
        )
        output = _repository_path(root, args.output, must_exist=False)
        write_work_order(output, order)
        print(
            json.dumps(
                {
                    "status": "prepared_not_executed",
                    "work_order": output.relative_to(root).as_posix(),
                    "work_order_sha256": order.work_order_sha256,
                    "action_id": order.request.action_id,
                    "scientific_qualification": None,
                    "production_submission_enabled": False,
                },
                sort_keys=True,
            )
        )
        return 0

    order, _source, _seed = load_work_order(
        _repository_path(root, args.work_order, must_exist=True),
        repository_root=root,
    )
    if args.command == "validate":
        print(
            json.dumps(
                {
                    "status": "validated_not_executed",
                    "action_id": order.request.action_id,
                    "work_order_sha256": order.work_order_sha256,
                    "production_submission_enabled": False,
                },
                sort_keys=True,
            )
        )
        return 0
    receipt = materialize_scout(order, repository_root=root)
    print(
        json.dumps(
            {
                "status": receipt.status,
                "action_id": receipt.action_id,
                "run": receipt.run.to_json_dict(),
                "completion_receipt": receipt.receipt_path,
                "completion_receipt_sha256": receipt.receipt_sha256,
                "scientific_qualification": None,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
