"""Prepare or execute one source-locked prospective exact reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    build_identity_registry,
)
from pipelines.time_dynamics.prospective_exact_reference_worker import (
    build_work_order,
    load_work_order,
    materialize_exact_reference,
    write_work_order,
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
            "Prepare or execute a canonical prospective Paper-II HH/Hubbard "
            "full-horizon exact reference."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare", help="write one immutable digest-bound exact-reference work order"
    )
    prepare.add_argument("--repo-root", default=".")
    prepare.add_argument("--request-json", required=True)
    prepare.add_argument("--source-epoch-receipt", required=True)
    prepare.add_argument("--seed-receipt", required=True)
    prepare.add_argument("--output", required=True)
    run = subparsers.add_parser(
        "run", help="execute one validated work order and seal exact evidence"
    )
    run.add_argument("--repo-root", default=".")
    run.add_argument("--work-order", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.repo_root).resolve(strict=True)
    if args.command == "prepare":
        request = _request(
            _repository_path(root, args.request_json, must_exist=True)
        )
        source = SourceEpochReceipt.load(
            _repository_path(
                root, args.source_epoch_receipt, must_exist=True
            ),
            repository_root=root,
        )
        registry = build_identity_registry()
        references = tuple(
            row
            for row in registry.references
            if row.reference_id == request.inputs.get("reference_id")
        )
        if len(references) != 1:
            raise ValueError("request does not name one canonical exact reference")
        reference = references[0]
        seed = registry.seed_for_cell(reference.cell_id)
        seed_receipt = SeedQualificationReceipt.load(
            _repository_path(root, args.seed_receipt, must_exist=True),
            repository_root=root,
            expected_seed=seed,
            expected_source_epoch=source.source_epoch,
        )
        order = build_work_order(
            request,
            source_epoch_receipt=source,
            seed_receipt=seed_receipt,
        )
        output = _repository_path(root, args.output, must_exist=False)
        write_work_order(output, order)
        summary = {
            "status": "prepared_not_executed",
            "work_order": output.relative_to(root).as_posix(),
            "work_order_sha256": order.work_order_sha256,
            "reference_id": order.reference.reference_id,
            "seed_receipt_sha256": order.seed_receipt_sha256,
            "remote_raw_evidence_policy": "preserve",
            "production_submission_enabled": False,
        }
    else:
        order, _source, _seed = load_work_order(
            _repository_path(root, args.work_order, must_exist=True),
            repository_root=root,
        )
        receipt = materialize_exact_reference(order, repository_root=root)
        summary = {
            "status": "verified",
            "reference_id": receipt.reference_id,
            "exact_reference_receipt": receipt.receipt_path,
            "exact_reference_receipt_sha256": receipt.receipt_sha256,
            "manifest_sha256": receipt.manifest.sha256,
            "data_sha256": receipt.data.sha256,
            "remote_raw_evidence_policy": "preserve",
        }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
