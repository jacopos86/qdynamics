"""Prepare or run one digest-bound prospective full-APM/exchange action."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from pipelines.time_dynamics.prospective_apm_exchange_worker import (
    build_work_order,
    load_work_order,
    materialize_apm_exchange,
    write_work_order,
)
from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    build_identity_registry,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    ExecutionRequest,
    SeedQualificationReceipt,
    SourceEpochReceipt,
    VerifiedFile,
)
from pipelines.time_dynamics.prospective_scientific_planner import (
    ProspectiveScientificBindings,
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


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request(path: Path) -> ExecutionRequest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("execution request JSON must contain one object")
    return ExecutionRequest.from_mapping(payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="write one immutable source-composed APM work order"
    )
    prepare.add_argument("--repo-root", default=".")
    prepare.add_argument("--request-json", required=True)
    prepare.add_argument("--source-epoch-receipt", required=True)
    prepare.add_argument("--seed-receipt", required=True)
    prepare.add_argument("--resume-run")
    prepare.add_argument("--resume-receipt")
    prepare.add_argument("--output", required=True)
    run = commands.add_parser(
        "run", help="execute and seal one completed-unscored APM trajectory"
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
        bindings = ProspectiveScientificBindings.from_json_dict(request.inputs)
        source = SourceEpochReceipt.load(
            _repository_path(
                root, args.source_epoch_receipt, must_exist=True
            ),
            repository_root=root,
        )
        registry = build_identity_registry()
        seeds = tuple(
            row for row in registry.seeds if row.seed_id == bindings.seed_id
        )
        if len(seeds) != 1:
            raise ValueError("APM request names no canonical seed")
        seed_receipt = SeedQualificationReceipt.load(
            _repository_path(root, args.seed_receipt, must_exist=True),
            repository_root=root,
            expected_seed=seeds[0],
            expected_source_epoch=source.source_epoch,
        )

        def optional_file(value: str | None) -> VerifiedFile | None:
            if value is None:
                return None
            path = _repository_path(root, value, must_exist=True)
            return VerifiedFile(
                path.relative_to(root).as_posix(), _file_sha(path)
            )

        order = build_work_order(
            request,
            source_epoch_receipt=source,
            seed_receipt=seed_receipt,
            repository_root=root,
            resume_run=optional_file(args.resume_run),
            resume_receipt=optional_file(args.resume_receipt),
        )
        output = _repository_path(root, args.output, must_exist=False)
        write_work_order(output, order)
        summary = {
            "status": "prepared_not_executed",
            "work_order": output.relative_to(root).as_posix(),
            "work_order_sha256": order.work_order_sha256,
            "action_id": order.request.action_id,
            "base_seed_source_epoch_identity_sha256": (
                source.source_epoch.identity_sha256
            ),
            "executor_overlay_revision": (
                order.executor_overlay.overlay_revision
            ),
            "executor_overlay_identity_sha256": (
                order.executor_overlay.identity_sha256
            ),
            "scheduler_contacted": False,
            "production_submission_enabled": False,
        }
    else:
        order, _source, _seed = load_work_order(
            _repository_path(root, args.work_order, must_exist=True),
            repository_root=root,
        )
        receipt = materialize_apm_exchange(order, repository_root=root)
        summary = {
            "status": receipt.status,
            "action_id": receipt.action_id,
            "run": receipt.run.path,
            "run_sha256": receipt.run.sha256,
            "completion_receipt": receipt.receipt_path,
            "completion_receipt_sha256": receipt.receipt_sha256,
            "base_source_epoch_identity_sha256": (
                receipt.source_epoch_identity_sha256
            ),
            "executed_overlay_revision": receipt.executor_overlay_revision,
            "scientific_qualification": None,
            "qiskit_compilation_used": False,
        }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
