"""Prepare or execute one source-locked prospective Paper-II HH seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from pipelines.time_dynamics.prospective_execution_bindings import (
    ExecutionRequest,
    SourceEpochReceipt,
)
from pipelines.time_dynamics.prospective_hh_seed_worker import (
    build_work_order,
    load_work_order,
    materialize_seed,
    write_work_order,
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
            "Prepare or execute a canonical prospective Paper-II Hubbard--Holstein "
            "generator-derived fixed-structure VQE seed."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="validate a canonical request and write an immutable work order"
    )
    prepare.add_argument("--repo-root", default=".")
    prepare.add_argument("--request-json", required=True)
    prepare.add_argument("--source-epoch-receipt", required=True)
    prepare.add_argument("--output", required=True)

    run = subparsers.add_parser(
        "run", help="execute one validated work order and write seed evidence"
    )
    run.add_argument("--repo-root", default=".")
    run.add_argument("--work-order", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.repo_root).resolve(strict=True)
    if args.command == "prepare":
        request_path = _repository_path(root, args.request_json, must_exist=True)
        source_path = _repository_path(
            root, args.source_epoch_receipt, must_exist=True
        )
        output_path = _repository_path(root, args.output, must_exist=False)
        request = _request(request_path)
        source_receipt = SourceEpochReceipt.load(
            source_path, repository_root=root
        )
        order = build_work_order(request, source_epoch_receipt=source_receipt)
        write_work_order(output_path, order)
        print(
            json.dumps(
                {
                    "status": "prepared_not_executed",
                    "work_order": output_path.relative_to(root).as_posix(),
                    "work_order_sha256": order.work_order_sha256,
                    "seed_id": order.seed.seed_id,
                    "architecture_policy_sha256": order.architecture[
                        "architecture_policy_sha256"
                    ],
                    "production_submission_enabled": False,
                },
                sort_keys=True,
            )
        )
        return 0

    order, _source_receipt = load_work_order(
        _repository_path(root, args.work_order, must_exist=True),
        repository_root=root,
    )
    receipt = materialize_seed(order, repository_root=root)
    print(
        json.dumps(
            {
                "status": "qualified",
                "seed_id": receipt.seed_id,
                "seed_receipt": receipt.receipt_path,
                "seed_receipt_sha256": receipt.receipt_sha256,
                "energy_error": receipt.energy_error,
                "exact_state_infidelity": receipt.exact_state_infidelity,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
