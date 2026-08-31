#!/usr/bin/env python3
"""Materialize or revalidate one prospective exact trajectory score."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from pipelines.time_dynamics.prospective_execution_bindings import (
    ExecutionRequest,
    ProspectiveExecutionError,
    VerifiedFile,
)
from pipelines.time_dynamics.prospective_scientific_planner import (
    ProspectiveScientificBindings,
)
from pipelines.time_dynamics.prospective_trajectory_scoring_worker import (
    materialization_request,
    score_completed_trajectory,
)


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(path: Path, *, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProspectiveExecutionError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ProspectiveExecutionError(f"{label} must be a JSON object")
    return value


def _inside(root: Path, value: str, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ProspectiveExecutionError(f"{label} is outside repository") from exc
    return path


def _verified(root: Path, value: str, *, label: str) -> VerifiedFile:
    path = _inside(root, value, label=label)
    return VerifiedFile(path.relative_to(root).as_posix(), _file_sha(path))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--bindings-json", required=True)
    parser.add_argument("--completed-run-receipt", required=True)
    parser.add_argument("--raw-completion-receipt", required=True)
    parser.add_argument("--exact-reference-receipt", required=True)
    parser.add_argument("--planner-receipt", required=True)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--score-receipt-output", required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "require existing outputs, then recompute and byte-verify the "
            "idempotent score without replacing evidence"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.repository_root).resolve(strict=True)
    request_path = _inside(root, args.request_json, label="score request")
    bindings_path = _inside(root, args.bindings_json, label="scientific bindings")
    request = ExecutionRequest.from_mapping(
        _mapping(request_path, label="score request")
    )
    bindings = ProspectiveScientificBindings.from_json_dict(
        _mapping(bindings_path, label="scientific bindings")
    )
    materialization_request(request)
    metrics = Path(args.metrics_output)
    receipt = Path(args.score_receipt_output)
    if not metrics.is_absolute():
        metrics = root / metrics
    if not receipt.is_absolute():
        receipt = root / receipt
    if args.validate_only and (not metrics.is_file() or not receipt.is_file()):
        raise ProspectiveExecutionError(
            "validate-only requires existing score metrics and receipt"
        )
    result = score_completed_trajectory(
        request,
        bindings=bindings,
        completed_run_receipt=_verified(
            root, args.completed_run_receipt, label="completed run receipt"
        ),
        raw_completion_receipt=_verified(
            root, args.raw_completion_receipt, label="raw completion receipt"
        ),
        exact_reference_receipt=_verified(
            root, args.exact_reference_receipt, label="exact reference receipt"
        ),
        planner_receipt=_verified(
            root, args.planner_receipt, label="scientific planner receipt"
        ),
        metrics_path=metrics,
        score_receipt_path=receipt,
        repository_root=root,
    )
    print(
        json.dumps(
            {
                "status": "validated" if args.validate_only else "materialized",
                "score_receipt_path": result.receipt_path,
                "score_receipt_sha256": result.receipt_sha256,
                "metrics_path": result.metrics.path,
                "metrics_sha256": result.metrics.sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through ``main``
    raise SystemExit(main())
