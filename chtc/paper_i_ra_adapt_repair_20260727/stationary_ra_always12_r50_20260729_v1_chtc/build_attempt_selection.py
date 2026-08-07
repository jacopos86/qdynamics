#!/usr/bin/env python3
"""Bind an explicit human-selected attempt per cell; never auto-select."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

PACKAGE_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from package_contract import (  # noqa: E402
    DIRECT_EXECUTION_COUNT,
    FETCH_VALIDATION_SCHEMA,
    PACKAGE_ID,
    PackageContractError,
    atomic_write_json,
    canonical_json_bytes,
    digested,
    direct_execution_ids,
    load_json_object,
    verify_self_digest,
)


SELECTION_SCHEMA = "paper_i_ra_adapt_stationary_core_attempt_selection_v1"


def build_selection(
    *,
    validation_path: Path,
    choices_path: Path,
    output: Path,
) -> dict[str, Any]:
    validation = load_json_object(
        validation_path, label="fetched validation receipt"
    )
    validation_sha = verify_self_digest(
        validation, label="fetched validation receipt"
    )
    choices = load_json_object(
        choices_path, label="explicit attempt choices"
    )
    choices_sha = verify_self_digest(
        choices, label="explicit attempt choices"
    )
    if (
        validation.get("schema") != FETCH_VALIDATION_SCHEMA
        or validation.get("package_id") != PACKAGE_ID
        or validation.get("automatic_attempt_selection_performed") is not False
        or validation.get("paper_evidence_adopted") is not False
    ):
        raise PackageContractError("Fetched validation authority drifted.")
    if (
        choices.get("schema")
        != "paper_i_ra_adapt_stationary_core_explicit_attempt_choices_v1"
        or choices.get("package_id") != PACKAGE_ID
        or choices.get("selection_authorized_by_user") is not True
        or choices.get("paper_evidence_adoption_authorized") is not False
    ):
        raise PackageContractError(
            "Attempt choices lack explicit selection-only authority."
        )
    raw_choices = choices.get("choices")
    if not isinstance(raw_choices, Mapping) or set(raw_choices) != set(
        direct_execution_ids()
    ):
        raise PackageContractError(
            "Explicit choices must cover exactly all 12 execution ids."
        )
    attempts = validation.get("attempts")
    if not isinstance(attempts, list):
        raise PackageContractError("Fetched validation has no attempts.")
    passed = {
        (str(row["execution_id"]), str(row["sha256"])): row
        for row in attempts
        if isinstance(row, Mapping) and row.get("status") == "passed"
    }
    selected: list[dict[str, Any]] = []
    for execution_id in direct_execution_ids():
        choice = raw_choices[execution_id]
        if not isinstance(choice, Mapping):
            raise PackageContractError(
                f"Choice is invalid: {execution_id}"
            )
        key = (execution_id, str(choice.get("attempt_sha256", "")))
        attempt = passed.get(key)
        if (
            attempt is None
            or choice.get("attempt_path") != attempt.get("path")
        ):
            raise PackageContractError(
                f"Choice is not a passed validated attempt: {execution_id}"
            )
        selected.append(
            {
                "execution_id": execution_id,
                "attempt_path": attempt["path"],
                "attempt_sha256": attempt["sha256"],
                "worker_receipt_sha256": attempt[
                    "worker_receipt_sha256"
                ],
            }
        )
    receipt = digested(
        {
            "schema": SELECTION_SCHEMA,
            "package_id": PACKAGE_ID,
            "fetched_validation_sha256": validation_sha,
            "explicit_choices_file_sha256": choices_sha,
            "selected_count": DIRECT_EXECUTION_COUNT,
            "selected_attempts": selected,
            "automatic_attempt_selection_performed": False,
            "paper_evidence_adopted": False,
            "status": "explicit_selection_bound_not_adopted",
        }
    )
    atomic_write_json(output, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--choices", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = build_selection(
            validation_path=args.validation.resolve(),
            choices_path=args.choices.resolve(),
            output=args.output.resolve(),
        )
        print(canonical_json_bytes(result).decode("utf-8"))
        return 0
    except (OSError, PackageContractError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
