#!/usr/bin/env python3
"""DAGMan POST gate: validate CONTROL before REUSE can be submitted."""

from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
from pathlib import Path

from pair_contract import (
    CONTROL_MODE,
    PAIR_ID,
    SOURCE_ARCHIVE_SHA256,
    bundle_dir,
    dump_json,
    load_json,
    sha256,
)
from validate_fetched import safe_extract, validate_output_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(
            "refusing a stale control gate; remove it before a new DAG submission"
        )
    with tempfile.TemporaryDirectory(prefix="sr-outer-control-post-") as tmp:
        root = Path(tmp)
        safe_extract(args.archive, root)
        output_root = root / "raw_outputs" / bundle_dir().name / CONTROL_MODE
        receipt = validate_output_root(output_root, CONTROL_MODE)
        runtime_gate = load_json(output_root / "anchor_gate.json")
        fetched_validation_receipt_sha256 = sha256(output_root / "validation.json")
    if receipt.get("status") != "pass" or runtime_gate.get("status") != "pass":
        raise ValueError("control transfer did not pass the terminal anchor gate")
    gate = {
        **runtime_gate,
        "schema": "paper_i_sr_outer_information_control_gate_v1",
        "status": "pass",
        "pair_id": PAIR_ID,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "control_transfer_archive_sha256": sha256(args.archive),
        "fetched_validation_receipt_sha256": fetched_validation_receipt_sha256,
        "current_runtime_control_validated": True,
        "historical_anchor_reproduction_status": "not_claimed",
    }
    dump_json(args.output, gate)
    print(json.dumps(gate, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
