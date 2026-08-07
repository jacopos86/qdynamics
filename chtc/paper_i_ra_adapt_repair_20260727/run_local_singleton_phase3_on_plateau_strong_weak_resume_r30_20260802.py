#!/usr/bin/env python3
"""Continue the strong--weak singleton control/target pair from r20 to r30."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from chtc.paper_i_ra_adapt_repair_20260727 import (  # noqa: E402
    run_local_singleton_phase3_on_plateau_strong_weak_resume_r20_20260802 as leg,
)


SOURCE_OUTPUT_ROOT = (
    REPO_ROOT
    / "output/local_runs"
    / "paper_i_ra_adapt_singleton_phase3_on_plateau_strong_weak_r20_resume_"
    "local_20260802_v1"
)
SOURCE_MATERIALIZATION_ROOT = SOURCE_OUTPUT_ROOT / "materialization"
SOURCE_RUNS_ROOT = SOURCE_OUTPUT_ROOT / "runs"
SOURCE_CELL_IDS = {
    "control": (
        "phase3_plateau_control_r20__strong_weak_u8__nph3__"
        "ra_singleton_plateau"
    ),
    "target": (
        "phase3_plateau_target_r20__strong_weak_u8__nph3__"
        "ra_singleton_plateau"
    ),
}
OUTPUT_ROOT = (
    REPO_ROOT
    / "output/local_runs"
    / "paper_i_ra_adapt_singleton_phase3_on_plateau_strong_weak_r30_resume_"
    "local_20260802_v1"
)
MATERIALIZATION_ROOT = OUTPUT_ROOT / "materialization"
RUNS_ROOT = OUTPUT_ROOT / "runs"
BUNDLE_ID = (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_strong_weak_r30_resume_"
    "local_v1"
)
CELL_IDS = {
    "control": (
        "phase3_plateau_control_r30__strong_weak_u8__nph3__"
        "ra_singleton_plateau"
    ),
    "target": (
        "phase3_plateau_target_r30__strong_weak_u8__nph3__"
        "ra_singleton_plateau"
    ),
}
SOURCE_ROUND = 20
TARGET_ROUND = 30


def _configure() -> None:
    leg.SOURCE_OUTPUT_ROOT = SOURCE_OUTPUT_ROOT
    leg.SOURCE_MATERIALIZATION_ROOT = SOURCE_MATERIALIZATION_ROOT
    leg.SOURCE_RUNS_ROOT = SOURCE_RUNS_ROOT
    leg.SOURCE_CELL_IDS = SOURCE_CELL_IDS
    leg.OUTPUT_ROOT = OUTPUT_ROOT
    leg.MATERIALIZATION_ROOT = MATERIALIZATION_ROOT
    leg.RUNS_ROOT = RUNS_ROOT
    leg.BUNDLE_ID = BUNDLE_ID
    leg.CELL_IDS = CELL_IDS
    leg.SOURCE_ROUND = SOURCE_ROUND
    leg.TARGET_ROUND = TARGET_ROUND


def _validate_source(role: str) -> dict[str, Any]:
    from pipelines.static_adapt.ra_adapt.contracts import (
        resolved_ra_adapt_protocol_from_mapping,
    )

    _configure()
    leg._configure_base()
    source_cell = SOURCE_CELL_IDS[role]
    target_cell = CELL_IDS[role]
    source_root = SOURCE_RUNS_ROOT / source_cell
    source_protocol = resolved_ra_adapt_protocol_from_mapping(
        leg._protocol_payload(SOURCE_MATERIALIZATION_ROOT, source_cell)
    )
    target_protocol = leg.base._load_bound_protocol(target_cell)
    terminal = leg._require_digest(
        source_root / "terminal_receipt.json",
        label=f"{role} round-20 terminal",
    )
    manifest = leg._require_digest(
        source_root / "run_manifest.json",
        label=f"{role} round-20 manifest",
    )
    authorization = leg._require_digest(
        source_root / "execution_authorization.json",
        label=f"{role} round-20 authorization",
    )
    if (
        terminal.get("status") != "passed"
        or terminal.get("cell_id") != source_cell
        or int(terminal.get("accepted_controller_rounds", -1))
        != SOURCE_ROUND
        or terminal.get("protocol_sha256") != source_protocol.sha256
        or manifest.get("protocol_sha256") != source_protocol.sha256
        or manifest.get("execution_authorization_sha256")
        != authorization.get("sha256")
        or target_protocol.route_contract["sha256"]
        != source_protocol.route_contract["sha256"]
        or target_protocol.problem != source_protocol.problem
    ):
        raise leg.ContinuationContractError(
            f"{role} round-20 source binding drifted."
        )
    checkpoint = source_root / "checkpoint.json"
    result_path = source_root / "result.json"
    summary_path = source_root / "paper_i_summary.json"
    bindings = {
        "checkpoint": leg._binding(checkpoint, root=source_root),
        "result": leg._binding(result_path, root=source_root),
        "paper_i_summary": leg._binding(summary_path, root=source_root),
        "terminal": leg._binding(
            source_root / "terminal_receipt.json", root=source_root
        ),
    }
    for artifact in ("checkpoint", "result", "paper_i_summary"):
        if bindings[artifact] != terminal.get(artifact):
            raise leg.ContinuationContractError(
                f"{role} round-20 {artifact} drifted."
            )
    sidecars = leg._checkpoint_sidecars(
        checkpoint,
        expected_depth=SOURCE_ROUND,
    )
    if sidecars != terminal.get("checkpoint_sidecars"):
        raise leg.ContinuationContractError(
            f"{role} round-20 checkpoint sidecars drifted."
        )
    source_result = leg.base._load_json(result_path)
    source_run = source_result.get("run")
    if not isinstance(source_run, Mapping):
        raise leg.ContinuationContractError(
            f"{role} round-20 result has no run payload."
        )
    for key in (
        "accepted_trajectory",
        "accepted_transitions",
        "scientific_replay",
    ):
        if (
            not isinstance(source_run.get(key), list)
            or len(source_run[key]) != SOURCE_ROUND
        ):
            raise leg.ContinuationContractError(
                f"{role} round-20 {key} is incomplete."
            )
    receipts = source_result.get("scientific_receipts", {}).get(
        "accepted_round_receipts"
    )
    if not isinstance(receipts, list) or len(receipts) != SOURCE_ROUND:
        raise leg.ContinuationContractError(
            f"{role} round-20 accepted receipts are incomplete."
        )
    return {
        "source_cell": source_cell,
        "target_cell": target_cell,
        "source_root": source_root,
        "source_protocol": source_protocol,
        "target_protocol": target_protocol,
        "terminal": terminal,
        "manifest": manifest,
        "authorization": authorization,
        "bindings": bindings,
        "sidecars": sidecars,
        "result": source_result,
    }


def _install() -> None:
    _configure()
    leg._validate_source = _validate_source


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--materialize", action="store_true")
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run-cell", choices=("control", "target"))
    action.add_argument("--finalize", action="store_true")
    parser.add_argument("--execution-authorized", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    _install()
    if args.materialize:
        if args.execution_authorized:
            raise leg.ContinuationContractError(
                "Materialization cannot carry execution authorization."
            )
        result = leg.materialize()
    elif args.preflight:
        if args.execution_authorized:
            raise leg.ContinuationContractError(
                "Preflight cannot carry execution authorization."
            )
        result = leg.preflight()
    elif args.run_cell is not None:
        if not args.execution_authorized:
            raise leg.ContinuationContractError(
                "Scientific continuation requires --execution-authorized."
            )
        result = leg.run_cell(args.run_cell)
    else:
        if args.execution_authorized:
            raise leg.ContinuationContractError(
                "Finalization does not carry execution authorization."
            )
        result = leg.finalize()
    print(leg.base._canonical_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
