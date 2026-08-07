#!/usr/bin/env python3
"""Continue the authenticated strong--weak target from r30 to r50."""

from __future__ import annotations

import argparse
import copy
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
from chtc.paper_i_ra_adapt_repair_20260727 import (  # noqa: E402
    run_local_singleton_phase3_on_plateau_strong_weak_resume_r30_20260802 as r30,
)
from chtc.paper_i_ra_adapt_repair_20260727 import (  # noqa: E402
    run_local_singleton_phase3_on_plateau_strong_weak_r30_retry_20260802 as r30_retry,
)


SOURCE_OUTPUT_ROOT = r30_retry.OUTPUT_ROOT
SOURCE_MATERIALIZATION_ROOT = r30.MATERIALIZATION_ROOT
SOURCE_RUNS_ROOT = SOURCE_OUTPUT_ROOT / "runs"
SOURCE_PROTOCOL_CELL_IDS = copy.deepcopy(r30.CELL_IDS)
SOURCE_RUN_DIRS = {"control": "control", "target": "target"}
OUTPUT_ROOT = (
    REPO_ROOT
    / "output/local_runs"
    / "paper_i_ra_adapt_singleton_phase3_on_plateau_strong_weak_r50_resume_"
    "local_20260802_v1"
)
MATERIALIZATION_ROOT = OUTPUT_ROOT / "materialization"
RUNS_ROOT = OUTPUT_ROOT / "runs"
BUNDLE_ID = (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_strong_weak_r50_resume_"
    "local_v1"
)
CELL_IDS = {
    "control": (
        "phase3_plateau_control_r50__strong_weak_u8__nph3__"
        "ra_singleton_plateau"
    ),
    "target": (
        "phase3_plateau_target_r50__strong_weak_u8__nph3__"
        "ra_singleton_plateau"
    ),
}
SOURCE_ROUND = 30
TARGET_ROUND = 50
EXACT_ENERGY = leg.EXACT_ENERGY


def _configure() -> None:
    leg.SOURCE_OUTPUT_ROOT = SOURCE_OUTPUT_ROOT
    leg.SOURCE_MATERIALIZATION_ROOT = SOURCE_MATERIALIZATION_ROOT
    leg.SOURCE_RUNS_ROOT = SOURCE_RUNS_ROOT
    leg.SOURCE_CELL_IDS = SOURCE_PROTOCOL_CELL_IDS
    leg.OUTPUT_ROOT = OUTPUT_ROOT
    leg.MATERIALIZATION_ROOT = MATERIALIZATION_ROOT
    leg.RUNS_ROOT = RUNS_ROOT
    leg.BUNDLE_ID = BUNDLE_ID
    leg.CELL_IDS = CELL_IDS
    leg.SOURCE_ROUND = SOURCE_ROUND
    leg.TARGET_ROUND = TARGET_ROUND
    leg._configure_base()


def _validate_source(role: str) -> dict[str, Any]:
    from pipelines.static_adapt.ra_adapt.contracts import (
        resolved_ra_adapt_protocol_from_mapping,
    )

    _configure()
    source_cell = SOURCE_PROTOCOL_CELL_IDS[role]
    target_cell = CELL_IDS[role]
    source_root = SOURCE_RUNS_ROOT / SOURCE_RUN_DIRS[role]
    source_protocol = resolved_ra_adapt_protocol_from_mapping(
        leg._protocol_payload(SOURCE_MATERIALIZATION_ROOT, source_cell)
    )
    target_protocol = leg.base._load_bound_protocol(target_cell)
    terminal = leg._require_digest(
        source_root / "terminal_receipt.json",
        label=f"{role} authenticated round-30 terminal",
    )
    manifest = leg._require_digest(
        source_root / "run_manifest.json",
        label=f"{role} authenticated round-30 manifest",
    )
    authorization = leg._require_digest(
        source_root / "execution_authorization.json",
        label=f"{role} authenticated round-30 authorization",
    )
    completion = leg._require_digest(
        SOURCE_OUTPUT_ROOT / "completion_receipt.json",
        label="authenticated round-30 completion",
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
        or authorization.get("execution_authorized") is not True
        or completion.get("status") != "passed"
        or int(completion.get("target_round", -1)) != SOURCE_ROUND
        or completion.get(f"{role}_terminal_sha256") != terminal["sha256"]
        or target_protocol.route_contract["sha256"]
        != source_protocol.route_contract["sha256"]
        or target_protocol.problem != source_protocol.problem
        or target_protocol.algorithm_id != source_protocol.algorithm_id
    ):
        raise leg.ContinuationContractError(
            f"{role} authenticated round-30 source binding drifted."
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
                f"{role} authenticated round-30 {artifact} drifted."
            )
    sidecars = leg._checkpoint_sidecars(
        checkpoint,
        expected_depth=SOURCE_ROUND,
    )
    if sidecars != terminal.get("checkpoint_sidecars"):
        raise leg.ContinuationContractError(
            f"{role} authenticated round-30 checkpoint sidecars drifted."
        )
    source_result = leg.base._load_json(result_path)
    source_run = source_result.get("run")
    if not isinstance(source_run, Mapping):
        raise leg.ContinuationContractError(
            f"{role} authenticated round-30 result has no run payload."
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
                f"{role} authenticated round-30 {key} is incomplete."
            )
    receipts = source_result.get("scientific_receipts", {}).get(
        "accepted_round_receipts"
    )
    if not isinstance(receipts, list) or len(receipts) != SOURCE_ROUND:
        raise leg.ContinuationContractError(
            f"{role} authenticated round-30 receipts are incomplete."
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


def finalize_target() -> dict[str, Any]:
    _install()
    terminal = leg._require_digest(
        RUNS_ROOT / CELL_IDS["target"] / "terminal_receipt.json",
        label="target round-50 terminal",
    )
    activation = leg._require_digest(
        RUNS_ROOT / CELL_IDS["target"] / "activation_validation.json",
        label="target round-50 activation validation",
    )
    source_completion = leg._require_digest(
        SOURCE_OUTPUT_ROOT / "completion_receipt.json",
        label="source round-30 completion",
    )
    if (
        terminal.get("status") != "passed"
        or int(terminal.get("accepted_controller_rounds", -1))
        != TARGET_ROUND
        or activation.get("status") != "passed"
        or terminal.get("activation_validation_sha256")
        != activation.get("sha256")
    ):
        raise leg.ContinuationContractError(
            "Target round-50 continuation is incomplete."
        )
    target_delta = float(terminal["final_same_cutoff_delta_e"])
    append_delta = leg._append_delta_e()
    completion = leg.base._digested(
        {
            "schema": (
                "paper_i_ra_adapt_phase3_plateau_target_continuation_"
                "completion_v1"
            ),
            "status": "passed",
            "source_round": SOURCE_ROUND,
            "target_round": TARGET_ROUND,
            "source_completion_sha256": source_completion["sha256"],
            "target_terminal_sha256": terminal["sha256"],
            "target_activation_validation_sha256": activation["sha256"],
            "target_same_cutoff_delta_e": target_delta,
            "append_same_cutoff_delta_e": append_delta,
            "target_over_append_ratio": target_delta / append_delta,
            "target_open_rounds": [
                int(row["controller_round"])
                for row in activation["rounds"]
                if row["insertion_plateau_domain_open"]
            ],
            "runner": leg._binding(Path(__file__).resolve(), root=REPO_ROOT),
            "execution_authorized": True,
            "submission_authorized": False,
            "completed_at_utc": leg.base._utc_now(),
        }
    )
    path = OUTPUT_ROOT / "completion_receipt.json"
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {path}")
    leg.base._write_json(path, completion)
    return completion


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--materialize", action="store_true")
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run-target", action="store_true")
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
    elif args.run_target:
        if not args.execution_authorized:
            raise leg.ContinuationContractError(
                "Scientific continuation requires --execution-authorized."
            )
        result = leg.run_cell("target")
    else:
        if args.execution_authorized:
            raise leg.ContinuationContractError(
                "Finalization does not carry execution authorization."
            )
        result = finalize_target()
    print(leg.base._canonical_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
