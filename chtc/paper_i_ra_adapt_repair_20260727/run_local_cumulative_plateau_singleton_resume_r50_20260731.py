#!/usr/bin/env python3
"""Resume the strong--strong cumulative-plateau singleton to round 50."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "off"
os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"

from chtc.paper_i_ra_adapt_repair_20260727 import (  # noqa: E402
    run_local_cumulative_plateau_pair_20260731 as base,
)
from chtc.paper_i_ra_adapt_repair_20260727 import (  # noqa: E402
    run_local_cumulative_plateau_singleton_resume_r30_20260731 as resume30,
)


CELL_ID = "core__strong_strong_u8__nph7__ra_singleton_plateau"
SOURCE_ROOT = (
    REPO_ROOT
    / "output/local_runs"
    / "paper_i_ra_adapt_cumulative_plateau_singleton_r30_finalized_local_20260731_v5"
)
SOURCE_CHECKPOINT_ROOT = SOURCE_ROOT / "canonical_resume_checkpoint"
SOURCE_CHECKPOINT = SOURCE_CHECKPOINT_ROOT / "checkpoint.json"
OUTPUT_ROOT = (
    REPO_ROOT
    / "output/local_runs"
    / "paper_i_ra_adapt_cumulative_plateau_singleton_r50_resume_local_20260731_v1"
)
SOURCE_ROUND = 30
TARGET_ROUND = 50
PLATEAU_RATIO = 1.0e-4


class ContinuationContractError(RuntimeError):
    """Fail-closed continuation contract violation."""


def _require_digest(path: Path, *, label: str) -> dict[str, Any]:
    value = base._load_json(path)
    base._verify_digest(value, label=label)
    return value


def _validate_source() -> dict[str, Any]:
    protocol = base._load_bound_protocol(CELL_ID)
    terminal = _require_digest(
        SOURCE_ROOT / "terminal_receipt.json",
        label="round-30 recovered terminal receipt",
    )
    trajectory = _require_digest(
        SOURCE_ROOT / "accepted_trajectory.json",
        label="round-30 recovered trajectory",
    )
    repair_receipt = _require_digest(
        SOURCE_CHECKPOINT_ROOT / "repair_receipt.json",
        label="round-30 checkpoint repair receipt",
    )
    checkpoint_binding = resume30._binding(
        SOURCE_CHECKPOINT, root=SOURCE_CHECKPOINT_ROOT
    )
    sidecars = resume30._checkpoint_sidecars(
        SOURCE_CHECKPOINT, expected_depth=SOURCE_ROUND
    )
    source_rows = trajectory.get("accepted_trajectory")
    points = trajectory.get("same_cutoff_delta_e_points")
    if (
        terminal.get("status") != "passed"
        or terminal.get("protocol_sha256") != protocol.sha256
        or int(terminal.get("accepted_controller_rounds", -1))
        != SOURCE_ROUND
        or trajectory.get("status") != "passed"
        or trajectory.get("protocol_sha256") != protocol.sha256
        or not isinstance(source_rows, list)
        or len(source_rows) != SOURCE_ROUND
        or not isinstance(points, list)
        or len(points) != SOURCE_ROUND
        or repair_receipt.get("status") != "passed"
        or repair_receipt.get("scientific_state_changed") is not False
        or repair_receipt.get("canonical_resume_validation") != "passed"
        or repair_receipt.get("corrected_checkpoint") != checkpoint_binding
    ):
        raise ContinuationContractError(
            "The finalized round-30 continuation source drifted."
        )
    terminal_checkpoint = terminal.get("canonical_resume_checkpoint")
    if (
        not isinstance(terminal_checkpoint, Mapping)
        or terminal_checkpoint.get("sha256") != checkpoint_binding["sha256"]
        or terminal.get("canonical_resume_repair_receipt_sha256")
        != repair_receipt["sha256"]
    ):
        raise ContinuationContractError(
            "The round-30 terminal receipt does not bind its resume source."
        )
    return {
        "protocol": protocol,
        "terminal": terminal,
        "trajectory": trajectory,
        "source_rows": source_rows,
        "checkpoint": checkpoint_binding,
        "checkpoint_sidecars": sidecars,
        "repair_receipt": repair_receipt,
    }


def preflight() -> dict[str, Any]:
    source = _validate_source()
    return base._digested(
        {
            "schema": "paper_i_ra_adapt_local_continuation_preflight_v1",
            "status": "passed",
            "cell_id": CELL_ID,
            "protocol_sha256": source["protocol"].sha256,
            "source_round": SOURCE_ROUND,
            "target_round": TARGET_ROUND,
            "source_checkpoint": source["checkpoint"],
            "source_terminal_receipt_sha256": source["terminal"]["sha256"],
            "plateau_cumulative_decrease_ratio_threshold": PLATEAU_RATIO,
            "output_root": OUTPUT_ROOT.relative_to(REPO_ROOT).as_posix(),
            "output_root_absent": not OUTPUT_ROOT.exists(),
            "execution_authorized": False,
            "submission_authorized": False,
        }
    )


def run() -> dict[str, Any]:
    from pipelines.static_adapt.ra_adapt import (
        RAAdaptOperationalControls,
        run_ra_adapt,
    )
    from pipelines.static_adapt.sr_snake import (
        AcceptedStateResume,
        CheckpointObservation,
        SRObservationPolicy,
    )

    source = _validate_source()
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    protocol = source["protocol"]
    authorization = base._digested(
        {
            "schema": "paper_i_ra_adapt_local_continuation_authorization_v1",
            "cell_id": CELL_ID,
            "protocol_sha256": protocol.sha256,
            "source_terminal_receipt_sha256": source["terminal"]["sha256"],
            "source_checkpoint_sha256": source["checkpoint"]["sha256"],
            "source_round": SOURCE_ROUND,
            "target_round": TARGET_ROUND,
            "authorization_source": "explicit_user_request_2026-07-31",
            "execution_authorized": True,
            "submission_authorized": False,
            "authorized_at_utc": base._utc_now(),
        }
    )
    base._write_json(OUTPUT_ROOT / "execution_authorization.json", authorization)
    checkpoint = OUTPUT_ROOT / "checkpoint.json"
    try:
        manifest = base._digested(
            {
                "schema": "paper_i_ra_adapt_local_continuation_manifest_v1",
                "run_class": "diagnostic_continuation",
                "cell_id": CELL_ID,
                "protocol_sha256": protocol.sha256,
                "active_gradient_policy": protocol.active_gradient_policy,
                "resource_weighting_scope": protocol.resource_weighting_scope,
                "candidate_representation": protocol.candidate_representation,
                "optimizer": protocol.optimizer,
                "optimizer_maxiter": protocol.optimizer_maxiter,
                "adapt_seed": protocol.seeds["adapt"],
                "protocol_horizon": protocol.horizon,
                "source_round": SOURCE_ROUND,
                "target_round": TARGET_ROUND,
                "plateau_cumulative_decrease_ratio_threshold": PLATEAU_RATIO,
                "resume_input": source["checkpoint"],
                "resume_input_sidecars": source["checkpoint_sidecars"],
                "resume_repair_receipt_sha256": source["repair_receipt"][
                    "sha256"
                ],
                "source_terminal_receipt_sha256": source["terminal"]["sha256"],
                "checkpoint_path": "checkpoint.json",
                "result_path": "result.json",
                "summary_path": "paper_i_summary.json",
                "exact_same_cutoff_energy": base.EXACT_ENERGIES[CELL_ID],
                "execution_authorization_sha256": authorization["sha256"],
                "started_at_utc": base._utc_now(),
            }
        )
        base._write_json(OUTPUT_ROOT / "run_manifest.json", manifest)
        controls = RAAdaptOperationalControls(
            maximum_controller_rounds=TARGET_ROUND,
            resume=AcceptedStateResume(
                checkpoint_path=SOURCE_CHECKPOINT,
                checkpoint_sha256=source["checkpoint"]["sha256"],
            ),
            observation=SRObservationPolicy(
                checkpoint=CheckpointObservation(
                    path=checkpoint,
                    every_controller_rounds=1,
                    keep_history_tail=100,
                ),
                estimator_ledger=None,
                resource_rounds=(TARGET_ROUND,),
            ),
        )
        result = run_ra_adapt(
            base._problem_from_receipt(protocol.problem),
            protocol,
            operational_controls=controls,
        )
        payload = result.to_dict()
        resumed_run = payload.get("run")
        if not isinstance(resumed_run, Mapping):
            raise ContinuationContractError("Result run payload is absent.")
        rows = resumed_run.get("accepted_trajectory")
        if not isinstance(rows, list) or len(rows) != TARGET_ROUND:
            raise ContinuationContractError(
                "Continuation did not close its trajectory through round 50."
            )
        if rows[:SOURCE_ROUND] != source["source_rows"]:
            raise ContinuationContractError(
                "Continuation changed the authenticated trajectory prefix."
            )
        base._write_json(OUTPUT_ROOT / "result.json", payload)
        if result.run.paper_i_summary is None:
            raise ContinuationContractError(
                "Continuation returned no Paper-I summary."
            )
        base._write_json(
            OUTPUT_ROOT / "paper_i_summary.json",
            result.run.paper_i_summary.to_dict(),
        )
        writer_sidecars = resume30._checkpoint_sidecars(
            checkpoint, expected_depth=TARGET_ROUND
        )
        canonical = resume30._materialize_occurrence_corrected_checkpoint(
            source_checkpoint=checkpoint,
            destination_root=OUTPUT_ROOT / "canonical_resume_checkpoint",
            expected_depth=TARGET_ROUND,
            protocol=protocol,
            provenance_role="round_50_canonical_resume_checkpoint",
        )
        delta_e = abs(
            float(result.final_state.energy) - base.EXACT_ENERGIES[CELL_ID]
        )
        terminal = base._digested(
            {
                "schema": "paper_i_ra_adapt_local_continuation_terminal_v1",
                "status": "passed",
                "cell_id": CELL_ID,
                "source_round": SOURCE_ROUND,
                "accepted_controller_rounds": TARGET_ROUND,
                "final_same_cutoff_delta_e": delta_e,
                "protocol_sha256": protocol.sha256,
                "manifest_sha256": manifest["sha256"],
                "source_terminal_receipt_sha256": source["terminal"]["sha256"],
                "source_checkpoint_sha256": source["checkpoint"]["sha256"],
                "writer_checkpoint": resume30._binding(
                    checkpoint, root=OUTPUT_ROOT
                ),
                "writer_checkpoint_sidecars": writer_sidecars,
                "canonical_resume_checkpoint": canonical["checkpoint"],
                "canonical_resume_checkpoint_sidecars": canonical[
                    "checkpoint_sidecars"
                ],
                "canonical_resume_repair_receipt": canonical["repair_receipt"],
                "canonical_resume_repair_receipt_sha256": canonical[
                    "repair_receipt_sha256"
                ],
                "result": resume30._binding(
                    OUTPUT_ROOT / "result.json", root=OUTPUT_ROOT
                ),
                "paper_i_summary": resume30._binding(
                    OUTPUT_ROOT / "paper_i_summary.json", root=OUTPUT_ROOT
                ),
                "completed_at_utc": base._utc_now(),
            }
        )
        base._write_json(OUTPUT_ROOT / "terminal_receipt.json", terminal)
        return terminal
    except BaseException as exc:
        failure = base._digested(
            {
                "schema": "paper_i_ra_adapt_local_continuation_failure_v1",
                "status": "failed",
                "cell_id": CELL_ID,
                "source_round": SOURCE_ROUND,
                "target_round": TARGET_ROUND,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "checkpoint_present": checkpoint.is_file(),
                "failed_at_utc": base._utc_now(),
            }
        )
        base._write_json(OUTPUT_ROOT / "failure_receipt.json", failure)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--execution-authorized", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.preflight:
        if args.execution_authorized:
            raise ContinuationContractError(
                "Preflight cannot carry execution authorization."
            )
        print(base._canonical_bytes(preflight()).decode("utf-8"))
        return 0
    if not args.execution_authorized:
        raise ContinuationContractError(
            "Continuation requires --execution-authorized."
        )
    print(base._canonical_bytes(run()).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
