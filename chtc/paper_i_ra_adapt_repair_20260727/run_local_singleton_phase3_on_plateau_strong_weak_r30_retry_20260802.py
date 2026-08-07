#!/usr/bin/env python3
"""Resume the disk-interrupted strong--weak pair from authenticated r27 to r30."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import traceback
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


SOURCE_ROOT = r30.OUTPUT_ROOT
SOURCE_RUNS_ROOT = SOURCE_ROOT / "runs"
SOURCE_CELL_IDS = copy.deepcopy(r30.CELL_IDS)
OUTPUT_ROOT = (
    REPO_ROOT
    / "output/local_runs"
    / "paper_i_ra_adapt_singleton_phase3_on_plateau_strong_weak_r30_retry_"
    "local_20260802_v3"
)
RUNS_ROOT = OUTPUT_ROOT / "runs"
SOURCE_ROUND = 27
TARGET_ROUND = 30
EXACT_ENERGY = leg.EXACT_ENERGY


def _install() -> None:
    r30._install()
    leg.SOURCE_ROUND = SOURCE_ROUND
    leg.TARGET_ROUND = TARGET_ROUND
    leg._configure_base()


def _source(role: str) -> dict[str, Any]:
    _install()
    cell_id = SOURCE_CELL_IDS[role]
    run_root = SOURCE_RUNS_ROOT / cell_id
    failure = leg._require_digest(
        run_root / "failure_receipt.json",
        label=f"{role} disk-interrupted failure",
    )
    if (
        failure.get("status") != "failed"
        or failure.get("error_type") != "OSError"
        or "No space left on device" not in str(failure.get("error"))
        or failure.get("checkpoint_present") is not True
    ):
        raise leg.ContinuationContractError(
            f"{role} source is not the known disk interruption."
        )
    protocol = leg.base._load_bound_protocol(cell_id)
    checkpoint = run_root / "checkpoint.json"
    checkpoint_binding = leg._binding(checkpoint)
    sidecars = leg._checkpoint_sidecars(
        checkpoint,
        expected_depth=SOURCE_ROUND,
    )
    manifest = leg._require_digest(
        run_root / "run_manifest.json",
        label=f"{role} interrupted manifest",
    )
    authorization = leg._require_digest(
        run_root / "execution_authorization.json",
        label=f"{role} interrupted authorization",
    )
    if (
        manifest.get("protocol_sha256") != protocol.sha256
        or manifest.get("execution_authorization_sha256")
        != authorization.get("sha256")
        or int(manifest.get("target_round", -1)) != TARGET_ROUND
    ):
        raise leg.ContinuationContractError(
            f"{role} interrupted source provenance drifted."
        )
    return {
        "cell_id": cell_id,
        "run_root": run_root,
        "protocol": protocol,
        "checkpoint": checkpoint,
        "checkpoint_binding": checkpoint_binding,
        "sidecars": sidecars,
        "failure": failure,
        "manifest": manifest,
        "authorization": authorization,
    }


def preflight() -> dict[str, Any]:
    sources = {role: _source(role) for role in ("control", "target")}
    return leg.base._digested(
        {
            "schema": "paper_i_ra_adapt_disk_retry_preflight_v1",
            "status": "passed",
            "source_round": SOURCE_ROUND,
            "target_round": TARGET_ROUND,
            "sources": {
                role: {
                    "cell_id": row["cell_id"],
                    "protocol_sha256": row["protocol"].sha256,
                    "checkpoint": row["checkpoint_binding"],
                    "checkpoint_sidecars": row["sidecars"],
                    "failure_receipt_sha256": row["failure"]["sha256"],
                }
                for role, row in sources.items()
            },
            "output_root": OUTPUT_ROOT.relative_to(REPO_ROOT).as_posix(),
            "output_root_absent": not OUTPUT_ROOT.exists(),
            "execution_authorized": False,
            "submission_authorized": False,
        }
    )


def _trajectory_from_checkpoint(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    adapt = payload.get("adapt_vqe")
    if not isinstance(adapt, Mapping):
        raise leg.ContinuationContractError("Checkpoint has no ADAPT payload.")
    history = adapt.get("history")
    if not isinstance(history, list) or len(history) != SOURCE_ROUND:
        raise leg.ContinuationContractError("Checkpoint history is incomplete.")
    insertion_positions: list[int] = []
    trajectory: list[dict[str, Any]] = []
    for round_index, row in enumerate(history, start=1):
        effective = row.get("selected_effective_positions")
        original = row.get("selected_positions")
        features = row.get("selected_feature_rows")
        if (
            not isinstance(effective, list)
            or not isinstance(original, list)
            or not isinstance(features, list)
            or not (len(effective) == len(original) == len(features))
        ):
            raise leg.ContinuationContractError(
                f"Round {round_index} admission projection is incomplete."
            )
        for effective_position, original_position in zip(
            effective, original, strict=True
        ):
            insertion_positions.insert(
                int(effective_position), int(original_position)
            )
        prune = row.get("post_admission_prune")
        if isinstance(prune, Mapping) and int(prune.get("accepted_count", 0)):
            deleted = prune.get("deleted_indices")
            if not isinstance(deleted, list) or len(deleted) != 1:
                raise leg.ContinuationContractError(
                    f"Round {round_index} prune projection is unsupported."
                )
            del insertion_positions[int(deleted[0])]
        prefix = row.get("active_prefix_checkpoint")
        if not isinstance(prefix, Mapping):
            raise leg.ContinuationContractError(
                f"Round {round_index} signed prefix is absent."
            )
        operators = prefix.get("ordered_active_operators")
        if not isinstance(operators, list):
            raise leg.ContinuationContractError(
                f"Round {round_index} active operators are absent."
            )
        trajectory.append(
            {
                "controller_round": round_index,
                "energy": float(row["energy_after_opt"]),
                "generator_ids": [str(item["generator_id"]) for item in operators],
                "insertion_positions": list(insertion_positions),
                "logical_parameters": [
                    float(value)
                    for value in prefix["signed_unwrapped_logical_parameters"]
                ],
                "operators": [
                    str(value)
                    for value in prefix["ordered_active_operator_labels"]
                ],
                "projective_state_fingerprint": str(
                    prefix["projective_state_fingerprint"]
                ),
                "runtime_parameters": [
                    float(value)
                    for value in prefix["signed_unwrapped_runtime_parameters"]
                ],
            }
        )
    return trajectory


def _materialize_ledger_closure_repair(
    *,
    source: Mapping[str, Any],
    destination_root: Path,
) -> dict[str, Any]:
    from pipelines.static_adapt.current_checkpoint import (
        _publish_active_cli_current_checkpoint,
    )
    from pipelines.static_adapt.sr_snake import AcceptedStateResume
    from pipelines.static_adapt.sr_snake._resume import (
        load_canonical_accepted_state_resume,
    )

    payload = leg.base._load_json(source["checkpoint"])
    repaired = copy.deepcopy(payload)
    continuation = repaired["adapt_vqe"].get("continuation")
    if not isinstance(continuation, dict):
        raise leg.ContinuationContractError(
            "Interrupted checkpoint has no continuation payload."
        )
    if continuation.get("active_prefix_estimator_ledger_closure") is not None:
        raise leg.ContinuationContractError(
            "Ledger-closure repair source is not missing its closure."
        )
    receipt_rows = continuation.get(
        "all_active_prefix_estimator_ledger_receipts"
    )
    terminal_prefix = continuation.get("terminal_active_prefix_checkpoint")
    if (
        not isinstance(receipt_rows, list)
        or not receipt_rows
        or not isinstance(terminal_prefix, Mapping)
        or int(terminal_prefix.get("outer_iteration", -1)) != SOURCE_ROUND
        or terminal_prefix.get("checkpoint_kind") != "post_admission_prune"
        or terminal_prefix.get("estimator_ledger_receipt") != receipt_rows[-1]
        or receipt_rows[-1].get("checkpoint_kind") != "post_admission_prune"
        or int(receipt_rows[-1].get("outer_iteration", -1)) != SOURCE_ROUND
    ):
        raise leg.ContinuationContractError(
            "Interrupted checkpoint terminal-prefix seam is not the expected "
            "post-admission partial state."
        )
    prior_receipt = copy.deepcopy(receipt_rows[-1])
    raw_cumulative = copy.deepcopy(
        prior_receipt["cumulative_raw_occurrences"]
    )
    unique_cumulative = copy.deepcopy(
        prior_receipt["cumulative_unique_primitives"]
    )
    components = ("N_H_outer", "N_H_refit", "N_grad", "N_metric")
    zero_components = {component: 0 for component in components}
    terminal_receipt = copy.deepcopy(prior_receipt)
    terminal_receipt.update(
        {
            "checkpoint_sequence": int(
                prior_receipt["checkpoint_sequence"]
            )
            + 1,
            "occurrence_sequence_start_exclusive": int(
                raw_cumulative["total"]
            ),
            "occurrence_sequence_end_inclusive": int(
                raw_cumulative["total"]
            ),
            "raw_occurrence_delta": {
                "components": copy.deepcopy(zero_components),
                "total": 0,
            },
            "executed_query_delta": {
                "components": copy.deepcopy(zero_components),
                "S_alg": 0,
            },
            "unique_primitive_delta": {
                "components": copy.deepcopy(zero_components),
                "S_unique": 0,
            },
            "cumulative_raw_occurrences": raw_cumulative,
            "cumulative_executed_queries": {
                "components": copy.deepcopy(raw_cumulative["components"]),
                "S_alg": int(raw_cumulative["total"]),
                "unit": "executed_logical_scalar_estimator_invocation",
            },
            "cumulative_unique_primitives": unique_cumulative,
            "outer_iteration": SOURCE_ROUND,
            "checkpoint_kind": "terminal_post_final_refit_and_prune",
        }
    )
    receipt_rows.append(copy.deepcopy(terminal_receipt))
    repaired_terminal_prefix = copy.deepcopy(dict(terminal_prefix))
    repaired_terminal_prefix["checkpoint_kind"] = (
        "terminal_post_final_refit_and_prune"
    )
    repaired_terminal_prefix["estimator_ledger_receipt"] = copy.deepcopy(
        terminal_receipt
    )
    repaired_terminal_prefix.pop("checkpoint_sha256", None)
    repaired_terminal_prefix["checkpoint_sha256"] = hashlib.sha256(
        json.dumps(
            repaired_terminal_prefix,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    continuation["terminal_active_prefix_checkpoint"] = copy.deepcopy(
        repaired_terminal_prefix
    )
    repaired["adapt_vqe"]["terminal_active_prefix_checkpoint"] = (
        copy.deepcopy(repaired_terminal_prefix)
    )
    pointer = repaired["checkpoint"]["estimator_call_ledger_checkpoint"]
    ledger_path = source["checkpoint"].parent / str(pointer["path"])
    ledger_sidecar = leg.base._load_json(ledger_path)
    ledger = ledger_sidecar.get("ledger")
    if not isinstance(ledger, Mapping):
        raise leg.ContinuationContractError(
            "Interrupted checkpoint ledger payload is absent."
        )
    receipts = [
        row
        for row in continuation["all_active_prefix_estimator_ledger_receipts"]
        if bool(row.get("enabled", False))
    ]
    raw_components = {
        component: sum(
            int(row["raw_occurrence_delta"]["components"].get(component, 0))
            for row in receipts
        )
        for component in components
    }
    unique_components = {
        component: sum(
            int(
                row["unique_primitive_delta"]["components"].get(component, 0)
            )
            for row in receipts
        )
        for component in components
    }
    raw_total = sum(int(row["raw_occurrence_delta"]["total"]) for row in receipts)
    unique_total = sum(
        int(
            row["unique_primitive_delta"].get(
                "S_unique", row["unique_primitive_delta"].get("S_alg", 0)
            )
        )
        for row in receipts
    )
    occurrence = ledger["occurrence_summary"]
    summary = ledger["summary"]
    terminal_raw_components = {
        component: int(occurrence["component_occurrence_counts"][component])
        for component in components
    }
    terminal_unique_components = {
        component: int(summary[component]) for component in components
    }
    terminal_raw_total = int(occurrence["total_call_occurrences"])
    terminal_unique_total = int(summary["S_unique"])
    if (
        raw_components != terminal_raw_components
        or unique_components != terminal_unique_components
        or raw_total != terminal_raw_total
        or unique_total != terminal_unique_total
    ):
        raise leg.ContinuationContractError(
            "Interrupted checkpoint receipts do not close to its ledger."
        )
    closure = {
        "schema": "paper_i_active_prefix_estimator_ledger_closure_v1",
        "enabled": True,
        "status": "complete",
        "passed": True,
        "receipt_count": len(receipts),
        "summed_raw_occurrences": {
            "components": raw_components,
            "total": raw_total,
        },
        "summed_unique_primitives": {
            "components": unique_components,
            "S_unique": unique_total,
        },
        "terminal_raw_occurrences": {
            "components": terminal_raw_components,
            "total": terminal_raw_total,
        },
        "terminal_unique_primitives": {
            "components": terminal_unique_components,
            "S_unique": terminal_unique_total,
        },
        "includes_discarded_branch_checkpoints": False,
    }
    continuation["active_prefix_estimator_ledger_closure"] = closure
    destination_root.mkdir(parents=True, exist_ok=False)
    destination = destination_root / "checkpoint.json"
    _publish_active_cli_current_checkpoint(
        repaired,
        ledger_payload=ledger,
        path=destination,
        keep_history_tail=100,
    )
    destination_binding = leg._binding(destination, root=destination_root)
    destination_sidecars = leg._checkpoint_sidecars(
        destination,
        expected_depth=SOURCE_ROUND,
    )
    hydration = load_canonical_accepted_state_resume(
        AcceptedStateResume(
            checkpoint_path=destination,
            checkpoint_sha256=destination_binding["sha256"],
        ),
        expected_problem=leg.base._problem_from_receipt(
            source["protocol"].problem
        ),
        expected_route_profile=str(
            source["protocol"].route_contract["route_profile"]
        ),
        expected_route_contract_sha256=str(
            source["protocol"].route_contract["sha256"]
        ),
    )
    if int(hydration.controller_round) != SOURCE_ROUND:
        raise leg.ContinuationContractError(
            "Ledger-closure repaired checkpoint hydrated at the wrong round."
        )
    receipt = leg.base._digested(
        {
            "schema": "paper_i_ra_adapt_checkpoint_ledger_closure_repair_v1",
            "status": "passed",
            "scientific_state_changed": False,
            "repair_scope": (
                "missing_terminal_prefix_receipt_and_estimator_ledger_"
                "closure_metadata_v1"
            ),
            "source_checkpoint": source["checkpoint_binding"],
            "source_checkpoint_sidecars": source["sidecars"],
            "repaired_checkpoint": destination_binding,
            "repaired_checkpoint_sidecars": destination_sidecars,
            "controller_round": SOURCE_ROUND,
            "closure": closure,
            "canonical_resume_validation": "passed",
            "created_at_utc": leg.base._utc_now(),
        }
    )
    leg.base._write_json(destination_root / "repair_receipt.json", receipt)
    return {
        "checkpoint_path": destination,
        "checkpoint": destination_binding,
        "sidecars": destination_sidecars,
        "receipt": leg._binding(
            destination_root / "repair_receipt.json", root=destination_root
        ),
        "receipt_sha256": receipt["sha256"],
    }


def run_cell(role: str) -> dict[str, Any]:
    from pipelines.static_adapt.ra_adapt import (
        RAAdaptOperationalControls,
        run_ra_adapt,
    )
    from pipelines.static_adapt.sr_snake import (
        AcceptedStateResume,
        CheckpointObservation,
        SRObservationPolicy,
    )

    source = _source(role)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    run_root = RUNS_ROOT / role
    if run_root.exists() or run_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    protocol = source["protocol"]
    authorization = leg.base._digested(
        {
            "schema": "paper_i_ra_adapt_disk_retry_authorization_v1",
            "role": role,
            "cell_id": source["cell_id"],
            "protocol_sha256": protocol.sha256,
            "source_round": SOURCE_ROUND,
            "target_round": TARGET_ROUND,
            "source_checkpoint_sha256": source["checkpoint_binding"]["sha256"],
            "source_failure_sha256": source["failure"]["sha256"],
            "authorization_source": "explicit_user_continuation_2026-08-02",
            "execution_authorized": True,
            "submission_authorized": False,
            "authorized_at_utc": leg.base._utc_now(),
        }
    )
    leg.base._write_json(run_root / "execution_authorization.json", authorization)
    manifest = leg.base._digested(
        {
            "schema": "paper_i_ra_adapt_disk_retry_manifest_v1",
            "role": role,
            "cell_id": source["cell_id"],
            "protocol_sha256": protocol.sha256,
            "route_contract_sha256": protocol.route_contract["sha256"],
            "source_round": SOURCE_ROUND,
            "target_round": TARGET_ROUND,
            "source_checkpoint": source["checkpoint_binding"],
            "source_checkpoint_sidecars": source["sidecars"],
            "source_failure_sha256": source["failure"]["sha256"],
            "execution_authorization_sha256": authorization["sha256"],
            "started_at_utc": leg.base._utc_now(),
        }
    )
    leg.base._write_json(run_root / "run_manifest.json", manifest)
    checkpoint = run_root / "checkpoint.json"
    try:
        repaired_source = _materialize_ledger_closure_repair(
            source=source,
            destination_root=run_root / "resume_source",
        )
        result = run_ra_adapt(
            leg.base._problem_from_receipt(protocol.problem),
            protocol,
            operational_controls=RAAdaptOperationalControls(
                maximum_controller_rounds=TARGET_ROUND,
                resume=AcceptedStateResume(
                    checkpoint_path=repaired_source["checkpoint_path"],
                    checkpoint_sha256=repaired_source["checkpoint"]["sha256"],
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
            ),
        )
        payload = result.to_dict()
        leg.base._write_json(run_root / "result.json", payload)
        if result.run.paper_i_summary is not None:
            leg.base._write_json(
                run_root / "paper_i_summary.json",
                result.run.paper_i_summary.to_dict(),
            )
        source_trajectory = _trajectory_from_checkpoint(
            leg.base._load_json(source["checkpoint"])
        )
        resumed_trajectory = payload["run"]["accepted_trajectory"]
        prefix_passed = bool(
            len(resumed_trajectory) == TARGET_ROUND
            and resumed_trajectory[:SOURCE_ROUND] == source_trajectory
        )
        prefix = leg.base._digested(
            {
                "schema": "paper_i_ra_adapt_disk_retry_prefix_validation_v1",
                "status": "passed" if prefix_passed else "failed",
                "role": role,
                "source_round": SOURCE_ROUND,
                "target_round": TARGET_ROUND,
                "checkpoint_trajectory_exact_prefix_match": prefix_passed,
            }
        )
        leg.base._write_json(run_root / "prefix_validation.json", prefix)
        if not prefix_passed:
            raise leg.ContinuationContractError(
                f"{role} disk retry changed its authenticated prefix."
            )
        sidecars = leg._checkpoint_sidecars(
            checkpoint,
            expected_depth=TARGET_ROUND,
        )
        activation = None
        if role == "target":
            activation = leg._strict_activation_validation(payload)
            leg.base._write_json(
                run_root / "activation_validation.json", activation
            )
            if activation["status"] != "passed":
                raise leg.ContinuationContractError(
                    "Target activation validation failed after disk retry."
                )
        delta_e = abs(float(result.final_state.energy) - EXACT_ENERGY)
        terminal = leg.base._digested(
            {
                "schema": "paper_i_ra_adapt_disk_retry_terminal_v1",
                "status": "passed",
                "role": role,
                "cell_id": source["cell_id"],
                "source_round": SOURCE_ROUND,
                "accepted_controller_rounds": TARGET_ROUND,
                "final_same_cutoff_delta_e": delta_e,
                "protocol_sha256": protocol.sha256,
                "manifest_sha256": manifest["sha256"],
                "source_checkpoint_sha256": source["checkpoint_binding"][
                    "sha256"
                ],
                "source_failure_sha256": source["failure"]["sha256"],
                "resume_repair_receipt": repaired_source["receipt"],
                "resume_repair_receipt_sha256": repaired_source[
                    "receipt_sha256"
                ],
                "checkpoint": leg._binding(checkpoint, root=run_root),
                "checkpoint_sidecars": sidecars,
                "result": leg._binding(
                    run_root / "result.json", root=run_root
                ),
                "paper_i_summary": leg._binding(
                    run_root / "paper_i_summary.json", root=run_root
                ),
                "prefix_validation_sha256": prefix["sha256"],
                "activation_validation_sha256": (
                    None if activation is None else activation["sha256"]
                ),
                "completed_at_utc": leg.base._utc_now(),
            }
        )
        leg.base._write_json(run_root / "terminal_receipt.json", terminal)
        return terminal
    except BaseException as exc:
        failure = leg.base._digested(
            {
                "schema": "paper_i_ra_adapt_disk_retry_failure_v1",
                "status": "failed",
                "role": role,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "checkpoint_present": checkpoint.is_file(),
                "failed_at_utc": leg.base._utc_now(),
            }
        )
        leg.base._write_json(run_root / "failure_receipt.json", failure)
        raise


def finalize() -> dict[str, Any]:
    _install()
    terminals: dict[str, dict[str, Any]] = {}
    for role in ("control", "target"):
        terminal = leg._require_digest(
            RUNS_ROOT / role / "terminal_receipt.json",
            label=f"{role} disk-retry terminal",
        )
        if terminal.get("status") != "passed":
            raise leg.ContinuationContractError(
                f"{role} disk retry is incomplete."
            )
        terminals[role] = terminal
    append_delta = leg._append_delta_e()
    control_delta = float(terminals["control"]["final_same_cutoff_delta_e"])
    target_delta = float(terminals["target"]["final_same_cutoff_delta_e"])
    completion = leg.base._digested(
        {
            "schema": "paper_i_ra_adapt_disk_retry_completion_v1",
            "status": "passed",
            "source_round": SOURCE_ROUND,
            "target_round": TARGET_ROUND,
            "control_same_cutoff_delta_e": control_delta,
            "target_same_cutoff_delta_e": target_delta,
            "append_same_cutoff_delta_e": append_delta,
            "target_over_control_ratio": target_delta / control_delta,
            "target_over_append_ratio": target_delta / append_delta,
            "control_terminal_sha256": terminals["control"]["sha256"],
            "target_terminal_sha256": terminals["target"]["sha256"],
            "submission_authorized": False,
            "completed_at_utc": leg.base._utc_now(),
        }
    )
    leg.base._write_json(OUTPUT_ROOT / "completion_receipt.json", completion)
    return completion


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run-cell", choices=("control", "target"))
    action.add_argument("--finalize", action="store_true")
    parser.add_argument("--execution-authorized", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.preflight:
        if args.execution_authorized:
            raise leg.ContinuationContractError(
                "Preflight cannot carry execution authorization."
            )
        result = preflight()
    elif args.run_cell is not None:
        if not args.execution_authorized:
            raise leg.ContinuationContractError(
                "Disk retry requires --execution-authorized."
            )
        result = run_cell(args.run_cell)
    else:
        if args.execution_authorized:
            raise leg.ContinuationContractError(
                "Finalization does not carry execution authorization."
            )
        result = finalize()
    print(leg.base._canonical_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
