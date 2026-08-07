#!/usr/bin/env python3
"""Recover the authenticated r40 boundary and finish the target at r50."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LOCKED_SOURCE_ROOT = (
    REPO_ROOT
    / "output/local_runs"
    / "paper_i_ra_adapt_singleton_phase3_on_plateau_strong_weak_r50_"
    "storage_retry_local_20260802_v2"
    / "locked_runtime_source_inventory_1abcefba"
)
if str(LOCKED_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCKED_SOURCE_ROOT))
sys.dont_write_bytecode = True

from chtc.paper_i_ra_adapt_repair_20260727 import (  # noqa: E402
    run_local_singleton_phase3_on_plateau_strong_weak_resume_r20_20260802 as leg,
)
from chtc.paper_i_ra_adapt_repair_20260727 import (  # noqa: E402
    run_local_singleton_phase3_on_plateau_strong_weak_r30_retry_20260802 as repair,
)
from chtc.paper_i_ra_adapt_repair_20260727 import (  # noqa: E402
    run_local_singleton_phase3_on_plateau_strong_weak_resume_r50_20260802 as r50,
)


SOURCE_RUN_ROOT = r50.RUNS_ROOT / r50.CELL_IDS["target"]
OUTPUT_ROOT = (
    REPO_ROOT
    / "output/local_runs"
    / "paper_i_ra_adapt_singleton_phase3_on_plateau_strong_weak_r50_"
    "storage_retry_local_20260802_v2"
)
RESUME_SOURCE_ROOT = OUTPUT_ROOT / "resume_source"
RUN_ROOT = OUTPUT_ROOT / "run_no_checkpoint"
CHECKPOINT = RUN_ROOT / "checkpoint.json"
SOURCE_ROUND = 40
TARGET_ROUND = 50
EXACT_ENERGY = leg.EXACT_ENERGY
CHECKPOINT_CADENCE = TARGET_ROUND - SOURCE_ROUND


def _install() -> None:
    r50._install()
    leg.SOURCE_ROUND = SOURCE_ROUND
    leg.TARGET_ROUND = TARGET_ROUND
    leg._configure_base()
    repair.SOURCE_ROUND = SOURCE_ROUND
    repair.TARGET_ROUND = TARGET_ROUND


def _protocol() -> Any:
    _install()
    plan = leg.base._load_json(
        leg.base.MATERIALIZATION_ROOT / "materialization_plan.json"
    )
    locks = leg.base._load_json(
        leg.base.MATERIALIZATION_ROOT / "source_locks_snapshot.json"
    )
    validation = leg.base._load_json(
        leg.base.MATERIALIZATION_ROOT / "validation_report.json"
    )
    receipt = leg.base._load_json(
        leg.base.MATERIALIZATION_ROOT / "materialization_receipt.json"
    )
    for label, payload in (
        ("plan", plan),
        ("source locks", locks),
        ("validation", validation),
        ("materialization receipt", receipt),
    ):
        leg.base._verify_digest(payload, label=label)
    if (
        receipt["plan_sha256"] != plan["sha256"]
        or receipt["source_locks_sha256"] != locks["sha256"]
        or receipt["validation_sha256"] != validation["sha256"]
        or validation["status"] != "passed"
    ):
        raise leg.ContinuationContractError(
            "Locked materialization bindings drifted."
        )
    from pipelines.static_adapt.ra_adapt.bundles import (
        _implementation_source_inventory,
    )

    if (
        _implementation_source_inventory(LOCKED_SOURCE_ROOT)
        != locks["implementation_sources"]
    ):
        raise leg.ContinuationContractError(
            "Isolated implementation source inventory drifted."
        )
    for binding in plan["source_bindings"].values():
        leg.base._verify_binding(binding)

    original_validator = leg.base._validate_materialization
    leg.base._validate_materialization = lambda: (plan, locks)
    try:
        return leg.base._load_bound_protocol(r50.CELL_IDS["target"])
    finally:
        leg.base._validate_materialization = original_validator


def _source() -> dict[str, Any]:
    _install()
    protocol = _protocol()
    failure = leg._require_digest(
        SOURCE_RUN_ROOT / "failure_receipt.json",
        label="round-41 disk-publication failure",
    )
    manifest = leg._require_digest(
        SOURCE_RUN_ROOT / "run_manifest.json",
        label="round-50 continuation manifest",
    )
    authorization = leg._require_digest(
        SOURCE_RUN_ROOT / "execution_authorization.json",
        label="round-50 continuation authorization",
    )
    checkpoint = SOURCE_RUN_ROOT / "checkpoint.json"
    checkpoint_binding = leg._binding(checkpoint)
    sidecars = leg._checkpoint_sidecars(
        checkpoint,
        expected_depth=SOURCE_ROUND,
    )
    if (
        failure.get("status") != "failed"
        or failure.get("error_type") != "OSError"
        or "No space left on device" not in str(failure.get("error"))
        or failure.get("checkpoint_present") is not True
        or manifest.get("protocol_sha256") != protocol.sha256
        or manifest.get("execution_authorization_sha256")
        != authorization.get("sha256")
        or authorization.get("execution_authorized") is not True
        or int(manifest.get("source_round", -1)) != 30
        or int(manifest.get("target_round", -1)) != TARGET_ROUND
    ):
        raise leg.ContinuationContractError(
            "Round-40 disk-interrupted source binding drifted."
        )
    return {
        "cell_id": r50.CELL_IDS["target"],
        "run_root": SOURCE_RUN_ROOT,
        "protocol": protocol,
        "checkpoint": checkpoint,
        "checkpoint_binding": checkpoint_binding,
        "sidecars": sidecars,
        "failure": failure,
        "manifest": manifest,
        "authorization": authorization,
    }


def preflight() -> dict[str, Any]:
    source = _source()
    return leg.base._digested(
        {
            "schema": "paper_i_ra_adapt_r50_storage_retry_preflight_v1",
            "status": "passed",
            "source_round": SOURCE_ROUND,
            "target_round": TARGET_ROUND,
            "source_checkpoint": source["checkpoint_binding"],
            "source_checkpoint_sidecars": source["sidecars"],
            "source_failure_sha256": source["failure"]["sha256"],
            "protocol_sha256": source["protocol"].sha256,
            "checkpoint_cadence": CHECKPOINT_CADENCE,
            "checkpoint_cadence_scope": "observation_only_storage_safety_v1",
            "output_root": OUTPUT_ROOT.relative_to(REPO_ROOT).as_posix(),
            "output_root_absent": not OUTPUT_ROOT.exists(),
            "execution_authorized": False,
            "submission_authorized": False,
        }
    )


def prepare_source() -> dict[str, Any]:
    source = _source()
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    repaired = repair._materialize_ledger_closure_repair(
        source=source,
        destination_root=RESUME_SOURCE_ROOT,
    )
    receipt = leg.base._digested(
        {
            "schema": "paper_i_ra_adapt_r50_storage_retry_source_v1",
            "status": "passed",
            "source_round": SOURCE_ROUND,
            "target_round": TARGET_ROUND,
            "scientific_state_changed": False,
            "source_checkpoint": source["checkpoint_binding"],
            "source_checkpoint_sidecars": source["sidecars"],
            "source_failure_sha256": source["failure"]["sha256"],
            "repaired_checkpoint": repaired["checkpoint"],
            "repaired_checkpoint_sidecars": repaired["sidecars"],
            "repair_receipt": repaired["receipt"],
            "repair_receipt_sha256": repaired["receipt_sha256"],
            "canonical_resume_validation": "passed",
            "prepared_at_utc": leg.base._utc_now(),
        }
    )
    leg.base._write_json(OUTPUT_ROOT / "prepared_source_receipt.json", receipt)
    return receipt


def _prepared_source() -> dict[str, Any]:
    _install()
    receipt = leg._require_digest(
        OUTPUT_ROOT / "prepared_source_receipt.json",
        label="prepared round-40 resume source",
    )
    checkpoint = RESUME_SOURCE_ROOT / "checkpoint.json"
    binding = leg._binding(checkpoint, root=RESUME_SOURCE_ROOT)
    sidecars = leg._checkpoint_sidecars(
        checkpoint,
        expected_depth=SOURCE_ROUND,
    )
    if (
        receipt.get("status") != "passed"
        or receipt.get("scientific_state_changed") is not False
        or receipt.get("canonical_resume_validation") != "passed"
        or receipt.get("repaired_checkpoint") != binding
        or receipt.get("repaired_checkpoint_sidecars") != sidecars
    ):
        raise leg.ContinuationContractError(
            "Prepared round-40 resume source drifted."
        )
    return {
        "receipt": receipt,
        "checkpoint": checkpoint,
        "binding": binding,
        "sidecars": sidecars,
    }


def _independent_activation_validation(
    result_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild every plateau decision from accepted transition energies."""
    serialized = leg._strict_activation_validation(result_payload)
    receipts = result_payload["scientific_receipts"][
        "accepted_round_receipts"
    ]
    transitions = result_payload["run"]["accepted_transitions"]
    trajectory = result_payload["run"]["accepted_trajectory"]
    rows: list[dict[str, Any]] = []
    accepted_drops: list[float] = []
    cardinality_passed = bool(
        len(receipts) == len(transitions) == len(trajectory) == TARGET_ROUND
    )
    for index, (receipt, transition, state) in enumerate(
        zip(receipts, transitions, trajectory, strict=False),
        start=1,
    ):
        before = float(transition["energy_before"])
        after = float(transition["energy_after"])
        current_drop = before - after
        plateau = receipt["insertion_commutation_plateau"]
        activation = receipt["phase3_population_activation"]
        population = receipt["projected_phase3_population_receipt"]
        threshold = float(plateau["prior_mean_decrease_ratio_threshold"])

        prior_mean = None
        expected_ratio = None
        expected_open = False
        if len(accepted_drops) >= 2:
            prior_mean = sum(accepted_drops[:-1]) / len(
                accepted_drops[:-1]
            )
            expected_ratio = accepted_drops[-1] / prior_mean
            expected_open = expected_ratio < threshold

        serialized_ratio = plateau.get(
            "marginal_to_prior_mean_decrease_ratio"
        )
        ratio_passed = bool(
            (serialized_ratio is None and expected_ratio is None)
            or (
                serialized_ratio is not None
                and expected_ratio is not None
                and math.isclose(
                    float(serialized_ratio),
                    expected_ratio,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-15,
                )
            )
        )
        prior_mean_serialized = plateau.get("prior_mean_energy_decrease")
        prior_mean_passed = bool(
            (prior_mean_serialized is None and prior_mean is None)
            or (
                prior_mean_serialized is not None
                and prior_mean is not None
                and math.isclose(
                    float(prior_mean_serialized),
                    prior_mean,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-15,
                )
            )
        )
        trigger_passed = index == 1
        if index > 1:
            previous = transitions[index - 2]
            trigger_passed = bool(
                math.isclose(
                    float(plateau["trigger_energy_before"]),
                    float(previous["energy_before"]),
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                )
                and math.isclose(
                    float(plateau["trigger_energy_after"]),
                    float(previous["energy_after"]),
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                )
            )
        available_count = int(
            population["phase2_available_shortlist_count"]
        )
        expected_count = available_count if expected_open else 1
        passed = bool(
            int(receipt["accepted_round_ordinal"]) == index
            and int(transition["controller_round"]) == index
            and int(state["controller_round"]) == index
            and math.isclose(
                float(state["energy"]),
                after,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            and threshold == 1.0e-4
            and ratio_passed
            and prior_mean_passed
            and trigger_passed
            and bool(plateau["domain_open"]) is expected_open
            and bool(activation["competitive_population_live"])
            is expected_open
            and int(population["competitive_population_input_count"])
            == expected_count
            and int(population["phase3_evaluated_candidate_count"])
            == expected_count
        )
        rows.append(
            {
                "controller_round": index,
                "accepted_energy_drop": current_drop,
                "independent_prior_mean_energy_decrease": prior_mean,
                "independent_trigger_ratio": expected_ratio,
                "independent_domain_open": expected_open,
                "passed": passed,
            }
        )
        accepted_drops.append(current_drop)

    passed = bool(
        cardinality_passed
        and serialized.get("status") == "passed"
        and all(row["passed"] for row in rows)
    )
    return leg.base._digested(
        {
            "schema": (
                "paper_i_ra_adapt_r50_independent_phase3_activation_"
                "validation_v1"
            ),
            "status": "passed" if passed else "failed",
            "serialized_validation_sha256": serialized["sha256"],
            "cardinality_passed": cardinality_passed,
            "rounds": rows,
        }
    )


def run_target() -> dict[str, Any]:
    from pipelines.static_adapt.ra_adapt import (
        RAAdaptOperationalControls,
        run_ra_adapt,
    )
    from pipelines.static_adapt.sr_snake import (
        AcceptedStateResume,
        SRObservationPolicy,
    )

    _install()
    prepared = _prepared_source()
    protocol = _protocol()
    if RUN_ROOT.exists() or RUN_ROOT.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {RUN_ROOT}")
    RUN_ROOT.mkdir(parents=True, exist_ok=False)
    authorization = leg.base._digested(
        {
            "schema": "paper_i_ra_adapt_r50_storage_retry_authorization_v1",
            "protocol_sha256": protocol.sha256,
            "source_round": SOURCE_ROUND,
            "target_round": TARGET_ROUND,
            "prepared_source_sha256": prepared["receipt"]["sha256"],
            "authorization_source": "explicit_user_continuation_2026-08-02",
            "execution_authorized": True,
            "submission_authorized": False,
            "authorized_at_utc": leg.base._utc_now(),
        }
    )
    leg.base._write_json(RUN_ROOT / "execution_authorization.json", authorization)
    manifest = leg.base._digested(
        {
            "schema": "paper_i_ra_adapt_r50_storage_retry_run_v1",
            "status": "running",
            "protocol_sha256": protocol.sha256,
            "route_contract_sha256": protocol.route_contract["sha256"],
            "source_round": SOURCE_ROUND,
            "target_round": TARGET_ROUND,
            "source_checkpoint": prepared["binding"],
            "source_checkpoint_sidecars": prepared["sidecars"],
            "prepared_source_sha256": prepared["receipt"]["sha256"],
            "checkpoint_publication_enabled": False,
            "checkpoint_publication_scope": (
                "terminal_result_only_storage_safety_v1"
            ),
            "same_cutoff_exact_energy": EXACT_ENERGY,
            "execution_authorization_sha256": authorization["sha256"],
            "started_at_utc": leg.base._utc_now(),
        }
    )
    leg.base._write_json(RUN_ROOT / "run_manifest.json", manifest)
    try:
        result = run_ra_adapt(
            leg.base._problem_from_receipt(protocol.problem),
            protocol,
            operational_controls=RAAdaptOperationalControls(
                maximum_controller_rounds=TARGET_ROUND,
                resume=AcceptedStateResume(
                    checkpoint_path=prepared["checkpoint"],
                    checkpoint_sha256=prepared["binding"]["sha256"],
                ),
                observation=SRObservationPolicy(
                    checkpoint=None,
                    estimator_ledger=None,
                    resource_rounds=(TARGET_ROUND,),
                ),
            ),
        )
        payload = result.to_dict()
        leg.base._write_json(RUN_ROOT / "result.json", payload)
        if result.run.paper_i_summary is not None:
            leg.base._write_json(
                RUN_ROOT / "paper_i_summary.json",
                result.run.paper_i_summary.to_dict(),
            )
        source_trajectory = repair._trajectory_from_checkpoint(
            leg.base._load_json(prepared["checkpoint"])
        )
        resumed_trajectory = payload["run"]["accepted_trajectory"]
        prefix_passed = bool(
            len(resumed_trajectory) == TARGET_ROUND
            and resumed_trajectory[:SOURCE_ROUND] == source_trajectory
        )
        prefix = leg.base._digested(
            {
                "schema": "paper_i_ra_adapt_r50_storage_retry_prefix_v1",
                "status": "passed" if prefix_passed else "failed",
                "source_round": SOURCE_ROUND,
                "target_round": TARGET_ROUND,
                "checkpoint_trajectory_exact_prefix_match": prefix_passed,
            }
        )
        leg.base._write_json(RUN_ROOT / "prefix_validation.json", prefix)
        if not prefix_passed:
            raise leg.ContinuationContractError(
                "Storage retry changed the authenticated round-40 prefix."
            )
        activation = _independent_activation_validation(payload)
        leg.base._write_json(
            RUN_ROOT / "activation_validation.json", activation
        )
        if activation["status"] != "passed":
            raise leg.ContinuationContractError(
                "Storage retry Phase-III activation validation failed."
            )
        terminal = leg.base._digested(
            {
                "schema": "paper_i_ra_adapt_r50_storage_retry_terminal_v1",
                "status": "passed",
                "accepted_controller_rounds": TARGET_ROUND,
                "final_same_cutoff_delta_e": abs(
                    float(result.final_state.energy) - EXACT_ENERGY
                ),
                "protocol_sha256": protocol.sha256,
                "manifest_sha256": manifest["sha256"],
                "prepared_source_sha256": prepared["receipt"]["sha256"],
                "checkpoint_publication_enabled": False,
                "source_resume_checkpoint": prepared["binding"],
                "source_resume_checkpoint_sidecars": prepared["sidecars"],
                "result": leg._binding(
                    RUN_ROOT / "result.json", root=RUN_ROOT
                ),
                "paper_i_summary": leg._binding(
                    RUN_ROOT / "paper_i_summary.json", root=RUN_ROOT
                ),
                "prefix_validation_sha256": prefix["sha256"],
                "activation_validation_sha256": activation["sha256"],
                "completed_at_utc": leg.base._utc_now(),
            }
        )
        leg.base._write_json(RUN_ROOT / "terminal_receipt.json", terminal)
        return terminal
    except BaseException as exc:
        failure = leg.base._digested(
            {
                "schema": "paper_i_ra_adapt_r50_storage_retry_failure_v1",
                "status": "failed",
                "source_round": SOURCE_ROUND,
                "target_round": TARGET_ROUND,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "checkpoint_publication_enabled": False,
                "checkpoint_present": False,
                "failed_at_utc": leg.base._utc_now(),
            }
        )
        leg.base._write_json(RUN_ROOT / "failure_receipt.json", failure)
        raise


def finalize() -> dict[str, Any]:
    _install()
    terminal = leg._require_digest(
        RUN_ROOT / "terminal_receipt.json",
        label="round-50 storage-retry terminal",
    )
    activation = leg._require_digest(
        RUN_ROOT / "activation_validation.json",
        label="round-50 storage-retry activation validation",
    )
    prefix = leg._require_digest(
        RUN_ROOT / "prefix_validation.json",
        label="round-50 storage-retry prefix validation",
    )
    if (
        terminal.get("status") != "passed"
        or int(terminal.get("accepted_controller_rounds", -1))
        != TARGET_ROUND
        or activation.get("status") != "passed"
        or prefix.get("status") != "passed"
        or terminal.get("activation_validation_sha256")
        != activation.get("sha256")
        or terminal.get("prefix_validation_sha256")
        != prefix.get("sha256")
    ):
        raise leg.ContinuationContractError(
            "Round-50 storage retry is incomplete."
        )
    target_delta = float(terminal["final_same_cutoff_delta_e"])
    append_delta = leg._append_delta_e()
    completion = leg.base._digested(
        {
            "schema": "paper_i_ra_adapt_r50_storage_retry_completion_v1",
            "status": "passed",
            "source_round": SOURCE_ROUND,
            "target_round": TARGET_ROUND,
            "target_terminal_sha256": terminal["sha256"],
            "target_activation_validation_sha256": activation["sha256"],
            "target_same_cutoff_delta_e": target_delta,
            "append_same_cutoff_delta_e": append_delta,
            "target_over_append_ratio": target_delta / append_delta,
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
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--prepare-source", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--finalize", action="store_true")
    parser.add_argument("--execution-authorized", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.preflight:
        result = preflight()
    elif args.prepare_source:
        result = prepare_source()
    elif args.run:
        if not args.execution_authorized:
            raise leg.ContinuationContractError(
                "Scientific retry requires --execution-authorized."
            )
        result = run_target()
    else:
        result = finalize()
    print(leg.base._canonical_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
