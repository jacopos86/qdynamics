#!/usr/bin/env python3
"""Continue the source-matched strong--weak singleton pair from r10 to r20."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import tarfile
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
    run_local_singleton_phase3_on_plateau_strong_weak_r10_20260802 as base,
)


SOURCE_OUTPUT_ROOT = base.OUTPUT_ROOT
SOURCE_MATERIALIZATION_ROOT = base.MATERIALIZATION_ROOT
SOURCE_RUNS_ROOT = base.RUNS_ROOT
SOURCE_CELL_IDS = {
    "control": base.CONTROL_CELL_ID,
    "target": base.TARGET_CELL_ID,
}
OUTPUT_ROOT = (
    REPO_ROOT
    / "output/local_runs"
    / "paper_i_ra_adapt_singleton_phase3_on_plateau_strong_weak_r20_resume_"
    "local_20260802_v1"
)
MATERIALIZATION_ROOT = OUTPUT_ROOT / "materialization"
RUNS_ROOT = OUTPUT_ROOT / "runs"
BUNDLE_ID = (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_strong_weak_r20_resume_"
    "local_v1"
)
CELL_IDS = {
    "control": (
        "phase3_plateau_control_r20__strong_weak_u8__nph3__"
        "ra_singleton_plateau"
    ),
    "target": (
        "phase3_plateau_target_r20__strong_weak_u8__nph3__"
        "ra_singleton_plateau"
    ),
}
SOURCE_ROUND = 10
TARGET_ROUND = 20
EXACT_ENERGY = base.EXACT_ENERGY
APPEND_ARCHIVE = (
    REPO_ROOT
    / "chtc/paper_i_ra_adapt_repair_20260727"
    / "retrieved_chtc_20260731_append_r70_strong_weak"
    / "r70_fresh__strong_weak_u8__nph3__append_singleton__cluster_9398375__"
    "proc_5.tar.gz"
)
APPEND_RESULT_MEMBER = "worker_outputs/payload/result.json"

R10_TO_R20_ALLOWED_DIFFS = {
    ("bundle_id",),
    ("bundle_manifest_sha256",),
    ("bundle_materialization", "bundle_id"),
    ("bundle_materialization", "bundle_manifest_sha256"),
    ("bundle_materialization", "cell_id"),
    ("bundle_materialization", "sha256"),
    ("horizon",),
    ("request", "execution", "stop", "maximum_controller_rounds"),
    ("request", "observation", "checkpoint", "path"),
    ("request", "observation", "estimator_ledger", "path"),
    ("sha256",),
    ("stopping_rule", "maximum_controller_rounds"),
}
REQUIRED_HORIZON_DIFFS = {
    ("horizon",),
    ("request", "execution", "stop", "maximum_controller_rounds"),
    ("stopping_rule", "maximum_controller_rounds"),
}


class ContinuationContractError(RuntimeError):
    """Fail-closed continuation contract violation."""


def _configure_base() -> None:
    """Point the reusable materializer at this noncolliding r20 package."""

    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.MATERIALIZATION_ROOT = MATERIALIZATION_ROOT
    base.RUNS_ROOT = RUNS_ROOT
    base.BUNDLE_ID = BUNDLE_ID
    base.CONTROL_CELL_ID = CELL_IDS["control"]
    base.TARGET_CELL_ID = CELL_IDS["target"]
    base.CELLS = tuple(CELL_IDS.values())
    base.MAXIMUM_CONTROLLER_ROUNDS = TARGET_ROUND


def _binding(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ContinuationContractError(f"Unsafe or missing artifact: {path}")
    display = (
        resolved.as_posix()
        if root is None
        else resolved.relative_to(root.resolve()).as_posix()
    )
    return {
        "path": display,
        "sha256": base._sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _require_digest(path: Path, *, label: str) -> dict[str, Any]:
    value = base._load_json(path)
    base._verify_digest(value, label=label)
    return value


def _safe_child(root: Path, relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ContinuationContractError(f"{label} path is unavailable.")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ContinuationContractError(f"{label} path is unsafe: {relative}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ContinuationContractError(
            f"{label} path escapes its run root."
        ) from exc
    return resolved


def _checkpoint_sidecars(
    checkpoint_path: Path,
    *,
    expected_depth: int,
) -> dict[str, dict[str, Any]]:
    payload = base._load_json(checkpoint_path)
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ContinuationContractError("Checkpoint envelope is unavailable.")
    if (
        int(checkpoint.get("depth", -1)) != expected_depth
        or int(checkpoint.get("ansatz_depth", -1)) != expected_depth
        or checkpoint.get("reason") != "iteration_done"
    ):
        raise ContinuationContractError(
            f"Checkpoint is not a finalized round-{expected_depth} boundary."
        )
    ledger = checkpoint.get("estimator_call_ledger_checkpoint")
    if not isinstance(ledger, Mapping) or ledger.get("status") != "complete":
        raise ContinuationContractError("Checkpoint ledger is incomplete.")
    ledger_path = _safe_child(
        checkpoint_path.parent,
        ledger.get("path"),
        label="estimator ledger sidecar",
    )
    ledger_binding = _binding(ledger_path, root=checkpoint_path.parent)
    if ledger_binding["sha256"] != ledger.get("sha256"):
        raise ContinuationContractError("Checkpoint ledger sidecar drifted.")

    adapt_vqe = payload.get("adapt_vqe")
    resume = (
        None
        if not isinstance(adapt_vqe, Mapping)
        else adapt_vqe.get("verified_singleton_resume_sidecar")
    )
    if not isinstance(resume, Mapping) or resume.get("status") != "complete":
        raise ContinuationContractError(
            "Verified singleton-resume sidecar is unavailable."
        )
    resume_path = _safe_child(
        checkpoint_path.parent,
        resume.get("path"),
        label="verified resume sidecar",
    )
    resume_binding = _binding(resume_path, root=checkpoint_path.parent)
    if resume_binding["sha256"] != resume.get("sha256"):
        raise ContinuationContractError("Verified resume sidecar drifted.")
    return {
        "estimator_ledger_checkpoint": ledger_binding,
        "verified_singleton_resume": resume_binding,
    }


def _protocol_payload(root: Path, cell_id: str) -> dict[str, Any]:
    return base._load_json(root / "protocols" / f"{cell_id}.json")


def materialize() -> dict[str, Any]:
    _configure_base()
    materialization = base.materialize()
    projections: dict[str, Any] = {}
    for role in ("control", "target"):
        source = _protocol_payload(
            SOURCE_MATERIALIZATION_ROOT,
            SOURCE_CELL_IDS[role],
        )
        target = _protocol_payload(MATERIALIZATION_ROOT, CELL_IDS[role])
        differences = base._scalar_differences(source, target)
        paths = {row[0] for row in differences}
        unexpected = paths - R10_TO_R20_ALLOWED_DIFFS
        if unexpected or not REQUIRED_HORIZON_DIFFS.issubset(paths):
            raise ContinuationContractError(
                f"{role} r10-to-r20 projection drifted: "
                f"unexpected={sorted(unexpected, key=str)!r}; "
                f"paths={sorted(paths, key=str)!r}"
            )
        if (
            source["algorithm_id"] != target["algorithm_id"]
            or source["route_contract"]["sha256"]
            != target["route_contract"]["sha256"]
            or int(source["horizon"]) != SOURCE_ROUND
            or int(target["horizon"]) != TARGET_ROUND
        ):
            raise ContinuationContractError(
                f"{role} scientific identity changed across continuation."
            )
        projections[role] = [
            {"path": list(path), "before": before, "after": after}
            for path, before, after in differences
        ]
    projection = base._digested(
        {
            "schema": "paper_i_ra_adapt_r10_to_r20_projection_validation_v1",
            "status": "passed",
            "source_round": SOURCE_ROUND,
            "target_round": TARGET_ROUND,
            "projections": projections,
        }
    )
    base._write_json(
        MATERIALIZATION_ROOT / "continuation_projection_validation.json",
        projection,
    )
    plan = base._digested(
        {
            "schema": "paper_i_ra_adapt_phase3_plateau_continuation_plan_v1",
            "status": "passed",
            "source_output": SOURCE_OUTPUT_ROOT.relative_to(REPO_ROOT).as_posix(),
            "target_output": OUTPUT_ROOT.relative_to(REPO_ROOT).as_posix(),
            "source_round": SOURCE_ROUND,
            "target_round": TARGET_ROUND,
            "cells": copy.deepcopy(CELL_IDS),
            "source_cells": copy.deepcopy(SOURCE_CELL_IDS),
            "projection_validation_sha256": projection["sha256"],
            "runner": _binding(Path(__file__).resolve(), root=REPO_ROOT),
            "execution_authorized": False,
            "submission_authorized": False,
            "created_at_utc": base._utc_now(),
        }
    )
    base._write_json(MATERIALIZATION_ROOT / "continuation_plan.json", plan)
    return base._digested(
        {
            "schema": "paper_i_ra_adapt_phase3_plateau_continuation_materialization_v1",
            "status": "passed",
            "base_materialization_sha256": materialization["sha256"],
            "continuation_plan_sha256": plan["sha256"],
            "projection_validation_sha256": projection["sha256"],
            "execution_authorized": False,
            "submission_authorized": False,
        }
    )


def _validate_source(role: str) -> dict[str, Any]:
    from pipelines.static_adapt.ra_adapt.contracts import (
        resolved_ra_adapt_protocol_from_mapping,
    )

    _configure_base()
    source_cell = SOURCE_CELL_IDS[role]
    target_cell = CELL_IDS[role]
    source_root = SOURCE_RUNS_ROOT / source_cell
    source_protocol = resolved_ra_adapt_protocol_from_mapping(
        _protocol_payload(SOURCE_MATERIALIZATION_ROOT, source_cell)
    )
    target_protocol = base._load_bound_protocol(target_cell)
    terminal = _require_digest(
        source_root / "terminal_receipt.json",
        label=f"{role} source terminal",
    )
    manifest = _require_digest(
        source_root / "run_manifest.json",
        label=f"{role} source manifest",
    )
    authorization = _require_digest(
        source_root / "execution_authorization.json",
        label=f"{role} source authorization",
    )
    if (
        terminal.get("status") != "passed"
        or terminal.get("cell_id") != source_cell
        or int(terminal.get("accepted_controller_rounds", -1)) != SOURCE_ROUND
        or terminal.get("protocol_sha256") != source_protocol.sha256
        or manifest.get("protocol_sha256") != source_protocol.sha256
        or manifest.get("execution_authorization_sha256")
        != authorization.get("sha256")
        or target_protocol.route_contract["sha256"]
        != source_protocol.route_contract["sha256"]
        or target_protocol.problem != source_protocol.problem
    ):
        raise ContinuationContractError(f"{role} source binding drifted.")
    checkpoint = source_root / "checkpoint.json"
    result_path = source_root / "result.json"
    ledger = source_root / "estimator_ledger.json"
    bindings = {
        "checkpoint": _binding(checkpoint),
        "result": _binding(result_path),
        "estimator_ledger": _binding(ledger),
        "terminal": _binding(source_root / "terminal_receipt.json"),
    }
    for artifact, receipt_key in (
        ("checkpoint", "checkpoint_sha256"),
        ("result", "result_sha256"),
        ("estimator_ledger", "estimator_ledger_sha256"),
    ):
        if bindings[artifact]["sha256"] != terminal.get(receipt_key):
            raise ContinuationContractError(
                f"{role} source {artifact} drifted from its terminal receipt."
            )
    sidecars = _checkpoint_sidecars(
        checkpoint,
        expected_depth=SOURCE_ROUND,
    )
    source_result = base._load_json(result_path)
    source_run = source_result.get("run")
    if not isinstance(source_run, Mapping):
        raise ContinuationContractError(f"{role} source result has no run.")
    for key in (
        "accepted_trajectory",
        "accepted_transitions",
        "scientific_replay",
    ):
        if not isinstance(source_run.get(key), list) or len(source_run[key]) != 10:
            raise ContinuationContractError(f"{role} source {key} is incomplete.")
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


def preflight() -> dict[str, Any]:
    _configure_base()
    plan = _require_digest(
        MATERIALIZATION_ROOT / "continuation_plan.json",
        label="continuation plan",
    )
    projection = _require_digest(
        MATERIALIZATION_ROOT / "continuation_projection_validation.json",
        label="continuation projection",
    )
    sources = {role: _validate_source(role) for role in ("control", "target")}
    return base._digested(
        {
            "schema": "paper_i_ra_adapt_phase3_plateau_continuation_preflight_v1",
            "status": "passed",
            "source_round": SOURCE_ROUND,
            "target_round": TARGET_ROUND,
            "continuation_plan_sha256": plan["sha256"],
            "projection_validation_sha256": projection["sha256"],
            "sources": {
                role: {
                    "source_terminal_sha256": row["terminal"]["sha256"],
                    "source_checkpoint": row["bindings"]["checkpoint"],
                    "source_checkpoint_sidecars": row["sidecars"],
                    "target_protocol_sha256": row["target_protocol"].sha256,
                }
                for role, row in sources.items()
            },
            "execution_authorized": False,
            "submission_authorized": False,
        }
    )


def _prefix_validation(
    *,
    role: str,
    source_result: Mapping[str, Any],
    resumed_result: Mapping[str, Any],
    source_checkpoint_sha256: str,
) -> dict[str, Any]:
    def transition_operators(row: Mapping[str, Any]) -> list[Any]:
        values = row.get("selected_operators")
        return list(values) if isinstance(values, list) else [row.get("selected_operator")]

    def transition_indices(
        row: Mapping[str, Any], plural: str, singular: str
    ) -> list[Any]:
        values = row.get(plural)
        return list(values) if isinstance(values, list) else [row.get(singular)]

    source_run = source_result.get("run")
    resumed_run = resumed_result.get("run")
    if not isinstance(source_run, Mapping) or not isinstance(
        resumed_run, Mapping
    ):
        raise ContinuationContractError("Result run payload is absent.")
    source_trajectory = source_run.get("accepted_trajectory")
    resumed_trajectory = resumed_run.get("accepted_trajectory")
    fields: dict[str, bool] = {
        "accepted_trajectory": bool(
            isinstance(source_trajectory, list)
            and isinstance(resumed_trajectory, list)
            and len(source_trajectory) == SOURCE_ROUND
            and len(resumed_trajectory) == TARGET_ROUND
            and resumed_trajectory[:SOURCE_ROUND] == source_trajectory
        )
    }
    source_receipts = source_result.get("scientific_receipts", {}).get(
        "accepted_round_receipts"
    )
    resumed_receipts = resumed_result.get("scientific_receipts", {}).get(
        "accepted_round_receipts"
    )
    fields["accepted_round_receipts"] = bool(
        isinstance(source_receipts, list)
        and isinstance(resumed_receipts, list)
        and len(source_receipts) == SOURCE_ROUND
        and len(resumed_receipts) == TARGET_ROUND
        and resumed_receipts[:SOURCE_ROUND] == source_receipts
    )
    source_transitions = source_run.get("accepted_transitions")
    resumed_transitions = resumed_run.get("accepted_transitions")
    fields["resume_transition_projection"] = bool(
        isinstance(source_transitions, list)
        and isinstance(resumed_transitions, list)
        and isinstance(source_trajectory, list)
        and len(resumed_transitions) == TARGET_ROUND
        and all(
            row.get("controller_round") == index
            and row.get("accepted_state") == source_trajectory[index - 1]
            and row.get("energy_after")
            == source_transitions[index - 1].get("energy_after")
            and row.get("energy_before")
            == source_transitions[index - 1].get("energy_before")
            and row.get("cumulative_s_alg")
            == source_transitions[index - 1].get("cumulative_s_alg")
            and row.get("route_family") == "ra_adapt"
            and row.get("selected_operators")
            == transition_operators(source_transitions[index - 1])
            and row.get("selected_pool_indices")
            == transition_indices(
                source_transitions[index - 1],
                "selected_pool_indices",
                "pool_index",
            )
            and row.get("selected_positions")
            == transition_indices(
                source_transitions[index - 1],
                "selected_positions",
                "insertion_position",
            )
            and row.get("source_checkpoint_sha256")
            == source_checkpoint_sha256
            for index, row in enumerate(
                resumed_transitions[:SOURCE_ROUND], start=1
            )
        )
    )
    source_replay = source_run.get("scientific_replay")
    resumed_replay = resumed_run.get("scientific_replay")
    fields["resume_scientific_replay_projection"] = bool(
        isinstance(source_replay, list)
        and isinstance(resumed_replay, list)
        and isinstance(source_trajectory, list)
        and len(resumed_replay) == TARGET_ROUND
        and all(
            row.get("controller_round") == index
            and row.get("accepted_state") == source_trajectory[index - 1]
            and row.get("accepted_refit")
            == source_replay[index - 1].get("accepted_refit")
            and row.get("checkpoint")
            == source_replay[index - 1].get("checkpoint")
            and row.get("phase") == source_replay[index - 1].get("phase")
            and row.get("trust_solve")
            == source_replay[index - 1].get("trust_solve")
            and row.get("selected_operators")
            == transition_operators(source_replay[index - 1])
            and row.get("source_checkpoint_sha256")
            == source_checkpoint_sha256
            for index, row in enumerate(resumed_replay[:SOURCE_ROUND], start=1)
        )
    )
    validation = base._digested(
        {
            "schema": "paper_i_ra_adapt_continuation_prefix_validation_v1",
            "status": "passed" if all(fields.values()) else "failed",
            "role": role,
            "source_round": SOURCE_ROUND,
            "target_round": TARGET_ROUND,
            "exact_prefix_fields": fields,
            "resume_wrapper_note": (
                "Accepted-state hydration serializes authenticated prefix "
                "transitions/replay in reconstruction wrappers; their exact "
                "scientific projections are validated instead of requiring "
                "the fresh-run wrapper shape."
            ),
        }
    )
    return validation


def _strict_activation_validation(
    result_payload: Mapping[str, Any],
) -> dict[str, Any]:
    receipts = result_payload["scientific_receipts"][
        "accepted_round_receipts"
    ]
    rows: list[dict[str, Any]] = []
    for expected_round, receipt in enumerate(receipts, start=1):
        plateau = receipt["insertion_commutation_plateau"]
        activation = receipt["phase3_population_activation"]
        population = receipt["projected_phase3_population_receipt"]
        domain_open = bool(plateau["domain_open"])
        available_count = int(population["phase2_available_shortlist_count"])
        input_count = int(population["competitive_population_input_count"])
        evaluated_count = int(population["phase3_evaluated_candidate_count"])
        ratio = plateau.get("marginal_to_prior_mean_decrease_ratio")
        recomputed_ratio = None
        ratio_passed = ratio is None
        if ratio is not None:
            trigger_decrease = float(plateau["trigger_energy_before"]) - float(
                plateau["trigger_energy_after"]
            )
            prior_mean = float(plateau["prior_mean_energy_decrease"])
            recomputed_ratio = trigger_decrease / prior_mean
            ratio_passed = bool(
                math.isclose(
                    float(ratio),
                    recomputed_ratio,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-15,
                )
                and domain_open
                is (
                    recomputed_ratio
                    < float(plateau["prior_mean_decrease_ratio_threshold"])
                )
            )
        expected_authority = (
            None if domain_open else "phase2_raw_score_top_rank_v1"
        )
        expected_winner_policy = (
            None if domain_open else "phase2_winner_only_refit_geometry_v1"
        )
        expected_count = available_count if domain_open else 1
        passed = bool(
            int(receipt["accepted_round_ordinal"]) == expected_round
            and activation.get("schema")
            == "ra_phase3_population_activation_receipt_v1"
            and activation.get("policy")
            == "same_round_insertion_plateau_predicate_v1"
            and activation.get("activation_source")
            == "same_round_authenticated_insertion_plateau_domain_open_v1"
            and activation.get("hysteresis_active") is False
            and activation.get("independent_latch_active") is False
            and bool(activation.get("competitive_population_live"))
            is domain_open
            and bool(activation.get("insertion_plateau_domain_open"))
            is domain_open
            and activation.get("preplateau_admission_authority")
            == expected_authority
            and activation.get("winner_materialization_policy")
            == expected_winner_policy
            and population.get("competitive_population_activation")
            == activation
            and input_count == expected_count
            and evaluated_count == expected_count
            and available_count >= 1
            and int(population["phase1_retained_parent_count"]) >= 1
            and plateau.get("hysteresis_active") is False
            and plateau.get("exact_reference_used") is False
            and ratio_passed
        )
        rows.append(
            {
                "controller_round": expected_round,
                "insertion_plateau_domain_open": domain_open,
                "competitive_phase3_population_live": bool(
                    activation["competitive_population_live"]
                ),
                "competitive_population_input_count": input_count,
                "phase2_available_shortlist_count": available_count,
                "phase3_evaluated_candidate_count": evaluated_count,
                "serialized_trigger_ratio": ratio,
                "recomputed_trigger_ratio": recomputed_ratio,
                "passed": passed,
            }
        )
    return base._digested(
        {
            "schema": "paper_i_ra_adapt_phase3_activation_validation_v2",
            "status": (
                "passed"
                if len(rows) == TARGET_ROUND
                and all(row["passed"] for row in rows)
                else "failed"
            ),
            "rounds": rows,
            "first_open_round": next(
                (
                    row["controller_round"]
                    for row in rows
                    if row["insertion_plateau_domain_open"]
                ),
                None,
            ),
            "continuation_open_rounds": [
                row["controller_round"]
                for row in rows[SOURCE_ROUND:]
                if row["insertion_plateau_domain_open"]
            ],
        }
    )


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

    source = _validate_source(role)
    run_root = RUNS_ROOT / CELL_IDS[role]
    if run_root.exists() or run_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    protocol = source["target_protocol"]
    authorization = base._digested(
        {
            "schema": "paper_i_ra_adapt_local_continuation_authorization_v1",
            "role": role,
            "cell_id": CELL_IDS[role],
            "protocol_sha256": protocol.sha256,
            "source_terminal_sha256": source["terminal"]["sha256"],
            "source_checkpoint_sha256": source["bindings"]["checkpoint"][
                "sha256"
            ],
            "source_round": SOURCE_ROUND,
            "target_round": TARGET_ROUND,
            "authorization_source": "explicit_user_request_2026-08-02",
            "execution_authorized": True,
            "submission_authorized": False,
            "authorized_at_utc": base._utc_now(),
        }
    )
    base._write_json(run_root / "execution_authorization.json", authorization)
    checkpoint = run_root / "checkpoint.json"
    manifest = base._digested(
        {
            "schema": "paper_i_ra_adapt_phase3_plateau_continuation_run_v1",
            "run_class": "diagnostic_continuation",
            "role": role,
            "cell_id": CELL_IDS[role],
            "protocol_sha256": protocol.sha256,
            "route_contract_sha256": protocol.route_contract["sha256"],
            "candidate_representation": protocol.candidate_representation,
            "active_gradient_policy": protocol.active_gradient_policy,
            "resource_weighting_scope": protocol.resource_weighting_scope,
            "optimizer": protocol.optimizer,
            "optimizer_maxiter": protocol.optimizer_maxiter,
            "adapt_seed": protocol.seeds["adapt"],
            "source_round": SOURCE_ROUND,
            "target_round": TARGET_ROUND,
            "source_checkpoint": source["bindings"]["checkpoint"],
            "source_checkpoint_sidecars": source["sidecars"],
            "same_cutoff_exact_energy": EXACT_ENERGY,
            "execution_authorization_sha256": authorization["sha256"],
            "started_at_utc": base._utc_now(),
        }
    )
    base._write_json(run_root / "run_manifest.json", manifest)
    try:
        controls = RAAdaptOperationalControls(
            maximum_controller_rounds=TARGET_ROUND,
            resume=AcceptedStateResume(
                checkpoint_path=source["source_root"] / "checkpoint.json",
                checkpoint_sha256=source["bindings"]["checkpoint"]["sha256"],
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
        base._write_json(run_root / "result.json", payload)
        if result.run.paper_i_summary is not None:
            base._write_json(
                run_root / "paper_i_summary.json",
                result.run.paper_i_summary.to_dict(),
            )
        prefix = _prefix_validation(
            role=role,
            source_result=source["result"],
            resumed_result=payload,
            source_checkpoint_sha256=source["bindings"]["checkpoint"][
                "sha256"
            ],
        )
        base._write_json(run_root / "prefix_validation.json", prefix)
        if prefix["status"] != "passed":
            raise ContinuationContractError(
                f"{role} continuation changed its authenticated r10 prefix."
            )
        sidecars = _checkpoint_sidecars(
            checkpoint,
            expected_depth=TARGET_ROUND,
        )
        activation = None
        if role == "target":
            activation = _strict_activation_validation(payload)
            if len(activation["rounds"]) != TARGET_ROUND:
                raise ContinuationContractError(
                    "Target activation receipt horizon is incomplete."
                )
            base._write_json(run_root / "activation_validation.json", activation)
            if activation["status"] != "passed":
                raise ContinuationContractError(
                    "Target Phase-III activation receipts failed."
                )
        delta_e = abs(float(result.final_state.energy) - EXACT_ENERGY)
        terminal = base._digested(
            {
                "schema": "paper_i_ra_adapt_phase3_plateau_continuation_terminal_v1",
                "status": "passed",
                "role": role,
                "cell_id": CELL_IDS[role],
                "source_round": SOURCE_ROUND,
                "accepted_controller_rounds": TARGET_ROUND,
                "final_same_cutoff_delta_e": delta_e,
                "protocol_sha256": protocol.sha256,
                "manifest_sha256": manifest["sha256"],
                "source_terminal_sha256": source["terminal"]["sha256"],
                "source_checkpoint_sha256": source["bindings"]["checkpoint"][
                    "sha256"
                ],
                "checkpoint": _binding(checkpoint, root=run_root),
                "checkpoint_sidecars": sidecars,
                "result": _binding(run_root / "result.json", root=run_root),
                "prefix_validation_sha256": prefix["sha256"],
                "activation_validation_sha256": (
                    None if activation is None else activation["sha256"]
                ),
                "completed_at_utc": base._utc_now(),
            }
        )
        base._write_json(run_root / "terminal_receipt.json", terminal)
        return terminal
    except BaseException as exc:
        failure = base._digested(
            {
                "schema": "paper_i_ra_adapt_phase3_plateau_continuation_failure_v1",
                "status": "failed",
                "role": role,
                "cell_id": CELL_IDS[role],
                "source_round": SOURCE_ROUND,
                "target_round": TARGET_ROUND,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "checkpoint_present": checkpoint.is_file(),
                "failed_at_utc": base._utc_now(),
            }
        )
        base._write_json(run_root / "failure_receipt.json", failure)
        raise


def recover_completed_cell(role: str) -> dict[str, Any]:
    """Finalize science completed before the overstrict v1 wrapper check."""

    source = _validate_source(role)
    run_root = RUNS_ROOT / CELL_IDS[role]
    failure = _require_digest(
        run_root / "failure_receipt.json",
        label=f"{role} known post-run validation failure",
    )
    if (
        failure.get("status") != "failed"
        or failure.get("error_type") != "ContinuationContractError"
        or failure.get("error")
        != f"{role} continuation changed its authenticated r10 prefix."
        or failure.get("checkpoint_present") is not True
    ):
        raise ContinuationContractError(
            f"{role} is not the known completed-science wrapper failure."
        )
    authorization = _require_digest(
        run_root / "execution_authorization.json",
        label=f"{role} continuation authorization",
    )
    manifest = _require_digest(
        run_root / "run_manifest.json",
        label=f"{role} continuation manifest",
    )
    if (
        authorization.get("execution_authorized") is not True
        or int(authorization.get("source_round", -1)) != SOURCE_ROUND
        or int(authorization.get("target_round", -1)) != TARGET_ROUND
        or manifest.get("execution_authorization_sha256")
        != authorization.get("sha256")
        or manifest.get("protocol_sha256")
        != source["target_protocol"].sha256
    ):
        raise ContinuationContractError(
            f"{role} completed-science provenance drifted."
        )
    result_path = run_root / "result.json"
    checkpoint = run_root / "checkpoint.json"
    summary_path = run_root / "paper_i_summary.json"
    payload = base._load_json(result_path)
    prefix = _prefix_validation(
        role=role,
        source_result=source["result"],
        resumed_result=payload,
        source_checkpoint_sha256=source["bindings"]["checkpoint"]["sha256"],
    )
    if prefix["status"] != "passed":
        raise ContinuationContractError(
            f"{role} recovered prefix projection failed."
        )
    prefix_path = run_root / "prefix_validation_v2.json"
    if prefix_path.exists() or prefix_path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {prefix_path}")
    base._write_json(prefix_path, prefix)
    sidecars = _checkpoint_sidecars(checkpoint, expected_depth=TARGET_ROUND)
    activation = None
    if role == "target":
        activation = _strict_activation_validation(payload)
        if activation["status"] != "passed":
            raise ContinuationContractError(
                "Recovered target Phase-III activation validation failed."
            )
        activation_path = run_root / "activation_validation.json"
        if activation_path.exists() or activation_path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {activation_path}")
        base._write_json(activation_path, activation)
    run = payload.get("run")
    if not isinstance(run, Mapping):
        raise ContinuationContractError("Recovered result has no run payload.")
    final_state = run.get("final_state")
    if not isinstance(final_state, Mapping):
        raise ContinuationContractError("Recovered result has no final state.")
    delta_e = abs(float(final_state["energy"]) - EXACT_ENERGY)
    terminal = base._digested(
        {
            "schema": (
                "paper_i_ra_adapt_phase3_plateau_recovered_terminal_v1"
            ),
            "status": "passed",
            "role": role,
            "cell_id": CELL_IDS[role],
            "source_round": SOURCE_ROUND,
            "accepted_controller_rounds": TARGET_ROUND,
            "final_same_cutoff_delta_e": delta_e,
            "protocol_sha256": source["target_protocol"].sha256,
            "manifest_sha256": manifest["sha256"],
            "source_terminal_sha256": source["terminal"]["sha256"],
            "source_checkpoint_sha256": source["bindings"]["checkpoint"][
                "sha256"
            ],
            "checkpoint": _binding(checkpoint, root=run_root),
            "checkpoint_sidecars": sidecars,
            "result": _binding(result_path, root=run_root),
            "paper_i_summary": _binding(summary_path, root=run_root),
            "prefix_validation_sha256": prefix["sha256"],
            "activation_validation_sha256": (
                None if activation is None else activation["sha256"]
            ),
            "known_post_run_wrapper_failure": _binding(
                run_root / "failure_receipt.json", root=run_root
            ),
            "scientific_rounds_rerun": False,
            "recovery_scope": (
                "post_science_overstrict_resume_wrapper_shape_check_v1"
            ),
            "completed_at_utc": base._utc_now(),
        }
    )
    terminal_path = run_root / "terminal_receipt.json"
    if terminal_path.exists() or terminal_path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {terminal_path}")
    base._write_json(terminal_path, terminal)
    return terminal


def _append_delta_e() -> float:
    if not APPEND_ARCHIVE.is_file() or APPEND_ARCHIVE.is_symlink():
        raise ContinuationContractError("Append comparator archive is absent.")
    with tarfile.open(APPEND_ARCHIVE, mode="r:gz") as archive:
        stream = archive.extractfile(APPEND_RESULT_MEMBER)
        if stream is None:
            raise ContinuationContractError("Append result member is absent.")
        payload = json.load(stream)
    history = payload["result_payload"]["history"]
    if len(history) < TARGET_ROUND:
        raise ContinuationContractError("Append comparator has no round 20.")
    energy = float(history[TARGET_ROUND - 1]["accepted_refit"]["final_energy"])
    return abs(energy - EXACT_ENERGY)


def finalize() -> dict[str, Any]:
    terminals: dict[str, dict[str, Any]] = {}
    for role in ("control", "target"):
        terminal = _require_digest(
            RUNS_ROOT / CELL_IDS[role] / "terminal_receipt.json",
            label=f"{role} continuation terminal",
        )
        if (
            terminal.get("status") != "passed"
            or int(terminal.get("accepted_controller_rounds", -1))
            != TARGET_ROUND
        ):
            raise ContinuationContractError(f"{role} continuation is incomplete.")
        terminals[role] = terminal
    target_activation = _require_digest(
        RUNS_ROOT / CELL_IDS["target"] / "activation_validation.json",
        label="target activation validation",
    )
    if target_activation.get("status") != "passed":
        raise ContinuationContractError("Target activation validation failed.")
    control_delta = float(terminals["control"]["final_same_cutoff_delta_e"])
    target_delta = float(terminals["target"]["final_same_cutoff_delta_e"])
    append_delta = _append_delta_e()
    completion = base._digested(
        {
            "schema": "paper_i_ra_adapt_phase3_plateau_continuation_completion_v1",
            "status": "passed",
            "source_round": SOURCE_ROUND,
            "target_round": TARGET_ROUND,
            "control_terminal_sha256": terminals["control"]["sha256"],
            "target_terminal_sha256": terminals["target"]["sha256"],
            "target_activation_validation_sha256": target_activation["sha256"],
            "control_same_cutoff_delta_e": control_delta,
            "target_same_cutoff_delta_e": target_delta,
            "append_same_cutoff_delta_e": append_delta,
            "target_minus_control_delta_e": target_delta - control_delta,
            "target_over_control_ratio": target_delta / control_delta,
            "target_over_append_ratio": target_delta / append_delta,
            "target_open_rounds": [
                int(row["controller_round"])
                for row in target_activation["rounds"]
                if row["insertion_plateau_domain_open"]
            ],
            "execution_authorized": True,
            "submission_authorized": False,
            "completed_at_utc": base._utc_now(),
        }
    )
    path = OUTPUT_ROOT / "completion_receipt.json"
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {path}")
    base._write_json(path, completion)
    return completion


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--materialize", action="store_true")
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run-cell", choices=("control", "target"))
    action.add_argument("--recover-cell", choices=("control", "target"))
    action.add_argument("--finalize", action="store_true")
    parser.add_argument("--execution-authorized", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.materialize:
        if args.execution_authorized:
            raise ContinuationContractError(
                "Materialization cannot carry execution authorization."
            )
        result = materialize()
    elif args.preflight:
        if args.execution_authorized:
            raise ContinuationContractError(
                "Preflight cannot carry execution authorization."
            )
        result = preflight()
    elif args.run_cell is not None:
        if not args.execution_authorized:
            raise ContinuationContractError(
                "Scientific continuation requires --execution-authorized."
            )
        result = run_cell(args.run_cell)
    elif args.recover_cell is not None:
        if args.execution_authorized:
            raise ContinuationContractError(
                "Completed-science recovery does not carry execution "
                "authorization."
            )
        result = recover_completed_cell(args.recover_cell)
    else:
        if args.execution_authorized:
            raise ContinuationContractError(
                "Finalization does not carry execution authorization."
            )
        result = finalize()
    print(base._canonical_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
