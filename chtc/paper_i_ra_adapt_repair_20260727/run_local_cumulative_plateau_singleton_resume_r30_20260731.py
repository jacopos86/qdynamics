#!/usr/bin/env python3
"""Resume the strong--strong cumulative-plateau singleton from round 20 to 30."""

from __future__ import annotations

import argparse
import copy
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


CELL_ID = "core__strong_strong_u8__nph7__ra_singleton_plateau"
SOURCE_RUN_ROOT = base.RUNS_ROOT / CELL_ID
OUTPUT_ROOT = (
    REPO_ROOT
    / "output/local_runs"
    / "paper_i_ra_adapt_cumulative_plateau_singleton_r30_resume_local_20260731_v4"
)
SOURCE_ROUND = 20
TARGET_ROUND = 30
PLATEAU_RATIO = 1.0e-4
COMPLETED_ROUND30_ROOT = (
    REPO_ROOT
    / "output/local_runs"
    / "paper_i_ra_adapt_cumulative_plateau_singleton_r30_resume_local_20260731_v4"
)
FINALIZED_ROUND30_ROOT = (
    REPO_ROOT
    / "output/local_runs"
    / "paper_i_ra_adapt_cumulative_plateau_singleton_r30_finalized_local_20260731_v5"
)


class ContinuationContractError(RuntimeError):
    """Fail-closed continuation contract violation."""


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
        raise ContinuationContractError("Checkpoint ledger is not complete.")
    ledger_path = _safe_child(
        checkpoint_path.parent,
        ledger.get("path"),
        label="estimator ledger sidecar",
    )
    ledger_binding = _binding(ledger_path, root=checkpoint_path.parent)
    if ledger_binding["sha256"] != ledger.get("sha256"):
        raise ContinuationContractError("Checkpoint ledger sidecar drifted.")

    adapt_vqe = payload.get("adapt_vqe")
    if not isinstance(adapt_vqe, Mapping):
        raise ContinuationContractError("Checkpoint ADAPT payload is absent.")
    resume = adapt_vqe.get("verified_singleton_resume_sidecar")
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


def _repair_active_prefix_owner_occurrences(
    payload: Mapping[str, Any],
    *,
    expected_depth: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replace label-registry owner echoes with admission-occurrence owners."""

    repaired = copy.deepcopy(dict(payload))
    adapt = repaired.get("adapt_vqe")
    if not isinstance(adapt, dict):
        raise ContinuationContractError("Repair source has no ADAPT payload.")
    history_raw = adapt.get("history")
    if not isinstance(history_raw, list) or len(history_raw) != expected_depth:
        raise ContinuationContractError("Repair source history is incomplete.")
    occurrences: list[dict[str, Any]] = []
    repaired_history: list[dict[str, Any]] = []
    repaired_prefixes: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    for round_index, row_raw in enumerate(history_raw, start=1):
        if not isinstance(row_raw, Mapping):
            raise ContinuationContractError("Repair history row is invalid.")
        row = copy.deepcopy(dict(row_raw))
        labels = row.get("selected_ops")
        positions = row.get("selected_effective_positions")
        features = row.get("selected_feature_rows")
        if (
            not isinstance(labels, list)
            or not isinstance(positions, list)
            or not isinstance(features, list)
            or not (len(labels) == len(positions) == len(features))
            or not labels
        ):
            raise ContinuationContractError(
                f"Round {round_index} admission rows are incomplete."
            )
        for label, position_raw, feature_raw in zip(
            labels, positions, features, strict=True
        ):
            if not isinstance(feature_raw, Mapping):
                raise ContinuationContractError(
                    f"Round {round_index} feature row is invalid."
                )
            metadata = feature_raw.get("generator_metadata")
            if not isinstance(metadata, Mapping):
                raise ContinuationContractError(
                    f"Round {round_index} generator metadata is absent."
                )
            owner = metadata.get("ra_retained_parent_owner")
            if not isinstance(owner, Mapping):
                raise ContinuationContractError(
                    f"Round {round_index} retained-parent owner is absent."
                )
            position = int(position_raw)
            if position < 0 or position > len(occurrences):
                raise ContinuationContractError(
                    f"Round {round_index} insertion position is invalid."
                )
            occurrences.insert(
                position,
                {
                    "label": str(label),
                    "generator_id": str(feature_raw.get("generator_id")),
                    "parent_generator_id": str(
                        owner.get("parent_generator_identity")
                    ),
                    "ra_retained_parent_owner": copy.deepcopy(dict(owner)),
                },
            )
        prune = row.get("post_admission_prune")
        if isinstance(prune, Mapping) and int(prune.get("accepted_count", 0)):
            deleted = prune.get("deleted_indices")
            if not isinstance(deleted, list) or len(deleted) != 1:
                raise ContinuationContractError(
                    f"Round {round_index} prune receipt is unsupported."
                )
            del occurrences[int(deleted[0])]

        prefix_raw = row.get("active_prefix_checkpoint")
        if not isinstance(prefix_raw, Mapping):
            raise ContinuationContractError(
                f"Round {round_index} active prefix is absent."
            )
        prefix = copy.deepcopy(dict(prefix_raw))
        operator_rows = prefix.get("ordered_active_operators")
        if not isinstance(operator_rows, list) or len(operator_rows) != len(
            occurrences
        ):
            raise ContinuationContractError(
                f"Round {round_index} active operator rows are incomplete."
            )
        old_prefix_sha = str(prefix.get("checkpoint_sha256", ""))
        changed_positions: list[dict[str, Any]] = []
        for position, (operator, occurrence) in enumerate(
            zip(operator_rows, occurrences, strict=True)
        ):
            if (
                not isinstance(operator, dict)
                or operator.get("label") != occurrence["label"]
                or operator.get("generator_id") != occurrence["generator_id"]
            ):
                raise ContinuationContractError(
                    f"Round {round_index} operator occurrence order drifted."
                )
            previous_owner = operator.get("ra_retained_parent_owner")
            required_owner = occurrence["ra_retained_parent_owner"]
            if previous_owner != required_owner:
                changed_positions.append(
                    {
                        "active_position": position,
                        "generator_id": occurrence["generator_id"],
                        "old_owner_sha256": (
                            None
                            if not isinstance(previous_owner, Mapping)
                            else previous_owner.get("sha256")
                        ),
                        "new_owner_sha256": required_owner.get("sha256"),
                    }
                )
            operator["parent_generator_id"] = occurrence[
                "parent_generator_id"
            ]
            operator["ra_retained_parent_owner"] = copy.deepcopy(
                required_owner
            )
        unsigned_prefix = dict(prefix)
        unsigned_prefix.pop("checkpoint_sha256", None)
        prefix["checkpoint_sha256"] = base._canonical_sha256(unsigned_prefix)
        if changed_positions:
            changes.append(
                {
                    "controller_round": round_index,
                    "checkpoint_kind": "post_admission_prune",
                    "old_prefix_sha256": old_prefix_sha,
                    "new_prefix_sha256": prefix["checkpoint_sha256"],
                    "changed_positions": changed_positions,
                }
            )
        row["active_prefix_checkpoint"] = prefix
        repaired_history.append(row)
        repaired_prefixes.append(copy.deepcopy(prefix))

    adapt["history"] = repaired_history
    adapt["history_tail"] = copy.deepcopy(repaired_history)
    adapt["active_prefix_checkpoints"] = copy.deepcopy(repaired_prefixes)
    terminal_raw = adapt.get("terminal_active_prefix_checkpoint")
    if not isinstance(terminal_raw, Mapping):
        raise ContinuationContractError("Terminal active prefix is absent.")
    terminal = copy.deepcopy(dict(terminal_raw))
    terminal_operator_rows = terminal.get("ordered_active_operators")
    if not isinstance(terminal_operator_rows, list) or len(
        terminal_operator_rows
    ) != len(occurrences):
        raise ContinuationContractError(
            "Terminal active operator rows are incomplete."
        )
    terminal_changed_positions: list[dict[str, Any]] = []
    terminal_old_sha = str(terminal.get("checkpoint_sha256", ""))
    for position, (operator, occurrence) in enumerate(
        zip(terminal_operator_rows, occurrences, strict=True)
    ):
        if (
            not isinstance(operator, dict)
            or operator.get("label") != occurrence["label"]
            or operator.get("generator_id") != occurrence["generator_id"]
        ):
            raise ContinuationContractError(
                "Terminal operator occurrence order drifted."
            )
        previous_owner = operator.get("ra_retained_parent_owner")
        required_owner = occurrence["ra_retained_parent_owner"]
        if previous_owner != required_owner:
            terminal_changed_positions.append(
                {
                    "active_position": position,
                    "generator_id": occurrence["generator_id"],
                    "old_owner_sha256": (
                        None
                        if not isinstance(previous_owner, Mapping)
                        else previous_owner.get("sha256")
                    ),
                    "new_owner_sha256": required_owner.get("sha256"),
                }
            )
        operator["parent_generator_id"] = occurrence["parent_generator_id"]
        operator["ra_retained_parent_owner"] = copy.deepcopy(required_owner)
    terminal_unsigned = dict(terminal)
    terminal_unsigned.pop("checkpoint_sha256", None)
    terminal["checkpoint_sha256"] = base._canonical_sha256(
        terminal_unsigned
    )
    if terminal_changed_positions:
        changes.append(
            {
                "controller_round": expected_depth,
                "checkpoint_kind": "terminal_post_final_refit_and_prune",
                "old_prefix_sha256": terminal_old_sha,
                "new_prefix_sha256": terminal["checkpoint_sha256"],
                "changed_positions": terminal_changed_positions,
            }
        )
    adapt["terminal_active_prefix_checkpoint"] = terminal
    continuation = adapt.get("continuation")
    if not isinstance(continuation, dict):
        raise ContinuationContractError("Continuation payload is absent.")
    continuation["active_prefix_checkpoints"] = copy.deepcopy(
        repaired_prefixes
    )
    continuation["terminal_active_prefix_checkpoint"] = copy.deepcopy(
        terminal
    )
    adapt.pop("verified_singleton_resume_sidecar", None)
    repaired["adapt_vqe"] = adapt
    return repaired, changes


def _materialize_occurrence_corrected_checkpoint(
    *,
    source_checkpoint: Path,
    destination_root: Path,
    expected_depth: int,
    protocol: Any,
    provenance_role: str,
) -> dict[str, Any]:
    from pipelines.static_adapt.current_checkpoint import (
        _publish_active_cli_current_checkpoint,
    )
    from pipelines.static_adapt.sr_snake._resume import (
        load_canonical_accepted_state_resume,
    )
    from pipelines.static_adapt.sr_snake import AcceptedStateResume

    source_binding = _binding(source_checkpoint)
    source_payload = base._load_json(source_checkpoint)
    source_sidecars = _checkpoint_sidecars(
        source_checkpoint, expected_depth=expected_depth
    )
    ledger_sidecar_path = _safe_child(
        source_checkpoint.parent,
        source_sidecars["estimator_ledger_checkpoint"]["path"],
        label="repair-source ledger sidecar",
    )
    ledger_sidecar = base._load_json(ledger_sidecar_path)
    ledger_payload = ledger_sidecar.get("ledger")
    if not isinstance(ledger_payload, Mapping):
        raise ContinuationContractError(
            "Repair-source ledger payload is absent."
        )
    repaired_payload, changes = _repair_active_prefix_owner_occurrences(
        source_payload, expected_depth=expected_depth
    )
    if not changes:
        raise ContinuationContractError(
            "Occurrence repair found no checkpoint-owner mismatch."
        )
    destination_root.mkdir(parents=True, exist_ok=False)
    destination = destination_root / "checkpoint.json"
    _publish_active_cli_current_checkpoint(
        repaired_payload,
        ledger_payload=ledger_payload,
        path=destination,
        keep_history_tail=100,
    )
    destination_binding = _binding(destination, root=destination_root)
    destination_sidecars = _checkpoint_sidecars(
        destination, expected_depth=expected_depth
    )
    hydration = load_canonical_accepted_state_resume(
        AcceptedStateResume(
            checkpoint_path=destination,
            checkpoint_sha256=destination_binding["sha256"],
        ),
        expected_problem=base._problem_from_receipt(protocol.problem),
        expected_route_profile=str(protocol.route_contract["route_profile"]),
        expected_route_contract_sha256=str(protocol.route_contract["sha256"]),
    )
    if int(hydration.controller_round) != expected_depth:
        raise ContinuationContractError(
            "Occurrence-corrected checkpoint did not hydrate at its depth."
        )
    receipt = base._digested(
        {
            "schema": "paper_i_ra_adapt_occurrence_owner_repair_v1",
            "status": "passed",
            "provenance_role": provenance_role,
            "scientific_state_changed": False,
            "repair_scope": (
                "signed_active_prefix_ra_retained_parent_owner_occurrences"
            ),
            "cause": (
                "label_keyed_registry_overwrote_an_older_duplicate_child_"
                "occurrence_owner_receipt"
            ),
            "source_checkpoint": source_binding,
            "source_checkpoint_sidecars": source_sidecars,
            "corrected_checkpoint": destination_binding,
            "corrected_checkpoint_sidecars": destination_sidecars,
            "controller_round": expected_depth,
            "changed_prefix_rounds": [
                int(value)
                for value in sorted(
                    {int(row["controller_round"]) for row in changes}
                )
            ],
            "changes": changes,
            "canonical_resume_validation": "passed",
            "created_at_utc": base._utc_now(),
        }
    )
    base._write_json(destination_root / "repair_receipt.json", receipt)
    return {
        "checkpoint_path": destination,
        "checkpoint": destination_binding,
        "checkpoint_sidecars": destination_sidecars,
        "repair_receipt": _binding(
            destination_root / "repair_receipt.json", root=destination_root
        ),
        "repair_receipt_sha256": receipt["sha256"],
        "changed_prefix_rounds": receipt["changed_prefix_rounds"],
    }


def _validate_source() -> dict[str, Any]:
    for path, label in (
        (
            base.MATERIALIZATION_ROOT / "materialization_plan.json",
            "materialization plan",
        ),
        (
            base.MATERIALIZATION_ROOT / "source_locks_snapshot.json",
            "source-lock snapshot",
        ),
        (
            base.MATERIALIZATION_ROOT / "validation_report.json",
            "materialization validation",
        ),
        (
            base.MATERIALIZATION_ROOT / "materialization_receipt.json",
            "materialization receipt",
        ),
    ):
        _require_digest(path, label=label)
    protocol = base._load_bound_protocol(CELL_ID)
    manifest = _require_digest(
        SOURCE_RUN_ROOT / "run_manifest.json", label="source run manifest"
    )
    terminal = _require_digest(
        SOURCE_RUN_ROOT / "terminal_receipt.json",
        label="source terminal receipt",
    )
    authorization = _require_digest(
        SOURCE_RUN_ROOT / "execution_authorization.json",
        label="source execution authorization",
    )
    if (
        terminal.get("status") != "passed"
        or terminal.get("cell_id") != CELL_ID
        or int(terminal.get("accepted_controller_rounds", -1))
        != SOURCE_ROUND
        or manifest.get("cell_id") != CELL_ID
        or int(manifest.get("operational_maximum_controller_rounds", -1))
        != SOURCE_ROUND
        or float(
            manifest.get(
                "plateau_cumulative_decrease_ratio_threshold", -1.0
            )
        )
        != PLATEAU_RATIO
        or terminal.get("protocol_sha256") != protocol.sha256
        or manifest.get("protocol_sha256") != protocol.sha256
        or terminal.get("manifest_sha256") != manifest.get("sha256")
        or manifest.get("execution_authorization_sha256")
        != authorization.get("sha256")
    ):
        raise ContinuationContractError("The round-20 source binding drifted.")

    checkpoint = SOURCE_RUN_ROOT / "checkpoint.json"
    result = SOURCE_RUN_ROOT / "result.json"
    ledger = SOURCE_RUN_ROOT / "estimator_ledger.json"
    bindings = {
        "checkpoint": _binding(checkpoint),
        "result": _binding(result),
        "estimator_ledger": _binding(ledger),
        "terminal_receipt": _binding(
            SOURCE_RUN_ROOT / "terminal_receipt.json"
        ),
    }
    for role, receipt_key in (
        ("checkpoint", "checkpoint_sha256"),
        ("result", "result_sha256"),
        ("estimator_ledger", "estimator_ledger_sha256"),
    ):
        if bindings[role]["sha256"] != terminal.get(receipt_key):
            raise ContinuationContractError(
                f"Source {role} no longer matches its terminal receipt."
            )
    sidecars = _checkpoint_sidecars(checkpoint, expected_depth=SOURCE_ROUND)
    source_result = base._load_json(result)
    run = source_result.get("run")
    if not isinstance(run, Mapping):
        raise ContinuationContractError("Source result has no run payload.")
    for role in (
        "accepted_trajectory",
        "accepted_transitions",
        "scientific_replay",
    ):
        rows = run.get(role)
        if not isinstance(rows, list) or len(rows) != SOURCE_ROUND:
            raise ContinuationContractError(
                f"Source result has no complete {role} prefix."
            )
    return {
        "protocol": protocol,
        "manifest": manifest,
        "terminal": terminal,
        "bindings": bindings,
        "sidecars": sidecars,
        "source_result": source_result,
    }


def preflight() -> dict[str, Any]:
    source = _validate_source()
    return base._digested(
        {
            "schema": "paper_i_ra_adapt_local_continuation_preflight_v1",
            "status": "passed",
            "cell_id": CELL_ID,
            "source_round": SOURCE_ROUND,
            "target_round": TARGET_ROUND,
            "protocol_sha256": source["protocol"].sha256,
            "source_terminal_receipt_sha256": source["terminal"]["sha256"],
            "source_checkpoint": source["bindings"]["checkpoint"],
            "source_checkpoint_sidecars": source["sidecars"],
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
            "source_checkpoint_sha256": source["bindings"]["checkpoint"][
                "sha256"
            ],
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
        repaired_source = _materialize_occurrence_corrected_checkpoint(
            source_checkpoint=SOURCE_RUN_ROOT / "checkpoint.json",
            destination_root=OUTPUT_ROOT / "resume_source",
            expected_depth=SOURCE_ROUND,
            protocol=protocol,
            provenance_role="round_20_resume_source",
        )
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
                "immutable_source_checkpoint": source["bindings"][
                    "checkpoint"
                ],
                "resume_input": repaired_source["checkpoint"],
                "resume_input_sidecars": repaired_source[
                    "checkpoint_sidecars"
                ],
                "resume_repair_receipt": repaired_source["repair_receipt"],
                "resume_repair_receipt_sha256": repaired_source[
                    "repair_receipt_sha256"
                ],
                "source_terminal_receipt": source["bindings"][
                    "terminal_receipt"
                ],
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
        )
        result = run_ra_adapt(
            base._problem_from_receipt(protocol.problem),
            protocol,
            operational_controls=controls,
        )
        payload = result.to_dict()
        resumed_run = payload.get("run")
        source_run = source["source_result"].get("run")
        if not isinstance(resumed_run, Mapping) or not isinstance(
            source_run, Mapping
        ):
            raise ContinuationContractError("Result run payload is absent.")
        for role in (
            "accepted_trajectory",
            "accepted_transitions",
            "scientific_replay",
        ):
            rows = resumed_run.get(role)
            source_rows = source_run.get(role)
            if not isinstance(rows, list) or len(rows) != TARGET_ROUND:
                raise ContinuationContractError(
                    f"Continuation did not close {role} through round 30."
                )
            if rows[:SOURCE_ROUND] != source_rows:
                raise ContinuationContractError(
                    f"Continuation changed the authenticated {role} prefix."
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
        writer_checkpoint_sidecars = _checkpoint_sidecars(
            checkpoint, expected_depth=TARGET_ROUND
        )
        canonical_checkpoint = _materialize_occurrence_corrected_checkpoint(
            source_checkpoint=checkpoint,
            destination_root=OUTPUT_ROOT / "canonical_resume_checkpoint",
            expected_depth=TARGET_ROUND,
            protocol=protocol,
            provenance_role="round_30_canonical_resume_checkpoint",
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
                "source_terminal_receipt_sha256": source["terminal"][
                    "sha256"
                ],
                "source_checkpoint_sha256": source["bindings"][
                    "checkpoint"
                ]["sha256"],
                "writer_checkpoint": _binding(
                    checkpoint, root=OUTPUT_ROOT
                ),
                "writer_checkpoint_sidecars": writer_checkpoint_sidecars,
                "canonical_resume_checkpoint": canonical_checkpoint[
                    "checkpoint"
                ],
                "canonical_resume_checkpoint_sidecars": (
                    canonical_checkpoint["checkpoint_sidecars"]
                ),
                "canonical_resume_repair_receipt": canonical_checkpoint[
                    "repair_receipt"
                ],
                "canonical_resume_repair_receipt_sha256": (
                    canonical_checkpoint["repair_receipt_sha256"]
                ),
                "result": _binding(
                    OUTPUT_ROOT / "result.json", root=OUTPUT_ROOT
                ),
                "paper_i_summary": _binding(
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


def _trajectory_from_checkpoint(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    adapt = payload.get("adapt_vqe")
    if not isinstance(adapt, Mapping):
        raise ContinuationContractError("Trajectory checkpoint has no ADAPT block.")
    history = adapt.get("history")
    if not isinstance(history, list):
        raise ContinuationContractError("Trajectory checkpoint has no history.")
    insertion_positions: list[int] = []
    trajectory: list[dict[str, Any]] = []
    for round_index, row in enumerate(history, start=1):
        if not isinstance(row, Mapping):
            raise ContinuationContractError("Trajectory history row is invalid.")
        effective = row.get("selected_effective_positions")
        original = row.get("selected_positions")
        if not isinstance(effective, list) or not isinstance(original, list):
            raise ContinuationContractError(
                f"Round {round_index} insertion receipt is absent."
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
                raise ContinuationContractError(
                    f"Round {round_index} prune receipt is unsupported."
                )
            del insertion_positions[int(deleted[0])]
        prefix = row.get("active_prefix_checkpoint")
        if not isinstance(prefix, Mapping):
            raise ContinuationContractError(
                f"Round {round_index} signed prefix is absent."
            )
        operators = prefix.get("ordered_active_operators")
        if not isinstance(operators, list) or len(operators) != len(
            insertion_positions
        ):
            raise ContinuationContractError(
                f"Round {round_index} active operator rows are incomplete."
            )
        trajectory.append(
            {
                "controller_round": round_index,
                "energy": float(row["energy_after_opt"]),
                "generator_ids": [
                    str(operator["generator_id"]) for operator in operators
                ],
                "insertion_positions": list(insertion_positions),
                "logical_parameters": [
                    float(value)
                    for value in prefix[
                        "signed_unwrapped_logical_parameters"
                    ]
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
                    for value in prefix[
                        "signed_unwrapped_runtime_parameters"
                    ]
                ],
            }
        )
    return trajectory


def finalize_completed_round30() -> dict[str, Any]:
    source = _validate_source()
    protocol = source["protocol"]
    if FINALIZED_ROUND30_ROOT.exists() or FINALIZED_ROUND30_ROOT.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite {FINALIZED_ROUND30_ROOT}"
        )
    completed_checkpoint = COMPLETED_ROUND30_ROOT / "checkpoint.json"
    completed_payload = base._load_json(completed_checkpoint)
    if int(completed_payload["checkpoint"]["depth"]) != TARGET_ROUND:
        raise ContinuationContractError(
            "Completed continuation checkpoint is not at round 30."
        )
    completed_authorization = _require_digest(
        COMPLETED_ROUND30_ROOT / "execution_authorization.json",
        label="completed continuation authorization",
    )
    completed_manifest = _require_digest(
        COMPLETED_ROUND30_ROOT / "run_manifest.json",
        label="completed continuation manifest",
    )
    completed_failure = _require_digest(
        COMPLETED_ROUND30_ROOT / "failure_receipt.json",
        label="completed continuation post-validation failure",
    )
    if (
        completed_authorization.get("execution_authorized") is not True
        or int(completed_authorization.get("target_round", -1)) != TARGET_ROUND
        or completed_manifest.get("execution_authorization_sha256")
        != completed_authorization.get("sha256")
        or completed_failure.get("error_type")
        != "ContinuationContractError"
        or completed_failure.get("error")
        != (
            "Continuation changed the authenticated accepted_transitions "
            "prefix."
        )
    ):
        raise ContinuationContractError(
            "Completed continuation provenance is not the known post-run "
            "validation failure."
        )

    FINALIZED_ROUND30_ROOT.mkdir(parents=True, exist_ok=False)
    canonical = _materialize_occurrence_corrected_checkpoint(
        source_checkpoint=completed_checkpoint,
        destination_root=(
            FINALIZED_ROUND30_ROOT / "canonical_resume_checkpoint"
        ),
        expected_depth=TARGET_ROUND,
        protocol=protocol,
        provenance_role="completed_round_30_canonical_resume_checkpoint",
    )
    canonical_payload = base._load_json(canonical["checkpoint_path"])
    trajectory = _trajectory_from_checkpoint(canonical_payload)
    if len(trajectory) != TARGET_ROUND:
        raise ContinuationContractError(
            "Recovered trajectory does not close through round 30."
        )
    source_prefix = source["source_result"]["run"]["accepted_trajectory"]
    if trajectory[:SOURCE_ROUND] != source_prefix:
        raise ContinuationContractError(
            "Recovered trajectory changed the exact round-20 prefix."
        )
    exact_energy = base.EXACT_ENERGIES[CELL_ID]
    points = [
        {
            "controller_round": int(row["controller_round"]),
            "delta_e": abs(float(row["energy"]) - exact_energy),
        }
        for row in trajectory
    ]
    trajectory_payload = base._digested(
        {
            "schema": "paper_i_ra_adapt_recovered_accepted_trajectory_v1",
            "status": "passed",
            "cell_id": CELL_ID,
            "accepted_controller_rounds": TARGET_ROUND,
            "protocol_sha256": protocol.sha256,
            "exact_round_20_prefix_match": True,
            "accepted_trajectory": trajectory,
            "same_cutoff_delta_e_points": points,
            "estimator_accounting": copy.deepcopy(
                canonical_payload["adapt_vqe"]["estimator_call_accounting"]
            ),
            "not_paper_evidence": True,
        }
    )
    base._write_json(
        FINALIZED_ROUND30_ROOT / "accepted_trajectory.json",
        trajectory_payload,
    )
    final_delta = float(points[-1]["delta_e"])
    terminal = base._digested(
        {
            "schema": "paper_i_ra_adapt_local_recovered_terminal_v1",
            "status": "passed",
            "cell_id": CELL_ID,
            "accepted_controller_rounds": TARGET_ROUND,
            "final_same_cutoff_delta_e": final_delta,
            "protocol_sha256": protocol.sha256,
            "source_round_20_terminal_receipt_sha256": source["terminal"][
                "sha256"
            ],
            "completed_round_30_writer_checkpoint": _binding(
                completed_checkpoint
            ),
            "completed_run_authorization": _binding(
                COMPLETED_ROUND30_ROOT / "execution_authorization.json"
            ),
            "completed_run_manifest": _binding(
                COMPLETED_ROUND30_ROOT / "run_manifest.json"
            ),
            "post_run_validation_failure": _binding(
                COMPLETED_ROUND30_ROOT / "failure_receipt.json"
            ),
            "canonical_resume_checkpoint": canonical["checkpoint"],
            "canonical_resume_checkpoint_sidecars": canonical[
                "checkpoint_sidecars"
            ],
            "canonical_resume_repair_receipt": canonical["repair_receipt"],
            "canonical_resume_repair_receipt_sha256": canonical[
                "repair_receipt_sha256"
            ],
            "accepted_trajectory": _binding(
                FINALIZED_ROUND30_ROOT / "accepted_trajectory.json",
                root=FINALIZED_ROUND30_ROOT,
            ),
            "recovery_scope": (
                "post_science_result_serialization_after_overstrict_local_"
                "accepted_transition_prefix_assertion"
            ),
            "scientific_rounds_rerun": False,
            "not_paper_evidence": True,
            "completed_at_utc": base._utc_now(),
        }
    )
    base._write_json(
        FINALIZED_ROUND30_ROOT / "terminal_receipt.json", terminal
    )
    return terminal


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--finalize-completed-round30", action="store_true")
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
    if args.finalize_completed_round30:
        if not args.execution_authorized:
            raise ContinuationContractError(
                "Round-30 finalization requires --execution-authorized."
            )
        print(
            base._canonical_bytes(finalize_completed_round30()).decode(
                "utf-8"
            )
        )
        return 0
    if not args.execution_authorized:
        raise ContinuationContractError(
            "Continuation requires --execution-authorized."
        )
    print(base._canonical_bytes(run()).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
