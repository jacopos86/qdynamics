"""Artifact boundary for the replayable Paper-II staged campaign.

The pure stage machine knows only ledger records.  This adapter validates that
an emitted ``run.json`` actually realized the requested registered command,
constructs post-run exact-reference errors, and appends one immutable record.
"""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.time_dynamics import paper_ii_runs as _runs
from pipelines.time_dynamics.diagnostics.physical_errors import (
    physical_errors_from_artifacts,
)
from pipelines.time_dynamics.staged_campaign import (
    ACTION_RECORD_SAFEGUARD,
    ACTION_RUN,
    CampaignLedger,
    CellAction,
    RungRecord,
    STATUS_SAFEGUARD_TRIPPED,
    StagedCampaign,
)


def _matches(actual: Any, expected: Any) -> bool:
    if expected is None:
        return actual is None
    if isinstance(expected, bool):
        return bool(actual) is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return math.isclose(
                float(actual),
                float(expected),
                rel_tol=1.0e-12,
                abs_tol=1.0e-14,
            )
        except (TypeError, ValueError):
            return False
    return str(actual) == str(expected)


def _read_nonempty_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file() or target.stat().st_size <= 0:
        raise ValueError(f"completed run artifact is missing or empty: {target}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"completed run artifact is not a nonempty object: {target}")
    return payload


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_resume_support_boundary(
    *,
    action: CellAction,
    run: Mapping[str, Any],
) -> dict[str, Any]:
    """Require a continuation to begin on the complete saved terminal support.

    A resume artifact serializes every runtime coordinate of the preceding
    leg, including drive-aligned children.  Reapplying fresh-run augmentation
    at the shared boundary silently changes the manifold before propagation;
    a run may then finish without triggering the duplicate-label guard.  The
    artifact boundary therefore checks the realized initial support itself,
    rather than trusting that the runner was invoked with a resume flag.
    """

    if action.resume_from_run_path is None:
        return {
            "resume_support_boundary_verified": False,
            "resume_runtime_coordinate_count": None,
        }

    source = _read_nonempty_json(action.resume_from_run_path)
    source_resume = dict(source.get("resume_state") or {})
    coordinates = tuple(source_resume.get("coordinates") or ())
    expected_labels = tuple(
        str(dict(coordinate).get("label") or "") for coordinate in coordinates
    )
    if not expected_labels or any(not label for label in expected_labels):
        raise ValueError(
            "resume source has no complete labeled runtime-coordinate support"
        )
    if len(set(expected_labels)) != len(expected_labels):
        raise ValueError("resume source contains duplicate runtime-coordinate labels")

    initial = dict(run.get("initial_state") or {})
    actual_labels = tuple(
        str(label) for label in (initial.get("runtime_coordinate_labels") or ())
    )
    actual_selected = tuple(
        str(label) for label in (initial.get("selected_term_labels") or ())
    )
    actual_count = initial.get("runtime_parameter_count")
    actual_logical_count = initial.get("logical_parameter_count")
    augmentation = dict(run.get("drive_aligned_ansatz") or {})
    mismatches: list[str] = []
    if actual_labels != expected_labels:
        mismatches.append(
            "initial runtime-coordinate labels differ from the saved support"
        )
    if actual_selected != expected_labels:
        mismatches.append(
            "initial selected-term labels differ from the saved support"
        )
    if not _matches(actual_count, len(expected_labels)):
        mismatches.append(
            f"initial runtime count {actual_count!r} != saved count {len(expected_labels)}"
        )
    if not _matches(actual_logical_count, len(expected_labels)):
        mismatches.append(
            f"initial logical count {actual_logical_count!r} != saved count {len(expected_labels)}"
        )
    if bool(augmentation.get("applied")):
        mismatches.append("drive-aligned augmentation was reapplied during resume")
    if augmentation.get("reason") != (
        "resume_terminal_support_already_contains_drive_alignment"
    ):
        mismatches.append(
            "drive-aligned augmentation did not record the resume-boundary skip"
        )
    if mismatches:
        raise ValueError(
            "resumed run does not preserve its saved support boundary: "
            + "; ".join(mismatches)
        )
    return {
        "resume_support_boundary_verified": True,
        "resume_runtime_coordinate_count": len(expected_labels),
    }


def verify_run_receipt(
    *,
    action: CellAction,
    command: _runs.RunCommand,
    run: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail if a run silently realized another arm, cut, or shared protocol."""

    trajectory = dict(run.get("trajectory") or {})
    support = dict(trajectory.get("support_patch_config") or {})
    lock = dict(run.get("run_lock") or {})
    physics = dict(lock.get("physics") or {})
    numerics = dict(lock.get("numerics") or {})
    inverse = dict(numerics.get("inverse_policy") or {})
    solve_repair = dict(numerics.get("solve_repair") or {})
    shared = dict(lock.get("shared_protocol") or {})
    controller = dict(shared.get("time_step_controller") or {})
    time_grid = dict(physics.get("time_grid") or {})
    solve_repair_enabled = bool(solve_repair.get("enabled"))

    expected_method = "avqds" if command.arm.is_comparator else "exchange"
    cut_name = "avqds_l2_cut" if command.arm.is_comparator else "insertion_l2_cut"
    expected_point_count = (
        376
        if action.resume_from_run_path is not None
        and action.horizon == "t20"
        and command.horizon.t_final == 20.0
        else int(command.horizon.num_times)
    )
    expected = {
        "method": expected_method,
        "activation_cut": float(action.threshold),
        "drive_A": float(command.drive.amplitude),
        "drive_omega": float(command.drive.omega),
        "integrator": str(command.numerics.integrator),
        "pinv_rcond": float(command.numerics.pinv_rcond),
        "ridge_lambda": float(command.numerics.ridge_lambda),
        "solve_damping": float(command.numerics.solve_damping),
        "solve_repair_enabled": bool(command.step_control.solve_repair_enabled),
        "repair_ridge_ladder": [float(command.numerics.ridge_lambda)],
        "repair_pinv_rcond_ladder": [float(command.numerics.pinv_rcond)],
        "repair_solve_damping_ladder": [float(command.numerics.solve_damping)],
        "parameter_displacement_max": command.step_control.parameter_step_max,
        "state_motion_l2_step_max": command.step_control.state_motion_l2_step_max,
        "max_local_subdivisions": command.step_control.max_local_subdivisions,
        "joint_patch_evaluations": int(
            command.search_profile.joint_patch_evaluations
        ),
        "certification_attempts_per_level": int(
            command.search_profile.certification_attempts_per_level
        ),
        "certification_attempts_per_deletion_branch": int(
            command.search_profile.certification_attempts_per_deletion_branch
        ),
        "rounds_per_checkpoint": int(command.search_profile.rounds_per_checkpoint),
        "insertion_cardinality": int(command.search_profile.insertion_cardinality),
        "t_initial": (
            5.0 if action.resume_from_run_path is not None else 0.0
        ),
        "t_final": float(command.horizon.t_final),
        "point_count": expected_point_count,
    }
    actual = {
        "method": support.get("dynamics_policy"),
        "activation_cut": support.get(cut_name),
        "drive_A": dict(physics.get("drive_profile") or {}).get("A"),
        "drive_omega": dict(physics.get("drive_profile") or {}).get("omega"),
        "integrator": trajectory.get("integrator_method"),
        "pinv_rcond": inverse.get("pinv_rcond"),
        "ridge_lambda": inverse.get("ridge_lambda"),
        "solve_damping": inverse.get("solve_damping"),
        "solve_repair_enabled": solve_repair_enabled,
        "repair_ridge_ladder": solve_repair.get("ridge_ladder"),
        "repair_pinv_rcond_ladder": solve_repair.get("pinv_rcond_ladder"),
        "repair_solve_damping_ladder": solve_repair.get(
            "solve_damping_ladder"
        ),
        "parameter_displacement_max": controller.get(
            "composed_parameter_step_max"
            if solve_repair_enabled
            else "parameter_displacement_max"
        ),
        # Older run-lock schemas serialize raw state-controller defaults even
        # when solve repair is disabled.  They are configuration-surface
        # values, not an active time-step controller, so normalize them away
        # only after independently checking the recorded activation bit.
        "state_motion_l2_step_max": (
            controller.get("state_motion_l2_step_max")
            if solve_repair_enabled
            else None
        ),
        "max_local_subdivisions": (
            controller.get("max_local_subdivisions")
            if solve_repair_enabled
            else None
        ),
        "joint_patch_evaluations": support.get("max_joint_patch_evaluations"),
        "certification_attempts_per_level": support.get(
            "max_certification_attempts_per_level"
        ),
        "certification_attempts_per_deletion_branch": support.get(
            "max_certification_attempts_per_deletion_branch"
        ),
        "rounds_per_checkpoint": support.get("max_insertion_rounds_per_checkpoint"),
        "insertion_cardinality": support.get("max_insertion_batch_size"),
        "t_initial": time_grid.get("t_initial"),
        "t_final": time_grid.get("t_final"),
        "point_count": time_grid.get("point_count"),
    }
    mismatches = {
        name: {"expected": expected[name], "actual": actual[name]}
        for name in expected
        if not _matches(actual[name], expected[name])
    }
    if mismatches:
        details = ", ".join(
            f"{name}={row['actual']!r} (expected {row['expected']!r})"
            for name, row in mismatches.items()
        )
        raise ValueError(f"run artifact does not realize its staged action: {details}")

    plot_rows = tuple(run.get("plot_rows") or ())
    if len(plot_rows) != expected_point_count:
        raise ValueError(
            f"run plot-row count {len(plot_rows)} != expected {expected_point_count}"
        )
    if not all("psi_real" in row and "psi_imag" in row for row in plot_rows):
        raise ValueError("staged campaign run did not record every checkpoint statevector")
    controller_resume = dict(run.get("resume_state") or {}).get("controller_state")
    if not isinstance(controller_resume, Mapping):
        raise ValueError("staged campaign run has no decision-identical controller resume")
    resume_boundary = _verify_resume_support_boundary(action=action, run=run)
    return {**actual, **resume_boundary, "cut_field": cut_name}


def record_completed_action(
    *,
    campaign: StagedCampaign,
    ledger: CampaignLedger,
    action: CellAction,
    command: _runs.RunCommand,
    run_path: str | Path,
    seed_path: str | Path,
    reference_substeps: int = 40,
    reference_store_root: str | Path = "output/paper_ii_staged/exact_references",
) -> CampaignLedger:
    """Validate and ingest one completed staged action."""

    if action.action != ACTION_RUN:
        raise ValueError("only run actions can ingest a run artifact")
    run = _read_nonempty_json(run_path)
    achieved = verify_run_receipt(action=action, command=command, run=run)
    metrics = physical_errors_from_artifacts(
        run_path=run_path,
        seed_path=seed_path,
        screen_run_path=action.resume_from_run_path,
        reference_substeps=int(reference_substeps),
        reference_store_root=reference_store_root,
    )
    return ledger.append(
        RungRecord.completed(
            key=action.key,
            horizon=action.horizon,
            threshold=action.threshold,
            run_path=str(run_path),
            metrics=metrics,
            policy=campaign.accuracy_policy,
            resume_from_run_path=action.resume_from_run_path,
            achieved_configuration=achieved,
        )
    ).reclassified(campaign.accuracy_policy)


def record_run_outcome(
    *,
    campaign: StagedCampaign,
    ledger: CampaignLedger,
    action: CellAction,
    command: _runs.RunCommand,
    run_path: str | Path,
    seed_path: str | Path,
    reference_substeps: int = 40,
    reference_store_root: str | Path = "output/paper_ii_staged/exact_references",
) -> CampaignLedger:
    """Ingest a runner artifact, classifying an uncured step as a safeguard.

    The runner writes ``run.json`` before returning exit status 3 for an
    exhausted subdivision budget.  Such a trajectory is useful provenance but
    is not accuracy evidence, so it is recorded without computing physical
    errors and without opening another structural-threshold rung.
    """

    if action.action != ACTION_RUN:
        raise ValueError("only run actions can ingest a runner outcome")
    run = _read_nonempty_json(run_path)
    achieved = verify_run_receipt(action=action, command=command, run=run)
    uncured = int(
        (run.get("summary") or {}).get("uncured_local_subdivision_count") or 0
    )
    if uncured > 0:
        return ledger.append(
            RungRecord(
                key=action.key,
                horizon=action.horizon,
                threshold=action.threshold,
                status=STATUS_SAFEGUARD_TRIPPED,
                run_path=str(run_path),
                reason=(
                    f"{uncured} uncured local-subdivision step(s) exhausted "
                    "the declared controller budget"
                ),
                resume_from_run_path=action.resume_from_run_path,
                achieved_configuration=achieved,
            )
        )
    metrics = physical_errors_from_artifacts(
        run_path=run_path,
        seed_path=seed_path,
        screen_run_path=action.resume_from_run_path,
        reference_substeps=int(reference_substeps),
        reference_store_root=reference_store_root,
    )
    return ledger.append(
        RungRecord.completed(
            key=action.key,
            horizon=action.horizon,
            threshold=action.threshold,
            run_path=str(run_path),
            metrics=metrics,
            policy=campaign.accuracy_policy,
            resume_from_run_path=action.resume_from_run_path,
            achieved_configuration=achieved,
        )
    ).reclassified(campaign.accuracy_policy)


def record_safeguard_action(
    *,
    ledger: CampaignLedger,
    action: CellAction,
) -> CampaignLedger:
    if action.action != ACTION_RECORD_SAFEGUARD:
        raise ValueError("only safeguard actions can record a safeguard event")
    return ledger.append(
        RungRecord(
            key=action.key,
            horizon=action.horizon,
            threshold=action.threshold,
            status=STATUS_SAFEGUARD_TRIPPED,
            reason=action.reason,
            resume_from_run_path=action.resume_from_run_path,
        )
    )


def record_prepared_action_outputs(
    *,
    campaign: StagedCampaign,
    ledger: CampaignLedger,
    action_rows: Sequence[Mapping[str, Any]],
    artifact_root: str | Path,
    seed_path: str | Path,
    numerics: str,
    gate: str,
    search_profile: str,
    reference_substeps: int = 40,
    reference_store_root: str | Path = "output/paper_ii_staged/exact_references",
) -> tuple[CampaignLedger, tuple[dict[str, Any], ...]]:
    """Validate and ingest every artifact emitted by one prepared package.

    The package's original ``action`` payload—not its transport-rewritten
    resume path—is replayed against the live ledger.  Trajectory paths must be
    relative to the verified extraction root, so a malformed manifest cannot
    make ingestion read another attempt.  Replaying the same verified
    extraction is idempotent when its exact run path is already present.
    """

    root = Path(artifact_root)
    if not root.is_dir():
        raise ValueError(f"artifact extraction root does not exist: {root}")
    rows = tuple(dict(row) for row in action_rows)
    ids = tuple(str(row.get("action_id") or "") for row in rows)
    if not rows or any(not action_id for action_id in ids):
        raise ValueError("prepared action rows must be nonempty and labeled")
    if len(set(ids)) != len(ids):
        raise ValueError("prepared action rows contain duplicate action identifiers")

    current_ledger = ledger
    receipts: list[dict[str, Any]] = []
    for row in rows:
        action = CellAction.from_json_dict(dict(row.get("action") or {}))
        if str(row["action_id"]) != action.action_id:
            raise ValueError(
                "prepared row action_id differs from its scientific action payload"
            )
        declared_path = Path(str(row.get("trajectory_json") or ""))
        if (
            not str(declared_path)
            or declared_path.is_absolute()
            or ".." in declared_path.parts
        ):
            raise ValueError(
                f"prepared trajectory path must stay under the extraction root: "
                f"{declared_path}"
            )
        run_path = root / declared_path

        previous = next(
            (
                record
                for record in current_ledger.records
                if record.run_path is not None
                and Path(record.run_path) == run_path
            ),
            None,
        )
        if previous is not None:
            if (
                previous.key != action.key
                or previous.horizon != action.horizon
                or not _matches(previous.threshold, action.threshold)
            ):
                raise ValueError(
                    "previous ledger record at the extracted run path differs "
                    "from the prepared action"
                )
            receipts.append(
                {
                    "action_id": action.action_id,
                    "run_path": str(run_path),
                    "run_sha256": _sha256_file(run_path),
                    "status": "already_recorded",
                    "ledger_status": previous.status,
                }
            )
            continue

        current = {
            candidate.action_id: candidate
            for candidate in campaign.next_actions(current_ledger)
        }
        if action.action_id not in current:
            raise ValueError(
                f"prepared action {action.action_id!r} is stale for this ledger"
            )
        if current[action.action_id] != action:
            raise ValueError(
                f"prepared action {action.action_id!r} differs from ledger replay"
            )
        command = campaign.build_run_command(
            action,
            seed_path=str(seed_path),
            output_json=str(run_path),
            numerics=str(numerics),
            gate=str(gate),
            search_profile=str(search_profile),
        )
        current_ledger = record_run_outcome(
            campaign=campaign,
            ledger=current_ledger,
            action=action,
            command=command,
            run_path=run_path,
            seed_path=seed_path,
            reference_substeps=int(reference_substeps),
            reference_store_root=reference_store_root,
        )
        record = current_ledger.records[-1]
        receipts.append(
            {
                "action_id": action.action_id,
                "run_path": str(run_path),
                "run_sha256": _sha256_file(run_path),
                "status": "recorded",
                "ledger_status": record.status,
            }
        )
    return current_ledger, tuple(receipts)


__all__ = [
    "record_completed_action",
    "record_prepared_action_outputs",
    "record_run_outcome",
    "record_safeguard_action",
    "verify_run_receipt",
]
