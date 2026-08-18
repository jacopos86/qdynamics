"""Run append-first AP-McLachlan from a static scaffold artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.scaffold.runtime_loader import (
    load_scaffold_runtime_input,
    load_scaffold_runtime_input_from_payload,
)
from pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory import (
    APPEND_LADDER_PREFILTER_POLICY_V1,
    APPEND_LADDER_SELECTION_POLICY_V1,
    APPEND_MACRO_SCOUT_SCORE_MODE_PARENT_TANGENT_SCHUR_GAIN,
    DEFAULT_APPEND_RESIDUAL_RATIO_THRESHOLD,
    PRUNE_PERSISTENCE_ATOM_HISTORY,
    PRUNE_PERSISTENCE_EXACT_BATCH,
    PRUNE_TARGET_POLICIES,
    AppendControllerConfig,
    SolveRepairConfig,
    SupportPatchControllerConfig,
    run_append_mclachlan_trajectory,
)
from pipelines.time_dynamics.ap_mclachlan.drive_aligned import (
    augment_state_with_drive_aligned_generator,
)
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import (
    time_dependent_hamiltonian_from_runtime_input,
)
from pipelines.time_dynamics.ap_mclachlan.integrators import (
    INTEGRATOR_EULER,
    SUPPORTED_INTEGRATORS,
)
from pipelines.time_dynamics.ap_mclachlan.inverse import (
    DEFAULT_MCLACHLAN_RIDGE_LAMBDA,
    DEFAULT_MCLACHLAN_SOLVE_DAMPING,
    McLachlanInversePolicy,
)
from pipelines.time_dynamics.ap_mclachlan.observables import (
    build_site_doublon_observable_plan,
    observable_row_fields,
)
from pipelines.time_dynamics.ap_mclachlan.reference_diagnostics import (
    ReferenceEnergyTrajectory,
    attach_reference_energy_diagnostics,
    attach_reference_energy_diagnostics_with_prefix,
    load_reference_energy_trajectory,
    reference_energy_summary,
    reference_energy_trajectory_from_payload,
)
from pipelines.time_dynamics.ap_mclachlan.support_frontier import (
    APPEND_MACRO_SCOUT_SCORE_MODES,
)
from pipelines.time_dynamics.ap_mclachlan.state import (
    AP_PARAMETERIZATION_PER_PAULI_TERM,
    AP_SUPPORTED_PARAMETERIZATION_MODES,
    state_from_scaffold_runtime_input,
)
from pipelines.time_dynamics.ap_mclachlan.support_atoms import (
    APPEND_OCCURRENCE_POLICIES,
    APPEND_OCCURRENCE_POLICY_LAYER_REUSE,
)
from pipelines.time_dynamics.normalized_pauli_pool import (
    NORMALIZED_POOL_PROFILES,
    build_normalized_pauli_pool,
    runtime_input_with_normalized_candidate_pool,
)
from pipelines.time_dynamics.fixed_vqe_conditioning import (
    fixed_vqe_stress_provenance_from_runtime_input as _fixed_vqe_conditioning_stress_provenance,
)
from pipelines.time_dynamics.runners.ap_fixed_from_adapt_artifact import (
    _drive_config_from_args,
    _parse_times,
)


# The online zero-angle redundancy injection path is gone.  Ansatz redundancy is
# now constructed offline as a real fixed-structure VQE ansatz and consumed as an
# ordinary serialized seed artifact.
REMOVED_ONLINE_REDUNDANCY_FLAGS = (
    "--diagnostic-redundancy-layer-count",
    "--diagnostic-redundancy-pool-profile",
    "--diagnostic-redundancy-layout-mode",
    "--diagnostic-redundancy-state-parity-atol",
)
REMOVED_ONLINE_REDUNDANCY_MESSAGE = (
    "the online zero-angle ANZATS redundancy injection path has been removed; it "
    "manufactured redundancy at run time instead of constructing a genuine fixed "
    "ansatz. Build a fixed-VQE conditioning-stress seed offline with "
    "pipelines/time_dynamics/runners/build_fixed_vqe_conditioning_seed.py and pass "
    "the resulting artifact to --artifact-json."
)


def reject_removed_online_redundancy_flags(argv: Sequence[str]) -> None:
    """Reject the deleted ``--diagnostic-redundancy-*`` command line explicitly."""

    used = sorted(
        {
            str(flag)
            for token in argv
            for flag in REMOVED_ONLINE_REDUNDANCY_FLAGS
            if str(token) == flag or str(token).startswith(f"{flag}=")
        }
    )
    if used:
        raise SystemExit(f"error: {REMOVED_ONLINE_REDUNDANCY_MESSAGE} Rejected flags: {used}.")


RUNNER_SCHEMA_V1 = "ap_mclachlan_append_from_adapt_artifact_v1"


def run_append_ap_mclachlan_from_runtime_input(
    runtime_input: Any,
    *,
    times: Sequence[float],
    integrator_method: str = INTEGRATOR_EULER,
    pinv_rcond: float = 1.0e-10,
    ridge_lambda: float = DEFAULT_MCLACHLAN_RIDGE_LAMBDA,
    solve_damping: float = DEFAULT_MCLACHLAN_SOLVE_DAMPING,
    enable_drive: bool = False,
    drive_config: Any | None = None,
    drive_aligned_ansatz: bool = True,
    parameterization_mode: str = AP_PARAMETERIZATION_PER_PAULI_TERM,
    controller_config: AppendControllerConfig = AppendControllerConfig(),
    support_patch_config: SupportPatchControllerConfig | None = None,
    solve_repair_config: SolveRepairConfig = SolveRepairConfig(),
    reference_energy_trajectory: Any | None = None,
    reference_energy_atol: float = 1.0e-12,
    seed_reference_energy_trajectory: Any | None = None,
    seed_reference_energy_atol: float = 1.0e-12,
    progress_log_every: int = 0,
    progress_log_events: bool = True,
    normalized_candidate_pool_profile: str | None = None,
    runner_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run append-first AP-McLachlan from an already-loaded scaffold contract."""

    if bool(enable_drive) and drive_config is None:
        raise ValueError("enable_drive=True requires a drive_config.")
    hamiltonian = time_dependent_hamiltonian_from_runtime_input(
        runtime_input,
        drive_config=(drive_config if bool(enable_drive) else None),
    )
    normalized_pool_contract = None
    if normalized_candidate_pool_profile not in {None, "", "none"}:
        normalized_pool_contract = build_normalized_pauli_pool(
            profile=str(normalized_candidate_pool_profile),
            static_poly=hamiltonian.static_poly,
            drive_poly=hamiltonian.drive_poly,
            candidate_pool_terms=tuple(
                getattr(runtime_input, "candidate_pool_terms", ()) or ()
            ),
        )
        runtime_input = runtime_input_with_normalized_candidate_pool(
            runtime_input,
            normalized_pool_contract,
        )
    state = state_from_scaffold_runtime_input(
        runtime_input,
        parameterization_mode=str(parameterization_mode),
    )
    drive_augmentation = augment_state_with_drive_aligned_generator(
        state,
        hamiltonian=hamiltonian,
        enabled=bool(enable_drive) and bool(drive_aligned_ansatz),
    )
    state = drive_augmentation.state
    inverse_policy = McLachlanInversePolicy(
        pinv_rcond=float(pinv_rcond),
        ridge_lambda=float(ridge_lambda),
        solve_damping=float(solve_damping),
    )
    reference = _coerce_reference_energy_trajectory(reference_energy_trajectory)
    seed_reference = _coerce_reference_energy_trajectory(seed_reference_energy_trajectory)
    progress_callback = _build_progress_callback(
        enabled=bool(int(progress_log_every) > 0),
        every=int(progress_log_every),
        log_events=bool(progress_log_events),
        total_points=len(tuple(float(t) for t in times)),
        reference=reference,
        reference_atol=float(reference_energy_atol),
        seed_reference=seed_reference,
        seed_reference_atol=float(seed_reference_energy_atol),
    )
    trajectory = run_append_mclachlan_trajectory(
        state=state,
        hamiltonian=hamiltonian,
        times=tuple(float(t) for t in times),
        inverse_policy=inverse_policy,
        integrator_method=str(integrator_method),
        controller_config=controller_config,
        support_patch_config=support_patch_config,
        solve_repair_config=solve_repair_config,
        metadata={
            "runner_schema": RUNNER_SCHEMA_V1,
            **dict(runner_metadata or {}),
        },
        progress_callback=progress_callback,
    )
    rows = attach_reference_energy_diagnostics(
        plot_rows=_plot_rows(trajectory, initial_state=state),
        reference=reference,
        atol=float(reference_energy_atol),
    )
    rows = attach_reference_energy_diagnostics_with_prefix(
        plot_rows=rows,
        reference=seed_reference,
        atol=float(seed_reference_energy_atol),
        field_prefix="seed_",
    )
    summary = _summary_from_rows(
        rows,
        initial_state=state,
        final_state=trajectory.final_state,
        hamiltonian=hamiltonian,
        integrator_method=str(integrator_method),
        inverse_policy=inverse_policy,
        controller_config=controller_config,
        support_patch_config=support_patch_config,
        solve_repair_config=solve_repair_config,
    )
    normalized_pool_payload = (
        None
        if normalized_pool_contract is None
        else normalized_pool_contract.to_json_dict(include_atoms=False)
    )
    if normalized_pool_payload is not None:
        summary["normalized_candidate_pool"] = dict(normalized_pool_payload)
    summary["fixed_vqe_conditioning_stress"] = _fixed_vqe_conditioning_stress_provenance(
        runtime_input
    )
    return {
        "schema": RUNNER_SCHEMA_V1,
        "initial_state": state.to_json_dict(),
        "final_state": trajectory.final_state.to_json_dict(),
        "normalized_candidate_pool": normalized_pool_payload,
        "drive_aligned_ansatz": drive_augmentation.to_json_dict(),
        "fixed_vqe_conditioning_stress": _fixed_vqe_conditioning_stress_provenance(
            runtime_input
        ),
        "hamiltonian": hamiltonian.to_json_dict(),
        "trajectory": trajectory.to_json_dict(),
        "plot_rows": rows,
        "summary": summary,
        "decision_data_flow": {
            "uses_reference_for_decision": False,
            "uses_exact_reference_for_decision": False,
            "uses_future_exact_forecast_for_decision": False,
            "uses_statevector_as_ideal_observable_estimator": True,
            "reference_energy_error_scope": "post_run_reporting",
            "seed_reference_energy_error_scope": "post_run_reporting",
        },
    }


def _build_progress_callback(
    *,
    enabled: bool,
    every: int,
    log_events: bool,
    total_points: int,
    reference: ReferenceEnergyTrajectory | None,
    reference_atol: float,
    seed_reference: ReferenceEnergyTrajectory | None,
    seed_reference_atol: float,
) -> Any | None:
    if not bool(enabled):
        return None
    interval = max(1, int(every))
    total = max(0, int(total_points))

    def callback(payload: Mapping[str, Any]) -> None:
        phase = str(payload.get("phase", ""))
        index = int(payload.get("index", 0))
        event_due = bool(log_events) and phase == "checkpoint_done" and bool(
            payload.get("patch_accepted", False)
        )
        periodic_due = (index % interval == 0) or (total > 0 and index + 1 == total)
        if not event_due and not periodic_due:
            return
        fields = _progress_energy_fields(
            payload,
            reference=reference,
            reference_atol=float(reference_atol),
            seed_reference=seed_reference,
            seed_reference_atol=float(seed_reference_atol),
        )
        message = _format_progress_message(
            payload,
            total_points=total,
            fields=fields,
        )
        print(message, file=sys.stderr, flush=True)

    return callback


def _progress_energy_fields(
    payload: Mapping[str, Any],
    *,
    reference: ReferenceEnergyTrajectory | None,
    reference_atol: float,
    seed_reference: ReferenceEnergyTrajectory | None,
    seed_reference_atol: float,
) -> dict[str, float | None]:
    time_value = float(payload.get("time", 0.0))
    energy = float(payload.get("energy_expectation", 0.0))
    ref_energy = _progress_reference_energy(
        time_value,
        reference=reference,
        atol=float(reference_atol),
    )
    seed_ref_energy = _progress_reference_energy(
        time_value,
        reference=seed_reference,
        atol=float(seed_reference_atol),
    )
    return {
        "reference_energy": ref_energy,
        "energy_error": None if ref_energy is None else float(energy - ref_energy),
        "abs_energy_error": None if ref_energy is None else float(abs(energy - ref_energy)),
        "seed_reference_energy": seed_ref_energy,
        "seed_energy_error": (
            None if seed_ref_energy is None else float(energy - seed_ref_energy)
        ),
        "seed_abs_energy_error": (
            None if seed_ref_energy is None else float(abs(energy - seed_ref_energy))
        ),
    }


def _progress_reference_energy(
    time_value: float,
    *,
    reference: ReferenceEnergyTrajectory | None,
    atol: float,
) -> float | None:
    if reference is None:
        return None
    best_energy: float | None = None
    best_delta: float | None = None
    for point in reference.points:
        delta = abs(float(point.time) - float(time_value))
        if best_delta is None or delta < best_delta:
            best_delta = float(delta)
            best_energy = float(point.energy)
    if best_energy is None or best_delta is None or best_delta > float(atol):
        return None
    return float(best_energy)


def _format_progress_message(
    payload: Mapping[str, Any],
    *,
    total_points: int,
    fields: Mapping[str, float | None],
) -> str:
    index = int(payload.get("index", 0))
    phase = str(payload.get("phase", "checkpoint"))
    phase_label = "start" if phase == "checkpoint_start" else "done"
    total_label = "?" if int(total_points) <= 0 else str(max(0, int(total_points) - 1))
    parts = [
        "[ap-progress]",
        phase_label,
        f"k={index}/{total_label}",
        f"t={float(payload.get('time', 0.0)):.6g}",
        f"E={float(payload.get('energy_expectation', 0.0)):.10g}",
        f"dE={_format_progress_float(fields.get('energy_error'))}",
        f"|dE|={_format_progress_float(fields.get('abs_energy_error'))}",
        f"seed_dE={_format_progress_float(fields.get('seed_energy_error'))}",
        f"rho={_format_progress_float(payload.get('mclachlan_residual_ratio'))}",
        f"params={int(payload.get('runtime_parameter_count', 0))}",
    ]
    if phase == "checkpoint_done":
        parts.extend(
            [
                f"patch={payload.get('patch_kind')}",
                f"accepted={bool(payload.get('patch_accepted', False))}",
                f"+{int(payload.get('patch_appended_count', 0) or 0)}",
                f"rung={payload.get('patch_selected_rung_size')}",
                f"scored={int(payload.get('patch_scored_count', 0) or 0)}",
                f"reason={payload.get('patch_reason')}",
            ]
        )
    return " ".join(str(part) for part in parts)


def _format_progress_float(value: Any) -> str:
    finite = _finite_or_none(value)
    if finite is None:
        return "n/a"
    return f"{float(finite):.6g}"


def _plot_rows(trajectory: Any, *, initial_state: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    observable_plan = build_site_doublon_observable_plan(
        initial_state.resolved_problem,
        dimension=int(np.asarray(initial_state.psi_ref, dtype=complex).reshape(-1).size),
    )
    for point in trajectory.points:
        theta = np.asarray(point.theta_runtime, dtype=float).reshape(-1)
        theta_dot = np.asarray(point.fixed_step.theta_dot, dtype=float).reshape(-1)
        decision = point.patch_decision
        selected_score = decision.selected_score
        batch = decision.batch_evaluation
        decision_metadata = dict(getattr(decision, "metadata", {}) or {})
        batch_metadata = {} if batch is None else dict(batch.metadata or {})
        selected_candidate = None if batch is None else batch.selected_score
        selected_metadata = (
            {} if selected_candidate is None else dict(selected_candidate.metadata or {})
        )
        patch_metadata = {**selected_metadata, **decision_metadata}
        selected_cost = (
            {}
            if selected_candidate is None
            else dict(selected_metadata.get("append_cost", {}) or {})
        )
        selected_prune_cost = (
            {}
            if selected_candidate is None
            else dict(selected_metadata.get("prune_cost", {}) or {})
        )
        selected_rank_payload = selected_cost or selected_prune_cost
        selected_cost_raw = dict(selected_cost.get("raw_components", {}) or {})
        selected_cost_bar = dict(selected_cost.get("bar_components", {}) or {})
        selected_cost_lambdas = dict(selected_cost.get("lambdas", {}) or {})
        selected_prune_raw = dict(selected_prune_cost.get("raw_components", {}) or {})
        selected_prune_bar = dict(selected_prune_cost.get("bar_components", {}) or {})
        selected_prune_conditioning = dict(
            selected_prune_cost.get("conditioning_components", {}) or {}
        )
        integration = point.integration_to_next
        integration_repair_summary = (
            {} if integration is None else dict(integration.repair_summary or {})
        )
        repair_schedule = point.fixed_step.solve_repair_response_schedule
        row = {
                "index": int(point.index),
                "time": float(point.time),
                "energy_expectation": float(point.energy_expectation),
                "theta_l2": float(np.linalg.norm(theta)),
                "theta_dot_l2": float(np.linalg.norm(theta_dot)),
                "mclachlan_gamma": float(point.fixed_step.gamma),
                "mclachlan_residual_sq": float(point.fixed_step.residual_sq),
                "mclachlan_residual_ratio": float(point.fixed_step.residual_ratio),
                "mclachlan_legacy_objective_residual_sq": _finite_or_none(
                    point.fixed_step.legacy_objective_residual_sq
                ),
                "mclachlan_legacy_objective_residual_ratio": _finite_or_none(
                    point.fixed_step.legacy_objective_residual_ratio
                ),
                "mclachlan_realized_residual_sq": _finite_or_none(
                    point.fixed_step.realized_residual_sq
                ),
                "mclachlan_rho_real": _finite_or_none(point.fixed_step.rho_real),
                "mclachlan_best_case_residual_sq": _finite_or_none(
                    point.fixed_step.best_case_residual_sq
                ),
                "mclachlan_rho_expr": _finite_or_none(point.fixed_step.rho_expr),
                "mclachlan_rho_num": _finite_or_none(point.fixed_step.rho_num),
                "state_velocity_l2": _finite_or_none(
                    point.fixed_step.projected_velocity_l2
                ),
                "state_motion_l2_step": _finite_or_none(
                    point.fixed_step.state_motion_l2_step
                ),
                "state_space_kink_eta": _finite_or_none(
                    point.fixed_step.state_space_kink_eta
                ),
                "rank": int(point.fixed_step.rank),
                "condition_number": (
                    None
                    if point.fixed_step.condition_number is None
                    else float(point.fixed_step.condition_number)
                ),
                "effective_pinv_rcond": float(point.fixed_step.inverse_policy.pinv_rcond),
                "effective_ridge_lambda": float(point.fixed_step.inverse_policy.ridge_lambda),
                "effective_solve_damping": float(point.fixed_step.inverse_policy.solve_damping),
                "solve_repair_enabled": bool(point.fixed_step.solve_repair_enabled),
                "solve_repair_applied": bool(point.fixed_step.solve_repair_applied),
                "solve_repair_unsupported": bool(point.fixed_step.solve_repair_unsupported),
                "solve_repair_reason": str(point.fixed_step.solve_repair_reason),
                "solve_repair_attempt_count": int(len(point.fixed_step.solve_repair_attempts)),
                "solve_repair_response_lanes": (
                    None
                    if repair_schedule is None
                    else [str(lane) for lane in repair_schedule.active_lanes]
                ),
                "solve_repair_response_severity": (
                    None if repair_schedule is None else float(repair_schedule.severity)
                ),
                "solve_repair_response_breadth": (
                    None if repair_schedule is None else int(repair_schedule.breadth)
                ),
                "solve_repair_inverse_policy_breadth": (
                    None
                    if repair_schedule is None
                    else int(repair_schedule.inverse_policy_breadth)
                ),
                "solve_repair_local_subdivision_breadth": (
                    None
                    if repair_schedule is None
                    else int(repair_schedule.local_subdivision_breadth)
                ),
                "solve_mode": str(point.fixed_step.solve_mode),
                "solve_guard_g_empty": bool(point.fixed_step.solve_guard_g_empty),
                "solve_guard_g_kappa": bool(point.fixed_step.solve_guard_g_kappa),
                "solve_guard_g_delta": bool(point.fixed_step.solve_guard_g_delta),
                "solve_guard_g_rho": bool(point.fixed_step.solve_guard_g_rho),
                "solve_guard_g_kink": bool(point.fixed_step.solve_guard_g_kink),
                "integration_local_subdivision_applied": (
                    None if integration is None else bool(integration.local_subdivision_applied)
                ),
                "integration_local_subdivision_depth": (
                    None if integration is None else int(integration.local_subdivision_depth)
                ),
                "integration_local_substep_count": (
                    None if integration is None else int(integration.local_substep_count)
                ),
                "integration_local_subdivision_reason": (
                    None
                    if integration is None or integration.local_subdivision_reason is None
                    else str(integration.local_subdivision_reason)
                ),
                "integration_rhs_evaluation_count": (
                    None if integration is None else int(integration.rhs_evaluation_count)
                ),
                "integration_prospective_state_motion_l2_step_initial": _finite_or_none(
                    integration_repair_summary.get(
                        "prospective_state_motion_l2_step_initial"
                    )
                ),
                "integration_max_prospective_state_motion_l2_step": _finite_or_none(
                    integration_repair_summary.get(
                        "max_prospective_state_motion_l2_step"
                    )
                ),
                "integration_prospective_state_motion_above_max": bool(
                    integration_repair_summary.get(
                        "prospective_state_motion_above_max",
                        False,
                    )
                ),
                "integration_prospective_state_motion_triggered": bool(
                    integration_repair_summary.get(
                        "prospective_state_motion_triggered",
                        False,
                    )
                ),
                "runtime_parameter_count": int(point.runtime_parameter_count),
                "logical_parameter_count": int(point.logical_parameter_count),
                "patch_kind": str(decision.patch_kind),
                "patch_accepted": bool(decision.accepted),
                "patch_selected_label": decision.selected_label,
                "patch_reason": str(decision.reason),
                "patch_candidate_count": int(decision.candidate_count),
                "patch_scored_count": int(decision.scored_count),
                "patch_batch_reason": None if batch is None else str(batch.reason),
                "patch_batch_selected_index": (
                    None if batch is None or batch.selected_index is None else int(batch.selected_index)
                ),
                "patch_batch_score_count": (
                    0 if batch is None else int(len(batch.candidate_scores))
                ),
                "patch_selection_policy": (
                    None if batch is None else str(batch.selection_policy)
                ),
                "patch_inserted_count": (
                    0 if selected_score is None else int(selected_score.inserted_count)
                ),
                "patch_appended_count": (
                    0 if selected_score is None else int(selected_score.inserted_count)
                ),
                "patch_deleted_count": (
                    0
                    if selected_score is None
                    else int(len(selected_score.removed_runtime_indices))
                ),
                "patch_removed_count": (
                    0
                    if selected_score is None
                    else int(len(selected_score.removed_runtime_indices))
                ),
                "support_patch_deleted_count": (
                    0
                    if selected_score is None
                    else int(len(selected_score.removed_runtime_indices))
                ),
                "support_patch_removed_count": (
                    0
                    if selected_score is None
                    else int(len(selected_score.removed_runtime_indices))
                ),
                "patch_rung_count": (
                    0 if batch is None else int(len(batch.rung_diagnostics))
                ),
                "patch_selected_rung_size": (
                    None
                    if selected_candidate is None
                    else selected_candidate.metadata.get("rung_size")
                ),
                "patch_append_ladder_mode": (
                    None
                    if batch is None
                    else batch.metadata.get("append_ladder_mode")
                ),
                "append_macro_scout_enabled": bool(
                    batch_metadata.get("macro_scout_enabled", False)
                ),
                "append_macro_scout_score_mode": batch_metadata.get(
                    "macro_scout_score_mode"
                ),
                "append_macro_scout_applied": bool(
                    batch_metadata.get("macro_scout_applied", False)
                ),
                "append_macro_scout_reason": batch_metadata.get("macro_scout_reason"),
                "append_macro_scout_fail_open_applied": bool(
                    batch_metadata.get("macro_scout_fail_open_applied", False)
                ),
                "append_macro_scout_exchange_fail_open_applied": bool(
                    batch_metadata.get(
                        "macro_scout_exchange_fail_open_applied", False
                    )
                ),
                "append_macro_scout_exchange_fail_open_frontier_preserved": bool(
                    batch_metadata.get(
                        "macro_scout_exchange_fail_open_frontier_preserved", False
                    )
                ),
                "append_macro_scout_exchange_filtering_diagnostic_only": bool(
                    batch_metadata.get(
                        "macro_scout_exchange_filtering_diagnostic_only", False
                    )
                ),
                "append_macro_scout_exchange_filtering_certification": (
                    batch_metadata.get(
                        "macro_scout_exchange_filtering_certification"
                    )
                ),
                "append_macro_scout_parent_count_total": (
                    None
                    if batch_metadata.get("macro_scout_parent_count_total") is None
                    else int(batch_metadata.get("macro_scout_parent_count_total"))
                ),
                "append_macro_scout_parent_count_scored": (
                    None
                    if batch_metadata.get("macro_scout_parent_count_scored") is None
                    else int(batch_metadata.get("macro_scout_parent_count_scored"))
                ),
                "append_macro_scout_parent_count_selected": (
                    None
                    if batch_metadata.get("macro_scout_parent_count_selected") is None
                    else int(batch_metadata.get("macro_scout_parent_count_selected"))
                ),
                "append_macro_scout_child_count_before": (
                    None
                    if batch_metadata.get("macro_scout_child_count_before") is None
                    else int(batch_metadata.get("macro_scout_child_count_before"))
                ),
                "append_macro_scout_child_count_after": (
                    None
                    if batch_metadata.get("macro_scout_child_count_after") is None
                    else int(batch_metadata.get("macro_scout_child_count_after"))
                ),
                "append_macro_scout_diagnostic_full_child_set_scoring": bool(
                    batch_metadata.get(
                        "macro_scout_diagnostic_full_child_set_scoring", False
                    )
                ),
                "append_macro_scout_measurement_saving_score_available": bool(
                    batch_metadata.get(
                        "macro_scout_measurement_saving_score_available", False
                    )
                ),
                "patch_insertion_gain": (
                    None
                    if selected_score is None or selected_score.insertion_gain is None
                    else float(selected_score.insertion_gain)
                ),
                "patch_append_gain": (
                    None
                    if selected_score is None or selected_score.insertion_gain is None
                    else float(selected_score.insertion_gain)
                ),
                "patch_deletion_loss": (
                    None
                    if selected_score is None or selected_score.deletion_loss is None
                    else float(selected_score.deletion_loss)
                ),
                "patch_rank_score": (
                    None
                    if selected_candidate is None or selected_candidate.rank_score is None
                    else float(selected_candidate.rank_score)
                ),
                "patch_raw_support_rank_score": (
                    None
                    if selected_score is None or selected_score.rank_score is None
                    else float(selected_score.rank_score)
                ),
                "patch_rank_utility": (
                    _finite_or_none(selected_rank_payload.get("rank_utility"))
                    if selected_rank_payload
                    else (
                        None
                        if selected_candidate is None or selected_candidate.rank_score is None
                        else float(selected_candidate.rank_score)
                    )
                ),
                "patch_rank_score_kind": selected_rank_payload.get("rank_score_kind"),
                "patch_cost_model_effective": selected_rank_payload.get("cost_model_effective"),
                "patch_cost_normalization_mode": selected_rank_payload.get(
                    "cost_normalization_mode"
                ),
                "patch_hardware_cost_denominator": _finite_or_none(
                    selected_cost.get("hardware_cost_denominator")
                ),
                "patch_hardware_cost_excess_sum": _finite_or_none(
                    selected_cost.get("hardware_cost_excess_sum")
                ),
                "patch_append_cost_alpha": _finite_or_none(
                    selected_cost.get("append_cost_alpha")
                ),
                "patch_utility_denominator": _finite_or_none(
                    selected_cost.get("utility_denominator")
                ),
                "patch_cost_lambda_source": selected_cost.get("lambda_source"),
                "patch_cost_raw_2q": _finite_or_none(selected_cost_raw.get("2q")),
                "patch_cost_raw_d": _finite_or_none(selected_cost_raw.get("d")),
                "patch_cost_raw_1q": _finite_or_none(selected_cost_raw.get("1q")),
                "patch_cost_raw_theta": _finite_or_none(
                    selected_cost_raw.get("theta")
                ),
                "patch_cost_raw_shot": _finite_or_none(selected_cost_raw.get("shot")),
                "patch_cost_bar_2q": _finite_or_none(selected_cost_bar.get("2q")),
                "patch_cost_bar_d": _finite_or_none(selected_cost_bar.get("d")),
                "patch_cost_bar_1q": _finite_or_none(selected_cost_bar.get("1q")),
                "patch_cost_bar_theta": _finite_or_none(
                    selected_cost_bar.get("theta")
                ),
                "patch_cost_bar_shot": _finite_or_none(selected_cost_bar.get("shot")),
                "patch_cost_lambda_2q": _finite_or_none(selected_cost_lambdas.get("2q")),
                "patch_cost_lambda_d": _finite_or_none(selected_cost_lambdas.get("d")),
                "patch_cost_lambda_1q": _finite_or_none(selected_cost_lambdas.get("1q")),
                "patch_cost_lambda_theta": _finite_or_none(
                    selected_cost_lambdas.get("theta")
                ),
                "patch_cost_lambda_shot": _finite_or_none(
                    selected_cost_lambdas.get("shot")
                ),
                "patch_prune_loss_full": (
                    None
                    if selected_candidate is None
                    else _finite_or_none(
                        (selected_candidate.metadata or {}).get("deletion_loss_full")
                    )
                ),
                "patch_prune_historical_loss": _finite_or_none(
                    selected_prune_cost.get("historical_deletion_loss")
                ),
                "patch_prune_history_count": (
                    None
                    if not selected_prune_cost
                    else int(selected_prune_cost.get("history_count", 0))
                ),
                "patch_prune_cost_pressure": _finite_or_none(
                    selected_prune_cost.get("saved_cost_pressure")
                ),
                "patch_prune_rank_score_kind": selected_prune_cost.get(
                    "rank_score_kind"
                ),
                "patch_prune_utility_denominator": _finite_or_none(
                    selected_prune_cost.get("utility_denominator")
                ),
                "patch_prune_conditioning_multiplier": _finite_or_none(
                    selected_prune_cost.get("conditioning_pressure_multiplier")
                ),
                "patch_prune_conditioning_damage_penalty": _finite_or_none(
                    selected_prune_cost.get("conditioning_damage_penalty")
                ),
                "patch_prune_d_kappa_rel": _finite_or_none(
                    selected_prune_conditioning.get("d_kappa_rel")
                ),
                "patch_prune_d_schur": _finite_or_none(
                    selected_prune_conditioning.get("d_schur")
                ),
                "patch_prune_d_kappa_schur_hist": _finite_or_none(
                    selected_prune_conditioning.get("d_kappa_schur_hist")
                ),
                "patch_prune_d_kappa_dam": _finite_or_none(
                    selected_prune_conditioning.get("d_kappa_dam")
                ),
                "patch_prune_cost_raw_2q": _finite_or_none(selected_prune_raw.get("2q")),
                "patch_prune_cost_raw_d": _finite_or_none(selected_prune_raw.get("d")),
                "patch_prune_cost_raw_1q": _finite_or_none(selected_prune_raw.get("1q")),
                "patch_prune_cost_bar_2q": _finite_or_none(selected_prune_bar.get("2q")),
                "patch_prune_cost_bar_d": _finite_or_none(selected_prune_bar.get("d")),
                "patch_prune_cost_bar_1q": _finite_or_none(selected_prune_bar.get("1q")),
                "patch_prune_persistence_count": (
                    None
                    if selected_candidate is None
                    else (
                        selected_candidate.metadata or {}
                    ).get("prune_persistence_count")
                ),
                "patch_prune_persistence_required": (
                    None
                    if selected_candidate is None
                    else (
                        selected_candidate.metadata or {}
                    ).get("prune_persistence_required")
                ),
                "patch_prune_persistence_mode": (
                    None
                    if selected_candidate is None
                    else (
                        selected_candidate.metadata or {}
                    ).get("prune_persistence_mode")
                ),
                "patch_prune_atom_history_pass_count": (
                    None
                    if selected_candidate is None
                    else (
                        selected_candidate.metadata or {}
                    ).get("prune_atom_history_pass_count")
                ),
                "patch_prune_atom_history_total_count": (
                    None
                    if selected_candidate is None
                    else (
                        selected_candidate.metadata or {}
                    ).get("prune_atom_history_total_count")
                ),
                "patch_prune_atom_history_fraction": (
                    None
                    if selected_candidate is None
                    else (
                        selected_candidate.metadata or {}
                    ).get("prune_atom_history_fraction")
                ),
                "patch_prune_atom_history_fraction_required": (
                    None
                    if selected_candidate is None
                    else (
                        selected_candidate.metadata or {}
                    ).get("prune_atom_history_fraction_required")
                ),
                "patch_prune_history_transition": patch_metadata.get(
                    "prune_history_transition"
                ),
                "patch_prune_atom_history_preserved_count": (
                    None
                    if patch_metadata.get("prune_atom_history_preserved_count")
                    is None
                    else int(patch_metadata.get("prune_atom_history_preserved_count"))
                ),
                "patch_prune_atom_history_dropped_count": (
                    None
                    if patch_metadata.get("prune_atom_history_dropped_count") is None
                    else int(patch_metadata.get("prune_atom_history_dropped_count"))
                ),
                "patch_prune_geometry_history_cleared_due_to_support_change": (
                    None
                    if patch_metadata.get(
                        "prune_geometry_history_cleared_due_to_support_change"
                    )
                    is None
                    else bool(
                        patch_metadata.get(
                            "prune_geometry_history_cleared_due_to_support_change"
                        )
                    )
                ),
                "patch_prune_cooldown_preserved_count": (
                    None
                    if patch_metadata.get("prune_cooldown_preserved_count") is None
                    else int(patch_metadata.get("prune_cooldown_preserved_count"))
                ),
                "patch_prune_cooldown_dropped_count": (
                    None
                    if patch_metadata.get("prune_cooldown_dropped_count") is None
                    else int(patch_metadata.get("prune_cooldown_dropped_count"))
                ),
                "patch_prune_safety_reason": (
                    None
                    if str(decision.patch_kind) not in {"delete", "exchange"}
                    else str(decision.reason)
                ),
                "patch_prune_safety_commit_eligible": bool(
                    str(decision.patch_kind) in {"delete", "exchange"}
                    and bool(decision.accepted)
                ),
                "patch_prune_smoothness_status": patch_metadata.get(
                    "prune_patch_smoothness_status"
                ),
                "patch_prune_smoothness_available": (
                    None
                    if patch_metadata.get("prune_patch_smoothness_available") is None
                    else bool(patch_metadata.get("prune_patch_smoothness_available"))
                ),
                "patch_prune_smoothness_eta": _finite_or_none(
                    patch_metadata.get("prune_patch_smoothness_eta")
                ),
                "patch_prune_smoothness_eta_threshold": _finite_or_none(
                    patch_metadata.get("prune_patch_smoothness_eta_threshold")
                ),
                "patch_prune_smoothness_severity": _finite_or_none(
                    patch_metadata.get("prune_patch_smoothness_severity")
                ),
                "patch_prune_smoothness_passed": (
                    None
                    if patch_metadata.get("prune_patch_smoothness_passed") is None
                    else bool(patch_metadata.get("prune_patch_smoothness_passed"))
                ),
                "patch_prune_smoothness_deferred": bool(
                    patch_metadata.get("prune_patch_smoothness_deferred", False)
                ),
                "patch_prune_smoothness_retry_from_deferred": bool(
                    patch_metadata.get(
                        "prune_patch_smoothness_retry_from_deferred", False
                    )
                ),
                "patch_prune_smoothness_attempt_count": (
                    None
                    if patch_metadata.get("prune_patch_smoothness_attempt_count") is None
                    else int(patch_metadata.get("prune_patch_smoothness_attempt_count"))
                ),
                "patch_prune_smoothness_cooldown_steps": (
                    None
                    if patch_metadata.get("prune_patch_smoothness_cooldown_steps") is None
                    else int(patch_metadata.get("prune_patch_smoothness_cooldown_steps"))
                ),
                "patch_prune_smoothness_cooldown_until_index": (
                    None
                    if patch_metadata.get(
                        "prune_patch_smoothness_cooldown_until_index"
                    )
                    is None
                    else int(
                        patch_metadata.get(
                            "prune_patch_smoothness_cooldown_until_index"
                        )
                    )
                ),
                "patch_prune_smoothness_trend_direction": patch_metadata.get(
                    "prune_patch_smoothness_trend_direction"
                ),
                "patch_prune_smoothness_trend_slope_per_index": _finite_or_none(
                    patch_metadata.get("prune_patch_smoothness_trend_slope_per_index")
                ),
                "patch_prune_refit_mode": patch_metadata.get("prune_patch_refit_mode"),
            }
        row.update(
            observable_row_fields(
                np.asarray(point.geometry.psi, dtype=complex),
                plan=observable_plan,
            )
        )
        rows.append(row)
    return rows


def _summary_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    initial_state: Any,
    final_state: Any,
    hamiltonian: Any,
    integrator_method: str,
    inverse_policy: McLachlanInversePolicy,
    controller_config: AppendControllerConfig,
    support_patch_config: SupportPatchControllerConfig | None,
    solve_repair_config: SolveRepairConfig,
) -> dict[str, Any]:
    accepted = [row for row in rows if bool(row.get("patch_accepted", False))]
    append_ladder_enabled = bool(
        support_patch_config is not None
        and str(support_patch_config.append_ladder_mode).strip().lower() == "combinatorial"
    )
    append_ladder_mode = (
        "legacy_singleton"
        if support_patch_config is None
        else str(support_patch_config.append_ladder_mode)
    )
    residuals = [
        float(row["mclachlan_residual_ratio"])
        for row in rows
        if row.get("mclachlan_residual_ratio") is not None
    ]
    rho_nums = _finite_row_values(rows, "mclachlan_rho_num")
    rho_reals = _finite_row_values(rows, "mclachlan_rho_real")
    rho_exprs = _finite_row_values(rows, "mclachlan_rho_expr")
    state_motions = _finite_row_values(rows, "state_motion_l2_step")
    prospective_state_motions = _finite_row_values(
        rows,
        "integration_max_prospective_state_motion_l2_step",
    )
    kink_etas = _finite_row_values(rows, "state_space_kink_eta")
    prune_patch_smoothness_etas = _finite_row_values(
        rows, "patch_prune_smoothness_eta"
    )
    prune_patch_smoothness_severities = _finite_row_values(
        rows, "patch_prune_smoothness_severity"
    )
    if not rows:
        summary = {
            "point_count": 0,
            "parameterization_mode": str(initial_state.parameterization_mode),
            "parameterization_label": str(initial_state.parameterization_label),
            "active_parameter_count_initial": int(initial_state.active_parameter_count),
            "active_parameter_count_final": int(final_state.active_parameter_count),
            "runtime_parameter_count_initial": int(initial_state.runtime_parameter_count),
            "runtime_parameter_count_final": int(final_state.runtime_parameter_count),
            "runtime_pauli_parameter_count_initial": int(initial_state.runtime_pauli_parameter_count),
            "runtime_pauli_parameter_count_final": int(final_state.runtime_pauli_parameter_count),
            "logical_parameter_count_initial": int(initial_state.logical_parameter_count),
            "logical_parameter_count_final": int(final_state.logical_parameter_count),
            "controller_config": controller_config.to_json_dict(),
            "support_patch_config": (
                None
                if support_patch_config is None
                else support_patch_config.to_json_dict()
            ),
            "solve_repair_config": solve_repair_config.to_json_dict(),
            "solve_repair_enabled": bool(solve_repair_config.enabled),
            "append_ladder_enabled": append_ladder_enabled,
            "append_ladder_mode": append_ladder_mode,
            "active_prune_enabled": bool(
                support_patch_config is not None and support_patch_config.prune_enabled
            ),
            "active_prune_commit_enabled": bool(
                support_patch_config is not None
                and support_patch_config.prune_commit_enabled
            ),
            "prune_ladder_enabled": bool(
                support_patch_config is not None
                and support_patch_config.prune_enabled
                and int(support_patch_config.max_prune_batch_size) > 0
            ),
            "prune_patch_smoothness_enabled": bool(
                support_patch_config is not None
                and support_patch_config.prune_patch_smoothness_enabled
            ),
            "prune_patch_smoothness_deferred_count": 0,
            "prune_patch_smoothness_unavailable_count": 0,
            "prune_patch_smoothness_passed_count": 0,
            "prune_patch_smoothness_retry_count": 0,
            "prune_patch_smoothness_accepted_after_retry_count": 0,
        }
        summary.update(reference_energy_summary(rows))
        summary.update(reference_energy_summary(rows, field_prefix="seed_", summary_prefix="seed_"))
        return summary
    summary = {
        "point_count": int(len(rows)),
        "time_initial": float(rows[0]["time"]),
        "time_final": float(rows[-1]["time"]),
        "energy_initial": float(rows[0]["energy_expectation"]),
        "energy_final": float(rows[-1]["energy_expectation"]),
        "parameterization_mode": str(initial_state.parameterization_mode),
        "parameterization_label": str(initial_state.parameterization_label),
        "active_parameter_count_initial": int(initial_state.active_parameter_count),
        "active_parameter_count_final": int(final_state.active_parameter_count),
        "runtime_parameter_count_initial": int(initial_state.runtime_parameter_count),
        "runtime_parameter_count_final": int(final_state.runtime_parameter_count),
        "runtime_pauli_parameter_count_initial": int(initial_state.runtime_pauli_parameter_count),
        "runtime_pauli_parameter_count_final": int(final_state.runtime_pauli_parameter_count),
        "logical_parameter_count_initial": int(initial_state.logical_parameter_count),
        "logical_parameter_count_final": int(final_state.logical_parameter_count),
        "candidate_pool_term_count": int(len(initial_state.candidate_pool_terms)),
        "candidate_pool_complete": bool(initial_state.can_structural_edit),
        "drive_enabled": bool(hamiltonian.drive_enabled),
        "integrator_method": str(integrator_method).lower(),
        "pinv_rcond": float(inverse_policy.pinv_rcond),
        "ridge_lambda": float(inverse_policy.ridge_lambda),
        "solve_damping": float(inverse_policy.solve_damping),
        "solve_repair_config": solve_repair_config.to_json_dict(),
        "solve_repair_enabled": bool(solve_repair_config.enabled),
        "solve_repair_applied_count": int(
            sum(1 for row in rows if bool(row.get("solve_repair_applied", False)))
        ),
        "solve_repair_unsupported_count": int(
            sum(1 for row in rows if bool(row.get("solve_repair_unsupported", False)))
        ),
        "solve_guard_g_empty_count": int(
            sum(1 for row in rows if bool(row.get("solve_guard_g_empty", False)))
        ),
        "solve_guard_g_kappa_count": int(
            sum(1 for row in rows if bool(row.get("solve_guard_g_kappa", False)))
        ),
        "solve_guard_g_delta_count": int(
            sum(1 for row in rows if bool(row.get("solve_guard_g_delta", False)))
        ),
        "solve_guard_g_rho_count": int(
            sum(1 for row in rows if bool(row.get("solve_guard_g_rho", False)))
        ),
        "solve_guard_g_kink_count": int(
            sum(1 for row in rows if bool(row.get("solve_guard_g_kink", False)))
        ),
        "local_subdivision_applied_count": int(
            sum(
                1
                for row in rows
                if bool(row.get("integration_local_subdivision_applied", False))
            )
        ),
        "max_mclachlan_rho_num": None if not rho_nums else float(max(rho_nums)),
        "max_mclachlan_rho_real": None if not rho_reals else float(max(rho_reals)),
        "max_mclachlan_rho_expr": None if not rho_exprs else float(max(rho_exprs)),
        "max_state_motion_l2_step": (
            None if not state_motions else float(max(state_motions))
        ),
        "max_prospective_state_motion_l2_step": (
            None
            if not prospective_state_motions
            else float(max(prospective_state_motions))
        ),
        "prospective_state_motion_trigger_count": int(
            sum(
                1
                for row in rows
                if bool(
                    row.get("integration_prospective_state_motion_triggered", False)
                )
            )
        ),
        "max_state_space_kink_eta": None if not kink_etas else float(max(kink_etas)),
        "accepted_patch_count": int(len(accepted)),
        "accepted_append_count": int(
            sum(1 for row in accepted if _is_append_patch_kind(row.get("patch_kind")))
        ),
        "accepted_insert_count": int(
            sum(1 for row in accepted if _is_append_patch_kind(row.get("patch_kind")))
        ),
        "accepted_appended_coordinate_count": int(
            sum(
                int(row.get("patch_appended_count") or row.get("patch_inserted_count") or 0)
                for row in accepted
                if _is_append_patch_kind(row.get("patch_kind"))
            )
        ),
        "accepted_inserted_coordinate_count": int(
            sum(
                int(row.get("patch_appended_count") or row.get("patch_inserted_count") or 0)
                for row in accepted
                if _is_append_patch_kind(row.get("patch_kind"))
            )
        ),
        "accepted_delete_count": int(
            sum(1 for row in accepted if str(row.get("patch_kind")) == "delete")
        ),
        "accepted_exchange_count": int(
            sum(1 for row in accepted if str(row.get("patch_kind")) == "exchange")
        ),
        "accepted_deleted_coordinate_count": int(
            sum(
                int(row.get("patch_deleted_count") or row.get("patch_removed_count") or 0)
                for row in accepted
                if str(row.get("patch_kind")) == "delete"
            )
        ),
        "accepted_exchange_appended_coordinate_count": int(
            sum(
                int(row.get("patch_appended_count") or row.get("patch_inserted_count") or 0)
                for row in accepted
                if str(row.get("patch_kind")) == "exchange"
            )
        ),
        "accepted_exchange_deleted_coordinate_count": int(
            sum(
                int(row.get("patch_deleted_count") or row.get("patch_removed_count") or 0)
                for row in accepted
                if str(row.get("patch_kind")) == "exchange"
            )
        ),
        "active_prune_enabled": bool(
            support_patch_config is not None and support_patch_config.prune_enabled
        ),
        "active_prune_commit_enabled": bool(
            support_patch_config is not None and support_patch_config.prune_commit_enabled
        ),
        "prune_ladder_enabled": bool(
            support_patch_config is not None
            and support_patch_config.prune_enabled
            and int(support_patch_config.max_prune_batch_size) > 0
        ),
        "prune_patch_smoothness_enabled": bool(
            support_patch_config is not None
            and support_patch_config.prune_patch_smoothness_enabled
        ),
        "prune_patch_smoothness_deferred_count": int(
            sum(
                1
                for row in rows
                if bool(row.get("patch_prune_smoothness_deferred", False))
            )
        ),
        "prune_patch_smoothness_unavailable_count": int(
            sum(
                1
                for row in rows
                if row.get("patch_prune_smoothness_available") is False
            )
        ),
        "prune_patch_smoothness_passed_count": int(
            sum(
                1
                for row in rows
                if row.get("patch_prune_smoothness_passed") is True
            )
        ),
        "prune_patch_smoothness_retry_count": int(
            sum(
                1
                for row in rows
                if bool(row.get("patch_prune_smoothness_retry_from_deferred", False))
            )
        ),
        "prune_patch_smoothness_accepted_after_retry_count": int(
            sum(
                1
                for row in rows
                if bool(row.get("patch_accepted", False))
                and str(row.get("patch_kind")) in {"delete", "exchange"}
                and bool(row.get("patch_prune_smoothness_retry_from_deferred", False))
            )
        ),
        "max_prune_patch_smoothness_eta": (
            None
            if not prune_patch_smoothness_etas
            else float(max(prune_patch_smoothness_etas))
        ),
        "max_prune_patch_smoothness_severity": (
            None
            if not prune_patch_smoothness_severities
            else float(max(prune_patch_smoothness_severities))
        ),
        "prune_scored_candidate_count": int(
            sum(
                int(row.get("patch_scored_count") or 0)
                for row in rows
                if str(row.get("patch_kind")) in {"delete", "exchange"}
            )
        ),
        "prune_commit_disabled_selected_count": int(
            sum(
                1
                for row in rows
                if str(row.get("patch_kind")) in {"delete", "exchange"}
                and str(row.get("patch_reason")) == "prune_commit_disabled"
            )
        ),
        "prune_persistence_wait_count": int(
            sum(
                1
                for row in rows
                if str(row.get("patch_kind")) in {"delete", "exchange"}
                and str(row.get("patch_reason")) == "prune_persistence_wait"
            )
        ),
        "prune_safety_rejection_count": int(
            sum(
                1
                for row in rows
                if str(row.get("patch_kind")) in {"delete", "exchange"}
                and not bool(row.get("patch_accepted", False))
                and str(row.get("patch_reason"))
                not in {
                    "prune_commit_disabled",
                    "prune_persistence_wait",
                    "no_finite_prune_ladder_score",
                    "prune_loss_above_threshold",
                }
            )
        ),
        "max_prune_deletion_loss_full": (
            None
            if not _finite_row_values(rows, "patch_prune_loss_full")
            else float(max(_finite_row_values(rows, "patch_prune_loss_full")))
        ),
        "max_prune_rank_score": (
            None
            if not [
                float(row["patch_rank_score"])
                for row in rows
                if str(row.get("patch_kind")) in {"delete", "exchange"}
                and row.get("patch_rank_score") is not None
            ]
            else float(
                max(
                    float(row["patch_rank_score"])
                    for row in rows
                    if str(row.get("patch_kind")) in {"delete", "exchange"}
                    and row.get("patch_rank_score") is not None
                )
            )
        ),
        "accepted_patch_labels": [
            str(row.get("patch_selected_label"))
            for row in accepted
            if row.get("patch_selected_label") is not None
        ],
        "max_mclachlan_residual_ratio": (
            None if not residuals else float(max(residuals))
        ),
        "final_mclachlan_residual_ratio": (
            None if not residuals else float(residuals[-1])
        ),
        "controller_config": controller_config.to_json_dict(),
        "support_patch_config": (
            None
            if support_patch_config is None
            else support_patch_config.to_json_dict()
        ),
        "append_ladder_enabled": append_ladder_enabled,
        "append_ladder_mode": append_ladder_mode,
        "append_selection_policy": (
            None
            if not append_ladder_enabled
            else APPEND_LADDER_SELECTION_POLICY_V1
        ),
        "support_patch_config_scope": (
            None
            if not append_ladder_enabled
            else (
                "append_prune_ladder_fields"
                if support_patch_config is not None and support_patch_config.prune_enabled
                else "append_ladder_fields_only"
            )
        ),
    }
    summary.update(reference_energy_summary(rows))
    summary.update(reference_energy_summary(rows, field_prefix="seed_", summary_prefix="seed_"))
    return summary


def _is_append_patch_kind(value: Any) -> bool:
    return str(value).strip().lower() in {"append", "insert"}


def _load_runtime_input_or_raise(
    artifact_path: Path,
    *,
    loader_mode: str | None,
    tag: str | None,
    generator_family: str,
    fallback_family: str,
    replay_candidate_pool_mode: str | None,
) -> Any:
    if replay_candidate_pool_mode not in {None, ""}:
        payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"Expected JSON object payload at {artifact_path}.")
        payload = dict(payload)
        payload["replay_candidate_pool_mode"] = str(replay_candidate_pool_mode)
        return load_scaffold_runtime_input_from_payload(
            payload,
            artifact_json=artifact_path,
            loader_mode=loader_mode,
            tag=tag,
            generator_family=str(generator_family),
            fallback_family=str(fallback_family),
        )
    return load_scaffold_runtime_input(
        artifact_path,
        loader_mode=loader_mode,
        tag=tag,
        generator_family=str(generator_family),
        fallback_family=str(fallback_family),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run append-first AP-McLachlan from a static scaffold artifact."
    )
    parser.add_argument("--artifact-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--loader-mode", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--generator-family", default="match_adapt")
    parser.add_argument("--fallback-family", default="full_meta")
    parser.add_argument("--replay-candidate-pool-mode", default=None)
    parser.add_argument(
        "--diagnostic-append-pool-mode",
        choices=("none", "replay_family_pool"),
        default="none",
        help=(
            "Diagnostic AP append-only override. replay_family_pool exposes the "
            "loader replay family pool to support-patch append decisions even "
            "when the imported artifact marks the default candidate pool as "
            "selected-only."
        ),
    )
    parser.add_argument(
        "--normalized-candidate-pool-profile",
        choices=("none", *sorted(NORMALIZED_POOL_PROFILES)),
        default="none",
        help=(
            "Diagnostic comparison contract for the future append pool. "
            "hamiltonian_drive_pauli locks the unique Hamiltonian/drive Pauli "
            "set; full_meta_pauli_children locks the deduplicated replay/full-meta "
            "Pauli-child set. Selected seed support is unchanged."
        ),
    )
    parser.add_argument("--times", default=None, help="Comma-separated time grid. Overrides --t-final/--num-times.")
    parser.add_argument("--t-final", type=float, default=0.2)
    parser.add_argument("--num-times", type=int, default=3)
    parser.add_argument("--integrator", choices=SUPPORTED_INTEGRATORS, default=INTEGRATOR_EULER)
    parser.add_argument("--pinv-rcond", type=float, default=1.0e-10)
    parser.add_argument("--ridge-lambda", type=float, default=DEFAULT_MCLACHLAN_RIDGE_LAMBDA)
    parser.add_argument("--solve-damping", type=float, default=DEFAULT_MCLACHLAN_SOLVE_DAMPING)
    parser.add_argument(
        "--parameterization-mode",
        choices=AP_SUPPORTED_PARAMETERIZATION_MODES,
        default=AP_PARAMETERIZATION_PER_PAULI_TERM,
        help=(
            "AP variational coordinate mode: per_pauli_term is per Pauli/polynomial "
            "term; logical_shared is per logical/macro generator."
        ),
    )
    parser.add_argument("--max-append-candidates", type=int, default=8)
    parser.add_argument("--append-min-time", type=float, default=0.0)
    parser.add_argument(
        "--residual-ratio-threshold",
        type=float,
        default=DEFAULT_APPEND_RESIDUAL_RATIO_THRESHOLD,
    )
    parser.add_argument("--min-logical-parameter-count", type=int, default=1)
    parser.add_argument("--require-complete-candidate-pool", action="store_true")
    parser.add_argument(
        "--append-occurrence-policy",
        choices=APPEND_OCCURRENCE_POLICIES,
        default=APPEND_OCCURRENCE_POLICY_LAYER_REUSE,
        help=(
            "layer_reuse lets a pool atom appear again at a later ANZATS layer "
            "while keeping one occurrence per proposed batch; unique_support is "
            "the one-use compatibility policy."
        ),
    )
    parser.add_argument("--max-append-batch-size", type=int, default=10)
    parser.add_argument("--append-schur-max-condition-number", type=float, default=1.0e12)
    parser.add_argument("--append-cost-alpha", type=float, default=1.0)
    parser.add_argument(
        "--append-cost-normalization-mode",
        default="family_robust_v1",
        choices=("family_robust_v1", "raw_legacy_v1"),
    )
    parser.add_argument("--append-cost-lambda-2q", type=float, default=0.05)
    parser.add_argument("--append-cost-lambda-d", type=float, default=0.05)
    parser.add_argument("--append-cost-lambda-1q", type=float, default=0.025)
    parser.add_argument("--append-cost-lambda-theta", type=float, default=0.0)
    parser.add_argument("--append-cost-lambda-shot", type=float, default=0.02)
    parser.add_argument("--append-cost-scale-floor", type=float, default=1.0e-12)
    parser.add_argument("--prune-cooldown-steps", type=int, default=2)
    parser.add_argument("--min-runtime-parameter-count", type=int, default=1)
    parser.add_argument("--prune-cost-alpha", type=float, default=1.0)
    parser.add_argument("--eps-loss", type=float, default=1.0e-14)
    parser.add_argument("--prune-ray-distance-tol", type=float, default=5.0e-2)
    parser.add_argument("--prune-history-window", type=int, default=3)
    parser.add_argument("--prune-history-lambda", type=float, default=1.0)
    parser.add_argument(
        "--prune-condition-lambda-kappa-rel", type=float, default=0.0
    )
    parser.add_argument(
        "--prune-condition-lambda-kappa-dam", type=float, default=0.0
    )
    parser.add_argument(
        "--certification-refit",
        action="store_true",
        help=(
            "Enable the bounded local trust-region refit of materialized "
            "finalists toward the frozen checkpoint ray before commit gates."
        ),
    )
    parser.add_argument(
        "--certification-refit-trust-radius", type=float, default=0.1
    )
    parser.add_argument(
        "--certification-refit-max-iterations", type=int, default=15
    )
    parser.add_argument(
        "--max-certification-attempts-per-level",
        type=int,
        default=None,
        help=(
            "Bound how many finalists one selector level may materialize "
            "before the level is declared exhausted (None = unbounded)."
        ),
    )
    parser.add_argument("--prune-patch-smoothness-eta-max", type=float, default=1.0e-3)
    parser.add_argument("--patch-utility-delta-weight", type=float, default=1.0)
    parser.add_argument(
        "--max-insertion-batch-size",
        type=int,
        default=None,
        help=(
            "Upper bound on inserted child occurrences in one structural "
            "patch. None falls back to --max-append-batch-size. Zero leaves "
            "stay plus pure deletion."
        ),
    )
    parser.add_argument(
        "--interaction-frontier-widths",
        default=None,
        help=(
            "Comma-separated strictly increasing child-frontier widths for "
            "multi-child insertion plans; default None resolves to "
            "2,4,8,...,|eligible universe|."
        ),
    )
    parser.add_argument(
        "--structural-score-floor",
        type=float,
        default=0.0,
        help=(
            "Structural score a candidate must exceed to reach certification "
            "(tau_score). The floor, not certification, excludes numerical-"
            "noise candidates such as no-op zero-angle insertions."
        ),
    )
    parser.add_argument(
        "--max-joint-patch-evaluations",
        type=int,
        default=None,
        help=(
            "Sole computational cap on structural enumeration: a complete "
            "deletion rung or insertion frontier is admitted only when the "
            "cumulative unique-candidate count stays within this budget; "
            "families are never partially sampled. None disables the guard "
            "and is intended for small systems and oracles only."
        ),
    )
    parser.add_argument(
        "--support-patch-scoring-workers",
        type=int,
        default=2,
        help=(
            "Classical support-patch candidate scoring workers. Threads cover "
            "append/prune/exchange candidate scoring only, and reduce in "
            "canonical task order, so worker count never changes a decision; "
            "1 remains the serial reference execution of the same code path. "
            "The default of 2 is calibrated rather than assumed: on an 8-core "
            "Apple-Accelerate host one L=2 conditioning-stress trajectory "
            "measured 10.7s at 1 worker, 8.5s at 2, 9.3s at 4 and 10.3s at 8, "
            "because candidate scoring is GIL-bound and extra workers cost "
            "more contention than they recover. Recalibrate for another host "
            "or problem size with "
            "pipelines.time_dynamics.diagnostics.ap_runtime_benchmark."
        ),
    )
    parser.add_argument(
        "--solve-repair",
        action="store_true",
        help=(
            "Enable the AP-McLachlan Paper-II solve-repair candidate set. "
            "Finite diagnostic runs continue with unsupported telemetry when "
            "no candidate satisfies the state-space acceptability predicate."
        ),
    )
    parser.add_argument("--solve-repair-condition-number-max", type=float, default=1.0e6)
    parser.add_argument("--solve-repair-condition-number-fail", type=float, default=None)
    parser.add_argument("--solve-repair-strict-finite-shot-validation", action="store_true")
    parser.add_argument("--solve-repair-theta-dot-l2-max", type=float, default=None)
    parser.add_argument("--solve-repair-rho-num-max", type=float, default=1.0e-2)
    parser.add_argument("--solve-repair-state-motion-l2-step-max", type=float, default=5.0e-2)
    parser.add_argument("--solve-repair-kink-eta-max", type=float, default=1.0e-2)
    parser.add_argument(
        "--solve-repair-local-subdivision",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--solve-repair-max-local-subdivisions", type=int, default=4)
    parser.add_argument("--solve-repair-local-subdivision-factor", type=int, default=2)
    parser.add_argument("--solve-repair-min-local-dt", type=float, default=1.0e-6)
    parser.add_argument("--solve-repair-release-patience-min", type=int, default=1)
    parser.add_argument("--solve-repair-release-patience-max", type=int, default=5)
    parser.add_argument("--solve-repair-release-kink-threshold-scale", type=float, default=0.5)
    parser.add_argument("--solve-repair-release-kink-severity-scale", type=float, default=4.0)
    parser.add_argument(
        "--solve-repair-ridge-ladder",
        default="1e-7,3e-8,1e-8,0,3e-7,1e-6,3e-6,1e-5",
    )
    parser.add_argument(
        "--solve-repair-pinv-rcond-ladder",
        default="1e-10,1e-11,1e-12,1e-9,1e-8,1e-7",
    )
    parser.add_argument(
        "--solve-repair-damping-ladder",
        default="0",
    )
    parser.add_argument("--enable-drive", action="store_true")
    parser.add_argument("--drive-A", type=float, default=0.0)
    parser.add_argument("--drive-omega", type=float, default=1.0)
    parser.add_argument("--drive-tbar", type=float, default=1.0)
    parser.add_argument("--drive-phi", type=float, default=0.0)
    parser.add_argument("--drive-pattern", default="staggered")
    parser.add_argument("--drive-custom-weights", default=None)
    parser.add_argument("--drive-include-identity", action="store_true")
    parser.add_argument("--drive-time-sampling", default="midpoint")
    parser.add_argument("--drive-t0", type=float, default=0.0)
    parser.add_argument("--drive-n-sites", type=int, default=None)
    parser.add_argument("--drive-ordering", default=None)
    parser.add_argument(
        "--drive-aligned-ansatz",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For driven AP-McLachlan, append the resolved drive operator as a "
            "zero-angle ansatz generator before append/prune decisions."
        ),
    )
    parser.add_argument("--reference-energy-json", default=None)
    parser.add_argument("--reference-energy-atol", type=float, default=1.0e-12)
    parser.add_argument(
        "--seed-reference-energy-json",
        default=None,
        help=(
            "Optional same-seed exact propagation reference for reporting only. "
            "This never enters AP decisions."
        ),
    )
    parser.add_argument("--seed-reference-energy-atol", type=float, default=1.0e-12)
    parser.add_argument(
        "--progress-log-every",
        type=int,
        default=0,
        help=(
            "Print report-only AP progress diagnostics every N checkpoints; "
            "0 disables progress logging."
        ),
    )
    parser.add_argument(
        "--progress-log-events",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When progress logging is enabled, also print accepted support-patch "
            "event checkpoints."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    reject_removed_online_redundancy_flags(raw_argv)
    parser = _build_parser()
    args = parser.parse_args(raw_argv)
    artifact_path = Path(args.artifact_json)
    try:
        replay_candidate_pool_mode = args.replay_candidate_pool_mode
        if str(args.diagnostic_append_pool_mode) == "replay_family_pool":
            if replay_candidate_pool_mode not in {None, "", "diagnostic_replay_family_pool", "family_pool", "replay_family_pool"}:
                raise ValueError(
                    "--diagnostic-append-pool-mode replay_family_pool conflicts "
                    f"with --replay-candidate-pool-mode={replay_candidate_pool_mode!r}."
                )
            replay_candidate_pool_mode = "diagnostic_replay_family_pool"
        runtime_input = _load_runtime_input_or_raise(
            artifact_path,
            loader_mode=args.loader_mode,
            tag=args.tag,
            generator_family=str(args.generator_family),
            fallback_family=str(args.fallback_family),
            replay_candidate_pool_mode=replay_candidate_pool_mode,
        )
        drive_config = _drive_config_from_args(args, runtime_input)
        controller_config = AppendControllerConfig(
            max_append_candidates=int(args.max_append_candidates),
            append_min_time=float(args.append_min_time),
            residual_ratio_threshold=float(args.residual_ratio_threshold),
            min_logical_parameter_count=int(args.min_logical_parameter_count),
            allow_incomplete_candidate_pool=not bool(args.require_complete_candidate_pool),
        )
        support_patch_config = SupportPatchControllerConfig(
            append_ladder_mode="combinatorial",
            append_occurrence_policy=str(args.append_occurrence_policy),
            max_append_batch_size=int(args.max_append_batch_size),
            max_insertion_batch_size=(
                None
                if args.max_insertion_batch_size is None
                else int(args.max_insertion_batch_size)
            ),
            interaction_frontier_widths=(
                None
                if args.interaction_frontier_widths in (None, "")
                else tuple(
                    int(token)
                    for token in str(args.interaction_frontier_widths).split(",")
                    if token.strip()
                )
            ),
            structural_score_floor=float(args.structural_score_floor),
            max_joint_patch_evaluations=(
                None
                if args.max_joint_patch_evaluations is None
                else int(args.max_joint_patch_evaluations)
            ),
            append_schur_max_condition_number=float(
                args.append_schur_max_condition_number
            ),
            support_patch_scoring_workers=int(args.support_patch_scoring_workers),
            cost_normalization_mode=str(args.append_cost_normalization_mode),
            append_cost_alpha=float(args.append_cost_alpha),
            append_cost_lambda_2q=float(args.append_cost_lambda_2q),
            append_cost_lambda_d=float(args.append_cost_lambda_d),
            append_cost_lambda_1q=float(args.append_cost_lambda_1q),
            append_cost_lambda_theta=float(args.append_cost_lambda_theta),
            append_cost_lambda_shot=float(args.append_cost_lambda_shot),
            append_cost_scale_floor=float(args.append_cost_scale_floor),
            append_min_time=float(args.append_min_time),
            residual_ratio_threshold=float(args.residual_ratio_threshold),
            prune_cooldown_steps=int(args.prune_cooldown_steps),
            min_runtime_parameter_count=int(args.min_runtime_parameter_count),
            prune_ray_distance_tol=float(args.prune_ray_distance_tol),
            prune_patch_smoothness_eta_max=float(
                args.prune_patch_smoothness_eta_max
            ),
            patch_utility_delta_weight=float(args.patch_utility_delta_weight),
            prune_cost_alpha=float(args.prune_cost_alpha),
            prune_history_window=int(args.prune_history_window),
            prune_history_lambda=float(args.prune_history_lambda),
            prune_condition_lambda_kappa_rel=float(
                args.prune_condition_lambda_kappa_rel
            ),
            prune_condition_lambda_kappa_dam=float(
                args.prune_condition_lambda_kappa_dam
            ),
            certification_refit_enabled=bool(args.certification_refit),
            certification_refit_trust_radius=float(
                args.certification_refit_trust_radius
            ),
            certification_refit_max_iterations=int(
                args.certification_refit_max_iterations
            ),
            max_certification_attempts_per_level=(
                None
                if args.max_certification_attempts_per_level is None
                else int(args.max_certification_attempts_per_level)
            ),
            eps_loss=float(args.eps_loss),
            cost_required_for_decisions=False,
            allow_incomplete_candidate_pool=not bool(args.require_complete_candidate_pool),
        )
        solve_repair_config = SolveRepairConfig(
            enabled=bool(args.solve_repair),
            condition_number_max=float(args.solve_repair_condition_number_max),
            condition_number_fail=(
                None
                if args.solve_repair_condition_number_fail is None
                else float(args.solve_repair_condition_number_fail)
            ),
            strict_finite_shot_validation=bool(
                args.solve_repair_strict_finite_shot_validation
            ),
            theta_dot_l2_max=(
                None
                if args.solve_repair_theta_dot_l2_max is None
                else float(args.solve_repair_theta_dot_l2_max)
            ),
            rho_num_max=float(args.solve_repair_rho_num_max),
            state_motion_l2_step_max=float(args.solve_repair_state_motion_l2_step_max),
            state_space_kink_eta_max=float(args.solve_repair_kink_eta_max),
            local_subdivision_enabled=bool(args.solve_repair_local_subdivision),
            max_local_subdivisions=int(args.solve_repair_max_local_subdivisions),
            local_subdivision_factor=int(args.solve_repair_local_subdivision_factor),
            min_local_dt=float(args.solve_repair_min_local_dt),
            release_patience_min=int(args.solve_repair_release_patience_min),
            release_patience_max=int(args.solve_repair_release_patience_max),
            release_kink_threshold_scale=float(
                args.solve_repair_release_kink_threshold_scale
            ),
            release_kink_severity_scale=float(
                args.solve_repair_release_kink_severity_scale
            ),
            ridge_ladder=_parse_float_ladder(args.solve_repair_ridge_ladder),
            pinv_rcond_ladder=_parse_float_ladder(args.solve_repair_pinv_rcond_ladder),
            solve_damping_ladder=_parse_float_ladder(args.solve_repair_damping_ladder),
        )
        payload = run_append_ap_mclachlan_from_runtime_input(
            runtime_input,
            times=_parse_times(args),
            integrator_method=str(args.integrator),
            pinv_rcond=float(args.pinv_rcond),
            ridge_lambda=float(args.ridge_lambda),
            solve_damping=float(args.solve_damping),
            enable_drive=bool(args.enable_drive),
            drive_config=drive_config,
            drive_aligned_ansatz=bool(args.drive_aligned_ansatz),
            parameterization_mode=str(args.parameterization_mode),
            controller_config=controller_config,
            support_patch_config=support_patch_config,
            solve_repair_config=solve_repair_config,
            reference_energy_trajectory=(
                None
                if args.reference_energy_json in {None, ""}
                else load_reference_energy_trajectory(args.reference_energy_json)
            ),
            reference_energy_atol=float(args.reference_energy_atol),
            seed_reference_energy_trajectory=(
                None
                if args.seed_reference_energy_json in {None, ""}
                else load_reference_energy_trajectory(args.seed_reference_energy_json)
            ),
            seed_reference_energy_atol=float(args.seed_reference_energy_atol),
            progress_log_every=int(args.progress_log_every),
            progress_log_events=bool(args.progress_log_events),
            normalized_candidate_pool_profile=(
                None
                if str(args.normalized_candidate_pool_profile) == "none"
                else str(args.normalized_candidate_pool_profile)
            ),
            runner_metadata={
                "artifact_json": str(artifact_path),
                "loader_mode": args.loader_mode,
                "tag": args.tag,
                "generator_family": str(args.generator_family),
                "fallback_family": str(args.fallback_family),
                "replay_candidate_pool_mode": replay_candidate_pool_mode,
                "diagnostic_append_pool_mode": str(args.diagnostic_append_pool_mode),
                "normalized_candidate_pool_profile": str(
                    args.normalized_candidate_pool_profile
                ),
                "online_redundancy_injection_available": False,
                "parameterization_mode": str(args.parameterization_mode),
                "append_min_time": float(args.append_min_time),
                "prune_patch_smoothness_eta_max": float(
                    args.prune_patch_smoothness_eta_max
                ),
                "patch_utility_delta_weight": float(args.patch_utility_delta_weight),
                "structural_score_floor": float(args.structural_score_floor),
                "max_joint_patch_evaluations": (
                    None
                    if args.max_joint_patch_evaluations is None
                    else int(args.max_joint_patch_evaluations)
                ),
                "max_insertion_batch_size": (
                    None
                    if args.max_insertion_batch_size is None
                    else int(args.max_insertion_batch_size)
                ),
                "interaction_frontier_widths": (
                    None
                    if args.interaction_frontier_widths in (None, "")
                    else str(args.interaction_frontier_widths)
                ),
                "support_patch_scoring_workers": int(
                    args.support_patch_scoring_workers
                ),
                "solve_damping": float(args.solve_damping),
                "solve_repair_requested": bool(args.solve_repair),
                "solve_repair_config": solve_repair_config.to_json_dict(),
                "drive_aligned_ansatz_requested": bool(args.drive_aligned_ansatz),
                "reference_energy_json": args.reference_energy_json,
                "seed_reference_energy_json": args.seed_reference_energy_json,
                "progress_log_every": int(args.progress_log_every),
                "progress_log_events": bool(args.progress_log_events),
            },
        )
        payload["source_artifact_json"] = str(artifact_path)
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


def _coerce_reference_energy_trajectory(value: Any | None) -> ReferenceEnergyTrajectory | None:
    if value is None:
        return None
    if isinstance(value, ReferenceEnergyTrajectory):
        return value
    if isinstance(value, Mapping):
        return reference_energy_trajectory_from_payload(value)
    raise TypeError("reference_energy_trajectory must be a ReferenceEnergyTrajectory, mapping, or None.")


def _parse_float_ladder(raw: Any) -> tuple[float, ...]:
    if raw in {None, ""}:
        raise ValueError("repair ladder fields must not be empty.")
    values = tuple(float(chunk.strip()) for chunk in str(raw).split(",") if chunk.strip())
    if not values:
        raise ValueError("repair ladder fields must contain at least one value.")
    if any(not np.isfinite(float(v)) or float(v) < 0.0 for v in values):
        raise ValueError("repair ladder fields must contain finite non-negative values.")
    return values


def _finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _finite_row_values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _finite_or_none(row.get(key))
        if value is not None:
            values.append(value)
    return values


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        out = float(value)
    except (TypeError, ValueError):
        return str(value)
    return out if np.isfinite(out) else None


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REMOVED_ONLINE_REDUNDANCY_FLAGS",
    "REMOVED_ONLINE_REDUNDANCY_MESSAGE",
    "RUNNER_SCHEMA_V1",
    "main",
    "reject_removed_online_redundancy_flags",
    "run_append_ap_mclachlan_from_runtime_input",
]
