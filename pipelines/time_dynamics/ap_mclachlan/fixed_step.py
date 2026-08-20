"""Eq. (8) fixed-support AP-McLachlan solve."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from pipelines.time_dynamics.ap_mclachlan.geometry import (
    McLachlanGeometry,
    StateSpaceSolveMetrics,
    residual_denominator,
    state_space_kink_eta,
    state_space_solve_metrics,
)
from pipelines.time_dynamics.ap_mclachlan.inverse import (
    DEFAULT_MCLACHLAN_RIDGE_LAMBDA,
    DEFAULT_MCLACHLAN_SOLVE_DAMPING,
    McLachlanInversePolicy,
    solve_theta_dot,
)


FIXED_STEP_SCHEMA_V1 = "ap_mclachlan_fixed_step_v1"
FIXED_STEP_SCHEMA_V2 = "ap_mclachlan_fixed_step_v2"
FIXED_STEP_EQUATION_ID = "eq8_fixed_support_mclachlan"
SOLVE_REPAIR_PROFILE_V1 = "ap_mclachlan_solve_repair_ladder_v1"
SOLVE_REPAIR_PROFILE_V2 = "ap_mclachlan_state_space_solve_repair_v2"
SOLVE_REPAIR_SELECTION_STATE_SPACE_SCORE_V1 = "state_space_candidate_set_score_v1"
# Compatibility export only. The active Paper-II implementation does not use
# ordered first-passing guard lists.
SOLVE_REPAIR_SELECTION_FIRST_PASSING_V1 = "archaic_first_passing_guard_order_v1"


@dataclass(frozen=True)
class SolveGuardReport:
    """State-space solve-repair guard report for one inverse-policy attempt."""

    repair_dt: float | None
    g_empty: bool
    g_kappa: bool
    g_delta: bool
    g_rho: bool
    g_kink: bool
    retained_support_empty: bool
    state_motion_l2_step: float | None
    state_space_kink_eta: float | None
    rho_real: float | None
    rho_expr: float | None
    rho_num: float | None
    projected_velocity_l2: float | None
    realized_residual_sq: float | None
    best_case_residual_sq: float | None
    guard_reason: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "repair_dt": _finite_or_none(self.repair_dt),
            "g_empty": bool(self.g_empty),
            "g_kappa": bool(self.g_kappa),
            "g_delta": bool(self.g_delta),
            "g_rho": bool(self.g_rho),
            "g_kink": bool(self.g_kink),
            "retained_support_empty": bool(self.retained_support_empty),
            "state_motion_l2_step": _finite_or_none(self.state_motion_l2_step),
            "state_space_kink_eta": _finite_or_none(self.state_space_kink_eta),
            "rho_real": _finite_or_none(self.rho_real),
            "rho_expr": _finite_or_none(self.rho_expr),
            "rho_num": _finite_or_none(self.rho_num),
            "projected_velocity_l2": _finite_or_none(self.projected_velocity_l2),
            "realized_residual_sq": _finite_or_none(self.realized_residual_sq),
            "best_case_residual_sq": _finite_or_none(self.best_case_residual_sq),
            "guard_reason": str(self.guard_reason),
        }


@dataclass(frozen=True)
class SolveRepairAttempt:
    """One attempted inverse policy in the McLachlan solve-repair candidate set."""

    attempt_index: int
    inverse_policy: McLachlanInversePolicy
    accepted: bool
    reason: str
    theta_dot_l2: float | None = None
    condition_number: float | None = None
    residual_ratio: float | None = None
    rank: int | None = None
    error: str | None = None
    guard_report: SolveGuardReport | None = None
    nominated_by: str = ""
    attempt_kind: str = "same_checkpoint_policy_rung"
    repair_dt: float | None = None
    theta_dot_l2_limit_exceeded_diagnostic: bool | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "attempt_index": int(self.attempt_index),
            "accepted": bool(self.accepted),
            "reason": str(self.reason),
            "pinv_rcond": float(self.inverse_policy.pinv_rcond),
            "ridge_lambda": float(self.inverse_policy.ridge_lambda),
            "solve_damping": float(self.inverse_policy.solve_damping),
            "theta_dot_l2": _finite_or_none(self.theta_dot_l2),
            "condition_number": _finite_or_none(self.condition_number),
            "residual_ratio": _finite_or_none(self.residual_ratio),
            "rank": None if self.rank is None else int(self.rank),
            "error": None if self.error is None else str(self.error),
            "guard_report": (
                None if self.guard_report is None else self.guard_report.to_json_dict()
            ),
            "nominated_by": str(self.nominated_by),
            "attempt_kind": str(self.attempt_kind),
            "repair_dt": _finite_or_none(self.repair_dt),
            "theta_dot_l2_limit_exceeded_diagnostic": self.theta_dot_l2_limit_exceeded_diagnostic,
        }


@dataclass(frozen=True)
class SolveRepairResponseSchedule:
    """Paper-II repair response schedule from the base rung."""

    active_lanes: tuple[str, ...]
    severity: float
    breadth: int
    inverse_policy_breadth: int
    local_subdivision_breadth: int

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "active_lanes": [str(lane) for lane in self.active_lanes],
            "severity": float(self.severity),
            "breadth": int(self.breadth),
            "inverse_policy_breadth": int(self.inverse_policy_breadth),
            "local_subdivision_breadth": int(self.local_subdivision_breadth),
        }


@dataclass(frozen=True)
class SolveRepairConfig:
    """Optional finite candidate-set repair for McLachlan solves."""

    enabled: bool = False
    condition_number_max: float | None = 1.0e6
    condition_number_fail: float | None = None
    strict_finite_shot_validation: bool = False
    theta_dot_l2_max: float | None = None
    rho_num_max: float | None = 1.0e-2
    state_motion_l2_step_max: float | None = 5.0e-2
    state_space_kink_eta_max: float | None = 1.0e-2
    local_subdivision_enabled: bool = True
    max_local_subdivisions: int = 4
    local_subdivision_factor: int = 2
    min_local_dt: float = 1.0e-6
    release_patience_min: int = 1
    release_patience_max: int = 5
    release_kink_threshold_scale: float = 0.5
    release_kink_severity_scale: float = 4.0
    score_cost_weight: float = 1.0e-3
    score_rho_weight: float = 1.0
    score_state_motion_weight: float = 1.0
    score_temporal_kink_weight: float = 1.0
    score_candidate_kink_weight: float = 0.25
    score_kappa_weight: float = 0.25
    ridge_ladder: tuple[float, ...] = (
        DEFAULT_MCLACHLAN_RIDGE_LAMBDA,
        3.0e-8,
        1.0e-8,
        0.0,
        3.0e-7,
        1.0e-6,
        3.0e-6,
        1.0e-5,
    )
    pinv_rcond_ladder: tuple[float, ...] = (
        1.0e-10,
        1.0e-11,
        1.0e-12,
        1.0e-9,
        1.0e-8,
        1.0e-7,
    )
    solve_damping_ladder: tuple[float, ...] = (DEFAULT_MCLACHLAN_SOLVE_DAMPING,)
    profile_id: str = SOLVE_REPAIR_PROFILE_V2
    selection_policy: str = SOLVE_REPAIR_SELECTION_STATE_SPACE_SCORE_V1

    @classmethod
    def minimal_profile(cls, **overrides: object) -> "SolveRepairConfig":
        """Repair reduced to the mechanism that measurably acts.

        Audit of 885 integration steps across 33 trajectories: 11,499 repair
        candidates were evaluated and 6 were applied, and the inverse-policy
        ladders left the base policy on 883/885 steps while damping never
        moved.  Local subdivision is what repeatedly changes a step, through
        both of its triggers: 372 subdivisions opened on excessive prospective
        state motion and 254 on the temporal-kink diagnostic.  This profile
        therefore keeps both subdivision triggers and pins the inverse policy
        to its base rung, retiring only the candidate search that measurement
        shows inert.  The full search remains available for diagnosis.
        """

        params: dict[str, object] = dict(
            enabled=True,
            # Both subdivision triggers are retained: across 33 trajectories
            # they opened 372 (state motion) and 254 (temporal kink)
            # subdivisions, so both act on real steps.
            state_motion_l2_step_max=1.0e-2,
            state_space_kink_eta_max=5.0e-3,
            local_subdivision_enabled=True,
            # What never acts is the inverse-candidate search: 6 of 11,499
            # candidates were accepted and the base rung held on 883 of 885
            # steps, so the ladders collapse to their base convention and the
            # conditioning/numerical-miss channels no longer open a search.
            condition_number_max=None,
            rho_num_max=None,
            ridge_ladder=(DEFAULT_MCLACHLAN_RIDGE_LAMBDA,),
            pinv_rcond_ladder=(1.0e-10,),
            solve_damping_ladder=(DEFAULT_MCLACHLAN_SOLVE_DAMPING,),
        )
        params.update(overrides)
        return cls(**params)  # type: ignore[arg-type]

    def __post_init__(self) -> None:
        for name in (
            "condition_number_max",
            "condition_number_fail",
            "theta_dot_l2_max",
            "rho_num_max",
            "state_motion_l2_step_max",
            "state_space_kink_eta_max",
            "min_local_dt",
            "release_kink_threshold_scale",
            "release_kink_severity_scale",
            "score_cost_weight",
            "score_rho_weight",
            "score_state_motion_weight",
            "score_temporal_kink_weight",
            "score_candidate_kink_weight",
            "score_kappa_weight",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive when set.")
        if int(self.max_local_subdivisions) < 0:
            raise ValueError("max_local_subdivisions must be non-negative.")
        if int(self.local_subdivision_factor) < 2:
            raise ValueError("local_subdivision_factor must be at least 2.")
        if int(self.release_patience_min) < 0:
            raise ValueError("release_patience_min must be non-negative.")
        if int(self.release_patience_max) < int(self.release_patience_min):
            raise ValueError("release_patience_max must be >= release_patience_min.")
        if float(self.release_kink_threshold_scale) > 1.0:
            raise ValueError("release_kink_threshold_scale must be <= 1.0.")
        for name in ("ridge_ladder", "pinv_rcond_ladder", "solve_damping_ladder"):
            values = tuple(float(v) for v in getattr(self, name))
            if not values:
                raise ValueError(f"{name} must contain at least one value.")
            if any((not np.isfinite(v) or v < 0.0) for v in values):
                raise ValueError(f"{name} values must be finite and non-negative.")
            object.__setattr__(self, name, values)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "profile_id": str(self.profile_id),
            "selection_policy": str(self.selection_policy),
            "condition_number_max": _finite_or_none(self.condition_number_max),
            "condition_number_max_status": "soft_kappa_warn_not_repair_trigger",
            "condition_number_fail": _finite_or_none(self.condition_number_fail),
            "strict_finite_shot_validation": bool(self.strict_finite_shot_validation),
            "theta_dot_l2_max": _finite_or_none(self.theta_dot_l2_max),
            "theta_dot_l2_max_status": "archaic_diagnostic_only_not_repair_guard",
            "rho_num_max": _finite_or_none(self.rho_num_max),
            "state_motion_l2_step_max": _finite_or_none(self.state_motion_l2_step_max),
            "state_space_kink_eta_max": _finite_or_none(self.state_space_kink_eta_max),
            "local_subdivision_enabled": bool(self.local_subdivision_enabled),
            "max_local_subdivisions": int(self.max_local_subdivisions),
            "local_subdivision_factor": int(self.local_subdivision_factor),
            "min_local_dt": float(self.min_local_dt),
            "release_patience_min": int(self.release_patience_min),
            "release_patience_max": int(self.release_patience_max),
            "release_kink_threshold_scale": float(self.release_kink_threshold_scale),
            "release_kink_severity_scale": float(self.release_kink_severity_scale),
            "score_cost_weight": float(self.score_cost_weight),
            "score_rho_weight": float(self.score_rho_weight),
            "score_state_motion_weight": float(self.score_state_motion_weight),
            "score_temporal_kink_weight": float(self.score_temporal_kink_weight),
            "score_candidate_kink_weight": float(self.score_candidate_kink_weight),
            "score_kappa_weight": float(self.score_kappa_weight),
            "ridge_ladder": [float(v) for v in self.ridge_ladder],
            "pinv_rcond_ladder": [float(v) for v in self.pinv_rcond_ladder],
            "solve_damping_ladder": [float(v) for v in self.solve_damping_ladder],
        }


@dataclass(frozen=True)
class FixedMcLachlanStep:
    """Result of ``theta_dot = K^+ f`` on a fixed active support."""

    theta_dot: np.ndarray
    gamma: float
    residual_sq: float
    residual_ratio: float
    rank: int
    condition_number: float | None
    geometry: McLachlanGeometry
    inverse_policy: McLachlanInversePolicy
    solve_repair_enabled: bool = False
    solve_repair_applied: bool = False
    solve_repair_unsupported: bool = False
    solve_repair_reason: str = "not_enabled"
    solve_repair_attempts: tuple[SolveRepairAttempt, ...] = ()
    solve_repair_config: SolveRepairConfig | None = None
    solve_repair_response_schedule: SolveRepairResponseSchedule | None = None
    state_space_metrics: StateSpaceSolveMetrics | None = None
    legacy_objective_residual_sq: float | None = None
    legacy_objective_residual_ratio: float | None = None
    projected_velocity_sq: float | None = None
    projected_velocity_l2: float | None = None
    realized_residual_sq: float | None = None
    rho_real: float | None = None
    best_case_residual_sq: float | None = None
    rho_expr: float | None = None
    rho_num: float | None = None
    state_motion_l2_step: float | None = None
    state_space_kink_eta: float | None = None
    solve_mode: str = "direct"
    solve_guard_g_empty: bool = False
    solve_guard_g_kappa: bool = False
    solve_guard_g_delta: bool = False
    solve_guard_g_rho: bool = False
    solve_guard_g_kink: bool = False

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": FIXED_STEP_SCHEMA_V2,
            "equation_id": FIXED_STEP_EQUATION_ID,
            "time": None if self.geometry.time is None else float(self.geometry.time),
            "theta_dot": [float(x) for x in self.theta_dot.tolist()],
            "gamma": float(self.gamma),
            "residual_sq": float(self.residual_sq),
            "residual_ratio": float(self.residual_ratio),
            "legacy_objective_residual_sq": _finite_or_none(
                self.legacy_objective_residual_sq
            ),
            "legacy_objective_residual_ratio": _finite_or_none(
                self.legacy_objective_residual_ratio
            ),
            "projected_velocity_sq": _finite_or_none(self.projected_velocity_sq),
            "projected_velocity_l2": _finite_or_none(self.projected_velocity_l2),
            "realized_residual_sq": _finite_or_none(self.realized_residual_sq),
            "rho_real": _finite_or_none(self.rho_real),
            "best_case_residual_sq": _finite_or_none(self.best_case_residual_sq),
            "rho_expr": _finite_or_none(self.rho_expr),
            "rho_num": _finite_or_none(self.rho_num),
            "state_motion_l2_step": _finite_or_none(self.state_motion_l2_step),
            "state_space_kink_eta": _finite_or_none(self.state_space_kink_eta),
            "norm_b_sq": float(self.geometry.norm_b_sq),
            "rank": int(self.rank),
            "condition_number": (
                None if self.condition_number is None else float(self.condition_number)
            ),
            "support_indices": [int(i) for i in self.geometry.support_indices or ()],
            "support_labels": [str(label) for label in self.geometry.support_labels],
            "inverse_policy_id": str(self.inverse_policy.policy_id),
            "pinv_rcond": float(self.inverse_policy.pinv_rcond),
            "ridge_lambda": float(self.inverse_policy.ridge_lambda),
            "solve_damping": float(self.inverse_policy.solve_damping),
            "solve_repair_enabled": bool(self.solve_repair_enabled),
            "solve_repair_applied": bool(self.solve_repair_applied),
            "solve_repair_unsupported": bool(self.solve_repair_unsupported),
            "solve_repair_reason": str(self.solve_repair_reason),
            "solve_repair_attempt_count": int(len(self.solve_repair_attempts)),
            "solve_mode": str(self.solve_mode),
            "solve_guard_g_empty": bool(self.solve_guard_g_empty),
            "solve_guard_g_kappa": bool(self.solve_guard_g_kappa),
            "solve_guard_g_delta": bool(self.solve_guard_g_delta),
            "solve_guard_g_rho": bool(self.solve_guard_g_rho),
            "solve_guard_g_kink": bool(self.solve_guard_g_kink),
            "solve_repair_attempts": [
                attempt.to_json_dict() for attempt in self.solve_repair_attempts
            ],
            "solve_repair_response_schedule": (
                None
                if self.solve_repair_response_schedule is None
                else self.solve_repair_response_schedule.to_json_dict()
            ),
            "solve_repair_config": (
                None
                if self.solve_repair_config is None
                else self.solve_repair_config.to_json_dict()
            ),
        }


def solve_fixed_mclachlan_step(
    geometry: McLachlanGeometry,
    *,
    inverse_policy: McLachlanInversePolicy = McLachlanInversePolicy(),
    step_dt: float | None = None,
    kink_reference_theta_dot: np.ndarray | Sequence[float] | None = None,
) -> FixedMcLachlanStep:
    """Solve Eq. (8) on the current fixed active support."""

    solve = solve_theta_dot(geometry.K, geometry.f, policy=inverse_policy)
    legacy_residual_sq = float(max(0.0, float(geometry.norm_b_sq) - float(solve.gamma)))
    legacy_residual_ratio = float(
        legacy_residual_sq / residual_denominator(geometry.norm_b_sq, inverse_policy.epsilon)
    )
    best_residual_sq = _best_case_residual_sq(geometry, inverse_policy)
    metrics = state_space_solve_metrics(
        geometry,
        solve.theta_dot,
        best_case_residual_sq=best_residual_sq,
        epsilon=float(inverse_policy.epsilon),
    )
    state_motion = None
    if step_dt is not None:
        state_motion = abs(float(step_dt)) * float(metrics.projected_velocity_l2)
    kink_eta = None
    if kink_reference_theta_dot is not None:
        kink_eta = state_space_kink_eta(
            geometry,
            solve.theta_dot,
            kink_reference_theta_dot,
            epsilon=float(inverse_policy.epsilon),
        )
    return FixedMcLachlanStep(
        theta_dot=np.asarray(solve.theta_dot, dtype=float).reshape(-1),
        gamma=float(solve.gamma),
        residual_sq=float(metrics.realized_residual_sq),
        residual_ratio=float(metrics.rho_real),
        rank=int(solve.inverse.rank),
        condition_number=solve.inverse.condition_number,
        geometry=geometry,
        inverse_policy=inverse_policy,
        state_space_metrics=metrics,
        legacy_objective_residual_sq=legacy_residual_sq,
        legacy_objective_residual_ratio=legacy_residual_ratio,
        projected_velocity_sq=float(metrics.projected_velocity_sq),
        projected_velocity_l2=float(metrics.projected_velocity_l2),
        realized_residual_sq=float(metrics.realized_residual_sq),
        rho_real=float(metrics.rho_real),
        best_case_residual_sq=float(metrics.best_case_residual_sq),
        rho_expr=float(metrics.rho_expr),
        rho_num=float(metrics.rho_num),
        state_motion_l2_step=state_motion,
        state_space_kink_eta=kink_eta,
    )


def solve_fixed_mclachlan_step_with_repair(
    geometry: McLachlanGeometry,
    *,
    inverse_policy: McLachlanInversePolicy = McLachlanInversePolicy(),
    repair_config: SolveRepairConfig = SolveRepairConfig(),
    repair_dt: float | None = None,
    kink_reference_theta_dot: np.ndarray | Sequence[float] | None = None,
) -> FixedMcLachlanStep:
    """Solve the fixed step, optionally scoring finite Paper-II repair candidates."""

    if not bool(repair_config.enabled):
        return solve_fixed_mclachlan_step(
            geometry,
            inverse_policy=inverse_policy,
            step_dt=repair_dt,
            kink_reference_theta_dot=kink_reference_theta_dot,
        )

    attempts: list[SolveRepairAttempt] = []
    valid_candidates: list[
        tuple[FixedMcLachlanStep, SolveGuardReport, int, str]
    ] = []
    base_step: FixedMcLachlanStep | None = None
    base_report: SolveGuardReport | None = None
    try:
        base_step = solve_fixed_mclachlan_step(
            geometry,
            inverse_policy=inverse_policy,
            step_dt=repair_dt,
            kink_reference_theta_dot=kink_reference_theta_dot,
        )
        base_report = _solve_guard_report(
            base_step,
            repair_config=repair_config,
            repair_dt=repair_dt,
        )
        if _step_is_execution_valid(base_step, base_report):
            valid_candidates.append((base_step, base_report, 0, "base"))
        accepted = _paper_ii_acceptability(base_report)
        attempts.append(
            _attempt_from_step(
                attempt_index=0,
                step=base_step,
                accepted=accepted,
                reason=base_report.guard_reason,
                guard_report=base_report,
                nominated_by="base",
                attempt_kind="base_policy",
                repair_config=repair_config,
                repair_dt=repair_dt,
            )
        )
        schedule = _repair_response_schedule(
            base_report,
            repair_config=repair_config,
        )
        if accepted:
            return _step_with_repair_telemetry(
                _step_with_guard_report(base_step, base_report),
                repair_config=repair_config,
                attempts=attempts,
                applied=False,
                unsupported=False,
                reason="accepted_base_policy",
                response_schedule=schedule,
            )
        if not _paper_ii_repair_entry_needed(base_report):
            return _step_with_repair_telemetry(
                _step_with_guard_report(base_step, base_report),
                repair_config=repair_config,
                attempts=attempts,
                applied=False,
                unsupported=True,
                reason=f"continued_with_base_policy_unsupported:no_repair_lane:{base_report.guard_reason}",
                response_schedule=schedule,
            )
    except (ValueError, np.linalg.LinAlgError) as exc:
        base_report = _failed_guard_report(repair_dt=repair_dt, reason="solve_failed")
        schedule = None
        attempts.append(
            SolveRepairAttempt(
                attempt_index=0,
                inverse_policy=inverse_policy,
                accepted=False,
                reason="solve_failed",
                error=str(exc),
                guard_report=base_report,
                nominated_by="base",
                attempt_kind="base_policy",
                repair_dt=repair_dt,
            )
        )

    last_reason = "no_repair_attempts"
    schedule = (
        None
        if base_report is None
        else _repair_response_schedule(base_report, repair_config=repair_config)
    )
    candidates = _repair_policy_candidates(
        inverse_policy,
        repair_config,
        inverse_policy_breadth=(
            None if schedule is None else int(schedule.inverse_policy_breadth)
        ),
        hard_invalidity=bool(base_report is not None and base_report.g_empty),
    )
    attempt_index = 1
    for policy, nominated_by in candidates:
        if _same_policy(policy, inverse_policy):
            continue
        try:
            step = solve_fixed_mclachlan_step(
                geometry,
                inverse_policy=policy,
                step_dt=repair_dt,
                kink_reference_theta_dot=kink_reference_theta_dot,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            last_reason = "solve_failed"
            attempts.append(
                SolveRepairAttempt(
                    attempt_index=int(attempt_index),
                    inverse_policy=policy,
                    accepted=False,
                    reason=last_reason,
                    error=str(exc),
                    guard_report=_failed_guard_report(
                        repair_dt=repair_dt,
                        reason=last_reason,
                    ),
                    nominated_by=nominated_by,
                    repair_dt=repair_dt,
                )
            )
            attempt_index += 1
            continue
        report = _solve_guard_report(
            step,
            repair_config=repair_config,
            repair_dt=repair_dt,
        )
        if _step_is_execution_valid(step, report):
            valid_candidates.append((step, report, attempt_index, nominated_by))
        accepted = _paper_ii_acceptability(report)
        attempts.append(
            _attempt_from_step(
                attempt_index=attempt_index,
                step=step,
                accepted=accepted,
                reason=report.guard_reason,
                guard_report=report,
                nominated_by=nominated_by,
                attempt_kind="same_checkpoint_policy_rung",
                repair_config=repair_config,
                repair_dt=repair_dt,
            )
        )
        last_reason = report.guard_reason
        attempt_index += 1

    if valid_candidates:
        acceptable_candidates = [
            candidate for candidate in valid_candidates if _paper_ii_acceptability(candidate[1])
        ]
        selection_pool = acceptable_candidates or valid_candidates
        selected_step, selected_report, selected_index, selected_nomination = min(
            selection_pool,
            key=lambda candidate: _paper_ii_repair_score(
                step=candidate[0],
                report=candidate[1],
                attempt_index=candidate[2],
                base_step=base_step,
                base_policy=inverse_policy,
                repair_config=repair_config,
            ),
        )
        applied = int(selected_index) != 0
        unsupported = not bool(acceptable_candidates)
        if unsupported:
            reason = (
                "continued_with_least_bad_finite_repair_unsupported"
                if applied
                else "continued_with_base_policy_unsupported"
            )
        else:
            reason = (
                "selected_acceptable_repair"
                if applied
                else "accepted_base_policy_after_candidate_score"
            )
        guarded_step = _step_with_guard_report(selected_step, selected_report)
        return _step_with_repair_telemetry(
            guarded_step,
            repair_config=repair_config,
            attempts=attempts,
            applied=applied,
            unsupported=unsupported,
            reason=f"{reason}:{selected_report.guard_reason}:{selected_nomination}",
            response_schedule=schedule,
        )

    raise SolveRepairUnsupportedError(
        "McLachlan solve repair failed before producing any finite executable "
        f"inverse-policy rung; last_reason={last_reason}.",
        attempts=tuple(attempts),
        reducible_by_subdivision=False,
        reason=last_reason,
    )


class SolveRepairUnsupportedError(ValueError):
    """Raised when same-checkpoint solve repair cannot produce any finite step."""

    def __init__(
        self,
        message: str,
        *,
        attempts: Sequence[SolveRepairAttempt],
        reducible_by_subdivision: bool,
        reason: str,
    ) -> None:
        super().__init__(message)
        self.attempts = tuple(attempts)
        self.reducible_by_subdivision = bool(reducible_by_subdivision)
        self.reason = str(reason)


def _step_with_repair_telemetry(
    step: FixedMcLachlanStep,
    *,
    repair_config: SolveRepairConfig,
    attempts: Sequence[SolveRepairAttempt],
    applied: bool,
    unsupported: bool = False,
    reason: str,
    response_schedule: SolveRepairResponseSchedule | None = None,
) -> FixedMcLachlanStep:
    return FixedMcLachlanStep(
        theta_dot=np.asarray(step.theta_dot, dtype=float).reshape(-1),
        gamma=float(step.gamma),
        residual_sq=float(step.residual_sq),
        residual_ratio=float(step.residual_ratio),
        rank=int(step.rank),
        condition_number=step.condition_number,
        geometry=step.geometry,
        inverse_policy=step.inverse_policy,
        solve_repair_enabled=True,
        solve_repair_applied=bool(applied),
        solve_repair_unsupported=bool(unsupported),
        solve_repair_reason=str(reason),
        solve_repair_attempts=tuple(attempts),
        solve_repair_config=repair_config,
        solve_repair_response_schedule=response_schedule,
        state_space_metrics=step.state_space_metrics,
        legacy_objective_residual_sq=step.legacy_objective_residual_sq,
        legacy_objective_residual_ratio=step.legacy_objective_residual_ratio,
        projected_velocity_sq=step.projected_velocity_sq,
        projected_velocity_l2=step.projected_velocity_l2,
        realized_residual_sq=step.realized_residual_sq,
        rho_real=step.rho_real,
        best_case_residual_sq=step.best_case_residual_sq,
        rho_expr=step.rho_expr,
        rho_num=step.rho_num,
        state_motion_l2_step=step.state_motion_l2_step,
        state_space_kink_eta=step.state_space_kink_eta,
        solve_mode=step.solve_mode,
        solve_guard_g_empty=bool(step.solve_guard_g_empty),
        solve_guard_g_kappa=bool(step.solve_guard_g_kappa),
        solve_guard_g_delta=bool(step.solve_guard_g_delta),
        solve_guard_g_rho=bool(step.solve_guard_g_rho),
        solve_guard_g_kink=bool(step.solve_guard_g_kink),
    )


def _step_with_guard_report(
    step: FixedMcLachlanStep,
    report: SolveGuardReport,
) -> FixedMcLachlanStep:
    return FixedMcLachlanStep(
        theta_dot=np.asarray(step.theta_dot, dtype=float).reshape(-1),
        gamma=float(step.gamma),
        residual_sq=float(step.residual_sq),
        residual_ratio=float(step.residual_ratio),
        rank=int(step.rank),
        condition_number=step.condition_number,
        geometry=step.geometry,
        inverse_policy=step.inverse_policy,
        solve_repair_unsupported=bool(step.solve_repair_unsupported),
        solve_repair_response_schedule=step.solve_repair_response_schedule,
        state_space_metrics=step.state_space_metrics,
        legacy_objective_residual_sq=step.legacy_objective_residual_sq,
        legacy_objective_residual_ratio=step.legacy_objective_residual_ratio,
        projected_velocity_sq=step.projected_velocity_sq,
        projected_velocity_l2=step.projected_velocity_l2,
        realized_residual_sq=step.realized_residual_sq,
        rho_real=step.rho_real,
        best_case_residual_sq=step.best_case_residual_sq,
        rho_expr=step.rho_expr,
        rho_num=step.rho_num,
        state_motion_l2_step=step.state_motion_l2_step,
        state_space_kink_eta=step.state_space_kink_eta,
        solve_mode="accepted" if report.guard_reason == "accepted" else "guarded",
        solve_guard_g_empty=bool(report.g_empty),
        solve_guard_g_kappa=bool(report.g_kappa),
        solve_guard_g_delta=bool(report.g_delta),
        solve_guard_g_rho=bool(report.g_rho),
        solve_guard_g_kink=bool(report.g_kink),
    )


def _attempt_from_step(
    *,
    attempt_index: int,
    step: FixedMcLachlanStep,
    accepted: bool,
    reason: str,
    guard_report: SolveGuardReport,
    nominated_by: str,
    attempt_kind: str,
    repair_config: SolveRepairConfig,
    repair_dt: float | None,
) -> SolveRepairAttempt:
    theta_dot_l2 = float(np.linalg.norm(np.asarray(step.theta_dot, dtype=float)))
    theta_limit = repair_config.theta_dot_l2_max
    return SolveRepairAttempt(
        attempt_index=int(attempt_index),
        inverse_policy=step.inverse_policy,
        accepted=bool(accepted),
        reason=str(reason),
        theta_dot_l2=theta_dot_l2,
        condition_number=step.condition_number,
        residual_ratio=float(step.residual_ratio),
        rank=int(step.rank),
        guard_report=guard_report,
        nominated_by=str(nominated_by),
        attempt_kind=str(attempt_kind),
        repair_dt=repair_dt,
        theta_dot_l2_limit_exceeded_diagnostic=(
            None if theta_limit is None else bool(theta_dot_l2 > float(theta_limit))
        ),
    )


def _solve_guard_report(
    step: FixedMcLachlanStep,
    *,
    repair_config: SolveRepairConfig,
    repair_dt: float | None,
) -> SolveGuardReport:
    metrics = step.state_space_metrics
    theta_dot_l2 = float(np.linalg.norm(np.asarray(step.theta_dot, dtype=float)))
    theta_dot_finite = bool(np.isfinite(theta_dot_l2))
    retained_empty = bool(int(step.rank) == 0)
    norm_nonzero = float(step.geometry.norm_b_sq) > float(step.inverse_policy.epsilon)
    g_empty = bool(retained_empty and norm_nonzero)
    cond_fail = repair_config.condition_number_fail
    g_kappa = bool(
        bool(repair_config.strict_finite_shot_validation)
        and cond_fail is not None
        and step.condition_number is not None
        and float(step.condition_number) > float(cond_fail)
    )
    state_motion = step.state_motion_l2_step
    motion_max = repair_config.state_motion_l2_step_max
    g_delta = bool(
        motion_max is not None
        and state_motion is not None
        and np.isfinite(float(state_motion))
        and float(state_motion) > float(motion_max)
    )
    rho_num = None if metrics is None else float(metrics.rho_num)
    rho_max = repair_config.rho_num_max
    g_rho = bool(
        rho_max is not None
        and rho_num is not None
        and np.isfinite(float(rho_num))
        and float(rho_num) > float(rho_max)
    )
    kink_eta = step.state_space_kink_eta
    kink_max = repair_config.state_space_kink_eta_max
    g_kink = bool(
        kink_max is not None
        and kink_eta is not None
        and np.isfinite(float(kink_eta))
        and float(kink_eta) > float(kink_max)
    )
    reason = "accepted"
    if not theta_dot_finite:
        reason = "theta_dot_nonfinite"
    elif g_empty:
        reason = "empty_retained_support"
    elif g_kappa:
        reason = "condition_number_strict_finite_shot_fail"
    elif g_delta:
        reason = "state_motion_step_above_max"
    elif g_kink:
        reason = "state_space_temporal_kink_above_max"
    elif g_rho:
        reason = "rho_num_above_max"
    return SolveGuardReport(
        repair_dt=repair_dt,
        g_empty=g_empty,
        g_kappa=g_kappa,
        g_delta=g_delta,
        g_rho=g_rho,
        g_kink=g_kink,
        retained_support_empty=retained_empty,
        state_motion_l2_step=state_motion,
        state_space_kink_eta=kink_eta,
        rho_real=None if metrics is None else float(metrics.rho_real),
        rho_expr=None if metrics is None else float(metrics.rho_expr),
        rho_num=rho_num,
        projected_velocity_l2=None if metrics is None else float(metrics.projected_velocity_l2),
        realized_residual_sq=None if metrics is None else float(metrics.realized_residual_sq),
        best_case_residual_sq=None if metrics is None else float(metrics.best_case_residual_sq),
        guard_reason=reason,
    )


def _failed_guard_report(*, repair_dt: float | None, reason: str) -> SolveGuardReport:
    return SolveGuardReport(
        repair_dt=repair_dt,
        g_empty=False,
        g_kappa=False,
        g_delta=False,
        g_rho=False,
        g_kink=False,
        retained_support_empty=False,
        state_motion_l2_step=None,
        state_space_kink_eta=None,
        rho_real=None,
        rho_expr=None,
        rho_num=None,
        projected_velocity_l2=None,
        realized_residual_sq=None,
        best_case_residual_sq=None,
        guard_reason=str(reason),
    )


def _step_is_execution_valid(
    step: FixedMcLachlanStep,
    report: SolveGuardReport,
) -> bool:
    if str(report.guard_reason) == "theta_dot_nonfinite":
        return False
    theta_dot = np.asarray(step.theta_dot, dtype=float).reshape(-1)
    if not np.all(np.isfinite(theta_dot)):
        return False
    required_scalars = (
        step.gamma,
        step.residual_sq,
        step.residual_ratio,
        step.rho_real,
        step.rho_expr,
        step.rho_num,
        step.realized_residual_sq,
        step.best_case_residual_sq,
    )
    return all(value is not None and np.isfinite(float(value)) for value in required_scalars)


def _paper_ii_acceptability(report: SolveGuardReport) -> bool:
    return not bool(
        report.g_empty
        or report.g_kappa
        or report.g_delta
        or report.g_rho
        or report.g_kink
        or str(report.guard_reason) in {"theta_dot_nonfinite", "solve_failed"}
    )


def _paper_ii_repair_entry_needed(report: SolveGuardReport) -> bool:
    return bool(report.g_empty or report.g_rho or report.g_delta or report.g_kink)


def _repair_response_schedule(
    report: SolveGuardReport,
    *,
    repair_config: SolveRepairConfig,
) -> SolveRepairResponseSchedule:
    rho = _threshold_ratio(report.rho_num, repair_config.rho_num_max)
    delta = _threshold_ratio(
        report.state_motion_l2_step,
        repair_config.state_motion_l2_step_max,
    )
    temporal_kink_raw = _threshold_ratio(
        report.state_space_kink_eta,
        repair_config.state_space_kink_eta_max,
    )
    temporal = float(np.sqrt(max(0.0, temporal_kink_raw)))
    components = {
        "rho": rho,
        "delta": delta,
        "time": temporal,
    }
    lanes = tuple(label for label, value in components.items() if float(value) > 1.0)
    severity = float(max([1.0, *components.values()]))
    breadth = 0
    if np.isfinite(severity) and severity > 1.0:
        breadth = int(max(0, np.ceil(np.log2(severity))))
    local_breadth = (
        min(breadth, int(repair_config.max_local_subdivisions))
        if any(lane in {"delta", "time"} for lane in lanes)
        else 0
    )
    inverse_breadth = breadth if "rho" in lanes else 0
    return SolveRepairResponseSchedule(
        active_lanes=lanes,
        severity=severity,
        breadth=breadth,
        inverse_policy_breadth=inverse_breadth,
        local_subdivision_breadth=local_breadth,
    )


def _paper_ii_repair_score(
    *,
    step: FixedMcLachlanStep,
    report: SolveGuardReport,
    attempt_index: int,
    base_step: FixedMcLachlanStep | None,
    base_policy: McLachlanInversePolicy,
    repair_config: SolveRepairConfig,
) -> tuple[float, int]:
    cost = _intervention_cost(base_policy, step.inverse_policy)
    rho = _threshold_ratio(report.rho_num, repair_config.rho_num_max)
    delta = _threshold_ratio(
        report.state_motion_l2_step,
        repair_config.state_motion_l2_step_max,
    )
    temporal_kink = _threshold_ratio(
        report.state_space_kink_eta,
        repair_config.state_space_kink_eta_max,
    )
    candidate_kink = _candidate_kink_ratio(
        step=step,
        base_step=base_step,
        repair_config=repair_config,
    )
    kappa = _kappa_soft_penalty(step.condition_number, repair_config)
    hard_penalty = 0.0 if _paper_ii_acceptability(report) else 1.0e6
    score = (
        hard_penalty
        + float(repair_config.score_cost_weight) * cost
        + float(repair_config.score_rho_weight) * rho
        + float(repair_config.score_state_motion_weight) * delta
        + float(repair_config.score_temporal_kink_weight) * temporal_kink
        + float(repair_config.score_candidate_kink_weight) * candidate_kink
        + float(repair_config.score_kappa_weight) * kappa
    )
    return (float(score), int(attempt_index))


def _intervention_cost(
    base_policy: McLachlanInversePolicy,
    candidate_policy: McLachlanInversePolicy,
) -> float:
    cost = 0.0
    for base, candidate in (
        (base_policy.pinv_rcond, candidate_policy.pinv_rcond),
        (base_policy.ridge_lambda, candidate_policy.ridge_lambda),
        (base_policy.solve_damping, candidate_policy.solve_damping),
    ):
        base_f = float(base)
        candidate_f = float(candidate)
        if base_f == candidate_f:
            continue
        cost += 1.0 + abs(_safe_log10(candidate_f) - _safe_log10(base_f))
    return float(cost)


def _threshold_ratio(value: float | None, limit: float | None) -> float:
    if value is None:
        return 0.0
    value_f = float(value)
    if not np.isfinite(value_f):
        return float("inf")
    if limit is None:
        return max(0.0, value_f)
    limit_f = float(limit)
    if not np.isfinite(limit_f) or limit_f <= 0.0:
        return float("inf")
    return float(max(0.0, value_f / limit_f))


def _candidate_kink_ratio(
    *,
    step: FixedMcLachlanStep,
    base_step: FixedMcLachlanStep | None,
    repair_config: SolveRepairConfig,
) -> float:
    if base_step is None or _same_policy(step.inverse_policy, base_step.inverse_policy):
        return 0.0
    try:
        eta = state_space_kink_eta(
            step.geometry,
            step.theta_dot,
            base_step.theta_dot,
            epsilon=float(step.inverse_policy.epsilon),
        )
    except ValueError:
        return float("inf")
    return _threshold_ratio(eta, repair_config.state_space_kink_eta_max)


def _kappa_soft_penalty(
    condition_number: float | None,
    repair_config: SolveRepairConfig,
) -> float:
    if condition_number is None:
        return 0.0
    kappa = float(condition_number)
    warn = repair_config.condition_number_max
    if warn is None or not np.isfinite(kappa):
        return 0.0 if np.isfinite(kappa) else float("inf")
    warn_f = float(warn)
    if warn_f <= 0.0 or kappa <= warn_f:
        return 0.0
    fail = repair_config.condition_number_fail
    fail_f = float(fail) if fail is not None else warn_f * 100.0
    if not np.isfinite(fail_f) or fail_f <= warn_f:
        return 1.0
    return float(np.clip((_safe_log10(kappa) - _safe_log10(warn_f)) / (_safe_log10(fail_f) - _safe_log10(warn_f)), 0.0, 1.0))


def _safe_log10(value: float) -> float:
    value_f = float(value)
    if not np.isfinite(value_f):
        return float("inf")
    return float(np.log10(max(value_f, 1.0e-300)))


def _best_case_residual_sq(
    geometry: McLachlanGeometry,
    inverse_policy: McLachlanInversePolicy,
) -> float:
    best_policy = McLachlanInversePolicy(
        pinv_rcond=float(inverse_policy.pinv_rcond),
        ridge_lambda=0.0,
        solve_damping=0.0,
        epsilon=float(inverse_policy.epsilon),
        policy_id=str(inverse_policy.policy_id),
    )
    solve = solve_theta_dot(geometry.K, geometry.f, policy=best_policy)
    return float(max(0.0, float(geometry.norm_b_sq) - float(solve.gamma)))


def _repair_policy_candidates(
    base_policy: McLachlanInversePolicy,
    repair_config: SolveRepairConfig,
    *,
    inverse_policy_breadth: int | None,
    hard_invalidity: bool = False,
) -> tuple[tuple[McLachlanInversePolicy, str], ...]:
    if not bool(hard_invalidity):
        breadth = 0 if inverse_policy_breadth is None else int(inverse_policy_breadth)
        if breadth <= 0:
            return ()
    else:
        breadth = max(
            len(repair_config.ridge_ladder),
            len(repair_config.pinv_rcond_ladder),
            len(repair_config.solve_damping_ladder),
        )
    ridges = _ladder_values(repair_config.ridge_ladder, float(base_policy.ridge_lambda))
    rconds = _ladder_values(repair_config.pinv_rcond_ladder, float(base_policy.pinv_rcond))
    dampings = _ladder_values(
        repair_config.solve_damping_ladder,
        float(base_policy.solve_damping),
    )
    ridge_up = _nearest_ladder_values_above(ridges, float(base_policy.ridge_lambda), breadth)
    ridge_down = _nearest_ladder_values_below(ridges, float(base_policy.ridge_lambda), breadth)
    rcond_up = _nearest_ladder_values_above(rconds, float(base_policy.pinv_rcond), breadth)
    rcond_down = _nearest_ladder_values_below(rconds, float(base_policy.pinv_rcond), breadth)
    damping_up = _nearest_ladder_values_above(
        dampings,
        float(base_policy.solve_damping),
        breadth,
    )
    damping_down = _nearest_ladder_values_below(
        dampings,
        float(base_policy.solve_damping),
        breadth,
    )
    policies: list[tuple[McLachlanInversePolicy, str]] = []
    seen: set[tuple[float, float, float]] = set()

    def add(policy: McLachlanInversePolicy, nominated_by: str) -> None:
        key = (
            float(policy.pinv_rcond),
            float(policy.ridge_lambda),
            float(policy.solve_damping),
        )
        if key in seen:
            return
        seen.add(key)
        policies.append((policy, str(nominated_by)))

    def make(rcond: float, ridge: float, damping: float) -> McLachlanInversePolicy:
        return McLachlanInversePolicy(
            pinv_rcond=float(rcond),
            ridge_lambda=float(ridge),
            solve_damping=float(damping),
            epsilon=float(base_policy.epsilon),
            policy_id=str(base_policy.policy_id),
        )

    for ridge in ridge_up:
        add(make(base_policy.pinv_rcond, ridge, base_policy.solve_damping), "ridge_up")
    for ridge in ridge_down:
        add(make(base_policy.pinv_rcond, ridge, base_policy.solve_damping), "ridge_down")
    for rcond in rcond_up:
        add(make(rcond, base_policy.ridge_lambda, base_policy.solve_damping), "pinv_rcond_up")
    for rcond in rcond_down:
        add(make(rcond, base_policy.ridge_lambda, base_policy.solve_damping), "pinv_rcond_down")
    for damping in damping_up:
        add(make(base_policy.pinv_rcond, base_policy.ridge_lambda, damping), "solve_damping_up")
    for damping in damping_down:
        add(make(base_policy.pinv_rcond, base_policy.ridge_lambda, damping), "solve_damping_down")

    return tuple(policies)


def _ladder_values(values: Sequence[float], base_value: float) -> tuple[float, ...]:
    out: list[float] = [float(base_value)]
    for value in tuple(values):
        candidate = float(value)
        if all(abs(candidate - existing) > 0.0 for existing in out):
            out.append(candidate)
    return tuple(out)


def _ladder_values_above(values: Sequence[float], base_value: float) -> tuple[float, ...]:
    base = float(base_value)
    return tuple(float(value) for value in tuple(values) if float(value) > base)


def _ladder_values_below(values: Sequence[float], base_value: float) -> tuple[float, ...]:
    base = float(base_value)
    return tuple(float(value) for value in tuple(values) if float(value) < base)


def _nearest_ladder_values_above(
    values: Sequence[float],
    base_value: float,
    count: int,
) -> tuple[float, ...]:
    if int(count) <= 0:
        return ()
    base = float(base_value)
    candidates = [float(value) for value in tuple(values) if float(value) > base]
    return tuple(
        sorted(candidates, key=lambda value: (_log_distance(value, base), value))[: int(count)]
    )


def _nearest_ladder_values_below(
    values: Sequence[float],
    base_value: float,
    count: int,
) -> tuple[float, ...]:
    if int(count) <= 0:
        return ()
    base = float(base_value)
    candidates = [float(value) for value in tuple(values) if float(value) < base]
    return tuple(
        sorted(candidates, key=lambda value: (_log_distance(value, base), -value))[: int(count)]
    )


def _log_distance(value: float, base_value: float) -> float:
    return abs(_safe_log10(float(value)) - _safe_log10(float(base_value)))


def _same_policy(left: McLachlanInversePolicy, right: McLachlanInversePolicy) -> bool:
    return (
        float(left.pinv_rcond) == float(right.pinv_rcond)
        and float(left.ridge_lambda) == float(right.ridge_lambda)
        and float(left.solve_damping) == float(right.solve_damping)
        and float(left.epsilon) == float(right.epsilon)
        and str(left.policy_id) == str(right.policy_id)
    )


def _attempts_reducible_by_subdivision(attempts: Sequence[SolveRepairAttempt]) -> bool:
    reports = [
        attempt.guard_report
        for attempt in tuple(attempts)
        if attempt.guard_report is not None and str(attempt.reason) != "solve_failed"
    ]
    if not reports:
        return False
    if not any(bool(report.g_delta or report.g_kink) for report in reports):
        return False
    for report in reports:
        if bool(report.g_empty or report.g_kappa or report.g_rho):
            return False
    return True


def _finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


__all__ = [
    "FIXED_STEP_EQUATION_ID",
    "FIXED_STEP_SCHEMA_V1",
    "FIXED_STEP_SCHEMA_V2",
    "FixedMcLachlanStep",
    "SOLVE_REPAIR_PROFILE_V1",
    "SOLVE_REPAIR_PROFILE_V2",
    "SOLVE_REPAIR_SELECTION_FIRST_PASSING_V1",
    "SOLVE_REPAIR_SELECTION_STATE_SPACE_SCORE_V1",
    "SolveGuardReport",
    "SolveRepairAttempt",
    "SolveRepairConfig",
    "SolveRepairResponseSchedule",
    "SolveRepairUnsupportedError",
    "solve_fixed_mclachlan_step",
    "solve_fixed_mclachlan_step_with_repair",
]
