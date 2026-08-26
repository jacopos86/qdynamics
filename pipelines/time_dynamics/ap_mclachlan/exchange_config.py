"""Typed AP adapter configuration for the generalized-exchange algorithm.

The route CLI still produces one transport object, but the live adapter never
receives that undifferentiated bag.  It partitions settings by the question
they answer: ordering, eligibility, certification, or bounded search.  The
pure mathematical rule remains in ``pipelines.time_dynamics.generalized_exchange``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ExchangeScoreConfig:
    """Quantities that change the mathematical preference ordering."""

    l2_cut: float
    debt_policy: str
    append_cost_alpha: float
    prune_cost_alpha: float
    delta_weight: float
    history_weight: float
    history_window: int
    epsilon_loss: float
    condition_relief_weight: float
    condition_damage_weight: float


@dataclass(frozen=True)
class ExchangeEligibilityConfig:
    """Rules that determine which support edits may be nominated."""

    insertion_gate_mode: str
    residual_ratio_threshold: float
    accumulated_drift_threshold: float | None
    deletion_enabled: bool
    target_policy: str
    occurrence_policy: str
    cooldown_steps: int
    minimum_surviving_support: int
    protect_drive_aligned: bool
    allow_incomplete_candidate_pool: bool


@dataclass(frozen=True)
class ExchangeCertificationConfig:
    """Hard gates that a materialized finalist must pass."""

    ray_distance_max: float
    velocity_change_max: float
    condition_number_max: float | None
    refit_enabled: bool
    refit_trust_radius: float
    refit_max_iterations: int


@dataclass(frozen=True)
class ExchangeSearchBudget:
    """Finite-work bounds on the combinatorial search approximation."""

    pool_size: int | None
    insertion_cardinality: int
    joint_patch_evaluations: int | None
    certification_attempts_per_level: int | None
    certification_attempts_per_deletion_branch: int | None
    rounds_per_checkpoint: int
    interaction_frontier_widths: tuple[int, ...] | None
    scoring_workers: int
    structural_score_floor: float


@dataclass(frozen=True)
class APGeneralizedExchangeConfig:
    """The four AP-specific inputs to one generalized-exchange operation."""

    score: ExchangeScoreConfig
    eligibility: ExchangeEligibilityConfig
    certification: ExchangeCertificationConfig
    search: ExchangeSearchBudget

    def to_json_dict(self) -> dict[str, object]:
        """Return the four-class provenance object."""

        return asdict(self)

    @classmethod
    def from_route_config(cls, config: Any) -> "APGeneralizedExchangeConfig":
        insertion_cap = getattr(config, "max_insertion_batch_size", 1)
        if insertion_cap is None:
            insertion_cap = 1
        return cls(
            score=ExchangeScoreConfig(
                l2_cut=float(config.insertion_l2_cut),
                debt_policy=str(config.debt_policy),
                append_cost_alpha=float(config.append_cost_alpha),
                prune_cost_alpha=float(config.prune_cost_alpha),
                delta_weight=float(config.patch_utility_delta_weight),
                history_weight=float(config.prune_history_lambda),
                history_window=int(config.prune_history_window),
                epsilon_loss=float(config.eps_loss),
                condition_relief_weight=float(
                    config.prune_condition_lambda_kappa_rel
                ),
                condition_damage_weight=float(
                    config.prune_condition_lambda_kappa_dam
                ),
            ),
            eligibility=ExchangeEligibilityConfig(
                insertion_gate_mode=str(config.insertion_gate_mode),
                residual_ratio_threshold=float(config.residual_ratio_threshold),
                accumulated_drift_threshold=(
                    None
                    if config.escalation_accumulated_drift_threshold is None
                    else float(config.escalation_accumulated_drift_threshold)
                ),
                deletion_enabled=bool(config.deletion_enabled),
                target_policy=str(config.prune_appended_origin_target_policy),
                occurrence_policy=str(config.append_occurrence_policy),
                cooldown_steps=int(config.prune_cooldown_steps),
                minimum_surviving_support=int(
                    config.min_runtime_parameter_count
                ),
                protect_drive_aligned=bool(config.protect_drive_aligned_atoms),
                allow_incomplete_candidate_pool=bool(
                    config.allow_incomplete_candidate_pool
                ),
            ),
            certification=ExchangeCertificationConfig(
                ray_distance_max=float(config.prune_ray_distance_tol),
                velocity_change_max=float(
                    config.prune_patch_smoothness_eta_max
                ),
                condition_number_max=(
                    None
                    if config.append_schur_max_condition_number is None
                    else float(config.append_schur_max_condition_number)
                ),
                refit_enabled=bool(config.certification_refit_enabled),
                refit_trust_radius=float(
                    config.certification_refit_trust_radius
                ),
                refit_max_iterations=int(
                    config.certification_refit_max_iterations
                ),
            ),
            search=ExchangeSearchBudget(
                pool_size=(
                    None
                    if config.max_structural_pool_size is None
                    else int(config.max_structural_pool_size)
                ),
                insertion_cardinality=int(insertion_cap),
                joint_patch_evaluations=(
                    None
                    if config.max_joint_patch_evaluations is None
                    else int(config.max_joint_patch_evaluations)
                ),
                certification_attempts_per_level=(
                    None
                    if config.max_certification_attempts_per_level is None
                    else int(config.max_certification_attempts_per_level)
                ),
                certification_attempts_per_deletion_branch=(
                    None
                    if config.max_certification_attempts_per_deletion_branch is None
                    else int(config.max_certification_attempts_per_deletion_branch)
                ),
                rounds_per_checkpoint=int(
                    config.max_insertion_rounds_per_checkpoint
                ),
                interaction_frontier_widths=(
                    None
                    if config.interaction_frontier_widths is None
                    else tuple(int(w) for w in config.interaction_frontier_widths)
                ),
                scoring_workers=int(config.support_patch_scoring_workers),
                structural_score_floor=float(config.structural_score_floor),
            ),
        )


__all__ = [
    "APGeneralizedExchangeConfig",
    "ExchangeCertificationConfig",
    "ExchangeEligibilityConfig",
    "ExchangeScoreConfig",
    "ExchangeSearchBudget",
]
