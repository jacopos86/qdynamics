"""Typed AP adapter configuration for the generalized-exchange algorithm.

The route CLI still produces one transport object, but the live adapter never
receives that undifferentiated bag.  It partitions settings by the question
they answer: ordering, eligibility, certification, or bounded search.  The
pure mathematical rule remains in ``pipelines.time_dynamics.generalized_exchange``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping


# --- Closed route bundles -----------------------------------------------------
#
# A bundle fixes the physical ordinate, the cost model, and the ranking kernel
# together.  They are registered here and nowhere else, so an objective, a
# cost model, and a kernel can never be mixed by hand: a run names one bundle
# id and inherits the whole contract.  The historical bundle is the route as it
# ran before 2026-09-02 and is never reinterpreted; its cost term is the
# singleton-normalized proxy that is identically 1 (measured, see
# docs/reports/PAPER_II_FULL_APM_PARALLELIZATION_20260901.md 8b.0).

EXCHANGE_ROUTE_BUNDLE_HISTORICAL = "paper_ii_generalized_exchange_v2"
EXCHANGE_ROUTE_BUNDLE_LEGACY_PROXY_FIXED = "paper_ii_exchange_legacy_proxy_fixed_v1"
EXCHANGE_ROUTE_BUNDLE_Q_COMPILED = "paper_ii_exchange_q_compiled_v1"
EXCHANGE_ROUTE_BUNDLE_GAMMA_COMPILED = "paper_ii_exchange_gamma_compiled_v1"

OBJECTIVE_LEGACY_COMPOSITE = "legacy_composite_utility_v2"
OBJECTIVE_REALIZED_Q = "realized_captured_drift_q_v1"
OBJECTIVE_IDEAL_GAMMA = "ideal_projection_gamma_v1"

COST_MODEL_INERT_SINGLETON_PROXY = "paper_i_proxy_singleton_inert_v1"
COST_MODEL_PROXY_POPULATION = "paper_i_proxy_population_v1"
COST_MODEL_COMPILED_POPULATION = "compiled_marrakesh_population_v1"

KERNEL_LEGACY_UTILITY_DEBT = "legacy_utility_debt_ordering_v2"
KERNEL_SHIFTED_PAPER_I = "shifted_paper_i_v1"

# Sealed compile contract, mirrored from the Paper-I precedent
# (pipelines/static_adapt/ra_adapt/marginal_qiskit_cost.py).  A run that needs
# different values declares a new bundle; these are not tunable per run.
COMPILE_BACKEND_NAME = "FakeMarrakesh"
COMPILE_OPTIMIZATION_LEVEL = 1
COMPILE_TRANSPILE_SEED = 7
COMPILE_STRUCTURAL_ANGLE = 1.0
COMPILE_TIMEOUT_SECONDS = 300.0


class _FrozenDict(dict):
    """Pickle/deepcopy-safe immutable dictionary for sealed route data."""

    @staticmethod
    def _immutable(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("sealed route mappings are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __deepcopy__(self, _memo: dict[int, Any]) -> "_FrozenDict":
        return self

    def __reduce__(self) -> tuple[Any, tuple[dict[str, float]]]:
        return (_FrozenDict, (dict(self),))


COMPILE_COORDINATE_WEIGHTS: Mapping[str, float] = _FrozenDict({
    "delta_n2q": 0.30,
    "delta_d2q": 0.30,
    "delta_dc": 0.25,
})
COMPILE_SCALE_FLOOR = 1.0e-12


@dataclass(frozen=True)
class CompileContract:
    """The sealed compiler settings a compiled-cost bundle carries."""

    backend_name: str = COMPILE_BACKEND_NAME
    optimization_level: int = COMPILE_OPTIMIZATION_LEVEL
    transpile_seed: int = COMPILE_TRANSPILE_SEED
    structural_angle: float = COMPILE_STRUCTURAL_ANGLE
    coordinate_weights: Mapping[str, float] = None  # type: ignore[assignment]
    scale_floor: float = COMPILE_SCALE_FLOOR
    compile_timeout_seconds: float = COMPILE_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        weights = (
            dict(COMPILE_COORDINATE_WEIGHTS)
            if self.coordinate_weights is None
            else {str(k): float(v) for k, v in self.coordinate_weights.items()}
        )
        object.__setattr__(self, "coordinate_weights", _FrozenDict(weights))
        if float(self.structural_angle) == 0.0:
            raise ValueError("structural_angle must be nonzero.")
        if (
            not math.isfinite(float(self.compile_timeout_seconds))
            or float(self.compile_timeout_seconds) <= 0.0
        ):
            raise ValueError("compile_timeout_seconds must be finite and positive.")


@dataclass(frozen=True)
class ExchangeRouteBundle:
    """One closed (objective, cost model, kernel, compiler) contract."""

    bundle_id: str
    objective_id: str
    cost_model_id: str
    kernel_id: str
    compiler: CompileContract | None

    @property
    def is_historical(self) -> bool:
        return self.bundle_id == EXCHANGE_ROUTE_BUNDLE_HISTORICAL

    @property
    def uses_population_scoring(self) -> bool:
        return not self.is_historical

    @property
    def needs_compiled_cost(self) -> bool:
        return self.cost_model_id == COST_MODEL_COMPILED_POPULATION

    @property
    def needs_gamma(self) -> bool:
        return self.objective_id == OBJECTIVE_IDEAL_GAMMA


EXCHANGE_ROUTE_BUNDLES: Mapping[str, ExchangeRouteBundle] = MappingProxyType({
    b.bundle_id: b
    for b in (
        ExchangeRouteBundle(
            bundle_id=EXCHANGE_ROUTE_BUNDLE_HISTORICAL,
            objective_id=OBJECTIVE_LEGACY_COMPOSITE,
            cost_model_id=COST_MODEL_INERT_SINGLETON_PROXY,
            kernel_id=KERNEL_LEGACY_UTILITY_DEBT,
            compiler=None,
        ),
        ExchangeRouteBundle(
            bundle_id=EXCHANGE_ROUTE_BUNDLE_LEGACY_PROXY_FIXED,
            objective_id=OBJECTIVE_LEGACY_COMPOSITE,
            cost_model_id=COST_MODEL_PROXY_POPULATION,
            kernel_id=KERNEL_LEGACY_UTILITY_DEBT,
            compiler=None,
        ),
        ExchangeRouteBundle(
            bundle_id=EXCHANGE_ROUTE_BUNDLE_Q_COMPILED,
            objective_id=OBJECTIVE_REALIZED_Q,
            cost_model_id=COST_MODEL_COMPILED_POPULATION,
            kernel_id=KERNEL_SHIFTED_PAPER_I,
            compiler=CompileContract(),
        ),
        ExchangeRouteBundle(
            bundle_id=EXCHANGE_ROUTE_BUNDLE_GAMMA_COMPILED,
            objective_id=OBJECTIVE_IDEAL_GAMMA,
            cost_model_id=COST_MODEL_COMPILED_POPULATION,
            kernel_id=KERNEL_SHIFTED_PAPER_I,
            compiler=CompileContract(),
        ),
    )
})


def resolve_route_bundle(bundle_id: str) -> ExchangeRouteBundle:
    """Return the registered bundle; unknown ids fail closed."""

    key = str(bundle_id)
    if key not in EXCHANGE_ROUTE_BUNDLES:
        raise ValueError(
            f"unknown exchange route bundle {key!r}; "
            f"known: {sorted(EXCHANGE_ROUTE_BUNDLES)}"
        )
    return EXCHANGE_ROUTE_BUNDLES[key]


@dataclass(frozen=True)
class ExchangeRouteConfig:
    """The bundle in force plus its two declared budgets/tolerances."""

    bundle: ExchangeRouteBundle
    # Maintenance-regime loss tolerance tau_del (normalized captured drift).
    # ``requested`` is what the run declared (None = inherit the route's
    # normalized Schur-loss tolerance); ``resolved`` is the number in force.
    maintenance_loss_tolerance_requested: float | None
    maintenance_loss_tolerance: float
    # Compile work guard per checkpoint; admits families whole like the
    # joint-work guard.  None = unbounded (small systems and smokes only).
    max_compiled_patches_per_checkpoint: int | None
    # Execution resource (no scientific effect): compile pool size.  Compile
    # warming is sequential with the solve warm on this lineage.
    compile_workers: int


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
    route: ExchangeRouteConfig

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
            route=ExchangeRouteConfig(
                bundle=resolve_route_bundle(
                    getattr(
                        config,
                        "exchange_route_bundle",
                        EXCHANGE_ROUTE_BUNDLE_HISTORICAL,
                    )
                ),
                maintenance_loss_tolerance_requested=(
                    None
                    if getattr(config, "maintenance_loss_tolerance", None) is None
                    else float(config.maintenance_loss_tolerance)
                ),
                maintenance_loss_tolerance=float(
                    config.prune_patch_smoothness_eta_max
                    if getattr(config, "maintenance_loss_tolerance", None) is None
                    else config.maintenance_loss_tolerance
                ),
                max_compiled_patches_per_checkpoint=(
                    None
                    if getattr(config, "max_compiled_patches_per_checkpoint", None)
                    is None
                    else int(config.max_compiled_patches_per_checkpoint)
                ),
                compile_workers=int(
                    config.support_patch_scoring_workers
                    if getattr(config, "compile_workers", None) is None
                    else config.compile_workers
                ),
            ),
        )


__all__ = [
    "APGeneralizedExchangeConfig",
    "COMPILE_BACKEND_NAME",
    "COMPILE_COORDINATE_WEIGHTS",
    "COMPILE_OPTIMIZATION_LEVEL",
    "COMPILE_SCALE_FLOOR",
    "COMPILE_STRUCTURAL_ANGLE",
    "COMPILE_TIMEOUT_SECONDS",
    "COMPILE_TRANSPILE_SEED",
    "COST_MODEL_COMPILED_POPULATION",
    "COST_MODEL_INERT_SINGLETON_PROXY",
    "COST_MODEL_PROXY_POPULATION",
    "CompileContract",
    "EXCHANGE_ROUTE_BUNDLES",
    "EXCHANGE_ROUTE_BUNDLE_GAMMA_COMPILED",
    "EXCHANGE_ROUTE_BUNDLE_HISTORICAL",
    "EXCHANGE_ROUTE_BUNDLE_LEGACY_PROXY_FIXED",
    "EXCHANGE_ROUTE_BUNDLE_Q_COMPILED",
    "ExchangeRouteBundle",
    "ExchangeRouteConfig",
    "KERNEL_LEGACY_UTILITY_DEBT",
    "KERNEL_SHIFTED_PAPER_I",
    "OBJECTIVE_IDEAL_GAMMA",
    "OBJECTIVE_LEGACY_COMPOSITE",
    "OBJECTIVE_REALIZED_Q",
    "resolve_route_bundle",
    "ExchangeCertificationConfig",
    "ExchangeEligibilityConfig",
    "ExchangeScoreConfig",
    "ExchangeSearchBudget",
]
