"""Typed Paper-I SNAKE configuration layers.

The manuscript is design authority, not a runtime configuration file.  This
module records the corresponding code contract without parsing LaTeX or mixing
historical compatibility controls into canonical Route A defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from pipelines.static_adapt.lane_routes import STATIC_LANE_ROUTE_PHYSICAL_OPERATOR_TYPE
from pipelines.static_adapt.route_identity import (
    ROUTE_ID_A,
    ROUTE_ID_B_LEGACY_PAIRWISE,
    ROUTE_ID_C,
)


PAPER_I_CONFIG_SCHEMA = "paper_i_static_snake_configuration_v1"
PAPER_I_MANUSCRIPT_COST_SEMANTICS = "K_t = 1 + weighted_normalized_burden"
PAPER_I_IMPLEMENTATION_COST_SEMANTICS = (
    "denominator = 1 + K_t_excess; K_t_excess is weighted_normalized_burden"
)


def _validated_subset_sizes(values: tuple[int, ...], *, field_name: str) -> tuple[int, ...]:
    resolved = tuple(int(value) for value in values)
    if not resolved:
        raise ValueError(f"{field_name} must contain at least one Pauli-word subset cardinality.")
    if any(value < 1 for value in resolved):
        raise ValueError(f"{field_name} values must all be positive integers.")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{field_name} must not contain duplicate cardinalities.")
    return resolved


@dataclass(frozen=True)
class PaperICostWeights:
    lambda_2q: float = 0.20
    lambda_d: float = 0.20
    lambda_1q: float = 0.05
    lambda_theta: float = 0.05
    lambda_shot: float = 0.15

    def __post_init__(self) -> None:
        for name, value in self.as_lambda_dict().items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"Paper-I cost weight {name!r} must be finite and nonnegative.")

    @property
    def enabled(self) -> bool:
        return any(float(value) > 0.0 for value in self.as_lambda_dict().values())

    def as_lambda_dict(self) -> dict[str, float]:
        return {
            "2q": float(self.lambda_2q),
            "d": float(self.lambda_d),
            "1q": float(self.lambda_1q),
            "theta": float(self.lambda_theta),
            "shot": float(self.lambda_shot),
        }

    @classmethod
    def disabled(cls) -> "PaperICostWeights":
        return cls(
            lambda_2q=0.0,
            lambda_d=0.0,
            lambda_1q=0.0,
            lambda_theta=0.0,
            lambda_shot=0.0,
        )


@dataclass(frozen=True)
class PaperIMethodCapabilities:
    phase_ids: tuple[int, ...] = (0, 1, 2, 3)
    cost_normalized_scoring: bool = True
    insertion_position_search: bool = True
    batching: bool = True
    beam_search: bool = True
    pruning: bool = True
    bounded_refit_windows: bool = True
    pauli_word_child_subsets: bool = True
    hard_symmetry_guard: bool = True
    pluggable_inner_optimizer: bool = True
    separate_noise_wrapper: bool = True
    hierarchical_macro_to_child_funnel: bool = True
    joint_ansatz_plus_batch_selector: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase_ids": [int(value) for value in self.phase_ids],
            "cost_normalized_scoring": bool(self.cost_normalized_scoring),
            "insertion_position_search": bool(self.insertion_position_search),
            "batching": bool(self.batching),
            "beam_search": bool(self.beam_search),
            "pruning": bool(self.pruning),
            "bounded_refit_windows": bool(self.bounded_refit_windows),
            "pauli_word_child_subsets": bool(self.pauli_word_child_subsets),
            "hard_symmetry_guard": bool(self.hard_symmetry_guard),
            "pluggable_inner_optimizer": bool(self.pluggable_inner_optimizer),
            "separate_noise_wrapper": bool(self.separate_noise_wrapper),
            "hierarchical_macro_to_child_funnel": bool(
                self.hierarchical_macro_to_child_funnel
            ),
            "joint_ansatz_plus_batch_selector": bool(
                self.joint_ansatz_plus_batch_selector
            ),
        }


@dataclass(frozen=True)
class CanonicalSnakeDefaults:
    route_id: str = ROUTE_ID_A
    first_class_route_ids: tuple[str, ...] = (ROUTE_ID_A,)
    benchmark_sibling_algorithms: tuple[str, ...] = ("append_only_adapt", "geo_adapt")
    cost_enabled: bool = True
    cost_weights: PaperICostWeights = field(default_factory=PaperICostWeights)
    canonical_stage_sequence: tuple[str, ...] = (
        "macro_phase1",
        "macro_phase2",
        "singleton_child_expansion",
        "global_child_phase1",
        "global_child_phase2",
        "joint_ansatz_plus_batch_selector",
        "admission_full_refit_prune",
    )
    phase0_enabled: bool = False
    macro_phase3_enabled: bool = False
    child_phase3_enabled: bool = False
    phase_score_family: str = "DeltaE_t/(1+K_t)"
    phase0_score_formula: str = "disabled_in_canonical_route"
    phase1_score_formula: str = "DeltaE1_TR/(1+K1)"
    phase2_score_formula: str = "DeltaE2_TR/(1+K2)"
    phase3_score_formula: str = "disabled_in_canonical_route"
    batch_score_formula: str = "DeltaE_joint_relaxed(B)/(1+K(B))"
    final_selection_authority: str = "joint_ansatz_plus_batch_schur_v1"
    joint_batch_context_mode: str = "full_ansatz_v1"
    beam_score_formula: str = "(E_root - E_b) / (1 + DeltaK_b)"
    batching_enabled: bool = True
    batch_selection_mode: str = "combinatorial_reduced_plane"
    beam_enabled: bool = False
    pruning_enabled: bool = True
    pauli_child_pool_enabled: bool = True
    phase3_candidate_population: str = "global_child_only_after_macro_phase2_v1"
    reoptimization_policy: str = "full"
    insertion_position_scope: str = "full"
    geometry_window_scope: str = "full"
    static_lane_route: str = STATIC_LANE_ROUTE_PHYSICAL_OPERATOR_TYPE
    shortlist_cap_policy: str = "fixed_explicit_phase_caps"
    maturity_scheduling_enabled: bool = False
    pauli_word_subset_sizes: tuple[int, ...] = (1,)
    child_symmetry_policy: str = "hard_guard"
    noise_execution_surface: str = "separate_wrapper"
    stop_policy: str = "max_iterations_primary_gradient_plateau_optional"

    def __post_init__(self) -> None:
        resolved = _validated_subset_sizes(
            self.pauli_word_subset_sizes,
            field_name="pauli_word_subset_sizes",
        )
        object.__setattr__(self, "pauli_word_subset_sizes", resolved)
        if bool(self.cost_enabled) and not self.cost_weights.enabled:
            raise ValueError("Canonical Paper-I cost is enabled but every cost weight is zero.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "route_id": str(self.route_id),
            "first_class_route_ids": [str(value) for value in self.first_class_route_ids],
            "benchmark_sibling_algorithms": [
                str(value) for value in self.benchmark_sibling_algorithms
            ],
            "cost_enabled": bool(self.cost_enabled),
            "cost_weights": self.cost_weights.as_lambda_dict(),
            "phase_score_family": str(self.phase_score_family),
            "canonical_stage_sequence": [
                str(value) for value in self.canonical_stage_sequence
            ],
            "phase0_enabled": bool(self.phase0_enabled),
            "macro_phase3_enabled": bool(self.macro_phase3_enabled),
            "child_phase3_enabled": bool(self.child_phase3_enabled),
            "phase0_score_formula": str(self.phase0_score_formula),
            "phase1_score_formula": str(self.phase1_score_formula),
            "phase2_score_formula": str(self.phase2_score_formula),
            "phase3_score_formula": str(self.phase3_score_formula),
            "batch_score_formula": str(self.batch_score_formula),
            "final_selection_authority": str(self.final_selection_authority),
            "joint_batch_context_mode": str(self.joint_batch_context_mode),
            "beam_score_formula": str(self.beam_score_formula),
            "batching_enabled": bool(self.batching_enabled),
            "batch_selection_mode": str(self.batch_selection_mode),
            "beam_enabled": bool(self.beam_enabled),
            "pruning_enabled": bool(self.pruning_enabled),
            "pauli_child_pool_enabled": bool(self.pauli_child_pool_enabled),
            "phase3_candidate_population": str(self.phase3_candidate_population),
            "reoptimization_policy": str(self.reoptimization_policy),
            "insertion_position_scope": str(self.insertion_position_scope),
            "geometry_window_scope": str(self.geometry_window_scope),
            "static_lane_route": str(self.static_lane_route),
            "shortlist_cap_policy": str(self.shortlist_cap_policy),
            "maturity_scheduling_enabled": bool(self.maturity_scheduling_enabled),
            "pauli_word_subset_sizes": [int(value) for value in self.pauli_word_subset_sizes],
            "pauli_word_subset_size_semantics": "exact_allowed_cardinalities",
            "child_symmetry_policy": str(self.child_symmetry_policy),
            "noise_execution_surface": str(self.noise_execution_surface),
            "stop_policy": str(self.stop_policy),
        }


@dataclass(frozen=True)
class PaperIHHDisplayedResultSettings:
    max_adapt_iterations: int = 30
    batching_enabled: bool = False
    beam_enabled: bool = False
    pruning_enabled: bool = True
    pauli_child_pool_enabled: bool = True
    phase3_candidate_population: str = "global_child_only_after_phase2_v1"
    full_coordinate_reoptimization: bool = True
    full_insertion_position_search: bool = True
    full_geometry_window: bool = True
    maturity_scheduling_enabled: bool = False
    pauli_word_subset_sizes: tuple[int, ...] = (1,)
    child_symmetry_policy: str = "hard_guard"
    qiskit_costing_is_post_run: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pauli_word_subset_sizes",
            _validated_subset_sizes(
                self.pauli_word_subset_sizes,
                field_name="pauli_word_subset_sizes",
            ),
        )
        if int(self.max_adapt_iterations) < 1:
            raise ValueError("max_adapt_iterations must be positive.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_adapt_iterations": int(self.max_adapt_iterations),
            "batching_enabled": bool(self.batching_enabled),
            "beam_enabled": bool(self.beam_enabled),
            "pruning_enabled": bool(self.pruning_enabled),
            "pauli_child_pool_enabled": bool(self.pauli_child_pool_enabled),
            "phase3_candidate_population": str(self.phase3_candidate_population),
            "full_coordinate_reoptimization": bool(self.full_coordinate_reoptimization),
            "full_insertion_position_search": bool(self.full_insertion_position_search),
            "full_geometry_window": bool(self.full_geometry_window),
            "maturity_scheduling_enabled": bool(self.maturity_scheduling_enabled),
            "pauli_word_subset_sizes": [int(value) for value in self.pauli_word_subset_sizes],
            "pauli_word_subset_size_semantics": "exact_allowed_cardinalities",
            "child_symmetry_policy": str(self.child_symmetry_policy),
            "qiskit_costing_is_post_run": bool(self.qiskit_costing_is_post_run),
        }


@dataclass(frozen=True)
class PaperIMechanismOverrides:
    cost_enabled: bool | None = None
    batching_enabled: bool | None = None
    beam_enabled: bool | None = None
    pruning_enabled: bool | None = None
    pauli_child_pool_enabled: bool | None = None
    batch_selection_mode: str | None = None
    geometry_window_size: int | None = None
    pauli_word_subset_sizes: tuple[int, ...] | None = None
    child_symmetry_policy: str | None = None

    def __post_init__(self) -> None:
        if self.geometry_window_size is not None and int(self.geometry_window_size) < 1:
            raise ValueError("geometry_window_size must be positive when specified.")
        if self.pauli_word_subset_sizes is not None:
            object.__setattr__(
                self,
                "pauli_word_subset_sizes",
                _validated_subset_sizes(
                    self.pauli_word_subset_sizes,
                    field_name="pauli_word_subset_sizes",
                ),
            )
        if self.batch_selection_mode is not None and self.batch_selection_mode not in {
            "greedy_reduced_plane",
            "combinatorial_reduced_plane",
        }:
            raise ValueError(
                "batch_selection_mode must be one of "
                "{'greedy_reduced_plane','combinatorial_reduced_plane'}."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "cost_enabled": self.cost_enabled,
            "batching_enabled": self.batching_enabled,
            "beam_enabled": self.beam_enabled,
            "pruning_enabled": self.pruning_enabled,
            "pauli_child_pool_enabled": self.pauli_child_pool_enabled,
            "batch_selection_mode": self.batch_selection_mode,
            "geometry_window_size": self.geometry_window_size,
            "pauli_word_subset_sizes": (
                None
                if self.pauli_word_subset_sizes is None
                else [int(value) for value in self.pauli_word_subset_sizes]
            ),
            "pauli_word_subset_size_semantics": "exact_allowed_cardinalities",
            "child_symmetry_policy": self.child_symmetry_policy,
        }


@dataclass(frozen=True)
class HistoricalCompatibilitySettings:
    legacy_route_ids: tuple[str, ...] = (ROUTE_ID_B_LEGACY_PAIRWISE, ROUTE_ID_C)
    legacy_algorithm_ids: tuple[str, ...] = ("tetris_adapt", "qeb_adapt", "ceo_adapt")
    legacy_routes_importable: bool = True
    legacy_routes_first_class: bool = False
    legacy_cli_quarantine_complete: bool = False
    legacy_cli_requires_explicit_gate: bool = True
    historical_payload_reading_enabled: bool = True
    legacy_beam_lambda_affects_route_a: bool = False
    maturity_scheduling_is_canonical: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "legacy_route_ids": [str(value) for value in self.legacy_route_ids],
            "legacy_algorithm_ids": [str(value) for value in self.legacy_algorithm_ids],
            "legacy_routes_importable": bool(self.legacy_routes_importable),
            "legacy_routes_first_class": bool(self.legacy_routes_first_class),
            "legacy_cli_quarantine_complete": bool(self.legacy_cli_quarantine_complete),
            "legacy_cli_requires_explicit_gate": bool(
                self.legacy_cli_requires_explicit_gate
            ),
            "historical_payload_reading_enabled": bool(
                self.historical_payload_reading_enabled
            ),
            "legacy_beam_lambda_affects_route_a": bool(
                self.legacy_beam_lambda_affects_route_a
            ),
            "maturity_scheduling_is_canonical": bool(
                self.maturity_scheduling_is_canonical
            ),
        }


@dataclass(frozen=True)
class PaperIConfiguration:
    capabilities: PaperIMethodCapabilities = field(default_factory=PaperIMethodCapabilities)
    canonical: CanonicalSnakeDefaults = field(default_factory=CanonicalSnakeDefaults)
    hh_displayed_results: PaperIHHDisplayedResultSettings = field(
        default_factory=PaperIHHDisplayedResultSettings
    )
    mechanism_overrides: PaperIMechanismOverrides = field(
        default_factory=PaperIMechanismOverrides
    )
    historical_compatibility: HistoricalCompatibilitySettings = field(
        default_factory=HistoricalCompatibilitySettings
    )

    def cost_weights(self, *, enabled: bool | None = None) -> PaperICostWeights:
        use_cost = bool(self.canonical.cost_enabled if enabled is None else enabled)
        return self.canonical.cost_weights if use_cost else PaperICostWeights.disabled()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": PAPER_I_CONFIG_SCHEMA,
            "authority": "typed_code_contract_not_runtime_manuscript_parse",
            "manuscript_cost_semantics": PAPER_I_MANUSCRIPT_COST_SEMANTICS,
            "implementation_cost_semantics": PAPER_I_IMPLEMENTATION_COST_SEMANTICS,
            "capabilities": self.capabilities.as_dict(),
            "canonical": self.canonical.as_dict(),
            "hh_displayed_results": self.hh_displayed_results.as_dict(),
            "mechanism_overrides": self.mechanism_overrides.as_dict(),
            "historical_compatibility": self.historical_compatibility.as_dict(),
        }


PAPER_I_CONFIGURATION = PaperIConfiguration()
PAPER_I_CANONICAL_COST_WEIGHTS = PAPER_I_CONFIGURATION.canonical.cost_weights


__all__ = [
    "CanonicalSnakeDefaults",
    "HistoricalCompatibilitySettings",
    "PAPER_I_CANONICAL_COST_WEIGHTS",
    "PAPER_I_CONFIGURATION",
    "PAPER_I_CONFIG_SCHEMA",
    "PAPER_I_IMPLEMENTATION_COST_SEMANTICS",
    "PAPER_I_MANUSCRIPT_COST_SEMANTICS",
    "PaperIConfiguration",
    "PaperICostWeights",
    "PaperIHHDisplayedResultSettings",
    "PaperIMechanismOverrides",
    "PaperIMethodCapabilities",
]
