#!/usr/bin/env python3
"""Shared continuation datamodel for HH ADAPT -> replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class PhaseControllerSnapshot:
    step_index: int
    depth_local: int
    depth_left: int
    runway_ratio: float
    early_coordinate: float
    late_coordinate: float
    frontier_ratio: float
    u_stag: float = 0.0
    m_t: float = 0.0
    s_t: float = 0.0
    rho_t: float = 1.0
    gamma_t: float = 0.0
    u_front: float = 1.0
    n_rem_hat: float = 0.0
    useful_horizon: float = 0.0
    runway_fraction: float = 0.0
    H_t: float = 0.0
    phase_thresholds: dict[str, float] = field(default_factory=dict)
    phase_caps: dict[str, int] = field(default_factory=dict)
    phase_shots: dict[str, int] = field(default_factory=dict)
    phase_uncertainty: dict[str, float] = field(default_factory=dict)
    snapshot_version: str = "phase123_controller_v1"
    depth_runway_ratio: float = 0.0
    n_rem_low: float = 0.0
    n_rem_high: float = 0.0
    confidence_ratio: float = 1.0
    phase_live: dict[str, bool] = field(default_factory=dict)
    terminal_phase: int = 3
    phase_null_reasons: dict[str, str] = field(default_factory=dict)
    phase_null_streaks: dict[str, int] = field(default_factory=dict)
    phase_caps_scheduled: dict[str, int] = field(default_factory=dict)
    phase_shots_maturity_floor: dict[str, int] = field(default_factory=dict)
    phase_shots_scheduled: dict[str, int] = field(default_factory=dict)
    phase_shots_snr: dict[str, int] = field(default_factory=dict)
    phase_shots_effective: dict[str, int] = field(default_factory=dict)
    phase_shot_uplift: dict[str, float] = field(default_factory=dict)
    phase_shot_fraction: dict[str, float] = field(default_factory=dict)
    phase_signal: dict[str, float] = field(default_factory=dict)
    phase_signal_floor: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ScaffoldCoordinateMetadata:
    candidate_label: str
    generator_id: str | None
    admission_step: int
    first_seen_step: int
    selector_score: float
    selector_burden: float
    cooldown_remaining: int = 0
    cumulative_abs_motion: float = 0.0
    recent_abs_motion: float = 0.0
    stagnation_score: float = 0.0
    current_abs_theta: float = 0.0
    previous_abs_theta: float = 0.0
    peak_abs_theta: float = 0.0
    amplitude_observation_count: int = 0


@dataclass(frozen=True)
class MaturePruneTrial:
    selector_step: int
    gate_open: bool
    probe_indices: list[int]
    selected_index: int | None
    selected_label: str | None
    frozen_regression: float | None
    refit_energy: float | None
    retained_gain: float | None
    accepted: bool
    rollback_reason: str | None = None
    amplitude_witness_ok: bool = False
    amplitude_witness_reason: str | None = None
    amplitude_witness: dict[str, Any] | None = None


@dataclass(frozen=True)
class CandidateFeatures:
    stage_name: str
    candidate_label: str
    candidate_family: str
    candidate_pool_index: int
    position_id: int
    append_position: int
    positions_considered: list[int]
    g_signed: float
    g_abs: float
    g_lcb: float
    sigma_hat: float
    F_metric: float
    metric_proxy: float
    novelty: float | None
    curvature_mode: str
    novelty_mode: str
    refit_window_indices: list[int]
    compiled_position_cost_proxy: dict[str, float]
    measurement_cache_stats: dict[str, float]
    leakage_penalty: float
    stage_gate_open: bool
    leakage_gate_open: bool
    trough_probe_triggered: bool
    trough_detected: bool
    simple_score: float | None
    score_version: str
    F_raw: float | None = None
    h_eff: float | None = None
    F_red: float | None = None
    ridge_used: float | None = None
    cheap_score: float | None = None
    cheap_score_version: str = "simple_v1"
    cheap_metric_proxy: float = 0.0
    cheap_benefit_proxy: float | None = None
    cheap_burden_total: float | None = None
    phase1_score_mode: str = "trust_region_v1"
    phase1_active_score: float | None = None
    phase1_legacy_simple_score: float | None = None
    phase1_trust_region_gain: float | None = None
    phase1_trust_region_score: float | None = None
    phase1_rho: float | None = None
    phase1_burden_total: float | None = None
    h_hat: float | None = None
    b_hat: list[float] | None = None
    H_window: list[list[float]] | None = None
    phase2_joint_geometry_reuse: dict[str, Any] = field(default_factory=dict)
    historical_singleton_phase2_coordinate_model: dict[str, Any] = field(
        default_factory=dict
    )
    depth_cost: float = 0.0
    new_group_cost: float = 0.0
    new_shot_cost: float = 0.0
    opt_dim_cost: float = 0.0
    reuse_count_cost: float = 0.0
    c_hat_2q: float = 0.0
    c_hat_d: float = 0.0
    c_hat_1q: float = 0.0
    c_hat_theta: float = 0.0
    c_hat_shot: float = 0.0
    c_bar_2q: float = 0.0
    c_bar_d: float = 0.0
    c_bar_1q: float = 0.0
    c_bar_theta: float = 0.0
    c_bar_shot: float = 0.0
    hardware_cost_excess_sum: float = 0.0
    hardware_cost_denominator: float = 1.0
    hardware_cost_normalization: dict[str, Any] = field(default_factory=dict)
    hardware_cost_lambdas: dict[str, float] = field(default_factory=dict)
    hardware_cost_lambda_source: str = "unresolved"
    hardware_cost_source: str = "unset"
    family_repeat_cost: float = 0.0
    full_v2_score: float | None = None
    shortlist_rank: int | None = None
    shortlist_size: int | None = None
    route_a_shortlist_unit: str | None = None
    route_a_shortlist_identity: str | None = None
    route_a_identity_rank: int | None = None
    route_a_identity_shortlist_size: int | None = None
    route_a_identity_position_rank: int | None = None
    route_a_identity_position_count: int | None = None
    actual_fallback_mode: str = "simple_v1_only"
    compatibility_penalty_total: float = 0.0
    generator_id: str | None = None
    template_id: str | None = None
    is_macro_generator: bool = False
    parent_generator_id: str | None = None
    runtime_split_mode: str = "off"
    runtime_split_parent_label: str | None = None
    runtime_split_child_index: int | None = None
    runtime_split_child_count: int | None = None
    runtime_split_chosen_representation: str = "parent"
    runtime_split_child_indices: list[int] = field(default_factory=list)
    runtime_split_child_labels: list[str] = field(default_factory=list)
    runtime_split_child_generator_ids: list[str] = field(default_factory=list)
    generator_metadata: dict[str, Any] | None = None
    symmetry_spec: dict[str, Any] | None = None
    symmetry_mode: str = "none"
    symmetry_mitigation_mode: str = "off"
    motif_metadata: dict[str, Any] | None = None
    motif_bonus: float = 0.0
    motif_source: str = "none"
    remaining_evaluations_proxy: float = 0.0
    remaining_evaluations_proxy_mode: str = "none"
    lifetime_cost_mode: str = "off"
    lifetime_weight_components: dict[str, float] = field(default_factory=dict)
    placeholder_hooks: dict[str, bool] = field(default_factory=dict)
    compile_cost_source: str = "proxy"
    compile_cost_total: float = 0.0
    compile_gate_open: bool = True
    compile_failure_reason: str | None = None
    compiled_position_cost_backend: dict[str, Any] | None = None
    phase_score_components: dict[str, float] = field(default_factory=dict)
    phase_cost_components: dict[str, float] = field(default_factory=dict)
    confidence_factor: float = 1.0
    phase2_raw_overlap_max: float | None = None
    phase2_raw_novelty: float | None = None
    phase2_novelty_mode: str = "collective_span_v1"
    phase2_novelty_source: str = "collective_span_v1"
    phase2_novelty_fallback_reason: str | None = None
    phase2_span_projection_z: float | None = None
    phase2_novelty_ridge_used: float | None = None
    phase2_raw_F_effective: float | None = None
    phase2_legacy_pairwise_novelty: float | None = None
    phase2_confidence_applied: bool = False
    phase2_raw_score_formula: str = "DeltaE_TR_raw * N2 / (1 + K2)"
    phase2_raw_trust_gain: float | None = None
    phase2_raw_score: float | None = None
    phase2_burden_total: float | None = None
    phase3_reduced_novelty: float | None = None
    phase3_reduced_trust_gain: float | None = None
    phase3_burden_total: float | None = None
    phase3_primary_score: float | None = None
    phase3_tie_break_score: float = 0.0
    phase3_auxiliary_score_mode: str = "tie_break_only"
    phase3_canonical_score_formula: str = "DeltaE_TR * N3 / (1 + K3)"
    selector_score: float | None = None
    selector_burden: float | None = None
    selector_geometry_mode: str = "reduced"
    controller_snapshot: dict[str, Any] | None = None
    refit_window_basis: str = "old_pre_geometry_alias"
    phase2_geometry_window_indices: list[int] = field(default_factory=list)
    phase2_geometry_window_policy: str = "legacy_refit_window_alias"
    schur_window_indices: list[int] = field(default_factory=list)
    schur_window_policy: str = "phase3_geometry_refit_window_alias"
    window_origin: str = "legacy"
    window_new_indices: list[int] = field(default_factory=list)
    window_age_indices: list[int] = field(default_factory=list)
    phase1_shortlisted: bool = False
    phase2_shortlisted: bool = False
    phase3_shortlisted: bool = False
    child_phase1_shortlisted: bool = False
    child_phase2_shortlisted: bool = False
    child_phase3_shortlisted: bool = False
    phase3_duplicate_penalty: float = 0.0
    algebraic_lane: str | None = None
    algebraic_quality: str | None = None
    algebraic_context_counts: dict[str, int] = field(default_factory=dict)
    algebraic_context_labels: list[str] = field(default_factory=list)
    algebraic_lane_health: float | None = None
    algebraic_lane_relative_health: float | None = None
    algebraic_lane_live: bool | None = None
    static_lane_route: str = "algebraic"
    physical_operator_lane: str | None = None
    physical_operator_quality: str | None = None
    physical_operator_hh_full_meta_class: str | None = None
    physical_operator_classifier_version: str | None = None
    physical_operator_classifier_label: str | None = None
    physical_operator_lane_source: str | None = None
    physical_operator_lane_health: float | None = None
    physical_operator_lane_relative_health: float | None = None
    physical_operator_lane_live: bool | None = None
    nested_refit_window: dict[str, Any] | None = None
    nested_window_accounting: dict[str, Any] | None = None
    route_c_plateau_acquisition: dict[str, Any] | None = None
    nested_refit_window_status: str = "policy_inactive"
    inherited_refit_window_indices: list[int] = field(default_factory=list)
    active_post_refit_indices: list[int] = field(default_factory=list)
    selection_inherited_old_indices: list[int] = field(default_factory=list)
    phase3_geometry_window_policy: str = "legacy_coupled"
    phase3_geometry_window_size: int = 0
    phase3_geometry_refit_window_indices: list[int] = field(default_factory=list)
    phase3_geometry_active_post_indices: list[int] = field(default_factory=list)
    phase3_geometry_nested_refit_window: dict[str, Any] | None = None
    phase3_geometry_window_accounting: dict[str, Any] | None = None
    w3_wopt_decoupled: bool = False
    optimizer_active_refit_indices: list[int] = field(default_factory=list)
    optimizer_active_refit_count: int = 0
    compile_proxy_basis: str = "legacy"
    compile_proxy_refit_count: int = 0
    phase3_selector_policy: str = "hardware_resolvable_v1"
    phase3_score_policy: str = "hardware_resolvable_v1"
    epsilon_g_shot: float = 0.0
    b_g_hw: float = 0.0
    b_g_drift: float = 0.0
    epsilon_g_res: float = 0.0
    g_hw_lcb: float = 0.0
    g_lcb_legacy_shot: float = 0.0
    hardware_resolution_mode: str = "ideal"
    hardware_resolution_source: str = "legacy_unset"
    phase0_pilot_schema: str = "inactive"
    phase0_pilot_enabled: bool = False
    phase0_pilot_retained: bool | None = None
    phase0_filter_reason: str = "not_evaluated"
    phase0_scope_label: str | None = None
    phase0_pilot_rank: int | None = None
    phase0_pilot_size: int | None = None
    phase0_lane_rank: int | None = None
    phase0_lane_size: int | None = None
    phase0_lane_budget: int | None = None
    phase0_raw_gradient_signed: float | None = None
    phase0_raw_gradient_abs: float | None = None
    phase0_sigma_hat: float = 0.0
    phase0_sigma_hat_available: bool = False
    phase0_sigma_source: str = "unavailable"
    phase0_epsilon_g_shot: float = 0.0
    phase0_b_g_hw: float = 0.0
    phase0_b_g_drift: float = 0.0
    phase0_epsilon_g_res: float = 0.0
    phase0_g_upper_hw: float | None = None
    phase0_delta_e_upper_hw: float | None = None
    phase0_novelty: float = 1.0
    phase0_score: float | None = None
    phase0_score_formula: str = "DeltaE0_upper * N0 / K0"
    phase0_K0: float = 1.0
    phase0_hardware_cost_denominator: float = 1.0
    phase0_hardware_cost_excess_sum: float = 0.0
    phase0_cost_raw_components: dict[str, float] = field(default_factory=dict)
    phase0_cost_bar_components: dict[str, float] = field(default_factory=dict)
    phase0_cost_lambdas: dict[str, float] = field(default_factory=dict)
    phase0_cost_lambda_source: str = "unresolved"
    phase0_cost_normalization_schema: str = "unresolved"
    phase0_cost_enabled: bool = False
    phase0_alpha: float | None = None
    phase0_threshold: float | None = None
    phase0_hardware_resolution_mode: str = "ideal"
    phase0_hardware_resolution_source: str = "legacy_unset"
    phase0_algebraic_lane: str | None = None
    phase0_algebraic_quality: str | None = None
    phase0_algebraic_context_counts: dict[str, int] = field(default_factory=dict)
    phase0_algebraic_context_labels: list[str] = field(default_factory=list)
    schur_window_solve: list[float] | None = None
    phase1_order_rescue_shortlisted: bool = False


@dataclass(frozen=True)
class MeasurementPlan:
    plan_version: str
    group_keys: list[str]
    nominal_shots_per_group: int
    grouping_mode: str


@dataclass(frozen=True)
class MeasurementGroupSpec:
    group_key: str
    coeff_l2: float = 1.0
    term_count: int = 1
    source: str = "legacy_string_key"


@dataclass(frozen=True)
class MeasurementCacheStats:
    groups_total: int
    groups_reused: int
    groups_new: int
    shots_reused: float
    shots_new: float
    reuse_count_cost: float
    shot_cost_proxy: float = 0.0
    new_group_coeff_l2_sum: float = 0.0
    sigma_star: float = 1.0
    new_group_term_count: int = 0
    measurement_cost_source: str = "legacy_string_key"


@dataclass(frozen=True)
class CompileCostEstimate:
    new_pauli_actions: float
    new_rotation_steps: float
    position_shift_span: float
    refit_active_count: float
    proxy_total: float
    cx_proxy_total: float = 0.0
    sq_proxy_total: float = 0.0
    gate_proxy_total: float = 0.0
    max_pauli_weight: float = 0.0
    c_hat_2q: float = 0.0
    c_hat_d: float = 0.0
    c_hat_1q: float = 0.0
    c_hat_theta: float = 0.0
    hardware_cost_source: str = "proxy_logical_ladder_span_v1"
    source_mode: str = "proxy"
    penalty_total: float | None = None
    depth_surrogate: float | None = None
    compile_gate_open: bool = True
    failure_reason: str | None = None
    selected_backend_name: str | None = None
    selected_resolution_kind: str | None = None
    aggregation_mode: str = "proxy"
    target_backend_names: list[str] = field(default_factory=list)
    successful_target_count: int = 0
    failed_target_count: int = 0
    raw_delta_compiled_count_2q: float | None = None
    delta_compiled_count_2q: float | None = None
    raw_delta_compiled_depth: float | None = None
    delta_compiled_depth: float | None = None
    raw_delta_compiled_depth_2q: float | None = None
    delta_compiled_depth_2q: float | None = None
    raw_delta_compiled_size: float | None = None
    delta_compiled_size: float | None = None
    delta_compiled_cx_count: float | None = None
    delta_compiled_ecr_count: float | None = None
    base_compiled_count_2q: float | None = None
    base_compiled_depth: float | None = None
    base_compiled_size: float | None = None
    trial_compiled_count_2q: float | None = None
    trial_compiled_depth: float | None = None
    trial_compiled_size: float | None = None
    proxy_baseline: dict[str, float] | None = None
    selected_backend_row: dict[str, Any] | None = None


@dataclass(frozen=True)
class ScaffoldFingerprintLite:
    selected_operator_labels: list[str]
    selected_generator_ids: list[str]
    num_parameters: int
    generator_family: str
    continuation_mode: str
    compiled_pauli_cache_size: int
    measurement_plan_version: str
    post_prune: bool
    split_event_count: int = 0
    motif_record_ids: list[str] = field(default_factory=list)
    compile_cost_mode: str = "proxy"
    backend_target_names: list[str] = field(default_factory=list)
    backend_reduction_mode: str = "none"


@dataclass(frozen=True)
class PruneDecision:
    index: int
    label: str
    accepted: bool
    energy_before: float
    energy_after: float
    regression: float
    reason: str
    safe_regression_ok: bool = True
    retained_gain_ok: bool = True
    regression_threshold: float | None = None
    retained_gain: float | None = None
    retained_gain_threshold: float | None = None
    amplitude_witness_ok: bool = True
    amplitude_witness_reason: str | None = None
    amplitude_current_abs: float | None = None
    amplitude_peak_abs: float | None = None
    amplitude_drop_abs: float | None = None
    amplitude_collapse_ratio: float | None = None
    amplitude_observation_count: int | None = None
    rung_index: int | None = None
    rung_kind: str | None = None
    confidence_model: str = "deterministic_sigma0"
    confidence_sigma: float = 0.0
    confidence_upper_regression: float | None = None
    confidence_guard_ok: bool = True
    curvature_guard_mode: str = "off"
    curvature_guard_active: bool = False
    curvature_guard_ok: bool = True
    curvature_guard_reason: str = "guard_off"
    acceptance_source: str = "remove_refit_energy_safety"
    surrogate_used_for_acceptance: bool = False


@dataclass(frozen=True)
class ReplayPlan:
    continuation_mode: str
    seed_policy_resolved: str
    handoff_state_kind: str
    freeze_scaffold_steps: int
    unfreeze_steps: int
    full_replay_steps: int
    trust_radius_initial: float
    trust_radius_growth: float
    trust_radius_max: float
    scaffold_block_indices: list[int]
    residual_block_indices: list[int]
    qn_spsa_refresh_every: int
    trust_radius_schedule: list[float]
    optimizer_memory_source: str = "unavailable"
    optimizer_memory_reused: bool = False
    refresh_mode: str = "disabled"
    symmetry_mitigation_mode: str = "off"
    generator_ids: list[str] = field(default_factory=list)
    motif_reference_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReplayPhaseTelemetry:
    phase_name: str
    nfev: int
    nit: int
    success: bool
    energy_before: float
    energy_after: float
    delta_abs_before: float | None
    delta_abs_after: float | None
    active_count: int
    frozen_count: int
    optimizer_memory_reused: bool = False
    optimizer_memory_source: str = "unavailable"
    qn_spsa_refresh_points: list[int] = field(default_factory=list)
    residual_zero_initialized: bool = True


class NoveltyOracle(Protocol):
    def estimate(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:  # pragma: no cover - interface
        ...


class CurvatureOracle(Protocol):
    def estimate(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:  # pragma: no cover - interface
        ...


class OptimizerMemoryAdapter(Protocol):
    def unavailable(self, *, method: str, parameter_count: int, reason: str) -> dict[str, Any]:
        ...

    def from_result(
        self,
        result: Any,
        *,
        method: str,
        parameter_count: int,
        source: str,
    ) -> dict[str, Any]:
        ...

    def remap_insert(
        self,
        state: Mapping[str, Any] | None,
        *,
        position_id: int,
        count: int = 1,
    ) -> dict[str, Any]:
        ...

    def remap_remove(
        self,
        state: Mapping[str, Any] | None,
        *,
        indices: Sequence[int],
    ) -> dict[str, Any]:
        ...

    def select_active(
        self,
        state: Mapping[str, Any] | None,
        *,
        active_indices: Sequence[int],
        source: str,
    ) -> dict[str, Any]:
        ...

    def merge_active(
        self,
        base_state: Mapping[str, Any] | None,
        *,
        active_indices: Sequence[int],
        active_state: Mapping[str, Any] | None,
        source: str,
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class QNSPSARefreshPlan:
    enabled: bool = False
    refresh_every: int = 0
    mode: str = "disabled"
    skip_reason: str = ""
    refresh_points: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class MotifMetadata:
    enabled: bool = False
    motif_tags: list[str] = field(default_factory=list)
    motif_ids: list[str] = field(default_factory=list)
    motif_source: str = "none"
    tiled_from_num_sites: int | None = None
    target_num_sites: int | None = None
    boundary_behavior: str | None = None
    transfer_mode: str = "exact_match_v1"


@dataclass(frozen=True)
class SymmetrySpec:
    spec_version: str = "phase3_symmetry_v1"
    particle_number_mode: str = "preserving"
    spin_sector_mode: str = "preserving"
    phonon_number_mode: str = "not_conserved"
    leakage_risk: float = 0.0
    mitigation_eligible: bool = False
    grouping_eligible: bool = True
    hard_guard: bool = False
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GeneratorMetadata:
    generator_id: str
    family_id: str
    template_id: str
    candidate_label: str
    support_qubits: list[int]
    support_sites: list[int]
    support_site_offsets: list[int]
    is_macro_generator: bool
    split_policy: str
    parent_generator_id: str | None = None
    symmetry_spec: dict[str, Any] | None = None
    compile_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratorSplitEvent:
    parent_generator_id: str
    child_generator_ids: list[str]
    reason: str
    split_mode: str
    probe_trigger: str | None = None
    choice_reason: str | None = None
    parent_score: float | None = None
    child_scores: dict[str, float] = field(default_factory=dict)
    admissible_child_subsets: list[list[str]] = field(default_factory=list)
    chosen_representation: str = "parent"
    chosen_child_ids: list[str] = field(default_factory=list)
    split_margin: float | None = None
    symmetry_gate_results: dict[str, Any] = field(default_factory=dict)
    parent_collapse_diagnostic: dict[str, Any] = field(default_factory=dict)
    compiled_cost_parent: float | None = None
    compiled_cost_children: float | None = None
    insertion_positions: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class MotifRecord:
    motif_id: str
    family_id: str
    template_id: str
    source_num_sites: int
    relative_order: int
    support_site_offsets: list[int]
    mean_theta: float
    mean_abs_theta: float
    sign_hint: int
    generator_ids: list[str]
    symmetry_spec: dict[str, Any] | None = None
    boundary_behavior: str = "interior_only"
    source_tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MotifLibrary:
    library_version: str
    source_tag: str
    source_num_sites: int
    ordering: str
    boson_encoding: str
    source_tags: list[str] = field(default_factory=list)
    records: list[MotifRecord] = field(default_factory=list)


@dataclass(frozen=True)
class RescueDiagnostic:
    enabled: bool = False
    triggered: bool = False
    reason: str = "disabled"
    shortlisted_labels: list[str] = field(default_factory=list)
    selected_label: str | None = None
    selected_position: int | None = None
    overlap_gain: float = 0.0


class Phase2OptimizerMemoryAdapter:
    """Deterministic remapping adapter for persisted optimizer memory."""

    _VECTOR_KEYS = (
        "preconditioner_diag",
        "grad_sq_ema",
    )

    def unavailable(self, *, method: str, parameter_count: int, reason: str) -> dict[str, Any]:
        return {
            "version": "phase2_optimizer_memory_v1",
            "optimizer": str(method),
            "parameter_count": int(max(0, parameter_count)),
            "available": False,
            "reason": str(reason),
            "source": "unavailable",
            "reused": False,
            "preconditioner_diag": [1.0] * int(max(0, parameter_count)),
            "grad_sq_ema": [0.0] * int(max(0, parameter_count)),
            "history_tail": [],
            "refresh_points": [],
            "remap_events": [],
        }

    def from_result(
        self,
        result: Any,
        *,
        method: str,
        parameter_count: int,
        source: str,
    ) -> dict[str, Any]:
        raw = getattr(result, "optimizer_memory", None)
        if not isinstance(raw, Mapping):
            return self.unavailable(
                method=str(method),
                parameter_count=int(parameter_count),
                reason="optimizer_memory_missing",
            )
        state = self._normalize(raw, parameter_count=int(parameter_count))
        state["source"] = str(source)
        return state

    def remap_insert(
        self,
        state: Mapping[str, Any] | None,
        *,
        position_id: int,
        count: int = 1,
    ) -> dict[str, Any]:
        base = self._normalize(state, parameter_count=self._parameter_count(state))
        n = int(base["parameter_count"])
        pos = max(0, min(int(position_id), n))
        add_n = int(max(0, count))
        for key, default in (("preconditioner_diag", 1.0), ("grad_sq_ema", 0.0)):
            vec = list(base.get(key, []))
            base[key] = vec[:pos] + ([float(default)] * add_n) + vec[pos:]
        base["parameter_count"] = int(n + add_n)
        self._append_remap_event(base, {"op": "insert", "position_id": int(pos), "count": int(add_n)})
        return base

    def remap_remove(
        self,
        state: Mapping[str, Any] | None,
        *,
        indices: Sequence[int],
    ) -> dict[str, Any]:
        base = self._normalize(state, parameter_count=self._parameter_count(state))
        n = int(base["parameter_count"])
        drop = sorted({int(i) for i in indices if 0 <= int(i) < n})
        keep = [i for i in range(n) if i not in set(drop)]
        for key in self._VECTOR_KEYS:
            vec = list(base.get(key, []))
            base[key] = [float(vec[i]) for i in keep]
        base["parameter_count"] = int(len(keep))
        self._append_remap_event(base, {"op": "remove", "indices": [int(i) for i in drop]})
        return base

    def select_active(
        self,
        state: Mapping[str, Any] | None,
        *,
        active_indices: Sequence[int],
        source: str,
    ) -> dict[str, Any]:
        base = self._normalize(state, parameter_count=self._parameter_count(state))
        n = int(base["parameter_count"])
        active = [int(i) for i in active_indices if 0 <= int(i) < n]
        out = {
            **base,
            "parameter_count": int(len(active)),
            "preconditioner_diag": [float(base["preconditioner_diag"][i]) for i in active],
            "grad_sq_ema": [float(base["grad_sq_ema"][i]) for i in active],
            "source": str(source),
            "reused": bool(base.get("available", False) and len(active) > 0),
            "active_indices": [int(i) for i in active],
        }
        self._append_remap_event(out, {"op": "select_active", "active_indices": [int(i) for i in active]})
        return out

    def merge_active(
        self,
        base_state: Mapping[str, Any] | None,
        *,
        active_indices: Sequence[int],
        active_state: Mapping[str, Any] | None,
        source: str,
    ) -> dict[str, Any]:
        base = self._normalize(base_state, parameter_count=self._parameter_count(base_state))
        active_norm = self._normalize(active_state, parameter_count=len(list(active_indices)))
        n = int(base["parameter_count"])
        active = [int(i) for i in active_indices if 0 <= int(i) < n]
        for key, default in (("preconditioner_diag", 1.0), ("grad_sq_ema", 0.0)):
            vec = list(base.get(key, [float(default)] * n))
            active_vec = list(active_norm.get(key, []))
            for k, idx in enumerate(active):
                if k < len(active_vec):
                    vec[idx] = float(active_vec[k])
            base[key] = vec
        base["source"] = str(source)
        base["available"] = bool(base.get("available", False) or active_norm.get("available", False))
        base["reused"] = bool(active_norm.get("reused", False))
        refresh = list(base.get("refresh_points", []))
        refresh.extend(int(x) for x in active_norm.get("refresh_points", []) if int(x) not in refresh)
        base["refresh_points"] = refresh
        self._append_remap_event(base, {"op": "merge_active", "active_indices": [int(i) for i in active]})
        return base

    def _parameter_count(self, state: Mapping[str, Any] | None) -> int:
        if isinstance(state, Mapping) and state.get("parameter_count") is not None:
            return int(max(0, int(state.get("parameter_count", 0))))
        if isinstance(state, Mapping):
            for key in self._VECTOR_KEYS:
                raw = state.get(key, None)
                if isinstance(raw, Sequence):
                    return int(len(list(raw)))
        return 0

    def _normalize(self, state: Mapping[str, Any] | None, *, parameter_count: int) -> dict[str, Any]:
        n = int(max(0, parameter_count))
        if not isinstance(state, Mapping):
            return self.unavailable(method="unknown", parameter_count=n, reason="missing_state")
        out = {
            "version": str(state.get("version", "phase2_optimizer_memory_v1")),
            "optimizer": str(state.get("optimizer", "unknown")),
            "parameter_count": int(n),
            "available": bool(state.get("available", False)),
            "reason": str(state.get("reason", "")),
            "source": str(state.get("source", "")),
            "reused": bool(state.get("reused", False)),
            "history_tail": [dict(x) for x in state.get("history_tail", []) if isinstance(x, Mapping)][-32:],
            "refresh_points": [int(x) for x in state.get("refresh_points", [])],
            "remap_events": [dict(x) for x in state.get("remap_events", []) if isinstance(x, Mapping)][-32:],
        }
        for key, default in (("preconditioner_diag", 1.0), ("grad_sq_ema", 0.0)):
            raw = list(state.get(key, [])) if isinstance(state.get(key, []), Sequence) else []
            vec = [float(default)] * n
            for i in range(min(n, len(raw))):
                vec[i] = float(raw[i])
            out[key] = vec
        return out

    def _append_remap_event(self, state: dict[str, Any], event: Mapping[str, Any]) -> None:
        events = [dict(x) for x in state.get("remap_events", []) if isinstance(x, Mapping)]
        events.append({str(k): v for k, v in event.items()})
        state["remap_events"] = events[-32:]
