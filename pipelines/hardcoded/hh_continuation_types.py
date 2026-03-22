#!/usr/bin/env python3
"""Shared continuation datamodel for HH ADAPT -> replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


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
    h_hat: float | None = None
    b_hat: list[float] | None = None
    H_window: list[list[float]] | None = None
    depth_cost: float = 0.0
    new_group_cost: float = 0.0
    new_shot_cost: float = 0.0
    opt_dim_cost: float = 0.0
    reuse_count_cost: float = 0.0
    full_v2_score: float | None = None
    shortlist_rank: int | None = None
    shortlist_size: int | None = None
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


@dataclass(frozen=True)
class MeasurementPlan:
    plan_version: str
    group_keys: list[str]
    nominal_shots_per_group: int
    grouping_mode: str


@dataclass(frozen=True)
class MeasurementCacheStats:
    groups_total: int
    groups_reused: int
    groups_new: int
    shots_reused: float
    shots_new: float
    reuse_count_cost: float


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
