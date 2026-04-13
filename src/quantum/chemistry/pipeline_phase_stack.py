from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import (
    CompiledPolynomialAction,
    apply_compiled_polynomial,
    compile_polynomial_action,
)
from src.quantum.pauli_actions import apply_compiled_pauli


@dataclass(frozen=True)
class PhaseControllerSnapshot:
    step_index: int
    depth_local: int
    depth_left: int
    runway_ratio: float
    early_coordinate: float
    late_coordinate: float
    frontier_ratio: float
    phase_thresholds: dict[str, float] = field(default_factory=dict)
    phase_caps: dict[str, int] = field(default_factory=dict)
    phase_shots: dict[str, int] = field(default_factory=dict)
    phase_uncertainty: dict[str, float] = field(default_factory=dict)
    snapshot_version: str = "chemistry_phase123_controller_v1"


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
    h_hat: float | None = None
    b_hat: list[float] | None = None
    H_window: list[list[float]] | None = None
    depth_cost: float = 0.0
    new_group_cost: float = 0.0
    new_shot_cost: float = 0.0
    opt_dim_cost: float = 0.0
    reuse_count_cost: float = 0.0
    family_repeat_cost: float = 0.0
    full_v2_score: float | None = None
    shortlist_rank: int | None = None
    shortlist_size: int | None = None
    actual_fallback_mode: str = "simple_v1_only"
    compatibility_penalty_total: float = 0.0
    generator_id: str | None = None
    generator_metadata: dict[str, Any] | None = None
    remaining_evaluations_proxy: float = 0.0
    remaining_evaluations_proxy_mode: str = "none"
    lifetime_cost_mode: str = "off"
    lifetime_weight_components: dict[str, float] = field(default_factory=dict)
    compile_cost_source: str = "proxy"
    compile_cost_total: float = 0.0
    compile_gate_open: bool = True
    compile_failure_reason: str | None = None
    phase_score_components: dict[str, float] = field(default_factory=dict)
    phase_cost_components: dict[str, float] = field(default_factory=dict)
    confidence_factor: float = 1.0
    phase2_raw_overlap_max: float | None = None
    phase2_raw_novelty: float | None = None
    phase2_raw_trust_gain: float | None = None
    phase2_raw_score: float | None = None
    phase2_burden_total: float | None = None
    phase3_reduced_novelty: float | None = None
    phase3_reduced_trust_gain: float | None = None
    phase3_burden_total: float | None = None
    selector_score: float | None = None
    selector_burden: float | None = None
    selector_geometry_mode: str = "reduced"
    controller_snapshot: dict[str, Any] | None = None
    phase1_shortlisted: bool = False
    phase2_shortlisted: bool = False
    phase3_shortlisted: bool = False


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
    proxy_baseline: dict[str, float] | None = None


@dataclass(frozen=True)
class SimpleScoreConfig:
    lambda_F: float = 1.0
    lambda_compile: float = 0.05
    lambda_measure: float = 0.02
    lambda_leak: float = 0.0
    z_alpha: float = 0.0
    wD: float = 0.0
    wG: float = 0.0
    wC: float = 0.0
    wc: float = 0.0
    depth_ref: float = 1.0
    group_ref: float = 1.0
    shot_ref: float = 1.0
    family_ref: float = 1.0
    lifetime_cost_mode: str = "off"
    score_version: str = "simple_v1"


@dataclass(frozen=True)
class FullScoreConfig:
    z_alpha: float = 0.0
    lambda_F: float = 1.0
    lambda_H: float = 1e-6
    rho: float = 0.25
    eta_L: float = 0.0
    gamma_N: float = 1.0
    wD: float = 0.2
    wG: float = 0.15
    wC: float = 0.15
    wP: float = 0.1
    wc: float = 0.1
    depth_ref: float = 1.0
    group_ref: float = 1.0
    shot_ref: float = 1.0
    optdim_ref: float = 1.0
    reuse_ref: float = 1.0
    family_ref: float = 1.0
    novelty_eps: float = 1e-6
    cheap_score_eps: float = 1e-12
    shortlist_fraction: float = 0.2
    shortlist_size: int = 12
    phase2_frontier_ratio: float = 0.9
    phase3_frontier_ratio: float = 0.9
    leakage_cap: float = 1e6
    lifetime_cost_mode: str = "off"
    remaining_evaluations_proxy_mode: str = "none"
    lifetime_weight: float = 0.05
    metric_floor: float = 1e-12
    reduced_metric_collapse_rel_tol: float = 1e-8
    ridge_growth_factor: float = 10.0
    ridge_max_steps: int = 12
    phase3_selector_geometry_mode: str = "reduced"
    score_version: str = "full_v2"


@dataclass(frozen=True)
class StageControllerConfig:
    plateau_patience: int = 2
    weak_drop_threshold: float = 1e-9
    probe_margin_ratio: float = 1.0
    max_probe_positions: int = 6
    append_admit_threshold: float = 0.05
    family_repeat_patience: int = 2
    tau_phase1_min: float = 0.0
    tau_phase1_max: float = 0.0
    tau_phase2_min: float = 0.0
    tau_phase2_max: float = 0.0
    tau_phase3_min: float = 0.0
    tau_phase3_max: float = 0.0
    cap_phase1_min: int = 1
    cap_phase1_max: int = 12
    cap_phase2_min: int = 1
    cap_phase2_max: int = 12
    cap_phase3_min: int = 1
    cap_phase3_max: int = 12
    shot_min: int = 1
    shot_max: int = 1
    runway_power_early: float = 1.0
    runway_power_late: float = 1.0


@dataclass(frozen=True)
class _ScaffoldDerivativeContext:
    psi_state: np.ndarray
    hpsi_state: np.ndarray
    refit_window_indices: tuple[int, ...]
    dpsi_window: tuple[np.ndarray, ...]
    tangents_window: tuple[np.ndarray, ...]
    Q_window: np.ndarray
    H_window_hessian: np.ndarray


def _replace_feature(feat: CandidateFeatures, **updates: Any) -> CandidateFeatures:
    return replace(feat, **updates)


def allowed_positions(
    *,
    n_params: int,
    append_position: int,
    active_window_indices: Iterable[int],
    max_positions: int,
) -> list[int]:
    positions = [int(append_position)]
    if int(n_params) <= 0:
        return [0]
    positions.append(0)
    for idx in active_window_indices:
        positions.append(int(idx))
    out: list[int] = []
    for p in positions:
        p_clamped = max(0, min(int(append_position), int(p)))
        if p_clamped not in out:
            out.append(p_clamped)
        if len(out) >= int(max_positions):
            break
    return out


def detect_trough(
    *,
    append_score: float,
    best_non_append_score: float,
    best_non_append_g_lcb: float,
    margin_ratio: float,
    append_admit_threshold: float,
) -> bool:
    if float(best_non_append_g_lcb) <= 0.0:
        return False
    if float(best_non_append_score) >= float(margin_ratio) * float(append_score):
        return True
    return (
        float(append_score) < float(append_admit_threshold)
        and float(best_non_append_score) >= float(append_admit_threshold)
    )


def should_probe_positions(
    *,
    stage_name: str,
    drop_plateau_hits: int,
    max_grad: float,
    eps_grad: float,
    append_score: float,
    finite_angle_flat: bool,
    repeated_family_flat: bool,
    cfg: StageControllerConfig,
) -> tuple[bool, str]:
    del append_score
    if str(stage_name) == "residual":
        return False, "residual_stage"
    if int(drop_plateau_hits) >= int(cfg.plateau_patience):
        return True, "drop_plateau"
    if float(max_grad) < float(eps_grad) and bool(finite_angle_flat):
        return True, "eps_grad_flat"
    if bool(repeated_family_flat):
        return True, "family_repeat_flat"
    return False, "default_append_only"


class StageController:
    def __init__(self, cfg: StageControllerConfig) -> None:
        self.cfg = cfg
        self._stage = "core"
        self._admission_deltas: list[float] = []
        self._last_snapshot: PhaseControllerSnapshot | None = None

    @property
    def stage_name(self) -> str:
        return str(self._stage)

    def start_with_seed(self) -> None:
        self._stage = "seed"

    def clone(self) -> "StageController":
        cloned = StageController(self.cfg)
        cloned._stage = str(self._stage)
        cloned._admission_deltas = [float(x) for x in self._admission_deltas]
        cloned._last_snapshot = self._last_snapshot
        return cloned

    def _runway_ratio(self, *, depth_local: int, max_depth: int) -> float:
        depth_now = max(0, int(depth_local))
        depth_cap = max(depth_now, int(max_depth))
        depth_left = max(0, depth_cap - depth_now)
        if depth_cap <= 0:
            return 0.0
        return float(max(0.0, min(1.0, depth_left / float(depth_cap))))

    def pre_step_snapshot(self, *, depth_local: int, max_depth: int) -> PhaseControllerSnapshot:
        depth_now = max(0, int(depth_local))
        depth_cap = max(depth_now, int(max_depth))
        depth_left = max(0, depth_cap - depth_now)
        runway_ratio = self._runway_ratio(depth_local=depth_now, max_depth=depth_cap)
        early = float(runway_ratio ** float(max(self.cfg.runway_power_early, 1e-12)))
        late = float((1.0 - runway_ratio) ** float(max(self.cfg.runway_power_late, 1e-12)))
        snapshot = PhaseControllerSnapshot(
            step_index=int(len(self._admission_deltas)),
            depth_local=int(depth_now),
            depth_left=int(depth_left),
            runway_ratio=float(runway_ratio),
            early_coordinate=float(early),
            late_coordinate=float(late),
            frontier_ratio=1.0,
            phase_thresholds={
                "phase1": float(self.cfg.tau_phase1_min + (self.cfg.tau_phase1_max - self.cfg.tau_phase1_min) * early),
                "phase2": float(self.cfg.tau_phase2_min + (self.cfg.tau_phase2_max - self.cfg.tau_phase2_min) * early),
                "phase3": float(self.cfg.tau_phase3_min + (self.cfg.tau_phase3_max - self.cfg.tau_phase3_min) * early),
            },
            phase_caps={
                "phase1": int(round(self.cfg.cap_phase1_min + (self.cfg.cap_phase1_max - self.cfg.cap_phase1_min) * late)),
                "phase2": int(round(self.cfg.cap_phase2_min + (self.cfg.cap_phase2_max - self.cfg.cap_phase2_min) * late)),
                "phase3": int(round(self.cfg.cap_phase3_min + (self.cfg.cap_phase3_max - self.cfg.cap_phase3_min) * late)),
            },
            phase_shots={"phase1": 1, "phase2": 1, "phase3": 1},
            phase_uncertainty={"phase2": 0.0, "phase3": 0.0},
        )
        self._last_snapshot = snapshot
        return snapshot

    def finalize_step_snapshot(
        self,
        *,
        pre_snapshot: PhaseControllerSnapshot,
        phase1_raw_scores: Iterable[float],
        u_sigma_phase2: float | None = None,
        u_sigma_phase3: float | None = None,
    ) -> PhaseControllerSnapshot:
        scores = sorted([float(x) for x in phase1_raw_scores if math.isfinite(float(x))], reverse=True)
        top = float(scores[0]) if scores else 0.0
        second = float(scores[1]) if len(scores) > 1 else 0.0
        frontier_ratio = float((second + 1e-12) / (top + 1e-12)) if (scores or top == 0.0) else 1.0
        snapshot = replace(
            pre_snapshot,
            frontier_ratio=float(max(0.0, min(1.0, frontier_ratio))),
            phase_uncertainty={
                "phase2": float(max(0.0, u_sigma_phase2 or 0.0)),
                "phase3": float(max(0.0, u_sigma_phase3 or 0.0)),
            },
        )
        self._last_snapshot = snapshot
        return snapshot

    def record_admission(self, *, selector_step: int, energy_before: float, energy_after_refit: float) -> None:
        del selector_step
        self._admission_deltas.append(float(energy_before) - float(energy_after_refit))

    def resolve_stage_transition(
        self,
        *,
        drop_plateau_hits: int,
        trough_detected: bool,
        residual_opened: bool,
    ) -> tuple[str, str]:
        if self._stage == "seed":
            self._stage = "core"
            return self._stage, "seed_complete"
        if self._stage == "core":
            if int(drop_plateau_hits) >= int(self.cfg.plateau_patience) and (not bool(trough_detected)):
                self._stage = "residual"
                return self._stage, "plateau_without_trough"
            return self._stage, "stay_core"
        if self._stage == "residual":
            if bool(residual_opened):
                return self._stage, "stay_residual"
            return self._stage, "residual_closed"
        return self._stage, "unknown_stage"


def _pauli_weight_exyz(label_exyz: str) -> int:
    return int(sum(1 for ch in str(label_exyz) if ch in {"x", "y", "z"}))


def _pauli_labels_from_term(term: Any) -> list[str]:
    if term is None or not hasattr(term, "polynomial"):
        return []
    return [str(poly_term.pw2strng()) for poly_term in term.polynomial.return_polynomial()]


def _measurement_basis_key_covers(required_key: str, seen_key: str) -> bool:
    req = str(required_key)
    seen = str(seen_key)
    if len(req) != len(seen):
        return False
    return all((r == "e") or (r == s) for r, s in zip(req, seen))


def _measurement_basis_key_merge(lhs_key: str, rhs_key: str) -> str | None:
    lhs = str(lhs_key)
    rhs = str(rhs_key)
    if len(lhs) != len(rhs):
        return None
    merged: list[str] = []
    for lhs_ch, rhs_ch in zip(lhs, rhs):
        if lhs_ch == "e":
            merged.append(rhs_ch)
            continue
        if rhs_ch in {"e", lhs_ch}:
            merged.append(lhs_ch)
            continue
        return None
    return "".join(merged)


def _compress_measurement_group_keys(group_keys: Iterable[str]) -> list[str]:
    ordered = sorted(
        {str(key) for key in group_keys if str(key) != ""},
        key=lambda key: (-_pauli_weight_exyz(str(key)), str(key)),
    )
    kept: list[str] = []
    for key in ordered:
        if any(_measurement_basis_key_covers(str(key), existing) for existing in kept):
            continue
        kept = [existing for existing in kept if not _measurement_basis_key_covers(existing, str(key))]
        kept.append(str(key))
    return kept


def _measurement_group_keys_from_labels(labels: Sequence[str]) -> list[str]:
    active_labels = sorted(
        {str(lbl) for lbl in labels if _pauli_weight_exyz(str(lbl)) > 0},
        key=lambda lbl: (-_pauli_weight_exyz(lbl), lbl),
    )
    groups: list[str] = []
    for label in active_labels:
        best_idx: int | None = None
        best_key: str | None = None
        best_delta: tuple[int, int] | None = None
        for idx, group_key in enumerate(groups):
            merged = _measurement_basis_key_merge(str(group_key), str(label))
            if merged is None:
                continue
            delta = (_pauli_weight_exyz(merged) - _pauli_weight_exyz(str(group_key)), idx)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_idx = int(idx)
                best_key = str(merged)
        if best_idx is None or best_key is None:
            groups.append(str(label))
        else:
            groups[best_idx] = str(best_key)
    return _compress_measurement_group_keys(groups)


def measurement_group_keys_for_term(term: Any) -> list[str]:
    return _measurement_group_keys_from_labels(_pauli_labels_from_term(term))


class Phase1CompileCostOracle:
    @staticmethod
    def _pauli_xy_count(label_exyz: str) -> int:
        return int(sum(1 for ch in str(label_exyz) if ch in {"x", "y"}))

    @classmethod
    def _cx_proxy_term(cls, label_exyz: str) -> int:
        return int(2 * max(_pauli_weight_exyz(label_exyz) - 1, 0))

    @classmethod
    def _sq_proxy_term(cls, label_exyz: str) -> int:
        weight = _pauli_weight_exyz(label_exyz)
        if weight <= 0:
            return 0
        return int(2 * cls._pauli_xy_count(label_exyz) + 1)

    def estimate(
        self,
        *,
        candidate_term_count: int,
        position_id: int,
        append_position: int,
        refit_active_count: int,
        candidate_term: Any | None = None,
    ) -> CompileCostEstimate:
        candidate_labels = _pauli_labels_from_term(candidate_term)
        active_labels = [str(lbl) for lbl in candidate_labels if _pauli_weight_exyz(str(lbl)) > 0]
        if active_labels:
            new_pauli_actions = float(len(active_labels))
            new_rotation_steps = float(len(active_labels))
            cx_proxy_total = float(sum(self._cx_proxy_term(lbl) for lbl in active_labels))
            sq_proxy_total = float(sum(self._sq_proxy_term(lbl) for lbl in active_labels))
            gate_proxy_total = float(cx_proxy_total + 0.5 * sq_proxy_total)
            max_pauli_weight = float(max(_pauli_weight_exyz(lbl) for lbl in active_labels))
        else:
            fallback_count = float(max(1, int(candidate_term_count)))
            new_pauli_actions = fallback_count
            new_rotation_steps = fallback_count
            cx_proxy_total = fallback_count
            sq_proxy_total = fallback_count
            gate_proxy_total = fallback_count
            max_pauli_weight = 0.0
        position_shift_span = float(abs(int(append_position) - int(position_id)))
        refit_active = float(max(0, int(refit_active_count)))
        total = float(gate_proxy_total + position_shift_span + refit_active)
        return CompileCostEstimate(
            new_pauli_actions=new_pauli_actions,
            new_rotation_steps=new_rotation_steps,
            position_shift_span=position_shift_span,
            refit_active_count=refit_active,
            proxy_total=total,
            cx_proxy_total=cx_proxy_total,
            sq_proxy_total=sq_proxy_total,
            gate_proxy_total=gate_proxy_total,
            max_pauli_weight=max_pauli_weight,
        )


class MeasurementCacheAudit:
    def __init__(self, nominal_shots_per_group: int = 1) -> None:
        self._seen_groups: set[str] = set()
        self._nominal_shots = int(max(1, nominal_shots_per_group))

    def estimate(self, group_keys: Iterable[str]) -> MeasurementCacheStats:
        unique_keys = list(_compress_measurement_group_keys(group_keys))
        groups_total = int(len(unique_keys))
        groups_reused = 0
        seen_keys = list(self._seen_groups)
        for key in unique_keys:
            if any(_measurement_basis_key_covers(str(key), str(seen)) for seen in seen_keys):
                groups_reused += 1
        groups_new = int(groups_total - groups_reused)
        shots_reused = float(groups_reused * self._nominal_shots)
        shots_new = float(groups_new * self._nominal_shots)
        reuse_count_cost = float(groups_new)
        return MeasurementCacheStats(
            groups_total=groups_total,
            groups_reused=int(groups_reused),
            groups_new=int(groups_new),
            shots_reused=shots_reused,
            shots_new=shots_new,
            reuse_count_cost=reuse_count_cost,
        )

    def commit(self, group_keys: Iterable[str]) -> None:
        for key in _compress_measurement_group_keys(group_keys):
            key_s = str(key)
            if key_s == "":
                continue
            if any(_measurement_basis_key_covers(key_s, seen) for seen in self._seen_groups):
                continue
            covered = {seen for seen in self._seen_groups if _measurement_basis_key_covers(seen, key_s)}
            if covered:
                self._seen_groups -= covered
            self._seen_groups.add(key_s)


def simple_v1_score(feat: CandidateFeatures, cfg: SimpleScoreConfig) -> float:
    if not bool(feat.stage_gate_open):
        return float("-inf")
    if not bool(feat.leakage_gate_open):
        return float("-inf")
    if not bool(feat.compile_gate_open):
        return float("-inf")
    compile_proxy = float(feat.compile_cost_total)
    groups_new = float(feat.measurement_cache_stats.get("groups_new", 0.0))
    shots_new = float(feat.measurement_cache_stats.get("shots_new", 0.0))
    reuse_count_cost = float(feat.measurement_cache_stats.get("reuse_count_cost", 0.0))
    leakage_penalty = float(feat.leakage_penalty)
    burden_total = (
        float(cfg.lambda_compile) * compile_proxy
        + float(cfg.lambda_measure) * (groups_new + shots_new + reuse_count_cost)
        + float(cfg.lambda_leak) * leakage_penalty
    )
    return float(float(feat.g_abs) / float(1.0 + max(0.0, burden_total)))


def normalize(value: float, ref: float) -> float:
    denom = float(ref)
    if not math.isfinite(denom) or denom <= 0.0:
        return float(max(0.0, value))
    return float(max(0.0, value) / denom)


def trust_region_drop(g_lcb: float, h_eff: float, F: float, rho: float) -> float:
    if float(g_lcb) <= 0.0 or float(F) <= 0.0:
        return 0.0
    h_eff_pos = float(max(0.0, h_eff))
    alpha_max = float(rho) / float(math.sqrt(float(F)))
    if h_eff_pos > 0.0:
        alpha_newton = float(g_lcb) / h_eff_pos
        if alpha_newton <= alpha_max:
            return float(0.5 * float(g_lcb) * float(g_lcb) / h_eff_pos)
    alpha = float(alpha_max)
    return float(float(g_lcb) * alpha - 0.5 * h_eff_pos * alpha * alpha)


def remaining_evaluations_proxy(*, current_depth: int | None, max_depth: int | None, mode: str) -> float:
    mode_key = str(mode).strip().lower()
    if mode_key == "none":
        return 0.0
    depth_now = 0 if current_depth is None else int(max(0, current_depth))
    depth_cap = depth_now if max_depth is None else int(max(depth_now, max_depth))
    if mode_key == "remaining_depth":
        return float(max(1, depth_cap - depth_now + 1))
    raise ValueError("remaining_evaluations_proxy_mode must be 'none' or 'remaining_depth'")


def family_repeat_cost_from_history(*, history_rows: Sequence[Mapping[str, Any]], candidate_family: str) -> float:
    fam = str(candidate_family).strip()
    if fam == "":
        return 0.0
    tail = [row for row in history_rows if isinstance(row, Mapping) and row.get("candidate_family") is not None]
    if not tail:
        return 0.0
    if str(tail[-1].get("candidate_family", "")).strip() != fam:
        return 0.0
    streak = 0
    for row in reversed(tail):
        if str(row.get("candidate_family", "")).strip() != fam:
            break
        streak += 1
    return float(streak)


def lifetime_weight_components(feat: CandidateFeatures, cfg: FullScoreConfig) -> dict[str, float]:
    if str(cfg.lifetime_cost_mode).strip().lower() == "off":
        return {
            "remaining_evaluations_proxy": float(feat.remaining_evaluations_proxy),
            "compiled": 0.0,
            "measurement": 0.0,
            "optimizer_dim": 0.0,
            "total": 0.0,
        }
    rem = float(max(0.0, feat.remaining_evaluations_proxy))
    compiled = rem * normalize(float(feat.depth_cost), float(cfg.depth_ref))
    measurement = rem * (
        normalize(float(feat.new_group_cost), float(cfg.group_ref))
        + normalize(float(feat.new_shot_cost), float(cfg.shot_ref))
        + normalize(float(feat.reuse_count_cost), float(cfg.reuse_ref))
    )
    optimizer_dim = rem * normalize(float(feat.opt_dim_cost), float(cfg.optdim_ref))
    total = compiled + measurement + optimizer_dim
    return {
        "remaining_evaluations_proxy": float(rem),
        "compiled": float(compiled),
        "measurement": float(measurement),
        "optimizer_dim": float(optimizer_dim),
        "total": float(total),
    }


def _cheap_burden_total(feat: CandidateFeatures, cfg: FullScoreConfig) -> float:
    K = (
        1.0
        + float(cfg.wD) * normalize(float(feat.depth_cost), float(cfg.depth_ref))
        + float(cfg.wG) * normalize(float(feat.new_group_cost), float(cfg.group_ref))
        + float(cfg.wC) * normalize(float(feat.new_shot_cost), float(cfg.shot_ref))
        + float(cfg.wP) * normalize(float(feat.opt_dim_cost), float(cfg.optdim_ref))
        + float(cfg.wc) * normalize(float(feat.reuse_count_cost), float(cfg.reuse_ref))
    )
    lifetime_components = lifetime_weight_components(feat, cfg)
    K = float(K + float(cfg.lifetime_weight) * float(lifetime_components.get("total", 0.0)))
    return float(K)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _phase_confidence_factor(g_abs: float, sigma_hat: float, *, z_alpha: float, eps: float = 1e-12) -> float:
    denom = float(max(abs(float(g_abs)), float(eps)))
    return _clip01(1.0 - float(z_alpha) * float(max(0.0, sigma_hat)) / denom)


def phase2_raw_geometry_score(
    feat: CandidateFeatures,
    *,
    F_raw: float,
    h_raw: float,
    q_window: Sequence[float],
    Q_window: np.ndarray,
    cfg: FullScoreConfig,
) -> dict[str, float]:
    F_safe = float(max(float(F_raw), float(cfg.metric_floor)))
    q_vec = np.asarray(q_window, dtype=float).reshape(-1)
    Q_mat = np.asarray(Q_window, dtype=float)
    overlap_max = 0.0
    if q_vec.size > 0 and Q_mat.size > 0:
        diag = np.diag(Q_mat)
        denom_base = math.sqrt(F_safe)
        for idx, q_val in enumerate(q_vec.tolist()):
            diag_val = float(diag[idx]) if idx < int(diag.size) else 0.0
            denom = denom_base * math.sqrt(max(float(diag_val), float(cfg.metric_floor)))
            if denom <= 0.0:
                continue
            overlap_max = max(overlap_max, abs(float(q_val)) / denom)
    raw_novelty = _clip01(1.0 - overlap_max)
    confidence = _phase_confidence_factor(float(feat.g_abs), float(feat.sigma_hat), z_alpha=float(cfg.z_alpha))
    burden_total = float(_cheap_burden_total(feat, cfg))
    trust_gain = float(trust_region_drop(float(feat.g_lcb), float(max(0.0, h_raw)), F_safe, float(cfg.rho)))
    score = float(trust_gain * confidence * raw_novelty / max(burden_total, float(cfg.cheap_score_eps)))
    return {
        "confidence_factor": float(confidence),
        "phase2_raw_overlap_max": float(overlap_max),
        "phase2_raw_novelty": float(raw_novelty),
        "phase2_raw_trust_gain": float(trust_gain),
        "phase2_burden_total": float(burden_total),
        "phase2_raw_score": float(score),
    }


def phase_shortlist_records(
    records: Sequence[Mapping[str, Any]],
    *,
    score_key: str,
    threshold: float,
    cap: int,
    frontier_ratio: float,
    tie_break_score_key: str | None = None,
    shortlist_flag: str | None = None,
    score_eps: float = 1e-12,
) -> list[dict[str, Any]]:
    def _record_score(rec: Mapping[str, Any], key: str | None, default: float = float("-inf")) -> float:
        if key is None:
            return 0.0
        raw = rec.get(key, default)
        if raw is None:
            return float(default)
        return float(raw)

    ranked = sorted(
        [dict(rec) for rec in records if float(_record_score(rec, score_key)) >= float(threshold)],
        key=lambda rec: (
            -_record_score(rec, score_key),
            -_record_score(rec, tie_break_score_key),
            int(rec.get("candidate_pool_index", -1)),
            int(rec.get("position_id", -1)),
        ),
    )
    if not ranked:
        return []
    cap_eff = int(max(1, min(int(cap), len(ranked))))
    shortlist_size = int(cap_eff)
    frontier_cut = float(max(0.0, min(1.0, frontier_ratio)))
    if cap_eff > 1 and frontier_cut > 0.0:
        for idx in range(cap_eff - 1):
            s_cur = float(_record_score(ranked[idx], score_key))
            s_next = float(_record_score(ranked[idx + 1], score_key))
            ratio = float((s_next + float(score_eps)) / (s_cur + float(score_eps)))
            if ratio <= frontier_cut:
                shortlist_size = int(idx + 1)
                break
    out: list[dict[str, Any]] = []
    for idx, rec in enumerate(ranked[:shortlist_size], start=1):
        updated = dict(rec)
        feat = updated.get("feature")
        if isinstance(feat, CandidateFeatures):
            repl: dict[str, Any] = {"shortlist_rank": int(idx), "shortlist_size": int(shortlist_size)}
            if shortlist_flag is not None and hasattr(feat, str(shortlist_flag)):
                repl[str(shortlist_flag)] = True
            updated["feature"] = _replace_feature(feat, **repl)
        out.append(updated)
    return out


def shortlist_records(
    records: Sequence[Mapping[str, Any]],
    *,
    cfg: FullScoreConfig,
    score_key: str = "simple_score",
    tie_break_score_key: str | None = "simple_score",
) -> list[dict[str, Any]]:
    def _record_score(rec: Mapping[str, Any], key: str | None, default: float = float("-inf")) -> float:
        if key is None:
            return 0.0
        raw = rec.get(key, default)
        if raw is None:
            return float(default)
        return float(raw)

    ranked = sorted(
        [dict(rec) for rec in records],
        key=lambda rec: (
            -_record_score(rec, score_key),
            -_record_score(rec, tie_break_score_key),
            int(rec.get("candidate_pool_index", -1)),
            int(rec.get("position_id", -1)),
        ),
    )
    if not ranked:
        return []
    total = int(len(ranked))
    target = int(max(1, min(total, cfg.shortlist_size, math.ceil(float(cfg.shortlist_fraction) * total))))
    out: list[dict[str, Any]] = []
    for idx, rec in enumerate(ranked[:target], start=1):
        updated = dict(rec)
        feat = updated.get("feature")
        if isinstance(feat, CandidateFeatures):
            updated["feature"] = _replace_feature(feat, shortlist_rank=int(idx), shortlist_size=int(target))
        out.append(updated)
    return out


def build_candidate_features(
    *,
    stage_name: str,
    candidate_label: str,
    candidate_family: str,
    candidate_pool_index: int,
    position_id: int,
    append_position: int,
    positions_considered: list[int],
    gradient_signed: float,
    metric_proxy: float,
    sigma_hat: float,
    refit_window_indices: list[int],
    compile_cost: CompileCostEstimate,
    measurement_stats: MeasurementCacheStats,
    leakage_penalty: float,
    stage_gate_open: bool,
    leakage_gate_open: bool,
    trough_probe_triggered: bool,
    trough_detected: bool,
    family_repeat_cost: float = 0.0,
    cfg: SimpleScoreConfig = SimpleScoreConfig(),
    cheap_score_cfg: FullScoreConfig | None = None,
    generator_metadata: Mapping[str, Any] | None = None,
    current_depth: int | None = None,
    max_depth: int | None = None,
    lifetime_cost_mode: str = "off",
    remaining_evaluations_proxy_mode: str = "none",
) -> CandidateFeatures:
    g_abs = float(abs(float(gradient_signed)))
    g_lcb = max(g_abs - float(cfg.z_alpha) * float(max(0.0, sigma_hat)), 0.0)
    remaining_eval_proxy = remaining_evaluations_proxy(
        current_depth=current_depth,
        max_depth=max_depth,
        mode=str(remaining_evaluations_proxy_mode),
    )
    proxy_cost = {
        "new_pauli_actions": float(compile_cost.new_pauli_actions),
        "new_rotation_steps": float(compile_cost.new_rotation_steps),
        "position_shift_span": float(compile_cost.position_shift_span),
        "refit_active_count": float(compile_cost.refit_active_count),
        "cx_proxy_total": float(compile_cost.cx_proxy_total),
        "sq_proxy_total": float(compile_cost.sq_proxy_total),
        "gate_proxy_total": float(compile_cost.gate_proxy_total),
        "max_pauli_weight": float(compile_cost.max_pauli_weight),
        "proxy_total": float(compile_cost.proxy_total),
    }
    compile_cost_total = (
        float(compile_cost.penalty_total)
        if compile_cost.penalty_total is not None
        else float(compile_cost.proxy_total)
    )
    depth_cost_value = (
        float(compile_cost.depth_surrogate)
        if compile_cost.depth_surrogate is not None
        else float(
            (float(proxy_cost.get("gate_proxy_total", 0.0)) if float(proxy_cost.get("gate_proxy_total", 0.0)) > 0.0 else float(proxy_cost.get("new_rotation_steps", 0.0)))
            + float(proxy_cost.get("position_shift_span", 0.0))
        )
    )
    feat = CandidateFeatures(
        stage_name=str(stage_name),
        candidate_label=str(candidate_label),
        candidate_family=str(candidate_family),
        candidate_pool_index=int(candidate_pool_index),
        position_id=int(position_id),
        append_position=int(append_position),
        positions_considered=[int(x) for x in positions_considered],
        g_signed=float(gradient_signed),
        g_abs=float(g_abs),
        g_lcb=float(g_lcb),
        sigma_hat=float(max(0.0, sigma_hat)),
        F_metric=float(max(0.0, metric_proxy)),
        metric_proxy=float(max(0.0, metric_proxy)),
        novelty=None,
        curvature_mode="lambda_F_metric_proxy_only",
        novelty_mode="none",
        refit_window_indices=[int(i) for i in refit_window_indices],
        compiled_position_cost_proxy={str(k): float(v) for k, v in proxy_cost.items()},
        measurement_cache_stats={
            "groups_total": float(measurement_stats.groups_total),
            "groups_reused": float(measurement_stats.groups_reused),
            "groups_new": float(measurement_stats.groups_new),
            "shots_reused": float(measurement_stats.shots_reused),
            "shots_new": float(measurement_stats.shots_new),
            "reuse_count_cost": float(measurement_stats.reuse_count_cost),
        },
        leakage_penalty=float(max(0.0, leakage_penalty)),
        stage_gate_open=bool(stage_gate_open),
        leakage_gate_open=bool(leakage_gate_open),
        trough_probe_triggered=bool(trough_probe_triggered),
        trough_detected=bool(trough_detected),
        simple_score=None,
        score_version=str(cfg.score_version),
        cheap_score=None,
        cheap_score_version=str(cfg.score_version),
        cheap_metric_proxy=float(max(0.0, metric_proxy)),
        depth_cost=float(depth_cost_value),
        new_group_cost=float(measurement_stats.groups_new),
        new_shot_cost=float(measurement_stats.shots_new),
        opt_dim_cost=float(len(refit_window_indices)),
        reuse_count_cost=float(measurement_stats.reuse_count_cost),
        family_repeat_cost=float(family_repeat_cost),
        generator_id=(
            str(generator_metadata.get("generator_id"))
            if isinstance(generator_metadata, Mapping) and generator_metadata.get("generator_id") is not None
            else None
        ),
        generator_metadata=(dict(generator_metadata) if isinstance(generator_metadata, Mapping) else None),
        remaining_evaluations_proxy=float(remaining_eval_proxy),
        remaining_evaluations_proxy_mode=str(remaining_evaluations_proxy_mode),
        lifetime_cost_mode=str(lifetime_cost_mode),
        lifetime_weight_components={"remaining_evaluations_proxy": float(remaining_eval_proxy)},
        compile_cost_source=str(compile_cost.source_mode),
        compile_cost_total=float(compile_cost_total),
        compile_gate_open=bool(compile_cost.compile_gate_open),
        compile_failure_reason=(None if compile_cost.failure_reason is None else str(compile_cost.failure_reason)),
        phase_score_components={},
        phase_cost_components={},
    )
    score = simple_v1_score(feat, cfg)
    phase_cost_components = {
        "compile_proxy": float(compile_cost_total),
        "measurement_groups_new": float(measurement_stats.groups_new),
        "measurement_shots_new": float(measurement_stats.shots_new),
        "measurement_reuse_cost": float(measurement_stats.reuse_count_cost),
        "leakage_penalty": float(max(0.0, leakage_penalty)),
    }
    feat = _replace_feature(
        feat,
        simple_score=float(score),
        cheap_score=float(score),
        cheap_score_version=str(cfg.score_version),
        phase_score_components={"phase1_gradient_abs": float(g_abs), "phase1_score": float(score)},
        phase_cost_components=dict(phase_cost_components),
    )
    if cheap_score_cfg is not None:
        burden = _cheap_burden_total(feat, cheap_score_cfg)
        benefit = float(
            float(feat.g_lcb) * float(feat.g_lcb)
            / (2.0 * float(max(float(cheap_score_cfg.lambda_F), float(cheap_score_cfg.cheap_score_eps))) * float(max(feat.metric_proxy, cheap_score_cfg.metric_floor)))
        ) if float(feat.g_lcb) > 0.0 and float(max(feat.metric_proxy, cheap_score_cfg.metric_floor)) > 0.0 else 0.0
        feat = _replace_feature(
            feat,
            cheap_benefit_proxy=float(benefit),
            cheap_burden_total=float(burden),
            cheap_score=float(benefit / max(float(burden), float(cheap_score_cfg.cheap_score_eps))),
        )
    return feat


def _compiled_for_label(
    *,
    label: str,
    polynomial: Any,
    compiled_cache: dict[str, CompiledPolynomialAction] | None,
    pauli_action_cache: dict[str, Any] | None,
) -> CompiledPolynomialAction:
    cache = compiled_cache if compiled_cache is not None else {}
    key = str(label)
    compiled = cache.get(key)
    if compiled is None:
        compiled = compile_polynomial_action(polynomial, tol=1e-12, pauli_action_cache=pauli_action_cache)
        cache[key] = compiled
    return compiled


def _tangent_data(
    *,
    psi_state: np.ndarray,
    label: str,
    polynomial: Any,
    compiled_cache: dict[str, CompiledPolynomialAction] | None,
    pauli_action_cache: dict[str, Any] | None,
) -> tuple[np.ndarray, float]:
    psi = np.asarray(psi_state, dtype=complex).reshape(-1)
    compiled = _compiled_for_label(
        label=str(label),
        polynomial=polynomial,
        compiled_cache=compiled_cache,
        pauli_action_cache=pauli_action_cache,
    )
    apsi = apply_compiled_polynomial(psi, compiled)
    mean = complex(np.vdot(psi, apsi))
    centered = np.asarray(apsi - mean * psi, dtype=complex)
    F = float(max(0.0, np.real(np.vdot(centered, centered))))
    return centered, F


def raw_f_metric_from_state(
    *,
    psi_state: np.ndarray,
    candidate_label: str,
    candidate_term: Any,
    compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
) -> float:
    _tangent, F_metric = _tangent_data(
        psi_state=psi_state,
        label=str(candidate_label),
        polynomial=candidate_term.polynomial,
        compiled_cache=compiled_cache,
        pauli_action_cache=pauli_action_cache,
    )
    return float(F_metric)


def _tangent_overlap_matrix(tangents: Sequence[np.ndarray]) -> np.ndarray:
    n = int(len(tangents))
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            val = float(np.real(np.vdot(tangents[i], tangents[j])))
            out[i, j] = val
            out[j, i] = val
    return out


def _executor_for_terms(
    terms: Sequence[Any],
    *,
    pauli_action_cache: dict[str, Any] | None,
    parameterization_mode: str,
) -> CompiledAnsatzExecutor:
    return CompiledAnsatzExecutor(
        list(terms),
        coefficient_tolerance=1e-12,
        ignore_identity=True,
        sort_terms=True,
        pauli_action_cache=pauli_action_cache,
        parameterization_mode=str(parameterization_mode),
    )


def _rotation_triplet(vec: np.ndarray, step: Any, theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vec_arr = np.asarray(vec, dtype=complex).reshape(-1)
    coeff = float(step.coeff_real)
    pvec = apply_compiled_pauli(vec_arr, step.action)
    phi = float(theta) * coeff
    c = math.cos(phi)
    s = math.sin(phi)
    u_vec = c * vec_arr - 1j * s * pvec
    d_vec = -coeff * s * vec_arr - 1j * coeff * c * pvec
    s_vec = -(coeff * coeff) * u_vec
    return np.asarray(u_vec, dtype=complex), np.asarray(d_vec, dtype=complex), np.asarray(s_vec, dtype=complex)


def _horizontal_tangent(psi_state: np.ndarray, dpsi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi_state, dtype=complex).reshape(-1)
    dpsi_vec = np.asarray(dpsi, dtype=complex).reshape(-1)
    overlap = complex(np.vdot(psi, dpsi_vec))
    return np.asarray(dpsi_vec - overlap * psi, dtype=complex)


def _energy_hessian_entry(
    *,
    dpsi_left: np.ndarray,
    dpsi_right: np.ndarray,
    d2psi: np.ndarray,
    hpsi_state: np.ndarray,
    hdpsi_right: np.ndarray,
) -> float:
    return float(
        2.0
        * np.real(
            np.vdot(np.asarray(d2psi, dtype=complex), np.asarray(hpsi_state, dtype=complex))
            + np.vdot(np.asarray(dpsi_left, dtype=complex), np.asarray(hdpsi_right, dtype=complex))
        )
    )


def _propagate_executor_derivatives(
    *,
    executor: CompiledAnsatzExecutor,
    theta: np.ndarray,
    psi_ref: np.ndarray,
    active_indices: Sequence[int],
) -> tuple[np.ndarray, list[np.ndarray], list[list[np.ndarray]]]:
    theta_vec = np.asarray(theta, dtype=float).reshape(-1)
    active = [int(i) for i in active_indices]
    psi = np.asarray(psi_ref, dtype=complex).reshape(-1).copy()
    n_active = int(len(active))
    dpsi = [np.zeros_like(psi, dtype=complex) for _ in range(n_active)]
    d2psi = [[np.zeros_like(psi, dtype=complex) for _ in range(n_active)] for __ in range(n_active)]
    if n_active == 0:
        return executor.prepare_state(theta_vec, psi), dpsi, d2psi
    active_map = {int(global_idx): int(local_idx) for local_idx, global_idx in enumerate(active)}
    plans = list(getattr(executor, "_plans", []))
    if len(plans) != int(theta_vec.size):
        raise ValueError(f"theta length mismatch: got {theta_vec.size}, expected {len(plans)}.")
    for global_idx, plan in enumerate(plans):
        theta_k = float(theta_vec[global_idx])
        local = active_map.get(int(global_idx), None)
        for step in getattr(plan, "steps", ()):
            old_psi = psi
            old_dpsi = dpsi
            old_d2psi = d2psi
            psi_u, psi_d, psi_s = _rotation_triplet(old_psi, step, theta_k)
            psi = psi_u
            next_dpsi: list[np.ndarray] = []
            d_old: list[np.ndarray] = []
            for idx in range(n_active):
                vec_u, vec_d, _vec_s = _rotation_triplet(old_dpsi[idx], step, theta_k)
                next_dpsi.append(vec_u)
                d_old.append(vec_d)
            if local is not None:
                next_dpsi[int(local)] = np.asarray(next_dpsi[int(local)] + psi_d, dtype=complex)
            next_d2psi: list[list[np.ndarray]] = [
                [np.zeros_like(psi, dtype=complex) for _ in range(n_active)]
                for __ in range(n_active)
            ]
            for row in range(n_active):
                for col in range(n_active):
                    vec_u, _vec_d, _vec_s = _rotation_triplet(old_d2psi[row][col], step, theta_k)
                    updated = vec_u
                    if local is not None:
                        if row == int(local):
                            updated = np.asarray(updated + d_old[col], dtype=complex)
                        if col == int(local):
                            updated = np.asarray(updated + d_old[row], dtype=complex)
                        if row == int(local) and col == int(local):
                            updated = np.asarray(updated + psi_s, dtype=complex)
                    next_d2psi[row][col] = np.asarray(updated, dtype=complex)
            dpsi = next_dpsi
            d2psi = next_d2psi
    return np.asarray(psi, dtype=complex), dpsi, d2psi


def _propagate_append_candidate(
    *,
    candidate_term: Any,
    psi_state: np.ndarray,
    window_dpsi: Sequence[np.ndarray],
    pauli_action_cache: dict[str, Any] | None,
    parameterization_mode: str,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    cand_exec = _executor_for_terms(
        [candidate_term],
        pauli_action_cache=pauli_action_cache,
        parameterization_mode=str(parameterization_mode),
    )
    plan = list(getattr(cand_exec, "_plans", []))
    if not plan:
        zero = np.zeros_like(np.asarray(psi_state, dtype=complex).reshape(-1), dtype=complex)
        return zero, zero, [np.zeros_like(zero) for _ in window_dpsi]
    steps = list(getattr(plan[0], "steps", ()))
    psi = np.asarray(psi_state, dtype=complex).reshape(-1).copy()
    cand_dpsi = np.zeros_like(psi, dtype=complex)
    cand_d2psi = np.zeros_like(psi, dtype=complex)
    win_dpsi = [np.asarray(vec, dtype=complex).reshape(-1).copy() for vec in window_dpsi]
    cand_win_d2 = [np.zeros_like(psi) for _ in window_dpsi]
    for step in steps:
        old_psi = psi
        old_cand_dpsi = cand_dpsi
        old_cand_d2psi = cand_d2psi
        old_win_dpsi = win_dpsi
        old_cand_win_d2 = cand_win_d2
        psi_u, psi_d, psi_s = _rotation_triplet(old_psi, step, 0.0)
        cand_u, cand_d, _cand_s = _rotation_triplet(old_cand_dpsi, step, 0.0)
        cand2_u, _cand2_d, _cand2_s = _rotation_triplet(old_cand_d2psi, step, 0.0)
        psi = psi_u
        cand_dpsi = np.asarray(cand_u + psi_d, dtype=complex)
        cand_d2psi = np.asarray(cand2_u + cand_d + cand_d + psi_s, dtype=complex)
        next_win_dpsi: list[np.ndarray] = []
        next_cand_win_d2: list[np.ndarray] = []
        for idx, win_vec in enumerate(old_win_dpsi):
            win_u, win_d, _win_s = _rotation_triplet(win_vec, step, 0.0)
            cross_u, _cross_d, _cross_s = _rotation_triplet(old_cand_win_d2[idx], step, 0.0)
            next_win_dpsi.append(np.asarray(win_u, dtype=complex))
            next_cand_win_d2.append(np.asarray(cross_u + win_d, dtype=complex))
        win_dpsi = next_win_dpsi
        cand_win_d2 = next_cand_win_d2
    return cand_dpsi, cand_d2psi, cand_win_d2


def _regularized_solve(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    base_ridge: float,
    growth_factor: float,
    max_steps: int,
    require_pd: bool,
) -> tuple[np.ndarray, float, np.ndarray]:
    mat = np.asarray(matrix, dtype=float)
    vec = np.asarray(rhs, dtype=float).reshape(-1)
    n = int(mat.shape[0])
    if n == 0:
        return np.zeros(0, dtype=float), float(max(base_ridge, 0.0)), np.zeros((0, 0), dtype=float)
    eye = np.eye(n, dtype=float)
    ridge = float(max(base_ridge, 0.0))
    if ridge == 0.0:
        ridge = 1e-12
    mat_sym = 0.5 * (mat + mat.T)
    for _ in range(int(max(1, max_steps))):
        trial = mat_sym + ridge * eye
        try:
            if require_pd:
                np.linalg.cholesky(trial)
            sol = np.linalg.solve(trial, vec)
            return np.asarray(sol, dtype=float), float(ridge), np.asarray(trial, dtype=float)
        except Exception:
            ridge *= float(max(growth_factor, 2.0))
    trial = mat_sym + ridge * eye
    if require_pd:
        np.linalg.cholesky(trial)
    sol = np.linalg.solve(trial, vec)
    return np.asarray(sol, dtype=float), float(ridge), np.asarray(trial, dtype=float)


class Phase2NoveltyOracle:
    def prepare_scaffold_context(
        self,
        *,
        selected_ops: Sequence[Any],
        theta: np.ndarray,
        psi_ref: np.ndarray,
        psi_state: np.ndarray,
        h_compiled: CompiledPolynomialAction,
        hpsi_state: np.ndarray,
        refit_window_indices: Sequence[int],
        pauli_action_cache: dict[str, Any] | None = None,
        parameterization_mode: str = "per_pauli_term",
    ) -> _ScaffoldDerivativeContext:
        inherited_window = [int(i) for i in refit_window_indices]
        psi_current = np.asarray(psi_state, dtype=complex).reshape(-1)
        hpsi_current = np.asarray(hpsi_state, dtype=complex).reshape(-1)
        if not inherited_window:
            return _ScaffoldDerivativeContext(
                psi_state=psi_current,
                hpsi_state=hpsi_current,
                refit_window_indices=tuple(),
                dpsi_window=tuple(),
                tangents_window=tuple(),
                Q_window=np.zeros((0, 0), dtype=float),
                H_window_hessian=np.zeros((0, 0), dtype=float),
            )
        executor = _executor_for_terms(
            selected_ops,
            pauli_action_cache=pauli_action_cache,
            parameterization_mode=str(parameterization_mode),
        )
        _psi_final, dpsi_window, d2psi_window = _propagate_executor_derivatives(
            executor=executor,
            theta=np.asarray(theta, dtype=float),
            psi_ref=np.asarray(psi_ref, dtype=complex),
            active_indices=inherited_window,
        )
        tangents_window = [_horizontal_tangent(psi_current, dpsi_vec) for dpsi_vec in dpsi_window]
        q_window = _tangent_overlap_matrix(tangents_window)
        hdpsi_window = [
            apply_compiled_polynomial(np.asarray(dpsi_vec, dtype=complex), h_compiled)
            for dpsi_vec in dpsi_window
        ]
        m = int(len(inherited_window))
        hess = np.zeros((m, m), dtype=float)
        for row in range(m):
            for col in range(m):
                hess[row, col] = _energy_hessian_entry(
                    dpsi_left=dpsi_window[row],
                    dpsi_right=dpsi_window[col],
                    d2psi=d2psi_window[row][col],
                    hpsi_state=hpsi_current,
                    hdpsi_right=hdpsi_window[col],
                )
        hess = 0.5 * (hess + hess.T)
        return _ScaffoldDerivativeContext(
            psi_state=psi_current,
            hpsi_state=hpsi_current,
            refit_window_indices=tuple(inherited_window),
            dpsi_window=tuple(np.asarray(x, dtype=complex) for x in dpsi_window),
            tangents_window=tuple(np.asarray(x, dtype=complex) for x in tangents_window),
            Q_window=np.asarray(q_window, dtype=float),
            H_window_hessian=np.asarray(hess, dtype=float),
        )

    def estimate(
        self,
        *,
        scaffold_context: _ScaffoldDerivativeContext,
        candidate_label: str,
        candidate_term: Any,
        compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
        pauli_action_cache: dict[str, Any] | None = None,
        novelty_eps: float = 1e-6,
        parameterization_mode: str = "per_pauli_term",
    ) -> Mapping[str, Any]:
        del candidate_label, compiled_cache, novelty_eps
        cand_dpsi, cand_d2psi, cand_window_d2 = _propagate_append_candidate(
            candidate_term=candidate_term,
            psi_state=scaffold_context.psi_state,
            window_dpsi=list(scaffold_context.dpsi_window),
            pauli_action_cache=pauli_action_cache,
            parameterization_mode=str(parameterization_mode),
        )
        cand_tangent = _horizontal_tangent(scaffold_context.psi_state, cand_dpsi)
        q_window = np.asarray(
            [float(np.real(np.vdot(tang_j, cand_tangent))) for tang_j in scaffold_context.tangents_window],
            dtype=float,
        )
        F_raw = float(max(0.0, np.real(np.vdot(cand_tangent, cand_tangent))))
        return {
            "novelty_mode": "append_exact_tangent_context_v1",
            "candidate_dpsi": np.asarray(cand_dpsi, dtype=complex),
            "candidate_d2psi": np.asarray(cand_d2psi, dtype=complex),
            "candidate_window_d2": [np.asarray(x, dtype=complex) for x in cand_window_d2],
            "candidate_tangent": np.asarray(cand_tangent, dtype=complex),
            "F_raw": float(F_raw),
            "Q_window": np.asarray(scaffold_context.Q_window, dtype=float),
            "q_window": np.asarray(q_window, dtype=float),
        }


class Phase2CurvatureOracle:
    def estimate(
        self,
        *,
        base_feature: CandidateFeatures,
        novelty_info: Mapping[str, Any],
        scaffold_context: _ScaffoldDerivativeContext,
        h_compiled: CompiledPolynomialAction,
        cfg: FullScoreConfig,
        optimizer_memory: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        del optimizer_memory
        F_raw = float(max(0.0, novelty_info.get("F_raw", base_feature.F_raw or base_feature.F_metric)))
        q_window = np.asarray(novelty_info.get("q_window", []), dtype=float).reshape(-1)
        Q_window = np.asarray(novelty_info.get("Q_window", scaffold_context.Q_window), dtype=float)
        cand_dpsi = np.asarray(novelty_info.get("candidate_dpsi"), dtype=complex).reshape(-1)
        cand_d2psi = np.asarray(novelty_info.get("candidate_d2psi"), dtype=complex).reshape(-1)
        cand_window_d2 = [
            np.asarray(x, dtype=complex).reshape(-1)
            for x in novelty_info.get("candidate_window_d2", [])
        ]
        hdpsi_candidate = apply_compiled_polynomial(cand_dpsi, h_compiled)
        h_raw = _energy_hessian_entry(
            dpsi_left=cand_dpsi,
            dpsi_right=cand_dpsi,
            d2psi=cand_d2psi,
            hpsi_state=scaffold_context.hpsi_state,
            hdpsi_right=hdpsi_candidate,
        )
        b_mixed = np.zeros(len(scaffold_context.refit_window_indices), dtype=float)
        for idx, dpsi_window in enumerate(scaffold_context.dpsi_window):
            if idx >= len(cand_window_d2):
                break
            b_mixed[idx] = _energy_hessian_entry(
                dpsi_left=dpsi_window,
                dpsi_right=cand_dpsi,
                d2psi=cand_window_d2[idx],
                hpsi_state=scaffold_context.hpsi_state,
                hdpsi_right=hdpsi_candidate,
            )
        H_window = np.asarray(scaffold_context.H_window_hessian, dtype=float)
        if H_window.size == 0:
            h_eff = float(h_raw)
            F_red = float(max(F_raw, float(cfg.metric_floor)))
            novelty = 1.0
            ridge_used = float(max(cfg.lambda_H, 0.0))
            mode = "append_exact_empty_window"
        else:
            minv_b, ridge_used, _M_window = _regularized_solve(
                H_window,
                b_mixed,
                base_ridge=float(max(cfg.lambda_H, 0.0)),
                growth_factor=float(max(cfg.ridge_growth_factor, 2.0)),
                max_steps=int(max(1, cfg.ridge_max_steps)),
                require_pd=True,
            )
            h_eff = float(h_raw - float(b_mixed.T @ minv_b))
            F_red_exact = float(F_raw - 2.0 * float(q_window.T @ minv_b) + float(minv_b.T @ Q_window @ minv_b))
            F_red = float(max(F_red_exact, float(cfg.metric_floor)))
            q_reduced = np.asarray(q_window - Q_window @ minv_b, dtype=float)
            collapse_floor = max(
                float(cfg.metric_floor),
                float(cfg.reduced_metric_collapse_rel_tol) * float(max(F_raw, float(cfg.metric_floor))),
            )
            metric_collapse = bool(F_red_exact <= collapse_floor)
            if metric_collapse:
                novelty = 0.0
                mode = "append_exact_metric_collapse_v1"
            else:
                qsol, _nov_ridge, _Qreg = _regularized_solve(
                    Q_window,
                    q_reduced,
                    base_ridge=float(max(cfg.novelty_eps, 0.0)),
                    growth_factor=float(max(cfg.ridge_growth_factor, 2.0)),
                    max_steps=int(max(1, cfg.ridge_max_steps)),
                    require_pd=True,
                )
                novelty_raw = 1.0 - float(q_reduced.T @ qsol) / float(F_red)
                novelty = float(min(1.0, max(0.0, novelty_raw)))
                mode = (
                    "append_exact_window_hessian_ridge_grown_v1"
                    if float(ridge_used) > float(max(cfg.lambda_H, 0.0))
                    else "append_exact_window_hessian_v1"
                )
        return {
            "h_raw": float(h_raw),
            "b_mixed": [float(x) for x in b_mixed.tolist()],
            "H_window_hessian": [[float(x) for x in row] for row in H_window.tolist()],
            "h_eff": float(h_eff),
            "F_red": float(F_red),
            "novelty": float(novelty),
            "ridge_used": float(ridge_used),
            "curvature_mode": str(mode),
        }


def full_v2_score(feat: CandidateFeatures, cfg: FullScoreConfig) -> tuple[float, str]:
    if (not bool(feat.stage_gate_open)) or (not bool(feat.leakage_gate_open)):
        return float("-inf"), "blocked_stage_or_leakage_gate"
    if not bool(feat.compile_gate_open):
        return float("-inf"), "compile_gate_closed"
    if float(feat.leakage_penalty) > float(cfg.leakage_cap):
        return float("-inf"), "leakage_cap"
    g_lcb = max(float(feat.g_abs) - float(cfg.z_alpha) * float(max(0.0, feat.sigma_hat)), 0.0)
    if g_lcb <= 0.0:
        return 0.0, "nonpositive_gradient"
    F_red_raw = feat.F_red if feat.F_red is not None else feat.F_raw
    if F_red_raw is None:
        if float(feat.F_metric) <= 0.0:
            return 0.0, "nonpositive_metric"
        F_red = float(max(float(feat.F_metric), float(cfg.metric_floor)))
        h_eff = float(max(0.0, feat.h_hat if feat.h_hat is not None else float(cfg.lambda_F) * float(feat.F_metric)))
        novelty = 1.0 if feat.novelty is None else min(max(float(feat.novelty), 0.0), 1.0)
        fallback_mode = "legacy_metric_path"
    else:
        h_eff = float(feat.h_eff if feat.h_eff is not None else (feat.h_hat if feat.h_hat is not None else 0.0))
        F_red = float(max(float(F_red_raw), float(cfg.metric_floor)))
        novelty = 1.0 if feat.novelty is None else float(min(1.0, max(0.0, feat.novelty)))
        if str(feat.curvature_mode).startswith("append_exact_metric_collapse") or novelty <= 0.0:
            return 0.0, "reduced_metric_collapse"
        fallback_mode = (
            "append_exact_reduced_path_ridge_grown"
            if feat.ridge_used is not None and float(feat.ridge_used) > float(max(cfg.lambda_H, 0.0))
            else ("append_exact_empty_window" if len(feat.refit_window_indices) == 0 else "append_exact_reduced_path")
        )
    delta_e = trust_region_drop(g_lcb, float(max(0.0, h_eff)), F_red, float(cfg.rho))
    if delta_e <= 0.0:
        return 0.0, fallback_mode
    K = (
        1.0
        + float(cfg.wD) * normalize(float(feat.depth_cost), float(cfg.depth_ref))
        + float(cfg.wG) * normalize(float(feat.new_group_cost), float(cfg.group_ref))
        + float(cfg.wC) * normalize(float(feat.new_shot_cost), float(cfg.shot_ref))
        + float(cfg.wP) * normalize(float(feat.opt_dim_cost), float(cfg.optdim_ref))
        + float(cfg.wc) * normalize(float(feat.reuse_count_cost), float(cfg.reuse_ref))
    )
    lifetime_components = lifetime_weight_components(feat, cfg)
    K = float(K + float(cfg.lifetime_weight) * float(lifetime_components.get("total", 0.0)))
    score = (
        math.exp(-float(cfg.eta_L) * float(feat.leakage_penalty))
        * (float(novelty) ** float(cfg.gamma_N))
        * float(delta_e)
        / float(K)
    )
    return float(score), str(fallback_mode)


def build_full_candidate_features(
    *,
    base_feature: CandidateFeatures,
    candidate_term: Any,
    cfg: FullScoreConfig,
    novelty_oracle: Phase2NoveltyOracle,
    curvature_oracle: Phase2CurvatureOracle,
    scaffold_context: _ScaffoldDerivativeContext,
    h_compiled: CompiledPolynomialAction,
    compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
    optimizer_memory: Mapping[str, Any] | None = None,
    parameterization_mode: str = "per_pauli_term",
) -> CandidateFeatures:
    novelty_info = novelty_oracle.estimate(
        scaffold_context=scaffold_context,
        candidate_label=str(base_feature.candidate_label),
        candidate_term=candidate_term,
        compiled_cache=compiled_cache,
        pauli_action_cache=pauli_action_cache,
        novelty_eps=float(cfg.novelty_eps),
        parameterization_mode=str(parameterization_mode),
    )
    curvature_info = curvature_oracle.estimate(
        base_feature=base_feature,
        novelty_info=novelty_info,
        scaffold_context=scaffold_context,
        h_compiled=h_compiled,
        cfg=cfg,
        optimizer_memory=optimizer_memory,
    )
    raw_geometry = phase2_raw_geometry_score(
        base_feature,
        F_raw=float(max(0.0, novelty_info.get("F_raw", base_feature.F_metric))),
        h_raw=float(curvature_info.get("h_raw", 0.0)),
        q_window=list(novelty_info.get("q_window", [])),
        Q_window=np.asarray(novelty_info.get("Q_window", scaffold_context.Q_window), dtype=float),
        cfg=cfg,
    )
    feat = _replace_feature(
        base_feature,
        novelty=float(curvature_info.get("novelty", 1.0)),
        novelty_mode=str(novelty_info.get("novelty_mode", "append_exact_tangent_context_v1")),
        curvature_mode=str(curvature_info.get("curvature_mode", "append_exact_window_hessian_v1")),
        F_metric=float(max(0.0, novelty_info.get("F_raw", base_feature.F_metric))),
        metric_proxy=float(max(0.0, novelty_info.get("F_raw", base_feature.metric_proxy))),
        F_raw=float(max(0.0, novelty_info.get("F_raw", base_feature.F_raw or base_feature.F_metric))),
        h_eff=float(curvature_info.get("h_eff", 0.0)),
        F_red=float(curvature_info.get("F_red", novelty_info.get("F_raw", 0.0))),
        ridge_used=float(curvature_info.get("ridge_used", max(cfg.lambda_H, 0.0))),
        h_hat=float(curvature_info.get("h_raw", 0.0)),
        b_hat=[float(x) for x in curvature_info.get("b_mixed", [])],
        H_window=[[float(x) for x in row] for row in curvature_info.get("H_window_hessian", [])],
        score_version=str(cfg.score_version),
        confidence_factor=float(raw_geometry.get("confidence_factor", 1.0)),
        phase2_raw_overlap_max=float(raw_geometry.get("phase2_raw_overlap_max", 0.0)),
        phase2_raw_novelty=float(raw_geometry.get("phase2_raw_novelty", 1.0)),
        phase2_raw_trust_gain=float(raw_geometry.get("phase2_raw_trust_gain", 0.0)),
        phase2_raw_score=float(raw_geometry.get("phase2_raw_score", 0.0)),
        phase2_burden_total=float(raw_geometry.get("phase2_burden_total", 1.0)),
    )
    feat = _replace_feature(
        feat,
        lifetime_weight_components=dict(lifetime_weight_components(feat, cfg)),
        lifetime_cost_mode=str(cfg.lifetime_cost_mode),
        remaining_evaluations_proxy_mode=str(cfg.remaining_evaluations_proxy_mode),
    )
    score, fallback_mode = full_v2_score(feat, cfg)
    selector_geometry_mode = str(getattr(cfg, "phase3_selector_geometry_mode", "reduced")).strip().lower()
    if selector_geometry_mode not in {"reduced", "raw_exact"}:
        selector_geometry_mode = "reduced"
    phase3_burden_total = float(_cheap_burden_total(feat, cfg))
    selector_score = float(score)
    selector_burden = float(phase3_burden_total)
    if selector_geometry_mode == "raw_exact":
        selector_score = float(feat.phase2_raw_score or 0.0)
        selector_burden = float(feat.phase2_burden_total or phase3_burden_total)
    return _replace_feature(
        feat,
        full_v2_score=float(score),
        phase3_reduced_novelty=float(feat.novelty if feat.novelty is not None else 1.0),
        phase3_reduced_trust_gain=float(
            trust_region_drop(
                float(max(0.0, feat.g_lcb)),
                float(max(0.0, feat.h_eff or 0.0)),
                float(max(feat.F_red or feat.F_metric, cfg.metric_floor)),
                float(cfg.rho),
            )
        ),
        phase3_burden_total=float(phase3_burden_total),
        selector_score=float(selector_score),
        selector_burden=float(selector_burden),
        selector_geometry_mode=str(selector_geometry_mode),
        actual_fallback_mode=str(fallback_mode),
    )


def resolve_reopt_active_indices(
    *,
    policy: str,
    n: int,
    theta: np.ndarray,
    window_size: int = 3,
    window_topk: int = 0,
    periodic_full_refit_triggered: bool = False,
) -> tuple[list[int], str]:
    policy_key = str(policy).strip().lower()
    if n <= 0:
        return [], policy_key
    if policy_key == "append_only":
        return [n - 1], "append_only"
    if policy_key == "full":
        return list(range(n)), "full"
    if policy_key != "windowed":
        raise ValueError(f"Unknown reopt policy '{policy_key}'.")
    if periodic_full_refit_triggered:
        return list(range(n)), "windowed_periodic_full"
    w_eff = min(int(window_size), n)
    newest = list(range(n - w_eff, n))
    older_start = n - w_eff
    if older_start <= 0 or int(window_topk) <= 0:
        return sorted(newest), "windowed"
    older_candidates = list(range(0, older_start))
    older_ranked = sorted(older_candidates, key=lambda i: (-abs(float(theta[i])), i))
    k_eff = min(int(window_topk), len(older_ranked))
    selected_older = older_ranked[:k_eff]
    active = sorted(set(newest) | set(selected_older))
    return active, "windowed"


def predict_reopt_window_for_position(
    *,
    theta: np.ndarray,
    position_id: int,
    policy: str,
    window_size: int,
    window_topk: int,
    periodic_full_refit_triggered: bool,
) -> list[int]:
    theta_arr = np.asarray(theta, dtype=float).reshape(-1)
    theta_plus = np.insert(theta_arr, int(position_id), 0.0)
    active, _mode = resolve_reopt_active_indices(
        policy=str(policy),
        n=int(theta_plus.size),
        theta=np.asarray(theta_plus, dtype=float),
        window_size=int(window_size),
        window_topk=int(window_topk),
        periodic_full_refit_triggered=bool(periodic_full_refit_triggered),
    )
    return [int(i) for i in active]


def make_reduced_objective(
    full_theta: np.ndarray,
    active_indices: list[int],
    obj_fn: Any,
) -> tuple[Any, np.ndarray]:
    frozen_theta = np.array(full_theta, copy=True)
    active_idx = list(active_indices)
    x0 = np.array([float(frozen_theta[i]) for i in active_idx], dtype=float)
    if len(active_idx) == len(frozen_theta):
        return obj_fn, np.array(frozen_theta, copy=True)

    def _reduced(x_active: np.ndarray) -> float:
        full = np.array(frozen_theta, copy=True)
        x_arr = np.asarray(x_active, dtype=float).ravel()
        for k, idx in enumerate(active_idx):
            full[idx] = float(x_arr[k])
        return float(obj_fn(full))

    return _reduced, x0
