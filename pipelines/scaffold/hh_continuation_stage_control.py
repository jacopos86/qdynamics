#!/usr/bin/env python3
"""Stage and position-probe policy for HH continuation Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from pipelines.scaffold.hh_continuation_types import PhaseControllerSnapshot


@dataclass(frozen=True)
class _RunwayForecastConfig:
    tau_delta: float = 1e-6
    rho_min: float = 0.25
    s_min: float = 1e-3
    ewma_alpha: float = 0.4
    recent_window: int = 3
    older_window: int = 6
    low_streak_patience: int = 3
    plateau_eta0: float = 0.85
    plateau_eta1: float = 0.15
    beta_0: float = -2.5
    beta_stag: float = 2.0
    beta_front: float = 1.25
    eta_h: float = 0.35
    eps: float = 1e-12


_RUNWAY_FORECAST_CFG = _RunwayForecastConfig()
_PASSIVE_PHASE_LIVE = {
    "phase1": True,
    "phase2": True,
    "phase3": True,
}
_PASSIVE_PHASE_NULL_STREAKS = {"phase2": 0, "phase3": 0}
_PASSIVE_PHASE_NULL_REASONS = {
    "phase1": "phase_live_retired_non_authoritative",
    "phase2": "phase_live_retired_non_authoritative",
    "phase3": "phase_live_retired_non_authoritative",
}


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
    runway_envelope_delta_h: float = 0.0
    runway_envelope_delta_m: float = 0.0
    runway_envelope_delta_s: float = 0.0
    shot_cap_phase1: int = 0
    shot_cap_phase2: int = 0
    shot_cap_phase3: int = 0
    shot_frontier_uplift_phase1: float = 0.0
    shot_frontier_uplift_phase2: float = 0.0
    shot_frontier_uplift_phase3: float = 0.0
    shot_sigma_uplift_phase2: float = 0.0
    shot_sigma_uplift_phase3: float = 0.0
    shot_snr_kappa_phase1: float = 0.0
    shot_snr_kappa_phase2: float = 0.0
    shot_snr_kappa_phase3: float = 0.0
    shot_delta_floor_phase1: float = 1e-12
    shot_delta_floor_phase2: float = 1e-12
    shot_delta_floor_phase3: float = 1e-12


@dataclass(frozen=True)
class PositionProbeDecision:
    should_probe: bool
    reason: str
    positions: list[int]


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _sigmoid(value: float) -> float:
    if float(value) >= 0.0:
        z = math.exp(-float(value))
        return float(1.0 / (1.0 + z))
    z = math.exp(float(value))
    return float(z / (1.0 + z))


def _normal_cdf(value: float) -> float:
    return float(0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0))))


def _ewma(values: list[float], *, alpha: float) -> float:
    if not values:
        return 0.0
    out = float(values[0])
    for value in values[1:]:
        out = float(alpha) * float(value) + (1.0 - float(alpha)) * float(out)
    return float(out)


def _mad(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    mid = len(ordered) // 2
    median = float(ordered[mid]) if len(ordered) % 2 else float(0.5 * (ordered[mid - 1] + ordered[mid]))
    deviations = sorted(abs(float(v) - median) for v in ordered)
    dmid = len(deviations) // 2
    return float(deviations[dmid]) if len(deviations) % 2 else float(0.5 * (deviations[dmid - 1] + deviations[dmid]))


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
    def __init__(
        self,
        cfg: StageControllerConfig,
        *,
        configured_terminal_phase: int = 3,
    ) -> None:
        self.cfg = cfg
        terminal_phase = int(configured_terminal_phase)
        if terminal_phase not in {1, 2, 3}:
            raise ValueError("configured_terminal_phase must be 1, 2, or 3.")
        self._configured_terminal_phase = terminal_phase
        self._stage = "core"
        self._admission_deltas: list[float] = []
        self._last_snapshot: PhaseControllerSnapshot | None = None

    def clone(self) -> "StageController":
        cloned = StageController(
            self.cfg,
            configured_terminal_phase=self._configured_terminal_phase,
        )
        cloned._stage = str(self._stage)
        cloned._admission_deltas = [float(x) for x in self._admission_deltas]
        cloned._last_snapshot = self._last_snapshot
        return cloned

    def snapshot(self) -> dict[str, object]:
        return {
            "cfg": self.cfg,
            "stage": str(self._stage),
            "admission_deltas": [float(x) for x in self._admission_deltas],
            "last_snapshot": self._last_snapshot,
            "configured_terminal_phase": self._configured_terminal_phase,
            "phase_live": dict(_PASSIVE_PHASE_LIVE),
            "phase_null_streaks": dict(_PASSIVE_PHASE_NULL_STREAKS),
            "phase_null_reasons": dict(_PASSIVE_PHASE_NULL_REASONS),
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, object]) -> "StageController":
        cfg = snapshot.get("cfg")
        if not isinstance(cfg, StageControllerConfig):
            raise TypeError("StageController snapshot missing StageControllerConfig.")
        out = cls(
            cfg,
            configured_terminal_phase=int(
                snapshot.get("configured_terminal_phase", 3)
            ),
        )
        out._stage = str(snapshot.get("stage", "core"))
        out._admission_deltas = [float(x) for x in snapshot.get("admission_deltas", [])]
        last_snapshot = snapshot.get("last_snapshot")
        out._last_snapshot = last_snapshot if isinstance(last_snapshot, PhaseControllerSnapshot) else None
        return out

    @property
    def stage_name(self) -> str:
        return str(self._stage)

    def start_with_seed(self) -> None:
        self._stage = "seed"

    def begin_core(self) -> None:
        self._stage = "core"

    def _runway_ratio(self, *, depth_local: int, max_depth: int) -> float:
        depth_now = max(0, int(depth_local))
        depth_cap = max(depth_now, int(max_depth))
        depth_left = max(0, depth_cap - depth_now)
        if depth_cap <= 0:
            return 0.0
        return float(max(0.0, min(1.0, depth_left / float(depth_cap))))

    def _runway_telemetry(
        self,
        *,
        depth_left: int,
        depth_cap: int,
        frontier_ratio: float,
    ) -> dict[str, float]:
        model = _RUNWAY_FORECAST_CFG
        depth_left_val = max(0, int(depth_left))
        depth_cap_val = max(1, int(depth_cap))
        frontier_val = _clip01(frontier_ratio)
        tau_use = float(max(float(self.cfg.weak_drop_threshold), float(model.eps)))
        tau_low_plus = float(math.asinh(tau_use / float(model.tau_delta)))
        clean_deltas = [
            float(max(0.0, float(delta)))
            for delta in self._admission_deltas
            if math.isfinite(float(delta))
        ]
        transformed = [
            float(math.asinh(float(delta) / float(model.tau_delta)))
            for delta in clean_deltas
        ]
        if transformed:
            m_t = float(_ewma(transformed, alpha=float(model.ewma_alpha)))
            s_t = float(max(float(model.s_min), _mad(transformed)))
        else:
            m_t = float(tau_low_plus + float(model.s_min))
            s_t = float(model.s_min)
        recent_raw = clean_deltas[-int(max(1, model.recent_window)) :]
        older_stop = len(clean_deltas) - len(recent_raw)
        older_start = max(0, older_stop - int(max(1, model.older_window)))
        older_raw = clean_deltas[older_start:older_stop]
        if recent_raw:
            recent_ewma = float(_ewma(recent_raw, alpha=float(model.ewma_alpha)))
        else:
            recent_ewma = float(max(tau_use, float(model.eps)))
        if older_raw:
            older_ewma = float(_ewma(older_raw, alpha=float(model.ewma_alpha)))
        else:
            older_ewma = float(recent_ewma)
        rho_t = float(
            max(
                float(model.rho_min),
                min(1.0, float(recent_ewma) / float(max(older_ewma, float(model.eps)))),
            )
        )
        gamma_t = float(max(0.0, -math.log(max(float(rho_t), float(model.eps)))))
        plateau_window = max(
            1,
            int(math.ceil(int(max(1, self.cfg.plateau_patience)) * float(depth_left_val) / float(depth_cap_val))),
        )
        low_window = max(
            1,
            int(
                math.ceil(
                    int(max(1, model.low_streak_patience)) * float(depth_left_val) / float(depth_cap_val)
                )
            ),
        )
        tau_plat_plus = float(
            (float(model.plateau_eta0) + float(model.plateau_eta1) * (1.0 - float(depth_left_val) / float(depth_cap_val)))
            * float(m_t)
        )
        plateau_hits = 0
        for value in reversed(transformed[-plateau_window:]):
            if float(value) <= float(tau_plat_plus):
                plateau_hits += 1
            else:
                break
        low_hits = 0
        for value in reversed(transformed[-low_window:]):
            if float(value) <= float(tau_low_plus):
                low_hits += 1
            else:
                break
        plateau_streak = (
            float(max(1, self.cfg.plateau_patience)) / float(plateau_window) * float(plateau_hits)
        )
        low_streak = (
            float(max(1, model.low_streak_patience)) / float(low_window) * float(low_hits)
        )
        u_stag = _clip01(
            max(
                float(plateau_streak) / float(max(1, self.cfg.plateau_patience)),
                float(low_streak) / float(max(1, model.low_streak_patience)),
            )
        )
        survival = 1.0
        n_rem_hat = 0.0
        for k in range(1, depth_left_val + 1):
            q_t = _normal_cdf(
                (
                    float(m_t)
                    - float(gamma_t) * float(k - 1)
                    - float(tau_low_plus)
                )
                / float(max(float(s_t), float(model.s_min)))
            )
            n_rem_hat += float(survival) * float(q_t)
            h_t = _sigmoid(
                float(model.beta_0)
                + float(model.beta_stag) * float(u_stag)
                + float(model.beta_front) * float(frontier_val)
                + float(model.eta_h) * float(k - 1)
            )
            survival *= float(max(0.0, 1.0 - float(h_t)))
        n_rem_hat = float(max(0.0, min(float(depth_left_val), float(n_rem_hat))))
        useful_horizon = float(min(float(depth_left_val), float(n_rem_hat)))
        runway_fraction = _clip01(float(n_rem_hat) / float(max(float(depth_left_val), float(model.eps))))
        confidence_delta = float(
            max(0.0, float(self.cfg.runway_envelope_delta_h))
            + max(0.0, float(self.cfg.runway_envelope_delta_m)) * float(frontier_val)
            + max(0.0, float(self.cfg.runway_envelope_delta_s)) * float(u_stag)
        )
        n_rem_low = float(max(0.0, float(n_rem_hat) - float(confidence_delta)))
        n_rem_high = float(min(float(depth_left_val), float(n_rem_hat) + float(confidence_delta)))
        confidence_ratio = _clip01(
            1.0 - float(confidence_delta) / float(max(float(depth_left_val), float(model.eps)))
        )
        return {
            "u_stag": float(u_stag),
            "m_t": float(m_t),
            "s_t": float(s_t),
            "rho_t": float(rho_t),
            "gamma_t": float(gamma_t),
            "u_front": float(frontier_val),
            "n_rem_hat": float(n_rem_hat),
            "n_rem_low": float(n_rem_low),
            "n_rem_high": float(n_rem_high),
            "confidence_ratio": float(confidence_ratio),
            "useful_horizon": float(useful_horizon),
            "runway_fraction": float(runway_fraction),
            "H_t": float(useful_horizon),
        }

    def _phase_shot_cap(self, phase_name: str) -> int:
        configured = {
            "phase1": int(self.cfg.shot_cap_phase1),
            "phase2": int(self.cfg.shot_cap_phase2),
            "phase3": int(self.cfg.shot_cap_phase3),
        }.get(str(phase_name), 0)
        if configured > 0:
            return int(configured)
        return int(max(int(self.cfg.shot_min), int(self.cfg.shot_max), 1))

    def _phase_shot_uplift(self, *, phase_name: str, u_front: float, phase_uncertainty: dict[str, float]) -> float:
        frontier_weight = {
            "phase1": float(self.cfg.shot_frontier_uplift_phase1),
            "phase2": float(self.cfg.shot_frontier_uplift_phase2),
            "phase3": float(self.cfg.shot_frontier_uplift_phase3),
        }.get(str(phase_name), 0.0)
        sigma_weight = {
            "phase1": 0.0,
            "phase2": float(self.cfg.shot_sigma_uplift_phase2),
            "phase3": float(self.cfg.shot_sigma_uplift_phase3),
        }.get(str(phase_name), 0.0)
        return _clip01(
            float(frontier_weight) * float(u_front)
            + float(sigma_weight) * float(max(0.0, phase_uncertainty.get(str(phase_name), 0.0)))
        )

    def _phase_snr_shots(self, *, phase_name: str, signal: float, sigma: float) -> int:
        kappa = {
            "phase1": float(self.cfg.shot_snr_kappa_phase1),
            "phase2": float(self.cfg.shot_snr_kappa_phase2),
            "phase3": float(self.cfg.shot_snr_kappa_phase3),
        }.get(str(phase_name), 0.0)
        if kappa <= 0.0:
            return 0
        delta_floor = {
            "phase1": float(self.cfg.shot_delta_floor_phase1),
            "phase2": float(self.cfg.shot_delta_floor_phase2),
            "phase3": float(self.cfg.shot_delta_floor_phase3),
        }.get(str(phase_name), 1e-12)
        denom = max(float(signal) * float(signal), float(delta_floor) * float(delta_floor), 1e-300)
        return int(math.ceil(float(kappa) * float(kappa) * float(sigma) * float(sigma) / float(denom)))

    def _phase_thresholds_caps_shots(
        self,
        *,
        early: float,
        late: float,
        u_front: float,
        phase_uncertainty: dict[str, float],
    ) -> dict[str, dict[str, float | int]]:
        thresholds = {
            "phase1": float(self.cfg.tau_phase1_min + (self.cfg.tau_phase1_max - self.cfg.tau_phase1_min) * late),
            "phase2": float(self.cfg.tau_phase2_min + (self.cfg.tau_phase2_max - self.cfg.tau_phase2_min) * late),
            "phase3": float(self.cfg.tau_phase3_min + (self.cfg.tau_phase3_max - self.cfg.tau_phase3_min) * late),
        }
        cap_scheduled = {
            "phase1": int(math.ceil(self.cfg.cap_phase1_min + (self.cfg.cap_phase1_max - self.cfg.cap_phase1_min) * early)),
            "phase2": int(math.ceil(self.cfg.cap_phase2_min + (self.cfg.cap_phase2_max - self.cfg.cap_phase2_min) * early)),
            "phase3": int(math.ceil(self.cfg.cap_phase3_min + (self.cfg.cap_phase3_max - self.cfg.cap_phase3_min) * early)),
        }
        cap_effective = {
            phase: int(max(0, cap_scheduled[phase]))
            for phase in ("phase1", "phase2", "phase3")
        }
        maturity_floor: dict[str, int] = {}
        scheduled: dict[str, int] = {}
        snr: dict[str, int] = {}
        effective: dict[str, int] = {}
        uplift: dict[str, float] = {}
        fraction: dict[str, float] = {}
        signal: dict[str, float] = {}
        signal_floor: dict[str, float] = {}
        for phase in ("phase1", "phase2", "phase3"):
            maturity_floor[phase] = int(
                math.ceil(int(self.cfg.shot_min) + (int(self.cfg.shot_max) - int(self.cfg.shot_min)) * float(late))
            )
            uplift[phase] = self._phase_shot_uplift(
                phase_name=phase,
                u_front=float(u_front),
                phase_uncertainty=phase_uncertainty,
            )
            fraction[phase] = _clip01(float(late) + (1.0 - float(late)) * float(uplift[phase]))
            scheduled[phase] = int(
                math.ceil(int(self.cfg.shot_min) + (int(self.cfg.shot_max) - int(self.cfg.shot_min)) * float(fraction[phase]))
            )
            sigma = float(max(0.0, phase_uncertainty.get(phase, 0.0)))
            signal[phase] = float(max(0.0, 1.0 - float(uplift[phase])))
            signal_floor[phase] = {
                "phase1": float(self.cfg.shot_delta_floor_phase1),
                "phase2": float(self.cfg.shot_delta_floor_phase2),
                "phase3": float(self.cfg.shot_delta_floor_phase3),
            }[phase]
            snr[phase] = self._phase_snr_shots(phase_name=phase, signal=signal[phase], sigma=sigma)
            effective[phase] = int(
                min(
                    self._phase_shot_cap(phase),
                    max(scheduled[phase], snr[phase]),
                )
            )
        return {
            "thresholds": thresholds,
            "cap_scheduled": cap_scheduled,
            "cap_effective": cap_effective,
            "maturity_floor": maturity_floor,
            "scheduled": scheduled,
            "snr": snr,
            "effective": effective,
            "uplift": uplift,
            "fraction": fraction,
            "signal": signal,
            "signal_floor": signal_floor,
        }

    def pre_step_snapshot(self, *, depth_local: int, max_depth: int) -> PhaseControllerSnapshot:
        depth_now = max(0, int(depth_local))
        depth_cap = max(depth_now, int(max_depth))
        depth_left = max(0, depth_cap - depth_now)
        depth_runway_ratio = self._runway_ratio(depth_local=depth_now, max_depth=depth_cap)
        runway = self._runway_telemetry(
            depth_left=int(depth_left),
            depth_cap=int(depth_cap),
            frontier_ratio=(
                float(self._last_snapshot.frontier_ratio)
                if isinstance(self._last_snapshot, PhaseControllerSnapshot)
                else 1.0
            ),
        )
        runway_ratio = float(runway["runway_fraction"])
        early = float(runway_ratio ** float(max(self.cfg.runway_power_early, 1e-12)))
        late = float((1.0 - runway_ratio) ** float(max(self.cfg.runway_power_late, 1e-12)))
        phase_uncertainty = {"phase2": 0.0, "phase3": 0.0}
        phase_laws = self._phase_thresholds_caps_shots(
            early=float(early),
            late=float(late),
            u_front=float(runway["u_front"]),
            phase_uncertainty=phase_uncertainty,
        )
        snapshot = PhaseControllerSnapshot(
            step_index=int(len(self._admission_deltas)),
            depth_local=int(depth_now),
            depth_left=int(depth_left),
            runway_ratio=float(runway_ratio),
            early_coordinate=float(early),
            late_coordinate=float(late),
            frontier_ratio=(
                float(self._last_snapshot.frontier_ratio)
                if isinstance(self._last_snapshot, PhaseControllerSnapshot)
                else 1.0
            ),
            u_stag=float(runway["u_stag"]),
            m_t=float(runway["m_t"]),
            s_t=float(runway["s_t"]),
            rho_t=float(runway["rho_t"]),
            gamma_t=float(runway["gamma_t"]),
            u_front=float(runway["u_front"]),
            n_rem_hat=float(runway["n_rem_hat"]),
            n_rem_low=float(runway["n_rem_low"]),
            n_rem_high=float(runway["n_rem_high"]),
            confidence_ratio=float(runway["confidence_ratio"]),
            useful_horizon=float(runway["useful_horizon"]),
            runway_fraction=float(runway["runway_fraction"]),
            H_t=float(runway["H_t"]),
            phase_thresholds=dict(phase_laws["thresholds"]),
            phase_caps=dict(phase_laws["cap_effective"]),
            phase_shots=dict(phase_laws["effective"]),
            phase_uncertainty=phase_uncertainty,
            snapshot_version="phase123_controller_maturity_v2",
            depth_runway_ratio=float(depth_runway_ratio),
            phase_live=dict(_PASSIVE_PHASE_LIVE),
            terminal_phase=self._configured_terminal_phase,
            phase_null_reasons=dict(_PASSIVE_PHASE_NULL_REASONS),
            phase_null_streaks=dict(_PASSIVE_PHASE_NULL_STREAKS),
            phase_caps_scheduled=dict(phase_laws["cap_scheduled"]),
            phase_shots_maturity_floor=dict(phase_laws["maturity_floor"]),
            phase_shots_scheduled=dict(phase_laws["scheduled"]),
            phase_shots_snr=dict(phase_laws["snr"]),
            phase_shots_effective=dict(phase_laws["effective"]),
            phase_shot_uplift=dict(phase_laws["uplift"]),
            phase_shot_fraction=dict(phase_laws["fraction"]),
            phase_signal=dict(phase_laws["signal"]),
            phase_signal_floor=dict(phase_laws["signal_floor"]),
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
        scores = sorted(
            [float(x) for x in phase1_raw_scores if float(x) == float(x)],
            reverse=True,
        )
        top = float(scores[0]) if scores else 0.0
        second = float(scores[1]) if len(scores) > 1 else 0.0
        frontier_ratio = float((second + 1e-12) / (top + 1e-12)) if (scores or top == 0.0) else 1.0
        runway = self._runway_telemetry(
            depth_left=int(pre_snapshot.depth_left),
            depth_cap=int(max(int(pre_snapshot.depth_local) + int(pre_snapshot.depth_left), 1)),
            frontier_ratio=float(max(0.0, min(1.0, frontier_ratio))),
        )
        runway_ratio = float(runway["runway_fraction"])
        early = float(runway_ratio ** float(max(self.cfg.runway_power_early, 1e-12)))
        late = float((1.0 - runway_ratio) ** float(max(self.cfg.runway_power_late, 1e-12)))
        phase_uncertainty = {
            "phase2": float(max(0.0, u_sigma_phase2 or 0.0)),
            "phase3": float(max(0.0, u_sigma_phase3 or 0.0)),
        }
        phase_laws = self._phase_thresholds_caps_shots(
            early=float(early),
            late=float(late),
            u_front=float(runway["u_front"]),
            phase_uncertainty=phase_uncertainty,
        )
        snapshot = PhaseControllerSnapshot(
            step_index=int(pre_snapshot.step_index),
            depth_local=int(pre_snapshot.depth_local),
            depth_left=int(pre_snapshot.depth_left),
            runway_ratio=float(runway_ratio),
            early_coordinate=float(early),
            late_coordinate=float(late),
            frontier_ratio=float(max(0.0, min(1.0, frontier_ratio))),
            u_stag=float(runway["u_stag"]),
            m_t=float(runway["m_t"]),
            s_t=float(runway["s_t"]),
            rho_t=float(runway["rho_t"]),
            gamma_t=float(runway["gamma_t"]),
            u_front=float(runway["u_front"]),
            n_rem_hat=float(runway["n_rem_hat"]),
            n_rem_low=float(runway["n_rem_low"]),
            n_rem_high=float(runway["n_rem_high"]),
            confidence_ratio=float(runway["confidence_ratio"]),
            useful_horizon=float(runway["useful_horizon"]),
            runway_fraction=float(runway["runway_fraction"]),
            H_t=float(runway["H_t"]),
            phase_thresholds=dict(phase_laws["thresholds"]),
            phase_caps=dict(phase_laws["cap_effective"]),
            phase_shots=dict(phase_laws["effective"]),
            phase_uncertainty=phase_uncertainty,
            snapshot_version="phase123_controller_maturity_v2",
            depth_runway_ratio=float(pre_snapshot.depth_runway_ratio),
            phase_live=dict(_PASSIVE_PHASE_LIVE),
            terminal_phase=self._configured_terminal_phase,
            phase_null_reasons=dict(_PASSIVE_PHASE_NULL_REASONS),
            phase_null_streaks=dict(_PASSIVE_PHASE_NULL_STREAKS),
            phase_caps_scheduled=dict(phase_laws["cap_scheduled"]),
            phase_shots_maturity_floor=dict(phase_laws["maturity_floor"]),
            phase_shots_scheduled=dict(phase_laws["scheduled"]),
            phase_shots_snr=dict(phase_laws["snr"]),
            phase_shots_effective=dict(phase_laws["effective"]),
            phase_shot_uplift=dict(phase_laws["uplift"]),
            phase_shot_fraction=dict(phase_laws["fraction"]),
            phase_signal=dict(phase_laws["signal"]),
            phase_signal_floor=dict(phase_laws["signal_floor"]),
        )
        self._last_snapshot = snapshot
        return snapshot

    def record_admission(
        self,
        *,
        selector_step: int,
        energy_before: float,
        energy_after_refit: float,
    ) -> None:
        del selector_step
        self._admission_deltas.append(float(energy_before) - float(energy_after_refit))

    def resolve_stage_transition(
        self,
        *,
        drop_plateau_hits: int,
        trough_detected: bool,
        residual_opened: bool,
        residual_stage_available: bool,
    ) -> tuple[str, str]:
        if self._stage == "seed":
            self._stage = "core"
            return self._stage, "seed_complete"
        if self._stage == "core":
            if (
                int(drop_plateau_hits) >= int(self.cfg.plateau_patience)
                and (not bool(trough_detected))
                and bool(residual_stage_available)
            ):
                self._stage = "residual"
                return self._stage, "plateau_without_trough"
            if (
                int(drop_plateau_hits) >= int(self.cfg.plateau_patience)
                and (not bool(trough_detected))
                and (not bool(residual_stage_available))
            ):
                return self._stage, "stay_core_no_residual_stage"
            return self._stage, "stay_core"
        if self._stage == "residual":
            if bool(residual_opened):
                return self._stage, "stay_residual"
            return self._stage, "residual_closed"
        return self._stage, "unknown_stage"
