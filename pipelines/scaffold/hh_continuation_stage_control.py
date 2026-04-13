#!/usr/bin/env python3
"""Stage and position-probe policy for HH continuation Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from pipelines.hardcoded.hh_continuation_types import PhaseControllerSnapshot


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
class PositionProbeDecision:
    should_probe: bool
    reason: str
    positions: list[int]


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
    def __init__(self, cfg: StageControllerConfig) -> None:
        self.cfg = cfg
        self._stage = "core"
        self._admission_deltas: list[float] = []
        self._last_snapshot: PhaseControllerSnapshot | None = None

    def clone(self) -> "StageController":
        cloned = StageController(self.cfg)
        cloned._stage = str(self._stage)
        cloned._admission_deltas = [float(x) for x in self._admission_deltas]
        cloned._last_snapshot = self._last_snapshot
        return cloned

    def snapshot(self) -> dict[str, str | StageControllerConfig]:
        return {
            "cfg": self.cfg,
            "stage": str(self._stage),
            "admission_deltas": [float(x) for x in self._admission_deltas],
            "last_snapshot": self._last_snapshot,
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, object]) -> "StageController":
        cfg = snapshot.get("cfg")
        if not isinstance(cfg, StageControllerConfig):
            raise TypeError("StageController snapshot missing StageControllerConfig.")
        out = cls(cfg)
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
            phase_shots={
                "phase1": int(round(self.cfg.shot_min + (self.cfg.shot_max - self.cfg.shot_min) * late)),
                "phase2": int(round(self.cfg.shot_min + (self.cfg.shot_max - self.cfg.shot_min) * late)),
                "phase3": int(round(self.cfg.shot_min + (self.cfg.shot_max - self.cfg.shot_min) * late)),
            },
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
        scores = sorted(
            [float(x) for x in phase1_raw_scores if float(x) == float(x)],
            reverse=True,
        )
        top = float(scores[0]) if scores else 0.0
        second = float(scores[1]) if len(scores) > 1 else 0.0
        frontier_ratio = float((second + 1e-12) / (top + 1e-12)) if (scores or top == 0.0) else 1.0
        snapshot = PhaseControllerSnapshot(
            step_index=int(pre_snapshot.step_index),
            depth_local=int(pre_snapshot.depth_local),
            depth_left=int(pre_snapshot.depth_left),
            runway_ratio=float(pre_snapshot.runway_ratio),
            early_coordinate=float(pre_snapshot.early_coordinate),
            late_coordinate=float(pre_snapshot.late_coordinate),
            frontier_ratio=float(max(0.0, min(1.0, frontier_ratio))),
            phase_thresholds=dict(pre_snapshot.phase_thresholds),
            phase_caps={k: max(1, int(v)) for k, v in pre_snapshot.phase_caps.items()},
            phase_shots=dict(pre_snapshot.phase_shots),
            phase_uncertainty={
                "phase2": float(max(0.0, u_sigma_phase2 or 0.0)),
                "phase3": float(max(0.0, u_sigma_phase3 or 0.0)),
            },
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
