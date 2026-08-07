#!/usr/bin/env python3
"""Legacy motion telemetry helpers for the old checkpoint controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from pipelines.scaffold.hh_continuation_scoring import FullScoreConfig
from pipelines.time_dynamics.legacy.checkpoint_types import RealtimeCheckpointConfig


@dataclass(frozen=True)
class MotionSchedulerTelemetry:
    regime: str
    direction_cosine: float | None
    rate_change_l2: float | None
    rate_change_ratio: float | None
    acceleration_l2: float | None
    curvature_cosine: float | None
    direction_reversal: bool
    curvature_sign_flip: bool
    kink_score: float


"Built Math: (x, y) -> (x_pad, y_pad) with shared width max(dim(x), dim(y))."
def align_theta_vectors(
    lhs: np.ndarray | Sequence[float],
    rhs: np.ndarray | Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(lhs, dtype=float).reshape(-1)
    right = np.asarray(rhs, dtype=float).reshape(-1)
    width = max(int(left.size), int(right.size))
    out_left = np.zeros(int(width), dtype=float)
    out_right = np.zeros(int(width), dtype=float)
    out_left[: int(left.size)] = left
    out_right[: int(right.size)] = right
    return out_left, out_right


"Built Math: cos(x, y) = <x,y> / (||x|| ||y||), with None for degenerate norms."
def cosine_similarity(
    lhs: np.ndarray | Sequence[float],
    rhs: np.ndarray | Sequence[float],
) -> float | None:
    left, right = align_theta_vectors(lhs, rhs)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1.0e-12 or right_norm <= 1.0e-12:
        return None
    return float(np.dot(left, right) / max(left_norm * right_norm, 1.0e-12))


"Built Math: motion(theta_dot_k) -> telemetry via Δθ̇, cosines, acceleration, and thresholded regime classification."
def build_motion_telemetry(
    *,
    cfg: RealtimeCheckpointConfig,
    theta_dot: np.ndarray | Sequence[float],
    theta_dot_history: Sequence[np.ndarray | Sequence[float]],
    predicted_displacement: float,
) -> MotionSchedulerTelemetry:
    current = np.asarray(theta_dot, dtype=float).reshape(-1)
    if not theta_dot_history:
        return MotionSchedulerTelemetry(
            regime="bootstrap",
            direction_cosine=None,
            rate_change_l2=None,
            rate_change_ratio=None,
            acceleration_l2=None,
            curvature_cosine=None,
            direction_reversal=False,
            curvature_sign_flip=False,
            kink_score=0.0,
        )
    previous = np.asarray(theta_dot_history[-1], dtype=float).reshape(-1)
    current_aligned, previous_aligned = align_theta_vectors(current, previous)
    delta = np.asarray(current_aligned - previous_aligned, dtype=float)
    rate_change_l2 = float(np.linalg.norm(delta))
    previous_norm = float(np.linalg.norm(previous_aligned))
    current_norm = float(np.linalg.norm(current_aligned))
    rate_change_denom = max(previous_norm, current_norm, 1.0e-12)
    rate_change_ratio = float(rate_change_l2 / rate_change_denom)
    direction_cosine = cosine_similarity(current_aligned, previous_aligned)
    direction_reversal = bool(
        direction_cosine is not None
        and float(direction_cosine) <= float(cfg.motion_direction_reversal_cosine_threshold)
        and float(rate_change_ratio) >= float(cfg.motion_calm_rate_change_ratio_threshold)
        and float(rate_change_l2) >= float(cfg.motion_acceleration_l2_threshold)
    )
    acceleration_l2 = float(np.linalg.norm(delta))
    curvature_cosine: float | None = None
    curvature_sign_flip = False
    if len(theta_dot_history) >= 2:
        previous_previous = np.asarray(theta_dot_history[-2], dtype=float).reshape(-1)
        max_width = max(int(current.size), int(previous.size), int(previous_previous.size))
        current_pad = np.zeros(int(max_width), dtype=float)
        previous_pad = np.zeros(int(max_width), dtype=float)
        previous_previous_pad = np.zeros(int(max_width), dtype=float)
        current_pad[: int(current.size)] = current
        previous_pad[: int(previous.size)] = previous
        previous_previous_pad[: int(previous_previous.size)] = previous_previous
        acceleration = np.asarray(current_pad - previous_pad, dtype=float)
        previous_acceleration = np.asarray(previous_pad - previous_previous_pad, dtype=float)
        acceleration_l2 = float(np.linalg.norm(acceleration))
        previous_acceleration_l2 = float(np.linalg.norm(previous_acceleration))
        curvature_cosine = cosine_similarity(acceleration, previous_acceleration)
        curvature_sign_flip = bool(
            curvature_cosine is not None
            and float(curvature_cosine) <= float(cfg.motion_curvature_flip_cosine_threshold)
            and float(acceleration_l2) >= float(cfg.motion_acceleration_l2_threshold)
            and float(previous_acceleration_l2) >= float(cfg.motion_acceleration_l2_threshold)
        )
    calm = bool(
        direction_cosine is not None
        and float(direction_cosine) >= float(cfg.motion_calm_direction_cosine_threshold)
        and float(rate_change_ratio) <= float(cfg.motion_calm_rate_change_ratio_threshold)
        and not direction_reversal
        and not curvature_sign_flip
        and float(predicted_displacement) <= 0.05
    )
    kink_score = float(
        max(
            0.0,
            float(rate_change_ratio),
            0.0 if direction_cosine is None else float(max(0.0, -direction_cosine)),
            0.0 if curvature_cosine is None else float(max(0.0, -curvature_cosine)),
        )
    )
    large_change = float(rate_change_l2) >= float(cfg.motion_acceleration_l2_threshold)
    if bool(direction_reversal) or bool(curvature_sign_flip) or (
        bool(large_change)
        and float(rate_change_ratio) >= float(cfg.motion_kink_rate_change_ratio_threshold)
    ):
        regime = "kink"
    elif bool(calm):
        regime = "calm"
    else:
        regime = "steady"
    return MotionSchedulerTelemetry(
        regime=str(regime),
        direction_cosine=(None if direction_cosine is None else float(direction_cosine)),
        rate_change_l2=float(rate_change_l2),
        rate_change_ratio=float(rate_change_ratio),
        acceleration_l2=float(acceleration_l2),
        curvature_cosine=(None if curvature_cosine is None else float(curvature_cosine)),
        direction_reversal=bool(direction_reversal),
        curvature_sign_flip=bool(curvature_sign_flip),
        kink_score=float(kink_score),
    )


"Built Math: refresh = max(base_refresh, motion_floor(regime)) under low<medium<high ordering."
def effective_refresh_pressure(
    *,
    base_refresh_pressure: str,
    motion: MotionSchedulerTelemetry,
) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    base = str(base_refresh_pressure).strip().lower()
    motion_floor = (
        "high"
        if str(motion.regime) == "kink"
        else ("medium" if str(motion.regime) == "bootstrap" else "low")
    )
    return max((base, motion_floor), key=lambda item: int(order.get(str(item), 1)))


"Built Math: shortlist_motion = scale(shortlist_base, regime)."
def shortlist_cfg_for_motion(
    *,
    cfg: RealtimeCheckpointConfig,
    base_shortlist_cfg: FullScoreConfig,
    motion: MotionSchedulerTelemetry,
) -> FullScoreConfig:
    base_size = int(base_shortlist_cfg.shortlist_size)
    base_fraction = float(base_shortlist_cfg.shortlist_fraction)
    if str(motion.regime) == "calm":
        return FullScoreConfig(
            shortlist_fraction=float(
                max(0.05, base_fraction * float(cfg.motion_calm_shortlist_scale))
            ),
            shortlist_size=max(
                1,
                int(np.ceil(float(base_size) * float(cfg.motion_calm_shortlist_scale))),
            ),
        )
    if str(motion.regime) == "kink":
        return FullScoreConfig(
            shortlist_fraction=float(min(1.0, base_fraction * 1.5)),
            shortlist_size=max(1, int(base_size) + int(cfg.motion_kink_shortlist_bonus)),
        )
    return base_shortlist_cfg


"Built Math: confirm_limit = policy_motion(count, refresh, regime)."
def oracle_confirm_limit_for_motion(
    *,
    confirmed_count: int,
    refresh_pressure: str,
    motion: MotionSchedulerTelemetry,
) -> int:
    count = max(0, int(confirmed_count))
    if count <= 0:
        return 0
    if str(motion.regime) == "kink" or str(refresh_pressure) == "high":
        return int(count)
    if str(refresh_pressure) == "medium":
        return min(2, int(count))
    if str(motion.regime) == "calm":
        return min(1, int(count))
    return min(2, int(count))


"Built Math: oracle_budget = scale_motion(refresh, regime)."
def oracle_budget_scale_for_motion(
    *,
    cfg: RealtimeCheckpointConfig,
    refresh_pressure: str,
    motion: MotionSchedulerTelemetry,
) -> float:
    if str(motion.regime) == "kink" or str(refresh_pressure) == "high":
        return float(max(1.0, float(cfg.motion_kink_oracle_budget_scale)))
    if str(motion.regime) == "calm" and str(refresh_pressure) == "low":
        return float(max(0.25, float(cfg.motion_calm_oracle_budget_scale)))
    return 1.0
