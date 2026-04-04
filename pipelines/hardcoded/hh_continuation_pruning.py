#!/usr/bin/env python3
"""Mature-scaffold pruning helpers for HH continuation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import numpy as np

from pipelines.hardcoded.hh_continuation_types import PruneDecision


@dataclass(frozen=True)
class PruneConfig:
    max_candidates: int = 6
    min_candidates: int = 2
    fraction_candidates: float = 0.25
    max_regression: float = 1e-8
    retained_gain_ratio: float = 0.5
    protect_steps: int = 2
    stale_age: int = 2
    stagnation_threshold: float = 0.0
    small_theta_abs: float = 1e-3
    small_theta_relative: float = 0.5
    cooldown_steps: int = 2
    local_window_size: int = 4
    old_fraction: float = 0.25


def _wrap_to_pi(value: float) -> float:
    return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


def cheap_prune_score(*, frozen_regression: float, selector_burden: float) -> float:
    return float(max(0.0, float(frozen_regression)) / (1.0 + max(0.0, float(selector_burden))))


def rank_prune_candidates(
    *,
    theta: np.ndarray,
    labels: list[str],
    marginal_proxy_benefit: list[float] | None,
    max_candidates: int,
    min_candidates: int,
    fraction_candidates: float,
    selector_burden: Sequence[float] | None = None,
    admission_steps: Sequence[int] | None = None,
    first_seen_steps: Sequence[int] | None = None,
    cooldown_remaining: Sequence[int] | None = None,
    stagnation_scores: Sequence[float] | None = None,
    current_step: int | None = None,
    protect_steps: int = 0,
    stale_age: int = 0,
    stagnation_threshold: float = 0.0,
    small_theta_abs: float = 0.0,
    small_theta_relative: float = 1.0,
) -> list[int]:
    n = int(theta.size)
    if n <= 0:
        return []
    benefits = list(marginal_proxy_benefit) if marginal_proxy_benefit is not None else []

    def _benefit_key(i: int) -> float:
        if i >= len(benefits):
            return float("inf")
        val = float(benefits[i])
        if not np.isfinite(val):
            return float("inf")
        return float(val)

    wrapped = np.asarray([_wrap_to_pi(float(x)) for x in np.asarray(theta, dtype=float).tolist()], dtype=float)
    target = int(np.ceil(float(fraction_candidates) * float(n)))
    target = max(int(min_candidates), target)
    target = min(int(max_candidates), target, n)

    metadata_ready = all(
        seq is not None
        for seq in (
            admission_steps,
            first_seen_steps,
            cooldown_remaining,
            stagnation_scores,
        )
    ) and current_step is not None
    if not metadata_ready:
        order = sorted(
            range(n),
            key=lambda i: (
                abs(float(wrapped[i])),
                _benefit_key(int(i)),
                str(labels[i]),
            ),
        )
        return [int(i) for i in order[:target]]

    admission_steps_list = [int(x) for x in list(admission_steps or [])[:n]]
    first_seen_steps_list = [int(x) for x in list(first_seen_steps or [])[:n]]
    cooldown_list = [int(x) for x in list(cooldown_remaining or [])[:n]]
    stagnation_list = [float(x) for x in list(stagnation_scores or [])[:n]]
    while len(admission_steps_list) < n:
        admission_steps_list.append(0)
    while len(first_seen_steps_list) < n:
        first_seen_steps_list.append(0)
    while len(cooldown_list) < n:
        cooldown_list.append(0)
    while len(stagnation_list) < n:
        stagnation_list.append(0.0)

    eligible = [
        int(i)
        for i in range(n)
        if int(current_step) - int(first_seen_steps_list[i]) >= int(stale_age)
        and float(stagnation_list[i]) >= float(stagnation_threshold)
        and int(current_step) - int(admission_steps_list[i]) >= int(protect_steps)
        and int(cooldown_list[i]) <= 0
    ]
    if not eligible:
        return []

    theta_small = float(
        max(
            float(max(0.0, small_theta_abs)),
            float(max(0.0, small_theta_relative))
            * float(np.median(np.abs(wrapped[np.asarray(eligible, dtype=int)]))),
        )
    )
    angle_pool = [int(i) for i in eligible if abs(float(wrapped[i])) <= float(theta_small) + 1e-15]
    if not angle_pool:
        return []

    burden_list = [float(x) for x in list(selector_burden or [])[:n]]
    while len(burden_list) < n:
        burden_list.append(0.0)
    target = min(int(max_candidates), len(angle_pool))
    order = sorted(
        angle_pool,
        key=lambda i: (
            abs(float(wrapped[i])),
            _benefit_key(int(i)),
            float(max(0.0, burden_list[i])),
            str(labels[i]),
        ),
    )
    return [int(i) for i in order[:target]]


def apply_pruning(
    *,
    theta: np.ndarray,
    labels: list[str],
    candidate_indices: list[int],
    eval_with_removal: Callable[..., tuple[float, np.ndarray]],
    energy_before: float,
    max_regression: float,
    retained_reference_energy: float | None = None,
    admitted_gain: float | None = None,
    retained_gain_ratio: float = 0.0,
) -> tuple[np.ndarray, list[str], list[PruneDecision], float]:
    cur_theta = np.asarray(theta, dtype=float).copy()
    cur_labels = list(labels)
    decisions: list[PruneDecision] = []
    cur_energy = float(energy_before)
    removed_so_far = 0

    for idx0 in candidate_indices:
        idx = int(idx0) - int(removed_so_far)
        if idx < 0 or idx >= len(cur_labels):
            continue
        try:
            trial_energy, trial_theta = eval_with_removal(idx, cur_theta, list(cur_labels))
        except TypeError:
            trial_energy, trial_theta = eval_with_removal(idx, cur_theta)
        regression = float(trial_energy - cur_energy)
        retained_gain = (
            None
            if retained_reference_energy is None
            else float(retained_reference_energy) - float(trial_energy)
        )
        retained_gain_threshold = (
            None
            if admitted_gain is None
            else float(retained_gain_ratio) * float(max(0.0, admitted_gain))
        )
        safe_regression_ok = bool(regression <= float(max_regression))
        retained_ok = True
        if retained_gain is not None and retained_gain_threshold is not None:
            retained_ok = bool(float(retained_gain) >= float(retained_gain_threshold))
        accepted = bool(safe_regression_ok) and bool(retained_ok)
        if accepted:
            reason = "accepted"
        elif (not bool(safe_regression_ok)) and (not bool(retained_ok)):
            reason = "safe_regression_and_retained_gain_failed"
        elif not bool(safe_regression_ok):
            reason = "safe_regression_exceeded"
        else:
            reason = "retained_gain_below_ratio"
        label = str(cur_labels[idx])
        decisions.append(
            PruneDecision(
                index=int(idx),
                label=label,
                accepted=bool(accepted),
                energy_before=float(cur_energy),
                energy_after=float(trial_energy),
                regression=float(regression),
                reason=str(reason),
                safe_regression_ok=bool(safe_regression_ok),
                retained_gain_ok=bool(retained_ok),
                regression_threshold=float(max_regression),
                retained_gain=(None if retained_gain is None else float(retained_gain)),
                retained_gain_threshold=(
                    None if retained_gain_threshold is None else float(retained_gain_threshold)
                ),
            )
        )
        if accepted:
            cur_theta = np.asarray(trial_theta, dtype=float).copy()
            del cur_labels[idx]
            cur_energy = float(trial_energy)
            removed_so_far += 1

    return cur_theta, cur_labels, decisions, float(cur_energy)


def post_prune_refit(
    *,
    theta: np.ndarray,
    refit_fn: Callable[[np.ndarray], tuple[np.ndarray, float]],
) -> tuple[np.ndarray, float]:
    return refit_fn(np.asarray(theta, dtype=float))
