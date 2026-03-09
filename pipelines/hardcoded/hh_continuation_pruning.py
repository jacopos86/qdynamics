#!/usr/bin/env python3
"""Prune-before-replay helpers for HH continuation Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from pipelines.hardcoded.hh_continuation_types import PruneDecision


@dataclass(frozen=True)
class PruneConfig:
    max_candidates: int = 6
    min_candidates: int = 2
    fraction_candidates: float = 0.25
    max_regression: float = 1e-8


def rank_prune_candidates(
    *,
    theta: np.ndarray,
    labels: list[str],
    marginal_proxy_benefit: list[float] | None,
    max_candidates: int,
    min_candidates: int,
    fraction_candidates: float,
) -> list[int]:
    n = int(theta.size)
    if n <= 0:
        return []
    target = int(np.ceil(float(fraction_candidates) * float(n)))
    target = max(int(min_candidates), target)
    target = min(int(max_candidates), target, n)
    benefits = list(marginal_proxy_benefit) if marginal_proxy_benefit is not None else []

    def _benefit_key(i: int) -> float:
        if i >= len(benefits):
            return float("inf")
        val = float(benefits[i])
        if not np.isfinite(val):
            return float("inf")
        return float(val)

    order = sorted(
        range(n),
        key=lambda i: (
            abs(float(theta[i])),
            _benefit_key(int(i)),
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
        accepted = bool(regression <= float(max_regression))
        reason = "accepted" if accepted else "regression_exceeded"
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
