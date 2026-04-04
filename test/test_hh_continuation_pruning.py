from __future__ import annotations

import pytest
import numpy as np

from pipelines.hardcoded.hh_continuation_pruning import (
    apply_pruning,
    cheap_prune_score,
    post_prune_refit,
    rank_prune_candidates,
)


def test_rank_prune_candidates_prefers_small_theta_without_metadata() -> None:
    theta = np.array([0.8, 0.01, 0.02, 0.5], dtype=float)
    labels = ["a", "b", "c", "d"]
    idx = rank_prune_candidates(
        theta=theta,
        labels=labels,
        marginal_proxy_benefit=None,
        max_candidates=3,
        min_candidates=2,
        fraction_candidates=0.5,
    )
    assert idx[0] == 1
    assert 2 in idx


def test_rank_prune_candidates_uses_proxy_benefit_as_tiebreak_without_metadata() -> None:
    theta = np.array([0.01, 0.01, 0.5], dtype=float)
    labels = ["a", "b", "c"]
    idx = rank_prune_candidates(
        theta=theta,
        labels=labels,
        marginal_proxy_benefit=[0.3, 0.1, 1.0],
        max_candidates=2,
        min_candidates=2,
        fraction_candidates=0.5,
    )
    assert idx == [1, 0]


def test_rank_prune_candidates_metadata_path_filters_and_prefers_small_angles() -> None:
    theta = np.array([0.004, 0.03, 0.001, 0.002], dtype=float)
    labels = ["a", "b", "c", "d"]
    idx = rank_prune_candidates(
        theta=theta,
        labels=labels,
        marginal_proxy_benefit=[0.3, 0.2, 0.1, 0.4],
        max_candidates=4,
        min_candidates=1,
        fraction_candidates=0.5,
        selector_burden=[0.0, 0.0, 0.0, 0.0],
        admission_steps=[1, 4, 1, 1],
        first_seen_steps=[1, 1, 4, 1],
        cooldown_remaining=[0, 0, 0, 2],
        stagnation_scores=[0.8, 0.8, 0.8, 0.8],
        current_step=5,
        protect_steps=2,
        stale_age=2,
        stagnation_threshold=0.5,
        small_theta_abs=0.0,
        small_theta_relative=1.0,
    )
    assert idx == [0]


def test_cheap_prune_score_divides_frozen_regression_by_selector_burden() -> None:
    assert cheap_prune_score(frozen_regression=0.3, selector_burden=2.0) == pytest.approx(0.1)
    assert cheap_prune_score(frozen_regression=-1.0, selector_burden=2.0) == pytest.approx(0.0)


def test_apply_pruning_accepts_when_regression_small() -> None:
    theta = np.array([0.2, 0.1, 0.05], dtype=float)
    labels = ["a", "b", "c"]

    def _eval(idx_remove: int, theta_cur: np.ndarray) -> tuple[float, np.ndarray]:
        theta_new = np.delete(theta_cur, idx_remove)
        return float(np.sum(theta_new**2)), theta_new

    theta_out, labels_out, decisions, energy_out = apply_pruning(
        theta=theta,
        labels=labels,
        candidate_indices=[2],
        eval_with_removal=_eval,
        energy_before=float(np.sum(theta**2)),
        max_regression=1e-3,
    )
    assert len(theta_out) == 2
    assert labels_out == ["a", "b"]
    assert decisions[0].accepted is True
    assert energy_out <= float(np.sum(theta**2))


def test_apply_pruning_rejects_when_retained_gain_fails() -> None:
    theta = np.array([0.2, 0.1, 0.05], dtype=float)
    labels = ["a", "b", "c"]

    def _eval(
        idx_remove: int,
        theta_cur: np.ndarray,
        labels_cur: list[str],
    ) -> tuple[float, np.ndarray]:
        theta_new = np.delete(theta_cur, idx_remove)
        return 0.95, theta_new

    theta_out, labels_out, decisions, energy_out = apply_pruning(
        theta=theta,
        labels=labels,
        candidate_indices=[2],
        eval_with_removal=_eval,
        energy_before=0.80,
        max_regression=0.20,
        retained_reference_energy=1.00,
        admitted_gain=0.30,
        retained_gain_ratio=0.75,
    )
    assert np.allclose(theta_out, theta)
    assert labels_out == labels
    assert decisions[0].accepted is False
    assert decisions[0].reason == "retained_gain_below_ratio"
    assert decisions[0].safe_regression_ok is True
    assert decisions[0].retained_gain_ok is False
    assert decisions[0].retained_gain == pytest.approx(0.05)
    assert decisions[0].retained_gain_threshold == pytest.approx(0.225)
    assert energy_out == 0.80


def test_post_prune_refit_returns_callback_result() -> None:
    theta = np.array([0.1, 0.2], dtype=float)
    theta_new, e = post_prune_refit(
        theta=theta,
        refit_fn=lambda x: (x * 0.5, float(np.sum((x * 0.5) ** 2))),
    )
    assert np.allclose(theta_new, [0.05, 0.1])
    assert e > 0.0
