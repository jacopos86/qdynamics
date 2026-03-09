from __future__ import annotations

import numpy as np

from pipelines.hardcoded.hh_continuation_pruning import (
    apply_pruning,
    post_prune_refit,
    rank_prune_candidates,
)


def test_rank_prune_candidates_prefers_small_theta() -> None:
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


def test_rank_prune_candidates_uses_proxy_benefit_as_tiebreak() -> None:
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


def test_post_prune_refit_returns_callback_result() -> None:
    theta = np.array([0.1, 0.2], dtype=float)
    theta_new, e = post_prune_refit(
        theta=theta,
        refit_fn=lambda x: (x * 0.5, float(np.sum((x * 0.5) ** 2))),
    )
    assert np.allclose(theta_new, [0.05, 0.1])
    assert e > 0.0
