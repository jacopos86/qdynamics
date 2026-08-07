from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.spsa_optimizer import spsa_minimize


def test_spsa_minimize_quadratic_converges() -> None:
    rng = np.random.default_rng(123)
    x_star = np.array([0.75, -1.25, 0.5, 1.0, -0.4], dtype=float)
    x0 = rng.normal(size=5)

    def quadratic(x: np.ndarray) -> float:
        diff = np.asarray(x, dtype=float) - x_star
        return float(np.dot(diff, diff))

    initial_fun = quadratic(x0)
    result = spsa_minimize(
        quadratic,
        x0,
        maxiter=300,
        seed=7,
        a=0.2,
        c=0.1,
    )

    assert result.success is True
    assert np.isfinite(result.fun)
    assert result.fun < initial_fun
    assert float(np.linalg.norm(result.x - x_star)) < 1e-1
    assert isinstance(result.optimizer_memory, dict)


def test_spsa_never_returns_worse_than_x0() -> None:
    """SPSA result.fun must be <= f(x0), i.e. the optimizer must seed its
    best-tracking with the baseline evaluation.

    Regression test for the x0-seeding bug: if best_y_observed was
    initialized to inf without evaluating f(x0), a short run on a hard
    landscape could return a point worse than the starting point.
    """
    # Construct a 'tilted basin' where the minimum is far from x0.
    # With very few iterations the SPSA walk is unlikely to reach the
    # minimum, but it must never report worse than f(x0).
    x0 = np.array([5.0, 5.0, 5.0], dtype=float)

    def hard_landscape(x: np.ndarray) -> float:
        # Global minimum at origin, but x0 is far away.
        return float(np.sum(x ** 2))

    f_x0 = hard_landscape(x0)  # 75.0

    for trial_seed in range(20):
        result = spsa_minimize(
            hard_landscape,
            x0,
            maxiter=5,  # intentionally very short
            seed=trial_seed,
            a=0.2,
            c=0.1,
            avg_last=0,
        )
        assert result.fun <= f_x0 + 1e-12, (
            f"SPSA returned fun={result.fun} > f(x0)={f_x0} "
            f"(seed={trial_seed}). x0-seeding is broken."
        )


def test_spsa_x0_seeding_counts_nfev() -> None:
    """The initial f(x0) evaluation from seeding must be counted in nfev."""
    x0 = np.array([1.0], dtype=float)
    call_count = [0]

    def counting_fn(x: np.ndarray) -> float:
        call_count[0] += 1
        return float(np.sum(x ** 2))

    result = spsa_minimize(
        counting_fn,
        x0,
        maxiter=3,
        seed=42,
        avg_last=0,
    )
    # 1 seeding eval + 3 iterations × 2 evals (plus, minus) = 7
    assert result.nfev == 7
    assert call_count[0] == 7


def test_spsa_optimizer_memory_refresh_points_recorded() -> None:
    x0 = np.array([1.0, -1.0], dtype=float)

    def quad(x: np.ndarray) -> float:
        return float(np.dot(x, x))

    result = spsa_minimize(
        quad,
        x0,
        maxiter=4,
        seed=11,
        memory={
            "available": True,
            "preconditioner_diag": [1.0, 1.0],
            "grad_sq_ema": [0.1, 0.1],
            "refresh_points": [],
            "remap_events": [],
        },
        refresh_every=2,
        precondition_mode="diag_rms_grad",
    )
    assert result.optimizer_memory is not None
    assert result.optimizer_memory["refresh_points"] == [2, 4]
