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
