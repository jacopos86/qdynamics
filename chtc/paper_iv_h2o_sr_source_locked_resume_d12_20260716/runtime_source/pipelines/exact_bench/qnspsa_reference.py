#!/usr/bin/env python3
"""Benchmark-local Qiskit QNSPSA reference harness.

This module is intentionally isolated to ``pipelines.exact_bench``.  It imports
``qiskit_algorithms`` lazily inside helper functions and is not a production
static-ADAPT dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

SCHEMA_VERSION = "qiskit_qnspsa_reference_v1"
QISKIT_BOUNDARY = "pipelines.exact_bench_only"
_DEPENDENCY_MESSAGE = "Qiskit QNSPSA reference support requires qiskit_algorithms.optimizers.QNSPSA."


class QiskitQNSPSAUnavailable(ImportError):
    """Raised when optional benchmark-only Qiskit QNSPSA support is unavailable."""


def import_qiskit_qnspsa_components() -> tuple[Any, Any]:
    """Import optional Qiskit QNSPSA components lazily."""
    try:
        from qiskit_algorithms.optimizers import QNSPSA
        from qiskit_algorithms.utils import algorithm_globals
    except Exception as exc:  # pragma: no cover - exact optional-dep failure varies
        raise QiskitQNSPSAUnavailable(_DEPENDENCY_MESSAGE) from exc
    return QNSPSA, algorithm_globals


def has_qiskit_qnspsa_support() -> bool:
    """Return whether optional benchmark-local Qiskit QNSPSA support is importable."""
    try:
        import_qiskit_qnspsa_components()
    except Exception:
        return False
    return True


def _schedule(*, scale: float, power: float, offset: float) -> Iterator[float]:
    k = 0
    while True:
        yield float(scale) / ((float(offset) + float(k) + 1.0) ** float(power))
        k += 1


def _write_payload(output_dir: Path | None, payload: dict[str, Any]) -> None:
    if output_dir is None:
        return
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qnspsa_reference.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _base_payload(*, seed: int, maxiter: int) -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "status": "not_started",
        "qiskit_boundary": QISKIT_BOUNDARY,
        "optimizer_class": "qiskit_algorithms.optimizers.QNSPSA",
        "seed": int(seed),
        "maxiter": int(maxiter),
    }


def run_qiskit_qnspsa_quadratic_reference(
    *,
    output_dir: Path | None = None,
    seed: int = 7,
    maxiter: int = 50,
) -> dict[str, Any]:
    """Run a tiny analytic-fidelity Qiskit QNSPSA sanity reference.

    The objective and fidelity are deterministic NumPy callables.  Exact target
    data is not consulted; this is only a benchmark-local optimizer API sanity
    check and optional-dependency probe.
    """
    payload = _base_payload(seed=int(seed), maxiter=int(maxiter))
    try:
        QNSPSA, algorithm_globals = import_qiskit_qnspsa_components()
    except QiskitQNSPSAUnavailable as exc:
        payload.update(
            {
                "status": "skipped_optional_dependency",
                "qiskit_available": False,
                "skip_reason": str(exc),
            }
        )
        _write_payload(output_dir, payload)
        return payload

    try:
        try:
            algorithm_globals.random_seed = int(seed)
        except Exception:
            pass
        x_star = np.array([0.25, -0.5], dtype=float)
        x0 = np.array([1.1, 0.9], dtype=float)

        def objective(x: np.ndarray) -> float:
            diff = np.asarray(x, dtype=float).reshape(-1) - x_star
            return float(np.dot(diff, diff))

        def fidelity(x: np.ndarray, y: np.ndarray) -> float:
            diff = np.asarray(x, dtype=float).reshape(-1) - np.asarray(y, dtype=float).reshape(-1)
            return float(np.exp(-float(np.dot(diff, diff))))

        initial_fun = float(objective(x0))
        optimizer = QNSPSA(
            fidelity,
            maxiter=int(maxiter),
            blocking=False,
            learning_rate=lambda: _schedule(scale=0.2, power=0.602, offset=10.0),
            perturbation=lambda: _schedule(scale=0.1, power=0.101, offset=0.0),
            resamplings=1,
            regularization=1e-3,
            initial_hessian=np.eye(int(x0.size), dtype=float),
        )
        result = optimizer.minimize(objective, x0=x0)
        final_fun = float(getattr(result, "fun", objective(np.asarray(getattr(result, "x", x0), dtype=float))))
        payload.update(
            {
                "status": "completed",
                "qiskit_available": True,
                "initial_fun": float(initial_fun),
                "final_fun": float(final_fun),
                "improved": bool(final_fun <= initial_fun),
                "x": [float(v) for v in np.asarray(getattr(result, "x", x0), dtype=float).reshape(-1).tolist()],
                "nfev": int(getattr(result, "nfev", 0) or 0),
                "nit": int(getattr(result, "nit", 0) or 0),
            }
        )
    except Exception as exc:  # pragma: no cover - optional Qiskit API variance
        payload.update(
            {
                "status": "failed",
                "qiskit_available": True,
                "exception_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    _write_payload(output_dir, payload)
    return payload


__all__ = [
    "QISKIT_BOUNDARY",
    "SCHEMA_VERSION",
    "QiskitQNSPSAUnavailable",
    "has_qiskit_qnspsa_support",
    "import_qiskit_qnspsa_components",
    "run_qiskit_qnspsa_quadratic_reference",
]
