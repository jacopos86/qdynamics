from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.static_adapt import adapt_pipeline as hardcoded_adapt
from pipelines.static_adapt import adapt_pipeline
from pipelines.static_adapt import prune_derivatives


def test_prune_derivative_helper_remains_available_through_wrappers() -> None:
    assert (
        adapt_pipeline._propagate_runtime_prune_derivatives
        is prune_derivatives._propagate_runtime_prune_derivatives
    )
    assert (
        hardcoded_adapt._propagate_runtime_prune_derivatives
        is prune_derivatives._propagate_runtime_prune_derivatives
    )


def test_propagate_runtime_prune_derivatives_empty_runtime_returns_zero_derivatives() -> None:
    executor = SimpleNamespace(runtime_parameter_count=2, _plans=[])
    psi_ref = np.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex)

    psi, dpsi, d2psi = prune_derivatives._propagate_runtime_prune_derivatives(
        executor=executor,
        theta=np.asarray([0.1, -0.2], dtype=float),
        psi_ref_state=psi_ref,
        active_indices=[1, 0],
    )

    np.testing.assert_allclose(psi, psi_ref)
    assert len(dpsi) == 2
    assert len(d2psi) == 2
    assert all(len(row) == 2 for row in d2psi)
    for vec in dpsi:
        np.testing.assert_allclose(vec, np.zeros_like(psi_ref))
    for row in d2psi:
        for vec in row:
            np.testing.assert_allclose(vec, np.zeros_like(psi_ref))


def test_propagate_runtime_prune_derivatives_rejects_theta_runtime_mismatch() -> None:
    executor = SimpleNamespace(runtime_parameter_count=3, _plans=[])

    with pytest.raises(ValueError, match="runtime theta length mismatch"):
        prune_derivatives._propagate_runtime_prune_derivatives(
            executor=executor,
            theta=np.asarray([0.1, -0.2], dtype=float),
            psi_ref_state=np.asarray([1.0 + 0.0j], dtype=complex),
            active_indices=[],
        )
