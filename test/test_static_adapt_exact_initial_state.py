from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.static_adapt import adapt_pipeline
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


def _z_hamiltonian() -> PauliPolynomial:
    return PauliPolynomial("JW", [PauliTerm(1, ps="z", pc=1.0)])


def _resolution(*, state: np.ndarray | None, available: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        state=state,
        available=available,
        source="fixture_exact_state",
        comparison_space_label="unit_test_sector",
        skip_reason=None if available else "missing",
    )


def test_required_exact_initial_state_is_validated_against_hamiltonian(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adapt_pipeline,
        "resolve_exact_reference_state_for_problem",
        lambda *_args, **_kwargs: _resolution(
            state=np.array([0.0, 1.0], dtype=complex)
        ),
    )

    state = adapt_pipeline._resolve_required_exact_initial_state(
        _z_hamiltonian(),
        resolved_problem=SimpleNamespace(),
        exact_energy=-1.0,
    )

    assert np.allclose(state, [0.0, 1.0])


def test_required_exact_initial_state_fails_closed_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adapt_pipeline,
        "resolve_exact_reference_state_for_problem",
        lambda *_args, **_kwargs: _resolution(state=None, available=False),
    )

    with pytest.raises(ValueError, match="requested but is unavailable"):
        adapt_pipeline._resolve_required_exact_initial_state(
            _z_hamiltonian(),
            resolved_problem=SimpleNamespace(),
            exact_energy=-1.0,
        )


def test_required_exact_initial_state_rejects_reference_state_masquerade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adapt_pipeline,
        "resolve_exact_reference_state_for_problem",
        lambda *_args, **_kwargs: _resolution(
            state=np.array([1.0, 0.0], dtype=complex)
        ),
    )

    with pytest.raises(ValueError, match="does not match the exact target energy"):
        adapt_pipeline._resolve_required_exact_initial_state(
            _z_hamiltonian(),
            resolved_problem=SimpleNamespace(),
            exact_energy=-1.0,
        )
