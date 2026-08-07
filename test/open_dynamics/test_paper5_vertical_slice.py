from __future__ import annotations

import numpy as np

from paper5.stability.matrix_reference import (
    CLOSED_SCALAR_STATE_NAMES,
    closed_scalar_to_matrix_state,
)
from pipelines.open_dynamics import (
    ANOMALOUS_PHONON_KEY,
    COHERENT_PHONON_KEY,
    CONNECTED_EPH_KEY,
    NORMAL_PHONON_KEY,
    arrays_from_matrix_states,
    from_paper5_dimer,
    require_comparable,
    validate_trajectory,
)


def _matrix_states():
    first = np.linspace(-0.1, 0.1, len(CLOSED_SCALAR_STATE_NAMES))
    second = first + np.linspace(0.01, 0.02, first.size)
    return (
        closed_scalar_to_matrix_state(first),
        closed_scalar_to_matrix_state(second),
    )


def _provenance():
    return {
        "paper_stated": {"source.gamma": 0.5},
        "repository_choices": {"numerics.step": 0.1},
        "citations": ("https://arxiv.org/html/2606.22233v1",),
    }


def test_full_adapter_matches_legacy_matrix_fields_field_by_field() -> None:
    states = _matrix_states()
    arrays = arrays_from_matrix_states(
        time=np.array([0.0, 0.1]),
        matrix_states=states,
        include_fluctuation_moments=True,
    )
    trajectory = from_paper5_dimer(
        arrays,
        level="equal_time_full",
        **_provenance(),
    )

    np.testing.assert_array_equal(
        trajectory.electron_1rdm,
        np.asarray([state.electron_density for state in states]),
    )
    np.testing.assert_array_equal(
        trajectory.moments[NORMAL_PHONON_KEY].values,
        np.asarray([state.phonon_density for state in states]),
    )
    np.testing.assert_array_equal(
        trajectory.moments[ANOMALOUS_PHONON_KEY].values,
        np.asarray([state.anomalous_phonon_density for state in states]),
    )
    np.testing.assert_array_equal(
        trajectory.moments[CONNECTED_EPH_KEY].values,
        np.asarray([state.electron_phonon_correlation for state in states]),
    )


def test_coherent_adapter_omits_unavailable_fields() -> None:
    arrays = arrays_from_matrix_states(
        time=np.array([0.0, 0.1]),
        matrix_states=_matrix_states(),
        include_fluctuation_moments=False,
    )
    trajectory = from_paper5_dimer(
        arrays,
        level="coherent_only",
        **_provenance(),
    )

    assert set(trajectory.moments) == {COHERENT_PHONON_KEY}
    assert trajectory.method.reference_access == "none"


def test_exact_adapter_is_truncated_offline_and_exposes_no_wavefunction() -> None:
    arrays = arrays_from_matrix_states(
        time=np.array([0.0, 0.1]),
        matrix_states=_matrix_states(),
        include_fluctuation_moments=True,
    )
    trajectory = from_paper5_dimer(
        arrays,
        level="exact",
        **_provenance(),
    )

    assert trajectory.method.family == "exact_reference"
    assert trajectory.method.reference_access == "offline_only"
    assert "cutoff" in trajectory.method.approximation
    for forbidden in ("state_vector", "state_derivative", "hamiltonian", "model"):
        assert not hasattr(trajectory, forbidden)


def test_three_level_vertical_slice_is_structurally_valid_and_comparable(
    paper5_vertical_slice,
) -> None:
    result = paper5_vertical_slice
    require_comparable(*result.trajectories)

    for trajectory in result.trajectories:
        assert validate_trajectory(trajectory).structure_issues == ()
    assert result.exact.method.reference_access == "offline_only"
    assert result.coherent_only.method.reference_access == "none"
    assert result.equal_time_full.method.reference_access == "none"
    assert result.full_coordinate_error.shape == (
        result.exact.time.size,
        len(CLOSED_SCALAR_STATE_NAMES),
    )
    assert result.full_block_error_norms.shape == (result.exact.time.size, 5)
    np.testing.assert_allclose(
        result.exact.electron_1rdm[0],
        result.equal_time_full.electron_1rdm[0],
        atol=1e-12,
    )
