from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.excited_dynamics.paper_iii_advisor_demo import (
    _half_filled_sector_indices,
    _midpoint_step,
)
from pipelines.scaffold.qse_root_refit import _validate_qse_manifest


def test_half_filled_binary_sector_dimension_for_hh_dimer() -> None:
    indices = _half_filled_sector_indices(
        num_sites=2,
        n_ph_max=3,
        boson_encoding="binary",
        ordering="blocked",
        nq_total=8,
    )

    # C(2,1)^2 electronic states times four phonon levels at each site.
    assert indices.size == 4 * 4 * 4
    assert len(set(int(index) for index in indices)) == int(indices.size)


def test_midpoint_step_is_unitary_and_matches_two_level_solution() -> None:
    hamiltonian = np.array([[0.0, 0.3], [0.3, 1.0]], dtype=complex)
    initial = np.array([1.0, 0.0], dtype=complex)

    evolved = _midpoint_step(initial, hamiltonian, 0.2)
    energies, vectors = np.linalg.eigh(hamiltonian)
    expected = vectors @ (np.exp(-1.0j * 0.2 * energies) * (vectors.conj().T @ initial))

    np.testing.assert_allclose(np.linalg.norm(evolved), 1.0, atol=1.0e-13)
    np.testing.assert_allclose(evolved, expected, atol=1.0e-13)


def test_qse_refit_validation_treats_q0_root_zero_as_excited_candidate() -> None:
    policy = {
        "reference_projection": "q0",
        "basis_vector_normalization": "raw_projected",
        "sector_projection": "identity",
        "sector_label": "unit_test_sector",
    }
    payload = {
        "schema_version": "qse_spectra_v1",
        "pipeline": "qse_spectra",
        "backend": "ideal_statevector",
        "uses_qiskit": False,
        "settings": {"basis_vector_policy": dict(policy)},
        "diagnostics": {
            "num_qubits": 1,
            "hilbert_dim": 2,
            "basis_size": 1,
            "retained_rank": 1,
            "basis_vector_policy": dict(policy),
        },
        "operator_basis": [{"basis_index": 0}],
        "eigenvalues": [
            {
                "state_index": 0,
                "energy": 1.0,
                "generalized_residual_norm": 0.0,
                "basis_coefficients": [
                    {"basis_index": 0, "re": 1.0, "im": 0.0}
                ],
            }
        ],
    }

    nq, basis_size, selected = _validate_qse_manifest(
        payload,
        state_index=0,
        allow_ground_state=False,
    )

    assert nq == 1
    assert basis_size == 1
    assert selected["state_index"] == 0

    payload["settings"]["basis_vector_policy"]["reference_projection"] = "none"
    payload["diagnostics"]["basis_vector_policy"]["reference_projection"] = "none"
    with pytest.raises(ValueError, match="ground Ritz state"):
        _validate_qse_manifest(payload, state_index=0, allow_ground_state=False)
