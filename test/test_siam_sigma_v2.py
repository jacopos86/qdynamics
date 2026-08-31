"""Versioned offline-reporting correction for the SIAM screening sum rule."""

from __future__ import annotations

import numpy as np
import pytest

from pipelines.time_dynamics.diagnostics.siam_exact_reference_store import (
    SIAM_EXACT_PREFIX_VIEW_V2,
    SIAM_EXACT_REFERENCE_STORE_V1,
    SIAM_EXACT_REFERENCE_STORE_V2,
    SIAMExactReferenceStore,
)
from pipelines.time_dynamics.siam_screening import SIAMScreeningConfig
from pipelines.time_dynamics.siam_seed import write_siam_seed_manifest
from src.quantum.siam_model import SIAMLayout, project_to_sector
from src.quantum.siam_observables import (
    SIAM_OBSERVABLES_V1,
    SIAM_OBSERVABLES_V2,
    impurity_double_occupancy_operator,
    impurity_number_operator,
    impurity_spin_square_operator,
    integrated_screening,
    screening_observable_operators,
    screening_observable_operators_v1,
)
from src.quantum.siam_reference import SIAM_REFERENCE_V2, exact_reference
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix


def _sector_matrix(operator, n_bath: int = 2) -> np.ndarray:
    return project_to_sector(
        np.asarray(hamiltonian_matrix(operator), dtype=complex), n_bath
    )


def test_sigma_v2_observable_contract_includes_impurity_self_spin() -> None:
    layout = SIAMLayout(2)
    v1 = screening_observable_operators_v1(layout, v=1.0 / np.sqrt(2.0))
    v2 = screening_observable_operators(layout, v=1.0 / np.sqrt(2.0))

    assert SIAM_OBSERVABLES_V1 != SIAM_OBSERVABLES_V2
    assert "N_d" not in v1 and "S_d2" not in v1
    assert set(v2) == {
        "m_d",
        "D_d",
        "N_d",
        "S_d2",
        "C_1",
        "C_2",
        "Sigma_1",
        "Sigma_2",
        "I_d",
    }
    n_d = _sector_matrix(impurity_number_operator(layout))
    d_d = _sector_matrix(impurity_double_occupancy_operator(layout))
    s_d2 = _sector_matrix(impurity_spin_square_operator(layout))
    assert np.allclose(s_d2, 0.75 * (n_d - 2.0 * d_d), atol=1.0e-12)
    assert np.allclose(
        _sector_matrix(v2["Sigma_2"]),
        _sector_matrix(v2["S_d2"] + v2["C_1"] + v2["C_2"]),
        atol=1.0e-12,
    )
    assert integrated_screening(
        {"C_1": -0.2, "C_2": -0.1},
        2,
        impurity_spin_square=0.6,
    ) == pytest.approx(0.3)


def test_exact_reference_is_explicitly_sigma_v2() -> None:
    reference = exact_reference(2, u=1.5, times=(0.0, 0.2))

    assert reference.schema == SIAM_REFERENCE_V2
    assert set(reference.observables) >= {"N_d", "S_d2", "Sigma_1", "Sigma_2"}
    n_d = np.asarray(reference.at("N_d"))
    d_d = np.asarray(reference.at("D_d"))
    s_d2 = np.asarray(reference.at("S_d2"))
    assert np.allclose(s_d2, 0.75 * (n_d - 2.0 * d_d), atol=1.0e-12)
    assert np.allclose(
        reference.at("Sigma_2"),
        s_d2 + np.asarray(reference.at("C_1")) + np.asarray(reference.at("C_2")),
        atol=1.0e-12,
    )


def test_exact_store_content_identity_and_views_are_sigma_v2(tmp_path) -> None:
    seed_path = tmp_path / "seed.json"
    write_siam_seed_manifest(
        seed_path,
        SIAMScreeningConfig(n_bath=2, t_final=0.2, num_times=3),
    )
    store = SIAMExactReferenceStore(tmp_path / "exact")
    artifact = store.get_or_build(seed_path=seed_path, times=(0.0, 0.1, 0.2))

    assert SIAM_EXACT_REFERENCE_STORE_V1 != SIAM_EXACT_REFERENCE_STORE_V2
    assert artifact.manifest["schema"] == SIAM_EXACT_REFERENCE_STORE_V2
    assert artifact.manifest["observable_schema"] == SIAM_OBSERVABLES_V2
    assert artifact.manifest["reference_schema"] == SIAM_REFERENCE_V2
    n_d = artifact.observable("N_d")
    d_d = artifact.observable("D_d")
    s_d2 = artifact.observable("S_d2")
    assert np.allclose(s_d2, 0.75 * (n_d - 2.0 * d_d), atol=1.0e-12)
    assert np.allclose(
        artifact.observable("Sigma_2"),
        s_d2 + artifact.observable("C_1") + artifact.observable("C_2"),
        atol=1.0e-12,
    )
    prefix = store.prefix_view(artifact, times=(0.0, 0.1))
    assert prefix.receipt["schema"] == SIAM_EXACT_PREFIX_VIEW_V2
    assert prefix.receipt["source"]["observable_schema"] == SIAM_OBSERVABLES_V2
