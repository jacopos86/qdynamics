"""N_b = 2 proof fixture for the finite-box SIAM screening application.

Six qubits, sector dimension 9. Every check here is cheap enough to run on
every commit, and each one pins a convention that a later change could silently
break: fermionic signs, the symmetry sector, the spin-correlation definition,
and the equivalence of the fermionic and Jordan-Wigner-encoded operators.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.quantum.siam_model import (
    DEFAULT_HOPPING,
    DOWN,
    QUENCHED_HYBRIDIZATION,
    SIAMLayout,
    UP,
    bath_eigenmodes,
    build_initial_state,
    build_siam_hamiltonian,
    embed_sector_vector,
    number_operator,
    project_to_sector,
    sector_indices,
    sector_vector,
    total_number_operator,
)
from src.quantum.siam_observables import (
    impurity_double_occupancy_operator,
    impurity_magnetization_operator,
    screening_observable_operators,
    spin_correlation_operator,
    spin_lower_operator,
    spin_raise_operator,
    spin_z_operator,
)
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix

N_BATH = 2
U_FIXTURE = 1.5


def _dense(poly) -> np.ndarray:
    return np.asarray(hamiltonian_matrix(poly), dtype=complex)


@pytest.fixture(scope="module")
def layout() -> SIAMLayout:
    return SIAMLayout(N_BATH)


@pytest.fixture(scope="module")
def hamiltonian() -> np.ndarray:
    return _dense(
        build_siam_hamiltonian(N_BATH, u=U_FIXTURE, v=QUENCHED_HYBRIDIZATION)
    )


# --- register and sector -------------------------------------------------


def test_register_size_and_sector_dimension(layout: SIAMLayout) -> None:
    assert layout.n_qubits == 6
    assert layout.electron_counts() == (2, 1)
    assert len(sector_indices(N_BATH)) == 9


def test_spin_blocked_layout_places_impurities_at_block_heads(layout) -> None:
    assert layout.impurity(UP) == 0
    assert layout.bath(1, UP) == 1
    assert layout.bath(2, UP) == 2
    assert layout.impurity(DOWN) == 3
    assert layout.bath(2, DOWN) == 5


def test_odd_or_zero_bath_is_rejected() -> None:
    for bad in (0, 1, 3, -2):
        with pytest.raises(ValueError, match="closed shell"):
            SIAMLayout(bad)


def test_bath_site_index_is_one_based_and_bounded(layout) -> None:
    with pytest.raises(ValueError, match="1..2"):
        layout.bath(0, UP)
    with pytest.raises(ValueError, match="1..2"):
        layout.bath(3, UP)


# --- Hamiltonian ---------------------------------------------------------


def test_hamiltonian_is_hermitian(hamiltonian: np.ndarray) -> None:
    assert np.allclose(hamiltonian, hamiltonian.conj().T, atol=1e-12)


@pytest.mark.parametrize("spin", [UP, DOWN])
def test_hamiltonian_conserves_each_spin_species(hamiltonian, layout, spin) -> None:
    number = _dense(total_number_operator(layout, spin))
    assert np.allclose(hamiltonian @ number - number @ hamiltonian, 0.0, atol=1e-10)


def test_hamiltonian_conserves_total_sz(hamiltonian, layout) -> None:
    sz = 0.5 * (
        _dense(total_number_operator(layout, UP))
        - _dense(total_number_operator(layout, DOWN))
    )
    assert np.allclose(hamiltonian @ sz - sz @ hamiltonian, 0.0, atol=1e-10)


def test_pre_quench_hamiltonian_drops_only_the_hybridization(layout) -> None:
    quenched = _dense(
        build_siam_hamiltonian(N_BATH, u=U_FIXTURE, v=QUENCHED_HYBRIDIZATION)
    )
    decoupled = _dense(build_siam_hamiltonian(N_BATH, u=U_FIXTURE, v=0.0))
    from src.quantum.siam_model import build_hybridization

    hybridization = _dense(
        build_hybridization(layout, v=QUENCHED_HYBRIDIZATION)
    )
    assert np.allclose(quenched - decoupled, hybridization, atol=1e-12)


def test_noninteracting_limit_is_one_body(layout) -> None:
    """At U=0 the Hamiltonian is quadratic, so its sector spectrum is a sum of
    single-particle energies drawn from the impurity-plus-bath chain."""

    h_sector = project_to_sector(
        _dense(build_siam_hamiltonian(N_BATH, u=0.0, v=QUENCHED_HYBRIDIZATION)),
        N_BATH,
    )
    size = N_BATH + 1
    single = np.zeros((size, size))
    single[0, 1] = single[1, 0] = -QUENCHED_HYBRIDIZATION
    for j in range(1, size - 1):
        single[j, j + 1] = single[j + 1, j] = -DEFAULT_HOPPING
    eps = np.linalg.eigvalsh(single)
    n_up, n_dn = SIAMLayout(N_BATH).electron_counts()
    from itertools import combinations

    expected = sorted(
        sum(eps[list(a)]) + sum(eps[list(b)])
        for a in combinations(range(size), n_up)
        for b in combinations(range(size), n_dn)
    )
    assert np.allclose(np.linalg.eigvalsh(h_sector), expected, atol=1e-10)


# --- prepared state ------------------------------------------------------


def test_initial_state_is_normalized_and_inside_the_sector() -> None:
    psi = build_initial_state(N_BATH)
    assert float(np.vdot(psi, psi).real) == pytest.approx(1.0, abs=1e-12)
    inside = float(np.linalg.norm(sector_vector(psi, N_BATH)) ** 2)
    assert inside == pytest.approx(1.0, abs=1e-12)


def test_initial_occupations_match_the_written_state(layout) -> None:
    """One spin-up impurity electron, a half-filled bath, nothing else."""

    psi = build_initial_state(N_BATH)
    expect = float(np.real(np.vdot(psi, _dense(
        number_operator(layout, layout.impurity(UP))) @ psi)))
    assert expect == pytest.approx(1.0, abs=1e-12)
    down = float(np.real(np.vdot(psi, _dense(
        number_operator(layout, layout.impurity(DOWN))) @ psi)))
    assert down == pytest.approx(0.0, abs=1e-12)
    for spin, count in ((UP, N_BATH // 2), (DOWN, N_BATH // 2)):
        total = sum(
            float(np.real(np.vdot(psi, _dense(
                number_operator(layout, layout.bath(site, spin))) @ psi)))
            for site in range(1, N_BATH + 1)
        )
        assert total == pytest.approx(count, abs=1e-10)


def test_bath_eigenmodes_match_the_open_chain_closed_form() -> None:
    energies, _vectors = bath_eigenmodes(N_BATH)
    expected = sorted(
        -2.0 * DEFAULT_HOPPING * np.cos(np.pi * m / (N_BATH + 1))
        for m in range(1, N_BATH + 1)
    )
    assert np.allclose(energies, expected, atol=1e-12)


def test_sector_round_trip_is_lossless() -> None:
    psi = build_initial_state(N_BATH)
    assert np.allclose(embed_sector_vector(sector_vector(psi, N_BATH), N_BATH), psi)


# --- spin algebra and observables ----------------------------------------


def test_spin_operators_satisfy_su2_on_the_impurity(layout) -> None:
    sz = _dense(spin_z_operator(layout, "d"))
    sp = _dense(spin_raise_operator(layout, "d"))
    sm = _dense(spin_lower_operator(layout, "d"))
    assert np.allclose(sz @ sp - sp @ sz, sp, atol=1e-12)
    assert np.allclose(sz @ sm - sm @ sz, -sm, atol=1e-12)
    assert np.allclose(sp @ sm - sm @ sp, 2.0 * sz, atol=1e-12)


def test_spin_correlation_includes_the_transverse_channel(layout) -> None:
    """The scalar product must not collapse to 3 S^z S^z."""

    full = _dense(spin_correlation_operator(layout, 1))
    longitudinal_only = 3.0 * (
        _dense(spin_z_operator(layout, "d")) @ _dense(spin_z_operator(layout, 1))
    )
    assert not np.allclose(full, longitudinal_only, atol=1e-8)
    transverse = full - (
        _dense(spin_z_operator(layout, "d")) @ _dense(spin_z_operator(layout, 1))
    )
    assert np.linalg.norm(transverse) > 1e-6


@pytest.mark.parametrize("site", [1, 2])
def test_spin_correlation_is_hermitian(layout, site: int) -> None:
    matrix = _dense(spin_correlation_operator(layout, site))
    assert np.allclose(matrix, matrix.conj().T, atol=1e-12)


def test_initial_observables_have_their_decoupled_values(layout) -> None:
    psi = build_initial_state(N_BATH)
    value = lambda poly: float(np.real(np.vdot(psi, _dense(poly) @ psi)))
    assert value(impurity_magnetization_operator(layout)) == pytest.approx(1.0, abs=1e-12)
    assert value(impurity_double_occupancy_operator(layout)) == pytest.approx(0.0, abs=1e-12)
    for site in (1, 2):
        assert value(spin_correlation_operator(layout, site)) == pytest.approx(0.0, abs=1e-12)


def test_every_screening_observable_is_hermitian(layout) -> None:
    for name, poly in screening_observable_operators(
        layout, v=QUENCHED_HYBRIDIZATION
    ).items():
        matrix = _dense(poly)
        assert np.allclose(matrix, matrix.conj().T, atol=1e-12), name
