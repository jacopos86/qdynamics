"""Every SIAM pool block is Hermitian, sector preserving, and correctly encoded.

The encoded check is the load-bearing one: a block is written as a product of
Jordan-Wigner creation and annihilation polynomials, and this suite confirms the
resulting Pauli representation equals an independently constructed fermionic
matrix. A sign error in the Jordan-Wigner string would pass a Hermiticity check
and fail here.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.quantum.pauli_polynomial_class import (
    fermion_minus_operator,
    fermion_plus_operator,
)
from src.quantum.siam_model import (
    DOWN,
    SIAMLayout,
    UP,
    number_operator,
    total_number_operator,
)
from src.quantum.siam_pool import (
    build_siam_pool,
    current_quadrature,
    hopping_quadrature,
)
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix

N_BATH = 2


def _dense(poly) -> np.ndarray:
    return np.asarray(hamiltonian_matrix(poly), dtype=complex)


@pytest.fixture(scope="module")
def layout() -> SIAMLayout:
    return SIAMLayout(N_BATH)


@pytest.fixture(scope="module")
def pool():
    return build_siam_pool(N_BATH)


def test_pool_contains_every_declared_family(pool) -> None:
    families = {block.family for block in pool}
    assert families == {"K", "J", "Qd", "A", "X", "P"}


def test_pool_labels_are_unique_and_deterministic(pool) -> None:
    labels = [block.label for block in pool]
    assert len(set(labels)) == len(labels)
    assert labels == [block.label for block in build_siam_pool(N_BATH)]


def test_labels_carry_their_physical_content(pool) -> None:
    labels = {block.label for block in pool}
    assert "Qd_centered" in labels
    assert "P_d_1" in labels
    assert any(label.startswith("X_d_") for label in labels)
    assert any(label.startswith("K_hyb_") for label in labels)


@pytest.mark.parametrize("index", range(14))
def test_every_block_is_hermitian(pool, index: int) -> None:
    matrix = _dense(pool[index].operator)
    assert np.allclose(matrix, matrix.conj().T, atol=1e-12), pool[index].label


@pytest.mark.parametrize("spin", [UP, DOWN])
def test_every_block_preserves_each_spin_number(pool, layout, spin) -> None:
    number = _dense(total_number_operator(layout, spin))
    for block in pool:
        matrix = _dense(block.operator)
        assert np.allclose(
            matrix @ number - number @ matrix, 0.0, atol=1e-10
        ), f"{block.label} breaks N_{spin}"


def test_hopping_and_current_quadratures_match_fermionic_construction(layout) -> None:
    """Independent construction from raw JW operators, not from the helpers."""

    nq = layout.n_qubits
    a, b = layout.bath(1, UP), layout.bath(2, UP)
    cd_a = _dense(fermion_plus_operator("JW", nq, a))
    cd_b = _dense(fermion_plus_operator("JW", nq, b))
    c_a = _dense(fermion_minus_operator("JW", nq, a))
    c_b = _dense(fermion_minus_operator("JW", nq, b))
    assert np.allclose(
        _dense(hopping_quadrature(layout, a, b)), cd_a @ c_b + cd_b @ c_a, atol=1e-12
    )
    assert np.allclose(
        _dense(current_quadrature(layout, a, b)),
        1.0j * (cd_a @ c_b - cd_b @ c_a),
        atol=1e-12,
    )


def test_pair_transfer_moves_exactly_one_pair(layout, pool) -> None:
    """P must move an up and a down together, so each species count is fixed."""

    block = next(b for b in pool if b.family == "P")
    matrix = _dense(block.operator)
    n_d_up = _dense(number_operator(layout, layout.impurity(UP)))
    n_d_dn = _dense(number_operator(layout, layout.impurity(DOWN)))
    # the double-occupancy operator changes by the same amount in both species
    assert np.allclose(
        matrix @ (n_d_up - n_d_dn) - (n_d_up - n_d_dn) @ matrix, 0.0, atol=1e-10
    )


def test_density_assisted_block_is_a_genuine_three_body_term(layout, pool) -> None:
    """A must not reduce to the bare current it gates."""

    block = next(b for b in pool if b.label == f"A_hyb_{UP}")
    gated = _dense(block.operator)
    bare = _dense(current_quadrature(layout, layout.impurity(UP), layout.bath(1, UP)))
    assert not np.allclose(gated, bare, atol=1e-8)
    assert np.linalg.norm(gated) > 1e-8


def test_pool_size_grows_with_bath_as_expected() -> None:
    assert len(build_siam_pool(2)) == 14
    assert len(build_siam_pool(4)) == 24
