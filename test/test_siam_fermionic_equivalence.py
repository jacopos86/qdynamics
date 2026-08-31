"""Independent occupation-basis checks for the encoded SIAM operators.

The expected matrices in this module are built directly by acting on bitstring
occupations.  In particular, the expected side does not import or call the
repository's Jordan--Wigner creation/annihilation helpers.  This makes the
tests sensitive to qubit-ordering, parity-string, coefficient, and operator-
ordering mistakes in the production Pauli polynomials.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.quantum.siam_model import build_siam_hamiltonian
from src.quantum.siam_pool import build_siam_pool
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix


N_BATH = 2
N_QUBITS = 2 * (N_BATH + 1)
DIMENSION = 1 << N_QUBITS

# Documented spin-blocked mode order for N_b = 2:
# (d_up, c1_up, c2_up, d_down, c1_down, c2_down).
D_UP, C1_UP, C2_UP, D_DOWN, C1_DOWN, C2_DOWN = range(N_QUBITS)

EXPECTED_POOL_LABELS = (
    "K_hyb_up_0_1",
    "K_bath_up_1_2",
    "K_hyb_down_3_4",
    "K_bath_down_4_5",
    "J_hyb_up_0_1",
    "J_bath_up_1_2",
    "J_hyb_down_3_4",
    "J_bath_down_4_5",
    "Qd_centered",
    "A_hyb_up",
    "A_hyb_down",
    "X_d_1",
    "X_d_2",
    "P_d_1",
)


def _annihilation_matrix(mode: int) -> np.ndarray:
    """Return c_mode by direct occupation-bit action (mode 0 is the LSB)."""

    matrix = np.zeros((DIMENSION, DIMENSION), dtype=complex)
    lower_modes = (1 << int(mode)) - 1
    for ket in range(DIMENSION):
        if not ((ket >> int(mode)) & 1):
            continue
        bra = ket ^ (1 << int(mode))
        parity = (ket & lower_modes).bit_count()
        matrix[bra, ket] = -1.0 if parity % 2 else 1.0
    return matrix


def _creation_matrix(mode: int) -> np.ndarray:
    """Return c_mode^dagger by direct occupation-bit action."""

    matrix = np.zeros((DIMENSION, DIMENSION), dtype=complex)
    lower_modes = (1 << int(mode)) - 1
    for ket in range(DIMENSION):
        if (ket >> int(mode)) & 1:
            continue
        bra = ket ^ (1 << int(mode))
        parity = (ket & lower_modes).bit_count()
        matrix[bra, ket] = -1.0 if parity % 2 else 1.0
    return matrix


ANNIHILATE = tuple(_annihilation_matrix(mode) for mode in range(N_QUBITS))
CREATE = tuple(_creation_matrix(mode) for mode in range(N_QUBITS))
NUMBER = tuple(CREATE[mode] @ ANNIHILATE[mode] for mode in range(N_QUBITS))
IDENTITY = np.eye(DIMENSION, dtype=complex)


def _hopping_quadrature(mode_a: int, mode_b: int) -> np.ndarray:
    return (
        CREATE[mode_a] @ ANNIHILATE[mode_b]
        + CREATE[mode_b] @ ANNIHILATE[mode_a]
    )


def _current_quadrature(mode_a: int, mode_b: int) -> np.ndarray:
    return 1.0j * (
        CREATE[mode_a] @ ANNIHILATE[mode_b]
        - CREATE[mode_b] @ ANNIHILATE[mode_a]
    )


def _expected_hamiltonian(*, u: float, v: float, hopping: float) -> np.ndarray:
    bath = -float(hopping) * (
        _hopping_quadrature(C1_UP, C2_UP)
        + _hopping_quadrature(C1_DOWN, C2_DOWN)
    )
    interaction = float(u) * (
        (NUMBER[D_UP] - 0.5 * IDENTITY)
        @ (NUMBER[D_DOWN] - 0.5 * IDENTITY)
    )
    hybridization = -float(v) * (
        _hopping_quadrature(D_UP, C1_UP)
        + _hopping_quadrature(D_DOWN, C1_DOWN)
    )
    return bath + interaction + hybridization


def _spin_correlation(site_up: int, site_down: int) -> np.ndarray:
    impurity_z = 0.5 * (NUMBER[D_UP] - NUMBER[D_DOWN])
    site_z = 0.5 * (NUMBER[site_up] - NUMBER[site_down])
    impurity_raise = CREATE[D_UP] @ ANNIHILATE[D_DOWN]
    impurity_lower = CREATE[D_DOWN] @ ANNIHILATE[D_UP]
    site_raise = CREATE[site_up] @ ANNIHILATE[site_down]
    site_lower = CREATE[site_down] @ ANNIHILATE[site_up]
    return impurity_z @ site_z + 0.5 * (
        impurity_raise @ site_lower + impurity_lower @ site_raise
    )


def _expected_pool() -> dict[str, tuple[str, np.ndarray]]:
    centered_interaction = (
        (NUMBER[D_UP] - 0.5 * IDENTITY)
        @ (NUMBER[D_DOWN] - 0.5 * IDENTITY)
    )
    pair_forward = (
        CREATE[D_UP]
        @ CREATE[D_DOWN]
        @ ANNIHILATE[C1_DOWN]
        @ ANNIHILATE[C1_UP]
    )
    pair_backward = (
        CREATE[C1_UP]
        @ CREATE[C1_DOWN]
        @ ANNIHILATE[D_DOWN]
        @ ANNIHILATE[D_UP]
    )
    return {
        "K_hyb_up_0_1": ("K", _hopping_quadrature(D_UP, C1_UP)),
        "K_bath_up_1_2": ("K", _hopping_quadrature(C1_UP, C2_UP)),
        "K_hyb_down_3_4": ("K", _hopping_quadrature(D_DOWN, C1_DOWN)),
        "K_bath_down_4_5": ("K", _hopping_quadrature(C1_DOWN, C2_DOWN)),
        "J_hyb_up_0_1": ("J", _current_quadrature(D_UP, C1_UP)),
        "J_bath_up_1_2": ("J", _current_quadrature(C1_UP, C2_UP)),
        "J_hyb_down_3_4": ("J", _current_quadrature(D_DOWN, C1_DOWN)),
        "J_bath_down_4_5": ("J", _current_quadrature(C1_DOWN, C2_DOWN)),
        "Qd_centered": ("Qd", centered_interaction),
        "A_hyb_up": (
            "A",
            NUMBER[D_DOWN] @ _current_quadrature(D_UP, C1_UP),
        ),
        "A_hyb_down": (
            "A",
            NUMBER[D_UP] @ _current_quadrature(D_DOWN, C1_DOWN),
        ),
        "X_d_1": ("X", _spin_correlation(C1_UP, C1_DOWN)),
        "X_d_2": ("X", _spin_correlation(C2_UP, C2_DOWN)),
        "P_d_1": ("P", pair_forward + pair_backward),
    }


def _production_matrix(polynomial) -> np.ndarray:
    return np.asarray(hamiltonian_matrix(polynomial), dtype=complex)


def test_full_siam_hamiltonian_matches_direct_occupation_basis_matrix() -> None:
    parameters = {"u": 1.375, "v": 0.625, "hopping": 0.875}
    actual = _production_matrix(build_siam_hamiltonian(N_BATH, **parameters))
    expected = _expected_hamiltonian(**parameters)

    assert actual.shape == expected.shape == (DIMENSION, DIMENSION)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)


def test_expected_pool_covers_the_complete_production_pool() -> None:
    actual = {block.label: block for block in build_siam_pool(N_BATH)}
    expected = _expected_pool()

    assert tuple(actual) == EXPECTED_POOL_LABELS
    assert set(actual) == set(expected)


@pytest.mark.parametrize("label", EXPECTED_POOL_LABELS)
def test_every_logical_pool_block_matches_direct_occupation_basis_matrix(
    label: str,
) -> None:
    actual = {block.label: block for block in build_siam_pool(N_BATH)}[label]
    expected_family, expected_matrix = _expected_pool()[label]

    assert actual.family == expected_family
    actual_matrix = _production_matrix(actual.operator)
    assert actual_matrix.shape == expected_matrix.shape == (DIMENSION, DIMENSION)
    np.testing.assert_allclose(
        actual_matrix,
        expected_matrix,
        rtol=0.0,
        atol=1.0e-12,
        err_msg=label,
    )
