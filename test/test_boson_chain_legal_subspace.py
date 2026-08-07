from __future__ import annotations

import numpy as np
import pytest

from src.quantum.operator_pools.boson_chains import (
    boson_chain_illegal_probability,
    boson_chain_legal_basis_indices,
    boson_chain_legal_probability,
    build_boson_chain_fock_statevector,
)


def test_binary_cutoff_two_has_one_illegal_local_codeword_per_site() -> None:
    indices = boson_chain_legal_basis_indices(num_sites=1, n_ph_max=2, boson_encoding="binary")
    assert tuple(indices.tolist()) == (0, 1, 2)

    illegal = np.zeros(4, dtype=complex)
    illegal[3] = 1.0
    assert boson_chain_legal_probability(illegal, num_sites=1, n_ph_max=2, boson_encoding="binary") == pytest.approx(0.0)
    assert boson_chain_illegal_probability(illegal, num_sites=1, n_ph_max=2, boson_encoding="binary") == pytest.approx(1.0)


def test_product_fock_state_is_legal_for_binary_and_unary() -> None:
    for encoding in ("binary", "unary"):
        psi = build_boson_chain_fock_statevector(
            num_sites=2,
            n_ph_max=2,
            boson_encoding=encoding,
            occupations=(0, 2),
        )
        assert boson_chain_legal_probability(psi, num_sites=2, n_ph_max=2, boson_encoding=encoding) == pytest.approx(1.0)
        assert boson_chain_illegal_probability(psi, num_sites=2, n_ph_max=2, boson_encoding=encoding) == pytest.approx(0.0)


def test_legal_probability_counts_only_legal_tensor_product_codes() -> None:
    legal = build_boson_chain_fock_statevector(num_sites=1, n_ph_max=2, boson_encoding="binary", occupations=(1,))
    illegal = np.zeros_like(legal)
    illegal[3] = 1.0
    psi = (legal + illegal) / np.sqrt(2.0)
    assert boson_chain_legal_probability(psi, num_sites=1, n_ph_max=2, boson_encoding="binary") == pytest.approx(0.5)
