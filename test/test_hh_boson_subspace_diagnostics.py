from __future__ import annotations

import numpy as np
import pytest

from pipelines.static_adapt.adapt_pipeline import _attach_boson_chain_subspace_diagnostics
from src.quantum.hartree_fock_reference_state import hubbard_holstein_reference_state


def test_hh_nph2_boson_subspace_diagnostics_use_full_register_layout() -> None:
    psi_reference = hubbard_holstein_reference_state(
        dims=2,
        num_particles=(1, 1),
        n_ph_max=2,
        boson_encoding="binary",
        indexing="blocked",
    )
    fermion_register_width = 4
    fermion_index = int(np.argmax(np.abs(psi_reference))) & ((1 << fermion_register_width) - 1)
    illegal_boson_site0_codeword = 3  # nph=2 binary has legal local codes 0,1,2 only.
    illegal_index = int(illegal_boson_site0_codeword << fermion_register_width) | int(fermion_index)
    psi_final = np.zeros_like(psi_reference)
    psi_final[int(np.argmax(np.abs(psi_reference)))] = 1.0 / np.sqrt(2.0)
    psi_final[illegal_index] = 1.0 / np.sqrt(2.0)

    adapt_payload: dict[str, object] = {}
    _attach_boson_chain_subspace_diagnostics(
        adapt_payload,
        problem_key="hh",
        psi_adapt=psi_final,
        psi_reference=psi_reference,
        num_sites=2,
        n_ph_max=2,
        boson_encoding="binary",
    )

    diag = adapt_payload["boson_subspace_diagnostics"]
    assert isinstance(diag, dict)
    assert diag["available"] is True
    assert diag["problem"] == "hh"
    assert diag["legal_subspace_scope"] == "boson_codewords_with_full_fermion_register"
    assert diag["total_register_width"] == 8
    assert diag["non_boson_register_width"] == 4
    assert diag["boson_register_width"] == 4
    assert diag["bits_per_boson_site"] == 2
    assert diag["boson_legal_codeword_count"] == 9
    assert diag["legal_subspace_dim"] == 144
    assert diag["illegal_state_count"] == 112
    assert diag["reference_legal_probability"] == pytest.approx(1.0)
    assert diag["reference_illegal_probability"] == pytest.approx(0.0)
    assert diag["final_legal_probability"] == pytest.approx(0.5)
    assert diag["final_illegal_probability"] == pytest.approx(0.5)
    assert diag["boson_legal_probability_min"] == pytest.approx(0.5)
    assert diag["boson_illegal_probability_max"] == pytest.approx(0.5)


def test_hh_nph2_legal_prepared_state_emits_near_zero_illegal_probability() -> None:
    psi_reference = hubbard_holstein_reference_state(
        dims=2,
        num_particles=(1, 1),
        n_ph_max=2,
        boson_encoding="binary",
        indexing="blocked",
    )

    adapt_payload: dict[str, object] = {}
    _attach_boson_chain_subspace_diagnostics(
        adapt_payload,
        problem_key="hh",
        psi_adapt=psi_reference,
        psi_reference=psi_reference,
        num_sites=2,
        n_ph_max=2,
        boson_encoding="binary",
    )

    diag = adapt_payload["boson_subspace_diagnostics"]
    assert isinstance(diag, dict)
    assert diag["final_legal_probability"] == pytest.approx(1.0)
    assert diag["final_illegal_probability"] == pytest.approx(0.0, abs=1e-12)
    assert diag["boson_illegal_probability_max"] == pytest.approx(0.0, abs=1e-12)
