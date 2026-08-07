from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.exact_bench import generic_static_adapt_variants as variants
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


def _boson_context(hamiltonian: PauliPolynomial) -> SimpleNamespace:
    return SimpleNamespace(
        family_key="bose_hubbard",
        request=SimpleNamespace(
            problem_key="bose_hubbard",
            num_sites=1,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
        ),
        layout=SimpleNamespace(
            total_qubits=1,
            fermion_qubits=0,
        ),
        sector=SimpleNamespace(
            label="truncated_boson_register",
            num_particles=None,
        ),
        default_num_particles=(0, 0),
        hamiltonian=hamiltonian,
    )


def _fidelity_fields(
    *, hamiltonian: PauliPolynomial, state: np.ndarray
) -> dict:
    compiled = variants.compile_polynomial_action(hamiltonian)
    return variants._dense_exact_state_fidelity_for_selected(
        context=_boson_context(hamiltonian),
        selected=(),
        theta=np.zeros(0, dtype=float),
        psi_ref=np.asarray(state, dtype=complex),
        h_compiled=compiled,
        pauli_action_cache={},
        exact_energy=None,
        max_qubits=2,
    )


def test_generic_terminal_unique_ground_state_uses_physical_projector_receipt() -> None:
    # (I-Z)/2 has the unique |0> ground state.
    hamiltonian = PauliPolynomial(
        "JW",
        [
            PauliTerm(1, ps="e", pc=0.5),
            PauliTerm(1, ps="z", pc=-0.5),
        ],
    )
    fields = _fidelity_fields(
        hamiltonian=hamiltonian,
        state=np.asarray([1.0, 0.0], dtype=complex),
    )

    assert fields["infidelity_status"] == "computed_ground_space_projector"
    assert fields["exact_state_fidelity"] == pytest.approx(1.0)
    assert fields["exact_state_fidelity_source"] == (
        "physical_sector_ground_space_projector_same_cutoff"
    )
    receipt = fields["ground_space_fidelity"]
    assert receipt["ground_space_multiplicity"] == 1
    assert receipt["ground_space_unique_proved"] is True
    assert receipt["s_alg_charged"] is False


def test_generic_terminal_degenerate_ground_space_is_basis_independent() -> None:
    # A constant Hamiltonian has a two-dimensional ground space.  Arbitrary
    # normalized states must have unit projector fidelity.
    hamiltonian = PauliPolynomial(
        "JW", [PauliTerm(1, ps="e", pc=2.0)]
    )
    state = np.asarray([1.0, 1.0j], dtype=complex) / np.sqrt(2.0)
    fields = _fidelity_fields(hamiltonian=hamiltonian, state=state)

    assert fields["exact_state_fidelity"] == pytest.approx(1.0)
    assert fields["infidelity_exact"] == pytest.approx(0.0)
    receipt = fields["ground_space_fidelity"]
    assert receipt["reference_convention"] == "degenerate_ground_space_projector"
    assert receipt["ground_space_multiplicity"] == 2
    assert receipt["ground_space_unique_proved"] is False

