from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.qse_spectra import (
    QSEBasisVectorPolicy,
    computational_basis_state,
    compute_qse_spectra,
    compute_transition_observables,
    pauli_string_basis_element,
    pauli_string_observable,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


def _poly(nq: int, terms: list[tuple[str, complex]]) -> PauliPolynomial:
    out = PauliPolynomial("JW")
    for label, coeff in terms:
        out.add_term(PauliTerm(int(nq), ps=str(label), pc=complex(coeff)))
    return out


def test_public_transition_helper_matches_embedded_scalar_output() -> None:
    hamiltonian = _poly(1, [("z", 1.0)])
    source = computational_basis_state(1, "0")
    basis = [pauli_string_basis_element("X", nq=1, name="x_excitation")]
    observable = pauli_string_observable("X", nq=1, name="dipole")
    policy = QSEBasisVectorPolicy(reference_projection="q0", basis_vector_normalization="raw_projected")

    embedded = compute_qse_spectra(
        hamiltonian,
        source,
        basis,
        basis_vector_policy=policy,
        transition_observables=[observable],
    )
    base = compute_qse_spectra(
        hamiltonian,
        source,
        basis,
        basis_vector_policy=policy,
    )
    helper_results = compute_transition_observables(base, source, [observable])

    assert len(embedded.transition_observables) == 1
    assert len(helper_results) == 1
    embedded_transition = embedded.transition_observables[0]
    helper_transition = helper_results[0]
    assert helper_transition.observable.name == embedded_transition.observable.name
    assert np.allclose(helper_transition.observable_matrix, embedded_transition.observable_matrix, atol=1e-12)
    assert np.allclose(helper_transition.transition_vector, embedded_transition.transition_vector, atol=1e-12)
    assert np.allclose(helper_transition.transition_amplitudes, embedded_transition.transition_amplitudes, atol=1e-12)
    assert np.allclose(helper_transition.transition_strengths, embedded_transition.transition_strengths, atol=1e-12)


def test_transition_helper_accepts_source_state_distinct_from_qse_build_state() -> None:
    hamiltonian = _poly(1, [("z", 1.0)])
    build_state = computational_basis_state(1, "0")
    distinct_source_state = computational_basis_state(1, "1")
    policy = QSEBasisVectorPolicy(reference_projection="q0", basis_vector_normalization="raw_projected")
    result = compute_qse_spectra(
        hamiltonian,
        build_state,
        [pauli_string_basis_element("X", nq=1, name="x_excitation")],
        basis_vector_policy=policy,
    )

    from_build_state = compute_transition_observables(
        result,
        build_state,
        [pauli_string_observable("I", nq=1, name="identity")],
    )[0]
    from_distinct_state = compute_transition_observables(
        result,
        distinct_source_state,
        [pauli_string_observable("I", nq=1, name="identity")],
    )[0]

    assert result.eigenvalues.tolist() == pytest.approx([-1.0], abs=1e-12)
    assert from_build_state.transition_vector[0] == pytest.approx(0.0 + 0.0j, abs=1e-12)
    assert from_build_state.transition_amplitudes[0] == pytest.approx(0.0 + 0.0j, abs=1e-12)
    assert from_build_state.transition_strengths[0] == pytest.approx(0.0, abs=1e-12)
    assert from_distinct_state.transition_vector[0] == pytest.approx(1.0 + 0.0j, abs=1e-12)
    assert from_distinct_state.transition_amplitudes[0] == pytest.approx(1.0 + 0.0j, abs=1e-12)
    assert from_distinct_state.transition_strengths[0] == pytest.approx(1.0, abs=1e-12)
