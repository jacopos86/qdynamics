from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

from pipelines.time_dynamics import pure_hubbard_fixed_vqe as pure
from pipelines.time_dynamics.fixed_vqe_conditioning import FixedVQEModelConfig
from src.quantum.hubbard_latex_python_pairs import (
    build_hubbard_hamiltonian,
    build_hubbard_holstein_hamiltonian,
)
from src.quantum.vqe_latex_python_pairs import VQEResult, exact_ground_energy_sector


def _coefficient_map(polynomial) -> dict[str, complex]:  # noqa: ANN001
    coefficients: defaultdict[str, complex] = defaultdict(complex)
    for term in polynomial.return_polynomial():
        coefficients[str(term.pw2strng())] += complex(term.p_coeff)
    return {
        word: coefficient
        for word, coefficient in coefficients.items()
        if abs(coefficient) > 1.0e-12
    }


def _sector_basis_indices() -> tuple[int, ...]:
    return tuple(
        basis_index
        for basis_index in range(1 << 8)
        if sum((basis_index >> qubit) & 1 for qubit in range(4)) == 2
        and sum((basis_index >> qubit) & 1 for qubit in range(4, 8)) == 2
    )


def _apply_pauli_word_to_index(word: str, basis_index: int) -> tuple[int, complex]:
    target = int(basis_index)
    phase = 1.0 + 0.0j
    for qubit in range(len(word)):
        operator = word[len(word) - 1 - qubit]
        bit = (int(basis_index) >> qubit) & 1
        if operator == "z" and bit:
            phase = -phase
        elif operator == "x":
            target ^= 1 << qubit
        elif operator == "y":
            target ^= 1 << qubit
            phase *= 1.0j if bit == 0 else -1.0j
    return target, phase


def _polynomial_action_on_index(polynomial, basis_index: int) -> dict[int, complex]:  # noqa: ANN001
    output: defaultdict[int, complex] = defaultdict(complex)
    for term in polynomial.return_polynomial():
        target, phase = _apply_pauli_word_to_index(
            str(term.pw2strng()), int(basis_index)
        )
        output[target] += complex(term.p_coeff) * phase
    return {
        target: amplitude
        for target, amplitude in output.items()
        if abs(amplitude) > 1.0e-12
    }


def test_hh_nph_zero_is_not_the_frozen_pure_hubbard_register_or_operator() -> None:
    pure_hamiltonian = build_hubbard_hamiltonian(
        dims=4,
        t=1.0,
        U=4.0,
        v=0.0,
        indexing="blocked",
        pbc=False,
    )
    hh_hamiltonian = build_hubbard_holstein_hamiltonian(
        dims=4,
        J=1.0,
        U=4.0,
        omega0=1.0,
        g=0.0,
        n_ph_max=0,
        boson_encoding="binary",
        v_t=None,
        v0=None,
        repr_mode="JW",
        indexing="blocked",
        pbc=False,
        include_zero_point=True,
    )

    hh_config = FixedVQEModelConfig(
        num_sites=4,
        t=1.0,
        u=4.0,
        g_ep=0.0,
        omega0=1.0,
        n_ph_max=0,
        ordering="blocked",
        boundary="open",
        sector_n_up=2,
        sector_n_dn=2,
    )
    assert hh_config.total_qubits == 12
    assert {term.nqubit() for term in pure_hamiltonian.return_polynomial()} == {8}
    assert {term.nqubit() for term in hh_hamiltonian.return_polynomial()} == {12}

    pure_padded = {
        "e" * 4 + word: coefficient
        for word, coefficient in _coefficient_map(pure_hamiltonian).items()
    }
    hh_coefficients = _coefficient_map(hh_hamiltonian)
    all_words = set(pure_padded) | set(hh_coefficients)
    difference = {
        word: hh_coefficients.get(word, 0.0) - pure_padded.get(word, 0.0)
        for word in all_words
        if abs(hh_coefficients.get(word, 0.0) - pure_padded.get(word, 0.0)) > 1.0e-12
    }
    assert difference == {"e" * 12: pytest.approx(2.0 + 0.0j)}


def test_frozen_l4_problem_is_exactly_the_direct_pure_hubbard_construction() -> None:
    config = pure.PureHubbardFixedVQEConfig()
    problem = pure.build_pure_hubbard_fixed_vqe_problem(config)
    direct_hamiltonian = build_hubbard_hamiltonian(
        dims=4,
        t=1.0,
        U=4.0,
        v=0.0,
        repr_mode="JW",
        indexing="blocked",
        pbc=False,
    )
    direct_exact = exact_ground_energy_sector(
        direct_hamiltonian,
        num_sites=4,
        num_particles=(2, 2),
        indexing="blocked",
    )

    assert config.total_qubits == 8
    assert config.num_particles == (2, 2)
    assert problem.reference_state.shape == (256,)
    assert np.linalg.norm(problem.reference_state) == pytest.approx(1.0)
    assert _coefficient_map(problem.hamiltonian) == pytest.approx(
        _coefficient_map(direct_hamiltonian)
    )
    assert problem.exact_energy == pytest.approx(direct_exact, abs=1.0e-12)
    assert problem.ansatz.nq == 8
    assert problem.sector_dimension == 36
    assert problem.unique_generator_count == 113
    assert len(problem.generator_terms) == 226
    assert problem.ansatz.num_parameters == 226
    assert len(problem.generator_pool_sha256) == 64
    assert [term.label for term in problem.generator_terms[:3]] == [
        "layer::00::uccsd_sing(alpha:0->2)",
        "layer::00::uccsd_sing(alpha:0->3)",
        "layer::00::uccsd_sing(alpha:1->2)",
    ]
    assert problem.generator_terms[112].label == "layer::00::hva_block::onsite_layer"
    assert problem.generator_terms[113].label == "layer::01::uccsd_sing(alpha:0->2)"
    assert problem.generator_terms[-1].label == "layer::01::hva_block::onsite_layer"
    assert all(term.execution_mode == "grouped_exact" for term in problem.generator_terms)
    assert all(
        term.nqubit() == 8
        for generator in problem.generator_terms
        for term in generator.polynomial.return_polynomial()
    )


def test_filtered_generator_pool_preserves_and_connects_the_entire_fixed_spin_sector() -> None:
    generators = pure.build_spin_sector_preserving_hubbard_generator_pool(
        pure.PureHubbardFixedVQEConfig()
    )
    basis = _sector_basis_indices()
    basis_set = set(basis)
    adjacency = {basis_index: set() for basis_index in basis}

    for generator in generators:
        for source in basis:
            output = _polynomial_action_on_index(generator.polynomial, source)
            assert set(output).issubset(basis_set), generator.label
            for target in output:
                if target != source:
                    adjacency[source].add(target)
                    adjacency[target].add(source)

    reached = {basis[0]}
    frontier = [basis[0]]
    while frontier:
        current = frontier.pop()
        unseen = adjacency[current] - reached
        reached.update(unseen)
        frontier.extend(unseen)

    assert len(basis) == 36
    assert reached == basis_set
    assert len(generators) > len(basis) - 1


@pytest.mark.parametrize(
    ("signed_error", "expected_qualified"),
    [(5.0e-7, True), (1.0e-6, True), (1.000001e-6, False)],
)
def test_generic_vqe_path_uses_pure_problem_and_enforces_frozen_gate(
    monkeypatch: pytest.MonkeyPatch,
    signed_error: float,
    expected_qualified: bool,
) -> None:
    captured: dict[str, object] = {}
    original_build = pure.build_pure_hubbard_fixed_vqe_problem
    expected_problem = original_build(pure.PureHubbardFixedVQEConfig())

    monkeypatch.setattr(
        pure,
        "build_pure_hubbard_fixed_vqe_problem",
        lambda config: expected_problem,
    )

    def _fake_vqe_minimize(H, ansatz, psi_ref, **kwargs):  # noqa: ANN001, ANN003
        captured.update(H=H, ansatz=ansatz, psi_ref=psi_ref, kwargs=kwargs)
        return VQEResult(
            energy=float(expected_problem.exact_energy + signed_error),
            theta=np.zeros(expected_problem.ansatz.num_parameters, dtype=float),
            success=True,
            message="mocked",
            nfev=1,
            nit=1,
            best_restart=0,
        )

    monkeypatch.setattr(pure, "vqe_minimize", _fake_vqe_minimize)
    result = pure.run_pure_hubbard_fixed_vqe(expected_problem.config)

    assert captured["H"] is expected_problem.hamiltonian
    assert captured["ansatz"] is expected_problem.ansatz
    assert captured["psi_ref"] is expected_problem.reference_state
    assert captured["kwargs"] == {
        "restarts": 4,
        "seed": 1_293_143_593,
        "initial_point_stddev": 0.3,
        "method": "SLSQP",
        "maxiter": 1800,
    }
    assert result.absolute_energy_error == pytest.approx(abs(signed_error), abs=1.0e-12)
    assert result.energy_gate == 1.0e-6
    assert result.qualified is expected_qualified


def test_frozen_config_serializes_no_boson_or_qiskit_surface() -> None:
    payload = pure.PureHubbardFixedVQEConfig().to_json_dict()
    assert payload["problem"] == "hubbard"
    assert payload["L"] == 4
    assert payload["u"] == 4.0
    assert payload["sector_n_up"] == payload["sector_n_down"] == 2
    assert payload["total_qubits"] == 8
    assert payload["initializer"] == "generator_derived_fixed_structure_vqe_v1"
    assert payload["generator_pool"] == "uccsd_qeb_hva_blocks"
    assert payload["generator_filter"] == "fixed_spin_sector_logical_generators_v1"
    assert payload["generator_order"] == "repository_deduplicated_pool_order_v1"
    assert payload["generator_execution"] == "grouped_exact_logical_shared_v1"
    assert payload["reps"] == 2
    assert payload["optimizer_method"] == "SLSQP"
    assert payload["maxiter"] == 1800
    assert payload["qiskit_compilation"] is False
    assert not any("phonon" in key or "boson" in key for key in payload)
