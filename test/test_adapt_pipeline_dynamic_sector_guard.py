from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipelines.contracts.problem import (
    FixedCountConstraint,
    ProblemRequest,
    RegisterBlockSpec,
    RegisterLayoutSpec,
    SectorSelection,
)
from pipelines.static_adapt.adapt_pipeline import _assert_ansatz_term_sector_safe
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _one_particle_context():
    return SimpleNamespace(
        layout=RegisterLayoutSpec(
            total_qubits=3,
            fermion_qubits=3,
            boson_qubits=0,
            ordering="blocked",
            boson_encoding=None,
            blocks=(
                RegisterBlockSpec(
                    name="fermion",
                    kind="fermion",
                    start_qubit=0,
                    stop_qubit=3,
                ),
            ),
        ),
        sector=SectorSelection(
            label="one_particle",
            comparison_space_label="one_particle",
            constraints=(FixedCountConstraint(quantity="n_f", value=1),),
            num_particles=None,
        ),
        request=ProblemRequest(
            problem_key="test",
            num_sites=3,
            t=0.0,
            u=0.0,
            dv=0.0,
            omega0=0.0,
            g_ep=0.0,
            n_ph_max=0,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
        ),
    )


def _two_bond_term(*, execution_mode: str) -> AnsatzTerm:
    return AnsatzTerm(
        label=f"two_bonds::{execution_mode}",
        polynomial=PauliPolynomial(
            "JW",
            [
                PauliTerm(3, ps="exx", pc=1.0),
                PauliTerm(3, ps="eyy", pc=1.0),
                PauliTerm(3, ps="xxe", pc=1.0),
                PauliTerm(3, ps="yye", pc=1.0),
            ],
        ),
        execution_mode=execution_mode,
    )


def test_dynamic_sector_guard_rejects_unsafe_termwise_execution():
    with pytest.raises(RuntimeError, match="execution_passed=False"):
        _assert_ansatz_term_sector_safe(
            _two_bond_term(execution_mode="termwise_product"),
            resolved_problem=_one_particle_context(),
            source="unit_test_dynamic_child",
        )


def test_dynamic_sector_guard_accepts_sector_safe_grouped_execution():
    audit = _assert_ansatz_term_sector_safe(
        _two_bond_term(execution_mode="grouped_exact"),
        resolved_problem=_one_particle_context(),
        source="unit_test_dynamic_child",
    )

    assert audit["passed"] is True
    assert audit["execution_passed"] is True


def test_dynamic_sector_guard_rejects_grouped_algebra_violation():
    term = AnsatzTerm(
        label="single_x",
        polynomial=PauliPolynomial(
            "JW",
            [PauliTerm(3, ps="eex", pc=1.0)],
        ),
        execution_mode="grouped_exact",
    )

    with pytest.raises(RuntimeError, match="grouped_passed=False"):
        _assert_ansatz_term_sector_safe(
            term,
            resolved_problem=_one_particle_context(),
            source="unit_test_dynamic_child",
        )
