from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.qse_spectra.core import (
    QSEBasisVectorPolicy,
    computational_basis_state,
    pauli_string_basis_element,
)
from pipelines.qse_spectra.exchange_maintenance import (
    QSEExchangeConfig,
    exchange_maintenance_payload,
    run_qse_exchange_maintenance,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm

_Q0 = QSEBasisVectorPolicy(reference_projection="q0", basis_vector_normalization="raw_projected")


def _hamiltonian() -> PauliPolynomial:
    out = PauliPolynomial("JW")
    out.add_term(PauliTerm(1, ps="z", pc=1.0))
    out.add_term(PauliTerm(1, ps="x", pc=0.5))
    return out


def _pool():
    return [
        pauli_string_basis_element("I", nq=1, name="identity"),
        pauli_string_basis_element("Z", nq=1, name="parallel_z"),
        pauli_string_basis_element("Y", nq=1, name="flip_expensive"),
        pauli_string_basis_element("X", nq=1, name="flip_cheap"),
    ]


def test_exchange_swaps_expensive_redundant_operator_for_cheaper_equivalent() -> None:
    # Y|0> and X|0> span the same projected direction; Y is priced higher, so
    # the certified exchange should swap Y -> X at unchanged root energy and
    # drop the useless identity via pure deletion.
    psi = computational_basis_state(1, "0")
    costs = (0.0, 1.0, 5.0, 1.0)

    result = run_qse_exchange_maintenance(
        _pool(),
        (0, 2),
        costs,
        hamiltonian=_hamiltonian(),
        prepared_state=psi,
        basis_vector_policy=_Q0,
    )

    assert 3 in result.final_indices
    assert 2 not in result.final_indices
    assert result.final_summary["total_compiled_cost"] < result.initial_summary["total_compiled_cost"]
    assert result.final_summary["root0_energy"] == pytest.approx(
        result.initial_summary["root0_energy"], abs=1.0e-9
    )

    payload = exchange_maintenance_payload(result)
    assert payload["schema_version"] == "qse_exchange_maintenance_v1"
    assert payload["controller_boundary"]["feeds_controller_decisions"] is False
    assert payload["committed_patch_count"] >= 1
    rejected = [
        patch
        for round_record in payload["rounds"]
        for patch in round_record["evaluated_patches"]
        if not patch["accepted"]
    ]
    assert rejected and all(patch["rejection_reasons"] for patch in rejected)


def test_exchange_accepts_accuracy_improving_joint_patch() -> None:
    # H = -Z_q0 - 0.4 Z_q1 on |00>: the q1 flip (|10>, E=-0.6) is the true
    # first excitation, the q0 flip (|01>, E=+0.6) a higher one. Starting from
    # {dead Z_q1 direction, q0 flip}, the coupled delete--add patch must swap
    # the dead direction for the q1 flip, lowering the target root.
    hamiltonian = PauliPolynomial("JW")
    hamiltonian.add_term(PauliTerm(2, ps="ez", pc=-1.0))
    hamiltonian.add_term(PauliTerm(2, ps="ze", pc=-0.4))
    psi = computational_basis_state(2, "00")
    pool = [
        pauli_string_basis_element("II", nq=2, name="identity"),
        pauli_string_basis_element("ZI", nq=2, name="dead_z_q1"),
        pauli_string_basis_element("IX", nq=2, name="flip_q0"),
        pauli_string_basis_element("XI", nq=2, name="flip_q1"),
    ]
    costs = (0.0, 3.0, 1.0, 1.0)

    result = run_qse_exchange_maintenance(
        pool,
        (1, 2),
        costs,
        hamiltonian=hamiltonian,
        prepared_state=psi,
        basis_vector_policy=_Q0,
        config=QSEExchangeConfig(max_rounds=4),
    )

    assert 3 in result.final_indices
    assert 1 not in result.final_indices
    assert result.final_summary["root0_energy"] == pytest.approx(-0.6, abs=1.0e-9)
    assert result.final_summary["root0_energy"] < result.initial_summary["root0_energy"] - 1.0


def test_exchange_rejects_all_patches_when_support_is_optimal() -> None:
    psi = computational_basis_state(1, "0")
    costs = (0.0, 1.0, 5.0, 1.0)

    result = run_qse_exchange_maintenance(
        _pool(),
        (3,),
        costs,
        hamiltonian=_hamiltonian(),
        prepared_state=psi,
        basis_vector_policy=_Q0,
    )

    assert result.final_indices == (3,)
    assert exchange_maintenance_payload(result)["committed_patch_count"] == 0


def test_exchange_config_validation() -> None:
    with pytest.raises(ValueError, match="max_rounds"):
        QSEExchangeConfig(max_rounds=0)
    with pytest.raises(ValueError, match="condition_slack_factor"):
        QSEExchangeConfig(condition_slack_factor=0.5)
    with pytest.raises(ValueError, match="min_retained_rank_fraction"):
        QSEExchangeConfig(min_retained_rank_fraction=0.0)
    with pytest.raises(ValueError, match="compiled_costs"):
        run_qse_exchange_maintenance(
            _pool(),
            (0,),
            (1.0, 2.0),
            hamiltonian=_hamiltonian(),
            prepared_state=computational_basis_state(1, "0"),
            basis_vector_policy=_Q0,
        )
