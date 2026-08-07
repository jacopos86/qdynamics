from __future__ import annotations

from pipelines.static_adapt.adapt_pipeline import _beam_executor_cache_key
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _term(
    *,
    label: str,
    pauli: str,
    coefficient: float,
    execution_mode: str = "termwise_product",
) -> AnsatzTerm:
    return AnsatzTerm(
        label=label,
        polynomial=PauliPolynomial(
            "JW",
            [PauliTerm(len(pauli), ps=pauli, pc=coefficient)],
        ),
        execution_mode=execution_mode,
    )


def test_beam_executor_cache_key_distinguishes_same_label_generator_semantics() -> None:
    baseline = _term(label="runtime-child", pauli="xe", coefficient=0.5)
    changed_word = _term(label="runtime-child", pauli="ze", coefficient=0.5)
    changed_coefficient = _term(
        label="runtime-child",
        pauli="xe",
        coefficient=0.75,
    )
    changed_execution = _term(
        label="runtime-child",
        pauli="xe",
        coefficient=0.5,
        execution_mode="grouped_exact",
    )

    baseline_key = _beam_executor_cache_key([baseline])

    assert baseline_key != _beam_executor_cache_key([changed_word])
    assert baseline_key != _beam_executor_cache_key([changed_coefficient])
    assert baseline_key != _beam_executor_cache_key([changed_execution])


def test_beam_executor_cache_key_preserves_operator_order() -> None:
    first = _term(label="first", pauli="xe", coefficient=0.5)
    second = _term(label="second", pauli="ez", coefficient=-0.25)

    assert _beam_executor_cache_key([first, second]) != _beam_executor_cache_key(
        [second, first]
    )
