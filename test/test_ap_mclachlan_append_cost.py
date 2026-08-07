from __future__ import annotations

import pytest

from pipelines.time_dynamics.ap_mclachlan.append_cost import (
    AP_APPEND_COST_NORMALIZATION_RAW_LEGACY_V1,
    AP_APPEND_NO_MEASUREMENT_COST_SOURCE_V1,
    AP_APPEND_RANK_SCORE_KIND_V1,
    AppendCostSettings,
    append_cost_telemetry_for_family,
    estimate_append_atom_set_cost,
)
from pipelines.time_dynamics.ap_mclachlan.support_atoms import SupportAtom
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _term(label: str) -> AnsatzTerm:
    poly = PauliPolynomial("JW")
    poly.add_term(PauliTerm(len(str(label)), ps=str(label), pc=1.0))
    poly._reduce()
    return AnsatzTerm(label=f"candidate_{label}", polynomial=poly)


def _atom(label: str) -> SupportAtom:
    term = _term(label)
    return SupportAtom(
        atom_id=f"pauli:{label}",
        atom_label=str(term.label),
        parent_label=str(term.label),
        term=term,
        parameterization_mode="per_pauli_term",
        runtime_count=1,
        origin_kind="test",
    )


def test_append_cost_estimate_reuses_paper_i_pauli_proxy() -> None:
    raw = estimate_append_atom_set_cost((_atom("xz"),))

    assert raw.raw_components["2q"] == pytest.approx(2.0)
    assert raw.raw_components["d"] == pytest.approx(2.0)
    assert raw.raw_components["1q"] == pytest.approx(3.0)
    assert raw.raw_components["theta"] == pytest.approx(1.0)
    assert raw.raw_components["shot"] == pytest.approx(0.0)
    assert raw.component_sources["shot"] == AP_APPEND_NO_MEASUREMENT_COST_SOURCE_V1


def test_append_cost_utility_is_gain_divided_by_denominator() -> None:
    raw = estimate_append_atom_set_cost((_atom("xz"),))
    settings = AppendCostSettings(
        cost_normalization_mode=AP_APPEND_COST_NORMALIZATION_RAW_LEGACY_V1,
        append_cost_alpha=1.0,
        lambda_2q=0.05,
        lambda_d=0.05,
        lambda_1q=0.025,
        lambda_theta=0.0,
        lambda_shot=0.02,
    )

    telemetry = append_cost_telemetry_for_family(
        (raw,),
        insertion_gains=(0.5,),
        settings=settings,
    )[0]

    expected_denominator = 1.0 + 0.05 * 2.0 + 0.05 * 2.0 + 0.025 * 3.0
    assert telemetry.rank_score_kind == AP_APPEND_RANK_SCORE_KIND_V1
    assert telemetry.hardware_cost_denominator == pytest.approx(expected_denominator)
    assert telemetry.rank_utility == pytest.approx(0.5 / expected_denominator)
