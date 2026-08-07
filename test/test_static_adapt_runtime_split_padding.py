from __future__ import annotations

from itertools import combinations

import pytest

from pipelines.scaffold.hh_continuation_generators import (
    serialize_polynomial_terms_exyz,
)
from pipelines.static_adapt.builders.legal_subspace_filter import (
    legal_subspace_basis_for_problem,
    pauli_action_on_basis_index,
)
from pipelines.static_adapt.route_a_child_padding import (
    ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_NPH2_V1,
    ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1,
    RouteAChildPaddingConfig,
)
from pipelines.static_adapt.runtime_split import (
    project_and_deduplicate_runtime_split_child_sets,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


def _child_set(
    *,
    label: str,
    terms: list[tuple[str, float]],
    parent_label: str,
    child_indices: list[int],
    position_id: int = 0,
) -> dict[str, object]:
    polynomial = PauliPolynomial(
        "JW",
        [
            PauliTerm(len(pauli_label), ps=pauli_label, pc=coefficient)
            for pauli_label, coefficient in terms
        ],
    )
    serialized = serialize_polynomial_terms_exyz(polynomial)
    return {
        "candidate_label": label,
        "candidate_polynomial": polynomial,
        "candidate_generator_metadata": {
            "generator_id": f"gen:raw:{label}",
            "family_id": "test_runtime_split",
            "parent_generator_id": f"gen:parent:{parent_label}",
            "symmetry_spec": None,
            "compile_metadata": {
                "serialized_terms_exyz": serialized,
                "runtime_split": {
                    "mode": "shortlist_pauli_children_v1",
                    "parent_label": parent_label,
                    "child_indices": list(child_indices),
                    "child_labels": [label],
                    "child_generator_ids": [f"gen:raw:{label}"],
                    "representation": "child_set",
                },
            },
        },
        "child_indices": list(child_indices),
        "child_labels": [label],
        "child_generator_ids": [f"gen:raw:{label}"],
        "position_id": int(position_id),
        "recommended_execution_mode": "termwise_product",
        "symmetry_gate": {"checked": True, "passed": True},
    }


def _pauli_labels_commute(left: str, right: str) -> bool:
    anti_count = 0
    for left_symbol, right_symbol in zip(left, right, strict=True):
        if (
            left_symbol == "e"
            or right_symbol == "e"
            or left_symbol == right_symbol
        ):
            continue
        anti_count += 1
    return bool(anti_count % 2 == 0)


def _assert_polynomial_preserves_legal_subspace(
    polynomial: PauliPolynomial,
    *,
    n_ph_max: int,
    total_register_width: int,
) -> None:
    layout = legal_subspace_basis_for_problem(
        problem_key="hh",
        num_sites=2,
        n_ph_max=int(n_ph_max),
        boson_encoding="binary",
        total_register_width=int(total_register_width),
    )
    legal_indices = tuple(int(value) for value in layout["legal_indices"])
    legal_set = set(legal_indices)
    terms = list(polynomial.return_polynomial())
    for basis_index in legal_indices:
        amplitudes: dict[int, complex] = {}
        for term in terms:
            out_index, phase = pauli_action_on_basis_index(
                str(term.pw2strng()),
                basis_index,
            )
            amplitudes[out_index] = amplitudes.get(
                out_index,
                0.0 + 0.0j,
            ) + complex(term.p_coeff) * phase
        assert all(
            abs(amplitude) <= 1e-12 or out_index in legal_set
            for out_index, amplitude in amplitudes.items()
        )


@pytest.mark.parametrize(
    ("n_ph_max", "total_register_width", "qpb", "raw_label", "alias_label"),
    [
        (2, 8, 2, "exeeeeee", "zxeeeeee"),
        (4, 10, 3, "eexeeeeeee", "zexeeeeeee"),
    ],
)
def test_exact_projection_deduplicates_per_parent_and_preserves_raw_lineage(
    n_ph_max: int,
    total_register_width: int,
    qpb: int,
    raw_label: str,
    alias_label: str,
) -> None:
    config = RouteAChildPaddingConfig(
        policy=ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1,
        problem_key="hh",
        num_sites=2,
        n_ph_max=int(n_ph_max),
        boson_encoding="binary",
        total_register_width=int(total_register_width),
    )
    candidates = [
        _child_set(
            label="parent-a::child_set[0]",
            terms=[(raw_label, 1.0)],
            parent_label="parent-a",
            child_indices=[0],
        ),
        _child_set(
            label="parent-a::child_set[1]",
            terms=[(alias_label, -3.0)],
            parent_label="parent-a",
            child_indices=[1],
        ),
        _child_set(
            label="parent-b::child_set[1]",
            terms=[(alias_label, 2.0)],
            parent_label="parent-b",
            child_indices=[1],
        ),
    ]

    projected, telemetry = project_and_deduplicate_runtime_split_child_sets(
        candidates,
        config=config,
        num_sites=2,
        ordering="blocked",
        qpb=int(qpb),
        fixed_num_particles=(1, 1),
    )

    assert len(projected) == 2
    assert telemetry["projection_input_count"] == 3
    assert telemetry["projection_output_count"] == 3
    assert telemetry["projection_zero_rejection_count"] == 0
    assert telemetry["deduplicated_candidate_count"] == 1
    assert telemetry["retained_candidate_count"] == 2
    assert telemetry["all_retained_execution_modes_grouped_exact"] is True
    assert telemetry["raw_term_order_and_coefficients_preserved_in_lineage"] is True
    assert telemetry["deduplication_scope"] == (
        "per_parent_and_position_projected_identity_v1"
    )

    parent_a = next(
        row
        for row in projected
        if row["route_a_child_padding_source_labels"][0].startswith("parent-a")
    )
    parent_b = next(
        row
        for row in projected
        if row["route_a_child_padding_source_labels"][0].startswith("parent-b")
    )
    assert parent_a["route_a_child_padding_source_labels"] == [
        "parent-a::child_set[0]",
        "parent-a::child_set[1]",
    ]
    assert parent_a["route_a_child_padding_source_count"] == 2
    assert parent_b["route_a_child_padding_source_count"] == 1
    lineage = parent_a["route_a_child_padding_source_lineage"]
    assert [row["source_ordinal"] for row in lineage] == [0, 1]
    assert [
        row["raw_serialized_terms_exyz"][0]["coeff_re"] for row in lineage
    ] == pytest.approx([1.0, -3.0])
    assert [row["raw_child_indices"] for row in lineage] == [[0], [1]]
    assert all(
        row["selected_execution_mode"] == "grouped_exact" for row in lineage
    )

    for row in projected:
        assert str(row["candidate_label"]).endswith("::legal_projected")
        assert row["recommended_execution_mode"] == "grouped_exact"
        polynomial = row["candidate_polynomial"]
        assert isinstance(polynomial, PauliPolynomial)
        _assert_polynomial_preserves_legal_subspace(
            polynomial,
            n_ph_max=int(n_ph_max),
            total_register_width=int(total_register_width),
        )
        labels = [
            str(term.pw2strng()) for term in polynomial.return_polynomial()
        ]
        assert all(
            _pauli_labels_commute(left, right)
            for left, right in combinations(labels, 2)
        )
        metadata = row["candidate_generator_metadata"]
        selected_terms = serialize_polynomial_terms_exyz(polynomial)
        assert metadata["compile_metadata"]["serialized_terms_exyz"] == (
            selected_terms
        )
        assert metadata["compile_metadata"]["runtime_split"][
            "padding_projection_policy"
        ] == ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1


def test_exact_projection_reports_zero_candidate_with_full_raw_coefficients() -> None:
    config = RouteAChildPaddingConfig(
        policy=ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1,
        problem_key="hh",
        num_sites=2,
        n_ph_max=2,
        boson_encoding="binary",
        total_register_width=8,
    )
    candidate = _child_set(
        label="parent-a::child_set[0,1]",
        terms=[("exeeeeee", 1.0), ("zxeeeeee", -1.0)],
        parent_label="parent-a",
        child_indices=[0, 1],
    )
    raw_serialized = serialize_polynomial_terms_exyz(
        candidate["candidate_polynomial"]
    )

    projected, telemetry = project_and_deduplicate_runtime_split_child_sets(
        [candidate],
        config=config,
        num_sites=2,
        ordering="blocked",
        qpb=2,
        fixed_num_particles=(1, 1),
    )

    assert projected == []
    assert telemetry["projection_zero_rejection_count"] == 1
    assert telemetry["retained_candidate_count"] == 0
    rejection = telemetry["zero_rejections"][0]
    assert rejection["reason"] == "projection_is_zero"
    assert rejection["raw_child_indices"] == [0, 1]
    assert rejection["raw_serialized_terms_exyz"] == raw_serialized
    assert {
        row["pauli_exyz"]: row["coeff_re"]
        for row in rejection["raw_serialized_terms_exyz"]
    } == pytest.approx({"exeeeeee": 1.0, "zxeeeeee": -1.0})
    assert rejection["raw_term_order"] == "polynomial_iteration_order"


def test_archival_helper_rejects_cutoff_specific_legacy_projection_policy() -> None:
    config = RouteAChildPaddingConfig(
        policy=ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_NPH2_V1,
        problem_key="hh",
        num_sites=2,
        n_ph_max=2,
        boson_encoding="binary",
        total_register_width=8,
    )
    candidate = _child_set(
        label="parent-a::child_set[0]",
        terms=[("exeeeeee", 1.0)],
        parent_label="parent-a",
        child_indices=[0],
    )

    with pytest.raises(ValueError, match="cutoff-generic policy"):
        project_and_deduplicate_runtime_split_child_sets(
            [candidate],
            config=config,
            num_sites=2,
            ordering="blocked",
            qpb=2,
            fixed_num_particles=(1, 1),
        )
