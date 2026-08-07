from __future__ import annotations

import itertools
from types import SimpleNamespace

import numpy as np
import pytest

import pipelines.static_adapt.route_a_schur_selector as schur_selector_mod
from pipelines.static_adapt.builders.legal_subspace_filter import (
    legal_subspace_basis_for_problem,
    pauli_action_on_basis_index,
)
from pipelines.static_adapt.route_a_child_padding import (
    ROUTE_A_CHILD_PADDING_HARD_FILTER_NPH2_V1,
    ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_NPH2_V1,
    ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1,
    ROUTE_A_CHILD_PADDING_UNCHECKED_DIAGNOSTIC_V1,
    RouteAChildPaddingConfig,
    project_route_a_child_polynomial,
)
from pipelines.static_adapt.route_a_funnel import (
    ROUTE_A_FUNNEL_CHILD_12_JOINT_RESPONSE_V2,
    ROUTE_A_FUNNEL_CHILD_12_SCHUR_V1,
    ROUTE_A_FUNNEL_DIRECT_CHILD_PHASE3_V1,
    ROUTE_A_FUNNEL_HIERARCHICAL_CHILD_123_V1,
    ROUTE_A_PHASE0_DISABLED,
    ROUTE_A_PHASE3_POPULATION_CHILD_ONLY_V1,
    ROUTE_A_PHASE3_POPULATION_PARENT_PLUS_CHILD_ABLATION_V1,
    RouteAFunnelConfig,
    run_route_a_child_funnel,
)
from pipelines.static_adapt.route_a_shortlists import (
    CHILD_IDENTITY_POLICY_GLOBAL_PAULI_WORD_V1,
    PAULI_CHILD_IDENTITY_NORMALIZATION_PROJECTIVE_V1,
    deduplicate_child_position_records,
    pauli_child_identity,
)
from pipelines.static_adapt.runtime_split import (
    build_global_child_records_for_parent,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.pauli_words import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _child(
    child_id: int,
    *,
    parent: str,
    position: int,
    score: float,
) -> dict[str, object]:
    words = ("xeee", "yeee", "zxee", "zyxe", "zzxe", "zzzx")
    word = words[int(child_id)]
    return {
        "candidate_label": f"{parent}::child:{child_id}",
        "candidate_term": AnsatzTerm(
            label=f"{parent}::child:{child_id}",
            polynomial=PauliPolynomial(
                "JW",
                [PauliTerm(len(word), ps=word, pc=1.0)],
            ),
        ),
        "candidate_pool_index": int(child_id),
        "position_id": int(position),
        "runtime_split_mode": "shortlist_pauli_children_v1",
        "runtime_split_parent_label": str(parent),
        "phase1_active_score": float(score),
        "phase2_raw_score": float(score),
        "full_v2_score": float(score),
        "phase3_tie_break_score": 0.0,
        "simple_score": float(score),
    }


def _parent(score: float = 100.0) -> dict[str, object]:
    return {
        "candidate_label": "macro-parent",
        "candidate_pool_index": 99,
        "position_id": 0,
        "runtime_split_mode": "off",
        "phase1_active_score": float(score),
        "phase2_raw_score": float(score),
        "full_v2_score": float(score),
        "phase3_tie_break_score": 0.0,
        "simple_score": float(score),
    }


def _direction_child(
    terms: list[tuple[str, complex]],
    *,
    parent: str,
    position: int,
    score: float,
    pool_index: int,
) -> dict[str, object]:
    label = f"{parent}::direction:{pool_index}"
    return {
        "candidate_label": label,
        "candidate_term": AnsatzTerm(
            label=label,
            polynomial=PauliPolynomial(
                "JW",
                [
                    PauliTerm(len(word), ps=word, pc=coefficient)
                    for word, coefficient in terms
                ],
            ),
        ),
        "candidate_pool_index": int(pool_index),
        "position_id": int(position),
        "runtime_split_mode": "shortlist_pauli_children_v1",
        "runtime_split_parent_label": str(parent),
        "phase1_active_score": float(score),
        "phase2_raw_score": float(score),
        "full_v2_score": float(score),
        "phase3_tie_break_score": 0.0,
        "simple_score": float(score),
    }


def _population() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for child_id in range(6):
        parent = "parent-a" if child_id < 3 else "parent-b"
        for position in (0, 1):
            records.append(
                _child(
                    child_id,
                    parent=parent,
                    position=position,
                    score=float(20 - child_id - position / 10),
                )
            )
    return records


def test_hierarchical_child_funnel_runs_123_without_lanes() -> None:
    full_evaluation_keys: list[tuple[str, int]] = []

    def full_record_evaluator(record):  # noqa: ANN001 - test callback
        full_evaluation_keys.append(
            (str(record["candidate_label"]), int(record["position_id"]))
        )
        return dict(record)

    result = run_route_a_child_funnel(
        _population(),
        config=RouteAFunnelConfig(
            mode=ROUTE_A_FUNNEL_HIERARCHICAL_CHILD_123_V1,
            population_mode=ROUTE_A_PHASE3_POPULATION_CHILD_ONLY_V1,
            child_identity_policy=CHILD_IDENTITY_POLICY_GLOBAL_PAULI_WORD_V1,
            child_phase1_cap=4,
            child_phase2_cap=3,
            child_phase3_cap=2,
        ),
        full_record_evaluator=full_record_evaluator,
    )

    assert result.telemetry["stage_order"] == [
        "child_phase1",
        "child_phase2",
        "child_phase3",
    ]
    assert len(result.child_phase1_records) == 8
    assert len(result.child_phase2_records) == 6
    assert len(result.child_phase3_records) == 4
    assert all(
        stage["physical_lanes_applied"] is False
        and stage["parent_family_quota_applied"] is False
        for stage in result.telemetry["stages"]
    )
    assert result.query_work["N_grad_record_proxy"] == 12
    assert result.query_work["N_metric_record_proxy"] == 8
    assert result.query_work["N_grad_probe"] == 12
    assert result.query_work["N_metric_probe"] == 8
    assert [event.event_kind for event in result.query_events] == [
        "route_a_child_phase1_gradient",
        "route_a_child_phase2_metric",
        "route_a_child_phase3_metric",
    ]
    assert [event.reused_record_count for event in result.query_events] == [0, 8, 6]
    assert [len(event.records) for event in result.query_events] == [12, 8, 0]
    assert len(result.query_events[2].reused_records) == 6
    assert len(full_evaluation_keys) == 8
    assert result.telemetry["full_record_evaluation_count"] == 8


def test_canonical_child_12_funnel_skips_phase3_and_exposes_phase2_selection() -> None:
    result = run_route_a_child_funnel(
        _population(),
        config=RouteAFunnelConfig(
            mode=ROUTE_A_FUNNEL_CHILD_12_SCHUR_V1,
            phase0_policy=ROUTE_A_PHASE0_DISABLED,
            population_mode=ROUTE_A_PHASE3_POPULATION_CHILD_ONLY_V1,
            child_identity_policy=CHILD_IDENTITY_POLICY_GLOBAL_PAULI_WORD_V1,
            child_phase1_cap=4,
            child_phase2_cap=3,
        ),
        full_record_evaluator=lambda record: dict(record),
        phase2_population_evaluator=lambda _records: pytest.fail(
            "legacy child_12 route must not invoke the joint-response evaluator"
        ),
    )

    assert result.telemetry["phase0_executed"] is False
    assert result.telemetry["macro_phase3_skipped"] is True
    assert result.telemetry["child_phase3_skipped"] is True
    assert result.telemetry["stage_order"] == ["child_phase1", "child_phase2"]
    assert result.child_phase3_records == ()
    assert result.selection_records == result.child_phase2_records
    assert "phase2_selector_mode" not in result.telemetry["config"]
    assert "phase2_selector" not in result.telemetry
    assert [event.phase for event in result.query_events] == ["phase1", "phase2"]
    assert all(
        stage["physical_lanes_applied"] is False
        and stage["parent_family_quota_applied"] is False
        for stage in result.telemetry["stages"]
    )


def test_experimental_joint_response_changes_only_phase2_authority() -> None:
    population = _population()
    common = dict(
        phase0_policy=ROUTE_A_PHASE0_DISABLED,
        population_mode=ROUTE_A_PHASE3_POPULATION_CHILD_ONLY_V1,
        child_identity_policy=CHILD_IDENTITY_POLICY_GLOBAL_PAULI_WORD_V1,
        child_phase1_cap=4,
        child_phase2_cap=2,
    )
    legacy = run_route_a_child_funnel(
        population,
        config=RouteAFunnelConfig(
            mode=ROUTE_A_FUNNEL_CHILD_12_SCHUR_V1,
            **common,
        ),
        full_record_evaluator=lambda record: dict(record),
    )

    def _joint_response(records):  # noqa: ANN001, ANN202
        evaluated = []
        for record in records:
            legacy_score = float(record["phase2_raw_score"])
            evaluated.append(
                {
                    **dict(record),
                    "phase2_legacy_product_score": legacy_score,
                    "phase2_raw_score": -legacy_score,
                    "phase2_selector_mode": "joint_response_singleton_v1",
                }
            )
        return SimpleNamespace(
            records=tuple(evaluated),
            telemetry={
                "schema": "route_a_phase2_joint_response_population_v1",
                "scope": "child_phase2",
            },
        )

    experimental = run_route_a_child_funnel(
        population,
        config=RouteAFunnelConfig(
            mode=ROUTE_A_FUNNEL_CHILD_12_JOINT_RESPONSE_V2,
            **common,
        ),
        full_record_evaluator=lambda record: dict(record),
        phase2_population_evaluator=_joint_response,
    )
    experimental_repeat = run_route_a_child_funnel(
        population,
        config=RouteAFunnelConfig(
            mode=ROUTE_A_FUNNEL_CHILD_12_JOINT_RESPONSE_V2,
            **common,
        ),
        full_record_evaluator=lambda record: dict(record),
        phase2_population_evaluator=_joint_response,
    )

    assert [
        record["candidate_label"] for record in experimental.child_phase1_records
    ] == [record["candidate_label"] for record in legacy.child_phase1_records]
    legacy_order = [
        (record["candidate_label"], record["position_id"])
        for record in legacy.child_phase2_records
    ]
    experimental_order = [
        (record["candidate_label"], record["position_id"])
        for record in experimental.child_phase2_records
    ]
    assert legacy_order == [
        ("parent-a::child:0", 0),
        ("parent-a::child:0", 1),
        ("parent-a::child:1", 0),
        ("parent-a::child:1", 1),
    ]
    assert experimental_order == [
        ("parent-b::child:3", 1),
        ("parent-b::child:3", 0),
        ("parent-a::child:2", 1),
        ("parent-a::child:2", 0),
    ]
    assert [
        (record["candidate_label"], record["position_id"])
        for record in experimental_repeat.child_phase2_records
    ] == experimental_order
    assert experimental.query_work == legacy.query_work
    assert experimental.telemetry["phase2_selector"]["active"] is True
    assert experimental.telemetry["phase2_selector"]["scope"] == "child_phase2"
    assert experimental.telemetry["child_phase3_skipped"] is True
    assert experimental.selection_records == experimental.child_phase2_records


def test_experimental_joint_response_requires_typed_population_evaluator() -> None:
    with pytest.raises(ValueError, match="typed Phase-II population evaluator"):
        run_route_a_child_funnel(
            _population(),
            config=RouteAFunnelConfig(
                mode=ROUTE_A_FUNNEL_CHILD_12_JOINT_RESPONSE_V2,
                phase0_policy=ROUTE_A_PHASE0_DISABLED,
                population_mode=ROUTE_A_PHASE3_POPULATION_CHILD_ONLY_V1,
                child_identity_policy=CHILD_IDENTITY_POLICY_GLOBAL_PAULI_WORD_V1,
                child_phase1_cap=4,
                child_phase2_cap=2,
            ),
            full_record_evaluator=lambda record: dict(record),
        )


def test_children_from_both_parents_compete_globally_and_siblings_can_survive() -> None:
    result = run_route_a_child_funnel(
        _population(),
        config=RouteAFunnelConfig(
            mode=ROUTE_A_FUNNEL_DIRECT_CHILD_PHASE3_V1,
            child_identity_policy=CHILD_IDENTITY_POLICY_GLOBAL_PAULI_WORD_V1,
            child_phase3_cap=4,
        ),
    )

    parents = {
        str(record["runtime_split_parent_label"])
        for record in result.child_phase3_records
    }
    assert parents == {"parent-a", "parent-b"}
    assert sum(
        str(record["runtime_split_parent_label"]) == "parent-a"
        for record in result.child_phase3_records
    ) >= 2
    assert result.telemetry["stage_order"] == ["child_phase3"]
    assert [event.probe_role for event in result.query_events] == [
        "gradient",
        "metric",
    ]
    assert result.query_events[1].reused_record_count == len(result.child_population)


def test_parent_plus_child_isolated_behind_explicit_ablation() -> None:
    child_only = run_route_a_child_funnel(
        _population(),
        parent_records=[_parent()],
        config=RouteAFunnelConfig(
            mode=ROUTE_A_FUNNEL_DIRECT_CHILD_PHASE3_V1,
            population_mode=ROUTE_A_PHASE3_POPULATION_CHILD_ONLY_V1,
            child_phase3_cap=1,
        ),
    )
    ablation = run_route_a_child_funnel(
        _population(),
        parent_records=[_parent()],
        config=RouteAFunnelConfig(
            mode=ROUTE_A_FUNNEL_DIRECT_CHILD_PHASE3_V1,
            population_mode=ROUTE_A_PHASE3_POPULATION_PARENT_PLUS_CHILD_ABLATION_V1,
            child_phase3_cap=1,
        ),
    )

    assert all(
        record["route_a_candidate_representation"] == "pauli_child"
        for record in child_only.child_phase3_records
    )
    assert ablation.child_phase3_records[0]["route_a_candidate_representation"] == "parent_macro"
    assert child_only.telemetry["parent_records_admissible"] is False
    assert ablation.telemetry["parent_records_admissible"] is True


def test_global_child_identity_normalizes_scalar_sign_phase_and_keeps_positions() -> None:
    records = [
        _direction_child(
            [("xeee", 1.0)], parent="parent-a", position=0, score=5.0, pool_index=1
        ),
        _direction_child(
            [("xeee", -2.0)], parent="parent-b", position=0, score=4.0, pool_index=2
        ),
        _direction_child(
            [("xeee", 3.0j)], parent="parent-c", position=0, score=3.0, pool_index=3
        ),
        _direction_child(
            [("xeee", 1.0)], parent="parent-d", position=1, score=2.0, pool_index=4
        ),
        _direction_child(
            [("yeee", 1.0)], parent="parent-a", position=0, score=1.0, pool_index=5
        ),
    ]

    deduplicated, telemetry = deduplicate_child_position_records(
        records,
        score_key="phase1_active_score",
        identity_policy=CHILD_IDENTITY_POLICY_GLOBAL_PAULI_WORD_V1,
    )

    assert len(deduplicated) == 3
    x_rows = [
        row
        for row in deduplicated
        if pauli_child_identity(row) == pauli_child_identity(records[0])
    ]
    assert {int(row["position_id"]) for row in x_rows} == {0, 1}
    position_zero = next(row for row in x_rows if int(row["position_id"]) == 0)
    assert position_zero["route_a_child_parent_labels"] == [
        "parent-a",
        "parent-b",
        "parent-c",
    ]
    assert int(position_zero["route_a_child_parent_count"]) == 3
    assert pauli_child_identity(records[0]) == pauli_child_identity(records[1])
    assert pauli_child_identity(records[0]) == pauli_child_identity(records[2])
    assert pauli_child_identity(records[0]) != pauli_child_identity(records[4])
    assert telemetry["duplicate_record_count"] == 2
    assert telemetry["unique_child_identity_count"] == 2
    assert telemetry["identity_normalization"] == (
        PAULI_CHILD_IDENTITY_NORMALIZATION_PROJECTIVE_V1
    )


def test_nph2_child_padding_filter_runs_after_global_dedup_before_child_phase1() -> None:
    records = [
        _direction_child(
            [("eeexeeee", 1.0)],
            parent="parent-a",
            position=0,
            score=10.0,
            pool_index=1,
        ),
        _direction_child(
            [("eeexeeee", -2.0)],
            parent="parent-b",
            position=0,
            score=9.0,
            pool_index=2,
        ),
        _direction_child(
            [("eeexeeee", 1.0)],
            parent="parent-c",
            position=1,
            score=8.0,
            pool_index=3,
        ),
        _direction_child(
            [("eeeeeeex", 1.0)],
            parent="parent-a",
            position=0,
            score=7.0,
            pool_index=4,
        ),
        _direction_child(
            [("eeezeeee", 1.0)],
            parent="parent-a",
            position=0,
            score=6.0,
            pool_index=5,
        ),
    ]
    config = RouteAFunnelConfig(
        mode=ROUTE_A_FUNNEL_CHILD_12_SCHUR_V1,
        phase0_policy=ROUTE_A_PHASE0_DISABLED,
        child_identity_policy=CHILD_IDENTITY_POLICY_GLOBAL_PAULI_WORD_V1,
        child_phase1_cap=10,
        child_phase2_cap=10,
        child_padding=RouteAChildPaddingConfig(
            policy=ROUTE_A_CHILD_PADDING_HARD_FILTER_NPH2_V1,
            problem_key="hh",
            num_sites=2,
            n_ph_max=2,
            boson_encoding="binary",
            total_register_width=8,
        ),
    )

    result = run_route_a_child_funnel(records, config=config)

    population = result.telemetry["child_population"]
    padding = result.telemetry["child_padding_filter"]
    assert population["input_record_count"] == 5
    assert population["deduplicated_record_count"] == 4
    assert population["pre_padding_filter_record_count"] == 4
    assert population["post_padding_filter_record_count"] == 2
    assert population["pre_padding_filter_unique_child_identity_count"] == 3
    assert population["post_padding_filter_unique_child_identity_count"] == 2
    assert padding["applied_after_global_child_deduplication"] is True
    assert padding["applied_before_child_phase1"] is True
    assert padding["rejected_record_count"] == 2
    assert padding["rejected_identity_count"] == 1
    assert len(result.child_population) == 2
    assert len(result.child_phase1_records) == 2
    assert len(result.child_phase2_records) == 2
    assert result.query_work["N_grad_probe"] == 2
    assert result.query_work["N_metric_probe"] == 2
    rejected_position_zero = next(
        row for row in padding["rejected_records"] if row["position_id"] == 0
    )
    assert rejected_position_zero["parent_labels"] == ["parent-a", "parent-b"]
    assert rejected_position_zero["reason"] == (
        "pauli_word_maps_legal_codeword_to_padding"
    )


def test_nph2_child_padding_unchecked_mode_is_diagnostic_only() -> None:
    result = run_route_a_child_funnel(
        [
            _direction_child(
                [("eeexeeee", 1.0)],
                parent="parent-a",
                position=0,
                score=1.0,
                pool_index=1,
            )
        ],
        config=RouteAFunnelConfig(
            mode=ROUTE_A_FUNNEL_CHILD_12_SCHUR_V1,
            phase0_policy=ROUTE_A_PHASE0_DISABLED,
            child_identity_policy=CHILD_IDENTITY_POLICY_GLOBAL_PAULI_WORD_V1,
            child_padding=RouteAChildPaddingConfig(
                policy=ROUTE_A_CHILD_PADDING_UNCHECKED_DIAGNOSTIC_V1,
            ),
        ),
    )

    assert len(result.child_population) == 1
    assert result.telemetry["child_padding_filter"]["active"] is False
    assert result.telemetry["child_padding_filter"]["reason"] == (
        "diagnostic_compatibility_mode"
    )


def test_nph2_exact_projection_preserves_transition_as_grouped_child() -> None:
    config = RouteAChildPaddingConfig(
        policy=ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_NPH2_V1,
        problem_key="hh",
        num_sites=2,
        n_ph_max=2,
        boson_encoding="binary",
        total_register_width=8,
    )
    raw = PauliPolynomial(
        "JW",
        [PauliTerm(8, ps="eeexeeee", pc=1.0)],
    )

    projected, projection = project_route_a_child_polynomial(
        raw,
        config=config,
    )

    assert projected is not None
    coefficients = {
        str(term.pw2strng()): complex(term.p_coeff)
        for term in projected.return_polynomial()
    }
    assert coefficients == pytest.approx(
        {
            "eeexeeee": 0.5,
            "eezxeeee": 0.5,
        }
    )
    assert projection["recommended_execution_mode"] == "grouped_exact"
    record = _direction_child(
        [("eeexeeee", 1.0)],
        parent="parent-a",
        position=0,
        score=1.0,
        pool_index=1,
    )
    record["candidate_term"] = AnsatzTerm(
        label="projected-child",
        polynomial=projected,
        execution_mode="grouped_exact",
    )
    result = run_route_a_child_funnel(
        [record],
        config=RouteAFunnelConfig(
            mode=ROUTE_A_FUNNEL_CHILD_12_SCHUR_V1,
            phase0_policy=ROUTE_A_PHASE0_DISABLED,
            child_identity_policy=CHILD_IDENTITY_POLICY_GLOBAL_PAULI_WORD_V1,
            child_padding=config,
        ),
    )
    assert len(result.child_population) == 1
    assert result.telemetry["child_padding_filter"]["rejected_record_count"] == 0
    assert result.telemetry["child_padding_filter"]["validation_mode"] == (
        "grouped_polynomial_legal_action_v1"
    )


def test_cutoff_generic_exact_projection_supports_nph4() -> None:
    config = RouteAChildPaddingConfig(
        policy=ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1,
        problem_key="hh",
        num_sites=2,
        n_ph_max=4,
        boson_encoding="binary",
        total_register_width=10,
    )
    raw = PauliPolynomial(
        "JW",
        [PauliTerm(10, ps="eexeeeeeee", pc=1.0)],
    )

    projected, projection = project_route_a_child_polynomial(raw, config=config)

    assert projected is not None
    assert projected.count_number_terms() > 1
    assert projection["recommended_execution_mode"] == "grouped_exact"
    record = _direction_child(
        [("eexeeeeeee", 1.0)],
        parent="parent-a",
        position=0,
        score=1.0,
        pool_index=1,
    )
    record["candidate_term"] = AnsatzTerm(
        label="projected-nph4-child",
        polynomial=projected,
        execution_mode="grouped_exact",
    )
    result = run_route_a_child_funnel(
        [record],
        config=RouteAFunnelConfig(
            mode=ROUTE_A_FUNNEL_CHILD_12_SCHUR_V1,
            phase0_policy=ROUTE_A_PHASE0_DISABLED,
            child_identity_policy=CHILD_IDENTITY_POLICY_GLOBAL_PAULI_WORD_V1,
            child_padding=config,
        ),
    )
    assert len(result.child_population) == 1
    assert result.telemetry["child_padding_filter"]["rejected_record_count"] == 0


def test_legacy_nph2_projection_policy_rejects_nph4() -> None:
    with pytest.raises(ValueError, match="requires n_ph_max=2"):
        RouteAChildPaddingConfig(
            policy=ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_NPH2_V1,
            problem_key="hh",
            num_sites=2,
            n_ph_max=4,
            boson_encoding="binary",
            total_register_width=10,
        )


def test_nph2_projection_matches_legal_block_for_every_local_pauli() -> None:
    config = RouteAChildPaddingConfig(
        policy=ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_NPH2_V1,
        problem_key="hh",
        num_sites=2,
        n_ph_max=2,
        boson_encoding="binary",
        total_register_width=8,
    )
    layout = legal_subspace_basis_for_problem(
        problem_key="hh",
        num_sites=2,
        n_ph_max=2,
        boson_encoding="binary",
        total_register_width=8,
    )
    legal_indices = tuple(int(value) for value in layout["legal_indices"])
    legal_set = set(legal_indices)

    for local_symbols in itertools.product("exyz", repeat=2):
        raw_label = "ee" + "".join(local_symbols) + "eeee"
        projected, _telemetry = project_route_a_child_polynomial(
            PauliPolynomial(
                "JW",
                [PauliTerm(8, ps=raw_label, pc=1.0)],
            ),
            config=config,
        )
        assert projected is not None
        for basis_index in legal_indices:
            raw_out, raw_phase = pauli_action_on_basis_index(
                raw_label,
                basis_index,
            )
            expected = (
                {raw_out: raw_phase}
                if raw_out in legal_set
                else {}
            )
            actual: dict[int, complex] = {}
            for term in projected.return_polynomial():
                out_index, phase = pauli_action_on_basis_index(
                    str(term.pw2strng()),
                    basis_index,
                )
                actual[out_index] = actual.get(out_index, 0.0 + 0.0j) + (
                    complex(term.p_coeff) * phase
                )
            actual = {
                index: amplitude
                for index, amplitude in actual.items()
                if abs(amplitude) > 1e-12
            }
            assert set(actual).issubset(legal_set)
            assert set(actual) == set(expected)
            for out_index in expected:
                assert actual[out_index] == pytest.approx(
                    expected[out_index],
                    abs=1e-12,
                )


def test_global_child_builder_scores_projected_operator_not_raw_singleton() -> None:
    config = RouteAChildPaddingConfig(
        policy=ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_NPH2_V1,
        problem_key="hh",
        num_sites=2,
        n_ph_max=2,
        boson_encoding="binary",
        total_register_width=8,
    )
    evaluated_terms: list[AnsatzTerm] = []

    def evaluate_candidate(**kwargs):  # noqa: ANN003 - test callback
        term = kwargs["candidate_term"]
        evaluated_terms.append(term)
        return {
            "candidate_label": str(kwargs["candidate_label"]),
            "candidate_term": term,
            "candidate_pool_index": len(evaluated_terms),
            "position_id": 0,
            "phase1_active_score": 1.0,
            "phase2_raw_score": 1.0,
            "simple_score": 1.0,
        }

    records, telemetry = build_global_child_records_for_parent(
        parent_label="parent",
        parent_term=AnsatzTerm(
            label="parent",
            polynomial=PauliPolynomial(
                "JW",
                [
                    PauliTerm(8, ps="eeexeeee", pc=1.0),
                    PauliTerm(8, ps="eeeeeeex", pc=1.0),
                ],
            ),
        ),
        parent_family_id="test",
        parent_generator_metadata=None,
        parent_symmetry_spec=None,
        child_set_symmetry_policy="off",
        subset_sizes=(1,),
        num_sites=2,
        ordering="blocked",
        qpb=2,
        problem_key="hh",
        fixed_num_particles=None,
        evaluate_candidate=evaluate_candidate,
        child_padding_config=config,
    )

    assert records
    assert len(evaluated_terms) == len(records)
    assert all(term.execution_mode == "grouped_exact" for term in evaluated_terms)
    assert any(
        term.polynomial.count_number_terms() > 1
        for term in evaluated_terms
    )
    assert all(
        str(record["candidate_label"]).endswith("::legal_projected")
        for record in records
    )
    assert telemetry["child_padding_projection_requested"] is True
    assert telemetry["child_padding_projection_input_count"] == len(records)
    assert telemetry["child_padding_projection_output_count"] == len(records)


def test_global_child_builder_reports_deferred_phase1_as_unmeasured() -> None:
    config = RouteAChildPaddingConfig(
        policy=ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_NPH2_V1,
        problem_key="hh",
        num_sites=2,
        n_ph_max=2,
        boson_encoding="binary",
        total_register_width=8,
    )
    evaluation_count = 0

    def evaluate_candidate(**_kwargs):  # noqa: ANN003 - test callback
        nonlocal evaluation_count
        evaluation_count += 1
        return {}

    records, telemetry = build_global_child_records_for_parent(
        parent_label="parent",
        parent_term=AnsatzTerm(
            label="parent",
            polynomial=PauliPolynomial(
                "JW",
                [
                    PauliTerm(8, ps="eeexeeee", pc=1.0),
                    PauliTerm(8, ps="eeeeeeex", pc=1.0),
                ],
            ),
        ),
        parent_family_id="test",
        parent_generator_metadata=None,
        parent_symmetry_spec=None,
        child_set_symmetry_policy="off",
        subset_sizes=(1,),
        num_sites=2,
        ordering="blocked",
        qpb=2,
        problem_key="hh",
        fixed_num_particles=None,
        evaluate_candidate=evaluate_candidate,
        child_padding_config=config,
        defer_phase1_evaluation=True,
        base_record={"candidate_pool_index": 1, "position_id": 0},
    )

    assert records
    assert evaluation_count == 0
    assert telemetry["staged_child_set_count"] == len(records)
    assert telemetry["phase1_evaluated_child_set_count"] == 0
    assert telemetry["evaluated_child_set_count"] == 0
    assert telemetry[
        "phase1_evaluation_deferred_until_after_global_deduplication"
    ] is True


def test_projected_child_identity_deduplicates_equivalent_raw_paulis_once() -> None:
    config = RouteAChildPaddingConfig(
        policy=ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_NPH2_V1,
        problem_key="hh",
        num_sites=2,
        n_ph_max=2,
        boson_encoding="binary",
        total_register_width=8,
    )
    records = []
    for pool_index, (raw_label, parent, coefficient) in enumerate(
        (
            ("eeexeeee", "parent-a", 0.5),
            ("eezxeeee", "parent-b", -1.5),
        ),
        start=1,
    ):
        projected, _telemetry = project_route_a_child_polynomial(
            PauliPolynomial(
                "JW",
                [PauliTerm(8, ps=raw_label, pc=float(coefficient))],
            ),
            config=config,
        )
        assert projected is not None
        record = _direction_child(
            [(raw_label, 1.0)],
            parent=parent,
            position=0,
            score=float(3 - pool_index),
            pool_index=pool_index,
        )
        record["candidate_term"] = AnsatzTerm(
            label=f"{parent}::projected",
            polynomial=projected,
            execution_mode="grouped_exact",
        )
        records.append(record)

    evaluated: list[dict[str, object]] = []
    evaluated_norms: list[float] = []

    def _evaluate_phase1(record):  # noqa: ANN001 - test callback
        evaluated.append(dict(record))
        evaluated_norms.append(
            float(
                sum(
                    abs(complex(term.p_coeff)) ** 2
                    for term in record[
                        "candidate_term"
                    ].polynomial.return_polynomial()
                )
                ** 0.5
            )
        )
        return {
            **dict(record),
            "phase1_active_score": 1.0,
            "simple_score": 1.0,
            "route_a_child_phase1_evaluation_deferred": False,
        }

    result = run_route_a_child_funnel(
        records,
        config=RouteAFunnelConfig(
            mode=ROUTE_A_FUNNEL_CHILD_12_SCHUR_V1,
            phase0_policy=ROUTE_A_PHASE0_DISABLED,
            child_identity_policy=CHILD_IDENTITY_POLICY_GLOBAL_PAULI_WORD_V1,
            child_padding=config,
        ),
        phase1_record_evaluator=_evaluate_phase1,
    )

    assert len(result.child_population) == 1
    assert len(evaluated) == 1
    assert evaluated_norms == pytest.approx([1.0], abs=1e-12)
    assert result.child_population[0]["route_a_child_parent_labels"] == [
        "parent-a",
        "parent-b",
    ]
    assert result.query_work["N_grad_probe"] == 1
    assert result.query_work["N_metric_probe"] == 1
    assert result.telemetry["child_phase1_evaluation"] == {
        "deferred_until_after_global_deduplication": True,
        "global_deduplication_complete_before_evaluation": True,
        "padding_policy_complete_before_evaluation": True,
        "cooldown_filter_complete_before_evaluation": True,
        "input_record_count": 1,
        "evaluated_record_count": 1,
    }


def test_global_child_identity_preserves_multiword_relative_coefficients() -> None:
    base = _direction_child(
        [("xe", 1.0), ("ey", 2.0j)],
        parent="parent-a",
        position=0,
        score=1.0,
        pool_index=1,
    )
    scaled_reordered = _direction_child(
        [("ey", 6.0), ("xe", -3.0j)],
        parent="parent-b",
        position=0,
        score=1.0,
        pool_index=2,
    )
    different_relative_coefficients = _direction_child(
        [("xe", 1.0), ("ey", 3.0j)],
        parent="parent-c",
        position=0,
        score=1.0,
        pool_index=3,
    )

    assert pauli_child_identity(base) == pauli_child_identity(scaled_reordered)
    assert pauli_child_identity(base) != pauli_child_identity(
        different_relative_coefficients
    )


def test_global_dedup_precedes_child_measurement_and_charges_once() -> None:
    records = [
        _direction_child(
            [("xeee", coefficient)],
            parent=parent,
            position=0,
            score=score,
            pool_index=index,
        )
        for index, (parent, coefficient, score) in enumerate(
            (
                ("parent-a", 1.0, 3.0),
                ("parent-b", -1.0, 2.0),
                ("parent-c", 2.0j, 1.0),
            ),
            start=1,
        )
    ]
    evaluated: list[dict[str, object]] = []

    def _evaluate(record):  # noqa: ANN001 - test callback
        evaluated.append(dict(record))
        return dict(record)

    result = run_route_a_child_funnel(
        records,
        config=RouteAFunnelConfig(
            mode=ROUTE_A_FUNNEL_CHILD_12_SCHUR_V1,
            phase0_policy=ROUTE_A_PHASE0_DISABLED,
            population_mode=ROUTE_A_PHASE3_POPULATION_CHILD_ONLY_V1,
            child_identity_policy=CHILD_IDENTITY_POLICY_GLOBAL_PAULI_WORD_V1,
            child_phase1_cap=10,
            child_phase2_cap=10,
        ),
        full_record_evaluator=_evaluate,
    )

    assert len(result.child_population) == 1
    assert len(result.child_phase1_records) == 1
    assert len(result.child_phase2_records) == 1
    assert len(result.selection_records) == 1
    assert len(evaluated) == 1
    assert len(result.query_events[0].records) == 1
    assert len(result.query_events[1].records) == 1
    assert result.query_work["N_grad_probe"] == 1
    assert result.query_work["N_metric_probe"] == 1
    population = result.telemetry["child_population"]
    assert population["applied_before_child_phase1"] is True
    assert population["applied_before_search_pool_construction"] is True
    assert population["applied_before_joint_gram_hessian_measurement"] is True
    assert result.selection_records[0]["route_a_child_parent_labels"] == [
        "parent-a",
        "parent-b",
        "parent-c",
    ]


def test_schur_selector_safety_dedup_precedes_workspace_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        _direction_child(
            [("xeee", 1.0)], parent="parent-a", position=0, score=2.0, pool_index=1
        ),
        _direction_child(
            [("xeee", -1.0)], parent="parent-b", position=0, score=1.0, pool_index=2
        ),
    ]
    captured: list[dict[str, object]] = []

    def _capture(rows, **_kwargs):  # noqa: ANN001 - test callback
        captured.extend(dict(row) for row in rows)
        return [], {"selection_mode": "combinatorial_reduced_plane"}

    monkeypatch.setattr(
        schur_selector_mod,
        "select_phase2_batch_record_proposals",
        _capture,
    )
    monkeypatch.setattr(
        schur_selector_mod,
        "route_a_schur_score_config",
        lambda base, *, config: base,
    )
    _, summary = schur_selector_mod.select_route_a_schur_proposals(
        records,
        config=schur_selector_mod.RouteASchurSelectorConfig(),
        score_config=object(),
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=np.asarray([1.0], dtype=complex),
        psi_state=np.asarray([1.0], dtype=complex),
        h_compiled=object(),
        novelty_oracle=object(),
        curvature_oracle=object(),
    )

    assert len(captured) == 1
    assert captured[0]["route_a_child_parent_labels"] == ["parent-a", "parent-b"]
    assert summary["child_phase2_survivor_count_input"] == 2
    assert summary["child_phase2_survivor_count"] == 1
    assert summary["global_dedup_applied_before_search_pool"] is True
    assert summary["global_dedup_applied_before_joint_geometry_workspace"] is True


def test_schur_selector_exhaustion_retry_expands_l25_to_all_and_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        _direction_child(
            [("xeee", 1.0)], parent="parent-a", position=0, score=2.0, pool_index=1
        )
    ]
    observed_pool_sizes: list[int] = []

    def _score_config(base, *, config):  # noqa: ANN001 - test callback
        del base
        return SimpleNamespace(batch_search_pool_size=int(config.batch_search_pool_size))

    def _empty(rows, *, cfg, **_kwargs):  # noqa: ANN001 - test callback
        assert rows
        observed_pool_sizes.append(int(cfg.batch_search_pool_size))
        return [], {"selection_mode": "combinatorial_reduced_plane"}

    monkeypatch.setattr(
        schur_selector_mod,
        "route_a_schur_score_config",
        _score_config,
    )
    monkeypatch.setattr(
        schur_selector_mod,
        "select_phase2_batch_record_proposals",
        _empty,
    )
    proposals, summary = schur_selector_mod.select_route_a_schur_proposals(
        records,
        config=schur_selector_mod.RouteASchurSelectorConfig(
            batch_search_pool_size=25,
            exhaustion_retry_policy=(
                schur_selector_mod.ROUTE_A_SELECTOR_EXHAUSTION_EXPAND_ALL_THEN_FORCE_SINGLETON_V1
            ),
        ),
        score_config=object(),
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=np.asarray([1.0], dtype=complex),
        psi_state=np.asarray([1.0], dtype=complex),
        h_compiled=object(),
        novelty_oracle=object(),
        curvature_oracle=object(),
    )

    assert proposals == []
    assert observed_pool_sizes == [25, 0]
    retry = summary["exhaustion_retry"]
    assert retry["triggered"] is True
    assert retry["expanded_to_all_child_phase2_survivors"] is True
    assert retry["forced_singleton_required"] is True


def test_schur_selector_canonical_default_disables_additivity_penalty() -> None:
    config = schur_selector_mod.RouteASchurSelectorConfig()

    assert config.additivity_policy == schur_selector_mod.ROUTE_A_ADDITIVITY_OFF
    assert config.lambda_add == 0.0


def test_hierarchical_phase3_pre_rescore_precedes_exact_record_evaluation() -> None:
    observed: list[str] = []

    def _phase3_pre(rows):  # noqa: ANN001 - test callback
        observed.append("pre")
        return [{**dict(row), "phase3_family_normalized": True} for row in rows]

    def _phase3_evaluate(row):  # noqa: ANN001 - test callback
        assert row["phase3_family_normalized"] is True
        observed.append("evaluate")
        return {**dict(row), "full_v2_score": 3.0}

    result = run_route_a_child_funnel(
        [_child(0, parent="parent-a", position=0, score=1.0)],
        config=RouteAFunnelConfig(
            mode=ROUTE_A_FUNNEL_HIERARCHICAL_CHILD_123_V1,
            population_mode=ROUTE_A_PHASE3_POPULATION_CHILD_ONLY_V1,
            child_identity_policy=CHILD_IDENTITY_POLICY_GLOBAL_PAULI_WORD_V1,
            child_phase1_cap=1,
            child_phase2_cap=1,
            child_phase3_cap=1,
        ),
        phase2_record_evaluator=lambda row: dict(row),
        phase3_pre_evaluation_rescorer=_phase3_pre,
        phase3_record_evaluator=_phase3_evaluate,
    )

    assert observed == ["pre", "evaluate"]
    assert result.selection_records[0]["full_v2_score"] == pytest.approx(3.0)
