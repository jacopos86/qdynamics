from __future__ import annotations

from dataclasses import replace
import math

from pipelines.static_adapt.route_a_shortlists import (
    CHILD_IDENTITY_POLICY_GLOBAL_PAULI_WORD_V1,
    CHILD_IDENTITY_POLICY_PARENT_QUALIFIED_LEGACY_V1,
    ROUTE_A_SHORTLIST_UNIT_MACRO_OPERATOR,
    deduplicate_child_position_records,
    expand_selected_identities,
    identity_population,
    macro_operator_identity,
)
from pipelines.scaffold.hh_continuation_scoring import FullScoreConfig
from pipelines.static_adapt.phase_shortlists import (
    PhaseShortlistRuntime,
    _phase1_lane_shortlist_with_legacy_hook,
    _phase2_lane_health_shortlist_with_legacy_hook,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.pauli_words import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _record(pool_index: int, position: int, score: float) -> dict[str, object]:
    return {
        "candidate_pool_index": int(pool_index),
        "candidate_label": f"macro:{pool_index}",
        "position_id": int(position),
        "score": float(score),
    }


def _child_record(
    *,
    label: str,
    pauli_word: str,
    position: int,
    parent: str,
    score: float,
    coefficient: float = 1.0,
) -> dict[str, object]:
    term = AnsatzTerm(
        label=str(label),
        polynomial=PauliPolynomial(
            "JW",
            [
                PauliTerm(
                    len(pauli_word),
                    ps=str(pauli_word),
                    pc=float(coefficient),
                )
            ],
        ),
    )
    return {
        "candidate_label": str(label),
        "candidate_term": term,
        "position_id": int(position),
        "runtime_split_parent_label": str(parent),
        "score": float(score),
    }


def test_identity_shortlist_caps_operators_and_preserves_positions() -> None:
    records = [
        _record(0, 0, 0.8),
        _record(0, 1, 0.7),
        _record(1, 0, 0.9),
        _record(1, 1, 0.1),
        _record(2, 0, 0.6),
        _record(2, 1, 0.5),
    ]
    population = identity_population(
        records,
        identity_key=macro_operator_identity,
        score_key="score",
    )

    selected = population.representatives[:2]
    expanded = expand_selected_identities(
        population,
        selected,
        shortlist_flag="selected",
        shortlist_unit=ROUTE_A_SHORTLIST_UNIT_MACRO_OPERATOR,
    )

    assert population.identity_count == 3
    assert population.record_count == 6
    assert {row["candidate_pool_index"] for row in expanded} == {0, 1}
    assert len(expanded) == 4
    assert {row["route_a_identity_shortlist_size"] for row in expanded} == {2}
    assert all(row["selected"] is True for row in expanded)


def test_global_child_dedup_merges_parents_but_preserves_positions() -> None:
    records = [
        _child_record(
            label="parent-a::child",
            pauli_word="x",
            position=0,
            parent="parent-a",
            score=0.8,
        ),
        _child_record(
            label="parent-b::child",
            pauli_word="x",
            position=0,
            parent="parent-b",
            score=0.9,
        ),
        _child_record(
            label="parent-a::child",
            pauli_word="x",
            position=1,
            parent="parent-a",
            score=0.7,
        ),
    ]

    deduplicated, telemetry = deduplicate_child_position_records(
        records,
        score_key="score",
        identity_policy=CHILD_IDENTITY_POLICY_GLOBAL_PAULI_WORD_V1,
    )

    assert len(deduplicated) == 2
    assert telemetry["duplicate_record_count"] == 1
    assert telemetry["unique_child_identity_count"] == 1
    by_position = {row["position_id"]: row for row in deduplicated}
    assert by_position[0]["candidate_label"] == "parent-b::child"
    assert by_position[0]["route_a_child_parent_labels"] == ["parent-a", "parent-b"]
    assert by_position[1]["route_a_child_parent_labels"] == ["parent-a"]
    assert (
        by_position[0]["route_a_global_pauli_identity"]
        == by_position[1]["route_a_global_pauli_identity"]
    )

    legacy, legacy_telemetry = deduplicate_child_position_records(
        records,
        score_key="score",
        identity_policy=CHILD_IDENTITY_POLICY_PARENT_QUALIFIED_LEGACY_V1,
    )
    assert len(legacy) == 3
    assert legacy_telemetry["unique_child_identity_count"] == 2
    assert len({row["route_a_global_pauli_identity"] for row in legacy}) == 1


def test_global_child_dedup_uses_unit_norm_direction_representative() -> None:
    records = [
        _child_record(
            label="parent-a::child",
            pauli_word="x",
            position=0,
            parent="parent-a",
            score=1.0,
            coefficient=0.5,
        ),
        _child_record(
            label="parent-b::child",
            pauli_word="x",
            position=0,
            parent="parent-b",
            score=2.0,
            coefficient=-1.5,
        ),
    ]

    deduplicated, telemetry = deduplicate_child_position_records(
        records,
        score_key="score",
        identity_policy=CHILD_IDENTITY_POLICY_GLOBAL_PAULI_WORD_V1,
    )

    assert len(deduplicated) == 1
    term = deduplicated[0]["candidate_term"]
    coefficients = [
        complex(pauli_term.p_coeff)
        for pauli_term in term.polynomial.return_polynomial()
    ]
    assert math.sqrt(sum(abs(value) ** 2 for value in coefficients)) == 1.0
    assert coefficients == [1.0 + 0.0j]
    normalization = deduplicated[0][
        "route_a_child_direction_normalization"
    ]
    assert normalization["source_coefficient_l2_norm"] == 1.5
    assert normalization["canonical_anchor_positive_real"] is True
    assert telemetry["canonical_direction_representative"] == (
        "unit_l2_norm_positive_anchor_v1"
    )


def _physical_runtime() -> PhaseShortlistRuntime:
    return PhaseShortlistRuntime(
        phase2_score_cfg=FullScoreConfig(),
        feature_updater=lambda feature, _updates: feature,
        lane_policy_active=True,
        lane_summary={},
        phase1_lane_quota_pressure=0.0,
        phase2_lane_quota_pressure=0.0,
        phase2_lane_rel_threshold=0.9,
        shortlist_lane_route="physical_operator_type",
        shortlist_lane_key="physical_operator_lane",
        shortlist_lanes=("lane_a", "lane_b"),
        shortlist_fallback_lane="lane_b",
        shortlist_lane_health_key_prefix="physical_operator",
    )


def test_physical_phase_shortlists_cap_identities_not_position_records() -> None:
    phase1_records: list[dict[str, object]] = []
    for operator in range(30):
        for position in (0, 1):
            score = float(100 - operator - position / 10)
            phase1_records.append(
                {
                    **_record(operator, position, score),
                    "phase1_active_score": score,
                    "simple_score": score,
                    "physical_operator_lane": (
                        "lane_a" if operator % 2 == 0 else "lane_b"
                    ),
                }
            )

    phase1 = _phase1_lane_shortlist_with_legacy_hook(
        phase1_records,
        runtime=_physical_runtime(),
        score_key="phase1_active_score",
        threshold=1.0e9,
        cap=24,
        frontier_ratio=0.99,
        tie_break_score_key="simple_score",
        shortlist_flag="phase1_shortlisted",
    )
    assert len(phase1) == 48
    assert len({macro_operator_identity(row) for row in phase1}) == 24

    phase2_records = [
        {
            **row,
            "phase2_raw_score": float(row["phase1_active_score"]),
        }
        for row in phase1
    ]
    phase2 = _phase2_lane_health_shortlist_with_legacy_hook(
        phase2_records,
        runtime=_physical_runtime(),
        score_key="phase2_raw_score",
        threshold=1.0e9,
        cap=12,
        frontier_ratio=0.99,
        tie_break_score_key="simple_score",
        shortlist_flag="phase2_shortlisted",
    )
    assert len(phase2) == 24
    assert len({macro_operator_identity(row) for row in phase2}) == 12


def test_historical_physical_shortlists_cap_position_records() -> None:
    records: list[dict[str, object]] = []
    for operator in range(12):
        for position in (0, 1):
            score = float(100 - 2 * operator - position)
            records.append(
                {
                    **_record(operator, position, score),
                    "phase1_active_score": score,
                    "phase2_raw_score": score,
                    "simple_score": score,
                    "physical_operator_lane": (
                        "lane_a" if operator % 2 == 0 else "lane_b"
                    ),
                }
            )

    historical_runtime = replace(
        _physical_runtime(),
        physical_operator_identity_caps_enabled=False,
    )
    phase1 = _phase1_lane_shortlist_with_legacy_hook(
        records,
        runtime=historical_runtime,
        score_key="phase1_active_score",
        threshold=0.0,
        cap=8,
        frontier_ratio=0.0,
        tie_break_score_key="simple_score",
        shortlist_flag="phase1_shortlisted",
    )
    phase2 = _phase2_lane_health_shortlist_with_legacy_hook(
        phase1,
        runtime=historical_runtime,
        score_key="phase2_raw_score",
        threshold=0.0,
        cap=4,
        frontier_ratio=0.0,
        tie_break_score_key="simple_score",
        shortlist_flag="phase2_shortlisted",
    )

    assert len(phase1) == 8
    assert len(phase2) == 4
    assert len({macro_operator_identity(row) for row in phase1}) < len(phase1)
    assert len({macro_operator_identity(row) for row in phase2}) < len(phase2)
    assert all("route_a_shortlist_unit" not in row for row in phase1)
    assert all("route_a_shortlist_unit" not in row for row in phase2)
    assert all(row["physical_operator_lane_live"] is True for row in phase2)
    runtime_summary = historical_runtime.lane_summary[
        "shortlist_runtime"
    ]
    assert runtime_summary["lane_route"] == "physical_operator_type"
    assert runtime_summary["phase1_last_shortlist_size"] == 8
    assert runtime_summary["phase2_last_shortlist_size"] == 4
    assert "phase1_shortlist_unit" not in runtime_summary
    assert "phase2_shortlist_unit" not in runtime_summary


def test_post_split_children_use_global_parent_cap_without_lane_authority() -> None:
    records = [
        {
            **_record(pool_index, 0, score),
            "candidate_label": f"parent-{pool_index}::child-{child}",
            "phase1_active_score": score,
            "phase2_raw_score": score,
            "simple_score": score,
            "physical_operator_lane": lane,
            "runtime_split_mode": "shortlist_pauli_children_v1",
            "runtime_split_chosen_representation": "child_atom",
            "runtime_split_parent_label": f"parent-{pool_index}",
        }
        for pool_index, child, score, lane in (
            (0, 0, 10.0, "lane_b"),
            (0, 1, 9.0, "lane_a"),
            (1, 0, 8.0, "lane_a"),
            (2, 0, 7.0, "lane_b"),
        )
    ]
    runtime = _physical_runtime()

    phase2 = _phase2_lane_health_shortlist_with_legacy_hook(
        records,
        runtime=runtime,
        score_key="phase2_raw_score",
        threshold=0.0,
        cap=2,
        frontier_ratio=0.0,
        tie_break_score_key="simple_score",
        shortlist_flag="phase2_shortlisted",
    )
    relabeled = [
        {
            **record,
            "physical_operator_lane": (
                "lane_a"
                if record["physical_operator_lane"] == "lane_b"
                else "lane_b"
            ),
        }
        for record in records
    ]
    relabeled_phase2 = _phase2_lane_health_shortlist_with_legacy_hook(
        relabeled,
        runtime=_physical_runtime(),
        score_key="phase2_raw_score",
        threshold=0.0,
        cap=2,
        frontier_ratio=0.0,
        tie_break_score_key="simple_score",
        shortlist_flag="phase2_shortlisted",
    )

    assert [row["candidate_pool_index"] for row in phase2] == [0, 1]
    assert [row["candidate_label"] for row in phase2] == [
        "parent-0::child-0",
        "parent-1::child-0",
    ]
    assert [row["candidate_label"] for row in relabeled_phase2] == [
        row["candidate_label"] for row in phase2
    ]
    assert len({macro_operator_identity(row) for row in phase2}) == len(phase2)
    assert "shortlist_runtime" not in runtime.lane_summary
    assert all(
        "physical_operator_lane_live" not in row
        for row in (*phase2, *relabeled_phase2)
    )
