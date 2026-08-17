from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from pipelines.static_adapt.ra_adapt.adaptive_phase_shortlist import (
    ADAPTIVE_PHASE123_SHORTLIST_POLICY_V1,
    AdaptivePhaseCandidateScore,
    AdaptivePhaseSelectionMappingReceipt,
    adaptive_phase_selection_receipt_from_mapping,
    adaptive_phase_shortlist_receipt_from_mapping,
    select_adaptive_phase_shortlist,
    validate_adaptive_phase_shortlist_receipt,
)
from pipelines.static_adapt.ra_adapt.contracts import canonical_sha256
from pipelines.static_adapt import adapt_pipeline
from pipelines.static_adapt import phase_shortlists
from pipelines.static_adapt.sr_snake._selection import (
    _CandidatePositionRecord,
    _PhaseSelectionReceipt,
    _ShortlistRankReceipt,
)


def _score(
    record_id: str,
    score: float,
    *,
    pool_index: int,
    position: int = 0,
    tie_break: float | None = None,
) -> AdaptivePhaseCandidateScore:
    return AdaptivePhaseCandidateScore(
        record_id=record_id,
        pool_index=pool_index,
        insertion_position=position,
        active_score=score,
        tie_break_score=score if tie_break is None else tie_break,
    )


def test_inverse_simpson_phase_shortlist_changes_cardinality_not_ranking() -> None:
    scores = (
        _score("r0", 1.0, pool_index=0),
        _score("r1", 1.0, pool_index=1),
        _score("r2", 0.1, pool_index=2),
    )

    decision = select_adaptive_phase_shortlist(
        scores,
        phase="phase_i",
        score_key="phase1_active_score",
        hard_cap=24,
        threshold=0.0,
        frontier_ratio=1.0,
    )

    assert decision.ranked_record_ids == ("r0", "r1", "r2")
    assert decision.retained_record_ids == ("r0", "r1")
    assert decision.receipt.policy == ADAPTIVE_PHASE123_SHORTLIST_POLICY_V1
    assert decision.receipt.rounded_effective_population_size == 2
    assert decision.receipt.retained_count == 2
    assert decision.receipt.saturated is False
    assert (
        validate_adaptive_phase_shortlist_receipt(
            decision.receipt,
            scores,
        )
        == decision.receipt
    )


def test_frontier_ratio_is_an_eligibility_guard_before_adaptive_cardinality() -> None:
    scores = (
        _score("r0", 1.0, pool_index=0),
        _score("r1", 0.95, pool_index=1),
        _score("r2", 0.40, pool_index=2),
        _score("r3", 0.39, pool_index=3),
    )

    decision = select_adaptive_phase_shortlist(
        scores,
        phase="phase_ii",
        score_key="phase2_raw_score",
        hard_cap=12,
        threshold=0.0,
        frontier_ratio=0.9,
    )

    assert decision.receipt.frontier_ratio_role == "eligibility_only"
    assert decision.receipt.frontier_eligible_record_ids == ("r0", "r1")
    assert decision.receipt.effective_population_score_record_ids == (
        "r0",
        "r1",
    )
    assert decision.retained_record_ids == ("r0", "r1")


def test_oversized_exact_boundary_tie_is_deterministically_saturated() -> None:
    scores = tuple(
        _score(f"r{index:02d}", 1.0, pool_index=index)
        for index in range(30)
    )

    decision = select_adaptive_phase_shortlist(
        scores,
        phase="phase_iii",
        score_key="selector_score",
        hard_cap=12,
        threshold=0.0,
        frontier_ratio=1.0,
    )

    assert decision.retained_record_ids == tuple(
        f"r{index:02d}" for index in range(12)
    )
    assert decision.receipt.saturated is True
    assert decision.receipt.saturation_reason == (
        "exact_boundary_tie_shell_exceeds_hard_cap"
    )
    assert decision.receipt.boundary_tie_shell_size == 30
    assert decision.receipt.hard_cap == 12
    assert adaptive_phase_shortlist_receipt_from_mapping(
        decision.receipt.to_dict()
    ) == decision.receipt
    tampered = decision.receipt.to_dict()
    tampered["retained_count"] = 11
    with pytest.raises(ValueError, match="receipt mapping"):
        adaptive_phase_shortlist_receipt_from_mapping(tampered)


def test_adaptive_phase_receipt_rejects_tampering_and_excludes_bad_scores() -> None:
    scores = (
        _score("good", 1.0, pool_index=0),
        _score("zero", 0.0, pool_index=1),
        _score("negative", -1.0, pool_index=2),
        _score("infinite", float("inf"), pool_index=3),
    )
    decision = select_adaptive_phase_shortlist(
        scores,
        phase="phase_i",
        score_key="phase1_active_score",
        hard_cap=24,
        threshold=0.0,
        frontier_ratio=1.0,
    )

    assert decision.retained_record_ids == ("good",)
    assert decision.receipt.active_feasible_record_ids == ("good", "zero")
    assert decision.receipt.positive_score_record_ids == ("good",)
    assert decision.receipt.excluded_record_ids == ("negative", "infinite")
    with pytest.raises(ValueError, match="adaptive Phase-shortlist receipt"):
        validate_adaptive_phase_shortlist_receipt(
            replace(decision.receipt, retained_count=2),
            scores,
        )


@pytest.mark.parametrize(
    ("phase", "cap"),
    (("phase_i", 24), ("phase_ii", 12), ("phase_iii", 12)),
)
def test_phase_hard_caps_are_typed_and_phase_specific(
    phase: str,
    cap: int,
) -> None:
    scores = tuple(
        _score(f"r{index:02d}", 1.0 / (index + 1), pool_index=index)
        for index in range(30)
    )
    decision = select_adaptive_phase_shortlist(
        scores,
        phase=phase,
        score_key=f"{phase}_active_score",
        hard_cap=cap,
        threshold=0.0,
        frontier_ratio=1.0,
    )

    assert decision.receipt.phase == phase
    assert decision.receipt.hard_cap == cap
    assert len(decision.retained_record_ids) <= cap


def test_live_record_adapter_changes_only_cardinality_and_marks_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        phase_shortlists,
        "_notify_legacy_shortlist_hook",
        lambda *_args, **_kwargs: None,
    )
    runtime = SimpleNamespace(
        feature_updater=lambda feature, _updates: feature,
    )
    records = [
        {
            "candidate_label": f"g{index}",
            "candidate_pool_index": index,
            "position_id": 0,
            "phase2_raw_score": score,
            "simple_score": score,
        }
        for index, score in enumerate((1.0, 1.0, 0.1))
    ]

    retained, receipt = (
        phase_shortlists._adaptive_phase_shortlist_with_receipt(
            records,
            runtime=runtime,
            phase="phase_ii",
            score_key="phase2_raw_score",
            threshold=0.0,
            hard_cap=12,
            frontier_ratio=1.0,
            tie_break_score_key="simple_score",
            shortlist_flag="phase2_shortlisted",
        )
    )

    assert [row["candidate_label"] for row in retained] == ["g0", "g1"]
    assert [row["shortlist_rank"] for row in retained] == [1, 2]
    assert all(row["phase2_shortlisted"] is True for row in retained)
    assert receipt.retained_count == 2
    assert receipt.qiskit_work_performed is False
    assert receipt.metric_work_performed is False
    assert receipt.compile_work_in_s_alg is False


def test_typed_phase_receipts_serialize_deep_adaptive_evidence() -> None:
    records = tuple(
        _CandidatePositionRecord(
            domain_record_id=f"d{index}",
            generator_id=f"g{index}",
            parent_generator_id=None,
            pool_index=index,
            pool_label=f"g{index}",
            insertion_position=0,
            symmetry_identity=f"sym{index}",
            lineage_identity=(f"g{index}",),
        )
        for index in range(2)
    )

    def _phase_receipt(phase: str, cap: int) -> _PhaseSelectionReceipt:
        decision = select_adaptive_phase_shortlist(
            tuple(
                _score(
                    f"g{index}|pool:{index}|position:0",
                    1.0 - 0.1 * index,
                    pool_index=index,
                )
                for index in range(2)
            ),
            phase=phase,
            score_key=f"{phase}_active_score",
            hard_cap=cap,
            threshold=0.0,
            frontier_ratio=1.0,
        )
        shortlist = records if phase != "phase_iii" else records[:1]
        ranking = tuple(
            _ShortlistRankReceipt(
                record_key=(row.domain_record_id, row.generator_id),
                shortlist_rank=index,
                primary_score=1.0 - 0.1 * (index - 1),
                tie_break_score=1.0 - 0.1 * (index - 1),
                pool_index=row.pool_index,
                insertion_position=row.insertion_position,
            )
            for index, row in enumerate(shortlist, start=1)
        )
        return _PhaseSelectionReceipt(
            phase=phase,
            population=records,
            shortlist=shortlist,
            shortlist_ranking=ranking,
            estimator_event_ids=(),
            adaptive_shortlist=decision.receipt,
            adaptive_live_scores=decision.receipt.input_scores,
        )

    payload = adapt_pipeline._scored_insertion_position_population_receipt(
        SimpleNamespace(
            phase_i=_phase_receipt("phase_i", 24),
            phase_ii=_phase_receipt("phase_ii", 12),
            phase_iii=_phase_receipt("phase_iii", 12),
        ),
        append_position=0,
    )

    assert [
        row["adaptive_shortlist"]["phase"] for row in payload["phases"]
    ] == ["phase_i", "phase_ii", "phase_iii"]
    assert all(
        row["adaptive_shortlist"]["sha256"]
        for row in payload["phases"]
    )


def test_typed_phase_receipt_rejects_detached_scores_and_phase3_winner() -> None:
    records = tuple(
        _CandidatePositionRecord(
            domain_record_id=f"d{index}",
            generator_id=f"g{index}",
            parent_generator_id=None,
            pool_index=index,
            pool_label=f"g{index}",
            insertion_position=0,
            symmetry_identity=f"sym{index}",
            lineage_identity=(f"g{index}",),
        )
        for index in range(2)
    )
    scores = tuple(
        _score(
            f"g{index}|pool:{index}|position:0",
            1.0 - 0.1 * index,
            pool_index=index,
        )
        for index in range(2)
    )
    decision = select_adaptive_phase_shortlist(
        scores,
        phase="phase_iii",
        score_key="full_v2_score",
        hard_cap=12,
        threshold=0.0,
        frontier_ratio=0.9,
    )

    def _ranking(record: _CandidatePositionRecord, score: float) -> tuple[_ShortlistRankReceipt, ...]:
        return (
            _ShortlistRankReceipt(
                record_key=(record.domain_record_id, record.generator_id),
                shortlist_rank=1,
                primary_score=score,
                tie_break_score=score,
                pool_index=record.pool_index,
                insertion_position=record.insertion_position,
            ),
        )

    detached_scores = (
        replace(scores[0], active_score=0.5, tie_break_score=0.5),
        scores[1],
    )
    with pytest.raises(ValueError, match="adaptive shortlist receipt"):
        _PhaseSelectionReceipt(
            phase="phase_iii",
            population=records,
            shortlist=records[:1],
            shortlist_ranking=_ranking(records[0], 0.5),
            estimator_event_ids=(),
            adaptive_shortlist=decision.receipt,
            adaptive_live_scores=detached_scores,
        )
    with pytest.raises(ValueError, match="prefix champion"):
        _PhaseSelectionReceipt(
            phase="phase_iii",
            population=records,
            shortlist=records[1:],
            shortlist_ranking=_ranking(records[1], 0.9),
            estimator_event_ids=(),
            adaptive_shortlist=decision.receipt,
            adaptive_live_scores=scores,
        )


def _serialized_phase_mapping(
    *,
    phase: str,
    cap: int,
) -> dict[str, object]:
    score_key = {
        "phase_i": "phase1_active_score",
        "phase_ii": "phase2_raw_score",
        "phase_iii": "full_v2_score",
    }[phase]
    decision = select_adaptive_phase_shortlist(
        (
            _score(
                "g0|pool:0|position:2",
                1.0,
                pool_index=0,
                position=2,
            ),
            _score(
                "g1|pool:1|position:2",
                0.95,
                pool_index=1,
                position=2,
            ),
        ),
        phase=phase,
        score_key=score_key,
        hard_cap=cap,
        threshold=0.0,
        frontier_ratio=0.9,
    )
    records = [
        {
            "domain_record_id": f"d{index}",
            "generator_id": f"g{index}",
            "pool_index": index,
            "pool_label": f"g{index}",
            "insertion_position": 2,
            "adaptive_record_id": f"g{index}|pool:{index}|position:2",
            "position_class": "append",
        }
        for index in range(2)
    ]
    shortlist = records if phase != "phase_iii" else records[:1]
    live_scores = [row.to_dict() for row in decision.receipt.input_scores]
    return {
        "phase": phase,
        "population_count": len(records),
        "records": records,
        "shortlist_count": len(shortlist),
        "shortlist_records": shortlist,
        "adaptive_shortlist": decision.receipt.to_dict(),
        "adaptive_population_scores": live_scores,
        "ordered_adaptive_population_scores_sha256": canonical_sha256(
            live_scores
        ),
        "final_admission_record_id": (
            shortlist[0]["adaptive_record_id"]
            if phase == "phase_iii"
            else None
        ),
        "estimator_event_ids": [],
        "ordered_population_sha256": canonical_sha256(records),
    }


@pytest.mark.parametrize(
    ("phase", "score_key", "cap", "shortlist_count"),
    (
        ("phase_i", "phase1_active_score", 24, 2),
        ("phase_ii", "phase2_raw_score", 12, 2),
        ("phase_iii", "full_v2_score", 12, 1),
    ),
)
def test_public_serialized_phase_mapper_deeply_binds_live_selection(
    phase: str,
    score_key: str,
    cap: int,
    shortlist_count: int,
) -> None:
    phase_mapping = _serialized_phase_mapping(phase=phase, cap=cap)

    validated = adaptive_phase_selection_receipt_from_mapping(
        phase_mapping,
        expected_phase=phase,
        expected_score_key=score_key,
        expected_hard_cap=cap,
        expected_frontier_ratio=0.9,
    )

    assert isinstance(validated, AdaptivePhaseSelectionMappingReceipt)
    assert validated.population_count == 2
    assert validated.adaptive_retained_count == 2
    assert validated.final_singleton_count == shortlist_count
    assert validated.mapping_sha256 == canonical_sha256(phase_mapping)


@pytest.mark.parametrize(
    ("phase", "score_key", "cap"),
    (
        ("phase_i", "phase1_active_score", 24),
        ("phase_ii", "phase2_raw_score", 12),
    ),
)
def test_public_serialized_phase_mapper_rejects_no_positive_before_phase_three(
    phase: str,
    score_key: str,
    cap: int,
) -> None:
    phase_mapping = _serialized_phase_mapping(phase=phase, cap=cap)
    zero = select_adaptive_phase_shortlist(
        (
            _score(
                f"g{index}|pool:{index}|position:2",
                0.0,
                pool_index=index,
                position=2,
            )
            for index in range(2)
        ),
        phase=phase,
        score_key=score_key,
        hard_cap=cap,
        threshold=0.0,
        frontier_ratio=0.9,
    ).receipt
    live_scores = [row.to_dict() for row in zero.input_scores]
    phase_mapping.update(
        {
            "shortlist_count": 0,
            "shortlist_records": [],
            "adaptive_shortlist": zero.to_dict(),
            "adaptive_population_scores": live_scores,
            "ordered_adaptive_population_scores_sha256": canonical_sha256(
                live_scores
            ),
        }
    )

    with pytest.raises(ValueError, match="mapping"):
        adaptive_phase_selection_receipt_from_mapping(
            phase_mapping,
            expected_phase=phase,
            expected_score_key=score_key,
            expected_hard_cap=cap,
            expected_frontier_ratio=0.9,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (
            lambda row: row["adaptive_population_scores"].reverse(),
            "mapping",
        ),
        (
            lambda row: row["records"][0].__setitem__(
                "adaptive_record_id", "detached"
            ),
            "mapping",
        ),
        (
            lambda row: row.__setitem__(
                "final_admission_record_id",
                "g1|pool:1|position:2",
            ),
            "mapping",
        ),
    ),
)
def test_public_serialized_phase_mapper_rejects_detached_evidence(
    mutation: object,
    match: str,
) -> None:
    phase_mapping = _serialized_phase_mapping(phase="phase_iii", cap=12)
    mutation(phase_mapping)

    with pytest.raises(ValueError, match=match):
        adaptive_phase_selection_receipt_from_mapping(
            phase_mapping,
            expected_phase="phase_iii",
            expected_score_key="full_v2_score",
            expected_hard_cap=12,
            expected_frontier_ratio=0.9,
        )
