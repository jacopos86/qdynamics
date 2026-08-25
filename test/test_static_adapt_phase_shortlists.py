from __future__ import annotations
from functools import lru_cache
import json
from typing import Any, Mapping

import pytest

from pipelines.scaffold.hh_continuation_scoring import FullScoreConfig
from pipelines.scaffold.hh_continuation_types import CandidateFeatures
from pipelines.static_adapt import phase_shortlists
from pipelines.static_adapt.ra_adapt.semantic_closure_routes import (
    build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request,
    build_paper_i_ra_hh_regime_problem,
    materialize_paper_i_ra_semantic_protocol,
)


PHYSICAL_TEST_LANES = ("family_a", "family_b", "family_c", "other")


@lru_cache(maxsize=1)
def _natural_terminal_authority(
) -> phase_shortlists.Phase3NaturalTerminalAuthority:
    protocol = materialize_paper_i_ra_semantic_protocol(
        build_paper_i_ra_hh_regime_problem("weak_weak"),
        build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request(
            insertion_policy="append_only",
            maximum_controller_rounds=1,
        ),
    )
    return phase_shortlists.Phase3NaturalTerminalAuthority.from_route_contract(
        protocol.route_contract,
        expected_route_contract_sha256=protocol.route_contract["sha256"],
    )


def test_natural_terminal_authority_rejects_noncanonical_contract_bytes() -> None:
    authority = _natural_terminal_authority()
    payload = json.loads(authority.route_contract_json)
    noncanonical = json.dumps(payload, indent=2, sort_keys=True)

    with pytest.raises(ValueError, match="canonical JSON"):
        phase_shortlists.Phase3NaturalTerminalAuthority(
            route_contract_json=noncanonical,
            route_contract_sha256=authority.route_contract_sha256,
        )


def _feature(**overrides: object) -> CandidateFeatures:
    values: dict[str, object] = dict(
        stage_name="phase3",
        candidate_label="a",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        g_signed=0.25,
        g_abs=0.25,
        g_lcb=0.25,
        sigma_hat=0.0,
        F=1.0,
        novelty=0.8,
        curvature_mode="current_curv",
        novelty_mode="current_novelty",
        refit_window_indices=[0],
        compiled_position_cost_proxy={},
        measurement_cache_stats={},
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        simple_score=0.25,
        score_version="test_v1",
        phase2_raw_score=0.25,
        full_v2_score=0.25,
        selector_score=0.25,
        phase_score_components={"existing": 1.0},
        actual_fallback_mode="exact_reduced",
    )
    values.update(overrides)
    return CandidateFeatures(**values)


def _runtime(
    *,
    active: bool = False,
    summary: dict[str, Any] | None = None,
    lane_summary: dict[str, Any] | None = None,
    lane_route: str = "physical_operator_type",
    lane_key: str = "physical_operator_lane",
    lanes: tuple[str, ...] = PHYSICAL_TEST_LANES,
    fallback_lane: str = "other",
    lane_health_key_prefix: str = "physical_operator",
    phase1_pressure: float = 1.0,
    phase2_pressure: float = 1.0,
    phase2_rel: float = 0.0,
    phase1_lane_retention_enabled: bool = True,
    physical_operator_identity_caps_enabled: bool = True,
    natural_terminal: bool = False,
) -> phase_shortlists.PhaseShortlistRuntime:
    def feature_updater(feat: Any, updates: Mapping[str, Any]) -> Any:
        if not isinstance(feat, CandidateFeatures):
            return feat
        return CandidateFeatures(**{**feat.__dict__, **dict(updates)})

    return phase_shortlists.PhaseShortlistRuntime(
        phase2_score_cfg=FullScoreConfig(shortlist_size=2, shortlist_fraction=0.5),
        feature_updater=feature_updater,
        lane_policy_active=bool(active),
        lane_summary=(
            lane_summary
            if lane_summary is not None
            else (summary if summary is not None else {})
        ),
        phase1_lane_quota_pressure=float(phase1_pressure),
        phase2_lane_quota_pressure=float(phase2_pressure),
        phase2_lane_rel_threshold=float(phase2_rel),
        shortlist_lane_route=str(lane_route),
        shortlist_lane_key=str(lane_key),
        shortlist_lanes=tuple(str(x) for x in lanes),
        shortlist_fallback_lane=str(fallback_lane),
        shortlist_lane_health_key_prefix=str(lane_health_key_prefix),
        phase1_lane_retention_enabled=bool(
            phase1_lane_retention_enabled
        ),
        physical_operator_identity_caps_enabled=bool(
            physical_operator_identity_caps_enabled
        ),
        phase3_natural_terminal_authority=(
            _natural_terminal_authority() if natural_terminal else None
        ),
    )


def _record(
    label: str,
    idx: int,
    *,
    score: float,
    phase2: float | None = None,
    full: float | None = None,
    lane: str = "other",
    position: int = 0,
    snapshot: Mapping[str, Any] | None = None,
    feature: CandidateFeatures | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "candidate_label": str(label),
        "candidate_pool_index": int(idx),
        "position_id": int(position),
        "simple_score": float(score),
        "cheap_score": float(score) - 0.1,
        "phase1_active_score": float(score) + 0.1,
        "phase2_raw_score": float(score if phase2 is None else phase2),
        "full_v2_score": float(score if full is None else full),
        "physical_operator_lane": str(lane),
    }
    if snapshot is not None:
        row["controller_snapshot"] = dict(snapshot)
    if feature is not None:
        row["feature"] = feature
    return row


def _phase3_sort(row: Mapping[str, Any]) -> tuple[float, int, int]:
    return (
        -float(row.get("full_v2_score", float("-inf"))),
        int(row.get("candidate_pool_index", -1)),
        int(row.get("position_id", -1)),
    )


def _phase2_sort(row: Mapping[str, Any]) -> tuple[float, int, int]:
    return (
        -float(row.get("phase2_raw_score", float("-inf"))),
        int(row.get("candidate_pool_index", -1)),
        int(row.get("position_id", -1)),
    )


def test_adaptive_phase3_no_positive_requires_authenticated_v2_authority() -> None:
    records = [
        _record(
            "zero-score",
            0,
            score=0.0,
            full=0.0,
            feature=_feature(
                candidate_label="zero-score",
                candidate_pool_index=0,
                position_id=0,
                simple_score=0.0,
                full_v2_score=0.0,
            ),
        )
    ]
    kwargs = {
        "phase": "phase_iii",
        "score_key": "full_v2_score",
        "threshold": 0.0,
        "hard_cap": 12,
        "frontier_ratio": 0.9,
        "tie_break_score_key": "simple_score",
        "shortlist_flag": "phase3_shortlisted",
    }

    with pytest.raises(
        RuntimeError,
        match=(
            "Adaptive phase_iii shortlist has no positive feasible candidate"
        ),
    ):
        phase_shortlists._adaptive_phase_shortlist_with_receipt(
            records,
            runtime=_runtime(),
            **kwargs,
        )

    retained, receipt = (
        phase_shortlists._adaptive_phase_shortlist_with_receipt(
            records,
            runtime=_runtime(natural_terminal=True),
            **kwargs,
        )
    )

    assert retained == []
    assert receipt.phase == "phase_iii"
    assert receipt.status == "no_positive_population"
    assert receipt.retained_record_ids == ()


def test_phase1_score_value_prefers_active_then_cheap_then_simple() -> None:
    assert phase_shortlists._phase1_shortlist_score_key() == "phase1_active_score"
    assert phase_shortlists._phase1_record_score_value({"phase1_active_score": 3.0, "cheap_score": 4.0}) == 3.0
    assert phase_shortlists._phase1_record_score_value({"cheap_score": 4.0, "simple_score": 5.0}) == 4.0
    assert phase_shortlists._phase1_record_score_value({"simple_score": 5.0}) == 5.0
    assert phase_shortlists._phase1_record_score_value({}, default=-7.0) == -7.0


def test_phase1_eval_payload_reads_candidate_features_and_append_split() -> None:
    feature = _feature(candidate_family="append_family", g_lcb=0.7, g_hw_lcb=0.8)
    records = [
        _record("append", 1, score=1.0, position=2, feature=feature),
        _record("insert", 2, score=2.0, position=0),
    ]

    payload = phase_shortlists._phase1_eval_payload_from_records(
        records,
        append_position_value=2,
    )

    assert payload["best_idx"] == 2
    assert payload["best_position"] == 0
    assert payload["best_score"] == pytest.approx(2.1)
    assert payload["append_best_g_lcb"] == pytest.approx(0.8)
    assert payload["append_best_family"] == "append_family"
    assert payload["best_non_append_g_lcb"] == pytest.approx(0.0)


def test_generic_shortlist_calls_legacy_hook_and_falls_back_to_best_record(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_shortlist(records: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
        calls.append({"records": records, **kwargs})
        return []

    monkeypatch.setattr(phase_shortlists, "shortlist_records", fake_shortlist)
    records = [_record("a", 0, score=1.0), _record("b", 1, score=2.0)]

    selected = phase_shortlists._phase_shortlist_with_legacy_hook(
        records,
        runtime=_runtime(),
        score_key="simple_score",
        threshold=10.0,
        cap=2,
        frontier_ratio=0.0,
        tie_break_score_key="cheap_score",
        shortlist_flag="kept",
    )

    assert len(calls) == 1
    assert calls[0]["cfg"].shortlist_size == 2
    assert calls[0]["score_key"] == "simple_score"
    assert [row["candidate_label"] for row in selected] == ["b"]


def test_inactive_lane_phase_wrappers_delegate_to_generic_shortlist() -> None:
    records = [
        _record("a", 0, score=1.0),
        _record(
            "b",
            1,
            score=2.0,
            feature=_feature(candidate_label="b", candidate_pool_index=1, simple_score=2.0),
        ),
    ]

    phase1 = phase_shortlists._phase1_lane_shortlist_with_legacy_hook(
        records,
        runtime=_runtime(active=False),
        score_key="simple_score",
        threshold=0.0,
        cap=1,
        frontier_ratio=0.0,
        shortlist_flag="phase1_shortlisted",
    )
    phase2 = phase_shortlists._phase2_lane_health_shortlist_with_legacy_hook(
        records,
        runtime=_runtime(active=False),
        score_key="phase2_raw_score",
        threshold=0.0,
        cap=1,
        frontier_ratio=0.0,
        shortlist_flag="phase2_shortlisted",
    )

    assert [row["candidate_label"] for row in phase1] == ["b"]
    assert phase1[0]["feature"].phase1_shortlisted is True
    assert [row["candidate_label"] for row in phase2] == ["b"]
    assert phase2[0]["feature"].phase2_shortlisted is True


def test_active_physical_phase1_records_lane_budgets_and_updates_features() -> None:
    summary: dict[str, Any] = {}
    feature = _feature(candidate_label="flat", candidate_pool_index=0, simple_score=2.0)
    records = [
        _record("family-a", 0, score=2.0, lane="family_a", feature=feature),
        _record("family-b", 1, score=1.5, lane="family_b"),
        _record("family-c", 2, score=1.0, lane="family_c"),
    ]

    selected = phase_shortlists._phase1_lane_shortlist_with_legacy_hook(
        records,
        runtime=_runtime(active=True, summary=summary),
        score_key="simple_score",
        threshold=0.0,
        cap=2,
        frontier_ratio=1.0,
        shortlist_flag="phase1_shortlisted",
    )

    assert len(selected) == 2
    assert selected[0]["phase1_shortlisted"] is True
    assert isinstance(selected[0]["feature"], CandidateFeatures)
    assert selected[0]["feature"].phase1_shortlisted is True
    assert summary["shortlist_runtime"]["phase1_last_shortlist_size"] == 2
    assert summary["shortlist_runtime"]["phase1_last_budget_target"] == 2


def test_active_physical_phase2_records_lane_health_and_runtime_summary() -> None:
    summary: dict[str, Any] = {}
    records = [
        _record("family-a", 0, score=2.0, phase2=2.0, lane="family_a"),
        _record("family-b", 1, score=1.5, phase2=1.5, lane="family_b"),
        _record("weak", 2, score=0.1, phase2=0.1, lane="family_c"),
    ]

    selected = phase_shortlists._phase2_lane_health_shortlist_with_legacy_hook(
        records,
        runtime=_runtime(active=True, summary=summary, phase2_rel=0.5),
        score_key="phase2_raw_score",
        threshold=0.0,
        cap=2,
        frontier_ratio=1.0,
        shortlist_flag="phase2_shortlisted",
    )

    assert {row["candidate_label"] for row in selected} == {
        "family-a",
        "family-b",
    }
    assert all(row["phase2_shortlisted"] is True for row in selected)
    assert all(
        row["physical_operator_lane_live"] is True for row in selected
    )
    assert summary["shortlist_runtime"]["phase2_last_shortlist_size"] == 2


def test_neutral_lane_budget_helper_preserves_physical_family_coverage() -> None:
    records = [
        _record("a-best", 0, score=10.0, lane="family_a"),
        _record("a-second", 1, score=9.0, lane="family_a"),
        _record("b", 2, score=8.0, lane="family_b"),
        _record("c", 3, score=7.0, lane="family_c"),
    ]

    budgets = phase_shortlists.lane_quota_pressure_budgets(
        records,
        cap=3,
        score_key="simple_score",
        lane_key="physical_operator_lane",
        lanes=PHYSICAL_TEST_LANES,
        fallback_lane="other",
        pressure=1.0,
    )
    assert budgets == {
        "family_a": 1,
        "family_b": 1,
        "family_c": 1,
        "other": 0,
    }

    selected = phase_shortlists.lane_phase1_shortlist_records(
        records,
        score_key="simple_score",
        threshold=0.0,
        cap=3,
        frontier_ratio=1.0,
        lane_key="physical_operator_lane",
        lanes=PHYSICAL_TEST_LANES,
        fallback_lane="other",
        lane_budgets=budgets,
    )
    assert [row["candidate_label"] for row in selected] == [
        "a-best",
        "b",
        "c",
    ]


def test_neutral_lane_helpers_require_an_explicit_valid_fallback() -> None:
    with pytest.raises(ValueError, match="Fallback lane"):
        phase_shortlists.lane_quota_pressure_budgets(
            [_record("a", 0, score=1.0, lane="family_a")],
            cap=1,
            score_key="simple_score",
            lane_key="physical_operator_lane",
            lanes=("family_a",),
            fallback_lane="other",
        )


def test_active_physical_lane_route_records_physical_health_and_runtime_summary() -> None:
    summary: dict[str, Any] = {}
    lanes = ("uccsd_correlation", "phonon_displacement", "dressed_phonon_correlation", "other")
    records = [
        {
            **_record("uccsd", 0, score=3.0, phase2=3.0, feature=_feature()),
            "physical_operator_lane": "uccsd_correlation",
        },
        {
            **_record("phonon", 1, score=2.5, phase2=2.5, feature=_feature(candidate_pool_index=1)),
            "physical_operator_lane": "phonon_displacement",
        },
        {
            **_record("dressed", 2, score=0.2, phase2=0.2, feature=_feature(candidate_pool_index=2)),
            "physical_operator_lane": "dressed_phonon_correlation",
        },
    ]

    selected = phase_shortlists._phase2_lane_health_shortlist_with_legacy_hook(
        records,
        runtime=_runtime(
            active=True,
            lane_summary=summary,
            lane_route="physical_operator_type",
            lane_key="physical_operator_lane",
            lanes=lanes,
            fallback_lane="other",
            lane_health_key_prefix="physical_operator",
            phase2_rel=0.5,
        ),
        score_key="phase2_raw_score",
        threshold=0.0,
        cap=2,
        frontier_ratio=1.0,
        shortlist_flag="phase2_shortlisted",
    )

    assert {row["candidate_label"] for row in selected} == {"uccsd", "phonon"}
    assert all(row["physical_operator_lane_live"] is True for row in selected)
    assert all(row["feature"].physical_operator_lane_live is True for row in selected)
    runtime = summary["shortlist_runtime"]
    assert runtime["lane_route"] == "physical_operator_type"
    assert runtime["phase2_last_shortlist_size"] == 2
    assert runtime["phase2_last_lane_budgets"]["uccsd_correlation"] == 1
    assert runtime["phase2_last_lane_budgets"]["phonon_displacement"] == 1


def test_phase1_lane_retention_off_uses_global_ranking_and_keeps_lane_telemetry() -> None:
    summary: dict[str, Any] = {}
    records = [
        {
            **_record("lane-a-best", 0, score=10.0),
            "physical_operator_lane": "lane_a",
        },
        {
            **_record("lane-a-second", 1, score=9.0),
            "physical_operator_lane": "lane_a",
        },
        {
            **_record("lane-b", 2, score=1.0),
            "physical_operator_lane": "lane_b",
        },
    ]
    runtime = _runtime(
        active=True,
        lane_summary=summary,
        lane_route="physical_operator_type",
        lane_key="physical_operator_lane",
        lanes=("lane_a", "lane_b"),
        fallback_lane="lane_b",
        lane_health_key_prefix="physical_operator",
        phase1_lane_retention_enabled=False,
        physical_operator_identity_caps_enabled=False,
    )

    selected = phase_shortlists._phase1_lane_shortlist_with_legacy_hook(
        records,
        runtime=runtime,
        score_key="phase1_active_score",
        threshold=0.0,
        cap=2,
        frontier_ratio=1.0,
        tie_break_score_key="simple_score",
        shortlist_flag="phase1_shortlisted",
    )

    assert [row["candidate_label"] for row in selected] == [
        "lane-a-best",
        "lane-a-second",
    ]
    audit = summary["phase1_lane_retention_audits"][0]
    assert audit["classification_active"] is True
    assert audit["lane_retention_enabled"] is False
    assert audit["applied_selection_mode"] == "global"
    assert audit["shortlists_differ"] is True
    assert [
        row["candidate_label"]
        for row in audit["lane_protected_counterfactual_shortlist"]
    ] == ["lane-a-best", "lane-b"]
    assert audit["lane_protected_only_selection_keys"] == [
        "pool:2@position:0"
    ]
    assert summary["shortlist_runtime"]["lane_route"] == (
        "physical_operator_type"
    )


def test_phase1_lane_retention_off_does_not_disable_phase2_lane_health() -> None:
    summary: dict[str, Any] = {}
    records = [
        {
            **_record("lane-a-best", 0, score=10.0, phase2=10.0),
            "physical_operator_lane": "lane_a",
        },
        {
            **_record("lane-a-second", 1, score=9.0, phase2=9.0),
            "physical_operator_lane": "lane_a",
        },
        {
            **_record("lane-b", 2, score=1.0, phase2=1.0),
            "physical_operator_lane": "lane_b",
        },
    ]
    runtime = _runtime(
        active=True,
        lane_summary=summary,
        lane_route="physical_operator_type",
        lane_key="physical_operator_lane",
        lanes=("lane_a", "lane_b"),
        fallback_lane="lane_b",
        lane_health_key_prefix="physical_operator",
        phase1_lane_retention_enabled=False,
        physical_operator_identity_caps_enabled=False,
    )

    selected = phase_shortlists._phase2_lane_health_shortlist_with_legacy_hook(
        records,
        runtime=runtime,
        score_key="phase2_raw_score",
        threshold=0.0,
        cap=2,
        frontier_ratio=1.0,
        tie_break_score_key="simple_score",
        shortlist_flag="phase2_shortlisted",
    )

    assert [row["candidate_label"] for row in selected] == [
        "lane-a-best",
        "lane-b",
    ]
    assert summary["shortlist_runtime"]["phase2_last_lane_budgets"] == {
        "lane_a": 1,
        "lane_b": 1,
    }


def test_selection_pool_preserves_legacy_first_positive_fallback_without_duplicates() -> None:
    shortlist = [_record("a", 0, score=1.0, full=1.0)]
    full = [
        _record("a", 0, score=1.0, full=1.0),
        _record("b", 1, score=0.8, full=0.8),
        _record("c", 2, score=-1.0, full=-1.0),
    ]

    selected = phase_shortlists._selection_pool_from_shortlist(
        shortlist,
        full,
        selector_score_key="full_v2_score",
        record_sort_key=_phase3_sort,
    )

    assert [row["candidate_label"] for row in selected] == ["a"]
    assert phase_shortlists._selection_record_key(selected[0]) == ("a", 0, 0)
    assert [row["candidate_label"] for row in phase_shortlists._positive_phase3_selector_records(full, selector_score_key="full_v2_score")] == ["a", "b"]
