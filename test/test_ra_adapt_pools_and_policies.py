from __future__ import annotations

import json
from typing import Any

import pytest

from pipelines.contracts.problem import ProblemRequest
from pipelines.scaffold.hh_continuation_scoring import (
    SimpleScoreConfig,
    phase1_score_payload,
)
from pipelines.scaffold.hh_continuation_types import CandidateFeatures
from pipelines.static_adapt.builders.problem_registry import (
    resolve_problem_context,
)
from pipelines.static_adapt.ra_adapt.adapters import (
    GlobalSinglePauliWordCandidateAdapter,
    MacroCandidateAdapter,
    SinglePauliWordCandidateAdapter,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    ACTIVE_GRADIENT_STATIONARY,
    CANDIDATE_REPRESENTATION_MACRO,
    CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    CandidateLineageReceipt,
    PolicyEchoReceipt,
    RESOURCE_WEIGHTING_LATE,
)
from pipelines.static_adapt.ra_adapt.pools import (
    build_candidate_inventory_lineage_receipt,
    build_executable_macro_pool,
    build_parent_template_inventory,
)


POOL_FIXTURES = {
    3: {
        "parent_count": 123,
        "parent_labels_sha256": (
            "17cc97b744f8e6b50b686b24edd28426ca2c055bc2c31054fd353ddfa10efbe3"
        ),
        "parent_pool_sha256": (
            "b533c4e08e57683bfb42de7a811ef106ba0eaa94f75d9a57f907cc36370fa67d"
        ),
        "macro_count": 102,
        "macro_removed_count": 21,
        "macro_labels_sha256": (
            "a8831528590e870a09ce08492b6f61da4a4d377e63fa8983b30ca9698af5d3d9"
        ),
        "macro_pool_sha256": (
            "1549f2e108406f494c2d4f884212c1026dbaa42f12eb92189f18eaf2a62b17df"
        ),
        "guarded_count": 948,
        "guarded_labels_sha256": (
            "02995a2c570d4322e46e55e3a532381ff7eff85dc3c2de8cb2b30ed888b76906"
        ),
        "guarded_pool_sha256": (
            "66ea9d6b058b562ba913e221124285de7be0ec13a972d30b9812ab29874a58e0"
        ),
    },
    7: {
        "parent_count": 171,
        "parent_labels_sha256": (
            "389ce1382b57b916e15e170c641f3884ed1ce33e9913d6eb709f24490739e93f"
        ),
        "parent_pool_sha256": (
            "831817f5a6a072ad2a43f4413b34fa6da558120081bbebce2831261ac03d680e"
        ),
        "macro_count": 148,
        "macro_removed_count": 23,
        "macro_labels_sha256": (
            "e6de937476653868f7d3974ad67c467c2f2e2496770e256671b2e807a5b5b03a"
        ),
        "macro_pool_sha256": (
            "e30e879dabf4d6eb234be92aae1cea76998172b67e8c679b241f5cdc6641d14e"
        ),
        "guarded_count": 6508,
        "guarded_labels_sha256": (
            "079478057eea213139dc2f3c7486097496454421a44677c290b5dc55860accb7"
        ),
        "guarded_pool_sha256": (
            "8e1fe54be4b089d759d334399add40fc5edea8faa31af9ea70f1f2cc36834e93"
        ),
    },
}


def _problem(*, n_ph_max: int) -> Any:
    return resolve_problem_context(
        ProblemRequest(
            problem_key="hh",
            num_sites=2,
            t=1.0,
            u=0.5,
            dv=0.0,
            omega0=1.0,
            g_ep=0.2,
            n_ph_max=n_ph_max,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
        )
    )


def _feature_with_nontrivial_resource_cost() -> CandidateFeatures:
    return CandidateFeatures(
        stage_name="phase1",
        candidate_label="fixture",
        candidate_family="fixture",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        g_signed=0.8,
        g_abs=0.8,
        g_lcb=0.8,
        sigma_hat=0.0,
        F_metric=1.0,
        metric_proxy=1.0,
        novelty=1.0,
        curvature_mode="fixture",
        novelty_mode="fixture",
        refit_window_indices=[],
        compiled_position_cost_proxy={},
        measurement_cache_stats={},
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        simple_score=None,
        score_version="fixture_v1",
        c_bar_2q=3.0,
    )


@pytest.mark.parametrize("n_ph_max", [3, 7])
def test_parent_macro_and_global_guarded_pool_identities(
    n_ph_max: int,
) -> None:
    expected = POOL_FIXTURES[n_ph_max]
    problem = _problem(n_ph_max=n_ph_max)
    macro_parent = build_parent_template_inventory(
        problem,
        representation_id=CANDIDATE_REPRESENTATION_MACRO,
    )
    singleton_parent = build_parent_template_inventory(
        problem,
        representation_id=CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    )
    macro = build_executable_macro_pool(problem)
    guarded = GlobalSinglePauliWordCandidateAdapter().executable_pool(
        problem
    )

    assert macro_parent.receipt.count == expected["parent_count"]
    assert macro_parent.receipt.ordered_labels_sha256 == (
        expected["parent_labels_sha256"]
    )
    assert macro_parent.receipt.ordered_pool_sha256 == (
        expected["parent_pool_sha256"]
    )
    assert singleton_parent.receipt.ordered_labels == (
        macro_parent.receipt.ordered_labels
    )
    assert singleton_parent.receipt.ordered_labels_sha256 == (
        macro_parent.receipt.ordered_labels_sha256
    )
    assert singleton_parent.receipt.ordered_pool_sha256 == (
        macro_parent.receipt.ordered_pool_sha256
    )

    assert macro.receipt.count == expected["macro_count"]
    assert len(macro.receipt.removed_labels) == (
        expected["macro_removed_count"]
    )
    assert macro.receipt.ordered_labels_sha256 == (
        expected["macro_labels_sha256"]
    )
    assert macro.receipt.ordered_pool_sha256 == expected["macro_pool_sha256"]
    assert macro.receipt.source_parent_ordered_labels_sha256 == (
        macro_parent.receipt.ordered_labels_sha256
    )
    removed = set(macro.receipt.removed_labels)
    assert removed.isdisjoint(macro.receipt.ordered_labels)
    assert [
        label
        for label in macro_parent.receipt.ordered_labels
        if label not in removed
    ] == list(macro.receipt.ordered_labels)
    assert set(macro_parent.receipt.ordered_labels) == (
        set(macro.receipt.ordered_labels) | removed
    )

    assert guarded.receipt.count == expected["guarded_count"]
    assert guarded.receipt.ordered_labels_sha256 == (
        expected["guarded_labels_sha256"]
    )
    assert guarded.receipt.ordered_pool_sha256 == (
        expected["guarded_pool_sha256"]
    )
    assert guarded.receipt.source_parent_ordered_labels_sha256 == (
        singleton_parent.receipt.ordered_labels_sha256
    )
    parent_generator_identities = {
        candidate.generator_identity
        for candidate in singleton_parent.candidates
    }
    assert all(candidate.parent_identities for candidate in guarded.candidates)
    assert all(
        set(candidate.parent_identities) <= parent_generator_identities
        for candidate in guarded.candidates
    )
    assert all(
        len(candidate.serialized_terms_exyz) == 1
        for candidate in guarded.candidates
    )

    macro_lineage = build_candidate_inventory_lineage_receipt(macro)
    guarded_lineage = build_candidate_inventory_lineage_receipt(guarded)
    assert macro_lineage.count == macro.receipt.count
    assert macro_lineage.pool_inventory_sha256 == macro.receipt.sha256
    assert macro_lineage.authority_binding()["sha256"] == (
        macro_lineage.sha256
    )
    assert [row.label for row in macro_lineage.ordered_rows] == list(
        macro.receipt.ordered_labels
    )
    assert guarded_lineage.count == guarded.receipt.count
    assert guarded_lineage.pool_inventory_sha256 == guarded.receipt.sha256
    assert any(
        len(row.parent_identities) > 1
        for row in guarded_lineage.ordered_rows
    )
    multi_parent = next(
        row
        for row in guarded_lineage.ordered_rows
        if len(row.parent_identities) > 1
    )
    admitted = CandidateLineageReceipt(
        representation_id=multi_parent.representation_id,
        candidate_label=multi_parent.label,
        generator_identity=multi_parent.generator_identity,
        parent_identities=multi_parent.parent_identities,
        insertion_position=2,
    )
    assert admitted.parent_identities == multi_parent.parent_identities
    assert admitted.insertion_position == 2
    assert admitted.to_dict()["sha256"] == admitted.sha256


def test_adapter_exposure_preserves_macro_identity_and_stages_singletons() -> None:
    problem = _problem(n_ph_max=3)
    macro_adapter = MacroCandidateAdapter()
    macro = macro_adapter.executable_pool(problem)
    retained_macro = macro.candidates[:3]
    exposed_macro = macro_adapter.expose_children(retained_macro)
    assert exposed_macro.candidates == retained_macro
    assert exposed_macro.metadata["exposure_policy"] == (
        "identity_on_retained_parents_v1"
    )

    singleton_adapter = SinglePauliWordCandidateAdapter()
    parents = singleton_adapter.parent_inventory(problem)
    retained_parents = parents.candidates[:3]
    exposed = singleton_adapter.expose_children(
        retained_parents,
        problem=problem,
    )
    retained_generator_identities = {
        candidate.generator_identity for candidate in retained_parents
    }
    assert exposed.metadata["exposure_scope"] == (
        "ra_retained_parent_shortlist_v1"
    )
    assert exposed.metadata["source_parent_count"] == len(retained_parents)
    assert exposed.metadata["exposure_policy"] == (
        "split_guard_project_canonicalize_dedupe_"
        "across_retained_parents_v1"
    )
    assert exposed.candidates
    assert all(
        set(candidate.parent_identities) <= retained_generator_identities
        for candidate in exposed.candidates
    )
    assert all(
        len(candidate.parent_identities) == 1
        for candidate in exposed.candidates
    )


def test_staged_exposure_differs_only_by_retained_parent_supply() -> None:
    problem = _problem(n_ph_max=3)
    adapter = SinglePauliWordCandidateAdapter()
    parents = adapter.parent_inventory(problem)
    staged_all = adapter.expose_children(
        parents.candidates,
        problem=problem,
    )
    global_pool = adapter.global_executable_pool(problem)

    staged_words = tuple(
        candidate.serialized_terms_exyz for candidate in staged_all.candidates
    )
    global_words = tuple(
        candidate.serialized_terms_exyz for candidate in global_pool.candidates
    )
    assert staged_words == global_words
    staged_word_identities = tuple(
        json.dumps(word, sort_keys=True) for word in staged_words
    )
    assert len(staged_word_identities) == len(set(staged_word_identities))
    assert all(
            not (
                len(word) == 1
                and set(str(word[0]["pauli_exyz"])) <= {"e"}
            )
        for word in staged_words
    )


def test_guarded_singleton_identity_is_independent_of_parent_supply() -> None:
    """One canonical Pauli direction keeps one identity across RA stages."""

    problem = _problem(n_ph_max=3)
    adapter = SinglePauliWordCandidateAdapter()
    parents = adapter.parent_inventory(problem)
    parent_by_identity = {
        candidate.generator_identity: candidate
        for candidate in parents.candidates
    }
    global_pool = adapter.global_executable_pool(problem)
    target = next(
        candidate
        for candidate in global_pool.candidates
        if candidate.label == "guarded_singleton::eeeeeexy"
    )

    staged_identities = set()
    for parent_identity in target.parent_identities:
        staged = adapter.expose_children(
            (parent_by_identity[parent_identity],),
            problem=problem,
        )
        staged_target = next(
            candidate
            for candidate in staged.candidates
            if candidate.label == target.label
        )
        staged_identities.add(staged_target.generator_identity)

    assert staged_identities == {target.generator_identity}


def test_late_resource_weighting_is_energy_only_in_phase_one() -> None:
    feature = _feature_with_nontrivial_resource_cost()
    common = {
        "lambda_2q": 1.0,
        "lambda_d": 0.0,
        "lambda_1q": 0.0,
        "lambda_theta": 0.0,
        "lambda_shot": 0.0,
    }
    late = phase1_score_payload(
        feature,
        SimpleScoreConfig(
            **common,
            resource_weighting_scope="late_resource_weighting_v1",
        ),
    )
    all_phase = phase1_score_payload(
        feature,
        SimpleScoreConfig(
            **common,
            resource_weighting_scope="all_phase_resource_weighting_v1",
        ),
    )

    assert late["resource_weighting_scope"] == (
        "late_resource_weighting_v1"
    )
    assert late["phase1_resource_weighting_active"] is False
    assert late["phase1_effective_cost_factor"] == pytest.approx(1.0)
    assert late["phase1_effective_burden"] == pytest.approx(1.0)
    assert late["phase1_raw_burden"] == pytest.approx(4.0)
    assert late["trust_region_score"] == pytest.approx(
        late["trust_region_gain"]
    )

    assert all_phase["resource_weighting_scope"] == (
        "all_phase_resource_weighting_v1"
    )
    assert all_phase["phase1_resource_weighting_active"] is True
    assert all_phase["phase1_effective_burden"] == pytest.approx(4.0)
    assert all_phase["phase1_raw_burden"] == pytest.approx(4.0)
    assert all_phase["trust_region_score"] == pytest.approx(
        float(all_phase["trust_region_gain"]) / 4.0
    )


def test_stationary_policy_receipt_forbids_active_gradient_work() -> None:
    receipt = PolicyEchoReceipt(
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_LATE,
    )
    assert receipt.active_gradient_indices_acquired == ()
    assert receipt.active_gradient_charge == 0

    with pytest.raises(ValueError, match="cannot acquire or charge"):
        PolicyEchoReceipt(
            active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
            resource_weighting_scope=RESOURCE_WEIGHTING_LATE,
            active_gradient_indices_acquired=(0,),
            active_gradient_charge=1,
        )
