from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.scaffold.hh_continuation_generators import (
    build_generator_metadata,
    build_runtime_split_child_sets,
    build_runtime_split_children,
)
from pipelines.scaffold.hh_continuation_scoring import (
    FullScoreConfig,
    SimpleScoreConfig,
    resolve_hardware_cost_lambdas,
)
from pipelines.static_adapt import beam_search, cli_config, engine_support
from pipelines.static_adapt.builders.shared_pauli_pool_contract import (
    SHARED_PAULI_POOL_MODE_CHILD_SETS_V1,
    SharedPauliPoolParent,
    build_shared_pauli_child_pool,
)
from src.quantum.pauli_polynomial_class import (
    PauliPolynomial,
    fermion_minus_operator,
    fermion_plus_operator,
)
from src.quantum.pauli_words import PauliTerm


def _branch(*, branch_id: int, energy: float, cost: float, label: str) -> SimpleNamespace:
    return SimpleNamespace(
        branch_id=int(branch_id),
        depth_local=1,
        energy_current=float(energy),
        cumulative_beam_cost=float(cost),
        cumulative_selector_score=0.0,
        cumulative_selector_burden=float(1.0 + cost),
        selected_ops=[SimpleNamespace(label=str(label))],
        theta=np.asarray([0.1 * branch_id], dtype=float),
    )


def _mixed_macro_poly() -> PauliPolynomial:
    preserving = (-1j) * (
        fermion_plus_operator("JW", 4, 1) * fermion_minus_operator("JW", 4, 0)
        - fermion_plus_operator("JW", 4, 0) * fermion_minus_operator("JW", 4, 1)
    )
    return preserving + PauliPolynomial("JW", [PauliTerm(4, ps="zeee", pc=0.25)])


def test_canonical_beam_replaces_legacy_scalar_with_gain_per_added_cost() -> None:
    low_energy_high_cost = _branch(branch_id=1, energy=-8.0, cost=9.0, label="deep")
    efficient = _branch(branch_id=2, energy=-5.0, cost=1.0, label="efficient")

    legacy, _ = engine_support._beam_prune_energy_cost_pareto_with_audit(
        [low_energy_high_cost, efficient],
        cap=1,
        lambda_beam=0.1,
    )
    canonical, audit = engine_support._beam_prune_gain_per_added_cost_pareto_with_audit(
        [low_energy_high_cost, efficient],
        cap=1,
        energy_root=0.0,
        legacy_lambda_beam=0.1,
    )

    assert [branch.branch_id for branch in legacy] == [1]
    assert [branch.branch_id for branch in canonical] == [2]
    assert audit["phase_local_normalized_scores_accumulated"] is False
    by_id = {row["branch_id"]: row for row in audit["input_branch_keys"]}
    assert by_id[1]["beam_survival_score"] == pytest.approx(0.8)
    assert by_id[2]["beam_survival_score"] == pytest.approx(2.5)


def test_canonical_beam_pareto_filters_realized_energy_and_added_burden() -> None:
    dominant = _branch(branch_id=1, energy=-3.0, cost=1.0, label="dominant")
    dominated = _branch(branch_id=2, energy=-2.0, cost=2.0, label="dominated")

    kept, audit = engine_support._beam_prune_gain_per_added_cost_pareto_with_audit(
        [dominated, dominant],
        cap=2,
        energy_root=0.0,
    )

    assert [branch.branch_id for branch in kept] == [1]
    assert audit["dominance_events"][0]["dominating_branch_id"] == 1
    assert audit["dominance_events"][0]["dominated_branch_id"] == 2


def test_canonical_beam_ignores_legacy_lambda_and_batch_structure() -> None:
    branches = [
        _branch(branch_id=1, energy=-4.0, cost=5.0, label="a"),
        _branch(branch_id=2, energy=-3.0, cost=1.0, label="b"),
    ]

    low_lambda, low_audit = beam_search._beam_prune_for_policy(
        branches,
        cap=1,
        ordered_batch_beam_mode=True,
        lambda_beam=0.0,
        canonical_beam_survival=True,
        energy_root=0.0,
    )
    high_lambda, high_audit = beam_search._beam_prune_for_policy(
        branches,
        cap=1,
        ordered_batch_beam_mode=False,
        lambda_beam=1e6,
        canonical_beam_survival=True,
        energy_root=0.0,
    )

    assert [branch.branch_id for branch in low_lambda] == [2]
    assert [branch.branch_id for branch in high_lambda] == [2]
    assert low_audit is not None and high_audit is not None
    assert low_audit["kept_branch_ids"] == high_audit["kept_branch_ids"]
    assert low_audit["legacy_lambda_beam_effect"] == "ignored"
    assert high_audit["legacy_lambda_beam_effect"] == "ignored"


def test_historical_depth8_beam_event_replays_under_both_survival_rules() -> None:
    energy_low = -0.9173376028478099
    energy_efficient = -0.91732335502389
    cost_high = 0.8456834108213811
    cost_low = 0.8043586257671329
    branches = [
        _branch(branch_id=25, energy=energy_low, cost=cost_high, label="branch25"),
        _branch(branch_id=26, energy=energy_efficient, cost=cost_low, label="branch26"),
        _branch(branch_id=27, energy=energy_low, cost=cost_high, label="branch27"),
        _branch(branch_id=28, energy=energy_efficient, cost=cost_low, label="branch28"),
    ]

    legacy_lambda_0p005, _ = engine_support._beam_prune_energy_cost_pareto_with_audit(
        branches,
        cap=3,
        lambda_beam=0.005,
    )
    legacy_lambda_0, _ = engine_support._beam_prune_energy_cost_pareto_with_audit(
        branches,
        cap=3,
        lambda_beam=0.0,
    )
    canonical, audit = engine_support._beam_prune_gain_per_added_cost_pareto_with_audit(
        branches,
        cap=3,
        energy_root=1.25,
        legacy_lambda_beam=0.005,
    )

    assert [branch.branch_id for branch in legacy_lambda_0p005] == [26, 28]
    assert [branch.branch_id for branch in legacy_lambda_0] == [25, 27]
    assert [branch.branch_id for branch in canonical] == [26, 28, 25]
    assert audit["dominated_count"] == 0
    scores = {
        row["branch_id"]: row["beam_survival_score"]
        for row in audit["input_branch_keys"]
    }
    assert scores[26] == pytest.approx(scores[28])
    assert scores[25] == pytest.approx(scores[27])
    assert scores[26] > scores[25]


def test_exact_pauli_word_subset_cardinalities_do_not_fall_back_to_singletons() -> None:
    parent = _mixed_macro_poly()
    parent_meta = build_generator_metadata(
        label="macro",
        polynomial=parent,
        family_id="uccsd",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        symmetry_spec=None,
    )
    children = build_runtime_split_children(
        parent_label="macro",
        polynomial=parent,
        family_id="uccsd",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        split_mode="shortlist_pauli_children_v1",
        parent_generator_metadata=parent_meta.__dict__,
        symmetry_spec=None,
    )

    pairs = build_runtime_split_child_sets(
        parent_label="macro",
        family_id="uccsd",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        split_mode="shortlist_pauli_children_v1",
        children=children,
        parent_generator_metadata=parent_meta.__dict__,
        symmetry_spec=None,
        subset_sizes=(2,),
        max_subset_size=1,
    )
    singleton_and_pair = build_runtime_split_child_sets(
        parent_label="macro",
        family_id="uccsd",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        split_mode="shortlist_pauli_children_v1",
        children=children,
        parent_generator_metadata=parent_meta.__dict__,
        symmetry_spec=None,
        subset_sizes=(1, 2),
    )
    unavailable = build_runtime_split_child_sets(
        parent_label="macro",
        family_id="uccsd",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        split_mode="shortlist_pauli_children_v1",
        children=children,
        parent_generator_metadata=parent_meta.__dict__,
        symmetry_spec=None,
        subset_sizes=(4,),
    )

    assert {tuple(row["child_indices"]) for row in pairs} == {
        (0, 1),
        (0, 2),
        (1, 2),
    }
    assert all(row["subset_cardinality"] == 2 for row in pairs)
    assert len(singleton_and_pair) == 6
    assert {row["subset_cardinality"] for row in singleton_and_pair} == {1, 2}
    assert unavailable == []


def test_shared_pauli_pool_preserves_exact_pair_only_request() -> None:
    result = build_shared_pauli_child_pool(
        parents=(
            SharedPauliPoolParent(
                label="macro",
                polynomial=_mixed_macro_poly(),
                family_id="uccsd",
            ),
        ),
        mode=SHARED_PAULI_POOL_MODE_CHILD_SETS_V1,
        symmetry_policy="off",
        subset_sizes=(2,),
        max_subset_size=1,
        problem_key="hh",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        max_terms=10,
    )

    children = [
        candidate
        for candidate in result.candidates
        if candidate.representation == "child_set"
    ]
    assert result.meta["subset_sizes"] == [2]
    assert result.meta["subset_size_semantics"] == (
        "exact_allowed_pauli_word_cardinalities"
    )
    assert {candidate.child_indices for candidate in children} == {
        (0, 1),
        (0, 2),
        (1, 2),
    }
    assert all(len(candidate.child_indices) == 2 for candidate in children)


def test_shared_canonical_cost_weights_and_cli_exact_subset_inputs() -> None:
    expected = {"2q": 0.20, "d": 0.20, "1q": 0.05, "theta": 0.05, "shot": 0.15}
    assert resolve_hardware_cost_lambdas(SimpleScoreConfig())[0] == pytest.approx(expected)
    assert resolve_hardware_cost_lambdas(FullScoreConfig())[0] == pytest.approx(expected)

    parser = cli_config._build_adapt_arg_parser(adapt_gradient_parity_rtol=1e-8)
    defaults = parser.parse_args([])
    assert (
        defaults.cost_lambda_2q,
        defaults.cost_lambda_d,
        defaults.cost_lambda_1q,
        defaults.cost_lambda_theta,
        defaults.cost_lambda_shot,
    ) == pytest.approx((0.20, 0.20, 0.05, 0.05, 0.15))
    exact = parser.parse_args(
        [
            "--phase3-runtime-split-subset-sizes",
            "2",
            "--adapt-child-pool-expansion-subset-sizes",
            "1,2",
            "--shared-pauli-pool-subset-sizes",
            "3",
        ]
    )
    assert exact.phase3_runtime_split_subset_sizes == "2"
    assert exact.adapt_child_pool_expansion_subset_sizes == "1,2"
    assert exact.shared_pauli_pool_subset_sizes == "3"
