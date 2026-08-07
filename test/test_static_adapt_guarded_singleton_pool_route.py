from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipelines.exact_bench import generic_static_adapt_variants as variants
from pipelines.exact_bench.table_i_canonical_cases import (
    TABLE_I_PAPER_I_HH_COMPLETION_SAME_CUTOFF_PROFILE,
    table_i_canonical_specs,
)
from pipelines.static_adapt.builders.shared_pauli_pool_contract import (
    SHARED_PAULI_POOL_MODE_GUARDED_SINGLETON_CHILDREN_ONLY_V1,
)
from pipelines.static_adapt.cli_config import _build_adapt_arg_parser
from pipelines.static_adapt.sr_snake_route_profile import (
    CANONICAL_SR_SNAKE_GUARDED_SINGLETON_POOL_V1_EXECUTION_SETTINGS,
    CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_V1_EXECUTION_SETTINGS,
    SR_ROUTE_PROFILE_GUARDED_SINGLETON_POOL_V1,
    SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_V1,
    canonical_sr_snake_guarded_singleton_pool_v1_contract,
    canonical_sr_snake_guarded_singleton_pool_v1_contract_sha256,
    canonical_sr_snake_macro_only_physical_lanes_v1_contract,
    canonical_sr_snake_macro_only_physical_lanes_v1_contract_sha256,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


def _parser():
    return _build_adapt_arg_parser(adapt_gradient_parity_rtol=1.0e-7)


_RETIRED_PASSIVE_PROFILE_FIELDS = {
    "phase_live_hysteresis_enabled",
    "phase1_prune_stale_age",
    "phase1_prune_stagnation_threshold",
    "phase1_prune_small_theta_abs",
    "phase1_prune_small_theta_relative",
}


def _assert_active_profile_settings(
    args: SimpleNamespace,
    settings: dict[str, object],
) -> None:
    for field, expected in settings.items():
        if field in _RETIRED_PASSIVE_PROFILE_FIELDS:
            assert not hasattr(args, field), field
        else:
            assert getattr(args, field) == expected, field


def _candidate(
    label: str,
    terms: list[tuple[str, float]],
) -> variants._PoolCandidate:
    polynomial = PauliPolynomial(
        "JW",
        [
            PauliTerm(len(pauli), ps=pauli, pc=coefficient)
            for pauli, coefficient in terms
        ],
    )
    return variants._PoolCandidate(
        label=label,
        polynomial=polynomial,
        support=tuple(
            sorted(
                {
                    qubit
                    for pauli, _coefficient in terms
                    for qubit in range(len(pauli))
                    if pauli[len(pauli) - 1 - qubit] != "e"
                }
            )
        ),
        pauli_labels_exyz=tuple(pauli for pauli, _coefficient in terms),
        construction="full_meta::full_meta",
    )


def _hh_nph3_context() -> SimpleNamespace:
    return SimpleNamespace(
        family_key="hh",
        request=SimpleNamespace(
            problem_key="hh",
            num_sites=2,
            ordering="blocked",
            n_ph_max=3,
            boson_encoding="binary",
        ),
        layout=SimpleNamespace(total_qubits=8, fermion_qubits=4),
        sector=SimpleNamespace(num_particles=(1, 1)),
    )


def _expand(
    parents: tuple[variants._PoolCandidate, ...],
):
    return variants._expand_pool_with_shared_pauli_children(
        pool=parents,
        context=_hh_nph3_context(),
        config=variants._get_config("static_full_meta_append_adapt_vqe"),
        mode=SHARED_PAULI_POOL_MODE_GUARDED_SINGLETON_CHILDREN_ONLY_V1,
        symmetry_policy="hard_guard",
        max_subset_size=1,
        max_terms=64,
    )


def test_guarded_singleton_pool_globally_deduplicates_and_preserves_lineage() -> None:
    parents = (
        _candidate(
            "full_meta::parent_a",
            [("zeeeeeee", 0.25), ("eeeeeeez", 0.5)],
        ),
        _candidate(
            "full_meta::parent_b",
            [("zeeeeeee", -0.75), ("ezeeeeee", 0.125)],
        ),
    )

    expanded, meta = _expand(parents)

    assert {candidate.label for candidate in expanded} == {
        "guarded_singleton::ezeeeeee",
        "guarded_singleton::zeeeeeee",
        "guarded_singleton::eeeeeeez",
    }
    assert all(
        candidate.runtime_split_representation == "guarded_singleton_child"
        for candidate in expanded
    )
    assert all(len(candidate.pauli_labels_exyz) == 1 for candidate in expanded)
    assert all(
        candidate.generator_metadata["is_macro_generator"] is False
        for candidate in expanded
    )
    duplicate = next(
        candidate
        for candidate in expanded
        if candidate.pauli_labels_exyz == ("zeeeeeee",)
    )
    duplicate_contract = duplicate.generator_metadata[
        "shared_pauli_pool_contract"
    ]
    assert duplicate_contract["parent_labels"] == [
        "full_meta::parent_a",
        "full_meta::parent_b",
    ]
    assert meta["guarded_singleton_global_duplicate_count"] == 1
    assert meta["parent_candidate_count"] == 0
    assert meta["guarded_singleton_projection_applied"] is False
    assert meta["candidate_representation_counts"] == {
        "parent": 0,
        "child_set": 0,
        "projected_singleton_child": 0,
        "guarded_singleton_child": 3,
    }


def test_guarded_singleton_pool_hard_rejects_sector_breaking_child() -> None:
    parents = (
        _candidate(
            "full_meta::mixed_sector",
            [("eeeeeeez", 0.5), ("eeeeeeex", 0.5)],
        ),
    )

    expanded, meta = _expand(parents)

    assert [candidate.pauli_labels_exyz for candidate in expanded] == [
        ("eeeeeeez",)
    ]
    assert meta["guarded_singleton_symmetry_rejected_count"] == 1
    assert expanded[0].runtime_split_symmetry_gate["checked"] is True
    assert expanded[0].runtime_split_symmetry_gate["passed"] is True


def test_guarded_singleton_pool_excludes_identity_without_projection() -> None:
    parents = (
        _candidate(
            "full_meta::identity_plus_direction",
            [("eeeeeeee", 0.5), ("eeeeeeez", 0.5)],
        ),
    )

    expanded, meta = _expand(parents)

    assert [candidate.pauli_labels_exyz for candidate in expanded] == [
        ("eeeeeeez",)
    ]
    assert meta["guarded_singleton_null_identity_count"] == 1
    assert meta["guarded_singleton_null_exclusions"][0]["reason"] == (
        "raw_singleton_is_identity_global_phase_direction"
    )
    assert meta["guarded_singleton_projection_applied"] is False
    assert meta["guarded_singleton_padding_filter"][
        "projection_applied_before_child_phase1_evaluation"
    ] is False


def test_guarded_singleton_route_is_global_and_has_no_lanes() -> None:
    args = _parser().parse_args(
        [
            "--sr-route-profile",
            "sr_snake_guarded_singleton_pool_v1",
            "--adapt-max-depth",
            "50",
        ]
    )

    assert args.sr_route_profile_request == (
        SR_ROUTE_PROFILE_GUARDED_SINGLETON_POOL_V1
    )
    assert args.sr_route_profile_resolved == (
        SR_ROUTE_PROFILE_GUARDED_SINGLETON_POOL_V1
    )
    assert args.sr_route_profile_contract == (
        canonical_sr_snake_guarded_singleton_pool_v1_contract()
    )
    assert args.sr_route_profile_contract_sha256 == (
        canonical_sr_snake_guarded_singleton_pool_v1_contract_sha256()
    )
    _assert_active_profile_settings(
        args,
        CANONICAL_SR_SNAKE_GUARDED_SINGLETON_POOL_V1_EXECUTION_SETTINGS,
    )
    assert args.adapt_max_depth == 50
    assert args.static_lane_route == "algebraic"
    assert args.phase3_selector_policy == "hardware_resolvable_v1"
    assert args.phase3_runtime_split_mode == "off"
    assert args.shared_pauli_pool_mode == (
        SHARED_PAULI_POOL_MODE_GUARDED_SINGLETON_CHILDREN_ONLY_V1
    )
    invariants = args.sr_route_profile_contract["semantic_invariants"]
    assert invariants["shortlist_population_policy"] == (
        "single_global_population_v1"
    )
    assert invariants["physical_operator_lanes_active"] is False
    assert invariants["algebraic_lanes_active"] is False


@pytest.mark.parametrize(
    "override",
    [
        ("--static-lane-route", "physical_operator_type"),
        ("--phase3-selector-policy", "algebraic_nested_v1"),
        ("--shared-pauli-pool-mode", "projected_singleton_children_only_v1"),
    ],
)
def test_guarded_singleton_route_rejects_lane_or_projection_drift(
    override: tuple[str, str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _parser().parse_args(
            [
                "--sr-route-profile",
                "sr_snake_guarded_singleton_pool_v1",
                "--adapt-max-depth",
                "50",
                *override,
            ]
        )
    assert exc_info.value.code == 2


def test_completion_real_guarded_singleton_pool_hashes_nph3_and_nph7() -> None:
    expected = {
        "hh_L2_nph3_completion_weak_weak": (
            123,
            948,
            682,
            "1d4445224f4e937891e0e4a8877aeb44a74beb1f07728e7d5eca64c55b2ae40a",
        ),
        "hh_L2_nph7_completion_weak_strong": (
            171,
            6508,
            2248,
            "4f1d20f06550d99e9f9116ccfdcec607a8056f4e2f27f48883cefb23a064a82c",
        ),
    }
    specs = table_i_canonical_specs(
        TABLE_I_PAPER_I_HH_COMPLETION_SAME_CUTOFF_PROFILE
    )
    for spec in (specs[0], specs[3]):
        context = variants._resolve_context_from_spec(spec)
        parents = variants.build_full_meta_candidate_pool(context, max_terms=None)
        children, meta = variants._expand_pool_with_shared_pauli_children(
            pool=parents,
            context=context,
            config=variants._get_config("static_full_meta_append_adapt_vqe"),
            mode=SHARED_PAULI_POOL_MODE_GUARDED_SINGLETON_CHILDREN_ONLY_V1,
            symmetry_policy="hard_guard",
            max_subset_size=1,
            max_terms=9000,
        )
        parent_count, child_count, duplicate_count, pool_hash = expected[
            str(spec.benchmark_id)
        ]
        assert len(parents) == parent_count
        assert len(children) == child_count
        assert meta["guarded_singleton_global_duplicate_count"] == duplicate_count
        assert meta["guarded_singleton_padding_rejected_count"] == 0
        assert meta["guarded_singleton_null_identity_count"] == 1
        assert meta["ordered_pool_hash"] == pool_hash
        assert all(
            child.runtime_split_representation == "guarded_singleton_child"
            for child in children
        )
        assert all(len(child.pauli_labels_exyz) == 1 for child in children)


def test_macro_only_route_retains_physical_lanes_and_never_builds_children() -> None:
    args = _parser().parse_args(
        [
            "--sr-route-profile",
            "sr_snake_macro_only_physical_lanes_v1",
            "--adapt-max-depth",
            "50",
        ]
    )

    assert args.sr_route_profile_request == (
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_V1
    )
    assert args.sr_route_profile_resolved == (
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_V1
    )
    assert args.sr_route_profile_contract == (
        canonical_sr_snake_macro_only_physical_lanes_v1_contract()
    )
    assert args.sr_route_profile_contract_sha256 == (
        canonical_sr_snake_macro_only_physical_lanes_v1_contract_sha256()
    )
    _assert_active_profile_settings(
        args,
        CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_V1_EXECUTION_SETTINGS,
    )
    assert args.static_lane_route == "physical_operator_type"
    assert args.physical_lane_shortlist_aggressiveness == 3
    assert args.phase3_runtime_split_mode == "off"
    assert args.adapt_child_pool_expansion_mode == "off"
    assert args.shared_pauli_pool_mode == "off"
    invariants = args.sr_route_profile_contract["semantic_invariants"]
    assert invariants["candidate_representation"] == (
        "intact_logical_parent_generator_v1"
    )
    assert invariants["generated_pauli_children_active"] is False
    assert invariants["physical_operator_lanes_active"] is True


def test_macro_only_one_sided_cost_route_changes_only_cost_normalization() -> None:
    baseline = _parser().parse_args(
        [
            "--sr-route-profile",
            "sr_snake_macro_only_physical_lanes_v1",
            "--adapt-max-depth",
            "50",
        ]
    )
    ablation = _parser().parse_args(
        [
            "--sr-route-profile",
            "sr_snake_macro_only_physical_lanes_one_sided_cost_v1",
            "--adapt-max-depth",
            "50",
        ]
    )

    baseline_settings = dict(baseline.sr_route_profile_contract["execution_settings"])
    ablation_settings = dict(ablation.sr_route_profile_contract["execution_settings"])
    assert baseline_settings.pop("phase3_hardware_cost_normalization_mode") == (
        "family_robust_symmetric_arctan_v1"
    )
    assert ablation_settings.pop("phase3_hardware_cost_normalization_mode") == (
        "family_robust_v1"
    )
    assert ablation_settings == baseline_settings
    assert ablation.static_lane_route == "physical_operator_type"
    assert ablation.phase3_runtime_split_mode == "off"
    assert ablation.adapt_child_pool_expansion_mode == "off"
    assert ablation.shared_pauli_pool_mode == "off"


@pytest.mark.parametrize(
    "override",
    [
        ("--static-lane-route", "algebraic"),
        ("--phase3-runtime-split-mode", "shortlist_pauli_children_v1"),
        ("--adapt-child-pool-expansion-mode", "pauli_children_v1"),
        ("--shared-pauli-pool-mode", "guarded_singleton_children_only_v1"),
    ],
)
def test_macro_only_route_rejects_child_or_lane_drift(
    override: tuple[str, str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _parser().parse_args(
            [
                "--sr-route-profile",
                "sr_snake_macro_only_physical_lanes_v1",
                "--adapt-max-depth",
                "50",
                *override,
            ]
        )
    assert exc_info.value.code == 2


def test_completion_real_macro_parent_pools_remain_intact_at_nph3_and_nph7() -> None:
    expected = {
        "hh_L2_nph3_completion_weak_weak": (123, 34, 89),
        "hh_L2_nph7_completion_weak_strong": (171, 80, 91),
    }
    specs = table_i_canonical_specs(
        TABLE_I_PAPER_I_HH_COMPLETION_SAME_CUTOFF_PROFILE
    )
    for spec in (specs[0], specs[3]):
        context = variants._resolve_context_from_spec(spec)
        parents = variants.build_full_meta_candidate_pool(context, max_terms=None)
        parent_count, singleton_parent_count, macro_parent_count = expected[
            str(spec.benchmark_id)
        ]
        assert len(parents) == parent_count
        assert sum(len(parent.pauli_labels_exyz) == 1 for parent in parents) == (
            singleton_parent_count
        )
        assert sum(len(parent.pauli_labels_exyz) > 1 for parent in parents) == (
            macro_parent_count
        )
        assert all(
            parent.runtime_split_representation == "parent" for parent in parents
        )
