from __future__ import annotations

import json

import pytest

from pipelines.static_adapt.paper_i_config import (
    CanonicalSnakeDefaults,
    PAPER_I_CONFIGURATION,
    PAPER_I_CONFIG_SCHEMA,
    PaperICostWeights,
    PaperIMechanismOverrides,
)


def test_paper_i_configuration_separates_capabilities_from_hh_reproduction() -> None:
    config = PAPER_I_CONFIGURATION

    assert config.capabilities.batching is True
    assert config.capabilities.beam_search is True
    assert config.hh_displayed_results.batching_enabled is False
    assert config.hh_displayed_results.beam_enabled is False
    assert config.canonical.first_class_route_ids == ("route_a",)
    assert config.canonical.benchmark_sibling_algorithms == (
        "append_only_adapt",
        "geo_adapt",
    )


def test_paper_i_canonical_defaults_match_approved_route_a_contract() -> None:
    canonical = PAPER_I_CONFIGURATION.canonical

    assert canonical.cost_enabled is True
    assert canonical.cost_weights.as_lambda_dict() == pytest.approx(
        {"2q": 0.20, "d": 0.20, "1q": 0.05, "theta": 0.05, "shot": 0.15}
    )
    assert canonical.reoptimization_policy == "full"
    assert canonical.batching_enabled is True
    assert canonical.batch_selection_mode == "combinatorial_reduced_plane"
    assert canonical.beam_enabled is False
    assert canonical.pruning_enabled is True
    assert canonical.pauli_child_pool_enabled is True
    assert canonical.phase0_enabled is False
    assert canonical.macro_phase3_enabled is False
    assert canonical.child_phase3_enabled is False
    assert canonical.canonical_stage_sequence == (
        "macro_phase1",
        "macro_phase2",
        "singleton_child_expansion",
        "global_child_phase1",
        "global_child_phase2",
        "joint_ansatz_plus_batch_selector",
        "admission_full_refit_prune",
    )
    assert canonical.final_selection_authority == (
        "joint_ansatz_plus_batch_schur_v1"
    )
    assert canonical.joint_batch_context_mode == "full_ansatz_v1"
    assert canonical.batch_score_formula == (
        "DeltaE_joint_relaxed(B)/(1+K(B))"
    )
    assert canonical.insertion_position_scope == (
        "full_logical_ansatz_commutation_classes_every_depth_v2"
    )
    assert canonical.geometry_window_scope == "full"
    assert canonical.static_lane_route == "physical_operator_type"
    assert canonical.maturity_scheduling_enabled is False
    assert canonical.pauli_word_subset_sizes == (1,)
    assert canonical.child_symmetry_policy == "hard_guard"
    assert canonical.noise_execution_surface == "separate_wrapper"
    assert (
        PAPER_I_CONFIGURATION.hh_displayed_results
        .commutation_reduced_insertion_position_search
        is True
    )
    assert not hasattr(
        PAPER_I_CONFIGURATION.hh_displayed_results,
        "full_insertion_position_search",
    )


def test_paper_i_cost_disable_zeroes_weights_without_changing_contract_type() -> None:
    disabled = PAPER_I_CONFIGURATION.cost_weights(enabled=False)

    assert isinstance(disabled, PaperICostWeights)
    assert disabled.enabled is False
    assert disabled.as_lambda_dict() == pytest.approx(
        {"2q": 0.0, "d": 0.0, "1q": 0.0, "theta": 0.0, "shot": 0.0}
    )


def test_paper_i_subset_sizes_are_exact_and_never_gain_singletons() -> None:
    pair_only = PaperIMechanismOverrides(pauli_word_subset_sizes=(2,))
    singleton_and_pair = PaperIMechanismOverrides(pauli_word_subset_sizes=(1, 2))

    assert pair_only.pauli_word_subset_sizes == (2,)
    assert singleton_and_pair.pauli_word_subset_sizes == (1, 2)
    with pytest.raises(ValueError, match="positive"):
        PaperIMechanismOverrides(pauli_word_subset_sizes=(0, 1))
    with pytest.raises(ValueError, match="duplicate"):
        CanonicalSnakeDefaults(pauli_word_subset_sizes=(1, 1))
    with pytest.raises(ValueError, match="batch_selection_mode"):
        PaperIMechanismOverrides(batch_selection_mode="legacy_batch")


def test_paper_i_compatibility_is_quarantined_but_readable() -> None:
    compatibility = PAPER_I_CONFIGURATION.historical_compatibility

    assert compatibility.legacy_routes_importable is True
    assert compatibility.legacy_routes_first_class is False
    assert compatibility.legacy_cli_quarantine_complete is False
    assert compatibility.legacy_cli_requires_explicit_gate is True
    assert compatibility.historical_payload_reading_enabled is True
    assert compatibility.legacy_beam_lambda_affects_route_a is False
    assert compatibility.maturity_scheduling_is_canonical is False


def test_paper_i_configuration_payload_is_stable_json_data() -> None:
    payload = PAPER_I_CONFIGURATION.as_dict()

    assert payload["schema"] == PAPER_I_CONFIG_SCHEMA
    assert payload["authority"] == "typed_code_contract_not_runtime_manuscript_parse"
    assert payload["canonical"]["pauli_word_subset_size_semantics"] == (
        "exact_allowed_cardinalities"
    )
    assert json.loads(json.dumps(payload)) == payload
