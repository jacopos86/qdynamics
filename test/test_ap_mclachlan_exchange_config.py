import pytest
from dataclasses import replace
"""Typed-boundary contract for the AP generalized-exchange adapter."""

from pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory import (
    SupportPatchControllerConfig,
)
from pipelines.time_dynamics.ap_mclachlan.exchange_config import (
    APGeneralizedExchangeConfig,
    ExchangeCertificationConfig,
    ExchangeEligibilityConfig,
    ExchangeScoreConfig,
    ExchangeSearchBudget,
)


@pytest.mark.parametrize("limit", [None, 0, 1, 3])
def test_deletion_cardinality_flows_through_raw_and_nested_provenance(limit) -> None:
    route = SupportPatchControllerConfig(max_deletion_batch_size=limit)
    typed = APGeneralizedExchangeConfig.from_route_config(route)
    assert typed.search.deletion_cardinality == limit
    raw = route.to_json_dict()
    nested = raw["generalized_exchange"]["search"]
    if limit is None:
        assert "max_deletion_batch_size" not in raw
        assert "deletion_cardinality" not in nested
        assert raw == SupportPatchControllerConfig().to_json_dict()
    else:
        assert raw["max_deletion_batch_size"] == limit
        assert nested["deletion_cardinality"] == limit


@pytest.mark.parametrize("limit", [-1, 0.5, 1.0, True, "1"])
def test_deletion_cardinality_is_validated_at_raw_and_typed_boundaries(limit) -> None:
    with pytest.raises(ValueError, match="max_deletion_batch_size"):
        SupportPatchControllerConfig(max_deletion_batch_size=limit)
    base = APGeneralizedExchangeConfig.from_route_config(SupportPatchControllerConfig())
    with pytest.raises(ValueError, match="deletion_cardinality"):
        replace(base.search, deletion_cardinality=limit)


@pytest.mark.parametrize("policy", ["avqds", "fixed"])
def test_explicit_deletion_cardinality_cannot_be_ignored_by_another_dynamics_policy(policy) -> None:
    with pytest.raises(ValueError, match="max_deletion_batch_size requires"):
        SupportPatchControllerConfig(dynamics_policy=policy, max_deletion_batch_size=1)


def test_route_transport_is_partitioned_into_four_exchange_inputs() -> None:
    route = SupportPatchControllerConfig(
        debt_policy="drift_ranked",
        prune_history_lambda=0.0,
        max_insertion_batch_size=1,
    )
    config = APGeneralizedExchangeConfig.from_route_config(route)

    assert isinstance(config.score, ExchangeScoreConfig)
    assert isinstance(config.eligibility, ExchangeEligibilityConfig)
    assert isinstance(config.certification, ExchangeCertificationConfig)
    assert isinstance(config.search, ExchangeSearchBudget)
    assert config.score.debt_policy == "drift_ranked"
    assert config.score.history_weight == 0.0
    assert config.search.insertion_cardinality == 1
    recorded = route.to_json_dict()["generalized_exchange"]
    # Five partitions since 2026-09-02: the closed route bundle joined the
    # four question-scoped inputs (see exchange_config.ExchangeRouteConfig).
    assert set(recorded) == {"score", "eligibility", "certification", "search", "route"}


def test_retired_selector_fields_are_not_transport_configuration() -> None:
    route = SupportPatchControllerConfig()
    for retired in (
        "append_rung_set_cap",
        "append_prefilter_size",
        "append_gain_threshold",
        "append_batch_score_threshold",
        "prune_rung_set_cap",
        "prune_prefilter_size",
        "max_prune_commits",
        "max_exchange_append_branches",
        "max_exchange_prune_branches",
        "max_exchange_pair_count",
        "exchange_append_score_min",
        "exchange_prune_score_min",
        "exchange_residual_dominance_tol",
        "exchange_cost_dominance_tol",
        "patch_utility_refit_weight",
        "patch_utility_velocity_weight",
        "patch_utility_threshold",
        "exchange_cost_alpha",
        "prune_enabled",
        "prune_commit_enabled",
        "max_prune_batch_size",
        "prune_projection_enabled",
        "prune_shadow_enabled",
        "prune_persistence_required",
        "prune_persistence_mode",
        "prune_atom_history_fraction",
        "prune_patch_smoothness_enabled",
    ):
        assert not hasattr(route, retired), retired


def test_scoring_workers_flow_from_route_config_into_search_budget() -> None:
    from pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory import (
        SupportPatchControllerConfig,
    )

    config = APGeneralizedExchangeConfig.from_route_config(
        SupportPatchControllerConfig(support_patch_scoring_workers=4)
    )
    assert config.search.scoring_workers == 4
    assert config.to_json_dict()["search"]["scoring_workers"] == 4


def test_route_bundles_are_closed_and_resolve_from_route_config() -> None:
    from pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory import (
        SupportPatchControllerConfig,
    )
    from pipelines.time_dynamics.ap_mclachlan.exchange_config import (
        EXCHANGE_ROUTE_BUNDLES,
        EXCHANGE_ROUTE_BUNDLE_GAMMA_COMPILED,
        EXCHANGE_ROUTE_BUNDLE_HISTORICAL,
        EXCHANGE_ROUTE_BUNDLE_LEGACY_PROXY_FIXED,
        EXCHANGE_ROUTE_BUNDLE_Q_COMPILED,
        resolve_route_bundle,
    )

    assert set(EXCHANGE_ROUTE_BUNDLES) == {
        EXCHANGE_ROUTE_BUNDLE_HISTORICAL,
        EXCHANGE_ROUTE_BUNDLE_LEGACY_PROXY_FIXED,
        EXCHANGE_ROUTE_BUNDLE_Q_COMPILED,
        EXCHANGE_ROUTE_BUNDLE_GAMMA_COMPILED,
    }
    historical = resolve_route_bundle(EXCHANGE_ROUTE_BUNDLE_HISTORICAL)
    assert historical.is_historical and not historical.uses_population_scoring
    assert historical.compiler is None and not historical.needs_compiled_cost
    q = resolve_route_bundle(EXCHANGE_ROUTE_BUNDLE_Q_COMPILED)
    gamma = resolve_route_bundle(EXCHANGE_ROUTE_BUNDLE_GAMMA_COMPILED)
    # Q and Gamma share the identical cost map and kernel; only the ordinate differs.
    assert q.cost_model_id == gamma.cost_model_id
    assert q.kernel_id == gamma.kernel_id
    assert q.compiler == gamma.compiler
    assert q.objective_id != gamma.objective_id
    assert gamma.needs_gamma and not q.needs_gamma
    assert q.compiler.backend_name == "FakeMarrakesh"
    assert q.compiler.optimization_level == 1 and q.compiler.transpile_seed == 7
    assert q.compiler.structural_angle == 1.0
    assert q.compiler.compile_timeout_seconds == 300.0
    with pytest.raises(TypeError):
        q.compiler.coordinate_weights["delta_n2q"] = 9.0
    with pytest.raises(TypeError):
        EXCHANGE_ROUTE_BUNDLES["made_up"] = q
    with pytest.raises(ValueError, match="unknown exchange route bundle"):
        resolve_route_bundle("paper_ii_exchange_made_up_v9")

    # Default config resolves to the historical bundle and inherits tau_del.
    base = SupportPatchControllerConfig(prune_patch_smoothness_eta_max=2.5e-3)
    resolved = APGeneralizedExchangeConfig.from_route_config(base)
    assert resolved.route.bundle.is_historical
    assert resolved.route.maintenance_loss_tolerance_requested is None
    assert resolved.route.maintenance_loss_tolerance == 2.5e-3
    assert resolved.route.max_compiled_patches_per_checkpoint is None
    assert resolved.to_json_dict()["route"]["bundle"]["bundle_id"] == (
        EXCHANGE_ROUTE_BUNDLE_HISTORICAL
    )

    explicit = SupportPatchControllerConfig(
        exchange_route_bundle=EXCHANGE_ROUTE_BUNDLE_GAMMA_COMPILED,
        maintenance_loss_tolerance=4.0e-4,
        max_compiled_patches_per_checkpoint=5000,
    )
    resolved = APGeneralizedExchangeConfig.from_route_config(explicit)
    assert resolved.route.bundle.needs_gamma
    assert resolved.route.maintenance_loss_tolerance == 4.0e-4
    assert resolved.route.max_compiled_patches_per_checkpoint == 5000
    provenance = explicit.to_json_dict()
    assert provenance["exchange_route_bundle"] == EXCHANGE_ROUTE_BUNDLE_GAMMA_COMPILED
    assert provenance["maintenance_loss_tolerance"] == 4.0e-4
    with pytest.raises(ValueError):
        SupportPatchControllerConfig(exchange_route_bundle="nope")
    with pytest.raises(ValueError):
        SupportPatchControllerConfig(max_compiled_patches_per_checkpoint=0)


def test_compile_execution_resources_flow_and_default() -> None:
    from pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory import (
        SupportPatchControllerConfig,
    )

    default = APGeneralizedExchangeConfig.from_route_config(
        SupportPatchControllerConfig(support_patch_scoring_workers=3)
    )
    assert default.route.compile_workers == 3
    explicit = APGeneralizedExchangeConfig.from_route_config(
        SupportPatchControllerConfig(
            support_patch_scoring_workers=2, compile_workers=8
        )
    )
    assert explicit.route.compile_workers == 8
    with pytest.raises(ValueError):
        SupportPatchControllerConfig(compile_workers=0)


def test_default_search_axes_are_omitted_from_provenance() -> None:
    route = SupportPatchControllerConfig()
    config = APGeneralizedExchangeConfig.from_route_config(route)
    assert config.search.search_policy == "family_frontiers"
    assert config.search.gain_evaluation == "joint_batch"
    for payload in (route.to_json_dict(), config.to_json_dict()["search"]):
        assert "search_policy" not in payload
        assert "gain_evaluation" not in payload


@pytest.mark.parametrize("policy", ("greedy", "block_exchange", "bounded_exhaustive"))
@pytest.mark.parametrize("gain", ("joint_batch", "singleton_factorized"))
def test_search_axes_flow_independently_to_compiled_q_transport(policy, gain) -> None:
    route = SupportPatchControllerConfig(
        search_policy=policy,
        gain_evaluation=gain,
        exchange_route_bundle="paper_ii_exchange_q_compiled_v1",
        max_insertion_batch_size=3,
    )
    config = APGeneralizedExchangeConfig.from_route_config(route)
    assert config.search.insertion_cardinality == 3
    for payload in (route.to_json_dict(), config.to_json_dict()["search"]):
        assert payload["search_policy"] == policy
        assert payload["gain_evaluation"] == gain


@pytest.mark.parametrize("overrides, message", (
    ({"search_policy": "exhaustive"}, "search_policy must"),
    ({"search_policy": None}, "search_policy must"),
    ({"gain_evaluation": "factorized"}, "gain_evaluation must"),
    ({"gain_evaluation": None}, "gain_evaluation must"),
    ({"search_policy": "greedy"}, "compiled-cost bundle"),
    ({"gain_evaluation": "singleton_factorized"}, "unsupported by family_frontiers"),
    ({"search_policy": "greedy", "gain_evaluation": "singleton_factorized",
      "exchange_route_bundle": "paper_ii_exchange_gamma_compiled_v1"}, "compiled realized-Q"),
    ({"search_policy": "greedy", "dynamics_policy": "avqds",
      "exchange_route_bundle": "paper_ii_exchange_q_compiled_v1"}, "dynamics_policy='exchange'"),
    ({"search_policy": "greedy", "exchange_insertion_frontier": 8,
      "exchange_route_bundle": "paper_ii_exchange_q_compiled_v1"}, "frontier settings"),
    ({"search_policy": "greedy", "interaction_frontier_widths": (8,),
      "exchange_route_bundle": "paper_ii_exchange_q_compiled_v1"}, "frontier settings"),
))
def test_invalid_search_axis_combinations_fail_at_route_construction(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        SupportPatchControllerConfig(**overrides)


def test_joint_search_allows_explicit_gamma_experiment() -> None:
    route = SupportPatchControllerConfig(
        search_policy="bounded_exhaustive",
        exchange_route_bundle="paper_ii_exchange_gamma_compiled_v1",
    )
    assert APGeneralizedExchangeConfig.from_route_config(route).route.bundle.needs_gamma


def test_typed_search_config_cannot_bypass_axis_validation() -> None:
    from dataclasses import replace

    config = APGeneralizedExchangeConfig.from_route_config(SupportPatchControllerConfig())
    with pytest.raises(ValueError, match="search_policy must"):
        replace(config.search, search_policy="made_up")
    with pytest.raises(ValueError, match="compiled-cost bundle"):
        replace(config, search=replace(config.search, search_policy="greedy"))
    with pytest.raises(ValueError, match="frontier settings"):
        replace(config.search, search_policy="greedy", exchange_insertion_frontier=2)
