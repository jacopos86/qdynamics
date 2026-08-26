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


def test_route_transport_is_partitioned_into_four_exchange_inputs() -> None:
    route = SupportPatchControllerConfig(
        debt_policy="drift_ranked",
        prune_history_lambda=0.0,
        max_structural_pool_size=128,
        max_insertion_batch_size=1,
    )
    config = APGeneralizedExchangeConfig.from_route_config(route)

    assert isinstance(config.score, ExchangeScoreConfig)
    assert isinstance(config.eligibility, ExchangeEligibilityConfig)
    assert isinstance(config.certification, ExchangeCertificationConfig)
    assert isinstance(config.search, ExchangeSearchBudget)
    assert config.score.debt_policy == "drift_ranked"
    assert config.score.history_weight == 0.0
    assert config.search.pool_size == 128
    assert config.search.insertion_cardinality == 1
    recorded = route.to_json_dict()["generalized_exchange"]
    assert set(recorded) == {"score", "eligibility", "certification", "search"}


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
