"""The pre-permission ray threshold is decoupled from the post-refit certificate.

Undeclared, the threshold reuses ``prune_ray_distance_tol`` and every
provenance object stays byte-identical to the stored lineages.  Declared, it
changes only the measurement-free permission bound of an unrefit deletion.
"""

from __future__ import annotations

import pytest

from pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory import (
    SupportPatchControllerConfig,
)
from pipelines.time_dynamics.ap_mclachlan.exchange_config import (
    APGeneralizedExchangeConfig,
    permission_ray_distance_max,
)
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
    _build_parser,
)


def test_undeclared_threshold_reuses_the_post_refit_tolerance_and_is_silent():
    config = SupportPatchControllerConfig()
    exchange = APGeneralizedExchangeConfig.from_route_config(config)
    assert exchange.eligibility.permission_ray_distance_max is None
    assert permission_ray_distance_max(exchange) == pytest.approx(
        float(config.prune_ray_distance_tol)
    )
    payload = config.to_json_dict()
    assert "exchange_permission_ray_distance_max" not in payload
    assert (
        "permission_ray_distance_max"
        not in payload["generalized_exchange"]["eligibility"]
    )


def test_declared_threshold_changes_only_the_permission_bound():
    control = SupportPatchControllerConfig()
    config = SupportPatchControllerConfig(
        exchange_permission_ray_distance_max=5.0e-2
    )
    exchange = APGeneralizedExchangeConfig.from_route_config(config)
    assert permission_ray_distance_max(exchange) == pytest.approx(5.0e-2)
    assert exchange.certification.ray_distance_max == pytest.approx(
        float(control.prune_ray_distance_tol)
    )
    assert exchange.certification.velocity_change_max == pytest.approx(
        float(control.prune_patch_smoothness_eta_max)
    )
    payload = config.to_json_dict()
    assert payload["exchange_permission_ray_distance_max"] == pytest.approx(
        5.0e-2
    )
    assert payload["generalized_exchange"]["eligibility"][
        "permission_ray_distance_max"
    ] == pytest.approx(5.0e-2)
    stripped = dict(payload)
    stripped.pop("exchange_permission_ray_distance_max")
    stripped["generalized_exchange"]["eligibility"].pop(
        "permission_ray_distance_max"
    )
    assert stripped == control.to_json_dict()


@pytest.mark.parametrize("value", [0.0, -1.0e-3])
def test_declared_threshold_must_be_positive(value: float):
    with pytest.raises(ValueError, match="must be positive"):
        SupportPatchControllerConfig(exchange_permission_ray_distance_max=value)


def test_runner_flag_defaults_to_none_and_parses_a_float():
    base = ["--artifact-json", "seed.json", "--output-json", "out.json"]
    parser = _build_parser()
    assert parser.parse_args(base).exchange_permission_ray_distance_max is None
    parsed = parser.parse_args(
        base + ["--exchange-permission-ray-distance-max", "0.05"]
    )
    assert parsed.exchange_permission_ray_distance_max == pytest.approx(0.05)
