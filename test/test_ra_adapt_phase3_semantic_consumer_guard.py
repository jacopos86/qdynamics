from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.scaffold.hh_continuation_scoring import (
    PHASE3_ZERO_CENTERED_SIGNED_FACTOR_CONSUMER_SEMANTIC_VERSION,
    require_phase3_signed_factor_consumer_semantic_version,
)
from pipelines.static_adapt import adapt_pipeline
from pipelines.static_adapt.ra_adapt.contracts import (
    resolved_ra_adapt_protocol_from_mapping,
)
from pipelines.static_adapt.ra_adapt.semantic_closure_routes import (
    PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2,
    PAPER_I_RA_SEMANTIC_IMPLEMENTATION_VERSION_V2,
    build_paper_i_ra_strong_weak_always_k5_request,
    build_paper_i_ra_strong_weak_nph3_problem,
    preflight_paper_i_ra_strong_weak_always_k5,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_AFFECTED_PROTOCOL = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_page12_matched_singleton12_r50_20260815_v1_local/"
    "bundle_materialization/paper_i_page12_matched_singleton12_r50_v1/"
    "protocols/global_singleton_gradient_phase0_phase23_qiskit_no_lanes__"
    "intermediate_strong__nph7__ra_global_singleton_gradient_phase0_"
    "phase123_qiskit_phase23_plateau.json"
)


def _score_config_from_route(route_contract: dict[str, object]):
    runtime = adapt_pipeline._thaw_canonical_sr_snake_infrastructure(
        adapt_pipeline._CANONICAL_SR_SNAKE_RUNTIME_INFRASTRUCTURE
    )
    execution = route_contract.get("execution_settings")
    assert isinstance(execution, dict)
    runtime.update(execution)
    return adapt_pipeline._default_no_prune_full_score_config(
        runtime,
        phase2_shortlist_size=24,
        phase2_shortlist_fraction=0.2,
    )


def test_authority_loaded_affected_route_refuses_corrected_phase3_consumer() -> None:
    protocol = resolved_ra_adapt_protocol_from_mapping(
        json.loads(HISTORICAL_AFFECTED_PROTOCOL.read_text(encoding="utf-8"))
    )
    route = protocol.route_contract
    assert isinstance(route, dict)
    assert route["sha256"] == (
        "9811652b332b592bee048a8e5f3048972256abae186921ed7efea52bfd5f3dd8"
    )
    assert route["execution_settings"].get(  # type: ignore[union-attr]
        "ra_semantic_implementation_version"
    ) is None

    with pytest.raises(RuntimeError, match="historical affected route digests"):
        _score_config_from_route(route)


def test_new_semantic_route_authorizes_exact_corrected_phase3_consumer() -> None:
    protocol = preflight_paper_i_ra_strong_weak_always_k5(
        build_paper_i_ra_strong_weak_nph3_problem(),
        build_paper_i_ra_strong_weak_always_k5_request(
            PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2
        ),
    )
    route = protocol.route_contract
    assert isinstance(route, dict)
    assert route["execution_settings"][  # type: ignore[index]
        "ra_semantic_implementation_version"
    ] == PAPER_I_RA_SEMANTIC_IMPLEMENTATION_VERSION_V2

    score_cfg = _score_config_from_route(route)

    assert score_cfg.phase3_signed_factor_consumer_semantic_version == (
        PHASE3_ZERO_CENTERED_SIGNED_FACTOR_CONSUMER_SEMANTIC_VERSION
    )
    assert require_phase3_signed_factor_consumer_semantic_version(score_cfg) == (
        PHASE3_ZERO_CENTERED_SIGNED_FACTOR_CONSUMER_SEMANTIC_VERSION
    )
