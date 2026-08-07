from __future__ import annotations

import json

import pytest

from pipelines.static_adapt.historical_formal_manifold_provenance import (
    FORMAL_MANIFOLD_ROUTE,
    FORMAL_MANIFOLD_ROUTE_COMPOSITION_SCHEMA,
    FORMAL_MANIFOLD_ROUTE_FAMILY,
    FORMAL_MANIFOLD_ROUTE_PROFILE_SCHEMA,
    FORMAL_MANIFOLD_SR_SELECTOR_FAMILY,
    FormalManifoldRouteComposition,
)
from pipelines.static_adapt.output_artifacts import (
    _resolved_output_formal_manifold_route_composition,
)
from pipelines.static_adapt.resume_scaffold import (
    extract_formal_manifold_route_composition,
)


def _historical_profile_payload() -> dict[str, object]:
    return {
        "schema": FORMAL_MANIFOLD_ROUTE_PROFILE_SCHEMA,
        "route_family": FORMAL_MANIFOLD_ROUTE_FAMILY,
        "route_profile": "historical_fm_profile_v1",
        "adapt_reoptimization_route": FORMAL_MANIFOLD_ROUTE,
        "candidate_selector_family": FORMAL_MANIFOLD_SR_SELECTOR_FAMILY,
        "candidate_selector_profile": "historical_selector_profile_v1",
        "phase2_enable_batching": False,
        "phase3_enable_batching": False,
        "structural_rollback_enabled": False,
    }


def _composition_payload() -> dict[str, object]:
    profile = _historical_profile_payload()
    return FormalManifoldRouteComposition(
        route_profile=str(profile["route_profile"]),
        candidate_selector_family=str(profile["candidate_selector_family"]),
        candidate_selector_profile=str(profile["candidate_selector_profile"]),
        singleton_response_selector=profile,
    ).as_dict()


def test_historical_composition_round_trip_verifies_canonical_digest() -> None:
    payload = _composition_payload()

    assert payload["schema"] == FORMAL_MANIFOLD_ROUTE_COMPOSITION_SCHEMA
    assert len(str(payload["sha256"])) == 64
    assert FormalManifoldRouteComposition.from_mapping(payload).as_dict() == payload
    json.dumps(payload, allow_nan=False, sort_keys=True)


def test_historical_profile_alias_parses_without_profile_resolution() -> None:
    profile = _historical_profile_payload()
    expected = _composition_payload()

    parsed = FormalManifoldRouteComposition.from_mapping(profile).as_dict()

    assert parsed == expected
    assert parsed["singleton_response_selector"] == profile


def test_output_and_resume_passive_aliases_agree() -> None:
    profile = _historical_profile_payload()
    composition = _composition_payload()
    adapt_payload = {
        "formal_manifold_route_composition": composition,
        "static_route_identity": profile,
    }

    assert (
        _resolved_output_formal_manifold_route_composition(adapt_payload)
        == composition
    )
    assert (
        extract_formal_manifold_route_composition(
            {
                "adapt_vqe": adapt_payload,
                "settings": {},
            }
        )
        == composition
    )


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("schema", "unknown_schema_v1", "schema"),
        ("route_family", "singleton_response_snake", "route_family"),
        ("adapt_reoptimization_route", "off", "adapt_reoptimization_route"),
        ("candidate_selector_family", "other_selector", "selector family"),
    ],
)
def test_historical_composition_rejects_identity_drift(
    field: str,
    replacement: object,
    error: str,
) -> None:
    payload = _composition_payload()
    payload.pop("sha256")
    payload[field] = replacement

    with pytest.raises(ValueError, match=error):
        FormalManifoldRouteComposition.from_mapping(payload)


def test_historical_composition_rejects_tampered_digest() -> None:
    payload = _composition_payload()
    payload["singleton_response_selector"]["phase2_enable_batching"] = True

    with pytest.raises(ValueError, match="digest mismatch"):
        FormalManifoldRouteComposition.from_mapping(payload)


def test_historical_composition_rejects_partial_selector_and_nonfinite_payload() -> None:
    with pytest.raises(ValueError, match="declared together"):
        FormalManifoldRouteComposition(
            candidate_selector_family=FORMAL_MANIFOLD_SR_SELECTOR_FAMILY,
        )
    with pytest.raises(ValueError, match="finite JSON"):
        FormalManifoldRouteComposition(
            singleton_response_selector={"invalid": float("nan")},
        )


def test_historical_provenance_public_surface_is_parse_only() -> None:
    from pipelines.static_adapt import historical_formal_manifold_provenance

    assert set(historical_formal_manifold_provenance.__all__) == {
        "FORMAL_MANIFOLD_ROUTE",
        "FORMAL_MANIFOLD_ROUTE_COMPOSITION_SCHEMA",
        "FORMAL_MANIFOLD_ROUTE_FAMILY",
        "FORMAL_MANIFOLD_ROUTE_PROFILE_OFF",
        "FORMAL_MANIFOLD_ROUTE_PROFILE_SCHEMA",
        "FORMAL_MANIFOLD_SR_SELECTOR_FAMILY",
        "FormalManifoldRouteComposition",
    }
    assert not hasattr(
        historical_formal_manifold_provenance,
        "resolve_formal_manifold_route_profile",
    )
    assert not hasattr(
        historical_formal_manifold_provenance,
        "FormalManifoldConfig",
    )
