"""Quarantined parser for preserved Formal-Manifold route provenance.

This module is intentionally inert.  It recognizes and authenticates the
stable serialized identity of historical Formal-Manifold artifacts; it does
not resolve profiles, construct settings, authorize resume, or expose runtime
configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping


FORMAL_MANIFOLD_ROUTE_PROFILE_OFF = "off"
FORMAL_MANIFOLD_ROUTE = "formal_manifold_warm_start_v1"
FORMAL_MANIFOLD_ROUTE_FAMILY = "formal_manifold_snake"
FORMAL_MANIFOLD_SR_SELECTOR_FAMILY = "singleton_response_snake"
FORMAL_MANIFOLD_ROUTE_COMPOSITION_SCHEMA = (
    "formal_manifold_route_composition_v1"
)
FORMAL_MANIFOLD_ROUTE_PROFILE_SCHEMA = "formal_manifold_route_profile_v1"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _json_safe_mapping(
    value: Mapping[str, Any],
    *,
    field_name: str,
) -> dict[str, Any]:
    try:
        normalized = json.loads(_canonical_json_bytes(dict(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be finite JSON data.") from exc
    if not isinstance(normalized, dict):
        raise TypeError(f"{field_name} must be a mapping.")
    return normalized


@dataclass(frozen=True)
class FormalManifoldRouteComposition:
    """Validated, JSON-safe identity for one historical FM composition."""

    route_family: str = FORMAL_MANIFOLD_ROUTE_FAMILY
    route_profile: str = FORMAL_MANIFOLD_ROUTE
    candidate_selector_family: str | None = None
    candidate_selector_profile: str | None = None
    adapt_reoptimization_route: str = FORMAL_MANIFOLD_ROUTE
    singleton_response_selector: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        route_family = str(self.route_family)
        route_profile = str(self.route_profile)
        reoptimization_route = str(self.adapt_reoptimization_route)
        if route_family != FORMAL_MANIFOLD_ROUTE_FAMILY:
            raise ValueError(
                "historical formal-manifold composition route_family must be "
                f"{FORMAL_MANIFOLD_ROUTE_FAMILY!r}."
            )
        if not route_profile.strip():
            raise ValueError(
                "historical formal-manifold composition route_profile is required."
            )
        if reoptimization_route != FORMAL_MANIFOLD_ROUTE:
            raise ValueError(
                "historical formal-manifold composition requires "
                f"adapt_reoptimization_route={FORMAL_MANIFOLD_ROUTE!r}."
            )

        selector_family = (
            None
            if self.candidate_selector_family in {None, ""}
            else str(self.candidate_selector_family)
        )
        selector_profile = (
            None
            if self.candidate_selector_profile in {None, ""}
            else str(self.candidate_selector_profile)
        )
        if (selector_family is None) != (selector_profile is None):
            raise ValueError(
                "candidate selector family and profile must be declared together."
            )
        if (
            selector_family is not None
            and selector_family != FORMAL_MANIFOLD_SR_SELECTOR_FAMILY
        ):
            raise ValueError(
                "historical formal-manifold composition candidate selector "
                f"family must be {FORMAL_MANIFOLD_SR_SELECTOR_FAMILY!r}."
            )
        if not isinstance(self.singleton_response_selector, Mapping):
            raise TypeError("singleton_response_selector must be a mapping.")

        object.__setattr__(self, "route_family", route_family)
        object.__setattr__(self, "route_profile", route_profile)
        object.__setattr__(
            self,
            "candidate_selector_family",
            selector_family,
        )
        object.__setattr__(
            self,
            "candidate_selector_profile",
            selector_profile,
        )
        object.__setattr__(
            self,
            "adapt_reoptimization_route",
            reoptimization_route,
        )
        object.__setattr__(
            self,
            "singleton_response_selector",
            _json_safe_mapping(
                self.singleton_response_selector,
                field_name="singleton_response_selector",
            ),
        )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any] | "FormalManifoldRouteComposition",
    ) -> "FormalManifoldRouteComposition":
        if isinstance(payload, cls):
            return payload
        if not isinstance(payload, Mapping):
            raise TypeError(
                "historical formal-manifold route composition must be a mapping."
            )

        raw = dict(payload)
        nested = raw.get("formal_manifold_route_composition")
        if nested is not None:
            if not isinstance(nested, Mapping):
                raise TypeError(
                    "formal_manifold_route_composition must be a mapping."
                )
            raw = dict(nested)

        source_schema = raw.get("schema")
        if source_schema not in {
            None,
            "",
            FORMAL_MANIFOLD_ROUTE_COMPOSITION_SCHEMA,
            FORMAL_MANIFOLD_ROUTE_PROFILE_SCHEMA,
        }:
            raise ValueError(
                "unsupported historical formal-manifold provenance schema: "
                f"{source_schema!r}."
            )

        selector_block = raw.get("singleton_response_selector")
        if selector_block is not None and not isinstance(selector_block, Mapping):
            raise TypeError("singleton_response_selector must be a mapping.")
        if selector_block is None:
            mechanisms = raw.get("mechanisms")
            if mechanisms is not None and not isinstance(mechanisms, Mapping):
                raise TypeError("mechanisms must be a mapping.")
            if isinstance(mechanisms, Mapping):
                selector_block = mechanisms.get("singleton_response_selector")
                if selector_block is not None and not isinstance(
                    selector_block,
                    Mapping,
                ):
                    raise TypeError(
                        "mechanisms.singleton_response_selector must be a mapping."
                    )
        if (
            selector_block is None
            and source_schema == FORMAL_MANIFOLD_ROUTE_PROFILE_SCHEMA
        ):
            # Historical ``static_route_identity`` aliases serialized the
            # already-resolved profile itself.  Preserve that payload as the
            # selector receipt without consulting the retired profile registry.
            selector_block = raw
        if selector_block is None:
            selector_block = {}

        selector_family = raw.get(
            "candidate_selector_family",
            raw.get("sr_route_family"),
        )
        selector_profile = raw.get(
            "candidate_selector_profile",
            raw.get("sr_route_profile"),
        )
        route_profile = raw.get("route_profile")
        if route_profile in {None, ""}:
            route_profile = (
                selector_profile
                if selector_profile not in {None, ""}
                else FORMAL_MANIFOLD_ROUTE
            )

        composition = cls(
            route_family=str(
                raw.get("route_family") or FORMAL_MANIFOLD_ROUTE_FAMILY
            ),
            route_profile=str(route_profile),
            candidate_selector_family=(
                None if selector_family in {None, ""} else str(selector_family)
            ),
            candidate_selector_profile=(
                None if selector_profile in {None, ""} else str(selector_profile)
            ),
            adapt_reoptimization_route=str(
                raw.get("adapt_reoptimization_route") or FORMAL_MANIFOLD_ROUTE
            ),
            singleton_response_selector=selector_block,
        )

        supplied_digest = raw.get("sha256")
        if (
            source_schema == FORMAL_MANIFOLD_ROUTE_COMPOSITION_SCHEMA
            and supplied_digest not in {None, ""}
            and str(supplied_digest) != composition.sha256
        ):
            raise ValueError(
                "historical formal-manifold route composition digest mismatch."
            )
        return composition

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": FORMAL_MANIFOLD_ROUTE_COMPOSITION_SCHEMA,
            "route_family": self.route_family,
            "route_profile": self.route_profile,
            "candidate_selector_family": self.candidate_selector_family,
            "candidate_selector_profile": self.candidate_selector_profile,
            "adapt_reoptimization_route": self.adapt_reoptimization_route,
            "singleton_response_selector": _json_safe_mapping(
                self.singleton_response_selector,
                field_name="singleton_response_selector",
            ),
        }
        payload["sha256"] = _json_sha256(payload)
        return payload

    @property
    def sha256(self) -> str:
        return str(self.as_dict()["sha256"])


__all__ = [
    "FORMAL_MANIFOLD_ROUTE",
    "FORMAL_MANIFOLD_ROUTE_COMPOSITION_SCHEMA",
    "FORMAL_MANIFOLD_ROUTE_FAMILY",
    "FORMAL_MANIFOLD_ROUTE_PROFILE_OFF",
    "FORMAL_MANIFOLD_ROUTE_PROFILE_SCHEMA",
    "FORMAL_MANIFOLD_SR_SELECTOR_FAMILY",
    "FormalManifoldRouteComposition",
]
