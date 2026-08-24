"""Stable SR-SNAKE Phase-I/II energy-model policy identifiers.

Phase I uses the measured Fubini--Study metric only to set the trust radius;
Phase II uses measured directional curvature.  Keeping these identifiers in a
small dependency-free module lets CLI, route resolution, scoring, checkpoint,
and result code share one fail-closed vocabulary.
"""

from __future__ import annotations

from typing import Any


PHASE1_SCORE_MODE_TRUST_REGION_V1 = "trust_region_v1"
PHASE1_SCORE_MODE_LEGACY_SIMPLE_V1 = "legacy_simple_v1"
PHASE1_SCORE_MODE_CHOICES = (
    PHASE1_SCORE_MODE_TRUST_REGION_V1,
    PHASE1_SCORE_MODE_LEGACY_SIMPLE_V1,
)

PHASE1_ENERGY_MODEL_FIRST_ORDER_FS_TRUST_V1 = "first_order_fs_trust_v1"
PHASE1_ENERGY_MODEL_CHOICES = (
    PHASE1_ENERGY_MODEL_FIRST_ORDER_FS_TRUST_V1,
)

PHASE2_CURVATURE_POLICY_LEGACY_OPTIONAL_V1 = "legacy_optional_v1"
PHASE2_CURVATURE_POLICY_MEASURED_REQUIRED_FAIL_CLOSED_V1 = (
    "measured_required_fail_closed_v1"
)
PHASE2_CURVATURE_POLICY_CHOICES = (
    PHASE2_CURVATURE_POLICY_LEGACY_OPTIONAL_V1,
    PHASE2_CURVATURE_POLICY_MEASURED_REQUIRED_FAIL_CLOSED_V1,
)

PHASE2_CHEAP_CURVATURE_PROXY_POLICY_OFF = "off"
PHASE2_CHEAP_CURVATURE_PROXY_POLICY_CHOICES = (
    PHASE2_CHEAP_CURVATURE_PROXY_POLICY_OFF,
)

PHASE2_DIRECTIONAL_CURVATURE_RECEIPT_SCHEMA = (
    "sr_snake_phase2_directional_curvature_receipt_v1"
)
PHASE2_DIRECTIONAL_CURVATURE_PROVENANCE_SCHEMA = (
    "sr_snake_phase2_directional_curvature_provenance_v1"
)


def _normalize_policy(raw: Any, *, choices: tuple[str, ...], field: str) -> str:
    if raw in {None, ""}:
        raise ValueError(f"{field} is required.")
    key = str(raw).strip().lower().replace("-", "_")
    if key not in choices:
        raise ValueError(f"{field} must be one of {list(choices)}; got {raw!r}.")
    return key


def normalize_phase1_energy_model(raw: Any) -> str:
    return _normalize_policy(
        raw,
        choices=PHASE1_ENERGY_MODEL_CHOICES,
        field="phase1_energy_model",
    )


def normalize_phase1_score_mode_policy(raw: Any) -> str:
    return _normalize_policy(
        raw,
        choices=PHASE1_SCORE_MODE_CHOICES,
        field="phase1_score_mode",
    )


def normalize_phase2_curvature_policy(raw: Any) -> str:
    return _normalize_policy(
        raw,
        choices=PHASE2_CURVATURE_POLICY_CHOICES,
        field="phase2_curvature_policy",
    )


def normalize_phase2_cheap_curvature_proxy_policy(raw: Any) -> str:
    return _normalize_policy(
        raw,
        choices=PHASE2_CHEAP_CURVATURE_PROXY_POLICY_CHOICES,
        field="phase2_cheap_curvature_proxy_policy",
    )


__all__ = [
    "PHASE1_SCORE_MODE_CHOICES",
    "PHASE1_SCORE_MODE_LEGACY_SIMPLE_V1",
    "PHASE1_SCORE_MODE_TRUST_REGION_V1",
    "PHASE1_ENERGY_MODEL_CHOICES",
    "PHASE1_ENERGY_MODEL_FIRST_ORDER_FS_TRUST_V1",
    "PHASE2_CHEAP_CURVATURE_PROXY_POLICY_CHOICES",
    "PHASE2_CHEAP_CURVATURE_PROXY_POLICY_OFF",
    "PHASE2_CURVATURE_POLICY_CHOICES",
    "PHASE2_CURVATURE_POLICY_LEGACY_OPTIONAL_V1",
    "PHASE2_CURVATURE_POLICY_MEASURED_REQUIRED_FAIL_CLOSED_V1",
    "PHASE2_DIRECTIONAL_CURVATURE_PROVENANCE_SCHEMA",
    "PHASE2_DIRECTIONAL_CURVATURE_RECEIPT_SCHEMA",
    "normalize_phase1_energy_model",
    "normalize_phase1_score_mode_policy",
    "normalize_phase2_cheap_curvature_proxy_policy",
    "normalize_phase2_curvature_policy",
]
