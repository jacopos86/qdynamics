#!/usr/bin/env python3
"""Shared symmetry metadata and verify-only mitigation hooks for HH continuation."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

from pipelines.hardcoded.hh_continuation_types import SymmetrySpec


_LOW_RISK_FAMILIES = {
    "paop",
    "paop_min",
    "paop_std",
    "paop_full",
    "paop_lf",
    "paop_lf_std",
    "paop_lf2_std",
    "paop_lf_full",
    "uccsd_paop_lf_full",
    "uccsd",
    "hva",
    "core",
}

_ALLOWED_SYMMETRY_MODES = {"off", "verify_only", "postselect_diag_v1", "projector_renorm_v1"}


def normalize_phase3_symmetry_mitigation_mode(mode: str | None) -> str:
    mode_key = "off" if mode is None else str(mode).strip().lower()
    if mode_key == "":
        return "off"
    if mode_key not in _ALLOWED_SYMMETRY_MODES:
        raise ValueError(
            "phase3_symmetry_mitigation_mode must be one of "
            f"{sorted(_ALLOWED_SYMMETRY_MODES)}."
        )
    return str(mode_key)


def build_symmetry_spec(
    *,
    family_id: str,
    mitigation_mode: str = "off",
) -> SymmetrySpec:
    family_key = str(family_id).strip().lower()
    mitigation_mode_key = normalize_phase3_symmetry_mitigation_mode(mitigation_mode)
    if family_key in _LOW_RISK_FAMILIES:
        leakage_risk = 0.0
    elif family_key in {"residual", "full_meta", "full_hamiltonian"}:
        leakage_risk = 0.1
    else:
        leakage_risk = 0.2
    tags = ["fermion_number", "spin_sector"]
    if mitigation_mode_key in {"postselect_diag_v1", "projector_renorm_v1"}:
        tags.append("active_symmetry_requested")
    return SymmetrySpec(
        particle_number_mode="preserving",
        spin_sector_mode="preserving",
        phonon_number_mode="not_conserved",
        leakage_risk=float(leakage_risk),
        mitigation_eligible=bool(mitigation_mode_key != "off"),
        grouping_eligible=bool(leakage_risk <= 0.2),
        hard_guard=False,
        tags=list(tags),
    )


def symmetry_spec_to_dict(spec: SymmetrySpec | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if isinstance(spec, SymmetrySpec):
        return asdict(spec)
    if isinstance(spec, Mapping):
        return dict(spec)
    return None


def leakage_penalty_from_spec(spec: SymmetrySpec | Mapping[str, Any] | None) -> float:
    if isinstance(spec, SymmetrySpec):
        return float(spec.leakage_risk)
    if isinstance(spec, Mapping):
        return float(spec.get("leakage_risk", 0.0))
    return 0.0


def verify_symmetry_sequence(
    *,
    generator_metadata: Sequence[Mapping[str, Any]],
    mitigation_mode: str,
) -> dict[str, Any]:
    mode = normalize_phase3_symmetry_mitigation_mode(mitigation_mode)
    if mode == "off":
        return {
            "mode": "off",
            "executed": False,
            "active": False,
            "passed": True,
            "high_risk_count": 0,
            "max_leakage_risk": 0.0,
        }
    leakage_values = []
    for meta in generator_metadata:
        spec = meta.get("symmetry_spec", None) if isinstance(meta, Mapping) else None
        leakage_values.append(float(leakage_penalty_from_spec(spec)))
    max_risk = float(max(leakage_values) if leakage_values else 0.0)
    high_risk_count = int(sum(1 for val in leakage_values if float(val) > 0.5))
    return {
        "mode": str(mode),
        "executed": True,
        "active": bool(mode in {"postselect_diag_v1", "projector_renorm_v1"}),
        "passed": bool(high_risk_count == 0),
        "high_risk_count": int(high_risk_count),
        "max_leakage_risk": float(max_risk),
    }
