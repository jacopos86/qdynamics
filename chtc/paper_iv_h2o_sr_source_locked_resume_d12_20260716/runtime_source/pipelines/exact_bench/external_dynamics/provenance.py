#!/usr/bin/env python3
"""Reference catalog for diagnostic external dynamics parity checks.

The catalog is metadata only.  It records optional public reference surfaces for
implementation-parity diagnostics and does not clone, import, or execute any
third-party code at module import time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

Availability = Literal["local_optional", "repo_native_only"]
ReferenceTier = Literal["author_code", "literature_limit", "repo_native_diagnostic"]
AdapterStatus = Literal[
    "source_probe_adapter",
    "api_probe_adapter",
    "repo_native_rhs_limit_adapter",
]


@dataclass(frozen=True)
class ExternalDynamicsReferenceSpec:
    """Metadata for an optional dynamics reference implementation."""

    reference_id: str
    display_name: str
    availability: Availability
    reference_tier: ReferenceTier
    url: str
    clone_url: str | None
    default_local_subdir: str | None
    intended_algorithm_ids: tuple[str, ...]
    adapter_status: AdapterStatus
    policy: str = "diagnostic_only_not_paper_evidence"
    license_note: str = "verify from local checkout before using any artifact outside diagnostics"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_REFERENCE_CATALOG: tuple[ExternalDynamicsReferenceSpec, ...] = (
    ExternalDynamicsReferenceSpec(
        reference_id="dalin27_adaptive_pvqd",
        display_name="Adaptive projected variational quantum dynamics reference code",
        availability="local_optional",
        reference_tier="author_code",
        url="https://github.com/dalin27/adaptive-pvqd",
        clone_url="https://github.com/dalin27/adaptive-pvqd.git",
        default_local_subdir="adaptive-pvqd",
        intended_algorithm_ids=("dyn_adaptive_pvqd",),
        adapter_status="source_probe_adapter",
        notes=(
            "Used only to validate adaptive-pVQD algorithm structure and, when a "
            "compatible legacy Qiskit environment is available, component-level "
            "projection behavior.  It is never table evidence."
        ),
    ),
    ExternalDynamicsReferenceSpec(
        reference_id="gqce_ames_avqds",
        display_name="GQCE/Ames Adaptive Variational Quantum Dynamics Simulations reference code",
        availability="local_optional",
        reference_tier="author_code",
        url="https://gitlab.com/gqce/avqds",
        clone_url="https://gitlab.com/gqce/avqds.git",
        default_local_subdir="avqds",
        intended_algorithm_ids=("dyn_avqds",),
        adapter_status="source_probe_adapter",
        notes=(
            "Base AVQDS reference surface for RHS-tangent/checkpoint behavior. "
            "This does not provide an AVQDS-T implementation."
        ),
    ),
    ExternalDynamicsReferenceSpec(
        reference_id="repo_native_avqds_t_rhs_limit",
        display_name="Repo-native AVQDS-T RHS/small-step limit diagnostic",
        availability="repo_native_only",
        reference_tier="repo_native_diagnostic",
        url="internal:repo-native-avqds-vs-avqds-t-rhs-limit",
        clone_url=None,
        default_local_subdir=None,
        intended_algorithm_ids=("dyn_avqds_t", "dyn_avqds"),
        adapter_status="repo_native_rhs_limit_adapter",
        license_note="not applicable; repo-native diagnostic helper",
        notes=(
            "Compares AVQDS-T product-formula target tangent against the base "
            "AVQDS RHS/projective tangent in a small-step diagnostic limit."
        ),
    ),
)


def external_dynamics_reference_catalog() -> tuple[ExternalDynamicsReferenceSpec, ...]:
    return _REFERENCE_CATALOG


def get_external_dynamics_reference_spec(reference_id: str) -> ExternalDynamicsReferenceSpec:
    key = str(reference_id).strip()
    for spec in _REFERENCE_CATALOG:
        if spec.reference_id == key:
            return spec
    known = ", ".join(spec.reference_id for spec in _REFERENCE_CATALOG)
    raise ValueError(f"Unknown external dynamics reference {reference_id!r}. Known references: {known}")


def reference_specs_for_algorithm(algorithm_id: str) -> tuple[ExternalDynamicsReferenceSpec, ...]:
    alg = str(algorithm_id).strip()
    return tuple(spec for spec in _REFERENCE_CATALOG if alg in spec.intended_algorithm_ids)
