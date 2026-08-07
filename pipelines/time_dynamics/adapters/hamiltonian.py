"""Realtime Hamiltonian-family adapter/capability helpers.

This module is intentionally data-oriented.  The static ADAPT registry owns the
family capability contract, while realtime consumers use this thin adapter view
to avoid scattering independent family dispatch tables through observables,
drive terms, and controller policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pipelines.contracts.problem import HamiltonianFamilyCapabilities
from pipelines.static_adapt.builders.problem_registry import get_problem_family_spec

SPINFUL_LATTICE_FAMILIES = frozenset(
    {"hubbard", "ionic_hubbard", "extended_hubbard", "ttprime_hubbard"}
)
SPINLESS_LATTICE_FAMILIES = frozenset({"spinless_tv"})
BOSON_CHAIN_FAMILIES = frozenset({"bose_hubbard", "harmonic_kerr_chain"})
SPIN_BOSON_FAMILIES = frozenset({"spin_boson"})
HUBBARD_HOLSTEIN_FAMILIES = frozenset({"hh"})
MOLECULAR_FLOW_FAMILIES = frozenset({"molecular_vibronic_h2"})
HAMILTONIAN_FLOW_FAMILIES = frozenset(
    tuple(SPINFUL_LATTICE_FAMILIES)
    + tuple(SPINLESS_LATTICE_FAMILIES)
    + tuple(BOSON_CHAIN_FAMILIES)
    + tuple(SPIN_BOSON_FAMILIES)
    + tuple(HUBBARD_HOLSTEIN_FAMILIES)
    + tuple(MOLECULAR_FLOW_FAMILIES)
)


@dataclass(frozen=True)
class RealtimeHamiltonianAdapter:
    """Resolved family view used by realtime algorithms.

    Methods stay deliberately small: algorithm modules should keep their local
    physics implementations, but ask this adapter for support/capability facts.
    """

    family_key: str
    capabilities: HamiltonianFamilyCapabilities

    @property
    def observable_kind(self) -> str:
        return str(self.capabilities.observable_kind)

    @property
    def drive_operator_kind(self) -> str | None:
        kind = self.capabilities.drive_operator_kind
        return None if kind in {None, ""} else str(kind)

    @property
    def supports_measurement_observables(self) -> bool:
        return bool(self.capabilities.supports_measurement_observables)

    @property
    def supports_driven_realtime(self) -> bool:
        return bool(self.capabilities.supports_driven_realtime)

    @property
    def supports_hamiltonian_flow_projective(self) -> bool:
        return bool(self.capabilities.supports_hamiltonian_flow_projective)


def _normalize_family_key(family_key: Any | None, *, default: str = "hh") -> str:
    if family_key in {None, ""}:
        return str(default)
    return str(family_key).strip().lower()


def family_capabilities_for_key(family_key: Any | None) -> HamiltonianFamilyCapabilities:
    """Return registered capabilities, falling back to an unsupported contract."""

    key = _normalize_family_key(family_key)
    try:
        return get_problem_family_spec(key).capabilities
    except ValueError:
        return HamiltonianFamilyCapabilities()


DRIVEN_HAMILTONIAN_FLOW_FAMILIES = frozenset(
    family
    for family in HAMILTONIAN_FLOW_FAMILIES
    if family_capabilities_for_key(family).supports_driven_realtime
)


def adapter_for_family_key(family_key: Any | None) -> RealtimeHamiltonianAdapter:
    key = _normalize_family_key(family_key)
    return RealtimeHamiltonianAdapter(
        family_key=key,
        capabilities=family_capabilities_for_key(key),
    )


def adapter_for_resolved_problem(
    resolved_problem: Any | None,
    *,
    default_family: str = "hh",
) -> RealtimeHamiltonianAdapter:
    """Resolve the realtime adapter for a registry context or lightweight test stub."""

    family_key = getattr(resolved_problem, "family_key", None)
    key = _normalize_family_key(family_key, default=str(default_family))
    capabilities = getattr(resolved_problem, "capabilities", None)
    if isinstance(capabilities, HamiltonianFamilyCapabilities):
        return RealtimeHamiltonianAdapter(family_key=key, capabilities=capabilities)
    return adapter_for_family_key(key)


def family_supports_hamiltonian_flow_projective(family_key: Any | None) -> bool:
    return bool(adapter_for_family_key(family_key).supports_hamiltonian_flow_projective)


def family_supports_driven_hamiltonian_flow(family_key: Any | None) -> bool:
    adapter = adapter_for_family_key(family_key)
    return bool(
        adapter.family_key in HAMILTONIAN_FLOW_FAMILIES
        and adapter.supports_driven_realtime
    )


__all__ = [
    "BOSON_CHAIN_FAMILIES",
    "DRIVEN_HAMILTONIAN_FLOW_FAMILIES",
    "HAMILTONIAN_FLOW_FAMILIES",
    "HUBBARD_HOLSTEIN_FAMILIES",
    "MOLECULAR_FLOW_FAMILIES",
    "RealtimeHamiltonianAdapter",
    "SPINFUL_LATTICE_FAMILIES",
    "SPINLESS_LATTICE_FAMILIES",
    "adapter_for_family_key",
    "adapter_for_resolved_problem",
    "family_capabilities_for_key",
    "family_supports_driven_hamiltonian_flow",
    "family_supports_hamiltonian_flow_projective",
]
