"""State-vector observables for active AP-McLachlan diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


HH_SITE_DOUBLON_OBSERVABLE_SCHEMA_V1 = "hh_site_doublon_observables_v1"


@dataclass(frozen=True)
class SiteDoublonObservableSnapshot:
    """HH site occupations and doublon evaluated from one state vector."""

    n_up_site: np.ndarray
    n_dn_site: np.ndarray
    site_occupations: np.ndarray
    doublon: float
    staggered: float
    primary_density: float

    def to_json_dict(self) -> dict[str, Any]:
        n_up = np.asarray(self.n_up_site, dtype=float).reshape(-1)
        n_dn = np.asarray(self.n_dn_site, dtype=float).reshape(-1)
        n_site = np.asarray(self.site_occupations, dtype=float).reshape(-1)
        return {
            "observable_schema": HH_SITE_DOUBLON_OBSERVABLE_SCHEMA_V1,
            "observable_telemetry_supported": True,
            "observable_family": "hh_spinful_site_doublon",
            "n_up_site": [float(x) for x in n_up.tolist()],
            "n_dn_site": [float(x) for x in n_dn.tolist()],
            "site_occupations": [float(x) for x in n_site.tolist()],
            "site_occupations_up": [float(x) for x in n_up.tolist()],
            "site_occupations_dn": [float(x) for x in n_dn.tolist()],
            "doublon": float(self.doublon),
            "staggered": float(self.staggered),
            "primary_density": float(self.primary_density),
            "site_occupations_label": "n_up+n_dn per site",
            "site_occupations_component_labels": [
                f"site_{index}" for index in range(int(n_site.size))
            ],
            "observable_evaluation_policy": "same_statevector_single_pass",
        }


@dataclass(frozen=True)
class SiteDoublonObservablePlan:
    """Precomputed diagonal masks for HH site/doublon observables."""

    num_sites: int
    ordering: str
    dimension: int
    up_masks: np.ndarray
    dn_masks: np.ndarray

    def evaluate(self, psi: np.ndarray) -> SiteDoublonObservableSnapshot:
        psi_vec = np.asarray(psi, dtype=complex).reshape(-1)
        if int(psi_vec.size) != int(self.dimension):
            raise ValueError(
                "Observable state dimension mismatch: "
                f"got {psi_vec.size}, expected {self.dimension}."
            )
        probs = np.abs(psi_vec) ** 2
        norm = float(np.sum(probs))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("Observable state must have positive finite norm.")
        probs = np.asarray(probs / norm, dtype=float).reshape(-1)
        up = np.asarray(self.up_masks @ probs, dtype=float).reshape(-1)
        dn = np.asarray(self.dn_masks @ probs, dtype=float).reshape(-1)
        site = np.asarray(up + dn, dtype=float).reshape(-1)
        doublon = float(np.sum((self.up_masks * self.dn_masks) @ probs))
        staggered = _staggered_density(site)
        primary_density = _primary_density(site)
        return SiteDoublonObservableSnapshot(
            n_up_site=up,
            n_dn_site=dn,
            site_occupations=site,
            doublon=float(doublon),
            staggered=float(staggered),
            primary_density=float(primary_density),
        )


def build_site_doublon_observable_plan(
    resolved_problem: Any,
    *,
    dimension: int,
) -> SiteDoublonObservablePlan | None:
    """Build an HH observable plan, or return ``None`` when unsupported."""

    family_key = str(getattr(resolved_problem, "family_key", "")).strip().lower()
    capabilities = getattr(resolved_problem, "capabilities", None)
    observable_kind = str(getattr(capabilities, "observable_kind", "")).strip().lower()
    if family_key != "hh" and observable_kind != "hh_spinful_boson":
        return None

    request = getattr(resolved_problem, "request", None)
    layout = getattr(resolved_problem, "layout", None)
    if request is None or layout is None:
        return None
    num_sites = int(getattr(request, "num_sites"))
    ordering = str(getattr(request, "ordering", getattr(layout, "ordering", "blocked")))
    if num_sites <= 0:
        raise ValueError("HH observable num_sites must be positive.")
    total_qubits = int(getattr(layout, "total_qubits"))
    expected_dimension = 1 << total_qubits
    if int(dimension) != expected_dimension:
        raise ValueError(
            "HH observable dimension does not match resolved register layout: "
            f"{dimension} vs 2**{total_qubits}={expected_dimension}."
        )
    fermion_start = _fermion_start_qubit(layout)
    indices = np.arange(int(dimension), dtype=np.uint64)
    up_masks = []
    dn_masks = []
    for site in range(num_sites):
        up_bit = int(fermion_start + _spin_orbital_bit_index(site, 0, num_sites, ordering))
        dn_bit = int(fermion_start + _spin_orbital_bit_index(site, 1, num_sites, ordering))
        if max(up_bit, dn_bit) >= total_qubits:
            raise ValueError(
                "HH observable spin-orbital bit exceeds register layout: "
                f"site={site}, up_bit={up_bit}, dn_bit={dn_bit}, total_qubits={total_qubits}."
            )
        up_masks.append(((indices >> np.uint64(up_bit)) & np.uint64(1)).astype(float))
        dn_masks.append(((indices >> np.uint64(dn_bit)) & np.uint64(1)).astype(float))
    return SiteDoublonObservablePlan(
        num_sites=int(num_sites),
        ordering=str(ordering),
        dimension=int(dimension),
        up_masks=np.asarray(up_masks, dtype=float),
        dn_masks=np.asarray(dn_masks, dtype=float),
    )


def observable_row_fields(
    psi: np.ndarray,
    *,
    plan: SiteDoublonObservablePlan | None,
) -> dict[str, Any]:
    """Return row-ready observable fields without extra propagation."""

    if plan is None:
        return {}
    return plan.evaluate(psi).to_json_dict()


def _fermion_start_qubit(layout: Any) -> int:
    block = None
    block_getter = getattr(layout, "block", None)
    if callable(block_getter):
        block = block_getter("fermion")
    if block is None:
        for candidate in tuple(getattr(layout, "blocks", ()) or ()):
            if str(getattr(candidate, "kind", "")) == "fermion" or str(getattr(candidate, "name", "")) == "fermion":
                block = candidate
                break
    if block is None:
        return 0
    return int(getattr(block, "start_qubit", 0))


def _spin_orbital_bit_index(site: int, spin: int, num_sites: int, ordering: str) -> int:
    ord_norm = str(ordering).strip().lower()
    if ord_norm == "blocked":
        return int(site) if int(spin) == 0 else int(num_sites) + int(site)
    if ord_norm == "interleaved":
        return (2 * int(site)) + int(spin)
    raise ValueError(f"Unsupported HH fermion ordering {ordering!r}.")


def _staggered_density(site_occupations: np.ndarray) -> float:
    occ = np.asarray(site_occupations, dtype=float).reshape(-1)
    if int(occ.size) == 0:
        return float("nan")
    signs = np.asarray(
        [1.0 if (site % 2 == 0) else -1.0 for site in range(int(occ.size))],
        dtype=float,
    )
    return float(np.sum(signs * occ) / float(occ.size))


def _primary_density(site_occupations: np.ndarray) -> float:
    occ = np.asarray(site_occupations, dtype=float).reshape(-1)
    if int(occ.size) == 0:
        return float("nan")
    if int(occ.size) == 1:
        return float(occ[0])
    if int(occ.size) == 2:
        return float(occ[0] - occ[1])
    return _staggered_density(occ)


__all__ = [
    "HH_SITE_DOUBLON_OBSERVABLE_SCHEMA_V1",
    "SiteDoublonObservablePlan",
    "SiteDoublonObservableSnapshot",
    "build_site_doublon_observable_plan",
    "observable_row_fields",
]
