"""Controlled zero-angle ANZATS redundancy for Paper-II diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np

from pipelines.time_dynamics.ap_mclachlan.state import (
    AP_PARAMETERIZATION_PER_PAULI_TERM,
    APMcLachlanState,
)
from pipelines.time_dynamics.ap_mclachlan.support_atoms import (
    APPEND_OCCURRENCE_POLICY_LAYER_REUSE,
    candidate_append_atoms,
    state_with_appended_atoms,
)
from pipelines.time_dynamics.normalized_pauli_pool import (
    NORMALIZED_POOL_HAMILTONIAN_DRIVE,
    NormalizedPauliPoolContract,
    normalized_pool_candidate_terms,
)


REDUNDANCY_STRESS_SCHEMA_V1 = "paper_ii_zero_angle_redundancy_stress_v1"
DEFAULT_REDUNDANCY_STRESS_PARITY_ATOL = 1.0e-10
REDUNDANCY_STRESS_LAYOUT_LAYERED = "layered"
REDUNDANCY_STRESS_LAYOUT_ADJACENT_DUPLICATES = "adjacent_duplicates"
REDUNDANCY_STRESS_LAYOUTS = (
    REDUNDANCY_STRESS_LAYOUT_LAYERED,
    REDUNDANCY_STRESS_LAYOUT_ADJACENT_DUPLICATES,
)


@dataclass(frozen=True)
class RedundancyStressConfig:
    """Diagnostic-only repeated-layer stress configuration."""

    layer_count: int = 0
    pool_profile: str = NORMALIZED_POOL_HAMILTONIAN_DRIVE
    layout_mode: str = REDUNDANCY_STRESS_LAYOUT_LAYERED
    state_parity_atol: float = DEFAULT_REDUNDANCY_STRESS_PARITY_ATOL

    def __post_init__(self) -> None:
        if int(self.layer_count) < 0:
            raise ValueError("redundancy stress layer_count must be nonnegative")
        if not str(self.pool_profile).strip():
            raise ValueError("redundancy stress pool_profile must be nonempty")
        if str(self.layout_mode) not in set(REDUNDANCY_STRESS_LAYOUTS):
            raise ValueError(
                "redundancy stress layout_mode must be one of "
                f"{REDUNDANCY_STRESS_LAYOUTS!r}; got {self.layout_mode!r}"
            )
        if not np.isfinite(float(self.state_parity_atol)) or float(
            self.state_parity_atol
        ) < 0.0:
            raise ValueError(
                "redundancy stress state_parity_atol must be finite and nonnegative"
            )

    @property
    def enabled(self) -> bool:
        return int(self.layer_count) > 0

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": REDUNDANCY_STRESS_SCHEMA_V1,
            "enabled": bool(self.enabled),
            "layer_count": int(self.layer_count),
            "pool_profile": str(self.pool_profile),
            "layout_mode": str(self.layout_mode),
            "state_parity_atol": float(self.state_parity_atol),
        }


@dataclass(frozen=True)
class RedundancyStressResult:
    """Augmented AP state and its non-destructive stress receipt."""

    state: APMcLachlanState
    receipt: Mapping[str, Any]


def redundancy_stress_config_from_metadata(
    metadata: Mapping[str, Any] | None,
) -> RedundancyStressConfig:
    """Resolve the shared diagnostic fixture from benchmark metadata."""

    payload = dict(metadata or {})
    return RedundancyStressConfig(
        layer_count=int(payload.get("diagnostic_redundancy_layer_count", 0)),
        pool_profile=str(
            payload.get(
                "diagnostic_redundancy_pool_profile",
                NORMALIZED_POOL_HAMILTONIAN_DRIVE,
            )
        ),
        layout_mode=str(
            payload.get(
                "diagnostic_redundancy_layout_mode",
                REDUNDANCY_STRESS_LAYOUT_LAYERED,
            )
        ),
        state_parity_atol=float(
            payload.get(
                "diagnostic_redundancy_state_parity_atol",
                DEFAULT_REDUNDANCY_STRESS_PARITY_ATOL,
            )
        ),
    )


def inject_zero_angle_redundancy_layers(
    state: APMcLachlanState,
    *,
    pool_contract: NormalizedPauliPoolContract | None,
    config: RedundancyStressConfig,
) -> RedundancyStressResult:
    """Append repeated zero-angle copies of one deterministic normalized pool."""

    if not config.enabled:
        return RedundancyStressResult(
            state=state,
            receipt={
                **config.to_json_dict(),
                "applied": False,
                "reason": "disabled",
                "appended_coordinate_count": 0,
            },
        )
    if state.parameterization_mode != AP_PARAMETERIZATION_PER_PAULI_TERM:
        raise ValueError(
            "controlled redundancy stress requires parameterization_mode="
            "'per_pauli_term'"
        )
    if pool_contract is None:
        raise ValueError(
            "controlled redundancy stress requires an explicit normalized pool contract"
        )
    if str(pool_contract.profile) != str(config.pool_profile):
        raise ValueError(
            "redundancy stress pool profile mismatch: "
            f"{pool_contract.profile!r} vs {config.pool_profile!r}"
        )

    prepared_before = state.prepare_state(state.theta_runtime)
    original_pool_terms = tuple(state.candidate_pool_terms)
    working = replace(
        state,
        candidate_pool_terms=normalized_pool_candidate_terms(pool_contract),
    )
    theta = np.asarray(working.theta_runtime, dtype=float).reshape(-1)
    if str(config.layout_mode) == REDUNDANCY_STRESS_LAYOUT_ADJACENT_DUPLICATES:
        working, theta, layer_receipts = _append_adjacent_duplicate_groups(
            working,
            theta_runtime=theta,
            pool_atom_count=len(pool_contract.atoms),
            occurrence_count=int(config.layer_count),
        )
    else:
        working, theta, layer_receipts = _append_complete_layers(
            working,
            theta_runtime=theta,
            pool_atom_count=len(pool_contract.atoms),
            layer_count=int(config.layer_count),
        )

    augmented = replace(
        working,
        candidate_pool_terms=original_pool_terms,
        candidate_pool_source=state.candidate_pool_source,
    )
    prepared_after = augmented.prepare_state(theta)
    before_norm = float(np.linalg.norm(prepared_before))
    after_norm = float(np.linalg.norm(prepared_after))
    if before_norm <= 0.0 or after_norm <= 0.0:
        raise ValueError("zero-angle redundancy stress produced a zero-norm state")
    prepared_before_unit = prepared_before / before_norm
    prepared_after_unit = prepared_after / after_norm
    overlap = complex(np.vdot(prepared_before_unit, prepared_after_unit))
    overlap_abs_raw = float(abs(overlap))
    phase = (
        1.0 + 0.0j
        if overlap_abs_raw <= 0.0
        else overlap / overlap_abs_raw
    )
    phase_aligned_l2 = float(
        np.linalg.norm(
            prepared_before_unit - np.conjugate(phase) * prepared_after_unit
        )
    )
    # The orthogonal projection is stable near unit overlap, unlike
    # sqrt(1 - |<psi|phi>|^2), which can amplify roundoff at parity.
    ray_distance = float(
        np.linalg.norm(prepared_after_unit - overlap * prepared_before_unit)
    )
    parity_passed = bool(
        phase_aligned_l2 <= float(config.state_parity_atol)
        and ray_distance <= float(config.state_parity_atol)
    )
    if not parity_passed:
        raise ValueError(
            "zero-angle redundancy stress changed the prepared state: "
            f"phase_aligned_l2={phase_aligned_l2}, ray_distance={ray_distance}, "
            f"atol={config.state_parity_atol}"
        )

    receipt = {
        **config.to_json_dict(),
        "applied": True,
        "reason": (
            "adjacent_zero_angle_duplicate_coordinates"
            if str(config.layout_mode)
            == REDUNDANCY_STRESS_LAYOUT_ADJACENT_DUPLICATES
            else "repeated_zero_angle_normalized_pauli_layers"
        ),
        "pool_atom_count": int(len(pool_contract.atoms)),
        "pool_ordered_atom_contract_sha256": str(
            pool_contract.ordered_atom_contract_sha256
        ),
        "appended_coordinate_count": int(
            int(config.layer_count) * len(pool_contract.atoms)
        ),
        "runtime_parameter_count_before": int(state.runtime_parameter_count),
        "runtime_parameter_count_after": int(augmented.runtime_parameter_count),
        "all_appended_angles_zero": bool(
            np.allclose(
                theta[int(state.runtime_parameter_count) :],
                0.0,
                rtol=0.0,
                atol=0.0,
            )
        ),
        "prepared_state_phase_aligned_l2": float(phase_aligned_l2),
        "prepared_state_ray_distance": float(ray_distance),
        "prepared_state_parity_passed": bool(parity_passed),
        "layers": layer_receipts,
    }
    extensions = dict(augmented.extensions or {})
    extensions[REDUNDANCY_STRESS_SCHEMA_V1] = receipt
    return RedundancyStressResult(
        state=replace(augmented, extensions=extensions),
        receipt=receipt,
    )


def _append_complete_layers(
    state: APMcLachlanState,
    *,
    theta_runtime: np.ndarray,
    pool_atom_count: int,
    layer_count: int,
) -> tuple[APMcLachlanState, np.ndarray, list[dict[str, Any]]]:
    working = state
    theta = np.asarray(theta_runtime, dtype=float).reshape(-1)
    receipts: list[dict[str, Any]] = []
    expected_base_atom_ids: tuple[str, ...] | None = None
    for layer_index in range(int(layer_count)):
        frontier = _redundancy_frontier(working, pool_atom_count=pool_atom_count)
        base_atom_ids = tuple(_base_atom_id(atom) for atom in frontier)
        if expected_base_atom_ids is None:
            expected_base_atom_ids = base_atom_ids
        elif base_atom_ids != expected_base_atom_ids:
            raise ValueError(
                "redundancy stress layers do not share the same ordered base atoms"
            )
        working, theta = state_with_appended_atoms(
            working,
            frontier,
            theta_runtime=theta,
        )
        receipts.append(
            {
                "layer_index": int(layer_index),
                "coordinate_count": int(len(frontier)),
                "atom_labels": [str(atom.atom_label) for atom in frontier],
                "base_atom_ids": list(base_atom_ids),
                "occurrence_indices": [_occurrence_index(atom) for atom in frontier],
            }
        )
    return working, theta, receipts


def _append_adjacent_duplicate_groups(
    state: APMcLachlanState,
    *,
    theta_runtime: np.ndarray,
    pool_atom_count: int,
    occurrence_count: int,
) -> tuple[APMcLachlanState, np.ndarray, list[dict[str, Any]]]:
    working = state
    theta = np.asarray(theta_runtime, dtype=float).reshape(-1)
    initial_frontier = _redundancy_frontier(
        working,
        pool_atom_count=pool_atom_count,
    )
    ordered_base_atom_ids = tuple(_base_atom_id(atom) for atom in initial_frontier)
    receipts: list[dict[str, Any]] = []
    for group_index, base_atom_id in enumerate(ordered_base_atom_ids):
        labels: list[str] = []
        occurrence_indices: list[int] = []
        for _ in range(int(occurrence_count)):
            frontier = _redundancy_frontier(
                working,
                pool_atom_count=pool_atom_count,
            )
            matches = tuple(
                atom for atom in frontier if _base_atom_id(atom) == base_atom_id
            )
            if len(matches) != 1:
                raise ValueError(
                    "adjacent redundancy stress could not resolve one pool atom for "
                    f"{base_atom_id!r}; found {len(matches)}"
                )
            atom = matches[0]
            working, theta = state_with_appended_atoms(
                working,
                (atom,),
                theta_runtime=theta,
            )
            labels.append(str(atom.atom_label))
            occurrence_indices.append(_occurrence_index(atom))
        receipts.append(
            {
                "group_index": int(group_index),
                "coordinate_count": int(len(labels)),
                "atom_labels": labels,
                "base_atom_ids": [str(base_atom_id)] * len(labels),
                "occurrence_indices": occurrence_indices,
            }
        )
    return working, theta, receipts


def _redundancy_frontier(
    state: APMcLachlanState,
    *,
    pool_atom_count: int,
) -> tuple[Any, ...]:
    frontier = candidate_append_atoms(
        state,
        allow_incomplete_candidate_pool=True,
        occurrence_policy=APPEND_OCCURRENCE_POLICY_LAYER_REUSE,
    )
    if len(frontier) != int(pool_atom_count):
        raise ValueError(
            "redundancy stress frontier does not match normalized pool size: "
            f"{len(frontier)} vs {pool_atom_count}"
        )
    return tuple(frontier)


def _base_atom_id(atom: Any) -> str:
    return str(dict(atom.metadata).get("base_atom_id", ""))


def _occurrence_index(atom: Any) -> int:
    return int(dict(atom.metadata).get("occurrence_index", 1))


__all__ = [
    "DEFAULT_REDUNDANCY_STRESS_PARITY_ATOL",
    "REDUNDANCY_STRESS_SCHEMA_V1",
    "REDUNDANCY_STRESS_LAYOUT_ADJACENT_DUPLICATES",
    "REDUNDANCY_STRESS_LAYOUT_LAYERED",
    "REDUNDANCY_STRESS_LAYOUTS",
    "RedundancyStressConfig",
    "RedundancyStressResult",
    "inject_zero_angle_redundancy_layers",
    "redundancy_stress_config_from_metadata",
]
