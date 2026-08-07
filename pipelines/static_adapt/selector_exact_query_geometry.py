"""Exact-state provider adapter for Formal-Manifold query closure.

This module performs the execution-chart derivative work that must stay out of
the nested ADAPT closures.  It returns the typed, round-local anchor/candidate
records consumed by :mod:`selector_query_closure`; it does not perform ranking
or charge classical contractions as estimator work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.scaffold.hh_continuation_scoring import (
    _array_fingerprint,
    _compiled_polynomial_fingerprint,
    _executor_for_terms,
    _horizontal_tangent,
    _ordered_scaffold_fingerprint,
)
from pipelines.static_adapt.exact_state_backend import (
    ExactOuterAnchorState,
    FORMAL_OUTER_EXACT_ANCHOR_SCHEMA,
)
from pipelines.static_adapt.geometry_fingerprints import (
    candidate_coordinate_fingerprint,
    candidate_generator_fingerprint,
    compiled_hamiltonian_fingerprint,
    fingerprint_jsonable as _fingerprint,
    ordered_scaffold_fingerprint,
)
from pipelines.static_adapt.selector_query_closure import (
    CandidateTangentRecord,
    QueryPrimitiveLedger,
    QueryReceipt,
    SelectorGeometryAnchor,
    build_candidate_tangent_record,
    projective_state_fingerprint,
)


@dataclass(frozen=True)
class ExactQueryClosedCandidateGeometry:
    anchor: SelectorGeometryAnchor
    candidate_record: CandidateTangentRecord
    state_reconstruction_delta_norm: float
    active_combined_indices: tuple[int, ...]
    candidate_combined_index: int
    retained_anchor_reused: bool

    def portable_payload(self) -> dict[str, Any]:
        return {
            "schema": "formal_manifold_exact_query_closed_candidate_geometry_v1",
            "anchor": self.anchor.portable_payload(),
            "candidate_record": self.candidate_record.portable_payload(),
            "state_reconstruction_delta_norm": float(
                self.state_reconstruction_delta_norm
            ),
            "active_combined_indices": list(self.active_combined_indices),
            "candidate_combined_index": int(self.candidate_combined_index),
            "retained_anchor_reused": bool(self.retained_anchor_reused),
            "live_tangent_handles_serialized": False,
            "dense_statevector_serialized": False,
        }


@dataclass(frozen=True)
class FormalOuterAnchorReuse:
    """FM-only adapter from committed terminal geometry to selector scope."""

    anchor: SelectorGeometryAnchor | None
    geometry_reuse_receipt: QueryReceipt | None
    energy_reuse_receipt: QueryReceipt | None
    precharged_energy_primitive_id: str | None
    status: str
    reason: str
    mismatched_fields: tuple[str, ...] = ()

    @property
    def reusable(self) -> bool:
        return bool(
            self.anchor is not None
            and self.geometry_reuse_receipt is not None
            and self.energy_reuse_receipt is not None
            and self.precharged_energy_primitive_id is not None
        )

    def portable_payload(self) -> dict[str, Any]:
        return {
            "schema": "formal_manifold_outer_anchor_reuse_v1",
            "status": str(self.status),
            "reason": str(self.reason),
            "reusable": bool(self.reusable),
            "mismatched_fields": list(self.mismatched_fields),
            "precharged_energy_primitive_id": (
                self.precharged_energy_primitive_id
            ),
            "geometry_source_primitive_ids": (
                []
                if self.geometry_reuse_receipt is None
                else list(self.geometry_reuse_receipt.all_primitive_ids)
            ),
            "measurement_origin_branch_id": (
                None if self.anchor is None else str(self.anchor.branch_id)
            ),
        }


def build_retained_selector_anchor_from_formal_state(
    *,
    state: ExactOuterAnchorState | None,
    source_query_ledger: QueryPrimitiveLedger,
    psi_state: np.ndarray,
    theta: np.ndarray,
    branch_id: str,
    ordered_scaffold_fingerprint: str,
    coordinate_registry_fingerprint: str,
    parameterization_tie_map_fingerprint: str,
    hamiltonian_fingerprint: str,
    active_coordinate_indices: Sequence[int],
) -> FormalOuterAnchorReuse:
    """Reuse a committed FM endpoint anchor or return a typed refresh reason.

    This function does not execute selector logic and never creates a new
    primitive identity.  It only re-exposes exact arrays already committed by
    FM under their original measurement IDs.
    """

    def unavailable(
        reason: str, mismatches: Sequence[str] = ()
    ) -> FormalOuterAnchorReuse:
        return FormalOuterAnchorReuse(
            anchor=None,
            geometry_reuse_receipt=None,
            energy_reuse_receipt=None,
            precharged_energy_primitive_id=None,
            status="refresh_required",
            reason=str(reason),
            mismatched_fields=tuple(str(value) for value in mismatches),
        )

    if state is None:
        return unavailable("no_committed_formal_state")
    if not isinstance(state, ExactOuterAnchorState):
        return unavailable("committed_formal_state_type_mismatch")
    if not isinstance(source_query_ledger, QueryPrimitiveLedger):
        raise TypeError("source_query_ledger must be a QueryPrimitiveLedger.")
    provenance = state.outer_exact_anchor
    if provenance is None:
        return unavailable("no_committed_exact_outer_anchor")
    if str(provenance.schema) != FORMAL_OUTER_EXACT_ANCHOR_SCHEMA:
        return unavailable("outer_anchor_schema_mismatch")
    if not (
        bool(provenance.exact_metric_observed)
        and bool(provenance.endpoint_tangents_observed)
    ):
        return unavailable("outer_anchor_is_not_exact_endpoint_geometry")
    supplied_state = np.asarray(psi_state, dtype=complex).reshape(-1)
    supplied_theta = np.asarray(theta, dtype=float).reshape(-1)
    active_indices = tuple(int(index) for index in active_coordinate_indices)
    if len(set(active_indices)) != len(active_indices) or any(
        index < 0 or index >= supplied_theta.size for index in active_indices
    ):
        return unavailable("outer_anchor_active_indices_invalid")
    state_fingerprint = projective_state_fingerprint(supplied_state)
    state_theta_fingerprint = _fingerprint(
        [float(value) for value in np.asarray(state.theta, dtype=float).tolist()]
    )
    expected_scope: Mapping[str, Any] = {
        "physical_state_fingerprint": state_fingerprint,
        "formal_theta_fingerprint": state_theta_fingerprint,
        "coordinate_registry_fingerprint": str(
            coordinate_registry_fingerprint
        ),
        "manifold_id": str(state.manifold_id),
        "parameterization_mode": str(state.parameterization_mode),
        "ordered_scaffold_fingerprint": str(
            ordered_scaffold_fingerprint
        ),
        "parameterization_tie_map_fingerprint": str(
            parameterization_tie_map_fingerprint
        ),
        "hamiltonian_fingerprint": str(hamiltonian_fingerprint),
        "measurement_origin_branch_id": str(branch_id),
        "whitening_id": str(state.whitening_id),
        "frame_id": str(state.frame_id),
        "logical_range_id": str(state.logical_range_id),
        "coordinate_count": int(supplied_theta.size),
        "rank": int(state.rank),
    }
    mismatches = [
        name
        for name, expected in expected_scope.items()
        if getattr(provenance, name) != expected
    ]
    if mismatches:
        return unavailable("outer_anchor_scope_mismatch", mismatches)
    if tuple(state.registry) and len(state.registry) != supplied_theta.size:
        return unavailable("outer_anchor_registry_dimension_mismatch")
    if np.asarray(state.theta).shape != supplied_theta.shape or not np.allclose(
        np.asarray(state.theta, dtype=float),
        supplied_theta,
        rtol=0.0,
        atol=1.0e-12,
    ):
        return unavailable("outer_anchor_theta_value_mismatch")
    if projective_state_fingerprint(state.statevector) != state_fingerprint:
        return unavailable("outer_anchor_state_value_mismatch")
    if (
        projective_state_fingerprint(state.frame_anchor_statevector)
        != state_fingerprint
    ):
        return unavailable("outer_anchor_frame_state_mismatch")
    tangents_full = np.asarray(state.tangents, dtype=complex)
    if tangents_full.shape != (supplied_state.size, supplied_theta.size):
        return unavailable("outer_anchor_tangent_shape_mismatch")
    if np.asarray(state.b).shape != (supplied_theta.size,):
        return unavailable("outer_anchor_differential_shape_mismatch")
    source_ids = set(provenance.source_primitive_ids)
    unknown_ids = source_ids - set(source_query_ledger.unique_primitive_ids)
    if unknown_ids:
        return unavailable(
            "outer_anchor_source_ids_missing_from_ledger",
            tuple(sorted(unknown_ids)),
        )
    ledger_kind_mismatches = [
        primitive_id
        for primitive_id, expected_kind in provenance.primitive_kind_by_id
        if source_query_ledger.primitive_kind(primitive_id) != expected_kind
    ]
    if ledger_kind_mismatches:
        return unavailable(
            "outer_anchor_source_kind_mismatch",
            ledger_kind_mismatches,
        )
    active_tangents = tuple(
        np.asarray(tangents_full[:, index], dtype=complex).copy()
        for index in active_indices
    )
    active_matrix = (
        np.column_stack(active_tangents)
        if active_tangents
        else np.zeros((supplied_state.size, 0), dtype=complex)
    )
    G_AA = np.asarray(
        np.real(np.conjugate(active_matrix).T @ active_matrix), dtype=float
    )
    b_A = np.asarray(state.b, dtype=float)[list(active_indices)]
    kind_map = provenance.kind_map
    geometry_receipt = QueryReceipt(
        primitive_ids_requested=(),
        primitive_ids_reused=provenance.geometry_source_primitive_ids,
        returned_fields=("b_A", "G_AA", "active_tangent_handles"),
        closure_capabilities=(
            "committed_exact_terminal_tangent_handle",
            "cross_outer_iteration_geometry_reuse",
            "common_state_tangent_contraction",
        ),
        provenance_by_field=(
            ("b_A", "committed_exact_terminal_coordinate_differential"),
            ("G_AA", "committed_exact_terminal_fubini_study_metric"),
            (
                "active_tangent_handles",
                "committed_exact_terminal_horizontal_tangents",
            ),
        ),
        primitive_kind_by_id=tuple(
            (primitive_id, kind_map[primitive_id])
            for primitive_id in provenance.geometry_source_primitive_ids
        ),
        provider_kind="formal_manifold_committed_exact_anchor_v1",
        statevector_shortcut_used=True,
    )
    energy_receipt = QueryReceipt(
        primitive_ids_requested=(),
        primitive_ids_reused=(provenance.energy_primitive_id,),
        returned_fields=("energy",),
        closure_capabilities=("cross_outer_iteration_energy_reuse",),
        provenance_by_field=(
            ("energy", "committed_exact_terminal_energy"),
        ),
        primitive_kind_by_id=(
            (
                provenance.energy_primitive_id,
                kind_map[provenance.energy_primitive_id],
            ),
        ),
        provider_kind="formal_manifold_committed_exact_anchor_v1",
        statevector_shortcut_used=True,
    )
    anchor = SelectorGeometryAnchor(
        state_fingerprint=state_fingerprint,
        branch_id=str(branch_id),
        manifold_id=str(state.manifold_id),
        ordered_scaffold_fingerprint=str(
            ordered_scaffold_fingerprint
        ),
        theta_fingerprint=_array_fingerprint(supplied_theta),
        coordinate_registry_fingerprint=str(
            coordinate_registry_fingerprint
        ),
        parameterization_mode=str(state.parameterization_mode),
        parameterization_tie_map_fingerprint=str(
            parameterization_tie_map_fingerprint
        ),
        hamiltonian_fingerprint=str(hamiltonian_fingerprint),
        active_coordinate_indices=active_indices,
        active_tangent_handles=active_tangents,
        G_AA=G_AA,
        b_A=b_A,
        gram_provenance="committed_exact_terminal_metric_reuse_v1",
        differential_provenance=(
            "committed_exact_terminal_coordinate_differential_reuse_v1"
        ),
        source_query_receipts=(geometry_receipt,),
    )
    return FormalOuterAnchorReuse(
        anchor=anchor,
        geometry_reuse_receipt=geometry_receipt,
        energy_reuse_receipt=energy_receipt,
        precharged_energy_primitive_id=str(provenance.energy_primitive_id),
        status="reused",
        reason="committed_terminal_exact_geometry_scope_match",
        mismatched_fields=(),
    )


def build_exact_query_closed_candidate_geometry(
    *,
    selected_ops: Sequence[Any],
    theta: np.ndarray,
    psi_ref: np.ndarray,
    psi_state: np.ndarray,
    hpsi_state: np.ndarray,
    candidate_term: Any,
    insertion_position: int,
    active_coordinate_indices: Sequence[int],
    branch_id: str,
    manifold_id: str,
    coordinate_registry_fingerprint: str,
    parameterization_tie_map_fingerprint: str,
    h_compiled: Any,
    candidate_query_receipt: QueryReceipt,
    anchor_query_receipts: Sequence[QueryReceipt] = (),
    retained_anchor: SelectorGeometryAnchor | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
    state_consistency_tolerance: float = 1.0e-8,
) -> ExactQueryClosedCandidateGeometry:
    """Build first-order insertion geometry without requesting second derivatives."""

    selected = list(selected_ops)
    theta_old = np.asarray(theta, dtype=float).reshape(-1)
    if theta_old.size != len(selected):
        raise ValueError(
            "exact query-closure currently requires one logical coordinate per "
            "selected generator."
        )
    position = int(insertion_position)
    if position < 0 or position > len(selected):
        raise ValueError("candidate insertion position is outside the scaffold.")
    active_source = tuple(int(index) for index in active_coordinate_indices)
    if len(set(active_source)) != len(active_source) or any(
        index < 0 or index >= len(selected) for index in active_source
    ):
        raise ValueError("active coordinate indices are invalid for the scaffold.")
    combined_ops = [*selected[:position], candidate_term, *selected[position:]]
    combined_theta = np.insert(theta_old, position, 0.0)
    active_combined = tuple(
        int(index if index < position else index + 1) for index in active_source
    )
    requested = (
        (position,)
        if retained_anchor is not None
        else tuple((*active_combined, position))
    )
    executor = _executor_for_terms(
        combined_ops,
        pauli_action_cache=pauli_action_cache,
    )
    reconstructed, tangents_by_index = executor.prepare_state_with_parameter_tangents(
        combined_theta,
        np.asarray(psi_ref, dtype=complex),
        parameter_indices=requested,
    )
    reconstructed_state = np.asarray(reconstructed, dtype=complex).reshape(-1)
    supplied_state = np.asarray(psi_state, dtype=complex).reshape(-1)
    if reconstructed_state.shape != supplied_state.shape:
        raise ValueError("reconstructed and supplied states have incompatible shapes.")
    overlap = complex(np.vdot(supplied_state, reconstructed_state))
    phase = overlap / abs(overlap) if abs(overlap) > 0.0 else 1.0 + 0.0j
    aligned_state = reconstructed_state / phase
    state_delta = float(np.linalg.norm(aligned_state - supplied_state))
    tolerance = float(max(1.0e-12, state_consistency_tolerance))
    if state_delta > tolerance:
        raise ValueError(
            "exact Phase-I insertion geometry reconstructed a different state: "
            f"delta={state_delta:.6g}, tolerance={tolerance:.6g}."
        )
    aligned_derivatives = {
        int(index): np.asarray(value / phase, dtype=complex)
        for index, value in tangents_by_index.items()
    }
    if retained_anchor is None:
        active_derivatives = tuple(
            aligned_derivatives[index] for index in active_combined
        )
        active_tangents = tuple(
            _horizontal_tangent(supplied_state, derivative)
            for derivative in active_derivatives
        )
    else:
        expected_anchor_scope = {
            "state_fingerprint": projective_state_fingerprint(supplied_state),
            "branch_id": str(branch_id),
            "manifold_id": str(manifold_id),
            "ordered_scaffold_fingerprint": (
                _ordered_scaffold_fingerprint(selected)
            ),
            "theta_fingerprint": _array_fingerprint(theta_old),
            "coordinate_registry_fingerprint": str(
                coordinate_registry_fingerprint
            ),
            "parameterization_mode": str(executor.parameterization_mode),
            "parameterization_tie_map_fingerprint": str(
                parameterization_tie_map_fingerprint
            ),
            "hamiltonian_fingerprint": _compiled_polynomial_fingerprint(
                h_compiled
            ),
            "active_coordinate_indices": active_source,
        }
        mismatches = [
            name
            for name, expected in expected_anchor_scope.items()
            if getattr(retained_anchor, name) != expected
        ]
        if mismatches:
            raise ValueError(
                "retained selector anchor scope mismatch: "
                + ", ".join(mismatches)
            )
        if len(retained_anchor.active_tangent_handles) != len(active_source):
            raise ValueError(
                "retained selector anchor lacks its round-local tangent handles."
            )
        active_derivatives = ()
        active_tangents = tuple(retained_anchor.active_tangent_handles)
    candidate_derivative = aligned_derivatives[position]
    candidate_tangent = _horizontal_tangent(
        supplied_state, candidate_derivative
    )
    active_count = len(active_tangents)
    if retained_anchor is None:
        G_AA = np.zeros((active_count, active_count), dtype=float)
        for left in range(active_count):
            for right in range(left, active_count):
                value = float(
                    np.real(np.vdot(active_tangents[left], active_tangents[right]))
                )
                G_AA[left, right] = value
                G_AA[right, left] = value
    else:
        G_AA = np.asarray(retained_anchor.G_AA, dtype=float)
    hpsi = np.asarray(hpsi_state, dtype=complex).reshape(-1)
    b_A = (
        np.asarray(
            [
                2.0 * float(np.real(np.vdot(value, hpsi)))
                for value in active_derivatives
            ],
            dtype=float,
        )
        if retained_anchor is None
        else np.asarray(retained_anchor.b_A, dtype=float)
    )
    b_B = 2.0 * float(np.real(np.vdot(candidate_derivative, hpsi)))
    state_fingerprint = projective_state_fingerprint(supplied_state)
    scaffold_fingerprint = _ordered_scaffold_fingerprint(selected)
    theta_fingerprint = _array_fingerprint(theta_old)
    anchor = (
        retained_anchor
        if retained_anchor is not None
        else SelectorGeometryAnchor(
            state_fingerprint=state_fingerprint,
            branch_id=str(branch_id),
            manifold_id=str(manifold_id),
            ordered_scaffold_fingerprint=scaffold_fingerprint,
            theta_fingerprint=theta_fingerprint,
            coordinate_registry_fingerprint=str(
                coordinate_registry_fingerprint
            ),
            parameterization_mode=str(executor.parameterization_mode),
            parameterization_tie_map_fingerprint=str(
                parameterization_tie_map_fingerprint
            ),
            hamiltonian_fingerprint=_compiled_polynomial_fingerprint(
                h_compiled
            ),
            active_coordinate_indices=active_source,
            active_tangent_handles=active_tangents,
            G_AA=G_AA,
            b_A=b_A,
            gram_provenance="exact_state_retained_anchor",
            differential_provenance="exact_state_coordinate_differential",
            source_query_receipts=tuple(anchor_query_receipts),
        )
    )
    generator_fingerprint = candidate_generator_fingerprint(candidate_term)
    record = build_candidate_tangent_record(
        anchor=anchor,
        candidate_fingerprint=generator_fingerprint,
        candidate_registry_entry_fingerprint=_fingerprint(
            {
                "candidate_generator_fingerprint": generator_fingerprint,
                "insertion_position": position,
                "parameterization_tie_map_fingerprint": (
                    parameterization_tie_map_fingerprint
                ),
            }
        ),
        insertion_position=position,
        tangent_handle=candidate_tangent,
        differential=b_B,
        query_receipts=(candidate_query_receipt,),
    )
    return ExactQueryClosedCandidateGeometry(
        anchor=anchor,
        candidate_record=record,
        state_reconstruction_delta_norm=state_delta,
        active_combined_indices=active_combined,
        candidate_combined_index=position,
        retained_anchor_reused=bool(retained_anchor is not None),
    )


__all__ = [
    "ExactQueryClosedCandidateGeometry",
    "FormalOuterAnchorReuse",
    "build_retained_selector_anchor_from_formal_state",
    "build_exact_query_closed_candidate_geometry",
    "candidate_generator_fingerprint",
    "candidate_coordinate_fingerprint",
    "compiled_hamiltonian_fingerprint",
    "ordered_scaffold_fingerprint",
]
