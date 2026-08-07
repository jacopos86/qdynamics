"""Exact-state provider adapter for Formal-Manifold query closure.

This module performs the execution-chart derivative work that must stay out of
the nested ADAPT closures.  It returns the typed, round-local anchor/candidate
records consumed by :mod:`selector_query_closure`; it does not perform ranking
or charge classical contractions as estimator work.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Sequence

import numpy as np

from pipelines.scaffold.hh_continuation_scoring import (
    _array_fingerprint,
    _candidate_coordinate_fingerprint,
    _compiled_polynomial_fingerprint,
    _executor_for_terms,
    _horizontal_tangent,
    _ordered_scaffold_fingerprint,
)
from pipelines.static_adapt.selector_query_closure import (
    CandidateTangentRecord,
    QueryReceipt,
    SelectorGeometryAnchor,
    build_candidate_tangent_record,
    projective_state_fingerprint,
)


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def candidate_generator_fingerprint(candidate_term: Any) -> str:
    polynomial = getattr(candidate_term, "polynomial", None)
    term_provider = getattr(polynomial, "return_polynomial", None)
    polynomial_terms = (
        tuple(term_provider())
        if callable(term_provider)
        else tuple(getattr(polynomial, "terms", ()))
    )
    terms = []
    for term in polynomial_terms:
        coefficient = complex(
            getattr(term, "p_coeff", getattr(term, "coeff", 0.0))
        )
        word_builder = getattr(term, "pw2strng", None)
        if callable(word_builder):
            pauli_word = str(word_builder())
        else:
            pauli_word = str(
                getattr(term, "pauli", getattr(term, "pauli_exyz", ""))
            )
        nq_builder = getattr(term, "nqubit", None)
        nq = int(nq_builder()) if callable(nq_builder) else int(
            getattr(term, "nq", len(pauli_word))
        )
        terms.append(
            {
                "coeff_real": float(coefficient.real),
                "coeff_imag": float(coefficient.imag),
                "nq": nq,
                "pauli": pauli_word,
            }
        )
    return _fingerprint(
        {
            "label": str(getattr(candidate_term, "label", "")),
            "execution_mode": str(
                getattr(candidate_term, "execution_mode", "termwise_product")
                or "termwise_product"
            ).strip().lower(),
            "terms": terms,
        }
    )


def candidate_coordinate_fingerprint(
    candidate_term: Any, *, insertion_position: int
) -> str:
    return str(
        _candidate_coordinate_fingerprint(
            candidate_term,
            position_id=int(insertion_position),
        )
    )


def compiled_hamiltonian_fingerprint(h_compiled: Any) -> str:
    return str(_compiled_polynomial_fingerprint(h_compiled))


def ordered_scaffold_fingerprint(selected_ops: Sequence[Any]) -> str:
    return str(_ordered_scaffold_fingerprint(selected_ops))


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
    "build_exact_query_closed_candidate_geometry",
    "candidate_generator_fingerprint",
    "candidate_coordinate_fingerprint",
    "compiled_hamiltonian_fingerprint",
    "ordered_scaffold_fingerprint",
]
