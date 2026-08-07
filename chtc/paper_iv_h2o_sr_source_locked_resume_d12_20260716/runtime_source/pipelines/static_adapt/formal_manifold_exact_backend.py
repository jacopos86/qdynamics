"""Exact-state adapter for the formal manifold reoptimization route.

The adaptive pipeline stores parameters in the expanded runtime layout even
when a :class:`~src.quantum.compiled_ansatz.CompiledAnsatzExecutor` exposes one
shared coordinate per logical generator.  This module is the boundary between
those two conventions: manifold coordinates always match the executor's
coordinates, while :meth:`CompiledExactManifoldAdapter.lift_to_runtime`
returns the storage representation expected by the existing pipeline.

All derivatives are analytic executor tangents.  Finite differences are not a
fallback in this adapter.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.static_adapt.formal_manifold_warm_start import (
    ExactStateBackend,
    ExactStateEvaluation,
)
from src.quantum.ansatz_parameterization import (
    AnsatzParameterLayout,
    expand_legacy_logical_theta,
    project_runtime_theta_block_mean,
    serialize_layout,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import (
    CompiledPolynomialAction,
    energy_via_one_apply,
)


EXACT_BACKEND_SCHEMA = "formal_manifold_compiled_exact_backend_v1"
COORDINATE_REGISTRY_SCHEMA = "formal_manifold_coordinate_registry_v1"
EXACT_STATE_PROVENANCE = "exact_state_computed"


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


def _finite_real_vector(
    value: np.ndarray | Sequence[float],
    *,
    expected_size: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if int(array.size) != int(expected_size):
        raise ValueError(
            f"{name} length mismatch: got {array.size}, expected {expected_size}."
        )
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must contain only finite real values.")
    return np.asarray(array, dtype=float).copy()


def _coordinate_registry_payload(
    layout: AnsatzParameterLayout,
    *,
    parameterization_mode: str,
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    """Return insertion-stable coordinate ids and their JSON-safe records.

    Current logical/runtime positions belong only in the records.  Identity is
    derived from the labeled block and its ordered gate occurrences, so
    inserting an unrelated block does not rename inherited coordinates.  The
    duplicate ordinal is a deterministic fallback for repeated identical
    blocks; the v1 route preflight may disallow repeats altogether.
    """

    coordinate_ids: list[str] = []
    records: list[dict[str, Any]] = []
    duplicate_count_by_block_fingerprint: dict[str, int] = {}
    block_identities: list[tuple[str, int, list[dict[str, Any]]]] = []
    for block in layout.blocks:
        identity_terms = [
            {
                "coefficient_real": float(spec.coeff_real),
                "nq": int(spec.nq),
                "pauli_exyz": str(spec.pauli_exyz),
                "runtime_occurrence_index": int(local_index),
            }
            for local_index, spec in enumerate(block.terms)
        ]
        block_fingerprint = _json_sha256(
            {
                "candidate_label": str(block.candidate_label),
                "runtime_terms": identity_terms,
            }
        )
        duplicate_ordinal = int(
            duplicate_count_by_block_fingerprint.get(block_fingerprint, 0)
        )
        duplicate_count_by_block_fingerprint[block_fingerprint] = (
            duplicate_ordinal + 1
        )
        block_identities.append(
            (block_fingerprint, duplicate_ordinal, identity_terms)
        )

    if parameterization_mode == "logical_shared":
        for block, block_identity in zip(layout.blocks, block_identities):
            block_fingerprint, duplicate_ordinal, identity_terms = block_identity
            runtime_terms = [
                {
                    **identity_term,
                    "runtime_index": int(block.runtime_start + local_index),
                }
                for local_index, identity_term in enumerate(identity_terms)
            ]
            coordinate_id = (
                f"logical:block:{block_fingerprint}:"
                f"instance:{duplicate_ordinal}"
            )
            coordinate_ids.append(coordinate_id)
            records.append(
                {
                    "block_fingerprint_sha256": block_fingerprint,
                    "candidate_label": str(block.candidate_label),
                    "coordinate_id": coordinate_id,
                    "coordinate_index": int(block.logical_index),
                    "coordinate_kind": "logical_shared",
                    "duplicate_instance_ordinal": duplicate_ordinal,
                    "logical_index": int(block.logical_index),
                    "runtime_count": int(block.runtime_count),
                    "runtime_indices": list(
                        range(int(block.runtime_start), int(block.runtime_stop))
                    ),
                    "runtime_start": int(block.runtime_start),
                    "runtime_terms": runtime_terms,
                }
            )
    elif parameterization_mode == "per_pauli_term":
        for block, block_identity in zip(layout.blocks, block_identities):
            block_fingerprint, duplicate_ordinal, identity_terms = block_identity
            for local_index, (spec, identity_term) in enumerate(
                zip(block.terms, identity_terms)
            ):
                runtime_index = int(block.runtime_start + local_index)
                gate_fingerprint = _json_sha256(
                    {
                        "block_fingerprint_sha256": block_fingerprint,
                        "gate_occurrence": identity_term,
                    }
                )
                coordinate_id = (
                    f"runtime:block:{block_fingerprint}:"
                    f"instance:{duplicate_ordinal}:"
                    f"gate:{gate_fingerprint}"
                )
                coordinate_ids.append(coordinate_id)
                records.append(
                    {
                        "block_fingerprint_sha256": block_fingerprint,
                        "candidate_label": str(block.candidate_label),
                        "coefficient_real": float(spec.coeff_real),
                        "coordinate_id": coordinate_id,
                        "coordinate_index": runtime_index,
                        "coordinate_kind": "per_pauli_term",
                        "duplicate_instance_ordinal": duplicate_ordinal,
                        "gate_fingerprint_sha256": gate_fingerprint,
                        "logical_index": int(block.logical_index),
                        "nq": int(spec.nq),
                        "pauli_exyz": str(spec.pauli_exyz),
                        "runtime_index": runtime_index,
                        "runtime_occurrence_index": int(local_index),
                    }
                )
    else:  # Defensive: constructor validation should make this unreachable.
        raise ValueError(
            "parameterization_mode must be 'logical_shared' or 'per_pauli_term'."
        )
    if len(set(coordinate_ids)) != len(coordinate_ids):
        raise ValueError("coordinate registry ids must be unique within a layout.")
    return tuple(coordinate_ids), records


class CompiledExactManifoldAdapter:
    """Bind a compiled ansatz and Hamiltonian to the exact manifold backend.

    Parameters are deliberately supplied as expanded runtime storage.  The
    adapter derives the executor-coordinate anchor ``x0`` and never presents
    redundant runtime storage as the formal route's coordinate chart.
    """

    def __init__(
        self,
        *,
        executor: CompiledAnsatzExecutor,
        layout: AnsatzParameterLayout,
        theta_runtime: np.ndarray | Sequence[float],
        psi_ref: np.ndarray | Sequence[complex],
        h_compiled: CompiledPolynomialAction,
        manifold_id: str = "compiled_ansatz_exact_manifold_v1",
    ) -> None:
        mode = str(executor.parameterization_mode)
        if mode not in {"logical_shared", "per_pauli_term"}:
            raise ValueError(
                "executor parameterization_mode must be 'logical_shared' or "
                f"'per_pauli_term', got {mode!r}."
            )
        if not isinstance(layout, AnsatzParameterLayout):
            raise TypeError("layout must be an AnsatzParameterLayout.")
        layout_payload = serialize_layout(layout)
        executor_layout_payload = serialize_layout(executor.layout)
        if _canonical_json_bytes(layout_payload) != _canonical_json_bytes(
            executor_layout_payload
        ):
            raise ValueError("layout must exactly match executor.layout.")
        if not str(manifold_id).strip():
            raise ValueError("manifold_id must be a non-empty string.")
        if not math.isfinite(float(layout.coefficient_tolerance)) or float(
            layout.coefficient_tolerance
        ) < 0.0:
            raise ValueError("layout coefficient_tolerance must be finite and nonnegative.")

        theta_runtime_array = _finite_real_vector(
            theta_runtime,
            expected_size=int(layout.runtime_parameter_count),
            name="theta_runtime",
        )
        if mode == "logical_shared":
            x0 = np.asarray(
                project_runtime_theta_block_mean(theta_runtime_array, layout),
                dtype=float,
            )
            coordinate_count = int(layout.logical_parameter_count)
        else:
            x0 = theta_runtime_array.copy()
            coordinate_count = int(layout.runtime_parameter_count)
        if int(executor.num_parameters) != coordinate_count:
            raise ValueError(
                "executor coordinate count disagrees with the resolved layout: "
                f"{executor.num_parameters} vs {coordinate_count}."
            )

        psi_reference = np.asarray(psi_ref, dtype=complex).reshape(-1).copy()
        if not bool(
            np.all(np.isfinite(psi_reference.real))
            and np.all(np.isfinite(psi_reference.imag))
        ):
            raise ValueError("psi_ref must contain only finite complex values.")
        if int(h_compiled.nq) < 0:
            raise ValueError("h_compiled.nq must be nonnegative.")
        expected_dimension = 1 << int(h_compiled.nq)
        if int(psi_reference.size) != expected_dimension:
            raise ValueError(
                "psi_ref length mismatch: "
                f"got {psi_reference.size}, expected {expected_dimension} "
                f"for h_compiled.nq={h_compiled.nq}."
            )
        if executor.nq is not None and int(executor.nq) != int(h_compiled.nq):
            raise ValueError(
                "executor and Hamiltonian qubit counts disagree: "
                f"{executor.nq} vs {h_compiled.nq}."
            )
        norm = float(np.linalg.norm(psi_reference))
        if not math.isfinite(norm) or not np.isclose(
            norm, 1.0, rtol=1.0e-10, atol=1.0e-12
        ):
            raise ValueError(
                f"psi_ref must be normalized for Fubini--Study geometry; norm={norm}."
            )
        h_coefficients = [complex(term.coeff) for term in h_compiled.terms]
        if not all(
            math.isfinite(float(coefficient.real))
            and math.isfinite(float(coefficient.imag))
            for coefficient in h_coefficients
        ):
            raise ValueError("h_compiled coefficients must be finite.")
        h_imag_max = max(
            (abs(float(coefficient.imag)) for coefficient in h_coefficients),
            default=0.0,
        )
        if h_imag_max > 1.0e-12:
            raise ValueError(
                "h_compiled must be Hermitian in the Pauli basis (real "
                f"coefficients); max imaginary coefficient={h_imag_max}."
            )

        coordinate_ids, coordinate_records = _coordinate_registry_payload(
            layout,
            parameterization_mode=mode,
        )
        if len(coordinate_ids) != coordinate_count:
            raise ValueError(
                "coordinate registry length mismatch: "
                f"got {len(coordinate_ids)}, expected {coordinate_count}."
            )
        layout_sha256 = _json_sha256(layout_payload)
        coordinate_registry_sha256 = _json_sha256(list(coordinate_ids))
        coordinate_records_sha256 = _json_sha256(coordinate_records)
        metadata: dict[str, Any] = {
            "coordinate_count": coordinate_count,
            "coordinate_identity_position_independent": True,
            "coordinate_identity_repeat_policy": (
                "block_fingerprint_then_duplicate_instance_ordinal"
            ),
            "coordinate_records": coordinate_records,
            "coordinate_records_sha256": coordinate_records_sha256,
            "coordinate_registry_schema": COORDINATE_REGISTRY_SCHEMA,
            "coordinate_registry_sha256": coordinate_registry_sha256,
            "derivative_method": "analytic_compiled_parameter_tangents_v1",
            "energy_method": "energy_via_one_apply",
            "finite_differences_used": False,
            "layout": layout_payload,
            "layout_sha256": layout_sha256,
            "manifold_id": str(manifold_id),
            "parameterization_mode": mode,
            "provenance": EXACT_STATE_PROVENANCE,
            "route_coordinate_convention": "executor_coordinates",
            "runtime_lift_method": (
                "expand_legacy_logical_theta"
                if mode == "logical_shared"
                else "identity"
            ),
            "schema": EXACT_BACKEND_SCHEMA,
            "storage_to_route_method": (
                "project_runtime_theta_block_mean"
                if mode == "logical_shared"
                else "identity"
            ),
            "theta_runtime_count": int(layout.runtime_parameter_count),
        }

        self.executor = executor
        self.layout = layout
        self.h_compiled = h_compiled
        self.manifold_id = str(manifold_id)
        self.parameterization_mode = mode
        self._coordinate_count = coordinate_count
        self._coordinate_records = deepcopy(coordinate_records)
        self._coordinate_registry = coordinate_ids
        self._layout_sha256 = layout_sha256
        self._metadata = deepcopy(metadata)
        self._psi_ref = psi_reference
        self._theta_runtime0 = theta_runtime_array
        self._x0 = np.asarray(x0, dtype=float).reshape(-1).copy()
        self.backend = ExactStateBackend(
            evaluate_fn=self._evaluate,
            coordinate_registry=self._coordinate_registry,
            manifold_id=self.manifold_id,
            parameterization_mode=self.parameterization_mode,
            metadata=deepcopy(metadata),
        )

    @property
    def x0(self) -> np.ndarray:
        """Initial point in executor coordinates, returned as a safe copy."""

        return self._x0.copy()

    @property
    def coordinate_registry(self) -> tuple[str, ...]:
        return tuple(self._coordinate_registry)

    @property
    def summary(self) -> dict[str, Any]:
        """Stable, JSON-safe adapter identity and storage-coordinate anchor."""

        return {
            **deepcopy(self._metadata),
            "coordinate_registry": list(self._coordinate_registry),
            "theta_runtime0": self._theta_runtime0.astype(float).tolist(),
            "x0": self._x0.astype(float).tolist(),
        }

    def lift_to_runtime(
        self,
        x: np.ndarray | Sequence[float],
    ) -> np.ndarray:
        """Lift executor coordinates back to expanded pipeline storage."""

        coordinate = _finite_real_vector(
            x,
            expected_size=self._coordinate_count,
            name="x",
        )
        if self.parameterization_mode == "logical_shared":
            return np.asarray(
                expand_legacy_logical_theta(coordinate, self.layout),
                dtype=float,
            )
        return coordinate.copy()

    def _evaluate(self, x: np.ndarray) -> ExactStateEvaluation:
        coordinate = _finite_real_vector(
            x,
            expected_size=self._coordinate_count,
            name="theta",
        )
        statevector, tangent_by_index = (
            self.executor.prepare_state_with_parameter_tangents(
                coordinate,
                self._psi_ref,
            )
        )
        psi = np.asarray(statevector, dtype=complex).reshape(-1)
        if self._coordinate_count:
            tangents = np.column_stack(
                [
                    np.asarray(tangent_by_index[index], dtype=complex).reshape(-1)
                    for index in range(self._coordinate_count)
                ]
            )
        else:
            tangents = np.zeros((int(psi.size), 0), dtype=complex)
        if tangents.shape != (int(psi.size), self._coordinate_count):
            raise ValueError(
                "compiled tangent matrix shape mismatch: "
                f"got {tangents.shape}, expected "
                f"{(int(psi.size), self._coordinate_count)}."
            )
        energy, hpsi = energy_via_one_apply(psi, self.h_compiled)
        gradient = np.asarray(
            2.0 * np.real(np.conjugate(tangents).T @ hpsi),
            dtype=float,
        ).reshape(-1)
        evaluation_metadata: Mapping[str, Any] = {
            "coordinate_registry_sha256": self._metadata[
                "coordinate_registry_sha256"
            ],
            "derivative_identity": "2*Re(<dpsi_i|Hpsi>)",
            "derivative_method": "analytic_compiled_parameter_tangents_v1",
            "energy_method": "energy_via_one_apply",
            "energy_provenance": EXACT_STATE_PROVENANCE,
            "finite_differences_used": False,
            "gradient_provenance": EXACT_STATE_PROVENANCE,
            "hamiltonian_apply_count": 1,
            "layout_sha256": self._layout_sha256,
            "statevector_provenance": EXACT_STATE_PROVENANCE,
            "tangent_provenance": EXACT_STATE_PROVENANCE,
        }
        return ExactStateEvaluation(
            energy=float(energy),
            gradient=gradient,
            statevector=psi,
            tangents=tangents,
            metadata=evaluation_metadata,
        )


def build_compiled_exact_manifold_adapter(
    executor: CompiledAnsatzExecutor,
    layout: AnsatzParameterLayout,
    theta_runtime: np.ndarray | Sequence[float],
    psi_ref: np.ndarray | Sequence[complex],
    h_compiled: CompiledPolynomialAction,
    *,
    manifold_id: str = "compiled_ansatz_exact_manifold_v1",
) -> CompiledExactManifoldAdapter:
    """Construct the exact compiled-state manifold adapter."""

    return CompiledExactManifoldAdapter(
        executor=executor,
        layout=layout,
        theta_runtime=theta_runtime,
        psi_ref=psi_ref,
        h_compiled=h_compiled,
        manifold_id=manifold_id,
    )


__all__ = [
    "COORDINATE_REGISTRY_SCHEMA",
    "CompiledExactManifoldAdapter",
    "EXACT_BACKEND_SCHEMA",
    "EXACT_STATE_PROVENANCE",
    "build_compiled_exact_manifold_adapter",
]
