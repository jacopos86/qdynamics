"""Neutral exact-state adapter for compiled ansatz geometry.

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
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.static_adapt.exact_state_backend import (
    ExactEnergyEvaluation,
    ExactGradientEvaluation,
    ExactStateBackend,
    ExactStateEvaluation,
)
from pipelines.static_adapt.geometry_fingerprints import (
    candidate_generator_fingerprint,
    compiled_hamiltonian_fingerprint,
    ordered_scaffold_fingerprint,
)
from src.quantum.ansatz_parameterization import (
    AnsatzParameterLayout,
    build_parameter_layout,
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
COORDINATE_REGISTRY_OVERRIDE_SCHEMA = (
    "formal_manifold_coordinate_registry_override_v1"
)
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


@dataclass(frozen=True)
class CoordinateRegistryOverride:
    """Explicit insertion-stable identities for one FM executor chart.

    Duplicate order is not coordinate identity.  The caller supplies the
    inherited-coordinate injection certified by the accepted growth receipt;
    only genuinely admitted positions receive newly allocated deterministic
    ids.  The immutable payload can therefore be previewed by a speculative
    beam child and discarded on rollback without mutating its parent.
    """

    parameterization_mode: str
    layout_fingerprint_sha256: str
    coordinate_ids: tuple[str, ...]
    inherited_coordinate_ids: tuple[str, ...]
    old_to_new_registry_mapping: tuple[int, ...]
    admitted_coordinate_positions: tuple[int, ...]
    parent_generator_fingerprints: tuple[str, ...]
    current_generator_fingerprints: tuple[str, ...]
    old_to_new_generator_mapping: tuple[int, ...]
    parent_registry_fingerprint_sha256: str
    admission_context_fingerprint_sha256: str
    allocation_records: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        mode = str(self.parameterization_mode)
        if mode not in {"logical_shared", "per_pauli_term"}:
            raise ValueError("unsupported coordinate-registry override mode.")
        coordinate_ids = tuple(str(value) for value in self.coordinate_ids)
        inherited_ids = tuple(
            str(value) for value in self.inherited_coordinate_ids
        )
        mapping = tuple(int(value) for value in self.old_to_new_registry_mapping)
        admitted = tuple(int(value) for value in self.admitted_coordinate_positions)
        generator_mapping = tuple(
            int(value) for value in self.old_to_new_generator_mapping
        )
        if len(set(coordinate_ids)) != len(coordinate_ids):
            raise ValueError("coordinate-registry override ids must be unique.")
        if len(mapping) != len(inherited_ids):
            raise ValueError(
                "coordinate-registry override inherited ids and mapping disagree."
            )
        if len(set(mapping)) != len(mapping):
            raise ValueError("old-to-new coordinate mapping must be injective.")
        if tuple(sorted(mapping)) != mapping:
            raise ValueError(
                "inherited coordinate positions must preserve gate order."
            )
        all_positions = set(range(len(coordinate_ids)))
        if not set(mapping).issubset(all_positions):
            raise ValueError("old-to-new coordinate mapping is out of range.")
        if set(admitted) != all_positions.difference(mapping):
            raise ValueError(
                "admitted positions must be the complement of inherited positions."
            )
        for old_index, new_index in enumerate(mapping):
            if coordinate_ids[new_index] != inherited_ids[old_index]:
                raise ValueError(
                    "coordinate-registry override renamed an inherited instance."
                )
        if len(self.allocation_records) != len(admitted):
            raise ValueError(
                "coordinate-registry override allocation records are incomplete."
            )
        if len(generator_mapping) != len(self.parent_generator_fingerprints):
            raise ValueError(
                "parent generators require one old-to-new generator position each."
            )
        if tuple(sorted(generator_mapping)) != generator_mapping or len(
            set(generator_mapping)
        ) != len(generator_mapping):
            raise ValueError(
                "inherited generator positions must be a strictly increasing map."
            )
        if any(
            position < 0 or position >= len(self.current_generator_fingerprints)
            for position in generator_mapping
        ):
            raise ValueError("old-to-new generator mapping is out of range.")
        for old_index, new_index in enumerate(generator_mapping):
            if str(self.parent_generator_fingerprints[old_index]) != str(
                self.current_generator_fingerprints[new_index]
            ):
                raise ValueError(
                    "mapped inherited generator does not match the accepted parent."
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": COORDINATE_REGISTRY_OVERRIDE_SCHEMA,
            "parameterization_mode": str(self.parameterization_mode),
            "layout_fingerprint_sha256": str(self.layout_fingerprint_sha256),
            "coordinate_ids": list(self.coordinate_ids),
            "coordinate_registry_fingerprint_sha256": _json_sha256(
                list(self.coordinate_ids)
            ),
            "inherited_coordinate_ids": list(self.inherited_coordinate_ids),
            "old_to_new_registry_mapping": list(
                self.old_to_new_registry_mapping
            ),
            "admitted_coordinate_positions": list(
                self.admitted_coordinate_positions
            ),
            "parent_generator_fingerprints": list(
                self.parent_generator_fingerprints
            ),
            "current_generator_fingerprints": list(
                self.current_generator_fingerprints
            ),
            "old_to_new_generator_mapping": list(
                self.old_to_new_generator_mapping
            ),
            "parent_registry_fingerprint_sha256": str(
                self.parent_registry_fingerprint_sha256
            ),
            "admission_context_fingerprint_sha256": str(
                self.admission_context_fingerprint_sha256
            ),
            "allocation_records": [
                deepcopy(dict(record)) for record in self.allocation_records
            ],
        }

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> "CoordinateRegistryOverride":
        data = deepcopy(dict(payload))
        if data.get("schema") != COORDINATE_REGISTRY_OVERRIDE_SCHEMA:
            raise ValueError("unsupported coordinate-registry override schema.")
        override = cls(
            parameterization_mode=str(data.get("parameterization_mode", "")),
            layout_fingerprint_sha256=str(
                data.get("layout_fingerprint_sha256", "")
            ),
            coordinate_ids=tuple(
                str(value) for value in data.get("coordinate_ids", ())
            ),
            inherited_coordinate_ids=tuple(
                str(value)
                for value in data.get("inherited_coordinate_ids", ())
            ),
            old_to_new_registry_mapping=tuple(
                int(value)
                for value in data.get("old_to_new_registry_mapping", ())
            ),
            admitted_coordinate_positions=tuple(
                int(value)
                for value in data.get("admitted_coordinate_positions", ())
            ),
            parent_generator_fingerprints=tuple(
                str(value)
                for value in data.get("parent_generator_fingerprints", ())
            ),
            current_generator_fingerprints=tuple(
                str(value)
                for value in data.get("current_generator_fingerprints", ())
            ),
            old_to_new_generator_mapping=tuple(
                int(value)
                for value in data.get("old_to_new_generator_mapping", ())
            ),
            parent_registry_fingerprint_sha256=str(
                data.get("parent_registry_fingerprint_sha256", "")
            ),
            admission_context_fingerprint_sha256=str(
                data.get("admission_context_fingerprint_sha256", "")
            ),
            allocation_records=tuple(
                deepcopy(dict(record))
                for record in data.get("allocation_records", ())
            ),
        )
        if str(data.get("coordinate_registry_fingerprint_sha256", "")) != str(
            _json_sha256(list(override.coordinate_ids))
        ):
            raise ValueError(
                "coordinate-registry override fingerprint mismatch."
            )
        if override.parent_registry_fingerprint_sha256 != _json_sha256(
            list(override.inherited_coordinate_ids)
        ):
            raise ValueError(
                "coordinate-registry override parent fingerprint mismatch."
            )
        return override


def build_coordinate_registry_override(
    layout: AnsatzParameterLayout,
    *,
    parameterization_mode: str,
    inherited_coordinate_ids: Sequence[str] = (),
    old_to_new_registry_mapping: Sequence[int] = (),
    parent_generator_fingerprints: Sequence[str] = (),
    current_generator_fingerprints: Sequence[str] = (),
    old_to_new_generator_mapping: Sequence[int] = (),
    admission_context: Mapping[str, Any] | str | None = None,
) -> CoordinateRegistryOverride:
    """Allocate stable ids from an explicit inherited-coordinate injection."""

    mode = str(parameterization_mode)
    default_ids, records = _coordinate_registry_payload(
        layout, parameterization_mode=mode
    )
    coordinate_count = len(default_ids)
    inherited = tuple(str(value) for value in inherited_coordinate_ids)
    mapping = tuple(int(value) for value in old_to_new_registry_mapping)
    if len(mapping) != len(inherited):
        raise ValueError(
            "inherited coordinate ids require one old-to-new position each."
        )
    if len(set(inherited)) != len(inherited):
        raise ValueError("inherited coordinate ids must be unique.")
    if len(set(mapping)) != len(mapping) or any(
        position < 0 or position >= coordinate_count for position in mapping
    ):
        raise ValueError("old-to-new coordinate mapping is not a valid injection.")
    if tuple(sorted(mapping)) != mapping:
        raise ValueError(
            "old-to-new coordinate mapping must preserve inherited gate order."
        )
    parent_generators = tuple(
        str(value) for value in parent_generator_fingerprints
    )
    current_generators = tuple(
        str(value) for value in current_generator_fingerprints
    )
    generator_mapping = tuple(int(value) for value in old_to_new_generator_mapping)
    if parent_generators or current_generators or generator_mapping:
        if len(generator_mapping) != len(parent_generators):
            raise ValueError(
                "parent generator fingerprints require an explicit mapping."
            )
        if tuple(sorted(generator_mapping)) != generator_mapping or len(
            set(generator_mapping)
        ) != len(generator_mapping):
            raise ValueError(
                "old-to-new generator mapping must be strictly increasing."
            )
        if any(
            position < 0 or position >= len(current_generators)
            for position in generator_mapping
        ):
            raise ValueError("old-to-new generator mapping is out of range.")
        for old_index, new_index in enumerate(generator_mapping):
            if parent_generators[old_index] != current_generators[new_index]:
                raise ValueError(
                    "mapped inherited generator fingerprint changed across growth."
                )
    admitted_positions = tuple(
        position
        for position in range(coordinate_count)
        if position not in set(mapping)
    )
    parent_fingerprint = _json_sha256(list(inherited))
    if admission_context is None:
        context_payload: Any = {
            "layout_fingerprint_sha256": _json_sha256(serialize_layout(layout)),
            "old_to_new_registry_mapping": list(mapping),
        }
    elif isinstance(admission_context, Mapping):
        context_payload = deepcopy(dict(admission_context))
    else:
        context_payload = str(admission_context)
    context_fingerprint = _json_sha256(context_payload)

    coordinate_ids: list[str | None] = [None] * coordinate_count
    for old_index, new_index in enumerate(mapping):
        coordinate_ids[new_index] = inherited[old_index]
    allocations: list[dict[str, Any]] = []
    for admission_ordinal, new_index in enumerate(admitted_positions):
        record = records[new_index]
        local_identity = {
            "block_fingerprint_sha256": str(
                record["block_fingerprint_sha256"]
            ),
            "gate_fingerprint_sha256": record.get(
                "gate_fingerprint_sha256"
            ),
            "runtime_occurrence_index": record.get(
                "runtime_occurrence_index"
            ),
        }
        instance_fingerprint = _json_sha256(
            {
                "schema": COORDINATE_REGISTRY_OVERRIDE_SCHEMA,
                "parameterization_mode": mode,
                "parent_registry_fingerprint_sha256": parent_fingerprint,
                "admission_context_fingerprint_sha256": context_fingerprint,
                "admission_ordinal": int(admission_ordinal),
                "new_coordinate_position": int(new_index),
                "local_identity": local_identity,
            }
        )
        prefix = "logical" if mode == "logical_shared" else "runtime"
        coordinate_id = f"{prefix}:instance:{instance_fingerprint}"
        if coordinate_id in inherited or coordinate_id in coordinate_ids:
            raise ValueError("deterministic coordinate-instance id collision.")
        coordinate_ids[new_index] = coordinate_id
        allocations.append(
            {
                "coordinate_id": coordinate_id,
                "coordinate_position": int(new_index),
                "admission_ordinal": int(admission_ordinal),
                **local_identity,
            }
        )
    resolved_ids = tuple(str(value) for value in coordinate_ids)
    return CoordinateRegistryOverride(
        parameterization_mode=mode,
        layout_fingerprint_sha256=_json_sha256(serialize_layout(layout)),
        coordinate_ids=resolved_ids,
        inherited_coordinate_ids=inherited,
        old_to_new_registry_mapping=mapping,
        admitted_coordinate_positions=admitted_positions,
        parent_generator_fingerprints=parent_generators,
        current_generator_fingerprints=current_generators,
        old_to_new_generator_mapping=generator_mapping,
        parent_registry_fingerprint_sha256=parent_fingerprint,
        admission_context_fingerprint_sha256=context_fingerprint,
        allocation_records=tuple(allocations),
    )


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
        hamiltonian_fingerprint: str | None = None,
        branch_id: str = "single_frontier:0",
        coordinate_registry_override: CoordinateRegistryOverride | None = None,
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
        if not str(branch_id).strip():
            raise ValueError("branch_id must be a non-empty string.")
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
        coordinate_override_payload: dict[str, Any] | None = None
        if coordinate_registry_override is not None:
            if not isinstance(
                coordinate_registry_override, CoordinateRegistryOverride
            ):
                raise TypeError(
                    "coordinate_registry_override must be a "
                    "CoordinateRegistryOverride."
                )
            if str(coordinate_registry_override.parameterization_mode) != mode:
                raise ValueError(
                    "coordinate-registry override parameterization mode mismatch."
                )
            if str(coordinate_registry_override.layout_fingerprint_sha256) != str(
                _json_sha256(layout_payload)
            ):
                raise ValueError(
                    "coordinate-registry override layout fingerprint mismatch."
                )
            if len(coordinate_registry_override.coordinate_ids) != len(
                coordinate_ids
            ):
                raise ValueError(
                    "coordinate-registry override coordinate count mismatch."
                )
            coordinate_ids = tuple(coordinate_registry_override.coordinate_ids)
            inherited_positions = set(
                coordinate_registry_override.old_to_new_registry_mapping
            )
            for index, coordinate_id in enumerate(coordinate_ids):
                coordinate_records[index]["coordinate_id"] = str(coordinate_id)
                coordinate_records[index]["coordinate_instance_origin"] = (
                    "inherited" if index in inherited_positions else "admitted"
                )
            coordinate_override_payload = coordinate_registry_override.as_dict()
        if len(coordinate_ids) != coordinate_count:
            raise ValueError(
                "coordinate registry length mismatch: "
                f"got {len(coordinate_ids)}, expected {coordinate_count}."
            )
        layout_sha256 = _json_sha256(layout_payload)
        coordinate_registry_sha256 = _json_sha256(list(coordinate_ids))
        coordinate_records_sha256 = _json_sha256(coordinate_records)
        scaffold_sha256 = ordered_scaffold_fingerprint(executor.terms)
        generator_fingerprints = [
            candidate_generator_fingerprint(term) for term in executor.terms
        ]
        if coordinate_registry_override is not None:
            declared_generators = tuple(
                coordinate_registry_override.current_generator_fingerprints
            )
            if declared_generators != tuple(generator_fingerprints):
                raise ValueError(
                    "coordinate-registry override generator sequence disagrees "
                    "with the compiled executor."
                )
        empty_layout_sha256 = _json_sha256(
            serialize_layout(
                build_parameter_layout(
                    (),
                    ignore_identity=bool(executor.ignore_identity),
                    coefficient_tolerance=float(executor.coefficient_tolerance),
                    sort_terms=bool(executor.sort_terms),
                )
            )
        )
        metadata: dict[str, Any] = {
            "coordinate_count": coordinate_count,
            "coordinate_identity_position_independent": True,
            "coordinate_identity_repeat_policy": (
                "explicit_inherited_mapping_hash_allocated_v1"
                if coordinate_override_payload is not None
                else "block_fingerprint_then_duplicate_instance_ordinal"
            ),
            "coordinate_registry_override": deepcopy(
                coordinate_override_payload
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
            "ordered_scaffold_fingerprint": scaffold_sha256,
            "candidate_generator_fingerprints": generator_fingerprints,
            "empty_parent_ordered_scaffold_fingerprint": (
                ordered_scaffold_fingerprint(())
            ),
            "empty_parent_parameterization_tie_map_fingerprint": (
                empty_layout_sha256
            ),
            "parameterization_tie_map_fingerprint": layout_sha256,
            "provider_backend_id": "compiled_exact_state_v1",
            "estimator_precision_contract": "analytic_exact_float64_v1",
            "branch_id": str(branch_id),
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
        if hamiltonian_fingerprint is not None:
            resolved_hamiltonian_fingerprint = str(
                hamiltonian_fingerprint
            ).strip()
            if not resolved_hamiltonian_fingerprint:
                raise ValueError(
                    "hamiltonian_fingerprint must be nonempty when supplied."
                )
            metadata["hamiltonian_fingerprint"] = (
                resolved_hamiltonian_fingerprint
            )
        else:
            metadata["hamiltonian_fingerprint"] = (
                compiled_hamiltonian_fingerprint(h_compiled)
            )

        self.executor = executor
        self.layout = layout
        self.h_compiled = h_compiled
        self.manifold_id = str(manifold_id)
        self.branch_id = str(branch_id)
        self.parameterization_mode = mode
        self._coordinate_count = coordinate_count
        self._coordinate_records = deepcopy(coordinate_records)
        self._coordinate_registry = coordinate_ids
        self._coordinate_registry_override = coordinate_registry_override
        self._layout_sha256 = layout_sha256
        self._metadata = deepcopy(metadata)
        self._psi_ref = psi_reference
        self._theta_runtime0 = theta_runtime_array
        self._x0 = np.asarray(x0, dtype=float).reshape(-1).copy()
        self.backend = ExactStateBackend(
            evaluate_fn=self._evaluate,
            energy_fn=self._evaluate_energy,
            gradient_fn=self._evaluate_gradient,
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
    def coordinate_registry_override(self) -> CoordinateRegistryOverride | None:
        """Return immutable coordinate-instance provenance, when supplied."""

        return self._coordinate_registry_override

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

    def _coordinate(self, x: np.ndarray) -> np.ndarray:
        return _finite_real_vector(
            x,
            expected_size=self._coordinate_count,
            name="theta",
        )

    def _evaluation_metadata(self, *, primitive: str) -> dict[str, Any]:
        return {
            "coordinate_registry_sha256": self._metadata[
                "coordinate_registry_sha256"
            ],
            "energy_method": "energy_via_one_apply",
            "energy_provenance": EXACT_STATE_PROVENANCE,
            "finite_differences_used": False,
            "hamiltonian_apply_count": 1,
            "layout_sha256": self._layout_sha256,
            "statevector_provenance": EXACT_STATE_PROVENANCE,
            "typed_estimator_primitive": str(primitive),
        }

    def _evaluate_energy(self, x: np.ndarray) -> ExactEnergyEvaluation:
        coordinate = self._coordinate(x)
        psi = np.asarray(
            self.executor.prepare_state(coordinate, self._psi_ref), dtype=complex
        ).reshape(-1)
        energy, _hpsi = energy_via_one_apply(psi, self.h_compiled)
        return ExactEnergyEvaluation(
            energy=float(energy),
            statevector=psi,
            metadata=self._evaluation_metadata(primitive="energy"),
        )

    def _state_gradient_tangents(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        coordinate = self._coordinate(x)
        coordinate = _finite_real_vector(
            coordinate,
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
        return psi, gradient, tangents, float(energy)

    def _evaluate_gradient(self, x: np.ndarray) -> ExactGradientEvaluation:
        psi, gradient, _tangents, energy = self._state_gradient_tangents(x)
        metadata = {
            **self._evaluation_metadata(primitive="coordinate_gradient"),
            "derivative_identity": "2*Re(<dpsi_i|Hpsi>)",
            "derivative_method": "analytic_compiled_parameter_tangents_v1",
            "gradient_provenance": EXACT_STATE_PROVENANCE,
            "full_metric_formed": False,
        }
        return ExactGradientEvaluation(
            energy=float(energy),
            gradient=gradient,
            statevector=psi,
            metadata=metadata,
        )

    def _evaluate(self, x: np.ndarray) -> ExactStateEvaluation:
        psi, gradient, tangents, energy = self._state_gradient_tangents(x)
        evaluation_metadata: Mapping[str, Any] = {
            **self._evaluation_metadata(primitive="tangent_or_metric"),
            "derivative_identity": "2*Re(<dpsi_i|Hpsi>)",
            "derivative_method": "analytic_compiled_parameter_tangents_v1",
            "gradient_provenance": EXACT_STATE_PROVENANCE,
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
    hamiltonian_fingerprint: str | None = None,
    branch_id: str = "single_frontier:0",
    coordinate_registry_override: CoordinateRegistryOverride | None = None,
) -> CompiledExactManifoldAdapter:
    """Construct the exact compiled-state manifold adapter."""

    return CompiledExactManifoldAdapter(
        executor=executor,
        layout=layout,
        theta_runtime=theta_runtime,
        psi_ref=psi_ref,
        h_compiled=h_compiled,
        manifold_id=manifold_id,
        hamiltonian_fingerprint=hamiltonian_fingerprint,
        branch_id=branch_id,
        coordinate_registry_override=coordinate_registry_override,
    )


__all__ = [
    "COORDINATE_REGISTRY_SCHEMA",
    "COORDINATE_REGISTRY_OVERRIDE_SCHEMA",
    "CompiledExactManifoldAdapter",
    "CoordinateRegistryOverride",
    "EXACT_BACKEND_SCHEMA",
    "EXACT_STATE_PROVENANCE",
    "build_coordinate_registry_override",
    "build_compiled_exact_manifold_adapter",
]
