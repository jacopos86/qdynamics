"""Typed, query-closed selector geometry for Formal-Manifold SNAKE.

This module is deliberately route-neutral numerical/data infrastructure.  It
does not call an estimator.  Provider responses enter as immutable primitive
identities and receipts; every contraction, factorization, Schur operation,
and subset solve performed here has zero quantum-query charge.

The ordinary coordinate Hessian used by Phase II is intentionally a different
type from the inverse raised-curvature prior used by the manifold optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import itertools
import json
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from pipelines.static_adapt.joint_linear_solve import (
    JointLinearSolveConfig,
    JointLinearSolveResult,
    factor_supported_metric,
    solve_joint_linear_model,
)


QUERY_RECEIPT_SCHEMA = "formal_manifold_query_receipt_v1"
QUERY_CLOSURE_WORKSPACE_SCHEMA = "formal_manifold_query_closed_workspace_v1"
FORMAL_GROWTH_RECEIPT_SCHEMA = "formal_manifold_growth_geometry_receipt_v1"

PRIMITIVE_KINDS = frozenset(
    {
        "energy",
        "coordinate_gradient",
        "tangent_or_metric",
        "coordinate_second_derivative",
        "hessian_vector",
        "cross_state_tangent",
    }
)
LEDGER_CATEGORY_BY_KIND = {
    "energy": "N_E",
    "coordinate_gradient": "N_grad",
    "tangent_or_metric": "N_G",
    "coordinate_second_derivative": "N_Q",
    "hessian_vector": "N_Hv",
    "cross_state_tangent": "N_cross",
}

CAPABILITY_LIVE_TANGENT = "round_local_live_tangent_handle"
CAPABILITY_ACTIVE_CANDIDATE_GRAM = "active_candidate_gram_available"
CAPABILITY_COMMON_TANGENT_CONTRACTION = "common_state_tangent_contraction"
ORDINARY_HESSIAN_PROVENANCE = "ordinary_coordinate_hessian"
OPTIMIZER_INVERSE_CURVATURE_PROVENANCE = (
    "transported_or_regularized_inverse_raised_curvature"
)
OPTIMIZER_MIXED_BLOCK_STATUS = "unknown_prior_zero"


def _require_text(name: str, value: str, *, allow_empty: bool = False) -> str:
    resolved = str(value)
    if not allow_empty and not resolved:
        raise ValueError(f"{name} must be nonempty.")
    return resolved


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest_payload(schema: str, payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(str(schema).encode("utf-8"))
    digest.update(b"\0")
    digest.update(_canonical_json(payload).encode("utf-8"))
    return digest.hexdigest()


def projective_state_fingerprint(value: Any) -> str:
    """Stable identity for a pure state modulo its unobservable global phase."""

    state = np.asarray(value, dtype=np.complex128).reshape(-1)
    if not (
        np.all(np.isfinite(np.real(state)))
        and np.all(np.isfinite(np.imag(state)))
    ):
        raise ValueError("state fingerprint input must be finite.")
    if state.size:
        pivot = int(np.argmax(np.abs(state)))
        pivot_value = complex(state[pivot])
        if abs(pivot_value) > 0.0:
            state = state * (np.conjugate(pivot_value) / abs(pivot_value))
    # Phase cancellation may differ by a few ulps for algebraically identical
    # inputs.  Quantization belongs only to provenance identity, never to the
    # numerical state used for contractions.
    real = np.round(np.real(state), decimals=14)
    imag = np.round(np.imag(state), decimals=14)
    real[np.abs(real) == 0.0] = 0.0
    imag[np.abs(imag) == 0.0] = 0.0
    return _digest_payload(
        "projective_pure_state_fingerprint_v1",
        {
            "shape": list(state.shape),
            "real": real.tolist(),
            "imag": imag.tolist(),
        },
    )


def _readonly_array(
    value: Any,
    *,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=float).copy()
    if ndim is not None and array.ndim != int(ndim):
        raise ValueError(f"array must have ndim={ndim}, got {array.ndim}.")
    if shape is not None and array.shape != shape:
        raise ValueError(f"array must have shape {shape}, got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("array contains nonfinite values.")
    array.setflags(write=False)
    return array


def _readonly_symmetric(value: Any, *, dimension: int) -> np.ndarray:
    array = _readonly_array(value, ndim=2, shape=(dimension, dimension))
    result = np.asarray(0.5 * (array + array.T), dtype=float)
    result.setflags(write=False)
    return result


def _sorted_unique_text(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({str(value) for value in values}))


def _normalized_pairs(
    values: Mapping[str, str] | Iterable[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    rows = values.items() if isinstance(values, Mapping) else values
    return tuple(sorted((str(key), str(value)) for key, value in rows))


def _receipt_ids(receipts: Iterable["QueryReceipt"]) -> frozenset[str]:
    return frozenset(
        primitive_id
        for receipt in receipts
        for primitive_id in receipt.all_primitive_ids
    )


def _real_tangent_inner_product(left: Any, right: Any) -> float:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        raise ValueError("tangent handles have incompatible shapes.")
    value = float(np.real(np.vdot(left_array.reshape(-1), right_array.reshape(-1))))
    if not math.isfinite(value):
        raise ValueError("tangent contraction is nonfinite.")
    return value


@dataclass(frozen=True)
class EstimatorPrimitiveIdentity:
    """Immutable logical estimator identity; equality is identity, not value."""

    primitive_kind: str
    physical_state_fingerprint: str
    branch_id: str
    ordered_scaffold_fingerprint: str
    theta_fingerprint: str
    coordinate_registry_fingerprint: str
    candidate_generator_fingerprint: str
    candidate_insertion_position: int | None
    parameterization_tie_map_fingerprint: str
    hamiltonian_fingerprint: str
    provider_backend_id: str
    estimator_precision_contract: str
    formula_primitive_identity: str

    def __post_init__(self) -> None:
        if str(self.primitive_kind) not in PRIMITIVE_KINDS:
            raise ValueError(f"primitive_kind must be one of {sorted(PRIMITIVE_KINDS)}.")
        for name in (
            "physical_state_fingerprint",
            "branch_id",
            "ordered_scaffold_fingerprint",
            "theta_fingerprint",
            "coordinate_registry_fingerprint",
            "parameterization_tie_map_fingerprint",
            "hamiltonian_fingerprint",
            "provider_backend_id",
            "estimator_precision_contract",
            "formula_primitive_identity",
        ):
            _require_text(name, getattr(self, name))
        _require_text(
            "candidate_generator_fingerprint",
            self.candidate_generator_fingerprint,
            allow_empty=True,
        )
        if self.candidate_insertion_position is not None and int(
            self.candidate_insertion_position
        ) < 0:
            raise ValueError("candidate_insertion_position must be nonnegative.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "primitive_kind": str(self.primitive_kind),
            "physical_state_fingerprint": str(self.physical_state_fingerprint),
            "branch_id": str(self.branch_id),
            "ordered_scaffold_fingerprint": str(
                self.ordered_scaffold_fingerprint
            ),
            "theta_fingerprint": str(self.theta_fingerprint),
            "coordinate_registry_fingerprint": str(
                self.coordinate_registry_fingerprint
            ),
            "candidate_generator_fingerprint": str(
                self.candidate_generator_fingerprint
            ),
            "candidate_insertion_position": (
                None
                if self.candidate_insertion_position is None
                else int(self.candidate_insertion_position)
            ),
            "parameterization_tie_map_fingerprint": str(
                self.parameterization_tie_map_fingerprint
            ),
            "hamiltonian_fingerprint": str(self.hamiltonian_fingerprint),
            "provider_backend_id": str(self.provider_backend_id),
            "estimator_precision_contract": str(
                self.estimator_precision_contract
            ),
            "formula_primitive_identity": str(self.formula_primitive_identity),
        }

    @property
    def primitive_id(self) -> str:
        return _digest_payload("logical_estimator_primitive_identity_v1", self.as_dict())


@dataclass(frozen=True)
class QueryReceipt:
    """Portable provider receipt keyed by logical estimator primitive IDs."""

    primitive_ids_requested: tuple[str, ...]
    primitive_ids_reused: tuple[str, ...]
    returned_fields: tuple[str, ...]
    closure_capabilities: tuple[str, ...]
    provenance_by_field: tuple[tuple[str, str], ...]
    primitive_kind_by_id: tuple[tuple[str, str], ...]
    provider_kind: str
    statevector_shortcut_used: bool = False
    schema: str = QUERY_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "primitive_ids_requested", _sorted_unique_text(self.primitive_ids_requested)
        )
        object.__setattr__(
            self, "primitive_ids_reused", _sorted_unique_text(self.primitive_ids_reused)
        )
        object.__setattr__(self, "returned_fields", _sorted_unique_text(self.returned_fields))
        object.__setattr__(
            self, "closure_capabilities", _sorted_unique_text(self.closure_capabilities)
        )
        object.__setattr__(
            self, "provenance_by_field", _normalized_pairs(self.provenance_by_field)
        )
        object.__setattr__(
            self, "primitive_kind_by_id", _normalized_pairs(self.primitive_kind_by_id)
        )
        _require_text("provider_kind", self.provider_kind)
        if str(self.schema) != QUERY_RECEIPT_SCHEMA:
            raise ValueError(f"schema must be {QUERY_RECEIPT_SCHEMA!r}.")
        overlap = set(self.primitive_ids_requested) & set(self.primitive_ids_reused)
        if overlap:
            raise ValueError("a primitive cannot be both requested and reused in one receipt.")
        kind_map = dict(self.primitive_kind_by_id)
        all_ids = self.all_primitive_ids
        if set(kind_map) != set(all_ids):
            raise ValueError("primitive_kind_by_id must cover every receipt primitive exactly.")
        invalid = sorted({kind for kind in kind_map.values() if kind not in PRIMITIVE_KINDS})
        if invalid:
            raise ValueError(f"invalid primitive kinds: {invalid}.")

    @classmethod
    def from_primitives(
        cls,
        *,
        requested: Sequence[EstimatorPrimitiveIdentity] = (),
        reused: Sequence[EstimatorPrimitiveIdentity] = (),
        returned_fields: Sequence[str],
        closure_capabilities: Sequence[str] = (),
        provenance_by_field: Mapping[str, str] | Iterable[tuple[str, str]] = (),
        provider_kind: str,
        statevector_shortcut_used: bool = False,
    ) -> "QueryReceipt":
        requested_by_id = {item.primitive_id: item for item in requested}
        reused_by_id = {item.primitive_id: item for item in reused}
        overlap = set(requested_by_id) & set(reused_by_id)
        if overlap:
            raise ValueError("requested and reused primitive identities overlap.")
        catalog = {**requested_by_id, **reused_by_id}
        return cls(
            primitive_ids_requested=tuple(requested_by_id),
            primitive_ids_reused=tuple(reused_by_id),
            returned_fields=tuple(returned_fields),
            closure_capabilities=tuple(closure_capabilities),
            provenance_by_field=_normalized_pairs(provenance_by_field),
            primitive_kind_by_id=tuple(
                (primitive_id, item.primitive_kind)
                for primitive_id, item in catalog.items()
            ),
            provider_kind=str(provider_kind),
            statevector_shortcut_used=bool(statevector_shortcut_used),
        )

    @property
    def all_primitive_ids(self) -> tuple[str, ...]:
        return _sorted_unique_text(
            (*self.primitive_ids_requested, *self.primitive_ids_reused)
        )

    @property
    def provenance(self) -> dict[str, str]:
        return dict(self.provenance_by_field)

    @property
    def kind_map(self) -> dict[str, str]:
        return dict(self.primitive_kind_by_id)

    def portable_payload(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "primitive_ids_requested": list(self.primitive_ids_requested),
            "primitive_ids_reused": list(self.primitive_ids_reused),
            "returned_fields": list(self.returned_fields),
            "closure_capabilities": list(self.closure_capabilities),
            "provenance_by_field": dict(self.provenance_by_field),
            "primitive_kind_by_id": dict(self.primitive_kind_by_id),
            "provider_kind": str(self.provider_kind),
            "statevector_shortcut_used": bool(self.statevector_shortcut_used),
        }


@dataclass(frozen=True)
class SelectorGeometryAnchor:
    state_fingerprint: str
    branch_id: str
    manifold_id: str
    ordered_scaffold_fingerprint: str
    theta_fingerprint: str
    coordinate_registry_fingerprint: str
    parameterization_mode: str
    parameterization_tie_map_fingerprint: str
    hamiltonian_fingerprint: str
    active_coordinate_indices: tuple[int, ...]
    active_tangent_handles: tuple[Any, ...]
    G_AA: np.ndarray
    b_A: np.ndarray
    gram_provenance: str
    differential_provenance: str
    source_query_receipts: tuple[QueryReceipt, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "state_fingerprint",
            "branch_id",
            "manifold_id",
            "ordered_scaffold_fingerprint",
            "theta_fingerprint",
            "coordinate_registry_fingerprint",
            "parameterization_mode",
            "parameterization_tie_map_fingerprint",
            "hamiltonian_fingerprint",
            "gram_provenance",
            "differential_provenance",
        ):
            _require_text(name, getattr(self, name))
        indices = tuple(int(index) for index in self.active_coordinate_indices)
        if len(set(indices)) != len(indices) or any(index < 0 for index in indices):
            raise ValueError("active_coordinate_indices must be unique and nonnegative.")
        object.__setattr__(self, "active_coordinate_indices", indices)
        dimension = len(indices)
        object.__setattr__(self, "G_AA", _readonly_symmetric(self.G_AA, dimension=dimension))
        object.__setattr__(self, "b_A", _readonly_array(self.b_A, ndim=1, shape=(dimension,)))
        handles = tuple(self.active_tangent_handles)
        if handles and len(handles) != dimension:
            raise ValueError("active_tangent_handles must be empty or match active dimension.")
        object.__setattr__(self, "active_tangent_handles", handles)
        object.__setattr__(self, "source_query_receipts", tuple(self.source_query_receipts))

    @property
    def active_dimension(self) -> int:
        return len(self.active_coordinate_indices)

    @property
    def source_primitive_ids(self) -> frozenset[str]:
        return _receipt_ids(self.source_query_receipts)

    def portable_payload(self) -> dict[str, Any]:
        """Return the cache-safe form; live tangent handles are never serialized."""

        return {
            "state_fingerprint": self.state_fingerprint,
            "branch_id": self.branch_id,
            "manifold_id": self.manifold_id,
            "ordered_scaffold_fingerprint": self.ordered_scaffold_fingerprint,
            "theta_fingerprint": self.theta_fingerprint,
            "coordinate_registry_fingerprint": self.coordinate_registry_fingerprint,
            "parameterization_mode": self.parameterization_mode,
            "parameterization_tie_map_fingerprint": self.parameterization_tie_map_fingerprint,
            "hamiltonian_fingerprint": self.hamiltonian_fingerprint,
            "active_coordinate_indices": list(self.active_coordinate_indices),
            "G_AA": self.G_AA.tolist(),
            "b_A": self.b_A.tolist(),
            "gram_provenance": self.gram_provenance,
            "differential_provenance": self.differential_provenance,
            "source_query_receipts": [
                receipt.portable_payload() for receipt in self.source_query_receipts
            ],
        }


@dataclass(frozen=True)
class CandidateTangentRecord:
    candidate_fingerprint: str
    candidate_registry_entry_fingerprint: str
    insertion_position: int
    state_fingerprint: str
    branch_id: str
    manifold_id: str
    ordered_scaffold_fingerprint: str
    theta_fingerprint: str
    coordinate_registry_fingerprint: str
    parameterization_mode: str
    parameterization_tie_map_fingerprint: str
    hamiltonian_fingerprint: str
    tangent_handle: Any = field(compare=False, repr=False)
    b_B: float = 0.0
    G_AB: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    G_BB: float = 0.0
    query_receipts: tuple[QueryReceipt, ...] = ()
    closure_capabilities: tuple[str, ...] = ()
    provenance_by_block: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "candidate_fingerprint",
            "candidate_registry_entry_fingerprint",
            "state_fingerprint",
            "branch_id",
            "manifold_id",
            "ordered_scaffold_fingerprint",
            "theta_fingerprint",
            "coordinate_registry_fingerprint",
            "parameterization_mode",
            "parameterization_tie_map_fingerprint",
            "hamiltonian_fingerprint",
        ):
            _require_text(name, getattr(self, name))
        if int(self.insertion_position) < 0:
            raise ValueError("insertion_position must be nonnegative.")
        object.__setattr__(self, "insertion_position", int(self.insertion_position))
        object.__setattr__(self, "G_AB", _readonly_array(self.G_AB, ndim=1))
        for name in ("b_B", "G_BB"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "query_receipts", tuple(self.query_receipts))
        object.__setattr__(
            self, "closure_capabilities", _sorted_unique_text(self.closure_capabilities)
        )
        object.__setattr__(
            self, "provenance_by_block", _normalized_pairs(self.provenance_by_block)
        )

    @property
    def candidate_key(self) -> str:
        return _digest_payload(
            "candidate_position_identity_v1",
            {
                "candidate_fingerprint": self.candidate_fingerprint,
                "candidate_registry_entry_fingerprint": (
                    self.candidate_registry_entry_fingerprint
                ),
                "insertion_position": self.insertion_position,
                "parameterization_tie_map_fingerprint": (
                    self.parameterization_tie_map_fingerprint
                ),
            },
        )

    @property
    def source_primitive_ids(self) -> frozenset[str]:
        return _receipt_ids(self.query_receipts)

    @property
    def supports_active_cross_gram(self) -> bool:
        return bool(
            self.tangent_handle is not None
            or CAPABILITY_ACTIVE_CANDIDATE_GRAM in self.closure_capabilities
        )

    def compatibility_mismatches(self, anchor: SelectorGeometryAnchor) -> tuple[str, ...]:
        fields = (
            "state_fingerprint",
            "branch_id",
            "manifold_id",
            "ordered_scaffold_fingerprint",
            "theta_fingerprint",
            "coordinate_registry_fingerprint",
            "parameterization_mode",
            "parameterization_tie_map_fingerprint",
            "hamiltonian_fingerprint",
        )
        mismatches = [
            field_name
            for field_name in fields
            if getattr(self, field_name) != getattr(anchor, field_name)
        ]
        if self.G_AB.shape != (anchor.active_dimension,):
            mismatches.append("active_candidate_gram_shape")
        return tuple(mismatches)

    def portable_payload(self) -> dict[str, Any]:
        """Bounded payload excluding live provider handles and statevectors."""

        return {
            "candidate_fingerprint": self.candidate_fingerprint,
            "candidate_registry_entry_fingerprint": (
                self.candidate_registry_entry_fingerprint
            ),
            "candidate_key": self.candidate_key,
            "insertion_position": self.insertion_position,
            "state_fingerprint": self.state_fingerprint,
            "branch_id": self.branch_id,
            "manifold_id": self.manifold_id,
            "ordered_scaffold_fingerprint": self.ordered_scaffold_fingerprint,
            "theta_fingerprint": self.theta_fingerprint,
            "coordinate_registry_fingerprint": self.coordinate_registry_fingerprint,
            "parameterization_mode": self.parameterization_mode,
            "parameterization_tie_map_fingerprint": (
                self.parameterization_tie_map_fingerprint
            ),
            "hamiltonian_fingerprint": self.hamiltonian_fingerprint,
            "b_B": self.b_B,
            "G_AB": self.G_AB.tolist(),
            "G_BB": self.G_BB,
            "query_receipts": [receipt.portable_payload() for receipt in self.query_receipts],
            "closure_capabilities": list(self.closure_capabilities),
            "provenance_by_block": dict(self.provenance_by_block),
        }


def build_candidate_tangent_record(
    *,
    anchor: SelectorGeometryAnchor,
    candidate_fingerprint: str,
    candidate_registry_entry_fingerprint: str,
    insertion_position: int,
    tangent_handle: Any,
    differential: float,
    query_receipts: Sequence[QueryReceipt],
    inner_product: Callable[[Any, Any], float] = _real_tangent_inner_product,
    extra_capabilities: Sequence[str] = (),
) -> CandidateTangentRecord:
    """Close candidate Gram blocks by classical contraction of live tangents."""

    if tangent_handle is None:
        raise ValueError("a live tangent_handle is required for query-free closure.")
    if anchor.active_dimension and not anchor.active_tangent_handles:
        raise ValueError("the anchor has no live active tangent frame.")
    G_AB = np.asarray(
        [inner_product(handle, tangent_handle) for handle in anchor.active_tangent_handles],
        dtype=float,
    )
    G_BB = float(inner_product(tangent_handle, tangent_handle))
    return CandidateTangentRecord(
        candidate_fingerprint=str(candidate_fingerprint),
        candidate_registry_entry_fingerprint=str(
            candidate_registry_entry_fingerprint
        ),
        insertion_position=int(insertion_position),
        state_fingerprint=anchor.state_fingerprint,
        branch_id=anchor.branch_id,
        manifold_id=anchor.manifold_id,
        ordered_scaffold_fingerprint=anchor.ordered_scaffold_fingerprint,
        theta_fingerprint=anchor.theta_fingerprint,
        coordinate_registry_fingerprint=anchor.coordinate_registry_fingerprint,
        parameterization_mode=anchor.parameterization_mode,
        parameterization_tie_map_fingerprint=(
            anchor.parameterization_tie_map_fingerprint
        ),
        hamiltonian_fingerprint=anchor.hamiltonian_fingerprint,
        tangent_handle=tangent_handle,
        b_B=float(differential),
        G_AB=G_AB,
        G_BB=G_BB,
        query_receipts=tuple(query_receipts),
        closure_capabilities=(
            CAPABILITY_LIVE_TANGENT,
            CAPABILITY_ACTIVE_CANDIDATE_GRAM,
            CAPABILITY_COMMON_TANGENT_CONTRACTION,
            *tuple(extra_capabilities),
        ),
        provenance_by_block=(
            ("b_B", "provider_coordinate_differential"),
            ("G_AB", "query_free_live_tangent_contraction"),
            ("G_BB", "query_free_live_tangent_contraction"),
        ),
    )


@dataclass
class QueryClosedPopulationWorkspace:
    """State/branch-scoped population geometry and its query provenance."""

    anchor: SelectorGeometryAnchor
    candidate_records: tuple[CandidateTangentRecord, ...]
    G_AC: np.ndarray
    G_CC: np.ndarray
    b_C: np.ndarray
    missing_primitive_requests: tuple[EstimatorPrimitiveIdentity, ...]
    source_query_receipts: tuple[QueryReceipt, ...]
    query_free_derived_fields: tuple[str, ...]
    complete_pair_gram: bool
    schema: str = QUERY_CLOSURE_WORKSPACE_SCHEMA
    derived_feature_cache: dict[Any, Any] = field(default_factory=dict, repr=False)
    subset_solve_cache: dict[Any, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        records = tuple(self.candidate_records)
        keys = [record.candidate_key for record in records]
        if len(set(keys)) != len(keys):
            raise ValueError("candidate-position identities must be unique.")
        for record in records:
            mismatches = record.compatibility_mismatches(self.anchor)
            if mismatches:
                raise ValueError(
                    "candidate record is incompatible with anchor: "
                    + ", ".join(mismatches)
                )
        active = self.anchor.active_dimension
        candidate_count = len(records)
        self.G_AC = _readonly_array(
            self.G_AC, ndim=2, shape=(active, candidate_count)
        )
        raw_G_CC = np.asarray(self.G_CC, dtype=float).copy()
        if raw_G_CC.shape != (candidate_count, candidate_count):
            raise ValueError("G_CC shape does not match candidate population.")
        finite_or_nan = np.isfinite(raw_G_CC) | np.isnan(raw_G_CC)
        if not np.all(finite_or_nan):
            raise ValueError("G_CC contains invalid values.")
        finite_pair = np.isfinite(raw_G_CC) & np.isfinite(raw_G_CC.T)
        if np.any(finite_pair):
            rows, columns = np.where(finite_pair)
            raw_G_CC[rows, columns] = 0.5 * (
                raw_G_CC[rows, columns] + raw_G_CC.T[rows, columns]
            )
        raw_G_CC.setflags(write=False)
        self.G_CC = raw_G_CC
        self.b_C = _readonly_array(
            self.b_C, ndim=1, shape=(candidate_count,)
        )
        self.candidate_records = records
        self.missing_primitive_requests = tuple(self.missing_primitive_requests)
        self.source_query_receipts = tuple(self.source_query_receipts)
        self.query_free_derived_fields = _sorted_unique_text(
            self.query_free_derived_fields
        )
        if str(self.schema) != QUERY_CLOSURE_WORKSPACE_SCHEMA:
            raise ValueError(f"schema must be {QUERY_CLOSURE_WORKSPACE_SCHEMA!r}.")

    @property
    def candidate_keys(self) -> tuple[str, ...]:
        return tuple(record.candidate_key for record in self.candidate_records)

    @property
    def source_primitive_ids(self) -> frozenset[str]:
        return _receipt_ids(self.source_query_receipts)

    @property
    def unique_primitive_id_set(self) -> frozenset[str]:
        return self.source_primitive_ids

    @property
    def workspace_fingerprint(self) -> str:
        return _digest_payload(
            QUERY_CLOSURE_WORKSPACE_SCHEMA,
            {
                "state_fingerprint": self.anchor.state_fingerprint,
                "branch_id": self.anchor.branch_id,
                "manifold_id": self.anchor.manifold_id,
                "ordered_scaffold_fingerprint": (
                    self.anchor.ordered_scaffold_fingerprint
                ),
                "theta_fingerprint": self.anchor.theta_fingerprint,
                "coordinate_registry_fingerprint": (
                    self.anchor.coordinate_registry_fingerprint
                ),
                "parameterization_tie_map_fingerprint": (
                    self.anchor.parameterization_tie_map_fingerprint
                ),
                "hamiltonian_fingerprint": self.anchor.hamiltonian_fingerprint,
                "candidate_keys": list(self.candidate_keys),
                "G_AC": self.G_AC.tolist(),
                "G_CC": self.G_CC.tolist(),
                "b_C": self.b_C.tolist(),
            },
        )

    def subset_geometry(
        self, candidate_indices: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = tuple(int(index) for index in candidate_indices)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("candidate_indices must be nonempty and unique.")
        if min(indices) < 0 or max(indices) >= len(self.candidate_records):
            raise IndexError("candidate index is out of range.")
        G_AB = np.asarray(self.G_AC[:, indices], dtype=float)
        G_BB = np.asarray(self.G_CC[np.ix_(indices, indices)], dtype=float)
        b_B = np.asarray(self.b_C[list(indices)], dtype=float)
        return G_AB, G_BB, b_B


def build_query_closed_population_workspace(
    *,
    anchor: SelectorGeometryAnchor,
    candidate_records: Sequence[CandidateTangentRecord],
    inner_product: Callable[[Any, Any], float] = _real_tangent_inner_product,
    provided_pair_gram: Mapping[tuple[str, str], float] | None = None,
    provided_pair_receipts: Mapping[tuple[str, str], QueryReceipt] | None = None,
    missing_pair_primitive_factory: (
        Callable[
            [CandidateTangentRecord, CandidateTangentRecord],
            EstimatorPrimitiveIdentity,
        ]
        | None
    ) = None,
) -> QueryClosedPopulationWorkspace:
    """Build one population Gram workspace before combinatorial subset search.

    When both records carry common-state live tangents, every off-diagonal
    candidate Gram entry is a classical contraction and no primitive is added.
    Scalar-only providers can supply measured pair entries or declare the exact
    missing primitive through ``missing_pair_primitive_factory``.
    """

    records = tuple(candidate_records)
    for record in records:
        mismatches = record.compatibility_mismatches(anchor)
        if mismatches:
            raise ValueError(
                "candidate record is incompatible with anchor: "
                + ", ".join(mismatches)
            )
    count = len(records)
    G_AC = (
        np.column_stack([record.G_AB for record in records])
        if records
        else np.zeros((anchor.active_dimension, 0), dtype=float)
    )
    G_CC = np.full((count, count), np.nan, dtype=float)
    for index, record in enumerate(records):
        G_CC[index, index] = float(record.G_BB)
    normalized_pair_values: dict[tuple[str, str], float] = {}
    for pair, value in (provided_pair_gram or {}).items():
        normalized_pair_values[tuple(sorted((str(pair[0]), str(pair[1]))))] = float(
            value
        )
    normalized_pair_receipts: dict[tuple[str, str], QueryReceipt] = {}
    for pair, receipt in (provided_pair_receipts or {}).items():
        normalized_pair_receipts[
            tuple(sorted((str(pair[0]), str(pair[1]))))
        ] = receipt
    source_receipts: list[QueryReceipt] = [*anchor.source_query_receipts]
    source_receipts.extend(
        receipt for record in records for receipt in record.query_receipts
    )
    missing: list[EstimatorPrimitiveIdentity] = []
    pair_contraction_count = 0
    pair_measurement_count = 0
    for left_index, right_index in itertools.combinations(range(count), 2):
        left = records[left_index]
        right = records[right_index]
        pair_key = tuple(sorted((left.candidate_key, right.candidate_key)))
        common_live_capability = bool(
            left.tangent_handle is not None
            and right.tangent_handle is not None
            and CAPABILITY_COMMON_TANGENT_CONTRACTION
            in left.closure_capabilities
            and CAPABILITY_COMMON_TANGENT_CONTRACTION
            in right.closure_capabilities
        )
        if common_live_capability:
            value = float(inner_product(left.tangent_handle, right.tangent_handle))
            pair_contraction_count += 1
        elif pair_key in normalized_pair_values:
            value = float(normalized_pair_values[pair_key])
            receipt = normalized_pair_receipts.get(pair_key)
            if receipt is None:
                raise ValueError("a provided pair Gram entry requires a QueryReceipt.")
            source_receipts.append(receipt)
            pair_measurement_count += 1
        else:
            if missing_pair_primitive_factory is not None:
                missing.append(missing_pair_primitive_factory(left, right))
            continue
        if not math.isfinite(value):
            raise ValueError("candidate-pair Gram contraction is nonfinite.")
        G_CC[left_index, right_index] = value
        G_CC[right_index, left_index] = value
    complete = bool(np.all(np.isfinite(G_CC)))
    derived_fields = [
        "G_AC_from_retained_anchor_and_candidate_records",
        "G_CC_candidate_diagonal_reuse",
    ]
    if pair_contraction_count:
        derived_fields.append("G_CC_live_tangent_pair_contractions")
    return QueryClosedPopulationWorkspace(
        anchor=anchor,
        candidate_records=records,
        G_AC=G_AC,
        G_CC=G_CC,
        b_C=np.asarray([record.b_B for record in records], dtype=float),
        missing_primitive_requests=tuple(missing),
        source_query_receipts=tuple(source_receipts),
        query_free_derived_fields=tuple(derived_fields),
        complete_pair_gram=complete,
        derived_feature_cache={
            "candidate_pair_classical_contraction_count": pair_contraction_count,
            "candidate_pair_measured_entry_count": pair_measurement_count,
        },
    )


@dataclass(frozen=True)
class ResidualizedCandidateBlock:
    S_B: np.ndarray
    residual_differential: np.ndarray
    active_supported_rank: int
    augmented_supported_rank: int
    rank_gain: int
    support_threshold: float
    augmented_spectrum: np.ndarray
    retained_condition_number: float | None

    def __post_init__(self) -> None:
        dimension = int(np.asarray(self.residual_differential).size)
        object.__setattr__(self, "S_B", _readonly_symmetric(self.S_B, dimension=dimension))
        object.__setattr__(
            self,
            "residual_differential",
            _readonly_array(self.residual_differential, ndim=1, shape=(dimension,)),
        )
        object.__setattr__(
            self, "augmented_spectrum", _readonly_array(self.augmented_spectrum, ndim=1)
        )


def residualize_candidate_block(
    *,
    anchor: SelectorGeometryAnchor,
    G_AB: np.ndarray,
    G_BB: np.ndarray,
    b_B: np.ndarray,
    rank_relative_tolerance: float = 1e-6,
    metric_regularization: float = 1e-9,
) -> ResidualizedCandidateBlock:
    """Compute the Gram Schur residual with the shared raw-metric rank rule."""

    active = anchor.active_dimension
    cross = _readonly_array(G_AB, ndim=2)
    if cross.shape[0] != active:
        raise ValueError("G_AB active dimension does not match anchor.")
    candidate_count = int(cross.shape[1])
    candidate_metric = _readonly_symmetric(G_BB, dimension=candidate_count)
    candidate_differential = _readonly_array(
        b_B, ndim=1, shape=(candidate_count,)
    )
    if active:
        active_factor = factor_supported_metric(
            anchor.G_AA,
            rank_relative_tolerance=rank_relative_tolerance,
            metric_regularization=metric_regularization,
        )
        active_pinv = active_factor.raw_metric_pseudoinverse
        active_rank = active_factor.rank
    else:
        active_pinv = np.zeros((0, 0), dtype=float)
        active_rank = 0
    S_B = np.asarray(
        candidate_metric - cross.T @ active_pinv @ cross,
        dtype=float,
    )
    S_B = 0.5 * (S_B + S_B.T)
    residual_differential = np.asarray(
        candidate_differential - cross.T @ active_pinv @ anchor.b_A,
        dtype=float,
    )
    augmented_metric = np.block(
        [[anchor.G_AA, cross], [cross.T, candidate_metric]]
    )
    augmented_factor = factor_supported_metric(
        augmented_metric,
        rank_relative_tolerance=rank_relative_tolerance,
        metric_regularization=metric_regularization,
    )
    return ResidualizedCandidateBlock(
        S_B=S_B,
        residual_differential=residual_differential,
        active_supported_rank=active_rank,
        augmented_supported_rank=augmented_factor.rank,
        rank_gain=int(augmented_factor.rank - active_rank),
        support_threshold=float(augmented_factor.support_threshold),
        augmented_spectrum=augmented_factor.raw_eigenvalues,
        retained_condition_number=augmented_factor.retained_condition_number,
    )


@dataclass(frozen=True)
class Phase1QueryClosedScore:
    feasible: bool
    reason: str
    candidate_key: str
    schur_metric: float
    residual_differential: float
    response: float
    trust_gain: float
    resource_burden: float
    score: float
    rank_gain: int
    support_threshold: float
    novelty_fraction: float
    augmented_spectrum: np.ndarray
    retained_condition_number: float | None
    source_primitive_ids: tuple[str, ...]
    baseline_primitive_ids: tuple[str, ...]
    primitive_set_reconciled: bool | None
    incremental_query_charge: int | None
    query_free_derived_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "augmented_spectrum", _readonly_array(self.augmented_spectrum, ndim=1)
        )


def evaluate_phase1_query_closed_score(
    *,
    anchor: SelectorGeometryAnchor,
    candidate: CandidateTangentRecord,
    trust_radius: float,
    resource_burden: float = 0.0,
    rank_relative_tolerance: float = 1e-6,
    metric_regularization: float = 1e-9,
    baseline_primitive_ids: Iterable[str] | None = None,
) -> Phase1QueryClosedScore:
    """Evaluate the authoritative formal-route first-order trust score."""

    radius = float(trust_radius)
    burden = float(resource_burden)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("trust_radius must be finite and positive.")
    if not math.isfinite(burden) or burden < 0.0:
        raise ValueError("resource_burden must be finite and nonnegative.")
    mismatches = candidate.compatibility_mismatches(anchor)
    source_ids = tuple(sorted(anchor.source_primitive_ids | candidate.source_primitive_ids))
    baseline_ids = (
        () if baseline_primitive_ids is None else tuple(sorted(set(baseline_primitive_ids)))
    )
    reconciliation = (
        None if baseline_primitive_ids is None else set(source_ids) == set(baseline_ids)
    )
    incremental_charge = (
        None
        if baseline_primitive_ids is None
        else len(set(source_ids) - set(baseline_ids))
    )

    def failed(reason: str) -> Phase1QueryClosedScore:
        return Phase1QueryClosedScore(
            feasible=False,
            reason=reason,
            candidate_key=candidate.candidate_key,
            schur_metric=0.0,
            residual_differential=0.0,
            response=0.0,
            trust_gain=0.0,
            resource_burden=burden,
            score=0.0,
            rank_gain=0,
            support_threshold=0.0,
            novelty_fraction=0.0,
            augmented_spectrum=np.zeros(0, dtype=float),
            retained_condition_number=None,
            source_primitive_ids=source_ids,
            baseline_primitive_ids=baseline_ids,
            primitive_set_reconciled=reconciliation,
            incremental_query_charge=incremental_charge,
            query_free_derived_fields=(),
        )

    if mismatches:
        return failed("anchor_candidate_fingerprint_mismatch:" + ",".join(mismatches))
    if not candidate.supports_active_cross_gram:
        return failed("missing_tangent_handle_or_active_candidate_gram")
    block = residualize_candidate_block(
        anchor=anchor,
        G_AB=candidate.G_AB.reshape(anchor.active_dimension, 1),
        G_BB=np.asarray([[candidate.G_BB]], dtype=float),
        b_B=np.asarray([candidate.b_B], dtype=float),
        rank_relative_tolerance=rank_relative_tolerance,
        metric_regularization=metric_regularization,
    )
    schur_metric = float(block.S_B[0, 0])
    residual_differential = float(block.residual_differential[0])
    rank_feasible = bool(
        block.rank_gain == 1 and schur_metric > float(block.support_threshold)
    )
    if not rank_feasible:
        result = failed("candidate_does_not_add_supported_metric_rank")
        return Phase1QueryClosedScore(
            **{
                **result.__dict__,
                "schur_metric": schur_metric,
                "residual_differential": residual_differential,
                "rank_gain": block.rank_gain,
                "support_threshold": block.support_threshold,
                "augmented_spectrum": block.augmented_spectrum,
                "retained_condition_number": block.retained_condition_number,
            }
        )
    denominator = float(schur_metric + metric_regularization)
    if denominator <= 0.0:
        return failed("nonpositive_regularized_schur_metric")
    response = float((residual_differential**2) / denominator)
    trust_gain = float(radius * math.sqrt(max(0.0, response)))
    score = float(trust_gain / (1.0 + burden))
    novelty = float(schur_metric / candidate.G_BB) if candidate.G_BB > 0.0 else 0.0
    return Phase1QueryClosedScore(
        feasible=True,
        reason="query_closed_first_order_metric_trust_response",
        candidate_key=candidate.candidate_key,
        schur_metric=schur_metric,
        residual_differential=residual_differential,
        response=response,
        trust_gain=trust_gain,
        resource_burden=burden,
        score=score,
        rank_gain=block.rank_gain,
        support_threshold=block.support_threshold,
        novelty_fraction=novelty,
        augmented_spectrum=block.augmented_spectrum,
        retained_condition_number=block.retained_condition_number,
        source_primitive_ids=source_ids,
        baseline_primitive_ids=baseline_ids,
        primitive_set_reconciled=reconciliation,
        incremental_query_charge=incremental_charge,
        query_free_derived_fields=(
            "augmented_gram",
            "schur_residual_metric",
            "residual_differential",
            "supported_rank_and_spectrum",
            "query_closed_phase1_response",
            "query_closed_phase1_score",
        ),
    )


@dataclass(frozen=True)
class Phase2OrdinaryHessianBlocks:
    """Candidate-population ordinary coordinate Hessian, never optimizer B."""

    workspace_fingerprint: str
    candidate_keys: tuple[str, ...]
    Q_AA: np.ndarray
    Q_AC: np.ndarray
    Q_CC: np.ndarray
    source_query_receipts: tuple[QueryReceipt, ...]
    provenance_by_block: tuple[tuple[str, str], ...]
    hessian_provenance: str = ORDINARY_HESSIAN_PROVENANCE

    def __post_init__(self) -> None:
        _require_text("workspace_fingerprint", self.workspace_fingerprint)
        keys = tuple(str(key) for key in self.candidate_keys)
        if len(set(keys)) != len(keys):
            raise ValueError("candidate_keys must be unique.")
        object.__setattr__(self, "candidate_keys", keys)
        if str(self.hessian_provenance) != ORDINARY_HESSIAN_PROVENANCE:
            raise ValueError(
                "Phase2OrdinaryHessianBlocks provenance must be "
                f"{ORDINARY_HESSIAN_PROVENANCE!r}."
            )
        Q_AA_raw = np.asarray(self.Q_AA, dtype=float)
        if Q_AA_raw.ndim != 2 or Q_AA_raw.shape[0] != Q_AA_raw.shape[1]:
            raise ValueError("Q_AA must be square.")
        active = int(Q_AA_raw.shape[0])
        candidate_count = len(keys)
        object.__setattr__(self, "Q_AA", _readonly_symmetric(self.Q_AA, dimension=active))
        object.__setattr__(
            self,
            "Q_AC",
            _readonly_array(self.Q_AC, ndim=2, shape=(active, candidate_count)),
        )
        object.__setattr__(self, "Q_CC", _readonly_symmetric(self.Q_CC, dimension=candidate_count))
        object.__setattr__(self, "source_query_receipts", tuple(self.source_query_receipts))
        if not self.source_query_receipts:
            raise ValueError(
                "ordinary Hessian blocks require a coordinate-second-derivative receipt."
            )
        source_kind_map: dict[str, str] = {}
        for receipt in self.source_query_receipts:
            source_kind_map.update(receipt.kind_map)
        if not source_kind_map or not any(
            kind == "coordinate_second_derivative"
            for kind in source_kind_map.values()
        ):
            raise ValueError(
                "ordinary Hessian sources must include a coordinate_second_derivative primitive."
            )
        object.__setattr__(
            self, "provenance_by_block", _normalized_pairs(self.provenance_by_block)
        )
        for block_name in ("Q_AA", "Q_AC", "Q_CC"):
            provenance = dict(self.provenance_by_block).get(block_name)
            if provenance != ORDINARY_HESSIAN_PROVENANCE:
                raise ValueError(
                    f"{block_name} provenance must remain ordinary_coordinate_hessian."
                )

    @property
    def source_primitive_ids(self) -> frozenset[str]:
        return _receipt_ids(self.source_query_receipts)

    @property
    def provenance_id(self) -> str:
        return _digest_payload(
            "phase2_ordinary_coordinate_hessian_blocks_v1",
            {
                "workspace_fingerprint": self.workspace_fingerprint,
                "candidate_keys": list(self.candidate_keys),
                "Q_AA": self.Q_AA.tolist(),
                "Q_AC": self.Q_AC.tolist(),
                "Q_CC": self.Q_CC.tolist(),
                "source_primitive_ids": sorted(self.source_primitive_ids),
                "provenance": self.hessian_provenance,
            },
        )

    def subset(
        self, candidate_indices: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = tuple(int(index) for index in candidate_indices)
        return (
            np.asarray(self.Q_AA, dtype=float),
            np.asarray(self.Q_AC[:, indices], dtype=float),
            np.asarray(self.Q_CC[np.ix_(indices, indices)], dtype=float),
        )

    def portable_payload(self) -> dict[str, Any]:
        return {
            "schema": "phase2_ordinary_coordinate_hessian_blocks_v1",
            "workspace_fingerprint": self.workspace_fingerprint,
            "candidate_keys": list(self.candidate_keys),
            "Q_AA": self.Q_AA.tolist(),
            "Q_AC": self.Q_AC.tolist(),
            "Q_CC": self.Q_CC.tolist(),
            "hessian_provenance": self.hessian_provenance,
            "provenance_by_block": dict(self.provenance_by_block),
            "source_query_receipts": [
                receipt.portable_payload() for receipt in self.source_query_receipts
            ],
        }


@dataclass(frozen=True)
class OptimizerInverseCurvaturePrior:
    """Physical-frame inverse raised-curvature prior used after growth."""

    B_plus: np.ndarray
    inherited_active_rank: int
    candidate_rank: int
    active_source: str
    candidate_scale: float
    provenance: str = OPTIMIZER_INVERSE_CURVATURE_PROVENANCE
    mixed_block_status: str = OPTIMIZER_MIXED_BLOCK_STATUS

    def __post_init__(self) -> None:
        dimension = int(self.inherited_active_rank) + int(self.candidate_rank)
        object.__setattr__(self, "B_plus", _readonly_symmetric(self.B_plus, dimension=dimension))
        if self.provenance != OPTIMIZER_INVERSE_CURVATURE_PROVENANCE:
            raise ValueError("optimizer curvature provenance is invalid.")
        if self.mixed_block_status != OPTIMIZER_MIXED_BLOCK_STATUS:
            raise ValueError("optimizer mixed block status is invalid.")
        if not math.isfinite(float(self.candidate_scale)) or float(
            self.candidate_scale
        ) <= 0.0:
            raise ValueError("candidate_scale must be finite and positive.")


def build_optimizer_inverse_curvature_prior(
    *,
    active_inverse_curvature: np.ndarray | None,
    active_rank: int,
    candidate_rank: int,
    candidate_scale: float,
    reset_active_scale: float | None = None,
    active_provenance: str = "transported_inverse_rbfgs",
) -> OptimizerInverseCurvaturePrior:
    """Construct mandatory ``diag(B_A, beta I)`` without consulting Phase-II Q."""

    active_count = int(active_rank)
    candidate_count = int(candidate_rank)
    if active_count < 0 or candidate_count <= 0:
        raise ValueError("active_rank must be nonnegative and candidate_rank positive.")
    beta = float(candidate_scale)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("candidate_scale must be finite and positive.")
    active_source = str(active_provenance)
    use_reset = active_inverse_curvature is None
    if not use_reset:
        active_block = np.asarray(active_inverse_curvature, dtype=float)
        use_reset = bool(
            active_block.shape != (active_count, active_count)
            or not np.all(np.isfinite(active_block))
        )
    if use_reset:
        reset_scale = beta if reset_active_scale is None else float(reset_active_scale)
        if not math.isfinite(reset_scale) or reset_scale <= 0.0:
            raise ValueError("reset_active_scale must be finite and positive.")
        active_block = reset_scale * np.eye(active_count, dtype=float)
        active_source = "regularized_isotropic_reset_prior"
    else:
        active_block = 0.5 * (active_block + active_block.T)
    B_plus = np.zeros((active_count + candidate_count,) * 2, dtype=float)
    B_plus[:active_count, :active_count] = active_block
    B_plus[active_count:, active_count:] = beta * np.eye(candidate_count)
    return OptimizerInverseCurvaturePrior(
        B_plus=B_plus,
        inherited_active_rank=active_count,
        candidate_rank=candidate_count,
        active_source=active_source,
        candidate_scale=beta,
    )


@dataclass(frozen=True)
class Phase2QueryClosedSolve:
    feasible: bool
    reason: str
    candidate_indices: tuple[int, ...]
    candidate_keys: tuple[str, ...]
    joint_step: np.ndarray
    predicted_reduction: float
    resource_burden: float
    score: float
    trust_lambda: float
    fubini_study_displacement_sq: float
    direct_schur_step_difference: float
    shared_direct_step_difference: float
    direct_kkt_residual: float
    schur_kkt_residual: float
    ordinary_hessian_provenance: str
    optimizer_curvature_used: bool
    source_primitive_ids: tuple[str, ...]
    query_free_derived_fields: tuple[str, ...]
    solve_result: JointLinearSolveResult | None = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "joint_step", _readonly_array(self.joint_step, ndim=1))
        if self.ordinary_hessian_provenance != ORDINARY_HESSIAN_PROVENANCE:
            raise ValueError("ordinary Hessian provenance was relabeled.")
        if self.optimizer_curvature_used:
            raise ValueError("Phase-II selection must not use optimizer curvature B.")


def _direct_schur_steps(
    *,
    matrix: np.ndarray,
    rhs: np.ndarray,
    active_count: int,
    rcond: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    M = np.asarray(matrix, dtype=float)
    b = np.asarray(rhs, dtype=float)
    direct = np.asarray(np.linalg.pinv(M, rcond=rcond) @ b, dtype=float)
    if active_count == 0:
        schur = direct.copy()
    else:
        M_AA = M[:active_count, :active_count]
        M_AC = M[:active_count, active_count:]
        M_CA = M[active_count:, :active_count]
        M_CC = M[active_count:, active_count:]
        rhs_A = b[:active_count]
        rhs_C = b[active_count:]
        inverse_A = np.linalg.pinv(M_AA, rcond=rcond)
        schur_matrix = M_CC - M_CA @ inverse_A @ M_AC
        schur_rhs = rhs_C - M_CA @ inverse_A @ rhs_A
        step_C = np.asarray(
            np.linalg.pinv(schur_matrix, rcond=rcond) @ schur_rhs,
            dtype=float,
        )
        step_A = np.asarray(inverse_A @ (rhs_A - M_AC @ step_C), dtype=float)
        schur = np.concatenate([step_A, step_C])
    difference = float(np.linalg.norm(direct - schur))
    return (
        direct,
        schur,
        difference,
        float(np.linalg.norm(M @ direct - b)),
        float(np.linalg.norm(M @ schur - b)),
    )


def solve_phase2_query_closed_subset(
    *,
    workspace: QueryClosedPopulationWorkspace,
    ordinary_hessian: Phase2OrdinaryHessianBlocks,
    candidate_indices: Sequence[int],
    resource_burden: float = 0.0,
    solve_config: JointLinearSolveConfig | None = None,
    schur_parity_tolerance: float = 1e-8,
) -> Phase2QueryClosedSolve:
    """Solve the mandatory ordinary-Hessian metric trust model for a subset."""

    config = solve_config if solve_config is not None else JointLinearSolveConfig()
    indices = tuple(int(index) for index in candidate_indices)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("candidate_indices must be nonempty and unique.")
    if min(indices) < 0 or max(indices) >= len(workspace.candidate_records):
        raise IndexError("candidate index is out of range.")
    keys = tuple(workspace.candidate_keys[index] for index in indices)
    source_ids = tuple(
        sorted(workspace.source_primitive_ids | ordinary_hessian.source_primitive_ids)
    )
    burden = float(resource_burden)
    if not math.isfinite(burden) or burden < 0.0:
        raise ValueError("resource_burden must be finite and nonnegative.")

    def failed(reason: str) -> Phase2QueryClosedSolve:
        return Phase2QueryClosedSolve(
            feasible=False,
            reason=reason,
            candidate_indices=indices,
            candidate_keys=keys,
            joint_step=np.zeros(workspace.anchor.active_dimension + len(indices)),
            predicted_reduction=0.0,
            resource_burden=burden,
            score=0.0,
            trust_lambda=0.0,
            fubini_study_displacement_sq=0.0,
            direct_schur_step_difference=math.inf,
            shared_direct_step_difference=math.inf,
            direct_kkt_residual=math.inf,
            schur_kkt_residual=math.inf,
            ordinary_hessian_provenance=ORDINARY_HESSIAN_PROVENANCE,
            optimizer_curvature_used=False,
            source_primitive_ids=source_ids,
            query_free_derived_fields=(),
            solve_result=None,
        )

    if ordinary_hessian.workspace_fingerprint != workspace.workspace_fingerprint:
        return failed("ordinary_hessian_workspace_fingerprint_mismatch")
    if ordinary_hessian.candidate_keys != workspace.candidate_keys:
        return failed("ordinary_hessian_candidate_registry_mismatch")
    G_AB, G_BB, b_B = workspace.subset_geometry(indices)
    if not np.all(np.isfinite(G_BB)):
        return failed("missing_candidate_pair_gram_primitive")
    Q_AA, Q_AB, Q_BB = ordinary_hessian.subset(indices)
    if not (
        np.all(np.isfinite(Q_AA))
        and np.all(np.isfinite(Q_AB))
        and np.all(np.isfinite(Q_BB))
    ):
        return failed("missing_phase2_ordinary_hessian_primitive")
    gram = np.block([[workspace.anchor.G_AA, G_AB], [G_AB.T, G_BB]])
    active_factor = factor_supported_metric(
        workspace.anchor.G_AA,
        rank_relative_tolerance=float(config.rank_relative_tolerance),
        metric_regularization=float(config.metric_regularization),
    )
    joint_factor = factor_supported_metric(
        gram,
        rank_relative_tolerance=float(config.rank_relative_tolerance),
        metric_regularization=float(config.metric_regularization),
    )
    active_factor_valid = bool(
        active_factor.feasible
        or active_factor.reason == "empty_supported_metric_subspace"
    )
    joint_factor_valid = bool(
        joint_factor.feasible
        or joint_factor.reason == "empty_supported_metric_subspace"
    )
    if not active_factor_valid or not joint_factor_valid:
        return failed("subset_supported_rank_factorization_failed")
    if int(joint_factor.rank - active_factor.rank) != len(indices):
        return failed("candidate_subset_supported_rank_gate_failed")
    hessian = np.block([[Q_AA, Q_AB], [Q_AB.T, Q_BB]])
    differential = np.concatenate([workspace.anchor.b_A, b_B])
    solved = solve_joint_linear_model(
        gram=gram,
        hessian=hessian,
        gradient=-differential,
        active_coordinate_count=workspace.anchor.active_dimension,
        config=config,
    )
    if not solved.feasible:
        result = failed("joint_trust_solve_failed:" + solved.reason)
        return Phase2QueryClosedSolve(
            **{**result.__dict__, "solve_result": solved}
        )
    whitening = np.asarray(joint_factor.whitening, dtype=float)
    trust_matrix = (
        whitening.T @ hessian @ whitening
        + float(solved.trust_lambda)
        * np.eye(int(joint_factor.rank), dtype=float)
    )
    supported_rhs = np.asarray(
        whitening.T @ (-differential), dtype=float
    )
    rcond = float(max(config.energy_regularization, 1e-15))
    direct_supported, schur_supported, supported_difference, direct_residual, schur_residual = _direct_schur_steps(
        matrix=trust_matrix,
        rhs=supported_rhs,
        active_count=min(workspace.anchor.active_dimension, int(joint_factor.rank)),
        rcond=rcond,
    )
    direct = np.asarray(whitening @ direct_supported, dtype=float)
    schur = np.asarray(whitening @ schur_supported, dtype=float)
    difference = float(np.linalg.norm(direct - schur))
    scale = float(max(1.0, np.linalg.norm(direct), np.linalg.norm(solved.joint_step)))
    shared_difference = float(np.linalg.norm(solved.joint_step - direct))
    parity_ok = bool(
        difference <= float(schur_parity_tolerance) * scale
        and shared_difference <= float(schur_parity_tolerance) * scale
        and direct_residual <= float(schur_parity_tolerance) * scale
        and schur_residual <= float(schur_parity_tolerance) * scale
    )
    if not parity_ok:
        result = failed("supported_direct_schur_or_shared_parity_failed")
        return Phase2QueryClosedSolve(
            **{
                **result.__dict__,
                "joint_step": solved.joint_step,
                "trust_lambda": solved.trust_lambda,
                "direct_schur_step_difference": difference,
                "shared_direct_step_difference": shared_difference,
                "direct_kkt_residual": direct_residual,
                "schur_kkt_residual": schur_residual,
                "solve_result": solved,
            }
        )
    predicted = float(solved.predicted_reduction)
    return Phase2QueryClosedSolve(
        feasible=True,
        reason="ordinary_coordinate_hessian_metric_trust_response",
        candidate_indices=indices,
        candidate_keys=keys,
        joint_step=solved.joint_step,
        predicted_reduction=predicted,
        resource_burden=burden,
        score=float(predicted / (1.0 + burden)),
        trust_lambda=float(solved.trust_lambda),
        fubini_study_displacement_sq=float(solved.fubini_study_displacement_sq),
        direct_schur_step_difference=difference,
        shared_direct_step_difference=shared_difference,
        direct_kkt_residual=direct_residual,
        schur_kkt_residual=schur_residual,
        ordinary_hessian_provenance=ORDINARY_HESSIAN_PROVENANCE,
        optimizer_curvature_used=False,
        source_primitive_ids=source_ids,
        query_free_derived_fields=(
            "joint_metric_assembly",
            "joint_ordinary_hessian_assembly",
            "supported_metric_whitening",
            "direct_and_schur_trust_solve",
            "phase2_query_closed_score",
        ),
        solve_result=solved,
    )


@dataclass(frozen=True)
class CombinatorialBatchSelection:
    feasible: bool
    reason: str
    selected: Phase2QueryClosedSolve | None
    subsets_searched: int
    feasible_subset_count: int
    ordered_results: tuple[Phase2QueryClosedSolve, ...]
    authoritative_policy: str = "deterministic_combinatorial_subset_argmax_v1"


def select_combinatorial_query_closed_batch(
    *,
    workspace: QueryClosedPopulationWorkspace,
    ordinary_hessian: Phase2OrdinaryHessianBlocks,
    max_batch_size: int,
    candidate_resource_burdens: Mapping[str, float] | None = None,
    solve_config: JointLinearSolveConfig | None = None,
    eligible_candidate_keys: Sequence[str] | None = None,
    schur_parity_tolerance: float = 1e-8,
) -> CombinatorialBatchSelection:
    """Exhaustively select the best feasible subset under the cardinality cap."""

    cap = int(max_batch_size)
    if cap <= 0:
        raise ValueError("max_batch_size must be positive.")
    allowed = (
        set(workspace.candidate_keys)
        if eligible_candidate_keys is None
        else {str(key) for key in eligible_candidate_keys}
    )
    unknown = allowed - set(workspace.candidate_keys)
    if unknown:
        raise ValueError(f"eligible_candidate_keys contains unknown keys: {sorted(unknown)}")
    population_indices = tuple(
        index
        for index, key in enumerate(workspace.candidate_keys)
        if key in allowed
    )
    burdens = {str(key): float(value) for key, value in (candidate_resource_burdens or {}).items()}
    results: list[Phase2QueryClosedSolve] = []
    config = solve_config if solve_config is not None else JointLinearSolveConfig()
    for cardinality in range(1, min(cap, len(population_indices)) + 1):
        for indices in itertools.combinations(population_indices, cardinality):
            keys = tuple(workspace.candidate_keys[index] for index in indices)
            burden = float(sum(burdens.get(key, 0.0) for key in keys))
            cache_key = (
                "phase2_query_closed_subset_v1",
                ordinary_hessian.provenance_id,
                keys,
                _canonical_json(config.as_dict()),
                burden,
                float(schur_parity_tolerance),
            )
            cached = workspace.subset_solve_cache.get(cache_key)
            if cached is None:
                cached = solve_phase2_query_closed_subset(
                    workspace=workspace,
                    ordinary_hessian=ordinary_hessian,
                    candidate_indices=indices,
                    resource_burden=burden,
                    solve_config=config,
                    schur_parity_tolerance=schur_parity_tolerance,
                )
                workspace.subset_solve_cache[cache_key] = cached
            results.append(cached)
    feasible = [result for result in results if result.feasible]
    if not feasible:
        return CombinatorialBatchSelection(
            feasible=False,
            reason="no_feasible_second_order_subset",
            selected=None,
            subsets_searched=len(results),
            feasible_subset_count=0,
            ordered_results=tuple(results),
        )
    selected = sorted(
        feasible,
        key=lambda result: (
            -float(result.score),
            -float(result.predicted_reduction),
            result.candidate_keys,
        ),
    )[0]
    return CombinatorialBatchSelection(
        feasible=True,
        reason="combinatorial_query_closed_argmax",
        selected=selected,
        subsets_searched=len(results),
        feasible_subset_count=len(feasible),
        ordered_results=tuple(results),
    )


@dataclass(frozen=True)
class FormalGrowthGeometryReceipt:
    state_fingerprint: str
    branch_id: str
    manifold_id: str
    ordered_scaffold_fingerprint: str
    theta_fingerprint: str
    old_coordinate_registry_fingerprint: str
    new_coordinate_registry_fingerprint: str
    parameterization_tie_map_fingerprint: str
    hamiltonian_fingerprint: str
    candidate_keys: tuple[str, ...]
    insertion_positions: tuple[int, ...]
    old_to_new_registry_mapping: tuple[int, ...]
    G_AA: np.ndarray
    G_AB: np.ndarray
    G_BB: np.ndarray
    candidate_gradients: np.ndarray
    rank_rule_fingerprint: str
    retained_spectrum: np.ndarray
    metric_convention: str
    G_AA_provenance: str
    source_primitive_ids: tuple[str, ...]
    query_free_derived_fields: tuple[str, ...]
    missing_or_regularized_fields: tuple[str, ...]
    zero_new_coordinates: bool
    old_gate_subsequence_unchanged: bool
    schema: str = FORMAL_GROWTH_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "state_fingerprint",
            "branch_id",
            "manifold_id",
            "ordered_scaffold_fingerprint",
            "theta_fingerprint",
            "old_coordinate_registry_fingerprint",
            "new_coordinate_registry_fingerprint",
            "parameterization_tie_map_fingerprint",
            "hamiltonian_fingerprint",
            "rank_rule_fingerprint",
            "metric_convention",
            "G_AA_provenance",
        ):
            _require_text(name, getattr(self, name))
        if self.schema != FORMAL_GROWTH_RECEIPT_SCHEMA:
            raise ValueError(f"schema must be {FORMAL_GROWTH_RECEIPT_SCHEMA!r}.")
        candidate_count = len(self.candidate_keys)
        active = int(np.asarray(self.G_AA).shape[0])
        object.__setattr__(self, "G_AA", _readonly_symmetric(self.G_AA, dimension=active))
        object.__setattr__(
            self,
            "G_AB",
            _readonly_array(self.G_AB, ndim=2, shape=(active, candidate_count)),
        )
        object.__setattr__(self, "G_BB", _readonly_symmetric(self.G_BB, dimension=candidate_count))
        object.__setattr__(
            self,
            "candidate_gradients",
            _readonly_array(self.candidate_gradients, ndim=1, shape=(candidate_count,)),
        )
        object.__setattr__(self, "retained_spectrum", _readonly_array(self.retained_spectrum, ndim=1))
        if len(self.insertion_positions) != candidate_count:
            raise ValueError("insertion_positions must match candidates.")
        if len(self.old_to_new_registry_mapping) != active:
            raise ValueError("old_to_new_registry_mapping must match active coordinates.")

    @property
    def receipt_fingerprint(self) -> str:
        return _digest_payload(
            FORMAL_GROWTH_RECEIPT_SCHEMA,
            self.portable_payload(include_fingerprint=False),
        )

    def portable_payload(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "state_fingerprint": self.state_fingerprint,
            "branch_id": self.branch_id,
            "manifold_id": self.manifold_id,
            "ordered_scaffold_fingerprint": self.ordered_scaffold_fingerprint,
            "theta_fingerprint": self.theta_fingerprint,
            "old_coordinate_registry_fingerprint": (
                self.old_coordinate_registry_fingerprint
            ),
            "new_coordinate_registry_fingerprint": (
                self.new_coordinate_registry_fingerprint
            ),
            "parameterization_tie_map_fingerprint": (
                self.parameterization_tie_map_fingerprint
            ),
            "hamiltonian_fingerprint": self.hamiltonian_fingerprint,
            "candidate_keys": list(self.candidate_keys),
            "insertion_positions": list(self.insertion_positions),
            "old_to_new_registry_mapping": list(self.old_to_new_registry_mapping),
            "G_AA": self.G_AA.tolist(),
            "G_AB": self.G_AB.tolist(),
            "G_BB": self.G_BB.tolist(),
            "candidate_gradients": self.candidate_gradients.tolist(),
            "rank_rule_fingerprint": self.rank_rule_fingerprint,
            "retained_spectrum": self.retained_spectrum.tolist(),
            "metric_convention": self.metric_convention,
            "G_AA_provenance": self.G_AA_provenance,
            "source_primitive_ids": list(self.source_primitive_ids),
            "query_free_derived_fields": list(self.query_free_derived_fields),
            "missing_or_regularized_fields": list(
                self.missing_or_regularized_fields
            ),
            "zero_new_coordinates": bool(self.zero_new_coordinates),
            "old_gate_subsequence_unchanged": bool(
                self.old_gate_subsequence_unchanged
            ),
        }
        if include_fingerprint:
            payload["receipt_fingerprint"] = self.receipt_fingerprint
        return payload


def build_formal_growth_geometry_receipt(
    *,
    workspace: QueryClosedPopulationWorkspace,
    selected: Phase2QueryClosedSolve,
    old_to_new_registry_mapping: Sequence[int],
    new_coordinate_registry_fingerprint: str,
    rank_relative_tolerance: float,
    metric_regularization: float,
    zero_new_coordinates: bool,
    old_gate_subsequence_unchanged: bool,
    metric_convention: str = "raw_fubini_study_supported_metric_whitened_v1",
) -> FormalGrowthGeometryReceipt:
    if not selected.feasible:
        raise ValueError("growth receipt requires a feasible selected subset.")
    indices = selected.candidate_indices
    G_AB, G_BB, b_B = workspace.subset_geometry(indices)
    full_metric = np.block(
        [[workspace.anchor.G_AA, G_AB], [G_AB.T, G_BB]]
    )
    factor = factor_supported_metric(
        full_metric,
        rank_relative_tolerance=rank_relative_tolerance,
        metric_regularization=metric_regularization,
    )
    rank_rule_fingerprint = _digest_payload(
        "supported_metric_rank_rule_v1",
        {
            "rank_relative_tolerance": float(rank_relative_tolerance),
            "metric_regularization": float(metric_regularization),
            "metric_convention": str(metric_convention),
        },
    )
    records = tuple(workspace.candidate_records[index] for index in indices)
    selected_metric_source_ids = set(workspace.anchor.source_primitive_ids)
    for record in records:
        selected_metric_source_ids.update(record.source_primitive_ids)
    missing_fields: list[str] = []
    if metric_regularization > 0.0:
        missing_fields.append("metric_ridge_regularized_supported_coordinates")
    return FormalGrowthGeometryReceipt(
        state_fingerprint=workspace.anchor.state_fingerprint,
        branch_id=workspace.anchor.branch_id,
        manifold_id=workspace.anchor.manifold_id,
        ordered_scaffold_fingerprint=(
            workspace.anchor.ordered_scaffold_fingerprint
        ),
        theta_fingerprint=workspace.anchor.theta_fingerprint,
        old_coordinate_registry_fingerprint=(
            workspace.anchor.coordinate_registry_fingerprint
        ),
        new_coordinate_registry_fingerprint=str(
            new_coordinate_registry_fingerprint
        ),
        parameterization_tie_map_fingerprint=(
            workspace.anchor.parameterization_tie_map_fingerprint
        ),
        hamiltonian_fingerprint=workspace.anchor.hamiltonian_fingerprint,
        candidate_keys=tuple(record.candidate_key for record in records),
        insertion_positions=tuple(record.insertion_position for record in records),
        old_to_new_registry_mapping=tuple(
            int(index) for index in old_to_new_registry_mapping
        ),
        G_AA=workspace.anchor.G_AA,
        G_AB=G_AB,
        G_BB=G_BB,
        candidate_gradients=b_B,
        rank_rule_fingerprint=rank_rule_fingerprint,
        retained_spectrum=factor.retained_eigenvalues,
        metric_convention=str(metric_convention),
        G_AA_provenance=workspace.anchor.gram_provenance,
        # Growth consumes first-order metric geometry only.  The selector's
        # ordinary-Hessian Q primitives must never be relabeled as growth work.
        source_primitive_ids=tuple(sorted(selected_metric_source_ids)),
        query_free_derived_fields=tuple(
            sorted(
                set(workspace.query_free_derived_fields)
                | {
                    "selected_growth_metric_blocks",
                    "growth_supported_rank_and_spectrum",
                    "growth_projector_enlargement_inputs",
                }
            )
        ),
        missing_or_regularized_fields=tuple(missing_fields),
        zero_new_coordinates=bool(zero_new_coordinates),
        old_gate_subsequence_unchanged=bool(old_gate_subsequence_unchanged),
    )


@dataclass(frozen=True)
class GrowthReceiptExpectation:
    state_fingerprint: str
    branch_id: str
    manifold_id: str
    ordered_scaffold_fingerprint: str
    theta_fingerprint: str
    old_coordinate_registry_fingerprint: str
    new_coordinate_registry_fingerprint: str
    parameterization_tie_map_fingerprint: str
    hamiltonian_fingerprint: str
    candidate_keys: tuple[str, ...]
    insertion_positions: tuple[int, ...]
    old_to_new_registry_mapping: tuple[int, ...]
    rank_rule_fingerprint: str
    metric_convention: str
    zero_new_coordinates: bool = True
    old_gate_subsequence_unchanged: bool = True

    @classmethod
    def from_receipt(
        cls, receipt: FormalGrowthGeometryReceipt
    ) -> "GrowthReceiptExpectation":
        return cls(
            state_fingerprint=receipt.state_fingerprint,
            branch_id=receipt.branch_id,
            manifold_id=receipt.manifold_id,
            ordered_scaffold_fingerprint=receipt.ordered_scaffold_fingerprint,
            theta_fingerprint=receipt.theta_fingerprint,
            old_coordinate_registry_fingerprint=(
                receipt.old_coordinate_registry_fingerprint
            ),
            new_coordinate_registry_fingerprint=(
                receipt.new_coordinate_registry_fingerprint
            ),
            parameterization_tie_map_fingerprint=(
                receipt.parameterization_tie_map_fingerprint
            ),
            hamiltonian_fingerprint=receipt.hamiltonian_fingerprint,
            candidate_keys=receipt.candidate_keys,
            insertion_positions=receipt.insertion_positions,
            old_to_new_registry_mapping=receipt.old_to_new_registry_mapping,
            rank_rule_fingerprint=receipt.rank_rule_fingerprint,
            metric_convention=receipt.metric_convention,
            zero_new_coordinates=True,
            old_gate_subsequence_unchanged=True,
        )


@dataclass(frozen=True)
class GrowthReceiptValidation:
    valid: bool
    reason: str
    mismatched_fields: tuple[str, ...]
    query_reuse_allowed: bool
    incremental_query_charge: int | None


def validate_formal_growth_geometry_receipt(
    receipt: FormalGrowthGeometryReceipt,
    expectation: GrowthReceiptExpectation,
) -> GrowthReceiptValidation:
    fields = (
        "state_fingerprint",
        "branch_id",
        "manifold_id",
        "ordered_scaffold_fingerprint",
        "theta_fingerprint",
        "old_coordinate_registry_fingerprint",
        "new_coordinate_registry_fingerprint",
        "parameterization_tie_map_fingerprint",
        "hamiltonian_fingerprint",
        "candidate_keys",
        "insertion_positions",
        "old_to_new_registry_mapping",
        "rank_rule_fingerprint",
        "metric_convention",
        "zero_new_coordinates",
        "old_gate_subsequence_unchanged",
    )
    mismatches = tuple(
        name
        for name in fields
        if getattr(receipt, name) != getattr(expectation, name)
    )
    valid = not mismatches
    return GrowthReceiptValidation(
        valid=valid,
        reason=("growth_geometry_reuse_valid" if valid else "growth_geometry_reuse_invalid"),
        mismatched_fields=mismatches,
        query_reuse_allowed=valid,
        incremental_query_charge=(0 if valid else None),
    )


class QueryPrimitiveLedger:
    """Unique-logical-primitive accounting; classical work never charges it."""

    def __init__(self) -> None:
        self._kind_by_id: dict[str, str] = {}
        self._requested_ids: set[str] = set()
        self._reused_ids: set[str] = set()
        self._ids_by_phase: dict[str, set[str]] = {}
        self._returned_fields_by_id: dict[str, set[str]] = {}
        self._consumer_phases_by_id: dict[str, set[str]] = {}
        self._derived_fields: set[str] = set()
        self._matrix_element_diagnostics: dict[str, int] = {}
        self._statevector_shortcut_used = False

    def consume_receipt(self, receipt: QueryReceipt, *, consumer_phase: str) -> None:
        phase = _require_text("consumer_phase", consumer_phase)
        kind_map = receipt.kind_map
        for primitive_id, kind in kind_map.items():
            previous = self._kind_by_id.get(primitive_id)
            if previous is not None and previous != kind:
                raise ValueError("one primitive ID was assigned conflicting kinds.")
            self._kind_by_id[primitive_id] = kind
            self._returned_fields_by_id.setdefault(primitive_id, set()).update(
                receipt.returned_fields
            )
            self._consumer_phases_by_id.setdefault(primitive_id, set()).add(phase)
        self._requested_ids.update(receipt.primitive_ids_requested)
        self._reused_ids.update(receipt.primitive_ids_reused)
        self._ids_by_phase.setdefault(phase, set()).update(receipt.all_primitive_ids)
        self._statevector_shortcut_used = bool(
            self._statevector_shortcut_used or receipt.statevector_shortcut_used
        )

    def reuse_known_primitives(
        self,
        primitive_ids: Iterable[str],
        *,
        consumer_phase: str,
    ) -> None:
        """Record a later consumer without manufacturing a new receipt."""

        phase = _require_text("consumer_phase", consumer_phase)
        ids = {str(value) for value in primitive_ids}
        unknown = ids - set(self._kind_by_id)
        if unknown:
            raise ValueError(
                "cannot reuse unknown primitive IDs: " + ", ".join(sorted(unknown))
            )
        self._reused_ids.update(ids)
        self._ids_by_phase.setdefault(phase, set()).update(ids)
        for primitive_id in ids:
            self._consumer_phases_by_id.setdefault(primitive_id, set()).add(phase)

    def merge(self, other: "QueryPrimitiveLedger") -> None:
        """Union a branch-local ledger into this ledger with identity checks."""

        if not isinstance(other, QueryPrimitiveLedger):
            raise TypeError("other must be a QueryPrimitiveLedger.")
        for primitive_id, kind in other._kind_by_id.items():
            previous = self._kind_by_id.get(primitive_id)
            if previous is not None and previous != kind:
                raise ValueError("merged ledgers assign conflicting primitive kinds.")
            self._kind_by_id[primitive_id] = kind
        self._requested_ids.update(other._requested_ids)
        self._reused_ids.update(other._reused_ids)
        for phase, ids in other._ids_by_phase.items():
            self._ids_by_phase.setdefault(phase, set()).update(ids)
        for primitive_id, fields in other._returned_fields_by_id.items():
            self._returned_fields_by_id.setdefault(primitive_id, set()).update(fields)
        for primitive_id, phases in other._consumer_phases_by_id.items():
            self._consumer_phases_by_id.setdefault(primitive_id, set()).update(phases)
        self._derived_fields.update(other._derived_fields)
        for name, count in other._matrix_element_diagnostics.items():
            self._matrix_element_diagnostics[name] = int(
                self._matrix_element_diagnostics.get(name, 0) + int(count)
            )
        self._statevector_shortcut_used = bool(
            self._statevector_shortcut_used or other._statevector_shortcut_used
        )

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "schema": "formal_manifold_query_primitive_ledger_checkpoint_v1",
            "kind_by_id": dict(sorted(self._kind_by_id.items())),
            "requested_ids": sorted(self._requested_ids),
            "reused_ids": sorted(self._reused_ids),
            "ids_by_phase": {
                key: sorted(value) for key, value in sorted(self._ids_by_phase.items())
            },
            "returned_fields_by_id": {
                key: sorted(value)
                for key, value in sorted(self._returned_fields_by_id.items())
            },
            "consumer_phases_by_id": {
                key: sorted(value)
                for key, value in sorted(self._consumer_phases_by_id.items())
            },
            "derived_fields": sorted(self._derived_fields),
            "matrix_element_diagnostics": dict(
                sorted(self._matrix_element_diagnostics.items())
            ),
            "statevector_shortcut_used": bool(self._statevector_shortcut_used),
        }

    @classmethod
    def from_checkpoint_payload(
        cls, payload: Mapping[str, Any]
    ) -> "QueryPrimitiveLedger":
        data = dict(payload)
        if data.get("schema") != "formal_manifold_query_primitive_ledger_checkpoint_v1":
            raise ValueError("unsupported query-ledger checkpoint schema.")
        ledger = cls()
        kind_by_id = {
            str(key): str(value)
            for key, value in dict(data.get("kind_by_id", {})).items()
        }
        if any(kind not in PRIMITIVE_KINDS for kind in kind_by_id.values()):
            raise ValueError("query-ledger checkpoint contains an invalid primitive kind.")
        ledger._kind_by_id = kind_by_id
        known = set(kind_by_id)
        ledger._requested_ids = {str(value) for value in data.get("requested_ids", [])}
        ledger._reused_ids = {str(value) for value in data.get("reused_ids", [])}
        if not (ledger._requested_ids | ledger._reused_ids).issubset(known):
            raise ValueError("query-ledger checkpoint references unknown primitive IDs.")
        ledger._ids_by_phase = {
            str(key): {str(value) for value in values}
            for key, values in dict(data.get("ids_by_phase", {})).items()
        }
        ledger._returned_fields_by_id = {
            str(key): {str(value) for value in values}
            for key, values in dict(data.get("returned_fields_by_id", {})).items()
        }
        ledger._consumer_phases_by_id = {
            str(key): {str(value) for value in values}
            for key, values in dict(data.get("consumer_phases_by_id", {})).items()
        }
        if any(
            not ids.issubset(known) for ids in ledger._ids_by_phase.values()
        ) or not set(ledger._returned_fields_by_id).issubset(known) or not set(
            ledger._consumer_phases_by_id
        ).issubset(known):
            raise ValueError("query-ledger checkpoint provenance is internally inconsistent.")
        ledger._derived_fields = {
            str(value) for value in data.get("derived_fields", [])
        }
        ledger._matrix_element_diagnostics = {
            str(key): int(value)
            for key, value in dict(
                data.get("matrix_element_diagnostics", {})
            ).items()
        }
        if any(value < 0 for value in ledger._matrix_element_diagnostics.values()):
            raise ValueError("query-ledger diagnostic counts must be nonnegative.")
        ledger._statevector_shortcut_used = bool(
            data.get("statevector_shortcut_used", False)
        )
        return ledger

    def record_query_free_derived_fields(self, fields: Iterable[str]) -> None:
        self._derived_fields.update(str(field) for field in fields)

    def record_matrix_element_diagnostic(self, name: str, count: int) -> None:
        resolved = int(count)
        if resolved < 0:
            raise ValueError("matrix-element diagnostic counts must be nonnegative.")
        self._matrix_element_diagnostics[str(name)] = resolved

    @property
    def unique_primitive_ids(self) -> frozenset[str]:
        return frozenset(self._kind_by_id)

    def telemetry(
        self,
        *,
        expected_actual_operator_probe_count: int | None = None,
        baseline_primitive_ids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        counts = {name: 0 for name in LEDGER_CATEGORY_BY_KIND.values()}
        for kind in self._kind_by_id.values():
            counts[LEDGER_CATEGORY_BY_KIND[kind]] += 1
        actual = len(self._kind_by_id)
        expected = (
            actual
            if expected_actual_operator_probe_count is None
            else int(expected_actual_operator_probe_count)
        )
        baseline = (
            None
            if baseline_primitive_ids is None
            else set(str(value) for value in baseline_primitive_ids)
        )
        current = set(self._kind_by_id)
        reconciliation = {
            "actual_operator_probe_count": actual,
            "expected_actual_operator_probe_count": expected,
            "count_equal": actual == expected,
            "baseline_set_equal": None if baseline is None else current == baseline,
            "new_vs_baseline": [] if baseline is None else sorted(current - baseline),
            "missing_vs_baseline": [] if baseline is None else sorted(baseline - current),
        }
        phase = self._ids_by_phase
        return {
            **counts,
            "actual_operator_probe_count": actual,
            "unique_primitive_ids_requested": sorted(self._requested_ids),
            "unique_primitive_ids_reused": sorted(self._reused_ids),
            "unique_primitive_ids": sorted(current),
            "query_free_derived_fields": sorted(self._derived_fields),
            "primitive_to_returned_fields": {
                key: sorted(value) for key, value in sorted(self._returned_fields_by_id.items())
            },
            "primitive_to_consumer_phases": {
                key: sorted(value) for key, value in sorted(self._consumer_phases_by_id.items())
            },
            "phase1_to_phase2_reuse_count": len(
                phase.get("phase1", set()) & phase.get("phase2", set())
            ),
            "phase2_to_batch_reuse_count": len(
                phase.get("phase2", set()) & phase.get("batch", set())
            ),
            "batch_to_growth_reuse_count": len(
                phase.get("batch", set()) & phase.get("growth", set())
            ),
            "matrix_element_diagnostics": dict(
                sorted(self._matrix_element_diagnostics.items())
            ),
            "statevector_shortcut_used": bool(self._statevector_shortcut_used),
            "primitive_count_reconciliation": reconciliation,
        }


def reconcile_primitive_id_sets(
    *,
    baseline_primitive_ids: Iterable[str],
    enriched_primitive_ids: Iterable[str],
) -> dict[str, Any]:
    baseline = {str(value) for value in baseline_primitive_ids}
    enriched = {str(value) for value in enriched_primitive_ids}
    return {
        "set_equal": baseline == enriched,
        "baseline_count": len(baseline),
        "enriched_count": len(enriched),
        "new_unique_primitive_ids": sorted(enriched - baseline),
        "missing_primitive_ids": sorted(baseline - enriched),
        "zero_incremental_queries": baseline == enriched,
    }


__all__ = [
    "CAPABILITY_ACTIVE_CANDIDATE_GRAM",
    "CAPABILITY_COMMON_TANGENT_CONTRACTION",
    "CAPABILITY_LIVE_TANGENT",
    "CandidateTangentRecord",
    "CombinatorialBatchSelection",
    "EstimatorPrimitiveIdentity",
    "FormalGrowthGeometryReceipt",
    "GrowthReceiptExpectation",
    "GrowthReceiptValidation",
    "LEDGER_CATEGORY_BY_KIND",
    "OPTIMIZER_INVERSE_CURVATURE_PROVENANCE",
    "OPTIMIZER_MIXED_BLOCK_STATUS",
    "ORDINARY_HESSIAN_PROVENANCE",
    "OptimizerInverseCurvaturePrior",
    "PRIMITIVE_KINDS",
    "Phase1QueryClosedScore",
    "Phase2OrdinaryHessianBlocks",
    "projective_state_fingerprint",
    "Phase2QueryClosedSolve",
    "QueryClosedPopulationWorkspace",
    "QueryPrimitiveLedger",
    "QueryReceipt",
    "ResidualizedCandidateBlock",
    "SelectorGeometryAnchor",
    "build_candidate_tangent_record",
    "build_formal_growth_geometry_receipt",
    "build_optimizer_inverse_curvature_prior",
    "build_query_closed_population_workspace",
    "evaluate_phase1_query_closed_score",
    "reconcile_primitive_id_sets",
    "residualize_candidate_block",
    "select_combinatorial_query_closed_batch",
    "solve_phase2_query_closed_subset",
    "validate_formal_growth_geometry_receipt",
]
