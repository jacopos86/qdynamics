"""Accepted-ansatz refit policies independent of selector geometry.

The static selector may use a local active window while the accepted ansatz is
refit in a different optimizer chart.  This module keeps that distinction
explicit.  Its supported-FS chart is fixed for one optimizer invocation and
uses the shared raw-metric support convention from ``joint_linear_solve``.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import pipelines.static_adapt.ra_adapt.support as ra_support
from pipelines.static_adapt.exact_geometry_backend import (
    build_compiled_exact_manifold_adapter,
)
from pipelines.static_adapt.joint_linear_solve import (
    JointLinearSolveConfig,
)
from pipelines.static_adapt.sr_snake_escape_controller import (
    SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1,
    SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1,
)
from src.quantum.ansatz_parameterization import AnsatzParameterLayout
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import CompiledPolynomialAction


ACCEPTED_REFIT_SCOPE_SELECTOR_POLICY_V1 = "selector_policy_v1"
ACCEPTED_REFIT_SCOPE_FULL_ANSATZ_V1 = "full_ansatz_v1"
ACCEPTED_REFIT_SCOPE_CHOICES = (
    ACCEPTED_REFIT_SCOPE_SELECTOR_POLICY_V1,
    ACCEPTED_REFIT_SCOPE_FULL_ANSATZ_V1,
)

ACCEPTED_REFIT_CHART_NATIVE_V1 = "native_v1"
ACCEPTED_REFIT_CHART_SUPPORTED_FS_WHITENED_FIXED_V1 = (
    "supported_fs_whitened_fixed_v1"
)
ACCEPTED_REFIT_CHART_CHOICES = (
    ACCEPTED_REFIT_CHART_NATIVE_V1,
    ACCEPTED_REFIT_CHART_SUPPORTED_FS_WHITENED_FIXED_V1,
)
ACCEPTED_REFIT_BASE_CHART_CHOICES = (
    SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1,
    SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1,
)

ACCEPTED_REFIT_CONFIG_SCHEMA = "accepted_ansatz_refit_config_v1"
SUPPORTED_FS_POWELL_CHART_SCHEMA = "supported_fs_powell_chart_v1"
SUPPORTED_FS_JOINT_STEP_MAP_RECEIPT_SCHEMA = (
    "supported_fs_joint_step_map_receipt_v1"
)
ACCEPTED_REFIT_FIXED_CHART_RECEIPT_SCHEMA = (
    "accepted_refit_fixed_chart_receipt_v1"
)
EXTERNAL_LOGICAL_FS_GRAM_RECEIPT_SCHEMA = (
    "supported_fs_powell_external_logical_gram_receipt_v1"
)


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _portable_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _portable_payload(item) for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_portable_payload(item) for item in value]
    if isinstance(value, np.ndarray):
        return np.asarray(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    return deepcopy(value)


def _immutable_payload(value: Any) -> Any:
    portable = _portable_payload(value)
    if isinstance(portable, dict):
        return MappingProxyType(
            {str(key): _immutable_payload(item) for key, item in portable.items()}
        )
    if isinstance(portable, list):
        return tuple(_immutable_payload(item) for item in portable)
    return portable


def _readonly_array(value: Any, *, dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _fubini_study_gram(
    state: Sequence[complex] | np.ndarray,
    tangents: np.ndarray,
) -> np.ndarray:
    """Return the real pullback Fubini--Study Gram matrix."""

    psi = np.asarray(state, dtype=complex).reshape(-1)
    tangent_matrix = np.asarray(tangents, dtype=complex)
    if tangent_matrix.ndim != 2 or tangent_matrix.shape[0] != psi.size:
        raise ValueError(
            "Fubini--Study tangents must be a state-dimension by coordinate "
            "matrix."
        )
    norm_sq = float(np.real(np.vdot(psi, psi)))
    if not math.isfinite(norm_sq) or norm_sq <= 0.0:
        raise ValueError("Fubini--Study origin state must have positive norm.")
    overlaps = np.conjugate(psi) @ tangent_matrix
    gram_complex = (
        np.conjugate(tangent_matrix).T @ tangent_matrix / norm_sq
        - np.outer(np.conjugate(overlaps), overlaps) / (norm_sq * norm_sq)
    )
    gram = np.asarray(np.real(gram_complex), dtype=float)
    return 0.5 * (gram + gram.T)


def _sha256_text(value: Any, *, field_name: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{field_name} must be a 64-character SHA-256 digest.")
    return text


@dataclass(frozen=True)
class ExternalLogicalFSGramReceipt:
    """Authoritative external inputs for one fixed supported-FS Powell chart.

    The receipt deliberately carries the complete logical Gram matrix and the
    origin endpoint values used only for chart provenance.  It does not carry
    a factorization: support restriction, expanded-runtime projection, and
    whitening remain owned by :func:`build_supported_fs_powell_chart` and
    therefore use exactly the same shared numerical convention as an acquired
    backend evaluation.
    """

    logical_gram: np.ndarray
    origin_state: np.ndarray
    origin_energy: float
    origin_gradient: np.ndarray
    origin_logical_theta: np.ndarray
    origin_runtime_theta: np.ndarray
    coordinate_registry: tuple[str, ...]
    layout_fingerprint_sha256: str
    coordinate_registry_fingerprint_sha256: str
    hamiltonian_fingerprint_sha256: str
    ordered_scaffold_fingerprint_sha256: str
    provenance_schema: str
    provenance_id: str
    source_primitive_ids: tuple[str, ...] = ()
    provenance_payload: Mapping[str, Any] = field(default_factory=dict)
    receipt_id: str = ""
    schema: str = EXTERNAL_LOGICAL_FS_GRAM_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if str(self.schema) != EXTERNAL_LOGICAL_FS_GRAM_RECEIPT_SCHEMA:
            raise ValueError("external logical FS Gram receipt schema mismatch.")
        registry = tuple(str(value) for value in self.coordinate_registry)
        if any(not value for value in registry):
            raise ValueError("external receipt coordinate ids must be nonempty.")
        if len(set(registry)) != len(registry):
            raise ValueError("external receipt coordinate registry must be unique.")
        count = len(registry)
        gram = np.asarray(self.logical_gram, dtype=float)
        if gram.shape != (count, count):
            raise ValueError(
                "external receipt logical_gram shape must match the coordinate "
                f"registry: got {gram.shape}, expected {(count, count)}."
            )
        if not bool(np.all(np.isfinite(gram))):
            raise ValueError("external receipt logical_gram must be finite.")
        if not np.allclose(gram, gram.T, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError("external receipt logical_gram must be symmetric.")
        gram = 0.5 * (gram + gram.T)

        gradient = np.asarray(self.origin_gradient, dtype=float).reshape(-1)
        logical_theta = np.asarray(
            self.origin_logical_theta, dtype=float
        ).reshape(-1)
        runtime_theta = np.asarray(
            self.origin_runtime_theta, dtype=float
        ).reshape(-1)
        state = np.asarray(self.origin_state, dtype=complex).reshape(-1)
        if gradient.shape != (count,):
            raise ValueError(
                "external receipt origin_gradient length must match the "
                "coordinate registry."
            )
        if logical_theta.shape != (count,):
            raise ValueError(
                "external receipt origin_logical_theta length must match the "
                "coordinate registry."
            )
        if runtime_theta.ndim != 1:
            raise ValueError("external receipt origin_runtime_theta must be a vector.")
        if state.ndim != 1 or int(state.size) <= 0:
            raise ValueError("external receipt origin_state must be a nonempty vector.")
        for name, array in (
            ("origin_gradient", gradient),
            ("origin_logical_theta", logical_theta),
            ("origin_runtime_theta", runtime_theta),
        ):
            if not bool(np.all(np.isfinite(array))):
                raise ValueError(f"external receipt {name} must be finite.")
        if not bool(
            np.all(np.isfinite(state.real)) and np.all(np.isfinite(state.imag))
        ):
            raise ValueError("external receipt origin_state must be finite.")
        state_norm = float(np.linalg.norm(state))
        if not math.isfinite(state_norm) or not np.isclose(
            state_norm, 1.0, rtol=1.0e-10, atol=1.0e-12
        ):
            raise ValueError(
                "external receipt origin_state must be normalized; "
                f"norm={state_norm}."
            )
        energy = float(self.origin_energy)
        if not math.isfinite(energy):
            raise ValueError("external receipt origin_energy must be finite.")

        layout_sha = _sha256_text(
            self.layout_fingerprint_sha256,
            field_name="layout_fingerprint_sha256",
        )
        registry_sha = _sha256_text(
            self.coordinate_registry_fingerprint_sha256,
            field_name="coordinate_registry_fingerprint_sha256",
        )
        if registry_sha != _json_sha256(list(registry)):
            raise ValueError(
                "external receipt coordinate-registry fingerprint does not "
                "match its registry."
            )
        hamiltonian_sha = _sha256_text(
            self.hamiltonian_fingerprint_sha256,
            field_name="hamiltonian_fingerprint_sha256",
        )
        scaffold_sha = _sha256_text(
            self.ordered_scaffold_fingerprint_sha256,
            field_name="ordered_scaffold_fingerprint_sha256",
        )
        provenance_schema = str(self.provenance_schema).strip()
        provenance_id = str(self.provenance_id).strip()
        if not provenance_schema:
            raise ValueError("external receipt provenance_schema must be nonempty.")
        if not provenance_id:
            raise ValueError("external receipt provenance_id must be nonempty.")
        source_primitive_ids = tuple(
            sorted({str(value).strip() for value in self.source_primitive_ids})
        )
        if any(not value for value in source_primitive_ids):
            raise ValueError(
                "external receipt source_primitive_ids must be nonempty strings."
            )
        provenance = _portable_payload(self.provenance_payload)
        if not isinstance(provenance, Mapping):
            raise TypeError("external receipt provenance_payload must be a mapping.")
        # Validate JSON safety now, rather than after the run reaches telemetry.
        json.dumps(
            provenance,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        receipt_payload = {
            "schema": EXTERNAL_LOGICAL_FS_GRAM_RECEIPT_SCHEMA,
            "logical_gram_sha256": _array_sha256(gram),
            "origin_state_sha256": _array_sha256(state),
            "origin_energy": energy,
            "origin_gradient_sha256": _array_sha256(gradient),
            "origin_logical_theta_sha256": _array_sha256(logical_theta),
            "origin_runtime_theta_sha256": _array_sha256(runtime_theta),
            "coordinate_registry": list(registry),
            "layout_fingerprint_sha256": layout_sha,
            "coordinate_registry_fingerprint_sha256": registry_sha,
            "hamiltonian_fingerprint_sha256": hamiltonian_sha,
            "ordered_scaffold_fingerprint_sha256": scaffold_sha,
            "provenance_schema": provenance_schema,
            "provenance_id": provenance_id,
            "source_primitive_ids": list(source_primitive_ids),
            "provenance_payload": provenance,
        }
        computed_receipt_id = _json_sha256(receipt_payload)
        supplied_receipt_id = str(self.receipt_id).strip().lower()
        if supplied_receipt_id and supplied_receipt_id != computed_receipt_id:
            raise ValueError("external logical FS Gram receipt id mismatch.")

        object.__setattr__(self, "coordinate_registry", registry)
        object.__setattr__(self, "logical_gram", _readonly_array(gram, dtype=float))
        object.__setattr__(
            self, "origin_gradient", _readonly_array(gradient, dtype=float)
        )
        object.__setattr__(
            self,
            "origin_logical_theta",
            _readonly_array(logical_theta, dtype=float),
        )
        object.__setattr__(
            self,
            "origin_runtime_theta",
            _readonly_array(runtime_theta, dtype=float),
        )
        object.__setattr__(
            self, "origin_state", _readonly_array(state, dtype=complex)
        )
        object.__setattr__(self, "origin_energy", energy)
        object.__setattr__(self, "layout_fingerprint_sha256", layout_sha)
        object.__setattr__(
            self, "coordinate_registry_fingerprint_sha256", registry_sha
        )
        object.__setattr__(
            self, "hamiltonian_fingerprint_sha256", hamiltonian_sha
        )
        object.__setattr__(
            self, "ordered_scaffold_fingerprint_sha256", scaffold_sha
        )
        object.__setattr__(self, "provenance_schema", provenance_schema)
        object.__setattr__(self, "provenance_id", provenance_id)
        object.__setattr__(
            self, "source_primitive_ids", source_primitive_ids
        )
        object.__setattr__(
            self, "provenance_payload", _immutable_payload(provenance)
        )
        object.__setattr__(self, "receipt_id", computed_receipt_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "receipt_id": str(self.receipt_id),
            "logical_parameter_count": len(self.coordinate_registry),
            "runtime_parameter_count": int(self.origin_runtime_theta.size),
            "statevector_dimension": int(self.origin_state.size),
            "logical_gram_sha256": _array_sha256(self.logical_gram),
            "origin_state_sha256": _array_sha256(self.origin_state),
            "origin_energy": float(self.origin_energy),
            "origin_gradient_sha256": _array_sha256(self.origin_gradient),
            "origin_logical_theta_sha256": _array_sha256(
                self.origin_logical_theta
            ),
            "origin_runtime_theta_sha256": _array_sha256(
                self.origin_runtime_theta
            ),
            "coordinate_registry": list(self.coordinate_registry),
            "layout_fingerprint_sha256": str(
                self.layout_fingerprint_sha256
            ),
            "coordinate_registry_fingerprint_sha256": str(
                self.coordinate_registry_fingerprint_sha256
            ),
            "hamiltonian_fingerprint_sha256": str(
                self.hamiltonian_fingerprint_sha256
            ),
            "ordered_scaffold_fingerprint_sha256": str(
                self.ordered_scaffold_fingerprint_sha256
            ),
            "provenance_schema": str(self.provenance_schema),
            "provenance_id": str(self.provenance_id),
            "source_primitive_ids": list(self.source_primitive_ids),
            "provenance_payload": _portable_payload(self.provenance_payload),
        }


def _validate_external_logical_fs_gram_receipt(
    receipt: ExternalLogicalFSGramReceipt,
    *,
    adapter: Any,
    supplied_runtime_theta: np.ndarray,
    h_compiled: CompiledPolynomialAction,
) -> None:
    """Fail closed before an external metric can enter the Powell chart."""

    if not isinstance(receipt, ExternalLogicalFSGramReceipt):
        raise TypeError(
            "external_logical_fs_gram_receipt must be an "
            "ExternalLogicalFSGramReceipt."
        )
    summary = adapter.summary
    registry = tuple(adapter.coordinate_registry)
    logical_count = int(adapter.x0.size)
    runtime_count = int(supplied_runtime_theta.size)
    expected_state_dimension = 1 << int(h_compiled.nq)
    if tuple(receipt.coordinate_registry) != registry:
        raise ValueError(
            "external logical FS Gram receipt coordinate registry mismatch."
        )
    comparisons = (
        (
            "layout fingerprint",
            receipt.layout_fingerprint_sha256,
            summary.get("layout_sha256"),
        ),
        (
            "coordinate-registry fingerprint",
            receipt.coordinate_registry_fingerprint_sha256,
            summary.get("coordinate_registry_sha256"),
        ),
        (
            "Hamiltonian fingerprint",
            receipt.hamiltonian_fingerprint_sha256,
            summary.get("hamiltonian_fingerprint"),
        ),
        (
            "ordered-scaffold fingerprint",
            receipt.ordered_scaffold_fingerprint_sha256,
            summary.get("ordered_scaffold_fingerprint"),
        ),
    )
    for label, observed, expected in comparisons:
        if str(observed) != str(expected):
            raise ValueError(
                f"external logical FS Gram receipt {label} mismatch."
            )
    if receipt.logical_gram.shape != (logical_count, logical_count):
        raise ValueError(
            "external logical FS Gram receipt metric dimension mismatch."
        )
    if receipt.origin_gradient.shape != (logical_count,):
        raise ValueError(
            "external logical FS Gram receipt gradient dimension mismatch."
        )
    if receipt.origin_logical_theta.shape != (logical_count,):
        raise ValueError(
            "external logical FS Gram receipt logical-theta dimension mismatch."
        )
    if receipt.origin_runtime_theta.shape != (runtime_count,):
        raise ValueError(
            "external logical FS Gram receipt runtime-theta dimension mismatch."
        )
    if receipt.origin_state.shape != (expected_state_dimension,):
        raise ValueError(
            "external logical FS Gram receipt state dimension mismatch."
        )
    expected_logical = np.asarray(adapter.x0, dtype=float).reshape(-1)
    expected_runtime = np.asarray(
        adapter.lift_to_runtime(expected_logical), dtype=float
    ).reshape(-1)
    if _array_sha256(receipt.origin_logical_theta) != _array_sha256(
        expected_logical
    ):
        raise ValueError(
            "external logical FS Gram receipt logical-theta fingerprint mismatch."
        )
    if _array_sha256(receipt.origin_runtime_theta) != _array_sha256(
        supplied_runtime_theta
    ):
        raise ValueError(
            "external logical FS Gram receipt supplied runtime-theta fingerprint "
            "mismatch."
        )
    if _array_sha256(receipt.origin_runtime_theta) != _array_sha256(
        expected_runtime
    ):
        raise ValueError(
            "external logical FS Gram receipt lifted runtime-theta fingerprint "
            "mismatch."
        )


@dataclass(frozen=True)
class AcceptedRefitConfig:
    """Typed accepted-refit controls with legacy-safe defaults."""

    scope: str = ACCEPTED_REFIT_SCOPE_SELECTOR_POLICY_V1
    coordinate_chart: str = ACCEPTED_REFIT_CHART_NATIVE_V1
    base_chart_policy: str = SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
    supported_metric: JointLinearSolveConfig = JointLinearSolveConfig()

    def __post_init__(self) -> None:
        scope = str(self.scope).strip().lower()
        chart = str(self.coordinate_chart).strip().lower()
        base_chart = str(self.base_chart_policy).strip().lower()
        if scope not in ACCEPTED_REFIT_SCOPE_CHOICES:
            raise ValueError(
                "accepted refit scope must be one of "
                f"{list(ACCEPTED_REFIT_SCOPE_CHOICES)}."
            )
        if chart not in ACCEPTED_REFIT_CHART_CHOICES:
            raise ValueError(
                "accepted refit coordinate chart must be one of "
                f"{list(ACCEPTED_REFIT_CHART_CHOICES)}."
            )
        if base_chart not in ACCEPTED_REFIT_BASE_CHART_CHOICES:
            raise ValueError(
                "accepted refit base chart must be one of "
                f"{list(ACCEPTED_REFIT_BASE_CHART_CHOICES)}."
            )
        if not isinstance(self.supported_metric, JointLinearSolveConfig):
            raise TypeError("supported_metric must be JointLinearSolveConfig.")
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "coordinate_chart", chart)
        object.__setattr__(self, "base_chart_policy", base_chart)

    @property
    def full_ansatz(self) -> bool:
        return self.scope == ACCEPTED_REFIT_SCOPE_FULL_ANSATZ_V1

    @property
    def supported_fs_whitened(self) -> bool:
        return (
            self.coordinate_chart
            == ACCEPTED_REFIT_CHART_SUPPORTED_FS_WHITENED_FIXED_V1
        )

    @property
    def diagnostic_active(self) -> bool:
        return bool(self.full_ansatz or self.supported_fs_whitened)

    def resolve_logical_indices(
        self,
        *,
        selector_active_indices: Sequence[int],
        logical_parameter_count: int,
    ) -> tuple[int, ...]:
        count = int(logical_parameter_count)
        if count < 0:
            raise ValueError("logical_parameter_count must be nonnegative.")
        selector = tuple(int(value) for value in selector_active_indices)
        if len(set(selector)) != len(selector):
            raise ValueError("selector_active_indices must be unique.")
        if any(value < 0 or value >= count for value in selector):
            raise ValueError("selector_active_indices contain an out-of-range value.")
        if self.full_ansatz:
            return tuple(range(count))
        return selector

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ACCEPTED_REFIT_CONFIG_SCHEMA,
            "scope": str(self.scope),
            "coordinate_chart": str(self.coordinate_chart),
            "base_chart_policy": str(self.base_chart_policy),
            "base_chart_applied": (
                str(self.base_chart_policy)
                if self.supported_fs_whitened
                else None
            ),
            "full_ansatz": bool(self.full_ansatz),
            "supported_fs_whitened": bool(self.supported_fs_whitened),
            "supported_metric": self.supported_metric.as_dict(),
        }


@dataclass(frozen=True)
class AcceptedRefitFixedChartReceipt:
    """Immutable identity of the chart held fixed for one optimizer call."""

    scope: str
    coordinate_chart: str
    base_chart_policy: str
    manifold_id: str
    construction_hashes: Mapping[str, str]
    support_factorization_provenance_id: str
    support_receipt_provenance_id: str
    external_gram_receipt_id: str | None
    sha256: str = ""
    schema: str = ACCEPTED_REFIT_FIXED_CHART_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ACCEPTED_REFIT_FIXED_CHART_RECEIPT_SCHEMA:
            raise ValueError("accepted-refit fixed-chart schema mismatch.")
        if self.scope not in ACCEPTED_REFIT_SCOPE_CHOICES:
            raise ValueError("accepted-refit fixed-chart scope is invalid.")
        if self.coordinate_chart not in ACCEPTED_REFIT_CHART_CHOICES:
            raise ValueError(
                "accepted-refit fixed-chart coordinate chart is invalid."
            )
        if self.base_chart_policy not in ACCEPTED_REFIT_BASE_CHART_CHOICES:
            raise ValueError(
                "accepted-refit fixed-chart base chart is invalid."
            )
        if not str(self.manifold_id):
            raise ValueError(
                "accepted-refit fixed-chart manifold id must be nonempty."
            )
        hashes = {
            str(key): _sha256_text(value, field_name=str(key))
            for key, value in self.construction_hashes.items()
        }
        if not hashes:
            raise ValueError(
                "accepted-refit fixed-chart construction hashes are empty."
            )
        object.__setattr__(
            self,
            "construction_hashes",
            MappingProxyType(hashes),
        )
        object.__setattr__(
            self,
            "support_factorization_provenance_id",
            _sha256_text(
                self.support_factorization_provenance_id,
                field_name="support_factorization_provenance_id",
            ),
        )
        object.__setattr__(
            self,
            "support_receipt_provenance_id",
            _sha256_text(
                self.support_receipt_provenance_id,
                field_name="support_receipt_provenance_id",
            ),
        )
        if self.external_gram_receipt_id is not None and not str(
            self.external_gram_receipt_id
        ):
            raise ValueError(
                "external_gram_receipt_id must be nonempty when present."
            )
        expected = _json_sha256(self.digest_payload())
        if self.sha256:
            if _sha256_text(
                self.sha256,
                field_name="accepted_refit_fixed_chart_sha256",
            ) != expected:
                raise ValueError(
                    "accepted-refit fixed-chart receipt digest mismatch."
                )
        else:
            object.__setattr__(self, "sha256", expected)

    def digest_payload(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "scope": str(self.scope),
            "coordinate_chart": str(self.coordinate_chart),
            "base_chart_policy": str(self.base_chart_policy),
            "manifold_id": str(self.manifold_id),
            "construction_hashes": dict(self.construction_hashes),
            "support_factorization_provenance_id": str(
                self.support_factorization_provenance_id
            ),
            "support_receipt_provenance_id": str(
                self.support_receipt_provenance_id
            ),
            "external_gram_receipt_id": self.external_gram_receipt_id,
            "chart_lifetime": (
                "fixed_for_one_optimizer_invocation_then_discarded_v1"
            ),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.digest_payload(),
            "sha256": str(self.sha256),
        }


@dataclass(frozen=True)
class SupportedFSPowellChart:
    """A fixed raw-FS-orthonormal Powell chart at one accepted endpoint."""

    objective: Callable[[np.ndarray], float]
    x0: np.ndarray
    lift_to_runtime: Callable[[np.ndarray], np.ndarray]
    coordinate_mode: str
    active_logical_indices: tuple[int, ...]
    active_runtime_indices: tuple[int, ...]
    active_optimizer_indices: tuple[int, ...]
    reduced_positions_by_logical: Mapping[int, tuple[int, ...]]
    origin_state: np.ndarray
    origin_logical_theta: np.ndarray
    origin_runtime_theta: np.ndarray
    whitened_to_logical_map: np.ndarray
    logical_to_whitened_map: np.ndarray
    coordinate_registry: tuple[str, ...]
    base_telemetry: Mapping[str, Any]

    def result_telemetry(
        self,
        *,
        optimizer_x: Sequence[float] | np.ndarray,
        final_runtime_theta: Sequence[float] | np.ndarray,
        final_energy: float,
    ) -> dict[str, Any]:
        x = np.asarray(optimizer_x, dtype=float).reshape(-1)
        if int(x.size) != int(self.x0.size):
            raise ValueError("optimizer_x length does not match the supported rank.")
        runtime = np.asarray(final_runtime_theta, dtype=float).reshape(-1)
        mapped_runtime = np.asarray(self.lift_to_runtime(x), dtype=float).reshape(-1)
        if runtime.shape != mapped_runtime.shape or not np.allclose(
            runtime,
            mapped_runtime,
            rtol=0.0,
            atol=2.0e-12,
        ):
            raise ValueError("final runtime theta does not match the whitened chart map.")
        logical = np.asarray(
            self.origin_logical_theta + self.whitened_to_logical_map @ x,
            dtype=float,
        )
        return {
            **dict(self.base_telemetry),
            "optimizer_displacement_whitened": [float(value) for value in x],
            "optimizer_displacement_norm": float(np.linalg.norm(x)),
            "final_logical_theta": [float(value) for value in logical],
            "final_runtime_theta": [float(value) for value in runtime],
            "final_runtime_theta_sha256": _array_sha256(runtime),
            "final_energy": float(final_energy),
        }


def map_phase_order_joint_step_to_supported_fs(
    *,
    chart: SupportedFSPowellChart,
    phase_order_joint_step: Sequence[float] | np.ndarray,
    phase3_to_post_logical_permutation: Sequence[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map one applied Phase-III joint step into the fixed Powell chart.

    ``phase3_to_post_logical_permutation[source]`` gives the post-admission
    logical index of the corresponding Phase-III coordinate.  The returned
    point is a displacement in the already fixed supported-FS chart; no chart
    factorization or quantum measurement is repeated here.
    """

    if not isinstance(chart, SupportedFSPowellChart):
        raise TypeError("chart must be a SupportedFSPowellChart.")
    phase_step = np.asarray(
        phase_order_joint_step, dtype=float
    ).reshape(-1)
    permutation = tuple(
        int(value) for value in phase3_to_post_logical_permutation
    )
    logical_count = int(chart.origin_logical_theta.size)
    supported_rank = int(chart.x0.size)
    if (
        phase_step.size != logical_count
        or len(permutation) != logical_count
        or tuple(sorted(permutation)) != tuple(range(logical_count))
    ):
        raise ValueError(
            "Phase-III joint-step permutation must cover the complete "
            "post-admission logical chart exactly once."
        )
    if not np.all(np.isfinite(phase_step)):
        raise ValueError("Phase-III joint step must be finite.")
    whitened_to_logical = np.asarray(
        chart.whitened_to_logical_map, dtype=float
    )
    logical_to_whitened = np.asarray(
        chart.logical_to_whitened_map, dtype=float
    )
    incumbent = np.asarray(chart.x0, dtype=float).reshape(-1)
    if (
        whitened_to_logical.shape != (logical_count, supported_rank)
        or logical_to_whitened.shape != (supported_rank, logical_count)
        or incumbent.shape != (supported_rank,)
        or not np.all(np.isfinite(whitened_to_logical))
        or not np.all(np.isfinite(logical_to_whitened))
        or not np.all(np.isfinite(incumbent))
    ):
        raise ValueError("Supported-FS chart maps are malformed or nonfinite.")
    post_logical_step = np.zeros(logical_count, dtype=float)
    for source_index, post_index in enumerate(permutation):
        post_logical_step[int(post_index)] = float(
            phase_step[int(source_index)]
        )
    whitened_displacement = np.asarray(
        logical_to_whitened @ post_logical_step,
        dtype=float,
    )
    supported_logical_step = np.asarray(
        whitened_to_logical @ whitened_displacement,
        dtype=float,
    )
    discarded_logical_step = np.asarray(
        post_logical_step - supported_logical_step,
        dtype=float,
    )
    mapped_x0 = np.asarray(
        incumbent + whitened_displacement,
        dtype=float,
    )
    if not (
        np.all(np.isfinite(whitened_displacement))
        and np.all(np.isfinite(supported_logical_step))
        and np.all(np.isfinite(mapped_x0))
    ):
        raise ValueError("Supported-FS joint-step map produced nonfinite data.")
    logical_scale = float(max(1.0, np.linalg.norm(post_logical_step)))
    projection_tolerance = float(
        max(
            1.0e-10,
            4096.0
            * np.finfo(float).eps
            * max(1, logical_count, supported_rank)
            * logical_scale,
        )
    )
    discarded_norm = float(np.linalg.norm(discarded_logical_step))
    receipt = {
        "schema": SUPPORTED_FS_JOINT_STEP_MAP_RECEIPT_SCHEMA,
        "source_coordinate_order": "phase3_active_then_selected_batch_v1",
        "target_coordinate_order": "post_admission_logical_v1",
        "phase3_to_post_logical_permutation": [
            int(value) for value in permutation
        ],
        "logical_parameter_count": int(logical_count),
        "supported_rank": int(supported_rank),
        "phase_order_joint_step": [
            float(value) for value in phase_step.tolist()
        ],
        "requested_post_logical_step": [
            float(value) for value in post_logical_step.tolist()
        ],
        "supported_post_logical_step": [
            float(value) for value in supported_logical_step.tolist()
        ],
        "discarded_null_logical_step": [
            float(value) for value in discarded_logical_step.tolist()
        ],
        "discarded_null_logical_step_norm": float(discarded_norm),
        "projection_tolerance": float(projection_tolerance),
        "source_step_within_supported_chart": bool(
            discarded_norm <= projection_tolerance
        ),
        "whitened_displacement": [
            float(value) for value in whitened_displacement.tolist()
        ],
        "mapped_optimizer_x0": [
            float(value) for value in mapped_x0.tolist()
        ],
        "whitened_to_logical_map_sha256": _array_sha256(
            whitened_to_logical
        ),
        "logical_to_whitened_map_sha256": _array_sha256(
            logical_to_whitened
        ),
        "classical_quantum_query_charge": 0,
    }
    return mapped_x0, receipt


def build_supported_fs_powell_chart(
    *,
    executor: CompiledAnsatzExecutor,
    layout: AnsatzParameterLayout,
    theta_runtime: Sequence[float] | np.ndarray,
    psi_ref: Sequence[complex] | np.ndarray,
    h_compiled: CompiledPolynomialAction,
    runtime_objective: Callable[[np.ndarray], float],
    config: AcceptedRefitConfig,
    manifold_id: str,
    external_logical_fs_gram_receipt: ExternalLogicalFSGramReceipt | None = None,
) -> SupportedFSPowellChart:
    """Build one fixed supported raw-FS chart for a Powell invocation."""

    if not config.supported_fs_whitened:
        raise ValueError("supported-FS chart requested with a native refit config.")
    if str(executor.parameterization_mode) != "logical_shared":
        raise ValueError(
            "supported-FS accepted refit currently requires logical_shared "
            "parameterization."
        )
    adapter = build_compiled_exact_manifold_adapter(
        executor=executor,
        layout=layout,
        theta_runtime=np.asarray(theta_runtime, dtype=float),
        psi_ref=np.asarray(psi_ref, dtype=complex),
        h_compiled=h_compiled,
        manifold_id=str(manifold_id),
    )
    logical_count = int(layout.logical_parameter_count)
    runtime_count = int(layout.runtime_parameter_count)
    supplied_runtime = np.asarray(theta_runtime, dtype=float).reshape(-1).copy()
    if external_logical_fs_gram_receipt is None:
        evaluation = adapter.backend.evaluate(adapter.x0)
        tangents = np.asarray(evaluation.tangents, dtype=complex)
        logical_gram = _fubini_study_gram(
            np.asarray(evaluation.statevector, dtype=complex),
            tangents,
        )
        origin_state = np.asarray(evaluation.statevector, dtype=complex)
        origin_energy = float(evaluation.energy)
        origin_gradient = np.asarray(evaluation.gradient, dtype=float)
        metric_input_status = "acquired"
        metric_input_mode = "exact_backend_evaluation_v1"
        metric_evaluation_provenance = dict(evaluation.metadata)
        metric_evaluation_provenance[
            "metric_tensor_convention"
        ] = "horizontal_projective_fubini_study_v1"
        external_receipt_telemetry = None
    else:
        _validate_external_logical_fs_gram_receipt(
            external_logical_fs_gram_receipt,
            adapter=adapter,
            supplied_runtime_theta=supplied_runtime,
            h_compiled=h_compiled,
        )
        logical_gram = np.asarray(
            external_logical_fs_gram_receipt.logical_gram, dtype=float
        ).copy()
        logical_gram = 0.5 * (logical_gram + logical_gram.T)
        origin_state = np.asarray(
            external_logical_fs_gram_receipt.origin_state, dtype=complex
        )
        origin_energy = float(external_logical_fs_gram_receipt.origin_energy)
        origin_gradient = np.asarray(
            external_logical_fs_gram_receipt.origin_gradient, dtype=float
        )
        metric_input_status = "reused"
        metric_input_mode = "external_logical_fs_gram_receipt_v1"
        external_receipt_telemetry = (
            external_logical_fs_gram_receipt.as_dict()
        )
        metric_evaluation_provenance = {
            "schema": EXTERNAL_LOGICAL_FS_GRAM_RECEIPT_SCHEMA,
            "metric_tensor_convention": (
                "horizontal_projective_fubini_study_v1"
            ),
            "receipt_id": str(external_logical_fs_gram_receipt.receipt_id),
            "provenance_schema": str(
                external_logical_fs_gram_receipt.provenance_schema
            ),
            "provenance_id": str(
                external_logical_fs_gram_receipt.provenance_id
            ),
            "provenance_payload": _portable_payload(
                external_logical_fs_gram_receipt.provenance_payload
            ),
        }
    base_chart_policy = str(config.base_chart_policy)
    if (
        base_chart_policy
        == SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
    ):
        base_to_logical = np.eye(logical_count, dtype=float)
        base_coordinate_registry = tuple(adapter.coordinate_registry)
        base_coordinate_kind = "logical_shared_reduced"
    else:
        base_to_logical = np.zeros((logical_count, runtime_count), dtype=float)
        expanded_registry: list[str] = []
        for block, logical_coordinate_id in zip(
            layout.blocks, adapter.coordinate_registry
        ):
            runtime_block_count = int(block.runtime_count)
            if runtime_block_count <= 0:
                raise ValueError(
                    "expanded-runtime projected logical chart cannot represent "
                    f"an empty block at logical_index={int(block.logical_index)}."
                )
            for runtime_index in range(
                int(block.runtime_start), int(block.runtime_stop)
            ):
                base_to_logical[int(block.logical_index), runtime_index] = (
                    1.0 / float(runtime_block_count)
                )
                expanded_registry.append(
                    f"{logical_coordinate_id}:projected_runtime:{runtime_index}"
                )
        base_coordinate_registry = tuple(expanded_registry)
        base_coordinate_kind = "expanded_runtime_projected_logical"
    base_gram = np.asarray(
        base_to_logical.T @ logical_gram @ base_to_logical,
        dtype=float,
    )
    base_gram = 0.5 * (base_gram + base_gram.T)
    retained_support = ra_support.factor_retained_support(
        base_gram,
        rank_relative_tolerance=float(
            config.supported_metric.rank_relative_tolerance
        ),
        metric_regularization=float(config.supported_metric.metric_regularization),
        source_provenance_id=(
            "accepted_refit_full_post_admission_gram:"
            + str(manifold_id)
            + ":"
            + base_chart_policy
        ),
    )
    factor = retained_support.factorization
    if not factor.feasible or int(factor.rank) <= 0:
        raise RuntimeError(
            "accepted-refit raw FS metric has no usable supported subspace: "
            f"{factor.reason}."
        )

    origin_logical = np.asarray(adapter.x0, dtype=float).reshape(-1)
    origin_runtime = np.asarray(
        adapter.lift_to_runtime(origin_logical), dtype=float
    ).reshape(-1)
    if supplied_runtime.shape != origin_runtime.shape or not np.allclose(
        supplied_runtime,
        origin_runtime,
        rtol=0.0,
        atol=2.0e-12,
    ):
        raise ValueError(
            "accepted-refit inherited runtime point is not a uniform logical "
            "alias and cannot define an exact fixed chart origin."
        )
    whitened_to_base = np.asarray(
        factor.raw_orthonormalizer,
        dtype=float,
    )
    whitened_to_logical = np.asarray(
        base_to_logical @ whitened_to_base,
        dtype=float,
    )
    logical_to_whitened = np.asarray(
        np.linalg.pinv(whitened_to_logical, rcond=1.0e-12),
        dtype=float,
    )
    raw_metric_in_chart = np.asarray(
        whitened_to_base.T @ base_gram @ whitened_to_base,
        dtype=float,
    )
    identity_residual = float(
        np.linalg.norm(raw_metric_in_chart - np.eye(int(factor.rank)), ord="fro")
    )
    if not math.isfinite(identity_residual) or identity_residual > 5.0e-8:
        raise FloatingPointError(
            "accepted-refit raw FS chart failed orthonormality: "
            f"residual={identity_residual}."
        )

    def _lift(value: np.ndarray) -> np.ndarray:
        displacement = np.asarray(value, dtype=float).reshape(-1)
        if int(displacement.size) != int(factor.rank):
            raise ValueError(
                "whitened Powell coordinate length must match supported rank."
            )
        logical = origin_logical + whitened_to_logical @ displacement
        return np.asarray(adapter.lift_to_runtime(logical), dtype=float)

    def _objective(value: np.ndarray) -> float:
        return float(runtime_objective(_lift(np.asarray(value, dtype=float))))

    active_logical = tuple(range(logical_count))
    active_runtime = tuple(range(runtime_count))
    optimizer_indices = tuple(range(int(factor.rank)))
    map_tol = 64.0 * np.finfo(float).eps
    positions_by_logical = {
        int(logical_index): tuple(
            int(position)
            for position in optimizer_indices
            if abs(float(whitened_to_logical[logical_index, position])) > map_tol
        )
        for logical_index in active_logical
    }
    fixed_chart_receipt = AcceptedRefitFixedChartReceipt(
        scope=str(config.scope),
        coordinate_chart=str(config.coordinate_chart),
        base_chart_policy=base_chart_policy,
        manifold_id=str(manifold_id),
        construction_hashes={
            "origin_state_sha256": _array_sha256(origin_state),
            "origin_logical_theta_sha256": _array_sha256(
                origin_logical
            ),
            "origin_runtime_theta_sha256": _array_sha256(
                origin_runtime
            ),
            "logical_coordinate_registry_sha256": _json_sha256(
                list(adapter.coordinate_registry)
            ),
            "base_coordinate_registry_sha256": _json_sha256(
                list(base_coordinate_registry)
            ),
            "raw_logical_fs_metric_sha256": _array_sha256(
                logical_gram
            ),
            "raw_base_metric_sha256": _array_sha256(base_gram),
            "base_to_logical_map_sha256": _array_sha256(
                base_to_logical
            ),
            "whitened_to_base_map_sha256": _array_sha256(
                whitened_to_base
            ),
            "whitened_to_logical_map_sha256": _array_sha256(
                whitened_to_logical
            ),
            "logical_to_whitened_map_sha256": _array_sha256(
                logical_to_whitened
            ),
            "raw_metric_in_powell_chart_sha256": _array_sha256(
                raw_metric_in_chart
            ),
            "metric_evaluation_provenance_sha256": _json_sha256(
                metric_evaluation_provenance
            ),
        },
        support_factorization_provenance_id=str(
            retained_support.receipt.factorization_provenance_id
        ),
        support_receipt_provenance_id=str(
            retained_support.receipt.receipt_provenance_id
        ),
        external_gram_receipt_id=(
            None
            if external_logical_fs_gram_receipt is None
            else str(external_logical_fs_gram_receipt.receipt_id)
        ),
    )
    telemetry = {
        "schema": SUPPORTED_FS_POWELL_CHART_SCHEMA,
        "policy": ACCEPTED_REFIT_CHART_SUPPORTED_FS_WHITENED_FIXED_V1,
        "base_chart_policy": base_chart_policy,
        "base_coordinate_kind": base_coordinate_kind,
        "chart_fixed_within_powell_invocation": True,
        "chart_recomputed_after_next_admission": True,
        "origin_kind": "inherited_zero_growth_state_v1",
        "origin_energy": float(origin_energy),
        "origin_logical_theta": [float(value) for value in origin_logical],
        "origin_runtime_theta": [float(value) for value in origin_runtime],
        "supplied_runtime_theta": [float(value) for value in supplied_runtime],
        "origin_logical_theta_sha256": _array_sha256(origin_logical),
        "origin_runtime_theta_sha256": _array_sha256(origin_runtime),
        "origin_state_sha256": _array_sha256(origin_state),
        "parameterization_mode": str(adapter.backend.parameterization_mode),
        "logical_coordinate_registry": list(adapter.coordinate_registry),
        "base_coordinate_registry": list(base_coordinate_registry),
        "base_coordinate_registry_sha256": _json_sha256(
            list(base_coordinate_registry)
        ),
        "logical_parameter_count": logical_count,
        "runtime_parameter_count": runtime_count,
        "base_parameter_count": int(base_gram.shape[0]),
        "supported_rank": int(factor.rank),
        "base_to_logical_map": base_to_logical.tolist(),
        "whitened_to_base_map": whitened_to_base.tolist(),
        "whitened_to_logical_map": whitened_to_logical.tolist(),
        "logical_to_whitened_map": logical_to_whitened.tolist(),
        "raw_logical_fs_metric": logical_gram.tolist(),
        "raw_logical_energy_gradient": [
            float(value) for value in origin_gradient
        ],
        "raw_base_metric": base_gram.tolist(),
        "raw_metric_in_powell_chart": raw_metric_in_chart.tolist(),
        "raw_metric_identity_residual": identity_residual,
        "metric_element_count": int(logical_count * (logical_count + 1) // 2),
        "metric_input_status": metric_input_status,
        "metric_input_mode": metric_input_mode,
        "metric_backend_evaluation_performed": (
            external_logical_fs_gram_receipt is None
        ),
        "metric_element_count_acquired_for_chart": (
            int(logical_count * (logical_count + 1) // 2)
            if external_logical_fs_gram_receipt is None
            else 0
        ),
        "metric_element_count_reused_for_chart": (
            0
            if external_logical_fs_gram_receipt is None
            else int(logical_count * (logical_count + 1) // 2)
        ),
        "external_logical_fs_gram_receipt": external_receipt_telemetry,
        "metric_evaluation_provenance": metric_evaluation_provenance,
        "classical_factorization_quantum_query_charge": 0,
        "retained_support_receipt": retained_support.receipt.as_dict(),
        "accepted_refit_fixed_chart_receipt": (
            fixed_chart_receipt.as_dict()
        ),
        "accepted_refit_fixed_chart_sha256": str(
            fixed_chart_receipt.sha256
        ),
        **factor.telemetry(),
    }
    return SupportedFSPowellChart(
        objective=_objective,
        x0=np.zeros(int(factor.rank), dtype=float),
        lift_to_runtime=_lift,
        coordinate_mode=f"supported_fs_whitened:{base_chart_policy}",
        active_logical_indices=active_logical,
        active_runtime_indices=active_runtime,
        active_optimizer_indices=optimizer_indices,
        reduced_positions_by_logical=positions_by_logical,
        origin_state=np.asarray(origin_state, dtype=complex).copy(),
        origin_logical_theta=origin_logical.copy(),
        origin_runtime_theta=origin_runtime.copy(),
        whitened_to_logical_map=whitened_to_logical.copy(),
        logical_to_whitened_map=logical_to_whitened.copy(),
        coordinate_registry=base_coordinate_registry,
        base_telemetry=telemetry,
    )


__all__ = [
    "ACCEPTED_REFIT_BASE_CHART_CHOICES",
    "ACCEPTED_REFIT_CHART_CHOICES",
    "ACCEPTED_REFIT_CHART_NATIVE_V1",
    "ACCEPTED_REFIT_CHART_SUPPORTED_FS_WHITENED_FIXED_V1",
    "ACCEPTED_REFIT_CONFIG_SCHEMA",
    "ACCEPTED_REFIT_FIXED_CHART_RECEIPT_SCHEMA",
    "ACCEPTED_REFIT_SCOPE_CHOICES",
    "ACCEPTED_REFIT_SCOPE_FULL_ANSATZ_V1",
    "ACCEPTED_REFIT_SCOPE_SELECTOR_POLICY_V1",
    "AcceptedRefitConfig",
    "AcceptedRefitFixedChartReceipt",
    "EXTERNAL_LOGICAL_FS_GRAM_RECEIPT_SCHEMA",
    "ExternalLogicalFSGramReceipt",
    "SUPPORTED_FS_POWELL_CHART_SCHEMA",
    "SUPPORTED_FS_JOINT_STEP_MAP_RECEIPT_SCHEMA",
    "SupportedFSPowellChart",
    "build_supported_fs_powell_chart",
    "map_phase_order_joint_step_to_supported_fs",
]
