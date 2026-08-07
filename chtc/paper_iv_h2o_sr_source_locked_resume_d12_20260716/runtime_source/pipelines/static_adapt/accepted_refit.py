"""Accepted-ansatz refit policies independent of selector geometry.

The static selector may use a local active window while the accepted ansatz is
refit in a different optimizer chart.  This module keeps that distinction
explicit.  Its supported-FS chart is fixed for one optimizer invocation and
uses the shared raw-metric support convention from ``joint_linear_solve``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pipelines.static_adapt.formal_manifold_exact_backend import (
    build_compiled_exact_manifold_adapter,
)
from pipelines.static_adapt.joint_linear_solve import (
    JointLinearSolveConfig,
    factor_supported_metric,
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
    evaluation = adapter.backend.evaluate(adapter.x0)
    tangents = np.asarray(evaluation.tangents, dtype=complex)
    logical_gram = np.asarray(
        np.real(np.conjugate(tangents).T @ tangents),
        dtype=float,
    )
    logical_gram = 0.5 * (logical_gram + logical_gram.T)
    logical_count = int(layout.logical_parameter_count)
    runtime_count = int(layout.runtime_parameter_count)
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
    factor = factor_supported_metric(
        base_gram,
        rank_relative_tolerance=float(
            config.supported_metric.rank_relative_tolerance
        ),
        metric_regularization=float(config.supported_metric.metric_regularization),
    )
    if not factor.feasible or int(factor.rank) <= 0:
        raise RuntimeError(
            "accepted-refit raw FS metric has no usable supported subspace: "
            f"{factor.reason}."
        )

    origin_logical = np.asarray(adapter.x0, dtype=float).reshape(-1)
    supplied_runtime = np.asarray(theta_runtime, dtype=float).reshape(-1).copy()
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
    telemetry = {
        "schema": SUPPORTED_FS_POWELL_CHART_SCHEMA,
        "policy": ACCEPTED_REFIT_CHART_SUPPORTED_FS_WHITENED_FIXED_V1,
        "base_chart_policy": base_chart_policy,
        "base_coordinate_kind": base_coordinate_kind,
        "chart_fixed_within_powell_invocation": True,
        "chart_recomputed_after_next_admission": True,
        "origin_kind": "inherited_zero_growth_state_v1",
        "origin_energy": float(evaluation.energy),
        "origin_logical_theta": [float(value) for value in origin_logical],
        "origin_runtime_theta": [float(value) for value in origin_runtime],
        "supplied_runtime_theta": [float(value) for value in supplied_runtime],
        "origin_logical_theta_sha256": _array_sha256(origin_logical),
        "origin_runtime_theta_sha256": _array_sha256(origin_runtime),
        "origin_state_sha256": _array_sha256(evaluation.statevector),
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
        "raw_base_metric": base_gram.tolist(),
        "raw_metric_in_powell_chart": raw_metric_in_chart.tolist(),
        "raw_metric_identity_residual": identity_residual,
        "metric_element_count": int(logical_count * (logical_count + 1) // 2),
        "metric_evaluation_provenance": dict(evaluation.metadata),
        "classical_factorization_quantum_query_charge": 0,
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
        origin_state=np.asarray(evaluation.statevector, dtype=complex).copy(),
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
    "ACCEPTED_REFIT_SCOPE_CHOICES",
    "ACCEPTED_REFIT_SCOPE_FULL_ANSATZ_V1",
    "ACCEPTED_REFIT_SCOPE_SELECTOR_POLICY_V1",
    "AcceptedRefitConfig",
    "SUPPORTED_FS_POWELL_CHART_SCHEMA",
    "SupportedFSPowellChart",
    "build_supported_fs_powell_chart",
]
