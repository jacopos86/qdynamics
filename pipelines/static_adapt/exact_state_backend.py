"""Neutral exact-state evaluation types and validation.

This module owns the exact-state callback interface shared by compiled
geometry, accepted refit, and retained compatibility routes.  It deliberately
contains no route controller, optimizer, or candidate-selection behavior.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


# Stable serialized identity retained for compatibility with committed
# formal-manifold endpoint receipts.  Ownership here prevents neutral selector
# geometry from importing the retired route implementation.
FORMAL_OUTER_EXACT_ANCHOR_SCHEMA = "formal_manifold_outer_exact_anchor_v1"


def finite_real_vector(value: Any, *, name: str) -> np.ndarray:
    """Return a copied, flattened finite real vector."""

    array = np.asarray(value, dtype=float).reshape(-1)
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must contain only finite real values.")
    return array.copy()


def finite_complex_array(value: Any, *, name: str) -> np.ndarray:
    """Return a copied finite complex array."""

    array = np.asarray(value, dtype=complex)
    if not bool(
        np.all(np.isfinite(array.real)) and np.all(np.isfinite(array.imag))
    ):
        raise ValueError(f"{name} must contain only finite complex values.")
    return array.copy()


# Private compatibility aliases used by the formal-manifold implementation.
_finite_real_vector = finite_real_vector
_finite_complex_array = finite_complex_array


@dataclass(frozen=True)
class ExactEnergyEvaluation:
    energy: float
    statevector: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExactGradientEvaluation:
    energy: float
    gradient: np.ndarray
    statevector: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExactStateEvaluation:
    energy: float
    gradient: np.ndarray
    statevector: np.ndarray
    tangents: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ExactStateBackend:
    """Validated exact-state callbacks with typed estimator boundaries.

    ``evaluate_fn`` is the full tangent/metric primitive retained for exact
    anchors. Sparse metric mode additionally requires distinct ``energy_fn``
    and ``gradient_fn`` callbacks so a trial or accepted gradient endpoint can
    never silently fall back to a full tangent evaluation.
    """

    def __init__(
        self,
        *,
        evaluate_fn: Callable[[np.ndarray], ExactStateEvaluation],
        energy_fn: Callable[[np.ndarray], ExactEnergyEvaluation] | None = None,
        gradient_fn: Callable[[np.ndarray], ExactGradientEvaluation] | None = None,
        coordinate_registry: Sequence[str],
        manifold_id: str,
        parameterization_mode: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not callable(evaluate_fn):
            raise TypeError("evaluate_fn must be callable.")
        registry = tuple(str(item) for item in coordinate_registry)
        if len(set(registry)) != len(registry):
            raise ValueError("coordinate_registry entries must be unique.")
        if not str(manifold_id).strip():
            raise ValueError("manifold_id must be non-empty.")
        self._evaluate_fn = evaluate_fn
        self._energy_fn = energy_fn
        self._gradient_fn = gradient_fn
        self.coordinate_registry = registry
        self.manifold_id = str(manifold_id)
        self.parameterization_mode = str(parameterization_mode)
        self.metadata = deepcopy(dict(metadata or {}))

    @property
    def supports_sparse_endpoint_primitives(self) -> bool:
        return bool(callable(self._energy_fn) and callable(self._gradient_fn))

    def _coordinate(self, theta: np.ndarray | Sequence[float]) -> np.ndarray:
        coordinate = finite_real_vector(theta, name="theta")
        if int(coordinate.size) != len(self.coordinate_registry):
            raise ValueError(
                "theta length does not match coordinate registry: "
                f"{coordinate.size} vs {len(self.coordinate_registry)}."
            )
        return coordinate

    @staticmethod
    def _validated_statevector(value: Any) -> np.ndarray:
        state = finite_complex_array(value, name="statevector").reshape(-1)
        norm = float(np.linalg.norm(state))
        if not np.isclose(norm, 1.0, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError(
                f"exact-state backend must return a normalized state; norm={norm}."
            )
        return state

    def evaluate_energy(
        self, theta: np.ndarray | Sequence[float]
    ) -> ExactEnergyEvaluation:
        coordinate = self._coordinate(theta)
        if self._energy_fn is None:
            full = self.evaluate(coordinate)
            return ExactEnergyEvaluation(
                energy=float(full.energy),
                statevector=np.asarray(full.statevector, dtype=complex).copy(),
                metadata={
                    **deepcopy(dict(full.metadata)),
                    "typed_primitive_fallback": "full_geometry_evaluation",
                },
            )
        raw = self._energy_fn(coordinate.copy())
        if not isinstance(raw, ExactEnergyEvaluation):
            raise TypeError("energy_fn must return ExactEnergyEvaluation.")
        energy = float(raw.energy)
        if not math.isfinite(energy):
            raise ValueError("exact-state energy must be finite.")
        return ExactEnergyEvaluation(
            energy=energy,
            statevector=self._validated_statevector(raw.statevector),
            metadata=deepcopy(dict(raw.metadata)),
        )

    def evaluate_gradient(
        self, theta: np.ndarray | Sequence[float]
    ) -> ExactGradientEvaluation:
        coordinate = self._coordinate(theta)
        if self._gradient_fn is None:
            full = self.evaluate(coordinate)
            return ExactGradientEvaluation(
                energy=float(full.energy),
                gradient=np.asarray(full.gradient, dtype=float).copy(),
                statevector=np.asarray(full.statevector, dtype=complex).copy(),
                metadata={
                    **deepcopy(dict(full.metadata)),
                    "typed_primitive_fallback": "full_geometry_evaluation",
                },
            )
        raw = self._gradient_fn(coordinate.copy())
        if not isinstance(raw, ExactGradientEvaluation):
            raise TypeError("gradient_fn must return ExactGradientEvaluation.")
        energy = float(raw.energy)
        if not math.isfinite(energy):
            raise ValueError("exact-state energy must be finite.")
        gradient = finite_real_vector(raw.gradient, name="gradient")
        if int(gradient.size) != int(coordinate.size):
            raise ValueError("gradient length must match the coordinate registry.")
        return ExactGradientEvaluation(
            energy=energy,
            gradient=gradient,
            statevector=self._validated_statevector(raw.statevector),
            metadata=deepcopy(dict(raw.metadata)),
        )

    def evaluate(self, theta: np.ndarray | Sequence[float]) -> ExactStateEvaluation:
        coordinate = self._coordinate(theta)
        raw = self._evaluate_fn(coordinate.copy())
        return self.validate_supplied_evaluation(coordinate, raw)

    def validate_supplied_evaluation(
        self,
        theta: np.ndarray | Sequence[float],
        evaluation: ExactStateEvaluation,
    ) -> ExactStateEvaluation:
        """Validate an exact endpoint supplied by a same-process receipt."""

        coordinate = self._coordinate(theta)
        raw = evaluation
        if not isinstance(raw, ExactStateEvaluation):
            raise TypeError("evaluation must be ExactStateEvaluation.")
        energy = float(raw.energy)
        if not math.isfinite(energy):
            raise ValueError("exact-state energy must be finite.")
        state = self._validated_statevector(raw.statevector)
        gradient = finite_real_vector(raw.gradient, name="gradient")
        if int(gradient.size) != int(coordinate.size):
            raise ValueError("gradient length must match the coordinate registry.")
        tangents_raw = finite_complex_array(raw.tangents, name="tangents")
        if tangents_raw.shape != (int(state.size), int(coordinate.size)):
            raise ValueError(
                "tangent matrix must have shape (state_dimension, coordinate_count), "
                f"got {tangents_raw.shape}."
            )
        overlaps = np.conjugate(state) @ tangents_raw
        tangents = tangents_raw - np.outer(state, overlaps)
        horizontal_residual = float(
            np.linalg.norm(np.conjugate(state) @ tangents)
        )
        metadata = {
            **deepcopy(dict(raw.metadata)),
            "horizontalization": "state_projector_v1",
            "horizontal_residual": horizontal_residual,
        }
        return ExactStateEvaluation(
            energy=energy,
            gradient=gradient,
            statevector=state,
            tangents=tangents,
            metadata=metadata,
        )


@runtime_checkable
class ExactOuterAnchorState(Protocol):
    """Structural interface consumed by retained exact-anchor reuse."""

    theta: np.ndarray
    registry: tuple[str, ...]
    manifold_id: str
    parameterization_mode: str
    statevector: np.ndarray
    frame_anchor_statevector: np.ndarray
    tangents: np.ndarray
    b: np.ndarray
    whitening_id: str
    frame_id: str
    logical_range_id: str
    rank: int
    outer_exact_anchor: Any


__all__ = [
    "ExactEnergyEvaluation",
    "ExactGradientEvaluation",
    "ExactOuterAnchorState",
    "ExactStateBackend",
    "ExactStateEvaluation",
    "FORMAL_OUTER_EXACT_ANCHOR_SCHEMA",
    "finite_complex_array",
    "finite_real_vector",
]
