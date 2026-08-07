"""Exact-state formal-manifold warm start for accepted ADAPT ansaetze.

This module implements the deterministic, constant-rank part of the formal
manifold proposal in ``formal_manifold_warm_start.md``.  It is deliberately a
reoptimization route, not a candidate-selection route.  The authoritative
metric is refreshed from analytic state tangents at every accepted endpoint;
the published qBroyden recurrence is retained only as labeled shadow
telemetry.  Objective curvature is an explicitly tagged intrinsic model in
the resolved tangent frame: either the default SPD inverse-RBFGS operator or a
diagnostic direct-SR1 operator globalized by the shared supported-metric
eigentrust kernel.  The representations are mutually exclusive authoritative
modes.

Hardware/statistical rank certification and Gram-only growth remain outside
this exact-state route and are reported as unsupported capabilities rather
than silently approximated.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
import uuid
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pipelines.static_adapt.joint_linear_solve import (
    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1,
    JointLinearSolveConfig,
    JointLinearSolveResult,
    SupportedMetricWhitening,
    factor_supported_metric,
    solve_joint_linear_model,
)
from pipelines.static_adapt.formal_manifold_route_profile import (
    FM_ROUTE_PROFILE_SR_PHASE2_PHASE3_WHITENED_ADAPTIVE_TRUST_NO_N2_V1,
    FM_ROUTE_PROFILE_SR_SOURCE_LOCKED_SUPPORTED_WHITENED_ADAPTIVE_TRUST_V1,
    FORMAL_MANIFOLD_ROUTE_FAMILY as PROFILE_FORMAL_MANIFOLD_ROUTE_FAMILY,
    FORMAL_MANIFOLD_ROUTE_PROFILE_CHOICES,
    FORMAL_MANIFOLD_ROUTE_PROFILE_OFF,
    FORMAL_MANIFOLD_SR_CANDIDATE_SELECTOR_FAMILY,
    FORMAL_MANIFOLD_SR_CANDIDATE_SELECTOR_PROFILE,
    FormalManifoldRouteProfile,
    resolve_formal_manifold_route_profile,
)
from pipelines.static_adapt.selector_query_closure import (
    FormalGrowthGeometryReceipt,
    GrowthReceiptExpectation,
    QueryPrimitiveLedger,
    projective_state_fingerprint,
    validate_formal_growth_geometry_receipt,
)


FORMAL_MANIFOLD_WARM_START_OFF = "off"
FORMAL_MANIFOLD_ROUTE = "formal_manifold_warm_start_v1"
FORMAL_MANIFOLD_WARM_START_ROUTE = FORMAL_MANIFOLD_ROUTE
FORMAL_MANIFOLD_ROUTE_FAMILY = "formal_manifold_snake"
FORMAL_MANIFOLD_SR_NO_N2_PROFILE = (
    FM_ROUTE_PROFILE_SR_PHASE2_PHASE3_WHITENED_ADAPTIVE_TRUST_NO_N2_V1
)
FORMAL_MANIFOLD_SR_SOURCE_LOCKED_PROFILE = (
    FM_ROUTE_PROFILE_SR_SOURCE_LOCKED_SUPPORTED_WHITENED_ADAPTIVE_TRUST_V1
)
FORMAL_MANIFOLD_SR_SELECTOR_FAMILY = (
    FORMAL_MANIFOLD_SR_CANDIDATE_SELECTOR_FAMILY
)
FORMAL_MANIFOLD_SR_SELECTOR_PROFILE = (
    FORMAL_MANIFOLD_SR_CANDIDATE_SELECTOR_PROFILE
)
FORMAL_MANIFOLD_ROUTE_COMPOSITION_SCHEMA = (
    "formal_manifold_route_composition_v1"
)

if FORMAL_MANIFOLD_ROUTE_FAMILY != PROFILE_FORMAL_MANIFOLD_ROUTE_FAMILY:
    raise RuntimeError("formal-manifold route-family constants disagree.")
FORMAL_CURVATURE_INVERSE_RBFGS = "inverse_rbfgs_raised_covariant_hessian_v1"
FORMAL_CURVATURE_DIRECT_SR1 = "direct_sr1_raised_covariant_hessian_v1"
FORMAL_CURVATURE_BRANCHES = frozenset(
    {FORMAL_CURVATURE_INVERSE_RBFGS, FORMAL_CURVATURE_DIRECT_SR1}
)
FORMAL_MANIFOLD_ROUTE_CHOICES = (
    FORMAL_MANIFOLD_WARM_START_OFF,
    FORMAL_MANIFOLD_ROUTE,
)

_FORMAL_MANIFOLD_SR_SELECTOR_MECHANISM_FIELDS = (
    "historical_singleton_coordinate_solve_policy",
    "historical_singleton_coordinate_solve_scope",
    "historical_singleton_trust_region_update_policy",
    "sr_powell_coordinate_chart_policy",
    "sr_escape_mode",
    "phase2_novelty_mode",
    "phase3_novelty_ablation_mode",
    "phase2_novelty_multiplier_policy",
    "phase3_novelty_multiplier_policy",
    "phase2_gram_novelty_policy",
    "phase3_gram_novelty_policy",
    "phase0_pilot_enabled",
    "phase2_enable_batching",
    "phase3_enable_batching",
    "structural_rollback_enabled",
    "route_a_funnel_active",
    "phase3_runtime_split_mode",
    "phase3_runtime_split_selection_mode",
    "phase3_runtime_split_subset_sizes",
    "phase3_runtime_split_child_set_symmetry_policy",
    "phase3_runtime_split_child_padding_policy",
    "candidate_response_model",
    "admission_cardinality",
    "prune_policy",
    "measured_n2_retained",
    "additional_n3_multiplier_applied",
)


def _registered_formal_manifold_route_profile(
    value: Any,
) -> FormalManifoldRouteProfile | None:
    """Resolve a registered composed profile, excluding the registry's off row."""

    route_profile = str(value)
    if route_profile == FORMAL_MANIFOLD_ROUTE_PROFILE_OFF:
        return None
    if route_profile not in FORMAL_MANIFOLD_ROUTE_PROFILE_CHOICES:
        return None
    return resolve_formal_manifold_route_profile(route_profile)


def normalize_reoptimization_route(value: str | None) -> str:
    key = str(value or FORMAL_MANIFOLD_WARM_START_OFF).strip().lower()
    aliases = {
        "": FORMAL_MANIFOLD_WARM_START_OFF,
        "none": FORMAL_MANIFOLD_WARM_START_OFF,
        "legacy": FORMAL_MANIFOLD_WARM_START_OFF,
        "manifold_qbroyd_rbfgs_exact_v1": FORMAL_MANIFOLD_ROUTE,
        FORMAL_MANIFOLD_ROUTE: FORMAL_MANIFOLD_ROUTE,
    }
    resolved = aliases.get(key, key)
    if resolved not in FORMAL_MANIFOLD_ROUTE_CHOICES:
        raise ValueError(
            "adapt_reoptimization_route must be one of "
            f"{set(FORMAL_MANIFOLD_ROUTE_CHOICES)}, got {value!r}."
        )
    return resolved


def _sym(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    return np.asarray(0.5 * (array + array.T), dtype=float)


def _finite_real_vector(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must contain only finite real values.")
    return array.copy()


def _finite_complex_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=complex)
    if not bool(np.all(np.isfinite(array.real)) and np.all(np.isfinite(array.imag))):
        raise ValueError(f"{name} must contain only finite complex values.")
    return array.copy()


def _json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _clone_query_ledger(ledger: QueryPrimitiveLedger) -> QueryPrimitiveLedger:
    """Clone a query ledger without sharing any mutable branch state."""

    if not isinstance(ledger, QueryPrimitiveLedger):
        raise TypeError("ledger must be a QueryPrimitiveLedger.")
    return QueryPrimitiveLedger.from_checkpoint_payload(
        ledger.checkpoint_payload()
    )


@dataclass(frozen=True)
class FormalManifoldRouteComposition:
    """Normalized FM outer-route and candidate-selector identity.

    The FM optimizer owns its manifold state.  The selector identity is stored
    beside it so an FM+SR composition cannot be mislabeled as an SR outer route
    or resume under a different selector profile.
    """

    route_family: str = FORMAL_MANIFOLD_ROUTE_FAMILY
    route_profile: str = FORMAL_MANIFOLD_ROUTE
    candidate_selector_family: str | None = None
    candidate_selector_profile: str | None = None
    adapt_reoptimization_route: str = FORMAL_MANIFOLD_ROUTE
    singleton_response_selector: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if str(self.route_family) != FORMAL_MANIFOLD_ROUTE_FAMILY:
            raise ValueError(
                "formal-manifold composition route_family must be "
                f"{FORMAL_MANIFOLD_ROUTE_FAMILY!r}."
            )
        if normalize_reoptimization_route(self.adapt_reoptimization_route) != (
            FORMAL_MANIFOLD_ROUTE
        ):
            raise ValueError(
                "formal-manifold composition requires the formal reoptimization route."
            )
        selector_family = (
            None
            if self.candidate_selector_family in {None, ""}
            else str(self.candidate_selector_family)
        )
        selector_profile = (
            None
            if self.candidate_selector_profile in {None, ""}
            else str(self.candidate_selector_profile)
        )
        if (selector_family is None) != (selector_profile is None):
            raise ValueError(
                "candidate selector family and profile must be declared together."
            )
        resolved_profile = _registered_formal_manifold_route_profile(
            self.route_profile
        )
        if resolved_profile is not None:
            if selector_family != resolved_profile.candidate_selector_family:
                raise ValueError(
                    f"the FM route profile {self.route_profile!r} requires "
                    "its resolved SR selector family."
                )
            if selector_profile != resolved_profile.candidate_selector_profile:
                raise ValueError(
                    f"the FM route profile {self.route_profile!r} requires "
                    "the resolved SR selector profile "
                    f"{resolved_profile.candidate_selector_profile!r}."
                )
        if not isinstance(self.singleton_response_selector, Mapping):
            raise TypeError("singleton_response_selector must be a mapping.")
        if resolved_profile is not None:
            expected = resolved_profile.as_dict()
            selector_payload = dict(self.singleton_response_selector)
            missing = [
                field_name
                for field_name in _FORMAL_MANIFOLD_SR_SELECTOR_MECHANISM_FIELDS
                if field_name not in selector_payload
            ]
            if missing:
                raise ValueError(
                    "the FM composition lacks resolved selector fields: "
                    + ", ".join(missing)
                )
            mismatched = [
                field_name
                for field_name in _FORMAL_MANIFOLD_SR_SELECTOR_MECHANISM_FIELDS
                if _json_hash(selector_payload[field_name])
                != _json_hash(expected[field_name])
            ]
            if mismatched:
                raise ValueError(
                    "the FM selector mechanism disagrees with the resolved "
                    "profile: "
                    + ", ".join(mismatched)
                )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any] | "FormalManifoldRouteComposition" | None,
    ) -> "FormalManifoldRouteComposition":
        if isinstance(payload, cls):
            return payload
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError("formal-manifold route composition must be a mapping.")
        raw = dict(payload)
        nested = raw.get("formal_manifold_route_composition")
        if isinstance(nested, Mapping):
            raw = dict(nested)
        selector_block = raw.get("singleton_response_selector")
        if not isinstance(selector_block, Mapping):
            mechanisms = raw.get("mechanisms")
            selector_block = (
                mechanisms.get("singleton_response_selector", {})
                if isinstance(mechanisms, Mapping)
                else {}
            )
        resolved_fields = _registered_formal_manifold_route_profile(
            raw.get("route_profile", "")
        )
        if not selector_block and resolved_fields is not None:
            selector_block = {
                key: deepcopy(raw[key])
                for key in (
                    resolved_fields.as_dict().keys()
                )
                if key in raw
            }
        selector_family = raw.get(
            "candidate_selector_family", raw.get("sr_route_family")
        )
        selector_profile = raw.get(
            "candidate_selector_profile", raw.get("sr_route_profile")
        )
        route_profile = raw.get("route_profile")
        if route_profile in {None, ""}:
            route_profile = (
                selector_profile
                if selector_profile not in {None, ""}
                else FORMAL_MANIFOLD_ROUTE
            )
        return cls(
            route_family=str(
                raw.get("route_family") or FORMAL_MANIFOLD_ROUTE_FAMILY
            ),
            route_profile=str(route_profile),
            candidate_selector_family=(
                None if selector_family in {None, ""} else str(selector_family)
            ),
            candidate_selector_profile=(
                None if selector_profile in {None, ""} else str(selector_profile)
            ),
            adapt_reoptimization_route=str(
                raw.get("adapt_reoptimization_route") or FORMAL_MANIFOLD_ROUTE
            ),
            singleton_response_selector=deepcopy(dict(selector_block)),
        )

    def as_dict(self) -> dict[str, Any]:
        selector_payload = json.loads(
            json.dumps(
                dict(self.singleton_response_selector),
                allow_nan=False,
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        payload = {
            "schema": FORMAL_MANIFOLD_ROUTE_COMPOSITION_SCHEMA,
            "route_family": str(self.route_family),
            "route_profile": str(self.route_profile),
            "candidate_selector_family": self.candidate_selector_family,
            "candidate_selector_profile": self.candidate_selector_profile,
            "adapt_reoptimization_route": str(self.adapt_reoptimization_route),
            "singleton_response_selector": selector_payload,
        }
        payload["sha256"] = _json_hash(payload)
        return payload

    @property
    def sha256(self) -> str:
        return str(self.as_dict()["sha256"])


def normalize_formal_manifold_route_composition(
    payload: Mapping[str, Any] | FormalManifoldRouteComposition | None,
) -> dict[str, Any]:
    """Return the canonical JSON-safe composed FM identity."""

    return FormalManifoldRouteComposition.from_mapping(payload).as_dict()


def _frame_provenance_id(
    statevector: np.ndarray,
    frame: np.ndarray,
    coordinate_to_frame: np.ndarray,
) -> str:
    """Fingerprint the actual endpoint frame, invariant to global state phase."""

    state = np.asarray(statevector, dtype=complex).reshape(-1)
    tangent_frame = np.asarray(frame, dtype=complex)
    pivot = int(np.argmax(np.abs(state))) if state.size else 0
    phase = (
        complex(state[pivot] / abs(state[pivot]))
        if state.size and abs(state[pivot]) > 0.0
        else 1.0 + 0.0j
    )
    state_aligned = np.asarray(state / phase, dtype=np.complex128)
    frame_aligned = np.asarray(tangent_frame / phase, dtype=np.complex128)
    digest = hashlib.sha256()
    digest.update(b"formal_manifold_physical_frame_v1\0")
    for array in (
        np.asarray(state_aligned.real, dtype="<f8"),
        np.asarray(state_aligned.imag, dtype="<f8"),
        np.asarray(frame_aligned.real, dtype="<f8"),
        np.asarray(frame_aligned.imag, dtype="<f8"),
        np.asarray(coordinate_to_frame, dtype="<f8"),
    ):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(contiguous.tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _relative_norm(delta: np.ndarray, reference: np.ndarray, floor: float) -> float:
    return float(
        np.linalg.norm(np.asarray(delta), ord="fro")
        / (np.linalg.norm(np.asarray(reference), ord="fro") + float(floor))
    )


@dataclass(frozen=True)
class RankRule:
    tau_abs: float = 1.0e-12
    tau_rel: float = 1.0e-10
    tau_gap: float = 1.0e-8

    def __post_init__(self) -> None:
        for name in ("tau_abs", "tau_rel", "tau_gap"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"RankRule.{name} must be finite and nonnegative.")

    def as_dict(self) -> dict[str, float]:
        return {
            "tau_abs": float(self.tau_abs),
            "tau_rel": float(self.tau_rel),
            "tau_gap": float(self.tau_gap),
        }


@dataclass(frozen=True)
class FormalManifoldConfig:
    rank_rule: RankRule = field(default_factory=RankRule)
    supported_metric: JointLinearSolveConfig = field(
        default_factory=JointLinearSolveConfig
    )
    curvature_branch: str = FORMAL_CURVATURE_INVERSE_RBFGS
    numerical_floor: float = 1.0e-14
    qbroyd_epsilon0: float = 0.15
    metric_innovation_soft: float = 0.10
    metric_innovation_hard: float = 0.50
    alignment_sigma_min: float = 1.0e-7
    curvature_guard: float = 1.0e-10
    postcondition_tol: float = 2.0e-8
    powell_eta: float = 0.20
    initial_inverse_curvature: float = 1.0
    inverse_curvature_min: float = 1.0e-6
    inverse_curvature_max: float = 1.0e6
    initial_trust_radius: float = 0.25
    min_trust_radius: float = 1.0e-8
    max_trust_radius: float = 2.0
    trust_shrink: float = 0.5
    trust_expand: float = 1.5
    armijo_c1: float = 1.0e-4
    line_search_shrink: float = 0.5
    line_search_max_steps: int = 18
    gradient_tol: float = 1.0e-8
    step_tol: float = 1.0e-11
    growth_identity_tol: float = 2.0e-7
    inherited_geometry_tol: float = 2.0e-7
    max_rejections: int = 3

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any] | None
    ) -> "FormalManifoldConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError("formal-manifold config payload must be a mapping.")
        data = deepcopy(dict(payload))
        data.pop("rank_rule_effective_fields", None)
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(
                "unknown FormalManifoldConfig fields: " + ", ".join(unknown)
            )
        if isinstance(data.get("rank_rule"), Mapping):
            data["rank_rule"] = RankRule(**dict(data["rank_rule"]))
        if isinstance(data.get("supported_metric"), Mapping):
            data["supported_metric"] = JointLinearSolveConfig(
                **dict(data["supported_metric"])
            )
        return cls(**data)

    def __post_init__(self) -> None:
        if not isinstance(self.supported_metric, JointLinearSolveConfig):
            raise TypeError("supported_metric must be JointLinearSolveConfig.")
        if (
            str(self.supported_metric.policy)
            != JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1
        ):
            raise ValueError(
                "Formal-Manifold requires the shared supported-metric "
                "whitened-eigh policy."
            )
        if str(self.curvature_branch) not in FORMAL_CURVATURE_BRANCHES:
            raise ValueError(
                "curvature_branch must be one of "
                f"{sorted(FORMAL_CURVATURE_BRANCHES)}."
            )
        positive = (
            "numerical_floor",
            "alignment_sigma_min",
            "postcondition_tol",
            "initial_inverse_curvature",
            "inverse_curvature_min",
            "inverse_curvature_max",
            "initial_trust_radius",
            "min_trust_radius",
            "max_trust_radius",
            "gradient_tol",
            "step_tol",
            "growth_identity_tol",
            "inherited_geometry_tol",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"FormalManifoldConfig.{name} must be finite and positive.")
        for name in (
            "metric_innovation_soft",
            "metric_innovation_hard",
            "curvature_guard",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"FormalManifoldConfig.{name} must be finite and nonnegative.")
        if not 0.0 <= float(self.qbroyd_epsilon0) < 1.0:
            raise ValueError("qbroyd_epsilon0 must satisfy 0 <= epsilon < 1.")
        if not 0.0 < float(self.powell_eta) < 1.0:
            raise ValueError("powell_eta must lie strictly between zero and one.")
        if not 0.0 < float(self.armijo_c1) < 1.0:
            raise ValueError("armijo_c1 must lie strictly between zero and one.")
        if not 0.0 < float(self.line_search_shrink) < 1.0:
            raise ValueError("line_search_shrink must lie strictly between zero and one.")
        if not 0.0 < float(self.trust_shrink) < 1.0:
            raise ValueError("trust_shrink must lie strictly between zero and one.")
        if float(self.trust_expand) <= 1.0:
            raise ValueError("trust_expand must exceed one.")
        if float(self.metric_innovation_hard) < float(self.metric_innovation_soft):
            raise ValueError("metric_innovation_hard must not be below the soft threshold.")
        if float(self.inverse_curvature_max) < float(self.inverse_curvature_min):
            raise ValueError("inverse_curvature_max must not be below its minimum.")
        if float(self.max_trust_radius) < float(self.initial_trust_radius):
            raise ValueError("max_trust_radius must not be below the initial radius.")
        if float(self.initial_trust_radius) < float(self.min_trust_radius):
            raise ValueError("initial_trust_radius must not be below the minimum radius.")
        if int(self.line_search_max_steps) < 1 or int(self.max_rejections) < 1:
            raise ValueError("line-search and rejection counts must be positive integers.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank_rule": self.rank_rule.as_dict(),
            "rank_rule_effective_fields": ["tau_gap"],
            "supported_metric": self.supported_metric.as_dict(),
            "curvature_branch": str(self.curvature_branch),
            "numerical_floor": float(self.numerical_floor),
            "qbroyd_epsilon0": float(self.qbroyd_epsilon0),
            "metric_innovation_soft": float(self.metric_innovation_soft),
            "metric_innovation_hard": float(self.metric_innovation_hard),
            "alignment_sigma_min": float(self.alignment_sigma_min),
            "curvature_guard": float(self.curvature_guard),
            "postcondition_tol": float(self.postcondition_tol),
            "powell_eta": float(self.powell_eta),
            "initial_inverse_curvature": float(self.initial_inverse_curvature),
            "inverse_curvature_min": float(self.inverse_curvature_min),
            "inverse_curvature_max": float(self.inverse_curvature_max),
            "initial_trust_radius": float(self.initial_trust_radius),
            "min_trust_radius": float(self.min_trust_radius),
            "max_trust_radius": float(self.max_trust_radius),
            "trust_shrink": float(self.trust_shrink),
            "trust_expand": float(self.trust_expand),
            "armijo_c1": float(self.armijo_c1),
            "line_search_shrink": float(self.line_search_shrink),
            "line_search_max_steps": int(self.line_search_max_steps),
            "gradient_tol": float(self.gradient_tol),
            "step_tol": float(self.step_tol),
            "growth_identity_tol": float(self.growth_identity_tol),
            "inherited_geometry_tol": float(self.inherited_geometry_tol),
            "max_rejections": int(self.max_rejections),
        }


@dataclass(frozen=True)
class ExactStateEvaluation:
    energy: float
    gradient: np.ndarray
    statevector: np.ndarray
    tangents: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ExactStateBackend:
    """Validated exact-state/tangent callback used by the route."""

    def __init__(
        self,
        *,
        evaluate_fn: Callable[[np.ndarray], ExactStateEvaluation],
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
        self.coordinate_registry = registry
        self.manifold_id = str(manifold_id)
        self.parameterization_mode = str(parameterization_mode)
        self.metadata = deepcopy(dict(metadata or {}))

    def evaluate(self, theta: np.ndarray | Sequence[float]) -> ExactStateEvaluation:
        coordinate = _finite_real_vector(theta, name="theta")
        if int(coordinate.size) != len(self.coordinate_registry):
            raise ValueError(
                "theta length does not match coordinate registry: "
                f"{coordinate.size} vs {len(self.coordinate_registry)}."
            )
        raw = self._evaluate_fn(coordinate.copy())
        if not isinstance(raw, ExactStateEvaluation):
            raise TypeError("evaluate_fn must return ExactStateEvaluation.")
        energy = float(raw.energy)
        if not math.isfinite(energy):
            raise ValueError("exact-state energy must be finite.")
        state = _finite_complex_array(raw.statevector, name="statevector").reshape(-1)
        norm = float(np.linalg.norm(state))
        if not np.isclose(norm, 1.0, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError(f"exact-state backend must return a normalized state; norm={norm}.")
        gradient = _finite_real_vector(raw.gradient, name="gradient")
        if int(gradient.size) != int(coordinate.size):
            raise ValueError("gradient length must match the coordinate registry.")
        tangents_raw = _finite_complex_array(raw.tangents, name="tangents")
        if tangents_raw.shape != (int(state.size), int(coordinate.size)):
            raise ValueError(
                "tangent matrix must have shape (state_dimension, coordinate_count), "
                f"got {tangents_raw.shape}."
            )
        # The physical frame is horizontal.  This removes coordinate-dependent
        # phase derivatives without selecting a global state phase.
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


@dataclass(frozen=True)
class ExactFrame:
    statevector: np.ndarray
    horizontal_tangents: np.ndarray
    frame: np.ndarray
    L: np.ndarray
    Z: np.ndarray
    M_R: np.ndarray
    gram_raw: np.ndarray
    gram_retained: np.ndarray
    rank: int
    spectrum: np.ndarray
    threshold: float
    retained_gap: float | None
    gap_status: str
    discarded_gram_residual: float
    whitening: np.ndarray
    whitening_pseudoinverse: np.ndarray
    raw_orthonormalizer: np.ndarray
    regularized_to_raw_frame: np.ndarray
    raw_whitened_metric: np.ndarray
    regularized_reduced_inverse_metric: np.ndarray
    whitening_id: str
    frame_id: str
    whitening_telemetry: Mapping[str, Any]


def build_exact_frame(
    statevector: np.ndarray,
    tangents: np.ndarray,
    *,
    rank_rule: RankRule | None = None,
    supported_metric: JointLinearSolveConfig | None = None,
    gram_override: np.ndarray | None = None,
    numerical_floor: float = 1.0e-14,
) -> ExactFrame:
    """Resolve a gauge-invariant real FS tangent quotient from exact tangents."""

    rule = rank_rule or RankRule()
    state = _finite_complex_array(statevector, name="statevector").reshape(-1)
    tangent_matrix = _finite_complex_array(tangents, name="tangents")
    if tangent_matrix.ndim != 2 or tangent_matrix.shape[0] != state.size:
        raise ValueError("tangents must be a two-dimensional ambient-by-coordinate matrix.")
    d = int(tangent_matrix.shape[1])
    overlaps = np.conjugate(state) @ tangent_matrix
    horizontal = tangent_matrix - np.outer(state, overlaps)
    if gram_override is None:
        gram = _sym(np.real(np.conjugate(horizontal).T @ horizontal))
    else:
        gram = _sym(np.asarray(gram_override, dtype=float))
        if gram.shape != (d, d) or not np.all(np.isfinite(gram)):
            raise ValueError(
                "gram_override must be a finite coordinate-square matrix."
            )
    shared_config = supported_metric or JointLinearSolveConfig()
    if (
        str(shared_config.policy)
        != JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1
    ):
        raise ValueError(
            "build_exact_frame requires the shared supported-metric policy."
        )
    factorization = factor_supported_metric(
        gram,
        rank_relative_tolerance=float(shared_config.rank_relative_tolerance),
        metric_regularization=float(shared_config.metric_regularization),
    )
    if not factorization.feasible and factorization.reason not in {
        "empty_supported_metric_subspace",
    }:
        raise ValueError(
            "exact tangent metric failed supported factorization: "
            f"{factorization.reason}."
        )
    eigenvalues = np.asarray(factorization.raw_eigenvalues, dtype=float)
    lambda_max = float(max(np.max(eigenvalues), 0.0)) if eigenvalues.size else 0.0
    threshold = float(factorization.support_threshold)
    rank = int(factorization.rank)
    if rank:
        values = np.asarray(factorization.retained_eigenvalues, dtype=float)
        vectors = np.asarray(factorization.retained_vectors, dtype=float)
        frame = horizontal @ factorization.raw_orthonormalizer
        L = np.asarray(np.diag(np.sqrt(values)) @ vectors.T, dtype=float)
        gram_retained = _sym(vectors @ np.diag(values) @ vectors.T)
        Z = vectors.copy()
        M_R = _sym(np.diag(1.0 / values))
    else:
        frame = np.zeros((int(state.size), 0), dtype=complex)
        L = np.zeros((0, d), dtype=float)
        Z = np.zeros((d, 0), dtype=float)
        M_R = np.zeros((0, 0), dtype=float)
        gram_retained = np.zeros((d, d), dtype=float)
    if rank == 0:
        retained_gap = None
        gap_status = "stable"
    else:
        discarded = np.asarray(
            eigenvalues[~factorization.retained_mask], dtype=float
        )
        below = float(max(np.max(discarded), 0.0)) if discarded.size else 0.0
        retained_gap = float(
            (float(np.min(factorization.retained_eigenvalues)) - below)
            / (lambda_max + float(numerical_floor))
        )
        gap_status = "stable" if retained_gap >= float(rule.tau_gap) else "unstable"
    frame_orthogonality = (
        float(np.linalg.norm(np.real(np.conjugate(frame).T @ frame) - np.eye(rank)))
        if rank
        else 0.0
    )
    if frame_orthogonality > 5.0e-8:
        raise FloatingPointError(
            "resolved tangent frame failed orthogonality postcondition: "
            f"{frame_orthogonality}."
        )
    discarded_residual = float(np.linalg.norm(gram - gram_retained, ord="fro"))
    return ExactFrame(
        statevector=state,
        horizontal_tangents=horizontal,
        frame=frame,
        L=L,
        Z=Z,
        M_R=M_R,
        gram_raw=gram,
        gram_retained=gram_retained,
        rank=rank,
        spectrum=eigenvalues,
        threshold=threshold,
        retained_gap=retained_gap,
        gap_status=gap_status,
        discarded_gram_residual=discarded_residual,
        whitening=np.asarray(factorization.whitening, dtype=float).copy(),
        whitening_pseudoinverse=np.asarray(
            factorization.whitening_pseudoinverse, dtype=float
        ).copy(),
        raw_orthonormalizer=np.asarray(
            factorization.raw_orthonormalizer, dtype=float
        ).copy(),
        regularized_to_raw_frame=np.asarray(
            factorization.regularized_to_raw_frame, dtype=float
        ).copy(),
        raw_whitened_metric=np.asarray(
            factorization.raw_whitened_metric, dtype=float
        ).copy(),
        regularized_reduced_inverse_metric=(
            np.diag(
                1.0
                / (
                    factorization.retained_eigenvalues
                    + float(factorization.metric_ridge)
                )
            )
            if factorization.rank
            else np.zeros((0, 0), dtype=float)
        ),
        whitening_id=str(factorization.provenance_id),
        frame_id=_frame_provenance_id(state, frame, L),
        whitening_telemetry=deepcopy(factorization.telemetry()),
    )


def density_tangent_cross_gram(
    state_left: np.ndarray,
    frame_left: np.ndarray,
    state_right: np.ndarray,
    frame_right: np.ndarray,
) -> np.ndarray:
    """Return ``1/2 Tr(X_left X_right)`` without materializing densities."""

    psi_l = _finite_complex_array(state_left, name="state_left").reshape(-1)
    psi_r = _finite_complex_array(state_right, name="state_right").reshape(-1)
    e_l = _finite_complex_array(frame_left, name="frame_left")
    e_r = _finite_complex_array(frame_right, name="frame_right")
    if e_l.ndim != 2 or e_r.ndim != 2:
        raise ValueError("endpoint frames must be two-dimensional matrices.")
    if e_l.shape[0] != psi_l.size or e_r.shape[0] != psi_r.size:
        raise ValueError("each endpoint frame must share its state's ambient dimension.")
    if psi_l.size != psi_r.size:
        raise ValueError("endpoint states must share one ambient Hilbert space.")
    left_on_right = np.conjugate(psi_l) @ e_r
    right_on_left = np.conjugate(psi_r) @ e_l
    state_overlap = np.vdot(psi_l, psi_r)
    tangent_overlap = np.conjugate(e_l).T @ e_r
    return np.asarray(
        np.real(
            np.outer(right_on_left, left_on_right)
            + state_overlap * np.conjugate(tangent_overlap)
        ),
        dtype=float,
    )


@dataclass(frozen=True)
class ProcrustesResult:
    Q: np.ndarray
    singular_values: np.ndarray
    sigma_min: float
    valid: bool
    reason: str


@dataclass(frozen=True)
class WhitenedTransport:
    P: np.ndarray
    condition_number: float
    pairing_residual: float
    valid: bool
    reason: str


def endpoint_procrustes(
    cross_gram: np.ndarray,
    *,
    alignment_floor: float = 1.0e-7,
) -> ProcrustesResult:
    cross = np.asarray(cross_gram, dtype=float)
    if cross.ndim != 2 or cross.shape[0] != cross.shape[1]:
        raise ValueError("full endpoint Procrustes transport requires a square cross Gram.")
    if not bool(np.all(np.isfinite(cross))):
        raise ValueError("cross Gram must be finite.")
    rank = int(cross.shape[0])
    if rank == 0:
        return ProcrustesResult(
            Q=np.zeros((0, 0), dtype=float),
            singular_values=np.zeros(0, dtype=float),
            sigma_min=1.0,
            valid=True,
            reason="rank_zero_identity",
        )
    u, singular_values, vh = np.linalg.svd(cross, full_matrices=False)
    q = np.asarray(u @ vh, dtype=float)
    sigma_min = float(np.min(singular_values))
    orthogonality = float(np.linalg.norm(q.T @ q - np.eye(rank), ord="fro"))
    valid = bool(
        sigma_min > float(alignment_floor)
        and orthogonality <= 5.0e-8
    )
    return ProcrustesResult(
        Q=q,
        singular_values=np.asarray(singular_values, dtype=float),
        sigma_min=sigma_min,
        valid=valid,
        reason="ok" if valid else "singular_or_ill_conditioned_endpoint_alignment",
    )


def supported_whitened_transport(
    raw_frame_transport: np.ndarray,
    regularized_to_raw_old: np.ndarray,
    regularized_to_raw_new: np.ndarray,
    *,
    condition_limit: float = 1.0e12,
) -> WhitenedTransport:
    """Translate a raw-FS frame map into shared whitened coordinates."""

    Q = np.asarray(raw_frame_transport, dtype=float)
    C_old = np.asarray(regularized_to_raw_old, dtype=float)
    C_new = np.asarray(regularized_to_raw_new, dtype=float)
    if Q.ndim != 2:
        raise ValueError("raw-frame transport must be a matrix.")
    if C_old.shape != (Q.shape[1], Q.shape[1]):
        raise ValueError("old raw-frame bridge has an incompatible shape.")
    if C_new.shape != (Q.shape[0], Q.shape[0]):
        raise ValueError("new raw-frame bridge has an incompatible shape.")
    if Q.size == 0:
        return WhitenedTransport(
            P=np.zeros(Q.shape, dtype=float),
            condition_number=1.0,
            pairing_residual=0.0,
            valid=True,
            reason="rank_zero_transport",
        )
    try:
        P = np.linalg.solve(C_new, Q @ C_old)
    except np.linalg.LinAlgError:
        return WhitenedTransport(
            P=np.zeros(Q.shape, dtype=float),
            condition_number=float("inf"),
            pairing_residual=float("inf"),
            valid=False,
            reason="singular_whitening_bridge",
        )
    singular_values = np.linalg.svd(P, compute_uv=False)
    sigma_min = float(np.min(singular_values))
    condition = (
        float(np.max(singular_values) / sigma_min)
        if sigma_min > 0.0
        else float("inf")
    )
    if P.shape[0] == P.shape[1] and sigma_min > 0.0:
        probe = np.arange(1, P.shape[0] + 1, dtype=float)
        transported_vector = P @ probe
        transported_covector = np.linalg.solve(P.T, probe)
        pairing_residual = float(
            abs(transported_covector @ transported_vector - probe @ probe)
            / (abs(float(probe @ probe)) + np.finfo(float).tiny)
        )
    else:
        pairing_residual = 0.0
    valid = bool(
        np.all(np.isfinite(P))
        and math.isfinite(condition)
        and condition <= float(condition_limit)
        and pairing_residual <= 5.0e-10
    )
    return WhitenedTransport(
        P=np.asarray(P, dtype=float),
        condition_number=float(condition),
        pairing_residual=float(pairing_residual),
        valid=valid,
        reason="ok" if valid else "ill_conditioned_whitened_frame_transport",
    )


def qbroyd_inverse_update(
    M_R: np.ndarray,
    b_R: np.ndarray,
    epsilon: float,
    *,
    numerical_floor: float = 1.0e-14,
) -> np.ndarray:
    """Published qBroyden recurrence translated to inverse FS-metric units."""

    metric_inverse = _sym(np.asarray(M_R, dtype=float))
    differential = _finite_real_vector(b_R, name="b_R")
    if metric_inverse.shape != (differential.size, differential.size):
        raise ValueError("M_R and b_R dimensions disagree.")
    eps = float(epsilon)
    if not 0.0 <= eps < 1.0:
        raise ValueError("qBroyden epsilon must satisfy 0 <= epsilon < 1.")
    if differential.size == 0:
        return np.zeros((0, 0), dtype=float)
    eigenvalues = np.linalg.eigvalsh(metric_inverse)
    if float(np.min(eigenvalues)) <= float(numerical_floor):
        raise ValueError("qBroyden requires an SPD reduced inverse metric.")
    one_minus = float(1.0 - eps)
    mb = metric_inverse @ differential
    denominator = float(
        one_minus + (eps / 4.0) * float(differential @ mb)
    )
    if denominator <= float(numerical_floor):
        raise FloatingPointError("qBroyden Sherman-Morrison denominator is nonpositive.")
    updated = (1.0 / one_minus) * (
        metric_inverse - (eps / 4.0) * np.outer(mb, mb) / denominator
    )
    updated = _sym(updated)
    if float(np.min(np.linalg.eigvalsh(updated))) <= float(numerical_floor):
        raise FloatingPointError("qBroyden inverse update lost positive definiteness.")
    return updated


@dataclass(frozen=True)
class RBFGSUpdate:
    B: np.ndarray
    y_used: np.ndarray
    applied: bool
    damped: bool
    curvature_raw: float
    curvature_used: float
    postcondition_residual: float | None
    reason: str


@dataclass(frozen=True)
class DirectSR1Update:
    """Guarded update of one direct raised Hessian operator.

    ``A`` is represented in the raw-FS orthonormal physical frame.  It is
    symmetric but intentionally need not be positive definite.
    """

    A: np.ndarray
    q: np.ndarray
    applied: bool
    denominator: float
    guard_threshold: float
    postcondition_residual: float | None
    reason: str


def guarded_direct_sr1(
    A: np.ndarray,
    s: np.ndarray,
    y: np.ndarray,
    *,
    curvature_guard: float = 1.0e-10,
    postcondition_tol: float = 2.0e-8,
    numerical_floor: float = 1.0e-14,
) -> DirectSR1Update:
    """Apply direct SR1, permitting negative eigenvalues by construction."""

    direct_hessian = _sym(np.asarray(A, dtype=float))
    displacement = _finite_real_vector(s, name="s")
    gradient_difference = _finite_real_vector(y, name="y")
    rank = int(displacement.size)
    if direct_hessian.shape != (rank, rank) or gradient_difference.size != rank:
        raise ValueError("A, s, and y dimensions disagree.")
    if not bool(np.all(np.isfinite(direct_hessian))):
        raise ValueError("direct SR1 operator must be finite.")
    if rank == 0:
        return DirectSR1Update(
            A=np.zeros((0, 0), dtype=float),
            q=np.zeros(0, dtype=float),
            applied=False,
            denominator=0.0,
            guard_threshold=0.0,
            postcondition_residual=0.0,
            reason="rank_zero",
        )
    q = np.asarray(gradient_difference - direct_hessian @ displacement, dtype=float)
    denominator = float(q @ displacement)
    guard_threshold = float(curvature_guard) * max(
        float(np.linalg.norm(q) * np.linalg.norm(displacement)),
        float(numerical_floor),
    )
    if abs(denominator) <= guard_threshold:
        return DirectSR1Update(
            A=direct_hessian,
            q=q,
            applied=False,
            denominator=denominator,
            guard_threshold=guard_threshold,
            postcondition_residual=None,
            reason="sr1_denominator_guard",
        )
    updated = _sym(direct_hessian + np.outer(q, q) / denominator)
    if not bool(np.all(np.isfinite(updated))):
        return DirectSR1Update(
            A=direct_hessian,
            q=q,
            applied=False,
            denominator=denominator,
            guard_threshold=guard_threshold,
            postcondition_residual=None,
            reason="nonfinite_sr1_update",
        )
    residual = float(
        np.linalg.norm(updated @ displacement - gradient_difference)
        / (
            np.linalg.norm(gradient_difference)
            + np.linalg.norm(updated @ displacement)
            + float(numerical_floor)
        )
    )
    if residual > float(postcondition_tol):
        return DirectSR1Update(
            A=direct_hessian,
            q=q,
            applied=False,
            denominator=denominator,
            guard_threshold=guard_threshold,
            postcondition_residual=residual,
            reason="secant_postcondition_failure",
        )
    return DirectSR1Update(
        A=updated,
        q=q,
        applied=True,
        denominator=denominator,
        guard_threshold=guard_threshold,
        postcondition_residual=residual,
        reason="applied",
    )


def solve_direct_sr1_trust_step(
    A: np.ndarray,
    gradient: np.ndarray,
    *,
    trust_radius: float,
    supported_metric: JointLinearSolveConfig,
) -> JointLinearSolveResult:
    """Globalize a direct model through the shared supported eigentrust core."""

    direct_hessian = _sym(np.asarray(A, dtype=float))
    covector = _finite_real_vector(gradient, name="gradient")
    if direct_hessian.shape != (covector.size, covector.size):
        raise ValueError("A and gradient dimensions disagree.")
    radius = float(trust_radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("trust_radius must be finite and positive.")
    solve_config = JointLinearSolveConfig(
        policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1,
        rank_relative_tolerance=float(supported_metric.rank_relative_tolerance),
        metric_regularization=float(supported_metric.metric_regularization),
        energy_regularization=float(supported_metric.energy_regularization),
        max_fubini_study_step=radius,
    )
    return solve_joint_linear_model(
        gram=np.eye(covector.size, dtype=float),
        hessian=direct_hessian,
        # The shared kernel maximizes predicted reduction g^T z - z^T A z/2.
        gradient=-covector,
        active_coordinate_count=int(covector.size),
        config=solve_config,
    )


def powell_damped_inverse_rbfgs(
    B: np.ndarray,
    s: np.ndarray,
    y: np.ndarray,
    *,
    eta: float = 0.20,
    curvature_guard: float = 1.0e-10,
    postcondition_tol: float = 2.0e-8,
    numerical_floor: float = 1.0e-14,
) -> RBFGSUpdate:
    """Powell-damped inverse BFGS in one orthonormal physical frame."""

    inverse_hessian = _sym(np.asarray(B, dtype=float))
    displacement = _finite_real_vector(s, name="s")
    gradient_difference = _finite_real_vector(y, name="y")
    rank = int(displacement.size)
    if inverse_hessian.shape != (rank, rank) or gradient_difference.size != rank:
        raise ValueError("B, s, and y dimensions disagree.")
    if rank == 0:
        return RBFGSUpdate(
            B=np.zeros((0, 0), dtype=float),
            y_used=np.zeros(0, dtype=float),
            applied=False,
            damped=False,
            curvature_raw=0.0,
            curvature_used=0.0,
            postcondition_residual=0.0,
            reason="rank_zero",
        )
    if float(np.min(np.linalg.eigvalsh(inverse_hessian))) <= numerical_floor:
        raise ValueError("inverse RBFGS requires an SPD inverse-Hessian model.")
    a_s = np.linalg.solve(inverse_hessian, displacement)
    delta = float(displacement @ a_s)
    sy = float(displacement @ gradient_difference)
    if delta <= numerical_floor:
        return RBFGSUpdate(
            B=inverse_hessian,
            y_used=gradient_difference,
            applied=False,
            damped=False,
            curvature_raw=sy,
            curvature_used=sy,
            postcondition_residual=None,
            reason="nonpositive_model_curvature",
        )
    damped = bool(sy < float(eta) * delta)
    if damped:
        denominator = float(delta - sy)
        if denominator <= numerical_floor:
            return RBFGSUpdate(
                B=inverse_hessian,
                y_used=gradient_difference,
                applied=False,
                damped=True,
                curvature_raw=sy,
                curvature_used=sy,
                postcondition_residual=None,
                reason="powell_denominator_failure",
            )
        theta = float((1.0 - float(eta)) * delta / denominator)
        y_used = theta * gradient_difference + (1.0 - theta) * a_s
    else:
        y_used = gradient_difference.copy()
    sy_used = float(displacement @ y_used)
    guard = float(curvature_guard) * max(
        float(np.linalg.norm(displacement) * np.linalg.norm(y_used)),
        numerical_floor,
    )
    if sy_used <= guard:
        return RBFGSUpdate(
            B=inverse_hessian,
            y_used=y_used,
            applied=False,
            damped=damped,
            curvature_raw=sy,
            curvature_used=sy_used,
            postcondition_residual=None,
            reason="curvature_guard",
        )
    rho = float(1.0 / sy_used)
    identity = np.eye(rank, dtype=float)
    left = identity - rho * np.outer(displacement, y_used)
    updated = left @ inverse_hessian @ left.T + rho * np.outer(
        displacement, displacement
    )
    updated = _sym(updated)
    min_eigenvalue = float(np.min(np.linalg.eigvalsh(updated)))
    if min_eigenvalue <= numerical_floor:
        return RBFGSUpdate(
            B=inverse_hessian,
            y_used=y_used,
            applied=False,
            damped=damped,
            curvature_raw=sy,
            curvature_used=sy_used,
            postcondition_residual=None,
            reason="spd_postcondition_failure",
        )
    residual = float(
        np.linalg.norm(updated @ y_used - displacement)
        / (
            np.linalg.norm(displacement)
            + np.linalg.norm(updated @ y_used)
            + numerical_floor
        )
    )
    if residual > float(postcondition_tol):
        return RBFGSUpdate(
            B=inverse_hessian,
            y_used=y_used,
            applied=False,
            damped=damped,
            curvature_raw=sy,
            curvature_used=sy_used,
            postcondition_residual=residual,
            reason="secant_postcondition_failure",
        )
    return RBFGSUpdate(
        B=updated,
        y_used=y_used,
        applied=True,
        damped=damped,
        curvature_raw=sy,
        curvature_used=sy_used,
        postcondition_residual=residual,
        reason="applied",
    )


@dataclass(frozen=True)
class GrowthMap:
    theta_plus: np.ndarray
    old_positions: tuple[int, ...]
    admitted_positions: tuple[int, ...]
    permutation_old_then_new_to_plus: np.ndarray


def grow_zero_coordinates(
    theta_minus: np.ndarray | Sequence[float],
    registry_minus: Sequence[str],
    registry_plus: Sequence[str],
    *,
    zero_tolerance: float = 1.0e-12,
    theta_plus: np.ndarray | Sequence[float] | None = None,
) -> GrowthMap:
    """Build the explicit logical injection/permutation for zero growth."""

    old_registry = tuple(str(item) for item in registry_minus)
    new_registry = tuple(str(item) for item in registry_plus)
    if len(set(old_registry)) != len(old_registry) or len(set(new_registry)) != len(new_registry):
        raise ValueError("growth registries must contain unique coordinate ids.")
    old_theta = _finite_real_vector(theta_minus, name="theta_minus")
    if old_theta.size != len(old_registry):
        raise ValueError("theta_minus and registry_minus lengths disagree.")
    if not set(old_registry).issubset(set(new_registry)):
        raise ValueError("registry_plus must contain every inherited coordinate exactly once.")
    old_positions = tuple(new_registry.index(item) for item in old_registry)
    old_position_set = set(old_positions)
    admitted_positions = tuple(
        index for index in range(len(new_registry)) if index not in old_position_set
    )
    if len(admitted_positions) + len(old_positions) != len(new_registry):
        raise ValueError("invalid growth injection.")
    expected = np.zeros(len(new_registry), dtype=float)
    for old_index, new_index in enumerate(old_positions):
        expected[new_index] = float(old_theta[old_index])
    if theta_plus is not None:
        supplied = _finite_real_vector(theta_plus, name="theta_plus")
        if supplied.size != len(new_registry):
            raise ValueError("theta_plus and registry_plus lengths disagree.")
        if not np.allclose(
            supplied[list(old_positions)], old_theta, rtol=0.0, atol=zero_tolerance
        ):
            raise ValueError("inherited coordinates changed during declared zero growth.")
        if admitted_positions and not np.allclose(
            supplied[list(admitted_positions)], 0.0, rtol=0.0, atol=zero_tolerance
        ):
            raise ValueError("every admitted coordinate must be zero at exact growth.")
        expected = supplied.copy()
    order = list(old_positions) + list(admitted_positions)
    permutation = np.eye(len(new_registry), dtype=float)[:, order]
    return GrowthMap(
        theta_plus=expected,
        old_positions=old_positions,
        admitted_positions=admitted_positions,
        permutation_old_then_new_to_plus=permutation,
    )


def _receipt_candidate_positions(
    *,
    old_coordinate_count: int,
    insertion_positions: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Reproduce the route's ordered multi-insertion registry mapping."""

    sequence: list[tuple[str, int]] = [
        ("old", int(index)) for index in range(int(old_coordinate_count))
    ]
    previous_original_positions: list[int] = []
    for candidate_index, original_position_raw in enumerate(insertion_positions):
        original_position = int(original_position_raw)
        if original_position < 0 or original_position > int(old_coordinate_count):
            raise ValueError("growth receipt insertion position is outside the old chart.")
        effective_position = int(
            original_position
            + sum(
                1
                for previous_position in previous_original_positions
                if int(previous_position) <= original_position
            )
        )
        if effective_position < 0 or effective_position > len(sequence):
            raise ValueError("growth receipt produced an invalid effective insertion.")
        sequence.insert(effective_position, ("candidate", int(candidate_index)))
        previous_original_positions.append(original_position)
    old_positions = tuple(
        int(sequence.index(("old", old_index)))
        for old_index in range(int(old_coordinate_count))
    )
    candidate_positions = tuple(
        int(sequence.index(("candidate", candidate_index)))
        for candidate_index in range(len(insertion_positions))
    )
    return old_positions, candidate_positions


def _coordinate_gram_from_growth_receipt(
    receipt: FormalGrowthGeometryReceipt,
    *,
    new_coordinate_count: int,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    old_count = int(np.asarray(receipt.G_AA).shape[0])
    candidate_count = len(receipt.candidate_keys)
    if int(new_coordinate_count) != old_count + candidate_count:
        raise ValueError(
            "growth receipt coordinate count does not match the enlarged chart: "
            f"new={new_coordinate_count}, old={old_count}, candidates={candidate_count}."
        )
    old_positions, candidate_positions = _receipt_candidate_positions(
        old_coordinate_count=old_count,
        insertion_positions=receipt.insertion_positions,
    )
    if old_positions != tuple(receipt.old_to_new_registry_mapping):
        raise ValueError(
            "growth receipt old-coordinate mapping disagrees with its insertions."
        )
    gram = np.zeros(
        (int(new_coordinate_count), int(new_coordinate_count)), dtype=float
    )
    gram[np.ix_(old_positions, old_positions)] = np.asarray(
        receipt.G_AA, dtype=float
    )
    gram[np.ix_(old_positions, candidate_positions)] = np.asarray(
        receipt.G_AB, dtype=float
    )
    gram[np.ix_(candidate_positions, old_positions)] = np.asarray(
        receipt.G_AB, dtype=float
    ).T
    gram[np.ix_(candidate_positions, candidate_positions)] = np.asarray(
        receipt.G_BB, dtype=float
    )
    return _sym(gram), old_positions, candidate_positions


@dataclass
class FormalManifoldWarmState:
    theta: np.ndarray
    registry: tuple[str, ...]
    manifold_id: str
    parameterization_mode: str
    energy: float
    statevector: np.ndarray
    tangents: np.ndarray
    frame: np.ndarray
    L: np.ndarray
    Z: np.ndarray
    M_R: np.ndarray
    whitening: np.ndarray
    whitening_pseudoinverse: np.ndarray
    raw_orthonormalizer: np.ndarray
    regularized_to_raw_frame: np.ndarray
    raw_whitened_metric: np.ndarray
    whitening_id: str
    frame_id: str
    logical_range_id: str
    b: np.ndarray
    curvature_branch: str
    # Exactly one of these matrices is authoritative.  ``B`` remains an
    # explicit field for compatibility with existing inverse-RBFGS consumers;
    # the inactive representation is a typed empty (0, 0) matrix.
    B: np.ndarray
    A: np.ndarray
    qbroyd_inverse_metric: np.ndarray
    trust_radius: float
    rank: int
    spectrum: np.ndarray
    retained_gap: float | None
    gap_status: str
    qbroyd_age: int
    metadata: dict[str, Any]


@dataclass
class FormalManifoldResult:
    x: np.ndarray
    fun: float
    nfev: int
    nit: int
    success: bool
    message: str
    warm_state: FormalManifoldWarmState
    info: dict[str, Any]
    _session_token: str = field(repr=False)


def _isotropic_inverse_curvature(rank: int, config: FormalManifoldConfig) -> np.ndarray:
    return float(config.initial_inverse_curvature) * np.eye(int(rank), dtype=float)


def _isotropic_direct_curvature(rank: int, config: FormalManifoldConfig) -> np.ndarray:
    scale = float(1.0 / config.initial_inverse_curvature)
    return scale * np.eye(int(rank), dtype=float)


def _reset_curvature(
    rank: int,
    config: FormalManifoldConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if str(config.curvature_branch) == FORMAL_CURVATURE_INVERSE_RBFGS:
        return (
            _isotropic_inverse_curvature(rank, config),
            np.zeros((0, 0), dtype=float),
        )
    return (
        np.zeros((0, 0), dtype=float),
        _isotropic_direct_curvature(rank, config),
    )


def _clipped_direct_curvature_scale(
    direct_hessian: np.ndarray,
    config: FormalManifoldConfig,
) -> float:
    eigenvalues = np.linalg.eigvalsh(_sym(np.asarray(direct_hessian, dtype=float)))
    default = float(1.0 / config.initial_inverse_curvature)
    if eigenvalues.size == 0:
        return default
    median = float(np.median(eigenvalues))
    magnitude_min = float(1.0 / config.inverse_curvature_max)
    magnitude_max = float(1.0 / config.inverse_curvature_min)
    if not math.isfinite(median) or abs(median) < magnitude_min:
        return default
    return float(math.copysign(np.clip(abs(median), magnitude_min, magnitude_max), median))


def _warm_state_from_frame(
    *,
    theta: np.ndarray,
    backend: ExactStateBackend,
    evaluation: ExactStateEvaluation,
    frame: ExactFrame,
    B: np.ndarray,
    A: np.ndarray | None = None,
    curvature_branch: str = FORMAL_CURVATURE_INVERSE_RBFGS,
    qbroyd_inverse_metric: np.ndarray | None = None,
    trust_radius: float,
    qbroyd_age: int,
    metadata: Mapping[str, Any],
) -> FormalManifoldWarmState:
    branch = str(curvature_branch)
    if branch not in FORMAL_CURVATURE_BRANCHES:
        raise ValueError(f"unsupported curvature branch {curvature_branch!r}.")
    inverse_matrix = _sym(np.asarray(B, dtype=float))
    direct_matrix = _sym(
        np.zeros((0, 0), dtype=float) if A is None else np.asarray(A, dtype=float)
    )
    if branch == FORMAL_CURVATURE_INVERSE_RBFGS:
        if inverse_matrix.shape != (frame.rank, frame.rank) or direct_matrix.shape != (0, 0):
            raise ValueError("inverse-RBFGS state requires only a rank-by-rank B matrix.")
    elif direct_matrix.shape != (frame.rank, frame.rank) or inverse_matrix.shape != (0, 0):
        raise ValueError("direct-SR1 state requires only a rank-by-rank A matrix.")
    logical_range_id = _json_hash(
        {
            "schema": "formal_manifold_logical_range_v1",
            "registry": list(backend.coordinate_registry),
            "retained_mask": frame.whitening_telemetry.get(
                "metric_retained_mask", []
            ),
            "Z": np.asarray(frame.Z, dtype=float).tolist(),
        }
    )
    frame_id = _json_hash(
        {
            "geometry_frame_id": str(frame.frame_id),
            "registry": list(backend.coordinate_registry),
        }
    )
    resolved_metadata = deepcopy(dict(metadata))
    resolved_metadata.update(
        {
            "whitening_id": str(frame.whitening_id),
            "curvature_frame_id": str(frame_id),
            "curvature_branch": branch,
            "qbroyd_logical_range_id": str(logical_range_id),
        }
    )
    return FormalManifoldWarmState(
        theta=np.asarray(theta, dtype=float).copy(),
        registry=tuple(backend.coordinate_registry),
        manifold_id=str(backend.manifold_id),
        parameterization_mode=str(backend.parameterization_mode),
        energy=float(evaluation.energy),
        statevector=np.asarray(evaluation.statevector, dtype=complex).copy(),
        tangents=np.asarray(frame.horizontal_tangents, dtype=complex).copy(),
        frame=np.asarray(frame.frame, dtype=complex).copy(),
        L=np.asarray(frame.L, dtype=float).copy(),
        Z=np.asarray(frame.Z, dtype=float).copy(),
        M_R=np.asarray(frame.M_R, dtype=float).copy(),
        whitening=np.asarray(frame.whitening, dtype=float).copy(),
        whitening_pseudoinverse=np.asarray(
            frame.whitening_pseudoinverse, dtype=float
        ).copy(),
        raw_orthonormalizer=np.asarray(
            frame.raw_orthonormalizer, dtype=float
        ).copy(),
        regularized_to_raw_frame=np.asarray(
            frame.regularized_to_raw_frame, dtype=float
        ).copy(),
        raw_whitened_metric=np.asarray(
            frame.raw_whitened_metric, dtype=float
        ).copy(),
        whitening_id=str(frame.whitening_id),
        frame_id=str(frame_id),
        logical_range_id=str(logical_range_id),
        b=np.asarray(evaluation.gradient, dtype=float).copy(),
        curvature_branch=branch,
        B=inverse_matrix,
        A=direct_matrix,
        qbroyd_inverse_metric=_sym(
            frame.regularized_reduced_inverse_metric
            if qbroyd_inverse_metric is None
            else np.asarray(qbroyd_inverse_metric, dtype=float)
        ),
        trust_radius=float(trust_radius),
        rank=int(frame.rank),
        spectrum=np.asarray(frame.spectrum, dtype=float).copy(),
        retained_gap=(None if frame.retained_gap is None else float(frame.retained_gap)),
        gap_status=str(frame.gap_status),
        qbroyd_age=int(qbroyd_age),
        metadata=resolved_metadata,
    )


def _state_projective_distance(left: np.ndarray, right: np.ndarray) -> float:
    overlap = min(1.0, abs(complex(np.vdot(left, right))))
    # sin(d_FS) has ordinary distance scaling near coincident rays.  In
    # contrast, 1-|overlap| has a quadratic small-distance scale.
    return float(math.sqrt(max(0.0, 1.0 - overlap * overlap)))


def _gradient_frame(state: FormalManifoldWarmState) -> np.ndarray:
    if state.rank == 0:
        return np.zeros(0, dtype=float)
    return np.asarray(state.raw_orthonormalizer.T @ state.b, dtype=float)


def _compact_state_summary(state: FormalManifoldWarmState | None) -> dict[str, Any]:
    if state is None:
        return {"active": False}
    return {
        "active": True,
        "coordinate_count": int(state.theta.size),
        "coordinate_registry_sha256": _json_hash(list(state.registry)),
        "curvature_branch": str(state.curvature_branch),
        "energy": float(state.energy),
        "gap_status": str(state.gap_status),
        "manifold_id": str(state.manifold_id),
        "parameterization_mode": str(state.parameterization_mode),
        "qbroyd_age": int(state.qbroyd_age),
        "rank": int(state.rank),
        "retained_gap": (
            None if state.retained_gap is None else float(state.retained_gap)
        ),
        "trust_radius": float(state.trust_radius),
        "whitening_id": str(state.whitening_id),
        "frame_id": str(state.frame_id),
        "logical_range_id": str(state.logical_range_id),
        "curvature_coordinate_system": "supported_raw_fs_orthonormal_frame_v1",
        "shared_solver_coordinate_system": "supported_regularized_metric_v1",
        "curvature_whitening_id": str(
            state.metadata.get("curvature_whitening_id", state.whitening_id)
        ),
        "curvature_frame_id": str(
            state.metadata.get("curvature_frame_id", state.frame_id)
        ),
        "qbroyd_whitening_id": str(
            state.metadata.get("qbroyd_whitening_id", state.whitening_id)
        ),
        "qbroyd_logical_range_id": str(
            state.metadata.get("qbroyd_logical_range_id", state.logical_range_id)
        ),
        "raw_metric_condition_number": state.metadata.get(
            "raw_metric_condition_number"
        ),
        "retained_metric_condition_number": state.metadata.get(
            "retained_metric_condition_number"
        ),
        "valid_curvature": bool(
            (
                state.curvature_branch == FORMAL_CURVATURE_INVERSE_RBFGS
                and state.B.shape == (state.rank, state.rank)
                and state.A.shape == (0, 0)
                and (
                    state.rank == 0
                    or float(np.min(np.linalg.eigvalsh(state.B))) > 0.0
                )
            )
            or (
                state.curvature_branch == FORMAL_CURVATURE_DIRECT_SR1
                and state.A.shape == (state.rank, state.rank)
                and state.B.shape == (0, 0)
                and bool(np.all(np.isfinite(state.A)))
                and bool(np.allclose(state.A, state.A.T))
            )
        ),
        "valid_metric": True,
    }


def _validate_warm_state(
    state: FormalManifoldWarmState,
    config: FormalManifoldConfig,
) -> None:
    rank = int(state.rank)
    dimension = int(state.theta.size)
    expected = {
        "frame": (int(state.statevector.size), rank),
        "L": (rank, dimension),
        "Z": (dimension, rank),
        "M_R": (rank, rank),
        "whitening": (dimension, rank),
        "whitening_pseudoinverse": (rank, dimension),
        "raw_orthonormalizer": (dimension, rank),
        "regularized_to_raw_frame": (rank, rank),
        "raw_whitened_metric": (rank, rank),
        "qbroyd_inverse_metric": (rank, rank),
    }
    for name, shape in expected.items():
        if np.asarray(getattr(state, name)).shape != shape:
            raise ValueError(
                f"checkpoint {name} shape is incompatible with its whitening provenance."
            )
    branch = str(state.curvature_branch)
    if branch != str(config.curvature_branch):
        raise ValueError(
            "checkpoint curvature branch does not match the session configuration."
        )
    if str(state.metadata.get("curvature_branch", branch)) != branch:
        raise ValueError("checkpoint curvature branch provenance is incompatible.")
    if branch == FORMAL_CURVATURE_INVERSE_RBFGS:
        if state.B.shape != (rank, rank) or state.A.shape != (0, 0):
            raise ValueError(
                "checkpoint inverse-RBFGS state must contain B and no direct A."
            )
    elif branch == FORMAL_CURVATURE_DIRECT_SR1:
        if state.A.shape != (rank, rank) or state.B.shape != (0, 0):
            raise ValueError(
                "checkpoint direct-SR1 state must contain A and no inverse B."
            )
        if not bool(np.all(np.isfinite(state.A))) or not np.allclose(
            state.A, state.A.T, rtol=0.0, atol=float(config.postcondition_tol)
        ):
            raise ValueError("checkpoint direct SR1 operator must be finite and symmetric.")
    else:
        raise ValueError(f"unsupported checkpoint curvature branch {branch!r}.")
    stored_config = state.metadata.get("supported_metric_config")
    if stored_config is not None and dict(stored_config) != config.supported_metric.as_dict():
        raise ValueError(
            "checkpoint supported-metric configuration does not match the session."
        )
    curvature_frame_id = state.metadata.get("curvature_frame_id")
    if curvature_frame_id is not None and str(curvature_frame_id) != str(state.frame_id):
        raise ValueError(
            "checkpoint curvature_frame_id is incompatible with the physical frame."
        )
    qbroyd_range_id = state.metadata.get("qbroyd_logical_range_id")
    if qbroyd_range_id is not None and str(qbroyd_range_id) != str(
        state.logical_range_id
    ):
        raise ValueError(
            "checkpoint qBroyden logical-range provenance is incompatible."
        )
    if rank:
        identity_residual = float(
            np.linalg.norm(
                state.whitening_pseudoinverse @ state.whitening - np.eye(rank),
                ord="fro",
            )
        )
        if identity_residual > float(config.postcondition_tol):
            raise ValueError(
                "checkpoint whitening/pseudoinverse identity postcondition failed."
            )
        positive_names = ["qbroyd_inverse_metric"]
        if branch == FORMAL_CURVATURE_INVERSE_RBFGS:
            positive_names.append("B")
        for name in positive_names:
            if float(np.min(np.linalg.eigvalsh(_sym(getattr(state, name))))) <= 0.0:
                raise ValueError(f"checkpoint {name} must be SPD.")


class FormalManifoldSession:
    """Persistent accepted-ansatz warm state with explicit commit/rollback."""

    def __init__(
        self,
        state: FormalManifoldWarmState | None = None,
        *,
        config: FormalManifoldConfig | None = None,
        branch_id: str = "single_frontier:0",
        parent_branch_id: str | None = None,
        route_composition: (
            Mapping[str, Any] | FormalManifoldRouteComposition | None
        ) = None,
    ) -> None:
        self.config = config or FormalManifoldConfig()
        self.route_composition = FormalManifoldRouteComposition.from_mapping(
            route_composition
        )
        if state is not None:
            _validate_warm_state(state, self.config)
        self.state = deepcopy(state)
        self._pending: FormalManifoldResult | None = None
        self._last_reset_reason: str | None = None
        self._reset_count = 0
        self._commit_count = 0
        self._rollback_count = 0
        self._branch_id = str(branch_id)
        self._parent_branch_id = (
            None if parent_branch_id is None else str(parent_branch_id)
        )

    def checkpoint(self) -> FormalManifoldWarmState | None:
        return deepcopy(self.state)

    @property
    def branch_id(self) -> str:
        return str(self._branch_id)

    @property
    def parent_branch_id(self) -> str | None:
        return (
            None
            if self._parent_branch_id is None
            else str(self._parent_branch_id)
        )

    def fork(self, *, branch_id: str) -> "FormalManifoldSession":
        """Clone committed FM state for one speculative beam child."""

        if self._pending is not None:
            raise RuntimeError(
                "cannot fork a formal-manifold session with a pending proposal."
            )
        forked = FormalManifoldSession(
            state=self.state,
            config=self.config,
            branch_id=str(branch_id),
            parent_branch_id=str(self._branch_id),
            route_composition=self.route_composition,
        )
        forked._last_reset_reason = self._last_reset_reason
        forked._reset_count = int(self._reset_count)
        forked._commit_count = int(self._commit_count)
        forked._rollback_count = int(self._rollback_count)
        return forked

    def checkpoint_payload(self) -> dict[str, Any] | None:
        """Return a JSON-safe persistent curvature/whitening checkpoint."""

        if self.state is None:
            return None
        state = self.state
        _validate_warm_state(state, self.config)
        composition = self.route_composition.as_dict()
        config_payload = self.config.as_dict()
        return {
            "schema": "formal_manifold_warm_state_checkpoint_v1",
            "route": FORMAL_MANIFOLD_ROUTE,
            "branch_id": str(self._branch_id),
            "parent_branch_id": self.parent_branch_id,
            "route_composition": composition,
            "route_composition_sha256": str(composition["sha256"]),
            "formal_manifold_config": config_payload,
            "formal_manifold_config_sha256": _json_hash(config_payload),
            "transaction_state": {
                "schema": "formal_manifold_transaction_state_v1",
                "pending": False,
                "last_reset_reason": self._last_reset_reason,
                "reset_count": int(self._reset_count),
                "commit_count": int(self._commit_count),
                "rollback_count": int(self._rollback_count),
                "structural_rollback_supported": False,
                "rollback_scope": "pending_proposal_only",
            },
            "supported_metric_config": self.config.supported_metric.as_dict(),
            "theta": [float(value) for value in state.theta.tolist()],
            "registry": list(state.registry),
            "registry_sha256": _json_hash(list(state.registry)),
            "manifold_id": str(state.manifold_id),
            "parameterization_mode": str(state.parameterization_mode),
            "energy": float(state.energy),
            "rank": int(state.rank),
            "spectrum": [float(value) for value in state.spectrum.tolist()],
            "retained_gap": (
                None if state.retained_gap is None else float(state.retained_gap)
            ),
            "gap_status": str(state.gap_status),
            "whitening_id": str(state.whitening_id),
            "frame_id": str(state.frame_id),
            "logical_range_id": str(state.logical_range_id),
            "whitening": state.whitening.tolist(),
            "whitening_pseudoinverse": state.whitening_pseudoinverse.tolist(),
            "raw_orthonormalizer": state.raw_orthonormalizer.tolist(),
            "regularized_to_raw_frame": state.regularized_to_raw_frame.tolist(),
            "raw_whitened_metric": state.raw_whitened_metric.tolist(),
            "curvature_branch": str(state.curvature_branch),
            "inverse_curvature": (
                state.B.tolist()
                if state.curvature_branch == FORMAL_CURVATURE_INVERSE_RBFGS
                else None
            ),
            "direct_curvature": (
                state.A.tolist()
                if state.curvature_branch == FORMAL_CURVATURE_DIRECT_SR1
                else None
            ),
            "qbroyd_inverse_metric": state.qbroyd_inverse_metric.tolist(),
            "qbroyd_age": int(state.qbroyd_age),
            "trust_radius": float(state.trust_radius),
            "curvature_whitening_id": str(
                state.metadata.get("curvature_whitening_id", state.whitening_id)
            ),
            "curvature_frame_id": str(
                state.metadata.get("curvature_frame_id", state.frame_id)
            ),
            "qbroyd_whitening_id": str(
                state.metadata.get("qbroyd_whitening_id", state.whitening_id)
            ),
            "qbroyd_logical_range_id": str(
                state.metadata.get(
                    "qbroyd_logical_range_id", state.logical_range_id
                )
            ),
            "metadata": deepcopy(dict(state.metadata)),
            "no_statevector_serialized": True,
            "no_credentials_serialized": True,
        }

    def transaction_payload(self) -> dict[str, Any]:
        """Return the branch-local transactional state without curvature data."""

        if self._pending is not None:
            raise RuntimeError(
                "cannot checkpoint a formal-manifold session with a pending proposal."
            )
        composition = self.route_composition.as_dict()
        config_payload = self.config.as_dict()
        return {
            "schema": "formal_manifold_transaction_state_v1",
            "branch_id": str(self._branch_id),
            "parent_branch_id": self.parent_branch_id,
            "pending": False,
            "last_reset_reason": self._last_reset_reason,
            "reset_count": int(self._reset_count),
            "commit_count": int(self._commit_count),
            "rollback_count": int(self._rollback_count),
            "structural_rollback_supported": False,
            "rollback_scope": "pending_proposal_only",
            "route_composition": composition,
            "route_composition_sha256": str(composition["sha256"]),
            "formal_manifold_config": config_payload,
            "formal_manifold_config_sha256": _json_hash(config_payload),
        }

    def restore_transaction_payload(self, payload: Mapping[str, Any]) -> None:
        """Restore branch identity/counters for a session with no warm state."""

        if self.state is not None or self._pending is not None:
            raise RuntimeError(
                "empty-session transaction restore requires no warm or pending state."
            )
        data = dict(payload)
        if data.get("schema") != "formal_manifold_transaction_state_v1":
            raise ValueError("unsupported formal-manifold transaction schema.")
        if bool(data.get("pending", False)):
            raise ValueError("transaction checkpoint cannot contain a pending proposal.")
        if bool(data.get("structural_rollback_supported", False)):
            raise ValueError("structural rollback is not supported by Formal-Manifold.")
        if str(data.get("rollback_scope", "pending_proposal_only")) != (
            "pending_proposal_only"
        ):
            raise ValueError("transaction checkpoint has an unsupported rollback scope.")
        stored_composition = FormalManifoldRouteComposition.from_mapping(
            data.get("route_composition")
        ).as_dict()
        if str(data.get("route_composition_sha256")) != str(
            stored_composition["sha256"]
        ):
            raise ValueError("transaction route-composition fingerprint mismatch.")
        if stored_composition != self.route_composition.as_dict():
            raise ValueError("transaction route composition does not match the session.")
        stored_config = dict(data.get("formal_manifold_config", {}))
        if str(data.get("formal_manifold_config_sha256")) != _json_hash(
            stored_config
        ):
            raise ValueError("transaction formal-manifold config fingerprint mismatch.")
        if stored_config != self.config.as_dict():
            raise ValueError("transaction formal-manifold config does not match the session.")
        restored_counts: dict[str, int] = {}
        for field_name in ("reset_count", "commit_count", "rollback_count"):
            try:
                value = int(data.get(field_name, 0))
            except (TypeError, ValueError):
                raise ValueError(
                    f"transaction {field_name} must be an integer."
                ) from None
            if value < 0:
                raise ValueError(
                    f"transaction {field_name} must be nonnegative."
                )
            restored_counts[field_name] = value
        self._branch_id = str(data.get("branch_id", self._branch_id))
        parent_branch = data.get("parent_branch_id")
        self._parent_branch_id = (
            None if parent_branch is None else str(parent_branch)
        )
        self._last_reset_reason = (
            None
            if data.get("last_reset_reason") is None
            else str(data.get("last_reset_reason"))
        )
        self._reset_count = restored_counts["reset_count"]
        self._commit_count = restored_counts["commit_count"]
        self._rollback_count = restored_counts["rollback_count"]

    def restore_checkpoint_payload(
        self,
        payload: Mapping[str, Any],
        backend: ExactStateBackend,
    ) -> int:
        """Restore a checkpoint after recomputing and validating its exact frame.

        Returns the single exact backend evaluation used for validation so the
        outer query/nfev ledger can charge it explicitly.
        """

        if self._pending is not None:
            raise RuntimeError("cannot restore while a proposal is pending.")
        data = dict(payload)
        if data.get("schema") != "formal_manifold_warm_state_checkpoint_v1":
            raise ValueError("unsupported formal-manifold checkpoint schema.")
        if normalize_reoptimization_route(str(data.get("route"))) != FORMAL_MANIFOLD_ROUTE:
            raise ValueError("checkpoint route does not match Formal-Manifold.")
        stored_composition_raw = data.get("route_composition")
        expected_composition = self.route_composition.as_dict()
        if stored_composition_raw is None:
            if self.route_composition.candidate_selector_profile is not None:
                raise ValueError(
                    "checkpoint lacks the required formal-manifold route composition."
                )
        else:
            if not isinstance(stored_composition_raw, Mapping):
                raise ValueError("checkpoint route composition must be a mapping.")
            stored_composition = FormalManifoldRouteComposition.from_mapping(
                stored_composition_raw
            ).as_dict()
            if str(data.get("route_composition_sha256")) != str(
                stored_composition["sha256"]
            ):
                raise ValueError("checkpoint route composition fingerprint mismatch.")
            if stored_composition != expected_composition:
                raise ValueError(
                    "checkpoint route composition does not match the session."
                )
        config_payload = self.config.as_dict()
        stored_config_raw = data.get("formal_manifold_config")
        if stored_config_raw is None:
            if self.route_composition.candidate_selector_profile is not None:
                raise ValueError(
                    "checkpoint lacks the required full formal-manifold config."
                )
        else:
            if not isinstance(stored_config_raw, Mapping):
                raise ValueError("checkpoint formal-manifold config must be a mapping.")
            stored_config = dict(stored_config_raw)
            if str(data.get("formal_manifold_config_sha256")) != _json_hash(
                stored_config
            ):
                raise ValueError("checkpoint formal-manifold config fingerprint mismatch.")
            if stored_config != config_payload:
                raise ValueError(
                    "checkpoint full formal-manifold config does not match the session."
                )
        transaction_raw = data.get("transaction_state", {})
        if not isinstance(transaction_raw, Mapping):
            raise ValueError("checkpoint transaction state must be a mapping.")
        transaction = dict(transaction_raw)
        if bool(transaction.get("pending", False)):
            raise ValueError("checkpoint cannot contain a pending FM proposal.")
        if bool(transaction.get("structural_rollback_supported", False)):
            raise ValueError("checkpoint requests unsupported structural rollback state.")
        restored_counts: dict[str, int] = {}
        for field_name in ("reset_count", "commit_count", "rollback_count"):
            try:
                field_value = int(transaction.get(field_name, 0))
            except (TypeError, ValueError):
                raise ValueError(
                    f"checkpoint transaction {field_name} must be an integer."
                ) from None
            if field_value < 0:
                raise ValueError(
                    f"checkpoint transaction {field_name} must be nonnegative."
                )
            restored_counts[field_name] = field_value
        if dict(data.get("supported_metric_config", {})) != self.config.supported_metric.as_dict():
            raise ValueError("checkpoint supported-metric configuration mismatch.")
        registry = tuple(str(value) for value in data.get("registry", []))
        if registry != tuple(backend.coordinate_registry):
            raise ValueError("checkpoint coordinate registry does not match the backend.")
        if str(data.get("registry_sha256")) != _json_hash(list(registry)):
            raise ValueError("checkpoint coordinate registry fingerprint mismatch.")
        if str(data.get("manifold_id")) != str(backend.manifold_id):
            raise ValueError("checkpoint manifold id does not match the backend.")
        if str(data.get("parameterization_mode")) != str(backend.parameterization_mode):
            raise ValueError("checkpoint parameterization mode does not match the backend.")
        theta = _finite_real_vector(data.get("theta", []), name="checkpoint theta")
        evaluation = backend.evaluate(theta)
        frame = self._frame_for(evaluation)
        if str(data.get("whitening_id")) != str(frame.whitening_id):
            raise ValueError("checkpoint whitening provenance does not match the rebuilt frame.")
        if int(data.get("rank", -1)) != int(frame.rank):
            raise ValueError("checkpoint supported rank does not match the rebuilt frame.")
        rebuilt_frame_id = _json_hash(
            {
                "geometry_frame_id": str(frame.frame_id),
                "registry": list(backend.coordinate_registry),
            }
        )
        rebuilt_range_id = _json_hash(
            {
                "schema": "formal_manifold_logical_range_v1",
                "registry": list(backend.coordinate_registry),
                "retained_mask": frame.whitening_telemetry.get(
                    "metric_retained_mask", []
                ),
                "Z": np.asarray(frame.Z, dtype=float).tolist(),
            }
        )
        if str(data.get("frame_id")) != str(rebuilt_frame_id):
            raise ValueError(
                "checkpoint physical-frame provenance does not match the rebuilt frame."
            )
        if str(data.get("logical_range_id")) != str(rebuilt_range_id):
            raise ValueError(
                "checkpoint logical-range provenance does not match the rebuilt frame."
            )
        stored_arrays = {
            "whitening": frame.whitening,
            "whitening_pseudoinverse": frame.whitening_pseudoinverse,
            "raw_orthonormalizer": frame.raw_orthonormalizer,
            "regularized_to_raw_frame": frame.regularized_to_raw_frame,
            "raw_whitened_metric": frame.raw_whitened_metric,
        }
        for key, rebuilt in stored_arrays.items():
            stored = np.asarray(data.get(key), dtype=float)
            if stored.shape != rebuilt.shape or not np.allclose(
                stored,
                rebuilt,
                rtol=0.0,
                atol=float(self.config.postcondition_tol),
            ):
                raise ValueError(
                    f"checkpoint {key} does not match the rebuilt whitening frame."
                )
        branch = str(
            data.get("curvature_branch", FORMAL_CURVATURE_INVERSE_RBFGS)
        )
        if branch != str(self.config.curvature_branch):
            raise ValueError(
                "checkpoint curvature branch does not match the session configuration."
            )
        if branch == FORMAL_CURVATURE_INVERSE_RBFGS:
            if data.get("direct_curvature") is not None:
                raise ValueError("inverse checkpoint cannot contain a direct SR1 operator.")
            B = _sym(np.asarray(data.get("inverse_curvature"), dtype=float))
            A = np.zeros((0, 0), dtype=float)
        elif branch == FORMAL_CURVATURE_DIRECT_SR1:
            if data.get("inverse_curvature") is not None:
                raise ValueError("direct checkpoint cannot contain an inverse B operator.")
            B = np.zeros((0, 0), dtype=float)
            A = _sym(np.asarray(data.get("direct_curvature"), dtype=float))
        else:
            raise ValueError(f"unsupported checkpoint curvature branch {branch!r}.")
        M_q = _sym(np.asarray(data.get("qbroyd_inverse_metric"), dtype=float))
        active_curvature = B if branch == FORMAL_CURVATURE_INVERSE_RBFGS else A
        if active_curvature.shape != (frame.rank, frame.rank) or M_q.shape != (frame.rank, frame.rank):
            raise ValueError("checkpoint curvature/qBroyden shapes do not match the rank.")
        metadata = deepcopy(dict(data.get("metadata", {})))
        metadata.update(
            {
                "supported_metric_config": self.config.supported_metric.as_dict(),
                "whitening_id": str(frame.whitening_id),
                "curvature_whitening_id": str(data.get("curvature_whitening_id")),
                "qbroyd_whitening_id": str(data.get("qbroyd_whitening_id")),
                "curvature_frame_id": str(data.get("curvature_frame_id")),
                "curvature_branch": branch,
                "qbroyd_logical_range_id": str(
                    data.get("qbroyd_logical_range_id")
                ),
                "checkpoint_restore": "validated_exact_frame_v1",
            }
        )
        restored = _warm_state_from_frame(
            theta=theta,
            backend=backend,
            evaluation=evaluation,
            frame=frame,
            B=B,
            A=A,
            curvature_branch=branch,
            qbroyd_inverse_metric=M_q,
            trust_radius=float(data.get("trust_radius")),
            qbroyd_age=int(data.get("qbroyd_age", 0)),
            metadata=metadata,
        )
        _validate_warm_state(restored, self.config)
        self.state = restored
        self._branch_id = str(data.get("branch_id", self._branch_id))
        parent_branch_raw = data.get("parent_branch_id", self._parent_branch_id)
        self._parent_branch_id = (
            None if parent_branch_raw is None else str(parent_branch_raw)
        )
        self._last_reset_reason = (
            None
            if transaction.get("last_reset_reason") is None
            else str(transaction.get("last_reset_reason"))
        )
        self._reset_count = int(restored_counts["reset_count"])
        self._commit_count = int(restored_counts["commit_count"])
        self._rollback_count = int(restored_counts["rollback_count"])
        return 1

    def reset(self, reason: str) -> None:
        self.state = None
        self._pending = None
        self._last_reset_reason = str(reason)
        self._reset_count += 1

    def rollback(self) -> None:
        self._pending = None
        self._rollback_count += 1

    def commit(self, result: FormalManifoldResult) -> None:
        if self._pending is None:
            raise RuntimeError("no pending formal-manifold proposal to commit.")
        if result._session_token != self._pending._session_token:
            raise ValueError("result does not belong to the pending session proposal.")
        # Commit the session-owned snapshot, not caller-mutable result arrays.
        self.state = deepcopy(self._pending.warm_state)
        self._pending = None
        self._commit_count += 1

    def summary(self) -> dict[str, Any]:
        return {
            "schema": "formal_manifold_session_summary_v1",
            "route": FORMAL_MANIFOLD_ROUTE,
            "branch_id": str(self._branch_id),
            "parent_branch_id": self.parent_branch_id,
            "route_composition": self.route_composition.as_dict(),
            "formal_manifold_config_sha256": _json_hash(self.config.as_dict()),
            "authoritative_metric": "exact_state_endpoint_refresh_v1",
            "supported_metric_whitening": (
                JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1
            ),
            "optimizer_coordinate_system": "supported_raw_fs_orthonormal_frame_v1",
            "shared_solver_coordinate_system": "supported_regularized_metric_v1",
            "qbroyd_mode": "shadow_predictor_exact_refresh_v1",
            "qbang_momentum_active": False,
            "curvature_branch": str(self.config.curvature_branch),
            "curvature_branches_mutually_exclusive": True,
            "candidate_scoring_unchanged": False,
            "candidate_scoring_policy": (
                "formal_manifold_query_closed_phase_models_v1"
            ),
            "static_route_identity_unchanged": True,
            "statistically_calibrated": False,
            "unsupported_capabilities": [
                "hardware_statistical_rank_certification",
                "gram_only_growth_without_transport_handle",
            ],
            "config": self.config.as_dict(),
            "state": _compact_state_summary(self.state),
            "pending": bool(self._pending is not None),
            "last_reset_reason": self._last_reset_reason,
            "reset_count": int(self._reset_count),
            "commit_count": int(self._commit_count),
            "rollback_count": int(self._rollback_count),
        }

    def _frame_for(
        self,
        evaluation: ExactStateEvaluation,
        *,
        gram_override: np.ndarray | None = None,
    ) -> ExactFrame:
        return build_exact_frame(
            evaluation.statevector,
            evaluation.tangents,
            rank_rule=self.config.rank_rule,
            supported_metric=self.config.supported_metric,
            gram_override=gram_override,
            numerical_floor=self.config.numerical_floor,
        )

    def _anchor(
        self,
        backend: ExactStateBackend,
        theta: np.ndarray,
        evaluation: ExactStateEvaluation,
        *,
        growth_receipt: FormalGrowthGeometryReceipt | None = None,
    ) -> tuple[FormalManifoldWarmState, dict[str, Any]]:
        previous = self.state
        gram_override: np.ndarray | None = None
        receipt_old_positions: tuple[int, ...] = ()
        receipt_candidate_positions: tuple[int, ...] = ()
        receipt_validation_payload: dict[str, Any] | None = None
        if growth_receipt is not None:
            if not isinstance(growth_receipt, FormalGrowthGeometryReceipt):
                raise TypeError(
                    "growth_receipt must be a FormalGrowthGeometryReceipt."
                )
            if not growth_receipt.candidate_keys:
                raise ValueError("growth receipt must admit at least one candidate.")
            old_registry = () if previous is None else tuple(previous.registry)
            growth = grow_zero_coordinates(
                np.zeros(0, dtype=float)
                if previous is None
                else np.asarray(previous.theta, dtype=float),
                old_registry,
                backend.coordinate_registry,
                theta_plus=theta,
                zero_tolerance=float(self.config.growth_identity_tol),
            )
            gram_override, receipt_old_positions, receipt_candidate_positions = (
                _coordinate_gram_from_growth_receipt(
                    growth_receipt,
                    new_coordinate_count=len(backend.coordinate_registry),
                )
            )
            if receipt_old_positions != tuple(growth.old_positions):
                raise ValueError(
                    "growth receipt mapping disagrees with the exact coordinate registry."
                )
            if set(receipt_candidate_positions) != set(growth.admitted_positions):
                raise ValueError(
                    "growth receipt candidates do not match admitted coordinates."
                )
            expected = GrowthReceiptExpectation(
                state_fingerprint=projective_state_fingerprint(
                    evaluation.statevector
                ),
                branch_id=str(growth_receipt.branch_id),
                manifold_id=str(backend.manifold_id),
                ordered_scaffold_fingerprint=str(
                    growth_receipt.ordered_scaffold_fingerprint
                ),
                theta_fingerprint=str(growth_receipt.theta_fingerprint),
                old_coordinate_registry_fingerprint=_json_hash(
                    list(old_registry)
                ),
                new_coordinate_registry_fingerprint=_json_hash(
                    list(backend.coordinate_registry)
                ),
                parameterization_tie_map_fingerprint=str(
                    growth_receipt.parameterization_tie_map_fingerprint
                ),
                hamiltonian_fingerprint=str(
                    growth_receipt.hamiltonian_fingerprint
                ),
                candidate_keys=tuple(growth_receipt.candidate_keys),
                insertion_positions=tuple(growth_receipt.insertion_positions),
                old_to_new_registry_mapping=tuple(growth.old_positions),
                rank_rule_fingerprint=str(growth_receipt.rank_rule_fingerprint),
                metric_convention=(
                    "raw_fubini_study_supported_metric_whitened_v1"
                ),
                zero_new_coordinates=True,
                old_gate_subsequence_unchanged=True,
            )
            receipt_validation = validate_formal_growth_geometry_receipt(
                growth_receipt, expected
            )
            if not receipt_validation.valid:
                raise ValueError(
                    "growth geometry receipt provenance mismatch: "
                    + ", ".join(receipt_validation.mismatched_fields)
                )
            horizontal = np.asarray(evaluation.tangents, dtype=complex)
            exact_gram = _sym(
                np.real(np.conjugate(horizontal).T @ horizontal)
            )
            gram_relative_error = _relative_norm(
                gram_override - exact_gram,
                exact_gram,
                self.config.numerical_floor,
            )
            if gram_relative_error > float(self.config.inherited_geometry_tol):
                raise ValueError(
                    "growth geometry receipt Gram disagrees with exact tangents; "
                    f"relative error={gram_relative_error}, "
                    f"receipt={gram_override.tolist()}, exact={exact_gram.tolist()}."
                )
            receipt_candidate_gradient = np.asarray(
                growth_receipt.candidate_gradients, dtype=float
            )
            exact_candidate_gradient = np.asarray(
                evaluation.gradient, dtype=float
            )[list(receipt_candidate_positions)]
            candidate_gradient_delta = (
                exact_candidate_gradient - receipt_candidate_gradient
            )
            candidate_gradient_absolute_error = float(
                np.linalg.norm(candidate_gradient_delta)
            )
            candidate_gradient_scale = float(
                max(
                    np.linalg.norm(receipt_candidate_gradient),
                    np.linalg.norm(exact_candidate_gradient),
                )
            )
            candidate_gradient_error = float(
                candidate_gradient_absolute_error
                / (candidate_gradient_scale + float(self.config.numerical_floor))
            )
            candidate_gradient_relative_tolerance = 10.0 * float(
                self.config.inherited_geometry_tol
            )
            candidate_gradient_absolute_tolerance = float(
                self.config.numerical_floor
            )
            candidate_gradient_comparison_tolerance = float(
                candidate_gradient_absolute_tolerance
                + candidate_gradient_relative_tolerance * candidate_gradient_scale
            )
            if (
                candidate_gradient_absolute_error
                > candidate_gradient_comparison_tolerance
            ):
                raise ValueError(
                    "growth geometry receipt candidate differential mismatch; "
                    f"scale-aware error={candidate_gradient_error}, "
                    f"absolute error={candidate_gradient_absolute_error}, "
                    f"allowed absolute error="
                    f"{candidate_gradient_comparison_tolerance}, "
                    f"scale={candidate_gradient_scale}."
                )
            receipt_validation_payload = {
                "schema": "formal_manifold_growth_receipt_validation_v1",
                "valid": True,
                "reason": str(receipt_validation.reason),
                "receipt_fingerprint": str(
                    growth_receipt.receipt_fingerprint
                ),
                "source_primitive_ids": list(
                    growth_receipt.source_primitive_ids
                ),
                "incremental_query_charge": 0,
                "gram_relative_error": float(gram_relative_error),
                "candidate_gradient_relative_error": float(
                    candidate_gradient_error
                ),
                "candidate_gradient_absolute_error": float(
                    candidate_gradient_absolute_error
                ),
                "candidate_gradient_scale": float(candidate_gradient_scale),
                "candidate_gradient_absolute_tolerance": float(
                    candidate_gradient_absolute_tolerance
                ),
                "candidate_gradient_relative_tolerance": float(
                    candidate_gradient_relative_tolerance
                ),
                "candidate_gradient_comparison_tolerance": float(
                    candidate_gradient_comparison_tolerance
                ),
                "old_positions": list(receipt_old_positions),
                "candidate_positions": list(receipt_candidate_positions),
            }
        frame = self._frame_for(evaluation, gram_override=gram_override)
        if growth_receipt is not None and not np.allclose(
            np.asarray(frame.spectrum, dtype=float)[
                np.asarray(frame.whitening_telemetry.get("metric_retained_mask", []), dtype=bool)
            ],
            np.asarray(growth_receipt.retained_spectrum, dtype=float),
            rtol=2.0e-8,
            atol=2.0e-10,
        ):
            raise ValueError(
                "growth receipt retained spectrum disagrees with shared whitening."
            )
        transition: dict[str, Any] = {
            "kind": (
                "query_closed_initial_growth_anchor"
                if growth_receipt is not None and previous is None
                else "initialize_exact_anchor"
            ),
            "curvature_action": "isotropic_reset",
            "growth_exact_identity_verified": bool(
                growth_receipt is not None and previous is None
            ),
            "rank_before": None if previous is None else int(previous.rank),
            "rank_after": int(frame.rank),
        }
        if receipt_validation_payload is not None:
            transition["query_closure_growth_receipt"] = deepcopy(
                receipt_validation_payload
            )
        B, A = _reset_curvature(frame.rank, self.config)
        qbroyd_inverse_metric = np.asarray(frame.M_R, dtype=float).copy()
        qbroyd_age = 0
        trust_radius = float(self.config.initial_trust_radius)
        if previous is not None:
            same_surface = bool(
                previous.manifold_id == backend.manifold_id
                and previous.parameterization_mode == backend.parameterization_mode
            )
            old_set = set(previous.registry)
            new_set = set(backend.coordinate_registry)
            is_growth = bool(
                same_surface
                and len(new_set) > len(old_set)
                and old_set.issubset(new_set)
            )
            same_registry = bool(
                same_surface and previous.registry == backend.coordinate_registry
            )
            if is_growth:
                growth = grow_zero_coordinates(
                    previous.theta,
                    previous.registry,
                    backend.coordinate_registry,
                    theta_plus=theta,
                )
                projective_distance = _state_projective_distance(
                    previous.statevector, evaluation.statevector
                )
                old_positions = list(growth.old_positions)
                inherited_gram = frame.gram_raw[np.ix_(old_positions, old_positions)]
                old_gram = _sym(
                    np.real(np.conjugate(previous.tangents).T @ previous.tangents)
                )
                inherited_gram_error = _relative_norm(
                    inherited_gram - old_gram,
                    old_gram,
                    self.config.numerical_floor,
                )
                inherited_cross = density_tangent_cross_gram(
                    evaluation.statevector,
                    frame.horizontal_tangents[:, old_positions],
                    previous.statevector,
                    previous.tangents,
                )
                inherited_tangent_map_error = _relative_norm(
                    inherited_cross - old_gram,
                    old_gram,
                    self.config.numerical_floor,
                )
                inherited_gradient_error = float(
                    np.linalg.norm(evaluation.gradient[old_positions] - previous.b)
                    / (
                        max(
                            float(np.linalg.norm(previous.b)),
                            float(np.linalg.norm(evaluation.gradient[old_positions])),
                            1.0,
                        )
                        + float(self.config.numerical_floor)
                    )
                )
                if projective_distance > float(self.config.growth_identity_tol):
                    raise ValueError(
                        "declared zero-coordinate growth changed the physical state; "
                        f"projective distance={projective_distance}."
                    )
                if inherited_gram_error > float(self.config.inherited_geometry_tol):
                    raise ValueError(
                        "declared zero-coordinate growth failed the inherited Gram identity; "
                        f"relative error={inherited_gram_error}."
                    )
                if inherited_tangent_map_error > float(
                    self.config.inherited_geometry_tol
                ):
                    raise ValueError(
                        "declared zero-coordinate growth failed the inherited "
                        "density-tangent map identity; "
                        f"relative error={inherited_tangent_map_error}."
                    )
                if inherited_gradient_error > 10.0 * float(
                    self.config.inherited_geometry_tol
                ):
                    raise ValueError(
                        "declared zero-coordinate growth failed the inherited differential identity; "
                        f"relative error={inherited_gradient_error}."
                    )
                cross_new_old = density_tangent_cross_gram(
                    evaluation.statevector,
                    frame.frame,
                    previous.statevector,
                    previous.frame,
                )
                embedding_error = float(
                    np.linalg.norm(cross_new_old.T @ cross_new_old - np.eye(previous.rank))
                )
                q = int(frame.rank - previous.rank)
                white_embedding = supported_whitened_transport(
                    cross_new_old,
                    previous.regularized_to_raw_frame,
                    frame.regularized_to_raw_frame,
                )
                raw_embedding = np.asarray(cross_new_old, dtype=float)
                embedding_transport_valid = bool(
                    q >= 0
                    and embedding_error <= 5.0e-6
                    and white_embedding.valid
                )
                if not embedding_transport_valid:
                    B, A = _reset_curvature(frame.rank, self.config)
                    qbroyd_inverse_metric = np.asarray(
                        frame.M_R, dtype=float
                    ).copy()
                    qbroyd_age = 0
                    trust_radius = float(
                        max(
                            self.config.min_trust_radius,
                            previous.trust_radius * self.config.trust_shrink,
                        )
                    )
                    if q < 0:
                        reset_reason = "supported_rank_decreased"
                    elif embedding_error > 5.0e-6:
                        reset_reason = "nonisometric_supported_frame_embedding"
                    else:
                        reset_reason = str(white_embedding.reason)
                    transition.update(
                        {
                            "kind": (
                                "exact_zero_coordinate_growth_frame_reset"
                            ),
                            "curvature_action": (
                                "incompatible_supported_frame_isotropic_reset"
                            ),
                            "growth_exact_identity_verified": True,
                            "supported_frame_transport_valid": False,
                            "supported_frame_reset_reason": str(reset_reason),
                            "old_positions": old_positions,
                            "admitted_positions": list(
                                growth.admitted_positions
                            ),
                            "projective_distance": projective_distance,
                            "inherited_gram_relative_error": (
                                inherited_gram_error
                            ),
                            "inherited_tangent_map_relative_error": (
                                inherited_tangent_map_error
                            ),
                            "inherited_gradient_relative_error": (
                                inherited_gradient_error
                            ),
                            "embedding_isometry_residual": embedding_error,
                            "whitened_embedding_condition_number": float(
                                white_embedding.condition_number
                            ),
                            "whitened_embedding_pairing_residual": float(
                                white_embedding.pairing_residual
                            ),
                            "whitened_embedding_reason": str(
                                white_embedding.reason
                            ),
                            "whitening_before": str(previous.whitening_id),
                            "whitening_after": str(frame.whitening_id),
                            "rank_gain": q,
                        }
                    )
                else:
                    if q:
                        if previous.rank:
                            u_columns, _singular, _vh = np.linalg.svd(
                                raw_embedding, full_matrices=False
                            )
                            projector = _sym(u_columns @ u_columns.T)
                        else:
                            projector = np.zeros(
                                (frame.rank, frame.rank), dtype=float
                            )
                        complement_projector = _sym(
                            np.eye(frame.rank, dtype=float) - projector
                        )
                    else:
                        complement_projector = np.zeros(
                            (frame.rank, frame.rank), dtype=float
                        )
                    if self.config.curvature_branch == FORMAL_CURVATURE_INVERSE_RBFGS:
                        if previous.rank:
                            B = raw_embedding @ previous.B @ raw_embedding.T
                        else:
                            B = np.zeros(
                                (frame.rank, frame.rank), dtype=float
                            )
                        if q:
                            if previous.rank:
                                scale = float(
                                    np.median(
                                        np.linalg.eigvalsh(previous.B)
                                    )
                                )
                            else:
                                scale = float(
                                    self.config.initial_inverse_curvature
                                )
                            scale = float(
                                np.clip(
                                    scale,
                                    self.config.inverse_curvature_min,
                                    self.config.inverse_curvature_max,
                                )
                            )
                            B = B + scale * complement_projector
                        B = _sym(B)
                        A = np.zeros((0, 0), dtype=float)
                    else:
                        if previous.rank:
                            A = raw_embedding @ previous.A @ raw_embedding.T
                        else:
                            A = np.zeros(
                                (frame.rank, frame.rank), dtype=float
                            )
                        if q:
                            A = A + _clipped_direct_curvature_scale(
                                previous.A, self.config
                            ) * complement_projector
                        A = _sym(A)
                        B = np.zeros((0, 0), dtype=float)
                    qbroyd_inverse_metric = np.asarray(
                        frame.M_R, dtype=float
                    ).copy()
                    qbroyd_age = 0
                    trust_radius = float(previous.trust_radius)
                    transition.update(
                        {
                            "kind": "exact_zero_coordinate_growth",
                            "curvature_action": "old_physical_block_plus_isotropic_residual_prior",
                            "growth_exact_identity_verified": True,
                            "supported_frame_transport_valid": True,
                            "old_positions": old_positions,
                            "admitted_positions": list(
                                growth.admitted_positions
                            ),
                            "projective_distance": projective_distance,
                            "inherited_gram_relative_error": inherited_gram_error,
                            "inherited_tangent_map_relative_error": (
                                inherited_tangent_map_error
                            ),
                            "inherited_gradient_relative_error": inherited_gradient_error,
                            "embedding_isometry_residual": embedding_error,
                            "whitened_embedding_condition_number": float(
                                white_embedding.condition_number
                            ),
                            "whitened_embedding_pairing_residual": float(
                                white_embedding.pairing_residual
                            ),
                            "whitening_before": str(previous.whitening_id),
                            "whitening_after": str(frame.whitening_id),
                            "rank_gain": q,
                        }
                    )
            elif same_registry and np.allclose(
                theta, previous.theta, rtol=0.0, atol=self.config.growth_identity_tol
            ):
                if frame.rank == previous.rank:
                    cross = density_tangent_cross_gram(
                        evaluation.statevector,
                        frame.frame,
                        previous.statevector,
                        previous.frame,
                    )
                    transport = endpoint_procrustes(
                        cross, alignment_floor=self.config.alignment_sigma_min
                    )
                    white_transport = supported_whitened_transport(
                        transport.Q,
                        previous.regularized_to_raw_frame,
                        frame.regularized_to_raw_frame,
                    )
                    if transport.valid:
                        if self.config.curvature_branch == FORMAL_CURVATURE_INVERSE_RBFGS:
                            B = _sym(transport.Q @ previous.B @ transport.Q.T)
                            A = np.zeros((0, 0), dtype=float)
                        else:
                            A = _sym(transport.Q @ previous.A @ transport.Q.T)
                            B = np.zeros((0, 0), dtype=float)
                        qbroyd_inverse_metric = np.asarray(
                            frame.M_R, dtype=float
                        ).copy()
                        qbroyd_age = 0
                        trust_radius = float(previous.trust_radius)
                        transition.update(
                            {
                                "kind": "same_endpoint_exact_reuse",
                                "curvature_action": "endpoint_frame_transport",
                                "transport_sigma_min": float(transport.sigma_min),
                                "whitened_transport_condition_number": float(
                                    white_transport.condition_number
                                ),
                                "whitening_before": str(previous.whitening_id),
                                "whitening_after": str(frame.whitening_id),
                            }
                        )
                    else:
                        transition.update(
                            {
                                "kind": "same_endpoint_alignment_reset",
                                "transport_sigma_min": float(transport.sigma_min),
                                "whitened_transport_reason": str(
                                    white_transport.reason
                                ),
                            }
                        )
                else:
                    transition["kind"] = "same_registry_rank_reset"
            else:
                transition.update(
                    {
                        "kind": "exact_reanchor_after_external_structural_change",
                        "reason": "registry_or_coordinate_anchor_changed",
                    }
                )
        if frame.gap_status == "unstable":
            B, A = _reset_curvature(frame.rank, self.config)
            qbroyd_inverse_metric = np.asarray(frame.M_R, dtype=float).copy()
            qbroyd_age = 0
            trust_radius = float(
                max(
                    self.config.min_trust_radius,
                    min(trust_radius, self.config.initial_trust_radius)
                    * self.config.trust_shrink,
                )
            )
            transition["curvature_action"] = (
                "retained_gap_unstable_isotropic_reset"
            )
            transition["rank_instability"] = True
        metadata = {
            "schema": "formal_manifold_warm_state_metadata_v1",
            "metric_provenance": (
                "query_closed_growth_receipt_reused_and_exact_validated"
                if growth_receipt is not None
                else "exact_state_computed"
            ),
            "curvature_provenance": (
                "regularized_prior"
                if "reset" in str(transition["curvature_action"])
                else "transported_or_growth_prior"
            ),
            "curvature_branch": str(self.config.curvature_branch),
            "qbroyd_mode": "shadow_predictor_exact_refresh_v1",
            "statistically_calibrated": False,
            "transition": deepcopy(transition),
            "discarded_gram_residual": float(frame.discarded_gram_residual),
            "rank_threshold": float(frame.threshold),
            "supported_metric_config": self.config.supported_metric.as_dict(),
            "whitening_id": str(frame.whitening_id),
            "curvature_whitening_id": str(frame.whitening_id),
            "qbroyd_whitening_id": str(frame.whitening_id),
            "curvature_coordinate_system": "supported_raw_fs_orthonormal_frame_v1",
            "shared_solver_coordinate_system": "supported_regularized_metric_v1",
            "raw_metric_condition_number": frame.whitening_telemetry.get(
                "raw_metric_condition_number"
            ),
            "retained_metric_condition_number": frame.whitening_telemetry.get(
                "retained_metric_condition_number"
            ),
            "metric_retained_mask": frame.whitening_telemetry.get(
                "metric_retained_mask"
            ),
        }
        if receipt_validation_payload is not None:
            metadata["query_closure_growth_receipt"] = deepcopy(
                receipt_validation_payload
            )
        state = _warm_state_from_frame(
            theta=theta,
            backend=backend,
            evaluation=evaluation,
            frame=frame,
            B=B,
            A=A,
            curvature_branch=str(self.config.curvature_branch),
            qbroyd_inverse_metric=qbroyd_inverse_metric,
            trust_radius=trust_radius,
            qbroyd_age=qbroyd_age,
            metadata=metadata,
        )
        return state, transition

    def propose(
        self,
        backend: ExactStateBackend,
        x0: np.ndarray | Sequence[float],
        *,
        maxiter: int,
        callback: Callable[[Mapping[str, Any]], None] | None = None,
        growth_receipt: FormalGrowthGeometryReceipt | None = None,
    ) -> FormalManifoldResult:
        if self._pending is not None:
            raise RuntimeError("commit or rollback the pending proposal before proposing again.")
        if not isinstance(backend, ExactStateBackend):
            raise TypeError("backend must be ExactStateBackend.")
        theta = _finite_real_vector(x0, name="x0")
        if theta.size != len(backend.coordinate_registry):
            raise ValueError("x0 length must match the backend coordinate registry.")
        max_iterations = int(maxiter)
        if max_iterations < 0:
            raise ValueError("maxiter must be nonnegative.")
        evaluation = backend.evaluate(theta)
        nfev = 1
        current, transition = self._anchor(
            backend,
            theta,
            evaluation,
            growth_receipt=growth_receipt,
        )
        step_rows: list[dict[str, Any]] = []
        accepted_steps = 0
        rejection_count = 0
        success = False
        message = "maximum iterations reached"

        for iteration in range(max_iterations):
            grad = _gradient_frame(current)
            grad_norm = float(np.linalg.norm(grad))
            if grad_norm <= float(self.config.gradient_tol):
                success = True
                message = "resolved Riemannian gradient tolerance reached"
                break
            alpha = 1.0
            accepted_eval: ExactStateEvaluation | None = None
            accepted_theta: np.ndarray | None = None
            line_evals = 0
            direct_solve: JointLinearSolveResult | None = None
            model_predicted_drop: float | None = None
            model_ratio: float | None = None
            rejection_reason = "armijo_rejection"
            if current.curvature_branch == FORMAL_CURVATURE_INVERSE_RBFGS:
                z_trial = -np.asarray(current.B @ grad, dtype=float)
                z_norm = float(np.linalg.norm(z_trial))
                if z_norm > float(current.trust_radius):
                    z_trial *= float(current.trust_radius / z_norm)
                    z_norm = float(current.trust_radius)
                if z_norm <= float(self.config.step_tol):
                    success = True
                    message = "resolved tangent step tolerance reached"
                    break
                p_trial = np.asarray(
                    current.raw_orthonormalizer @ z_trial, dtype=float
                )
                directional = float(current.b @ p_trial)
                if not directional < 0.0:
                    current.B = _isotropic_inverse_curvature(
                        current.rank, self.config
                    )
                    z_trial = -grad
                    z_norm = float(np.linalg.norm(z_trial))
                    if z_norm > float(current.trust_radius):
                        z_trial *= float(current.trust_radius / z_norm)
                    p_trial = np.asarray(
                        current.raw_orthonormalizer @ z_trial, dtype=float
                    )
                    directional = float(current.b @ p_trial)
                if not directional < 0.0:
                    message = "failed to construct a descent direction"
                    break
                for _line_index in range(int(self.config.line_search_max_steps)):
                    theta_trial = np.asarray(
                        current.theta + alpha * p_trial, dtype=float
                    )
                    trial_eval = backend.evaluate(theta_trial)
                    nfev += 1
                    line_evals += 1
                    if float(trial_eval.energy) <= float(
                        current.energy
                        + float(self.config.armijo_c1) * alpha * directional
                    ):
                        accepted_eval = trial_eval
                        accepted_theta = theta_trial
                        break
                    alpha *= float(self.config.line_search_shrink)
            else:
                direct_solve = solve_direct_sr1_trust_step(
                    current.A,
                    grad,
                    trust_radius=float(current.trust_radius),
                    supported_metric=self.config.supported_metric,
                )
                if direct_solve.feasible:
                    z_trial = np.asarray(direct_solve.joint_step, dtype=float)
                    z_norm = float(np.linalg.norm(z_trial))
                    if z_norm <= float(self.config.step_tol):
                        success = True
                        message = "resolved tangent step tolerance reached"
                        break
                    p_trial = np.asarray(
                        current.raw_orthonormalizer @ z_trial, dtype=float
                    )
                    directional = float(current.b @ p_trial)
                    theta_trial = np.asarray(current.theta + p_trial, dtype=float)
                    trial_eval = backend.evaluate(theta_trial)
                    nfev += 1
                    line_evals = 1
                    model_predicted_drop = float(direct_solve.predicted_reduction)
                    actual_trial_drop = float(current.energy - trial_eval.energy)
                    model_ratio = (
                        float(actual_trial_drop / model_predicted_drop)
                        if model_predicted_drop > 0.0
                        else None
                    )
                    if (
                        model_ratio is not None
                        and actual_trial_drop > 0.0
                        and model_ratio >= float(self.config.armijo_c1)
                    ):
                        accepted_eval = trial_eval
                        accepted_theta = theta_trial
                    else:
                        rejection_reason = "direct_sr1_trust_model_rejection"
                else:
                    z_trial = np.zeros(current.rank, dtype=float)
                    z_norm = 0.0
                    p_trial = np.zeros(current.theta.size, dtype=float)
                    directional = 0.0
                    rejection_reason = (
                        "direct_sr1_shared_trust_solve_failure:"
                        f"{direct_solve.reason}"
                    )

            if accepted_eval is None or accepted_theta is None:
                rejection_count += 1
                current.trust_radius = float(
                    max(
                        self.config.min_trust_radius,
                        current.trust_radius * self.config.trust_shrink,
                    )
                )
                row = {
                    "iteration": int(iteration + 1),
                    "accepted": False,
                    "energy": float(current.energy),
                    "gradient_norm": grad_norm,
                    "line_search_evaluations": int(line_evals),
                    "trust_radius": float(current.trust_radius),
                    "reason": rejection_reason,
                    "curvature_branch": str(current.curvature_branch),
                    "direct_trust_solve": (
                        None
                        if direct_solve is None
                        else {
                            "feasible": bool(direct_solve.feasible),
                            "reason": str(direct_solve.reason),
                            "predicted_reduction": float(
                                direct_solve.predicted_reduction
                            ),
                            "trust_lambda": float(direct_solve.trust_lambda),
                            "model_ratio": model_ratio,
                            "telemetry": deepcopy(direct_solve.telemetry),
                        }
                    ),
                }
                step_rows.append(row)
                if callback is not None:
                    callback(dict(row))
                if rejection_count >= int(self.config.max_rejections):
                    current.B, current.A = _reset_curvature(
                        current.rank, self.config
                    )
                    current.qbroyd_inverse_metric = np.asarray(
                        current.M_R, dtype=float
                    ).copy()
                    current.qbroyd_age = 0
                    current.metadata["last_reset_reason"] = (
                        "repeated_globalization_rejection"
                    )
                    message = "repeated globalization rejection; curvature reset"
                    break
                continue

            endpoint_frame = self._frame_for(accepted_eval)
            p_accepted = np.asarray(alpha * p_trial, dtype=float)
            x_sec = np.asarray(alpha * z_trial, dtype=float)
            raw_frame_secant = x_sec.copy()
            actual_drop = float(current.energy - accepted_eval.energy)
            qbroyd_info: dict[str, Any] = {
                "mode": "shadow_predictor_exact_refresh_v1",
                "observed_endpoint_metric": True,
                "epsilon": None,
                "innovation": None,
                "severity": "none",
                "coordinate_system": "shared_retained_coordinate_quotient_v1",
                "observation_overwrites_prediction": True,
            }
            M_pred_old: np.ndarray | None = None
            if current.rank:
                epsilon = float(
                    self.config.qbroyd_epsilon0 / (current.qbroyd_age + 1)
                )
                try:
                    b_R = np.asarray(current.Z.T @ current.b, dtype=float)
                    M_pred_old = qbroyd_inverse_update(
                        current.qbroyd_inverse_metric,
                        b_R,
                        epsilon,
                        numerical_floor=self.config.numerical_floor,
                    )
                    qbroyd_info.update(
                        {
                            "epsilon": epsilon,
                            "shadow_age": int(current.qbroyd_age + 1),
                        }
                    )
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    qbroyd_info.update(
                        {
                            "epsilon": epsilon,
                            "severity": "hard",
                            "reason": f"qbroyd_shadow_failure:{type(exc).__name__}",
                        }
                    )

            rank_transition = bool(
                endpoint_frame.rank != current.rank
                or endpoint_frame.gap_status == "unstable"
                or current.gap_status == "unstable"
            )
            transport_info: dict[str, Any] = {
                "valid": False,
                "sigma_min": None,
                "reason": "rank_transition",
            }
            update_info: RBFGSUpdate | None = None
            sr1_update_info: DirectSR1Update | None = None
            qbroyd_inverse_metric_new = np.asarray(
                endpoint_frame.M_R, dtype=float
            ).copy()
            qbroyd_age_new = 0
            if (
                not rank_transition
                and current.rank
                and M_pred_old is not None
            ):
                try:
                    G_pred_coordinate = _sym(
                        current.Z @ np.linalg.inv(M_pred_old) @ current.Z.T
                    )
                    innovation = _relative_norm(
                        endpoint_frame.gram_retained - G_pred_coordinate,
                        endpoint_frame.gram_retained,
                        self.config.numerical_floor,
                    )
                    severity = (
                        "hard"
                        if innovation > float(self.config.metric_innovation_hard)
                        else (
                            "soft"
                            if innovation > float(self.config.metric_innovation_soft)
                            else "none"
                        )
                    )
                    qbroyd_info.update(
                        {
                            "innovation": float(innovation),
                            "severity": severity,
                            "prediction_basis": "current_retained_coordinate_range",
                            "endpoint_reanchor_basis": "observed_endpoint_retained_range",
                        }
                    )
                except np.linalg.LinAlgError:
                    qbroyd_info.update(
                        {
                            "severity": "hard",
                            "reason": "qbroyd_coordinate_inverse_failure",
                        }
                    )
            if rank_transition:
                B_new, A_new = _reset_curvature(endpoint_frame.rank, self.config)
                trust_new = float(
                    max(
                        self.config.min_trust_radius,
                        current.trust_radius * self.config.trust_shrink,
                    )
                )
                curvature_action = "rank_change_reset_at_accepted_endpoint"
            else:
                cross = density_tangent_cross_gram(
                    accepted_eval.statevector,
                    endpoint_frame.frame,
                    current.statevector,
                    current.frame,
                )
                transport = endpoint_procrustes(
                    cross, alignment_floor=self.config.alignment_sigma_min
                )
                transport_info = {
                    "valid": bool(transport.valid),
                    "sigma_min": float(transport.sigma_min),
                    "reason": str(transport.reason),
                    "max_principal_angle": float(
                        math.acos(max(-1.0, min(1.0, transport.sigma_min)))
                    ),
                }
                white_transport = supported_whitened_transport(
                    transport.Q,
                    current.regularized_to_raw_frame,
                    endpoint_frame.regularized_to_raw_frame,
                )
                transport_info.update(
                    {
                        "whitened_valid": bool(white_transport.valid),
                        "whitened_reason": str(white_transport.reason),
                        "whitened_condition_number": float(
                            white_transport.condition_number
                        ),
                        "whitened_pairing_residual": float(
                            white_transport.pairing_residual
                        ),
                        "whitening_before": str(current.whitening_id),
                        "whitening_after": str(endpoint_frame.whitening_id),
                    }
                )
                if not transport.valid:
                    B_new, A_new = _reset_curvature(
                        endpoint_frame.rank, self.config
                    )
                    trust_new = float(
                        max(
                            self.config.min_trust_radius,
                            current.trust_radius * self.config.trust_shrink,
                        )
                    )
                    curvature_action = "transport_failure_reset_at_accepted_endpoint"
                elif str(qbroyd_info.get("severity", "none")) == "hard":
                    B_new, A_new = _reset_curvature(
                        endpoint_frame.rank, self.config
                    )
                    trust_new = float(
                        max(
                            self.config.min_trust_radius,
                            current.trust_radius * self.config.trust_shrink,
                        )
                    )
                    curvature_action = (
                        "hard_metric_innovation_reset_skip_crossing_secant"
                    )
                else:
                    Q = np.asarray(transport.Q, dtype=float)
                    grad_new = (
                        np.asarray(
                            endpoint_frame.raw_orthonormalizer.T
                            @ accepted_eval.gradient,
                            dtype=float,
                        )
                        if endpoint_frame.rank
                        else np.zeros(0, dtype=float)
                    )
                    s = np.asarray(Q @ x_sec, dtype=float)
                    y = np.asarray(grad_new - Q @ grad, dtype=float)
                    invalidating_update = False
                    if current.curvature_branch == FORMAL_CURVATURE_INVERSE_RBFGS:
                        B_tilde = _sym(Q @ current.B @ Q.T)
                        update_info = powell_damped_inverse_rbfgs(
                            B_tilde,
                            s,
                            y,
                            eta=self.config.powell_eta,
                            curvature_guard=self.config.curvature_guard,
                            postcondition_tol=self.config.postcondition_tol,
                            numerical_floor=self.config.numerical_floor,
                        )
                        invalidating_update_reasons = {
                            "powell_denominator_failure",
                            "spd_postcondition_failure",
                            "secant_postcondition_failure",
                        }
                        invalidating_update = bool(
                            str(update_info.reason) in invalidating_update_reasons
                        )
                        if invalidating_update:
                            B_new, A_new = _reset_curvature(
                                endpoint_frame.rank, self.config
                            )
                            curvature_action = (
                                "rbfgs_postcondition_failure_isotropic_reset"
                            )
                        else:
                            B_new = np.asarray(update_info.B, dtype=float)
                            A_new = np.zeros((0, 0), dtype=float)
                            curvature_action = (
                                "powell_damped_inverse_rbfgs"
                                if update_info.applied
                                else "transported_curvature_secant_skipped"
                            )
                        try:
                            A_model = np.linalg.inv(current.B)
                            predicted_drop = float(
                                -(
                                    grad @ x_sec
                                    + 0.5 * float(x_sec @ A_model @ x_sec)
                                )
                            )
                        except np.linalg.LinAlgError:
                            predicted_drop = float("nan")
                        ratio = (
                            float(actual_drop / predicted_drop)
                            if math.isfinite(predicted_drop)
                            and predicted_drop > 0.0
                            else None
                        )
                    else:
                        A_tilde = _sym(Q @ current.A @ Q.T)
                        sr1_update_info = guarded_direct_sr1(
                            A_tilde,
                            s,
                            y,
                            curvature_guard=self.config.curvature_guard,
                            postcondition_tol=self.config.postcondition_tol,
                            numerical_floor=self.config.numerical_floor,
                        )
                        invalidating_update = bool(
                            sr1_update_info.reason
                            in {"nonfinite_sr1_update", "secant_postcondition_failure"}
                        )
                        if invalidating_update:
                            B_new, A_new = _reset_curvature(
                                endpoint_frame.rank, self.config
                            )
                            curvature_action = (
                                "direct_sr1_postcondition_failure_isotropic_reset"
                            )
                        else:
                            B_new = np.zeros((0, 0), dtype=float)
                            A_new = np.asarray(sr1_update_info.A, dtype=float)
                            curvature_action = (
                                "guarded_direct_sr1"
                                if sr1_update_info.applied
                                else "transported_direct_curvature_secant_skipped"
                            )
                        predicted_drop = (
                            float(model_predicted_drop)
                            if model_predicted_drop is not None
                            else float(
                                -(
                                    grad @ x_sec
                                    + 0.5 * float(x_sec @ current.A @ x_sec)
                                )
                            )
                        )
                        ratio = (
                            float(model_ratio)
                            if model_ratio is not None
                            else (
                                float(actual_drop / predicted_drop)
                                if math.isfinite(predicted_drop)
                                and predicted_drop > 0.0
                                else None
                            )
                        )
                    trust_new = float(current.trust_radius)
                    if ratio is None or ratio < 0.25:
                        trust_new *= float(self.config.trust_shrink)
                    elif ratio > 0.75 and z_norm >= 0.8 * float(current.trust_radius):
                        trust_new *= float(self.config.trust_expand)
                    trust_new = float(
                        np.clip(
                            trust_new,
                            self.config.min_trust_radius,
                            self.config.max_trust_radius,
                        )
                    )
                    if invalidating_update:
                        trust_new = float(
                            max(
                                self.config.min_trust_radius,
                                trust_new * self.config.trust_shrink,
                            )
                        )

            endpoint_metadata = {
                "schema": "formal_manifold_warm_state_metadata_v1",
                "metric_provenance": "exact_state_computed",
                "qbroyd_mode": "shadow_predictor_exact_refresh_v1",
                "qbroyd_shadow": deepcopy(qbroyd_info),
                "curvature_branch": str(current.curvature_branch),
                "curvature_action": curvature_action,
                "curvature_provenance": (
                    "secant_inferred"
                    if (
                        (update_info is not None and update_info.applied)
                        or (sr1_update_info is not None and sr1_update_info.applied)
                    )
                    else "transported_or_regularized_prior"
                ),
                "transport": deepcopy(transport_info),
                "rank_transition": bool(rank_transition),
                "discarded_gram_residual": float(
                    endpoint_frame.discarded_gram_residual
                ),
                "rank_threshold": float(endpoint_frame.threshold),
                "statistically_calibrated": False,
                "supported_metric_config": self.config.supported_metric.as_dict(),
                "whitening_id": str(endpoint_frame.whitening_id),
                "curvature_whitening_id": str(endpoint_frame.whitening_id),
                "qbroyd_whitening_id": str(endpoint_frame.whitening_id),
                "curvature_coordinate_system": "supported_raw_fs_orthonormal_frame_v1",
                "shared_solver_coordinate_system": "supported_regularized_metric_v1",
                "raw_metric_condition_number": endpoint_frame.whitening_telemetry.get(
                    "raw_metric_condition_number"
                ),
                "retained_metric_condition_number": endpoint_frame.whitening_telemetry.get(
                    "retained_metric_condition_number"
                ),
                "metric_retained_mask": endpoint_frame.whitening_telemetry.get(
                    "metric_retained_mask"
                ),
            }
            growth_receipt_origin = current.metadata.get(
                "query_closure_growth_receipt_origin",
                current.metadata.get("query_closure_growth_receipt"),
            )
            if isinstance(growth_receipt_origin, Mapping):
                endpoint_metadata["query_closure_growth_receipt_origin"] = {
                    **deepcopy(dict(growth_receipt_origin)),
                    "applies_to_current_endpoint": False,
                    "scope": "accepted_zero_growth_anchor_origin",
                }
            current = _warm_state_from_frame(
                theta=accepted_theta,
                backend=backend,
                evaluation=accepted_eval,
                frame=endpoint_frame,
                B=B_new,
                A=A_new,
                curvature_branch=str(current.curvature_branch),
                qbroyd_inverse_metric=qbroyd_inverse_metric_new,
                trust_radius=trust_new,
                qbroyd_age=qbroyd_age_new,
                metadata=endpoint_metadata,
            )
            accepted_steps += 1
            rejection_count = 0
            row = {
                "iteration": int(iteration + 1),
                "accepted": True,
                "alpha": float(alpha),
                "energy": float(current.energy),
                "actual_drop": actual_drop,
                "gradient_norm_before": grad_norm,
                "line_search_evaluations": int(line_evals),
                "rank": int(current.rank),
                "gap_status": str(current.gap_status),
                "trust_radius": float(current.trust_radius),
                "transport": deepcopy(transport_info),
                "qbroyd_shadow": deepcopy(qbroyd_info),
                "coordinate_directional_pairing_residual": float(
                    abs(float(alpha * directional) - float(grad @ x_sec))
                ),
                "raw_frame_secant_norm": float(
                    np.linalg.norm(raw_frame_secant)
                ),
                "curvature_action": curvature_action,
                "curvature_branch": str(current.curvature_branch),
                "direct_trust_solve": (
                    None
                    if direct_solve is None
                    else {
                        "feasible": bool(direct_solve.feasible),
                        "reason": str(direct_solve.reason),
                        "predicted_reduction": float(
                            direct_solve.predicted_reduction
                        ),
                        "model_ratio": model_ratio,
                        "trust_lambda": float(direct_solve.trust_lambda),
                        "telemetry": deepcopy(direct_solve.telemetry),
                    }
                ),
                "rbfgs": (
                    None
                    if update_info is None
                    else {
                        "applied": bool(update_info.applied),
                        "damped": bool(update_info.damped),
                        "reason": str(update_info.reason),
                        "curvature_raw": float(update_info.curvature_raw),
                        "curvature_used": float(update_info.curvature_used),
                        "postcondition_residual": (
                            None
                            if update_info.postcondition_residual is None
                            else float(update_info.postcondition_residual)
                        ),
                    }
                ),
                "direct_sr1": (
                    None
                    if sr1_update_info is None
                    else {
                        "applied": bool(sr1_update_info.applied),
                        "reason": str(sr1_update_info.reason),
                        "denominator": float(sr1_update_info.denominator),
                        "guard_threshold": float(
                            sr1_update_info.guard_threshold
                        ),
                        "postcondition_residual": (
                            None
                            if sr1_update_info.postcondition_residual is None
                            else float(sr1_update_info.postcondition_residual)
                        ),
                        "minimum_eigenvalue": float(
                            np.min(np.linalg.eigvalsh(sr1_update_info.A))
                        )
                        if sr1_update_info.A.size
                        else None,
                    }
                ),
            }
            step_rows.append(row)
            if callback is not None:
                callback(dict(row))

        else:
            if max_iterations == 0:
                message = "zero optimization iterations requested"

        final_gradient_norm = float(np.linalg.norm(_gradient_frame(current)))
        if final_gradient_norm <= float(self.config.gradient_tol):
            success = True
            if message == "maximum iterations reached":
                message = "resolved Riemannian gradient tolerance reached"
        token = str(uuid.uuid4())
        info = {
            "schema": "formal_manifold_reoptimization_result_v1",
            "route": FORMAL_MANIFOLD_ROUTE,
            "transition": deepcopy(transition),
            "rank": int(current.rank),
            "retained_gap": (
                None if current.retained_gap is None else float(current.retained_gap)
            ),
            "gap_status": str(current.gap_status),
            "trust_radius": float(current.trust_radius),
            "qbroyd_age": int(current.qbroyd_age),
            "qbroyd_mode": "shadow_predictor_exact_refresh_v1",
            "curvature_branch": str(current.curvature_branch),
            "curvature_branches_mutually_exclusive": True,
            "authoritative_metric": "exact_state_endpoint_refresh_v1",
            "supported_metric_whitening_policy": (
                JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1
            ),
            "optimizer_coordinate_system": "supported_raw_fs_orthonormal_frame_v1",
            "shared_solver_coordinate_system": "supported_regularized_metric_v1",
            "whitening_id": str(current.whitening_id),
            "frame_id": str(current.frame_id),
            "logical_range_id": str(current.logical_range_id),
            "curvature_whitening_id": str(
                current.metadata.get("curvature_whitening_id")
            ),
            "curvature_frame_id": str(
                current.metadata.get("curvature_frame_id")
            ),
            "qbroyd_whitening_id": str(
                current.metadata.get("qbroyd_whitening_id")
            ),
            "qbroyd_logical_range_id": str(
                current.metadata.get("qbroyd_logical_range_id")
            ),
            "whitening_telemetry": {
                "raw_metric_condition_number": current.metadata.get(
                    "raw_metric_condition_number"
                ),
                "retained_metric_condition_number": current.metadata.get(
                    "retained_metric_condition_number"
                ),
                "metric_retained_mask": deepcopy(
                    current.metadata.get("metric_retained_mask")
                ),
            },
            "qbang_momentum_active": False,
            "accepted_inner_steps": int(accepted_steps),
            "final_gradient_norm": final_gradient_norm,
            "steps": deepcopy(step_rows),
            "candidate_scoring_unchanged": False,
            "candidate_scoring_policy": (
                "formal_manifold_query_closed_phase_models_v1"
            ),
            "static_route_identity_unchanged": True,
            "statistically_calibrated": False,
        }
        result = FormalManifoldResult(
            x=np.asarray(current.theta, dtype=float).copy(),
            fun=float(current.energy),
            nfev=int(nfev),
            nit=int(len(step_rows)),
            success=bool(success),
            message=str(message),
            warm_state=deepcopy(current),
            info=info,
            _session_token=token,
        )
        self._pending = deepcopy(result)
        return result


@dataclass
class FormalManifoldBranchRuntime:
    """Branch-owned FM transaction, curvature, and query-accounting state.

    Forking deep-copies the accepted manifold state and query ledger.  A
    rollback only discards the pending proposal on this branch; it never
    removes an accepted generator or restores an earlier ansatz structure.
    """

    branch_id: str
    query_origin_branch_id: str
    session: FormalManifoldSession
    query_ledger: QueryPrimitiveLedger = field(default_factory=QueryPrimitiveLedger)

    def __post_init__(self) -> None:
        self.branch_id = str(self.branch_id)
        self.query_origin_branch_id = str(self.query_origin_branch_id)
        if not self.branch_id:
            raise ValueError("formal-manifold branch_id must be nonempty.")
        if not self.query_origin_branch_id:
            raise ValueError(
                "formal-manifold query_origin_branch_id must be nonempty."
            )
        if not isinstance(self.session, FormalManifoldSession):
            raise TypeError("session must be a FormalManifoldSession.")
        if self.session.branch_id != self.branch_id:
            raise ValueError(
                "formal-manifold runtime/session branch provenance disagrees."
            )
        if not isinstance(self.query_ledger, QueryPrimitiveLedger):
            raise TypeError("query_ledger must be a QueryPrimitiveLedger.")

    @classmethod
    def root(
        cls,
        *,
        session: FormalManifoldSession,
        query_ledger: QueryPrimitiveLedger | None = None,
        branch_id: str = "beam_branch:0",
    ) -> "FormalManifoldBranchRuntime":
        """Bind one committed single-frontier session to the beam root."""

        if not isinstance(session, FormalManifoldSession):
            raise TypeError("session must be a FormalManifoldSession.")
        rooted = session.fork(branch_id=str(branch_id))
        return cls(
            branch_id=str(branch_id),
            query_origin_branch_id=str(branch_id),
            session=rooted,
            query_ledger=(
                QueryPrimitiveLedger()
                if query_ledger is None
                else _clone_query_ledger(query_ledger)
            ),
        )

    def fork(self, *, branch_id: str) -> "FormalManifoldBranchRuntime":
        """Create a non-aliasing speculative child runtime."""

        if self.session.branch_id != self.branch_id:
            raise RuntimeError(
                "formal-manifold runtime/session branch provenance disagrees."
            )
        child_id = str(branch_id)
        return FormalManifoldBranchRuntime(
            branch_id=child_id,
            query_origin_branch_id=str(self.branch_id),
            session=self.session.fork(branch_id=child_id),
            query_ledger=_clone_query_ledger(self.query_ledger),
        )

    def propose(
        self,
        backend: ExactStateBackend,
        x0: np.ndarray | Sequence[float],
        *,
        maxiter: int,
        callback: Callable[[Mapping[str, Any]], None] | None = None,
        growth_receipt: FormalGrowthGeometryReceipt | None = None,
    ) -> FormalManifoldResult:
        return self.session.propose(
            backend,
            x0,
            maxiter=int(maxiter),
            callback=callback,
            growth_receipt=growth_receipt,
        )

    def commit(self, result: FormalManifoldResult) -> QueryPrimitiveLedger:
        self.session.commit(result)
        self.query_origin_branch_id = str(self.branch_id)
        return _clone_query_ledger(self.query_ledger)

    def rollback(self) -> QueryPrimitiveLedger:
        self.session.rollback()
        self.query_origin_branch_id = str(self.branch_id)
        return _clone_query_ledger(self.query_ledger)

    def checkpoint_payload(self) -> dict[str, Any]:
        """Serialize one branch without live statevectors or shared aliases."""

        transaction = self.session.transaction_payload()
        warm_state = self.session.checkpoint_payload()
        return {
            "schema": "formal_manifold_beam_branch_runtime_checkpoint_v1",
            "branch_id": str(self.branch_id),
            "parent_branch_id": self.session.parent_branch_id,
            "query_origin_branch_id": str(self.query_origin_branch_id),
            "transaction_state": transaction,
            "warm_state": warm_state,
            "query_ledger": self.query_ledger.checkpoint_payload(),
            "route_composition": self.session.route_composition.as_dict(),
            "formal_manifold_config_sha256": _json_hash(
                self.session.config.as_dict()
            ),
            "structural_rollback_supported": False,
            "rollback_scope": "pending_proposal_only",
        }

    @classmethod
    def restore_checkpoint_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        backend: ExactStateBackend | None = None,
        config: FormalManifoldConfig | None = None,
        expected_route_composition: (
            Mapping[str, Any] | FormalManifoldRouteComposition | None
        ) = None,
    ) -> tuple["FormalManifoldBranchRuntime", int]:
        """Restore and validate one branch-local checkpoint.

        A nonempty warm state requires ``backend`` so whitening and frame
        provenance can be rebuilt exactly.  The returned integer is the exact
        evaluation charge incurred by that validation.
        """

        data = dict(payload)
        if data.get("schema") != (
            "formal_manifold_beam_branch_runtime_checkpoint_v1"
        ):
            raise ValueError("unsupported formal-manifold branch checkpoint schema.")
        if bool(data.get("structural_rollback_supported", False)):
            raise ValueError("structural rollback is not supported by Formal-Manifold.")
        if str(data.get("rollback_scope", "pending_proposal_only")) != (
            "pending_proposal_only"
        ):
            raise ValueError("formal-manifold branch checkpoint rollback scope drifted.")
        transaction_raw = data.get("transaction_state")
        if not isinstance(transaction_raw, Mapping):
            raise ValueError("branch checkpoint lacks transaction state.")
        transaction = dict(transaction_raw)
        composition = FormalManifoldRouteComposition.from_mapping(
            data.get("route_composition", transaction.get("route_composition"))
        )
        if expected_route_composition is not None:
            expected = FormalManifoldRouteComposition.from_mapping(
                expected_route_composition
            )
            if composition.as_dict() != expected.as_dict():
                raise ValueError(
                    "branch checkpoint formal-manifold route composition drifted."
                )
        resolved_config = (
            FormalManifoldConfig.from_mapping(
                transaction.get("formal_manifold_config")
            )
            if config is None
            else config
        )
        if not isinstance(resolved_config, FormalManifoldConfig):
            raise TypeError("config must be a FormalManifoldConfig.")
        config_sha = _json_hash(resolved_config.as_dict())
        if str(data.get("formal_manifold_config_sha256")) != config_sha:
            raise ValueError("branch checkpoint formal-manifold config drifted.")
        branch_id = str(data.get("branch_id", ""))
        if not branch_id:
            raise ValueError("branch checkpoint lacks branch_id.")
        if str(transaction.get("branch_id")) != branch_id:
            raise ValueError("branch checkpoint transaction branch identity drifted.")
        parent_branch_raw = data.get(
            "parent_branch_id", transaction.get("parent_branch_id")
        )
        session = FormalManifoldSession(
            config=resolved_config,
            branch_id=branch_id,
            parent_branch_id=(
                None if parent_branch_raw is None else str(parent_branch_raw)
            ),
            route_composition=composition,
        )
        warm_state = data.get("warm_state")
        validation_nfev = 0
        if warm_state is None:
            session.restore_transaction_payload(transaction)
        else:
            if not isinstance(warm_state, Mapping):
                raise ValueError("branch checkpoint warm state must be a mapping.")
            if backend is None:
                raise ValueError(
                    "backend is required to restore a nonempty FM warm state."
                )
            validation_nfev = session.restore_checkpoint_payload(
                warm_state, backend
            )
            restored_transaction = session.transaction_payload()
            for field_name in (
                "branch_id",
                "parent_branch_id",
                "last_reset_reason",
                "reset_count",
                "commit_count",
                "rollback_count",
                "rollback_scope",
            ):
                if restored_transaction.get(field_name) != transaction.get(
                    field_name
                ):
                    raise ValueError(
                        "branch checkpoint warm/transaction state disagrees for "
                        f"{field_name}."
                    )
        query_ledger_raw = data.get("query_ledger")
        if not isinstance(query_ledger_raw, Mapping):
            raise ValueError("branch checkpoint lacks a query-ledger checkpoint.")
        runtime = cls(
            branch_id=branch_id,
            query_origin_branch_id=str(
                data.get("query_origin_branch_id", branch_id)
            ),
            session=session,
            query_ledger=QueryPrimitiveLedger.from_checkpoint_payload(
                query_ledger_raw
            ),
        )
        return runtime, int(validation_nfev)

    def behavioral_fingerprint_payload(self) -> dict[str, Any]:
        """Return all branch-local state that can change later FM behavior."""

        warm = self.session.checkpoint_payload()
        base = {
            "schema": "formal_manifold_beam_behavior_v1",
            "active": bool(warm is not None),
            "query_origin_branch_id": str(self.query_origin_branch_id),
            "route_composition_sha256": self.session.route_composition.sha256,
            "formal_manifold_config_sha256": _json_hash(
                self.session.config.as_dict()
            ),
            "query_ledger_sha256": _json_hash(
                self.query_ledger.checkpoint_payload()
            ),
        }
        if warm is None:
            return base
        base.update(
            {
                "registry_sha256": warm.get("registry_sha256"),
                "whitening_id": warm.get("whitening_id"),
                "frame_id": warm.get("frame_id"),
                "logical_range_id": warm.get("logical_range_id"),
                "curvature_branch": warm.get("curvature_branch"),
                "inverse_curvature": warm.get("inverse_curvature"),
                "direct_curvature": warm.get("direct_curvature"),
                "qbroyd_inverse_metric": warm.get("qbroyd_inverse_metric"),
                "qbroyd_age": warm.get("qbroyd_age"),
                "trust_radius": warm.get("trust_radius"),
            }
        )
        return base

    def summary(self) -> dict[str, Any]:
        return {
            "schema": "formal_manifold_beam_branch_runtime_summary_v1",
            "branch_id": str(self.branch_id),
            "query_origin_branch_id": str(self.query_origin_branch_id),
            "session": self.session.summary(),
            "behavioral_fingerprint": self.behavioral_fingerprint_payload(),
            "checkpoint_sha256": _json_hash(self.checkpoint_payload()),
        }


__all__ = [
    "DirectSR1Update",
    "ExactFrame",
    "ExactStateBackend",
    "ExactStateEvaluation",
    "FORMAL_MANIFOLD_ROUTE",
    "FORMAL_MANIFOLD_ROUTE_CHOICES",
    "FORMAL_MANIFOLD_ROUTE_COMPOSITION_SCHEMA",
    "FORMAL_MANIFOLD_ROUTE_FAMILY",
    "FORMAL_MANIFOLD_SR_NO_N2_PROFILE",
    "FORMAL_MANIFOLD_SR_SOURCE_LOCKED_PROFILE",
    "FORMAL_MANIFOLD_SR_SELECTOR_FAMILY",
    "FORMAL_MANIFOLD_SR_SELECTOR_PROFILE",
    "FORMAL_MANIFOLD_WARM_START_OFF",
    "FORMAL_MANIFOLD_WARM_START_ROUTE",
    "FORMAL_CURVATURE_BRANCHES",
    "FORMAL_CURVATURE_DIRECT_SR1",
    "FORMAL_CURVATURE_INVERSE_RBFGS",
    "FormalManifoldConfig",
    "FormalManifoldBranchRuntime",
    "FormalManifoldResult",
    "FormalManifoldRouteComposition",
    "FormalManifoldSession",
    "FormalManifoldWarmState",
    "GrowthMap",
    "ProcrustesResult",
    "RBFGSUpdate",
    "RankRule",
    "WhitenedTransport",
    "build_exact_frame",
    "density_tangent_cross_gram",
    "endpoint_procrustes",
    "grow_zero_coordinates",
    "guarded_direct_sr1",
    "normalize_formal_manifold_route_composition",
    "normalize_reoptimization_route",
    "powell_damped_inverse_rbfgs",
    "qbroyd_inverse_update",
    "solve_direct_sr1_trust_step",
    "supported_whitened_transport",
]
