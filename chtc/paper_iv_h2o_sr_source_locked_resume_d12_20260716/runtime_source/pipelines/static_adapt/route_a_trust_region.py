"""Branch-local adaptive trust-radius state for canonical Route A."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.scaffold.hh_continuation_scoring import (
    BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1,
)
from pipelines.static_adapt.schur_warm_start import scalar_schur_trust_step
from pipelines.static_adapt.route_a_schur_selector import (
    ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_V1,
    ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2,
    ROUTE_A_TRUST_REGION_FIXED,
    RouteASchurSelectorConfig,
    TrustRegionUpdateConfig,
)
from pipelines.static_adapt.joint_linear_solve import (
    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
)


TRUST_REGION_STATE_SCHEMA = "route_a_trust_region_state_v1"
TRUST_REGION_UPDATE_SCHEMA = "route_a_trust_region_update_v1"
ROUND_TRUST_REGION_SNAPSHOT_SCHEMA = "route_a_round_trust_region_snapshot_v1"
ROUND_TRUST_REGION_STAGE_RECEIPT_SCHEMA = (
    "route_a_round_trust_region_stage_receipt_v1"
)
ROUND_TRUST_REGION_STAGE_NAMES = (
    "macro_phase1",
    "macro_phase2",
    "child_phase1",
    "child_phase2",
    "final_selector",
)
HISTORICAL_SINGLETON_SCALAR_TRUST_CONTEXT_V1 = (
    "historical_singleton_scalar_trust_context_v1"
)
HISTORICAL_SINGLETON_GEOMETRY_EXPANSION_CONTEXT_V1 = (
    "historical_singleton_geometry_expansion_v1"
)
SR_ACTIVE_ONLY_CORRECTION_CONTEXT_V1 = (
    "sr_active_only_shared_support_correction_v1"
)
SR_ACTIVE_STATIONARITY_BACKTRACKING_CONTEXT_V1 = (
    "sr_active_stationarity_backtracking_refinement_v1"
)
HISTORICAL_SINGLETON_SCALAR_SELECTOR_SUMMARY_SCHEMA = (
    "historical_singleton_scalar_selector_summary_v1"
)


def _finite_float(value: Any) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if math.isfinite(resolved) else None


def _normalized_state(state: np.ndarray) -> np.ndarray:
    vector = np.asarray(state, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("State must have finite nonzero norm.")
    return vector / norm


def exact_fubini_study_distance(
    state_before: np.ndarray,
    state_after: np.ndarray,
) -> float:
    """Return arccos(|<before|after>|), invariant to global phase."""

    before = _normalized_state(state_before)
    after = _normalized_state(state_after)
    if before.shape != after.shape:
        raise ValueError("Fubini-Study states must have matching dimensions.")
    overlap = float(np.clip(abs(np.vdot(before, after)), 0.0, 1.0))
    if 1.0 - overlap <= 1e-14:
        return 0.0
    return float(math.acos(overlap))


def state_fingerprint(state: np.ndarray) -> str:
    """Hash a normalized state after removing its arbitrary global phase."""

    vector = _normalized_state(state).copy()
    nonzero = np.flatnonzero(np.abs(vector) > 1e-14)
    if nonzero.size:
        anchor = vector[int(nonzero[0])]
        vector *= np.exp(-1j * np.angle(anchor))
    packed = np.column_stack((vector.real, vector.imag)).astype("<f8", copy=False)
    return hashlib.sha256(packed.tobytes(order="C")).hexdigest()


@dataclass
class RouteATrustRegionState:
    radius: float
    reference_radius: float | None = None
    update_count: int = 0
    last_update: dict[str, Any] | None = None
    initialization_reason: str = "configured_initial_radius"

    def __post_init__(self) -> None:
        radius = float(self.radius)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("Trust-region radius must be finite and positive.")
        self.radius = radius
        reference_radius = (
            radius
            if self.reference_radius is None
            else float(self.reference_radius)
        )
        if not math.isfinite(reference_radius) or reference_radius <= 0.0:
            raise ValueError(
                "Trust-region reference radius must be finite and positive."
            )
        self.reference_radius = reference_radius
        self.update_count = int(max(0, int(self.update_count)))
        self.last_update = (
            None if self.last_update is None else copy.deepcopy(dict(self.last_update))
        )
        self.initialization_reason = str(self.initialization_reason)

    def clone(self) -> "RouteATrustRegionState":
        return RouteATrustRegionState.from_dict(
            self.as_dict(),
            initial_radius=float(self.reference_radius),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": TRUST_REGION_STATE_SCHEMA,
            "radius": float(self.radius),
            "reference_radius": float(self.reference_radius),
            "update_count": int(self.update_count),
            "last_update": copy.deepcopy(self.last_update),
            "initialization_reason": str(self.initialization_reason),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | None,
        *,
        initial_radius: float,
    ) -> "RouteATrustRegionState":
        if not isinstance(payload, Mapping):
            return cls(
                radius=float(initial_radius),
                initialization_reason="legacy_checkpoint_missing_state",
            )
        if "radius" not in payload:
            return cls(
                radius=float(initial_radius),
                initialization_reason="legacy_checkpoint_missing_state",
            )
        radius = _finite_float(payload.get("radius"))
        if radius is None or radius <= 0.0:
            return cls(
                radius=float(initial_radius),
                initialization_reason="invalid_checkpoint_state_reset",
            )
        reference_radius = _finite_float(
            payload.get("reference_radius", initial_radius)
        )
        if reference_radius is None or reference_radius <= 0.0:
            reference_radius = float(initial_radius)
        return cls(
            radius=float(radius),
            reference_radius=float(reference_radius),
            update_count=int(payload.get("update_count", 0)),
            last_update=(
                dict(payload["last_update"])
                if isinstance(payload.get("last_update"), Mapping)
                else None
            ),
            initialization_reason=str(
                payload.get("initialization_reason", "restored_checkpoint_state")
            ),
        )


@dataclass(frozen=True)
class RouteARoundTrustRegionSnapshot:
    """Immutable branch-local trust state resolved once for one ADAPT round."""

    radius: float
    update_count: int
    source: str

    def __post_init__(self) -> None:
        radius = float(self.radius)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("Round trust-region radius must be finite and positive.")
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "update_count", int(max(0, int(self.update_count))))
        object.__setattr__(self, "source", str(self.source))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ROUND_TRUST_REGION_SNAPSHOT_SCHEMA,
            "radius": float(self.radius),
            "update_count": int(self.update_count),
            "source": str(self.source),
        }


def resolve_round_trust_region_snapshot(
    state: RouteATrustRegionState | None,
    *,
    fallback_radius: float,
) -> RouteARoundTrustRegionSnapshot:
    """Freeze the radius that every metric-bearing stage reads this round."""

    if state is None:
        return RouteARoundTrustRegionSnapshot(
            radius=float(fallback_radius),
            update_count=0,
            source="configured_fallback",
        )
    return RouteARoundTrustRegionSnapshot(
        radius=float(state.radius),
        update_count=int(state.update_count),
        source="branch_local_state",
    )


def score_config_with_round_trust_radius(
    config: Any,
    snapshot: RouteARoundTrustRegionSnapshot,
) -> Any:
    """Project one frozen round radius into a score dataclass without mutation."""

    if not hasattr(config, "rho"):
        raise TypeError("Round trust-radius score config must expose rho.")
    return replace(config, rho=float(snapshot.radius))


def selector_config_with_round_trust_radius(
    config: RouteASchurSelectorConfig,
    snapshot: RouteARoundTrustRegionSnapshot,
) -> RouteASchurSelectorConfig:
    """Project one frozen round radius into a stateless joint selector call."""

    return replace(
        config,
        max_fubini_study_step=float(snapshot.radius),
    )


def round_trust_region_stage_receipt(
    snapshot: RouteARoundTrustRegionSnapshot,
    *,
    stage_radii: Mapping[str, float],
) -> dict[str, Any]:
    """Validate and serialize the one-radius-per-round stage contract."""

    missing = [
        stage
        for stage in ROUND_TRUST_REGION_STAGE_NAMES
        if stage not in stage_radii
    ]
    if missing:
        raise ValueError(
            "Round trust-radius receipt is missing stages: " + ", ".join(missing)
        )
    resolved: dict[str, float] = {}
    for stage in ROUND_TRUST_REGION_STAGE_NAMES:
        value = float(stage_radii[stage])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Round trust radius for {stage} must be positive.")
        if value != float(snapshot.radius):
            raise ValueError(
                f"Round trust-radius drift at {stage}: "
                f"{value} != {snapshot.radius}."
            )
        resolved[stage] = value
    return {
        "schema": ROUND_TRUST_REGION_STAGE_RECEIPT_SCHEMA,
        "radius": float(snapshot.radius),
        "update_count_at_round_start": int(snapshot.update_count),
        "source": str(snapshot.source),
        "stage_radii": resolved,
    }


def initialize_trust_region_state(
    *,
    initial_radius: float,
    checkpoint_payload: Mapping[str, Any] | None = None,
) -> RouteATrustRegionState:
    if checkpoint_payload is None:
        return RouteATrustRegionState(radius=float(initial_radius))
    restored = RouteATrustRegionState.from_dict(
        checkpoint_payload,
        initial_radius=float(initial_radius),
    )
    if restored.initialization_reason not in {
        "legacy_checkpoint_missing_state",
        "invalid_checkpoint_state_reset",
    }:
        restored.initialization_reason = "restored_checkpoint_state"
    return restored


def selector_config_with_trust_radius(
    config: RouteASchurSelectorConfig,
    state: RouteATrustRegionState | None,
) -> RouteASchurSelectorConfig:
    """Project the branch-local runtime radius into one stateless selector call."""

    snapshot = resolve_round_trust_region_snapshot(
        state,
        fallback_radius=float(config.max_fubini_study_step),
    )
    return selector_config_with_round_trust_radius(config, snapshot)


def _selected_record_identities(
    records: Sequence[Mapping[str, Any]],
) -> list[str]:
    identities: list[str] = []
    for record in records:
        value = (
            record.get("route_a_global_pauli_identity")
            or record.get("route_a_child_identity")
            or record.get("generator_id")
            or record.get("candidate_label")
            or record.get("operator_label")
            or ""
        )
        identities.append(str(value))
    return identities


def _feature_row_value(feature_row: Any, name: str) -> Any:
    if isinstance(feature_row, Mapping):
        nested = feature_row.get("feature")
        if nested is not None:
            return _feature_row_value(nested, name)
        return feature_row.get(name)
    return getattr(feature_row, name, None)


def _required_finite_feature_value(feature_row: Any, name: str) -> float:
    value = _finite_float(_feature_row_value(feature_row, name))
    if value is None:
        raise ValueError(
            "Historical singleton scalar trust context requires finite "
            f"feature field {name!r}."
        )
    return float(value)


def _crosschecked_component_value(
    components: Mapping[str, Any],
    names: Sequence[str],
    *,
    description: str,
) -> tuple[float, str]:
    resolved: list[tuple[str, float]] = []
    for name in names:
        if name not in components:
            continue
        value = _finite_float(components.get(name))
        if value is None:
            raise ValueError(
                f"Historical singleton {description} component {name!r} "
                "must be finite."
            )
        resolved.append((str(name), float(value)))
    if not resolved:
        raise ValueError(
            f"Historical singleton scalar trust context is missing {description}."
        )
    authority_name, authority_value = resolved[0]
    for name, value in resolved[1:]:
        if not math.isclose(
            value,
            authority_value,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"Historical singleton {description} components disagree: "
                f"{authority_name}={authority_value!r}, {name}={value!r}."
            )
    return float(authority_value), str(authority_name)


def historical_singleton_scalar_selector_summary(
    feature_row: Any,
    *,
    radius: float,
    metric_floor: float,
    reduced_metric_collapse_rel_tol: float,
) -> dict[str, Any]:
    """Build updater receipt keys from the saved historical scalar model.

    Whitened Phase-III overlays replace the live ``DeltaE_TR`` field.  When
    present, the explicitly preserved historical scalar gain is therefore the
    crosscheck authority; unmodified historical rows use their original
    ``DeltaE_TR`` aliases.  Missing or inconsistent geometry fails closed.
    """

    radius_value = _finite_float(radius)
    floor_value = _finite_float(metric_floor)
    collapse_value = _finite_float(reduced_metric_collapse_rel_tol)
    if radius_value is None or radius_value <= 0.0:
        raise ValueError(
            "Historical singleton scalar trust radius must be finite and positive."
        )
    if floor_value is None or floor_value < 0.0:
        raise ValueError(
            "Historical singleton scalar metric floor must be finite and "
            "nonnegative."
        )
    if collapse_value is None or collapse_value < 0.0:
        raise ValueError(
            "Historical singleton scalar collapse tolerance must be finite and "
            "nonnegative."
        )
    components_raw = _feature_row_value(feature_row, "phase_score_components")
    if not isinstance(components_raw, Mapping):
        raise ValueError(
            "Historical singleton scalar trust context requires saved Phase-III "
            "score components."
        )
    components = dict(components_raw)
    preserved_names = (
        "phase3_historical_scalar_DeltaE_TR",
        "historical_scalar_DeltaE_TR",
    )
    if any(name in components for name in preserved_names):
        stored_gain, stored_gain_source = _crosschecked_component_value(
            components,
            preserved_names,
            description="preserved scalar gain",
        )
    else:
        stored_gain, stored_gain_source = _crosschecked_component_value(
            components,
            ("DeltaE_TR", "phase3_delta_e_tr"),
            description="scalar gain",
        )
        feature_gain = _finite_float(
            _feature_row_value(feature_row, "phase3_reduced_trust_gain")
        )
        if feature_gain is not None and not math.isclose(
            feature_gain,
            stored_gain,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Historical singleton scalar feature gain disagrees with its "
                "saved Phase-III component."
            )

    g_component = _finite_float(components.get("phase3_g_hw_lcb"))
    g_lcb = (
        float(g_component)
        if g_component is not None
        else _required_finite_feature_value(feature_row, "g_lcb")
    )
    h_eff = _required_finite_feature_value(feature_row, "h_eff")
    f_red = _required_finite_feature_value(feature_row, "F_red")
    f_raw = _required_finite_feature_value(feature_row, "F_raw")
    step = scalar_schur_trust_step(
        g_lcb=float(g_lcb),
        h_eff=float(h_eff),
        F_red=float(f_red),
        F_raw=float(f_raw),
        rho=float(radius_value),
        metric_floor=float(floor_value),
        reduced_metric_collapse_rel_tol=float(collapse_value),
    )
    if str(step.status) != "available":
        raise ValueError(
            "Historical singleton scalar trust step is unavailable: "
            f"{step.reason}."
        )
    if not math.isclose(
        float(step.predicted_gain),
        float(stored_gain),
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Historical singleton scalar trust-step gain does not reproduce "
            "the saved scalar Phase-III gain."
        )
    f_safe = float(max(f_red, floor_value))
    displacement = float(step.alpha_abs * math.sqrt(f_safe))
    displacement_sq = float(displacement * displacement)
    schur_indices_raw = _feature_row_value(feature_row, "schur_window_indices")
    if schur_indices_raw is None:
        schur_indices: list[int] = []
    elif isinstance(schur_indices_raw, Sequence) and not isinstance(
        schur_indices_raw,
        (str, bytes),
    ):
        schur_indices = [int(value) for value in schur_indices_raw]
    else:
        raise ValueError(
            "Historical singleton schur_window_indices must be a sequence."
        )
    return {
        "schema": HISTORICAL_SINGLETON_SCALAR_SELECTOR_SUMMARY_SCHEMA,
        "context_mode": HISTORICAL_SINGLETON_SCALAR_TRUST_CONTEXT_V1,
        "authority": "saved_historical_scalar_schur_model_v1",
        "joint_fubini_study_displacement_sq": float(displacement_sq),
        "predicted_fubini_study_displacement": float(displacement),
        "applied_predicted_reduction": float(step.predicted_gain),
        "joint_gain": float(step.predicted_gain),
        "trust_regularization_applied": bool(step.at_trust_boundary),
        "trust_clipped": bool(step.at_trust_boundary),
        "trust_radius_binding": bool(step.at_trust_boundary),
        "active_context_indices_effective": schur_indices,
        "historical_scalar_gain_crosscheck": float(stored_gain),
        "historical_scalar_gain_source": str(stored_gain_source),
        "scalar_coordinate_step_abs": float(step.alpha_abs),
        "scalar_step_status": str(step.status),
        "scalar_step_reason": str(step.reason),
        "scalar_step_telemetry": dict(step.telemetry),
        "radius": float(radius_value),
        "g_lcb": float(g_lcb),
        "h_eff": float(h_eff),
        "F_red": float(f_red),
        "F_safe": float(f_safe),
        "F_raw": float(f_raw),
        "metric_floor": float(floor_value),
        "reduced_metric_collapse_rel_tol": float(collapse_value),
    }


def _sr_v2_stabilized_trust_transaction(
    summary: Mapping[str, Any],
    *,
    realized_joint_step: Sequence[float] | None,
    radius_before: float,
) -> dict[str, Any] | None:
    """Resolve predicted/realized motion in the exact v2 trust metric."""

    policy = str(summary.get("joint_linear_solve_policy_effective", ""))
    if policy != JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2:
        return None
    if realized_joint_step is None:
        raise ValueError(
            "SR v2 adaptive trust requires the accepted realized joint "
            "coordinate step."
        )
    certified_radius_sq = _finite_float(summary.get("trust_radius_sq"))
    certified_radius_tolerance_sq = _finite_float(
        summary.get("trust_radius_binding_tolerance_sq")
    )
    live_radius_sq = float(radius_before) ** 2
    if certified_radius_sq is None or certified_radius_sq <= 0.0:
        raise ValueError(
            "SR v2 trust transaction is missing its certified trust radius."
        )
    radius_match_tolerance_sq = float(
        max(
            1e-14,
            certified_radius_tolerance_sq or 0.0,
            4096.0
            * np.finfo(float).eps
            * max(1.0, certified_radius_sq, live_radius_sq),
        )
    )
    radius_sq_residual = float(abs(certified_radius_sq - live_radius_sq))
    if radius_sq_residual > radius_match_tolerance_sq:
        raise ValueError(
            "SR v2 certified trust radius does not match the branch-local "
            "radius used for the accepted path."
        )
    G_AA = np.asarray(summary.get("G_AA_raw", []), dtype=float)
    G_AB = np.asarray(summary.get("G_AB_raw", []), dtype=float)
    G_BB = np.asarray(summary.get("G_BB_raw", []), dtype=float)
    if G_AA.ndim != 2 or G_AA.shape[0] != G_AA.shape[1]:
        raise ValueError("SR v2 trust transaction is missing square G_AA_raw.")
    active_count = int(G_AA.shape[0])
    if G_BB.ndim != 2 or G_BB.shape[0] != G_BB.shape[1]:
        raise ValueError("SR v2 trust transaction is missing square G_BB_raw.")
    batch_count = int(G_BB.shape[0])
    if G_AB.shape != (active_count, batch_count):
        raise ValueError("SR v2 trust transaction has inconsistent G_AB_raw.")
    raw_metric = np.block([[G_AA, G_AB], [G_AB.T, G_BB]])
    raw_metric = 0.5 * (raw_metric + raw_metric.T)
    dimension = int(raw_metric.shape[0])
    if dimension <= 0 or not np.all(np.isfinite(raw_metric)):
        raise ValueError("SR v2 trust transaction has invalid raw metric.")

    provenance_id = str(
        summary.get("supported_metric_whitening_provenance_id", "")
    )
    if not provenance_id:
        raise ValueError(
            "SR v2 trust transaction is missing whitening provenance."
        )
    retained_mask = np.asarray(
        summary.get("metric_retained_mask", []), dtype=bool
    ).reshape(-1)
    raw_eigenvalues_reference = np.asarray(
        summary.get("raw_metric_eigenvalues", []), dtype=float
    ).reshape(-1)
    ridge = _finite_float(summary.get("metric_whitening_ridge"))
    stabilization_lambda = _finite_float(
        summary.get("metric_stabilization_lambda_G")
    )
    if (
        retained_mask.size != dimension
        or raw_eigenvalues_reference.size != dimension
        or ridge is None
        or ridge < 0.0
    ):
        raise ValueError(
            "SR v2 trust transaction has incomplete stabilized-metric "
            "telemetry."
        )
    raw_eigenvalues, raw_eigenvectors = np.linalg.eigh(raw_metric)
    eigen_scale = float(
        max(
            1.0,
            np.max(np.abs(raw_eigenvalues))
            if raw_eigenvalues.size
            else 0.0,
        )
    )
    reconstruction_tolerance = float(
        max(
            _finite_float(summary.get("raw_metric_support_epsilon_G"))
            or 0.0,
            4096.0
            * np.finfo(float).eps
            * max(1, dimension)
            * eigen_scale,
        )
    )
    eigenvalue_residual = float(
        np.max(np.abs(raw_eigenvalues - raw_eigenvalues_reference))
    )
    if eigenvalue_residual > reconstruction_tolerance:
        raise ValueError(
            "SR v2 stabilized trust metric does not reproduce the certified "
            "raw spectrum."
        )
    retained_values = np.asarray(raw_eigenvalues[retained_mask], dtype=float)
    retained_vectors = np.asarray(
        raw_eigenvectors[:, retained_mask], dtype=float
    )
    denominators_telemetry = np.asarray(
        summary.get("whitening_denominators", []), dtype=float
    ).reshape(-1)
    if denominators_telemetry.size:
        if (
            denominators_telemetry.size != retained_values.size
            or not np.all(np.isfinite(denominators_telemetry))
        ):
            raise ValueError(
                "SR v2 trust transaction has invalid certified whitening "
                "denominators."
            )
        denominators = denominators_telemetry
        denominator_authority = "solver_whitening_denominators_v1"
    else:
        # Compatibility for old checkpoint/test summaries.  Fresh v2 solves
        # always emit the derived denominators explicitly.
        denominators = retained_values + float(ridge)
        denominator_authority = "legacy_reconstructed_eigenvalue_plus_ridge"
    declared_lambda = float(
        ridge if stabilization_lambda is None else stabilization_lambda
    )
    denominator_crosscheck = retained_values + declared_lambda
    denominator_tolerance = float(
        max(
            reconstruction_tolerance,
            4096.0
            * np.finfo(float).eps
            * max(1, dimension)
            * max(
                1.0,
                float(np.max(np.abs(denominators)))
                if denominators.size
                else 0.0,
            ),
        )
    )
    denominator_residual = float(
        np.max(np.abs(denominators - denominator_crosscheck))
    )
    if denominator_residual > denominator_tolerance:
        raise ValueError(
            "SR v2 certified whitening denominators disagree with the "
            "derived stabilization telemetry."
        )
    if not denominators.size or np.any(denominators <= 0.0):
        raise ValueError("SR v2 stabilized trust metric has invalid support.")
    stabilized_metric = np.asarray(
        retained_vectors @ np.diag(denominators) @ retained_vectors.T,
        dtype=float,
    )
    predicted_step = np.asarray(summary.get("joint_step", []), dtype=float).reshape(-1)
    realized_step = np.asarray(realized_joint_step, dtype=float).reshape(-1)
    if (
        predicted_step.size != dimension
        or realized_step.size != dimension
        or not np.all(np.isfinite(predicted_step))
        or not np.all(np.isfinite(realized_step))
    ):
        raise ValueError(
            "SR v2 trust transaction coordinate steps do not match the "
            "certified metric dimension."
        )
    predicted_sq = float(
        max(0.0, predicted_step.T @ stabilized_metric @ predicted_step)
    )
    realized_sq = float(
        max(0.0, realized_step.T @ stabilized_metric @ realized_step)
    )
    predicted = float(math.sqrt(predicted_sq))
    realized = float(math.sqrt(realized_sq))
    whitened_norm = _finite_float(summary.get("whitened_step_norm"))
    predicted_crosscheck_tolerance = float(
        4096.0
        * np.finfo(float).eps
        * max(1, dimension)
        * max(1.0, predicted, whitened_norm or 0.0)
    )
    if (
        whitened_norm is not None
        and abs(float(whitened_norm) - predicted)
        > predicted_crosscheck_tolerance
    ):
        raise ValueError(
            "SR v2 stabilized predicted displacement disagrees with the "
            "certified whitened-step norm."
        )
    raw_predicted_sq = float(max(0.0, predicted_step.T @ raw_metric @ predicted_step))
    raw_realized_sq = float(max(0.0, realized_step.T @ raw_metric @ realized_step))
    return {
        "schema": "sr_v2_stabilized_trust_accepted_path_transaction_v1",
        "joint_linear_solve_policy": str(policy),
        "supported_metric_whitening_provenance_id": str(provenance_id),
        "certified_trust_radius": float(math.sqrt(certified_radius_sq)),
        "certified_trust_radius_sq": float(certified_radius_sq),
        "branch_trust_radius_before": float(radius_before),
        "trust_radius_sq_match_residual": float(radius_sq_residual),
        "trust_radius_sq_match_tolerance": float(radius_match_tolerance_sq),
        "trust_radius_provenance": (
            "coordinate_summary_trust_radius_sq_matched_to_branch_state_v1"
        ),
        "metric_whitening_ridge": float(ridge),
        "metric_stabilization_lambda_G": float(declared_lambda),
        "whitening_denominators": [
            float(value) for value in denominators.tolist()
        ],
        "whitening_denominator_authority": str(denominator_authority),
        "whitening_denominator_crosscheck_residual": float(
            denominator_residual
        ),
        "whitening_denominator_crosscheck_tolerance": float(
            denominator_tolerance
        ),
        "metric_retained_mask": [
            bool(value) for value in retained_mask.tolist()
        ],
        "raw_metric_eigenvalue_reconstruction_residual": float(
            eigenvalue_residual
        ),
        "raw_metric_eigenvalue_reconstruction_tolerance": float(
            reconstruction_tolerance
        ),
        "predicted_stabilized_trust_displacement": float(predicted),
        "predicted_stabilized_trust_displacement_sq": float(predicted_sq),
        "realized_stabilized_trust_displacement": float(realized),
        "realized_stabilized_trust_displacement_sq": float(realized_sq),
        "predicted_whitened_step_norm_crosscheck": whitened_norm,
        "predicted_whitened_step_norm_crosscheck_tolerance": float(
            predicted_crosscheck_tolerance
        ),
        "predicted_raw_metric_local_displacement": float(
            math.sqrt(raw_predicted_sq)
        ),
        "realized_raw_metric_local_displacement": float(
            math.sqrt(raw_realized_sq)
        ),
        "predicted_joint_step": [
            float(value) for value in predicted_step.tolist()
        ],
        "accepted_realized_joint_step": [
            float(value) for value in realized_step.tolist()
        ],
        "adaptive_radius_rescale_authority": (
            "regularized_supported_metric_whitened_coordinates_v1"
        ),
        "exact_endpoint_fubini_study_distance_role": "diagnostic_only",
        "raw_gram_local_displacement_role": "diagnostic_only",
        "stabilized_trust_transaction_complete": True,
    }


def hold_trust_region_state_after_failed_transaction(
    state: RouteATrustRegionState,
    *,
    config: TrustRegionUpdateConfig,
    context_mode: str,
    reason: str,
    failure_kind: str,
    audit_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Record a failed/no-state trust service without changing its radius."""

    radius = float(state.radius)
    payload = {
        "schema": TRUST_REGION_UPDATE_SCHEMA,
        "policy": str(config.policy),
        "context_mode": str(context_mode),
        "radius_before": float(radius),
        "radius_after": float(radius),
        "update_factor": 1.0,
        "update_reason": str(reason),
        "trust_transaction_status": "held",
        "trust_transaction_failure_kind": str(failure_kind),
        "numerical_or_mapping_failure": bool(
            str(failure_kind)
            in {
                "mapping",
                "mapping_contract",
                "nonfinite_path",
                "stabilized_path",
                "state_consistency",
            }
        ),
        "no_state_transition": True,
        "scientific_radius_lower_bound": (
            0.0
            if str(config.policy)
            == ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2
            else float(config.radius_min)
        ),
        "scientific_radius_upper_bound": None,
        "rate_limit_applied": False,
        "numerical_floor_applied": False,
        "radius_hold_authority": (
            "section8_num_map_or_no_state_priority_hold_v1"
        ),
        "audit_payload": copy.deepcopy(dict(audit_payload or {})),
    }
    state.update_count = int(state.update_count) + 1
    state.last_update = copy.deepcopy(payload)
    return payload


def update_trust_region_state(
    state: RouteATrustRegionState,
    *,
    config: TrustRegionUpdateConfig,
    context_mode: str,
    selector_summary: Mapping[str, Any] | None,
    state_before: np.ndarray,
    state_after_refit: np.ndarray,
    energy_before: float,
    energy_after_refit: float,
    energy_improvement_tolerance: float,
    full_coordinate_refit: bool,
    selected_records: Sequence[Mapping[str, Any]] = (),
    selected_effective_positions: Sequence[int] = (),
    allow_historical_singleton_context: bool = False,
    realized_joint_step: Sequence[float] | None = None,
    seed_guard_payload: Mapping[str, Any] | None = None,
    optimizer_outcome_payload: Mapping[str, Any] | None = None,
    accepted_path_prediction: Mapping[str, Any] | None = None,
    transaction_failure_reason: str | None = None,
) -> dict[str, Any]:
    """Update one branch after full refit and before any prune mutation."""

    summary = dict(selector_summary or {})
    radius_before = float(state.radius)
    predicted_sq_raw_metric = _finite_float(
        summary.get("joint_fubini_study_displacement_sq")
    )
    predicted_displacement = (
        None
        if predicted_sq_raw_metric is None or predicted_sq_raw_metric < 0.0
        else float(math.sqrt(max(0.0, predicted_sq_raw_metric)))
    )
    predicted_energy_reduction = _finite_float(
        summary.get("applied_predicted_reduction")
    )
    selector_regularization_applied = bool(
        summary.get(
            "trust_regularization_applied",
            summary.get("trust_clipped", False),
        )
    )
    radius_binding_value = summary.get("trust_radius_binding")
    if radius_binding_value is None:
        radius_sq = float(radius_before) ** 2
        binding_tolerance_sq = float(max(1e-14, radius_sq * 1e-8))
        trust_radius_binding = bool(
            selector_regularization_applied
            and predicted_sq_raw_metric is not None
            and abs(float(predicted_sq_raw_metric) - radius_sq)
            <= binding_tolerance_sq
        )
    else:
        trust_radius_binding = bool(radius_binding_value)
    realized_energy_reduction = float(energy_before) - float(energy_after_refit)
    realized_fs_displacement_exact: float | None
    try:
        realized_fs_displacement_exact = exact_fubini_study_distance(
            state_before,
            state_after_refit,
        )
    except (TypeError, ValueError, FloatingPointError):
        realized_fs_displacement_exact = None
    try:
        pre_state_fingerprint = state_fingerprint(state_before)
        post_state_fingerprint = state_fingerprint(state_after_refit)
    except (TypeError, ValueError, FloatingPointError):
        pre_state_fingerprint = None
        post_state_fingerprint = None
        if transaction_failure_reason in {None, ""}:
            transaction_failure_reason = "nonfinite_or_invalid_state_path"
    stabilized_trust_transaction: dict[str, Any] | None = None
    stabilized_trust_transaction_failure: str | None = None
    try:
        stabilized_trust_transaction = _sr_v2_stabilized_trust_transaction(
            summary,
            realized_joint_step=realized_joint_step,
            radius_before=float(radius_before),
        )
    except (TypeError, ValueError, FloatingPointError) as exc:
        stabilized_trust_transaction_failure = (
            f"{exc.__class__.__name__}:{str(exc)}"
        )
    if stabilized_trust_transaction is None:
        realized_displacement = realized_fs_displacement_exact
        displacement_authority = "endpoint_fubini_study_distance_v1"
    else:
        predicted_displacement = float(
            stabilized_trust_transaction[
                "predicted_stabilized_trust_displacement"
            ]
        )
        realized_displacement = float(
            stabilized_trust_transaction[
                "realized_stabilized_trust_displacement"
            ]
        )
        displacement_authority = str(
            stabilized_trust_transaction[
                "adaptive_radius_rescale_authority"
            ]
        )

    ratio: float | None = None
    update_factor = 1.0
    radius_after = float(radius_before)
    update_reason = "fixed_policy"
    direction_guard_available = False
    direction_guard_passed = False
    policy = str(config.policy)
    eps = float(config.displacement_epsilon)
    if (
        stabilized_trust_transaction is not None
        and predicted_displacement is not None
        and predicted_displacement > eps
        and realized_displacement is not None
        and math.isfinite(realized_displacement)
    ):
        ratio = float(realized_displacement / (predicted_displacement + eps))

    numerical_radius_floor = max(
        float(np.finfo(float).tiny),
        float(math.ulp(max(radius_before, float(np.finfo(float).tiny)))),
    )
    numerical_floor_applied = False
    rate_limit_applied = False
    adaptive_policy = policy in {
        ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_V1,
        ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2,
    }
    unbounded_policy = bool(
        policy
        == ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2
    )
    sr_v2_policy = bool(
        str(summary.get("joint_linear_solve_policy_effective", ""))
        == JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2
    )
    seed_guard = dict(seed_guard_payload or {})
    optimizer_outcome = dict(optimizer_outcome_payload or {})
    path_prediction = dict(accepted_path_prediction or {})
    seed_guard_status = str(seed_guard.get("status", ""))
    seed_guard_failure_kind = str(
        seed_guard.get("transaction_failure_kind", "")
    )
    explicit_transaction_failure = (
        None
        if transaction_failure_reason in {None, ""}
        else str(transaction_failure_reason)
    )
    mapped_seed_exact_reduction = _finite_float(
        seed_guard.get("mapped_seed_exact_gain")
    )
    mapped_seed_predicted_reduction = _finite_float(
        seed_guard.get(
            "mapped_seed_predicted_reduction",
            predicted_energy_reduction,
        )
    )
    accepted_state_source = str(
        optimizer_outcome.get(
            "post_refit_safe_source",
            optimizer_outcome.get("selected_source", ""),
        )
    )
    if not accepted_state_source and seed_guard_status == "accepted":
        accepted_state_source = "mapped_downhill_seed"
    powell_endpoint_energy = _finite_float(
        optimizer_outcome.get("optimizer_energy")
    )
    powell_endpoint_reduction_diagnostic = (
        None
        if powell_endpoint_energy is None
        else float(float(energy_before) - powell_endpoint_energy)
    )
    accepted_path_prediction_certified = bool(
        path_prediction.get("certified", False)
    )
    accepted_path_predicted_reduction = _finite_float(
        path_prediction.get("predicted_reduction")
    )
    model_agreement_ratio: float | None = None
    model_agreement_authority = "unavailable"
    section8_displacement_fallback_applied = False

    context_supported = bool(
        (
            str(context_mode) == BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1
            and bool(full_coordinate_refit)
        )
        or (
            str(context_mode) == SR_ACTIVE_ONLY_CORRECTION_CONTEXT_V1
            and bool(full_coordinate_refit)
        )
        or (
            bool(allow_historical_singleton_context)
            and str(context_mode)
            == HISTORICAL_SINGLETON_SCALAR_TRUST_CONTEXT_V1
        )
    )

    if adaptive_policy and sr_v2_policy:
        numerical_or_mapping_failure = bool(
            explicit_transaction_failure is not None
            or stabilized_trust_transaction_failure is not None
            or seed_guard_failure_kind
            in {
                "mapping",
                "mapping_contract",
                "nonfinite_path",
                "stabilized_path",
                "state_consistency",
            }
            or seed_guard_status == "unavailable"
        )
        if numerical_or_mapping_failure:
            update_reason = "numerical_or_mapping_failure_hold"
        elif not context_supported:
            update_reason = "context_mode_not_supported"
        elif stabilized_trust_transaction is None:
            update_reason = "stabilized_trust_path_unavailable_hold"
        elif seed_guard_status != "accepted":
            update_reason = "mapped_seed_transaction_unavailable_hold"
        else:
            realized_trust_displacement = float(
                stabilized_trust_transaction[
                    "realized_stabilized_trust_displacement"
                ]
            )
            exact_fs_motion_resolved = bool(
                realized_fs_displacement_exact is not None
                and math.isfinite(realized_fs_displacement_exact)
                and float(realized_fs_displacement_exact) > eps
            )
            trust_motion_resolved = bool(
                math.isfinite(realized_trust_displacement)
                and realized_trust_displacement > eps
            )
            seed_selected = accepted_state_source in {
                "mapped_downhill_seed",
                "mapped_active_restriction_seed",
                "mapped_retained_joint_candidate",
            }
            optimizer_selected = accepted_state_source == "optimizer_result"
            if seed_selected:
                if (
                    mapped_seed_predicted_reduction is None
                    or mapped_seed_predicted_reduction <= 0.0
                    or mapped_seed_exact_reduction is None
                    or mapped_seed_exact_reduction
                    <= float(max(0.0, energy_improvement_tolerance))
                    or not trust_motion_resolved
                    or not exact_fs_motion_resolved
                ):
                    update_reason = "mapped_seed_agreement_unresolved_hold"
                else:
                    model_agreement_ratio = float(
                        mapped_seed_exact_reduction
                        / mapped_seed_predicted_reduction
                    )
                    radius_after = float(
                        realized_trust_displacement
                        * math.sqrt(max(0.0, model_agreement_ratio))
                    )
                    if not math.isfinite(radius_after) or radius_after <= 0.0:
                        radius_after = float(radius_before)
                        update_factor = 1.0
                        update_reason = "mapped_seed_calibration_nonfinite_hold"
                    else:
                        update_factor = float(radius_after / radius_before)
                        update_reason = "mapped_seed_model_agreement_calibrated"
                        model_agreement_authority = (
                            "quadratic_prediction_vs_exact_mapped_seed_v1"
                        )
            elif optimizer_selected:
                if (
                    accepted_path_prediction_certified
                    and accepted_path_predicted_reduction is not None
                    and accepted_path_predicted_reduction > 0.0
                    and realized_energy_reduction
                    > float(max(0.0, energy_improvement_tolerance))
                    and trust_motion_resolved
                    and exact_fs_motion_resolved
                ):
                    model_agreement_ratio = float(
                        realized_energy_reduction
                        / accepted_path_predicted_reduction
                    )
                    radius_after = float(
                        realized_trust_displacement
                        * math.sqrt(max(0.0, model_agreement_ratio))
                    )
                    if not math.isfinite(radius_after) or radius_after <= 0.0:
                        radius_after = float(radius_before)
                        update_factor = 1.0
                        update_reason = "post_powell_path_calibration_nonfinite_hold"
                    else:
                        update_factor = float(radius_after / radius_before)
                        update_reason = "post_powell_certified_path_model_calibrated"
                        model_agreement_authority = (
                            "certified_accepted_path_prediction_vs_post_powell_v1"
                        )
                elif (
                    realized_energy_reduction
                    > float(max(0.0, energy_improvement_tolerance))
                    and trust_motion_resolved
                    and exact_fs_motion_resolved
                ):
                    radius_after = float(realized_trust_displacement)
                    update_factor = float(radius_after / radius_before)
                    update_reason = (
                        "post_powell_stabilized_displacement_fallback"
                    )
                    model_agreement_authority = (
                        "no_corresponding_powell_path_prediction_v1"
                    )
                    section8_displacement_fallback_applied = True
                else:
                    update_reason = "post_powell_path_unresolved_hold"
            else:
                update_reason = "accepted_state_source_unresolved_hold"
    elif adaptive_policy:
        if realized_displacement is None or not math.isfinite(realized_displacement):
            if unbounded_policy:
                # Unbounded-v2 numerical failures are no longer contracted to
                # an implementation floor.  The defective path is repaired or
                # remeasured while rho remains unchanged.
                update_factor = 1.0
                radius_after = float(radius_before)
                numerical_floor_applied = False
                update_reason = "invalid_measurement_contract_hold"
            else:
                update_factor = float(config.contraction_factor_min)
                radius_after = max(
                    float(config.radius_min),
                    float(radius_before) * update_factor,
                )
                update_reason = "invalid_measurement_contract"
        elif not context_supported:
            update_reason = "context_mode_not_supported"
        elif predicted_displacement is None or predicted_displacement <= eps:
            update_reason = "invalid_measurement_contract"
        else:
            ratio = float(realized_displacement / (predicted_displacement + eps))
            if ratio < 1.0:
                raw_factor = float(math.sqrt(max(0.0, ratio)))
                if unbounded_policy:
                    update_factor = raw_factor
                    radius_after = max(
                        float(numerical_radius_floor),
                        float(radius_before) * update_factor,
                    )
                    numerical_floor_applied = bool(
                        float(radius_before) * update_factor
                        < float(numerical_radius_floor)
                    )
                else:
                    update_factor = float(
                        max(
                            float(config.contraction_factor_min),
                            raw_factor,
                        )
                    )
                    rate_limit_applied = bool(update_factor != raw_factor)
                    radius_after = max(
                        float(config.radius_min),
                        float(radius_before) * update_factor,
                    )
                update_reason = "realized_displacement_smaller"
            elif ratio > 1.0 and trust_radius_binding:
                energy_descent_ok = bool(
                    realized_energy_reduction
                    > float(max(0.0, energy_improvement_tolerance))
                )
                direction_ok = bool(
                    not config.require_direction_for_expansion
                    or (
                        direction_guard_available
                        and direction_guard_passed
                    )
                )
                if not energy_descent_ok:
                    update_reason = "energy_veto_hold"
                elif not direction_ok:
                    update_reason = "direction_veto_hold"
                else:
                    raw_factor = float(math.sqrt(max(0.0, ratio)))
                    update_factor = (
                        raw_factor
                        if unbounded_policy
                        else float(
                            min(
                                float(config.expansion_factor_max),
                                raw_factor,
                            )
                        )
                    )
                    rate_limit_applied = bool(
                        (not unbounded_policy) and update_factor != raw_factor
                    )
                    radius_after = float(radius_before) * update_factor
                    update_reason = (
                        "binding_radius_realized_displacement_larger"
                    )
            else:
                update_reason = "radius_inactive_hold"
    elif policy != ROUTE_A_TRUST_REGION_FIXED:
        raise ValueError(f"Unsupported trust-region update policy: {policy}")

    payload = {
        "schema": TRUST_REGION_UPDATE_SCHEMA,
        "policy": str(policy),
        "context_mode": str(context_mode),
        "radius_before": float(radius_before),
        "radius_after": float(radius_after),
        "update_factor": float(update_factor),
        "update_reason": str(update_reason),
        "scientific_radius_lower_bound": (
            0.0 if unbounded_policy else float(config.radius_min)
        ),
        "scientific_radius_upper_bound": None,
        "rate_limit_applied": bool(rate_limit_applied),
        "numerical_radius_floor": float(numerical_radius_floor),
        "numerical_floor_applied": bool(numerical_floor_applied),
        "trust_clipped": bool(trust_radius_binding),
        "trust_radius_binding": bool(trust_radius_binding),
        "selector_regularization_applied": bool(
            selector_regularization_applied
        ),
        "predicted_fs_displacement": (
            predicted_displacement
            if stabilized_trust_transaction is None
            else None
        ),
        "predicted_raw_metric_local_displacement": (
            None
            if predicted_sq_raw_metric is None
            or predicted_sq_raw_metric < 0.0
            else float(math.sqrt(max(0.0, predicted_sq_raw_metric)))
        ),
        "predicted_stabilized_trust_displacement": (
            None
            if stabilized_trust_transaction is None
            else float(predicted_displacement)
        ),
        "realized_fs_displacement_exact": realized_fs_displacement_exact,
        "realized_stabilized_trust_displacement": (
            None
            if stabilized_trust_transaction is None
            else float(realized_displacement)
        ),
        "realized_fs_displacement_local": None,
        "displacement_ratio": ratio,
        "displacement_ratio_metric": str(displacement_authority),
        "metric_direction_cosine": None,
        "direction_guard_available": bool(direction_guard_available),
        "direction_guard_passed": bool(direction_guard_passed),
        "direction_guard_required": bool(
            config.require_direction_for_expansion
        ),
        "predicted_energy_reduction": predicted_energy_reduction,
        "mapped_seed_predicted_reduction": mapped_seed_predicted_reduction,
        "mapped_seed_exact_reduction": mapped_seed_exact_reduction,
        "post_powell_realized_reduction": float(realized_energy_reduction),
        "powell_endpoint_reduction_diagnostic": (
            powell_endpoint_reduction_diagnostic
        ),
        "accepted_state_source": (
            None if not accepted_state_source else str(accepted_state_source)
        ),
        "accepted_path_prediction_certified": bool(
            accepted_path_prediction_certified
        ),
        "accepted_path_predicted_reduction": (
            accepted_path_predicted_reduction
        ),
        "model_agreement_ratio": model_agreement_ratio,
        "model_agreement_authority": str(model_agreement_authority),
        "section8_displacement_fallback_applied": bool(
            section8_displacement_fallback_applied
        ),
        "section8_energy_comparison_order": (
            "prediction_vs_exact_mapped_seed_before_post_powell_v1"
            if sr_v2_policy
            else None
        ),
        "stabilized_trust_transaction_failure": (
            stabilized_trust_transaction_failure
        ),
        "transaction_failure_reason": explicit_transaction_failure,
        "realized_energy_reduction_pre_prune": float(
            realized_energy_reduction
        ),
        "full_coordinate_refit": bool(full_coordinate_refit),
        "historical_singleton_context_opt_in": bool(
            allow_historical_singleton_context
        ),
        "historical_singleton_context_accepted": bool(
            allow_historical_singleton_context
            and str(context_mode)
            == HISTORICAL_SINGLETON_SCALAR_TRUST_CONTEXT_V1
        ),
        "pre_admission_state_fingerprint": pre_state_fingerprint,
        "post_refit_pre_prune_state_fingerprint": post_state_fingerprint,
        "active_context_indices": [
            int(value)
            for value in (
                summary.get("geometry_workspace", {}).get("active_indices", [])
                if isinstance(summary.get("geometry_workspace"), Mapping)
                else summary.get("active_context_indices_effective", [])
            )
        ],
        "selected_record_identities": _selected_record_identities(
            selected_records
        ),
        "selected_effective_positions": [
            int(value) for value in selected_effective_positions
        ],
        "stabilized_trust_transaction": (
            None
            if stabilized_trust_transaction is None
            else dict(stabilized_trust_transaction)
        ),
    }
    state.radius = float(radius_after)
    state.update_count = int(state.update_count) + 1
    state.last_update = copy.deepcopy(payload)
    return payload


def update_sr_active_only_trust_region_state(
    state: RouteATrustRegionState,
    *,
    config: TrustRegionUpdateConfig,
    global_coordinate_summary: Mapping[str, Any],
    active_restriction_summary: Mapping[str, Any],
    active_indices: Sequence[int],
    realized_joint_step: Sequence[float],
    state_before: np.ndarray,
    state_after_refit: np.ndarray,
    energy_before: float,
    energy_after_refit: float,
    energy_improvement_tolerance: float,
    seed_guard_payload: Mapping[str, Any] | None = None,
    optimizer_outcome_payload: Mapping[str, Any] | None = None,
    accepted_path_prediction: Mapping[str, Any] | None = None,
    transaction_failure_reason: str | None = None,
) -> dict[str, Any]:
    """Update trust after a certified no-admission active-only correction.

    The correction is a full refit only within the certified active
    restriction of the already-built global supported metric.  No candidate
    coordinate or selected-record identity is accepted by this transaction.
    """

    global_summary = dict(global_coordinate_summary)
    summary = dict(active_restriction_summary)
    active = [int(value) for value in active_indices]
    if len(set(active)) != len(active):
        raise ValueError("SR active-only indices must be unique.")
    if any(value < 0 for value in active):
        raise ValueError("SR active-only indices must be nonnegative.")
    if not active:
        raise ValueError("SR active-only correction requires active coordinates.")
    if not bool(summary.get("valid", False)) or not bool(
        summary.get("trust_global_optimality_certified", False)
    ):
        raise ValueError(
            "SR active-only trust update requires a valid global-optimality "
            "certificate."
        )
    active_count = int(summary.get("active_coordinate_count", len(active)))
    if active_count != len(active):
        raise ValueError(
            "SR active-only active-coordinate count does not match its "
            "workspace indices."
        )
    joint_step = np.asarray(summary.get("joint_step", []), dtype=float).reshape(-1)
    if int(joint_step.size) < int(active_count):
        raise ValueError("SR active-only joint step is shorter than its active block.")
    if not np.all(np.isfinite(joint_step)):
        raise ValueError("SR active-only joint step must be finite.")
    batch_step = np.asarray(joint_step[active_count:], dtype=float)
    batch_zero_tolerance = float(
        max(
            0.0,
            _finite_float(
                summary.get("active_restriction_batch_zero_tolerance", 0.0)
            )
            or 0.0,
        )
    )
    if batch_step.size and float(np.linalg.norm(batch_step)) > batch_zero_tolerance:
        raise ValueError(
            "SR active-only correction contains a nonzero candidate block."
        )
    predicted_reduction = _finite_float(summary.get("predicted_reduction"))
    predicted_displacement_sq = _finite_float(
        summary.get("joint_fubini_study_displacement_sq")
    )
    if predicted_reduction is None or predicted_reduction <= 0.0:
        raise ValueError(
            "SR active-only correction requires positive predicted reduction."
        )
    if predicted_displacement_sq is None or predicted_displacement_sq <= 0.0:
        raise ValueError(
            "SR active-only correction requires positive predicted displacement."
        )
    energy_tolerance = float(max(0.0, energy_improvement_tolerance))
    if float(energy_after_refit) > float(energy_before) + energy_tolerance:
        raise ValueError("SR active-only correction cannot worsen the incumbent.")

    restricted_solve = summary.get("restricted_coordinate_trust_solve")
    restricted_payload = (
        dict(restricted_solve) if isinstance(restricted_solve, Mapping) else {}
    )
    updater_summary = {
        **global_summary,
        **summary,
        "joint_fubini_study_displacement_sq": float(
            predicted_displacement_sq
        ),
        "applied_predicted_reduction": float(predicted_reduction),
        "trust_regularization_applied": bool(
            restricted_payload.get(
                "trust_regularization_applied",
                restricted_payload.get("trust_clipped", False),
            )
        ),
        "trust_clipped": bool(restricted_payload.get("trust_clipped", False)),
        "trust_radius_binding": bool(
            restricted_payload.get("trust_radius_binding", False)
        ),
        "geometry_workspace": {
            "active_indices": [int(value) for value in active],
        },
    }
    if restricted_payload.get("whitened_step_norm") is not None:
        updater_summary["whitened_step_norm"] = float(
            restricted_payload["whitened_step_norm"]
        )
    elif str(
        updater_summary.get("joint_linear_solve_policy_effective", "")
    ) == JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2:
        # The accepted-path transaction reconstructs this norm from the exact
        # full-support metric.  Do not retain the enclosing candidate solve's
        # whitened norm, which describes a different joint step.
        updater_summary.pop("whitened_step_norm", None)
    payload = update_trust_region_state(
        state,
        config=config,
        context_mode=SR_ACTIVE_ONLY_CORRECTION_CONTEXT_V1,
        selector_summary=updater_summary,
        state_before=np.asarray(state_before, dtype=complex),
        state_after_refit=np.asarray(state_after_refit, dtype=complex),
        energy_before=float(energy_before),
        energy_after_refit=float(energy_after_refit),
        energy_improvement_tolerance=float(energy_tolerance),
        full_coordinate_refit=True,
        selected_records=(),
        selected_effective_positions=(),
        allow_historical_singleton_context=False,
        realized_joint_step=realized_joint_step,
        seed_guard_payload=seed_guard_payload,
        optimizer_outcome_payload=optimizer_outcome_payload,
        accepted_path_prediction=accepted_path_prediction,
        transaction_failure_reason=transaction_failure_reason,
    )
    enriched = {
        **dict(payload),
        "sr_active_only_correction": True,
        "singleton_consumed": False,
        "active_indices": [int(value) for value in active],
        "active_coordinate_count": int(active_count),
        "candidate_coordinate_count": int(joint_step.size - active_count),
        "candidate_block_zero_certified": True,
        "active_restriction_source": str(
            summary.get("active_restriction_source", "")
        ),
        "active_restriction_provenance_id": str(
            summary.get("active_restriction_provenance_id", "")
        ),
    }
    state.last_update = copy.deepcopy(enriched)
    return enriched


def update_geometry_expansion_trust_region_state(
    state: RouteATrustRegionState,
    *,
    config: TrustRegionUpdateConfig,
    state_before: np.ndarray,
    state_after_refit: np.ndarray,
    energy_before: float,
    energy_after_refit: float,
    energy_improvement_tolerance: float,
    full_coordinate_refit: bool,
    selected_records: Sequence[Mapping[str, Any]] = (),
    selected_effective_positions: Sequence[int] = (),
) -> dict[str, Any]:
    """Update adaptive trust after a novelty-only manifold expansion.

    There is no predicted coordinate displacement in geometry-expansion mode,
    so the ordinary predicted/realized ratio is undefined.  A successful full
    refit recalibrates the next radius to the realized Fubini--Study motion;
    otherwise the pre-admission radius is held.  No scalar/unwhitened model is
    substituted.
    """

    policy = str(config.policy)
    adaptive_policies = {
        ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_V1,
        ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2,
    }
    if policy not in adaptive_policies:
        raise ValueError(
            "Geometry-expansion trust handling requires an adaptive "
            "trust-region update policy."
        )
    if not bool(full_coordinate_refit):
        raise ValueError(
            "Geometry-expansion admission requires a full-coordinate refit."
        )

    radius_before = float(state.radius)
    unbounded_policy = bool(
        policy
        == ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2
    )
    numerical_radius_floor = max(
        float(np.finfo(float).tiny),
        float(math.ulp(max(radius_before, float(np.finfo(float).tiny)))),
    )
    realized_displacement: float | None
    try:
        realized_displacement = exact_fubini_study_distance(
            state_before,
            state_after_refit,
        )
    except (TypeError, ValueError, FloatingPointError):
        realized_displacement = None
    realized_energy_reduction = float(energy_before) - float(energy_after_refit)
    energy_descent_ok = bool(
        realized_energy_reduction
        > float(max(0.0, energy_improvement_tolerance))
    )
    displacement_valid = bool(
        realized_displacement is not None
        and math.isfinite(float(realized_displacement))
        and float(realized_displacement) > float(config.displacement_epsilon)
    )

    radius_after = float(radius_before)
    update_reason = "geometry_expansion_no_descent_hold"
    if not displacement_valid:
        update_reason = "geometry_expansion_invalid_displacement_hold"
    elif energy_descent_ok:
        radius_after = float(
            max(
                numerical_radius_floor
                if unbounded_policy
                else float(config.radius_min),
                float(realized_displacement),
            )
        )
        update_reason = (
            "geometry_expansion_descent_recalibrated_to_realized_displacement"
        )

    update_factor = float(radius_after / radius_before)
    payload = {
        "schema": TRUST_REGION_UPDATE_SCHEMA,
        "policy": str(policy),
        "context_mode": HISTORICAL_SINGLETON_GEOMETRY_EXPANSION_CONTEXT_V1,
        "geometry_expansion_active": True,
        "geometry_expansion_score_formula": "N3/(1+K3)",
        "scalar_or_unwhitened_fallback_used": False,
        "radius_before": float(radius_before),
        "radius_after": float(radius_after),
        "update_factor": float(update_factor),
        "update_reason": str(update_reason),
        "scientific_radius_lower_bound": (
            0.0 if unbounded_policy else float(config.radius_min)
        ),
        "scientific_radius_upper_bound": None,
        "rate_limit_applied": False,
        "numerical_radius_floor": float(numerical_radius_floor),
        "numerical_floor_applied": False,
        "trust_clipped": False,
        "trust_radius_binding": False,
        "selector_regularization_applied": False,
        "predicted_fs_displacement": None,
        "realized_fs_displacement_exact": realized_displacement,
        "realized_fs_displacement_local": None,
        "displacement_ratio": None,
        "metric_direction_cosine": None,
        "direction_guard_available": False,
        "direction_guard_passed": False,
        "direction_guard_required": False,
        "predicted_energy_reduction": None,
        "realized_energy_reduction_pre_prune": float(
            realized_energy_reduction
        ),
        "full_coordinate_refit": True,
        "historical_singleton_context_opt_in": False,
        "historical_singleton_context_accepted": False,
        "pre_admission_state_fingerprint": state_fingerprint(state_before),
        "post_refit_pre_prune_state_fingerprint": state_fingerprint(
            state_after_refit
        ),
        "active_context_indices": [],
        "selected_record_identities": _selected_record_identities(
            selected_records
        ),
        "selected_effective_positions": [
            int(value) for value in selected_effective_positions
        ],
    }
    state.radius = float(radius_after)
    state.update_count = int(state.update_count) + 1
    state.last_update = copy.deepcopy(payload)
    return payload


def contract_rejected_active_stationarity_trust_region_state(
    state: RouteATrustRegionState,
    *,
    config: TrustRegionUpdateConfig,
    guard_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Refine a branch radius after finite active-endpoint disagreement.

    The exact dyadic guard has already shown that every tested nonlinear
    endpoint is finite but none simultaneously satisfies exact descent and the
    physical endpoint-radius gate.  This is not an admission or a state
    transition.  Only the branch-local radius is halved so the supported model
    can be rebuilt at the next controller service.  Mapping, state, and
    nonfinite failures remain on the existing fail-closed hold path.
    """

    if str(config.policy) != ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2:
        raise ValueError(
            "Rejected SR active-stationarity refinement requires the "
            "unbounded-v2 adaptive trust policy."
        )
    guard = dict(guard_payload)
    failure_kind = str(guard.get("transaction_failure_kind", ""))
    fail_closed_kinds = {
        "mapping",
        "mapping_contract",
        "nonfinite_path",
        "stabilized_path",
        "state_consistency",
    }
    finite_disagreement_certified = bool(
        failure_kind == "finite_nonlinear_model_disagreement"
        and bool(guard.get("nonlinear_backtracking_exhausted", False))
        and bool(guard.get("all_backtracking_candidates_finite", False))
        and str(guard.get("trust_action", "")) == "contract_branch_radius"
    )
    if failure_kind in fail_closed_kinds or not finite_disagreement_certified:
        return hold_trust_region_state_after_failed_transaction(
            state,
            config=config,
            context_mode=SR_ACTIVE_STATIONARITY_BACKTRACKING_CONTEXT_V1,
            reason="sr_active_stationarity_backtracking_certificate_hold",
            failure_kind=(failure_kind or "certificate_unresolved"),
            audit_payload=guard,
        )

    radius_before = float(state.radius)
    if not math.isfinite(radius_before) or radius_before <= 0.0:
        raise ValueError(
            "SR active-stationarity refinement requires a finite positive "
            "branch radius."
        )
    numerical_radius_floor = float(
        max(
            np.finfo(float).tiny,
            math.ulp(max(radius_before, float(np.finfo(float).tiny))),
        )
    )
    unconstrained_radius_after = float(0.5 * radius_before)
    radius_after = float(
        max(numerical_radius_floor, unconstrained_radius_after)
    )
    refinement_progressed = bool(radius_after < radius_before)
    payload = {
        "schema": TRUST_REGION_UPDATE_SCHEMA,
        "policy": str(config.policy),
        "context_mode": SR_ACTIVE_STATIONARITY_BACKTRACKING_CONTEXT_V1,
        "radius_before": float(radius_before),
        "radius_after": float(radius_after),
        "update_factor": float(radius_after / radius_before),
        "update_reason": (
            "finite_nonlinear_endpoint_disagreement_half_radius_retry"
            if refinement_progressed
            else "finite_nonlinear_endpoint_disagreement_numerical_floor"
        ),
        "trust_transaction_status": "radius_refinement",
        "trust_transaction_failure_kind": str(failure_kind),
        "numerical_or_mapping_failure": False,
        "no_state_transition": True,
        "singleton_consumed": False,
        "ansatz_state_unchanged": True,
        "parameter_state_unchanged": True,
        "controller_depth_mutated": False,
        "admission_history_unchanged": True,
        "scientific_radius_lower_bound": 0.0,
        "scientific_radius_upper_bound": None,
        "rate_limit_applied": False,
        "numerical_radius_floor": float(numerical_radius_floor),
        "numerical_floor_applied": bool(
            radius_after > unconstrained_radius_after
        ),
        "radius_refinement_progressed": bool(refinement_progressed),
        "radius_refinement_factor_authority": (
            "largest_first_dyadic_endpoint_backtracking_half_radius_v1"
        ),
        "audit_payload": copy.deepcopy(guard),
    }
    state.radius = float(radius_after)
    state.update_count = int(state.update_count) + 1
    state.last_update = copy.deepcopy(payload)
    return payload


def contract_rejected_saddle_trust_region_state(
    state: RouteATrustRegionState,
    *,
    config: TrustRegionUpdateConfig,
    guard_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Backtrack an SR saddle radius after two finite non-downhill signs.

    This is a controller refinement, not an admission.  It deliberately
    mutates only the branch-local trust radius and its audit receipt; ansatz,
    parameter, depth, and energy state remain the caller's responsibility and
    must be preserved unchanged.
    """

    if str(config.policy) != ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2:
        raise ValueError(
            "Rejected SR saddle contraction requires the unbounded-v2 adaptive "
            "trust policy."
        )
    failure_kind_raw = guard_payload.get("transaction_failure_kind")
    failure_kind = (
        "" if failure_kind_raw is None else str(failure_kind_raw)
    )
    if failure_kind:
        return hold_trust_region_state_after_failed_transaction(
            state,
            config=config,
            context_mode="sr_saddle_rejected_step_v1",
            reason="sr_saddle_transaction_failure_marker_hold",
            failure_kind=failure_kind,
            audit_payload=guard_payload,
        )
    if not bool(guard_payload.get("all_mapped_signs_finite", False)) or not bool(
        guard_payload.get("all_mapped_signs_non_downhill", False)
    ):
        return hold_trust_region_state_after_failed_transaction(
            state,
            config=config,
            context_mode="sr_saddle_rejected_step_v1",
            reason="sr_saddle_incomplete_or_nonfinite_sign_pair_hold",
            failure_kind=str(
                guard_payload.get("transaction_failure_kind", "mapping")
            ),
            audit_payload=guard_payload,
        )

    radius_before = float(state.radius)
    certificate_raw = guard_payload.get(
        "saddle_taylor_contraction_certificate"
    )
    certificate = (
        dict(certificate_raw) if isinstance(certificate_raw, Mapping) else {}
    )
    mu_lower = _finite_float(
        certificate.get("negative_curvature_magnitude_lower_bound")
    )
    l3_upper = _finite_float(
        certificate.get("third_derivative_upper_bound")
    )
    comparison_width = _finite_float(
        certificate.get("energy_comparison_width")
    )
    radius_after = _finite_float(certificate.get("certified_radius_after"))
    certificate_valid = bool(
        certificate.get("valid", False)
        and mu_lower is not None
        and mu_lower > 0.0
        and l3_upper is not None
        and l3_upper >= 0.0
        and comparison_width is not None
        and comparison_width >= 0.0
        and radius_after is not None
        and 0.0 < radius_after < radius_before
    )
    taylor_lower_bound = (
        None
        if not certificate_valid
        else float(
            0.5 * float(mu_lower) * float(radius_after) ** 2
            - (1.0 / 6.0) * float(l3_upper) * float(radius_after) ** 3
        )
    )
    taylor_contraction_available = bool(
        certificate_valid
        and taylor_lower_bound is not None
        and taylor_lower_bound > float(comparison_width)
    )
    if not taylor_contraction_available and l3_upper is None:
        # Section 8 permits ordinary numerical backtracking when no mapped
        # third-derivative bound is available.  This is a typed refinement of
        # rho only; it is not a Taylor certificate and it never applies to an
        # incomplete, nonfinite, or unmappable sign pair (handled above).
        radius_after_numerical = float(0.5 * radius_before)
        if (
            not math.isfinite(radius_after_numerical)
            or radius_after_numerical <= 0.0
            or radius_after_numerical >= radius_before
        ):
            return hold_trust_region_state_after_failed_transaction(
                state,
                config=config,
                context_mode="sr_saddle_exact_sign_model_disagreement_v1",
                reason="sr_saddle_numerical_backtracking_unrepresentable_hold",
                failure_kind="numerical_radius_resolution",
                audit_payload={
                    "guard_payload": copy.deepcopy(dict(guard_payload)),
                    "radius_before": float(radius_before),
                },
            )
        numerical_receipt = {
            "schema": "sr_saddle_numerical_backtracking_receipt_v1",
            "valid": True,
            "reason": "third_derivative_bound_unavailable",
            "backtracking_rule": "ordinary_geometric_half_radius_v1",
            "backtracking_factor": 0.5,
            "radius_before": float(radius_before),
            "radius_after": float(radius_after_numerical),
            "certified_taylor_guarantee": False,
            "all_mapped_signs_finite": True,
            "all_mapped_signs_non_downhill": True,
            "energy_comparison_width": copy.deepcopy(
                guard_payload.get("energy_comparison_width", {})
            ),
        }
        payload = {
            "schema": TRUST_REGION_UPDATE_SCHEMA,
            "policy": str(config.policy),
            "context_mode": "sr_saddle_exact_sign_model_disagreement_v1",
            "radius_before": float(radius_before),
            "radius_after": float(radius_after_numerical),
            "update_factor": 0.5,
            "update_reason": "sr_saddle_numerical_backtracking_contract_refine",
            "contraction_basis": (
                "ordinary_geometric_backtracking_without_taylor_guarantee_v1"
            ),
            "certified_taylor_radius_available": False,
            "saddle_taylor_contraction_certificate": copy.deepcopy(certificate),
            "saddle_numerical_backtracking_receipt": numerical_receipt,
            "requires_refinement": True,
            "ansatz_state_mutated": False,
            "parameter_state_mutated": False,
            "depth_mutated": False,
            "scientific_radius_lower_bound": 0.0,
            "scientific_radius_upper_bound": None,
            "numerical_radius_floor": None,
            "numerical_floor_applied": False,
            "mapped_seed_incumbent_energy": _finite_float(
                guard_payload.get("mapped_seed_incumbent_energy")
            ),
            "sign_evaluations": copy.deepcopy(
                list(guard_payload.get("sign_evaluations", []))
            ),
            "sr_saddle_transaction_outcome": (
                "radius_contract_refinement_no_state_mutation"
            ),
        }
        state.radius = float(radius_after_numerical)
        state.update_count = int(state.update_count) + 1
        state.last_update = copy.deepcopy(payload)
        return payload
    if not taylor_contraction_available:
        return hold_trust_region_state_after_failed_transaction(
            state,
            config=config,
            context_mode="sr_saddle_exact_sign_model_disagreement_v1",
            reason="sr_saddle_taylor_contraction_certificate_unavailable_hold",
            failure_kind="certificate_unresolved",
            audit_payload={
                "guard_payload": copy.deepcopy(dict(guard_payload)),
                "saddle_taylor_contraction_certificate": certificate,
                "taylor_lower_bound": taylor_lower_bound,
            },
        )
    assert radius_after is not None
    payload = {
        "schema": TRUST_REGION_UPDATE_SCHEMA,
        "policy": str(config.policy),
        "context_mode": "sr_saddle_exact_sign_model_disagreement_v1",
        "radius_before": float(radius_before),
        "radius_after": float(radius_after),
        "update_factor": float(radius_after / radius_before),
        "update_reason": "sr_saddle_taylor_certified_contract_refine",
        "contraction_basis": "section8_certified_taylor_radius_v1",
        "certified_taylor_radius_available": True,
        "saddle_taylor_contraction_certificate": copy.deepcopy(certificate),
        "certified_negative_curvature_magnitude_lower_bound": float(mu_lower),
        "certified_third_derivative_upper_bound": float(l3_upper),
        "energy_comparison_width": float(comparison_width),
        "certified_taylor_lower_bound": float(taylor_lower_bound),
        "requires_refinement": True,
        "ansatz_state_mutated": False,
        "parameter_state_mutated": False,
        "depth_mutated": False,
        "scientific_radius_lower_bound": 0.0,
        "scientific_radius_upper_bound": None,
        "numerical_radius_floor": None,
        "numerical_floor_applied": False,
        "mapped_seed_incumbent_energy": _finite_float(
            guard_payload.get("mapped_seed_incumbent_energy")
        ),
        "sign_evaluations": copy.deepcopy(
            list(guard_payload.get("sign_evaluations", []))
        ),
        "sr_saddle_transaction_outcome": (
            "radius_contract_refinement_no_state_mutation"
        ),
    }
    state.radius = float(radius_after)
    state.update_count = int(state.update_count) + 1
    state.last_update = copy.deepcopy(payload)
    return payload


__all__ = [
    "HISTORICAL_SINGLETON_GEOMETRY_EXPANSION_CONTEXT_V1",
    "HISTORICAL_SINGLETON_SCALAR_SELECTOR_SUMMARY_SCHEMA",
    "HISTORICAL_SINGLETON_SCALAR_TRUST_CONTEXT_V1",
    "ROUND_TRUST_REGION_SNAPSHOT_SCHEMA",
    "ROUND_TRUST_REGION_STAGE_NAMES",
    "ROUND_TRUST_REGION_STAGE_RECEIPT_SCHEMA",
    "RouteARoundTrustRegionSnapshot",
    "RouteATrustRegionState",
    "TRUST_REGION_STATE_SCHEMA",
    "TRUST_REGION_UPDATE_SCHEMA",
    "exact_fubini_study_distance",
    "historical_singleton_scalar_selector_summary",
    "initialize_trust_region_state",
    "resolve_round_trust_region_snapshot",
    "round_trust_region_stage_receipt",
    "score_config_with_round_trust_radius",
    "selector_config_with_round_trust_radius",
    "selector_config_with_trust_radius",
    "state_fingerprint",
    "contract_rejected_saddle_trust_region_state",
    "hold_trust_region_state_after_failed_transaction",
    "update_sr_active_only_trust_region_state",
    "update_trust_region_state",
    "update_geometry_expansion_trust_region_state",
]
