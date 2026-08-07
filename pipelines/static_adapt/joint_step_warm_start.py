"""Exact applied joint-selector step mapping for optimizer initialization."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pipelines.static_adapt.accepted_refit import (
    SupportedFSPowellChart,
    map_phase_order_joint_step_to_supported_fs,
)
from pipelines.static_adapt.nested_windows import (
    NestedWindowError,
    build_composed_batch_window_payload,
)
from pipelines.static_adapt.schur_warm_start import SeedProposal, select_guarded_seed
from src.quantum.ansatz_parameterization import AnsatzParameterLayout


ROUTE_A_JOINT_STEP_WARM_START_OFF = "off"
ROUTE_A_JOINT_STEP_WARM_START_EXACT_GUARDED_V1 = (
    "exact_applied_joint_step_guarded_v1"
)
ROUTE_A_JOINT_STEP_WARM_START_MODES = frozenset(
    {
        ROUTE_A_JOINT_STEP_WARM_START_OFF,
        ROUTE_A_JOINT_STEP_WARM_START_EXACT_GUARDED_V1,
    }
)


@dataclass(frozen=True)
class RouteAJointStepWarmStartConfig:
    mode: str = ROUTE_A_JOINT_STEP_WARM_START_OFF
    guard_abs_tol: float = 1.0e-12
    guard_rel_tol: float = 1.0e-12

    def __post_init__(self) -> None:
        if str(self.mode) not in ROUTE_A_JOINT_STEP_WARM_START_MODES:
            raise ValueError(
                "joint-step warm-start mode must be one of "
                f"{sorted(ROUTE_A_JOINT_STEP_WARM_START_MODES)}."
            )
        for field_name in ("guard_abs_tol", "guard_rel_tol"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and nonnegative.")

    @property
    def enabled(self) -> bool:
        return str(self.mode) == ROUTE_A_JOINT_STEP_WARM_START_EXACT_GUARDED_V1

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": str(self.mode),
            "guard_abs_tol": float(self.guard_abs_tol),
            "guard_rel_tol": float(self.guard_rel_tol),
        }


def _record_label(record: Mapping[str, Any]) -> str:
    value = record.get("candidate_label")
    if value not in {None, ""}:
        return str(value)
    feature = record.get("feature")
    feature_value = getattr(feature, "candidate_label", None)
    if feature_value not in {None, ""}:
        return str(feature_value)
    term = record.get("candidate_term")
    term_value = getattr(term, "label", None)
    return "" if term_value in {None, ""} else str(term_value)


def _sequence_of_floats(value: Any) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        return None
    try:
        resolved = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return resolved if np.all(np.isfinite(np.asarray(resolved, dtype=float))) else None


def _sequence_of_ints(value: Any) -> list[int] | None:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        return None
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        return None


def _finite_float(value: Any) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if math.isfinite(resolved) else None


def certify_exact_physical_transition(
    *,
    guard_payload: Mapping[str, Any],
    mapped_optimizer_x: np.ndarray,
    transition_optimizer_x: np.ndarray,
    transition_runtime_parameters: np.ndarray,
    transition_energy: float,
    transition_state_source: str,
    declared_operator_labels: Sequence[str],
    realized_operator_labels: Sequence[str],
    operator_semantic_order_match: bool,
    operator_identity_order_match: bool | None = None,
    declared_logical_parameter_count: int,
    realized_logical_parameter_count: int,
    declared_runtime_parameter_count: int,
    realized_runtime_parameter_count: int,
    declared_order_state: np.ndarray | None,
    realized_order_state: np.ndarray | None,
    state_consistency_tolerance: float,
    state_consistency_tolerance_source: str,
    declared_state_fingerprint: str | None = None,
    realized_state_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Certify the exact, post-map state before an SR transition is committed.

    The quadratic/local controller is not a physical-state authority.  This
    certificate becomes true only after its joint step has passed the exact
    objective guard and the selected final point has been replayed against an
    independently reconstructed declared operator order.  A global phase is
    removed before the two replay states are compared.
    """

    declared_labels = [str(value) for value in declared_operator_labels]
    realized_labels = [str(value) for value in realized_operator_labels]
    tolerance = _finite_float(state_consistency_tolerance)
    tolerance_valid = bool(tolerance is not None and tolerance >= 0.0)
    tolerance_value = (
        float(max(1.0e-12, tolerance)) if tolerance_valid else float("nan")
    )

    def _finite_vector(value: Any) -> tuple[np.ndarray | None, bool]:
        try:
            array = np.asarray(value, dtype=float).reshape(-1).copy()
        except (TypeError, ValueError):
            return None, False
        return array, bool(array.size > 0 and np.all(np.isfinite(array)))

    mapped_x, mapped_x_finite = _finite_vector(mapped_optimizer_x)
    transition_x, transition_x_finite = _finite_vector(transition_optimizer_x)
    transition_runtime, transition_runtime_finite = _finite_vector(
        transition_runtime_parameters
    )
    optimizer_shape_match = bool(
        mapped_x is not None
        and transition_x is not None
        and mapped_x.shape == transition_x.shape
    )
    transition_energy_value = _finite_float(transition_energy)
    mapped_energy_value = _finite_float(
        guard_payload.get("mapped_seed_proposal_energy")
    )
    exact_guard_accepted = bool(
        str(guard_payload.get("status", "")) == "accepted"
        and not bool(guard_payload.get("fallback_to_incumbent", True))
        and mapped_energy_value is not None
    )

    declared_logical_count = int(declared_logical_parameter_count)
    realized_logical_count = int(realized_logical_parameter_count)
    declared_runtime_count = int(declared_runtime_parameter_count)
    realized_runtime_count = int(realized_runtime_parameter_count)
    layout_counts_finite = bool(
        min(
            declared_logical_count,
            realized_logical_count,
            declared_runtime_count,
            realized_runtime_count,
        )
        >= 0
    )
    parameter_layout_match = bool(
        layout_counts_finite
        and declared_logical_count == realized_logical_count
        and declared_runtime_count == realized_runtime_count
    )
    runtime_parameter_count_match = bool(
        transition_runtime is not None
        and int(transition_runtime.size) == realized_runtime_count
    )
    operator_label_order_match = bool(declared_labels == realized_labels)
    declared_order_preserved = bool(
        operator_label_order_match and operator_semantic_order_match
    )

    def _state_array(value: Any) -> tuple[np.ndarray | None, bool]:
        if value is None:
            return None, False
        try:
            state = np.asarray(value, dtype=complex).reshape(-1).copy()
        except (TypeError, ValueError):
            return None, False
        finite = bool(
            state.size > 0
            and np.all(np.isfinite(state.real))
            and np.all(np.isfinite(state.imag))
        )
        return state, finite

    declared_state, declared_state_finite = _state_array(declared_order_state)
    realized_state, realized_state_finite = _state_array(realized_order_state)
    state_shape_match = bool(
        declared_state is not None
        and realized_state is not None
        and declared_state.shape == realized_state.shape
    )
    declared_state_norm = (
        None
        if declared_state is None or not declared_state_finite
        else float(np.linalg.norm(declared_state))
    )
    realized_state_norm = (
        None
        if realized_state is None or not realized_state_finite
        else float(np.linalg.norm(realized_state))
    )
    normalization_tolerance = (
        float(max(1.0e-10, tolerance_value))
        if tolerance_valid
        else float("nan")
    )
    states_normalized = bool(
        tolerance_valid
        and declared_state_norm is not None
        and realized_state_norm is not None
        and abs(float(declared_state_norm) - 1.0) <= normalization_tolerance
        and abs(float(realized_state_norm) - 1.0) <= normalization_tolerance
    )

    overlap_abs: float | None = None
    alignment_real: float | None = None
    alignment_imag: float | None = None
    phase_aligned_distance: float | None = None
    if (
        declared_state_finite
        and realized_state_finite
        and state_shape_match
        and declared_state is not None
        and realized_state is not None
    ):
        overlap = complex(np.vdot(declared_state, realized_state))
        overlap_abs = float(abs(overlap))
        alignment = (
            complex(np.conjugate(overlap) / abs(overlap))
            if abs(overlap) > 0.0
            else complex(1.0, 0.0)
        )
        aligned_realized = np.asarray(alignment * realized_state, dtype=complex)
        alignment_real = float(alignment.real)
        alignment_imag = float(alignment.imag)
        phase_aligned_distance = float(
            np.linalg.norm(declared_state - aligned_realized)
        )
    state_consistency_certified = bool(
        tolerance_valid
        and declared_state_finite
        and realized_state_finite
        and state_shape_match
        and states_normalized
        and phase_aligned_distance is not None
        and math.isfinite(float(phase_aligned_distance))
        and float(phase_aligned_distance) <= tolerance_value
    )
    resulting_circuit_evaluated = bool(
        mapped_energy_value is not None and transition_energy_value is not None
    )
    finite_map_certified = bool(
        mapped_x_finite
        and transition_x_finite
        and optimizer_shape_match
        and parameter_layout_match
        and transition_runtime_finite
        and runtime_parameter_count_match
    )

    checks = {
        "exact_guard_accepted": bool(exact_guard_accepted),
        "finite_map_certified": bool(finite_map_certified),
        "resulting_circuit_evaluated": bool(resulting_circuit_evaluated),
        "declared_operator_order_preserved": bool(declared_order_preserved),
        "state_consistency_certified": bool(state_consistency_certified),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    physical_transition_certified = bool(not failed_checks)
    return {
        "schema": "sr_saddle_physical_transition_certificate_v1",
        "status": "certified" if physical_transition_certified else "unavailable",
        "reason": (
            "exact_mapped_physical_transition_certified"
            if physical_transition_certified
            else f"physical_transition_check_failed::{failed_checks[0]}"
        ),
        "certification_scope": (
            "accepted_exact_guard_then_seed_preserving_final_state_v1"
        ),
        "physical_transition_certified": bool(physical_transition_certified),
        "failed_checks": list(failed_checks),
        "exact_guard_status": str(guard_payload.get("status", "")),
        "exact_guard_accepted": bool(exact_guard_accepted),
        "mapped_seed_energy": mapped_energy_value,
        "transition_energy": transition_energy_value,
        "transition_state_source": str(transition_state_source),
        "resulting_circuit_evaluated": bool(resulting_circuit_evaluated),
        "finite_map_status": "certified" if finite_map_certified else "failed",
        "finite_map_certified": bool(finite_map_certified),
        "mapped_optimizer_point_finite": bool(mapped_x_finite),
        "transition_optimizer_point_finite": bool(transition_x_finite),
        "transition_runtime_parameter_vector_finite": bool(
            transition_runtime_finite
        ),
        "transition_runtime_parameter_count_match": bool(
            runtime_parameter_count_match
        ),
        "optimizer_coordinate_shape_match": bool(optimizer_shape_match),
        "declared_logical_parameter_count": int(declared_logical_count),
        "realized_logical_parameter_count": int(realized_logical_count),
        "declared_runtime_parameter_count": int(declared_runtime_count),
        "realized_runtime_parameter_count": int(realized_runtime_count),
        "parameter_layout_match": bool(parameter_layout_match),
        "declared_operator_labels": list(declared_labels),
        "realized_operator_labels": list(realized_labels),
        "operator_label_order_match": bool(operator_label_order_match),
        "operator_semantic_order_match": bool(operator_semantic_order_match),
        "operator_identity_order_match": (
            None
            if operator_identity_order_match is None
            else bool(operator_identity_order_match)
        ),
        "declared_operator_order_preserved": bool(declared_order_preserved),
        "declared_state_fingerprint": (
            None
            if declared_state_fingerprint in {None, ""}
            else str(declared_state_fingerprint)
        ),
        "realized_state_fingerprint": (
            None
            if realized_state_fingerprint in {None, ""}
            else str(realized_state_fingerprint)
        ),
        "declared_state_finite": bool(declared_state_finite),
        "realized_state_finite": bool(realized_state_finite),
        "state_shape_match": bool(state_shape_match),
        "declared_state_norm": declared_state_norm,
        "realized_state_norm": realized_state_norm,
        "states_normalized": bool(states_normalized),
        "normalization_tolerance": (
            float(normalization_tolerance) if tolerance_valid else None
        ),
        "phase_alignment_overlap_abs": overlap_abs,
        "phase_alignment_factor_real": alignment_real,
        "phase_alignment_factor_imag": alignment_imag,
        "phase_aligned_state_distance": phase_aligned_distance,
        "state_consistency_tolerance": (
            float(tolerance_value) if tolerance_valid else None
        ),
        "state_consistency_tolerance_source": str(
            state_consistency_tolerance_source
        ),
        "state_consistency_certified": bool(state_consistency_certified),
    }


def _energy_comparison_width_payload(
    *,
    numerical_width: float,
    optimizer_reproducibility_allowance: float = 0.0,
    event_schema: str,
) -> dict[str, Any]:
    """Type the simultaneous width used by an exact-energy comparison."""

    numerical = float(max(0.0, float(numerical_width)))
    optimizer_allowance = float(
        max(0.0, float(optimizer_reproducibility_allowance))
    )
    aggregate = float(numerical + optimizer_allowance)
    return {
        "schema": "sr_simultaneous_energy_comparison_width_v1",
        "comparison_event_schema": str(event_schema),
        "numerical_energy_comparison_width": numerical,
        "optimizer_reproducibility_allowance": optimizer_allowance,
        "aggregate_simultaneous_comparison_width": aggregate,
        "optimizer_allowance_authority": (
            "not_applicable_before_disposable_optimizer_probe"
            if optimizer_allowance == 0.0
            else "measured_optimizer_reproducibility_allowance"
        ),
    }


def guard_atomic_joint_step_candidates(
    *,
    objective: Callable[[np.ndarray], float],
    canonical_x0: np.ndarray,
    joint_steps: Sequence[Sequence[float]],
    map_joint_step: Callable[
        [Sequence[float]],
        tuple[np.ndarray | None, Mapping[str, Any]],
    ],
    predicted_reductions: Sequence[float | None] | None = None,
    candidate_roles: Sequence[str] | None = None,
    guard_abs_tol: float = 1.0e-12,
    guard_rel_tol: float = 1.0e-12,
    incumbent_energy: float | None = None,
    endpoint_distance_from_incumbent: Callable[[np.ndarray], float] | None = None,
    max_endpoint_distance: float | None = None,
) -> tuple[np.ndarray, dict[str, Any], int]:
    """Atomically map and exactly compare every retained joint candidate.

    This is the coordinate-agnostic guard used by active-only SR corrections.
    The caller owns the physical joint-to-optimizer map.  Every retained
    candidate must map before *any* objective call, and every mapped objective
    must be finite before a downhill candidate can be selected.  When an
    endpoint-distance service is supplied, every mapped endpoint distance must
    also be finite and the selected candidate must lie within the declared
    budget.  Consequently an insertion/map failure or a nonfinite path is a
    typed no-state/hold outcome rather than an accidental one-sided sign
    choice.
    """

    incumbent = np.asarray(canonical_x0, dtype=float).reshape(-1).copy()
    abs_tol = float(guard_abs_tol)
    rel_tol = float(guard_rel_tol)
    if not np.all(np.isfinite(incumbent)):
        raise ValueError("Atomic joint-step incumbent must be finite.")
    if not math.isfinite(abs_tol) or abs_tol < 0.0:
        raise ValueError("guard_abs_tol must be finite and nonnegative.")
    if not math.isfinite(rel_tol) or rel_tol < 0.0:
        raise ValueError("guard_rel_tol must be finite and nonnegative.")
    precomputed_incumbent_energy = _finite_float(incumbent_energy)
    if incumbent_energy is not None and precomputed_incumbent_energy is None:
        raise ValueError("incumbent_energy must be finite when provided.")
    endpoint_gate_requested = bool(
        endpoint_distance_from_incumbent is not None
        or max_endpoint_distance is not None
    )
    if endpoint_gate_requested and (
        endpoint_distance_from_incumbent is None
        or max_endpoint_distance is None
    ):
        raise ValueError(
            "Endpoint-distance guarding requires both a distance service and "
            "a maximum distance."
        )
    endpoint_distance_budget = _finite_float(max_endpoint_distance)
    if endpoint_gate_requested and (
        endpoint_distance_budget is None or endpoint_distance_budget < 0.0
    ):
        raise ValueError("max_endpoint_distance must be finite and nonnegative.")
    endpoint_distance_tolerance = (
        None
        if endpoint_distance_budget is None
        else float(
            max(
                1.0e-14,
                4096.0
                * np.finfo(float).eps
                * max(1.0, float(endpoint_distance_budget)),
            )
        )
    )

    requested = list(joint_steps)
    predictions = (
        [None] * len(requested)
        if predicted_reductions is None
        else list(predicted_reductions)
    )
    roles = (
        ["retained_joint_candidate"] * len(requested)
        if candidate_roles is None
        else [str(value) for value in candidate_roles]
    )
    payload: dict[str, Any] = {
        "schema": "sr_atomic_retained_joint_step_guard_v1",
        "status": "unavailable",
        "reason": "missing_retained_joint_candidates",
        "attempted": bool(requested),
        "atomic_candidate_set_required": True,
        "retained_candidate_count": int(len(requested)),
        "candidate_evaluations": [],
        "fallback_to_incumbent": True,
        "selected_candidate_index": None,
        "mapped_seed_exact_gain": None,
        "mapped_seed_predicted_reduction": None,
        "transaction_failure_kind": None,
        "trust_action": "hold",
        "no_state_transition": True,
        "guard_objective_evals": 0,
        "guard_endpoint_state_evals": 0,
        "endpoint_distance_gate_required": bool(endpoint_gate_requested),
        "max_endpoint_distance": endpoint_distance_budget,
        "endpoint_distance_tolerance": endpoint_distance_tolerance,
    }
    if not requested:
        return incumbent, payload, 0
    if len(predictions) != len(requested) or len(roles) != len(requested):
        return incumbent, {
            **payload,
            "reason": "retained_candidate_metadata_count_mismatch",
            "transaction_failure_kind": "mapping_contract",
        }, 0

    mapped_records: list[dict[str, Any]] = []
    mapped_points: list[np.ndarray] = []
    mapping_failed = False
    for index, raw_step in enumerate(requested):
        step = _sequence_of_floats(raw_step)
        record: dict[str, Any] = {
            "candidate_index": int(index),
            "candidate_role": str(roles[index]),
            "joint_step": None if step is None else list(step),
            "predicted_reduction": _finite_float(predictions[index]),
            "status": "unavailable",
            "reason": "invalid_joint_step",
            "mapping": None,
            "mapped_optimizer_x": None,
            "energy": None,
            "mapped_seed_exact_gain": None,
            "exact_endpoint_distance": None,
            "endpoint_distance_within_budget": None,
        }
        if step is None:
            mapping_failed = True
            mapped_records.append(record)
            continue
        try:
            mapped_x, mapping = map_joint_step(step)
        except (TypeError, ValueError, IndexError, FloatingPointError) as exc:
            mapped_x = None
            mapping = {
                "status": "unavailable",
                "reason": "joint_step_mapping_exception",
                "exception": exc.__class__.__name__,
            }
        mapping_payload = dict(mapping)
        record["mapping"] = mapping_payload
        point = (
            None
            if mapped_x is None
            else np.asarray(mapped_x, dtype=float).reshape(-1).copy()
        )
        if (
            point is None
            or point.shape != incumbent.shape
            or not np.all(np.isfinite(point))
        ):
            mapping_failed = True
            record["reason"] = str(
                mapping_payload.get("reason", "joint_step_mapping_failed")
            )
            mapped_records.append(record)
            continue
        record.update(
            {
                "status": "mapped",
                "reason": "joint_step_mapped",
                "mapped_optimizer_x": [float(value) for value in point.tolist()],
            }
        )
        mapped_points.append(point)
        mapped_records.append(record)

    payload["candidate_evaluations"] = [dict(record) for record in mapped_records]
    if mapping_failed or len(mapped_points) != len(requested):
        return incumbent, {
            **payload,
            "reason": "atomic_retained_candidate_mapping_failed",
            "transaction_failure_kind": "mapping",
            "candidate_evaluations": [dict(record) for record in mapped_records],
        }, 0

    if precomputed_incumbent_energy is None:
        try:
            incumbent_energy_value = float(objective(incumbent))
        except (TypeError, ValueError, FloatingPointError):
            incumbent_energy_value = float("nan")
        incumbent_nfev = 1
        incumbent_energy_source = "guard_objective_evaluation"
    else:
        incumbent_energy_value = float(precomputed_incumbent_energy)
        incumbent_nfev = 0
        incumbent_energy_source = "reused_prior_guard_evaluation"
    energies: list[float] = []
    for point in mapped_points:
        try:
            energies.append(float(objective(point)))
        except (TypeError, ValueError, FloatingPointError):
            energies.append(float("nan"))
    nfev = int(incumbent_nfev + len(mapped_points))
    if not math.isfinite(incumbent_energy_value) or any(
        not math.isfinite(value) for value in energies
    ):
        for record, energy in zip(mapped_records, energies):
            record.update(
                {
                    "status": "evaluated" if math.isfinite(energy) else "unavailable",
                    "reason": (
                        "exact_objective_evaluated"
                        if math.isfinite(energy)
                        else "nonfinite_objective"
                    ),
                    "energy": float(energy) if math.isfinite(energy) else None,
                }
            )
        return incumbent, {
            **payload,
            "reason": "atomic_retained_candidate_nonfinite_objective",
            "transaction_failure_kind": "nonfinite_path",
            "mapped_seed_incumbent_energy": (
                float(incumbent_energy_value)
                if math.isfinite(incumbent_energy_value)
                else None
            ),
            "incumbent_energy_source": str(incumbent_energy_source),
            "candidate_evaluations": [dict(record) for record in mapped_records],
            "guard_objective_evals": int(nfev),
        }, int(nfev)

    endpoint_distances: list[float] = []
    if endpoint_distance_from_incumbent is not None:
        for point in mapped_points:
            try:
                endpoint_distances.append(
                    float(endpoint_distance_from_incumbent(point))
                )
            except (TypeError, ValueError, RuntimeError, FloatingPointError):
                endpoint_distances.append(float("nan"))
        if any(not math.isfinite(value) or value < 0.0 for value in endpoint_distances):
            for record, energy, distance in zip(
                mapped_records,
                energies,
                endpoint_distances,
            ):
                record.update(
                    {
                        "status": "evaluated",
                        "reason": (
                            "exact_endpoint_distance_evaluated"
                            if math.isfinite(distance) and distance >= 0.0
                            else "nonfinite_endpoint_distance"
                        ),
                        "energy": float(energy),
                        "exact_endpoint_distance": (
                            float(distance)
                            if math.isfinite(distance) and distance >= 0.0
                            else None
                        ),
                    }
                )
            return incumbent, {
                **payload,
                "reason": "atomic_retained_candidate_nonfinite_endpoint_distance",
                "transaction_failure_kind": "state_consistency",
                "mapped_seed_incumbent_energy": float(incumbent_energy_value),
                "incumbent_energy_source": str(incumbent_energy_source),
                "candidate_evaluations": [dict(record) for record in mapped_records],
                "guard_objective_evals": int(nfev),
                "guard_endpoint_state_evals": int(len(endpoint_distances)),
            }, int(nfev)
    else:
        endpoint_distances = [0.0] * len(mapped_points)

    comparison_scale = max(
        1.0,
        abs(float(incumbent_energy_value)),
        *(abs(float(value)) for value in energies),
    )
    tolerance = float(abs_tol + rel_tol * comparison_scale)
    comparison_width = _energy_comparison_width_payload(
        numerical_width=tolerance,
        event_schema="sr_atomic_retained_joint_step_exact_guard_v1",
    )
    energy_downhill = [
        index
        for index, energy in enumerate(energies)
        if float(energy) < float(incumbent_energy_value) - tolerance
    ]
    downhill = [
        index
        for index in energy_downhill
        if endpoint_distance_budget is None
        or float(endpoint_distances[index])
        <= float(endpoint_distance_budget)
        + float(endpoint_distance_tolerance or 0.0)
    ]
    selected_index = (
        min(downhill, key=lambda index: (float(energies[index]), int(index)))
        if downhill
        else None
    )
    for record, energy, endpoint_distance in zip(
        mapped_records,
        energies,
        endpoint_distances,
    ):
        exact_gain = float(incumbent_energy_value - energy)
        distance_within_budget = bool(
            endpoint_distance_budget is None
            or float(endpoint_distance)
            <= float(endpoint_distance_budget)
            + float(endpoint_distance_tolerance or 0.0)
        )
        record.update(
            {
                "status": "evaluated",
                "reason": "exact_objective_evaluated",
                "energy": float(energy),
                "mapped_seed_exact_gain": float(exact_gain),
                "materially_downhill": bool(
                    float(energy) < float(incumbent_energy_value) - tolerance
                ),
                "exact_endpoint_distance": (
                    None
                    if endpoint_distance_budget is None
                    else float(endpoint_distance)
                ),
                "endpoint_distance_within_budget": (
                    None
                    if endpoint_distance_budget is None
                    else bool(distance_within_budget)
                ),
                "eligible_for_selection": bool(
                    float(energy) < float(incumbent_energy_value) - tolerance
                    and distance_within_budget
                ),
            }
        )
    if selected_index is None:
        any_energy_downhill = bool(energy_downhill)
        rejected_for_endpoint_distance = bool(
            endpoint_distance_budget is not None and any_energy_downhill
        )
        return incumbent, {
            **payload,
            "status": "rejected",
            "reason": (
                "all_materially_downhill_candidates_exceed_exact_endpoint_distance_budget"
                if rejected_for_endpoint_distance
                else "all_retained_candidates_certified_non_downhill"
            ),
            "mapped_seed_incumbent_energy": float(incumbent_energy_value),
            "incumbent_energy_source": str(incumbent_energy_source),
            "candidate_evaluations": [dict(record) for record in mapped_records],
            "comparison_tolerance": float(tolerance),
            "energy_comparison_width": dict(comparison_width),
            **{
                key: value
                for key, value in comparison_width.items()
                if key != "schema"
            },
            "all_mapped_candidates_finite": True,
            "all_mapped_candidates_non_downhill": bool(
                not any_energy_downhill
            ),
            "any_mapped_candidate_materially_downhill": bool(
                any_energy_downhill
            ),
            "all_materially_downhill_candidates_outside_endpoint_distance_budget": bool(
                rejected_for_endpoint_distance
            ),
            "sr_transaction_outcome": (
                "no_state_refinement_finite_endpoint_rejection"
                if rejected_for_endpoint_distance
                else "no_state_refinement_trust_hold"
            ),
            "guard_objective_evals": int(nfev),
            "guard_endpoint_state_evals": int(
                len(endpoint_distances) if endpoint_gate_requested else 0
            ),
        }, int(nfev)

    selected_record = mapped_records[int(selected_index)]
    selected_point = mapped_points[int(selected_index)]
    exact_gain = float(
        incumbent_energy_value - energies[int(selected_index)]
    )
    predicted = _finite_float(selected_record.get("predicted_reduction"))
    ratio = (
        None
        if predicted is None or predicted <= 0.0
        else float(exact_gain / predicted)
    )
    return selected_point, {
        **payload,
        "status": "accepted",
        "reason": "retained_candidate_exactly_downhill",
        "fallback_to_incumbent": False,
        "selected_candidate_index": int(selected_index),
        "selected_candidate_role": str(roles[int(selected_index)]),
        "mapped_seed_incumbent_energy": float(incumbent_energy_value),
        "incumbent_energy_source": str(incumbent_energy_source),
        "mapped_seed_proposal_energy": float(energies[int(selected_index)]),
        "mapped_seed_exact_gain": float(exact_gain),
        "mapped_seed_exact_endpoint_distance": (
            None
            if endpoint_distance_budget is None
            else float(endpoint_distances[int(selected_index)])
        ),
        "mapped_seed_predicted_reduction": predicted,
        "exact_to_prediction_ratio": ratio,
        "candidate_evaluations": [dict(record) for record in mapped_records],
        "comparison_tolerance": float(tolerance),
        "energy_comparison_width": dict(comparison_width),
        **{
            key: value
            for key, value in comparison_width.items()
            if key != "schema"
        },
        "all_mapped_candidates_finite": True,
        "all_mapped_candidates_non_downhill": False,
        "no_state_transition": False,
        "trust_action": "accepted_transition",
        "sr_transaction_outcome": "mapped_downhill_seed_ready",
        "guard_objective_evals": int(nfev),
        "guard_endpoint_state_evals": int(
            len(endpoint_distances) if endpoint_gate_requested else 0
        ),
    }, int(nfev)


def propose_exact_joint_step_seed(
    *,
    canonical_x0: np.ndarray,
    post_layout: AnsatzParameterLayout,
    reopt_runtime_active_indices: Sequence[int],
    pre_parameter_count: int,
    positions_in_commit_order: Sequence[int],
    selected_records: Sequence[Mapping[str, Any]],
    selector_summary: Mapping[str, Any] | None,
    optimizer_coordinate_mode: str = "runtime",
    zero_tolerance: float = 1.0e-12,
) -> tuple[SeedProposal | None, dict[str, Any]]:
    """Map the selector's applied ``(delta theta_A, alpha_B)`` into Powell x0."""

    telemetry: dict[str, Any] = {
        "schema": "route_a_joint_step_warm_start_v1",
        "status": "unavailable",
        "reason": "missing_joint_selector_summary",
        "step_source": "selector_applied_joint_solve_v1",
        "prediction_authority": "initialization_only_full_refit_authoritative",
        "objective_guard_required": True,
        "legacy_diagonal_geometry_used": False,
    }
    if not isinstance(selector_summary, Mapping):
        return None, telemetry
    active_step = _sequence_of_floats(
        selector_summary.get("active_parameter_relaxation")
    )
    batch_step = _sequence_of_floats(selector_summary.get("batch_coordinate_step"))
    geometry_workspace = selector_summary.get("geometry_workspace")
    active_pre_indices = (
        _sequence_of_ints(geometry_workspace.get("active_indices"))
        if isinstance(geometry_workspace, Mapping)
        else None
    )
    selected_labels_raw = selector_summary.get("selected_labels")
    selector_predicted_gain = _finite_float(
        selector_summary.get(
            "applied_predicted_reduction",
            selector_summary.get("joint_gain"),
        )
    )
    selected_labels = (
        [str(value) for value in selected_labels_raw]
        if isinstance(selected_labels_raw, Sequence)
        and not isinstance(selected_labels_raw, (str, bytes, bytearray))
        else None
    )
    if active_step is None or batch_step is None or active_pre_indices is None:
        return None, {
            **telemetry,
            "reason": "missing_applied_joint_step",
        }
    record_labels = [_record_label(record) for record in selected_records]
    if len(batch_step) != len(selected_records) or len(positions_in_commit_order) != len(
        selected_records
    ):
        return None, {**telemetry, "reason": "batch_coordinate_count_mismatch"}
    if len(active_step) != len(active_pre_indices):
        return None, {**telemetry, "reason": "active_coordinate_count_mismatch"}
    if selected_labels is not None and selected_labels != record_labels:
        return None, {
            **telemetry,
            "reason": "selected_record_order_mismatch",
            "selector_labels": selected_labels,
            "admission_labels": record_labels,
        }
    try:
        batch_window = build_composed_batch_window_payload(
            pre_parameter_count=int(pre_parameter_count),
            positions_in_commit_order=[int(value) for value in positions_in_commit_order],
            old_pre_indices=[int(value) for value in active_pre_indices],
        )
    except (NestedWindowError, TypeError, ValueError) as exc:
        return None, {
            **telemetry,
            "reason": "invalid_insertion_mapping",
            "exception": exc.__class__.__name__,
        }
    if int(post_layout.logical_parameter_count) != int(
        batch_window["post_parameter_count"]
    ):
        return None, {**telemetry, "reason": "post_layout_count_mismatch"}

    x0 = np.asarray(canonical_x0, dtype=float).reshape(-1).copy()
    coordinate_mode = str(optimizer_coordinate_mode).strip().lower()
    if coordinate_mode not in {"runtime", "logical_shared"}:
        return None, {
            **telemetry,
            "reason": "unsupported_optimizer_coordinate_mode",
            "optimizer_coordinate_mode": coordinate_mode,
        }
    optimizer_to_reduced = {
        int(coordinate_index): int(reduced_index)
        for reduced_index, coordinate_index in enumerate(
            reopt_runtime_active_indices
        )
    }

    def reduced_positions(logical_index: int) -> list[int] | None:
        if logical_index < 0 or logical_index >= int(post_layout.logical_parameter_count):
            return None
        if coordinate_mode == "logical_shared":
            mapped_logical = optimizer_to_reduced.get(int(logical_index))
            return None if mapped_logical is None else [int(mapped_logical)]
        block = post_layout.blocks[int(logical_index)]
        runtime_positions = range(int(block.runtime_start), int(block.runtime_stop))
        mapped = [optimizer_to_reduced.get(int(index)) for index in runtime_positions]
        if not mapped or any(value is None for value in mapped):
            return None
        return [int(value) for value in mapped if value is not None]

    old_post_indices = [int(value) for value in batch_window["old_post_indices"]]
    candidate_post_indices = [
        int(value) for value in batch_window["candidate_post_indices"]
    ]
    active_position_groups = [reduced_positions(index) for index in old_post_indices]
    candidate_position_groups = [
        reduced_positions(index) for index in candidate_post_indices
    ]
    if any(group is None for group in [*active_position_groups, *candidate_position_groups]):
        return None, {
            **telemetry,
            "reason": "optimizer_window_missing_joint_coordinate",
            "batch_window": dict(batch_window),
        }
    candidate_reduced_positions = [
        int(position)
        for group in candidate_position_groups
        if group is not None
        for position in group
    ]
    if any(abs(float(x0[position])) > float(zero_tolerance) for position in candidate_reduced_positions):
        return None, {
            **telemetry,
            "reason": "candidate_not_initialized_at_zero",
            "candidate_seed_values": [
                float(x0[position]) for position in candidate_reduced_positions
            ],
        }
    proposal_x0 = np.asarray(x0, dtype=float).copy()
    for delta, group in zip(active_step, active_position_groups):
        assert group is not None
        for reduced_position in group:
            proposal_x0[int(reduced_position)] += float(delta)
    for alpha, group in zip(batch_step, candidate_position_groups):
        assert group is not None
        for reduced_position in group:
            proposal_x0[int(reduced_position)] += float(alpha)
    if not np.all(np.isfinite(proposal_x0)):
        return None, {**telemetry, "reason": "nonfinite_mapped_joint_step"}

    optimizer_delta = np.asarray(proposal_x0 - x0, dtype=float)
    optimizer_delta_l2 = float(np.linalg.norm(optimizer_delta))
    runtime_delta_l2 = float(optimizer_delta_l2)
    if coordinate_mode == "logical_shared":
        runtime_delta_squared = 0.0
        logical_groups = [
            *(zip(old_post_indices, active_position_groups)),
            *(zip(candidate_post_indices, candidate_position_groups)),
        ]
        for logical_index, group in logical_groups:
            assert group is not None
            if len(group) != 1:
                return None, {
                    **telemetry,
                    "reason": "logical_shared_coordinate_group_not_singleton",
                    "logical_index": int(logical_index),
                    "reduced_position_group": [int(value) for value in group],
                }
            delta_value = float(optimizer_delta[int(group[0])])
            runtime_count = int(
                post_layout.blocks[int(logical_index)].runtime_count
            )
            runtime_delta_squared += float(runtime_count) * delta_value * delta_value
        runtime_delta_l2 = float(math.sqrt(max(0.0, runtime_delta_squared)))

    mapped_payload = {
        **telemetry,
        "status": "available",
        "reason": "exact_applied_joint_step_mapped",
        "batch_window": dict(batch_window),
        "active_pre_logical_indices": [int(value) for value in active_pre_indices],
        "active_post_logical_indices": old_post_indices,
        "candidate_post_logical_indices": candidate_post_indices,
        "active_reduced_position_groups": active_position_groups,
        "candidate_reduced_position_groups": candidate_position_groups,
        "active_parameter_relaxation": [float(value) for value in active_step],
        "batch_coordinate_step": [float(value) for value in batch_step],
        "selected_labels": record_labels,
        "selector_applied_predicted_reduction": selector_predicted_gain,
        "optimizer_coordinate_mode": coordinate_mode,
        "optimizer_coordinate_delta_l2": float(optimizer_delta_l2),
        "runtime_delta_l2": float(runtime_delta_l2),
    }
    return (
        SeedProposal(
            name="route_a_exact_applied_joint_step_v1",
            x0=proposal_x0,
            telemetry=dict(mapped_payload),
        ),
        mapped_payload,
    )


def guard_exact_joint_step_seed(
    *,
    objective: Callable[[np.ndarray], float],
    canonical_x0: np.ndarray,
    config: RouteAJointStepWarmStartConfig,
    post_layout: AnsatzParameterLayout,
    reopt_runtime_active_indices: Sequence[int],
    pre_parameter_count: int,
    positions_in_commit_order: Sequence[int],
    selected_records: Sequence[Mapping[str, Any]],
    selector_summary: Mapping[str, Any] | None,
    optimizer_coordinate_mode: str = "runtime",
) -> tuple[np.ndarray, dict[str, Any], int]:
    """Build and objective-guard the exact applied joint-selector seed."""

    incumbent = np.asarray(canonical_x0, dtype=float).reshape(-1).copy()
    payload: dict[str, Any] = {
        "schema": "route_a_joint_step_warm_start_v1",
        "enabled": bool(config.enabled),
        "mode": str(config.mode),
        "attempted": False,
        "status": "disabled",
        "reason": "policy_off",
        "guard_objective_evals": 0,
        "fallback_to_incumbent": True,
        "physical_transition_certified": False,
    }
    if not config.enabled:
        return incumbent, payload, 0
    proposal, proposal_telemetry = propose_exact_joint_step_seed(
        canonical_x0=incumbent,
        post_layout=post_layout,
        reopt_runtime_active_indices=reopt_runtime_active_indices,
        pre_parameter_count=int(pre_parameter_count),
        positions_in_commit_order=positions_in_commit_order,
        selected_records=selected_records,
        selector_summary=selector_summary,
        optimizer_coordinate_mode=str(optimizer_coordinate_mode),
    )
    payload.update(dict(proposal_telemetry))
    payload.update(
        {
            "enabled": True,
            "mode": str(config.mode),
            "attempted": True,
        }
    )
    if proposal is None:
        payload.update(
            {
                "transaction_failure_kind": "mapping",
                "trust_action": "hold",
                "sr_transaction_outcome": "no_state_refinement_trust_hold",
            }
        )
        return incumbent, payload, 0
    result = select_guarded_seed(
        objective=objective,
        incumbent_x0=incumbent,
        proposals=[proposal],
        guard_abs_tol=float(config.guard_abs_tol),
        guard_rel_tol=float(config.guard_rel_tol),
    )
    nfev = int(result.telemetry.get("guard_objective_evals", 0))
    guard_evaluations = result.telemetry.get("evaluations")
    proposal_energy: float | None = None
    if isinstance(guard_evaluations, Sequence) and not isinstance(
        guard_evaluations, (str, bytes, bytearray)
    ):
        for evaluation in guard_evaluations:
            if not isinstance(evaluation, Mapping):
                continue
            if str(evaluation.get("name", "")) != str(proposal.name):
                continue
            proposal_energy = _finite_float(evaluation.get("energy"))
            if proposal_energy is not None:
                break
    incumbent_energy = _finite_float(result.telemetry.get("incumbent_energy"))
    mapped_seed_exact_gain = (
        None
        if incumbent_energy is None or proposal_energy is None
        else float(incumbent_energy - proposal_energy)
    )
    selector_predicted_gain = _finite_float(
        proposal_telemetry.get("selector_applied_predicted_reduction")
    )
    prediction_to_exact_seed_ratio = (
        None
        if selector_predicted_gain is None
        or mapped_seed_exact_gain is None
        or mapped_seed_exact_gain <= 0.0
        else float(selector_predicted_gain / mapped_seed_exact_gain)
    )
    guard_tolerance = float(
        max(0.0, result.telemetry.get("guard_tolerance", 0.0))
    )
    comparison_width = _energy_comparison_width_payload(
        numerical_width=guard_tolerance,
        event_schema="route_a_exact_joint_step_seed_guard_v1",
    )
    payload.update(
        {
            "status": str(result.status),
            "reason": str(result.reason),
            "chosen_source": str(result.chosen_source),
            "guard_objective_evals": int(nfev),
            "fallback_to_incumbent": bool(result.status != "accepted"),
            "guard": dict(result.telemetry),
            "mapped_seed_incumbent_energy": incumbent_energy,
            "mapped_seed_proposal_energy": proposal_energy,
            "mapped_seed_exact_gain": mapped_seed_exact_gain,
            "prediction_to_exact_seed_ratio": prediction_to_exact_seed_ratio,
            "mapped_seed_predicted_reduction": selector_predicted_gain,
            "energy_comparison_width": dict(comparison_width),
            **{
                key: value
                for key, value in comparison_width.items()
                if key != "schema"
            },
            "transaction_failure_kind": (
                None
                if str(result.status) in {"accepted", "rejected"}
                else "nonfinite_path"
            ),
            "trust_action": (
                "accepted_transition"
                if str(result.status) == "accepted"
                else "hold"
            ),
            "sr_transaction_outcome": (
                "mapped_downhill_seed_ready"
                if str(result.status) == "accepted"
                else "no_state_refinement_trust_hold"
            ),
        }
    )
    return np.asarray(result.x0, dtype=float).copy(), payload, int(nfev)


def guard_supported_fs_full_joint_step_seed(
    *,
    objective: Callable[[np.ndarray], float],
    incumbent_energy: float,
    chart: SupportedFSPowellChart,
    config: RouteAJointStepWarmStartConfig,
    selected_records: Sequence[Mapping[str, Any]],
    selector_summary: Mapping[str, Any] | None,
    phase3_to_post_logical_permutation: Sequence[int],
) -> tuple[np.ndarray, dict[str, Any], int]:
    """Guard the complete Phase-III response in a fixed supported-FS chart."""

    incumbent = np.asarray(chart.x0, dtype=float).reshape(-1).copy()
    payload: dict[str, Any] = {
        "schema": "accepted_refit_joint_response_initialization_v1",
        "enabled": bool(config.enabled),
        "policy": str(config.mode),
        "attempted": False,
        "status": "disabled",
        "reason": "policy_off",
        "chosen_source": "inherited_zero_growth",
        "fallback_to_incumbent": True,
        "guard_objective_evals": 0,
        "incumbent_energy": float(incumbent_energy),
        "incumbent_energy_source": (
            "accepted_pre_refit_endpoint_reuse_v1"
        ),
        "selection_mutated": False,
        "prediction_authority": (
            "initialization_only_full_powell_refit_authoritative_v1"
        ),
    }
    if not config.enabled:
        return incumbent, payload, 0
    if not isinstance(selector_summary, Mapping):
        return incumbent, {
            **payload,
            "attempted": True,
            "status": "unavailable",
            "reason": "missing_joint_selector_summary",
        }, 0
    active_step = _sequence_of_floats(
        selector_summary.get("active_parameter_relaxation")
    )
    batch_step = _sequence_of_floats(
        selector_summary.get("batch_coordinate_step")
    )
    joint_step = _sequence_of_floats(selector_summary.get("joint_step"))
    if active_step is None or batch_step is None or joint_step is None:
        return incumbent, {
            **payload,
            "attempted": True,
            "status": "unavailable",
            "reason": "missing_applied_full_joint_step",
        }, 0
    concatenated = [*active_step, *batch_step]
    if len(joint_step) != len(concatenated) or not np.allclose(
        np.asarray(joint_step, dtype=float),
        np.asarray(concatenated, dtype=float),
        rtol=0.0,
        atol=2.0e-12,
    ):
        return incumbent, {
            **payload,
            "attempted": True,
            "status": "unavailable",
            "reason": "joint_step_component_mismatch",
        }, 0
    record_labels = [_record_label(record) for record in selected_records]
    selected_labels_raw = selector_summary.get("selected_labels")
    selected_labels = (
        [str(value) for value in selected_labels_raw]
        if isinstance(selected_labels_raw, Sequence)
        and not isinstance(
            selected_labels_raw, (str, bytes, bytearray)
        )
        else None
    )
    if (
        len(batch_step) != len(record_labels)
        or selected_labels is None
        or selected_labels != record_labels
    ):
        return incumbent, {
            **payload,
            "attempted": True,
            "status": "unavailable",
            "reason": "selected_record_order_mismatch",
            "selector_labels": selected_labels,
            "admission_labels": record_labels,
        }, 0
    predicted_reduction = _finite_float(
        selector_summary.get(
            "applied_predicted_reduction",
            selector_summary.get("predicted_reduction"),
        )
    )
    candidate_gain_receipt = selector_summary.get(
        "phase3_candidate_gain_receipt"
    )
    try:
        mapped_x0, mapping_receipt = (
            map_phase_order_joint_step_to_supported_fs(
                chart=chart,
                phase_order_joint_step=np.asarray(
                    joint_step, dtype=float
                ),
                phase3_to_post_logical_permutation=(
                    phase3_to_post_logical_permutation
                ),
            )
        )
    except (TypeError, ValueError, IndexError) as exc:
        return incumbent, {
            **payload,
            "attempted": True,
            "status": "unavailable",
            "reason": "supported_fs_joint_step_mapping_failed",
            "mapping_exception": exc.__class__.__name__,
            "mapped_seed_predicted_full_joint_reduction": (
                predicted_reduction
            ),
            "phase3_candidate_gain_receipt": (
                dict(candidate_gain_receipt)
                if isinstance(candidate_gain_receipt, Mapping)
                else None
            ),
        }, 0
    if not bool(
        mapping_receipt.get("source_step_within_supported_chart", False)
    ):
        return incumbent, {
            **payload,
            "attempted": True,
            "status": "unavailable",
            "reason": "joint_step_outside_refit_supported_chart",
            "supported_fs_mapping": dict(mapping_receipt),
            "mapped_seed_predicted_full_joint_reduction": (
                predicted_reduction
            ),
            "phase3_candidate_gain_receipt": (
                dict(candidate_gain_receipt)
                if isinstance(candidate_gain_receipt, Mapping)
                else None
            ),
        }, 0
    proposal = SeedProposal(
        name="phase3_full_joint_response_supported_fs_v1",
        x0=np.asarray(mapped_x0, dtype=float),
        telemetry={
            "mapping": dict(mapping_receipt),
            "predicted_full_joint_reduction": predicted_reduction,
            "phase3_candidate_gain_receipt": (
                dict(candidate_gain_receipt)
                if isinstance(candidate_gain_receipt, Mapping)
                else None
            ),
        },
    )
    result = select_guarded_seed(
        objective=objective,
        incumbent_x0=incumbent,
        proposals=[proposal],
        guard_abs_tol=float(config.guard_abs_tol),
        guard_rel_tol=float(config.guard_rel_tol),
        incumbent_energy=float(incumbent_energy),
        max_objective_evals=1,
    )
    nfev = int(result.telemetry.get("guard_objective_evals", 0))
    proposal_energy: float | None = None
    evaluations = result.telemetry.get("evaluations")
    if isinstance(evaluations, Sequence) and not isinstance(
        evaluations, (str, bytes, bytearray)
    ):
        for evaluation in evaluations:
            if not isinstance(evaluation, Mapping):
                continue
            if str(evaluation.get("name", "")) != proposal.name:
                continue
            proposal_energy = _finite_float(evaluation.get("energy"))
            break
    exact_gain = (
        None
        if proposal_energy is None
        else float(float(incumbent_energy) - proposal_energy)
    )
    return np.asarray(result.x0, dtype=float).copy(), {
        **payload,
        "attempted": True,
        "status": str(result.status),
        "reason": str(result.reason),
        "chosen_source": str(result.chosen_source),
        "fallback_to_incumbent": bool(result.status != "accepted"),
        "guard_objective_evals": int(nfev),
        "mapped_seed_proposal_energy": proposal_energy,
        "mapped_seed_exact_gain": exact_gain,
        "mapped_seed_predicted_full_joint_reduction": (
            predicted_reduction
        ),
        "phase3_candidate_gain_receipt": (
            dict(candidate_gain_receipt)
            if isinstance(candidate_gain_receipt, Mapping)
            else None
        ),
        "supported_fs_mapping": dict(mapping_receipt),
        "guard": dict(result.telemetry),
    }, int(nfev)


def guard_exact_joint_step_sign_candidates(
    *,
    objective: Callable[[np.ndarray], float],
    canonical_x0: np.ndarray,
    config: RouteAJointStepWarmStartConfig,
    post_layout: AnsatzParameterLayout,
    reopt_runtime_active_indices: Sequence[int],
    pre_parameter_count: int,
    positions_in_commit_order: Sequence[int],
    selected_records: Sequence[Mapping[str, Any]],
    selector_summary: Mapping[str, Any] | None,
    joint_steps: Sequence[Sequence[float]] | None = None,
    optimizer_coordinate_mode: str = "runtime",
) -> tuple[np.ndarray, dict[str, Any], int]:
    """Map and guard both exact hard-case signs as one atomic action.

    The joint solver stores hard-case candidates in the concatenated
    ``(delta theta_A, alpha_B)`` coordinate order.  This helper derives the
    active/batch split from the applied selector summary, maps every sign with
    :func:`propose_exact_joint_step_seed`, and asks the existing objective guard
    to choose the lowest exact-energy sign.  A missing or unmappable sign makes
    the whole pair unavailable: accepting only half of a hard-case pair would
    manufacture a sign preference from an implementation failure.

    The returned vector is a protected optimizer seed.  The caller must retain
    it in the post-optimizer outcome set so a worse Powell result cannot erase
    a mapped downhill state.
    """

    incumbent = np.asarray(canonical_x0, dtype=float).reshape(-1).copy()
    payload: dict[str, Any] = {
        "schema": "route_a_joint_step_sign_guard_v1",
        "enabled": bool(config.enabled),
        "mode": str(config.mode),
        "attempted": False,
        "status": "disabled",
        "reason": "policy_off",
        "guard_objective_evals": 0,
        "fallback_to_incumbent": True,
        "atomic_sign_pair_required": True,
        "step_source": "selector_hard_case_sign_candidates_joint_v1",
        "prediction_authority": "initialization_only_full_refit_authoritative",
        "post_powell_role": "initialization_only_seed_preserving_guard",
        "objective_guard_required": True,
        "sign_evaluations": [],
        "selected_sign": None,
        "mapped_seed_exact_gain": None,
        "prediction_to_exact_seed_ratio": None,
        "physical_transition_certified": False,
    }
    if not config.enabled:
        return incumbent, payload, 0
    payload.update({"enabled": True, "attempted": True})
    if not isinstance(selector_summary, Mapping):
        return incumbent, {
            **payload,
            "status": "unavailable",
            "reason": "missing_joint_selector_summary",
        }, 0

    applied_active = _sequence_of_floats(
        selector_summary.get("active_parameter_relaxation")
    )
    applied_batch = _sequence_of_floats(
        selector_summary.get("batch_coordinate_step")
    )
    if applied_active is None or applied_batch is None:
        return incumbent, {
            **payload,
            "status": "unavailable",
            "reason": "missing_joint_step_partition",
        }, 0
    active_count = int(len(applied_active))
    batch_count = int(len(applied_batch))
    joint_count = int(active_count + batch_count)
    payload.update(
        {
            "active_coordinate_count": active_count,
            "batch_coordinate_count": batch_count,
            "joint_coordinate_count": joint_count,
        }
    )

    source_raw = selector_summary.get("hard_case_sign_candidates_joint")
    source_candidates_raw = (
        list(source_raw)
        if isinstance(source_raw, Sequence)
        and not isinstance(source_raw, (str, bytes, bytearray))
        else []
    )
    requested_raw = (
        list(joint_steps) if joint_steps is not None else source_candidates_raw
    )
    if len(requested_raw) != 2:
        return incumbent, {
            **payload,
            "status": "unavailable",
            "reason": "hard_case_sign_candidate_count_mismatch",
            "hard_case_sign_candidate_count": int(len(requested_raw)),
        }, 0

    source_candidates = [_sequence_of_floats(value) for value in source_candidates_raw]
    source_predictions_raw = selector_summary.get(
        "hard_case_sign_candidate_predicted_reductions"
    )
    source_predictions = (
        [_finite_float(value) for value in source_predictions_raw]
        if isinstance(source_predictions_raw, Sequence)
        and not isinstance(source_predictions_raw, (str, bytes, bytearray))
        else []
    )
    applied_joint = np.asarray([*applied_active, *applied_batch], dtype=float)
    selected_sign_raw = selector_summary.get("hard_case_selected_sign")
    try:
        selected_sign_hint = int(selected_sign_raw)
    except (TypeError, ValueError):
        selected_sign_hint = 0
    if selected_sign_hint not in {-1, 1}:
        selected_sign_hint = 0

    # The solver's canonical source order is (+, -).  When the summary carries
    # its selected sign, match it to the applied joint step so reordered input
    # sequences retain physical sign identity.
    source_signs = [1 if index == 0 else -1 for index in range(len(source_candidates))]
    if selected_sign_hint and len(source_candidates) == 2:
        applied_matches = [
            bool(
                candidate is not None
                and len(candidate) == joint_count
                and np.allclose(
                    np.asarray(candidate, dtype=float),
                    applied_joint,
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )
            )
            for candidate in source_candidates
        ]
        if sum(applied_matches) == 1:
            applied_index = int(applied_matches.index(True))
            source_signs[applied_index] = int(selected_sign_hint)
            source_signs[1 - applied_index] = int(-selected_sign_hint)

    requested_candidates = [_sequence_of_floats(value) for value in requested_raw]
    explicit_applied_index: int | None = None
    if len(source_candidates) == 0 and selected_sign_hint:
        explicit_applied_matches = [
            bool(
                candidate is not None
                and len(candidate) == joint_count
                and np.allclose(
                    np.asarray(candidate, dtype=float),
                    applied_joint,
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )
            )
            for candidate in requested_candidates
        ]
        if sum(explicit_applied_matches) == 1:
            explicit_applied_index = int(explicit_applied_matches.index(True))
    used_source_indices: set[int] = set()
    candidate_records: list[dict[str, Any]] = []
    mapping_failed = False
    for input_index, joint_step in enumerate(requested_candidates):
        record: dict[str, Any] = {
            "input_index": int(input_index),
            "status": "unavailable",
            "reason": "invalid_joint_step",
            "sign": None,
            "joint_step": None if joint_step is None else list(joint_step),
            "active_parameter_relaxation": None,
            "batch_coordinate_step": None,
            "mapped_optimizer_delta": None,
            "mapped_seed": None,
            "energy": None,
            "mapped_seed_exact_gain": None,
            "predicted_reduction": None,
            "prediction_to_exact_seed_ratio": None,
        }
        if joint_step is None or len(joint_step) != joint_count:
            mapping_failed = True
            record["reason"] = "joint_step_coordinate_count_mismatch"
            candidate_records.append(record)
            continue

        source_index: int | None = None
        for candidate_index, source_candidate in enumerate(source_candidates):
            if candidate_index in used_source_indices or source_candidate is None:
                continue
            if len(source_candidate) != len(joint_step):
                continue
            if np.allclose(
                np.asarray(source_candidate, dtype=float),
                np.asarray(joint_step, dtype=float),
                rtol=1.0e-12,
                atol=1.0e-12,
            ):
                source_index = int(candidate_index)
                used_source_indices.add(source_index)
                break
        if source_index is not None and source_index < len(source_signs):
            sign = int(source_signs[source_index])
        elif len(source_candidates) == 0:
            # An explicit pair may be supplied without stored candidates.  The
            # applied candidate still carries the selector's selected sign.
            if explicit_applied_index is not None:
                sign = (
                    int(selected_sign_hint)
                    if input_index == explicit_applied_index
                    else int(-selected_sign_hint)
                )
            else:
                sign = 1 if input_index == 0 else -1
        else:
            mapping_failed = True
            record["reason"] = "joint_step_not_in_selector_sign_pair"
            candidate_records.append(record)
            continue
        predicted_reduction = (
            source_predictions[source_index]
            if source_index is not None and source_index < len(source_predictions)
            else None
        )
        active_step = [float(value) for value in joint_step[:active_count]]
        batch_step = [float(value) for value in joint_step[active_count:]]
        sign_summary = dict(selector_summary)
        sign_summary["active_parameter_relaxation"] = active_step
        sign_summary["batch_coordinate_step"] = batch_step
        sign_summary.pop("applied_predicted_reduction", None)
        sign_summary.pop("joint_gain", None)
        if predicted_reduction is not None:
            sign_summary["applied_predicted_reduction"] = float(
                predicted_reduction
            )
        proposal, proposal_telemetry = propose_exact_joint_step_seed(
            canonical_x0=incumbent,
            post_layout=post_layout,
            reopt_runtime_active_indices=reopt_runtime_active_indices,
            pre_parameter_count=int(pre_parameter_count),
            positions_in_commit_order=positions_in_commit_order,
            selected_records=selected_records,
            selector_summary=sign_summary,
            optimizer_coordinate_mode=str(optimizer_coordinate_mode),
        )
        record.update(
            {
                "sign": int(sign),
                "active_parameter_relaxation": active_step,
                "batch_coordinate_step": batch_step,
                "predicted_reduction": predicted_reduction,
                "mapping": dict(proposal_telemetry),
            }
        )
        if proposal is None:
            mapping_failed = True
            record["reason"] = str(proposal_telemetry.get("reason", "mapping_failed"))
            candidate_records.append(record)
            continue
        proposal_x0 = np.asarray(proposal.x0, dtype=float).reshape(-1).copy()
        proposal_name = (
            "route_a_exact_hard_case_sign_plus_v1"
            if int(sign) > 0
            else "route_a_exact_hard_case_sign_minus_v1"
        )
        record.update(
            {
                "status": "mapped",
                "reason": "exact_joint_step_mapped",
                "mapped_optimizer_delta": [
                    float(value) for value in (proposal_x0 - incumbent).tolist()
                ],
                "mapped_seed": [float(value) for value in proposal_x0.tolist()],
                "proposal_name": proposal_name,
            }
        )
        record["proposal"] = SeedProposal(
            name=proposal_name,
            x0=proposal_x0,
            telemetry={
                **dict(proposal.telemetry),
                "hard_case_sign": int(sign),
                "hard_case_joint_step": [float(value) for value in joint_step],
                "predicted_reduction": predicted_reduction,
            },
        )
        candidate_records.append(record)

    def public_record(record: Mapping[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in record.items() if key != "proposal"}

    payload["sign_evaluations"] = [
        public_record(record) for record in candidate_records
    ]
    if mapping_failed or len(candidate_records) != 2 or any(
        "proposal" not in record for record in candidate_records
    ):
        return incumbent, {
            **payload,
            "status": "unavailable",
            "reason": "atomic_sign_pair_mapping_failed",
            "fallback_to_incumbent": True,
            "transaction_failure_kind": "mapping",
            "trust_action": "hold",
            "sr_saddle_transaction_outcome": "no_state_refinement_trust_hold",
        }, 0

    # Stable sign ordering makes equal-energy tie behavior independent of input
    # order without claiming a physical preference between symmetric signs.
    candidate_records.sort(
        key=lambda record: (
            0 if int(record["sign"]) > 0 else 1,
            tuple(float(value) for value in record["joint_step"]),
        )
    )
    proposals = [record["proposal"] for record in candidate_records]
    result = select_guarded_seed(
        objective=objective,
        incumbent_x0=incumbent,
        proposals=proposals,
        guard_abs_tol=float(config.guard_abs_tol),
        guard_rel_tol=float(config.guard_rel_tol),
    )
    nfev = int(result.telemetry.get("guard_objective_evals", 0))
    incumbent_energy = _finite_float(result.telemetry.get("incumbent_energy"))
    evaluations = result.telemetry.get("evaluations")
    energy_by_name: dict[str, float] = {}
    if isinstance(evaluations, Sequence) and not isinstance(
        evaluations, (str, bytes, bytearray)
    ):
        for evaluation in evaluations:
            if not isinstance(evaluation, Mapping):
                continue
            energy = _finite_float(evaluation.get("energy"))
            if energy is not None:
                energy_by_name[str(evaluation.get("name", ""))] = energy

    for record in candidate_records:
        energy = energy_by_name.get(str(record["proposal_name"]))
        exact_gain = (
            None
            if incumbent_energy is None or energy is None
            else float(incumbent_energy - energy)
        )
        predicted_reduction = _finite_float(record.get("predicted_reduction"))
        ratio = (
            None
            if predicted_reduction is None
            or exact_gain is None
            or exact_gain <= 0.0
            else float(predicted_reduction / exact_gain)
        )
        record.update(
            {
                "status": "evaluated" if energy is not None else "evaluation_failed",
                "reason": (
                    "exact_objective_evaluated"
                    if energy is not None
                    else "nonfinite_objective"
                ),
                "energy": energy,
                "mapped_seed_exact_gain": exact_gain,
                "prediction_to_exact_seed_ratio": ratio,
            }
        )

    finite_records = [
        record for record in candidate_records if record.get("energy") is not None
    ]
    if len(finite_records) != 2:
        return incumbent, {
            **payload,
            "status": "unavailable",
            "reason": "atomic_sign_pair_nonfinite_objective",
            "fallback_to_incumbent": True,
            "transaction_failure_kind": "nonfinite_path",
            "trust_action": "hold",
            "finite_mapped_sign_count": int(len(finite_records)),
            "all_mapped_signs_finite": False,
            "all_mapped_signs_non_downhill": False,
            "sign_evaluations": [
                public_record(record) for record in candidate_records
            ],
            "sr_saddle_transaction_outcome": "no_state_refinement_trust_hold",
        }, int(nfev)
    lowest_record = (
        min(
            finite_records,
            key=lambda record: (
                float(record["energy"]),
                0 if int(record["sign"]) > 0 else 1,
            ),
        )
        if finite_records
        else None
    )
    guard_tolerance = float(
        max(0.0, result.telemetry.get("guard_tolerance", 0.0))
    )
    comparison_width = _energy_comparison_width_payload(
        numerical_width=guard_tolerance,
        event_schema="route_a_exact_joint_step_sign_pair_guard_v1",
    )
    aggregate_comparison_width = float(
        comparison_width["aggregate_simultaneous_comparison_width"]
    )
    all_finite_point_estimate_non_downhill = bool(
        len(finite_records) == 2
        and incumbent_energy is not None
        and all(
            float(record["energy"]) >= float(incumbent_energy)
            for record in finite_records
        )
    )
    for record in finite_records:
        exact_gain = _finite_float(record.get("mapped_seed_exact_gain"))
        exact_gain_upper_bound = (
            None
            if exact_gain is None
            else float(exact_gain + aggregate_comparison_width)
        )
        record["mapped_seed_exact_gain_upper_bound"] = exact_gain_upper_bound
        record["certified_non_downhill"] = bool(
            exact_gain_upper_bound is not None
            and exact_gain_upper_bound <= 0.0
        )
    all_finite_certified_non_downhill = bool(
        len(finite_records) == 2
        and all(
            bool(record.get("certified_non_downhill", False))
            for record in finite_records
        )
    )
    sign_pair_comparison_unresolved = bool(
        str(result.status) == "rejected"
        and len(finite_records) == 2
        and not all_finite_certified_non_downhill
    )
    taylor_certificate_raw = selector_summary.get(
        "saddle_taylor_contraction_certificate"
    )
    taylor_certificate = (
        copy.deepcopy(dict(taylor_certificate_raw))
        if isinstance(taylor_certificate_raw, Mapping)
        else {}
    )
    valid_taylor_certificate = bool(taylor_certificate.get("valid", False))
    payload.update(
        {
            "status": str(result.status),
            "reason": str(result.reason),
            "chosen_source": str(result.chosen_source),
            "guard_objective_evals": int(nfev),
            "fallback_to_incumbent": bool(result.status != "accepted"),
            "guard": dict(result.telemetry),
            "mapped_seed_incumbent_energy": incumbent_energy,
            "sign_evaluations": [
                public_record(record) for record in candidate_records
            ],
            "selected_sign": (
                None if lowest_record is None else int(lowest_record["sign"])
            ),
            "selected_sign_accepted": bool(result.status == "accepted"),
            "mapped_seed_proposal_energy": (
                None if lowest_record is None else float(lowest_record["energy"])
            ),
            "mapped_seed_exact_gain": (
                None
                if lowest_record is None
                else lowest_record["mapped_seed_exact_gain"]
            ),
            "prediction_to_exact_seed_ratio": (
                None
                if lowest_record is None
                else lowest_record["prediction_to_exact_seed_ratio"]
            ),
            "mapped_seed_predicted_reduction": (
                None
                if lowest_record is None
                else lowest_record.get("predicted_reduction")
            ),
            "finite_mapped_sign_count": int(len(finite_records)),
            "all_mapped_signs_finite": bool(len(finite_records) == 2),
            "all_mapped_signs_point_estimate_non_downhill": bool(
                all_finite_point_estimate_non_downhill
            ),
            "all_mapped_signs_non_downhill": bool(
                all_finite_certified_non_downhill
            ),
            "sign_pair_energy_comparison_resolved": bool(
                str(result.status) == "accepted"
                or all_finite_certified_non_downhill
            ),
            "sign_pair_energy_comparison_unresolved": bool(
                sign_pair_comparison_unresolved
            ),
            "transaction_failure_kind": (
                "comparison_unresolved"
                if sign_pair_comparison_unresolved
                else None
            ),
            "energy_comparison_width": dict(comparison_width),
            **{
                key: value
                for key, value in comparison_width.items()
                if key != "schema"
            },
            "sr_saddle_transaction_outcome": (
                "radius_contract_refinement_no_state_mutation"
                if (
                    str(result.status) == "rejected"
                    and all_finite_certified_non_downhill
                )
                else "mapped_downhill_seed_ready"
                if str(result.status) == "accepted"
                else "no_state_refinement_trust_hold"
            ),
            "saddle_taylor_contraction_certificate": taylor_certificate,
            "trust_action": (
                "accepted_transition"
                if str(result.status) == "accepted"
                else "contract_with_taylor_certificate"
                if (
                    str(result.status) == "rejected"
                    and all_finite_certified_non_downhill
                    and valid_taylor_certificate
                )
                else "contract_with_numerical_backtracking"
                if (
                    str(result.status) == "rejected"
                    and all_finite_certified_non_downhill
                )
                else "hold_for_comparison_refinement"
                if sign_pair_comparison_unresolved
                else "hold"
            ),
        }
    )
    return np.asarray(result.x0, dtype=float).copy(), payload, int(nfev)


def retain_seed_preserving_optimizer_outcome(
    *,
    mapped_seed_x0: np.ndarray,
    mapped_seed_energy: float,
    optimizer_x: np.ndarray,
    optimizer_energy: float,
    incumbent_energy: float,
    guard_abs_tol: float = 1.0e-12,
    guard_rel_tol: float = 1.0e-12,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Retain the safer exact state from a downhill seed and its optimizer.

    The mapped seed has already passed the exact objective guard.  Powell is a
    disposable continuation from that seed: it replaces the seed only when its
    returned point is finite and materially lower.  Ties and comparison-scale
    overlaps retain the seed, matching the seed-preserving transaction in the
    SR-SNAKE saddle controller.
    """

    seed_x = np.asarray(mapped_seed_x0, dtype=float).reshape(-1).copy()
    optimizer_value = np.asarray(optimizer_x, dtype=float).reshape(-1).copy()
    seed_energy = _finite_float(mapped_seed_energy)
    optimizer_energy_finite = _finite_float(optimizer_energy)
    incumbent_energy_finite = _finite_float(incumbent_energy)
    if (
        seed_energy is None
        or incumbent_energy_finite is None
        or not np.all(np.isfinite(seed_x))
    ):
        raise ValueError("Mapped downhill seed transaction requires finite inputs.")
    if seed_x.shape != optimizer_value.shape:
        optimizer_energy_finite = None
    optimizer_point_finite = bool(
        seed_x.shape == optimizer_value.shape
        and np.all(np.isfinite(optimizer_value))
    )
    if not optimizer_point_finite:
        optimizer_energy_finite = None

    comparison_scale = max(
        1.0,
        abs(float(seed_energy)),
        abs(float(optimizer_energy_finite))
        if optimizer_energy_finite is not None
        else 0.0,
    )
    comparison_tolerance = float(guard_abs_tol) + float(
        guard_rel_tol
    ) * comparison_scale
    seed_downhill_margin = float(incumbent_energy_finite) - float(seed_energy)
    if seed_downhill_margin <= float(comparison_tolerance):
        raise ValueError(
            "Mapped seed must be materially downhill before seed-preserving Powell."
        )

    optimizer_materially_lower = bool(
        optimizer_energy_finite is not None
        and float(optimizer_energy_finite)
        < float(seed_energy) - float(comparison_tolerance)
    )
    if optimizer_materially_lower:
        selected_x = optimizer_value
        selected_energy = float(optimizer_energy_finite)
        safe_source = "optimizer_result"
    else:
        selected_x = seed_x
        selected_energy = float(seed_energy)
        safe_source = "mapped_downhill_seed"

    payload = {
        "schema": "sr_saddle_seed_preserving_optimizer_outcome_v1",
        "mapped_seed_energy": float(seed_energy),
        "optimizer_energy": optimizer_energy_finite,
        "incumbent_energy": float(incumbent_energy_finite),
        "comparison_tolerance": float(comparison_tolerance),
        "mapped_seed_downhill_margin": float(seed_downhill_margin),
        "optimizer_point_finite": bool(optimizer_point_finite),
        "optimizer_materially_lower": bool(optimizer_materially_lower),
        "post_refit_safe_source": str(safe_source),
        "mapped_seed_retained": bool(not optimizer_materially_lower),
        "optimizer_result_discarded": bool(not optimizer_materially_lower),
        "selected_energy": float(selected_energy),
        "state_selection_rule": (
            "optimizer_only_if_materially_below_mapped_downhill_seed_v1"
        ),
    }
    return np.asarray(selected_x, dtype=float).copy(), float(selected_energy), payload


__all__ = [
    "ROUTE_A_JOINT_STEP_WARM_START_EXACT_GUARDED_V1",
    "ROUTE_A_JOINT_STEP_WARM_START_MODES",
    "ROUTE_A_JOINT_STEP_WARM_START_OFF",
    "RouteAJointStepWarmStartConfig",
    "certify_exact_physical_transition",
    "guard_exact_joint_step_seed",
    "guard_exact_joint_step_sign_candidates",
    "guard_supported_fs_full_joint_step_seed",
    "guard_atomic_joint_step_candidates",
    "propose_exact_joint_step_seed",
    "retain_seed_preserving_optimizer_outcome",
]
