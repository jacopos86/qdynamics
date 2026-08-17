"""Static ADAPT structural-resume helpers.

This module intentionally keeps resume loading, validation, digesting, and
secret-value guards out of ``adapt_pipeline.py``.  It is the first narrow slice
of a static ADAPT continuation route; it does not build IBM account
orchestration.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pipelines.scaffold.imported_artifact_resolution import ImportedArtifactResolution
from pipelines.scaffold.runtime_contract import ScaffoldRuntimeInput
from pipelines.scaffold.runtime_loader import (
    load_scaffold_runtime_input_from_payload,
)
from pipelines.scaffold.hh_continuation_types import PhaseControllerSnapshot
from pipelines.scaffold.hh_continuation_pruning import (
    AFFINE_DELETION_FS_TRUST_TRIAL_RECEIPT_V1,
)
from pipelines.static_adapt.geometry_fingerprints import (
    candidate_generator_fingerprint,
)
from pipelines.static_adapt.builders.primitive_pools import _polynomial_signature
from pipelines.static_adapt.estimator_call_ledger import (
    EstimatorCallKey,
    EstimatorCallLedger,
    projective_state_fingerprint,
)
from pipelines.static_adapt.historical_formal_manifold_provenance import (
    FORMAL_MANIFOLD_ROUTE,
    FORMAL_MANIFOLD_ROUTE_FAMILY,
    FORMAL_MANIFOLD_ROUTE_PROFILE_OFF,
    FORMAL_MANIFOLD_SR_SELECTOR_FAMILY,
    FormalManifoldRouteComposition as HistoricalFormalManifoldRouteComposition,
)
from pipelines.static_adapt.selector_measurement_proxy import (
    controller_proxy_from_history_rows,
)
from pipelines.static_adapt.sr_snake_escape_controller import (
    SR_COORDINATE_SOLVE_SCOPE_PHASE2_AND_PHASE3_V1,
    SR_POWELL_COORDINATE_CHART_AUTO,
    SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1,
    SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1,
    SR_POWELL_COORDINATE_CHART_POLICY_CHOICES,
    SR_ROUTE_FAMILY,
    SR_ROUTE_PROFILE_CONFORMANCE_CHOICES,
    SR_ROUTE_PROFILE_CONFORMANCE_UNPROMOTED_EXPLICIT_ABLATION,
    SR_ROUTE_PROFILE_DISABLED,
    SR_ROUTE_PROFILE_PHASE2_PHASE3_WHITENED,
    SR_ROUTE_PROFILE_REDUCED_POWELL,
    SR_ROUTE_PROFILE_SADDLE_ONLY,
    SR_ROUTE_PROFILE_SADDLE_PLUS_MODELED_MINIMUM,
)
from pipelines.static_adapt.sr_snake_route_profile import (
    HISTORICAL_SR_SNAKE_PHASE12_ENERGY_MODEL_SETTINGS,
    HISTORICAL_SR_SNAKE_RESPONSE_SCOPE_SETTINGS,
    PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1,
    SR_ROUTE_PROFILE_CANONICAL_V1,
    SR_ROUTE_PROFILE_CONVENTIONAL_V2,
    SR_ROUTE_PROFILE_CONVENTIONAL_V3,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1,
    SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1,
    SR_ROUTE_PROFILE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1,
    SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2,
    SR_ROUTE_PROFILE_CANDIDATE_V4,
    SR_ROUTE_PROFILE_REQUEST_OFF,
    canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract_sha256 as _active_verified_resume_contract_sha256,
    normalize_phase3_response_coordinate_scope,
    normalize_sr_route_profile_request,
    validate_sr_route_profile_contract,
)
from pipelines.static_adapt.sr_snake_phase12_policy import (
    PHASE1_ENERGY_MODEL_FIRST_ORDER_FS_TRUST_V1,
    PHASE1_SCORE_MODE_TRUST_REGION_V1,
    PHASE2_CHEAP_CURVATURE_PROXY_POLICY_OFF,
    PHASE2_CURVATURE_POLICY_MEASURED_REQUIRED_FAIL_CLOSED_V1,
    normalize_phase1_energy_model,
    normalize_phase1_score_mode_policy,
    normalize_phase2_cheap_curvature_proxy_policy,
    normalize_phase2_curvature_policy,
)
from src.quantum.ansatz_parameterization import (
    AnsatzParameterLayout,
    expand_legacy_logical_theta,
    serialize_layout,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.vqe_latex_python_pairs import (
    AnsatzTerm,
    PauliPolynomial,
    PauliTerm,
)


_SECRET_MARKER_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:qiskit[_-]?ibm[_-]?)?"
    r"(?:token|api[_-]?key|apikey|secret|password|credential)"
    r"(?:[^a-z0-9]|$)",
    re.IGNORECASE,
)

_STATIC_RESUME_ALLOWED_PROBLEMS = frozenset(
    {
        "hh",
        "molecular_vibronic_h2o_linear_fd",
    }
)

_OBSOLETE_ADMISSION_ROLLBACK_FIELDS = frozenset(
    {
        "adapt_rollback_mode",
        "adapt_rollback_tolerance",
        "structural_rollback",
        "depth_rollback",
        "zero_gain_duplicate_filter",
        "zero_gain_duplicate_guard",
        "duplicate_cooldown_policy",
        "pre_child_phase1_filter",
        "cooldown_excluded_record_count",
        "skipped_structural_rollback_rows",
    }
)

_MODELED_MINIMUM_EXECUTION_CHECKPOINT_FIELD = (
    "modeled_minimum_execution_checkpoint"
)


def _assert_no_unsupported_modeled_minimum_execution_checkpoint(
    payload: Mapping[str, Any],
    *,
    context: str,
) -> None:
    """Fail closed before a legacy resume path can discard Stage-B state.

    Modeled-minimum execution checkpoints carry separate incumbent and working
    states plus scheduler service state.  The legacy scaffold and preserved
    best-frontier loaders cannot round-trip that contract, so accepting one
    would silently resume only the incumbent/static-ADAPT portion.
    """

    field = _MODELED_MINIMUM_EXECUTION_CHECKPOINT_FIELD
    present_paths: list[str] = []
    if field in payload:
        present_paths.append(field)
    adapt = payload.get("adapt_vqe", None)
    if isinstance(adapt, Mapping) and field in adapt:
        present_paths.append(f"adapt_vqe.{field}")
    checkpoint = payload.get("checkpoint", None)
    if isinstance(checkpoint, Mapping) and field in checkpoint:
        present_paths.append(f"checkpoint.{field}")
    if present_paths:
        raise ValueError(
            f"{context} cannot consume a modeled-minimum execution checkpoint; "
            "the legacy/scaffold resume contract cannot preserve separate "
            "incumbent/working state and scheduler service state. Rejected "
            "field path(s): "
            + ", ".join(present_paths)
        )


def _drop_obsolete_admission_rollback_state(
    value: Any,
    *,
    path: tuple[str, ...] = (),
    removed_counts: dict[str, int] | None = None,
) -> tuple[Any, dict[str, int]]:
    counts = {} if removed_counts is None else removed_counts
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            remove = key in _OBSOLETE_ADMISSION_ROLLBACK_FIELDS
            remove = remove or (
                key == "suppressed_reason" and str(child) == "structural_rollback"
            )
            remove = remove or (
                key == "rollback"
                and path
                and path[-1] in {"final_full_refit", "resume_boundary_refit"}
            )
            if remove:
                counts[key] = int(counts.get(key, 0)) + 1
                continue
            cleaned_child, counts = _drop_obsolete_admission_rollback_state(
                child,
                path=(*path, key),
                removed_counts=counts,
            )
            cleaned[key] = cleaned_child
        return cleaned, counts
    if isinstance(value, list):
        cleaned_list: list[Any] = []
        for index, child in enumerate(value):
            cleaned_child, counts = _drop_obsolete_admission_rollback_state(
                child,
                path=(*path, str(index)),
                removed_counts=counts,
            )
            cleaned_list.append(cleaned_child)
        return cleaned_list, counts
    if isinstance(value, tuple):
        cleaned_tuple, counts = _drop_obsolete_admission_rollback_state(
            list(value),
            path=path,
            removed_counts=counts,
        )
        return tuple(cleaned_tuple), counts
    return value, counts


@dataclass(frozen=True)
class ResumeScaffoldSource:
    artifact_json: Path
    artifact_sha256: str
    payload: Mapping[str, Any]
    runtime_input: ScaffoldRuntimeInput
    import_summary: Mapping[str, Any]


@dataclass(frozen=True)
class ResumeMatchedScaffold:
    selected_ops: tuple[AnsatzTerm, ...]
    selected_layout: AnsatzParameterLayout
    theta_runtime: np.ndarray
    theta_logical: np.ndarray | None
    selected_pool_indices: tuple[int, ...]
    selected_generator_contracts: Mapping[str, Mapping[str, Any]]
    validation: Mapping[str, Any]


@dataclass(frozen=True)
class ResumeBestFrontierCheckpoint:
    """Strict, source-complete state for one preserved best beam branch."""

    history: tuple[Mapping[str, Any], ...]
    controller_round: int
    ansatz_depth: int
    branch_id: int
    parent_branch_id: int | None
    operator_labels: tuple[str, ...]
    theta_runtime: tuple[float, ...]
    theta_logical: tuple[float, ...]
    route_a_trust_region_state: Mapping[str, Any]
    beam_checkpoint_branch: Mapping[str, Any]
    frontier_prune_key: Mapping[str, Any]
    source_energy: float
    initial_state_digest: str
    ansatz_input_state_digest: str
    powell_coordinate_chart_policy: str | None
    route_profile_conformance: str | None
    sr_route_profile_request: str | None
    sr_route_profile_contract: Mapping[str, Any] | None
    sr_route_profile_contract_sha256: str | None
    phase3_response_coordinate_scope: str | None
    phase1_score_mode: str | None
    phase1_energy_model: str | None
    phase2_curvature_policy: str | None
    phase2_cheap_curvature_proxy_policy: str | None
    validation: Mapping[str, Any]


@dataclass(frozen=True)
class ResumeVerifiedSingletonCheckpoint:
    """Strict state for one complete beam-disabled singleton checkpoint."""

    history: tuple[Mapping[str, Any], ...]
    controller_round: int
    ansatz_depth: int
    branch_id: None
    parent_branch_id: None
    operator_labels: tuple[str, ...]
    theta_runtime: tuple[float, ...]
    theta_logical: tuple[float, ...]
    route_a_trust_region_state: Mapping[str, Any]
    controller_measurement_work_summary: Mapping[str, Any]
    phase1_residual_opened: bool
    phase1_stage_name: str
    maturity_controller_snapshot: Mapping[str, Any]
    maturity_controller_state_provenance: Mapping[str, Any]
    selection_parent_pool_size: int
    selected_parent_pool_indices: tuple[int, ...]
    selected_logical_candidate_indices: tuple[int, ...]
    selection_count_state_provenance: Mapping[str, Any]
    estimator_call_ledger_payload: Mapping[str, Any]
    estimator_call_ledger_provenance: Mapping[str, Any]
    source_energy: float
    strict_replay: Mapping[str, Any]
    initial_state_digest: str
    ansatz_input_state_digest: str
    powell_coordinate_chart_policy: str | None
    route_profile_conformance: str | None
    sr_route_profile_request: str | None
    sr_route_profile_contract_sha256: str | None
    phase3_response_coordinate_scope: str | None
    phase1_score_mode: str | None
    phase1_energy_model: str | None
    phase2_curvature_policy: str | None
    phase2_cheap_curvature_proxy_policy: str | None
    validation: Mapping[str, Any]


@dataclass(frozen=True)
class ResumeCompileSmokeResult:
    required: bool
    executed: bool
    success: bool
    backend_name: str | None
    compiled_depth: int | None
    compiled_size: int | None
    compiled_count_2q: int | None
    output_json: str | None
    error: str | None

    def to_payload(self) -> dict[str, Any]:
        return dict(asdict(self))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(x) for x in value.reshape(-1).tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"re": float(np.real(value)), "im": float(np.imag(value))}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def digest_jsonable(value: Any) -> str:
    """Return a stable SHA256 digest for a JSON-like value."""

    encoded = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_latest_v4_prune_trust_state(
    *,
    history: Sequence[Mapping[str, Any]],
    artifact_profile_request: str | None,
    context: str,
) -> dict[str, Any] | None:
    """Require the latest serialized v4 prune radius/mu/counter state.

    Fresh execution initializes from the registered profile before any history
    exists.  A structural resume necessarily has history and therefore must
    restore the latest state exactly; it may not search backward or substitute
    profile defaults when any field is absent.
    """

    if artifact_profile_request not in {
        SR_ROUTE_PROFILE_CANDIDATE_V4,
        SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1,
    }:
        return None
    if not history:
        raise ValueError(f"{context} v4 resume history is empty.")
    latest_index = int(len(history) - 1)
    latest_row = history[latest_index]
    prune_raw = latest_row.get("post_admission_prune")
    if not isinstance(prune_raw, Mapping):
        raise ValueError(
            f"{context} history[{latest_index}] lacks v4 post-admission prune state."
        )
    state_raw = prune_raw.get("phase1_prune_trust_state_after")
    if not isinstance(state_raw, Mapping):
        raise ValueError(
            f"{context} history[{latest_index}] lacks the latest v4 prune "
            "radius/mu/update_count state."
        )
    if str(state_raw.get("schema", "")) != "affine_deletion_fs_trust_state_v1":
        raise ValueError(
            f"{context} history[{latest_index}] latest v4 prune trust-state "
            "schema is unsupported."
        )
    try:
        radius = float(state_raw["radius"])
        metric_damping = float(state_raw["metric_damping"])
        update_count = int(state_raw["update_count"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            f"{context} history[{latest_index}] latest v4 prune "
            "radius/mu/update_count state cannot be reconstructed."
        ) from None
    if (
        not math.isfinite(radius)
        or radius <= 0.0
        or not math.isfinite(metric_damping)
        or metric_damping < 0.0
        or update_count < 0
    ):
        raise ValueError(
            f"{context} history[{latest_index}] latest v4 prune trust state "
            "contains invalid values."
        )
    if (
        artifact_profile_request == SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1
        and metric_damping != 0.0
    ):
        raise ValueError(
            f"{context} undamped pruning profile restored nonzero metric damping."
        )
    return {
        "schema": "affine_deletion_fs_trust_state_v1",
        "radius": float(radius),
        "metric_damping": float(metric_damping),
        "update_count": int(update_count),
    }


def _validate_authenticated_v4_pruned_lineage(
    *,
    history: Sequence[Mapping[str, Any]],
    operator_labels: Sequence[str],
    ansatz_depth: int,
    artifact_profile_request: str | None,
    context: str,
) -> dict[str, Any]:
    """Validate v4 depth divergence from signed accepted-prune events only."""

    if artifact_profile_request not in {
        SR_ROUTE_PROFILE_CANDIDATE_V4,
        SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1,
    }:
        raise ValueError(
            f"{context} accepts an accepted prune deletion lineage-depth "
            "divergence only for "
            "authenticated SR-SNAKE v4 artifacts."
        )

    def _int_value(value: Any, *, field: str, minimum: int = 0) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{context} {field} is not an integer.") from None
        if parsed < int(minimum):
            raise ValueError(f"{context} {field} must be >= {minimum}.")
        return int(parsed)

    def _trust_state_tuple(value: Any, *, field: str) -> tuple[float, float, int]:
        if not isinstance(value, Mapping):
            raise ValueError(f"{context} {field} is missing or malformed.")
        if str(value.get("schema", "")) != "affine_deletion_fs_trust_state_v1":
            raise ValueError(f"{context} {field} schema is unsupported.")
        try:
            radius = float(value["radius"])
            metric_damping = float(value["metric_damping"])
            update_count = int(value["update_count"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{context} {field} cannot be reconstructed.") from None
        if (
            not math.isfinite(radius)
            or radius <= 0.0
            or not math.isfinite(metric_damping)
            or metric_damping < 0.0
            or update_count < 0
        ):
            raise ValueError(f"{context} {field} contains invalid values.")
        return float(radius), float(metric_damping), int(update_count)

    active_labels: list[str] = []
    accepted_prune_count_total = 0
    previous_trust_state: tuple[float, float, int] | None = None
    final_prefix_checkpoint: dict[str, Any] | None = None

    for row_index, row_raw in enumerate(history):
        row = dict(row_raw)
        round_number = int(row_index + 1)
        if _int_value(
            row.get("depth"),
            field=f"history[{row_index}].depth",
            minimum=1,
        ) != round_number:
            raise ValueError(
                f"{context} v4 history depths must be contiguous from one."
            )
        if _int_value(
            row.get("batch_size"),
            field=f"history[{row_index}].batch_size",
            minimum=1,
        ) != 1:
            raise ValueError(f"{context} v4 resume requires singleton admissions.")
        selected_label = str(row.get("selected_op", ""))
        if not selected_label:
            raise ValueError(
                f"{context} history[{row_index}].selected_op is missing."
            )
        selected_position = _int_value(
            row.get("selected_position"),
            field=f"history[{row_index}].selected_position",
            minimum=0,
        )
        if selected_position > len(active_labels):
            raise ValueError(
                f"{context} history[{row_index}] insertion position is outside "
                "the active prefix."
            )
        active_labels.insert(int(selected_position), str(selected_label))

        checkpoint_raw = row.get("active_prefix_checkpoint")
        if not isinstance(checkpoint_raw, Mapping):
            raise ValueError(
                f"{context} accepted v4 prune lineage requires a signed active "
                f"prefix checkpoint at history[{row_index}]."
            )
        checkpoint = dict(checkpoint_raw)
        if (
            str(checkpoint.get("schema", ""))
            != "paper_i_signed_active_prefix_checkpoint_v1"
            or str(checkpoint.get("checkpoint_kind", ""))
            != "post_admission_prune"
        ):
            raise ValueError(
                f"{context} history[{row_index}] active-prefix schema is invalid."
            )
        if _int_value(
            checkpoint.get("outer_iteration"),
            field=f"history[{row_index}].active_prefix_checkpoint.outer_iteration",
            minimum=1,
        ) != round_number:
            raise ValueError(
                f"{context} history[{row_index}] active-prefix round disagrees."
            )
        checkpoint_sha256 = str(checkpoint.get("checkpoint_sha256", "")).lower()
        unsigned_checkpoint = dict(checkpoint)
        unsigned_checkpoint.pop("checkpoint_sha256", None)
        if (
            len(checkpoint_sha256) != 64
            or checkpoint_sha256 != digest_jsonable(unsigned_checkpoint)
        ):
            raise ValueError(
                f"{context} history[{row_index}] active-prefix checksum failed."
            )

        compact_prune_raw = row.get("post_admission_prune")
        checkpoint_prune_raw = checkpoint.get("post_admission_prune")
        if not isinstance(compact_prune_raw, Mapping) or not isinstance(
            checkpoint_prune_raw, Mapping
        ):
            raise ValueError(
                f"{context} history[{row_index}] v4 prune evidence is missing."
            )
        compact_prune = dict(compact_prune_raw)
        checkpoint_prune = dict(checkpoint_prune_raw)
        compact_accepted_count = _int_value(
            compact_prune.get("accepted_count", 0),
            field=f"history[{row_index}].post_admission_prune.accepted_count",
        )
        checkpoint_accepted_count = _int_value(
            checkpoint_prune.get("accepted_count", 0),
            field=(
                f"history[{row_index}].active_prefix_checkpoint."
                "post_admission_prune.accepted_count"
            ),
        )
        if (
            compact_accepted_count != checkpoint_accepted_count
            or compact_accepted_count not in {0, 1}
        ):
            raise ValueError(
                f"{context} history[{row_index}] accepted-prune count is invalid."
            )

        trust_before = _trust_state_tuple(
            checkpoint_prune.get("phase1_prune_trust_state_before"),
            field=f"history[{row_index}].phase1_prune_trust_state_before",
        )
        trust_after = _trust_state_tuple(
            checkpoint_prune.get("phase1_prune_trust_state_after"),
            field=f"history[{row_index}].phase1_prune_trust_state_after",
        )
        compact_trust_after = _trust_state_tuple(
            compact_prune.get("phase1_prune_trust_state_after"),
            field=(
                f"history[{row_index}].post_admission_prune."
                "phase1_prune_trust_state_after"
            ),
        )
        if compact_trust_after != trust_after:
            raise ValueError(
                f"{context} history[{row_index}] compact prune trust state "
                "disagrees with the signed active prefix."
            )
        if previous_trust_state is not None and trust_before != previous_trust_state:
            raise ValueError(
                f"{context} history[{row_index}] prune trust state is not continuous."
            )
        if trust_after[0] > trust_before[0] or trust_after[1] < trust_before[1]:
            raise ValueError(
                f"{context} history[{row_index}] prune trust update violates "
                "contraction-only/monotone-damping semantics."
            )
        if (
            artifact_profile_request
            == SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1
            and (trust_before[1] != 0.0 or trust_after[1] != 0.0)
        ):
            raise ValueError(
                f"{context} history[{row_index}] undamped pruning profile "
                "contains nonzero metric damping."
            )

        if compact_accepted_count == 1:
            if not bool(checkpoint_prune.get("executed", False)):
                raise ValueError(
                    f"{context} history[{row_index}] accepted prune was not executed."
                )
            trial_receipt = checkpoint_prune.get("phase1_prune_trial_receipt")
            if not isinstance(trial_receipt, Mapping) or str(
                trial_receipt.get("schema", "")
            ) != AFFINE_DELETION_FS_TRUST_TRIAL_RECEIPT_V1:
                raise ValueError(
                    f"{context} history[{row_index}] accepted prune lacks its "
                    "same-trial receipt."
                )
            trial_id = str(trial_receipt.get("trial_id", ""))
            if (
                not trial_id
                or str(trial_receipt.get("prediction_trial_id", "")) != trial_id
                or str(trial_receipt.get("realization_trial_id", "")) != trial_id
                or trial_receipt.get("prediction_complete") is not True
                or trial_receipt.get("realization_complete") is not True
                or trial_receipt.get("energy_receipt_complete") is not True
                or trial_receipt.get("measured_delete_refit_is_acceptance_authority")
                is not True
            ):
                raise ValueError(
                    f"{context} history[{row_index}] accepted prune receipt is "
                    "incomplete or mismatched."
                )
            decisions = checkpoint_prune.get("decisions")
            if not isinstance(decisions, Sequence) or isinstance(
                decisions, (str, bytes, bytearray)
            ) or len(decisions) != 1 or not isinstance(decisions[0], Mapping):
                raise ValueError(
                    f"{context} history[{row_index}] accepted prune decision is missing."
                )
            decision = dict(decisions[0])
            if decision.get("accepted") is not True:
                raise ValueError(
                    f"{context} history[{row_index}] prune decision was not accepted."
                )
            deletion_index = _int_value(
                decision.get("index"),
                field=f"history[{row_index}].prune_decision.index",
                minimum=0,
            )
            if deletion_index >= len(active_labels):
                raise ValueError(
                    f"{context} history[{row_index}] prune deletion index is invalid."
                )
            deletion_label = str(decision.get("label", ""))
            if not deletion_label or active_labels[deletion_index] != deletion_label:
                raise ValueError(
                    f"{context} history[{row_index}] prune deletion identity disagrees."
                )
            if trust_after[2] != trust_before[2] + 1:
                raise ValueError(
                    f"{context} history[{row_index}] accepted prune trust update "
                    "count did not advance exactly once."
                )
            del active_labels[deletion_index]
            accepted_prune_count_total += 1
        elif bool(checkpoint_prune.get("executed", False)):
            if trust_after[2] != trust_before[2] + 1:
                raise ValueError(
                    f"{context} history[{row_index}] rejected prune trust update "
                    "count did not advance exactly once."
                )
        elif trust_after != trust_before:
            raise ValueError(
                f"{context} history[{row_index}] non-executed prune mutated trust state."
            )

        checkpoint_labels_raw = checkpoint.get("ordered_active_operator_labels")
        if not isinstance(checkpoint_labels_raw, Sequence) or isinstance(
            checkpoint_labels_raw, (str, bytes, bytearray)
        ):
            raise ValueError(
                f"{context} history[{row_index}] active-prefix labels are invalid."
            )
        checkpoint_labels = [str(label) for label in checkpoint_labels_raw]
        if checkpoint_labels != active_labels or _int_value(
            checkpoint.get("active_ansatz_depth"),
            field=f"history[{row_index}].active_prefix_checkpoint.active_ansatz_depth",
        ) != len(active_labels):
            raise ValueError(
                f"{context} history[{row_index}] active-prefix reconstruction disagrees."
            )
        previous_trust_state = trust_after
        final_prefix_checkpoint = checkpoint

    if accepted_prune_count_total <= 0:
        raise ValueError(f"{context} v4 pruned-lineage validator found no deletion.")
    if int(ansatz_depth) != len(active_labels) or tuple(active_labels) != tuple(
        str(label) for label in operator_labels
    ):
        raise ValueError(f"{context} final active prefix does not match saved operators.")
    if int(ansatz_depth) != len(history) - int(accepted_prune_count_total):
        raise ValueError(
            f"{context} history/ansatz depth divergence is not explained exactly "
            "by authenticated accepted v4 prune events."
        )
    assert final_prefix_checkpoint is not None
    assert previous_trust_state is not None
    return {
        "schema": "authenticated_sr_v4_pruned_resume_lineage_v1",
        "accepted_prune_count": int(accepted_prune_count_total),
        "active_operator_labels": list(active_labels),
        "final_active_prefix_checkpoint": dict(final_prefix_checkpoint),
        "restored_prune_trust_state": {
            "schema": "affine_deletion_fs_trust_state_v1",
            "radius": float(previous_trust_state[0]),
            "metric_damping": float(previous_trust_state[1]),
            "update_count": int(previous_trust_state[2]),
        },
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contains_secret_marker(value: Any) -> bool:
    if value is None:
        return False
    text = str(value)
    if text == "":
        return False
    return bool(_SECRET_MARKER_RE.search(text))


def _iter_secret_value_hits(value: Any, *, path: str) -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            hits.extend(_iter_secret_value_hits(child, path=f"{path}.{key}"))
        return hits
    if isinstance(value, (list, tuple)):
        for idx, child in enumerate(value):
            hits.extend(_iter_secret_value_hits(child, path=f"{path}[{idx}]"))
        return hits
    if isinstance(value, (str, Path)) and contains_secret_marker(value):
        hits.append(path)
    return hits


def assert_no_secret_material(value: Any, *, context: str = "resume_scaffold") -> None:
    """Reject secret-like string values.

    The scan is intentionally value-only so audit keys like
    ``credential_audit`` can exist while raw token/API-key-like material cannot.
    """

    hits = _iter_secret_value_hits(value, path=str(context))
    if hits:
        preview = ", ".join(hits[:6])
        raise ValueError(
            "Secret-like token/API-key/credential value is not allowed in "
            f"{context}. Offending value path(s): {preview}"
        )


def assert_no_secret_cli_values(args_or_mapping: Any) -> None:
    mapping = vars(args_or_mapping) if not isinstance(args_or_mapping, Mapping) else args_or_mapping
    assert_no_secret_material(mapping, context="CLI arguments")


def _read_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Resume scaffold artifact must be a JSON object: {path}")
    return dict(payload)


def _adapt_block(payload: Mapping[str, Any]) -> dict[str, Any]:
    block = payload.get("adapt_vqe", {})
    if not isinstance(block, Mapping):
        raise ValueError("Resume scaffold artifact is missing adapt_vqe object.")
    return dict(block)


def _settings_block(payload: Mapping[str, Any]) -> dict[str, Any]:
    block = payload.get("settings", {})
    if not isinstance(block, Mapping):
        raise ValueError("Resume scaffold artifact is missing settings object.")
    return dict(block)


def _state_digest(payload: Mapping[str, Any], key: str) -> str | None:
    block = payload.get(key, None)
    if not isinstance(block, Mapping):
        return None
    return digest_jsonable(block)


def _source_depth(payload: Mapping[str, Any], runtime_input: ScaffoldRuntimeInput) -> int:
    adapt = _adapt_block(payload)
    raw = adapt.get("ansatz_depth", None)
    try:
        depth = int(raw)
        if depth >= 0:
            return depth
    except Exception:
        pass
    return int(len(runtime_input.selected_terms))


def _source_controller_round(payload: Mapping[str, Any]) -> int:
    adapt = _adapt_block(payload)
    for raw in (
        adapt.get("history_count"),
        payload.get("checkpoint", {}).get("depth")
        if isinstance(payload.get("checkpoint"), Mapping)
        else None,
    ):
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            return value
    history = adapt.get("history", ())
    if isinstance(history, Sequence) and not isinstance(
        history,
        (str, bytes, bytearray),
    ):
        return int(len(history))
    return 0


def _source_continuation_mode(payload: Mapping[str, Any]) -> str | None:
    settings = _settings_block(payload)
    adapt = _adapt_block(payload)
    continuation = adapt.get("continuation", {}) if isinstance(adapt.get("continuation", None), Mapping) else {}
    candidates: list[tuple[str, str]] = []
    for label, raw in (
        ("settings.adapt_continuation_mode", settings.get("adapt_continuation_mode")),
        ("settings.continuation_mode", settings.get("continuation_mode")),
        ("adapt_vqe.continuation_mode", adapt.get("continuation_mode")),
        ("adapt_vqe.continuation.mode", continuation.get("mode")),
        ("adapt_vqe.continuation.continuation_mode", continuation.get("continuation_mode")),
    ):
        if raw not in {None, ""}:
            candidates.append((label, str(raw).strip()))
    if not candidates:
        return None
    normalized = {mode.lower() for _label, mode in candidates}
    if len(normalized) > 1:
        raise ValueError(
            "Resume artifact has conflicting continuation mode fields: "
            + json.dumps({label: mode for label, mode in candidates}, sort_keys=True)
        )
    return candidates[0][1]


_SR_POWELL_CHART_POLICY_PATHS: tuple[tuple[str, ...], ...] = (
    ("settings", "sr_powell_coordinate_chart_policy"),
    (
        "settings",
        "formal_manifold_route_composition",
        "singleton_response_selector",
        "sr_powell_coordinate_chart_policy",
    ),
    (
        "settings",
        "historical_singleton_coordinate_trust_overlay",
        "powell_coordinate_chart_policy",
    ),
    ("settings", "static_route_identity", "powell_coordinate_chart_policy"),
    ("adapt_vqe", "static_route_identity", "powell_coordinate_chart_policy"),
    (
        "adapt_vqe",
        "formal_manifold_route_composition",
        "singleton_response_selector",
        "sr_powell_coordinate_chart_policy",
    ),
    (
        "adapt_vqe",
        "historical_singleton_coordinate_trust_overlay",
        "powell_coordinate_chart_policy",
    ),
    ("adapt_vqe", "optimizer_coordinate_chart", "powell_coordinate_chart_policy"),
    (
        "adapt_vqe",
        "terminal_active_prefix_checkpoint",
        "optimizer_coordinate_chart",
        "powell_coordinate_chart_policy",
    ),
    (
        "adapt_vqe",
        "continuation",
        "terminal_active_prefix_checkpoint",
        "optimizer_coordinate_chart",
        "powell_coordinate_chart_policy",
    ),
    ("checkpoint", "powell_coordinate_chart_policy"),
    ("checkpoint", "static_route_identity", "powell_coordinate_chart_policy"),
    ("checkpoint", "optimizer_coordinate_chart", "powell_coordinate_chart_policy"),
    (
        "checkpoint",
        "historical_singleton_coordinate_trust_overlay",
        "powell_coordinate_chart_policy",
    ),
    (
        "terminal_active_prefix_checkpoint",
        "optimizer_coordinate_chart",
        "powell_coordinate_chart_policy",
    ),
    ("static_route_identity", "powell_coordinate_chart_policy"),
    (
        "formal_manifold_route_composition",
        "singleton_response_selector",
        "sr_powell_coordinate_chart_policy",
    ),
    ("optimizer_coordinate_chart", "powell_coordinate_chart_policy"),
)

_SR_ROUTE_PROFILE_CONFORMANCE_PATHS: tuple[tuple[str, ...], ...] = (
    ("settings", "route_profile_conformance"),
    ("settings", "sr_powell_route_instance", "route_profile_conformance"),
    ("settings", "static_route_identity", "route_profile_conformance"),
    (
        "settings",
        "historical_singleton_coordinate_trust_overlay",
        "route_profile_conformance",
    ),
    ("adapt_vqe", "static_route_identity", "route_profile_conformance"),
    ("adapt_vqe", "sr_powell_route_instance", "route_profile_conformance"),
    (
        "adapt_vqe",
        "historical_singleton_coordinate_trust_overlay",
        "route_profile_conformance",
    ),
    ("checkpoint", "route_profile_conformance"),
    ("checkpoint", "static_route_identity", "route_profile_conformance"),
    ("checkpoint", "sr_powell_route_instance", "route_profile_conformance"),
    ("static_route_identity", "route_profile_conformance"),
    ("sr_powell_route_instance", "route_profile_conformance"),
)

_SR_COORDINATE_SOLVE_SCOPE_PATHS: tuple[tuple[str, ...], ...] = (
    ("settings", "historical_singleton_coordinate_solve_scope"),
    ("settings", "sr_powell_route_instance", "coordinate_solve_scope"),
    ("settings", "static_route_identity", "coordinate_solve_scope"),
    (
        "settings",
        "historical_singleton_coordinate_trust_overlay",
        "coordinate_solve_scope",
    ),
    ("adapt_vqe", "static_route_identity", "coordinate_solve_scope"),
    ("adapt_vqe", "sr_powell_route_instance", "coordinate_solve_scope"),
    (
        "adapt_vqe",
        "historical_singleton_coordinate_trust_overlay",
        "coordinate_solve_scope",
    ),
    ("checkpoint", "static_route_identity", "coordinate_solve_scope"),
    ("checkpoint", "sr_powell_route_instance", "coordinate_solve_scope"),
    ("static_route_identity", "coordinate_solve_scope"),
    ("sr_powell_route_instance", "coordinate_solve_scope"),
)

_SR_ROUTE_FAMILY_PATHS: tuple[tuple[str, ...], ...] = (
    ("settings", "route_family"),
    ("settings", "formal_manifold_route_composition", "route_family"),
    ("settings", "static_route_identity", "route_family"),
    ("adapt_vqe", "formal_manifold_route_composition", "route_family"),
    ("adapt_vqe", "static_route_identity", "route_family"),
    ("checkpoint", "static_route_identity", "route_family"),
    ("static_route_identity", "route_family"),
    ("formal_manifold_route_composition", "route_family"),
)

_SR_ROUTE_PROFILE_PATHS: tuple[tuple[str, ...], ...] = (
    ("settings", "route_profile"),
    ("settings", "formal_manifold_route_composition", "route_profile"),
    ("settings", "static_route_identity", "route_profile"),
    ("adapt_vqe", "static_route_identity", "route_profile"),
    ("adapt_vqe", "formal_manifold_route_composition", "route_profile"),
    ("checkpoint", "static_route_identity", "route_profile"),
    ("static_route_identity", "route_profile"),
    ("formal_manifold_route_composition", "route_profile"),
)

_SR_ROUTE_PROFILES = frozenset(
    {
        SR_ROUTE_PROFILE_DISABLED,
        SR_ROUTE_PROFILE_CONVENTIONAL_V2,
        SR_ROUTE_PROFILE_CONVENTIONAL_V3,
        SR_ROUTE_PROFILE_CANDIDATE_V4,
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
        SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1,
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1,
        SR_ROUTE_PROFILE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1,
        SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2,
        SR_ROUTE_PROFILE_REDUCED_POWELL,
        SR_ROUTE_PROFILE_PHASE2_PHASE3_WHITENED,
        SR_ROUTE_PROFILE_SADDLE_ONLY,
        SR_ROUTE_PROFILE_SADDLE_PLUS_MODELED_MINIMUM,
    }
)

_SR_ROUTE_PROFILE_TO_POWELL_CHART = {
    SR_ROUTE_PROFILE_DISABLED: (
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    ),
    SR_ROUTE_PROFILE_CONVENTIONAL_V2: (
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    ),
    SR_ROUTE_PROFILE_CONVENTIONAL_V3: (
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    ),
    SR_ROUTE_PROFILE_CANDIDATE_V4: (
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    ),
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1: (
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    ),
    SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1: (
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    ),
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1: (
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    ),
    SR_ROUTE_PROFILE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1: (
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    ),
    SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2: (
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    ),
    SR_ROUTE_PROFILE_REDUCED_POWELL: (
        SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
    ),
    SR_ROUTE_PROFILE_PHASE2_PHASE3_WHITENED: (
        SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
    ),
    SR_ROUTE_PROFILE_SADDLE_ONLY: (
        SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
    ),
    SR_ROUTE_PROFILE_SADDLE_PLUS_MODELED_MINIMUM: (
        SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
    ),
}

_SOURCE_LOCK_INDICATOR_PATHS: tuple[tuple[str, ...], ...] = (
    ("settings", "phase3_source_lock_preferred_sequence"),
    ("adapt_vqe", "phase3_source_lock_preferred_sequence"),
    ("phase3_source_lock_preferred_sequence",),
    ("phase3_source_lock",),
)

_SR_ROUTE_PROFILE_REQUEST_PATHS: tuple[tuple[str, ...], ...] = (
    ("settings", "sr_route_profile_request"),
    (
        "settings",
        "historical_singleton_coordinate_trust_overlay",
        "sr_route_profile_request",
    ),
    ("settings", "static_route_identity", "sr_route_profile_request"),
    ("adapt_vqe", "sr_route_profile_request"),
    (
        "adapt_vqe",
        "historical_singleton_coordinate_trust_overlay",
        "sr_route_profile_request",
    ),
    ("adapt_vqe", "static_route_identity", "sr_route_profile_request"),
    ("checkpoint", "sr_route_profile_request"),
    ("checkpoint", "settings", "sr_route_profile_request"),
    ("checkpoint", "static_route_identity", "sr_route_profile_request"),
    ("sr_route_profile_request",),
    ("static_route_identity", "sr_route_profile_request"),
)

_SR_ROUTE_PROFILE_CONTRACT_PATHS: tuple[tuple[str, ...], ...] = (
    ("settings", "sr_route_profile_contract"),
    (
        "settings",
        "historical_singleton_coordinate_trust_overlay",
        "sr_route_profile_contract",
    ),
    ("settings", "static_route_identity", "sr_route_profile_contract"),
    ("adapt_vqe", "sr_route_profile_contract"),
    (
        "adapt_vqe",
        "historical_singleton_coordinate_trust_overlay",
        "sr_route_profile_contract",
    ),
    ("adapt_vqe", "static_route_identity", "sr_route_profile_contract"),
    ("checkpoint", "sr_route_profile_contract"),
    ("checkpoint", "settings", "sr_route_profile_contract"),
    ("checkpoint", "static_route_identity", "sr_route_profile_contract"),
    ("sr_route_profile_contract",),
    ("static_route_identity", "sr_route_profile_contract"),
)

_SR_ROUTE_PROFILE_CONTRACT_SHA256_PATHS: tuple[tuple[str, ...], ...] = (
    ("settings", "sr_route_profile_contract_sha256"),
    (
        "settings",
        "historical_singleton_coordinate_trust_overlay",
        "sr_route_profile_contract_sha256",
    ),
    (
        "settings",
        "static_route_identity",
        "sr_route_profile_contract_sha256",
    ),
    ("adapt_vqe", "sr_route_profile_contract_sha256"),
    (
        "adapt_vqe",
        "historical_singleton_coordinate_trust_overlay",
        "sr_route_profile_contract_sha256",
    ),
    (
        "adapt_vqe",
        "static_route_identity",
        "sr_route_profile_contract_sha256",
    ),
    ("checkpoint", "sr_route_profile_contract_sha256"),
    ("checkpoint", "settings", "sr_route_profile_contract_sha256"),
    (
        "checkpoint",
        "static_route_identity",
        "sr_route_profile_contract_sha256",
    ),
    ("sr_route_profile_contract_sha256",),
    ("static_route_identity", "sr_route_profile_contract_sha256"),
)

_PHASE3_RESPONSE_COORDINATE_SCOPE_PATHS: tuple[tuple[str, ...], ...] = (
    ("settings", "phase3_response_coordinate_scope"),
    ("settings", "static_route_identity", "phase3_response_coordinate_scope"),
    (
        "settings",
        "historical_singleton_coordinate_trust_overlay",
        "phase3_response_coordinate_scope",
    ),
    ("adapt_vqe", "phase3_response_coordinate_scope"),
    (
        "adapt_vqe",
        "static_route_identity",
        "phase3_response_coordinate_scope",
    ),
    (
        "adapt_vqe",
        "historical_singleton_coordinate_trust_overlay",
        "phase3_response_coordinate_scope",
    ),
    (
        "adapt_vqe",
        "terminal_active_prefix_checkpoint",
        "phase3_response_coordinate_scope",
    ),
    (
        "adapt_vqe",
        "continuation",
        "terminal_active_prefix_checkpoint",
        "phase3_response_coordinate_scope",
    ),
    ("checkpoint", "phase3_response_coordinate_scope"),
    ("checkpoint", "settings", "phase3_response_coordinate_scope"),
    (
        "checkpoint",
        "static_route_identity",
        "phase3_response_coordinate_scope",
    ),
    ("phase3_response_coordinate_scope",),
    ("static_route_identity", "phase3_response_coordinate_scope"),
)

_PHASE12_ENERGY_MODEL_POLICY_PATHS = {
    key: tuple(
        (*path[:-1], key) for path in _PHASE3_RESPONSE_COORDINATE_SCOPE_PATHS
    )
    for key in (
        "phase1_score_mode",
        "phase1_energy_model",
        "phase2_curvature_policy",
        "phase2_cheap_curvature_proxy_policy",
    )
}


def _nested_payload_value(
    payload: Mapping[str, Any], path: tuple[str, ...]
) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _path_label(path: tuple[str, ...]) -> str:
    return ".".join(path)


def validate_resume_phase12_energy_model_policies(
    payload: Mapping[str, Any],
    *,
    expected_profile_request: str | None = None,
    context: str = "Static scaffold resume",
) -> dict[str, Any]:
    normalizers = {
        "phase1_score_mode": normalize_phase1_score_mode_policy,
        "phase1_energy_model": normalize_phase1_energy_model,
        "phase2_curvature_policy": normalize_phase2_curvature_policy,
        "phase2_cheap_curvature_proxy_policy": (
            normalize_phase2_cheap_curvature_proxy_policy
        ),
    }
    serialized: dict[str, dict[str, str]] = {
        key: {} for key in normalizers
    }
    for key, normalizer in normalizers.items():
        for path in _PHASE12_ENERGY_MODEL_POLICY_PATHS[key]:
            raw = _nested_payload_value(payload, path)
            if raw in {None, ""}:
                continue
            try:
                serialized[key][_path_label(path)] = normalizer(raw)
            except ValueError as exc:
                raise ValueError(
                    f"{context} has an unknown {key} at {_path_label(path)}: "
                    f"{raw!r}."
                ) from exc
        if len(set(serialized[key].values())) > 1:
            raise ValueError(
                f"{context} has conflicting {key} values: "
                + json.dumps(serialized[key], sort_keys=True)
            )

    artifact_profiles: set[str] = set()
    for path in _SR_ROUTE_PROFILE_REQUEST_PATHS:
        raw = _nested_payload_value(payload, path)
        if raw not in {None, ""}:
            artifact_profiles.add(normalize_sr_route_profile_request(raw))
    if len(artifact_profiles) > 1:
        raise ValueError(
            f"{context} has conflicting SR profiles while validating Phase-I/II "
            "energy-model policies."
        )
    artifact_profile = next(iter(artifact_profiles), None)
    expected_profile = (
        None
        if expected_profile_request in {None, ""}
        else normalize_sr_route_profile_request(expected_profile_request)
    )
    effective_profile = (
        expected_profile
        if expected_profile not in {None, SR_ROUTE_PROFILE_REQUEST_OFF}
        else artifact_profile
    )
    v4_expected = {
        "phase1_score_mode": PHASE1_SCORE_MODE_TRUST_REGION_V1,
        "phase1_energy_model": PHASE1_ENERGY_MODEL_FIRST_ORDER_FS_TRUST_V1,
        "phase2_curvature_policy": (
            PHASE2_CURVATURE_POLICY_MEASURED_REQUIRED_FAIL_CLOSED_V1
        ),
        "phase2_cheap_curvature_proxy_policy": (
            PHASE2_CHEAP_CURVATURE_PROXY_POLICY_OFF
        ),
    }
    if effective_profile in {
        SR_ROUTE_PROFILE_CANDIDATE_V4,
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
        SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1,
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1,
    }:
        missing = [key for key, values in serialized.items() if not values]
        if missing:
            raise ValueError(
                f"{context} strict SR-SNAKE artifact is missing explicit Phase-I/II "
                "energy-model policies: " + ",".join(missing)
            )
        resolved = {
            key: next(iter(values.values()))
            for key, values in serialized.items()
        }
        if resolved != v4_expected:
            raise ValueError(
                f"{context} strict SR-SNAKE Phase-I/II policies drifted: "
                f"{resolved!r}."
            )
        source = "serialized_artifact"
    elif effective_profile in {
        SR_ROUTE_PROFILE_CANONICAL_V1,
        SR_ROUTE_PROFILE_CONVENTIONAL_V2,
        SR_ROUTE_PROFILE_CONVENTIONAL_V3,
        SR_ROUTE_PROFILE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1,
        SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2,
    }:
        resolved = dict(HISTORICAL_SR_SNAKE_PHASE12_ENERGY_MODEL_SETTINGS)
        for key, values in serialized.items():
            if values and next(iter(values.values())) != resolved[key]:
                raise ValueError(
                    f"{context} historical SR-SNAKE {key} drifted from its "
                    "versioned replay contract."
                )
        source = (
            "serialized_artifact"
            if all(bool(values) for values in serialized.values())
            else "versioned_historical_profile_contract"
        )
    else:
        resolved = {
            key: (next(iter(values.values())) if values else None)
            for key, values in serialized.items()
        }
        source = "serialized_artifact" if all(resolved.values()) else "unresolved"
    return {
        "schema": "static_adapt_resume_phase12_energy_model_policies_v1",
        "resolved": dict(resolved),
        "resolution_source": str(source),
        "serialized_fields": {
            key: dict(sorted(values.items()))
            for key, values in serialized.items()
        },
        "inferred": False,
    }


def validate_resume_sr_route_profile_contract(
    payload: Mapping[str, Any],
    *,
    expected_profile_request: str | None = None,
    expected_contract: Mapping[str, Any] | None = None,
    expected_contract_sha256: str | None = None,
    context: str = "Static scaffold resume",
) -> dict[str, Any]:
    """Validate a registered SR-SNAKE replay identity without inference.

    The Powell-chart validator protects one important optimizer setting.  The
    route selector additionally locks the complete execution contract.  Every
    serialized alias must agree, and an invocation that explicitly requests a
    registered SR-SNAKE profile may not consume a legacy artifact that lacks
    that profile's contract.
    """

    request_fields: dict[str, str] = {}
    for path in _SR_ROUTE_PROFILE_REQUEST_PATHS:
        raw = _nested_payload_value(payload, path)
        if raw in {None, ""}:
            continue
        try:
            value = normalize_sr_route_profile_request(raw)
        except ValueError as exc:
            raise ValueError(
                f"{context} has an unknown SR route-profile request at "
                f"{_path_label(path)}: {raw!r}."
            ) from exc
        request_fields[_path_label(path)] = value
    distinct_requests = sorted(set(request_fields.values()))
    if len(distinct_requests) > 1:
        raise ValueError(
            f"{context} has conflicting SR route-profile requests: "
            + json.dumps(request_fields, sort_keys=True)
        )
    artifact_request = distinct_requests[0] if distinct_requests else None

    contract_fields: dict[str, Mapping[str, Any]] = {}
    for path in _SR_ROUTE_PROFILE_CONTRACT_PATHS:
        raw = _nested_payload_value(payload, path)
        if raw is None:
            continue
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"{context} has a non-object SR route-profile contract at "
                f"{_path_label(path)}."
            )
        contract_fields[_path_label(path)] = dict(raw)
    contract_digests = {
        digest_jsonable(dict(contract)) for contract in contract_fields.values()
    }
    if len(contract_digests) > 1:
        raise ValueError(
            f"{context} has conflicting serialized SR route-profile contracts "
            f"at {sorted(contract_fields)}."
        )
    artifact_contract = (
        dict(next(iter(contract_fields.values()))) if contract_fields else None
    )

    sha_fields: dict[str, str] = {}
    for path in _SR_ROUTE_PROFILE_CONTRACT_SHA256_PATHS:
        raw = _nested_payload_value(payload, path)
        if raw in {None, ""}:
            continue
        value = str(raw).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(
                f"{context} has an invalid SR route-profile contract SHA-256 "
                f"at {_path_label(path)}: {raw!r}."
            )
        sha_fields[_path_label(path)] = value
    distinct_sha = sorted(set(sha_fields.values()))
    if len(distinct_sha) > 1:
        raise ValueError(
            f"{context} has conflicting SR route-profile contract SHA-256 "
            "fields: "
            + json.dumps(sha_fields, sort_keys=True)
        )
    artifact_sha256 = distinct_sha[0] if distinct_sha else None

    expected_request = (
        None
        if expected_profile_request in {None, ""}
        else normalize_sr_route_profile_request(expected_profile_request)
    )
    registered_profiles = frozenset(
        {
            SR_ROUTE_PROFILE_CANONICAL_V1,
            SR_ROUTE_PROFILE_CONVENTIONAL_V2,
            SR_ROUTE_PROFILE_CONVENTIONAL_V3,
            SR_ROUTE_PROFILE_CANDIDATE_V4,
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1,
            SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1,
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1,
            SR_ROUTE_PROFILE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1,
            SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2,
        }
    )
    registered_present = bool(
        artifact_request in registered_profiles
        or artifact_contract is not None
        or artifact_sha256 is not None
    )
    registered_expected = expected_request in registered_profiles

    if registered_present:
        if artifact_request is None:
            raise ValueError(
                f"{context} has an SR route-profile contract but no serialized "
                "route-profile request."
            )
        if artifact_request not in registered_profiles:
            raise ValueError(
                f"{context} associates a registered SR contract with "
                f"profile request {artifact_request!r}."
            )
        validate_sr_route_profile_contract(
            profile_request=artifact_request,
            contract=artifact_contract,
            contract_sha256=artifact_sha256,
        )

    if registered_expected:
        if not registered_present:
            raise ValueError(
                f"{context} was invoked as {expected_request!r}, but the resume "
                "artifact lacks its complete route-profile contract."
            )
        if artifact_request != expected_request:
            raise ValueError(
                f"{context} resume artifact uses {artifact_request!r}, but the "
                f"current invocation requests {expected_request!r}."
            )
        expected_payload = validate_sr_route_profile_contract(
            profile_request=expected_request,
            contract=expected_contract,
            contract_sha256=expected_contract_sha256,
        )
        if artifact_contract != expected_payload:
            raise ValueError(
                f"{context} SR-SNAKE route-profile contract does not match "
                "the current invocation."
            )
        if artifact_sha256 != str(expected_contract_sha256 or "").lower():
            raise ValueError(
                f"{context} SR-SNAKE route-profile contract SHA-256 does "
                "not match the current invocation."
            )
    elif expected_request == SR_ROUTE_PROFILE_REQUEST_OFF and registered_present:
        raise ValueError(
            f"{context} artifact uses a registered SR-SNAKE profile, but the current "
            "invocation did not explicitly request that route profile."
        )

    phase3_response_scope_validation = (
        validate_resume_phase3_response_coordinate_scope(
            payload,
            expected_profile_request=(
                expected_request
                if expected_request not in {None, SR_ROUTE_PROFILE_REQUEST_OFF}
                else artifact_request
            ),
            context=context,
        )
    )
    phase12_energy_model_validation = (
        validate_resume_phase12_energy_model_policies(
            payload,
            expected_profile_request=(
                expected_request
                if expected_request not in {None, SR_ROUTE_PROFILE_REQUEST_OFF}
                else artifact_request
            ),
            context=context,
        )
    )

    return {
        "schema_version": "static_adapt_resume_sr_route_profile_contract_v1",
        "status": "pass" if registered_present else "not_applicable",
        "artifact_profile_request": artifact_request,
        "expected_profile_request": expected_request,
        "contract_sha256": artifact_sha256,
        "expected_contract_sha256": (
            None
            if expected_contract_sha256 in {None, ""}
            else str(expected_contract_sha256).strip().lower()
        ),
        "request_fields": dict(sorted(request_fields.items())),
        "contract_source_fields": sorted(contract_fields),
        "contract_sha256_fields": dict(sorted(sha_fields.items())),
        "phase3_response_coordinate_scope": (
            phase3_response_scope_validation["resolved_scope"]
        ),
        "phase3_response_coordinate_scope_validation": (
            phase3_response_scope_validation
        ),
        "phase12_energy_model_policies": dict(
            phase12_energy_model_validation["resolved"]
        ),
        "phase12_energy_model_policy_validation": (
            phase12_energy_model_validation
        ),
        "inferred": False,
    }


def validate_resume_phase3_response_coordinate_scope(
    payload: Mapping[str, Any],
    *,
    expected_scope: str | None = None,
    expected_profile_request: str | None = None,
    required: bool | None = None,
    context: str = "Static scaffold resume",
) -> dict[str, Any]:
    """Validate the serialized Phase-III response-coordinate policy.

    SR-SNAKE v3 must carry the explicit full-active-plus-singleton policy in
    its artifact.  Frozen v1/v2 artifacts may predate this telemetry; their
    versioned identities resolve only to the explicit legacy-coupled policy,
    never from a window-size sentinel.
    """

    scope_fields: dict[str, str] = {}
    for path in _PHASE3_RESPONSE_COORDINATE_SCOPE_PATHS:
        raw = _nested_payload_value(payload, path)
        if raw in {None, ""}:
            continue
        try:
            value = normalize_phase3_response_coordinate_scope(raw)
        except ValueError as exc:
            raise ValueError(
                f"{context} has an unknown Phase-III response-coordinate "
                f"scope at {_path_label(path)}: {raw!r}."
            ) from exc
        scope_fields[_path_label(path)] = value
    distinct_scopes = sorted(set(scope_fields.values()))
    if len(distinct_scopes) > 1:
        raise ValueError(
            f"{context} has conflicting Phase-III response-coordinate scopes: "
            + json.dumps(scope_fields, sort_keys=True)
        )

    request_values: list[str] = []
    for path in _SR_ROUTE_PROFILE_REQUEST_PATHS:
        raw = _nested_payload_value(payload, path)
        if raw in {None, ""}:
            continue
        request_values.append(normalize_sr_route_profile_request(raw))
    artifact_profiles = sorted(set(request_values))
    if len(artifact_profiles) > 1:
        raise ValueError(
            f"{context} has conflicting SR route-profile requests while "
            "validating the Phase-III response scope."
        )
    artifact_profile = artifact_profiles[0] if artifact_profiles else None
    expected_profile = (
        None
        if expected_profile_request in {None, ""}
        else normalize_sr_route_profile_request(expected_profile_request)
    )
    effective_profile = (
        expected_profile
        if expected_profile not in {None, SR_ROUTE_PROFILE_REQUEST_OFF}
        else artifact_profile
    )
    if (
        artifact_profile not in {None, SR_ROUTE_PROFILE_REQUEST_OFF}
        and expected_profile not in {None, SR_ROUTE_PROFILE_REQUEST_OFF}
        and artifact_profile != expected_profile
    ):
        raise ValueError(
            f"{context} Phase-III response scope belongs to profile "
            f"{artifact_profile!r}, but the invocation requests "
            f"{expected_profile!r}."
        )

    explicit_scope = distinct_scopes[0] if distinct_scopes else None
    historical_profiles = {
        SR_ROUTE_PROFILE_CANONICAL_V1,
        SR_ROUTE_PROFILE_CONVENTIONAL_V2,
    }
    historical_scope = HISTORICAL_SR_SNAKE_RESPONSE_SCOPE_SETTINGS[
        "phase3_response_coordinate_scope"
    ]
    if effective_profile in {
        SR_ROUTE_PROFILE_CONVENTIONAL_V3,
        SR_ROUTE_PROFILE_CANDIDATE_V4,
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
        SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1,
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1,
        SR_ROUTE_PROFILE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1,
        SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2,
    }:
        if explicit_scope is None:
            raise ValueError(
                f"{context} full-response SR-SNAKE artifact is missing "
                "phase3_response_coordinate_scope."
            )
        if (
            explicit_scope
            != PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1
        ):
            raise ValueError(
                f"{context} full-response SR-SNAKE profile requires "
                "full_active_plus_singleton_v1; artifact has "
                f"{explicit_scope!r}."
            )
        resolved = explicit_scope
        resolution_source = "serialized_artifact"
    elif effective_profile in historical_profiles:
        if explicit_scope is not None and explicit_scope != historical_scope:
            raise ValueError(
                f"{context} historical SR-SNAKE profile requires "
                f"{historical_scope!r}; artifact has {explicit_scope!r}."
            )
        resolved = historical_scope
        resolution_source = (
            "serialized_artifact"
            if explicit_scope is not None
            else "versioned_historical_profile_contract"
        )
    elif explicit_scope is not None:
        resolved = explicit_scope
        resolution_source = "serialized_artifact"
    elif bool(required):
        raise ValueError(
            f"{context} is missing phase3_response_coordinate_scope."
        )
    else:
        resolved = None
        resolution_source = "not_applicable"

    normalized_expected = (
        None
        if expected_scope in {None, ""}
        else normalize_phase3_response_coordinate_scope(expected_scope)
    )
    if (
        normalized_expected is not None
        and resolved is not None
        and resolved != normalized_expected
    ):
        raise ValueError(
            f"{context} Phase-III response-coordinate scope drifted: "
            f"artifact={resolved!r}, expected={normalized_expected!r}."
        )

    return {
        "schema_version": (
            "static_adapt_resume_phase3_response_coordinate_scope_v1"
        ),
        "status": "pass" if resolved is not None else "not_applicable",
        "resolved_scope": resolved,
        "resolution_source": resolution_source,
        "artifact_profile_request": artifact_profile,
        "expected_profile_request": expected_profile,
        "expected_scope": normalized_expected,
        "scope_fields": dict(sorted(scope_fields.items())),
        "inferred_from_window_or_refit_schedule": False,
    }


def _resume_powell_chart_policy_required(payload: Mapping[str, Any]) -> bool:
    route_families = {
        str(value).strip().lower()
        for path in _SR_ROUTE_FAMILY_PATHS
        if (value := _nested_payload_value(payload, path)) is not None
        and value != ""
    }
    if str(SR_ROUTE_FAMILY).strip().lower() in route_families:
        return True
    if str(FORMAL_MANIFOLD_ROUTE_FAMILY).strip().lower() in route_families:
        composition = extract_formal_manifold_route_composition(payload)
        if composition is not None and str(
            composition.get("candidate_selector_family") or ""
        ).strip().lower() == str(
            FORMAL_MANIFOLD_SR_SELECTOR_FAMILY
        ).strip().lower():
            return True
    route_profiles = {
        str(value).strip().lower()
        for path in _SR_ROUTE_PROFILE_PATHS
        if (value := _nested_payload_value(payload, path)) is not None
        and value != ""
    }
    if route_profiles & {
        str(value).strip().lower() for value in _SR_ROUTE_PROFILES
    }:
        return True
    for path in _SOURCE_LOCK_INDICATOR_PATHS:
        value = _nested_payload_value(payload, path)
        if isinstance(value, str):
            if value.strip():
                return True
        elif value:
            return True
    return False


def validate_resume_powell_coordinate_chart_policy(
    payload: Mapping[str, Any],
    *,
    expected_policy: str | None = None,
    expected_route_profile_conformance: str | None = None,
    required: bool | None = None,
    context: str = "Static scaffold resume",
) -> dict[str, Any]:
    """Extract and strictly validate serialized SR-SNAKE Powell-chart identity.

    A historical or source-locked SR replay must never infer this execution
    policy from the live optimizer.  Every serialized alias is therefore
    collected and required to agree.  Callers may additionally supply the
    already-resolved current route policy to detect resume drift.
    """

    required_value = (
        _resume_powell_chart_policy_required(payload)
        if required is None
        else bool(required)
    )
    source_fields: dict[str, str] = {}
    for path in _SR_POWELL_CHART_POLICY_PATHS:
        raw = _nested_payload_value(payload, path)
        if raw is None or raw == "":
            continue
        value = str(raw).strip().lower()
        if value not in SR_POWELL_COORDINATE_CHART_POLICY_CHOICES:
            raise ValueError(
                f"{context} has unknown Powell coordinate-chart policy at "
                f"{_path_label(path)}: {raw!r}. Expected one of "
                f"{list(SR_POWELL_COORDINATE_CHART_POLICY_CHOICES)}."
            )
        source_fields[_path_label(path)] = value

    distinct = sorted(set(source_fields.values()))
    if len(distinct) > 1:
        raise ValueError(
            f"{context} has conflicting Powell coordinate-chart policies: "
            + json.dumps(source_fields, sort_keys=True)
        )
    resolved = distinct[0] if distinct else None
    if required_value and resolved is None:
        raise ValueError(
            f"{context} is SR/source-locked but is missing the explicit "
            "Powell coordinate-chart policy; replay fails closed."
        )

    conformance_fields: dict[str, str] = {}
    for path in _SR_ROUTE_PROFILE_CONFORMANCE_PATHS:
        raw = _nested_payload_value(payload, path)
        if raw in {None, ""}:
            continue
        value = str(raw).strip().lower()
        if value not in SR_ROUTE_PROFILE_CONFORMANCE_CHOICES:
            raise ValueError(
                f"{context} has unknown route-profile conformance marker at "
                f"{_path_label(path)}: {raw!r}. Expected one of "
                f"{list(SR_ROUTE_PROFILE_CONFORMANCE_CHOICES)}."
            )
        conformance_fields[_path_label(path)] = value
    distinct_conformance = sorted(set(conformance_fields.values()))
    if len(distinct_conformance) > 1:
        raise ValueError(
            f"{context} has conflicting route-profile conformance markers: "
            + json.dumps(conformance_fields, sort_keys=True)
        )
    route_profile_conformance = (
        distinct_conformance[0] if distinct_conformance else None
    )

    coordinate_scope_fields: dict[str, str] = {}
    for path in _SR_COORDINATE_SOLVE_SCOPE_PATHS:
        raw = _nested_payload_value(payload, path)
        if raw in {None, ""}:
            continue
        coordinate_scope_fields[_path_label(path)] = str(raw).strip().lower()
    distinct_coordinate_scopes = sorted(set(coordinate_scope_fields.values()))
    if len(distinct_coordinate_scopes) > 1:
        raise ValueError(
            f"{context} has conflicting SR coordinate-solve scopes: "
            + json.dumps(coordinate_scope_fields, sort_keys=True)
        )
    coordinate_solve_scope = (
        distinct_coordinate_scopes[0] if distinct_coordinate_scopes else None
    )

    route_profile_fields: dict[str, str] = {}
    for path in _SR_ROUTE_PROFILE_PATHS:
        raw = _nested_payload_value(payload, path)
        if raw in {None, ""}:
            continue
        profile = str(raw).strip().lower()
        if profile in _SR_ROUTE_PROFILE_TO_POWELL_CHART:
            route_profile_fields[_path_label(path)] = profile
    profile_expected_charts = {
        _SR_ROUTE_PROFILE_TO_POWELL_CHART[profile]
        for profile in route_profile_fields.values()
    }
    if len(profile_expected_charts) > 1:
        raise ValueError(
            f"{context} has SR route profiles with incompatible Powell "
            "coordinate-chart policies: "
            + json.dumps(route_profile_fields, sort_keys=True)
        )
    route_profile_expected = (
        next(iter(profile_expected_charts)) if profile_expected_charts else None
    )
    phase2_phase3_expanded_pair = bool(
        resolved
        == SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
        and route_profile_fields
        and set(route_profile_fields.values())
        == {SR_ROUTE_PROFILE_PHASE2_PHASE3_WHITENED}
    )
    explicit_unpromoted_ablation = bool(
        phase2_phase3_expanded_pair
        and route_profile_conformance
        == SR_ROUTE_PROFILE_CONFORMANCE_UNPROMOTED_EXPLICIT_ABLATION
        and coordinate_solve_scope
        == SR_COORDINATE_SOLVE_SCOPE_PHASE2_AND_PHASE3_V1
    )
    if phase2_phase3_expanded_pair and not explicit_unpromoted_ablation:
        raise ValueError(
            f"{context} explicit Phase-II+III expanded-chart resume requires "
            "the serialized unpromoted route-profile conformance marker and "
            f"coordinate_solve_scope={SR_COORDINATE_SOLVE_SCOPE_PHASE2_AND_PHASE3_V1!r}; "
            "replay refuses to infer either field."
        )
    if (
        route_profile_conformance
        == SR_ROUTE_PROFILE_CONFORMANCE_UNPROMOTED_EXPLICIT_ABLATION
        and not explicit_unpromoted_ablation
    ):
        raise ValueError(
            f"{context} serialized the unpromoted explicit-ablation marker "
            "without the exact Phase-II+III expanded-chart route instance."
        )
    if (
        resolved is not None
        and route_profile_expected is not None
        and resolved != route_profile_expected
        and not explicit_unpromoted_ablation
    ):
        raise ValueError(
            f"{context} route-profile/Powell-chart mismatch: "
            f"profiles={json.dumps(route_profile_fields, sort_keys=True)}, "
            f"chart={resolved!r}, expected={route_profile_expected!r}."
        )

    expected: str | None = None
    if expected_policy not in {None, ""}:
        expected = str(expected_policy).strip().lower()
        if expected not in SR_POWELL_COORDINATE_CHART_POLICY_CHOICES:
            raise ValueError(
                f"{context} expected Powell coordinate-chart policy is unknown: "
                f"{expected_policy!r}."
            )
        if resolved is None:
            raise ValueError(
                f"{context} is missing the Powell coordinate-chart policy "
                f"required by the current route ({expected})."
            )
        if resolved != expected:
            raise ValueError(
                f"{context} Powell coordinate-chart policy mismatch: "
                f"artifact={resolved!r}, current={expected!r}."
            )

    expected_conformance: str | None = None
    if expected_route_profile_conformance not in {None, ""}:
        expected_conformance = str(
            expected_route_profile_conformance
        ).strip().lower()
        if expected_conformance not in SR_ROUTE_PROFILE_CONFORMANCE_CHOICES:
            raise ValueError(
                f"{context} expected route-profile conformance marker is "
                f"unknown: {expected_route_profile_conformance!r}."
            )
        if route_profile_conformance is None:
            raise ValueError(
                f"{context} is missing the route-profile conformance marker "
                f"required by the current route ({expected_conformance})."
            )
        if route_profile_conformance != expected_conformance:
            raise ValueError(
                f"{context} route-profile conformance mismatch: "
                f"artifact={route_profile_conformance!r}, "
                f"current={expected_conformance!r}."
            )

    return {
        "schema_version": "static_adapt_resume_powell_chart_policy_v1",
        "status": "pass" if resolved is not None else "not_applicable",
        "required": bool(required_value or expected is not None),
        "resolved_policy": resolved,
        "expected_policy": expected,
        "source_fields": dict(sorted(source_fields.items())),
        "source_field_count": int(len(source_fields)),
        "route_profile_fields": dict(sorted(route_profile_fields.items())),
        "route_profile_expected_policy": route_profile_expected,
        "route_profile_conformance": route_profile_conformance,
        "expected_route_profile_conformance": expected_conformance,
        "route_profile_conformance_fields": dict(
            sorted(conformance_fields.items())
        ),
        "coordinate_solve_scope": coordinate_solve_scope,
        "coordinate_solve_scope_fields": dict(
            sorted(coordinate_scope_fields.items())
        ),
        "explicit_unpromoted_ablation": bool(explicit_unpromoted_ablation),
        "inferred": False,
    }


def build_resume_import_summary(
    source: ResumeScaffoldSource | None = None,
    *,
    artifact_json: str | Path | None = None,
    artifact_sha256: str | None = None,
    payload: Mapping[str, Any] | None = None,
    runtime_input: ScaffoldRuntimeInput | None = None,
    validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if source is not None:
        artifact_json = source.artifact_json
        artifact_sha256 = source.artifact_sha256
        payload = source.payload
        runtime_input = source.runtime_input
    if artifact_json is None or artifact_sha256 is None or payload is None or runtime_input is None:
        raise ValueError("build_resume_import_summary requires source or explicit artifact/payload/runtime_input.")
    adapt = _adapt_block(payload)
    settings = _settings_block(payload)
    operator_labels = [str(x) for x in adapt.get("operators", [])]
    theta_runtime = np.asarray(runtime_input.theta_runtime, dtype=float).reshape(-1)
    theta_logical = (
        None
        if runtime_input.theta_logical is None
        else np.asarray(runtime_input.theta_logical, dtype=float).reshape(-1)
    )
    parameterization = adapt.get("parameterization", None)
    powell_chart_validation = validate_resume_powell_coordinate_chart_policy(
        payload,
        context="Resume import summary",
    )
    sr_route_profile_validation = validate_resume_sr_route_profile_contract(
        payload,
        context="Resume import summary",
    )
    phase3_response_scope_validation = (
        validate_resume_phase3_response_coordinate_scope(
            payload,
            expected_profile_request=sr_route_profile_validation[
                "artifact_profile_request"
            ],
            context="Resume import summary",
        )
    )
    _cleaned_payload, removed_obsolete_counts = (
        _drop_obsolete_admission_rollback_state(payload)
    )
    summary = {
        "schema_version": "static_hh_adapt_resume_import_v1",
        "path": str(Path(artifact_json)),
        "artifact_sha256": str(artifact_sha256),
        "source_ansatz_depth": int(_source_depth(payload, runtime_input)),
        "source_controller_round": int(_source_controller_round(payload)),
        "source_num_parameters": int(theta_runtime.size),
        "source_logical_num_parameters": int(runtime_input.base_layout.logical_parameter_count),
        "source_pool_type": (
            str(adapt.get("pool_type"))
            if adapt.get("pool_type") not in {None, ""}
            else (
                str(settings.get("adapt_pool"))
                if settings.get("adapt_pool") not in {None, ""}
                else None
            )
        ),
        "source_continuation_mode": _source_continuation_mode(payload),
        "powell_coordinate_chart_policy": powell_chart_validation[
            "resolved_policy"
        ],
        "route_profile_conformance": powell_chart_validation[
            "route_profile_conformance"
        ],
        "powell_coordinate_chart_policy_validation": powell_chart_validation,
        "sr_route_profile_request": sr_route_profile_validation[
            "artifact_profile_request"
        ],
        "sr_route_profile_contract_sha256": sr_route_profile_validation[
            "contract_sha256"
        ],
        "sr_route_profile_contract_validation": sr_route_profile_validation,
        "phase3_response_coordinate_scope": (
            phase3_response_scope_validation["resolved_scope"]
        ),
        "phase3_response_coordinate_scope_validation": (
            phase3_response_scope_validation
        ),
        "phase12_energy_model_policies": dict(
            sr_route_profile_validation["phase12_energy_model_policies"]
        ),
        "phase12_energy_model_policy_validation": dict(
            sr_route_profile_validation[
                "phase12_energy_model_policy_validation"
            ]
        ),
        "operator_count": int(len(operator_labels)),
        "operator_labels_digest": digest_jsonable(operator_labels),
        "parameterization_digest": digest_jsonable(parameterization),
        "theta_runtime_digest": digest_jsonable([float(x) for x in theta_runtime.tolist()]),
        "theta_logical_digest": (
            None if theta_logical is None else digest_jsonable([float(x) for x in theta_logical.tolist()])
        ),
        "initial_state_digest": _state_digest(payload, "initial_state"),
        "ansatz_input_state_digest": _state_digest(payload, "ansatz_input_state"),
        "runtime_loader_provenance": dict(getattr(runtime_input, "provenance", {}) or {}),
        "obsolete_admission_rollback_state_migration": {
            "schema": "obsolete_admission_rollback_state_drop_v1",
            "applied": bool(removed_obsolete_counts),
            "behavior": "ignored_and_dropped_before_resume",
            "removed_field_counts": dict(sorted(removed_obsolete_counts.items())),
        },
        "validation": dict(validation or {}),
        "no_credentials_serialized": True,
    }
    assert_no_secret_material(summary, context="resume import summary")
    return summary


def load_static_resume_source(
    artifact_json: str | Path,
    *,
    loader_mode: str = "replay_family",
    settings_overrides: Mapping[str, Any] | None = None,
) -> ResumeScaffoldSource:
    path = Path(artifact_json)
    raw_payload = _read_json_object(path)
    payload_cleaned, removed_obsolete_counts = (
        _drop_obsolete_admission_rollback_state(raw_payload)
    )
    payload = dict(payload_cleaned)
    runtime_payload: Mapping[str, Any]
    if settings_overrides:
        payload_copy = dict(payload)
        settings_copy = dict(_settings_block(payload))
        for key, value in settings_overrides.items():
            if value is None or value == "":
                continue
            settings_copy[str(key)] = value
        payload_copy["settings"] = settings_copy
        runtime_payload = payload_copy
    else:
        runtime_payload = payload
    assert_no_secret_material(raw_payload, context=f"resume artifact {path}")
    assert_no_secret_material(runtime_payload, context=f"resume runtime payload {path}")
    runtime_input = load_scaffold_runtime_input_from_payload(
        runtime_payload,
        artifact_json=path,
        loader_mode=str(loader_mode),
        generator_family="match_adapt",
        fallback_family="full_meta",
    )
    sha = file_sha256(path)
    summary = build_resume_import_summary(
        artifact_json=path,
        artifact_sha256=sha,
        payload=runtime_payload,
        runtime_input=runtime_input,
    )
    summary["obsolete_admission_rollback_state_migration"] = {
        "schema": "obsolete_admission_rollback_state_drop_v1",
        "applied": bool(removed_obsolete_counts),
        "behavior": "ignored_and_dropped_before_resume",
        "removed_field_counts": dict(sorted(removed_obsolete_counts.items())),
    }
    if settings_overrides:
        summary["loader_settings_overrides"] = {
            str(key): str(value)
            for key, value in settings_overrides.items()
            if value is not None and value != ""
        }
        summary["artifact_payload_runtime_patched"] = True
        assert_no_secret_material(summary, context="resume import summary with overrides")
    return ResumeScaffoldSource(
        artifact_json=path,
        artifact_sha256=sha,
        payload=runtime_payload,
        runtime_input=runtime_input,
        import_summary=summary,
    )


def _arg_value(args: Any, name: str, default: Any = None) -> Any:
    if args is None:
        return default
    if isinstance(args, Mapping):
        return args.get(name, default)
    return getattr(args, name, default)


def extract_formal_manifold_route_composition(
    payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Extract one internally consistent normalized FM composition identity."""

    if not isinstance(payload, Mapping):
        raise TypeError("resume payload must be a mapping.")
    adapt = _adapt_block(payload)
    settings = _settings_block(payload)
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    for owner_name, owner in (
        ("payload", payload),
        ("adapt_vqe", adapt),
        ("settings", settings),
    ):
        direct = owner.get("formal_manifold_route_composition")
        if isinstance(direct, Mapping):
            candidates.append(
                (f"{owner_name}.formal_manifold_route_composition", direct)
            )
        identity = owner.get("static_route_identity")
        if isinstance(identity, Mapping) and (
            str(identity.get("route_family", ""))
            == FORMAL_MANIFOLD_ROUTE_FAMILY
            or str(identity.get("adapt_reoptimization_route", ""))
            == FORMAL_MANIFOLD_ROUTE
        ):
            candidates.append((f"{owner_name}.static_route_identity", identity))
    if not candidates:
        for owner_name, owner in (("adapt_vqe", adapt), ("settings", settings)):
            route_profile = str(
                owner.get("formal_manifold_route_profile")
                or owner.get("route_profile")
                or ""
            ).strip().lower()
            selector_family = owner.get("candidate_selector_family")
            selector_profile = owner.get("candidate_selector_profile")
            has_selector_pair = bool(selector_family) and bool(selector_profile)
            has_registered_profile_request = route_profile not in {
                "",
                FORMAL_MANIFOLD_ROUTE_PROFILE_OFF,
                FORMAL_MANIFOLD_ROUTE,
            }
            if (
                has_selector_pair
                or (
                    has_registered_profile_request
                    and (
                        str(owner.get("route_family", ""))
                        == FORMAL_MANIFOLD_ROUTE_FAMILY
                        or str(owner.get("adapt_reoptimization_route", ""))
                        == FORMAL_MANIFOLD_ROUTE
                    )
                )
            ):
                candidates.append((owner_name, owner))
    if not candidates:
        return None
    normalized: list[tuple[str, dict[str, Any]]] = []
    for field_path, candidate in candidates:
        try:
            resolved = HistoricalFormalManifoldRouteComposition.from_mapping(
                candidate
            ).as_dict()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid Formal-Manifold route composition at {field_path}: {exc}"
            ) from exc
        normalized.append((field_path, resolved))
    reference_path, reference = normalized[0]
    disagreements = [
        field_path
        for field_path, candidate in normalized[1:]
        if candidate != reference
    ]
    if disagreements:
        raise ValueError(
            "Formal-Manifold route composition fields disagree: "
            + ", ".join([reference_path, *disagreements])
        )
    return dict(reference)


def _compare_setting(
    mismatches: list[dict[str, Any]],
    *,
    name: str,
    artifact_value: Any,
    current_value: Any,
    cast: Callable[[Any], Any] = str,
    abs_tol: float | None = None,
) -> None:
    try:
        artifact_cast = cast(artifact_value)
        current_cast = cast(current_value)
    except Exception:
        artifact_cast = artifact_value
        current_cast = current_value
    if abs_tol is not None:
        try:
            if abs(float(artifact_cast) - float(current_cast)) <= float(abs_tol):
                return
        except Exception:
            pass
    if artifact_cast != current_cast:
        mismatches.append(
            {
                "field": str(name),
                "artifact": artifact_value,
                "current": current_value,
            }
        )


def validate_static_hh_resume_source(
    source: ResumeScaffoldSource,
    *,
    args: Any | None = None,
    continuation_mode: str | None = None,
    expected_phase3_response_coordinate_scope: str | None = None,
    expected_powell_coordinate_chart_policy: str | None = None,
    expected_route_profile_conformance: str | None = None,
) -> dict[str, Any]:
    payload = source.payload
    _assert_no_unsupported_modeled_minimum_execution_checkpoint(
        payload,
        context="Static scaffold resume",
    )
    settings = _settings_block(payload)
    adapt = _adapt_block(payload)
    artifact_fm_composition = extract_formal_manifold_route_composition(payload)
    if artifact_fm_composition is not None:
        raise ValueError(
            "Static scaffold resume does not execute author-retired "
            "Formal-Manifold route checkpoints."
        )
    effective_expected_powell_policy = expected_powell_coordinate_chart_policy
    if effective_expected_powell_policy in {None, ""} and args is not None:
        requested_powell_policy = _arg_value(
            args,
            "sr_powell_coordinate_chart_policy",
            None,
        )
        requested_powell_policy_key = str(
            requested_powell_policy or ""
        ).strip().lower()
        if requested_powell_policy_key in SR_POWELL_COORDINATE_CHART_POLICY_CHOICES:
            effective_expected_powell_policy = requested_powell_policy_key
        elif requested_powell_policy_key not in {
            "",
            SR_POWELL_COORDINATE_CHART_AUTO,
        }:
            raise ValueError(
                "Static scaffold resume current Powell coordinate-chart request "
                f"is unknown: {requested_powell_policy!r}."
            )
    powell_chart_validation = validate_resume_powell_coordinate_chart_policy(
        payload,
        expected_policy=effective_expected_powell_policy,
        expected_route_profile_conformance=(
            expected_route_profile_conformance
        ),
        context="Static scaffold resume",
    )
    expected_sr_profile_request = None
    expected_sr_contract = None
    expected_sr_contract_sha256 = None
    if args is not None:
        expected_sr_profile_request = _arg_value(
            args,
            "sr_route_profile_request",
            SR_ROUTE_PROFILE_REQUEST_OFF,
        )
        expected_sr_contract = _arg_value(
            args,
            "sr_route_profile_contract",
            None,
        )
        expected_sr_contract_sha256 = _arg_value(
            args,
            "sr_route_profile_contract_sha256",
            None,
        )
    sr_route_profile_validation = validate_resume_sr_route_profile_contract(
        payload,
        expected_profile_request=expected_sr_profile_request,
        expected_contract=expected_sr_contract,
        expected_contract_sha256=expected_sr_contract_sha256,
        context="Static scaffold resume",
    )
    expected_phase3_response_scope = expected_phase3_response_coordinate_scope
    if expected_phase3_response_scope in {None, ""} and args is not None:
        expected_phase3_response_scope = _arg_value(
            args,
            "phase3_response_coordinate_scope",
            None,
        )
    phase3_response_scope_validation = (
        validate_resume_phase3_response_coordinate_scope(
            payload,
            expected_scope=expected_phase3_response_scope,
            expected_profile_request=expected_sr_profile_request,
            context="Static scaffold resume",
        )
    )
    problem = str(settings.get("problem", "")).strip().lower()
    if problem not in _STATIC_RESUME_ALLOWED_PROBLEMS:
        raise ValueError(
            "Static scaffold resume only supports "
            f"{sorted(_STATIC_RESUME_ALLOWED_PROBLEMS)} for this slice; artifact problem={problem!r}."
        )
    current_problem = _arg_value(args, "problem", problem)
    if current_problem is not None and str(current_problem).strip().lower() != problem:
        raise ValueError(
            "Resume artifact problem does not match current static request: "
            f"{problem!r} != {str(current_problem).strip().lower()!r}."
        )
    if str(_arg_value(args, "adapt_resume_mode", "scaffold_v1")) != "scaffold_v1":
        raise ValueError("Only --adapt-resume-mode scaffold_v1 is supported.")
    if adapt.get("logical_parameterization") not in {None, "", "single_term"}:
        raise ValueError("seq2p/logical-product resume artifacts are out of scope for scaffold_v1.")
    if not isinstance(adapt.get("parameterization", None), Mapping):
        raise ValueError("Structural resume requires adapt_vqe.parameterization.")
    theta_runtime = np.asarray(source.runtime_input.theta_runtime, dtype=float).reshape(-1)
    if int(theta_runtime.size) != int(source.runtime_input.base_layout.runtime_parameter_count):
        raise ValueError("Resume theta length does not match runtime layout parameter count.")
    if not isinstance(payload.get("ansatz_input_state", None), Mapping):
        raise ValueError("Structural resume requires ansatz_input_state in the source artifact.")
    if not isinstance(payload.get("initial_state", None), Mapping):
        raise ValueError("Structural resume requires initial_state in the source artifact.")

    mismatches: list[dict[str, Any]] = []
    if args is not None:
        generic_comparisons: tuple[tuple[str, str, Callable[[Any], Any]], ...] = (
            ("L", "L", int),
            ("n_ph_max", "n_ph_max", int),
            ("boson_encoding", "boson_encoding", str),
            ("ordering", "ordering", str),
            ("include_zero_point", "include_zero_point", bool),
        )
        hh_comparisons: tuple[tuple[str, str, Callable[[Any], Any]], ...] = (
            ("t", "t", float),
            ("u", "u", float),
            ("dv", "dv", float),
            ("omega0", "omega0", float),
            ("g_ep", "g_ep", float),
            ("boundary", "boundary", str),
        )
        h2o_comparisons: tuple[tuple[str, str, Callable[[Any], Any]], ...] = (
            (
                "molecular_vibronic_h2o_linear_fd_fixture_json",
                "molecular_vibronic_h2o_linear_fd_fixture_json",
                str,
            ),
        )
        comparisons = list(generic_comparisons)
        if problem == "hh":
            comparisons.extend(hh_comparisons)
        elif problem == "molecular_vibronic_h2o_linear_fd":
            comparisons.extend(h2o_comparisons)
        for setting_name, arg_name, caster in comparisons:
            if setting_name not in settings:
                # Older ADAPT artifacts did not always serialize defaults such
                # as include_zero_point.  Missing legacy defaults are recorded
                # in validation metadata instead of causing a hard failure;
                # fields that are present must still match exactly.
                continue
            _compare_setting(
                mismatches,
                name=setting_name,
                artifact_value=settings.get(setting_name),
                current_value=_arg_value(args, arg_name),
                cast=caster,
                abs_tol=(1e-10 if caster is float else None),
            )
        artifact_coordinate_scope = settings.get(
            "historical_singleton_coordinate_solve_scope"
        )
        if artifact_coordinate_scope in {None, ""}:
            artifact_overlay = settings.get(
                "historical_singleton_coordinate_trust_overlay"
            )
            if isinstance(artifact_overlay, Mapping):
                artifact_coordinate_scope = artifact_overlay.get(
                    "coordinate_solve_scope"
                )
        if artifact_coordinate_scope in {None, ""}:
            artifact_coordinate_scope = "phase3_only_v1"
        _compare_setting(
            mismatches,
            name="historical_singleton_coordinate_solve_scope",
            artifact_value=artifact_coordinate_scope,
            current_value=_arg_value(
                args,
                "historical_singleton_coordinate_solve_scope",
                "phase3_only_v1",
            ),
            cast=lambda value: str(value).strip().lower(),
        )
        artifact_pool = settings.get("adapt_pool", adapt.get("pool_type", None))
        if str(artifact_pool).strip().lower() in {"phase3_v1", "phase2_v1", "legacy_v0"}:
            # Some current-json snapshots written before structural resume
            # separated pool and continuation metadata stored the continuation
            # mode in the pool field.  Do not reject an otherwise replayable
            # scaffold on that legacy serialization error.
            artifact_pool = None
        current_pool = _arg_value(args, "adapt_pool", None)
        if current_pool not in {None, ""} and artifact_pool not in {None, ""}:
            _compare_setting(
                mismatches,
                name="adapt_pool",
                artifact_value=artifact_pool,
                current_value=current_pool,
                cast=lambda x: str(x).strip().lower(),
            )
    if mismatches:
        raise ValueError(
            "Resume artifact settings do not match current static HH request: "
            + json.dumps(mismatches[:8], sort_keys=True)
        )
    source_mode = _source_continuation_mode(payload)
    if continuation_mode not in {None, ""} and source_mode not in {None, ""}:
        if str(source_mode).strip().lower() != str(continuation_mode).strip().lower():
            raise ValueError(
                "Resume artifact continuation mode does not match current request: "
                f"{source_mode!r} != {continuation_mode!r}."
            )

    validation = {
        "schema_version": "static_hh_adapt_resume_validation_v1",
        "problem": str(problem),
        "settings_match": True,
        "continuation_mode": source_mode,
        "current_continuation_mode": continuation_mode,
        "powell_coordinate_chart_policy": powell_chart_validation[
            "resolved_policy"
        ],
        "route_profile_conformance": powell_chart_validation[
            "route_profile_conformance"
        ],
        "powell_coordinate_chart_policy_validation": powell_chart_validation,
        "sr_route_profile_request": sr_route_profile_validation[
            "artifact_profile_request"
        ],
        "sr_route_profile_contract_sha256": sr_route_profile_validation[
            "contract_sha256"
        ],
        "sr_route_profile_contract_validation": sr_route_profile_validation,
        "phase3_response_coordinate_scope": (
            phase3_response_scope_validation["resolved_scope"]
        ),
        "phase3_response_coordinate_scope_validation": (
            phase3_response_scope_validation
        ),
        "phase12_energy_model_policies": dict(
            sr_route_profile_validation["phase12_energy_model_policies"]
        ),
        "phase12_energy_model_policy_validation": dict(
            sr_route_profile_validation[
                "phase12_energy_model_policy_validation"
            ]
        ),
        "runtime_parameter_count": int(theta_runtime.size),
        "logical_parameter_count": int(source.runtime_input.base_layout.logical_parameter_count),
        "selected_term_count": int(len(source.runtime_input.selected_terms)),
        "candidate_pool_complete": bool(source.runtime_input.candidate_pool_source.candidate_pool_complete),
        "no_credentials_serialized": True,
    }
    assert_no_secret_material(validation, context="resume validation")
    return validation


def _normalize_resume_parameterization_mode(value: Any, *, field: str) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "logical_shared": "logical_shared",
        "logical_shared_v1": "logical_shared",
        "per_pauli_term": "per_pauli_term",
        "per_pauli_term_v1": "per_pauli_term",
    }
    resolved = aliases.get(raw)
    if resolved is None:
        raise ValueError(
            f"Unsupported resume parameterization mode in {field}: {value!r}. "
            "Expected logical_shared or per_pauli_term."
        )
    return str(resolved)


def _explicit_resume_parameterization_modes(
    payload: Mapping[str, Any],
) -> dict[str, str]:
    """Resolve every serialized execution-mode alias without guessing.

    Older checkpoints used one of three locations.  Multiple locations are
    allowed only when they resolve to the same execution contract.
    """

    adapt = payload.get("adapt_vqe", {})
    checkpoint = payload.get("checkpoint", {})
    raw_fields: tuple[tuple[str, Any], ...] = (
        (
            "adapt_vqe.parameterization_execution_mode",
            adapt.get("parameterization_execution_mode")
            if isinstance(adapt, Mapping)
            else None,
        ),
        (
            "adapt_vqe.parameterization_mode",
            adapt.get("parameterization_mode")
            if isinstance(adapt, Mapping)
            else None,
        ),
        (
            "checkpoint.parameterization_execution_mode",
            checkpoint.get("parameterization_execution_mode")
            if isinstance(checkpoint, Mapping)
            else None,
        ),
        (
            "checkpoint.parameterization_mode",
            checkpoint.get("parameterization_mode")
            if isinstance(checkpoint, Mapping)
            else None,
        ),
    )
    resolved: dict[str, str] = {}
    for field, value in raw_fields:
        if value is None or value == "":
            continue
        resolved[str(field)] = _normalize_resume_parameterization_mode(
            value,
            field=str(field),
        )
    return resolved


def _replay_resume_in_expected_parameterization(
    *,
    selected_ops: Sequence[AnsatzTerm],
    selected_layout: AnsatzParameterLayout,
    theta_runtime: np.ndarray,
    theta_logical: np.ndarray | None,
    psi_ref: np.ndarray,
    psi_initial: np.ndarray,
    expected_parameterization_mode: str,
    atol: float = 1.0e-10,
) -> dict[str, Any]:
    mode = _normalize_resume_parameterization_mode(
        expected_parameterization_mode,
        field="expected_parameterization_mode",
    )
    theta_exec = np.asarray(theta_runtime, dtype=float).reshape(-1)
    if mode == "logical_shared":
        if theta_logical is None:
            raise ValueError(
                "logical_shared resume cannot replay without logical theta."
            )
        theta_exec = np.asarray(theta_logical, dtype=float).reshape(-1)
    if not np.all(np.isfinite(theta_exec)):
        raise ValueError("Resume theta contains a non-finite value.")

    try:
        executor = CompiledAnsatzExecutor(
            list(selected_ops),
            coefficient_tolerance=float(selected_layout.coefficient_tolerance),
            ignore_identity=bool(selected_layout.ignore_identity),
            sort_terms=(str(selected_layout.term_order).strip().lower() == "sorted"),
            parameterization_mode=mode,
            parameterization_layout=selected_layout,
        )
        replayed = np.asarray(
            executor.prepare_state(
                theta_exec,
                np.asarray(psi_ref, dtype=complex).reshape(-1),
            ),
            dtype=complex,
        ).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Resume scaffold cannot execute under the expected parameterization "
            f"mode {mode!r}: {exc}"
        ) from exc

    expected = np.asarray(psi_initial, dtype=complex).reshape(-1)
    if int(replayed.size) != int(expected.size):
        raise ValueError(
            "Resume expected-mode replay state dimension mismatch: "
            f"replayed={replayed.size}, checkpoint={expected.size}."
        )
    replay_norm = float(np.linalg.norm(replayed))
    checkpoint_norm = float(np.linalg.norm(expected))
    if not (math.isfinite(replay_norm) and math.isfinite(checkpoint_norm)):
        raise ValueError("Resume expected-mode replay produced a non-finite norm.")
    if replay_norm <= float(atol) or checkpoint_norm <= float(atol):
        raise ValueError("Resume expected-mode replay cannot compare a zero-norm state.")

    overlap = complex(np.vdot(expected, replayed))
    alignment = (
        complex(np.exp(-1.0j * np.angle(overlap)))
        if abs(overlap) > float(atol)
        else 1.0 + 0.0j
    )
    replay_error = float(np.linalg.norm(expected - alignment * replayed))
    if not math.isfinite(replay_error) or replay_error > float(atol):
        raise ValueError(
            "Resume scaffold strict expected-mode replay failed up to global "
            f"phase: mode={mode!r}, l2_error={replay_error:.3e}, "
            f"tolerance={float(atol):.3e}."
        )
    return {
        "schema_version": "static_adapt_resume_expected_mode_replay_v1",
        "passed": True,
        "parameterization_mode": str(mode),
        "global_phase_invariant": True,
        "l2_error_up_to_global_phase": float(replay_error),
        "tolerance": float(atol),
        "checkpoint_state_norm": float(checkpoint_norm),
        "replayed_state_norm": float(replay_norm),
        "overlap_abs": float(abs(overlap)),
    }


def _generator_component_signature(
    term: AnsatzTerm,
) -> tuple[tuple[int, str, float, float], ...] | None:
    polynomial = getattr(term, "polynomial", None)
    term_provider = getattr(polynomial, "return_polynomial", None)
    if not callable(term_provider):
        return None
    components: list[tuple[int, str, float, float]] = []
    for component in term_provider():
        word_builder = getattr(component, "pw2strng", None)
        nq_builder = getattr(component, "nqubit", None)
        if not callable(word_builder) or not callable(nq_builder):
            return None
        word = str(word_builder()).strip().lower()
        nq = int(nq_builder())
        coefficient = complex(
            getattr(component, "p_coeff", getattr(component, "coeff", 0.0))
        )
        if (
            len(word) != nq
            or any(symbol not in {"e", "x", "y", "z"} for symbol in word)
            or not math.isfinite(float(coefficient.real))
            or not math.isfinite(float(coefficient.imag))
        ):
            return None
        components.append(
            (
                int(nq),
                word,
                float(coefficient.real),
                float(coefficient.imag),
            )
        )
    return tuple(sorted(components))


def _generator_semantic_signature(
    term: AnsatzTerm,
) -> tuple[str, tuple[tuple[str, float], ...]]:
    execution_mode = str(
        getattr(term, "execution_mode", "termwise_product")
        or "termwise_product"
    ).strip().lower()
    return execution_mode, _polynomial_signature(term.polynomial)


def _pauli_words_commute_for_resume(left: str, right: str) -> bool:
    if len(left) != len(right):
        return False
    anticommuting_positions = sum(
        1
        for left_symbol, right_symbol in zip(left, right)
        if left_symbol != "e"
        and right_symbol != "e"
        and left_symbol != right_symbol
    )
    return bool(anticommuting_positions % 2 == 0)


def _load_verified_active_prefix_sidecar(
    source: ResumeScaffoldSource,
) -> dict[str, Any]:
    """Load the exact signed prefix from a sidecar or current checkpoint.

    Old current checkpoints omitted per-operator execution and runtime-split
    metadata from their serialized parameterization.  The sibling terminal
    result retains the exact signed post-admission prefix checkpoint.  A
    no-refit resume must use that record rather than infer semantics from a
    label or from current pool defaults.  Newer nonterminal checkpoints carry
    the same signed record in the winning history row.  That embedded record is
    authoritative when its digest, controller round, operator order,
    parameters, replay receipt, and projective-state fingerprint all close.
    """

    sidecar_path = source.artifact_json.with_name(
        "signed_active_prefix_checkpoint.json"
    )
    if sidecar_path.is_file():
        sidecar_payload = _read_json_object(sidecar_path)
        if str(sidecar_payload.get("schema", "")) != (
            "static_adapt_signed_active_prefix_resume_sidecar_v1"
        ):
            raise ValueError(
                "Compact signed-prefix resume sidecar schema is not supported."
            )
        source_result_json = sidecar_payload.get("source_result_json")
        source_result_sha256 = str(
            sidecar_payload.get("source_result_sha256", "")
        )
        if (
            not isinstance(source_result_json, str)
            or not source_result_json.strip()
            or re.fullmatch(r"[0-9a-f]{64}", source_result_sha256) is None
        ):
            raise ValueError(
                "Compact signed-prefix resume sidecar lacks typed source-result provenance."
            )
        checkpoint_raw = sidecar_payload.get("checkpoint")
        prefix_source = "sibling_signed_prefix_sidecar_v1"
        prefix_json = sidecar_path
        prefix_sha256 = file_sha256(sidecar_path)
    else:
        adapt_payload = source.payload.get("adapt_vqe", {})
        history_raw = (
            adapt_payload.get("history", [])
            if isinstance(adapt_payload, Mapping)
            else []
        )
        last_history = (
            history_raw[-1]
            if isinstance(history_raw, Sequence)
            and not isinstance(history_raw, (str, bytes, bytearray))
            and len(history_raw) > 0
            and isinstance(history_raw[-1], Mapping)
            else None
        )
        checkpoint_raw = (
            last_history.get("active_prefix_checkpoint")
            if isinstance(last_history, Mapping)
            else None
        )
        if not isinstance(checkpoint_raw, Mapping):
            raise ValueError(
                "Verified no-refit resume requires either the compact sibling "
                "signed-prefix sidecar or an exact signed active-prefix "
                "checkpoint in the final winning history row."
            )
        observed_source_sha256 = file_sha256(source.artifact_json)
        if (
            re.fullmatch(r"[0-9a-f]{64}", str(source.artifact_sha256)) is None
            or observed_source_sha256 != str(source.artifact_sha256)
        ):
            raise ValueError(
                "Embedded signed-prefix checkpoint source SHA-256 mismatch."
            )
        source_result_json = str(source.artifact_json)
        source_result_sha256 = str(source.artifact_sha256)
        prefix_source = "embedded_current_winning_history_v1"
        prefix_json = source.artifact_json
        prefix_sha256 = observed_source_sha256
    source_labels = tuple(
        str(term.label) for term in source.runtime_input.selected_terms
    )
    source_depth = int(len(source_labels))
    source_controller_round = int(_source_controller_round(source.payload))
    if not isinstance(checkpoint_raw, Mapping):
        raise ValueError(
            "Signed-prefix provenance lacks its exact checkpoint."
        )
    checkpoint = dict(checkpoint_raw)
    if (
        int(checkpoint.get("outer_iteration", -1)) != source_controller_round
        or str(checkpoint.get("checkpoint_kind", ""))
        != "post_admission_prune"
    ):
        raise ValueError(
            "Compact signed-prefix resume sidecar does not contain the requested "
            f"round-{source_controller_round} post-admission checkpoint."
        )
    if str(checkpoint.get("schema", "")) != (
        "paper_i_signed_active_prefix_checkpoint_v1"
    ):
        raise ValueError("Signed-prefix checkpoint schema is not supported.")
    expected_checkpoint_sha = str(
        checkpoint.get("checkpoint_sha256", "")
    )
    hash_input = dict(checkpoint)
    hash_input.pop("checkpoint_sha256", None)
    observed_checkpoint_sha = digest_jsonable(hash_input)
    if (
        not expected_checkpoint_sha
        or observed_checkpoint_sha != expected_checkpoint_sha
    ):
        raise ValueError(
            "Signed-prefix checkpoint SHA-256 mismatch: "
            f"embedded={expected_checkpoint_sha!r}, "
            f"observed={observed_checkpoint_sha!r}."
        )
    checkpoint_labels = tuple(
        str(value)
        for value in checkpoint.get("ordered_active_operator_labels", [])
    )
    adapt_labels = tuple(
        str(value) for value in source.payload.get("adapt_vqe", {}).get(
            "operators", []
        )
    ) if isinstance(source.payload.get("adapt_vqe", {}), Mapping) else tuple()
    if (
        checkpoint_labels != source_labels
        or checkpoint_labels != adapt_labels
        or int(checkpoint.get("active_ansatz_depth", -1)) != source_depth
    ):
        raise ValueError(
            "Signed-prefix checkpoint operator order/depth does not match current.json."
        )
    runtime_theta = np.asarray(
        checkpoint.get("signed_unwrapped_runtime_parameters", []),
        dtype=float,
    ).reshape(-1)
    source_runtime_theta = np.asarray(
        source.runtime_input.theta_runtime, dtype=float
    ).reshape(-1)
    logical_theta = np.asarray(
        checkpoint.get("signed_unwrapped_logical_parameters", []),
        dtype=float,
    ).reshape(-1)
    source_logical_theta = np.asarray(
        source.runtime_input.theta_logical, dtype=float
    ).reshape(-1)
    if (
        runtime_theta.shape != source_runtime_theta.shape
        or logical_theta.shape != source_logical_theta.shape
        or not np.allclose(runtime_theta, source_runtime_theta, atol=1e-12, rtol=0.0)
        or not np.allclose(logical_theta, source_logical_theta, atol=1e-12, rtol=0.0)
    ):
        raise ValueError(
            "Signed-prefix checkpoint parameters do not match current.json."
        )
    if str(checkpoint.get("parameterization_execution_mode", "")) != str(
        source.payload.get("adapt_vqe", {}).get(
            "parameterization_execution_mode", ""
        )
    ):
        raise ValueError(
            "Signed-prefix checkpoint parameterization mode does not match current.json."
        )
    strict_replay = checkpoint.get("strict_replay", {})
    state_sector = checkpoint.get("state_sector_contract", {})
    active_sector = checkpoint.get("active_generator_sector_contract", {})
    if not (
        isinstance(strict_replay, Mapping)
        and strict_replay.get("passed") is True
        and isinstance(state_sector, Mapping)
        and state_sector.get("passed") is True
        and isinstance(active_sector, Mapping)
        and active_sector.get("passed_with_parameterization") is True
    ):
        raise ValueError(
            "Signed-prefix checkpoint did not preserve passing replay/state/generator audits."
        )
    expected_fingerprint = str(
        checkpoint.get("projective_state_fingerprint", "")
    )
    observed_fingerprint = projective_state_fingerprint(
        np.asarray(source.runtime_input.psi_initial, dtype=complex).reshape(-1)
    )
    if expected_fingerprint != observed_fingerprint:
        raise ValueError(
            "Signed-prefix checkpoint projective state does not match current.json."
        )

    operator_rows = checkpoint.get("ordered_active_operators", [])
    if not isinstance(operator_rows, Sequence) or isinstance(
        operator_rows, (str, bytes)
    ) or len(operator_rows) != source_depth:
        raise ValueError(
            "Signed-prefix checkpoint ordered_active_operators is incomplete."
        )
    terms: list[AnsatzTerm] = []
    contracts: list[dict[str, Any]] = []
    for position, raw_row in enumerate(operator_rows):
        if not isinstance(raw_row, Mapping):
            raise ValueError("Signed-prefix operator row is not a mapping.")
        label = str(raw_row.get("label", ""))
        if (
            int(raw_row.get("active_position", -1)) != position
            or label != checkpoint_labels[position]
        ):
            raise ValueError(
                "Signed-prefix operator row position/label is inconsistent."
            )
        execution_mode = str(raw_row.get("execution_mode", ""))
        if execution_mode not in {"grouped_exact", "termwise_product"}:
            raise ValueError(
                f"Signed-prefix operator has unsupported execution mode {execution_mode!r}."
            )
        serialized_terms = raw_row.get(
            "serialized_terms_exyz_in_execution_order", []
        )
        if not isinstance(serialized_terms, Sequence) or isinstance(
            serialized_terms, (str, bytes)
        ) or len(serialized_terms) == 0:
            raise ValueError(
                f"Signed-prefix operator {label!r} has no serialized terms."
            )
        polynomial = PauliPolynomial("JW")
        normalized_terms: list[dict[str, Any]] = []
        for raw_term in serialized_terms:
            if not isinstance(raw_term, Mapping):
                raise ValueError("Signed-prefix serialized term is not a mapping.")
            coefficient_imag = float(raw_term.get("coeff_im", 0.0))
            coefficient_real = float(raw_term.get("coeff_re", 0.0))
            nq = int(raw_term.get("nq", 0))
            pauli = str(raw_term.get("pauli_exyz", "")).strip().lower()
            if (
                abs(coefficient_imag) > 1e-14
                or not math.isfinite(coefficient_real)
                or nq <= 0
                or len(pauli) != nq
                or any(symbol not in {"e", "x", "y", "z"} for symbol in pauli)
            ):
                raise ValueError(
                    f"Signed-prefix operator {label!r} has an invalid serialized term."
                )
            polynomial.add_term(
                PauliTerm(nq, ps=pauli, pc=coefficient_real)
            )
            normalized_terms.append(
                {
                    "pauli_exyz": pauli,
                    "coeff_re": coefficient_real,
                    "coeff_im": coefficient_imag,
                    "nq": nq,
                }
            )
        polynomial._reduce()
        terms.append(
            AnsatzTerm(
                label=label,
                polynomial=polynomial,
                execution_mode=execution_mode,
            )
        )
        runtime_split = raw_row.get("runtime_split")
        runtime_split = (
            dict(runtime_split)
            if isinstance(runtime_split, Mapping)
            else None
        )
        padding_lineage = raw_row.get("route_a_child_padding_lineage")
        padding_lineage = (
            dict(padding_lineage)
            if isinstance(padding_lineage, Mapping)
            else None
        )
        symmetry_gate = (
            runtime_split.get("symmetry_gate", {})
            if isinstance(runtime_split, Mapping)
            else {}
        )
        hard_guard = bool(
            isinstance(symmetry_gate, Mapping)
            and symmetry_gate.get("hard_guard_required") is True
            and symmetry_gate.get("hard_guard_present") is True
            and symmetry_gate.get("passed") is True
        )
        if isinstance(runtime_split, Mapping):
            recommended_mode = runtime_split.get(
                "recommended_execution_mode"
            )
            if (
                recommended_mode not in {None, ""}
                and str(recommended_mode) != execution_mode
            ):
                raise ValueError(
                    "Signed-prefix runtime-split execution mode disagrees "
                    f"with ordered operator {label!r}."
                )
        if isinstance(padding_lineage, Mapping):
            lineage_mode = padding_lineage.get("selected_execution_mode")
            if (
                lineage_mode not in {None, ""}
                and str(lineage_mode) != execution_mode
            ):
                raise ValueError(
                    "Signed-prefix padding-lineage execution mode disagrees "
                    f"with ordered operator {label!r}."
                )
        contract_provenance = {
            "source_result_json": str(source_result_json),
            "source_result_sha256": source_result_sha256,
            "resume_prefix_source": str(prefix_source),
            "resume_prefix_json": str(prefix_json),
            "resume_prefix_sha256": str(prefix_sha256),
            "checkpoint_sha256": expected_checkpoint_sha,
            "active_position": position,
        }
        if prefix_source == "sibling_signed_prefix_sidecar_v1":
            contract_provenance.update(
                {
                    "resume_sidecar_json": str(sidecar_path),
                    "resume_sidecar_sha256": str(prefix_sha256),
                }
            )
        contracts.append(
            {
                "schema": "resume_selected_generator_contract_v1",
                "generator_id": raw_row.get("generator_id"),
                "label": label,
                "parent_generator_id": raw_row.get("parent_generator_id"),
                "execution_mode": execution_mode,
                "symmetry_spec": {
                    "particle_number_mode": (
                        "preserving" if hard_guard else "off"
                    ),
                    "spin_sector_mode": (
                        "preserving" if hard_guard else "off"
                    ),
                    "hard_guard": hard_guard,
                    "source": "signed_active_prefix_checkpoint_v1",
                },
                "compile_metadata": {
                    "serialized_terms_exyz": normalized_terms,
                    "runtime_terms_exyz": normalized_terms,
                    "runtime_split": runtime_split,
                    "route_a_child_padding_lineage": padding_lineage,
                },
                "resume_contract_provenance": contract_provenance,
            }
        )
    expected_guarded_labels = [
        str(value)
        for value in active_sector.get(
            "fixed_sector_guarded_generator_labels", []
        )
    ]
    observed_guarded_labels = [
        str(contract["label"])
        for contract in contracts
        if bool(contract["symmetry_spec"]["hard_guard"])
    ]
    if (
        int(
            active_sector.get(
                "fixed_sector_guarded_generator_count", -1
            )
        )
        != len(expected_guarded_labels)
        or observed_guarded_labels != expected_guarded_labels
    ):
        raise ValueError(
            "Signed-prefix hard-guard label order/multiplicity does not close "
            "against active_generator_sector_contract."
        )
    return {
        "terms": tuple(terms),
        "contracts": tuple(contracts),
        "provenance": {
            "schema": "verified_resume_signed_active_prefix_v2",
            "source_result_json": str(source_result_json),
            "source_result_sha256": source_result_sha256,
            "resume_prefix_source": str(prefix_source),
            "resume_prefix_json": str(prefix_json),
            "resume_prefix_sha256": str(prefix_sha256),
            "checkpoint_sha256": expected_checkpoint_sha,
            "outer_iteration": source_controller_round,
            "operator_count": len(terms),
            "projective_state_fingerprint": expected_fingerprint,
        },
    }


def _load_verified_singleton_resume_sidecar(
    source: ResumeScaffoldSource,
    *,
    context: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Load legacy v1 or authenticate the active writer's no-cycle v2 proof."""

    pointer_raw = _adapt_block(source.payload).get(
        "verified_singleton_resume_sidecar"
    )
    if pointer_raw is None:
        adapt = _adapt_block(source.payload)
        active_v2_contract_sha256 = (
            _active_verified_resume_contract_sha256()
        )
        if (
            str(adapt.get("sr_route_profile_request", ""))
            == SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1
            and str(adapt.get("sr_route_profile_contract_sha256", ""))
            == active_v2_contract_sha256
        ):
            raise ValueError(
                f"{context} active fd5ec3fa route requires an authenticated "
                "v2 resume-sidecar pointer; legacy v1 downgrade is forbidden."
            )
        sidecar_path = source.artifact_json.with_name(
            "signed_active_prefix_checkpoint.json"
        )
        if not sidecar_path.is_file():
            raise ValueError(
                f"{context} requires sibling "
                "signed_active_prefix_checkpoint.json."
            )
        sidecar = _read_json_object(sidecar_path)
        if str(sidecar.get("schema", "")) != (
            "static_adapt_signed_active_prefix_resume_sidecar_v1"
        ):
            raise ValueError(f"{context} legacy sidecar schema is unsupported.")
        source_result_json = sidecar.get("source_result_json")
        source_result_sha256 = str(
            sidecar.get("source_result_sha256", "")
        )
        if (
            not isinstance(source_result_json, str)
            or not source_result_json.strip()
            or re.fullmatch(r"[0-9a-f]{64}", source_result_sha256) is None
        ):
            raise ValueError(f"{context} lacks source-result provenance.")
        return sidecar_path, sidecar, {
            "schema": "verified_singleton_resume_source_binding_v1",
            "status": "legacy_external_source_provenance",
            "required": False,
            "no_credentials_serialized": True,
        }
    if not isinstance(pointer_raw, Mapping):
        raise ValueError(f"{context} v2 sidecar pointer is invalid.")
    pointer = dict(pointer_raw)
    if (
        str(pointer.get("schema", ""))
        != "static_adapt_verified_singleton_resume_sidecar_pointer_v1"
        or pointer.get("enabled") is not True
        or str(pointer.get("status", "")) != "complete"
        or str(pointer.get("sidecar_schema", ""))
        != "static_adapt_signed_active_prefix_resume_sidecar_v2"
        or str(pointer.get("source_projection_schema", ""))
        != "static_adapt_verified_singleton_resume_source_projection_v1"
        or pointer.get("no_credentials_serialized") is not True
    ):
        raise ValueError(f"{context} v2 sidecar pointer contract is unsupported.")
    relative_path = str(pointer.get("path", "")).strip()
    if (
        not relative_path
        or Path(relative_path).is_absolute()
        or Path(relative_path).name != relative_path
    ):
        raise ValueError(f"{context} v2 sidecar pointer path is not a sibling.")
    expected_sidecar_sha256 = str(pointer.get("sha256", ""))
    expected_projection_sha256 = str(
        pointer.get("source_projection_sha256", "")
    )
    if (
        re.fullmatch(r"[0-9a-f]{64}", expected_sidecar_sha256) is None
        or re.fullmatch(r"[0-9a-f]{64}", expected_projection_sha256) is None
    ):
        raise ValueError(f"{context} v2 sidecar pointer hashes are invalid.")
    sidecar_path = source.artifact_json.with_name(relative_path)
    try:
        sidecar_bytes = sidecar_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{context} v2 sidecar cannot be read.") from exc
    if hashlib.sha256(sidecar_bytes).hexdigest() != expected_sidecar_sha256:
        raise ValueError(f"{context} v2 sidecar SHA-256 mismatch.")
    try:
        sidecar_raw = json.loads(sidecar_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} v2 sidecar is not valid JSON.") from exc
    if not isinstance(sidecar_raw, Mapping):
        raise ValueError(f"{context} v2 sidecar is not an object.")
    sidecar = dict(sidecar_raw)
    if str(sidecar.get("schema", "")) != str(pointer["sidecar_schema"]):
        raise ValueError(f"{context} v2 sidecar schema disagrees with pointer.")
    if str(sidecar.get("source_result_digest_scope", "")) != str(
        pointer["source_projection_schema"]
    ):
        raise ValueError(
            f"{context} v2 sidecar projection scope disagrees with pointer."
        )
    source_result_json = sidecar.get("source_result_json")
    source_result_sha256 = str(sidecar.get("source_result_sha256", ""))
    if not isinstance(source_result_json, str) or not source_result_json.strip():
        raise ValueError(f"{context} lacks source-result path provenance.")
    recorded_path = Path(source_result_json).expanduser()
    if not recorded_path.is_absolute():
        recorded_path = source.artifact_json.parent / recorded_path
    try:
        recorded_resolved = recorded_path.resolve(strict=False)
        artifact_resolved = source.artifact_json.resolve(strict=False)
    except OSError as exc:
        raise ValueError(
            f"{context} cannot resolve source-result path provenance."
        ) from exc
    if recorded_resolved != artifact_resolved:
        raise ValueError(
            f"{context} source-result path does not name the loaded artifact."
        )
    source_projection = _read_json_object(source.artifact_json)
    projection_adapt = _adapt_block(source_projection)
    projection_adapt.pop("verified_singleton_resume_sidecar", None)
    source_projection["adapt_vqe"] = projection_adapt
    observed_projection_sha256 = digest_jsonable(source_projection)
    if (
        source_result_sha256 != expected_projection_sha256
        or observed_projection_sha256 != expected_projection_sha256
    ):
        raise ValueError(
            f"{context} v2 source-projection SHA-256 mismatch."
        )
    binding = {
        "schema": "verified_singleton_resume_source_binding_v1",
        "status": "authenticated_v2_pointer_and_source_projection",
        "required": True,
        "source_result_json": str(source_result_json),
        "source_result_sha256": source_result_sha256,
        "resume_sidecar_json": str(sidecar_path),
        "resume_sidecar_sha256": expected_sidecar_sha256,
        "no_credentials_serialized": True,
    }
    assert_no_secret_material(
        binding,
        context=f"{context} v2 source binding",
    )
    return sidecar_path, sidecar, binding


def _load_verified_singleton_controller_state(
    source: ResumeScaffoldSource,
    *,
    authenticated_pruned_lineage: bool = False,
) -> dict[str, Any]:
    """Load the exact nonbeam controller state omitted from ``current.json``.

    The preserved round checkpoint contains the prefix and energies but legacy
    ``current.json`` did not serialize the maturity-controller state or the
    Phase-I residual-lane flag.  The compact signed-prefix sidecar is
    source-result authenticated by the locked bundle, so it also carries the
    unique final-round controller snapshot and the source history fields that
    prove the residual lane remained closed.
    """

    context = "Verified singleton controller resume"
    sidecar_path, sidecar, source_binding = (
        _load_verified_singleton_resume_sidecar(
            source,
            context=context,
        )
    )
    source_result_json = sidecar.get("source_result_json")
    source_result_sha256 = str(sidecar.get("source_result_sha256", ""))

    adapt = _adapt_block(source.payload)
    try:
        controller_round = int(source.payload["checkpoint"]["depth"])
        ansatz_depth = int(adapt["ansatz_depth"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            f"{context} cannot resolve current checkpoint round/depth."
        ) from None
    depth_relation_valid = (
        0 < ansatz_depth <= controller_round
        if bool(authenticated_pruned_lineage)
        else ansatz_depth == controller_round
    )
    if controller_round <= 0 or not depth_relation_valid:
        raise ValueError(f"{context} checkpoint round/depth is inconsistent.")

    snapshot_raw = sidecar.get("controller_snapshot")
    if not isinstance(snapshot_raw, Mapping):
        raise ValueError(f"{context} lacks controller_snapshot.")
    snapshot = dict(snapshot_raw)
    expected_snapshot_digest = str(
        sidecar.get("controller_snapshot_sha256", "")
    )
    observed_snapshot_digest = digest_jsonable(snapshot)
    if (
        re.fullmatch(r"[0-9a-f]{64}", expected_snapshot_digest) is None
        or expected_snapshot_digest != observed_snapshot_digest
    ):
        raise ValueError(f"{context} controller snapshot digest mismatch.")
    expected_snapshot_fields = {
        str(field.name) for field in fields(PhaseControllerSnapshot)
    }
    if set(snapshot) != expected_snapshot_fields:
        missing = sorted(expected_snapshot_fields.difference(snapshot))
        extra = sorted(set(snapshot).difference(expected_snapshot_fields))
        raise ValueError(
            f"{context} controller snapshot fields disagree; "
            f"missing={missing}, extra={extra}."
        )
    try:
        typed_snapshot = PhaseControllerSnapshot(**snapshot)
    except (TypeError, ValueError):
        raise ValueError(
            f"{context} controller snapshot is not typed."
        ) from None
    if str(typed_snapshot.snapshot_version) != "phase123_controller_maturity_v2":
        raise ValueError(f"{context} maturity snapshot version is unsupported.")
    if (
        int(typed_snapshot.step_index) != controller_round - 1
        or int(typed_snapshot.depth_local) != controller_round - 1
        or int(typed_snapshot.depth_left) != 1
    ):
        raise ValueError(
            f"{context} snapshot is not the unique pre-round-{controller_round} state."
        )
    numeric_snapshot_fields = (
        "runway_ratio",
        "early_coordinate",
        "late_coordinate",
        "frontier_ratio",
        "u_stag",
        "m_t",
        "s_t",
        "rho_t",
        "gamma_t",
        "u_front",
        "n_rem_hat",
        "useful_horizon",
        "runway_fraction",
        "H_t",
        "depth_runway_ratio",
        "n_rem_low",
        "n_rem_high",
        "confidence_ratio",
    )
    if any(
        not math.isfinite(float(getattr(typed_snapshot, field)))
        for field in numeric_snapshot_fields
    ):
        raise ValueError(f"{context} snapshot contains nonfinite telemetry.")
    phase_live = dict(typed_snapshot.phase_live)
    phase_null_streaks = dict(typed_snapshot.phase_null_streaks)
    phase_null_reasons = dict(typed_snapshot.phase_null_reasons)
    if (
        set(phase_live) != {"phase1", "phase2", "phase3"}
        or any(type(value) is not bool for value in phase_live.values())
        or set(phase_null_streaks) != {"phase2", "phase3"}
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or int(value) < 0
            for value in phase_null_streaks.values()
        )
        or set(phase_null_reasons) != {"phase1", "phase2", "phase3"}
        or any(
            not isinstance(value, str)
            for value in phase_null_reasons.values()
        )
        or int(typed_snapshot.terminal_phase) not in {1, 2, 3}
    ):
        raise ValueError(f"{context} passive phase-state shape is invalid.")
    retired_reason = "phase_live_retired_non_authoritative"
    if set(phase_null_reasons.values()) == {retired_reason}:
        if (
            phase_live != {"phase1": True, "phase2": True, "phase3": True}
            or phase_null_streaks != {"phase2": 0, "phase3": 0}
        ):
            raise ValueError(
                f"{context} retired passive phase-state values are invalid."
            )

    controller_state_raw = sidecar.get("controller_state")
    if not isinstance(controller_state_raw, Mapping):
        raise ValueError(f"{context} lacks controller_state evidence.")
    controller_state = dict(controller_state_raw)
    if str(controller_state.get("schema", "")) != (
        "static_adapt_singleton_controller_resume_state_v1"
    ):
        raise ValueError(f"{context} controller_state schema is unsupported.")
    evidence_raw = controller_state.get("source_history_row_evidence")
    if not isinstance(evidence_raw, Mapping):
        raise ValueError(f"{context} lacks source history-row evidence.")
    evidence = dict(evidence_raw)
    expected_evidence_digest = str(
        controller_state.get("source_history_row_evidence_sha256", "")
    )
    if (
        re.fullmatch(r"[0-9a-f]{64}", expected_evidence_digest) is None
        or digest_jsonable(evidence) != expected_evidence_digest
    ):
        raise ValueError(f"{context} history-row evidence digest mismatch.")
    expected_evidence = {
        "depth": controller_round,
        "drop_policy_enabled": False,
        "drop_plateau_hits": 0,
        "stage_name": "core",
        "stage_transition_reason": "stay_core",
        "controller_snapshot_count": 1,
        "selected_feature_row_index": 0,
    }
    if evidence != expected_evidence:
        raise ValueError(
            f"{context} cannot prove the Phase-I residual lane stayed closed."
        )
    try:
        state_round = int(controller_state.get("controller_round"))
        source_max_depth = int(controller_state.get("source_max_depth"))
    except (TypeError, ValueError):
        raise ValueError(f"{context} controller_state round is invalid.") from None
    if (
        state_round != controller_round
        or source_max_depth != controller_round
        or controller_state.get("phase1_residual_opened") is not False
        or str(controller_state.get("phase1_stage_name", "")) != "core"
    ):
        raise ValueError(f"{context} controller_state is inconsistent.")

    provenance = {
        "schema": "verified_singleton_controller_resume_state_v1",
        "source_result_json": str(source_result_json),
        "source_result_sha256": source_result_sha256,
        "resume_sidecar_json": str(sidecar_path),
        "resume_sidecar_sha256": file_sha256(sidecar_path),
        "controller_snapshot_sha256": expected_snapshot_digest,
        "source_history_row_evidence_sha256": expected_evidence_digest,
        "controller_round": controller_round,
        "source_max_depth": source_max_depth,
        "phase1_residual_opened": False,
        "phase1_stage_name": "core",
        "drop_policy_enabled": False,
        "drop_plateau_hits": 0,
        "source_binding": source_binding,
        "no_credentials_serialized": True,
    }
    assert_no_secret_material(
        provenance,
        context="verified singleton controller resume provenance",
    )
    return {
        "phase1_residual_opened": False,
        "phase1_stage_name": "core",
        "controller_snapshot": asdict(typed_snapshot),
        "provenance": provenance,
    }


def _load_verified_singleton_selection_state(
    source: ResumeScaffoldSource,
) -> dict[str, Any]:
    """Load the exact accepted-parent counters for a singleton continuation."""

    context = "Verified singleton selection-count resume"
    sidecar_path, sidecar, source_binding = (
        _load_verified_singleton_resume_sidecar(
            source,
            context=context,
        )
    )
    state_raw = sidecar.get("selection_state")
    if not isinstance(state_raw, Mapping):
        raise ValueError(f"{context} lacks selection_state evidence.")
    state = dict(state_raw)
    if str(state.get("schema", "")) != (
        "static_adapt_singleton_selection_count_resume_state_v1"
    ):
        raise ValueError(f"{context} selection_state schema is unsupported.")
    adapt = _adapt_block(source.payload)
    try:
        controller_round = int(source.payload["checkpoint"]["depth"])
        source_pool_size = int(adapt["pool_size"])
        state_round = int(state["controller_round"])
        state_pool_size = int(state["pool_size"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{context} round/pool size is invalid.") from None
    if (
        controller_round <= 0
        or source_pool_size <= 0
        or state_round != controller_round
        or state_pool_size != source_pool_size
    ):
        raise ValueError(f"{context} round/pool size disagrees.")

    ordered_raw = state.get("ordered_parent_pool_indices")
    if not isinstance(ordered_raw, Sequence) or isinstance(
        ordered_raw, (str, bytes, bytearray)
    ):
        raise ValueError(f"{context} parent-index sequence is invalid.")
    try:
        ordered_parent = tuple(int(value) for value in ordered_raw)
    except (TypeError, ValueError):
        raise ValueError(f"{context} parent-index sequence is invalid.") from None
    if (
        len(ordered_parent) != controller_round
        or any(index < 0 or index >= source_pool_size for index in ordered_parent)
    ):
        raise ValueError(f"{context} parent-index sequence is incomplete/out of range.")
    parent_digest = str(
        state.get("ordered_parent_pool_indices_sha256", "")
    )
    if (
        re.fullmatch(r"[0-9a-f]{64}", parent_digest) is None
        or digest_jsonable(list(ordered_parent)) != parent_digest
    ):
        raise ValueError(f"{context} parent-index digest mismatch.")
    record_counts_raw = state.get("selected_feature_row_count_per_round")
    if not isinstance(record_counts_raw, Sequence) or isinstance(
        record_counts_raw, (str, bytes, bytearray)
    ):
        raise ValueError(f"{context} selected-feature-row counts are invalid.")
    try:
        record_counts = tuple(int(value) for value in record_counts_raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"{context} selected-feature-row counts are invalid."
        ) from None
    if record_counts != tuple(1 for _ in range(controller_round)):
        raise ValueError(
            f"{context} requires exactly one selected feature row per round."
        )

    if state.get("seq2p_logical_mode") is not False:
        raise ValueError(
            f"{context} requires explicit non-seq2p full-meta route evidence."
        )
    logical_raw = state.get("ordered_logical_candidate_indices")
    if not isinstance(logical_raw, Sequence) or isinstance(
        logical_raw, (str, bytes, bytearray)
    ):
        raise ValueError(f"{context} logical-index sequence is invalid.")
    try:
        ordered_logical = tuple(int(value) for value in logical_raw)
    except (TypeError, ValueError):
        raise ValueError(f"{context} logical-index sequence is invalid.") from None
    logical_digest = str(
        state.get("ordered_logical_candidate_indices_sha256", "")
    )
    if ordered_logical or (
        re.fullmatch(r"[0-9a-f]{64}", logical_digest) is None
        or digest_jsonable([]) != logical_digest
    ):
        raise ValueError(
            f"{context} non-seq2p route must carry an authenticated empty logical sequence."
        )

    source_result_json = sidecar.get("source_result_json")
    source_result_sha256 = str(sidecar.get("source_result_sha256", ""))
    if (
        not isinstance(source_result_json, str)
        or not source_result_json.strip()
        or re.fullmatch(r"[0-9a-f]{64}", source_result_sha256) is None
    ):
        raise ValueError(f"{context} lacks source-result provenance.")
    provenance = {
        "schema": "verified_singleton_selection_count_resume_state_v1",
        "source_result_json": str(source_result_json),
        "source_result_sha256": source_result_sha256,
        "resume_sidecar_json": str(sidecar_path),
        "resume_sidecar_sha256": file_sha256(sidecar_path),
        "controller_round": controller_round,
        "pool_size": source_pool_size,
        "ordered_parent_pool_indices_sha256": parent_digest,
        "ordered_logical_candidate_indices_sha256": logical_digest,
        "selection_count_total": len(ordered_parent),
        "seq2p_logical_mode": False,
        "source_binding": source_binding,
        "no_credentials_serialized": True,
    }
    assert_no_secret_material(
        provenance,
        context="verified singleton selection-count resume provenance",
    )
    return {
        "pool_size": source_pool_size,
        "ordered_parent_pool_indices": ordered_parent,
        "ordered_logical_candidate_indices": ordered_logical,
        "provenance": provenance,
    }


def _assert_repeated_resume_contract_consistency(
    contracts: Sequence[Mapping[str, Any]],
) -> None:
    """Fail closed before per-position contracts collapse into a label map."""

    by_label: dict[str, dict[str, Any]] = {}
    for raw_contract in contracts:
        contract = dict(raw_contract)
        label = str(contract.get("label", ""))
        scientific = dict(contract)
        scientific.pop("resume_contract_provenance", None)
        previous = by_label.get(label)
        if previous is not None and digest_jsonable(previous) != digest_jsonable(
            scientific
        ):
            raise ValueError(
                "Repeated signed-prefix label has conflicting execution/guard "
                f"contracts: {label!r}."
            )
        by_label[label] = scientific


def match_resume_scaffold_to_pool(
    source: ResumeScaffoldSource,
    *,
    pool: Sequence[AnsatzTerm],
    build_selected_layout: Callable[[list[AnsatzTerm]], AnsatzParameterLayout],
    expected_parameterization_mode: str,
    require_source_generator_contract: bool = False,
) -> ResumeMatchedScaffold:
    verified_active_prefix = (
        _load_verified_active_prefix_sidecar(source)
        if bool(require_source_generator_contract)
        else None
    )
    selected_terms = tuple(
        verified_active_prefix["terms"]
        if isinstance(verified_active_prefix, Mapping)
        else source.runtime_input.selected_terms
    )
    verified_contracts = tuple(
        verified_active_prefix["contracts"]
        if isinstance(verified_active_prefix, Mapping)
        else ()
    )
    _assert_repeated_resume_contract_consistency(verified_contracts)
    if len(selected_terms) == 0:
        raise ValueError("Structural resume requires at least one selected scaffold generator.")
    by_label: dict[str, list[int]] = {}
    for idx, term in enumerate(pool):
        by_label.setdefault(str(term.label), []).append(int(idx))
    selected_ops: list[AnsatzTerm] = []
    selected_pool_indices: list[int] = []
    missing: list[str] = []
    selected_outside_pool_labels: list[str] = []
    selected_outside_pool_records: list[dict[str, Any]] = []
    legacy_execution_mode_rebind_records: list[dict[str, Any]] = []
    selected_generator_contracts: dict[str, dict[str, Any]] = {}
    adapt_payload = source.payload.get("adapt_vqe", {})
    adapt_payload = adapt_payload if isinstance(adapt_payload, Mapping) else {}
    parameterization_payload = adapt_payload.get("parameterization", {})
    parameterization_payload = (
        parameterization_payload
        if isinstance(parameterization_payload, Mapping)
        else {}
    )
    raw_blocks = parameterization_payload.get("blocks", [])
    raw_blocks = (
        list(raw_blocks)
        if isinstance(raw_blocks, Sequence)
        and not isinstance(raw_blocks, (str, bytes))
        else []
    )
    legacy_missing_execution_mode_positions = {
        int(index)
        for index, raw_block in enumerate(raw_blocks)
        if isinstance(raw_block, Mapping) and "execution_mode" not in raw_block
    }
    def _restore_outside_pool_source_contract(
        *,
        selected_position: int,
        serialized_term: AnsatzTerm,
    ) -> tuple[AnsatzTerm, dict[str, Any] | None]:
        label = str(serialized_term.label)
        if not bool(require_source_generator_contract):
            return serialized_term, None
        if selected_position >= len(verified_contracts):
            raise ValueError(
                "Verified resume cannot restore selected generator contract: "
                f"sidecar has no row for position {selected_position}."
            )
        contract = verified_contracts[selected_position]
        if not isinstance(contract, Mapping):
            raise ValueError(
                "Verified resume signed operator contract is not a mapping."
            )
        if str(contract.get("label", "")) != label:
            raise ValueError(
                "Verified resume signed operator contract label mismatch at "
                f"position {selected_position}."
            )
        return serialized_term, dict(contract)

    for selected_position, term in enumerate(selected_terms):
        label = str(term.label)
        matches = by_label.get(label, [])
        term_fingerprint = candidate_generator_fingerprint(term)
        term_semantics = _generator_semantic_signature(term)
        if not matches:
            if "::child_set[" in label:
                restored_term, restored_contract = (
                    _restore_outside_pool_source_contract(
                        selected_position=int(selected_position),
                        serialized_term=term,
                    )
                )
                selected_ops.append(restored_term)
                if restored_contract is not None:
                    selected_generator_contracts[label] = dict(
                        restored_contract
                    )
                selected_outside_pool_labels.append(label)
                selected_outside_pool_records.append(
                    {
                        "label": label,
                        "reason": "runtime_split_child_set_label_absent",
                        "serialized_generator_fingerprint": term_fingerprint,
                        "restored_execution_mode": str(
                            getattr(
                                restored_term,
                                "execution_mode",
                                "termwise_product",
                            )
                        ),
                        "source_generator_contract_restored": bool(
                            restored_contract is not None
                        ),
                        "same_label_pool_fingerprints": [],
                    }
                )
            else:
                missing.append(label)
            continue
        semantic_matches = [
            int(idx)
            for idx in matches
            if _generator_semantic_signature(pool[int(idx)]) == term_semantics
        ]
        if not semantic_matches:
            if (
                selected_position in legacy_missing_execution_mode_positions
            ):
                polynomial_matches = [
                    int(idx)
                    for idx in matches
                    if _polynomial_signature(pool[int(idx)].polynomial)
                    == _polynomial_signature(term.polynomial)
                ]
                if len(polynomial_matches) > 1:
                    raise ValueError(
                        "Legacy resume execution-mode recovery found multiple "
                        f"exact polynomial matches for label {label!r}."
                    )
                if len(polynomial_matches) == 1:
                    idx = int(polynomial_matches[0])
                    selected_pool_indices.append(idx)
                    selected_ops.append(pool[idx])
                    legacy_execution_mode_rebind_records.append(
                        {
                            "selected_position": int(selected_position),
                            "label": label,
                            "serialized_execution_mode": str(
                                getattr(
                                    term,
                                    "execution_mode",
                                    "termwise_product",
                                )
                            ),
                            "rebound_execution_mode": str(
                                getattr(
                                    pool[idx],
                                    "execution_mode",
                                    "termwise_product",
                                )
                            ),
                            "equivalence": "exact_label_and_polynomial_v1",
                        }
                    )
                    continue
            selected_ops.append(term)
            selected_outside_pool_labels.append(label)
            selected_outside_pool_records.append(
                {
                    "label": label,
                    "reason": "same_label_pool_semantics_mismatch",
                    "serialized_generator_fingerprint": term_fingerprint,
                    "same_label_pool_fingerprints": [
                        candidate_generator_fingerprint(pool[int(idx)])
                        for idx in matches
                    ],
                }
            )
            continue
        idx = int(semantic_matches[0])
        selected_pool_indices.append(idx)
        selected_ops.append(pool[idx])
    if missing:
        raise ValueError(
            "Resume scaffold selected generator(s) are absent from the current pool: "
            + ", ".join(missing[:8])
        )
    selected_layout = build_selected_layout(list(selected_ops))
    artifact_layout_payload = serialize_layout(
        source.runtime_input.base_layout
    )
    current_layout_payload = serialize_layout(selected_layout)
    legacy_execution_modes_restored = bool(
        legacy_execution_mode_rebind_records
        or selected_generator_contracts
    )
    if legacy_execution_modes_restored:
        for layout_payload in (
            artifact_layout_payload,
            current_layout_payload,
        ):
            for block in layout_payload.get("blocks", []):
                if isinstance(block, dict):
                    block.pop("execution_mode", None)
    artifact_layout_digest = digest_jsonable(artifact_layout_payload)
    current_layout_digest = digest_jsonable(current_layout_payload)
    if artifact_layout_digest != current_layout_digest:
        raise ValueError(
            "Resume scaffold layout does not match current pool reconstruction "
            f"(artifact={artifact_layout_digest}, current={current_layout_digest})."
        )
    theta_runtime = np.asarray(source.runtime_input.theta_runtime, dtype=float).reshape(-1)
    if int(theta_runtime.size) != int(selected_layout.runtime_parameter_count):
        raise ValueError(
            "Resume runtime theta length does not match reconstructed selected layout."
        )
    theta_logical = (
        None
        if source.runtime_input.theta_logical is None
        else np.asarray(source.runtime_input.theta_logical, dtype=float).reshape(-1)
    )
    expected_mode = _normalize_resume_parameterization_mode(
        expected_parameterization_mode,
        field="expected_parameterization_mode",
    )
    source_mode_fields = _explicit_resume_parameterization_modes(source.payload)
    source_modes = set(source_mode_fields.values())
    if len(source_modes) > 1:
        raise ValueError(
            "Resume scaffold has conflicting explicit parameterization modes: "
            + json.dumps(source_mode_fields, sort_keys=True)
        )
    source_mode = next(iter(source_modes), None)
    if source_mode is not None and source_mode != expected_mode:
        raise ValueError(
            "Resume scaffold parameterization mode does not match the current "
            f"route: source={source_mode!r}, expected={expected_mode!r}."
        )

    logical_alias_max_abs_error: float | None = None
    if expected_mode == "logical_shared":
        if theta_logical is None:
            raise ValueError(
                "logical_shared resume requires an explicit logical_optimal_point "
                "vector; projecting an independent runtime vector is not allowed."
            )
        if int(theta_logical.size) != int(selected_layout.logical_parameter_count):
            raise ValueError(
                "logical_shared resume logical theta length does not match the "
                "reconstructed selected layout."
            )
        runtime_alias = np.asarray(
            expand_legacy_logical_theta(theta_logical, selected_layout),
            dtype=float,
        ).reshape(-1)
        logical_alias_max_abs_error = float(
            np.max(np.abs(theta_runtime - runtime_alias))
            if int(theta_runtime.size)
            else 0.0
        )
        if not np.allclose(
            theta_runtime,
            runtime_alias,
            atol=1.0e-10,
            rtol=0.0,
        ):
            raise ValueError(
                "logical_shared resume requires runtime theta to be a blockwise "
                "alias of logical theta; found max_abs_error="
                f"{logical_alias_max_abs_error:.3e}."
            )

    replay = _replay_resume_in_expected_parameterization(
        selected_ops=selected_ops,
        selected_layout=selected_layout,
        theta_runtime=theta_runtime,
        theta_logical=theta_logical,
        psi_ref=np.asarray(source.runtime_input.psi_ref, dtype=complex).reshape(-1),
        psi_initial=np.asarray(
            source.runtime_input.psi_initial,
            dtype=complex,
        ).reshape(-1),
        expected_parameterization_mode=expected_mode,
    )
    source_mode_inferred = source_mode is None
    validation = {
        "schema_version": "static_hh_adapt_resume_pool_match_v2",
        "selected_pool_indices": [int(x) for x in selected_pool_indices],
        "operator_labels_digest": digest_jsonable([str(op.label) for op in selected_ops]),
        "parameterization_digest": current_layout_digest,
        "theta_runtime_digest": digest_jsonable([float(x) for x in theta_runtime.tolist()]),
        "selected_term_count": int(len(selected_ops)),
        "selected_terms_outside_pool_count": int(len(selected_outside_pool_labels)),
        "selected_terms_outside_pool_reason": (
            None
            if not selected_outside_pool_labels
            else (
                "runtime_split_child_set_terms_are_terminal_scaffold_terms"
                if all(
                    record["reason"] == "runtime_split_child_set_label_absent"
                    for record in selected_outside_pool_records
                )
                else "serialized_selected_generator_semantics_preserved_v1"
            )
        ),
        "selected_terms_outside_pool_labels": [str(x) for x in selected_outside_pool_labels],
        "selected_terms_outside_pool_records": selected_outside_pool_records,
        "legacy_execution_mode_omission_repaired": bool(
            legacy_execution_modes_restored
        ),
        "legacy_execution_mode_rebind_records": (
            legacy_execution_mode_rebind_records
        ),
        "selected_generator_contract_restore_count": int(
            len(selected_generator_contracts)
        ),
        "selected_generator_contract_restore_labels": sorted(
            selected_generator_contracts
        ),
        "signed_active_prefix_sidecar": (
            dict(verified_active_prefix["provenance"])
            if isinstance(verified_active_prefix, Mapping)
            else None
        ),
        "runtime_parameter_count": int(theta_runtime.size),
        "expected_parameterization_mode": str(expected_mode),
        "source_parameterization_mode": str(source_mode or expected_mode),
        "source_parameterization_mode_fields": dict(source_mode_fields),
        "source_parameterization_mode_inferred": bool(source_mode_inferred),
        "source_parameterization_mode_resolution": (
            "expected_mode_strict_replay_inference"
            if source_mode_inferred
            else "explicit_source_metadata"
        ),
        "logical_runtime_block_alias_checked": bool(
            expected_mode == "logical_shared"
        ),
        "logical_runtime_block_alias_max_abs_error": logical_alias_max_abs_error,
        "strict_expected_mode_replay": replay,
        "no_credentials_serialized": True,
    }
    assert_no_secret_material(validation, context="resume pool match")
    return ResumeMatchedScaffold(
        selected_ops=tuple(selected_ops),
        selected_layout=selected_layout,
        theta_runtime=theta_runtime,
        theta_logical=theta_logical,
        selected_pool_indices=tuple(int(x) for x in selected_pool_indices),
        selected_generator_contracts={
            str(label): dict(contract)
            for label, contract in selected_generator_contracts.items()
        },
        validation=validation,
    )


def extract_best_frontier_resume_checkpoint(
    source: ResumeScaffoldSource,
    *,
    expected_phase3_response_coordinate_scope: str | None = None,
    expected_powell_coordinate_chart_policy: str | None = None,
    expected_route_profile_conformance: str | None = None,
    expected_sr_route_profile_request: str | None = None,
    expected_sr_route_profile_contract: Mapping[str, Any] | None = None,
    expected_sr_route_profile_contract_sha256: str | None = None,
) -> ResumeBestFrontierCheckpoint:
    """Validate and extract one complete ``beam_round_done`` winning lineage.

    This seam is intentionally stricter than the general scaffold loader.  It
    accepts only a complete singleton, append-only winning lineage with no
    accepted prune deletion.  Those restrictions let a caller resume the
    preserved best branch without pretending that the discarded beam frontier
    or an unrecorded structural transition can be reconstructed.
    """

    payload = source.payload
    _assert_no_unsupported_modeled_minimum_execution_checkpoint(
        payload,
        context="Best-frontier resume",
    )
    if extract_formal_manifold_route_composition(payload) is not None:
        raise ValueError(
            "Best-frontier resume does not execute author-retired "
            "Formal-Manifold route checkpoints."
        )
    powell_chart_validation = validate_resume_powell_coordinate_chart_policy(
        payload,
        expected_policy=expected_powell_coordinate_chart_policy,
        expected_route_profile_conformance=(
            expected_route_profile_conformance
        ),
        context="Best-frontier resume",
    )
    sr_route_profile_validation = validate_resume_sr_route_profile_contract(
        payload,
        expected_profile_request=expected_sr_route_profile_request,
        expected_contract=expected_sr_route_profile_contract,
        expected_contract_sha256=expected_sr_route_profile_contract_sha256,
        context="Best-frontier resume",
    )
    phase3_response_scope_validation = (
        validate_resume_phase3_response_coordinate_scope(
            payload,
            expected_scope=expected_phase3_response_coordinate_scope,
            expected_profile_request=expected_sr_route_profile_request,
            context="Best-frontier resume",
        )
    )
    adapt = _adapt_block(payload)
    checkpoint_raw = payload.get("checkpoint", None)
    if not isinstance(checkpoint_raw, Mapping):
        raise ValueError("Best-frontier resume requires a checkpoint object.")
    checkpoint = dict(checkpoint_raw)

    if str(checkpoint.get("reason", "")) != "beam_round_done":
        raise ValueError(
            "Best-frontier resume requires checkpoint.reason='beam_round_done'."
        )
    if str(checkpoint.get("checkpoint_branch_policy", "")) != "best_frontier_branch":
        raise ValueError(
            "Best-frontier resume requires checkpoint_branch_policy="
            "'best_frontier_branch'."
        )
    if not bool(checkpoint.get("beam_enabled", False)):
        raise ValueError("Best-frontier resume requires a beam-enabled checkpoint.")
    if not bool(adapt.get("partial_checkpoint", False)):
        raise ValueError("Best-frontier resume requires adapt_vqe.partial_checkpoint=true.")
    if not bool(adapt.get("adapt_beam_enabled", False)):
        raise ValueError("Best-frontier resume requires adapt_vqe.adapt_beam_enabled=true.")
    if adapt.get("history_checkpoint_complete", None) is not True:
        raise ValueError(
            "Best-frontier resume requires history_checkpoint_complete=true."
        )

    history_raw = adapt.get("history", None)
    if not isinstance(history_raw, Sequence) or isinstance(
        history_raw, (str, bytes, bytearray)
    ):
        raise ValueError("Best-frontier resume requires adapt_vqe.history array.")
    if any(not isinstance(row, Mapping) for row in history_raw):
        raise ValueError("Best-frontier resume history rows must be JSON objects.")
    history_cleaned_raw, _removed = _drop_obsolete_admission_rollback_state(
        [dict(row) for row in history_raw]
    )
    history = [dict(row) for row in history_cleaned_raw]
    if not history:
        raise ValueError("Best-frontier resume history must be non-empty.")
    latest_v4_prune_trust_state = _validate_latest_v4_prune_trust_state(
        history=history,
        artifact_profile_request=sr_route_profile_validation[
            "artifact_profile_request"
        ],
        context="Best-frontier resume",
    )

    def _int_field(
        block: Mapping[str, Any],
        field: str,
        *,
        context: str,
        minimum: int = 0,
    ) -> int:
        try:
            value = int(block.get(field))
        except (TypeError, ValueError):
            raise ValueError(
                f"Best-frontier resume requires integer {context}.{field}."
            ) from None
        if value < int(minimum):
            raise ValueError(
                f"Best-frontier resume requires {context}.{field} >= {minimum}."
            )
        return value

    history_count = _int_field(
        adapt,
        "history_count",
        context="adapt_vqe",
        minimum=1,
    )
    if history_count != len(history):
        raise ValueError(
            "Best-frontier resume history_count does not match complete history length."
        )
    controller_round = _int_field(
        checkpoint,
        "depth",
        context="checkpoint",
        minimum=1,
    )
    if controller_round != history_count:
        raise ValueError(
            "Best-frontier resume checkpoint depth does not match history_count."
        )

    branch_id = _int_field(
        checkpoint,
        "branch_id",
        context="checkpoint",
        minimum=0,
    )
    parent_raw = checkpoint.get("parent_branch_id", None)
    parent_branch_id = None if parent_raw is None else int(parent_raw)
    if _int_field(adapt, "branch_id", context="adapt_vqe") != branch_id:
        raise ValueError("Best-frontier resume branch_id fields disagree.")
    if adapt.get("parent_branch_id", None) != parent_raw:
        raise ValueError("Best-frontier resume parent_branch_id fields disagree.")

    operator_labels_raw = adapt.get("operators", None)
    if not isinstance(operator_labels_raw, Sequence) or isinstance(
        operator_labels_raw, (str, bytes, bytearray)
    ):
        raise ValueError("Best-frontier resume requires adapt_vqe.operators array.")
    operator_labels = tuple(str(label) for label in operator_labels_raw)
    if not operator_labels or any(label == "" for label in operator_labels):
        raise ValueError("Best-frontier resume operator labels must be non-empty.")
    ansatz_depth = _int_field(
        adapt,
        "ansatz_depth",
        context="adapt_vqe",
        minimum=1,
    )
    if ansatz_depth != len(operator_labels):
        raise ValueError(
            "Best-frontier resume ansatz_depth does not match operator count."
        )
    if _int_field(
        checkpoint,
        "ansatz_depth",
        context="checkpoint",
        minimum=1,
    ) != ansatz_depth:
        raise ValueError("Best-frontier resume checkpoint ansatz depth disagrees.")
    accepted_prune_present = any(
        isinstance(row.get("post_admission_prune"), Mapping)
        and int(row["post_admission_prune"].get("accepted_count", 0) or 0) > 0
        for row in history
    )
    authenticated_v4_pruned_lineage = None
    if accepted_prune_present:
        authenticated_v4_pruned_lineage = _validate_authenticated_v4_pruned_lineage(
            history=history,
            operator_labels=operator_labels,
            ansatz_depth=int(ansatz_depth),
            artifact_profile_request=sr_route_profile_validation[
                "artifact_profile_request"
            ],
            context="Best-frontier resume",
        )
        if (
            latest_v4_prune_trust_state
            != authenticated_v4_pruned_lineage["restored_prune_trust_state"]
        ):
            raise ValueError(
                "Best-frontier resume latest v4 prune trust state disagrees "
                "with its authenticated deletion lineage."
            )
    elif ansatz_depth < history_count:
        raise ValueError(
            "Best-frontier singleton resume requires ansatz depth to be at "
            "least the completed admission count."
        )
    preserved_seed_prefix_depth = (
        0
        if authenticated_v4_pruned_lineage is not None
        else int(ansatz_depth - history_count)
    )

    selected_labels: list[str] = []
    previous_branch_id: int | None = None
    for index, row in enumerate(history):
        expected_round = int(index + 1)
        if _int_field(row, "depth", context=f"history[{index}]", minimum=1) != expected_round:
            raise ValueError(
                "Best-frontier resume history depths must be contiguous from one."
            )
        row_branch_id = _int_field(
            row,
            "branch_id",
            context=f"history[{index}]",
            minimum=0,
        )
        row_parent_id = _int_field(
            row,
            "parent_branch_id",
            context=f"history[{index}]",
            minimum=0,
        )
        if previous_branch_id is not None and row_parent_id != previous_branch_id:
            raise ValueError(
                "Best-frontier resume history is not one continuous winning branch."
            )
        previous_branch_id = row_branch_id
        if _int_field(row, "batch_size", context=f"history[{index}]", minimum=1) != 1:
            raise ValueError(
                "Best-frontier resume only supports preserved singleton admissions."
            )
        label = str(row.get("selected_op", ""))
        if label == "":
            raise ValueError(
                f"Best-frontier resume history[{index}].selected_op is missing."
            )
        selected_labels.append(label)
        try:
            selected_position = int(row.get("selected_position"))
        except (TypeError, ValueError):
            raise ValueError(
                f"Best-frontier resume history[{index}].selected_position is missing."
            ) from None
        if (
            authenticated_v4_pruned_lineage is None
            and selected_position != preserved_seed_prefix_depth + index
        ):
            raise ValueError(
                "Best-frontier resume requires append-only ordered insertion "
                "positions after its preserved seed prefix."
            )
        prune = row.get("post_admission_prune", {})
        if not isinstance(prune, Mapping):
            raise ValueError(
                f"Best-frontier resume history[{index}].post_admission_prune is invalid."
            )
        try:
            accepted_prune_count = int(prune.get("accepted_count", 0) or 0)
        except (TypeError, ValueError):
            raise ValueError(
                f"Best-frontier resume history[{index}] prune count is invalid."
            ) from None
        if (
            accepted_prune_count != 0
            and authenticated_v4_pruned_lineage is None
        ):
            raise ValueError(
                "Best-frontier resume cannot reconstruct a lineage with accepted prune deletion."
            )
    if previous_branch_id != branch_id:
        raise ValueError(
            "Best-frontier resume last history branch does not match checkpoint branch."
        )
    if (
        authenticated_v4_pruned_lineage is None
        and tuple(selected_labels)
        != operator_labels[preserved_seed_prefix_depth:]
    ):
        raise ValueError(
            "Best-frontier resume ordered history operators do not match the "
            "active operators after its preserved seed prefix."
        )

    def _finite_vector(value: Any, *, field: str) -> np.ndarray:
        if not isinstance(value, Sequence) or isinstance(
            value, (str, bytes, bytearray)
        ):
            raise ValueError(f"Best-frontier resume requires {field} array.")
        try:
            vector = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            raise ValueError(f"Best-frontier resume {field} is not numeric.") from None
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            raise ValueError(f"Best-frontier resume {field} must be finite and non-empty.")
        return vector

    theta_runtime = _finite_vector(
        adapt.get("optimal_point", None),
        field="adapt_vqe.optimal_point",
    )
    theta_logical = _finite_vector(
        adapt.get("logical_optimal_point", None),
        field="adapt_vqe.logical_optimal_point",
    )
    if authenticated_v4_pruned_lineage is not None:
        final_prefix = authenticated_v4_pruned_lineage[
            "final_active_prefix_checkpoint"
        ]
        prefix_runtime = _finite_vector(
            final_prefix.get("signed_unwrapped_runtime_parameters"),
            field="active_prefix_checkpoint.signed_unwrapped_runtime_parameters",
        )
        prefix_logical = _finite_vector(
            final_prefix.get("signed_unwrapped_logical_parameters"),
            field="active_prefix_checkpoint.signed_unwrapped_logical_parameters",
        )
        if not np.array_equal(prefix_runtime, theta_runtime) or not np.array_equal(
            prefix_logical, theta_logical
        ):
            raise ValueError(
                "Best-frontier resume signed active-prefix parameters disagree "
                "with the saved optimizer state."
            )
    if _int_field(adapt, "num_parameters", context="adapt_vqe", minimum=1) != int(
        theta_runtime.size
    ):
        raise ValueError("Best-frontier resume runtime theta count disagrees.")
    if _int_field(
        adapt,
        "logical_num_parameters",
        context="adapt_vqe",
        minimum=1,
    ) != int(theta_logical.size):
        raise ValueError("Best-frontier resume logical theta count disagrees.")
    if int(theta_logical.size) != ansatz_depth:
        raise ValueError(
            "Best-frontier singleton resume requires one logical theta per operator."
        )
    parameterization = adapt.get("parameterization", None)
    if not isinstance(parameterization, Mapping):
        raise ValueError("Best-frontier resume requires adapt_vqe.parameterization.")
    if _int_field(
        parameterization,
        "runtime_parameter_count",
        context="adapt_vqe.parameterization",
        minimum=1,
    ) != int(theta_runtime.size):
        raise ValueError(
            "Best-frontier resume parameterization runtime count disagrees."
        )
    if _int_field(
        parameterization,
        "logical_operator_count",
        context="adapt_vqe.parameterization",
        minimum=1,
    ) != ansatz_depth:
        raise ValueError(
            "Best-frontier resume parameterization logical count disagrees."
        )

    runtime_labels = tuple(str(term.label) for term in source.runtime_input.selected_terms)
    if runtime_labels != operator_labels:
        raise ValueError(
            "Best-frontier resume runtime-loader operators disagree with the checkpoint."
        )
    runtime_theta = np.asarray(source.runtime_input.theta_runtime, dtype=float).reshape(-1)
    if not np.array_equal(runtime_theta, theta_runtime):
        raise ValueError(
            "Best-frontier resume runtime-loader theta disagrees with the checkpoint."
        )
    runtime_theta_logical = source.runtime_input.theta_logical
    if runtime_theta_logical is None or not np.array_equal(
        np.asarray(runtime_theta_logical, dtype=float).reshape(-1),
        theta_logical,
    ):
        raise ValueError(
            "Best-frontier resume runtime-loader logical theta disagrees with the checkpoint."
        )

    state_digests: dict[str, str] = {}
    state_nq: dict[str, int] = {}
    for state_name, runtime_state in (
        ("initial_state", source.runtime_input.psi_initial),
        ("ansatz_input_state", source.runtime_input.psi_ref),
    ):
        state_manifest = payload.get(state_name, None)
        if not isinstance(state_manifest, Mapping):
            raise ValueError(f"Best-frontier resume requires {state_name} manifest.")
        nq_total = _int_field(
            state_manifest,
            "nq_total",
            context=state_name,
            minimum=1,
        )
        state_nq[state_name] = nq_total
        try:
            norm = float(state_manifest.get("norm"))
        except (TypeError, ValueError):
            raise ValueError(
                f"Best-frontier resume {state_name}.norm is invalid."
            ) from None
        if not math.isfinite(norm) or abs(norm - 1.0) > 1.0e-8:
            raise ValueError(
                f"Best-frontier resume {state_name} must be normalized."
            )
        runtime_size = int(np.asarray(runtime_state, dtype=complex).reshape(-1).size)
        if runtime_size != (1 << nq_total):
            raise ValueError(
                f"Best-frontier resume {state_name} dimension disagrees with nq_total."
            )
        state_digests[state_name] = digest_jsonable(state_manifest)
    if state_nq["initial_state"] != state_nq["ansatz_input_state"]:
        raise ValueError("Best-frontier resume state manifests use different qubit counts.")

    trust_raw = adapt.get("route_a_trust_region_state", None)
    if not isinstance(trust_raw, Mapping):
        raise ValueError(
            "Best-frontier resume requires route_a_trust_region_state."
        )
    trust = dict(trust_raw)
    if str(trust.get("schema", "")) != "route_a_trust_region_state_v1":
        raise ValueError("Best-frontier resume trust-state schema is unsupported.")
    trust_update_count = _int_field(
        trust,
        "update_count",
        context="route_a_trust_region_state",
        minimum=1,
    )
    if trust_update_count != controller_round:
        raise ValueError(
            "Best-frontier resume trust update_count does not match controller round."
        )
    for field, allow_zero in (("radius", True), ("reference_radius", False)):
        try:
            value = float(trust.get(field))
        except (TypeError, ValueError):
            raise ValueError(
                f"Best-frontier resume trust-state {field} is invalid."
            ) from None
        if not math.isfinite(value) or value < 0.0 or (not allow_zero and value == 0.0):
            raise ValueError(
                f"Best-frontier resume trust-state {field} is invalid."
            )
    last_update = trust.get("last_update", None)
    if not isinstance(last_update, Mapping):
        raise ValueError("Best-frontier resume trust state lacks last_update.")
    history_last_update = history[-1].get("route_a_trust_region_update", None)
    if not isinstance(history_last_update, Mapping):
        raise ValueError(
            "Best-frontier resume last history trust update is missing."
        )
    history_last_update_state = dict(history_last_update)
    endpoint_overlap_query_accounting = history_last_update_state.pop(
        "endpoint_overlap_query_accounting", None
    )
    if endpoint_overlap_query_accounting is not None:
        if not isinstance(endpoint_overlap_query_accounting, Mapping):
            raise ValueError(
                "Best-frontier resume endpoint-overlap query accounting is malformed."
            )
        overlap_receipt = dict(endpoint_overlap_query_accounting)
        if (
            str(overlap_receipt.get("schema", ""))
            != "adaptive_trust_overlap_query_accounting_v1"
            or overlap_receipt.get("enabled") is not True
            or str(overlap_receipt.get("status", "")) != "complete"
            or str(overlap_receipt.get("component", "")) != "N_metric"
            or str(overlap_receipt.get("formal_query_category", ""))
            != "N_cross"
            or not str(overlap_receipt.get("primitive_id", ""))
        ):
            raise ValueError(
                "Best-frontier resume endpoint-overlap query accounting is not "
                "a complete typed receipt."
            )
    if digest_jsonable(history_last_update_state) != digest_jsonable(last_update):
        raise ValueError(
            "Best-frontier resume last history trust update disagrees with trust state."
        )

    beam_replay = adapt.get("beam_replay_telemetry", None)
    if not isinstance(beam_replay, Mapping):
        raise ValueError("Best-frontier resume lacks beam replay telemetry.")
    beam_branch_raw = beam_replay.get("checkpoint_branch", None)
    if not isinstance(beam_branch_raw, Mapping):
        raise ValueError("Best-frontier resume lacks checkpoint branch telemetry.")
    beam_branch = dict(beam_branch_raw)
    if str(beam_branch.get("status", "")) != "frontier" or bool(
        beam_branch.get("terminated", False)
    ):
        raise ValueError("Best-frontier resume checkpoint branch is not live frontier state.")
    for field, expected in (
        ("branch_id", branch_id),
        ("depth_local", controller_round),
        ("history_count", history_count),
        ("ansatz_depth", ansatz_depth),
    ):
        if _int_field(
            beam_branch,
            field,
            context="beam_replay_telemetry.checkpoint_branch",
            minimum=0,
        ) != expected:
            raise ValueError(
                f"Best-frontier resume beam checkpoint {field} disagrees."
            )
    if beam_branch.get("parent_branch_id", None) != parent_raw:
        raise ValueError("Best-frontier resume beam parent branch disagrees.")
    beam_labels = beam_branch.get("operator_labels", None)
    if not isinstance(beam_labels, Sequence) or isinstance(
        beam_labels, (str, bytes, bytearray)
    ) or tuple(str(label) for label in beam_labels) != operator_labels:
        raise ValueError("Best-frontier resume beam operator labels disagree.")
    beam_trust = beam_branch.get("route_a_trust_region_state", None)
    if not isinstance(beam_trust, Mapping) or digest_jsonable(beam_trust) != digest_jsonable(
        trust
    ):
        raise ValueError("Best-frontier resume beam trust state disagrees.")
    for context, tail_raw, tail_count_raw in (
        (
            "adapt_vqe",
            adapt.get("history_tail", None),
            adapt.get("history_tail_count", None),
        ),
        (
            "beam checkpoint",
            beam_branch.get("history_tail", None),
            beam_branch.get("history_tail_count", None),
        ),
    ):
        if not isinstance(tail_raw, Sequence) or isinstance(
            tail_raw, (str, bytes, bytearray)
        ) or any(not isinstance(row, Mapping) for row in tail_raw):
            raise ValueError(f"Best-frontier resume {context} history tail is invalid.")
        tail_cleaned, _tail_removed = _drop_obsolete_admission_rollback_state(
            [dict(row) for row in tail_raw]
        )
        try:
            tail_count = int(tail_count_raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"Best-frontier resume {context} history tail count is invalid."
            ) from None
        if tail_count != len(tail_cleaned) or tail_count > history_count:
            raise ValueError(
                f"Best-frontier resume {context} history tail count disagrees."
            )
        if digest_jsonable(tail_cleaned) != digest_jsonable(history[-tail_count:]):
            raise ValueError(
                f"Best-frontier resume {context} history tail is not the winning-lineage suffix."
            )

    frontier_prune_key_raw = beam_branch.get("frontier_prune_key", None)
    if not isinstance(frontier_prune_key_raw, Mapping):
        raise ValueError("Best-frontier resume lacks frontier_prune_key.")
    frontier_prune_key = dict(frontier_prune_key_raw)
    prune_labels = frontier_prune_key.get("labels", None)
    if not isinstance(prune_labels, Sequence) or isinstance(
        prune_labels, (str, bytes, bytearray)
    ) or tuple(str(label) for label in prune_labels) != operator_labels:
        raise ValueError("Best-frontier resume frontier-prune operators disagree.")
    theta_round10 = _finite_vector(
        frontier_prune_key.get("theta_round10", None),
        field="frontier_prune_key.theta_round10",
    )
    try:
        theta_round_digits = int(frontier_prune_key.get("theta_round10_digits"))
    except (TypeError, ValueError):
        raise ValueError(
            "Best-frontier resume frontier-prune theta precision is invalid."
        ) from None
    if theta_round10.size != theta_runtime.size or theta_round_digits < 0:
        raise ValueError("Best-frontier resume frontier-prune theta count disagrees.")
    round_tolerance = max(
        np.finfo(float).eps,
        0.51 * (10.0 ** (-theta_round_digits)),
    )
    if not np.allclose(
        theta_round10,
        theta_runtime,
        rtol=0.0,
        atol=round_tolerance,
    ):
        raise ValueError("Best-frontier resume frontier-prune theta disagrees.")

    try:
        source_energy = float(adapt.get("energy"))
        beam_energy = float(beam_branch.get("energy"))
    except (TypeError, ValueError):
        raise ValueError("Best-frontier resume source energy is invalid.") from None
    if not math.isfinite(source_energy) or not math.isfinite(beam_energy):
        raise ValueError("Best-frontier resume source energy must be finite.")
    if abs(source_energy - beam_energy) > 1.0e-12:
        raise ValueError("Best-frontier resume beam energy disagrees with source energy.")

    validation = {
        "schema_version": "static_adapt_best_frontier_resume_checkpoint_v1",
        "checkpoint_reason": "beam_round_done",
        "checkpoint_branch_policy": "best_frontier_branch",
        "history_checkpoint_complete": True,
        "history_count": int(history_count),
        "history_digest": digest_jsonable(history),
        "controller_round": int(controller_round),
        "ansatz_depth": int(ansatz_depth),
        "branch_id": int(branch_id),
        "parent_branch_id": parent_branch_id,
        "operator_labels_digest": digest_jsonable(operator_labels),
        "theta_runtime_digest": digest_jsonable(theta_runtime.tolist()),
        "theta_logical_digest": digest_jsonable(theta_logical.tolist()),
        "initial_state_digest": state_digests["initial_state"],
        "ansatz_input_state_digest": state_digests["ansatz_input_state"],
        "route_a_trust_region_state_digest": digest_jsonable(trust),
        "frontier_prune_key_digest": digest_jsonable(frontier_prune_key),
        "powell_coordinate_chart_policy": powell_chart_validation[
            "resolved_policy"
        ],
        "route_profile_conformance": powell_chart_validation[
            "route_profile_conformance"
        ],
        "powell_coordinate_chart_policy_validation": powell_chart_validation,
        "sr_route_profile_request": sr_route_profile_validation[
            "artifact_profile_request"
        ],
        "sr_route_profile_contract_sha256": sr_route_profile_validation[
            "contract_sha256"
        ],
        "sr_route_profile_contract_validation": sr_route_profile_validation,
        "phase3_response_coordinate_scope": (
            phase3_response_scope_validation["resolved_scope"]
        ),
        "phase3_response_coordinate_scope_validation": (
            phase3_response_scope_validation
        ),
        "phase12_energy_model_policies": dict(
            sr_route_profile_validation["phase12_energy_model_policies"]
        ),
        "phase12_energy_model_policy_validation": dict(
            sr_route_profile_validation[
                "phase12_energy_model_policy_validation"
            ]
        ),
        "discarded_frontier_reconstructed": False,
        "lineage_scope": "preserved_best_frontier_branch_only",
        "preserved_seed_prefix_depth": int(preserved_seed_prefix_depth),
        "authenticated_v4_pruned_lineage": (
            None
            if authenticated_v4_pruned_lineage is None
            else {
                key: value
                for key, value in authenticated_v4_pruned_lineage.items()
                if key != "final_active_prefix_checkpoint"
            }
        ),
        "latest_v4_prune_trust_state": (
            None
            if latest_v4_prune_trust_state is None
            else dict(latest_v4_prune_trust_state)
        ),
        "no_credentials_serialized": True,
    }
    assert_no_secret_material(validation, context="best-frontier resume validation")
    return ResumeBestFrontierCheckpoint(
        history=tuple(dict(row) for row in history),
        controller_round=int(controller_round),
        ansatz_depth=int(ansatz_depth),
        branch_id=int(branch_id),
        parent_branch_id=parent_branch_id,
        operator_labels=operator_labels,
        theta_runtime=tuple(float(value) for value in theta_runtime.tolist()),
        theta_logical=tuple(float(value) for value in theta_logical.tolist()),
        route_a_trust_region_state=trust,
        beam_checkpoint_branch=beam_branch,
        frontier_prune_key=frontier_prune_key,
        source_energy=float(source_energy),
        initial_state_digest=state_digests["initial_state"],
        ansatz_input_state_digest=state_digests["ansatz_input_state"],
        powell_coordinate_chart_policy=powell_chart_validation[
            "resolved_policy"
        ],
        route_profile_conformance=powell_chart_validation[
            "route_profile_conformance"
        ],
        sr_route_profile_request=sr_route_profile_validation[
            "artifact_profile_request"
        ],
        sr_route_profile_contract=(
            None
            if sr_route_profile_validation["contract_sha256"] is None
            else dict(
                next(
                    iter(
                        {
                            _path_label(path): _nested_payload_value(payload, path)
                            for path in _SR_ROUTE_PROFILE_CONTRACT_PATHS
                            if isinstance(_nested_payload_value(payload, path), Mapping)
                        }.values()
                    )
                )
            )
        ),
        sr_route_profile_contract_sha256=sr_route_profile_validation[
            "contract_sha256"
        ],
        phase3_response_coordinate_scope=(
            phase3_response_scope_validation["resolved_scope"]
        ),
        phase1_score_mode=(
            sr_route_profile_validation["phase12_energy_model_policies"].get(
                "phase1_score_mode"
            )
        ),
        phase1_energy_model=(
            sr_route_profile_validation["phase12_energy_model_policies"].get(
                "phase1_energy_model"
            )
        ),
        phase2_curvature_policy=(
            sr_route_profile_validation["phase12_energy_model_policies"].get(
                "phase2_curvature_policy"
            )
        ),
        phase2_cheap_curvature_proxy_policy=(
            sr_route_profile_validation["phase12_energy_model_policies"].get(
                "phase2_cheap_curvature_proxy_policy"
            )
        ),
        validation=validation,
    )


_TERMINAL_ESTIMATOR_CONSUMER_SCOPES = frozenset(
    {
        "accepted_refit:terminal_full_refit",
        "energy:final_full_refit",
        "final_state_verification",
    }
)


def load_checkpoint_estimator_call_ledger(
    source: ResumeScaffoldSource,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Load the exact state-keyed ledger linked by a completed checkpoint.

    This route-neutral loader deliberately does not fall back to terminal
    accounting artifacts.  A structural continuation may safely restore only
    the immutable ledger snapshot that the published completed-round
    ``current.json`` authenticated.
    """

    context = "Structural resume"
    adapt = _adapt_block(source.payload)
    checkpoint_pointer_raw = adapt.get("estimator_call_ledger_checkpoint")
    if checkpoint_pointer_raw is None:
        return None
    if not isinstance(checkpoint_pointer_raw, Mapping):
        raise ValueError(
            f"{context} estimator-ledger checkpoint pointer is malformed."
        )
    checkpoint_pointer = dict(checkpoint_pointer_raw)
    if checkpoint_pointer.get("enabled") is False:
        return None
    pointer_schema = str(checkpoint_pointer.get("schema", ""))
    if (
        pointer_schema
        not in {
            "paper_i_estimator_call_ledger_checkpoint_pointer_v1",
            "paper_i_estimator_call_ledger_checkpoint_pointer_v2",
        }
        or checkpoint_pointer.get("enabled") is not True
        or checkpoint_pointer.get("status") != "complete"
        or checkpoint_pointer.get("current_round_finalized") is not True
    ):
        raise ValueError(
            f"{context} estimator-ledger checkpoint pointer is incomplete."
        )
    relative_path = str(checkpoint_pointer.get("path", "")).strip()
    if (
        not relative_path
        or Path(relative_path).is_absolute()
        or Path(relative_path).name != relative_path
    ):
        raise ValueError(
            f"{context} estimator-ledger checkpoint path is not a sibling."
        )
    sidecar = Path(source.artifact_json).with_name(relative_path)
    if not sidecar.is_file():
        raise ValueError(
            f"{context} estimator-ledger checkpoint sidecar is missing."
        )
    expected_sha256 = str(checkpoint_pointer.get("sha256", ""))
    observed_sha256 = file_sha256(sidecar)
    if not expected_sha256 or observed_sha256 != expected_sha256:
        raise ValueError(
            f"{context} estimator-ledger checkpoint hash mismatch."
        )
    sidecar_payload = _read_json_object(sidecar)
    sidecar_schema = str(sidecar_payload.get("schema", ""))
    if (
        sidecar_schema
        not in {
            "paper_i_estimator_call_ledger_checkpoint_sidecar_v1",
            "paper_i_estimator_call_ledger_checkpoint_sidecar_v2",
        }
        or sidecar_payload.get("no_credentials_serialized") is not True
    ):
        raise ValueError(
            f"{context} estimator-ledger checkpoint schema is unsupported."
        )
    checkpoint_meta_raw = sidecar_payload.get("checkpoint")
    if not isinstance(checkpoint_meta_raw, Mapping):
        raise ValueError(
            f"{context} estimator-ledger checkpoint metadata is missing."
        )
    checkpoint_meta = dict(checkpoint_meta_raw)
    source_round = _source_controller_round(source.payload)
    if (
        int(checkpoint_meta.get("depth", -1)) != int(source_round)
        or int(checkpoint_pointer.get("checkpoint_depth", -1))
        != int(source_round)
        or str(checkpoint_pointer.get("checkpoint_reason", ""))
        != str(checkpoint_meta.get("reason", ""))
        or checkpoint_meta.get("current_round_finalized") is not True
    ):
        raise ValueError(
            f"{context} estimator-ledger checkpoint depth disagrees."
        )
    ledger_raw = sidecar_payload.get("ledger")
    if not isinstance(ledger_raw, Mapping):
        raise ValueError(
            f"{context} estimator-ledger checkpoint payload is missing."
        )
    if str(checkpoint_pointer.get("ledger_schema", "")) != str(
        ledger_raw.get("schema", "")
    ):
        raise ValueError(
            f"{context} estimator-ledger checkpoint ledger schema disagrees."
        )
    checkpoint_ledger = EstimatorCallLedger.from_payload(ledger_raw)
    checkpoint_payload = checkpoint_ledger.to_payload()
    checkpoint_summary = checkpoint_ledger.summary()
    checkpoint_occurrences = checkpoint_ledger.occurrence_summary()
    ledger_fingerprint = str(
        checkpoint_payload.get("ledger_fingerprint", "")
    )
    expected_fingerprint = str(
        checkpoint_pointer.get("ledger_fingerprint", "")
    )
    if (
        not ledger_fingerprint
        or expected_fingerprint != ledger_fingerprint
        or str(sidecar_payload.get("ledger_fingerprint", ""))
        != ledger_fingerprint
    ):
        raise ValueError(
            f"{context} estimator-ledger checkpoint fingerprint mismatch."
        )
    expected_unique = int(
        checkpoint_pointer.get("unique_primitive_count", -1)
    )
    expected_occurrences = int(
        checkpoint_pointer.get("raw_occurrence_count", -1)
    )
    expected_s_alg = int(checkpoint_pointer.get("S_alg", -1))
    expected_s_unique = int(
        checkpoint_pointer.get(
            "S_unique",
            # A v1 checkpoint used S_alg for the identity-collapsed count.
            expected_s_alg if pointer_schema.endswith("_v1") else -1,
        )
    )
    observed_unique = int(checkpoint_summary["S_unique"])
    observed_s_alg = int(checkpoint_occurrences["total_call_occurrences"])
    expected_s_alg_observed = (
        observed_unique if pointer_schema.endswith("_v1") else observed_s_alg
    )
    sidecar_s_unique = int(
        sidecar_payload.get(
            "S_unique",
            expected_s_alg if sidecar_schema.endswith("_v1") else -1,
        )
    )
    if (
        expected_unique
        != int(checkpoint_summary["unique_primitive_count"])
        or expected_occurrences
        != int(checkpoint_occurrences["total_call_occurrences"])
        or expected_s_alg != expected_s_alg_observed
        or expected_s_unique != observed_unique
        or int(sidecar_payload.get("unique_primitive_count", -1))
        != expected_unique
        or int(sidecar_payload.get("raw_occurrence_count", -1))
        != expected_occurrences
        or int(sidecar_payload.get("S_alg", -1)) != expected_s_alg
        or sidecar_s_unique != expected_s_unique
    ):
        raise ValueError(
            f"{context} estimator-ledger checkpoint counts do not close."
        )
    provenance = {
        "schema_version": "static_adapt_checkpoint_estimator_ledger_v1",
        "source_kind": "completed_round_checkpoint_sidecar",
        "path": str(sidecar),
        "sha256": str(observed_sha256),
        "source_current_json": str(source.artifact_json),
        "source_current_sha256": str(source.artifact_sha256),
        "restored_prefix_ledger_fingerprint": str(ledger_fingerprint),
        "restored_prefix_occurrence_count": int(expected_occurrences),
        "restored_prefix_S_alg": int(observed_s_alg),
        "restored_prefix_S_unique": int(observed_unique),
        "excluded_terminal_occurrence_count": 0,
        "excluded_terminal_consumer_scopes": [],
        "discarded_frontier_reconstructed": False,
        "current_round_finalized": True,
        "checkpoint_depth": int(source_round),
        "no_credentials_serialized": True,
    }
    assert_no_secret_material(
        provenance,
        context="structural resume estimator-ledger checkpoint provenance",
    )
    return checkpoint_payload, provenance


def _load_verified_singleton_estimator_ledger(
    source: ResumeScaffoldSource,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Restore the exact pre-terminal ledger from the preserved sidecar."""

    context = "Verified singleton resume"
    checkpoint_result = load_checkpoint_estimator_call_ledger(source)
    if checkpoint_result is not None:
        checkpoint_payload, provenance = checkpoint_result
        provenance = dict(provenance)
        provenance["schema_version"] = (
            "static_adapt_verified_singleton_estimator_ledger_v2"
        )
        return checkpoint_payload, provenance

    sidecar = Path(source.artifact_json).with_name("estimator_call_ledger.json")
    if not sidecar.is_file():
        raise ValueError(
            f"{context} requires sibling estimator_call_ledger.json."
        )
    sidecar_payload = _read_json_object(sidecar)
    if sidecar_payload.get("schema") not in {
        "paper_i_estimator_call_ledger_sidecar_v1",
        "paper_i_estimator_call_ledger_sidecar_v2",
    }:
        raise ValueError(f"{context} estimator-ledger sidecar schema is unsupported.")
    if sidecar_payload.get("adapt_success") is not True or sidecar_payload.get(
        "adapt_error"
    ) not in {None, ""}:
        raise ValueError(f"{context} estimator-ledger sidecar is not complete.")
    accounting_raw = sidecar_payload.get("accounting")
    if not isinstance(accounting_raw, Mapping):
        raise ValueError(f"{context} estimator-ledger accounting is missing.")
    accounting = dict(accounting_raw)
    if (
        accounting.get("enabled") is not True
        or accounting.get("complete") is not True
        or list(accounting.get("exact_blockers", []))
    ):
        raise ValueError(f"{context} estimator-ledger accounting is unresolved.")
    ledger_raw = sidecar_payload.get("ledger")
    if not isinstance(ledger_raw, Mapping):
        raise ValueError(f"{context} estimator-ledger payload is missing.")
    full_ledger = EstimatorCallLedger.from_payload(ledger_raw)
    full_payload = full_ledger.to_payload()

    identity_by_id: dict[str, EstimatorCallKey] = {}
    entries = full_payload.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(f"{context} estimator-ledger entries are invalid.")
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(
            entry.get("identity"), Mapping
        ):
            raise ValueError(f"{context} estimator-ledger entry is invalid.")
        primitive_id = str(entry.get("primitive_id", ""))
        identity = EstimatorCallKey.from_dict(entry["identity"])
        if primitive_id != identity.primitive_id or primitive_id in identity_by_id:
            raise ValueError(f"{context} estimator-ledger identity map is invalid.")
        identity_by_id[primitive_id] = identity

    occurrences = full_payload.get("occurrences", [])
    if not isinstance(occurrences, list) or not occurrences:
        raise ValueError(f"{context} estimator-ledger occurrences are missing.")
    first_terminal_index: int | None = None
    for index, occurrence in enumerate(occurrences):
        if not isinstance(occurrence, Mapping):
            raise ValueError(f"{context} estimator-ledger occurrence is invalid.")
        scope = str(occurrence.get("consumer_scope", ""))
        if scope in _TERMINAL_ESTIMATOR_CONSUMER_SCOPES:
            first_terminal_index = index
            break
    prefix_occurrences = (
        occurrences
        if first_terminal_index is None
        else occurrences[:first_terminal_index]
    )
    terminal_occurrences = (
        [] if first_terminal_index is None else occurrences[first_terminal_index:]
    )
    if not prefix_occurrences:
        raise ValueError(f"{context} estimator-ledger prefix is empty.")
    if any(
        str(row.get("consumer_scope", ""))
        not in _TERMINAL_ESTIMATOR_CONSUMER_SCOPES
        for row in terminal_occurrences
        if isinstance(row, Mapping)
    ):
        raise ValueError(
            f"{context} estimator-ledger terminal suffix is not contiguous."
        )

    prefix_ledger = EstimatorCallLedger()
    for expected_sequence, occurrence in enumerate(prefix_occurrences, start=1):
        if not isinstance(occurrence, Mapping):
            raise ValueError(f"{context} estimator-ledger occurrence is invalid.")
        if int(occurrence.get("sequence", 0)) != expected_sequence:
            raise ValueError(
                f"{context} estimator-ledger sequence is not contiguous."
            )
        primitive_id = str(occurrence.get("primitive_id", ""))
        identity = identity_by_id.get(primitive_id)
        if identity is None:
            raise ValueError(
                f"{context} estimator-ledger occurrence identity is missing."
            )
        receipt = prefix_ledger.record_call(
            identity,
            component=str(occurrence.get("component", "")),
            consumer_scope=str(occurrence.get("consumer_scope", "")),
            branch_id=(
                None
                if occurrence.get("branch_id") is None
                else str(occurrence.get("branch_id"))
            ),
        )
        if bool(receipt.charged) != bool(occurrence.get("charged", False)):
            raise ValueError(
                f"{context} estimator-ledger charge identity does not replay."
            )
    prefix_payload = prefix_ledger.to_payload()
    full_summary = full_ledger.summary()
    prefix_summary = prefix_ledger.summary()
    full_occurrence_summary = full_ledger.occurrence_summary()
    prefix_occurrence_summary = prefix_ledger.occurrence_summary()
    accounting_all = accounting.get("all_branch_search_work")
    accounting_schema = str(accounting.get("schema", ""))
    expected_stored_s_alg = (
        int(full_summary["S_unique"])
        if accounting_schema.endswith("_v1")
        else int(full_occurrence_summary["total_call_occurrences"])
    )
    if not isinstance(accounting_all, Mapping) or int(
        accounting_all.get("S_alg", -1)
    ) != expected_stored_s_alg:
        raise ValueError(
            f"{context} estimator-ledger accounting summary does not close."
        )
    provenance = {
        "schema_version": "static_adapt_verified_singleton_estimator_ledger_v1",
        "path": str(sidecar),
        "sha256": file_sha256(sidecar),
        "full_ledger_fingerprint": str(full_payload["ledger_fingerprint"]),
        "restored_prefix_ledger_fingerprint": str(
            prefix_payload["ledger_fingerprint"]
        ),
        "full_occurrence_count": int(len(occurrences)),
        "restored_prefix_occurrence_count": int(len(prefix_occurrences)),
        "excluded_terminal_occurrence_count": int(len(terminal_occurrences)),
        "excluded_terminal_consumer_scopes": sorted(
            {
                str(row.get("consumer_scope", ""))
                for row in terminal_occurrences
                if isinstance(row, Mapping)
            }
        ),
        "full_S_alg": int(full_occurrence_summary["total_call_occurrences"]),
        "restored_prefix_S_alg": int(
            prefix_occurrence_summary["total_call_occurrences"]
        ),
        "full_S_unique": int(full_summary["S_unique"]),
        "restored_prefix_S_unique": int(prefix_summary["S_unique"]),
        "discarded_frontier_reconstructed": False,
        "no_credentials_serialized": True,
    }
    assert_no_secret_material(
        provenance,
        context="verified singleton estimator-ledger provenance",
    )
    return prefix_payload, provenance


def extract_verified_singleton_resume_checkpoint(
    source: ResumeScaffoldSource,
    *,
    expected_phase3_response_coordinate_scope: str | None = None,
    expected_powell_coordinate_chart_policy: str | None = None,
    expected_route_profile_conformance: str | None = None,
    expected_sr_route_profile_request: str | None = None,
    expected_sr_route_profile_contract: Mapping[str, Any] | None = None,
    expected_sr_route_profile_contract_sha256: str | None = None,
) -> ResumeVerifiedSingletonCheckpoint:
    """Validate one complete beam-disabled ``iteration_done`` checkpoint.

    Only an ordered, append-only singleton lineage is accepted.  The source
    must contain its complete history, exact runtime/logical coordinates,
    passing strict-replay receipt, trust state, and a controller-work summary
    that closes against the preserved history.  No branch frontier is inferred
    and optional branch identifiers remain null.
    """

    payload = source.payload
    context = "Verified singleton resume"
    _assert_no_unsupported_modeled_minimum_execution_checkpoint(
        payload,
        context=context,
    )
    powell_chart_validation = validate_resume_powell_coordinate_chart_policy(
        payload,
        expected_policy=expected_powell_coordinate_chart_policy,
        expected_route_profile_conformance=(
            expected_route_profile_conformance
        ),
        context=context,
    )
    sr_route_profile_validation = validate_resume_sr_route_profile_contract(
        payload,
        expected_profile_request=expected_sr_route_profile_request,
        expected_contract=expected_sr_route_profile_contract,
        expected_contract_sha256=expected_sr_route_profile_contract_sha256,
        context=context,
    )
    phase3_response_scope_validation = (
        validate_resume_phase3_response_coordinate_scope(
            payload,
            expected_scope=expected_phase3_response_coordinate_scope,
            expected_profile_request=expected_sr_route_profile_request,
            context=context,
        )
    )
    if extract_formal_manifold_route_composition(payload) is not None:
        raise ValueError(
            f"{context} does not support a Formal-Manifold route checkpoint."
        )

    adapt = _adapt_block(payload)
    checkpoint_raw = payload.get("checkpoint")
    if not isinstance(checkpoint_raw, Mapping):
        raise ValueError(f"{context} requires a checkpoint object.")
    checkpoint = dict(checkpoint_raw)

    def _int_field(
        block: Mapping[str, Any],
        field: str,
        *,
        owner: str,
        minimum: int = 0,
    ) -> int:
        try:
            value = int(block.get(field))
        except (TypeError, ValueError):
            raise ValueError(
                f"{context} requires integer {owner}.{field}."
            ) from None
        if value < int(minimum):
            raise ValueError(
                f"{context} requires {owner}.{field} >= {minimum}."
            )
        return value

    def _finite_vector(value: Any, *, field: str) -> np.ndarray:
        if not isinstance(value, Sequence) or isinstance(
            value, (str, bytes, bytearray)
        ):
            raise ValueError(f"{context} requires {field} array.")
        try:
            vector = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            raise ValueError(f"{context} {field} is not numeric.") from None
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            raise ValueError(
                f"{context} {field} must be finite and non-empty."
            )
        return vector

    if str(checkpoint.get("reason", "")) != "iteration_done":
        raise ValueError(
            f"{context} requires checkpoint.reason='iteration_done'."
        )
    if checkpoint.get("beam_enabled") is not False:
        raise ValueError(f"{context} requires checkpoint.beam_enabled=false.")
    if checkpoint.get("checkpoint_branch_policy") not in {None, ""}:
        raise ValueError(
            f"{context} forbids checkpoint branch/frontier policy metadata."
        )
    if checkpoint.get("branch_id") is not None or checkpoint.get(
        "parent_branch_id"
    ) is not None:
        raise ValueError(f"{context} requires null checkpoint branch IDs.")
    if checkpoint.get("complete") is not False:
        raise ValueError(
            f"{context} requires a pre-terminal current checkpoint."
        )
    if not bool(adapt.get("partial_checkpoint", False)):
        raise ValueError(f"{context} requires adapt_vqe.partial_checkpoint=true.")
    if adapt.get("adapt_beam_enabled") is not False:
        raise ValueError(f"{context} requires adapt_vqe.adapt_beam_enabled=false.")
    if adapt.get("branch_id") is not None or adapt.get("parent_branch_id") is not None:
        raise ValueError(f"{context} requires null adapt_vqe branch IDs.")
    if str(adapt.get("checkpoint_reason", "")) != "iteration_done":
        raise ValueError(
            f"{context} adapt_vqe checkpoint reason disagrees."
        )
    if adapt.get("history_checkpoint_complete") is not True:
        raise ValueError(
            f"{context} requires history_checkpoint_complete=true."
        )
    if adapt.get("stop_reason") not in {None, ""}:
        raise ValueError(f"{context} requires a live, non-terminal checkpoint.")
    beam_replay_telemetry = adapt.get("beam_replay_telemetry")
    if beam_replay_telemetry is not None and beam_replay_telemetry != {}:
        raise ValueError(f"{context} forbids beam replay/frontier telemetry.")
    for field in (
        "formal_manifold_runtime_checkpoint",
        "formal_manifold_warm_state_checkpoint",
        "formal_manifold_query_closure_checkpoint",
    ):
        field_value = adapt.get(field)
        if field_value is not None and field_value != {}:
            raise ValueError(f"{context} does not support adapt_vqe.{field}.")

    final_refit = adapt.get("final_full_refit")
    if not isinstance(final_refit, Mapping):
        raise ValueError(f"{context} requires final_full_refit checkpoint state.")
    if bool(final_refit.get("attempted", False)) or bool(
        final_refit.get("executed", False)
    ):
        raise ValueError(f"{context} rejects a terminal final-refit checkpoint.")
    if int(final_refit.get("nfev", 0) or 0) != 0:
        raise ValueError(f"{context} final-refit nfev must be zero.")

    history_raw = adapt.get("history")
    if not isinstance(history_raw, Sequence) or isinstance(
        history_raw, (str, bytes, bytearray)
    ) or any(not isinstance(row, Mapping) for row in history_raw):
        raise ValueError(f"{context} requires adapt_vqe.history object array.")
    history_cleaned_raw, _removed = _drop_obsolete_admission_rollback_state(
        [dict(row) for row in history_raw]
    )
    history = [dict(row) for row in history_cleaned_raw]
    if not history:
        raise ValueError(f"{context} history must be non-empty.")
    latest_v4_prune_trust_state = _validate_latest_v4_prune_trust_state(
        history=history,
        artifact_profile_request=sr_route_profile_validation[
            "artifact_profile_request"
        ],
        context=context,
    )
    history_count = _int_field(
        adapt,
        "history_count",
        owner="adapt_vqe",
        minimum=1,
    )
    if history_count != len(history):
        raise ValueError(
            f"{context} history_count does not match complete history length."
        )
    history_tail_raw = adapt.get("history_tail")
    if not isinstance(history_tail_raw, Sequence) or isinstance(
        history_tail_raw, (str, bytes, bytearray)
    ) or any(not isinstance(row, Mapping) for row in history_tail_raw):
        raise ValueError(f"{context} history tail is invalid.")
    history_tail_cleaned, _tail_removed = _drop_obsolete_admission_rollback_state(
        [dict(row) for row in history_tail_raw]
    )
    history_tail_count = _int_field(
        adapt,
        "history_tail_count",
        owner="adapt_vqe",
        minimum=0,
    )
    retention = adapt.get("history_tail_retention")
    compact_tail = (
        isinstance(retention, Mapping)
        and str(retention.get("schema", ""))
        == "static_adapt_verified_resume_history_retention_v2"
    )
    if compact_tail:
        requested_limit = _int_field(
            retention,
            "requested_limit",
            owner="adapt_vqe.history_tail_retention",
            minimum=0,
        )
        expected_tail_count = min(requested_limit, history_count)
        expected_tail = (
            []
            if expected_tail_count == 0
            else history[-expected_tail_count:]
        )
        if (
            int(retention.get("serialized_complete_history_count", -1))
            != history_count
            or int(retention.get("serialized_tail_count", -1))
            != expected_tail_count
            or int(retention.get("requested_window_count", -1))
            != expected_tail_count
            or history_tail_count != expected_tail_count
            or len(history_tail_cleaned) != expected_tail_count
            or digest_jsonable(history_tail_cleaned)
            != digest_jsonable(expected_tail)
        ):
            raise ValueError(
                f"{context} compact history tail does not authenticate the "
                "declared complete history suffix."
            )
    else:
        if (
            history_tail_count != history_count
            or len(history_tail_cleaned) != history_count
        ):
            raise ValueError(f"{context} requires the complete history tail.")
        if digest_jsonable(history_tail_cleaned) != digest_jsonable(history):
            raise ValueError(f"{context} history and history tail disagree.")

    controller_round = _int_field(
        checkpoint,
        "depth",
        owner="checkpoint",
        minimum=1,
    )
    if controller_round != history_count:
        raise ValueError(f"{context} checkpoint depth disagrees with history.")
    operator_labels_raw = adapt.get("operators")
    if not isinstance(operator_labels_raw, Sequence) or isinstance(
        operator_labels_raw, (str, bytes, bytearray)
    ):
        raise ValueError(f"{context} requires adapt_vqe.operators array.")
    operator_labels = tuple(str(label) for label in operator_labels_raw)
    if not operator_labels or any(label == "" for label in operator_labels):
        raise ValueError(f"{context} operator labels must be non-empty.")
    ansatz_depth = _int_field(
        adapt,
        "ansatz_depth",
        owner="adapt_vqe",
        minimum=1,
    )
    if ansatz_depth != len(operator_labels):
        raise ValueError(
            f"{context} singleton ansatz depth/operator counts disagree."
        )
    accepted_prune_present = any(
        isinstance(row.get("post_admission_prune"), Mapping)
        and int(row["post_admission_prune"].get("accepted_count", 0) or 0) > 0
        for row in history
    )
    authenticated_v4_pruned_lineage = None
    if accepted_prune_present:
        authenticated_v4_pruned_lineage = _validate_authenticated_v4_pruned_lineage(
            history=history,
            operator_labels=operator_labels,
            ansatz_depth=int(ansatz_depth),
            artifact_profile_request=sr_route_profile_validation[
                "artifact_profile_request"
            ],
            context=context,
        )
        if (
            latest_v4_prune_trust_state
            != authenticated_v4_pruned_lineage["restored_prune_trust_state"]
        ):
            raise ValueError(
                f"{context} latest v4 prune trust state disagrees with its "
                "authenticated deletion lineage."
            )
    elif ansatz_depth != history_count:
        raise ValueError(
            f"{context} singleton ansatz depth/operator/history counts disagree."
        )
    if _int_field(
        checkpoint,
        "ansatz_depth",
        owner="checkpoint",
        minimum=1,
    ) != ansatz_depth:
        raise ValueError(f"{context} checkpoint ansatz depth disagrees.")

    selected_labels: list[str] = []
    for index, row in enumerate(history):
        expected_round = index + 1
        if _int_field(
            row,
            "depth",
            owner=f"history[{index}]",
            minimum=1,
        ) != expected_round:
            raise ValueError(
                f"{context} history depths must be contiguous from one."
            )
        if row.get("branch_id") is not None or row.get("parent_branch_id") is not None:
            raise ValueError(f"{context} history branch IDs must remain null.")
        if _int_field(
            row,
            "batch_size",
            owner=f"history[{index}]",
            minimum=1,
        ) != 1:
            raise ValueError(f"{context} only supports singleton admissions.")
        label = str(row.get("selected_op", ""))
        if label == "":
            raise ValueError(f"{context} history[{index}].selected_op is missing.")
        selected_labels.append(label)
        selected_position = _int_field(
            row,
            "selected_position",
            owner=f"history[{index}]",
            minimum=0,
        )
        if authenticated_v4_pruned_lineage is None and selected_position != index:
            raise ValueError(
                f"{context} requires append-only ordered insertion positions."
            )
        prune = row.get("post_admission_prune")
        if not isinstance(prune, Mapping):
            raise ValueError(f"{context} history[{index}] prune state is invalid.")
        try:
            accepted_prune_count = int(prune.get("accepted_count", 0) or 0)
        except (TypeError, ValueError):
            raise ValueError(
                f"{context} history[{index}] prune count is invalid."
            ) from None
        if (
            accepted_prune_count != 0
            and authenticated_v4_pruned_lineage is None
        ):
            raise ValueError(
                f"{context} cannot reconstruct accepted prune deletion."
            )
        for energy_field in ("energy_before_opt", "energy_after_opt"):
            try:
                energy_value = float(row.get(energy_field))
            except (TypeError, ValueError):
                raise ValueError(
                    f"{context} history[{index}].{energy_field} is invalid."
                ) from None
            if not math.isfinite(energy_value):
                raise ValueError(
                    f"{context} history[{index}].{energy_field} is nonfinite."
                )
    if (
        authenticated_v4_pruned_lineage is None
        and tuple(selected_labels) != operator_labels
    ):
        raise ValueError(
            f"{context} ordered history operators do not match active operators."
        )

    theta_runtime = _finite_vector(
        adapt.get("optimal_point"),
        field="adapt_vqe.optimal_point",
    )
    theta_logical = _finite_vector(
        adapt.get("logical_optimal_point"),
        field="adapt_vqe.logical_optimal_point",
    )
    if authenticated_v4_pruned_lineage is not None:
        final_prefix = authenticated_v4_pruned_lineage[
            "final_active_prefix_checkpoint"
        ]
        prefix_runtime = _finite_vector(
            final_prefix.get("signed_unwrapped_runtime_parameters"),
            field="active_prefix_checkpoint.signed_unwrapped_runtime_parameters",
        )
        prefix_logical = _finite_vector(
            final_prefix.get("signed_unwrapped_logical_parameters"),
            field="active_prefix_checkpoint.signed_unwrapped_logical_parameters",
        )
        if not np.array_equal(prefix_runtime, theta_runtime) or not np.array_equal(
            prefix_logical, theta_logical
        ):
            raise ValueError(
                f"{context} signed active-prefix parameters disagree with the "
                "saved optimizer state."
            )
    if _int_field(
        adapt,
        "num_parameters",
        owner="adapt_vqe",
        minimum=1,
    ) != int(theta_runtime.size):
        raise ValueError(f"{context} runtime theta count disagrees.")
    if _int_field(
        adapt,
        "logical_num_parameters",
        owner="adapt_vqe",
        minimum=1,
    ) != int(theta_logical.size):
        raise ValueError(f"{context} logical theta count disagrees.")
    if int(theta_logical.size) != ansatz_depth:
        raise ValueError(
            f"{context} requires one logical theta per accepted singleton."
        )
    parameterization = adapt.get("parameterization")
    if not isinstance(parameterization, Mapping):
        raise ValueError(f"{context} requires adapt_vqe.parameterization.")
    if _int_field(
        parameterization,
        "runtime_parameter_count",
        owner="adapt_vqe.parameterization",
        minimum=1,
    ) != int(theta_runtime.size):
        raise ValueError(f"{context} parameterization runtime count disagrees.")
    if _int_field(
        parameterization,
        "logical_operator_count",
        owner="adapt_vqe.parameterization",
        minimum=1,
    ) != ansatz_depth:
        raise ValueError(f"{context} parameterization logical count disagrees.")

    runtime_labels = tuple(
        str(term.label) for term in source.runtime_input.selected_terms
    )
    if runtime_labels != operator_labels:
        raise ValueError(f"{context} runtime-loader operators disagree.")
    runtime_theta = np.asarray(
        source.runtime_input.theta_runtime,
        dtype=float,
    ).reshape(-1)
    if not np.array_equal(runtime_theta, theta_runtime):
        raise ValueError(f"{context} runtime-loader theta disagrees.")
    runtime_theta_logical = source.runtime_input.theta_logical
    if runtime_theta_logical is None or not np.array_equal(
        np.asarray(runtime_theta_logical, dtype=float).reshape(-1),
        theta_logical,
    ):
        raise ValueError(f"{context} runtime-loader logical theta disagrees.")

    state_digests: dict[str, str] = {}
    state_nq: dict[str, int] = {}
    for state_name, runtime_state in (
        ("initial_state", source.runtime_input.psi_initial),
        ("ansatz_input_state", source.runtime_input.psi_ref),
    ):
        state_manifest = payload.get(state_name)
        if not isinstance(state_manifest, Mapping):
            raise ValueError(f"{context} requires {state_name} manifest.")
        nq_total = _int_field(
            state_manifest,
            "nq_total",
            owner=state_name,
            minimum=1,
        )
        state_nq[state_name] = nq_total
        try:
            norm = float(state_manifest.get("norm"))
        except (TypeError, ValueError):
            raise ValueError(f"{context} {state_name}.norm is invalid.") from None
        if not math.isfinite(norm) or abs(norm - 1.0) > 1.0e-8:
            raise ValueError(f"{context} {state_name} must be normalized.")
        runtime_size = int(np.asarray(runtime_state, dtype=complex).reshape(-1).size)
        if runtime_size != (1 << nq_total):
            raise ValueError(
                f"{context} {state_name} dimension disagrees with nq_total."
            )
        state_digests[state_name] = digest_jsonable(state_manifest)
    if state_nq["initial_state"] != state_nq["ansatz_input_state"]:
        raise ValueError(f"{context} state qubit counts disagree.")

    strict_replay_raw = adapt.get("strict_replay")
    if not isinstance(strict_replay_raw, Mapping):
        raise ValueError(f"{context} requires strict_replay telemetry.")
    strict_replay = dict(strict_replay_raw)
    if str(strict_replay.get("schema", "")) != "static_adapt_strict_state_replay_v1":
        raise ValueError(f"{context} strict-replay schema is unsupported.")
    if strict_replay.get("passed") is not True:
        raise ValueError(f"{context} strict replay did not pass.")
    try:
        replay_tolerance = float(strict_replay.get("tolerance"))
        replay_l2 = float(strict_replay.get("phase_aligned_l2"))
        replay_fidelity = float(strict_replay.get("fidelity"))
    except (TypeError, ValueError):
        raise ValueError(f"{context} strict-replay values are invalid.") from None
    if (
        not math.isfinite(replay_tolerance)
        or replay_tolerance <= 0.0
        or not math.isfinite(replay_l2)
        or replay_l2 < 0.0
        or replay_l2 > replay_tolerance
        or not math.isfinite(replay_fidelity)
        or replay_fidelity < 0.0
        or replay_fidelity > 1.0 + replay_tolerance
        or (1.0 - replay_fidelity) > max(1.0e-12, 10.0 * replay_tolerance)
    ):
        raise ValueError(f"{context} strict-replay receipt is inconsistent.")

    trust_raw = adapt.get("route_a_trust_region_state")
    if not isinstance(trust_raw, Mapping):
        raise ValueError(f"{context} requires route_a_trust_region_state.")
    trust = dict(trust_raw)
    if str(trust.get("schema", "")) != "route_a_trust_region_state_v1":
        raise ValueError(f"{context} trust-state schema is unsupported.")
    trust_update_count = _int_field(
        trust,
        "update_count",
        owner="route_a_trust_region_state",
        minimum=0,
    )
    for field, allow_zero in (("radius", True), ("reference_radius", False)):
        try:
            value = float(trust.get(field))
        except (TypeError, ValueError):
            raise ValueError(f"{context} trust-state {field} is invalid.") from None
        if not math.isfinite(value) or value < 0.0 or (not allow_zero and value == 0.0):
            raise ValueError(f"{context} trust-state {field} is invalid.")
    last_update = trust.get("last_update")
    if trust_update_count == 0:
        if last_update is not None:
            raise ValueError(
                f"{context} zero-update trust state has a last_update record."
            )
    elif not isinstance(last_update, Mapping):
        raise ValueError(f"{context} trust state lacks last_update.")

    try:
        source_energy = float(adapt.get("energy"))
        last_history_energy = float(history[-1].get("energy_after_opt"))
    except (TypeError, ValueError):
        raise ValueError(f"{context} source energy is invalid.") from None
    if not math.isfinite(source_energy) or not math.isfinite(last_history_energy):
        raise ValueError(f"{context} source energy must be finite.")
    if abs(source_energy - last_history_energy) > 1.0e-12:
        raise ValueError(f"{context} saved energy disagrees with history.")
    source_nfev = _int_field(adapt, "nfev_total", owner="adapt_vqe", minimum=0)
    try:
        history_nfev = int(history[-1].get("nfev_total_after_step"))
    except (TypeError, ValueError):
        raise ValueError(f"{context} last history nfev is invalid.") from None
    if history_nfev != source_nfev:
        raise ValueError(f"{context} saved nfev ledger disagrees with history.")

    source_work_raw = adapt.get("controller_measurement_work_summary")
    if not isinstance(source_work_raw, Mapping):
        raise ValueError(
            f"{context} requires controller_measurement_work_summary."
        )
    source_work = dict(source_work_raw)
    if str(source_work.get("schema", "")) != "controller_measurement_work_proxy_v1":
        raise ValueError(f"{context} controller-work schema is unsupported.")
    if str(source_work.get("beam_run_scope", "")) != "single_route":
        raise ValueError(f"{context} controller-work scope is not single-route.")
    if _int_field(
        source_work,
        "checkpoint_depth",
        owner="controller_measurement_work_summary",
        minimum=1,
    ) != controller_round:
        raise ValueError(f"{context} controller-work checkpoint depth disagrees.")
    reconstructed_work = controller_proxy_from_history_rows(history)
    closure_fields = (
        "events_count",
        "records_evaluated",
        "records_with_group_keys",
        "groups_total",
        "groups_reused",
        "groups_new",
        "shots_total",
        "shots_reused",
        "shots_new",
        "total_groups_new",
        "total_shots_new",
        "reuse_count_cost",
        "candidate_count_total",
        "evaluated_count_total",
        "pre_shortlist_count_total",
        "shortlist_size_total",
        "retained_count_total",
        "rejected_count_total",
        "candidate_work_event_count",
        "candidate_work_missing_event_count",
    )
    for field in closure_fields:
        try:
            source_value = float(source_work.get(field))
            replay_value = float(reconstructed_work.get(field))
        except (TypeError, ValueError):
            raise ValueError(
                f"{context} controller-work field {field} is unresolved."
            ) from None
        if not (
            math.isfinite(source_value)
            and math.isfinite(replay_value)
            and math.isclose(source_value, replay_value, rel_tol=0.0, abs_tol=1.0e-9)
        ):
            raise ValueError(
                f"{context} controller-work ledger does not close for {field}."
            )

    controller_resume_state = _load_verified_singleton_controller_state(
        source,
        authenticated_pruned_lineage=bool(
            authenticated_v4_pruned_lineage is not None
        ),
    )
    selection_resume_state = _load_verified_singleton_selection_state(source)
    (
        estimator_call_ledger_payload,
        estimator_call_ledger_provenance,
    ) = _load_verified_singleton_estimator_ledger(source)

    validation = {
        "schema_version": "static_adapt_verified_singleton_resume_checkpoint_v1",
        "checkpoint_reason": "iteration_done",
        "checkpoint_complete_history": True,
        "history_checkpoint_complete": True,
        "history_count": int(history_count),
        "history_digest": digest_jsonable(history),
        "controller_round": int(controller_round),
        "ansatz_depth": int(ansatz_depth),
        "branch_id": None,
        "parent_branch_id": None,
        "operator_labels_digest": digest_jsonable(operator_labels),
        "theta_runtime_digest": digest_jsonable(theta_runtime.tolist()),
        "theta_logical_digest": digest_jsonable(theta_logical.tolist()),
        "initial_state_digest": state_digests["initial_state"],
        "ansatz_input_state_digest": state_digests["ansatz_input_state"],
        "strict_replay_digest": digest_jsonable(strict_replay),
        "strict_replay_passed": True,
        "saved_energy": float(source_energy),
        "saved_energy_history_abs_discrepancy": float(
            abs(source_energy - last_history_energy)
        ),
        "saved_energy_reconstruction_tolerance": 1.0e-8,
        "route_a_trust_region_state_digest": digest_jsonable(trust),
        "controller_measurement_work_summary_digest": digest_jsonable(source_work),
        "controller_measurement_work_reconstructed_digest": digest_jsonable(
            reconstructed_work
        ),
        "controller_measurement_work_closure_fields": list(closure_fields),
        "controller_measurement_work_closed": True,
        "phase1_residual_opened": bool(
            controller_resume_state["phase1_residual_opened"]
        ),
        "phase1_stage_name": str(
            controller_resume_state["phase1_stage_name"]
        ),
        "maturity_controller_snapshot_digest": digest_jsonable(
            controller_resume_state["controller_snapshot"]
        ),
        "maturity_controller_state_provenance": dict(
            controller_resume_state["provenance"]
        ),
        "selection_parent_pool_size": int(selection_resume_state["pool_size"]),
        "selected_parent_pool_indices_digest": digest_jsonable(
            list(selection_resume_state["ordered_parent_pool_indices"])
        ),
        "selected_logical_candidate_indices_digest": digest_jsonable(
            list(selection_resume_state["ordered_logical_candidate_indices"])
        ),
        "selection_count_state_provenance": dict(
            selection_resume_state["provenance"]
        ),
        "estimator_call_ledger_provenance": dict(
            estimator_call_ledger_provenance
        ),
        "powell_coordinate_chart_policy": powell_chart_validation[
            "resolved_policy"
        ],
        "route_profile_conformance": powell_chart_validation[
            "route_profile_conformance"
        ],
        "powell_coordinate_chart_policy_validation": powell_chart_validation,
        "sr_route_profile_request": sr_route_profile_validation[
            "artifact_profile_request"
        ],
        "sr_route_profile_contract_sha256": sr_route_profile_validation[
            "contract_sha256"
        ],
        "sr_route_profile_contract_validation": sr_route_profile_validation,
        "phase3_response_coordinate_scope": (
            phase3_response_scope_validation["resolved_scope"]
        ),
        "phase3_response_coordinate_scope_validation": (
            phase3_response_scope_validation
        ),
        "phase12_energy_model_policies": dict(
            sr_route_profile_validation["phase12_energy_model_policies"]
        ),
        "phase12_energy_model_policy_validation": dict(
            sr_route_profile_validation[
                "phase12_energy_model_policy_validation"
            ]
        ),
        "discarded_frontier_reconstructed": False,
        "lineage_scope": "complete_singleton_branch_only",
        "authenticated_v4_pruned_lineage": (
            None
            if authenticated_v4_pruned_lineage is None
            else {
                key: value
                for key, value in authenticated_v4_pruned_lineage.items()
                if key != "final_active_prefix_checkpoint"
            }
        ),
        "latest_v4_prune_trust_state": (
            None
            if latest_v4_prune_trust_state is None
            else dict(latest_v4_prune_trust_state)
        ),
        "no_credentials_serialized": True,
    }
    assert_no_secret_material(
        validation,
        context="verified singleton resume validation",
    )
    return ResumeVerifiedSingletonCheckpoint(
        history=tuple(dict(row) for row in history),
        controller_round=int(controller_round),
        ansatz_depth=int(ansatz_depth),
        branch_id=None,
        parent_branch_id=None,
        operator_labels=operator_labels,
        theta_runtime=tuple(float(value) for value in theta_runtime.tolist()),
        theta_logical=tuple(float(value) for value in theta_logical.tolist()),
        route_a_trust_region_state=trust,
        controller_measurement_work_summary=source_work,
        phase1_residual_opened=bool(
            controller_resume_state["phase1_residual_opened"]
        ),
        phase1_stage_name=str(controller_resume_state["phase1_stage_name"]),
        maturity_controller_snapshot=dict(
            controller_resume_state["controller_snapshot"]
        ),
        maturity_controller_state_provenance=dict(
            controller_resume_state["provenance"]
        ),
        selection_parent_pool_size=int(selection_resume_state["pool_size"]),
        selected_parent_pool_indices=tuple(
            int(value)
            for value in selection_resume_state[
                "ordered_parent_pool_indices"
            ]
        ),
        selected_logical_candidate_indices=tuple(
            int(value)
            for value in selection_resume_state[
                "ordered_logical_candidate_indices"
            ]
        ),
        selection_count_state_provenance=dict(
            selection_resume_state["provenance"]
        ),
        estimator_call_ledger_payload=estimator_call_ledger_payload,
        estimator_call_ledger_provenance=estimator_call_ledger_provenance,
        source_energy=float(source_energy),
        strict_replay=strict_replay,
        initial_state_digest=state_digests["initial_state"],
        ansatz_input_state_digest=state_digests["ansatz_input_state"],
        powell_coordinate_chart_policy=powell_chart_validation[
            "resolved_policy"
        ],
        route_profile_conformance=powell_chart_validation[
            "route_profile_conformance"
        ],
        sr_route_profile_request=sr_route_profile_validation[
            "artifact_profile_request"
        ],
        sr_route_profile_contract_sha256=sr_route_profile_validation[
            "contract_sha256"
        ],
        phase3_response_coordinate_scope=(
            phase3_response_scope_validation["resolved_scope"]
        ),
        phase1_score_mode=(
            sr_route_profile_validation["phase12_energy_model_policies"].get(
                "phase1_score_mode"
            )
        ),
        phase1_energy_model=(
            sr_route_profile_validation["phase12_energy_model_policies"].get(
                "phase1_energy_model"
            )
        ),
        phase2_curvature_policy=(
            sr_route_profile_validation["phase12_energy_model_policies"].get(
                "phase2_curvature_policy"
            )
        ),
        phase2_cheap_curvature_proxy_policy=(
            sr_route_profile_validation["phase12_energy_model_policies"].get(
                "phase2_cheap_curvature_proxy_policy"
            )
        ),
        validation=validation,
    )


def _extract_continuation_block(payload: Mapping[str, Any]) -> dict[str, Any]:
    top = payload.get("continuation", None)
    if isinstance(top, Mapping):
        return dict(top)
    adapt = payload.get("adapt_vqe", None)
    if isinstance(adapt, Mapping) and isinstance(adapt.get("continuation", None), Mapping):
        return dict(adapt.get("continuation", {}))
    return {}


def extract_resume_history(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    continuation = _extract_continuation_block(payload)
    rows = continuation.get("selected_scaffold_history", [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    cleaned_rows, _removed = _drop_obsolete_admission_rollback_state(
        [dict(row) for row in rows if isinstance(row, Mapping)]
    )
    return [dict(row) for row in cleaned_rows if isinstance(row, Mapping)]


def extract_resume_optimizer_memory(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    continuation = _extract_continuation_block(payload)
    memory = continuation.get("optimizer_memory", None)
    return dict(memory) if isinstance(memory, Mapping) else None


def run_resume_compile_smoke(
    source: ResumeScaffoldSource,
    *,
    mode: str,
    backend_name: str = "FakeMarrakesh",
    seed_transpiler: int = 7,
    optimization_level: int = 1,
) -> ResumeCompileSmokeResult:
    smoke_mode = str(mode).strip().lower()
    required = bool(smoke_mode == "required")
    if smoke_mode == "off":
        return ResumeCompileSmokeResult(
            required=False,
            executed=False,
            success=False,
            backend_name=str(backend_name),
            compiled_depth=None,
            compiled_size=None,
            compiled_count_2q=None,
            output_json=None,
            error=None,
        )
    try:
        from pipelines.scaffold.adapt_circuit_cost import (
            CompileScoutConfig,
            run_compile_scout,
        )

        cfg = CompileScoutConfig(
            source=ImportedArtifactResolution(
                mode="imported_artifact",
                requested_json=Path(source.artifact_json),
                resolved_json=Path(source.artifact_json),
                source_kind="direct_payload",
                default_subject=False,
            ),
            requested_backend_name=str(backend_name),
            candidate_backends=(str(backend_name),),
            sweep_backends=False,
            seed_transpiler=int(seed_transpiler),
            optimization_level=int(optimization_level),
            output_json=Path(source.artifact_json).with_name(
                f"{Path(source.artifact_json).stem}_resume_compile_smoke.json"
            ),
        )
        payload = run_compile_scout(cfg)
        selected = payload.get("selected_backend", {}) if isinstance(payload, Mapping) else {}
        if not isinstance(selected, Mapping):
            selected = {}
        result = ResumeCompileSmokeResult(
            required=required,
            executed=True,
            success=bool(payload.get("success", False)) if isinstance(payload, Mapping) else False,
            backend_name=(
                str(selected.get("transpile_backend"))
                if selected.get("transpile_backend") not in {None, ""}
                else str(backend_name)
            ),
            compiled_depth=(
                None if selected.get("compiled_depth") is None else int(selected.get("compiled_depth"))
            ),
            compiled_size=(
                None if selected.get("compiled_size") is None else int(selected.get("compiled_size"))
            ),
            compiled_count_2q=(
                None if selected.get("compiled_count_2q") is None else int(selected.get("compiled_count_2q"))
            ),
            output_json=(
                str(payload.get("artifacts", {}).get("output_json"))
                if isinstance(payload.get("artifacts", None), Mapping)
                and payload.get("artifacts", {}).get("output_json") is not None
                else None
            ),
            error=None,
        )
        assert_no_secret_material(result.to_payload(), context="resume compile smoke result")
        return result
    except Exception as exc:
        return ResumeCompileSmokeResult(
            required=required,
            executed=False,
            success=False,
            backend_name=str(backend_name),
            compiled_depth=None,
            compiled_size=None,
            compiled_count_2q=None,
            output_json=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def build_credential_audit() -> dict[str, Any]:
    return {
        "schema_version": "static_hh_adapt_runtime_audit_v1",
        "cli_accepts_credentials": False,
        "environment_serialized": False,
        "runtime_credentials_serialized": False,
        "no_credentials_serialized": True,
    }


__all__ = [
    "ResumeBestFrontierCheckpoint",
    "ResumeCompileSmokeResult",
    "ResumeMatchedScaffold",
    "ResumeScaffoldSource",
    "ResumeVerifiedSingletonCheckpoint",
    "assert_no_secret_cli_values",
    "assert_no_secret_material",
    "build_credential_audit",
    "build_resume_import_summary",
    "contains_secret_marker",
    "digest_jsonable",
    "extract_best_frontier_resume_checkpoint",
    "extract_verified_singleton_resume_checkpoint",
    "extract_formal_manifold_route_composition",
    "extract_resume_history",
    "extract_resume_optimizer_memory",
    "file_sha256",
    "load_static_resume_source",
    "match_resume_scaffold_to_pool",
    "run_resume_compile_smoke",
    "validate_resume_powell_coordinate_chart_policy",
    "validate_resume_phase12_energy_model_policies",
    "validate_resume_phase3_response_coordinate_scope",
    "validate_resume_sr_route_profile_contract",
    "validate_static_hh_resume_source",
]
