"""Optional Paper-I controller extensions.

The mathematical singleton route does not own batch, prune, or beam choices.
An enabled extension is resolved here from an authenticated route contract;
an absent extension carries no dormant policy values.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from pipelines.scaffold.hh_continuation_scoring import (
    BATCH_GEOMETRY_FULL_RESIDUAL_GRAM_HESSIAN_V1,
    BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1,
    BATCH_SEARCH_FEASIBILITY_JOINT_SUBSET_GATE_V1,
    BATCH_SEARCH_POPULATION_RANKED_CHILD_PHASE2_V1,
    BatchSelectionProposal,
    FullScoreConfig,
    combinatorial_reduced_plane_batch_proposals,
    greedy_reduced_plane_batch_proposals,
)

from pipelines.scaffold.hh_continuation_pruning import (
    PRUNE_METRIC_COST_WEIGHT_ANSATZ_ENTRY_DENOMINATOR_V1,
    PRUNE_METRIC_COST_WEIGHT_OFF,
    PRUNE_METRIC_SCHUR_SOLVE_AFFINE_DELETION_GLOBAL_TRUST_V1,
    PRUNE_METRIC_SCHUR_SOLVE_GRADIENT_CORRECTED_V1,
    PRUNE_METRIC_SCHUR_SOLVE_STATIONARY_GW_ZERO_V1,
    PRUNE_POLICY_RECOVERABILITY_LADDER_V1,
    PRUNE_SCHUR_ROUTE_FULL_LOGICAL_FS_TRUST_DELETE_REFIT_V1,
    PRUNE_SCHUR_ROUTE_HESSIAN_COUPLING_V1,
    PRUNE_SCHUR_ROUTE_METRIC_REGULARIZED_V1,
    PruneConfig,
    resolve_prune_tolerance_mode,
)
from pipelines.static_adapt.prune_risk_dataset import (
    PRUNE_PREFILTER_MOTIF_RISK_V1,
    PRUNE_PREFILTER_OFF,
    load_prune_prefilter_profile,
)
from pipelines.static_adapt.batch_ordering import _batch_admission_record_key
from pipelines.static_adapt.joint_linear_solve import (
    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1,
)
from pipelines.static_adapt.sr_snake._controller import (
    _AcceptedPrefixAllWork,
    _ControllerOutcome,
    _DefaultControllerNumericalRuntime,
    _PreparedSelection,
    _ProjectedAcceptedRound,
    _assert_projected_event,
    _configured_stop_receipt,
    _selection_state_matches_accepted,
)
from pipelines.static_adapt.sr_snake._selection import (
    _CombinatorialBatchAdmissionDecision,
    _GreedyBatchAdmissionDecision,
    _SingletonAdmissionDecision,
    _select_combinatorial_batch,
    _select_greedy_batch,
    _select_singleton,
)
from pipelines.static_adapt.sr_snake._transition import (
    _AcceptedCombinatorialBatchTransition,
    _AcceptedGreedyBatchTransition,
    _AcceptedSingletonTransition,
    _AcceptedStateSnapshot,
    _CheckpointReadyAcceptedStateEvent,
    _TransitionWorkspace,
    _transition_combinatorial_batch,
    _transition_greedy_batch,
    _transition_singleton,
)
from pipelines.static_adapt.sr_snake.contracts import (
    CombinatorialBatchAdmission,
    GreedyBatchAdmission,
    SRStopPolicy,
    SingletonAdmission,
    StopReceipt,
)


PRUNING_SETTING_NAMES = (
    "phase1_prune_policy",
    "phase1_prune_mode",
    "phase1_prune_fraction",
    "phase1_prune_min_candidates",
    "phase1_prune_max_candidates",
    "phase1_prune_max_regression",
    "phase1_prune_tolerance_mode",
    "phase1_prune_tolerance_shot_coeff",
    "phase1_prune_tolerance_screen_coeff",
    "phase1_prune_tolerance_chem",
    "phase1_prune_tolerance_rel_coeff",
    "phase1_prune_tolerance_target_energy",
    "phase1_prune_retained_gain_ratio",
    "phase1_prune_protect_steps",
    "phase1_prune_cooldown_steps",
    "phase1_prune_local_window_size",
    "phase1_prune_recovery_trust_radius",
    "phase1_prune_schur_nomination_route",
    "phase1_prune_metric_schur_mu",
    "phase1_prune_metric_schur_solve_mode",
    "phase1_prune_metric_schur_cost_weighting",
    "phase1_prune_trust_update_policy",
    "phase1_prune_metric_mu_update_policy",
    "phase1_prune_endpoint_overlap_policy",
    "phase1_prune_old_fraction",
    "phase1_prune_checkpoint_period",
    "phase1_prune_live_min_depth",
    "phase1_prune_maturity_threshold",
    "phase1_prune_snr_threshold",
    "phase1_prune_prefilter_policy",
    "phase1_prune_prefilter_json",
    "phase1_prune_risk_threshold",
    "phase1_prune_prefilter_max_candidates",
)

PRUNING_RUNTIME_KEYS = frozenset(
    {"phase1_prune_enabled", *PRUNING_SETTING_NAMES}
)

BATCH_RUNTIME_KEYS = frozenset(
    {
        "phase2_enable_batching",
        "phase3_enable_batching",
        "phase2_batch_selection_mode",
        "phase3_batch_selection_mode",
        "phase3_batch_prefilter_mode",
        "phase3_batch_order_selection_mode",
        "phase3_batch_order_max_permutations",
        "phase2_batch_target_size",
        "phase2_batch_size_cap",
        "phase2_batch_near_degenerate_ratio",
        "phase2_batch_rank_rel_tol",
        "phase2_batch_additivity_tol",
        "phase2_compat_overlap_weight",
        "phase2_compat_comm_weight",
        "phase2_compat_curv_weight",
        "phase2_compat_sched_weight",
        "phase2_compat_measure_weight",
    }
)

BEAM_RUNTIME_KEYS = frozenset(
    {
        "adapt_beam_live_branches",
        "adapt_beam_children_per_parent",
        "adapt_beam_terminated_keep",
        "adapt_beam_terminal_archive_mode",
        "adapt_beam_lambda",
        "adapt_beam_parent_workers",
        "phase3_tie_beam_score_ratio",
        "phase3_tie_beam_abs_tol",
        "phase3_tie_beam_max_branches",
        "phase3_tie_beam_max_late_coordinate",
        "phase3_tie_beam_min_depth_left",
    }
)


@dataclass(frozen=True, slots=True)
class BatchExtension:
    """Complete Paper-I choices for one enabled batch admission policy."""

    strategy: str
    maximum_size: int
    search_window_size: int | None

    def __post_init__(self) -> None:
        strategy = str(self.strategy).strip().lower()
        if strategy not in {"greedy", "combinatorial"}:
            raise ValueError("Batch strategy must be greedy or combinatorial.")
        maximum_size = int(self.maximum_size)
        if isinstance(self.maximum_size, bool) or maximum_size < 1:
            raise ValueError("Batch maximum_size must be a positive integer.")
        if maximum_size > 5:
            raise ValueError("Batch maximum_size must not exceed 5.")
        search_window_size = self.search_window_size
        if search_window_size is not None:
            resolved_window = int(search_window_size)
            if isinstance(search_window_size, bool) or resolved_window < 1:
                raise ValueError(
                    "Batch search_window_size must be a positive integer."
                )
            search_window_size = resolved_window
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "maximum_size", maximum_size)
        object.__setattr__(self, "search_window_size", search_window_size)


@dataclass(frozen=True, slots=True)
class BeamExtension:
    """Complete Paper-I choices for one enabled fork-local beam."""

    live_parent_branches: int
    admission_children_per_parent: int
    maximum_admission_children_per_round: int
    s_alg_weight: float
    calibration_status: str = field(
        default="uncalibrated_default",
        init=False,
    )

    def __post_init__(self) -> None:
        choices = {
            "live_parent_branches": self.live_parent_branches,
            "admission_children_per_parent": self.admission_children_per_parent,
            "maximum_admission_children_per_round": (
                self.maximum_admission_children_per_round
            ),
        }
        resolved: dict[str, int] = {}
        for name, value in choices.items():
            normalized = int(value)
            if isinstance(value, bool) or normalized < 1:
                raise ValueError(f"{name} must be a positive integer.")
            resolved[name] = normalized
        if resolved["admission_children_per_parent"] < 2:
            raise ValueError(
                "fork-local beam requires at least two admission children "
                "per parent."
            )
        if (
            resolved["maximum_admission_children_per_round"]
            < resolved["admission_children_per_parent"]
        ):
            raise ValueError(
                "maximum_admission_children_per_round must be at least "
                "admission_children_per_parent."
            )
        weight = float(self.s_alg_weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("s_alg_weight must be finite and positive.")
        for name, value in resolved.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "s_alg_weight", weight)


@dataclass(frozen=True, slots=True)
class _DisabledLegacyBeamState:
    """Un-enableable state retained only while the old monolith is removed."""

    live_branches_requested: int = field(default=1, init=False)
    children_per_parent_requested: None = field(default=None, init=False)
    terminated_keep_requested: None = field(default=None, init=False)
    live_branches_effective: int = field(default=1, init=False)
    children_per_parent_effective: int = field(default=1, init=False)
    terminated_keep_effective: int = field(default=0, init=False)
    beam_enabled: bool = field(default=False, init=False)
    terminal_archive_mode: str = field(default="disabled", init=False)


_DISABLED_LEGACY_BEAM_STATE = _DisabledLegacyBeamState()


def beam_extension_from_policy(policy: object) -> BeamExtension | None:
    """Translate the public typed beam choice into one extension value."""

    kind = str(getattr(policy, "kind", "")).strip().lower()
    if kind == "off":
        return None
    if kind != "fork_local":
        raise TypeError("Unknown beam policy for the beam extension.")
    required = (
        "live_parent_branches",
        "admission_children_per_parent",
        "maximum_admission_children_per_round",
        "s_alg_weight",
    )
    missing = [name for name in required if getattr(policy, name, None) is None]
    if missing:
        raise ValueError(
            "Enabled beam is missing required choices: " + ", ".join(missing)
        )
    return BeamExtension(
        live_parent_branches=int(policy.live_parent_branches),
        admission_children_per_parent=int(policy.admission_children_per_parent),
        maximum_admission_children_per_round=int(
            policy.maximum_admission_children_per_round
        ),
        s_alg_weight=float(policy.s_alg_weight),
    )


def batch_extension_from_admission(admission: object) -> BatchExtension | None:
    """Translate the public typed admission choice into one extension value."""

    kind = str(getattr(admission, "kind", "")).strip().lower()
    if kind == "singleton":
        return None
    if kind not in {"greedy_batch", "combinatorial_batch"}:
        raise TypeError("Unknown admission policy for the batch extension.")
    maximum_size = getattr(admission, "maximum_size", None)
    if maximum_size is None:
        raise ValueError("Enabled batching requires maximum_size.")
    if kind == "combinatorial_batch":
        search_window_size = getattr(
            admission,
            "resolved_search_window_size",
            None,
        )
    else:
        search_window_size = getattr(admission, "search_window_size", None)
    return BatchExtension(
        strategy=("greedy" if kind == "greedy_batch" else "combinatorial"),
        maximum_size=int(maximum_size),
        search_window_size=(
            None if search_window_size is None else int(search_window_size)
        ),
    )


def run_batch_proposals(
    extension: BatchExtension,
    ranked_records: Sequence[Mapping[str, Any]],
    *,
    phase2_score_config: FullScoreConfig,
    selected_ops: Sequence[Any],
    theta: np.ndarray,
    reference_state: np.ndarray,
    current_state: np.ndarray,
    compiled_hamiltonian: Any,
    novelty_oracle: Any,
    curvature_oracle: Any,
    compiled_cache: Any,
    pauli_action_cache: Any,
    tie_break_score_key: Any,
    estimator_service: Any,
) -> tuple[BatchSelectionProposal, Mapping[str, Any]]:
    """Score the bounded, generator-distinct Paper-I batch feasible set."""

    strategy = extension.strategy
    if strategy == "greedy":
        strategy_label = "Greedy"
        proposal_builder = greedy_reduced_plane_batch_proposals
        pair_consumer_scope = "greedy_batch_pair_geometry_all_evaluated"
        accounting_schema = "greedy_batch_pair_estimator_accounting_v1"
    elif strategy == "combinatorial":
        strategy_label = "Combinatorial"
        proposal_builder = combinatorial_reduced_plane_batch_proposals
        pair_consumer_scope = (
            "combinatorial_batch_pair_geometry_all_evaluated"
        )
        accounting_schema = "combinatorial_batch_pair_estimator_accounting_v1"
    else:  # guarded by BatchExtension, retained as a closed-world assertion.
        raise AssertionError(f"Unsupported batch strategy: {strategy!r}")

    ranked = [dict(record) for record in ranked_records]
    if extension.search_window_size is None:
        search_population = ranked
    else:
        search_population = ranked[: int(extension.search_window_size)]
    if not search_population:
        raise RuntimeError(
            f"{strategy_label} batch search received an empty ranked window."
        )
    selected_ops_now = list(selected_ops)
    theta_now = np.asarray(theta, dtype=float)
    current_state_now = np.asarray(current_state, dtype=complex)
    batch_cfg = replace(
        phase2_score_config,
        batch_target_size=int(extension.maximum_size),
        batch_size_cap=int(extension.maximum_size),
        batch_near_degenerate_ratio=0.0,
        batch_search_pool_size=(
            0
            if extension.search_window_size is None
            else int(extension.search_window_size)
        ),
        batch_search_population_mode=(
            BATCH_SEARCH_POPULATION_RANKED_CHILD_PHASE2_V1
        ),
        batch_search_feasibility_policy=(
            BATCH_SEARCH_FEASIBILITY_JOINT_SUBSET_GATE_V1
        ),
        batch_geometry_mode=BATCH_GEOMETRY_FULL_RESIDUAL_GRAM_HESSIAN_V1,
        batch_joint_linear_solve_policy=(
            JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1
        ),
        batch_joint_context_mode=BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1,
        batch_active_context_indices=tuple(range(len(selected_ops_now))),
    )
    pair_accounting_rows: list[dict[str, Any]] = []

    def _observe_joint_pair(payload: Mapping[str, Any]) -> None:
        left_record = payload.get("left_record")
        right_record = payload.get("right_record")
        if not isinstance(left_record, Mapping) or not isinstance(
            right_record,
            Mapping,
        ):
            raise RuntimeError(
                f"{strategy_label} pair observer lost its measured records."
            )
        accounting = estimator_service._record_candidate_pair_geometry_primitives(
            state=current_state_now,
            selected_ops_now=selected_ops_now,
            logical_theta_now=theta_now,
            left_record=left_record,
            right_record=right_record,
            consumer_scope=pair_consumer_scope,
            pair_cache_key=str(payload.get("cache_key", "")),
            winning_pair=False,
            batch_kind=strategy,
        )
        pair_accounting_rows.append(
            {
                **dict(accounting),
                "left_record_key": list(
                    _batch_admission_record_key(left_record)
                ),
                "right_record_key": list(
                    _batch_admission_record_key(right_record)
                ),
                "cache_hit": bool(payload.get("cache_hit", False)),
                "physical_evaluation_performed": bool(
                    payload.get("physical_evaluation_performed", False)
                ),
                "gram_entry": float(payload["gram_entry"]),
                "hessian_entry": float(payload["hessian_entry"]),
                "state_reconstruction_delta_norm": float(
                    payload["state_reconstruction_delta_norm"]
                ),
            }
        )

    proposals, raw_summary = proposal_builder(
        search_population,
        cfg=batch_cfg,
        selected_ops=selected_ops_now,
        theta=theta_now,
        psi_ref=np.asarray(reference_state, dtype=complex),
        psi_state=current_state_now,
        h_compiled=compiled_hamiltonian,
        novelty_oracle=novelty_oracle,
        curvature_oracle=curvature_oracle,
        compiled_cache=compiled_cache,
        pauli_action_cache=pauli_action_cache,
        tie_break_score_key=tie_break_score_key,
        max_proposals=1,
        joint_pair_observer=_observe_joint_pair,
    )
    if not proposals:
        raise RuntimeError(
            f"{strategy_label} reduced-plane search found no feasible proposal."
        )
    winning_keys = {
        _batch_admission_record_key(record) for record in proposals[0].records
    }
    for row in pair_accounting_rows:
        left_key = tuple(row["left_record_key"])
        right_key = tuple(row["right_record_key"])
        row["winning_pair"] = bool(
            left_key in winning_keys and right_key in winning_keys
        )
    geometry_workspace = raw_summary.get("geometry_workspace", {})
    geometry_workspace = (
        geometry_workspace
        if isinstance(geometry_workspace, Mapping)
        else {}
    )
    expected_pair_receipts = [
        dict(row)
        for row in geometry_workspace.get("joint_pair_receipts", ())
        if isinstance(row, Mapping)
    ]
    if len(pair_accounting_rows) != len(expected_pair_receipts):
        raise RuntimeError(
            f"{strategy_label} pair estimator accounting does not match "
            "the scoring workspace."
        )
    physical_evaluation_count = sum(
        int(bool(row["physical_evaluation_performed"]))
        for row in pair_accounting_rows
    )
    if physical_evaluation_count != int(
        geometry_workspace.get("joint_pair_cache_miss_count", 0)
    ):
        raise RuntimeError(
            f"{strategy_label} pair physical-evaluation sentinel does not "
            "match the scoring cache misses."
        )
    pair_estimator_accounting = {
        "schema": accounting_schema,
        "all_evaluated_pair_count": len(pair_accounting_rows),
        "winning_pair_count": sum(
            int(bool(row["winning_pair"])) for row in pair_accounting_rows
        ),
        "physical_pair_evaluation_count": physical_evaluation_count,
        "ledger_occurrence_count": sum(
            len(row["primitive_rows"]) for row in pair_accounting_rows
        ),
        "occurrences_per_pair": 2,
        "primitive_semantics": (
            "one_metric_and_one_energy_hessian_occurrence_per_"
            "required_candidate_pair_v1"
        ),
        "all_vs_winning_semantics": (
            "all_required_search_pairs_are_ledgered_once_and_winning_"
            "pairs_are_an_attribution_subset_v1"
        ),
        "pairs": pair_accounting_rows,
    }
    summary = {
        **dict(raw_summary),
        "public_search_window_size": extension.search_window_size,
        "ranked_population_count": len(ranked),
        "ranked_window_count": len(search_population),
        "ranked_window_truncated": bool(
            len(search_population) < len(ranked)
        ),
        "near_degenerate_shell_active": False,
        "pair_geometry_estimator_accounting": pair_estimator_accounting,
    }
    return proposals[0], summary


@dataclass(frozen=True, slots=True)
class PruningExtension:
    """Complete choices for one enabled pruning policy.

    There are deliberately no field defaults.  The extension is either absent
    or its authenticated contract supplies every value used by the controller.
    """

    settings: Mapping[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        values = {str(key): value for key, value in self.settings.items()}
        missing = [name for name in PRUNING_SETTING_NAMES if name not in values]
        if missing:
            raise ValueError(
                "Enabled pruning is missing required choices: "
                + ", ".join(missing)
            )
        unknown = sorted(set(values).difference(PRUNING_SETTING_NAMES))
        if unknown:
            raise ValueError(
                "Enabled pruning received unknown choices: "
                + ", ".join(unknown)
            )
        object.__setattr__(self, "settings", MappingProxyType(values))

    def __getitem__(self, name: str) -> Any:
        return self.settings[str(name)]

    def to_runtime_dict(self) -> dict[str, Any]:
        return dict(self.settings)


@dataclass(frozen=True, slots=True)
class Extensions:
    """Resolved optional behavior composed after the singleton route."""

    batch: BatchExtension | None = None
    pruning: PruningExtension | None = None
    beam: BeamExtension | None = None

    def __iter__(self) -> Iterator[object]:
        if self.batch is not None:
            yield self.batch
        if self.pruning is not None:
            yield self.pruning
        if self.beam is not None:
            yield self.beam


NO_EXTENSIONS = Extensions()


@dataclass(frozen=True, slots=True)
class PruningRuntime:
    """Validated internal state for an enabled pruning extension."""

    config: PruneConfig
    mode: str
    checkpoint_period: int
    live_min_depth: int
    maturity_threshold: float
    snr_threshold: float
    prefilter_policy: str
    prefilter_path: Path | None
    prefilter_profile: Mapping[str, Any] | None
    risk_threshold: float
    prefilter_max_candidates: int
    trust_update_policy: str
    metric_mu_update_policy: str
    endpoint_overlap_policy: str

    def modes(self, *, phase1_enabled: bool) -> tuple[bool, bool]:
        enabled = bool(phase1_enabled)
        return (
            bool(enabled and self.mode in {"live", "both"}),
            bool(enabled and self.mode in {"final", "both"}),
        )


def _nonnegative_finite(value: Any, *, name: str) -> float:
    if value is None:
        return 0.0
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return resolved


def resolve_pruning_runtime(
    extension: PruningExtension | None,
    *,
    repo_root: Path,
) -> PruningRuntime | None:
    """Validate the conditional pruning interview only when it is enabled."""

    if extension is None:
        return None
    values = extension.settings
    policy = str(values["phase1_prune_policy"]).strip().lower()
    mode = str(values["phase1_prune_mode"]).strip().lower()
    if mode not in {"live", "final", "both"}:
        raise ValueError(f"Unsupported phase1_prune_mode: {mode}")
    prefilter_policy = str(
        values["phase1_prune_prefilter_policy"]
    ).strip().lower()
    if prefilter_policy not in {
        PRUNE_PREFILTER_OFF,
        PRUNE_PREFILTER_MOTIF_RISK_V1,
    }:
        raise ValueError(
            "Unsupported phase1_prune_prefilter_policy: "
            f"{prefilter_policy}"
        )
    tolerance_mode_requested = str(
        values["phase1_prune_tolerance_mode"]
    ).strip().lower()
    tolerance_mode = resolve_prune_tolerance_mode(
        mode=tolerance_mode_requested,
        prune_policy=policy,
    )
    nomination_route = str(
        values["phase1_prune_schur_nomination_route"]
    ).strip().lower()
    if nomination_route not in {
        PRUNE_SCHUR_ROUTE_HESSIAN_COUPLING_V1,
        PRUNE_SCHUR_ROUTE_METRIC_REGULARIZED_V1,
        PRUNE_SCHUR_ROUTE_FULL_LOGICAL_FS_TRUST_DELETE_REFIT_V1,
    }:
        raise ValueError(
            "Unsupported phase1_prune_schur_nomination_route: "
            f"{nomination_route}"
        )
    solve_mode = str(
        values["phase1_prune_metric_schur_solve_mode"]
    ).strip().lower()
    if solve_mode not in {
        PRUNE_METRIC_SCHUR_SOLVE_STATIONARY_GW_ZERO_V1,
        PRUNE_METRIC_SCHUR_SOLVE_GRADIENT_CORRECTED_V1,
        PRUNE_METRIC_SCHUR_SOLVE_AFFINE_DELETION_GLOBAL_TRUST_V1,
    }:
        raise ValueError(
            "Unsupported phase1_prune_metric_schur_solve_mode: "
            f"{solve_mode}"
        )
    cost_weighting = str(
        values["phase1_prune_metric_schur_cost_weighting"]
    ).strip().lower()
    if cost_weighting not in {
        PRUNE_METRIC_COST_WEIGHT_ANSATZ_ENTRY_DENOMINATOR_V1,
        PRUNE_METRIC_COST_WEIGHT_OFF,
    }:
        raise ValueError(
            "Unsupported phase1_prune_metric_schur_cost_weighting: "
            f"{cost_weighting}"
        )
    trust_update = str(
        values["phase1_prune_trust_update_policy"]
    ).strip().lower()
    if trust_update not in {"off", "modeled_local_fs_conservative_v1"}:
        raise ValueError(
            "Unsupported phase1_prune_trust_update_policy: "
            f"{trust_update}"
        )
    metric_mu_update = str(
        values["phase1_prune_metric_mu_update_policy"]
    ).strip().lower()
    if metric_mu_update not in {
        "off",
        "same_trial_underprediction_monotone_v1",
    }:
        raise ValueError(
            "Unsupported phase1_prune_metric_mu_update_policy: "
            f"{metric_mu_update}"
        )
    endpoint_overlap = str(
        values["phase1_prune_endpoint_overlap_policy"]
    ).strip().lower()
    if endpoint_overlap not in {"off", "energy_safe_trial_only_v1"}:
        raise ValueError(
            "Unsupported phase1_prune_endpoint_overlap_policy: "
            f"{endpoint_overlap}"
        )
    local_window_size = max(
        0, int(values["phase1_prune_local_window_size"])
    )
    trust_radius = _nonnegative_finite(
        values["phase1_prune_recovery_trust_radius"],
        name="phase1_prune_recovery_trust_radius",
    )
    if nomination_route == PRUNE_SCHUR_ROUTE_FULL_LOGICAL_FS_TRUST_DELETE_REFIT_V1:
        if solve_mode != PRUNE_METRIC_SCHUR_SOLVE_AFFINE_DELETION_GLOBAL_TRUST_V1:
            raise ValueError(
                "Full-logical FS trust pruning requires "
                "affine_deletion_global_trust_v1."
            )
        if local_window_size != 0:
            raise ValueError(
                "Full-logical FS trust pruning requires local_window_size=0."
            )
        if trust_radius <= 0.0:
            raise ValueError(
                "Full-logical FS trust pruning requires a positive radius."
            )
        if endpoint_overlap != "off":
            raise ValueError(
                "The query-neutral prune route forbids endpoint-overlap probes."
            )

    target_energy = values["phase1_prune_tolerance_target_energy"]
    if target_energy is not None and not math.isfinite(float(target_energy)):
        raise ValueError(
            "phase1_prune_tolerance_target_energy must be finite."
        )
    prefilter_path: Path | None = None
    prefilter_profile: Mapping[str, Any] | None = None
    if prefilter_policy == PRUNE_PREFILTER_MOTIF_RISK_V1:
        raw_path = values["phase1_prune_prefilter_json"]
        if raw_path in {None, ""}:
            raise ValueError(
                "motif_risk_v1 pruning requires phase1_prune_prefilter_json."
            )
        prefilter_path = Path(str(raw_path))
        if not prefilter_path.is_absolute():
            prefilter_path = repo_root / prefilter_path
        prefilter_profile = load_prune_prefilter_profile(prefilter_path)

    recoverability = policy == PRUNE_POLICY_RECOVERABILITY_LADDER_V1
    maximum_candidates = max(
        1, int(values["phase1_prune_max_candidates"])
    )
    config = PruneConfig(
        policy=policy,
        max_candidates=maximum_candidates,
        min_candidates=max(1, int(values["phase1_prune_min_candidates"])),
        fraction_candidates=max(
            0.0, float(values["phase1_prune_fraction"])
        ),
        max_regression=_nonnegative_finite(
            values["phase1_prune_max_regression"],
            name="phase1_prune_max_regression",
        ),
        tolerance_mode_requested=tolerance_mode_requested,
        tolerance_mode=tolerance_mode,
        tolerance_shot_coeff=_nonnegative_finite(
            values["phase1_prune_tolerance_shot_coeff"],
            name="phase1_prune_tolerance_shot_coeff",
        ),
        tolerance_screen_coeff=_nonnegative_finite(
            values["phase1_prune_tolerance_screen_coeff"],
            name="phase1_prune_tolerance_screen_coeff",
        ),
        tolerance_chem=_nonnegative_finite(
            values["phase1_prune_tolerance_chem"],
            name="phase1_prune_tolerance_chem",
        ),
        tolerance_rel_coeff=_nonnegative_finite(
            values["phase1_prune_tolerance_rel_coeff"],
            name="phase1_prune_tolerance_rel_coeff",
        ),
        tolerance_target_energy=(
            None if target_energy is None else float(target_energy)
        ),
        retained_gain_ratio=max(
            0.0, float(values["phase1_prune_retained_gain_ratio"])
        ),
        protect_steps=max(0, int(values["phase1_prune_protect_steps"])),
        cooldown_steps=max(0, int(values["phase1_prune_cooldown_steps"])),
        local_window_size=local_window_size,
        surrogate_recovery_trust_radius=trust_radius,
        schur_nomination_route=nomination_route,
        metric_schur_mu=_nonnegative_finite(
            values["phase1_prune_metric_schur_mu"],
            name="phase1_prune_metric_schur_mu",
        ),
        metric_schur_solve_mode=solve_mode,
        metric_schur_cost_weighting=cost_weighting,
        old_fraction=min(
            1.0, max(0.0, float(values["phase1_prune_old_fraction"]))
        ),
        surrogate_enabled=recoverability,
        surrogate_nomination_gate_enabled=recoverability,
        surrogate_nomination_gate_factor=1.0,
        surrogate_exact_trial_cap=(1 if recoverability else maximum_candidates),
    )
    return PruningRuntime(
        config=config,
        mode=mode,
        checkpoint_period=max(
            1, int(values["phase1_prune_checkpoint_period"])
        ),
        live_min_depth=max(0, int(values["phase1_prune_live_min_depth"])),
        maturity_threshold=max(
            0.0, float(values["phase1_prune_maturity_threshold"])
        ),
        snr_threshold=max(0.0, float(values["phase1_prune_snr_threshold"])),
        prefilter_policy=prefilter_policy,
        prefilter_path=prefilter_path,
        prefilter_profile=prefilter_profile,
        risk_threshold=_nonnegative_finite(
            values["phase1_prune_risk_threshold"],
            name="phase1_prune_risk_threshold",
        ),
        prefilter_max_candidates=max(
            0, int(values["phase1_prune_prefilter_max_candidates"])
        ),
        trust_update_policy=trust_update,
        metric_mu_update_policy=metric_mu_update,
        endpoint_overlap_policy=endpoint_overlap,
    )


def extensions_from_route_contract(
    contract: Mapping[str, Any] | None,
) -> Extensions:
    """Resolve enabled extensions without retaining disabled-policy choices."""

    if contract is None:
        return NO_EXTENSIONS
    execution_settings = contract.get("execution_settings")
    if not isinstance(execution_settings, Mapping):
        raise ValueError("The route contract lacks execution settings.")
    if not bool(execution_settings.get("phase1_prune_enabled", False)):
        return NO_EXTENSIONS
    return Extensions(
        batch=None,
        pruning=PruningExtension(
            {
                name: execution_settings[name]
                for name in PRUNING_SETTING_NAMES
                if name in execution_settings
            }
        )
    )


def without_extension_runtime_keys(
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Project core runtime settings without optional-extension controls."""

    return {
        str(key): value
        for key, value in settings.items()
        if str(key) not in PRUNING_RUNTIME_KEYS
        and str(key) not in BATCH_RUNTIME_KEYS
        and str(key) not in BEAM_RUNTIME_KEYS
    }

@dataclass(frozen=True, slots=True)
class _ForkLocalBeamBranch:
    """One live accepted lineage in the canonical bounded beam."""

    runtime: _DefaultControllerNumericalRuntime = field(repr=False)
    state: _AcceptedStateSnapshot
    accepted_states: tuple[_AcceptedStateSnapshot, ...]
    transitions: tuple[
        _AcceptedSingletonTransition
        | _AcceptedGreedyBatchTransition
        | _AcceptedCombinatorialBatchTransition,
        ...,
    ]
    events: tuple[_CheckpointReadyAcceptedStateEvent, ...]
    projected_rounds: tuple[_ProjectedAcceptedRound, ...]
    branch_ids: tuple[str, ...]
    lineage_s_alg: int
    comparison_score: float
    stop: StopReceipt | None

    def __post_init__(self) -> None:
        cardinalities = {
            len(self.accepted_states),
            len(self.transitions),
            len(self.events),
            len(self.projected_rounds),
            len(self.branch_ids),
        }
        if len(cardinalities) != 1:
            raise ValueError(
                "Beam lineage states, transitions, events, projections, and "
                "branch IDs must be one-to-one."
            )
        if self.lineage_s_alg < 0:
            raise ValueError("Beam lineage S_alg must be nonnegative.")


def _beam_branch_sort_key(
    branch: _ForkLocalBeamBranch,
) -> tuple[float, float, int, tuple[str, ...]]:
    """Return the deterministic settled fork-local comparison key."""

    return (
        float(branch.comparison_score),
        float(branch.state.accepted_energy),
        int(branch.lineage_s_alg),
        tuple(branch.branch_ids),
    )


def _run_default_fork_local_beam_controller(
    runtime: _DefaultControllerNumericalRuntime,
    stop_policy: SRStopPolicy,
    admission: (
        SingletonAdmission
        | GreedyBatchAdmission
        | CombinatorialBatchAdmission
    ),
    beam: BeamExtension,
) -> _ControllerOutcome:
    """Run the bounded direct-controller beam with global work accounting.

    Every parent is replaced by accepted children.  No unchanged parent is
    retained, every evaluated child remains in the shared estimator ledger,
    and only the selected lineage is projected as the scientific trajectory.
    """

    if not isinstance(stop_policy, SRStopPolicy):
        raise TypeError("stop_policy must be an SRStopPolicy")
    if not isinstance(
        admission,
        (
            SingletonAdmission,
            GreedyBatchAdmission,
            CombinatorialBatchAdmission,
        ),
    ):
        raise TypeError("beam admission policy has the wrong type")
    if not isinstance(beam, BeamExtension):
        raise TypeError("beam must be a BeamExtension")

    initial_state = runtime.initial_accepted_state
    initial_s_alg = int(runtime.beam_executed_s_alg())
    resume_branch_ids, resume_lineage_s_alg = runtime.beam_resume_seed()
    frontier = [
        _ForkLocalBeamBranch(
            runtime=runtime,
            state=initial_state,
            accepted_states=(),
            transitions=(),
            events=(),
            projected_rounds=(),
            branch_ids=(),
            lineage_s_alg=int(resume_lineage_s_alg),
            comparison_score=float(
                initial_state.accepted_energy
                + float(beam.s_alg_weight) * resume_lineage_s_alg
            ),
            stop=None,
        )
    ]
    branch_counter = 0
    round_audits: list[dict[str, Any]] = []
    accepted_prefix_all_work: list[_AcceptedPrefixAllWork] = []
    winner: _ForkLocalBeamBranch | None = None
    owned_children: list[_DefaultControllerNumericalRuntime] = []

    def _close_owned_child(
        child: _DefaultControllerNumericalRuntime,
    ) -> None:
        if child is runtime:
            return
        for index, owned_child in enumerate(owned_children):
            if owned_child is child:
                del owned_children[index]
                child.close()
                return

    def _select(
        prepared: _PreparedSelection,
    ) -> (
        _SingletonAdmissionDecision
        | _GreedyBatchAdmissionDecision
        | _CombinatorialBatchAdmissionDecision
    ):
        if isinstance(admission, GreedyBatchAdmission):
            return _select_greedy_batch(
                prepared.controller_state,
                prepared.workspace,
                maximum_size=admission.maximum_size,
                search_window_size=admission.search_window_size,
            )
        if isinstance(admission, CombinatorialBatchAdmission):
            return _select_combinatorial_batch(
                prepared.controller_state,
                prepared.workspace,
                maximum_size=admission.maximum_size,
                search_window_size=(
                    admission.resolved_search_window_size
                ),
            )
        return _select_singleton(
            prepared.controller_state,
            prepared.workspace,
        )

    def _transition(
        state: _AcceptedStateSnapshot,
        decision: (
            _SingletonAdmissionDecision
            | _GreedyBatchAdmissionDecision
            | _CombinatorialBatchAdmissionDecision
        ),
        workspace: _TransitionWorkspace,
    ) -> (
        _AcceptedSingletonTransition
        | _AcceptedGreedyBatchTransition
        | _AcceptedCombinatorialBatchTransition
    ):
        if isinstance(decision, _GreedyBatchAdmissionDecision):
            return _transition_greedy_batch(state, decision, workspace)
        if isinstance(decision, _CombinatorialBatchAdmissionDecision):
            return _transition_combinatorial_batch(
                state,
                decision,
                workspace,
            )
        return _transition_singleton(state, decision, workspace)

    try:
        while winner is None:
            children: list[_ForkLocalBeamBranch] = []
            child_audit_rows: list[dict[str, Any]] = []
            parent_rows: list[dict[str, Any]] = []
            for parent_index, parent in enumerate(
                sorted(frontier, key=_beam_branch_sort_key)
            ):
                if (
                    len(children)
                    >= beam.maximum_admission_children_per_round
                ):
                    break
                excluded_pool_indices: set[int] = set()
                parent_child_count = 0
                parent_id = (
                    (
                        *resume_branch_ids,
                        *parent.branch_ids,
                    )[-1]
                    if resume_branch_ids or parent.branch_ids
                    else None
                )
                for child_ordinal in range(
                    beam.admission_children_per_parent
                ):
                    if (
                        len(children)
                        >= beam.maximum_admission_children_per_round
                    ):
                        break
                    branch_counter += 1
                    branch_id = (
                        f"canonical_beam:r{int(parent.state.controller_round) + 1}:"
                        f"p{parent_index}:c{child_ordinal}:n{branch_counter}"
                    )
                    before_s_alg = int(runtime.beam_executed_s_alg())
                    child_runtime, child_input_state = (
                        parent.runtime.fork_beam_branch(
                            parent.state,
                            branch_id=branch_id,
                            parent_branch_id=parent_id,
                            excluded_pool_indices=tuple(
                                sorted(excluded_pool_indices)
                            ),
                        )
                    )
                    owned_children.append(child_runtime)
                    try:
                        prepared = child_runtime.prepare_selection(
                            child_input_state
                        )
                        if not _selection_state_matches_accepted(
                            prepared.controller_state,
                            child_input_state,
                        ):
                            raise RuntimeError(
                                "Beam child selection identifies a different "
                                "accepted fork state."
                            )
                        decision = _select(prepared)
                        selected = (
                            decision.selected
                            if isinstance(
                                decision,
                                (
                                    _GreedyBatchAdmissionDecision,
                                    _CombinatorialBatchAdmissionDecision,
                                ),
                            )
                            else (decision.selected,)
                        )
                        selected_pool_indices = tuple(
                            int(value.pool_index) for value in selected
                        )
                        if excluded_pool_indices.intersection(
                            selected_pool_indices
                        ):
                            raise RuntimeError(
                                "Beam siblings selected an excluded admission."
                            )
                        transition_workspace = (
                            child_runtime.prepare_transition(
                                child_input_state,
                                decision,
                            )
                        )
                        transition = _transition(
                            child_input_state,
                            decision,
                            transition_workspace,
                        )
                        projection = child_runtime.project_accepted_event(
                            transition.checkpoint_event,
                            transition,
                        )
                        _assert_projected_event(
                            projection,
                            transition.checkpoint_event,
                        )
                    except Exception:
                        child_runtime.clear_beam_branch_context()
                        _close_owned_child(child_runtime)
                        raise
                    child_runtime.clear_beam_branch_context()
                    excluded_pool_indices.update(selected_pool_indices)
                    after_s_alg = int(runtime.beam_executed_s_alg())
                    fork_local_delta = int(after_s_alg - before_s_alg)
                    if fork_local_delta <= 0:
                        raise RuntimeError(
                            "A measured beam child performed no estimator work."
                        )
                    lineage_s_alg = int(
                        parent.lineage_s_alg + fork_local_delta
                    )
                    next_state = transition.next_state
                    comparison_score = float(
                        next_state.accepted_energy
                        + float(beam.s_alg_weight) * lineage_s_alg
                    )
                    stop = _configured_stop_receipt(
                        stop_policy,
                        next_state,
                        accepted_states=(
                            *parent.accepted_states,
                            next_state,
                        ),
                    )
                    child = _ForkLocalBeamBranch(
                        runtime=child_runtime,
                        state=next_state,
                        accepted_states=(
                            *parent.accepted_states,
                            next_state,
                        ),
                        transitions=(
                            *parent.transitions,
                            transition,
                        ),
                        events=(
                            *parent.events,
                            transition.checkpoint_event,
                        ),
                        projected_rounds=(
                            *parent.projected_rounds,
                            projection,
                        ),
                        branch_ids=(
                            *parent.branch_ids,
                            branch_id,
                        ),
                        lineage_s_alg=lineage_s_alg,
                        comparison_score=comparison_score,
                        stop=stop,
                    )
                    children.append(child)
                    parent_child_count += 1
                    child_audit_rows.append(
                        {
                            "branch_id": branch_id,
                            "parent_branch_id": parent_id,
                            "accepted_energy": float(
                                next_state.accepted_energy
                            ),
                            "fork_local_s_alg_delta": fork_local_delta,
                            "lineage_s_alg": lineage_s_alg,
                            "comparison_score": comparison_score,
                            "selected_pool_indices": list(
                                selected_pool_indices
                            ),
                            "stop_reasons": list(stop.fired_reasons),
                        }
                    )
                parent_rows.append(
                    {
                        "parent_branch_id": parent_id,
                        "children_executed": parent_child_count,
                        "unchanged_parent_retained": False,
                    }
                )
                if parent.runtime is not runtime:
                    _close_owned_child(parent.runtime)

            if not children:
                raise RuntimeError(
                    "Fork-local beam produced no accepted child."
                )
            ranked_children = sorted(children, key=_beam_branch_sort_key)
            exact_hits = [
                child
                for child in ranked_children
                if child.stop is not None
                and "exact_ed_target_reached"
                in child.stop.fired_reasons
            ]
            maximum_hits = [
                child
                for child in ranked_children
                if child.stop is not None
                and "maximum_controller_rounds"
                in child.stop.fired_reasons
            ]
            if exact_hits:
                winner = exact_hits[0]
                survivor_ids = {id(winner.runtime)}
                terminal_reason = "exact_ed_target_reached"
            elif len(maximum_hits) == len(ranked_children):
                winner = ranked_children[0]
                survivor_ids = {id(winner.runtime)}
                terminal_reason = "maximum_controller_rounds"
            else:
                frontier = ranked_children[
                    : beam.live_parent_branches
                ]
                survivor_ids = {
                    id(branch.runtime) for branch in frontier
                }
                terminal_reason = None
            for child in children:
                if id(child.runtime) not in survivor_ids:
                    _close_owned_child(child.runtime)
            round_audits.append(
                {
                    "controller_round": int(
                        ranked_children[0].state.controller_round
                    ),
                    "parent_rows": parent_rows,
                    "children": child_audit_rows,
                    "children_executed": len(children),
                    "survivor_branch_ids": [
                        branch.branch_ids[-1]
                        for branch in (
                            (winner,) if winner is not None else frontier
                        )
                    ],
                    "terminal_reason": terminal_reason,
                }
            )
            all_work_components = (
                runtime.beam_executed_s_alg_components()
            )
            accepted_prefix_all_work.append(
                _AcceptedPrefixAllWork(
                    components=all_work_components,
                    s_alg=int(runtime.beam_executed_s_alg()),
                )
            )

        if winner.stop is None or not winner.stop.fired_reasons:
            raise RuntimeError("Beam winner lacks a configured stop receipt.")
        diagnostics = {
            "schema": "paper_i_canonical_fork_local_beam_search_v1",
            "comparison": "accepted_energy_plus_weight_times_lineage_s_alg",
            "s_alg_scope": "fork_local_cumulative_lineage",
            "s_alg_weight": float(beam.s_alg_weight),
            "calibration_status": str(beam.calibration_status),
            "live_parent_branches": int(beam.live_parent_branches),
            "admission_children_per_parent": int(
                beam.admission_children_per_parent
            ),
            "maximum_admission_children_per_round": int(
                beam.maximum_admission_children_per_round
            ),
            "unchanged_parent_survival": False,
            "phase_live_hysteresis": False,
            "initial_unbranched_s_alg": initial_s_alg,
            "resume_winning_branch_ids": list(resume_branch_ids),
            "resume_winning_lineage_s_alg": int(
                resume_lineage_s_alg
            ),
            "all_executed_s_alg": int(runtime.beam_executed_s_alg()),
            "winning_branch_ids": [
                *resume_branch_ids,
                *winner.branch_ids,
            ],
            "winning_lineage_s_alg": int(winner.lineage_s_alg),
            "winning_comparison_score": float(
                winner.comparison_score
            ),
            "rounds": round_audits,
        }
        winner.runtime.configure_beam_winner(
            winning_branch_ids=(
                *resume_branch_ids,
                *winner.branch_ids,
            ),
            diagnostics=diagnostics,
            observation_owner=runtime,
        )
        finalization = winner.runtime.finalize(
            final_state=winner.state,
            transitions=winner.transitions,
            events=winner.events,
            projected_rounds=winner.projected_rounds,
            stop=winner.stop,
        )
        _close_owned_child(winner.runtime)
        return _ControllerOutcome(
            initial_state=initial_state,
            final_state=winner.state,
            accepted_states=winner.accepted_states,
            transitions=winner.transitions,
            events=winner.events,
            projected_rounds=winner.projected_rounds,
            accepted_prefix_all_work=tuple(accepted_prefix_all_work),
            stop=winner.stop,
            finalization=finalization,
        )
    finally:
        for child_runtime in tuple(owned_children):
            try:
                _close_owned_child(child_runtime)
            except Exception:
                # Preserve the controller's scientific/implementation failure;
                # cleanup is best-effort once a fork close itself fails.
                pass
        runtime.clear_beam_branch_context()
        runtime.close()
