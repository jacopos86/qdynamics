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

    def __iter__(self) -> Iterator[object]:
        if self.batch is not None:
            yield self.batch
        if self.pruning is not None:
            yield self.pruning


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
    }
