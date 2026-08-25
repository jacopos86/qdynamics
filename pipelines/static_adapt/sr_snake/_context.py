"""Private one-pass dependency resolution for the SR-SNAKE run seam."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

from pipelines.contracts.problem import ResolvedProblemContext
from pipelines.static_adapt.extensions import beam_extension_from_policy
from pipelines.static_adapt.sr_snake._controller import (
    _DefaultControllerNumericalRuntime,
)
from pipelines.static_adapt.sr_snake.contracts import (
    AcceptedStateResume,
    AlwaysCommutationReducedInsertion,
    AppendCommutationReducedInsertion,
    AppendOnlyInsertion,
    BeamOff,
    CombinatorialBatchAdmission,
    ForkLocalBeam,
    FreshStart,
    GreedyBatchAdmission,
    MetricPruning,
    PlateauCommutationInsertion,
    PruningOff,
    RecoverabilityPruning,
    ResolvedProblemReceipt,
    SRObservationPolicy,
    SRRunRequest,
    SRStopPolicy,
    SingletonAdmission,
    TrustRegionPruning,
)
from pipelines.static_adapt.sr_snake_route_profile import (
    SR_ROUTE_PROFILE_COMBINATORIAL_BATCH_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1
    as _ACTIVE_COMBINATORIAL_PROFILE,
    SR_ROUTE_PROFILE_GREEDY_BATCH_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1
    as _ACTIVE_GREEDY_PROFILE,
    SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V2
    as _ACTIVE_INSERTION_PROFILE,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_V1
    as _ACTIVE_ALWAYS_INSERTION_PROFILE,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1
    as _ACTIVE_PROFILE,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_ALIAS_V1
    as _ACTIVE_PROFILE_REQUEST,
    SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_FS_PRUNE_VERIFY_ALIAS_V1
    as _ACTIVE_PRUNE_PROFILE_REQUEST,
    SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_FS_PRUNE_VERIFY_V1
    as _ACTIVE_PRUNE_PROFILE,
    canonical_sr_snake_combinatorial_batch_projected_phase3_no_overlap_trust_v1_contract,
    canonical_sr_snake_combinatorial_batch_projected_phase3_no_overlap_trust_v1_contract_sha256,
    canonical_sr_snake_greedy_batch_projected_phase3_no_overlap_trust_v1_contract,
    canonical_sr_snake_greedy_batch_projected_phase3_no_overlap_trust_v1_contract_sha256,
    canonical_sr_snake_insertion_commutation_plateau_v2_contract,
    canonical_sr_snake_insertion_commutation_plateau_v2_contract_sha256,
    canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_commutation_reduced_insertion_v1_contract,
    canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_commutation_reduced_insertion_v1_contract_sha256,
    canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract,
    canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract_sha256,
    canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_full_geometry_fs_prune_verify_v1_contract,
    canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_full_geometry_fs_prune_verify_v1_contract_sha256,
)


class _DefaultControllerRuntimeFactory(Protocol):
    """Construct the exact-gated numerical runtime for the public facade."""

    def __call__(
        self,
        *,
        admission_policy: (
            SingletonAdmission
            | GreedyBatchAdmission
            | CombinatorialBatchAdmission
        ),
        stop_policy: SRStopPolicy,
        executor_kwargs: Mapping[str, Any],
        resume_hydration: Any | None = None,
        candidate_adapter: Any | None = None,
    ) -> _DefaultControllerNumericalRuntime: ...


@dataclass(frozen=True)
class _FrozenList:
    items: tuple[Any, ...]


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, list):
        return _FrozenList(tuple(_freeze(item) for item in value))
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _thaw(item)
            for key, item in value.items()
        }
    if isinstance(value, _FrozenList):
        return [_thaw(item) for item in value.items]
    if isinstance(value, tuple):
        return tuple(_thaw(item) for item in value)
    if isinstance(value, frozenset):
        return {_thaw(item) for item in value}
    return value


@dataclass(frozen=True)
class _ResolvedRouteDependency:
    family: str
    profile_request: str
    profile: str
    contract_sha256: str
    pool_key: str
    contract: Mapping[str, Any] = field(repr=False)


@dataclass(frozen=True)
class _ResolvedNumericalDependencies:
    hamiltonian: Any = field(repr=False, compare=False)
    default_runtime_factory: _DefaultControllerRuntimeFactory = field(
        repr=False,
        compare=False,
    )
    legacy_executor: Callable[..., tuple[dict[str, Any], Any]] = field(
        repr=False,
        compare=False,
    )
    state_backend: str
    finite_angle: float
    response_coordinate_scope: str
    trust_policy: str
    candidate_adapter: Any | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    resume_hydration: Any | None = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class _ResolvedOptimizerPolicy:
    name: str
    maximum_iterations: int
    maximum_function_evaluations: int
    seed: int


@dataclass(frozen=True)
class _ResolvedAcceptedRefitPolicy:
    policy: str
    scope: str
    coordinate_chart: str
    base_chart_policy: str


@dataclass(frozen=True)
class _ResolvedEstimatorLedgerDependency:
    enabled: bool
    destination: Path | None


@dataclass(frozen=True)
class _ResolvedInitialAcceptedState:
    source_label: str
    state_kind: str
    build_state: Callable[[], Any] = field(repr=False, compare=False)


@dataclass(frozen=True)
class _ResolvedExecutionContext:
    """Immutable active dependencies plus direct controller runtime inputs."""

    problem: ResolvedProblemContext = field(repr=False, compare=False)
    problem_receipt: ResolvedProblemReceipt
    request: SRRunRequest
    route: _ResolvedRouteDependency
    numerical: _ResolvedNumericalDependencies
    optimizer: _ResolvedOptimizerPolicy
    accepted_refit: _ResolvedAcceptedRefitPolicy
    estimator_ledger: _ResolvedEstimatorLedgerDependency
    observation: SRObservationPolicy
    stop: SRStopPolicy
    initial_state: _ResolvedInitialAcceptedState
    exact_same_cutoff_energy: float
    runtime_kwargs: Mapping[str, Any] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        pool_key = str(self.route.pool_key)
        if pool_key not in self.problem.admissible_pool_keys:
            raise ValueError(
                f"Resolved SR-SNAKE pool {pool_key!r} is not admissible for "
                f"problem family {self.problem.family_key!r}."
            )
        if self.request.execution.stop is not self.stop:
            raise ValueError(
                "Resolved SR-SNAKE stop policy is not the request stop policy."
            )
        if self.request.observation is not self.observation:
            raise ValueError(
                "Resolved SR-SNAKE observation policy is not the request "
                "observation policy."
            )
        if self.runtime_kwargs.get("h_poly") is not self.numerical.hamiltonian:
            raise ValueError(
                "Resolved SR-SNAKE numerical Hamiltonian disagrees with the "
                "direct runtime input."
            )
        if str(self.runtime_kwargs.get("adapt_pool")) != pool_key:
            raise ValueError(
                "Resolved SR-SNAKE pool disagrees with the direct runtime input."
            )
        if int(self.runtime_kwargs.get("max_depth", -1)) != int(
            self.stop.maximum_controller_rounds
        ):
            raise ValueError(
                "Resolved SR-SNAKE stop policy disagrees with the direct "
                "runtime input."
            )

    def canonical_runtime_kwargs(self) -> dict[str, Any]:
        """Return one mutable direct-runtime copy for the public facade."""

        resolved = _thaw(self.runtime_kwargs)
        if not isinstance(resolved, dict):
            raise AssertionError("Canonical runtime inputs must thaw to a dictionary.")
        return resolved

    def legacy_executor_kwargs(self) -> dict[str, Any]:
        """Project direct state onto the explicit compatibility boundary."""

        from pipelines.static_adapt import adapt_pipeline

        return adapt_pipeline._canonical_sr_snake_legacy_executor_kwargs(
            self.canonical_runtime_kwargs()
        )

    def build_default_controller_runtime(
        self,
    ) -> _DefaultControllerNumericalRuntime:
        """Construct the exact-gated direct session for the public facade."""

        return self.numerical.default_runtime_factory(
            admission_policy=self.request.method.admission,
            stop_policy=self.stop,
            executor_kwargs=self.canonical_runtime_kwargs(),
            resume_hydration=self.numerical.resume_hydration,
            candidate_adapter=self.numerical.candidate_adapter,
        )


def _validate_supported_request(request: SRRunRequest) -> None:
    admission = request.method.admission
    if not isinstance(
        admission,
        (
            SingletonAdmission,
            GreedyBatchAdmission,
            CombinatorialBatchAdmission,
        ),
    ):
        raise TypeError("Unsupported SR-SNAKE admission policy.")

    if not isinstance(
        request.method.pruning,
        (
            PruningOff,
            MetricPruning,
            TrustRegionPruning,
            RecoverabilityPruning,
        ),
    ):
        raise TypeError("Unsupported SR-SNAKE pruning policy.")
    if not isinstance(
        request.method.insertion,
        (
            PlateauCommutationInsertion,
            AlwaysCommutationReducedInsertion,
            AppendCommutationReducedInsertion,
            AppendOnlyInsertion,
        ),
    ):
        raise TypeError("Unsupported SR-SNAKE insertion policy.")
    if not isinstance(request.method.beam, (BeamOff, ForkLocalBeam)):
        raise TypeError("Unsupported SR-SNAKE beam policy.")

    if not isinstance(
        request.execution.resume,
        (FreshStart, AcceptedStateResume),
    ):
        raise TypeError("Unsupported SR-SNAKE resume policy.")


def _validate_exact_source(
    problem: ResolvedProblemReceipt,
    request: SRRunRequest,
) -> None:
    exact = request.execution.stop.exact_ed_target
    if exact is None:
        return
    source = exact.source
    mismatches: list[str] = []
    if source.problem_request_sha256 != problem.problem_request_sha256:
        mismatches.append("problem_request_sha256")
    if source.sector_label != problem.sector_label:
        mismatches.append("sector_label")
    if source.comparison_space_label != problem.comparison_space_label:
        mismatches.append("comparison_space_label")
    if source.n_ph_max != problem.n_ph_max:
        mismatches.append("n_ph_max")
    if mismatches:
        raise ValueError(
            "ExactEDStop source does not describe the resolved physical "
            "problem, sector, and cutoff: "
            + ", ".join(mismatches)
        )


def _route_contract_sha256(contract: Mapping[str, Any]) -> str:
    """Return the deterministic digest used by canonical composed contracts."""

    payload = json.dumps(
        contract,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_route_contract_for_request(
    request: SRRunRequest,
) -> tuple[str, str, dict[str, Any], str]:
    """Resolve one policy composition without probing compatibility routes.

    Exact historical contracts retain their frozen identities.  Every other
    typed composition is a deterministic child of the accepted fd5ec parent;
    no alias enumeration or compatibility fallback participates in this
    decision.
    """

    admission = request.method.admission
    insertion = request.method.insertion
    pruning = request.method.pruning
    beam = request.method.beam

    if (
        isinstance(admission, SingletonAdmission)
        and isinstance(insertion, AlwaysCommutationReducedInsertion)
        and isinstance(pruning, PruningOff)
        and isinstance(beam, BeamOff)
    ):
        contract = canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_commutation_reduced_insertion_v1_contract()
        return (
            _ACTIVE_ALWAYS_INSERTION_PROFILE,
            _ACTIVE_ALWAYS_INSERTION_PROFILE,
            contract,
            canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_commutation_reduced_insertion_v1_contract_sha256(),
        )
    if (
        isinstance(admission, SingletonAdmission)
        and isinstance(insertion, PlateauCommutationInsertion)
        and isinstance(pruning, PruningOff)
        and isinstance(beam, BeamOff)
    ):
        contract = canonical_sr_snake_insertion_commutation_plateau_v2_contract()
        return (
            _ACTIVE_INSERTION_PROFILE,
            _ACTIVE_INSERTION_PROFILE,
            contract,
            canonical_sr_snake_insertion_commutation_plateau_v2_contract_sha256(),
        )
    if (
        isinstance(admission, SingletonAdmission)
        and isinstance(insertion, AppendOnlyInsertion)
        and isinstance(pruning, PruningOff)
        and isinstance(beam, BeamOff)
    ):
        contract = (
            canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract()
        )
        return (
            _ACTIVE_PROFILE_REQUEST,
            _ACTIVE_PROFILE,
            contract,
            canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract_sha256(),
        )
    if (
        isinstance(admission, GreedyBatchAdmission)
        and isinstance(insertion, AppendOnlyInsertion)
        and isinstance(pruning, PruningOff)
        and isinstance(beam, BeamOff)
    ):
        contract = (
            canonical_sr_snake_greedy_batch_projected_phase3_no_overlap_trust_v1_contract(
                maximum_size=admission.maximum_size,
                search_window_size=admission.search_window_size,
            )
        )
        return (
            _ACTIVE_GREEDY_PROFILE,
            _ACTIVE_GREEDY_PROFILE,
            contract,
            canonical_sr_snake_greedy_batch_projected_phase3_no_overlap_trust_v1_contract_sha256(
                maximum_size=admission.maximum_size,
                search_window_size=admission.search_window_size,
            ),
        )
    if (
        isinstance(admission, CombinatorialBatchAdmission)
        and isinstance(insertion, AppendOnlyInsertion)
        and isinstance(pruning, PruningOff)
        and isinstance(beam, BeamOff)
    ):
        contract = (
            canonical_sr_snake_combinatorial_batch_projected_phase3_no_overlap_trust_v1_contract(
                maximum_size=admission.maximum_size,
                search_window_size=admission.resolved_search_window_size,
            )
        )
        return (
            _ACTIVE_COMBINATORIAL_PROFILE,
            _ACTIVE_COMBINATORIAL_PROFILE,
            contract,
            canonical_sr_snake_combinatorial_batch_projected_phase3_no_overlap_trust_v1_contract_sha256(
                maximum_size=admission.maximum_size,
                search_window_size=admission.resolved_search_window_size,
            ),
        )
    if (
        isinstance(admission, SingletonAdmission)
        and isinstance(insertion, AppendOnlyInsertion)
        and isinstance(pruning, RecoverabilityPruning)
        and isinstance(beam, BeamOff)
    ):
        contract = (
            canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_full_geometry_fs_prune_verify_v1_contract()
        )
        return (
            _ACTIVE_PRUNE_PROFILE_REQUEST,
            _ACTIVE_PRUNE_PROFILE,
            contract,
            canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_full_geometry_fs_prune_verify_v1_contract_sha256(),
        )

    if isinstance(admission, GreedyBatchAdmission):
        contract = (
            canonical_sr_snake_greedy_batch_projected_phase3_no_overlap_trust_v1_contract(
                maximum_size=admission.maximum_size,
                search_window_size=admission.search_window_size,
            )
        )
        admission_tag = "greedy_batch"
    elif isinstance(admission, CombinatorialBatchAdmission):
        contract = (
            canonical_sr_snake_combinatorial_batch_projected_phase3_no_overlap_trust_v1_contract(
                maximum_size=admission.maximum_size,
                search_window_size=admission.resolved_search_window_size,
            )
        )
        admission_tag = "combinatorial_batch"
    else:
        contract = (
            canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract()
        )
        admission_tag = "singleton"

    execution_settings = dict(contract["execution_settings"])
    semantic_invariants = dict(contract["semantic_invariants"])

    if isinstance(insertion, AlwaysCommutationReducedInsertion):
        insertion_contract = canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_commutation_reduced_insertion_v1_contract()
        execution_settings["adapt_insertion_mode"] = (
            insertion_contract["execution_settings"]["adapt_insertion_mode"]
        )
        for key in (
            "diagnostic_position_ablation",
            "insertion_position_scope",
            "insertion_equivalence_policy",
        ):
            semantic_invariants[key] = insertion_contract[
                "semantic_invariants"
            ][key]
        insertion_tag = "always_commutation_reduced"
    elif isinstance(insertion, PlateauCommutationInsertion):
        insertion_contract = (
            canonical_sr_snake_insertion_commutation_plateau_v2_contract()
        )
        execution_settings["adapt_insertion_mode"] = (
            insertion_contract["execution_settings"]["adapt_insertion_mode"]
        )
        for key in (
            "experimental_insertion_policy",
            "insertion_position_scope",
            "insertion_equivalence_policy",
            "plateau_trigger_source",
            "plateau_prior_mean_decrease_ratio_threshold",
            "plateau_threshold_comparison",
            "plateau_threshold_calibration_status",
            "plateau_patience",
            "plateau_hysteresis_active",
            "online_exact_reference_used",
        ):
            semantic_invariants[key] = insertion_contract[
                "semantic_invariants"
            ][key]
        insertion_tag = "plateau_commutation"
    elif isinstance(insertion, AppendCommutationReducedInsertion):
        execution_settings["adapt_insertion_mode"] = (
            insertion.runtime_mode
        )
        semantic_invariants.update(
            {
                "diagnostic_position_ablation": True,
                "experimental_insertion_policy": insertion.kind,
                "insertion_position_scope": insertion.position_scope,
                "insertion_equivalence_policy": (
                    insertion.equivalence_policy
                ),
            }
        )
        insertion_tag = insertion.kind
    elif isinstance(insertion, AppendOnlyInsertion):
        execution_settings["adapt_insertion_mode"] = "append_only"
        insertion_tag = "append_only"
    else:
        raise TypeError("Unsupported SR-SNAKE insertion policy.")

    if isinstance(pruning, PruningOff):
        pruning_tag = "off"
    else:
        prune_contract = (
            canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_full_geometry_fs_prune_verify_v1_contract()
        )
        execution_settings.update(
            {
                key: value
                for key, value in prune_contract["execution_settings"].items()
                if str(key).startswith("phase1_prune_")
            }
        )
        for key, value in prune_contract["semantic_invariants"].items():
            if str(key).startswith("prune_"):
                semantic_invariants[key] = value
        semantic_invariants["pruning_active"] = True
        if isinstance(pruning, MetricPruning):
            pruning_tag = "metric"
            nomination_route = (
                "regularized_local_metric_response_delete_refit_v1"
            )
            execution_settings["phase1_prune_schur_nomination_route"] = (
                "metric_regularized_v1"
            )
            execution_settings["phase1_prune_metric_schur_mu"] = 1.0e-6
            execution_settings["phase1_prune_metric_schur_solve_mode"] = (
                "gradient_corrected_v1"
            )
        elif isinstance(pruning, TrustRegionPruning):
            pruning_tag = "trust_region"
            nomination_route = "full_logical_fs_trust_delete_refit_v1"
        else:
            pruning_tag = "recoverability"
            nomination_route = "full_logical_fs_trust_delete_refit_v1"
        semantic_invariants["prune_nomination_authority"] = nomination_route
        semantic_invariants["prune_acceptance_authority"] = (
            "measured_delete_and_complete_refit_v1"
        )

    if isinstance(beam, ForkLocalBeam):
        beam_tag = "fork_local"
        execution_settings.update(
            {
                "adapt_beam_live_branches": beam.live_parent_branches,
                "adapt_beam_children_per_parent": (
                    beam.admission_children_per_parent
                ),
                "adapt_beam_lambda": beam.s_alg_weight,
                "adapt_beam_terminal_archive_mode": "disabled",
                "adapt_beam_terminated_keep": 0,
            }
        )
        semantic_invariants.update(
            {
                "beam_shape": (
                    "three_live_two_children_per_parent_v1"
                    if (
                        beam.live_parent_branches == 3
                        and beam.admission_children_per_parent == 2
                    )
                    else (
                        f"typed_{beam.live_parent_branches}x"
                        f"{beam.admission_children_per_parent}_v1"
                    )
                ),
                "beam_comparison": (
                    "accepted_energy_plus_weight_times_fork_local_s_alg_v1"
                ),
                "beam_global_accounting": (
                    "all_executed_branch_occurrences_in_global_s_alg_v1"
                ),
                "beam_live_parent_branches": beam.live_parent_branches,
                "beam_admission_children_per_parent": (
                    beam.admission_children_per_parent
                ),
                "beam_maximum_admission_children_per_round": (
                    beam.maximum_admission_children_per_round
                ),
                "beam_s_alg_weight": beam.s_alg_weight,
                "beam_calibration_status": beam.calibration_status,
                "beam_unchanged_parent_survival": False,
                "beam_phase_live_hysteresis": False,
            }
        )
    else:
        beam_tag = "off"

    profile = (
        "paper_i_canonical_sr_snake"
        f"__admission-{admission_tag}"
        f"__insertion-{insertion_tag}"
        f"__pruning-{pruning_tag}"
        f"__beam-{beam_tag}_v1"
    )
    semantic_invariants.update(
        {
            "canonical_interface": "run_sr_snake_problem_request_v1",
            "canonical_composition_schema": (
                "paper_i_canonical_policy_composition_v1"
            ),
            "canonical_admission_policy": admission.kind,
            "canonical_insertion_policy": insertion.kind,
            "canonical_pruning_policy": pruning.kind,
            "canonical_beam_policy": beam.kind,
            "compatibility_resolution_active": False,
        }
    )
    contract.update(
        {
            "route_family": (
                "singleton_response_snake"
                if isinstance(admission, SingletonAdmission)
                else str(contract["route_family"])
            ),
            "route_profile": profile,
            "execution_settings": execution_settings,
            "semantic_invariants": semantic_invariants,
            "lineage_authority": {
                "parent_route_profile": _ACTIVE_PROFILE,
                "parent_contract_sha256": (
                    canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract_sha256()
                ),
                "typed_policy_composition": request.method.to_dict(),
                "scientific_result_anchor_claimed": False,
            },
        }
    )
    normalized = json.loads(json.dumps(contract, sort_keys=True))
    return (
        profile,
        profile,
        normalized,
        _route_contract_sha256(normalized),
    )


def _resolve_execution_context(
    problem: ResolvedProblemContext,
    request: SRRunRequest,
    *,
    route_override: tuple[str, str, Mapping[str, Any], str] | None = None,
    candidate_adapter: Any | None = None,
) -> _ResolvedExecutionContext:
    """Resolve the active typed request into one immutable executor context."""

    if not isinstance(problem, ResolvedProblemContext):
        raise TypeError("problem must be a ResolvedProblemContext.")
    if not isinstance(request, SRRunRequest):
        raise TypeError("request must be an SRRunRequest.")
    family_key = str(problem.family_key).strip().lower()
    ordinary_paper_i_problem = bool(
        family_key == "hh" and int(problem.request.num_sites) == 2
    )
    application_family_key = str(
        getattr(candidate_adapter, "application_family_key", "")
    ).strip().lower()
    lane_owned_application = bool(
        route_override is not None
        and candidate_adapter is not None
        and application_family_key == family_key
    )
    if not ordinary_paper_i_problem and not lane_owned_application:
        raise ValueError(
            "The ordinary Paper-I SR-SNAKE facade is locked to the canonical "
            "Hubbard--Holstein L=2 problem. Other families and sizes require "
            "an explicitly named compatibility or lane-owned route."
        )
    _validate_supported_request(request)

    problem_receipt = ResolvedProblemReceipt.from_problem(problem)
    _validate_exact_source(problem_receipt, request)

    from pipelines.static_adapt import adapt_pipeline

    exact = request.execution.stop.exact_ed_target
    exact_energy = (
        float(exact.energy)
        if exact is not None
        else float(
            problem.exact_target.resolve_energy(ai_log=adapt_pipeline._ai_log)
        )
    )

    if route_override is None:
        (
            route_profile_request,
            route_profile,
            route_contract,
            route_contract_sha256,
        ) = _canonical_route_contract_for_request(request)
    else:
        if (
            not isinstance(route_override, tuple)
            or len(route_override) != 4
            or not isinstance(route_override[2], Mapping)
        ):
            raise TypeError(
                "route_override must be a four-field typed route identity."
            )
        (
            route_profile_request,
            route_profile,
            route_contract_raw,
            route_contract_sha256,
        ) = route_override
        route_contract = json.loads(
            json.dumps(
                dict(route_contract_raw),
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        if (
            str(route_profile_request) != str(route_profile)
            or str(route_contract.get("route_profile", ""))
            != str(route_profile)
            or _route_contract_sha256(route_contract)
            != str(route_contract_sha256)
        ):
            raise ValueError(
                "The internal RA-ADAPT route override failed identity "
                "authentication."
            )
    if candidate_adapter is not None:
        from pipelines.static_adapt.ra_adapt.adapters import (
            CandidateRepresentationAdapter,
        )

        if not isinstance(candidate_adapter, CandidateRepresentationAdapter):
            raise TypeError(
                "candidate_adapter must implement the canonical RA-ADAPT "
                "representation adapter protocol."
            )
        invariants = route_contract.get("semantic_invariants", {})
        route_representation = (
            invariants.get("candidate_representation")
            if isinstance(invariants, Mapping)
            else None
        )
        if route_representation != str(
            candidate_adapter.candidate_representation_id
        ):
            raise ValueError(
                "The RA-ADAPT route and candidate adapter representation "
                "identities disagree."
            )
    checkpoint = request.observation.checkpoint
    kwargs = adapt_pipeline._build_canonical_sr_snake_runtime_kwargs(
        resolved_problem_context=problem,
        maximum_controller_rounds=(
            request.execution.stop.maximum_controller_rounds
        ),
        exact_energy=exact_energy,
        exact_target_absolute_tolerance=(
            None if exact is None else float(exact.absolute_tolerance)
        ),
        exact_target_energy=(
            None if exact is None else float(exact.energy)
        ),
        checkpoint_path=None if checkpoint is None else checkpoint.path,
        checkpoint_every_controller_rounds=(
            None
            if checkpoint is None
            else checkpoint.every_controller_rounds
        ),
        checkpoint_keep_history_tail=(
            None if checkpoint is None else checkpoint.keep_history_tail
        ),
        # The direct runtime keeps the resolved profile at this internal
        # executor boundary, matching the characterized exact route.
        route_profile_request=route_profile,
        route_profile=route_profile,
        route_contract=route_contract,
        route_contract_sha256=route_contract_sha256,
        beam_extension=beam_extension_from_policy(request.method.beam),
        gradient_tolerance=request.execution.stop.gradient_tolerance,
    )

    contract_raw = kwargs.get("sr_route_profile_contract")
    if not isinstance(contract_raw, Mapping):
        raise ValueError(
            "The active SR-SNAKE route did not resolve a route contract."
        )
    contract = _freeze(contract_raw)
    if not isinstance(contract, Mapping):
        raise AssertionError("Frozen route contract must remain a mapping.")
    pool_key = str(kwargs.get("adapt_pool", ""))
    route = _ResolvedRouteDependency(
        family=str(contract.get("route_family", "")),
        profile_request=route_profile_request,
        profile=str(kwargs.get("sr_route_profile_resolved", "")),
        contract_sha256=str(
            kwargs.get("sr_route_profile_contract_sha256", "")
        ),
        pool_key=pool_key,
        contract=contract,
    )
    resume_hydration: Any | None = None
    if isinstance(request.execution.resume, AcceptedStateResume):
        from pipelines.static_adapt.sr_snake._resume import (
            load_canonical_accepted_state_resume,
        )

        resume_hydration = load_canonical_accepted_state_resume(
            request.execution.resume,
            expected_problem=problem,
            expected_route_profile=route.profile,
            expected_route_contract_sha256=route.contract_sha256,
        )
        if int(resume_hydration.controller_round) >= int(
            request.execution.stop.maximum_controller_rounds
        ):
            raise ValueError(
                "Accepted-state resume requires maximum_controller_rounds "
                "to exceed the authenticated checkpoint round."
            )
        exact_target = request.execution.stop.exact_ed_target
        if (
            exact_target is not None
            and abs(
                float(resume_hydration.accepted_energy)
                - float(exact_target.energy)
            )
            <= float(exact_target.absolute_tolerance)
        ):
            raise ValueError(
                "Accepted-state resume checkpoint already satisfies the "
                "configured exact-ED stop target."
            )
    frozen_kwargs = _freeze(kwargs)
    if not isinstance(frozen_kwargs, Mapping):
        raise AssertionError("Frozen legacy executor inputs must be a mapping.")

    def _retained_optional_policy_executor(
        **executor_kwargs: Any,
    ) -> tuple[dict[str, Any], Any]:
        return adapt_pipeline._run_hardcoded_adapt_vqe(**executor_kwargs)

    return _ResolvedExecutionContext(
        problem=problem,
        problem_receipt=problem_receipt,
        request=request,
        route=route,
        numerical=_ResolvedNumericalDependencies(
            hamiltonian=problem.hamiltonian,
            default_runtime_factory=(
                adapt_pipeline
                ._build_default_sr_controller_numerical_runtime
            ),
            legacy_executor=_retained_optional_policy_executor,
            state_backend=str(kwargs["adapt_state_backend"]),
            finite_angle=float(kwargs["finite_angle"]),
            response_coordinate_scope=str(
                kwargs["phase3_response_coordinate_scope"]
            ),
            trust_policy=str(
                kwargs["historical_singleton_trust_region_update_policy"]
            ),
            candidate_adapter=candidate_adapter,
            resume_hydration=resume_hydration,
        ),
        optimizer=_ResolvedOptimizerPolicy(
            name=str(kwargs["adapt_inner_optimizer"]),
            maximum_iterations=int(kwargs["maxiter"]),
            maximum_function_evaluations=int(kwargs["adapt_scipy_maxfev"]),
            seed=int(kwargs["seed"]),
        ),
        accepted_refit=_ResolvedAcceptedRefitPolicy(
            policy=str(kwargs["adapt_reopt_policy"]),
            scope=str(kwargs["adapt_accepted_refit_scope"]),
            coordinate_chart=str(
                kwargs["adapt_accepted_refit_coordinate_chart"]
            ),
            base_chart_policy=str(
                kwargs["adapt_accepted_refit_base_chart_policy"]
            ),
        ),
        estimator_ledger=_ResolvedEstimatorLedgerDependency(
            enabled=bool(kwargs["adapt_estimator_call_ledger_enabled"]),
            destination=(
                None
                if request.observation.estimator_ledger is None
                else request.observation.estimator_ledger.path
            ),
        ),
        observation=request.observation,
        stop=request.execution.stop,
        initial_state=_ResolvedInitialAcceptedState(
            source_label=str(problem.reference_state.source_label),
            state_kind=str(problem.reference_state.state_kind),
            build_state=problem.reference_state.build_state,
        ),
        exact_same_cutoff_energy=float(exact_energy),
        runtime_kwargs=frozen_kwargs,
    )


__all__: list[str] = []
