"""Shared Paper-I RA-ADAPT execution engine.

The numerical controller remains the characterized static-ADAPT controller.
This module owns representation resolution, repaired route authentication,
bundle-only study policies, and result provenance.
"""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import math
from typing import Any, Mapping

from pipelines.contracts.problem import ResolvedProblemContext
from pipelines.static_adapt.adaptive_phase_contracts import (
    ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1,
)
from pipelines.static_adapt.ra_adapt.adapters import (
    GLOBAL_SINGLE_PAULI_ADAPTER_ID,
    GLOBAL_SINGLETON_ABSOLUTE_GRADIENT_PHASE0_POLICY,
    GLOBAL_SINGLETON_GRADIENT_PHASE0_ADAPTER_ID,
    MACRO_GRADIENT_PHASE0_ADAPTER_ID,
    MACRO_PHASE0_STANDARD_ADAPT_ABS_GRADIENT_POLICY,
    PHASE_I_SUPPLY_GLOBAL_GUARDED_SINGLETON,
    PHASE_I_VISIBILITY_ALL_EXECUTABLE,
    PHASE_II_EXPOSURE_RETAINED_SINGLETON_IDENTITY,
    GlobalSinglePauliWordCandidateAdapter,
    GlobalSingletonGradientPhase0CandidateAdapter,
    H2OLinearFDSectorCompletePauliBlockCandidateAdapter,
    H2OLinearFDSymmetryCompleteCandidateAdapter,
    MacroCandidateAdapter,
    MacroGradientPhase0CandidateAdapter,
    MacroGradientPhase0ThenSingletonCandidateAdapter,
    MacroThenSingletonPhaseICandidateAdapter,
    SinglePauliWordCandidateAdapter,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    ACTIVE_GRADIENT_MEASURED,
    ACTIVE_GRADIENT_STATIONARY,
    APPEND_ADAPT_PROTOCOL_SCHEMA,
    BundleProtocolMaterializationAuthority,
    CANDIDATE_REPRESENTATION_MACRO,
    CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    CandidateInventoryLineageReceipt,
    CandidateLineageReceipt,
    ENDPOINT_OVERLAP_DISPLACEMENT_TRUST,
    EXACT_ORDERED_INSERTION_CHART,
    EXPANDED_RUNTIME_PROJECTED_LOGICAL_BASE_CHART,
    FULL_ENLARGED_ACCEPTED_REFIT,
    PhaseIIIStabilizationReceipt,
    PhaseIIIMultiplierContract,
    PolicyEchoReceipt,
    PROJECTED_GENERALIZED_SOLVER,
    RA_ADAPT_FULL_RESPONSE_REFIT_INITIALIZATION_POLICY,
    RA_ADAPT_INCREMENTAL_CANDIDATE_GAIN_POLICY,
    RA_ADAPT_NONSTATIONARY_INCREMENTAL_FULL_RESPONSE_ALGORITHM_ID,
    RA_ADAPT_POST_LATCH_INSERTION_TRIGGER_SCOPE,
    RA_ADAPT_PHASE3_POPULATION_ON_INSERTION_PLATEAU,
    RA_ADAPT_PHASE3_POPULATION_LATCHED_ON_PROGRESS_PLATEAU,
    RA_ADAPT_PHASE3_PREPLATEAU_WINNER_MATERIALIZATION,
    RA_ADAPT_PROTOCOL_SCHEMA,
    RA_ADAPT_PROTOCOL_SCHEMAS,
    RA_ADAPT_PROTOCOL_SCHEMA_V2,
    RA_ADAPT_RESULT_SCHEMA_V1,
    RA_ADAPT_RESULT_SCHEMA_V2,
    RA_ADAPT_ROUTE_CONTRACT_SCHEMA_V1,
    RA_ADAPT_ROUTE_CONTRACT_SCHEMA_V2,
    RA_ADAPT_SINGLETON_LATCHED_PHASE3_ALGORITHM_ID,
    RA_ADAPT_SINGLETON_PHASE3_PLATEAU_ALGORITHM_ID,
    RA_STAGED_SELECTOR_ID,
    RAAdaptOperationalControls,
    RAAdaptRequest,
    RAAdaptResult,
    RESOURCE_WEIGHTING_ALL_PHASE,
    RESOURCE_WEIGHTING_LATE,
    ResolvedRAAdaptProtocol,
    SOURCE_GRAM_NO_OVERLAP_TRUST,
    SUPPORTED_FS_WHITENED_REFIT_CHART,
    canonical_sha256,
    require_protocol_materialization_authority,
)
from pipelines.static_adapt.ra_adapt.pools import (
    CandidateInventory,
    CandidateRecord,
    build_candidate_inventory_lineage_receipt,
    guarded_singleton_generator_identity,
    require_h2o_symmetry_complete_problem,
)
from pipelines.static_adapt.ra_adapt.replay_evidence import (
    build_ra_controller_replay_evidence,
)
from pipelines.static_adapt.ra_adapt.exact_reference_isolation import (
    build_study1_exact_reference_isolation_receipt,
    is_study1_protocol,
)
from pipelines.static_adapt.ra_adapt.insertion_geometry import (
    APPEND_COMMUTATION_REDUCED_MODE,
    APPEND_COMMUTATION_REDUCED_POLICY,
    APPEND_ENDPOINT_POSITION_SCOPE,
    EXACT_TERM_COMMUTATION_EQUIVALENCE,
    validate_commutation_reduced_insertion_receipt,
)
from pipelines.static_adapt.ra_adapt.l3_page12 import (
    PAPER_I_L3_PAGE12_ADAPTER_ID,
    PaperIL3Page12GlobalSingletonGradientPhase0CandidateAdapter,
    is_paper_i_l3_page12_application,
    require_paper_i_l3_page12_materialization,
)
from pipelines.static_adapt.ra_adapt.pure_hubbard_noise_page12 import (
    PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ADAPTER_ID,
    PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ALGORITHM_ID,
    PaperIPureHubbardNoisePage12CandidateAdapter,
    is_paper_i_pure_hubbard_noise_page12_application,
    pure_hubbard_noise_level_contract,
    require_paper_i_pure_hubbard_noise_page12_materialization,
)
from pipelines.static_adapt.ra_adapt.semantic_closure_routes import (
    PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1,
    PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_FORCED_K50_V1,
    PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
    PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2,
    PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2,
    PAPER_I_RA_PHASE0_EXECUTABLE_ROUTE_VARIANTS,
    PAPER_I_RA_PHASE0_POSITION_ROUTE_VARIANTS,
    PAPER_I_RA_PHASE0_V2_ROUTE_VARIANTS,
    PAPER_I_RA_SEMANTIC_ADAPTER_ID,
    PAPER_I_RA_SEMANTIC_ALGORITHM_IDS,
    PaperIRASemanticClosureGlobalSingletonCandidateAdapter,
    build_semantic_closure_route_contract,
    canonical_semantic_execution_problem,
    is_semantic_closure_adapter,
    semantic_closure_native_bundle_digest,
    semantic_closure_native_bundle_id,
    semantic_closure_route_identity,
    semantic_closure_route_identity_from_algorithm,
    validate_semantic_closure_materialization_authority,
    validate_semantic_final_selector_accounting,
    validate_semantic_gradient_adaptive_phase0_receipt,
    validate_semantic_position_phase0_receipt,
    validate_semantic_proxy_phase0_receipt,
    validate_semantic_projected_phase123_receipt,
)
from pipelines.static_adapt.deferred_gram_fallback import (
    DEFERRED_GRAM_ALL_MODELS_INFEASIBLE_FALLBACK_V1,
)
from pipelines.static_adapt.hh_backend_compile_oracle import (
    BACKEND_COMPILE_SCOPE_PHASE2_PHASE3_QISKIT_ONLY_V1,
    BACKEND_COMPILE_SCOPE_PHASE3_QISKIT_ONLY_V1,
    BACKEND_COMPILE_SCOPE_SHARED_ALL_PHASES_V1,
    MARRAKESH_GRAPH_SPAN_MODE,
    ONE_QUBIT_COORDINATE_COMPILED_POSITIVE_DELTA_V1,
)
from pipelines.static_adapt.numerical_physical_integrity import (
    build_ra_numerical_physical_integrity,
)
from pipelines.static_adapt.ra_adapt.support import (
    RetainedSupportReceipt,
    validate_retained_support_receipt,
)
from pipelines.static_adapt.ra_adapt.trust import (
    SourceGramNoOverlapTrustReceipt,
    source_gram_no_overlap_trust_receipt_from_mapping,
)
from pipelines.static_adapt.route_a_schur_selector import (
    ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2,
    ROUTE_A_TRUST_REGION_SOURCE_METRIC_INVERSE_SQRT_NO_OVERLAP_V1,
)
from pipelines.static_adapt.route_a_trust_region import (
    GEOMETRY_EXPANSION_NO_OVERLAP_HOLD_REASON,
    GEOMETRY_EXPANSION_SOURCE_METRIC_LIMITATION,
    HISTORICAL_SINGLETON_GEOMETRY_EXPANSION_CONTEXT_V1,
)
from pipelines.static_adapt.route_a_child_padding import (
    ROUTE_A_CHILD_PADDING_UNCHECKED_DIAGNOSTIC_V1,
)
from pipelines.static_adapt.sr_snake._context import (
    _canonical_route_contract_for_request,
    _resolve_execution_context,
)
from pipelines.static_adapt.sr_snake.contracts import (
    ForkLocalBeam,
    MetricPruning,
    RecoverabilityPruning,
    TrustRegionPruning,
    AcceptedStateResume,
    AlwaysCommutationReducedInsertion,
    AppendCommutationReducedInsertion,
    AppendOnlyInsertion,
    BeamOff,
    EndpointOverlapDisplacementTrust,
    FreshStart,
    PlateauCommutationInsertion,
    PruningOff,
    ResolvedProblemReceipt,
    SRRunRequest,
    SingletonAdmission,
)
from pipelines.static_adapt.ra_adapt.runtime import (
    _execute_resolved_context,
)
from pipelines.static_adapt.sr_snake_route_profile import (
    canonical_sr_snake_insertion_commutation_plateau_v1_contract,
    canonical_sr_snake_insertion_commutation_plateau_v1_contract_sha256,
    canonical_sr_snake_macro_only_always_insertion_fs_prune_beam3x2_v1_contract,
    canonical_sr_snake_macro_only_always_insertion_fs_prune_beam3x2_v1_contract_sha256,
    canonical_sr_snake_macro_only_physical_lanes_commutation_reduced_insertion_diagnostic_v2_contract,
    canonical_sr_snake_macro_only_physical_lanes_commutation_reduced_insertion_diagnostic_v2_contract_sha256,
    canonical_sr_snake_macro_only_physical_lanes_insertion_commutation_plateau_v1_contract,
    canonical_sr_snake_macro_only_physical_lanes_insertion_commutation_plateau_v1_contract_sha256,
    canonical_sr_snake_macro_only_physical_lanes_insertion_commutation_plateau_v2_contract,
    canonical_sr_snake_macro_only_physical_lanes_insertion_commutation_plateau_v2_contract_sha256,
    canonical_sr_snake_macro_only_physical_lanes_v1_contract,
    canonical_sr_snake_macro_only_physical_lanes_v1_contract_sha256,
    canonical_sr_snake_h2o_derivative_resolved_paper_i_v3_contract,
    canonical_sr_snake_h2o_derivative_resolved_paper_i_v3_contract_sha256,
)


RA_ADAPT_ROUTE_CONTRACT_SCHEMA = RA_ADAPT_ROUTE_CONTRACT_SCHEMA_V1
RA_ADAPT_LEGACY_ALGORITHM_ID = "paper_i_ra_adapt_v1"
RA_ADAPT_ALGORITHM_ID = (
    RA_ADAPT_NONSTATIONARY_INCREMENTAL_FULL_RESPONSE_ALGORITHM_ID
)
RA_ADAPT_ORDINARY_BUNDLE_ID = (
    "ordinary_facade_nonstationary_incremental_full_response_v2"
)
RA_ADAPT_LEGACY_ORDINARY_BUNDLE_ID = (
    "ordinary_facade_historical_compatible_v1"
)
PHASE3_CANDIDATE_GAIN_JOINT_MINUS_ACTIVE_ONLY_V1 = (
    RA_ADAPT_INCREMENTAL_CANDIDATE_GAIN_POLICY
)
ROUTE_A_JOINT_STEP_WARM_START_EXACT_GUARDED_V1 = (
    RA_ADAPT_FULL_RESPONSE_REFIT_INITIALIZATION_POLICY
)
RA_ADAPT_GLOBAL_SINGLETON_INSERTION_KIND_BY_ALGORITHM_ID = {
    (
        "paper_i_ra_adapt_global_singleton_"
        "append_commutation_reduced_v1"
    ): AppendCommutationReducedInsertion.kind,
    (
        "paper_i_ra_adapt_global_singleton_plateau_commutation_v1"
    ): PlateauCommutationInsertion.kind,
    (
        "paper_i_ra_adapt_global_singleton_append_commutation_"
        "reduced_qiskit_transpile_cost_v1"
    ): AppendCommutationReducedInsertion.kind,
    (
        "paper_i_ra_adapt_global_singleton_plateau_commutation_"
        "qiskit_transpile_cost_v1"
    ): PlateauCommutationInsertion.kind,
    (
        "paper_i_ra_adapt_global_singleton_plateau_commutation_"
        "qiskit_phase3_only_v1"
    ): PlateauCommutationInsertion.kind,
    (
        "paper_i_ra_adapt_global_singleton_plateau_commutation_"
        "qiskit_phase3_denominator_no_lanes_tau1em6_v1"
    ): PlateauCommutationInsertion.kind,
}
RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_ALGORITHM_ID = (
    "paper_i_ra_adapt_global_singleton_plateau_commutation_"
    "qiskit_phase3_only_v1"
)
RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_DENOMINATOR_NO_LANES_ALGORITHM_ID = (
    "paper_i_ra_adapt_global_singleton_plateau_commutation_"
    "qiskit_phase3_denominator_no_lanes_tau1em6_v1"
)
RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_SOURCE_ALGORITHM_ID = (
    "paper_i_ra_adapt_global_singleton_plateau_commutation_v1"
)
RA_ADAPT_MACRO_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID = (
    "paper_i_ra_adapt_macro_phase1_singleton_phase1_phase2_phase3_"
    "qiskit_phase2_phase3_plateau_no_lanes_v1"
)
RA_ADAPT_MACRO_GRADIENT_PHASE0_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID = (
    "paper_i_ra_adapt_macro_gradient_phase0_singleton_phase1_phase2_phase3_"
    "qiskit_phase2_phase3_plateau_no_lanes_v1"
)
RA_ADAPT_MACRO_GRADIENT_PHASE0_PROXY_NO_LANES_ALGORITHM_ID = (
    "paper_i_ra_adapt_macro_gradient_phase0_macro_phase1_phase2_phase3_"
    "proxy_plateau_no_lanes_v1"
)
RA_ADAPT_MACRO_GRADIENT_PHASE0_MACRO_PHASE23_QISKIT_ALGORITHM_ID = (
    "paper_i_ra_adapt_macro_gradient_phase0_macro_phase1_phase2_phase3_"
    "qiskit_phase2_phase3_plateau_no_lanes_v1"
)
RA_ADAPT_GLOBAL_SINGLETON_GRADIENT_PHASE0_PHASE23_QISKIT_ALGORITHM_ID = (
    "paper_i_ra_adapt_global_singleton_gradient_phase0_phase1_phase2_phase3_"
    "qiskit_phase2_phase3_plateau_no_lanes_v1"
)
RA_ADAPT_PHASE23_QISKIT_ALGORITHM_IDS = frozenset(
    {
        RA_ADAPT_MACRO_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID,
        RA_ADAPT_MACRO_GRADIENT_PHASE0_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID,
        RA_ADAPT_MACRO_GRADIENT_PHASE0_MACRO_PHASE23_QISKIT_ALGORITHM_ID,
        RA_ADAPT_GLOBAL_SINGLETON_GRADIENT_PHASE0_PHASE23_QISKIT_ALGORITHM_ID,
        PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ALGORITHM_ID,
    }
)
RA_ADAPT_GLOBAL_SINGLETON_QISKIT_COST_ALGORITHM_IDS = frozenset(
    algorithm_id
    for algorithm_id in RA_ADAPT_GLOBAL_SINGLETON_INSERTION_KIND_BY_ALGORITHM_ID
    if algorithm_id.endswith("_qiskit_transpile_cost_v1")
)
RA_ADAPT_MACRO_QISKIT_COST_INSERTION_KIND_BY_ALGORITHM_ID = {
    (
        "paper_i_ra_adapt_macro_append_only_"
        "qiskit_transpile_cost_v1"
    ): AppendOnlyInsertion.kind,
    (
        "paper_i_ra_adapt_macro_plateau_insertion_"
        "qiskit_transpile_cost_v1"
    ): PlateauCommutationInsertion.kind,
    (
        "paper_i_ra_adapt_macro_always_insertion_"
        "qiskit_transpile_cost_v1"
    ): AlwaysCommutationReducedInsertion.kind,
    (
        "paper_i_ra_adapt_macro_always_insertion_no_lanes_"
        "qiskit_transpile_cost_v1"
    ): AlwaysCommutationReducedInsertion.kind,
}
RA_ADAPT_MACRO_NO_LANES_ALGORITHM_ID = (
    "paper_i_ra_adapt_macro_always_insertion_no_lanes_"
    "qiskit_transpile_cost_v1"
)
RA_ADAPT_MACRO_NO_LANES_ROUTE_SUFFIX = (
    "macro_only_no_lanes_global_single_population_v1"
)
RA_ADAPT_QISKIT_COST_ROUTE_SUFFIX = (
    "qiskit_full_ansatz_transpile_cost_all_phases_v1"
)
RA_ADAPT_QISKIT_COST_POLICY = (
    "qiskit_full_trial_ansatz_delta_all_phases_v1"
)
RA_ADAPT_QISKIT_COST_PHASE_REUSE = (
    "phase_i_once_then_phase_ii_phase_iii_reuse_v1"
)
RA_ADAPT_PHASE3_QISKIT_COST_ROUTE_SUFFIX = (
    "qiskit_full_ansatz_transpile_cost_phase3_only_v1"
)
RA_ADAPT_PHASE3_QISKIT_COST_POLICY = (
    "qiskit_full_trial_ansatz_positive_clipped_delta_phase3_only_v1"
)
RA_ADAPT_PHASE3_QISKIT_COST_PHASE_REUSE = (
    "phase_i_phase_ii_graph_span_then_phase_iii_recompile_population_v1"
)
RA_ADAPT_PHASE3_QISKIT_DENOMINATOR_ROUTE_SUFFIX = (
    "qiskit_full_ansatz_positive_marginal_denominator_phase3_only_"
    "no_lanes_tau1em6_v1"
)
RA_ADAPT_PHASE3_QISKIT_DENOMINATOR_POLICY = (
    "qiskit_positive_marginal_family_robust_denominator_phase3_only_v1"
)
RA_ADAPT_PHASE23_QISKIT_COST_ROUTE_SUFFIX = (
    "macro_phase1_then_singleton_phase1_then_qiskit_phase2_phase3_"
    "no_lanes_v1"
)
RA_ADAPT_PHASE23_QISKIT_COST_POLICY = (
    "qiskit_full_trial_ansatz_signed_marginal_phase2_phase3_v1"
)
RA_ADAPT_PHASE23_QISKIT_COST_PHASE_REUSE = (
    "phase_ii_phase_iii_shared_oracle_snapshot_and_cache_v1"
)
RA_ADAPT_GRADIENT_PHASE0_SHORTLIST_SIZE = 24
RA_ADAPT_PHASE3_QISKIT_ALGORITHM_IDS = frozenset(
    {
        RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_ALGORITHM_ID,
        RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_DENOMINATOR_NO_LANES_ALGORITHM_ID,
    }
)
RA_ADAPT_ESTIMATOR_ACCOUNTING = (
    "s_alg_equals_n_h_outer_plus_n_h_refit_plus_n_grad_plus_n_metric_v1"
)
RA_ADAPT_COMPILE_IDENTITY = {
    "policy": "table_i_basis_gate_transpile_v1",
    "optimization_level": 0,
    "transpiler_seed": 7,
    "basis_gates": [
        "id",
        "x",
        "sx",
        "rx",
        "ry",
        "rz",
        "h",
        "s",
        "sdg",
        "cx",
        "cz",
    ],
    "reference_preparation_included": True,
    "coupling_map": None,
    "initial_layout": None,
    "routing_method": None,
}
H2O_LINEAR_FD_FAMILY = "molecular_vibronic_h2o_linear_fd"
H2O_RA_MACRO_APPLICATION_LANE = "paper_iv_h2o_linear_fd_ra_adapt_v2"
H2O_RA_SECTOR_COMPLETE_PAULI_BLOCK_APPLICATION_LANE = (
    "paper_iv_h2o_linear_fd_ra_adapt_sector_complete_pauli_block_v1"
)
ALWAYS_REDUCED_INSERTION_MODE = "full_commutation_reduced"
ALWAYS_REDUCED_INSERTION_SCOPE = (
    "full_logical_ansatz_commutation_classes_every_depth_v2"
)
ALWAYS_REDUCED_INSERTION_EQUIVALENCE = (
    "termwise_cross_component_commutation_earliest_representative_v1"
)
GLOBAL_SINGLETON_ROUTE_SUFFIX = (
    "global_guarded_singleton_phase_i__identity_phase_ii"
)
RA_ADAPT_MACRO_GRADIENT_PHASE0_PHASE23_QISKIT_ROUTE_SUFFIX = (
    "macro_abs_gradient_phase0_then_guarded_singleton_phase1_"
    "then_qiskit_phase2_phase3_no_lanes_v1"
)
RA_ADAPT_MACRO_GRADIENT_PHASE0_PROXY_NO_LANES_ROUTE_SUFFIX = (
    "macro_abs_gradient_phase0_then_macro_phase1_then_identity_macro_"
    "phase2_phase3_proxy_no_lanes_v1"
)
RA_ADAPT_MACRO_GRADIENT_PHASE0_MACRO_PHASE23_QISKIT_ROUTE_SUFFIX = (
    "macro_abs_gradient_phase0_then_macro_phase1_then_identity_macro_"
    "phase2_phase3_qiskit_no_lanes_v1"
)
RA_ADAPT_GLOBAL_SINGLETON_GRADIENT_PHASE0_PHASE23_QISKIT_ROUTE_SUFFIX = (
    "global_singleton_abs_gradient_phase0_then_singleton_phase1_"
    "then_qiskit_phase2_phase3_no_lanes_v1"
)


def _global_singleton_supply_contract(
    adapter: Any,
) -> dict[str, str] | None:
    """Return the source identity for the explicit global-singleton RA arm."""

    if not isinstance(adapter, GlobalSinglePauliWordCandidateAdapter):
        return None
    observed = {
        "candidate_adapter_id": str(adapter.adapter_id),
        "phase_i_candidate_supply": str(
            adapter.phase_i_candidate_supply_id
        ),
        "phase_i_candidate_visibility": str(
            adapter.phase_i_candidate_visibility_id
        ),
        "phase_ii_candidate_exposure": str(
            adapter.phase_ii_candidate_exposure_id
        ),
    }
    expected_adapter_id = (
        PAPER_I_RA_SEMANTIC_ADAPTER_ID
        if is_semantic_closure_adapter(adapter)
        else PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ADAPTER_ID
        if isinstance(
            adapter,
            PaperIPureHubbardNoisePage12CandidateAdapter,
        )
        else PAPER_I_L3_PAGE12_ADAPTER_ID
        if isinstance(
            adapter,
            PaperIL3Page12GlobalSingletonGradientPhase0CandidateAdapter,
        )
        else GLOBAL_SINGLETON_GRADIENT_PHASE0_ADAPTER_ID
        if isinstance(
            adapter,
            GlobalSingletonGradientPhase0CandidateAdapter,
        )
        else GLOBAL_SINGLE_PAULI_ADAPTER_ID
    )
    expected = {
        "candidate_adapter_id": expected_adapter_id,
        "phase_i_candidate_supply": (
            PHASE_I_SUPPLY_GLOBAL_GUARDED_SINGLETON
        ),
        "phase_i_candidate_visibility": (
            PHASE_I_VISIBILITY_ALL_EXECUTABLE
        ),
        "phase_ii_candidate_exposure": (
            PHASE_II_EXPOSURE_RETAINED_SINGLETON_IDENTITY
        ),
    }
    if observed != expected:
        raise ValueError(
            "Global-singleton RA candidate-supply identity drifted."
        )
    return expected


def _route_sha256(contract: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(contract),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _h2o_application_active(
    problem: ResolvedProblemContext | None,
    request: RAAdaptRequest,
) -> bool:
    if problem is None:
        return False
    family = str(problem.family_key).strip().lower()
    if family != H2O_LINEAR_FD_FAMILY:
        return False
    if not isinstance(
        request.adapter,
        (
            H2OLinearFDSectorCompletePauliBlockCandidateAdapter,
            H2OLinearFDSymmetryCompleteCandidateAdapter,
        ),
    ):
        raise ValueError(
            "The H2O RA application requires its named derivative-resolved "
            "macro or staged sector-complete Pauli-block adapter."
        )
    require_h2o_symmetry_complete_problem(problem)
    if not (
        isinstance(request.method.admission, SingletonAdmission)
        and isinstance(
            request.method.insertion,
            PlateauCommutationInsertion,
        )
        and isinstance(request.method.pruning, PruningOff)
        and isinstance(request.method.beam, BeamOff)
    ):
        raise ValueError(
            "The H2O RA application is locked to singleton admission, "
            "plateau insertion, pruning off, and beam off."
        )
    return True


def _validate_executed_insertion_contract(
    request: RAAdaptRequest,
    route_contract: Mapping[str, Any],
    *,
    algorithm_id: str,
) -> None:
    execution = route_contract.get("execution_settings")
    invariants = route_contract.get("semantic_invariants")
    if not isinstance(execution, Mapping) or not isinstance(
        invariants, Mapping
    ):
        raise RuntimeError(
            "Canonical RA route omitted its insertion contract surfaces."
        )
    insertion = request.method.insertion
    if isinstance(insertion, AlwaysCommutationReducedInsertion):
        if (
            execution.get("adapt_insertion_mode")
            != ALWAYS_REDUCED_INSERTION_MODE
            or invariants.get("insertion_position_scope")
            != ALWAYS_REDUCED_INSERTION_SCOPE
            or invariants.get("insertion_equivalence_policy")
            != ALWAYS_REDUCED_INSERTION_EQUIVALENCE
        ):
            raise RuntimeError(
                "Always-insertion route is not bound to the full logical "
                "commutation-reduced domain."
            )
    elif isinstance(insertion, PlateauCommutationInsertion):
        declared_plateau_policy = str(
            invariants.get("experimental_insertion_policy", "")
        ).strip()
        if str(route_contract.get("route_profile", "")).startswith(
            "paper_iv_h2o_ra_adapt__"
        ):
            expected_plateau_mode = "insertion_commutation_plateau_v1"
        elif declared_plateau_policy in {
            "insertion_commutation_plateau_v1",
            "insertion_commutation_plateau_v2",
        }:
            expected_plateau_mode = declared_plateau_policy
        else:
            expected_plateau_mode = "insertion_commutation_plateau_v2"
        if (
            execution.get("adapt_insertion_mode")
            != expected_plateau_mode
            or invariants.get("insertion_equivalence_policy")
            != ALWAYS_REDUCED_INSERTION_EQUIVALENCE
        ):
            raise RuntimeError(
                "Plateau-insertion route is not bound to the shared "
                "commutation reducer."
            )
    elif isinstance(insertion, AppendCommutationReducedInsertion):
        if (
            execution.get("adapt_insertion_mode")
            != APPEND_COMMUTATION_REDUCED_MODE
            or invariants.get("insertion_position_scope")
            != APPEND_ENDPOINT_POSITION_SCOPE
            or invariants.get("insertion_equivalence_policy")
            != EXACT_TERM_COMMUTATION_EQUIVALENCE
        ):
            raise RuntimeError(
                "Commutation-reduced append route is not bound to the "
                "shared endpoint reducer."
            )
    elif isinstance(insertion, AppendOnlyInsertion):
        if execution.get("adapt_insertion_mode") != "append_only":
            raise RuntimeError(
                "Append-only route changed its insertion domain."
            )
    else:
        raise RuntimeError("Canonical RA insertion policy is unsupported.")
    if str(algorithm_id) in RA_ADAPT_PHASE3_QISKIT_ALGORITHM_IDS:
        denominator_no_lanes = bool(
            str(algorithm_id)
            == RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_DENOMINATOR_NO_LANES_ALGORITHM_ID
        )
        expected_normalization = (
            "family_robust_v1"
            if denominator_no_lanes
            else "family_robust_symmetric_arctan_v1"
        )
        expected_policy = (
            RA_ADAPT_PHASE3_QISKIT_DENOMINATOR_POLICY
            if denominator_no_lanes
            else RA_ADAPT_PHASE3_QISKIT_COST_POLICY
        )
        if (
            execution.get("phase3_backend_cost_mode")
            != MARRAKESH_GRAPH_SPAN_MODE
            or execution.get("phase3_backend_cost_scope")
            != BACKEND_COMPILE_SCOPE_PHASE3_QISKIT_ONLY_V1
            or execution.get("phase3_backend_name") != "FakeMarrakesh"
            or execution.get("phase3_backend_optimization_level") != 1
            or execution.get("phase3_backend_transpile_seed") != 7
            or execution.get(
                "phase3_hardware_cost_normalization_mode"
            )
            != expected_normalization
            or invariants.get("selector_compile_cost_policy")
            != expected_policy
            or invariants.get("selector_compile_cost_phase_reuse")
            != RA_ADAPT_PHASE3_QISKIT_COST_PHASE_REUSE
            or invariants.get("selector_compile_cost_scope")
            != BACKEND_COMPILE_SCOPE_PHASE3_QISKIT_ONLY_V1
            or invariants.get("phase_iii_qiskit_backend_fallback_allowed")
            is not False
            or invariants.get(
                "phase_iii_qiskit_negative_delta_reward_enabled"
            )
            is not False
            or invariants.get(
                "phase_iii_qiskit_raw_signed_telemetry_required"
            )
            is not True
            or invariants.get("phase_iii_qiskit_structure_theta_value") != 1.0
            or invariants.get(
                "phase_iii_qiskit_full_base_trial_ansatz_transpile"
            )
            is not True
            or invariants.get(
                "phase_iii_qiskit_independent_base_trial_layouts"
            )
            is not True
            or invariants.get(
                "phase_iii_qiskit_one_qubit_coordinate_policy"
            )
            != ONE_QUBIT_COORDINATE_COMPILED_POSITIVE_DELTA_V1
            or invariants.get(
                "phase_iii_qiskit_selector_circuit_coordinates"
            )
            != [
                "positive_clip_delta_N2q",
                "positive_clip_delta_D2q",
                "positive_clip_delta_N1q",
            ]
            or invariants.get("phase_iii_qiskit_population_rescore_policy")
            != "complete_evaluated_phase3_population_before_ranking_v1"
            or invariants.get(
                "phase_iii_qiskit_population_normalization_policy"
            )
            != expected_normalization
            or invariants.get("phase_iii_qiskit_failure_policy")
            != "abort_run_v1"
            or (
                denominator_no_lanes
                and (
                    execution.get("static_lane_route")
                    != "global_single_population"
                    or "physical_lane_shortlist_aggressiveness" in execution
                    or invariants.get("physical_operator_lanes_active")
                    is not False
                    or invariants.get("shortlist_population_policy")
                    != "single_global_population_v1"
                    or invariants.get(
                        "plateau_prior_mean_decrease_ratio_threshold"
                    )
                    != 1.0e-6
                    or invariants.get("phase_iii_score_formula")
                    != (
                        "B3/(1+lambda_2q*cbar_2q+lambda_d*cbar_d+"
                        "lambda_1q*cbar_1q)"
                    )
                    or invariants.get(
                        "phase_iii_qiskit_theta_and_shot_lambdas"
                    )
                    != {"theta": 0.0, "shot": 0.0}
                )
            )
        ):
            raise RuntimeError(
                "The Phase-III-only Qiskit route lost its source-locked "
                "compile-cost contract."
            )
    if (
        str(algorithm_id)
        == RA_ADAPT_MACRO_GRADIENT_PHASE0_PROXY_NO_LANES_ALGORITHM_ID
    ):
        if (
            type(request.adapter) is not MacroGradientPhase0CandidateAdapter
            or execution.get("ra_phase0_gradient_shortlist_policy")
            != MACRO_PHASE0_STANDARD_ADAPT_ABS_GRADIENT_POLICY
            or execution.get("ra_phase0_gradient_shortlist_size")
            != RA_ADAPT_GRADIENT_PHASE0_SHORTLIST_SIZE
            or execution.get("phase3_backend_cost_mode")
            != MARRAKESH_GRAPH_SPAN_MODE
            or "phase3_backend_cost_scope" in execution
            or execution.get("static_lane_route")
            != "global_single_population"
            or "physical_lane_shortlist_aggressiveness" in execution
            or invariants.get("phase0_active") is not True
            or invariants.get("phase0_fubini_metric_active") is not False
            or invariants.get("phase0_compile_cost_active") is not False
            or invariants.get("phase0_estimator_components") != ["N_grad"]
            or invariants.get("selector_qiskit_compile_cost_active")
            is not False
            or invariants.get("physical_operator_lanes_active") is not False
            or invariants.get("shortlist_population_policy")
            != "single_global_population_v1"
            or invariants.get("macro_generator_identity_preserved_all_phases")
            is not True
            or invariants.get("singleton_child_exposure_active") is not False
            or invariants.get(
                "plateau_prior_mean_decrease_ratio_threshold"
            )
            != 1.0e-4
        ):
            raise RuntimeError(
                "The macro-only gradient-Phase-0 proxy/no-lanes route "
                "drifted from its typed contract."
            )
    if (
        str(algorithm_id)
        == RA_ADAPT_MACRO_GRADIENT_PHASE0_MACRO_PHASE23_QISKIT_ALGORITHM_ID
    ):
        if (
            type(request.adapter) is not MacroGradientPhase0CandidateAdapter
            or execution.get("ra_phase0_gradient_shortlist_policy")
            != MACRO_PHASE0_STANDARD_ADAPT_ABS_GRADIENT_POLICY
            or execution.get("ra_phase0_gradient_shortlist_size")
            != RA_ADAPT_GRADIENT_PHASE0_SHORTLIST_SIZE
            or execution.get("phase3_backend_cost_mode")
            != MARRAKESH_GRAPH_SPAN_MODE
            or execution.get("phase3_backend_cost_scope")
            != BACKEND_COMPILE_SCOPE_PHASE2_PHASE3_QISKIT_ONLY_V1
            or execution.get("phase3_backend_name") != "FakeMarrakesh"
            or execution.get("phase3_backend_optimization_level") != 1
            or execution.get("phase3_backend_transpile_seed") != 7
            or execution.get("static_lane_route")
            != "global_single_population"
            or "physical_lane_shortlist_aggressiveness" in execution
            or invariants.get("phase0_active") is not True
            or invariants.get("phase0_fubini_metric_active") is not False
            or invariants.get("phase0_compile_cost_active") is not False
            or invariants.get("phase0_estimator_components") != ["N_grad"]
            or invariants.get("selector_qiskit_compile_cost_active")
            is not True
            or invariants.get("selector_compile_cost_policy")
            != RA_ADAPT_PHASE23_QISKIT_COST_POLICY
            or invariants.get("selector_compile_cost_phase_reuse")
            != RA_ADAPT_PHASE23_QISKIT_COST_PHASE_REUSE
            or invariants.get("selector_compile_cost_scope")
            != BACKEND_COMPILE_SCOPE_PHASE2_PHASE3_QISKIT_ONLY_V1
            or invariants.get("phase_i_compile_cost_source")
            != "structural_proxy_v1"
            or invariants.get("phase_ii_compile_cost_source")
            != "backend_transpile_v1"
            or invariants.get("phase_iii_compile_cost_source")
            != "backend_transpile_v1"
            or invariants.get(
                "phase_ii_phase_iii_qiskit_negative_delta_reward_enabled"
            )
            is not True
            or invariants.get(
                "phase_ii_phase_iii_qiskit_backend_fallback_allowed"
            )
            is not False
            or invariants.get(
                "phase_ii_phase_iii_qiskit_population_normalization_policy"
            )
            != "zero_centered_signed_arctan_v1"
            or invariants.get("physical_operator_lanes_active") is not False
            or invariants.get("shortlist_population_policy")
            != "single_global_population_v1"
            or invariants.get("macro_generator_identity_preserved_all_phases")
            is not True
            or invariants.get("singleton_child_exposure_active") is not False
            or invariants.get(
                "plateau_prior_mean_decrease_ratio_threshold"
            )
            != 1.0e-4
        ):
            raise RuntimeError(
                "The macro-only gradient-Phase-0 Phase-II/III Qiskit route "
                "drifted from its typed contract."
            )
    if str(algorithm_id) == RA_ADAPT_SINGLETON_PHASE3_PLATEAU_ALGORITHM_ID:
        expected_execution = {
            "ra_phase3_population_activation_policy": (
                RA_ADAPT_PHASE3_POPULATION_ON_INSERTION_PLATEAU
            ),
            "ra_phase3_preplateau_materialization_policy": (
                RA_ADAPT_PHASE3_PREPLATEAU_WINNER_MATERIALIZATION
            ),
        }
        expected_invariants = {
            "candidate_representation": CANDIDATE_REPRESENTATION_SINGLE_PAULI,
            "phase1_activation_scope": "all_controller_rounds_v1",
            "phase2_activation_scope": "all_controller_rounds_v1",
            "phase3_competitive_population_activation": (
                RA_ADAPT_PHASE3_POPULATION_ON_INSERTION_PLATEAU
            ),
            "phase3_activation_source": (
                "same_round_authenticated_insertion_plateau_domain_open_v1"
            ),
            "phase3_preplateau_admission_authority": (
                "phase2_raw_score_top_rank_v1"
            ),
            "phase3_preplateau_materialization_policy": (
                RA_ADAPT_PHASE3_PREPLATEAU_WINNER_MATERIALIZATION
            ),
            "phase3_activation_independent_latch": False,
            "phase3_activation_hysteresis_active": False,
        }
        if (
            not isinstance(request.adapter, SinglePauliWordCandidateAdapter)
            or not isinstance(request.method.admission, SingletonAdmission)
            or not isinstance(request.method.insertion, PlateauCommutationInsertion)
            or any(
                execution.get(key) != value
                for key, value in expected_execution.items()
            )
            or any(
                invariants.get(key) != value
                for key, value in expected_invariants.items()
            )
        ):
            raise RuntimeError(
                "The singleton Phase-III-on-plateau ablation drifted from "
                "its typed route contract."
            )
    if str(algorithm_id) == RA_ADAPT_SINGLETON_LATCHED_PHASE3_ALGORITHM_ID:
        expected_execution = {
            "ra_phase3_population_activation_policy": (
                RA_ADAPT_PHASE3_POPULATION_LATCHED_ON_PROGRESS_PLATEAU
            ),
            "ra_phase3_preplateau_materialization_policy": (
                RA_ADAPT_PHASE3_PREPLATEAU_WINNER_MATERIALIZATION
            ),
            "ra_insertion_plateau_history_scope": (
                RA_ADAPT_POST_LATCH_INSERTION_TRIGGER_SCOPE
            ),
        }
        expected_invariants = {
            "candidate_representation": CANDIDATE_REPRESENTATION_SINGLE_PAULI,
            "phase1_activation_scope": "all_controller_rounds_v1",
            "phase2_activation_scope": "all_controller_rounds_v1",
            "phase3_competitive_population_activation": (
                RA_ADAPT_PHASE3_POPULATION_LATCHED_ON_PROGRESS_PLATEAU
            ),
            "phase3_activation_source": (
                "first_authenticated_progress_plateau_domain_open_latched_v1"
            ),
            "phase3_preplateau_admission_authority": (
                "phase2_raw_score_top_rank_v1"
            ),
            "phase3_preplateau_materialization_policy": (
                RA_ADAPT_PHASE3_PREPLATEAU_WINNER_MATERIALIZATION
            ),
            "phase3_activation_independent_latch": True,
            "phase3_activation_hysteresis_active": False,
            "phase3_latch_retirement_policy": "never_close_v1",
            "insertion_plateau_history_scope": (
                RA_ADAPT_POST_LATCH_INSERTION_TRIGGER_SCOPE
            ),
            "insertion_activation_requires_prior_phase3_latch": True,
            "insertion_activation_changes_phase3_latch": False,
        }
        if (
            not isinstance(request.adapter, SinglePauliWordCandidateAdapter)
            or not isinstance(request.method.admission, SingletonAdmission)
            or not isinstance(request.method.insertion, PlateauCommutationInsertion)
            or any(
                execution.get(key) != value
                for key, value in expected_execution.items()
            )
            or any(
                invariants.get(key) != value
                for key, value in expected_invariants.items()
            )
        ):
            raise RuntimeError(
                "The latched Phase-III/separate-insertion ablation drifted "
                "from its typed route contract."
            )
    if str(algorithm_id) == RA_ADAPT_ALGORITHM_ID and (
        route_contract.get("algorithm_id") != RA_ADAPT_ALGORITHM_ID
        or execution.get("ra_active_gradient_policy")
        != ACTIVE_GRADIENT_MEASURED
        or execution.get("ra_phase3_candidate_gain_policy")
        != PHASE3_CANDIDATE_GAIN_JOINT_MINUS_ACTIVE_ONLY_V1
        or execution.get("ra_accepted_refit_initialization_policy")
        != ROUTE_A_JOINT_STEP_WARM_START_EXACT_GUARDED_V1
        or invariants.get("phase3_candidate_gain_policy")
        != PHASE3_CANDIDATE_GAIN_JOINT_MINUS_ACTIVE_ONLY_V1
        or invariants.get("accepted_refit_initialization_policy")
        != ROUTE_A_JOINT_STEP_WARM_START_EXACT_GUARDED_V1
        or invariants.get("accepted_refit_initialization_coordinate_scope")
        != "full_existing_active_plus_new_batch_coordinates_v1"
        or invariants.get("accepted_refit_initialization_exact_guard")
        is not True
    ):
        raise RuntimeError(
            "Canonical RA v2 execution lost its incremental-gain or "
            "full-response initialization contract."
        )


def _sr_request(request: RAAdaptRequest) -> SRRunRequest:
    return SRRunRequest(
        method=request.method,
        execution=request.execution,
        observation=request.observation,
    )


def _compose_macro_plateau_pruning_and_beam(
    request: RAAdaptRequest,
    *,
    parent_contract: Mapping[str, Any],
    parent_digest: str,
) -> tuple[dict[str, Any], str]:
    """Overlay typed pruning/beam semantics on the macro plateau parent."""

    if isinstance(request.method.pruning, PruningOff) and isinstance(
        request.method.beam, BeamOff
    ):
        return dict(parent_contract), str(parent_digest)

    (
        _policy_request,
        _policy_profile,
        policy_contract,
        _policy_digest,
    ) = _canonical_route_contract_for_request(_sr_request(request))
    normalized = json.loads(json.dumps(parent_contract, sort_keys=True))

    execution = dict(normalized.get("execution_settings", {}))
    for key in tuple(execution):
        if str(key).startswith(("phase1_prune_", "adapt_beam_")):
            execution.pop(key)
    execution.update(
        {
            str(key): value
            for key, value in policy_contract["execution_settings"].items()
            if str(key).startswith(("phase1_prune_", "adapt_beam_"))
        }
    )
    normalized["execution_settings"] = execution

    invariants = dict(normalized.get("semantic_invariants", {}))
    policy_exact_keys = {
        "pruning_active",
        "terminal_prune_active",
        "canonical_admission_policy",
        "canonical_insertion_policy",
        "canonical_pruning_policy",
        "canonical_beam_policy",
        "canonical_composition_schema",
        "compatibility_resolution_active",
    }
    for key in tuple(invariants):
        if (
            str(key).startswith(("prune_", "beam_"))
            or key in policy_exact_keys
        ):
            invariants.pop(key)
    invariants.update(
        {
            str(key): value
            for key, value in policy_contract["semantic_invariants"].items()
            if (
                str(key).startswith(("prune_", "beam_"))
                or key in policy_exact_keys
            )
        }
    )
    normalized["semantic_invariants"] = invariants
    normalized["route_profile"] = (
        f"{parent_contract['route_profile']}"
        f"__pruning-{request.method.pruning.kind}"
        f"__beam-{request.method.beam.kind}"
    )
    normalized["lineage_authority"] = {
        "parent_route_profile": str(parent_contract["route_profile"]),
        "parent_contract_sha256": str(parent_digest),
        "typed_policy_composition": request.method.to_dict(),
        "scientific_result_anchor_claimed": False,
    }
    normalized = json.loads(json.dumps(normalized, sort_keys=True))
    return normalized, _route_sha256(normalized)


def _macro_parent_contract(
    request: RAAdaptRequest,
    *,
    algorithm_id: str,
) -> tuple[dict[str, Any], str]:
    # The algorithm identifier is provenance only.  Executable semantics are
    # selected exclusively by the typed request contract.
    _ = algorithm_id
    if isinstance(
        request.method.insertion,
        AlwaysCommutationReducedInsertion,
    ):
        if isinstance(request.method.beam, ForkLocalBeam) and isinstance(
            request.method.pruning, (MetricPruning, TrustRegionPruning)
        ):
            nomination_route = (
                "metric_regularized_v1"
                if isinstance(request.method.pruning, MetricPruning)
                else "full_logical_fs_trust_delete_refit_v1"
            )
            # Always-insertion with the sealed 3x2 fork-local beam and live
            # recoverability pruning. Selected by the typed request, not by
            # the algorithm identifier.
            return (
                canonical_sr_snake_macro_only_always_insertion_fs_prune_beam3x2_v1_contract(
                    nomination_route=nomination_route
                ),
                canonical_sr_snake_macro_only_always_insertion_fs_prune_beam3x2_v1_contract_sha256(
                    nomination_route=nomination_route
                ),
            )
        return (
            canonical_sr_snake_macro_only_physical_lanes_commutation_reduced_insertion_diagnostic_v2_contract(),
            canonical_sr_snake_macro_only_physical_lanes_commutation_reduced_insertion_diagnostic_v2_contract_sha256(),
        )
    if isinstance(request.method.insertion, PlateauCommutationInsertion):
        if str(algorithm_id) == RA_ADAPT_LEGACY_ALGORITHM_ID:
            parent_contract = (
                canonical_sr_snake_macro_only_physical_lanes_insertion_commutation_plateau_v1_contract()
            )
            parent_digest = (
                canonical_sr_snake_macro_only_physical_lanes_insertion_commutation_plateau_v1_contract_sha256()
            )
        else:
            parent_contract = (
                canonical_sr_snake_macro_only_physical_lanes_insertion_commutation_plateau_v2_contract()
            )
            parent_digest = (
                canonical_sr_snake_macro_only_physical_lanes_insertion_commutation_plateau_v2_contract_sha256()
            )
        return _compose_macro_plateau_pruning_and_beam(
            request,
            parent_contract=parent_contract,
            parent_digest=parent_digest,
        )
    if isinstance(
        request.method.insertion,
        AppendCommutationReducedInsertion,
    ):
        contract = (
            canonical_sr_snake_macro_only_physical_lanes_v1_contract()
        )
        normalized = json.loads(json.dumps(contract, sort_keys=True))
        execution = dict(normalized.get("execution_settings", {}))
        execution["adapt_insertion_mode"] = (
            APPEND_COMMUTATION_REDUCED_MODE
        )
        normalized["execution_settings"] = execution
        invariants = dict(normalized.get("semantic_invariants", {}))
        invariants.update(
            {
                "insertion_position_scope": (
                    APPEND_ENDPOINT_POSITION_SCOPE
                ),
                "insertion_equivalence_policy": (
                    EXACT_TERM_COMMUTATION_EQUIVALENCE
                ),
            }
        )
        normalized["semantic_invariants"] = invariants
        return normalized, _route_sha256(normalized)
    if isinstance(request.method.insertion, AppendOnlyInsertion):
        return (
            canonical_sr_snake_macro_only_physical_lanes_v1_contract(),
            canonical_sr_snake_macro_only_physical_lanes_v1_contract_sha256(),
        )
    raise ValueError("The macro RA route has no authorized insertion policy.")


def _repaired_route_contract(
    request: RAAdaptRequest,
    *,
    active_gradient_policy: str,
    resource_weighting_scope: str,
    algorithm_id: str,
    problem: ResolvedProblemContext | None = None,
) -> tuple[str, str, dict[str, Any], str]:
    representation = str(request.adapter.candidate_representation_id)
    global_singleton_supply = _global_singleton_supply_contract(
        request.adapter
    )
    algorithm_identity = str(algorithm_id)
    semantic_closure_active = bool(
        algorithm_identity in PAPER_I_RA_SEMANTIC_ALGORITHM_IDS
    )
    endpoint_overlap_trust_active = isinstance(
        request.method.trust_update,
        EndpointOverlapDisplacementTrust,
    )
    resolved_trust_policy_id = (
        ENDPOINT_OVERLAP_DISPLACEMENT_TRUST
        if endpoint_overlap_trust_active
        else SOURCE_GRAM_NO_OVERLAP_TRUST
    )
    resolved_runtime_trust_policy = (
        ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2
        if endpoint_overlap_trust_active
        else ROUTE_A_TRUST_REGION_SOURCE_METRIC_INVERSE_SQRT_NO_OVERLAP_V1
    )
    phase3_population_plateau_ablation = bool(
        algorithm_identity
        == RA_ADAPT_SINGLETON_PHASE3_PLATEAU_ALGORITHM_ID
    )
    latched_phase3_separate_insertion_ablation = bool(
        algorithm_identity
        == RA_ADAPT_SINGLETON_LATCHED_PHASE3_ALGORITHM_ID
    )
    phase3_population_controlled_ablation = bool(
        phase3_population_plateau_ablation
        or latched_phase3_separate_insertion_ablation
    )
    canonical_full_response_v2 = bool(
        algorithm_identity == RA_ADAPT_ALGORITHM_ID
    )
    if phase3_population_controlled_ablation:
        if (
            not isinstance(request.adapter, SinglePauliWordCandidateAdapter)
            or isinstance(request.adapter, GlobalSinglePauliWordCandidateAdapter)
            or representation != CANDIDATE_REPRESENTATION_SINGLE_PAULI
            or not isinstance(request.method.admission, SingletonAdmission)
            or not isinstance(
                request.method.insertion,
                PlateauCommutationInsertion,
            )
            or str(active_gradient_policy) != ACTIVE_GRADIENT_STATIONARY
            or str(resource_weighting_scope) != RESOURCE_WEIGHTING_LATE
        ):
            raise ValueError(
                "The plateau-controlled Phase-III ablation requires the Paper-I "
                "single-Pauli adapter, singleton admission, plateau-v2 "
                "insertion, stationary active response, and late resource "
                "weighting."
            )
    if canonical_full_response_v2 and (
        str(active_gradient_policy) != ACTIVE_GRADIENT_MEASURED
    ):
        raise ValueError(
            "The canonical RA-ADAPT v2 algorithm requires measured active "
            "response."
        )
    expected_macro_insertion_kind = (
        RA_ADAPT_MACRO_QISKIT_COST_INSERTION_KIND_BY_ALGORITHM_ID.get(
            algorithm_identity
        )
    )
    qiskit_cost_active = bool(
        algorithm_identity
        in RA_ADAPT_GLOBAL_SINGLETON_QISKIT_COST_ALGORITHM_IDS
        or expected_macro_insertion_kind is not None
    )
    phase3_only_qiskit_cost_active = bool(
        algorithm_identity in RA_ADAPT_PHASE3_QISKIT_ALGORITHM_IDS
    )
    phase23_qiskit_cost_active = bool(
        algorithm_identity in RA_ADAPT_PHASE23_QISKIT_ALGORITHM_IDS
        or semantic_closure_active
    )
    macro_gradient_phase0_active = bool(
        algorithm_identity
        == (
            RA_ADAPT_MACRO_GRADIENT_PHASE0_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID
        )
    )
    macro_only_gradient_phase0_proxy_active = bool(
        algorithm_identity
        == RA_ADAPT_MACRO_GRADIENT_PHASE0_PROXY_NO_LANES_ALGORITHM_ID
    )
    macro_only_gradient_phase0_qiskit_active = bool(
        algorithm_identity
        == RA_ADAPT_MACRO_GRADIENT_PHASE0_MACRO_PHASE23_QISKIT_ALGORITHM_ID
    )
    macro_only_gradient_phase0_active = bool(
        macro_only_gradient_phase0_proxy_active
        or macro_only_gradient_phase0_qiskit_active
    )
    global_singleton_gradient_phase0_active = bool(
        algorithm_identity
        in {
            RA_ADAPT_GLOBAL_SINGLETON_GRADIENT_PHASE0_PHASE23_QISKIT_ALGORITHM_ID,
            PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ALGORITHM_ID,
            *PAPER_I_RA_SEMANTIC_ALGORITHM_IDS,
        }
    )
    pure_hubbard_noise_application = bool(
        algorithm_identity
        == PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ALGORITHM_ID
    )
    if bool(
        isinstance(
            request.adapter,
            MacroGradientPhase0ThenSingletonCandidateAdapter,
        )
    ) != macro_gradient_phase0_active:
        raise ValueError(
            "The macro gradient-Phase-0 adapter and algorithm identity must "
            "be selected together."
        )
    if pure_hubbard_noise_application != bool(
        type(request.adapter) is PaperIPureHubbardNoisePage12CandidateAdapter
    ):
        raise ValueError(
            "The pure-Hubbard full-noise adapter and algorithm identity must "
            "be selected together."
        )
    if bool(
        type(request.adapter) is MacroGradientPhase0CandidateAdapter
    ) != macro_only_gradient_phase0_active:
        raise ValueError(
            "The macro-only gradient-Phase-0 adapter and algorithm identity "
            "must be selected together."
        )
    gradient_phase0_adapter_active = bool(
        isinstance(
            request.adapter,
            GlobalSingletonGradientPhase0CandidateAdapter,
        )
        or is_semantic_closure_adapter(request.adapter)
    )
    if (
        gradient_phase0_adapter_active
        != global_singleton_gradient_phase0_active
        or semantic_closure_active
        != is_semantic_closure_adapter(request.adapter)
    ):
        raise ValueError(
            "The global-singleton gradient-Phase-0 adapter and algorithm "
            "identity must be selected together."
        )
    phase3_qiskit_denominator_no_lanes_active = bool(
        algorithm_identity
        == RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_DENOMINATOR_NO_LANES_ALGORITHM_ID
    )
    any_qiskit_cost_active = bool(
        qiskit_cost_active
        or phase3_only_qiskit_cost_active
        or phase23_qiskit_cost_active
    )
    if (
        algorithm_identity
        == RA_ADAPT_MACRO_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID
    ) and (
        not isinstance(
            request.adapter,
            MacroThenSingletonPhaseICandidateAdapter,
        )
        or isinstance(
            request.adapter,
            MacroGradientPhase0ThenSingletonCandidateAdapter,
        )
        or not isinstance(request.method.admission, SingletonAdmission)
        or not isinstance(
            request.method.insertion,
            PlateauCommutationInsertion,
        )
        or str(active_gradient_policy) != ACTIVE_GRADIENT_STATIONARY
        or str(resource_weighting_scope) != RESOURCE_WEIGHTING_ALL_PHASE
    ):
        raise ValueError(
            "The staged Phase-II/III Qiskit route requires macro Phase-I "
            "prefiltering, singleton Phase-I/II/III, plateau-v2 insertion, "
            "stationary active response, and all-phase resource weighting."
        )
    if macro_gradient_phase0_active and (
        not isinstance(
            request.adapter,
            MacroGradientPhase0ThenSingletonCandidateAdapter,
        )
        or str(request.adapter.macro_phase0_policy_id)
        != MACRO_PHASE0_STANDARD_ADAPT_ABS_GRADIENT_POLICY
        or not isinstance(request.method.admission, SingletonAdmission)
        or not isinstance(
            request.method.insertion,
            PlateauCommutationInsertion,
        )
        or str(active_gradient_policy) != ACTIVE_GRADIENT_STATIONARY
        or str(resource_weighting_scope) != RESOURCE_WEIGHTING_ALL_PHASE
    ):
        raise ValueError(
            "The macro gradient-Phase-0 route requires the exact |g|-only "
            "macro screen, guarded singleton Phase-I/II/III, plateau-v2 "
            "insertion, stationary active response, and all-phase resource "
            "weighting."
        )
    if macro_only_gradient_phase0_active and (
        type(request.adapter) is not MacroGradientPhase0CandidateAdapter
        or str(request.adapter.macro_phase0_policy_id)
        != MACRO_PHASE0_STANDARD_ADAPT_ABS_GRADIENT_POLICY
        or representation != CANDIDATE_REPRESENTATION_MACRO
        or not isinstance(request.method.admission, SingletonAdmission)
        or not isinstance(
            request.method.insertion,
            PlateauCommutationInsertion,
        )
        or str(active_gradient_policy) != ACTIVE_GRADIENT_STATIONARY
        or str(resource_weighting_scope) != RESOURCE_WEIGHTING_ALL_PHASE
    ):
        raise ValueError(
            "The macro-only gradient-Phase-0 route requires the exact "
            "|g|-only macro screen, intact macro Phase-I/II/III, singleton "
            "admission, plateau-v2 insertion, stationary active response, "
            "and all-phase resource weighting."
        )
    if (
        global_singleton_gradient_phase0_active
        and not semantic_closure_active
        and (
            not isinstance(
                request.adapter,
                GlobalSingletonGradientPhase0CandidateAdapter,
            )
            or str(request.adapter.phase0_shortlist_policy_id)
            != GLOBAL_SINGLETON_ABSOLUTE_GRADIENT_PHASE0_POLICY
            or not isinstance(request.method.admission, SingletonAdmission)
            or not isinstance(
                request.method.insertion,
                PlateauCommutationInsertion,
            )
            or str(active_gradient_policy) != ACTIVE_GRADIENT_STATIONARY
            or str(resource_weighting_scope) != RESOURCE_WEIGHTING_ALL_PHASE
        )
    ):
        raise ValueError(
            "The initialized-singleton gradient-Phase-0 route requires the "
            "exact |g|-only global-singleton screen, singleton Phase-I/II/III, "
            "plateau-v2 insertion, stationary active response, and all-phase "
            "resource weighting."
        )
    if phase3_only_qiskit_cost_active and (
        global_singleton_supply is None
        or not isinstance(request.method.admission, SingletonAdmission)
        or not isinstance(
            request.method.insertion, PlateauCommutationInsertion
        )
        or str(active_gradient_policy) != ACTIVE_GRADIENT_STATIONARY
        or str(resource_weighting_scope) != RESOURCE_WEIGHTING_ALL_PHASE
    ):
        raise ValueError(
            "The Phase-III-only Qiskit candidate is source-locked to the "
            "stationary all-phase global-singleton plateau-v2 route."
        )
    if (
        (
            algorithm_identity
            in RA_ADAPT_GLOBAL_SINGLETON_QISKIT_COST_ALGORITHM_IDS
            or phase3_only_qiskit_cost_active
        )
        and global_singleton_supply is None
    ):
        raise ValueError(
            "The Qiskit full-ansatz selector-cost algorithm is restricted "
            "to the global-singleton candidate-supply route."
        )
    if expected_macro_insertion_kind is not None:
        if (
            not isinstance(request.adapter, MacroCandidateAdapter)
            or representation != CANDIDATE_REPRESENTATION_MACRO
        ):
            raise ValueError(
                "The macro Qiskit selector-cost algorithm requires the "
                "macro candidate adapter."
            )
        observed_insertion_kind = str(request.method.insertion.kind)
        if observed_insertion_kind != expected_macro_insertion_kind:
            raise ValueError(
                "The macro Qiskit selector-cost algorithm insertion policy "
                f"requires {expected_macro_insertion_kind!r}, observed "
                f"{observed_insertion_kind!r}."
            )
    if (
        any_qiskit_cost_active
        and str(resource_weighting_scope) != RESOURCE_WEIGHTING_ALL_PHASE
    ):
        raise ValueError(
            "The Qiskit selector-cost algorithms retain the source route's "
            "all-phase resource weighting."
        )
    sr_request = _sr_request(request)
    h2o_application = _h2o_application_active(problem, request)
    if h2o_application:
        parent_contract = (
            canonical_sr_snake_h2o_derivative_resolved_paper_i_v3_contract()
        )
        parent_digest = (
            canonical_sr_snake_h2o_derivative_resolved_paper_i_v3_contract_sha256()
        )
        policy_parent_contract = json.loads(
            json.dumps(parent_contract, sort_keys=True)
        )
        parent_execution = dict(
            policy_parent_contract.get("execution_settings", {})
        )
        parent_execution["adapt_insertion_mode"] = (
            "insertion_commutation_plateau_v1"
        )
        policy_parent_contract["execution_settings"] = parent_execution
    elif representation == CANDIDATE_REPRESENTATION_MACRO:
        parent_contract, parent_digest = _macro_parent_contract(
            request, algorithm_id=algorithm_id
        )
        policy_parent_contract = parent_contract
    elif representation == CANDIDATE_REPRESENTATION_SINGLE_PAULI:
        if pure_hubbard_noise_application or (
            algorithm_identity == RA_ADAPT_LEGACY_ALGORITHM_ID
            and isinstance(
                request.method.insertion,
                PlateauCommutationInsertion,
            )
        ):
            parent_contract = (
                canonical_sr_snake_insertion_commutation_plateau_v1_contract()
            )
            parent_digest = (
                canonical_sr_snake_insertion_commutation_plateau_v1_contract_sha256()
            )
        else:
            (
                _profile_request,
                _profile,
                parent_contract,
                parent_digest,
            ) = _canonical_route_contract_for_request(sr_request)
        policy_parent_contract = parent_contract
    else:
        raise ValueError("Unknown RA candidate representation.")

    if pure_hubbard_noise_application:
        policy_parent_contract = json.loads(
            json.dumps(policy_parent_contract, sort_keys=True)
        )
        pure_execution = dict(
            policy_parent_contract.get("execution_settings", {})
        )
        pure_execution["adapt_insertion_mode"] = (
            "insertion_commutation_plateau_v1"
        )
        policy_parent_contract["execution_settings"] = pure_execution

    insertion_mode = str(
        policy_parent_contract.get("execution_settings", {}).get(
            "adapt_insertion_mode", ""
        )
    )
    h2o_candidate_route_suffix = (
        "__sector_complete_pauli_block_exposure_v1"
        if h2o_application
        and isinstance(
            request.adapter,
            H2OLinearFDSectorCompletePauliBlockCandidateAdapter,
        )
        else "__symmetry_complete_macro_v1"
        if h2o_application
        else ""
    )
    candidate_route_suffix = (
        "__" + GLOBAL_SINGLETON_ROUTE_SUFFIX
        if global_singleton_supply is not None
        else ""
    )
    phase23_qiskit_route_suffix = (
        RA_ADAPT_MACRO_GRADIENT_PHASE0_PHASE23_QISKIT_ROUTE_SUFFIX
        if macro_gradient_phase0_active
        else RA_ADAPT_MACRO_GRADIENT_PHASE0_MACRO_PHASE23_QISKIT_ROUTE_SUFFIX
        if macro_only_gradient_phase0_qiskit_active
        else RA_ADAPT_GLOBAL_SINGLETON_GRADIENT_PHASE0_PHASE23_QISKIT_ROUTE_SUFFIX
        if global_singleton_gradient_phase0_active
        else RA_ADAPT_PHASE23_QISKIT_COST_ROUTE_SUFFIX
    )
    policy_composition_route_suffix = (
        (
            f"__pruning-{request.method.pruning.kind}"
            f"__beam-{request.method.beam.kind}"
        )
        if (
            not isinstance(request.method.pruning, PruningOff)
            or not isinstance(request.method.beam, BeamOff)
        )
        else ""
    )
    profile = (
        (
            "paper_iv_h2o_ra_adapt__"
            if h2o_application
            else "paper_i_ra_adapt__"
        )
        + representation
        + h2o_candidate_route_suffix
        + "__"
        + insertion_mode
        + candidate_route_suffix
        + "__"
        + str(active_gradient_policy)
        + "__"
        + str(resource_weighting_scope)
        + policy_composition_route_suffix
        + (
            "__endpoint_overlap_displacement_trust_v1"
            if endpoint_overlap_trust_active
            else ""
        )
        + (
            "__incremental_active_baseline__exact_guarded_full_response"
            if canonical_full_response_v2
            else ""
        )
        + (
            "__" + RA_ADAPT_MACRO_GRADIENT_PHASE0_PROXY_NO_LANES_ROUTE_SUFFIX
            if macro_only_gradient_phase0_proxy_active
            else ""
        )
        + (
            "__" + phase23_qiskit_route_suffix
            if phase23_qiskit_cost_active
            else
            "__"
            + (
                RA_ADAPT_PHASE3_QISKIT_DENOMINATOR_ROUTE_SUFFIX
                if phase3_qiskit_denominator_no_lanes_active
                else RA_ADAPT_PHASE3_QISKIT_COST_ROUTE_SUFFIX
            )
            if phase3_only_qiskit_cost_active
            else "__" + RA_ADAPT_QISKIT_COST_ROUTE_SUFFIX
            if qiskit_cost_active
            else ""
        )
        + (
            "__phase3_population_on_insertion_plateau_v1"
            if phase3_population_plateau_ablation
            else (
                "__phase3_population_latched_on_progress_plateau_v1"
                "__insertion_on_phase3_plateau_v1"
            )
            if latched_phase3_separate_insertion_ablation
            else ""
        )
    )
    contract = json.loads(json.dumps(policy_parent_contract, sort_keys=True))
    contract["schema"] = (
        RA_ADAPT_ROUTE_CONTRACT_SCHEMA_V2
        if canonical_full_response_v2
        else RA_ADAPT_ROUTE_CONTRACT_SCHEMA_V1
    )
    contract["route_family"] = "ra_adapt"
    contract["route_profile"] = profile
    if canonical_full_response_v2:
        contract["algorithm_id"] = algorithm_identity
    execution = dict(contract.get("execution_settings", {}))
    execution.update(
        {
            "historical_singleton_coordinate_solve_policy": (
                PROJECTED_GENERALIZED_SOLVER
            ),
            "historical_singleton_trust_region_update_policy": (
                resolved_runtime_trust_policy
            ),
            "adapt_accepted_refit_scope": FULL_ENLARGED_ACCEPTED_REFIT,
            "adapt_accepted_refit_coordinate_chart": (
                SUPPORTED_FS_WHITENED_REFIT_CHART
            ),
            "adapt_accepted_refit_base_chart_policy": (
                EXPANDED_RUNTIME_PROJECTED_LOGICAL_BASE_CHART
            ),
            "ra_active_gradient_policy": str(active_gradient_policy),
            "ra_resource_weighting_scope": str(resource_weighting_scope),
        }
    )
    if canonical_full_response_v2:
        execution.update(
            {
                "ra_phase3_candidate_gain_policy": (
                    PHASE3_CANDIDATE_GAIN_JOINT_MINUS_ACTIVE_ONLY_V1
                ),
                "ra_accepted_refit_initialization_policy": (
                    ROUTE_A_JOINT_STEP_WARM_START_EXACT_GUARDED_V1
                ),
            }
        )
    if phase3_population_controlled_ablation:
        execution.update(
            {
                "ra_phase3_population_activation_policy": (
                    RA_ADAPT_PHASE3_POPULATION_LATCHED_ON_PROGRESS_PLATEAU
                    if latched_phase3_separate_insertion_ablation
                    else RA_ADAPT_PHASE3_POPULATION_ON_INSERTION_PLATEAU
                ),
                "ra_phase3_preplateau_materialization_policy": (
                    RA_ADAPT_PHASE3_PREPLATEAU_WINNER_MATERIALIZATION
                ),
            }
        )
    if latched_phase3_separate_insertion_ablation:
        execution["ra_insertion_plateau_history_scope"] = (
            RA_ADAPT_POST_LATCH_INSERTION_TRIGGER_SCOPE
        )
    if h2o_application:
        execution.update(
            {
                "problem": H2O_LINEAR_FD_FAMILY,
                "adapt_pool": "full_meta_derivative_resolved_v2",
                "adapt_parallel_gradient_workers": 8,
                "allow_archival_phase3_runtime_split": False,
                "phase3_runtime_split_mode": "off",
                "phase3_runtime_split_selection_mode": "off",
                "phase3_backend_cost_mode": "proxy",
            }
        )
    if phase23_qiskit_cost_active:
        execution.update(
            {
                "phase3_backend_cost_mode": MARRAKESH_GRAPH_SPAN_MODE,
                "phase3_backend_cost_scope": (
                    BACKEND_COMPILE_SCOPE_PHASE2_PHASE3_QISKIT_ONLY_V1
                ),
                "phase3_backend_name": "FakeMarrakesh",
                "phase3_backend_optimization_level": 1,
                "phase3_backend_transpile_seed": 7,
                "adapt_parallel_gradient_workers": 4,
                "phase3_hardware_cost_normalization_mode": (
                    "zero_centered_signed_arctan_v1"
                ),
            }
        )
        if (
            macro_gradient_phase0_active
            or macro_only_gradient_phase0_qiskit_active
        ):
            execution.update(
                {
                    "ra_phase0_gradient_shortlist_policy": (
                        MACRO_PHASE0_STANDARD_ADAPT_ABS_GRADIENT_POLICY
                    ),
                    "ra_phase0_gradient_shortlist_size": (
                        RA_ADAPT_GRADIENT_PHASE0_SHORTLIST_SIZE
                    ),
                }
            )
        elif global_singleton_gradient_phase0_active:
            execution.update(
                {
                    "ra_phase0_gradient_shortlist_policy": (
                        GLOBAL_SINGLETON_ABSOLUTE_GRADIENT_PHASE0_POLICY
                    ),
                    "ra_phase0_gradient_shortlist_size": (
                        RA_ADAPT_GRADIENT_PHASE0_SHORTLIST_SIZE
                    ),
                }
            )
        if pure_hubbard_noise_application:
            adapter = request.adapter
            assert isinstance(
                adapter,
                PaperIPureHubbardNoisePage12CandidateAdapter,
            )
            execution.update(
                {
                    "problem": "hubbard",
                    "adapt_pool": "full_meta",
                    "adapt_parallel_gradient_workers": 1,
                    "allow_archival_phase3_runtime_split": False,
                    "phase3_runtime_split_mode": "off",
                    "phase3_runtime_split_selection_mode": "off",
                    "phase3_runtime_split_child_padding_policy": (
                        ROUTE_A_CHILD_PADDING_UNCHECKED_DIAGNOSTIC_V1
                    ),
                    "ra_controller_noise_contract": (
                        {
                            **pure_hubbard_noise_level_contract(
                                adapter.noise_level_id
                            ),
                            "surface": {
                                "candidate_gradient_scoring": "noisy",
                                "powell_refit_objective": "noisy",
                                "geometry_and_gram": "exact",
                                "reported_energy": "exact_diagnostic",
                            },
                        }
                    ),
                }
            )
    elif phase3_only_qiskit_cost_active:
        execution.update(
            {
                "phase3_backend_cost_mode": MARRAKESH_GRAPH_SPAN_MODE,
                "phase3_backend_cost_scope": (
                    BACKEND_COMPILE_SCOPE_PHASE3_QISKIT_ONLY_V1
                ),
                "phase3_backend_name": "FakeMarrakesh",
                "phase3_backend_optimization_level": 1,
                "phase3_backend_transpile_seed": 7,
                "adapt_parallel_gradient_workers": 4,
                "phase3_hardware_cost_normalization_mode": (
                    "family_robust_v1"
                    if phase3_qiskit_denominator_no_lanes_active
                    else "family_robust_symmetric_arctan_v1"
                ),
            }
        )
    elif qiskit_cost_active:
        execution.update(
            {
                "phase3_backend_cost_mode": "transpile_single_v1",
                "phase3_backend_name": "FakeMarrakesh",
                "phase3_backend_optimization_level": 1,
                "phase3_backend_transpile_seed": 7,
                "adapt_parallel_gradient_workers": 4,
            }
        )
    if macro_only_gradient_phase0_proxy_active:
        execution.update(
            {
                "ra_phase0_gradient_shortlist_policy": (
                    MACRO_PHASE0_STANDARD_ADAPT_ABS_GRADIENT_POLICY
                ),
                "ra_phase0_gradient_shortlist_size": (
                    RA_ADAPT_GRADIENT_PHASE0_SHORTLIST_SIZE
                ),
                "phase3_backend_cost_mode": MARRAKESH_GRAPH_SPAN_MODE,
            }
        )
        execution.pop("phase3_backend_cost_scope", None)
    if str(algorithm_identity) in {
        RA_ADAPT_MACRO_NO_LANES_ALGORITHM_ID,
        RA_ADAPT_MACRO_GRADIENT_PHASE0_PROXY_NO_LANES_ALGORITHM_ID,
        RA_ADAPT_MACRO_GRADIENT_PHASE0_MACRO_PHASE23_QISKIT_ALGORITHM_ID,
        RA_ADAPT_MACRO_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID,
        RA_ADAPT_MACRO_GRADIENT_PHASE0_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID,
        RA_ADAPT_GLOBAL_SINGLETON_GRADIENT_PHASE0_PHASE23_QISKIT_ALGORITHM_ID,
        PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ALGORITHM_ID,
        RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_DENOMINATOR_NO_LANES_ALGORITHM_ID,
        *PAPER_I_RA_SEMANTIC_ALGORITHM_IDS,
    }:
        # Lanes-off arm of the macro always-insertion ablation. The single
        # executable change is the Phase-I shortlist population: one global
        # ranking instead of the nine physical operator lanes.
        from pipelines.static_adapt.lane_routes import (
            STATIC_LANE_ROUTE_GLOBAL_SINGLE_POPULATION,
        )

        execution["static_lane_route"] = (
            STATIC_LANE_ROUTE_GLOBAL_SINGLE_POPULATION
        )
        execution.pop("physical_lane_shortlist_aggressiveness", None)
    contract["execution_settings"] = execution
    invariants = dict(contract.get("semantic_invariants", {}))
    for retired_key in (
        "all_energy_models_infeasible_novelty_fallback_active",
        "all_energy_models_infeasible_novelty_fallback_telemetry_required",
        "all_energy_models_infeasible_novelty_fallback_policy",
    ):
        invariants.pop(retired_key, None)
    invariants.update(
        {
            "canonical_interface": "run_ra_adapt_problem_request_v1",
            "candidate_representation": representation,
            "result_candidate_representation": representation,
            "candidate_geometry_chart": EXACT_ORDERED_INSERTION_CHART,
            "phase3_solver": PROJECTED_GENERALIZED_SOLVER,
            "phase3_metric_ridge": 0.0,
            "phase3_support_projection_active": True,
            "phase3_supported_whitening_active": False,
            "phase3_supported_metric_inverse_sqrt_active": False,
            "phase3_metric_ridge_active": False,
            "phase3_whitening_active": False,
            "phase3_inverse_sqrt_constructed": False,
            "trust_policy": resolved_trust_policy_id,
            "endpoint_overlap_required": endpoint_overlap_trust_active,
            "endpoint_overlap_measurement_active": (
                endpoint_overlap_trust_active
            ),
            "endpoint_overlap_query_charge_required": (
                1 if endpoint_overlap_trust_active else 0
            ),
            "adaptive_trust_policy": resolved_runtime_trust_policy,
            "adaptive_trust_predicted_displacement": (
                "phase3_joint_predicted_fubini_study_displacement_v1"
                if endpoint_overlap_trust_active
                else "phase3_joint_step_source_supported_gram_norm_v1"
            ),
            "adaptive_trust_realized_displacement": (
                "exact_post_refit_endpoint_fubini_study_displacement_v1"
                if endpoint_overlap_trust_active
                else "post_refit_parameter_step_source_supported_gram_norm_v1"
            ),
            "adaptive_trust_radius_exponent": (
                0.5 if endpoint_overlap_trust_active else -0.5
            ),
            "accepted_refit_scope": FULL_ENLARGED_ACCEPTED_REFIT,
            "accepted_refit_coordinate_chart": (
                SUPPORTED_FS_WHITENED_REFIT_CHART
            ),
            "accepted_refit_base_chart_policy": (
                EXPANDED_RUNTIME_PROJECTED_LOGICAL_BASE_CHART
            ),
            "active_gradient_policy": str(active_gradient_policy),
            "resource_weighting_scope": str(resource_weighting_scope),
            "selector_identity": RA_STAGED_SELECTOR_ID,
            "deferred_gram_fallback_receipt_schema": (
                DEFERRED_GRAM_ALL_MODELS_INFEASIBLE_FALLBACK_V1
            ),
        }
    )
    if canonical_full_response_v2:
        invariants.update(
            {
                "phase3_candidate_gain_policy": (
                    PHASE3_CANDIDATE_GAIN_JOINT_MINUS_ACTIVE_ONLY_V1
                ),
                "phase3_candidate_gain_semantics": (
                    "full_joint_minus_candidate_independent_active_only_v1"
                ),
                "phase3_active_only_baseline_solver": (
                    "same_supported_projected_generalized_solver_support_"
                    "tolerances_and_trust_radius_v1"
                ),
                "phase3_active_only_baseline_quantum_query_charge": 0,
                "accepted_refit_initialization_policy": (
                    ROUTE_A_JOINT_STEP_WARM_START_EXACT_GUARDED_V1
                ),
                "accepted_refit_initialization_coordinate_scope": (
                    "full_existing_active_plus_new_batch_coordinates_v1"
                ),
                "accepted_refit_initialization_map": (
                    "phase3_order_to_post_logical_to_fixed_supported_fs_v1"
                ),
                "accepted_refit_initialization_exact_guard": True,
                "accepted_refit_initialization_authority": (
                    "initialization_only_powell_refit_authoritative_v1"
                ),
            }
        )
    if phase3_population_controlled_ablation:
        invariants.update(
            {
                "phase1_activation_scope": "all_controller_rounds_v1",
                "phase2_activation_scope": "all_controller_rounds_v1",
                "phase3_competitive_population_activation": (
                    RA_ADAPT_PHASE3_POPULATION_LATCHED_ON_PROGRESS_PLATEAU
                    if latched_phase3_separate_insertion_ablation
                    else RA_ADAPT_PHASE3_POPULATION_ON_INSERTION_PLATEAU
                ),
                "phase3_activation_source": (
                    "first_authenticated_progress_plateau_domain_open_latched_v1"
                    if latched_phase3_separate_insertion_ablation
                    else "same_round_authenticated_insertion_plateau_domain_open_v1"
                ),
                "phase3_preplateau_admission_authority": (
                    "phase2_raw_score_top_rank_v1"
                ),
                "phase3_preplateau_materialization_policy": (
                    RA_ADAPT_PHASE3_PREPLATEAU_WINNER_MATERIALIZATION
                ),
                "phase3_activation_independent_latch": (
                    latched_phase3_separate_insertion_ablation
                ),
                "phase3_activation_hysteresis_active": False,
            }
        )
    if latched_phase3_separate_insertion_ablation:
        invariants.update(
            {
                "phase3_latch_retirement_policy": "never_close_v1",
                "insertion_plateau_history_scope": (
                    RA_ADAPT_POST_LATCH_INSERTION_TRIGGER_SCOPE
                ),
                "insertion_activation_requires_prior_phase3_latch": True,
                "insertion_activation_changes_phase3_latch": False,
            }
        )
    if h2o_application:
        h2o_sector_blocks = isinstance(
            request.adapter,
            H2OLinearFDSectorCompletePauliBlockCandidateAdapter,
        )
        for retired_key in (
            "child_padding_policy",
            "child_padding_projection_required",
            "child_padding_reason",
        ):
            invariants.pop(retired_key, None)
        invariants.update(
            {
                "application_lane": (
                    H2O_RA_SECTOR_COMPLETE_PAULI_BLOCK_APPLICATION_LANE
                    if h2o_sector_blocks
                    else H2O_RA_MACRO_APPLICATION_LANE
                ),
                "problem_family": H2O_LINEAR_FD_FAMILY,
                "operator_pool_identity": "full_meta_derivative_resolved_v2",
                "singleton_admission": True,
                "staged_singleton_exposure": False,
                "staged_sector_complete_pauli_block_exposure": (
                    h2o_sector_blocks
                ),
                "candidate_generator_semantics": (
                    "sector_complete_pauli_block_v1"
                    if h2o_sector_blocks
                    else "symmetry_complete_derivative_resolved_v1"
                ),
                "raw_single_pauli_child_exposure": False,
                "guarded_single_pauli_child_exposure": False,
                "sector_complete_pauli_block_exposure": h2o_sector_blocks,
                "normal_mode_count": 3,
            }
        )
    if global_singleton_supply is not None:
        invariants.update(global_singleton_supply)
    if pure_hubbard_noise_application:
        # The shared Page-12 parent is calibrated to the later prior-mean v2
        # trigger.  This named noise application deliberately reuses the
        # original cumulative-relative v1 trigger, so remove every inherited
        # v2-only statement before binding the replacement semantics.
        for inherited_v2_key in (
            "plateau_prior_mean_decrease_ratio_threshold",
            "plateau_threshold_comparison",
            "plateau_trigger_source",
            "plateau_threshold_calibration_status",
        ):
            invariants.pop(inherited_v2_key, None)
        invariants.update(
            {
                "application_lane": (
                    "paper_i_pure_hubbard_page12_full_noise_v1"
                ),
                "problem_family": "hubbard",
                "controller_noise_active": True,
                "controller_noise_candidate_gradient_scoring": "noisy",
                "controller_noise_powell_refit_objective": "noisy",
                "controller_noise_geometry_and_gram": "exact",
                "reported_energy_semantics": "exact_diagnostic",
                "value_noise_iid_not_frozen_keyed": True,
                "optimizer_evaluation_order": "serial_v1",
                "experimental_insertion_policy": (
                    "insertion_commutation_plateau_v1"
                ),
                "plateau_progress_statistic": (
                    "marginal_to_prior_cumulative_energy_decrease_v1"
                ),
                "plateau_cumulative_decrease_ratio_threshold": 1.0e-4,
                "plateau_threshold_comparison": (
                    "marginal_to_prior_cumulative_strictly_below_v1"
                ),
                "plateau_trigger_source": (
                    "immediately_preceding_marginal_over_prior_cumulative_"
                    "accepted_post_full_refit_energy_decrease_v1"
                ),
                "plateau_threshold_calibration_status": (
                    "source_locked_completed_trajectory_replay_v1"
                ),
                "plateau_energy_source": (
                    "persisted_noisy_controller_energy_before_after_v1"
                ),
            }
        )
    if phase23_qiskit_cost_active:
        candidate_funnel_order = (
            "macro_gradient_phase0_shortlist_then_guarded_singleton_"
            "phase1_shortlist_then_singleton_phase2_then_singleton_phase3_v1"
            if macro_gradient_phase0_active
            else "macro_gradient_phase0_shortlist_then_macro_phase1_then_"
            "identity_macro_phase2_then_macro_phase3_v1"
            if macro_only_gradient_phase0_qiskit_active
            else "global_singleton_gradient_phase0_shortlist_then_singleton_"
            "phase1_shortlist_then_singleton_phase2_then_singleton_phase3_v1"
            if global_singleton_gradient_phase0_active
            else "macro_phase1_shortlist_then_guarded_singleton_"
            "phase1_shortlist_then_singleton_phase2_then_singleton_phase3_v1"
        )
        invariants.update(
            {
                "selector_compile_cost_policy": (
                    RA_ADAPT_PHASE23_QISKIT_COST_POLICY
                ),
                "selector_compile_cost_scope": (
                    BACKEND_COMPILE_SCOPE_PHASE2_PHASE3_QISKIT_ONLY_V1
                ),
                "selector_compile_cost_phase_reuse": (
                    RA_ADAPT_PHASE23_QISKIT_COST_PHASE_REUSE
                ),
                "phase_i_compile_cost_source": "structural_proxy_v1",
                "phase_ii_compile_cost_source": "backend_transpile_v1",
                "phase_iii_compile_cost_source": "backend_transpile_v1",
                "phase_ii_phase_iii_qiskit_negative_delta_reward_enabled": (
                    True
                ),
                "phase_ii_phase_iii_qiskit_backend_fallback_allowed": False,
                "phase_ii_phase_iii_qiskit_structure_theta_value": 1.0,
                "phase_ii_phase_iii_qiskit_full_base_trial_ansatz_transpile": (
                    True
                ),
                "phase_ii_phase_iii_qiskit_population_normalization_policy": (
                    "zero_centered_signed_arctan_v1"
                ),
                "candidate_funnel_order": candidate_funnel_order,
            }
        )
        if isinstance(
            request.adapter,
            MacroThenSingletonPhaseICandidateAdapter,
        ):
            invariants["post_exposure_singleton_phase_i_policy"] = str(
                request.adapter.post_exposure_phase_i_shortlist_id
            )
        if (
            macro_gradient_phase0_active
            or macro_only_gradient_phase0_qiskit_active
            or global_singleton_gradient_phase0_active
        ):
            invariants.update(
                {
                    "phase0_active": True,
                    "phase0_score": "standard_adapt_absolute_gradient_v1",
                    "phase0_fubini_metric_active": False,
                    "phase0_resource_cost_active": False,
                    "phase0_compile_cost_active": False,
                    "phase0_shortlist_size": (
                        RA_ADAPT_GRADIENT_PHASE0_SHORTLIST_SIZE
                    ),
                    "phase0_estimator_components": ["N_grad"],
                }
            )
    elif phase3_only_qiskit_cost_active:
        invariants.update(
            {
                "selector_compile_cost_policy": (
                    RA_ADAPT_PHASE3_QISKIT_DENOMINATOR_POLICY
                    if phase3_qiskit_denominator_no_lanes_active
                    else RA_ADAPT_PHASE3_QISKIT_COST_POLICY
                ),
                "selector_compile_cost_phase_reuse": (
                    RA_ADAPT_PHASE3_QISKIT_COST_PHASE_REUSE
                ),
                "selector_compile_cost_scope": (
                    BACKEND_COMPILE_SCOPE_PHASE3_QISKIT_ONLY_V1
                ),
                "phase_i_phase_ii_compile_cost_source": (
                    MARRAKESH_GRAPH_SPAN_MODE
                ),
                "phase_iii_compile_cost_source": "backend_transpile_v1",
                "phase_iii_qiskit_backend_fallback_allowed": False,
                "phase_iii_qiskit_negative_delta_reward_enabled": False,
                "phase_iii_qiskit_raw_signed_telemetry_required": True,
                "phase_iii_qiskit_structure_theta_value": 1.0,
                "phase_iii_qiskit_full_base_trial_ansatz_transpile": True,
                "phase_iii_qiskit_independent_base_trial_layouts": True,
                "phase_iii_qiskit_one_qubit_coordinate_policy": (
                    ONE_QUBIT_COORDINATE_COMPILED_POSITIVE_DELTA_V1
                ),
                "phase_iii_qiskit_selector_circuit_coordinates": [
                    "positive_clip_delta_N2q",
                    "positive_clip_delta_D2q",
                    "positive_clip_delta_N1q",
                ],
                "phase_iii_qiskit_population_rescore_policy": (
                    "complete_evaluated_phase3_population_before_ranking_v1"
                ),
                "phase_iii_qiskit_population_normalization_policy": (
                    "family_robust_v1"
                    if phase3_qiskit_denominator_no_lanes_active
                    else "family_robust_symmetric_arctan_v1"
                ),
                "phase_iii_qiskit_failure_policy": "abort_run_v1",
            }
        )
        if phase3_qiskit_denominator_no_lanes_active:
            invariants.update(
                {
                    "phase_iii_score_formula": (
                        "B3/(1+lambda_2q*cbar_2q+lambda_d*cbar_d+"
                        "lambda_1q*cbar_1q)"
                    ),
                    "phase_iii_qiskit_theta_and_shot_lambdas": {
                        "theta": 0.0,
                        "shot": 0.0,
                    },
                    "plateau_prior_mean_decrease_ratio_threshold": 1.0e-6,
                    "plateau_threshold_calibration_status": (
                        "weak_weak_counterfactual_confirmation_tau1em6_"
                        "20260806"
                    ),
                }
            )
    elif qiskit_cost_active:
        invariants.update(
            {
                "selector_compile_cost_policy": (
                    RA_ADAPT_QISKIT_COST_POLICY
                ),
                "selector_compile_cost_phase_reuse": (
                    RA_ADAPT_QISKIT_COST_PHASE_REUSE
                ),
            }
        )
    if macro_only_gradient_phase0_active:
        invariants.update(
            {
                "phase0_active": True,
                "phase0_score": "standard_adapt_absolute_gradient_v1",
                "phase0_fubini_metric_active": False,
                "phase0_resource_cost_active": False,
                "phase0_compile_cost_active": False,
                "phase0_shortlist_size": (
                    RA_ADAPT_GRADIENT_PHASE0_SHORTLIST_SIZE
                ),
                "phase0_estimator_components": ["N_grad"],
                "candidate_funnel_order": (
                    "macro_gradient_phase0_shortlist_then_macro_phase1_"
                    "then_identity_macro_phase2_then_macro_phase3_v1"
                ),
                "selector_qiskit_compile_cost_active": bool(
                    macro_only_gradient_phase0_qiskit_active
                ),
                "macro_generator_identity_preserved_all_phases": True,
                "singleton_child_exposure_active": False,
            }
        )
        if macro_only_gradient_phase0_proxy_active:
            invariants["phase_i_phase_ii_phase_iii_cost_source"] = (
                "marrakesh_graph_span_structural_proxy_v1"
            )
    if str(algorithm_identity) in {
        RA_ADAPT_MACRO_NO_LANES_ALGORITHM_ID,
        RA_ADAPT_MACRO_GRADIENT_PHASE0_PROXY_NO_LANES_ALGORITHM_ID,
        RA_ADAPT_MACRO_GRADIENT_PHASE0_MACRO_PHASE23_QISKIT_ALGORITHM_ID,
        RA_ADAPT_MACRO_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID,
        RA_ADAPT_MACRO_GRADIENT_PHASE0_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID,
        RA_ADAPT_GLOBAL_SINGLETON_GRADIENT_PHASE0_PHASE23_QISKIT_ALGORITHM_ID,
        PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ALGORITHM_ID,
        RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_DENOMINATOR_NO_LANES_ALGORITHM_ID,
        *PAPER_I_RA_SEMANTIC_ALGORITHM_IDS,
    }:
        invariants.update(
            {
                "physical_operator_lanes_active": False,
                "shortlist_population_policy": (
                    "single_global_population_v1"
                ),
            }
        )
        invariants.pop("physical_lane_shortlist_aggressiveness", None)
    contract["semantic_invariants"] = invariants
    only_intended_scientific_changes = [
        "exact_ordered_insertion_geometry_at_recorded_position",
        "projected_generalized_raw_gram_phase3_solve",
        "source_gram_no_endpoint_overlap_trust",
        "bundle_locked_active_gradient_policy",
        "bundle_locked_resource_weighting_scope",
    ]
    if canonical_full_response_v2:
        only_intended_scientific_changes.extend(
            [
                "measured_nonstationary_active_response",
                "candidate_gain_subtracts_active_only_supported_trust_baseline",
                "guarded_full_existing_plus_new_coordinate_refit_seed",
            ]
        )
    if macro_only_gradient_phase0_active:
        only_intended_scientific_changes.extend(
            [
                "standard_adapt_absolute_gradient_macro_phase0_cap24",
                "retained_macro_identity_preserved_phase1_phase2_phase3",
                "single_global_shortlist_population_no_physical_lanes",
                (
                    "phase_i_structural_proxy_then_qiskit_phase2_phase3"
                    if macro_only_gradient_phase0_qiskit_active
                    else "structural_graph_span_proxy_without_qiskit_"
                    "selector_compile"
                ),
            ]
        )
    if phase3_population_plateau_ablation:
        only_intended_scientific_changes.append(
            "phase3_competitive_population_activates_on_same_round_"
            "insertion_plateau"
        )
    if latched_phase3_separate_insertion_ablation:
        only_intended_scientific_changes.extend(
            [
                "phase3_competitive_population_first_open_progress_"
                "plateau_latched",
                "commutation_reduced_insertion_requires_prior_full_"
                "phase3_plateau_transition",
            ]
        )
    if h2o_application:
        only_intended_scientific_changes.extend(
            [
                "h2o_linear_fd_problem_application",
                "derivative_resolved_h2o_parent_pool",
                (
                    "staged_sector_complete_pauli_block_exposure"
                    if isinstance(
                        request.adapter,
                        H2OLinearFDSectorCompletePauliBlockCandidateAdapter,
                    )
                    else "singleton_admission_of_symmetry_complete_generators"
                ),
            ]
        )
    if global_singleton_supply is not None:
        only_intended_scientific_changes.extend(
            [
                "global_guarded_singleton_phase_i_candidate_supply",
                "identity_preserving_phase_ii_singleton_exposure",
            ]
        )
    if pure_hubbard_noise_application:
        only_intended_scientific_changes.extend(
            [
                "pure_hubbard_l2_named_application",
                "fixed_full_noise_candidate_gradient_scoring",
                "fixed_full_noise_serial_powell_refit_objective",
                "exact_geometry_and_gram_retained",
                "exact_diagnostic_energy_separated_from_controller_energy",
                "iid_value_noise_rng_checkpoint_restore",
            ]
        )
    if phase23_qiskit_cost_active:
        only_intended_scientific_changes.extend(
            [
                "qiskit_full_trial_ansatz_signed_delta_in_phase2_phase3",
                "phase_i_phase_ii_phase_iii_single_global_shortlist_population",
            ]
        )
        if macro_gradient_phase0_active:
            only_intended_scientific_changes.extend(
                [
                    "standard_adapt_absolute_gradient_macro_phase0",
                    "macro_phase0_shortlist_before_guarded_singleton_exposure",
                    "fresh_singleton_phase1_shortlist_before_phase2",
                    "phase0_omits_fubini_metric_and_resource_cost",
                ]
            )
        elif macro_only_gradient_phase0_qiskit_active:
            only_intended_scientific_changes.extend(
                [
                    "standard_adapt_absolute_gradient_macro_phase0",
                    "intact_macro_identity_phase1_phase2_phase3",
                    "phase0_omits_fubini_metric_and_resource_cost",
                ]
            )
        elif global_singleton_gradient_phase0_active:
            only_intended_scientific_changes.extend(
                [
                    "initialized_global_singleton_standard_adapt_gradient_phase0",
                    "phase0_shortlist_before_singleton_phase1",
                    "phase0_omits_fubini_metric_and_resource_cost",
                ]
            )
        else:
            only_intended_scientific_changes.extend(
                [
                    "macro_phase1_shortlist_before_guarded_singleton_exposure",
                    "fresh_singleton_phase1_shortlist_before_phase2",
                ]
            )
    elif phase3_only_qiskit_cost_active:
        only_intended_scientific_changes.append(
            "phase3_selector_cost_graph_span_to_qiskit_positive_clipped_"
            "marginal_transpile"
        )
        if phase3_qiskit_denominator_no_lanes_active:
            only_intended_scientific_changes.extend(
                [
                    "phase3_qiskit_marginals_enter_literal_paper_i_"
                    "denominator",
                    "phase_i_phase_ii_single_global_shortlist_population",
                    "plateau_prior_mean_ratio_threshold_tau1em6",
                ]
            )
    elif qiskit_cost_active:
        only_intended_scientific_changes.append(
            "qiskit_full_trial_ansatz_delta_all_phases"
        )
    lineage_parent_profile = str(parent_contract["route_profile"])
    lineage_parent_digest = str(parent_digest)
    if phase3_only_qiskit_cost_active:
        (
            _source_request_profile,
            lineage_parent_profile,
            _source_contract,
            lineage_parent_digest,
        ) = _repaired_route_contract(
            request,
            active_gradient_policy=active_gradient_policy,
            resource_weighting_scope=resource_weighting_scope,
            algorithm_id=(
                RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_ALGORITHM_ID
                if phase3_qiskit_denominator_no_lanes_active
                else RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_SOURCE_ALGORITHM_ID
            ),
            problem=problem,
        )
    elif latched_phase3_separate_insertion_ablation:
        (
            _source_request_profile,
            lineage_parent_profile,
            _source_contract,
            lineage_parent_digest,
        ) = _repaired_route_contract(
            request,
            active_gradient_policy=active_gradient_policy,
            resource_weighting_scope=resource_weighting_scope,
            algorithm_id=RA_ADAPT_SINGLETON_PHASE3_PLATEAU_ALGORITHM_ID,
            problem=problem,
        )
    contract["lineage_authority"] = {
        "parent_route_profile": lineage_parent_profile,
        "parent_contract_sha256": lineage_parent_digest,
        "supersession_reason": (
            (
                "paper_iv_h2o_ra_adapt_nonstationary_full_response_v2_"
                "20260731"
            )
            if h2o_application and canonical_full_response_v2
            else "paper_i_ra_adapt_nonstationary_full_response_v2_20260731"
            if canonical_full_response_v2
            else (
                "paper_i_ra_adapt_macro_gradient_phase0_then_singleton_"
                "phase123_phase23_qiskit_candidate_20260807"
                if macro_gradient_phase0_active
                else "paper_i_ra_adapt_macro_gradient_phase0_macro_phase123_"
                "phase23_qiskit_candidate_20260811"
                if macro_only_gradient_phase0_qiskit_active
                else "paper_i_ra_adapt_global_singleton_gradient_phase0_"
                "phase123_phase23_qiskit_candidate_20260807"
                if global_singleton_gradient_phase0_active
                else "paper_i_ra_adapt_macro_then_singleton_phase123_"
                "phase23_qiskit_candidate_20260807"
            )
            if phase23_qiskit_cost_active
            else (
                "paper_i_ra_adapt_phase3_qiskit_denominator_no_lanes_"
                "tau1em6_candidate_20260806"
                if phase3_qiskit_denominator_no_lanes_active
                else "paper_i_ra_adapt_phase3_only_qiskit_cost_candidate_20260806"
            )
            if phase3_only_qiskit_cost_active
            else (
                "paper_i_ra_adapt_macro_gradient_phase0_proxy_no_lanes_"
                "candidate_20260810"
            )
            if macro_only_gradient_phase0_active
            else "paper_i_ra_adapt_singleton_phase3_plateau_ablation_20260802"
            if phase3_population_plateau_ablation
            else (
                "paper_i_ra_adapt_singleton_latched_phase3_separate_"
                "plateau_insertion_20260804"
            )
            if latched_phase3_separate_insertion_ablation
            else "paper_i_ra_adapt_unification_repair_20260727"
        ),
        "only_intended_scientific_changes": (
            only_intended_scientific_changes
        ),
        "scientific_result_anchor_claimed": False,
    }
    if semantic_closure_active:
        return build_semantic_closure_route_contract(
            request,
            algorithm_id=algorithm_identity,
            active_gradient_policy=str(active_gradient_policy),
            resource_weighting_scope=str(resource_weighting_scope),
            parent_contract=contract,
            parent_contract_sha256=_route_sha256(contract),
        )
    normalized = json.loads(json.dumps(contract, sort_keys=True))
    return profile, profile, normalized, _route_sha256(normalized)


def _ordinary_bundle_digest() -> str:
    return canonical_sha256(
        {
            "schema": "ordinary_ra_adapt_facade_authority_v2",
            "bundle_id": RA_ADAPT_ORDINARY_BUNDLE_ID,
            "algorithm_id": RA_ADAPT_ALGORITHM_ID,
            "active_gradient_policy": ACTIVE_GRADIENT_MEASURED,
            "phase3_candidate_gain_policy": (
                PHASE3_CANDIDATE_GAIN_JOINT_MINUS_ACTIVE_ONLY_V1
            ),
            "accepted_refit_initialization_policy": (
                ROUTE_A_JOINT_STEP_WARM_START_EXACT_GUARDED_V1
            ),
        }
    )


def _legacy_ordinary_bundle_digest() -> str:
    return canonical_sha256(
        {
            "schema": "ordinary_ra_adapt_facade_authority_v1",
            "bundle_id": RA_ADAPT_LEGACY_ORDINARY_BUNDLE_ID,
        }
    )


def build_resolved_ra_protocol(
    problem: ResolvedProblemContext,
    request: RAAdaptRequest,
    *,
    materialization_authority: (
        BundleProtocolMaterializationAuthority | None
    ) = None,
) -> ResolvedRAAdaptProtocol:
    """Resolve and digest one immutable RA protocol.

    Bundle materialization supplies one private typed authority rather than
    individual study-policy or provenance knobs.  The public request has no
    corresponding fields.
    """

    if not isinstance(request, RAAdaptRequest):
        raise TypeError("request must be RAAdaptRequest.")
    semantic_adapter = (
        request.adapter
        if isinstance(
            request.adapter,
            PaperIRASemanticClosureGlobalSingletonCandidateAdapter,
        )
        else None
    )
    if materialization_authority is None:
        active_gradient_policy = (
            ACTIVE_GRADIENT_STATIONARY
            if semantic_adapter is not None
            else ACTIVE_GRADIENT_MEASURED
        )
        resource_weighting_scope = RESOURCE_WEIGHTING_ALL_PHASE
        algorithm_id = (
            semantic_adapter.algorithm_id
            if semantic_adapter is not None
            else RA_ADAPT_ALGORITHM_ID
        )
        bundle_id = (
            semantic_closure_native_bundle_id(
                semantic_adapter.route_variant
            )
            if semantic_adapter is not None
            else RA_ADAPT_ORDINARY_BUNDLE_ID
        )
        bundle_manifest_sha256 = (
            semantic_closure_native_bundle_digest(
                semantic_adapter.route_variant
            )
            if semantic_adapter is not None
            else _ordinary_bundle_digest()
        )
        source_locks: Mapping[str, str] = {}
        materialization_receipt = None
    else:
        if not isinstance(
            materialization_authority,
            BundleProtocolMaterializationAuthority,
        ):
            raise TypeError(
                "materialization_authority must be minted by "
                "ra_adapt.bundles."
            )
        materialization_receipt = materialization_authority.receipt
        if materialization_receipt.protocol_schema not in (
            RA_ADAPT_PROTOCOL_SCHEMAS
        ):
            raise ValueError(
                "RA protocol received an authority for another protocol "
                "schema."
            )
        if (
            materialization_receipt.candidate_representation
            != str(request.adapter.candidate_representation_id)
            or materialization_receipt.selector_identity
            != RA_STAGED_SELECTOR_ID
        ):
            raise ValueError(
                "RA materialization authority does not match the request."
            )
        active_gradient_policy = (
            materialization_receipt.active_gradient_policy
        )
        resource_weighting_scope = (
            materialization_receipt.resource_weighting_scope
        )
        algorithm_id = materialization_receipt.algorithm_id
        if (
            semantic_adapter is None
            and str(algorithm_id) in PAPER_I_RA_SEMANTIC_ALGORITHM_IDS
        ) or (
            semantic_adapter is not None
            and str(algorithm_id) != semantic_adapter.algorithm_id
        ):
            raise ValueError(
                "Semantic-closure adapter and materialized algorithm identity "
                "must match."
            )
        if semantic_adapter is not None:
            validate_semantic_closure_materialization_authority(
                problem,
                request,
                receipt=materialization_receipt,
                source_lock_refs=materialization_authority.source_lock_refs,
            )
        bundle_id = materialization_receipt.bundle_id
        bundle_manifest_sha256 = (
            materialization_receipt.bundle_manifest_sha256
        )
        source_locks = materialization_authority.source_lock_refs
    l3_page12_application = is_paper_i_l3_page12_application(
        problem,
        request,
    )
    pure_hubbard_noise_application = (
        is_paper_i_pure_hubbard_noise_page12_application(
            problem,
            request,
        )
    )
    if l3_page12_application:
        require_paper_i_l3_page12_materialization(
            problem=problem,
            request=request,
            algorithm_id=str(algorithm_id),
            active_gradient_policy=str(active_gradient_policy),
            resource_weighting_scope=str(resource_weighting_scope),
            source_locks=source_locks,
        )
    if pure_hubbard_noise_application:
        require_paper_i_pure_hubbard_noise_page12_materialization(
            problem=problem,
            request=request,
            algorithm_id=str(algorithm_id),
            active_gradient_policy=str(active_gradient_policy),
            resource_weighting_scope=str(resource_weighting_scope),
            source_locks=source_locks,
        )
    parent = request.adapter.parent_inventory(problem)
    executable = request.adapter.executable_pool(problem)
    candidate_inventory_lineage = (
        build_candidate_inventory_lineage_receipt(executable)
    )
    global_singleton_supply = _global_singleton_supply_contract(
        request.adapter
    )
    sr_request = _sr_request(request)
    horizon = int(sr_request.execution.stop.maximum_controller_rounds)
    if horizon < 1:
        raise ValueError("RA protocols require a positive controller horizon.")
    h2o_application = _h2o_application_active(problem, request)
    if h2o_application:
        h2o_parent = (
            canonical_sr_snake_h2o_derivative_resolved_paper_i_v3_contract()
        )
        parent_route = str(h2o_parent["route_profile"])
        parent_digest = (
            canonical_sr_snake_h2o_derivative_resolved_paper_i_v3_contract_sha256()
        )
    else:
        (
            _route_request,
            parent_route,
            _parent_contract,
            parent_digest,
        ) = _canonical_route_contract_for_request(sr_request)
    if (
        not h2o_application
        and
        request.adapter.candidate_representation_id
        == CANDIDATE_REPRESENTATION_MACRO
    ):
        macro_contract, macro_digest = _macro_parent_contract(
            request, algorithm_id=algorithm_id
        )
        parent_route = str(macro_contract["route_profile"])
        parent_digest = str(macro_digest)
    if str(algorithm_id) in RA_ADAPT_PHASE3_QISKIT_ALGORITHM_IDS:
        phase3_parent_algorithm_id = (
            RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_ALGORITHM_ID
            if str(algorithm_id)
            == RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_DENOMINATOR_NO_LANES_ALGORITHM_ID
            else RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_SOURCE_ALGORITHM_ID
        )
        (
            _page7_request_profile,
            parent_route,
            _page7_route_contract,
            parent_digest,
        ) = _repaired_route_contract(
            request,
            active_gradient_policy=str(active_gradient_policy),
            resource_weighting_scope=str(resource_weighting_scope),
            algorithm_id=phase3_parent_algorithm_id,
            problem=problem,
        )
    elif str(algorithm_id) == (
        RA_ADAPT_SINGLETON_LATCHED_PHASE3_ALGORITHM_ID
    ):
        (
            _page8_request_profile,
            parent_route,
            _page8_route_contract,
            parent_digest,
        ) = _repaired_route_contract(
            request,
            active_gradient_policy=str(active_gradient_policy),
            resource_weighting_scope=str(resource_weighting_scope),
            algorithm_id=RA_ADAPT_SINGLETON_PHASE3_PLATEAU_ALGORITHM_ID,
            problem=problem,
        )
    lineage_authority: dict[str, Any] = {
        "parent_route_profile": str(parent_route),
        "parent_contract_sha256": str(parent_digest),
        "candidate_inventory_lineage": (
            candidate_inventory_lineage.authority_binding()
        ),
    }
    if global_singleton_supply is not None:
        lineage_authority["candidate_supply"] = dict(
            global_singleton_supply
        )
    if str(algorithm_id) == RA_ADAPT_ALGORITHM_ID:
        lineage_authority["algorithm_semantics"] = {
            "active_response": ACTIVE_GRADIENT_MEASURED,
            "candidate_gain_policy": (
                PHASE3_CANDIDATE_GAIN_JOINT_MINUS_ACTIVE_ONLY_V1
            ),
            "accepted_refit_initialization_policy": (
                ROUTE_A_JOINT_STEP_WARM_START_EXACT_GUARDED_V1
            ),
            "full_response_coordinate_scope": (
                "existing_active_plus_new_batch_v1"
            ),
        }
    protocol_schema = (
        RA_ADAPT_PROTOCOL_SCHEMA
        if semantic_adapter is not None
        and materialization_receipt is None
        else RA_ADAPT_PROTOCOL_SCHEMA_V2
        if materialization_receipt is None
        else str(materialization_receipt.protocol_schema)
    )
    resolved_trust_policy_id = (
        ENDPOINT_OVERLAP_DISPLACEMENT_TRUST
        if isinstance(
            request.method.trust_update,
            EndpointOverlapDisplacementTrust,
        )
        else SOURCE_GRAM_NO_OVERLAP_TRUST
    )
    payload: dict[str, Any] = {
        "schema": protocol_schema,
        "algorithm_id": str(algorithm_id),
        "candidate_representation": str(
            request.adapter.candidate_representation_id
        ),
        "adapter_id": str(request.adapter.adapter_id),
        "selector_identity": RA_STAGED_SELECTOR_ID,
        "active_gradient_policy": str(active_gradient_policy),
        "resource_weighting_scope": str(resource_weighting_scope),
        "derivative_chart_id": EXACT_ORDERED_INSERTION_CHART,
        "trust_policy_id": resolved_trust_policy_id,
        "phase3_solver_id": PROJECTED_GENERALIZED_SOLVER,
        "phase3_multiplier_contract": PhaseIIIMultiplierContract(),
        "accepted_refit_scope": FULL_ENLARGED_ACCEPTED_REFIT,
        "accepted_refit_coordinate_chart": SUPPORTED_FS_WHITENED_REFIT_CHART,
        "accepted_refit_base_chart_policy": (
            EXPANDED_RUNTIME_PROJECTED_LOGICAL_BASE_CHART
        ),
        "problem": ResolvedProblemReceipt.from_problem(problem),
        "parent_inventory": parent.receipt,
        "executable_pool": executable.receipt,
        "optimizer": "powell",
        "optimizer_maxiter": 200,
        "stopping_rule": sr_request.execution.stop.to_dict(),
        "horizon": horizon,
        "seeds": {"adapt": 7, "transpiler": 7},
        "estimator_accounting_convention": RA_ADAPT_ESTIMATOR_ACCOUNTING,
        "compile_identity": dict(RA_ADAPT_COMPILE_IDENTITY),
        "lineage_authority": lineage_authority,
        "source_locks": dict(source_locks),
        "bundle_id": str(bundle_id),
        "bundle_manifest_sha256": str(bundle_manifest_sha256),
        "execution_authorized": False,
        "request": request,
    }
    if str(algorithm_id) in {
        RA_ADAPT_ALGORITHM_ID,
        RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_ALGORITHM_ID,
        RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_DENOMINATOR_NO_LANES_ALGORITHM_ID,
        RA_ADAPT_MACRO_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID,
        RA_ADAPT_MACRO_GRADIENT_PHASE0_PROXY_NO_LANES_ALGORITHM_ID,
        RA_ADAPT_MACRO_GRADIENT_PHASE0_MACRO_PHASE23_QISKIT_ALGORITHM_ID,
        RA_ADAPT_MACRO_GRADIENT_PHASE0_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID,
        RA_ADAPT_GLOBAL_SINGLETON_GRADIENT_PHASE0_PHASE23_QISKIT_ALGORITHM_ID,
        PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ALGORITHM_ID,
        *PAPER_I_RA_SEMANTIC_ALGORITHM_IDS,
    }:
        (
            _bound_route_request,
            _bound_route_profile,
            bound_route_contract,
            bound_route_sha256,
        ) = _repaired_route_contract(
            request,
            active_gradient_policy=str(active_gradient_policy),
            resource_weighting_scope=str(resource_weighting_scope),
            algorithm_id=str(algorithm_id),
            problem=problem,
        )
        payload["route_contract"] = {
            **bound_route_contract,
            "sha256": str(bound_route_sha256),
        }
    if materialization_receipt is not None:
        payload["bundle_materialization"] = materialization_receipt
    digest = canonical_sha256(payload)
    return ResolvedRAAdaptProtocol(
        **payload,
        sha256=digest,
        _materialization_authority=materialization_authority,
    )


def _validate_resolved_pool_identity(
    problem: ResolvedProblemContext,
    protocol: ResolvedRAAdaptProtocol,
) -> tuple[CandidateInventoryLineageReceipt, CandidateInventory]:
    request = protocol.request
    if not isinstance(request, RAAdaptRequest):
        raise TypeError("Resolved RA protocol lost its typed request.")
    if int(request.execution.stop.maximum_controller_rounds) != int(
        protocol.horizon
    ):
        raise ValueError("Resolved RA protocol horizon drifted.")
    parent_inventory = request.adapter.parent_inventory(problem)
    executable_inventory = request.adapter.executable_pool(problem)
    parent = parent_inventory.receipt
    executable = executable_inventory.receipt
    for name, observed, expected in (
        ("parent", parent, protocol.parent_inventory),
        ("executable", executable, protocol.executable_pool),
    ):
        if observed != expected:
            raise ValueError(f"Resolved RA {name} pool identity drifted.")
    lineage = build_candidate_inventory_lineage_receipt(
        executable_inventory
    )
    if protocol.lineage_authority.get(
        "candidate_inventory_lineage"
    ) != lineage.authority_binding():
        raise ValueError(
            "Resolved RA candidate inventory lineage drifted."
        )
    expected_supply = _global_singleton_supply_contract(request.adapter)
    observed_supply = protocol.lineage_authority.get("candidate_supply")
    if (
        expected_supply is not None
        and observed_supply != expected_supply
    ):
        raise ValueError(
            "Resolved RA global-singleton candidate-supply identity drifted."
        )
    return lineage, executable_inventory


def _find_mapping(
    value: Any,
    *,
    key: str,
) -> list[Mapping[str, Any]]:
    found: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        if key in value:
            found.append(value)
        for item in value.values():
            found.extend(_find_mapping(item, key=key))
    elif isinstance(value, (tuple, list)):
        for item in value:
            found.extend(_find_mapping(item, key=key))
    return found


def _policy_receipt(
    *,
    protocol: ResolvedRAAdaptProtocol,
    finalization: Mapping[str, Any],
) -> PolicyEchoReceipt:
    rows = _find_mapping(
        finalization, key="active_gradient_query_accounting"
    )
    accounting: Mapping[str, Any] = {}
    if rows:
        raw = rows[-1].get("active_gradient_query_accounting")
        if isinstance(raw, Mapping):
            accounting = raw
    indices_raw = accounting.get("active_gradient_indices_acquired", ())
    indices = (
        tuple(int(value) for value in indices_raw)
        if isinstance(indices_raw, (tuple, list))
        else ()
    )
    charge = int(accounting.get("new_unique_gradients_charged", 0) or 0)
    return PolicyEchoReceipt(
        active_gradient_policy=protocol.active_gradient_policy,
        resource_weighting_scope=protocol.resource_weighting_scope,
        active_gradient_indices_acquired=indices,
        active_gradient_charge=charge,
        phase3_candidate_gain_policy=(
            PHASE3_CANDIDATE_GAIN_JOINT_MINUS_ACTIVE_ONLY_V1
            if protocol.algorithm_id == RA_ADAPT_ALGORITHM_ID
            else None
        ),
        accepted_refit_initialization_policy=(
            ROUTE_A_JOINT_STEP_WARM_START_EXACT_GUARDED_V1
            if protocol.algorithm_id == RA_ADAPT_ALGORITHM_ID
            else None
        ),
    )


def _required_retained_support(
    row: Mapping[str, Any],
    *,
    trust: SourceGramNoOverlapTrustReceipt | None,
) -> RetainedSupportReceipt:
    if trust is None:
        selected_rows = row.get("selected_feature_rows")
        if (
            not isinstance(selected_rows, (tuple, list))
            or len(selected_rows) != 1
            or not isinstance(selected_rows[0], Mapping)
        ):
            raise RuntimeError(
                "Geometry-expansion trust requires exactly one selected "
                "feature row."
            )
        selected_geometry = selected_rows[0].get(
            "phase2_joint_geometry_reuse"
        )
        if not isinstance(selected_geometry, Mapping):
            raise RuntimeError(
                "Geometry-expansion trust lost its selected support model."
            )
        raw_support = selected_geometry.get("retained_support_receipt")
        if not isinstance(raw_support, Mapping):
            raise RuntimeError(
                "Geometry-expansion trust lost its retained-support receipt."
            )
        receipt = validate_retained_support_receipt(raw_support)
        if float(receipt.metric_regularization) != 0.0:
            raise RuntimeError(
                "Accepted RA selector support must be factorized without a "
                "metric ridge."
            )
        return receipt

    matches: dict[str, RetainedSupportReceipt] = {}
    for owner in _find_mapping(row, key="retained_support_receipt"):
        raw = owner.get("retained_support_receipt")
        if not isinstance(raw, Mapping):
            raise RuntimeError(
                "Accepted RA round has a malformed retained-support receipt."
            )
        receipt = validate_retained_support_receipt(raw)
        if (
            receipt.factorization_provenance_id
            == trust.support_provenance_id
        ):
            matches[receipt.receipt_provenance_id] = receipt
    if len(matches) != 1:
        raise RuntimeError(
            "Accepted RA round must carry exactly one selector-support "
            "receipt matching its trust transaction."
        )
    receipt = next(iter(matches.values()))
    if float(receipt.metric_regularization) != 0.0:
        raise RuntimeError(
            "Accepted RA selector support must be factorized without a "
            "metric ridge."
        )
    if (
        receipt.retained_mask != trust.retained_mask
        or int(receipt.rank) != int(trust.supported_rank)
    ):
        raise RuntimeError(
            "Accepted RA trust transaction disagrees with selector support."
        )
    return receipt


def _geometry_expansion_trust_limitation(
    trust_update: Mapping[str, Any],
) -> bool:
    failure = trust_update.get("source_metric_trust_transaction_failure")
    if failure != GEOMETRY_EXPANSION_SOURCE_METRIC_LIMITATION:
        return False
    endpoint_overlap_query_charge = trust_update.get(
        "endpoint_overlap_query_charge"
    )
    if (
        trust_update.get("source_metric_trust_transaction") is not None
        or trust_update.get("geometry_expansion_active") is not True
        or str(trust_update.get("context_mode", ""))
        != HISTORICAL_SINGLETON_GEOMETRY_EXPANSION_CONTEXT_V1
        or str(trust_update.get("policy", ""))
        != ROUTE_A_TRUST_REGION_SOURCE_METRIC_INVERSE_SQRT_NO_OVERLAP_V1
        or str(trust_update.get("update_reason", ""))
        != GEOMETRY_EXPANSION_NO_OVERLAP_HOLD_REASON
        or trust_update.get("scalar_or_unwhitened_fallback_used") is not False
        or str(trust_update.get("model_agreement_authority", ""))
        != "unavailable_without_coordinate_prediction"
        or trust_update.get("endpoint_overlap_measurement_required")
        is not False
        or trust_update.get("endpoint_overlap_measurement_performed")
        is not False
        or isinstance(endpoint_overlap_query_charge, bool)
        or not isinstance(endpoint_overlap_query_charge, int)
        or endpoint_overlap_query_charge != 0
    ):
        raise RuntimeError(
            "Accepted RA geometry-expansion trust limitation is malformed."
        )
    return True


def _required_phase3_stabilization(
    row: Mapping[str, Any],
    *,
    support: RetainedSupportReceipt,
) -> PhaseIIIStabilizationReceipt | None:
    """Project stabilization from the accepted winner, not candidate telemetry.

    The plateau-gated singleton producer materializes its accepted Phase-II
    winner before the plateau opens and its accepted Phase-III winner after it
    opens in ``selected_feature_rows``.  The trust update may echo that same
    stabilization receipt.  Other recursively nested summaries can describe
    candidates that were scored but not admitted, so they are not evidence for
    the accepted round.
    """

    candidates: list[tuple[str, Mapping[str, Any]]] = []
    selected_rows = row.get("selected_feature_rows")
    if isinstance(selected_rows, (tuple, list)):
        for selected in selected_rows:
            if not isinstance(selected, Mapping):
                continue
            winner_geometry = selected.get("phase2_joint_geometry_reuse")
            if isinstance(winner_geometry, Mapping):
                candidates.append(("selected_geometry", winner_geometry))
    trust_update = row.get("route_a_trust_region_update")
    if isinstance(trust_update, Mapping):
        accepted_path_echo = trust_update.get(
            "phase3_stabilization_receipt"
        )
        if isinstance(accepted_path_echo, Mapping):
            candidates.append(("trust_echo", accepted_path_echo))

    matches: list[PhaseIIIStabilizationReceipt] = []
    missing_receipt_geometries: list[Mapping[str, Any]] = []
    seen: set[tuple[float, float, float, bool]] = set()
    quartet = (
        "kappa_stabilization_shift",
        "trust_boundary_multiplier_lambda",
        "total_metric_multiplier_mu",
        "trust_boundary_active",
    )
    for source, candidate in candidates:
        if (
            str(
                candidate.get(
                    "joint_linear_solve_policy_effective",
                    "",
                )
            )
            != PROJECTED_GENERALIZED_SOLVER
            or str(
                candidate.get(
                    "supported_metric_projection_provenance_id",
                    "",
                )
            )
            != support.factorization_provenance_id
        ):
            continue
        quartet_presence = tuple(field in candidate for field in quartet)
        if any(quartet_presence) and not all(quartet_presence):
            raise RuntimeError(
                "Accepted RA Phase-III stabilization receipt is incomplete."
            )
        if not any(quartet_presence):
            if (
                source != "selected_geometry"
                or candidate.get("feasible") is not False
            ):
                raise RuntimeError(
                    "Accepted RA Phase-III stabilization receipt is incomplete."
                )
            missing_receipt_geometries.append(candidate)
            continue
        try:
            key = (
                float(candidate["kappa_stabilization_shift"]),
                float(candidate["trust_boundary_multiplier_lambda"]),
                float(candidate["total_metric_multiplier_mu"]),
                bool(candidate["trust_boundary_active"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Accepted RA Phase-III stabilization receipt is incomplete."
            ) from exc
        if key in seen:
            continue
        seen.add(key)
        matches.append(
            PhaseIIIStabilizationReceipt(
                solver_policy=PROJECTED_GENERALIZED_SOLVER,
                kappa_stabilization_shift=key[0],
                trust_boundary_multiplier_lambda=key[1],
                total_metric_multiplier_mu=key[2],
                trust_boundary_active=key[3],
                metric_whitening_active=bool(
                    candidate.get(
                        "supported_metric_whitening_active",
                        True,
                    )
                ),
                metric_inverse_sqrt_constructed=bool(
                    candidate.get(
                        "supported_metric_inverse_sqrt_constructed",
                        True,
                    )
                ),
            )
        )
    if missing_receipt_geometries:
        if (
            matches
            or (
                isinstance(trust_update, Mapping)
                and "phase3_stabilization_receipt" in trust_update
            )
        ):
            raise RuntimeError(
                "Accepted RA Phase-III stabilization receipt conflicts with "
                "its geometry-expansion limitation."
            )
        if (
            len(missing_receipt_geometries) == 1
            and isinstance(trust_update, Mapping)
            and _geometry_expansion_trust_limitation(trust_update)
        ):
            return None
        raise RuntimeError(
            "Accepted RA Phase-III stabilization receipt is incomplete."
        )
    if len(matches) != 1:
        raise RuntimeError(
            "Accepted RA round must carry exactly one Phase-III "
            "stabilization receipt matching selector support."
        )
    return matches[0]


def _accepted_candidate_lineage_receipts(
    row: Mapping[str, Any],
    *,
    candidate_representation: str | None,
    executable_inventory: CandidateInventory,
) -> list[CandidateLineageReceipt]:
    """Project accepted rows through authenticated or preserved lineage."""

    raw_rows = row.get("selected_feature_rows")
    if not isinstance(raw_rows, (tuple, list)):
        raise RuntimeError(
            "Accepted RA round is missing selected candidate feature rows."
        )
    by_identity: dict[tuple[str, str], CandidateRecord] = {}
    by_label: dict[str, CandidateRecord] = {}
    for candidate in executable_inventory.candidates:
        key = (str(candidate.label), str(candidate.generator_identity))
        if key in by_identity:
            raise RuntimeError(
                "Authenticated RA executable inventory has duplicate "
                "candidate lineage identities."
            )
        by_identity[key] = candidate
        label_key = str(candidate.label)
        if label_key in by_label:
            raise RuntimeError(
                "Authenticated RA executable inventory has duplicate "
                "candidate labels."
            )
        by_label[label_key] = candidate
    receipts: list[CandidateLineageReceipt] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, Mapping):
            raise RuntimeError(
                "Accepted RA candidate lineage row is not a mapping."
            )
        metadata = raw_row.get("generator_metadata")
        if not isinstance(metadata, Mapping):
            raise RuntimeError(
                "Accepted RA candidate is missing generator metadata."
            )
        try:
            label = str(raw_row["candidate_label"])
            generator_identity = str(metadata["generator_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Accepted RA candidate lacks its executable-pool identity."
            ) from exc
        candidate = by_identity.get((label, generator_identity))
        if candidate is None:
            if (
                candidate_representation
                != CANDIDATE_REPRESENTATION_SINGLE_PAULI
            ):
                raise RuntimeError(
                    "Accepted RA candidate is absent from the authenticated "
                    "executable inventory."
                )
            adapter_representation = metadata.get(
                "ra_candidate_representation"
            )
            if adapter_representation is not None:
                raw_manifest = metadata.get("ra_candidate_manifest")
                if not isinstance(raw_manifest, Mapping):
                    raise RuntimeError(
                        "Accepted staged RA child is missing its "
                        "candidate-manifest binding."
                    )
                required_manifest_fields = {
                    "label",
                    "representation_id",
                    "generator_identity",
                    "parent_identities",
                    "family_id",
                    "stage_family",
                    "construction",
                    "execution_mode",
                    "serialized_terms_exyz",
                    "symmetry_receipt",
                }
                if set(raw_manifest) != required_manifest_fields:
                    raise RuntimeError(
                        "Accepted staged RA child has a malformed "
                        "candidate manifest."
                    )
                raw_parent_identities = raw_manifest.get(
                    "parent_identities"
                )
                raw_serialized_terms = raw_manifest.get(
                    "serialized_terms_exyz"
                )
                raw_symmetry_receipt = raw_manifest.get(
                    "symmetry_receipt"
                )
                if (
                    not isinstance(
                        raw_parent_identities, (tuple, list)
                    )
                    or not raw_parent_identities
                    or not isinstance(
                        raw_serialized_terms, (tuple, list)
                    )
                    or len(raw_serialized_terms) != 1
                    or not all(
                        isinstance(term, Mapping)
                        for term in raw_serialized_terms
                    )
                    or not isinstance(
                        raw_symmetry_receipt, Mapping
                    )
                ):
                    raise RuntimeError(
                        "Accepted staged RA child manifest has invalid "
                        "lineage, terms, or symmetry evidence."
                    )
                representation = str(
                    raw_manifest["representation_id"]
                )
                parent_identities = tuple(
                    str(value)
                    for value in raw_parent_identities
                )
                compile_metadata = metadata.get("compile_metadata")
                runtime_split = (
                    compile_metadata.get("runtime_split")
                    if isinstance(compile_metadata, Mapping)
                    else None
                )
                compiled_terms = (
                    compile_metadata.get(
                        "serialized_terms_exyz"
                    )
                    if isinstance(compile_metadata, Mapping)
                    else None
                )
                compiled_symmetry_gate = (
                    compile_metadata.get("symmetry_gate")
                    if isinstance(compile_metadata, Mapping)
                    else None
                )
                shared_contract = metadata.get(
                    "shared_pauli_pool_contract"
                )
                raw_parent_labels = (
                    shared_contract.get("parent_labels")
                    if isinstance(shared_contract, Mapping)
                    else None
                )
                if (
                    not isinstance(runtime_split, Mapping)
                    or runtime_split.get("mode")
                    != "guarded_singleton_children_only_v1"
                    or runtime_split.get("representation")
                    != "guarded_singleton_child"
                    or not isinstance(compiled_terms, (tuple, list))
                    or [
                        dict(term) for term in compiled_terms
                    ]
                    != [
                        dict(term) for term in raw_serialized_terms
                    ]
                    or not isinstance(
                        compiled_symmetry_gate, Mapping
                    )
                    or not isinstance(shared_contract, Mapping)
                    or shared_contract.get("mode")
                    != "guarded_singleton_children_only_v1"
                    or shared_contract.get("representation")
                    != "guarded_singleton_child"
                    or not isinstance(
                        raw_parent_labels, (tuple, list)
                    )
                    or not raw_parent_labels
                ):
                    raise RuntimeError(
                        "Accepted staged RA child manifest differs from "
                        "its executed guarded-child metadata."
                    )
                parent_labels = tuple(
                    str(value) for value in raw_parent_labels
                )
                if any(
                    parent_label not in by_label
                    for parent_label in parent_labels
                ):
                    raise RuntimeError(
                        "Accepted staged RA child names an "
                        "unauthenticated parent label."
                    )
                parent_identities_from_labels = tuple(
                    str(by_label[parent_label].generator_identity)
                    for parent_label in parent_labels
                )
                intrinsic_generator_identity = (
                    guarded_singleton_generator_identity(
                        label=label,
                        serialized_terms_exyz=(
                            raw_serialized_terms
                        ),
                    )
                )
                common_gate_fields = (
                    "checked",
                    "passed",
                    "particle_number_preserving",
                    "spin_sector_preserving",
                    "commutator_l1_total",
                    "commutator_l1_up",
                    "commutator_l1_dn",
                    "globally_particle_number_commuting",
                    "globally_spin_sector_commuting",
                    "gate_scope",
                    "fixed_count_sector",
                    "required_particle_number",
                    "required_spin_sector",
                )
                if (
                    parent_identities_from_labels
                    != parent_identities
                    or generator_identity
                    != intrinsic_generator_identity
                    or raw_symmetry_receipt.get(
                        "hard_guard_required"
                    )
                    is not True
                    or raw_symmetry_receipt.get(
                        "hard_guard_present"
                    )
                    is not True
                    or raw_symmetry_receipt.get("checked")
                    is not True
                    or raw_symmetry_receipt.get("passed")
                    is not True
                    or raw_symmetry_receipt.get("rejected")
                    is not False
                    or any(
                        raw_symmetry_receipt.get(field)
                        != compiled_symmetry_gate.get(field)
                        for field in common_gate_fields
                    )
                ):
                    raise RuntimeError(
                        "Accepted staged RA child failed its intrinsic "
                        "identity, parent, or hard-guard binding."
                    )
                authenticated_parent_identities = {
                    str(row.generator_identity)
                    for row in executable_inventory.candidates
                }
                if (
                    representation
                    != CANDIDATE_REPRESENTATION_SINGLE_PAULI
                    or str(raw_manifest["label"]) != label
                    or str(raw_manifest["generator_identity"])
                    != generator_identity
                    or not set(parent_identities)
                    <= authenticated_parent_identities
                ):
                    raise RuntimeError(
                        "Accepted staged RA child manifest is not bound "
                        "to the authenticated parent inventory."
                    )
                raw_metadata_parents = metadata.get(
                    "ra_parent_generator_ids"
                )
                if (
                    not isinstance(
                        raw_metadata_parents, (tuple, list)
                    )
                    or tuple(
                        str(value)
                        for value in raw_metadata_parents
                    )
                    != parent_identities
                ):
                    raise RuntimeError(
                        "Accepted staged RA child metadata and manifest "
                        "parent lineage differ."
                    )
                manifest_sha256 = canonical_sha256(
                    dict(raw_manifest)
                )
                if (
                    str(
                        metadata.get(
                            "ra_candidate_manifest_sha256", ""
                        )
                    )
                    != manifest_sha256
                ):
                    raise RuntimeError(
                        "Accepted staged RA child candidate-manifest "
                        "digest drifted."
                    )
            else:
                raw_plural_parents = metadata.get(
                    "parent_generator_ids"
                )
                if isinstance(raw_plural_parents, (tuple, list)):
                    parent_identities = tuple(
                        str(value) for value in raw_plural_parents
                    )
                else:
                    raw_parent = metadata.get("parent_generator_id")
                    parent_identities = (
                        ()
                        if raw_parent is None
                        else (str(raw_parent),)
                    )
                compile_metadata = metadata.get("compile_metadata")
                serialized_terms = (
                    compile_metadata.get("serialized_terms_exyz")
                    if isinstance(compile_metadata, Mapping)
                    else None
                )
                symmetry_receipt = metadata.get("symmetry_spec")
                if (
                    not isinstance(serialized_terms, (tuple, list))
                    or len(serialized_terms) != 1
                    or not all(
                        isinstance(term, Mapping)
                        for term in serialized_terms
                    )
                    or (
                        symmetry_receipt is not None
                        and not isinstance(
                            symmetry_receipt, Mapping
                        )
                    )
                ):
                    raise RuntimeError(
                        "Historical-compatible singleton lineage cannot be "
                        "canonicalized from its accepted execution record."
                    )
                serialized_term_rows = [
                    dict(term) for term in serialized_terms
                ]
                if not parent_identities:
                    preserved_parent = by_label.get(label)
                    if (
                        str(metadata.get("split_policy", ""))
                        != "preserve"
                        or metadata.get("is_macro_generator") is not False
                        or preserved_parent is None
                        or preserved_parent.representation_id
                        != CANDIDATE_REPRESENTATION_SINGLE_PAULI
                        or canonical_sha256(serialized_term_rows)
                        != canonical_sha256(
                            [
                                dict(term)
                                for term in (
                                    preserved_parent
                                    .serialized_terms_exyz
                                )
                            ]
                        )
                        or str(metadata.get("family_id", ""))
                        != str(preserved_parent.family_id)
                    ):
                        raise RuntimeError(
                            "Historical-compatible parentless singleton "
                            "lineage does not match the authenticated "
                            "executable inventory."
                        )
                representation = CANDIDATE_REPRESENTATION_SINGLE_PAULI
                manifest_row = {
                    "label": label,
                    "representation_id": representation,
                    "generator_identity": generator_identity,
                    "parent_identities": list(parent_identities),
                    "family_id": str(
                        metadata.get(
                            "family_id",
                            raw_row.get("candidate_family", ""),
                        )
                    ),
                    "stage_family": str(
                        raw_row.get("stage_name", "")
                    ),
                    "construction": str(
                        metadata.get("template_id", "")
                    ),
                    "execution_mode": str(
                        metadata.get("split_policy", "")
                    ),
                    "serialized_terms_exyz": serialized_term_rows,
                    "symmetry_receipt": (
                        None
                        if symmetry_receipt is None
                        else dict(symmetry_receipt)
                    ),
                }
                if any(
                    not str(manifest_row[field]).strip()
                    for field in (
                        "family_id",
                        "stage_family",
                        "construction",
                        "execution_mode",
                    )
                ):
                    raise RuntimeError(
                        "Historical-compatible singleton candidate "
                        "manifest is incomplete."
                    )
                manifest_sha256 = canonical_sha256(manifest_row)
        else:
            representation = str(candidate.representation_id)
            parent_identities = tuple(candidate.parent_identities)
            manifest_sha256 = canonical_sha256(candidate.manifest_row())
        if (
            candidate_representation is None
            or str(representation) != str(candidate_representation)
        ):
            raise RuntimeError(
                "Accepted RA candidate representation differs from its "
                "resolved protocol."
            )
        metadata_representation = metadata.get(
            "ra_candidate_representation"
        )
        if (
            metadata_representation is not None
            and str(metadata_representation) != representation
        ):
            raise RuntimeError(
                "Accepted RA adapter representation lineage differs from "
                "the authenticated executable inventory."
            )
        raw_parents = metadata.get("ra_parent_generator_ids")
        if raw_parents is not None and (
            not isinstance(raw_parents, (tuple, list))
            or tuple(str(value) for value in raw_parents)
            != tuple(str(value) for value in parent_identities)
        ):
            raise RuntimeError(
                "Accepted RA adapter parent lineage differs from the "
                "authenticated executable inventory."
            )
        metadata_manifest_sha256 = metadata.get(
            "ra_candidate_manifest_sha256"
        )
        if (
            metadata_manifest_sha256 is not None
            and str(metadata_manifest_sha256) != manifest_sha256
        ):
            raise RuntimeError(
                "Accepted RA adapter candidate-manifest digest differs from "
                "the authenticated executable inventory."
            )
        try:
            receipt = CandidateLineageReceipt(
                representation_id=representation,
                candidate_label=label,
                generator_identity=generator_identity,
                parent_identities=parent_identities,
                insertion_position=int(raw_row["position_id"]),
                candidate_manifest_sha256=manifest_sha256,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Accepted RA candidate lineage is incomplete or invalid."
            ) from exc
        receipts.append(receipt)

    expected_count = int(row.get("selected_logical_size", len(receipts)))
    if len(receipts) != expected_count:
        raise RuntimeError(
            "Accepted RA candidate-lineage cardinality differs from the "
            "admission."
        )
    raw_positions = row.get("selected_positions")
    if receipts and isinstance(raw_positions, (tuple, list)):
        if tuple(
            receipt.insertion_position for receipt in receipts
        ) != tuple(int(value) for value in raw_positions):
            raise RuntimeError(
                "Accepted RA candidate-lineage positions differ from the "
                "admission."
            )
    return receipts


def _required_accepted_refit_fixed_chart(
    row: Mapping[str, Any],
) -> Any:
    """Validate the one immutable accepted-refit chart for an admission."""

    from pipelines.static_adapt.accepted_refit import (
        ACCEPTED_REFIT_FIXED_CHART_RECEIPT_SCHEMA,
        AcceptedRefitFixedChartReceipt,
    )

    accepted_refit = row.get("accepted_refit")
    if not isinstance(accepted_refit, Mapping):
        raise RuntimeError(
            "Accepted RA round is missing its accepted-refit result."
        )
    owners = _find_mapping(
        accepted_refit,
        key="accepted_refit_fixed_chart_receipt",
    )
    if len(owners) != 1 or owners[0] is not accepted_refit:
        raise RuntimeError(
            "Accepted RA round must carry exactly one fixed refit chart."
        )
    raw_receipt = accepted_refit.get(
        "accepted_refit_fixed_chart_receipt"
    )
    raw_sha256 = accepted_refit.get(
        "accepted_refit_fixed_chart_sha256"
    )
    if not isinstance(raw_receipt, Mapping):
        raise RuntimeError(
            "Accepted RA fixed refit chart receipt is malformed."
        )
    try:
        receipt = AcceptedRefitFixedChartReceipt(
            schema=str(raw_receipt["schema"]),
            scope=str(raw_receipt["scope"]),
            coordinate_chart=str(raw_receipt["coordinate_chart"]),
            base_chart_policy=str(raw_receipt["base_chart_policy"]),
            manifold_id=str(raw_receipt["manifold_id"]),
            construction_hashes=dict(raw_receipt["construction_hashes"]),
            support_factorization_provenance_id=str(
                raw_receipt["support_factorization_provenance_id"]
            ),
            support_receipt_provenance_id=str(
                raw_receipt["support_receipt_provenance_id"]
            ),
            external_gram_receipt_id=(
                None
                if raw_receipt.get("external_gram_receipt_id") is None
                else str(raw_receipt["external_gram_receipt_id"])
            ),
            sha256=str(raw_receipt["sha256"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Accepted RA fixed refit chart receipt is invalid."
        ) from exc
    if (
        receipt.schema != ACCEPTED_REFIT_FIXED_CHART_RECEIPT_SCHEMA
        or receipt.scope != FULL_ENLARGED_ACCEPTED_REFIT
        or receipt.coordinate_chart
        != SUPPORTED_FS_WHITENED_REFIT_CHART
        or receipt.base_chart_policy
        != EXPANDED_RUNTIME_PROJECTED_LOGICAL_BASE_CHART
        or receipt.as_dict() != dict(raw_receipt)
        or str(raw_sha256) != str(receipt.sha256)
        or accepted_refit.get("chart_fixed_within_powell_invocation")
        is not True
        or accepted_refit.get("chart_recomputed_after_next_admission")
        is not True
    ):
        raise RuntimeError(
            "Accepted RA refit did not use one complete fixed supported-FS "
            "chart for exactly one optimizer invocation."
        )
    return receipt


def _phase0_reduction_validation_population(
    scored_population: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a named gradient Phase0 domain to its retained scored subset."""

    raw_phase0 = scored_population.get("phase0_gradient_screen")
    if not isinstance(raw_phase0, Mapping) or raw_phase0.get("schema") != (
        "paper_i_scored_gradient_phase0_population_v1"
    ):
        raise RuntimeError(
            "Gradient Phase0 accepted round is missing its scored-domain "
            "receipt."
        )

    def _validated_rows(
        key: str,
        count_key: str,
        digest_key: str,
    ) -> list[dict[str, Any]]:
        raw_rows = raw_phase0.get(key)
        if not isinstance(raw_rows, list) or not raw_rows or any(
            not isinstance(row, Mapping) for row in raw_rows
        ):
            raise RuntimeError(
                f"Gradient Phase0 {key} is missing or malformed."
            )
        rows = [dict(row) for row in raw_rows]
        record_keys = [
            (
                str(row.get("domain_record_id", "")),
                str(row.get("generator_id", "")),
            )
            for row in rows
        ]
        position_keys = [
            (
                int(row.get("pool_index", -1)),
                int(row.get("insertion_position", -1)),
            )
            for row in rows
        ]
        if (
            any(not all(record_key) for record_key in record_keys)
            or any(min(position_key) < 0 for position_key in position_keys)
            or len(set(record_keys)) != len(record_keys)
            or len(set(position_keys)) != len(position_keys)
            or int(raw_phase0.get(count_key, -1)) != len(rows)
            or raw_phase0.get(digest_key) != canonical_sha256(rows)
        ):
            raise RuntimeError(
                f"Gradient Phase0 {key} identity or digest drifted."
            )
        return rows

    population = _validated_rows(
        "population",
        "population_count",
        "ordered_population_sha256",
    )
    shortlist = _validated_rows(
        "shortlist",
        "shortlist_count",
        "ordered_shortlist_sha256",
    )
    population_record_keys = {
        (str(row["domain_record_id"]), str(row["generator_id"]))
        for row in population
    }
    if any(
        (str(row["domain_record_id"]), str(row["generator_id"]))
        not in population_record_keys
        for row in shortlist
    ):
        raise RuntimeError(
            "Gradient Phase0 shortlist escaped its original population."
        )

    retained_positions = {
        (int(row["pool_index"]), int(row["insertion_position"]))
        for row in shortlist
    }
    raw_phases = scored_population.get("phases")
    if not isinstance(raw_phases, list) or not raw_phases:
        raise RuntimeError(
            "Gradient Phase0 accepted round has no scored phases."
        )
    later_phases: list[dict[str, Any]] = []
    for raw_phase in raw_phases:
        if not isinstance(raw_phase, Mapping):
            raise RuntimeError(
                "Gradient Phase0 accepted round has a malformed scored phase."
            )
        raw_records = raw_phase.get("records")
        if not isinstance(raw_records, list) or not raw_records or any(
            not isinstance(row, Mapping) for row in raw_records
        ):
            raise RuntimeError(
                "Gradient Phase0 accepted round has malformed scored records."
            )
        if any(
            (
                int(row.get("pool_index", -1)),
                int(row.get("insertion_position", -1)),
            )
            not in retained_positions
            for row in raw_records
        ):
            raise RuntimeError(
                "Gradient Phase0 scored phase escaped the retained shortlist."
            )
        if raw_phase.get("phase") != "phase_i":
            later_phases.append(dict(raw_phase))

    return {
        "phases": [
            {
                "phase": "phase_i",
                "records": population,
            },
            *later_phases,
        ]
    }


def _validated_gradient_phase0_round_receipt(
    row: Mapping[str, Any],
    *,
    scored_population: Mapping[str, Any],
    algorithm_id: str,
) -> dict[str, Any]:
    """Validate and retain the named pure-gradient Phase0 evidence."""

    semantic_identity = None
    if str(algorithm_id) in PAPER_I_RA_SEMANTIC_ALGORITHM_IDS:
        semantic_identity = semantic_closure_route_identity_from_algorithm(
            str(algorithm_id)
        )
        raw = row.get("ra_gradient_phase0_shortlist")
        if not isinstance(raw, Mapping):
            raise RuntimeError(
                "Accepted round is missing semantic Phase0 evidence."
            )
        if (
            semantic_identity.route_variant
            not in PAPER_I_RA_PHASE0_EXECUTABLE_ROUTE_VARIANTS
        ):
            raise RuntimeError("The v1 semantic Phase0 route is retired.")
        if semantic_identity.route_variant in {
            PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2,
            PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1,
            PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
            PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_FORCED_K50_V1,
        }:
            return validate_semantic_gradient_adaptive_phase0_receipt(
                raw,
                scored_population=scored_population,
            )
        if semantic_identity.route_variant in PAPER_I_RA_PHASE0_POSITION_ROUTE_VARIANTS:
            return validate_semantic_position_phase0_receipt(
                raw,
                scored_population=scored_population,
            )
        if (
            semantic_identity.route_variant
            != PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2
        ):
            return validate_semantic_proxy_phase0_receipt(
                raw,
                scored_population=scored_population,
            )

    expected = {
        RA_ADAPT_MACRO_GRADIENT_PHASE0_PROXY_NO_LANES_ALGORITHM_ID: (
            "paper_i_macro_gradient_phase0_receipt_v1",
            MACRO_PHASE0_STANDARD_ADAPT_ABS_GRADIENT_POLICY,
        ),
        RA_ADAPT_MACRO_GRADIENT_PHASE0_MACRO_PHASE23_QISKIT_ALGORITHM_ID: (
            "paper_i_macro_gradient_phase0_receipt_v1",
            MACRO_PHASE0_STANDARD_ADAPT_ABS_GRADIENT_POLICY,
        ),
        RA_ADAPT_MACRO_GRADIENT_PHASE0_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID: (
            "paper_i_macro_gradient_phase0_receipt_v1",
            MACRO_PHASE0_STANDARD_ADAPT_ABS_GRADIENT_POLICY,
        ),
        RA_ADAPT_GLOBAL_SINGLETON_GRADIENT_PHASE0_PHASE23_QISKIT_ALGORITHM_ID: (
            "paper_i_global_singleton_gradient_phase0_receipt_v1",
            GLOBAL_SINGLETON_ABSOLUTE_GRADIENT_PHASE0_POLICY,
        ),
        PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ALGORITHM_ID: (
            "paper_i_global_singleton_gradient_phase0_receipt_v1",
            GLOBAL_SINGLETON_ABSOLUTE_GRADIENT_PHASE0_POLICY,
        ),
    }.get(str(algorithm_id))
    if semantic_identity is not None:
        expected = (
            "paper_i_global_singleton_gradient_phase0_receipt_v1",
            GLOBAL_SINGLETON_ABSOLUTE_GRADIENT_PHASE0_POLICY,
        )
    if expected is None:
        raise RuntimeError("Unknown gradient Phase0 algorithm identity.")
    raw = row.get("ra_gradient_phase0_shortlist")
    if not isinstance(raw, Mapping):
        raise RuntimeError("Accepted round is missing gradient Phase0 evidence.")
    receipt = dict(raw)
    observed_sha256 = receipt.pop("sha256", None)
    accounting = receipt.get("estimator_accounting")
    components = (
        accounting.get("components")
        if isinstance(accounting, Mapping)
        else None
    )
    input_indices = receipt.get("input_pool_indices")
    retained_indices = receipt.get("retained_pool_indices")
    event_ids = receipt.get("estimator_event_ids")
    input_count = int(receipt.get("input_candidate_count", -1))
    retained_count = int(receipt.get("retained_candidate_count", -1))
    if (
        receipt.get("schema") != expected[0]
        or receipt.get("policy") != expected[1]
        or observed_sha256 != canonical_sha256(receipt)
        or receipt.get("score")
        != "absolute_coordinate_energy_gradient_v1"
        or receipt.get("metric_policy") != "off"
        or receipt.get("compile_cost_policy") != "off"
        or receipt.get("measurement_cost_policy") != "off"
        or int(receipt.get("requested_shortlist_size", -1))
        != RA_ADAPT_GRADIENT_PHASE0_SHORTLIST_SIZE
        or not isinstance(input_indices, list)
        or not isinstance(retained_indices, list)
        or input_count != len(input_indices)
        or retained_count != len(retained_indices)
        or int(receipt.get("effective_shortlist_size", -1))
        != retained_count
        or retained_count
        != min(RA_ADAPT_GRADIENT_PHASE0_SHORTLIST_SIZE, input_count)
        or len(set(int(value) for value in input_indices)) != input_count
        or len(set(int(value) for value in retained_indices))
        != retained_count
        or not set(int(value) for value in retained_indices).issubset(
            int(value) for value in input_indices
        )
        or not isinstance(event_ids, list)
        or len(event_ids) != input_count
        or len(set(str(value) for value in event_ids)) != input_count
        or components
        != {
            "N_H_outer": 0,
            "N_H_refit": 0,
            "N_grad": input_count,
            "N_metric": 0,
        }
        or accounting.get("S_alg") != input_count
        or accounting.get("zero_metric_measurements") is not True
    ):
        raise RuntimeError("Accepted gradient Phase0 evidence is invalid.")

    screen = scored_population.get("phase0_gradient_screen")
    if not isinstance(screen, Mapping):
        raise RuntimeError("Accepted gradient Phase0 scored domain is absent.")
    population = screen.get("population")
    shortlist = screen.get("shortlist")
    if (
        not isinstance(population, list)
        or not isinstance(shortlist, list)
        or {
            int(record.get("pool_index", -1))
            for record in population
            if isinstance(record, Mapping)
        }
        != {int(value) for value in input_indices}
        or {
            int(record.get("pool_index", -1))
            for record in shortlist
            if isinstance(record, Mapping)
        }
        != {int(value) for value in retained_indices}
    ):
        raise RuntimeError(
            "Gradient Phase0 detailed receipt and scored domain disagree."
        )
    return dict(raw)


def _accepted_round_scientific_receipts(
    finalization: Mapping[str, Any],
    *,
    adapter_id: str,
    candidate_representation: str | None = None,
    executable_inventory: CandidateInventory,
    algorithm_id: str | None = None,
    trust_policy_id: str = SOURCE_GRAM_NO_OVERLAP_TRUST,
) -> list[dict[str, Any]]:
    raw_history = finalization.get("history")
    stationary_semantic = bool(
        str(algorithm_id or "") in PAPER_I_RA_SEMANTIC_ALGORITHM_IDS
        and finalization.get("terminal_controller_outcome")
        == "phase0_stationary_no_competitive_candidate_v1"
    )
    phase3_no_admission_semantic = bool(
        str(algorithm_id or "") in PAPER_I_RA_SEMANTIC_ALGORITHM_IDS
        and finalization.get("terminal_controller_outcome")
        == ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1
        and isinstance(
            finalization.get("terminal_phase3_selection_receipt"),
            Mapping,
        )
    )
    if not isinstance(raw_history, (tuple, list)) or (
        not raw_history
        and not (stationary_semantic or phase3_no_admission_semantic)
    ):
        raise RuntimeError(
            "Canonical RA finalization is missing accepted-round history."
        )
    receipts: list[dict[str, Any]] = []
    latched_phase3_ablation = bool(
        str(algorithm_id or "")
        == RA_ADAPT_SINGLETON_LATCHED_PHASE3_ALGORITHM_ID
    )
    phase3_only_qiskit_cost = bool(
        str(algorithm_id or "") in RA_ADAPT_PHASE3_QISKIT_ALGORITHM_IDS
    )
    phase3_qiskit_denominator_no_lanes = bool(
        str(algorithm_id or "")
        == RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_DENOMINATOR_NO_LANES_ALGORITHM_ID
    )
    pure_hubbard_controller_noise = bool(
        str(algorithm_id or "")
        == PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ALGORITHM_ID
    )
    phase3_latched = False
    first_latch_round: int | None = None
    endpoint_overlap_trust_active = bool(
        str(trust_policy_id) == ENDPOINT_OVERLAP_DISPLACEMENT_TRUST
    )
    if str(trust_policy_id) not in {
        SOURCE_GRAM_NO_OVERLAP_TRUST,
        ENDPOINT_OVERLAP_DISPLACEMENT_TRUST,
    }:
        raise RuntimeError("Accepted RA history has an unknown trust policy.")
    for row_index, raw_row in enumerate(raw_history, start=1):
        if not isinstance(raw_row, Mapping):
            raise RuntimeError("Accepted RA history row is not a mapping.")
        trust_update = raw_row.get("route_a_trust_region_update")
        if not isinstance(trust_update, Mapping):
            raise RuntimeError(
                "Accepted RA round is missing its trust update."
            )
        raw_transaction = trust_update.get(
            "source_metric_trust_transaction"
        )
        trust: SourceGramNoOverlapTrustReceipt | None
        trust_projection: dict[str, Any] | None
        endpoint_overlap_projection: dict[str, Any] | None = None
        if isinstance(raw_transaction, Mapping):
            trust = source_gram_no_overlap_trust_receipt_from_mapping(
                raw_transaction,
                adapter_id=adapter_id,
            )
            trust_projection = (
                None
                if endpoint_overlap_trust_active
                else trust.as_dict()
            )
        else:
            trust = None
            if endpoint_overlap_trust_active:
                trust_projection = None
            elif _geometry_expansion_trust_limitation(trust_update):
                trust_projection = None
            else:
                failure = trust_update.get(
                    "source_metric_trust_transaction_failure"
                )
                raise RuntimeError(
                    "Accepted RA round is missing the required real "
                    "source-Gram trust transaction "
                    f"(round={row_index}, failure={failure!r})."
                )
        if endpoint_overlap_trust_active:
            accounting = trust_update.get(
                "endpoint_overlap_query_accounting"
            )
            try:
                predicted_fs = trust_update.get(
                    "predicted_fs_displacement"
                )
                realized_fs = float(
                    trust_update["realized_fs_displacement_exact"]
                )
                radius_before = float(trust_update["radius_before"])
                radius_after = float(trust_update["radius_after"])
                update_factor = float(trust_update["update_factor"])
                overlap_charge = trust_update[
                    "endpoint_overlap_query_charge"
                ]
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Accepted RA endpoint-overlap trust receipt is incomplete."
                ) from exc
            geometry_expansion = bool(
                trust_update.get("geometry_expansion_active") is True
            )
            if predicted_fs is not None:
                predicted_fs = float(predicted_fs)
            ratio_metric = str(
                trust_update.get("displacement_ratio_metric", "")
            )
            if geometry_expansion and not ratio_metric:
                ratio_metric = (
                    "geometry_expansion_endpoint_fubini_study_recalibration_v1"
                )
            if (
                str(trust_update.get("policy", ""))
                != ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2
                or trust_update.get("endpoint_overlap_measurement_required")
                is not True
                or trust_update.get("endpoint_overlap_measurement_performed")
                is not True
                or isinstance(overlap_charge, bool)
                or overlap_charge != 1
                or not isinstance(accounting, Mapping)
                or accounting.get("status") != "complete"
                or accounting.get("performed") is not True
                or accounting.get("component") != "N_metric"
                or accounting.get("formal_query_category") != "N_cross"
                or not all(
                    math.isfinite(value) and value >= 0.0
                    for value in (
                        realized_fs,
                        radius_before,
                        radius_after,
                        update_factor,
                    )
                )
                or (
                    predicted_fs is not None
                    and (
                        not math.isfinite(predicted_fs)
                        or predicted_fs < 0.0
                    )
                )
                or (
                    not geometry_expansion
                    and (
                        predicted_fs is None
                        or ratio_metric
                        != (
                            "predicted_fubini_study_vs_endpoint_"
                            "fubini_study_v1"
                        )
                    )
                )
            ):
                raise RuntimeError(
                    "Accepted RA endpoint-overlap trust receipt is malformed."
                )
            endpoint_payload = {
                "schema": "ra_adapt_endpoint_overlap_trust_receipt_v1",
                "policy": str(trust_update["policy"]),
                "context_mode": str(trust_update.get("context_mode", "")),
                "geometry_expansion_active": geometry_expansion,
                "predicted_fubini_study_displacement": predicted_fs,
                "realized_fubini_study_displacement": realized_fs,
                "displacement_ratio": trust_update.get(
                    "displacement_ratio"
                ),
                "displacement_ratio_metric": ratio_metric,
                "radius_before": radius_before,
                "radius_after": radius_after,
                "update_factor": update_factor,
                "update_reason": str(trust_update["update_reason"]),
                "endpoint_overlap_query_charge": 1,
                "endpoint_overlap_query_accounting": copy.deepcopy(
                    dict(accounting)
                ),
                "source_metric_prediction_receipt": (
                    None
                    if raw_transaction is None
                    else copy.deepcopy(dict(raw_transaction))
                ),
            }
            endpoint_overlap_projection = {
                **endpoint_payload,
                "sha256": canonical_sha256(endpoint_payload),
            }
        support = _required_retained_support(raw_row, trust=trust)
        stabilization = _required_phase3_stabilization(
            raw_row,
            support=support,
        )
        candidate_lineage = _accepted_candidate_lineage_receipts(
            raw_row,
            candidate_representation=candidate_representation,
            executable_inventory=executable_inventory,
        )
        canonical_full_response_v2 = bool(
            str(algorithm_id or "") == RA_ADAPT_ALGORITHM_ID
        )
        phase3_population_plateau_ablation = bool(
            str(algorithm_id or "")
            == RA_ADAPT_SINGLETON_PHASE3_PLATEAU_ALGORITHM_ID
        )
        fixed_refit_chart = _required_accepted_refit_fixed_chart(raw_row)
        scored_positions = raw_row.get(
            "scored_insertion_position_population"
        )
        if not isinstance(scored_positions, Mapping):
            raise RuntimeError(
                "Accepted RA round is missing its scored insertion-position "
                "population."
            )
        scored_payload = dict(scored_positions)
        scored_sha256 = scored_payload.pop("sha256", None)
        phases = scored_payload.get("phases")
        if (
            scored_payload.get("schema")
            != "paper_i_scored_insertion_position_population_v1"
            or scored_payload.get("coordinate_chart")
            != EXACT_ORDERED_INSERTION_CHART
            or scored_payload.get("phase_order")
            != ["phase_i", "phase_ii", "phase_iii"]
            or not isinstance(phases, list)
            or len(phases) != 3
            or scored_sha256 != canonical_sha256(scored_payload)
        ):
            raise RuntimeError(
                "Accepted RA scored insertion-position receipt is invalid."
            )
        observed_count = 0
        interior_count = 0
        append_count = 0
        append_position = int(scored_payload.get("append_position", -1))
        for phase_index, phase in enumerate(phases):
            if not isinstance(phase, Mapping):
                raise RuntimeError(
                    "Accepted RA scored insertion phase is malformed."
                )
            records = phase.get("records")
            if (
                phase.get("phase")
                != ("phase_i", "phase_ii", "phase_iii")[phase_index]
                or not isinstance(records, list)
                or not records
                or int(phase.get("population_count", -1)) != len(records)
                or phase.get("ordered_population_sha256")
                != canonical_sha256(records)
            ):
                raise RuntimeError(
                    "Accepted RA scored insertion phase population drifted."
                )
            for record in records:
                if not isinstance(record, Mapping):
                    raise RuntimeError(
                        "Accepted RA scored insertion record is malformed."
                    )
                position = int(record.get("insertion_position", -1))
                position_class = str(record.get("position_class", ""))
                if (
                    position < 0
                    or position > append_position
                    or position_class
                    != ("interior" if position < append_position else "append")
                ):
                    raise RuntimeError(
                        "Accepted RA scored insertion record is out of range."
                    )
                observed_count += 1
                interior_count += int(position_class == "interior")
                append_count += int(position_class == "append")
        if (
            observed_count != int(scored_payload.get("scored_record_count", -1))
            or interior_count
            != int(scored_payload.get("interior_scored_count", -1))
            or append_count
            != int(scored_payload.get("append_scored_count", -1))
        ):
            raise RuntimeError(
                "Accepted RA scored insertion-position totals do not close."
            )
        round_receipt: dict[str, Any] = {
            "accepted_round_ordinal": int(row_index),
            "accepted_candidate_lineage": [
                receipt.to_dict() for receipt in candidate_lineage
            ],
            "retained_support": support.as_dict(),
            "phase3_stabilization": (
                None if stabilization is None else stabilization.to_dict()
            ),
            "source_gram_no_overlap_trust": trust_projection,
            "accepted_refit_fixed_chart_receipt": (
                fixed_refit_chart.as_dict()
            ),
            "accepted_refit_fixed_chart_sha256": str(
                fixed_refit_chart.sha256
            ),
            "scored_insertion_position_population": copy.deepcopy(
                dict(scored_positions)
            ),
        }
        if pure_hubbard_controller_noise:
            controller_noise = raw_row.get("controller_noise")
            runtime_delta = (
                controller_noise.get("runtime_delta")
                if isinstance(controller_noise, Mapping)
                else None
            )
            delta_records = (
                runtime_delta.get("evaluation_records_delta")
                if isinstance(runtime_delta, Mapping)
                else None
            )
            delta_compile_receipts = (
                runtime_delta.get("compiled_noise_receipts_delta")
                if isinstance(runtime_delta, Mapping)
                else None
            )
            rng_state_after = (
                runtime_delta.get("rng_state_after")
                if isinstance(runtime_delta, Mapping)
                else None
            )
            if (
                not isinstance(controller_noise, Mapping)
                or controller_noise.get("schema")
                != "paper_i_pure_hubbard_controller_noise_transition_v1"
                or not isinstance(runtime_delta, Mapping)
                or runtime_delta.get("schema")
                != (
                    "paper_i_pure_hubbard_controller_noise_"
                    "transition_delta_v1"
                )
                or not isinstance(delta_records, list)
                or not delta_records
                or runtime_delta.get("evaluation_records_delta_sha256")
                != canonical_sha256(delta_records)
                or not isinstance(delta_compile_receipts, Mapping)
                or runtime_delta.get(
                    "compiled_noise_receipts_delta_sha256"
                )
                != canonical_sha256(delta_compile_receipts)
                or not isinstance(rng_state_after, Mapping)
                or runtime_delta.get("rng_state_after_sha256")
                != canonical_sha256(rng_state_after)
                or float(controller_noise.get("exact_diagnostic_energy_before"))
                != float(raw_row["energy_before_opt"])
                or float(controller_noise.get("exact_diagnostic_energy_after"))
                != float(raw_row["energy_after_opt"])
            ):
                raise RuntimeError(
                    "Accepted pure-Hubbard round lost its controller-noise "
                    "transition closure."
                )
            round_receipt["controller_noise"] = copy.deepcopy(
                dict(controller_noise)
            )
        if str(algorithm_id or "") in {
            RA_ADAPT_MACRO_GRADIENT_PHASE0_PROXY_NO_LANES_ALGORITHM_ID,
            RA_ADAPT_MACRO_GRADIENT_PHASE0_MACRO_PHASE23_QISKIT_ALGORITHM_ID,
            RA_ADAPT_MACRO_GRADIENT_PHASE0_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID,
            RA_ADAPT_GLOBAL_SINGLETON_GRADIENT_PHASE0_PHASE23_QISKIT_ALGORITHM_ID,
            PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ALGORITHM_ID,
            *PAPER_I_RA_SEMANTIC_ALGORITHM_IDS,
        }:
            round_receipt["ra_gradient_phase0_shortlist"] = (
                _validated_gradient_phase0_round_receipt(
                    raw_row,
                    scored_population=scored_payload,
                    algorithm_id=str(algorithm_id),
                )
            )
        if str(algorithm_id or "") in PAPER_I_RA_SEMANTIC_ALGORITHM_IDS:
            projected_phase123 = raw_row.get(
                "projected_phase3_population_receipt"
            )
            if not isinstance(projected_phase123, Mapping):
                raise RuntimeError(
                    "Accepted semantic round lost its Phase-I--III "
                    "population evidence."
                )
            round_receipt["projected_phase3_population_receipt"] = (
                validate_semantic_projected_phase123_receipt(
                    projected_phase123
                )
            )
        if endpoint_overlap_trust_active:
            round_receipt["endpoint_overlap_trust"] = (
                endpoint_overlap_projection
            )
        plateau = raw_row.get("insertion_commutation_plateau")
        if plateau is not None:
            plateau_policy = (
                str(plateau.get("policy", "")).strip()
                if isinstance(plateau, Mapping)
                else ""
            )
            if plateau_policy not in {
                "insertion_commutation_plateau_v1",
                "insertion_commutation_plateau_v2",
            }:
                raise RuntimeError(
                    "Accepted RA plateau-insertion policy is unknown."
                )
            try:
                validated_phase0 = round_receipt.get(
                    "ra_gradient_phase0_shortlist"
                )
                position_phase0 = bool(
                    isinstance(validated_phase0, Mapping)
                    and validated_phase0.get("route_variant")
                    in PAPER_I_RA_PHASE0_POSITION_ROUTE_VARIANTS
                )
                reduction_scored_population = (
                    scored_payload
                    if position_phase0
                    else (
                        _phase0_reduction_validation_population(
                            scored_payload
                        )
                        if str(algorithm_id or "")
                        in {
                            RA_ADAPT_MACRO_GRADIENT_PHASE0_PROXY_NO_LANES_ALGORITHM_ID,
                            RA_ADAPT_MACRO_GRADIENT_PHASE0_MACRO_PHASE23_QISKIT_ALGORITHM_ID,
                            RA_ADAPT_MACRO_GRADIENT_PHASE0_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID,
                            RA_ADAPT_GLOBAL_SINGLETON_GRADIENT_PHASE0_PHASE23_QISKIT_ALGORITHM_ID,
                            PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ALGORITHM_ID,
                            *PAPER_I_RA_SEMANTIC_ALGORITHM_IDS,
                        }
                        else scored_payload
                    )
                )
                validate_commutation_reduced_insertion_receipt(
                    plateau,
                    expected_policy=plateau_policy,
                    expected_requested_positions=(
                        list(range(append_position + 1))
                        if isinstance(plateau, Mapping)
                        and plateau.get("domain_open") is True
                        else [append_position]
                    ),
                    scored_population=reduction_scored_population,
                    expected_representative_pairs=(
                        [
                            (
                                int(row["pool_index"]),
                                int(row["insertion_position"]),
                            )
                            for row in validated_phase0["population"]
                        ]
                        if position_phase0
                        else None
                    ),
                    expected_phase_i_pairs=(
                        [
                            (
                                int(row["pool_index"]),
                                int(row["insertion_position"]),
                            )
                            for row in validated_phase0["retained_records"]
                        ]
                        if position_phase0
                        else None
                    ),
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Accepted RA plateau-insertion receipt is invalid."
                ) from exc
            round_receipt["insertion_commutation_plateau"] = (
                copy.deepcopy(dict(plateau))
            )
        if phase3_only_qiskit_cost:
            projected = raw_row.get(
                "projected_phase3_population_receipt"
            )
            qiskit_receipt = (
                projected.get("phase3_qiskit_selector_cost_receipt")
                if isinstance(projected, Mapping)
                else None
            )
            rows = (
                qiskit_receipt.get("rows")
                if isinstance(qiskit_receipt, Mapping)
                else None
            )
            evaluated_count = (
                int(projected.get("phase3_evaluated_candidate_count", -1))
                if isinstance(projected, Mapping)
                else -1
            )
            phase3_scored_records = (
                phases[2].get("records")
                if isinstance(phases[2], Mapping)
                else None
            )
            full_population_identity_rows = (
                projected.get("phase3_evaluated_population_identities")
                if isinstance(projected, Mapping)
                else None
            )
            if (
                not isinstance(projected, Mapping)
                or projected.get("schema")
                != "paper_i_projected_phase3_population_receipt_v2"
                or not isinstance(qiskit_receipt, Mapping)
                or qiskit_receipt.get("schema")
                != "paper_i_phase3_qiskit_marginal_compile_receipt_v1"
                or qiskit_receipt.get("scope")
                != BACKEND_COMPILE_SCOPE_PHASE3_QISKIT_ONLY_V1
                or qiskit_receipt.get("phase_i_phase_ii_cost_source")
                != MARRAKESH_GRAPH_SPAN_MODE
                or qiskit_receipt.get("phase_iii_cost_source")
                != "backend_transpile_v1"
                or qiskit_receipt.get("resolved_backend_name")
                != "FakeMarrakesh"
                or qiskit_receipt.get("resolution_kind") != "fake_exact"
                or qiskit_receipt.get("optimization_level") != 1
                or qiskit_receipt.get("seed_transpiler") != 7
                or qiskit_receipt.get("structure_theta_value") != 1.0
                or qiskit_receipt.get(
                    "accepted_base_and_trial_full_ansatz_transpiled"
                )
                is not True
                or qiskit_receipt.get("independent_base_trial_layouts")
                is not True
                or qiskit_receipt.get("preferred_backend_fallback_allowed")
                is not False
                or qiskit_receipt.get("negative_delta_reward_enabled")
                is not False
                or qiskit_receipt.get("one_qubit_coordinate_policy")
                != ONE_QUBIT_COORDINATE_COMPILED_POSITIVE_DELTA_V1
                or qiskit_receipt.get("raw_signed_telemetry_retained")
                is not True
                or qiskit_receipt.get("selector_circuit_coordinates")
                != [
                    "positive_clip_delta_N2q",
                    "positive_clip_delta_D2q",
                    "positive_clip_delta_N1q",
                ]
                or qiskit_receipt.get("population_normalization")
                != (
                    "family_robust_v1"
                    if phase3_qiskit_denominator_no_lanes
                    else "family_robust_symmetric_arctan_v1"
                )
                or qiskit_receipt.get("excluded_from_s_alg") is not True
                or not isinstance(rows, list)
                or len(rows) != evaluated_count
                or not isinstance(phase3_scored_records, list)
                or not isinstance(full_population_identity_rows, list)
                or len(full_population_identity_rows) != evaluated_count
                or projected.get(
                    "phase3_evaluated_population_identities_sha256"
                )
                != canonical_sha256(full_population_identity_rows)
                or qiskit_receipt.get("phase3_evaluated_candidate_count")
                != evaluated_count
                or qiskit_receipt.get("phase3_qiskit_estimate_count_delta")
                != evaluated_count
                or qiskit_receipt.get("rows_sha256")
                != canonical_sha256(rows)
            ):
                raise RuntimeError(
                    "Accepted Phase-III-only Qiskit round is missing its "
                    "closed exact-population compile-cost receipt."
                )
            scored_identities = [
                (
                    str(record.get("pool_label", "")),
                    int(record.get("pool_index", -1)),
                    str(record.get("generator_id", "")),
                    int(record.get("insertion_position", -1)),
                )
                for record in phase3_scored_records
                if isinstance(record, Mapping)
            ]
            qiskit_identities = [
                (
                    str(record.get("candidate_label", "")),
                    int(record.get("candidate_pool_index", -1)),
                    str(record.get("generator_id", "")),
                    int(record.get("position_id", -1)),
                )
                for record in rows
                if isinstance(record, Mapping)
            ]
            full_population_identities = [
                (
                    str(record.get("candidate_label", "")),
                    int(record.get("candidate_pool_index", -1)),
                    str(record.get("generator_id", "")),
                    int(record.get("position_id", -1)),
                )
                for record in full_population_identity_rows
                if isinstance(record, Mapping)
            ]
            if (
                len(qiskit_identities) != evaluated_count
                or len(full_population_identities) != evaluated_count
                or len(set(full_population_identities)) != evaluated_count
                or qiskit_identities != full_population_identities
                or not scored_identities
                or any(
                    identity not in set(full_population_identities)
                    for identity in scored_identities
                )
            ):
                raise RuntimeError(
                    "Phase-III Qiskit telemetry identities differ from its "
                    "authenticated full evaluated Phase-III population."
                )
            population_hashes: set[str] = set()
            for qiskit_row in rows:
                if not isinstance(qiskit_row, Mapping):
                    raise RuntimeError(
                        "Phase-III Qiskit telemetry row is malformed."
                    )
                try:
                    raw_2q = float(
                        qiskit_row["raw_delta_compiled_count_2q"]
                    )
                    raw_d = float(
                        qiskit_row["raw_delta_compiled_depth_2q"]
                    )
                    raw_1q = float(
                        qiskit_row["raw_delta_compiled_count_1q"]
                    )
                    clipped = (
                        float(qiskit_row["c_hat_2q"]),
                        float(qiskit_row["c_hat_d"]),
                        float(qiskit_row["c_hat_1q"]),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "Phase-III Qiskit telemetry is incomplete."
                    ) from exc
                if not all(
                    math.isfinite(value)
                    for value in (raw_2q, raw_d, raw_1q, *clipped)
                ) or any(
                    not math.isclose(
                        observed,
                        max(0.0, raw),
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                    for observed, raw in zip(
                        clipped,
                        (raw_2q, raw_d, raw_1q),
                        strict=True,
                    )
                ):
                    raise RuntimeError(
                        "Phase-III Qiskit selector coordinates are not the "
                        "positive-clipped signed transpiler deltas."
                    )
                base_structure_key = qiskit_row.get(
                    "base_structure_key"
                )
                trial_structure_key = qiskit_row.get(
                    "trial_structure_key"
                )
                base_layout = qiskit_row.get(
                    "base_logical_to_physical"
                )
                trial_layout = qiskit_row.get(
                    "trial_logical_to_physical"
                )
                if (
                    not isinstance(base_structure_key, str)
                    or len(base_structure_key) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in base_structure_key
                    )
                    or not isinstance(trial_structure_key, str)
                    or len(trial_structure_key) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in trial_structure_key
                    )
                    or base_structure_key == trial_structure_key
                    or qiskit_row.get("base_initial_layout") is not None
                    or qiskit_row.get("trial_initial_layout") is not None
                    or not isinstance(base_layout, list)
                    or not isinstance(trial_layout, list)
                    or not base_layout
                    or len(base_layout) != len(trial_layout)
                    or qiskit_row.get(
                        "base_trial_layout_coupling_policy"
                    )
                    != "independent_unconstrained_full_transpiles_v1"
                ):
                    raise RuntimeError(
                        "Phase-III Qiskit full-base/full-trial layout "
                        "telemetry is unauthenticated."
                    )
                population_hash = qiskit_row.get(
                    "hardware_cost_population_hash"
                )
                if phase3_qiskit_denominator_no_lanes:
                    try:
                        denominator = float(
                            qiskit_row["hardware_cost_denominator"]
                        )
                        recomputed = 1.0 + sum(
                            float(qiskit_row[f"lambda_{key}"])
                            * float(qiskit_row[f"c_bar_{key}"])
                            for key in ("2q", "d", "1q")
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        raise RuntimeError(
                            "Phase-III Qiskit denominator telemetry is incomplete."
                        ) from exc
                    if (
                        qiskit_row.get("hardware_cost_policy")
                        != "family_robust_v1"
                        or float(qiskit_row.get("lambda_theta", -1.0)) != 0.0
                        or float(qiskit_row.get("lambda_shot", -1.0)) != 0.0
                        or not math.isfinite(denominator)
                        or denominator < 1.0
                        or not math.isclose(
                            denominator,
                            recomputed,
                            rel_tol=0.0,
                            abs_tol=1.0e-12,
                        )
                    ):
                        raise RuntimeError(
                            "Phase-III Qiskit selector lost its literal "
                            "marginal-cost denominator."
                        )
                else:
                    if (
                        not isinstance(population_hash, str)
                        or not population_hash
                    ):
                        raise RuntimeError(
                            "Phase-III Qiskit population normalization is "
                            "unauthenticated."
                        )
                    population_hashes.add(population_hash)
            if (
                not phase3_qiskit_denominator_no_lanes
                and len(population_hashes) != 1
            ):
                raise RuntimeError(
                    "Phase-III Qiskit candidates were not normalized over "
                    "one common evaluated population."
                )
            round_receipt["projected_phase3_population_receipt"] = (
                copy.deepcopy(dict(projected))
            )
        if phase3_population_plateau_ablation:
            activation = raw_row.get("phase3_population_activation")
            projected = raw_row.get(
                "projected_phase3_population_receipt"
            )
            if (
                not isinstance(plateau, Mapping)
                or not isinstance(activation, Mapping)
                or not isinstance(projected, Mapping)
                or activation.get("schema")
                != "ra_phase3_population_activation_receipt_v1"
                or activation.get("policy")
                != RA_ADAPT_PHASE3_POPULATION_ON_INSERTION_PLATEAU
                or activation.get("competitive_population_live")
                is not plateau.get("domain_open")
                or activation.get("insertion_plateau_domain_open")
                is not plateau.get("domain_open")
                or activation.get("independent_latch_active") is not False
                or activation.get("hysteresis_active") is not False
                or projected.get("competitive_population_activation")
                != activation
            ):
                raise RuntimeError(
                    "Accepted singleton Phase-III activation did not project "
                    "the authenticated plateau predicate."
                )
            phase3_live = bool(activation["competitive_population_live"])
            competitive_count = int(
                projected.get("competitive_population_input_count", -1)
            )
            available_count = int(
                projected.get("phase2_available_shortlist_count", -1)
            )
            if (
                competitive_count < 1
                or available_count < competitive_count
                or (not phase3_live and competitive_count != 1)
                or (
                    not phase3_live
                    and activation.get("preplateau_admission_authority")
                    != "phase2_raw_score_top_rank_v1"
                )
                or (
                    not phase3_live
                    and activation.get("winner_materialization_policy")
                    != RA_ADAPT_PHASE3_PREPLATEAU_WINNER_MATERIALIZATION
                )
            ):
                raise RuntimeError(
                    "Accepted singleton Phase-III competitive population "
                    "receipt is inconsistent."
                )
            round_receipt["phase3_population_activation"] = (
                copy.deepcopy(dict(activation))
            )
            round_receipt["projected_phase3_population_receipt"] = (
                copy.deepcopy(dict(projected))
            )
        if latched_phase3_ablation:
            activation = raw_row.get("phase3_population_activation")
            projected = raw_row.get(
                "projected_phase3_population_receipt"
            )
            entry = (
                activation.get("entry_plateau_receipt")
                if isinstance(activation, Mapping)
                else None
            )
            expected_after = bool(
                phase3_latched
                or (
                    isinstance(entry, Mapping)
                    and entry.get("domain_open") is True
                )
            )
            expected_opened = bool(not phase3_latched and expected_after)
            if expected_opened:
                first_latch_round = row_index
            raw_entry_open = bool(
                isinstance(entry, Mapping)
                and entry.get("domain_open") is True
            )
            expected_insertion_open = bool(
                phase3_latched and raw_entry_open
            )
            compared_entry_fields = (
                "policy",
                "trigger_energy_before",
                "trigger_energy_after",
                "trigger_energy_decrease",
                "prior_cumulative_energy_decrease",
                "prior_accepted_transition_count",
                "prior_mean_energy_decrease",
                "marginal_to_prior_mean_decrease_ratio",
                "prior_mean_decrease_ratio_threshold",
                "threshold_comparison",
                "calibration_status",
            )
            if (
                not isinstance(plateau, Mapping)
                or not isinstance(activation, Mapping)
                or not isinstance(projected, Mapping)
                or not isinstance(entry, Mapping)
                or activation.get("schema")
                != "ra_phase3_population_activation_receipt_v2"
                or activation.get("policy")
                != RA_ADAPT_PHASE3_POPULATION_LATCHED_ON_PROGRESS_PLATEAU
                or entry.get("policy")
                != "insertion_commutation_plateau_v2"
                or activation.get("entry_plateau_domain_open")
                is not raw_entry_open
                or activation.get("phase3_latched_before_round")
                is not phase3_latched
                or activation.get("phase3_latched_after_round")
                is not expected_after
                or activation.get("phase3_latch_opened_this_round")
                is not expected_opened
                or activation.get("competitive_population_live")
                is not expected_after
                or activation.get("first_latch_accepted_round")
                != first_latch_round
                or activation.get("independent_latch_active") is not True
                or activation.get("deactivation_allowed") is not False
                or activation.get("hysteresis_active") is not False
                or plateau.get("insertion_trigger_scope")
                != RA_ADAPT_POST_LATCH_INSERTION_TRIGGER_SCOPE
                or plateau.get("phase3_latched_before_round")
                is not phase3_latched
                or plateau.get("preceding_full_phase3_transition_eligible")
                is not phase3_latched
                or plateau.get("insertion_latch_active") is not False
                or plateau.get("raw_progress_plateau_domain_open")
                is not raw_entry_open
                or plateau.get("domain_open") is not expected_insertion_open
                or activation.get("insertion_plateau_domain_open")
                is not expected_insertion_open
                or activation.get("insertion_trigger_eligible")
                is not phase3_latched
                or any(
                    entry.get(key) != plateau.get(key)
                    for key in compared_entry_fields
                )
                or projected.get("competitive_population_activation")
                != activation
            ):
                raise RuntimeError(
                    "Accepted latched Phase-III/separate-insertion receipts "
                    "did not preserve their causal ordering."
                )
            competitive_count = int(
                projected.get("competitive_population_input_count", -1)
            )
            available_count = int(
                projected.get("phase2_available_shortlist_count", -1)
            )
            if (
                competitive_count < 1
                or available_count < 1
                or (
                    expected_after
                    and competitive_count != available_count
                )
                or (
                    not expected_after and competitive_count != 1
                )
                or (
                    not expected_after
                    and activation.get("preplateau_admission_authority")
                    != "phase2_raw_score_top_rank_v1"
                )
                or (
                    not expected_after
                    and activation.get("winner_materialization_policy")
                    != RA_ADAPT_PHASE3_PREPLATEAU_WINNER_MATERIALIZATION
                )
                or (
                    expected_after
                    and activation.get("preplateau_admission_authority")
                    is not None
                )
                or (
                    expected_after
                    and activation.get("winner_materialization_policy")
                    is not None
                )
            ):
                raise RuntimeError(
                    "Accepted latched Phase-III competitive population "
                    "receipt is inconsistent."
                )
            phase3_latched = expected_after
            round_receipt["phase3_population_activation"] = (
                copy.deepcopy(dict(activation))
            )
            round_receipt["projected_phase3_population_receipt"] = (
                copy.deepcopy(dict(projected))
            )
        reduced = raw_row.get("insertion_commutation_reduced")
        if reduced is not None:
            reduced_policy = (
                str(reduced.get("policy", ""))
                if isinstance(reduced, Mapping)
                else ""
            )
            if reduced_policy == "always_commutation_reduced":
                expected_reduced_positions = list(
                    range(append_position + 1)
                )
            elif reduced_policy == APPEND_COMMUTATION_REDUCED_POLICY:
                expected_reduced_positions = [append_position]
            else:
                raise RuntimeError(
                    "Accepted RA insertion-reduction policy is unknown."
                )
            try:
                validated_phase0 = round_receipt.get(
                    "ra_gradient_phase0_shortlist"
                )
                position_phase0 = bool(
                    isinstance(validated_phase0, Mapping)
                    and validated_phase0.get("route_variant")
                    in PAPER_I_RA_PHASE0_POSITION_ROUTE_VARIANTS
                )
                validate_commutation_reduced_insertion_receipt(
                    reduced,
                    expected_policy=reduced_policy,
                    expected_requested_positions=(
                        expected_reduced_positions
                    ),
                    scored_population=scored_payload,
                    expected_representative_pairs=(
                        [
                            (
                                int(row["pool_index"]),
                                int(row["insertion_position"]),
                            )
                            for row in validated_phase0["population"]
                        ]
                        if position_phase0
                        else None
                    ),
                    expected_phase_i_pairs=(
                        [
                            (
                                int(row["pool_index"]),
                                int(row["insertion_position"]),
                            )
                            for row in validated_phase0["retained_records"]
                        ]
                        if position_phase0
                        else None
                    ),
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Accepted RA always-insertion reduction receipt is invalid."
                ) from exc
            round_receipt["insertion_commutation_reduced"] = (
                copy.deepcopy(dict(reduced))
            )
        accepted_refit = raw_row.get("accepted_refit")
        if canonical_full_response_v2:
            initialization = (
                accepted_refit.get("accepted_refit_initialization")
                if isinstance(accepted_refit, Mapping)
                else None
            )
            if not isinstance(initialization, Mapping):
                raise RuntimeError(
                    "Canonical RA v2 accepted refit is missing its full-response "
                    "initialization receipt."
                )
            initialization_receipt = copy.deepcopy(dict(initialization))
            status = str(initialization_receipt.get("status", ""))
            try:
                guard_nfev = int(
                    initialization_receipt.get("guard_objective_evals", -1)
                )
                refit_guard_nfev = int(
                    accepted_refit.get(
                        "accepted_refit_initialization_guard_nfev", -1
                    )
                )
                row_guard_nfev = int(
                    raw_row.get(
                        "nfev_accepted_refit_initialization_guard", -1
                    )
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Canonical RA v2 accepted-refit guard accounting is invalid."
                ) from exc
            mapping = initialization_receipt.get("supported_fs_mapping")
            candidate_gain = initialization_receipt.get(
                "phase3_candidate_gain_receipt"
            )
            if not isinstance(candidate_gain, Mapping):
                raise RuntimeError(
                    "Canonical RA v2 initialization lost the incremental-gain "
                    "receipt."
                )
            if (
                initialization_receipt.get("schema")
                != "accepted_refit_joint_response_initialization_v1"
                or initialization_receipt.get("enabled") is not True
                or initialization_receipt.get("attempted") is not True
                or initialization_receipt.get("policy")
                != ROUTE_A_JOINT_STEP_WARM_START_EXACT_GUARDED_V1
                or initialization_receipt.get("selection_mutated") is not False
                or initialization_receipt.get("prediction_authority")
                != "initialization_only_full_powell_refit_authoritative_v1"
                or initialization_receipt.get("guard_objective_stage")
                != "accepted_refit_joint_response_guard"
                or status not in {"accepted", "rejected", "unavailable"}
                or guard_nfev not in {0, 1}
                or guard_nfev != refit_guard_nfev
                or guard_nfev != row_guard_nfev
            ):
                raise RuntimeError(
                    "Canonical RA v2 accepted-refit initialization contract "
                    "drifted."
                )
            if status in {"accepted", "rejected"} and guard_nfev != 1:
                raise RuntimeError(
                    "A mapped RA v2 full-response seed must receive exactly one "
                    "exact objective guard evaluation."
                )
            if status == "unavailable" and guard_nfev != 0:
                raise RuntimeError(
                    "An unavailable RA v2 full-response seed cannot spend a "
                    "guard evaluation."
                )
            if status == "accepted" and (
                initialization_receipt.get("fallback_to_incumbent") is not False
                or not math.isfinite(
                    float(
                        initialization_receipt.get(
                            "mapped_seed_proposal_energy", float("nan")
                        )
                    )
                )
                or float(
                    initialization_receipt.get(
                        "mapped_seed_exact_gain", float("nan")
                    )
                )
                <= 0.0
            ):
                raise RuntimeError(
                    "Accepted RA v2 full-response initialization is not exactly "
                    "certified downhill."
                )
            if status == "rejected" and (
                initialization_receipt.get("fallback_to_incumbent") is not True
            ):
                raise RuntimeError(
                    "Rejected RA v2 initialization did not retain the incumbent."
                )
            if status in {"accepted", "rejected"}:
                if (
                    not isinstance(mapping, Mapping)
                    or mapping.get("schema")
                    != "supported_fs_joint_step_map_receipt_v1"
                    or mapping.get("source_step_within_supported_chart") is not True
                    or int(mapping.get("classical_quantum_query_charge", -1)) != 0
                    or int(mapping.get("logical_parameter_count", -1))
                    != len(mapping.get("phase_order_joint_step", ()))
                    or int(mapping.get("logical_parameter_count", -1))
                    != int(
                        candidate_gain.get("active_only_baseline", {}).get(
                            "active_coordinate_count", -1
                        )
                    )
                    + len(candidate_lineage)
                ):
                    raise RuntimeError(
                        "Canonical RA v2 full-response chart mapping is invalid."
                    )
            try:
                full_gain = float(candidate_gain["full_joint_trust_gain"])
                active_gain = float(candidate_gain["active_only_trust_gain"])
                raw_increment = float(
                    candidate_gain["incremental_candidate_gain_raw"]
                )
                incremental_gain = float(
                    candidate_gain["incremental_candidate_gain"]
                )
                selected_gain = float(candidate_gain["selected_gain"])
                comparison_tolerance = float(
                    candidate_gain["comparison_tolerance"]
                )
                predicted_full_gain = float(
                    initialization_receipt[
                        "mapped_seed_predicted_full_joint_reduction"
                    ]
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Canonical RA v2 incremental-gain receipt is incomplete."
                ) from exc
            baseline = candidate_gain.get("active_only_baseline")
            if (
                candidate_gain.get("schema")
                != "phase3_candidate_gain_receipt_v1"
                or candidate_gain.get("policy")
                != PHASE3_CANDIDATE_GAIN_JOINT_MINUS_ACTIVE_ONLY_V1
                or candidate_gain.get("joint_gain_semantics")
                != "incremental_candidate_gain_v1"
                or candidate_gain.get("comparison_feasible") is not True
                or int(candidate_gain.get("classical_quantum_query_charge", -1))
                != 0
                or not isinstance(baseline, Mapping)
                or baseline.get("candidate_independent") is not True
                or int(baseline.get("classical_quantum_query_charge", -1)) != 0
                or any(
                    not math.isfinite(value) or value < 0.0
                    for value in (
                        full_gain,
                        active_gain,
                        incremental_gain,
                        selected_gain,
                        comparison_tolerance,
                    )
                )
                or not math.isfinite(raw_increment)
                or abs(raw_increment - (full_gain - active_gain))
                > comparison_tolerance
                or abs(incremental_gain - max(0.0, raw_increment))
                > comparison_tolerance
                or abs(selected_gain - incremental_gain)
                > comparison_tolerance
                or abs(predicted_full_gain - full_gain)
                > comparison_tolerance
            ):
                raise RuntimeError(
                    "Canonical RA v2 full-vs-active candidate-gain receipt does "
                    "not close."
                )
            round_receipt["accepted_refit_initialization"] = (
                initialization_receipt
            )
            round_receipt[
                "accepted_refit_initialization_sha256"
            ] = canonical_sha256(initialization_receipt)
            round_receipt[
                "accepted_refit_initialization_guard_nfev"
            ] = int(guard_nfev)
        invocation = (
            accepted_refit.get("accepted_refit_invocation")
            if isinstance(accepted_refit, Mapping)
            else None
        )
        metric_accounting = (
            invocation.get("metric_query_accounting")
            if isinstance(invocation, Mapping)
            else None
        )
        if isinstance(metric_accounting, Mapping):
            round_receipt["accepted_refit_metric_query_accounting"] = (
                copy.deepcopy(dict(metric_accounting))
            )
        receipts.append(round_receipt)
    return receipts


def _validate_endpoint_only_accepted_round(
    round_receipt: Mapping[str, Any],
) -> None:
    """Reject interior scoring or admission on an append-only round."""

    scored = round_receipt.get("scored_insertion_position_population")
    if not isinstance(scored, Mapping):
        raise RuntimeError(
            "Endpoint-only accepted round has no scored-position population."
        )
    append_position = int(scored.get("append_position", -1))
    phases = scored.get("phases")
    if append_position < 0 or not isinstance(phases, list) or not phases:
        raise RuntimeError(
            "Endpoint-only accepted round has an invalid append endpoint."
        )
    for phase in phases:
        records = phase.get("records") if isinstance(phase, Mapping) else None
        if not isinstance(records, list) or not records:
            raise RuntimeError(
                "Endpoint-only accepted round has an invalid scored phase."
            )
        if any(
            not isinstance(record, Mapping)
            or int(record.get("insertion_position", -1)) != append_position
            or record.get("position_class") != "append"
            for record in records
        ):
            raise RuntimeError(
                "Endpoint-only accepted round scored an interior position."
            )
    lineage = round_receipt.get("accepted_candidate_lineage")
    if (
        not isinstance(lineage, list)
        or not lineage
        or any(
            not isinstance(record, Mapping)
            or int(record.get("insertion_position", -1)) != append_position
            for record in lineage
        )
    ):
        raise RuntimeError(
            "Endpoint-only accepted round admitted an interior position."
        )


def _pure_hubbard_controller_noise_scientific_receipt(
    *,
    accepted_round_receipts: list[dict[str, Any]],
    route_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Close the named sequential-noise trace at the outer RA boundary."""

    execution = route_contract.get("execution_settings")
    noise_contract = (
        execution.get("ra_controller_noise_contract")
        if isinstance(execution, Mapping)
        else None
    )
    if not isinstance(noise_contract, Mapping):
        raise RuntimeError(
            "Pure-Hubbard finalization lost its controller-noise contract."
        )
    terminal_records: list[dict[str, Any]] = []
    compiled_noise_receipts: dict[str, dict[str, Any]] = {}
    rng_state: dict[str, Any] | None = None
    for ordinal, round_receipt in enumerate(
        accepted_round_receipts, start=1
    ):
        transition = round_receipt.get("controller_noise")
        delta = (
            transition.get("runtime_delta")
            if isinstance(transition, Mapping)
            else None
        )
        records_delta = (
            delta.get("evaluation_records_delta")
            if isinstance(delta, Mapping)
            else None
        )
        compile_delta = (
            delta.get("compiled_noise_receipts_delta")
            if isinstance(delta, Mapping)
            else None
        )
        rng_state_after = (
            delta.get("rng_state_after")
            if isinstance(delta, Mapping)
            else None
        )
        if (
            not isinstance(delta, Mapping)
            or delta.get("schema")
            != "paper_i_pure_hubbard_controller_noise_transition_delta_v1"
            or delta.get("noise_contract_sha256")
            != noise_contract.get("sha256")
            or not isinstance(records_delta, list)
            or not records_delta
            or int(delta.get("evaluation_count_before", -1))
            != len(terminal_records)
            or int(delta.get("evaluation_count_after", -1))
            != len(terminal_records) + len(records_delta)
            or delta.get("evaluation_records_delta_sha256")
            != canonical_sha256(records_delta)
            or not isinstance(compile_delta, Mapping)
            or int(
                delta.get("compiled_noise_receipt_count_before", -1)
            )
            != len(compiled_noise_receipts)
            or any(
                plan_digest in compiled_noise_receipts
                for plan_digest in compile_delta
            )
            or int(
                delta.get("compiled_noise_receipt_count_after", -1)
            )
            != len(compiled_noise_receipts) + len(compile_delta)
            or delta.get("compiled_noise_receipts_delta_sha256")
            != canonical_sha256(compile_delta)
            or not isinstance(rng_state_after, Mapping)
            or delta.get("rng_state_after_sha256")
            != canonical_sha256(rng_state_after)
        ):
            raise RuntimeError(
                "Pure-Hubbard controller-noise round trace does not close "
                f"at accepted round {ordinal}."
            )
        terminal_records.extend(copy.deepcopy(records_delta))
        compiled_noise_receipts.update(
            {
                str(plan_digest): copy.deepcopy(dict(receipt))
                for plan_digest, receipt in compile_delta.items()
                if isinstance(receipt, Mapping)
            }
        )
        if (
            len(compiled_noise_receipts)
            != int(delta["compiled_noise_receipt_count_after"])
            or delta.get("cumulative_evaluation_records_sha256")
            != canonical_sha256(terminal_records)
            or delta.get("cumulative_compiled_noise_receipts_sha256")
            != canonical_sha256(compiled_noise_receipts)
        ):
            raise RuntimeError(
                "Pure-Hubbard controller-noise cumulative delta closure "
                f"failed at accepted round {ordinal}."
            )
        rng_state = copy.deepcopy(dict(rng_state_after))
    if not terminal_records or rng_state is None:
        raise RuntimeError("Pure-Hubbard controller-noise trace is absent.")
    if (
        not compiled_noise_receipts
    ):
        raise RuntimeError(
            "Pure-Hubbard compiled-noise receipt map is invalid."
        )
    for plan_digest, compile_receipt in compiled_noise_receipts.items():
        coherent = (
            compile_receipt.get("synthetic_coherent")
            if isinstance(compile_receipt, Mapping)
            else None
        )
        metrics = (
            compile_receipt.get("compile_metrics")
            if isinstance(compile_receipt, Mapping)
            else None
        )
        unsigned_compile_receipt = (
            {
                key: value
                for key, value in compile_receipt.items()
                if key != "sha256"
            }
            if isinstance(compile_receipt, Mapping)
            else {}
        )
        if (
            not isinstance(plan_digest, str)
            or not plan_digest
            or not isinstance(compile_receipt, Mapping)
            or compile_receipt.get("parameterized_plan_digest")
            != plan_digest
            or compile_receipt.get("sha256")
            != canonical_sha256(unsigned_compile_receipt)
            or not isinstance(coherent, Mapping)
            or int(coherent.get("inserted_count", 0)) <= 0
            or not isinstance(
                compile_receipt.get("inserted_errors_sha256"), str
            )
            or not isinstance(compile_receipt.get("compile_signature"), Mapping)
            or compile_receipt["compile_signature"].get(
                "synthetic_coherent_inserted_after_transpile"
            )
            is not True
            or not isinstance(metrics, Mapping)
            or any(
                not isinstance(metrics.get(key), int)
                or int(metrics[key]) < 0
                for key in (
                    "compiled_depth",
                    "compiled_size",
                    "compiled_two_qubit_count",
                    "compiled_cx_count",
                    "compiled_ecr_count",
                )
            )
            or not isinstance(metrics.get("compiled_op_counts"), Mapping)
        ):
            raise RuntimeError(
                "Pure-Hubbard applied coherent-noise compile receipt is "
                "invalid."
            )
    draw_count = (
        int(rng_state.get("draw_count", -1))
        if isinstance(rng_state, Mapping)
        else -1
    )
    expected_draw_start = 0
    for evaluation_ordinal, record in enumerate(
        terminal_records, start=1
    ):
        value_noise = (
            record.get("value_noise")
            if isinstance(record, Mapping)
            else None
        )
        depolarizing = (
            record.get("synthetic_depolarizing")
            if isinstance(record, Mapping)
            else None
        )
        coherent = (
            record.get("synthetic_coherent")
            if isinstance(record, Mapping)
            else None
        )
        if (
            not isinstance(record, Mapping)
            or record.get("schema")
            != "paper_i_pure_hubbard_controller_noise_evaluation_v1"
            or record.get("evaluation_ordinal") != evaluation_ordinal
            or not isinstance(record.get("stage"), str)
            or not record["stage"]
            or not isinstance(value_noise, Mapping)
            or value_noise.get("model") != "gaussian_iid_v1"
            or value_noise.get("draw_index_start") != expected_draw_start
            or value_noise.get("draw_index_stop")
            != expected_draw_start + 1
            or value_noise.get("n_draws") != 1
            or not isinstance(depolarizing, Mapping)
            or not isinstance(coherent, Mapping)
            or record.get("parameterized_plan_digest")
            not in compiled_noise_receipts
            or record.get("compiled_noise_receipt_sha256")
            != compiled_noise_receipts[
                record["parameterized_plan_digest"]
            ].get("sha256")
        ):
            raise RuntimeError(
                "Pure-Hubbard per-evaluation controller-noise evidence is "
                f"invalid at ordinal {evaluation_ordinal}."
            )
        expected_draw_start += 1
    if draw_count != expected_draw_start:
        raise RuntimeError(
            "Pure-Hubbard RNG draw cursor does not close to its ordered "
            "evaluation trace."
        )
    payload = {
        "schema": "paper_i_pure_hubbard_controller_noise_receipt_v1",
        "candidate_gradient_scoring": "noisy",
        "powell_refit_objective": "noisy",
        "plateau_energy_source": (
            "persisted_noisy_controller_energy_before_after_v1"
        ),
        "geometry_and_gram": "exact",
        "reported_energy": "exact_diagnostic",
        "same_circuit_incumbent": True,
        "optimizer_evaluation_order": "serial_v1",
        "candidate_record_cache": "off_fail_closed_v1",
        "noise_contract": copy.deepcopy(dict(noise_contract)),
        "noise_contract_sha256": str(noise_contract["sha256"]),
        "effective_oracle_config": copy.deepcopy(
            dict(noise_contract["effective_oracle_config"])
        ),
        "evaluation_count": len(terminal_records),
        "evaluation_records": terminal_records,
        "evaluation_records_sha256": canonical_sha256(terminal_records),
        "compiled_noise_receipts": copy.deepcopy(
            dict(compiled_noise_receipts)
        ),
        "compiled_noise_receipts_sha256": canonical_sha256(
            compiled_noise_receipts
        ),
        "value_noise": {
            "model": "gaussian_iid_v1",
            "seed": int(noise_contract["value_noise"]["seed"]),
            "std": float(noise_contract["value_noise"]["std"]),
            "draw_count": draw_count,
            "rng_state": copy.deepcopy(dict(rng_state)),
        },
        "accepted_round_count": len(accepted_round_receipts),
        "final_controller_energy": float(
            accepted_round_receipts[-1]["controller_noise"][
                "controller_energy_after"
            ]
        ),
        "final_exact_diagnostic_energy": float(
            accepted_round_receipts[-1]["controller_noise"][
                "exact_diagnostic_energy_after"
            ]
        ),
    }
    return {**payload, "sha256": canonical_sha256(payload)}


def _validate_reduced_accepted_round_admission(
    round_receipt: Mapping[str, Any],
    *,
    reduction_key: str,
) -> None:
    """Bind every admitted lineage to one scored class representative."""

    reduction = round_receipt.get(reduction_key)
    raw_plans = (
        reduction.get("candidate_position_plans")
        if isinstance(reduction, Mapping)
        else None
    )
    if not isinstance(raw_plans, list) or not raw_plans:
        raise RuntimeError(
            "Reduced accepted round has no representative plans."
        )
    plans_by_pool_index: dict[int, Mapping[str, Any]] = {}
    for raw_plan in raw_plans:
        if not isinstance(raw_plan, Mapping):
            raise RuntimeError(
                "Reduced accepted round has a malformed representative plan."
            )
        pool_index = int(raw_plan.get("candidate_pool_index", -1))
        if pool_index < 0 or pool_index in plans_by_pool_index:
            raise RuntimeError(
                "Reduced accepted round has ambiguous representative plans."
            )
        plans_by_pool_index[pool_index] = raw_plan

    scored = round_receipt.get("scored_insertion_position_population")
    phases = scored.get("phases") if isinstance(scored, Mapping) else None
    phase_iii_rows = (
        [
            phase
            for phase in phases
            if isinstance(phase, Mapping)
            and phase.get("phase") == "phase_iii"
        ]
        if isinstance(phases, list)
        else []
    )
    if len(phase_iii_rows) != 1 or not isinstance(
        phase_iii_rows[0].get("records"),
        list,
    ):
        raise RuntimeError(
            "Reduced accepted round has no unique Phase-III population."
        )
    phase_iii_records = phase_iii_rows[0]["records"]

    lineage = round_receipt.get("accepted_candidate_lineage")
    if not isinstance(lineage, list) or not lineage:
        raise RuntimeError(
            "Reduced accepted round has no admitted candidate lineage."
        )
    for admitted in lineage:
        if not isinstance(admitted, Mapping):
            raise RuntimeError(
                "Reduced accepted round has malformed admitted lineage."
            )
        candidate_label = str(admitted.get("candidate_label", ""))
        generator_identity = str(admitted.get("generator_identity", ""))
        insertion_position = int(admitted.get("insertion_position", -1))
        matches = [
            record
            for record in phase_iii_records
            if isinstance(record, Mapping)
            and record.get("pool_label") == candidate_label
            and record.get("generator_id") == generator_identity
            and int(record.get("insertion_position", -1))
            == insertion_position
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "Reduced accepted lineage is not one exact Phase-III scored "
                "candidate-position identity."
            )
        pool_index = int(matches[0].get("pool_index", -1))
        plan = plans_by_pool_index.get(pool_index)
        representatives = (
            plan.get("representative_positions")
            if isinstance(plan, Mapping)
            else None
        )
        if (
            not isinstance(representatives, list)
            or insertion_position
            not in {int(value) for value in representatives}
        ):
            raise RuntimeError(
                "Reduced accepted lineage is not an authenticated "
                "commutation-class representative."
            )


def _required_deferred_gram_fallback(
    finalization: Mapping[str, Any],
) -> dict[str, Any]:
    raw = finalization.get(
        DEFERRED_GRAM_ALL_MODELS_INFEASIBLE_FALLBACK_V1
    )
    if not isinstance(raw, Mapping):
        raise RuntimeError(
            "Canonical RA finalization is missing its deferred-Gram "
            "fallback summary."
        )
    receipt = dict(raw)
    if (
        str(receipt.get("schema", ""))
        != DEFERRED_GRAM_ALL_MODELS_INFEASIBLE_FALLBACK_V1
        or str(receipt.get("scope", "")) != "run"
    ):
        raise RuntimeError(
            "Canonical RA deferred-Gram fallback summary is invalid."
        )
    for key in ("enabled", "fired", "rounds", "charge"):
        if key not in receipt:
            raise RuntimeError(
                "Canonical RA deferred-Gram fallback summary is incomplete."
            )
    return receipt


def run_ra_adapt(
    problem: ResolvedProblemContext,
    request: RAAdaptRequest | ResolvedRAAdaptProtocol | None = None,
    *,
    operational_controls: RAAdaptOperationalControls | None = None,
) -> RAAdaptResult:
    """Execute one canonical Paper-I RA-ADAPT request.

    ``operational_controls`` is available only for an already validated,
    bundle-resolved protocol.  It can shorten (never extend) the authorized
    horizon, authenticate a resume checkpoint, and redirect observation
    sidecars without mutating the protocol or its digest.
    """

    if not isinstance(problem, ResolvedProblemContext):
        raise TypeError("problem must be a ResolvedProblemContext.")
    family_key = str(problem.family_key).strip().lower()
    if not (
        (family_key == "hh" and int(problem.request.num_sites) == 2)
        or (family_key == "hh" and int(problem.request.num_sites) == 3)
        or (family_key == "hubbard" and int(problem.request.num_sites) == 2)
        or family_key == H2O_LINEAR_FD_FAMILY
    ):
        raise ValueError(
            "The ordinary Paper-I RA-ADAPT facade is locked to the canonical "
            "Hubbard--Holstein L=2 problem. Other families and sizes require "
            "an explicitly named lane-owned application."
        )
    if family_key == "hubbard":
        named_request = (
            request.request
            if isinstance(request, ResolvedRAAdaptProtocol)
            else request
        )
        if not (
            isinstance(named_request, RAAdaptRequest)
            and is_paper_i_pure_hubbard_noise_page12_application(
                problem,
                named_request,
            )
        ):
            raise ValueError(
                "The ordinary Paper-I RA-ADAPT facade is locked to the "
                "canonical Hubbard--Holstein L=2 problem. Pure Hubbard is "
                "executable only through the exact named Page-12 full-noise "
                "application adapter."
            )
    bundle_resolved = isinstance(request, ResolvedRAAdaptProtocol)
    if request is None:
        public_request = RAAdaptRequest()
        protocol = build_resolved_ra_protocol(problem, public_request)
    elif isinstance(request, RAAdaptRequest):
        public_request = request
        protocol = build_resolved_ra_protocol(problem, public_request)
    elif isinstance(request, ResolvedRAAdaptProtocol):
        if request.schema not in RA_ADAPT_PROTOCOL_SCHEMAS:
            raise ValueError("run_ra_adapt requires an RA protocol.")
        protocol = request
        public_request = request.request
        if not isinstance(public_request, RAAdaptRequest):
            raise TypeError("Resolved RA protocol lost its request.")
    else:
        raise TypeError(
            "request must be RAAdaptRequest, a bundle-resolved RA protocol, "
            "or None."
        )
    _h2o_application_active(problem, public_request)
    l3_page12_application = is_paper_i_l3_page12_application(
        problem,
        public_request,
    )
    pure_hubbard_noise_application = (
        is_paper_i_pure_hubbard_noise_page12_application(
            problem,
            public_request,
        )
    )
    if (
        family_key == "hh"
        and int(problem.request.num_sites) == 3
        and not l3_page12_application
    ):
        raise ValueError(
            "Hubbard--Holstein L=3 is executable only through the exact "
            "named Page-12 L=3 application adapter."
        )
    if family_key == "hubbard" and not pure_hubbard_noise_application:
        raise ValueError(
            "The ordinary Paper-I RA-ADAPT facade is locked to the canonical "
            "Hubbard--Holstein L=2 problem. Pure Hubbard is executable only "
            "through the exact named Page-12 full-noise application adapter."
        )

    if operational_controls is not None:
        if not bundle_resolved:
            raise ValueError(
                "operational_controls requires a validated, bundle-resolved "
                "RA protocol."
            )
        if not isinstance(
            operational_controls,
            RAAdaptOperationalControls,
        ):
            raise TypeError(
                "operational_controls must be "
                "RAAdaptOperationalControls or None."
            )

    require_protocol_materialization_authority(
        protocol,
        ordinary_algorithm_id=RA_ADAPT_ALGORITHM_ID,
        ordinary_bundle_id=(
            RA_ADAPT_ORDINARY_BUNDLE_ID
        ),
        ordinary_bundle_manifest_sha256=_ordinary_bundle_digest(),
        additional_ordinary_identities=(
            (
                RA_ADAPT_LEGACY_ALGORITHM_ID,
                RA_ADAPT_LEGACY_ORDINARY_BUNDLE_ID,
                _legacy_ordinary_bundle_digest(),
            ),
        ),
    )
    if protocol.algorithm_id in PAPER_I_RA_SEMANTIC_ALGORITHM_IDS:
        if protocol.bundle_materialization is None:
            raise ValueError(
                "Semantic-closure execution requires its source-bound "
                "materialization receipt."
            )
        validate_semantic_closure_materialization_authority(
            problem,
            public_request,
            receipt=protocol.bundle_materialization,
            source_lock_refs=protocol.source_locks,
        )
        problem = canonical_semantic_execution_problem(problem)
    if l3_page12_application:
        require_paper_i_l3_page12_materialization(
            problem=problem,
            request=public_request,
            algorithm_id=str(protocol.algorithm_id),
            active_gradient_policy=str(protocol.active_gradient_policy),
            resource_weighting_scope=str(
                protocol.resource_weighting_scope
            ),
            source_locks=protocol.source_locks,
        )
    if pure_hubbard_noise_application:
        require_paper_i_pure_hubbard_noise_page12_materialization(
            problem=problem,
            request=public_request,
            algorithm_id=str(protocol.algorithm_id),
            active_gradient_policy=str(protocol.active_gradient_policy),
            resource_weighting_scope=str(
                protocol.resource_weighting_scope
            ),
            source_locks=protocol.source_locks,
        )
    if operational_controls is not None:
        authorized_rounds = int(
            protocol.request.execution.stop.maximum_controller_rounds
        )
        if (
            int(protocol.horizon) != authorized_rounds
            or operational_controls.maximum_controller_rounds
            > authorized_rounds
        ):
            raise ValueError(
                "Operational controls may only shorten the authorized "
                "controller horizon."
            )
        if not isinstance(protocol.request.execution.resume, FreshStart):
            raise ValueError(
                "A bundle-resolved RA protocol must retain fresh-start "
                "execution authority; resume belongs only to operational "
                "controls."
            )
        public_request = replace(
            protocol.request,
            execution=replace(
                protocol.request.execution,
                stop=replace(
                    protocol.request.execution.stop,
                    maximum_controller_rounds=(
                        operational_controls.maximum_controller_rounds
                    ),
                ),
                resume=operational_controls.resume,
            ),
            observation=operational_controls.observation,
        )
        if isinstance(
            operational_controls.resume,
            AcceptedStateResume,
        ) and operational_controls.maximum_controller_rounds < 2:
            raise ValueError(
                "Accepted-state continuation requires an operational "
                "horizon of at least two controller rounds."
            )
    (
        candidate_inventory_lineage,
        executable_inventory,
    ) = _validate_resolved_pool_identity(
        problem,
        protocol,
    )
    if (
        isinstance(request, ResolvedRAAdaptProtocol)
        and protocol.bundle_materialization is None
        and protocol.algorithm_id == RA_ADAPT_ALGORITHM_ID
    ):
        expected_protocol = build_resolved_ra_protocol(
            problem,
            protocol.request,
        )
        if protocol != expected_protocol:
            raise ValueError(
                "Ordinary canonical RA v2 protocol drifted from deterministic "
                "resolution."
            )
    sr_request = _sr_request(public_request)
    historical_singleton_passthrough = bool(
        protocol.bundle_materialization is None
        and protocol.algorithm_id == RA_ADAPT_LEGACY_ALGORITHM_ID
        and protocol.bundle_id == RA_ADAPT_LEGACY_ORDINARY_BUNDLE_ID
        and protocol.bundle_manifest_sha256
        == _legacy_ordinary_bundle_digest()
        and isinstance(
            public_request.adapter, SinglePauliWordCandidateAdapter
        )
        and protocol.active_gradient_policy == ACTIVE_GRADIENT_MEASURED
        and protocol.resource_weighting_scope
        == RESOURCE_WEIGHTING_ALL_PHASE
    )
    if historical_singleton_passthrough:
        legacy_contract = (
            canonical_sr_snake_insertion_commutation_plateau_v1_contract()
        )
        route_override = (
            str(legacy_contract["route_profile"]),
            str(legacy_contract["route_profile"]),
            legacy_contract,
            canonical_sr_snake_insertion_commutation_plateau_v1_contract_sha256(),
        )
        context = _resolve_execution_context(
            problem,
            sr_request,
            route_override=route_override,
        )
        completed = _execute_resolved_context(context, sr_request)
    else:
        route_contract_request = (
            protocol.request
            if protocol.algorithm_id in PAPER_I_RA_SEMANTIC_ALGORITHM_IDS
            else public_request
        )
        route_override = _repaired_route_contract(
            route_contract_request,
            active_gradient_policy=protocol.active_gradient_policy,
            resource_weighting_scope=protocol.resource_weighting_scope,
            algorithm_id=protocol.algorithm_id,
            problem=problem,
        )
        if protocol.algorithm_id in {
            RA_ADAPT_ALGORITHM_ID,
            RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_ALGORITHM_ID,
            RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_DENOMINATOR_NO_LANES_ALGORITHM_ID,
        } or protocol.algorithm_id in PAPER_I_RA_SEMANTIC_ALGORITHM_IDS:
            bound_route = protocol.route_contract
            if not isinstance(bound_route, Mapping):
                raise ValueError(
                    "Canonical RA v2 protocol is missing its bound route."
                )
            bound_route_payload = dict(bound_route)
            bound_route_sha256 = str(
                bound_route_payload.pop("sha256", "")
            )
            if (
                bound_route_payload != route_override[2]
                or bound_route_sha256 != route_override[3]
            ):
                raise ValueError(
                    "Canonical RA v2 bound route drifted before execution."
                )
        context = _resolve_execution_context(
            problem,
            sr_request,
            route_override=route_override,
            candidate_adapter=public_request.adapter,
        )
        completed = _execute_resolved_context(context, sr_request)

    finalization = completed.finalization.to_serialization_mapping()
    policy = _policy_receipt(
        protocol=protocol,
        finalization=finalization,
    )
    accepted_round_receipts = _accepted_round_scientific_receipts(
        finalization,
        adapter_id=protocol.adapter_id,
        candidate_representation=protocol.candidate_representation,
        executable_inventory=executable_inventory,
        algorithm_id=protocol.algorithm_id,
        trust_policy_id=protocol.trust_policy_id,
    )
    route_contract = finalization.get("sr_route_profile_contract")
    if not isinstance(route_contract, Mapping):
        raise RuntimeError(
            "Canonical RA finalization is missing its resolved route contract."
        )
    _validate_executed_insertion_contract(
        public_request,
        route_contract,
        algorithm_id=protocol.algorithm_id,
    )
    route_execution = route_contract.get("execution_settings")
    if not isinstance(route_execution, Mapping):
        raise RuntimeError(
            "Canonical RA finalization omitted route execution settings."
        )
    expected_plateau_policy = str(
        route_execution.get("adapt_insertion_mode", "")
    )
    insertion_policy = public_request.method.insertion
    for round_receipt in accepted_round_receipts:
        has_plateau = "insertion_commutation_plateau" in round_receipt
        has_always_reduced = (
            "insertion_commutation_reduced" in round_receipt
        )
        if isinstance(
            insertion_policy,
            AlwaysCommutationReducedInsertion,
        ):
            valid_insertion_receipts = (
                has_always_reduced and not has_plateau
            )
        elif isinstance(insertion_policy, PlateauCommutationInsertion):
            valid_insertion_receipts = (
                has_plateau and not has_always_reduced
            )
        elif isinstance(insertion_policy, AppendOnlyInsertion):
            valid_insertion_receipts = (
                not has_plateau and not has_always_reduced
            )
        elif isinstance(
            insertion_policy,
            AppendCommutationReducedInsertion,
        ):
            valid_insertion_receipts = (
                has_always_reduced and not has_plateau
            )
        else:
            valid_insertion_receipts = False
        if not valid_insertion_receipts:
            raise RuntimeError(
                "Accepted RA insertion receipts do not match the typed "
                "insertion policy."
            )
        if isinstance(insertion_policy, AppendOnlyInsertion):
            _validate_endpoint_only_accepted_round(round_receipt)
        elif isinstance(
            insertion_policy,
            AppendCommutationReducedInsertion,
        ):
            reduced = round_receipt.get(
                "insertion_commutation_reduced"
            )
            if (
                not isinstance(reduced, Mapping)
                or reduced.get("policy")
                != APPEND_COMMUTATION_REDUCED_POLICY
            ):
                raise RuntimeError(
                    "Commutation-reduced append admission lost its typed "
                    "endpoint receipt."
                )
            _validate_reduced_accepted_round_admission(
                round_receipt,
                reduction_key="insertion_commutation_reduced",
            )
            _validate_endpoint_only_accepted_round(round_receipt)
        elif isinstance(insertion_policy, PlateauCommutationInsertion):
            plateau = round_receipt.get("insertion_commutation_plateau")
            if (
                not isinstance(plateau, Mapping)
                or plateau.get("policy") != expected_plateau_policy
            ):
                raise RuntimeError(
                    "Plateau-insertion receipt disagrees with its executed "
                    "route policy."
                )
            _validate_reduced_accepted_round_admission(
                round_receipt,
                reduction_key="insertion_commutation_plateau",
            )
            if (
                isinstance(plateau, Mapping)
                and plateau.get("domain_open") is False
            ):
                _validate_endpoint_only_accepted_round(round_receipt)
        elif isinstance(
            insertion_policy,
            AlwaysCommutationReducedInsertion,
        ):
            _validate_reduced_accepted_round_admission(
                round_receipt,
                reduction_key="insertion_commutation_reduced",
            )
    latest_round = (
        None if not accepted_round_receipts else accepted_round_receipts[-1]
    )
    selector_compile_cost_accounting = finalization.get(
        "selector_compile_cost_accounting"
    )
    semantic_selector_accounting_closure = None
    if protocol.algorithm_id in PAPER_I_RA_SEMANTIC_ALGORITHM_IDS:
        semantic_selector_accounting_closure = (
            validate_semantic_final_selector_accounting(
                algorithm_id=protocol.algorithm_id,
                route_contract=(
                    protocol.route_contract
                    if isinstance(protocol.route_contract, Mapping)
                    else {}
                ),
                selector_compile_cost_accounting=(
                    selector_compile_cost_accounting
                    if isinstance(
                        selector_compile_cost_accounting,
                        Mapping,
                    )
                    else {}
                ),
                finalization=finalization,
                accepted_round_receipts=accepted_round_receipts,
            )
        )
    elif (
        protocol.algorithm_id
        == RA_ADAPT_MACRO_GRADIENT_PHASE0_PROXY_NO_LANES_ALGORITHM_ID
    ):
        phase12_compile = (
            selector_compile_cost_accounting.get("phase_i_phase_ii")
            if isinstance(selector_compile_cost_accounting, Mapping)
            else None
        )
        if (
            not isinstance(selector_compile_cost_accounting, Mapping)
            or selector_compile_cost_accounting.get("schema")
            != "paper_i_selector_compile_cost_accounting_v1"
            or selector_compile_cost_accounting.get("scope")
            != BACKEND_COMPILE_SCOPE_SHARED_ALL_PHASES_V1
            or selector_compile_cost_accounting.get("excluded_from_s_alg")
            is not True
            or selector_compile_cost_accounting.get("phase0_cost_source")
            != "none_standard_adapt_absolute_gradient_v1"
            or selector_compile_cost_accounting.get("qiskit_applied_phases")
            != []
            or selector_compile_cost_accounting.get(
                "phase_iii_reuses_phase_i_phase_ii_oracle"
            )
            is not True
            or not isinstance(phase12_compile, Mapping)
            or phase12_compile.get("mode") != MARRAKESH_GRAPH_SPAN_MODE
            or selector_compile_cost_accounting.get("phase_iii") is not None
        ):
            raise RuntimeError(
                "The macro-only gradient-Phase-0 route lost its shared "
                "structural graph-span cost accounting."
            )
    elif protocol.algorithm_id in RA_ADAPT_PHASE3_QISKIT_ALGORITHM_IDS:
        phase12_compile = (
            selector_compile_cost_accounting.get("phase_i_phase_ii")
            if isinstance(selector_compile_cost_accounting, Mapping)
            else None
        )
        phase3_compile = (
            selector_compile_cost_accounting.get("phase_iii")
            if isinstance(selector_compile_cost_accounting, Mapping)
            else None
        )
        phase3_targets = (
            phase3_compile.get("targets")
            if isinstance(phase3_compile, Mapping)
            else None
        )
        if (
            not isinstance(selector_compile_cost_accounting, Mapping)
            or selector_compile_cost_accounting.get("schema")
            != "paper_i_selector_compile_cost_accounting_v1"
            or selector_compile_cost_accounting.get("scope")
            != BACKEND_COMPILE_SCOPE_PHASE3_QISKIT_ONLY_V1
            or selector_compile_cost_accounting.get("excluded_from_s_alg")
            is not True
            or not isinstance(phase12_compile, Mapping)
            or phase12_compile.get("mode") != MARRAKESH_GRAPH_SPAN_MODE
            or phase12_compile.get("preferred_backend_fallback_allowed")
            is not False
            or not isinstance(phase3_compile, Mapping)
            or phase3_compile.get("mode") != "transpile_single_v1"
            or phase3_compile.get("optimization_level") != 1
            or phase3_compile.get("seed_transpiler") != 7
            or phase3_compile.get("structure_theta_value") != 1.0
            or phase3_compile.get("negative_delta_reward_enabled") is not False
            or phase3_compile.get("preferred_backend_fallback_allowed")
            is not False
            or phase3_compile.get("one_qubit_coordinate_policy")
            != ONE_QUBIT_COORDINATE_COMPILED_POSITIVE_DELTA_V1
            or not isinstance(phase3_targets, list)
            or len(phase3_targets) != 1
            or not isinstance(phase3_targets[0], Mapping)
            or phase3_targets[0].get("resolved_name") != "FakeMarrakesh"
            or phase3_targets[0].get("resolution_kind") != "fake_exact"
        ):
            raise RuntimeError(
                "Canonical RA finalization lost the Phase-III-only Qiskit "
                "compile-oracle accounting."
            )
    elif protocol.algorithm_id in RA_ADAPT_PHASE23_QISKIT_ALGORITHM_IDS:
        gradient_phase0_algorithm = protocol.algorithm_id in {
            RA_ADAPT_MACRO_GRADIENT_PHASE0_MACRO_PHASE23_QISKIT_ALGORITHM_ID,
            RA_ADAPT_MACRO_GRADIENT_PHASE0_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID,
            RA_ADAPT_GLOBAL_SINGLETON_GRADIENT_PHASE0_PHASE23_QISKIT_ALGORITHM_ID,
            PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ALGORITHM_ID,
        }
        phase12_compile = (
            selector_compile_cost_accounting.get("phase_i_phase_ii")
            if isinstance(selector_compile_cost_accounting, Mapping)
            else None
        )
        phase23_compile = (
            selector_compile_cost_accounting.get("phase_iii")
            if isinstance(selector_compile_cost_accounting, Mapping)
            else None
        )
        phase23_targets = (
            phase23_compile.get("targets")
            if isinstance(phase23_compile, Mapping)
            else None
        )
        if (
            not isinstance(selector_compile_cost_accounting, Mapping)
            or selector_compile_cost_accounting.get("schema")
            != "paper_i_selector_compile_cost_accounting_v1"
            or selector_compile_cost_accounting.get("scope")
            != BACKEND_COMPILE_SCOPE_PHASE2_PHASE3_QISKIT_ONLY_V1
            or selector_compile_cost_accounting.get("excluded_from_s_alg")
            is not True
            or phase12_compile is not None
            or selector_compile_cost_accounting.get("phase_i_cost_source")
            != "structural_proxy_v1"
            or selector_compile_cost_accounting.get("phase0_cost_source")
            != (
                "none_standard_adapt_absolute_gradient_v1"
                if gradient_phase0_algorithm
                else None
            )
            or selector_compile_cost_accounting.get("qiskit_applied_phases")
            != ["phase_ii", "phase_iii"]
            or not isinstance(phase23_compile, Mapping)
            or phase23_compile.get("role") != "phase_ii_phase_iii"
            or phase23_compile.get("mode") != "transpile_single_v1"
            or phase23_compile.get("optimization_level") != 1
            or phase23_compile.get("seed_transpiler") != 7
            or phase23_compile.get("structure_theta_value") != 1.0
            or phase23_compile.get("negative_delta_reward_enabled") is not True
            or phase23_compile.get("preferred_backend_fallback_allowed")
            is not False
            or phase23_compile.get("one_qubit_coordinate_policy")
            != ONE_QUBIT_COORDINATE_COMPILED_POSITIVE_DELTA_V1
            or not isinstance(phase23_targets, list)
            or len(phase23_targets) != 1
            or not isinstance(phase23_targets[0], Mapping)
            or phase23_targets[0].get("resolved_name") != "FakeMarrakesh"
            or phase23_targets[0].get("resolution_kind") != "fake_exact"
        ):
            raise RuntimeError(
                "Canonical RA finalization lost the Phase-II/III Qiskit "
                "compile-oracle accounting."
            )
    deferred_fallback = _required_deferred_gram_fallback(finalization)
    numerical_physical_integrity = (
        build_ra_numerical_physical_integrity(
            run=completed.result,
            finalization=finalization,
        )
    )
    controller_replay_evidence = build_ra_controller_replay_evidence(
        protocol=protocol,
        run=completed.result,
        finalization=finalization,
    )
    study1_g8 = (
        build_study1_exact_reference_isolation_receipt(
            protocol=protocol,
            method="ra_adapt",
            finalized_controller_rounds=len(
                completed.result.accepted_trajectory
            ),
            exact_same_cutoff_energy=(
                completed.result.canonical_reporting.exact_same_cutoff_energy
            ),
        )
        if is_study1_protocol(protocol)
        else None
    )
    controller_noise_receipt = (
        _pure_hubbard_controller_noise_scientific_receipt(
            accepted_round_receipts=accepted_round_receipts,
            route_contract=route_contract,
        )
        if pure_hubbard_noise_application
        else None
    )
    return RAAdaptResult(
        schema=(
            RA_ADAPT_RESULT_SCHEMA_V2
            if protocol.algorithm_id == RA_ADAPT_ALGORITHM_ID
            else RA_ADAPT_RESULT_SCHEMA_V1
        ),
        protocol=protocol,
        selector_identity=RA_STAGED_SELECTOR_ID,
        parent_inventory=protocol.parent_inventory,
        executable_pool=protocol.executable_pool,
        policy=policy,
        run=completed.result,
        numerical_physical_integrity=numerical_physical_integrity,
        scientific_receipts={
            "route_contract": completed.result.route.to_dict(),
            "resolved_route_contract": dict(route_contract),
            "candidate_geometry_chart": EXACT_ORDERED_INSERTION_CHART,
            "trust_policy": protocol.trust_policy_id,
            "phase3_solver": PROJECTED_GENERALIZED_SOLVER,
            "accepted_refit_scope": FULL_ENLARGED_ACCEPTED_REFIT,
            "accepted_refit_coordinate_chart": (
                SUPPORTED_FS_WHITENED_REFIT_CHART
            ),
            **(
                {
                    "phase3_candidate_gain_policy": (
                        PHASE3_CANDIDATE_GAIN_JOINT_MINUS_ACTIVE_ONLY_V1
                    ),
                    "accepted_refit_initialization_policy": (
                        ROUTE_A_JOINT_STEP_WARM_START_EXACT_GUARDED_V1
                    ),
                    "active_response_policy": ACTIVE_GRADIENT_MEASURED,
                }
                if protocol.algorithm_id == RA_ADAPT_ALGORITHM_ID
                else {}
            ),
            "candidate_inventory_lineage": (
                candidate_inventory_lineage.to_dict()
            ),
            "accepted_round_receipts": accepted_round_receipts,
            **(
                {
                    "semantic_selector_accounting_closure": (
                        semantic_selector_accounting_closure
                    )
                }
                if semantic_selector_accounting_closure is not None
                else {}
            ),
            **(
                {
                    "terminal_phase3_selection_receipt": copy.deepcopy(
                        dict(
                            finalization[
                                "terminal_phase3_selection_receipt"
                            ]
                        )
                    )
                }
                if isinstance(
                    finalization.get("terminal_phase3_selection_receipt"),
                    Mapping,
                )
                else {}
            ),
            **(
                {"controller_noise": controller_noise_receipt}
                if controller_noise_receipt is not None
                else {}
            ),
            **(
                {
                    "selector_compile_cost_accounting": copy.deepcopy(
                        dict(selector_compile_cost_accounting)
                    )
                }
                if (
                    protocol.algorithm_id in RA_ADAPT_PHASE3_QISKIT_ALGORITHM_IDS
                    or protocol.algorithm_id
                    in RA_ADAPT_PHASE23_QISKIT_ALGORITHM_IDS
                    or protocol.algorithm_id
                    == RA_ADAPT_MACRO_GRADIENT_PHASE0_PROXY_NO_LANES_ALGORITHM_ID
                    or protocol.algorithm_id
                    in PAPER_I_RA_SEMANTIC_ALGORITHM_IDS
                )
                else {}
            ),
            **(
                {
                    "retained_support": latest_round["retained_support"],
                    "phase3_stabilization": latest_round[
                        "phase3_stabilization"
                    ],
                    "source_gram_no_overlap_trust": latest_round[
                        "source_gram_no_overlap_trust"
                    ],
                }
                if latest_round is not None
                else (
                    {
                        "terminal_phase0_selection_receipt": copy.deepcopy(
                            finalization[
                                "terminal_phase0_selection_receipt"
                            ]
                        )
                    }
                    if isinstance(
                        finalization.get(
                            "terminal_phase0_selection_receipt"
                        ),
                        Mapping,
                    )
                    else {}
                )
            ),
            **(
                {
                    "endpoint_overlap_trust": latest_round[
                        "endpoint_overlap_trust"
                    ]
                }
                if latest_round is not None
                and protocol.trust_policy_id
                == ENDPOINT_OVERLAP_DISPLACEMENT_TRUST
                else {}
            ),
            DEFERRED_GRAM_ALL_MODELS_INFEASIBLE_FALLBACK_V1: (
                deferred_fallback
            ),
            "controller_replay_evidence": controller_replay_evidence,
            "controller_replay_evidence_sha256": (
                controller_replay_evidence["sha256"]
            ),
            **(
                {
                    "study1_g8_exact_reference_isolation": (
                        study1_g8.to_dict()
                    )
                }
                if study1_g8 is not None
                else {}
            ),
            "numerical_physical_integrity": (
                numerical_physical_integrity.to_dict()
            ),
            "numerical_physical_integrity_sha256": canonical_sha256(
                numerical_physical_integrity
            ),
            "policy": policy.to_dict(),
        },
    )


__all__ = [
    "RA_ADAPT_ALGORITHM_ID",
    "RA_ADAPT_LEGACY_ALGORITHM_ID",
    "RA_ADAPT_ORDINARY_BUNDLE_ID",
    "RA_ADAPT_COMPILE_IDENTITY",
    "RA_ADAPT_ESTIMATOR_ACCOUNTING",
    "RA_ADAPT_GLOBAL_SINGLETON_QISKIT_COST_ALGORITHM_IDS",
    "RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_ALGORITHM_ID",
    "RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_DENOMINATOR_NO_LANES_ALGORITHM_ID",
    "RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_SOURCE_ALGORITHM_ID",
    "RA_ADAPT_GLOBAL_SINGLETON_GRADIENT_PHASE0_PHASE23_QISKIT_ALGORITHM_ID",
    "RA_ADAPT_GLOBAL_SINGLETON_GRADIENT_PHASE0_PHASE23_QISKIT_ROUTE_SUFFIX",
    "RA_ADAPT_MACRO_GRADIENT_PHASE0_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID",
    "RA_ADAPT_MACRO_GRADIENT_PHASE0_PHASE23_QISKIT_ROUTE_SUFFIX",
    "RA_ADAPT_MACRO_GRADIENT_PHASE0_MACRO_PHASE23_QISKIT_ALGORITHM_ID",
    "RA_ADAPT_MACRO_GRADIENT_PHASE0_MACRO_PHASE23_QISKIT_ROUTE_SUFFIX",
    "RA_ADAPT_MACRO_GRADIENT_PHASE0_PROXY_NO_LANES_ALGORITHM_ID",
    "RA_ADAPT_MACRO_GRADIENT_PHASE0_PROXY_NO_LANES_ROUTE_SUFFIX",
    "RA_ADAPT_MACRO_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID",
    "RA_ADAPT_PHASE23_QISKIT_ALGORITHM_IDS",
    "RA_ADAPT_GRADIENT_PHASE0_SHORTLIST_SIZE",
    "RA_ADAPT_MACRO_QISKIT_COST_INSERTION_KIND_BY_ALGORITHM_ID",
    "RA_ADAPT_QISKIT_COST_PHASE_REUSE",
    "RA_ADAPT_QISKIT_COST_POLICY",
    "RA_ADAPT_QISKIT_COST_ROUTE_SUFFIX",
    "RA_ADAPT_PHASE3_QISKIT_COST_PHASE_REUSE",
    "RA_ADAPT_PHASE3_QISKIT_COST_POLICY",
    "RA_ADAPT_PHASE3_QISKIT_COST_ROUTE_SUFFIX",
    "RA_ADAPT_PHASE3_QISKIT_DENOMINATOR_POLICY",
    "RA_ADAPT_PHASE3_QISKIT_DENOMINATOR_ROUTE_SUFFIX",
    "RA_ADAPT_ROUTE_CONTRACT_SCHEMA",
    "build_resolved_ra_protocol",
    "run_ra_adapt",
]
