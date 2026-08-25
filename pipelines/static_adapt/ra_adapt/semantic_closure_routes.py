"""Versioned native routes for the Paper-I cost-semantic closure.

The route family keeps Phase 0 independent from the later insertion policy.
All variants evaluate the same ordered append-endpoint generator-gradient
population and keep Qiskit and the Fubini--Study metric out of Phase 0.  The
four executable v2 arms independently vary the Phase-0 score and cardinality;
retired v1 identities remain provenance-only.  Phases I--III share one signed
full-trial Qiskit cost contract.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from pipelines.contracts.problem import (
    ProblemRequest,
    ResolvedProblemContext,
)
from pipelines.static_adapt.builders.problem_registry import (
    resolve_problem_context,
)
from pipelines.static_adapt.adaptive_phase_contracts import (
    ADAPTIVE_HORIZON_POLICY_EXACT_TARGET_V1,
    ADAPTIVE_HORIZON_POLICY_MAXIMUM_V1,
    ADAPTIVE_PHASE3_NO_POSITIVE_POLICY_FORCED_ADMISSION_V1,
    ADAPTIVE_PHASE3_NO_POSITIVE_POLICY_RAISE_V1,
    ADAPTIVE_PHASE3_NO_POSITIVE_POLICY_TYPED_TERMINAL_V1,
)
from pipelines.static_adapt.ra_adapt.adaptive_append_endpoint_shortlist import (
    ADAPTIVE_APPEND_ENDPOINT_SHORTLIST_POLICY,
    ADAPTIVE_PHASE0_ACTIVE_SCORE_SHORTLIST_POLICY_V2,
    AdaptivePhase0ActiveScore,
    AppendEndpointGeneratorScore,
    select_adaptive_phase0_active_score_shortlist,
    select_adaptive_append_endpoint_shortlist,
)
from pipelines.static_adapt.ra_adapt.adaptive_phase_shortlist import (
    ADAPTIVE_PHASE123_SHORTLIST_POLICY_V1,
    ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1,
    adaptive_phase_record_id,
    adaptive_phase_selection_receipt_from_mapping,
)
from pipelines.static_adapt.ra_adapt.adapters import (
    GlobalSinglePauliWordCandidateAdapter,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    ACTIVE_GRADIENT_STATIONARY,
    CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    RA_ADAPT_PHASE3_POPULATION_ALL_ROUNDS,
    RA_ADAPT_PHASE3_POPULATION_LATCHED_ON_PROGRESS_PLATEAU,
    RA_ADAPT_PHASE3_POPULATION_ON_INSERTION_PLATEAU,
    RA_ADAPT_PROTOCOL_SCHEMA,
    RA_STAGED_SELECTOR_ID,
    RAAdaptRequest,
    RESOURCE_WEIGHTING_ALL_PHASE,
    ResolvedRAAdaptProtocol,
    canonical_sha256,
)
from pipelines.static_adapt.ra_adapt.insertion_geometry import (
    APPEND_COMMUTATION_REDUCED_POLICY,
    validate_commutation_reduced_insertion_receipt,
)
from pipelines.static_adapt.hh_backend_compile_oracle import (
    BACKEND_COMPILE_SCOPE_PHASE123_QISKIT_V1,
    ONE_QUBIT_COORDINATE_COMPILED_POSITIVE_DELTA_V1,
)
from pipelines.static_adapt.sr_snake.contracts import (
    AlwaysCommutationReducedInsertion,
    AppendCommutationReducedInsertion,
    AppendOnlyInsertion,
    BeamOff,
    FreshStart,
    PlateauCommutationInsertion,
    PruningOff,
    SingletonAdmission,
    SRExecutionPolicy,
    SRMethodPolicy,
    SRObservationPolicy,
    SRStopPolicy,
)


PAPER_I_RA_SEMANTIC_IMPLEMENTATION_VERSION = (
    "paper_i_ra_phase0_proxy_ablation_phase123_qiskit_semantic_closure_v1"
)
PAPER_I_RA_SEMANTIC_IMPLEMENTATION_VERSION_V2 = (
    "paper_i_ra_phase0_score_cardinality_matrix_phase123_qiskit_"
    "semantic_closure_v2"
)
PAPER_I_RA_SEMANTIC_IMPLEMENTATION_VERSION_POSITION_V1 = (
    "paper_i_ra_phase0_placement_score_cardinality_matrix_phase123_qiskit_"
    "semantic_closure_v1"
)
PAPER_I_RA_ALL_PHASE_ADAPTIVE_IMPLEMENTATION_VERSION_V1 = (
    "paper_i_ra_all_phase_adaptive_gradient_phase0_phase123_qiskit_"
    "semantic_closure_v1"
)
PAPER_I_RA_ALL_PHASE_ADAPTIVE_NATURAL_TERMINAL_IMPLEMENTATION_VERSION_V2 = (
    "paper_i_ra_all_phase_adaptive_gradient_phase0_phase123_qiskit_"
    "natural_terminal_semantic_closure_v2"
)
PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_IMPLEMENTATION_VERSION_V1 = (
    "paper_i_ra_all_phase_adaptive_position_gradient_phase0_phase123_"
    "qiskit_semantic_closure_v1"
)
PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_NATURAL_TERMINAL_IMPLEMENTATION_VERSION_V2 = (
    "paper_i_ra_all_phase_adaptive_position_gradient_phase0_phase123_"
    "qiskit_natural_terminal_semantic_closure_v2"
)
PAPER_I_RA_ALL_PHASE_ADAPTIVE_FORCED_K50_IMPLEMENTATION_VERSION_V1 = (
    "paper_i_ra_all_phase_adaptive_gradient_phase0_phase123_qiskit_"
    "forced_admission_k50_semantic_closure_v1"
)
PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_FORCED_K50_IMPLEMENTATION_VERSION_V1 = (
    "paper_i_ra_all_phase_adaptive_position_gradient_phase0_phase123_"
    "qiskit_forced_admission_k50_semantic_closure_v1"
)
PAPER_I_RA_ALL_PHASE_ADAPTIVE_MIN_FLOORS_IMPLEMENTATION_VERSION_V1 = (
    "paper_i_ra_all_phase_adaptive_gradient_phase0_phase123_qiskit_"
    "min_floors_natural_terminal_semantic_closure_v1"
)
PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_MIN_FLOORS_IMPLEMENTATION_VERSION_V1 = (
    "paper_i_ra_all_phase_adaptive_position_gradient_phase0_phase123_"
    "qiskit_min_floors_natural_terminal_semantic_closure_v1"
)
PAPER_I_RA_SEMANTIC_ROUTE_CONTRACT_SCHEMA = (
    "paper_i_ra_phase0_proxy_ablation_phase123_qiskit_route_contract_v1"
)
PAPER_I_RA_SEMANTIC_SOURCE_INVENTORY_SCHEMA = (
    "paper_i_ra_semantic_closure_source_implementation_inventory_v1"
)
PAPER_I_RA_SEMANTIC_MATERIALIZATION_CONTRACT_SCHEMA = (
    "paper_i_ra_semantic_closure_native_materialization_contract_v1"
)
PAPER_I_RA_SEMANTIC_ADAPTER_ID = (
    "paper_i_ra_semantic_closure_global_singleton_candidate_adapter_v1"
)
PAPER_I_RA_SEMANTIC_NATIVE_BUNDLE_ID = (
    "paper_i_ra_phase0_proxy_ablation_phase123_qiskit_native_v1"
)
PAPER_I_RA_SEMANTIC_NATIVE_BUNDLE_ID_V2 = (
    "paper_i_ra_phase0_score_cardinality_matrix_phase123_qiskit_native_v2"
)
PAPER_I_RA_SEMANTIC_NATIVE_EIGHT_ARM_BUNDLE_ID_V1 = (
    "paper_i_ra_phase0_placement_score_cardinality_matrix_phase123_qiskit_"
    "native_v1"
)
PAPER_I_RA_ALL_PHASE_ADAPTIVE_NATIVE_BUNDLE_ID_V1 = (
    "paper_i_ra_all_phase_adaptive_gradient_phase0_phase123_qiskit_"
    "native_v1"
)
PAPER_I_RA_ALL_PHASE_ADAPTIVE_NATURAL_TERMINAL_NATIVE_BUNDLE_ID_V2 = (
    "paper_i_ra_all_phase_adaptive_gradient_phase0_phase123_qiskit_"
    "natural_terminal_native_v2"
)
PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_NATIVE_BUNDLE_ID_V1 = (
    "paper_i_ra_all_phase_adaptive_position_gradient_phase0_phase123_"
    "qiskit_native_v1"
)
PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_NATURAL_TERMINAL_NATIVE_BUNDLE_ID_V2 = (
    "paper_i_ra_all_phase_adaptive_position_gradient_phase0_phase123_"
    "qiskit_natural_terminal_native_v2"
)
PAPER_I_RA_ALL_PHASE_ADAPTIVE_FORCED_K50_NATIVE_BUNDLE_ID_V1 = (
    "paper_i_ra_all_phase_adaptive_gradient_phase0_phase123_qiskit_"
    "forced_admission_k50_native_v1"
)
PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_FORCED_K50_NATIVE_BUNDLE_ID_V1 = (
    "paper_i_ra_all_phase_adaptive_position_gradient_phase0_phase123_"
    "qiskit_forced_admission_k50_native_v1"
)
PAPER_I_RA_ALL_PHASE_ADAPTIVE_MIN_FLOORS_NATIVE_BUNDLE_ID_V1 = (
    "paper_i_ra_all_phase_adaptive_gradient_phase0_phase123_qiskit_"
    "min_floors_natural_terminal_native_v1"
)
PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_MIN_FLOORS_NATIVE_BUNDLE_ID_V1 = (
    "paper_i_ra_all_phase_adaptive_position_gradient_phase0_phase123_"
    "qiskit_min_floors_natural_terminal_native_v1"
)
PAPER_I_RA_PHASE123_QISKIT_COMPILE_SCOPE = (
    BACKEND_COMPILE_SCOPE_PHASE123_QISKIT_V1
)
PAPER_I_RA_PHASE123_QISKIT_COST_POLICY = (
    "qiskit_full_trial_ansatz_signed_marginal_phase1_phase2_phase3_v1"
)
PAPER_I_RA_PHASE123_QISKIT_CACHE_POLICY = (
    "reuse_only_exact_base_trial_candidate_position_identity_v1"
)
PAPER_I_RA_SEMANTIC_PHASE0_PROXY_RECEIPT_SCHEMA = (
    "paper_i_ra_semantic_closure_proxy_phase0_receipt_v1"
)
PAPER_I_RA_SEMANTIC_PHASE0_GRADIENT_ADAPTIVE_RECEIPT_SCHEMA_V2 = (
    "paper_i_ra_semantic_closure_gradient_adaptive_phase0_receipt_v2"
)
PAPER_I_RA_SEMANTIC_POSITION_PHASE0_RECEIPT_SCHEMA = (
    "paper_i_ra_semantic_closure_position_record_phase0_receipt_v1"
)
PAPER_I_RA_SEMANTIC_PHASE0_CONSUMER_SCOPE = (
    "phase0_append_endpoint_generator_gradient_surface_v1"
)
PAPER_I_RA_SEMANTIC_PHASE0_POPULATION_SCOPE = (
    "current_available_append_endpoint_generators_v1"
)
_PHASE0_GRAPH_COST_COMPONENTS = ("2q", "d", "1q", "theta", "shot")
_SEMANTIC_IMPLEMENTATION_SOURCE_PATHS = (
    ("semantic_route", "pipelines/static_adapt/ra_adapt/semantic_closure_routes.py"),
    ("problem_contract", "pipelines/contracts/problem.py"),
    ("adaptive_shortlist", "pipelines/static_adapt/ra_adapt/adaptive_append_endpoint_shortlist.py"),
    (
        "adaptive_phase123_shortlist",
        "pipelines/static_adapt/ra_adapt/adaptive_phase_shortlist.py",
    ),
    ("gradient_phase0", "pipelines/static_adapt/ra_adapt/phase0.py"),
    ("typed_adapter", "pipelines/static_adapt/ra_adapt/adapters.py"),
    ("executable_pool", "pipelines/static_adapt/ra_adapt/pools.py"),
    (
        "pool_resolution",
        "pipelines/static_adapt/builders/pool_resolution.py",
    ),
    (
        "problem_registry",
        "pipelines/static_adapt/builders/problem_registry.py",
    ),
    (
        "problem_setup",
        "pipelines/static_adapt/builders/problem_setup.py",
    ),
    ("typed_contracts", "pipelines/static_adapt/ra_adapt/contracts.py"),
    ("protocol_engine", "pipelines/static_adapt/ra_adapt/engine.py"),
    ("authority_materializer", "pipelines/static_adapt/ra_adapt/bundles.py"),
    ("runtime_facade", "pipelines/static_adapt/ra_adapt/runtime.py"),
    (
        "controller_replay_evidence",
        "pipelines/static_adapt/ra_adapt/replay_evidence.py",
    ),
    (
        "insertion_geometry",
        "pipelines/static_adapt/ra_adapt/insertion_geometry.py",
    ),
    ("selection_runtime", "pipelines/static_adapt/adapt_pipeline.py"),
    ("phase_shortlist_runtime", "pipelines/static_adapt/phase_shortlists.py"),
    (
        "candidate_record_cache_identity",
        "pipelines/static_adapt/adapt_candidate_record_cache.py",
    ),
    (
        "estimator_call_ledger",
        "pipelines/static_adapt/estimator_call_ledger.py",
    ),
    (
        "numerical_physical_integrity",
        "pipelines/static_adapt/numerical_physical_integrity.py",
    ),
    (
        "deferred_gram_fallback",
        "pipelines/static_adapt/deferred_gram_fallback.py",
    ),
    ("route_context", "pipelines/static_adapt/sr_snake/_context.py"),
    ("resume_hydration", "pipelines/static_adapt/sr_snake/_resume.py"),
    (
        "observation_projection",
        "pipelines/static_adapt/sr_snake/_observation.py",
    ),
    ("controller_runtime", "pipelines/static_adapt/sr_snake/_controller.py"),
    ("selection_contract", "pipelines/static_adapt/sr_snake/_selection.py"),
    ("transition_runtime", "pipelines/static_adapt/sr_snake/_transition.py"),
    ("sr_contracts", "pipelines/static_adapt/sr_snake/contracts.py"),
    (
        "route_profile_runtime",
        "pipelines/static_adapt/sr_snake_route_profile.py",
    ),
    (
        "current_checkpoint_io",
        "pipelines/static_adapt/current_checkpoint.py",
    ),
    (
        "resume_checkpoint_scaffold",
        "pipelines/static_adapt/resume_scaffold.py",
    ),
    ("qiskit_compile_oracle", "pipelines/static_adapt/hh_backend_compile_oracle.py"),
    (
        "qiskit_ansatz_circuit_builder",
        "pipelines/hardcoded/adapt_circuit_execution.py",
    ),
    ("qiskit_backend_tools", "pipelines/qiskit_backend_tools.py"),
    ("signed_cost_consumer", "pipelines/scaffold/hh_continuation_scoring.py"),
    ("scoring_types", "pipelines/scaffold/hh_continuation_types.py"),
    (
        "phase_controller_thresholds",
        "pipelines/scaffold/hh_continuation_stage_control.py",
    ),
    (
        "candidate_generator_semantics",
        "pipelines/scaffold/hh_continuation_generators.py",
    ),
    (
        "candidate_symmetry_semantics",
        "pipelines/scaffold/hh_continuation_symmetry.py",
    ),
    (
        "candidate_motif_semantics",
        "pipelines/scaffold/hh_continuation_motifs.py",
    ),
    ("nested_phase_windows", "pipelines/static_adapt/nested_windows.py"),
    (
        "measurement_proxy",
        "pipelines/static_adapt/selector_measurement_proxy.py",
    ),
    (
        "commutation_metadata",
        "pipelines/static_adapt/commutation_metadata.py",
    ),
    ("compiled_ansatz", "src/quantum/compiled_ansatz.py"),
    ("compiled_polynomial", "src/quantum/compiled_polynomial.py"),
    ("pauli_actions", "src/quantum/pauli_actions.py"),
    (
        "ansatz_parameterization",
        "src/quantum/ansatz_parameterization.py",
    ),
    ("ansatz_terms", "src/quantum/vqe_latex_python_pairs.py"),
    (
        "hh_hamiltonian_builder",
        "src/quantum/hubbard_latex_python_pairs.py",
    ),
    (
        "hh_exact_diagonalization",
        "src/quantum/ed_hubbard_holstein.py",
    ),
    (
        "pauli_polynomial_representation",
        "src/quantum/pauli_polynomial_class.py",
    ),
    (
        "pauli_term_representation",
        "src/quantum/qubitization_module.py",
    ),
)
_SEMANTIC_IMPLEMENTATION_SOURCE_ROOTS = (
    "pipelines",
    "src",
)
_SEMANTIC_IMPLEMENTATION_EXTRA_SOURCE_PATHS: tuple[str, ...] = ()

PAPER_I_RA_PHASE0_GRADIENT_FIXED24 = "gradient_only_fixed24_v1"
PAPER_I_RA_PHASE0_PROXY_FIXED24_SHADOW = (
    "structural_proxy_cost_fixed24_adaptive_shadow_v1"
)
PAPER_I_RA_PHASE0_PROXY_ADAPTIVE = (
    "structural_proxy_cost_adaptive_shortlist_v1"
)
PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2 = "gradient_only_fixed24_v2"
PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2 = (
    "gradient_only_adaptive_shortlist_v2"
)
PAPER_I_RA_PHASE0_PROXY_FIXED24_V2 = "structural_proxy_cost_fixed24_v2"
PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2 = (
    "structural_proxy_cost_adaptive_shortlist_v2"
)
PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1 = (
    "gradient_only_adaptive_shortlist_phase123_adaptive_v1"
)
PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2 = (
    "gradient_only_adaptive_shortlist_phase123_adaptive_natural_terminal_v2"
)
PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1 = (
    "position_records_gradient_only_adaptive_shortlist_phase123_adaptive_v1"
)
PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2 = (
    "position_records_gradient_only_adaptive_shortlist_phase123_adaptive_"
    "natural_terminal_v2"
)
PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_FORCED_K50_V1 = (
    "gradient_only_adaptive_shortlist_phase123_adaptive_"
    "forced_admission_k50_v1"
)
PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_FORCED_K50_V1 = (
    "position_records_gradient_only_adaptive_shortlist_phase123_adaptive_"
    "forced_admission_k50_v1"
)
PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1 = (
    "gradient_only_adaptive_shortlist_phase123_adaptive_min_floors_"
    "natural_terminal_v1"
)
PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1 = (
    "position_records_gradient_only_adaptive_shortlist_phase123_adaptive_"
    "min_floors_natural_terminal_v1"
)
PAPER_I_RA_PHASE_SHORTLIST_MINIMUMS_V1 = {
    "phase0": 10,
    "phase_i": 7,
    "phase_ii": 4,
}
PAPER_I_RA_PHASE0_POSITION_GRADIENT_FIXED24_V1 = (
    "position_records_gradient_only_fixed24_v1"
)
PAPER_I_RA_PHASE0_POSITION_GRADIENT_ADAPTIVE_V1 = (
    "position_records_gradient_only_adaptive_shortlist_v1"
)
PAPER_I_RA_PHASE0_POSITION_PROXY_FIXED24_V1 = (
    "position_records_structural_proxy_cost_fixed24_v1"
)
PAPER_I_RA_PHASE0_POSITION_PROXY_ADAPTIVE_V1 = (
    "position_records_structural_proxy_cost_adaptive_shortlist_v1"
)
PAPER_I_RA_PHASE0_LEGACY_ROUTE_VARIANTS = frozenset(
    {
        PAPER_I_RA_PHASE0_GRADIENT_FIXED24,
        PAPER_I_RA_PHASE0_PROXY_FIXED24_SHADOW,
        PAPER_I_RA_PHASE0_PROXY_ADAPTIVE,
    }
)
PAPER_I_RA_PHASE0_V2_ROUTE_VARIANTS = frozenset(
    {
        PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2,
        PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2,
        PAPER_I_RA_PHASE0_PROXY_FIXED24_V2,
        PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2,
    }
)
PAPER_I_RA_PHASE0_POSITION_ROUTE_VARIANTS = frozenset(
    {
        PAPER_I_RA_PHASE0_POSITION_GRADIENT_FIXED24_V1,
        PAPER_I_RA_PHASE0_POSITION_GRADIENT_ADAPTIVE_V1,
        PAPER_I_RA_PHASE0_POSITION_PROXY_FIXED24_V1,
        PAPER_I_RA_PHASE0_POSITION_PROXY_ADAPTIVE_V1,
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1,
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_FORCED_K50_V1,
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
    }
)
PAPER_I_RA_ALL_PHASE_ADAPTIVE_ROUTE_VARIANTS = frozenset(
    {
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1,
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_FORCED_K50_V1,
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1,
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_FORCED_K50_V1,
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
    }
)
PAPER_I_RA_PHASE_SHORTLIST_MIN_FLOORS_ROUTE_VARIANTS = frozenset(
    {
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
    }
)
PAPER_I_RA_PHASE3_FORCED_ADMISSION_ROUTE_VARIANTS = frozenset(
    {
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_FORCED_K50_V1,
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_FORCED_K50_V1,
    }
)
PAPER_I_RA_PHASE3_NATURAL_TERMINAL_ROUTE_VARIANTS = frozenset(
    {
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
    }
)
PAPER_I_RA_PHASE0_EXECUTABLE_ROUTE_VARIANTS = frozenset(
    PAPER_I_RA_PHASE0_V2_ROUTE_VARIANTS
    | PAPER_I_RA_PHASE0_POSITION_ROUTE_VARIANTS
    | PAPER_I_RA_ALL_PHASE_ADAPTIVE_ROUTE_VARIANTS
)
PAPER_I_RA_SEMANTIC_ROUTE_VARIANTS = frozenset(
    PAPER_I_RA_PHASE0_LEGACY_ROUTE_VARIANTS
    | PAPER_I_RA_PHASE0_EXECUTABLE_ROUTE_VARIANTS
)


@dataclass(frozen=True, slots=True)
class PaperIRASemanticClosureRouteIdentity:
    route_variant: str
    algorithm_id: str
    route_id: str
    route_profile: str
    semantic_implementation_version: str


_ROUTE_IDENTITIES: Mapping[str, PaperIRASemanticClosureRouteIdentity] = {
    PAPER_I_RA_PHASE0_GRADIENT_FIXED24: (
        PaperIRASemanticClosureRouteIdentity(
            route_variant=PAPER_I_RA_PHASE0_GRADIENT_FIXED24,
            algorithm_id=(
                "paper_i_ra_global_singleton_append_endpoint_gradient_"
                "fixed24_qiskit_phase123_semantic_closure_v1"
            ),
            route_id=(
                "paper_i_ra_phase0_gradient_fixed24_phase123_qiskit_"
                "semantic_closure_v1"
            ),
            route_profile=(
                "paper_i_ra__global_singleton__phase0_gradient_fixed24__"
                "qiskit_phase123_signed__semantic_closure_v1"
            ),
            semantic_implementation_version=(
                PAPER_I_RA_SEMANTIC_IMPLEMENTATION_VERSION
            ),
        )
    ),
    PAPER_I_RA_PHASE0_PROXY_FIXED24_SHADOW: (
        PaperIRASemanticClosureRouteIdentity(
            route_variant=PAPER_I_RA_PHASE0_PROXY_FIXED24_SHADOW,
            algorithm_id=(
                "paper_i_ra_global_singleton_append_endpoint_proxy_cost_"
                "fixed24_qiskit_phase123_semantic_closure_v1"
            ),
            route_id=(
                "paper_i_ra_phase0_proxy_cost_fixed24_phase123_qiskit_"
                "semantic_closure_v1"
            ),
            route_profile=(
                "paper_i_ra__global_singleton__phase0_proxy_cost_fixed24__"
                "qiskit_phase123_signed__semantic_closure_v1"
            ),
            semantic_implementation_version=(
                PAPER_I_RA_SEMANTIC_IMPLEMENTATION_VERSION
            ),
        )
    ),
    PAPER_I_RA_PHASE0_PROXY_ADAPTIVE: (
        PaperIRASemanticClosureRouteIdentity(
            route_variant=PAPER_I_RA_PHASE0_PROXY_ADAPTIVE,
            algorithm_id=(
                "paper_i_ra_global_singleton_append_endpoint_proxy_cost_"
                "adaptive_qiskit_phase123_semantic_closure_v1"
            ),
            route_id=(
                "paper_i_ra_phase0_proxy_cost_adaptive_phase123_qiskit_"
                "semantic_closure_v1"
            ),
            route_profile=(
                "paper_i_ra__global_singleton__phase0_proxy_cost_adaptive__"
                "qiskit_phase123_signed__semantic_closure_v1"
            ),
            semantic_implementation_version=(
                PAPER_I_RA_SEMANTIC_IMPLEMENTATION_VERSION
            ),
        )
    ),
    PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2: (
        PaperIRASemanticClosureRouteIdentity(
            route_variant=PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2,
            algorithm_id=(
                "paper_i_ra_global_singleton_append_endpoint_gradient_"
                "fixed24_qiskit_phase123_semantic_closure_v2"
            ),
            route_id=(
                "paper_i_ra_phase0_gradient_fixed24_phase123_qiskit_"
                "semantic_closure_v2"
            ),
            route_profile=(
                "paper_i_ra__global_singleton__phase0_gradient_fixed24__"
                "qiskit_phase123_signed__semantic_closure_v2"
            ),
            semantic_implementation_version=(
                PAPER_I_RA_SEMANTIC_IMPLEMENTATION_VERSION_V2
            ),
        )
    ),
    PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2: (
        PaperIRASemanticClosureRouteIdentity(
            route_variant=PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2,
            algorithm_id=(
                "paper_i_ra_global_singleton_append_endpoint_gradient_"
                "adaptive_qiskit_phase123_semantic_closure_v2"
            ),
            route_id=(
                "paper_i_ra_phase0_gradient_adaptive_phase123_qiskit_"
                "semantic_closure_v2"
            ),
            route_profile=(
                "paper_i_ra__global_singleton__phase0_gradient_adaptive__"
                "qiskit_phase123_signed__semantic_closure_v2"
            ),
            semantic_implementation_version=(
                PAPER_I_RA_SEMANTIC_IMPLEMENTATION_VERSION_V2
            ),
        )
    ),
    PAPER_I_RA_PHASE0_PROXY_FIXED24_V2: (
        PaperIRASemanticClosureRouteIdentity(
            route_variant=PAPER_I_RA_PHASE0_PROXY_FIXED24_V2,
            algorithm_id=(
                "paper_i_ra_global_singleton_append_endpoint_proxy_cost_"
                "fixed24_qiskit_phase123_semantic_closure_v2"
            ),
            route_id=(
                "paper_i_ra_phase0_proxy_cost_fixed24_phase123_qiskit_"
                "semantic_closure_v2"
            ),
            route_profile=(
                "paper_i_ra__global_singleton__phase0_proxy_cost_fixed24__"
                "qiskit_phase123_signed__semantic_closure_v2"
            ),
            semantic_implementation_version=(
                PAPER_I_RA_SEMANTIC_IMPLEMENTATION_VERSION_V2
            ),
        )
    ),
    PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2: (
        PaperIRASemanticClosureRouteIdentity(
            route_variant=PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2,
            algorithm_id=(
                "paper_i_ra_global_singleton_append_endpoint_proxy_cost_"
                "adaptive_qiskit_phase123_semantic_closure_v2"
            ),
            route_id=(
                "paper_i_ra_phase0_proxy_cost_adaptive_phase123_qiskit_"
                "semantic_closure_v2"
            ),
            route_profile=(
                "paper_i_ra__global_singleton__phase0_proxy_cost_adaptive__"
                "qiskit_phase123_signed__semantic_closure_v2"
            ),
            semantic_implementation_version=(
                PAPER_I_RA_SEMANTIC_IMPLEMENTATION_VERSION_V2
            ),
        )
    ),
    PAPER_I_RA_PHASE0_POSITION_GRADIENT_FIXED24_V1: (
        PaperIRASemanticClosureRouteIdentity(
            route_variant=PAPER_I_RA_PHASE0_POSITION_GRADIENT_FIXED24_V1,
            algorithm_id=(
                "paper_i_ra_global_singleton_position_records_gradient_"
                "fixed24_qiskit_phase123_semantic_closure_v1"
            ),
            route_id=(
                "paper_i_ra_phase0_position_gradient_fixed24_phase123_"
                "qiskit_semantic_closure_v1"
            ),
            route_profile=(
                "paper_i_ra__global_singleton__phase0_position_gradient_"
                "fixed24__qiskit_phase123_signed__semantic_closure_v1"
            ),
            semantic_implementation_version=(
                PAPER_I_RA_SEMANTIC_IMPLEMENTATION_VERSION_POSITION_V1
            ),
        )
    ),
    PAPER_I_RA_PHASE0_POSITION_GRADIENT_ADAPTIVE_V1: (
        PaperIRASemanticClosureRouteIdentity(
            route_variant=PAPER_I_RA_PHASE0_POSITION_GRADIENT_ADAPTIVE_V1,
            algorithm_id=(
                "paper_i_ra_global_singleton_position_records_gradient_"
                "adaptive_qiskit_phase123_semantic_closure_v1"
            ),
            route_id=(
                "paper_i_ra_phase0_position_gradient_adaptive_phase123_"
                "qiskit_semantic_closure_v1"
            ),
            route_profile=(
                "paper_i_ra__global_singleton__phase0_position_gradient_"
                "adaptive__qiskit_phase123_signed__semantic_closure_v1"
            ),
            semantic_implementation_version=(
                PAPER_I_RA_SEMANTIC_IMPLEMENTATION_VERSION_POSITION_V1
            ),
        )
    ),
    PAPER_I_RA_PHASE0_POSITION_PROXY_FIXED24_V1: (
        PaperIRASemanticClosureRouteIdentity(
            route_variant=PAPER_I_RA_PHASE0_POSITION_PROXY_FIXED24_V1,
            algorithm_id=(
                "paper_i_ra_global_singleton_position_records_proxy_cost_"
                "fixed24_qiskit_phase123_semantic_closure_v1"
            ),
            route_id=(
                "paper_i_ra_phase0_position_proxy_cost_fixed24_phase123_"
                "qiskit_semantic_closure_v1"
            ),
            route_profile=(
                "paper_i_ra__global_singleton__phase0_position_proxy_cost_"
                "fixed24__qiskit_phase123_signed__semantic_closure_v1"
            ),
            semantic_implementation_version=(
                PAPER_I_RA_SEMANTIC_IMPLEMENTATION_VERSION_POSITION_V1
            ),
        )
    ),
    PAPER_I_RA_PHASE0_POSITION_PROXY_ADAPTIVE_V1: (
        PaperIRASemanticClosureRouteIdentity(
            route_variant=PAPER_I_RA_PHASE0_POSITION_PROXY_ADAPTIVE_V1,
            algorithm_id=(
                "paper_i_ra_global_singleton_position_records_proxy_cost_"
                "adaptive_qiskit_phase123_semantic_closure_v1"
            ),
            route_id=(
                "paper_i_ra_phase0_position_proxy_cost_adaptive_phase123_"
                "qiskit_semantic_closure_v1"
            ),
            route_profile=(
                "paper_i_ra__global_singleton__phase0_position_proxy_cost_"
                "adaptive__qiskit_phase123_signed__semantic_closure_v1"
            ),
            semantic_implementation_version=(
                PAPER_I_RA_SEMANTIC_IMPLEMENTATION_VERSION_POSITION_V1
            ),
        )
    ),
    PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1: (
        PaperIRASemanticClosureRouteIdentity(
            route_variant=PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1,
            algorithm_id=(
                "paper_i_ra_global_singleton_append_endpoint_gradient_"
                "adaptive_phase123_adaptive_qiskit_semantic_closure_v1"
            ),
            route_id=(
                "paper_i_ra_phase0_gradient_adaptive_phase123_adaptive_"
                "qiskit_semantic_closure_v1"
            ),
            route_profile=(
                "paper_i_ra__global_singleton__phase0_gradient_adaptive__"
                "phase123_adaptive__qiskit_signed__semantic_closure_v1"
            ),
            semantic_implementation_version=(
                PAPER_I_RA_ALL_PHASE_ADAPTIVE_IMPLEMENTATION_VERSION_V1
            ),
        )
    ),
    PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2: (
        PaperIRASemanticClosureRouteIdentity(
            route_variant=(
                PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2
            ),
            algorithm_id=(
                "paper_i_ra_global_singleton_append_endpoint_gradient_"
                "adaptive_phase123_adaptive_qiskit_natural_terminal_"
                "semantic_closure_v2"
            ),
            route_id=(
                "paper_i_ra_phase0_gradient_adaptive_phase123_adaptive_"
                "qiskit_natural_terminal_semantic_closure_v2"
            ),
            route_profile=(
                "paper_i_ra__global_singleton__phase0_gradient_adaptive__"
                "phase123_adaptive__qiskit_signed__natural_terminal__"
                "semantic_closure_v2"
            ),
            semantic_implementation_version=(
                PAPER_I_RA_ALL_PHASE_ADAPTIVE_NATURAL_TERMINAL_IMPLEMENTATION_VERSION_V2
            ),
        )
    ),
    PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1: (
        PaperIRASemanticClosureRouteIdentity(
            route_variant=(
                PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1
            ),
            algorithm_id=(
                "paper_i_ra_global_singleton_position_records_gradient_"
                "adaptive_phase123_adaptive_qiskit_semantic_closure_v1"
            ),
            route_id=(
                "paper_i_ra_phase0_position_gradient_adaptive_phase123_"
                "adaptive_qiskit_semantic_closure_v1"
            ),
            route_profile=(
                "paper_i_ra__global_singleton__phase0_position_gradient_"
                "adaptive__phase123_adaptive__qiskit_signed__semantic_"
                "closure_v1"
            ),
            semantic_implementation_version=(
                PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_IMPLEMENTATION_VERSION_V1
            ),
        )
    ),
    PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2: (
        PaperIRASemanticClosureRouteIdentity(
            route_variant=(
                PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2
            ),
            algorithm_id=(
                "paper_i_ra_global_singleton_position_records_gradient_"
                "adaptive_phase123_adaptive_qiskit_natural_terminal_"
                "semantic_closure_v2"
            ),
            route_id=(
                "paper_i_ra_phase0_position_gradient_adaptive_phase123_"
                "adaptive_qiskit_natural_terminal_semantic_closure_v2"
            ),
            route_profile=(
                "paper_i_ra__global_singleton__phase0_position_gradient_"
                "adaptive__phase123_adaptive__qiskit_signed__natural_"
                "terminal__semantic_closure_v2"
            ),
            semantic_implementation_version=(
                PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_NATURAL_TERMINAL_IMPLEMENTATION_VERSION_V2
            ),
        )
    ),
    PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_FORCED_K50_V1: (
        PaperIRASemanticClosureRouteIdentity(
            route_variant=(
                PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_FORCED_K50_V1
            ),
            algorithm_id=(
                "paper_i_ra_global_singleton_append_endpoint_gradient_"
                "adaptive_phase123_adaptive_qiskit_forced_admission_k50_"
                "semantic_closure_v1"
            ),
            route_id=(
                "paper_i_ra_phase0_gradient_adaptive_phase123_adaptive_"
                "qiskit_forced_admission_k50_semantic_closure_v1"
            ),
            route_profile=(
                "paper_i_ra__global_singleton__phase0_gradient_adaptive__"
                "phase123_adaptive__qiskit_signed__forced_admission_k50__"
                "semantic_closure_v1"
            ),
            semantic_implementation_version=(
                PAPER_I_RA_ALL_PHASE_ADAPTIVE_FORCED_K50_IMPLEMENTATION_VERSION_V1
            ),
        )
    ),
    PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1: (
        PaperIRASemanticClosureRouteIdentity(
            route_variant=(
                PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1
            ),
            algorithm_id=(
                "paper_i_ra_global_singleton_append_endpoint_gradient_"
                "adaptive_phase123_adaptive_qiskit_min_floors_natural_"
                "terminal_semantic_closure_v1"
            ),
            route_id=(
                "paper_i_ra_phase0_gradient_adaptive_phase123_adaptive_"
                "qiskit_min_floors_natural_terminal_semantic_closure_v1"
            ),
            route_profile=(
                "paper_i_ra__global_singleton__phase0_gradient_adaptive__"
                "phase123_adaptive__qiskit_signed__min_floors__natural_"
                "terminal__semantic_closure_v1"
            ),
            semantic_implementation_version=(
                PAPER_I_RA_ALL_PHASE_ADAPTIVE_MIN_FLOORS_IMPLEMENTATION_VERSION_V1
            ),
        )
    ),
    PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1: (
        PaperIRASemanticClosureRouteIdentity(
            route_variant=(
                PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1
            ),
            algorithm_id=(
                "paper_i_ra_global_singleton_position_records_gradient_"
                "adaptive_phase123_adaptive_qiskit_min_floors_natural_"
                "terminal_semantic_closure_v1"
            ),
            route_id=(
                "paper_i_ra_phase0_position_gradient_adaptive_phase123_"
                "adaptive_qiskit_min_floors_natural_terminal_semantic_"
                "closure_v1"
            ),
            route_profile=(
                "paper_i_ra__global_singleton__phase0_position_gradient_"
                "adaptive__phase123_adaptive__qiskit_signed__min_floors__"
                "natural_terminal__semantic_closure_v1"
            ),
            semantic_implementation_version=(
                PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_MIN_FLOORS_IMPLEMENTATION_VERSION_V1
            ),
        )
    ),
    PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_FORCED_K50_V1: (
        PaperIRASemanticClosureRouteIdentity(
            route_variant=(
                PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_FORCED_K50_V1
            ),
            algorithm_id=(
                "paper_i_ra_global_singleton_position_records_gradient_"
                "adaptive_phase123_adaptive_qiskit_forced_admission_k50_"
                "semantic_closure_v1"
            ),
            route_id=(
                "paper_i_ra_phase0_position_gradient_adaptive_phase123_"
                "adaptive_qiskit_forced_admission_k50_semantic_closure_v1"
            ),
            route_profile=(
                "paper_i_ra__global_singleton__phase0_position_gradient_"
                "adaptive__phase123_adaptive__qiskit_signed__forced_"
                "admission_k50__semantic_closure_v1"
            ),
            semantic_implementation_version=(
                PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_FORCED_K50_IMPLEMENTATION_VERSION_V1
            ),
        )
    ),
}

PAPER_I_RA_SEMANTIC_ALGORITHM_IDS = frozenset(
    identity.algorithm_id for identity in _ROUTE_IDENTITIES.values()
)


def semantic_closure_route_identity(
    route_variant: str,
) -> PaperIRASemanticClosureRouteIdentity:
    try:
        return _ROUTE_IDENTITIES[str(route_variant)]
    except KeyError as exc:
        raise ValueError("Unknown Paper-I semantic-closure route variant.") from exc


def semantic_closure_route_identity_from_algorithm(
    algorithm_id: str,
) -> PaperIRASemanticClosureRouteIdentity:
    matches = tuple(
        identity
        for identity in _ROUTE_IDENTITIES.values()
        if identity.algorithm_id == str(algorithm_id)
    )
    if len(matches) != 1:
        raise ValueError("Unknown Paper-I semantic-closure algorithm identity.")
    return matches[0]


def semantic_phase3_no_positive_policy(route_variant: str) -> str:
    """Resolve the authenticated Phase-III exhaustion policy for one route."""

    identity = semantic_closure_route_identity(route_variant)
    if (
        identity.route_variant
        in PAPER_I_RA_PHASE3_NATURAL_TERMINAL_ROUTE_VARIANTS
    ):
        return ADAPTIVE_PHASE3_NO_POSITIVE_POLICY_TYPED_TERMINAL_V1
    if (
        identity.route_variant
        in PAPER_I_RA_PHASE3_FORCED_ADMISSION_ROUTE_VARIANTS
    ):
        return ADAPTIVE_PHASE3_NO_POSITIVE_POLICY_FORCED_ADMISSION_V1
    return ADAPTIVE_PHASE3_NO_POSITIVE_POLICY_RAISE_V1


def semantic_phase_shortlist_minimums(
    route_variant: str,
) -> dict[str, int] | None:
    """Resolve the declared minimum-retention floors for one route."""

    identity = semantic_closure_route_identity(route_variant)
    if (
        identity.route_variant
        in PAPER_I_RA_PHASE_SHORTLIST_MIN_FLOORS_ROUTE_VARIANTS
    ):
        return dict(PAPER_I_RA_PHASE_SHORTLIST_MINIMUMS_V1)
    return None


def semantic_phase0_minimum_retained(route_variant: str) -> int:
    """Resolve the Phase-0 minimum-retention floor for one route."""

    minimums = semantic_phase_shortlist_minimums(route_variant)
    return int(minimums.get("phase0", 0)) if minimums else 0


def semantic_controller_horizon_policy(route_variant: str) -> str:
    """Resolve exact-target versus maximum-horizon completion semantics."""

    identity = semantic_closure_route_identity(route_variant)
    return (
        ADAPTIVE_HORIZON_POLICY_MAXIMUM_V1
        if identity.route_variant
        in PAPER_I_RA_PHASE3_NATURAL_TERMINAL_ROUTE_VARIANTS
        else ADAPTIVE_HORIZON_POLICY_EXACT_TARGET_V1
    )


def semantic_closure_native_bundle_id(route_variant: str) -> str:
    """Resolve the provenance-only v1 or executable v2 bundle identity."""

    identity = semantic_closure_route_identity(route_variant)
    if identity.route_variant == (
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2
    ):
        return PAPER_I_RA_ALL_PHASE_ADAPTIVE_NATURAL_TERMINAL_NATIVE_BUNDLE_ID_V2
    if identity.route_variant == (
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2
    ):
        return (
            PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_NATURAL_TERMINAL_NATIVE_BUNDLE_ID_V2
        )
    if identity.route_variant == (
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1
    ):
        return PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_NATIVE_BUNDLE_ID_V1
    if identity.route_variant == PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1:
        return PAPER_I_RA_ALL_PHASE_ADAPTIVE_NATIVE_BUNDLE_ID_V1
    if identity.route_variant == (
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_FORCED_K50_V1
    ):
        return PAPER_I_RA_ALL_PHASE_ADAPTIVE_FORCED_K50_NATIVE_BUNDLE_ID_V1
    if identity.route_variant == (
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_FORCED_K50_V1
    ):
        return (
            PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_FORCED_K50_NATIVE_BUNDLE_ID_V1
        )
    if identity.route_variant == (
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1
    ):
        return PAPER_I_RA_ALL_PHASE_ADAPTIVE_MIN_FLOORS_NATIVE_BUNDLE_ID_V1
    if identity.route_variant == (
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1
    ):
        return (
            PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_MIN_FLOORS_NATIVE_BUNDLE_ID_V1
        )
    return (
        PAPER_I_RA_SEMANTIC_NATIVE_EIGHT_ARM_BUNDLE_ID_V1
        if identity.route_variant in PAPER_I_RA_PHASE0_EXECUTABLE_ROUTE_VARIANTS
        else PAPER_I_RA_SEMANTIC_NATIVE_BUNDLE_ID
    )


@dataclass(frozen=True)
class PaperIRASemanticClosureGlobalSingletonCandidateAdapter(
    GlobalSinglePauliWordCandidateAdapter
):
    """Named global-singleton adapter carrying one Phase-0 route identity."""

    route_variant: str = PAPER_I_RA_PHASE0_GRADIENT_FIXED24
    semantic_implementation_version: str = ""
    candidate_representation_id: str = CANDIDATE_REPRESENTATION_SINGLE_PAULI
    adapter_id: str = PAPER_I_RA_SEMANTIC_ADAPTER_ID

    def __post_init__(self) -> None:
        identity = semantic_closure_route_identity(self.route_variant)
        if not self.semantic_implementation_version:
            object.__setattr__(
                self,
                "semantic_implementation_version",
                identity.semantic_implementation_version,
            )
        if self.semantic_implementation_version != (
            identity.semantic_implementation_version
        ):
            raise ValueError(
                "Paper-I semantic implementation version drifted."
            )
        if (
            self.candidate_representation_id
            != CANDIDATE_REPRESENTATION_SINGLE_PAULI
            or self.adapter_id != PAPER_I_RA_SEMANTIC_ADAPTER_ID
        ):
            raise ValueError(
                "Paper-I semantic-closure adapter identity fields are fixed."
            )

    @property
    def algorithm_id(self) -> str:
        return semantic_closure_route_identity(
            self.route_variant
        ).algorithm_id

    @property
    def phase0_shortlist_policy_id(self) -> str:
        if self.route_variant in {
            PAPER_I_RA_PHASE0_GRADIENT_FIXED24,
            PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2,
        }:
            return "global_singleton_absolute_gradient_shortlist_v1"
        if self.route_variant == PAPER_I_RA_PHASE0_PROXY_FIXED24_SHADOW:
            return "append_endpoint_graph_weighted_fixed24_adaptive_shadow_v1"
        if self.route_variant == PAPER_I_RA_PHASE0_PROXY_ADAPTIVE:
            return ADAPTIVE_APPEND_ENDPOINT_SHORTLIST_POLICY
        if self.route_variant == PAPER_I_RA_PHASE0_PROXY_FIXED24_V2:
            return "append_endpoint_graph_weighted_fixed24_v2"
        if self.route_variant in {
            PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2,
            PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2,
            PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1,
            PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
            PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_FORCED_K50_V1,
            PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
        }:
            return ADAPTIVE_PHASE0_ACTIVE_SCORE_SHORTLIST_POLICY_V2
        if self.route_variant in {
            PAPER_I_RA_PHASE0_POSITION_GRADIENT_FIXED24_V1,
            PAPER_I_RA_PHASE0_POSITION_PROXY_FIXED24_V1,
        }:
            return "position_record_active_score_fixed24_v1"
        if self.route_variant in PAPER_I_RA_PHASE0_POSITION_ROUTE_VARIANTS:
            return "position_record_active_score_adaptive_shortlist_v1"
        raise RuntimeError("Semantic Phase-0 shortlist policy is unknown.")


def is_semantic_closure_adapter(value: Any) -> bool:
    return isinstance(
        value,
        PaperIRASemanticClosureGlobalSingletonCandidateAdapter,
    )


def _finite(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _same_float(left: Any, right: Any) -> bool:
    try:
        return math.isclose(
            float(left),
            float(right),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
    except (TypeError, ValueError):
        return False


def _normalize_semantic_phase0_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        pool_index = int(row.get("pool_index", -1))
        append_position = int(row.get("append_position", -1))
        generator_id = str(row.get("generator_id", ""))
        pool_label = str(row.get("pool_label", ""))
        gradient = _finite(
            row.get("append_gradient_signed"),
            label="Append-endpoint gradient",
        )
        denominator = _finite(
            row.get("graph_proxy_denominator"),
            label="Graph-proxy denominator",
        )
        source = str(row.get("graph_proxy_source", ""))
        if (
            pool_index < 0
            or append_position < 0
            or not generator_id
            or not pool_label
            or denominator <= 0.0
            or source != "proxy_logical_ladder_span_v1"
        ):
            raise ValueError(
                "Append-endpoint scored-row identity or source is invalid."
            )
        raw_cost = dict(row.get("graph_proxy_raw", {}))
        bars = dict(row.get("graph_proxy_bars", {}))
        if set(raw_cost) != set(_PHASE0_GRAPH_COST_COMPONENTS) or set(
            bars
        ) != set(_PHASE0_GRAPH_COST_COMPONENTS):
            raise ValueError("Graph-proxy cost components are incomplete.")
        raw_cost = {
            key: _finite(raw_cost[key], label=f"Raw graph cost {key}")
            for key in _PHASE0_GRAPH_COST_COMPONENTS
        }
        bars = {
            key: _finite(bars[key], label=f"Normalized graph cost {key}")
            for key in _PHASE0_GRAPH_COST_COMPONENTS
        }
        if any(value < 0.0 for value in (*raw_cost.values(), *bars.values())):
            raise ValueError(
                "Graph-proxy cost components must be nonnegative."
            )
        excess = _finite(
            row.get("graph_proxy_cost_excess_sum"),
            label="Graph-proxy excess",
        )
        if excess < 0.0:
            raise ValueError("Graph-proxy excess must be nonnegative.")
        normalized.append(
            {
                "pool_index": pool_index,
                "generator_id": generator_id,
                "pool_label": pool_label,
                "append_position": append_position,
                "append_gradient_signed": gradient,
                "graph_proxy_source": source,
                "graph_proxy_raw": raw_cost,
                "graph_proxy_bars": bars,
                "graph_proxy_cost_excess_sum": excess,
                "graph_proxy_denominator": denominator,
            }
        )
    normalized.sort(key=lambda row: int(row["pool_index"]))
    indices = [int(row["pool_index"]) for row in normalized]
    generator_ids = [str(row["generator_id"]) for row in normalized]
    positions = {int(row["append_position"]) for row in normalized}
    if (
        not normalized
        or len(set(indices)) != len(indices)
        or len(set(generator_ids)) != len(generator_ids)
        or len(positions) != 1
    ):
        raise ValueError(
            "Append-endpoint scored population identity is invalid."
        )
    return normalized


def _validate_semantic_graph_proxy_normalization(
    rows: Sequence[Mapping[str, Any]],
    normalization: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(normalization)
    normalization_rows = payload.get("rows")
    denominators = payload.get("denominators")
    medians = dict(payload.get("medians", {}))
    scales = dict(payload.get("scales", {}))
    lambdas = dict(payload.get("lambdas", {}))
    scale_floor = _finite(
        payload.get("scale_floor"),
        label="Graph-proxy normalization scale floor",
    )
    if (
        payload.get("schema")
        != "snake_hardware_cost_candidate_record_denominator_v1"
        or payload.get("scope") != "candidate_records"
        or payload.get("normalization_schema")
        != "snake_hardware_cost_family_robust_v1"
        or not str(payload.get("lambda_source", ""))
        or scale_floor <= 0.0
        or set(medians) != set(_PHASE0_GRAPH_COST_COMPONENTS)
        or set(scales) != set(_PHASE0_GRAPH_COST_COMPONENTS)
        or set(lambdas) != set(_PHASE0_GRAPH_COST_COMPONENTS)
        or not isinstance(normalization_rows, list)
        or not isinstance(denominators, list)
        or len(normalization_rows) != len(rows)
        or len(denominators) != len(rows)
    ):
        raise ValueError("Graph-proxy normalization contract is invalid.")
    median_values = {
        key: _finite(medians[key], label=f"Graph-proxy median {key}")
        for key in _PHASE0_GRAPH_COST_COMPONENTS
    }
    scale_values = {
        key: _finite(scales[key], label=f"Graph-proxy scale {key}")
        for key in _PHASE0_GRAPH_COST_COMPONENTS
    }
    lambda_values = {
        key: _finite(lambdas[key], label=f"Graph-proxy lambda {key}")
        for key in _PHASE0_GRAPH_COST_COMPONENTS
    }
    if any(value < 0.0 for value in median_values.values()) or any(
        value < scale_floor for value in scale_values.values()
    ):
        raise ValueError("Graph-proxy normalization scale is invalid.")
    for key in _PHASE0_GRAPH_COST_COMPONENTS:
        raw_values = [float(row["graph_proxy_raw"][key]) for row in rows]
        expected_median = float(statistics.median(raw_values))
        positive_excesses = [
            value - expected_median
            for value in raw_values
            if value > expected_median
        ]
        expected_scale = max(
            scale_floor,
            float(statistics.median(positive_excesses))
            if positive_excesses
            else scale_floor,
        )
        if not _same_float(
            median_values[key], expected_median
        ) or not _same_float(scale_values[key], expected_scale):
            raise ValueError(
                "Graph-proxy normalization statistics drifted."
            )
    for index, (row, raw_norm_row, raw_denominator) in enumerate(
        zip(rows, normalization_rows, denominators, strict=True)
    ):
        if not isinstance(raw_norm_row, Mapping):
            raise ValueError("Graph-proxy normalization row is malformed.")
        norm_row = dict(raw_norm_row)
        expected_bars = {
            key: float(
                math.asinh(
                    max(
                        0.0,
                        float(row["graph_proxy_raw"][key])
                        - median_values[key],
                    )
                    / scale_values[key]
                )
            )
            for key in _PHASE0_GRAPH_COST_COMPONENTS
        }
        expected_excess = float(
            max(
                0.0,
                sum(
                    lambda_values[key] * expected_bars[key]
                    for key in _PHASE0_GRAPH_COST_COMPONENTS
                ),
            )
        )
        expected_denominator = float(max(1.0, 1.0 + expected_excess))
        if (
            int(norm_row.get("index", -1)) != index
            or str(norm_row.get("label", "")) != str(row["pool_label"])
            or int(norm_row.get("candidate_pool_index", -1))
            != int(row["pool_index"])
            or int(norm_row.get("position_id", -1))
            != int(row["append_position"])
            or dict(norm_row.get("raw", {})) != dict(row["graph_proxy_raw"])
            or any(
                not _same_float(
                    dict(norm_row.get("bars", {})).get(key),
                    expected_bars[key],
                )
                or not _same_float(
                    row["graph_proxy_bars"][key], expected_bars[key]
                )
                for key in _PHASE0_GRAPH_COST_COMPONENTS
            )
            or not _same_float(
                norm_row.get("hardware_cost_excess_sum"), expected_excess
            )
            or not _same_float(
                row["graph_proxy_cost_excess_sum"], expected_excess
            )
            or not _same_float(
                norm_row.get("hardware_cost_denominator"),
                expected_denominator,
            )
            or not _same_float(raw_denominator, expected_denominator)
            or not _same_float(
                row["graph_proxy_denominator"], expected_denominator
            )
        ):
            raise ValueError("Graph-proxy normalization row drifted.")
    return payload


def select_semantic_proxy_phase0_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    route_variant: str,
    cap: int = 24,
) -> dict[str, Any]:
    """Select the fixed-shadow or active-adaptive proxy shortlist."""

    variant = str(route_variant)
    if variant not in {
        PAPER_I_RA_PHASE0_PROXY_FIXED24_SHADOW,
        PAPER_I_RA_PHASE0_PROXY_ADAPTIVE,
        PAPER_I_RA_PHASE0_PROXY_FIXED24_V2,
        PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2,
    }:
        raise ValueError("Route variant is not a proxy Phase-0 policy.")
    if isinstance(cap, bool):
        raise ValueError("Append-endpoint Phase-0 cap must be an integer, not bool.")
    cap_value = int(cap)
    if cap_value < 1:
        raise ValueError("Append-endpoint Phase-0 cap must be positive.")
    scores = [
        AppendEndpointGeneratorScore(
            generator_index=int(row.get("pool_index", -1)),
            append_gradient=_finite(
                row.get("append_gradient_signed"),
                label="Append-endpoint Phase-0 gradient",
            ),
            graph_cost=_finite(
                row.get("graph_proxy_denominator"),
                label="Append-endpoint graph-proxy denominator",
            ),
        )
        for row in rows
    ]
    if not scores or any(
        score.generator_index < 0 or score.graph_cost <= 0.0
        for score in scores
    ):
        raise ValueError("Append-endpoint Phase-0 population is invalid.")
    if variant in {
        PAPER_I_RA_PHASE0_PROXY_FIXED24_V2,
        PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2,
    }:
        active_score_policy = (
            "absolute_append_gradient_over_graph_proxy_cost_v1"
        )
        v2_adaptive = select_adaptive_phase0_active_score_shortlist(
            tuple(
                AdaptivePhase0ActiveScore(
                    generator_index=int(row.generator_index),
                    active_score=(
                        abs(float(row.append_gradient))
                        / float(row.graph_cost)
                    ),
                )
                for row in scores
            ),
            cap=cap_value,
            active_score_policy=active_score_policy,
        )
        v2_receipt = v2_adaptive.to_receipt()
        ranked = list(v2_adaptive.ranked_generator_indices)
        if variant == PAPER_I_RA_PHASE0_PROXY_FIXED24_V2:
            retained = ranked[: min(cap_value, len(ranked))]
            active_policy = (
                "fixed_top_k_by_absolute_gradient_graph_proxy_utility_v2"
            )
            adaptive_role = "off"
        else:
            retained = list(v2_adaptive.retained_generator_indices)
            active_policy = ADAPTIVE_PHASE0_ACTIVE_SCORE_SHORTLIST_POLICY_V2
            adaptive_role = "active"
        active_ranking = [
            {
                "generator_index": int(row["generator_index"]),
                "utility": float(row["active_score"]),
                "utility_log": row["active_score_log"],
                "utility_relative_to_champion": float(
                    row["active_score_relative_to_champion"]
                ),
            }
            for row in v2_receipt["ranking"]
        ]
        return {
            "route_variant": variant,
            "cap": cap_value,
            "active_shortlist_policy": active_policy,
            "active_score": active_score_policy,
            "active_ranking": active_ranking,
            "adaptive_decision_role": adaptive_role,
            "status": "stationary" if not retained else "competitive",
            "ranked_pool_indices": ranked,
            "retained_pool_indices": retained,
            "adaptive_decision": (
                v2_receipt
                if variant in {
                    PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2,
                }
                else None
            ),
        }
    adaptive = select_adaptive_append_endpoint_shortlist(
        scores,
        cap=cap_value,
    )
    adaptive_receipt = adaptive.to_receipt()
    if variant == PAPER_I_RA_PHASE0_PROXY_FIXED24_SHADOW:
        fixed_log_utility = {
            int(row.generator_index): (
                math.log(abs(float(row.append_gradient)))
                - math.log(float(row.graph_cost))
                if abs(float(row.append_gradient)) > 0.0
                else -math.inf
            )
            for row in scores
        }
        ranked = sorted(
            (int(row.generator_index) for row in scores),
            key=lambda index: (-fixed_log_utility[index], index),
        )
        positive_logs = [
            value
            for value in fixed_log_utility.values()
            if math.isfinite(value)
        ]
        champion_log = max(positive_logs) if positive_logs else -math.inf
        active_ranking = [
            {
                "generator_index": index,
                "utility": (
                    math.exp(fixed_log_utility[index])
                    if math.isfinite(fixed_log_utility[index])
                    else 0.0
                ),
                "utility_log": (
                    fixed_log_utility[index]
                    if math.isfinite(fixed_log_utility[index])
                    else None
                ),
                "utility_relative_to_champion": (
                    math.exp(fixed_log_utility[index] - champion_log)
                    if math.isfinite(fixed_log_utility[index])
                    else 0.0
                ),
            }
            for index in ranked
        ]
        retained = ranked[: min(cap_value, len(ranked))]
        active_policy = (
            "fixed_top_k_by_absolute_gradient_graph_proxy_utility_v1"
        )
        active_score = (
            "absolute_append_gradient_over_graph_proxy_cost_v1"
        )
        adaptive_role = "shadow"
    else:
        ranked = list(adaptive.ranked_generator_indices)
        active_ranking = [
            {
                "generator_index": int(row["generator_index"]),
                "utility": float(row["utility"]),
                "utility_log": row["utility_log"],
                "utility_relative_to_champion": float(
                    row["utility_relative_to_champion"]
                ),
            }
            for row in adaptive_receipt["ranking"]
        ]
        retained = list(adaptive.retained_generator_indices)
        active_policy = "adaptive_effective_competition_v1"
        active_score = "squared_append_gradient_over_graph_proxy_cost_v1"
        adaptive_role = "active"
    return {
        "route_variant": variant,
        "cap": cap_value,
        "active_shortlist_policy": active_policy,
        "active_score": active_score,
        "active_ranking": active_ranking,
        "adaptive_decision_role": adaptive_role,
        "status": "stationary" if not retained else "competitive",
        "ranked_pool_indices": ranked,
        "retained_pool_indices": retained,
        "adaptive_decision": adaptive_receipt,
    }


def filter_semantic_phase0_position_domain(
    admissible_domain: Sequence[Any],
    *,
    ranked_pool_indices: Sequence[int],
    retained_pool_indices: Sequence[int],
) -> tuple[Any, ...]:
    """Retain every downstream position of each selected generator."""

    population = tuple(admissible_domain)
    ranked = tuple(int(value) for value in ranked_pool_indices)
    retained = tuple(int(value) for value in retained_pool_indices)
    if not population or len(set(ranked)) != len(ranked):
        raise ValueError("Append-endpoint Phase-0 domain identity is invalid.")
    if len(set(retained)) != len(retained) or not set(retained).issubset(ranked):
        raise ValueError("Append-endpoint Phase-0 retained identity is invalid.")
    if {int(record.pool_index) for record in population} != set(ranked):
        raise ValueError(
            "Append-endpoint Phase-0 generator population differs from its "
            "immutable position domain."
        )
    retained_set = set(retained)
    rank_by_pool = {
        pool_index: rank for rank, pool_index in enumerate(ranked)
    }
    return tuple(
        sorted(
            (
                record
                for record in population
                if int(record.pool_index) in retained_set
            ),
            key=lambda record: (
                rank_by_pool[int(record.pool_index)],
                int(record.insertion_position),
                str(record.domain_record_id),
            ),
        )
    )


def build_semantic_position_phase0_receipt(
    rows: Sequence[Mapping[str, Any]],
    *,
    estimator_event_ids: Sequence[str],
    route_variant: str,
    cap: int = 24,
) -> dict[str, Any]:
    """Close one native position-record Phase-0 population and shortlist."""

    variant = str(route_variant)
    if variant not in PAPER_I_RA_PHASE0_POSITION_ROUTE_VARIANTS:
        raise ValueError("Route variant is not a position-record Phase-0 policy.")
    if isinstance(cap, bool) or int(cap) != 24:
        raise ValueError("Position-record Phase-0 requires the hard cap 24.")
    proxy_active = variant in {
        PAPER_I_RA_PHASE0_POSITION_PROXY_FIXED24_V1,
        PAPER_I_RA_PHASE0_POSITION_PROXY_ADAPTIVE_V1,
    }
    adaptive = variant in {
        PAPER_I_RA_PHASE0_POSITION_GRADIENT_ADAPTIVE_V1,
        PAPER_I_RA_PHASE0_POSITION_PROXY_ADAPTIVE_V1,
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1,
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_FORCED_K50_V1,
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
    }
    normalized: list[dict[str, Any]] = []
    identities: set[tuple[int, int, str]] = set()
    for raw in rows:
        row = dict(raw)
        pool_index = int(row.get("pool_index", -1))
        position = int(row.get("insertion_position", -1))
        record_id = str(row.get("domain_record_id", ""))
        generator_id = str(row.get("generator_id", ""))
        pool_label = str(row.get("pool_label", ""))
        position_class = str(row.get("position_class", ""))
        gradient = _finite(
            row.get("gradient_signed"),
            label="Position-record Phase-0 gradient",
        )
        denominator = _finite(
            row.get("graph_proxy_denominator", 1.0),
            label="Position-record Phase-0 graph denominator",
        )
        identity = (pool_index, position, record_id)
        if (
            pool_index < 0
            or position < 0
            or not record_id
            or not generator_id
            or not pool_label
            or position_class not in {"interior", "append"}
            or identity in identities
            or denominator <= 0.0
        ):
            raise ValueError("Position-record Phase-0 row identity is invalid.")
        identities.add(identity)
        active_score = abs(gradient) / denominator if proxy_active else abs(gradient)
        normalized.append(
            {
                "domain_record_id": record_id,
                "generator_id": generator_id,
                "pool_index": pool_index,
                "pool_label": pool_label,
                "insertion_position": position,
                "position_class": position_class,
                "gradient_signed": gradient,
                "gradient_abs": abs(gradient),
                "graph_proxy_denominator": denominator,
                "active_score": active_score,
            }
        )
    if not normalized:
        raise ValueError("Position-record Phase-0 population must be non-empty.")
    normalized.sort(
        key=lambda row: (
            int(row["pool_index"]),
            int(row["insertion_position"]),
            str(row["domain_record_id"]),
        )
    )
    events = [str(value) for value in estimator_event_ids]
    if (
        len(events) != len(normalized)
        or len(set(events)) != len(events)
        or any(not value for value in events)
    ):
        raise ValueError("Position-record Phase-0 estimator events are invalid.")
    score_policy = _phase0_policy(variant)["score"]
    decision = select_adaptive_phase0_active_score_shortlist(
        tuple(
            AdaptivePhase0ActiveScore(index, float(row["active_score"]))
            for index, row in enumerate(normalized)
        ),
        cap=int(cap),
        active_score_policy=str(score_policy),
        min_retained=semantic_phase0_minimum_retained(variant),
    )
    decision_receipt = decision.to_receipt()
    ranked_offsets = [int(value) for value in decision.ranked_generator_indices]
    retained_offsets = (
        list(decision.retained_generator_indices)
        if adaptive
        else ranked_offsets[: min(int(cap), len(ranked_offsets))]
    )
    retained_set = set(retained_offsets)
    ranking = [
        {
            **dict(normalized[offset]),
            "rank": rank,
            "retained": offset in retained_set,
        }
        for rank, offset in enumerate(ranked_offsets, start=1)
    ]
    retained_rows = [dict(normalized[offset]) for offset in retained_offsets]
    components = {
        "N_H_outer": 0,
        "N_H_refit": 0,
        "N_grad": len(normalized),
        "N_metric": 0,
    }
    identity = semantic_closure_route_identity(variant)
    payload: dict[str, Any] = {
        "schema": PAPER_I_RA_SEMANTIC_POSITION_PHASE0_RECEIPT_SCHEMA,
        "policy": _phase0_policy(variant)["shortlist"],
        "route_variant": variant,
        "route_id": identity.route_id,
        "algorithm_id": identity.algorithm_id,
        "semantic_implementation_version": identity.semantic_implementation_version,
        "population_scope": (
            "current_commutation_reduced_candidate_position_records_v1"
        ),
        "consumer_scope": "phase0_candidate_position_gradient_surface_v1",
        "gradient_surface": "commutation_reduced_candidate_position_records_v1",
        "position_aware_gradient_surface": True,
        "generator_level_reexpansion_after_phase0": False,
        "score": str(score_policy),
        "ranking_order": (
            "descending_active_score_then_pool_position_record_identity_v1"
        ),
        "graph_proxy_cost_policy": (
            "paper_i_structural_graph_proxy_transform_v1"
            if proxy_active
            else "off"
        ),
        "qiskit_compile_cost_policy": "off",
        "metric_policy": "off",
        "measurement_cost_policy": "off",
        "requested_cap": int(cap),
        "adaptive_decision_role": "active" if adaptive else "off",
        "status": str(decision.status) if adaptive else "competitive",
        "frontier_saturated": (
            bool(decision.frontier_saturated) if adaptive else len(normalized) > int(cap)
        ),
        "input_candidate_count": len(normalized),
        "retained_candidate_count": len(retained_rows),
        "effective_shortlist_size": len(retained_rows),
        "input_population_sha256": canonical_sha256(normalized),
        "retained_population_sha256": canonical_sha256(retained_rows),
        "population": normalized,
        "ranking": ranking,
        "retained_records": retained_rows,
        "adaptive_decision": decision_receipt if adaptive else None,
        "estimator_event_ids": events,
        "estimator_accounting": {
            "unit": "executed_logical_scalar_estimator_invocation",
            "components": components,
            **components,
            "S_alg": int(sum(components.values())),
            "zero_metric_measurements": True,
        },
    }
    if adaptive and decision.status == "stationary":
        payload["terminal_controller_outcome"] = (
            "phase0_stationary_no_competitive_candidate_v1"
        )
    payload["sha256"] = canonical_sha256(payload)
    return payload


def validate_semantic_position_phase0_receipt(
    raw_receipt: Mapping[str, Any],
    *,
    scored_population: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute a position-record receipt from its authenticated population."""

    try:
        receipt = copy.deepcopy(dict(raw_receipt))
        if receipt.get("schema") != (
            PAPER_I_RA_SEMANTIC_POSITION_PHASE0_RECEIPT_SCHEMA
        ):
            raise RuntimeError(
                "Position-record Phase-0 receipt schema is invalid."
            )
        observed_sha = receipt.get("sha256")
        unsigned = dict(receipt)
        unsigned.pop("sha256", None)
        if observed_sha != canonical_sha256(unsigned):
            raise RuntimeError(
                "Position-record Phase-0 receipt digest is invalid."
            )
        rebuilt = build_semantic_position_phase0_receipt(
            receipt.get("population", []),
            estimator_event_ids=receipt.get("estimator_event_ids", []),
            route_variant=str(receipt.get("route_variant", "")),
            cap=int(receipt.get("requested_cap", -1)),
        )
        if rebuilt != receipt:
            raise RuntimeError(
                "Position-record Phase-0 receipt failed recomputation."
            )
        if scored_population is not None:
            _validate_semantic_scored_position_record_projection(
                receipt,
                scored_population,
            )
    except RuntimeError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            "Position-record Phase-0 receipt failed recomputation."
        ) from exc
    return receipt


def _validate_semantic_scored_position_record_projection(
    receipt: Mapping[str, Any],
    scored_population: Mapping[str, Any],
) -> None:
    """Bind a retained position-record shortlist to the Phase-I input seam."""

    screen = scored_population.get("phase0_gradient_screen")
    if not isinstance(screen, Mapping) or screen.get("schema") != (
        "paper_i_scored_gradient_phase0_population_v1"
    ):
        raise RuntimeError("Position-record Phase-0 scored domain is absent.")
    population = screen.get("population")
    shortlist = screen.get("shortlist")
    if (
        not isinstance(population, list)
        or not population
        or any(not isinstance(row, Mapping) for row in population)
        or not isinstance(shortlist, list)
        or any(not isinstance(row, Mapping) for row in shortlist)
        or int(screen.get("population_count", -1)) != len(population)
        or int(screen.get("shortlist_count", -1)) != len(shortlist)
        or screen.get("ordered_population_sha256")
        != canonical_sha256(population)
        or screen.get("ordered_shortlist_sha256")
        != canonical_sha256(shortlist)
    ):
        raise RuntimeError(
            "Position-record Phase-0 scored domain is malformed."
        )

    projection_fields = (
        "domain_record_id",
        "generator_id",
        "pool_index",
        "pool_label",
        "insertion_position",
        "position_class",
    )

    def _project(row: Mapping[str, Any]) -> dict[str, Any]:
        return {field: row[field] for field in projection_fields}

    expected_population = [
        _project(row) for row in receipt.get("population", [])
    ]
    expected_shortlist = [
        _project(row) for row in receipt.get("retained_records", [])
    ]
    if [_project(row) for row in population] != expected_population:
        raise RuntimeError(
            "Semantic Phase-0 position-record population domain drifted."
        )
    if [_project(row) for row in shortlist] != expected_shortlist:
        raise RuntimeError(
            "Semantic Phase-0 retained position-record domain drifted."
        )


def build_semantic_proxy_phase0_receipt(
    rows: Sequence[Mapping[str, Any]],
    *,
    graph_proxy_normalization: Mapping[str, Any],
    estimator_event_ids: Sequence[str],
    route_variant: str,
    cap: int = 24,
) -> dict[str, Any]:
    """Build a self-digesting native proxy Phase-0 receipt."""

    normalized = _normalize_semantic_phase0_rows(rows)
    normalization = _validate_semantic_graph_proxy_normalization(
        normalized,
        graph_proxy_normalization,
    )
    decision = select_semantic_proxy_phase0_rows(
        normalized,
        route_variant=str(route_variant),
        cap=cap,
    )
    event_ids = [str(value) for value in estimator_event_ids]
    if (
        len(event_ids) != len(normalized)
        or len(set(event_ids)) != len(event_ids)
        or any(not value for value in event_ids)
    ):
        raise ValueError(
            "Append-endpoint Phase-0 estimator events do not close N_grad."
        )
    active_ranking = {
        int(row["generator_index"]): dict(row)
        for row in decision["active_ranking"]
    }
    retained_set = set(
        int(value) for value in decision["retained_pool_indices"]
    )
    row_by_pool = {
        int(row["pool_index"]): dict(row) for row in normalized
    }
    ranking: list[dict[str, Any]] = []
    for rank, pool_index in enumerate(
        decision["ranked_pool_indices"],
        start=1,
    ):
        population_row = row_by_pool[int(pool_index)]
        active_row = active_ranking[int(pool_index)]
        ranking.append(
            {
                **population_row,
                "append_gradient_abs": float(
                    abs(float(population_row["append_gradient_signed"]))
                ),
                "utility": float(active_row["utility"]),
                "utility_log": active_row["utility_log"],
                "utility_relative_to_champion": float(
                    active_row["utility_relative_to_champion"]
                ),
                "rank": int(rank),
                "active_retained": int(pool_index) in retained_set,
            }
        )
    components = {
        "N_H_outer": 0,
        "N_H_refit": 0,
        "N_grad": len(normalized),
        "N_metric": 0,
    }
    zero_components = {key: 0 for key in components}
    identity = semantic_closure_route_identity(str(route_variant))
    payload: dict[str, Any] = {
        "schema": PAPER_I_RA_SEMANTIC_PHASE0_PROXY_RECEIPT_SCHEMA,
        "policy": _phase0_policy(str(route_variant))["shortlist"],
        "route_variant": str(route_variant),
        "route_id": identity.route_id,
        "algorithm_id": identity.algorithm_id,
        "semantic_implementation_version": identity.semantic_implementation_version,
        "population_scope": PAPER_I_RA_SEMANTIC_PHASE0_POPULATION_SCOPE,
        "consumer_scope": PAPER_I_RA_SEMANTIC_PHASE0_CONSUMER_SCOPE,
        "gradient_surface": "append_endpoint_generators_v1",
        "position_aware_gradient_surface": False,
        "insertion_position_scope": (
            "append_endpoint_generator_screen_before_downstream_position_policy_v1"
        ),
        "downstream_insertion_policy": "independent_unmodified_v1",
        "score": (
            "absolute_append_gradient_over_graph_proxy_denominator_v1"
            if str(route_variant)
            in {
                PAPER_I_RA_PHASE0_PROXY_FIXED24_SHADOW,
                PAPER_I_RA_PHASE0_PROXY_FIXED24_V2,
                PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2,
            }
            else "squared_append_gradient_over_graph_proxy_denominator_v1"
        ),
        "ranking_order": (
            "descending_absolute_gradient_over_graph_proxy_then_pool_index_v1"
            if str(route_variant)
            in {
                PAPER_I_RA_PHASE0_PROXY_FIXED24_SHADOW,
                PAPER_I_RA_PHASE0_PROXY_FIXED24_V2,
                PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2,
            }
            else "descending_squared_gradient_over_graph_proxy_then_pool_index_v1"
        ),
        "graph_proxy_cost_policy": "family_robust_positive_denominator_v1",
        "graph_proxy_compile_source": "phase1_logical_graph_proxy_v1",
        "qiskit_compile_cost_policy": "off",
        "metric_policy": "off",
        "measurement_cost_policy": "off",
        "native_semantic_closure_bound": True,
        "requested_cap": int(decision["cap"]),
        "active_shortlist_policy": str(
            decision["active_shortlist_policy"]
        ),
        "adaptive_decision_role": str(
            decision["adaptive_decision_role"]
        ),
        "status": str(decision["status"]),
        "append_position": int(normalized[0]["append_position"]),
        "input_candidate_count": len(normalized),
        "retained_candidate_count": len(retained_set),
        "effective_shortlist_size": len(retained_set),
        "input_pool_indices": [
            int(row["pool_index"]) for row in normalized
        ],
        "ranked_pool_indices": [
            int(value) for value in decision["ranked_pool_indices"]
        ],
        "retained_pool_indices": [
            int(value) for value in decision["retained_pool_indices"]
        ],
        "input_population_sha256": canonical_sha256(normalized),
        "retained_population_sha256": canonical_sha256(
            [
                row
                for row in normalized
                if int(row["pool_index"]) in retained_set
            ]
        ),
        "population": normalized,
        "ranking": ranking,
        "graph_proxy_normalization": normalization,
        "adaptive_decision": (
            dict(decision["adaptive_decision"])
            if isinstance(decision["adaptive_decision"], Mapping)
            else None
        ),
        "estimator_event_ids": event_ids,
        "estimator_accounting": {
            "unit": "executed_logical_scalar_estimator_invocation",
            "components": components,
            **components,
            "S_alg": int(sum(components.values())),
            "zero_metric_measurements": True,
        },
        "adaptive_shadow_accounting": {
            "source": (
                "classical_reuse_of_active_gradient_and_proxy_population_v1"
                if decision["adaptive_decision_role"] == "shadow"
                else "off_v2"
            ),
            "components": zero_components,
            **zero_components,
            "S_alg": 0,
        },
    }
    if str(decision["status"]) == "stationary":
        payload["terminal_controller_outcome"] = (
            "phase0_stationary_no_competitive_candidate_v1"
        )
    payload["sha256"] = canonical_sha256(payload)
    return payload


def build_semantic_gradient_adaptive_phase0_receipt(
    *,
    available_indices: Sequence[int],
    gradients: Sequence[float],
    pool_labels: Sequence[str],
    estimator_event_ids: Sequence[str],
    route_variant: str,
    cap: int = 24,
) -> dict[str, Any]:
    """Close the v2 standard-ADAPT adaptive-cardinality Phase-0 screen."""

    variant = str(route_variant)
    if variant not in {
        PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2,
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1,
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_FORCED_K50_V1,
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
    }:
        raise ValueError("Route variant is not the v2 gradient-adaptive policy.")
    indices = tuple(sorted(int(value) for value in available_indices))
    values = tuple(float(value) for value in gradients)
    if (
        not indices
        or len(set(indices)) != len(indices)
        or any(index < 0 or index >= len(values) for index in indices)
        or any(not math.isfinite(values[index]) for index in indices)
        or any(index >= len(pool_labels) for index in indices)
    ):
        raise ValueError("Gradient-adaptive Phase-0 population is invalid.")
    event_ids = [str(value) for value in estimator_event_ids]
    if (
        len(event_ids) != len(indices)
        or len(set(event_ids)) != len(event_ids)
        or any(not value for value in event_ids)
    ):
        raise ValueError(
            "Gradient-adaptive Phase-0 estimator events do not close N_grad."
        )
    active_score_policy = "absolute_append_endpoint_generator_gradient_v1"
    decision = select_adaptive_phase0_active_score_shortlist(
        tuple(
            AdaptivePhase0ActiveScore(index, abs(values[index]))
            for index in indices
        ),
        cap=int(cap),
        active_score_policy=active_score_policy,
        min_retained=semantic_phase0_minimum_retained(route_variant),
    )
    decision_receipt = decision.to_receipt()
    retained = set(decision.retained_generator_indices)
    gradient_by_index = {index: values[index] for index in indices}
    identity = semantic_closure_route_identity(variant)
    ranking = [
        {
            "pool_index": int(row["generator_index"]),
            "pool_label": str(pool_labels[int(row["generator_index"])]),
            "gradient_signed": float(
                gradient_by_index[int(row["generator_index"])]
            ),
            "gradient_abs": float(row["active_score"]),
            "active_score": float(row["active_score"]),
            "active_score_log": row["active_score_log"],
            "active_score_relative_to_champion": float(
                row["active_score_relative_to_champion"]
            ),
            "rank": int(row["rank"]),
            "retained": int(row["generator_index"]) in retained,
        }
        for row in decision_receipt["ranking"]
    ]
    components = {
        "N_H_outer": 0,
        "N_H_refit": 0,
        "N_grad": len(indices),
        "N_metric": 0,
    }
    payload: dict[str, Any] = {
        "schema": PAPER_I_RA_SEMANTIC_PHASE0_GRADIENT_ADAPTIVE_RECEIPT_SCHEMA_V2,
        "policy": ADAPTIVE_PHASE0_ACTIVE_SCORE_SHORTLIST_POLICY_V2,
        "route_variant": variant,
        "route_id": identity.route_id,
        "algorithm_id": identity.algorithm_id,
        "semantic_implementation_version": identity.semantic_implementation_version,
        "population_scope": PAPER_I_RA_SEMANTIC_PHASE0_POPULATION_SCOPE,
        "consumer_scope": PAPER_I_RA_SEMANTIC_PHASE0_CONSUMER_SCOPE,
        "score": active_score_policy,
        "ranking_order": "descending_absolute_gradient_then_pool_index_v1",
        "adaptive_law": "inverse_simpson_active_score_population_v2",
        "retention_policy": (
            "ranked_prefix_with_exact_boundary_tie_closure_subject_to_"
            "hard_cap_v2"
        ),
        "position_aware_gradient_surface": False,
        "graph_proxy_cost_policy": "off",
        "qiskit_compile_cost_policy": "off",
        "compile_cost_policy": "off",
        "metric_policy": "off",
        "measurement_cost_policy": "off",
        "requested_cap": int(cap),
        "status": str(decision.status),
        "input_candidate_count": len(indices),
        "retained_candidate_count": len(retained),
        "effective_shortlist_size": len(retained),
        "input_pool_indices": list(indices),
        "ranked_pool_indices": list(decision.ranked_generator_indices),
        "retained_pool_indices": list(decision.retained_generator_indices),
        "ranking": ranking,
        "adaptive_decision": decision_receipt,
        "estimator_event_ids": event_ids,
        "estimator_accounting": {
            "unit": "executed_logical_scalar_estimator_invocation",
            "components": components,
            **components,
            "S_alg": int(sum(components.values())),
            "zero_metric_measurements": True,
        },
    }
    if decision.status == "stationary":
        payload["terminal_controller_outcome"] = (
            "phase0_stationary_no_competitive_candidate_v1"
        )
    payload["sha256"] = canonical_sha256(payload)
    return payload


def validate_semantic_gradient_adaptive_phase0_receipt(
    raw_receipt: Mapping[str, Any],
    *,
    scored_population: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute and validate one v2 adaptive-gradient Phase-0 receipt."""

    try:
        receipt = dict(raw_receipt)
        observed_sha256 = receipt.get("sha256")
        unsigned = dict(receipt)
        unsigned.pop("sha256", None)
        if observed_sha256 != canonical_sha256(unsigned):
            raise RuntimeError("Semantic Phase-0 receipt digest is invalid.")

        raw_indices = receipt.get("input_pool_indices")
        raw_ranking = receipt.get("ranking")
        if (
            not isinstance(raw_indices, list)
            or not raw_indices
            or not isinstance(raw_ranking, list)
            or len(raw_ranking) != len(raw_indices)
            or any(not isinstance(row, Mapping) for row in raw_ranking)
        ):
            raise RuntimeError(
                "Gradient-adaptive Phase-0 population is malformed."
            )
        indices = tuple(int(value) for value in raw_indices)
        ranking_indices = tuple(
            int(row.get("pool_index", -1)) for row in raw_ranking
        )
        if (
            len(set(indices)) != len(indices)
            or set(ranking_indices) != set(indices)
            or len(set(ranking_indices)) != len(ranking_indices)
            or any(index < 0 for index in indices)
        ):
            raise RuntimeError(
                "Gradient-adaptive Phase-0 population is malformed."
            )

        extent = max(indices) + 1
        gradients = [0.0] * extent
        pool_labels = [""] * extent
        for row in raw_ranking:
            pool_index = int(row["pool_index"])
            gradients[pool_index] = float(row["gradient_signed"])
            pool_labels[pool_index] = str(row["pool_label"])
        expected = build_semantic_gradient_adaptive_phase0_receipt(
            available_indices=indices,
            gradients=gradients,
            pool_labels=pool_labels,
            estimator_event_ids=receipt.get("estimator_event_ids", []),
            route_variant=str(receipt.get("route_variant", "")),
            cap=receipt.get("requested_cap", 0),
        )
        if receipt != expected:
            raise RuntimeError(
                "Semantic Phase-0 receipt failed decision recomputation."
            )
        if scored_population is not None:
            _validate_semantic_scored_position_projection(
                receipt,
                scored_population,
            )
    except RuntimeError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            "Semantic Phase-0 receipt failed decision recomputation."
        ) from exc
    return dict(raw_receipt)


def _validate_semantic_scored_position_projection(
    receipt: Mapping[str, Any],
    scored_population: Mapping[str, Any],
) -> None:
    screen = scored_population.get("phase0_gradient_screen")
    if not isinstance(screen, Mapping) or screen.get("schema") != (
        "paper_i_scored_gradient_phase0_population_v1"
    ):
        raise RuntimeError("Semantic Phase-0 scored domain is absent.")
    population = screen.get("population")
    shortlist = screen.get("shortlist")
    if (
        not isinstance(population, list)
        or not population
        or any(not isinstance(row, Mapping) for row in population)
        or not isinstance(shortlist, list)
        or any(not isinstance(row, Mapping) for row in shortlist)
        or int(screen.get("population_count", -1)) != len(population)
        or int(screen.get("shortlist_count", -1)) != len(shortlist)
        or screen.get("ordered_population_sha256")
        != canonical_sha256(population)
        or screen.get("ordered_shortlist_sha256")
        != canonical_sha256(shortlist)
    ):
        raise RuntimeError("Semantic Phase-0 scored domain is malformed.")
    input_indices = set(
        int(value) for value in receipt["input_pool_indices"]
    )
    ranked_indices = [
        int(value) for value in receipt["ranked_pool_indices"]
    ]
    retained_indices = set(
        int(value) for value in receipt["retained_pool_indices"]
    )
    if {int(row.get("pool_index", -1)) for row in population} != input_indices:
        raise RuntimeError("Semantic Phase-0 population domain drifted.")
    rank_by_pool = {
        pool_index: rank for rank, pool_index in enumerate(ranked_indices)
    }
    expected_shortlist = sorted(
        (
            dict(row)
            for row in population
            if int(row.get("pool_index", -1)) in retained_indices
        ),
        key=lambda row: (
            rank_by_pool[int(row["pool_index"])],
            int(row["insertion_position"]),
            str(row["domain_record_id"]),
        ),
    )
    if [dict(row) for row in shortlist] != expected_shortlist:
        raise RuntimeError(
            "Semantic Phase-0 changed a retained generator's position domain."
        )


def validate_semantic_proxy_phase0_receipt(
    raw_receipt: Mapping[str, Any],
    *,
    scored_population: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute and validate one native proxy Phase-0 receipt."""

    try:
        receipt = dict(raw_receipt)
        observed_sha256 = receipt.get("sha256")
        unsigned = dict(receipt)
        unsigned.pop("sha256", None)
        if observed_sha256 != canonical_sha256(unsigned):
            raise RuntimeError("Semantic Phase-0 receipt digest is invalid.")
        expected = build_semantic_proxy_phase0_receipt(
            receipt.get("population", []),
            graph_proxy_normalization=receipt.get(
                "graph_proxy_normalization", {}
            ),
            estimator_event_ids=receipt.get("estimator_event_ids", []),
            route_variant=str(receipt.get("route_variant", "")),
            cap=receipt.get("requested_cap", 0),
        )
        if receipt != expected:
            raise RuntimeError(
                "Semantic Phase-0 receipt failed decision recomputation."
            )
        if scored_population is not None:
            _validate_semantic_scored_position_projection(
                receipt,
                scored_population,
            )
    except RuntimeError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            "Semantic Phase-0 receipt failed decision recomputation."
        ) from exc
    return dict(raw_receipt)


def validate_semantic_phase0_runtime_binding(
    adapter: PaperIRASemanticClosureGlobalSingletonCandidateAdapter,
    route_contract: Mapping[str, Any],
) -> str:
    """Fail closed unless runtime settings match the typed route identity."""

    if not is_semantic_closure_adapter(adapter):
        raise RuntimeError("Semantic Phase-0 requires its typed adapter.")
    identity = semantic_closure_route_identity(adapter.route_variant)
    if identity.route_variant not in PAPER_I_RA_PHASE0_EXECUTABLE_ROUTE_VARIANTS:
        raise RuntimeError(
            "The v1 Phase-0 semantic runtime is retired; use a v2 route."
        )
    route = dict(route_contract)
    execution = route.get("execution_settings")
    invariants = route.get("semantic_invariants")
    native = route.get("native_semantic_contract")
    policy = _phase0_policy(adapter.route_variant)
    if (
        route.get("algorithm_id") != identity.algorithm_id
        or route.get("route_id") != identity.route_id
        or route.get("semantic_implementation_version")
        != identity.semantic_implementation_version
        or not isinstance(execution, Mapping)
        or not isinstance(invariants, Mapping)
        or not isinstance(native, Mapping)
        or native.get("route_variant") != adapter.route_variant
        or native.get("algorithm_id") != identity.algorithm_id
        or native.get("phase0_policy") != policy
        or native.get("compile_scope")
        != PAPER_I_RA_PHASE123_QISKIT_COMPILE_SCOPE
        or native.get("hardware_cost_normalization")
        != "zero_centered_signed_arctan_v1"
        or execution.get("ra_semantic_route_variant")
        != adapter.route_variant
        or execution.get("ra_semantic_implementation_version")
        != identity.semantic_implementation_version
        or execution.get("ra_phase0_gradient_shortlist_policy")
        != adapter.phase0_shortlist_policy_id
        or execution.get("ra_phase0_gradient_shortlist_size") != 24
        or execution.get("ra_phase0_adaptive_shadow_receipt")
        is not bool(policy["adaptive_shadow_receipt"])
        or execution.get("phase3_backend_cost_scope")
        != PAPER_I_RA_PHASE123_QISKIT_COMPILE_SCOPE
        or execution.get("phase3_hardware_cost_normalization_mode")
        != "zero_centered_signed_arctan_v1"
        or invariants.get("phase0_active") is not True
        or invariants.get("phase0_population") != policy["population"]
        or invariants.get("phase0_score") != policy["score"]
        or invariants.get("phase0_fubini_metric_active") is not False
        or invariants.get("phase0_compile_cost_active") is not False
        or invariants.get("phase0_structural_proxy_cost_active")
        is not (policy["graph_proxy_cost"] != "off")
        or invariants.get("phase0_resource_cost_active")
        is not (policy["graph_proxy_cost"] != "off")
        or invariants.get("phase0_estimator_components") != ["N_grad"]
    ):
        raise RuntimeError("Semantic Phase-0 route binding drifted.")
    return adapter.route_variant


def execute_semantic_phase0_runtime(
    transaction: Any,
    *,
    admissible_domain: Sequence[Any],
) -> Any:
    """Dispatch one typed semantic route through the native transaction."""

    adapter = transaction.context.candidate_adapter
    variant = validate_semantic_phase0_runtime_binding(
        adapter,
        transaction.context.route_contract,
    )
    if variant == PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2:
        from pipelines.static_adapt.ra_adapt.phase0 import (
            GLOBAL_SINGLETON_ABSOLUTE_GRADIENT_PHASE0_POLICY,
            GLOBAL_SINGLETON_GRADIENT_PHASE0_RECEIPT_SCHEMA,
        )

        return transaction.run_absolute_gradient_phase0(
            admissible_domain=tuple(admissible_domain),
            shortlist_size=24,
            policy=GLOBAL_SINGLETON_ABSOLUTE_GRADIENT_PHASE0_POLICY,
            receipt_schema=GLOBAL_SINGLETON_GRADIENT_PHASE0_RECEIPT_SCHEMA,
            consumer_scope=PAPER_I_RA_SEMANTIC_PHASE0_CONSUMER_SCOPE,
            population_scope=PAPER_I_RA_SEMANTIC_PHASE0_POPULATION_SCOPE,
        )
    if variant in {
        PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2,
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1,
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_FORCED_K50_V1,
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
    }:
        return transaction.run_adaptive_gradient_phase0(
            admissible_domain=tuple(admissible_domain),
            route_variant=variant,
            shortlist_size=24,
        )
    if variant in PAPER_I_RA_PHASE0_POSITION_ROUTE_VARIANTS:
        return transaction.run_position_record_phase0(
            admissible_domain=tuple(admissible_domain),
            route_variant=variant,
            shortlist_size=24,
        )
    return transaction.run_semantic_proxy_phase0(
        admissible_domain=tuple(admissible_domain),
        route_variant=variant,
        shortlist_size=24,
    )


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _validate_phase123_population_receipt(
    raw_receipt: Mapping[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    receipt = dict(raw_receipt)
    rows = receipt.get("rows")
    if (
        receipt.get("schema")
        != "paper_i_phase123_qiskit_population_normalization_v1"
        or receipt.get("scope") != PAPER_I_RA_PHASE123_QISKIT_COMPILE_SCOPE
        or receipt.get("phase") != str(phase)
        or receipt.get("normalization_count") != 1
        or receipt.get("normalization_policy")
        != "zero_centered_signed_arctan_v1"
        or receipt.get("negative_delta_reward_enabled") is not True
        or receipt.get("full_base_trial_at_recorded_insertion") is not True
        or receipt.get("excluded_from_s_alg") is not True
        or "S_alg" in receipt
        or not _is_sha256(receipt.get("population_hash"))
        or not isinstance(rows, list)
        or not rows
        or any(not isinstance(row, Mapping) for row in rows)
        or int(receipt.get("population_count", -1)) != len(rows)
        or receipt.get("rows_sha256") != canonical_sha256(rows)
    ):
        raise RuntimeError("Phase-I--III population receipt is invalid.")
    population_hash = str(receipt["population_hash"])
    identities: set[tuple[int, int, str]] = set()
    base_keys: set[str] = set()
    for raw_row in rows:
        row = dict(raw_row)
        identity = (
            int(row.get("candidate_pool_index", -1)),
            int(row.get("position_id", -1)),
            str(row.get("generator_id", "")),
        )
        base_key = row.get("base_structure_key")
        trial_key = row.get("trial_structure_key")
        candidate_label = str(row.get("candidate_label", ""))
        compile_cache_generator_id = str(
            row.get("compile_cache_generator_id", "")
        )
        compile_cache_identity = row.get("compile_cache_identity")
        expected_cache_identity = {
            "schema": "phase123_qiskit_candidate_position_compile_cache_v1",
            "scope": PAPER_I_RA_PHASE123_QISKIT_COMPILE_SCOPE,
            "candidate_label": candidate_label,
            "generator_id": compile_cache_generator_id,
            "position_id": identity[1],
            "base_structure_key": str(base_key),
            "trial_structure_key": str(trial_key),
        }
        numeric_keys = (
            "raw_delta_compiled_count_2q",
            "raw_delta_compiled_depth_2q",
            "raw_delta_compiled_count_1q",
            "hardware_cost_signed_index",
            "hardware_cost_score_factor",
        )
        try:
            numeric_values = tuple(float(row[key]) for key in numeric_keys)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Phase-I--III population row is incomplete."
            ) from exc
        if (
            min(identity[:2]) < 0
            or not identity[2]
            or not candidate_label
            or compile_cache_generator_id
            != f"{identity[2]}::pool[{identity[0]}]"
            or identity in identities
            or not _is_sha256(base_key)
            or not _is_sha256(trial_key)
            or str(base_key) == str(trial_key)
            or row.get("hardware_cost_population_hash") != population_hash
            or not all(math.isfinite(value) for value in numeric_values)
            or float(row["hardware_cost_score_factor"]) <= 0.0
            or not isinstance(compile_cache_identity, Mapping)
            or dict(compile_cache_identity) != expected_cache_identity
            or row.get("compile_cache_identity_sha256")
            != canonical_sha256(expected_cache_identity)
        ):
            raise RuntimeError("Phase-I--III population row is invalid.")
        identities.add(identity)
        base_keys.add(str(base_key))
    if len(base_keys) != 1:
        raise RuntimeError(
            "Phase-I--III population mixes full-base compile identities."
        )
    return receipt


def validate_semantic_projected_phase123_receipt(
    raw_projected: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and retain the complete serialized Phase-I--III evidence."""

    projected = copy.deepcopy(dict(raw_projected))
    required_phases = ("phase_i", "phase_ii", "phase_iii")
    phase_receipts = projected.get(
        "phase123_qiskit_population_normalization_receipts"
    )
    qiskit_receipt = projected.get("phase3_qiskit_selector_cost_receipt")
    linked_phase_iii = (
        qiskit_receipt.get("phase123_population_normalization_receipt")
        if isinstance(qiskit_receipt, Mapping)
        else None
    )
    if (
        projected.get("schema")
        != "paper_i_projected_phase3_population_receipt_v2"
        or not isinstance(phase_receipts, Mapping)
        or set(phase_receipts) != set(required_phases)
        or linked_phase_iii != phase_receipts.get("phase_iii")
    ):
        raise RuntimeError(
            "Phase-I--III normalization receipt set is incomplete."
        )
    validated = {
        phase: _validate_phase123_population_receipt(
            phase_receipts[phase],
            phase=phase,
        )
        for phase in required_phases
    }
    projected[
        "phase123_qiskit_population_normalization_receipts"
    ] = copy.deepcopy(validated)
    projected["phase3_qiskit_selector_cost_receipt"] = {
        **copy.deepcopy(dict(qiskit_receipt)),
        "phase123_population_normalization_receipt": copy.deepcopy(
            validated["phase_iii"]
        ),
    }
    return projected


def _validate_terminal_controller_measurement_work_proxy(
    raw_proxy: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate the failed attempt's complete controller-work summary."""

    if not isinstance(raw_proxy, Mapping):
        raise ValueError("Terminal controller work must be a mapping.")
    proxy = copy.deepcopy(dict(raw_proxy))
    by_phase = proxy.get("by_phase")
    per_phase = proxy.get("per_phase")
    by_scope = proxy.get("by_scope")
    numeric = proxy.get("numeric_validation")
    controller_numeric = proxy.get("controller_numeric_validation")
    if (
        proxy.get("schema") != "controller_measurement_work_proxy_v1"
        or proxy.get("source") != "native_controller_live_decision_work_v1"
        or proxy.get("source_kind") != "native_controller_work"
        or proxy.get("legacy_fallback_used") is not False
        or "events" in proxy
        or not isinstance(by_phase, Mapping)
        or not isinstance(per_phase, Mapping)
        or dict(by_phase) != dict(per_phase)
        or not {"phase1", "phase2", "phase3"}.issubset(by_phase)
        or not isinstance(by_scope, Mapping)
        or proxy.get("work_scope_count") != len(by_scope)
        or not isinstance(numeric, Mapping)
        or dict(controller_numeric or {}) != dict(numeric)
        or numeric.get("schema")
        != "controller_measurement_work_numeric_validation_v1"
        or numeric.get("status") != "ok"
        or numeric.get("paper_i_table_work_status") != "ok"
        or numeric.get("missing_required_fields") != []
        or numeric.get("invalid_fields") != []
        or proxy.get("controller_numeric_validation_status") != "ok"
        or proxy.get("paper_i_controller_numeric_validation_status") != "ok"
        or proxy.get("candidate_work_ledger_schema")
        != "controller_candidate_work_ledger_v1"
        or proxy.get("candidate_work_ledger_status")
        != "explicit_candidate_work_ledger_v1"
    ):
        raise ValueError("Terminal controller work identity drifted.")

    integer_counters = (
        "events_count",
        "candidate_work_event_count",
        "candidate_work_missing_event_count",
        "candidate_count_total",
        "evaluated_count_total",
        "pre_shortlist_count_total",
        "shortlist_size_total",
        "retained_count_total",
        "rejected_count_total",
    )
    for key in integer_counters:
        value = proxy.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Terminal controller-work counter is malformed.")
        if not float(value).is_integer() or int(value) < 0:
            raise ValueError("Terminal controller-work counter is invalid.")
    if (
        int(proxy["events_count"]) <= 0
        or int(proxy["candidate_work_event_count"])
        != int(proxy["events_count"])
        or int(proxy["candidate_work_missing_event_count"]) != 0
    ):
        raise ValueError("Terminal controller-work event closure drifted.")

    additive_keys = (
        "events_count",
        "records_evaluated",
        "records_with_group_keys",
        "groups_total",
        "groups_reused",
        "groups_cache_missed",
        "groups_topup",
        "groups_new",
        "total_groups_new",
        "expanded_measurement_group_probe_count",
        "expanded_measurement_group_probe_count_total",
        "shots_total",
        "shots_reused",
        "shots_new",
        "total_shots_new",
        "reuse_count_cost",
        "candidate_work_event_count",
        "candidate_work_missing_event_count",
        "candidate_count_total",
        "evaluated_count_total",
        "pre_shortlist_count_total",
        "shortlist_size_total",
        "retained_count_total",
        "rejected_count_total",
        "actual_operator_probe_count_total",
        "actual_evaluated_candidate_count_total",
        "reused_operator_probe_count_total",
        "method_input_candidate_count_total",
        "method_shortlist_candidate_count_total",
        "method_retained_candidate_count_total",
        "method_rejected_candidate_count_total",
    )

    def _closed_over(children: Mapping[str, Any]) -> None:
        rows = list(children.values())
        if not rows or any(not isinstance(row, Mapping) for row in rows):
            raise ValueError("Terminal controller-work children are malformed.")
        for key in additive_keys:
            if key not in proxy:
                continue
            values = [row.get(key, 0) for row in rows]
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
                for value in values
            ):
                raise ValueError("Terminal controller-work child drifted.")
            observed = proxy.get(key)
            if (
                isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or not math.isfinite(float(observed))
                or not math.isclose(
                    float(observed),
                    math.fsum(float(value) for value in values),
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
            ):
                raise ValueError("Terminal controller-work totals drifted.")

    _closed_over(by_phase)
    _closed_over(by_scope)
    return proxy


def validate_semantic_phase3_no_positive_terminal_receipt(
    raw_receipt: Mapping[str, Any],
    *,
    route_variant: str,
    route_contract: Mapping[str, Any],
    expected_route_contract_sha256: str,
    accepted_round_count: int,
    terminal_active_prefix_checkpoint: Mapping[str, Any],
    finalization: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate a no-admission Phase-III attempt separately from history."""

    try:
        if not isinstance(raw_receipt, Mapping):
            raise TypeError("terminal receipt must be a mapping")
        receipt = copy.deepcopy(dict(raw_receipt))
        supplied_sha = receipt.pop("sha256", None)
        if supplied_sha != canonical_sha256(receipt):
            raise ValueError("terminal receipt digest drifted")
        if set(receipt) != {
            "schema",
            "terminal_controller_outcome",
            "accepted_controller_round",
            "attempted_controller_round",
            "accepted_state_fingerprint",
            "accepted_operator_count",
            "accepted_state_unchanged",
            "final_admission_record_id",
            "phase0_gradient_shortlist",
            "insertion_mode",
            "insertion_commutation_plateau",
            "insertion_commutation_reduced",
            "phase3_population_activation",
            "controller_measurement_work_proxy",
            "scored_insertion_position_population",
            "projected_phase3_population_receipt",
            "phase123_qiskit_population_normalization_receipts",
            "estimator_event_ids",
            "estimator_event_count",
            "estimator_event_ids_sha256",
            "terminal_active_prefix_checkpoint_sha256",
            "terminal_estimator_prefix_receipt",
            "terminal_estimator_prefix_receipt_sha256",
        }:
            raise ValueError("terminal receipt keys drifted")
        accepted_count = int(accepted_round_count)
        route = validate_semantic_phase3_natural_terminal_route_contract(
            route_contract,
            expected_route_contract_sha256=(
                expected_route_contract_sha256
            ),
        )
        identity = semantic_closure_route_identity(route_variant)
        native = route.get("native_semantic_contract")
        execution = route.get("execution_settings")
        invariants = route.get("semantic_invariants")
        if (
            identity.route_variant
            not in PAPER_I_RA_PHASE3_NATURAL_TERMINAL_ROUTE_VARIANTS
            or not isinstance(native, Mapping)
            or not isinstance(execution, Mapping)
            or not isinstance(invariants, Mapping)
            or native.get("route_variant") != identity.route_variant
            or native.get("phase3_no_positive_policy")
            != ADAPTIVE_PHASE3_NO_POSITIVE_POLICY_TYPED_TERMINAL_V1
            or native.get("controller_horizon_policy")
            != ADAPTIVE_HORIZON_POLICY_MAXIMUM_V1
            or execution.get("ra_phase3_no_positive_policy")
            != ADAPTIVE_PHASE3_NO_POSITIVE_POLICY_TYPED_TERMINAL_V1
            or execution.get("ra_controller_horizon_policy")
            != ADAPTIVE_HORIZON_POLICY_MAXIMUM_V1
            or invariants.get("phase3_no_positive_policy")
            != ADAPTIVE_PHASE3_NO_POSITIVE_POLICY_TYPED_TERMINAL_V1
            or invariants.get("controller_horizon_policy")
            != ADAPTIVE_HORIZON_POLICY_MAXIMUM_V1
        ):
            raise ValueError(
                "terminal receipt requires the authenticated V2 natural-terminal route"
            )
        insertion_mode = receipt["insertion_mode"]
        if (
            not isinstance(execution, Mapping)
            or not isinstance(insertion_mode, str)
            or not insertion_mode
            or insertion_mode != execution.get("adapt_insertion_mode")
        ):
            raise ValueError("terminal insertion mode drifted")
        _validate_terminal_controller_measurement_work_proxy(
            receipt["controller_measurement_work_proxy"]
        )
        checkpoint = copy.deepcopy(dict(terminal_active_prefix_checkpoint))
        unsigned_checkpoint = dict(checkpoint)
        checkpoint_sha256 = unsigned_checkpoint.pop(
            "checkpoint_sha256",
            None,
        )
        event_ids = receipt["estimator_event_ids"]
        prefix = receipt["terminal_estimator_prefix_receipt"]
        if (
            receipt["schema"]
            != "paper_i_ra_phase3_no_positive_selection_terminal_v1"
            or receipt["terminal_controller_outcome"]
            != ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1
            or receipt["accepted_controller_round"] != accepted_count
            or receipt["attempted_controller_round"] != accepted_count + 1
            or not isinstance(receipt["accepted_state_fingerprint"], str)
            or not receipt["accepted_state_fingerprint"]
            or isinstance(receipt["accepted_operator_count"], bool)
            or int(receipt["accepted_operator_count"]) < 0
            or receipt["accepted_state_unchanged"] is not True
            or receipt["final_admission_record_id"] is not None
            or not isinstance(event_ids, list)
            or any(not isinstance(value, str) or not value for value in event_ids)
            or len(set(event_ids)) != len(event_ids)
            or receipt["estimator_event_count"] != len(event_ids)
            or receipt["estimator_event_ids_sha256"]
            != canonical_sha256(event_ids)
            or not isinstance(prefix, Mapping)
            or prefix.get("checkpoint_kind")
            != "terminal_phase3_no_positive"
            or receipt["terminal_estimator_prefix_receipt_sha256"]
            != canonical_sha256(prefix)
            or checkpoint.get("checkpoint_kind")
            != "terminal_phase3_no_positive"
            or checkpoint_sha256 != canonical_sha256(unsigned_checkpoint)
            or checkpoint.get("outer_iteration") != accepted_count
            or checkpoint.get("active_ansatz_depth")
            != receipt["accepted_operator_count"]
            or checkpoint.get("projective_state_fingerprint")
            != receipt["accepted_state_fingerprint"]
            or receipt["terminal_active_prefix_checkpoint_sha256"]
            != canonical_sha256(checkpoint)
        ):
            raise ValueError("terminal controller evidence drifted")

        finalized = copy.deepcopy(dict(finalization))
        continuation = finalized.get("continuation")
        accounting = finalized.get("estimator_call_accounting")
        history = finalized.get("history")
        accepted_checkpoint = (
            history[-1].get("active_prefix_checkpoint")
            if isinstance(history, (list, tuple))
            and len(history) == accepted_count
            and history
            and isinstance(history[-1], Mapping)
            else None
        )
        accepted_history_closed = bool(
            isinstance(history, (list, tuple))
            and len(history) == accepted_count
            and (
                (
                    accepted_count == 0
                    and not history
                    and checkpoint.get("outer_iteration") == 0
                    and checkpoint.get("active_ansatz_depth") == 0
                )
                or (
                    accepted_count > 0
                    and isinstance(accepted_checkpoint, Mapping)
                    and accepted_checkpoint.get("outer_iteration")
                    == accepted_count
                    and accepted_checkpoint.get("active_ansatz_depth")
                    == checkpoint.get("active_ansatz_depth")
                    and accepted_checkpoint.get(
                        "projective_state_fingerprint"
                    )
                    == checkpoint.get("projective_state_fingerprint")
                )
            )
        )
        all_prefixes = (
            continuation.get(
                "all_active_prefix_estimator_ledger_receipts"
            )
            if isinstance(continuation, Mapping)
            else None
        )
        full_ledger = (
            accounting.get("full_ledger")
            if isinstance(accounting, Mapping)
            else None
        )
        occurrences = (
            full_ledger.get("occurrences")
            if isinstance(full_ledger, Mapping)
            else None
        )
        components = ("N_H_outer", "N_H_refit", "N_grad", "N_metric")
        if (
            not isinstance(continuation, Mapping)
            or continuation.get("terminal_phase3_selection_receipt")
            != raw_receipt
            or not accepted_history_closed
            or not isinstance(all_prefixes, list)
            or not all_prefixes
            or all_prefixes[-1] != prefix
            or not isinstance(accounting, Mapping)
            or not isinstance(full_ledger, Mapping)
            or full_ledger.get("schema") != "estimator_call_ledger_v1"
            or full_ledger.get("component_contract") != list(components)
            or not isinstance(occurrences, list)
            or not occurrences
            or any(not isinstance(row, Mapping) for row in occurrences)
        ):
            raise ValueError("terminal estimator evidence is incomplete")
        if set(prefix) != {
            "schema",
            "enabled",
            "status",
            "checkpoint_sequence",
            "outer_iteration",
            "checkpoint_kind",
            "branch_id",
            "parent_branch_id",
            "occurrence_sequence_start_exclusive",
            "occurrence_sequence_end_inclusive",
            "raw_occurrence_delta",
            "executed_query_delta",
            "unique_primitive_delta",
            "cumulative_raw_occurrences",
            "cumulative_executed_queries",
            "cumulative_unique_primitives",
            "runtime_estimator_occurrence_contract",
            "physical_identity_collapse_is_diagnostic_only",
            "raw_occurrences_preserved",
        }:
            raise ValueError("terminal estimator-prefix keys drifted")
        start = int(prefix["occurrence_sequence_start_exclusive"])
        end = int(prefix["occurrence_sequence_end_inclusive"])
        if (
            prefix["schema"]
            != "paper_i_active_prefix_estimator_ledger_receipt_v2"
            or prefix["enabled"] is not True
            or prefix["status"] != "complete"
            or prefix["checkpoint_sequence"] != len(all_prefixes)
            or prefix["outer_iteration"] != accepted_count
            or prefix["checkpoint_kind"]
            != "terminal_phase3_no_positive"
            or prefix["branch_id"] is not None
            or prefix["parent_branch_id"] is not None
            or prefix["runtime_estimator_occurrence_contract"]
            != "all_instrumented_logical_scalar_estimator_calls_v1"
            or prefix["physical_identity_collapse_is_diagnostic_only"]
            is not True
            or prefix["raw_occurrences_preserved"] is not True
            or start < 0
            or end != len(occurrences)
            or start >= end
        ):
            raise ValueError("terminal estimator-prefix provenance drifted")
        normalized_occurrences: list[dict[str, Any]] = []
        for expected_sequence, raw_occurrence in enumerate(
            occurrences,
            start=1,
        ):
            occurrence = dict(raw_occurrence)
            if (
                occurrence.get("sequence") != expected_sequence
                or not isinstance(occurrence.get("primitive_id"), str)
                or not occurrence["primitive_id"]
                or occurrence.get("component") not in components
                or not isinstance(occurrence.get("consumer_scope"), str)
                or not occurrence["consumer_scope"]
                or not isinstance(occurrence.get("charged"), bool)
            ):
                raise ValueError("terminal ledger occurrence drifted")
            normalized_occurrences.append(occurrence)
        delta_occurrences = normalized_occurrences[start:end]

        def _component_counts(
            rows: Sequence[Mapping[str, Any]],
            *,
            charged_only: bool,
        ) -> dict[str, int]:
            return {
                component: sum(
                    1
                    for row in rows
                    if row["component"] == component
                    and (not charged_only or row["charged"] is True)
                )
                for component in components
            }

        delta_raw = _component_counts(
            delta_occurrences,
            charged_only=False,
        )
        cumulative_raw = _component_counts(
            normalized_occurrences,
            charged_only=False,
        )
        delta_unique = _component_counts(
            delta_occurrences,
            charged_only=True,
        )
        cumulative_unique = _component_counts(
            normalized_occurrences,
            charged_only=True,
        )
        if (
            prefix["raw_occurrence_delta"]
            != {"components": delta_raw, "total": len(delta_occurrences)}
            or prefix["executed_query_delta"]
            != {"components": delta_raw, "S_alg": len(delta_occurrences)}
            or prefix["unique_primitive_delta"]
            != {
                "components": delta_unique,
                "S_unique": sum(delta_unique.values()),
            }
            or prefix["cumulative_raw_occurrences"]
            != {
                "components": cumulative_raw,
                "total": len(normalized_occurrences),
            }
            or prefix["cumulative_executed_queries"]
            != {
                "components": cumulative_raw,
                "S_alg": len(normalized_occurrences),
                "unit": "executed_logical_scalar_estimator_invocation",
            }
            or prefix["cumulative_unique_primitives"]
            != {
                "components": cumulative_unique,
                "S_unique": sum(cumulative_unique.values()),
            }
        ):
            raise ValueError("terminal estimator-prefix counters drifted")
        delta_event_ids = [
            f"estimator:{row['sequence']}:{row['primitive_id']}"
            for row in delta_occurrences
        ]
        if (
            not event_ids
            or len(event_ids) > len(delta_event_ids)
            or event_ids != delta_event_ids[-len(event_ids) :]
        ):
            raise ValueError("terminal selection events left the ledger tail")

        scored = receipt["scored_insertion_position_population"]
        if not isinstance(scored, Mapping):
            raise TypeError("terminal scored population must be a mapping")
        scored_payload = copy.deepcopy(dict(scored))
        scored_sha = scored_payload.pop("sha256", None)
        if scored_sha != canonical_sha256(scored_payload):
            raise ValueError("terminal scored population digest drifted")
        phase_rows = scored.get("phases")
        if (
            scored.get("schema")
            != "paper_i_scored_insertion_position_population_v1"
            or not isinstance(phase_rows, list)
            or len(phase_rows) != 3
            or [row.get("phase") for row in phase_rows]
            != ["phase_i", "phase_ii", "phase_iii"]
        ):
            raise ValueError("terminal phase population order drifted")
        selections = {}
        for phase_row, phase_name, score_key, hard_cap in zip(
            phase_rows,
            ("phase_i", "phase_ii", "phase_iii"),
            ("phase1_active_score", "phase2_raw_score", "full_v2_score"),
            (24, 12, 12),
            strict=True,
        ):
            selections[phase_name] = (
                adaptive_phase_selection_receipt_from_mapping(
                    phase_row,
                    expected_phase=phase_name,
                    expected_score_key=score_key,
                    expected_hard_cap=hard_cap,
                    expected_frontier_ratio=0.9,
                )
            )
        phase3_selection = selections["phase_iii"]
        if (
            phase3_selection.adaptive_shortlist.status
            != "no_positive_population"
            or phase3_selection.adaptive_retained_count != 0
            or phase3_selection.final_singleton_count != 0
            or phase_rows[2].get("terminal_outcome")
            != ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1
        ):
            raise ValueError("terminal Phase-III decision drifted")

        phase0 = receipt["phase0_gradient_shortlist"]
        if not isinstance(phase0, Mapping):
            raise TypeError("terminal Phase-0 receipt is missing")
        if route_variant in PAPER_I_RA_PHASE0_POSITION_ROUTE_VARIANTS:
            validate_semantic_position_phase0_receipt(
                phase0,
                scored_population=scored,
            )
            retained_rows = phase0.get("retained_records")
            phase_i_rows = phase_rows[0].get("records")
            if (
                not isinstance(retained_rows, list)
                or not retained_rows
                or not isinstance(phase_i_rows, list)
                or len(retained_rows) != len(phase_i_rows)
            ):
                raise ValueError("terminal Phase-0 to Phase-I link is missing")

            def _coordinate(row: Mapping[str, Any]) -> tuple[str, int, int, str]:
                return (
                    str(row.get("domain_record_id", "")),
                    int(row.get("pool_index", -1)),
                    int(row.get("insertion_position", -1)),
                    str(row.get("position_class", "")),
                )

            phase0_by_coordinate = {
                _coordinate(row): row for row in retained_rows
            }
            phase_i_by_coordinate = {
                _coordinate(row): row for row in phase_i_rows
            }
            if (
                len(phase0_by_coordinate) != len(retained_rows)
                or len(phase_i_by_coordinate) != len(phase_i_rows)
                or set(phase0_by_coordinate) != set(phase_i_by_coordinate)
            ):
                raise ValueError("terminal direct Phase-I membership drifted")
            for coordinate in phase0_by_coordinate:
                pool_index = coordinate[1]
                phase0_row = phase0_by_coordinate[coordinate]
                phase_i_row = phase_i_by_coordinate[coordinate]
                physical_id = str(phase_i_row.get("generator_id", ""))
                if (
                    not physical_id
                    or str(phase0_row.get("generator_id", ""))
                    != f"{physical_id}::pool[{pool_index}]"
                    or str(phase0_row.get("pool_label", ""))
                    != str(phase_i_row.get("pool_label", ""))
                ):
                    raise ValueError("terminal Phase-0 identity drifted")
        elif route_variant in {
            PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2,
            PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1,
            PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
            PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
        }:
            validate_semantic_gradient_adaptive_phase0_receipt(
                phase0,
                scored_population=scored,
            )
        else:
            validate_semantic_proxy_phase0_receipt(
                phase0,
                scored_population=scored,
            )

        append_position = int(scored.get("append_position", -1))
        if append_position < 0:
            raise ValueError("terminal append position drifted")

        def _record_pairs(rows: Any) -> list[tuple[int, int]]:
            if not isinstance(rows, list) or not rows:
                raise ValueError("terminal insertion records are missing")
            pairs: list[tuple[int, int]] = []
            for row in rows:
                if not isinstance(row, Mapping):
                    raise ValueError("terminal insertion record is malformed")
                pool_index = row.get("pool_index")
                position = row.get("insertion_position")
                if (
                    isinstance(pool_index, bool)
                    or isinstance(position, bool)
                    or not isinstance(pool_index, int)
                    or not isinstance(position, int)
                    or pool_index < 0
                    or position < 0
                    or position > append_position
                ):
                    raise ValueError("terminal insertion coordinate drifted")
                pairs.append((pool_index, position))
            if len(pairs) != len(set(pairs)):
                raise ValueError("terminal insertion coordinates repeat")
            return pairs

        phase_i_pairs = _record_pairs(phase_rows[0].get("records"))
        if route_variant in PAPER_I_RA_PHASE0_POSITION_ROUTE_VARIANTS:
            representative_pairs = _record_pairs(phase0.get("population"))
            expected_phase_i_pairs = _record_pairs(
                phase0.get("retained_records")
            )
        else:
            representative_pairs = None
            retained_indices = phase0.get("retained_pool_indices")
            if (
                not isinstance(retained_indices, list)
                or not retained_indices
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in retained_indices
                )
                or len(retained_indices) != len(set(retained_indices))
            ):
                raise ValueError("terminal Phase-0 retained generators drifted")
            retained_pool_indices = set(retained_indices)
            expected_phase_i_pairs = []

        plateau_receipt = receipt["insertion_commutation_plateau"]
        reduced_receipt = receipt["insertion_commutation_reduced"]
        domain_receipt: Mapping[str, Any] | None = None
        expected_policy: str | None = None
        if insertion_mode == "append_only":
            if plateau_receipt is not None or reduced_receipt is not None:
                raise ValueError("append-only terminal gained a domain receipt")
            if route_variant not in PAPER_I_RA_PHASE0_POSITION_ROUTE_VARIANTS:
                expected_phase_i_pairs = [
                    (pool_index, append_position)
                    for pool_index in retained_indices
                ]
            if set(phase_i_pairs) != set(expected_phase_i_pairs):
                raise ValueError("append-only terminal Phase-I domain drifted")
            for phase_row in phase_rows:
                if any(
                    position != append_position
                    for _pool_index, position in _record_pairs(
                        phase_row.get("records")
                    )
                ):
                    raise ValueError("append-only terminal left the endpoint")
        elif insertion_mode in {
            "insertion_commutation_plateau_v1",
            "insertion_commutation_plateau_v2",
        }:
            if not isinstance(plateau_receipt, Mapping) or reduced_receipt is not None:
                raise ValueError("plateau terminal domain receipt drifted")
            domain_receipt = plateau_receipt
            expected_policy = insertion_mode
        elif insertion_mode == "full_commutation_reduced":
            if plateau_receipt is not None or not isinstance(
                reduced_receipt, Mapping
            ):
                raise ValueError("always-open terminal domain receipt drifted")
            domain_receipt = reduced_receipt
            expected_policy = "always_commutation_reduced"
        elif insertion_mode == "append_commutation_reduced":
            if plateau_receipt is not None or not isinstance(
                reduced_receipt, Mapping
            ):
                raise ValueError("append-reduced terminal domain receipt drifted")
            domain_receipt = reduced_receipt
            expected_policy = APPEND_COMMUTATION_REDUCED_POLICY
        else:
            raise ValueError("terminal insertion mode is unsupported")

        if domain_receipt is not None:
            domain_open = bool(domain_receipt.get("domain_open", False))
            expected_positions = (
                list(range(append_position + 1))
                if insertion_mode == "full_commutation_reduced"
                or (
                    insertion_mode.startswith("insertion_commutation_plateau_")
                    and domain_open
                )
                else [append_position]
            )
            if route_variant not in PAPER_I_RA_PHASE0_POSITION_ROUTE_VARIANTS:
                raw_representatives = domain_receipt.get(
                    "retained_representatives"
                )
                if not isinstance(raw_representatives, list):
                    raise ValueError("terminal insertion representatives drifted")
                expected_phase_i_pairs = [
                    (int(row["candidate_pool_index"]), int(position))
                    for row in raw_representatives
                    if isinstance(row, Mapping)
                    and int(row.get("candidate_pool_index", -1))
                    in retained_pool_indices
                    for position in row.get("positions", [])
                ]
            validate_commutation_reduced_insertion_receipt(
                domain_receipt,
                expected_policy=expected_policy,
                expected_requested_positions=expected_positions,
                scored_population=scored,
                expected_representative_pairs=representative_pairs,
                expected_phase_i_pairs=expected_phase_i_pairs,
            )

        activation = receipt["phase3_population_activation"]
        expected_activation_policy = execution.get(
            "ra_phase3_population_activation_policy",
            RA_ADAPT_PHASE3_POPULATION_ALL_ROUNDS,
        )
        if (
            not isinstance(activation, Mapping)
            or activation.get("policy") != expected_activation_policy
            or not isinstance(
                activation.get("competitive_population_live"), bool
            )
        ):
            raise ValueError("terminal Phase-III activation drifted")
        if expected_activation_policy == RA_ADAPT_PHASE3_POPULATION_ALL_ROUNDS:
            expected_activation = {
                "schema": "ra_phase3_population_activation_receipt_v1",
                "policy": RA_ADAPT_PHASE3_POPULATION_ALL_ROUNDS,
                "competitive_population_live": True,
                "activation_source": "route_default_all_rounds_v1",
                "preplateau_admission_authority": None,
                "winner_materialization_policy": None,
                "insertion_plateau_domain_open": None,
                "independent_latch_active": False,
                "hysteresis_active": False,
            }
            if dict(activation) != expected_activation:
                raise ValueError("terminal all-round activation drifted")
        elif expected_activation_policy == (
            RA_ADAPT_PHASE3_POPULATION_ON_INSERTION_PLATEAU
        ):
            if (
                not isinstance(plateau_receipt, Mapping)
                or activation.get("schema")
                != "ra_phase3_population_activation_receipt_v1"
                or activation.get("competitive_population_live")
                is not plateau_receipt.get("domain_open")
                or activation.get("insertion_plateau_domain_open")
                is not plateau_receipt.get("domain_open")
                or activation.get("independent_latch_active") is not False
                or activation.get("hysteresis_active") is not False
            ):
                raise ValueError("terminal plateau activation drifted")
        elif expected_activation_policy == (
            RA_ADAPT_PHASE3_POPULATION_LATCHED_ON_PROGRESS_PLATEAU
        ):
            entry = activation.get("entry_plateau_receipt")
            if (
                not isinstance(plateau_receipt, Mapping)
                or not isinstance(entry, Mapping)
                or activation.get("schema")
                != "ra_phase3_population_activation_receipt_v2"
                or activation.get("independent_latch_active") is not True
                or activation.get("deactivation_allowed") is not False
                or activation.get("hysteresis_active") is not False
                or activation.get("phase3_latched_after_round")
                is not activation.get("competitive_population_live")
                or activation.get("insertion_plateau_domain_open")
                is not plateau_receipt.get("domain_open")
            ):
                raise ValueError("terminal latched activation drifted")
        else:
            raise ValueError("terminal Phase-III activation policy is unknown")

        projected_raw = receipt["projected_phase3_population_receipt"]
        if not isinstance(projected_raw, Mapping):
            raise TypeError("terminal projected population is missing")
        projected = validate_semantic_projected_phase123_receipt(
            projected_raw
        )
        projected_activation = projected.get(
            "competitive_population_activation"
        )
        if expected_activation_policy in {
            RA_ADAPT_PHASE3_POPULATION_ON_INSERTION_PLATEAU,
            RA_ADAPT_PHASE3_POPULATION_LATCHED_ON_PROGRESS_PLATEAU,
        }:
            competitive_count = projected.get(
                "competitive_population_input_count"
            )
            available_count = projected.get(
                "phase2_available_shortlist_count"
            )
            live = bool(activation["competitive_population_live"])
            if (
                projected_activation != activation
                or isinstance(competitive_count, bool)
                or not isinstance(competitive_count, int)
                or isinstance(available_count, bool)
                or not isinstance(available_count, int)
                or competitive_count < 1
                or available_count < competitive_count
                or (live and competitive_count != available_count)
                or (not live and competitive_count != 1)
            ):
                raise ValueError("terminal projected activation drifted")
        elif projected_activation is not None:
            raise ValueError("terminal all-round activation was reprojected")
        qiskit = projected[
            "phase123_qiskit_population_normalization_receipts"
        ]
        if receipt[
            "phase123_qiskit_population_normalization_receipts"
        ] != qiskit:
            raise ValueError("terminal Qiskit population evidence drifted")
        population_link_sha256 = {}
        for phase_name, phase_row in zip(
            ("phase_i", "phase_ii", "phase_iii"),
            phase_rows,
            strict=True,
        ):
            scored_rows = phase_row["records"]
            qiskit_rows = qiskit[phase_name]["rows"]
            scored_identities = {
                (
                    str(row.get("generator_id", "")),
                    int(row.get("pool_index", -1)),
                    int(row.get("insertion_position", -1)),
                    str(row.get("pool_label", "")),
                )
                for row in scored_rows
            }
            qiskit_identities = {
                (
                    str(row.get("generator_id", "")),
                    int(row.get("candidate_pool_index", -1)),
                    int(row.get("position_id", -1)),
                    str(row.get("candidate_label", "")),
                )
                for row in qiskit_rows
            }
            if (
                len(scored_identities) != len(scored_rows)
                or len(qiskit_identities) != len(qiskit_rows)
                or scored_identities != qiskit_identities
            ):
                raise ValueError("terminal adaptive/Qiskit populations drifted")
            population_link_sha256[phase_name] = canonical_sha256(
                {
                    "phase": phase_name,
                    "identities": sorted(scored_identities),
                }
            )
        return {
            **receipt,
            "sha256": str(supplied_sha),
            "phase123_adaptive_shortlist_receipt_sha256": {
                phase: selection.adaptive_shortlist.sha256
                for phase, selection in selections.items()
            },
            "phase123_qiskit_adaptive_population_link_sha256": (
                population_link_sha256
            ),
        }
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Invalid semantic Phase-III no-positive terminal receipt: "
            f"{exc}"
        ) from exc


def validate_semantic_final_selector_accounting(
    *,
    algorithm_id: str,
    route_contract: Mapping[str, Any],
    selector_compile_cost_accounting: Mapping[str, Any],
    finalization: Mapping[str, Any],
    accepted_round_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Close native Phase-0 and Phase-I--III Qiskit evidence at finalization."""

    try:
        identity = semantic_closure_route_identity_from_algorithm(
            str(algorithm_id)
        )
        if identity.route_variant not in PAPER_I_RA_PHASE0_EXECUTABLE_ROUTE_VARIANTS:
            raise RuntimeError("The v1 semantic accounting route is retired.")
        route = dict(route_contract)
        native = route.get("native_semantic_contract")
        execution = route.get("execution_settings")
        route_sha256 = route.get("sha256")
        unsigned_route = dict(route)
        unsigned_route.pop("sha256", None)
        executed_route = finalization.get("sr_route_profile_contract")
        executed_route_sha256 = finalization.get(
            "sr_route_profile_contract_sha256"
        )
        if (
            route.get("route_id") != identity.route_id
            or route.get("algorithm_id") != identity.algorithm_id
            or route.get("semantic_implementation_version")
            != identity.semantic_implementation_version
            or route_sha256 != canonical_sha256(unsigned_route)
            or not isinstance(executed_route, Mapping)
            or dict(executed_route) != unsigned_route
            or executed_route_sha256 != route_sha256
            or not isinstance(native, Mapping)
            or native.get("route_variant") != identity.route_variant
            or native.get("compile_scope")
            != PAPER_I_RA_PHASE123_QISKIT_COMPILE_SCOPE
            or native.get("hardware_cost_normalization")
            != "zero_centered_signed_arctan_v1"
            or native.get("compile_work_in_s_alg") is not False
            or not isinstance(execution, Mapping)
            or execution.get("phase3_backend_cost_scope")
            != PAPER_I_RA_PHASE123_QISKIT_COMPILE_SCOPE
            or execution.get("phase3_hardware_cost_normalization_mode")
            != "zero_centered_signed_arctan_v1"
        ):
            raise RuntimeError("Semantic route identity is invalid.")

        accounting = dict(selector_compile_cost_accounting)
        qiskit = accounting.get("phase_iii")
        targets = qiskit.get("targets") if isinstance(qiskit, Mapping) else None
        if (
            accounting.get("schema")
            != "paper_i_selector_compile_cost_accounting_v1"
            or accounting.get("scope")
            != PAPER_I_RA_PHASE123_QISKIT_COMPILE_SCOPE
            or accounting.get("excluded_from_s_alg") is not True
            or "S_alg" in accounting
            or accounting.get("phase_i_cost_source")
            != "backend_transpile_v1"
            or accounting.get("qiskit_applied_phases")
            != ["phase_i", "phase_ii", "phase_iii"]
            or accounting.get("phase_i_phase_ii") is not None
            or accounting.get("phase_iii_reuses_phase_i_phase_ii_oracle")
            is not False
            or not isinstance(qiskit, Mapping)
            or qiskit.get("role") != "phase_i_phase_ii_phase_iii"
            or qiskit.get("mode") != "transpile_single_v1"
            or qiskit.get("optimization_level") != 1
            or qiskit.get("seed_transpiler") != 7
            or qiskit.get("structure_theta_value") != 1.0
            or qiskit.get("negative_delta_reward_enabled") is not True
            or qiskit.get("preferred_backend_fallback_allowed") is not False
            or qiskit.get("one_qubit_coordinate_policy")
            != ONE_QUBIT_COORDINATE_COMPILED_POSITIVE_DELTA_V1
            or not isinstance(targets, list)
            or len(targets) != 1
            or not isinstance(targets[0], Mapping)
            or targets[0].get("resolved_name") != "FakeMarrakesh"
            or targets[0].get("resolution_kind") != "fake_exact"
        ):
            raise RuntimeError("Selector compile accounting is invalid.")

        history = finalization.get("history")
        accepted = tuple(accepted_round_receipts)
        terminal_outcome = finalization.get("terminal_controller_outcome")
        terminal_phase0 = finalization.get(
            "terminal_phase0_selection_receipt"
        )
        terminal_phase3 = finalization.get(
            "terminal_phase3_selection_receipt"
        )
        stationary_terminal = bool(
            terminal_outcome
            == "phase0_stationary_no_competitive_candidate_v1"
        )
        phase3_no_positive_terminal = bool(
            terminal_outcome
            == ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1
        )
        if (
            not isinstance(history, (list, tuple))
            or len(history) != len(accepted)
            or any(not isinstance(row, Mapping) for row in history)
            or any(not isinstance(row, Mapping) for row in accepted)
            or (
                not history
                and not (
                    stationary_terminal or phase3_no_positive_terminal
                )
            )
        ):
            raise RuntimeError("Accepted-round accounting is incomplete.")
        if stationary_terminal:
            adaptive_variants = {
                PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2,
                PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2,
                PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1,
                PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
                PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
                PAPER_I_RA_PHASE0_POSITION_GRADIENT_ADAPTIVE_V1,
                PAPER_I_RA_PHASE0_POSITION_PROXY_ADAPTIVE_V1,
                PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1,
                PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
                PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
            }
            if (
                identity.route_variant not in adaptive_variants
                or not isinstance(terminal_phase0, Mapping)
                or terminal_phase0.get("status") != "stationary"
                or terminal_phase0.get("retained_candidate_count") != 0
                or (
                    terminal_phase0.get("retained_records") != []
                    if identity.route_variant
                    in PAPER_I_RA_PHASE0_POSITION_ROUTE_VARIANTS
                    else terminal_phase0.get("retained_pool_indices") != []
                )
                or terminal_phase0.get("terminal_controller_outcome")
                != terminal_outcome
            ):
                raise RuntimeError(
                    "Stationary Phase-0 terminal evidence is invalid."
                )
            if identity.route_variant in PAPER_I_RA_PHASE0_POSITION_ROUTE_VARIANTS:
                validate_semantic_position_phase0_receipt(terminal_phase0)
            elif identity.route_variant in {
                PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2,
                PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1,
                PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
                PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_FORCED_K50_V1,
                PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
            }:
                validate_semantic_gradient_adaptive_phase0_receipt(
                    terminal_phase0
                )
            else:
                validate_semantic_proxy_phase0_receipt(terminal_phase0)
        elif phase3_no_positive_terminal:
            if (
                terminal_phase0 is not None
                or identity.route_variant
                not in PAPER_I_RA_PHASE3_NATURAL_TERMINAL_ROUTE_VARIANTS
                or not isinstance(terminal_phase3, Mapping)
                or not isinstance(
                    finalization.get("terminal_active_prefix_checkpoint"),
                    Mapping,
                )
            ):
                raise RuntimeError(
                    "Phase-III no-positive terminal evidence is incomplete."
                )
        elif (
            terminal_phase0 is not None
            or terminal_phase3 is not None
            or terminal_outcome is not None
        ):
            raise RuntimeError("Unexpected semantic terminal evidence.")

        closed_rounds: list[dict[str, Any]] = []
        required_phases = ("phase_i", "phase_ii", "phase_iii")
        for round_index, (raw_history, accepted_round) in enumerate(
            zip(history, accepted, strict=True),
            start=1,
        ):
            phase0 = accepted_round.get("ra_gradient_phase0_shortlist")
            scored = accepted_round.get(
                "scored_insertion_position_population"
            )
            if not isinstance(phase0, Mapping) or not isinstance(
                scored, Mapping
            ):
                raise RuntimeError("Phase-0 accepted evidence is missing.")
            if identity.route_variant in PAPER_I_RA_PHASE0_POSITION_ROUTE_VARIANTS:
                validate_semantic_position_phase0_receipt(
                    phase0,
                    scored_population=scored,
                )
            elif identity.route_variant == PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2:
                from pipelines.static_adapt.ra_adapt.phase0 import (
                    GLOBAL_SINGLETON_ABSOLUTE_GRADIENT_PHASE0_POLICY,
                    GLOBAL_SINGLETON_GRADIENT_PHASE0_RECEIPT_SCHEMA,
                )

                phase0_accounting = phase0.get("estimator_accounting")
                if (
                    phase0.get("schema")
                    != GLOBAL_SINGLETON_GRADIENT_PHASE0_RECEIPT_SCHEMA
                    or phase0.get("policy")
                    != GLOBAL_SINGLETON_ABSOLUTE_GRADIENT_PHASE0_POLICY
                    or phase0.get("compile_cost_policy") != "off"
                    or phase0.get("metric_policy") != "off"
                    or phase0.get("requested_shortlist_size") != 24
                    or not isinstance(phase0_accounting, Mapping)
                    or phase0_accounting.get("S_alg")
                    != phase0.get("input_candidate_count")
                ):
                    raise RuntimeError("Gradient Phase-0 receipt is invalid.")
            elif identity.route_variant in {
                PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2,
                PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1,
                PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
                PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_FORCED_K50_V1,
                PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
            }:
                validate_semantic_gradient_adaptive_phase0_receipt(
                    phase0,
                    scored_population=scored,
                )
            else:
                validate_semantic_proxy_phase0_receipt(
                    phase0,
                    scored_population=scored,
                )

            adaptive_phase123_sha256: dict[str, str] = {}
            adaptive_phase123_selections: dict[str, Any] = {}
            phase0_phase_i_direct_population_link_sha256: str | None = None
            if identity.route_variant in (
                PAPER_I_RA_ALL_PHASE_ADAPTIVE_ROUTE_VARIANTS
            ):
                scored_phases = scored.get("phases")
                if (
                    not isinstance(scored_phases, list)
                    or any(not isinstance(row, Mapping) for row in scored_phases)
                    or [row.get("phase") for row in scored_phases]
                    != list(required_phases)
                ):
                    raise RuntimeError(
                        "Adaptive Phase-I--III scored populations are incomplete."
                    )
                for phase_name, phase_row in zip(
                    required_phases,
                    scored_phases,
                    strict=True,
                ):
                    expected_score_key = {
                        "phase_i": "phase1_active_score",
                        "phase_ii": "phase2_raw_score",
                        "phase_iii": "full_v2_score",
                    }[phase_name]
                    expected_cap = {
                        "phase_i": 24,
                        "phase_ii": 12,
                        "phase_iii": 12,
                    }[phase_name]
                    validated_adaptive = (
                        adaptive_phase_selection_receipt_from_mapping(
                            phase_row,
                            expected_phase=phase_name,
                            expected_score_key=expected_score_key,
                            expected_hard_cap=expected_cap,
                            expected_frontier_ratio=0.9,
                        )
                    )
                    adaptive_phase123_sha256[phase_name] = (
                        validated_adaptive.adaptive_shortlist.sha256
                    )
                    adaptive_phase123_selections[phase_name] = (
                        validated_adaptive
                    )
                if identity.route_variant in {
                    PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1,
                    PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
                    PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_FORCED_K50_V1,
                    PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
                    PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
                }:
                    retained_rows = phase0.get("retained_records")
                    if (
                        not isinstance(retained_rows, list)
                        or not retained_rows
                        or any(
                            not isinstance(row, Mapping)
                            for row in retained_rows
                        )
                    ):
                        raise RuntimeError(
                            "Position-record Phase-0 retained population is "
                            "missing."
                        )
                    phase_i_rows = scored_phases[0].get("records")
                    if (
                        not isinstance(phase_i_rows, list)
                        or any(
                            not isinstance(row, Mapping)
                            for row in phase_i_rows
                        )
                    ):
                        raise RuntimeError(
                            "Position-record Phase-I population is missing."
                        )

                    def _domain_coordinate(
                        row: Mapping[str, Any],
                    ) -> tuple[str, int, int, str]:
                        return (
                            str(row.get("domain_record_id", "")),
                            int(row.get("pool_index", -1)),
                            int(row.get("insertion_position", -1)),
                            str(row.get("position_class", "")),
                        )

                    phase0_record_ids = tuple(
                        _domain_coordinate(row) for row in retained_rows
                    )
                    phase_i_record_ids = tuple(
                        _domain_coordinate(row) for row in phase_i_rows
                    )
                    phase0_by_coordinate = {
                        _domain_coordinate(row): row
                        for row in retained_rows
                    }
                    phase_i_by_coordinate = {
                        _domain_coordinate(row): row
                        for row in phase_i_rows
                    }
                    if (
                        len(set(phase0_record_ids))
                        != len(phase0_record_ids)
                        or len(set(phase_i_record_ids))
                        != len(phase_i_record_ids)
                        or any(
                            not domain_record_id
                            or pool_index < 0
                            or insertion_position < 0
                            or position_class not in {"interior", "append"}
                            for (
                                domain_record_id,
                                pool_index,
                                insertion_position,
                                position_class,
                            ) in (*phase0_record_ids, *phase_i_record_ids)
                        )
                        or len(phase_i_record_ids) != len(phase0_record_ids)
                        or set(phase_i_record_ids) != set(phase0_record_ids)
                    ):
                        raise RuntimeError(
                            "Position-record Phase-0 was not passed directly "
                            "to Phase I."
                        )
                    for coordinate in phase0_record_ids:
                        phase0_row = phase0_by_coordinate[coordinate]
                        phase_i_row = phase_i_by_coordinate[coordinate]
                        pool_index = int(coordinate[1])
                        phase0_generator_id = str(
                            phase0_row.get("generator_id", "")
                        )
                        phase_i_generator_id = str(
                            phase_i_row.get("generator_id", "")
                        )
                        phase0_pool_label = str(
                            phase0_row.get("pool_label", "")
                        )
                        phase_i_pool_label = str(
                            phase_i_row.get("pool_label", "")
                        )
                        if (
                            not phase_i_generator_id
                            or phase0_generator_id
                            != (
                                f"{phase_i_generator_id}::pool["
                                f"{pool_index}]"
                            )
                            or not phase0_pool_label
                            or phase0_pool_label != phase_i_pool_label
                        ):
                            raise RuntimeError(
                                "Position-record Phase-0 controller identity "
                                "did not normalize to the Phase-I physical "
                                "identity."
                            )
                    phase0_phase_i_direct_population_link_sha256 = (
                        canonical_sha256(
                            {
                                "phase0_retained_domain_coordinates": list(
                                    phase0_record_ids
                                ),
                                "phase_i_population_domain_coordinates": list(
                                    phase_i_record_ids
                                ),
                                "phase0_retained_controller_generator_ids": [
                                    str(
                                        phase0_by_coordinate[coordinate].get(
                                            "generator_id", ""
                                        )
                                    )
                                    for coordinate in phase0_record_ids
                                ],
                                "phase_i_population_physical_generator_ids": [
                                    str(
                                        phase_i_by_coordinate[coordinate].get(
                                            "generator_id", ""
                                        )
                                    )
                                    for coordinate in phase_i_record_ids
                                ],
                                "phase0_retained_pool_labels": [
                                    str(
                                        phase0_by_coordinate[coordinate].get(
                                            "pool_label", ""
                                        )
                                    )
                                    for coordinate in phase0_record_ids
                                ],
                                "phase_i_population_pool_labels": [
                                    str(
                                        phase_i_by_coordinate[coordinate].get(
                                            "pool_label", ""
                                        )
                                    )
                                    for coordinate in phase_i_record_ids
                                ],
                            }
                        )
                    )

            projected = raw_history.get(
                "projected_phase3_population_receipt"
            )
            if not isinstance(projected, Mapping):
                raise RuntimeError(
                    "Phase-I--III projected population receipt is missing."
                )
            validated_projected = (
                validate_semantic_projected_phase123_receipt(projected)
            )
            if accepted_round.get(
                "projected_phase3_population_receipt"
            ) != validated_projected:
                raise RuntimeError(
                    "Accepted Phase-I--III population evidence drifted from "
                    "the validated runtime receipt."
                )
            validated = validated_projected[
                "phase123_qiskit_population_normalization_receipts"
            ]
            phase123_population_link_sha256: dict[str, str] = {}
            if adaptive_phase123_selections:
                for phase_name in required_phases:
                    selection = adaptive_phase123_selections[phase_name]
                    qiskit_rows = validated[phase_name].get("rows")
                    if not isinstance(qiskit_rows, list):
                        raise RuntimeError(
                            "Adaptive Qiskit population rows are missing."
                        )
                    qiskit_record_ids = tuple(
                        adaptive_phase_record_id(
                            generator_id=str(row.get("generator_id", "")),
                            pool_index=int(
                                row.get("candidate_pool_index", -1)
                            ),
                            insertion_position=int(
                                row.get("position_id", -1)
                            ),
                        )
                        for row in qiskit_rows
                    )
                    scored_record_ids = tuple(
                        selection.population_record_ids
                    )
                    scored_phase_row = scored_phases[
                        required_phases.index(phase_name)
                    ]
                    scored_rows = scored_phase_row.get("records")
                    if (
                        not isinstance(scored_rows, list)
                        or any(
                            not isinstance(row, Mapping)
                            for row in scored_rows
                        )
                    ):
                        raise RuntimeError(
                            "Adaptive scored phase population rows are missing."
                        )
                    scored_identity_rows = tuple(
                        (
                            str(row.get("generator_id", "")),
                            int(row.get("pool_index", -1)),
                            int(row.get("insertion_position", -1)),
                            str(row.get("pool_label", "")),
                        )
                        for row in scored_rows
                    )
                    qiskit_identity_rows = tuple(
                        (
                            str(row.get("generator_id", "")),
                            int(row.get("candidate_pool_index", -1)),
                            int(row.get("position_id", -1)),
                            str(row.get("candidate_label", "")),
                        )
                        for row in qiskit_rows
                    )
                    expected_identity_rows = scored_identity_rows
                    expected_record_ids = scored_record_ids
                    child_identity_rows: tuple[
                        tuple[str, int, int, str], ...
                    ] = ()
                    if phase_name == "phase_iii":
                        # Phase-III parent splits evaluate additional child
                        # candidates whose Qiskit compile costs are measured
                        # under authenticated ``child:`` identities.  The
                        # evaluated-population receipt is the authority for
                        # what Phase III measured: it must contain the scored
                        # parents exactly, plus only declared child rows.
                        evaluated_rows = validated_projected.get(
                            "phase3_evaluated_population_identities"
                        )
                        if isinstance(evaluated_rows, list) and evaluated_rows:
                            if validated_projected.get(
                                "phase3_evaluated_population_identities_sha256"
                            ) != canonical_sha256(evaluated_rows):
                                raise RuntimeError(
                                    "Phase-III evaluated population identity "
                                    "digest drifted."
                                )
                            evaluated_identity_rows = tuple(
                                (
                                    str(row.get("generator_id", "")),
                                    int(row.get("candidate_pool_index", -1)),
                                    int(row.get("position_id", -1)),
                                    str(row.get("candidate_label", "")),
                                )
                                for row in evaluated_rows
                            )
                            child_identity_rows = tuple(
                                row
                                for row in evaluated_identity_rows
                                if row not in set(scored_identity_rows)
                            )
                            if (
                                len(set(evaluated_identity_rows))
                                != len(evaluated_identity_rows)
                                or not set(scored_identity_rows)
                                <= set(evaluated_identity_rows)
                                or any(
                                    not row[0].startswith("child:")
                                    for row in child_identity_rows
                                )
                            ):
                                raise RuntimeError(
                                    "Phase-III child evaluation identities "
                                    "drifted."
                                )
                            expected_identity_rows = evaluated_identity_rows
                            expected_record_ids = tuple(
                                adaptive_phase_record_id(
                                    generator_id=row[0],
                                    pool_index=row[1],
                                    insertion_position=row[2],
                                )
                                for row in evaluated_identity_rows
                            )
                    if (
                        len(qiskit_record_ids) != len(expected_record_ids)
                        or len(set(qiskit_record_ids))
                        != len(qiskit_record_ids)
                        or set(qiskit_record_ids)
                        != set(expected_record_ids)
                        or len(scored_identity_rows)
                        != selection.population_count
                        or len(set(scored_identity_rows))
                        != len(scored_identity_rows)
                        or len(set(qiskit_identity_rows))
                        != len(qiskit_identity_rows)
                        or set(qiskit_identity_rows)
                        != set(expected_identity_rows)
                    ):
                        raise RuntimeError(
                            "Adaptive and Qiskit phase populations drifted."
                        )
                    link_payload: dict[str, Any] = {
                        "phase": phase_name,
                        "scored_record_ids": list(
                            scored_record_ids
                        ),
                        "qiskit_record_ids": list(
                            qiskit_record_ids
                        ),
                        "scored_identity_rows": list(
                            scored_identity_rows
                        ),
                        "qiskit_identity_rows": list(
                            qiskit_identity_rows
                        ),
                    }
                    if child_identity_rows:
                        link_payload["phase3_child_identity_rows"] = list(
                            child_identity_rows
                        )
                    phase123_population_link_sha256[phase_name] = (
                        canonical_sha256(link_payload)
                    )
            phase0_sha = phase0.get("sha256")
            if not _is_sha256(phase0_sha):
                raise RuntimeError("Phase-0 receipt digest is invalid.")
            closed_round = {
                "accepted_round": round_index,
                "phase0_receipt_sha256": str(phase0_sha),
                "phase123_population_receipt_sha256": {
                    phase: canonical_sha256(validated[phase])
                    for phase in required_phases
                },
            }
            if adaptive_phase123_sha256:
                closed_round[
                    "phase123_adaptive_shortlist_receipt_sha256"
                ] = adaptive_phase123_sha256
                closed_round[
                    "phase123_qiskit_adaptive_population_link_sha256"
                ] = phase123_population_link_sha256
            if phase0_phase_i_direct_population_link_sha256 is not None:
                closed_round[
                    "phase0_phase_i_direct_population_link_sha256"
                ] = phase0_phase_i_direct_population_link_sha256
            closed_rounds.append(closed_round)

        closure: dict[str, Any] = {
            "schema": (
                "paper_i_ra_semantic_final_selector_accounting_closure_v1"
            ),
            "route_id": identity.route_id,
            "algorithm_id": identity.algorithm_id,
            "route_variant": identity.route_variant,
            "semantic_implementation_version": (
                identity.semantic_implementation_version
            ),
            "route_contract_sha256": str(route_sha256),
            "selector_compile_cost_accounting_sha256": canonical_sha256(
                accounting
            ),
            "validated_round_count": len(closed_rounds),
            "qiskit_phases": list(required_phases),
            "population_normalization_count_per_phase_per_round": 1,
            "qiskit_compile_work_excluded_from_s_alg": True,
            "rounds": closed_rounds,
        }
        if stationary_terminal:
            closure.update(
                {
                    "terminal_controller_outcome": str(
                        terminal_outcome
                    ),
                    "terminal_phase0_receipt_sha256": str(
                        terminal_phase0["sha256"]
                    ),
                    "phase_i_entered_after_terminal_phase0": False,
                }
            )
        if phase3_no_positive_terminal:
            validated_terminal = (
                validate_semantic_phase3_no_positive_terminal_receipt(
                    terminal_phase3,
                    route_variant=identity.route_variant,
                    route_contract=route,
                    expected_route_contract_sha256=str(route_sha256),
                    accepted_round_count=len(closed_rounds),
                    terminal_active_prefix_checkpoint=finalization[
                        "terminal_active_prefix_checkpoint"
                    ],
                    finalization=finalization,
                )
            )
            closure.update(
                {
                    "terminal_controller_outcome": str(
                        terminal_outcome
                    ),
                    "terminal_accepted_controller_round": int(
                        validated_terminal["accepted_controller_round"]
                    ),
                    "terminal_attempted_controller_round": int(
                        validated_terminal["attempted_controller_round"]
                    ),
                    "terminal_phase3_selection_receipt_sha256": str(
                        validated_terminal["sha256"]
                    ),
                    "terminal_phase123_adaptive_shortlist_receipt_sha256": (
                        validated_terminal[
                            "phase123_adaptive_shortlist_receipt_sha256"
                        ]
                    ),
                    "terminal_phase123_qiskit_adaptive_population_link_sha256": (
                        validated_terminal[
                            "phase123_qiskit_adaptive_population_link_sha256"
                        ]
                    ),
                    "terminal_final_admission_record_id": None,
                }
            )
        closure["sha256"] = canonical_sha256(closure)
        return closure
    except RuntimeError as exc:
        raise RuntimeError(
            "Paper-I semantic final selector accounting is invalid."
        ) from exc
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            "Paper-I semantic final selector accounting is invalid."
        ) from exc


def _phase0_policy(route_variant: str) -> dict[str, Any]:
    common = {
        "population": "same_ordered_append_endpoint_generator_population_v1",
        "benefit": "absolute_append_endpoint_generator_gradient_v1",
        "fubini_study_metric": "off",
        "qiskit_compile": "off",
    }
    if route_variant == PAPER_I_RA_PHASE0_GRADIENT_FIXED24:
        return {
            **common,
            "graph_proxy_cost": "off",
            "score": "absolute_append_endpoint_generator_gradient_v1",
            "shortlist": "fixed_top_24_v1",
            "adaptive_shadow_receipt": False,
        }
    if route_variant == PAPER_I_RA_PHASE0_PROXY_FIXED24_SHADOW:
        return {
            **common,
            "graph_proxy_cost": "paper_i_structural_graph_proxy_transform_v1",
            "score": "absolute_append_gradient_over_graph_proxy_cost_v1",
            "shortlist": "fixed_top_24_v1",
            "adaptive_shadow_receipt": True,
        }
    if route_variant == PAPER_I_RA_PHASE0_PROXY_ADAPTIVE:
        return {
            **common,
            "benefit": "squared_append_endpoint_generator_gradient_v1",
            "graph_proxy_cost": "paper_i_structural_graph_proxy_transform_v1",
            "score": "squared_append_gradient_over_graph_proxy_cost_v1",
            "shortlist": ADAPTIVE_APPEND_ENDPOINT_SHORTLIST_POLICY,
            "adaptive_shadow_receipt": False,
        }
    if route_variant == PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2:
        return {
            **common,
            "graph_proxy_cost": "off",
            "score": "absolute_append_endpoint_generator_gradient_v1",
            "shortlist": "fixed_top_24_v1",
            "adaptive_shadow_receipt": False,
        }
    if route_variant in {
        PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2,
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1,
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_FORCED_K50_V1,
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
    }:
        return {
            **common,
            "graph_proxy_cost": "off",
            "score": "absolute_append_endpoint_generator_gradient_v1",
            "shortlist": ADAPTIVE_PHASE0_ACTIVE_SCORE_SHORTLIST_POLICY_V2,
            "adaptive_shadow_receipt": False,
        }
    if route_variant == PAPER_I_RA_PHASE0_PROXY_FIXED24_V2:
        return {
            **common,
            "graph_proxy_cost": "paper_i_structural_graph_proxy_transform_v1",
            "score": "absolute_append_gradient_over_graph_proxy_cost_v1",
            "shortlist": "fixed_top_24_v1",
            "adaptive_shadow_receipt": False,
        }
    if route_variant == PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2:
        return {
            **common,
            "graph_proxy_cost": "paper_i_structural_graph_proxy_transform_v1",
            "score": "absolute_append_gradient_over_graph_proxy_cost_v1",
            "shortlist": ADAPTIVE_PHASE0_ACTIVE_SCORE_SHORTLIST_POLICY_V2,
            "adaptive_shadow_receipt": False,
        }
    if route_variant in PAPER_I_RA_PHASE0_POSITION_ROUTE_VARIANTS:
        proxy = route_variant in {
            PAPER_I_RA_PHASE0_POSITION_PROXY_FIXED24_V1,
            PAPER_I_RA_PHASE0_POSITION_PROXY_ADAPTIVE_V1,
        }
        adaptive = route_variant in {
            PAPER_I_RA_PHASE0_POSITION_GRADIENT_ADAPTIVE_V1,
            PAPER_I_RA_PHASE0_POSITION_PROXY_ADAPTIVE_V1,
            PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1,
            PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
            PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_FORCED_K50_V1,
            PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
        }
        policy = {
            "population": (
                "current_commutation_reduced_candidate_position_records_v1"
            ),
            "benefit": "absolute_position_record_gradient_v1",
            "fubini_study_metric": "off",
            "qiskit_compile": "off",
            "graph_proxy_cost": (
                "paper_i_structural_graph_proxy_transform_v1"
                if proxy
                else "off"
            ),
            "score": (
                "absolute_position_gradient_over_graph_proxy_cost_v1"
                if proxy
                else "absolute_position_record_gradient_v1"
            ),
            "shortlist": (
                ADAPTIVE_PHASE0_ACTIVE_SCORE_SHORTLIST_POLICY_V2
                if adaptive
                else "fixed_top_24_v1"
            ),
            "adaptive_shadow_receipt": False,
        }
        if route_variant in {
            PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1,
            PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
            PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_FORCED_K50_V1,
            PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
        }:
            policy.update(
                {
                    "placement_activation": (
                        "append_record_when_closed_full_commutation_reduced_"
                        "records_when_open_v1"
                    ),
                    "generator_level_reexpansion_after_phase0": False,
                }
            )
        return policy
    raise ValueError("Unknown Paper-I semantic-closure route variant.")


def _require_semantic_closure_request(
    request: RAAdaptRequest,
    *,
    algorithm_id: str,
    active_gradient_policy: str,
    resource_weighting_scope: str,
) -> PaperIRASemanticClosureRouteIdentity:
    if not isinstance(request, RAAdaptRequest):
        raise TypeError("request must be an RAAdaptRequest.")
    adapter = request.adapter
    if not is_semantic_closure_adapter(adapter):
        raise ValueError(
            "Semantic-closure algorithm requires its typed global-singleton adapter."
        )
    assert isinstance(
        adapter,
        PaperIRASemanticClosureGlobalSingletonCandidateAdapter,
    )
    identity = semantic_closure_route_identity(adapter.route_variant)
    if identity.algorithm_id != str(algorithm_id):
        raise ValueError(
            "Semantic-closure adapter and algorithm identity must match."
        )
    if (
        str(active_gradient_policy) != ACTIVE_GRADIENT_STATIONARY
        or str(resource_weighting_scope) != RESOURCE_WEIGHTING_ALL_PHASE
    ):
        raise ValueError(
            "Semantic-closure routes require stationary source response and "
            "all-phase resource weighting."
        )
    if not (
        isinstance(request.method.admission, SingletonAdmission)
        and isinstance(request.method.pruning, PruningOff)
        and isinstance(request.method.beam, BeamOff)
        and isinstance(request.execution.resume, FreshStart)
        and request.execution.stop.exact_ed_target is None
        and isinstance(
            request.method.insertion,
            (
                AlwaysCommutationReducedInsertion,
                AppendCommutationReducedInsertion,
                AppendOnlyInsertion,
                PlateauCommutationInsertion,
            ),
        )
    ):
        raise ValueError(
            "Semantic-closure routes require singleton admission, pruning/beam "
            "off, a fresh unconditional horizon, and a typed insertion policy."
        )
    return identity


def build_semantic_closure_route_contract(
    request: RAAdaptRequest,
    *,
    algorithm_id: str,
    active_gradient_policy: str,
    resource_weighting_scope: str,
    parent_contract: Mapping[str, Any],
    parent_contract_sha256: str,
) -> tuple[str, str, dict[str, Any], str]:
    """Overlay one complete native semantic contract on a resolved parent."""

    identity = _require_semantic_closure_request(
        request,
        algorithm_id=algorithm_id,
        active_gradient_policy=active_gradient_policy,
        resource_weighting_scope=resource_weighting_scope,
    )
    route = copy.deepcopy(dict(parent_contract))
    insertion_kind = str(request.method.insertion.kind)
    profile = f"{identity.route_profile}__insertion-{insertion_kind}"
    phase0 = _phase0_policy(identity.route_variant)
    phase3_no_positive_policy = semantic_phase3_no_positive_policy(
        identity.route_variant
    )
    controller_horizon_policy = semantic_controller_horizon_policy(
        identity.route_variant
    )
    natural_terminal_route = (
        identity.route_variant
        in PAPER_I_RA_PHASE3_NATURAL_TERMINAL_ROUTE_VARIANTS
    )
    declares_no_positive_policy = natural_terminal_route or (
        identity.route_variant
        in PAPER_I_RA_PHASE3_FORCED_ADMISSION_ROUTE_VARIANTS
    )
    shortlist_minimums = semantic_phase_shortlist_minimums(
        identity.route_variant
    )
    native = {
        "schema": PAPER_I_RA_SEMANTIC_ROUTE_CONTRACT_SCHEMA,
        "route_variant": identity.route_variant,
        "route_id": identity.route_id,
        "algorithm_id": identity.algorithm_id,
        "semantic_implementation_version": identity.semantic_implementation_version,
        "candidate_population": "global_guarded_singleton_pool_v1",
        "candidate_representation": CANDIDATE_REPRESENTATION_SINGLE_PAULI,
        "phase0_policy": phase0,
        "phase0_adaptive_cap": 24,
        "phase0_estimator_components": ["N_grad"],
        "compile_scope": PAPER_I_RA_PHASE123_QISKIT_COMPILE_SCOPE,
        "qiskit_active_phases": ["phase_i", "phase_ii", "phase_iii"],
        "qiskit_full_trial_compile_semantics": (
            "full_base_and_trial_ansatz_at_recorded_insertion_position_v1"
        ),
        "signed_compile_deltas": ["delta_N2q", "delta_D2q", "delta_N1q"],
        "hardware_cost_normalization": "zero_centered_signed_arctan_v1",
        "negative_compile_delta_policy": "negative_delta_is_reward_v1",
        "phase_score_closure": {
            "phase_i": "benefit_times_hardware_cost_factor_over_burden_v1",
            "phase_ii": "benefit_times_hardware_cost_factor_over_burden_v1",
            "phase_iii": "benefit_times_hardware_cost_factor_over_burden_v1",
        },
        "population_normalization": (
            "one_complete_evaluated_population_once_before_ranking_per_phase_v1"
        ),
        "compile_cache_policy": PAPER_I_RA_PHASE123_QISKIT_CACHE_POLICY,
        "compile_failure_policy": "abort_run_v1",
        "compile_work_in_s_alg": False,
        "s_alg": "N_H_outer+N_H_refit+N_grad+N_metric",
        "insertion_policy": insertion_kind,
        "optimizer": "powell",
        "optimizer_maxiter": 200,
        "seeds": {"adapt": 7, "transpiler": 7},
        "horizon": int(
            request.execution.stop.maximum_controller_rounds
        ),
    }
    if declares_no_positive_policy:
        native.update(
            {
                "phase3_no_positive_policy": phase3_no_positive_policy,
                "controller_horizon_policy": controller_horizon_policy,
            }
        )
    if shortlist_minimums is not None:
        native["phase_shortlist_minimums"] = dict(shortlist_minimums)
    if identity.route_variant in PAPER_I_RA_PHASE0_EXECUTABLE_ROUTE_VARIANTS:
        native.update(
            {
                "optimizer_options": {
                    "xtol": 1.0e-4,
                    "ftol": 1.0e-8,
                    "maxfev": None,
                },
                "phase_shortlist_maxima": {"phase_i": 24, "phase_ii": 12},
                "phase_frontier_ratios": {"phase_ii": 0.9, "phase_iii": 0.9},
            }
        )
    if identity.route_variant in PAPER_I_RA_ALL_PHASE_ADAPTIVE_ROUTE_VARIANTS:
        native.update(
            {
                "phase123_shortlist_policy": (
                    ADAPTIVE_PHASE123_SHORTLIST_POLICY_V1
                ),
                "phase_shortlist_maxima": {
                    "phase_i": 24,
                    "phase_ii": 12,
                    "phase_iii": 12,
                },
                "phase_frontier_ratio_role": "eligibility_only",
                "phase_frontier_ratios": {
                    "phase_i": 0.9,
                    "phase_ii": 0.9,
                    "phase_iii": 0.9,
                },
            }
        )

    route["route_family"] = "ra_adapt"
    route["route_id"] = identity.route_id
    route["route_profile"] = profile
    route["algorithm_id"] = identity.algorithm_id
    route["semantic_implementation_version"] = (
        identity.semantic_implementation_version
    )
    route["native_semantic_contract"] = native

    execution = dict(route.get("execution_settings", {}))
    execution.update(
        {
            "ra_semantic_route_variant": identity.route_variant,
            "ra_semantic_implementation_version": (
                identity.semantic_implementation_version
            ),
            "ra_phase0_gradient_shortlist_policy": (
                request.adapter.phase0_shortlist_policy_id
            ),
            "ra_phase0_gradient_shortlist_size": 24,
            "ra_phase0_adaptive_shadow_receipt": bool(
                phase0["adaptive_shadow_receipt"]
            ),
            "phase3_backend_cost_scope": (
                PAPER_I_RA_PHASE123_QISKIT_COMPILE_SCOPE
            ),
            "phase3_hardware_cost_normalization_mode": (
                "zero_centered_signed_arctan_v1"
            ),
        }
    )
    if declares_no_positive_policy:
        execution.update(
            {
                "ra_phase3_no_positive_policy": phase3_no_positive_policy,
                "ra_controller_horizon_policy": controller_horizon_policy,
            }
        )
    if shortlist_minimums is not None:
        execution["ra_phase_shortlist_minimums"] = dict(shortlist_minimums)
    if identity.route_variant in PAPER_I_RA_ALL_PHASE_ADAPTIVE_ROUTE_VARIANTS:
        execution["ra_phase123_shortlist_policy"] = (
            ADAPTIVE_PHASE123_SHORTLIST_POLICY_V1
        )
    route["execution_settings"] = execution

    invariants = dict(route.get("semantic_invariants", {}))
    invariants.update(
        {
            "semantic_implementation_version": (
                identity.semantic_implementation_version
            ),
            "phase0_active": True,
            "phase0_population": phase0["population"],
            "phase0_score": phase0["score"],
            "phase0_shortlist_policy": phase0["shortlist"],
            "phase0_fubini_metric_active": False,
            "phase0_compile_cost_active": False,
            "phase0_structural_proxy_cost_active": (
                phase0["graph_proxy_cost"] != "off"
            ),
            "phase0_resource_cost_active": (
                phase0["graph_proxy_cost"] != "off"
            ),
            "phase0_adaptive_shadow_receipt_active": bool(
                phase0["adaptive_shadow_receipt"]
            ),
            "phase0_estimator_components": ["N_grad"],
            "selector_compile_cost_policy": (
                PAPER_I_RA_PHASE123_QISKIT_COST_POLICY
            ),
            "selector_compile_cost_scope": (
                PAPER_I_RA_PHASE123_QISKIT_COMPILE_SCOPE
            ),
            "selector_compile_cost_phase_reuse": (
                PAPER_I_RA_PHASE123_QISKIT_CACHE_POLICY
            ),
            "phase_i_compile_cost_source": "backend_transpile_v1",
            "phase_ii_compile_cost_source": "backend_transpile_v1",
            "phase_iii_compile_cost_source": "backend_transpile_v1",
            "phase_i_phase_ii_phase_iii_qiskit_negative_delta_reward_enabled": True,
            "phase_i_phase_ii_phase_iii_qiskit_backend_fallback_allowed": False,
            "phase_i_phase_ii_phase_iii_qiskit_full_base_trial_ansatz_transpile": True,
            "phase_i_phase_ii_phase_iii_qiskit_population_normalization_policy": (
                "zero_centered_signed_arctan_v1"
            ),
            "qiskit_compile_work_excluded_from_s_alg": True,
            "estimator_accounting_convention": (
                "s_alg_equals_n_h_outer_plus_n_h_refit_plus_n_grad_plus_n_metric_v1"
            ),
        }
    )
    if declares_no_positive_policy:
        invariants.update(
            {
                "phase3_no_positive_policy": phase3_no_positive_policy,
                "controller_horizon_policy": controller_horizon_policy,
            }
        )
    if shortlist_minimums is not None:
        invariants["phase_shortlist_minimums"] = dict(shortlist_minimums)
    route["semantic_invariants"] = invariants
    route["lineage_authority"] = {
        "parent_route_profile": str(parent_contract.get("route_profile", "")),
        "parent_contract_sha256": str(parent_contract_sha256),
        "supersession_reason": (
            (
                "paper_i_ra_position_gradient_phase0_all_phase_adaptive_"
                "semantic_closure_20260817"
                if identity.route_variant
                in {
                    PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1,
                    PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
                    PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_FORCED_K50_V1,
                    PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
                }
                else (
                    "paper_i_ra_gradient_phase0_all_phase_adaptive_semantic_"
                    "closure_20260816"
                )
            )
            if identity.route_variant
            in PAPER_I_RA_ALL_PHASE_ADAPTIVE_ROUTE_VARIANTS
            else (
                "paper_i_ra_phase0_proxy_ablation_phase123_qiskit_semantic_"
                "closure_20260816"
            )
        ),
        "only_intended_scientific_changes": [
            "phase3_zero_centered_signed_factor_consumption_repair",
            "phase_i_full_trial_qiskit_compile_cost_activation",
            "declared_phase0_policy",
            *(
                [
                    (
                        "phase0_commutation_reduced_position_record_absolute_"
                        "gradient_adaptive_cardinality_direct_pass_through"
                        if identity.route_variant
                        in {
                            PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1,
                            PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
                        }
                        else (
                            "phase0_standard_append_endpoint_absolute_gradient_"
                            "adaptive_cardinality"
                        )
                    ),
                    "phase_i_phase_ii_phase_iii_adaptive_cardinality",
                ]
                if identity.route_variant
                in PAPER_I_RA_ALL_PHASE_ADAPTIVE_ROUTE_VARIANTS
                else []
            ),
        ],
        "scientific_result_anchor_claimed": False,
    }
    normalized = copy.deepcopy(route)
    return profile, profile, normalized, canonical_sha256(normalized)


def validate_semantic_closure_native_route_contract(
    route_contract: Mapping[str, Any],
    *,
    candidate_adapter: Any | None = None,
) -> dict[str, Any]:
    """Authenticate one semantic route at the numerical-runtime boundary."""

    if not isinstance(route_contract, Mapping):
        raise ValueError("Semantic runtime requires a route contract.")
    route = copy.deepcopy(dict(route_contract))
    native = route.get("native_semantic_contract")
    execution = route.get("execution_settings")
    invariants = route.get("semantic_invariants")
    lineage = route.get("lineage_authority")
    if not all(
        isinstance(value, Mapping)
        for value in (native, execution, invariants, lineage)
    ):
        raise ValueError("Semantic native route contract is incomplete.")
    assert isinstance(native, Mapping)
    assert isinstance(execution, Mapping)
    assert isinstance(invariants, Mapping)
    assert isinstance(lineage, Mapping)

    variant = str(native.get("route_variant", ""))
    identity = semantic_closure_route_identity(variant)
    insertion_kind = str(native.get("insertion_policy", ""))
    expected_profile = (
        f"{identity.route_profile}__insertion-{insertion_kind}"
    )
    horizon = native.get("horizon")
    expected_phase3_no_positive_policy = semantic_phase3_no_positive_policy(
        variant
    )
    expected_controller_horizon_policy = semantic_controller_horizon_policy(
        variant
    )
    natural_terminal_route = (
        variant in PAPER_I_RA_PHASE3_NATURAL_TERMINAL_ROUTE_VARIANTS
    )
    declares_no_positive_policy = natural_terminal_route or (
        variant in PAPER_I_RA_PHASE3_FORCED_ADMISSION_ROUTE_VARIANTS
    )
    expected_shortlist_minimums = semantic_phase_shortlist_minimums(variant)
    shortlist_minimum_fields_valid = bool(
        (
            expected_shortlist_minimums is not None
            and native.get("phase_shortlist_minimums")
            == expected_shortlist_minimums
            and execution.get("ra_phase_shortlist_minimums")
            == expected_shortlist_minimums
            and invariants.get("phase_shortlist_minimums")
            == expected_shortlist_minimums
        )
        or (
            expected_shortlist_minimums is None
            and "phase_shortlist_minimums" not in native
            and "ra_phase_shortlist_minimums" not in execution
            and "phase_shortlist_minimums" not in invariants
        )
    )
    natural_terminal_policy_fields_valid = bool(
        (
            declares_no_positive_policy
            and native.get("phase3_no_positive_policy")
            == expected_phase3_no_positive_policy
            and native.get("controller_horizon_policy")
            == expected_controller_horizon_policy
            and execution.get("ra_phase3_no_positive_policy")
            == expected_phase3_no_positive_policy
            and execution.get("ra_controller_horizon_policy")
            == expected_controller_horizon_policy
            and invariants.get("phase3_no_positive_policy")
            == expected_phase3_no_positive_policy
            and invariants.get("controller_horizon_policy")
            == expected_controller_horizon_policy
        )
        or (
            not declares_no_positive_policy
            and "phase3_no_positive_policy" not in native
            and "controller_horizon_policy" not in native
            and "ra_phase3_no_positive_policy" not in execution
            and "ra_controller_horizon_policy" not in execution
            and "phase3_no_positive_policy" not in invariants
            and "controller_horizon_policy" not in invariants
        )
    )
    adaptive_phase123 = variant in PAPER_I_RA_ALL_PHASE_ADAPTIVE_ROUTE_VARIANTS
    expected_phase_maxima = (
        {"phase_i": 24, "phase_ii": 12, "phase_iii": 12}
        if adaptive_phase123
        else {"phase_i": 24, "phase_ii": 12}
    )
    expected_frontier_ratios = (
        {"phase_i": 0.9, "phase_ii": 0.9, "phase_iii": 0.9}
        if adaptive_phase123
        else {"phase_ii": 0.9, "phase_iii": 0.9}
    )
    if (
        route.get("schema") != "paper_i_ra_adapt_route_contract_v1"
        or route.get("route_family") != "ra_adapt"
        or route.get("route_id") != identity.route_id
        or route.get("route_profile") != expected_profile
        or route.get("algorithm_id") != identity.algorithm_id
        or route.get("semantic_implementation_version")
        != identity.semantic_implementation_version
        or native.get("schema")
        != PAPER_I_RA_SEMANTIC_ROUTE_CONTRACT_SCHEMA
        or native.get("route_id") != identity.route_id
        or native.get("algorithm_id") != identity.algorithm_id
        or native.get("semantic_implementation_version")
        != identity.semantic_implementation_version
        or native.get("candidate_population")
        != "global_guarded_singleton_pool_v1"
        or native.get("candidate_representation")
        != CANDIDATE_REPRESENTATION_SINGLE_PAULI
        or native.get("phase0_policy") != _phase0_policy(variant)
        or native.get("phase0_adaptive_cap") != 24
        or native.get("phase0_estimator_components") != ["N_grad"]
        or not natural_terminal_policy_fields_valid
        or not shortlist_minimum_fields_valid
        or native.get("compile_scope")
        != PAPER_I_RA_PHASE123_QISKIT_COMPILE_SCOPE
        or native.get("qiskit_active_phases")
        != ["phase_i", "phase_ii", "phase_iii"]
        or native.get("qiskit_full_trial_compile_semantics")
        != "full_base_and_trial_ansatz_at_recorded_insertion_position_v1"
        or native.get("hardware_cost_normalization")
        != "zero_centered_signed_arctan_v1"
        or native.get("optimizer") != "powell"
        or native.get("optimizer_maxiter") != 200
        or native.get("optimizer_options")
        != {"xtol": 1.0e-4, "ftol": 1.0e-8, "maxfev": None}
        or native.get("phase_shortlist_maxima")
        != expected_phase_maxima
        or native.get("phase_frontier_ratios")
        != expected_frontier_ratios
        or native.get("phase123_shortlist_policy")
        != (
            ADAPTIVE_PHASE123_SHORTLIST_POLICY_V1
            if adaptive_phase123
            else None
        )
        or native.get("phase_frontier_ratio_role")
        != ("eligibility_only" if adaptive_phase123 else None)
        or execution.get("ra_phase123_shortlist_policy")
        != (
            ADAPTIVE_PHASE123_SHORTLIST_POLICY_V1
            if adaptive_phase123
            else None
        )
        or native.get("negative_compile_delta_policy")
        != "negative_delta_is_reward_v1"
        or native.get("compile_cache_policy")
        != PAPER_I_RA_PHASE123_QISKIT_CACHE_POLICY
        or native.get("compile_failure_policy") != "abort_run_v1"
        or native.get("compile_work_in_s_alg") is not False
        or native.get("s_alg")
        != "N_H_outer+N_H_refit+N_grad+N_metric"
        or native.get("optimizer") != "powell"
        or native.get("optimizer_maxiter") != 200
        or native.get("seeds") != {"adapt": 7, "transpiler": 7}
        or isinstance(horizon, bool)
        or not isinstance(horizon, int)
        or not 1 <= horizon <= 50
        or lineage.get("scientific_result_anchor_claimed") is not False
    ):
        raise ValueError("Semantic native route identity drifted.")

    expected_adapter = (
        PaperIRASemanticClosureGlobalSingletonCandidateAdapter(
            route_variant=variant
        )
        if candidate_adapter is None
        else candidate_adapter
    )
    if (
        not is_semantic_closure_adapter(expected_adapter)
        or expected_adapter.route_variant != variant
    ):
        raise ValueError("Semantic route and runtime adapter disagree.")
    try:
        validate_semantic_phase0_runtime_binding(
            expected_adapter,
            route,
        )
    except RuntimeError as exc:
        raise ValueError("Semantic native route binding drifted.") from exc

    observed_sha256 = route.pop("sha256", None)
    if (
        observed_sha256 is not None
        and str(observed_sha256) != canonical_sha256(route)
    ):
        raise ValueError("Semantic native route digest drifted.")
    return copy.deepcopy(dict(route_contract))


def validate_semantic_phase3_natural_terminal_route_contract(
    route_contract: Mapping[str, Any],
    *,
    expected_route_contract_sha256: str,
) -> dict[str, Any]:
    """Authenticate the sole route allowed to end on Phase-III exhaustion."""

    if not isinstance(route_contract, Mapping):
        raise ValueError(
            "Phase-III terminal requires a V2 natural-terminal route contract."
        )
    route = copy.deepcopy(dict(route_contract))
    embedded_sha256 = route.pop("sha256", None)
    expected_sha256 = str(expected_route_contract_sha256)
    if (
        len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
        or canonical_sha256(route) != expected_sha256
        or (
            embedded_sha256 is not None
            and str(embedded_sha256) != expected_sha256
        )
    ):
        raise ValueError("Phase-III natural-terminal route contract digest drifted.")

    validate_semantic_closure_native_route_contract(route)
    native = route.get("native_semantic_contract")
    execution = route.get("execution_settings")
    invariants = route.get("semantic_invariants")
    if (
        not isinstance(native, Mapping)
        or not isinstance(execution, Mapping)
        or not isinstance(invariants, Mapping)
        or native.get("route_variant")
        not in PAPER_I_RA_PHASE3_NATURAL_TERMINAL_ROUTE_VARIANTS
        or native.get("phase3_no_positive_policy")
        != ADAPTIVE_PHASE3_NO_POSITIVE_POLICY_TYPED_TERMINAL_V1
        or native.get("controller_horizon_policy")
        != ADAPTIVE_HORIZON_POLICY_MAXIMUM_V1
        or execution.get("ra_phase3_no_positive_policy")
        != ADAPTIVE_PHASE3_NO_POSITIVE_POLICY_TYPED_TERMINAL_V1
        or execution.get("ra_controller_horizon_policy")
        != ADAPTIVE_HORIZON_POLICY_MAXIMUM_V1
        or invariants.get("phase3_no_positive_policy")
        != ADAPTIVE_PHASE3_NO_POSITIVE_POLICY_TYPED_TERMINAL_V1
        or invariants.get("controller_horizon_policy")
        != ADAPTIVE_HORIZON_POLICY_MAXIMUM_V1
    ):
        raise ValueError(
            "Phase-III terminal requires the authenticated V2 natural-terminal route."
        )
    return route


_PHASE0_APPROVED_EXECUTION_KEYS = frozenset(
    {
        "ra_semantic_route_variant",
        "ra_phase0_gradient_shortlist_policy",
        "ra_phase0_gradient_shortlist_size",
        "ra_phase0_adaptive_shadow_receipt",
    }
)


def project_approved_phase0_ablation(
    route_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove only identity and declared Phase-0 fields for A/B comparison."""

    projected = copy.deepcopy(dict(route_contract))
    for key in (
        "sha256",
        "route_id",
        "route_profile",
        "algorithm_id",
    ):
        projected.pop(key, None)
    execution = dict(projected.get("execution_settings", {}))
    for key in _PHASE0_APPROVED_EXECUTION_KEYS:
        execution.pop(key, None)
    projected["execution_settings"] = execution
    invariants = dict(projected.get("semantic_invariants", {}))
    for key in tuple(invariants):
        if key.startswith("phase0_"):
            invariants.pop(key, None)
    projected["semantic_invariants"] = invariants
    native = dict(projected.get("native_semantic_contract", {}))
    for key in ("route_variant", "route_id", "algorithm_id", "phase0_policy"):
        native.pop(key, None)
    projected["native_semantic_contract"] = native
    return projected


def _semantic_closure_repository_root() -> Path:
    root = Path(__file__).resolve().parents[3]
    if not (root / "pipelines").is_dir():
        raise RuntimeError(
            "Paper-I semantic-closure source root is not a repository checkout."
        )
    return root


def semantic_closure_source_implementation_inventory(
    route_variant: str | None = None,
) -> dict[str, Any]:
    """Hash every implementation surface required by the native route.

    The inventory is recomputed from the checkout that imported this module.
    A source-locked future batch therefore binds the actual corrected
    consumer, transaction, authority loader, and route implementation rather
    than trusting an algorithm label alone.
    """

    root = _semantic_closure_repository_root()
    if route_variant is not None:
        # Validate the requested route identity, but do not make a source-tree
        # digest route-dependent.  A factorial campaign must bind one exact
        # byte inventory while each bundle manifest separately binds its own
        # semantic version and route identity.
        semantic_closure_route_identity(str(route_variant))
    source_specs: list[tuple[str, str]] = list(
        _SEMANTIC_IMPLEMENTATION_SOURCE_PATHS
    )
    covered_paths = {
        relative_path for _, relative_path in source_specs
    }
    for relative_root in _SEMANTIC_IMPLEMENTATION_SOURCE_ROOTS:
        source_root = (root / relative_root).resolve()
        try:
            source_root.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                "Semantic implementation source root escaped the repository."
            ) from exc
        if not source_root.is_dir() or source_root.is_symlink():
            raise RuntimeError(
                f"Semantic implementation source root is missing: {relative_root}"
            )
        for path in sorted(source_root.rglob("*.py")):
            relative_path = path.relative_to(root).as_posix()
            if relative_path not in covered_paths:
                source_specs.append(
                    (f"runtime_tree:{relative_path}", relative_path)
                )
                covered_paths.add(relative_path)
    for relative_path in _SEMANTIC_IMPLEMENTATION_EXTRA_SOURCE_PATHS:
        if relative_path not in covered_paths:
            source_specs.append((f"runtime_extra:{relative_path}", relative_path))
            covered_paths.add(relative_path)

    sources: list[dict[str, Any]] = []
    for role, relative_path in source_specs:
        source_path = root / relative_path
        path = source_path.resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                "Semantic implementation source escaped the repository root."
            ) from exc
        if not path.is_file() or source_path.is_symlink():
            raise RuntimeError(
                f"Semantic implementation source is missing: {relative_path}"
            )
        payload = path.read_bytes()
        sources.append(
            {
                "role": role,
                "path": relative_path,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    inventory: dict[str, Any] = {
        "schema": PAPER_I_RA_SEMANTIC_SOURCE_INVENTORY_SCHEMA,
        "semantic_implementation_scope": (
            "paper_i_ra_phase0_placement_score_cardinality_matrix_all_"
            "native_routes_v1"
        ),
        "coverage": "conservative_production_python_tree_v1",
        "source_roots": list(_SEMANTIC_IMPLEMENTATION_SOURCE_ROOTS),
        "extra_source_paths": list(
            _SEMANTIC_IMPLEMENTATION_EXTRA_SOURCE_PATHS
        ),
        "source_count": len(sources),
        "sources": sources,
    }
    inventory["sha256"] = canonical_sha256(inventory)
    return inventory


def semantic_closure_native_bundle_manifest(
    route_variant: str,
) -> dict[str, Any]:
    """Return the source-bound, non-serializable-authority bundle manifest."""

    identity = semantic_closure_route_identity(route_variant)
    inventory = semantic_closure_source_implementation_inventory(
        identity.route_variant
    )
    manifest: dict[str, Any] = {
        "schema": "paper_i_ra_semantic_closure_native_authority_v2",
        "bundle_id": semantic_closure_native_bundle_id(
            identity.route_variant
        ),
        "route_variant": identity.route_variant,
        "algorithm_id": identity.algorithm_id,
        "semantic_implementation_version": identity.semantic_implementation_version,
        "source_implementation_inventory": inventory,
        "source_implementation_inventory_sha256": inventory["sha256"],
        "serialized_protocol_execution_authorized": False,
        "execution_requires_private_materialization_authority": True,
        "materialization_authority_serializable": False,
    }
    manifest["sha256"] = canonical_sha256(manifest)
    return manifest


def semantic_closure_native_bundle_digest(route_variant: str) -> str:
    return str(semantic_closure_native_bundle_manifest(route_variant)["sha256"])


def _semantic_hamiltonian_terms_from_value(
    hamiltonian: Any,
) -> list[dict[str, Any]]:
    from src.quantum.pauli_polynomial_class import PauliPolynomial
    from src.quantum.qubitization_module import PauliTerm

    if type(hamiltonian) is not PauliPolynomial:
        raise ValueError("Canonical Hamiltonian term inventory is unavailable.")
    terms = getattr(hamiltonian, "_pol", None)
    if not isinstance(terms, list):
        raise ValueError("Canonical Hamiltonian term inventory is unavailable.")
    rows: list[dict[str, Any]] = []
    for term in terms:
        if type(term) is not PauliTerm:
            raise ValueError("Canonical Hamiltonian term is malformed.")
        coefficient = complex(getattr(term, "p_coeff", 0.0))
        nq = getattr(term, "_PauliTerm__nq", None)
        letters = getattr(term, "pw", None)
        if (
            isinstance(nq, bool)
            or not isinstance(nq, int)
            or not isinstance(letters, list)
            or len(letters) != nq
            or not math.isfinite(float(coefficient.real))
            or not math.isfinite(float(coefficient.imag))
        ):
            raise ValueError("Canonical Hamiltonian term is malformed.")
        symbols = tuple(str(getattr(letter, "symbol", "")) for letter in letters)
        if any(symbol not in {"e", "x", "y", "z"} for symbol in symbols):
            raise ValueError("Canonical Hamiltonian term is malformed.")
        rows.append(
            {
                "pauli": "".join(symbols),
                "nq": nq,
                "coefficient_real_hex": float(coefficient.real).hex(),
                "coefficient_imag_hex": float(coefficient.imag).hex(),
            }
        )
    if not rows:
        raise ValueError("Canonical Hamiltonian term inventory is empty.")
    return sorted(
        rows,
        key=lambda row: (
            str(row["pauli"]),
            str(row["coefficient_real_hex"]),
            str(row["coefficient_imag_hex"]),
        ),
    )


def _semantic_hamiltonian_terms(
    problem: ResolvedProblemContext,
) -> list[dict[str, Any]]:
    return _semantic_hamiltonian_terms_from_value(problem.hamiltonian)


def _semantic_problem_scientific_content(
    problem: ResolvedProblemContext,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "paper_i_ra_semantic_problem_scientific_content_v2",
        "problem_request_sha256": canonical_sha256(asdict(problem.request)),
        "layout": asdict(problem.layout),
        "sector": asdict(problem.sector),
        "hamiltonian_terms": _semantic_hamiltonian_terms(problem),
        "reference_state": {
            "kind": str(problem.reference_state.kind),
            "source_label": str(problem.reference_state.source_label),
            "state_kind": str(problem.reference_state.state_kind),
        },
        "exact_target": {
            "kind": str(problem.exact_target.kind),
            "comparison_space_label": str(
                problem.exact_target.comparison_space_label
            ),
            "exact_state_policy": str(
                problem.exact_target.exact_state_policy
            ),
            "fallback_policy": str(problem.exact_target.fallback_policy),
        },
    }
    payload["hamiltonian_terms_sha256"] = canonical_sha256(
        payload["hamiltonian_terms"]
    )
    payload["sha256"] = canonical_sha256(payload)
    return payload


def _require_canonical_semantic_problem_content(
    problem: ResolvedProblemContext,
) -> dict[str, Any]:
    canonical = resolve_problem_context(problem.request)
    try:
        observed = _semantic_problem_scientific_content(problem)
        expected = _semantic_problem_scientific_content(canonical)
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError(
            "Semantic execution requires canonical Paper-I scientific "
            "Hamiltonian, reference-state, and target content."
        ) from exc
    if observed != expected:
        raise ValueError(
            "Semantic execution requires canonical Paper-I scientific "
            "Hamiltonian, reference-state, and target content."
        )
    return expected


def canonical_semantic_execution_problem(
    problem: ResolvedProblemContext,
) -> ResolvedProblemContext:
    """Return a fresh canonical context after non-executing identity checks."""

    _require_canonical_semantic_problem_content(problem)
    return resolve_problem_context(problem.request)


def _canonical_paper_i_hh_regime_id(
    problem: ResolvedProblemContext,
) -> str:
    if not isinstance(problem, ResolvedProblemContext):
        raise TypeError("problem must be a ResolvedProblemContext.")
    request = problem.request
    invariant = (
        str(problem.family_key) == "hh"
        and str(request.problem_key) == "hh"
        and int(request.num_sites) == 2
        and math.isclose(float(request.t), 1.0, rel_tol=0.0, abs_tol=1.0e-15)
        and math.isclose(float(request.dv), 0.0, rel_tol=0.0, abs_tol=1.0e-15)
        and math.isclose(
            float(request.omega0), 1.0, rel_tol=0.0, abs_tol=1.0e-15
        )
        and str(request.boson_encoding) == "binary"
        and str(request.ordering) == "blocked"
        and str(request.boundary) == "open"
        and bool(request.include_zero_point) is True
        and math.isclose(float(request.v_nn), 0.0, rel_tol=0.0, abs_tol=1.0e-15)
        and math.isclose(
            float(request.t_prime), 0.0, rel_tol=0.0, abs_tol=1.0e-15
        )
        and request.n_fermions is None
    )
    hubbard_regimes = (
        ("weak", 0.25),
        ("intermediate", 1.25),
        ("strong", 8.0),
    )
    holstein_regimes = (
        ("weak", math.sqrt(0.125), 3),
        ("strong", math.sqrt(0.625), 7),
    )
    hubbard = next(
        (
            label
            for label, value in hubbard_regimes
            if math.isclose(
                float(request.u), value, rel_tol=0.0, abs_tol=1.0e-12
            )
        ),
        None,
    )
    holstein = next(
        (
            label
            for label, value, cutoff in holstein_regimes
            if int(request.n_ph_max) == cutoff
            and math.isclose(
                float(request.g_ep), value, rel_tol=0.0, abs_tol=1.0e-12
            )
        ),
        None,
    )
    if not invariant or hubbard is None or holstein is None:
        raise ValueError(
            "Semantic authority is restricted to the canonical Paper-I L=2 "
            "Hubbard--Holstein six-regime matrix."
        )
    return (
        f"{hubbard}_{holstein}_u8"
        if hubbard == "strong"
        else f"{hubbard}_{holstein}"
    )


def _require_canonical_paper_i_semantic_application(
    problem: ResolvedProblemContext,
    request: RAAdaptRequest,
) -> tuple[PaperIRASemanticClosureRouteIdentity, str]:
    adapter = request.adapter if isinstance(request, RAAdaptRequest) else None
    if not isinstance(
        adapter,
        PaperIRASemanticClosureGlobalSingletonCandidateAdapter,
    ):
        raise ValueError(
            "Canonical Paper-I semantic materialization requires its typed "
            "adapter."
        )
    identity = semantic_closure_route_identity(adapter.route_variant)
    if identity.route_variant not in PAPER_I_RA_PHASE0_EXECUTABLE_ROUTE_VARIANTS:
        raise ValueError(
            "The v1 Phase-0 semantic route is retired; use an explicit v2 "
            "score/cardinality route."
        )
    _require_semantic_closure_request(
        request,
        algorithm_id=identity.algorithm_id,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
    )
    insertion = request.method.insertion
    horizon = request.execution.stop.maximum_controller_rounds
    if (
        not isinstance(
            insertion,
            (
                AlwaysCommutationReducedInsertion,
                PlateauCommutationInsertion,
                AppendOnlyInsertion,
            ),
        )
        or isinstance(horizon, bool)
        or not 1 <= int(horizon) <= 50
    ):
        raise ValueError(
            "Canonical Paper-I semantic materialization requires an "
            "always-open, plateau, or append-only fresh request with horizon "
            "1..50."
        )
    return identity, _canonical_paper_i_hh_regime_id(problem)


def _require_exact_strong_weak_always_k5_application(
    problem: ResolvedProblemContext,
    request: RAAdaptRequest,
) -> PaperIRASemanticClosureRouteIdentity:
    adapter = request.adapter if isinstance(request, RAAdaptRequest) else None
    if not isinstance(
        adapter,
        PaperIRASemanticClosureGlobalSingletonCandidateAdapter,
    ):
        raise ValueError(
            "The strong--weak always-open k=5 application requires the "
            "semantic adapter."
        )
    expected = build_paper_i_ra_strong_weak_always_k5_request(
        adapter.route_variant
    )
    identity, regime_id = _require_canonical_paper_i_semantic_application(
        problem,
        request,
    )
    problem_request = problem.request
    if (
        request.to_dict() != expected.to_dict()
        or regime_id != "strong_weak_u8"
    ):
        raise ValueError(
            "The strong--weak always-open k=5 request or problem drifted."
        )
    return identity


def _require_exact_strong_weak_plateau_k5_application(
    problem: ResolvedProblemContext,
    request: RAAdaptRequest,
) -> PaperIRASemanticClosureRouteIdentity:
    adapter = request.adapter if isinstance(request, RAAdaptRequest) else None
    if not isinstance(
        adapter,
        PaperIRASemanticClosureGlobalSingletonCandidateAdapter,
    ):
        raise ValueError(
            "The strong--weak plateau k=5 application requires the "
            "semantic adapter."
        )
    expected = build_paper_i_ra_strong_weak_plateau_k5_request(
        adapter.route_variant
    )
    identity, regime_id = _require_canonical_paper_i_semantic_application(
        problem,
        request,
    )
    if request.to_dict() != expected.to_dict() or regime_id != "strong_weak_u8":
        raise ValueError(
            "The strong--weak plateau k=5 request or problem drifted."
        )
    return identity


def semantic_closure_materialization_contract(
    problem: ResolvedProblemContext,
    request: RAAdaptRequest,
) -> dict[str, Any]:
    """Build the exact source-lock contract accepted by the native loader."""

    identity, regime_id = _require_canonical_paper_i_semantic_application(
        problem,
        request,
    )
    scientific_content = _require_canonical_semantic_problem_content(problem)
    bundle = semantic_closure_native_bundle_manifest(identity.route_variant)
    inventory = bundle["source_implementation_inventory"]
    insertion_kind = str(request.method.insertion.kind)
    horizon = int(request.execution.stop.maximum_controller_rounds)
    cell_id = (
        f"{identity.route_id}__{regime_id}__"
        f"nph{int(problem.request.n_ph_max)}__{insertion_kind}__k{horizon}"
    )
    source_lock_id = f"{cell_id}__source_v1"
    cell_lock: dict[str, Any] = {
        "schema": "paper_i_ra_semantic_closure_cell_source_lock_v1",
        "cell_id": cell_id,
        "source_lock_id": source_lock_id,
        "route_variant": identity.route_variant,
        "route_id": identity.route_id,
        "algorithm_id": identity.algorithm_id,
        "semantic_implementation_version": identity.semantic_implementation_version,
        "problem_request_sha256": canonical_sha256(asdict(problem.request)),
        "ra_request_sha256": canonical_sha256(request.to_dict()),
        "problem_scientific_content_sha256": scientific_content["sha256"],
        "source_implementation_inventory_sha256": inventory["sha256"],
        "bundle_manifest_sha256": bundle["sha256"],
    }
    cell_lock["sha256"] = canonical_sha256(cell_lock)
    source_locks_manifest: dict[str, Any] = {
        "schema": "paper_i_ra_semantic_closure_source_locks_v1",
        "bundle_manifest_sha256": bundle["sha256"],
        "source_implementation_inventory_sha256": inventory["sha256"],
        "cell_source_lock": cell_lock,
    }
    source_locks_manifest["sha256"] = canonical_sha256(source_locks_manifest)
    refs = {
        "source_locks_manifest_sha256": source_locks_manifest["sha256"],
        "implementation_source_inventory_sha256": inventory["sha256"],
        "cell_source_lock_id": source_lock_id,
        "cell_source_lock_sha256": cell_lock["sha256"],
        "semantic_bundle_manifest_sha256": bundle["sha256"],
        "problem_scientific_content_sha256": scientific_content["sha256"],
    }
    contract: dict[str, Any] = {
        "schema": PAPER_I_RA_SEMANTIC_MATERIALIZATION_CONTRACT_SCHEMA,
        "bundle_id": bundle["bundle_id"],
        "bundle_manifest_sha256": bundle["sha256"],
        "cell_id": cell_id,
        "source_lock_id": source_lock_id,
        "source_locks_sha256": source_locks_manifest["sha256"],
        "source_lock_refs": refs,
        "source_implementation_inventory_sha256": inventory["sha256"],
        "problem_scientific_content_sha256": scientific_content["sha256"],
        "route_variant": identity.route_variant,
        "route_id": identity.route_id,
        "algorithm_id": identity.algorithm_id,
        "semantic_implementation_version": identity.semantic_implementation_version,
        "serialized_protocol_execution_authorized": False,
        "execution_requires_private_materialization_authority": True,
    }
    contract["sha256"] = canonical_sha256(contract)
    return contract


def validate_semantic_closure_materialization_authority(
    problem: ResolvedProblemContext,
    request: RAAdaptRequest,
    *,
    receipt: Any,
    source_lock_refs: Mapping[str, str],
) -> dict[str, Any]:
    """Reject any semantic capability not minted from the exact native lock."""

    expected = semantic_closure_materialization_contract(problem, request)
    refs = {
        str(key): str(value)
        for key, value in sorted(source_lock_refs.items())
    }
    expected_receipt_fields = {
        "bundle_id": expected["bundle_id"],
        "bundle_manifest_sha256": expected["bundle_manifest_sha256"],
        "source_locks_sha256": expected["source_locks_sha256"],
        "cell_id": expected["cell_id"],
        "source_lock_id": expected["source_lock_id"],
        "protocol_schema": RA_ADAPT_PROTOCOL_SCHEMA,
        "algorithm_id": expected["algorithm_id"],
        "candidate_representation": CANDIDATE_REPRESENTATION_SINGLE_PAULI,
        "selector_identity": RA_STAGED_SELECTOR_ID,
        "active_gradient_policy": ACTIVE_GRADIENT_STATIONARY,
        "resource_weighting_scope": RESOURCE_WEIGHTING_ALL_PHASE,
    }
    if (
        refs != expected["source_lock_refs"]
        or getattr(receipt, "source_lock_refs_sha256", None)
        != canonical_sha256(refs)
        or any(
            getattr(receipt, field, None) != value
            for field, value in expected_receipt_fields.items()
        )
    ):
        raise ValueError(
            "Semantic-closure materialization authority drifted from its "
            "exact source-bound native contract."
        )
    return expected


def build_paper_i_ra_strong_weak_nph3_problem() -> ResolvedProblemContext:
    """Build the exact L=2 strong--weak, ``nph=3`` diagnostic problem."""

    return resolve_problem_context(
        ProblemRequest(
            problem_key="hh",
            num_sites=2,
            t=1.0,
            u=8.0,
            dv=0.0,
            omega0=1.0,
            g_ep=math.sqrt(0.125),
            n_ph_max=3,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
            include_zero_point=True,
            v_nn=0.0,
            t_prime=0.0,
            n_fermions=None,
        )
    )


PAPER_I_RA_CANONICAL_REGIME_IDS = (
    "weak_weak",
    "intermediate_weak",
    "strong_weak_u8",
    "weak_strong",
    "intermediate_strong",
    "strong_strong_u8",
)


def build_paper_i_ra_hh_regime_problem(
    regime_id: str,
) -> ResolvedProblemContext:
    """Build one exact canonical Paper-I Hubbard--Holstein regime."""

    regime = str(regime_id)
    settings: dict[str, tuple[float, float, int]] = {
        "weak_weak": (0.25, math.sqrt(0.125), 3),
        "intermediate_weak": (1.25, math.sqrt(0.125), 3),
        "strong_weak_u8": (8.0, math.sqrt(0.125), 3),
        "weak_strong": (0.25, math.sqrt(0.625), 7),
        "intermediate_strong": (1.25, math.sqrt(0.625), 7),
        "strong_strong_u8": (8.0, math.sqrt(0.625), 7),
    }
    try:
        u_value, g_value, n_ph_max = settings[regime]
    except KeyError as exc:
        raise ValueError("Unknown canonical Paper-I HH regime identity.") from exc
    problem = resolve_problem_context(
        ProblemRequest(
            problem_key="hh",
            num_sites=2,
            t=1.0,
            u=u_value,
            dv=0.0,
            omega0=1.0,
            g_ep=g_value,
            n_ph_max=n_ph_max,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
            include_zero_point=True,
            v_nn=0.0,
            t_prime=0.0,
            n_fermions=None,
        )
    )
    if _canonical_paper_i_hh_regime_id(problem) != regime:
        raise RuntimeError("Canonical Paper-I HH regime builder drifted.")
    return problem


def build_paper_i_ra_all_phase_adaptive_request(
    *,
    insertion_policy: str,
    maximum_controller_rounds: int = 50,
) -> RAAdaptRequest:
    """Build the production all-phase-adaptive append or plateau request."""

    insertion_key = str(insertion_policy)
    insertion_types = {
        "append_only": AppendOnlyInsertion,
        "plateau_commutation": PlateauCommutationInsertion,
    }
    try:
        insertion = insertion_types[insertion_key]()
    except KeyError as exc:
        raise ValueError(
            "All-phase adaptive insertion_policy must be 'append_only' or "
            "'plateau_commutation'."
        ) from exc
    horizon = maximum_controller_rounds
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise ValueError("All-phase adaptive horizon must be an integer.")
    if not 1 <= horizon <= 50:
        raise ValueError("All-phase adaptive horizon must be in [1, 50].")
    return RAAdaptRequest(
        adapter=PaperIRASemanticClosureGlobalSingletonCandidateAdapter(
            route_variant=PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1
        ),
        method=SRMethodPolicy(
            admission=SingletonAdmission(),
            insertion=insertion,
            pruning=PruningOff(),
            beam=BeamOff(),
        ),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(maximum_controller_rounds=horizon),
            resume=FreshStart(),
        ),
        observation=SRObservationPolicy(),
    )


def build_paper_i_ra_all_phase_adaptive_natural_terminal_request(
    *,
    insertion_policy: str,
    maximum_controller_rounds: int = 50,
) -> RAAdaptRequest:
    """Build the append-endpoint Phase-0 maximum-horizon V2 request."""

    insertion_key = str(insertion_policy)
    insertion_types = {
        "append_only": AppendOnlyInsertion,
        "plateau_commutation": PlateauCommutationInsertion,
    }
    try:
        insertion = insertion_types[insertion_key]()
    except KeyError as exc:
        raise ValueError(
            "All-phase natural-terminal insertion_policy must be "
            "'append_only' or 'plateau_commutation'."
        ) from exc
    horizon = maximum_controller_rounds
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise ValueError(
            "All-phase natural-terminal horizon must be an integer."
        )
    if not 1 <= horizon <= 50:
        raise ValueError(
            "All-phase natural-terminal horizon must be in [1, 50]."
        )
    return RAAdaptRequest(
        adapter=PaperIRASemanticClosureGlobalSingletonCandidateAdapter(
            route_variant=(
                PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2
            )
        ),
        method=SRMethodPolicy(
            admission=SingletonAdmission(),
            insertion=insertion,
            pruning=PruningOff(),
            beam=BeamOff(),
        ),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(maximum_controller_rounds=horizon),
            resume=FreshStart(),
        ),
        observation=SRObservationPolicy(),
    )


def build_paper_i_ra_all_phase_adaptive_forced_k50_request(
    *,
    insertion_policy: str,
    maximum_controller_rounds: int = 50,
) -> RAAdaptRequest:
    """Build the append-endpoint Phase-0 forced-admission exact-k50 request."""

    insertion_key = str(insertion_policy)
    insertion_types = {
        "append_only": AppendOnlyInsertion,
        "plateau_commutation": PlateauCommutationInsertion,
    }
    try:
        insertion = insertion_types[insertion_key]()
    except KeyError as exc:
        raise ValueError(
            "All-phase forced-k50 insertion_policy must be "
            "'append_only' or 'plateau_commutation'."
        ) from exc
    horizon = maximum_controller_rounds
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise ValueError("All-phase forced-k50 horizon must be an integer.")
    if not 1 <= horizon <= 50:
        raise ValueError("All-phase forced-k50 horizon must be in [1, 50].")
    return RAAdaptRequest(
        adapter=PaperIRASemanticClosureGlobalSingletonCandidateAdapter(
            route_variant=(
                PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_FORCED_K50_V1
            )
        ),
        method=SRMethodPolicy(
            admission=SingletonAdmission(),
            insertion=insertion,
            pruning=PruningOff(),
            beam=BeamOff(),
        ),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(maximum_controller_rounds=horizon),
            resume=FreshStart(),
        ),
        observation=SRObservationPolicy(),
    )


def build_paper_i_ra_all_phase_adaptive_min_floors_request(
    *,
    insertion_policy: str,
    maximum_controller_rounds: int = 50,
) -> RAAdaptRequest:
    """Build the endpoint-P0 min-floors natural-terminal request."""

    insertion_key = str(insertion_policy)
    insertion_types = {
        "append_only": AppendOnlyInsertion,
        "plateau_commutation": PlateauCommutationInsertion,
    }
    try:
        insertion = insertion_types[insertion_key]()
    except KeyError as exc:
        raise ValueError(
            "All-phase min-floors insertion_policy must be "
            "'append_only' or 'plateau_commutation'."
        ) from exc
    horizon = maximum_controller_rounds
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise ValueError("All-phase min-floors horizon must be an integer.")
    if not 1 <= horizon <= 50:
        raise ValueError("All-phase min-floors horizon must be in [1, 50].")
    return RAAdaptRequest(
        adapter=PaperIRASemanticClosureGlobalSingletonCandidateAdapter(
            route_variant=(
                PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1
            )
        ),
        method=SRMethodPolicy(
            admission=SingletonAdmission(),
            insertion=insertion,
            pruning=PruningOff(),
            beam=BeamOff(),
        ),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(maximum_controller_rounds=horizon),
            resume=FreshStart(),
        ),
        observation=SRObservationPolicy(),
    )


def build_paper_i_ra_all_phase_position_adaptive_min_floors_request(
    *,
    insertion_policy: str,
    maximum_controller_rounds: int = 50,
) -> RAAdaptRequest:
    """Build the position-P0 min-floors natural-terminal request."""

    insertion_key = str(insertion_policy)
    insertion_types = {
        "append_only": AppendOnlyInsertion,
        "plateau_commutation": PlateauCommutationInsertion,
        "always_commutation_reduced": AlwaysCommutationReducedInsertion,
    }
    try:
        insertion = insertion_types[insertion_key]()
    except KeyError as exc:
        raise ValueError(
            "Position min-floors insertion_policy must be 'append_only', "
            "'plateau_commutation', or 'always_commutation_reduced'."
        ) from exc
    horizon = maximum_controller_rounds
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise ValueError("Position min-floors horizon must be an integer.")
    if not 1 <= horizon <= 50:
        raise ValueError("Position min-floors horizon must be in [1, 50].")
    return RAAdaptRequest(
        adapter=PaperIRASemanticClosureGlobalSingletonCandidateAdapter(
            route_variant=(
                PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1
            )
        ),
        method=SRMethodPolicy(
            admission=SingletonAdmission(),
            insertion=insertion,
            pruning=PruningOff(),
            beam=BeamOff(),
        ),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(maximum_controller_rounds=horizon),
            resume=FreshStart(),
        ),
        observation=SRObservationPolicy(),
    )


def build_paper_i_ra_all_phase_position_adaptive_forced_k50_request(
    *,
    insertion_policy: str,
    maximum_controller_rounds: int = 50,
) -> RAAdaptRequest:
    """Build the position Phase-0 forced-admission exact-k50 request."""

    insertion_key = str(insertion_policy)
    insertion_types = {
        "append_only": AppendOnlyInsertion,
        "plateau_commutation": PlateauCommutationInsertion,
        "always_commutation_reduced": AlwaysCommutationReducedInsertion,
    }
    try:
        insertion = insertion_types[insertion_key]()
    except KeyError as exc:
        raise ValueError(
            "Position forced-k50 insertion_policy must be 'append_only', "
            "'plateau_commutation', or 'always_commutation_reduced'."
        ) from exc
    horizon = maximum_controller_rounds
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise ValueError("Position forced-k50 horizon must be an integer.")
    if not 1 <= horizon <= 50:
        raise ValueError("Position forced-k50 horizon must be in [1, 50].")
    return RAAdaptRequest(
        adapter=PaperIRASemanticClosureGlobalSingletonCandidateAdapter(
            route_variant=(
                PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_FORCED_K50_V1
            )
        ),
        method=SRMethodPolicy(
            admission=SingletonAdmission(),
            insertion=insertion,
            pruning=PruningOff(),
            beam=BeamOff(),
        ),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(maximum_controller_rounds=horizon),
            resume=FreshStart(),
        ),
        observation=SRObservationPolicy(),
    )


def build_paper_i_ra_all_phase_position_adaptive_request(
    *,
    insertion_policy: str,
    maximum_controller_rounds: int = 50,
) -> RAAdaptRequest:
    """Build the position-aware all-phase-adaptive append or plateau request."""

    insertion_key = str(insertion_policy)
    insertion_types = {
        "append_only": AppendOnlyInsertion,
        "plateau_commutation": PlateauCommutationInsertion,
    }
    try:
        insertion = insertion_types[insertion_key]()
    except KeyError as exc:
        raise ValueError(
            "Position-aware all-phase adaptive insertion_policy must be "
            "'append_only' or 'plateau_commutation'."
        ) from exc
    horizon = maximum_controller_rounds
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise ValueError(
            "Position-aware all-phase adaptive horizon must be an integer."
        )
    if not 1 <= horizon <= 50:
        raise ValueError(
            "Position-aware all-phase adaptive horizon must be in [1, 50]."
        )
    return RAAdaptRequest(
        adapter=PaperIRASemanticClosureGlobalSingletonCandidateAdapter(
            route_variant=(
                PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1
            )
        ),
        method=SRMethodPolicy(
            admission=SingletonAdmission(),
            insertion=insertion,
            pruning=PruningOff(),
            beam=BeamOff(),
        ),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(maximum_controller_rounds=horizon),
            resume=FreshStart(),
        ),
        observation=SRObservationPolicy(),
    )


def build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request(
    *,
    insertion_policy: str,
    maximum_controller_rounds: int = 50,
) -> RAAdaptRequest:
    """Build the opt-in maximum-horizon natural-terminal V2 request."""

    insertion_key = str(insertion_policy)
    insertion_types = {
        "append_only": AppendOnlyInsertion,
        "plateau_commutation": PlateauCommutationInsertion,
        "always_commutation_reduced": AlwaysCommutationReducedInsertion,
    }
    try:
        insertion = insertion_types[insertion_key]()
    except KeyError as exc:
        raise ValueError(
            "Position-aware natural-terminal insertion_policy must be "
            "'append_only', 'plateau_commutation', or "
            "'always_commutation_reduced'."
        ) from exc
    horizon = maximum_controller_rounds
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise ValueError(
            "Position-aware natural-terminal horizon must be an integer."
        )
    if not 1 <= horizon <= 50:
        raise ValueError(
            "Position-aware natural-terminal horizon must be in [1, 50]."
        )
    return RAAdaptRequest(
        adapter=PaperIRASemanticClosureGlobalSingletonCandidateAdapter(
            route_variant=(
                PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2
            )
        ),
        method=SRMethodPolicy(
            admission=SingletonAdmission(),
            insertion=insertion,
            pruning=PruningOff(),
            beam=BeamOff(),
        ),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(maximum_controller_rounds=horizon),
            resume=FreshStart(),
        ),
        observation=SRObservationPolicy(),
    )


def build_paper_i_ra_strong_weak_always_k5_request(
    route_variant: str,
) -> RAAdaptRequest:
    """Return the exact typed strong--weak always-open ``k=5`` request."""

    semantic_closure_route_identity(route_variant)
    return RAAdaptRequest(
        adapter=PaperIRASemanticClosureGlobalSingletonCandidateAdapter(
            route_variant=route_variant
        ),
        method=SRMethodPolicy(
            admission=SingletonAdmission(),
            insertion=AlwaysCommutationReducedInsertion(),
            pruning=PruningOff(),
            beam=BeamOff(),
        ),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(maximum_controller_rounds=5),
            resume=FreshStart(),
        ),
        observation=SRObservationPolicy(),
    )


def build_paper_i_ra_strong_weak_plateau_k5_request(
    route_variant: str,
) -> RAAdaptRequest:
    """Return the exact typed strong--weak plateau-controlled ``k=5`` request."""

    semantic_closure_route_identity(route_variant)
    return RAAdaptRequest(
        adapter=PaperIRASemanticClosureGlobalSingletonCandidateAdapter(
            route_variant=route_variant
        ),
        method=SRMethodPolicy(
            admission=SingletonAdmission(),
            insertion=PlateauCommutationInsertion(),
            pruning=PruningOff(),
            beam=BeamOff(),
        ),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(maximum_controller_rounds=5),
            resume=FreshStart(),
        ),
        observation=SRObservationPolicy(),
    )


def preflight_paper_i_ra_strong_weak_always_k5(
    problem: ResolvedProblemContext,
    request: RAAdaptRequest,
) -> ResolvedRAAdaptProtocol:
    """Resolve the exact k=5 request without granting execution authority."""

    _require_exact_strong_weak_always_k5_application(
        problem,
        request,
    )
    return preflight_paper_i_ra_semantic(problem, request)


def preflight_paper_i_ra_strong_weak_plateau_k5(
    problem: ResolvedProblemContext,
    request: RAAdaptRequest,
) -> ResolvedRAAdaptProtocol:
    """Resolve the exact plateau-controlled k=5 request without authority."""

    _require_exact_strong_weak_plateau_k5_application(problem, request)
    return preflight_paper_i_ra_semantic(problem, request)


def preflight_paper_i_ra_semantic(
    problem: ResolvedProblemContext,
    request: RAAdaptRequest,
) -> ResolvedRAAdaptProtocol:
    """Resolve one canonical six-regime semantic request without authority."""

    identity, _regime_id = _require_canonical_paper_i_semantic_application(
        problem,
        request,
    )
    adapter = request.adapter
    assert isinstance(
        adapter,
        PaperIRASemanticClosureGlobalSingletonCandidateAdapter,
    )
    from pipelines.static_adapt.ra_adapt.engine import (
        build_resolved_ra_protocol,
    )

    protocol = build_resolved_ra_protocol(problem, request)
    route = protocol.route_contract
    execution = (
        route.get("execution_settings", {})
        if isinstance(route, Mapping)
        else {}
    )
    if (
        protocol.algorithm_id != identity.algorithm_id
        or protocol.active_gradient_policy != ACTIVE_GRADIENT_STATIONARY
        or protocol.resource_weighting_scope != RESOURCE_WEIGHTING_ALL_PHASE
        or protocol.execution_authorized is not False
        or not isinstance(route, Mapping)
        or route.get("route_id") != identity.route_id
        or route.get("native_semantic_contract", {}).get(
            "semantic_implementation_version"
        )
        != identity.semantic_implementation_version
        or execution.get("phase3_backend_cost_scope")
        != BACKEND_COMPILE_SCOPE_PHASE123_QISKIT_V1
        or execution.get("phase3_hardware_cost_normalization_mode")
        != "zero_centered_signed_arctan_v1"
    ):
        raise RuntimeError(
            "Native semantic-closure protocol failed its canonical preflight."
        )
    return protocol


def materialize_paper_i_ra_semantic_protocol(
    problem: ResolvedProblemContext,
    request: RAAdaptRequest,
) -> ResolvedRAAdaptProtocol:
    """Mint one private source-bound capability for a canonical matrix cell."""

    from pipelines.static_adapt.ra_adapt.bundles import (
        materialize_semantic_closure_protocol,
    )

    return materialize_semantic_closure_protocol(problem, request)


def materialize_paper_i_ra_strong_weak_always_k5_protocol(
    problem: ResolvedProblemContext,
    request: RAAdaptRequest,
) -> ResolvedRAAdaptProtocol:
    """Mint a source-bound in-memory capability; never execute or write.

    The serialized protocol deliberately retains ``execution_authorized=False``.
    ``run_ra_adapt`` accepts it only while the private, non-serializable
    materialization authority returned by the validated bundle seam remains
    attached.
    """

    _require_exact_strong_weak_always_k5_application(problem, request)
    return materialize_paper_i_ra_semantic_protocol(problem, request)


__all__ = [
    "PAPER_I_RA_ALL_PHASE_ADAPTIVE_NATURAL_TERMINAL_IMPLEMENTATION_VERSION_V2",
    "PAPER_I_RA_ALL_PHASE_ADAPTIVE_NATURAL_TERMINAL_NATIVE_BUNDLE_ID_V2",
    "PAPER_I_RA_ALL_PHASE_ADAPTIVE_IMPLEMENTATION_VERSION_V1",
    "PAPER_I_RA_ALL_PHASE_ADAPTIVE_NATIVE_BUNDLE_ID_V1",
    "PAPER_I_RA_ALL_PHASE_ADAPTIVE_ROUTE_VARIANTS",
    "PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1",
    "PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_FORCED_K50_V1",
    "PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1",
    "PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2",
    "PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_IMPLEMENTATION_VERSION_V1",
    "PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_NATIVE_BUNDLE_ID_V1",
    "PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1",
    "PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_NATURAL_TERMINAL_IMPLEMENTATION_VERSION_V2",
    "PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_NATURAL_TERMINAL_NATIVE_BUNDLE_ID_V2",
    "PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_FORCED_K50_V1",
    "PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1",
    "PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2",
    "PAPER_I_RA_CANONICAL_REGIME_IDS",
    "PAPER_I_RA_PHASE0_GRADIENT_FIXED24",
    "PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2",
    "PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2",
    "PAPER_I_RA_PHASE0_LEGACY_ROUTE_VARIANTS",
    "PAPER_I_RA_PHASE0_PROXY_ADAPTIVE",
    "PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2",
    "PAPER_I_RA_PHASE0_PROXY_FIXED24_V2",
    "PAPER_I_RA_PHASE0_PROXY_FIXED24_SHADOW",
    "PAPER_I_RA_PHASE0_POSITION_GRADIENT_ADAPTIVE_V1",
    "PAPER_I_RA_PHASE0_POSITION_GRADIENT_FIXED24_V1",
    "PAPER_I_RA_PHASE0_POSITION_PROXY_ADAPTIVE_V1",
    "PAPER_I_RA_PHASE0_POSITION_PROXY_FIXED24_V1",
    "PAPER_I_RA_PHASE0_POSITION_ROUTE_VARIANTS",
    "PAPER_I_RA_PHASE0_EXECUTABLE_ROUTE_VARIANTS",
    "PAPER_I_RA_PHASE0_V2_ROUTE_VARIANTS",
    "PAPER_I_RA_PHASE3_NATURAL_TERMINAL_ROUTE_VARIANTS",
    "PAPER_I_RA_PHASE123_QISKIT_CACHE_POLICY",
    "PAPER_I_RA_PHASE123_QISKIT_COMPILE_SCOPE",
    "PAPER_I_RA_PHASE123_QISKIT_COST_POLICY",
    "PAPER_I_RA_SEMANTIC_PHASE0_CONSUMER_SCOPE",
    "PAPER_I_RA_SEMANTIC_PHASE0_GRADIENT_ADAPTIVE_RECEIPT_SCHEMA_V2",
    "PAPER_I_RA_SEMANTIC_POSITION_PHASE0_RECEIPT_SCHEMA",
    "PAPER_I_RA_SEMANTIC_PHASE0_POPULATION_SCOPE",
    "PAPER_I_RA_SEMANTIC_PHASE0_PROXY_RECEIPT_SCHEMA",
    "PAPER_I_RA_SEMANTIC_ADAPTER_ID",
    "PAPER_I_RA_SEMANTIC_ALGORITHM_IDS",
    "PAPER_I_RA_SEMANTIC_IMPLEMENTATION_VERSION",
    "PAPER_I_RA_SEMANTIC_IMPLEMENTATION_VERSION_V2",
    "PAPER_I_RA_SEMANTIC_IMPLEMENTATION_VERSION_POSITION_V1",
    "PAPER_I_RA_SEMANTIC_NATIVE_BUNDLE_ID",
    "PAPER_I_RA_SEMANTIC_NATIVE_BUNDLE_ID_V2",
    "PAPER_I_RA_SEMANTIC_NATIVE_EIGHT_ARM_BUNDLE_ID_V1",
    "PAPER_I_RA_SEMANTIC_ROUTE_CONTRACT_SCHEMA",
    "PAPER_I_RA_SEMANTIC_ROUTE_VARIANTS",
    "PAPER_I_RA_SEMANTIC_MATERIALIZATION_CONTRACT_SCHEMA",
    "PAPER_I_RA_SEMANTIC_SOURCE_INVENTORY_SCHEMA",
    "PaperIRASemanticClosureGlobalSingletonCandidateAdapter",
    "PaperIRASemanticClosureRouteIdentity",
    "build_paper_i_ra_all_phase_adaptive_request",
    "build_paper_i_ra_all_phase_adaptive_forced_k50_request",
    "build_paper_i_ra_all_phase_adaptive_min_floors_request",
    "build_paper_i_ra_all_phase_adaptive_natural_terminal_request",
    "build_paper_i_ra_all_phase_position_adaptive_request",
    "build_paper_i_ra_all_phase_position_adaptive_forced_k50_request",
    "build_paper_i_ra_all_phase_position_adaptive_min_floors_request",
    "build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request",
    "build_paper_i_ra_hh_regime_problem",
    "build_paper_i_ra_strong_weak_always_k5_request",
    "build_paper_i_ra_strong_weak_plateau_k5_request",
    "build_paper_i_ra_strong_weak_nph3_problem",
    "build_semantic_gradient_adaptive_phase0_receipt",
    "build_semantic_position_phase0_receipt",
    "build_semantic_proxy_phase0_receipt",
    "build_semantic_closure_route_contract",
    "canonical_semantic_execution_problem",
    "execute_semantic_phase0_runtime",
    "filter_semantic_phase0_position_domain",
    "is_semantic_closure_adapter",
    "materialize_paper_i_ra_strong_weak_always_k5_protocol",
    "materialize_paper_i_ra_semantic_protocol",
    "preflight_paper_i_ra_semantic",
    "preflight_paper_i_ra_strong_weak_always_k5",
    "preflight_paper_i_ra_strong_weak_plateau_k5",
    "project_approved_phase0_ablation",
    "semantic_closure_native_bundle_digest",
    "semantic_closure_native_bundle_id",
    "semantic_closure_native_bundle_manifest",
    "semantic_closure_materialization_contract",
    "semantic_closure_route_identity",
    "semantic_closure_route_identity_from_algorithm",
    "semantic_controller_horizon_policy",
    "semantic_phase3_no_positive_policy",
    "semantic_closure_source_implementation_inventory",
    "select_semantic_proxy_phase0_rows",
    "validate_semantic_phase0_runtime_binding",
    "validate_semantic_projected_phase123_receipt",
    "validate_semantic_final_selector_accounting",
    "validate_semantic_phase3_no_positive_terminal_receipt",
    "validate_semantic_phase3_natural_terminal_route_contract",
    "validate_semantic_position_phase0_receipt",
    "validate_semantic_closure_native_route_contract",
    "validate_semantic_closure_materialization_authority",
    "validate_semantic_gradient_adaptive_phase0_receipt",
    "validate_semantic_proxy_phase0_receipt",
]
