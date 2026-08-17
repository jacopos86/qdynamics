"""Finite, source-locked Paper-I RA-ADAPT run-bundle materialization.

This module is intentionally not a launcher.  It creates either the two
historical Study-1 handoff bundles or the user-selected stationary 48-cell
core, validates their immutable protocol surfaces, and writes non-executing
execution-manifest templates.  It never runs a scientific cell or submits a
scheduler job.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import platform
import sys
import tarfile
from typing import Any, Callable, Mapping, Sequence
import zipfile

from pipelines.static_adapt.ra_adapt.adapters import (
    GLOBAL_SINGLE_PAULI_ADAPTER_ID,
    PHASE_I_SUPPLY_GLOBAL_GUARDED_SINGLETON,
    PHASE_I_VISIBILITY_ALL_EXECUTABLE,
    PHASE_II_EXPOSURE_RETAINED_SINGLETON_IDENTITY,
    GlobalSinglePauliWordCandidateAdapter,
    MacroCandidateAdapter,
    SinglePauliWordCandidateAdapter,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    ACTIVE_GRADIENT_MEASURED,
    ACTIVE_GRADIENT_STATIONARY,
    APPEND_ADAPT_PROTOCOL_SCHEMA,
    APPEND_CONVENTIONAL_SELECTOR_ID,
    APPEND_CONVENTIONAL_SELECTOR_SCOPE,
    AppendAdaptRequest,
    BundleProtocolMaterializationAuthority,
    CANDIDATE_REPRESENTATION_MACRO,
    CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    EXACT_ORDERED_INSERTION_CHART,
    EXPANDED_RUNTIME_PROJECTED_LOGICAL_BASE_CHART,
    FULL_ENLARGED_ACCEPTED_REFIT,
    LOGICAL_SHARED_REDUCED_REFIT_BASE_CHART,
    NATIVE_REFIT_CHART,
    PhaseIIIMultiplierContract,
    PROJECTED_GENERALIZED_SOLVER,
    RA_ADAPT_NONSTATIONARY_INCREMENTAL_FULL_RESPONSE_ALGORITHM_ID,
    RA_ADAPT_PROTOCOL_SCHEMA,
    RA_ADAPT_PROTOCOL_SCHEMA_V2,
    RA_STAGED_SELECTOR_ID,
    RAAdaptRequest,
    RESOURCE_WEIGHTING_ALL_PHASE,
    RESOURCE_WEIGHTING_LATE,
    ResolvedRAAdaptProtocol,
    SOURCE_GRAM_NO_OVERLAP_TRUST,
    SUPPORTED_FS_WHITENED_REFIT_CHART,
    canonical_json_bytes,
    canonical_sha256,
    bundle_protocol_materialization_receipt,
    resolved_ra_adapt_protocol_from_mapping,
    _attach_validated_bundle_protocol_authority,
    _mint_bundle_protocol_materialization_authority,
)
from pipelines.static_adapt.ra_adapt.engine import (
    ALWAYS_REDUCED_INSERTION_EQUIVALENCE,
    ALWAYS_REDUCED_INSERTION_MODE,
    ALWAYS_REDUCED_INSERTION_SCOPE,
    RA_ADAPT_COMPILE_IDENTITY,
    RA_ADAPT_ESTIMATOR_ACCOUNTING,
    RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_ALGORITHM_ID,
    RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_SOURCE_ALGORITHM_ID,
    RA_ADAPT_PHASE3_QISKIT_COST_PHASE_REUSE,
    RA_ADAPT_PHASE3_QISKIT_COST_POLICY,
    RA_ADAPT_PHASE3_QISKIT_COST_ROUTE_SUFFIX,
    RA_ADAPT_QISKIT_COST_PHASE_REUSE,
    RA_ADAPT_QISKIT_COST_POLICY,
    RA_ADAPT_QISKIT_COST_ROUTE_SUFFIX,
    _repaired_route_contract,
    build_resolved_ra_protocol,
)
from pipelines.static_adapt.hh_backend_compile_oracle import (
    BACKEND_COMPILE_SCOPE_PHASE3_QISKIT_ONLY_V1,
    MARRAKESH_GRAPH_SPAN_MODE,
)
from pipelines.static_adapt.ra_adapt.pools import (
    GUARDED_SINGLETON_POOL_SCHEMA,
    PARENT_TEMPLATE_INVENTORY_SCHEMA,
)
from pipelines.static_adapt.ra_adapt.numerical_runtime import (
    NumericalRuntimeContractError,
    normalize_numerical_runtime_contract,
)
from pipelines.static_adapt.sr_snake.contracts import (
    PruningOff,
    BeamOff,
    TrustRegionPruning,
    MetricPruning,
    ForkLocalBeam,
    AlwaysCommutationReducedInsertion,
    AppendCommutationReducedInsertion,
    AppendOnlyInsertion,
    CheckpointObservation,
    EstimatorLedgerObservation,
    PlateauCommutationInsertion,
    SRExecutionPolicy,
    SRMethodPolicy,
    SRObservationPolicy,
    SRStopPolicy,
)


BUNDLE_SCHEMA = "ra_adapt_run_bundle_v1"
SOURCE_LOCK_SCHEMA = "ra_adapt_source_locks_v1"
EXPECTED_ARTIFACTS_SCHEMA = "ra_adapt_expected_artifacts_v1"
VALIDATION_REPORT_SCHEMA = "ra_adapt_bundle_validation_report_v1"
MATERIALIZATION_BINDING_SCHEMA = (
    "ra_adapt_bundle_materialization_binding_v1"
)
EXECUTION_TEMPLATE_SCHEMA = "ra_adapt_execution_manifest_template_v1"
BLOCKED_PROTOCOL_SCHEMA = "ra_adapt_blocked_protocol_v1"
STUDY1_EXECUTION_DEDUPE_SCHEMA = (
    "paper_i_ra_adapt_study1_execution_dedupe_v1"
)
PRESERVATION_EXECUTION_GATE_SCHEMA = (
    "paper_i_ra_adapt_preservation_execution_gate_v2"
)
PRESERVATION_MEASURED_GATE_ID = (
    "g13a_measured_same_problem_replay_v2"
)
PRESERVATION_STATIONARY_GATE_ID = (
    "g13b_stationary_zero_active_gradient_neutral_pair_v2"
)

STUDY_ID = "paper_i_ra_adapt_stationarity_comparison_v1"
RUN_CLASS = "candidate"
VISIBLE_TARGET_ID = (
    "paper_i_displayed_macro_rows_plus_targeted_singleton_preservation_v1"
)
CORE_CAMPAIGN_ID = "paper_i_ra_adapt_stationary_late_core_v1"
CORE_RUN_CLASS = "paper_facing"
CORE_VISIBLE_TARGET_ID = (
    "paper_i_stationary_source_phase_iii_active_gradient_full_48_cell_core_v1"
)
CORE_BUNDLE_ID = "ra_repair_stationary_late_core_v1"
CORE_SELECTION_AUTHORITY_PATH = (
    "agent_guidance/static-adapt/icm/ra-adapt-repair-20260727/"
    "user-review-stationary-core-20260728.json"
)
CORE_SELECTION_AUTHORITY_SHA256 = (
    "1b9e35d956ab7c93a1c02f0c4dd086906e8c7619cb182c064f259173b0fafad2"
)
FACTORIAL_CAMPAIGN_ID = (
    "paper_i_ra_adapt_always_stationarity_phase1_cost_factorial_v1"
)
FACTORIAL_RUN_CLASS = "diagnostic"
FACTORIAL_VISIBLE_TARGET_ID = (
    "paper_i_ra_adapt_always_stationarity_phase1_cost_factorial_v1"
)
GLOBAL_SINGLETON_CAMPAIGN_ID = (
    "paper_i_ra_adapt_global_singleton_insertion_comparison_v1"
)
GLOBAL_SINGLETON_BUNDLE_ID = (
    "ra_repair_global_singleton_stationary_all_phase_insertion_v1"
)
GLOBAL_SINGLETON_RUN_CLASS = "diagnostic"
GLOBAL_SINGLETON_VISIBLE_TARGET_ID = (
    "paper_i_ra_adapt_global_singleton_insertion_comparison_v1"
)
QISKIT_COST_PILOT_CAMPAIGN_ID = (
    "paper_i_ra_adapt_qiskit_cost_plateau_pilot_v1"
)
QISKIT_COST_PILOT_BUNDLE_ID = (
    "ra_adapt_qiskit_cost_plateau_pilot_v1"
)
QISKIT_COST_PILOT_RUN_CLASS = "diagnostic"
QISKIT_COST_PILOT_VISIBLE_TARGET_ID = (
    "paper_i_ra_adapt_qiskit_cost_plateau_pilot_v1"
)
QISKIT_COST_PILOT_EXECUTION_TARGET = "local"
QISKIT_COST_PILOT_MACRO_ALGORITHM_ID = (
    "paper_i_ra_adapt_macro_plateau_insertion_"
    "qiskit_transpile_cost_v1"
)
QISKIT_COST_PILOT_GLOBAL_SINGLETON_ALGORITHM_ID = (
    "paper_i_ra_adapt_global_singleton_plateau_commutation_"
    "qiskit_transpile_cost_v1"
)
QISKIT_COST_ALWAYS13_CAMPAIGN_ID = (
    "paper_i_ra_adapt_qiskit_cost_macro_always13_diagnostic_v1"
)
QISKIT_COST_ALWAYS13_BUNDLE_ID = (
    "ra_adapt_qiskit_cost_macro_always13_diagnostic_v1"
)
QISKIT_COST_ALWAYS13_RUN_CLASS = "diagnostic"
QISKIT_COST_ALWAYS13_VISIBLE_TARGET_ID = (
    "paper_i_ra_adapt_qiskit_cost_macro_always13_diagnostic_v1"
)
QISKIT_COST_ALWAYS13_EXECUTION_TARGET = "local"
QISKIT_COST_ALWAYS13_ALGORITHM_ID = (
    "paper_i_ra_adapt_macro_always_insertion_"
    "qiskit_transpile_cost_v1"
)
QISKIT_COST_ALWAYS13_HORIZON = 13
QISKIT_COST_ALWAYS6_CAMPAIGN_ID = (
    "paper_i_ra_adapt_qiskit_cost_macro_always6_diagnostic_v1"
)
QISKIT_COST_ALWAYS6_BUNDLE_ID = (
    "ra_adapt_qiskit_cost_macro_always6_diagnostic_v1"
)
QISKIT_COST_ALWAYS6_RUN_CLASS = "diagnostic"
QISKIT_COST_ALWAYS6_VISIBLE_TARGET_ID = (
    "paper_i_ra_adapt_qiskit_cost_macro_always6_diagnostic_v1"
)
QISKIT_COST_ALWAYS6_EXECUTION_TARGET = "local"
BEAMPRUNE_CAMPAIGN_ID = (
    "paper_i_ra_adapt_macro_always_beamprune_lanes_r50_v1"
)
BEAMPRUNE_BUNDLE_ID = "ra_adapt_macro_always_beamprune_lanes_r50_v1"
BEAMPRUNE_RUN_CLASS = "diagnostic"
BEAMPRUNE_VISIBLE_TARGET_ID = (
    "paper_i_ra_adapt_macro_always_beamprune_lanes_r50_v1"
)
BEAMPRUNE_EXECUTION_TARGET = "chtc"
BEAMPRUNE_HORIZON = 50
BEAMPRUNE_ARMS = (
    ("lanes_on_metric", "physical_operator_type", "metric"),
    ("lanes_off_metric", "global_single_population", "metric"),
    ("lanes_on_trust", "physical_operator_type", "trust_region"),
    ("lanes_off_trust", "global_single_population", "trust_region"),
)
LANES_ABLATION_CAMPAIGN_ID = (
    "paper_i_ra_adapt_macro_always_lanes_ablation_r50_v1"
)
LANES_ABLATION_BUNDLE_ID = (
    "ra_adapt_macro_always_lanes_ablation_r50_v1"
)
LANES_ABLATION_RUN_CLASS = "diagnostic"
LANES_ABLATION_VISIBLE_TARGET_ID = (
    "paper_i_ra_adapt_macro_always_lanes_ablation_r50_v1"
)
LANES_ABLATION_EXECUTION_TARGET = "chtc"
LANES_ABLATION_HORIZON = 50
LANES_ABLATION_LANES_ON_ALGORITHM_ID = (
    "paper_i_ra_adapt_macro_always_insertion_qiskit_transpile_cost_v1"
)
LANES_ABLATION_LANES_OFF_ALGORITHM_ID = (
    "paper_i_ra_adapt_macro_always_insertion_no_lanes_"
    "qiskit_transpile_cost_v1"
)
QISKIT_COST_ALWAYS6_HORIZON_BY_REGIME: dict[str, int] = {
    "weak_weak": 20,
    "intermediate_weak": 20,
    "strong_weak_u8": 15,
    "weak_strong": 20,
    "intermediate_strong": 20,
    "strong_strong_u8": 15,
}
QISKIT_COST_ALWAYS13_SOURCE_PROTOCOL_SHA256 = (
    "7ffc19e842f46ffa5a5317d76560550bcd7c98739903e6d529b5932e959ed2b3"
)
PHASE3_QISKIT_CAMPAIGN_ID = (
    "paper_i_ra_adapt_global_singleton_phase3_qiskit_"
    "mixed_horizon_v1"
)
PHASE3_QISKIT_BUNDLE_ID = (
    "ra_adapt_global_singleton_phase3_qiskit_mixed_horizon_v1"
)
PHASE3_QISKIT_RUN_CLASS = "candidate"
PHASE3_QISKIT_VISIBLE_TARGET_ID = (
    "paper_i_ra_adapt_global_singleton_phase3_qiskit_"
    "append_comparison_v1"
)
PHASE3_QISKIT_EXECUTION_TARGET = "chtc"
PHASE3_QISKIT_WEAK_HOLSTEIN_HORIZON = 50
PHASE3_QISKIT_STRONG_HOLSTEIN_HORIZON = 70
PHASE3_QISKIT_PAGE7_PARENT_ROUTE_PROFILE = (
    "paper_i_ra_adapt__single_pauli_word_v1__"
    "insertion_commutation_plateau_v2__"
    "global_guarded_singleton_phase_i__identity_phase_ii__"
    "stationary_source_response_v1__all_phase_resource_weighting_v1"
)
PHASE3_QISKIT_PAGE7_PARENT_CONTRACT_SHA256 = (
    "69af64db5bbaf5b811685b8353b82b748dc13d16306e4c08ddfe5ffde07f301b"
)
EXECUTION_TARGET = "chtc"
SUBMISSION_STATE = "not_submitted"
STATIONARY_BUNDLE_ID = "ra_repair_stationary_late_v1"
MEASURED_BUNDLE_ID = "ra_repair_measured_late_v1"
STUDY1_BUNDLE_POLICIES = (
    (STATIONARY_BUNDLE_ID, ACTIVE_GRADIENT_STATIONARY),
    (MEASURED_BUNDLE_ID, ACTIVE_GRADIENT_MEASURED),
)
FACTORIAL_STATIONARY_LATE_BUNDLE_ID = (
    "ra_repair_always_factorial_stationary_late_v1"
)
FACTORIAL_MEASURED_LATE_BUNDLE_ID = (
    "ra_repair_always_factorial_measured_late_v1"
)
FACTORIAL_STATIONARY_ALL_PHASE_BUNDLE_ID = (
    "ra_repair_always_factorial_stationary_all_phase_v1"
)
FACTORIAL_MEASURED_ALL_PHASE_BUNDLE_ID = (
    "ra_repair_always_factorial_measured_all_phase_v1"
)
FACTORIAL_BUNDLE_POLICIES = (
    (
        FACTORIAL_STATIONARY_LATE_BUNDLE_ID,
        ACTIVE_GRADIENT_STATIONARY,
        RESOURCE_WEIGHTING_LATE,
    ),
    (
        FACTORIAL_MEASURED_LATE_BUNDLE_ID,
        ACTIVE_GRADIENT_MEASURED,
        RESOURCE_WEIGHTING_LATE,
    ),
    (
        FACTORIAL_STATIONARY_ALL_PHASE_BUNDLE_ID,
        ACTIVE_GRADIENT_STATIONARY,
        RESOURCE_WEIGHTING_ALL_PHASE,
    ),
    (
        FACTORIAL_MEASURED_ALL_PHASE_BUNDLE_ID,
        ACTIVE_GRADIENT_MEASURED,
        RESOURCE_WEIGHTING_ALL_PHASE,
    ),
)
_FACTORIAL_AXIS_SUFFIXES = {
    (
        ACTIVE_GRADIENT_STATIONARY,
        RESOURCE_WEIGHTING_LATE,
    ): "gradient_stationary__phase1_cost_off",
    (
        ACTIVE_GRADIENT_MEASURED,
        RESOURCE_WEIGHTING_LATE,
    ): "gradient_measured__phase1_cost_off",
    (
        ACTIVE_GRADIENT_STATIONARY,
        RESOURCE_WEIGHTING_ALL_PHASE,
    ): "gradient_stationary__phase1_cost_on",
    (
        ACTIVE_GRADIENT_MEASURED,
        RESOURCE_WEIGHTING_ALL_PHASE,
    ): "gradient_measured__phase1_cost_on",
}

ROUTE_APPEND_MACRO = "append_macro"
ROUTE_RA_MACRO_APPEND_ONLY = "ra_macro_append_only"
ROUTE_RA_MACRO_PLATEAU = "ra_macro_plateau"
ROUTE_RA_MACRO_ALWAYS = "ra_macro_always"
ROUTE_SINGLETON_PLATEAU = "singleton_plateau"
ROUTE_APPEND_SINGLETON = "append_singleton"
ROUTE_RA_SINGLETON_APPEND_ONLY = "ra_singleton_append_only"
ROUTE_RA_SINGLETON_PLATEAU = "ra_singleton_plateau"
ROUTE_RA_SINGLETON_ALWAYS = "ra_singleton_always"
ROUTE_RA_GLOBAL_SINGLETON_APPEND_REDUCED = (
    "ra_global_singleton_append_commutation_reduced"
)
ROUTE_RA_GLOBAL_SINGLETON_PLATEAU = (
    "ra_global_singleton_plateau_commutation"
)
MACRO_ROUTE_IDS = (
    ROUTE_APPEND_MACRO,
    ROUTE_RA_MACRO_APPEND_ONLY,
    ROUTE_RA_MACRO_PLATEAU,
    ROUTE_RA_MACRO_ALWAYS,
)
VALIDATION_ROUTE_IDS = (*MACRO_ROUTE_IDS, ROUTE_SINGLETON_PLATEAU)
SINGLETON_CORE_ROUTE_IDS = (
    ROUTE_APPEND_SINGLETON,
    ROUTE_RA_SINGLETON_APPEND_ONLY,
    ROUTE_RA_SINGLETON_PLATEAU,
    ROUTE_RA_SINGLETON_ALWAYS,
)
GLOBAL_SINGLETON_INSERTION_ROUTE_IDS = (
    ROUTE_RA_GLOBAL_SINGLETON_APPEND_REDUCED,
    ROUTE_RA_GLOBAL_SINGLETON_PLATEAU,
)
QISKIT_COST_PILOT_ROUTE_IDS = (
    ROUTE_RA_MACRO_PLATEAU,
    ROUTE_RA_GLOBAL_SINGLETON_PLATEAU,
)
QISKIT_COST_ALWAYS13_ROUTE_IDS = (ROUTE_RA_MACRO_ALWAYS,)
PHASE3_QISKIT_ROUTE_IDS = (ROUTE_RA_GLOBAL_SINGLETON_PLATEAU,)

VALIDATION_REGIMES = ("strong_weak_u8", "strong_strong_u8")
CLAIM_FACING_REGIME_CUTOFF_PAIRS = (
    ("weak_weak", 3),
    ("intermediate_weak", 3),
    ("strong_weak_u8", 3),
    ("weak_strong", 7),
    ("intermediate_strong", 7),
    ("strong_strong_u8", 7),
)
FULL_VISIBLE_REGIME_CUTOFF_PAIRS = (
    ("weak_weak", 3),
    ("weak_weak", 7),
    ("intermediate_weak", 3),
    ("intermediate_weak", 7),
    ("strong_weak_u8", 3),
    ("strong_weak_u8", 7),
    ("weak_strong", 3),
    ("weak_strong", 7),
    ("intermediate_strong", 3),
    ("intermediate_strong", 7),
    ("strong_strong_u8", 3),
    ("strong_strong_u8", 7),
)
FULL_HORIZON = 50
EXPECTED_ARTIFACT_ROLES = (
    "execution_manifest",
    "checkpoint",
    "estimator_ledger",
    "result",
    "summary",
)
OBJECTIVE_EXECUTION_GATE_IDS = (
    "g1_physics_source_lock_equality_v2",
    "g2_same_cutoff_identity_v2",
    "g3_pool_identity_v2",
    "g4_refit_coordinate_convention_v2",
    "g5_insertion_position_correctness_v2",
    "g6_phase3_integrity_v2",
    "g7_policy_echo_v2",
    "g8_exact_reference_isolation_v2",
    "g9_numerical_physical_integrity_v2",
    "g10_accounting_closure_v2",
    "g11_checkpoint_replay_resume_v2",
    "g12_compile_identity_v2",
    "g13_same_physics_preservation_v2",
    "g14_completeness_v2",
)

# Membership identity is independent of Hamiltonian coefficients.  The full
# ordered-pool digest intentionally includes coefficient-bearing generator
# rows and is therefore recomputed and compared only within one resolved
# problem/regime.
MACRO_POOL_MEMBERSHIP_BY_NPH: Mapping[int, Mapping[str, Any]] = {
    3: {
        "count": 102,
        "ordered_labels_sha256": (
            "a8831528590e870a09ce08492b6f61da4a4d377e63fa8983b30ca9698af5d3d9"
        ),
    },
    7: {
        "count": 148,
        "ordered_labels_sha256": (
            "e6de937476653868f7d3974ad67c467c2f2e2496770e256671b2e807a5b5b03a"
        ),
    },
}
SINGLETON_PARENT_MEMBERSHIP_BY_NPH: Mapping[int, Mapping[str, Any]] = {
    3: {
        "count": 123,
        "ordered_labels_sha256": (
            "17cc97b744f8e6b50b686b24edd28426ca2c055bc2c31054fd353ddfa10efbe3"
        ),
    },
    7: {
        "count": 171,
        "ordered_labels_sha256": (
            "389ce1382b57b916e15e170c641f3884ed1ce33e9913d6eb709f24490739e93f"
        ),
    },
}
GLOBAL_SINGLETON_POOL_MEMBERSHIP_BY_NPH: Mapping[
    int, Mapping[str, Any]
] = {
    3: {
        "count": 948,
        "ordered_labels_sha256": (
            "02995a2c570d4322e46e55e3a532381ff7eff85dc3c2de8cb2b30ed888b76906"
        ),
    },
    7: {
        "count": 6508,
        "ordered_labels_sha256": (
            "079478057eea213139dc2f3c7486097496454421a44677c290b5dc55860accb7"
        ),
    },
}
GLOBAL_SINGLETON_ORDERED_POOL_SHA256_BY_REGIME: Mapping[str, str] = {
    "weak_weak": (
        "74880d215fd350fba57c2560eef6b6225d1caa69a7103d32624f70f6f3dfce84"
    ),
    "intermediate_weak": (
        "816dfa970a2b40e7c781f5440fcdfb33690a236f9ae853dfdd155e2f53c7e67f"
    ),
    "strong_weak_u8": (
        "62a24f68adc8a71f78fa5d3afb28356d15b988a2003e1c97e69871a65726e90c"
    ),
    "weak_strong": (
        "2b7416a82f70814e5d507ef6524bd8c8bd436c624dfc25495f7a4974188152c0"
    ),
    "intermediate_strong": (
        "078aa89647ee0449b73e3951d1c367d61a41eefeed67565ea3d8caecd81ded1a"
    ),
    "strong_strong_u8": (
        "7a0e3dacc93ef0e5af82c4f76d6956d5113844e2517723c06f26fb41a8568c59"
    ),
}

_SETTLED_CHANGE_IDS = frozenset(
    {
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "bundle_output_paths_and_labels",
        "bundle_horizon",
        "study1_axis",
        "study1_insertion_policy_variant",
        "study_authorized_cutoff_change",
        "approved_validation_cutoff_override",
        "core_stationary_gradient_policy",
        "core_candidate_representation_axis",
        "core_insertion_policy_variant",
        "core_conventional_append_baseline",
        "core_fixed_horizon",
        "global_singleton_candidate_adapter",
        "global_singleton_phase_i_candidate_supply",
        "global_singleton_phase_i_candidate_visibility",
        "global_singleton_phase_ii_candidate_exposure",
        "global_singleton_route_identity",
        "global_singleton_insertion_policy_variant",
        "qiskit_selector_cost_oracle",
        "qiskit_cost_all_phase_scope",
        "qiskit_cost_pilot_exact_cell_selection",
        "qiskit_cost_always13_insertion_policy",
        "qiskit_cost_always13_horizon",
        "qiskit_cost_always13_exact_cell_selection",
        "phase3_qiskit_selector_cost_scope",
        "phase3_qiskit_exact_cell_selection",
    }
)

_EXECUTION_ENTRYPOINTS = {
    "append_adapt": "pipelines.static_adapt.ra_adapt.run_append_adapt",
    "ra_adapt": "pipelines.static_adapt.ra_adapt.run_ra_adapt",
}

GLOBAL_SOURCE_LOCKS: Mapping[str, Mapping[str, str]] = {
    "macro_visible_provenance": {
        "path": (
            "MATH/paper_details/figures/"
            "paper_i_hh_macro_common_accuracy_20260723/"
            "paper_i_hh_macro_common_accuracy_20260723_provenance.json"
        ),
        "sha256": (
            "0153bbf8e1ea73c6c73bd559548d5e8b9e80b185effe9c4e8cd9ce6b7a1cae2e"
        ),
    },
    "macro_provenance_tracker": {
        "path": (
            "output/pdf/"
            "paper_i_hh_sr_snake_no_prune_no_beam_no_ordinary_novelty_tracking_20260715/"
            "paper_i_hh_sr_snake_no_prune_no_beam_no_ordinary_novelty_tracking_20260715.json"
        ),
        "sha256": (
            "8e057da082f9d81b308fc6563029ef9f1f309950b54191bd60bc47abfec95922"
        ),
    },
    "ed_cutoff_reference": {
        "path": (
            "MATH/paper_facing/paper_I_static_scaffold/"
            "paper_i_hh_ed_cutoff_reference_six_regime_20260727.json"
        ),
        "sha256": (
            "66a6409790affffd6ce8928d7fb46cc945b57d50e210d3cb215e8039a63c5573"
        ),
    },
    "visible_settings_resolver": {
        "path": (
            "agent_guidance/skills/shared/scripts/"
            "resolve_visible_settings.py"
        ),
        "sha256": (
            "7471668cf8a6a0b902e4884deb8e6fb52bad753c0269ef665684fbb250f9dd9a"
        ),
    },
}

_ALGORITHM_IDS = {
    ROUTE_APPEND_MACRO: "paper_i_append_adapt_v1",
    ROUTE_RA_MACRO_APPEND_ONLY: (
        "paper_i_ra_adapt_macro_append_only_repair_v1"
    ),
    ROUTE_RA_MACRO_PLATEAU: (
        "paper_i_ra_adapt_macro_plateau_insertion_repair_v1"
    ),
    # ``engine._macro_parent_contract`` uses this stable token to resolve the
    # characterized always-enabled insertion route.
    ROUTE_RA_MACRO_ALWAYS: (
        "paper_i_ra_adapt_macro_always_insertion_repair_v1"
    ),
    ROUTE_SINGLETON_PLATEAU: (
        "paper_i_ra_adapt_singleton_plateau_preservation_v1"
    ),
    ROUTE_APPEND_SINGLETON: "paper_i_append_adapt_v1",
    ROUTE_RA_SINGLETON_APPEND_ONLY: (
        "paper_i_ra_adapt_singleton_append_only_repair_v1"
    ),
    ROUTE_RA_SINGLETON_PLATEAU: (
        "paper_i_ra_adapt_singleton_plateau_insertion_repair_v1"
    ),
    ROUTE_RA_SINGLETON_ALWAYS: (
        "paper_i_ra_adapt_singleton_always_insertion_repair_v1"
    ),
    ROUTE_RA_GLOBAL_SINGLETON_APPEND_REDUCED: (
        "paper_i_ra_adapt_global_singleton_append_commutation_reduced_v1"
    ),
    ROUTE_RA_GLOBAL_SINGLETON_PLATEAU: (
        "paper_i_ra_adapt_global_singleton_plateau_commutation_v1"
    ),
}

_RA_INSERTION_KIND_BY_ROUTE = {
    ROUTE_RA_MACRO_APPEND_ONLY: AppendOnlyInsertion.kind,
    ROUTE_RA_MACRO_PLATEAU: PlateauCommutationInsertion.kind,
    ROUTE_RA_MACRO_ALWAYS: AlwaysCommutationReducedInsertion.kind,
    ROUTE_SINGLETON_PLATEAU: PlateauCommutationInsertion.kind,
    ROUTE_RA_SINGLETON_APPEND_ONLY: AppendOnlyInsertion.kind,
    ROUTE_RA_SINGLETON_PLATEAU: PlateauCommutationInsertion.kind,
    ROUTE_RA_SINGLETON_ALWAYS: AlwaysCommutationReducedInsertion.kind,
    ROUTE_RA_GLOBAL_SINGLETON_APPEND_REDUCED: (
        AppendCommutationReducedInsertion.kind
    ),
    ROUTE_RA_GLOBAL_SINGLETON_PLATEAU: (
        PlateauCommutationInsertion.kind
    ),
}


class BundleMaterializationError(ValueError):
    """Fail-closed bundle or source-lock validation error."""


def _protocol_schema_for_cell(cell: "BundleCellSpec") -> str:
    if cell.selector_family == "append_adapt":
        return APPEND_ADAPT_PROTOCOL_SCHEMA
    if (
        cell.algorithm_id
        == RA_ADAPT_NONSTATIONARY_INCREMENTAL_FULL_RESPONSE_ALGORITHM_ID
    ):
        return RA_ADAPT_PROTOCOL_SCHEMA_V2
    return RA_ADAPT_PROTOCOL_SCHEMA


@dataclass(frozen=True)
class BundleCellSpec:
    """One finite cell in a settled Study-1 bundle."""

    cell_id: str
    stage: str
    regime_id: str
    nph: int
    route_id: str
    algorithm_id: str
    selector_family: str
    candidate_representation: str
    horizon: int | None
    source_lock_id: str
    preservation_contract_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "cell_id": self.cell_id,
            "stage": self.stage,
            "regime_id": self.regime_id,
            "nph": int(self.nph),
            "route_id": self.route_id,
            "algorithm_id": self.algorithm_id,
            "selector_family": self.selector_family,
            "candidate_representation": self.candidate_representation,
            "horizon": self.horizon,
            "source_lock_id": self.source_lock_id,
        }
        if self.preservation_contract_id is not None:
            payload["preservation_contract_id"] = (
                self.preservation_contract_id
            )
        return payload


@dataclass(frozen=True)
class ProtocolResolutionContext:
    """Inputs supplied to one pure protocol resolver."""

    cell: BundleCellSpec
    problem: Any
    request: RAAdaptRequest | AppendAdaptRequest
    active_gradient_policy: str
    resource_weighting_scope: str
    bundle_id: str
    bundle_manifest_sha256: str
    source_lock_refs: Mapping[str, str]
    materialization_authority: BundleProtocolMaterializationAuthority


@dataclass(frozen=True)
class MaterializedBundleReceipt:
    bundle_id: str
    bundle_path: Path
    bundle_manifest_sha256: str
    source_locks_sha256: str
    expected_artifacts_sha256: str
    validation_report_sha256: str
    cell_count: int
    materialization_status: str


ProtocolResolver = Callable[[ProtocolResolutionContext], Any]
ProblemResolver = Callable[[str, int], Any]


def _require_sha256(value: Any, *, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise BundleMaterializationError(
            f"{label} must be a lowercase SHA-256 digest."
        )
    return digest


def _require_positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise BundleMaterializationError(f"{label} must be a positive integer.")
    resolved = int(value)
    if resolved != value or resolved < 1:
        raise BundleMaterializationError(f"{label} must be a positive integer.")
    return resolved


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_module_paths(
    repo_root: Path,
    *,
    module: str,
    imported_name: str | None = None,
) -> tuple[Path, ...]:
    """Resolve a Python import to existing repository-local source files."""

    parts = tuple(part for part in module.split(".") if part)
    if not parts or any(part in {".", ".."} for part in parts):
        return ()
    base = repo_root.joinpath(*parts)
    paths: list[Path] = []
    module_file = base.with_suffix(".py")
    package_file = base / "__init__.py"
    if module_file.is_file():
        paths.append(module_file)
    if package_file.is_file():
        paths.append(package_file)
    if imported_name and imported_name != "*":
        child = base / f"{imported_name}.py"
        child_package = base / imported_name / "__init__.py"
        if child.is_file():
            paths.append(child)
        if child_package.is_file():
            paths.append(child_package)
    return tuple(dict.fromkeys(paths))


def _module_with_package_initializers(
    repo_root: Path,
    path: Path,
) -> tuple[Path, ...]:
    """Return one local module plus every existing ancestor initializer."""

    root = repo_root.resolve()
    source = path.resolve()
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise BundleMaterializationError(
            f"Implementation source escaped the repository: {source}."
        ) from exc
    if not source.is_file():
        return ()
    paths = [source]
    directory = source.parent
    while directory != root:
        initializer = directory / "__init__.py"
        if initializer.is_file():
            paths.append(initializer.resolve())
        parent = directory.parent
        if parent == directory:
            raise BundleMaterializationError(
                f"Could not close package initializers for {source}."
            )
        directory = parent
    return tuple(dict.fromkeys(paths))


def _implementation_source_inventory(repo_root: Path) -> dict[str, Any]:
    """Hash the executable RA import closure, including package initializers."""

    package = repo_root / "pipelines" / "static_adapt" / "ra_adapt"
    roots = tuple(sorted(package.glob("*.py")))
    if not roots:
        raise BundleMaterializationError(
            "The canonical RA-ADAPT implementation package is missing."
        )
    pending = [
        source
        for root in roots
        for source in _module_with_package_initializers(repo_root, root)
    ]
    discovered: set[Path] = set()
    while pending:
        path = pending.pop()
        resolved = path.resolve()
        if resolved in discovered:
            continue
        discovered.add(resolved)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            raise BundleMaterializationError(
                f"Could not inventory implementation source {path}."
            ) from exc
        relative = path.relative_to(repo_root).with_suffix("")
        package_parts = relative.parts[:-1]
        if relative.name == "__init__":
            package_parts = relative.parts[:-1]
        for node in ast.walk(tree):
            candidates: tuple[Path, ...] = ()
            if isinstance(node, ast.Import):
                for alias in node.names:
                    candidates += _repo_module_paths(
                        repo_root, module=alias.name
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    keep = max(0, len(package_parts) - node.level + 1)
                    prefix = package_parts[:keep]
                    module_parts = tuple(
                        part for part in (node.module or "").split(".") if part
                    )
                    module = ".".join((*prefix, *module_parts))
                else:
                    module = str(node.module or "")
                for alias in node.names or ():
                    candidates += _repo_module_paths(
                        repo_root,
                        module=module,
                        imported_name=alias.name,
                    )
            for candidate in candidates:
                for source in _module_with_package_initializers(
                    repo_root, candidate
                ):
                    if source not in discovered:
                        pending.append(source)

    files = [
        {
            "path": path.relative_to(repo_root).as_posix(),
            "sha256": _hash_file(path),
        }
        for path in sorted(discovered)
    ]
    root_paths = [
        path.relative_to(repo_root).as_posix() for path in sorted(roots)
    ]
    package_initializer_paths = [
        row["path"]
        for row in files
        if row["path"].endswith("/__init__.py")
    ]
    return _digested(
        {
            "schema": "ra_adapt_implementation_source_inventory_v2",
            "resolution": (
                "static_repo_local_import_closure_with_package_"
                "initializers_v2"
            ),
            "root_paths": root_paths,
            "root_count": len(root_paths),
            "files": files,
            "file_count": len(files),
            "package_initializer_paths": package_initializer_paths,
            "package_initializer_count": len(
                package_initializer_paths
            ),
            "all_files_verified": True,
        }
    )


def _digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("sha256", None)
    result["sha256"] = canonical_sha256(result)
    return result


def _verify_digest(payload: Mapping[str, Any], *, label: str) -> str:
    observed = _require_sha256(payload.get("sha256"), label=f"{label}.sha256")
    digest_payload = dict(payload)
    digest_payload.pop("sha256", None)
    expected = canonical_sha256(digest_payload)
    if observed != expected:
        raise BundleMaterializationError(
            f"{label} canonical digest mismatch: {observed} != {expected}."
        )
    return observed


def _resolve_path(repo_root: Path, value: Any, *, label: str) -> Path:
    text = str(value).strip()
    if not text:
        raise BundleMaterializationError(f"{label} path is empty.")
    path = Path(text).expanduser()
    return path if path.is_absolute() else repo_root / path


def _hash_archive_member(archive: Path, member_path: str) -> str:
    """Stream one locked archive member into SHA-256.

    Displayed-row source members can be several GiB.  Verification must not
    materialize those bytes in memory or retain a second extracted catalog.
    """

    normalized = PurePosixPath(str(member_path))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise BundleMaterializationError(
            f"Unsafe archive member path: {member_path!r}."
        )
    digest = hashlib.sha256()
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as bundle:
            try:
                member = bundle.getmember(str(normalized))
            except KeyError as exc:
                raise BundleMaterializationError(
                    f"Archive {archive} has no member {member_path!r}."
                ) from exc
            if not member.isfile():
                raise BundleMaterializationError(
                    f"Archive member {member_path!r} is not a regular file."
                )
            stream = bundle.extractfile(member)
            if stream is None:
                raise BundleMaterializationError(
                    f"Could not read archive member {member_path!r}."
                )
            with stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive, "r") as bundle:
            try:
                with bundle.open(str(normalized), "r") as stream:
                    for block in iter(
                        lambda: stream.read(1024 * 1024),
                        b"",
                    ):
                        digest.update(block)
            except KeyError as exc:
                raise BundleMaterializationError(
                    f"Archive {archive} has no member {member_path!r}."
                ) from exc
        return digest.hexdigest()
    raise BundleMaterializationError(
        f"Unsupported source archive format: {archive}."
    )


def source_lock_id(regime_id: str, nph: int, route_id: str) -> str:
    return f"{regime_id}__nph{int(nph)}__{route_id}"


def build_study1_cell_specs(
    *,
    validation_horizon: int | None,
    full_horizon: int = FULL_HORIZON,
) -> tuple[BundleCellSpec, ...]:
    """Return the settled 10-validation + 48-full Study-1 matrix."""

    if validation_horizon is not None:
        validation_horizon = _require_positive_int(
            validation_horizon, label="validation_horizon"
        )
    full_horizon = _require_positive_int(full_horizon, label="full_horizon")
    if full_horizon != FULL_HORIZON:
        raise BundleMaterializationError(
            "The source-locked full-matrix horizon is fixed at 50."
        )

    cells: list[BundleCellSpec] = []
    for regime_id in VALIDATION_REGIMES:
        for route_id in VALIDATION_ROUTE_IDS:
            representation = (
                CANDIDATE_REPRESENTATION_SINGLE_PAULI
                if route_id == ROUTE_SINGLETON_PLATEAU
                else CANDIDATE_REPRESENTATION_MACRO
            )
            selector = (
                "append_adapt"
                if route_id == ROUTE_APPEND_MACRO
                else "ra_adapt"
            )
            cells.append(
                BundleCellSpec(
                    cell_id=(
                        f"validation__{regime_id}__nph3__{route_id}"
                    ),
                    stage="validation",
                    regime_id=regime_id,
                    nph=3,
                    route_id=route_id,
                    algorithm_id=_ALGORITHM_IDS[route_id],
                    selector_family=selector,
                    candidate_representation=representation,
                    horizon=validation_horizon,
                    source_lock_id=source_lock_id(
                        regime_id, 3, route_id
                    ),
                    preservation_contract_id=(
                        "historical_singleton_plateau_route_t13_v1"
                        if route_id == ROUTE_SINGLETON_PLATEAU
                        else None
                    ),
                )
            )

    for regime_id, nph in FULL_VISIBLE_REGIME_CUTOFF_PAIRS:
        for route_id in MACRO_ROUTE_IDS:
            cells.append(
                BundleCellSpec(
                    cell_id=f"full__{regime_id}__nph{nph}__{route_id}",
                    stage="full",
                    regime_id=regime_id,
                    nph=int(nph),
                    route_id=route_id,
                    algorithm_id=_ALGORITHM_IDS[route_id],
                    selector_family=(
                        "append_adapt"
                        if route_id == ROUTE_APPEND_MACRO
                        else "ra_adapt"
                    ),
                    candidate_representation=CANDIDATE_REPRESENTATION_MACRO,
                    horizon=full_horizon,
                    source_lock_id=source_lock_id(
                        regime_id, nph, route_id
                    ),
                )
            )
    if len(cells) != 58:
        raise AssertionError("The settled Study-1 matrix must contain 58 cells.")
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise AssertionError("Study-1 cell ids must be unique.")
    return tuple(cells)


def build_core_cell_specs(
    *,
    horizon: int = FULL_HORIZON,
) -> tuple[BundleCellSpec, ...]:
    """Return the selected-policy 48-cell claim-facing Paper-I core.

    This is deliberately a sibling of :func:`build_study1_cell_specs`.
    Study 1 remains two 58-cell logical bundles; the core is materialized
    only after the user selects one stationarity policy.
    """

    horizon = _require_positive_int(horizon, label="horizon")
    if horizon != FULL_HORIZON:
        raise BundleMaterializationError(
            "The claim-facing core horizon is fixed at 50."
        )
    routes_by_representation = (
        (CANDIDATE_REPRESENTATION_MACRO, MACRO_ROUTE_IDS),
        (
            CANDIDATE_REPRESENTATION_SINGLE_PAULI,
            SINGLETON_CORE_ROUTE_IDS,
        ),
    )
    cells: list[BundleCellSpec] = []
    for regime_id, nph in CLAIM_FACING_REGIME_CUTOFF_PAIRS:
        for representation, route_ids in routes_by_representation:
            for route_id in route_ids:
                cells.append(
                    BundleCellSpec(
                        cell_id=(
                            f"core__{regime_id}__nph{nph}__{route_id}"
                        ),
                        stage="core",
                        regime_id=regime_id,
                        nph=int(nph),
                        route_id=route_id,
                        algorithm_id=_ALGORITHM_IDS[route_id],
                        selector_family=(
                            "append_adapt"
                            if route_id
                            in {
                                ROUTE_APPEND_MACRO,
                                ROUTE_APPEND_SINGLETON,
                            }
                            else "ra_adapt"
                        ),
                        candidate_representation=representation,
                        horizon=horizon,
                        source_lock_id=source_lock_id(
                            regime_id, nph, route_id
                        ),
                    )
                )
    if len(cells) != 48:
        raise AssertionError(
            "The claim-facing Paper-I core must contain exactly 48 cells."
        )
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise AssertionError("Core cell ids must be unique.")
    return tuple(cells)


def build_factorial_always_cell_specs(
    *,
    active_gradient_policy: str,
    resource_weighting_scope: str,
    horizon: int = FULL_HORIZON,
) -> tuple[BundleCellSpec, ...]:
    """Return one exact 12-cell corrected-always factorial arm."""

    axis = (active_gradient_policy, resource_weighting_scope)
    try:
        suffix = _FACTORIAL_AXIS_SUFFIXES[axis]
    except KeyError as exc:
        raise BundleMaterializationError(
            "The corrected-always factorial accepts only the declared "
            "active-gradient and Phase-I resource-weighting axes."
        ) from exc
    horizon = _require_positive_int(horizon, label="horizon")
    if horizon != FULL_HORIZON:
        raise BundleMaterializationError(
            "The corrected-always factorial horizon is fixed at 50."
        )

    cells: list[BundleCellSpec] = []
    routes_by_representation = (
        (
            CANDIDATE_REPRESENTATION_MACRO,
            ROUTE_RA_MACRO_ALWAYS,
        ),
        (
            CANDIDATE_REPRESENTATION_SINGLE_PAULI,
            ROUTE_RA_SINGLETON_ALWAYS,
        ),
    )
    for regime_id, nph in CLAIM_FACING_REGIME_CUTOFF_PAIRS:
        for representation, route_id in routes_by_representation:
            base_id = f"core__{regime_id}__nph{nph}__{route_id}"
            cells.append(
                BundleCellSpec(
                    cell_id=f"{base_id}__{suffix}",
                    stage="factorial",
                    regime_id=regime_id,
                    nph=int(nph),
                    route_id=route_id,
                    algorithm_id=_ALGORITHM_IDS[route_id],
                    selector_family="ra_adapt",
                    candidate_representation=representation,
                    horizon=horizon,
                    source_lock_id=source_lock_id(
                        regime_id, nph, route_id
                    ),
                )
            )
    if len(cells) != 12:
        raise AssertionError(
            "Each corrected-always factorial arm must contain 12 cells."
        )
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise AssertionError("Factorial-arm cell ids must be unique.")
    return tuple(cells)


def build_global_singleton_insertion_cell_specs(
    *,
    horizon: int = FULL_HORIZON,
) -> tuple[BundleCellSpec, ...]:
    """Return the fixed 12-cell global-singleton insertion comparison."""

    horizon = _require_positive_int(horizon, label="horizon")
    if horizon != FULL_HORIZON:
        raise BundleMaterializationError(
            "The global-singleton insertion comparison horizon is fixed "
            "at 50."
        )
    cells = tuple(
        BundleCellSpec(
            cell_id=(
                f"global_singleton__{regime_id}__nph{nph}__{route_id}"
            ),
            stage="global_singleton_insertion",
            regime_id=regime_id,
            nph=int(nph),
            route_id=route_id,
            algorithm_id=_ALGORITHM_IDS[route_id],
            selector_family="ra_adapt",
            candidate_representation=CANDIDATE_REPRESENTATION_SINGLE_PAULI,
            horizon=horizon,
            source_lock_id=source_lock_id(regime_id, nph, route_id),
        )
        for regime_id, nph in CLAIM_FACING_REGIME_CUTOFF_PAIRS
        for route_id in GLOBAL_SINGLETON_INSERTION_ROUTE_IDS
    )
    if len(cells) != 12 or len({cell.cell_id for cell in cells}) != 12:
        raise AssertionError(
            "The global-singleton insertion comparison must contain "
            "exactly 12 unique cells."
        )
    return cells


def build_qiskit_cost_plateau_pilot_cell_specs(
    *,
    horizon: int = FULL_HORIZON,
) -> tuple[BundleCellSpec, ...]:
    """Return the exact local two-cell Qiskit-cost plateau diagnostic."""

    horizon = _require_positive_int(horizon, label="horizon")
    if horizon != FULL_HORIZON:
        raise BundleMaterializationError(
            "The Qiskit-cost plateau pilot horizon is fixed at 50."
        )
    cells = (
        BundleCellSpec(
            cell_id=(
                "qiskit_cost_pilot__strong_weak_u8__nph3__"
                f"{ROUTE_RA_MACRO_PLATEAU}"
            ),
            stage="qiskit_cost_plateau_pilot",
            regime_id="strong_weak_u8",
            nph=3,
            route_id=ROUTE_RA_MACRO_PLATEAU,
            algorithm_id=QISKIT_COST_PILOT_MACRO_ALGORITHM_ID,
            selector_family="ra_adapt",
            candidate_representation=CANDIDATE_REPRESENTATION_MACRO,
            horizon=horizon,
            source_lock_id=source_lock_id(
                "strong_weak_u8", 3, ROUTE_RA_MACRO_PLATEAU
            ),
        ),
        BundleCellSpec(
            cell_id=(
                "qiskit_cost_pilot__strong_strong_u8__nph7__"
                f"{ROUTE_RA_GLOBAL_SINGLETON_PLATEAU}"
            ),
            stage="qiskit_cost_plateau_pilot",
            regime_id="strong_strong_u8",
            nph=7,
            route_id=ROUTE_RA_GLOBAL_SINGLETON_PLATEAU,
            algorithm_id=(
                QISKIT_COST_PILOT_GLOBAL_SINGLETON_ALGORITHM_ID
            ),
            selector_family="ra_adapt",
            candidate_representation=CANDIDATE_REPRESENTATION_SINGLE_PAULI,
            horizon=horizon,
            source_lock_id=source_lock_id(
                "strong_strong_u8",
                7,
                ROUTE_RA_GLOBAL_SINGLETON_PLATEAU,
            ),
        ),
    )
    if len({cell.cell_id for cell in cells}) != 2:
        raise AssertionError(
            "The Qiskit-cost plateau pilot must contain two unique cells."
        )
    return cells


def build_qiskit_cost_always13_cell_specs(
    *,
    horizon: int = QISKIT_COST_ALWAYS13_HORIZON,
) -> tuple[BundleCellSpec, ...]:
    """Return the exact one-cell Qiskit-cost always-insertion diagnostic."""

    horizon = _require_positive_int(horizon, label="horizon")
    if horizon != QISKIT_COST_ALWAYS13_HORIZON:
        raise BundleMaterializationError(
            "The Qiskit-cost always13 diagnostic horizon is fixed at 13."
        )
    cells = (
        BundleCellSpec(
            cell_id=(
                "qiskit_cost_always13__strong_weak_u8__nph3__"
                f"{ROUTE_RA_MACRO_ALWAYS}"
            ),
            stage="qiskit_cost_always13_diagnostic",
            regime_id="strong_weak_u8",
            nph=3,
            route_id=ROUTE_RA_MACRO_ALWAYS,
            algorithm_id=QISKIT_COST_ALWAYS13_ALGORITHM_ID,
            selector_family="ra_adapt",
            candidate_representation=CANDIDATE_REPRESENTATION_MACRO,
            horizon=horizon,
            source_lock_id=source_lock_id(
                "strong_weak_u8", 3, ROUTE_RA_MACRO_ALWAYS
            ),
        ),
    )
    if len({cell.cell_id for cell in cells}) != 1:
        raise AssertionError(
            "The Qiskit-cost always13 diagnostic must contain one cell."
        )
    return cells


def build_phase3_qiskit_mixed_horizon_cell_specs(
    *,
    weak_holstein_horizon: int = PHASE3_QISKIT_WEAK_HOLSTEIN_HORIZON,
    strong_holstein_horizon: int = PHASE3_QISKIT_STRONG_HOLSTEIN_HORIZON,
) -> tuple[BundleCellSpec, ...]:
    """Return the exact six-cell Phase-III-Qiskit candidate campaign."""

    weak_holstein_horizon = _require_positive_int(
        weak_holstein_horizon,
        label="weak_holstein_horizon",
    )
    strong_holstein_horizon = _require_positive_int(
        strong_holstein_horizon,
        label="strong_holstein_horizon",
    )
    if (
        weak_holstein_horizon
        != PHASE3_QISKIT_WEAK_HOLSTEIN_HORIZON
        or strong_holstein_horizon
        != PHASE3_QISKIT_STRONG_HOLSTEIN_HORIZON
    ):
        raise BundleMaterializationError(
            "The Phase-III-Qiskit candidate horizons are fixed at 50 for "
            "the weak-Holstein sector and 70 for the strong-Holstein "
            "sector."
        )

    cells = tuple(
        BundleCellSpec(
            cell_id=(
                "phase3_qiskit_candidate__"
                f"{regime_id}__nph{nph}__"
                f"{ROUTE_RA_GLOBAL_SINGLETON_PLATEAU}"
            ),
            stage="phase3_qiskit_candidate",
            regime_id=regime_id,
            nph=int(nph),
            route_id=ROUTE_RA_GLOBAL_SINGLETON_PLATEAU,
            algorithm_id=(
                RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_ALGORITHM_ID
            ),
            selector_family="ra_adapt",
            candidate_representation=(
                CANDIDATE_REPRESENTATION_SINGLE_PAULI
            ),
            horizon=(
                weak_holstein_horizon
                if int(nph) == 3
                else strong_holstein_horizon
            ),
            source_lock_id=source_lock_id(
                regime_id,
                nph,
                ROUTE_RA_GLOBAL_SINGLETON_PLATEAU,
            ),
        )
        for regime_id, nph in CLAIM_FACING_REGIME_CUTOFF_PAIRS
    )
    if len(cells) != 6 or len({cell.cell_id for cell in cells}) != 6:
        raise AssertionError(
            "The Phase-III-Qiskit candidate campaign must contain "
            "exactly six unique cells."
        )
    return cells


def build_qiskit_cost_always6_cell_specs(
    *,
    horizon_by_regime: Mapping[str, int] | None = None,
) -> tuple[BundleCellSpec, ...]:
    """Return the exact six-cell macro always-insertion Qiskit-cost campaign."""

    horizons = dict(
        QISKIT_COST_ALWAYS6_HORIZON_BY_REGIME
        if horizon_by_regime is None
        else horizon_by_regime
    )
    if set(horizons) != set(QISKIT_COST_ALWAYS6_HORIZON_BY_REGIME):
        raise BundleMaterializationError(
            "The Qiskit-cost always6 diagnostic requires exactly the six "
            "claim-facing regimes."
        )
    for regime_id, horizon in horizons.items():
        resolved = _require_positive_int(horizon, label="horizon")
        if resolved != QISKIT_COST_ALWAYS6_HORIZON_BY_REGIME[regime_id]:
            raise BundleMaterializationError(
                "The Qiskit-cost always6 diagnostic horizons are fixed at "
                "20/20/15 for the weak-Holstein sector and 20/20/15 for "
                "the strong-Holstein sector."
            )

    cells = tuple(
        BundleCellSpec(
            cell_id=(
                "qiskit_cost_always6__"
                f"{regime_id}__nph{nph}__"
                f"{ROUTE_RA_MACRO_ALWAYS}"
            ),
            stage="qiskit_cost_always6_diagnostic",
            regime_id=regime_id,
            nph=int(nph),
            route_id=ROUTE_RA_MACRO_ALWAYS,
            algorithm_id=QISKIT_COST_ALWAYS13_ALGORITHM_ID,
            selector_family="ra_adapt",
            candidate_representation=CANDIDATE_REPRESENTATION_MACRO,
            horizon=int(horizons[regime_id]),
            source_lock_id=source_lock_id(
                regime_id, nph, ROUTE_RA_MACRO_ALWAYS
            ),
        )
        for regime_id, nph in CLAIM_FACING_REGIME_CUTOFF_PAIRS
    )
    if len(cells) != 6 or len({cell.cell_id for cell in cells}) != 6:
        raise AssertionError(
            "The Qiskit-cost always6 diagnostic must contain exactly six "
            "unique cells."
        )
    return cells


def build_beamprune_cell_specs(
    *,
    horizon: int = BEAMPRUNE_HORIZON,
) -> tuple[BundleCellSpec, ...]:
    """Return the exact 24-cell beam+prune lane ablation."""

    horizon = _require_positive_int(horizon, label="horizon")
    if horizon != BEAMPRUNE_HORIZON:
        raise BundleMaterializationError(
            "The beam+prune lane ablation horizon is fixed at 50."
        )
    cells = tuple(
        BundleCellSpec(
            cell_id=(
                f"beamprune__{arm}__{regime_id}__nph{nph}__"
                f"{ROUTE_RA_MACRO_ALWAYS}"
            ),
            stage="macro_always_beamprune_lane_ablation",
            regime_id=regime_id,
            nph=int(nph),
            route_id=ROUTE_RA_MACRO_ALWAYS,
            algorithm_id=(
                LANES_ABLATION_LANES_OFF_ALGORITHM_ID
                if lane_route == "global_single_population"
                else LANES_ABLATION_LANES_ON_ALGORITHM_ID
            ),
            selector_family="ra_adapt",
            candidate_representation=CANDIDATE_REPRESENTATION_MACRO,
            horizon=int(horizon),
            source_lock_id=source_lock_id(
                regime_id, nph, ROUTE_RA_MACRO_ALWAYS
            ),
        )
        for arm, lane_route, _prune in BEAMPRUNE_ARMS
        for regime_id, nph in CLAIM_FACING_REGIME_CUTOFF_PAIRS
    )
    if len(cells) != 24 or len({cell.cell_id for cell in cells}) != 24:
        raise AssertionError(
            "The beam+prune lane ablation must contain exactly 24 unique cells."
        )
    return cells


def _beamprune_contract(
    cells: Sequence[BundleCellSpec],
) -> dict[str, Any]:
    """Describe the crossed lane x prune-family beam ablation."""

    return {
        "schema": "paper_i_ra_adapt_macro_always_beamprune_lanes_v1",
        "comparison_shape": {
            "cell_count": 24,
            "arm_count": len(BEAMPRUNE_ARMS),
            "regime_count": 6,
            "candidate_representation_count": 1,
            "insertion_policy_count": 1,
        },
        "ablated_axes": [
            "physical_operator_lanes_active",
            "phase1_prune_schur_nomination_route",
        ],
        "held_fixed": {
            "insertion": AlwaysCommutationReducedInsertion.kind,
            "beam": "fork_local_3x2",
            "beam_live_parent_branches": 3,
            "beam_admission_children_per_parent": 2,
            "beam_maximum_admission_children_per_round": 6,
            "beam_s_alg_weight": 0.005,
            "phase1_prune_enabled": True,
            "phase1_prune_mode": "live",
            "phase1_prune_policy": "recoverability_ladder_v1",
            "backend_cost_mode": "transpile_single_v1",
        },
        "arms": [
            {
                "arm": arm,
                "static_lane_route": lane_route,
                "prune_family": prune,
                "phase1_prune_schur_nomination_route": (
                    "metric_regularized_v1"
                    if prune == "metric"
                    else "full_logical_fs_trust_delete_refit_v1"
                ),
            }
            for arm, lane_route, prune in BEAMPRUNE_ARMS
        ],
        "active_gradient_policy": ACTIVE_GRADIENT_STATIONARY,
        "resource_weighting_scope": RESOURCE_WEIGHTING_ALL_PHASE,
        "route_ids": [ROUTE_RA_MACRO_ALWAYS],
        "phase_i_shortlist_size": 24,
        "phase_ii_shortlist_size": 12,
        "horizon": BEAMPRUNE_HORIZON,
        "execution_target": BEAMPRUNE_EXECUTION_TARGET,
        "ordered_cell_ids": [cell.cell_id for cell in cells],
        "direct_execution_cell_count": len(cells),
        "scientific_role": (
            "macro_always_insertion_beam3x2_prune_lane_crossed_r50_v1"
        ),
    }


def build_lanes_ablation_cell_specs(
    *,
    horizon: int = LANES_ABLATION_HORIZON,
) -> tuple[BundleCellSpec, ...]:
    """Return the exact twelve-cell macro always-insertion lanes ablation."""

    horizon = _require_positive_int(horizon, label="horizon")
    if horizon != LANES_ABLATION_HORIZON:
        raise BundleMaterializationError(
            "The macro always-insertion lanes ablation horizon is fixed at 50."
        )
    arms = (
        ("lanes_on", LANES_ABLATION_LANES_ON_ALGORITHM_ID),
        ("lanes_off", LANES_ABLATION_LANES_OFF_ALGORITHM_ID),
    )
    cells = tuple(
        BundleCellSpec(
            cell_id=(
                f"lanes_ablation__{arm}__{regime_id}__nph{nph}__"
                f"{ROUTE_RA_MACRO_ALWAYS}"
            ),
            stage="macro_always_lanes_ablation",
            regime_id=regime_id,
            nph=int(nph),
            route_id=ROUTE_RA_MACRO_ALWAYS,
            algorithm_id=algorithm_id,
            selector_family="ra_adapt",
            candidate_representation=CANDIDATE_REPRESENTATION_MACRO,
            horizon=int(horizon),
            source_lock_id=source_lock_id(
                regime_id, nph, ROUTE_RA_MACRO_ALWAYS
            ),
        )
        for arm, algorithm_id in arms
        for regime_id, nph in CLAIM_FACING_REGIME_CUTOFF_PAIRS
    )
    if len(cells) != 12 or len({cell.cell_id for cell in cells}) != 12:
        raise AssertionError(
            "The macro always-insertion lanes ablation must contain exactly "
            "twelve unique cells."
        )
    return cells


def _lanes_ablation_contract(
    cells: Sequence[BundleCellSpec],
) -> dict[str, Any]:
    """Describe the one-axis lane ablation and its fixed controls."""

    return {
        "schema": "paper_i_ra_adapt_macro_always_lanes_ablation_v1",
        "comparison_shape": {
            "cell_count": 12,
            "arm_count": 2,
            "regime_count": 6,
            "candidate_representation_count": 1,
            "insertion_policy_count": 1,
        },
        "ablated_axis": "physical_operator_lanes_active",
        "arms": {
            "lanes_on": {
                "algorithm_id": LANES_ABLATION_LANES_ON_ALGORITHM_ID,
                "static_lane_route": "physical_operator_type",
                "physical_lane_shortlist_aggressiveness": 3,
                "physical_operator_lanes_active": True,
            },
            "lanes_off": {
                "algorithm_id": LANES_ABLATION_LANES_OFF_ALGORITHM_ID,
                "static_lane_route": "global_single_population",
                "physical_operator_lanes_active": False,
                "shortlist_population_policy": "single_global_population_v1",
            },
        },
        "active_gradient_policy": ACTIVE_GRADIENT_STATIONARY,
        "resource_weighting_scope": RESOURCE_WEIGHTING_ALL_PHASE,
        "phase1_cost_term": "enabled_v1",
        "candidate_representations": [CANDIDATE_REPRESENTATION_MACRO],
        "route_ids": [ROUTE_RA_MACRO_ALWAYS],
        "algorithm_ids": [
            LANES_ABLATION_LANES_ON_ALGORITHM_ID,
            LANES_ABLATION_LANES_OFF_ALGORITHM_ID,
        ],
        "typed_insertion_policy": AlwaysCommutationReducedInsertion.kind,
        "selector_compile_cost_policy": RA_ADAPT_QISKIT_COST_POLICY,
        "selector_compile_cost_phase_reuse": (
            RA_ADAPT_QISKIT_COST_PHASE_REUSE
        ),
        "backend_cost_mode": "transpile_single_v1",
        "backend_name": "FakeMarrakesh",
        "backend_optimization_level": 1,
        "backend_transpile_seed": 7,
        "parallel_gradient_workers": 4,
        "phase_i_shortlist_size": 24,
        "phase_ii_shortlist_size": 12,
        "horizon": LANES_ABLATION_HORIZON,
        "execution_target": LANES_ABLATION_EXECUTION_TARGET,
        "ordered_cell_ids": [cell.cell_id for cell in cells],
        "direct_execution_cell_count": len(cells),
        "scientific_role": (
            "macro_always_insertion_phase1_lane_ablation_paired_r50_v1"
        ),
    }


def _qiskit_cost_always6_contract(
    cells: Sequence[BundleCellSpec],
) -> dict[str, Any]:
    """Describe the six-regime macro always-insertion Qiskit-cost controls."""

    return {
        "schema": "paper_i_ra_adapt_qiskit_cost_macro_always6_v1",
        "comparison_shape": {
            "cell_count": 6,
            "candidate_representation_count": 1,
            "insertion_policy_count": 1,
        },
        "source_campaign_id": CORE_CAMPAIGN_ID,
        "source_route_id": ROUTE_RA_MACRO_ALWAYS,
        "changed_scientific_fields": [
            "route_contract.execution_settings.phase3_backend_cost_mode",
            "resource_weighting_scope",
            "request.execution.stop.maximum_controller_rounds",
        ],
        "active_gradient_policy": ACTIVE_GRADIENT_STATIONARY,
        "resource_weighting_scope": RESOURCE_WEIGHTING_ALL_PHASE,
        "phase1_cost_term": "enabled_v1",
        "candidate_representations": [CANDIDATE_REPRESENTATION_MACRO],
        "route_ids": [ROUTE_RA_MACRO_ALWAYS],
        "algorithm_ids": [QISKIT_COST_ALWAYS13_ALGORITHM_ID],
        "typed_insertion_policy": (
            AlwaysCommutationReducedInsertion.kind
        ),
        "selector_compile_cost_policy": RA_ADAPT_QISKIT_COST_POLICY,
        "selector_compile_cost_phase_reuse": (
            RA_ADAPT_QISKIT_COST_PHASE_REUSE
        ),
        "route_profile_suffix": RA_ADAPT_QISKIT_COST_ROUTE_SUFFIX,
        "backend_cost_mode": "transpile_single_v1",
        "backend_name": "FakeMarrakesh",
        "backend_optimization_level": 1,
        "backend_transpile_seed": 7,
        "parallel_gradient_workers": 4,
        "phase_i_shortlist_size": 24,
        "phase_ii_shortlist_size": 12,
        "horizon_by_regime": {
            cell.regime_id: int(cell.horizon) for cell in cells
        },
        "execution_target": QISKIT_COST_ALWAYS6_EXECUTION_TARGET,
        "ordered_cell_ids": [cell.cell_id for cell in cells],
        "direct_execution_cell_count": len(cells),
        "scientific_role": (
            "local_diagnostic_qiskit_cost_always_insertion_six_regime_v1"
        ),
    }


def _qiskit_cost_plateau_pilot_contract(
    cells: Sequence[BundleCellSpec],
) -> dict[str, Any]:
    """Describe the exact cost-oracle diagnostic and its fixed controls."""

    return {
        "schema": "paper_i_ra_adapt_qiskit_cost_plateau_pilot_v1",
        "comparison_shape": {
            "cell_count": 2,
            "candidate_representation_count": 2,
            "insertion_policy_count": 1,
        },
        "active_gradient_policy": ACTIVE_GRADIENT_STATIONARY,
        "resource_weighting_scope": RESOURCE_WEIGHTING_ALL_PHASE,
        "phase1_cost_term": "enabled_v1",
        "candidate_representations": [
            CANDIDATE_REPRESENTATION_MACRO,
            CANDIDATE_REPRESENTATION_SINGLE_PAULI,
        ],
        "route_ids": list(QISKIT_COST_PILOT_ROUTE_IDS),
        "algorithm_ids": [
            QISKIT_COST_PILOT_MACRO_ALGORITHM_ID,
            QISKIT_COST_PILOT_GLOBAL_SINGLETON_ALGORITHM_ID,
        ],
        "typed_insertion_policy": PlateauCommutationInsertion.kind,
        "selector_compile_cost_policy": RA_ADAPT_QISKIT_COST_POLICY,
        "selector_compile_cost_phase_reuse": (
            RA_ADAPT_QISKIT_COST_PHASE_REUSE
        ),
        "route_profile_suffix": RA_ADAPT_QISKIT_COST_ROUTE_SUFFIX,
        "backend_cost_mode": "transpile_single_v1",
        "backend_name": "FakeMarrakesh",
        "backend_optimization_level": 1,
        "backend_transpile_seed": 7,
        "parallel_gradient_workers": 4,
        "phase_i_shortlist_size": 24,
        "phase_ii_shortlist_size": 12,
        "horizon": FULL_HORIZON,
        "execution_target": QISKIT_COST_PILOT_EXECUTION_TARGET,
        "ordered_cell_ids": [cell.cell_id for cell in cells],
        "direct_execution_cell_count": len(cells),
        "scientific_role": (
            "local_diagnostic_qiskit_cost_oracle_ablation_v1"
        ),
    }


def _qiskit_cost_always13_contract(
    cells: Sequence[BundleCellSpec],
) -> dict[str, Any]:
    """Describe the exact insertion-and-horizon diagnostic controls."""

    return {
        "schema": "paper_i_ra_adapt_qiskit_cost_macro_always13_v1",
        "comparison_shape": {
            "cell_count": 1,
            "candidate_representation_count": 1,
            "insertion_policy_count": 1,
        },
        "source_campaign_id": QISKIT_COST_PILOT_CAMPAIGN_ID,
        "source_bundle_id": QISKIT_COST_PILOT_BUNDLE_ID,
        "source_route_id": ROUTE_RA_MACRO_PLATEAU,
        "source_algorithm_id": QISKIT_COST_PILOT_MACRO_ALGORITHM_ID,
        "changed_scientific_fields": [
            "request.method.insertion",
            "request.execution.stop.maximum_controller_rounds",
        ],
        "active_gradient_policy": ACTIVE_GRADIENT_STATIONARY,
        "resource_weighting_scope": RESOURCE_WEIGHTING_ALL_PHASE,
        "phase1_cost_term": "enabled_v1",
        "candidate_representations": [
            CANDIDATE_REPRESENTATION_MACRO,
        ],
        "route_ids": list(QISKIT_COST_ALWAYS13_ROUTE_IDS),
        "algorithm_ids": [QISKIT_COST_ALWAYS13_ALGORITHM_ID],
        "typed_insertion_policy": (
            AlwaysCommutationReducedInsertion.kind
        ),
        "selector_compile_cost_policy": RA_ADAPT_QISKIT_COST_POLICY,
        "selector_compile_cost_phase_reuse": (
            RA_ADAPT_QISKIT_COST_PHASE_REUSE
        ),
        "route_profile_suffix": RA_ADAPT_QISKIT_COST_ROUTE_SUFFIX,
        "backend_cost_mode": "transpile_single_v1",
        "backend_name": "FakeMarrakesh",
        "backend_optimization_level": 1,
        "backend_transpile_seed": 7,
        "parallel_gradient_workers": 4,
        "phase_i_shortlist_size": 24,
        "phase_ii_shortlist_size": 12,
        "horizon": QISKIT_COST_ALWAYS13_HORIZON,
        "execution_target": QISKIT_COST_ALWAYS13_EXECUTION_TARGET,
        "ordered_cell_ids": [cell.cell_id for cell in cells],
        "direct_execution_cell_count": len(cells),
        "scientific_role": (
            "local_diagnostic_qiskit_cost_always_insertion_prefix_v1"
        ),
    }


def _phase3_qiskit_mixed_horizon_contract(
    cells: Sequence[BundleCellSpec],
) -> dict[str, Any]:
    """Describe the source-locked candidate and Append comparison target."""

    return {
        "schema": (
            "paper_i_ra_adapt_global_singleton_phase3_qiskit_"
            "mixed_horizon_v1"
        ),
        "scientific_role": (
            "candidate_paper_facing_append_adapt_comparison_v1"
        ),
        "comparison_shape": {
            "regime_cutoff_pair_count": 6,
            "route_count": 1,
            "total_cell_count": 6,
        },
        "active_gradient_policy": ACTIVE_GRADIENT_STATIONARY,
        "resource_weighting_scope": RESOURCE_WEIGHTING_ALL_PHASE,
        "candidate_representation": (
            CANDIDATE_REPRESENTATION_SINGLE_PAULI
        ),
        "candidate_adapter_id": GLOBAL_SINGLE_PAULI_ADAPTER_ID,
        "phase_i_candidate_supply": (
            PHASE_I_SUPPLY_GLOBAL_GUARDED_SINGLETON
        ),
        "phase_i_candidate_visibility": (
            PHASE_I_VISIBILITY_ALL_EXECUTABLE
        ),
        "phase_ii_candidate_exposure": (
            PHASE_II_EXPOSURE_RETAINED_SINGLETON_IDENTITY
        ),
        "phase_iii_admission_cardinality": 1,
        "selector_identity": RA_STAGED_SELECTOR_ID,
        "route_ids": list(PHASE3_QISKIT_ROUTE_IDS),
        "algorithm_ids": [
            RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_ALGORITHM_ID
        ],
        "typed_insertion_policy": PlateauCommutationInsertion.kind,
        "selector_compile_cost_policy": (
            RA_ADAPT_PHASE3_QISKIT_COST_POLICY
        ),
        "selector_compile_cost_phase_reuse": (
            RA_ADAPT_PHASE3_QISKIT_COST_PHASE_REUSE
        ),
        "selector_compile_cost_scope": (
            BACKEND_COMPILE_SCOPE_PHASE3_QISKIT_ONLY_V1
        ),
        "phase_i_phase_ii_compile_cost_source": MARRAKESH_GRAPH_SPAN_MODE,
        "phase_iii_compile_cost_source": "backend_transpile_v1",
        "route_profile_suffix": RA_ADAPT_PHASE3_QISKIT_COST_ROUTE_SUFFIX,
        "backend_name": "FakeMarrakesh",
        "backend_optimization_level": 1,
        "backend_transpile_seed": 7,
        "backend_fallback_allowed": False,
        "negative_delta_reward_enabled": False,
        "raw_signed_telemetry_required": True,
        "weak_holstein": {
            "nph": 3,
            "horizon": PHASE3_QISKIT_WEAK_HOLSTEIN_HORIZON,
            "append_comparator_horizon": 50,
        },
        "strong_holstein": {
            "nph": 7,
            "horizon": PHASE3_QISKIT_STRONG_HOLSTEIN_HORIZON,
            "append_comparator_horizon": 70,
        },
        "comparison_metrics": {
            "primary": "final_same_cutoff_absolute_energy_error_v1",
            "secondary": [
                "s_alg_at_final_horizon_v1",
                "s_alg_at_first_common_error_crossing_v1",
            ],
        },
        "source_route_lineage": {
            "algorithm_id": (
                RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_SOURCE_ALGORITHM_ID
            ),
            "route_profile": PHASE3_QISKIT_PAGE7_PARENT_ROUTE_PROFILE,
            "route_contract_sha256": (
                PHASE3_QISKIT_PAGE7_PARENT_CONTRACT_SHA256
            ),
        },
        "ordered_cell_ids": [cell.cell_id for cell in cells],
        "direct_execution_cell_count": len(cells),
        "execution_target": PHASE3_QISKIT_EXECUTION_TARGET,
        "execution_authorized": False,
    }


def _global_singleton_insertion_contract(
    cells: Sequence[BundleCellSpec],
) -> dict[str, Any]:
    """Describe the one-insertion-axis scientific contract."""

    return {
        "schema": (
            "paper_i_ra_adapt_global_singleton_insertion_comparison_v1"
        ),
        "comparison_shape": {
            "regime_cutoff_pair_count": len(
                CLAIM_FACING_REGIME_CUTOFF_PAIRS
            ),
            "insertion_policy_count": len(
                GLOBAL_SINGLETON_INSERTION_ROUTE_IDS
            ),
            "cells_per_regime_cutoff_pair": 2,
            "total_cell_count": 12,
        },
        "active_gradient_policy": ACTIVE_GRADIENT_STATIONARY,
        "resource_weighting_scope": RESOURCE_WEIGHTING_ALL_PHASE,
        "phase1_cost_term": "enabled_v1",
        "candidate_representation": CANDIDATE_REPRESENTATION_SINGLE_PAULI,
        "candidate_adapter_id": GLOBAL_SINGLE_PAULI_ADAPTER_ID,
        "phase_i_candidate_supply": (
            PHASE_I_SUPPLY_GLOBAL_GUARDED_SINGLETON
        ),
        "phase_i_candidate_visibility": (
            PHASE_I_VISIBILITY_ALL_EXECUTABLE
        ),
        "phase_i_shortlist_size": 24,
        "phase_ii_candidate_exposure": (
            PHASE_II_EXPOSURE_RETAINED_SINGLETON_IDENTITY
        ),
        "phase_ii_shortlist_size": 12,
        "phase_iii_admission_cardinality": 1,
        "selector_identity": RA_STAGED_SELECTOR_ID,
        "regime_cutoff_pairs": [
            {"regime_id": regime_id, "nph": nph}
            for regime_id, nph in CLAIM_FACING_REGIME_CUTOFF_PAIRS
        ],
        "route_ids": list(GLOBAL_SINGLETON_INSERTION_ROUTE_IDS),
        "algorithm_ids": [
            _ALGORITHM_IDS[route_id]
            for route_id in GLOBAL_SINGLETON_INSERTION_ROUTE_IDS
        ],
        "insertion_policies": {
            ROUTE_RA_GLOBAL_SINGLETON_APPEND_REDUCED: {
                "typed_kind": AppendCommutationReducedInsertion.kind,
                "runtime_mode": (
                    AppendCommutationReducedInsertion.runtime_mode
                ),
                "position_scope": (
                    AppendCommutationReducedInsertion.position_scope
                ),
                "equivalence_policy": (
                    AppendCommutationReducedInsertion.equivalence_policy
                ),
            },
            ROUTE_RA_GLOBAL_SINGLETON_PLATEAU: {
                "typed_kind": PlateauCommutationInsertion.kind,
                "runtime_mode": "insertion_commutation_plateau_v2",
                "position_scope": (
                    "append_only_or_immediate_plateau_full_logical_"
                    "domain_v1"
                ),
                "equivalence_policy": (
                    "termwise_cross_component_commutation_"
                    "earliest_representative_v1"
                ),
                "prior_mean_decrease_ratio_threshold": 1e-4,
                "threshold_comparison": (
                    "marginal_to_prior_mean_strictly_below_v2"
                ),
                "patience": 1,
                "hysteresis_active": False,
            },
        },
        "parent_inventory_membership_by_nph": {
            str(nph): dict(contract)
            for nph, contract in SINGLETON_PARENT_MEMBERSHIP_BY_NPH.items()
        },
        "global_executable_pool_membership_by_nph": {
            str(nph): dict(contract)
            for nph, contract in (
                GLOBAL_SINGLETON_POOL_MEMBERSHIP_BY_NPH.items()
            )
        },
        "ordered_pool_sha256_by_regime": dict(
            GLOBAL_SINGLETON_ORDERED_POOL_SHA256_BY_REGIME
        ),
        "horizon": FULL_HORIZON,
        "ordered_cell_ids": [cell.cell_id for cell in cells],
        "direct_execution_cell_count": len(cells),
        "cross_arm_equality": (
            "all_common_scientific_fields_equal_outside_insertion_v1"
        ),
    }


def _factorial_policy_for_bundle(
    bundle_id: str,
) -> tuple[str, str]:
    for candidate_id, gradient_policy, resource_scope in (
        FACTORIAL_BUNDLE_POLICIES
    ):
        if bundle_id == candidate_id:
            return gradient_policy, resource_scope
    raise BundleMaterializationError(
        f"Unknown corrected-always factorial bundle id: {bundle_id!r}."
    )


def _factorial_arm_contract(
    *,
    active_gradient_policy: str,
    resource_weighting_scope: str,
    cells: Sequence[BundleCellSpec],
) -> dict[str, Any]:
    return {
        "schema": (
            "paper_i_ra_adapt_always_stationarity_"
            "phase1_cost_factorial_arm_v1"
        ),
        "factorial_shape": {
            "active_gradient_policy_count": 2,
            "resource_weighting_scope_count": 2,
            "regime_cutoff_pair_count": len(
                CLAIM_FACING_REGIME_CUTOFF_PAIRS
            ),
            "candidate_representation_count": 2,
            "total_cell_count": 48,
            "bundle_count": 4,
            "cells_per_bundle": 12,
        },
        "active_gradient_policy": active_gradient_policy,
        "resource_weighting_scope": resource_weighting_scope,
        "phase1_cost_term": (
            "enabled_v1"
            if resource_weighting_scope == RESOURCE_WEIGHTING_ALL_PHASE
            else "disabled_for_phase1_only_v1"
        ),
        "regime_cutoff_pairs": [
            {"regime_id": regime_id, "nph": nph}
            for regime_id, nph in CLAIM_FACING_REGIME_CUTOFF_PAIRS
        ],
        "candidate_representations": [
            CANDIDATE_REPRESENTATION_MACRO,
            CANDIDATE_REPRESENTATION_SINGLE_PAULI,
        ],
        "route_ids": [
            ROUTE_RA_MACRO_ALWAYS,
            ROUTE_RA_SINGLETON_ALWAYS,
        ],
        "typed_insertion_policy": (
            AlwaysCommutationReducedInsertion.kind
        ),
        "runtime_insertion_mode": ALWAYS_REDUCED_INSERTION_MODE,
        "insertion_position_scope": ALWAYS_REDUCED_INSERTION_SCOPE,
        "insertion_equivalence_policy": (
            ALWAYS_REDUCED_INSERTION_EQUIVALENCE
        ),
        "horizon": FULL_HORIZON,
        "ordered_cell_ids": [cell.cell_id for cell in cells],
        "direct_execution_cell_count": len(cells),
    }


def study1_shared_execution_dedupe_contract() -> dict[str, Any]:
    """Describe the two policy-inert Append executions shared in Study 1.

    Both bundle-bound protocols remain immutable and authoritative.  The
    execution layer runs the stationary-bundle Append cell once per regime
    and fulfills the measured-bundle duplicate through an explicit hashed
    result reference plus an Append scientific-equivalence receipt.
    """

    groups = []
    for regime_id in VALIDATION_REGIMES:
        cell_id = f"validation__{regime_id}__nph3__{ROUTE_APPEND_MACRO}"
        groups.append(
            {
                "group_id": f"study1_append_shared__{regime_id}__nph3",
                "scientific_cell_id": cell_id,
                "canonical_execution": {
                    "bundle_id": STATIONARY_BUNDLE_ID,
                    "cell_id": cell_id,
                },
                "shared_result_reference": {
                    "bundle_id": MEASURED_BUNDLE_ID,
                    "cell_id": cell_id,
                    "fulfillment_kind": "shared_result_reference_v1",
                    "completion_matrix_status_on_verified_reference": (
                        "done"
                    ),
                },
            }
        )
    return _digested(
        {
            "schema": STUDY1_EXECUTION_DEDUPE_SCHEMA,
            "materialized_validation_cell_count": 20,
            "unique_validation_execution_count": 18,
            "shared_execution_savings": 2,
            "bundle_protocol_authority": (
                "retained_independently_per_bundle_v1"
            ),
            "result_link_contract": (
                "hash_bound_reference_with_append_scientific_"
                "equivalence_receipt_v1"
            ),
            "scientific_equivalence_projection": {
                "selector_scope": (
                    "conventional_append_no_phase3_no_trust_v1"
                ),
                "policy_inert_fields": [
                    "active_gradient_policy",
                    "resource_weighting_scope",
                ],
                "bundle_binding_fields": [
                    "bundle_id",
                    "bundle_manifest_sha256",
                    "bundle_materialization",
                    "sha256",
                ],
                "required_equal_fields": [
                    "algorithm_id",
                    "candidate_representation",
                    "adapter_id",
                    "selector_identity",
                    "derivative_chart_id",
                    "accepted_refit_scope",
                    "accepted_refit_coordinate_chart",
                    "accepted_refit_base_chart_policy",
                    "problem",
                    "parent_inventory",
                    "executable_pool",
                    "optimizer",
                    "optimizer_maxiter",
                    "horizon",
                    "seeds",
                    "estimator_accounting_convention",
                    "compile_identity",
                ],
            },
            "groups": groups,
            "execution_authorized": False,
            "submission_state": SUBMISSION_STATE,
            "submitted": False,
        }
    )


def _study1_shared_execution_assignment(
    *,
    bundle_id: str,
    cell: BundleCellSpec,
) -> dict[str, Any]:
    contract = study1_shared_execution_dedupe_contract()
    for group in contract["groups"]:
        if group["scientific_cell_id"] != cell.cell_id:
            continue
        canonical = group["canonical_execution"]
        shared = group["shared_result_reference"]
        if bundle_id == canonical["bundle_id"]:
            return {
                "fulfillment_kind": "canonical_shared_execution_v1",
                "group_id": group["group_id"],
                "dedupe_contract_sha256": contract["sha256"],
                "canonical_execution": dict(canonical),
            }
        if bundle_id == shared["bundle_id"]:
            return {
                "fulfillment_kind": "shared_result_reference_v1",
                "group_id": group["group_id"],
                "dedupe_contract_sha256": contract["sha256"],
                "canonical_execution": dict(canonical),
                "completion_matrix_status_on_verified_reference": (
                    shared[
                        "completion_matrix_status_on_verified_reference"
                    ]
                ),
            }
    return {
        "fulfillment_kind": "direct_execution_v1",
        "dedupe_contract_sha256": contract["sha256"],
    }


def _execution_fulfillment_assignment(
    *,
    campaign_id: str,
    bundle_id: str,
    cell: BundleCellSpec,
) -> dict[str, Any]:
    if campaign_id == STUDY_ID:
        return _study1_shared_execution_assignment(
            bundle_id=bundle_id,
            cell=cell,
        )
    if campaign_id in {
        CORE_CAMPAIGN_ID,
        FACTORIAL_CAMPAIGN_ID,
        GLOBAL_SINGLETON_CAMPAIGN_ID,
        QISKIT_COST_PILOT_CAMPAIGN_ID,
        QISKIT_COST_ALWAYS13_CAMPAIGN_ID,
        QISKIT_COST_ALWAYS6_CAMPAIGN_ID,
        LANES_ABLATION_CAMPAIGN_ID,
        BEAMPRUNE_CAMPAIGN_ID,
        PHASE3_QISKIT_CAMPAIGN_ID,
    }:
        return {
            "fulfillment_kind": "direct_execution_v1",
            "canonical_execution": {
                "bundle_id": bundle_id,
                "cell_id": cell.cell_id,
            },
        }
    raise BundleMaterializationError(
        f"Unknown bundle campaign id: {campaign_id!r}."
    )


def _execution_target_for_campaign(campaign_id: str) -> str:
    if campaign_id == QISKIT_COST_PILOT_CAMPAIGN_ID:
        return QISKIT_COST_PILOT_EXECUTION_TARGET
    if campaign_id == QISKIT_COST_ALWAYS13_CAMPAIGN_ID:
        return QISKIT_COST_ALWAYS13_EXECUTION_TARGET
    if campaign_id == QISKIT_COST_ALWAYS6_CAMPAIGN_ID:
        return QISKIT_COST_ALWAYS6_EXECUTION_TARGET
    if campaign_id == LANES_ABLATION_CAMPAIGN_ID:
        return LANES_ABLATION_EXECUTION_TARGET
    if campaign_id == BEAMPRUNE_CAMPAIGN_ID:
        return BEAMPRUNE_EXECUTION_TARGET
    if campaign_id == PHASE3_QISKIT_CAMPAIGN_ID:
        return PHASE3_QISKIT_EXECUTION_TARGET
    if campaign_id in {
        STUDY_ID,
        CORE_CAMPAIGN_ID,
        FACTORIAL_CAMPAIGN_ID,
        GLOBAL_SINGLETON_CAMPAIGN_ID,
    }:
        return EXECUTION_TARGET
    raise BundleMaterializationError(
        f"Unknown bundle campaign id: {campaign_id!r}."
    )


def preservation_execution_gate_contract(
    *,
    active_gradient_policy: str,
) -> dict[str, Any]:
    """Return the policy-specific, same-physics G13 preservation contract.

    T13 remains a generic route characterization at its own U=2, g=1
    physics.  It is never used as a numerical baseline for the U=8 Study-1
    cells.
    """

    if active_gradient_policy == ACTIVE_GRADIENT_MEASURED:
        gate_id = PRESERVATION_MEASURED_GATE_ID
        requirements = {
            "same_problem_deterministic_replay_required": True,
            "paired_policy_comparison_required": True,
            "trajectory_deviation_is_pass_condition": False,
            "zero_active_gradient_indices_required": False,
            "zero_active_gradient_charge_required": False,
        }
    elif active_gradient_policy == ACTIVE_GRADIENT_STATIONARY:
        gate_id = PRESERVATION_STATIONARY_GATE_ID
        requirements = {
            "same_problem_deterministic_replay_required": True,
            "paired_policy_comparison_required": True,
            "trajectory_deviation_is_pass_condition": False,
            "zero_active_gradient_indices_required": True,
            "zero_active_gradient_charge_required": True,
        }
    else:
        raise BundleMaterializationError(
            "Unknown active-gradient policy for the G13 preservation gate: "
            f"{active_gradient_policy!r}."
        )
    return _digested(
        {
            "schema": PRESERVATION_EXECUTION_GATE_SCHEMA,
            "gate_id": gate_id,
            "generic_route_characterization": {
                "fixture_contract_id": (
                    "historical_singleton_plateau_route_t13_v1"
                ),
                "fixture_problem_role": (
                    "u2_g1_route_characterization_only_v1"
                ),
                "study1_numerical_baseline": False,
                "preflight_required": True,
            },
            "study1_comparison_space": (
                "same_resolved_problem_route_horizon_except_gradient_policy_v1"
            ),
            "active_gradient_policy": active_gradient_policy,
            "requirements": requirements,
        }
    )


def validate_preservation_execution_gate(
    *,
    active_gradient_policy: str,
    generic_t13_characterization_passed: bool,
    same_problem_deterministic_replay_passed: bool,
    paired_policy_comparison_available: bool,
    paired_trajectory_max_abs_deviation: float,
    active_gradient_indices_acquired: Sequence[int],
    active_gradient_charge: int,
) -> dict[str, Any]:
    """Validate one executed preservation cell against its G13 semantics."""

    booleans = (
        generic_t13_characterization_passed,
        same_problem_deterministic_replay_passed,
        paired_policy_comparison_available,
    )
    if any(not isinstance(value, bool) for value in booleans):
        raise BundleMaterializationError(
            "G13 characterization/replay/pair observations must be booleans."
        )
    deviation = float(paired_trajectory_max_abs_deviation)
    if not math.isfinite(deviation) or deviation < 0.0:
        raise BundleMaterializationError(
            "G13 paired trajectory deviation must be finite and nonnegative."
        )
    indices: list[int] = []
    for value in active_gradient_indices_acquired:
        if (
            isinstance(value, bool)
            or int(value) != value
            or int(value) < 0
        ):
            raise BundleMaterializationError(
                "G13 active-gradient indices must be nonnegative integers."
            )
        indices.append(int(value))
    if (
        isinstance(active_gradient_charge, bool)
        or int(active_gradient_charge) != active_gradient_charge
        or int(active_gradient_charge) < 0
    ):
        raise BundleMaterializationError(
            "G13 active-gradient charge must be a nonnegative integer."
        )
    charge = int(active_gradient_charge)
    contract = preservation_execution_gate_contract(
        active_gradient_policy=active_gradient_policy
    )
    passed = (
        generic_t13_characterization_passed
        and same_problem_deterministic_replay_passed
        and paired_policy_comparison_available
        and (
            active_gradient_policy == ACTIVE_GRADIENT_MEASURED
            or (indices == [] and charge == 0)
        )
    )
    if not passed:
        raise BundleMaterializationError(
            f"Preservation execution failed {contract['gate_id']}."
        )
    return _digested(
        {
            "schema": "paper_i_ra_adapt_preservation_gate_receipt_v2",
            "gate_contract_sha256": contract["sha256"],
            "gate_id": contract["gate_id"],
            "active_gradient_policy": active_gradient_policy,
            "generic_t13_characterization_passed": (
                generic_t13_characterization_passed
            ),
            "same_problem_deterministic_replay_passed": (
                same_problem_deterministic_replay_passed
            ),
            "paired_policy_comparison_available": (
                paired_policy_comparison_available
            ),
            "paired_trajectory_max_abs_deviation": deviation,
            "paired_trajectory_deviation_role": (
                "neutral_observation_not_pass_condition_v1"
            ),
            "active_gradient_indices_acquired": indices,
            "active_gradient_charge": charge,
            "status": "passed",
        }
    )


_BASELINE_PROTOCOL_BINDINGS: Mapping[str, tuple[str, str]] = {
    "L": ("problem.num_sites", "exact"),
    "t": ("problem.t", "exact"),
    "u": ("problem.u", "exact"),
    "dv": ("problem.dv", "exact"),
    "v_nn": ("problem.v_nn", "exact"),
    "t_prime": ("problem.t_prime", "exact"),
    "omega0": ("problem.omega0", "exact"),
    "g_ep": ("problem.g_ep", "exact"),
    "n_ph_max": ("problem.n_ph_max", "exact_or_cutoff_axis"),
    "boson_encoding": ("problem.boson_encoding", "exact"),
    "ordering": ("problem.ordering", "exact"),
    "boundary": ("problem.boundary", "exact"),
    "include_zero_point": ("problem.include_zero_point", "exact"),
    "adapt_inner_optimizer": ("optimizer", "optimizer_family"),
    "adapt_optimizer_kind": ("optimizer", "optimizer_family"),
    "optimizer": ("optimizer", "optimizer_family"),
    "optimizer_kind": ("optimizer", "optimizer_family"),
    "adapt_final_refit_maxiter": ("optimizer_maxiter", "exact"),
    "adapt_maxiter": ("optimizer_maxiter", "exact"),
    "optimizer_maxiter": ("optimizer_maxiter", "exact"),
    "maxiter": ("optimizer_maxiter", "exact"),
    "adapt_seed": ("seeds.adapt", "exact"),
    "seed": ("seeds.adapt", "exact"),
    "phase3_backend_transpile_seed": ("seeds.transpiler", "exact"),
}

_RA_ROUTE_BASELINE_BINDINGS: Mapping[str, tuple[str, str]] = {
    name: (f"route_contract.execution_settings.{name}", "exact")
    for name in (
        "adapt_final_full_refit",
        "adapt_finite_angle",
        "adapt_full_refit_every",
        "adapt_reopt_policy",
        "adapt_window_size",
        "adapt_window_topk",
        "phase1_prune_enabled",
        "phase2_enable_batching",
    )
}


def _declare_protocol_field_bindings(
    trace: Mapping[str, Any],
    *,
    cell: BundleCellSpec,
) -> tuple[dict[str, Any], ...]:
    reused = trace.get("settings_reused")
    settings = (
        reused.get("settings")
        if isinstance(reused, Mapping)
        else None
    )
    if not isinstance(settings, Mapping):
        raise BundleMaterializationError(
            f"Resolver trace has no settings payload for {cell.source_lock_id}."
        )
    changes = trace.get("settings_changed", ())
    change_ids = {
        str(change.get("id"))
        for change in changes
        if isinstance(change, Mapping)
    }
    unexpected = sorted(change_ids.difference(_SETTLED_CHANGE_IDS))
    if unexpected:
        raise BundleMaterializationError(
            f"Resolver trace has unapproved setting changes for "
            f"{cell.source_lock_id}: {unexpected}."
        )
    mapping = dict(_BASELINE_PROTOCOL_BINDINGS)
    if cell.selector_family == "ra_adapt":
        mapping.update(_RA_ROUTE_BASELINE_BINDINGS)
    bindings: list[dict[str, Any]] = []
    for source_name, (protocol_path, comparison) in mapping.items():
        if source_name not in settings:
            continue
        binding: dict[str, Any] = {
            "source_path": f"settings_reused.settings.{source_name}",
            "protocol_path": protocol_path,
            "comparison": comparison,
        }
        if (
            comparison == "exact_or_cutoff_axis"
            and int(settings[source_name]) != int(cell.nph)
        ):
            allowed = sorted(
                change_ids.intersection(
                    {
                        "study_authorized_cutoff_change",
                        "approved_validation_cutoff_override",
                    }
                )
            )
            if not allowed:
                raise BundleMaterializationError(
                    f"Cutoff drift for {cell.source_lock_id} is not tied to "
                    "a settled cutoff-axis receipt."
                )
            binding["authorized_delta_ids"] = allowed
        bindings.append(binding)
    if not bindings:
        raise BundleMaterializationError(
            f"Resolver trace exposes no exact reusable protocol field for "
            f"{cell.source_lock_id}."
        )
    return tuple(bindings)


def _verify_resolver_trace(
    trace: Mapping[str, Any],
    *,
    cell: BundleCellSpec,
    member_sha256: str,
) -> dict[str, Any]:
    required = {
        "source_map",
        "regime_or_case",
        "method",
        "source_json",
        "settings_reused",
        "settings_changed",
        "status",
        "problems",
    }
    missing = sorted(required.difference(trace))
    if missing:
        raise BundleMaterializationError(
            f"Source lock {cell.source_lock_id} has a resolver trace missing "
            f"{missing}."
        )
    if str(trace["regime_or_case"]) != cell.regime_id:
        raise BundleMaterializationError(
            f"Resolver trace regime drift for {cell.source_lock_id}."
        )
    if not str(trace["method"]).strip():
        raise BundleMaterializationError(
            f"Resolver trace method is empty for {cell.source_lock_id}."
        )
    if not isinstance(trace["settings_reused"], Mapping) or not trace[
        "settings_reused"
    ]:
        raise BundleMaterializationError(
            f"Resolver trace has no reusable settings for "
            f"{cell.source_lock_id}."
        )
    if not isinstance(trace["settings_changed"], (tuple, list)):
        raise BundleMaterializationError(
            f"Resolver trace settings_changed is not a list for "
            f"{cell.source_lock_id}."
        )
    if not isinstance(trace["problems"], (tuple, list)):
        raise BundleMaterializationError(
            f"Resolver trace problems is not a list for "
            f"{cell.source_lock_id}."
        )
    if str(trace["status"]) != "ok":
        raise BundleMaterializationError(
            f"Resolver trace is blocked for {cell.source_lock_id}."
        )
    same_cutoff_ed = trace.get("same_cutoff_ed_reference")
    if not isinstance(same_cutoff_ed, Mapping):
        raise BundleMaterializationError(
            f"Resolver trace has no same-cutoff ED reference for "
            f"{cell.source_lock_id}."
        )
    ed_authority = GLOBAL_SOURCE_LOCKS["ed_cutoff_reference"]
    if (
        str(same_cutoff_ed.get("path")) != ed_authority["path"]
        or _require_sha256(
            same_cutoff_ed.get("sha256"),
            label=(
                f"resolver trace {cell.source_lock_id} "
                "same-cutoff ED SHA-256"
            ),
        )
        != ed_authority["sha256"]
        or int(same_cutoff_ed.get("nph", -1)) != int(cell.nph)
        or same_cutoff_ed.get("required") is not True
        or same_cutoff_ed.get("reference_role")
        != "same_cutoff_reporting_reference"
    ):
        raise BundleMaterializationError(
            f"Resolver trace same-cutoff ED reference drifted for "
            f"{cell.source_lock_id}."
        )

    expected_hashes = [
        trace.get("source_sha256_expected"),
        trace.get("source_sha256_actual"),
    ]
    valid_expected_hashes = [
        _require_sha256(value, label="resolver trace source SHA-256")
        for value in expected_hashes
        if value is not None
    ]
    if valid_expected_hashes and member_sha256 not in valid_expected_hashes:
        raise BundleMaterializationError(
            f"Verified archive member for {cell.source_lock_id} does not "
            "match the resolver source SHA-256."
        )
    if trace.get("source_sha256_match") is False:
        raise BundleMaterializationError(
            f"Resolver trace records a SHA-256 mismatch for "
            f"{cell.source_lock_id}."
        )
    normalized = json.loads(canonical_json_bytes(trace))
    bindings = _declare_protocol_field_bindings(normalized, cell=cell)
    normalized["normalized_protocol_used_field_paths"] = [
        binding["source_path"] for binding in bindings
    ]
    normalized["protocol_field_bindings"] = list(bindings)
    normalized["protocol_used_field_audit"] = (
        "declared_pending_protocol_resolution"
    )
    return json.loads(canonical_json_bytes(normalized))


def normalize_and_verify_source_locks(
    source_locks: Mapping[str, Any],
    *,
    cells: Sequence[BundleCellSpec],
    repo_root: Path,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Validate exact global, archive, member, and resolver source locks.

    ``verify_files=False`` is an audit-only mode.  It is serialized as
    unverified and prevents a bundle from receiving a ``passed``
    materialization status.
    """

    if not isinstance(source_locks, Mapping):
        raise BundleMaterializationError("source_locks must be a mapping.")
    supplied_schema = source_locks.get("schema", SOURCE_LOCK_SCHEMA)
    if supplied_schema != SOURCE_LOCK_SCHEMA:
        raise BundleMaterializationError("Unknown RA-ADAPT source-lock schema.")

    implementation_sources = _implementation_source_inventory(repo_root)
    supplied_implementation = source_locks.get("implementation_sources")
    if supplied_implementation is not None:
        if not isinstance(supplied_implementation, Mapping):
            raise BundleMaterializationError(
                "implementation_sources must be a digested mapping."
            )
        supplied_implementation_sha = _verify_digest(
            supplied_implementation,
            label="implementation_sources",
        )
        if supplied_implementation_sha != implementation_sources["sha256"]:
            raise BundleMaterializationError(
                "RA-ADAPT implementation source inventory drifted."
            )

    supplied_globals = source_locks.get("global_sources", {})
    if not isinstance(supplied_globals, Mapping):
        raise BundleMaterializationError("global_sources must be a mapping.")
    normalized_globals: dict[str, Any] = {}
    for role, authority in GLOBAL_SOURCE_LOCKS.items():
        supplied = supplied_globals.get(role, authority)
        if not isinstance(supplied, Mapping):
            raise BundleMaterializationError(
                f"Global source lock {role!r} must be a mapping."
            )
        path_text = str(supplied.get("path", ""))
        digest = _require_sha256(
            supplied.get("sha256"), label=f"global_sources.{role}.sha256"
        )
        if path_text != authority["path"] or digest != authority["sha256"]:
            raise BundleMaterializationError(
                f"Global source lock authority drifted for {role!r}."
            )
        verified = False
        if verify_files:
            path = _resolve_path(repo_root, path_text, label=role)
            if not path.is_file():
                raise BundleMaterializationError(
                    f"Required global source is missing: {path}."
                )
            actual = _hash_file(path)
            if actual != digest:
                raise BundleMaterializationError(
                    f"Global source SHA-256 drift for {path}."
                )
            verified = True
        normalized_globals[role] = {
            "path": path_text,
            "sha256": digest,
            "verified": verified,
        }

    raw_cells = source_locks.get("cell_locks")
    if not isinstance(raw_cells, Mapping):
        raise BundleMaterializationError("cell_locks must be a mapping.")
    required_ids = {cell.source_lock_id for cell in cells}
    missing_ids = sorted(required_ids.difference(raw_cells))
    if missing_ids:
        raise BundleMaterializationError(
            "Missing source locks for finite bundle cells: "
            + ", ".join(missing_ids)
        )

    archive_hash_cache: dict[Path, str] = {}
    member_hash_cache: dict[tuple[Path, str], str] = {}
    normalized_cells: dict[str, Any] = {}
    representative_by_lock = {
        cell.source_lock_id: cell for cell in reversed(tuple(cells))
    }
    for lock_id in sorted(required_ids):
        cell = representative_by_lock[lock_id]
        raw = raw_cells[lock_id]
        if not isinstance(raw, Mapping):
            raise BundleMaterializationError(
                f"Cell source lock {lock_id!r} must be a mapping."
            )
        if str(raw.get("regime_id")) != cell.regime_id:
            raise BundleMaterializationError(
                f"Source-lock regime drift for {lock_id}."
            )
        if int(raw.get("nph", -1)) != int(cell.nph):
            raise BundleMaterializationError(
                f"Source-lock cutoff drift for {lock_id}."
            )
        if str(raw.get("route_id")) != cell.route_id:
            raise BundleMaterializationError(
                f"Source-lock route drift for {lock_id}."
            )
        archive_lock = raw.get("archive")
        member_lock = raw.get("member")
        trace = raw.get("resolver_trace")
        if not isinstance(archive_lock, Mapping):
            raise BundleMaterializationError(
                f"Source lock {lock_id} has no typed archive lock."
            )
        if not isinstance(member_lock, Mapping):
            raise BundleMaterializationError(
                f"Source lock {lock_id} has no typed member lock."
            )
        if not isinstance(trace, Mapping):
            raise BundleMaterializationError(
                f"Source lock {lock_id} has no resolver trace."
            )

        archive_path_text = str(archive_lock.get("path", "")).strip()
        archive_sha = _require_sha256(
            archive_lock.get("sha256"),
            label=f"cell_locks.{lock_id}.archive.sha256",
        )
        member_path = str(member_lock.get("path", "")).strip()
        member_sha = _require_sha256(
            member_lock.get("sha256"),
            label=f"cell_locks.{lock_id}.member.sha256",
        )
        archive_verified = False
        member_verified = False
        if verify_files:
            archive_path = _resolve_path(
                repo_root, archive_path_text, label=f"{lock_id}.archive"
            )
            if not archive_path.is_file():
                raise BundleMaterializationError(
                    f"Source archive is missing: {archive_path}."
                )
            if archive_path not in archive_hash_cache:
                archive_hash_cache[archive_path] = _hash_file(archive_path)
            actual_archive = archive_hash_cache[archive_path]
            if actual_archive != archive_sha:
                raise BundleMaterializationError(
                    f"Source archive SHA-256 drift for {archive_path}."
                )
            member_key = (archive_path, member_path)
            if member_key not in member_hash_cache:
                member_hash_cache[member_key] = _hash_archive_member(
                    archive_path,
                    member_path,
                )
            if member_hash_cache[member_key] != member_sha:
                raise BundleMaterializationError(
                    f"Source archive member SHA-256 drift for "
                    f"{archive_path}:{member_path}."
                )
            archive_verified = True
            member_verified = True

        normalized_trace = _verify_resolver_trace(
            trace, cell=cell, member_sha256=member_sha
        )
        normalized = _digested(
            {
                "regime_id": cell.regime_id,
                "nph": int(cell.nph),
                "route_id": cell.route_id,
                "archive": {
                    "path": archive_path_text,
                    "sha256": archive_sha,
                },
                "member": {
                    "path": member_path,
                    "sha256": member_sha,
                },
                "resolver_trace": normalized_trace,
                "verification": {
                    "archive_sha256_verified": archive_verified,
                    "member_sha256_verified": member_verified,
                    "resolver_trace_compatible": True,
                    "verification_mode": (
                        "local_exact_bytes_v1"
                        if verify_files
                        else "record_only_unverified_v1"
                    ),
                },
            }
        )
        supplied_digest = raw.get("sha256")
        if supplied_digest is not None and _require_sha256(
            supplied_digest, label=f"cell_locks.{lock_id}.sha256"
        ) != normalized["sha256"]:
            raise BundleMaterializationError(
                f"Cell source-lock canonical digest drift for {lock_id}."
            )
        normalized_cells[lock_id] = normalized

    payload = _digested(
        {
            "schema": SOURCE_LOCK_SCHEMA,
            "global_sources": normalized_globals,
            "implementation_sources": implementation_sources,
            "cell_locks": normalized_cells,
            "required_cell_lock_count": len(required_ids),
            "all_required_files_verified": bool(verify_files),
        }
    )
    supplied_digest = source_locks.get("sha256")
    if supplied_digest is not None and _require_sha256(
        supplied_digest, label="source_locks.sha256"
    ) != payload["sha256"]:
        raise BundleMaterializationError(
            "Source-lock manifest canonical digest drift."
        )
    return payload


def _bind_core_selection_authority(
    normalized_source_locks: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    authority_path = repo_root / CORE_SELECTION_AUTHORITY_PATH
    if not authority_path.is_file():
        raise BundleMaterializationError(
            "The stationary-core user-selection authority is missing."
        )
    observed = _hash_file(authority_path)
    if observed != CORE_SELECTION_AUTHORITY_SHA256:
        raise BundleMaterializationError(
            "The stationary-core user-selection authority SHA-256 drifted."
        )
    payload = dict(normalized_source_locks)
    payload.pop("sha256", None)
    payload["campaign_authorities"] = {
        "stationarity_selection": {
            "path": CORE_SELECTION_AUTHORITY_PATH,
            "sha256": CORE_SELECTION_AUTHORITY_SHA256,
            "verified": True,
        }
    }
    return _digested(payload)


def _default_environment_fingerprint() -> dict[str, Any]:
    payload = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
    }
    return _digested(
        {
            "schema": "ra_adapt_environment_fingerprint_v1",
            **payload,
        }
    )


def _dependency_lock_provenance(
    repo_root: Path,
    dependency_lock_paths: Sequence[str | Path] | None,
) -> dict[str, Any]:
    if dependency_lock_paths is None:
        requirements = repo_root / "requirements.txt"
        if requirements.is_file():
            lines = [
                line.strip()
                for line in requirements.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            dependency_lock_paths = (
                ("requirements.txt",)
                if lines and all("==" in line for line in lines)
                else ()
            )
        else:
            dependency_lock_paths = ()

    locks: list[dict[str, str]] = []
    for value in dependency_lock_paths:
        path = _resolve_path(repo_root, value, label="dependency lock")
        if not path.is_file():
            raise BundleMaterializationError(
                f"Explicit dependency lock is missing: {path}."
            )
        serialized = (
            str(path.relative_to(repo_root))
            if path.is_relative_to(repo_root)
            else str(path)
        )
        locks.append({"path": serialized, "sha256": _hash_file(path)})
    locks.sort(key=lambda row: row["path"])
    if not locks:
        return {
            "dependency_lock_sha256": None,
            "dependency_lock_status": (
                "missing_no_pinned_dependency_lock_detected"
            ),
            "dependency_locks": [],
            "dependency_lock_digest_convention": None,
        }
    digest = (
        locks[0]["sha256"]
        if len(locks) == 1
        else canonical_sha256(locks)
    )
    return {
        "dependency_lock_sha256": digest,
        "dependency_lock_status": "verified",
        "dependency_locks": locks,
        "dependency_lock_digest_convention": (
            "single_file_sha256_v1"
            if len(locks) == 1
            else "canonical_lock_list_sha256_v1"
        ),
    }


def _repository_state(
    repo_root: Path, supplied: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(supplied, Mapping):
        raise BundleMaterializationError(
            "repository_state must explicitly record git_commit and "
            "dirty_working_tree."
        )
    commit = str(supplied.get("git_commit", "")).strip().lower()
    if len(commit) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise BundleMaterializationError(
            "repository_state.git_commit must be a Git object digest."
        )
    dirty = supplied.get("dirty_working_tree")
    if not isinstance(dirty, bool):
        raise BundleMaterializationError(
            "repository_state.dirty_working_tree must be an observed boolean."
        )
    cwd = str(supplied.get("cwd", repo_root))
    return {
        "git_commit": commit,
        "dirty_working_tree": dirty,
        "cwd": cwd,
    }


def _artifact_paths(cell: BundleCellSpec) -> dict[str, str]:
    root = PurePosixPath("runs") / cell.cell_id
    return {
        "execution_manifest": str(root / "execution_manifest.json"),
        "checkpoint": str(root / "checkpoints" / "current.json"),
        "estimator_ledger": str(root / "result" / "estimator_ledger.json"),
        "result": str(root / "result" / "result.json"),
        "summary": str(root / "summary" / "summary.json"),
    }


def _build_request(
    cell: BundleCellSpec,
    *,
    bundle_dir: Path,
) -> RAAdaptRequest | AppendAdaptRequest:
    if cell.horizon is None:
        raise BundleMaterializationError(
            f"Cannot build a request for blocked cell {cell.cell_id}."
        )
    paths = _artifact_paths(cell)
    execution = SRExecutionPolicy(
        stop=SRStopPolicy(maximum_controller_rounds=int(cell.horizon))
    )
    observation = SRObservationPolicy(
        checkpoint=CheckpointObservation(
            path=Path(paths["checkpoint"]),
            every_controller_rounds=1,
            keep_history_tail=100,
        ),
        estimator_ledger=EstimatorLedgerObservation(
            path=Path(paths["estimator_ledger"])
        ),
    )
    adapter: (
        MacroCandidateAdapter
        | SinglePauliWordCandidateAdapter
        | GlobalSinglePauliWordCandidateAdapter
    )
    if cell.route_id in GLOBAL_SINGLETON_INSERTION_ROUTE_IDS:
        adapter = GlobalSinglePauliWordCandidateAdapter()
    elif (
        cell.candidate_representation
        == CANDIDATE_REPRESENTATION_MACRO
    ):
        adapter = MacroCandidateAdapter()
    else:
        adapter = SinglePauliWordCandidateAdapter()
    if cell.selector_family == "append_adapt":
        return AppendAdaptRequest(
            adapter=adapter,
            execution=execution,
            observation=observation,
        )
    if cell.route_id in {
        ROUTE_RA_MACRO_APPEND_ONLY,
        ROUTE_RA_SINGLETON_APPEND_ONLY,
    }:
        insertion = AppendOnlyInsertion()
    elif cell.route_id in {
        ROUTE_RA_MACRO_PLATEAU,
        ROUTE_SINGLETON_PLATEAU,
        ROUTE_RA_SINGLETON_PLATEAU,
    }:
        insertion = PlateauCommutationInsertion()
    elif cell.route_id in {
        ROUTE_RA_MACRO_ALWAYS,
        ROUTE_RA_SINGLETON_ALWAYS,
    }:
        insertion = AlwaysCommutationReducedInsertion()
    elif cell.route_id == ROUTE_RA_GLOBAL_SINGLETON_APPEND_REDUCED:
        insertion = AppendCommutationReducedInsertion()
    elif cell.route_id == ROUTE_RA_GLOBAL_SINGLETON_PLATEAU:
        insertion = PlateauCommutationInsertion()
    else:
        raise BundleMaterializationError(
            f"Unknown RA insertion route: {cell.route_id!r}."
        )
    beam = BeamOff()
    pruning = PruningOff()
    if cell.cell_id.startswith("beamprune__"):
        # Crossed beam+prune lane ablation: beam is fixed 3x2 fork-local and
        # the prune family is carried by the arm token in the cell id.
        beam = ForkLocalBeam(
            live_parent_branches=3,
            admission_children_per_parent=2,
            maximum_admission_children_per_round=6,
            s_alg_weight=0.005,
        )
        arm = cell.cell_id.split("__")[1]
        pruning = (
            MetricPruning() if arm.endswith("_metric") else TrustRegionPruning()
        )
    return RAAdaptRequest(
        adapter=adapter,
        method=SRMethodPolicy(
            insertion=insertion, beam=beam, pruning=pruning
        ),
        execution=execution,
        observation=observation,
    )


def _default_protocol_resolver(context: ProtocolResolutionContext) -> Any:
    if context.cell.selector_family == "ra_adapt":
        if not isinstance(context.request, RAAdaptRequest):
            raise TypeError("RA cell lost its RAAdaptRequest.")
        return build_resolved_ra_protocol(
            context.problem,
            context.request,
            materialization_authority=(
                context.materialization_authority
            ),
        )
    if not isinstance(context.request, AppendAdaptRequest):
        raise TypeError("Append cell lost its AppendAdaptRequest.")
    try:
        from pipelines.static_adapt.ra_adapt.append import (
            build_resolved_append_protocol,
        )
    except ImportError as exc:
        raise BundleMaterializationError(
            "Append protocol materialization requires "
            "ra_adapt.append.build_resolved_append_protocol."
        ) from exc
    return build_resolved_append_protocol(
        context.problem,
        context.request,
        materialization_authority=context.materialization_authority,
    )


def _as_protocol_payload(value: Any, *, cell: BundleCellSpec) -> dict[str, Any]:
    if isinstance(value, Mapping):
        payload = dict(value)
    elif callable(getattr(value, "to_dict", None)):
        payload = value.to_dict()
    else:
        raise BundleMaterializationError(
            f"Protocol resolver returned an unsupported value for "
            f"{cell.cell_id}."
        )
    # Round-trip through the canonical encoder to reject NaN and remove
    # non-JSON container types before validation/writing.
    payload = json.loads(canonical_json_bytes(payload))
    _verify_digest(payload, label=f"protocol {cell.cell_id}")
    return payload


def _route_contract_for_request(
    *,
    cell: BundleCellSpec,
    request: RAAdaptRequest | AppendAdaptRequest,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(request, RAAdaptRequest):
        _requested, _resolved, contract, engine_digest = (
            _repaired_route_contract(
                request,
                active_gradient_policy=str(
                    protocol["active_gradient_policy"]
                ),
                resource_weighting_scope=str(
                    protocol["resource_weighting_scope"]
                ),
                algorithm_id=cell.algorithm_id,
            )
        )
        if engine_digest != canonical_sha256(contract):
            raise BundleMaterializationError(
                f"Engine route-contract digest drifted for {cell.cell_id}."
            )
        route = _digested(contract)
        parent = route.get("lineage_authority")
        protocol_lineage = protocol.get("lineage_authority")
        if (
            not isinstance(parent, Mapping)
            or not isinstance(protocol_lineage, Mapping)
            or parent.get("parent_contract_sha256")
            != protocol_lineage.get("parent_contract_sha256")
        ):
            raise BundleMaterializationError(
                f"Route lineage drifted for {cell.cell_id}."
            )
        return route

    if not isinstance(request, AppendAdaptRequest):
        raise TypeError("Unknown typed request while resolving route contract.")
    return _digested(
        {
            "schema": "paper_i_append_adapt_route_contract_v1",
            "route_family": "append_adapt",
            "route_profile": (
                "paper_i_append_adapt__"
                f"{cell.candidate_representation}__"
                f"{protocol['active_gradient_policy']}__"
                f"{protocol['resource_weighting_scope']}"
            ),
            "execution_settings": {
                "optimizer": protocol["optimizer"],
                "optimizer_maxiter": protocol["optimizer_maxiter"],
                "maximum_controller_rounds": protocol["horizon"],
                "adapt_seed": protocol["seeds"]["adapt"],
                "phase3_backend_transpile_seed": (
                    protocol["seeds"]["transpiler"]
                ),
                "adapt_insertion_mode": "append_only",
            },
            "semantic_invariants": {
                "canonical_interface": (
                    "run_append_adapt_problem_request_v1"
                ),
                "candidate_representation": (
                    cell.candidate_representation
                ),
                "selector_identity": APPEND_CONVENTIONAL_SELECTOR_ID,
                "selector_scope": APPEND_CONVENTIONAL_SELECTOR_SCOPE,
                "selector_rule": (
                    "largest_absolute_commutator_gradient_full_pool_v1"
                ),
                "selection_with_replacement": True,
                "insertion_position": "append_only_v1",
                "ra_staged_funnel_invoked": False,
                "candidate_geometry_chart": EXACT_ORDERED_INSERTION_CHART,
                "accepted_refit_scope": FULL_ENLARGED_ACCEPTED_REFIT,
                "accepted_refit_coordinate_chart": NATIVE_REFIT_CHART,
                "accepted_refit_base_chart_policy": (
                    LOGICAL_SHARED_REDUCED_REFIT_BASE_CHART
                ),
                "active_gradient_policy": protocol[
                    "active_gradient_policy"
                ],
                "resource_weighting_scope": protocol[
                    "resource_weighting_scope"
                ],
            },
            "lineage_authority": dict(protocol["lineage_authority"]),
        }
    )


def _nested_value(payload: Mapping[str, Any], path: str) -> Any:
    value: Any = payload
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise BundleMaterializationError(
                f"Required field path {path!r} was not resolved."
            )
        value = value[component]
    return value


def _optimizer_family(value: Any) -> str:
    normalized = str(value).strip().lower()
    if "powell" in normalized:
        return "powell"
    return normalized


def _baseline_consumption_receipt(
    *,
    cell: BundleCellSpec,
    protocol: Mapping[str, Any],
    cell_source_lock: Mapping[str, Any],
) -> dict[str, Any]:
    trace = cell_source_lock.get("resolver_trace")
    if not isinstance(trace, Mapping):
        raise BundleMaterializationError(
            f"Source lock has no resolver trace for {cell.cell_id}."
        )
    bindings = trace.get("protocol_field_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise BundleMaterializationError(
            f"Source lock has no exact protocol bindings for {cell.cell_id}."
        )
    changes = trace.get("settings_changed", ())
    change_ids = sorted(
        {
            str(change.get("id"))
            for change in changes
            if isinstance(change, Mapping)
        }
    )
    unexpected = sorted(set(change_ids).difference(_SETTLED_CHANGE_IDS))
    if unexpected:
        raise BundleMaterializationError(
            f"Protocol {cell.cell_id} has unapproved source deltas "
            f"{unexpected}."
        )
    verified_bindings: list[dict[str, Any]] = []
    for raw_binding in bindings:
        if not isinstance(raw_binding, Mapping):
            raise BundleMaterializationError(
                f"Malformed protocol binding for {cell.cell_id}."
            )
        source_path = str(raw_binding.get("source_path", ""))
        protocol_path = str(raw_binding.get("protocol_path", ""))
        comparison = str(raw_binding.get("comparison", ""))
        source_value = _nested_value(trace, source_path)
        protocol_value = _nested_value(protocol, protocol_path)
        matched = (
            _optimizer_family(source_value)
            == _optimizer_family(protocol_value)
            if comparison == "optimizer_family"
            else source_value == protocol_value
        )
        disposition = "exact_reuse"
        delta_ids = tuple(raw_binding.get("authorized_delta_ids", ()))
        if not matched:
            if (
                comparison != "exact_or_cutoff_axis"
                or not delta_ids
                or any(
                    delta_id
                    not in {
                        "study_authorized_cutoff_change",
                        "approved_validation_cutoff_override",
                    }
                    or delta_id not in change_ids
                    for delta_id in delta_ids
                )
                or int(protocol_value) != int(cell.nph)
            ):
                raise BundleMaterializationError(
                    f"Source baseline field drifted for {cell.cell_id}: "
                    f"{source_path} -> {protocol_path}."
                )
            disposition = "settled_cutoff_axis_delta"
        verified_bindings.append(
            {
                "source_path": source_path,
                "protocol_path": protocol_path,
                "comparison": comparison,
                "source_value": source_value,
                "protocol_value": protocol_value,
                "disposition": disposition,
                **(
                    {"authorized_delta_ids": list(delta_ids)}
                    if delta_ids
                    else {}
                ),
            }
        )
    declared = list(trace.get("normalized_protocol_used_field_paths", ()))
    consumed = [row["source_path"] for row in verified_bindings]
    if declared != consumed or len(consumed) != len(set(consumed)):
        raise BundleMaterializationError(
            f"Protocol baseline consumption is incomplete for {cell.cell_id}."
        )
    return _digested(
        {
            "schema": "ra_adapt_protocol_baseline_consumption_v1",
            "source_lock_id": cell.source_lock_id,
            "source_lock_sha256": cell_source_lock["sha256"],
            "status": "passed",
            "declared_source_field_paths": declared,
            "consumed_source_field_paths": consumed,
            "bindings": verified_bindings,
            "settled_change_ids": change_ids,
            "unconsumed_declared_field_paths": [],
            "unapproved_change_ids": [],
        }
    )


def _decorate_protocol_payload(
    payload: Mapping[str, Any],
    *,
    cell: BundleCellSpec,
    request: RAAdaptRequest | AppendAdaptRequest,
    cell_source_lock: Mapping[str, Any],
    materialization_authority: BundleProtocolMaterializationAuthority,
) -> dict[str, Any]:
    result = dict(payload)
    serialized_materialization = (
        materialization_authority.receipt.to_dict()
    )
    existing_materialization = result.get("bundle_materialization")
    if (
        existing_materialization is not None
        and existing_materialization != serialized_materialization
    ):
        raise BundleMaterializationError(
            f"Protocol materialization authority drifted for {cell.cell_id}."
        )
    result["bundle_materialization"] = serialized_materialization
    for name in ("parent_inventory", "executable_pool"):
        pool = result.get(name)
        if not isinstance(pool, Mapping):
            raise BundleMaterializationError(
                f"Protocol {cell.cell_id} has no {name}."
            )
        if pool.get("sha256") is None:
            result[name] = _digested(pool)
        else:
            _verify_digest(pool, label=f"protocol {cell.cell_id}.{name}")
            result[name] = dict(pool)
    result["route_contract"] = _route_contract_for_request(
        cell=cell,
        request=request,
        protocol=result,
    )
    _verify_digest(
        result["route_contract"],
        label=f"protocol {cell.cell_id}.route_contract",
    )
    result["baseline_consumption"] = _baseline_consumption_receipt(
        cell=cell,
        protocol=result,
        cell_source_lock=cell_source_lock,
    )
    return _digested(result)


def _pool_count(payload: Mapping[str, Any], name: str) -> int:
    pool = payload.get(name)
    if not isinstance(pool, Mapping):
        raise BundleMaterializationError(
            f"Resolved protocol has no {name} receipt."
        )
    return int(pool.get("count", -1))


def _validate_protocol_payload(
    payload: Mapping[str, Any],
    *,
    cell: BundleCellSpec,
    bundle_id: str,
    bundle_manifest_sha256: str,
    active_gradient_policy: str,
    resource_weighting_scope: str,
    source_lock_refs: Mapping[str, str],
    cell_source_lock: Mapping[str, Any],
    source_locks_sha256: str,
) -> None:
    _verify_digest(payload, label=f"protocol {cell.cell_id}")
    expected_schema = _protocol_schema_for_cell(cell)
    expected_selector = (
        APPEND_CONVENTIONAL_SELECTOR_ID
        if cell.selector_family == "append_adapt"
        else RA_STAGED_SELECTOR_ID
    )
    checks = {
        "schema": expected_schema,
        "algorithm_id": cell.algorithm_id,
        "candidate_representation": cell.candidate_representation,
        "selector_identity": expected_selector,
        "active_gradient_policy": active_gradient_policy,
        "resource_weighting_scope": resource_weighting_scope,
        "derivative_chart_id": EXACT_ORDERED_INSERTION_CHART,
        "trust_policy_id": SOURCE_GRAM_NO_OVERLAP_TRUST,
        "phase3_solver_id": PROJECTED_GENERALIZED_SOLVER,
        "phase3_multiplier_contract": PhaseIIIMultiplierContract().to_dict(),
        "accepted_refit_scope": FULL_ENLARGED_ACCEPTED_REFIT,
        "accepted_refit_coordinate_chart": (
            NATIVE_REFIT_CHART
            if cell.selector_family == "append_adapt"
            else SUPPORTED_FS_WHITENED_REFIT_CHART
        ),
        "accepted_refit_base_chart_policy": (
            LOGICAL_SHARED_REDUCED_REFIT_BASE_CHART
            if cell.selector_family == "append_adapt"
            else EXPANDED_RUNTIME_PROJECTED_LOGICAL_BASE_CHART
        ),
        "optimizer": "powell",
        "optimizer_maxiter": 200,
        "horizon": int(cell.horizon or 0),
        "estimator_accounting_convention": RA_ADAPT_ESTIMATOR_ACCOUNTING,
        "compile_identity": dict(RA_ADAPT_COMPILE_IDENTITY),
        "bundle_id": bundle_id,
        "bundle_manifest_sha256": bundle_manifest_sha256,
        "execution_authorized": False,
    }
    if cell.selector_family == "append_adapt":
        checks["selector_scope"] = APPEND_CONVENTIONAL_SELECTOR_SCOPE
    elif payload.get("selector_scope") is not None:
        raise BundleMaterializationError(
            f"RA protocol unexpectedly carries selector_scope for "
            f"{cell.cell_id}."
        )
    for name, expected in checks.items():
        if payload.get(name) != expected:
            raise BundleMaterializationError(
                f"Protocol field drift for {cell.cell_id}.{name}: "
                f"{payload.get(name)!r} != {expected!r}."
            )
    protocol_source_locks = payload.get("source_locks")
    if not isinstance(protocol_source_locks, Mapping):
        raise BundleMaterializationError(
            f"Protocol source locks are missing for {cell.cell_id}."
        )
    for name, expected in source_lock_refs.items():
        if protocol_source_locks.get(name) != expected:
            raise BundleMaterializationError(
                f"Protocol source-lock drift for {cell.cell_id}.{name}."
            )
    for name, digest in protocol_source_locks.items():
        if name == "cell_source_lock_id":
            if not str(digest).strip():
                raise BundleMaterializationError(
                    f"Protocol source-lock id is empty for {cell.cell_id}."
                )
        else:
            _require_sha256(
                digest,
                label=f"protocol {cell.cell_id}.source_locks.{name}",
            )
    expected_materialization = (
        bundle_protocol_materialization_receipt(
            bundle_id=bundle_id,
            bundle_manifest_sha256=bundle_manifest_sha256,
            source_locks_sha256=source_locks_sha256,
            source_lock_refs=source_lock_refs,
            cell_id=cell.cell_id,
            source_lock_id=cell.source_lock_id,
            protocol_schema=expected_schema,
            algorithm_id=cell.algorithm_id,
            candidate_representation=cell.candidate_representation,
            selector_identity=expected_selector,
            active_gradient_policy=active_gradient_policy,
            resource_weighting_scope=resource_weighting_scope,
        ).to_dict()
    )
    if payload.get("bundle_materialization") != expected_materialization:
        raise BundleMaterializationError(
            f"Protocol bundle materialization drifted for {cell.cell_id}."
        )
    for pool_name in ("parent_inventory", "executable_pool"):
        pool = payload.get(pool_name)
        if not isinstance(pool, Mapping):
            raise BundleMaterializationError(
                f"Protocol pool is missing for {cell.cell_id}.{pool_name}."
            )
        _verify_digest(
            pool, label=f"protocol {cell.cell_id}.{pool_name}"
        )
        labels = pool.get("ordered_labels")
        if (
            not isinstance(labels, list)
            or int(pool.get("count", -1)) != len(labels)
            or pool.get("ordered_labels_sha256")
            != canonical_sha256(labels)
        ):
            raise BundleMaterializationError(
                f"Protocol pool label inventory drifted for "
                f"{cell.cell_id}.{pool_name}."
            )
    route_contract = payload.get("route_contract")
    if not isinstance(route_contract, Mapping):
        raise BundleMaterializationError(
            f"Protocol route contract is missing for {cell.cell_id}."
        )
    _verify_digest(
        route_contract,
        label=f"protocol {cell.cell_id}.route_contract",
    )
    route_invariants = route_contract.get("semantic_invariants")
    route_execution = route_contract.get("execution_settings")
    if (
        not isinstance(route_invariants, Mapping)
        or not isinstance(route_execution, Mapping)
        or route_invariants.get("candidate_representation")
        != cell.candidate_representation
        or route_invariants.get("selector_identity") != expected_selector
        or (
            cell.selector_family == "append_adapt"
            and route_invariants.get("selector_scope")
            != APPEND_CONVENTIONAL_SELECTOR_SCOPE
        )
        or route_invariants.get("candidate_geometry_chart")
        != EXACT_ORDERED_INSERTION_CHART
        or route_invariants.get("active_gradient_policy")
        != active_gradient_policy
        or route_invariants.get("resource_weighting_scope")
        != resource_weighting_scope
    ):
        raise BundleMaterializationError(
            f"Protocol route contract drifted for {cell.cell_id}."
        )
    if cell.route_id in {
        ROUTE_RA_MACRO_ALWAYS,
        ROUTE_RA_SINGLETON_ALWAYS,
    } and (
        route_execution.get("adapt_insertion_mode")
        != ALWAYS_REDUCED_INSERTION_MODE
        or route_invariants.get("insertion_position_scope")
        != ALWAYS_REDUCED_INSERTION_SCOPE
        or route_invariants.get("insertion_equivalence_policy")
        != ALWAYS_REDUCED_INSERTION_EQUIVALENCE
    ):
        raise BundleMaterializationError(
            "Always-insertion protocol is not bound to the full logical "
            f"commutation-reduced domain for {cell.cell_id}."
        )
    if cell.route_id == ROUTE_RA_GLOBAL_SINGLETON_APPEND_REDUCED and (
        route_execution.get("adapt_insertion_mode")
        != AppendCommutationReducedInsertion.runtime_mode
        or route_invariants.get("insertion_position_scope")
        != AppendCommutationReducedInsertion.position_scope
        or route_invariants.get("insertion_equivalence_policy")
        != AppendCommutationReducedInsertion.equivalence_policy
    ):
        raise BundleMaterializationError(
            "Global-singleton append arm is not bound to the "
            f"commutation-reduced endpoint for {cell.cell_id}."
        )
    if cell.route_id == ROUTE_RA_GLOBAL_SINGLETON_PLATEAU and (
        route_execution.get("adapt_insertion_mode")
        != "insertion_commutation_plateau_v2"
        or route_invariants.get("insertion_position_scope")
        != "append_only_or_immediate_plateau_full_logical_domain_v1"
        or route_invariants.get("insertion_equivalence_policy")
        != AppendCommutationReducedInsertion.equivalence_policy
        or route_invariants.get(
            "plateau_prior_mean_decrease_ratio_threshold"
        )
        != 1e-4
        or route_invariants.get("plateau_threshold_comparison")
        != "marginal_to_prior_mean_strictly_below_v2"
        or route_invariants.get("plateau_patience") != 1
        or route_invariants.get("plateau_hysteresis_active") is not False
    ):
        raise BundleMaterializationError(
            "Global-singleton plateau arm drifted from the fixed shared "
            f"commutation reducer and trigger for {cell.cell_id}."
        )
    if cell.route_id in GLOBAL_SINGLETON_INSERTION_ROUTE_IDS:
        expected_supply = {
            "candidate_adapter_id": GLOBAL_SINGLE_PAULI_ADAPTER_ID,
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
        if (
            payload.get("adapter_id") != GLOBAL_SINGLE_PAULI_ADAPTER_ID
            or any(
                route_invariants.get(name) != value
                for name, value in expected_supply.items()
            )
            or route_execution.get("phase1_shortlist_size") != 24
            or route_execution.get("phase2_shortlist_size") != 12
            or route_invariants.get("admission_cardinality") != 1
            or route_execution.get("phase1_prune_enabled") is not False
            or route_execution.get("phase2_enable_batching") is not False
            or route_execution.get("phase3_enable_batching") is not False
        ):
            raise BundleMaterializationError(
                "Global-singleton Phase-I/II/III funnel identity drifted "
                f"for {cell.cell_id}."
            )
    if (
        cell.selector_family == "append_adapt"
        and (
            route_invariants.get("accepted_refit_coordinate_chart")
            != NATIVE_REFIT_CHART
            or route_invariants.get("accepted_refit_base_chart_policy")
            != LOGICAL_SHARED_REDUCED_REFIT_BASE_CHART
        )
    ):
        raise BundleMaterializationError(
            f"Append route refit convention drifted for {cell.cell_id}."
        )
    baseline = payload.get("baseline_consumption")
    if not isinstance(baseline, Mapping):
        raise BundleMaterializationError(
            f"Protocol baseline receipt is missing for {cell.cell_id}."
        )
    _verify_digest(
        baseline,
        label=f"protocol {cell.cell_id}.baseline_consumption",
    )
    if (
        baseline.get("status") != "passed"
        or baseline.get("source_lock_id") != cell.source_lock_id
        or baseline.get("source_lock_sha256")
        != cell_source_lock.get("sha256")
        or baseline.get("unconsumed_declared_field_paths") != []
        or baseline.get("unapproved_change_ids") != []
    ):
        raise BundleMaterializationError(
            f"Protocol baseline consumption failed for {cell.cell_id}."
        )
    request = payload.get("request")
    if not isinstance(request, Mapping):
        raise BundleMaterializationError(
            f"Protocol request is missing for {cell.cell_id}."
        )
    expected_request_kind = (
        AppendAdaptRequest.kind
        if cell.selector_family == "append_adapt"
        else RAAdaptRequest.kind
    )
    if request.get("kind") != expected_request_kind:
        raise BundleMaterializationError(
            f"Protocol request discriminator drifted for {cell.cell_id}."
        )
    if cell.selector_family == "ra_adapt":
        method = request.get("method")
        if not isinstance(method, Mapping) or any(
            not isinstance(method.get(name), Mapping)
            or not str(method[name].get("kind", "")).strip()
            for name in ("admission", "insertion", "pruning", "beam")
        ):
            raise BundleMaterializationError(
                f"RA policy discriminator is missing for {cell.cell_id}."
            )
        expected_insertion_kind = _RA_INSERTION_KIND_BY_ROUTE.get(
            cell.route_id
        )
        if (
            expected_insertion_kind is None
            or method["insertion"].get("kind")
            != expected_insertion_kind
        ):
            raise BundleMaterializationError(
                "RA insertion policy does not match the materialized route "
                f"for {cell.cell_id}: "
                f"{method['insertion'].get('kind')!r} != "
                f"{expected_insertion_kind!r}."
            )
        request_adapter = request.get("adapter")
        if cell.route_id in GLOBAL_SINGLETON_INSERTION_ROUTE_IDS and (
            not isinstance(request_adapter, Mapping)
            or request_adapter.get("adapter_id")
            != GLOBAL_SINGLE_PAULI_ADAPTER_ID
            or request_adapter.get("candidate_representation_id")
            != CANDIDATE_REPRESENTATION_SINGLE_PAULI
        ):
            raise BundleMaterializationError(
                "Global-singleton request adapter discriminator drifted "
                f"for {cell.cell_id}."
            )
    seeds = payload.get("seeds")
    if not isinstance(seeds, Mapping) or (
        int(seeds.get("adapt", -1)) != 7
        or int(seeds.get("transpiler", -1)) != 7
    ):
        raise BundleMaterializationError(
            f"Protocol seeds drifted for {cell.cell_id}."
        )
    stopping = payload.get("stopping_rule")
    if not isinstance(stopping, Mapping) or int(
        stopping.get("maximum_controller_rounds", -1)
    ) != int(cell.horizon or 0):
        raise BundleMaterializationError(
            f"Protocol stopping horizon drifted for {cell.cell_id}."
        )
    problem = payload.get("problem")
    if not isinstance(problem, Mapping):
        raise BundleMaterializationError(
            f"Resolved problem receipt is missing for {cell.cell_id}."
        )
    if (
        str(problem.get("family_key", "")).strip().lower() != "hh"
        or int(problem.get("num_sites", -1)) != 2
        or int(problem.get("n_ph_max", -1)) != int(cell.nph)
        or not str(problem.get("reference_label", "")).strip()
    ):
        raise BundleMaterializationError(
            f"Resolved problem receipt drifted for {cell.cell_id}."
        )
    parent_expected = 123 if int(cell.nph) == 3 else 171
    if _pool_count(payload, "parent_inventory") != parent_expected:
        raise BundleMaterializationError(
            f"Parent inventory count drifted for {cell.cell_id}."
        )
    if cell.candidate_representation == CANDIDATE_REPRESENTATION_MACRO:
        executable_expected = 102 if int(cell.nph) == 3 else 148
        if _pool_count(payload, "executable_pool") != executable_expected:
            raise BundleMaterializationError(
                f"Executable macro pool count drifted for {cell.cell_id}."
            )


def _blocked_protocol(
    *,
    cell: BundleCellSpec,
    bundle_id: str,
    bundle_manifest_sha256: str,
    active_gradient_policy: str,
    resource_weighting_scope: str,
    source_lock_refs: Mapping[str, str],
) -> dict[str, Any]:
    return _digested(
        {
            "schema": BLOCKED_PROTOCOL_SCHEMA,
            "cell_id": cell.cell_id,
            "status": "blocked",
            "blocking_reason": (
                "validation_horizon_not_supplied_or_validated"
            ),
            "bundle_id": bundle_id,
            "bundle_manifest_sha256": bundle_manifest_sha256,
            "algorithm_id": cell.algorithm_id,
            "candidate_representation": cell.candidate_representation,
            "selector_family": cell.selector_family,
            "active_gradient_policy": active_gradient_policy,
            "resource_weighting_scope": resource_weighting_scope,
            "source_locks": dict(source_lock_refs),
            "execution_authorized": False,
            "submission_state": SUBMISSION_STATE,
            "submitted": False,
        }
    )


def _manifest_payload(
    *,
    bundle_id: str,
    active_gradient_policy: str,
    resource_weighting_scope: str,
    cells: Sequence[BundleCellSpec],
    source_locks_sha256: str,
    environment_fingerprint: Mapping[str, Any],
    dependency_provenance: Mapping[str, Any],
    repository_state: Mapping[str, Any],
    materialization_timestamp: str | None,
    campaign_id: str = STUDY_ID,
    numerical_runtime_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if campaign_id == STUDY_ID:
        if resource_weighting_scope != RESOURCE_WEIGHTING_LATE:
            raise BundleMaterializationError(
                "Study 1 remains fixed to late resource weighting."
            )
        campaign_surface = {
            "study_id": STUDY_ID,
            "campaign_id": STUDY_ID,
            "study_stage": 1,
            "run_class": RUN_CLASS,
            "visible_target": {
                "target_id": VISIBLE_TARGET_ID,
                "paper": "Paper I",
                "source_lock_role": "macro_visible_provenance",
                "source_lock_pointer": (
                    "source_locks.json"
                    "#/global_sources/macro_visible_provenance"
                ),
            },
            "stationarity_winner_selected": False,
            "study_2_included": False,
            "validation_cell_count": sum(
                cell.stage == "validation" for cell in cells
            ),
            "full_cell_count": sum(cell.stage == "full" for cell in cells),
            "study1_shared_execution_dedupe": (
                study1_shared_execution_dedupe_contract()
            ),
            "execution_progression_contract": {
                "order": "materialization_then_validation_then_full_v1",
                "full_matrix_requires_validation_status": (
                    "validation_passed"
                ),
                "required_objective_gate_ids": list(
                    OBJECTIVE_EXECUTION_GATE_IDS
                ),
                "occurrence_minimums": {
                    "interior_insertion_observed": 1,
                    "trust_contraction_observed": 1,
                },
                "fail_closed": True,
            },
            "post_study_1_user_decision_required": True,
        }
    elif campaign_id == CORE_CAMPAIGN_ID:
        if (
            bundle_id != CORE_BUNDLE_ID
            or active_gradient_policy != ACTIVE_GRADIENT_STATIONARY
            or resource_weighting_scope != RESOURCE_WEIGHTING_LATE
            or tuple(cells) != build_core_cell_specs()
        ):
            raise BundleMaterializationError(
                "The stationary core campaign surface is not the exact "
                "selected 48-cell contract."
            )
        campaign_surface = {
            "study_id": CORE_CAMPAIGN_ID,
            "campaign_id": CORE_CAMPAIGN_ID,
            "study_stage": "stationarity_winner_selected_core_v1",
            "run_class": CORE_RUN_CLASS,
            "visible_target": {
                "target_id": CORE_VISIBLE_TARGET_ID,
                "paper": "Paper I",
                "source_lock_role": "per_cell_exact_source_locks",
                "source_lock_pointer": "source_locks.json#/cell_locks",
            },
            "stationarity_winner_selected": True,
            "stationarity_selection": {
                "selected_policy": ACTIVE_GRADIENT_STATIONARY,
                "selection_status": "user_selected_final_v1",
                "resource_weighting_scope": RESOURCE_WEIGHTING_LATE,
                "authority": {
                    "path": CORE_SELECTION_AUTHORITY_PATH,
                    "sha256": CORE_SELECTION_AUTHORITY_SHA256,
                    "source_lock_pointer": (
                        "source_locks.json#/campaign_authorities/"
                        "stationarity_selection"
                    ),
                },
            },
            "core_cell_count": len(cells),
            "core_matrix_contract": {
                "schema": "paper_i_ra_adapt_stationary_core_matrix_v1",
                "regime_cutoff_pairs": [
                    {"regime_id": regime_id, "nph": nph}
                    for regime_id, nph in CLAIM_FACING_REGIME_CUTOFF_PAIRS
                ],
                "candidate_representations": [
                    CANDIDATE_REPRESENTATION_MACRO,
                    CANDIDATE_REPRESENTATION_SINGLE_PAULI,
                ],
                "macro_route_ids": list(MACRO_ROUTE_IDS),
                "singleton_route_ids": list(SINGLETON_CORE_ROUTE_IDS),
                "horizon": FULL_HORIZON,
                "direct_execution_cell_count": len(cells),
            },
        }
    elif campaign_id == FACTORIAL_CAMPAIGN_ID:
        expected_gradient, expected_resource = (
            _factorial_policy_for_bundle(bundle_id)
        )
        expected_cells = build_factorial_always_cell_specs(
            active_gradient_policy=expected_gradient,
            resource_weighting_scope=expected_resource,
        )
        if (
            active_gradient_policy != expected_gradient
            or resource_weighting_scope != expected_resource
            or tuple(cells) != expected_cells
        ):
            raise BundleMaterializationError(
                "The corrected-always factorial campaign surface is not "
                "the exact declared 12-cell arm."
            )
        campaign_surface = {
            "study_id": FACTORIAL_CAMPAIGN_ID,
            "campaign_id": FACTORIAL_CAMPAIGN_ID,
            "study_stage": (
                "always_stationarity_phase1_cost_factorial_v1"
            ),
            "run_class": FACTORIAL_RUN_CLASS,
            "visible_target": {
                "target_id": FACTORIAL_VISIBLE_TARGET_ID,
                "paper": "Paper I",
                "source_lock_role": "per_cell_exact_source_locks",
                "source_lock_pointer": "source_locks.json#/cell_locks",
            },
            "stationarity_winner_selected": False,
            "factorial_arm_cell_count": len(cells),
            "factorial_arm_contract": _factorial_arm_contract(
                active_gradient_policy=active_gradient_policy,
                resource_weighting_scope=resource_weighting_scope,
                cells=cells,
            ),
        }
    elif campaign_id == GLOBAL_SINGLETON_CAMPAIGN_ID:
        expected_cells = build_global_singleton_insertion_cell_specs()
        if (
            bundle_id != GLOBAL_SINGLETON_BUNDLE_ID
            or active_gradient_policy != ACTIVE_GRADIENT_STATIONARY
            or resource_weighting_scope != RESOURCE_WEIGHTING_ALL_PHASE
            or tuple(cells) != expected_cells
        ):
            raise BundleMaterializationError(
                "The global-singleton insertion campaign surface is not "
                "the exact fixed 12-cell contract."
            )
        campaign_surface = {
            "study_id": GLOBAL_SINGLETON_CAMPAIGN_ID,
            "campaign_id": GLOBAL_SINGLETON_CAMPAIGN_ID,
            "study_stage": (
                "global_guarded_singleton_insertion_comparison_v1"
            ),
            "run_class": GLOBAL_SINGLETON_RUN_CLASS,
            "visible_target": {
                "target_id": GLOBAL_SINGLETON_VISIBLE_TARGET_ID,
                "paper": "Paper I",
                "source_lock_role": "per_cell_exact_source_locks",
                "source_lock_pointer": "source_locks.json#/cell_locks",
            },
            "stationarity_condition": "always_applied_v1",
            "phase1_cost_term": "always_applied_v1",
            "global_singleton_insertion_cell_count": len(cells),
            "global_singleton_insertion_contract": (
                _global_singleton_insertion_contract(cells)
            ),
        }
    elif campaign_id == QISKIT_COST_PILOT_CAMPAIGN_ID:
        expected_cells = build_qiskit_cost_plateau_pilot_cell_specs()
        if (
            bundle_id != QISKIT_COST_PILOT_BUNDLE_ID
            or active_gradient_policy != ACTIVE_GRADIENT_STATIONARY
            or resource_weighting_scope != RESOURCE_WEIGHTING_ALL_PHASE
            or tuple(cells) != expected_cells
        ):
            raise BundleMaterializationError(
                "The Qiskit-cost plateau pilot surface is not the exact "
                "fixed two-cell local diagnostic."
            )
        campaign_surface = {
            "study_id": QISKIT_COST_PILOT_CAMPAIGN_ID,
            "campaign_id": QISKIT_COST_PILOT_CAMPAIGN_ID,
            "study_stage": "qiskit_cost_plateau_pilot_v1",
            "run_class": QISKIT_COST_PILOT_RUN_CLASS,
            "visible_target": {
                "target_id": QISKIT_COST_PILOT_VISIBLE_TARGET_ID,
                "paper": "Paper I",
                "source_lock_role": "per_cell_exact_source_locks",
                "source_lock_pointer": "source_locks.json#/cell_locks",
            },
            "stationarity_condition": "always_applied_v1",
            "phase1_cost_term": "always_applied_v1",
            "qiskit_cost_pilot_cell_count": len(cells),
            "qiskit_cost_plateau_pilot_contract": (
                _qiskit_cost_plateau_pilot_contract(cells)
            ),
        }
    elif campaign_id == QISKIT_COST_ALWAYS13_CAMPAIGN_ID:
        expected_cells = build_qiskit_cost_always13_cell_specs()
        if (
            bundle_id != QISKIT_COST_ALWAYS13_BUNDLE_ID
            or active_gradient_policy != ACTIVE_GRADIENT_STATIONARY
            or resource_weighting_scope != RESOURCE_WEIGHTING_ALL_PHASE
            or tuple(cells) != expected_cells
        ):
            raise BundleMaterializationError(
                "The Qiskit-cost always13 surface is not the exact fixed "
                "one-cell local diagnostic."
            )
        campaign_surface = {
            "study_id": QISKIT_COST_ALWAYS13_CAMPAIGN_ID,
            "campaign_id": QISKIT_COST_ALWAYS13_CAMPAIGN_ID,
            "study_stage": "qiskit_cost_macro_always13_diagnostic_v1",
            "run_class": QISKIT_COST_ALWAYS13_RUN_CLASS,
            "visible_target": {
                "target_id": QISKIT_COST_ALWAYS13_VISIBLE_TARGET_ID,
                "paper": "Paper I",
                "source_lock_role": "per_cell_exact_source_locks",
                "source_lock_pointer": "source_locks.json#/cell_locks",
            },
            "stationarity_condition": "always_applied_v1",
            "phase1_cost_term": "always_applied_v1",
            "qiskit_cost_always13_cell_count": len(cells),
            "qiskit_cost_always13_contract": (
                _qiskit_cost_always13_contract(cells)
            ),
        }
    elif campaign_id == QISKIT_COST_ALWAYS6_CAMPAIGN_ID:
        expected_cells = build_qiskit_cost_always6_cell_specs()
        if (
            bundle_id != QISKIT_COST_ALWAYS6_BUNDLE_ID
            or active_gradient_policy != ACTIVE_GRADIENT_STATIONARY
            or resource_weighting_scope != RESOURCE_WEIGHTING_ALL_PHASE
            or tuple(cells) != expected_cells
        ):
            raise BundleMaterializationError(
                "The Qiskit-cost always6 surface is not the exact fixed "
                "six-cell macro always-insertion diagnostic."
            )
        campaign_surface = {
            "study_id": QISKIT_COST_ALWAYS6_CAMPAIGN_ID,
            "campaign_id": QISKIT_COST_ALWAYS6_CAMPAIGN_ID,
            "study_stage": "qiskit_cost_macro_always6_diagnostic_v1",
            "run_class": QISKIT_COST_ALWAYS6_RUN_CLASS,
            "visible_target": {
                "target_id": QISKIT_COST_ALWAYS6_VISIBLE_TARGET_ID,
                "paper": "Paper I",
                "source_lock_role": "per_cell_exact_source_locks",
                "source_lock_pointer": "source_locks.json#/cell_locks",
            },
            "stationarity_condition": "always_applied_v1",
            "phase1_cost_term": "always_applied_v1",
            "qiskit_cost_always6_cell_count": len(cells),
            "qiskit_cost_always6_contract": (
                _qiskit_cost_always6_contract(cells)
            ),
        }
    elif campaign_id == BEAMPRUNE_CAMPAIGN_ID:
        expected_cells = build_beamprune_cell_specs()
        if (
            bundle_id != BEAMPRUNE_BUNDLE_ID
            or active_gradient_policy != ACTIVE_GRADIENT_STATIONARY
            or resource_weighting_scope != RESOURCE_WEIGHTING_ALL_PHASE
            or tuple(cells) != expected_cells
        ):
            raise BundleMaterializationError(
                "The beam+prune lane ablation surface is not the exact fixed "
                "24-cell crossed contract."
            )
        campaign_surface = {
            "study_id": BEAMPRUNE_CAMPAIGN_ID,
            "campaign_id": BEAMPRUNE_CAMPAIGN_ID,
            "study_stage": "macro_always_beamprune_lane_ablation_r50_v1",
            "run_class": BEAMPRUNE_RUN_CLASS,
            "visible_target": {
                "target_id": BEAMPRUNE_VISIBLE_TARGET_ID,
                "paper": "Paper I",
                "source_lock_role": "per_cell_exact_source_locks",
                "source_lock_pointer": "source_locks.json#/cell_locks",
            },
            "stationarity_condition": "always_applied_v1",
            "phase1_cost_term": "always_applied_v1",
            "beamprune_cell_count": len(cells),
            "beamprune_contract": _beamprune_contract(cells),
        }
    elif campaign_id == LANES_ABLATION_CAMPAIGN_ID:
        expected_cells = build_lanes_ablation_cell_specs()
        if (
            bundle_id != LANES_ABLATION_BUNDLE_ID
            or active_gradient_policy != ACTIVE_GRADIENT_STATIONARY
            or resource_weighting_scope != RESOURCE_WEIGHTING_ALL_PHASE
            or tuple(cells) != expected_cells
        ):
            raise BundleMaterializationError(
                "The macro always-insertion lanes ablation surface is not "
                "the exact fixed twelve-cell paired contract."
            )
        campaign_surface = {
            "study_id": LANES_ABLATION_CAMPAIGN_ID,
            "campaign_id": LANES_ABLATION_CAMPAIGN_ID,
            "study_stage": "macro_always_lanes_ablation_r50_v1",
            "run_class": LANES_ABLATION_RUN_CLASS,
            "visible_target": {
                "target_id": LANES_ABLATION_VISIBLE_TARGET_ID,
                "paper": "Paper I",
                "source_lock_role": "per_cell_exact_source_locks",
                "source_lock_pointer": "source_locks.json#/cell_locks",
            },
            "stationarity_condition": "always_applied_v1",
            "phase1_cost_term": "always_applied_v1",
            "lanes_ablation_cell_count": len(cells),
            "lanes_ablation_contract": _lanes_ablation_contract(cells),
        }
    elif campaign_id == PHASE3_QISKIT_CAMPAIGN_ID:
        expected_cells = build_phase3_qiskit_mixed_horizon_cell_specs()
        if (
            bundle_id != PHASE3_QISKIT_BUNDLE_ID
            or active_gradient_policy != ACTIVE_GRADIENT_STATIONARY
            or resource_weighting_scope != RESOURCE_WEIGHTING_ALL_PHASE
            or tuple(cells) != expected_cells
        ):
            raise BundleMaterializationError(
                "The Phase-III-Qiskit campaign surface is not the exact "
                "fixed six-cell mixed-horizon candidate contract."
            )
        campaign_surface = {
            "study_id": PHASE3_QISKIT_CAMPAIGN_ID,
            "campaign_id": PHASE3_QISKIT_CAMPAIGN_ID,
            "study_stage": (
                "global_singleton_phase3_qiskit_candidate_v1"
            ),
            "run_class": PHASE3_QISKIT_RUN_CLASS,
            "visible_target": {
                "target_id": PHASE3_QISKIT_VISIBLE_TARGET_ID,
                "paper": "Paper I",
                "source_lock_role": "per_cell_exact_source_locks",
                "source_lock_pointer": "source_locks.json#/cell_locks",
            },
            "stationarity_condition": "always_applied_v1",
            "phase1_cost_term": "always_applied_v1",
            "phase3_qiskit_candidate_cell_count": len(cells),
            "phase3_qiskit_mixed_horizon_contract": (
                _phase3_qiskit_mixed_horizon_contract(cells)
            ),
        }
    else:
        raise BundleMaterializationError(
            f"Unknown bundle campaign id: {campaign_id!r}."
        )

    payload = {
        "schema": BUNDLE_SCHEMA,
        "bundle_id": bundle_id,
        **campaign_surface,
        "active_gradient_policy": active_gradient_policy,
        "resource_weighting_scope": resource_weighting_scope,
        "execution_target": _execution_target_for_campaign(campaign_id),
        "execution_authorized": False,
        "submission_state": SUBMISSION_STATE,
        "submitted": False,
        "cell_count": len(cells),
        "cells": [
            {
                **cell.to_dict(),
                **(
                    {
                        "preservation_execution_gate": (
                            preservation_execution_gate_contract(
                                active_gradient_policy=(
                                    active_gradient_policy
                                )
                            )
                        )
                    }
                    if cell.preservation_contract_id is not None
                    else {}
                ),
                "protocol_path": f"protocols/{cell.cell_id}.json",
                "execution_template_path": (
                    f"execution_templates/{cell.cell_id}.json"
                ),
            }
            for cell in cells
        ],
        "ordered_cells_contract": {
            "pointer": "#/cells",
            "order_semantics": "serialized_list_order_v1",
            "identity_field": "cell_id",
        },
        "per_cell_provenance_pointers": {
            "typed_protocol": "protocols/<cell_id>.json",
            "protocol_digest": "protocols/<cell_id>.json#/sha256",
            "physics_receipt": "protocols/<cell_id>.json#/problem",
            "settings_lock": (
                "source_locks.json#/cell_locks/<source_lock_id>"
                "/resolver_trace/settings_reused"
            ),
            "same_cutoff_ed_reference": (
                "source_locks.json#/cell_locks/<source_lock_id>"
                "/resolver_trace/same_cutoff_ed_reference"
            ),
            "optimizer": "protocols/<cell_id>.json#/optimizer",
            "optimizer_budget": (
                "protocols/<cell_id>.json#/optimizer_maxiter"
            ),
            "seeds": "protocols/<cell_id>.json#/seeds",
            "reference_state": (
                "protocols/<cell_id>.json#/problem/reference_label"
            ),
            "expected_artifact_roles": (
                "expected_artifacts.json#/cells/<cell_id>"
                "/expected_run_artifacts"
            ),
            "execution_template": "execution_templates/<cell_id>.json",
        },
        "source_locks": {
            "path": "source_locks.json",
            "sha256": source_locks_sha256,
        },
        "expected_artifacts_path": "expected_artifacts.json",
        "validation_report_path": "validation_report.json",
        "compile_identity": dict(RA_ADAPT_COMPILE_IDENTITY),
        "phase3_multiplier_semantics": {
            "curvature_shift_field": "kappa_stabilization_shift",
            "trust_boundary_field": "trust_boundary_multiplier_lambda",
            "total_metric_multiplier_field": (
                "total_metric_multiplier_mu"
            ),
            "identity": "mu_equals_kappa_plus_lambda_v1",
            "trust_boundary_active_iff": "lambda_gt_zero",
        },
        "protocol_execution_separation": (
            "immutable_protocol_plus_observed_execution_manifest_v1"
        ),
        "environment_fingerprint": dict(environment_fingerprint),
        **(
            {
                "numerical_runtime_contract": dict(
                    numerical_runtime_contract
                )
            }
            if numerical_runtime_contract is not None
            else {}
        ),
        **dict(dependency_provenance),
        "repository_state_at_materialization": dict(repository_state),
        "materialization_timestamp": materialization_timestamp,
    }
    return _digested(payload)


def _source_lock_refs(
    normalized_source_locks: Mapping[str, Any],
    *,
    cell: BundleCellSpec,
) -> dict[str, str]:
    cell_lock = normalized_source_locks["cell_locks"][cell.source_lock_id]
    return {
        "source_locks_manifest_sha256": str(
            normalized_source_locks["sha256"]
        ),
        "implementation_source_inventory_sha256": str(
            normalized_source_locks["implementation_sources"]["sha256"]
        ),
        "cell_source_lock_id": cell.source_lock_id,
        "cell_source_lock_sha256": str(cell_lock["sha256"]),
        "visible_provenance_sha256": str(
            normalized_source_locks["global_sources"][
                "macro_visible_provenance"
            ]["sha256"]
        ),
        "provenance_tracker_sha256": str(
            normalized_source_locks["global_sources"][
                "macro_provenance_tracker"
            ]["sha256"]
        ),
        "ed_cutoff_reference_sha256": str(
            normalized_source_locks["global_sources"][
                "ed_cutoff_reference"
            ]["sha256"]
        ),
        "resolver_script_sha256": str(
            normalized_source_locks["global_sources"][
                "visible_settings_resolver"
            ]["sha256"]
        ),
    }


def _bundle_protocol_materialization_authority(
    *,
    cell: BundleCellSpec,
    bundle_id: str,
    bundle_manifest_sha256: str,
    source_locks_sha256: str,
    source_lock_refs: Mapping[str, str],
    active_gradient_policy: str,
    resource_weighting_scope: str,
    protocol_sha256: str | None = None,
) -> BundleProtocolMaterializationAuthority:
    protocol_schema = _protocol_schema_for_cell(cell)
    selector_identity = (
        APPEND_CONVENTIONAL_SELECTOR_ID
        if cell.selector_family == "append_adapt"
        else RA_STAGED_SELECTOR_ID
    )
    receipt = bundle_protocol_materialization_receipt(
        bundle_id=bundle_id,
        bundle_manifest_sha256=bundle_manifest_sha256,
        source_locks_sha256=source_locks_sha256,
        source_lock_refs=source_lock_refs,
        cell_id=cell.cell_id,
        source_lock_id=cell.source_lock_id,
        protocol_schema=protocol_schema,
        algorithm_id=cell.algorithm_id,
        candidate_representation=cell.candidate_representation,
        selector_identity=selector_identity,
        active_gradient_policy=active_gradient_policy,
        resource_weighting_scope=resource_weighting_scope,
    )
    return _mint_bundle_protocol_materialization_authority(
        receipt,
        source_lock_refs=source_lock_refs,
        protocol_sha256=protocol_sha256,
    )


def materialize_semantic_closure_protocol(
    problem: Any,
    request: RAAdaptRequest,
) -> ResolvedRAAdaptProtocol:
    """Mint one exact native semantic capability without I/O or execution.

    Unlike the on-disk campaign materializers, this narrow seam produces only
    an in-memory protocol.  The protocol remains publicly non-executing and
    gains authority solely through the private protocol-digest capability
    attached after the exact source inventory and application contract pass.
    """

    from pipelines.static_adapt.ra_adapt.semantic_closure_routes import (
        preflight_paper_i_ra_semantic,
        semantic_closure_materialization_contract,
    )

    preflight = preflight_paper_i_ra_semantic(problem, request)
    native = semantic_closure_materialization_contract(problem, request)
    if (
        native["bundle_id"] != preflight.bundle_id
        or native["bundle_manifest_sha256"]
        != preflight.bundle_manifest_sha256
        or native["algorithm_id"] != preflight.algorithm_id
    ):
        raise BundleMaterializationError(
            "Semantic native materialization drifted from preflight."
        )
    refs = dict(native["source_lock_refs"])
    receipt = bundle_protocol_materialization_receipt(
        bundle_id=str(native["bundle_id"]),
        bundle_manifest_sha256=str(native["bundle_manifest_sha256"]),
        source_locks_sha256=str(native["source_locks_sha256"]),
        source_lock_refs=refs,
        cell_id=str(native["cell_id"]),
        source_lock_id=str(native["source_lock_id"]),
        protocol_schema=RA_ADAPT_PROTOCOL_SCHEMA,
        algorithm_id=str(native["algorithm_id"]),
        candidate_representation=CANDIDATE_REPRESENTATION_SINGLE_PAULI,
        selector_identity=RA_STAGED_SELECTOR_ID,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
    )
    unbound_authority = _mint_bundle_protocol_materialization_authority(
        receipt,
        source_lock_refs=refs,
    )
    protocol = build_resolved_ra_protocol(
        problem,
        request,
        materialization_authority=unbound_authority,
    )
    if (
        protocol.execution_authorized is not False
        or protocol.bundle_materialization != receipt
        or protocol.bundle_id != native["bundle_id"]
        or protocol.bundle_manifest_sha256
        != native["bundle_manifest_sha256"]
        or protocol.route_contract != preflight.route_contract
    ):
        raise BundleMaterializationError(
            "Semantic native protocol failed its materialization binding."
        )
    bound_authority = _mint_bundle_protocol_materialization_authority(
        receipt,
        source_lock_refs=refs,
        protocol_sha256=protocol.sha256,
    )
    return _attach_validated_bundle_protocol_authority(
        protocol,
        bound_authority,
    )


def _execution_template(
    *,
    cell: BundleCellSpec,
    bundle_id: str,
    protocol_path: str,
    protocol_sha256: str,
    source_lock_refs: Mapping[str, str],
    repository_state: Mapping[str, Any],
    environment_fingerprint: Mapping[str, Any],
    dependency_provenance: Mapping[str, Any],
    campaign_id: str = STUDY_ID,
    numerical_runtime_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if campaign_id == STUDY_ID:
        study_id = STUDY_ID
        run_class = RUN_CLASS
    elif campaign_id == CORE_CAMPAIGN_ID:
        study_id = CORE_CAMPAIGN_ID
        run_class = CORE_RUN_CLASS
    elif campaign_id == FACTORIAL_CAMPAIGN_ID:
        study_id = FACTORIAL_CAMPAIGN_ID
        run_class = FACTORIAL_RUN_CLASS
    elif campaign_id == GLOBAL_SINGLETON_CAMPAIGN_ID:
        study_id = GLOBAL_SINGLETON_CAMPAIGN_ID
        run_class = GLOBAL_SINGLETON_RUN_CLASS
    elif campaign_id == QISKIT_COST_PILOT_CAMPAIGN_ID:
        study_id = QISKIT_COST_PILOT_CAMPAIGN_ID
        run_class = QISKIT_COST_PILOT_RUN_CLASS
    elif campaign_id == QISKIT_COST_ALWAYS13_CAMPAIGN_ID:
        study_id = QISKIT_COST_ALWAYS13_CAMPAIGN_ID
        run_class = QISKIT_COST_ALWAYS13_RUN_CLASS
    elif campaign_id == QISKIT_COST_ALWAYS6_CAMPAIGN_ID:
        study_id = QISKIT_COST_ALWAYS6_CAMPAIGN_ID
        run_class = QISKIT_COST_ALWAYS6_RUN_CLASS
    elif campaign_id == LANES_ABLATION_CAMPAIGN_ID:
        study_id = LANES_ABLATION_CAMPAIGN_ID
        run_class = LANES_ABLATION_RUN_CLASS
    elif campaign_id == BEAMPRUNE_CAMPAIGN_ID:
        study_id = BEAMPRUNE_CAMPAIGN_ID
        run_class = BEAMPRUNE_RUN_CLASS
    elif campaign_id == PHASE3_QISKIT_CAMPAIGN_ID:
        study_id = PHASE3_QISKIT_CAMPAIGN_ID
        run_class = PHASE3_QISKIT_RUN_CLASS
    else:
        raise BundleMaterializationError(
            f"Unknown bundle campaign id: {campaign_id!r}."
        )
    outputs = _artifact_paths(cell)
    return _digested(
        {
            "schema": EXECUTION_TEMPLATE_SCHEMA,
            "cell_id": cell.cell_id,
            "study_id": study_id,
            "campaign_id": campaign_id,
            "run_class": run_class,
            "execution_state": "not_started",
            "execution_target": _execution_target_for_campaign(campaign_id),
            "execution_entrypoint": _EXECUTION_ENTRYPOINTS[
                cell.selector_family
            ],
            "execution_authorized": False,
            "submission_state": SUBMISSION_STATE,
            "submitted": False,
            "protocol": {
                "path": protocol_path,
                "sha256": protocol_sha256,
            },
            "provenance_pointers": {
                "source_lock": (
                    "source_locks.json#/cell_locks/"
                    f"{cell.source_lock_id}"
                ),
                "physics_receipt": f"{protocol_path}#/problem",
                "settings_lock": (
                    "source_locks.json#/cell_locks/"
                    f"{cell.source_lock_id}/resolver_trace/settings_reused"
                ),
                "same_cutoff_ed_reference": (
                    "source_locks.json#/cell_locks/"
                    f"{cell.source_lock_id}/resolver_trace/"
                    "same_cutoff_ed_reference"
                ),
                "seeds": f"{protocol_path}#/seeds",
                "reference_state": (
                    f"{protocol_path}#/problem/reference_label"
                ),
            },
            "expected_artifact_contract": {
                "pointer": (
                    "expected_artifacts.json#/cells/"
                    f"{cell.cell_id}/expected_run_artifacts"
                ),
                "required_roles": list(EXPECTED_ARTIFACT_ROLES),
            },
            "execution_fulfillment": (
                _execution_fulfillment_assignment(
                    campaign_id=campaign_id,
                    bundle_id=bundle_id,
                    cell=cell,
                )
            ),
            "command_argv": None,
            "command_argv_status": "record_at_execution",
            "working_directory_policy": "bundle_root_v1",
            "cwd": None,
            "cwd_status": "record_at_execution",
            "seeds": {"adapt": 7, "transpiler": 7},
            "git_commit": None,
            "git_commit_status": "record_at_execution",
            "dirty_working_tree": None,
            "dirty_working_tree_status": "record_at_execution",
            "input_source_lock_hashes": dict(source_lock_refs),
            "output_artifacts": {
                name: {
                    "path": path,
                    "sha256": None,
                    "status": "not_produced",
                }
                for name, path in outputs.items()
                if name != "execution_manifest"
            },
            "timestamps": {"started_at": None, "finished_at": None},
            "timestamps_status": "record_at_execution",
            "exit_status": None,
            "exit_status_status": "record_at_execution",
            "environment_fingerprint": None,
            "environment_fingerprint_status": "record_at_execution",
            **(
                {
                    "numerical_runtime_contract": dict(
                        numerical_runtime_contract
                    ),
                    "numerical_runtime_receipt": None,
                    "numerical_runtime_receipt_status": (
                        "required_at_execution"
                    ),
                }
                if numerical_runtime_contract is not None
                else {}
            ),
            **dict(dependency_provenance),
        }
    )


def _validate_macro_pool_hash_equality(
    protocols: Mapping[str, Mapping[str, Any]],
    cells: Sequence[BundleCellSpec],
) -> None:
    groups = sorted(
        {
            (cell.regime_id, int(cell.nph))
            for cell in cells
            if (
                cell.candidate_representation
                == CANDIDATE_REPRESENTATION_MACRO
                and cell.horizon is not None
            )
        }
    )
    for regime_id, nph in groups:
        expected_membership = MACRO_POOL_MEMBERSHIP_BY_NPH.get(nph)
        if expected_membership is None:
            raise BundleMaterializationError(
                f"No stable macro-pool membership contract for nph={nph}."
            )
        seen: set[tuple[str, str]] = set()
        for cell in cells:
            if (
                cell.regime_id != regime_id
                or int(cell.nph) != nph
                or cell.candidate_representation
                != CANDIDATE_REPRESENTATION_MACRO
                or cell.horizon is None
            ):
                continue
            payload = protocols[cell.cell_id]
            pool = payload["executable_pool"]
            if (
                int(pool.get("count", -1))
                != int(expected_membership["count"])
                or pool.get("ordered_labels_sha256")
                != expected_membership["ordered_labels_sha256"]
            ):
                raise BundleMaterializationError(
                    "Macro executable-pool membership drift at "
                    f"{regime_id}, nph={nph}, cell={cell.cell_id}."
                )
            seen.add(
                (
                    str(pool.get("ordered_labels_sha256")),
                    str(pool.get("ordered_pool_sha256")),
                )
            )
        if len(seen) != 1:
            raise BundleMaterializationError(
                "Macro RA/Append executable-pool hash drift at "
                f"{regime_id}, nph={nph}."
            )


def _pool_identity(
    pool: Mapping[str, Any],
) -> tuple[str, str]:
    return (
        str(pool.get("ordered_labels_sha256", "")),
        str(pool.get("ordered_pool_sha256", "")),
    )


def _validate_singleton_pool_contracts(
    protocols: Mapping[str, Mapping[str, Any]],
    cells: Sequence[BundleCellSpec],
    *,
    expected_global_cells_per_group: int = 2,
) -> None:
    """Gate singleton ancestry plus RA-staged/Append-global exposure."""

    expected_global_cells_per_group = _require_positive_int(
        expected_global_cells_per_group,
        label="expected_global_cells_per_group",
    )
    groups = sorted(
        {
            (cell.regime_id, int(cell.nph))
            for cell in cells
            if (
                cell.candidate_representation
                == CANDIDATE_REPRESENTATION_SINGLE_PAULI
                and cell.horizon is not None
            )
        }
    )
    for regime_id, nph in groups:
        expected_membership = SINGLETON_PARENT_MEMBERSHIP_BY_NPH.get(nph)
        if expected_membership is None:
            raise BundleMaterializationError(
                f"No stable singleton-parent membership contract for nph={nph}."
            )
        group_cells = [
            cell
            for cell in cells
            if (
                cell.regime_id == regime_id
                and int(cell.nph) == nph
                and cell.candidate_representation
                == CANDIDATE_REPRESENTATION_SINGLE_PAULI
                and cell.horizon is not None
            )
        ]
        parent_identities: set[tuple[str, str]] = set()
        global_executable_identities: set[tuple[str, str]] = set()
        for cell in group_cells:
            payload = protocols[cell.cell_id]
            parent = payload.get("parent_inventory")
            executable = payload.get("executable_pool")
            if not isinstance(parent, Mapping) or not isinstance(
                executable, Mapping
            ):
                raise BundleMaterializationError(
                    f"Singleton pool receipts are missing for {cell.cell_id}."
                )
            expected_parent_count = int(expected_membership["count"])
            if (
                parent.get("schema") != PARENT_TEMPLATE_INVENTORY_SCHEMA
                or parent.get("candidate_representation")
                != CANDIDATE_REPRESENTATION_SINGLE_PAULI
                or int(parent.get("count", -1)) != expected_parent_count
                or parent.get("ordered_labels_sha256")
                != expected_membership["ordered_labels_sha256"]
            ):
                raise BundleMaterializationError(
                    f"Singleton parent ancestry drifted for {cell.cell_id}."
                )
            parent_identity = _pool_identity(parent)
            if not all(parent_identity):
                raise BundleMaterializationError(
                    f"Singleton parent hashes are missing for {cell.cell_id}."
                )
            parent_identities.add(parent_identity)

            lineage = payload.get("lineage_authority")
            if not isinstance(lineage, Mapping):
                raise BundleMaterializationError(
                    f"Singleton lineage authority is missing for "
                    f"{cell.cell_id}."
                )
            if payload.get("adapter_id") == GLOBAL_SINGLE_PAULI_ADAPTER_ID:
                global_membership = (
                    GLOBAL_SINGLETON_POOL_MEMBERSHIP_BY_NPH.get(nph)
                )
                expected_pool_sha = (
                    GLOBAL_SINGLETON_ORDERED_POOL_SHA256_BY_REGIME.get(
                        regime_id
                    )
                )
                candidate_supply = lineage.get("candidate_supply")
                route_contract = payload.get("route_contract")
                route_invariants = (
                    route_contract.get("semantic_invariants")
                    if isinstance(route_contract, Mapping)
                    else None
                )
                expected_supply = {
                    "candidate_adapter_id": (
                        GLOBAL_SINGLE_PAULI_ADAPTER_ID
                    ),
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
                if (
                    global_membership is None
                    or expected_pool_sha is None
                    or executable.get("schema")
                    != GUARDED_SINGLETON_POOL_SCHEMA
                    or executable.get("candidate_representation")
                    != CANDIDATE_REPRESENTATION_SINGLE_PAULI
                    or int(executable.get("count", -1))
                    != int(global_membership["count"])
                    or executable.get("ordered_labels_sha256")
                    != global_membership["ordered_labels_sha256"]
                    or executable.get("ordered_pool_sha256")
                    != expected_pool_sha
                    or executable.get(
                        "source_parent_ordered_labels_sha256"
                    )
                    != parent.get("ordered_labels_sha256")
                    or payload.get("selector_identity")
                    != RA_STAGED_SELECTOR_ID
                    or candidate_supply != expected_supply
                    or not isinstance(route_invariants, Mapping)
                    or any(
                        route_invariants.get(name) != value
                        for name, value in expected_supply.items()
                    )
                ):
                    raise BundleMaterializationError(
                        "Global-singleton RA must expose the exact guarded "
                        "Append inventory in Phase I and retain singleton "
                        f"identity in Phase II for {cell.cell_id}."
                    )
                global_executable_identities.add(
                    _pool_identity(executable)
                )
            elif cell.selector_family == "append_adapt":
                if (
                    executable.get("schema")
                    != GUARDED_SINGLETON_POOL_SCHEMA
                    or executable.get("candidate_representation")
                    != CANDIDATE_REPRESENTATION_SINGLE_PAULI
                    or executable.get(
                        "source_parent_ordered_labels_sha256"
                    )
                    != parent.get("ordered_labels_sha256")
                    or lineage.get("ra_staged_funnel_invoked") is not False
                    or payload.get("selector_identity")
                    != APPEND_CONVENTIONAL_SELECTOR_ID
                ):
                    raise BundleMaterializationError(
                        "Singleton Append must expose the global guarded child "
                        f"pool before selection for {cell.cell_id}."
                    )
            elif (
                executable.get("schema")
                != PARENT_TEMPLATE_INVENTORY_SCHEMA
                or _pool_identity(executable) != parent_identity
                or int(executable.get("count", -1))
                != expected_parent_count
                or payload.get("selector_identity") != RA_STAGED_SELECTOR_ID
            ):
                raise BundleMaterializationError(
                    "Singleton RA must retain the parent factory for staged "
                    f"child exposure in {cell.cell_id}."
                )
        if len(parent_identities) != 1:
            raise BundleMaterializationError(
                "Singleton RA/Append parent-ancestry hash drift at "
                f"{regime_id}, nph={nph}."
            )
        global_cells = [
            cell
            for cell in group_cells
            if protocols[cell.cell_id].get("adapter_id")
            == GLOBAL_SINGLE_PAULI_ADAPTER_ID
        ]
        if global_cells and (
            len(global_cells) != expected_global_cells_per_group
            or len(global_executable_identities) != 1
        ):
            raise BundleMaterializationError(
                "Global-singleton insertion arms lost their common exact "
                f"executable pool at {regime_id}, nph={nph}."
            )


def _validate_qiskit_cost_pilot_source_lock(
    *,
    cell: BundleCellSpec,
    trace: Mapping[str, Any],
) -> None:
    """Require the pilot's exact source anchor and declared cost deltas."""

    anchor = trace.get("qiskit_cost_pilot_source_anchor")
    changes = trace.get("settings_changed")
    if not isinstance(anchor, Mapping) or not isinstance(changes, list):
        raise BundleMaterializationError(
            f"Qiskit-cost pilot source anchor is missing for {cell.cell_id}."
        )
    change_by_id = {
        str(change.get("id")): change
        for change in changes
        if isinstance(change, Mapping)
    }
    is_macro = (
        cell.candidate_representation
        == CANDIDATE_REPRESENTATION_MACRO
    )
    expected_delta_ids = {
        "qiskit_selector_cost_oracle",
        "qiskit_cost_pilot_exact_cell_selection",
    }
    if is_macro:
        expected_delta_ids.add("qiskit_cost_all_phase_scope")
    source_campaign_id = (
        CORE_CAMPAIGN_ID
        if is_macro
        else GLOBAL_SINGLETON_CAMPAIGN_ID
    )
    source_bundle_id = (
        CORE_BUNDLE_ID if is_macro else GLOBAL_SINGLETON_BUNDLE_ID
    )
    source_algorithm_id = _ALGORITHM_IDS[cell.route_id]
    declared = anchor.get("declared_delta_ids")
    oracle_delta = change_by_id.get("qiskit_selector_cost_oracle")
    selection_delta = change_by_id.get(
        "qiskit_cost_pilot_exact_cell_selection"
    )
    scope_delta = change_by_id.get("qiskit_cost_all_phase_scope")
    if (
        anchor.get("schema")
        != "paper_i_ra_adapt_qiskit_cost_plateau_pilot_source_anchor_v1"
        or anchor.get("source_campaign_id") != source_campaign_id
        or anchor.get("source_bundle_id") != source_bundle_id
        or anchor.get("source_route_id") != cell.route_id
        or anchor.get("source_algorithm_id") != source_algorithm_id
        or anchor.get("target_campaign_id")
        != QISKIT_COST_PILOT_CAMPAIGN_ID
        or anchor.get("target_bundle_id") != QISKIT_COST_PILOT_BUNDLE_ID
        or anchor.get("target_algorithm_id") != cell.algorithm_id
        or anchor.get("regime_id") != cell.regime_id
        or int(anchor.get("nph", -1)) != int(cell.nph)
        or anchor.get("scientific_result_anchor_claimed") is not False
        or not isinstance(declared, list)
        or set(map(str, declared)) != expected_delta_ids
        or not expected_delta_ids.issubset(change_by_id)
        or not isinstance(oracle_delta, Mapping)
        or oracle_delta.get("field") != "selector_cost_policy"
        or oracle_delta.get("from") != "marrakesh_graph_span_v1"
        or oracle_delta.get("to") != RA_ADAPT_QISKIT_COST_POLICY
        or not isinstance(selection_delta, Mapping)
        or selection_delta.get("field") != "campaign_cell_selection"
        or selection_delta.get("to") != cell.cell_id
    ):
        raise BundleMaterializationError(
            f"Qiskit-cost pilot source derivation drifted for {cell.cell_id}."
        )
    if is_macro:
        if (
            not isinstance(scope_delta, Mapping)
            or scope_delta.get("field") != "resource_weighting_scope"
            or scope_delta.get("from") != RESOURCE_WEIGHTING_LATE
            or scope_delta.get("to") != RESOURCE_WEIGHTING_ALL_PHASE
        ):
            raise BundleMaterializationError(
                "Qiskit-cost macro pilot lost its explicit all-phase "
                f"scope delta for {cell.cell_id}."
            )
    elif scope_delta is not None:
        raise BundleMaterializationError(
            "Qiskit-cost global-singleton pilot must preserve its existing "
            f"all-phase scope for {cell.cell_id}."
        )


def _validate_qiskit_cost_pilot_protocols(
    protocols: Mapping[str, Mapping[str, Any]],
    cells: Sequence[BundleCellSpec],
) -> None:
    """Fail closed unless both pilot routes use the same Qiskit-cost seam."""

    expected_cells = build_qiskit_cost_plateau_pilot_cell_specs()
    if tuple(cells) != expected_cells:
        raise BundleMaterializationError(
            "Qiskit-cost protocol validation requires the exact ordered "
            "two-cell pilot."
        )
    for cell in cells:
        protocol = protocols.get(cell.cell_id)
        route = (
            protocol.get("route_contract")
            if isinstance(protocol, Mapping)
            else None
        )
        execution = (
            route.get("execution_settings")
            if isinstance(route, Mapping)
            else None
        )
        invariants = (
            route.get("semantic_invariants")
            if isinstance(route, Mapping)
            else None
        )
        lineage = (
            route.get("lineage_authority")
            if isinstance(route, Mapping)
            else None
        )
        request = (
            protocol.get("request")
            if isinstance(protocol, Mapping)
            else None
        )
        method = (
            request.get("method")
            if isinstance(request, Mapping)
            else None
        )
        insertion = (
            method.get("insertion")
            if isinstance(method, Mapping)
            else None
        )
        expected_adapter_id = (
            MacroCandidateAdapter.adapter_id
            if cell.candidate_representation
            == CANDIDATE_REPRESENTATION_MACRO
            else GLOBAL_SINGLE_PAULI_ADAPTER_ID
        )
        intended_changes = (
            lineage.get("only_intended_scientific_changes")
            if isinstance(lineage, Mapping)
            else None
        )
        if (
            not isinstance(protocol, Mapping)
            or protocol.get("algorithm_id") != cell.algorithm_id
            or protocol.get("adapter_id") != expected_adapter_id
            or not isinstance(route, Mapping)
            or not str(route.get("route_profile", "")).endswith(
                "__" + RA_ADAPT_QISKIT_COST_ROUTE_SUFFIX
            )
            or not isinstance(execution, Mapping)
            or execution.get("phase3_backend_cost_mode")
            != "transpile_single_v1"
            or execution.get("phase3_backend_name") != "FakeMarrakesh"
            or execution.get("phase3_backend_optimization_level") != 1
            or execution.get("phase3_backend_transpile_seed") != 7
            or execution.get("adapt_parallel_gradient_workers") != 4
            or execution.get("phase1_shortlist_size") != 24
            or execution.get("phase2_shortlist_size") != 12
            or not isinstance(invariants, Mapping)
            or invariants.get("selector_compile_cost_policy")
            != RA_ADAPT_QISKIT_COST_POLICY
            or invariants.get("selector_compile_cost_phase_reuse")
            != RA_ADAPT_QISKIT_COST_PHASE_REUSE
            or invariants.get("active_gradient_policy")
            != ACTIVE_GRADIENT_STATIONARY
            or invariants.get("resource_weighting_scope")
            != RESOURCE_WEIGHTING_ALL_PHASE
            or not isinstance(insertion, Mapping)
            or insertion.get("kind") != PlateauCommutationInsertion.kind
            or not isinstance(intended_changes, list)
            or "qiskit_full_trial_ansatz_delta_all_phases"
            not in intended_changes
        ):
            raise BundleMaterializationError(
                "Qiskit-cost plateau route contract drifted for "
                f"{cell.cell_id}."
            )


def _validate_qiskit_cost_always13_source_lock(
    *,
    cell: BundleCellSpec,
    trace: Mapping[str, Any],
) -> None:
    """Require an exact plateau-Qiskit source and only the named deltas."""

    anchor = trace.get("qiskit_cost_always13_source_anchor")
    changes = trace.get("settings_changed")
    if not isinstance(anchor, Mapping) or not isinstance(changes, list):
        raise BundleMaterializationError(
            "Qiskit-cost always13 source anchor is missing for "
            f"{cell.cell_id}."
        )
    change_by_id = {
        str(change.get("id")): change
        for change in changes
        if isinstance(change, Mapping)
    }
    expected_delta_ids = {
        "qiskit_cost_always13_insertion_policy",
        "qiskit_cost_always13_horizon",
        "qiskit_cost_always13_exact_cell_selection",
    }
    insertion_delta = change_by_id.get(
        "qiskit_cost_always13_insertion_policy"
    )
    horizon_delta = change_by_id.get("qiskit_cost_always13_horizon")
    selection_delta = change_by_id.get(
        "qiskit_cost_always13_exact_cell_selection"
    )
    declared = anchor.get("declared_delta_ids")
    changed_fields = anchor.get("changed_scientific_fields")
    if (
        anchor.get("schema")
        != "paper_i_ra_adapt_qiskit_cost_always13_source_anchor_v1"
        or anchor.get("source_campaign_id")
        != QISKIT_COST_PILOT_CAMPAIGN_ID
        or anchor.get("source_bundle_id") != QISKIT_COST_PILOT_BUNDLE_ID
        or anchor.get("source_route_id") != ROUTE_RA_MACRO_PLATEAU
        or anchor.get("source_algorithm_id")
        != QISKIT_COST_PILOT_MACRO_ALGORITHM_ID
        or anchor.get("source_protocol_sha256")
        != QISKIT_COST_ALWAYS13_SOURCE_PROTOCOL_SHA256
        or anchor.get("target_campaign_id")
        != QISKIT_COST_ALWAYS13_CAMPAIGN_ID
        or anchor.get("target_bundle_id")
        != QISKIT_COST_ALWAYS13_BUNDLE_ID
        or anchor.get("target_route_id") != cell.route_id
        or anchor.get("target_algorithm_id") != cell.algorithm_id
        or anchor.get("regime_id") != cell.regime_id
        or int(anchor.get("nph", -1)) != int(cell.nph)
        or anchor.get("scientific_result_anchor_claimed") is not False
        or not isinstance(declared, list)
        or set(map(str, declared)) != expected_delta_ids
        or not isinstance(changed_fields, list)
        or changed_fields
        != [
            "request.method.insertion",
            "request.execution.stop.maximum_controller_rounds",
        ]
        or not expected_delta_ids.issubset(change_by_id)
        or not isinstance(insertion_delta, Mapping)
        or insertion_delta.get("field") != "insertion_policy"
        or insertion_delta.get("from")
        != PlateauCommutationInsertion.kind
        or insertion_delta.get("to")
        != AlwaysCommutationReducedInsertion.kind
        or not isinstance(horizon_delta, Mapping)
        or horizon_delta.get("field") != "maximum_controller_rounds"
        or int(horizon_delta.get("from", -1)) != FULL_HORIZON
        or int(horizon_delta.get("to", -1))
        != QISKIT_COST_ALWAYS13_HORIZON
        or not isinstance(selection_delta, Mapping)
        or selection_delta.get("field") != "campaign_cell_selection"
        or selection_delta.get("from")
        != (
            "qiskit_cost_pilot__strong_weak_u8__nph3__"
            f"{ROUTE_RA_MACRO_PLATEAU}"
        )
        or selection_delta.get("to") != cell.cell_id
    ):
        raise BundleMaterializationError(
            "Qiskit-cost always13 source derivation drifted for "
            f"{cell.cell_id}."
        )


def _validate_qiskit_cost_always13_protocols(
    protocols: Mapping[str, Mapping[str, Any]],
    cells: Sequence[BundleCellSpec],
) -> None:
    """Fail closed unless the one target uses the exact requested route."""

    expected_cells = build_qiskit_cost_always13_cell_specs()
    if tuple(cells) != expected_cells:
        raise BundleMaterializationError(
            "Qiskit-cost always13 validation requires the exact ordered "
            "one-cell diagnostic."
        )
    cell = expected_cells[0]
    protocol = protocols.get(cell.cell_id)
    route = (
        protocol.get("route_contract")
        if isinstance(protocol, Mapping)
        else None
    )
    execution = (
        route.get("execution_settings")
        if isinstance(route, Mapping)
        else None
    )
    invariants = (
        route.get("semantic_invariants")
        if isinstance(route, Mapping)
        else None
    )
    lineage = (
        route.get("lineage_authority")
        if isinstance(route, Mapping)
        else None
    )
    request = (
        protocol.get("request")
        if isinstance(protocol, Mapping)
        else None
    )
    method = (
        request.get("method") if isinstance(request, Mapping) else None
    )
    insertion = (
        method.get("insertion") if isinstance(method, Mapping) else None
    )
    request_execution = (
        request.get("execution") if isinstance(request, Mapping) else None
    )
    stop = (
        request_execution.get("stop")
        if isinstance(request_execution, Mapping)
        else None
    )
    intended_changes = (
        lineage.get("only_intended_scientific_changes")
        if isinstance(lineage, Mapping)
        else None
    )
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("algorithm_id") != cell.algorithm_id
        or protocol.get("candidate_representation")
        != CANDIDATE_REPRESENTATION_MACRO
        or protocol.get("adapter_id") != MacroCandidateAdapter.adapter_id
        or int(protocol.get("horizon", -1))
        != QISKIT_COST_ALWAYS13_HORIZON
        or not isinstance(route, Mapping)
        or not str(route.get("route_profile", "")).endswith(
            "__" + RA_ADAPT_QISKIT_COST_ROUTE_SUFFIX
        )
        or not isinstance(execution, Mapping)
        or execution.get("adapt_insertion_mode")
        != "full_commutation_reduced"
        or execution.get("phase3_backend_cost_mode")
        != "transpile_single_v1"
        or execution.get("phase3_backend_name") != "FakeMarrakesh"
        or execution.get("phase3_backend_optimization_level") != 1
        or execution.get("phase3_backend_transpile_seed") != 7
        or execution.get("adapt_parallel_gradient_workers") != 4
        or execution.get("phase1_shortlist_size") != 24
        or execution.get("phase2_shortlist_size") != 12
        or not isinstance(invariants, Mapping)
        or invariants.get("selector_compile_cost_policy")
        != RA_ADAPT_QISKIT_COST_POLICY
        or invariants.get("selector_compile_cost_phase_reuse")
        != RA_ADAPT_QISKIT_COST_PHASE_REUSE
        or invariants.get("active_gradient_policy")
        != ACTIVE_GRADIENT_STATIONARY
        or invariants.get("resource_weighting_scope")
        != RESOURCE_WEIGHTING_ALL_PHASE
        or invariants.get("insertion_position_scope")
        != "full_logical_ansatz_commutation_classes_every_depth_v2"
        or not isinstance(insertion, Mapping)
        or insertion.get("kind")
        != AlwaysCommutationReducedInsertion.kind
        or not isinstance(stop, Mapping)
        or int(stop.get("maximum_controller_rounds", -1))
        != QISKIT_COST_ALWAYS13_HORIZON
        or not isinstance(intended_changes, list)
        or "qiskit_full_trial_ansatz_delta_all_phases"
        not in intended_changes
    ):
        raise BundleMaterializationError(
            "Qiskit-cost always13 route contract drifted for "
            f"{cell.cell_id}."
        )


def _validate_phase3_qiskit_source_lock(
    *,
    cell: BundleCellSpec,
    trace: Mapping[str, Any],
) -> None:
    """Require the exact page-7 parent and Phase-III-only declared delta."""

    anchor = trace.get("phase3_qiskit_source_anchor")
    changes = trace.get("settings_changed")
    if not isinstance(anchor, Mapping) or not isinstance(changes, list):
        raise BundleMaterializationError(
            "Phase-III-Qiskit source anchor is missing for "
            f"{cell.cell_id}."
        )
    change_by_id = {
        str(change.get("id")): change
        for change in changes
        if isinstance(change, Mapping)
    }
    expected_delta_ids = {
        "phase3_qiskit_selector_cost_scope",
        "phase3_qiskit_exact_cell_selection",
    }
    scope_delta = change_by_id.get(
        "phase3_qiskit_selector_cost_scope"
    )
    selection_delta = change_by_id.get(
        "phase3_qiskit_exact_cell_selection"
    )
    declared = anchor.get("declared_delta_ids")
    if (
        anchor.get("schema")
        != "paper_i_ra_adapt_phase3_qiskit_source_anchor_v1"
        or anchor.get("source_algorithm_id")
        != RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_SOURCE_ALGORITHM_ID
        or anchor.get("source_route_id")
        != ROUTE_RA_GLOBAL_SINGLETON_PLATEAU
        or anchor.get("source_route_profile")
        != PHASE3_QISKIT_PAGE7_PARENT_ROUTE_PROFILE
        or anchor.get("source_route_contract_sha256")
        != PHASE3_QISKIT_PAGE7_PARENT_CONTRACT_SHA256
        or anchor.get("target_campaign_id") != PHASE3_QISKIT_CAMPAIGN_ID
        or anchor.get("target_bundle_id") != PHASE3_QISKIT_BUNDLE_ID
        or anchor.get("target_algorithm_id") != cell.algorithm_id
        or anchor.get("regime_id") != cell.regime_id
        or int(anchor.get("nph", -1)) != int(cell.nph)
        or int(anchor.get("source_horizon", -1)) != int(cell.horizon or -1)
        or int(anchor.get("target_horizon", -1)) != int(cell.horizon or -1)
        or anchor.get("scientific_result_anchor_claimed") is not False
        or not isinstance(declared, list)
        or set(map(str, declared)) != expected_delta_ids
        or not expected_delta_ids.issubset(change_by_id)
        or not isinstance(scope_delta, Mapping)
        or scope_delta.get("field") != "selector_compile_cost_scope"
        or scope_delta.get("from") != "marrakesh_graph_span_all_phases_v1"
        or scope_delta.get("to")
        != BACKEND_COMPILE_SCOPE_PHASE3_QISKIT_ONLY_V1
        or not isinstance(selection_delta, Mapping)
        or selection_delta.get("field") != "campaign_cell_selection"
        or selection_delta.get("to") != cell.cell_id
    ):
        raise BundleMaterializationError(
            "Phase-III-Qiskit source derivation drifted for "
            f"{cell.cell_id}."
        )


def _validate_phase3_qiskit_protocols(
    protocols: Mapping[str, Mapping[str, Any]],
    cells: Sequence[BundleCellSpec],
) -> None:
    """Fail closed unless all six cells use only the Phase-III Qiskit seam."""

    expected_cells = build_phase3_qiskit_mixed_horizon_cell_specs()
    if tuple(cells) != expected_cells:
        raise BundleMaterializationError(
            "Phase-III-Qiskit validation requires the exact ordered "
            "six-cell mixed-horizon campaign."
        )
    for cell in cells:
        protocol = protocols.get(cell.cell_id)
        route = (
            protocol.get("route_contract")
            if isinstance(protocol, Mapping)
            else None
        )
        execution = (
            route.get("execution_settings")
            if isinstance(route, Mapping)
            else None
        )
        invariants = (
            route.get("semantic_invariants")
            if isinstance(route, Mapping)
            else None
        )
        lineage = (
            route.get("lineage_authority")
            if isinstance(route, Mapping)
            else None
        )
        request = (
            protocol.get("request")
            if isinstance(protocol, Mapping)
            else None
        )
        method = (
            request.get("method") if isinstance(request, Mapping) else None
        )
        insertion = (
            method.get("insertion")
            if isinstance(method, Mapping)
            else None
        )
        intended_changes = (
            lineage.get("only_intended_scientific_changes")
            if isinstance(lineage, Mapping)
            else None
        )
        if (
            not isinstance(protocol, Mapping)
            or protocol.get("algorithm_id")
            != RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_ALGORITHM_ID
            or protocol.get("adapter_id")
            != GLOBAL_SINGLE_PAULI_ADAPTER_ID
            or int(protocol.get("horizon", -1)) != int(cell.horizon or -1)
            or not isinstance(route, Mapping)
            or not str(route.get("route_profile", "")).endswith(
                "__" + RA_ADAPT_PHASE3_QISKIT_COST_ROUTE_SUFFIX
            )
            or not isinstance(execution, Mapping)
            or execution.get("phase3_backend_cost_mode")
            != MARRAKESH_GRAPH_SPAN_MODE
            or execution.get("phase3_backend_cost_scope")
            != BACKEND_COMPILE_SCOPE_PHASE3_QISKIT_ONLY_V1
            or execution.get("phase3_backend_name") != "FakeMarrakesh"
            or execution.get("phase3_backend_optimization_level") != 1
            or execution.get("phase3_backend_transpile_seed") != 7
            or execution.get("adapt_parallel_gradient_workers") != 4
            or execution.get("phase1_shortlist_size") != 24
            or execution.get("phase2_shortlist_size") != 12
            or not isinstance(invariants, Mapping)
            or invariants.get("selector_compile_cost_policy")
            != RA_ADAPT_PHASE3_QISKIT_COST_POLICY
            or invariants.get("selector_compile_cost_phase_reuse")
            != RA_ADAPT_PHASE3_QISKIT_COST_PHASE_REUSE
            or invariants.get("selector_compile_cost_scope")
            != BACKEND_COMPILE_SCOPE_PHASE3_QISKIT_ONLY_V1
            or invariants.get("phase_i_phase_ii_compile_cost_source")
            != MARRAKESH_GRAPH_SPAN_MODE
            or invariants.get("phase_iii_compile_cost_source")
            != "backend_transpile_v1"
            or invariants.get(
                "phase_iii_qiskit_population_normalization_policy"
            )
            != "family_robust_symmetric_arctan_v1"
            or invariants.get(
                "phase_iii_qiskit_backend_fallback_allowed"
            )
            is not False
            or invariants.get(
                "phase_iii_qiskit_negative_delta_reward_enabled"
            )
            is not False
            or invariants.get(
                "phase_iii_qiskit_raw_signed_telemetry_required"
            )
            is not True
            or invariants.get("active_gradient_policy")
            != ACTIVE_GRADIENT_STATIONARY
            or invariants.get("resource_weighting_scope")
            != RESOURCE_WEIGHTING_ALL_PHASE
            or not isinstance(insertion, Mapping)
            or insertion.get("kind") != PlateauCommutationInsertion.kind
            or not isinstance(lineage, Mapping)
            or lineage.get("parent_route_profile")
            != PHASE3_QISKIT_PAGE7_PARENT_ROUTE_PROFILE
            or lineage.get("parent_contract_sha256")
            != PHASE3_QISKIT_PAGE7_PARENT_CONTRACT_SHA256
            or not isinstance(intended_changes, list)
            or (
                "phase3_selector_cost_graph_span_to_qiskit_"
                "positive_clipped_marginal_transpile"
            )
            not in intended_changes
        ):
            raise BundleMaterializationError(
                "Phase-III-Qiskit route contract drifted for "
                f"{cell.cell_id}."
            )


_GLOBAL_SINGLETON_COMMON_PROTOCOL_FIELDS = (
    "schema",
    "candidate_representation",
    "adapter_id",
    "selector_identity",
    "active_gradient_policy",
    "resource_weighting_scope",
    "derivative_chart_id",
    "trust_policy_id",
    "phase3_solver_id",
    "phase3_multiplier_contract",
    "accepted_refit_scope",
    "accepted_refit_coordinate_chart",
    "accepted_refit_base_chart_policy",
    "problem",
    "parent_inventory",
    "executable_pool",
    "optimizer",
    "optimizer_maxiter",
    "stopping_rule",
    "horizon",
    "seeds",
    "estimator_accounting_convention",
    "compile_identity",
)
_GLOBAL_SINGLETON_INSERTION_ROUTE_EXECUTION_FIELDS = frozenset(
    {"adapt_insertion_mode"}
)
_GLOBAL_SINGLETON_INSERTION_ROUTE_INVARIANT_FIELDS = frozenset(
    {
        "experimental_insertion_policy",
        "canonical_admission_policy",
        "canonical_beam_policy",
        "canonical_composition_schema",
        "canonical_insertion_policy",
        "canonical_pruning_policy",
        "compatibility_resolution_active",
        "diagnostic_position_ablation",
        "insertion_position_scope",
        "insertion_equivalence_policy",
        "online_exact_reference_used",
        "plateau_cumulative_decrease_ratio_threshold",
        "plateau_prior_mean_decrease_ratio_threshold",
        "plateau_hysteresis_active",
        "plateau_patience",
        "plateau_threshold_calibration_status",
        "plateau_threshold_comparison",
        "plateau_trigger_source",
    }
)


def _global_singleton_cross_arm_projection(
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one arm onto fields that are not insertion semantics."""

    request = protocol.get("request")
    method = request.get("method") if isinstance(request, Mapping) else None
    route = protocol.get("route_contract")
    route_execution = (
        route.get("execution_settings")
        if isinstance(route, Mapping)
        else None
    )
    route_invariants = (
        route.get("semantic_invariants")
        if isinstance(route, Mapping)
        else None
    )
    lineage = protocol.get("lineage_authority")
    if (
        not isinstance(request, Mapping)
        or not isinstance(method, Mapping)
        or not isinstance(route_execution, Mapping)
        or not isinstance(route_invariants, Mapping)
        or not isinstance(lineage, Mapping)
    ):
        raise BundleMaterializationError(
            "Global-singleton protocol lacks a cross-arm comparison surface."
        )
    request_common = {
        name: method.get(name)
        for name in ("admission", "pruning", "beam")
    }
    request_common.update(
        {
            "kind": request.get("kind"),
            "adapter": request.get("adapter"),
            "execution": request.get("execution"),
        }
    )
    return {
        "protocol_fields": {
            name: protocol.get(name)
            for name in _GLOBAL_SINGLETON_COMMON_PROTOCOL_FIELDS
        },
        "request_outside_insertion": request_common,
        "candidate_inventory_lineage": lineage.get(
            "candidate_inventory_lineage"
        ),
        "candidate_supply": lineage.get("candidate_supply"),
        "route_execution_outside_insertion": {
            name: value
            for name, value in route_execution.items()
            if name
            not in _GLOBAL_SINGLETON_INSERTION_ROUTE_EXECUTION_FIELDS
        },
        "route_invariants_outside_insertion": {
            name: value
            for name, value in route_invariants.items()
            if name
            not in _GLOBAL_SINGLETON_INSERTION_ROUTE_INVARIANT_FIELDS
        },
    }


def _validate_global_singleton_cross_arm_equality(
    protocols: Mapping[str, Mapping[str, Any]],
    cells: Sequence[BundleCellSpec],
) -> None:
    """Require one exact append/plateau pair per regime and no other delta."""

    campaign_cells = [
        cell
        for cell in cells
        if cell.route_id in GLOBAL_SINGLETON_INSERTION_ROUTE_IDS
    ]
    if not campaign_cells:
        return
    if tuple(campaign_cells) != build_global_singleton_insertion_cell_specs():
        raise BundleMaterializationError(
            "Global-singleton cross-arm validation received a partial or "
            "reordered matrix."
        )
    for regime_id, nph in CLAIM_FACING_REGIME_CUTOFF_PAIRS:
        pair = [
            cell
            for cell in campaign_cells
            if cell.regime_id == regime_id and int(cell.nph) == int(nph)
        ]
        if [cell.route_id for cell in pair] != list(
            GLOBAL_SINGLETON_INSERTION_ROUTE_IDS
        ):
            raise BundleMaterializationError(
                "Global-singleton insertion pair ordering drifted at "
                f"{regime_id}, nph={nph}."
            )
        projections = [
            _global_singleton_cross_arm_projection(
                protocols[cell.cell_id]
            )
            for cell in pair
        ]
        if canonical_sha256(projections[0]) != canonical_sha256(
            projections[1]
        ):
            raise BundleMaterializationError(
                "Global-singleton append/plateau arms differ outside the "
                f"insertion policy at {regime_id}, nph={nph}."
            )


def _validate_core_singleton_source_anchor(
    *,
    cell: BundleCellSpec,
    trace: Mapping[str, Any],
) -> None:
    anchor = trace.get("core_source_anchor")
    if not isinstance(anchor, Mapping):
        raise BundleMaterializationError(
            f"Core singleton source anchor is missing for {cell.cell_id}."
        )
    changes = trace.get("settings_changed")
    change_ids = {
        str(change.get("id"))
        for change in changes
        if isinstance(change, Mapping)
    } if isinstance(changes, list) else set()
    required_common = {
        "core_stationary_gradient_policy",
        "core_candidate_representation_axis",
        "core_fixed_horizon",
    }
    if cell.route_id == ROUTE_APPEND_SINGLETON:
        expected_family = "canonical_append_registry_v1"
        expected_source_route = ROUTE_APPEND_SINGLETON
        expected_insertion = "conventional_unwhitened_append_v1"
        required_route_id = "core_conventional_append_baseline"
    else:
        expected_family = "chtc_9381198_singleton_plateau_v1"
        expected_source_route = ROUTE_RA_SINGLETON_PLATEAU
        expected_insertion = _RA_INSERTION_KIND_BY_ROUTE[cell.route_id]
        required_route_id = "core_insertion_policy_variant"
    derivation = anchor.get("route_derivation")
    declared = (
        derivation.get("declared_delta_ids")
        if isinstance(derivation, Mapping)
        else None
    )
    required_ids = {*required_common, required_route_id}
    if (
        anchor.get("schema")
        != "paper_i_ra_adapt_core_singleton_source_anchor_v1"
        or anchor.get("anchor_family") != expected_family
        or anchor.get("regime_id") != cell.regime_id
        or int(anchor.get("nph", -1)) != int(cell.nph)
        or anchor.get("scientific_result_anchor_claimed") is not False
        or not isinstance(derivation, Mapping)
        or derivation.get("source_route_id") != expected_source_route
        or derivation.get("target_route_id") != cell.route_id
        or derivation.get("target_insertion_policy")
        != expected_insertion
        or not isinstance(declared, list)
        or set(map(str, declared)) != required_ids
        or not required_ids.issubset(change_ids)
    ):
        raise BundleMaterializationError(
            f"Core singleton route derivation drifted for {cell.cell_id}."
        )


_GLOBAL_SINGLETON_COMMON_SOURCE_DELTA_IDS = frozenset(
    {
        "D5",
        "global_singleton_candidate_adapter",
        "global_singleton_phase_i_candidate_supply",
        "global_singleton_phase_i_candidate_visibility",
        "global_singleton_phase_ii_candidate_exposure",
        "global_singleton_route_identity",
    }
)


def _validate_global_singleton_source_anchor(
    *,
    cell: BundleCellSpec,
    trace: Mapping[str, Any],
) -> None:
    """Validate derivation from the exact sealed v13 singleton plateau lock."""

    anchor = trace.get("global_singleton_source_anchor")
    changes = trace.get("settings_changed")
    if not isinstance(anchor, Mapping) or not isinstance(changes, list):
        raise BundleMaterializationError(
            f"Global-singleton source anchor is missing for {cell.cell_id}."
        )
    change_by_id = {
        str(change.get("id")): change
        for change in changes
        if isinstance(change, Mapping)
    }
    expected_ids = set(_GLOBAL_SINGLETON_COMMON_SOURCE_DELTA_IDS)
    expected_insertion = _RA_INSERTION_KIND_BY_ROUTE[cell.route_id]
    if cell.route_id == ROUTE_RA_GLOBAL_SINGLETON_APPEND_REDUCED:
        expected_ids.add("global_singleton_insertion_policy_variant")
    derivation = anchor.get("route_derivation")
    predecessor = anchor.get("predecessor")
    declared = (
        derivation.get("declared_delta_ids")
        if isinstance(derivation, Mapping)
        else None
    )
    expected_change_values = {
        "D5": (
            "resource_weighting_scope",
            RESOURCE_WEIGHTING_ALL_PHASE,
        ),
        "global_singleton_candidate_adapter": (
            "candidate_adapter_id",
            GLOBAL_SINGLE_PAULI_ADAPTER_ID,
        ),
        "global_singleton_phase_i_candidate_supply": (
            "phase_i_candidate_supply",
            PHASE_I_SUPPLY_GLOBAL_GUARDED_SINGLETON,
        ),
        "global_singleton_phase_i_candidate_visibility": (
            "phase_i_candidate_visibility",
            PHASE_I_VISIBILITY_ALL_EXECUTABLE,
        ),
        "global_singleton_phase_ii_candidate_exposure": (
            "phase_ii_candidate_exposure",
            PHASE_II_EXPOSURE_RETAINED_SINGLETON_IDENTITY,
        ),
        "global_singleton_route_identity": (
            "route_id",
            cell.route_id,
        ),
    }
    change_values_match = all(
        isinstance(change_by_id.get(delta_id), Mapping)
        and change_by_id[delta_id].get("field") == field
        and change_by_id[delta_id].get("to") == value
        for delta_id, (field, value) in expected_change_values.items()
    )
    insertion_change = change_by_id.get(
        "global_singleton_insertion_policy_variant"
    )
    if (
        cell.route_id == ROUTE_RA_GLOBAL_SINGLETON_APPEND_REDUCED
        and (
            not isinstance(insertion_change, Mapping)
            or insertion_change.get("field") != "insertion_policy"
            or insertion_change.get("from")
            != PlateauCommutationInsertion.kind
            or insertion_change.get("to") != expected_insertion
        )
    ):
        change_values_match = False
    if (
        anchor.get("schema")
        != "paper_i_ra_adapt_global_singleton_source_anchor_v1"
        or anchor.get("anchor_family")
        != "sealed_stationary_core_v13_singleton_plateau_v1"
        or anchor.get("regime_id") != cell.regime_id
        or int(anchor.get("nph", -1)) != int(cell.nph)
        or anchor.get("scientific_result_anchor_claimed") is not False
        or not isinstance(predecessor, Mapping)
        or predecessor.get("materialization_id")
        != "ra_adapt_stationary_late_core_v13"
        or predecessor.get("source_route_id")
        != ROUTE_RA_SINGLETON_PLATEAU
        or predecessor.get("source_insertion_policy")
        != PlateauCommutationInsertion.kind
        or not isinstance(derivation, Mapping)
        or derivation.get("target_route_id") != cell.route_id
        or derivation.get("target_insertion_policy")
        != expected_insertion
        or not isinstance(declared, list)
        or set(map(str, declared)) != expected_ids
        or not expected_ids.issubset(change_by_id)
        or not change_values_match
    ):
        raise BundleMaterializationError(
            f"Global-singleton source derivation drifted for {cell.cell_id}."
        )


def _validate_global_singleton_source_lock_matrix(
    *,
    source_locks: Mapping[str, Any],
    cells: Sequence[BundleCellSpec],
) -> None:
    """Prove each insertion pair derives from the same locked source bytes."""

    if tuple(cells) != build_global_singleton_insertion_cell_specs():
        raise BundleMaterializationError(
            "Global-singleton source-lock validation requires the exact "
            "ordered 12-cell matrix."
        )
    cell_locks = source_locks.get("cell_locks")
    if not isinstance(cell_locks, Mapping) or len(cell_locks) != 12:
        raise BundleMaterializationError(
            "Global-singleton source-lock matrix must contain exactly "
            "12 locks."
        )
    for regime_id, nph in CLAIM_FACING_REGIME_CUTOFF_PAIRS:
        pair = [
            cell
            for cell in cells
            if cell.regime_id == regime_id and int(cell.nph) == int(nph)
        ]
        locks = [cell_locks.get(cell.source_lock_id) for cell in pair]
        if any(not isinstance(lock, Mapping) for lock in locks):
            raise BundleMaterializationError(
                "Global-singleton source-lock pair is incomplete at "
                f"{regime_id}, nph={nph}."
            )
        append_lock, plateau_lock = locks
        assert isinstance(append_lock, Mapping)
        assert isinstance(plateau_lock, Mapping)
        if (
            append_lock.get("archive") != plateau_lock.get("archive")
            or append_lock.get("member") != plateau_lock.get("member")
        ):
            raise BundleMaterializationError(
                "Global-singleton insertion arms changed locked source "
                f"bytes at {regime_id}, nph={nph}."
            )
        projections: list[dict[str, Any]] = []
        for cell, lock in zip(pair, locks, strict=True):
            assert isinstance(lock, Mapping)
            trace = lock.get("resolver_trace")
            if not isinstance(trace, Mapping):
                raise BundleMaterializationError(
                    f"Global-singleton source trace is missing for "
                    f"{cell.cell_id}."
                )
            _validate_global_singleton_source_anchor(
                cell=cell,
                trace=trace,
            )
            trace_projection = json.loads(canonical_json_bytes(trace))
            trace_projection["method"] = (
                "<global_singleton_insertion_axis>"
            )
            trace_projection.pop("settings_changed", None)
            trace_projection.pop("global_singleton_source_anchor", None)
            projections.append(
                {
                    "archive": lock.get("archive"),
                    "member": lock.get("member"),
                    "resolver_trace_outside_declared_deltas": (
                        trace_projection
                    ),
                }
            )
        if canonical_sha256(projections[0]) != canonical_sha256(
            projections[1]
        ):
            raise BundleMaterializationError(
                "Global-singleton source locks differ outside the declared "
                f"insertion axis at {regime_id}, nph={nph}."
            )


def _validate_factorial_source_axis_lock(
    *,
    cell: BundleCellSpec,
    trace: Mapping[str, Any],
    active_gradient_policy: str,
    resource_weighting_scope: str,
) -> None:
    changes = trace.get("settings_changed")
    if not isinstance(changes, list):
        raise BundleMaterializationError(
            f"Factorial source axes are missing for {cell.cell_id}."
        )
    d5_rows = [
        change
        for change in changes
        if isinstance(change, Mapping) and change.get("id") == "D5"
    ]
    gradient_rows = [
        change
        for change in changes
        if (
            isinstance(change, Mapping)
            and change.get("field") == "active_gradient_policy"
        )
    ]
    gradient_declared = False
    for row in gradient_rows:
        values = row.get("to_bundle_values", ())
        if row.get("to") == active_gradient_policy or (
            isinstance(values, (list, tuple))
            and active_gradient_policy in values
        ):
            gradient_declared = True
            break
    if (
        len(d5_rows) != 1
        or d5_rows[0].get("field") != "resource_weighting_scope"
        or d5_rows[0].get("to") != resource_weighting_scope
        or not gradient_declared
    ):
        raise BundleMaterializationError(
            "Factorial source-lock policy axes drifted for "
            f"{cell.cell_id}."
        )


def _factorial_source_lock_projection(
    source_locks: Mapping[str, Any],
) -> dict[str, Any]:
    projection = json.loads(canonical_json_bytes(source_locks))
    projection.pop("sha256", None)
    for lock in projection["cell_locks"].values():
        lock.pop("sha256", None)
        changes = lock["resolver_trace"]["settings_changed"]
        for change in changes:
            if change.get("id") == "D5":
                change["to"] = "<resource_weighting_scope_axis>"
            if change.get("field") == "active_gradient_policy":
                if "to" in change:
                    change["to"] = "<active_gradient_policy_axis>"
    return projection


def _validate_factorial_source_lock_matrix(
    *,
    source_locks_by_bundle: Mapping[str, Mapping[str, Any]],
    cells_by_bundle: Mapping[str, Sequence[BundleCellSpec]],
) -> None:
    if set(source_locks_by_bundle) != {
        bundle_id
        for bundle_id, _gradient, _resource
        in FACTORIAL_BUNDLE_POLICIES
    }:
        raise BundleMaterializationError(
            "The factorial source-lock matrix lost a declared bundle."
        )
    projections = []
    for bundle_id, gradient_policy, resource_scope in (
        FACTORIAL_BUNDLE_POLICIES
    ):
        source_locks = source_locks_by_bundle[bundle_id]
        cell_locks = source_locks.get("cell_locks")
        if not isinstance(cell_locks, Mapping):
            raise BundleMaterializationError(
                f"Factorial source locks are missing for {bundle_id}."
            )
        for cell in cells_by_bundle[bundle_id]:
            lock = cell_locks.get(cell.source_lock_id)
            trace = (
                lock.get("resolver_trace")
                if isinstance(lock, Mapping)
                else None
            )
            if not isinstance(trace, Mapping):
                raise BundleMaterializationError(
                    f"Factorial source trace is missing for {cell.cell_id}."
                )
            _validate_factorial_source_axis_lock(
                cell=cell,
                trace=trace,
                active_gradient_policy=gradient_policy,
                resource_weighting_scope=resource_scope,
            )
        projections.append(
            _factorial_source_lock_projection(source_locks)
        )
    if len({canonical_sha256(row) for row in projections}) != 1:
        raise BundleMaterializationError(
            "Factorial source locks differ beyond the two declared policy "
            "axes."
        )


def _validate_paper_i_materialization_gate(
    *,
    manifest: Mapping[str, Any],
    normalized_source_locks: Mapping[str, Any],
    protocols: Mapping[str, Mapping[str, Any]],
    execution_templates: Mapping[str, Mapping[str, Any]],
    expected: Mapping[str, Any],
    cells: Sequence[BundleCellSpec],
    blocked_cells: Sequence[str],
    campaign_id: str = STUDY_ID,
) -> dict[str, Any]:
    """Prove the Paper-I run gate from pointers, not copied defaults."""

    if campaign_id == STUDY_ID:
        expected_study_id = STUDY_ID
        expected_run_class = RUN_CLASS
        expected_visible_target_id = VISIBLE_TARGET_ID
    elif campaign_id == CORE_CAMPAIGN_ID:
        expected_study_id = CORE_CAMPAIGN_ID
        expected_run_class = CORE_RUN_CLASS
        expected_visible_target_id = CORE_VISIBLE_TARGET_ID
    elif campaign_id == FACTORIAL_CAMPAIGN_ID:
        expected_study_id = FACTORIAL_CAMPAIGN_ID
        expected_run_class = FACTORIAL_RUN_CLASS
        expected_visible_target_id = FACTORIAL_VISIBLE_TARGET_ID
    elif campaign_id == GLOBAL_SINGLETON_CAMPAIGN_ID:
        expected_study_id = GLOBAL_SINGLETON_CAMPAIGN_ID
        expected_run_class = GLOBAL_SINGLETON_RUN_CLASS
        expected_visible_target_id = (
            GLOBAL_SINGLETON_VISIBLE_TARGET_ID
        )
    elif campaign_id == QISKIT_COST_PILOT_CAMPAIGN_ID:
        expected_study_id = QISKIT_COST_PILOT_CAMPAIGN_ID
        expected_run_class = QISKIT_COST_PILOT_RUN_CLASS
        expected_visible_target_id = (
            QISKIT_COST_PILOT_VISIBLE_TARGET_ID
        )
    elif campaign_id == QISKIT_COST_ALWAYS13_CAMPAIGN_ID:
        expected_study_id = QISKIT_COST_ALWAYS13_CAMPAIGN_ID
        expected_run_class = QISKIT_COST_ALWAYS13_RUN_CLASS
        expected_visible_target_id = (
            QISKIT_COST_ALWAYS13_VISIBLE_TARGET_ID
        )
    elif campaign_id == QISKIT_COST_ALWAYS6_CAMPAIGN_ID:
        expected_study_id = QISKIT_COST_ALWAYS6_CAMPAIGN_ID
        expected_run_class = QISKIT_COST_ALWAYS6_RUN_CLASS
        expected_visible_target_id = (
            QISKIT_COST_ALWAYS6_VISIBLE_TARGET_ID
        )
    elif campaign_id == LANES_ABLATION_CAMPAIGN_ID:
        expected_study_id = LANES_ABLATION_CAMPAIGN_ID
        expected_run_class = LANES_ABLATION_RUN_CLASS
        expected_visible_target_id = LANES_ABLATION_VISIBLE_TARGET_ID
    elif campaign_id == BEAMPRUNE_CAMPAIGN_ID:
        expected_study_id = BEAMPRUNE_CAMPAIGN_ID
        expected_run_class = BEAMPRUNE_RUN_CLASS
        expected_visible_target_id = BEAMPRUNE_VISIBLE_TARGET_ID
    elif campaign_id == PHASE3_QISKIT_CAMPAIGN_ID:
        expected_study_id = PHASE3_QISKIT_CAMPAIGN_ID
        expected_run_class = PHASE3_QISKIT_RUN_CLASS
        expected_visible_target_id = PHASE3_QISKIT_VISIBLE_TARGET_ID
    else:
        raise BundleMaterializationError(
            f"Unknown bundle campaign id: {campaign_id!r}."
        )
    manifest_checks = {
        "run_class": expected_run_class,
        "study_id": expected_study_id,
        "campaign_id": campaign_id,
        "execution_target": _execution_target_for_campaign(campaign_id),
        "execution_authorized": False,
        "submission_state": SUBMISSION_STATE,
        "submitted": False,
    }
    for field, required in manifest_checks.items():
        if manifest.get(field) != required:
            raise BundleMaterializationError(
                f"Paper-I materialization gate drifted at manifest.{field}."
            )
    visible_target = manifest.get("visible_target")
    if not isinstance(visible_target, Mapping) or (
        visible_target.get("target_id") != expected_visible_target_id
    ):
        raise BundleMaterializationError(
            "Paper-I materialization gate has no visible-target source-lock "
            "pointer."
        )
    if campaign_id == STUDY_ID and (
        visible_target.get("source_lock_role")
        != "macro_visible_provenance"
    ):
        raise BundleMaterializationError(
            "Paper-I Study-1 visible-target source-lock role drifted."
        )
    study1_numerical_runtime_contract: Mapping[str, Any] | None = None
    if campaign_id == STUDY_ID:
        raw_runtime_contract = manifest.get(
            "numerical_runtime_contract"
        )
        if not isinstance(raw_runtime_contract, Mapping):
            raise BundleMaterializationError(
                "Paper-I Study-1 has no numerical_runtime_contract."
            )
        try:
            study1_numerical_runtime_contract = (
                normalize_numerical_runtime_contract(
                    raw_runtime_contract
                )
            )
        except NumericalRuntimeContractError as exc:
            raise BundleMaterializationError(
                f"Paper-I Study-1 numerical_runtime_contract drifted: {exc}"
            ) from exc
    if campaign_id in {
        CORE_CAMPAIGN_ID,
        FACTORIAL_CAMPAIGN_ID,
        GLOBAL_SINGLETON_CAMPAIGN_ID,
        QISKIT_COST_PILOT_CAMPAIGN_ID,
        QISKIT_COST_ALWAYS13_CAMPAIGN_ID,
        PHASE3_QISKIT_CAMPAIGN_ID,
    } and visible_target.get("source_lock_role") != (
        "per_cell_exact_source_locks"
    ):
        raise BundleMaterializationError(
            "Paper-I per-cell visible-target source-lock role drifted."
        )
    if campaign_id == CORE_CAMPAIGN_ID:
        forbidden = {
            "study1_shared_execution_dedupe",
            "execution_progression_contract",
            "post_study_1_user_decision_required",
            "validation_cell_count",
            "full_cell_count",
        }
        unexpected = sorted(forbidden.intersection(manifest))
        selection = manifest.get("stationarity_selection")
        authority = (
            selection.get("authority")
            if isinstance(selection, Mapping)
            else None
        )
        campaign_authorities = normalized_source_locks.get(
            "campaign_authorities"
        )
        locked_authority = (
            campaign_authorities.get("stationarity_selection")
            if isinstance(campaign_authorities, Mapping)
            else None
        )
        if (
            unexpected
            or manifest.get("bundle_id") != CORE_BUNDLE_ID
            or manifest.get("stationarity_winner_selected") is not True
            or manifest.get("active_gradient_policy")
            != ACTIVE_GRADIENT_STATIONARY
            or manifest.get("resource_weighting_scope")
            != RESOURCE_WEIGHTING_LATE
            or manifest.get("core_cell_count") != 48
            or tuple(cells) != build_core_cell_specs()
            or not isinstance(authority, Mapping)
            or authority.get("path") != CORE_SELECTION_AUTHORITY_PATH
            or authority.get("sha256")
            != CORE_SELECTION_AUTHORITY_SHA256
            or not isinstance(locked_authority, Mapping)
            or locked_authority.get("path")
            != CORE_SELECTION_AUTHORITY_PATH
            or locked_authority.get("sha256")
            != CORE_SELECTION_AUTHORITY_SHA256
            or locked_authority.get("verified") is not True
        ):
            raise BundleMaterializationError(
                "Paper-I stationary-core campaign surface drifted."
            )
    if campaign_id == FACTORIAL_CAMPAIGN_ID:
        bundle_id = str(manifest.get("bundle_id", ""))
        expected_gradient, expected_resource = (
            _factorial_policy_for_bundle(bundle_id)
        )
        expected_cells = build_factorial_always_cell_specs(
            active_gradient_policy=expected_gradient,
            resource_weighting_scope=expected_resource,
        )
        expected_arm_contract = _factorial_arm_contract(
            active_gradient_policy=expected_gradient,
            resource_weighting_scope=expected_resource,
            cells=expected_cells,
        )
        forbidden = {
            "study1_shared_execution_dedupe",
            "execution_progression_contract",
            "post_study_1_user_decision_required",
            "validation_cell_count",
            "full_cell_count",
            "stationarity_selection",
            "core_cell_count",
            "core_matrix_contract",
        }
        if (
            forbidden.intersection(manifest)
            or manifest.get("stationarity_winner_selected") is not False
            or manifest.get("active_gradient_policy")
            != expected_gradient
            or manifest.get("resource_weighting_scope")
            != expected_resource
            or manifest.get("factorial_arm_cell_count") != 12
            or manifest.get("factorial_arm_contract")
            != expected_arm_contract
            or tuple(cells) != expected_cells
        ):
            raise BundleMaterializationError(
                "Paper-I corrected-always factorial arm surface drifted."
            )
    if campaign_id == GLOBAL_SINGLETON_CAMPAIGN_ID:
        expected_cells = build_global_singleton_insertion_cell_specs()
        expected_contract = _global_singleton_insertion_contract(
            expected_cells
        )
        forbidden = {
            "study1_shared_execution_dedupe",
            "execution_progression_contract",
            "post_study_1_user_decision_required",
            "validation_cell_count",
            "full_cell_count",
            "stationarity_selection",
            "core_cell_count",
            "core_matrix_contract",
            "factorial_arm_cell_count",
            "factorial_arm_contract",
        }
        if (
            forbidden.intersection(manifest)
            or manifest.get("bundle_id")
            != GLOBAL_SINGLETON_BUNDLE_ID
            or manifest.get("active_gradient_policy")
            != ACTIVE_GRADIENT_STATIONARY
            or manifest.get("resource_weighting_scope")
            != RESOURCE_WEIGHTING_ALL_PHASE
            or manifest.get("stationarity_condition")
            != "always_applied_v1"
            or manifest.get("phase1_cost_term")
            != "always_applied_v1"
            or manifest.get("global_singleton_insertion_cell_count")
            != 12
            or manifest.get("global_singleton_insertion_contract")
            != expected_contract
            or tuple(cells) != expected_cells
        ):
            raise BundleMaterializationError(
                "Paper-I global-singleton insertion campaign surface "
                "drifted."
            )
    if campaign_id == QISKIT_COST_PILOT_CAMPAIGN_ID:
        expected_cells = build_qiskit_cost_plateau_pilot_cell_specs()
        expected_contract = _qiskit_cost_plateau_pilot_contract(
            expected_cells
        )
        forbidden = {
            "study1_shared_execution_dedupe",
            "execution_progression_contract",
            "post_study_1_user_decision_required",
            "validation_cell_count",
            "full_cell_count",
            "stationarity_selection",
            "core_cell_count",
            "core_matrix_contract",
            "factorial_arm_cell_count",
            "factorial_arm_contract",
            "global_singleton_insertion_cell_count",
            "global_singleton_insertion_contract",
        }
        if (
            forbidden.intersection(manifest)
            or manifest.get("bundle_id")
            != QISKIT_COST_PILOT_BUNDLE_ID
            or manifest.get("active_gradient_policy")
            != ACTIVE_GRADIENT_STATIONARY
            or manifest.get("resource_weighting_scope")
            != RESOURCE_WEIGHTING_ALL_PHASE
            or manifest.get("execution_target")
            != QISKIT_COST_PILOT_EXECUTION_TARGET
            or manifest.get("stationarity_condition")
            != "always_applied_v1"
            or manifest.get("phase1_cost_term")
            != "always_applied_v1"
            or manifest.get("qiskit_cost_pilot_cell_count") != 2
            or manifest.get("qiskit_cost_plateau_pilot_contract")
            != expected_contract
            or tuple(cells) != expected_cells
        ):
            raise BundleMaterializationError(
                "Paper-I Qiskit-cost plateau pilot surface drifted."
            )
    if campaign_id == QISKIT_COST_ALWAYS13_CAMPAIGN_ID:
        expected_cells = build_qiskit_cost_always13_cell_specs()
        expected_contract = _qiskit_cost_always13_contract(
            expected_cells
        )
        forbidden = {
            "study1_shared_execution_dedupe",
            "execution_progression_contract",
            "post_study_1_user_decision_required",
            "validation_cell_count",
            "full_cell_count",
            "stationarity_selection",
            "core_cell_count",
            "core_matrix_contract",
            "factorial_arm_cell_count",
            "factorial_arm_contract",
            "global_singleton_insertion_cell_count",
            "global_singleton_insertion_contract",
            "qiskit_cost_pilot_cell_count",
            "qiskit_cost_plateau_pilot_contract",
        }
        if (
            forbidden.intersection(manifest)
            or manifest.get("bundle_id")
            != QISKIT_COST_ALWAYS13_BUNDLE_ID
            or manifest.get("active_gradient_policy")
            != ACTIVE_GRADIENT_STATIONARY
            or manifest.get("resource_weighting_scope")
            != RESOURCE_WEIGHTING_ALL_PHASE
            or manifest.get("execution_target")
            != QISKIT_COST_ALWAYS13_EXECUTION_TARGET
            or manifest.get("stationarity_condition")
            != "always_applied_v1"
            or manifest.get("phase1_cost_term")
            != "always_applied_v1"
            or manifest.get("qiskit_cost_always13_cell_count") != 1
            or manifest.get("qiskit_cost_always13_contract")
            != expected_contract
            or tuple(cells) != expected_cells
        ):
            raise BundleMaterializationError(
                "Paper-I Qiskit-cost always13 diagnostic surface drifted."
            )
    if campaign_id == QISKIT_COST_ALWAYS6_CAMPAIGN_ID:
        expected_cells = build_qiskit_cost_always6_cell_specs()
        expected_contract = _qiskit_cost_always6_contract(expected_cells)
        forbidden = {
            "study1_shared_execution_dedupe",
            "execution_progression_contract",
            "post_study_1_user_decision_required",
            "validation_cell_count",
            "full_cell_count",
            "stationarity_selection",
            "core_cell_count",
            "core_matrix_contract",
            "factorial_arm_cell_count",
            "factorial_arm_contract",
            "global_singleton_insertion_cell_count",
            "global_singleton_insertion_contract",
            "qiskit_cost_pilot_cell_count",
            "qiskit_cost_plateau_pilot_contract",
            "qiskit_cost_always13_cell_count",
            "qiskit_cost_always13_contract",
        }
        if (
            forbidden.intersection(manifest)
            or manifest.get("bundle_id") != QISKIT_COST_ALWAYS6_BUNDLE_ID
            or manifest.get("active_gradient_policy")
            != ACTIVE_GRADIENT_STATIONARY
            or manifest.get("resource_weighting_scope")
            != RESOURCE_WEIGHTING_ALL_PHASE
            or manifest.get("execution_target")
            != QISKIT_COST_ALWAYS6_EXECUTION_TARGET
            or manifest.get("stationarity_condition")
            != "always_applied_v1"
            or manifest.get("phase1_cost_term") != "always_applied_v1"
            or manifest.get("qiskit_cost_always6_cell_count") != 6
            or manifest.get("qiskit_cost_always6_contract")
            != expected_contract
            or tuple(cells) != expected_cells
        ):
            raise BundleMaterializationError(
                "Paper-I Qiskit-cost always6 diagnostic surface drifted."
            )
    if campaign_id == BEAMPRUNE_CAMPAIGN_ID:
        expected_cells = build_beamprune_cell_specs()
        expected_contract = _beamprune_contract(expected_cells)
        if (
            manifest.get("bundle_id") != BEAMPRUNE_BUNDLE_ID
            or manifest.get("execution_target") != BEAMPRUNE_EXECUTION_TARGET
            or manifest.get("beamprune_cell_count") != 24
            or manifest.get("beamprune_contract") != expected_contract
            or tuple(cells) != expected_cells
        ):
            raise BundleMaterializationError(
                "Paper-I beam+prune lane ablation surface drifted."
            )
    if campaign_id == LANES_ABLATION_CAMPAIGN_ID:
        expected_cells = build_lanes_ablation_cell_specs()
        expected_contract = _lanes_ablation_contract(expected_cells)
        if (
            manifest.get("bundle_id") != LANES_ABLATION_BUNDLE_ID
            or manifest.get("active_gradient_policy")
            != ACTIVE_GRADIENT_STATIONARY
            or manifest.get("resource_weighting_scope")
            != RESOURCE_WEIGHTING_ALL_PHASE
            or manifest.get("execution_target")
            != LANES_ABLATION_EXECUTION_TARGET
            or manifest.get("lanes_ablation_cell_count") != 12
            or manifest.get("lanes_ablation_contract") != expected_contract
            or tuple(cells) != expected_cells
        ):
            raise BundleMaterializationError(
                "Paper-I macro always-insertion lanes ablation surface "
                "drifted."
            )
    if campaign_id == PHASE3_QISKIT_CAMPAIGN_ID:
        expected_cells = build_phase3_qiskit_mixed_horizon_cell_specs()
        expected_contract = _phase3_qiskit_mixed_horizon_contract(
            expected_cells
        )
        forbidden = {
            "study1_shared_execution_dedupe",
            "execution_progression_contract",
            "post_study_1_user_decision_required",
            "validation_cell_count",
            "full_cell_count",
            "stationarity_selection",
            "core_cell_count",
            "core_matrix_contract",
            "factorial_arm_cell_count",
            "factorial_arm_contract",
            "global_singleton_insertion_cell_count",
            "global_singleton_insertion_contract",
            "qiskit_cost_pilot_cell_count",
            "qiskit_cost_plateau_pilot_contract",
            "qiskit_cost_always13_cell_count",
            "qiskit_cost_always13_contract",
        }
        if (
            forbidden.intersection(manifest)
            or manifest.get("bundle_id") != PHASE3_QISKIT_BUNDLE_ID
            or manifest.get("active_gradient_policy")
            != ACTIVE_GRADIENT_STATIONARY
            or manifest.get("resource_weighting_scope")
            != RESOURCE_WEIGHTING_ALL_PHASE
            or manifest.get("execution_target")
            != PHASE3_QISKIT_EXECUTION_TARGET
            or manifest.get("stationarity_condition")
            != "always_applied_v1"
            or manifest.get("phase1_cost_term") != "always_applied_v1"
            or manifest.get("phase3_qiskit_candidate_cell_count") != 6
            or manifest.get("phase3_qiskit_mixed_horizon_contract")
            != expected_contract
            or tuple(cells) != expected_cells
        ):
            raise BundleMaterializationError(
                "Paper-I Phase-III-Qiskit candidate surface drifted."
            )
    ordered_ids = [cell.cell_id for cell in cells]
    manifest_rows = manifest.get("cells")
    if not isinstance(manifest_rows, list) or [
        row.get("cell_id") if isinstance(row, Mapping) else None
        for row in manifest_rows
    ] != ordered_ids:
        raise BundleMaterializationError(
            "Paper-I materialization gate lost ordered cells."
        )
    manifest_by_id = {
        str(row["cell_id"]): row
        for row in manifest_rows
        if isinstance(row, Mapping)
    }
    expected_dedupe: Mapping[str, Any] | None = None
    if campaign_id == STUDY_ID:
        serialized_dedupe = manifest.get("study1_shared_execution_dedupe")
        expected_dedupe = study1_shared_execution_dedupe_contract()
        if not isinstance(serialized_dedupe, Mapping):
            raise BundleMaterializationError(
                "Paper-I manifest has no Study-1 Append dedupe contract."
            )
        _verify_digest(
            serialized_dedupe, label="Study-1 Append dedupe contract"
        )
        if dict(serialized_dedupe) != expected_dedupe:
            raise BundleMaterializationError(
                "Paper-I Study-1 Append dedupe contract drifted."
            )

    blocked = set(blocked_cells)
    for cell in cells:
        lock = normalized_source_locks["cell_locks"].get(
            cell.source_lock_id
        )
        if not isinstance(lock, Mapping):
            raise BundleMaterializationError(
                f"Paper-I materialization gate has no source lock for "
                f"{cell.cell_id}."
            )
        trace = lock.get("resolver_trace")
        settings_lock = (
            trace.get("settings_reused")
            if isinstance(trace, Mapping)
            else None
        )
        same_cutoff_ed = (
            trace.get("same_cutoff_ed_reference")
            if isinstance(trace, Mapping)
            else None
        )
        if not isinstance(settings_lock, Mapping) or not settings_lock:
            raise BundleMaterializationError(
                f"Paper-I materialization gate has no settings lock for "
                f"{cell.cell_id}."
            )
        if not isinstance(same_cutoff_ed, Mapping):
            raise BundleMaterializationError(
                f"Paper-I materialization gate has no same-cutoff ED "
                f"reference for {cell.cell_id}."
            )
        if campaign_id == FACTORIAL_CAMPAIGN_ID:
            _validate_factorial_source_axis_lock(
                cell=cell,
                trace=trace,
                active_gradient_policy=str(
                    manifest["active_gradient_policy"]
                ),
                resource_weighting_scope=str(
                    manifest["resource_weighting_scope"]
                ),
            )
        if (
            campaign_id in {
                CORE_CAMPAIGN_ID,
                FACTORIAL_CAMPAIGN_ID,
            }
            and cell.candidate_representation
            == CANDIDATE_REPRESENTATION_SINGLE_PAULI
        ):
            _validate_core_singleton_source_anchor(
                cell=cell,
                trace=trace,
            )
        if campaign_id == GLOBAL_SINGLETON_CAMPAIGN_ID:
            _validate_global_singleton_source_anchor(
                cell=cell,
                trace=trace,
            )
        if campaign_id == QISKIT_COST_PILOT_CAMPAIGN_ID:
            _validate_qiskit_cost_pilot_source_lock(
                cell=cell,
                trace=trace,
            )
        if campaign_id == QISKIT_COST_ALWAYS13_CAMPAIGN_ID:
            _validate_qiskit_cost_always13_source_lock(
                cell=cell,
                trace=trace,
            )
        if campaign_id == PHASE3_QISKIT_CAMPAIGN_ID:
            _validate_phase3_qiskit_source_lock(
                cell=cell,
                trace=trace,
            )

        protocol = protocols.get(cell.cell_id)
        if not isinstance(protocol, Mapping):
            raise BundleMaterializationError(
                f"Paper-I materialization gate has no typed protocol for "
                f"{cell.cell_id}."
            )
        _verify_digest(protocol, label=f"protocol {cell.cell_id}")
        if protocol.get("execution_authorized") is not False:
            raise BundleMaterializationError(
                f"Paper-I protocol unexpectedly authorizes execution for "
                f"{cell.cell_id}."
            )
        if cell.cell_id in blocked:
            if protocol.get("schema") != BLOCKED_PROTOCOL_SCHEMA:
                raise BundleMaterializationError(
                    f"Blocked protocol schema drifted for {cell.cell_id}."
                )
        else:
            expected_schema = _protocol_schema_for_cell(cell)
            problem = protocol.get("problem")
            seeds = protocol.get("seeds")
            if (
                protocol.get("schema") != expected_schema
                or not isinstance(problem, Mapping)
                or not str(problem.get("reference_label", "")).strip()
                or not str(protocol.get("optimizer", "")).strip()
                or int(protocol.get("optimizer_maxiter", 0)) < 1
                or not isinstance(seeds, Mapping)
            ):
                raise BundleMaterializationError(
                    f"Paper-I typed protocol gate is incomplete for "
                    f"{cell.cell_id}."
                )

        template = execution_templates.get(cell.cell_id)
        if not isinstance(template, Mapping):
            raise BundleMaterializationError(
                f"Paper-I materialization gate has no execution template "
                f"for {cell.cell_id}."
            )
        template_checks = {
            "study_id": expected_study_id,
            "campaign_id": campaign_id,
            "run_class": expected_run_class,
            "execution_target": _execution_target_for_campaign(campaign_id),
            "execution_entrypoint": _EXECUTION_ENTRYPOINTS[
                cell.selector_family
            ],
            "execution_authorized": False,
            "submission_state": SUBMISSION_STATE,
            "submitted": False,
        }
        for field, required in template_checks.items():
            if template.get(field) != required:
                raise BundleMaterializationError(
                    "Paper-I materialization gate drifted at execution "
                    f"template {cell.cell_id}.{field}."
                )
        if campaign_id == STUDY_ID and (
            template.get("numerical_runtime_contract")
            != study1_numerical_runtime_contract
            or template.get("numerical_runtime_receipt") is not None
            or template.get("numerical_runtime_receipt_status")
            != "required_at_execution"
        ):
            raise BundleMaterializationError(
                "Paper-I Study-1 execution template lost the shared "
                f"numerical runtime gate for {cell.cell_id}."
            )
        if any(
            template.get(field) is not None
            for field in (
                "cwd",
                "git_commit",
                "dirty_working_tree",
                "environment_fingerprint",
                "exit_status",
            )
        ) or template.get("timestamps") != {
            "started_at": None,
            "finished_at": None,
        }:
            raise BundleMaterializationError(
                f"Execution template records unobserved execution state for "
                f"{cell.cell_id}."
            )
        if template.get("working_directory_policy") != "bundle_root_v1":
            raise BundleMaterializationError(
                f"Execution template is not bundle-portable for "
                f"{cell.cell_id}."
            )
        expected_fulfillment = _execution_fulfillment_assignment(
            campaign_id=campaign_id,
            bundle_id=str(manifest["bundle_id"]),
            cell=cell,
        )
        if template.get("execution_fulfillment") != expected_fulfillment:
            raise BundleMaterializationError(
                f"Execution fulfillment drifted for {cell.cell_id}."
            )
        protocol_pointer = template.get("protocol")
        if not isinstance(protocol_pointer, Mapping) or (
            protocol_pointer.get("path")
            != f"protocols/{cell.cell_id}.json"
            or protocol_pointer.get("sha256") != protocol.get("sha256")
        ):
            raise BundleMaterializationError(
                f"Execution template protocol pointer drifted for "
                f"{cell.cell_id}."
            )
        if cell.cell_id not in blocked:
            request = protocol.get("request")
            observation = (
                request.get("observation")
                if isinstance(request, Mapping)
                else None
            )
            checkpoint = (
                observation.get("checkpoint")
                if isinstance(observation, Mapping)
                else None
            )
            ledger = (
                observation.get("estimator_ledger")
                if isinstance(observation, Mapping)
                else None
            )
            for name, receipt in (
                ("checkpoint", checkpoint),
                ("estimator_ledger", ledger),
            ):
                path_text = (
                    receipt.get("path")
                    if isinstance(receipt, Mapping)
                    else None
                )
                if (
                    not isinstance(path_text, str)
                    or Path(path_text).is_absolute()
                    or not path_text.startswith(f"runs/{cell.cell_id}/")
                ):
                    raise BundleMaterializationError(
                        f"Protocol {name} path is not bundle-relative for "
                        f"{cell.cell_id}."
                    )
        artifact_contract = template.get("expected_artifact_contract")
        if not isinstance(artifact_contract, Mapping) or tuple(
            artifact_contract.get("required_roles", ())
        ) != EXPECTED_ARTIFACT_ROLES:
            raise BundleMaterializationError(
                f"Expected artifact role contract drifted for "
                f"{cell.cell_id}."
            )
        expected_cell = expected["cells"].get(cell.cell_id)
        expected_roles = (
            expected_cell.get("expected_run_artifacts")
            if isinstance(expected_cell, Mapping)
            else None
        )
        if not isinstance(expected_roles, Mapping) or set(
            expected_roles
        ) != set(EXPECTED_ARTIFACT_ROLES):
            raise BundleMaterializationError(
                f"Expected artifact roles are incomplete for {cell.cell_id}."
            )
        reference_fulfilled = (
            expected_fulfillment["fulfillment_kind"]
            == "shared_result_reference_v1"
        )
        for role, artifact in expected_roles.items():
            if (
                not isinstance(artifact, Mapping)
                or artifact.get("required") is not True
                or artifact.get("fulfillment_kind")
                != expected_fulfillment["fulfillment_kind"]
                or artifact.get("direct_file_required")
                is reference_fulfilled
                or artifact.get("reference_receipt_required")
                is not reference_fulfilled
            ):
                raise BundleMaterializationError(
                    "Expected artifact fulfillment drifted for "
                    f"{cell.cell_id}.{role}."
                )
        if (
            not isinstance(expected_cell, Mapping)
            or expected_cell.get("execution_fulfillment")
            != expected_fulfillment
        ):
            raise BundleMaterializationError(
                f"Expected execution fulfillment drifted for {cell.cell_id}."
            )
        manifest_cell = manifest_by_id[cell.cell_id]
        if cell.preservation_contract_id is not None:
            expected_gate = preservation_execution_gate_contract(
                active_gradient_policy=str(
                    manifest["active_gradient_policy"]
                )
            )
            if (
                manifest_cell.get("preservation_execution_gate")
                != expected_gate
                or expected_cell.get("preservation_execution_gate")
                != expected_gate
            ):
                raise BundleMaterializationError(
                    f"Preservation gate drifted for {cell.cell_id}."
                )
        elif (
            "preservation_execution_gate" in manifest_cell
            or "preservation_execution_gate" in expected_cell
        ):
            raise BundleMaterializationError(
                f"Unexpected preservation gate for {cell.cell_id}."
            )

    result = {
        "run_class": expected_run_class,
        "visible_target_id": expected_visible_target_id,
        "campaign_id": campaign_id,
        "ordered_cell_ids_sha256": canonical_sha256(ordered_ids),
        "cell_count": len(cells),
        "typed_protocol_count": len(protocols),
        "source_lock_count": len(
            normalized_source_locks["cell_locks"]
        ),
        "same_cutoff_ed_reference_count": len(cells),
        "execution_template_count": len(execution_templates),
        "expected_artifact_roles": list(EXPECTED_ARTIFACT_ROLES),
        "execution_target": _execution_target_for_campaign(campaign_id),
        "execution_authorized": False,
        "submission_state": SUBMISSION_STATE,
        "submitted": False,
        "implementation_source_inventory_sha256": (
            normalized_source_locks["implementation_sources"]["sha256"]
        ),
    }
    if expected_dedupe is not None:
        result["study1_shared_execution_dedupe_sha256"] = (
            expected_dedupe["sha256"]
        )
    if campaign_id == CORE_CAMPAIGN_ID:
        result["all_cells_direct_execution"] = True
        result["direct_execution_cell_count"] = len(cells)
        result["semantic_route_ids"] = list(
            (*MACRO_ROUTE_IDS, *SINGLETON_CORE_ROUTE_IDS)
        )
        result["stationarity_selection_authority_sha256"] = (
            CORE_SELECTION_AUTHORITY_SHA256
        )
    if campaign_id == FACTORIAL_CAMPAIGN_ID:
        result["all_cells_direct_execution"] = True
        result["direct_execution_cell_count"] = len(cells)
        result["semantic_route_ids"] = [
            ROUTE_RA_MACRO_ALWAYS,
            ROUTE_RA_SINGLETON_ALWAYS,
        ]
        result["active_gradient_policy"] = manifest[
            "active_gradient_policy"
        ]
        result["resource_weighting_scope"] = manifest[
            "resource_weighting_scope"
        ]
    if campaign_id == GLOBAL_SINGLETON_CAMPAIGN_ID:
        result["all_cells_direct_execution"] = True
        result["direct_execution_cell_count"] = len(cells)
        result["semantic_route_ids"] = list(
            GLOBAL_SINGLETON_INSERTION_ROUTE_IDS
        )
        result["candidate_adapter_id"] = (
            GLOBAL_SINGLE_PAULI_ADAPTER_ID
        )
        result["active_gradient_policy"] = (
            ACTIVE_GRADIENT_STATIONARY
        )
        result["resource_weighting_scope"] = (
            RESOURCE_WEIGHTING_ALL_PHASE
        )
    if campaign_id == QISKIT_COST_PILOT_CAMPAIGN_ID:
        result["all_cells_direct_execution"] = True
        result["direct_execution_cell_count"] = len(cells)
        result["semantic_route_ids"] = list(
            QISKIT_COST_PILOT_ROUTE_IDS
        )
        result["algorithm_ids"] = [
            QISKIT_COST_PILOT_MACRO_ALGORITHM_ID,
            QISKIT_COST_PILOT_GLOBAL_SINGLETON_ALGORITHM_ID,
        ]
        result["active_gradient_policy"] = (
            ACTIVE_GRADIENT_STATIONARY
        )
        result["resource_weighting_scope"] = (
            RESOURCE_WEIGHTING_ALL_PHASE
        )
        result["execution_target"] = (
            QISKIT_COST_PILOT_EXECUTION_TARGET
        )
    if campaign_id == QISKIT_COST_ALWAYS13_CAMPAIGN_ID:
        result["all_cells_direct_execution"] = True
        result["direct_execution_cell_count"] = len(cells)
        result["semantic_route_ids"] = list(
            QISKIT_COST_ALWAYS13_ROUTE_IDS
        )
        result["algorithm_ids"] = [
            QISKIT_COST_ALWAYS13_ALGORITHM_ID,
        ]
        result["active_gradient_policy"] = (
            ACTIVE_GRADIENT_STATIONARY
        )
        result["resource_weighting_scope"] = (
            RESOURCE_WEIGHTING_ALL_PHASE
        )
        result["execution_target"] = (
            QISKIT_COST_ALWAYS13_EXECUTION_TARGET
        )
    if campaign_id == PHASE3_QISKIT_CAMPAIGN_ID:
        result["all_cells_direct_execution"] = True
        result["direct_execution_cell_count"] = len(cells)
        result["semantic_route_ids"] = list(PHASE3_QISKIT_ROUTE_IDS)
        result["algorithm_ids"] = [
            RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_ALGORITHM_ID
        ]
        result["candidate_adapter_id"] = GLOBAL_SINGLE_PAULI_ADAPTER_ID
        result["active_gradient_policy"] = ACTIVE_GRADIENT_STATIONARY
        result["resource_weighting_scope"] = RESOURCE_WEIGHTING_ALL_PHASE
        result["execution_target"] = PHASE3_QISKIT_EXECUTION_TARGET
        result["source_parent_contract_sha256"] = (
            PHASE3_QISKIT_PAGE7_PARENT_CONTRACT_SHA256
        )
    return result


def _materialization_binding(
    *,
    manifest: Mapping[str, Any],
    normalized_source_locks: Mapping[str, Any],
    protocols: Mapping[str, Mapping[str, Any]],
    execution_templates: Mapping[str, Mapping[str, Any]],
    expected: Mapping[str, Any],
    cells: Sequence[BundleCellSpec],
    materialization_gate: Mapping[str, Any],
) -> dict[str, Any]:
    ordered_protocols = [
        {
            "cell_id": cell.cell_id,
            "sha256": protocols[cell.cell_id]["sha256"],
        }
        for cell in cells
    ]
    ordered_execution_templates = [
        {
            "cell_id": cell.cell_id,
            "sha256": execution_templates[cell.cell_id]["sha256"],
        }
        for cell in cells
    ]
    return _digested(
        {
            "schema": MATERIALIZATION_BINDING_SCHEMA,
            "bundle_id": manifest["bundle_id"],
            "bundle_manifest_sha256": manifest["sha256"],
            "source_locks_sha256": normalized_source_locks["sha256"],
            "expected_artifacts_sha256": expected["sha256"],
            "ordered_protocols_sha256": canonical_sha256(
                ordered_protocols
            ),
            "ordered_execution_templates_sha256": canonical_sha256(
                ordered_execution_templates
            ),
            "paper_i_run_materialization_gate_sha256": (
                materialization_gate["sha256"]
            ),
        }
    )


def _objective_execution_gates(
    *,
    active_gradient_policy: str | None = None,
) -> list[dict[str, Any]]:
    gates = [
        {
            "id": gate_id,
            "status": "not_run",
            "blocks_full_matrix": True,
        }
        for gate_id in OBJECTIVE_EXECUTION_GATE_IDS
    ]
    for gate in gates:
        if gate["id"] in {
            "g5_insertion_position_correctness_v2",
            "g6_phase3_integrity_v2",
        }:
            gate.update(
                {
                    "required_minimum_count": 1,
                    "observed_count": None,
                    "observation_status": "unobserved",
                }
            )
        if (
            gate["id"] == "g13_same_physics_preservation_v2"
            and active_gradient_policy is not None
        ):
            contract = preservation_execution_gate_contract(
                active_gradient_policy=active_gradient_policy
            )
            gate.update(
                {
                    "gate_contract_id": contract["gate_id"],
                    "gate_contract_sha256": contract["sha256"],
                }
            )
    return gates


def validate_full_matrix_progression(
    validation_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless an executed validation proves every objective gate."""

    if not isinstance(validation_report, Mapping):
        raise BundleMaterializationError(
            "Validation progression report must be a mapping."
        )
    _verify_digest(validation_report, label="validation progression report")
    if validation_report.get("materialization_status") != "passed":
        raise BundleMaterializationError(
            "Full-matrix progression requires passed materialization."
        )
    if (
        validation_report.get("execution_progression_status")
        != "validation_passed"
    ):
        raise BundleMaterializationError(
            "Full-matrix progression requires an executed, passed validation."
        )
    raw_gates = validation_report.get("objective_execution_gates")
    if not isinstance(raw_gates, list):
        raise BundleMaterializationError(
            "Validation progression has no objective execution gates."
        )
    gates = {
        str(gate.get("id")): gate
        for gate in raw_gates
        if isinstance(gate, Mapping)
    }
    if set(gates) != set(OBJECTIVE_EXECUTION_GATE_IDS):
        raise BundleMaterializationError(
            "Validation progression objective-gate set drifted."
        )
    for gate_id in OBJECTIVE_EXECUTION_GATE_IDS:
        gate = gates[gate_id]
        if gate.get("status") != "passed":
            raise BundleMaterializationError(
                f"Full-matrix progression blocked by {gate_id}."
            )
        if gate_id in {
            "g5_insertion_position_correctness_v2",
            "g6_phase3_integrity_v2",
        }:
            minimum = int(gate.get("required_minimum_count", 0))
            observed = gate.get("observed_count")
            if minimum != 1 or isinstance(observed, bool) or not isinstance(
                observed, int
            ) or observed < minimum:
                raise BundleMaterializationError(
                    f"Full-matrix progression requires a non-vacuous "
                    f"{gate_id} occurrence."
                )
    return _digested(
        {
            "schema": "ra_adapt_full_matrix_progression_receipt_v2",
            "bundle_id": validation_report.get("bundle_id"),
            "validation_report_sha256": validation_report["sha256"],
            "validation_status": "passed",
            "full_matrix_progression_ready": True,
            "objective_gate_ids": list(OBJECTIVE_EXECUTION_GATE_IDS),
            "interior_insertion_count": gates[
                "g5_insertion_position_correctness_v2"
            ]["observed_count"],
            "trust_contraction_count": gates[
                "g6_phase3_integrity_v2"
            ]["observed_count"],
            "execution_authorized": False,
            "submission_state": SUBMISSION_STATE,
            "submitted": False,
        }
    )


def _prepare_bundle(
    *,
    destination: Path,
    bundle_id: str,
    active_gradient_policy: str,
    cells: Sequence[BundleCellSpec],
    normalized_source_locks: Mapping[str, Any],
    problem_resolver: ProblemResolver,
    protocol_resolver: ProtocolResolver,
    repository_state: Mapping[str, Any],
    environment_fingerprint: Mapping[str, Any],
    dependency_provenance: Mapping[str, Any],
    materialization_timestamp: str | None,
    resource_weighting_scope: str = RESOURCE_WEIGHTING_LATE,
    campaign_id: str = STUDY_ID,
    numerical_runtime_contract: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Mapping[str, Any]], MaterializedBundleReceipt]:
    bundle_dir = destination / bundle_id
    manifest = _manifest_payload(
        bundle_id=bundle_id,
        active_gradient_policy=active_gradient_policy,
        resource_weighting_scope=resource_weighting_scope,
        cells=cells,
        source_locks_sha256=str(normalized_source_locks["sha256"]),
        environment_fingerprint=environment_fingerprint,
        dependency_provenance=dependency_provenance,
        repository_state=repository_state,
        materialization_timestamp=materialization_timestamp,
        campaign_id=campaign_id,
        numerical_runtime_contract=numerical_runtime_contract,
    )
    manifest_sha = str(manifest["sha256"])

    problem_cache: dict[tuple[str, int], Any] = {}
    protocols: dict[str, Mapping[str, Any]] = {}
    execution_templates: dict[str, Mapping[str, Any]] = {}
    expected_cells: dict[str, Any] = {}
    blocked_cells: list[str] = []
    for cell in cells:
        refs = _source_lock_refs(normalized_source_locks, cell=cell)
        protocol_path = f"protocols/{cell.cell_id}.json"
        if cell.horizon is None:
            protocol = _blocked_protocol(
                cell=cell,
                bundle_id=bundle_id,
                bundle_manifest_sha256=manifest_sha,
                active_gradient_policy=active_gradient_policy,
                resource_weighting_scope=resource_weighting_scope,
                source_lock_refs=refs,
            )
            blocked_cells.append(cell.cell_id)
        else:
            cache_key = (cell.regime_id, int(cell.nph))
            if cache_key not in problem_cache:
                problem_cache[cache_key] = problem_resolver(*cache_key)
            request = _build_request(cell, bundle_dir=bundle_dir)
            materialization_authority = (
                _bundle_protocol_materialization_authority(
                    cell=cell,
                    bundle_id=bundle_id,
                    bundle_manifest_sha256=manifest_sha,
                    source_locks_sha256=str(
                        normalized_source_locks["sha256"]
                    ),
                    source_lock_refs=refs,
                    active_gradient_policy=active_gradient_policy,
                    resource_weighting_scope=resource_weighting_scope,
                )
            )
            resolved = protocol_resolver(
                ProtocolResolutionContext(
                    cell=cell,
                    problem=problem_cache[cache_key],
                    request=request,
                    active_gradient_policy=active_gradient_policy,
                    resource_weighting_scope=resource_weighting_scope,
                    bundle_id=bundle_id,
                    bundle_manifest_sha256=manifest_sha,
                    source_lock_refs=refs,
                    materialization_authority=(
                        materialization_authority
                    ),
                )
            )
            protocol = _as_protocol_payload(resolved, cell=cell)
            cell_source_lock = normalized_source_locks["cell_locks"][
                cell.source_lock_id
            ]
            protocol = _decorate_protocol_payload(
                protocol,
                cell=cell,
                request=request,
                cell_source_lock=cell_source_lock,
                materialization_authority=(
                    materialization_authority
                ),
            )
            _validate_protocol_payload(
                protocol,
                cell=cell,
                bundle_id=bundle_id,
                bundle_manifest_sha256=manifest_sha,
                active_gradient_policy=active_gradient_policy,
                resource_weighting_scope=resource_weighting_scope,
                source_lock_refs=refs,
                cell_source_lock=cell_source_lock,
                source_locks_sha256=str(
                    normalized_source_locks["sha256"]
                ),
            )
        protocols[cell.cell_id] = protocol
        execution_template = _execution_template(
            cell=cell,
            bundle_id=bundle_id,
            protocol_path=protocol_path,
            protocol_sha256=str(protocol["sha256"]),
            source_lock_refs=refs,
            repository_state=repository_state,
            environment_fingerprint=environment_fingerprint,
            dependency_provenance=dependency_provenance,
            campaign_id=campaign_id,
            numerical_runtime_contract=numerical_runtime_contract,
        )
        execution_templates[cell.cell_id] = execution_template
        execution_fulfillment = _execution_fulfillment_assignment(
            campaign_id=campaign_id,
            bundle_id=bundle_id,
            cell=cell,
        )
        reference_fulfilled = (
            execution_fulfillment["fulfillment_kind"]
            == "shared_result_reference_v1"
        )
        expected_cells[cell.cell_id] = {
            "stage": cell.stage,
            "execution_fulfillment": execution_fulfillment,
            "protocol": {
                "path": protocol_path,
                "sha256": protocol["sha256"],
                "status": (
                    "blocked"
                    if cell.horizon is None
                    else "resolved"
                ),
            },
            "execution_template": {
                "path": f"execution_templates/{cell.cell_id}.json",
                "sha256": execution_template["sha256"],
            },
            "expected_run_artifacts": {
                name: {
                    "path": path,
                    "required": True,
                    "fulfillment_kind": execution_fulfillment[
                        "fulfillment_kind"
                    ],
                    "direct_file_required": not reference_fulfilled,
                    "reference_receipt_required": reference_fulfilled,
                }
                for name, path in _artifact_paths(cell).items()
            },
            **(
                {
                    "preservation_execution_gate": (
                        preservation_execution_gate_contract(
                            active_gradient_policy=active_gradient_policy
                        )
                    )
                }
                if cell.preservation_contract_id is not None
                else {}
            ),
        }

    if not blocked_cells:
        _validate_macro_pool_hash_equality(protocols, cells)
        _validate_singleton_pool_contracts(
            protocols,
            cells,
            expected_global_cells_per_group=(
                1
                if campaign_id
                in {
                    QISKIT_COST_PILOT_CAMPAIGN_ID,
                    PHASE3_QISKIT_CAMPAIGN_ID,
                }
                else 2
            ),
        )
        if campaign_id == GLOBAL_SINGLETON_CAMPAIGN_ID:
            _validate_global_singleton_source_lock_matrix(
                source_locks=normalized_source_locks,
                cells=cells,
            )
            _validate_global_singleton_cross_arm_equality(
                protocols, cells
            )
        if campaign_id == QISKIT_COST_PILOT_CAMPAIGN_ID:
            _validate_qiskit_cost_pilot_protocols(protocols, cells)
        if campaign_id == QISKIT_COST_ALWAYS13_CAMPAIGN_ID:
            _validate_qiskit_cost_always13_protocols(protocols, cells)
        if campaign_id == PHASE3_QISKIT_CAMPAIGN_ID:
            _validate_phase3_qiskit_protocols(protocols, cells)

    expected = _digested(
        {
            "schema": EXPECTED_ARTIFACTS_SCHEMA,
            "bundle_id": bundle_id,
            "cell_count": len(cells),
            "cells": expected_cells,
        }
    )
    materialization_status = "blocked" if blocked_cells else "passed"
    materialization_gate = _digested(
        _validate_paper_i_materialization_gate(
            manifest=manifest,
            normalized_source_locks=normalized_source_locks,
            protocols=protocols,
            execution_templates=execution_templates,
            expected=expected,
            cells=cells,
            blocked_cells=blocked_cells,
            campaign_id=campaign_id,
        )
    )
    materialization_binding = _materialization_binding(
        manifest=manifest,
        normalized_source_locks=normalized_source_locks,
        protocols=protocols,
        execution_templates=execution_templates,
        expected=expected,
        cells=cells,
        materialization_gate=materialization_gate,
    )
    common_checks = [
        {
            "id": "bundle_schema_and_digest",
            "status": "passed",
            "observed": BUNDLE_SCHEMA,
        },
        {
            "id": "source_locks_exact_bytes",
            "status": (
                "passed"
                if normalized_source_locks[
                    "all_required_files_verified"
                ]
                else "blocked"
            ),
            "observed": normalized_source_locks["sha256"],
        },
        {
            "id": "resolved_protocol_contracts",
            "status": "blocked" if blocked_cells else "passed",
            "observed": {
                "resolved": len(cells) - len(blocked_cells),
                "blocked": len(blocked_cells),
            },
        },
        {
            "id": "macro_pool_hash_equality",
            "status": "not_evaluated" if blocked_cells else "passed",
        },
        {
            "id": "singleton_pool_exposure_contracts",
            "status": "not_evaluated" if blocked_cells else "passed",
        },
    ]
    terminal_checks = [
        {
            "id": "protocol_execution_separation",
            "status": "passed",
            "observed": {
                "protocol_count": len(protocols),
                "execution_template_count": len(execution_templates),
                "execution_authorized": False,
                "submission_state": SUBMISSION_STATE,
                "submitted": False,
            },
        },
        {
            "id": "paper_i_run_materialization_gate",
            "status": "blocked" if blocked_cells else "passed",
            "observed": materialization_gate,
            "blocking_cells": blocked_cells,
        },
    ]
    if campaign_id == STUDY_ID:
        checks = [
            common_checks[0],
            {
                "id": "finite_cell_matrix",
                "status": "passed",
                "observed": {
                    "total": len(cells),
                    "validation": sum(
                        cell.stage == "validation" for cell in cells
                    ),
                    "full": sum(cell.stage == "full" for cell in cells),
                },
            },
            common_checks[1],
            {
                "id": "validation_horizon",
                "status": "blocked" if blocked_cells else "passed",
                "observed": (
                    None
                    if blocked_cells
                    else next(
                        cell.horizon
                        for cell in cells
                        if cell.stage == "validation"
                    )
                ),
                "blocking_cells": blocked_cells,
            },
            *common_checks[2:],
            {
                "id": "study1_append_shared_execution_dedupe",
                "status": "passed",
                "observed": study1_shared_execution_dedupe_contract(),
            },
            *terminal_checks,
        ]
        validation_surface = {
            "execution_progression_status": "not_run",
            "objective_execution_gates": _objective_execution_gates(
                active_gradient_policy=active_gradient_policy
            ),
            "stationarity_winner_selected": False,
            "user_decision_required_after_study_1": True,
        }
    elif campaign_id == CORE_CAMPAIGN_ID:
        direct_count = sum(
            template["execution_fulfillment"]["fulfillment_kind"]
            == "direct_execution_v1"
            for template in execution_templates.values()
        )
        core_validation_binding = _digested(
            {
                "schema": (
                    "paper_i_ra_adapt_stationary_core_validation_binding_v1"
                ),
                "campaign_id": CORE_CAMPAIGN_ID,
                "bundle_id": CORE_BUNDLE_ID,
                "bundle_manifest_sha256": manifest["sha256"],
                "source_locks_sha256": normalized_source_locks["sha256"],
                "expected_artifacts_sha256": expected["sha256"],
                "materialization_gate_sha256": materialization_gate["sha256"],
                "implementation_source_inventory_sha256": (
                    normalized_source_locks["implementation_sources"][
                        "sha256"
                    ]
                ),
                "stationarity_selection_authority_sha256": (
                    CORE_SELECTION_AUTHORITY_SHA256
                ),
                "semantic_route_ids": list(
                    (*MACRO_ROUTE_IDS, *SINGLETON_CORE_ROUTE_IDS)
                ),
                "cell_count": len(cells),
                "direct_execution_cell_count": direct_count,
                "materialization_status": materialization_status,
                "p3_execution_receipt_required": True,
                "execution_authorized": False,
                "submission_state": SUBMISSION_STATE,
                "submitted": False,
            }
        )
        checks = [
            common_checks[0],
            {
                "id": "exact_core_cell_matrix",
                "status": "passed",
                "observed": {
                    "total": len(cells),
                    "regime_cutoff_pair_count": len(
                        CLAIM_FACING_REGIME_CUTOFF_PAIRS
                    ),
                    "route_family_count": 8,
                    "horizon": FULL_HORIZON,
                },
            },
            common_checks[1],
            *common_checks[2:],
            {
                "id": "all_cells_direct_execution",
                "status": (
                    "passed" if direct_count == len(cells) else "blocked"
                ),
                "observed": direct_count,
            },
            *terminal_checks,
        ]
        validation_surface = {
            "stationarity_winner_selected": True,
            "core_validation_binding": core_validation_binding,
            "scientific_execution_status": "not_run",
        }
    elif campaign_id == FACTORIAL_CAMPAIGN_ID:
        direct_count = sum(
            template["execution_fulfillment"]["fulfillment_kind"]
            == "direct_execution_v1"
            for template in execution_templates.values()
        )
        arm_contract = manifest["factorial_arm_contract"]
        factorial_validation_binding = _digested(
            {
                "schema": (
                    "paper_i_ra_adapt_always_stationarity_phase1_cost_"
                    "factorial_validation_binding_v1"
                ),
                "campaign_id": FACTORIAL_CAMPAIGN_ID,
                "bundle_id": bundle_id,
                "bundle_manifest_sha256": manifest["sha256"],
                "source_locks_sha256": normalized_source_locks["sha256"],
                "expected_artifacts_sha256": expected["sha256"],
                "materialization_gate_sha256": materialization_gate["sha256"],
                "implementation_source_inventory_sha256": (
                    normalized_source_locks["implementation_sources"][
                        "sha256"
                    ]
                ),
                "factorial_arm_contract_sha256": canonical_sha256(
                    arm_contract
                ),
                "active_gradient_policy": active_gradient_policy,
                "resource_weighting_scope": resource_weighting_scope,
                "semantic_route_ids": [
                    ROUTE_RA_MACRO_ALWAYS,
                    ROUTE_RA_SINGLETON_ALWAYS,
                ],
                "cell_count": len(cells),
                "direct_execution_cell_count": direct_count,
                "materialization_status": materialization_status,
                "execution_authorized": False,
                "submission_state": SUBMISSION_STATE,
                "submitted": False,
            }
        )
        checks = [
            common_checks[0],
            {
                "id": "exact_factorial_arm_matrix",
                "status": "passed",
                "observed": {
                    "total": len(cells),
                    "regime_cutoff_pair_count": len(
                        CLAIM_FACING_REGIME_CUTOFF_PAIRS
                    ),
                    "candidate_representation_count": 2,
                    "route_ids": [
                        ROUTE_RA_MACRO_ALWAYS,
                        ROUTE_RA_SINGLETON_ALWAYS,
                    ],
                    "horizon": FULL_HORIZON,
                    "active_gradient_policy": active_gradient_policy,
                    "resource_weighting_scope": resource_weighting_scope,
                },
            },
            common_checks[1],
            *common_checks[2:],
            {
                "id": "all_cells_direct_execution",
                "status": (
                    "passed" if direct_count == len(cells) else "blocked"
                ),
                "observed": direct_count,
            },
            *terminal_checks,
        ]
        validation_surface = {
            "stationarity_winner_selected": False,
            "factorial_validation_binding": (
                factorial_validation_binding
            ),
            "scientific_execution_status": "not_run",
        }
    elif campaign_id == GLOBAL_SINGLETON_CAMPAIGN_ID:
        direct_count = sum(
            template["execution_fulfillment"]["fulfillment_kind"]
            == "direct_execution_v1"
            for template in execution_templates.values()
        )
        campaign_contract = manifest[
            "global_singleton_insertion_contract"
        ]
        validation_binding = _digested(
            {
                "schema": (
                    "paper_i_ra_adapt_global_singleton_insertion_"
                    "validation_binding_v1"
                ),
                "campaign_id": GLOBAL_SINGLETON_CAMPAIGN_ID,
                "bundle_id": GLOBAL_SINGLETON_BUNDLE_ID,
                "bundle_manifest_sha256": manifest["sha256"],
                "source_locks_sha256": normalized_source_locks["sha256"],
                "expected_artifacts_sha256": expected["sha256"],
                "materialization_gate_sha256": (
                    materialization_gate["sha256"]
                ),
                "implementation_source_inventory_sha256": (
                    normalized_source_locks["implementation_sources"][
                        "sha256"
                    ]
                ),
                "campaign_contract_sha256": canonical_sha256(
                    campaign_contract
                ),
                "candidate_adapter_id": (
                    GLOBAL_SINGLE_PAULI_ADAPTER_ID
                ),
                "active_gradient_policy": (
                    ACTIVE_GRADIENT_STATIONARY
                ),
                "resource_weighting_scope": (
                    RESOURCE_WEIGHTING_ALL_PHASE
                ),
                "semantic_route_ids": list(
                    GLOBAL_SINGLETON_INSERTION_ROUTE_IDS
                ),
                "cell_count": len(cells),
                "direct_execution_cell_count": direct_count,
                "materialization_status": materialization_status,
                "execution_authorized": False,
                "submission_state": SUBMISSION_STATE,
                "submitted": False,
            }
        )
        checks = [
            common_checks[0],
            {
                "id": "exact_global_singleton_insertion_matrix",
                "status": "passed",
                "observed": {
                    "total": len(cells),
                    "regime_cutoff_pair_count": len(
                        CLAIM_FACING_REGIME_CUTOFF_PAIRS
                    ),
                    "route_ids": list(
                        GLOBAL_SINGLETON_INSERTION_ROUTE_IDS
                    ),
                    "horizon": FULL_HORIZON,
                    "candidate_adapter_id": (
                        GLOBAL_SINGLE_PAULI_ADAPTER_ID
                    ),
                },
            },
            common_checks[1],
            *common_checks[2:],
            {
                "id": "global_singleton_source_lock_pair_equality",
                "status": (
                    "not_evaluated" if blocked_cells else "passed"
                ),
            },
            {
                "id": "global_singleton_cross_arm_scientific_equality",
                "status": (
                    "not_evaluated" if blocked_cells else "passed"
                ),
            },
            {
                "id": "all_cells_direct_execution",
                "status": (
                    "passed" if direct_count == len(cells) else "blocked"
                ),
                "observed": direct_count,
            },
            *terminal_checks,
        ]
        validation_surface = {
            "stationarity_condition": "always_applied_v1",
            "phase1_cost_term": "always_applied_v1",
            "global_singleton_insertion_validation_binding": (
                validation_binding
            ),
            "scientific_execution_status": "not_run",
        }
    elif campaign_id == QISKIT_COST_PILOT_CAMPAIGN_ID:
        direct_count = sum(
            template["execution_fulfillment"]["fulfillment_kind"]
            == "direct_execution_v1"
            for template in execution_templates.values()
        )
        campaign_contract = manifest[
            "qiskit_cost_plateau_pilot_contract"
        ]
        validation_binding = _digested(
            {
                "schema": (
                    "paper_i_ra_adapt_qiskit_cost_plateau_pilot_"
                    "validation_binding_v1"
                ),
                "campaign_id": QISKIT_COST_PILOT_CAMPAIGN_ID,
                "bundle_id": QISKIT_COST_PILOT_BUNDLE_ID,
                "bundle_manifest_sha256": manifest["sha256"],
                "source_locks_sha256": normalized_source_locks["sha256"],
                "expected_artifacts_sha256": expected["sha256"],
                "materialization_gate_sha256": (
                    materialization_gate["sha256"]
                ),
                "implementation_source_inventory_sha256": (
                    normalized_source_locks["implementation_sources"][
                        "sha256"
                    ]
                ),
                "campaign_contract_sha256": canonical_sha256(
                    campaign_contract
                ),
                "algorithm_ids": [
                    QISKIT_COST_PILOT_MACRO_ALGORITHM_ID,
                    QISKIT_COST_PILOT_GLOBAL_SINGLETON_ALGORITHM_ID,
                ],
                "active_gradient_policy": (
                    ACTIVE_GRADIENT_STATIONARY
                ),
                "resource_weighting_scope": (
                    RESOURCE_WEIGHTING_ALL_PHASE
                ),
                "execution_target": QISKIT_COST_PILOT_EXECUTION_TARGET,
                "semantic_route_ids": list(
                    QISKIT_COST_PILOT_ROUTE_IDS
                ),
                "cell_count": len(cells),
                "direct_execution_cell_count": direct_count,
                "materialization_status": materialization_status,
                "execution_authorized": False,
                "submission_state": SUBMISSION_STATE,
                "submitted": False,
            }
        )
        checks = [
            common_checks[0],
            {
                "id": "exact_qiskit_cost_plateau_pilot_matrix",
                "status": "passed",
                "observed": {
                    "total": len(cells),
                    "route_ids": list(QISKIT_COST_PILOT_ROUTE_IDS),
                    "algorithm_ids": [
                        QISKIT_COST_PILOT_MACRO_ALGORITHM_ID,
                        QISKIT_COST_PILOT_GLOBAL_SINGLETON_ALGORITHM_ID,
                    ],
                    "horizon": FULL_HORIZON,
                    "execution_target": (
                        QISKIT_COST_PILOT_EXECUTION_TARGET
                    ),
                },
            },
            common_checks[1],
            *common_checks[2:],
            {
                "id": "qiskit_cost_pilot_source_derivations",
                "status": (
                    "not_evaluated" if blocked_cells else "passed"
                ),
            },
            {
                "id": "qiskit_cost_route_contracts",
                "status": (
                    "not_evaluated" if blocked_cells else "passed"
                ),
            },
            {
                "id": "all_cells_direct_execution",
                "status": (
                    "passed" if direct_count == len(cells) else "blocked"
                ),
                "observed": direct_count,
            },
            *terminal_checks,
        ]
        validation_surface = {
            "stationarity_condition": "always_applied_v1",
            "phase1_cost_term": "always_applied_v1",
            "qiskit_cost_pilot_validation_binding": validation_binding,
            "scientific_execution_status": "not_run",
        }
    elif campaign_id == QISKIT_COST_ALWAYS13_CAMPAIGN_ID:
        direct_count = sum(
            template["execution_fulfillment"]["fulfillment_kind"]
            == "direct_execution_v1"
            for template in execution_templates.values()
        )
        campaign_contract = manifest["qiskit_cost_always13_contract"]
        validation_binding = _digested(
            {
                "schema": (
                    "paper_i_ra_adapt_qiskit_cost_always13_"
                    "validation_binding_v1"
                ),
                "campaign_id": QISKIT_COST_ALWAYS13_CAMPAIGN_ID,
                "bundle_id": QISKIT_COST_ALWAYS13_BUNDLE_ID,
                "bundle_manifest_sha256": manifest["sha256"],
                "source_locks_sha256": normalized_source_locks["sha256"],
                "expected_artifacts_sha256": expected["sha256"],
                "materialization_gate_sha256": (
                    materialization_gate["sha256"]
                ),
                "implementation_source_inventory_sha256": (
                    normalized_source_locks["implementation_sources"][
                        "sha256"
                    ]
                ),
                "campaign_contract_sha256": canonical_sha256(
                    campaign_contract
                ),
                "algorithm_ids": [
                    QISKIT_COST_ALWAYS13_ALGORITHM_ID,
                ],
                "active_gradient_policy": (
                    ACTIVE_GRADIENT_STATIONARY
                ),
                "resource_weighting_scope": (
                    RESOURCE_WEIGHTING_ALL_PHASE
                ),
                "execution_target": (
                    QISKIT_COST_ALWAYS13_EXECUTION_TARGET
                ),
                "semantic_route_ids": list(
                    QISKIT_COST_ALWAYS13_ROUTE_IDS
                ),
                "cell_count": len(cells),
                "direct_execution_cell_count": direct_count,
                "materialization_status": materialization_status,
                "execution_authorized": False,
                "submission_state": SUBMISSION_STATE,
                "submitted": False,
            }
        )
        checks = [
            common_checks[0],
            {
                "id": "exact_qiskit_cost_always13_matrix",
                "status": "passed",
                "observed": {
                    "total": len(cells),
                    "route_ids": list(
                        QISKIT_COST_ALWAYS13_ROUTE_IDS
                    ),
                    "algorithm_ids": [
                        QISKIT_COST_ALWAYS13_ALGORITHM_ID,
                    ],
                    "horizon": QISKIT_COST_ALWAYS13_HORIZON,
                    "execution_target": (
                        QISKIT_COST_ALWAYS13_EXECUTION_TARGET
                    ),
                },
            },
            common_checks[1],
            *common_checks[2:],
            {
                "id": "qiskit_cost_always13_source_derivation",
                "status": (
                    "not_evaluated" if blocked_cells else "passed"
                ),
            },
            {
                "id": "qiskit_cost_always13_route_contract",
                "status": (
                    "not_evaluated" if blocked_cells else "passed"
                ),
            },
            {
                "id": "all_cells_direct_execution",
                "status": (
                    "passed" if direct_count == len(cells) else "blocked"
                ),
                "observed": direct_count,
            },
            *terminal_checks,
        ]
        validation_surface = {
            "stationarity_condition": "always_applied_v1",
            "phase1_cost_term": "always_applied_v1",
            "qiskit_cost_always13_validation_binding": validation_binding,
            "scientific_execution_status": "not_run",
        }
    elif campaign_id == BEAMPRUNE_CAMPAIGN_ID:
        direct_count = sum(
            template["execution_fulfillment"]["fulfillment_kind"]
            == "direct_execution_v1"
            for template in execution_templates.values()
        )
        validation_binding = _digested(
            {
                "schema": (
                    "paper_i_ra_adapt_macro_always_beamprune_lanes_"
                    "validation_binding_v1"
                ),
                "campaign_id": BEAMPRUNE_CAMPAIGN_ID,
                "bundle_id": BEAMPRUNE_BUNDLE_ID,
                "bundle_manifest_sha256": manifest["sha256"],
                "source_locks_sha256": normalized_source_locks["sha256"],
                "expected_artifacts_sha256": expected["sha256"],
                "materialization_gate_sha256": (
                    materialization_gate["sha256"]
                ),
                "implementation_source_inventory_sha256": (
                    normalized_source_locks["implementation_sources"]["sha256"]
                ),
                "campaign_contract_sha256": canonical_sha256(
                    manifest["beamprune_contract"]
                ),
                "execution_target": BEAMPRUNE_EXECUTION_TARGET,
                "semantic_route_ids": [ROUTE_RA_MACRO_ALWAYS],
                "cell_count": len(cells),
                "direct_execution_cell_count": direct_count,
                "materialization_status": materialization_status,
                "execution_authorized": False,
                "submission_state": SUBMISSION_STATE,
                "submitted": False,
            }
        )
        checks = [
            common_checks[0],
            {
                "id": "exact_beamprune_matrix",
                "status": "passed",
                "observed": {
                    "total": len(cells),
                    "arms": [a for a, _l, _p in BEAMPRUNE_ARMS],
                    "horizon": BEAMPRUNE_HORIZON,
                    "execution_target": BEAMPRUNE_EXECUTION_TARGET,
                },
            },
            common_checks[1],
            *common_checks[2:],
            {"id": "beamprune_source_derivation",
             "status": ("not_evaluated" if blocked_cells else "passed")},
            {"id": "beamprune_route_contracts",
             "status": ("not_evaluated" if blocked_cells else "passed")},
            {"id": "all_cells_direct_execution",
             "status": ("passed" if direct_count == len(cells) else "blocked"),
             "observed": direct_count},
            *terminal_checks,
        ]
        validation_surface = {
            "stationarity_condition": "always_applied_v1",
            "phase1_cost_term": "always_applied_v1",
            "beamprune_validation_binding": validation_binding,
            "scientific_execution_status": "not_run",
        }
    elif campaign_id == LANES_ABLATION_CAMPAIGN_ID:
        direct_count = sum(
            template["execution_fulfillment"]["fulfillment_kind"]
            == "direct_execution_v1"
            for template in execution_templates.values()
        )
        campaign_contract = manifest["lanes_ablation_contract"]
        validation_binding = _digested(
            {
                "schema": (
                    "paper_i_ra_adapt_macro_always_lanes_ablation_"
                    "validation_binding_v1"
                ),
                "campaign_id": LANES_ABLATION_CAMPAIGN_ID,
                "bundle_id": LANES_ABLATION_BUNDLE_ID,
                "bundle_manifest_sha256": manifest["sha256"],
                "source_locks_sha256": normalized_source_locks["sha256"],
                "expected_artifacts_sha256": expected["sha256"],
                "materialization_gate_sha256": (
                    materialization_gate["sha256"]
                ),
                "implementation_source_inventory_sha256": (
                    normalized_source_locks["implementation_sources"][
                        "sha256"
                    ]
                ),
                "campaign_contract_sha256": canonical_sha256(
                    campaign_contract
                ),
                "algorithm_ids": [
                    LANES_ABLATION_LANES_ON_ALGORITHM_ID,
                    LANES_ABLATION_LANES_OFF_ALGORITHM_ID,
                ],
                "active_gradient_policy": ACTIVE_GRADIENT_STATIONARY,
                "resource_weighting_scope": RESOURCE_WEIGHTING_ALL_PHASE,
                "execution_target": LANES_ABLATION_EXECUTION_TARGET,
                "semantic_route_ids": [ROUTE_RA_MACRO_ALWAYS],
                "cell_count": len(cells),
                "direct_execution_cell_count": direct_count,
                "materialization_status": materialization_status,
                "execution_authorized": False,
                "submission_state": SUBMISSION_STATE,
                "submitted": False,
            }
        )
        checks = [
            common_checks[0],
            {
                "id": "exact_lanes_ablation_matrix",
                "status": "passed",
                "observed": {
                    "total": len(cells),
                    "route_ids": [ROUTE_RA_MACRO_ALWAYS],
                    "algorithm_ids": [
                        LANES_ABLATION_LANES_ON_ALGORITHM_ID,
                        LANES_ABLATION_LANES_OFF_ALGORITHM_ID,
                    ],
                    "horizon": LANES_ABLATION_HORIZON,
                    "execution_target": LANES_ABLATION_EXECUTION_TARGET,
                },
            },
            common_checks[1],
            *common_checks[2:],
            {
                "id": "lanes_ablation_source_derivation",
                "status": (
                    "not_evaluated" if blocked_cells else "passed"
                ),
            },
            {
                "id": "lanes_ablation_route_contracts",
                "status": (
                    "not_evaluated" if blocked_cells else "passed"
                ),
            },
            {
                "id": "all_cells_direct_execution",
                "status": (
                    "passed" if direct_count == len(cells) else "blocked"
                ),
                "observed": direct_count,
            },
            *terminal_checks,
        ]
        validation_surface = {
            "stationarity_condition": "always_applied_v1",
            "phase1_cost_term": "always_applied_v1",
            "lanes_ablation_validation_binding": validation_binding,
            "scientific_execution_status": "not_run",
        }
    elif campaign_id == QISKIT_COST_ALWAYS6_CAMPAIGN_ID:
        direct_count = sum(
            template["execution_fulfillment"]["fulfillment_kind"]
            == "direct_execution_v1"
            for template in execution_templates.values()
        )
        campaign_contract = manifest["qiskit_cost_always6_contract"]
        validation_binding = _digested(
            {
                "schema": (
                    "paper_i_ra_adapt_qiskit_cost_always6_"
                    "validation_binding_v1"
                ),
                "campaign_id": QISKIT_COST_ALWAYS6_CAMPAIGN_ID,
                "bundle_id": QISKIT_COST_ALWAYS6_BUNDLE_ID,
                "bundle_manifest_sha256": manifest["sha256"],
                "source_locks_sha256": normalized_source_locks["sha256"],
                "expected_artifacts_sha256": expected["sha256"],
                "materialization_gate_sha256": (
                    materialization_gate["sha256"]
                ),
                "implementation_source_inventory_sha256": (
                    normalized_source_locks["implementation_sources"][
                        "sha256"
                    ]
                ),
                "campaign_contract_sha256": canonical_sha256(
                    campaign_contract
                ),
                "algorithm_ids": [QISKIT_COST_ALWAYS13_ALGORITHM_ID],
                "active_gradient_policy": ACTIVE_GRADIENT_STATIONARY,
                "resource_weighting_scope": RESOURCE_WEIGHTING_ALL_PHASE,
                "execution_target": (
                    QISKIT_COST_ALWAYS6_EXECUTION_TARGET
                ),
                "semantic_route_ids": [ROUTE_RA_MACRO_ALWAYS],
                "cell_count": len(cells),
                "direct_execution_cell_count": direct_count,
                "materialization_status": materialization_status,
                "execution_authorized": False,
                "submission_state": SUBMISSION_STATE,
                "submitted": False,
            }
        )
        checks = [
            common_checks[0],
            {
                "id": "exact_qiskit_cost_always6_matrix",
                "status": "passed",
                "observed": {
                    "total": len(cells),
                    "route_ids": [ROUTE_RA_MACRO_ALWAYS],
                    "algorithm_ids": [QISKIT_COST_ALWAYS13_ALGORITHM_ID],
                    "horizon_by_regime": {
                        cell.regime_id: int(cell.horizon)
                        for cell in cells
                    },
                    "execution_target": (
                        QISKIT_COST_ALWAYS6_EXECUTION_TARGET
                    ),
                },
            },
            common_checks[1],
            *common_checks[2:],
            {
                "id": "qiskit_cost_always6_source_derivation",
                "status": (
                    "not_evaluated" if blocked_cells else "passed"
                ),
            },
            {
                "id": "qiskit_cost_always6_route_contracts",
                "status": (
                    "not_evaluated" if blocked_cells else "passed"
                ),
            },
            {
                "id": "all_cells_direct_execution",
                "status": (
                    "passed" if direct_count == len(cells) else "blocked"
                ),
                "observed": direct_count,
            },
            *terminal_checks,
        ]
        validation_surface = {
            "stationarity_condition": "always_applied_v1",
            "phase1_cost_term": "always_applied_v1",
            "qiskit_cost_always6_validation_binding": validation_binding,
            "scientific_execution_status": "not_run",
        }
    elif campaign_id == PHASE3_QISKIT_CAMPAIGN_ID:
        direct_count = sum(
            template["execution_fulfillment"]["fulfillment_kind"]
            == "direct_execution_v1"
            for template in execution_templates.values()
        )
        campaign_contract = manifest[
            "phase3_qiskit_mixed_horizon_contract"
        ]
        validation_binding = _digested(
            {
                "schema": (
                    "paper_i_ra_adapt_phase3_qiskit_mixed_horizon_"
                    "validation_binding_v1"
                ),
                "campaign_id": PHASE3_QISKIT_CAMPAIGN_ID,
                "bundle_id": PHASE3_QISKIT_BUNDLE_ID,
                "bundle_manifest_sha256": manifest["sha256"],
                "source_locks_sha256": normalized_source_locks["sha256"],
                "expected_artifacts_sha256": expected["sha256"],
                "materialization_gate_sha256": (
                    materialization_gate["sha256"]
                ),
                "implementation_source_inventory_sha256": (
                    normalized_source_locks["implementation_sources"][
                        "sha256"
                    ]
                ),
                "campaign_contract_sha256": canonical_sha256(
                    campaign_contract
                ),
                "algorithm_ids": [
                    RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_ALGORITHM_ID
                ],
                "active_gradient_policy": ACTIVE_GRADIENT_STATIONARY,
                "resource_weighting_scope": RESOURCE_WEIGHTING_ALL_PHASE,
                "execution_target": PHASE3_QISKIT_EXECUTION_TARGET,
                "semantic_route_ids": list(PHASE3_QISKIT_ROUTE_IDS),
                "source_parent_route_profile": (
                    PHASE3_QISKIT_PAGE7_PARENT_ROUTE_PROFILE
                ),
                "source_parent_contract_sha256": (
                    PHASE3_QISKIT_PAGE7_PARENT_CONTRACT_SHA256
                ),
                "cell_count": len(cells),
                "direct_execution_cell_count": direct_count,
                "materialization_status": materialization_status,
                "execution_authorized": False,
                "submission_state": SUBMISSION_STATE,
                "submitted": False,
            }
        )
        checks = [
            common_checks[0],
            {
                "id": "exact_phase3_qiskit_mixed_horizon_matrix",
                "status": "passed",
                "observed": {
                    "total": len(cells),
                    "route_ids": list(PHASE3_QISKIT_ROUTE_IDS),
                    "algorithm_ids": [
                        RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_ALGORITHM_ID
                    ],
                    "weak_holstein_horizon": (
                        PHASE3_QISKIT_WEAK_HOLSTEIN_HORIZON
                    ),
                    "strong_holstein_horizon": (
                        PHASE3_QISKIT_STRONG_HOLSTEIN_HORIZON
                    ),
                    "execution_target": PHASE3_QISKIT_EXECUTION_TARGET,
                },
            },
            common_checks[1],
            *common_checks[2:],
            {
                "id": "phase3_qiskit_source_derivations",
                "status": (
                    "not_evaluated" if blocked_cells else "passed"
                ),
            },
            {
                "id": "phase3_qiskit_route_contracts",
                "status": (
                    "not_evaluated" if blocked_cells else "passed"
                ),
            },
            {
                "id": "all_cells_direct_execution",
                "status": (
                    "passed" if direct_count == len(cells) else "blocked"
                ),
                "observed": direct_count,
            },
            *terminal_checks,
        ]
        validation_surface = {
            "stationarity_condition": "always_applied_v1",
            "phase1_cost_term": "always_applied_v1",
            "phase3_qiskit_validation_binding": validation_binding,
            "scientific_execution_status": "not_run",
        }
    else:
        raise BundleMaterializationError(
            f"Unknown bundle campaign id: {campaign_id!r}."
        )
    validation = _digested(
        {
            "schema": VALIDATION_REPORT_SCHEMA,
            "bundle_id": bundle_id,
            "campaign_id": campaign_id,
            "materialization_status": materialization_status,
            "execution_authorized": False,
            "submission_state": SUBMISSION_STATE,
            "submitted": False,
            "checks": checks,
            "materialization_binding": materialization_binding,
            **validation_surface,
        }
    )

    files: dict[str, Mapping[str, Any]] = {
        "bundle_manifest.json": manifest,
        "source_locks.json": normalized_source_locks,
        "expected_artifacts.json": expected,
        "validation_report.json": validation,
    }
    for cell_id, protocol in protocols.items():
        files[f"protocols/{cell_id}.json"] = protocol
    for cell_id, template in execution_templates.items():
        files[f"execution_templates/{cell_id}.json"] = template
    receipt = MaterializedBundleReceipt(
        bundle_id=bundle_id,
        bundle_path=bundle_dir,
        bundle_manifest_sha256=manifest_sha,
        source_locks_sha256=str(normalized_source_locks["sha256"]),
        expected_artifacts_sha256=str(expected["sha256"]),
        validation_report_sha256=str(validation["sha256"]),
        cell_count=len(cells),
        materialization_status=materialization_status,
    )
    return files, receipt


def _write_canonical_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _load_canonical_digested_mapping(
    path: Path,
    *,
    label: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise BundleMaterializationError(
            f"Required {label} is missing: {path}."
        )
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleMaterializationError(
            f"{label} is not valid UTF-8 JSON: {path}."
        ) from exc
    if not isinstance(payload, Mapping):
        raise BundleMaterializationError(
            f"{label} must contain a JSON object."
        )
    normalized = dict(payload)
    if raw != canonical_json_bytes(normalized) + b"\n":
        raise BundleMaterializationError(
            f"{label} is not canonical digested JSON."
        )
    _verify_digest(normalized, label=label)
    return normalized


def _cell_from_manifest_row(
    row: Mapping[str, Any],
) -> BundleCellSpec:
    return BundleCellSpec(
        cell_id=str(row["cell_id"]),
        stage=str(row["stage"]),
        regime_id=str(row["regime_id"]),
        nph=int(row["nph"]),
        route_id=str(row["route_id"]),
        algorithm_id=str(row["algorithm_id"]),
        selector_family=str(row["selector_family"]),
        candidate_representation=str(
            row["candidate_representation"]
        ),
        horizon=(
            None if row.get("horizon") is None else int(row["horizon"])
        ),
        source_lock_id=str(row["source_lock_id"]),
        preservation_contract_id=(
            None
            if row.get("preservation_contract_id") is None
            else str(row["preservation_contract_id"])
        ),
    )


def load_validated_bundle_protocol(
    protocol_path: str | Path,
) -> ResolvedRAAdaptProtocol:
    """Load one protocol only after validating its complete bundle binding.

    A protocol JSON loaded directly through ``contracts`` is suitable for
    inspection but intentionally lacks the in-memory execution capability.
    This loader validates the canonical protocol, bundle manifest,
    source-lock manifest, expected-artifact index, and materialization report,
    then binds the final protocol digest to the private authority accepted by
    the RA and Append facades.
    """

    path = Path(protocol_path).expanduser().resolve()
    if path.parent.name != "protocols":
        raise BundleMaterializationError(
            "A validated protocol must live under <bundle>/protocols/."
        )
    bundle_dir = path.parent.parent
    manifest = _load_canonical_digested_mapping(
        bundle_dir / "bundle_manifest.json",
        label="bundle manifest",
    )
    source_locks = _load_canonical_digested_mapping(
        bundle_dir / "source_locks.json",
        label="source-lock manifest",
    )
    expected = _load_canonical_digested_mapping(
        bundle_dir / "expected_artifacts.json",
        label="expected-artifact index",
    )
    validation = _load_canonical_digested_mapping(
        bundle_dir / "validation_report.json",
        label="bundle validation report",
    )
    protocol_payload = _load_canonical_digested_mapping(
        path,
        label="resolved protocol",
    )

    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise BundleMaterializationError("Unknown bundle manifest schema.")
    if source_locks.get("schema") != SOURCE_LOCK_SCHEMA:
        raise BundleMaterializationError("Unknown source-lock schema.")
    if expected.get("schema") != EXPECTED_ARTIFACTS_SCHEMA:
        raise BundleMaterializationError(
            "Unknown expected-artifact schema."
        )
    if validation.get("schema") != VALIDATION_REPORT_SCHEMA:
        raise BundleMaterializationError(
            "Unknown bundle validation-report schema."
        )
    campaign_id = str(manifest.get("campaign_id", ""))
    if campaign_id == STUDY_ID:
        expected_run_class = RUN_CLASS
    elif campaign_id == CORE_CAMPAIGN_ID:
        expected_run_class = CORE_RUN_CLASS
    elif campaign_id == FACTORIAL_CAMPAIGN_ID:
        expected_run_class = FACTORIAL_RUN_CLASS
    elif campaign_id == GLOBAL_SINGLETON_CAMPAIGN_ID:
        expected_run_class = GLOBAL_SINGLETON_RUN_CLASS
    elif campaign_id == QISKIT_COST_PILOT_CAMPAIGN_ID:
        expected_run_class = QISKIT_COST_PILOT_RUN_CLASS
    elif campaign_id == QISKIT_COST_ALWAYS13_CAMPAIGN_ID:
        expected_run_class = QISKIT_COST_ALWAYS13_RUN_CLASS
    elif campaign_id == QISKIT_COST_ALWAYS6_CAMPAIGN_ID:
        expected_run_class = QISKIT_COST_ALWAYS6_RUN_CLASS
    elif campaign_id == LANES_ABLATION_CAMPAIGN_ID:
        expected_run_class = LANES_ABLATION_RUN_CLASS
    elif campaign_id == BEAMPRUNE_CAMPAIGN_ID:
        expected_run_class = BEAMPRUNE_RUN_CLASS
    elif campaign_id == PHASE3_QISKIT_CAMPAIGN_ID:
        expected_run_class = PHASE3_QISKIT_RUN_CLASS
    else:
        raise BundleMaterializationError(
            "Unknown materialized bundle campaign."
        )
    if manifest.get("run_class") != expected_run_class:
        raise BundleMaterializationError(
            "Materialized bundle run class drifted."
        )
    bundle_id = str(manifest.get("bundle_id", ""))
    if not bundle_id or any(
        str(payload.get("bundle_id", "")) != bundle_id
        for payload in (expected, validation, protocol_payload)
    ):
        raise BundleMaterializationError(
            "Bundle id drifted across materialized files."
        )
    if bundle_dir.name != bundle_id:
        raise BundleMaterializationError(
            "Bundle directory name does not match the bundle id."
        )
    if validation.get("materialization_status") != "passed":
        raise BundleMaterializationError(
            "Protocol execution requires a passed bundle materialization."
        )
    if source_locks.get("all_required_files_verified") is not True:
        raise BundleMaterializationError(
            "Protocol source locks were not verified at materialization."
        )
    manifest_source_locks = manifest.get("source_locks")
    if (
        not isinstance(manifest_source_locks, Mapping)
        or manifest_source_locks.get("path") != "source_locks.json"
        or manifest_source_locks.get("sha256")
        != source_locks.get("sha256")
    ):
        raise BundleMaterializationError(
            "Bundle manifest is not bound to the source-lock manifest."
        )

    raw_rows = manifest.get("cells")
    if (
        not isinstance(raw_rows, list)
        or not raw_rows
        or any(not isinstance(row, Mapping) for row in raw_rows)
    ):
        raise BundleMaterializationError(
            "Bundle manifest has no ordered cell list."
        )
    cells = tuple(_cell_from_manifest_row(row) for row in raw_rows)
    if (
        len({cell.cell_id for cell in cells}) != len(cells)
        or int(manifest.get("cell_count", -1)) != len(cells)
        or int(expected.get("cell_count", -1)) != len(cells)
    ):
        raise BundleMaterializationError(
            "Bundle ordered-cell identity or count drifted."
        )
    matching_cells = [cell for cell in cells if cell.cell_id == path.stem]
    if len(matching_cells) != 1:
        raise BundleMaterializationError(
            "Protocol path does not identify exactly one bundle cell."
        )
    cell = matching_cells[0]
    if cell.horizon is None:
        raise BundleMaterializationError(
            "A blocked protocol cannot receive execution authority."
        )

    expected_cells = expected.get("cells")
    if (
        not isinstance(expected_cells, Mapping)
        or set(expected_cells) != {item.cell_id for item in cells}
    ):
        raise BundleMaterializationError(
            "Expected-artifact cell set drifted from the manifest."
        )

    active_gradient_policy = str(
        manifest.get("active_gradient_policy", "")
    )
    resource_weighting_scope = str(
        manifest.get("resource_weighting_scope", "")
    )
    if campaign_id in {STUDY_ID, CORE_CAMPAIGN_ID}:
        if resource_weighting_scope != RESOURCE_WEIGHTING_LATE:
            raise BundleMaterializationError(
                "This campaign accepts only late-weighting materialized "
                "contracts."
            )
    elif campaign_id == FACTORIAL_CAMPAIGN_ID:
        expected_gradient, expected_resource = (
            _factorial_policy_for_bundle(bundle_id)
        )
        if (
            active_gradient_policy != expected_gradient
            or resource_weighting_scope != expected_resource
        ):
            raise BundleMaterializationError(
                "Factorial bundle id and policy axes do not match."
            )
    elif campaign_id == GLOBAL_SINGLETON_CAMPAIGN_ID and (
        bundle_id != GLOBAL_SINGLETON_BUNDLE_ID
        or active_gradient_policy != ACTIVE_GRADIENT_STATIONARY
        or resource_weighting_scope != RESOURCE_WEIGHTING_ALL_PHASE
    ):
        raise BundleMaterializationError(
            "Global-singleton bundle id or fixed policy contract drifted."
        )
    elif campaign_id == QISKIT_COST_PILOT_CAMPAIGN_ID and (
        bundle_id != QISKIT_COST_PILOT_BUNDLE_ID
        or active_gradient_policy != ACTIVE_GRADIENT_STATIONARY
        or resource_weighting_scope != RESOURCE_WEIGHTING_ALL_PHASE
        or manifest.get("execution_target")
        != QISKIT_COST_PILOT_EXECUTION_TARGET
    ):
        raise BundleMaterializationError(
            "Qiskit-cost pilot bundle id or fixed policy contract drifted."
        )
    elif campaign_id == QISKIT_COST_ALWAYS13_CAMPAIGN_ID and (
        bundle_id != QISKIT_COST_ALWAYS13_BUNDLE_ID
        or active_gradient_policy != ACTIVE_GRADIENT_STATIONARY
        or resource_weighting_scope != RESOURCE_WEIGHTING_ALL_PHASE
        or manifest.get("execution_target")
        != QISKIT_COST_ALWAYS13_EXECUTION_TARGET
    ):
        raise BundleMaterializationError(
            "Qiskit-cost always13 bundle id or fixed policy contract "
            "drifted."
        )
    elif campaign_id == PHASE3_QISKIT_CAMPAIGN_ID and (
        bundle_id != PHASE3_QISKIT_BUNDLE_ID
        or active_gradient_policy != ACTIVE_GRADIENT_STATIONARY
        or resource_weighting_scope != RESOURCE_WEIGHTING_ALL_PHASE
        or manifest.get("execution_target")
        != PHASE3_QISKIT_EXECUTION_TARGET
    ):
        raise BundleMaterializationError(
            "Phase-III-Qiskit bundle id or fixed policy contract drifted."
        )
    cell_locks = source_locks.get("cell_locks")
    if not isinstance(cell_locks, Mapping):
        raise BundleMaterializationError(
            "Source-lock manifest has no cell-lock mapping."
        )

    protocols: dict[str, Mapping[str, Any]] = {}
    execution_templates: dict[str, Mapping[str, Any]] = {}
    for row, materialized_cell in zip(raw_rows, cells, strict=True):
        relative_protocol_path = (
            f"protocols/{materialized_cell.cell_id}.json"
        )
        relative_template_path = (
            f"execution_templates/{materialized_cell.cell_id}.json"
        )
        if (
            row.get("protocol_path") != relative_protocol_path
            or row.get("execution_template_path")
            != relative_template_path
        ):
            raise BundleMaterializationError(
                f"Bundle cell path drifted for "
                f"{materialized_cell.cell_id}."
            )
        materialized_protocol = (
            protocol_payload
            if materialized_cell.cell_id == cell.cell_id
            else _load_canonical_digested_mapping(
                bundle_dir / relative_protocol_path,
                label=(
                    "resolved protocol "
                    f"{materialized_cell.cell_id}"
                ),
            )
        )
        execution_template = _load_canonical_digested_mapping(
            bundle_dir / relative_template_path,
            label=(
                "execution template "
                f"{materialized_cell.cell_id}"
            ),
        )
        expected_cell = expected_cells.get(materialized_cell.cell_id)
        expected_protocol = (
            expected_cell.get("protocol")
            if isinstance(expected_cell, Mapping)
            else None
        )
        expected_template = (
            expected_cell.get("execution_template")
            if isinstance(expected_cell, Mapping)
            else None
        )
        if (
            not isinstance(expected_protocol, Mapping)
            or expected_protocol.get("path")
            != relative_protocol_path
            or expected_protocol.get("sha256")
            != materialized_protocol.get("sha256")
            or expected_protocol.get("status") != "resolved"
            or not isinstance(expected_template, Mapping)
            or expected_template.get("path") != relative_template_path
            or expected_template.get("sha256")
            != execution_template.get("sha256")
        ):
            raise BundleMaterializationError(
                "Expected-artifact index is not bound to cell "
                f"{materialized_cell.cell_id}."
            )
        refs_for_cell = _source_lock_refs(
            source_locks, cell=materialized_cell
        )
        cell_source_lock = cell_locks.get(
            materialized_cell.source_lock_id
        )
        if not isinstance(cell_source_lock, Mapping):
            raise BundleMaterializationError(
                "Protocol cell has no source-lock receipt: "
                f"{materialized_cell.cell_id}."
            )
        _validate_protocol_payload(
            materialized_protocol,
            cell=materialized_cell,
            bundle_id=bundle_id,
            bundle_manifest_sha256=str(manifest["sha256"]),
            active_gradient_policy=active_gradient_policy,
            resource_weighting_scope=resource_weighting_scope,
            source_lock_refs=refs_for_cell,
            cell_source_lock=cell_source_lock,
            source_locks_sha256=str(source_locks["sha256"]),
        )
        protocols[materialized_cell.cell_id] = materialized_protocol
        execution_templates[materialized_cell.cell_id] = (
            execution_template
        )

    _validate_macro_pool_hash_equality(protocols, cells)
    _validate_singleton_pool_contracts(
        protocols,
        cells,
        expected_global_cells_per_group=(
            1
            if campaign_id
            in {
                QISKIT_COST_PILOT_CAMPAIGN_ID,
                PHASE3_QISKIT_CAMPAIGN_ID,
            }
            else 2
        ),
    )
    if campaign_id == GLOBAL_SINGLETON_CAMPAIGN_ID:
        _validate_global_singleton_source_lock_matrix(
            source_locks=source_locks,
            cells=cells,
        )
        _validate_global_singleton_cross_arm_equality(protocols, cells)
    if campaign_id == QISKIT_COST_PILOT_CAMPAIGN_ID:
        _validate_qiskit_cost_pilot_protocols(protocols, cells)
    if campaign_id == QISKIT_COST_ALWAYS13_CAMPAIGN_ID:
        _validate_qiskit_cost_always13_protocols(protocols, cells)
    if campaign_id == PHASE3_QISKIT_CAMPAIGN_ID:
        _validate_phase3_qiskit_protocols(protocols, cells)
    recomputed_gate = _digested(
        _validate_paper_i_materialization_gate(
            manifest=manifest,
            normalized_source_locks=source_locks,
            protocols=protocols,
            execution_templates=execution_templates,
            expected=expected,
            cells=cells,
            blocked_cells=(),
            campaign_id=campaign_id,
        )
    )
    checks = validation.get("checks")
    checks_by_id = {
        str(check.get("id")): check
        for check in checks
        if isinstance(check, Mapping)
    } if isinstance(checks, list) else {}
    if campaign_id == STUDY_ID:
        required_passed_checks = {
            "bundle_schema_and_digest",
            "finite_cell_matrix",
            "source_locks_exact_bytes",
            "validation_horizon",
            "resolved_protocol_contracts",
            "macro_pool_hash_equality",
            "singleton_pool_exposure_contracts",
            "study1_append_shared_execution_dedupe",
            "protocol_execution_separation",
            "paper_i_run_materialization_gate",
        }
        if validation.get("objective_execution_gates") != (
            _objective_execution_gates(
                active_gradient_policy=active_gradient_policy
            )
        ):
            raise BundleMaterializationError(
                "Bundle preservation/objective execution gates drifted."
            )
    elif campaign_id == CORE_CAMPAIGN_ID:
        required_passed_checks = {
            "bundle_schema_and_digest",
            "exact_core_cell_matrix",
            "source_locks_exact_bytes",
            "resolved_protocol_contracts",
            "macro_pool_hash_equality",
            "singleton_pool_exposure_contracts",
            "all_cells_direct_execution",
            "protocol_execution_separation",
            "paper_i_run_materialization_gate",
        }
        forbidden_validation_fields = {
            "execution_progression_status",
            "objective_execution_gates",
            "user_decision_required_after_study_1",
        }
        if forbidden_validation_fields.intersection(validation):
            raise BundleMaterializationError(
                "Stationary-core validation contains obsolete Study-1 "
                "progression fields."
            )
    elif campaign_id == FACTORIAL_CAMPAIGN_ID:
        required_passed_checks = {
            "bundle_schema_and_digest",
            "exact_factorial_arm_matrix",
            "source_locks_exact_bytes",
            "resolved_protocol_contracts",
            "macro_pool_hash_equality",
            "singleton_pool_exposure_contracts",
            "all_cells_direct_execution",
            "protocol_execution_separation",
            "paper_i_run_materialization_gate",
        }
        forbidden_validation_fields = {
            "execution_progression_status",
            "objective_execution_gates",
            "user_decision_required_after_study_1",
            "core_validation_binding",
        }
        if forbidden_validation_fields.intersection(validation):
            raise BundleMaterializationError(
                "Factorial validation contains an unrelated campaign field."
            )
    elif campaign_id == GLOBAL_SINGLETON_CAMPAIGN_ID:
        required_passed_checks = {
            "bundle_schema_and_digest",
            "exact_global_singleton_insertion_matrix",
            "source_locks_exact_bytes",
            "resolved_protocol_contracts",
            "macro_pool_hash_equality",
            "singleton_pool_exposure_contracts",
            "global_singleton_source_lock_pair_equality",
            "global_singleton_cross_arm_scientific_equality",
            "all_cells_direct_execution",
            "protocol_execution_separation",
            "paper_i_run_materialization_gate",
        }
        forbidden_validation_fields = {
            "execution_progression_status",
            "objective_execution_gates",
            "user_decision_required_after_study_1",
            "core_validation_binding",
            "factorial_validation_binding",
        }
        if forbidden_validation_fields.intersection(validation):
            raise BundleMaterializationError(
                "Global-singleton validation contains an unrelated "
                "campaign field."
            )
    elif campaign_id == QISKIT_COST_PILOT_CAMPAIGN_ID:
        required_passed_checks = {
            "bundle_schema_and_digest",
            "exact_qiskit_cost_plateau_pilot_matrix",
            "source_locks_exact_bytes",
            "resolved_protocol_contracts",
            "macro_pool_hash_equality",
            "singleton_pool_exposure_contracts",
            "qiskit_cost_pilot_source_derivations",
            "qiskit_cost_route_contracts",
            "all_cells_direct_execution",
            "protocol_execution_separation",
            "paper_i_run_materialization_gate",
        }
        forbidden_validation_fields = {
            "execution_progression_status",
            "objective_execution_gates",
            "user_decision_required_after_study_1",
            "core_validation_binding",
            "factorial_validation_binding",
            "global_singleton_insertion_validation_binding",
        }
        if forbidden_validation_fields.intersection(validation):
            raise BundleMaterializationError(
                "Qiskit-cost pilot validation contains an unrelated "
                "campaign field."
            )
    elif campaign_id == QISKIT_COST_ALWAYS13_CAMPAIGN_ID:
        required_passed_checks = {
            "bundle_schema_and_digest",
            "exact_qiskit_cost_always13_matrix",
            "source_locks_exact_bytes",
            "resolved_protocol_contracts",
            "macro_pool_hash_equality",
            "singleton_pool_exposure_contracts",
            "qiskit_cost_always13_source_derivation",
            "qiskit_cost_always13_route_contract",
            "all_cells_direct_execution",
            "protocol_execution_separation",
            "paper_i_run_materialization_gate",
        }
        forbidden_validation_fields = {
            "execution_progression_status",
            "objective_execution_gates",
            "user_decision_required_after_study_1",
            "core_validation_binding",
            "factorial_validation_binding",
            "global_singleton_insertion_validation_binding",
            "qiskit_cost_pilot_validation_binding",
        }
        if forbidden_validation_fields.intersection(validation):
            raise BundleMaterializationError(
                "Qiskit-cost always13 validation contains an unrelated "
                "campaign field."
            )
    elif campaign_id == BEAMPRUNE_CAMPAIGN_ID:
        required_passed_checks = {
            "bundle_schema_and_digest",
            "exact_beamprune_matrix",
            "source_locks_exact_bytes",
            "resolved_protocol_contracts",
            "macro_pool_hash_equality",
            "singleton_pool_exposure_contracts",
            "beamprune_source_derivation",
            "beamprune_route_contracts",
            "all_cells_direct_execution",
            "protocol_execution_separation",
            "paper_i_run_materialization_gate",
        }
        forbidden_validation_fields = {
            "core_validation_binding",
            "lanes_ablation_validation_binding",
            "qiskit_cost_always6_validation_binding",
        }
        if forbidden_validation_fields.intersection(validation):
            raise BundleMaterializationError(
                "Beam+prune validation contains an unrelated campaign field."
            )
    elif campaign_id == LANES_ABLATION_CAMPAIGN_ID:
        required_passed_checks = {
            "bundle_schema_and_digest",
            "exact_lanes_ablation_matrix",
            "source_locks_exact_bytes",
            "resolved_protocol_contracts",
            "macro_pool_hash_equality",
            "singleton_pool_exposure_contracts",
            "lanes_ablation_source_derivation",
            "lanes_ablation_route_contracts",
            "all_cells_direct_execution",
            "protocol_execution_separation",
            "paper_i_run_materialization_gate",
        }
        forbidden_validation_fields = {
            "execution_progression_status",
            "objective_execution_gates",
            "user_decision_required_after_study_1",
            "core_validation_binding",
            "factorial_validation_binding",
            "global_singleton_insertion_validation_binding",
            "qiskit_cost_pilot_validation_binding",
            "qiskit_cost_always13_validation_binding",
            "qiskit_cost_always6_validation_binding",
        }
        if forbidden_validation_fields.intersection(validation):
            raise BundleMaterializationError(
                "Lanes-ablation validation contains an unrelated campaign "
                "field."
            )
    elif campaign_id == QISKIT_COST_ALWAYS6_CAMPAIGN_ID:
        required_passed_checks = {
            "bundle_schema_and_digest",
            "exact_qiskit_cost_always6_matrix",
            "source_locks_exact_bytes",
            "resolved_protocol_contracts",
            "macro_pool_hash_equality",
            "singleton_pool_exposure_contracts",
            "qiskit_cost_always6_source_derivation",
            "qiskit_cost_always6_route_contracts",
            "all_cells_direct_execution",
            "protocol_execution_separation",
            "paper_i_run_materialization_gate",
        }
        forbidden_validation_fields = {
            "execution_progression_status",
            "objective_execution_gates",
            "user_decision_required_after_study_1",
            "core_validation_binding",
            "factorial_validation_binding",
            "global_singleton_insertion_validation_binding",
            "qiskit_cost_pilot_validation_binding",
            "qiskit_cost_always13_validation_binding",
        }
        if forbidden_validation_fields.intersection(validation):
            raise BundleMaterializationError(
                "Qiskit-cost always6 validation contains an unrelated "
                "campaign field."
            )
    elif campaign_id == PHASE3_QISKIT_CAMPAIGN_ID:
        required_passed_checks = {
            "bundle_schema_and_digest",
            "exact_phase3_qiskit_mixed_horizon_matrix",
            "source_locks_exact_bytes",
            "resolved_protocol_contracts",
            "macro_pool_hash_equality",
            "singleton_pool_exposure_contracts",
            "phase3_qiskit_source_derivations",
            "phase3_qiskit_route_contracts",
            "all_cells_direct_execution",
            "protocol_execution_separation",
            "paper_i_run_materialization_gate",
        }
        forbidden_validation_fields = {
            "execution_progression_status",
            "objective_execution_gates",
            "user_decision_required_after_study_1",
            "core_validation_binding",
            "factorial_validation_binding",
            "global_singleton_insertion_validation_binding",
            "qiskit_cost_pilot_validation_binding",
            "qiskit_cost_always13_validation_binding",
        }
        if forbidden_validation_fields.intersection(validation):
            raise BundleMaterializationError(
                "Phase-III-Qiskit validation contains an unrelated "
                "campaign field."
            )
    else:
        raise BundleMaterializationError(
            f"Unknown bundle campaign id: {campaign_id!r}."
        )
    if set(checks_by_id) != required_passed_checks or any(
        checks_by_id[check_id].get("status") != "passed"
        for check_id in required_passed_checks
    ):
        raise BundleMaterializationError(
            "Bundle validation report does not contain the passed "
            "materialization check set."
        )
    if campaign_id == CORE_CAMPAIGN_ID:
        expected_core_binding = _digested(
            {
                "schema": (
                    "paper_i_ra_adapt_stationary_core_validation_binding_v1"
                ),
                "campaign_id": CORE_CAMPAIGN_ID,
                "bundle_id": CORE_BUNDLE_ID,
                "bundle_manifest_sha256": manifest["sha256"],
                "source_locks_sha256": source_locks["sha256"],
                "expected_artifacts_sha256": expected["sha256"],
                "materialization_gate_sha256": recomputed_gate["sha256"],
                "implementation_source_inventory_sha256": (
                    source_locks["implementation_sources"]["sha256"]
                ),
                "stationarity_selection_authority_sha256": (
                    CORE_SELECTION_AUTHORITY_SHA256
                ),
                "semantic_route_ids": list(
                    (*MACRO_ROUTE_IDS, *SINGLETON_CORE_ROUTE_IDS)
                ),
                "cell_count": len(cells),
                "direct_execution_cell_count": len(cells),
                "materialization_status": "passed",
                "p3_execution_receipt_required": True,
                "execution_authorized": False,
                "submission_state": SUBMISSION_STATE,
                "submitted": False,
            }
        )
        serialized_core_binding = validation.get(
            "core_validation_binding"
        )
        if not isinstance(serialized_core_binding, Mapping):
            raise BundleMaterializationError(
                "Stationary-core validation binding is missing."
            )
        _verify_digest(
            serialized_core_binding,
            label="stationary-core validation binding",
        )
        if dict(serialized_core_binding) != expected_core_binding:
            raise BundleMaterializationError(
                "Stationary-core validation binding drifted."
            )
    if campaign_id == FACTORIAL_CAMPAIGN_ID:
        expected_factorial_binding = _digested(
            {
                "schema": (
                    "paper_i_ra_adapt_always_stationarity_phase1_cost_"
                    "factorial_validation_binding_v1"
                ),
                "campaign_id": FACTORIAL_CAMPAIGN_ID,
                "bundle_id": bundle_id,
                "bundle_manifest_sha256": manifest["sha256"],
                "source_locks_sha256": source_locks["sha256"],
                "expected_artifacts_sha256": expected["sha256"],
                "materialization_gate_sha256": recomputed_gate["sha256"],
                "implementation_source_inventory_sha256": (
                    source_locks["implementation_sources"]["sha256"]
                ),
                "factorial_arm_contract_sha256": canonical_sha256(
                    manifest["factorial_arm_contract"]
                ),
                "active_gradient_policy": active_gradient_policy,
                "resource_weighting_scope": resource_weighting_scope,
                "semantic_route_ids": [
                    ROUTE_RA_MACRO_ALWAYS,
                    ROUTE_RA_SINGLETON_ALWAYS,
                ],
                "cell_count": len(cells),
                "direct_execution_cell_count": len(cells),
                "materialization_status": "passed",
                "execution_authorized": False,
                "submission_state": SUBMISSION_STATE,
                "submitted": False,
            }
        )
        serialized_factorial_binding = validation.get(
            "factorial_validation_binding"
        )
        if not isinstance(serialized_factorial_binding, Mapping):
            raise BundleMaterializationError(
                "Factorial validation binding is missing."
            )
        _verify_digest(
            serialized_factorial_binding,
            label="factorial validation binding",
        )
        if dict(serialized_factorial_binding) != (
            expected_factorial_binding
        ):
            raise BundleMaterializationError(
                "Factorial validation binding drifted."
            )
    if campaign_id == GLOBAL_SINGLETON_CAMPAIGN_ID:
        expected_global_binding = _digested(
            {
                "schema": (
                    "paper_i_ra_adapt_global_singleton_insertion_"
                    "validation_binding_v1"
                ),
                "campaign_id": GLOBAL_SINGLETON_CAMPAIGN_ID,
                "bundle_id": GLOBAL_SINGLETON_BUNDLE_ID,
                "bundle_manifest_sha256": manifest["sha256"],
                "source_locks_sha256": source_locks["sha256"],
                "expected_artifacts_sha256": expected["sha256"],
                "materialization_gate_sha256": recomputed_gate["sha256"],
                "implementation_source_inventory_sha256": (
                    source_locks["implementation_sources"]["sha256"]
                ),
                "campaign_contract_sha256": canonical_sha256(
                    manifest["global_singleton_insertion_contract"]
                ),
                "candidate_adapter_id": (
                    GLOBAL_SINGLE_PAULI_ADAPTER_ID
                ),
                "active_gradient_policy": (
                    ACTIVE_GRADIENT_STATIONARY
                ),
                "resource_weighting_scope": (
                    RESOURCE_WEIGHTING_ALL_PHASE
                ),
                "semantic_route_ids": list(
                    GLOBAL_SINGLETON_INSERTION_ROUTE_IDS
                ),
                "cell_count": len(cells),
                "direct_execution_cell_count": len(cells),
                "materialization_status": "passed",
                "execution_authorized": False,
                "submission_state": SUBMISSION_STATE,
                "submitted": False,
            }
        )
        serialized_global_binding = validation.get(
            "global_singleton_insertion_validation_binding"
        )
        if not isinstance(serialized_global_binding, Mapping):
            raise BundleMaterializationError(
                "Global-singleton validation binding is missing."
            )
        _verify_digest(
            serialized_global_binding,
            label="global-singleton validation binding",
        )
        if dict(serialized_global_binding) != expected_global_binding:
            raise BundleMaterializationError(
                "Global-singleton validation binding drifted."
            )
    if campaign_id == QISKIT_COST_PILOT_CAMPAIGN_ID:
        expected_pilot_binding = _digested(
            {
                "schema": (
                    "paper_i_ra_adapt_qiskit_cost_plateau_pilot_"
                    "validation_binding_v1"
                ),
                "campaign_id": QISKIT_COST_PILOT_CAMPAIGN_ID,
                "bundle_id": QISKIT_COST_PILOT_BUNDLE_ID,
                "bundle_manifest_sha256": manifest["sha256"],
                "source_locks_sha256": source_locks["sha256"],
                "expected_artifacts_sha256": expected["sha256"],
                "materialization_gate_sha256": recomputed_gate["sha256"],
                "implementation_source_inventory_sha256": (
                    source_locks["implementation_sources"]["sha256"]
                ),
                "campaign_contract_sha256": canonical_sha256(
                    manifest["qiskit_cost_plateau_pilot_contract"]
                ),
                "algorithm_ids": [
                    QISKIT_COST_PILOT_MACRO_ALGORITHM_ID,
                    QISKIT_COST_PILOT_GLOBAL_SINGLETON_ALGORITHM_ID,
                ],
                "active_gradient_policy": (
                    ACTIVE_GRADIENT_STATIONARY
                ),
                "resource_weighting_scope": (
                    RESOURCE_WEIGHTING_ALL_PHASE
                ),
                "execution_target": QISKIT_COST_PILOT_EXECUTION_TARGET,
                "semantic_route_ids": list(
                    QISKIT_COST_PILOT_ROUTE_IDS
                ),
                "cell_count": len(cells),
                "direct_execution_cell_count": len(cells),
                "materialization_status": "passed",
                "execution_authorized": False,
                "submission_state": SUBMISSION_STATE,
                "submitted": False,
            }
        )
        serialized_pilot_binding = validation.get(
            "qiskit_cost_pilot_validation_binding"
        )
        if not isinstance(serialized_pilot_binding, Mapping):
            raise BundleMaterializationError(
                "Qiskit-cost pilot validation binding is missing."
            )
        _verify_digest(
            serialized_pilot_binding,
            label="Qiskit-cost pilot validation binding",
        )
        if dict(serialized_pilot_binding) != expected_pilot_binding:
            raise BundleMaterializationError(
                "Qiskit-cost pilot validation binding drifted."
            )
    if campaign_id == QISKIT_COST_ALWAYS13_CAMPAIGN_ID:
        expected_always13_binding = _digested(
            {
                "schema": (
                    "paper_i_ra_adapt_qiskit_cost_always13_"
                    "validation_binding_v1"
                ),
                "campaign_id": QISKIT_COST_ALWAYS13_CAMPAIGN_ID,
                "bundle_id": QISKIT_COST_ALWAYS13_BUNDLE_ID,
                "bundle_manifest_sha256": manifest["sha256"],
                "source_locks_sha256": source_locks["sha256"],
                "expected_artifacts_sha256": expected["sha256"],
                "materialization_gate_sha256": recomputed_gate["sha256"],
                "implementation_source_inventory_sha256": (
                    source_locks["implementation_sources"]["sha256"]
                ),
                "campaign_contract_sha256": canonical_sha256(
                    manifest["qiskit_cost_always13_contract"]
                ),
                "algorithm_ids": [
                    QISKIT_COST_ALWAYS13_ALGORITHM_ID,
                ],
                "active_gradient_policy": (
                    ACTIVE_GRADIENT_STATIONARY
                ),
                "resource_weighting_scope": (
                    RESOURCE_WEIGHTING_ALL_PHASE
                ),
                "execution_target": (
                    QISKIT_COST_ALWAYS13_EXECUTION_TARGET
                ),
                "semantic_route_ids": list(
                    QISKIT_COST_ALWAYS13_ROUTE_IDS
                ),
                "cell_count": len(cells),
                "direct_execution_cell_count": len(cells),
                "materialization_status": "passed",
                "execution_authorized": False,
                "submission_state": SUBMISSION_STATE,
                "submitted": False,
            }
        )
        serialized_always13_binding = validation.get(
            "qiskit_cost_always13_validation_binding"
        )
        if not isinstance(serialized_always13_binding, Mapping):
            raise BundleMaterializationError(
                "Qiskit-cost always13 validation binding is missing."
            )
        _verify_digest(
            serialized_always13_binding,
            label="Qiskit-cost always13 validation binding",
        )
        if (
            dict(serialized_always13_binding)
            != expected_always13_binding
        ):
            raise BundleMaterializationError(
                "Qiskit-cost always13 validation binding drifted."
            )
    if campaign_id == PHASE3_QISKIT_CAMPAIGN_ID:
        expected_phase3_qiskit_binding = _digested(
            {
                "schema": (
                    "paper_i_ra_adapt_phase3_qiskit_mixed_horizon_"
                    "validation_binding_v1"
                ),
                "campaign_id": PHASE3_QISKIT_CAMPAIGN_ID,
                "bundle_id": PHASE3_QISKIT_BUNDLE_ID,
                "bundle_manifest_sha256": manifest["sha256"],
                "source_locks_sha256": source_locks["sha256"],
                "expected_artifacts_sha256": expected["sha256"],
                "materialization_gate_sha256": recomputed_gate["sha256"],
                "implementation_source_inventory_sha256": (
                    source_locks["implementation_sources"]["sha256"]
                ),
                "campaign_contract_sha256": canonical_sha256(
                    manifest["phase3_qiskit_mixed_horizon_contract"]
                ),
                "algorithm_ids": [
                    RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_ALGORITHM_ID
                ],
                "active_gradient_policy": ACTIVE_GRADIENT_STATIONARY,
                "resource_weighting_scope": RESOURCE_WEIGHTING_ALL_PHASE,
                "execution_target": PHASE3_QISKIT_EXECUTION_TARGET,
                "semantic_route_ids": list(PHASE3_QISKIT_ROUTE_IDS),
                "source_parent_route_profile": (
                    PHASE3_QISKIT_PAGE7_PARENT_ROUTE_PROFILE
                ),
                "source_parent_contract_sha256": (
                    PHASE3_QISKIT_PAGE7_PARENT_CONTRACT_SHA256
                ),
                "cell_count": len(cells),
                "direct_execution_cell_count": len(cells),
                "materialization_status": "passed",
                "execution_authorized": False,
                "submission_state": SUBMISSION_STATE,
                "submitted": False,
            }
        )
        serialized_phase3_qiskit_binding = validation.get(
            "phase3_qiskit_validation_binding"
        )
        if not isinstance(serialized_phase3_qiskit_binding, Mapping):
            raise BundleMaterializationError(
                "Phase-III-Qiskit validation binding is missing."
            )
        _verify_digest(
            serialized_phase3_qiskit_binding,
            label="Phase-III-Qiskit validation binding",
        )
        if (
            dict(serialized_phase3_qiskit_binding)
            != expected_phase3_qiskit_binding
        ):
            raise BundleMaterializationError(
                "Phase-III-Qiskit validation binding drifted."
            )
    if campaign_id == STUDY_ID and (
        checks_by_id["study1_append_shared_execution_dedupe"].get(
            "observed"
        )
        != study1_shared_execution_dedupe_contract()
    ):
        raise BundleMaterializationError(
            "Stale bundle validation report: Append dedupe contract drifted."
        )
    if (
        checks_by_id["paper_i_run_materialization_gate"].get(
            "observed"
        )
        != recomputed_gate
    ):
        raise BundleMaterializationError(
            "Stale bundle validation report: materialization gate drifted."
        )
    recomputed_binding = _materialization_binding(
        manifest=manifest,
        normalized_source_locks=source_locks,
        protocols=protocols,
        execution_templates=execution_templates,
        expected=expected,
        cells=cells,
        materialization_gate=recomputed_gate,
    )
    serialized_binding = validation.get("materialization_binding")
    if not isinstance(serialized_binding, Mapping):
        raise BundleMaterializationError(
            "Bundle validation report has no materialization binding."
        )
    _verify_digest(
        serialized_binding, label="materialization binding"
    )
    if dict(serialized_binding) != recomputed_binding:
        raise BundleMaterializationError(
            "Stale bundle validation report: cross-file binding drifted."
        )

    refs = _source_lock_refs(source_locks, cell=cell)
    protocol = resolved_ra_adapt_protocol_from_mapping(
        protocols[cell.cell_id]
    )
    authority = _bundle_protocol_materialization_authority(
        cell=cell,
        bundle_id=bundle_id,
        bundle_manifest_sha256=str(manifest["sha256"]),
        source_locks_sha256=str(source_locks["sha256"]),
        source_lock_refs=refs,
        active_gradient_policy=active_gradient_policy,
        resource_weighting_scope=resource_weighting_scope,
        protocol_sha256=protocol.sha256,
    )
    return _attach_validated_bundle_protocol_authority(
        protocol, authority
    )


def materialize_study1_bundles(
    destination: str | Path,
    *,
    problem_resolver: ProblemResolver,
    source_locks: Mapping[str, Any],
    validation_horizon: int | None,
    repository_state: Mapping[str, Any],
    repo_root: str | Path,
    protocol_resolver: ProtocolResolver | None = None,
    full_horizon: int = FULL_HORIZON,
    dependency_lock_paths: Sequence[str | Path] | None = None,
    environment_fingerprint: Mapping[str, Any] | None = None,
    numerical_runtime_contract: Mapping[str, Any] | None = None,
    materialization_timestamp: str | None = None,
    verify_source_files: bool = True,
) -> tuple[MaterializedBundleReceipt, MaterializedBundleReceipt]:
    """Materialize the two settled, matched Study-1 bundles.

    ``validation_horizon`` must be supplied for a fully passed handoff.  When
    it is ``None``, all ten validation protocols and the validation report are
    explicitly blocked rather than guessing a horizon.

    The function refuses to overwrite either bundle directory and contains no
    execution or scheduler seam.
    """

    root = Path(repo_root).expanduser().resolve()
    if not root.is_dir():
        raise BundleMaterializationError(f"repo_root is not a directory: {root}")
    destination_path = Path(destination).expanduser()
    if not destination_path.is_absolute():
        destination_path = root / destination_path
    destination_path = destination_path.resolve()
    cells = build_study1_cell_specs(
        validation_horizon=validation_horizon,
        full_horizon=full_horizon,
    )
    normalized_locks = normalize_and_verify_source_locks(
        source_locks,
        cells=cells,
        repo_root=root,
        verify_files=verify_source_files,
    )
    if not verify_source_files:
        # Audit-only materialization remains explicit and blocked.
        validation_horizon = None
        cells = build_study1_cell_specs(
            validation_horizon=None,
            full_horizon=full_horizon,
        )
    state = _repository_state(root, repository_state)
    if environment_fingerprint is None:
        environment = _default_environment_fingerprint()
    else:
        if not isinstance(environment_fingerprint, Mapping):
            raise BundleMaterializationError(
                "environment_fingerprint must be a mapping."
            )
        environment = _digested(
            {
                "schema": "ra_adapt_environment_fingerprint_v1",
                **dict(environment_fingerprint),
            }
        )
    dependencies = _dependency_lock_provenance(
        root, dependency_lock_paths
    )
    resolver = protocol_resolver or _default_protocol_resolver

    bundle_dirs = [
        destination_path / bundle_id
        for bundle_id, _policy in STUDY1_BUNDLE_POLICIES
    ]
    existing = [path for path in bundle_dirs if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing run bundle(s): "
            + ", ".join(str(path) for path in existing)
        )
    if numerical_runtime_contract is None:
        raise BundleMaterializationError(
            "numerical_runtime_contract is required for matched "
            "Append-ADAPT/RA-ADAPT materialization."
        )
    try:
        numerical_runtime = normalize_numerical_runtime_contract(
            numerical_runtime_contract
        )
    except NumericalRuntimeContractError as exc:
        raise BundleMaterializationError(
            f"numerical_runtime_contract is invalid: {exc}"
        ) from exc

    prepared: list[
        tuple[dict[str, Mapping[str, Any]], MaterializedBundleReceipt]
    ] = []
    for bundle_id, policy in STUDY1_BUNDLE_POLICIES:
        prepared.append(
            _prepare_bundle(
                destination=destination_path,
                bundle_id=bundle_id,
                active_gradient_policy=policy,
                cells=cells,
                normalized_source_locks=normalized_locks,
                problem_resolver=problem_resolver,
                protocol_resolver=resolver,
                repository_state=state,
                environment_fingerprint=environment,
                dependency_provenance=dependencies,
                materialization_timestamp=materialization_timestamp,
                numerical_runtime_contract=numerical_runtime,
            )
        )

    receipts: list[MaterializedBundleReceipt] = []
    for files, receipt in prepared:
        receipt.bundle_path.mkdir(parents=True, exist_ok=False)
        for relative_path, payload in sorted(files.items()):
            _write_canonical_json(
                receipt.bundle_path / relative_path, payload
            )
        receipts.append(receipt)
    return receipts[0], receipts[1]


def materialize_factorial_always_bundles(
    destination: str | Path,
    *,
    problem_resolver: ProblemResolver,
    source_locks: Mapping[str, Any] | None = None,
    source_locks_by_bundle: (
        Mapping[str, Mapping[str, Any]] | None
    ) = None,
    repository_state: Mapping[str, Any],
    repo_root: str | Path,
    protocol_resolver: ProtocolResolver | None = None,
    horizon: int = FULL_HORIZON,
    dependency_lock_paths: Sequence[str | Path] | None = None,
    environment_fingerprint: Mapping[str, Any] | None = None,
    materialization_timestamp: str | None = None,
    verify_source_files: bool = True,
) -> tuple[MaterializedBundleReceipt, ...]:
    """Materialize the four exact 12-cell corrected-always factorial arms.

    Per-bundle source locks are required when the D5 resource-scope
    declaration differs between arms.  ``source_locks`` remains an explicit
    shared-input convenience, but it is mutually exclusive with
    ``source_locks_by_bundle`` and still must pass every arm-specific D5 gate.
    """

    if not verify_source_files:
        raise BundleMaterializationError(
            "The corrected-always factorial requires exact source-byte "
            "verification."
        )
    if (source_locks is None) == (source_locks_by_bundle is None):
        raise BundleMaterializationError(
            "Supply exactly one of source_locks or "
            "source_locks_by_bundle."
        )
    root = Path(repo_root).expanduser().resolve()
    if not root.is_dir():
        raise BundleMaterializationError(
            f"repo_root is not a directory: {root}"
        )
    destination_path = Path(destination).expanduser()
    if not destination_path.is_absolute():
        destination_path = root / destination_path
    destination_path = destination_path.resolve()

    arm_specs = tuple(
        (
            bundle_id,
            active_gradient_policy,
            resource_weighting_scope,
            build_factorial_always_cell_specs(
                active_gradient_policy=active_gradient_policy,
                resource_weighting_scope=resource_weighting_scope,
                horizon=horizon,
            ),
        )
        for (
            bundle_id,
            active_gradient_policy,
            resource_weighting_scope,
        ) in FACTORIAL_BUNDLE_POLICIES
    )
    all_cells = tuple(
        cell
        for _bundle_id, _gradient, _resource, cells in arm_specs
        for cell in cells
    )
    if (
        len(all_cells) != 48
        or len({cell.cell_id for cell in all_cells}) != 48
    ):
        raise AssertionError(
            "The corrected-always factorial must contain 48 unique cells."
        )
    expected_bundle_ids = {
        bundle_id for bundle_id, _gradient, _resource, _cells in arm_specs
    }
    if source_locks_by_bundle is not None:
        if set(source_locks_by_bundle) != expected_bundle_ids:
            raise BundleMaterializationError(
                "source_locks_by_bundle must contain exactly the four "
                "factorial bundle ids."
            )
        raw_locks_by_bundle = dict(source_locks_by_bundle)
    else:
        assert source_locks is not None
        raw_locks_by_bundle = {
            bundle_id: source_locks
            for bundle_id in expected_bundle_ids
        }
    normalized_locks_by_bundle = {
        bundle_id: normalize_and_verify_source_locks(
            raw_locks_by_bundle[bundle_id],
            cells=cells,
            repo_root=root,
            verify_files=True,
        )
        for (
            bundle_id,
            _active_gradient_policy,
            _resource_weighting_scope,
            cells,
        ) in arm_specs
    }
    _validate_factorial_source_lock_matrix(
        source_locks_by_bundle=normalized_locks_by_bundle,
        cells_by_bundle={
            bundle_id: cells
            for bundle_id, _gradient, _resource, cells in arm_specs
        },
    )
    state = _repository_state(root, repository_state)
    if environment_fingerprint is None:
        environment = _default_environment_fingerprint()
    else:
        if not isinstance(environment_fingerprint, Mapping):
            raise BundleMaterializationError(
                "environment_fingerprint must be a mapping."
            )
        environment = _digested(
            {
                "schema": "ra_adapt_environment_fingerprint_v1",
                **dict(environment_fingerprint),
            }
        )
    dependencies = _dependency_lock_provenance(
        root, dependency_lock_paths
    )
    resolver = protocol_resolver or _default_protocol_resolver

    bundle_dirs = [
        destination_path / bundle_id
        for bundle_id, _gradient, _resource, _cells in arm_specs
    ]
    existing = [path for path in bundle_dirs if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing corrected-always factorial "
            "bundle(s): "
            + ", ".join(str(path) for path in existing)
        )

    prepared = [
        _prepare_bundle(
            destination=destination_path,
            bundle_id=bundle_id,
            active_gradient_policy=active_gradient_policy,
            resource_weighting_scope=resource_weighting_scope,
            cells=cells,
            normalized_source_locks=normalized_locks_by_bundle[bundle_id],
            problem_resolver=problem_resolver,
            protocol_resolver=resolver,
            repository_state=state,
            environment_fingerprint=environment,
            dependency_provenance=dependencies,
            materialization_timestamp=materialization_timestamp,
            campaign_id=FACTORIAL_CAMPAIGN_ID,
        )
        for (
            bundle_id,
            active_gradient_policy,
            resource_weighting_scope,
            cells,
        ) in arm_specs
    ]

    receipts: list[MaterializedBundleReceipt] = []
    for files, receipt in prepared:
        receipt.bundle_path.mkdir(parents=True, exist_ok=False)
        for relative_path, payload in sorted(files.items()):
            _write_canonical_json(
                receipt.bundle_path / relative_path, payload
            )
        receipts.append(receipt)
    return tuple(receipts)


def materialize_global_singleton_insertion_bundle(
    destination: str | Path,
    *,
    problem_resolver: ProblemResolver,
    source_locks: Mapping[str, Any],
    repository_state: Mapping[str, Any],
    repo_root: str | Path,
    protocol_resolver: ProtocolResolver | None = None,
    horizon: int = FULL_HORIZON,
    dependency_lock_paths: Sequence[str | Path] | None = None,
    environment_fingerprint: Mapping[str, Any] | None = None,
    materialization_timestamp: str | None = None,
    verify_source_files: bool = True,
) -> MaterializedBundleReceipt:
    """Materialize the inert 12-cell global-singleton insertion campaign."""

    if not verify_source_files:
        raise BundleMaterializationError(
            "The global-singleton insertion comparison requires exact "
            "source-byte verification."
        )
    root = Path(repo_root).expanduser().resolve()
    if not root.is_dir():
        raise BundleMaterializationError(
            f"repo_root is not a directory: {root}"
        )
    destination_path = Path(destination).expanduser()
    if not destination_path.is_absolute():
        destination_path = root / destination_path
    destination_path = destination_path.resolve()
    cells = build_global_singleton_insertion_cell_specs(horizon=horizon)
    normalized_locks = normalize_and_verify_source_locks(
        source_locks,
        cells=cells,
        repo_root=root,
        verify_files=True,
    )
    _validate_global_singleton_source_lock_matrix(
        source_locks=normalized_locks,
        cells=cells,
    )
    state = _repository_state(root, repository_state)
    if environment_fingerprint is None:
        environment = _default_environment_fingerprint()
    else:
        if not isinstance(environment_fingerprint, Mapping):
            raise BundleMaterializationError(
                "environment_fingerprint must be a mapping."
            )
        environment = _digested(
            {
                "schema": "ra_adapt_environment_fingerprint_v1",
                **dict(environment_fingerprint),
            }
        )
    dependencies = _dependency_lock_provenance(
        root, dependency_lock_paths
    )
    resolver = protocol_resolver or _default_protocol_resolver
    bundle_dir = destination_path / GLOBAL_SINGLETON_BUNDLE_ID
    if bundle_dir.exists():
        raise FileExistsError(
            "Refusing to overwrite existing global-singleton insertion "
            f"bundle: {bundle_dir}"
        )
    files, receipt = _prepare_bundle(
        destination=destination_path,
        bundle_id=GLOBAL_SINGLETON_BUNDLE_ID,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
        cells=cells,
        normalized_source_locks=normalized_locks,
        problem_resolver=problem_resolver,
        protocol_resolver=resolver,
        repository_state=state,
        environment_fingerprint=environment,
        dependency_provenance=dependencies,
        materialization_timestamp=materialization_timestamp,
        campaign_id=GLOBAL_SINGLETON_CAMPAIGN_ID,
    )
    receipt.bundle_path.mkdir(parents=True, exist_ok=False)
    for relative_path, payload in sorted(files.items()):
        _write_canonical_json(receipt.bundle_path / relative_path, payload)
    return receipt


def materialize_qiskit_cost_plateau_pilot_bundle(
    destination: str | Path,
    *,
    problem_resolver: ProblemResolver,
    source_locks: Mapping[str, Any],
    repository_state: Mapping[str, Any],
    repo_root: str | Path,
    protocol_resolver: ProtocolResolver | None = None,
    horizon: int = FULL_HORIZON,
    dependency_lock_paths: Sequence[str | Path] | None = None,
    environment_fingerprint: Mapping[str, Any] | None = None,
    materialization_timestamp: str | None = None,
    verify_source_files: bool = True,
) -> MaterializedBundleReceipt:
    """Materialize the inert two-cell local Qiskit-cost plateau pilot."""

    if not verify_source_files:
        raise BundleMaterializationError(
            "The Qiskit-cost plateau pilot requires exact source-byte "
            "verification."
        )
    root = Path(repo_root).expanduser().resolve()
    if not root.is_dir():
        raise BundleMaterializationError(
            f"repo_root is not a directory: {root}"
        )
    destination_path = Path(destination).expanduser()
    if not destination_path.is_absolute():
        destination_path = root / destination_path
    destination_path = destination_path.resolve()
    cells = build_qiskit_cost_plateau_pilot_cell_specs(
        horizon=horizon
    )
    normalized_locks = normalize_and_verify_source_locks(
        source_locks,
        cells=cells,
        repo_root=root,
        verify_files=True,
    )
    for cell in cells:
        trace = normalized_locks["cell_locks"][cell.source_lock_id][
            "resolver_trace"
        ]
        _validate_qiskit_cost_pilot_source_lock(
            cell=cell,
            trace=trace,
        )
    state = _repository_state(root, repository_state)
    if environment_fingerprint is None:
        environment = _default_environment_fingerprint()
    else:
        if not isinstance(environment_fingerprint, Mapping):
            raise BundleMaterializationError(
                "environment_fingerprint must be a mapping."
            )
        environment = _digested(
            {
                "schema": "ra_adapt_environment_fingerprint_v1",
                **dict(environment_fingerprint),
            }
        )
    dependencies = _dependency_lock_provenance(
        root, dependency_lock_paths
    )
    resolver = protocol_resolver or _default_protocol_resolver
    bundle_dir = destination_path / QISKIT_COST_PILOT_BUNDLE_ID
    if bundle_dir.exists():
        raise FileExistsError(
            "Refusing to overwrite existing Qiskit-cost plateau pilot "
            f"bundle: {bundle_dir}"
        )
    files, receipt = _prepare_bundle(
        destination=destination_path,
        bundle_id=QISKIT_COST_PILOT_BUNDLE_ID,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
        cells=cells,
        normalized_source_locks=normalized_locks,
        problem_resolver=problem_resolver,
        protocol_resolver=resolver,
        repository_state=state,
        environment_fingerprint=environment,
        dependency_provenance=dependencies,
        materialization_timestamp=materialization_timestamp,
        campaign_id=QISKIT_COST_PILOT_CAMPAIGN_ID,
    )
    receipt.bundle_path.mkdir(parents=True, exist_ok=False)
    for relative_path, payload in sorted(files.items()):
        _write_canonical_json(receipt.bundle_path / relative_path, payload)
    return receipt


def materialize_qiskit_cost_always13_bundle(
    destination: str | Path,
    *,
    problem_resolver: ProblemResolver,
    source_locks: Mapping[str, Any],
    repository_state: Mapping[str, Any],
    repo_root: str | Path,
    protocol_resolver: ProtocolResolver | None = None,
    horizon: int = QISKIT_COST_ALWAYS13_HORIZON,
    dependency_lock_paths: Sequence[str | Path] | None = None,
    environment_fingerprint: Mapping[str, Any] | None = None,
    materialization_timestamp: str | None = None,
    verify_source_files: bool = True,
) -> MaterializedBundleReceipt:
    """Materialize the inert one-cell Qiskit-cost always13 diagnostic."""

    if not verify_source_files:
        raise BundleMaterializationError(
            "The Qiskit-cost always13 diagnostic requires exact "
            "source-byte verification."
        )
    root = Path(repo_root).expanduser().resolve()
    if not root.is_dir():
        raise BundleMaterializationError(
            f"repo_root is not a directory: {root}"
        )
    destination_path = Path(destination).expanduser()
    if not destination_path.is_absolute():
        destination_path = root / destination_path
    destination_path = destination_path.resolve()
    cells = build_qiskit_cost_always13_cell_specs(horizon=horizon)
    normalized_locks = normalize_and_verify_source_locks(
        source_locks,
        cells=cells,
        repo_root=root,
        verify_files=True,
    )
    for cell in cells:
        trace = normalized_locks["cell_locks"][cell.source_lock_id][
            "resolver_trace"
        ]
        _validate_qiskit_cost_always13_source_lock(
            cell=cell,
            trace=trace,
        )
    state = _repository_state(root, repository_state)
    if environment_fingerprint is None:
        environment = _default_environment_fingerprint()
    else:
        if not isinstance(environment_fingerprint, Mapping):
            raise BundleMaterializationError(
                "environment_fingerprint must be a mapping."
            )
        environment = _digested(
            {
                "schema": "ra_adapt_environment_fingerprint_v1",
                **dict(environment_fingerprint),
            }
        )
    dependencies = _dependency_lock_provenance(
        root, dependency_lock_paths
    )
    resolver = protocol_resolver or _default_protocol_resolver
    bundle_dir = destination_path / QISKIT_COST_ALWAYS13_BUNDLE_ID
    if bundle_dir.exists():
        raise FileExistsError(
            "Refusing to overwrite existing Qiskit-cost always13 "
            f"bundle: {bundle_dir}"
        )
    files, receipt = _prepare_bundle(
        destination=destination_path,
        bundle_id=QISKIT_COST_ALWAYS13_BUNDLE_ID,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
        cells=cells,
        normalized_source_locks=normalized_locks,
        problem_resolver=problem_resolver,
        protocol_resolver=resolver,
        repository_state=state,
        environment_fingerprint=environment,
        dependency_provenance=dependencies,
        materialization_timestamp=materialization_timestamp,
        campaign_id=QISKIT_COST_ALWAYS13_CAMPAIGN_ID,
    )
    receipt.bundle_path.mkdir(parents=True, exist_ok=False)
    for relative_path, payload in sorted(files.items()):
        _write_canonical_json(receipt.bundle_path / relative_path, payload)
    return receipt


def materialize_qiskit_cost_always6_bundle(
    destination: str | Path,
    *,
    problem_resolver: ProblemResolver,
    source_locks: Mapping[str, Any],
    repository_state: Mapping[str, Any],
    repo_root: str | Path,
    protocol_resolver: ProtocolResolver | None = None,
    horizon_by_regime: Mapping[str, int] | None = None,
    dependency_lock_paths: Sequence[str | Path] | None = None,
    environment_fingerprint: Mapping[str, Any] | None = None,
    materialization_timestamp: str | None = None,
    verify_source_files: bool = True,
) -> MaterializedBundleReceipt:
    """Materialize the inert six-cell macro always-insertion Qiskit-cost run."""

    if not verify_source_files:
        raise BundleMaterializationError(
            "The Qiskit-cost always6 diagnostic requires exact "
            "source-byte verification."
        )
    root = Path(repo_root).expanduser().resolve()
    if not root.is_dir():
        raise BundleMaterializationError(
            f"repo_root is not a directory: {root}"
        )
    destination_path = Path(destination).expanduser()
    if not destination_path.is_absolute():
        destination_path = root / destination_path
    destination_path = destination_path.resolve()
    cells = build_qiskit_cost_always6_cell_specs(
        horizon_by_regime=horizon_by_regime
    )
    normalized_locks = normalize_and_verify_source_locks(
        source_locks,
        cells=cells,
        repo_root=root,
        verify_files=True,
    )
    state = _repository_state(root, repository_state)
    if environment_fingerprint is None:
        environment = _default_environment_fingerprint()
    else:
        if not isinstance(environment_fingerprint, Mapping):
            raise BundleMaterializationError(
                "environment_fingerprint must be a mapping."
            )
        environment = _digested(
            {
                "schema": "ra_adapt_environment_fingerprint_v1",
                **dict(environment_fingerprint),
            }
        )
    dependencies = _dependency_lock_provenance(
        root, dependency_lock_paths
    )
    resolver = protocol_resolver or _default_protocol_resolver
    bundle_dir = destination_path / QISKIT_COST_ALWAYS6_BUNDLE_ID
    if bundle_dir.exists():
        raise FileExistsError(
            "Refusing to overwrite existing Qiskit-cost always6 "
            f"bundle: {bundle_dir}"
        )
    files, receipt = _prepare_bundle(
        destination=destination_path,
        bundle_id=QISKIT_COST_ALWAYS6_BUNDLE_ID,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
        cells=cells,
        normalized_source_locks=normalized_locks,
        problem_resolver=problem_resolver,
        protocol_resolver=resolver,
        repository_state=state,
        environment_fingerprint=environment,
        dependency_provenance=dependencies,
        materialization_timestamp=materialization_timestamp,
        campaign_id=QISKIT_COST_ALWAYS6_CAMPAIGN_ID,
    )
    receipt.bundle_path.mkdir(parents=True, exist_ok=False)
    for relative_path, payload in sorted(files.items()):
        _write_canonical_json(receipt.bundle_path / relative_path, payload)
    return receipt


def materialize_beamprune_bundle(
    destination: str | Path,
    *,
    problem_resolver: ProblemResolver,
    source_locks: Mapping[str, Any],
    repository_state: Mapping[str, Any],
    repo_root: str | Path,
    protocol_resolver: ProtocolResolver | None = None,
    horizon: int = BEAMPRUNE_HORIZON,
    dependency_lock_paths: Sequence[str | Path] | None = None,
    environment_fingerprint: Mapping[str, Any] | None = None,
    materialization_timestamp: str | None = None,
    verify_source_files: bool = True,
) -> MaterializedBundleReceipt:
    """Materialize the inert 24-cell beam+prune lane ablation."""

    root = Path(repo_root).expanduser().resolve()
    destination_path = Path(destination).expanduser()
    if not destination_path.is_absolute():
        destination_path = root / destination_path
    destination_path = destination_path.resolve()
    cells = build_beamprune_cell_specs(horizon=horizon)
    normalized_locks = normalize_and_verify_source_locks(
        source_locks, cells=cells, repo_root=root, verify_files=True
    )
    state = _repository_state(root, repository_state)
    environment = (
        _default_environment_fingerprint()
        if environment_fingerprint is None
        else _digested({"schema": "ra_adapt_environment_fingerprint_v1",
                        **dict(environment_fingerprint)})
    )
    dependencies = _dependency_lock_provenance(root, dependency_lock_paths)
    bundle_dir = destination_path / BEAMPRUNE_BUNDLE_ID
    if bundle_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {bundle_dir}")
    files, receipt = _prepare_bundle(
        destination=destination_path,
        bundle_id=BEAMPRUNE_BUNDLE_ID,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
        cells=cells,
        normalized_source_locks=normalized_locks,
        problem_resolver=problem_resolver,
        protocol_resolver=protocol_resolver or _default_protocol_resolver,
        repository_state=state,
        environment_fingerprint=environment,
        dependency_provenance=dependencies,
        materialization_timestamp=materialization_timestamp,
        campaign_id=BEAMPRUNE_CAMPAIGN_ID,
    )
    receipt.bundle_path.mkdir(parents=True, exist_ok=False)
    for relative_path, payload in sorted(files.items()):
        _write_canonical_json(receipt.bundle_path / relative_path, payload)
    return receipt


def materialize_lanes_ablation_bundle(
    destination: str | Path,
    *,
    problem_resolver: ProblemResolver,
    source_locks: Mapping[str, Any],
    repository_state: Mapping[str, Any],
    repo_root: str | Path,
    protocol_resolver: ProtocolResolver | None = None,
    horizon: int = LANES_ABLATION_HORIZON,
    dependency_lock_paths: Sequence[str | Path] | None = None,
    environment_fingerprint: Mapping[str, Any] | None = None,
    materialization_timestamp: str | None = None,
    verify_source_files: bool = True,
) -> MaterializedBundleReceipt:
    """Materialize the inert twelve-cell macro always-insertion lane ablation."""

    if not verify_source_files:
        raise BundleMaterializationError(
            "The lanes ablation requires exact source-byte verification."
        )
    root = Path(repo_root).expanduser().resolve()
    if not root.is_dir():
        raise BundleMaterializationError(
            f"repo_root is not a directory: {root}"
        )
    destination_path = Path(destination).expanduser()
    if not destination_path.is_absolute():
        destination_path = root / destination_path
    destination_path = destination_path.resolve()
    cells = build_lanes_ablation_cell_specs(horizon=horizon)
    normalized_locks = normalize_and_verify_source_locks(
        source_locks,
        cells=cells,
        repo_root=root,
        verify_files=True,
    )
    state = _repository_state(root, repository_state)
    if environment_fingerprint is None:
        environment = _default_environment_fingerprint()
    else:
        if not isinstance(environment_fingerprint, Mapping):
            raise BundleMaterializationError(
                "environment_fingerprint must be a mapping."
            )
        environment = _digested(
            {
                "schema": "ra_adapt_environment_fingerprint_v1",
                **dict(environment_fingerprint),
            }
        )
    dependencies = _dependency_lock_provenance(root, dependency_lock_paths)
    resolver = protocol_resolver or _default_protocol_resolver
    bundle_dir = destination_path / LANES_ABLATION_BUNDLE_ID
    if bundle_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing lanes ablation bundle: {bundle_dir}"
        )
    files, receipt = _prepare_bundle(
        destination=destination_path,
        bundle_id=LANES_ABLATION_BUNDLE_ID,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
        cells=cells,
        normalized_source_locks=normalized_locks,
        problem_resolver=problem_resolver,
        protocol_resolver=resolver,
        repository_state=state,
        environment_fingerprint=environment,
        dependency_provenance=dependencies,
        materialization_timestamp=materialization_timestamp,
        campaign_id=LANES_ABLATION_CAMPAIGN_ID,
    )
    receipt.bundle_path.mkdir(parents=True, exist_ok=False)
    for relative_path, payload in sorted(files.items()):
        _write_canonical_json(receipt.bundle_path / relative_path, payload)
    return receipt


def materialize_phase3_qiskit_mixed_horizon_bundle(
    destination: str | Path,
    *,
    problem_resolver: ProblemResolver,
    source_locks: Mapping[str, Any],
    repository_state: Mapping[str, Any],
    repo_root: str | Path,
    protocol_resolver: ProtocolResolver | None = None,
    weak_holstein_horizon: int = PHASE3_QISKIT_WEAK_HOLSTEIN_HORIZON,
    strong_holstein_horizon: int = PHASE3_QISKIT_STRONG_HOLSTEIN_HORIZON,
    dependency_lock_paths: Sequence[str | Path] | None = None,
    environment_fingerprint: Mapping[str, Any] | None = None,
    materialization_timestamp: str | None = None,
    verify_source_files: bool = True,
) -> MaterializedBundleReceipt:
    """Materialize the inert six-cell Phase-III-Qiskit candidate bundle."""

    if not verify_source_files:
        raise BundleMaterializationError(
            "The Phase-III-Qiskit candidate campaign requires exact "
            "source-byte verification."
        )
    root = Path(repo_root).expanduser().resolve()
    if not root.is_dir():
        raise BundleMaterializationError(
            f"repo_root is not a directory: {root}"
        )
    destination_path = Path(destination).expanduser()
    if not destination_path.is_absolute():
        destination_path = root / destination_path
    destination_path = destination_path.resolve()
    cells = build_phase3_qiskit_mixed_horizon_cell_specs(
        weak_holstein_horizon=weak_holstein_horizon,
        strong_holstein_horizon=strong_holstein_horizon,
    )
    normalized_locks = normalize_and_verify_source_locks(
        source_locks,
        cells=cells,
        repo_root=root,
        verify_files=True,
    )
    for cell in cells:
        trace = normalized_locks["cell_locks"][cell.source_lock_id][
            "resolver_trace"
        ]
        _validate_phase3_qiskit_source_lock(
            cell=cell,
            trace=trace,
        )
    state = _repository_state(root, repository_state)
    if environment_fingerprint is None:
        environment = _default_environment_fingerprint()
    else:
        if not isinstance(environment_fingerprint, Mapping):
            raise BundleMaterializationError(
                "environment_fingerprint must be a mapping."
            )
        environment = _digested(
            {
                "schema": "ra_adapt_environment_fingerprint_v1",
                **dict(environment_fingerprint),
            }
        )
    dependencies = _dependency_lock_provenance(
        root, dependency_lock_paths
    )
    resolver = protocol_resolver or _default_protocol_resolver
    bundle_dir = destination_path / PHASE3_QISKIT_BUNDLE_ID
    if bundle_dir.exists():
        raise FileExistsError(
            "Refusing to overwrite existing Phase-III-Qiskit candidate "
            f"bundle: {bundle_dir}"
        )
    files, receipt = _prepare_bundle(
        destination=destination_path,
        bundle_id=PHASE3_QISKIT_BUNDLE_ID,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
        cells=cells,
        normalized_source_locks=normalized_locks,
        problem_resolver=problem_resolver,
        protocol_resolver=resolver,
        repository_state=state,
        environment_fingerprint=environment,
        dependency_provenance=dependencies,
        materialization_timestamp=materialization_timestamp,
        campaign_id=PHASE3_QISKIT_CAMPAIGN_ID,
    )
    receipt.bundle_path.mkdir(parents=True, exist_ok=False)
    for relative_path, payload in sorted(files.items()):
        _write_canonical_json(receipt.bundle_path / relative_path, payload)
    return receipt


def materialize_core_bundle(
    destination: str | Path,
    *,
    problem_resolver: ProblemResolver,
    source_locks: Mapping[str, Any],
    repository_state: Mapping[str, Any],
    repo_root: str | Path,
    protocol_resolver: ProtocolResolver | None = None,
    horizon: int = FULL_HORIZON,
    dependency_lock_paths: Sequence[str | Path] | None = None,
    environment_fingerprint: Mapping[str, Any] | None = None,
    materialization_timestamp: str | None = None,
    verify_source_files: bool = True,
) -> MaterializedBundleReceipt:
    """Materialize the selected stationary 48-cell Paper-I core.

    This paper-facing path has no audit-only mode: all global sources,
    per-cell archives and members, implementation sources, and the explicit
    user-selection receipt must verify before any bundle directory is
    written.  The resulting execution templates remain unauthorized.
    """

    if not verify_source_files:
        raise BundleMaterializationError(
            "The paper-facing stationary core requires exact source-byte "
            "verification."
        )
    root = Path(repo_root).expanduser().resolve()
    if not root.is_dir():
        raise BundleMaterializationError(
            f"repo_root is not a directory: {root}"
        )
    destination_path = Path(destination).expanduser()
    if not destination_path.is_absolute():
        destination_path = root / destination_path
    destination_path = destination_path.resolve()
    cells = build_core_cell_specs(horizon=horizon)
    normalized_locks = normalize_and_verify_source_locks(
        source_locks,
        cells=cells,
        repo_root=root,
        verify_files=True,
    )
    normalized_locks = _bind_core_selection_authority(
        normalized_locks,
        repo_root=root,
    )
    state = _repository_state(root, repository_state)
    if environment_fingerprint is None:
        environment = _default_environment_fingerprint()
    else:
        if not isinstance(environment_fingerprint, Mapping):
            raise BundleMaterializationError(
                "environment_fingerprint must be a mapping."
            )
        environment = _digested(
            {
                "schema": "ra_adapt_environment_fingerprint_v1",
                **dict(environment_fingerprint),
            }
        )
    dependencies = _dependency_lock_provenance(
        root, dependency_lock_paths
    )
    resolver = protocol_resolver or _default_protocol_resolver
    bundle_dir = destination_path / CORE_BUNDLE_ID
    if bundle_dir.exists():
        raise FileExistsError(
            "Refusing to overwrite existing stationary-core run bundle: "
            f"{bundle_dir}"
        )
    files, receipt = _prepare_bundle(
        destination=destination_path,
        bundle_id=CORE_BUNDLE_ID,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        cells=cells,
        normalized_source_locks=normalized_locks,
        problem_resolver=problem_resolver,
        protocol_resolver=resolver,
        repository_state=state,
        environment_fingerprint=environment,
        dependency_provenance=dependencies,
        materialization_timestamp=materialization_timestamp,
        campaign_id=CORE_CAMPAIGN_ID,
    )
    receipt.bundle_path.mkdir(parents=True, exist_ok=False)
    for relative_path, payload in sorted(files.items()):
        _write_canonical_json(receipt.bundle_path / relative_path, payload)
    return receipt


__all__ = [
    "BLOCKED_PROTOCOL_SCHEMA",
    "BUNDLE_SCHEMA",
    "BundleCellSpec",
    "BundleMaterializationError",
    "CLAIM_FACING_REGIME_CUTOFF_PAIRS",
    "CORE_BUNDLE_ID",
    "CORE_CAMPAIGN_ID",
    "CORE_RUN_CLASS",
    "CORE_SELECTION_AUTHORITY_PATH",
    "CORE_SELECTION_AUTHORITY_SHA256",
    "CORE_VISIBLE_TARGET_ID",
    "EXECUTION_TARGET",
    "EXECUTION_TEMPLATE_SCHEMA",
    "EXPECTED_ARTIFACT_ROLES",
    "EXPECTED_ARTIFACTS_SCHEMA",
    "FACTORIAL_BUNDLE_POLICIES",
    "FACTORIAL_CAMPAIGN_ID",
    "FACTORIAL_MEASURED_ALL_PHASE_BUNDLE_ID",
    "FACTORIAL_MEASURED_LATE_BUNDLE_ID",
    "FACTORIAL_RUN_CLASS",
    "FACTORIAL_STATIONARY_ALL_PHASE_BUNDLE_ID",
    "FACTORIAL_STATIONARY_LATE_BUNDLE_ID",
    "FACTORIAL_VISIBLE_TARGET_ID",
    "FULL_HORIZON",
    "FULL_VISIBLE_REGIME_CUTOFF_PAIRS",
    "GLOBAL_SOURCE_LOCKS",
    "GLOBAL_SINGLETON_BUNDLE_ID",
    "GLOBAL_SINGLETON_CAMPAIGN_ID",
    "GLOBAL_SINGLETON_INSERTION_ROUTE_IDS",
    "GLOBAL_SINGLETON_ORDERED_POOL_SHA256_BY_REGIME",
    "GLOBAL_SINGLETON_POOL_MEMBERSHIP_BY_NPH",
    "GLOBAL_SINGLETON_RUN_CLASS",
    "GLOBAL_SINGLETON_VISIBLE_TARGET_ID",
    "MACRO_ROUTE_IDS",
    "MATERIALIZATION_BINDING_SCHEMA",
    "MEASURED_BUNDLE_ID",
    "MaterializedBundleReceipt",
    "OBJECTIVE_EXECUTION_GATE_IDS",
    "PHASE3_QISKIT_BUNDLE_ID",
    "PHASE3_QISKIT_CAMPAIGN_ID",
    "PHASE3_QISKIT_EXECUTION_TARGET",
    "PHASE3_QISKIT_PAGE7_PARENT_CONTRACT_SHA256",
    "PHASE3_QISKIT_PAGE7_PARENT_ROUTE_PROFILE",
    "PHASE3_QISKIT_ROUTE_IDS",
    "PHASE3_QISKIT_RUN_CLASS",
    "PHASE3_QISKIT_STRONG_HOLSTEIN_HORIZON",
    "PHASE3_QISKIT_VISIBLE_TARGET_ID",
    "PHASE3_QISKIT_WEAK_HOLSTEIN_HORIZON",
    "PRESERVATION_EXECUTION_GATE_SCHEMA",
    "PRESERVATION_MEASURED_GATE_ID",
    "PRESERVATION_STATIONARY_GATE_ID",
    "ProtocolResolutionContext",
    "QISKIT_COST_PILOT_BUNDLE_ID",
    "QISKIT_COST_PILOT_CAMPAIGN_ID",
    "QISKIT_COST_PILOT_EXECUTION_TARGET",
    "QISKIT_COST_PILOT_GLOBAL_SINGLETON_ALGORITHM_ID",
    "QISKIT_COST_PILOT_MACRO_ALGORITHM_ID",
    "QISKIT_COST_PILOT_ROUTE_IDS",
    "QISKIT_COST_PILOT_RUN_CLASS",
    "QISKIT_COST_PILOT_VISIBLE_TARGET_ID",
    "QISKIT_COST_ALWAYS13_ALGORITHM_ID",
    "QISKIT_COST_ALWAYS13_BUNDLE_ID",
    "QISKIT_COST_ALWAYS13_CAMPAIGN_ID",
    "QISKIT_COST_ALWAYS13_EXECUTION_TARGET",
    "QISKIT_COST_ALWAYS13_HORIZON",
    "QISKIT_COST_ALWAYS13_ROUTE_IDS",
    "QISKIT_COST_ALWAYS13_RUN_CLASS",
    "QISKIT_COST_ALWAYS13_SOURCE_PROTOCOL_SHA256",
    "QISKIT_COST_ALWAYS13_VISIBLE_TARGET_ID",
    "RUN_CLASS",
    "ROUTE_APPEND_MACRO",
    "ROUTE_APPEND_SINGLETON",
    "ROUTE_RA_MACRO_ALWAYS",
    "ROUTE_RA_MACRO_APPEND_ONLY",
    "ROUTE_RA_MACRO_PLATEAU",
    "ROUTE_RA_SINGLETON_ALWAYS",
    "ROUTE_RA_SINGLETON_APPEND_ONLY",
    "ROUTE_RA_SINGLETON_PLATEAU",
    "ROUTE_RA_GLOBAL_SINGLETON_APPEND_REDUCED",
    "ROUTE_RA_GLOBAL_SINGLETON_PLATEAU",
    "ROUTE_SINGLETON_PLATEAU",
    "SINGLETON_CORE_ROUTE_IDS",
    "SOURCE_LOCK_SCHEMA",
    "STATIONARY_BUNDLE_ID",
    "STUDY1_BUNDLE_POLICIES",
    "STUDY_ID",
    "STUDY1_EXECUTION_DEDUPE_SCHEMA",
    "SUBMISSION_STATE",
    "VALIDATION_REGIMES",
    "VALIDATION_REPORT_SCHEMA",
    "VALIDATION_ROUTE_IDS",
    "VISIBLE_TARGET_ID",
    "build_core_cell_specs",
    "build_factorial_always_cell_specs",
    "build_global_singleton_insertion_cell_specs",
    "build_phase3_qiskit_mixed_horizon_cell_specs",
    "build_qiskit_cost_plateau_pilot_cell_specs",
    "build_qiskit_cost_always13_cell_specs",
    "build_study1_cell_specs",
    "load_validated_bundle_protocol",
    "materialize_semantic_closure_protocol",
    "materialize_core_bundle",
    "materialize_factorial_always_bundles",
    "materialize_global_singleton_insertion_bundle",
    "materialize_phase3_qiskit_mixed_horizon_bundle",
    "materialize_qiskit_cost_plateau_pilot_bundle",
    "materialize_qiskit_cost_always13_bundle",
    "materialize_study1_bundles",
    "normalize_and_verify_source_locks",
    "preservation_execution_gate_contract",
    "source_lock_id",
    "study1_shared_execution_dedupe_contract",
    "validate_full_matrix_progression",
    "validate_preservation_execution_gate",
]
