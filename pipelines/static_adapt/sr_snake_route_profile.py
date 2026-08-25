"""Executable route contracts for historical and conventional SR-SNAKE.

The older SR identity resolver intentionally classified only the local
coordinate/trust overlay.  That was not enough to reproduce the Paper-I
Hubbard--Holstein route: a partially configured invocation could acquire the
canonical profile name while using different optimizer, child, shortlist,
beam, prune, or fallback policies.

This module owns the fail-closed CLI normalization contracts.  It contains no
scientific implementation; it only materializes and hashes already implemented
settings before the runtime begins.  SR-SNAKE v1 preserves the historical
route.  Conventional SR-SNAKE v2 adds the full-accepted-ansatz, supported-FS
Powell chart used by the three 2026-07-15 weak-Holstein anchor runs.
Conventional SR-SNAKE v3 additionally decouples the Phase-III response model
from every Powell/refit window and requires the full active logical ansatz plus
the singleton candidate to enter the response model before Gram support
reduction.
Conventional SR-SNAKE v3.1 preserves that frozen v3 contract and records the
historical disabling of phase-live hysteresis.  That hashed field is passive:
configured phases are always live at runtime.  The unqualified ``sr_snake``
alias resolves to v3.1; explicit ``sr_snake_v3`` remains byte-stable for
historical replay.
SR-SNAKE v4 is an opt-in candidate profile layered on v3.  It combines the
bounded symmetric hardware-cost transform with live-only full-logical
Fubini--Study trust pruning and zero-query Phase-III damping diagnostics.  The
v4 validation campaign does not silently redefine the conventional alias.
The projected-Phase-III ablation is a one-setting child of the validated Main
SR no-prune/symmetric-cost contract: it retains raw supported metric modes and
solves their generalized FS trust problem without Phase-III whitening, while
leaving the accepted full-ansatz Powell refit whitened.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from pipelines.static_adapt.phase3_material_window import (
    DEFAULT_PHASE3_MATERIAL_WINDOW_POLICY,
)
from pipelines.static_adapt.sr_snake_phase12_policy import (
    PHASE1_ENERGY_MODEL_FIRST_ORDER_FS_TRUST_V1,
    PHASE1_SCORE_MODE_TRUST_REGION_V1,
    PHASE2_CHEAP_CURVATURE_PROXY_POLICY_OFF,
    PHASE2_CURVATURE_POLICY_LEGACY_OPTIONAL_V1,
    PHASE2_CURVATURE_POLICY_MEASURED_REQUIRED_FAIL_CLOSED_V1,
)


SR_ROUTE_PROFILE_REQUEST_OFF = "off"
INSERTION_COMMUTATION_PLATEAU_CUMULATIVE_DECREASE_RATIO_THRESHOLD = 1.0e-4
INSERTION_COMMUTATION_PLATEAU_CALIBRATION_STATUS = (
    "source_locked_completed_trajectory_replay_v1"
)
INSERTION_COMMUTATION_PLATEAU_PRIOR_MEAN_DECREASE_RATIO_THRESHOLD = 1.0e-4
INSERTION_COMMUTATION_PLATEAU_PRIOR_MEAN_CALIBRATION_STATUS = (
    "source_locked_counterfactual_trigger_replay_v2"
)
SR_ROUTE_PROFILE_CANONICAL_V1 = "supported_whitened_adaptive_trust_v1"
SR_ROUTE_PROFILE_CONVENTIONAL_V2 = (
    "supported_whitened_adaptive_trust_full_accepted_refit_v2"
)
SR_ROUTE_PROFILE_CONVENTIONAL_V3 = (
    "supported_whitened_adaptive_trust_full_response_full_accepted_refit_v3"
)
SR_ROUTE_PROFILE_CANDIDATE_V4 = (
    "supported_whitened_adaptive_trust_full_response_symmetric_cost_fs_prune_v4"
)
SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1 = (
    "supported_whitened_adaptive_trust_full_response_symmetric_cost_"
    "no_prune_v1"
)
SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_V1 = (
    "supported_projected_generalized_adaptive_trust_full_response_"
    "symmetric_cost_no_prune_v1"
)
SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1 = (
    "supported_projected_generalized_source_metric_no_overlap_trust_"
    "full_response_symmetric_cost_no_prune_v1"
)
SR_ROUTE_PROFILE_GREEDY_BATCH_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1 = (
    "supported_projected_generalized_source_metric_no_overlap_trust_"
    "full_response_symmetric_cost_no_prune_greedy_batch_v1"
)
SR_ROUTE_PROFILE_COMBINATORIAL_BATCH_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1 = (
    "supported_projected_generalized_source_metric_no_overlap_trust_"
    "full_response_symmetric_cost_no_prune_combinatorial_batch_v1"
)
SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_V1 = (
    "supported_projected_generalized_source_metric_no_overlap_trust_"
    "full_response_symmetric_cost_no_prune_commutation_reduced_insertion_v1"
)
SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V1 = (
    "supported_projected_generalized_source_metric_no_overlap_trust_"
    "full_response_symmetric_cost_no_prune_insertion_commutation_plateau_v1"
)
SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V2 = (
    "supported_projected_generalized_source_metric_no_overlap_trust_"
    "full_response_symmetric_cost_no_prune_insertion_commutation_plateau_v2"
)
SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_V1 = (
    "supported_projected_generalized_source_metric_no_overlap_trust_"
    "material_window_symmetric_cost_no_prune_v1"
)
SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_V1 = (
    "supported_projected_generalized_source_metric_no_overlap_trust_"
    "full_response_symmetric_cost_query_neutral_fs_prune_v1"
)
SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_FS_PRUNE_VERIFY_V1 = (
    "supported_projected_generalized_source_metric_no_overlap_trust_"
    "full_response_symmetric_cost_fs_prune_keep_verify_v1"
)
SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1 = (
    "supported_projected_generalized_source_metric_no_overlap_trust_"
    "material_window_symmetric_cost_fs_prune_keep_verify_v1"
)
SR_ROUTE_PROFILE_GUARDED_SINGLETON_POOL_V1 = (
    "supported_whitened_adaptive_trust_full_response_symmetric_cost_"
    "no_prune_guarded_singleton_pool_v1"
)
SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_V1 = (
    "supported_whitened_adaptive_trust_full_response_symmetric_cost_"
    "no_prune_macro_only_physical_lanes_v1"
)
SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_COMMUTATION_REDUCED_INSERTION_DIAGNOSTIC_V2 = (
    "supported_whitened_adaptive_trust_full_response_symmetric_cost_"
    "no_prune_macro_only_physical_lanes_commutation_reduced_insertion_"
    "diagnostic_v2"
)
SR_ROUTE_PROFILE_MACRO_ONLY_ALWAYS_INSERTION_FS_PRUNE_BEAM3X2_V1 = (
    "supported_whitened_adaptive_trust_full_response_symmetric_cost_"
    "fs_prune_nodamping_beam3x2_macro_only_commutation_reduced_"
    "insertion_v1"
)
SR_ROUTE_PROFILE_MACRO_ONLY_NO_LANES_COMMUTATION_REDUCED_INSERTION_V1 = (
    "supported_whitened_adaptive_trust_full_response_symmetric_cost_"
    "no_prune_macro_only_no_lanes_commutation_reduced_insertion_v1"
)
SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V1 = (
    "supported_whitened_adaptive_trust_full_response_symmetric_cost_"
    "no_prune_macro_only_physical_lanes_insertion_commutation_plateau_v1"
)
SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V2 = (
    "supported_whitened_adaptive_trust_full_response_symmetric_cost_"
    "no_prune_macro_only_physical_lanes_insertion_commutation_plateau_v2"
)
SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_ONE_SIDED_COST_V1 = (
    "supported_whitened_adaptive_trust_full_response_one_sided_cost_"
    "no_prune_macro_only_physical_lanes_v1"
)
SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_V1 = (
    "supported_whitened_adaptive_trust_full_response_symmetric_cost_"
    "fs_prune_nodamping_beam3x2_macro_only_physical_lanes_v1"
)
SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_ONE_SIDED_COST_V1 = (
    "supported_whitened_adaptive_trust_full_response_one_sided_cost_"
    "fs_prune_nodamping_beam3x2_macro_only_physical_lanes_v1"
)
SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1 = (
    "supported_whitened_adaptive_trust_full_response_symmetric_cost_"
    "fs_prune_nodamping_v1"
)
SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1 = (
    "supported_whitened_adaptive_trust_full_response_symmetric_cost_"
    "no_prune_beam_v1"
)
SR_ROUTE_PROFILE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1 = (
    "supported_whitened_adaptive_trust_full_response_no_novelty_"
    "metric_prune_beam_v1"
)
SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2 = (
    "supported_whitened_adaptive_trust_full_response_no_novelty_"
    "metric_prune_beam_h2o_derivative_resolved_v2"
)
SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3 = (
    "supported_whitened_adaptive_trust_full_response_symmetric_cost_"
    "no_prune_h2o_derivative_resolved_paper_i_v3"
)
SR_ROUTE_PROFILE_CANONICAL_ALIAS = "sr_snake_v1"
SR_ROUTE_PROFILE_CONVENTIONAL_ALIAS = "sr_snake"
SR_ROUTE_PROFILE_CONVENTIONAL_ALIAS_V2 = "sr_snake_v2"
SR_ROUTE_PROFILE_CONVENTIONAL_ALIAS_V3 = "sr_snake_v3"
SR_ROUTE_PROFILE_CANDIDATE_ALIAS_V4 = "sr_snake_v4"
SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_ALIAS_V1 = (
    "sr_snake_no_prune_symmetric_cost_v1"
)
SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_ALIAS_V1 = (
    "sr_snake_no_prune_symmetric_cost_projected_phase3_v1"
)
SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_ALIAS_V1 = (
    "sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1"
)
SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_ALIAS_V1 = (
    "sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_"
    "commutation_reduced_insertion_v1"
)
SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_ALIAS_V1 = (
    "insertion_commutation_plateau_v1"
)
SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_ALIAS_V2 = (
    "insertion_commutation_plateau_v2"
)
SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_ALIAS_V1 = (
    "sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_"
    "material_window_v1"
)
SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_ALIAS_V1 = (
    "sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_full_geometry_"
    "query_neutral_prune_v1"
)
SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_FS_PRUNE_VERIFY_ALIAS_V1 = (
    "sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_full_geometry_"
    "fs_prune_verify_v1"
)
SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_ALIAS_V1 = (
    "sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_material_"
    "window_fs_prune_verify_v1"
)
SR_ROUTE_PROFILE_GUARDED_SINGLETON_POOL_ALIAS_V1 = (
    "sr_snake_guarded_singleton_pool_v1"
)
SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_ALIAS_V1 = (
    "sr_snake_macro_only_physical_lanes_v1"
)
SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_COMMUTATION_REDUCED_INSERTION_DIAGNOSTIC_ALIAS_V2 = (
    "sr_snake_macro_only_physical_lanes_commutation_reduced_insertion_"
    "diagnostic_v2"
)
SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_ALIAS_V1 = (
    "sr_snake_macro_only_physical_lanes_insertion_commutation_plateau_v1"
)
SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_ALIAS_V2 = (
    "sr_snake_macro_only_physical_lanes_insertion_commutation_plateau_v2"
)
SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_ONE_SIDED_COST_ALIAS_V1 = (
    "sr_snake_macro_only_physical_lanes_one_sided_cost_v1"
)
SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_ALIAS_V1 = (
    "sr_snake_macro_only_physical_lanes_fs_prune_beam3x2_v1"
)
SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_ONE_SIDED_COST_ALIAS_V1 = (
    "sr_snake_macro_only_physical_lanes_fs_prune_beam3x2_one_sided_cost_v1"
)
SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_ALIAS_V1 = (
    "sr_snake_symmetric_cost_fs_prune_nodamping_v1"
)
SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_ALIAS_V1 = (
    "sr_snake_no_prune_symmetric_cost_beam_v1"
)
SR_ROUTE_PROFILE_NO_NOVELTY_METRIC_PRUNE_BEAM_ALIAS_V1 = (
    "sr_snake_no_novelty_metric_prune_beam_v1"
)
SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_ALIAS_V2 = (
    "sr_snake_h2o_derivative_resolved_v2"
)
SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_PAPER_I_ALIAS_V3 = (
    "sr_snake_h2o_derivative_resolved_paper_i_v3"
)

PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1 = (
    "full_active_plus_singleton_v1"
)
PHASE3_RESPONSE_COORDINATE_SCOPE_FIXED_LOCAL_WINDOW_V1 = "fixed_local_window_v1"
PHASE3_RESPONSE_COORDINATE_SCOPE_CANDIDATE_MATERIAL_COUPLING_WINDOW_V1 = (
    "candidate_material_coupling_window_v1"
)
PHASE3_RESPONSE_COORDINATE_SCOPE_LEGACY_REOPT_COUPLED_V1 = (
    "legacy_reopt_coupled_v1"
)
PHASE3_RESPONSE_COORDINATE_SCOPE_CHOICES = (
    PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1,
    PHASE3_RESPONSE_COORDINATE_SCOPE_FIXED_LOCAL_WINDOW_V1,
    PHASE3_RESPONSE_COORDINATE_SCOPE_CANDIDATE_MATERIAL_COUPLING_WINDOW_V1,
    PHASE3_RESPONSE_COORDINATE_SCOPE_LEGACY_REOPT_COUPLED_V1,
)
SR_ROUTE_PROFILE_REQUEST_CHOICES = (
    SR_ROUTE_PROFILE_REQUEST_OFF,
    SR_ROUTE_PROFILE_CANONICAL_ALIAS,
    SR_ROUTE_PROFILE_CANONICAL_V1,
    SR_ROUTE_PROFILE_CONVENTIONAL_ALIAS,
    SR_ROUTE_PROFILE_CONVENTIONAL_ALIAS_V2,
    SR_ROUTE_PROFILE_CONVENTIONAL_V2,
    SR_ROUTE_PROFILE_CONVENTIONAL_ALIAS_V3,
    SR_ROUTE_PROFILE_CONVENTIONAL_V3,
    SR_ROUTE_PROFILE_CANDIDATE_ALIAS_V4,
    SR_ROUTE_PROFILE_CANDIDATE_V4,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_ALIAS_V1,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_ALIAS_V1,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_V1,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_ALIAS_V1,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1,
    SR_ROUTE_PROFILE_GREEDY_BATCH_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1,
    SR_ROUTE_PROFILE_COMBINATORIAL_BATCH_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_ALIAS_V1,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_V1,
    SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_ALIAS_V1,
    SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V1,
    SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_ALIAS_V2,
    SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V2,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_ALIAS_V1,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_V1,
    SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_ALIAS_V1,
    SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_V1,
    SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_FS_PRUNE_VERIFY_ALIAS_V1,
    SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_FS_PRUNE_VERIFY_V1,
    SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_ALIAS_V1,
    SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1,
    SR_ROUTE_PROFILE_GUARDED_SINGLETON_POOL_ALIAS_V1,
    SR_ROUTE_PROFILE_GUARDED_SINGLETON_POOL_V1,
    SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_ALIAS_V1,
    SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_V1,
    SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_COMMUTATION_REDUCED_INSERTION_DIAGNOSTIC_ALIAS_V2,
    SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_COMMUTATION_REDUCED_INSERTION_DIAGNOSTIC_V2,
    SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_ALIAS_V1,
    SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V1,
    SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_ALIAS_V2,
    SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V2,
    SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_ONE_SIDED_COST_ALIAS_V1,
    SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_ONE_SIDED_COST_V1,
    SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_ALIAS_V1,
    SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_V1,
    SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_ONE_SIDED_COST_ALIAS_V1,
    SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_ONE_SIDED_COST_V1,
    SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_ALIAS_V1,
    SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_ALIAS_V1,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1,
    SR_ROUTE_PROFILE_NO_NOVELTY_METRIC_PRUNE_BEAM_ALIAS_V1,
    SR_ROUTE_PROFILE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1,
    SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_ALIAS_V2,
    SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2,
    SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_PAPER_I_ALIAS_V3,
    SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3,
)

SR_ROUTE_PROFILE_CONTRACT_SCHEMA = "sr_snake_route_profile_contract_v1"
SR_ROUTE_PROFILE_CONTRACT_DIGEST_SCHEMA = (
    "sr_snake_route_profile_contract_sha256_v1"
)

_HISTORICAL_COMMAND_SHA256 = (
    "37751de2805875337cb8a0034a7394b02344c893e1b0a583439b1954c7c8061e"
)
_HISTORICAL_RESULT_SHA256 = (
    "f8d2bb9756d395d7806bb2f365d95a5fcb4c5aa6de55e96f89ecfc35295b10da"
)
_SELF_CONTAINED_ARCHIVE_SHA256 = (
    "c290d9ee1b31cd211e41faad174cd2e311ca65cf351c46bbb84fbaaea9504c6c"
)
_CONVENTIONAL_V2_SOURCE_ARCHIVE_SHA256 = (
    "f0ced05fb7c4ab242ef10323c13ac0e3e3d5be2c15b255c931b73aa8a980cbae"
)
_CONVENTIONAL_V2_SOURCE_LOCK_ROOT = (
    "raw_outputs/paper_i_hh_sr_snake_weak_weak_full_accepted_refit_whitened_"
    "20260715/source_lock"
)
_CONVENTIONAL_V2_SOURCE_MANIFEST_SHA256 = (
    "7e0b8d4b72bcabb5de842ae89afd8d34ce861ebf0e2b9d16fe12f883c73aa416"
)

_CONVENTIONAL_V2_WEAK_HOLSTEIN_ANCHORS: tuple[dict[str, Any], ...] = (
    {
        "regime": "weak-weak",
        "u": 0.25,
        "n_ph_work": 2,
        "source_root": (
            "raw_outputs/paper_i_hh_sr_snake_weak_weak_full_accepted_refit_"
            "whitened_20260715/expanded_runtime_projected_logical_v1_r30"
        ),
        "command_sha256": (
            "7823df8a3cb4c900a0d9a21366c18e354cd982804bdc8ad52199541ec20bd800"
        ),
        "result_sha256": (
            "ef43248e5a9c9893a4555f753c2871d42aca95d4143b6878fd4be04dddac6172"
        ),
        "exit_status_sha256": (
            "60f06f5101b2fd22d0f3fb02b657b8f1a0f2153980cb9108800f7daafeec9e12"
        ),
        "settings_diff_sha256": (
            "de0f9cd5fefc88ea27557480b84eeb9d5bc4186e36792e0e5f18a0b4a117501f"
        ),
        "energy": -0.9183531184618378,
        "same_cutoff_reference_energy": -0.9183531194991743,
        "absolute_error": 1.0373365499916076e-9,
        "active_ansatz_depth": 25,
        "controller_rounds": 30,
    },
    {
        "regime": "intermediate-weak",
        "u": 1.25,
        "n_ph_work": 2,
        "source_root": (
            "raw_outputs/paper_i_hh_sr_snake_intermediate_weak_full_accepted_"
            "refit_whitened_20260715/expanded_runtime_projected_logical_v1_r30"
        ),
        "command_sha256": (
            "576e402f29117726f1dcaea82b2a0654a6089e2034d836a8d1a2f18ed63adfdd"
        ),
        "result_sha256": (
            "4fa47aa4285c40a53a9e88f0ffe57d10be8841cae59b9effd3f692da20318c8d"
        ),
        "exit_status_sha256": (
            "8843126145519c1f17fc0a235b89ecbb34ca2de78f61f0c79a117d4c653db945"
        ),
        "settings_diff_sha256": (
            "b2209ff810b4713698b9077576d771baf49fff92a771f9141eb29b591478c6f3"
        ),
        "energy": -0.49499563910841426,
        "same_cutoff_reference_energy": -0.49499563910870126,
        "absolute_error": 2.8699265186560297e-13,
        "result_embedded_reference_energy": -0.49499563910866023,
        "result_embedded_absolute_error": 2.4596991110570343e-13,
        "active_ansatz_depth": 27,
        "controller_rounds": 30,
    },
    {
        "regime": "strong-weak-u8",
        "u": 8.0,
        "n_ph_work": 2,
        "source_root": (
            "raw_outputs/paper_i_hh_sr_snake_strong_weak_u8_full_accepted_"
            "refit_whitened_20260715/expanded_runtime_projected_logical_v1_r30"
        ),
        "command_sha256": (
            "3340e423bf654ba562655ec820023eb76afbb23150941066de120a444b4bc3e1"
        ),
        "result_sha256": (
            "541966c89e0cf5523cb6b66260c5bb6a2f5b1cff2e4e81c617d34c984f351bc1"
        ),
        "exit_status_sha256": (
            "af4c2c374aaa07e1a31b1bb5016c89d39bc368ed8af212dd033fc970c653b159"
        ),
        "settings_diff_sha256": (
            "aa97c51a0310dbcffadf412dc163351f14d2ab4105370af73567397d9d4e1a61"
        ),
        "energy": 0.5264600841532714,
        "same_cutoff_reference_energy": 0.5264587007998404,
        "absolute_error": 1.3833534310281337e-6,
        "active_ansatz_depth": 24,
        "controller_rounds": 30,
    },
)


# Namespace destinations are used deliberately: this is the one executable
# source for expanding either versioned SR route profile.  Regime physics
# (U, g, n_ph_max, and the exact same-cutoff reference) remains outside the
# method profile and must be supplied by the regime's source lock.
CANONICAL_SR_SNAKE_V1_EXECUTION_SETTINGS: dict[str, Any] = {
    "problem": "hh",
    "adapt_pool": "full_meta",
    "adapt_pool_class_filter_json": None,
    "adapt_pool_label_filter_json": None,
    "adapt_selected_logical_source_json": None,
    "adapt_selected_logical_mode": "off",
    "adapt_continuation_mode": "phase3_v1",
    "static_route_id": "route_a",
    "static_meta_feature_profile": "paper_i_production_v1",
    "static_lane_route": "physical_operator_type",
    "physical_lane_shortlist_aggressiveness": 3,
    "adapt_reoptimization_route": "off",
    "adapt_formal_manifold_route_profile": "off",
    "adapt_formal_manifold_config_json": None,
    "historical_singleton_coordinate_solve_policy": (
        "supported_metric_whitened_eigh_v1"
    ),
    "historical_singleton_coordinate_solve_scope": "phase3_only_v1",
    "historical_singleton_trust_region_update_policy": (
        "displacement_calibrated_unbounded_v2"
    ),
    "sr_powell_coordinate_chart_policy": (
        "expanded_runtime_projected_logical_v1"
    ),
    "sr_escape_mode": "disabled",
    "sr_controller_ablation_contract": "off",
    "phase2_gram_novelty_policy": "ordinary_multiplier_v1",
    "phase3_gram_novelty_policy": "ordinary_multiplier_v1",
    "phase3_novelty_ablation_mode": "off",
    "phase2_novelty_mode": "collective_span_v1",
    "phase2_selector_gain_mode": "trust_region_v1",
    "phase2_rho": 0.25,
    "adapt_inner_optimizer": "POWELL",
    "adapt_maxiter": 200,
    "adapt_scipy_maxfev": 0,
    "adapt_state_backend": "compiled",
    "adapt_seed": 7,
    "adapt_reopt_policy": "windowed",
    "adapt_window_size": 3,
    "adapt_window_topk": 0,
    "adapt_full_refit_every": 8,
    "adapt_final_full_refit": "true",
    "adapt_final_refit_maxiter": 200,
    "adapt_insertion_mode": "append_only",
    "adapt_max_depth": 30,
    "adapt_allow_repeats": True,
    "adapt_finite_angle_fallback": True,
    "adapt_finite_angle": 0.1,
    "adapt_finite_angle_min_improvement": 1.0e-12,
    "adapt_disable_hh_seed": False,
    "phase0_pilot_enabled": False,
    "phase1_shortlist_size": 24,
    "phase2_shortlist_size": 12,
    "phase2_shortlist_fraction": 0.25,
    "phase2_enable_batching": False,
    "phase3_enable_batching": False,
    "phase3_runtime_split_mode": "shortlist_pauli_children_v1",
    "allow_archival_phase3_runtime_split": True,
    "phase3_runtime_split_selection_mode": "archival_child_set_forward_v1",
    "phase3_runtime_split_max_subset_size": 1,
    "phase3_runtime_split_subset_sizes": "1",
    "phase3_runtime_split_child_set_symmetry_policy": "hard_guard",
    "phase3_runtime_split_child_padding_policy": "exact_projected_grouped_v1",
    "adapt_child_pool_expansion_mode": "off",
    "adapt_child_pool_expansion_symmetry_policy": "off",
    "shared_pauli_pool_mode": "off",
    "shared_pauli_pool_symmetry_policy": "off",
    "phase1_prune_enabled": True,
    "phase1_prune_policy": "recoverability_ladder_v1",
    "phase1_prune_mode": "both",
    "phase1_prune_fraction": 0.25,
    "phase1_prune_min_candidates": 1,
    "phase1_prune_max_candidates": 6,
    "phase1_prune_max_regression": 1.0e-8,
    "phase1_prune_tolerance_mode": "auto",
    "phase1_prune_tolerance_shot_coeff": 0.0,
    "phase1_prune_tolerance_screen_coeff": 0.01,
    "phase1_prune_tolerance_chem": 0.0,
    "phase1_prune_tolerance_rel_coeff": 0.0,
    "phase1_prune_tolerance_target_energy": None,
    "phase1_prune_retained_gain_ratio": 0.5,
    "phase1_prune_protect_steps": 2,
    "phase1_prune_stale_age": 2,
    "phase1_prune_stagnation_threshold": 0.0,
    "phase1_prune_small_theta_abs": 1.0e-3,
    "phase1_prune_small_theta_relative": 0.5,
    "phase1_prune_cooldown_steps": 2,
    "phase1_prune_local_window_size": 4,
    "phase1_prune_old_fraction": 0.25,
    "phase1_prune_checkpoint_period": 3,
    "phase1_prune_maturity_threshold": 0.5,
    "phase1_prune_snr_threshold": 1.0,
    "phase1_prune_prefilter_policy": "off",
    "phase1_prune_prefilter_json": None,
    "phase1_prune_risk_threshold": 0.0,
    "phase1_prune_prefilter_max_candidates": 1,
    "adapt_beam_live_branches": 3,
    "adapt_beam_children_per_parent": 2,
    "adapt_beam_terminated_keep": None,
    "adapt_beam_terminal_archive_mode": "disabled",
    "adapt_beam_lambda": 0.005,
    "adapt_beam_parent_workers": 1,
    "phase3_selector_policy": "algebraic_nested_v1",
    "phase3_selector_geometry_mode": "reduced",
    "phase3_geometry_window_size": 0,
    "phase3_window_relaxation_mode": "reduced",
    "phase3_backend_cost_mode": "marrakesh_graph_span_v1",
    "phase3_backend_name": "FakeMarrakesh",
    "phase3_backend_transpile_seed": 7,
    "phase3_backend_optimization_level": 1,
    "phase3_hardware_cost_normalization_mode": "family_robust_v1",
    "phase3_lifetime_cost_mode": "phase3_v1",
    "phase3_enable_rescue": False,
    "phase3_symmetry_mitigation_mode": "off",
    "phase3_plateau_acquisition_mode": "off",
    "phase3_plateau_seed_probe_mode": "off",
    "phase3_shadow_legacy_geometry_mode": "off",
    "phase3_shadow_legacy_max_depth": 0,
    "phase3_parent_collapse_debug_max_depth": 0,
    "hardware_resolution_mode": "ideal",
    "gradient_hw_floor": 0.0,
    "gradient_drift_floor": 0.0,
}


# These values are intentionally outside the historical v1 contract payload so
# its published digest remains unchanged.  Explicit v1 requests nevertheless
# materialize and validate them, preventing a later accepted-refit feature from
# silently changing historical replay.
HISTORICAL_SR_SNAKE_V1_ACCEPTED_REFIT_SETTINGS: dict[str, Any] = {
    "adapt_accepted_refit_scope": "selector_policy_v1",
    "adapt_accepted_refit_coordinate_chart": "native_v1",
    "adapt_accepted_refit_base_chart_policy": "logical_shared_reduced_v1",
}


# The Phase-III response scope is deliberately outside the frozen v1/v2
# contract payloads so their published digests remain byte-for-byte stable.
# Both historical profiles nevertheless materialize and validate the explicit
# legacy policy at runtime.  This prevents a generic zero-valued window default
# from silently changing the meaning of a historical replay.
HISTORICAL_SR_SNAKE_RESPONSE_SCOPE_SETTINGS: dict[str, Any] = {
    "phase3_response_coordinate_scope": (
        PHASE3_RESPONSE_COORDINATE_SCOPE_LEGACY_REOPT_COUPLED_V1
    ),
}


# The retained historical profiles still resolve their score and measured-
# curvature requirements explicitly.  The retired metric-for-Hessian proxy
# policies are intentionally absent.
HISTORICAL_SR_SNAKE_PHASE12_ENERGY_MODEL_SETTINGS: dict[str, Any] = {
    "phase1_score_mode": PHASE1_SCORE_MODE_TRUST_REGION_V1,
    "phase2_curvature_policy": PHASE2_CURVATURE_POLICY_LEGACY_OPTIONAL_V1,
}


CANONICAL_SR_SNAKE_V2_EXECUTION_SETTINGS: dict[str, Any] = {
    **CANONICAL_SR_SNAKE_V1_EXECUTION_SETTINGS,
    "phase1_prune_recovery_trust_radius": 0.0,
    "phase1_prune_schur_nomination_route": "hessian_coupling_v1",
    "phase1_prune_metric_schur_mu": 1.0e-6,
    "phase1_prune_metric_schur_solve_mode": "stationary_gw_zero_v1",
    "phase1_prune_metric_schur_cost_weighting": (
        "ansatz_entry_denominator_v1"
    ),
    "phase1_prune_live_min_depth": 0,
    "adapt_accepted_refit_scope": "full_ansatz_v1",
    "adapt_accepted_refit_coordinate_chart": "supported_fs_whitened_fixed_v1",
    "adapt_accepted_refit_base_chart_policy": (
        "expanded_runtime_projected_logical_v1"
    ),
}


CANONICAL_SR_SNAKE_V3_EXECUTION_SETTINGS: dict[str, Any] = {
    **CANONICAL_SR_SNAKE_V2_EXECUTION_SETTINGS,
    "phase3_response_coordinate_scope": (
        PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1
    ),
}


CANONICAL_SR_SNAKE_V4_EXECUTION_SETTINGS: dict[str, Any] = {
    **CANONICAL_SR_SNAKE_V3_EXECUTION_SETTINGS,
    # V4 promises a full Phase-III response on every controller round.  The
    # stage controller must therefore never retire Phase III into a Phase-II-
    # only admission path.
    "phase_live_hysteresis_enabled": False,
    # Conventional SR-SNAKE grows exclusively through controller admissions.
    # The legacy HH quadrature preseed would add eight operators before round
    # one and make controller-round and ansatz-depth horizons disagree.
    "adapt_disable_hh_seed": True,
    # Phase I is a genuinely first-order energy screen.  Phase II must carry
    # a finite measured directional-curvature receipt for every scored
    # candidate.
    "phase1_score_mode": PHASE1_SCORE_MODE_TRUST_REGION_V1,
    "phase1_energy_model": PHASE1_ENERGY_MODEL_FIRST_ORDER_FS_TRUST_V1,
    "phase2_curvature_policy": (
        PHASE2_CURVATURE_POLICY_MEASURED_REQUIRED_FAIL_CLOSED_V1
    ),
    "phase2_cheap_curvature_proxy_policy": (
        PHASE2_CHEAP_CURVATURE_PROXY_POLICY_OFF
    ),
    # Finite-angle energy probing is outside the v4 response-model contract.
    # Keep it explicitly disabled so neither a generic parser default nor a
    # source-locked command can silently activate the guard.
    "adapt_finite_angle_fallback": False,
    # Candidate admission remains singleton/no-batch, but the ordinary
    # novelty multiplier is removed.  The all-models-infeasible geometry
    # fallback remains active through fallback_only_v1.
    "phase2_gram_novelty_policy": "fallback_only_v1",
    "phase3_gram_novelty_policy": "fallback_only_v1",
    "phase3_hardware_cost_normalization_mode": (
        "family_robust_symmetric_arctan_v1"
    ),
    # Hold a single live branch so beam work cannot alter the response model.
    "adapt_beam_live_branches": 1,
    "adapt_beam_children_per_parent": 1,
    "adapt_beam_terminated_keep": 0,
    "adapt_beam_terminal_archive_mode": "disabled",
    # Every accepted admission already receives a full supported-FS refit;
    # no terminal-only refit or prune mutation is permitted in this profile.
    "adapt_final_full_refit": "false",
    "phase1_prune_enabled": True,
    "phase1_prune_policy": "recoverability_ladder_v1",
    "phase1_prune_mode": "live",
    "phase1_prune_max_candidates": 1,
    "phase1_prune_local_window_size": 0,
    "phase1_prune_recovery_trust_radius": 0.125,
    "phase1_prune_schur_nomination_route": (
        "full_logical_fs_trust_delete_refit_v1"
    ),
    "phase1_prune_metric_schur_mu": 0.0,
    "phase1_prune_metric_schur_solve_mode": "affine_deletion_global_trust_v1",
    "phase1_prune_metric_schur_cost_weighting": "off",
    "phase1_prune_trust_update_policy": "modeled_local_fs_conservative_v1",
    "phase1_prune_metric_mu_update_policy": (
        "same_trial_underprediction_monotone_v1"
    ),
    "phase1_prune_endpoint_overlap_policy": "off",
    "phase3_shadow_damping_policy": "mapped_seed_zero_query_v1",
}


CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_V1_EXECUTION_SETTINGS: dict[
    str, Any
] = {
    # This experimental identity intentionally leaves the controller horizon
    # outside the method contract.  Each regime source lock must supply its
    # approved 30- or 50-round horizon explicitly.
    **{
        field: value
        for field, value in CANONICAL_SR_SNAKE_V3_EXECUTION_SETTINGS.items()
        if field != "adapt_max_depth"
    },
    "adapt_disable_hh_seed": True,
    "phase1_score_mode": PHASE1_SCORE_MODE_TRUST_REGION_V1,
    "phase1_energy_model": PHASE1_ENERGY_MODEL_FIRST_ORDER_FS_TRUST_V1,
    "phase2_curvature_policy": (
        PHASE2_CURVATURE_POLICY_MEASURED_REQUIRED_FAIL_CLOSED_V1
    ),
    "phase2_cheap_curvature_proxy_policy": (
        PHASE2_CHEAP_CURVATURE_PROXY_POLICY_OFF
    ),
    "adapt_finite_angle_fallback": False,
    # Ordinary novelty multiplication is disabled.  fallback_only_v1 retains
    # the bounded all-energy-models-infeasible safety path and its telemetry.
    "phase2_gram_novelty_policy": "fallback_only_v1",
    "phase3_gram_novelty_policy": "fallback_only_v1",
    "phase3_hardware_cost_normalization_mode": (
        "family_robust_symmetric_arctan_v1"
    ),
    "adapt_beam_live_branches": 1,
    "adapt_beam_children_per_parent": 1,
    "adapt_beam_terminated_keep": 0,
    "adapt_beam_terminal_archive_mode": "disabled",
    # The validated six-regime Main SR source lock (cluster 8887574) keeps
    # Phase III live for the complete controller horizon.
    "phase_live_hysteresis_enabled": False,
    # Full supported-FS refitting remains the ordinary post-admission action;
    # periodic and terminal-only refit mutations are disabled.
    "adapt_full_refit_every": 0,
    "adapt_final_full_refit": "false",
    "phase1_prune_enabled": False,
    "phase3_shadow_damping_policy": "off",
}


CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_V1_EXECUTION_SETTINGS: dict[
    str, Any
] = {
    **CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_V1_EXECUTION_SETTINGS,
    # The only scientific setting changed from the validated Main SR parent.
    # Phase III removes raw-Gram null directions, then solves its generalized
    # FS trust problem without constructing a metric inverse square root.
    "historical_singleton_coordinate_solve_policy": (
        "supported_metric_projected_generalized_trust_v1"
    ),
}


CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1_EXECUTION_SETTINGS: dict[
    str, Any
] = {
    **CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_V1_EXECUTION_SETTINGS,
    # Reuse the source-point supported Gram matrix to calibrate accepted
    # parameter motion.  No endpoint state overlap is measured or charged.
    "historical_singleton_trust_region_update_policy": (
        "source_metric_inverse_sqrt_no_overlap_v1"
    ),
}


CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_V1_EXECUTION_SETTINGS: dict[
    str, Any
] = {
    **CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1_EXECUTION_SETTINGS,
    "adapt_insertion_mode": "full_commutation_reduced",
}


CANONICAL_SR_SNAKE_INSERTION_COMMUTATION_PLATEAU_V1_EXECUTION_SETTINGS: dict[
    str, Any
] = {
    **CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1_EXECUTION_SETTINGS,
    # The cumulative-relative threshold is serialized below; the only
    # legacy-executor switch is its named insertion policy, leaving the active
    # Paper-I parent otherwise byte-for-byte intact.
    "adapt_insertion_mode": "insertion_commutation_plateau_v1",
}


CANONICAL_SR_SNAKE_INSERTION_COMMUTATION_PLATEAU_V2_EXECUTION_SETTINGS: dict[
    str, Any
] = {
    **CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1_EXECUTION_SETTINGS,
    # V2 compares the latest accepted decrease with the mean of all prior
    # accepted decreases.  Every other parent setting remains unchanged.
    "adapt_insertion_mode": "insertion_commutation_plateau_v2",
}


CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_V1_EXECUTION_SETTINGS: dict[
    str, Any
] = {
    **CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1_EXECUTION_SETTINGS,
    # Replace only the eager full Phase-III geometry construction.  The
    # material window is candidate-coupling driven and is independent of the
    # full accepted-ansatz Powell refit chart/window.
    "phase3_response_coordinate_scope": (
        PHASE3_RESPONSE_COORDINATE_SCOPE_CANDIDATE_MATERIAL_COUPLING_WINDOW_V1
    ),
}


CANONICAL_SR_SNAKE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_V1_EXECUTION_SETTINGS: dict[
    str, Any
] = {
    **CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1_EXECUTION_SETTINGS,
    # Reuse the selected full active-plus-singleton Phase-III model to
    # nominate at most one affine deletion.  The combined admission/deletion
    # state receives the route's one ordinary accepted-state Powell refit;
    # pruning does not launch a measured sibling or any dedicated estimator.
    "phase1_prune_enabled": True,
    "phase1_prune_policy": "recoverability_ladder_v1",
    "phase1_prune_mode": "live",
    "phase1_prune_max_candidates": 1,
    "phase1_prune_local_window_size": 0,
    # Start at the radius reached only after four rejected nominations in the
    # first-hit weak--weak smoke.  This makes the prune screen conservative
    # from round one without changing its zero-query source geometry.
    "phase1_prune_recovery_trust_radius": 0.00390625,
    "phase1_prune_schur_nomination_route": (
        "full_logical_fs_trust_delete_refit_v1"
    ),
    "phase1_prune_metric_schur_mu": 0.0,
    "phase1_prune_metric_schur_solve_mode": (
        "affine_deletion_global_trust_v1"
    ),
    "phase1_prune_metric_schur_cost_weighting": "off",
    "phase1_prune_trust_update_policy": (
        "modeled_local_fs_conservative_v1"
    ),
    "phase1_prune_metric_mu_update_policy": "off",
    "phase1_prune_endpoint_overlap_policy": "off",
}


CANONICAL_SR_SNAKE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_FS_PRUNE_VERIFY_V1_EXECUTION_SETTINGS: dict[
    str, Any
] = {
    **CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1_EXECUTION_SETTINGS,
    # One-factor child of the active fd5ec route. The post-admission full
    # refit remains unchanged; this policy nominates at most one old logical
    # coordinate and measures one immutable delete/refit sibling.
    "phase1_prune_enabled": True,
    "phase1_prune_policy": "recoverability_ladder_v1",
    "phase1_prune_mode": "live",
    "phase1_prune_max_candidates": 1,
    "phase1_prune_local_window_size": 0,
    "phase1_prune_recovery_trust_radius": 0.125,
    "phase1_prune_schur_nomination_route": (
        "full_logical_fs_trust_delete_refit_v1"
    ),
    "phase1_prune_metric_schur_mu": 0.0,
    "phase1_prune_metric_schur_solve_mode": (
        "affine_deletion_global_trust_v1"
    ),
    "phase1_prune_metric_schur_cost_weighting": "off",
    "phase1_prune_trust_update_policy": (
        "modeled_local_fs_conservative_v1"
    ),
    "phase1_prune_metric_mu_update_policy": "off",
    "phase1_prune_endpoint_overlap_policy": "off",
}


CANONICAL_SR_SNAKE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1_EXECUTION_SETTINGS: dict[
    str, Any
] = {
    **CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_V1_EXECUTION_SETTINGS,
    # Admission remains the inherited effective 1x1 route.  These controls
    # activate only one live deletion nomination whose measured delete/refit
    # sibling is compared with an immutable keep branch.
    "phase1_prune_enabled": True,
    "phase1_prune_policy": "recoverability_ladder_v1",
    "phase1_prune_mode": "live",
    "phase1_prune_max_candidates": 1,
    "phase1_prune_local_window_size": 0,
    "phase1_prune_recovery_trust_radius": 0.125,
    "phase1_prune_schur_nomination_route": (
        "full_logical_fs_trust_delete_refit_v1"
    ),
    "phase1_prune_metric_schur_mu": 0.0,
    "phase1_prune_metric_schur_solve_mode": (
        "affine_deletion_global_trust_v1"
    ),
    "phase1_prune_metric_schur_cost_weighting": "off",
    "phase1_prune_trust_update_policy": (
        "modeled_local_fs_conservative_v1"
    ),
    "phase1_prune_metric_mu_update_policy": "off",
    "phase1_prune_endpoint_overlap_policy": "off",
}


CANONICAL_SR_SNAKE_GUARDED_SINGLETON_POOL_V1_EXECUTION_SETTINGS: dict[
    str, Any
] = {
    # Controlled pool-contract ablation of the corrected no-prune,
    # symmetric-cost SR route.  The full_meta macros are decomposition sources
    # only: no macro candidate may enter Phase I.
    **CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_V1_EXECUTION_SETTINGS,
    # One global candidate population: neither physical-operator nor
    # algebraic lane allocation is active.
    "static_lane_route": "algebraic",
    # The algebraic static route normalizes the inactive physical-lane
    # aggressiveness to one.  Record that inert sentinel explicitly so runtime
    # conformance cannot mistake lane deactivation for settings drift.
    "physical_lane_shortlist_aggressiveness": 1,
    "phase3_selector_policy": "hardware_resolvable_v1",
    "phase_live_hysteresis_enabled": False,
    "phase3_runtime_split_mode": "off",
    "allow_archival_phase3_runtime_split": False,
    # The archival Phase-III splitter is disabled for this route, so retain its
    # existing inactive sentinel.  Raw children receive the separate, active
    # legal-codeword hard filter while the shared pool is constructed.
    "phase3_runtime_split_child_padding_policy": "unchecked_diagnostic_v1",
    "shared_pauli_pool_mode": "guarded_singleton_children_only_v1",
    "shared_pauli_pool_symmetry_policy": "hard_guard",
    "shared_pauli_pool_max_subset_size": 1,
    "shared_pauli_pool_subset_sizes": "1",
}


CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_V1_EXECUTION_SETTINGS: dict[
    str, Any
] = {
    # Logical complement of the guarded raw-child pool: retain the ordered
    # full_meta/HVA parent generators only and never materialize children.
    **CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_V1_EXECUTION_SETTINGS,
    "static_lane_route": "physical_operator_type",
    "phase_live_hysteresis_enabled": False,
    "phase3_runtime_split_mode": "off",
    "allow_archival_phase3_runtime_split": False,
    "phase3_runtime_split_child_padding_policy": "unchecked_diagnostic_v1",
    "adapt_child_pool_expansion_mode": "off",
    "shared_pauli_pool_mode": "off",
    "shared_pauli_pool_symmetry_policy": "off",
}


CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_COMMUTATION_REDUCED_INSERTION_DIAGNOSTIC_V2_EXECUTION_SETTINGS: dict[
    str, Any
] = {
    # Preserve the exhaustive V1 diagnostic while scoring only one canonical
    # representative from each exactly certified commuting insertion class.
    **CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_V1_EXECUTION_SETTINGS,
    "adapt_insertion_mode": "full_commutation_reduced",
}


CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V1_EXECUTION_SETTINGS: dict[
    str, Any
] = {
    # One-factor diagnostic: preserve the page-2 macro route and activate the
    # commutation-reduced insertion domain only after an accepted weak step.
    **CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_V1_EXECUTION_SETTINGS,
    "adapt_insertion_mode": "insertion_commutation_plateau_v1",
}


CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V2_EXECUTION_SETTINGS: dict[
    str, Any
] = {
    # Apply the same historical-average trigger to the macro parent without
    # changing its pool, selection, trust, or refit settings.
    **CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_V1_EXECUTION_SETTINGS,
    "adapt_insertion_mode": "insertion_commutation_plateau_v2",
}


CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_ONE_SIDED_COST_V1_EXECUTION_SETTINGS: dict[
    str, Any
] = {
    # Exact one-variable cost-normalization ablation of the macro-only route.
    **CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_V1_EXECUTION_SETTINGS,
    "phase3_hardware_cost_normalization_mode": "family_robust_v1",
}


CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_V1_EXECUTION_SETTINGS: dict[
    str, Any
] = {
    # Macro-only complement with the already registered undamped FS-trust
    # deletion model and exact historical 3x2 beam semantics.
    **CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_V1_EXECUTION_SETTINGS,
    "phase1_prune_enabled": True,
    "phase1_prune_policy": "recoverability_ladder_v1",
    "phase1_prune_mode": "live",
    "phase1_prune_max_candidates": 1,
    "phase1_prune_local_window_size": 0,
    "phase1_prune_recovery_trust_radius": 0.125,
    "phase1_prune_schur_nomination_route": (
        "full_logical_fs_trust_delete_refit_v1"
    ),
    "phase1_prune_metric_schur_mu": 0.0,
    "phase1_prune_metric_schur_solve_mode": (
        "affine_deletion_global_trust_v1"
    ),
    "phase1_prune_metric_schur_cost_weighting": "off",
    "phase1_prune_trust_update_policy": (
        "modeled_local_fs_conservative_v1"
    ),
    "phase1_prune_metric_mu_update_policy": "off",
    "phase1_prune_endpoint_overlap_policy": "off",
    "adapt_beam_live_branches": 3,
    "adapt_beam_children_per_parent": 2,
    "adapt_beam_terminated_keep": 3,
    "adapt_beam_terminal_archive_mode": "legacy",
    "adapt_beam_lambda": 0.005,
}


CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_ONE_SIDED_COST_V1_EXECUTION_SETTINGS: dict[
    str, Any
] = {
    # Exact one-variable cost-normalization control of the symmetric arm.
    **CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_V1_EXECUTION_SETTINGS,
    "phase3_hardware_cost_normalization_mode": "family_robust_v1",
}


CANONICAL_SR_SNAKE_SYMMETRIC_COST_FS_PRUNE_V1_EXECUTION_SETTINGS: dict[
    str, Any
] = {
    # Appendix-only one-factor pruning ablation of the main 1x1 profile.
    **CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_V1_EXECUTION_SETTINGS,
    "phase_live_hysteresis_enabled": False,
    "phase1_prune_enabled": True,
    "phase1_prune_policy": "recoverability_ladder_v1",
    "phase1_prune_mode": "live",
    "phase1_prune_max_candidates": 1,
    # Zero means the complete active logical ansatz enters the deletion model.
    "phase1_prune_local_window_size": 0,
    "phase1_prune_recovery_trust_radius": 0.125,
    "phase1_prune_schur_nomination_route": (
        "full_logical_fs_trust_delete_refit_v1"
    ),
    # Keep H + mu G scientific damping inactive.  The explicit FS radius is
    # the only geometric regularizer in this one-factor pruning ablation.
    "phase1_prune_metric_schur_mu": 0.0,
    "phase1_prune_metric_schur_solve_mode": (
        "affine_deletion_global_trust_v1"
    ),
    "phase1_prune_metric_schur_cost_weighting": "off",
    "phase1_prune_trust_update_policy": (
        "modeled_local_fs_conservative_v1"
    ),
    "phase1_prune_metric_mu_update_policy": "off",
    "phase1_prune_endpoint_overlap_policy": "off",
}


CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1_EXECUTION_SETTINGS: dict[
    str, Any
] = {
    # Appendix-only exact historical 3x2 beam ablation of the main 1x1
    # profile.  The legacy archive mode preserves the pre-2026-07-04
    # stop-or-admit branch semantics: every expanded parent also yields a
    # terminated stop child, and the best three terminated branches persist
    # cumulatively across later controller rounds.
    **CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_V1_EXECUTION_SETTINGS,
    "phase_live_hysteresis_enabled": False,
    "adapt_beam_live_branches": 3,
    "adapt_beam_children_per_parent": 2,
    "adapt_beam_terminated_keep": 3,
    "adapt_beam_terminal_archive_mode": "legacy",
    "adapt_beam_lambda": 0.005,
}


CANONICAL_SR_SNAKE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1_EXECUTION_SETTINGS: dict[
    str, Any
] = {
    **CANONICAL_SR_SNAKE_V3_EXECUTION_SETTINGS,
    "problem": "molecular_vibronic_h2o_linear_fd",
    "adapt_max_depth": 50,
    "phase3_runtime_split_child_padding_policy": "full_binary_code_space_v1",
    "phase3_backend_cost_mode": "proxy",
    # Preserve novelty measurements as diagnostics while removing them from
    # both Phase-II and Phase-III selection decisions.
    "phase3_novelty_ablation_mode": "all",
    # Retain the conventional SR 3x2 beam and singleton admission contract,
    # but use the metric-regularized deletion nomination requested for H2O.
    "phase1_prune_schur_nomination_route": "metric_regularized_v1",
    "phase1_prune_metric_schur_mu": 0.01,
}


CANONICAL_SR_SNAKE_H2O_DERIVATIVE_RESOLVED_V2_EXECUTION_SETTINGS: dict[
    str, Any
] = {
    **CANONICAL_SR_SNAKE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1_EXECUTION_SETTINGS,
    "adapt_pool": "full_meta_derivative_resolved_v2",
}


CANONICAL_SR_SNAKE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3_EXECUTION_SETTINGS: dict[
    str, Any
] = {
    **CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_V1_EXECUTION_SETTINGS,
    "problem": "molecular_vibronic_h2o_linear_fd",
    "adapt_max_depth": 50,
    "adapt_pool": "full_meta_derivative_resolved_v2",
    "phase3_runtime_split_child_padding_policy": "full_binary_code_space_v1",
    "phase3_backend_cost_mode": "proxy",
}


_DEST_OPTION_STRINGS: dict[str, tuple[str, ...]] = {
    "problem": ("--problem",),
    "adapt_pool": ("--adapt-pool",),
    "adapt_pool_class_filter_json": ("--adapt-pool-class-filter-json",),
    "adapt_pool_label_filter_json": ("--adapt-pool-label-filter-json",),
    "adapt_selected_logical_source_json": ("--adapt-selected-logical-source-json",),
    "adapt_selected_logical_mode": ("--adapt-selected-logical-mode",),
    "adapt_continuation_mode": ("--adapt-continuation-mode",),
    "static_route_id": ("--static-route-id",),
    "static_meta_feature_profile": ("--static-meta-feature-profile",),
    "static_lane_route": ("--static-lane-route",),
    "physical_lane_shortlist_aggressiveness": (
        "--physical-lane-shortlist-aggressiveness",
    ),
    "adapt_reoptimization_route": ("--adapt-reoptimization-route",),
    "adapt_formal_manifold_route_profile": (
        "--adapt-formal-manifold-route-profile",
    ),
    "adapt_formal_manifold_config_json": (
        "--adapt-formal-manifold-config-json",
    ),
    "historical_singleton_coordinate_solve_policy": (
        "--historical-singleton-coordinate-solve-policy",
    ),
    "historical_singleton_coordinate_solve_scope": (
        "--historical-singleton-coordinate-solve-scope",
    ),
    "historical_singleton_trust_region_update_policy": (
        "--historical-singleton-trust-region-update-policy",
    ),
    "sr_powell_coordinate_chart_policy": (
        "--sr-powell-coordinate-chart-policy",
    ),
    "sr_escape_mode": ("--sr-escape-mode",),
    "sr_controller_ablation_contract": (
        "--sr-controller-ablation-contract",
    ),
    "phase2_gram_novelty_policy": ("--phase2-gram-novelty-policy",),
    "phase3_gram_novelty_policy": ("--phase3-gram-novelty-policy",),
    "phase3_novelty_ablation_mode": ("--phase3-novelty-ablation-mode",),
    "phase2_novelty_mode": ("--phase2-novelty-mode",),
    "phase2_selector_gain_mode": ("--phase2-selector-gain-mode",),
    "phase2_rho": ("--phase2-rho",),
    "phase1_score_mode": ("--phase1-score-mode",),
    "phase1_energy_model": ("--phase1-energy-model",),
    "phase2_curvature_policy": ("--phase2-curvature-policy",),
    "phase2_cheap_curvature_proxy_policy": (
        "--phase2-cheap-curvature-proxy-policy",
    ),
    "adapt_inner_optimizer": ("--adapt-inner-optimizer",),
    "adapt_maxiter": ("--adapt-maxiter",),
    "adapt_scipy_maxfev": ("--adapt-scipy-maxfev",),
    "adapt_state_backend": ("--adapt-state-backend",),
    "adapt_seed": ("--adapt-seed",),
    "adapt_reopt_policy": ("--adapt-reopt-policy",),
    "adapt_window_size": ("--adapt-window-size",),
    "adapt_window_topk": ("--adapt-window-topk",),
    "adapt_full_refit_every": ("--adapt-full-refit-every",),
    "adapt_final_full_refit": ("--adapt-final-full-refit",),
    "adapt_final_refit_maxiter": ("--adapt-final-refit-maxiter",),
    "adapt_insertion_mode": ("--adapt-insertion-mode",),
    "adapt_max_depth": ("--adapt-max-depth",),
    "adapt_allow_repeats": ("--adapt-allow-repeats", "--adapt-no-repeats"),
    "adapt_finite_angle_fallback": (
        "--adapt-finite-angle-fallback",
        "--adapt-no-finite-angle-fallback",
    ),
    "adapt_finite_angle": ("--adapt-finite-angle",),
    "adapt_finite_angle_min_improvement": (
        "--adapt-finite-angle-min-improvement",
    ),
    "adapt_disable_hh_seed": ("--adapt-disable-hh-seed",),
    "phase0_pilot_enabled": ("--phase0-pilot-enabled", "--phase0-no-pilot"),
    "phase1_shortlist_size": ("--phase1-shortlist-size",),
    "phase2_shortlist_size": ("--phase2-shortlist-size",),
    "phase2_shortlist_fraction": ("--phase2-shortlist-fraction",),
    "phase2_enable_batching": (
        "--phase2-enable-batching",
        "--phase2-no-batching",
    ),
    "phase3_enable_batching": (
        "--phase3-enable-batching",
        "--phase3-no-batching",
    ),
    "phase3_runtime_split_mode": ("--phase3-runtime-split-mode",),
    "allow_archival_phase3_runtime_split": (
        "--allow-archival-phase3-runtime-split",
    ),
    "phase3_runtime_split_selection_mode": (
        "--phase3-runtime-split-selection-mode",
    ),
    "phase3_runtime_split_max_subset_size": (
        "--phase3-runtime-split-max-subset-size",
    ),
    "phase3_runtime_split_subset_sizes": (
        "--phase3-runtime-split-subset-sizes",
    ),
    "phase3_runtime_split_child_set_symmetry_policy": (
        "--phase3-runtime-split-child-set-symmetry-policy",
    ),
    "phase3_runtime_split_child_padding_policy": (
        "--phase3-runtime-split-child-padding-policy",
    ),
    "adapt_child_pool_expansion_mode": ("--adapt-child-pool-expansion-mode",),
    "adapt_child_pool_expansion_symmetry_policy": (
        "--adapt-child-pool-expansion-symmetry-policy",
    ),
    "shared_pauli_pool_mode": ("--shared-pauli-pool-mode",),
    "shared_pauli_pool_symmetry_policy": (
        "--shared-pauli-pool-symmetry-policy",
    ),
    "shared_pauli_pool_max_subset_size": (
        "--shared-pauli-pool-max-subset-size",
    ),
    "shared_pauli_pool_subset_sizes": (
        "--shared-pauli-pool-subset-sizes",
    ),
    "phase1_prune_enabled": ("--phase1-prune-enabled", "--phase1-no-prune"),
    "phase1_prune_policy": ("--phase1-prune-policy",),
    "phase1_prune_mode": ("--phase1-prune-mode",),
    "phase1_prune_fraction": ("--phase1-prune-fraction",),
    "phase1_prune_min_candidates": ("--phase1-prune-min-candidates",),
    "phase1_prune_max_candidates": ("--phase1-prune-max-candidates",),
    "phase1_prune_max_regression": ("--phase1-prune-max-regression",),
    "phase1_prune_tolerance_mode": ("--phase1-prune-tolerance-mode",),
    "phase1_prune_tolerance_shot_coeff": (
        "--phase1-prune-tolerance-shot-coeff",
    ),
    "phase1_prune_tolerance_screen_coeff": (
        "--phase1-prune-tolerance-screen-coeff",
    ),
    "phase1_prune_tolerance_chem": ("--phase1-prune-tolerance-chem",),
    "phase1_prune_tolerance_rel_coeff": (
        "--phase1-prune-tolerance-rel-coeff",
    ),
    "phase1_prune_tolerance_target_energy": (
        "--phase1-prune-tolerance-target-energy",
    ),
    "phase1_prune_retained_gain_ratio": (
        "--phase1-prune-retained-gain-ratio",
    ),
    "phase1_prune_protect_steps": ("--phase1-prune-protect-steps",),
    "phase1_prune_stale_age": ("--phase1-prune-stale-age",),
    "phase1_prune_stagnation_threshold": (
        "--phase1-prune-stagnation-threshold",
    ),
    "phase1_prune_small_theta_abs": ("--phase1-prune-small-theta-abs",),
    "phase1_prune_small_theta_relative": (
        "--phase1-prune-small-theta-relative",
    ),
    "phase1_prune_cooldown_steps": ("--phase1-prune-cooldown-steps",),
    "phase1_prune_local_window_size": (
        "--phase1-prune-local-window-size",
    ),
    "phase1_prune_recovery_trust_radius": (
        "--phase1-prune-recovery-trust-radius",
    ),
    "phase1_prune_schur_nomination_route": (
        "--phase1-prune-schur-nomination-route",
    ),
    "phase1_prune_metric_schur_mu": (
        "--phase1-prune-metric-schur-mu",
    ),
    "phase1_prune_metric_schur_solve_mode": (
        "--phase1-prune-metric-schur-solve-mode",
    ),
    "phase1_prune_metric_schur_cost_weighting": (
        "--phase1-prune-metric-schur-cost-weighting",
    ),
    "phase1_prune_trust_update_policy": (
        "--phase1-prune-trust-update-policy",
    ),
    "phase1_prune_metric_mu_update_policy": (
        "--phase1-prune-metric-mu-update-policy",
    ),
    "phase1_prune_endpoint_overlap_policy": (
        "--phase1-prune-endpoint-overlap-policy",
    ),
    "phase1_prune_old_fraction": ("--phase1-prune-old-fraction",),
    "phase1_prune_checkpoint_period": (
        "--phase1-prune-checkpoint-period",
    ),
    "phase1_prune_live_min_depth": ("--phase1-prune-live-min-depth",),
    "phase1_prune_maturity_threshold": (
        "--phase1-prune-maturity-threshold",
    ),
    "phase1_prune_snr_threshold": ("--phase1-prune-snr-threshold",),
    "phase1_prune_prefilter_policy": ("--phase1-prune-prefilter-policy",),
    "phase1_prune_prefilter_json": ("--phase1-prune-prefilter-json",),
    "phase1_prune_risk_threshold": ("--phase1-prune-risk-threshold",),
    "phase1_prune_prefilter_max_candidates": (
        "--phase1-prune-prefilter-max-candidates",
    ),
    "phase3_selector_policy": ("--phase3-selector-policy",),
    "phase3_selector_geometry_mode": ("--phase3-selector-geometry-mode",),
    "phase3_geometry_window_size": ("--phase3-geometry-window-size",),
    "phase3_response_coordinate_scope": (
        "--phase3-response-coordinate-scope",
    ),
    "phase3_window_relaxation_mode": ("--phase3-window-relaxation-mode",),
    "phase3_backend_cost_mode": ("--phase3-backend-cost-mode",),
    "phase3_backend_name": ("--phase3-backend-name",),
    "phase3_backend_transpile_seed": ("--phase3-backend-transpile-seed",),
    "phase3_backend_optimization_level": (
        "--phase3-backend-optimization-level",
    ),
    "phase3_hardware_cost_normalization_mode": (
        "--phase3-hardware-cost-normalization-mode",
    ),
    "phase3_shadow_damping_policy": (
        "--phase3-shadow-damping-policy",
    ),
    "phase3_lifetime_cost_mode": ("--phase3-lifetime-cost-mode",),
    "phase3_enable_rescue": ("--phase3-enable-rescue", "--phase3-no-rescue"),
    "phase3_symmetry_mitigation_mode": (
        "--phase3-symmetry-mitigation-mode",
    ),
    "phase3_plateau_acquisition_mode": (
        "--phase3-plateau-acquisition-mode",
    ),
    "phase3_plateau_seed_probe_mode": ("--phase3-plateau-seed-probe-mode",),
    "phase3_shadow_legacy_geometry_mode": (
        "--phase3-shadow-legacy-geometry-mode",
    ),
    "phase3_shadow_legacy_max_depth": ("--phase3-shadow-legacy-max-depth",),
    "phase3_parent_collapse_debug_max_depth": (
        "--phase3-parent-collapse-debug-max-depth",
    ),
    "hardware_resolution_mode": ("--hardware-resolution-mode",),
    "gradient_hw_floor": ("--gradient-hw-floor",),
    "gradient_drift_floor": ("--gradient-drift-floor",),
    "adapt_accepted_refit_scope": ("--adapt-accepted-refit-scope",),
    "adapt_accepted_refit_coordinate_chart": (
        "--adapt-accepted-refit-coordinate-chart",
    ),
    "adapt_accepted_refit_base_chart_policy": (
        "--adapt-accepted-refit-base-chart-policy",
    ),
}


_DISALLOWED_BOOLEAN_OPTIONS = frozenset(
    {
        "--adapt-no-repeats",
        "--phase0-pilot-enabled",
        "--phase2-enable-batching",
        "--phase3-enable-batching",
        "--phase3-enable-rescue",
    }
)


def _json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _phase3_material_window_policy_payload() -> dict[str, Any]:
    """Freeze the versioned numerical window policy into a route contract."""

    policy = DEFAULT_PHASE3_MATERIAL_WINDOW_POLICY
    return {
        "policy_version": str(policy.policy_version),
        "gram_entry_threshold": float(policy.gram_entry_threshold),
        "hessian_entry_threshold": float(policy.hessian_entry_threshold),
        "gram_omitted_l2_tolerance": float(
            policy.gram_omitted_l2_tolerance
        ),
        "hessian_omitted_l2_tolerance": float(
            policy.hessian_omitted_l2_tolerance
        ),
        "gram_cross_block_tolerance": float(
            policy.gram_cross_block_tolerance
        ),
        "hessian_cross_block_tolerance": float(
            policy.hessian_cross_block_tolerance
        ),
        "epsilon": float(policy.epsilon),
    }


def canonical_sr_snake_v1_contract() -> dict[str, Any]:
    """Return a fresh serialization-safe copy of the canonical contract."""

    payload: dict[str, Any] = {
        "schema": SR_ROUTE_PROFILE_CONTRACT_SCHEMA,
        "route_family": "singleton_response_snake",
        "route_profile": SR_ROUTE_PROFILE_CANONICAL_V1,
        "execution_settings": dict(CANONICAL_SR_SNAKE_V1_EXECUTION_SETTINGS),
        "semantic_invariants": {
            "regime_physics_source": "per_regime_source_lock",
            "same_cutoff_reference_required": True,
            "full_meta_hva_policy": "included_no_filters_v1",
            "phase2_score_policy": "collective_span_novelty_multiplier_v1",
            "phase3_ordinary_novelty_multiplier": "unit_n3_v1",
            "all_energy_models_infeasible_policy": (
                "collective_span_novelty_over_cost_v1"
            ),
            "geometry_expansion_refit_policy": "full_coordinate_refit_v1",
            "geometry_expansion_radius_policy": (
                "realized_fs_displacement_on_descent_hold_on_no_descent_v1"
            ),
            "scalar_unwhitened_fallback_allowed": False,
            "admission_cardinality": 1,
            "admission_rollback_supported": False,
            "repeated_generator_identity_allowed": True,
            "route_a_funnel_active": False,
        },
        "historical_authority": {
            "historical_command_sha256": _HISTORICAL_COMMAND_SHA256,
            "historical_result_sha256": _HISTORICAL_RESULT_SHA256,
            "self_contained_replay_archive_sha256": (
                _SELF_CONTAINED_ARCHIVE_SHA256
            ),
            "weak_weak_absolute_error": 4.472864776339236e-7,
        },
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def canonical_sr_snake_v1_contract_sha256() -> str:
    return _json_sha256(canonical_sr_snake_v1_contract())


def canonical_sr_snake_v2_contract() -> dict[str, Any]:
    """Return the conventional full-accepted-refit SR-SNAKE contract."""

    payload: dict[str, Any] = {
        "schema": SR_ROUTE_PROFILE_CONTRACT_SCHEMA,
        "route_family": "singleton_response_snake",
        "route_profile": SR_ROUTE_PROFILE_CONVENTIONAL_V2,
        "execution_settings": dict(CANONICAL_SR_SNAKE_V2_EXECUTION_SETTINGS),
        "semantic_invariants": {
            "regime_physics_source": "per_regime_source_lock",
            "same_cutoff_reference_required": True,
            "full_meta_hva_policy": "included_no_filters_v1",
            "phase2_score_policy": "collective_span_novelty_multiplier_v1",
            "phase3_ordinary_novelty_multiplier": "unit_n3_v1",
            "all_energy_models_infeasible_policy": (
                "collective_span_novelty_over_cost_v1"
            ),
            "selector_coordinate_solve_scope": "phase3_only_v1",
            "accepted_refit_scope": "full_ansatz_v1",
            "accepted_refit_coordinate_chart": "supported_fs_whitened_fixed_v1",
            "accepted_refit_base_chart_policy": (
                "expanded_runtime_projected_logical_v1"
            ),
            "accepted_refit_chart_rebuild_policy": (
                "once_per_accepted_refit_invocation_v1"
            ),
            "prune_nomination_route": "hessian_coupling_v1",
            "prune_acceptance_authority": "measured_delete_and_refit_v1",
            "terminal_prune_active": True,
            "scalar_unwhitened_fallback_allowed": False,
            "admission_cardinality": 1,
            "admission_rollback_supported": False,
            "repeated_generator_identity_allowed": True,
            "route_a_funnel_active": False,
            "negative_curvature_escape_active": False,
        },
        "anchor_authority": {
            "source_lock_root": _CONVENTIONAL_V2_SOURCE_LOCK_ROOT,
            "source_manifest_path": (
                f"{_CONVENTIONAL_V2_SOURCE_LOCK_ROOT}/source_manifest.json"
            ),
            "source_manifest_sha256": _CONVENTIONAL_V2_SOURCE_MANIFEST_SHA256,
            "source_archive_path": (
                f"{_CONVENTIONAL_V2_SOURCE_LOCK_ROOT}/source_tree.tar.gz"
            ),
            "source_archive_sha256": _CONVENTIONAL_V2_SOURCE_ARCHIVE_SHA256,
            "weak_holstein_anchors": [
                dict(anchor) for anchor in _CONVENTIONAL_V2_WEAK_HOLSTEIN_ANCHORS
            ],
        },
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def canonical_sr_snake_v2_contract_sha256() -> str:
    return _json_sha256(canonical_sr_snake_v2_contract())


def canonical_sr_snake_v3_contract() -> dict[str, Any]:
    """Return the full-response conventional SR-SNAKE contract."""

    payload: dict[str, Any] = {
        "schema": SR_ROUTE_PROFILE_CONTRACT_SCHEMA,
        "route_family": "singleton_response_snake",
        "route_profile": SR_ROUTE_PROFILE_CONVENTIONAL_V3,
        "execution_settings": dict(CANONICAL_SR_SNAKE_V3_EXECUTION_SETTINGS),
        "semantic_invariants": {
            "regime_physics_source": "per_regime_source_lock",
            "same_cutoff_reference_required": True,
            "full_meta_hva_policy": "included_no_filters_v1",
            "phase2_score_policy": "collective_span_novelty_multiplier_v1",
            "phase3_ordinary_novelty_multiplier": "unit_n3_v1",
            "all_energy_models_infeasible_policy": (
                "collective_span_novelty_over_cost_v1"
            ),
            "selector_coordinate_solve_scope": "phase3_only_v1",
            "phase3_response_coordinate_scope": (
                PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1
            ),
            "phase3_response_pre_support_invariant": (
                "response_count_equals_active_logical_count_plus_one_v1"
            ),
            "phase3_response_refit_schedule_independent": True,
            "accepted_refit_scope": "full_ansatz_v1",
            "accepted_refit_coordinate_chart": "supported_fs_whitened_fixed_v1",
            "accepted_refit_base_chart_policy": (
                "expanded_runtime_projected_logical_v1"
            ),
            "accepted_refit_chart_rebuild_policy": (
                "once_per_accepted_refit_invocation_v1"
            ),
            "prune_nomination_route": "hessian_coupling_v1",
            "prune_acceptance_authority": "measured_delete_and_refit_v1",
            "terminal_prune_active": True,
            "scalar_unwhitened_fallback_allowed": False,
            "admission_cardinality": 1,
            "admission_rollback_supported": False,
            "repeated_generator_identity_allowed": True,
            "route_a_funnel_active": False,
            "negative_curvature_escape_active": False,
        },
        "lineage_authority": {
            "parent_route_profile": SR_ROUTE_PROFILE_CONVENTIONAL_V2,
            "parent_contract_sha256": canonical_sr_snake_v2_contract_sha256(),
            "scientific_result_anchor_claimed": False,
        },
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def canonical_sr_snake_v3_contract_sha256() -> str:
    return _json_sha256(canonical_sr_snake_v3_contract())


def canonical_sr_snake_v4_contract() -> dict[str, Any]:
    """Return the opt-in symmetric-cost/trusted-prune SR-SNAKE contract."""

    payload: dict[str, Any] = {
        "schema": SR_ROUTE_PROFILE_CONTRACT_SCHEMA,
        "route_family": "singleton_response_snake",
        "route_profile": SR_ROUTE_PROFILE_CANDIDATE_V4,
        "execution_settings": dict(CANONICAL_SR_SNAKE_V4_EXECUTION_SETTINGS),
        "semantic_invariants": {
            "regime_physics_source": "per_regime_source_lock",
            "same_cutoff_reference_required": True,
            "full_meta_hva_policy": "included_no_filters_v1",
            "hh_preseed_policy": "disabled_singleton_growth_from_reference_v1",
            "phase3_response_coordinate_scope": (
                PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1
            ),
            "phase3_response_pre_support_invariant": (
                "response_count_equals_active_logical_count_plus_one_v1"
            ),
            "phase_live_hysteresis_enabled": False,
            "phase_retirement_policy": "disabled_v1",
            "phase1_energy_model": (
                PHASE1_ENERGY_MODEL_FIRST_ORDER_FS_TRUST_V1
            ),
            "phase1_fs_metric_role": "trust_domain_only_v1",
            "phase2_curvature_policy": (
                PHASE2_CURVATURE_POLICY_MEASURED_REQUIRED_FAIL_CLOSED_V1
            ),
            "phase2_cheap_curvature_proxy_policy": (
                PHASE2_CHEAP_CURVATURE_PROXY_POLICY_OFF
            ),
            "phase2_curvature_failure_policy": "abort_run_v1",
            "ordinary_gram_novelty_multiplier_active": False,
            "all_energy_models_infeasible_policy": (
                "collective_span_novelty_over_symmetric_cost_v1"
            ),
            "hardware_cost_policy": "family_robust_symmetric_arctan_v1",
            "hardware_cost_application_scope": "phase1_phase2_phase3_v1",
            "accepted_refit_scope": "full_ansatz_v1",
            "accepted_refit_coordinate_chart": "supported_fs_whitened_fixed_v1",
            "accepted_refit_base_chart_policy": (
                "expanded_runtime_projected_logical_v1"
            ),
            "beam_shape": "effective_1x1_v1",
            "prune_execution_scope": "live_only_v1",
            "prune_response_scope": "full_active_logical_v1",
            "prune_trust_constraint": "complete_affine_deletion_fs_v1",
            "prune_acceptance_authority": "measured_delete_and_refit_v1",
            "prune_endpoint_overlap_measurement_active": False,
            "prune_radius_update_measurement_delta": 0,
            "phase3_shadow_damping_active": True,
            "phase3_shadow_damping_applied_mu": 0.0,
            "phase3_shadow_damping_measurement_delta": 0,
            "finite_angle_fallback_active": False,
            "terminal_prune_active": False,
            "terminal_full_refit_active": False,
            "admission_cardinality": 1,
            "admission_rollback_supported": False,
            "repeated_generator_identity_allowed": True,
            "route_a_funnel_active": False,
            "negative_curvature_escape_active": False,
        },
        "lineage_authority": {
            "parent_route_profile": SR_ROUTE_PROFILE_CONVENTIONAL_V3,
            "parent_contract_sha256": canonical_sr_snake_v3_contract_sha256(),
            "scientific_result_anchor_claimed": False,
        },
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def canonical_sr_snake_v4_contract_sha256() -> str:
    return _json_sha256(canonical_sr_snake_v4_contract())


def canonical_sr_snake_no_prune_symmetric_cost_v1_contract() -> dict[str, Any]:
    """Return the full-response symmetric-cost, no-prune SR contract."""

    payload: dict[str, Any] = {
        "schema": SR_ROUTE_PROFILE_CONTRACT_SCHEMA,
        "route_family": "singleton_response_snake",
        "route_profile": SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
        "execution_settings": dict(
            CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_V1_EXECUTION_SETTINGS
        ),
        "semantic_invariants": {
            "regime_physics_source": "per_regime_source_lock",
            "same_cutoff_reference_required": True,
            "controller_horizon_source": "per_regime_source_lock",
            "full_meta_hva_policy": "included_no_filters_v1",
            "hh_preseed_policy": "disabled_singleton_growth_from_reference_v1",
            "phase3_response_coordinate_scope": (
                PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1
            ),
            "phase3_response_pre_support_invariant": (
                "response_count_equals_active_logical_count_plus_one_v1"
            ),
            "selector_coordinate_solve_scope": "phase3_only_v1",
            "phase2_supported_whitening_active": False,
            "phase3_supported_whitening_active": True,
            "adaptive_trust_policy": "displacement_calibrated_unbounded_v2",
            "phase1_energy_model": (
                PHASE1_ENERGY_MODEL_FIRST_ORDER_FS_TRUST_V1
            ),
            "phase1_fs_metric_role": "trust_domain_only_v1",
            "phase2_curvature_policy": (
                PHASE2_CURVATURE_POLICY_MEASURED_REQUIRED_FAIL_CLOSED_V1
            ),
            "phase2_cheap_curvature_proxy_policy": (
                PHASE2_CHEAP_CURVATURE_PROXY_POLICY_OFF
            ),
            "phase2_curvature_failure_policy": "abort_run_v1",
            "ordinary_phase2_novelty_multiplier_active": False,
            "ordinary_phase3_novelty_multiplier_active": False,
            "all_energy_models_infeasible_novelty_fallback_active": True,
            "all_energy_models_infeasible_novelty_fallback_telemetry_required": (
                True
            ),
            "all_energy_models_infeasible_novelty_fallback_policy": (
                "collective_span_novelty_over_symmetric_cost_v1"
            ),
            "hardware_cost_policy": "family_robust_symmetric_arctan_v1",
            "hardware_cost_application_scope": (
                "phase1_phase2_phase3_and_infeasible_fallback_v1"
            ),
            "accepted_refit_scope": "full_ansatz_v1",
            "accepted_refit_coordinate_chart": "supported_fs_whitened_fixed_v1",
            "accepted_refit_base_chart_policy": (
                "expanded_runtime_projected_logical_v1"
            ),
            "beam_shape": "effective_1x1_v1",
            "phase_live_hysteresis_enabled": False,
            "phase_retirement_policy": "disabled_v1",
            "periodic_full_refit_active": False,
            "pruning_active": False,
            "phase3_shadow_damping_active": False,
            "finite_angle_fallback_active": False,
            "terminal_prune_active": False,
            "terminal_full_refit_active": False,
            "admission_cardinality": 1,
            "admission_rollback_supported": False,
            "repeated_generator_identity_allowed": True,
            "route_a_funnel_active": False,
            "negative_curvature_escape_active": False,
        },
        "lineage_authority": {
            "parent_route_profile": SR_ROUTE_PROFILE_CONVENTIONAL_V3,
            "parent_contract_sha256": canonical_sr_snake_v3_contract_sha256(),
            "scientific_result_anchor_claimed": False,
        },
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def canonical_sr_snake_no_prune_symmetric_cost_v1_contract_sha256() -> str:
    return _json_sha256(canonical_sr_snake_no_prune_symmetric_cost_v1_contract())


def canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_v1_contract() -> dict[
    str, Any
]:
    """Return the support-projected, non-whitened Phase-III ablation."""

    payload = canonical_sr_snake_no_prune_symmetric_cost_v1_contract()
    payload["route_profile"] = (
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_V1
    )
    payload["execution_settings"] = dict(
        CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_V1_EXECUTION_SETTINGS
    )
    payload["semantic_invariants"].update(
        {
            "selector_coordinate_solve_scope": "phase3_only_v1",
            "phase2_supported_whitening_active": False,
            "phase3_support_projection_active": True,
            "phase3_supported_whitening_active": False,
            "phase3_supported_metric_inverse_sqrt_active": False,
            "phase3_metric_ridge_active": False,
            "phase3_trust_solve": (
                "raw_supported_metric_generalized_kkt_v1"
            ),
            # The coordinate-sensitive accepted Powell refit remains exactly
            # the parent's full-ansatz supported-FS-whitened operation.
            "accepted_refit_coordinate_chart": (
                "supported_fs_whitened_fixed_v1"
            ),
        }
    )
    payload["lineage_authority"] = {
        "parent_route_profile": SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
        "parent_contract_sha256": (
            canonical_sr_snake_no_prune_symmetric_cost_v1_contract_sha256()
        ),
        "only_intended_parent_setting_change": {
            "historical_singleton_coordinate_solve_policy": (
                "supported_metric_projected_generalized_trust_v1"
            )
        },
        "scientific_result_anchor_claimed": False,
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_v1_contract_sha256() -> str:
    return _json_sha256(
        canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_v1_contract()
    )


def canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract() -> dict[
    str, Any
]:
    """Return projected Phase III with source-metric no-overlap calibration."""

    payload = canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_v1_contract()
    payload["route_profile"] = (
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1
    )
    payload["execution_settings"] = dict(
        CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1_EXECUTION_SETTINGS
    )
    payload["semantic_invariants"].update(
        {
            "adaptive_trust_policy": (
                "source_metric_inverse_sqrt_no_overlap_v1"
            ),
            "adaptive_trust_predicted_displacement": (
                "phase3_joint_step_source_supported_gram_norm_v1"
            ),
            "adaptive_trust_realized_displacement": (
                "post_refit_parameter_step_source_supported_gram_norm_v1"
            ),
            "adaptive_trust_radius_exponent": -0.5,
            "adaptive_trust_expansion_requires_boundary": True,
            "adaptive_trust_expansion_energy_veto": (
                "positive_realized_descent_v1"
            ),
            "endpoint_overlap_measurement_active": False,
            "endpoint_overlap_query_charge_required": 0,
            "geometry_expansion_without_coordinate_prediction": (
                "hold_radius_without_overlap_v1"
            ),
        }
    )
    payload["lineage_authority"] = {
        "parent_route_profile": (
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_V1
        ),
        "parent_contract_sha256": (
            canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_v1_contract_sha256()
        ),
        "only_intended_parent_setting_change": {
            "historical_singleton_trust_region_update_policy": (
                "source_metric_inverse_sqrt_no_overlap_v1"
            )
        },
        "scientific_result_anchor_claimed": False,
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract_sha256() -> str:
    return _json_sha256(
        canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract()
    )


def canonical_sr_snake_greedy_batch_projected_phase3_no_overlap_trust_v1_contract(
    *,
    maximum_size: int = 3,
    search_window_size: int | None = None,
) -> dict[str, Any]:
    """Return the request-specific greedy-batch child of active fd5ec.

    The public facade owns these two typed controls.  Historical batching
    booleans and the near-degenerate score shell remain inactive, so this
    contract can vary the ranked search window without reopening legacy
    Route-A/JR policy.
    """

    maximum = int(maximum_size)
    if isinstance(maximum_size, bool) or maximum != maximum_size:
        raise ValueError("maximum_size must be an integer.")
    if not 1 <= maximum <= 5:
        raise ValueError("maximum_size must lie in the supported range 1..5.")
    if search_window_size is None:
        search_window: int | None = None
    else:
        search_window = int(search_window_size)
        if (
            isinstance(search_window_size, bool)
            or search_window != search_window_size
            or search_window < 1
        ):
            raise ValueError(
                "search_window_size must be a positive integer or None."
            )

    payload = (
        canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract()
    )
    payload["route_family"] = "greedy_batch_response_snake"
    payload["route_profile"] = (
        SR_ROUTE_PROFILE_GREEDY_BATCH_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1
    )
    payload["semantic_invariants"].update(
        {
            "admission_policy": "cost_weighted_greedy_reduced_plane_v1",
            "admission_cardinality": "one_to_configured_maximum_v1",
            "greedy_batch_maximum_size": maximum,
            "greedy_batch_search_window_size": search_window,
            "greedy_batch_search_window_semantics": (
                "full_ranked_phase3_population_v1"
                if search_window is None
                else "ranked_phase3_prefix_cardinality_v1"
            ),
            "greedy_batch_search_population": "ranked_phase3_candidates_v1",
            "greedy_batch_near_degenerate_shell_active": False,
            "greedy_batch_singleton_fallback_preserves_route_identity": True,
            "greedy_batch_commit_order": "proposal_order_v1",
            "greedy_batch_joint_response_coordinate_scope": (
                "full_active_plus_ordered_batch_v1"
            ),
            "controller_round_semantics": (
                "one_selection_one_atomic_batch_one_full_refit_v1"
            ),
            "legacy_phase2_batching_active": False,
            "legacy_phase3_batching_active": False,
        }
    )
    payload["lineage_authority"] = {
        "parent_route_profile": (
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1
        ),
        "parent_contract_sha256": (
            canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract_sha256()
        ),
        "only_intended_parent_policy_changes": {
            "admission_policy": "cost_weighted_greedy_reduced_plane_v1",
            "maximum_size": maximum,
            "search_window_size": search_window,
        },
        "scientific_result_anchor_claimed": False,
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def canonical_sr_snake_greedy_batch_projected_phase3_no_overlap_trust_v1_contract_sha256(
    *,
    maximum_size: int = 3,
    search_window_size: int | None = None,
) -> str:
    return _json_sha256(
        canonical_sr_snake_greedy_batch_projected_phase3_no_overlap_trust_v1_contract(
            maximum_size=maximum_size,
            search_window_size=search_window_size,
        )
    )


class _CombinatorialSearchWindowOmitted:
    __slots__ = ()


_COMBINATORIAL_SEARCH_WINDOW_OMITTED = (
    _CombinatorialSearchWindowOmitted()
)


def canonical_sr_snake_combinatorial_batch_projected_phase3_no_overlap_trust_v1_contract(
    *,
    maximum_size: int = 3,
    search_window_size: (
        int | None | _CombinatorialSearchWindowOmitted
    ) = _COMBINATORIAL_SEARCH_WINDOW_OMITTED,
) -> dict[str, Any]:
    """Return the request-specific exhaustive-batch child of active fd5ec."""

    maximum = int(maximum_size)
    if isinstance(maximum_size, bool) or maximum != maximum_size:
        raise ValueError("maximum_size must be an integer.")
    if not 1 <= maximum <= 5:
        raise ValueError("maximum_size must lie in the supported range 1..5.")
    if search_window_size is _COMBINATORIAL_SEARCH_WINDOW_OMITTED:
        search_window: int | None = min(2 * maximum, 10)
    elif search_window_size is None:
        search_window: int | None = None
    else:
        search_window = int(search_window_size)
        if (
            isinstance(search_window_size, bool)
            or search_window != search_window_size
            or search_window < 1
        ):
            raise ValueError(
                "search_window_size must be a positive integer or None."
            )

    payload = (
        canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract()
    )
    payload["route_family"] = "combinatorial_batch_response_snake"
    payload["route_profile"] = (
        SR_ROUTE_PROFILE_COMBINATORIAL_BATCH_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1
    )
    payload["semantic_invariants"].update(
        {
            "admission_policy": (
                "cost_weighted_combinatorial_reduced_plane_v1"
            ),
            "admission_cardinality": "one_to_configured_maximum_v1",
            "combinatorial_batch_maximum_size": maximum,
            "combinatorial_batch_search_window_size": search_window,
            "combinatorial_batch_search_window_semantics": (
                "full_ranked_phase3_population_v1"
                if search_window is None
                else "ranked_phase3_prefix_cardinality_v1"
            ),
            "combinatorial_batch_default_window_policy": (
                "min_two_times_maximum_size_and_ten_v1"
            ),
            "combinatorial_batch_search_population": (
                "ranked_phase3_candidates_v1"
            ),
            "combinatorial_batch_internal_order": (
                "ranked_child_phase2_within_fixed_phase3_membership_v1"
            ),
            "combinatorial_batch_enumeration": (
                "generator_distinct_subsets_not_permutations_v1"
            ),
            "combinatorial_batch_record_semantics": (
                "fixed_generator_plus_insertion_position_v1"
            ),
            "combinatorial_batch_commit_order": (
                "phase2_ranked_proposal_order_within_phase3_prefix_v1"
            ),
            "combinatorial_batch_joint_score": (
                "coupled_reduced_plane_gain_over_one_plus_symmetric_cost_v1"
            ),
            "combinatorial_batch_near_degenerate_shell_active": False,
            "combinatorial_batch_singleton_fallback_preserves_route_identity": (
                True
            ),
            "combinatorial_batch_joint_response_coordinate_scope": (
                "full_active_plus_ordered_batch_v1"
            ),
            "controller_round_semantics": (
                "one_selection_one_atomic_batch_one_full_refit_v1"
            ),
            "legacy_phase2_batching_active": False,
            "legacy_phase3_batching_active": False,
        }
    )
    payload["lineage_authority"] = {
        "parent_route_profile": (
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1
        ),
        "parent_contract_sha256": (
            canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract_sha256()
        ),
        "only_intended_parent_policy_changes": {
            "admission_policy": (
                "cost_weighted_combinatorial_reduced_plane_v1"
            ),
            "maximum_size": maximum,
            "search_window_size": search_window,
            "search_window_semantics": (
                "full_ranked_phase3_population_v1"
                if search_window is None
                else "ranked_phase3_prefix_cardinality_v1"
            ),
        },
        "scientific_result_anchor_claimed": False,
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def canonical_sr_snake_combinatorial_batch_projected_phase3_no_overlap_trust_v1_contract_sha256(
    *,
    maximum_size: int = 3,
    search_window_size: (
        int | None | _CombinatorialSearchWindowOmitted
    ) = _COMBINATORIAL_SEARCH_WINDOW_OMITTED,
) -> str:
    return _json_sha256(
        canonical_sr_snake_combinatorial_batch_projected_phase3_no_overlap_trust_v1_contract(
            maximum_size=maximum_size,
            search_window_size=search_window_size,
        )
    )


def canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_commutation_reduced_insertion_v1_contract() -> dict[
    str, Any
]:
    """Return the current singleton route with commutation-reduced insertion."""

    payload = (
        canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract()
    )
    payload["route_profile"] = (
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_V1
    )
    payload["execution_settings"] = dict(
        CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_V1_EXECUTION_SETTINGS
    )
    payload["semantic_invariants"].update(
        {
            "diagnostic_position_ablation": True,
            "insertion_position_scope": (
                "full_logical_ansatz_commutation_classes_every_depth_v2"
            ),
            "insertion_equivalence_policy": (
                "termwise_cross_component_commutation_earliest_representative_v1"
            ),
        }
    )
    payload["lineage_authority"] = {
        "parent_route_profile": (
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1
        ),
        "parent_contract_sha256": (
            canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract_sha256()
        ),
        "only_intended_parent_setting_changes": {
            "adapt_insertion_mode": "full_commutation_reduced",
        },
        "scientific_result_anchor_claimed": False,
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_commutation_reduced_insertion_v1_contract_sha256() -> str:
    return _json_sha256(
        canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_commutation_reduced_insertion_v1_contract()
    )


def canonical_sr_snake_insertion_commutation_plateau_v1_contract() -> dict[
    str, Any
]:
    """Return the immediate plateau-triggered insertion experiment."""

    payload = (
        canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract()
    )
    payload["route_profile"] = SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V1
    payload["execution_settings"] = dict(
        CANONICAL_SR_SNAKE_INSERTION_COMMUTATION_PLATEAU_V1_EXECUTION_SETTINGS
    )
    payload["semantic_invariants"].update(
        {
            "experimental_insertion_policy": (
                "insertion_commutation_plateau_v1"
            ),
            "insertion_position_scope": (
                "append_only_or_immediate_plateau_full_logical_domain_v1"
            ),
            "insertion_equivalence_policy": (
                "termwise_cross_component_commutation_earliest_representative_v1"
            ),
            "plateau_trigger_source": (
                "immediately_preceding_marginal_over_prior_cumulative_"
                "accepted_post_full_refit_energy_decrease_v1"
            ),
            "plateau_cumulative_decrease_ratio_threshold": (
                INSERTION_COMMUTATION_PLATEAU_CUMULATIVE_DECREASE_RATIO_THRESHOLD
            ),
            "plateau_threshold_comparison": (
                "marginal_to_prior_cumulative_strictly_below_v1"
            ),
            "plateau_threshold_calibration_status": (
                INSERTION_COMMUTATION_PLATEAU_CALIBRATION_STATUS
            ),
            "plateau_patience": 1,
            "plateau_hysteresis_active": False,
            "online_exact_reference_used": False,
        }
    )
    payload["lineage_authority"] = {
        "parent_route_profile": (
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1
        ),
        "parent_contract_sha256": (
            canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract_sha256()
        ),
        "only_intended_parent_setting_changes": {
            "adapt_insertion_mode": "insertion_commutation_plateau_v1",
        },
        "scientific_result_anchor_claimed": False,
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def canonical_sr_snake_insertion_commutation_plateau_v1_contract_sha256() -> str:
    return _json_sha256(
        canonical_sr_snake_insertion_commutation_plateau_v1_contract()
    )


def canonical_sr_snake_insertion_commutation_plateau_v2_contract() -> dict[
    str, Any
]:
    """Return the prior-historical-mean plateau insertion experiment."""

    payload = (
        canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract()
    )
    payload["route_profile"] = SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V2
    payload["execution_settings"] = dict(
        CANONICAL_SR_SNAKE_INSERTION_COMMUTATION_PLATEAU_V2_EXECUTION_SETTINGS
    )
    payload["semantic_invariants"].update(
        {
            "experimental_insertion_policy": (
                "insertion_commutation_plateau_v2"
            ),
            "insertion_position_scope": (
                "append_only_or_immediate_plateau_full_logical_domain_v1"
            ),
            "insertion_equivalence_policy": (
                "termwise_cross_component_commutation_earliest_representative_v1"
            ),
            "plateau_trigger_source": (
                "immediately_preceding_marginal_over_prior_mean_"
                "accepted_post_full_refit_energy_decrease_v2"
            ),
            "plateau_prior_mean_decrease_ratio_threshold": (
                INSERTION_COMMUTATION_PLATEAU_PRIOR_MEAN_DECREASE_RATIO_THRESHOLD
            ),
            "plateau_threshold_comparison": (
                "marginal_to_prior_mean_strictly_below_v2"
            ),
            "plateau_threshold_calibration_status": (
                INSERTION_COMMUTATION_PLATEAU_PRIOR_MEAN_CALIBRATION_STATUS
            ),
            "plateau_patience": 1,
            "plateau_hysteresis_active": False,
            "online_exact_reference_used": False,
        }
    )
    payload["lineage_authority"] = {
        "parent_route_profile": (
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1
        ),
        "parent_contract_sha256": (
            canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract_sha256()
        ),
        "only_intended_parent_setting_changes": {
            "adapt_insertion_mode": "insertion_commutation_plateau_v2",
        },
        "scientific_result_anchor_claimed": False,
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def canonical_sr_snake_insertion_commutation_plateau_v2_contract_sha256() -> str:
    return _json_sha256(
        canonical_sr_snake_insertion_commutation_plateau_v2_contract()
    )


def canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_full_geometry_query_neutral_prune_v1_contract() -> dict[
    str, Any
]:
    """Return the query-neutral full-geometry trust-prune candidate."""

    payload = (
        canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract()
    )
    payload["route_profile"] = (
        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_V1
    )
    payload["execution_settings"] = dict(
        CANONICAL_SR_SNAKE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_V1_EXECUTION_SETTINGS
    )
    payload["semantic_invariants"].update(
        {
            "pruning_active": True,
            "prune_execution_scope": "live_before_first_hit_v1",
            "prune_nomination_count_per_round_max": 1,
            "prune_response_scope": "full_active_plus_singleton_v1",
            "prune_source_geometry_policy": (
                "reuse_selected_phase3_full_active_plus_singleton_gram_"
                "hessian_gradient_v1"
            ),
            "phase3_material_window_policy": "off",
            "prune_trust_constraint": "complete_affine_deletion_fs_v1",
            "prune_initial_fs_radius": 0.00390625,
            "prune_radius_independent_from_admission_radius": True,
            "prune_radius_update_policy": (
                "combined_transition_rejection_contraction_only_half_to_"
                "1e-8_floor_v1"
            ),
            "prune_modeled_energy_change_max": -2.0e-6,
            "prune_modeled_energy_change_threshold_basis": (
                "one_percent_of_paper_i_target_abs_delta_e_v1"
            ),
            "prune_energy_nonincrease_absolute_tolerance": 1.0e-12,
            "benchmark_target_abs_delta_e": 2.0e-4,
            "benchmark_stop_contract": (
                "stop_immediately_after_first_accepted_target_hit_v1"
            ),
            "benchmark_max_round_safety_cap": 50,
            "prune_acceptance_authority": (
                "one_ordinary_complete_accepted_ansatz_powell_refit_energy_"
                "nonincrease_guard_v1"
            ),
            "prune_combined_transition": (
                "singleton_admission_plus_optional_deletion_one_refit_v1"
            ),
            "prune_failed_transition_policy": (
                "restore_pre_round_accepted_state_without_second_refit_v1"
            ),
            "prune_failed_coordinate_hysteresis": (
                "cooldown_until_state_motion_or_modeled_loss_improvement_v1"
            ),
            "prune_verification_beam": "off",
            "prune_delete_refit_sibling": "off",
            "prune_source_geometry_query_delta": 0,
            "prune_derivative_query_delta": 0,
            "prune_metric_query_delta": 0,
            "prune_hessian_query_delta": 0,
            "prune_energy_query_delta": 0,
            "prune_endpoint_overlap_measurement_active": False,
            "prune_endpoint_overlap_query_delta": 0,
            "prune_rollback_classical_query_charge": 0,
            "prune_explicit_query_delta": 0,
            "accepted_refit_scope": "full_ansatz_v1",
            "accepted_refit_coordinate_chart": (
                "supported_fs_whitened_fixed_v1"
            ),
            "beam_shape": "effective_1x1_v1",
            "historical_admission_beam_active": False,
            "terminal_prune_active": False,
            "terminal_full_refit_active": False,
        }
    )
    changed_keys = (
        "phase1_prune_enabled",
        "phase1_prune_mode",
        "phase1_prune_max_candidates",
        "phase1_prune_local_window_size",
        "phase1_prune_recovery_trust_radius",
        "phase1_prune_schur_nomination_route",
        "phase1_prune_metric_schur_mu",
        "phase1_prune_metric_schur_solve_mode",
        "phase1_prune_metric_schur_cost_weighting",
        "phase1_prune_trust_update_policy",
        "phase1_prune_metric_mu_update_policy",
        "phase1_prune_endpoint_overlap_policy",
    )
    payload["lineage_authority"] = {
        "parent_route_profile": (
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1
        ),
        "parent_contract_sha256": (
            canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract_sha256()
        ),
        "only_intended_parent_setting_changes": {
            key: payload["execution_settings"][key] for key in changed_keys
        },
        "scientific_result_anchor_claimed": False,
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_full_geometry_query_neutral_prune_v1_contract_sha256() -> str:
    return _json_sha256(
        canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_full_geometry_query_neutral_prune_v1_contract()
    )


def canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_full_geometry_fs_prune_verify_v1_contract() -> dict[
    str, Any
]:
    """Return the measured recoverability-prune child of active fd5ec."""

    payload = (
        canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract()
    )
    payload["route_profile"] = (
        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_FS_PRUNE_VERIFY_V1
    )
    payload["execution_settings"] = dict(
        CANONICAL_SR_SNAKE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_FS_PRUNE_VERIFY_V1_EXECUTION_SETTINGS
    )
    payload["semantic_invariants"].update(
        {
            "pruning_active": True,
            "prune_execution_scope": "live_only_v1",
            "prune_stage_order": (
                "accepted_full_refit_then_nomination_then_measured_"
                "delete_refit_then_transition_closure_v1"
            ),
            "prune_nomination_count_per_round_max": 1,
            "prune_response_scope": "full_active_logical_post_refit_v1",
            "prune_source_geometry_policy": (
                "fresh_post_refit_full_logical_gram_hessian_gradient_v1"
            ),
            "prune_trust_constraint": "complete_affine_deletion_fs_v1",
            "prune_initial_fs_radius": 0.125,
            "prune_radius_update_policy": (
                "rejection_contraction_only_half_to_1e-8_floor_v1"
            ),
            "prune_acceptance_authority": "measured_delete_and_refit_v1",
            "prune_verification_beam": (
                "minimal_immutable_keep_vs_one_delete_refit_sibling_v1"
            ),
            "prune_keep_branch_mutation_policy": (
                "immutable_never_destructively_mutated_v1"
            ),
            "prune_rejected_trial_policy": (
                "discard_sibling_without_survivor_restore_v1"
            ),
            "prune_branch_specific_delete_refit_measurements_are_real_work": True,
            "prune_rejected_branch_measurements_in_all_work_s_alg": True,
            "prune_metric_damping_active": False,
            "prune_metric_damping_update_active": False,
            "prune_endpoint_overlap_measurement_active": False,
            "prune_radius_update_measurement_delta": 0,
            "beam_shape": "effective_1x1_v1",
            "historical_admission_beam_active": False,
            "terminal_prune_active": False,
            "terminal_full_refit_active": False,
        }
    )
    changed_keys = (
        "phase1_prune_enabled",
        "phase1_prune_mode",
        "phase1_prune_max_candidates",
        "phase1_prune_local_window_size",
        "phase1_prune_recovery_trust_radius",
        "phase1_prune_schur_nomination_route",
        "phase1_prune_metric_schur_mu",
        "phase1_prune_metric_schur_solve_mode",
        "phase1_prune_metric_schur_cost_weighting",
        "phase1_prune_trust_update_policy",
        "phase1_prune_metric_mu_update_policy",
        "phase1_prune_endpoint_overlap_policy",
    )
    payload["lineage_authority"] = {
        "parent_route_profile": (
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1
        ),
        "parent_contract_sha256": (
            canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract_sha256()
        ),
        "only_intended_parent_setting_changes": {
            key: payload["execution_settings"][key] for key in changed_keys
        },
        "scientific_result_anchor_claimed": False,
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_full_geometry_fs_prune_verify_v1_contract_sha256() -> str:
    return _json_sha256(
        canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_full_geometry_fs_prune_verify_v1_contract()
    )


def canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_material_window_v1_contract() -> dict[
    str, Any
]:
    """Return the independent material-coupling Phase-III window study."""

    payload = (
        canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract()
    )
    payload["route_profile"] = (
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_V1
    )
    payload["execution_settings"] = dict(
        CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_V1_EXECUTION_SETTINGS
    )
    payload["semantic_invariants"].update(
        {
            "phase3_response_coordinate_scope": (
                PHASE3_RESPONSE_COORDINATE_SCOPE_CANDIDATE_MATERIAL_COUPLING_WINDOW_V1
            ),
            "phase3_response_pre_support_invariant": (
                "response_count_equals_retained_active_count_plus_one_v1"
            ),
            "phase3_material_window_policy": (
                _phase3_material_window_policy_payload()
            ),
            "phase3_material_window_threshold_source": (
                "validated_full_geometry_no_overlap_six_regime_telemetry_v1"
            ),
            "phase3_material_window_candidate_screen_domain": (
                "all_active_candidate_gram_and_hessian_couplings_v1"
            ),
            "phase3_material_window_source_anchor": (
                "full_active_plus_singleton_response_v1"
            ),
            "phase3_material_window_selection_rule": (
                "union_of_material_gram_and_hessian_coordinates_v1"
            ),
            "phase3_material_window_closure_diagnostics": (
                "measured_retained_to_omitted_gram_and_hessian_blocks_v1"
            ),
            "phase3_material_window_measurement_accounting": (
                "strict_identity_deduplicated_per_round_union_across_all_"
                "evaluated_candidates_v1"
            ),
            "phase3_material_window_per_candidate_summed_estimate_allowed": (
                False
            ),
            "s_alg_component_order": [
                "N_H_outer",
                "N_H_refit",
                "N_grad",
                "N_metric",
            ],
            "s_alg_aggregation": (
                "strict_identity_deduplicated_component_sum_v1"
            ),
            "phase3_material_window_full_refresh_triggers": [
                "supported_nullity_drift_v1",
                "gram_omitted_block_closure_failure_v1",
                "hessian_omitted_block_closure_failure_v1",
                "nonfinite_window_diagnostic_v1",
            ],
            "phase3_material_window_support_change_policy": (
                "full_geometry_refresh_on_unexpected_supported_nullity_drift_v1"
            ),
            "phase3_material_window_independent_from_powell_refit_window": True,
            "phase3_material_window_fallback_scope": (
                "full_active_plus_singleton_geometry_v1"
            ),
            "accepted_refit_scope": "full_ansatz_v1",
            "accepted_refit_coordinate_chart": (
                "supported_fs_whitened_fixed_v1"
            ),
            "beam_shape": "effective_1x1_v1",
            "historical_admission_beam_active": False,
            "pruning_active": False,
        }
    )
    payload["lineage_authority"] = {
        "parent_route_profile": (
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1
        ),
        "parent_contract_sha256": (
            canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract_sha256()
        ),
        "only_intended_parent_setting_change": {
            "phase3_response_coordinate_scope": (
                PHASE3_RESPONSE_COORDINATE_SCOPE_CANDIDATE_MATERIAL_COUPLING_WINDOW_V1
            )
        },
        "scientific_result_anchor_claimed": False,
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_material_window_v1_contract_sha256() -> str:
    return _json_sha256(
        canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_material_window_v1_contract()
    )


def canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_material_window_fs_prune_verify_v1_contract() -> dict[
    str, Any
]:
    """Return material-window SR with immutable keep/prune verification."""

    payload = (
        canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_material_window_v1_contract()
    )
    payload["route_profile"] = (
        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1
    )
    payload["execution_settings"] = dict(
        CANONICAL_SR_SNAKE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1_EXECUTION_SETTINGS
    )
    payload["semantic_invariants"].update(
        {
            "pruning_active": True,
            "prune_execution_scope": "live_only_v1",
            "prune_nomination_count_per_round_max": 1,
            "prune_response_scope": "full_active_logical_v1",
            "prune_source_geometry_policy": (
                "reuse_measured_source_active_gram_hessian_blocks_v1"
            ),
            "prune_trust_constraint": "complete_affine_deletion_fs_v1",
            "prune_initial_fs_radius": 0.125,
            "prune_radius_update_policy": (
                "rejection_contraction_only_half_to_1e-8_floor_v1"
            ),
            "prune_acceptance_authority": "measured_delete_and_refit_v1",
            "prune_verification_beam": (
                "minimal_immutable_keep_vs_one_delete_refit_sibling_v1"
            ),
            "prune_verification_parallelism": (
                "logical_parallel_keep_and_delete_refit_siblings_v1"
            ),
            "prune_keep_branch_mutation_policy": (
                "immutable_never_destructively_mutated_v1"
            ),
            "prune_rejected_trial_policy": (
                "discard_sibling_without_survivor_restore_v1"
            ),
            "prune_rollback_classical_query_charge": 0,
            "prune_branch_specific_delete_refit_measurements_are_real_work": True,
            "prune_rejected_branch_measurements_in_all_work_s_alg": True,
            "prune_s_alg_views_required": [
                "all_work_v1",
                "winning_lineage_v1",
                "rejected_or_discarded_prune_branch_v1",
                "shared_source_state_v1",
            ],
            "prune_s_alg_identity_deduplication": (
                "strict_estimator_primitive_identity_v1"
            ),
            "prune_metric_damping_active": False,
            "prune_metric_damping_update_active": False,
            "prune_endpoint_overlap_measurement_active": False,
            "prune_radius_update_measurement_delta": 0,
            "beam_shape": "effective_1x1_v1",
            "historical_admission_beam_active": False,
            "terminal_prune_active": False,
            "terminal_full_refit_active": False,
        }
    )
    changed_keys = (
        "phase1_prune_enabled",
        "phase1_prune_mode",
        "phase1_prune_max_candidates",
        "phase1_prune_local_window_size",
        "phase1_prune_recovery_trust_radius",
        "phase1_prune_schur_nomination_route",
        "phase1_prune_metric_schur_mu",
        "phase1_prune_metric_schur_solve_mode",
        "phase1_prune_metric_schur_cost_weighting",
        "phase1_prune_trust_update_policy",
        "phase1_prune_metric_mu_update_policy",
        "phase1_prune_endpoint_overlap_policy",
    )
    payload["lineage_authority"] = {
        "parent_route_profile": (
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_V1
        ),
        "parent_contract_sha256": (
            canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_material_window_v1_contract_sha256()
        ),
        "only_intended_parent_setting_changes": {
            key: payload["execution_settings"][key] for key in changed_keys
        },
        "scientific_result_anchor_claimed": False,
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_material_window_fs_prune_verify_v1_contract_sha256() -> str:
    return _json_sha256(
        canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_material_window_fs_prune_verify_v1_contract()
    )


def canonical_sr_snake_guarded_singleton_pool_v1_contract() -> dict[str, Any]:
    """Return the SR response route over a guarded raw-singleton pool."""

    payload = canonical_sr_snake_no_prune_symmetric_cost_v1_contract()
    payload["route_profile"] = SR_ROUTE_PROFILE_GUARDED_SINGLETON_POOL_V1
    payload["execution_settings"] = dict(
        CANONICAL_SR_SNAKE_GUARDED_SINGLETON_POOL_V1_EXECUTION_SETTINGS
    )
    payload["semantic_invariants"].update(
        {
            "diagnostic_pool_ablation": True,
            "phase_live_hysteresis_enabled": False,
            "phase_retirement_policy": "disabled_v1",
            "candidate_parent_pool": "unfiltered_full_meta_hva_included_v1",
            "macro_parent_role": "decomposition_source_only_v1",
            "macro_candidates_enter_phase1": False,
            "candidate_representation": "raw_single_pauli_word_v1",
            "candidate_subset_cardinality": 1,
            "candidate_pool_projection_active": False,
            "candidate_direction_normalization": (
                "projective_unit_direction_for_global_deduplication_v1"
            ),
            "fixed_sector_guard": "hard_fail_closed_v1",
            "binary_padding_policy": "legal_codeword_hard_filter_v1",
            "binary_padding_cutoff_source": "per_regime_nph3_or_nph7_v1",
            "global_child_identity_policy": "global_pauli_word_v1",
            "duplicate_parent_lineage_preserved": True,
            "pool_exposure_scope": "same_atomic_pool_phase1_phase2_phase3_v1",
            "shortlist_population_policy": "single_global_population_v1",
            "physical_operator_lanes_active": False,
            "algebraic_lanes_active": False,
            "phase3_runtime_split_active": False,
        }
    )
    payload["lineage_authority"] = {
        "parent_route_profile": SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
        "parent_contract_sha256": (
            canonical_sr_snake_no_prune_symmetric_cost_v1_contract_sha256()
        ),
        "only_intended_parent_setting_changes": {
            "static_lane_route": "algebraic",
            "physical_lane_shortlist_aggressiveness": 1,
            "phase3_selector_policy": "hardware_resolvable_v1",
            "phase_live_hysteresis_enabled": False,
            "phase3_runtime_split_mode": "off",
            "allow_archival_phase3_runtime_split": False,
            "phase3_runtime_split_child_padding_policy": (
                "unchecked_diagnostic_v1"
            ),
            "shared_pauli_pool_mode": "guarded_singleton_children_only_v1",
            "shared_pauli_pool_symmetry_policy": "hard_guard",
            "shared_pauli_pool_max_subset_size": 1,
            "shared_pauli_pool_subset_sizes": "1",
        },
        "scientific_result_anchor_claimed": False,
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def canonical_sr_snake_guarded_singleton_pool_v1_contract_sha256() -> str:
    return _json_sha256(canonical_sr_snake_guarded_singleton_pool_v1_contract())


def canonical_sr_snake_macro_only_physical_lanes_v1_contract() -> dict[str, Any]:
    """Return the SR response route over intact full-meta parent generators."""

    payload = canonical_sr_snake_no_prune_symmetric_cost_v1_contract()
    payload["route_profile"] = SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_V1
    payload["execution_settings"] = dict(
        CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_V1_EXECUTION_SETTINGS
    )
    payload["semantic_invariants"].update(
        {
            "diagnostic_pool_ablation": True,
            "phase_live_hysteresis_enabled": False,
            "phase_retirement_policy": "disabled_v1",
            "candidate_parent_pool": "unfiltered_full_meta_hva_included_v1",
            "candidate_representation": "intact_logical_parent_generator_v1",
            "parent_generators_enter_phase1": True,
            "generated_pauli_children_active": False,
            "phase3_runtime_split_active": False,
            "shared_child_pool_expansion_active": False,
            "candidate_pool_projection_active": False,
            "pool_exposure_scope": "same_parent_pool_phase1_phase2_phase3_v1",
            "physical_operator_lanes_active": True,
            "physical_lane_shortlist_aggressiveness": 3,
        }
    )
    payload["lineage_authority"] = {
        "parent_route_profile": SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
        "parent_contract_sha256": (
            canonical_sr_snake_no_prune_symmetric_cost_v1_contract_sha256()
        ),
        "only_intended_parent_setting_changes": {
            "phase_live_hysteresis_enabled": False,
            "phase3_runtime_split_mode": "off",
            "allow_archival_phase3_runtime_split": False,
            "phase3_runtime_split_child_padding_policy": (
                "unchecked_diagnostic_v1"
            ),
            "adapt_child_pool_expansion_mode": "off",
            "shared_pauli_pool_mode": "off",
            "shared_pauli_pool_symmetry_policy": "off",
        },
        "scientific_result_anchor_claimed": False,
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def canonical_sr_snake_macro_only_physical_lanes_v1_contract_sha256() -> str:
    return _json_sha256(
        canonical_sr_snake_macro_only_physical_lanes_v1_contract()
    )


def canonical_sr_snake_macro_only_physical_lanes_commutation_reduced_insertion_diagnostic_v2_contract() -> dict[str, Any]:
    """Return the macro-only route with commutation-reduced full insertion."""

    payload = canonical_sr_snake_macro_only_physical_lanes_v1_contract()
    payload["route_profile"] = (
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_COMMUTATION_REDUCED_INSERTION_DIAGNOSTIC_V2
    )
    payload["execution_settings"] = dict(
        CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_COMMUTATION_REDUCED_INSERTION_DIAGNOSTIC_V2_EXECUTION_SETTINGS
    )
    payload["semantic_invariants"].update(
        {
            "diagnostic_position_ablation": True,
            "insertion_position_scope": (
                "full_logical_ansatz_commutation_classes_every_depth_v2"
            ),
            "insertion_equivalence_policy": (
                "termwise_cross_component_commutation_earliest_representative_v1"
            ),
        }
    )
    payload["lineage_authority"] = {
        "parent_route_profile": SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_V1,
        "parent_contract_sha256": (
            canonical_sr_snake_macro_only_physical_lanes_v1_contract_sha256()
        ),
        "only_intended_parent_setting_changes": {
            "adapt_insertion_mode": "full_commutation_reduced",
        },
        "scientific_result_anchor_claimed": False,
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def canonical_sr_snake_macro_only_physical_lanes_commutation_reduced_insertion_diagnostic_v2_contract_sha256() -> str:
    return _json_sha256(
        canonical_sr_snake_macro_only_physical_lanes_commutation_reduced_insertion_diagnostic_v2_contract()
    )


def canonical_sr_snake_macro_only_always_insertion_fs_prune_beam3x2_v1_contract(
    *,
    nomination_route: str = "full_logical_fs_trust_delete_refit_v1",
) -> dict[str, Any]:
    """Return always-insertion macro with 3x2 beam and live FS pruning.

    Parent is the always-insertion macro contract. The intended changes are the
    beam and pruning axes only, taken verbatim from the sealed
    ``fs_prune_nodamping_beam3x2`` profile. That profile is append-only; the
    append-only insertion flip is deliberately NOT inherited here so the
    insertion policy stays comparable with the running lanes ablation.
    """

    payload = (
        canonical_sr_snake_macro_only_physical_lanes_commutation_reduced_insertion_diagnostic_v2_contract()
    )
    donor = (
        canonical_sr_snake_macro_only_physical_lanes_fs_prune_beam_v1_contract()
    )
    suffix = (
        "metric_regularized"
        if str(nomination_route) == "metric_regularized_v1"
        else "fs_trust"
    )
    payload["route_profile"] = (
        f"{SR_ROUTE_PROFILE_MACRO_ONLY_ALWAYS_INSERTION_FS_PRUNE_BEAM3X2_V1}"
        f"__{suffix}_nomination"
    )
    execution = dict(payload["execution_settings"])
    donor_execution = donor["execution_settings"]
    changed: dict[str, Any] = {}
    for key in sorted(donor_execution):
        if key == "adapt_insertion_mode":
            continue
        if execution.get(key) != donor_execution[key]:
            execution[key] = donor_execution[key]
            changed[key] = donor_execution[key]
    execution["phase1_prune_schur_nomination_route"] = str(nomination_route)
    changed["phase1_prune_schur_nomination_route"] = str(nomination_route)
    payload["execution_settings"] = execution

    invariants = dict(payload["semantic_invariants"])
    donor_invariants = donor["semantic_invariants"]
    for key in sorted(donor_invariants):
        if "insertion" in key:
            continue
        if invariants.get(key) != donor_invariants[key]:
            invariants[key] = donor_invariants[key]
    payload["semantic_invariants"] = invariants

    payload["lineage_authority"] = {
        "parent_route_profile": (
            SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_COMMUTATION_REDUCED_INSERTION_DIAGNOSTIC_V2
        ),
        "parent_contract_sha256": (
            canonical_sr_snake_macro_only_physical_lanes_commutation_reduced_insertion_diagnostic_v2_contract_sha256()
        ),
        "beam_prune_donor_route_profile": (
            SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_V1
        ),
        "beam_prune_donor_contract_sha256": (
            canonical_sr_snake_macro_only_physical_lanes_fs_prune_beam_v1_contract_sha256()
        ),
        "only_intended_parent_setting_changes": changed,
        "insertion_policy_deliberately_not_inherited": "append_only",
        "scientific_result_anchor_claimed": False,
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def canonical_sr_snake_macro_only_always_insertion_fs_prune_beam3x2_v1_contract_sha256(
    *,
    nomination_route: str = "full_logical_fs_trust_delete_refit_v1",
) -> str:
    return _json_sha256(
        canonical_sr_snake_macro_only_always_insertion_fs_prune_beam3x2_v1_contract(
            nomination_route=nomination_route
        )
    )



def canonical_sr_snake_macro_only_no_lanes_commutation_reduced_insertion_v1_contract() -> dict[str, Any]:
    """Return the lanes-off arm of the macro always-insertion ablation.

    The single intended change against the parent always-insertion macro
    contract is ``physical_operator_lanes_active = False``: the Phase-I
    shortlist becomes one global population instead of a lane-structured one.
    Representation, pool, insertion policy, gradients, weighting, optimizer,
    seeds, and shortlist sizes are inherited unchanged so the pair isolates the
    lane axis alone.
    """

    payload = (
        canonical_sr_snake_macro_only_physical_lanes_commutation_reduced_insertion_diagnostic_v2_contract()
    )
    payload["route_profile"] = (
        SR_ROUTE_PROFILE_MACRO_ONLY_NO_LANES_COMMUTATION_REDUCED_INSERTION_V1
    )
    execution = dict(payload["execution_settings"])
    from pipelines.static_adapt.lane_routes import (
        STATIC_LANE_ROUTE_GLOBAL_SINGLE_POPULATION,
    )

    execution["static_lane_route"] = STATIC_LANE_ROUTE_GLOBAL_SINGLE_POPULATION
    execution.pop("physical_lane_shortlist_aggressiveness", None)
    payload["execution_settings"] = execution
    payload["semantic_invariants"].update(
        {
            "physical_operator_lanes_active": False,
            "shortlist_population_policy": "single_global_population_v1",
        }
    )
    payload["semantic_invariants"].pop(
        "physical_lane_shortlist_aggressiveness", None
    )
    payload["lineage_authority"] = {
        "parent_route_profile": (
            SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_COMMUTATION_REDUCED_INSERTION_DIAGNOSTIC_V2
        ),
        "parent_contract_sha256": (
            canonical_sr_snake_macro_only_physical_lanes_commutation_reduced_insertion_diagnostic_v2_contract_sha256()
        ),
        "only_intended_parent_setting_changes": {
            "physical_operator_lanes_active": False,
        },
        "scientific_result_anchor_claimed": False,
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def canonical_sr_snake_macro_only_no_lanes_commutation_reduced_insertion_v1_contract_sha256() -> str:
    return _json_sha256(
        canonical_sr_snake_macro_only_no_lanes_commutation_reduced_insertion_v1_contract()
    )


def canonical_sr_snake_macro_only_physical_lanes_insertion_commutation_plateau_v1_contract() -> dict[str, Any]:
    """Return the page-2 macro route with plateau-triggered insertion."""

    payload = canonical_sr_snake_macro_only_physical_lanes_v1_contract()
    payload["route_profile"] = (
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V1
    )
    payload["execution_settings"] = dict(
        CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V1_EXECUTION_SETTINGS
    )
    payload["semantic_invariants"].update(
        {
            "experimental_insertion_policy": "insertion_commutation_plateau_v1",
            "insertion_position_scope": (
                "append_only_or_immediate_plateau_full_logical_domain_v1"
            ),
            "insertion_equivalence_policy": (
                "termwise_cross_component_commutation_earliest_representative_v1"
            ),
            "plateau_trigger_source": (
                "immediately_preceding_marginal_over_prior_cumulative_"
                "accepted_post_full_refit_energy_decrease_v1"
            ),
            "plateau_cumulative_decrease_ratio_threshold": (
                INSERTION_COMMUTATION_PLATEAU_CUMULATIVE_DECREASE_RATIO_THRESHOLD
            ),
            "plateau_threshold_comparison": (
                "marginal_to_prior_cumulative_strictly_below_v1"
            ),
            "plateau_threshold_calibration_status": (
                INSERTION_COMMUTATION_PLATEAU_CALIBRATION_STATUS
            ),
            "plateau_patience": 1,
            "plateau_hysteresis_active": False,
            "online_exact_reference_used": False,
        }
    )
    payload["lineage_authority"] = {
        "parent_route_profile": SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_V1,
        "parent_contract_sha256": (
            canonical_sr_snake_macro_only_physical_lanes_v1_contract_sha256()
        ),
        "only_intended_parent_setting_changes": {
            "adapt_insertion_mode": "insertion_commutation_plateau_v1",
        },
        "scientific_result_anchor_claimed": False,
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def canonical_sr_snake_macro_only_physical_lanes_insertion_commutation_plateau_v1_contract_sha256() -> str:
    return _json_sha256(
        canonical_sr_snake_macro_only_physical_lanes_insertion_commutation_plateau_v1_contract()
    )


def canonical_sr_snake_macro_only_physical_lanes_insertion_commutation_plateau_v2_contract() -> dict[str, Any]:
    """Return the macro route with the prior-historical-mean trigger."""

    payload = canonical_sr_snake_macro_only_physical_lanes_v1_contract()
    payload["route_profile"] = (
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V2
    )
    payload["execution_settings"] = dict(
        CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V2_EXECUTION_SETTINGS
    )
    payload["semantic_invariants"].update(
        {
            "experimental_insertion_policy": "insertion_commutation_plateau_v2",
            "insertion_position_scope": (
                "append_only_or_immediate_plateau_full_logical_domain_v1"
            ),
            "insertion_equivalence_policy": (
                "termwise_cross_component_commutation_earliest_representative_v1"
            ),
            "plateau_trigger_source": (
                "immediately_preceding_marginal_over_prior_mean_"
                "accepted_post_full_refit_energy_decrease_v2"
            ),
            "plateau_prior_mean_decrease_ratio_threshold": (
                INSERTION_COMMUTATION_PLATEAU_PRIOR_MEAN_DECREASE_RATIO_THRESHOLD
            ),
            "plateau_threshold_comparison": (
                "marginal_to_prior_mean_strictly_below_v2"
            ),
            "plateau_threshold_calibration_status": (
                INSERTION_COMMUTATION_PLATEAU_PRIOR_MEAN_CALIBRATION_STATUS
            ),
            "plateau_patience": 1,
            "plateau_hysteresis_active": False,
            "online_exact_reference_used": False,
        }
    )
    payload["lineage_authority"] = {
        "parent_route_profile": SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_V1,
        "parent_contract_sha256": (
            canonical_sr_snake_macro_only_physical_lanes_v1_contract_sha256()
        ),
        "only_intended_parent_setting_changes": {
            "adapt_insertion_mode": "insertion_commutation_plateau_v2",
        },
        "scientific_result_anchor_claimed": False,
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def canonical_sr_snake_macro_only_physical_lanes_insertion_commutation_plateau_v2_contract_sha256() -> str:
    return _json_sha256(
        canonical_sr_snake_macro_only_physical_lanes_insertion_commutation_plateau_v2_contract()
    )


def canonical_sr_snake_macro_only_physical_lanes_one_sided_cost_v1_contract() -> dict[str, Any]:
    """Return the macro-only route with the historical one-sided cost term."""

    payload = canonical_sr_snake_macro_only_physical_lanes_v1_contract()
    payload["route_profile"] = (
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_ONE_SIDED_COST_V1
    )
    payload["execution_settings"] = dict(
        CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_ONE_SIDED_COST_V1_EXECUTION_SETTINGS
    )
    payload["semantic_invariants"].update(
        {
            "hardware_cost_policy": "family_robust_v1",
            "cost_normalization_ablation": "one_sided_positive_excess_v1",
        }
    )
    payload["lineage_authority"] = {
        "parent_route_profile": SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_V1,
        "parent_contract_sha256": (
            canonical_sr_snake_macro_only_physical_lanes_v1_contract_sha256()
        ),
        "only_intended_parent_setting_changes": {
            "phase3_hardware_cost_normalization_mode": "family_robust_v1"
        },
        "scientific_result_anchor_claimed": False,
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def canonical_sr_snake_macro_only_physical_lanes_one_sided_cost_v1_contract_sha256() -> str:
    return _json_sha256(
        canonical_sr_snake_macro_only_physical_lanes_one_sided_cost_v1_contract()
    )


def canonical_sr_snake_macro_only_physical_lanes_fs_prune_beam_v1_contract() -> dict[str, Any]:
    """Return macro-only SR with historical beam and FS-trust pruning."""

    payload = canonical_sr_snake_macro_only_physical_lanes_v1_contract()
    payload["route_profile"] = (
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_V1
    )
    payload["execution_settings"] = dict(
        CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_V1_EXECUTION_SETTINGS
    )
    payload["semantic_invariants"].update(
        {
            "pruning_active": True,
            "prune_execution_scope": "live_only_v1",
            "prune_response_scope": "full_active_logical_v1",
            "prune_trust_constraint": "complete_affine_deletion_fs_v1",
            "prune_initial_fs_radius": 0.125,
            "prune_radius_update_policy": (
                "rejection_contraction_only_half_to_1e-8_floor_v1"
            ),
            "prune_acceptance_authority": "measured_delete_and_refit_v1",
            "prune_metric_damping_active": False,
            "prune_metric_damping_update_active": False,
            "prune_endpoint_overlap_measurement_active": False,
            "prune_radius_update_measurement_delta": 0,
            "terminal_prune_active": False,
            "terminal_full_refit_active": False,
            "beam_shape": "historical_3x2_v1",
            "beam_live_branch_cap": 3,
            "beam_children_per_parent": 2,
            "beam_expanded_child_cap_per_round": 6,
            "beam_terminated_keep": 3,
            "beam_terminal_archive_mode": "legacy",
            "beam_parent_stop_terminal_also_materialized": True,
            "beam_structural_mode": "stop_or_single_admission",
            "beam_terminal_archive_accumulation": "cumulative_across_rounds",
            "beam_terminal_archive_cap": 3,
            "beam_lambda": 0.005,
        }
    )
    payload["lineage_authority"] = {
        "parent_route_profile": SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_V1,
        "parent_contract_sha256": (
            canonical_sr_snake_macro_only_physical_lanes_v1_contract_sha256()
        ),
        "only_intended_parent_setting_changes": {
            key: CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_V1_EXECUTION_SETTINGS[
                key
            ]
            for key in (
                "phase1_prune_enabled",
                "phase1_prune_policy",
                "phase1_prune_mode",
                "phase1_prune_max_candidates",
                "phase1_prune_local_window_size",
                "phase1_prune_recovery_trust_radius",
                "phase1_prune_schur_nomination_route",
                "phase1_prune_metric_schur_mu",
                "phase1_prune_metric_schur_solve_mode",
                "phase1_prune_metric_schur_cost_weighting",
                "phase1_prune_trust_update_policy",
                "phase1_prune_metric_mu_update_policy",
                "phase1_prune_endpoint_overlap_policy",
                "adapt_beam_live_branches",
                "adapt_beam_children_per_parent",
                "adapt_beam_terminated_keep",
                "adapt_beam_terminal_archive_mode",
                "adapt_beam_lambda",
            )
        },
        "scientific_result_anchor_claimed": False,
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def canonical_sr_snake_macro_only_physical_lanes_fs_prune_beam_v1_contract_sha256() -> str:
    return _json_sha256(
        canonical_sr_snake_macro_only_physical_lanes_fs_prune_beam_v1_contract()
    )


def canonical_sr_snake_macro_only_physical_lanes_fs_prune_beam_one_sided_cost_v1_contract() -> dict[str, Any]:
    """Return the exact one-sided-cost control of macro beam-plus-prune SR."""

    payload = canonical_sr_snake_macro_only_physical_lanes_fs_prune_beam_v1_contract()
    payload["route_profile"] = (
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_ONE_SIDED_COST_V1
    )
    payload["execution_settings"] = dict(
        CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_ONE_SIDED_COST_V1_EXECUTION_SETTINGS
    )
    payload["semantic_invariants"].update(
        {
            "hardware_cost_policy": "family_robust_v1",
            "cost_normalization_ablation": "one_sided_positive_excess_v1",
            "all_energy_models_infeasible_novelty_fallback_policy": (
                "collective_span_novelty_over_cost_v1"
            ),
        }
    )
    payload["lineage_authority"] = {
        "parent_route_profile": (
            SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_V1
        ),
        "parent_contract_sha256": (
            canonical_sr_snake_macro_only_physical_lanes_fs_prune_beam_v1_contract_sha256()
        ),
        "only_intended_parent_setting_changes": {
            "phase3_hardware_cost_normalization_mode": "family_robust_v1"
        },
        "scientific_result_anchor_claimed": False,
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def canonical_sr_snake_macro_only_physical_lanes_fs_prune_beam_one_sided_cost_v1_contract_sha256() -> str:
    return _json_sha256(
        canonical_sr_snake_macro_only_physical_lanes_fs_prune_beam_one_sided_cost_v1_contract()
    )


def canonical_sr_snake_symmetric_cost_fs_prune_v1_contract() -> dict[str, Any]:
    """Return the one-factor, undamped FS-trust pruning appendix contract."""

    payload = canonical_sr_snake_no_prune_symmetric_cost_v1_contract()
    payload["route_profile"] = SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1
    payload["execution_settings"] = dict(
        CANONICAL_SR_SNAKE_SYMMETRIC_COST_FS_PRUNE_V1_EXECUTION_SETTINGS
    )
    payload["semantic_invariants"].update(
        {
            "appendix_one_factor_ablation": True,
            "phase_live_hysteresis_enabled": False,
            "phase_retirement_policy": "disabled_v1",
            "pruning_active": True,
            "prune_execution_scope": "live_only_v1",
            "prune_response_scope": "full_active_logical_v1",
            "prune_trust_constraint": "complete_affine_deletion_fs_v1",
            "prune_initial_fs_radius": 0.125,
            "prune_radius_update_policy": (
                "rejection_contraction_only_half_to_1e-8_floor_v1"
            ),
            "prune_acceptance_authority": "measured_delete_and_refit_v1",
            "prune_metric_damping_active": False,
            "prune_metric_damping_update_active": False,
            "prune_endpoint_overlap_measurement_active": False,
            "prune_radius_update_measurement_delta": 0,
            "terminal_prune_active": False,
            "terminal_full_refit_active": False,
        }
    )
    payload["lineage_authority"] = {
        "parent_route_profile": SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
        "parent_contract_sha256": (
            canonical_sr_snake_no_prune_symmetric_cost_v1_contract_sha256()
        ),
        "scientific_result_anchor_claimed": False,
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def canonical_sr_snake_symmetric_cost_fs_prune_v1_contract_sha256() -> str:
    return _json_sha256(canonical_sr_snake_symmetric_cost_fs_prune_v1_contract())


def canonical_sr_snake_no_prune_symmetric_cost_beam_v1_contract() -> dict[str, Any]:
    """Return the appendix-only historical 3x2 beam ablation contract."""

    payload = canonical_sr_snake_no_prune_symmetric_cost_v1_contract()
    payload["route_profile"] = SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1
    payload["execution_settings"] = dict(
        CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1_EXECUTION_SETTINGS
    )
    payload["semantic_invariants"].update(
        {
            "appendix_one_factor_ablation": True,
            "phase_live_hysteresis_enabled": False,
            "phase_retirement_policy": "disabled_v1",
            "beam_shape": "historical_3x2_v1",
            "beam_live_branch_cap": 3,
            "beam_children_per_parent": 2,
            "beam_expanded_child_cap_per_round": 6,
            "beam_terminated_keep": 3,
            "beam_terminal_archive_mode": "legacy",
            "beam_parent_stop_terminal_also_materialized": True,
            "beam_structural_mode": "stop_or_single_admission",
            "beam_terminal_archive_accumulation": "cumulative_across_rounds",
            "beam_terminal_archive_cap": 3,
            "beam_lambda": 0.005,
        }
    )
    payload["lineage_authority"] = {
        "parent_route_profile": SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
        "parent_contract_sha256": (
            canonical_sr_snake_no_prune_symmetric_cost_v1_contract_sha256()
        ),
        "historical_semantics_source_commit": (
            "1f1d93c1a0060f0db70da6736cae4ec5ffffc79b^"
        ),
        "scientific_result_anchor_claimed": False,
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def canonical_sr_snake_no_prune_symmetric_cost_beam_v1_contract_sha256() -> str:
    return _json_sha256(
        canonical_sr_snake_no_prune_symmetric_cost_beam_v1_contract()
    )


def canonical_sr_snake_no_novelty_metric_prune_beam_v1_contract() -> dict[str, Any]:
    """Return the explicit SR-v3 no-novelty metric-prune beam contract."""

    payload: dict[str, Any] = {
        "schema": SR_ROUTE_PROFILE_CONTRACT_SCHEMA,
        "route_family": "singleton_response_snake",
        "route_profile": SR_ROUTE_PROFILE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1,
        "execution_settings": dict(
            CANONICAL_SR_SNAKE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1_EXECUTION_SETTINGS
        ),
        "semantic_invariants": {
            "regime_physics_source": "per_regime_source_lock",
            "same_cutoff_reference_required": True,
            "full_meta_hva_policy": "included_no_filters_v1",
            "phase3_response_coordinate_scope": (
                PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1
            ),
            "phase3_response_pre_support_invariant": (
                "response_count_equals_active_logical_count_plus_one_v1"
            ),
            "ordinary_phase2_novelty_active": False,
            "ordinary_phase3_novelty_active": False,
            "all_energy_models_infeasible_novelty_fallback_active": False,
            "novelty_telemetry_retained": True,
            "accepted_refit_scope": "full_ansatz_v1",
            "accepted_refit_coordinate_chart": "supported_fs_whitened_fixed_v1",
            "accepted_refit_base_chart_policy": (
                "expanded_runtime_projected_logical_v1"
            ),
            "beam_shape": "three_live_two_children_per_parent_v1",
            "prune_nomination_route": "metric_regularized_v1",
            "prune_metric_schur_mu": 0.01,
            "prune_acceptance_authority": "measured_delete_and_refit_v1",
            "terminal_prune_active": True,
            "admission_cardinality": 1,
            "phase2_batching_active": False,
            "phase3_batching_active": False,
            "child_padding_policy": "full_binary_code_space_v1",
            "child_padding_projection_required": False,
            "child_padding_reason": "nph1_binary_register_is_fully_physical_v1",
            "phase3_backend_cost_mode": "proxy",
            "phase3_backend_cost_reason": "generic_non_hh_application_v1",
            "route_a_funnel_active": False,
            "negative_curvature_escape_active": False,
        },
        "lineage_authority": {
            "parent_route_profile": SR_ROUTE_PROFILE_CONVENTIONAL_V3,
            "parent_contract_sha256": canonical_sr_snake_v3_contract_sha256(),
            "scientific_result_anchor_claimed": False,
        },
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def canonical_sr_snake_no_novelty_metric_prune_beam_v1_contract_sha256() -> str:
    return _json_sha256(
        canonical_sr_snake_no_novelty_metric_prune_beam_v1_contract()
    )


def canonical_sr_snake_h2o_derivative_resolved_v2_contract() -> dict[str, Any]:
    """Return the H2O derivative-resolved pool overlay on the SR baseline."""

    payload = canonical_sr_snake_no_novelty_metric_prune_beam_v1_contract()
    payload["route_profile"] = SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2
    payload["execution_settings"] = dict(
        CANONICAL_SR_SNAKE_H2O_DERIVATIVE_RESOLVED_V2_EXECUTION_SETTINGS
    )
    payload["semantic_invariants"].update(
        {
            "operator_pool_identity": "full_meta_derivative_resolved_v2",
            "baseline_full_meta_pool_retained": True,
            "derivative_factorization": (
                "one_and_two_body_symmetric_spectral_factors_v1"
            ),
            "coordinate_conditioned_uccsd_active": True,
            "physical_lane_classifier": (
                "h2o_linear_fd_physical_operator_lanes_v2_derivative_resolved"
            ),
        }
    )
    payload["lineage_authority"] = {
        "parent_route_profile": (
            SR_ROUTE_PROFILE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1
        ),
        "parent_contract_sha256": (
            canonical_sr_snake_no_novelty_metric_prune_beam_v1_contract_sha256()
        ),
        "only_intended_parent_setting_change": {
            "adapt_pool": "full_meta_derivative_resolved_v2"
        },
        "scientific_result_anchor_claimed": False,
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def canonical_sr_snake_h2o_derivative_resolved_v2_contract_sha256() -> str:
    return _json_sha256(canonical_sr_snake_h2o_derivative_resolved_v2_contract())


def canonical_sr_snake_h2o_derivative_resolved_paper_i_v3_contract() -> dict[
    str, Any
]:
    """Return the H2O overlay on the Paper-I no-prune/no-beam SR route."""

    payload = canonical_sr_snake_no_prune_symmetric_cost_v1_contract()
    payload["route_profile"] = (
        SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3
    )
    payload["execution_settings"] = dict(
        CANONICAL_SR_SNAKE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3_EXECUTION_SETTINGS
    )
    payload["semantic_invariants"].update(
        {
            "regime_physics_source": "paper_iv_h2o_fixture_source_lock_v1",
            "controller_horizon_source": "paper_iv_depth_50_source_lock_v1",
            "operator_pool_identity": "full_meta_derivative_resolved_v2",
            "baseline_full_meta_pool_retained": True,
            "derivative_factorization": (
                "one_and_two_body_symmetric_spectral_factors_v1"
            ),
            "coordinate_conditioned_uccsd_active": True,
            "physical_lane_classifier": (
                "h2o_linear_fd_physical_operator_lanes_v2_derivative_resolved"
            ),
            "child_padding_policy": "full_binary_code_space_v1",
            "child_padding_projection_required": False,
            "child_padding_reason": (
                "nph1_binary_register_is_fully_physical_v1"
            ),
            "phase3_backend_cost_mode": "proxy",
            "phase3_backend_cost_reason": "generic_non_hh_application_v1",
        }
    )
    payload["lineage_authority"] = {
        "parent_route_profile": SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
        "parent_contract_sha256": (
            canonical_sr_snake_no_prune_symmetric_cost_v1_contract_sha256()
        ),
        "only_intended_parent_setting_changes": {
            "problem": "molecular_vibronic_h2o_linear_fd",
            "adapt_max_depth": 50,
            "adapt_pool": "full_meta_derivative_resolved_v2",
            "phase3_runtime_split_child_padding_policy": (
                "full_binary_code_space_v1"
            ),
            "phase3_backend_cost_mode": "proxy",
        },
        "scientific_result_anchor_claimed": False,
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def canonical_sr_snake_h2o_derivative_resolved_paper_i_v3_contract_sha256() -> str:
    return _json_sha256(
        canonical_sr_snake_h2o_derivative_resolved_paper_i_v3_contract()
    )


def canonical_sr_snake_contract(profile: Any) -> dict[str, Any]:
    """Return the exact canonical contract for a normalized profile id."""

    requested = normalize_sr_route_profile_request(profile)
    if requested == SR_ROUTE_PROFILE_CANONICAL_V1:
        return canonical_sr_snake_v1_contract()
    if requested == SR_ROUTE_PROFILE_CONVENTIONAL_V2:
        return canonical_sr_snake_v2_contract()
    if requested == SR_ROUTE_PROFILE_CONVENTIONAL_V3:
        return canonical_sr_snake_v3_contract()
    if requested == SR_ROUTE_PROFILE_CANDIDATE_V4:
        return canonical_sr_snake_v4_contract()
    if requested == SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1:
        return canonical_sr_snake_no_prune_symmetric_cost_v1_contract()
    if requested == SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_V1:
        return canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_v1_contract()
    if requested == (
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1
    ):
        return canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract()
    if requested == (
        SR_ROUTE_PROFILE_GREEDY_BATCH_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1
    ):
        return canonical_sr_snake_greedy_batch_projected_phase3_no_overlap_trust_v1_contract()
    if requested == (
        SR_ROUTE_PROFILE_COMBINATORIAL_BATCH_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1
    ):
        return canonical_sr_snake_combinatorial_batch_projected_phase3_no_overlap_trust_v1_contract()
    if requested == (
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_V1
    ):
        return canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_commutation_reduced_insertion_v1_contract()
    if requested == SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V1:
        return canonical_sr_snake_insertion_commutation_plateau_v1_contract()
    if requested == SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V2:
        return canonical_sr_snake_insertion_commutation_plateau_v2_contract()
    if requested == (
        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_V1
    ):
        return canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_full_geometry_query_neutral_prune_v1_contract()
    if requested == (
        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_FS_PRUNE_VERIFY_V1
    ):
        return canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_full_geometry_fs_prune_verify_v1_contract()
    if requested == (
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_V1
    ):
        return canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_material_window_v1_contract()
    if requested == (
        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1
    ):
        return canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_material_window_fs_prune_verify_v1_contract()
    if requested == SR_ROUTE_PROFILE_GUARDED_SINGLETON_POOL_V1:
        return canonical_sr_snake_guarded_singleton_pool_v1_contract()
    if requested == SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_V1:
        return canonical_sr_snake_macro_only_physical_lanes_v1_contract()
    if requested == (
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_COMMUTATION_REDUCED_INSERTION_DIAGNOSTIC_V2
    ):
        return canonical_sr_snake_macro_only_physical_lanes_commutation_reduced_insertion_diagnostic_v2_contract()
    if requested == (
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V1
    ):
        return canonical_sr_snake_macro_only_physical_lanes_insertion_commutation_plateau_v1_contract()
    if requested == (
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V2
    ):
        return canonical_sr_snake_macro_only_physical_lanes_insertion_commutation_plateau_v2_contract()
    if requested == SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_ONE_SIDED_COST_V1:
        return canonical_sr_snake_macro_only_physical_lanes_one_sided_cost_v1_contract()
    if requested == SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_V1:
        return canonical_sr_snake_macro_only_physical_lanes_fs_prune_beam_v1_contract()
    if requested == (
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_ONE_SIDED_COST_V1
    ):
        return canonical_sr_snake_macro_only_physical_lanes_fs_prune_beam_one_sided_cost_v1_contract()
    if requested == SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1:
        return canonical_sr_snake_symmetric_cost_fs_prune_v1_contract()
    if requested == SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1:
        return canonical_sr_snake_no_prune_symmetric_cost_beam_v1_contract()
    if requested == SR_ROUTE_PROFILE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1:
        return canonical_sr_snake_no_novelty_metric_prune_beam_v1_contract()
    if requested == SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2:
        return canonical_sr_snake_h2o_derivative_resolved_v2_contract()
    if requested == SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3:
        return canonical_sr_snake_h2o_derivative_resolved_paper_i_v3_contract()
    raise ValueError("The off SR route profile has no canonical contract.")


def canonical_sr_snake_contract_sha256(profile: Any) -> str:
    return _json_sha256(canonical_sr_snake_contract(profile))


def normalize_sr_route_profile_request(raw: Any) -> str:
    key = str(SR_ROUTE_PROFILE_REQUEST_OFF if raw in {None, ""} else raw)
    key = key.strip().lower().replace("-", "_")
    aliases = {
        "none": SR_ROUTE_PROFILE_REQUEST_OFF,
        "disabled": SR_ROUTE_PROFILE_REQUEST_OFF,
        "sr_snake_v1": SR_ROUTE_PROFILE_CANONICAL_V1,
        "canonical_v1": SR_ROUTE_PROFILE_CANONICAL_V1,
        SR_ROUTE_PROFILE_CANONICAL_V1: SR_ROUTE_PROFILE_CANONICAL_V1,
        "sr_snake": SR_ROUTE_PROFILE_CONVENTIONAL_V3,
        "conventional": SR_ROUTE_PROFILE_CONVENTIONAL_V3,
        "canonical": SR_ROUTE_PROFILE_CONVENTIONAL_V3,
        "sr_snake_v2": SR_ROUTE_PROFILE_CONVENTIONAL_V2,
        "canonical_v2": SR_ROUTE_PROFILE_CONVENTIONAL_V2,
        SR_ROUTE_PROFILE_CONVENTIONAL_V2: SR_ROUTE_PROFILE_CONVENTIONAL_V2,
        "sr_snake_v3": SR_ROUTE_PROFILE_CONVENTIONAL_V3,
        "canonical_v3": SR_ROUTE_PROFILE_CONVENTIONAL_V3,
        SR_ROUTE_PROFILE_CONVENTIONAL_V3: SR_ROUTE_PROFILE_CONVENTIONAL_V3,
        "sr_snake_v4": SR_ROUTE_PROFILE_CANDIDATE_V4,
        "candidate_v4": SR_ROUTE_PROFILE_CANDIDATE_V4,
        SR_ROUTE_PROFILE_CANDIDATE_V4: SR_ROUTE_PROFILE_CANDIDATE_V4,
        "sr_snake_no_prune_symmetric_cost_v1": (
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1
        ),
        "no_prune_symmetric_cost_v1": (
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1
        ),
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1: (
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1
        ),
        "sr_snake_no_prune_symmetric_cost_projected_phase3_v1": (
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_V1
        ),
        "no_prune_symmetric_cost_projected_phase3_v1": (
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_V1
        ),
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_V1: (
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_V1
        ),
        "sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1": (
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1
        ),
        "no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1": (
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1
        ),
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1: (
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1
        ),
        SR_ROUTE_PROFILE_GREEDY_BATCH_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1: (
            SR_ROUTE_PROFILE_GREEDY_BATCH_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1
        ),
        SR_ROUTE_PROFILE_COMBINATORIAL_BATCH_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1: (
            SR_ROUTE_PROFILE_COMBINATORIAL_BATCH_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1
        ),
        "sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_commutation_reduced_insertion_v1": (
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_V1
        ),
        "no_prune_symmetric_cost_projected_phase3_no_overlap_trust_commutation_reduced_insertion_v1": (
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_V1
        ),
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_V1: (
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_V1
        ),
        "insertion_commutation_plateau_v1": (
            SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V1
        ),
        SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V1: (
            SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V1
        ),
        "insertion_commutation_plateau_v2": (
            SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V2
        ),
        SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V2: (
            SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V2
        ),
        "sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_full_geometry_query_neutral_prune_v1": (
            SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_V1
        ),
        "symmetric_cost_projected_phase3_no_overlap_trust_full_geometry_query_neutral_prune_v1": (
            SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_V1
        ),
        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_V1: (
            SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_V1
        ),
        "sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_full_geometry_fs_prune_verify_v1": (
            SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_FS_PRUNE_VERIFY_V1
        ),
        "symmetric_cost_projected_phase3_no_overlap_trust_full_geometry_fs_prune_verify_v1": (
            SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_FS_PRUNE_VERIFY_V1
        ),
        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_FS_PRUNE_VERIFY_V1: (
            SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_FS_PRUNE_VERIFY_V1
        ),
        "sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_material_window_v1": (
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_V1
        ),
        "no_prune_symmetric_cost_projected_phase3_no_overlap_trust_material_window_v1": (
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_V1
        ),
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_V1: (
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_V1
        ),
        "sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_material_window_fs_prune_verify_v1": (
            SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1
        ),
        "symmetric_cost_projected_phase3_no_overlap_trust_material_window_fs_prune_verify_v1": (
            SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1
        ),
        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1: (
            SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1
        ),
        "sr_snake_guarded_singleton_pool_v1": (
            SR_ROUTE_PROFILE_GUARDED_SINGLETON_POOL_V1
        ),
        "guarded_singleton_pool_v1": (
            SR_ROUTE_PROFILE_GUARDED_SINGLETON_POOL_V1
        ),
        SR_ROUTE_PROFILE_GUARDED_SINGLETON_POOL_V1: (
            SR_ROUTE_PROFILE_GUARDED_SINGLETON_POOL_V1
        ),
        "sr_snake_macro_only_physical_lanes_v1": (
            SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_V1
        ),
        "macro_only_physical_lanes_v1": (
            SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_V1
        ),
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_V1: (
            SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_V1
        ),
        "sr_snake_macro_only_physical_lanes_commutation_reduced_insertion_diagnostic_v2": (
            SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_COMMUTATION_REDUCED_INSERTION_DIAGNOSTIC_V2
        ),
        "macro_only_physical_lanes_commutation_reduced_insertion_diagnostic_v2": (
            SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_COMMUTATION_REDUCED_INSERTION_DIAGNOSTIC_V2
        ),
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_COMMUTATION_REDUCED_INSERTION_DIAGNOSTIC_V2: (
            SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_COMMUTATION_REDUCED_INSERTION_DIAGNOSTIC_V2
        ),
        "sr_snake_macro_only_physical_lanes_insertion_commutation_plateau_v1": (
            SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V1
        ),
        "macro_only_physical_lanes_insertion_commutation_plateau_v1": (
            SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V1
        ),
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V1: (
            SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V1
        ),
        "sr_snake_macro_only_physical_lanes_insertion_commutation_plateau_v2": (
            SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V2
        ),
        "macro_only_physical_lanes_insertion_commutation_plateau_v2": (
            SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V2
        ),
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V2: (
            SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V2
        ),
        "sr_snake_macro_only_physical_lanes_one_sided_cost_v1": (
            SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_ONE_SIDED_COST_V1
        ),
        "macro_only_physical_lanes_one_sided_cost_v1": (
            SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_ONE_SIDED_COST_V1
        ),
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_ONE_SIDED_COST_V1: (
            SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_ONE_SIDED_COST_V1
        ),
        "sr_snake_macro_only_physical_lanes_fs_prune_beam3x2_v1": (
            SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_V1
        ),
        "macro_only_physical_lanes_fs_prune_beam3x2_v1": (
            SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_V1
        ),
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_V1: (
            SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_V1
        ),
        "sr_snake_macro_only_physical_lanes_fs_prune_beam3x2_one_sided_cost_v1": (
            SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_ONE_SIDED_COST_V1
        ),
        "macro_only_physical_lanes_fs_prune_beam3x2_one_sided_cost_v1": (
            SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_ONE_SIDED_COST_V1
        ),
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_ONE_SIDED_COST_V1: (
            SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_ONE_SIDED_COST_V1
        ),
        "sr_snake_symmetric_cost_fs_prune_nodamping_v1": (
            SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1
        ),
        "symmetric_cost_fs_prune_nodamping_v1": (
            SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1
        ),
        SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1: (
            SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1
        ),
        "sr_snake_no_prune_symmetric_cost_beam_v1": (
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1
        ),
        "no_prune_symmetric_cost_beam_v1": (
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1
        ),
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1: (
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1
        ),
        "sr_snake_no_novelty_metric_prune_beam_v1": (
            SR_ROUTE_PROFILE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1
        ),
        "no_novelty_metric_prune_beam_v1": (
            SR_ROUTE_PROFILE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1
        ),
        SR_ROUTE_PROFILE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1: (
            SR_ROUTE_PROFILE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1
        ),
        "sr_snake_h2o_derivative_resolved_v2": (
            SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2
        ),
        "h2o_derivative_resolved_v2": (
            SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2
        ),
        SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2: (
            SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2
        ),
        "sr_snake_h2o_derivative_resolved_paper_i_v3": (
            SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3
        ),
        "h2o_derivative_resolved_paper_i_v3": (
            SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3
        ),
        SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3: (
            SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3
        ),
    }
    key = aliases.get(key, key)
    if key not in {
        SR_ROUTE_PROFILE_REQUEST_OFF,
        SR_ROUTE_PROFILE_CANONICAL_V1,
        SR_ROUTE_PROFILE_CONVENTIONAL_V2,
        SR_ROUTE_PROFILE_CONVENTIONAL_V3,
        SR_ROUTE_PROFILE_CANDIDATE_V4,
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_V1,
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1,
            SR_ROUTE_PROFILE_GREEDY_BATCH_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1,
            SR_ROUTE_PROFILE_COMBINATORIAL_BATCH_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1,
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_V1,
        SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V1,
        SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V2,
            SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_V1,
        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_FS_PRUNE_VERIFY_V1,
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_V1,
        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1,
        SR_ROUTE_PROFILE_GUARDED_SINGLETON_POOL_V1,
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_V1,
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_COMMUTATION_REDUCED_INSERTION_DIAGNOSTIC_V2,
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V1,
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V2,
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_ONE_SIDED_COST_V1,
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_V1,
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_ONE_SIDED_COST_V1,
        SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1,
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1,
        SR_ROUTE_PROFILE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1,
        SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2,
        SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3,
    }:
        raise ValueError(
            "sr_route_profile must be one of "
            f"{list(SR_ROUTE_PROFILE_REQUEST_CHOICES)}; got {raw!r}."
        )
    return key


def normalize_phase3_response_coordinate_scope(raw: Any) -> str:
    """Normalize one explicit Phase-III response-coordinate scope."""

    if raw in {None, ""}:
        raise ValueError("phase3_response_coordinate_scope is required.")
    key = str(raw).strip().lower().replace("-", "_")
    if key not in PHASE3_RESPONSE_COORDINATE_SCOPE_CHOICES:
        raise ValueError(
            "phase3_response_coordinate_scope must be one of "
            f"{list(PHASE3_RESPONSE_COORDINATE_SCOPE_CHOICES)}; got {raw!r}."
        )
    return key


def _comparable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return value.strip()
    return value


def _equivalent(field: str, actual: Any, expected: Any) -> bool:
    actual_value = _comparable(actual)
    expected_value = _comparable(expected)
    if field == "sr_powell_coordinate_chart_policy" and actual_value == "auto":
        return True
    if isinstance(expected_value, float):
        try:
            return float(actual_value) == float(expected_value)
        except (TypeError, ValueError):
            return False
    return actual_value == expected_value


def normalize_sr_route_profile_namespace(namespace: Any) -> Any:
    """Materialize the canonical profile and reject explicit setting drift.

    The parser records option strings that were explicitly present.  Implicit
    generic defaults are replaced by the profile.  Explicit matching values
    are accepted; explicit conflicting values fail closed.
    """

    requested = normalize_sr_route_profile_request(
        getattr(namespace, "sr_route_profile_request", None)
    )
    setattr(namespace, "sr_route_profile_request", requested)
    if requested == SR_ROUTE_PROFILE_REQUEST_OFF:
        setattr(namespace, "sr_route_profile_resolved", None)
        setattr(namespace, "sr_route_profile_contract", None)
        setattr(namespace, "sr_route_profile_contract_sha256", None)
        return namespace

    explicit_raw = getattr(namespace, "_explicit_cli_options", None)
    if explicit_raw is None:
        # A hand-built Namespace has no way to distinguish an implicit parser
        # default from an explicit scientific override.  It must already carry
        # the complete contract instead of being silently rewritten.
        explicit_options: frozenset[str] | None = None
    else:
        explicit_options = frozenset(str(value) for value in explicit_raw)

    if requested == SR_ROUTE_PROFILE_CANDIDATE_V4:
        execution_settings = CANONICAL_SR_SNAKE_V4_EXECUTION_SETTINGS
    elif requested == SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1:
        execution_settings = (
            CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_V1_EXECUTION_SETTINGS
        )
    elif requested == SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_V1:
        execution_settings = (
            CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_V1_EXECUTION_SETTINGS
        )
    elif requested == (
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1
    ):
        execution_settings = (
            CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1_EXECUTION_SETTINGS
        )
    elif requested == (
        SR_ROUTE_PROFILE_GREEDY_BATCH_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1
    ):
        execution_settings = (
            CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1_EXECUTION_SETTINGS
        )
    elif requested == (
        SR_ROUTE_PROFILE_COMBINATORIAL_BATCH_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1
    ):
        execution_settings = (
            CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1_EXECUTION_SETTINGS
        )
    elif requested == (
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_V1
    ):
        execution_settings = (
            CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_V1_EXECUTION_SETTINGS
        )
    elif requested == SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V1:
        execution_settings = (
            CANONICAL_SR_SNAKE_INSERTION_COMMUTATION_PLATEAU_V1_EXECUTION_SETTINGS
        )
    elif requested == SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V2:
        execution_settings = (
            CANONICAL_SR_SNAKE_INSERTION_COMMUTATION_PLATEAU_V2_EXECUTION_SETTINGS
        )
    elif requested == (
        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_V1
    ):
        execution_settings = (
            CANONICAL_SR_SNAKE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_V1_EXECUTION_SETTINGS
        )
    elif requested == (
        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_FS_PRUNE_VERIFY_V1
    ):
        execution_settings = (
            CANONICAL_SR_SNAKE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_FS_PRUNE_VERIFY_V1_EXECUTION_SETTINGS
        )
    elif requested == (
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_V1
    ):
        execution_settings = (
            CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_V1_EXECUTION_SETTINGS
        )
    elif requested == (
        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1
    ):
        execution_settings = (
            CANONICAL_SR_SNAKE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1_EXECUTION_SETTINGS
        )
    elif requested == SR_ROUTE_PROFILE_GUARDED_SINGLETON_POOL_V1:
        execution_settings = (
            CANONICAL_SR_SNAKE_GUARDED_SINGLETON_POOL_V1_EXECUTION_SETTINGS
        )
    elif requested == SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_V1:
        execution_settings = (
            CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_V1_EXECUTION_SETTINGS
        )
    elif requested == (
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_COMMUTATION_REDUCED_INSERTION_DIAGNOSTIC_V2
    ):
        execution_settings = (
            CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_COMMUTATION_REDUCED_INSERTION_DIAGNOSTIC_V2_EXECUTION_SETTINGS
        )
    elif requested == (
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V1
    ):
        execution_settings = (
            CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V1_EXECUTION_SETTINGS
        )
    elif requested == (
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V2
    ):
        execution_settings = (
            CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V2_EXECUTION_SETTINGS
        )
    elif requested == SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_ONE_SIDED_COST_V1:
        execution_settings = (
            CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_ONE_SIDED_COST_V1_EXECUTION_SETTINGS
        )
    elif requested == SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_V1:
        execution_settings = (
            CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_V1_EXECUTION_SETTINGS
        )
    elif requested == (
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_ONE_SIDED_COST_V1
    ):
        execution_settings = (
            CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_ONE_SIDED_COST_V1_EXECUTION_SETTINGS
        )
    elif requested == SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1:
        execution_settings = (
            CANONICAL_SR_SNAKE_SYMMETRIC_COST_FS_PRUNE_V1_EXECUTION_SETTINGS
        )
    elif requested == SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1:
        execution_settings = (
            CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1_EXECUTION_SETTINGS
        )
    elif requested == SR_ROUTE_PROFILE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1:
        execution_settings = {
            **CANONICAL_SR_SNAKE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1_EXECUTION_SETTINGS,
            **HISTORICAL_SR_SNAKE_PHASE12_ENERGY_MODEL_SETTINGS,
        }
    elif requested == SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2:
        execution_settings = {
            **CANONICAL_SR_SNAKE_H2O_DERIVATIVE_RESOLVED_V2_EXECUTION_SETTINGS,
            **HISTORICAL_SR_SNAKE_PHASE12_ENERGY_MODEL_SETTINGS,
        }
    elif requested == SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3:
        execution_settings = (
            CANONICAL_SR_SNAKE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3_EXECUTION_SETTINGS
        )
    elif requested == SR_ROUTE_PROFILE_CONVENTIONAL_V3:
        execution_settings = {
            **CANONICAL_SR_SNAKE_V3_EXECUTION_SETTINGS,
            **HISTORICAL_SR_SNAKE_PHASE12_ENERGY_MODEL_SETTINGS,
        }
    elif requested == SR_ROUTE_PROFILE_CONVENTIONAL_V2:
        execution_settings = {
            **CANONICAL_SR_SNAKE_V2_EXECUTION_SETTINGS,
            **HISTORICAL_SR_SNAKE_RESPONSE_SCOPE_SETTINGS,
            **HISTORICAL_SR_SNAKE_PHASE12_ENERGY_MODEL_SETTINGS,
        }
    else:
        execution_settings = {
            **CANONICAL_SR_SNAKE_V1_EXECUTION_SETTINGS,
            **HISTORICAL_SR_SNAKE_V1_ACCEPTED_REFIT_SETTINGS,
            **HISTORICAL_SR_SNAKE_RESPONSE_SCOPE_SETTINGS,
            **HISTORICAL_SR_SNAKE_PHASE12_ENERGY_MODEL_SETTINGS,
        }
    # Historical hashed contracts retain retired fields byte-for-byte, but
    # they are passive provenance and must never populate the namespace.
    retired_execution_fields = {
        "phase_live_hysteresis_enabled",
        "phase1_prune_stale_age",
        "phase1_prune_stagnation_threshold",
        "phase1_prune_small_theta_abs",
        "phase1_prune_small_theta_relative",
        "phase1_prune_amplitude_witness_required",
        "phase1_prune_collapse_peak_abs_min",
        "phase1_prune_collapse_current_abs_max",
        "phase1_prune_collapse_ratio",
        "phase1_prune_collapse_min_abs_drop",
        "phase1_prune_collapse_min_observations",
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
    execution_settings = {
        field: value
        for field, value in execution_settings.items()
        if field not in retired_execution_fields
    }
    conflicts: list[dict[str, Any]] = []
    if requested in {
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_V1,
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1,
        SR_ROUTE_PROFILE_GREEDY_BATCH_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1,
        SR_ROUTE_PROFILE_COMBINATORIAL_BATCH_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1,
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_V1,
        SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V1,
        SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V2,
        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_V1,
        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_FS_PRUNE_VERIFY_V1,
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_V1,
        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1,
        SR_ROUTE_PROFILE_GUARDED_SINGLETON_POOL_V1,
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_V1,
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_COMMUTATION_REDUCED_INSERTION_DIAGNOSTIC_V2,
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V1,
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V2,
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_ONE_SIDED_COST_V1,
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_V1,
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_ONE_SIDED_COST_V1,
        SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1,
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1,
    }:
        horizon_options = frozenset(
            _DEST_OPTION_STRINGS.get("adapt_max_depth", ("--adapt-max-depth",))
        )
        horizon_explicit = bool(
            explicit_options is not None
            and explicit_options.intersection(horizon_options)
        )
        raw_horizon = getattr(namespace, "adapt_max_depth", None)
        try:
            source_locked_horizon = int(raw_horizon)
        except (TypeError, ValueError):
            source_locked_horizon = 0
        if not horizon_explicit or source_locked_horizon < 1:
            conflicts.append(
                {
                    "field": "adapt_max_depth",
                    "explicit_options": sorted(
                        horizon_options.intersection(explicit_options or ())
                    ),
                    "current": _comparable(raw_horizon),
                    "required": "explicit_positive_per_regime_source_lock",
                }
            )
    for field, expected in execution_settings.items():
        current = getattr(namespace, field, None)
        option_strings = _DEST_OPTION_STRINGS.get(field, ())
        field_explicit = bool(
            explicit_options is not None
            and explicit_options.intersection(option_strings)
        )
        disallowed = sorted(
            set(option_strings).intersection(_DISALLOWED_BOOLEAN_OPTIONS).intersection(
                explicit_options or ()
            )
        )
        if disallowed or (
            (field_explicit or explicit_options is None)
            and not _equivalent(field, current, expected)
        ):
            conflicts.append(
                {
                    "field": field,
                    "explicit_options": sorted(
                        set(option_strings).intersection(explicit_options or ())
                    ),
                    "current": _comparable(current),
                    "required": expected,
                }
            )
            continue
        setattr(namespace, field, expected)

    if conflicts:
        raise ValueError(
            "SR-SNAKE route profile conflicts with explicit or untracked "
            "scientific settings: "
            + json.dumps(conflicts, sort_keys=True, default=str)
        )

    contract = canonical_sr_snake_contract(requested)
    digest = canonical_sr_snake_contract_sha256(requested)
    setattr(namespace, "sr_route_profile_resolved", requested)
    setattr(namespace, "sr_route_profile_contract", contract)
    setattr(namespace, "sr_route_profile_contract_sha256", digest)
    return namespace


def validate_sr_route_profile_contract(
    *,
    profile_request: Any,
    contract: Mapping[str, Any] | None,
    contract_sha256: str | None,
) -> dict[str, Any] | None:
    """Validate an already-normalized runtime/checkpoint contract."""

    requested = normalize_sr_route_profile_request(profile_request)
    if requested == SR_ROUTE_PROFILE_REQUEST_OFF:
        if (
            (contract is not None and bool(dict(contract)))
            or contract_sha256 not in {None, ""}
        ):
            raise ValueError(
                "An SR route contract was supplied while sr_route_profile is off."
            )
        return None
    if not isinstance(contract, Mapping):
        raise ValueError(
            "The requested SR-SNAKE profile requires its complete "
            "route-profile contract."
        )
    payload = dict(contract)
    expected = canonical_sr_snake_contract(requested)
    if payload != expected:
        raise ValueError("SR-SNAKE route-profile contract drifted.")
    actual_digest = _json_sha256(payload)
    expected_digest = canonical_sr_snake_contract_sha256(requested)
    if actual_digest != expected_digest or str(contract_sha256 or "") != expected_digest:
        raise ValueError(
            "SR-SNAKE route-profile contract SHA-256 is missing or drifted."
        )
    return payload


def validate_sr_route_profile_runtime_settings(
    *,
    profile_request: Any,
    contract: Mapping[str, Any] | None,
    contract_sha256: str | None,
    runtime_settings: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Require supplied effective runtime settings to match the profile."""

    payload = validate_sr_route_profile_contract(
        profile_request=profile_request,
        contract=contract,
        contract_sha256=contract_sha256,
    )
    if payload is None:
        return None
    requested = normalize_sr_route_profile_request(profile_request)
    expected = dict(payload["execution_settings"])
    # Authenticate historical contract bytes unchanged while keeping the
    # retired phase-liveness and ordinary-novelty fields passive and outside
    # execution. The active runtime resolves only the explicitly retained
    # deferred-Gram fallback from the historical Gram-policy pair.
    retired_execution_fields = (
        "phase_live_hysteresis_enabled",
        "phase2_novelty_mode",
        "phase3_novelty_ablation_mode",
        "phase1_prune_stale_age",
        "phase1_prune_stagnation_threshold",
        "phase1_prune_small_theta_abs",
        "phase1_prune_small_theta_relative",
        "phase1_prune_amplitude_witness_required",
        "phase1_prune_collapse_peak_abs_min",
        "phase1_prune_collapse_current_abs_max",
        "phase1_prune_collapse_ratio",
        "phase1_prune_collapse_min_abs_drop",
        "phase1_prune_collapse_min_observations",
    )
    for field in retired_execution_fields:
        expected.pop(field, None)
    if requested == SR_ROUTE_PROFILE_CANONICAL_V1:
        expected.update(HISTORICAL_SR_SNAKE_V1_ACCEPTED_REFIT_SETTINGS)
    if requested in {
        SR_ROUTE_PROFILE_CANONICAL_V1,
        SR_ROUTE_PROFILE_CONVENTIONAL_V2,
    }:
        expected.update(HISTORICAL_SR_SNAKE_RESPONSE_SCOPE_SETTINGS)
    if requested in {
        SR_ROUTE_PROFILE_CANONICAL_V1,
        SR_ROUTE_PROFILE_CONVENTIONAL_V2,
        SR_ROUTE_PROFILE_CONVENTIONAL_V3,
        SR_ROUTE_PROFILE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1,
        SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2,
    }:
        expected.update(HISTORICAL_SR_SNAKE_PHASE12_ENERGY_MODEL_SETTINGS)
    runtime_settings_checked = dict(runtime_settings)
    for field in retired_execution_fields:
        runtime_settings_checked.pop(field, None)
    if requested in {
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_V1,
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1,
        SR_ROUTE_PROFILE_GREEDY_BATCH_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1,
        SR_ROUTE_PROFILE_COMBINATORIAL_BATCH_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1,
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_V1,
        SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V1,
        SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V2,
        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_V1,
        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_FS_PRUNE_VERIFY_V1,
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_V1,
        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1,
        SR_ROUTE_PROFILE_GUARDED_SINGLETON_POOL_V1,
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_V1,
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_COMMUTATION_REDUCED_INSERTION_DIAGNOSTIC_V2,
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V1,
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V2,
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_ONE_SIDED_COST_V1,
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_V1,
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_ONE_SIDED_COST_V1,
        SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1,
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1,
    }:
        if "adapt_max_depth" not in runtime_settings_checked:
            raise ValueError(
                "The no-prune symmetric-cost SR profile requires an explicit "
                "positive source-locked adapt_max_depth."
            )
        try:
            source_locked_horizon = int(
                runtime_settings_checked.pop("adapt_max_depth")
            )
        except (TypeError, ValueError):
            raise ValueError(
                "The no-prune symmetric-cost SR profile requires a positive "
                "source-locked adapt_max_depth."
            ) from None
        if source_locked_horizon < 1:
            raise ValueError(
                "The no-prune symmetric-cost SR profile requires a positive "
                "source-locked adapt_max_depth."
            )
    if requested in {
        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_V1,
        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_FS_PRUNE_VERIFY_V1,
        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1,
    }:
        required_runtime_fields = {
            key for key in expected if str(key).startswith("phase1_prune_")
        }.union(
            {
                "phase3_shadow_damping_policy",
            }
        )
        missing_required = sorted(
            required_runtime_fields.difference(runtime_settings_checked)
        )
        if missing_required:
            raise ValueError(
                "The detailed prune SR profile requires "
                "every detailed prune/source-lock runtime field: "
                + ",".join(missing_required)
            )
    missing = sorted(set(runtime_settings_checked).difference(expected))
    if missing:
        raise ValueError(
            "SR-SNAKE runtime validator received unknown contract fields: "
            + ",".join(missing)
        )
    mismatches = [
        {
            "field": field,
            "runtime": _comparable(value),
            "required": expected[field],
        }
        for field, value in runtime_settings_checked.items()
        if not _equivalent(field, value, expected[field])
    ]
    if mismatches:
        raise ValueError(
            "SR-SNAKE effective runtime settings drifted: "
            + json.dumps(mismatches, sort_keys=True, default=str)
        )
    return payload


__all__ = [
    "CANONICAL_SR_SNAKE_V1_EXECUTION_SETTINGS",
    "CANONICAL_SR_SNAKE_V2_EXECUTION_SETTINGS",
    "CANONICAL_SR_SNAKE_V3_EXECUTION_SETTINGS",
    "CANONICAL_SR_SNAKE_V4_EXECUTION_SETTINGS",
    "CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_V1_EXECUTION_SETTINGS",
    "CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_V1_EXECUTION_SETTINGS",
    "CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1_EXECUTION_SETTINGS",
    "CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_V1_EXECUTION_SETTINGS",
    "CANONICAL_SR_SNAKE_INSERTION_COMMUTATION_PLATEAU_V1_EXECUTION_SETTINGS",
    "CANONICAL_SR_SNAKE_INSERTION_COMMUTATION_PLATEAU_V2_EXECUTION_SETTINGS",
    "CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_V1_EXECUTION_SETTINGS",
    "CANONICAL_SR_SNAKE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_V1_EXECUTION_SETTINGS",
    "CANONICAL_SR_SNAKE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_FS_PRUNE_VERIFY_V1_EXECUTION_SETTINGS",
    "CANONICAL_SR_SNAKE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1_EXECUTION_SETTINGS",
    "CANONICAL_SR_SNAKE_GUARDED_SINGLETON_POOL_V1_EXECUTION_SETTINGS",
    "CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_V1_EXECUTION_SETTINGS",
    "CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_COMMUTATION_REDUCED_INSERTION_DIAGNOSTIC_V2_EXECUTION_SETTINGS",
    "CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V1_EXECUTION_SETTINGS",
    "CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V2_EXECUTION_SETTINGS",
    "CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_ONE_SIDED_COST_V1_EXECUTION_SETTINGS",
    "CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_V1_EXECUTION_SETTINGS",
    "CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_ONE_SIDED_COST_V1_EXECUTION_SETTINGS",
    "CANONICAL_SR_SNAKE_SYMMETRIC_COST_FS_PRUNE_V1_EXECUTION_SETTINGS",
    "CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1_EXECUTION_SETTINGS",
    "CANONICAL_SR_SNAKE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1_EXECUTION_SETTINGS",
    "CANONICAL_SR_SNAKE_H2O_DERIVATIVE_RESOLVED_V2_EXECUTION_SETTINGS",
    "CANONICAL_SR_SNAKE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3_EXECUTION_SETTINGS",
    "HISTORICAL_SR_SNAKE_PHASE12_ENERGY_MODEL_SETTINGS",
    "HISTORICAL_SR_SNAKE_V1_ACCEPTED_REFIT_SETTINGS",
    "HISTORICAL_SR_SNAKE_RESPONSE_SCOPE_SETTINGS",
    "INSERTION_COMMUTATION_PLATEAU_CALIBRATION_STATUS",
    "INSERTION_COMMUTATION_PLATEAU_CUMULATIVE_DECREASE_RATIO_THRESHOLD",
    "INSERTION_COMMUTATION_PLATEAU_PRIOR_MEAN_CALIBRATION_STATUS",
    "INSERTION_COMMUTATION_PLATEAU_PRIOR_MEAN_DECREASE_RATIO_THRESHOLD",
    "PHASE3_RESPONSE_COORDINATE_SCOPE_CHOICES",
    "PHASE3_RESPONSE_COORDINATE_SCOPE_CANDIDATE_MATERIAL_COUPLING_WINDOW_V1",
    "PHASE3_RESPONSE_COORDINATE_SCOPE_FIXED_LOCAL_WINDOW_V1",
    "PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1",
    "PHASE3_RESPONSE_COORDINATE_SCOPE_LEGACY_REOPT_COUPLED_V1",
    "SR_ROUTE_PROFILE_CANONICAL_ALIAS",
    "SR_ROUTE_PROFILE_CANONICAL_V1",
    "SR_ROUTE_PROFILE_CONVENTIONAL_ALIAS",
    "SR_ROUTE_PROFILE_CONVENTIONAL_ALIAS_V2",
    "SR_ROUTE_PROFILE_CONVENTIONAL_ALIAS_V3",
    "SR_ROUTE_PROFILE_CANDIDATE_ALIAS_V4",
    "SR_ROUTE_PROFILE_CANDIDATE_V4",
    "SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_ALIAS_V1",
    "SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1",
    "SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_ALIAS_V1",
    "SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_V1",
    "SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_ALIAS_V1",
    "SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1",
    "SR_ROUTE_PROFILE_GREEDY_BATCH_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1",
    "SR_ROUTE_PROFILE_COMBINATORIAL_BATCH_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1",
    "SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_ALIAS_V1",
    "SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_V1",
    "SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_ALIAS_V1",
    "SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V1",
    "SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_ALIAS_V2",
    "SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V2",
    "SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_ALIAS_V1",
    "SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_V1",
    "SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_ALIAS_V1",
    "SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_V1",
    "SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_FS_PRUNE_VERIFY_ALIAS_V1",
    "SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_FS_PRUNE_VERIFY_V1",
    "SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_ALIAS_V1",
    "SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1",
    "SR_ROUTE_PROFILE_GUARDED_SINGLETON_POOL_ALIAS_V1",
    "SR_ROUTE_PROFILE_GUARDED_SINGLETON_POOL_V1",
    "SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_ALIAS_V1",
    "SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_V1",
    "SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_COMMUTATION_REDUCED_INSERTION_DIAGNOSTIC_ALIAS_V2",
    "SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_COMMUTATION_REDUCED_INSERTION_DIAGNOSTIC_V2",
    "SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_ALIAS_V1",
    "SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V1",
    "SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_ALIAS_V2",
    "SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V2",
    "SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_ONE_SIDED_COST_ALIAS_V1",
    "SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_ONE_SIDED_COST_V1",
    "SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_ALIAS_V1",
    "SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_V1",
    "SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_ONE_SIDED_COST_ALIAS_V1",
    "SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_ONE_SIDED_COST_V1",
    "SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_ALIAS_V1",
    "SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1",
    "SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_ALIAS_V1",
    "SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1",
    "SR_ROUTE_PROFILE_NO_NOVELTY_METRIC_PRUNE_BEAM_ALIAS_V1",
    "SR_ROUTE_PROFILE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1",
    "SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_ALIAS_V2",
    "SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2",
    "SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_PAPER_I_ALIAS_V3",
    "SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3",
    "SR_ROUTE_PROFILE_CONVENTIONAL_V2",
    "SR_ROUTE_PROFILE_CONVENTIONAL_V3",
    "SR_ROUTE_PROFILE_CONTRACT_DIGEST_SCHEMA",
    "SR_ROUTE_PROFILE_CONTRACT_SCHEMA",
    "SR_ROUTE_PROFILE_REQUEST_CHOICES",
    "SR_ROUTE_PROFILE_REQUEST_OFF",
    "canonical_sr_snake_contract",
    "canonical_sr_snake_contract_sha256",
    "canonical_sr_snake_v1_contract",
    "canonical_sr_snake_v1_contract_sha256",
    "canonical_sr_snake_v2_contract",
    "canonical_sr_snake_v2_contract_sha256",
    "canonical_sr_snake_v3_contract",
    "canonical_sr_snake_v3_contract_sha256",
    "canonical_sr_snake_v4_contract",
    "canonical_sr_snake_v4_contract_sha256",
    "canonical_sr_snake_no_prune_symmetric_cost_v1_contract",
    "canonical_sr_snake_no_prune_symmetric_cost_v1_contract_sha256",
    "canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_v1_contract",
    "canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_v1_contract_sha256",
    "canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract",
    "canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract_sha256",
    "canonical_sr_snake_greedy_batch_projected_phase3_no_overlap_trust_v1_contract",
    "canonical_sr_snake_greedy_batch_projected_phase3_no_overlap_trust_v1_contract_sha256",
    "canonical_sr_snake_combinatorial_batch_projected_phase3_no_overlap_trust_v1_contract",
    "canonical_sr_snake_combinatorial_batch_projected_phase3_no_overlap_trust_v1_contract_sha256",
    "canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_commutation_reduced_insertion_v1_contract",
    "canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_commutation_reduced_insertion_v1_contract_sha256",
    "canonical_sr_snake_insertion_commutation_plateau_v1_contract",
    "canonical_sr_snake_insertion_commutation_plateau_v1_contract_sha256",
    "canonical_sr_snake_insertion_commutation_plateau_v2_contract",
    "canonical_sr_snake_insertion_commutation_plateau_v2_contract_sha256",
    "canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_full_geometry_query_neutral_prune_v1_contract",
    "canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_full_geometry_query_neutral_prune_v1_contract_sha256",
    "canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_full_geometry_fs_prune_verify_v1_contract",
    "canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_full_geometry_fs_prune_verify_v1_contract_sha256",
    "canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_material_window_v1_contract",
    "canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_material_window_v1_contract_sha256",
    "canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_material_window_fs_prune_verify_v1_contract",
    "canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_material_window_fs_prune_verify_v1_contract_sha256",
    "canonical_sr_snake_guarded_singleton_pool_v1_contract",
    "canonical_sr_snake_guarded_singleton_pool_v1_contract_sha256",
    "canonical_sr_snake_macro_only_physical_lanes_v1_contract",
    "canonical_sr_snake_macro_only_physical_lanes_v1_contract_sha256",
    "canonical_sr_snake_macro_only_physical_lanes_commutation_reduced_insertion_diagnostic_v2_contract",
    "canonical_sr_snake_macro_only_physical_lanes_commutation_reduced_insertion_diagnostic_v2_contract_sha256",
    "canonical_sr_snake_macro_only_physical_lanes_insertion_commutation_plateau_v1_contract",
    "canonical_sr_snake_macro_only_physical_lanes_insertion_commutation_plateau_v1_contract_sha256",
    "canonical_sr_snake_macro_only_physical_lanes_insertion_commutation_plateau_v2_contract",
    "canonical_sr_snake_macro_only_physical_lanes_insertion_commutation_plateau_v2_contract_sha256",
    "canonical_sr_snake_macro_only_physical_lanes_one_sided_cost_v1_contract",
    "canonical_sr_snake_macro_only_physical_lanes_one_sided_cost_v1_contract_sha256",
    "canonical_sr_snake_macro_only_physical_lanes_fs_prune_beam_v1_contract",
    "canonical_sr_snake_macro_only_physical_lanes_fs_prune_beam_v1_contract_sha256",
    "canonical_sr_snake_macro_only_physical_lanes_fs_prune_beam_one_sided_cost_v1_contract",
    "canonical_sr_snake_macro_only_physical_lanes_fs_prune_beam_one_sided_cost_v1_contract_sha256",
    "canonical_sr_snake_symmetric_cost_fs_prune_v1_contract",
    "canonical_sr_snake_symmetric_cost_fs_prune_v1_contract_sha256",
    "canonical_sr_snake_no_prune_symmetric_cost_beam_v1_contract",
    "canonical_sr_snake_no_prune_symmetric_cost_beam_v1_contract_sha256",
    "canonical_sr_snake_no_novelty_metric_prune_beam_v1_contract",
    "canonical_sr_snake_no_novelty_metric_prune_beam_v1_contract_sha256",
    "canonical_sr_snake_h2o_derivative_resolved_v2_contract",
    "canonical_sr_snake_h2o_derivative_resolved_v2_contract_sha256",
    "canonical_sr_snake_h2o_derivative_resolved_paper_i_v3_contract",
    "canonical_sr_snake_h2o_derivative_resolved_paper_i_v3_contract_sha256",
    "normalize_phase3_response_coordinate_scope",
    "normalize_sr_route_profile_namespace",
    "normalize_sr_route_profile_request",
    "validate_sr_route_profile_contract",
    "validate_sr_route_profile_runtime_settings",
]
