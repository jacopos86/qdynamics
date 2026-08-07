#!/usr/bin/env python3
"""Generate CHTC record files for the generic static Table-I benchmark queue.

The queue is intentionally benchmark/comparator-only by default.  The project
controller/SNAKE row can be included with ``--include-snake`` when we want a
single combined table run, but normal CHTC benchmark submissions exclude it
because those canonical SNAKE sweeps are managed separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chtc.phase3_optuna.paper_i_clean_ladder_contract import (  # noqa: E402
    PAPER_I_CLEAN_TAU_PHYS as _PAPER_I_CLEAN_TAU_PHYS,
    PAPER_I_CLEAN_TAU_TIGHT as _PAPER_I_CLEAN_TAU_TIGHT,
    PAPER_I_LADDER_STAGE_CONFIGS as _SHARED_PAPER_I_LADDER_STAGE_CONFIGS,
    PAPER_I_PHONON_FAMILIES,
    PHONON_CUTOFF_TSV_FIELDS,
)
from chtc.phase3_optuna.paper_i_table_i_audit_escalation import (  # noqa: E402
    TARGET_STAGE as _PAPER_I_ESCALATION_TARGET_STAGE,
    candidates_for_lane as _escalation_candidates_for_lane,
    generic_target_record_id as _escalation_generic_target_record_id,
    source_metadata_fields as _escalation_source_metadata_fields,
    target_case_ids as _escalation_target_case_ids,
)
from pipelines.exact_bench.benchmark_decision_noise import (  # noqa: E402
    BENCHMARK_DECISION_NOISE_MODEL_CHOICES,
    BENCHMARK_DECISION_NOISE_SEMANTIC as _BENCHMARK_DECISION_NOISE_SEMANTIC,
    BENCHMARK_DECISION_NOISE_TSV_FIELDS,
)
from pipelines.exact_bench.table_i_static_benchmark import (  # noqa: E402
    TABLE_I_STATIC_ALGORITHM_IDS,
    TABLE_I_STATIC_BENCHMARK_ALGORITHM_IDS,
    build_table_i_static_jobs,
    summarize_table_i_jobs,
    table_i_method_label,
)
from pipelines.exact_bench.static_reference_metrics import (  # noqa: E402
    exact_energy_for_spec,
    materialize_reference_energy_cache,
)
from pipelines.exact_bench.table_i_canonical_cases import (  # noqa: E402
    TABLE_I_CLEAN_NPH1_REF4_PROFILE,
    TABLE_I_CLEAN_NPH2_REF3_PROFILE,
    TABLE_I_CLEAN_NPH2_REF4_PROFILE,
    TABLE_I_CLEAN_NPH2_REF5_PROFILE,
    TABLE_I_CLEAN_NPH3_REF4_PROFILE,
    TABLE_I_CLEAN_NPH4_REF5_PROFILE,
    TABLE_I_CLEAN_NPH4_REF7_PROFILE,
    TABLE_I_CLEAN_NPH6_REF9_PROFILE,
    TABLE_I_NPH2_REF3_PROFILE,
    TABLE_I_PAPER_I_MAIN_TABLES_SPSA_PROFILE,
    TABLE_I_STANDARD_PROFILE,
    TABLE_I_STATIC_SUITE_PROFILE_ENV,
    TABLE_I_THREE_MODEL_HH_SYMMETRIC_PROFILE,
    table_i_canonical_spec_by_case_id,
    table_i_suite_profile,
)
from pipelines.exact_bench.paper_i_main_tables_spsa_profile import (  # noqa: E402
    PAPER_I_MAIN_TABLES_SPSA_BUDGET_DEFAULTS,
    PAPER_I_MAIN_TABLES_SPSA_CASE_IDS,
    PAPER_I_MAIN_TABLES_SPSA_COMPARATOR_ALGORITHM_IDS,
    PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_TSV_FIELDS,
    PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID,
    PAPER_I_MAIN_TABLES_SPSA_SMOKE_BUDGET_DEFAULTS,
    PAPER_I_MAIN_TABLES_SPSA_SMOKE_CASE_IDS_BY_FAMILY,
    PAPER_I_MAIN_TABLES_SPSA_SNAKE_ALGORITHM_ID,
    PAPER_I_MAIN_TABLES_SPSA_TARGET,
)
from pipelines.reporting.benchmark_manifest import BenchmarkJob  # noqa: E402
from pipelines.static_adapt.route_identity import (  # noqa: E402
    ROUTE_A_REQUIRED_COMPONENTS,
    ROUTE_A_VERSION,
    ROUTE_B_REQUIRED_COMPONENTS,
    ROUTE_ID_A,
    ROUTE_ID_B_LEGACY_PAIRWISE,
    ROUTE_ID_UNSPECIFIED,
)

DEFAULT_INPUT_DIR = REPO_ROOT / "chtc" / "phase3_optuna" / "input"
DEFAULT_QUEUE_OUTPUT_ROOT = Path("raw_outputs/generic_static_table")
DEFAULT_ENERGY_STOP_TARGET = 1e-8
DEFAULT_FIRST_HIT_THRESHOLDS = (1e-6, 1e-8)
PHASE3_ORACLE_TSV_FIELDS = (
    "phase3_oracle_gradient_mode",
    "phase3_oracle_backend_name",
    "phase3_oracle_use_fake_backend",
    "phase3_oracle_shots",
    "phase3_oracle_repeats",
    "phase3_oracle_aggregate",
    "phase3_oracle_seed",
    "phase3_oracle_execution_surface",
    "phase3_oracle_inner_objective_mode",
    "phase3_oracle_value_noise_model",
    "phase3_oracle_value_noise_std",
    "phase3_oracle_value_noise_seed",
)
PHASE3_BUDGET_TSV_FIELDS = (
    "phase3_adapt_max_depth",
    "phase3_adapt_maxiter",
    "phase3_refit_maxiter",
    "phase3_final_maxiter",
    "phase3_adapt_spsa_a",
    "phase3_adapt_spsa_c",
    "phase3_adapt_spsa_big_a",
    "phase3_adapt_spsa_alpha",
    "phase3_adapt_spsa_gamma",
    "phase3_adapt_spsa_eval_repeats",
    "phase3_adapt_spsa_avg_last",
    "phase3_adapt_allow_repeats",
)
PHASE3_RUNTIME_TSV_FIELDS = (
    "phase3_adapt_parallel_gradient_workers",
    "phase3_adapt_beam_parent_workers",
)
PHASE3_POLICY_JSON_TSV_FIELDS = ("phase3_policy_json",)
PHASE3_POS_GEO_TSV_FIELDS = ("phase3_pos_geo_position_policy",)
HARDWARE_RESOLUTION_PROFILE_TSV_FIELDS = (
    "hardware_resolution_mode",
    "hardware_resolution_profile_json",
    "hardware_resolution_profile_name",
)
STATIC_ROUTE_TSV_FIELDS = (
    "static_route_id",
    "route_base_pool_key",
    "canonical_snake_eligible_expected",
    "route_evidence_role",
    "phase3_selector_policy",
    "phase3_selector_geometry_mode",
    "algebraic_shortlisting_enabled",
    "hardware_resolution_schema",
    "phase2_raw_score_formula",
    "canonical_score_formula",
    "primary_selector_score_key",
    "auxiliary_terms_primary_mode",
    "phase3_novelty_ablation_mode",
    "phase3_window_relaxation_mode",
    "phase2_enable_batching",
    "phase3_enable_batching",
    "phase3_batch_selection_mode",
    "phase3_batch_prefilter_mode",
    "phase3_nested_window_application",
    "phase1_prune_enabled",
    "phase1_prune_policy",
    "phase1_prune_mode",
    "phase1_prune_amplitude_witness_required",
    "continuation_mode",
)
STATIC_ROUTE_RUNTIME_TSV_FIELDS = ("phase2_novelty_mode",)
SELECTED_LOGICAL_TSV_FIELDS = (
    "selected_logical_route",
    "selected_logical_source_json",
    "selected_logical_transfer_mode",
)
GENERIC_ADAPT_RUNTIME_SPLIT_TSV_FIELDS = (
    "generic_adapt_runtime_split_mode",
    "generic_adapt_runtime_split_symmetry_policy",
    "generic_adapt_runtime_split_max_subset_size",
)
SHARED_PAULI_POOL_TSV_FIELDS = (
    "shared_pauli_pool_mode",
    "shared_pauli_pool_symmetry_policy",
    "shared_pauli_pool_max_subset_size",
)
RESOURCE_GUARD_TSV_FIELDS = (
    "resource_qubit_cap",
    "resource_pool_term_cap",
)
HEA_TSV_FIELDS = (
    "hea_reps",
    "hea_maxiter",
)
BENCHMARK_VALUE_NOISE_TSV_FIELDS = (
    "benchmark_value_noise_model",
    "benchmark_value_noise_std",
    "benchmark_value_noise_seed",
)
_PAPER_I_LADDER_STAGE_CONFIGS = {
    name: config.asdict() for name, config in _SHARED_PAPER_I_LADDER_STAGE_CONFIGS.items()
}
_PHASE3_ORACLE_GRADIENT_MODE_CHOICES = {
    "off",
    "ideal",
    "shots",
    "aer_noise",
    "aer_density_matrix",
    "backend_scheduled",
    "runtime",
}
_PHASE3_ORACLE_EXECUTION_SURFACE_CHOICES = {"auto", "expectation_v1", "raw_measurement_v1"}
_PHASE3_ORACLE_VALUE_NOISE_MODEL_CHOICES = {"off", "gaussian_iid_v1"}
_BENCHMARK_VALUE_NOISE_MODEL_CHOICES = {"off", "gaussian_iid_v1"}
_BENCHMARK_VALUE_NOISE_SEMANTIC = "post_static_result_value_noise_not_physical_shots"
_BENCHMARK_DECISION_NOISE_MODEL_CHOICES = set(BENCHMARK_DECISION_NOISE_MODEL_CHOICES)
_PHASE3_SMOKE_BUDGET_PROFILE_CHOICES = {"off", "weak_local_v1"}
_PHASE3_POLICY_PROFILE_CHOICES = {"off", "spsa_prior_best_v1", "spsa_prior_depth12_v1"}
_GENERIC_ADAPT_BUDGET_PROFILE_CHOICES = {
    "off",
    "paper_i_first_hit_depth256_v1",
    "paper_i_first_hit_depth500_v1",
}
_GENERIC_ADAPT_SMOKE_BUDGET_PROFILE_CHOICES = {"off", "weak_local_v1"}
_GENERIC_ADAPT_RUNTIME_SPLIT_MODE_CHOICES = {"off", "shortlist_pauli_children_v1"}
_GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICY_CHOICES = {"off", "hard_guard"}
_GENERIC_ADAPT_RUNTIME_SPLIT_SUPPORTED_ALGORITHM_IDS = {
    "static_full_meta_append_adapt_vqe",
    "static_geo_adapt_vqe",
}
_GENERIC_ADAPT_RUNTIME_SPLIT_SUPPORTED_FAMILIES = {"hh", "hubbard"}
_SHARED_PAULI_POOL_MODE_CHOICES = {"off", "shared_pauli_child_sets_v1", "pauli_child_sets_v1", "global_pauli_child_sets_v1"}
_SHARED_PAULI_POOL_SYMMETRY_POLICY_CHOICES = {"off", "hard_guard"}
_SHARED_PAULI_POOL_SUPPORTED_ALGORITHM_IDS = {
    "static_family_native_adapt_phase3",
    "static_full_meta_append_adapt_vqe",
    "static_geo_adapt_vqe",
}
_SHARED_PAULI_POOL_SUPPORTED_FAMILIES = {"hh", "hubbard"}
_PHASE2_NOVELTY_MODE_CHOICES = {"", "collective_span_v1", "legacy_pairwise_v1"}
_CALIBRATION_PROFILE_CHOICES = {"off", "nph2_route_a_hk_hh_v1"}
_NPH2_ROUTE_A_HK_HH_CALIBRATION_TARGET_RECORD_IDS = (
    "static_table__harmonic_kerr_chain__harmonic_kerr_chain_L2_nph2__static_family_native_adapt_phase3",
    "static_table__harmonic_kerr_chain__harmonic_kerr_chain_L2_nph2_w0p75__static_family_native_adapt_phase3",
    "static_table__hh__hh_L2_nph2__static_family_native_adapt_phase3",
)
_NPH2_ROUTE_A_HK_HH_CALIBRATION_SMOKE_RECORD_IDS = (
    "static_table__harmonic_kerr_chain__harmonic_kerr_chain_L2_nph2__static_family_native_adapt_phase3",
    "static_table__hh__hh_L2_nph2__static_family_native_adapt_phase3",
)
_NPH2_ROUTE_A_HK_HH_CALIBRATION_ROUTE_IDENTITY = ROUTE_A_VERSION
_NPH2_ROUTE_A_HK_HH_CALIBRATION_WORKING_CUTOFF = 2
_NPH2_ROUTE_A_HK_HH_CALIBRATION_REF_CUTOFF = 3
_PHASE3_WEAK_LOCAL_BUDGET_FIELDS = {
    "phase3_adapt_max_depth": "1",
    "phase3_adapt_maxiter": "1",
    "phase3_refit_maxiter": "1",
    "phase3_final_maxiter": "1",
    "phase3_adapt_spsa_eval_repeats": "1",
    "phase3_adapt_spsa_avg_last": "0",
}
_PHASE3_SPSA_PRIOR_BEST_FIELDS = {
    "phase3_adapt_maxiter": "8000",
    "phase3_refit_maxiter": "8000",
    "phase3_final_maxiter": "8000",
    "phase3_adapt_spsa_a": "0.05",
    "phase3_adapt_spsa_c": "0.02",
    "phase3_adapt_spsa_big_a": "50.0",
    "phase3_adapt_spsa_alpha": "0.602",
    "phase3_adapt_spsa_gamma": "0.101",
    "phase3_adapt_spsa_eval_repeats": "1",
    "phase3_adapt_spsa_avg_last": "0",
    "phase3_adapt_allow_repeats": "true",
}
_PHASE3_SPSA_PRIOR_DEPTH12_FIELDS = {
    **_PHASE3_SPSA_PRIOR_BEST_FIELDS,
    "phase3_adapt_max_depth": "12",
}
_GENERIC_ADAPT_FIRST_HIT_DEPTH500_FIELDS = {
    "phase3_adapt_max_depth": "500",
    "phase3_adapt_maxiter": "5000",
    "phase3_adapt_allow_repeats": "true",
}
_GENERIC_ADAPT_FIRST_HIT_DEPTH256_FIELDS = {
    "phase3_adapt_max_depth": "256",
    "phase3_adapt_maxiter": "2500",
    "phase3_adapt_allow_repeats": "true",
}
_SELECTED_LOGICAL_SUPPORTED_ALGORITHM_IDS = {
    "static_full_meta_append_adapt_vqe",
    "static_tetris_qubit_adapt_vqe",
    "static_geo_adapt_vqe",
    "static_pos_geo_adapt_vqe",
    "static_family_native_adapt_phase3",
}
_PHASE3_STATIC_ADAPT_ALGORITHM_IDS = {"static_family_native_adapt_phase3", "static_append_only_adapt_phase3"}
_GENERIC_STATIC_ADAPT_VARIANT_ALGORITHM_IDS = {
    "static_full_meta_append_adapt_vqe",
    "static_tetris_qubit_adapt_vqe",
    "static_geo_adapt_vqe",
    "static_pos_geo_adapt_vqe",
    "static_qubit_qeb_adapt_vqe",
    "static_geo_qubit_adapt_vqe",
    "static_geo_qeb_adapt_vqe",
}
_SNAKE_TABLE_I_ALGORITHM_IDS = ("static_family_native_adapt_phase3",)


def _unique_float_thresholds(values: Sequence[float]) -> tuple[float, ...]:
    """Preserve order while removing duplicate numeric thresholds."""

    out: list[float] = []
    for value in values:
        number = float(value)
        if not any(math.isclose(number, seen, rel_tol=0.0, abs_tol=1e-15) for seen in out):
            out.append(number)
    return tuple(out)


def _smoke_record_ids_for_profile(profile: str) -> tuple[str, ...]:
    profile_key = table_i_suite_profile(profile)
    if profile_key == TABLE_I_PAPER_I_MAIN_TABLES_SPSA_PROFILE:
        return tuple(
            f"static_table__{family}__{case_id}__{algorithm_id}"
            for family, case_ids in PAPER_I_MAIN_TABLES_SPSA_SMOKE_CASE_IDS_BY_FAMILY.items()
            for case_id in case_ids
            for algorithm_id in PAPER_I_MAIN_TABLES_SPSA_COMPARATOR_ALGORITHM_IDS
        )
    if profile_key == TABLE_I_CLEAN_NPH2_REF4_PROFILE:
        return (
            "static_table__bose_hubbard__bose_hubbard_L2_nph2_clean_weak__static_hea_qiskit_vqe",
            "static_table__harmonic_kerr_chain__harmonic_kerr_chain_L2_nph2_clean_weak__static_family_informed_vqe",
            "static_table__hh__hh_L2_nph2_clean_weak__static_full_meta_append_adapt_vqe",
        )
    if profile_key == TABLE_I_CLEAN_NPH1_REF4_PROFILE:
        return (
            "static_table__bose_hubbard__bose_hubbard_L2_nph1_clean_weak__static_hea_qiskit_vqe",
            "static_table__harmonic_kerr_chain__harmonic_kerr_chain_L2_nph1_clean_weak__static_family_informed_vqe",
            "static_table__hh__hh_L2_nph1_clean_weak__static_full_meta_append_adapt_vqe",
        )
    if profile_key == TABLE_I_CLEAN_NPH2_REF5_PROFILE:
        return (
            "static_table__spin_boson__spin_boson_L2_nph2_clean_weak__static_hea_qiskit_vqe",
            "static_table__hh__hh_L2_nph2_clean_strong__static_full_meta_append_adapt_vqe",
        )
    if profile_key == TABLE_I_CLEAN_NPH3_REF4_PROFILE:
        return (
            "static_table__bose_hubbard__bose_hubbard_L2_nph3_clean_weak__static_hea_qiskit_vqe",
            "static_table__harmonic_kerr_chain__harmonic_kerr_chain_L2_nph3_clean_weak__static_family_informed_vqe",
            "static_table__hh__hh_L2_nph3_clean_weak__static_full_meta_append_adapt_vqe",
        )
    if profile_key == TABLE_I_CLEAN_NPH4_REF5_PROFILE:
        return (
            "static_table__bose_hubbard__bose_hubbard_L2_nph4_clean_weak__static_hea_qiskit_vqe",
            "static_table__harmonic_kerr_chain__harmonic_kerr_chain_L2_nph4_clean_weak__static_family_informed_vqe",
            "static_table__hh__hh_L2_nph4_clean_weak__static_full_meta_append_adapt_vqe",
        )
    if profile_key == TABLE_I_CLEAN_NPH4_REF7_PROFILE:
        return (
            "static_table__harmonic_kerr_chain__harmonic_kerr_chain_L2_nph4_clean_weak__static_hea_qiskit_vqe",
            "static_table__harmonic_kerr_chain__harmonic_kerr_chain_L2_nph4_clean_strong__static_family_informed_vqe",
        )
    if profile_key == TABLE_I_CLEAN_NPH6_REF9_PROFILE:
        return (
            "static_table__spin_boson__spin_boson_L2_nph6_clean_strong__static_hea_qiskit_vqe",
        )
    if profile_key == TABLE_I_CLEAN_NPH2_REF3_PROFILE:
        return (
            "static_table__bose_hubbard__bose_hubbard_L2_nph2_clean_weak__static_hea_qiskit_vqe",
            "static_table__harmonic_kerr_chain__harmonic_kerr_chain_L2_nph2_clean_weak__static_family_informed_vqe",
            "static_table__hh__hh_L2_nph2_clean_weak__static_full_meta_append_adapt_vqe",
        )
    if profile_key == TABLE_I_NPH2_REF3_PROFILE:
        return (
            "static_table__hubbard__hubbard_L2__static_hea_qiskit_vqe",
            "static_table__bose_hubbard__bose_hubbard_L2_nph2__static_hea_qiskit_vqe",
            "static_table__hubbard__hubbard_L2__static_family_informed_vqe",
            "static_table__bose_hubbard__bose_hubbard_L2_nph2__static_family_informed_vqe",
            "static_table__hh__hh_L2_nph2__static_full_meta_append_adapt_vqe",
            "static_table__spinless_tv__spinless_tv_L2__static_full_meta_append_adapt_vqe",
            "static_table__spin_boson__spin_boson_L1_nph2__static_full_meta_append_adapt_vqe",
            "static_table__spinless_tv__spinless_tv_L2__static_qubit_qeb_adapt_vqe",
            "static_table__hh__hh_L2_nph2__static_qubit_qeb_adapt_vqe",
            "static_table__spin_boson__spin_boson_L1_nph2__static_tetris_qubit_adapt_vqe",
            "static_table__hh__hh_L2_nph2__static_tetris_qubit_adapt_vqe",
            "static_table__hubbard__hubbard_L2__static_geo_adapt_vqe",
            "static_table__hh__hh_L2_nph2__static_geo_adapt_vqe",
            "static_table__bose_hubbard__bose_hubbard_L2_nph2__static_geo_adapt_vqe",
        )
    return (
        "static_table__hubbard__hubbard_L2__static_hea_qiskit_vqe",
        "static_table__bose_hubbard__bose_hubbard_L2__static_hea_qiskit_vqe",
        "static_table__hubbard__hubbard_L2__static_family_informed_vqe",
        "static_table__bose_hubbard__bose_hubbard_L2__static_family_informed_vqe",
        "static_table__hh__hh_L2__static_full_meta_append_adapt_vqe",
        "static_table__spinless_tv__spinless_tv_L2__static_full_meta_append_adapt_vqe",
        "static_table__spin_boson__spin_boson_L1__static_full_meta_append_adapt_vqe",
        "static_table__spinless_tv__spinless_tv_L2__static_qubit_qeb_adapt_vqe",
        "static_table__hh__hh_L2__static_qubit_qeb_adapt_vqe",
        "static_table__spin_boson__spin_boson_L1__static_tetris_qubit_adapt_vqe",
        "static_table__hh__hh_L2__static_tetris_qubit_adapt_vqe",
        "static_table__hubbard__hubbard_L2__static_geo_adapt_vqe",
        "static_table__hh__hh_L2__static_geo_adapt_vqe",
        "static_table__bose_hubbard__bose_hubbard_L2__static_geo_adapt_vqe",
    )


def _phase3_static_smoke_record_ids_for_profile(profile: str) -> tuple[str, ...]:
    profile_key = table_i_suite_profile(profile)
    if profile_key == TABLE_I_THREE_MODEL_HH_SYMMETRIC_PROFILE:
        return ("static_table__hh__hh_L2_nph2_three_model_sym_weak_weak__static_family_native_adapt_phase3",)
    case_id = "hh_L2_nph2" if profile_key == TABLE_I_NPH2_REF3_PROFILE else "hh_L2"
    return (f"static_table__hh__{case_id}__static_family_native_adapt_phase3",)


def _record_id(job: BenchmarkJob) -> str:
    return f"static_table__{job.family}__{job.case_id}__{job.algorithm_id}"


def _blank_phase3_oracle_fields() -> dict[str, str]:
    return {field: "" for field in PHASE3_ORACLE_TSV_FIELDS}


def _blank_phase3_budget_fields() -> dict[str, str]:
    return {field: "" for field in PHASE3_BUDGET_TSV_FIELDS}


def _blank_phase3_runtime_fields() -> dict[str, str]:
    return {field: "" for field in PHASE3_RUNTIME_TSV_FIELDS}


def _blank_phase3_policy_json_fields() -> dict[str, str]:
    return {field: "" for field in PHASE3_POLICY_JSON_TSV_FIELDS}


def _blank_phase3_pos_geo_fields() -> dict[str, str]:
    return {field: "" for field in PHASE3_POS_GEO_TSV_FIELDS}


def _blank_hardware_resolution_profile_fields() -> dict[str, str]:
    return {field: "" for field in HARDWARE_RESOLUTION_PROFILE_TSV_FIELDS}


def _blank_static_route_fields() -> dict[str, str]:
    return {field: "" for field in STATIC_ROUTE_TSV_FIELDS}


def _blank_static_route_runtime_fields() -> dict[str, str]:
    return {field: "" for field in STATIC_ROUTE_RUNTIME_TSV_FIELDS}


def _blank_generic_adapt_runtime_split_fields() -> dict[str, str]:
    return {field: "" for field in GENERIC_ADAPT_RUNTIME_SPLIT_TSV_FIELDS}


def _blank_shared_pauli_pool_fields() -> dict[str, str]:
    return {field: "" for field in SHARED_PAULI_POOL_TSV_FIELDS}


def _blank_benchmark_value_noise_fields() -> dict[str, str]:
    return {field: "" for field in BENCHMARK_VALUE_NOISE_TSV_FIELDS}


def _blank_resource_guard_fields() -> dict[str, str]:
    return {field: "" for field in RESOURCE_GUARD_TSV_FIELDS}


def _blank_hea_fields() -> dict[str, str]:
    return {field: "" for field in HEA_TSV_FIELDS}


def _blank_optimizer_fields() -> dict[str, str]:
    return {field: "" for field in PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_TSV_FIELDS}


def _optimizer_fields_for_algorithm(
    algorithm_id: str,
    *,
    optimizer_profile: str | None,
    smoke: bool = False,
) -> dict[str, str]:
    fields = _blank_optimizer_fields()
    profile = str(optimizer_profile or "").strip()
    if not profile:
        return fields
    if profile != PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID:
        raise ValueError(f"unsupported optimizer_profile for generic static records: {profile!r}")
    algorithm_key = str(algorithm_id).strip()
    if algorithm_key == PAPER_I_MAIN_TABLES_SPSA_SNAKE_ALGORITHM_ID:
        raise ValueError(
            f"optimizer_profile={profile} excludes SNAKE from generic comparator records; "
            "use the dedicated Route-A/SNAKE generator."
        )
    if algorithm_key == "static_pos_geo_adapt_vqe":
        raise ValueError(
            f"optimizer_profile={profile} excludes static_pos_geo_adapt_vqe; visible Geo-ADAPT is static_geo_adapt_vqe."
        )
    if algorithm_key not in PAPER_I_MAIN_TABLES_SPSA_COMPARATOR_ALGORITHM_IDS:
        raise ValueError(
            f"optimizer_profile={profile} supports only displayed non-SNAKE comparator algorithms; "
            f"got {algorithm_key!r}."
        )
    budgets = PAPER_I_MAIN_TABLES_SPSA_SMOKE_BUDGET_DEFAULTS if bool(smoke) else PAPER_I_MAIN_TABLES_SPSA_BUDGET_DEFAULTS
    fields["optimizer_profile"] = profile
    if algorithm_key == "static_hea_qiskit_vqe":
        defaults = budgets["hea"]
        fields["hea_optimizer"] = str(defaults["optimizer"])
        fields["hea_spsa_maxiter"] = str(int(defaults["spsa_maxiter"]))
        fields["hea_spsa_seed"] = str(int(defaults["spsa_seed"]))
    elif algorithm_key == "static_family_informed_vqe":
        defaults = budgets["family_informed"]
        fields["family_informed_optimizer"] = str(defaults["optimizer"])
        fields["family_informed_spsa_maxiter"] = str(int(defaults["spsa_maxiter"]))
        fields["family_informed_spsa_seed"] = str(int(defaults["spsa_seed"]))
    else:
        defaults = budgets["adapt"]
        fields["adapt_optimizer_kind"] = str(defaults["optimizer_kind"])
        fields["adapt_spsa_maxiter"] = str(int(defaults["spsa_maxiter"]))
        fields["adapt_spsa_seed"] = str(int(defaults["spsa_seed"]))
    return fields


def _apply_optimizer_profile_fields(
    record: Mapping[str, str],
    *,
    optimizer_profile: str | None,
    smoke: bool = False,
) -> dict[str, str]:
    row = dict(record)
    row.update(
        _optimizer_fields_for_algorithm(
            str(row.get("algorithm_id") or ""),
            optimizer_profile=optimizer_profile,
            smoke=smoke,
        )
    )
    return row


def _blank_benchmark_decision_noise_fields() -> dict[str, str]:
    return {field: "" for field in BENCHMARK_DECISION_NOISE_TSV_FIELDS}


def _blank_phonon_cutoff_fields() -> dict[str, str]:
    return {field: "" for field in PHONON_CUTOFF_TSV_FIELDS}


def _pipeline_arg_value(args: Sequence[str], flag: str) -> str | None:
    tokens = [str(x) for x in args]
    for idx, token in enumerate(tokens):
        if token == flag and idx + 1 < len(tokens):
            return tokens[idx + 1]
        prefix = f"{flag}="
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def _phonon_cutoff_fields_for_job(job: BenchmarkJob, *, suite_profile: str) -> dict[str, str]:
    fields = _blank_phonon_cutoff_fields()
    try:
        spec = table_i_canonical_spec_by_case_id(str(job.family), str(job.case_id), suite_profile)
    except Exception:
        return fields
    if not bool(getattr(getattr(spec, "features", None), "bosonic", False)):
        return fields
    work = _pipeline_arg_value(tuple(str(x) for x in getattr(spec, "base_pipeline_args", ())), "--n-ph-max")
    ref = getattr(spec, "exact_reference_n_ph_max", None)
    if work not in {None, ""}:
        fields["n_ph_work"] = str(int(float(str(work))))
    if ref not in {None, ""}:
        fields["n_ph_ref"] = str(int(float(str(ref))))
    if fields["n_ph_work"] and fields["n_ph_ref"]:
        fields["primary_energy_metric"] = "higher_cutoff_reference_abs_delta_e"
        fields["same_cutoff_error_role"] = "diagnostic_only"
    elif fields["n_ph_work"]:
        fields["primary_energy_metric"] = "same_cutoff_abs_delta_e_diagnostic_only"
        fields["same_cutoff_error_role"] = "primary_fallback_no_reference_cutoff"
    return fields


def _normalize_ladder_stage(value: str | None) -> tuple[str, dict[str, Any]]:
    stage = str(value or "off").strip().lower().replace("-", "_")
    if stage not in _PAPER_I_LADDER_STAGE_CONFIGS:
        raise ValueError(
            f"paper_i_cutoff_ladder_stage must be one of {sorted(_PAPER_I_LADDER_STAGE_CONFIGS)}."
        )
    return stage, dict(_PAPER_I_LADDER_STAGE_CONFIGS[stage])


def _apply_ladder_metadata(
    records: Sequence[dict[str, str]],
    *,
    stage: str,
    config: Mapping[str, Any],
    case_ids: Sequence[str],
    allow_ref5: bool,
    escalation_reason: str,
    snake_policy: str,
) -> list[dict[str, str]]:
    if not bool(config.get("enabled")):
        return [dict(record) for record in records]
    requested_case_ids = {str(case_id).strip() for case_id in case_ids if str(case_id).strip()}
    if bool(config.get("requires_prior_failure")) and not requested_case_ids:
        raise ValueError(f"paper_i_cutoff_ladder_stage={stage} requires explicit --paper-i-ladder-case-id values.")
    if bool(config.get("requires_ref5_allowance")) and not bool(allow_ref5):
        raise ValueError(f"paper_i_cutoff_ladder_stage={stage} requires --paper-i-ladder-allow-ref5.")
    if bool(config.get("requires_prior_failure")) and not str(escalation_reason or "").strip():
        raise ValueError(f"paper_i_cutoff_ladder_stage={stage} requires --paper-i-ladder-escalation-reason.")
    out: list[dict[str, str]] = []
    available_case_ids = {
        str(record.get("case_id") or "")
        for record in records
        if str(record.get("n_ph_work") or "").strip()
        and str(record.get("family") or "") in PAPER_I_PHONON_FAMILIES
    }
    for record in records:
        if str(record.get("family") or "") not in PAPER_I_PHONON_FAMILIES:
            continue
        if requested_case_ids and str(record.get("case_id") or "") not in requested_case_ids:
            continue
        if not str(record.get("n_ph_work") or "").strip():
            continue
        row = dict(record)
        row["paper_i_cutoff_ladder_stage"] = stage
        row["paper_i_ladder_acceptance_threshold"] = str(float(_PAPER_I_CLEAN_TAU_TIGHT))
        row["paper_i_ladder_requires_prior_failure"] = "true" if bool(config.get("requires_prior_failure")) else "false"
        row["paper_i_ladder_escalation_reason"] = str(escalation_reason or "")
        row["paper_i_ladder_allow_ref5"] = "true" if bool(allow_ref5) else "false"
        row["paper_i_ladder_snake_policy"] = str(snake_policy or "")
        row["tau_phys"] = str(float(_PAPER_I_CLEAN_TAU_PHYS))
        row["tau_tight"] = str(float(_PAPER_I_CLEAN_TAU_TIGHT))
        row["primary_energy_metric"] = "higher_cutoff_reference_abs_delta_e"
        row["same_cutoff_error_role"] = "diagnostic_only"
        out.append(row)
    if not out:
        raise ValueError(f"paper_i_cutoff_ladder_stage={stage} generated no phonon records.")
    emitted_case_ids = {str(row.get("case_id") or "") for row in out}
    missing_requested = sorted(requested_case_ids - emitted_case_ids)
    if missing_requested:
        raise ValueError(
            f"paper_i_cutoff_ladder_stage={stage} missing requested case_id(s): {missing_requested}; "
            f"available phonon case_ids: {sorted(available_case_ids)}"
        )
    return out


def _apply_reference_energy_metadata(
    records: Sequence[dict[str, str]],
    *,
    suite_profile: str,
    output_dir: Path,
) -> list[dict[str, str]]:
    if not records:
        return []
    specs_by_case = {
        (str(record.get("family") or ""), str(record.get("case_id") or "")): table_i_canonical_spec_by_case_id(
            str(record.get("family") or ""),
            str(record.get("case_id") or ""),
            suite_profile,
        )
        for record in records
        if str(record.get("n_ph_work") or "").strip()
    }
    cache_path = Path(output_dir) / "generic_static_reference_energy_cache.json"
    materialize_reference_energy_cache(tuple(specs_by_case.values()), output_json=cache_path)
    cache_rel = str(cache_path.relative_to(REPO_ROOT)) if cache_path.is_absolute() and cache_path.is_relative_to(REPO_ROOT) else str(cache_path)
    out: list[dict[str, str]] = []
    for record in records:
        row = dict(record)
        if not str(row.get("n_ph_work") or "").strip():
            out.append(row)
            continue
        spec = specs_by_case[(str(row.get("family") or ""), str(row.get("case_id") or ""))]
        work_nph = int(row["n_ph_work"])
        ref_nph = int(row["n_ph_ref"]) if str(row.get("n_ph_ref") or "").strip() else None
        same_energy, same_key, _same_payload = exact_energy_for_spec(spec, n_ph_max=work_nph)
        row["reference_energy_cache_json"] = cache_rel
        row["same_cutoff_reference_energy_key"] = same_key
        row["same_cutoff_exact_gs_energy"] = repr(float(same_energy))
        row["reference_energy_status"] = "same_cutoff_only"
        if ref_nph is not None:
            ref_energy, ref_key, _ref_payload = exact_energy_for_spec(spec, n_ph_max=ref_nph)
            row["reference_cutoff_energy_key"] = ref_key
            row["exact_reference_energy"] = repr(float(ref_energy))
            row["exact_reference_n_ph_max"] = str(int(ref_nph))
            row["reference_energy_status"] = "ok"
        out.append(row)
    return out


def _phase3_budget_overlay_requested(fields: Mapping[str, str] | dict[str, str]) -> bool:
    return any(str(fields.get(field) or "").strip() != "" for field in PHASE3_BUDGET_TSV_FIELDS)


def _phase3_runtime_overlay_requested(fields: Mapping[str, str] | dict[str, str]) -> bool:
    return any(str(fields.get(field) or "").strip() != "" for field in PHASE3_RUNTIME_TSV_FIELDS)


def _generic_adapt_runtime_split_overlay_requested(fields: Mapping[str, str] | dict[str, str]) -> bool:
    return any(str(fields.get(field) or "").strip() != "" for field in GENERIC_ADAPT_RUNTIME_SPLIT_TSV_FIELDS)


def _shared_pauli_pool_overlay_requested(fields: Mapping[str, str] | dict[str, str]) -> bool:
    return any(str(fields.get(field) or "").strip() != "" for field in SHARED_PAULI_POOL_TSV_FIELDS)


def _phase3_policy_json_overlay_requested(fields: Mapping[str, str] | dict[str, str]) -> bool:
    return any(str(fields.get(field) or "").strip() != "" for field in PHASE3_POLICY_JSON_TSV_FIELDS)


def _hardware_resolution_profile_overlay_requested(fields: Mapping[str, str] | dict[str, str]) -> bool:
    return any(str(fields.get(field) or "").strip() != "" for field in HARDWARE_RESOLUTION_PROFILE_TSV_FIELDS)


def _static_route_id_for_job(
    job: BenchmarkJob,
    *,
    hardware_resolution_profile_requested: bool,
    phase2_novelty_mode: str,
) -> str:
    if str(job.metadata.get("dispatch") or "") != "phase3_static_adapt":
        return ""
    if str(job.algorithm_id) == "static_family_native_adapt_phase3" and not bool(
        hardware_resolution_profile_requested
    ):
        if str(phase2_novelty_mode or "").strip().lower() == "legacy_pairwise_v1":
            return ROUTE_ID_B_LEGACY_PAIRWISE
        return ROUTE_ID_A
    return ROUTE_ID_UNSPECIFIED


def _route_component_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _static_route_component_fields(route_id: str) -> dict[str, str]:
    fields = _blank_static_route_fields()
    route_key = str(route_id or "").strip()
    if route_key == "":
        return fields
    fields["static_route_id"] = route_key
    if route_key == ROUTE_ID_A:
        required = ROUTE_A_REQUIRED_COMPONENTS
        fields["canonical_snake_eligible_expected"] = "true"
        fields["route_evidence_role"] = "canonical_current_route_candidate"
    elif route_key == ROUTE_ID_B_LEGACY_PAIRWISE:
        required = ROUTE_B_REQUIRED_COMPONENTS
        fields["canonical_snake_eligible_expected"] = "false"
        fields["route_evidence_role"] = "legacy_pairwise_control"
    else:
        return fields
    fields["route_base_pool_key"] = _route_component_text(required["base_pool_key"])
    for key, value in required.items():
        if key == "base_pool_key":
            continue
        if key in STATIC_ROUTE_TSV_FIELDS or key in STATIC_ROUTE_RUNTIME_TSV_FIELDS or key == "hardware_resolution_mode":
            fields[key] = _route_component_text(value)
    fields["phase2_enable_batching"] = _route_component_text(required["phase3_enable_batching"])
    return fields


def _benchmark_value_noise_overlay_requested(fields: Mapping[str, str] | dict[str, str]) -> bool:
    if str(fields.get("benchmark_value_noise_model") or "off").strip().lower() != "off":
        return True
    raw_std = fields.get("benchmark_value_noise_std")
    if raw_std not in {None, ""} and float(raw_std) != 0.0:
        return True
    return fields.get("benchmark_value_noise_seed") not in {None, ""}


def _benchmark_decision_noise_overlay_requested(fields: Mapping[str, str] | dict[str, str]) -> bool:
    if str(fields.get("benchmark_decision_noise_model") or "off").strip().lower() != "off":
        return True
    raw_std = fields.get("benchmark_decision_noise_std")
    if raw_std not in {None, ""} and float(raw_std) != 0.0:
        return True
    return fields.get("benchmark_decision_noise_seed") not in {None, ""}


def _phase3_oracle_overlay_requested(fields: Mapping[str, str] | dict[str, str]) -> bool:
    if str(fields.get("phase3_oracle_gradient_mode") or "off").strip().lower() != "off":
        return True
    if str(fields.get("phase3_oracle_inner_objective_mode") or "exact").strip().lower() != "exact":
        return True
    if str(fields.get("phase3_oracle_value_noise_model") or "off").strip().lower() != "off":
        return True
    raw_std = fields.get("phase3_oracle_value_noise_std")
    if raw_std not in {None, ""} and float(raw_std) != 0.0:
        return True
    return any(str(fields.get(field) or "").strip() != "" for field in PHASE3_ORACLE_TSV_FIELDS if field not in {
        "phase3_oracle_gradient_mode",
        "phase3_oracle_inner_objective_mode",
        "phase3_oracle_value_noise_model",
        "phase3_oracle_value_noise_std",
    })


def _positive_int_string(value: Any, *, field: str) -> str:
    if value in {None, ""}:
        return ""
    text = str(value).strip()
    if text.startswith("+"):
        text = text[1:]
    if not text.isdecimal():
        raise ValueError(f"{field} must be a positive integer when provided.")
    parsed = int(text)
    if parsed < 1:
        raise ValueError(f"{field} must be a positive integer when provided.")
    return str(int(parsed))


def _finite_float_string(value: Any, *, field: str) -> str:
    if value in {None, ""}:
        return ""
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite when provided.")
    return str(float(parsed))


def _phase3_budget_int_string(value: Any, *, field: str, min_value: int = 1) -> str:
    if value in {None, ""}:
        return ""
    text = str(value).strip()
    if text.startswith("+"):
        text = text[1:]
    if not text.isdecimal():
        raise ValueError(f"{field} must be an integer >= {int(min_value)} when provided.")
    parsed = int(text)
    if parsed < int(min_value):
        raise ValueError(f"{field} must be an integer >= {int(min_value)} when provided.")
    return str(int(parsed))


def _phase3_budget_bool_string(value: Any, *, field: str) -> str:
    if value in {None, ""}:
        return ""
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return "true"
    if text in {"0", "false", "no", "off"}:
        return "false"
    raise ValueError(f"{field} must be boolean when provided.")


def _normalize_phase3_budget_overlay_fields(fields: Mapping[str, Any]) -> dict[str, str]:
    normalized = _blank_phase3_budget_fields()
    normalized.update({str(key): "" if value is None else str(value) for key, value in dict(fields).items()})
    for field, value in list(normalized.items()):
        if field == "phase3_adapt_allow_repeats":
            normalized[field] = _phase3_budget_bool_string(value, field=field)
        elif field in {
            "phase3_adapt_spsa_a",
            "phase3_adapt_spsa_c",
            "phase3_adapt_spsa_big_a",
            "phase3_adapt_spsa_alpha",
            "phase3_adapt_spsa_gamma",
        }:
            normalized[field] = _finite_float_string(value, field=field)
        else:
            min_value = 0 if field == "phase3_adapt_spsa_avg_last" else 1
            normalized[field] = _phase3_budget_int_string(value, field=field, min_value=min_value)
    return normalized


def _normalize_phase3_smoke_budget_overlay(profile: str | None) -> tuple[str, dict[str, str]]:
    profile_key = str(profile or "off").strip().lower().replace("-", "_")
    if profile_key not in _PHASE3_SMOKE_BUDGET_PROFILE_CHOICES:
        raise ValueError(
            f"phase3_smoke_budget_profile must be one of {sorted(_PHASE3_SMOKE_BUDGET_PROFILE_CHOICES)}."
        )
    if profile_key == "off":
        return profile_key, _blank_phase3_budget_fields()
    return profile_key, _normalize_phase3_budget_overlay_fields(_PHASE3_WEAK_LOCAL_BUDGET_FIELDS)


def _normalize_phase3_policy_overlay(profile: str | None) -> tuple[str, dict[str, str]]:
    profile_key = str(profile or "off").strip().lower().replace("-", "_")
    if profile_key not in _PHASE3_POLICY_PROFILE_CHOICES:
        raise ValueError(f"phase3_policy_profile must be one of {sorted(_PHASE3_POLICY_PROFILE_CHOICES)}.")
    if profile_key == "off":
        return profile_key, _blank_phase3_budget_fields()
    if profile_key == "spsa_prior_depth12_v1":
        return profile_key, _normalize_phase3_budget_overlay_fields(_PHASE3_SPSA_PRIOR_DEPTH12_FIELDS)
    return profile_key, _normalize_phase3_budget_overlay_fields(_PHASE3_SPSA_PRIOR_BEST_FIELDS)


def _normalize_generic_adapt_budget_overlay(profile: str | None) -> tuple[str, dict[str, str]]:
    profile_key = str(profile or "off").strip().lower().replace("-", "_")
    if profile_key not in _GENERIC_ADAPT_BUDGET_PROFILE_CHOICES:
        raise ValueError(
            "generic_adapt_budget_profile must be one of "
            f"{sorted(_GENERIC_ADAPT_BUDGET_PROFILE_CHOICES)}."
        )
    if profile_key == "off":
        return profile_key, _blank_phase3_budget_fields()
    if profile_key == "paper_i_first_hit_depth256_v1":
        return profile_key, _normalize_phase3_budget_overlay_fields(_GENERIC_ADAPT_FIRST_HIT_DEPTH256_FIELDS)
    return profile_key, _normalize_phase3_budget_overlay_fields(_GENERIC_ADAPT_FIRST_HIT_DEPTH500_FIELDS)


def _normalize_generic_adapt_smoke_budget_overlay(profile: str | None) -> tuple[str, dict[str, str]]:
    profile_key = str(profile or "off").strip().lower().replace("-", "_")
    if profile_key not in _GENERIC_ADAPT_SMOKE_BUDGET_PROFILE_CHOICES:
        raise ValueError(
            "generic_adapt_smoke_budget_profile must be one of "
            f"{sorted(_GENERIC_ADAPT_SMOKE_BUDGET_PROFILE_CHOICES)}."
        )
    if profile_key == "off":
        return profile_key, _blank_phase3_budget_fields()
    return profile_key, _normalize_phase3_budget_overlay_fields(_PHASE3_WEAK_LOCAL_BUDGET_FIELDS)


def _normalize_phase3_policy_json_overlay(phase3_policy_json: str | None) -> dict[str, str]:
    fields = _blank_phase3_policy_json_fields()
    value = str(phase3_policy_json or "").strip()
    if value:
        fields["phase3_policy_json"] = value
    return fields


def _normalize_phase2_novelty_mode(value: str | None) -> str:
    mode = str(value or "").strip().lower()
    if mode not in _PHASE2_NOVELTY_MODE_CHOICES:
        raise ValueError(f"phase2_novelty_mode must be one of {sorted(_PHASE2_NOVELTY_MODE_CHOICES)}.")
    return mode


def _normalize_calibration_profile(profile: str | None) -> str:
    profile_key = str(profile or "off").strip().lower().replace("-", "_")
    if profile_key not in _CALIBRATION_PROFILE_CHOICES:
        raise ValueError(f"calibration_profile must be one of {sorted(_CALIBRATION_PROFILE_CHOICES)}.")
    return profile_key


def _filter_records_by_ids(
    records: Sequence[dict[str, str]],
    *,
    record_ids: Sequence[str],
    label: str,
) -> list[dict[str, str]]:
    by_id = {record["record_id"]: record for record in records}
    missing = [record_id for record_id in record_ids if record_id not in by_id]
    if missing:
        raise ValueError(f"{label} record selection missing generated records: {missing}")
    return [dict(by_id[record_id]) for record_id in record_ids]


def _filter_records_by_case_ids(
    records: Sequence[dict[str, str]],
    *,
    case_ids: Sequence[str],
    label: str,
) -> list[dict[str, str]]:
    requested = tuple(str(case_id).strip() for case_id in case_ids if str(case_id).strip())
    if not requested:
        return []
    requested_set = set(requested)
    selected = [dict(record) for record in records if str(record.get("case_id") or "") in requested_set]
    found = {str(record.get("case_id") or "") for record in selected}
    missing = [case_id for case_id in requested if case_id not in found]
    if missing:
        available = sorted({str(record.get("case_id") or "") for record in records})
        raise ValueError(f"{label} case selection missing generated case_id(s): {missing}; available: {available}")
    return selected


def _normalize_phase3_runtime_overlay(
    *,
    phase3_adapt_parallel_gradient_workers: int | str | None = None,
    phase3_adapt_beam_parent_workers: int | str | None = None,
) -> dict[str, str]:
    fields = _blank_phase3_runtime_fields()
    fields["phase3_adapt_parallel_gradient_workers"] = _positive_int_string(
        phase3_adapt_parallel_gradient_workers,
        field="phase3_adapt_parallel_gradient_workers",
    )
    fields["phase3_adapt_beam_parent_workers"] = _positive_int_string(
        phase3_adapt_beam_parent_workers,
        field="phase3_adapt_beam_parent_workers",
    )
    return fields


def _normalize_generic_adapt_runtime_split_overlay(
    *,
    generic_adapt_runtime_split_mode: str | None = None,
    generic_adapt_runtime_split_symmetry_policy: str | None = None,
    generic_adapt_runtime_split_max_subset_size: int | str | None = None,
) -> dict[str, str]:
    fields = _blank_generic_adapt_runtime_split_fields()
    mode = str(generic_adapt_runtime_split_mode or "off").strip().lower()
    if mode in {"", "none", "false", "0", "disabled"}:
        mode = "off"
    if mode not in _GENERIC_ADAPT_RUNTIME_SPLIT_MODE_CHOICES:
        raise ValueError(
            "generic_adapt_runtime_split_mode must be one of "
            f"{sorted(_GENERIC_ADAPT_RUNTIME_SPLIT_MODE_CHOICES)}."
        )
    policy = str(generic_adapt_runtime_split_symmetry_policy or "off").strip().lower().replace("-", "_")
    if policy in {"", "none", "false", "0", "disabled"}:
        policy = "off"
    if policy not in _GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICY_CHOICES:
        raise ValueError(
            "generic_adapt_runtime_split_symmetry_policy must be one of "
            f"{sorted(_GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICY_CHOICES)}."
        )
    if mode == "off":
        if policy != "off":
            raise ValueError("generic_adapt_runtime_split_symmetry_policy requires runtime split mode to be enabled.")
        if generic_adapt_runtime_split_max_subset_size not in {None, ""}:
            raise ValueError("generic_adapt_runtime_split_max_subset_size requires runtime split mode to be enabled.")
        return fields
    fields["generic_adapt_runtime_split_mode"] = mode
    fields["generic_adapt_runtime_split_symmetry_policy"] = policy
    fields["generic_adapt_runtime_split_max_subset_size"] = _positive_int_string(
        3 if generic_adapt_runtime_split_max_subset_size in {None, ""} else generic_adapt_runtime_split_max_subset_size,
        field="generic_adapt_runtime_split_max_subset_size",
    )
    return fields


def _normalize_shared_pauli_pool_overlay(
    *,
    shared_pauli_pool_mode: str | None = None,
    shared_pauli_pool_symmetry_policy: str | None = None,
    shared_pauli_pool_max_subset_size: int | str | None = None,
) -> dict[str, str]:
    fields = _blank_shared_pauli_pool_fields()
    mode = str(shared_pauli_pool_mode or "off").strip().lower().replace("-", "_")
    if mode in {"", "none", "false", "0", "disabled"}:
        mode = "off"
    if mode in {"pauli_child_sets_v1", "global_pauli_child_sets_v1"}:
        mode = "shared_pauli_child_sets_v1"
    if mode not in {"off", "shared_pauli_child_sets_v1"}:
        raise ValueError(
            "shared_pauli_pool_mode must be one of "
            f"{sorted(_SHARED_PAULI_POOL_MODE_CHOICES)}."
        )
    policy = str(shared_pauli_pool_symmetry_policy or "off").strip().lower().replace("-", "_")
    if policy in {"", "none", "false", "0", "disabled"}:
        policy = "off"
    if policy not in _SHARED_PAULI_POOL_SYMMETRY_POLICY_CHOICES:
        raise ValueError(
            "shared_pauli_pool_symmetry_policy must be one of "
            f"{sorted(_SHARED_PAULI_POOL_SYMMETRY_POLICY_CHOICES)}."
        )
    if mode == "off":
        if policy != "off":
            raise ValueError("shared_pauli_pool_symmetry_policy requires shared Pauli pool mode to be enabled.")
        if shared_pauli_pool_max_subset_size not in {None, ""}:
            raise ValueError("shared_pauli_pool_max_subset_size requires shared Pauli pool mode to be enabled.")
        return fields
    fields["shared_pauli_pool_mode"] = mode
    fields["shared_pauli_pool_symmetry_policy"] = policy
    fields["shared_pauli_pool_max_subset_size"] = _positive_int_string(
        3 if shared_pauli_pool_max_subset_size in {None, ""} else shared_pauli_pool_max_subset_size,
        field="shared_pauli_pool_max_subset_size",
    )
    return fields


def _normalize_hardware_resolution_profile_overlay(
    *,
    hardware_resolution_mode: str | None = None,
    hardware_resolution_profile_json: str | None = None,
    hardware_resolution_profile_name: str | None = None,
) -> dict[str, str]:
    fields = _blank_hardware_resolution_profile_fields()
    mode = str(hardware_resolution_mode or "").strip().lower()
    profile_json = str(hardware_resolution_profile_json or "").strip()
    profile_name = str(hardware_resolution_profile_name or "").strip()
    if mode == "" and profile_json == "" and profile_name == "":
        return fields
    if mode not in {"", "profile"}:
        raise ValueError("hardware_resolution_mode overlay must be 'profile' when provided.")
    if not profile_json or not profile_name:
        raise ValueError(
            "hardware_resolution profile overlay requires hardware_resolution_profile_json "
            "and hardware_resolution_profile_name together."
        )
    fields["hardware_resolution_mode"] = "profile"
    fields["hardware_resolution_profile_json"] = profile_json
    fields["hardware_resolution_profile_name"] = profile_name
    return fields


def _normalize_phase3_oracle_overlay(
    *,
    phase3_oracle_gradient_mode: str | None = None,
    phase3_oracle_backend_name: str | None = None,
    phase3_oracle_use_fake_backend: bool | None = None,
    phase3_oracle_shots: int | None = None,
    phase3_oracle_repeats: int | None = None,
    phase3_oracle_aggregate: str | None = None,
    phase3_oracle_seed: int | None = None,
    phase3_oracle_execution_surface: str | None = None,
    phase3_oracle_inner_objective_mode: str | None = None,
    phase3_oracle_value_noise_model: str | None = None,
    phase3_oracle_value_noise_std: float | None = None,
    phase3_oracle_value_noise_seed: int | None = None,
) -> dict[str, str]:
    fields = _blank_phase3_oracle_fields()
    mode = str(phase3_oracle_gradient_mode or "off").strip().lower()
    if mode not in _PHASE3_ORACLE_GRADIENT_MODE_CHOICES:
        raise ValueError(f"phase3_oracle_gradient_mode must be one of {sorted(_PHASE3_ORACLE_GRADIENT_MODE_CHOICES)}.")
    if phase3_oracle_gradient_mode not in {None, ""}:
        fields["phase3_oracle_gradient_mode"] = mode
    if phase3_oracle_backend_name not in {None, ""}:
        fields["phase3_oracle_backend_name"] = str(phase3_oracle_backend_name).strip()
    if phase3_oracle_use_fake_backend is not None:
        fields["phase3_oracle_use_fake_backend"] = "true" if bool(phase3_oracle_use_fake_backend) else "false"
    fields["phase3_oracle_shots"] = _positive_int_string(phase3_oracle_shots, field="phase3_oracle_shots")
    fields["phase3_oracle_repeats"] = _positive_int_string(phase3_oracle_repeats, field="phase3_oracle_repeats")
    aggregate = str(phase3_oracle_aggregate or "mean").strip().lower()
    if aggregate != "mean":
        raise ValueError("phase3_oracle_aggregate currently supports only 'mean'.")
    if phase3_oracle_aggregate not in {None, ""}:
        fields["phase3_oracle_aggregate"] = aggregate
    fields["phase3_oracle_seed"] = _positive_int_string(phase3_oracle_seed, field="phase3_oracle_seed")
    execution_surface = str(phase3_oracle_execution_surface or "auto").strip().lower()
    if execution_surface not in _PHASE3_ORACLE_EXECUTION_SURFACE_CHOICES:
        raise ValueError(
            f"phase3_oracle_execution_surface must be one of {sorted(_PHASE3_ORACLE_EXECUTION_SURFACE_CHOICES)}."
        )
    if phase3_oracle_execution_surface not in {None, ""}:
        fields["phase3_oracle_execution_surface"] = execution_surface
    inner_mode = str(phase3_oracle_inner_objective_mode or "exact").strip().lower()
    if inner_mode not in {"exact", "noisy_v1"}:
        raise ValueError("phase3_oracle_inner_objective_mode must be one of {'exact','noisy_v1'}.")
    if phase3_oracle_inner_objective_mode not in {None, ""}:
        fields["phase3_oracle_inner_objective_mode"] = inner_mode
    value_noise_model = str(phase3_oracle_value_noise_model or "off").strip().lower()
    if value_noise_model not in _PHASE3_ORACLE_VALUE_NOISE_MODEL_CHOICES:
        raise ValueError(
            f"phase3_oracle_value_noise_model must be one of {sorted(_PHASE3_ORACLE_VALUE_NOISE_MODEL_CHOICES)}."
        )
    if phase3_oracle_value_noise_model not in {None, ""}:
        fields["phase3_oracle_value_noise_model"] = value_noise_model
    fields["phase3_oracle_value_noise_std"] = _finite_float_string(
        phase3_oracle_value_noise_std,
        field="phase3_oracle_value_noise_std",
    )
    fields["phase3_oracle_value_noise_seed"] = _positive_int_string(
        phase3_oracle_value_noise_seed,
        field="phase3_oracle_value_noise_seed",
    )
    std = float(fields["phase3_oracle_value_noise_std"] or 0.0)
    if value_noise_model == "off":
        if fields["phase3_oracle_value_noise_seed"]:
            raise ValueError(
                "phase3_oracle_value_noise_seed requires phase3_oracle_value_noise_model='gaussian_iid_v1'."
            )
        if std != 0.0:
            raise ValueError("phase3_oracle_value_noise_model='off' requires phase3_oracle_value_noise_std == 0.")
    elif value_noise_model == "gaussian_iid_v1":
        if (not math.isfinite(std)) or std <= 0.0:
            raise ValueError(
                "phase3_oracle_value_noise_model='gaussian_iid_v1' requires finite phase3_oracle_value_noise_std > 0."
            )
        if mode == "off":
            raise ValueError("phase3_oracle_value_noise_model='gaussian_iid_v1' requires phase3_oracle_gradient_mode != 'off'.")
        if execution_surface == "raw_measurement_v1":
            raise ValueError("phase3 oracle value noise is post-expectation metadata and cannot use raw_measurement_v1.")
    if inner_mode == "noisy_v1" and mode == "off":
        raise ValueError("phase3_oracle_inner_objective_mode='noisy_v1' requires phase3_oracle_gradient_mode != 'off'.")
    return fields


def _normalize_benchmark_value_noise_overlay(
    *,
    benchmark_value_noise_model: str | None = None,
    benchmark_value_noise_std: float | None = None,
    benchmark_value_noise_seed: int | None = None,
) -> dict[str, str]:
    fields = _blank_benchmark_value_noise_fields()
    model = str(benchmark_value_noise_model or "off").strip().lower()
    if model not in _BENCHMARK_VALUE_NOISE_MODEL_CHOICES:
        raise ValueError(
            f"benchmark_value_noise_model must be one of {sorted(_BENCHMARK_VALUE_NOISE_MODEL_CHOICES)}."
        )
    if benchmark_value_noise_model not in {None, ""}:
        fields["benchmark_value_noise_model"] = model
    fields["benchmark_value_noise_std"] = _finite_float_string(
        benchmark_value_noise_std,
        field="benchmark_value_noise_std",
    )
    if benchmark_value_noise_seed not in {None, ""}:
        fields["benchmark_value_noise_seed"] = str(int(benchmark_value_noise_seed))
    std = float(fields["benchmark_value_noise_std"] or 0.0)
    if model == "off":
        if fields["benchmark_value_noise_seed"]:
            raise ValueError(
                "benchmark_value_noise_seed requires benchmark_value_noise_model='gaussian_iid_v1'."
            )
        if std != 0.0:
            raise ValueError("benchmark_value_noise_model='off' requires benchmark_value_noise_std == 0.")
    elif model == "gaussian_iid_v1":
        if (not math.isfinite(std)) or std <= 0.0:
            raise ValueError(
                "benchmark_value_noise_model='gaussian_iid_v1' requires finite benchmark_value_noise_std > 0."
            )
    return fields


def _normalize_benchmark_decision_noise_overlay(
    *,
    benchmark_decision_noise_model: str | None = None,
    benchmark_decision_noise_std: float | None = None,
    benchmark_decision_noise_seed: int | None = None,
) -> dict[str, str]:
    fields = _blank_benchmark_decision_noise_fields()
    model = str(benchmark_decision_noise_model or "off").strip().lower()
    if model not in _BENCHMARK_DECISION_NOISE_MODEL_CHOICES:
        raise ValueError(
            "benchmark_decision_noise_model must be one of "
            f"{sorted(_BENCHMARK_DECISION_NOISE_MODEL_CHOICES)}."
        )
    if benchmark_decision_noise_model not in {None, ""}:
        fields["benchmark_decision_noise_model"] = model
    fields["benchmark_decision_noise_std"] = _finite_float_string(
        benchmark_decision_noise_std,
        field="benchmark_decision_noise_std",
    )
    if benchmark_decision_noise_seed not in {None, ""}:
        fields["benchmark_decision_noise_seed"] = str(int(benchmark_decision_noise_seed))
    std = float(fields["benchmark_decision_noise_std"] or 0.0)
    if model == "off":
        if fields["benchmark_decision_noise_seed"]:
            raise ValueError(
                "benchmark_decision_noise_seed requires benchmark_decision_noise_model='gaussian_iid_v1'."
            )
        if std != 0.0:
            raise ValueError("benchmark_decision_noise_model='off' requires benchmark_decision_noise_std == 0.")
    elif model == "gaussian_iid_v1":
        if (not math.isfinite(std)) or std <= 0.0:
            raise ValueError(
                "benchmark_decision_noise_model='gaussian_iid_v1' requires finite "
                "benchmark_decision_noise_std > 0."
            )
    return fields


def _records_from_jobs(
    jobs: Sequence[BenchmarkJob],
    *,
    suite_profile: str,
    energy_stop_target: float,
    first_hit_thresholds: Sequence[float],
    phase3_oracle_overlay: Mapping[str, str] | None = None,
    phase3_policy_overlay: Mapping[str, str] | None = None,
    phase3_policy_json_overlay: Mapping[str, str] | None = None,
    phase3_runtime_overlay: Mapping[str, str] | None = None,
    generic_adapt_runtime_split_overlay: Mapping[str, str] | None = None,
    shared_pauli_pool_overlay: Mapping[str, str] | None = None,
    generic_adapt_budget_overlay: Mapping[str, str] | None = None,
    phase2_novelty_mode: str = "",
    hardware_resolution_profile_overlay: Mapping[str, str] | None = None,
    benchmark_value_noise_overlay: Mapping[str, str] | None = None,
    benchmark_decision_noise_overlay: Mapping[str, str] | None = None,
    hh_pos_geo_position_policy: str = "",
    disable_resource_guards: bool = False,
    hea_reps: int | None = None,
    hea_maxiter: int | None = None,
    optimizer_profile: str | None = None,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    thresholds = ",".join(str(float(x)) for x in _unique_float_thresholds(first_hit_thresholds))
    overlay = dict(phase3_oracle_overlay or _blank_phase3_oracle_fields())
    overlay_requested = _phase3_oracle_overlay_requested(overlay)
    phase3_policy = dict(phase3_policy_overlay or _blank_phase3_budget_fields())
    phase3_policy_requested = _phase3_budget_overlay_requested(phase3_policy)
    generic_adapt_budget = dict(generic_adapt_budget_overlay or _blank_phase3_budget_fields())
    generic_adapt_budget_requested = _phase3_budget_overlay_requested(generic_adapt_budget)
    phase3_policy_json = dict(phase3_policy_json_overlay or _blank_phase3_policy_json_fields())
    phase3_policy_json_requested = _phase3_policy_json_overlay_requested(phase3_policy_json)
    runtime_overlay = dict(phase3_runtime_overlay or _blank_phase3_runtime_fields())
    runtime_overlay_requested = _phase3_runtime_overlay_requested(runtime_overlay)
    runtime_split_overlay = dict(generic_adapt_runtime_split_overlay or _blank_generic_adapt_runtime_split_fields())
    runtime_split_overlay_requested = _generic_adapt_runtime_split_overlay_requested(runtime_split_overlay)
    shared_pool_overlay = dict(shared_pauli_pool_overlay or _blank_shared_pauli_pool_fields())
    shared_pool_overlay_requested = _shared_pauli_pool_overlay_requested(shared_pool_overlay)
    phase2_mode = _normalize_phase2_novelty_mode(phase2_novelty_mode) or "collective_span_v1"
    hardware_overlay = dict(hardware_resolution_profile_overlay or _blank_hardware_resolution_profile_fields())
    hardware_overlay_requested = _hardware_resolution_profile_overlay_requested(hardware_overlay)
    benchmark_overlay = dict(benchmark_value_noise_overlay or _blank_benchmark_value_noise_fields())
    benchmark_overlay_requested = _benchmark_value_noise_overlay_requested(benchmark_overlay)
    benchmark_decision_overlay = dict(benchmark_decision_noise_overlay or _blank_benchmark_decision_noise_fields())
    benchmark_decision_overlay_requested = _benchmark_decision_noise_overlay_requested(benchmark_decision_overlay)
    hh_pos_geo_position_policy_key = str(hh_pos_geo_position_policy or "").strip().lower()
    if hh_pos_geo_position_policy_key not in {"", "append", "best_insert_refit"}:
        raise ValueError("hh_pos_geo_position_policy must be blank, append, or best_insert_refit.")
    for job in jobs:
        if job.status != "runnable":
            continue
        row = {
            "record_id": _record_id(job),
            "family": str(job.family),
            "case_id": str(job.case_id),
            "algorithm_id": str(job.algorithm_id),
            "suite_profile": str(suite_profile),
            "energy_stop_target": str(float(energy_stop_target)),
            "first_hit_thresholds": thresholds,
            **_blank_phase3_oracle_fields(),
            **_blank_phase3_budget_fields(),
            **_blank_phase3_policy_json_fields(),
            **_blank_phase3_pos_geo_fields(),
            **_blank_phase3_runtime_fields(),
            **_blank_hardware_resolution_profile_fields(),
            **_blank_static_route_fields(),
            **_blank_static_route_runtime_fields(),
            **_blank_generic_adapt_runtime_split_fields(),
            **_blank_shared_pauli_pool_fields(),
            **_blank_resource_guard_fields(),
            **_blank_hea_fields(),
            **_blank_optimizer_fields(),
            **_blank_benchmark_value_noise_fields(),
            **_blank_benchmark_decision_noise_fields(),
            **_phonon_cutoff_fields_for_job(job, suite_profile=suite_profile),
        }
        if bool(disable_resource_guards):
            row["resource_qubit_cap"] = "0"
            row["resource_pool_term_cap"] = "0"
        if str(job.algorithm_id) == "static_hea_qiskit_vqe":
            if hea_reps is not None:
                row["hea_reps"] = str(int(hea_reps))
            if hea_maxiter is not None:
                row["hea_maxiter"] = str(int(hea_maxiter))
        row.update(
            _optimizer_fields_for_algorithm(
                str(job.algorithm_id),
                optimizer_profile=optimizer_profile,
                smoke=False,
            )
        )
        if overlay_requested and str(job.metadata.get("dispatch") or "") == "phase3_static_adapt":
            row.update(overlay)
        if (
            phase3_policy_requested
            and str(job.metadata.get("dispatch") or "") == "phase3_static_adapt"
            and str(job.algorithm_id) == "static_family_native_adapt_phase3"
        ):
            row.update(phase3_policy)
        if generic_adapt_budget_requested and str(job.metadata.get("dispatch") or "") == "generic_static_adapt_variants":
            row.update(generic_adapt_budget)
        if (
            runtime_split_overlay_requested
            and str(job.metadata.get("dispatch") or "") == "generic_static_adapt_variants"
            and str(job.family) in _GENERIC_ADAPT_RUNTIME_SPLIT_SUPPORTED_FAMILIES
            and str(job.algorithm_id) in _GENERIC_ADAPT_RUNTIME_SPLIT_SUPPORTED_ALGORITHM_IDS
        ):
            row.update(runtime_split_overlay)
        if (
            shared_pool_overlay_requested
            and str(job.family) in _SHARED_PAULI_POOL_SUPPORTED_FAMILIES
            and str(job.algorithm_id) in _SHARED_PAULI_POOL_SUPPORTED_ALGORITHM_IDS
            and str(job.metadata.get("dispatch") or "") in {"phase3_static_adapt", "generic_static_adapt_variants"}
        ):
            row.update(shared_pool_overlay)
        if (
            hh_pos_geo_position_policy_key
            and str(job.family) == "hh"
            and str(job.algorithm_id) == "static_pos_geo_adapt_vqe"
        ):
            row["phase3_pos_geo_position_policy"] = hh_pos_geo_position_policy_key
        if (
            phase3_policy_json_requested
            and str(job.metadata.get("dispatch") or "") == "phase3_static_adapt"
            and str(job.algorithm_id) == "static_family_native_adapt_phase3"
        ):
            row.update(phase3_policy_json)
        if (
            runtime_overlay_requested
            and str(job.metadata.get("dispatch") or "") == "phase3_static_adapt"
            and str(job.algorithm_id) == "static_family_native_adapt_phase3"
        ):
            row.update(runtime_overlay)
        if str(job.metadata.get("dispatch") or "") == "phase3_static_adapt":
            row["phase2_novelty_mode"] = phase2_mode
        route_id = _static_route_id_for_job(
            job,
            hardware_resolution_profile_requested=bool(hardware_overlay_requested),
            phase2_novelty_mode=phase2_mode,
        )
        row.update(_static_route_component_fields(route_id))
        if hardware_overlay_requested and str(job.metadata.get("dispatch") or "") == "phase3_static_adapt":
            row.update(hardware_overlay)
        if benchmark_overlay_requested:
            row.update(benchmark_overlay)
        if benchmark_decision_overlay_requested and str(job.algorithm_id) not in _PHASE3_STATIC_ADAPT_ALGORITHM_IDS:
            row.update(benchmark_decision_overlay)
        records.append(row)
    records.sort(key=lambda row: (row["algorithm_id"], row["family"], row["case_id"], row["record_id"]))
    return records


def _write_records(path: Path, records: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "record_id",
                "family",
                "case_id",
                "algorithm_id",
                "suite_profile",
                "energy_stop_target",
                "first_hit_thresholds",
                *PHASE3_ORACLE_TSV_FIELDS,
                *PHASE3_BUDGET_TSV_FIELDS,
                *PHASE3_POLICY_JSON_TSV_FIELDS,
                *PHASE3_POS_GEO_TSV_FIELDS,
                *PHASE3_RUNTIME_TSV_FIELDS,
                *HARDWARE_RESOLUTION_PROFILE_TSV_FIELDS,
                *STATIC_ROUTE_TSV_FIELDS,
                *STATIC_ROUTE_RUNTIME_TSV_FIELDS,
                *SELECTED_LOGICAL_TSV_FIELDS,
                *GENERIC_ADAPT_RUNTIME_SPLIT_TSV_FIELDS,
                *SHARED_PAULI_POOL_TSV_FIELDS,
                *RESOURCE_GUARD_TSV_FIELDS,
                *HEA_TSV_FIELDS,
                *PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_TSV_FIELDS,
                *BENCHMARK_VALUE_NOISE_TSV_FIELDS,
                *BENCHMARK_DECISION_NOISE_TSV_FIELDS,
                *PHONON_CUTOFF_TSV_FIELDS,
            ),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(records)


def _write_record_ids(path: Path, records: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{record['record_id']}\n" for record in records), encoding="utf-8")


def _select_smoke_records(
    records: Sequence[dict[str, str]],
    *,
    suite_profile: str,
    include_phase3_static: bool = False,
    phase3_budget_overlay: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, str]], int]:
    smoke_record_ids = _smoke_record_ids_for_profile(suite_profile)
    if bool(include_phase3_static):
        smoke_record_ids = (*smoke_record_ids, *_phase3_static_smoke_record_ids_for_profile(suite_profile))
    budget_overlay = dict(phase3_budget_overlay or _blank_phase3_budget_fields())
    budget_requested = _phase3_budget_overlay_requested(budget_overlay)
    by_id = {record["record_id"]: record for record in records}
    selected: list[dict[str, str]] = []
    phase3_budget_applied_count = 0
    for record_id in smoke_record_ids:
        if record_id not in by_id:
            continue
        row = dict(by_id[record_id])
        if budget_requested and str(row.get("algorithm_id") or "") in _PHASE3_STATIC_ADAPT_ALGORITHM_IDS:
            row.update(budget_overlay)
            phase3_budget_applied_count += 1
        selected.append(row)
    if len(selected) != len(smoke_record_ids):
        missing = [record_id for record_id in smoke_record_ids if record_id not in by_id]
        raise ValueError(f"Smoke record selection missing generated records: {missing}")
    return selected, phase3_budget_applied_count


def _status_by_algorithm(jobs: Sequence[BenchmarkJob]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for job in jobs:
        status_counts = out.setdefault(str(job.algorithm_id), {})
        status_counts[str(job.status)] = status_counts.get(str(job.status), 0) + 1
    return out


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return str(value)


def _portable_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def generate_records(
    *,
    output_dir: Path,
    include_snake: bool = False,
    snake_only: bool = False,
    algorithm_filter: Sequence[str] | None = None,
    queue_output_root: Path = DEFAULT_QUEUE_OUTPUT_ROOT,
    suite_profile: str = TABLE_I_STANDARD_PROFILE,
    family_filter: Sequence[str] | None = None,
    paper_i_ladder_candidate_manifest: Path | None = None,
    energy_stop_target: float = DEFAULT_ENERGY_STOP_TARGET,
    first_hit_thresholds: Sequence[float] = DEFAULT_FIRST_HIT_THRESHOLDS,
    phase3_oracle_gradient_mode: str | None = None,
    phase3_oracle_backend_name: str | None = None,
    phase3_oracle_use_fake_backend: bool | None = None,
    phase3_oracle_shots: int | None = None,
    phase3_oracle_repeats: int | None = None,
    phase3_oracle_aggregate: str | None = None,
    phase3_oracle_seed: int | None = None,
    phase3_oracle_execution_surface: str | None = None,
    phase3_oracle_inner_objective_mode: str | None = None,
    phase3_oracle_value_noise_model: str | None = None,
    phase3_oracle_value_noise_std: float | None = None,
    phase3_oracle_value_noise_seed: int | None = None,
    hardware_resolution_mode: str | None = None,
    hardware_resolution_profile_json: str | None = None,
    hardware_resolution_profile_name: str | None = None,
    benchmark_value_noise_model: str | None = None,
    benchmark_value_noise_std: float | None = None,
    benchmark_value_noise_seed: int | None = None,
    benchmark_decision_noise_model: str | None = None,
    benchmark_decision_noise_std: float | None = None,
    benchmark_decision_noise_seed: int | None = None,
    phase3_smoke_budget_profile: str = "off",
    phase3_policy_profile: str = "off",
    generic_adapt_budget_profile: str = "off",
    generic_adapt_smoke_budget_profile: str = "off",
    phase3_policy_json: str | None = None,
    phase2_novelty_mode: str | None = None,
    phase3_adapt_parallel_gradient_workers: int | str | None = None,
    phase3_adapt_beam_parent_workers: int | str | None = None,
    generic_adapt_runtime_split_mode: str | None = "off",
    generic_adapt_runtime_split_symmetry_policy: str | None = "off",
    generic_adapt_runtime_split_max_subset_size: int | str | None = None,
    shared_pauli_pool_mode: str | None = "off",
    shared_pauli_pool_symmetry_policy: str | None = "off",
    shared_pauli_pool_max_subset_size: int | str | None = None,
    hh_pos_geo_position_policy: str = "",
    disable_resource_guards: bool = False,
    hea_reps: int | None = None,
    hea_maxiter: int | None = None,
    calibration_profile: str = "off",
    paper_i_cutoff_ladder_stage: str = "off",
    paper_i_ladder_case_ids: Sequence[str] | None = None,
    paper_i_ladder_allow_ref5: bool = False,
    paper_i_ladder_escalation_reason: str = "",
    paper_i_ladder_benchmarks_only: bool = False,
    selected_logical_route: str = "standard",
    selected_logical_source_json: str = "",
    selected_logical_transfer_mode: str = "exact_match_v1",
    selected_logical_supported_algorithms_only: bool = False,
    smoke_case_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    calibration_profile_key = _normalize_calibration_profile(calibration_profile)
    ladder_stage, ladder_config = _normalize_ladder_stage(paper_i_cutoff_ladder_stage)
    requested_snake = bool(include_snake or snake_only)
    ladder_candidate_manifest_path = None if paper_i_ladder_candidate_manifest is None else Path(paper_i_ladder_candidate_manifest)
    ladder_candidate_records_by_id: dict[str, dict[str, Any]] = {}
    ladder_candidate_case_ids: tuple[str, ...] = ()
    ladder_candidate_summary: dict[str, Any] | None = None
    ladder_snake_policy = "not_applicable"
    suite_profile_key = table_i_suite_profile(
        TABLE_I_NPH2_REF3_PROFILE
        if calibration_profile_key == "nph2_route_a_hk_hh_v1"
        else ladder_config["suite_profile"]
        if bool(ladder_config.get("enabled"))
        else suite_profile
    )
    optimizer_profile_key = (
        PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID
        if suite_profile_key == TABLE_I_PAPER_I_MAIN_TABLES_SPSA_PROFILE
        else ""
    )
    if optimizer_profile_key:
        if calibration_profile_key != "off":
            raise ValueError(f"optimizer_profile={optimizer_profile_key} is incompatible with calibration_profile.")
        if bool(ladder_config.get("enabled")):
            raise ValueError(f"optimizer_profile={optimizer_profile_key} is incompatible with cutoff ladder stages.")
        if requested_snake:
            raise ValueError(
                f"optimizer_profile={optimizer_profile_key} excludes SNAKE from generic comparator records; "
                "use the dedicated Route-A/SNAKE generator."
            )
        if str(hh_pos_geo_position_policy or "").strip():
            raise ValueError(
                f"optimizer_profile={optimizer_profile_key} excludes PosGeo controls; "
                "visible Geo-ADAPT is static_geo_adapt_vqe."
            )
        if not math.isclose(float(energy_stop_target), float(DEFAULT_ENERGY_STOP_TARGET), rel_tol=0.0, abs_tol=1e-15) and not math.isclose(
            float(energy_stop_target), float(PAPER_I_MAIN_TABLES_SPSA_TARGET), rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError(f"optimizer_profile={optimizer_profile_key} requires energy_stop_target=2e-4.")
        thresholds_key = _unique_float_thresholds(first_hit_thresholds)
        if thresholds_key != DEFAULT_FIRST_HIT_THRESHOLDS and thresholds_key != (PAPER_I_MAIN_TABLES_SPSA_TARGET,):
            raise ValueError(f"optimizer_profile={optimizer_profile_key} requires first_hit_thresholds=(2e-4,).")
        energy_stop_target = PAPER_I_MAIN_TABLES_SPSA_TARGET
        first_hit_thresholds = (PAPER_I_MAIN_TABLES_SPSA_TARGET,)
    if bool(ladder_config.get("enabled")):
        if bool(paper_i_ladder_benchmarks_only) and requested_snake:
            raise ValueError("Use either --paper-i-ladder-benchmarks-only or SNAKE inclusion flags, not both.")
        if ladder_candidate_manifest_path is not None:
            if ladder_stage != _PAPER_I_ESCALATION_TARGET_STAGE:
                raise ValueError(
                    "--paper-i-ladder-candidate-manifest is only supported for "
                    f"{_PAPER_I_ESCALATION_TARGET_STAGE}."
                )
            if not bool(paper_i_ladder_benchmarks_only) or requested_snake:
                raise ValueError(
                    "Generic audit-driven ladder filtering is comparator-only; "
                    "use --paper-i-ladder-benchmarks-only and the Route-A generator for SNAKE rows."
                )
            candidates = _escalation_candidates_for_lane(
                ladder_candidate_manifest_path,
                lane="comparator",
                target_stage=_PAPER_I_ESCALATION_TARGET_STAGE,
            )
            available_case_ids = set(_escalation_target_case_ids(candidates))
            requested_case_ids = {
                str(case_id).strip()
                for case_id in (paper_i_ladder_case_ids or ())
                if str(case_id).strip()
            }
            if requested_case_ids:
                missing_requested = sorted(requested_case_ids - available_case_ids)
                if missing_requested:
                    raise ValueError(
                        "paper_i_ladder_candidate_manifest missing requested case_id(s): "
                        f"{missing_requested}; available: {sorted(available_case_ids)}"
                    )
                candidates = [
                    candidate
                    for candidate in candidates
                    if str(candidate.get("next_stage_case_id") or "") in requested_case_ids
                ]
            ladder_candidate_case_ids = _escalation_target_case_ids(candidates)
            ladder_candidate_records_by_id = {
                _escalation_generic_target_record_id(candidate): dict(candidate)
                for candidate in candidates
            }
            ladder_candidate_summary = {
                "manifest_path": str(ladder_candidate_manifest_path),
                "lane": "comparator",
                "target_stage": _PAPER_I_ESCALATION_TARGET_STAGE,
                "candidate_count": len(candidates),
                "target_case_ids": list(ladder_candidate_case_ids),
                "record_ids": sorted(ladder_candidate_records_by_id),
            }
        if not bool(paper_i_ladder_benchmarks_only) and not snake_only:
            include_snake = True
        energy_stop_target = _PAPER_I_CLEAN_TAU_TIGHT
        first_hit_thresholds = _unique_float_thresholds((_PAPER_I_CLEAN_TAU_PHYS, _PAPER_I_CLEAN_TAU_TIGHT))
    if calibration_profile_key == "nph2_route_a_hk_hh_v1":
        energy_stop_target = DEFAULT_ENERGY_STOP_TARGET
        first_hit_thresholds = DEFAULT_FIRST_HIT_THRESHOLDS
    normalized_family_filter = tuple(
        str(family).strip()
        for family in (family_filter or ())
        if str(family).strip()
    )
    if optimizer_profile_key and normalized_family_filter:
        raise ValueError(f"optimizer_profile={optimizer_profile_key} must generate the full visible main-table case set; family filters are not allowed.")
    molecular_vibronic_h2_only = set(normalized_family_filter) == {"molecular_vibronic_h2"}
    if suite_profile_key != TABLE_I_STANDARD_PROFILE and Path(queue_output_root) == DEFAULT_QUEUE_OUTPUT_ROOT:
        queue_output_root = Path(f"{DEFAULT_QUEUE_OUTPUT_ROOT}_{suite_profile_key}")
    phase3_oracle_overlay = _normalize_phase3_oracle_overlay(
        phase3_oracle_gradient_mode=phase3_oracle_gradient_mode,
        phase3_oracle_backend_name=phase3_oracle_backend_name,
        phase3_oracle_use_fake_backend=phase3_oracle_use_fake_backend,
        phase3_oracle_shots=phase3_oracle_shots,
        phase3_oracle_repeats=phase3_oracle_repeats,
        phase3_oracle_aggregate=phase3_oracle_aggregate,
        phase3_oracle_seed=phase3_oracle_seed,
        phase3_oracle_execution_surface=phase3_oracle_execution_surface,
        phase3_oracle_inner_objective_mode=phase3_oracle_inner_objective_mode,
        phase3_oracle_value_noise_model=phase3_oracle_value_noise_model,
        phase3_oracle_value_noise_std=phase3_oracle_value_noise_std,
        phase3_oracle_value_noise_seed=phase3_oracle_value_noise_seed,
    )
    phase3_oracle_requested = _phase3_oracle_overlay_requested(phase3_oracle_overlay)
    hardware_resolution_profile_overlay = _normalize_hardware_resolution_profile_overlay(
        hardware_resolution_mode=hardware_resolution_mode,
        hardware_resolution_profile_json=hardware_resolution_profile_json,
        hardware_resolution_profile_name=hardware_resolution_profile_name,
    )
    hardware_resolution_profile_requested = _hardware_resolution_profile_overlay_requested(hardware_resolution_profile_overlay)
    benchmark_value_noise_overlay = _normalize_benchmark_value_noise_overlay(
        benchmark_value_noise_model=benchmark_value_noise_model,
        benchmark_value_noise_std=benchmark_value_noise_std,
        benchmark_value_noise_seed=benchmark_value_noise_seed,
    )
    benchmark_value_noise_requested = _benchmark_value_noise_overlay_requested(benchmark_value_noise_overlay)
    benchmark_decision_noise_overlay = _normalize_benchmark_decision_noise_overlay(
        benchmark_decision_noise_model=benchmark_decision_noise_model,
        benchmark_decision_noise_std=benchmark_decision_noise_std,
        benchmark_decision_noise_seed=benchmark_decision_noise_seed,
    )
    benchmark_decision_noise_requested = _benchmark_decision_noise_overlay_requested(benchmark_decision_noise_overlay)
    phase3_budget_profile_key, phase3_budget_overlay = _normalize_phase3_smoke_budget_overlay(phase3_smoke_budget_profile)
    phase3_budget_requested = _phase3_budget_overlay_requested(phase3_budget_overlay)
    phase3_policy_profile_key, phase3_policy_overlay = _normalize_phase3_policy_overlay(phase3_policy_profile)
    phase3_policy_requested = _phase3_budget_overlay_requested(phase3_policy_overlay)
    generic_adapt_budget_profile_key, generic_adapt_budget_overlay = _normalize_generic_adapt_budget_overlay(
        generic_adapt_budget_profile
    )
    generic_adapt_budget_requested = _phase3_budget_overlay_requested(generic_adapt_budget_overlay)
    generic_adapt_smoke_budget_profile_key, generic_adapt_smoke_budget_overlay = (
        _normalize_generic_adapt_smoke_budget_overlay(generic_adapt_smoke_budget_profile)
    )
    generic_adapt_smoke_budget_requested = _phase3_budget_overlay_requested(generic_adapt_smoke_budget_overlay)
    phase3_policy_json_overlay = _normalize_phase3_policy_json_overlay(phase3_policy_json)
    phase3_policy_json_requested = _phase3_policy_json_overlay_requested(phase3_policy_json_overlay)
    phase2_mode = _normalize_phase2_novelty_mode(phase2_novelty_mode) or "collective_span_v1"
    phase3_runtime_overlay = _normalize_phase3_runtime_overlay(
        phase3_adapt_parallel_gradient_workers=phase3_adapt_parallel_gradient_workers,
        phase3_adapt_beam_parent_workers=phase3_adapt_beam_parent_workers,
    )
    phase3_runtime_requested = _phase3_runtime_overlay_requested(phase3_runtime_overlay)
    generic_adapt_runtime_split_overlay = _normalize_generic_adapt_runtime_split_overlay(
        generic_adapt_runtime_split_mode=generic_adapt_runtime_split_mode,
        generic_adapt_runtime_split_symmetry_policy=generic_adapt_runtime_split_symmetry_policy,
        generic_adapt_runtime_split_max_subset_size=generic_adapt_runtime_split_max_subset_size,
    )
    generic_adapt_runtime_split_requested = _generic_adapt_runtime_split_overlay_requested(
        generic_adapt_runtime_split_overlay
    )
    shared_pauli_pool_overlay = _normalize_shared_pauli_pool_overlay(
        shared_pauli_pool_mode=shared_pauli_pool_mode,
        shared_pauli_pool_symmetry_policy=shared_pauli_pool_symmetry_policy,
        shared_pauli_pool_max_subset_size=shared_pauli_pool_max_subset_size,
    )
    shared_pauli_pool_requested = _shared_pauli_pool_overlay_requested(shared_pauli_pool_overlay)
    if shared_pauli_pool_requested and generic_adapt_runtime_split_requested:
        raise ValueError("shared_pauli_pool overlay cannot be combined with generic_adapt_runtime_split overlay.")
    hh_pos_geo_position_policy_key = str(hh_pos_geo_position_policy or "").strip().lower()
    if hh_pos_geo_position_policy_key not in {"", "append", "best_insert_refit"}:
        raise ValueError("hh_pos_geo_position_policy must be blank, append, or best_insert_refit.")
    if snake_only and include_snake:
        raise ValueError("Use either --snake-only or --include-snake, not both.")
    if (
        not bool(ladder_config.get("enabled"))
        and suite_profile_key in {
        TABLE_I_CLEAN_NPH2_REF3_PROFILE,
        TABLE_I_CLEAN_NPH1_REF4_PROFILE,
        TABLE_I_CLEAN_NPH2_REF4_PROFILE,
        TABLE_I_CLEAN_NPH2_REF5_PROFILE,
        TABLE_I_CLEAN_NPH3_REF4_PROFILE,
        TABLE_I_CLEAN_NPH4_REF5_PROFILE,
        TABLE_I_CLEAN_NPH4_REF7_PROFILE,
        TABLE_I_CLEAN_NPH6_REF9_PROFILE,
        }
        and (
        include_snake or snake_only
        )
        and not molecular_vibronic_h2_only
    ):
        raise ValueError(
            "Clean Paper-I comparator profiles are benchmark-only. "
            "Use the dedicated Route-A/SNAKE Optuna generators for SNAKE rows."
        )
    if optimizer_profile_key:
        algorithms = PAPER_I_MAIN_TABLES_SPSA_COMPARATOR_ALGORITHM_IDS
    elif calibration_profile_key == "nph2_route_a_hk_hh_v1":
        algorithms = _SNAKE_TABLE_I_ALGORITHM_IDS
    else:
        algorithms = (
            _SNAKE_TABLE_I_ALGORITHM_IDS
            if snake_only
            else TABLE_I_STATIC_ALGORITHM_IDS
            if include_snake
            else TABLE_I_STATIC_BENCHMARK_ALGORITHM_IDS
        )
    requested_algorithms = tuple(
        str(value).strip()
        for value in (algorithm_filter or ())
        if str(value).strip()
    )
    if requested_algorithms:
        if optimizer_profile_key:
            disallowed = sorted(
                set(requested_algorithms)
                & {PAPER_I_MAIN_TABLES_SPSA_SNAKE_ALGORITHM_ID, "static_pos_geo_adapt_vqe"}
            )
            if disallowed:
                raise ValueError(
                    f"optimizer_profile={optimizer_profile_key} rejects SNAKE/PosGeo algorithm filter(s): {disallowed}."
                )
        available_algorithms = set(algorithms)
        missing_algorithms = sorted(set(requested_algorithms) - available_algorithms)
        if missing_algorithms:
            raise ValueError(
                f"algorithm filter requested unavailable algorithm(s) {missing_algorithms}; "
                f"available for this mode: {sorted(available_algorithms)}"
            )
        if optimizer_profile_key and set(requested_algorithms) != set(PAPER_I_MAIN_TABLES_SPSA_COMPARATOR_ALGORITHM_IDS):
            raise ValueError(
                f"optimizer_profile={optimizer_profile_key} must generate exactly the six displayed non-SNAKE comparators."
            )
        algorithms = tuple(algorithm for algorithm in algorithms if algorithm in set(requested_algorithms))
    old_profile = os.environ.get(TABLE_I_STATIC_SUITE_PROFILE_ENV)
    os.environ[TABLE_I_STATIC_SUITE_PROFILE_ENV] = suite_profile_key
    try:
        jobs = build_table_i_static_jobs(
            output_root=queue_output_root,
            algorithm_ids=algorithms,
            include_skipped=True,
        )
    finally:
        if old_profile is None:
            os.environ.pop(TABLE_I_STATIC_SUITE_PROFILE_ENV, None)
        else:
            os.environ[TABLE_I_STATIC_SUITE_PROFILE_ENV] = old_profile
    if normalized_family_filter:
        allowed_families = set(normalized_family_filter)
        available_families = {str(job.family) for job in jobs}
        jobs = [job for job in jobs if str(job.family) in allowed_families]
        emitted_families = {str(job.family) for job in jobs}
        missing_families = sorted(allowed_families - emitted_families)
        if missing_families:
            raise ValueError(
                f"family filter requested unavailable family/families {missing_families}; "
                f"available families for {suite_profile_key}: {sorted(available_families)}"
            )
    records = _records_from_jobs(
        jobs,
        suite_profile=suite_profile_key,
        energy_stop_target=float(energy_stop_target),
        first_hit_thresholds=first_hit_thresholds,
        phase3_oracle_overlay=phase3_oracle_overlay,
        phase3_policy_overlay=phase3_policy_overlay,
        phase3_policy_json_overlay=phase3_policy_json_overlay,
        phase3_runtime_overlay=phase3_runtime_overlay,
        generic_adapt_runtime_split_overlay=generic_adapt_runtime_split_overlay,
        shared_pauli_pool_overlay=shared_pauli_pool_overlay,
        generic_adapt_budget_overlay=generic_adapt_budget_overlay,
        phase2_novelty_mode=phase2_mode,
        hardware_resolution_profile_overlay=hardware_resolution_profile_overlay,
        benchmark_value_noise_overlay=benchmark_value_noise_overlay,
        benchmark_decision_noise_overlay=benchmark_decision_noise_overlay,
        hh_pos_geo_position_policy=hh_pos_geo_position_policy_key,
        disable_resource_guards=bool(disable_resource_guards),
        hea_reps=hea_reps,
        hea_maxiter=hea_maxiter,
        optimizer_profile=optimizer_profile_key,
    )
    if optimizer_profile_key:
        expected_pairs = {
            (case_id, algorithm_id)
            for case_id in PAPER_I_MAIN_TABLES_SPSA_CASE_IDS
            for algorithm_id in PAPER_I_MAIN_TABLES_SPSA_COMPARATOR_ALGORITHM_IDS
        }
        actual_pairs = {(str(record.get("case_id") or ""), str(record.get("algorithm_id") or "")) for record in records}
        if actual_pairs != expected_pairs or len(records) != 48:
            missing = sorted(expected_pairs - actual_pairs)
            extra = sorted(actual_pairs - expected_pairs)
            raise ValueError(
                f"optimizer_profile={optimizer_profile_key} must generate exactly 48 visible comparator rows; "
                f"got {len(records)}, missing={missing}, extra={extra}."
            )
    if calibration_profile_key == "nph2_route_a_hk_hh_v1":
        records = _filter_records_by_ids(
            records,
            record_ids=_NPH2_ROUTE_A_HK_HH_CALIBRATION_TARGET_RECORD_IDS,
            label="Calibration target",
        )
    if bool(ladder_config.get("enabled")):
        ladder_snake_policy = (
            "benchmarks_only_explicit"
            if bool(paper_i_ladder_benchmarks_only)
            else "snake_only"
            if bool(snake_only)
            else "included"
        )
        records = _apply_ladder_metadata(
            records,
            stage=ladder_stage,
            config=ladder_config,
            case_ids=ladder_candidate_case_ids or tuple(paper_i_ladder_case_ids or ()),
            allow_ref5=bool(paper_i_ladder_allow_ref5),
            escalation_reason=(
                str(paper_i_ladder_escalation_reason or "")
                or ("audit_driven_phonon_ladder_escalation" if ladder_candidate_manifest_path is not None else "")
            ),
            snake_policy=ladder_snake_policy,
        )
        records = _apply_reference_energy_metadata(
            records,
            suite_profile=suite_profile_key,
            output_dir=Path(output_dir),
        )
        if ladder_candidate_manifest_path is not None:
            records = _filter_records_by_ids(
                records,
                record_ids=tuple(sorted(ladder_candidate_records_by_id)),
                label="Paper-I audit escalation candidate",
            )
            for row in records:
                candidate = ladder_candidate_records_by_id[str(row["record_id"])]
                row.update(
                    _escalation_source_metadata_fields(
                        candidate,
                        candidate_manifest_json=ladder_candidate_manifest_path,
                    )
                )
                row["paper_i_ladder_escalation_reason"] = str(candidate.get("escalation_reason") or "")
            if ladder_candidate_summary is not None:
                ladder_candidate_summary["applied_record_count"] = len(records)
    elif molecular_vibronic_h2_only and any(str(row.get("n_ph_ref") or "").strip() for row in records):
        records = _apply_reference_energy_metadata(
            records,
            suite_profile=suite_profile_key,
            output_dir=Path(output_dir),
        )
    selected_route_key = str(selected_logical_route or "standard").strip().lower().replace("-", "_")
    selected_source = str(selected_logical_source_json or "").strip()
    selected_transfer = str(selected_logical_transfer_mode or "exact_match_v1").strip().lower()
    if selected_route_key not in {"standard", "historical_selected"}:
        raise ValueError("selected_logical_route must be standard or historical_selected")
    if selected_transfer not in {"exact_match_v1", "boundary_v1"}:
        raise ValueError("selected_logical_transfer_mode must be exact_match_v1 or boundary_v1")
    applied_selected_overlay_count = 0
    if selected_route_key == "historical_selected":
        if not selected_source:
            raise ValueError("selected_logical_source_json is required for historical_selected generic static rows")
        for row in records:
            if bool(selected_logical_supported_algorithms_only) and str(row.get("algorithm_id") or "") not in _SELECTED_LOGICAL_SUPPORTED_ALGORITHM_IDS:
                continue
            row["selected_logical_route"] = selected_route_key
            row["selected_logical_source_json"] = selected_source
            row["selected_logical_transfer_mode"] = selected_transfer
            applied_selected_overlay_count += 1
        if bool(selected_logical_supported_algorithms_only) and applied_selected_overlay_count == 0:
            raise ValueError("selected_logical_supported_algorithms_only selected no supported records")
    phase3_oracle_applied_count = sum(
        1
        for record in records
        if any(str(record.get(field) or "").strip() != "" for field in PHASE3_ORACLE_TSV_FIELDS)
    )
    if phase3_oracle_requested and phase3_oracle_applied_count == 0:
        raise ValueError("Phase3 oracle/value-noise overlay requested, but no phase3_static_adapt records were generated; use --include-snake for this smoke slice.")
    phase3_runtime_applied_count = sum(
        1
        for record in records
        if any(str(record.get(field) or "").strip() != "" for field in PHASE3_RUNTIME_TSV_FIELDS)
    )
    if phase3_runtime_requested and phase3_runtime_applied_count == 0:
        raise ValueError(
            "Phase3 runtime overlay requested, but no native SNAKE phase3_static_adapt records were generated; "
            "use --include-snake or --snake-only."
        )
    phase3_policy_json_applied_count = sum(
        1
        for record in records
        if any(str(record.get(field) or "").strip() != "" for field in PHASE3_POLICY_JSON_TSV_FIELDS)
    )
    if phase3_policy_json_requested and phase3_policy_json_applied_count == 0:
        raise ValueError(
            "Phase3 policy JSON overlay requested, but no native SNAKE phase3_static_adapt records were generated; "
            "use --include-snake or --snake-only."
        )
    hardware_resolution_profile_applied_count = sum(
        1
        for record in records
        if any(str(record.get(field) or "").strip() != "" for field in HARDWARE_RESOLUTION_PROFILE_TSV_FIELDS)
    )
    if hardware_resolution_profile_requested and hardware_resolution_profile_applied_count == 0:
        raise ValueError(
            "Hardware-resolution profile overlay requested, but no phase3_static_adapt records were generated; "
            "use --include-snake or --snake-only."
        )
    static_route_applied_count = sum(1 for record in records if str(record.get("static_route_id") or "").strip() != "")
    static_route_a_record_count = sum(1 for record in records if str(record.get("static_route_id") or "").strip() == ROUTE_ID_A)
    static_route_unspecified_record_count = sum(
        1 for record in records if str(record.get("static_route_id") or "").strip() == ROUTE_ID_UNSPECIFIED
    )
    hardware_profile_rows_marked_diagnostic = bool(
        hardware_resolution_profile_requested
        and hardware_resolution_profile_applied_count > 0
        and all(
            str(record.get("static_route_id") or "").strip() == ROUTE_ID_UNSPECIFIED
            for record in records
            if any(str(record.get(field) or "").strip() != "" for field in HARDWARE_RESOLUTION_PROFILE_TSV_FIELDS)
        )
    )
    benchmark_value_noise_applied_count = sum(
        1
        for record in records
        if any(str(record.get(field) or "").strip() != "" for field in BENCHMARK_VALUE_NOISE_TSV_FIELDS)
    )
    if benchmark_value_noise_requested and benchmark_value_noise_applied_count == 0:
        raise ValueError("Benchmark value-noise overlay requested, but no runnable generic static benchmark records were generated.")
    benchmark_decision_noise_applied_count = sum(
        1
        for record in records
        if any(str(record.get(field) or "").strip() != "" for field in BENCHMARK_DECISION_NOISE_TSV_FIELDS)
    )
    if benchmark_decision_noise_requested and benchmark_decision_noise_applied_count == 0:
        raise ValueError(
            "Benchmark decision-noise overlay requested, but no non-Phase3 benchmark records were generated; "
            "use phase3_oracle_value_noise_* for SNAKE/Phase3 rows."
        )
    phase3_policy_applied_count = (
        sum(
            1
            for record in records
            if all(
                str(record.get(field) or "").strip() == str(value or "").strip()
                for field, value in phase3_policy_overlay.items()
                if str(value or "").strip() != ""
            )
        )
        if phase3_policy_requested
        else 0
    )
    if phase3_policy_requested and phase3_policy_applied_count == 0:
        raise ValueError(
            "Phase3 policy overlay requested, but no native SNAKE phase3_static_adapt records were generated; "
            "use --include-snake or --snake-only."
        )
    generic_adapt_budget_applied_count = (
        sum(
            1
            for record in records
            if str(record.get("algorithm_id") or "") in _GENERIC_STATIC_ADAPT_VARIANT_ALGORITHM_IDS
            and all(
                str(record.get(field) or "").strip() == str(value or "").strip()
                for field, value in generic_adapt_budget_overlay.items()
                if str(value or "").strip() != ""
            )
        )
        if generic_adapt_budget_requested
        else 0
    )
    if generic_adapt_budget_requested and generic_adapt_budget_applied_count == 0:
        raise ValueError(
            "Generic ADAPT budget overlay requested, but no generic_static_adapt_variants records were generated."
        )
    generic_adapt_runtime_split_applied_count = (
        sum(
            1
            for record in records
            if str(record.get("family") or "") in _GENERIC_ADAPT_RUNTIME_SPLIT_SUPPORTED_FAMILIES
            and str(record.get("algorithm_id") or "") in _GENERIC_ADAPT_RUNTIME_SPLIT_SUPPORTED_ALGORITHM_IDS
            and all(
                str(record.get(field) or "").strip() == str(value or "").strip()
                for field, value in generic_adapt_runtime_split_overlay.items()
                if str(value or "").strip() != ""
            )
        )
        if generic_adapt_runtime_split_requested
        else 0
    )
    if generic_adapt_runtime_split_requested and generic_adapt_runtime_split_applied_count == 0:
        raise ValueError(
            "Generic ADAPT runtime split overlay requested, but no HH/Hubbard append/Geo generic_static_adapt_variants records were generated."
        )
    shared_pauli_pool_applied_count = (
        sum(
            1
            for record in records
            if str(record.get("family") or "") in _SHARED_PAULI_POOL_SUPPORTED_FAMILIES
            and str(record.get("algorithm_id") or "") in _SHARED_PAULI_POOL_SUPPORTED_ALGORITHM_IDS
            and all(
                str(record.get(field) or "").strip() == str(value or "").strip()
                for field, value in shared_pauli_pool_overlay.items()
                if str(value or "").strip() != ""
            )
        )
        if shared_pauli_pool_requested
        else 0
    )
    if shared_pauli_pool_requested and shared_pauli_pool_applied_count == 0:
        raise ValueError(
            "Shared Pauli pool overlay requested, but no HH/Hubbard SNAKE, append, or Geo records were generated."
        )
    has_effective_phase3_records = bool(
        include_snake or snake_only or calibration_profile_key == "nph2_route_a_hk_hh_v1"
    )
    if phase3_budget_requested and not has_effective_phase3_records:
        raise ValueError("Phase3 smoke budget overlay requested, but no phase3_static_adapt records were generated; use --include-snake for this smoke slice.")
    requested_smoke_case_ids = tuple(
        str(case_id).strip()
        for case_id in (smoke_case_ids or ())
        if str(case_id).strip()
    )
    if requested_smoke_case_ids:
        smoke_records = _filter_records_by_case_ids(
            records,
            case_ids=requested_smoke_case_ids,
            label="Smoke",
        )
        phase3_budget_applied_count = 0
    elif calibration_profile_key == "nph2_route_a_hk_hh_v1":
        smoke_records = _filter_records_by_ids(
            records,
            record_ids=_NPH2_ROUTE_A_HK_HH_CALIBRATION_SMOKE_RECORD_IDS,
            label="Calibration smoke",
        )
        phase3_budget_applied_count = 0
        if phase3_budget_requested:
            for row in smoke_records:
                row.update(phase3_budget_overlay)
                phase3_budget_applied_count += 1
    elif bool(ladder_config.get("enabled")):
        smoke_records = []
        seen_families: set[str] = set()
        for row in sorted(records, key=lambda item: (item["family"], item["case_id"], item["algorithm_id"])):
            family = str(row.get("family") or "")
            if str(row.get("algorithm_id") or "") in _PHASE3_STATIC_ADAPT_ALGORITHM_IDS:
                continue
            if family in seen_families:
                continue
            smoke_records.append(dict(row))
            seen_families.add(family)
        if not smoke_records:
            raise ValueError(f"Paper-I cutoff ladder stage {ladder_stage} has no smoke records.")
        phase3_budget_applied_count = 0
        if ladder_snake_policy in {"included", "snake_only"}:
            snake_candidates = [
                row
                for row in sorted(records, key=lambda item: (item["family"], item["case_id"], item["algorithm_id"]))
                if str(row.get("algorithm_id") or "") in _PHASE3_STATIC_ADAPT_ALGORITHM_IDS
            ]
            if not snake_candidates:
                raise ValueError(f"Paper-I cutoff ladder stage {ladder_stage} has no SNAKE smoke candidate.")
            snake_smoke = dict(snake_candidates[0])
            if not phase3_budget_requested:
                snake_smoke.update(_PHASE3_WEAK_LOCAL_BUDGET_FIELDS)
                phase3_budget_applied_count += 1
            elif phase3_budget_requested:
                snake_smoke.update(phase3_budget_overlay)
                phase3_budget_applied_count += 1
            smoke_records.append(snake_smoke)
    elif snake_only:
        smoke_records = []
        phase3_budget_applied_count = 0
        if normalized_family_filter:
            seen_families: set[str] = set()
            for row in sorted(records, key=lambda item: (item["family"], item["case_id"], item["algorithm_id"])):
                if str(row.get("algorithm_id") or "") not in _PHASE3_STATIC_ADAPT_ALGORITHM_IDS:
                    continue
                family = str(row.get("family") or "")
                if family in seen_families:
                    continue
                smoke = dict(row)
                if phase3_budget_requested:
                    smoke.update(phase3_budget_overlay)
                    phase3_budget_applied_count += 1
                smoke_records.append(smoke)
                seen_families.add(family)
            if not smoke_records:
                raise ValueError(f"Family-filtered SNAKE suite {suite_profile_key} has no smoke records.")
        else:
            by_id = {record["record_id"]: record for record in records}
            for record_id in _phase3_static_smoke_record_ids_for_profile(suite_profile_key):
                if record_id not in by_id:
                    raise ValueError(f"SNAKE smoke record selection missing generated record: {record_id}")
                row = dict(by_id[record_id])
                if phase3_budget_requested:
                    row.update(phase3_budget_overlay)
                    phase3_budget_applied_count += 1
                smoke_records.append(row)
    else:
        if normalized_family_filter:
            smoke_records = []
            seen_families: set[str] = set()
            for row in sorted(records, key=lambda item: (item["family"], item["case_id"], item["algorithm_id"])):
                family = str(row.get("family") or "")
                if family in seen_families:
                    continue
                smoke_records.append(dict(row))
                seen_families.add(family)
            if not smoke_records:
                raise ValueError(f"Family-filtered suite {suite_profile_key} has no smoke records.")
            phase3_budget_applied_count = 0
        else:
            smoke_records, phase3_budget_applied_count = _select_smoke_records(
                records,
                suite_profile=suite_profile_key,
                include_phase3_static=(
                    phase3_oracle_requested
                    or phase3_budget_requested
                    or phase3_policy_requested
                    or phase3_policy_json_requested
                    or phase3_runtime_requested
                    or hardware_resolution_profile_requested
                ),
                phase3_budget_overlay=phase3_budget_overlay,
            )
    if optimizer_profile_key:
        smoke_records = [
            _apply_optimizer_profile_fields(record, optimizer_profile=optimizer_profile_key, smoke=True)
            for record in smoke_records
        ]
    if phase3_budget_requested and phase3_budget_applied_count == 0:
        raise ValueError("Phase3 smoke budget overlay requested, but no phase3_static_adapt smoke records were generated; use --include-snake for this smoke slice.")
    generic_adapt_smoke_budget_applied_count = 0
    if generic_adapt_smoke_budget_requested:
        for row in smoke_records:
            if str(row.get("algorithm_id") or "") not in _GENERIC_STATIC_ADAPT_VARIANT_ALGORITHM_IDS:
                continue
            row.update(generic_adapt_smoke_budget_overlay)
            generic_adapt_smoke_budget_applied_count += 1
        if generic_adapt_smoke_budget_applied_count == 0:
            raise ValueError(
                "Generic ADAPT smoke budget overlay requested, but no generic_static_adapt_variants smoke records were generated."
            )

    output = Path(output_dir)
    full_records_path = output / "generic_static_table_records.tsv"
    full_ids_path = output / "generic_static_table_record_ids.txt"
    smoke_records_path = output / "generic_static_table_smoke_records.tsv"
    smoke_ids_path = output / "generic_static_table_smoke_record_ids.txt"
    summary_path = output / "generic_static_table_records_summary.json"

    _write_records(full_records_path, records)
    _write_record_ids(full_ids_path, records)
    _write_records(smoke_records_path, smoke_records)
    _write_record_ids(smoke_ids_path, smoke_records)

    table_summary = summarize_table_i_jobs(jobs, suite_profile=suite_profile_key)
    summary = {
        "schema": "generic_static_table_chtc_records_v1",
        "include_snake": bool(include_snake),
        "snake_only": bool(snake_only),
        "suite_profile": suite_profile_key,
        "family_filter": list(normalized_family_filter),
        "energy_stop_target": float(energy_stop_target),
        "first_hit_thresholds": [float(x) for x in _unique_float_thresholds(first_hit_thresholds)],
        "phonon_cutoff_fields": {
            "fieldnames": list(PHONON_CUTOFF_TSV_FIELDS),
            "explicit_record_fields": True,
            "primary_energy_metric_for_phonon_rows": "higher_cutoff_reference_abs_delta_e",
        },
        "resource_guard_overlay": {
            "disabled": bool(disable_resource_guards),
            "fields": {field: "0" for field in RESOURCE_GUARD_TSV_FIELDS} if bool(disable_resource_guards) else {},
            "semantics": "0 means unbounded resource guard for the exact-bench comparator runner.",
        },
        "hea_overlay": {
            "reps": None if hea_reps is None else int(hea_reps),
            "maxiter": None if hea_maxiter is None else int(hea_maxiter),
        },
        "optimizer_profile_overlay": {
            "profile": optimizer_profile_key,
            "fieldnames": list(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_TSV_FIELDS),
            "applied_record_count": sum(
                1 for record in records if str(record.get("optimizer_profile") or "") == optimizer_profile_key
            ) if optimizer_profile_key else 0,
            "smoke_record_count": sum(
                1 for record in smoke_records if str(record.get("optimizer_profile") or "") == optimizer_profile_key
            ) if optimizer_profile_key else 0,
        },
        "paper_i_cutoff_ladder": {
            "enabled": bool(ladder_config.get("enabled")),
            "stage": ladder_stage,
            "suite_profile": suite_profile_key if bool(ladder_config.get("enabled")) else None,
            "n_ph_work": ladder_config.get("n_ph_work"),
            "n_ph_ref": ladder_config.get("n_ph_ref"),
            "acceptance_threshold": float(_PAPER_I_CLEAN_TAU_TIGHT),
            "tau_phys": float(_PAPER_I_CLEAN_TAU_PHYS),
            "requires_prior_failure": bool(ladder_config.get("requires_prior_failure")),
            "case_ids": list(paper_i_ladder_case_ids or ()),
            "allow_ref5": bool(paper_i_ladder_allow_ref5),
            "escalation_reason": str(paper_i_ladder_escalation_reason or ""),
            "snake_policy": ladder_snake_policy,
            "benchmarks_only": bool(paper_i_ladder_benchmarks_only),
            "candidate_manifest_filter": ladder_candidate_summary,
        },
        "algorithm_ids": list(algorithms),
        "algorithm_filter": list(requested_algorithms),
        "method_labels": {algorithm_id: table_i_method_label(algorithm_id) for algorithm_id in algorithms},
        "job_count": len(jobs),
        "runnable_record_count": len(records),
        "smoke_record_count": len(smoke_records),
        "status_by_algorithm": _status_by_algorithm(jobs),
        "phase3_oracle_overlay": {
            "requested": bool(phase3_oracle_requested),
            "applied_record_count": int(phase3_oracle_applied_count),
            "fields": dict(phase3_oracle_overlay) if phase3_oracle_requested else _blank_phase3_oracle_fields(),
        },
        "phase3_budget_overlay": {
            "requested": bool(phase3_budget_requested),
            "applied": bool(phase3_budget_applied_count),
            "applied_record_count": int(phase3_budget_applied_count),
            "profile": phase3_budget_profile_key,
            "fields": dict(phase3_budget_overlay) if phase3_budget_requested else _blank_phase3_budget_fields(),
        },
        "phase3_policy_overlay": {
            "requested": bool(phase3_policy_requested),
            "applied": bool(phase3_policy_applied_count),
            "applied_record_count": int(phase3_policy_applied_count),
            "profile": phase3_policy_profile_key,
            "fields": dict(phase3_policy_overlay) if phase3_policy_requested else _blank_phase3_budget_fields(),
        },
        "generic_adapt_budget_overlay": {
            "requested": bool(generic_adapt_budget_requested),
            "applied": bool(generic_adapt_budget_applied_count),
            "applied_record_count": int(generic_adapt_budget_applied_count),
            "profile": generic_adapt_budget_profile_key,
            "fields": (
                dict(generic_adapt_budget_overlay)
                if generic_adapt_budget_requested
                else _blank_phase3_budget_fields()
            ),
            "semantics": (
                "generic_static_adapt_variants bounded first-hit repair budget; "
                "not applied to HEA/family-informed/SNAKE rows"
            ),
        },
        "generic_adapt_runtime_split_overlay": {
            "requested": bool(generic_adapt_runtime_split_requested),
            "applied": bool(generic_adapt_runtime_split_applied_count),
            "applied_record_count": int(generic_adapt_runtime_split_applied_count),
            "fields": (
                dict(generic_adapt_runtime_split_overlay)
                if generic_adapt_runtime_split_requested
                else _blank_generic_adapt_runtime_split_fields()
            ),
            "supported_algorithm_ids": sorted(_GENERIC_ADAPT_RUNTIME_SPLIT_SUPPORTED_ALGORITHM_IDS),
            "supported_families": sorted(_GENERIC_ADAPT_RUNTIME_SPLIT_SUPPORTED_FAMILIES),
            "semantics": "Paper-I HH/Hubbard full_meta append/Geo Pauli-child candidate expansion overlay",
        },
        "shared_pauli_pool_overlay": {
            "requested": bool(shared_pauli_pool_requested),
            "applied": bool(shared_pauli_pool_applied_count),
            "applied_record_count": int(shared_pauli_pool_applied_count),
            "fields": (
                dict(shared_pauli_pool_overlay)
                if shared_pauli_pool_requested
                else _blank_shared_pauli_pool_fields()
            ),
            "supported_algorithm_ids": sorted(_SHARED_PAULI_POOL_SUPPORTED_ALGORITHM_IDS),
            "supported_families": sorted(_SHARED_PAULI_POOL_SUPPORTED_FAMILIES),
            "semantics": "Canonical Paper-I shared parent-plus-Pauli-child-set pool overlay for SNAKE, Geo, and append; symmetry_policy=off is explicit no-guard, not disabled mode.",
        },
        "generic_adapt_smoke_budget_overlay": {
            "requested": bool(generic_adapt_smoke_budget_requested),
            "applied": bool(generic_adapt_smoke_budget_applied_count),
            "applied_record_count": int(generic_adapt_smoke_budget_applied_count),
            "profile": generic_adapt_smoke_budget_profile_key,
            "fields": (
                dict(generic_adapt_smoke_budget_overlay)
                if generic_adapt_smoke_budget_requested
                else _blank_phase3_budget_fields()
            ),
            "case_ids": list(requested_smoke_case_ids),
            "semantics": "smoke-record-only generic_static_adapt_variants budget overlay",
        },
        "phase3_runtime_overlay": {
            "requested": bool(phase3_runtime_requested),
            "applied": bool(phase3_runtime_applied_count),
            "applied_record_count": int(phase3_runtime_applied_count),
            "fields": dict(phase3_runtime_overlay) if phase3_runtime_requested else _blank_phase3_runtime_fields(),
        },
        "phase3_policy_json_overlay": {
            "requested": bool(phase3_policy_json_requested),
            "applied": bool(phase3_policy_json_applied_count),
            "applied_record_count": int(phase3_policy_json_applied_count),
            "fields": (
                dict(phase3_policy_json_overlay)
                if phase3_policy_json_requested
                else _blank_phase3_policy_json_fields()
            ),
        },
        "phase2_novelty_mode": phase2_mode,
        "hardware_resolution_profile_overlay": {
            "requested": bool(hardware_resolution_profile_requested),
            "applied": bool(hardware_resolution_profile_applied_count),
            "applied_record_count": int(hardware_resolution_profile_applied_count),
            "fields": (
                dict(hardware_resolution_profile_overlay)
                if hardware_resolution_profile_requested
                else _blank_hardware_resolution_profile_fields()
            ),
        },
        "static_route_overlay": {
            "applied_record_count": int(static_route_applied_count),
            "route_a_record_count": int(static_route_a_record_count),
            "unspecified_record_count": int(static_route_unspecified_record_count),
            "hardware_profile_rows_marked_diagnostic": bool(hardware_profile_rows_marked_diagnostic),
        },
        "selected_logical_overlay": {
            "route": selected_route_key,
            "source_json": selected_source,
            "transfer_mode": selected_transfer,
            "supported_algorithms_only": bool(selected_logical_supported_algorithms_only),
            "supported_algorithm_ids": sorted(_SELECTED_LOGICAL_SUPPORTED_ALGORITHM_IDS),
            "applied_record_count": int(applied_selected_overlay_count),
        },
        "benchmark_value_noise_overlay": {
            "requested": bool(benchmark_value_noise_requested),
            "applied": bool(benchmark_value_noise_applied_count),
            "applied_record_count": int(benchmark_value_noise_applied_count),
            "semantic": _BENCHMARK_VALUE_NOISE_SEMANTIC,
            "fields": dict(benchmark_value_noise_overlay) if benchmark_value_noise_requested else _blank_benchmark_value_noise_fields(),
        },
        "benchmark_decision_noise_overlay": {
            "requested": bool(benchmark_decision_noise_requested),
            "applied": bool(benchmark_decision_noise_applied_count),
            "applied_record_count": int(benchmark_decision_noise_applied_count),
            "semantic": _BENCHMARK_DECISION_NOISE_SEMANTIC,
            "fields": dict(benchmark_decision_noise_overlay) if benchmark_decision_noise_requested else _blank_benchmark_decision_noise_fields(),
        },
        "table_i": table_summary,
        "paths": {
            "records_tsv": _portable_path(full_records_path),
            "record_ids_txt": _portable_path(full_ids_path),
            "smoke_records_tsv": _portable_path(smoke_records_path),
            "smoke_record_ids_txt": _portable_path(smoke_ids_path),
            "summary_json": _portable_path(summary_path),
        },
    }
    if calibration_profile_key != "off":
        summary["calibration"] = {
            "profile": calibration_profile_key,
            "target_record_ids": list(_NPH2_ROUTE_A_HK_HH_CALIBRATION_TARGET_RECORD_IDS),
            "smoke_record_ids": list(_NPH2_ROUTE_A_HK_HH_CALIBRATION_SMOKE_RECORD_IDS),
            "expected_route_identity": (
                _NPH2_ROUTE_A_HK_HH_CALIBRATION_ROUTE_IDENTITY
                if int(static_route_a_record_count) > 0
                else None
            ),
            "canonical_route_a_expected": bool(int(static_route_a_record_count) > 0),
            "declared_static_route_ids": sorted(
                {
                    str(record.get("static_route_id") or "").strip()
                    for record in records
                    if str(record.get("static_route_id") or "").strip()
                }
            ),
            "diagnostic_hardware_profile_route": bool(hardware_profile_rows_marked_diagnostic),
            "working_cutoff": int(_NPH2_ROUTE_A_HK_HH_CALIBRATION_WORKING_CUTOFF),
            "ref_cutoff": int(_NPH2_ROUTE_A_HK_HH_CALIBRATION_REF_CUTOFF),
        }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate CHTC generic static Table-I benchmark records.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--include-snake", action="store_true", default=False)
    parser.add_argument("--snake-only", action="store_true", default=False)
    parser.add_argument("--algorithm-id", action="append", dest="algorithm_filter", default=None)
    parser.add_argument("--queue-output-root", type=Path, default=DEFAULT_QUEUE_OUTPUT_ROOT)
    parser.add_argument("--suite-profile", default=TABLE_I_STANDARD_PROFILE)
    parser.add_argument("--family", action="append", dest="family_filter", default=None)
    parser.add_argument("--paper-i-ladder-candidate-manifest", type=Path, default=None)
    parser.add_argument("--energy-stop-target", type=float, default=DEFAULT_ENERGY_STOP_TARGET)
    parser.add_argument("--first-hit-threshold", type=float, action="append", dest="first_hit_thresholds", default=None)
    parser.add_argument("--phase3-oracle-gradient-mode", default=None)
    parser.add_argument("--phase3-oracle-backend-name", default=None)
    parser.add_argument("--phase3-oracle-use-fake-backend", action="store_true", default=None)
    parser.add_argument("--phase3-oracle-shots", type=int, default=None)
    parser.add_argument("--phase3-oracle-repeats", type=int, default=None)
    parser.add_argument("--phase3-oracle-aggregate", default=None)
    parser.add_argument("--phase3-oracle-seed", type=int, default=None)
    parser.add_argument("--phase3-oracle-execution-surface", default=None)
    parser.add_argument("--phase3-oracle-inner-objective-mode", default=None)
    parser.add_argument("--phase3-oracle-value-noise-model", default=None)
    parser.add_argument("--phase3-oracle-value-noise-std", type=float, default=None)
    parser.add_argument("--phase3-oracle-value-noise-seed", type=int, default=None)
    parser.add_argument("--hardware-resolution-mode", default=None)
    parser.add_argument("--hardware-resolution-profile-json", default=None)
    parser.add_argument("--hardware-resolution-profile-name", default=None)
    parser.add_argument("--benchmark-value-noise-model", default=None)
    parser.add_argument("--benchmark-value-noise-std", type=float, default=None)
    parser.add_argument("--benchmark-value-noise-seed", type=int, default=None)
    parser.add_argument("--benchmark-decision-noise-model", default=None)
    parser.add_argument("--benchmark-decision-noise-std", type=float, default=None)
    parser.add_argument("--benchmark-decision-noise-seed", type=int, default=None)
    parser.add_argument("--phase3-smoke-budget-profile", choices=sorted(_PHASE3_SMOKE_BUDGET_PROFILE_CHOICES), default="off")
    parser.add_argument("--phase3-policy-profile", choices=sorted(_PHASE3_POLICY_PROFILE_CHOICES), default="off")
    parser.add_argument(
        "--generic-adapt-budget-profile",
        choices=sorted(_GENERIC_ADAPT_BUDGET_PROFILE_CHOICES),
        default="off",
    )
    parser.add_argument(
        "--generic-adapt-smoke-budget-profile",
        choices=sorted(_GENERIC_ADAPT_SMOKE_BUDGET_PROFILE_CHOICES),
        default="off",
    )
    parser.add_argument("--phase3-policy-json", default=None)
    parser.add_argument("--phase2-novelty-mode", choices=sorted(x for x in _PHASE2_NOVELTY_MODE_CHOICES if x), default=None)
    parser.add_argument("--phase3-adapt-parallel-gradient-workers", type=int, default=None)
    parser.add_argument("--phase3-adapt-beam-parent-workers", type=int, default=None)
    parser.add_argument(
        "--generic-adapt-runtime-split-mode",
        choices=sorted(_GENERIC_ADAPT_RUNTIME_SPLIT_MODE_CHOICES),
        default="off",
    )
    parser.add_argument(
        "--generic-adapt-runtime-split-symmetry-policy",
        choices=sorted(_GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICY_CHOICES),
        default="off",
    )
    parser.add_argument("--generic-adapt-runtime-split-max-subset-size", type=int, default=None)
    parser.add_argument(
        "--shared-pauli-pool-mode",
        choices=sorted(_SHARED_PAULI_POOL_MODE_CHOICES),
        default="off",
    )
    parser.add_argument(
        "--shared-pauli-pool-symmetry-policy",
        choices=sorted(_SHARED_PAULI_POOL_SYMMETRY_POLICY_CHOICES),
        default="off",
    )
    parser.add_argument("--shared-pauli-pool-max-subset-size", type=int, default=None)
    parser.add_argument("--hh-pos-geo-position-policy", choices=["", "append", "best_insert_refit"], default="")
    parser.add_argument("--disable-resource-guards", action="store_true", default=False)
    parser.add_argument("--hea-reps", type=int, default=None)
    parser.add_argument("--hea-maxiter", type=int, default=None)
    parser.add_argument("--calibration-profile", choices=sorted(_CALIBRATION_PROFILE_CHOICES), default="off")
    parser.add_argument("--paper-i-cutoff-ladder-stage", choices=sorted(_PAPER_I_LADDER_STAGE_CONFIGS), default="off")
    parser.add_argument("--paper-i-ladder-case-id", action="append", dest="paper_i_ladder_case_ids", default=None)
    parser.add_argument("--paper-i-ladder-allow-ref5", action="store_true", default=False)
    parser.add_argument("--paper-i-ladder-escalation-reason", default="")
    parser.add_argument("--paper-i-ladder-benchmarks-only", action="store_true", default=False)
    parser.add_argument("--selected-logical-route", default="standard")
    parser.add_argument("--selected-logical-source-json", default="")
    parser.add_argument("--selected-logical-transfer-mode", default="exact_match_v1")
    parser.add_argument("--selected-logical-supported-algorithms-only", action="store_true", default=False)
    parser.add_argument("--smoke-case-id", action="append", dest="smoke_case_ids", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = generate_records(
        output_dir=Path(args.output_dir),
        include_snake=bool(args.include_snake),
        snake_only=bool(args.snake_only),
        algorithm_filter=tuple(args.algorithm_filter or ()),
        queue_output_root=Path(args.queue_output_root),
        suite_profile=str(args.suite_profile),
        family_filter=tuple(args.family_filter or ()),
        paper_i_ladder_candidate_manifest=args.paper_i_ladder_candidate_manifest,
        energy_stop_target=float(args.energy_stop_target),
        first_hit_thresholds=tuple(args.first_hit_thresholds or DEFAULT_FIRST_HIT_THRESHOLDS),
        phase3_oracle_gradient_mode=args.phase3_oracle_gradient_mode,
        phase3_oracle_backend_name=args.phase3_oracle_backend_name,
        phase3_oracle_use_fake_backend=args.phase3_oracle_use_fake_backend,
        phase3_oracle_shots=args.phase3_oracle_shots,
        phase3_oracle_repeats=args.phase3_oracle_repeats,
        phase3_oracle_aggregate=args.phase3_oracle_aggregate,
        phase3_oracle_seed=args.phase3_oracle_seed,
        phase3_oracle_execution_surface=args.phase3_oracle_execution_surface,
        phase3_oracle_inner_objective_mode=args.phase3_oracle_inner_objective_mode,
        phase3_oracle_value_noise_model=args.phase3_oracle_value_noise_model,
        phase3_oracle_value_noise_std=args.phase3_oracle_value_noise_std,
        phase3_oracle_value_noise_seed=args.phase3_oracle_value_noise_seed,
        hardware_resolution_mode=args.hardware_resolution_mode,
        hardware_resolution_profile_json=args.hardware_resolution_profile_json,
        hardware_resolution_profile_name=args.hardware_resolution_profile_name,
        benchmark_value_noise_model=args.benchmark_value_noise_model,
        benchmark_value_noise_std=args.benchmark_value_noise_std,
        benchmark_value_noise_seed=args.benchmark_value_noise_seed,
        benchmark_decision_noise_model=args.benchmark_decision_noise_model,
        benchmark_decision_noise_std=args.benchmark_decision_noise_std,
        benchmark_decision_noise_seed=args.benchmark_decision_noise_seed,
        phase3_smoke_budget_profile=args.phase3_smoke_budget_profile,
        phase3_policy_profile=args.phase3_policy_profile,
        generic_adapt_budget_profile=args.generic_adapt_budget_profile,
        generic_adapt_smoke_budget_profile=args.generic_adapt_smoke_budget_profile,
        phase3_policy_json=args.phase3_policy_json,
        phase2_novelty_mode=args.phase2_novelty_mode,
        phase3_adapt_parallel_gradient_workers=args.phase3_adapt_parallel_gradient_workers,
        phase3_adapt_beam_parent_workers=args.phase3_adapt_beam_parent_workers,
        generic_adapt_runtime_split_mode=args.generic_adapt_runtime_split_mode,
        generic_adapt_runtime_split_symmetry_policy=args.generic_adapt_runtime_split_symmetry_policy,
        generic_adapt_runtime_split_max_subset_size=args.generic_adapt_runtime_split_max_subset_size,
        shared_pauli_pool_mode=args.shared_pauli_pool_mode,
        shared_pauli_pool_symmetry_policy=args.shared_pauli_pool_symmetry_policy,
        shared_pauli_pool_max_subset_size=args.shared_pauli_pool_max_subset_size,
        hh_pos_geo_position_policy=args.hh_pos_geo_position_policy,
        disable_resource_guards=bool(args.disable_resource_guards),
        hea_reps=args.hea_reps,
        hea_maxiter=args.hea_maxiter,
        calibration_profile=args.calibration_profile,
        paper_i_cutoff_ladder_stage=args.paper_i_cutoff_ladder_stage,
        paper_i_ladder_case_ids=tuple(args.paper_i_ladder_case_ids or ()),
        paper_i_ladder_allow_ref5=bool(args.paper_i_ladder_allow_ref5),
        paper_i_ladder_escalation_reason=args.paper_i_ladder_escalation_reason,
        paper_i_ladder_benchmarks_only=bool(args.paper_i_ladder_benchmarks_only),
        selected_logical_route=str(args.selected_logical_route or "standard"),
        selected_logical_source_json=str(args.selected_logical_source_json or ""),
        selected_logical_transfer_mode=str(args.selected_logical_transfer_mode or "exact_match_v1"),
        selected_logical_supported_algorithms_only=bool(args.selected_logical_supported_algorithms_only),
        smoke_case_ids=tuple(args.smoke_case_ids or ()),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
