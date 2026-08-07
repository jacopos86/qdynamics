#!/usr/bin/env python3
"""Internal source-role metadata for Paper-I exact-bench comparators.

This module is intentionally metadata-only.  It does not run comparators,
change optimizer/selector behavior, or promote manuscript/table artifacts.  The
fields emitted here are for agents and table-support tooling to distinguish:

* the surface that generated a displayed row;
* any external/library/reference surface available for parity;
* whether parity has actually been run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

ExecutionSurfaceRole = Literal[
    "primary_execution_surface",
    "parity_surface",
    "provenance_only_reference",
]

PAPER_I_STATIC_CONTRACT_ID = "paper_i_static_table_i_canonical_suite_v1"
PAPER_I_EXTERNAL_FIRST_SLICE_CONTRACT_ID = "paper_i_external_hubbard_L2_first_slice_only_v1"
PAPER_I_EXTERNAL_TETRIS_HUBBARD_L2_PARAMETERIZED_CONTRACT_ID = (
    "paper_i_external_tetris_hubbard_L2_parameterized_cases_v1"
)
PAPER_I_TARGET_PROFILE = "paper_i_phys_v1"
PAPER_I_TAU_PHYS = 2.0e-4


@dataclass(frozen=True)
class ComparatorSourceProfile:
    algorithm_id: str
    display_label: str
    algorithm_origin: str
    execution_surface: str
    execution_surface_role: ExecutionSurfaceRole
    external_reference_status: ExecutionSurfaceRole | None = None
    external_reference_id: str | None = None
    external_reference_url: str | None = None
    external_reference_requested_ref: str | None = None
    external_reference_license_status: str | None = None
    external_reference_adapter_status: str | None = None
    external_reference_cache_root: str | None = None
    repo_local_fallback_used: bool = False
    parity_status: str = "not_run"
    parity_scope: str | None = None
    parity_reference_algorithm_id: str | None = None
    same_suite_contract_id: str = PAPER_I_STATIC_CONTRACT_ID
    paper_i_target_profile: str = PAPER_I_TARGET_PROFILE
    paper_i_tau_phys: float = PAPER_I_TAU_PHYS
    exact_reference_role: str = "reporting_only_after_optimization"
    compiled_cost_source_kind: str = "not_applicable"
    first_hit_or_terminal_role: str = "terminal_row"
    promotable_to_manuscript: bool = False


_QISKIT_URL = "https://qiskit-community.github.io/qiskit-algorithms/"
_CEO_URL = "https://github.com/mafaldaramoa/ceo-adapt-vqe"
_HRGRIMSL_ADAPT_URL = "https://github.com/hrgrimsl/adapt"
_JORDANOV_QEB_URL = "https://github.com/JordanovSJ/VQE"
_OPENVQE_URL = "https://github.com/OpenVQE/OpenVQE"
_OVERLAP_URL = "https://www.nature.com/articles/s42005-023-01312-y"

COMPARATOR_SOURCE_PROFILES: dict[str, ComparatorSourceProfile] = {
    "static_hea_qiskit_vqe": ComparatorSourceProfile(
        algorithm_id="static_hea_qiskit_vqe",
        display_label="Qiskit hardware-efficient ansatz VQE",
        algorithm_origin="external_fixed_ansatz_qiskit_hea",
        execution_surface="qiskit_circuit_statevector_ansatz_with_repo_vqe_optimizer",
        execution_surface_role="primary_execution_surface",
        external_reference_status="primary_execution_surface",
        external_reference_id="qiskit_circuit_statevector_transpile",
        external_reference_url=_QISKIT_URL,
        external_reference_adapter_status="implemented_qiskit_hea_adapter_exact_bench_only",
        external_reference_license_status="dependency_license_managed_by_environment",
        parity_status="not_applicable_qiskit_primary_surface_no_pairwise_parity_required",
        parity_scope="ansatz_shape_statevector_energy_and_compiled_cost_smoke_tests",
        compiled_cost_source_kind="qiskit_terminal_transpile_sidecar",
        first_hit_or_terminal_role="terminal_fixed_ansatz_row",
    ),
    "static_family_informed_vqe": ComparatorSourceProfile(
        algorithm_id="static_family_informed_vqe",
        display_label="Family-informed fixed VQE",
        algorithm_origin="benchmark_local_fixed_ansatz_statevector_vqe",
        execution_surface="repo_local_fixed_ansatz_statevector_vqe",
        execution_surface_role="primary_execution_surface",
        external_reference_status="parity_surface",
        external_reference_id="qiskit_fixed_ansatz_evaluator_candidate",
        external_reference_url=_QISKIT_URL,
        external_reference_adapter_status="parity_adapter_pending",
        external_reference_license_status="dependency_license_managed_by_environment",
        repo_local_fallback_used=True,
        parity_status="not_run_qiskit_fixed_ansatz_parity_pending",
        parity_scope="fixed_generator_order_parameter_count_statevector_energy_state_overlap_compiled_cost",
        compiled_cost_source_kind="qiskit_terminal_transpile_sidecar",
        first_hit_or_terminal_role="terminal_fixed_ansatz_row",
    ),
    "static_full_meta_append_adapt_vqe": ComparatorSourceProfile(
        algorithm_id="static_full_meta_append_adapt_vqe",
        display_label="Append-only ADAPT-VQE (local full_meta)",
        algorithm_origin="benchmark_local_statevector_adapt_variant",
        execution_surface="repo_local_full_meta_append_only_statevector_adapt",
        execution_surface_role="primary_execution_surface",
        external_reference_status="parity_surface",
        external_reference_id="qiskit_algorithms_adaptvqe_full_meta",
        external_reference_url=_QISKIT_URL,
        external_reference_adapter_status="implemented_reference_row_parity_sidecar_pending",
        external_reference_license_status="dependency_license_managed_by_environment",
        repo_local_fallback_used=True,
        parity_status="not_run_qiskit_adaptvqe_parity_pending",
        parity_scope=(
            "undrained_operator_pool_with_replacement_selected_generators_powell_refit_"
            "energy_trajectory_final_state_fixed_horizon_compiled_cost"
        ),
        parity_reference_algorithm_id="static_qiskit_adapt_vqe",
        compiled_cost_source_kind="qiskit_terminal_transpile_sidecar",
        first_hit_or_terminal_role="first_hit_or_terminal_adapt_row",
    ),
    "static_qiskit_adapt_vqe": ComparatorSourceProfile(
        algorithm_id="static_qiskit_adapt_vqe",
        display_label="Qiskit AdaptVQE append-only ADAPT reference",
        algorithm_origin="qiskit_algorithms_adaptvqe_full_meta_exact_bench",
        execution_surface="qiskit_algorithms_adaptvqe_full_meta_sparse_pauli_ops",
        execution_surface_role="primary_execution_surface",
        external_reference_status="primary_execution_surface",
        external_reference_id="qiskit_algorithms_adaptvqe",
        external_reference_url=_QISKIT_URL,
        external_reference_adapter_status="implemented_reference_row_exact_bench_only",
        external_reference_license_status="dependency_license_managed_by_environment",
        parity_status="not_run_local_append_only_parity_pending",
        parity_scope="operator_pool_selected_generators_energy_trajectory_final_state_target_hit_stop_reason_compiled_cost",
        parity_reference_algorithm_id="static_full_meta_append_adapt_vqe",
        compiled_cost_source_kind="qiskit_terminal_transpile_sidecar",
        first_hit_or_terminal_role="terminal_library_adapt_reference_row",
    ),
    "static_qubit_qeb_adapt_vqe": ComparatorSourceProfile(
        algorithm_id="static_qubit_qeb_adapt_vqe",
        display_label="Qubit/QEB-ADAPT-VQE",
        algorithm_origin="benchmark_local_statevector_adapt_variant",
        execution_surface="repo_local_qeb_singles_doubles_statevector_adapt",
        execution_surface_role="primary_execution_surface",
        external_reference_status="parity_surface",
        external_reference_id="jordanovsj_vqe_or_openvqe_qeb_candidate",
        external_reference_url=f"{_JORDANOV_QEB_URL}; {_OPENVQE_URL}",
        external_reference_adapter_status="parity_adapter_pending",
        external_reference_license_status="not_checked_adapter_pending",
        repo_local_fallback_used=True,
        parity_status="not_run_external_qeb_conformance_pending",
        parity_scope="qeb_pool_construction_mapped_pauli_labels_selected_excitations_energy_state_overlap",
        compiled_cost_source_kind="qiskit_terminal_transpile_sidecar",
        first_hit_or_terminal_role="first_hit_or_terminal_adapt_row",
    ),
    "static_tetris_qubit_adapt_vqe": ComparatorSourceProfile(
        algorithm_id="static_tetris_qubit_adapt_vqe",
        display_label="TETRIS-ADAPT-VQE",
        algorithm_origin="benchmark_local_statevector_adapt_variant",
        execution_surface="repo_local_full_meta_tetris_style_statevector_adapt",
        execution_surface_role="primary_execution_surface",
        external_reference_status="parity_surface",
        external_reference_id="ceo_adapt_vqe_tetris_public_code_parameterized_hubbard_L2",
        external_reference_url=_CEO_URL,
        external_reference_adapter_status="implemented_hubbard_L2_tetris_public_code_parameterized_cases",
        external_reference_license_status="license_files_recorded_when_checkout_available",
        repo_local_fallback_used=True,
        parity_status="not_run_external_tetris_parameterized_conformance_pending",
        parity_scope="parameterized_hubbard_L2_batch_membership_disjoint_support_energy_optimizer_budget_compiled_cost",
        parity_reference_algorithm_id="static_tetris_adapt_phase3",
        compiled_cost_source_kind="qiskit_terminal_transpile_sidecar",
        first_hit_or_terminal_role="first_hit_or_terminal_adapt_row",
    ),
    "static_geo_adapt_vqe": ComparatorSourceProfile(
        algorithm_id="static_geo_adapt_vqe",
        display_label="Geo-ADAPT-VQE",
        algorithm_origin="benchmark_local_statevector_adapt_variant",
        execution_surface="repo_local_full_meta_projected_fs_geo_statevector_adapt",
        execution_surface_role="primary_execution_surface",
        external_reference_status="parity_surface",
        external_reference_id="independent_dense_statevector_geo_formula",
        external_reference_adapter_status="internal_conformance_formula_pending",
        external_reference_license_status="not_applicable_internal_formula",
        repo_local_fallback_used=True,
        parity_status="not_run_internal_geo_metric_conformance_pending",
        parity_scope=(
            "full_pool_tangent_metric_natural_gradient_score_post_selection_immediate_repeat_skip_"
            "powell_refit_fixed_horizon_energy_trajectory"
        ),
        compiled_cost_source_kind="qiskit_terminal_transpile_sidecar",
        first_hit_or_terminal_role="first_hit_or_terminal_adapt_row",
    ),
    "static_pos_geo_adapt_vqe": ComparatorSourceProfile(
        algorithm_id="static_pos_geo_adapt_vqe",
        display_label="Pos-Geo-ADAPT-VQE",
        algorithm_origin="benchmark_local_statevector_adapt_variant",
        execution_surface="repo_local_full_meta_position_optimized_geo_statevector_adapt",
        execution_surface_role="primary_execution_surface",
        external_reference_status="parity_surface",
        external_reference_id="independent_dense_statevector_geo_formula",
        external_reference_adapter_status="internal_conformance_formula_pending",
        external_reference_license_status="not_applicable_internal_formula",
        repo_local_fallback_used=True,
        parity_status="not_run_internal_pos_geo_conformance_pending",
        parity_scope="tangent_metric_natural_gradient_score_insertion_position_refit_energy_trajectory",
        compiled_cost_source_kind="qiskit_terminal_transpile_sidecar",
        first_hit_or_terminal_role="first_hit_or_terminal_adapt_row",
    ),
    "static_geo_qubit_adapt_vqe": ComparatorSourceProfile(
        algorithm_id="static_geo_qubit_adapt_vqe",
        display_label="legacy geometry diagnostic (removed from Table I)",
        algorithm_origin="benchmark_local_statevector_adapt_variant",
        execution_surface="repo_local_full_meta_legacy_geo_style_metric_selector",
        execution_surface_role="primary_execution_surface",
        external_reference_status="provenance_only_reference",
        external_reference_id="geo_adapt_literature_only_legacy_diagnostic",
        external_reference_adapter_status="no_external_adapter_legacy_diagnostic",
        external_reference_license_status="not_applicable_no_adapter",
        repo_local_fallback_used=True,
        parity_status="not_applicable_legacy_diagnostic_not_table_i_displayed",
        parity_scope="legacy_diagnostic_only",
        compiled_cost_source_kind="qiskit_terminal_transpile_sidecar",
        first_hit_or_terminal_role="first_hit_or_terminal_adapt_row",
    ),
    "static_geo_qeb_adapt_vqe": ComparatorSourceProfile(
        algorithm_id="static_geo_qeb_adapt_vqe",
        display_label="Geo-ADAPT-VQE (QEB reference)",
        algorithm_origin="benchmark_local_statevector_adapt_variant",
        execution_surface="repo_local_qeb_projected_fs_geo_statevector_adapt",
        execution_surface_role="primary_execution_surface",
        external_reference_status="parity_surface",
        external_reference_id="independent_dense_statevector_geo_formula",
        external_reference_adapter_status="internal_conformance_formula_pending",
        external_reference_license_status="not_applicable_internal_formula",
        repo_local_fallback_used=True,
        parity_status="not_run_internal_geo_qeb_conformance_pending",
        parity_scope="qeb_pool_tangent_metric_natural_gradient_score_selected_candidate_state_overlap",
        compiled_cost_source_kind="qiskit_terminal_transpile_sidecar",
        first_hit_or_terminal_role="first_hit_or_terminal_adapt_row",
    ),
    "static_family_native_adapt_phase3": ComparatorSourceProfile(
        algorithm_id="static_family_native_adapt_phase3",
        display_label="Resource-adaptive Phase3 ADAPT scaffold/controller",
        algorithm_origin="repo_native_phase3_static_adapt_snake",
        execution_surface="repo_native_phase3_static_adapt_controller",
        execution_surface_role="primary_execution_surface",
        external_reference_status=None,
        external_reference_id=None,
        external_reference_adapter_status="no_external_reference_expected_for_snake",
        external_reference_license_status="not_applicable_repo_native_method",
        parity_status="not_applicable_repo_native_no_external_reference_expected",
        parity_scope="route_identity_and_qiskit_resource_sidecars_only",
        compiled_cost_source_kind="qiskit_resource_sidecar_when_available",
        first_hit_or_terminal_role="first_hit_resource_extraction",
    ),
    "static_ceo_adapt_phase3": ComparatorSourceProfile(
        algorithm_id="static_ceo_adapt_phase3",
        display_label="CEO ADAPT benchmark",
        algorithm_origin="external_public_code_ceo_adapt_vqe",
        execution_surface="external_public_code_ceo_adapt_vqe_hubbard_L2_first_slice",
        execution_surface_role="primary_execution_surface",
        external_reference_status="primary_execution_surface",
        external_reference_id="ceo_adapt_vqe",
        external_reference_url=_CEO_URL,
        external_reference_adapter_status="implemented_hubbard_L2_public_code_first_slice_only",
        external_reference_license_status="license_files_recorded_when_checkout_available",
        parity_status="first_slice_conformance_only_not_full_paper_i_suite",
        parity_scope="hubbard_L2_public_code_first_slice_energy_selected_indices_gradients",
        same_suite_contract_id=PAPER_I_EXTERNAL_FIRST_SLICE_CONTRACT_ID,
        compiled_cost_source_kind="not_available_external_first_slice",
        first_hit_or_terminal_role="external_first_slice_terminal_row",
    ),
    "static_tetris_adapt_phase3": ComparatorSourceProfile(
        algorithm_id="static_tetris_adapt_phase3",
        display_label="TETRIS-ADAPT benchmark",
        algorithm_origin="external_public_code_ceo_adapt_vqe_tetris",
        execution_surface="external_public_code_ceo_adapt_vqe_tetris_hubbard_L2_parameterized_cases",
        execution_surface_role="primary_execution_surface",
        external_reference_status="primary_execution_surface",
        external_reference_id="ceo_adapt_vqe",
        external_reference_url=_CEO_URL,
        external_reference_adapter_status="implemented_hubbard_L2_tetris_public_code_parameterized_cases",
        external_reference_license_status="license_files_recorded_when_checkout_available",
        parity_status="parameterized_hubbard_L2_diagnostic_not_full_paper_i_suite",
        parity_scope="hubbard_L2_public_code_parameterized_tetris_batches_energy_selected_indices_gradients",
        parity_reference_algorithm_id="static_tetris_qubit_adapt_vqe",
        same_suite_contract_id=PAPER_I_EXTERNAL_TETRIS_HUBBARD_L2_PARAMETERIZED_CONTRACT_ID,
        compiled_cost_source_kind="not_available_external_parameterized_diagnostic",
        first_hit_or_terminal_role="external_parameterized_terminal_row",
    ),
    "static_overlap_adapt_phase3": ComparatorSourceProfile(
        algorithm_id="static_overlap_adapt_phase3",
        display_label="Overlap-ADAPT benchmark",
        algorithm_origin="external_overlap_adapt_request_only",
        execution_surface="no_runnable_execution_surface_request_only",
        execution_surface_role="provenance_only_reference",
        external_reference_status="provenance_only_reference",
        external_reference_id="overlap_adapt_vqe_request",
        external_reference_url=_OVERLAP_URL,
        external_reference_adapter_status="request_only_no_adapter",
        external_reference_license_status="not_available_request_only",
        parity_status="not_runnable_request_only",
        parity_scope="provenance_only_no_promoted_evidence",
        same_suite_contract_id=PAPER_I_EXTERNAL_FIRST_SLICE_CONTRACT_ID,
        compiled_cost_source_kind="not_available_request_only",
        first_hit_or_terminal_role="not_available_request_only",
    ),
    "static_append_original_adapt_provenance": ComparatorSourceProfile(
        algorithm_id="static_append_original_adapt_provenance",
        display_label="Original ADAPT provenance reference",
        algorithm_origin="provenance_only_original_adapt_source",
        execution_surface="no_displayed_row_provenance_only",
        execution_surface_role="provenance_only_reference",
        external_reference_status="provenance_only_reference",
        external_reference_id="hrgrimsl_adapt",
        external_reference_url=_HRGRIMSL_ADAPT_URL,
        external_reference_adapter_status="not_integrated_paper_i_full_suite",
        external_reference_license_status="not_checked_not_integrated",
        parity_status="not_run_provenance_only",
        parity_scope="citation_and_implementation_guidance_only",
    ),
}

REQUIRED_COMPARATOR_SOURCE_FIELDS: tuple[str, ...] = (
    "algorithm_origin",
    "execution_surface",
    "execution_surface_role",
    "external_reference_status",
    "external_reference_id",
    "external_reference_url",
    "external_reference_requested_ref",
    "external_reference_resolved_commit",
    "external_reference_license_status",
    "external_reference_adapter_status",
    "external_reference_cache_root",
    "repo_local_fallback_used",
    "parity_status",
    "parity_scope",
    "parity_reference_algorithm_id",
    "parity_reference_artifact",
    "parity_energy_abs_delta",
    "parity_state_infidelity",
    "parity_selected_generators_match",
    "parity_compiled_cost_match",
    "same_suite_contract_id",
    "paper_i_target_profile",
    "paper_i_tau_phys",
    "cutoff_pair",
    "exact_reference_role",
    "compiled_cost_source_kind",
    "first_hit_or_terminal_role",
    "promotable_to_manuscript",
)


def comparator_source_profile(algorithm_id: str) -> ComparatorSourceProfile:
    """Return the source profile for a known Paper-I comparator algorithm."""
    key = str(algorithm_id)
    try:
        return COMPARATOR_SOURCE_PROFILES[key]
    except KeyError as exc:
        known = ", ".join(sorted(COMPARATOR_SOURCE_PROFILES))
        raise ValueError(f"No comparator source profile for {algorithm_id!r}. Known profiles: {known}") from exc


def maybe_comparator_source_profile(algorithm_id: str) -> ComparatorSourceProfile | None:
    """Return a profile if this algorithm is a Paper-I comparator, otherwise ``None``."""
    return COMPARATOR_SOURCE_PROFILES.get(str(algorithm_id))


def comparator_source_fields(
    algorithm_id: str,
    *,
    runner_module: str | None = None,
    external_reference_resolved_commit: str | None = None,
    external_reference_cache_root: str | Path | None = None,
    external_reference_license_status: str | None = None,
    parity_reference_artifact: str | Path | None = None,
    parity_energy_abs_delta: float | None = None,
    parity_state_infidelity: float | None = None,
    parity_selected_generators_match: bool | None = None,
    parity_compiled_cost_match: bool | None = None,
    cutoff_pair: Mapping[str, Any] | None = None,
    promotable_to_manuscript: bool | None = None,
) -> dict[str, Any]:
    """Flatten comparator source metadata for row/result manifests.

    All optional parity quantities default to ``None``.  A caller should only set
    them after a real parity/conformance check has run.
    """
    profile = comparator_source_profile(algorithm_id)
    payload: dict[str, Any] = asdict(profile)
    payload.pop("display_label", None)
    payload["display_label"] = profile.display_label
    payload["runner_module"] = runner_module
    payload["external_reference_resolved_commit"] = external_reference_resolved_commit
    if external_reference_cache_root is not None:
        payload["external_reference_cache_root"] = str(external_reference_cache_root)
    if external_reference_license_status is not None:
        payload["external_reference_license_status"] = str(external_reference_license_status)
    payload["parity_reference_artifact"] = None if parity_reference_artifact is None else str(parity_reference_artifact)
    payload["parity_energy_abs_delta"] = parity_energy_abs_delta
    payload["parity_state_infidelity"] = parity_state_infidelity
    payload["parity_selected_generators_match"] = parity_selected_generators_match
    payload["parity_compiled_cost_match"] = parity_compiled_cost_match
    payload["cutoff_pair"] = None if cutoff_pair is None else dict(cutoff_pair)
    if promotable_to_manuscript is not None:
        payload["promotable_to_manuscript"] = bool(promotable_to_manuscript)
    for field in REQUIRED_COMPARATOR_SOURCE_FIELDS:
        payload.setdefault(field, None)
    return payload


__all__ = [
    "COMPARATOR_SOURCE_PROFILES",
    "ComparatorSourceProfile",
    "ExecutionSurfaceRole",
    "PAPER_I_EXTERNAL_FIRST_SLICE_CONTRACT_ID",
    "PAPER_I_EXTERNAL_TETRIS_HUBBARD_L2_PARAMETERIZED_CONTRACT_ID",
    "PAPER_I_STATIC_CONTRACT_ID",
    "PAPER_I_TARGET_PROFILE",
    "PAPER_I_TAU_PHYS",
    "REQUIRED_COMPARATOR_SOURCE_FIELDS",
    "comparator_source_fields",
    "comparator_source_profile",
    "maybe_comparator_source_profile",
]
