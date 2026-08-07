#!/usr/bin/env python3
"""Compatibility shim for generic dynamics benchmark rows.

Implementation ownership lives in :mod:`pipelines.time_dynamics.benchmarks`.
This module re-exports the historical API for existing callers and tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pipelines.time_dynamics.benchmarks import common as _common
from pipelines.time_dynamics.benchmarks import legacy_native as _legacy_native
from pipelines.time_dynamics.benchmarks import registry as _registry
from pipelines.time_dynamics.benchmarks.common import (
    CANDIDATE_POOL_REQUIRED_GENERIC_ALGORITHMS,
    CORRECTNESS_SIDECAR_FILENAMES,
    CORRECTNESS_SIDECAR_KEYS,
    DYNAMICS_TABLE_BUNDLE_SCHEMA,
    NATIVE_RESOURCE_POLICY,
    _active_hamiltonian_terms,
    _assert_native_case_supported,
    _build_layout_for_terms,
    _candidate_pool_completeness,
    _candidate_pool_is_complete,
    _compile_audit_from_resources,
    _compiled_executor_for_terms,
    _copy_theta_by_layout_blocks,
    _energy_from_matrix,
    _exact_states_for_times,
    _exact_step_from_eigendecomp,
    _float_or_none,
    _full_horizon_resources,
    _generic_parameter_manifest,
    _int_or_none,
    _load_runtime_input_for_case,
    _max_or_none,
    _mean_or_none,
    _metadata_float,
    _metadata_int,
    _metadata_optional_int,
    _min_or_none,
    _native_comparator_command,
    _normalize_state,
    _prepare_scaffold_state,
    _row_from_payload,
    _scaffold_resources_for_layouts,
    _sequence_resource_totals,
    _state_diagnostic_row,
    _table_fields,
    _term_label,
    _term_label_set,
    _trajectory_from_states,
    _trajectory_summary,
    _write_json,
    skipped_generic_dynamics_row,
    write_dynamics_row_bundle,
    write_skipped_generic_dynamics_row,
    build_fixed_mclachlan_correctness_sidecar,
)
from pipelines.time_dynamics.benchmarks.legacy_native import PRODUCT_FORMULA_CANDIDATE_ORDERS
from pipelines.time_dynamics.benchmarks.legacy_native import (
    DEFAULT_QDRIFT_RNG_SEED,
    DEFAULT_QDRIFT_SAMPLES_PER_INTERVAL,
)
from pipelines.time_dynamics.benchmarks.registry import (
    NATIVE_GENERIC_COMPARATOR_ALGORITHMS,
    REALTIME_GENERIC_DYNAMICS_ALGORITHMS,
    SUPPORTED_GENERIC_DYNAMICS_ALGORITHMS,
)
from pipelines.time_dynamics.tables.dynamics_benchmark_contract import DynamicsBenchmarkCase, DynamicsBenchmarkRow

# Historical monkeypatch targets.  run_generic_dynamics_row syncs these aliases
# back into the benchmark implementation before dispatching.
load_scaffold_runtime_input = _common.load_scaffold_runtime_input
realtime = _common.realtime

# Historical private helper aliases used by tests/downstream notebooks.
_product_formula_sequence = _legacy_native._product_formula_sequence
_simulate_product_formula_candidate = _legacy_native._simulate_product_formula_candidate
_selection_value = _legacy_native._selection_value
_product_formula_selection_key = _legacy_native._product_formula_selection_key
_build_product_formula_payload = _legacy_native._build_product_formula_payload
_simulate_qdrift = _legacy_native._simulate_qdrift
_build_qdrift_payload = _legacy_native._build_qdrift_payload
_projection_loss_for_state = _legacy_native._projection_loss_for_state
_coordinate_refine_projection = _legacy_native._coordinate_refine_projection
_fit_pvqd_projection_step = _legacy_native._fit_pvqd_projection_step
_build_fixed_pvqd_payload = _legacy_native._build_fixed_pvqd_payload
_candidate_indices_for_adaptive_pvqd = _legacy_native._candidate_indices_for_adaptive_pvqd
_build_adaptive_pvqd_payload = _legacy_native._build_adaptive_pvqd_payload
_solve_avqds_tangent_step = _legacy_native._solve_avqds_tangent_step
_candidate_indices_for_avqds = _legacy_native._candidate_indices_for_avqds
_build_avqds_payload = _legacy_native._build_avqds_payload
_build_avqds_correctness_sidecar = _legacy_native._build_avqds_correctness_sidecar
_solve_avqds_t_target_tangent_step = _legacy_native._solve_avqds_t_target_tangent_step
_build_avqds_t_payload = _legacy_native._build_avqds_t_payload
_build_avqds_t_correctness_sidecar = _legacy_native._build_avqds_t_correctness_sidecar


def _sync_compat_overrides() -> None:
    _common.load_scaffold_runtime_input = load_scaffold_runtime_input
    _common.realtime = realtime


def run_generic_dynamics_row(
    *,
    case: DynamicsBenchmarkCase,
    algorithm_id: str,
    output_dir: Path,
) -> DynamicsBenchmarkRow:
    _sync_compat_overrides()
    return _registry.run_generic_dynamics_row(
        case=case,
        algorithm_id=algorithm_id,
        output_dir=Path(output_dir),
    )


def __getattr__(name: str) -> Any:
    for module in (_common, _registry, _legacy_native):
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(name)


__all__ = [
    "CORRECTNESS_SIDECAR_FILENAMES",
    "CORRECTNESS_SIDECAR_KEYS",
    "DYNAMICS_TABLE_BUNDLE_SCHEMA",
    "NATIVE_GENERIC_COMPARATOR_ALGORITHMS",
    "PRODUCT_FORMULA_CANDIDATE_ORDERS",
    "REALTIME_GENERIC_DYNAMICS_ALGORITHMS",
    "SUPPORTED_GENERIC_DYNAMICS_ALGORITHMS",
    "run_generic_dynamics_row",
    "skipped_generic_dynamics_row",
    "write_dynamics_row_bundle",
    "write_skipped_generic_dynamics_row",
]
