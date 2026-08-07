#!/usr/bin/env python3
"""Isolated registry for non-HH generic dynamics benchmark runners."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from pipelines.time_dynamics.benchmarks.common import (
    NATIVE_GENERIC_COMPARATOR_ALGORITHMS,
    REALTIME_GENERIC_DYNAMICS_ALGORITHMS,
    SUPPORTED_GENERIC_DYNAMICS_ALGORITHMS,
    write_skipped_generic_dynamics_row,
)
from pipelines.time_dynamics.benchmarks.legacy_native import (
    LEGACY_NATIVE_BENCHMARK_RUNNER_MODULE_BY_ALGORITHM,
    LEGACY_NATIVE_GENERIC_ALGORITHMS,
)
from pipelines.time_dynamics.benchmarks.avqds_tetris import AVQDS_TETRIS_ALGORITHM_ID
from pipelines.time_dynamics.benchmarks.qiskit_native import (
    QISKIT_COMMUNITY_GENERIC_COMPARATOR_ALGORITHMS,
    QISKIT_COMMUNITY_RUNNER_MODULE_BY_ALGORITHM,
)
from pipelines.time_dynamics.tables.dynamics_benchmark_contract import DynamicsBenchmarkCase, DynamicsBenchmarkRow

GENERIC_CONTROLLER_ABLATION_MATRIX_ALGORITHM = "dyn_controller_ablation_matrix"
GENERIC_CONTROLLER_FULL_ALGORITHM = "dyn_controller_full"
GENERIC_CONTROLLER_ABLATION_ROW_ALGORITHMS: tuple[str, ...] = (
    GENERIC_CONTROLLER_FULL_ALGORITHM,
)
_CONTROLLER_ABLATION_MODULE = "pipelines.time_dynamics.benchmarks.legacy_native"
_AVQDS_TETRIS_MODULE = "pipelines.time_dynamics.benchmarks.avqds_tetris"

_RUNNER_MODULES: dict[str, str] = {
    **LEGACY_NATIVE_BENCHMARK_RUNNER_MODULE_BY_ALGORITHM,
    **QISKIT_COMMUNITY_RUNNER_MODULE_BY_ALGORITHM,
    AVQDS_TETRIS_ALGORITHM_ID: _AVQDS_TETRIS_MODULE,
    GENERIC_CONTROLLER_ABLATION_MATRIX_ALGORITHM: _CONTROLLER_ABLATION_MODULE,
    GENERIC_CONTROLLER_FULL_ALGORITHM: _CONTROLLER_ABLATION_MODULE,
}
SUPPORTED_ISOLATED_BENCHMARK_ALGORITHMS: tuple[str, ...] = tuple(_RUNNER_MODULES)


def runner_module_for_algorithm(algorithm_id: str) -> str | None:
    return _RUNNER_MODULES.get(str(algorithm_id))


def supports_isolated_benchmark(algorithm_id: str) -> bool:
    return str(algorithm_id) in _RUNNER_MODULES


def dispatch_label(algorithm_id: str) -> str:
    algorithm = str(algorithm_id)
    if algorithm in REALTIME_GENERIC_DYNAMICS_ALGORITHMS:
        return "generic_realtime_neutral"
    if algorithm in NATIVE_GENERIC_COMPARATOR_ALGORITHMS:
        return "generic_repo_native_comparator"
    if algorithm in QISKIT_COMMUNITY_GENERIC_COMPARATOR_ALGORITHMS:
        return "generic_qiskit_community_comparator"
    if algorithm == GENERIC_CONTROLLER_ABLATION_MATRIX_ALGORITHM:
        return "generic_controller_ablation_matrix"
    if algorithm in GENERIC_CONTROLLER_ABLATION_ROW_ALGORITHMS:
        return "generic_controller_ablation_row"
    return "generic_case_skipped"


def _controller_ablation_variant_id_for_algorithm(module: Any, algorithm_id: str) -> str:
    for variant in module.controller_ablation_variants():
        if str(variant.algorithm_id) == str(algorithm_id):
            return str(variant.variant_id)
    raise KeyError(str(algorithm_id))


def _load_runner(algorithm_id: str) -> Any:
    module_name = runner_module_for_algorithm(algorithm_id)
    if module_name is None:
        raise KeyError(str(algorithm_id))
    return importlib.import_module(module_name)


def run_isolated_benchmark(
    *,
    case: DynamicsBenchmarkCase,
    algorithm_id: str,
    output_dir: Path,
) -> DynamicsBenchmarkRow | dict[str, Any]:
    algorithm = str(algorithm_id)
    if algorithm not in _RUNNER_MODULES:
        return write_skipped_generic_dynamics_row(
            case=case,
            algorithm_id=algorithm,
            output_dir=Path(output_dir),
            status="skipped_no_runner",
            reason="generic dynamics isolated benchmark registry has no runner for this algorithm",
        )
    module = _load_runner(algorithm)
    if algorithm == GENERIC_CONTROLLER_ABLATION_MATRIX_ALGORITHM:
        return module.run_generic_controller_ablation_matrix(case=case, output_dir=Path(output_dir))
    if algorithm in GENERIC_CONTROLLER_ABLATION_ROW_ALGORITHMS:
        variant_id = _controller_ablation_variant_id_for_algorithm(module, algorithm)
        return module.run_generic_controller_ablation_row(
            case=case,
            variant_id=variant_id,
            output_dir=Path(output_dir),
        )
    if algorithm in QISKIT_COMMUNITY_GENERIC_COMPARATOR_ALGORITHMS:
        return module.run_benchmark_row(
            case=case,
            algorithm_id=algorithm,
            output_dir=Path(output_dir),
        )
    if algorithm in LEGACY_NATIVE_GENERIC_ALGORITHMS:
        return module.run_benchmark_row(
            case=case,
            algorithm_id=algorithm,
            output_dir=Path(output_dir),
        )
    return module.run_benchmark_row(case=case, output_dir=Path(output_dir))


def run_generic_dynamics_row(
    *,
    case: DynamicsBenchmarkCase,
    algorithm_id: str,
    output_dir: Path,
) -> DynamicsBenchmarkRow:
    result = run_isolated_benchmark(case=case, algorithm_id=algorithm_id, output_dir=Path(output_dir))
    if not isinstance(result, DynamicsBenchmarkRow):
        raise TypeError(f"algorithm {algorithm_id!r} did not return a DynamicsBenchmarkRow")
    return result


__all__ = [
    "GENERIC_CONTROLLER_ABLATION_MATRIX_ALGORITHM",
    "GENERIC_CONTROLLER_ABLATION_ROW_ALGORITHMS",
    "GENERIC_CONTROLLER_FULL_ALGORITHM",
    "NATIVE_GENERIC_COMPARATOR_ALGORITHMS",
    "QISKIT_COMMUNITY_GENERIC_COMPARATOR_ALGORITHMS",
    "REALTIME_GENERIC_DYNAMICS_ALGORITHMS",
    "SUPPORTED_GENERIC_DYNAMICS_ALGORITHMS",
    "SUPPORTED_ISOLATED_BENCHMARK_ALGORITHMS",
    "dispatch_label",
    "runner_module_for_algorithm",
    "run_generic_dynamics_row",
    "run_isolated_benchmark",
    "supports_isolated_benchmark",
]
