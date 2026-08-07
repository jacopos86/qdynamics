#!/usr/bin/env python3
"""Paper-II pinned Qiskit-community comparator dispatch.

This module is the single benchmark-package boundary for the Qiskit-native
comparators defined for Paper II.  Keep Qiskit implementation details behind
``pipelines.exact_bench.qiskit_community_dynamics_adapter`` and keep AP
McLachlan controller internals out of this file.
"""

from __future__ import annotations

from pathlib import Path

from pipelines.time_dynamics.tables.dynamics_benchmark_contract import DynamicsBenchmarkCase, DynamicsBenchmarkRow

QISKIT_TROTTER_QRTE_ALGORITHM = "dyn_qiskit_trotter_qrte"
QISKIT_PVQD_ALGORITHM = "dyn_qiskit_pvqd"
QISKIT_VARQRTE_ALGORITHM = "dyn_qiskit_varqrte"

QISKIT_COMMUNITY_GENERIC_COMPARATOR_ALGORITHMS: tuple[str, ...] = (
    QISKIT_TROTTER_QRTE_ALGORITHM,
    QISKIT_PVQD_ALGORITHM,
    QISKIT_VARQRTE_ALGORITHM,
)

QISKIT_COMMUNITY_RUNNER_MODULE_BY_ALGORITHM: dict[str, str] = {
    algorithm_id: "pipelines.time_dynamics.benchmarks.qiskit_native"
    for algorithm_id in QISKIT_COMMUNITY_GENERIC_COMPARATOR_ALGORITHMS
}

QISKIT_COMMUNITY_METHOD_LABELS: dict[str, str] = {
    QISKIT_TROTTER_QRTE_ALGORITHM: "Qiskit-community TrotterQRTE dynamics",
    QISKIT_PVQD_ALGORITHM: "Qiskit-community PVQD dynamics",
    QISKIT_VARQRTE_ALGORITHM: "Qiskit-community VarQRTE dynamics",
}

QISKIT_COMMUNITY_TABLE_LABELS: dict[str, str] = {
    QISKIT_TROTTER_QRTE_ALGORITHM: "Qiskit TrotterQRTE",
    QISKIT_PVQD_ALGORITHM: "Qiskit PVQD",
    QISKIT_VARQRTE_ALGORITHM: "Qiskit VarQRTE",
}


def has_qiskit_community_runner(algorithm_id: str) -> bool:
    return str(algorithm_id) in QISKIT_COMMUNITY_GENERIC_COMPARATOR_ALGORITHMS


def run_benchmark_row(
    *,
    case: DynamicsBenchmarkCase,
    algorithm_id: str,
    output_dir: Path,
) -> DynamicsBenchmarkRow:
    if not has_qiskit_community_runner(algorithm_id):
        raise KeyError(str(algorithm_id))

    from pipelines.time_dynamics.benchmarks import common

    return common.run_qiskit_community_comparator_row(
        case=case,
        algorithm_id=str(algorithm_id),
        output_dir=Path(output_dir),
    )


__all__ = [
    "QISKIT_COMMUNITY_GENERIC_COMPARATOR_ALGORITHMS",
    "QISKIT_COMMUNITY_METHOD_LABELS",
    "QISKIT_COMMUNITY_RUNNER_MODULE_BY_ALGORITHM",
    "QISKIT_COMMUNITY_TABLE_LABELS",
    "QISKIT_PVQD_ALGORITHM",
    "QISKIT_TROTTER_QRTE_ALGORITHM",
    "QISKIT_VARQRTE_ALGORITHM",
    "has_qiskit_community_runner",
    "run_benchmark_row",
]
