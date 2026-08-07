#!/usr/bin/env python3
"""Table-I static benchmark manifest surface.

Table I in ``MATH/paper_details/main_condensed`` is the manuscript-facing
static Pareto suite for currently honest exact-bench rows: hardware-efficient
VQE, family-informed fixed VQE, a benchmark-local full_meta append-only ADAPT comparator,
benchmark-local Qubit/QEB-ADAPT, TETRIS, full-meta Geo-ADAPT, and SNAKE.  The Qiskit
AdaptVQE row is retained as an explicit nondefault reference/parity target.  CEO-style/public-code competitors must be supplied
by benchmark-local library/public-code adapters, not by Phase3-emulated policy
toggles.  This module is deliberately manifest-only glue;
individual row execution still routes through ``generic_static_benchmark`` so the
main static ADAPT pipeline remains untouched.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from pipelines.exact_bench.generic_static_benchmark import build_static_jobs
from pipelines.exact_bench.table_i_canonical_cases import table_i_canonical_families, table_i_suite_profile
from pipelines.reporting.benchmark_manifest import BenchmarkJob, write_manifest_bundle
from pipelines.static_adapt.builders.problem_registry import available_problem_keys

TABLE_I_STATIC_ALGORITHM_IDS: tuple[str, ...] = (
    "static_hea_qiskit_vqe",
    "static_family_informed_vqe",
    "static_full_meta_append_adapt_vqe",
    "static_qubit_qeb_adapt_vqe",
    "static_tetris_qubit_adapt_vqe",
    "static_geo_adapt_vqe",
    "static_family_native_adapt_phase3",
)
TABLE_I_STATIC_BENCHMARK_ALGORITHM_IDS: tuple[str, ...] = tuple(
    algorithm_id
    for algorithm_id in TABLE_I_STATIC_ALGORITHM_IDS
    if algorithm_id != "static_family_native_adapt_phase3"
)

TABLE_I_METHOD_LABELS: dict[str, str] = {
    "static_hea_qiskit_vqe": "HEA VQE",
    "static_family_informed_vqe": "family-informed VQE",
    "static_full_meta_append_adapt_vqe": "append-only ADAPT",
    "static_qubit_qeb_adapt_vqe": "Qubit/QEB-ADAPT-VQE",
    "static_tetris_qubit_adapt_vqe": "TETRIS-ADAPT-VQE",
    "static_geo_adapt_vqe": "Geo-ADAPT-VQE",
    "static_family_native_adapt_phase3": "SNAKE",
}

TABLE_I_NONDEFAULT_METHOD_LABELS: dict[str, str] = {
    "static_qiskit_adapt_vqe": "Qiskit AdaptVQE append-only ADAPT reference",
    "static_geo_qubit_adapt_vqe": "legacy geometry diagnostic (removed from default Table I)",
    "static_geo_qeb_adapt_vqe": "Geo-ADAPT-VQE (QEB reference)",
    "static_pos_geo_adapt_vqe": "Pos-Geo-ADAPT-VQE diagnostic",
}


def table_i_method_label(algorithm_id: str) -> str:
    """Return a readable Table-I/legacy label for an algorithm id."""
    key = str(algorithm_id)
    return TABLE_I_METHOD_LABELS.get(key, TABLE_I_NONDEFAULT_METHOD_LABELS.get(key, key))


TABLE_I_CLASS_BY_FAMILY: dict[str, str] = {
    "hubbard": "fermionic",
    "ionic_hubbard": "fermionic",
    "extended_hubbard": "fermionic",
    "ttprime_hubbard": "fermionic",
    "spinless_tv": "fermionic",
    "bose_hubbard": "bosonic",
    "harmonic_kerr_chain": "bosonic",
    "hh": "fermion-boson",
    "spin_boson": "bosonic",
    "molecular_vibronic_h2": "fermion-boson",
}


def table_i_families(profile: str | None = None) -> tuple[str, ...]:
    """Return currently registered Hamiltonian families mapped into Table I."""
    registered = set(available_problem_keys())
    return tuple(family for family in table_i_canonical_families(profile) if family in registered)


def build_table_i_static_jobs(
    *,
    output_root: Path,
    families: Sequence[str] | None = None,
    algorithm_ids: Sequence[str] | None = None,
    include_skipped: bool = True,
) -> list[BenchmarkJob]:
    """Build the manuscript Table-I static job matrix."""
    fams = tuple(families) if families is not None else table_i_families()
    algs = tuple(algorithm_ids) if algorithm_ids is not None else TABLE_I_STATIC_ALGORITHM_IDS
    return build_static_jobs(
        output_root=Path(output_root),
        families=fams,
        algorithm_ids=algs,
        include_skipped=include_skipped,
    )


def summarize_table_i_jobs(jobs: Sequence[BenchmarkJob], *, suite_profile: str | None = None) -> dict[str, Any]:
    """Return compact coverage counts by manuscript class and method."""
    suite_profile_key = table_i_suite_profile(suite_profile)
    status_by_method: dict[str, Counter[str]] = defaultdict(Counter)
    status_by_class_method: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    cases_by_method: dict[str, set[str]] = defaultdict(set)
    job_algorithm_ids: list[str] = []
    job_families: list[str] = []
    for job in jobs:
        if job.algorithm_id not in job_algorithm_ids:
            job_algorithm_ids.append(job.algorithm_id)
        if job.family not in job_families:
            job_families.append(job.family)
        method = table_i_method_label(job.algorithm_id)
        klass = TABLE_I_CLASS_BY_FAMILY.get(job.family, "unmapped")
        status_by_method[method][job.status] += 1
        status_by_class_method[klass][method][job.status] += 1
        if job.status == "runnable":
            cases_by_method[method].add(f"{job.family}/{job.case_id}")
    return {
        "schema": "table_i_static_benchmark_summary_v1",
        "table_label": "main_condensed Table I / tab:static_claims",
        "suite_profile": suite_profile_key,
        "algorithm_ids": list(job_algorithm_ids),
        "catalog_algorithm_ids": list(TABLE_I_STATIC_ALGORITHM_IDS),
        "method_labels": {algorithm_id: table_i_method_label(algorithm_id) for algorithm_id in job_algorithm_ids},
        "catalog_method_labels": dict(TABLE_I_METHOD_LABELS),
        "nondefault_method_labels": dict(TABLE_I_NONDEFAULT_METHOD_LABELS),
        "families": list(job_families),
        "catalog_families": list(table_i_families(suite_profile_key)),
        "status_by_method": {method: dict(counter) for method, counter in status_by_method.items()},
        "status_by_class_method": {
            klass: {method: dict(counter) for method, counter in by_method.items()}
            for klass, by_method in status_by_class_method.items()
        },
        "runnable_case_count_by_method": {
            method: len(cases) for method, cases in cases_by_method.items()
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the manuscript Table-I static benchmark manifest.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--family", action="append", dest="families", default=None)
    parser.add_argument("--algorithm-id", action="append", dest="algorithm_ids", default=None)
    parser.add_argument(
        "--benchmarks-only",
        action="store_true",
        default=False,
        help="Omit the SNAKE/project-controller row and emit comparator benchmark rows only.",
    )
    parser.add_argument("--include-skipped", action="store_true", default=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    algorithm_ids = (
        TABLE_I_STATIC_BENCHMARK_ALGORITHM_IDS
        if bool(args.benchmarks_only) and args.algorithm_ids is None
        else args.algorithm_ids
    )
    jobs = build_table_i_static_jobs(
        output_root=Path(args.output_dir),
        families=args.families,
        algorithm_ids=algorithm_ids,
        include_skipped=bool(args.include_skipped),
    )
    summary = write_manifest_bundle(output_dir=args.output_dir, jobs=jobs, label="table_i_static_benchmark")
    table_summary = summarize_table_i_jobs(jobs)
    path = Path(args.output_dir) / "table_i_summary.json"
    path.write_text(json.dumps(table_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["paths"]["table_i_summary_json"] = str(path)
    summary["table_i"] = table_summary
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
