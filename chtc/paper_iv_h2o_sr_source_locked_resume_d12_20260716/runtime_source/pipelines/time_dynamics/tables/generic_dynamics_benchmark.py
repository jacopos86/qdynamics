#!/usr/bin/env python3
"""Generic dynamics benchmark manifest/dispatch surface.

This is CHTC-facing glue.  It records which Hamiltonian-family/dynamics
algorithm combinations are safe to submit and dispatches either:

* the existing HH t=8 benchmark wrappers, or
* generic fixture-backed rows for exact/fixed-McLachlan, product-formula/Suzuki,
  qDRIFT, fixed-pVQD, and AVQDS-style tangent comparators.

All other non-HH comparator rows remain skipped until concrete runners exist.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import subprocess  # compatibility monkeypatch surface for HH legacy wrapper tests
import sys
from pathlib import Path
from typing import Sequence

from pipelines.exact_bench.benchmark_algorithm_registry import (
    default_benchmark_algorithms,
    evaluate_algorithm_for_family,
)
from pipelines.reporting.benchmark_manifest import BenchmarkJob, write_manifest_bundle
from pipelines.static_adapt.builders.problem_registry import available_problem_keys
from pipelines.time_dynamics.benchmarks import legacy_native
from pipelines.time_dynamics.benchmarks import registry as benchmark_registry
from pipelines.time_dynamics.benchmarks.common import write_skipped_generic_dynamics_row
from pipelines.time_dynamics.benchmarks.registry import GENERIC_CONTROLLER_ABLATION_MATRIX_ALGORITHM
from pipelines.time_dynamics.tables.dynamics_benchmark_contract import DynamicsBenchmarkCase
from pipelines.time_dynamics.tables.dynamics_benchmark_contract import (
    DYNAMICS_CLASS_TUNING_DEFAULT_SOURCE,
    DYNAMICS_HH_LEGACY_TUNING_SOURCE,
    DYNAMICS_SKIPPED_TUNING_SOURCE,
    build_dynamics_tuning_provenance,
)
from pipelines.time_dynamics.tables.generic_dynamics_cases import (
    cases_for_family,
)
from pipelines.time_dynamics.tables.table_lock_contract import (
    table_lock_provenance_for_case,
    with_class_settings_lock_manifest,
)

# Compatibility aliases; ownership of the HH map lives in benchmarks.legacy_native.
_HH_DYNAMICS_CASES = legacy_native.HH_DYNAMICS_CASES
_HH_MODULE_MAP = legacy_native.HH_MODULE_MAP


def _families_from_args(values: Sequence[str] | None) -> tuple[str, ...]:
    return tuple(values) if values else available_problem_keys()


def _qiskit_qubit_cap_arg(value: str) -> int | None:
    text = str(value).strip().lower()
    if text in {"none", "null", "uncapped", "no_cap"}:
        return None
    return int(text)


def _dynamics_algorithms_from_args(values: Sequence[str] | None):
    algs = default_benchmark_algorithms(domain="dynamics")
    if not values:
        return algs
    wanted = set(str(v) for v in values)
    return tuple(alg for alg in algs if alg.algorithm_id in wanted)


def _case_ids_for_family(
    family: str,
    *,
    case_manifest: Path | str | None = None,
) -> tuple[str, ...]:
    cases = cases_for_family(family, case_manifest)
    if cases:
        return tuple(case.case_id for case in cases)
    if family == "hh":
        return _HH_DYNAMICS_CASES
    return (f"{family}_dynamics_default",)


def _generic_case_lookup(
    family: str,
    *,
    case_manifest: Path | str | None = None,
) -> dict[str, DynamicsBenchmarkCase]:
    return {case.case_id: case for case in cases_for_family(family, case_manifest)}


def _placeholder_case(*, family: str, case_id: str) -> DynamicsBenchmarkCase:
    return DynamicsBenchmarkCase(
        case_id=str(case_id),
        family=str(family),
        table_class="unclassified",
        artifact_json="",
        description="placeholder for skipped manifest row with no explicit fixture case",
    )


def _hh_legacy_case(*, case_id: str) -> DynamicsBenchmarkCase:
    return DynamicsBenchmarkCase(
        case_id=str(case_id),
        family="hh",
        table_class="hybrid",
        tuning_class="hybrid",
        artifact_json="",
        description="HH legacy dynamics wrapper; class-tuning provenance is not locked",
    )


def _job_tuning_provenance(
    *,
    family: str,
    case_id: str,
    algorithm_id: str,
    status: str,
    case: DynamicsBenchmarkCase | None,
) -> dict:
    if case is not None:
        source = (
            DYNAMICS_CLASS_TUNING_DEFAULT_SOURCE
            if str(status) == "runnable"
            else DYNAMICS_SKIPPED_TUNING_SOURCE
        )
        source_case = case
    elif str(family) == "hh":
        source = DYNAMICS_HH_LEGACY_TUNING_SOURCE
        source_case = _hh_legacy_case(case_id=case_id)
    else:
        source = DYNAMICS_SKIPPED_TUNING_SOURCE
        source_case = _placeholder_case(family=family, case_id=case_id)
    return build_dynamics_tuning_provenance(
        case=source_case,
        algorithm_id=algorithm_id,
        settings_kind="benchmark_job",
        settings_source=source,
        locked=False,
    )


def _command_for_job(
    *,
    family: str,
    case_id: str,
    algorithm_id: str,
    output_dir: Path,
    case_manifest: Path | str | None = None,
    class_settings_manifest: Path | str | None = None,
    require_locked_class_settings: bool = False,
    qiskit_dynamics_mode: str = "off",
    qiskit_qubit_cap: int | None = 12,
    qiskit_export_circuits: bool = False,
) -> tuple[str, ...]:
    command = [
        sys.executable,
        "-m",
        "pipelines.time_dynamics.tables.generic_dynamics_benchmark",
        "--run-single",
        "--family",
        family,
        "--case-id",
        case_id,
        "--algorithm-id",
        algorithm_id,
        "--output-dir",
        str(output_dir),
    ]
    if case_manifest is not None:
        command.extend(["--case-manifest", str(case_manifest)])
    if class_settings_manifest is not None:
        command.extend(["--class-settings-manifest", str(class_settings_manifest)])
    if bool(require_locked_class_settings):
        command.append("--require-locked-class-settings")
    if str(qiskit_dynamics_mode) != "off":
        command.extend(["--qiskit-dynamics-mode", str(qiskit_dynamics_mode)])
        command.extend([
            "--qiskit-qubit-cap",
            "none" if qiskit_qubit_cap is None else str(int(qiskit_qubit_cap)),
        ])
        if bool(qiskit_export_circuits):
            command.append("--qiskit-export-circuits")
    return tuple(command)


def _with_qiskit_dynamics_metadata(
    case: DynamicsBenchmarkCase,
    *,
    qiskit_dynamics_mode: str = "off",
    qiskit_qubit_cap: int | None = 12,
    qiskit_export_circuits: bool = False,
) -> DynamicsBenchmarkCase:
    mode = str(qiskit_dynamics_mode)
    if mode == "off":
        return case
    if mode not in {"parity", "parity_required"}:
        raise ValueError("Qiskit dynamics support is parity-only; use off, parity, or parity_required")
    metadata = dict(case.metadata or {})
    metadata["qiskit_dynamics"] = {
        "mode": mode,
        "qubit_cap": None if qiskit_qubit_cap is None else int(qiskit_qubit_cap),
        "export_circuits": bool(qiskit_export_circuits),
        "time_segmentation": "match_native_interval",
    }
    metadata["qiskit_dynamics_mode"] = mode
    if qiskit_qubit_cap is not None:
        metadata["qiskit_qubit_cap"] = int(qiskit_qubit_cap)
    metadata["qiskit_export_circuits"] = bool(qiskit_export_circuits)
    return replace(case, metadata=metadata)


def _generic_dispatch_label(algorithm_id: str) -> str:
    return benchmark_registry.dispatch_label(algorithm_id)


def build_dynamics_jobs(
    *,
    output_root: Path,
    families: Sequence[str] | None = None,
    algorithm_ids: Sequence[str] | None = None,
    include_skipped: bool = True,
    case_manifest: Path | str | None = None,
    class_settings_manifest: Path | str | None = None,
    require_locked_class_settings: bool = False,
    qiskit_dynamics_mode: str = "off",
    qiskit_qubit_cap: int | None = 12,
    qiskit_export_circuits: bool = False,
) -> list[BenchmarkJob]:
    fams = _families_from_args(families)
    algs = _dynamics_algorithms_from_args(algorithm_ids)
    jobs: list[BenchmarkJob] = []
    for family in fams:
        case_lookup = _generic_case_lookup(family, case_manifest=case_manifest)
        for case_id in _case_ids_for_family(family, case_manifest=case_manifest):
            case = case_lookup.get(case_id)
            for alg in algs:
                app = evaluate_algorithm_for_family(alg, family)
                status = app.status
                reason = app.reason
                command: tuple[str, ...] = ()
                job_output = output_root / "dynamics" / family / case_id / alg.algorithm_id
                generic_runner_available = case is not None and benchmark_registry.supports_isolated_benchmark(
                    alg.algorithm_id
                )
                hh_runner_available = (
                    family == "hh"
                    and case is None
                    and legacy_native.has_legacy_runner(alg.algorithm_id)
                )
                if status == "runnable" and not (hh_runner_available or generic_runner_available):
                    status = "skipped_no_runner"
                    reason = "no concrete dynamics dispatch mapping for this family/algorithm"
                if status == "runnable":
                    command = _command_for_job(
                        family=family,
                        case_id=case_id,
                        algorithm_id=alg.algorithm_id,
                        output_dir=job_output,
                        case_manifest=case_manifest,
                        class_settings_manifest=class_settings_manifest,
                        require_locked_class_settings=bool(require_locked_class_settings),
                        qiskit_dynamics_mode=str(qiskit_dynamics_mode),
                        qiskit_qubit_cap=qiskit_qubit_cap,
                        qiskit_export_circuits=bool(qiskit_export_circuits),
                    )
                if status != "runnable" and not include_skipped:
                    continue
                metadata = {
                    "required_pool_key": app.required_pool_key,
                    "resolved_pool_key": app.resolved_pool_key,
                }
                tuning = _job_tuning_provenance(
                    family=family,
                    case_id=case_id,
                    algorithm_id=alg.algorithm_id,
                    status=status,
                    case=case,
                )
                metadata.update(dict(tuning))
                metadata["tuning_provenance"] = dict(tuning)
                if case is not None:
                    metadata.update(table_lock_provenance_for_case(case))
                    if str(qiskit_dynamics_mode) != "off":
                        metadata["qiskit_dynamics"] = {
                            "mode": str(qiskit_dynamics_mode),
                            "qubit_cap": None if qiskit_qubit_cap is None else int(qiskit_qubit_cap),
                            "export_circuits": bool(qiskit_export_circuits),
                            "time_segmentation": "match_native_interval",
                            "support_scope": "benchmark_local_parity_only",
                        }
                    if class_settings_manifest is not None:
                        metadata["class_settings_lock_manifest"] = str(class_settings_manifest)
                        metadata["require_locked_class_settings"] = bool(require_locked_class_settings)
                    metadata.update(
                        {
                            "table_class": case.table_class,
                            "artifact_json": case.artifact_json,
                            "dispatch": _generic_dispatch_label(alg.algorithm_id)
                            if status == "runnable" and generic_runner_available
                            else "generic_case_skipped",
                        }
                    )
                elif family == "hh":
                    metadata["dispatch"] = (
                        "hh_legacy_wrapper"
                        if status == "runnable" and hh_runner_available
                        else "hh_case_skipped"
                    )
                else:
                    metadata["dispatch"] = "skipped_no_explicit_case"
                jobs.append(
                    BenchmarkJob(
                        job_id=f"dynamics__{family}__{case_id}__{alg.algorithm_id}",
                        domain="dynamics",
                        family=family,
                        case_id=case_id,
                        algorithm_id=alg.algorithm_id,
                        status=status,
                        reason=reason,
                        command=command,
                        output_dir=str(job_output),
                        runner_module=app.runner_module,
                        qpu_faithful=app.qpu_faithful,
                        exact_assisted=app.exact_assisted,
                        diagnostic=app.diagnostic,
                        hamiltonian_generic=app.hamiltonian_generic,
                        resources={"request_cpus": 1, "request_memory": "4GB", "request_disk": "4GB"},
                        metadata=metadata,
                    )
                )
    return jobs


def _write_skip_payload(
    *,
    family: str,
    case_id: str,
    algorithm_id: str,
    output_dir: Path,
    status: str,
    reason: str,
) -> dict:
    payload = {
        "schema": "generic_dynamics_benchmark_single_v1",
        "family": family,
        "case_id": case_id,
        "algorithm_id": algorithm_id,
        "status": status,
        "reason": reason,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "skip.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _row_payload(row) -> dict:
    if hasattr(row, "to_dict"):
        return row.to_dict()
    return dict(row)


def run_single(
    *,
    family: str,
    case_id: str,
    algorithm_id: str,
    output_dir: Path,
    case_manifest: Path | str | None = None,
    class_settings_manifest: Path | str | None = None,
    require_locked_class_settings: bool = False,
    qiskit_dynamics_mode: str = "off",
    qiskit_qubit_cap: int | None = 12,
    qiskit_export_circuits: bool = False,
) -> dict:
    app = evaluate_algorithm_for_family(algorithm_id, family)
    if app.status != "runnable":
        case = _generic_case_lookup(family, case_manifest=case_manifest).get(
            case_id,
            _placeholder_case(family=family, case_id=case_id),
        )
        if family != "hh" or case.artifact_json:
            case = with_class_settings_lock_manifest(
                case,
                manifest_path=class_settings_manifest,
                require_locked=bool(require_locked_class_settings),
            )
            case = _with_qiskit_dynamics_metadata(
                case,
                qiskit_dynamics_mode=str(qiskit_dynamics_mode),
                qiskit_qubit_cap=qiskit_qubit_cap,
                qiskit_export_circuits=bool(qiskit_export_circuits),
            )
            row = write_skipped_generic_dynamics_row(
                case=case,
                algorithm_id=algorithm_id,
                output_dir=output_dir,
                status=app.status,
                reason=app.reason,
            )
            return _row_payload(row)
        return _write_skip_payload(
            family=family,
            case_id=case_id,
            algorithm_id=algorithm_id,
            output_dir=output_dir,
            status=app.status,
            reason=app.reason,
        )

    case_lookup = _generic_case_lookup(family, case_manifest=case_manifest)
    case = case_lookup.get(case_id)
    if case is not None:
        if not benchmark_registry.supports_isolated_benchmark(algorithm_id):
            case = with_class_settings_lock_manifest(
                case,
                manifest_path=class_settings_manifest,
                require_locked=bool(require_locked_class_settings),
            )
            case = _with_qiskit_dynamics_metadata(
                case,
                qiskit_dynamics_mode=str(qiskit_dynamics_mode),
                qiskit_qubit_cap=qiskit_qubit_cap,
                qiskit_export_circuits=bool(qiskit_export_circuits),
            )
            row = write_skipped_generic_dynamics_row(
                case=case,
                algorithm_id=algorithm_id,
                output_dir=output_dir,
                status="skipped_no_runner",
                reason="no concrete generic dynamics row runner is wired for this algorithm",
            )
            return _row_payload(row)
        case = with_class_settings_lock_manifest(
            case,
            manifest_path=class_settings_manifest,
            require_locked=bool(require_locked_class_settings),
        )
        case = _with_qiskit_dynamics_metadata(
            case,
            qiskit_dynamics_mode=str(qiskit_dynamics_mode),
            qiskit_qubit_cap=qiskit_qubit_cap,
            qiskit_export_circuits=bool(qiskit_export_circuits),
        )
        result = benchmark_registry.run_isolated_benchmark(
            case=case,
            algorithm_id=algorithm_id,
            output_dir=Path(output_dir),
        )
        return _row_payload(result)

    if family != "hh":
        row = write_skipped_generic_dynamics_row(
            case=_placeholder_case(family=family, case_id=case_id),
            algorithm_id=algorithm_id,
            output_dir=output_dir,
            status="skipped_no_runner",
            reason="no explicit generic dynamics case is available for this family/case_id",
        )
        return _row_payload(row)

    if not legacy_native.has_legacy_runner(algorithm_id):
        return _write_skip_payload(
            family=family,
            case_id=case_id,
            algorithm_id=algorithm_id,
            output_dir=output_dir,
            status="skipped_no_runner",
            reason="no concrete HH dynamics module mapping",
        )
    # Preserve the historical monkeypatch surface where callers patch
    # generic_dynamics_benchmark.subprocess before invoking HH dispatch.
    legacy_native.subprocess = subprocess
    return legacy_native.run_legacy_hh_wrapper(
        case_id=case_id,
        algorithm_id=algorithm_id,
        output_dir=Path(output_dir),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or run generic dynamics benchmark jobs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--family", action="append", dest="families", default=None)
    parser.add_argument("--algorithm-id", action="append", dest="algorithm_ids", default=None)
    parser.add_argument("--case-id", type=str, default=None)
    parser.add_argument("--case-manifest", type=Path, default=None)
    parser.add_argument("--class-settings-manifest", type=Path, default=None)
    parser.add_argument("--require-locked-class-settings", action="store_true")
    parser.add_argument("--qiskit-dynamics-mode", choices=("off", "parity", "parity_required"), default="off")
    parser.add_argument("--qiskit-qubit-cap", type=_qiskit_qubit_cap_arg, default=12)
    parser.add_argument("--qiskit-export-circuits", action="store_true", default=False)
    parser.add_argument("--include-skipped", action="store_true", default=False)
    parser.add_argument("--run-single", action="store_true", default=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.run_single:
        if not args.families or len(args.families) != 1:
            raise SystemExit("--run-single requires exactly one --family")
        if not args.algorithm_ids or len(args.algorithm_ids) != 1:
            raise SystemExit("--run-single requires exactly one --algorithm-id")
        if not args.case_id:
            raise SystemExit("--run-single requires --case-id")
        result = run_single(
            family=str(args.families[0]),
            case_id=str(args.case_id),
            algorithm_id=str(args.algorithm_ids[0]),
            output_dir=Path(args.output_dir),
            case_manifest=args.case_manifest,
            class_settings_manifest=args.class_settings_manifest,
            require_locked_class_settings=bool(args.require_locked_class_settings),
            qiskit_dynamics_mode=str(args.qiskit_dynamics_mode),
            qiskit_qubit_cap=args.qiskit_qubit_cap,
            qiskit_export_circuits=bool(args.qiskit_export_circuits),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    jobs = build_dynamics_jobs(
        output_root=Path(args.output_dir),
        families=args.families,
        algorithm_ids=args.algorithm_ids,
        include_skipped=bool(args.include_skipped),
        case_manifest=args.case_manifest,
        class_settings_manifest=args.class_settings_manifest,
        require_locked_class_settings=bool(args.require_locked_class_settings),
        qiskit_dynamics_mode=str(args.qiskit_dynamics_mode),
        qiskit_qubit_cap=args.qiskit_qubit_cap,
        qiskit_export_circuits=bool(args.qiskit_export_circuits),
    )
    summary = write_manifest_bundle(output_dir=args.output_dir, jobs=jobs, label="generic_dynamics_benchmark")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
