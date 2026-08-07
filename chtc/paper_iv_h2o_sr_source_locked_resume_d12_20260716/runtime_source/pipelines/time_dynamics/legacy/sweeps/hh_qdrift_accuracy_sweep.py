#!/usr/bin/env python3
"""Simulation-only qDRIFT accuracy audit for the HH L=2 t=8 benchmark anchor.

This is not a replacement for the hardware-cost qDRIFT benchmark row.  It is a
paper-safety audit: hold the validated controller source, exact reporting grid,
and qDRIFT implementation fixed while sweeping stochastic sample budgets and RNG
seeds.  The sweep deliberately skips Qiskit transpilation so larger sample counts
can be checked without creating huge compiled circuits.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.legacy.hh_benchmarks import hh_qdrift_benchmark as qdrift


SCHEMA_VERSION = "hh_qdrift_accuracy_sweep_v1"
DEFAULT_SAMPLE_COUNTS = (16, 64, 256)
DEFAULT_RNG_SEEDS = (1, 7, 13)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    return qdrift._jsonable(value)  # noqa: SLF001 - audit wrapper over benchmark-local internals.


def _write_json(path: Path, payload: Any) -> Path:
    return qdrift._write_json(path, payload)  # noqa: SLF001 - same artifact convention as benchmark row.


def _finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _parse_int_list(values: Sequence[int] | None, default: Sequence[int]) -> tuple[int, ...]:
    raw = tuple(default if values is None else values)
    parsed = tuple(int(value) for value in raw)
    if not parsed:
        raise ValueError("at least one integer value is required")
    if any(value <= 0 for value in parsed):
        raise ValueError("all integer sweep values must be positive")
    return tuple(dict.fromkeys(parsed))


def _method_run_path(output_dir: Path, *, case_id: str, samples_per_interval: int, rng_seed: int) -> Path:
    method_id = qdrift.method_id_for_config(
        samples_per_interval=int(samples_per_interval),
        rng_seed=int(rng_seed),
    )
    return Path(output_dir) / "runs" / f"{case_id}__{method_id}.json"


def _base_case_from_args(args: argparse.Namespace) -> qdrift.QDriftBenchmarkCase:
    case = qdrift._case_by_id(str(args.case_id))  # noqa: SLF001 - uses benchmark-local case registry.
    return replace(
        case,
        controller_json=Path(args.controller_json) if args.controller_json is not None else case.controller_json,
        source_pdf=Path(args.source_pdf) if args.source_pdf is not None else case.source_pdf,
        trotter_steps=int(args.trotter_steps) if args.trotter_steps is not None else case.trotter_steps,
    )


def _load_source_context(case: qdrift.QDriftBenchmarkCase) -> dict[str, Any]:
    overlay = qdrift.overlay
    source_payload = overlay._load_source_payload(Path(case.controller_json))  # noqa: SLF001
    source_rows = overlay._state_sample_rows(source_payload)  # noqa: SLF001
    times = overlay._source_times(source_payload, source_rows)  # noqa: SLF001
    _ = overlay._uniform_dt(times, int(case.trotter_steps))  # noqa: SLF001 - validates the grid.
    context = overlay._rebuild_context(source_payload)  # noqa: SLF001
    drive_t0 = float((context.drive_profile or {}).get("t0", 0.0))
    drive_sampling = str((context.drive_profile or {}).get("time_sampling", "midpoint"))
    physical_times = overlay._source_physical_times(  # noqa: SLF001
        source_rows,
        fallback_drive_t0=float(drive_t0),
    )
    exact_energy = [
        qdrift._required_finite_float(  # noqa: SLF001
            row.get("energy_total_exact"),
            field=f"source_rows[{idx}].energy_total_exact",
        )
        for idx, row in enumerate(source_rows)
    ]
    return {
        "source_payload": source_payload,
        "source_rows": source_rows,
        "times": times,
        "context": context,
        "drive_t0": drive_t0,
        "drive_sampling": drive_sampling,
        "physical_times": physical_times,
        "exact_energy": exact_energy,
    }


def _row_from_simulation(
    *,
    case: qdrift.QDriftBenchmarkCase,
    samples_per_interval: int,
    rng_seed: int,
    simulation: qdrift.QDriftSimulationResult,
    run_json: Path,
) -> dict[str, Any]:
    summary = dict(simulation.summary)
    lambdas = [float(row.get("lambda", 0.0)) for row in simulation.intervals]
    taus = [float(row.get("tau", 0.0)) for row in simulation.intervals]
    sampled_rotation_count = int(len(simulation.intervals) * int(samples_per_interval))
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": str(case.case_id),
        "method_id": str(simulation.method),
        "method_kind": qdrift.METHOD_KIND,
        "status": "ok",
        "audit_scope": "simulation_only_accuracy_no_transpile",
        "compiled_costs_included": False,
        "randomization_family": qdrift.RANDOMIZATION_FAMILY,
        "samples_per_interval": int(samples_per_interval),
        "rng_seed": int(rng_seed),
        "trotter_steps": int(case.trotter_steps),
        "num_times": int(len(simulation.trajectory)),
        "final_energy_total": _finite_or_none(summary.get("final_energy_total")),
        "final_energy_total_exact": _finite_or_none(summary.get("final_energy_total_exact")),
        "final_abs_energy_total_error": _finite_or_none(summary.get("final_abs_energy_total_error")),
        "mean_abs_energy_total_error": _finite_or_none(summary.get("mean_abs_energy_total_error")),
        "max_abs_energy_total_error": _finite_or_none(summary.get("max_abs_energy_total_error")),
        "sampled_rotation_count": sampled_rotation_count,
        "state_at_time_sampled_rotation_count": int(samples_per_interval),
        "full_horizon_sampled_rotation_count": sampled_rotation_count,
        "lambda_mean": float(np.mean(lambdas)) if lambdas else None,
        "lambda_max": float(np.max(lambdas)) if lambdas else None,
        "tau_mean": float(np.mean(taus)) if taus else None,
        "tau_max": float(np.max(taus)) if taus else None,
        "exact_fields_reporting_only": True,
        "controller_decisions_modified": False,
        "artifact_run_json": str(run_json),
    }


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_sample: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        if str(row.get("status", "")).lower() != "ok":
            continue
        samples = int(row["samples_per_interval"])
        by_sample.setdefault(samples, []).append(row)

    sample_summaries: list[dict[str, Any]] = []
    for samples, group in sorted(by_sample.items()):
        mean_errors = np.asarray([float(row["mean_abs_energy_total_error"]) for row in group], dtype=float)
        final_errors = np.asarray([float(row["final_abs_energy_total_error"]) for row in group], dtype=float)
        best = min(group, key=lambda row: float(row["mean_abs_energy_total_error"]))
        sample_summaries.append(
            {
                "samples_per_interval": int(samples),
                "seed_count": int(len(group)),
                "mean_error_mean": float(np.mean(mean_errors)),
                "mean_error_median": float(np.median(mean_errors)),
                "mean_error_min": float(np.min(mean_errors)),
                "mean_error_max": float(np.max(mean_errors)),
                "final_error_mean": float(np.mean(final_errors)),
                "final_error_median": float(np.median(final_errors)),
                "final_error_min": float(np.min(final_errors)),
                "final_error_max": float(np.max(final_errors)),
                "best_seed_by_mean_error": int(best["rng_seed"]),
                "best_method_id_by_mean_error": str(best["method_id"]),
                "best_mean_abs_energy_total_error": float(best["mean_abs_energy_total_error"]),
                "sampled_rotation_count": int(best["sampled_rotation_count"]),
            }
        )

    ok_rows = [row for row in rows if str(row.get("status", "")).lower() == "ok"]
    best_overall = (
        min(ok_rows, key=lambda row: float(row["mean_abs_energy_total_error"])) if ok_rows else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": _now_utc(),
        "row_count": int(len(rows)),
        "status_counts": dict(Counter(str(row.get("status", "unknown")) for row in rows)),
        "sample_summaries": sample_summaries,
        "best_overall_by_mean_error": dict(best_overall) if best_overall is not None else None,
    }


def run_accuracy_sweep(
    *,
    output_dir: str | Path,
    case: qdrift.QDriftBenchmarkCase,
    sample_counts: Sequence[int] = DEFAULT_SAMPLE_COUNTS,
    rng_seeds: Sequence[int] = DEFAULT_RNG_SEEDS,
    command: str = "",
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    sample_counts = _parse_int_list(sample_counts, DEFAULT_SAMPLE_COUNTS)
    rng_seeds = _parse_int_list(rng_seeds, DEFAULT_RNG_SEEDS)
    source = _load_source_context(case)
    context = source["context"]

    rows: list[dict[str, Any]] = []
    for samples in sample_counts:
        for seed in rng_seeds:
            method_id = qdrift.method_id_for_config(samples_per_interval=int(samples), rng_seed=int(seed))
            simulation = qdrift._simulate_qdrift(  # noqa: SLF001 - this audit intentionally exercises the benchmark kernel.
                psi_initial=context.psi_initial,
                times=source["times"],
                exact_energy_total=source["exact_energy"],
                observation_physical_times=source["physical_times"],
                ordered_labels_exyz=context.ordered_labels_exyz,
                coeff_map_exyz=context.coeff_map_exyz,
                hmat_static=context.hmat,
                drive_coeff_provider_exyz=context.drive_coeff_provider_exyz,
                drive_t0=float(source["drive_t0"]),
                drive_time_sampling=str(source["drive_sampling"]),
                nq=int(context.nq),
                samples_per_interval=int(samples),
                rng_seed=int(seed),
                method_id=method_id,
            )
            run_json = _method_run_path(
                root,
                case_id=str(case.case_id),
                samples_per_interval=int(samples),
                rng_seed=int(seed),
            )
            run_payload = {
                "schema_version": SCHEMA_VERSION,
                "generated_utc": _now_utc(),
                "case": _jsonable(case),
                "method_id": method_id,
                "audit_scope": "simulation_only_accuracy_no_transpile",
                "trajectory": simulation.trajectory,
                "summary": simulation.summary,
                "qdrift_intervals": simulation.intervals,
            }
            _write_json(run_json, run_payload)
            rows.append(
                _row_from_simulation(
                    case=case,
                    samples_per_interval=int(samples),
                    rng_seed=int(seed),
                    simulation=simulation,
                    run_json=run_json,
                )
            )

    rows_json = root / "qdrift_accuracy_sweep_rows.json"
    summary_json = root / "qdrift_accuracy_sweep_summary.json"
    manifest_json = root / "qdrift_accuracy_sweep_manifest.json"
    summary = _summarize_rows(rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": _now_utc(),
        "benchmark": "hh_qdrift_accuracy_sweep",
        "case": _jsonable(case),
        "sample_counts": list(sample_counts),
        "rng_seeds": list(rng_seeds),
        "command": command,
        "contract": {
            "compiled_costs_included": False,
            "exact_reference_policy": "reporting_only_after_trajectory_energy",
            "controller_decisions_modified": False,
            "purpose": "diagnose whether the poor s16 qDRIFT row is under-sampling or implementation pathology",
        },
        "paths": {
            "manifest_json": str(manifest_json),
            "rows_json": str(rows_json),
            "summary_json": str(summary_json),
            "runs_dir": str(root / "runs"),
        },
    }
    _write_json(rows_json, rows)
    _write_json(summary_json, summary)
    _write_json(manifest_json, manifest)
    return {"manifest": manifest, "rows": rows, "summary": summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a simulation-only HH qDRIFT sample/seed accuracy sweep.")
    parser.add_argument("--case-id", type=str, default=qdrift.DEFAULT_CASE_ID)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--controller-json", type=Path, default=None)
    parser.add_argument("--source-pdf", type=Path, default=None)
    parser.add_argument("--trotter-steps", type=int, default=None)
    parser.add_argument("--samples-per-interval", type=int, action="append", default=None)
    parser.add_argument("--rng-seed", type=int, action="append", default=None)
    return parser


def _command_from_argv(argv: Sequence[str] | None) -> str:
    if argv is None:
        return " ".join([sys.executable, "-m", "pipelines.time_dynamics.legacy.sweeps.hh_qdrift_accuracy_sweep", *sys.argv[1:]])
    return " ".join(["python", "-m", "pipelines.time_dynamics.legacy.sweeps.hh_qdrift_accuracy_sweep", *map(str, argv)])


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    case = _base_case_from_args(args)
    result = run_accuracy_sweep(
        output_dir=Path(args.output_dir),
        case=case,
        sample_counts=_parse_int_list(args.samples_per_interval, DEFAULT_SAMPLE_COUNTS),
        rng_seeds=_parse_int_list(args.rng_seed, DEFAULT_RNG_SEEDS),
        command=_command_from_argv(argv),
    )
    best = result["summary"].get("best_overall_by_mean_error") or {}
    print(f"manifest_json={Path(args.output_dir) / 'qdrift_accuracy_sweep_manifest.json'}")
    print(f"rows_json={Path(args.output_dir) / 'qdrift_accuracy_sweep_rows.json'}")
    print(f"summary_json={Path(args.output_dir) / 'qdrift_accuracy_sweep_summary.json'}")
    print(f"row_count={len(result['rows'])}")
    if best:
        print(f"best_method_id_by_mean_error={best.get('method_id')}")
        print(f"best_mean_abs_energy_total_error={best.get('mean_abs_energy_total_error')}")
        print(f"best_final_abs_energy_total_error={best.get('final_abs_energy_total_error')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
