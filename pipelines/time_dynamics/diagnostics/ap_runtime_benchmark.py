"""Deterministic runtime/memory benchmark harness for the active AP-McLachlan route.

This harness measures; it does not propagate.  Every case executes the ordinary
runner ``pipelines.time_dynamics.runners.ap_append_from_adapt_artifact`` through
its own ``main(argv)`` entry point, so there is exactly one propagation engine in
the repository and no benchmark-only scientific path.

The harness makes no quality judgement.  It reports wall time, CPU time, peak
resident memory, worker counts, BLAS thread environment, and a bounded phase
profiling receipt.  Whether a measured speedup or a scientific trajectory is
good enough is the user's decision.

Each case runs in a **fresh subprocess** for two reasons:

* BLAS thread counts are read when the numeric libraries load, so they can only
  be controlled through the environment of a new process;
* peak resident memory is a process high-water mark, so one process per case is
  the only way to attribute it to a case.

Usage
-----

Run a case matrix and write an aggregate report::

    python3 -m pipelines.time_dynamics.diagnostics.ap_runtime_benchmark \
        --matrix-json matrix.json --output-json report.json

The parent process re-invokes this module with ``--single-case-json`` for each
case; that mode is an implementation detail of the harness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.ap_mclachlan.performance import (
    PROFILE_RECEIPT_SCHEMA_V1,
    profiling_session,
)


BENCHMARK_SCHEMA_V1 = "ap_mclachlan_runtime_benchmark_v1"
CASE_RESULT_SCHEMA_V1 = "ap_mclachlan_runtime_benchmark_case_v1"

#: Environment variables that bound BLAS/OpenMP thread pools.  All of them are
#: set explicitly for every case so a measurement is never taken against an
#: unrecorded ambient thread count.
BLAS_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",  # Apple Accelerate
    "NUMEXPR_NUM_THREADS",
)


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------


def _blas_identity() -> dict[str, Any]:
    """Return the BLAS/LAPACK backend NumPy is actually linked against."""

    try:
        config = np.show_config("dicts") or {}
    except Exception:  # pragma: no cover - older NumPy without dict config
        return {"available": False}
    build = dict(config.get("Build Dependencies", {}) or {})
    blas = dict(build.get("blas", {}) or {})
    lapack = dict(build.get("lapack", {}) or {})
    return {
        "available": True,
        "blas_name": blas.get("name"),
        "blas_version": blas.get("version"),
        "lapack_name": lapack.get("name"),
        "lapack_version": lapack.get("version"),
    }


def environment_record() -> dict[str, Any]:
    """Capture everything needed to interpret a timing number later."""

    try:
        import scipy  # noqa: PLC0415 - optional, recorded when present

        scipy_version: str | None = str(scipy.__version__)
    except Exception:
        scipy_version = None
    try:
        import numba  # noqa: PLC0415 - optional compiled-kernel backend

        numba_version: str | None = str(numba.__version__)
    except Exception:
        numba_version = None
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "numpy_version": np.__version__,
        "scipy_version": scipy_version,
        "numba_version": numba_version,
        "blas": _blas_identity(),
        "thread_env": {name: os.environ.get(name) for name in BLAS_THREAD_ENV_VARS},
    }


def _peak_rss_bytes() -> int:
    """Peak resident set size of this process, normalized to bytes.

    ``ru_maxrss`` is bytes on Darwin and kilobytes on Linux.
    """

    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def sha256_of_file(path: str | Path) -> str | None:
    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Case definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkCase:
    """One measured invocation of the ordinary AP-McLachlan runner."""

    case_id: str
    runner_argv: tuple[str, ...]
    #: Value of ``support_patch_scoring_workers`` this case is exercising.  It
    #: must already be present in ``runner_argv``; it is recorded separately so
    #: the report can be pivoted by worker count.
    worker_count: int = 1
    blas_threads: int = 1
    description: str = ""
    artifact_json: str | None = None
    output_json: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "case_id": str(self.case_id),
            "runner_argv": [str(token) for token in self.runner_argv],
            "worker_count": int(self.worker_count),
            "blas_threads": int(self.blas_threads),
            "description": str(self.description),
            "artifact_json": self.artifact_json,
            "output_json": self.output_json,
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "BenchmarkCase":
        return cls(
            case_id=str(payload["case_id"]),
            runner_argv=tuple(str(token) for token in payload["runner_argv"]),
            worker_count=int(payload.get("worker_count", 1)),
            blas_threads=int(payload.get("blas_threads", 1)),
            description=str(payload.get("description", "")),
            artifact_json=payload.get("artifact_json"),
            output_json=payload.get("output_json"),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


# ---------------------------------------------------------------------------
# Single-case execution (child process)
# ---------------------------------------------------------------------------


def run_case_in_process(case: BenchmarkCase, *, profile: bool = True) -> dict[str, Any]:
    """Execute one case through the ordinary runner and measure it.

    Imported lazily so that a parent process listing cases never pays the
    runner's import cost.
    """

    from pipelines.time_dynamics.runners import ap_append_from_adapt_artifact as runner

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    receipt: dict[str, Any] | None = None
    exit_code: int | None = None
    error: str | None = None

    try:
        if profile:
            with profiling_session(label=case.case_id) as profiler:
                exit_code = int(runner.main(list(case.runner_argv)))
            receipt = profiler.receipt()
        else:
            exit_code = int(runner.main(list(case.runner_argv)))
    except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised as data
        error = f"{type(exc).__name__}: {exc}"

    wall_seconds = time.perf_counter() - wall_start
    cpu_seconds = time.process_time() - cpu_start

    output_json = case.output_json
    return {
        "schema": CASE_RESULT_SCHEMA_V1,
        "case": case.to_json_dict(),
        "ok": bool(error is None and exit_code == 0),
        "exit_code": exit_code,
        "error": error,
        "wall_seconds": float(wall_seconds),
        "cpu_seconds": float(cpu_seconds),
        "cpu_utilization": (
            float(cpu_seconds / wall_seconds) if wall_seconds > 0.0 else None
        ),
        "peak_rss_bytes": int(_peak_rss_bytes()),
        "artifact_sha256": (
            sha256_of_file(case.artifact_json) if case.artifact_json else None
        ),
        "output_sha256": sha256_of_file(output_json) if output_json else None,
        "output_bytes": (
            int(Path(output_json).stat().st_size)
            if output_json and Path(output_json).is_file()
            else None
        ),
        "environment": environment_record(),
        "profile_receipt": receipt,
        "profile_receipt_schema": PROFILE_RECEIPT_SCHEMA_V1 if receipt else None,
    }


# ---------------------------------------------------------------------------
# Matrix execution (parent process)
# ---------------------------------------------------------------------------


def _child_env(case: BenchmarkCase) -> dict[str, str]:
    env = dict(os.environ)
    for name in BLAS_THREAD_ENV_VARS:
        env[name] = str(int(case.blas_threads))
    return env


def run_case_in_subprocess(
    case: BenchmarkCase,
    *,
    scratch_dir: Path,
    profile: bool = True,
    timeout_seconds: float | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Run one case in a fresh process so BLAS threads and peak RSS are its own."""

    scratch_dir.mkdir(parents=True, exist_ok=True)
    case_json = scratch_dir / f"case_{case.case_id}.json"
    result_json = scratch_dir / f"result_{case.case_id}.json"
    case_json.write_text(json.dumps(case.to_json_dict(), indent=2), encoding="utf-8")

    argv = [
        sys.executable,
        "-m",
        "pipelines.time_dynamics.diagnostics.ap_runtime_benchmark",
        "--single-case-json",
        str(case_json),
        "--output-json",
        str(result_json),
    ]
    if not profile:
        argv.append("--no-profile")

    started = time.perf_counter()
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        argv,
        cwd=str(repo_root) if repo_root is not None else None,
        env=_child_env(case),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    parent_wall = time.perf_counter() - started

    if result_json.is_file():
        result = json.loads(result_json.read_text(encoding="utf-8"))
    else:
        result = {
            "schema": CASE_RESULT_SCHEMA_V1,
            "case": case.to_json_dict(),
            "ok": False,
            "exit_code": int(completed.returncode),
            "error": "child process produced no result payload",
            "wall_seconds": None,
            "cpu_seconds": None,
            "peak_rss_bytes": None,
            "profile_receipt": None,
        }
    result["parent_wall_seconds"] = float(parent_wall)
    result["child_returncode"] = int(completed.returncode)
    result["child_stderr_tail"] = "\n".join(
        (completed.stderr or "").strip().splitlines()[-20:]
    )
    return result


def run_matrix(
    cases: Sequence[BenchmarkCase],
    *,
    scratch_dir: Path,
    profile: bool = True,
    repetitions: int = 1,
    timeout_seconds: float | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Run every case ``repetitions`` times and return an aggregate report."""

    if int(repetitions) < 1:
        raise ValueError("repetitions must be positive.")
    results: list[dict[str, Any]] = []
    for repetition in range(int(repetitions)):
        for case in cases:
            result = run_case_in_subprocess(
                case,
                scratch_dir=scratch_dir,
                profile=profile,
                timeout_seconds=timeout_seconds,
                repo_root=repo_root,
            )
            result["repetition"] = int(repetition)
            results.append(result)
    return {
        "schema": BENCHMARK_SCHEMA_V1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parent_environment": environment_record(),
        "case_count": int(len(cases)),
        "repetitions": int(repetitions),
        "profile_enabled": bool(profile),
        "matrix_hash": _stable_hash([case.to_json_dict() for case in cases]),
        "results": results,
        "summary": summarize(results),
        "interpretation_note": (
            "Measurements only. This harness does not judge whether a speedup or "
            "a scientific trajectory is sufficient."
        ),
    }


def summarize(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Collapse repetitions into one row per case, reporting the observed spread."""

    by_case: dict[str, list[Mapping[str, Any]]] = {}
    for result in results:
        case_id = str(dict(result.get("case", {})).get("case_id", "unknown"))
        by_case.setdefault(case_id, []).append(result)

    rows: list[dict[str, Any]] = []
    for case_id in sorted(by_case):
        runs = by_case[case_id]
        case = dict(runs[0].get("case", {}))
        walls = [
            float(run["wall_seconds"])
            for run in runs
            if run.get("ok") and run.get("wall_seconds") is not None
        ]
        cpus = [
            float(run["cpu_seconds"])
            for run in runs
            if run.get("ok") and run.get("cpu_seconds") is not None
        ]
        peaks = [
            int(run["peak_rss_bytes"])
            for run in runs
            if run.get("ok") and run.get("peak_rss_bytes") is not None
        ]
        rows.append(
            {
                "case_id": case_id,
                "description": case.get("description", ""),
                "worker_count": case.get("worker_count"),
                "blas_threads": case.get("blas_threads"),
                "ok_count": int(sum(1 for run in runs if run.get("ok"))),
                "run_count": int(len(runs)),
                "wall_seconds_min": min(walls) if walls else None,
                "wall_seconds_median": (
                    float(np.median(np.asarray(walls, dtype=float))) if walls else None
                ),
                "wall_seconds_max": max(walls) if walls else None,
                "cpu_seconds_median": (
                    float(np.median(np.asarray(cpus, dtype=float))) if cpus else None
                ),
                "cpu_utilization_median": (
                    float(
                        np.median(
                            np.asarray(cpus, dtype=float) / np.asarray(walls, dtype=float)
                        )
                    )
                    if walls and cpus and len(walls) == len(cpus)
                    else None
                ),
                "peak_rss_bytes_max": max(peaks) if peaks else None,
                "errors": sorted(
                    {str(run.get("error")) for run in runs if run.get("error")}
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure wall time, CPU time, peak RSS, and phase profile of the "
            "ordinary AP-McLachlan runner."
        )
    )
    parser.add_argument(
        "--matrix-json",
        default=None,
        help="JSON file holding {'cases': [...]} to execute.",
    )
    parser.add_argument(
        "--single-case-json",
        default=None,
        help="Internal: execute exactly one case in this process.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--scratch-dir", default=None)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="Measure wall/CPU/RSS without installing the phase profiler.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    if args.single_case_json:
        case = BenchmarkCase.from_json_dict(
            json.loads(Path(args.single_case_json).read_text(encoding="utf-8"))
        )
        result = run_case_in_process(case, profile=not bool(args.no_profile))
        output_json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        return 0 if result.get("ok") else 1

    if not args.matrix_json:
        parser.error("one of --matrix-json or --single-case-json is required.")

    payload = json.loads(Path(args.matrix_json).read_text(encoding="utf-8"))
    cases = [BenchmarkCase.from_json_dict(item) for item in payload["cases"]]
    scratch_dir = Path(
        args.scratch_dir or (output_json.parent / f"{output_json.stem}_scratch")
    )
    report = run_matrix(
        cases,
        scratch_dir=scratch_dir,
        profile=not bool(args.no_profile),
        repetitions=int(args.repetitions),
        timeout_seconds=args.timeout_seconds,
        repo_root=Path(args.repo_root) if args.repo_root else None,
    )
    output_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    for row in report["summary"]:
        print(
            f"{row['case_id']:<44} workers={row['worker_count']} "
            f"blas={row['blas_threads']} "
            f"wall_median={row['wall_seconds_median']} "
            f"cpu_util={row['cpu_utilization_median']} "
            f"peak_rss={row['peak_rss_bytes_max']} "
            f"ok={row['ok_count']}/{row['run_count']}"
        )
    return 0 if all(row["ok_count"] == row["run_count"] for row in report["summary"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
