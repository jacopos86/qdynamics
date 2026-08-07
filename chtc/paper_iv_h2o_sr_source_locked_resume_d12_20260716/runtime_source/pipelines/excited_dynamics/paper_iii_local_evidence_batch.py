"""Paper-III local-only evidence-scaling batch harness.

This sidecar is intentionally sequential and orchestration-only. It may call the
P7b local science-pilot wrapper, which in turn launches strict HH via subprocess,
but this module must not import realtime/controller, CHTC, Optuna, Qiskit, or
exact-benchmark surfaces.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from pipelines.excited_dynamics.io import write_json
from pipelines.excited_dynamics.paper_iii_local_science_pilot import (
    DEFAULT_HH_ARTIFACT_JSON,
    DEFAULT_NUM_TIMES,
    DEFAULT_T_FINAL,
    DEFAULT_TIMEOUT_SECONDS,
    PaperIIILocalSciencePilotConfig,
    run_paper_iii_local_science_pilot,
)


PAPER_III_LOCAL_EVIDENCE_BATCH_SCHEMA_VERSION = "paper_iii_local_evidence_batch_v1"
PAPER_III_LOCAL_EVIDENCE_BATCH_PIPELINE = "paper_iii_local_evidence_batch"
MODULE_NAME = "pipelines.excited_dynamics.paper_iii_local_evidence_batch"
DEFAULT_BATCH_OUTPUT_DIR = Path("artifacts/agent_runs/paper_iii_p8_local_evidence_scaling")
DEFAULT_SCOREBOARD_MD = Path("prompt-exports/optimize-paper-iii-local-jobs-runs.md")
DEFAULT_EXISTING_STRICT_OUTPUT_JSON = Path(
    "artifacts/agent_runs/paper_iii_p7b_local_science_pilot/hh_strict_realtime_pilot.json"
)
JOB_MODE_REPORT_ONLY = "report_only_existing_output"
JOB_MODE_STRICT_HH = "strict_hh"
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_FORBIDDEN_RAW_MARKERS = (
    "amplitudes_qn_to_q0",
    "raw_physical_state",
    "basis_matrix_vectors",
    "exact_target_trajectories",
    "exact_step_forecast",
    "state_at(",
)


class PaperIIILocalEvidenceBatchError(ValueError):
    """Raised when the local batch configuration or summary is unsafe."""


@dataclass(frozen=True)
class PaperIIILocalEvidenceJobConfig:
    job_id: str
    mode: str
    output_dir: Path
    artifact_json: Path
    t_final: float
    num_times: int
    timeout_seconds: int
    run_tag: str
    existing_strict_output_json: Path | None = None
    progress_json: Path | None = None
    partial_payload_json: Path | None = None


@dataclass(frozen=True)
class PaperIIILocalEvidenceBatchConfig:
    output_dir: Path = DEFAULT_BATCH_OUTPUT_DIR
    artifact_json: Path = DEFAULT_HH_ARTIFACT_JSON
    existing_strict_output_json: Path = DEFAULT_EXISTING_STRICT_OUTPUT_JSON
    scoreboard_md: Path = DEFAULT_SCOREBOARD_MD
    t_final: float = DEFAULT_T_FINAL
    num_times: int = DEFAULT_NUM_TIMES
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    include_second_strict: bool = False
    overwrite_job_dirs: bool = False

    @property
    def batch_summary_json(self) -> Path:
        return Path(self.output_dir) / "batch_summary.json"

    @property
    def batch_summary_md(self) -> Path:
        return Path(self.output_dir) / "batch_summary.md"

    @property
    def run_manifest_json(self) -> Path:
        return Path(self.output_dir) / "run_manifest.json"


PilotFn = Callable[[PaperIIILocalSciencePilotConfig], Mapping[str, Any]]


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _display_path(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _optional_display_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return _display_path(path)


def _shell_join(command: Sequence[str]) -> str:
    return shlex.join([str(part) for part in command])


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return []


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _rate_per_hour(count: int, seconds: float | None) -> float | None:
    if seconds is None or seconds <= 0.0:
        return None
    return float(count) * 3600.0 / float(seconds)


def _assert_no_raw_payload(value: Any, *, context: str) -> None:
    serialized = json.dumps(value, sort_keys=True, allow_nan=False)
    hits = [marker for marker in _FORBIDDEN_RAW_MARKERS if marker in serialized]
    if hits:
        raise PaperIIILocalEvidenceBatchError(f"{context} would emit forbidden raw/reference markers: {hits}")


def _validate_job_config(job: PaperIIILocalEvidenceJobConfig) -> None:
    if not _JOB_ID_RE.match(job.job_id):
        raise PaperIIILocalEvidenceBatchError(f"unsafe job_id: {job.job_id!r}")
    if job.mode not in {JOB_MODE_REPORT_ONLY, JOB_MODE_STRICT_HH}:
        raise PaperIIILocalEvidenceBatchError(f"unsupported job mode for {job.job_id}: {job.mode!r}")
    if job.mode == JOB_MODE_REPORT_ONLY and job.existing_strict_output_json is None:
        raise PaperIIILocalEvidenceBatchError(f"report-only job {job.job_id} requires existing_strict_output_json")
    if job.mode == JOB_MODE_STRICT_HH and job.existing_strict_output_json is not None:
        raise PaperIIILocalEvidenceBatchError(f"strict job {job.job_id} must not set existing_strict_output_json")


def build_default_job_plan(config: PaperIIILocalEvidenceBatchConfig) -> list[PaperIIILocalEvidenceJobConfig]:
    output_dir = Path(config.output_dir)
    strict_001_dir = output_dir / "baseline_strict_hh_001"
    jobs = [
        PaperIIILocalEvidenceJobConfig(
            job_id="p8-report-only-001",
            mode=JOB_MODE_REPORT_ONLY,
            output_dir=output_dir / "report_only_001",
            artifact_json=Path(config.artifact_json),
            t_final=float(config.t_final),
            num_times=int(config.num_times),
            timeout_seconds=int(config.timeout_seconds),
            run_tag="paper_iii_p8_report_only_001",
            existing_strict_output_json=Path(config.existing_strict_output_json),
        ),
        PaperIIILocalEvidenceJobConfig(
            job_id="p8-strict-001",
            mode=JOB_MODE_STRICT_HH,
            output_dir=strict_001_dir,
            artifact_json=Path(config.artifact_json),
            t_final=float(config.t_final),
            num_times=int(config.num_times),
            timeout_seconds=int(config.timeout_seconds),
            run_tag="paper_iii_p8_baseline_strict_hh_001",
            progress_json=strict_001_dir / "progress.json",
            partial_payload_json=strict_001_dir / "partial_payload.json",
        ),
    ]
    if config.include_second_strict:
        strict_002_dir = output_dir / "baseline_strict_hh_002"
        jobs.append(
            PaperIIILocalEvidenceJobConfig(
                job_id="p8-strict-002",
                mode=JOB_MODE_STRICT_HH,
                output_dir=strict_002_dir,
                artifact_json=Path(config.artifact_json),
                t_final=float(config.t_final),
                num_times=int(config.num_times),
                timeout_seconds=int(config.timeout_seconds),
                run_tag="paper_iii_p8_baseline_strict_hh_002",
                progress_json=strict_002_dir / "progress.json",
                partial_payload_json=strict_002_dir / "partial_payload.json",
            )
        )
    else:
        jobs.append(
            PaperIIILocalEvidenceJobConfig(
                job_id="p8-report-only-002",
                mode=JOB_MODE_REPORT_ONLY,
                output_dir=output_dir / "report_only_002",
                artifact_json=Path(config.artifact_json),
                t_final=float(config.t_final),
                num_times=int(config.num_times),
                timeout_seconds=int(config.timeout_seconds),
                run_tag="paper_iii_p8_report_only_002",
                existing_strict_output_json=strict_001_dir / "hh_strict_realtime_pilot.json",
            )
        )
    seen: set[str] = set()
    for job in jobs:
        _validate_job_config(job)
        if job.job_id in seen:
            raise PaperIIILocalEvidenceBatchError(f"duplicate job_id: {job.job_id}")
        seen.add(job.job_id)
    return jobs


def _pilot_config(job: PaperIIILocalEvidenceJobConfig) -> PaperIIILocalSciencePilotConfig:
    return PaperIIILocalSciencePilotConfig(
        artifact_json=Path(job.artifact_json),
        output_dir=Path(job.output_dir),
        t_final=float(job.t_final),
        num_times=int(job.num_times),
        timeout_seconds=int(job.timeout_seconds),
        run_tag=str(job.run_tag),
        report_only_existing_output=job.mode == JOB_MODE_REPORT_ONLY,
        progress_json=Path(job.progress_json) if job.progress_json is not None else None,
        partial_payload_json=Path(job.partial_payload_json) if job.partial_payload_json is not None else None,
        existing_strict_output_json=Path(job.existing_strict_output_json)
        if job.existing_strict_output_json is not None
        else None,
    )


def _leakage_checks(report: Mapping[str, Any]) -> dict[str, bool]:
    run = _mapping(_mapping(report.get("runs")).get("strict_hh_runtime_dynamics"))
    metrics = _mapping(run.get("metrics"))
    strict_route = _mapping(metrics.get("strict_route"))
    boundary = _mapping(report.get("controller_boundary"))
    return {
        "controller_exact_input_mode_off": strict_route.get("controller_exact_input_mode") == "off",
        "diagnostic_exact_reference_mode_benchmark_exact": strict_route.get("diagnostic_exact_reference_mode")
        == "benchmark_exact",
        "uses_reference_for_decision_false": strict_route.get("uses_reference_for_decision") is False,
        "uses_future_exact_forecast_for_decision_false": strict_route.get("uses_future_exact_forecast_for_decision")
        is False,
        "exact_decision_checkpoints_zero": int(strict_route.get("exact_decision_checkpoints", 0) or 0) == 0,
        "qpu_faithful_decisions_passed_true": strict_route.get("qpu_faithful_decisions_passed") is True,
        "strict_measurement_oracle_certified_true": strict_route.get("strict_measurement_oracle_certified") is True,
        "feeds_controller_decisions_false": report.get("feeds_controller_decisions") is False
        and boundary.get("feeds_controller_decisions") is False,
        "reference_comparisons_feed_controller_decisions_false": report.get(
            "reference_comparisons_feed_controller_decisions"
        )
        is False
        and boundary.get("reference_comparisons_feed_controller_decisions") is False,
    }


def _extract_job_summary(
    *,
    job: PaperIIILocalEvidenceJobConfig,
    report: Mapping[str, Any],
    started_utc: str,
    finished_utc: str,
    wallclock_seconds: float,
) -> dict[str, Any]:
    run = _mapping(_mapping(report.get("runs")).get("strict_hh_runtime_dynamics"))
    metrics = _mapping(run.get("metrics"))
    horizon = _mapping(metrics.get("horizon"))
    validation = _mapping(run.get("strict_validation"))
    blockers = [str(item) for item in _sequence(report.get("blockers"))]
    leakage_checks = _leakage_checks(report)
    leakage_failure = not all(leakage_checks.values())
    strict_passed = validation.get("passed") is True
    paper_science = report.get("paper_iii_science_benchmark") is True
    status = "completed" if strict_passed and paper_science and not blockers and not leakage_failure else "blocked"
    progress_json = Path(job.progress_json) if job.progress_json is not None else None
    partial_payload_json = Path(job.partial_payload_json) if job.partial_payload_json is not None else None
    summary = {
        "job_id": job.job_id,
        "mode": job.mode,
        "output_dir": _display_path(job.output_dir),
        "artifact_json": _display_path(job.artifact_json),
        "existing_strict_output_json": _optional_display_path(job.existing_strict_output_json),
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "wallclock_seconds": float(wallclock_seconds),
        "status": status,
        "command_status": run.get("command_status"),
        "command_wallclock_seconds": run.get("command_wallclock_seconds"),
        "strict_validation_passed": bool(strict_passed),
        "paper_iii_science_benchmark": bool(paper_science),
        "physical_row_count": horizon.get("physical_row_count"),
        "final_physical_time": horizon.get("final_physical_time"),
        "blocker_count": int(len(blockers)),
        "blockers": blockers,
        "controller_leakage_failure": bool(leakage_failure),
        "leakage_checks": leakage_checks,
        "artifacts": {
            "report_json": _display_path(Path(job.output_dir) / "paper_iii_local_science_pilot_report.json"),
            "report_md": _display_path(Path(job.output_dir) / "paper_iii_local_science_pilot_report.md"),
            "run_manifest_json": _display_path(Path(job.output_dir) / "run_manifest.json"),
            "command_log_md": _display_path(Path(job.output_dir) / "command_log.md"),
            "stdout_log": _display_path(Path(job.output_dir) / "logs" / "stdout.log"),
            "stderr_log": _display_path(Path(job.output_dir) / "logs" / "stderr.log"),
            "strict_payload_source_json": run.get("strict_payload_source_json") or run.get("output_json"),
            "progress_json": _optional_display_path(progress_json),
            "progress_json_exists": bool(progress_json is not None and progress_json.exists()),
            "partial_payload_json": _optional_display_path(partial_payload_json),
            "partial_payload_json_exists": bool(partial_payload_json is not None and partial_payload_json.exists()),
        },
    }
    _assert_no_raw_payload(summary, context=f"job_summary:{job.job_id}")
    return summary


def _exception_job_summary(
    *,
    job: PaperIIILocalEvidenceJobConfig,
    started_utc: str,
    finished_utc: str,
    wallclock_seconds: float,
    exc: BaseException,
) -> dict[str, Any]:
    summary = {
        "job_id": job.job_id,
        "mode": job.mode,
        "output_dir": _display_path(job.output_dir),
        "artifact_json": _display_path(job.artifact_json),
        "existing_strict_output_json": _optional_display_path(job.existing_strict_output_json),
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "wallclock_seconds": float(wallclock_seconds),
        "status": "blocked",
        "command_status": "exception",
        "command_wallclock_seconds": None,
        "strict_validation_passed": False,
        "paper_iii_science_benchmark": False,
        "physical_row_count": None,
        "final_physical_time": None,
        "blocker_count": 1,
        "blockers": [f"job_exception:{type(exc).__name__}:{exc}"],
        "controller_leakage_failure": False,
        "leakage_checks": {},
        "artifacts": {
            "report_json": _display_path(Path(job.output_dir) / "paper_iii_local_science_pilot_report.json"),
            "report_md": _display_path(Path(job.output_dir) / "paper_iii_local_science_pilot_report.md"),
            "run_manifest_json": _display_path(Path(job.output_dir) / "run_manifest.json"),
            "command_log_md": _display_path(Path(job.output_dir) / "command_log.md"),
            "stdout_log": _display_path(Path(job.output_dir) / "logs" / "stdout.log"),
            "stderr_log": _display_path(Path(job.output_dir) / "logs" / "stderr.log"),
            "strict_payload_source_json": None,
            "progress_json": _optional_display_path(job.progress_json),
            "progress_json_exists": False,
            "partial_payload_json": _optional_display_path(job.partial_payload_json),
            "partial_payload_json_exists": False,
        },
    }
    _assert_no_raw_payload(summary, context=f"job_exception_summary:{job.job_id}")
    return summary


def execute_local_evidence_job(
    job: PaperIIILocalEvidenceJobConfig,
    *,
    pilot_fn: PilotFn = run_paper_iii_local_science_pilot,
) -> dict[str, Any]:
    _validate_job_config(job)
    Path(job.output_dir).mkdir(parents=True, exist_ok=True)
    started_utc = _utc_now()
    started_perf = time.perf_counter()
    try:
        report = pilot_fn(_pilot_config(job))
        finished_utc = _utc_now()
        wallclock_seconds = max(0.0, float(time.perf_counter() - started_perf))
        summary = _extract_job_summary(
            job=job,
            report=report,
            started_utc=started_utc,
            finished_utc=finished_utc,
            wallclock_seconds=wallclock_seconds,
        )
    except Exception as exc:  # pragma: no cover - defensive fail-closed path
        finished_utc = _utc_now()
        wallclock_seconds = max(0.0, float(time.perf_counter() - started_perf))
        summary = _exception_job_summary(
            job=job,
            started_utc=started_utc,
            finished_utc=finished_utc,
            wallclock_seconds=wallclock_seconds,
            exc=exc,
        )
    write_json(Path(job.output_dir) / "job_summary.json", summary)
    return summary


def _should_continue_after_job(index: int, job: PaperIIILocalEvidenceJobConfig, job_summary: Mapping[str, Any]) -> bool:
    if job_summary.get("status") == "completed" and not job_summary.get("controller_leakage_failure"):
        return True
    blockers = [str(item) for item in _sequence(job_summary.get("blockers"))]
    if (
        index == 0
        and job.mode == JOB_MODE_REPORT_ONLY
        and any("existing_strict_output_missing" in blocker for blocker in blockers)
    ):
        return True
    return False


def _optimization_candidates_ranked() -> list[dict[str, Any]]:
    return [
        {
            "rank": 1,
            "candidate": "Local batch harness + timing + scoreboard",
            "expected_delta": "high",
            "risk": "low",
            "action": "implemented as setup/baseline scaffold",
        },
        {
            "rank": 2,
            "candidate": "P7b progress/partial forwarding",
            "expected_delta": "medium",
            "risk": "low",
            "action": "implemented for strict jobs",
        },
        {
            "rank": 3,
            "candidate": "Explicit existing strict JSON revalidation path",
            "expected_delta": "medium",
            "risk": "low",
            "action": "implemented for report-only jobs",
        },
        {
            "rank": 4,
            "candidate": "Second strict baseline for variance",
            "expected_delta": "medium",
            "risk": "low",
            "action": "optional --include-second-strict after strict-001 passes",
        },
        {
            "rank": 5,
            "candidate": "Labeled integrator/candidate ablations",
            "expected_delta": "unknown",
            "risk": "medium/high",
            "action": "defer; not part of this setup/baseline task",
        },
    ]


def _build_stop_continue_decision(jobs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    leakage_failures = sum(1 for job in jobs if job.get("controller_leakage_failure") is True)
    strict_jobs = [job for job in jobs if job.get("mode") == JOB_MODE_STRICT_HH]
    strict_completed = [job for job in strict_jobs if job.get("status") == "completed"]
    strict_progress_missing = [
        job.get("job_id")
        for job in strict_completed
        if not _mapping(job.get("artifacts")).get("progress_json_exists")
        or not _mapping(job.get("artifacts")).get("partial_payload_json_exists")
    ]
    blockers = [blocker for job in jobs for blocker in _sequence(job.get("blockers"))]
    if leakage_failures:
        return {
            "decision": "stop",
            "reasons": ["controller_leakage_failure_detected", "strict evidence must fail closed"],
        }
    if strict_jobs and not strict_completed:
        return {"decision": "stop", "reasons": ["no_strict_runtime_job_completed"]}
    if strict_progress_missing:
        return {
            "decision": "stop",
            "reasons": [f"strict_progress_or_partial_artifact_missing:{strict_progress_missing}"],
        }
    if blockers:
        return {"decision": "stop", "reasons": ["blockers_present"]}
    if strict_completed:
        return {
            "decision": "continue",
            "reasons": [
                "strict_runtime_baseline_completed",
                "zero_controller_leakage_failures",
                "progress_artifacts_present",
            ],
        }
    return {"decision": "stop", "reasons": ["no_strict_runtime_job_attempted"]}


def _batch_primary_metrics(*, jobs: Sequence[Mapping[str, Any]], total_wallclock_seconds: float) -> dict[str, Any]:
    validated_jobs = [job for job in jobs if job.get("status") == "completed"]
    strict_jobs = [job for job in jobs if job.get("mode") == JOB_MODE_STRICT_HH]
    validated_strict_jobs = [job for job in strict_jobs if job.get("status") == "completed"]
    strict_wall_seconds = sum(_finite_float(job.get("wallclock_seconds")) or 0.0 for job in strict_jobs)
    leakage_failure_count = sum(1 for job in jobs if job.get("controller_leakage_failure") is True)
    return {
        "attempted_job_count": int(len(jobs)),
        "validated_local_evidence_job_count": int(len(validated_jobs)),
        "validated_local_evidence_jobs_per_hour": _rate_per_hour(len(validated_jobs), total_wallclock_seconds),
        "validated_strict_runtime_job_count": int(len(validated_strict_jobs)),
        "validated_strict_runtime_jobs_per_hour": _rate_per_hour(len(validated_strict_jobs), strict_wall_seconds),
        "total_wallclock_seconds": float(total_wallclock_seconds),
        "strict_runtime_wallclock_seconds": float(strict_wall_seconds),
        "controller_leakage_failure_count": int(leakage_failure_count),
        "blocker_count": int(sum(int(job.get("blocker_count", 0) or 0) for job in jobs)),
    }


def _render_job_table_rows(jobs: Sequence[Mapping[str, Any]]) -> list[str]:
    rows: list[str] = []
    for job in jobs:
        artifacts = _mapping(job.get("artifacts"))
        leakage = "pass" if not job.get("controller_leakage_failure") else "fail"
        notes: list[str] = []
        if job.get("mode") == JOB_MODE_STRICT_HH:
            notes.append(
                "progress=" + ("yes" if artifacts.get("progress_json_exists") else "no")
                + ", partial="
                + ("yes" if artifacts.get("partial_payload_json_exists") else "no")
            )
        blockers = job.get("blocker_count", 0)
        rows.append(
            "| {job_id} | {mode} | P7b wrapper | {output_dir} | {started} | {finished} | {wall:.3f} | {status} | {strict_pass} | {rows} | {final_t} | {blockers} | {leakage} | {notes} |".format(
                job_id=job.get("job_id"),
                mode=job.get("mode"),
                output_dir=job.get("output_dir"),
                started=job.get("started_utc"),
                finished=job.get("finished_utc"),
                wall=float(job.get("wallclock_seconds") or 0.0),
                status=job.get("status"),
                strict_pass=str(job.get("strict_validation_passed")).lower(),
                rows=job.get("physical_row_count"),
                final_t=job.get("final_physical_time"),
                blockers=blockers,
                leakage=leakage,
                notes="; ".join(notes) if notes else "report-only validation",
            )
        )
    return rows


def _render_batch_markdown(summary: Mapping[str, Any]) -> str:
    primary = _mapping(summary.get("primary_metric"))
    lines = [
        "# Paper III Local Evidence Batch Summary",
        "",
        f"Generated UTC: `{summary.get('generated_utc')}`",
        "",
        "## Primary metrics",
        "",
        f"- attempted_job_count: `{primary.get('attempted_job_count')}`",
        f"- validated_local_evidence_job_count: `{primary.get('validated_local_evidence_job_count')}`",
        f"- validated_local_evidence_jobs_per_hour: `{primary.get('validated_local_evidence_jobs_per_hour')}`",
        f"- validated_strict_runtime_job_count: `{primary.get('validated_strict_runtime_job_count')}`",
        f"- validated_strict_runtime_jobs_per_hour: `{primary.get('validated_strict_runtime_jobs_per_hour')}`",
        f"- controller_leakage_failure_count: `{primary.get('controller_leakage_failure_count')}`",
        f"- blocker_count: `{primary.get('blocker_count')}`",
        "",
        "## Jobs",
        "",
        "| Run ID | Mode | Command source | Output dir | Started UTC | Finished UTC | Wall sec | Status | Strict pass | Physical rows | Final t | Blockers | Leakage checks | Notes |",
        "|---|---|---|---|---|---|---:|---|---|---:|---:|---:|---|---|",
    ]
    lines.extend(_render_job_table_rows(_sequence(summary.get("jobs"))))
    decision = _mapping(summary.get("stop_continue_decision"))
    lines.extend(
        [
            "",
            "## Stop/continue decision",
            "",
            f"- decision: `{decision.get('decision')}`",
            f"- reasons: `{', '.join(str(item) for item in _sequence(decision.get('reasons')))}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_scoreboard(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Paper III Local Evidence Scaling Runs",
        "",
        "Purpose: measure local-only strict/report evidence throughput while CHTC is unavailable.",
        "",
        "Primary metric: validated strict/local evidence jobs per wall-clock hour with zero controller-leakage failures.",
        "Secondary metrics: wall-clock seconds per job, physical row count, final time, strict validation pass/fail, blocker count, stdout/stderr/progress artifact presence, and report-only exact/reference separation.",
        "",
        "Guardrails:",
        "- No CHTC, Optuna, fresh ADAPT, remote-runner edits, or Qiskit/IBM runtime.",
        "- Strict HH evidence must preserve controller exact/reference separation and fail closed on leakage.",
        f"- Artifacts live under `{summary.get('output_dir')}`.",
        "- Exact/reference comparisons are report-only and never feed controller decisions.",
        "",
        "## Batch command",
        "",
        f"`{summary.get('batch_command') or 'not recorded'}`",
        "",
        "## Run table",
        "",
        "| Run ID | Mode | Command source | Output dir | Started UTC | Finished UTC | Wall sec | Status | Strict pass | Physical rows | Final t | Blockers | Leakage checks | Notes |",
        "|---|---|---|---|---|---|---:|---|---|---:|---:|---:|---|---|",
    ]
    lines.extend(_render_job_table_rows(_sequence(summary.get("jobs"))))
    decision = _mapping(summary.get("stop_continue_decision"))
    lines.extend(
        [
            "",
            "## Stop/continue decision",
            "",
            f"- `{decision.get('decision')}`: {', '.join(str(item) for item in _sequence(decision.get('reasons')))}",
            "",
            "## Ranked first-pass optimization candidates",
            "",
            "| Rank | Candidate | Expected delta | Risk | Action |",
            "|---:|---|---|---|---|",
        ]
    )
    for candidate in _sequence(summary.get("optimization_candidates_ranked")):
        candidate_map = _mapping(candidate)
        lines.append(
            f"| {candidate_map.get('rank')} | {candidate_map.get('candidate')} | {candidate_map.get('expected_delta')} | {candidate_map.get('risk')} | {candidate_map.get('action')} |"
        )
    return "\n".join(lines) + "\n"


def _build_run_manifest(summary: Mapping[str, Any], config: PaperIIILocalEvidenceBatchConfig) -> dict[str, Any]:
    primary = _mapping(summary.get("primary_metric"))
    return {
        "schema_version": "agent_run_manifest_v1",
        "slice": "paper_iii_p8_local_evidence_scaling",
        "generated_utc": summary.get("generated_utc"),
        "pipeline": PAPER_III_LOCAL_EVIDENCE_BATCH_PIPELINE,
        "local_only": True,
        "paper_iii_local_evidence_batch": True,
        "production_claim": False,
        "commands": {"local_evidence_batch": {"command": summary.get("batch_command"), "status": "completed"}},
        "artifacts": {
            "batch_summary_json": _display_path(config.batch_summary_json),
            "batch_summary_md": _display_path(config.batch_summary_md),
            "run_manifest_json": _display_path(config.run_manifest_json),
            "scoreboard_md": _display_path(config.scoreboard_md),
        },
        "output_summary": {
            "attempted_job_count": primary.get("attempted_job_count"),
            "validated_local_evidence_job_count": primary.get("validated_local_evidence_job_count"),
            "validated_strict_runtime_job_count": primary.get("validated_strict_runtime_job_count"),
            "validated_strict_runtime_jobs_per_hour": primary.get("validated_strict_runtime_jobs_per_hour"),
            "controller_leakage_failure_count": primary.get("controller_leakage_failure_count"),
            "blocker_count": primary.get("blocker_count"),
        },
        "guardrails": dict(_mapping(summary.get("guardrails"))),
        "stop_continue_decision": dict(_mapping(summary.get("stop_continue_decision"))),
    }


def run_paper_iii_local_evidence_batch(
    config: PaperIIILocalEvidenceBatchConfig,
    *,
    pilot_fn: PilotFn = run_paper_iii_local_science_pilot,
    command_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    jobs = build_default_job_plan(config)
    generated_utc = _utc_now()
    batch_started_utc = generated_utc
    batch_started_perf = time.perf_counter()
    completed_jobs: list[dict[str, Any]] = []
    skipped_job_ids: list[str] = []
    for index, job in enumerate(jobs):
        job_summary = execute_local_evidence_job(job, pilot_fn=pilot_fn)
        completed_jobs.append(job_summary)
        if not _should_continue_after_job(index, job, job_summary):
            skipped_job_ids = [remaining.job_id for remaining in jobs[index + 1 :]]
            break
    batch_finished_utc = _utc_now()
    total_wallclock_seconds = max(0.0, float(time.perf_counter() - batch_started_perf))
    primary = _batch_primary_metrics(jobs=completed_jobs, total_wallclock_seconds=total_wallclock_seconds)
    summary: dict[str, Any] = {
        "schema_version": PAPER_III_LOCAL_EVIDENCE_BATCH_SCHEMA_VERSION,
        "pipeline": PAPER_III_LOCAL_EVIDENCE_BATCH_PIPELINE,
        "generated_utc": generated_utc,
        "started_utc": batch_started_utc,
        "finished_utc": batch_finished_utc,
        "output_dir": _display_path(config.output_dir),
        "batch_command": _shell_join(command_argv) if command_argv else None,
        "local_only": True,
        "sequential_local_batch": True,
        "guardrails": {
            "chtc_used": False,
            "optuna_used": False,
            "fresh_adapt_run": False,
            "remote_runner_edits": False,
            "qiskit_ibm_runtime_used": False,
            "exact_bench_changed": False,
            "realtime_or_controller_defaults_changed": False,
            "adapt_static_defaults_changed": False,
            "parallel_job_fanout": False,
            "physical_statevectors_emitted_in_summaries": False,
        },
        "config": {
            "artifact_json": _display_path(config.artifact_json),
            "existing_strict_output_json": _display_path(config.existing_strict_output_json),
            "scoreboard_md": _display_path(config.scoreboard_md),
            "t_final": float(config.t_final),
            "num_times": int(config.num_times),
            "timeout_seconds": int(config.timeout_seconds),
            "include_second_strict": bool(config.include_second_strict),
            "overwrite_job_dirs": bool(config.overwrite_job_dirs),
        },
        "primary_metric": primary,
        "jobs": completed_jobs,
        "skipped_job_ids": skipped_job_ids,
        "stop_continue_decision": _build_stop_continue_decision(completed_jobs),
        "optimization_candidates_ranked": _optimization_candidates_ranked(),
    }
    _assert_no_raw_payload(summary, context="paper_iii_local_evidence_batch_summary")
    markdown = _render_batch_markdown(summary)
    scoreboard = _render_scoreboard(summary)
    manifest = _build_run_manifest(summary, config)
    _assert_no_raw_payload(markdown, context="paper_iii_local_evidence_batch_markdown")
    _assert_no_raw_payload(scoreboard, context="paper_iii_local_evidence_batch_scoreboard")
    _assert_no_raw_payload(manifest, context="paper_iii_local_evidence_batch_manifest")
    write_json(config.batch_summary_json, summary)
    Path(config.batch_summary_md).write_text(markdown, encoding="utf-8")
    write_json(config.run_manifest_json, manifest)
    Path(config.scoreboard_md).parent.mkdir(parents=True, exist_ok=True)
    Path(config.scoreboard_md).write_text(scoreboard, encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local-only sequential Paper-III evidence-scaling batch.")
    parser.add_argument("--output-dir", default=str(DEFAULT_BATCH_OUTPUT_DIR))
    parser.add_argument("--artifact-json", default=str(DEFAULT_HH_ARTIFACT_JSON))
    parser.add_argument("--existing-strict-output-json", default=str(DEFAULT_EXISTING_STRICT_OUTPUT_JSON))
    parser.add_argument("--t-final", type=float, default=DEFAULT_T_FINAL)
    parser.add_argument("--num-times", type=int, default=DEFAULT_NUM_TIMES)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--scoreboard-md", default=str(DEFAULT_SCOREBOARD_MD))
    parser.add_argument(
        "--include-second-strict",
        action="store_true",
        help="Run a second strict HH baseline instead of report-only revalidating strict-001.",
    )
    parser.add_argument(
        "--overwrite-job-dirs",
        action="store_true",
        help="Recorded for provenance only; this harness does not delete existing artifacts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    summary = run_paper_iii_local_evidence_batch(
        PaperIIILocalEvidenceBatchConfig(
            output_dir=Path(args.output_dir),
            artifact_json=Path(args.artifact_json),
            existing_strict_output_json=Path(args.existing_strict_output_json),
            scoreboard_md=Path(args.scoreboard_md),
            t_final=float(args.t_final),
            num_times=int(args.num_times),
            timeout_seconds=int(args.timeout_seconds),
            include_second_strict=bool(args.include_second_strict),
            overwrite_job_dirs=bool(args.overwrite_job_dirs),
        ),
        command_argv=[sys.executable, "-m", MODULE_NAME, *raw_argv],
    )
    primary = _mapping(summary.get("primary_metric"))
    strict_count = int(primary.get("validated_strict_runtime_job_count", 0) or 0)
    leakage_count = int(primary.get("controller_leakage_failure_count", 0) or 0)
    blocker_count = int(primary.get("blocker_count", 0) or 0)
    return 0 if strict_count >= 1 and leakage_count == 0 and blocker_count == 0 else 1


__all__ = [
    "DEFAULT_BATCH_OUTPUT_DIR",
    "DEFAULT_EXISTING_STRICT_OUTPUT_JSON",
    "DEFAULT_SCOREBOARD_MD",
    "JOB_MODE_REPORT_ONLY",
    "JOB_MODE_STRICT_HH",
    "MODULE_NAME",
    "PAPER_III_LOCAL_EVIDENCE_BATCH_PIPELINE",
    "PAPER_III_LOCAL_EVIDENCE_BATCH_SCHEMA_VERSION",
    "PaperIIILocalEvidenceBatchConfig",
    "PaperIIILocalEvidenceBatchError",
    "PaperIIILocalEvidenceJobConfig",
    "build_default_job_plan",
    "build_parser",
    "execute_local_evidence_job",
    "main",
    "run_paper_iii_local_evidence_batch",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
