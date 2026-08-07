#!/usr/bin/env python3
"""Shared benchmark job-manifest helpers for local and CHTC runs."""

from __future__ import annotations

import csv
import json
import shlex
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "benchmark_job_manifest_v1"


@dataclass(frozen=True)
class BenchmarkJob:
    job_id: str
    domain: str
    family: str
    case_id: str
    algorithm_id: str
    status: str
    reason: str
    command: tuple[str, ...] = ()
    output_dir: str = ""
    runner_module: str | None = None
    qpu_faithful: bool | None = None
    exact_assisted: bool = False
    diagnostic: bool = False
    hamiltonian_generic: bool = False
    resources: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["command"] = list(self.command)
        out["command_shell"] = command_to_shell(self.command) if self.command else ""
        return out


def command_to_shell(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return str(value)


def manifest_payload(*, jobs: Sequence[BenchmarkJob], label: str = "") -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for job in jobs:
        status_counts[job.status] = status_counts.get(job.status, 0) + 1
    return {
        "schema": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "label": str(label),
        "job_count": int(len(jobs)),
        "status_counts": dict(sorted(status_counts.items())),
        "runnable_count": int(status_counts.get("runnable", 0)),
    }


def write_job_jsonl(path: Path | str, jobs: Sequence[BenchmarkJob]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(json.dumps(job.to_dict(), sort_keys=True, default=_json_default) + "\n")


def write_job_csv(path: Path | str, jobs: Sequence[BenchmarkJob]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [job.to_dict() for job in jobs]
    fieldnames = [
        "job_id",
        "domain",
        "family",
        "case_id",
        "algorithm_id",
        "status",
        "reason",
        "runner_module",
        "output_dir",
        "command_shell",
        "qpu_faithful",
        "exact_assisted",
        "diagnostic",
        "hamiltonian_generic",
    ]
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_manifest_bundle(
    *,
    output_dir: Path | str,
    jobs: Sequence[BenchmarkJob],
    label: str,
    jsonl_name: str = "jobs.jsonl",
    csv_name: str = "jobs.csv",
    summary_name: str = "manifest_summary.json",
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    jsonl_path = root / jsonl_name
    csv_path = root / csv_name
    summary_path = root / summary_name
    write_job_jsonl(jsonl_path, jobs)
    write_job_csv(csv_path, jobs)
    summary = manifest_payload(jobs=jobs, label=label)
    summary["paths"] = {
        "jobs_jsonl": str(jsonl_path),
        "jobs_csv": str(csv_path),
        "summary_json": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


__all__ = [
    "BenchmarkJob",
    "SCHEMA_VERSION",
    "command_to_shell",
    "manifest_payload",
    "write_job_csv",
    "write_job_jsonl",
    "write_manifest_bundle",
]
