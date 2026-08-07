"""Worker/CPU limit helpers for static ADAPT execution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


_AUTO_WORKER_CAP_ENV = "STATIC_ADAPT_AUTO_WORKER_CAP"
_ALLOCATED_CPU_ENV_NAMES = (
    "STATIC_ADAPT_ALLOCATED_CPUS",
    "CONDOR_REQUEST_CPUS",
    "REQUEST_CPUS",
    "SLURM_CPUS_PER_TASK",
    "PBS_NP",
    "NSLOTS",
)


def _positive_int_from_text(raw: Any) -> int | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        value = int(float(text))
    except (TypeError, ValueError):
        return None
    return int(value) if int(value) > 0 else None


def _positive_int_env(name: str) -> int | None:
    return _positive_int_from_text(os.environ.get(str(name)))


def _condor_ad_cpu_count() -> int | None:
    for env_name in ("_CONDOR_JOB_AD", "_CONDOR_MACHINE_AD"):
        ad_path = str(os.environ.get(env_name, "") or "").strip()
        if not ad_path:
            continue
        path = Path(ad_path)
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for field in ("RequestCpus", "Cpus"):
            prefix = f"{field} ="
            for line in lines:
                if str(line).strip().startswith(prefix):
                    parsed = _positive_int_from_text(str(line).split("=", 1)[1])
                    if parsed is not None:
                        return int(parsed)
    return None


def _allocated_cpu_count() -> int:
    for name in _ALLOCATED_CPU_ENV_NAMES:
        parsed = _positive_int_env(name)
        if parsed is not None:
            return int(parsed)
    condor_cpus = _condor_ad_cpu_count()
    if condor_cpus is not None:
        return int(condor_cpus)
    get_affinity = getattr(os, "sched_getaffinity", None)
    if callable(get_affinity):
        try:
            affinity_count = len(get_affinity(0))
        except OSError:
            affinity_count = 0
        if int(affinity_count) > 0:
            return int(affinity_count)
    return int(os.cpu_count() or 1)


def _resolve_adapt_worker_limit(
    requested: int,
    *,
    name: str,
) -> tuple[int, dict[str, Any]]:
    requested_int = int(requested)
    if requested_int < 0:
        raise ValueError(f"{name} must be >= 0 (0=auto).")
    allocated_cpus = int(max(1, _allocated_cpu_count()))
    cap_env = _positive_int_env(_AUTO_WORKER_CAP_ENV)
    if requested_int > 0:
        configured_cap = int(requested_int)
        resolved = int(min(int(allocated_cpus), int(configured_cap)))
        source = "explicit"
    else:
        configured_cap = int(cap_env) if cap_env is not None else int(allocated_cpus)
        resolved = int(min(int(allocated_cpus), int(configured_cap)))
        source = "auto_allocated_cpu_count"
    resolved = int(max(1, resolved))
    return resolved, {
        "requested": int(requested_int),
        "resolved": int(resolved),
        "source": str(source),
        "allocated_cpus": int(allocated_cpus),
        "configured_cap": int(configured_cap),
        "cap_env": _AUTO_WORKER_CAP_ENV,
        "cap_env_value": (None if cap_env is None else int(cap_env)),
    }


def _cap_worker_limit_for_items(worker_limit: int, item_count: int) -> int:
    if int(item_count) <= 0:
        return 1
    return int(max(1, min(int(worker_limit), int(item_count))))
