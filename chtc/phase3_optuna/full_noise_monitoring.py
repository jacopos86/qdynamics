#!/usr/bin/env python3
"""Support-owned Paper-I full-noise monitoring harness.

This module intentionally lives outside the production ADAPT/oracle path.  It
collects existing runtime evidence, appends process/reliability scoreboard rows,
and can run a tiny direct oracle probe when an ADAPT depth-1 smoke would be too
expensive.  It reports shared gate-list normalization status without changing
source-lock semantics or physics/algorithm settings.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.noise_oracle_defaults import normalize_gate_name_tuple  # noqa: E402
from pipelines.static_adapt.builders.pool_resolution import (  # noqa: E402
    build_selected_logical_pool_match_report,
    resolve_pool_plan,
    resolve_requested_pool_filters,
)
from pipelines.static_adapt.builders.problem_registry import ProblemRequest, resolve_problem_context  # noqa: E402

SCOREBOARD_SCHEMA = "paper_i_full_noise_monitoring_scoreboard_v1"
DEFAULT_SCOREBOARD = REPO_ROOT / "prompt-exports" / "optimize-paper-i-full-noise-monitoring-runs.md"
DEFAULT_DEPTH1_SMOKE_ROOT = (
    REPO_ROOT
    / "monitor_artifacts"
    / "paper_i_full_noise_monitoring"
    / "depth1_source_lock_recoverable_json_smoke_v1"
)
DEFAULT_DEPTH1_SELECTED_LOGICAL_SOURCE = (
    REPO_ROOT
    / "chtc"
    / "phase3_optuna"
    / "input"
    / "paper_i_reduced_pool_selected_logical_20260524_v1"
    / "hh_L2_nph2.selected_logical.json"
)
REMOTE_RUNNER_STATUS_URL = "http://127.0.0.1:8765/api/status"
AI_LOG_RE = re.compile(r"(?:^|\s)AI_LOG\s+(\{.*\})\s*$")

SCOREBOARD_COLUMNS = (
    "timestamp_utc",
    "iteration_id",
    "attributed_change",
    "command",
    "case_id",
    "smoke_root",
    "expected_validity",
    "observed_run_validity",
    "invalid_run_detection_latency_s",
    "invalid_run_detection_depth",
    "full_noise_active_gate",
    "source_lock_active_gate",
    "json_recoverability_gate",
    "remote_runner_status_snapshot",
    "rejection_reason",
    "evidence_paths",
    "follow_up",
)

SCOREBOARD_SCAFFOLD = """# Paper-I Full-Noise Monitoring Optimization Runs

Append-only scoreboard for process/runtime reliability measurements. Do not edit prior rows; append corrections as new rows with a new `iteration_id` and a `follow_up` reference to the superseded row. This file records run validity and monitoring latency only, not physics quality.

## Schema v1

Required fields:

- `timestamp_utc`
- `iteration_id`
- `attributed_change`
- `command`
- `case_id`
- `smoke_root`
- `expected_validity`
- `observed_run_validity`
- `invalid_run_detection_latency_s`
- `invalid_run_detection_depth`
- `full_noise_active_gate`
- `source_lock_active_gate`
- `json_recoverability_gate`
- `remote_runner_status_snapshot`
- `rejection_reason`
- `evidence_paths`
- `follow_up`

## Rows

| timestamp_utc | iteration_id | attributed_change | command | case_id | smoke_root | expected_validity | observed_run_validity | invalid_run_detection_latency_s | invalid_run_detection_depth | full_noise_active_gate | source_lock_active_gate | json_recoverability_gate | remote_runner_status_snapshot | rejection_reason | evidence_paths | follow_up |
|---|---|---|---|---|---|---|---|---:|---:|---|---|---|---|---|---|---|
"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _safe_relative(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve(strict=False).relative_to(REPO_ROOT.resolve(strict=False)))
    except Exception:
        return str(p)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _safe_relative(value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _compact_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), default=str)


def _markdown_cell(value: Any) -> str:
    if value is None:
        text = "null"
    elif isinstance(value, (Mapping, list, tuple, set)):
        text = _compact_json(value)
    else:
        text = str(value)
    text = text.replace("\n", " ").replace("\r", " ")
    return text.replace("|", "\\|")


def ensure_scoreboard(path: str | Path = DEFAULT_SCOREBOARD) -> Path:
    board = _repo_path(path)
    board.parent.mkdir(parents=True, exist_ok=True)
    if not board.exists() or board.stat().st_size == 0:
        board.write_text(SCOREBOARD_SCAFFOLD, encoding="utf-8")
    return board


def append_scoreboard_row(path: str | Path, row: Mapping[str, Any]) -> Path:
    board = ensure_scoreboard(path)
    ordered = [_markdown_cell(row.get(column, "")) for column in SCOREBOARD_COLUMNS]
    with board.open("a", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(ordered) + " |\n")
    return board


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, Mapping):
        return None, "json_not_mapping"
    return dict(payload), None


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_ai_log_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    for line_no, line in enumerate(lines, start=1):
        match = AI_LOG_RE.search(line.strip())
        if match is None:
            continue
        try:
            payload = json.loads(match.group(1))
        except Exception:
            continue
        if not isinstance(payload, Mapping):
            continue
        event = dict(payload)
        event["_line_no"] = int(line_no)
        ts = _parse_ts(event.get("ts_utc") or event.get("timestamp_utc"))
        if ts is not None:
            event["_parsed_ts"] = ts
        events.append(event)
    return events


def _iter_json_files(root: Path, *, limit: int = 500) -> list[Path]:
    if root.is_file() and root.suffix.lower() == ".json":
        return [root]
    if not root.exists():
        return []
    out: list[Path] = []
    for path in sorted(root.rglob("*.json")):
        if len(out) >= limit:
            break
        if path.is_file():
            out.append(path)
    return out


def _oracle_config_candidates_from_mapping(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidates: list[Mapping[str, Any]] = []
    settings = payload.get("settings")
    if isinstance(settings, Mapping):
        candidates.append(settings)
    adapt = payload.get("adapt_vqe")
    if isinstance(adapt, Mapping):
        continuation = adapt.get("continuation")
        if isinstance(continuation, Mapping):
            cfg = continuation.get("oracle_gradient_config")
            if isinstance(cfg, Mapping):
                candidates.append(cfg)
            backend = continuation.get("oracle_backend_info")
            if isinstance(backend, Mapping):
                candidates.append({"oracle_backend_info": backend})
        cfg = adapt.get("oracle_gradient_config")
        if isinstance(cfg, Mapping):
            candidates.append(cfg)
        backend = adapt.get("oracle_backend_info")
        if isinstance(backend, Mapping):
            candidates.append({"oracle_backend_info": backend})
    if any(str(key).startswith("phase3_oracle_") for key in payload.keys()):
        candidates.append(payload)
    return candidates


def _oracle_config_candidates(events: Sequence[Mapping[str, Any]], json_payloads: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    candidates: list[Mapping[str, Any]] = []
    for payload in json_payloads:
        candidates.extend(_oracle_config_candidates_from_mapping(payload))
    for event in events:
        settings = event.get("settings")
        if isinstance(settings, Mapping):
            candidates.append(settings)
        if any(str(key).startswith("phase3_oracle_") for key in event.keys()):
            candidates.append(event)
    return candidates


def _lookup(config: Mapping[str, Any], *paths: str, default: Any = None) -> Any:
    for path in paths:
        current: Any = config
        ok = True
        for part in path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                ok = False
                break
            current = current[part]
        if ok:
            return current
    return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _gate_values(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    if isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        return [str(item).strip() for item in raw if str(item).strip()]
    return [str(raw).strip()] if str(raw).strip() else []


def _gate_list_status(raw: Any) -> dict[str, Any]:
    error: str | None = None
    try:
        values = list(normalize_gate_name_tuple(raw, default=None, field_name="monitor gate list"))
    except ValueError as exc:
        values = _gate_values(raw)
        error = str(exc)
    bad = [value for value in values if any(sep in value for sep in (" ", "\t", ",", ";"))]
    return {
        "values": values,
        "present": bool(values),
        "split": bool(values) and error is None and not bad,
        "tokens_with_separators": bad,
        "normalization_error": error,
    }


def _full_noise_gate(configs: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, Any]]:
    cfg: Mapping[str, Any] = {}
    for candidate in reversed(list(configs)):
        if any(
            key in candidate
            for key in (
                "value_noise",
                "synthetic_depolarizing",
                "synthetic_coherent",
            )
        ) or any(str(key).startswith("phase3_oracle_") for key in candidate.keys()):
            cfg = candidate
            break
    value_noise = _lookup(cfg, "value_noise", default={})
    if not isinstance(value_noise, Mapping):
        value_noise = {}
    value_model = _lookup(cfg, "value_noise.model", "phase3_oracle_value_noise_model", default=value_noise.get("model", "off"))
    sigma0 = _lookup(cfg, "value_noise.sigma0_abs", "phase3_oracle_value_noise_sigma0_abs", default=value_noise.get("sigma0_abs"))
    n_eff = _lookup(cfg, "value_noise.N_eff", "phase3_oracle_value_noise_n_eff", default=value_noise.get("N_eff"))
    physical_shots = _lookup(cfg, "value_noise.physical_shots_unchanged", "physical_shots_unchanged", default=True)
    fixed_gate_claim = _lookup(
        cfg,
        "value_noise.fixed_gate_error_reduction_claimed",
        "fixed_gate_error_reduction_claimed",
        default=False,
    )
    value_pass = (
        str(value_model).strip().lower() == "gaussian_iid_v1"
        and _as_float(sigma0) > 0.0
        and _as_float(n_eff) > 0.0
        and bool(physical_shots) is True
        and bool(fixed_gate_claim) is False
    )

    depol_1q_error = _as_float(_lookup(cfg, "synthetic_depolarizing.one_qubit_error", "phase3_oracle_synthetic_depolarizing_1q_error", default=0.0))
    depol_2q_error = _as_float(_lookup(cfg, "synthetic_depolarizing.two_qubit_error", "phase3_oracle_synthetic_depolarizing_2q_error", default=0.0))
    depol_1q_gates = _gate_list_status(_lookup(cfg, "synthetic_depolarizing.one_qubit_gates", "phase3_oracle_synthetic_depolarizing_1q_gates", default=()))
    depol_2q_gates = _gate_list_status(_lookup(cfg, "synthetic_depolarizing.two_qubit_gates", "phase3_oracle_synthetic_depolarizing_2q_gates", default=()))
    depol_probs_bound = 0.0 <= depol_1q_error <= 1.0 and 0.0 <= depol_2q_error <= 1.0
    depol_positive = depol_1q_error > 0.0 or depol_2q_error > 0.0
    depol_split = True
    if depol_1q_error > 0.0:
        depol_split = depol_split and bool(depol_1q_gates["split"])
    if depol_2q_error > 0.0:
        depol_split = depol_split and bool(depol_2q_gates["split"])
    depol_pass = depol_positive and depol_probs_bound and depol_split

    coherent_1q_std = _as_float(_lookup(cfg, "synthetic_coherent.one_qubit_angle_std", "phase3_oracle_synthetic_coherent_1q_angle_std", default=0.0))
    coherent_2q_std = _as_float(_lookup(cfg, "synthetic_coherent.two_qubit_angle_std", "phase3_oracle_synthetic_coherent_2q_angle_std", default=0.0))
    coherent_enabled = coherent_1q_std > 0.0 or coherent_2q_std > 0.0
    coherent_inserted_count: int | None = None
    backend_metadata_present = False
    for candidate in configs:
        backend = candidate.get("oracle_backend_info") if isinstance(candidate, Mapping) else None
        if isinstance(backend, Mapping):
            details = backend.get("details")
            if isinstance(details, Mapping):
                backend_metadata_present = True
                coherent = details.get("synthetic_coherent")
                if isinstance(coherent, Mapping) and "inserted_count" in coherent:
                    try:
                        coherent_inserted_count = int(coherent.get("inserted_count"))
                    except Exception:
                        coherent_inserted_count = None
    for event in events:
        text = f"{event.get('error_repr', '')} {event.get('error', '')}"
        if "inserted zero gate errors" in text or "inserted zero" in text:
            coherent_inserted_count = 0
    coherent_pass = bool(coherent_enabled and coherent_inserted_count is not None and coherent_inserted_count > 0)

    status = "pass" if value_pass and depol_pass and coherent_pass else "fail"
    details = {
        "status": status,
        "value_noise": {
            "pass": value_pass,
            "model": value_model,
            "sigma0_abs": sigma0,
            "N_eff": n_eff,
            "physical_shots_unchanged": bool(physical_shots),
            "fixed_gate_error_reduction_claimed": bool(fixed_gate_claim),
        },
        "depolarizing": {
            "pass": depol_pass,
            "positive": depol_positive,
            "probabilities_bound": depol_probs_bound,
            "split": depol_split,
            "one_qubit_error": depol_1q_error,
            "two_qubit_error": depol_2q_error,
            "one_qubit_gates": depol_1q_gates,
            "two_qubit_gates": depol_2q_gates,
        },
        "coherent": {
            "pass": coherent_pass,
            "enabled": coherent_enabled,
            "inserted_count": coherent_inserted_count,
            "backend_metadata_present": backend_metadata_present,
            "one_qubit_angle_std": coherent_1q_std,
            "two_qubit_angle_std": coherent_2q_std,
        },
    }
    summary = (
        f"{status}(value_noise={'pass' if value_pass else 'fail'}; "
        f"depolarizing_split={'pass' if depol_split else 'fail'}; "
        f"depolarizing_positive={'pass' if depol_positive else 'fail'}; "
        f"coherent_inserted_count={coherent_inserted_count})"
    )
    return summary, details


def _source_lock_gate(events: Sequence[Mapping[str, Any]], json_payloads: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, Any]]:
    if any(bool(payload.get("monitoring_source_lock_not_applicable")) for payload in json_payloads):
        return "not_applicable(direct_oracle_probe)", {"status": "not_applicable", "reason": "direct_oracle_probe"}
    metas: list[Mapping[str, Any]] = []
    for event in events:
        meta = event.get("adapt_selected_logical_filter")
        if isinstance(meta, Mapping):
            metas.append(meta)
    for payload in json_payloads:
        adapt = payload.get("adapt_vqe") if isinstance(payload, Mapping) else None
        if isinstance(adapt, Mapping):
            for key in ("selected_logical_filter_meta", "adapt_selected_logical_filter"):
                meta = adapt.get(key)
                if isinstance(meta, Mapping):
                    metas.append(meta)
    if not metas:
        return "ambiguous(no_selected_logical_metadata)", {"status": "ambiguous", "reason": "no_selected_logical_metadata"}
    meta = metas[-1]
    source = str(meta.get("source_json") or meta.get("selected_logical_source_json") or "").strip()
    source_exists = bool(source and _repo_path(source).exists())
    fallback = bool(meta.get("fallback_to_full_pool"))
    applied = bool(meta.get("applied"))
    status = "pass" if applied and source_exists and not fallback else "fail"
    details = {
        "status": status,
        "applied": applied,
        "fallback_to_full_pool": fallback,
        "source_json": source or None,
        "source_exists": source_exists,
        "mode": meta.get("mode"),
        "fallback_reason": meta.get("fallback_reason"),
        "fallback_detail": meta.get("fallback_detail"),
    }
    summary_parts = [
        f"applied={str(applied).lower()}",
        f"source_exists={str(source_exists).lower()}",
        f"fallback_to_full_pool={str(fallback).lower()}",
    ]
    for key in ("selected_record_count", "active_pool_label_count", "matched_count"):
        if key in meta and meta.get(key) is not None:
            details[key] = meta.get(key)
            summary_parts.append(f"{key}={meta.get(key)}")
    for key in ("selected_label_family_samples", "missing_label_family_samples", "match_method_counts"):
        if key in meta:
            details[key] = meta.get(key)
    return (
        f"{status}(" + "; ".join(summary_parts) + ")",
        details,
    )


def _json_recoverability_gate(root: Path, json_files: Sequence[Path], json_payloads: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, Any]]:
    current_paths = [path for path in json_files if "current" in path.name]
    result_paths = [path for path in json_files if "result" in path.name]
    backend_metadata_present = False
    for payload in json_payloads:
        for cfg in _oracle_config_candidates_from_mapping(payload):
            backend = cfg.get("oracle_backend_info") if isinstance(cfg, Mapping) else None
            if isinstance(backend, Mapping) and isinstance(backend.get("details"), Mapping):
                backend_metadata_present = True
    stdout_parseable = bool(events)
    current_parseable = bool(current_paths)
    result_parseable = bool(result_paths)
    status = "pass" if (current_parseable or result_parseable or stdout_parseable) and backend_metadata_present else "partial"
    if not (current_parseable or result_parseable or stdout_parseable):
        status = "fail"
    details = {
        "status": status,
        "stdout_ai_log_parseable": stdout_parseable,
        "json_file_count": len(json_files),
        "current_json_parseable": current_parseable,
        "current_json_paths": [_safe_relative(p) for p in current_paths[:10]],
        "result_json_parseable": result_parseable,
        "result_json_paths": [_safe_relative(p) for p in result_paths[:10]],
        "backend_metadata_present": backend_metadata_present,
        "artifact_root_exists": root.exists(),
    }
    return (
        f"{status}(stdout_ai_log_parseable={str(stdout_parseable).lower()}; current_json_parseable={str(current_parseable).lower()}; result_json_parseable={str(result_parseable).lower()}; backend_metadata_present={str(backend_metadata_present).lower()})",
        details,
    )


def _depth_from_events_or_json(events: Sequence[Mapping[str, Any]], json_payloads: Sequence[Mapping[str, Any]]) -> int | None:
    depths: list[int] = []
    for event in events:
        for key in ("depth", "ansatz_depth", "current_depth"):
            if key in event:
                try:
                    depths.append(int(event[key]))
                    break
                except Exception:
                    pass
    for payload in json_payloads:
        adapt = payload.get("adapt_vqe") if isinstance(payload, Mapping) else None
        if isinstance(adapt, Mapping):
            for key in ("ansatz_depth", "depth", "current_depth", "accepted_operator_count"):
                if key in adapt:
                    try:
                        depths.append(int(adapt[key]))
                        break
                    except Exception:
                        pass
    return max(depths) if depths else None


def _detection_from_events(events: Sequence[Mapping[str, Any]]) -> tuple[float | None, int | None, str | None]:
    if not events:
        return None, None, None
    start_ts = None
    for event in events:
        if isinstance(event.get("_parsed_ts"), datetime):
            start_ts = event["_parsed_ts"]
            break
    last_depth: int | None = None
    for event in events:
        if "depth" in event:
            try:
                last_depth = int(event.get("depth"))
            except Exception:
                pass
        name = str(event.get("event") or "")
        text = f"{event.get('error_repr', '')} {event.get('error', '')}"
        semantic_error = any(
            token in text
            for token in (
                "Synthetic coherent overrotation was configured but inserted zero gate errors",
                "inserted zero gate errors",
                "not a valid full-noise circuit-channel diagnostic",
            )
        )
        if semantic_error or "error_type" in event or name.endswith("_failed"):
            end_ts = event.get("_parsed_ts") if isinstance(event.get("_parsed_ts"), datetime) else None
            latency = None
            if start_ts is not None and end_ts is not None:
                latency = round(max(0.0, float((end_ts - start_ts).total_seconds())), 3)
            depth = last_depth
            if "depth" in event:
                try:
                    depth = int(event.get("depth"))
                except Exception:
                    pass
            reason = str(event.get("error_repr") or event.get("error") or name or "runtime_error")
            return latency, depth, reason
    return None, last_depth, None


def _stderr_tail(root: Path, *, max_chars: int = 1000) -> str | None:
    stderr = root / "stderr.log"
    if not stderr.exists():
        return None
    try:
        text = stderr.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    tail = text[-max_chars:].strip()
    return tail or None


def capture_remote_runner_status() -> dict[str, Any]:
    started = time.monotonic()
    try:
        result = subprocess.run(
            ["curl", "-fsS", REMOTE_RUNNER_STATUS_URL],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except Exception as exc:
        return {"status": "not_available", "error": f"{type(exc).__name__}: {exc}"}
    payload: Any = None
    if result.stdout.strip():
        try:
            payload = json.loads(result.stdout)
        except Exception:
            payload = result.stdout.strip()[:500]
    return {
        "status": "ok" if result.returncode == 0 else "error",
        "returncode": int(result.returncode),
        "elapsed_s": round(float(time.monotonic() - started), 3),
        "payload": payload,
        "stderr": result.stderr.strip()[:500] if result.stderr.strip() else None,
    }


def collect_artifact_metrics(root: str | Path) -> dict[str, Any]:
    artifact_root = _repo_path(root)
    stdout_path = artifact_root / "stdout.log"
    events = parse_ai_log_events(stdout_path)
    json_files = _iter_json_files(artifact_root)
    json_payloads: list[Mapping[str, Any]] = []
    json_errors: dict[str, str] = {}
    for path in json_files:
        payload, error = _read_json(path)
        if payload is None:
            if error:
                json_errors[_safe_relative(path)] = error
            continue
        json_payloads.append(payload)

    configs = _oracle_config_candidates(events, json_payloads)
    full_noise_summary, full_noise_details = _full_noise_gate(configs, events)
    source_summary, source_details = _source_lock_gate(events, json_payloads)
    json_summary, json_details = _json_recoverability_gate(artifact_root, json_files, json_payloads, events)
    latency, detection_depth, rejection_reason = _detection_from_events(events)
    depth = detection_depth if detection_depth is not None else _depth_from_events_or_json(events, json_payloads)
    if rejection_reason is None:
        rejection_reason = _stderr_tail(artifact_root)

    evidence_paths: list[str] = []
    for path in (stdout_path, artifact_root / "stderr.log"):
        if path.exists():
            evidence_paths.append(_safe_relative(path))
    evidence_paths.extend(_safe_relative(path) for path in json_files[:12])
    if len(json_files) > 12:
        evidence_paths.append(f"... {len(json_files) - 12} more json files under {_safe_relative(artifact_root)}")

    return {
        "artifact_root": artifact_root,
        "events": events,
        "json_files": json_files,
        "json_errors": json_errors,
        "json_payload_count": len(json_payloads),
        "full_noise_active_gate": full_noise_summary,
        "full_noise_active_gate_details": full_noise_details,
        "source_lock_active_gate": source_summary,
        "source_lock_active_gate_details": source_details,
        "json_recoverability_gate": json_summary,
        "json_recoverability_gate_details": json_details,
        "invalid_run_detection_latency_s": latency,
        "invalid_run_detection_depth": depth,
        "rejection_reason": rejection_reason,
        "evidence_paths": evidence_paths,
    }


def classify_observed_validity(expected: str, metrics: Mapping[str, Any]) -> str:
    expected_norm = str(expected or "unknown").strip().lower()
    reason = str(metrics.get("rejection_reason") or "")
    full_gate = str(metrics.get("full_noise_active_gate") or "")
    source_gate = str(metrics.get("source_lock_active_gate") or "")
    json_gate = str(metrics.get("json_recoverability_gate") or "")
    latency = metrics.get("invalid_run_detection_latency_s")
    semantic_rejection = bool(reason and ("Synthetic coherent" in reason or "full-noise" in reason or "ADAPT-VQE failed" in reason))
    if expected_norm == "invalid" and semantic_rejection:
        return "rejected"
    source_ok = source_gate.startswith("pass") or source_gate.startswith("not_applicable")
    if expected_norm == "valid" and full_gate.startswith("pass") and source_ok and json_gate.startswith("pass"):
        return "valid"
    if expected_norm == "invalid" and latency is None:
        return "ambiguous"
    if reason:
        return "failed"
    return "ambiguous"


def build_scoreboard_row(
    *,
    metrics: Mapping[str, Any],
    iteration_id: str,
    attributed_change: str,
    command: str,
    case_id: str,
    expected_validity: str,
    observed_run_validity: str | None = None,
    remote_runner_status_snapshot: Mapping[str, Any] | str | None = None,
    follow_up: str = "",
) -> dict[str, Any]:
    observed = observed_run_validity or classify_observed_validity(expected_validity, metrics)
    artifact_root = metrics.get("artifact_root")
    return {
        "timestamp_utc": utc_now_iso(),
        "iteration_id": iteration_id,
        "attributed_change": attributed_change,
        "command": command,
        "case_id": case_id,
        "smoke_root": _safe_relative(artifact_root) if artifact_root is not None else "",
        "expected_validity": expected_validity,
        "observed_run_validity": observed,
        "invalid_run_detection_latency_s": metrics.get("invalid_run_detection_latency_s"),
        "invalid_run_detection_depth": metrics.get("invalid_run_detection_depth"),
        "full_noise_active_gate": metrics.get("full_noise_active_gate"),
        "source_lock_active_gate": metrics.get("source_lock_active_gate"),
        "json_recoverability_gate": metrics.get("json_recoverability_gate"),
        "remote_runner_status_snapshot": remote_runner_status_snapshot or "not_captured",
        "rejection_reason": metrics.get("rejection_reason") or "",
        "evidence_paths": metrics.get("evidence_paths") or [],
        "follow_up": follow_up,
    }


def _write_probe_ai_log(path: Path, events: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write("AI_LOG " + json.dumps(_jsonable(dict(event)), sort_keys=True, default=str) + "\n")


def _backend_info_payload(backend_info: Any) -> dict[str, Any]:
    if backend_info is None:
        return {}
    if is_dataclass(backend_info):
        return dict(_jsonable(asdict(backend_info)))
    if isinstance(backend_info, Mapping):
        return dict(_jsonable(backend_info))
    payload: dict[str, Any] = {}
    for key in ("noise_mode", "backend_name", "using_fake_backend", "details"):
        if hasattr(backend_info, key):
            payload[key] = _jsonable(getattr(backend_info, key))
    return payload


def run_oracle_probe(output_root: str | Path, *, gate_style: str, expected_validity: str) -> dict[str, Any]:
    root = _repo_path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    start_ts = utc_now_iso()
    events: list[dict[str, Any]] = [
        {
            "event": "full_noise_monitoring_oracle_probe_start",
            "ts_utc": start_ts,
            "depth": 0,
            "gate_style": gate_style,
        }
    ]
    if gate_style == "split":
        oneq_gates = ("x", "sx", "rx", "ry", "h")
        twoq_gates = ("cx", "cz", "ecr")
    elif gate_style == "bad":
        oneq_gates = ("definitely_not_compiled_gate",)
        twoq_gates = ("definitely_not_compiled_gate",)
    else:
        oneq_gates = ("x sx rx ry h",)
        twoq_gates = ("cx cz ecr",)

    evidence: dict[str, Any] = {
        "schema": "paper_i_full_noise_monitoring_oracle_probe_v1",
        "monitoring_source_lock_not_applicable": True,
        "expected_validity": expected_validity,
        "gate_style": gate_style,
        "adapt_vqe": {
            "ansatz_depth": 0,
            "continuation": {
                "oracle_gradient_config": {
                    "noise_mode": "aer_density_matrix_synthetic_coherent",
                    "execution_surface": "expectation_v1",
                    "value_noise": {
                        "enabled": True,
                        "model": "gaussian_iid_v1",
                        "sigma0_abs": 1.0e-3,
                        "N_eff": 400.0,
                        "std": 5.0e-5,
                        "physical_shots_unchanged": True,
                        "fixed_gate_error_reduction_claimed": False,
                    },
                    "synthetic_depolarizing": {
                        "one_qubit_error": 1.0e-8,
                        "two_qubit_error": 1.0e-7,
                        "one_qubit_gates": list(oneq_gates),
                        "two_qubit_gates": list(twoq_gates),
                    },
                    "synthetic_coherent": {
                        "one_qubit_angle_std": 2.0e-4,
                        "two_qubit_angle_std": 6.0e-4,
                        "seed": 20260609,
                        "generator_mode": "random_pauli_frozen_v1",
                        "one_qubit_gates": list(oneq_gates),
                        "two_qubit_gates": list(twoq_gates),
                    },
                }
            },
        },
    }
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp
        from pipelines.exact_bench.noise_oracle_runtime import ExpectationOracle, OracleConfig

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        obs = SparsePauliOp.from_list([("ZZ", 1.0)])
        cfg = OracleConfig(
            noise_mode="aer_density_matrix_synthetic_coherent",
            seed=7,
            mitigation={"mode": "none"},
            symmetry_mitigation={"mode": "off"},
            value_noise_model="gaussian_iid_v1",
            value_noise_std=5.0e-5,
            value_noise_seed=20260610,
            value_noise_sigma0_abs=1.0e-3,
            value_noise_n_eff=400.0,
            value_noise_semantic="snake_function_value_noise_shot_equivalent_v1",
            value_noise_std_source="sigma0_abs_over_sqrt_N_eff",
            synthetic_depolarizing_1q_error=1.0e-8,
            synthetic_depolarizing_2q_error=1.0e-7,
            synthetic_depolarizing_1q_gates=oneq_gates,
            synthetic_depolarizing_2q_gates=twoq_gates,
            synthetic_coherent_1q_angle_std=2.0e-4,
            synthetic_coherent_2q_angle_std=6.0e-4,
            synthetic_coherent_seed=20260609,
            synthetic_coherent_1q_gates=oneq_gates,
            synthetic_coherent_2q_gates=twoq_gates,
        )
        with ExpectationOracle(cfg) as oracle:
            estimate = oracle.evaluate(qc, obs)
            backend_info = _backend_info_payload(oracle.backend_info)
        elapsed = round(float(time.monotonic() - started), 3)
        evidence["probe"] = {"status": "completed", "elapsed_s": elapsed, "estimate_mean": float(estimate.mean)}
        evidence["adapt_vqe"]["oracle_backend_info"] = backend_info
        events.append(
            {
                "event": "full_noise_monitoring_oracle_probe_completed",
                "ts_utc": utc_now_iso(),
                "depth": 0,
                "elapsed_s": elapsed,
            }
        )
    except Exception as exc:
        elapsed = round(float(time.monotonic() - started), 3)
        error = f"{type(exc).__name__}: {exc}"
        evidence["probe"] = {"status": "failed", "elapsed_s": elapsed, "error": error}
        events.append(
            {
                "event": "full_noise_monitoring_oracle_probe_failed",
                "ts_utc": utc_now_iso(),
                "depth": 0,
                "elapsed_s": elapsed,
                "error_type": type(exc).__name__,
                "error_repr": repr(exc),
                "error": error,
            }
        )
    (root / "probe_evidence.json").write_text(json.dumps(_jsonable(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_probe_ai_log(root / "stdout.log", events)
    return collect_artifact_metrics(root)


def _command_path_arg(path: str | Path) -> str:
    return _safe_relative(_repo_path(path))


def _depth1_smoke_noise_settings() -> dict[str, Any]:
    return {
        "phase3_oracle_gradient_mode": "aer_density_matrix_synthetic_coherent",
        "phase3_oracle_execution_surface": "expectation_v1",
        "phase3_oracle_value_noise_model": "gaussian_iid_v1",
        "phase3_oracle_value_noise_sigma0_abs": 0.001,
        "phase3_oracle_value_noise_n_eff": 400,
        "physical_shots_unchanged": True,
        "fixed_gate_error_reduction_claimed": False,
        "phase3_oracle_synthetic_depolarizing_1q_error": 1.0e-8,
        "phase3_oracle_synthetic_depolarizing_2q_error": 1.0e-7,
        "phase3_oracle_synthetic_depolarizing_1q_gates": "x,sx,rx,ry,h",
        "phase3_oracle_synthetic_depolarizing_2q_gates": "cx,cz,ecr",
        "phase3_oracle_synthetic_coherent_1q_angle_std": 2.0e-4,
        "phase3_oracle_synthetic_coherent_2q_angle_std": 6.0e-4,
        "phase3_oracle_synthetic_coherent_seed": 20260609,
        "phase3_oracle_synthetic_coherent_generator_mode": "random_pauli_frozen_v1",
        "phase3_oracle_synthetic_coherent_1q_gates": "x,sx,rx,ry,h",
        "phase3_oracle_synthetic_coherent_2q_gates": "cx,cz,ecr",
    }


def _build_depth1_adapt_smoke_command(
    output_root: str | Path,
    *,
    selected_source: str | Path = DEFAULT_DEPTH1_SELECTED_LOGICAL_SOURCE,
) -> list[str]:
    root = _repo_path(output_root)
    selected = _repo_path(selected_source)
    current_json = root / "json" / "current.json"
    result_json = root / "json" / "result.json"
    return [
        sys.executable,
        "-u",
        "-m",
        "pipelines.static_adapt.adapt_pipeline",
        "--problem",
        "hh",
        "--L",
        "2",
        "--t",
        "1.0",
        "--u",
        "1.25",
        "--dv",
        "0.0",
        "--omega0",
        "1.0",
        "--g-ep",
        "0.3535533905932738",
        "--n-ph-max",
        "2",
        "--boson-encoding",
        "binary",
        "--ordering",
        "blocked",
        "--boundary",
        "open",
        "--adapt-pool",
        "math_md_full_meta_v1",
        "--adapt-continuation-mode",
        "phase3_v1",
        "--adapt-segment-target-depth",
        "1",
        "--adapt-segment-max-new-admissions",
        "1",
        "--adapt-current-json",
        _command_path_arg(current_json),
        "--adapt-current-json-every-depth",
        "1",
        "--adapt-current-json-keep-history-tail",
        "100",
        "--output-json",
        _command_path_arg(result_json),
        "--skip-pdf",
        "--skip-trajectory",
        "--adapt-selected-logical-source-json",
        _command_path_arg(selected),
        "--adapt-selected-logical-mode",
        "family_closure_fail_closed",
        "--adapt-selected-logical-transfer-mode",
        "exact_match_v1",
        "--phase3-oracle-gradient-mode",
        "aer_density_matrix_synthetic_coherent",
        "--phase3-oracle-execution-surface",
        "expectation_v1",
        "--phase3-oracle-value-noise-model",
        "gaussian_iid_v1",
        "--phase3-oracle-value-noise-sigma0-abs",
        "0.001",
        "--phase3-oracle-value-noise-n-eff",
        "400",
        "--phase3-oracle-synthetic-depolarizing-1q-error",
        "1e-8",
        "--phase3-oracle-synthetic-depolarizing-2q-error",
        "1e-7",
        "--phase3-oracle-synthetic-depolarizing-1q-gates",
        "x,sx,rx,ry,h",
        "--phase3-oracle-synthetic-depolarizing-2q-gates",
        "cx,cz,ecr",
        "--phase3-oracle-synthetic-coherent-1q-angle-std",
        "2e-4",
        "--phase3-oracle-synthetic-coherent-2q-angle-std",
        "6e-4",
        "--phase3-oracle-synthetic-coherent-seed",
        "20260609",
        "--phase3-oracle-synthetic-coherent-generator-mode",
        "random_pauli_frozen_v1",
        "--phase3-oracle-synthetic-coherent-1q-gates",
        "x,sx,rx,ry,h",
        "--phase3-oracle-synthetic-coherent-2q-gates",
        "cx,cz,ecr",
    ]


def _depth1_source_lock_pool_match_report(selected_source: str | Path) -> dict[str, Any]:
    request = ProblemRequest(
        problem_key="hh",
        num_sites=2,
        t=1.0,
        u=1.25,
        dv=0.0,
        omega0=1.0,
        g_ep=0.3535533905932738,
        n_ph_max=2,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        include_zero_point=True,
    )
    resolved = resolve_problem_context(request)
    pool_filters = resolve_requested_pool_filters(
        problem_key="hh",
        num_sites=2,
        n_ph_max=2,
        adapt_pool="math_md_full_meta_v1",
        adapt_pool_class_filter_json=None,
        adapt_pool_label_filter_json=None,
    )
    pool_plan = resolve_pool_plan(
        resolved_problem=resolved,
        continuation_mode="phase3_v1",
        adapt_pool="math_md_full_meta_v1",
        paop_r=0,
        paop_split_paulis=False,
        paop_prune_eps=0.0,
        paop_normalization="none",
        phase3_symmetry_mitigation_mode="off",
        filter_resolution=pool_filters,
    )
    report = build_selected_logical_pool_match_report(
        pool=pool_plan.pool,
        pool_family_ids=pool_plan.pool_family_ids,
        pool_symmetry_specs=pool_plan.pool_symmetry_specs,
        pool_generator_registry=pool_plan.pool_generator_registry,
        request=resolved.request,
        qpb=int(pool_plan.qpb),
        mode="family_closure_fail_closed",
        source_json=_repo_path(selected_source),
        transfer_mode="exact_match_v1",
        paop_split_paulis=False,
    )
    if report.get("source_json"):
        report["source_json"] = _safe_relative(report.get("source_json"))
    report.update(
        {
            "smoke_problem": "hh",
            "smoke_L": 2,
            "smoke_n_ph_max": 2,
            "smoke_adapt_pool": "math_md_full_meta_v1",
            "smoke_continuation_mode": "phase3_v1",
            "smoke_ordering": "blocked",
            "smoke_boson_encoding": "binary",
            "pool_key": pool_plan.pool_key,
            "method_name": pool_plan.method_name,
        }
    )
    return report


def _preflight_depth1_selected_logical_source(selected_source: str | Path) -> dict[str, Any]:
    source = _repo_path(selected_source)
    evidence: dict[str, Any] = {
        "schema": "depth1_adapt_smoke_source_lock_preflight_v1",
        "attributed_change": "source_lock_pool_match_preflight_v1",
        "source_json": _safe_relative(source),
        "source_exists": source.exists(),
        "json_parseable": False,
        "record_count": None,
        "selected_record_count": None,
        "active_pool_label_count": None,
        "matched_count": None,
        "fallback_to_full_pool": False,
        "status": "fail",
        "reason": None,
    }
    if not source.exists():
        evidence["reason"] = "source_json_missing"
        return evidence
    try:
        report = _depth1_source_lock_pool_match_report(source)
    except Exception as exc:
        evidence["reason"] = "source_lock_pool_match_preflight_error"
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        return evidence
    evidence["pool_match_report"] = dict(report)
    evidence["json_parseable"] = report.get("reason") != "malformed_source_json"
    evidence["record_count"] = report.get("selected_record_count")
    evidence["selected_record_count"] = report.get("selected_record_count")
    evidence["active_pool_label_count"] = report.get("active_pool_label_count")
    evidence["matched_count"] = report.get("matched_count")
    evidence["selected_label_family_samples"] = report.get("selected_label_family_samples") or []
    evidence["missing_label_family_samples"] = report.get("missing_label_family_samples") or []
    evidence["active_pool_label_samples"] = report.get("active_pool_label_samples") or []
    evidence["match_method_counts"] = report.get("match_method_counts") or {}
    evidence["fallback_to_full_pool"] = bool(report.get("fallback_to_full_pool", False))
    if report.get("status") == "pass" and int(report.get("matched_count") or 0) > 0:
        evidence["status"] = "pass"
        evidence["reason"] = None
        return evidence
    evidence["reason"] = str(report.get("semantic_reason") or report.get("reason") or "source_lock_pool_match_failed")
    return evidence


def _source_lock_preflight_filter_meta(preflight: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "adapt_selected_logical_pool_filter_v1",
        "mode": "family_closure_fail_closed",
        "transfer_mode": "exact_match_v1",
        "applied": False,
        "fallback_to_full_pool": False,
        "fallback_reason": "source_lock_preflight_failed",
        "fallback_detail": preflight.get("reason"),
        "source_json": preflight.get("source_json"),
        "source_exists": preflight.get("source_exists"),
        "selected_record_count": preflight.get("selected_record_count") or preflight.get("record_count"),
        "active_pool_label_count": preflight.get("active_pool_label_count"),
        "matched_count": preflight.get("matched_count"),
        "selected_label_family_samples": preflight.get("selected_label_family_samples") or [],
        "missing_label_family_samples": preflight.get("missing_label_family_samples") or [],
        "match_method_counts": preflight.get("match_method_counts") or {},
    }


def _append_monitor_ai_log_event(stdout_path: str | Path, event_payload: Mapping[str, Any]) -> None:
    path = _repo_path(stdout_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    event = dict(event_payload)
    event.setdefault("ts_utc", utc_now_iso())
    with path.open("a", encoding="utf-8") as fh:
        fh.write("AI_LOG " + json.dumps(_jsonable(event), sort_keys=True, default=str) + "\n")


def _write_depth1_command_script(path: str | Path, command: Sequence[str]) -> Path:
    script = _repo_path(path)
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {shlex.quote(str(REPO_ROOT))}\n"
        f"{shlex.join([str(part) for part in command])}\n",
        encoding="utf-8",
    )
    try:
        script.chmod(0o755)
    except Exception:
        pass
    return script


def _run_depth1_adapt_smoke_subprocess(
    command: Sequence[str],
    output_root: str | Path,
    *,
    timeout_s: float | None = None,
) -> subprocess.CompletedProcess[str]:
    root = _repo_path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "json").mkdir(parents=True, exist_ok=True)
    _write_depth1_command_script(root / "command.sh", command)
    stdout_path = root / "stdout.log"
    stderr_path = root / "stderr.log"
    started = time.monotonic()
    _append_monitor_ai_log_event(
        stdout_path,
        {
            "event": "depth1_adapt_smoke_subprocess_start",
            "depth": 0,
            "command": [str(part) for part in command],
            "output_root": _safe_relative(root),
        },
    )
    try:
        with stdout_path.open("a", encoding="utf-8") as stdout_fh, stderr_path.open("w", encoding="utf-8") as stderr_fh:
            result = subprocess.run(
                [str(part) for part in command],
                cwd=str(REPO_ROOT),
                stdout=stdout_fh,
                stderr=stderr_fh,
                text=True,
                timeout=timeout_s,
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        elapsed = round(float(time.monotonic() - started), 3)
        with stderr_path.open("a", encoding="utf-8") as stderr_fh:
            stderr_fh.write(f"TimeoutExpired after {timeout_s}s: {exc}\n")
        _append_monitor_ai_log_event(
            stdout_path,
            {
                "event": "depth1_adapt_smoke_subprocess_failed",
                "depth": 0,
                "elapsed_s": elapsed,
                "returncode": 124,
                "error_type": "TimeoutExpired",
                "error": f"TimeoutExpired after {timeout_s}s",
            },
        )
        return subprocess.CompletedProcess([str(part) for part in command], 124)
    elapsed = round(float(time.monotonic() - started), 3)
    event_name = "depth1_adapt_smoke_subprocess_completed" if result.returncode == 0 else "depth1_adapt_smoke_subprocess_failed"
    _append_monitor_ai_log_event(
        stdout_path,
        {
            "event": event_name,
            "depth": 1 if result.returncode == 0 else 0,
            "elapsed_s": elapsed,
            "returncode": int(result.returncode),
        },
    )
    return result


def _source_lock_fail_gate_summary(
    *,
    source_exists: bool,
    selected_record_count: Any = None,
    active_pool_label_count: Any = None,
    matched_count: Any = None,
) -> str:
    parts = [
        "applied=false",
        f"source_exists={str(bool(source_exists)).lower()}",
        "fallback_to_full_pool=false",
    ]
    for key, value in (
        ("selected_record_count", selected_record_count),
        ("active_pool_label_count", active_pool_label_count),
        ("matched_count", matched_count),
    ):
        if value is not None:
            parts.append(f"{key}={value}")
    return "fail(" + "; ".join(parts) + ")"


def _override_metrics_for_source_lock_setup_failure(
    metrics: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    out = dict(metrics)
    source_exists = bool(preflight.get("source_exists"))
    selected_record_count = preflight.get("selected_record_count") or preflight.get("record_count")
    active_pool_label_count = preflight.get("active_pool_label_count")
    matched_count = preflight.get("matched_count")
    details = {
        "status": "fail",
        "applied": False,
        "fallback_to_full_pool": False,
        "source_json": preflight.get("source_json"),
        "source_exists": source_exists,
        "mode": "family_closure_fail_closed",
        "fallback_reason": "source_lock_preflight_failed",
        "fallback_detail": preflight.get("reason"),
        "selected_record_count": selected_record_count,
        "active_pool_label_count": active_pool_label_count,
        "matched_count": matched_count,
        "selected_label_family_samples": preflight.get("selected_label_family_samples") or [],
        "missing_label_family_samples": preflight.get("missing_label_family_samples") or [],
        "match_method_counts": preflight.get("match_method_counts") or {},
    }
    out["source_lock_active_gate"] = _source_lock_fail_gate_summary(
        source_exists=source_exists,
        selected_record_count=selected_record_count,
        active_pool_label_count=active_pool_label_count,
        matched_count=matched_count,
    )
    out["source_lock_active_gate_details"] = details
    out["full_noise_active_gate"] = "not_evaluated(source_lock_preflight_failed)"
    out["full_noise_active_gate_details"] = {
        "status": "not_evaluated",
        "reason": "source_lock_preflight_failed",
        "source_lock_reason": preflight.get("reason"),
    }
    out["invalid_run_detection_depth"] = 0
    evidence_paths = list(out.get("evidence_paths") or [])
    preflight_path = preflight.get("evidence_path")
    if preflight_path and str(preflight_path) not in evidence_paths:
        evidence_paths.append(str(preflight_path))
    reason_parts = [f"source_lock_preflight_failed:{preflight.get('reason') or 'unknown'}"]
    for key, value in (
        ("selected_record_count", selected_record_count),
        ("active_pool_label_count", active_pool_label_count),
        ("matched_count", matched_count),
    ):
        if value is not None:
            reason_parts.append(f"{key}={value}")
    if preflight_path:
        reason_parts.append(f"evidence_path={preflight_path}")
    out["rejection_reason"] = " ".join(reason_parts)
    out["evidence_paths"] = evidence_paths
    return out


def _is_source_lock_runtime_failure(reason: Any) -> bool:
    text = str(reason or "")
    return "selected-logical pool filter failed closed:" in text


def _override_metrics_for_source_lock_runtime_failure(
    metrics: Mapping[str, Any],
    *,
    selected_source: str | Path,
    reason: Any,
) -> dict[str, Any]:
    out = dict(metrics)
    source = _repo_path(selected_source)
    detail = str(reason or "selected-logical pool filter failed closed")
    out["source_lock_active_gate"] = _source_lock_fail_gate_summary(source_exists=source.exists())
    out["source_lock_active_gate_details"] = {
        "status": "fail",
        "applied": False,
        "fallback_to_full_pool": False,
        "source_json": _safe_relative(source),
        "source_exists": source.exists(),
        "mode": "family_closure_fail_closed",
        "fallback_reason": "source_lock_runtime_failed_closed",
        "fallback_detail": detail,
    }
    out["invalid_run_detection_depth"] = 0
    out["rejection_reason"] = f"source_lock_runtime_failed_closed:{detail}"
    return out


def _depth1_command_label(output_root: str | Path) -> str:
    return f"depth1-adapt-smoke:{_safe_relative(_repo_path(output_root) / 'command.sh')}"


def _cmd_depth1_adapt_smoke(args: argparse.Namespace) -> int:
    root = _repo_path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "json").mkdir(parents=True, exist_ok=True)
    selected_source = _repo_path(args.selected_logical_source_json)
    command = _build_depth1_adapt_smoke_command(root, selected_source=selected_source)
    _write_depth1_command_script(root / "command.sh", command)

    preflight = _preflight_depth1_selected_logical_source(selected_source)
    preflight_path = root / "source_lock_preflight_evidence.json"
    preflight["evidence_path"] = _safe_relative(preflight_path)
    preflight_path.write_text(json.dumps(_jsonable(preflight), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if preflight.get("status") != "pass":
        stdout_path = root / "stdout.log"
        (root / "stderr.log").write_text(
            f"source_lock_preflight_failed:{preflight.get('reason') or 'unknown'}\n",
            encoding="utf-8",
        )
        _append_monitor_ai_log_event(
            stdout_path,
            {
                "event": "depth1_adapt_smoke_source_lock_preflight_start",
                "depth": 0,
                "settings": _depth1_smoke_noise_settings(),
                "adapt_selected_logical_filter": _source_lock_preflight_filter_meta(preflight),
            },
        )
        _append_monitor_ai_log_event(
            stdout_path,
            {
                "event": "depth1_adapt_smoke_source_lock_preflight_failed",
                "depth": 0,
                "error_type": "SourceLockPreflightError",
                "error": f"source_lock_preflight_failed:{preflight.get('reason') or 'unknown'}",
                "adapt_selected_logical_filter": _source_lock_preflight_filter_meta(preflight),
            },
        )
        metrics = _override_metrics_for_source_lock_setup_failure(collect_artifact_metrics(root), preflight)
        remote_snapshot: Mapping[str, Any] | str = "not_captured"
        if args.capture_remote_runner_status:
            remote_snapshot = capture_remote_runner_status()
        row = build_scoreboard_row(
            metrics=metrics,
            iteration_id=args.iteration_id,
            attributed_change=args.attributed_change,
            command=_depth1_command_label(root),
            case_id=args.case_id,
            expected_validity=args.expected_validity,
            observed_run_validity=args.observed_run_validity or "rejected",
            remote_runner_status_snapshot=remote_snapshot,
            follow_up=args.follow_up,
        )
        append_scoreboard_row(args.scoreboard, row)
        print(json.dumps(_jsonable(row), indent=2, sort_keys=True))
        return 4

    result = _run_depth1_adapt_smoke_subprocess(command, root, timeout_s=args.timeout_s)
    metrics = collect_artifact_metrics(root)
    if _is_source_lock_runtime_failure(metrics.get("rejection_reason")):
        metrics = _override_metrics_for_source_lock_runtime_failure(
            metrics,
            selected_source=selected_source,
            reason=metrics.get("rejection_reason"),
        )
        observed_override = args.observed_run_validity or "rejected"
    else:
        observed_override = args.observed_run_validity
    if result.returncode != 0 and not metrics.get("rejection_reason"):
        metrics = dict(metrics)
        metrics["rejection_reason"] = f"depth1_adapt_smoke_returncode={int(result.returncode)}"
    remote_snapshot = "not_captured"
    if args.capture_remote_runner_status:
        remote_snapshot = capture_remote_runner_status()
    row = build_scoreboard_row(
        metrics=metrics,
        iteration_id=args.iteration_id,
        attributed_change=args.attributed_change,
        command=_depth1_command_label(root),
        case_id=args.case_id,
        expected_validity=args.expected_validity,
        observed_run_validity=observed_override,
        remote_runner_status_snapshot=remote_snapshot,
        follow_up=args.follow_up,
    )
    append_scoreboard_row(args.scoreboard, row)
    print(json.dumps(_jsonable(row), indent=2, sort_keys=True))
    if args.expected_validity == "valid" and row["observed_run_validity"] != "valid":
        return 2
    if args.expected_validity == "invalid" and row["observed_run_validity"] not in {"rejected", "failed"}:
        return 3
    return int(result.returncode)


def _cmd_init_scoreboard(args: argparse.Namespace) -> int:
    board = ensure_scoreboard(args.scoreboard)
    print(_safe_relative(board))
    return 0


def _cmd_collect_artifact(args: argparse.Namespace) -> int:
    metrics = collect_artifact_metrics(args.artifact_root)
    remote_snapshot: Mapping[str, Any] | str = "not_captured"
    if args.capture_remote_runner_status:
        remote_snapshot = capture_remote_runner_status()
    row = build_scoreboard_row(
        metrics=metrics,
        iteration_id=args.iteration_id,
        attributed_change=args.attributed_change,
        command=args.command,
        case_id=args.case_id,
        expected_validity=args.expected_validity,
        observed_run_validity=args.observed_run_validity,
        remote_runner_status_snapshot=remote_snapshot,
        follow_up=args.follow_up,
    )
    append_scoreboard_row(args.scoreboard, row)
    print(json.dumps(_jsonable(row), indent=2, sort_keys=True))
    return 0


def _cmd_oracle_probe(args: argparse.Namespace) -> int:
    metrics = run_oracle_probe(args.output_root, gate_style=args.gate_style, expected_validity=args.expected_validity)
    remote_snapshot: Mapping[str, Any] | str = "not_captured"
    if args.capture_remote_runner_status:
        remote_snapshot = capture_remote_runner_status()
    row = build_scoreboard_row(
        metrics=metrics,
        iteration_id=args.iteration_id,
        attributed_change=args.attributed_change,
        command=f"oracle-probe --gate-style {args.gate_style}",
        case_id=args.case_id,
        expected_validity=args.expected_validity,
        observed_run_validity=args.observed_run_validity,
        remote_runner_status_snapshot=remote_snapshot,
        follow_up=args.follow_up,
    )
    append_scoreboard_row(args.scoreboard, row)
    print(json.dumps(_jsonable(row), indent=2, sort_keys=True))
    if args.expected_validity == "valid" and row["observed_run_validity"] != "valid":
        return 2
    if args.expected_validity == "invalid" and row["observed_run_validity"] not in {"rejected", "failed"}:
        return 3
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command_name", required=True)

    init = sub.add_parser("init-scoreboard", help="Create the append-only scoreboard scaffold if missing.")
    init.add_argument("--scoreboard", default=str(DEFAULT_SCOREBOARD))
    init.set_defaults(func=_cmd_init_scoreboard)

    collect = sub.add_parser("collect-artifact", help="Collect monitoring gates from an existing artifact root and append one row.")
    collect.add_argument("--artifact-root", required=True)
    collect.add_argument("--scoreboard", default=str(DEFAULT_SCOREBOARD))
    collect.add_argument("--iteration-id", required=True)
    collect.add_argument("--attributed-change", default="baseline_no_change")
    collect.add_argument("--command", default="existing artifact collection")
    collect.add_argument("--case-id", required=True)
    collect.add_argument("--expected-validity", choices=("valid", "invalid", "unknown"), required=True)
    collect.add_argument("--observed-run-validity", choices=("valid", "rejected", "failed", "ambiguous", "environment_blocked"), default=None)
    collect.add_argument("--follow-up", default="")
    collect.add_argument("--capture-remote-runner-status", action="store_true")
    collect.set_defaults(func=_cmd_collect_artifact)

    probe = sub.add_parser("oracle-probe", help="Run a tiny direct full-noise oracle probe and append one row.")
    probe.add_argument("--output-root", required=True)
    probe.add_argument("--scoreboard", default=str(DEFAULT_SCOREBOARD))
    probe.add_argument("--iteration-id", required=True)
    probe.add_argument("--attributed-change", default="baseline_no_change")
    probe.add_argument("--case-id", default="direct_oracle_probe")
    probe.add_argument("--expected-validity", choices=("valid", "invalid"), required=True)
    probe.add_argument("--gate-style", choices=("split", "space", "bad"), required=True)
    probe.add_argument("--observed-run-validity", choices=("valid", "rejected", "failed", "ambiguous", "environment_blocked"), default=None)
    probe.add_argument("--follow-up", default="")
    probe.add_argument("--capture-remote-runner-status", action="store_true")
    probe.set_defaults(func=_cmd_oracle_probe)

    depth1 = sub.add_parser(
        "depth1-adapt-smoke",
        help="Run the monitor-owned depth-1 ADAPT/source-lock recoverable-JSON smoke and append one row.",
    )
    depth1.add_argument("--output-root", default=str(DEFAULT_DEPTH1_SMOKE_ROOT))
    depth1.add_argument("--selected-logical-source-json", default=str(DEFAULT_DEPTH1_SELECTED_LOGICAL_SOURCE))
    depth1.add_argument("--scoreboard", default=str(DEFAULT_SCOREBOARD))
    depth1.add_argument("--iteration-id", required=True)
    depth1.add_argument("--attributed-change", default="source_lock_pool_match_preflight_v1")
    depth1.add_argument("--case-id", required=True)
    depth1.add_argument("--expected-validity", choices=("valid", "invalid"), required=True)
    depth1.add_argument(
        "--observed-run-validity",
        choices=("valid", "rejected", "failed", "ambiguous", "environment_blocked"),
        default=None,
    )
    depth1.add_argument("--follow-up", default="")
    depth1.add_argument("--timeout-s", type=float, default=None)
    depth1.add_argument("--capture-remote-runner-status", action="store_true")
    depth1.set_defaults(func=_cmd_depth1_adapt_smoke)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
