#!/usr/bin/env python3
"""Focused recovery/planning helpers for fixed-scaffold HH runtime follow-ups.

This module stays in wrapper/report space. It does not alter the core HH operator
layers. It exists to support two practical follow-ups:
1) recover partial optimizer progress from interrupted Runtime job sequences
2) build a concise rerun plan from the current candidate/anchor evidence
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.exact_bench.noise_oracle_runtime import _fetch_runtime_job_record
from pipelines.hardcoded.imported_artifact_resolution import (
    resolve_default_hh_nighthawk_fixed_scaffold_artifact_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_JSON = REPO_ROOT / "artifacts" / "json"


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON payload at {path} is not an object.")
    return dict(payload)


def _repo_relative_str(path_like: str | Path | None) -> str | None:
    if path_like in {None, ""}:
        return None
    path = Path(path_like)
    if path.is_absolute():
        try:
            return str(path.resolve().relative_to(REPO_ROOT))
        except Exception:
            return str(path.resolve())
    return str(path)


def _artifact_adapt_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    adapt = payload.get("adapt_vqe", {}) if isinstance(payload, Mapping) else {}
    return adapt if isinstance(adapt, Mapping) else {}


def _artifact_parameterization(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    adapt = _artifact_adapt_payload(payload)
    param = adapt.get("parameterization", {}) if isinstance(adapt, Mapping) else {}
    return param if isinstance(param, Mapping) else {}


def _flatten_runtime_terms(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    param = _artifact_parameterization(payload)
    blocks = param.get("blocks", []) if isinstance(param, Mapping) else []
    out: list[dict[str, Any]] = []
    runtime_index = 0
    for block in blocks if isinstance(blocks, Sequence) else []:
        if not isinstance(block, Mapping):
            continue
        terms = block.get("runtime_terms_exyz", [])
        if not isinstance(terms, Sequence):
            continue
        for term in terms:
            if not isinstance(term, Mapping):
                continue
            out.append(
                {
                    "runtime_index": int(runtime_index),
                    "logical_index": int(block.get("logical_index", -1)),
                    "candidate_label": str(block.get("candidate_label", "")),
                    "pauli_exyz": str(term.get("pauli_exyz", "")),
                }
            )
            runtime_index += 1
    return out


def _aggregate_values(values: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {
            "mean": None,
            "stdev": None,
            "stderr": None,
            "n_samples": 0,
            "raw_values": [],
        }
    stdev = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    stderr = float(stdev / np.sqrt(float(arr.size))) if arr.size > 0 else float("nan")
    return {
        "mean": float(np.mean(arr)),
        "stdev": float(stdev),
        "stderr": float(stderr),
        "n_samples": int(arr.size),
        "raw_values": [float(x) for x in arr.tolist()],
    }


def _iter_json_payloads() -> list[tuple[Path, dict[str, Any]]]:
    rows: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(ARTIFACTS_JSON.glob("*.json")):
        try:
            rows.append((path, _load_json(path)))
        except Exception:
            continue
    rows.sort(key=lambda item: item[0].stat().st_mtime, reverse=True)
    return rows


def _find_latest_json_payload(predicate) -> tuple[Path | None, dict[str, Any] | None]:
    for path, payload in _iter_json_payloads():
        try:
            if bool(predicate(path, payload)):
                return path, payload
        except Exception:
            continue
    return None, None


def _find_compile_row(artifact_ref: str | Path) -> dict[str, Any] | None:
    path = ARTIFACTS_JSON / "investigation_marrakesh_compile_compare_20260323.json"
    if not path.exists():
        return None
    payload = _load_json(path)
    rows = payload.get("rows", []) if isinstance(payload, Mapping) else []
    target = _repo_relative_str(artifact_ref)
    for row in rows if isinstance(rows, Sequence) else []:
        if isinstance(row, Mapping) and str(row.get("artifact", "")) == str(target):
            return dict(row)
    return None


def _find_candidate_energy_only_runtime_eval(candidate_artifact_ref: str | Path) -> tuple[Path | None, dict[str, Any] | None]:
    target = _repo_relative_str(candidate_artifact_ref)
    return _find_latest_json_payload(
        lambda _path, payload: str(payload.get("pipeline", "")) == "hh_fixed_scaffold_energy_only_runtime_eval_v1"
        and str(payload.get("candidate_artifact_json", "")) == str(target)
    )


def _find_anchor_full_circuit_runtime_eval(anchor_artifact_ref: str | Path) -> tuple[Path | None, dict[str, Any] | None]:
    target = _repo_relative_str(anchor_artifact_ref)
    return _find_latest_json_payload(
        lambda _path, payload: str(payload.get("artifact_json", "")) == str(target)
        and isinstance(payload.get("result", {}), Mapping)
        and isinstance(payload.get("result", {}).get("final_observables", {}), Mapping)
        and "energy_static" in payload.get("result", {}).get("final_observables", {})
    )


def _find_anchor_savedparam_eval(anchor_artifact_ref: str | Path) -> tuple[Path | None, dict[str, Any] | None]:
    target = _repo_relative_str(anchor_artifact_ref)
    return _find_latest_json_payload(
        lambda _path, payload: str(payload.get("artifact_json", "")) == str(target)
        and isinstance(payload.get("unmitigated", None), Mapping)
        and isinstance(payload.get("mthree", None), Mapping)
        and "delta_mean" in payload.get("unmitigated", {})
        and "delta_mean" in payload.get("mthree", {})
    )


def _find_anchor_attribution(anchor_artifact_ref: str | Path) -> tuple[Path | None, dict[str, Any] | None]:
    target = _repo_relative_str(anchor_artifact_ref)

    def _pred(_path: Path, payload: Mapping[str, Any]) -> bool:
        rec = payload.get("fixed_scaffold_noise_attribution", {}) if isinstance(payload, Mapping) else {}
        return isinstance(rec, Mapping) and str(rec.get("artifact_json", "")) == str(target)

    return _find_latest_json_payload(_pred)


def _find_anchor_local_baseline(anchor_artifact_ref: str | Path) -> tuple[Path | None, dict[str, Any] | None]:
    target = _repo_relative_str(anchor_artifact_ref)
    return _find_latest_json_payload(
        lambda _path, payload: str(payload.get("artifact", payload.get("artifact_json", ""))) == str(target)
        and "exact_state_fidelity" in payload
    )


def _screen_budget(runtime_parameter_count: int, oracle_repeats: int) -> dict[str, Any]:
    rp = int(runtime_parameter_count)
    repeats = int(max(1, oracle_repeats))
    optimizer_calls = 1 + 2 * rp
    optimizer_jobs = repeats * optimizer_calls
    post_validation_jobs = repeats * 2
    return {
        "screen_maxiter": int(rp),
        "screen_objective_calls": int(optimizer_calls),
        "screen_optimizer_runtime_jobs": int(optimizer_jobs),
        "screen_post_validation_runtime_jobs": int(post_validation_jobs),
        "screen_total_runtime_jobs": int(optimizer_jobs + post_validation_jobs),
    }


def _sequence_not_text(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _extract_recovery_source_payload(payload: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    if isinstance(payload, Mapping) and (
        "objective_trace" in payload or "runtime_job_ids" in payload
    ):
        return dict(payload), "direct_recovery_payload"
    nested = payload.get("fixed_scaffold_noisy_replay", {}) if isinstance(payload, Mapping) else {}
    if isinstance(nested, Mapping) and (
        "objective_trace" in nested or "runtime_job_ids" in nested
    ):
        return dict(nested), "staged_fixed_scaffold_noisy_replay"
    return {}, "missing"


def _finite_float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    return numeric if np.isfinite(numeric) else None


def reconstruct_fixed_scaffold_runtime_recovery(
    *,
    artifact_json: str | Path,
    runtime_job_ids: Sequence[str] | None = None,
    recovery_source_json: str | Path | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    artifact_ref = _repo_relative_str(artifact_json)
    payload = _load_json(artifact_json)
    objective_trace: list[dict[str, Any]] = []
    recovery_granularity = "objective_call"
    recovery_source_payload: dict[str, Any] | None = None
    recovery_source_kind = "missing"
    explicit_runtime_job_ids_provided = any(
        str(job_id).strip()
        for job_id in (runtime_job_ids or [])
    )
    runtime_job_id_list = [
        str(job_id)
        for job_id in (runtime_job_ids or [])
        if str(job_id).strip()
    ]
    if recovery_source_json is not None and Path(recovery_source_json).exists():
        recovery_source_payload, recovery_source_kind = _extract_recovery_source_payload(
            _load_json(recovery_source_json)
        )
        src_trace = recovery_source_payload.get("objective_trace", [])
        if _sequence_not_text(src_trace):
            objective_trace = [dict(row) for row in src_trace if isinstance(row, Mapping)]
        if not runtime_job_id_list:
            src_runtime_job_ids = recovery_source_payload.get("runtime_job_ids", [])
            if _sequence_not_text(src_runtime_job_ids):
                runtime_job_id_list = [
                    str(job_id)
                    for job_id in src_runtime_job_ids
                    if str(job_id).strip()
                ]
    if explicit_runtime_job_ids_provided and objective_trace:
        merged_trace: list[dict[str, Any]] = []
        explicit_job_cursor = 0
        for idx, row in enumerate(objective_trace):
            merged_row = dict(row)
            source_runtime_jobs = (
                row.get("runtime_jobs", [])
                if isinstance(row, Mapping)
                else []
            )
            group_size = (
                len(source_runtime_jobs)
                if _sequence_not_text(source_runtime_jobs) and len(source_runtime_jobs) > 0
                else 1
            )
            row_job_ids = runtime_job_id_list[explicit_job_cursor : explicit_job_cursor + int(group_size)]
            explicit_job_cursor += len(row_job_ids)
            merged_row["_explicit_runtime_job_override"] = bool(row_job_ids)
            merged_row["runtime_jobs"] = [
                {"job_id": str(job_id)}
                for job_id in row_job_ids
            ]
            merged_trace.append(merged_row)
        for idx in range(explicit_job_cursor, len(runtime_job_id_list)):
            merged_trace.append(
                {
                    "trace_index": int(len(merged_trace) + 1),
                    "call_index": None,
                    "status": "pending",
                    "theta_runtime": None,
                    "theta_logical": None,
                    "_explicit_runtime_job_override": True,
                    "runtime_jobs": [{"job_id": str(runtime_job_id_list[idx])}],
                }
            )
        objective_trace = merged_trace
    if not objective_trace and runtime_job_id_list:
        recovery_granularity = "runtime_job"
        refreshed = [
            asdict(_fetch_runtime_job_record(str(job_id), require_result=True))
            for job_id in runtime_job_id_list
            if str(job_id).strip()
        ]
        refreshed.sort(key=lambda rec: str(rec.get("created_utc", "")))
        objective_trace = [
            {
                "trace_index": int(idx + 1),
                "call_index": None,
                "status": "completed" if rec.get("expectation_value", None) is not None else str(rec.get("status", "pending")).lower(),
                "theta_runtime": None,
                "theta_logical": None,
                "runtime_jobs": [dict(rec)],
            }
            for idx, rec in enumerate(refreshed)
        ]

    refreshed_rows: list[dict[str, Any]] = []
    best_so_far: dict[str, Any] | None = None
    runtime_jobs_total = 0
    runtime_jobs_completed = 0
    runtime_jobs_pending = 0
    runtime_jobs_failed = 0
    total_quantum_seconds = 0.0

    for row in sorted(
        objective_trace,
        key=lambda item: int(item.get("trace_index", item.get("call_index", 0) or 0)),
    ):
        explicit_job_override = bool(row.get("_explicit_runtime_job_override", False))
        runtime_jobs = row.get("runtime_jobs", []) if isinstance(row, Mapping) else []
        refreshed_jobs: list[dict[str, Any]] = []
        values: list[float] = []
        for job in runtime_jobs if isinstance(runtime_jobs, Sequence) else []:
            if not isinstance(job, Mapping):
                continue
            job_id = job.get("job_id", None)
            if job_id in {None, ""}:
                continue
            rec = asdict(_fetch_runtime_job_record(str(job_id), require_result=True))
            refreshed_jobs.append(rec)
            runtime_jobs_total += 1
            total_quantum_seconds += float(rec.get("usage_quantum_seconds") or 0.0)
            if rec.get("expectation_value", None) is not None:
                runtime_jobs_completed += 1
                values.append(float(rec["expectation_value"]))
            elif str(rec.get("status", "")).upper() in {"QUEUED", "RUNNING", "INITIALIZING", "VALIDATING"}:
                runtime_jobs_pending += 1
            else:
                runtime_jobs_failed += 1
        agg = _aggregate_values(values)
        source_status_raw = row.get("status", None)
        source_status = (
            str(source_status_raw).lower()
            if source_status_raw not in {None, ""}
            else ""
        )
        source_energy = _finite_float_or_none(row.get("energy_noisy_mean", None))
        source_stderr = _finite_float_or_none(row.get("energy_noisy_stderr", None))
        source_n_samples = 0
        try:
            source_n_samples = int(row.get("n_samples", 1) or 1)
        except Exception:
            source_n_samples = 1
        if source_n_samples < 1:
            source_n_samples = 1
        source_raw_values = row.get("raw_values", []) if isinstance(row, Mapping) else []
        source_raw_values_list = [
            float(x)
            for x in source_raw_values
            if _finite_float_or_none(x) is not None
        ] if _sequence_not_text(source_raw_values) else []

        status = "completed" if int(agg["n_samples"]) > 0 else str(source_status or "pending")
        if int(agg["n_samples"]) == 0 and source_energy is not None:
            if (
                source_status == "failed"
                or source_status == "pending"
                or source_status.startswith("partial")
            ):
                status = source_status
            else:
                status = "completed"
        if explicit_job_override and refreshed_jobs and int(agg["n_samples"]) == 0:
            if any(
                str(rec.get("status", "")).upper()
                in {"QUEUED", "RUNNING", "INITIALIZING", "VALIDATING"}
                for rec in refreshed_jobs
            ):
                status = "partial_pending"
            elif any(
                rec.get("error", None)
                or str(rec.get("status", "")).upper() in {"FAILED", "CANCELLED", "ERROR"}
                for rec in refreshed_jobs
            ):
                status = "failed"
            else:
                status = str(source_status or "pending")
        if status != "completed" and refreshed_jobs:
            if any(str(rec.get("status", "")).upper() in {"QUEUED", "RUNNING", "INITIALIZING", "VALIDATING"} for rec in refreshed_jobs):
                status = "partial_pending"
            elif any(rec.get("error", None) for rec in refreshed_jobs):
                status = "failed"
        energy_noisy_mean = agg["mean"]
        energy_noisy_stderr = agg["stderr"]
        sample_count = int(agg["n_samples"])
        raw_values = list(agg["raw_values"])
        if sample_count == 0 and source_energy is not None and not explicit_job_override:
            energy_noisy_mean = float(source_energy)
            energy_noisy_stderr = (
                float(source_stderr)
                if source_stderr is not None
                else 0.0
            )
            sample_count = int(source_n_samples)
            raw_values = (
                list(source_raw_values_list)
                if source_raw_values_list
                else [float(source_energy)]
            )
        refreshed_row = {
            "trace_index": int(row.get("trace_index", row.get("call_index", len(refreshed_rows) + 1) or len(refreshed_rows) + 1)),
            "call_index": (
                None
                if row.get("call_index", None) in {None, ""}
                else int(row.get("call_index"))
            ),
            "status": str(status),
            "theta_runtime": row.get("theta_runtime", None),
            "theta_logical": row.get("theta_logical", None),
            "elapsed_s": row.get("elapsed_s", None),
            "runtime_jobs": refreshed_jobs,
            "energy_noisy_mean": energy_noisy_mean,
            "energy_noisy_stderr": energy_noisy_stderr,
            "n_samples": int(sample_count),
            "raw_values": list(raw_values),
        }
        refreshed_rows.append(refreshed_row)
        if refreshed_row["energy_noisy_mean"] is None:
            continue
        challenger = {
            "trace_index": int(refreshed_row["trace_index"]),
            "call_index": refreshed_row.get("call_index", None),
            "energy_noisy_mean": float(refreshed_row["energy_noisy_mean"]),
            "energy_noisy_stderr": float(refreshed_row["energy_noisy_stderr"] or 0.0),
            "theta_runtime": refreshed_row.get("theta_runtime", None),
            "theta_logical": refreshed_row.get("theta_logical", None),
        }
        if best_so_far is None:
            best_so_far = challenger
            continue
        if float(challenger["energy_noisy_mean"]) < float(best_so_far["energy_noisy_mean"]):
            best_so_far = challenger
            continue
        if (
            float(challenger["energy_noisy_mean"]) == float(best_so_far["energy_noisy_mean"])
            and float(challenger["energy_noisy_stderr"]) < float(best_so_far["energy_noisy_stderr"])
        ):
            best_so_far = challenger

    result = {
        "generated_utc": _now_utc(),
        "pipeline": "hh_fixed_scaffold_runtime_recovery_v1",
        "artifact_json": str(artifact_ref),
        "recovery_source_json": _repo_relative_str(recovery_source_json),
        "reconstructed_from_runtime_jobs": bool(runtime_jobs_total > 0),
        "recovery_granularity": str(recovery_granularity),
        "success": True,
        "partial": True,
        "objective_trace": refreshed_rows,
        "best_so_far": best_so_far,
        "summary": {
            "trace_rows_total": int(len(refreshed_rows)),
            "trace_rows_completed": int(sum(1 for row in refreshed_rows if row.get("energy_noisy_mean") is not None)),
            "trace_rows_pending": int(sum(1 for row in refreshed_rows if str(row.get("status", "")).startswith("partial") or str(row.get("status", "")) == "pending")),
            "trace_rows_failed": int(sum(1 for row in refreshed_rows if str(row.get("status", "")) == "failed")),
            "objective_calls_total": (
                int(sum(1 for row in refreshed_rows if row.get("call_index", None) is not None))
                if str(recovery_granularity) == "objective_call"
                else None
            ),
            "objective_calls_completed": (
                int(
                    sum(
                        1
                        for row in refreshed_rows
                        if row.get("call_index", None) is not None
                        and row.get("energy_noisy_mean") is not None
                    )
                )
                if str(recovery_granularity) == "objective_call"
                else None
            ),
            "objective_calls_pending": (
                int(
                    sum(
                        1
                        for row in refreshed_rows
                        if row.get("call_index", None) is not None
                        and (
                            str(row.get("status", "")).startswith("partial")
                            or str(row.get("status", "")) == "pending"
                        )
                    )
                )
                if str(recovery_granularity) == "objective_call"
                else None
            ),
            "objective_calls_failed": (
                int(
                    sum(
                        1
                        for row in refreshed_rows
                        if row.get("call_index", None) is not None
                        and str(row.get("status", "")) == "failed"
                    )
                )
                if str(recovery_granularity) == "objective_call"
                else None
            ),
            "runtime_jobs_total": int(runtime_jobs_total),
            "runtime_jobs_completed": int(runtime_jobs_completed),
            "runtime_jobs_pending": int(runtime_jobs_pending),
            "runtime_jobs_failed": int(runtime_jobs_failed),
            "runtime_quantum_seconds_total": float(total_quantum_seconds),
        },
        "artifact_ground_truth": {
            "saved_energy": float(_artifact_adapt_payload(payload).get("energy", float("nan"))),
            "exact_energy": float(payload.get("ground_state", {}).get("exact_energy", float("nan"))),
            "runtime_parameter_count": int(_artifact_parameterization(payload).get("runtime_parameter_count", 0) or 0),
        },
        "notes": [
            (
                "Legacy interrupted runs without a theta journal can still recover job-level best-so-far energy, "
                "but they do not prove optimizer objective-call boundaries or a faithful theta trace."
                if str(recovery_granularity) == "runtime_job"
                else "Recovered from an objective-call trace source."
            ),
        ],
    }
    if str(recovery_source_kind) == "staged_fixed_scaffold_noisy_replay":
        result["notes"].append(
            "Recovered from fixed_scaffold_noisy_replay embedded in staged workflow output."
        )
    if bool(explicit_runtime_job_ids_provided):
        result["notes"].append(
            "Explicit runtime_job_ids took precedence over any embedded recovery-source job ids or objective trace."
        )
    if output_json is not None:
        _write_json(output_json, result)
    return result


def build_fixed_scaffold_rerun_plan(
    *,
    candidate_artifact_json: str | Path,
    recovery_json: str | Path | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    candidate_ref = _repo_relative_str(candidate_artifact_json)
    candidate_payload = _load_json(candidate_artifact_json)
    anchor_ref = candidate_payload.get("source_artifact_json", None)
    if anchor_ref in {None, ""}:
        resolved, _ = resolve_default_hh_nighthawk_fixed_scaffold_artifact_json()
        anchor_ref = _repo_relative_str(resolved)
    if anchor_ref in {None, ""}:
        raise ValueError("Unable to resolve anchor artifact for fixed-scaffold rerun plan.")
    anchor_path = REPO_ROOT / str(anchor_ref)
    anchor_payload = _load_json(anchor_path)

    candidate_terms = _flatten_runtime_terms(candidate_payload)
    anchor_terms = _flatten_runtime_terms(anchor_payload)
    candidate_term_keys = {(rec["logical_index"], rec["candidate_label"], rec["pauli_exyz"]) for rec in candidate_terms}
    omitted_terms = [
        rec for rec in anchor_terms
        if (rec["logical_index"], rec["candidate_label"], rec["pauli_exyz"]) not in candidate_term_keys
    ]

    candidate_runtime_count = int(_artifact_parameterization(candidate_payload).get("runtime_parameter_count", 0) or 0)
    anchor_runtime_count = int(_artifact_parameterization(anchor_payload).get("runtime_parameter_count", 0) or 0)
    candidate_logical_count = int(_artifact_parameterization(candidate_payload).get("logical_operator_count", 0) or 0)
    anchor_logical_count = int(_artifact_parameterization(anchor_payload).get("logical_operator_count", 0) or 0)

    candidate_eval_path, candidate_eval = _find_candidate_energy_only_runtime_eval(candidate_ref)
    anchor_eval_path, anchor_eval = _find_candidate_energy_only_runtime_eval(anchor_ref)
    anchor_full_path, anchor_full = _find_anchor_full_circuit_runtime_eval(anchor_ref)
    attr_path, attr_payload = _find_anchor_attribution(anchor_ref)
    savedparam_path, savedparam_payload = _find_anchor_savedparam_eval(anchor_ref)
    baseline_path, baseline_payload = _find_anchor_local_baseline(anchor_ref)
    candidate_compile = _find_compile_row(candidate_ref)
    anchor_compile = _find_compile_row(anchor_ref)

    full_delta = readout_delta = gate_delta = mthree_delta = None
    dominant_residual = None
    alignment_target = None
    if isinstance(attr_payload, Mapping):
        rec = attr_payload.get("fixed_scaffold_noise_attribution", {})
        slices = rec.get("slices", {}) if isinstance(rec, Mapping) else {}
        if isinstance(slices, Mapping):
            full_delta = slices.get("full", {}).get("delta_mean", None) if isinstance(slices.get("full", {}), Mapping) else None
            readout_delta = slices.get("readout_only", {}).get("delta_mean", None) if isinstance(slices.get("readout_only", {}), Mapping) else None
            gate_delta = slices.get("gate_stateprep_only", {}).get("delta_mean", None) if isinstance(slices.get("gate_stateprep_only", {}), Mapping) else None
    if isinstance(savedparam_payload, Mapping):
        mthree_delta = savedparam_payload.get("mthree", {}).get("delta_mean", None) if isinstance(savedparam_payload.get("mthree", {}), Mapping) else None
    if full_delta is not None and readout_delta is not None and gate_delta is not None:
        dominant_residual = "gate_stateprep" if float(gate_delta) > float(readout_delta) else "readout"
    if mthree_delta is not None and readout_delta is not None and gate_delta is not None:
        alignment_target = (
            "gate_stateprep_only"
            if abs(float(mthree_delta) - float(gate_delta)) <= abs(float(mthree_delta) - float(readout_delta))
            else "readout_only"
        )

    candidate_budget = _screen_budget(
        runtime_parameter_count=int(candidate_runtime_count),
        oracle_repeats=int(candidate_eval.get("noise_contract", {}).get("oracle_repeats", 1)) if isinstance(candidate_eval, Mapping) else 1,
    )
    anchor_budget = _screen_budget(runtime_parameter_count=int(anchor_runtime_count), oracle_repeats=1)

    recovery_payload = _load_json(recovery_json) if recovery_json is not None and Path(recovery_json).exists() else None
    recovery_summary = recovery_payload.get("summary", {}) if isinstance(recovery_payload, Mapping) else {}
    recovered_best = recovery_payload.get("best_so_far", None) if isinstance(recovery_payload, Mapping) else None
    recovered_backend = None
    if isinstance(recovery_payload, Mapping):
        for row in recovery_payload.get("objective_trace", []) if isinstance(recovery_payload.get("objective_trace", []), Sequence) else []:
            if not isinstance(row, Mapping):
                continue
            runtime_jobs = row.get("runtime_jobs", [])
            if isinstance(runtime_jobs, Sequence):
                for job in runtime_jobs:
                    if isinstance(job, Mapping) and job.get("backend_name", None) not in {None, ""}:
                        recovered_backend = str(job.get("backend_name"))
                        break
                if recovered_backend is not None:
                    break

    candidate_runtime_contract = "missing"
    candidate_runtime_backend = None
    if isinstance(candidate_eval, Mapping):
        candidate_runtime_contract = "energy_only_runtime"
        candidate_runtime_backend = candidate_eval.get("noise_contract", {}).get("backend_name", None) if isinstance(candidate_eval.get("noise_contract", {}), Mapping) else None

    anchor_runtime_contract = "missing"
    anchor_runtime_backend = None
    if isinstance(anchor_eval, Mapping):
        anchor_runtime_contract = "energy_only_runtime"
        anchor_runtime_backend = anchor_eval.get("noise_contract", {}).get("backend_name", None) if isinstance(anchor_eval.get("noise_contract", {}), Mapping) else None
    elif isinstance(anchor_full, Mapping):
        anchor_runtime_contract = "full_circuit_audit"
        anchor_runtime_backend = anchor_full.get("backend_name", None)

    target_backend = recovered_backend or candidate_runtime_backend or anchor_runtime_backend
    evidence_gaps: list[str] = []
    if anchor_runtime_contract != "energy_only_runtime":
        evidence_gaps.append("anchor_energy_only_runtime_baseline_missing")
    if recovered_backend is not None and candidate_runtime_backend is not None and str(recovered_backend) != str(candidate_runtime_backend):
        evidence_gaps.append("candidate_runtime_backend_mismatch")
    if target_backend is None:
        evidence_gaps.append("target_backend_unknown")
    backend_selection_required = bool(
        target_backend is None
        or "candidate_runtime_backend_mismatch" in evidence_gaps
    )

    if backend_selection_required:
        recommended_next_submission = "select_backend_then_anchor_7term_energy_only_fixedtheta_baseline"
    else:
        recommended_next_submission = (
            "anchor_7term_energy_only_fixedtheta_baseline"
            if anchor_runtime_contract != "energy_only_runtime"
            else "candidate_6term_runtime_spsa_screen"
        )
        recommended_next_submission = f"{recommended_next_submission}_on_{target_backend}"

    result = {
        "generated_utc": _now_utc(),
        "pipeline": "hh_fixed_scaffold_rerun_plan_v1",
        "candidate_artifact_json": str(candidate_ref),
        "anchor_artifact_json": str(anchor_ref),
        "decision_context": {
            "target_backend": target_backend,
            "stateprep_gate_noise_note": "Existing anchor attribution evidence indicates the dominant residual is gate/state-prep rather than readout.",
        },
        "candidate": {
            "fixed_scaffold_kind": _artifact_adapt_payload(candidate_payload).get("fixed_scaffold_kind", candidate_payload.get("fixed_scaffold_kind", None)),
            "local_exact_energy": float(_artifact_adapt_payload(candidate_payload).get("energy", float("nan"))),
            "local_abs_delta_e": float(_artifact_adapt_payload(candidate_payload).get("abs_delta_e", float("nan"))),
            "local_exact_state_fidelity": candidate_payload.get("analysis", {}).get("expected_local_exact_state_fidelity", candidate_payload.get("expected_local_exact_state_fidelity", None)) if isinstance(candidate_payload.get("analysis", {}), Mapping) else candidate_payload.get("expected_local_exact_state_fidelity", None),
            "runtime_parameter_count": int(candidate_runtime_count),
            "logical_parameter_count": int(candidate_logical_count),
            "compile_row": candidate_compile,
            "fixedtheta_runtime_eval_json": _repo_relative_str(candidate_eval_path),
            "fixedtheta_runtime_backend": candidate_runtime_backend,
            "fixedtheta_runtime_delta_mean": (None if not isinstance(candidate_eval, Mapping) else candidate_eval.get("runtime_eval", {}).get("delta_mean", None)),
        },
        "anchor": {
            "fixed_scaffold_kind": _artifact_adapt_payload(anchor_payload).get("fixed_scaffold_kind", anchor_payload.get("fixed_scaffold_kind", None)),
            "local_exact_energy": float(_artifact_adapt_payload(anchor_payload).get("energy", float("nan"))),
            "local_abs_delta_e": float(_artifact_adapt_payload(anchor_payload).get("abs_delta_e", float("nan"))),
            "local_exact_state_fidelity": (baseline_payload.get("exact_state_fidelity", None) if isinstance(baseline_payload, Mapping) else None),
            "runtime_parameter_count": int(anchor_runtime_count),
            "logical_parameter_count": int(anchor_logical_count),
            "compile_row": anchor_compile,
            "energy_only_runtime_eval_json": _repo_relative_str(anchor_eval_path),
            "fallback_runtime_eval_json": _repo_relative_str(anchor_full_path),
            "runtime_contract": str(anchor_runtime_contract),
            "runtime_backend": anchor_runtime_backend,
        },
        "structural_diff": {
            "anchor_runtime_parameter_count": int(anchor_runtime_count),
            "candidate_runtime_parameter_count": int(candidate_runtime_count),
            "runtime_parameter_delta": int(candidate_runtime_count - anchor_runtime_count),
            "anchor_logical_parameter_count": int(anchor_logical_count),
            "candidate_logical_parameter_count": int(candidate_logical_count),
            "omitted_runtime_terms_exyz": [str(rec.get("pauli_exyz", "")) for rec in omitted_terms],
            "affected_logical_blocks": sorted({str(rec.get("candidate_label", "")) for rec in omitted_terms}),
        },
        "noise_evidence": {
            "attribution_json": _repo_relative_str(attr_path),
            "savedparam_eval_json": _repo_relative_str(savedparam_path),
            "full_delta_mean": full_delta,
            "readout_only_delta_mean": readout_delta,
            "gate_stateprep_only_delta_mean": gate_delta,
            "dominant_residual_noise_source": dominant_residual,
            "mthree_savedparam_delta_mean": mthree_delta,
            "mthree_alignment_target": alignment_target,
            "readout_not_primary_limit": bool(dominant_residual == "gate_stateprep" and alignment_target == "gate_stateprep_only"),
        },
        "runtime_contracts": {
            "candidate_runtime_contract": str(candidate_runtime_contract),
            "candidate_runtime_backend": candidate_runtime_backend,
            "anchor_runtime_contract": str(anchor_runtime_contract),
            "anchor_runtime_backend": anchor_runtime_backend,
            "runtime_contract_match": bool(candidate_runtime_contract == anchor_runtime_contract == "energy_only_runtime" and candidate_runtime_backend == anchor_runtime_backend),
            "requires_matched_anchor_energy_only_baseline": bool(anchor_runtime_contract != "energy_only_runtime"),
        },
        "budget_plan": {
            "budget_unit": "runtime_estimator_jobs",
            "candidate_screen": candidate_budget,
            "anchor_screen": anchor_budget,
            "completed_candidate_trace_rows": int(recovery_summary.get("trace_rows_completed", 0)) if isinstance(recovery_summary, Mapping) else 0,
            "completed_candidate_objective_calls": (
                None
                if not isinstance(recovery_summary, Mapping) or recovery_summary.get("objective_calls_completed", None) is None
                else int(recovery_summary.get("objective_calls_completed", 0))
            ),
            "completed_candidate_runtime_jobs": int(recovery_summary.get("runtime_jobs_completed", 0)) if isinstance(recovery_summary, Mapping) else 0,
            "candidate_screen_runtime_jobs_remaining": (
                None
                if not isinstance(recovery_summary, Mapping)
                else max(0, int(candidate_budget["screen_total_runtime_jobs"]) - int(recovery_summary.get("runtime_jobs_completed", 0)))
            ),
            "candidate_resume_start_theta_source": (
                "recovered_best_so_far_runtime_theta"
                if (
                    isinstance(recovered_best, Mapping)
                    and recovered_best.get("theta_runtime", None) is not None
                    and list(recovered_best.get("theta_runtime", [])) != []
                )
                else "saved_runtime_theta"
            ),
            "backend_selection_required": backend_selection_required,
            "recommended_next_submission": recommended_next_submission,
            "recommended_next_optimizer_submission": (
                "select_backend_then_candidate_6term_runtime_spsa_screen"
                if backend_selection_required
                else f"candidate_6term_runtime_spsa_screen_on_{target_backend}"
            ),
        },
        "recovery_context": {
            "recovery_json": _repo_relative_str(recovery_json),
            "recovered_best_so_far": recovered_best,
        },
        "evidence_gaps": evidence_gaps,
        "summary": {
            "pareto_takeaway": "6-term is the lighter executable candidate, but the current backend evidence is not matched against a 7-term energy-only baseline on the same backend.",
            "noise_takeaway": "Readout mitigation is helpful, but the surviving residual is better explained by gate/state-prep error than by readout alone.",
        },
    }
    if output_json is not None:
        _write_json(output_json, result)
    return result
