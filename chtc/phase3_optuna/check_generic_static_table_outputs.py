#!/usr/bin/env python3
"""Validate fetched CHTC outputs for the generic static Table-I benchmark queue."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence
import math

from pipelines.exact_bench.benchmark_decision_noise import (
    BENCHMARK_DECISION_NOISE_MODEL_CHOICES,
    BENCHMARK_DECISION_NOISE_SEMANTIC as _BENCHMARK_DECISION_NOISE_SEMANTIC,
)

DEFAULT_RECORDS = Path("chtc/phase3_optuna/input/generic_static_table_records.tsv")
DEFAULT_FETCHED_ROOT = Path("raw_outputs/chtc_phase3_optuna/generic_static_table")
DEFAULT_LOCAL_ROOT = Path("raw_outputs/generic_static_table")
ENRICHMENT_FILENAME = "generic_static_metric_enrichment.json"
ENRICHMENT_SCHEMA_VERSION = "generic_static_metric_enrichment_v1"


_QUALITY_PASS_STATUSES = {"completed", "ok"}
_QUALITY_PASS_ROW_STATUSES = {"ok", "completed"}
_BENCHMARK_METRIC_KEYS = ("energy", "abs_delta_e", "count_2q", "circuit_depth")
_REQUIRED_PROXY_KEYS = ("shots_total", "compiled_depth_total", "compiled_count_2q_total")
_PHASE3_STATIC_ADAPT_ALGORITHM_IDS = {"static_family_native_adapt_phase3", "static_append_only_adapt_phase3"}
_PHASE3_VALUE_NOISE_SEMANTIC = "post_expectation_value_noise_not_physical_shots"
_BENCHMARK_VALUE_NOISE_SEMANTIC = "post_static_result_value_noise_not_physical_shots"
_BENCHMARK_VALUE_NOISE_MODEL_CHOICES = {"off", "gaussian_iid_v1"}
_BENCHMARK_DECISION_NOISE_MODEL_CHOICES = set(BENCHMARK_DECISION_NOISE_MODEL_CHOICES)
_BENCHMARK_VALUE_NOISE_EXACT_ENERGY_KEYS = (
    "exact_energy",
    "exact_gs_energy",
    "exact_reference_energy",
    "same_cutoff_exact_gs_energy",
    "target_exact_energy",
    "exact_energy_total",
)
_BENCHMARK_VALUE_NOISE_JSON_ARTIFACTS = (
    "generic_static_single.json",
    "result.json",
    "manifest.json",
    "rows.json",
    "hh_static_benchmark_result.json",
    "hh_static_benchmark_manifest.json",
    "hh_static_benchmark_rows.json",
)
_ALGORITHMIC_WORK_SCHEMA = "algorithmic_measurement_work_v1"
_ALGORITHMIC_WORK_COMPONENTS = ("N_H_outer_eval", "N_grad_probe", "N_metric_probe", "N_H_refit_eval")
_ALGORITHMIC_WORK_SOURCE_KINDS = {"explicit_components", "event_ledger"}


def _load_records(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing records file: {path}")
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines(), delimiter="\t"))
    required = {"record_id", "family", "case_id", "algorithm_id"}
    missing = required - set(rows[0].keys() if rows else ())
    if missing:
        raise ValueError(f"records file {path} missing columns: {sorted(missing)}")
    return rows


def _default_root() -> Path:
    if DEFAULT_FETCHED_ROOT.exists():
        return DEFAULT_FETCHED_ROOT
    return DEFAULT_LOCAL_ROOT


def _read_json(path: Path) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _payload_for_record(root: Path, record_id: str) -> tuple[Path, Mapping[str, Any] | None]:
    result_dir = root / record_id / "result"
    for name in (
        "generic_static_single.json",
        "hh_static_benchmark_result.json",
        "result.json",
        "manifest.json",
        "skip.json",
    ):
        path = result_dir / name
        payload = _read_json(path)
        if payload is not None:
            return path, payload
    return result_dir / "generic_static_single.json", None


def _payload_status(payload: Mapping[str, Any]) -> str:
    status = str(payload.get("status", ""))
    if status:
        return status
    result = payload.get("result")
    if isinstance(result, Mapping):
        return str(result.get("status", ""))
    rows = payload.get("rows")
    if isinstance(rows, list) and rows and isinstance(rows[0], Mapping):
        return str(rows[0].get("status", ""))
    return "unknown"


def _row_metric(payload: Mapping[str, Any], key: str) -> Any:
    result = payload.get("result")
    if isinstance(result, Mapping) and key in result:
        return result.get(key)
    rows = payload.get("rows")
    if isinstance(rows, list) and rows and isinstance(rows[0], Mapping):
        return rows[0].get(key)
    return payload.get(key)


def _is_quality_pass(payload: Mapping[str, Any]) -> bool:
    status = _payload_status(payload)
    if status in _QUALITY_PASS_STATUSES:
        return True
    if status in _QUALITY_PASS_ROW_STATUSES:
        return True
    result = payload.get("result")
    if isinstance(result, Mapping) and str(result.get("status", "")) in _QUALITY_PASS_ROW_STATUSES:
        return True
    return False


def _has_benchmark_metrics(payload: Mapping[str, Any]) -> bool:
    return any(_row_metric(payload, key) is not None for key in _BENCHMARK_METRIC_KEYS)


def _contract_violations(payload: Mapping[str, Any]) -> list[str]:
    violations: list[str] = []
    algorithm_id = str(_row_metric(payload, "algorithm_id") or payload.get("algorithm_id") or "")
    expected_phase3_called = bool(algorithm_id in _PHASE3_STATIC_ADAPT_ALGORITHM_IDS)
    if _row_metric(payload, "phase3_controller_called") is not expected_phase3_called:
        violations.append(f"phase3_controller_called_not_{str(expected_phase3_called).lower()}")
    for key in _REQUIRED_PROXY_KEYS:
        if _row_metric(payload, key) is None:
            violations.append(f"missing_{key}")
    return violations


def _enrichment_payload(enrichment_root: Path | None, record_id: str) -> tuple[Path | None, Mapping[str, Any] | None]:
    if enrichment_root is None:
        return None, None
    path = enrichment_root / record_id / "result" / ENRICHMENT_FILENAME
    return path, _read_json(path)


def _finite_num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _int_or_none(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _record_phase3_value_noise_requested(record: Mapping[str, Any]) -> bool:
    model = str(record.get("phase3_oracle_value_noise_model") or "off").strip().lower()
    if model != "off":
        return True
    raw_std = record.get("phase3_oracle_value_noise_std")
    std = _finite_num(raw_std)
    if raw_std not in {None, ""} and (std is None or std != 0.0):
        return True
    return record.get("phase3_oracle_value_noise_seed") not in {None, ""}


def _record_benchmark_value_noise_requested(record: Mapping[str, Any]) -> bool:
    model = str(record.get("benchmark_value_noise_model") or "off").strip().lower()
    if model != "off":
        return True
    raw_std = record.get("benchmark_value_noise_std")
    std = _finite_num(raw_std)
    if raw_std not in {None, ""} and (std is None or std != 0.0):
        return True
    return record.get("benchmark_value_noise_seed") not in {None, ""}


def _record_benchmark_decision_noise_requested(record: Mapping[str, Any]) -> bool:
    model = str(record.get("benchmark_decision_noise_model") or "off").strip().lower()
    if model != "off":
        return True
    raw_std = record.get("benchmark_decision_noise_std")
    std = _finite_num(raw_std)
    if raw_std not in {None, ""} and (std is None or std != 0.0):
        return True
    return record.get("benchmark_decision_noise_seed") not in {None, ""}


def _record_value_noise_requested(record: Mapping[str, Any]) -> bool:
    return _record_phase3_value_noise_requested(record) or _record_benchmark_value_noise_requested(record)


def _path_get(root: Any, path: str) -> Any:
    node = root
    for part in str(path).split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def _adapt_payload_for_record(payload_path: Path, payload: Mapping[str, Any]) -> tuple[Path | None, Mapping[str, Any] | None]:
    result = payload.get("result") if isinstance(payload.get("result"), Mapping) else {}
    result_json = None
    if isinstance(result, Mapping):
        result_json = result.get("result_json")
    if result_json in {None, ""}:
        result_json = payload.get("result_json")
    candidates: list[Path] = []
    if result_json not in {None, ""}:
        raw_path = Path(str(result_json))
        candidates.append(raw_path)
        if not raw_path.is_absolute():
            candidates.append(payload_path.parent / raw_path)
    benchmark_id = None
    if isinstance(result, Mapping):
        benchmark_id = result.get("benchmark_id")
    if benchmark_id in {None, ""}:
        benchmark_id = payload.get("benchmark_id")
    if benchmark_id not in {None, ""}:
        candidates.append(payload_path.parent / str(benchmark_id) / "json" / "result.json")
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        loaded = _read_json(candidate)
        if loaded is not None:
            return candidate, loaded
    return (candidates[0] if candidates else None), None


def _value_noise_payload(adapt_payload: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(adapt_payload, Mapping):
        return None
    for path in (
        "continuation.oracle_gradient_config.value_noise",
        "adapt_vqe.continuation.oracle_gradient_config.value_noise",
        "oracle_gradient_config.value_noise",
    ):
        payload = _path_get(adapt_payload, path)
        if isinstance(payload, Mapping):
            return payload
    return None


def _phase3_value_noise_status(
    *,
    record: Mapping[str, Any],
    payload_path: Path,
    payload: Mapping[str, Any],
    expected_model: str | None = None,
    expected_std: float | None = None,
    expected_seed: int | None = None,
) -> tuple[str, dict[str, Any]]:
    if not _record_phase3_value_noise_requested(record):
        return "not_requested", {}
    if str(record.get("algorithm_id") or "") not in _PHASE3_STATIC_ADAPT_ALGORITHM_IDS:
        return "requested_for_non_phase3_static_adapt", {}
    adapt_path, adapt_payload = _adapt_payload_for_record(payload_path, payload)
    if adapt_payload is None:
        return "missing_adapt_payload", {"expected_adapt_payload": None if adapt_path is None else str(adapt_path)}
    value_noise = _value_noise_payload(adapt_payload)
    if not isinstance(value_noise, Mapping):
        return "missing_value_noise_payload", {"adapt_payload": str(adapt_path)}
    if not bool(value_noise.get("enabled", False)):
        return "value_noise_not_enabled", {"adapt_payload": str(adapt_path), "value_noise": dict(value_noise)}
    actual_model = str(value_noise.get("model") or "").strip().lower()
    wanted_model = str(expected_model or record.get("phase3_oracle_value_noise_model") or "").strip().lower()
    if wanted_model and actual_model != wanted_model:
        return "model_mismatch", {"expected": wanted_model, "actual": actual_model, "adapt_payload": str(adapt_path)}
    actual_std = _finite_num(value_noise.get("std"))
    wanted_std = expected_std
    if wanted_std is None:
        wanted_std = _finite_num(record.get("phase3_oracle_value_noise_std"))
    if wanted_std is not None and (actual_std is None or not math.isclose(float(actual_std), float(wanted_std), rel_tol=1e-9, abs_tol=1e-15)):
        return "std_mismatch", {"expected": wanted_std, "actual": actual_std, "adapt_payload": str(adapt_path)}
    actual_seed = _int_or_none(value_noise.get("seed"))
    wanted_seed = expected_seed
    if wanted_seed is None:
        wanted_seed = _int_or_none(record.get("phase3_oracle_value_noise_seed"))
    if wanted_seed is not None and actual_seed != int(wanted_seed):
        return "seed_mismatch", {"expected": int(wanted_seed), "actual": actual_seed, "adapt_payload": str(adapt_path)}
    if str(value_noise.get("semantic") or "") != _PHASE3_VALUE_NOISE_SEMANTIC:
        return "semantic_mismatch", {"expected": _PHASE3_VALUE_NOISE_SEMANTIC, "actual": value_noise.get("semantic"), "adapt_payload": str(adapt_path)}
    return "ok", {"adapt_payload": str(adapt_path), "value_noise": dict(value_noise)}


def _primary_payload_row(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    rows = _payload_benchmark_rows(payload)
    return rows[0] if rows else None


def _payload_benchmark_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    result = payload.get("result")
    if isinstance(result, Mapping):
        out.append(result)
    rows = payload.get("rows")
    if isinstance(rows, list):
        out.extend(row for row in rows if isinstance(row, Mapping))
    if not out and any(key in payload for key in ("energy", "energy_ideal", "benchmark_value_noise", "benchmark_decision_noise")):
        out.append(payload)
    return out


def _benchmark_reference_energy(row: Mapping[str, Any]) -> tuple[str | None, float | None]:
    for key in _BENCHMARK_VALUE_NOISE_EXACT_ENERGY_KEYS:
        value = _finite_num(row.get(key))
        if value is not None:
            return key, float(value)
    return None, None


def _benchmark_value_noise_energy_baseline(row: Mapping[str, Any], row_noise: Mapping[str, Any]) -> float | None:
    for key in ("benchmark_value_noise_energy_ideal", "energy_pre_benchmark_value_noise"):
        value = _finite_num(row.get(key))
        if value is not None:
            return float(value)
    for key in ("benchmark_value_noise_energy_ideal", "energy_pre_benchmark_value_noise"):
        value = _finite_num(row_noise.get(key))
        if value is not None:
            return float(value)
    return _finite_num(row.get("energy_ideal"))


def _benchmark_value_noise_expected_seed(record: Mapping[str, Any], expected_seed: int | None) -> int | None:
    return expected_seed if expected_seed is not None else _int_or_none(record.get("benchmark_value_noise_seed"))


def _benchmark_value_noise_expected_std(record: Mapping[str, Any], expected_std: float | None) -> float | None:
    return expected_std if expected_std is not None else _finite_num(record.get("benchmark_value_noise_std"))


def _benchmark_value_noise_payload_status(
    *,
    record: Mapping[str, Any],
    payload: Mapping[str, Any],
    expected_model: str | None = None,
    expected_std: float | None = None,
    expected_seed: int | None = None,
) -> tuple[str, dict[str, Any]]:
    if not _record_benchmark_value_noise_requested(record):
        return "not_requested", {}
    rows = _payload_benchmark_rows(payload)
    if not rows:
        return "missing_benchmark_value_noise_row", {}
    payload_noise = payload.get("benchmark_value_noise")
    if not isinstance(payload_noise, Mapping):
        return "missing_benchmark_value_noise_top_level_payload", {"row_count": len(rows)}
    if str(payload_noise.get("status") or "") not in {"ok", "completed"}:
        return "benchmark_top_level_status_not_ok", {"payload_value_noise": dict(payload_noise)}
    applied_row_count = _int_or_none(payload_noise.get("applied_row_count"))
    row_target_count = _int_or_none(payload_noise.get("row_target_count"))
    if row_target_count is None or applied_row_count is None:
        return "benchmark_top_level_missing_counts", {"payload_value_noise": dict(payload_noise), "row_count": len(rows)}
    if row_target_count != len(rows) or applied_row_count != len(rows):
        return "benchmark_top_level_count_mismatch", {
            "payload_value_noise": dict(payload_noise),
            "row_count": len(rows),
            "row_target_count": row_target_count,
            "applied_row_count": applied_row_count,
        }
    if str(payload_noise.get("semantic") or "") != _BENCHMARK_VALUE_NOISE_SEMANTIC:
        return "benchmark_top_level_semantic_mismatch", {"payload_value_noise": dict(payload_noise)}
    if not bool(payload_noise.get("physical_shots_unchanged", False)):
        return "benchmark_top_level_physical_shots_not_marked_unchanged", {"payload_value_noise": dict(payload_noise)}

    wanted_model = str(expected_model or record.get("benchmark_value_noise_model") or "").strip().lower()
    wanted_std = _benchmark_value_noise_expected_std(record, expected_std)
    wanted_seed = _benchmark_value_noise_expected_seed(record, expected_seed)
    for idx, row in enumerate(rows):
        row_noise = row.get("benchmark_value_noise")
        if not isinstance(row_noise, Mapping):
            return "benchmark_row_missing_value_noise_payload", {"row_index": idx}
        if str(row.get("benchmark_value_noise_status") or "") != "ok":
            return "benchmark_row_status_not_ok", {"row_index": idx, "row_value_noise": dict(row_noise)}
        if not bool(row_noise.get("enabled", False)):
            return "benchmark_row_value_noise_not_enabled", {"row_index": idx, "row_value_noise": dict(row_noise)}
        actual_model = str(row_noise.get("model") or "").strip().lower()
        if wanted_model and actual_model != wanted_model:
            return "benchmark_model_mismatch", {"row_index": idx, "expected": wanted_model, "actual": actual_model, "row_value_noise": dict(row_noise)}
        if actual_model not in _BENCHMARK_VALUE_NOISE_MODEL_CHOICES or actual_model == "off":
            return "benchmark_model_invalid", {"row_index": idx, "actual": actual_model, "row_value_noise": dict(row_noise)}
        actual_std = _finite_num(row_noise.get("std"))
        if wanted_std is not None and (actual_std is None or not math.isclose(float(actual_std), float(wanted_std), rel_tol=1e-9, abs_tol=1e-15)):
            return "benchmark_std_mismatch", {"row_index": idx, "expected": wanted_std, "actual": actual_std, "row_value_noise": dict(row_noise)}
        actual_seed = _int_or_none(row_noise.get("seed"))
        if actual_seed is None:
            return "benchmark_seed_missing", {"row_index": idx, "row_value_noise": dict(row_noise)}
        if wanted_seed is not None and actual_seed != int(wanted_seed):
            return "benchmark_seed_mismatch", {"row_index": idx, "expected": int(wanted_seed), "actual": actual_seed, "row_value_noise": dict(row_noise)}
        if str(row_noise.get("semantic") or "") != _BENCHMARK_VALUE_NOISE_SEMANTIC:
            return "benchmark_semantic_mismatch", {"row_index": idx, "expected": _BENCHMARK_VALUE_NOISE_SEMANTIC, "actual": row_noise.get("semantic"), "row_value_noise": dict(row_noise)}
        if not bool(row_noise.get("physical_shots_unchanged", False)):
            return "benchmark_physical_shots_not_marked_unchanged", {"row_index": idx, "row_value_noise": dict(row_noise)}
        noise_draw = _finite_num(row_noise.get("noise_draw"))
        energy = _finite_num(row.get("energy"))
        energy_baseline = _benchmark_value_noise_energy_baseline(row, row_noise)
        if noise_draw is None or energy is None or energy_baseline is None:
            return "benchmark_missing_energy_or_noise_draw", {"row_index": idx, "row_value_noise": dict(row_noise)}
        if not math.isclose(float(energy), float(energy_baseline) + float(noise_draw), rel_tol=1e-9, abs_tol=1e-12):
            return "benchmark_energy_noise_draw_mismatch", {
                "row_index": idx,
                "energy": energy,
                "energy_baseline": energy_baseline,
                "noise_draw": noise_draw,
                "row_value_noise": dict(row_noise),
            }
        reference_key, reference_energy = _benchmark_reference_energy(row)
        if reference_energy is not None:
            wanted_delta = abs(float(energy) - float(reference_energy))
            for key in ("delta_E_abs", "abs_delta_e"):
                actual_delta = _finite_num(row.get(key))
                if actual_delta is None or not math.isclose(float(actual_delta), wanted_delta, rel_tol=1e-9, abs_tol=1e-12):
                    return "benchmark_delta_mismatch", {
                        "row_index": idx,
                        "field": key,
                        "expected": wanted_delta,
                        "actual": actual_delta,
                        "reference_key": reference_key,
                        "row_value_noise": dict(row_noise),
                    }
    return "ok", {"row_count": len(rows), "payload_value_noise": dict(payload_noise)}


def _benchmark_value_noise_artifacts_status(
    *,
    record: Mapping[str, Any],
    root: Path,
    record_id: str,
    expected_model: str | None = None,
    expected_std: float | None = None,
    expected_seed: int | None = None,
) -> tuple[str, dict[str, Any]]:
    result_dir = Path(root) / str(record_id) / "result"
    checked: list[str] = []
    skipped_no_rows: list[str] = []
    for name in _BENCHMARK_VALUE_NOISE_JSON_ARTIFACTS:
        path = result_dir / name
        if not path.exists():
            continue
        artifact_payload = _read_json(path)
        if not isinstance(artifact_payload, Mapping):
            return "benchmark_artifact_invalid_payload", {"artifact": str(path)}
        if not _payload_benchmark_rows(artifact_payload):
            skipped_no_rows.append(str(path))
            continue
        status, detail = _benchmark_value_noise_payload_status(
            record=record,
            payload=artifact_payload,
            expected_model=expected_model,
            expected_std=expected_std,
            expected_seed=expected_seed,
        )
        if status != "ok":
            return f"benchmark_artifact_{status}", {
                "artifact": str(path),
                "artifact_status": status,
                "artifact_detail": detail,
                "checked_artifacts": checked,
                "skipped_no_rows": skipped_no_rows,
            }
        checked.append(str(path))
    return "ok", {"checked_artifacts": checked, "skipped_no_rows": skipped_no_rows}


def _benchmark_decision_noise_expected_seed(record: Mapping[str, Any], expected_seed: int | None) -> int | None:
    return expected_seed if expected_seed is not None else _int_or_none(record.get("benchmark_decision_noise_seed"))


def _benchmark_decision_noise_expected_std(record: Mapping[str, Any], expected_std: float | None) -> float | None:
    return expected_std if expected_std is not None else _finite_num(record.get("benchmark_decision_noise_std"))


def _benchmark_decision_noise_metadata_status(
    metadata: Mapping[str, Any],
    *,
    record: Mapping[str, Any],
    expected_model: str | None = None,
    expected_std: float | None = None,
    expected_seed: int | None = None,
    label: str = "metadata",
) -> tuple[str, dict[str, Any]]:
    actual_status = str(metadata.get("status") or "").strip().lower()
    if actual_status not in {"ok", "unsupported"}:
        return "decision_status_not_handled", {"label": label, "decision_noise": dict(metadata)}
    if not bool(metadata.get("enabled", False)):
        return "decision_noise_not_enabled", {"label": label, "decision_noise": dict(metadata)}
    actual_model = str(metadata.get("model") or "").strip().lower()
    wanted_model = str(expected_model or record.get("benchmark_decision_noise_model") or "").strip().lower()
    if wanted_model and actual_model != wanted_model:
        return "decision_model_mismatch", {"label": label, "expected": wanted_model, "actual": actual_model, "decision_noise": dict(metadata)}
    if actual_model not in _BENCHMARK_DECISION_NOISE_MODEL_CHOICES or actual_model == "off":
        return "decision_model_invalid", {"label": label, "actual": actual_model, "decision_noise": dict(metadata)}
    actual_std = _finite_num(metadata.get("std"))
    wanted_std = _benchmark_decision_noise_expected_std(record, expected_std)
    if wanted_std is not None and (actual_std is None or not math.isclose(float(actual_std), float(wanted_std), rel_tol=1e-9, abs_tol=1e-15)):
        return "decision_std_mismatch", {"label": label, "expected": wanted_std, "actual": actual_std, "decision_noise": dict(metadata)}
    actual_seed = _int_or_none(metadata.get("seed"))
    if actual_seed is None:
        return "decision_seed_missing", {"label": label, "decision_noise": dict(metadata)}
    wanted_seed = _benchmark_decision_noise_expected_seed(record, expected_seed)
    if wanted_seed is not None and actual_seed != int(wanted_seed):
        return "decision_seed_mismatch", {"label": label, "expected": int(wanted_seed), "actual": actual_seed, "decision_noise": dict(metadata)}
    if str(metadata.get("semantic") or "") != _BENCHMARK_DECISION_NOISE_SEMANTIC:
        return "decision_semantic_mismatch", {"label": label, "expected": _BENCHMARK_DECISION_NOISE_SEMANTIC, "actual": metadata.get("semantic"), "decision_noise": dict(metadata)}
    if not bool(metadata.get("physical_shots_unchanged", False)):
        return "decision_physical_shots_not_marked_unchanged", {"label": label, "decision_noise": dict(metadata)}
    if str(metadata.get("algorithmic_measurement_work_schema") or "") != _ALGORITHMIC_WORK_SCHEMA:
        return "decision_algorithmic_work_schema_mismatch", {
            "label": label,
            "expected": _ALGORITHMIC_WORK_SCHEMA,
            "actual": metadata.get("algorithmic_measurement_work_schema"),
            "decision_noise": dict(metadata),
        }
    if not bool(metadata.get("algorithmic_measurement_work_unchanged", False)):
        return "decision_algorithmic_work_not_marked_unchanged", {"label": label, "decision_noise": dict(metadata)}
    if actual_status == "unsupported":
        if bool(metadata.get("supported", True)):
            return "decision_unsupported_marked_supported", {"label": label, "decision_noise": dict(metadata)}
        if bool(metadata.get("applied", True)):
            return "decision_unsupported_marked_applied", {"label": label, "decision_noise": dict(metadata)}
        if not bool(metadata.get("fail_closed", False)):
            return "decision_unsupported_not_fail_closed", {"label": label, "decision_noise": dict(metadata)}
        draw_count = _int_or_none(metadata.get("draw_count_total"))
        if draw_count is None or draw_count != 0:
            return "decision_unsupported_has_draws", {"label": label, "decision_noise": dict(metadata)}
        surfaces = metadata.get("surfaces_affected")
        if surfaces not in (None, ()) and surfaces != []:
            return "decision_unsupported_has_surfaces", {"label": label, "decision_noise": dict(metadata)}
        if not str(metadata.get("reason") or "").strip():
            return "decision_unsupported_missing_reason", {"label": label, "decision_noise": dict(metadata)}
        return "unsupported", {"label": label, "decision_noise": dict(metadata)}
    if not bool(metadata.get("supported", False)):
        return "decision_ok_not_marked_supported", {"label": label, "decision_noise": dict(metadata)}
    if not bool(metadata.get("applied", False)):
        return "decision_ok_not_marked_applied", {"label": label, "decision_noise": dict(metadata)}
    draw_count = _int_or_none(metadata.get("draw_count_total"))
    if draw_count is None or draw_count <= 0:
        return "decision_ok_missing_draws", {"label": label, "decision_noise": dict(metadata)}
    surfaces = metadata.get("surfaces_affected")
    if not isinstance(surfaces, list) or not surfaces:
        return "decision_ok_missing_surfaces", {"label": label, "decision_noise": dict(metadata)}
    return "ok", {"label": label, "decision_noise": dict(metadata)}


def _benchmark_decision_noise_payload_status(
    *,
    record: Mapping[str, Any],
    payload: Mapping[str, Any],
    expected_model: str | None = None,
    expected_std: float | None = None,
    expected_seed: int | None = None,
) -> tuple[str, dict[str, Any]]:
    if not _record_benchmark_decision_noise_requested(record):
        return "not_requested", {}
    if str(record.get("algorithm_id") or "") in _PHASE3_STATIC_ADAPT_ALGORITHM_IDS:
        return "requested_for_phase3_static_adapt_use_phase3_oracle_value_noise", {}
    rows = _payload_benchmark_rows(payload)
    if not rows:
        return "missing_benchmark_decision_noise_row", {}
    payload_noise = payload.get("benchmark_decision_noise")
    if not isinstance(payload_noise, Mapping):
        return "missing_benchmark_decision_noise_top_level_payload", {"row_count": len(rows)}
    top_status, top_detail = _benchmark_decision_noise_metadata_status(
        payload_noise,
        record=record,
        expected_model=expected_model,
        expected_std=expected_std,
        expected_seed=expected_seed,
        label="top_level",
    )
    if top_status not in {"ok", "unsupported"}:
        return top_status, top_detail
    top_field_status = str(payload.get("benchmark_decision_noise_status") or "").strip().lower()
    if top_field_status and top_field_status != top_status:
        return "decision_top_level_status_field_mismatch", {
            "expected": top_status,
            "actual": top_field_status,
            "payload_decision_noise": dict(payload_noise),
        }
    row_statuses: list[str] = []
    row_details: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        row_noise = row.get("benchmark_decision_noise")
        if not isinstance(row_noise, Mapping):
            return "benchmark_row_missing_decision_noise_payload", {"row_index": idx}
        row_status, row_detail = _benchmark_decision_noise_metadata_status(
            row_noise,
            record=record,
            expected_model=expected_model,
            expected_std=expected_std,
            expected_seed=expected_seed,
            label=f"row[{idx}]",
        )
        if row_status not in {"ok", "unsupported"}:
            return row_status, {"row_index": idx, **row_detail}
        row_field_status = str(row.get("benchmark_decision_noise_status") or "").strip().lower()
        if row_field_status and row_field_status != row_status:
            return "decision_row_status_field_mismatch", {
                "row_index": idx,
                "expected": row_status,
                "actual": row_field_status,
                "row_decision_noise": dict(row_noise),
            }
        row_statuses.append(row_status)
        row_details.append({"row_index": idx, **row_detail})
    if top_status == "unsupported" and all(status == "unsupported" for status in row_statuses):
        return "unsupported", {"row_count": len(rows), "payload_decision_noise": dict(payload_noise), "rows": row_details}
    if top_status == "ok" and all(status == "ok" for status in row_statuses):
        return "ok", {"row_count": len(rows), "payload_decision_noise": dict(payload_noise), "rows": row_details}
    return "decision_top_row_status_mismatch", {
        "top_status": top_status,
        "row_statuses": row_statuses,
        "payload_decision_noise": dict(payload_noise),
    }


def _benchmark_decision_noise_artifacts_status(
    *,
    record: Mapping[str, Any],
    root: Path,
    record_id: str,
    expected_model: str | None = None,
    expected_std: float | None = None,
    expected_seed: int | None = None,
) -> tuple[str, dict[str, Any]]:
    result_dir = Path(root) / str(record_id) / "result"
    checked: list[str] = []
    skipped_no_rows: list[str] = []
    accepted_statuses: list[str] = []
    for name in _BENCHMARK_VALUE_NOISE_JSON_ARTIFACTS:
        path = result_dir / name
        if not path.exists():
            continue
        artifact_payload = _read_json(path)
        if not isinstance(artifact_payload, Mapping):
            return "decision_artifact_invalid_payload", {"artifact": str(path)}
        if not _payload_benchmark_rows(artifact_payload):
            skipped_no_rows.append(str(path))
            continue
        status, detail = _benchmark_decision_noise_payload_status(
            record=record,
            payload=artifact_payload,
            expected_model=expected_model,
            expected_std=expected_std,
            expected_seed=expected_seed,
        )
        if status not in {"ok", "unsupported"}:
            return f"decision_artifact_{status}", {
                "artifact": str(path),
                "artifact_status": status,
                "artifact_detail": detail,
                "checked_artifacts": checked,
                "skipped_no_rows": skipped_no_rows,
            }
        checked.append(str(path))
        accepted_statuses.append(status)
    return "ok", {"checked_artifacts": checked, "skipped_no_rows": skipped_no_rows, "accepted_statuses": accepted_statuses}


def _is_explicit_decision_noise_unsupported_payload(payload: Mapping[str, Any]) -> bool:
    if str(payload.get("status") or "").strip().lower() == "skipped_unsupported_decision_noise":
        return True
    if str(payload.get("benchmark_decision_noise_status") or "").strip().lower() == "unsupported":
        return True
    row = _primary_payload_row(payload)
    if isinstance(row, Mapping) and str(row.get("benchmark_decision_noise_status") or "").strip().lower() == "unsupported":
        return True
    return False


def _value_noise_status(
    *,
    record: Mapping[str, Any],
    payload_path: Path,
    payload: Mapping[str, Any],
    expected_model: str | None = None,
    expected_std: float | None = None,
    expected_seed: int | None = None,
) -> tuple[str, dict[str, Any]]:
    phase3_requested = _record_phase3_value_noise_requested(record)
    benchmark_requested = _record_benchmark_value_noise_requested(record)
    if not phase3_requested and not benchmark_requested:
        return "not_requested", {}
    detail: dict[str, Any] = {}
    statuses: dict[str, str] = {}
    if phase3_requested:
        phase3_status, phase3_detail = _phase3_value_noise_status(
            record=record,
            payload_path=payload_path,
            payload=payload,
            expected_model=expected_model,
            expected_std=expected_std,
            expected_seed=expected_seed,
        )
        statuses["phase3_oracle"] = phase3_status
        detail["phase3_oracle"] = phase3_detail
    if benchmark_requested:
        benchmark_status, benchmark_detail = _benchmark_value_noise_payload_status(
            record=record,
            payload=payload,
            expected_model=expected_model,
            expected_std=expected_std,
            expected_seed=expected_seed,
        )
        statuses["benchmark"] = benchmark_status
        detail["benchmark"] = benchmark_detail
    if statuses and all(status == "ok" for status in statuses.values()):
        return "ok", detail
    if len(statuses) == 1:
        return next(iter(statuses.values())), detail
    for label, status in statuses.items():
        if status != "ok":
            return f"{label}_{status}", detail
    return "not_requested", detail


def _metric_status(enrichment: Mapping[str, Any] | None, record_id: str, *, metric_key: str, row_update_key: str) -> str:
    if not isinstance(enrichment, Mapping):
        return "missing"
    if enrichment.get("schema") != ENRICHMENT_SCHEMA_VERSION:
        return "invalid_schema"
    if str(enrichment.get("record_id") or "") != str(record_id):
        return "record_id_mismatch"
    statuses = enrichment.get("metric_statuses")
    row_updates = enrichment.get("row_updates")
    if not isinstance(statuses, Mapping) or not isinstance(row_updates, Mapping):
        return f"missing_{metric_key.lower()}_status"
    status = str(statuses.get(metric_key) or f"missing_{metric_key.lower()}_status")
    if status != "ok":
        return status
    if _finite_num(row_updates.get(row_update_key)) is None:
        return f"missing_{row_update_key}_value"
    return "ok"


def _s_norm_status(enrichment: Mapping[str, Any] | None, record_id: str) -> str:
    return _metric_status(enrichment, record_id, metric_key="S_norm", row_update_key="S_norm")


def _s_alg_status(enrichment: Mapping[str, Any] | None, record_id: str) -> str:
    status = _metric_status(enrichment, record_id, metric_key="S_alg", row_update_key="S_alg")
    if status != "ok":
        return status
    row_updates = enrichment.get("row_updates") if isinstance(enrichment, Mapping) else None
    if not isinstance(row_updates, Mapping):
        return "missing_s_alg_row_updates"
    s_alg = _finite_num(row_updates.get("S_alg"))
    component_values = []
    for component in (*_ALGORITHMIC_WORK_COMPONENTS, "N_other_quantum"):
        value = _finite_num(row_updates.get(f"S_alg_{component}"))
        if value is None:
            return f"missing_S_alg_{component}_value"
        component_values.append(float(value))
    if s_alg is None or not math.isclose(float(s_alg), float(sum(component_values)), rel_tol=1e-9, abs_tol=1e-9):
        return "S_alg_component_sum_mismatch"
    work = _path_get(enrichment, "metrics.algorithmic_measurement_work")
    if not isinstance(work, Mapping):
        return "missing_algorithmic_measurement_work"
    if str(work.get("schema") or "") != _ALGORITHMIC_WORK_SCHEMA:
        return "invalid_algorithmic_measurement_work_schema"
    if str(work.get("status") or "") != "ok":
        return str(work.get("status") or "invalid_algorithmic_measurement_work_status")
    if str(work.get("source_kind") or "") not in _ALGORITHMIC_WORK_SOURCE_KINDS:
        return "unsupported_algorithmic_measurement_work_source_kind"
    components = work.get("components")
    if not isinstance(components, Mapping):
        return "missing_algorithmic_measurement_work_components"
    for component in _ALGORITHMIC_WORK_COMPONENTS:
        if _finite_num(components.get(component)) is None:
            return f"missing_algorithmic_measurement_work_{component}"
    return "ok"


def validate_outputs(
    *,
    records_path: Path,
    root: Path,
    summary_path: Path | None = None,
    enrichment_root: Path | None = None,
    require_enrichment: bool = False,
    require_s_norm: bool = False,
    require_s_alg: bool = False,
    require_value_noise_applied: bool = False,
    expected_value_noise_model: str | None = None,
    expected_value_noise_std: float | None = None,
    expected_value_noise_seed: int | None = None,
    require_decision_noise_handled: bool = False,
    expected_decision_noise_model: str | None = None,
    expected_decision_noise_std: float | None = None,
    expected_decision_noise_seed: int | None = None,
) -> dict[str, Any]:
    records = _load_records(records_path)
    status_counts: Counter[str] = Counter()
    algorithm_counts: dict[str, Counter[str]] = defaultdict(Counter)
    missing: list[dict[str, str]] = []
    unusable: list[dict[str, Any]] = []
    benchmarked: list[dict[str, Any]] = []
    quality_nonpassing: list[dict[str, Any]] = []
    contract_violations: list[dict[str, Any]] = []
    enrichment_counts: Counter[str] = Counter()
    enrichment_violations: list[dict[str, Any]] = []
    s_norm_counts: Counter[str] = Counter()
    s_norm_by_algorithm: dict[str, Counter[str]] = defaultdict(Counter)
    s_alg_counts: Counter[str] = Counter()
    s_alg_by_algorithm: dict[str, Counter[str]] = defaultdict(Counter)
    value_noise_counts: Counter[str] = Counter()
    value_noise_by_algorithm: dict[str, Counter[str]] = defaultdict(Counter)
    value_noise_violations: list[dict[str, Any]] = []
    decision_noise_counts: Counter[str] = Counter()
    decision_noise_by_algorithm: dict[str, Counter[str]] = defaultdict(Counter)
    decision_noise_violations: list[dict[str, Any]] = []

    if bool(require_s_norm) and enrichment_root is None:
        for record in records:
            s_norm_counts["missing"] += 1
            s_norm_by_algorithm[str(record["algorithm_id"])]["missing"] += 1
            enrichment_violations.append({**record, "s_norm_status": "missing", "violation": "S_norm_required_without_enrichment_root"})
    if bool(require_s_alg) and enrichment_root is None:
        for record in records:
            s_alg_counts["missing"] += 1
            s_alg_by_algorithm[str(record["algorithm_id"])]["missing"] += 1
            enrichment_violations.append({**record, "s_alg_status": "missing", "violation": "S_alg_required_without_enrichment_root"})

    for record in records:
        record_id = str(record["record_id"])
        payload_path, payload = _payload_for_record(root, record_id)
        if payload is None:
            status_counts["missing"] += 1
            algorithm_counts[str(record["algorithm_id"])]["missing"] += 1
            missing.append({**record, "expected_payload": str(payload_path)})
            continue
        enrichment_path, enrichment = _enrichment_payload(enrichment_root, record_id)
        if enrichment_root is not None:
            if enrichment is None:
                enrichment_counts["missing"] += 1
                if bool(require_enrichment):
                    enrichment_violations.append({**record, "expected_enrichment": str(enrichment_path)})
            else:
                enrichment_status = str(enrichment.get("status") or "unknown")
                enrichment_counts[enrichment_status] += 1
                if bool(require_enrichment) and enrichment_status in {"failed", "payload_missing"}:
                    enrichment_violations.append({**record, "enrichment_path": str(enrichment_path), "enrichment_status": enrichment_status})
            s_status = _s_norm_status(enrichment, record_id)
            s_norm_counts[s_status] += 1
            s_norm_by_algorithm[str(record["algorithm_id"])][s_status] += 1
            if bool(require_s_norm) and s_status != "ok":
                enrichment_violations.append({
                    **record,
                    "enrichment_path": str(enrichment_path),
                    "s_norm_status": s_status,
                    "violation": "S_norm_required",
                })
            s_alg_status = _s_alg_status(enrichment, record_id)
            s_alg_counts[s_alg_status] += 1
            s_alg_by_algorithm[str(record["algorithm_id"])][s_alg_status] += 1
            if bool(require_s_alg) and s_alg_status != "ok":
                enrichment_violations.append({
                    **record,
                    "enrichment_path": str(enrichment_path),
                    "s_alg_status": s_alg_status,
                    "violation": "S_alg_required",
                })
        if bool(require_value_noise_applied) or _record_value_noise_requested(record):
            value_noise_status, value_noise_detail = _value_noise_status(
                record=record,
                payload_path=payload_path,
                payload=payload,
                expected_model=expected_value_noise_model,
                expected_std=expected_value_noise_std,
                expected_seed=expected_value_noise_seed,
            )
            if _record_benchmark_value_noise_requested(record):
                artifact_status, artifact_detail = _benchmark_value_noise_artifacts_status(
                    record=record,
                    root=root,
                    record_id=record_id,
                    expected_model=expected_value_noise_model,
                    expected_std=expected_value_noise_std,
                    expected_seed=expected_value_noise_seed,
                )
                value_noise_detail = {**value_noise_detail, "benchmark_artifacts": artifact_detail}
                if value_noise_status == "ok" and artifact_status != "ok":
                    value_noise_status = artifact_status
            value_noise_counts[value_noise_status] += 1
            value_noise_by_algorithm[str(record["algorithm_id"])][value_noise_status] += 1
            if bool(require_value_noise_applied) and value_noise_status not in {"ok", "not_requested"}:
                value_noise_violations.append({
                    **record,
                    "payload_path": str(payload_path),
                    "value_noise_status": value_noise_status,
                    "value_noise_detail": value_noise_detail,
                    "violation": "value_noise_required",
                })
        if bool(require_decision_noise_handled) or _record_benchmark_decision_noise_requested(record):
            decision_noise_status, decision_noise_detail = _benchmark_decision_noise_payload_status(
                record=record,
                payload=payload,
                expected_model=expected_decision_noise_model,
                expected_std=expected_decision_noise_std,
                expected_seed=expected_decision_noise_seed,
            )
            if _record_benchmark_decision_noise_requested(record):
                artifact_status, artifact_detail = _benchmark_decision_noise_artifacts_status(
                    record=record,
                    root=root,
                    record_id=record_id,
                    expected_model=expected_decision_noise_model,
                    expected_std=expected_decision_noise_std,
                    expected_seed=expected_decision_noise_seed,
                )
                decision_noise_detail = {**decision_noise_detail, "benchmark_decision_artifacts": artifact_detail}
                if decision_noise_status in {"ok", "unsupported"} and artifact_status != "ok":
                    decision_noise_status = artifact_status
            decision_noise_counts[decision_noise_status] += 1
            decision_noise_by_algorithm[str(record["algorithm_id"])][decision_noise_status] += 1
            if bool(require_decision_noise_handled) and decision_noise_status not in {"ok", "unsupported", "not_requested"}:
                decision_noise_violations.append({
                    **record,
                    "payload_path": str(payload_path),
                    "decision_noise_status": decision_noise_status,
                    "decision_noise_detail": decision_noise_detail,
                    "violation": "decision_noise_required",
                })
        status = _payload_status(payload)
        quality_pass = _is_quality_pass(payload)
        has_metrics = _has_benchmark_metrics(payload)
        contract_errors = _contract_violations(payload)
        decision_noise_unsupported = _is_explicit_decision_noise_unsupported_payload(payload)
        status_counts[status or "unknown"] += 1
        outcome_bucket = (
            "contract_violation"
            if contract_errors
            else "decision_noise_unsupported"
            if decision_noise_unsupported
            else "benchmarked_quality_pass"
            if quality_pass
            else "benchmarked_quality_nonpass"
            if has_metrics
            else "unusable"
        )
        algorithm_counts[str(record["algorithm_id"])][outcome_bucket] += 1
        entry = {
            **record,
            "payload_path": str(payload_path),
            "payload_status": status,
            "benchmark_outcome": outcome_bucket,
            "quality_gate_reason": _row_metric(payload, "quality_gate_reason"),
            "failure_reason": _row_metric(payload, "failure_reason"),
            "energy": _row_metric(payload, "energy"),
            "exact_energy": _row_metric(payload, "exact_energy"),
            "abs_delta_e": _row_metric(payload, "abs_delta_e"),
            "count_2q": _row_metric(payload, "count_2q"),
            "circuit_depth": _row_metric(payload, "circuit_depth"),
            "runtime_s": _row_metric(payload, "runtime_s"),
            "phase3_controller_called": _row_metric(payload, "phase3_controller_called"),
            "shots_total": _row_metric(payload, "shots_total"),
            "compiled_depth_total": _row_metric(payload, "compiled_depth_total"),
            "compiled_count_2q_total": _row_metric(payload, "compiled_count_2q_total"),
            "contract_violations": contract_errors,
        }
        if contract_errors:
            contract_violations.append(entry)
        if has_metrics:
            benchmarked.append(entry)
            if not quality_pass:
                quality_nonpassing.append(entry)
        elif not decision_noise_unsupported:
            unusable.append(entry)

    if bool(require_value_noise_applied) and value_noise_counts.get("ok", 0) == 0:
        value_noise_violations.append({
            "violation": "value_noise_required_no_ok_rows",
            "value_noise_status_counts": dict(sorted(value_noise_counts.items())),
        })
    if bool(require_decision_noise_handled) and (decision_noise_counts.get("ok", 0) + decision_noise_counts.get("unsupported", 0)) == 0:
        decision_noise_violations.append({
            "violation": "decision_noise_required_no_handled_rows",
            "decision_noise_status_counts": dict(sorted(decision_noise_counts.items())),
        })

    summary = {
        "schema": "generic_static_table_output_check_v2",
        "records_path": str(records_path),
        "output_root": str(root),
        "expected_count": len(records),
        "benchmarked_count": len(benchmarked),
        # Backwards-compatible alias: completed means usable benchmark payload present,
        # not that the algorithm met any quality or accuracy threshold.
        "completed_count": len(benchmarked),
        "missing_count": len(missing),
        "unusable_count": len(unusable),
        # Backwards-compatible alias: bad now means unusable payload, not poor result.
        "bad_count": len(unusable),
        "quality_nonpassing_count": len(quality_nonpassing),
        "contract_violation_count": len(contract_violations),
        "enrichment_root": None if enrichment_root is None else str(enrichment_root),
        "enrichment_status_counts": dict(sorted(enrichment_counts.items())),
        "s_norm_status_counts": dict(sorted(s_norm_counts.items())),
        "s_norm_status_by_algorithm": {alg: dict(counter) for alg, counter in sorted(s_norm_by_algorithm.items())},
        "s_alg_status_counts": dict(sorted(s_alg_counts.items())),
        "s_alg_status_by_algorithm": {alg: dict(counter) for alg, counter in sorted(s_alg_by_algorithm.items())},
        "value_noise_status_counts": dict(sorted(value_noise_counts.items())),
        "value_noise_status_by_algorithm": {alg: dict(counter) for alg, counter in sorted(value_noise_by_algorithm.items())},
        "value_noise_violation_count": len(value_noise_violations),
        "decision_noise_status_counts": dict(sorted(decision_noise_counts.items())),
        "decision_noise_status_by_algorithm": {alg: dict(counter) for alg, counter in sorted(decision_noise_by_algorithm.items())},
        "decision_noise_violation_count": len(decision_noise_violations),
        "enrichment_violation_count": len(enrichment_violations),
        "payload_status_counts": dict(sorted(status_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "status_by_algorithm": {alg: dict(counter) for alg, counter in sorted(algorithm_counts.items())},
        "missing": missing,
        "unusable": unusable,
        "bad": unusable,
        "quality_nonpassing": quality_nonpassing,
        "contract_violations": contract_violations,
        "enrichment_violations": enrichment_violations,
        "value_noise_violations": value_noise_violations,
        "decision_noise_violations": decision_noise_violations,
        "benchmarked_preview": benchmarked[:10],
        "completed_preview": benchmarked[:10],
    }
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check generic static Table-I CHTC outputs.")
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=Path("raw_outputs/generic_static_table_output_check.json"))
    parser.add_argument("--enrichment-root", type=Path, default=None)
    parser.add_argument("--require-enrichment", action="store_true", default=False)
    parser.add_argument("--require-s-norm", action="store_true", default=False)
    parser.add_argument("--require-s-alg", action="store_true", default=False)
    parser.add_argument("--require-value-noise-applied", action="store_true", default=False)
    parser.add_argument("--expected-value-noise-model", default=None)
    parser.add_argument("--expected-value-noise-std", type=float, default=None)
    parser.add_argument("--expected-value-noise-seed", type=int, default=None)
    parser.add_argument("--require-decision-noise-handled", action="store_true", default=False)
    parser.add_argument("--expected-decision-noise-model", default=None)
    parser.add_argument("--expected-decision-noise-std", type=float, default=None)
    parser.add_argument("--expected-decision-noise-seed", type=int, default=None)
    parser.add_argument("--allow-incomplete", action="store_true", default=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root) if args.root is not None else _default_root()
    summary = validate_outputs(
        records_path=Path(args.records),
        root=root,
        summary_path=Path(args.summary),
        enrichment_root=args.enrichment_root,
        require_enrichment=bool(args.require_enrichment),
        require_s_norm=bool(args.require_s_norm),
        require_s_alg=bool(args.require_s_alg),
        require_value_noise_applied=bool(args.require_value_noise_applied),
        expected_value_noise_model=args.expected_value_noise_model,
        expected_value_noise_std=args.expected_value_noise_std,
        expected_value_noise_seed=args.expected_value_noise_seed,
        require_decision_noise_handled=bool(args.require_decision_noise_handled),
        expected_decision_noise_model=args.expected_decision_noise_model,
        expected_decision_noise_std=args.expected_decision_noise_std,
        expected_decision_noise_seed=args.expected_decision_noise_seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    hard_fail = bool(
        summary["bad_count"]
        or summary.get("enrichment_violation_count", 0)
        or summary.get("value_noise_violation_count", 0)
        or summary.get("decision_noise_violation_count", 0)
    )
    if not bool(args.allow_incomplete):
        hard_fail = hard_fail or bool(summary["missing_count"] or summary["contract_violation_count"])
    if hard_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
