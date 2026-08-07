#!/usr/bin/env python3
"""Shallow feature extraction for large Paper-I HH SNAKE/Route-A JSONs.

The full result JSONs can be hundreds of MB because they embed large histories and
operator payloads. This extractor intentionally avoids full JSON loading. It reads
small head/tail byte windows, extracts scalar JSON fields with regex/JSON scalar
parsing, and emits interpretable ML rows with missingness/quality flags.

It is not a paper-evidence promotion tool. It is a feature inventory tool.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_HEAD_BYTES = 2_000_000
DEFAULT_TAIL_BYTES = 2_000_000

PAPER_REGIMES = {"weak-weak", "strong-weak", "weak-strong", "strong-strong"}

SETTING_KEYS = (
    "phase1_prune_fraction",
    "phase1_prune_enabled",
    "phase1_prune_policy",
    "phase1_prune_max_regression",
    "phase1_prune_retained_gain_ratio",
    "phase2_batch_near_degenerate_ratio",
    "phase2_batch_rank_rel_tol",
    "phase2_batch_additivity_tol",
    "phase2_batch_size_cap",
    "phase3_enable_batching",
    "phase2_gamma_N",
    "phase2_gamma_N_schedule_mode",
    "phase2_motif_bonus_weight",
    "lambda_K_scale",
    "cost_lambda_K_scale",
    "adapt_beam_enabled",
    "adapt_beam_live_branches",
    "adapt_beam_children_per_parent",
    "adapt_inner_optimizer",
    "adapt_pool",
    "static_meta_feature_profile",
    "benchmark_id",
    "n_ph_max",
    "n_ph_work",
    "u",
    "U_over_t",
    "u_over_t",
    "g_ep",
    "lambda",
    "omega0",
)

HEAD_OUTCOME_KEYS = (
    "abs_delta_e",
    "energy",
    "exact_gs_energy",
    "final_depth",
    "final_runtime_parameter_count",
    "new_admission_records",
    "stop_reason",
    "adapt_beam_enabled",
    "adapt_beam_live_branches",
    "adapt_beam_children_per_parent",
)

NUMERIC_EFFECTIVE_KEYS = (
    "phase1_prune_fraction",
    "phase1_prune_max_regression",
    "phase1_prune_retained_gain_ratio",
    "phase2_batch_near_degenerate_ratio",
    "phase2_batch_rank_rel_tol",
    "phase2_batch_additivity_tol",
    "lambda_K_scale",
    "cost_lambda_K_scale",
)


def _read_window(path: Path, *, head_bytes: int, tail_bytes: int) -> tuple[str, str, int]:
    size = path.stat().st_size
    with path.open("rb") as fh:
        head = fh.read(min(head_bytes, size))
        if size > tail_bytes:
            fh.seek(max(0, size - tail_bytes))
            tail = fh.read(tail_bytes)
        else:
            fh.seek(0)
            tail = fh.read(size)
    return head.decode("utf-8", "replace"), tail.decode("utf-8", "replace"), size


def _json_scalar_after_colon(text: str, pos: int) -> Any:
    colon = text.find(":", pos)
    if colon < 0:
        return None
    idx = colon + 1
    while idx < len(text) and text[idx].isspace():
        idx += 1
    if idx >= len(text):
        return None
    # Decode strings with JSON decoder for escapes.
    if text[idx] == '"':
        try:
            value, _end = json.JSONDecoder().raw_decode(text[idx: idx + 20_000])
            return value
        except Exception:
            return None
    match = re.match(r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|true|false|null", text[idx: idx + 200])
    if not match:
        return None
    raw = match.group(0)
    if raw == "true":
        return True
    if raw == "false":
        return False
    if raw == "null":
        return None
    try:
        if any(c in raw for c in ".eE"):
            return float(raw)
        return int(raw)
    except Exception:
        return None


def _extract_scalar(text: str, key: str, *, prefer: str = "first") -> Any:
    needle = f'"{key}"'
    pos = text.rfind(needle) if prefer == "last" else text.find(needle)
    if pos < 0:
        return None
    return _json_scalar_after_colon(text, pos)


def _extract_many(text: str, keys: Sequence[str], *, prefer: str) -> dict[str, Any]:
    return {key: _extract_scalar(text, key, prefer=prefer) for key in keys}


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _finite_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _normalize_regime(*values: Any) -> str:
    text = " ".join(_clean(v) for v in values).lower().replace("-", "_").replace("/", "_")
    for needle, label in (
        ("u8_strong_strong", "u8-strong-strong"),
        ("u8_strong_weak", "u8-strong-weak"),
        ("intermediate_strong", "intermediate-strong"),
        ("intermediate_weak", "intermediate-weak"),
        ("strong_strong", "strong-strong"),
        ("weak_strong", "weak-strong"),
        ("strong_weak", "strong-weak"),
        ("weak_weak", "weak-weak"),
    ):
        if needle in text:
            return label
    for pattern, label in (
        (r"(?:^|[_\W])(?:hh_)?ss(?:[_\W]|$)", "strong-strong"),
        (r"(?:^|[_\W])(?:hh_)?ws(?:[_\W]|$)", "weak-strong"),
        (r"(?:^|[_\W])(?:hh_)?sw(?:[_\W]|$)", "strong-weak"),
        (r"(?:^|[_\W])(?:hh_)?ww(?:[_\W]|$)", "weak-weak"),
    ):
        if re.search(pattern, text):
            return label
    return "unknown"


def _nph_from_regime(regime: str) -> int | None:
    if regime in {"weak-weak", "strong-weak"}:
        return 2
    if regime in {"weak-strong", "strong-strong"}:
        return 4
    return None


def _nph_from_text(text: str) -> int | None:
    match = re.search(r"(?:nph|n_ph)[_-]?(\d+)", text.lower())
    if match:
        return _finite_int(match.group(1))
    return None


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _quality_flags(row: Mapping[str, Any]) -> list[str]:
    flags: list[str] = []
    if row.get("regime") not in PAPER_REGIMES:
        flags.append("non_paper_i_or_unknown_regime")
    if row.get("n_ph") is None:
        flags.append("missing_n_ph")
    if row.get("delta_E_best_proxy") is None:
        flags.append("missing_delta_E_proxy")
    if row.get("history_len_proxy") in {None, 0}:
        flags.append("missing_history_len_proxy")
    missing = row.get("effective_knob_missing") or []
    if missing:
        flags.append("effective_knob_missingness")
    if row.get("path_u8_signal"):
        flags.append("u8_diagnostic_context")
    if row.get("path_first_crossing_signal"):
        flags.append("first_crossing_snapshot")
    if row.get("path_live_or_current_signal"):
        flags.append("live_or_current_snapshot")
    if row.get("phase2_motif_bonus_weight") not in {None, 0, 0.0}:
        flags.append("motif_bonus_nonzero")
    schedule = row.get("phase2_gamma_N_schedule_mode")
    if schedule not in {None, "", "fixed"}:
        flags.append("nonflat_or_scheduled_novelty")
    return sorted(set(flags))


def extract_large_json_features(path: Path, *, head_bytes: int = DEFAULT_HEAD_BYTES, tail_bytes: int = DEFAULT_TAIL_BYTES) -> dict[str, Any]:
    head, tail, size = _read_window(path, head_bytes=head_bytes, tail_bytes=tail_bytes)
    head_vals = _extract_many(head, HEAD_OUTCOME_KEYS, prefer="first")
    tail_vals = _extract_many(tail, SETTING_KEYS, prefer="last")
    path_text = path.as_posix()
    benchmark_id = _coalesce(tail_vals.get("benchmark_id"), _extract_scalar(head, "benchmark_id", prefer="first"))
    regime = _normalize_regime(benchmark_id, path.name, path_text)
    n_ph = _finite_int(_coalesce(tail_vals.get("n_ph_work"), tail_vals.get("n_ph_max"), _nph_from_text(path_text), _nph_from_regime(regime)))
    lambda_k = _finite_float(_coalesce(tail_vals.get("lambda_K_scale"), tail_vals.get("cost_lambda_K_scale")))
    effective_params = {
        "lambda_K_scale": lambda_k,
        "phase1_prune_fraction": _finite_float(tail_vals.get("phase1_prune_fraction")),
        "prune_recoverability_slack_scale": None,
        "batch_near_degenerate_ratio_shared": _finite_float(tail_vals.get("phase2_batch_near_degenerate_ratio")),
        "batch_rank_rel_tol_shared": _finite_float(tail_vals.get("phase2_batch_rank_rel_tol")),
        "batch_additivity_slack_scale": _finite_float(tail_vals.get("phase2_batch_additivity_tol")),
    }
    recovered = sorted(k for k, v in effective_params.items() if v is not None)
    delta = _finite_float(head_vals.get("abs_delta_e"))
    final_depth = _finite_int(_coalesce(head_vals.get("final_depth"), head_vals.get("new_admission_records")))
    row: dict[str, Any] = {
        "path": str(path),
        "file_name": path.name,
        "size_bytes": size,
        "parse_status": "shallow_large_json_ok",
        "method": "snake",
        "regime": regime,
        "n_ph": n_ph,
        "benchmark_id": benchmark_id,
        "delta_E_best_proxy": delta,
        "delta_E_final_proxy": delta,
        "k_best_energy_proxy": final_depth,
        "k_final_proxy": final_depth,
        "history_len_proxy": final_depth,
        "energy_proxy": _finite_float(head_vals.get("energy")),
        "exact_gs_energy_proxy": _finite_float(head_vals.get("exact_gs_energy")),
        "final_runtime_parameter_count": _finite_int(head_vals.get("final_runtime_parameter_count")),
        "stop_reason": head_vals.get("stop_reason"),
        "effective_params": effective_params,
        "effective_knob_recovered_count": len(recovered),
        "effective_knob_recovered": recovered,
        "effective_knob_missing": sorted(k for k, v in effective_params.items() if v is None),
        "phase1_prune_enabled": tail_vals.get("phase1_prune_enabled"),
        "phase1_prune_policy": tail_vals.get("phase1_prune_policy"),
        "phase2_gamma_N": _finite_float(tail_vals.get("phase2_gamma_N")),
        "phase2_gamma_N_schedule_mode": tail_vals.get("phase2_gamma_N_schedule_mode"),
        "phase2_motif_bonus_weight": _finite_float(tail_vals.get("phase2_motif_bonus_weight")),
        "phase3_enable_batching": tail_vals.get("phase3_enable_batching"),
        "adapt_beam_enabled": _coalesce(tail_vals.get("adapt_beam_enabled"), head_vals.get("adapt_beam_enabled")),
        "adapt_beam_live_branches": _finite_int(_coalesce(tail_vals.get("adapt_beam_live_branches"), head_vals.get("adapt_beam_live_branches"))),
        "adapt_beam_children_per_parent": _finite_int(_coalesce(tail_vals.get("adapt_beam_children_per_parent"), head_vals.get("adapt_beam_children_per_parent"))),
        "adapt_inner_optimizer": tail_vals.get("adapt_inner_optimizer"),
        "adapt_pool": tail_vals.get("adapt_pool"),
        "static_meta_feature_profile": tail_vals.get("static_meta_feature_profile"),
        "u_or_U_over_t": _finite_float(_coalesce(tail_vals.get("U_over_t"), tail_vals.get("u_over_t"), tail_vals.get("u"))),
        "g_or_lambda": _finite_float(_coalesce(tail_vals.get("lambda"), tail_vals.get("g_ep"))),
        "omega0": _finite_float(tail_vals.get("omega0")),
        "path_u8_signal": bool(re.search(r"u8", path_text, re.I)),
        "path_live_or_current_signal": bool(re.search(r"current|live", path_text, re.I)),
        "path_first_crossing_signal": bool(re.search(r"first_crossing", path_text, re.I)),
        "head_bytes": head_bytes,
        "tail_bytes": tail_bytes,
    }
    row["quality_flags"] = _quality_flags(row)
    row["include_for_ml"] = True
    row["include_for_strict_bound_training"] = (
        row["regime"] in PAPER_REGIMES
        and row["delta_E_best_proxy"] is not None
        and row["effective_knob_recovered_count"] >= 3
        and not row["path_first_crossing_signal"]
    )
    return row


def _load_inventory_rows(path: Path, statuses: set[str]) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"Inventory has no rows[] list: {path}")
    out = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if statuses and str(row.get("parse_status")) not in statuses:
            continue
        if row.get("path"):
            out.append(dict(row))
    return out


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", required=True, help="Inventory JSON with rows[].path entries.")
    p.add_argument("--output-ndjson", required=True, help="Output NDJSON feature rows.")
    p.add_argument("--summary-json", required=True, help="Output summary JSON.")
    p.add_argument("--status", action="append", default=["deferred_large_json"], help="Inventory parse_status to extract. Repeatable.")
    p.add_argument("--max-files", type=int, default=0, help="Optional cap; 0 means all matching rows.")
    p.add_argument("--head-bytes", type=int, default=DEFAULT_HEAD_BYTES)
    p.add_argument("--tail-bytes", type=int, default=DEFAULT_TAIL_BYTES)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    start = time.time()
    rows = _load_inventory_rows(Path(args.inventory), set(args.status or []))
    if int(args.max_files) > 0:
        rows = rows[: int(args.max_files)]
    features: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for idx, inv_row in enumerate(rows):
        path = Path(str(inv_row["path"]))
        try:
            row = extract_large_json_features(path, head_bytes=int(args.head_bytes), tail_bytes=int(args.tail_bytes))
            row["inventory_index"] = inv_row.get("inventory_index")
            row["extract_index"] = idx
            features.append(row)
        except Exception as exc:
            errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}", "extract_index": idx})
    _write_jsonl(Path(args.output_ndjson), features)
    summary: dict[str, Any] = {
        "schema": "paper_i_hh_snake_shallow_large_json_summary_v1",
        "inventory": str(args.inventory),
        "output_ndjson": str(args.output_ndjson),
        "requested_count": len(rows),
        "feature_count": len(features),
        "error_count": len(errors),
        "head_bytes": int(args.head_bytes),
        "tail_bytes": int(args.tail_bytes),
        "runtime_seconds": time.time() - start,
        "strict_bound_training_count": sum(1 for row in features if row.get("include_for_strict_bound_training")),
        "by_regime": {},
        "by_quality_flag": {},
        "errors_sample": errors[:50],
    }
    for row in features:
        regime = str(row.get("regime"))
        summary["by_regime"][regime] = summary["by_regime"].get(regime, 0) + 1
        for flag in row.get("quality_flags", []):
            summary["by_quality_flag"][flag] = summary["by_quality_flag"].get(flag, 0) + 1
    _write_json(Path(args.summary_json), summary)


if __name__ == "__main__":  # pragma: no cover
    main()
