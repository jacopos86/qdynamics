#!/usr/bin/env python3
"""Summarize compact SNAKE/Phase3 frontier diagnostics from static ADAPT artifacts.

The runtime already persists rich selected-branch and beam-search telemetry in the
native ADAPT result JSON.  This support parser extracts the small per-depth table
needed for nph2 convergence debugging without changing route selection logic.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "nph2_frontier_diagnostics_v1"
ROUND_COUNT_FIELDS = (
    "raw_candidate_record_count",
    "phase2_raw_candidate_record_count",
    "phase1_shortlist_size",
    "phase2_shortlist_size",
    "phase3_shortlist_size",
    "proposal_family_count",
    "proposals_selected_count",
    "frontier_input_count",
    "frontier_kept_count",
    "terminal_kept_count",
)
ROUND_SCORE_FIELDS = (
    "best_available_gradient",
    "best_available_simple_score",
    "best_available_phase2_raw_score",
    "best_available_full_v2_score",
    "best_available_gain",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_existing_path(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = (
        path / "generic_static_single.json",
        path / "result" / "generic_static_single.json",
        path / "result.json",
        path / "json" / "result.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No supported result JSON found under {path}")


def _resolve_artifact_path(raw: Any, *, anchor: Path) -> Path | None:
    if raw is None or raw == "":
        return None
    candidate = Path(str(raw))
    if candidate.is_absolute() and candidate.exists():
        return candidate
    search_roots = (Path.cwd(), anchor.parent, anchor.parent.parent)
    for root in search_roots:
        resolved = root / candidate
        if resolved.exists():
            return resolved
    return candidate if candidate.exists() else None


def _resolve_inputs(input_path: Path, stdout_log: Path | None = None) -> tuple[Path, dict[str, Any] | None, Path, dict[str, Any], Path | None]:
    payload_path = _first_existing_path(input_path)
    payload = _load_json(payload_path)
    generic_payload: dict[str, Any] | None = None
    adapt_path = payload_path
    adapt_payload = payload

    if "adapt_vqe" not in payload:
        generic_payload = payload
        result_payload = payload.get("result") if isinstance(payload.get("result"), Mapping) else {}
        result_json = result_payload.get("result_json") if isinstance(result_payload, Mapping) else None
        resolved = _resolve_artifact_path(result_json, anchor=payload_path)
        if resolved is None:
            case_id = str(payload.get("case_id") or (result_payload.get("benchmark_id") if isinstance(result_payload, Mapping) else ""))
            candidate = payload_path.parent / case_id / "json" / "result.json"
            if candidate.exists():
                resolved = candidate
        if resolved is None:
            raise FileNotFoundError(f"Could not resolve native ADAPT result_json from {payload_path}")
        adapt_path = resolved
        adapt_payload = _load_json(adapt_path)

    log_path = stdout_log
    if log_path is None:
        candidate = adapt_path.parent.parent / "logs" / "stdout.log"
        if candidate.exists():
            log_path = candidate
    return payload_path, generic_payload, adapt_path, adapt_payload, log_path


def _safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _rows(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _best(records: Sequence[Mapping[str, Any]], *keys: str) -> float | None:
    best_value: float | None = None
    for row in records:
        for key in keys:
            value = _safe_float(row.get(key))
            if value is None:
                continue
            if best_value is None or value > best_value:
                best_value = value
    return best_value


def _history_by_depth(adapt_vqe: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    out: dict[int, Mapping[str, Any]] = {}
    for row in _rows(adapt_vqe.get("history")):
        depth = _safe_int(row.get("depth"))
        if depth is not None:
            out[int(depth)] = row
    return out


def _ai_log_rounds(stdout_log: Path | None) -> list[dict[str, Any]]:
    if stdout_log is None or not stdout_log.exists():
        return []
    rounds: list[dict[str, Any]] = []
    for line in stdout_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("AI_LOG "):
            continue
        try:
            event = json.loads(line[len("AI_LOG ") :])
        except json.JSONDecodeError:
            continue
        if event.get("event") == "hardcoded_adapt_beam_round_done":
            rounds.append(dict(event))
    return rounds


def _terminal_surface_counts(beam_search: Mapping[str, Any], depth: int) -> dict[str, int | None]:
    """Best-effort fallback for old artifacts whose final empty round lacks new fields."""

    candidates = []
    for row in _rows(beam_search.get("finalist_summaries")):
        depth_local = _safe_int(row.get("depth_local"))
        if depth_local not in {depth, depth - 1}:
            continue
        candidates.append(row)
    if not candidates:
        return {}
    return {
        "phase2_raw_candidate_record_count": max(_safe_int(row.get("scored_surface_count")) or 0 for row in candidates),
        "phase3_shortlist_size": max(_safe_int(row.get("retained_shortlist_count")) or 0 for row in candidates),
    }


def _round_from_history(depth: int, history_row: Mapping[str, Any] | None) -> dict[str, Any]:
    if history_row is None:
        return {}
    scored = _rows(history_row.get("scored_surface_records")) or _rows(history_row.get("shortlisted_records"))
    retained = _rows(history_row.get("retained_shortlist_records"))
    return {
        "phase2_raw_candidate_record_count": _coalesce(_safe_int(history_row.get("scored_surface_size")), len(scored) if scored else None),
        "phase2_shortlist_size": _coalesce(_safe_int(history_row.get("shortlist_size")), len(scored) if scored else None),
        "phase3_shortlist_size": _coalesce(_safe_int(history_row.get("retained_shortlist_size")), len(retained) if retained else None),
        "best_available_gradient": _safe_float(history_row.get("max_grad")),
        "best_available_simple_score": _coalesce(_safe_float(history_row.get("simple_score")), _best(scored, "simple_score")),
        "best_available_phase2_raw_score": _coalesce(_safe_float(history_row.get("phase2_raw_score")), _best(scored, "phase2_raw_score")),
        "best_available_full_v2_score": _coalesce(_safe_float(history_row.get("full_v2_score")), _best(scored, "full_v2_score")),
        "best_available_gain": _coalesce(_safe_float(history_row.get("phase2_raw_trust_gain")), _safe_float(history_row.get("delta_abs_drop_from_prev"))),
    }


def _compact_summary_line(row: Mapping[str, Any] | None) -> str:
    if not row:
        return "frontier_diagnostics=unavailable"
    parts = [f"d{row.get('depth')}"]
    for key, label in (
        ("raw_candidate_record_count", "raw"),
        ("phase1_shortlist_size", "p1"),
        ("phase2_shortlist_size", "p2"),
        ("phase3_shortlist_size", "p3"),
    ):
        value = row.get(key)
        if value is not None:
            parts.append(f"{label}={value}")
    parts.append(f"props={row.get('proposals_selected_count')}/{row.get('proposal_family_count')}")
    parts.append(f"frontier={row.get('frontier_input_count')}->{row.get('frontier_kept_count')}")
    if row.get("best_available_gradient") is not None:
        parts.append(f"best_grad={float(row['best_available_gradient']):.3e}")
    if row.get("best_available_full_v2_score") is not None:
        parts.append(f"best_full_v2={float(row['best_available_full_v2_score']):.3e}")
    if row.get("stop_reason"):
        parts.append(f"stop={row.get('stop_reason')}")
    return "; ".join(parts)


def summarize_frontier_diagnostics(input_path: str | Path, *, stdout_log: str | Path | None = None) -> dict[str, Any]:
    payload_path, generic_payload, adapt_path, adapt_payload, log_path = _resolve_inputs(
        Path(input_path),
        Path(stdout_log) if stdout_log is not None else None,
    )
    adapt_vqe = adapt_payload.get("adapt_vqe") if isinstance(adapt_payload.get("adapt_vqe"), Mapping) else {}
    continuation = adapt_vqe.get("continuation") if isinstance(adapt_vqe.get("continuation"), Mapping) else {}
    beam_search = continuation.get("beam_search") if isinstance(continuation.get("beam_search"), Mapping) else {}
    history = _history_by_depth(adapt_vqe)
    rounds = _rows(beam_search.get("rounds")) or _ai_log_rounds(log_path)
    result_payload = generic_payload.get("result", {}) if isinstance(generic_payload, Mapping) else adapt_vqe
    if not isinstance(result_payload, Mapping):
        result_payload = {}
    boson_subspace = (
        adapt_vqe.get("boson_subspace_diagnostics")
        if isinstance(adapt_vqe.get("boson_subspace_diagnostics"), Mapping)
        else {}
    )

    rows: list[dict[str, Any]] = []
    for round_row in rounds:
        depth = _safe_int(round_row.get("depth"))
        if depth is None:
            continue
        fallback = _round_from_history(int(depth), history.get(int(depth)))
        terminal_fallback = _terminal_surface_counts(beam_search, int(depth))
        row: dict[str, Any] = {"depth": int(depth)}
        for field in ROUND_COUNT_FIELDS:
            value = _safe_int(round_row.get(field))
            if value is None:
                value = _safe_int(fallback.get(field))
            if value is None:
                value = _safe_int(terminal_fallback.get(field))
            row[field] = value
        for field in ROUND_SCORE_FIELDS:
            value = _safe_float(round_row.get(field))
            if value is None:
                value = _safe_float(fallback.get(field))
            row[field] = value
        row["stop_reason"] = round_row.get("stop_reason")
        if row["stop_reason"] is None and row.get("frontier_input_count") and row.get("frontier_kept_count") == 0:
            row["stop_reason"] = adapt_vqe.get("stop_reason") or beam_search.get("winner_stop_reason") or "frontier_empty"
        if isinstance(round_row.get("parent_stop_reason_counts"), Mapping):
            row["parent_stop_reason_counts"] = dict(round_row.get("parent_stop_reason_counts", {}))
        rows.append(row)

    collapse_rows = [row for row in rows if row.get("frontier_input_count") and row.get("frontier_kept_count") == 0]
    zero_proposal_rows = [row for row in rows if row.get("proposal_family_count") == 0 or row.get("proposals_selected_count") == 0]
    terminal_row = collapse_rows[0] if collapse_rows else (rows[-1] if rows else None)
    summary = {
        "schema": f"{SCHEMA}_summary",
        "row_count": len(rows),
        "collapse_depth": terminal_row.get("depth") if terminal_row and terminal_row.get("frontier_kept_count") == 0 else None,
        "first_zero_proposal_depth": zero_proposal_rows[0].get("depth") if zero_proposal_rows else None,
        "terminal_stop_reason": adapt_vqe.get("stop_reason") or result_payload.get("stop_reason") or beam_search.get("winner_stop_reason"),
        "pool_size": _coalesce(_safe_int(result_payload.get("pool_size")), _safe_int(adapt_vqe.get("pool_size"))),
        "ansatz_depth": _coalesce(_safe_int(result_payload.get("ansatz_depth")), _safe_int(adapt_vqe.get("ansatz_depth"))),
        "abs_delta_e_same_cutoff": _coalesce(_safe_float(result_payload.get("abs_delta_e_same_cutoff")), _safe_float(adapt_vqe.get("abs_delta_e"))),
        "abs_delta_e_ref": _coalesce(_safe_float(result_payload.get("abs_delta_e_reference")), _safe_float(result_payload.get("abs_delta_e")), _safe_float(adapt_vqe.get("abs_delta_e"))),
        "cutoff_abs_delta_e": _safe_float(result_payload.get("cutoff_abs_delta_e")),
        "boson_illegal_probability_max": _coalesce(
            _safe_float(result_payload.get("boson_illegal_probability_max")),
            _safe_float(adapt_vqe.get("boson_illegal_probability_max")),
            _safe_float(boson_subspace.get("boson_illegal_probability_max")),
        ),
        "frontier_summary": _compact_summary_line(terminal_row),
    }
    return {
        "schema": SCHEMA,
        "input_json": str(payload_path),
        "adapt_result_json": str(adapt_path),
        "stdout_log": str(log_path) if log_path is not None else None,
        "case_id": generic_payload.get("case_id") if isinstance(generic_payload, Mapping) else adapt_payload.get("settings", {}).get("benchmark_id"),
        "family": generic_payload.get("family") if isinstance(generic_payload, Mapping) else adapt_payload.get("settings", {}).get("problem"),
        "algorithm_id": generic_payload.get("algorithm_id") if isinstance(generic_payload, Mapping) else None,
        "summary": summary,
        "per_depth": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Phase3/SNAKE frontier and shortlist diagnostics.")
    parser.add_argument("--input", type=Path, required=True, help="generic_static_single.json, native result.json, or containing directory")
    parser.add_argument("--stdout-log", type=Path, default=None, help="optional stdout.log AI_LOG source")
    parser.add_argument("--output", type=Path, default=None, help="write JSON summary here")
    parser.add_argument("--print-summary", action="store_true", help="print one compact frontier summary line")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = summarize_frontier_diagnostics(args.input, stdout_log=args.stdout_log)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.print_summary or args.output is None:
        print(payload["summary"]["frontier_summary"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
