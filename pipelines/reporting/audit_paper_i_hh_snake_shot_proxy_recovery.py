#!/usr/bin/env python3
"""Audit whether HH Table-III SNAKE rows can recover comparable shots_total.

This is a reporting/provenance audit.  It does not launch runs and does not
infer missing deterministic-shot inputs from legacy controller/S_norm proxies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.exact_bench.snake_table_i_measurement_work import (
    snake_algorithmic_work_from_payload,
    snake_deterministic_shot_proxy_from_payload,
)

DEFAULT_SOURCE_MAP = Path("MATH/paper_facing/paper_I_static_scaffold/hh_tableiii_convergence_sources.json")
DEFAULT_OUTPUT_JSON = Path("output/pdf/paper_i_hh_tableiii_snake_shot_proxy_recovery_audit_20260612.json")
REGIME_ORDER = ("weak_weak", "strong_weak", "weak_strong", "strong_strong")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    if not parsed.is_integer() or parsed < 0:
        return None
    return int(parsed)


def _add_path(candidates: list[dict[str, Any]], *, kind: str, raw: Any) -> None:
    if raw is None or raw == "":
        return
    path = str(raw)
    if any(row["path"] == path for row in candidates):
        return
    candidates.append({"kind": kind, "path": path})


def _source_candidates(method: Mapping[str, Any], promotion_payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for key in (
        "source_json",
        "history_source_json",
        "strict_replay_json",
        "resource_source_json",
        "compiled_cost_json",
        "previous_json_backed_source_json",
        "base_source_json_before_continuation",
        "previous_source_json_before_stdout_continuation_20260601",
        "stdout_continuation_promotion_json",
        "promotion_json",
    ):
        _add_path(candidates, kind=f"method.{key}", raw=method.get(key))
    if isinstance(promotion_payload, Mapping):
        for key in ("source_current_json", "strict_replay_json", "source_json", "history_source_json"):
            _add_path(candidates, kind=f"promotion.{key}", raw=promotion_payload.get(key))
    return candidates


def _history_positions_from_mapping(mapping: Mapping[str, Any] | None) -> list[int]:
    if not isinstance(mapping, Mapping):
        return []
    out: list[int] = []
    for key in ("chosen_prefix_len", "history_position_tau", "k_tau", "k_pl", "table_display_iteration", "display_iteration"):
        value = _int_or_none(mapping.get(key))
        if value is not None and value > 0 and value not in out:
            out.append(value)
    table_prefix = mapping.get("table_display_prefix")
    if isinstance(table_prefix, Mapping):
        value = _int_or_none(table_prefix.get("k_pl") or table_prefix.get("history_position") or table_prefix.get("iteration"))
        if value is not None and value > 0 and value not in out:
            out.append(value)
    visible = mapping.get("visible_cells")
    if isinstance(visible, Mapping):
        value = _int_or_none(visible.get("k_pl") or visible.get("chosen_prefix_len") or visible.get("history_position"))
        if value is not None and value > 0 and value not in out:
            out.append(value)
    return out


def _audit_payload_scope(payload: Mapping[str, Any], *, scope: str, history_position: int | None, source_label: str) -> dict[str, Any]:
    try:
        work, work_audit = snake_algorithmic_work_from_payload(
            payload,
            scope=scope,
            history_position=history_position,
            source_label=source_label,
        )
        fields, shot_audit = snake_deterministic_shot_proxy_from_payload(
            payload,
            scope=scope,
            history_position=history_position,
            source_label=source_label,
        )
    except Exception as exc:  # fail closed and keep exact exception text in audit
        return {
            "scope": scope,
            "history_position": history_position,
            "status": "audit_exception",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "can_backfill": False,
        }
    shot_status = str(shot_audit.get("status") or "unknown")
    return {
        "scope": scope,
        "history_position": history_position,
        "status": "can_backfill" if shot_status == "ok" else "blocked",
        "can_backfill": shot_status == "ok",
        "shots_total": fields.get("shots_total"),
        "S_alg_status": work.get("S_alg_status"),
        "S_alg": work.get("S_alg"),
        "deterministic_shot_status": shot_status,
        "deterministic_shot_blocker": shot_audit.get("blocker"),
        "deterministic_shot_audit": shot_audit,
        "algorithmic_work_status": work_audit.get("status"),
    }


def _audit_candidate(candidate: Mapping[str, Any], *, history_positions: Sequence[int]) -> dict[str, Any]:
    raw_path = str(candidate["path"])
    path = Path(raw_path)
    base = {
        "kind": candidate.get("kind"),
        "path": raw_path,
        "exists_locally": path.exists() and path.is_file(),
    }
    if not base["exists_locally"]:
        base.update(status="source_not_available_locally", scopes=[])
        return base
    try:
        payload = _read_json(path)
    except Exception as exc:
        base.update(status="source_unreadable", exception_type=type(exc).__name__, exception=str(exc), scopes=[])
        return base
    if not isinstance(payload, Mapping):
        base.update(status="source_not_mapping", scopes=[])
        return base
    scopes = [_audit_payload_scope(payload, scope="terminal", history_position=None, source_label=raw_path)]
    for history_position in history_positions:
        scopes.append(_audit_payload_scope(payload, scope="display_prefix", history_position=int(history_position), source_label=raw_path))
    base.update(
        status="can_backfill" if any(scope.get("can_backfill") for scope in scopes) else "blocked",
        source_sha256=_sha256(path),
        top_level_keys=sorted(str(k) for k in payload.keys())[:80],
        scopes=scopes,
    )
    return base


def build_audit(source_map_path: Path = DEFAULT_SOURCE_MAP) -> dict[str, Any]:
    source_map = _read_json(source_map_path)
    if not isinstance(source_map, Mapping):
        raise ValueError(f"source map is not a JSON object: {source_map_path}")
    rows: list[dict[str, Any]] = []
    for regime in REGIME_ORDER:
        regime_block = source_map.get("regimes", {}).get(regime, {})
        method = regime_block.get("methods", {}).get("SNAKE", {}) if isinstance(regime_block, Mapping) else {}
        if not isinstance(method, Mapping):
            rows.append({"regime": regime, "status": "missing_snake_method_block", "can_backfill": False})
            continue
        promotion_payload = None
        source_json = method.get("source_json")
        if isinstance(source_json, str) and Path(source_json).exists():
            try:
                loaded = _read_json(Path(source_json))
                if isinstance(loaded, Mapping):
                    promotion_payload = loaded
            except Exception:
                promotion_payload = None
        history_positions = []
        for mapping in (method, promotion_payload):
            for value in _history_positions_from_mapping(mapping):
                if value not in history_positions:
                    history_positions.append(value)
        candidates = _source_candidates(method, promotion_payload)
        candidate_audits = [_audit_candidate(candidate, history_positions=history_positions) for candidate in candidates]
        can_backfill_candidates = [candidate for candidate in candidate_audits if candidate.get("status") == "can_backfill"]
        blockers = Counter()
        for candidate in candidate_audits:
            for scope in candidate.get("scopes", []):
                if scope.get("can_backfill"):
                    continue
                blockers[str(scope.get("deterministic_shot_status") or scope.get("status") or candidate.get("status"))] += 1
            if not candidate.get("scopes"):
                blockers[str(candidate.get("status"))] += 1
        rows.append(
            {
                "regime": regime,
                "method": "SNAKE",
                "status": "can_backfill" if can_backfill_candidates else "needs_replay_or_rerun_for_comparable_shots_total",
                "can_backfill": bool(can_backfill_candidates),
                "history_positions_checked": history_positions,
                "candidate_count": len(candidate_audits),
                "can_backfill_candidate_count": len(can_backfill_candidates),
                "blocker_counts": dict(sorted(blockers.items())),
                "candidates": candidate_audits,
            }
        )
    status_counts = Counter(str(row.get("status")) for row in rows)
    return {
        "schema": "paper_i_hh_tableiii_snake_shot_proxy_recovery_audit_v1",
        "source_map": str(source_map_path),
        "source_map_sha256_at_audit_time": _sha256(source_map_path),
        "table_label": source_map.get("table_label"),
        "policy": {
            "deterministic_shot_formula": "shots_total = shots_per_pauli_term_proxy * hamiltonian_pauli_term_count * (energy_eval_count_proxy + gradient_operator_probe_count_proxy + metric_operator_probe_count_proxy)",
            "legacy_controller_or_s_norm_proxies": "diagnostic_only_not_comparable_shots_total",
            "missing_required_inputs": "fail_closed_do_not_infer",
        },
        "row_count": len(rows),
        "can_backfill_count": sum(1 for row in rows if row.get("can_backfill")),
        "status_counts": dict(sorted(status_counts.items())),
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-map", type=Path, default=DEFAULT_SOURCE_MAP)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args(argv)
    audit = build_audit(args.source_map)
    _write_json(args.output_json, audit)
    print(json.dumps({"output_json": str(args.output_json), "status_counts": audit["status_counts"], "can_backfill_count": audit["can_backfill_count"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
