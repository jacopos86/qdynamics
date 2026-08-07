#!/usr/bin/env python3
"""Build a Paper-I SNAKE Route-A/SPSA historical Pareto ledger.

The output is reporting support for the approved Table-I PDF.  It keeps the
full nondominated SNAKE front per Hamiltonian/regime while also emitting one
deterministic display representative per cell for the rectangular PDF table.
Only rows with validated first-hit Qiskit sidecar costs are admitted.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.exact_bench import generic_static_metric_enrichment as enrich
from pipelines.static_adapt.route_identity import ROUTE_ID_A

SCHEMA = "paper_i_routea_spsa_historical_pareto_v1"
SNAKE_ALGORITHM_ID = "static_family_native_adapt_phase3"
DEFAULT_THRESHOLD = 2e-4
DEFAULT_ROOTS = (
    Path("raw_outputs"),
    Path("artifacts/agent_runs"),
    Path("artifacts/json"),
    Path("MATH/paper_facing/paper_I_static_scaffold"),
)

FAMILY_ALIASES = {
    "hubbard": "hubbard",
    "ionic_hubbard": "ionic_hubbard",
    "ionic_hubbard_chain": "ionic_hubbard",
    "ttprime_hubbard": "ttprime_hubbard",
    "t_tprime_hubbard": "ttprime_hubbard",
    "tprime_hubbard": "ttprime_hubbard",
    "extended_hubbard": "extended_hubbard",
    "spinless_tv": "spinless_tv",
    "spinless_t_v": "spinless_tv",
    "bose_hubbard": "bose_hubbard",
    "harmonic_kerr_chain": "harmonic_kerr_chain",
    "harmonic_kerr": "harmonic_kerr_chain",
    "spin_boson": "spin_boson",
    "hh": "hh",
    "hubbard_holstein": "hh",
    "molecular_vibronic_h2": "molecular_vibronic_h2",
    "vibronic_h2": "molecular_vibronic_h2",
    "molecular_vibronic": "molecular_vibronic_h2",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _canonical_family(row: Mapping[str, Any]) -> str | None:
    raw = row.get("family") or row.get("hamiltonian") or row.get("hamiltonian_id")
    if raw is not None:
        lowered = str(raw).strip().lower()
        return FAMILY_ALIASES.get(lowered, lowered)
    text = str(row.get("case_id") or row.get("benchmark_id") or row.get("record_id") or "").lower()
    for alias, canonical in sorted(FAMILY_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if alias in text:
            return canonical
    return None


def _regime(row: Mapping[str, Any]) -> str | None:
    raw = row.get("regime") or row.get("physics_regime")
    text = str(raw if raw is not None else row.get("case_id") or row.get("benchmark_id") or row.get("record_id") or "").lower()
    if "weak" in text:
        return "weak"
    if "strong" in text:
        return "strong"
    return None


def _resolve_path(raw: Any, *, base: Path | None = None) -> Path | None:
    if raw in {None, ""}:
        return None
    path = Path(str(raw))
    if path.exists():
        return path
    if base is not None:
        candidate = base.parent / path
        if candidate.exists():
            return candidate
    return None


def _flatten_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return " ".join(_flatten_text(v) for v in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return " ".join(_flatten_text(v) for v in value)
    return "" if value is None else str(value)


def _is_snake(row: Mapping[str, Any]) -> bool:
    return str(row.get("algorithm_id") or "") == SNAKE_ALGORITHM_ID or str(row.get("method") or "").strip().lower() == "snake"


def _has_route_a_provenance(row: Mapping[str, Any]) -> bool:
    text = _flatten_text(row).lower()
    route_a = str(ROUTE_ID_A).lower()
    return route_a in text or "route_a" in text or "route a" in text


def _has_spsa_provenance(row: Mapping[str, Any]) -> bool:
    return "spsa" in _flatten_text(row).lower()


def _threshold_matches(row: Mapping[str, Any], cost: Mapping[str, Any], threshold: float) -> bool:
    candidates = (
        row.get("threshold"),
        row.get("tau_phys"),
        cost.get("threshold"),
        cost.get("tau_phys"),
    )
    for value in candidates:
        parsed = _finite(value)
        if parsed is not None and abs(parsed - float(threshold)) <= 1e-12:
            return True
    sidecar = row.get("paper_i_first_crossing_compiled_cost")
    if isinstance(sidecar, Mapping):
        for value in (sidecar.get("tau_phys"), sidecar.get("threshold")):
            parsed = _finite(value)
            if parsed is not None and abs(parsed - float(threshold)) <= 1e-12:
                return True
    return False


def _rows_from_payload(payload: Any, *, source_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(payload, list):
        iterable = payload
    elif isinstance(payload, Mapping):
        raw_rows = payload.get("row_results") or payload.get("rows") or payload.get("records")
        if isinstance(raw_rows, list):
            iterable = raw_rows
        else:
            result = payload.get("result")
            if isinstance(result, Mapping):
                row = dict(result)
                for key in ("family", "case_id", "algorithm_id", "method", "regime"):
                    if key in payload and key not in row:
                        row[key] = payload[key]
                row.setdefault("payload_path", str(source_path))
                return [row]
            iterable = [payload]
    else:
        iterable = []
    for item in iterable:
        if not isinstance(item, Mapping):
            continue
        row = dict(item)
        row.setdefault("payload_path", str(source_path))
        rows.append(row)
    return rows


def _candidate_paths(*, roots: Sequence[Path], summary_jsons: Sequence[Path], input_jsons: Sequence[Path]) -> list[Path]:
    paths: dict[str, Path] = {}
    for path in [*summary_jsons, *input_jsons]:
        if path.exists() and path.is_file():
            paths[str(path.resolve())] = path
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            paths[str(root.resolve())] = root
            continue
        for pattern in (
            "**/generic_static_single.json",
            "**/table_i_fixed_accuracy_calibrated_summary*.json",
            "**/*fixed_accuracy*_summary*.json",
            "**/*snake*table*support*.json",
        ):
            for path in root.glob(pattern):
                if path.is_file():
                    paths[str(path.resolve())] = path
    return list(paths.values())


def _validated_candidate(row: Mapping[str, Any], *, threshold: float, source_path: Path) -> tuple[dict[str, Any] | None, str]:
    if not _is_snake(row):
        return None, "not_snake"
    if not _has_route_a_provenance(row):
        return None, "missing_route_a_provenance"
    if not _has_spsa_provenance(row):
        return None, "missing_spsa_provenance"
    family = _canonical_family(row)
    regime = _regime(row)
    if family is None or regime is None:
        return None, "missing_family_or_regime"
    payload_path = _resolve_path(
        row.get("payload_path") or row.get("source_payload_path") or row.get("result_json") or row.get("source_result_path"),
        base=source_path,
    )
    cost = enrich.table_i_threshold_cost_from_row(
        algorithm_id=SNAKE_ALGORITHM_ID,
        row=row,
        threshold=float(threshold),
        record={
            "record_id": row.get("record_id"),
            "case_id": row.get("case_id") or row.get("benchmark_id"),
            "algorithm_id": SNAKE_ALGORITHM_ID,
        },
        result_path=payload_path,
    )
    if not _threshold_matches(row, cost, threshold):
        return None, "threshold_mismatch"
    valid = (
        str(cost.get("threshold_status") or "") == "ok_native_first_hit"
        and str(cost.get("cost_source") or "") == "snake_audited_first_crossing_compiled_cost"
        and cost.get("resource_display_allowed") is True
        and str(cost.get("compiled_resource_validation_status") or "") == "ok"
        and str(cost.get("sidecar_validation_status") or "") == "ok"
        and cost.get("sidecar_hash_verified") is True
    )
    if not valid:
        return None, "invalid_or_missing_first_hit_sidecar"
    out = dict(row)
    out.update(
        {
            "family": family,
            "method": "SNAKE",
            "algorithm_id": SNAKE_ALGORITHM_ID,
            "regime": regime,
            "threshold": float(threshold),
            "threshold_status": "ok_native_first_hit",
            "cost_included": True,
            "abs_delta_e": cost.get("abs_delta_e"),
            "S_alg": cost.get("S_alg"),
            "S_norm": cost.get("S_norm"),
            "count_2q": cost.get("count_2q"),
            "depth_2q": cost.get("depth_2q"),
            "circuit_depth": cost.get("circuit_depth"),
            "cost_source": cost.get("cost_source"),
            "source": cost.get("source"),
            "first_hit_semantics": cost.get("first_hit_semantics"),
            "method_cost_semantics": cost.get("method_cost_semantics"),
            "resource_display_allowed": True,
            "compiled_resource_validation_status": cost.get("compiled_resource_validation_status"),
            "compiled_resource_validation_reason": cost.get("compiled_resource_validation_reason"),
            "first_hit_cost_source_kind": cost.get("first_hit_cost_source_kind"),
            "sidecar_validation_status": cost.get("sidecar_validation_status"),
            "sidecar_validation_reason": cost.get("sidecar_validation_reason"),
            "sidecar_hash_verified": cost.get("sidecar_hash_verified"),
            "sidecar_source_kind": cost.get("sidecar_source_kind"),
            "source_result_sha256": cost.get("source_result_sha256"),
            "historical_pareto_source_path": str(source_path),
            "historical_pareto_payload_path": None if payload_path is None else str(payload_path),
            "historical_pareto_role": "candidate",
        }
    )
    return out, "accepted"


def _dimension_tuple(row: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        _finite(row.get("abs_delta_e")) or math.inf,
        _finite(row.get("count_2q")) or math.inf,
        _finite(row.get("depth_2q")) or math.inf,
        _finite(row.get("circuit_depth")) or math.inf,
        _finite(row.get("S_norm")) or math.inf,
    )


def _dominates(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    da = _dimension_tuple(a)
    db = _dimension_tuple(b)
    return all(x <= y for x, y in zip(da, db)) and any(x < y for x, y in zip(da, db))


def _pareto_front(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    front: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if any(_dominates(other, row) for j, other in enumerate(rows) if j != idx):
            continue
        front.append(dict(row))
    return front


def _display_key(row: Mapping[str, Any]) -> tuple[float, float, float, float, float, str]:
    return (
        _finite(row.get("count_2q")) or math.inf,
        _finite(row.get("depth_2q")) or math.inf,
        _finite(row.get("circuit_depth")) or math.inf,
        _finite(row.get("S_norm")) or math.inf,
        _finite(row.get("abs_delta_e")) or math.inf,
        str(row.get("historical_pareto_source_path") or row.get("record_id") or ""),
    )


def build_historical_pareto_payload(
    *,
    roots: Sequence[Path],
    summary_jsons: Sequence[Path],
    input_jsons: Sequence[Path],
    threshold: float,
) -> dict[str, Any]:
    accepted_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    rejection_counts: Counter[str] = Counter()
    scanned_paths = _candidate_paths(roots=roots, summary_jsons=summary_jsons, input_jsons=input_jsons)
    for path in scanned_paths:
        payload = _load_json(path)
        if payload is None:
            rejection_counts["unreadable_json"] += 1
            continue
        for row in _rows_from_payload(payload, source_path=path):
            candidate, reason = _validated_candidate(row, threshold=float(threshold), source_path=path)
            if candidate is None:
                rejection_counts[reason] += 1
                continue
            accepted_by_key[(str(candidate["family"]), str(candidate["regime"]))].append(candidate)
    groups: list[dict[str, Any]] = []
    display_rows: list[dict[str, Any]] = []
    for (family, regime), rows in sorted(accepted_by_key.items()):
        front = _pareto_front(rows)
        display = dict(sorted(front, key=_display_key)[0]) if front else None
        if display is not None:
            display["historical_pareto_role"] = "display_representative"
            display["historical_pareto_selection_rule"] = "min_N_2q_then_D_2q_then_D_circ_then_S_norm_then_abs_delta_e"
            display["historical_pareto_front_size"] = len(front)
            display["historical_pareto_candidate_count"] = len(rows)
            display_rows.append(display)
        groups.append(
            {
                "family": family,
                "method": "SNAKE",
                "regime": regime,
                "candidate_count": len(rows),
                "pareto_front_size": len(front),
                "pareto_front": front,
                "display_representative": display,
            }
        )
    return {
        "schema": SCHEMA,
        "threshold": float(threshold),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_policy": "route_a_spsa_validated_first_hit_qiskit_sidecar_pareto_v1",
        "display_representative_rule": "min_N_2q_then_D_2q_then_D_circ_then_S_norm_then_abs_delta_e",
        "scanned_path_count": len(scanned_paths),
        "accepted_candidate_count": sum(len(rows) for rows in accepted_by_key.values()),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "groups": groups,
        "row_results": display_rows,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", default=None, help="Artifact root to scan; may be repeated.")
    parser.add_argument("--summary-json", type=Path, action="append", default=(), help="Summary JSON to include; may be repeated.")
    parser.add_argument("--input-json", type=Path, action="append", default=(), help="Single JSON payload to include; may be repeated.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    roots = tuple(args.root) if args.root else DEFAULT_ROOTS
    payload = build_historical_pareto_payload(
        roots=roots,
        summary_jsons=tuple(args.summary_json or ()),
        input_jsons=tuple(args.input_json or ()),
        threshold=float(args.threshold),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
