#!/usr/bin/env python3
"""Summarize Route-A Phase0 nph=1 Optuna outputs into a machine-readable scoreboard."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "routeA_phase0_nph1_scoreboard_v1"
THRESHOLDS = (5e-5, 1e-6, 1e-8)
FIELDS = (
    "record_id",
    "benchmark_id",
    "family",
    "class_key",
    "source_kind",
    "same_cutoff_abs_delta_e",
    "meets_5e-5",
    "meets_1e-6",
    "meets_1e-8",
    "compiled_two_qubit_count",
    "compiled_depth",
    "parameter_count",
    "S_alg",
    "measurement_shots_proxy",
    "S_phys",
    "phase0_pilot_max_records",
    "summary_path",
)


def _as_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if out == out and abs(out) != float("inf") else None


def _get(mapping: Mapping[str, Any], *paths: str) -> Any:
    for path in paths:
        node: Any = mapping
        ok = True
        for part in path.split("."):
            if not isinstance(node, Mapping) or part not in node:
                ok = False
                break
            node = node[part]
        if ok:
            return node
    return None


def _result_rows_from_summary(path: Path, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    record_id = None
    study_name = str(payload.get("study_name") or "")
    if "chtc_phase3_optuna/" in study_name:
        record_id = study_name.split("chtc_phase3_optuna/", 1)[1].split("/", 1)[0]
    phase0_max = _get(payload, "phase0.base_policy.phase0_pilot_max_records", "base_policy.static.phase0_pilot_max_records")
    class_key = None
    variant = str(_get(payload, "base_policy.static.algorithm_variant") or "")
    if "class_fermion_boson" in variant:
        class_key = "fermion_boson"
    elif "class_bosonic" in variant:
        class_key = "bosonic"
    elif "class_fermionic" in variant:
        class_key = "fermionic"

    trials = payload.get("trials")
    if not isinstance(trials, Sequence) or isinstance(trials, (str, bytes)):
        return rows
    for trial in trials:
        if not isinstance(trial, Mapping) or str(trial.get("state") or "") != "COMPLETE":
            continue
        attrs = trial.get("user_attrs") if isinstance(trial.get("user_attrs"), Mapping) else {}
        candidates: list[Mapping[str, Any]] = []
        result = attrs.get("result") if isinstance(attrs, Mapping) else None
        if isinstance(result, Mapping):
            candidates.append(result)
        results = attrs.get("results") if isinstance(attrs, Mapping) else None
        if isinstance(results, Mapping):
            candidates.extend(value for value in results.values() if isinstance(value, Mapping))
        for result_payload in candidates:
            abs_delta = _as_float(
                result_payload.get("abs_delta_e_same_cutoff")
                if result_payload.get("abs_delta_e_same_cutoff") is not None
                else result_payload.get("cutoff_abs_delta_e")
                if result_payload.get("cutoff_abs_delta_e") is not None
                else result_payload.get("abs_delta_e")
            )
            rows.append(
                {
                    "record_id": record_id,
                    "benchmark_id": result_payload.get("benchmark_id"),
                    "family": result_payload.get("family"),
                    "class_key": class_key,
                    "source_kind": "optuna_summary",
                    "same_cutoff_abs_delta_e": abs_delta,
                    "meets_5e-5": None if abs_delta is None else bool(abs_delta <= 5e-5),
                    "meets_1e-6": None if abs_delta is None else bool(abs_delta <= 1e-6),
                    "meets_1e-8": None if abs_delta is None else bool(abs_delta <= 1e-8),
                    "compiled_two_qubit_count": result_payload.get("count_2q"),
                    "compiled_depth": result_payload.get("circuit_depth"),
                    "parameter_count": result_payload.get("parameter_count") or result_payload.get("runtime_parameter_count"),
                    "S_alg": result_payload.get("measurement_groups_proxy"),
                    "measurement_shots_proxy": result_payload.get("measurement_shots_proxy") or result_payload.get("shot_cost_proxy"),
                    "S_phys": None,
                    "phase0_pilot_max_records": phase0_max,
                    "summary_path": str(path),
                }
            )
    return rows


def collect_rows(roots: Sequence[str | Path], *, include_known_hh_baseline: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if include_known_hh_baseline:
        rows.append(
            {
                "record_id": "known_local_hh_l2_nph1_routeA_snake_baseline",
                "benchmark_id": "hh_L2",
                "family": "hh",
                "class_key": "fermion_boson",
                "source_kind": "setup_plan_known_local_baseline",
                "same_cutoff_abs_delta_e": 7.32e-5,
                "meets_5e-5": False,
                "meets_1e-6": False,
                "meets_1e-8": False,
                "compiled_two_qubit_count": 253,
                "compiled_depth": None,
                "parameter_count": None,
                "S_alg": None,
                "measurement_shots_proxy": None,
                "S_phys": None,
                "phase0_pilot_max_records": None,
                "summary_path": None,
            }
        )
    for raw in roots:
        root = Path(raw)
        if not root.exists():
            continue
        candidates = (root,) if root.is_file() else tuple(sorted(root.rglob("summary.json")))
        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(payload, Mapping):
                rows.extend(_result_rows_from_summary(path, payload))
    return rows


def scoreboard_payload(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "thresholds": [float(x) for x in THRESHOLDS],
        "row_count": len(rows),
        "rows": [dict(row) for row in rows],
    }


def rows_to_tsv(rows: Sequence[Mapping[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(FIELDS), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in FIELDS})
    return buf.getvalue()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-tsv", type=Path, default=None)
    parser.add_argument("--include-known-hh-baseline", action="store_true")
    args = parser.parse_args(argv)

    rows = collect_rows(args.root, include_known_hh_baseline=bool(args.include_known_hh_baseline))
    payload = scoreboard_payload(rows)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_tsv is not None:
        args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
        args.output_tsv.write_text(rows_to_tsv(rows), encoding="utf-8")
    if args.output_json is None and args.output_tsv is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
