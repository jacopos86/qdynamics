#!/usr/bin/env python3
"""Generate Paper-I HH native-forced child/no-child comparison records.

This is the page-1 counterpart to the monotone/non-forced page-2 grid: six HH
regimes, Append/Geo/SNAKE, and explicit child/no-child route settings. SNAKE
rows do not request Schur warm-start refits here; this preserves the old-route
comparison axis.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chtc.phase3_optuna.generate_paper_i_hh_spsa_budget_ladder_records import (
    FIELDNAMES,
    NATIVE_ENGINE,
    configure_batch,
    output_paths,
    rel_or_abs,
)
from chtc.phase3_optuna.generate_paper_i_hh_native200_depth30_records import (
    APPEND_GEO_RECORDS,
    METHOD_ORDER,
    METHOD_TO_KEY,
    REGIME_ORDER,
    SNAKE_RECORDS,
    SUPPORT_JSON,
    manifest_path_for_source,
    read_json,
    read_records,
    support_rows,
    write_submit_file,
)


DEFAULT_BATCH_ID = "paper_i_hh_native_forced_child_matrix_depth30_20260623_v1"
CHILD_MODES = ("no_child", "polychildren")
CHILD_SPLIT_MODE = "shortlist_pauli_children_v1"
CHILD_SYMMETRY_POLICY = "off"
SNAKE_CHILD_POOL_MODE = "global_pauli_child_sets_v1"
SNAKE_CHILD_POOL_SYMMETRY_POLICY = "hard_guard"
GENERIC_STOP_POLICY = "fixed_horizon_no_target_v1"


def write_lines(path: Path, rows: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")


def source_record_index(paths: Sequence[Path]) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for path in paths:
        for row in read_records(path):
            if row.get("engine_key") != "native_forced" or row.get("budget") != "200":
                continue
            method_key = str(row.get("method_key"))
            if method_key not in {"append", "geo", "snake"}:
                continue
            out[(str(row.get("display_regime")), method_key)] = row
    return out


def child_enabled(child_mode: str) -> bool:
    if child_mode not in CHILD_MODES:
        raise ValueError(f"child_mode must be one of {CHILD_MODES}; got {child_mode!r}")
    return child_mode == "polychildren"


def changed_fields(method_key: str, child_mode: str) -> list[str]:
    fields = [
        "max_depth",
        "adapt_segment_target_depth",
        "adapt_segment_max_new_admissions",
        "disable_drop_stop",
        "disable_benchmark_target_stop",
    ]
    if method_key in {"append", "geo"}:
        fields.extend(["generic_adapt_stop_policy", "target_abs_delta_e"])
        if child_enabled(child_mode):
            fields.extend(
                [
                    "generic_adapt_runtime_split_mode",
                    "generic_adapt_runtime_split_symmetry_policy",
                    "resource_pool_term_cap",
                ]
            )
    else:
        fields.extend(
            [
                "snake_phase3_runtime_split_mode",
                "snake_adapt_child_pool_expansion_mode",
                "snake_adapt_child_pool_expansion_symmetry_policy",
                "snake_adapt_child_pool_expansion_max_subset_size",
            ]
        )
    return fields


def make_row(
    *,
    batch_id: str,
    regime: str,
    method: str,
    child_mode: str,
    anchor: Mapping[str, Any],
    source: Mapping[str, str],
) -> dict[str, str]:
    method_key = METHOD_TO_KEY[method]
    child_on = child_enabled(child_mode)
    record_id = (
        f"{batch_id}__{regime.replace('-', '_')}"
        f"__{method_key}__native_forced__maxiter200__depth30_noearlystop__{child_mode}"
    )
    row = dict(source)
    row.update(
        {
            "record_id": record_id,
            "batch_id": batch_id,
            "run_class": "diagnostic",
            "runnable": "true" if str(source.get("runnable") or "") == "true" else "false",
            "blocker": str(source.get("blocker") or ""),
            "engine_key": "native_forced",
            "engine_label": "native forced full budget",
            "spsa_refit_engine": NATIVE_ENGINE,
            "budget": "200",
            "max_depth": "30",
            "source_settings_status": "ok_native_forced_child_matrix_grid",
            "schedule_source_policy": "paper_i_native_forced_child_matrix_grid",
            "schedule_source_regime": regime,
            "schedule_source_method": method,
            "schedule_source_json": str(source.get("source_json") or anchor.get("source_json") or ""),
            "schedule_source_note": (
                "native-forced depth30 comparison grid; fixed-horizon execution; "
                f"child_mode={child_mode}"
            ),
            "anchor_source_json": str(anchor.get("source_json") or ""),
            "anchor_source_sha256": str(anchor.get("source_sha256") or ""),
            "anchor_cell_manifest_rel": manifest_path_for_source(str(anchor.get("source_json") or "")),
            "changed_fields_vs_anchor": ",".join(changed_fields(method_key, child_mode)),
            "source_contract_note": (
                "Comparative diagnostic grid: native-forced SPSA maxiter=200, depth30/no-target "
                "execution, and child/no-child route settings. Append/Geo child rows use generic "
                "child-set pool expansion; SNAKE child rows use global pre-Phase-1 child-set pool "
                "expansion and do not use Schur warm-start refits."
            ),
        }
    )
    if method_key in {"append", "geo"}:
        row["source_json"] = str(anchor.get("source_json") or row.get("source_json") or "")
        row["source_json_sha256"] = str(anchor.get("source_sha256") or row.get("source_json_sha256") or "")
        row["generic_adapt_runtime_split_mode"] = CHILD_SPLIT_MODE if child_on else "off"
        row["generic_adapt_runtime_split_symmetry_policy"] = CHILD_SYMMETRY_POLICY
        row["generic_adapt_runtime_split_max_subset_size"] = ""
        row["generic_adapt_stop_policy"] = GENERIC_STOP_POLICY
        row["resource_qubit_cap"] = ""
        row["resource_pool_term_cap"] = "0" if child_on else ""
        row["adapt_schur_warm_start_mode"] = ""
        row["snake_phase3_runtime_split_mode"] = ""
        row["snake_adapt_child_pool_expansion_mode"] = ""
        row["snake_adapt_child_pool_expansion_symmetry_policy"] = ""
        row["snake_adapt_child_pool_expansion_max_subset_size"] = ""
    else:
        row["source_json"] = str(source.get("source_json") or "")
        row["source_json_sha256"] = str(source.get("source_json_sha256") or "")
        row["generic_adapt_runtime_split_mode"] = ""
        row["generic_adapt_runtime_split_symmetry_policy"] = ""
        row["generic_adapt_runtime_split_max_subset_size"] = ""
        row["generic_adapt_stop_policy"] = ""
        row["resource_qubit_cap"] = ""
        row["resource_pool_term_cap"] = ""
        row["adapt_schur_warm_start_mode"] = ""
        row["snake_phase3_runtime_split_mode"] = "off"
        row["snake_adapt_child_pool_expansion_mode"] = SNAKE_CHILD_POOL_MODE if child_on else "off"
        row["snake_adapt_child_pool_expansion_symmetry_policy"] = (
            SNAKE_CHILD_POOL_SYMMETRY_POLICY if child_on else ""
        )
        row["snake_adapt_child_pool_expansion_max_subset_size"] = "3" if child_on else ""
    row.update(output_paths(record_id, method_key))
    for field in FIELDNAMES:
        row.setdefault(field, "")
    return row


def build_records(batch_id: str) -> list[dict[str, str]]:
    anchors = support_rows(SUPPORT_JSON)
    sources = source_record_index([APPEND_GEO_RECORDS, SNAKE_RECORDS])
    rows: list[dict[str, str]] = []
    missing: dict[str, list[str]] = defaultdict(list)
    for regime in REGIME_ORDER:
        for method in METHOD_ORDER:
            method_key = METHOD_TO_KEY[method]
            anchor = anchors.get((regime, method))
            source = sources.get((regime, method_key))
            if anchor is None:
                missing["anchor"].append(f"{regime}/{method}")
                continue
            if source is None:
                missing["source_record"].append(f"{regime}/{method_key}")
                continue
            for child_mode in CHILD_MODES:
                rows.append(
                    make_row(
                        batch_id=batch_id,
                        regime=regime,
                        method=method,
                        child_mode=child_mode,
                        anchor=anchor,
                        source=source,
                    )
                )
    if missing:
        raise SystemExit(json.dumps({"status": "blocked_missing_sources", "missing": missing}, indent=2))
    return rows


def write_records(batch_id: str, records: Sequence[dict[str, str]]) -> dict[str, Any]:
    input_dir = ROOT / "chtc" / "phase3_optuna" / "input" / batch_id
    records_tsv = input_dir / "paper_i_hh_spsa_budget_ladder_records.tsv"
    record_ids = input_dir / "paper_i_hh_spsa_budget_ladder_record_ids.txt"
    smoke_ids = input_dir / "paper_i_hh_spsa_budget_ladder_smoke_record_ids.txt"
    manifest_json = input_dir / "paper_i_hh_spsa_budget_ladder_manifest.json"
    submit_path = ROOT / "chtc" / "phase3_optuna" / f"submit_{batch_id}.sub"
    input_dir.mkdir(parents=True, exist_ok=True)
    with records_tsv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FIELDNAMES), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow(row)
    write_lines(record_ids, (row["record_id"] for row in records))
    write_lines(smoke_ids, (records[0]["record_id"],))
    by_method: dict[str, list[str]] = defaultdict(list)
    for row in records:
        by_method[str(row["method_key"])].append(row["record_id"])
    for method_key, ids in sorted(by_method.items()):
        write_lines(input_dir / f"paper_i_hh_native_forced_child_matrix_{method_key}_record_ids.txt", ids)
    manifest = {
        "schema": "paper_i_hh_native_forced_child_matrix_depth30_manifest_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch_id,
        "run_class": "diagnostic",
        "source_contract": {
            "support_json": rel_or_abs(SUPPORT_JSON),
            "support_schema": read_json(SUPPORT_JSON).get("schema"),
            "comparative_grid_note": (
                "Native-forced SPSA page-1 child/no-child Cartesian grid. "
                "SNAKE uses the old route without Schur warm-start refits."
            ),
            "preserved": {
                "spsa_engine": "native_forced",
                "spsa_refit_engine": NATIVE_ENGINE,
                "maxiter": 200,
                "methods": list(METHOD_ORDER),
                "regimes": list(REGIME_ORDER),
            },
            "append_geo_child_policy": {
                "no_child": {"generic_adapt_runtime_split_mode": "off"},
                "polychildren": {
                    "generic_adapt_runtime_split_mode": CHILD_SPLIT_MODE,
                    "generic_adapt_runtime_split_symmetry_policy": CHILD_SYMMETRY_POLICY,
                    "resource_pool_term_cap": 0,
                },
                "generic_adapt_stop_policy": GENERIC_STOP_POLICY,
            },
            "snake_child_policy": {
                "no_child": {"snake_phase3_runtime_split_mode": "off"},
                "polychildren": {
                    "snake_phase3_runtime_split_mode": "off",
                    "snake_adapt_child_pool_expansion_mode": SNAKE_CHILD_POOL_MODE,
                    "snake_adapt_child_pool_expansion_symmetry_policy": SNAKE_CHILD_POOL_SYMMETRY_POLICY,
                    "snake_adapt_child_pool_expansion_max_subset_size": 3,
                },
                "adapt_schur_warm_start_mode": "off",
            },
            "max_depth": 30,
        },
        "paths": {
            "records_tsv": rel_or_abs(records_tsv),
            "record_ids_txt": rel_or_abs(record_ids),
            "smoke_record_ids_txt": rel_or_abs(smoke_ids),
            "submit_file": rel_or_abs(submit_path),
        },
        "record_count": len(records),
        "intended_grid": {
            "regime_count": len(REGIME_ORDER),
            "method_count": len(METHOD_ORDER),
            "child_modes": list(CHILD_MODES),
            "engine_keys": ["native_forced"],
            "budgets": [200],
            "max_depth": 30,
            "intended_record_count": len(REGIME_ORDER) * len(METHOD_ORDER) * len(CHILD_MODES),
        },
        "records": [
            {
                "record_id": row["record_id"],
                "display_regime": row["display_regime"],
                "method_key": row["method_key"],
                "engine_key": row["engine_key"],
                "child_mode": "polychildren"
                if (
                    row.get("generic_adapt_runtime_split_mode") == CHILD_SPLIT_MODE
                    or row.get("snake_adapt_child_pool_expansion_mode") == SNAKE_CHILD_POOL_MODE
                )
                else "no_child",
                "snake_phase3_runtime_split_mode": row.get("snake_phase3_runtime_split_mode"),
                "snake_adapt_child_pool_expansion_mode": row.get("snake_adapt_child_pool_expansion_mode"),
                "budget": row["budget"],
                "max_depth": row["max_depth"],
            }
            for row in records
        ],
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_submit_file(batch_id=batch_id, submit_path=submit_path, records_tsv=records_tsv, record_ids=record_ids)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    batch_id = str(args.batch_id)
    configure_batch(batch_id)
    records = build_records(batch_id)
    manifest = write_records(batch_id, records)
    print(json.dumps({k: v for k, v in manifest.items() if k != "records"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
