#!/usr/bin/env python3
"""Generate class-canonical Route-A Phase0 nph=1 Optuna CHTC records."""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.table_i_canonical_cases import TABLE_I_STANDARD_PROFILE, table_i_executable_specs  # noqa: E402
from pipelines.static_adapt.route_identity import ROUTE_ID_A  # noqa: E402

from chtc.phase3_optuna.generate_routeA_phase0_nph1_oracle_records import (  # noqa: E402
    FIELDNAMES,
    TARGET_ABS_DELTA_E,
    _base_row,
    _ids_text,
    _tsv_text,
)

SCRIPT_DIR = Path(__file__).resolve().parent
BATCH_ID = "routeA_phase0_nph1_class_v1"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "input" / BATCH_ID
DEFAULT_ORACLE_SUMMARY_ROOT = "raw_outputs/routeA_phase0_nph1_oracle_v1"
FULL_RECORDS = "phase0_class_records.tsv"
FULL_IDS = "phase0_class_record_ids.txt"
SMOKE_RECORDS = "phase0_class_smoke_records.tsv"
SMOKE_IDS = "phase0_class_smoke_record_ids.txt"
MANIFEST = "phase0_class_manifest.json"
FULL_N_TRIALS = 48
SMOKE_N_TRIALS = 1

CLASS_DEFINITIONS = (
    {
        "key": "fermionic",
        "record_token": "fermionic",
        "canonical_lane": "fermionic",
        "families": ("hubbard", "ionic_hubbard", "extended_hubbard", "ttprime_hubbard", "spinless_tv"),
        "policy_search_profile": "fermionic_protected_correlation",
        "class_key": "fermionic",
    },
    {
        "key": "bosonic",
        "record_token": "bosonic",
        "canonical_lane": "bosonic",
        "families": ("bose_hubbard", "harmonic_kerr_chain", "spin_boson"),
        "policy_search_profile": "bosonic_fullmeta_compact",
        "class_key": "bosonic",
    },
    {
        "key": "fermion_boson_hh",
        "record_token": "fermion_boson_hh",
        "canonical_lane": "",
        "families": ("hh",),
        "policy_search_profile": "bosonic_fullmeta_compact",
        "class_key": "fermion_boson",
    },
)


def _benchmark_ids_for_families(families: Sequence[str]) -> tuple[str, ...]:
    family_set = set(families)
    return tuple(spec.benchmark_id for spec in table_i_executable_specs(TABLE_I_STANDARD_PROFILE) if spec.family in family_set)


def _row_for_class(defn: Mapping[str, Any], *, smoke: bool, oracle_summary_root: str) -> dict[str, str]:
    families = tuple(str(x) for x in defn["families"])
    benchmark_ids = _benchmark_ids_for_families(families)
    if not benchmark_ids:
        raise ValueError(f"class {defn['key']}: no benchmark IDs selected")
    record_id = f"routeA_phase0_nph1_class_{defn['record_token']}_{'smoke_' if smoke else ''}v1"
    row = _base_row(
        record_id=record_id,
        mode="global",
        families=families,
        benchmark_ids=benchmark_ids,
        n_trials=SMOKE_N_TRIALS if smoke else FULL_N_TRIALS,
        policy_search_profile=str(defn["policy_search_profile"]),
        canonical_lane=str(defn["canonical_lane"]),
        canonical_lane_stage="train",
        oracle_summary_root=str(oracle_summary_root),
        oracle_required_static_route_id=ROUTE_ID_A,
        oracle_required_suite_profile=TABLE_I_STANDARD_PROFILE,
        oracle_require_phase0_aware=True,
        oracle_require_compatible_warm_starts=True,
        algorithm_variant=f"A_current_collective_phase0_nph1_class_{defn['class_key']}",
        seed=90117 if smoke else 85117,
    )
    row["oracle_enqueue_limit"] = "8"
    row["required_target_benchmark_ids"] = " ".join(benchmark_ids)
    row["required_target_abs_delta_e"] = TARGET_ABS_DELTA_E
    return row


def build_rows(*, oracle_summary_root: str = DEFAULT_ORACLE_SUMMARY_ROOT) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    full_rows = [_row_for_class(defn, smoke=False, oracle_summary_root=oracle_summary_root) for defn in CLASS_DEFINITIONS]
    smoke_rows = [_row_for_class(defn, smoke=True, oracle_summary_root=oracle_summary_root) for defn in CLASS_DEFINITIONS]
    validate_rows(full_rows, smoke=False, oracle_summary_root=oracle_summary_root)
    validate_rows(smoke_rows, smoke=True, oracle_summary_root=oracle_summary_root)
    return full_rows, smoke_rows


def validate_rows(rows: Sequence[Mapping[str, str]], *, smoke: bool, oracle_summary_root: str) -> None:
    if len(rows) != len(CLASS_DEFINITIONS):
        raise ValueError(f"expected {len(CLASS_DEFINITIONS)} class rows")
    expected_trials = str(SMOKE_N_TRIALS if smoke else FULL_N_TRIALS)
    seen: set[str] = set()
    for row in rows:
        rid = str(row.get("record_id") or "")
        if rid in seen:
            raise ValueError(f"duplicate class record_id: {rid}")
        seen.add(rid)
        if "_mixed_" in rid or rid.startswith("mixed_"):
            raise ValueError(f"class record ID must not use legacy mixed marker: {rid}")
        if str(row.get("static_route_id") or "") != ROUTE_ID_A:
            raise ValueError(f"{rid}: static_route_id must be route_a")
        if str(row.get("suite_profile") or "") != TABLE_I_STANDARD_PROFILE:
            raise ValueError(f"{rid}: suite_profile must be standard")
        if str(row.get("oracle_summary_root") or "") != str(oracle_summary_root):
            raise ValueError(f"{rid}: oracle_summary_root mismatch")
        if str(row.get("oracle_required_static_route_id") or "") != ROUTE_ID_A:
            raise ValueError(f"{rid}: required oracle route guard missing")
        if str(row.get("oracle_required_suite_profile") or "") != TABLE_I_STANDARD_PROFILE:
            raise ValueError(f"{rid}: required oracle suite guard missing")
        if str(row.get("oracle_require_phase0_aware") or "") != "true":
            raise ValueError(f"{rid}: required phase0-aware guard missing")
        if str(row.get("oracle_require_compatible_warm_starts") or "") != "true":
            raise ValueError(f"{rid}: required compatible warm-start guard missing")
        if str(row.get("exact_reference_boson_cutoff") or "") != "0":
            raise ValueError(f"{rid}: exact_reference_boson_cutoff must be 0")
        if str(row.get("n_trials") or "") != expected_trials:
            raise ValueError(f"{rid}: n_trials must be {expected_trials}")
        if not str(row.get("benchmark_ids") or "").strip():
            raise ValueError(f"{rid}: benchmark_ids must be exact and non-empty")
        if "spin_boson" in str(row.get("families") or "") and str(row.get("canonical_lane") or "") != "bosonic":
            raise ValueError(f"{rid}: spin_boson must remain in the bosonic class row")
        if str(row.get("families") or "") == "hh" and str(row.get("canonical_lane") or ""):
            raise ValueError(f"{rid}: HH-only fermion-boson row must keep canonical_lane blank")


def _manifest_payload(
    full_rows: Sequence[Mapping[str, str]],
    smoke_rows: Sequence[Mapping[str, str]],
    *,
    oracle_summary_root: str,
) -> dict[str, Any]:
    return {
        "schema": "routeA_phase0_nph1_class_records_manifest_v1",
        "batch_id": BATCH_ID,
        "generated_by": "chtc/phase3_optuna/generate_routeA_phase0_nph1_class_records.py",
        "suite_profile": TABLE_I_STANDARD_PROFILE,
        "oracle_summary_root": str(oracle_summary_root),
        "record_ids": [row["record_id"] for row in full_rows],
        "smoke_record_ids": [row["record_id"] for row in smoke_rows],
        "constraints": {
            "working_n_ph_max": 1,
            "exact_reference_boson_cutoff": 0,
            "primary_energy_metric": "same_cutoff_abs_delta_e",
            "target_abs_delta_e": float(TARGET_ABS_DELTA_E),
            "static_route_id": ROUTE_ID_A,
            "phase0_is_route_identity": False,
            "oracle_required_static_route_id": ROUTE_ID_A,
            "oracle_required_suite_profile": TABLE_I_STANDARD_PROFILE,
            "oracle_require_phase0_aware": True,
            "oracle_require_compatible_warm_starts": True,
        },
        "classes": [
            {
                "key": str(defn["key"]),
                "class_key": str(defn["class_key"]),
                "canonical_lane": str(defn["canonical_lane"]),
                "families": list(defn["families"]),
                "benchmark_ids": list(_benchmark_ids_for_families(tuple(defn["families"]))),
            }
            for defn in CLASS_DEFINITIONS
        ],
    }


def render_artifacts(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    oracle_summary_root: str = DEFAULT_ORACLE_SUMMARY_ROOT,
) -> dict[Path, str]:
    output_dir = Path(output_dir)
    full_rows, smoke_rows = build_rows(oracle_summary_root=oracle_summary_root)
    manifest = _manifest_payload(full_rows, smoke_rows, oracle_summary_root=oracle_summary_root)
    return {
        output_dir / FULL_RECORDS: _tsv_text(full_rows),
        output_dir / FULL_IDS: _ids_text(full_rows),
        output_dir / SMOKE_RECORDS: _tsv_text(smoke_rows),
        output_dir / SMOKE_IDS: _ids_text(smoke_rows),
        output_dir / MANIFEST: json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    }


def write_artifacts(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    oracle_summary_root: str = DEFAULT_ORACLE_SUMMARY_ROOT,
) -> dict[str, str]:
    artifacts = render_artifacts(output_dir, oracle_summary_root=oracle_summary_root)
    for path, text in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return {path.name: str(path) for path in artifacts}


def check_artifacts(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    oracle_summary_root: str = DEFAULT_ORACLE_SUMMARY_ROOT,
) -> list[str]:
    errors: list[str] = []
    for path, expected in render_artifacts(output_dir, oracle_summary_root=oracle_summary_root).items():
        try:
            actual = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            errors.append(f"missing generated artifact: {path}")
            continue
        if actual != expected:
            errors.append(f"generated artifact is stale: {path}")
    return errors


def generate_records(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    oracle_summary_root: str = DEFAULT_ORACLE_SUMMARY_ROOT,
    write: bool = False,
) -> dict[str, Any]:
    artifacts = (
        write_artifacts(output_dir, oracle_summary_root=oracle_summary_root)
        if write
        else {path.name: str(path) for path in render_artifacts(output_dir, oracle_summary_root=oracle_summary_root)}
    )
    full_rows, smoke_rows = build_rows(oracle_summary_root=oracle_summary_root)
    return {
        "schema": "routeA_phase0_nph1_class_generation_summary_v1",
        "batch_id": BATCH_ID,
        "oracle_summary_root": str(oracle_summary_root),
        "full_record_ids": [row["record_id"] for row in full_rows],
        "smoke_record_ids": [row["record_id"] for row in smoke_rows],
        "paths": artifacts,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--oracle-summary-root", default=DEFAULT_ORACLE_SUMMARY_ROOT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    if args.check:
        errors = check_artifacts(args.output_dir, oracle_summary_root=str(args.oracle_summary_root))
        if errors:
            for error in errors:
                print(error)
            return 1
        print("Route-A Phase0 nph1 class artifacts are current")
        return 0
    if args.write:
        paths = write_artifacts(args.output_dir, oracle_summary_root=str(args.oracle_summary_root))
        print(f"wrote {len(paths)} Route-A Phase0 nph1 class artifacts")
        return 0
    print(
        json.dumps(
            generate_records(args.output_dir, oracle_summary_root=str(args.oracle_summary_root), write=False),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
