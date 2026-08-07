#!/usr/bin/env python3
"""Preflight and fetched-output checks for the L=2/nph=1 tripartite reset batch."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_RECORDS = REPO_ROOT / "chtc/phase3_optuna/input/global_tripartite_spsa_ab_l2nph1_reset_smoke_records.tsv"


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        return [dict(row) for row in reader if row.get("record_id")]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _split_words(value: str | None) -> list[str]:
    value = str(value or "").strip()
    return [] if not value else value.split()


def _pipeline_arg_value(args: Sequence[Any], flag: str) -> str | None:
    tokens = [str(x) for x in args]
    for idx, token in enumerate(tokens):
        if token == flag and idx + 1 < len(tokens):
            return tokens[idx + 1]
        prefix = f"{flag}="
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def _row_constraint_errors(row: Mapping[str, str]) -> list[str]:
    rid = str(row.get("record_id") or "")
    errors: list[str] = []
    if "fermionic" in rid:
        errors.append("record_id unexpectedly names a fermionic lane")
    if str(row.get("canonical_lane") or "").strip():
        errors.append("canonical_lane must be blank; reset records use explicit families")
    family = str(row.get("families") or "").strip()
    if family not in {"bose_hubbard", "hh"}:
        errors.append(f"families must be bose_hubbard or hh, got {family!r}")
    if str(row.get("sizes") or "").strip() != "2":
        errors.append("sizes must be exactly 2")
    if str(row.get("boson_cutoff") or "").strip() != "1":
        errors.append("boson_cutoff must be exactly 1")
    if str(row.get("boson_cutoffs") or "").strip():
        errors.append("boson_cutoffs must be blank")
    if str(row.get("exact_reference_boson_cutoff") or "").strip() != "0":
        errors.append("exact_reference_boson_cutoff must be 0")
    if str(row.get("fixed_inner_optimizer") or "").strip().upper() != "SPSA":
        errors.append("fixed_inner_optimizer must be SPSA")
    novelty = str(row.get("phase2_novelty_mode") or "").strip()
    if novelty not in {"collective_span_v1", "legacy_pairwise_v1"}:
        errors.append("phase2_novelty_mode must be collective_span_v1 or legacy_pairwise_v1")
    return [f"{rid}: {error}" for error in errors]


def _preflight_row(row: Mapping[str, str]) -> tuple[list[str], list[dict[str, Any]]]:
    from pipelines.static_adapt.optimization.phase3_policy_optuna import filter_static_benchmark_suite

    rid = str(row.get("record_id") or "")
    errors: list[str] = []
    specs_payload: list[dict[str, Any]] = []
    families = _split_words(row.get("families"))
    sizes = [int(x) for x in _split_words(row.get("sizes"))]
    boson_cutoff = int(str(row.get("boson_cutoff") or "1"))
    cutoffs = [int(x) for x in _split_words(row.get("boson_cutoffs"))]
    exact_ref = int(str(row.get("exact_reference_boson_cutoff") or "0"))
    specs = filter_static_benchmark_suite(
        families=families,
        sizes=sizes,
        boson_cutoff=None if cutoffs else boson_cutoff,
        boson_cutoffs=cutoffs or None,
        exact_reference_boson_cutoff=None if exact_ref <= 0 else exact_ref,
        physics_grid_profile=str(row.get("physics_grid_profile") or "canonical"),
    )
    if not specs:
        errors.append(f"{rid}: preflight selected no benchmark specs")
    for spec in specs:
        args = tuple(str(x) for x in spec.base_pipeline_args)
        nph_value = _pipeline_arg_value(args, "--n-ph-max")
        specs_payload.append(
            {
                "benchmark_id": spec.benchmark_id,
                "family": spec.family,
                "L": int(spec.features.L),
                "n_ph_max": None if nph_value is None else int(nph_value),
                "exact_reference_n_ph_max": spec.exact_reference_n_ph_max,
            }
        )
        if int(spec.features.L) == 3:
            errors.append(f"{rid}: preflight selected forbidden L=3 benchmark {spec.benchmark_id}")
        if nph_value is None:
            errors.append(f"{rid}: preflight benchmark {spec.benchmark_id} is missing --n-ph-max")
        elif int(nph_value) != 1:
            errors.append(f"{rid}: preflight selected forbidden n_ph_max={nph_value} for {spec.benchmark_id}")
        if spec.exact_reference_n_ph_max is not None:
            errors.append(
                f"{rid}: preflight selected high-cutoff exact reference n_ph_max={spec.exact_reference_n_ph_max} for {spec.benchmark_id}"
            )
    return errors, specs_payload


def _summary_errors(record_id: str, summary_path: Path) -> list[str]:
    errors: list[str] = []
    payload = _load_json(summary_path)
    benchmarks = payload.get("benchmarks", []) if isinstance(payload, Mapping) else []
    if not isinstance(benchmarks, Sequence) or isinstance(benchmarks, (str, bytes)):
        return [f"{record_id}: summary benchmarks field is not a list"]
    for idx, bench in enumerate(benchmarks):
        if not isinstance(bench, Mapping):
            errors.append(f"{record_id}: summary benchmark #{idx} is not an object")
            continue
        bench_id = str(bench.get("benchmark_id") or f"benchmark_{idx}")
        features = bench.get("features") if isinstance(bench.get("features"), Mapping) else {}
        L = features.get("L") if isinstance(features, Mapping) else bench.get("L")
        if L is not None and int(L) == 3:
            errors.append(f"{record_id}: summary selected forbidden L=3 benchmark {bench_id}")
        args = bench.get("base_pipeline_args") or ()
        nph = _pipeline_arg_value(args, "--n-ph-max") if isinstance(args, Sequence) and not isinstance(args, (str, bytes)) else None
        if nph is None:
            errors.append(f"{record_id}: summary benchmark {bench_id} is missing --n-ph-max")
        elif int(nph) != 1:
            errors.append(f"{record_id}: summary selected forbidden n_ph_max={nph} for {bench_id}")
        exact_ref = bench.get("exact_reference_n_ph_max")
        if exact_ref is not None:
            errors.append(f"{record_id}: summary benchmark {bench_id} has exact_reference_n_ph_max={exact_ref}")
    return errors


def _output_errors(row: Mapping[str, str], output_root: Path, *, require_complete: bool) -> tuple[list[str], list[str]]:
    rid = str(row.get("record_id") or "")
    record_dir = output_root / rid
    errors: list[str] = []
    warnings: list[str] = []
    if not record_dir.exists():
        message = f"{rid}: output directory is missing: {record_dir}"
        (errors if require_complete else warnings).append(message)
        return errors, warnings

    for rel in ("record_manifest.json", "heartbeat.json", "progress/trial_events.jsonl", "command.sh"):
        path = record_dir / rel
        if not path.exists():
            errors.append(f"{rid}: missing output artifact {rel}")
    for rel in ("record_manifest.json", "heartbeat.json"):
        path = record_dir / rel
        if path.exists():
            try:
                _load_json(path)
            except Exception as exc:
                errors.append(f"{rid}: failed to parse {rel}: {type(exc).__name__}: {exc}")
    command_path = record_dir / "command.sh"
    if command_path.exists():
        command = command_path.read_text(encoding="utf-8", errors="replace")
        if "--fixed-phase2-novelty-mode" not in command:
            errors.append(f"{rid}: command.sh does not include --fixed-phase2-novelty-mode")
        if "--sizes 2" not in command and "--sizes '2'" not in command:
            errors.append(f"{rid}: command.sh does not visibly restrict --sizes 2")
        if "--boson-cutoff 1" not in command and "--boson-cutoff '1'" not in command:
            errors.append(f"{rid}: command.sh does not visibly restrict --boson-cutoff 1")
    summary_path = record_dir / "summary.json"
    if summary_path.exists():
        try:
            errors.extend(_summary_errors(rid, summary_path))
        except Exception as exc:
            errors.append(f"{rid}: failed to parse summary.json: {type(exc).__name__}: {exc}")
    elif require_complete:
        errors.append(f"{rid}: summary.json is required but missing")
    else:
        warnings.append(f"{rid}: summary.json missing; treating as partial output")
    if require_complete and not (record_dir / "study.sqlite3").exists() and not (record_dir / "progress/study_snapshot.sqlite3").exists():
        errors.append(f"{rid}: neither study.sqlite3 nor progress/study_snapshot.sqlite3 exists")
    run_err = record_dir / "run.err"
    if run_err.exists():
        text = run_err.read_text(encoding="utf-8", errors="replace")
        bad_needles = (
            "unrecognized arguments",
            "fixed_phase2_novelty_mode does not match",
            "No module named",
            "ImportError",
        )
        for needle in bad_needles:
            if needle in text:
                errors.append(f"{rid}: run.err contains {needle!r}")
    return errors, warnings


def check_records(records_path: Path, output_root: Path, *, require_complete: bool, preflight_only: bool) -> dict[str, Any]:
    rows = _read_tsv(records_path)
    errors: list[str] = []
    warnings: list[str] = []
    preflight_specs: dict[str, list[dict[str, Any]]] = {}
    if len(rows) != 4:
        errors.append(f"expected exactly four reset records in {records_path}, got {len(rows)}")
    ids = [str(row.get("record_id") or "") for row in rows]
    if len(set(ids)) != len(ids):
        errors.append(f"duplicate record IDs in {records_path}: {ids}")
    for row in rows:
        rid = str(row.get("record_id") or "")
        errors.extend(_row_constraint_errors(row))
        row_errors, specs_payload = _preflight_row(row)
        errors.extend(row_errors)
        preflight_specs[rid] = specs_payload
        if not preflight_only:
            out_errors, out_warnings = _output_errors(row, output_root, require_complete=require_complete)
            errors.extend(out_errors)
            warnings.extend(out_warnings)
    return {
        "schema": "phase3_tripartite_spsa_ab_l2nph1_reset_check_v1",
        "records_path": str(records_path),
        "output_root": str(output_root),
        "record_count": len(rows),
        "preflight_specs": preflight_specs,
        "warnings": warnings,
        "errors": errors,
        "ok": not errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output-root", type=Path, default=Path("raw_outputs"))
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    payload = check_records(
        args.records,
        args.output_root,
        require_complete=bool(args.require_complete),
        preflight_only=bool(args.preflight_only),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
