#!/usr/bin/env python3
"""Collect Paper-II HH static seed regeneration outputs into a source registry."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chtc.generic_time_dynamics_table.paper_ii_seed_track_common import (  # noqa: E402
    HH_REGIMES_BY_ID,
    ROOT as COMMON_ROOT,
    SEED_TRACKS_BY_ID,
    SOURCE_REGISTRY_SCHEMA,
    SeedTrackValidationError,
    coerce_hh_seed_payload,
    dry_load_hh_seed_payload,
    read_json,
    repo_path,
    sha256_file,
    source_parameter_count,
    source_static_error,
    validate_hh_seed_source,
    write_json,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_records(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh, delimiter="\t")]


def _read_sources(path: Path | None, *, root: Path) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = read_json(path, root=root)
    if payload.get("schema") != SOURCE_REGISTRY_SCHEMA:
        raise SeedTrackValidationError(
            f"previous registry schema must be {SOURCE_REGISTRY_SCHEMA!r}; got {payload.get('schema')!r}"
        )
    sources = payload.get("sources", [])
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        raise SeedTrackValidationError("previous registry sources must be a list")
    return [dict(item) for item in sources if isinstance(item, Mapping)]


def _repo_rel(path: Path, *, root: Path) -> str:
    resolved = path if path.is_absolute() else root / path
    try:
        return str(resolved.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _candidate_paths(record: Mapping[str, str], *, root: Path) -> list[Path]:
    output_root = repo_path(str(record.get("output_root", "")), root=root)
    if not output_root.exists():
        return []
    kind = str(record.get("kind", "")).strip()
    names = ("result.json", "generic_static_single.json") if kind == "append_powell" else ("result.json",)
    out: list[Path] = []
    for name in names:
        out.extend(sorted(path for path in output_root.rglob(name) if path.is_file()))
    return out


def _source_entry_for_record(record: Mapping[str, str], *, artifact_json: str) -> dict[str, Any]:
    regime_id = str(record.get("hh_regime_id", "")).strip()
    seed_track = str(record.get("seed_track", "")).strip()
    if regime_id not in HH_REGIMES_BY_ID:
        raise SeedTrackValidationError(f"unknown HH regime id in regen record: {regime_id!r}")
    if seed_track not in SEED_TRACKS_BY_ID:
        raise SeedTrackValidationError(f"unknown seed track in regen record: {seed_track!r}")
    track = SEED_TRACKS_BY_ID[seed_track]
    return {
        "hh_regime_id": regime_id,
        "hh_static_case_id": str(record.get("hh_static_case_id") or HH_REGIMES_BY_ID[regime_id].static_case_id),
        "seed_track": seed_track,
        "static_algorithm_id": track.required_static_algorithm_id,
        "seed_selection_policy": track.seed_selection_policy,
        "source_artifact_json": artifact_json,
        "source_record_id": str(record.get("record_id", "")),
    }


def _evaluate_candidate(
    *,
    path: Path,
    record: Mapping[str, str],
    root: Path,
    hard_threshold: float,
    require_runtime_dry_load: bool,
) -> dict[str, Any]:
    artifact_json = _repo_rel(path, root=root)
    source_entry = _source_entry_for_record(record, artifact_json=artifact_json)
    try:
        raw = read_json(artifact_json, root=root)
        payload = coerce_hh_seed_payload(raw, source_entry=source_entry, source_path=artifact_json)
        validation = validate_hh_seed_source(payload, source_entry=source_entry, source_path=artifact_json)
        static_error = source_static_error(payload)
        if static_error is None:
            raise SeedTrackValidationError("candidate lacks static_abs_delta_e/abs_delta_e")
        if float(static_error) > float(hard_threshold):
            raise SeedTrackValidationError(
                f"candidate static error {float(static_error):.6g} exceeds threshold {float(hard_threshold):.6g}"
            )
        runtime = (
            dry_load_hh_seed_payload(payload, artifact_json=artifact_json)
            if require_runtime_dry_load
            else {"runtime_loadability_status": "not_run"}
        )
        source_entry.update(
            {
                "source_artifact_sha256": sha256_file(artifact_json, root=root),
                "static_abs_delta_e": float(static_error),
                "static_parameter_count": source_parameter_count(payload),
                "static_error_metric_source": "source_static_error",
                "regeneration_record_id": str(record.get("record_id", "")),
                "regeneration_output_root": str(record.get("output_root", "")),
            }
        )
        return {
            "accepted": True,
            "candidate_artifact_json": artifact_json,
            "source_entry": source_entry,
            "validation_summary": validation,
            "runtime_loadability": runtime,
            "static_abs_delta_e": float(static_error),
        }
    except Exception as exc:
        return {
            "accepted": False,
            "candidate_artifact_json": artifact_json,
            "reason": "validation_failed",
            "validation_error_type": type(exc).__name__,
            "validation_error": str(exc),
        }


def collect_outputs(
    *,
    records_tsv: Path,
    output_dir: Path,
    previous_accepted_registry: Path | None = None,
    root: Path = COMMON_ROOT,
    hard_threshold: float = 1.0e-4,
    require_runtime_dry_load: bool = True,
) -> dict[str, Any]:
    records = _read_records(records_tsv)
    accepted_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for source in _read_sources(previous_accepted_registry, root=root):
        key = (str(source.get("hh_regime_id", "")), str(source.get("seed_track", "")))
        if all(key):
            accepted_by_key[key] = source

    evaluations: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    missing_records: list[dict[str, Any]] = []
    for record in records:
        candidates = _candidate_paths(record, root=root)
        if not candidates:
            missing_records.append(
                {
                    "record_id": record.get("record_id"),
                    "hh_regime_id": record.get("hh_regime_id"),
                    "seed_track": record.get("seed_track"),
                    "output_root": record.get("output_root"),
                    "reason": "no_candidate_result_json_found",
                }
            )
            continue
        record_evals = [
            _evaluate_candidate(
                path=candidate,
                record=record,
                root=root,
                hard_threshold=float(hard_threshold),
                require_runtime_dry_load=bool(require_runtime_dry_load),
            )
            for candidate in candidates
        ]
        evaluations.extend(record_evals)
        accepted = [item for item in record_evals if item.get("accepted")]
        rejected.extend(item for item in record_evals if not item.get("accepted"))
        if not accepted:
            continue
        best = min(accepted, key=lambda item: float(item.get("static_abs_delta_e", float("inf"))))
        entry = dict(best["source_entry"])
        key = (str(entry.get("hh_regime_id", "")), str(entry.get("seed_track", "")))
        previous = accepted_by_key.get(key)
        if previous is None or float(entry.get("static_abs_delta_e", float("inf"))) <= float(
            previous.get("static_abs_delta_e", float("inf"))
        ):
            accepted_by_key[key] = entry
        selected.append(best)

    output_dir = repo_path(output_dir, root=root)
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted_registry_rel = Path(_repo_rel(output_dir / "hh_local_accepted_seed_sources.json", root=root))
    rejected_rel = Path(_repo_rel(output_dir / "rejected_candidates.json", root=root))
    summary_rel = Path(_repo_rel(output_dir / "summary.json", root=root))
    accepted_registry = {
        "schema": SOURCE_REGISTRY_SCHEMA,
        "generated_utc": _utc_now(),
        "previous_accepted_registry": str(previous_accepted_registry) if previous_accepted_registry else None,
        "records_tsv": str(records_tsv),
        "sources": [accepted_by_key[key] for key in sorted(accepted_by_key)],
    }
    rejected_payload = {
        "schema": "paper_ii_hh_static_seed_regen_rejected_candidates_v1",
        "generated_utc": _utc_now(),
        "records_tsv": str(records_tsv),
        "missing_records": missing_records,
        "rejected_candidates": rejected,
        "all_candidate_evaluations": evaluations,
    }
    summary = {
        "schema": "paper_ii_hh_static_seed_regen_collection_summary_v1",
        "generated_utc": _utc_now(),
        "records_tsv": str(records_tsv),
        "previous_accepted_registry": str(previous_accepted_registry) if previous_accepted_registry else None,
        "accepted_registry": str(accepted_registry_rel),
        "rejected_candidates": str(rejected_rel),
        "record_count": len(records),
        "selected_regenerated_count": len(selected),
        "accepted_source_count": len(accepted_registry["sources"]),
        "missing_record_count": len(missing_records),
        "rejected_candidate_count": len(rejected),
        "hard_threshold": float(hard_threshold),
        "require_runtime_dry_load": bool(require_runtime_dry_load),
    }
    write_json(accepted_registry_rel, accepted_registry, root=root)
    write_json(rejected_rel, rejected_payload, root=root)
    write_json(summary_rel, summary, root=root)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-tsv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--previous-accepted-registry", type=Path, default=None)
    parser.add_argument("--hard-threshold", type=float, default=1.0e-4)
    parser.add_argument("--no-runtime-dry-load", action="store_true")
    args = parser.parse_args(argv)
    summary = collect_outputs(
        records_tsv=args.records_tsv,
        output_dir=args.output_dir,
        previous_accepted_registry=args.previous_accepted_registry,
        hard_threshold=float(args.hard_threshold),
        require_runtime_dry_load=not bool(args.no_runtime_dry_load),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
