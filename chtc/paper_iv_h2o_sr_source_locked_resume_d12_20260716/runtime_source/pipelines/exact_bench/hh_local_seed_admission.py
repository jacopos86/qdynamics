#!/usr/bin/env python3
"""Local Paper-II HH static-seed admission gate.

This script is intentionally local-only.  It audits candidate static SNAKE and
append-only ADAPT seeds before they can be used to build Paper-II dynamics case
manifests.  It does not launch heavy static jobs and does not edit manuscript
files.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import shlex
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chtc.generic_time_dynamics_table.paper_ii_seed_track_common import (
    HH_LOCAL_PAPER_II_REGIMES,
    ROOT,
    SEED_TRACKS_BY_ID,
    SOURCE_REGISTRY_SCHEMA,
    SeedTrackValidationError,
    coerce_hh_seed_payload,
    dry_load_hh_seed_payload,
    read_json,
    repo_path,
    sha256_file,
    validate_hh_seed_source,
    write_json,
)

DEFAULT_SOURCE_REGISTRY = Path("chtc/generic_time_dynamics_table/input/paper_ii_hh_static_seed_sources_v1.json")
DEFAULT_OUTPUT_ROOT = Path("raw_outputs/paper_ii_hh_local_seed_admission_20260626")
DEFAULT_REGIME_IDS: tuple[str, ...] = tuple(regime.regime_id for regime in HH_LOCAL_PAPER_II_REGIMES)
DEFAULT_SEED_TRACKS: tuple[str, ...] = ("snake", "append")

SAME_CUTOFF_ERROR_KEYS: tuple[str, ...] = (
    "same_cutoff_abs_delta_e",
    "abs_delta_e_same_cutoff",
    "static_abs_delta_e_same_cutoff",
    "same_cutoff_delta_E_abs",
    "same_cutoff_delta_e_abs",
)
PRIMARY_STATIC_ERROR_KEYS: tuple[str, ...] = (
    "static_abs_delta_e",
    "abs_delta_e",
    "primary_abs_delta_e",
    "delta_E_abs",
    "delta_e_abs",
)

SNAKE_REGEN_REGIME_ARG: dict[str, str] = {
    "weak_weak": "weak-weak",
    "weak_strong": "weak-strong",
    "strong_weak_u8": "strong-weak-u8",
    "strong_strong_u8": "strong-strong-u8",
}

APPEND_CASE_PROFILE: dict[str, tuple[str, str]] = {
    "weak_weak": (
        "paper_i_three_model_hh_symmetric_20260527_v1",
        "hh_L2_nph2_three_model_sym_weak_weak",
    ),
    "weak_strong": (
        "paper_i_three_model_hh_symmetric_20260527_v1",
        "hh_L2_nph4_three_model_sym_weak_strong",
    ),
    "strong_weak_u8": (
        "paper_i_three_model_hh_symmetric_u8_20260611_v1",
        "hh_L2_nph2_three_model_sym_u8_strong_weak",
    ),
    "strong_strong_u8": (
        "paper_i_three_model_hh_symmetric_u8_20260611_v1",
        "hh_L2_nph4_three_model_sym_u8_strong_strong",
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_registry(path: Path, *, root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = read_json(path, root=root)
    if payload.get("schema") != SOURCE_REGISTRY_SCHEMA:
        raise SeedTrackValidationError(
            f"source registry schema must be {SOURCE_REGISTRY_SCHEMA!r}; got {payload.get('schema')!r}"
        )
    sources = payload.get("sources", [])
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        raise SeedTrackValidationError("source registry sources must be a list")
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in sources:
        if not isinstance(raw, Mapping):
            raise SeedTrackValidationError("every source registry entry must be an object")
        entry = dict(raw)
        key = (
            str(entry.get("hh_regime_id", entry.get("regime_id", ""))).strip(),
            str(entry.get("seed_track", "")).strip(),
        )
        if key in out:
            raise SeedTrackValidationError(f"duplicate source registry entry for {key}")
        out[key] = entry
    return out


def _block_dict(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    raw = payload.get(key, {})
    return dict(raw) if isinstance(raw, Mapping) else {}


def _finite_abs_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        out = abs(float(value))
    except (TypeError, ValueError):
        return None
    return out


def _extract_static_same_cutoff_error(payload: Mapping[str, Any]) -> tuple[float | None, str | None]:
    blocks: tuple[tuple[str, Mapping[str, Any]], ...] = (
        ("paper_ii_seed_lock", _block_dict(payload, "paper_ii_seed_lock")),
        ("paper_ii_static_seed_export", _block_dict(payload, "paper_ii_static_seed_export")),
        ("adapt_vqe", _block_dict(payload, "adapt_vqe")),
        ("result", _block_dict(payload, "result")),
        ("top", payload),
    )
    for block_name, block in blocks:
        for key in SAME_CUTOFF_ERROR_KEYS:
            value = _finite_abs_float(block.get(key))
            if value is not None:
                return value, f"{block_name}.{key}"
    for block_name, block in blocks:
        for key in PRIMARY_STATIC_ERROR_KEYS:
            value = _finite_abs_float(block.get(key))
            if value is not None:
                return value, f"{block_name}.{key}"
    return None, None


def _snake_regen_command(regime_id: str, *, output_root: Path, maxiter: int) -> list[str]:
    return [
        "python3",
        "pipelines/exact_bench/paper_i_hh_speed_optuna.py",
        "--regime",
        SNAKE_REGEN_REGIME_ARG[regime_id],
        "--n-trials",
        "1",
        "--max-depth",
        "30",
        "--benchmark-target-abs-delta-e",
        "1e-4",
        "--force-run-to-depth",
        "--maxiter",
        str(int(maxiter)),
        "--final-refit-maxiter",
        str(int(maxiter)),
        "--search-inner-optimizer",
        "POWELL",
        "--objective-mode",
        "energy",
        "--output-dir",
        str(output_root / "static_powell" / "snake" / regime_id),
    ]


def _append_regen_command(regime_id: str, *, output_root: Path, maxiter: int) -> list[str]:
    _, case_id = APPEND_CASE_PROFILE[regime_id]
    return [
        "env",
        "GENERIC_STATIC_TABLE_ADAPT_OPTIMIZER_KIND=powell",
        "GENERIC_STATIC_TABLE_PHASE3_ADAPT_MAX_DEPTH=30",
        f"GENERIC_STATIC_TABLE_PHASE3_ADAPT_MAXITER={int(maxiter)}",
        "GENERIC_STATIC_TABLE_ENERGY_STOP_TARGET=1e-4",
        "GENERIC_STATIC_TABLE_FIRST_HIT_THRESHOLDS=1e-4,1e-5",
        "GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_MODE=shortlist_pauli_children_v1",
        "GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICY=off",
        "GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_MAX_SUBSET_SIZE=3",
        "TABLE_I_STATIC_SUITE_PROFILE=" + APPEND_CASE_PROFILE[regime_id][0],
        "python3",
        "-m",
        "pipelines.exact_bench.generic_static_benchmark",
        "--run-single",
        "--family",
        "hh",
        "--case-id",
        case_id,
        "--algorithm-id",
        "static_full_meta_append_adapt_vqe",
        "--output-dir",
        str(output_root / "static_powell" / "append" / regime_id),
    ]


def _regen_command(regime_id: str, seed_track: str, *, output_root: Path, maxiter: int) -> list[str]:
    if seed_track == "snake":
        return _snake_regen_command(regime_id, output_root=output_root, maxiter=maxiter)
    if seed_track == "append":
        return _append_regen_command(regime_id, output_root=output_root, maxiter=maxiter)
    return []


def _shell_join(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in argv)


def _audit_one(
    *,
    entry: Mapping[str, Any] | None,
    regime_id: str,
    seed_track: str,
    root: Path,
    hard_threshold: float,
    preferred_threshold: float,
    require_runtime_dry_load: bool,
    output_root: Path,
    maxiter: int,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "hh_regime_id": regime_id,
        "seed_track": seed_track,
        "hard_threshold": float(hard_threshold),
        "preferred_threshold": float(preferred_threshold),
        "accepted": False,
        "status": "rejected",
    }
    if seed_track not in SEED_TRACKS_BY_ID:
        return {**base, "reason": "unknown_seed_track"}
    if regime_id not in SNAKE_REGEN_REGIME_ARG:
        return {**base, "reason": "unknown_regime_id"}
    regen = _regen_command(regime_id, seed_track, output_root=output_root, maxiter=maxiter)
    base["regeneration_command"] = regen
    base["regeneration_command_shell"] = _shell_join(regen) if regen else None
    if entry is None:
        return {**base, "reason": "missing_source_registry_entry"}
    entry_map = dict(entry)
    base["source_registry_entry"] = entry_map
    source = str(entry_map.get("source_artifact_json", "")).strip()
    base["source_artifact_json"] = source
    if not source:
        return {**base, "reason": "missing_source_artifact_json"}
    source_path = repo_path(source, root=root)
    if not source_path.exists():
        return {**base, "reason": "missing_source_artifact"}
    try:
        raw_payload = read_json(source, root=root)
        payload = coerce_hh_seed_payload(raw_payload, source_entry=entry_map, source_path=source)
        validation = validate_hh_seed_source(payload, source_entry=entry_map, source_path=source)
        static_error, metric_source = _extract_static_same_cutoff_error(payload)
        dry_load = (
            dry_load_hh_seed_payload(payload, artifact_json=source)
            if bool(require_runtime_dry_load)
            else {"runtime_loadability_status": "not_run"}
        )
        source_sha = sha256_file(source, root=root)
    except Exception as exc:
        return {
            **base,
            "reason": "validation_failed",
            "validation_error_type": type(exc).__name__,
            "validation_error": str(exc),
        }
    admitted = static_error is not None and static_error <= float(hard_threshold)
    status = "accepted" if admitted else "rejected"
    reason = "accepted" if admitted else (
        "static_abs_delta_e_above_threshold" if static_error is not None else "missing_static_same_cutoff_delta_e"
    )
    return {
        **base,
        "accepted": bool(admitted),
        "status": status,
        "reason": reason,
        "source_artifact_sha256": source_sha,
        "static_abs_delta_e": static_error,
        "static_error_metric_source": metric_source,
        "preferred_threshold_met": bool(static_error is not None and static_error <= float(preferred_threshold)),
        "validation_summary": validation,
        "runtime_loadability": dry_load,
    }


def build_admission(
    *,
    source_registry: Path,
    output_root: Path,
    root: Path = ROOT,
    regime_ids: Sequence[str] = DEFAULT_REGIME_IDS,
    seed_tracks: Sequence[str] = DEFAULT_SEED_TRACKS,
    hard_threshold: float = 1.0e-4,
    preferred_threshold: float = 1.0e-5,
    require_runtime_dry_load: bool = True,
    maxiter: int = 1000,
) -> dict[str, Any]:
    registry = _read_registry(source_registry, root=root)
    rows: list[dict[str, Any]] = []
    accepted_sources: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for regime_id in regime_ids:
        for seed_track in seed_tracks:
            entry = registry.get((str(regime_id), str(seed_track)))
            row = _audit_one(
                entry=entry,
                regime_id=str(regime_id),
                seed_track=str(seed_track),
                root=root,
                hard_threshold=float(hard_threshold),
                preferred_threshold=float(preferred_threshold),
                require_runtime_dry_load=bool(require_runtime_dry_load),
                output_root=Path(output_root),
                maxiter=int(maxiter),
            )
            rows.append(row)
            if row.get("accepted"):
                source_entry = dict(row.get("source_registry_entry", {}))
                source_entry["source_artifact_sha256"] = row.get("source_artifact_sha256")
                source_entry["static_abs_delta_e"] = row.get("static_abs_delta_e")
                source_entry["static_error_metric_source"] = row.get("static_error_metric_source")
                accepted_sources.append(source_entry)
            else:
                rejected.append(row)
    return {
        "schema": "hh_local_seed_admission_ledger_v1",
        "generated_utc": _utc_now(),
        "source_registry": str(source_registry),
        "regime_ids": list(regime_ids),
        "seed_tracks": list(seed_tracks),
        "static_delta_e_policy": {
            "metric": "same_cutoff_abs_delta_e_or_primary_static_abs_delta_e",
            "hard_max": float(hard_threshold),
            "preferred_max": float(preferred_threshold),
        },
        "require_runtime_dry_load": bool(require_runtime_dry_load),
        "accepted_count": len(accepted_sources),
        "rejected_count": len(rejected),
        "rows": rows,
        "accepted_source_registry": {
            "schema": SOURCE_REGISTRY_SCHEMA,
            "generated_utc": _utc_now(),
            "sources": accepted_sources,
        },
        "rejected_rows": rejected,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-registry", type=Path, default=DEFAULT_SOURCE_REGISTRY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--regime-id", action="append", default=None)
    parser.add_argument("--seed-track", action="append", default=None)
    parser.add_argument("--hard-threshold", type=float, default=1.0e-4)
    parser.add_argument("--preferred-threshold", type=float, default=1.0e-5)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--no-runtime-dry-load", action="store_true")
    args = parser.parse_args(argv)

    output_root = repo_path(args.output_root, root=ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    regime_ids = tuple(args.regime_id or DEFAULT_REGIME_IDS)
    seed_tracks = tuple(args.seed_track or DEFAULT_SEED_TRACKS)
    ledger = build_admission(
        source_registry=args.source_registry,
        output_root=args.output_root,
        regime_ids=regime_ids,
        seed_tracks=seed_tracks,
        hard_threshold=float(args.hard_threshold),
        preferred_threshold=float(args.preferred_threshold),
        require_runtime_dry_load=not bool(args.no_runtime_dry_load),
        maxiter=int(args.maxiter),
    )
    ledger_rel = Path(args.output_root) / "hh_local_seed_admission_ledger.json"
    accepted_rel = Path(args.output_root) / "hh_local_accepted_seed_sources.json"
    rejected_rel = Path(args.output_root) / "hh_local_seed_regeneration_plan.json"
    write_json(ledger_rel, ledger, root=ROOT)
    write_json(accepted_rel, ledger["accepted_source_registry"], root=ROOT)
    write_json(
        rejected_rel,
        {
            "schema": "hh_local_seed_regeneration_plan_v1",
            "generated_utc": ledger["generated_utc"],
            "rows": ledger["rejected_rows"],
        },
        root=ROOT,
    )
    print(
        json.dumps(
            {
                "ledger": str(ledger_rel),
                "accepted_source_registry": str(accepted_rel),
                "regeneration_plan": str(rejected_rel),
                "accepted_count": ledger["accepted_count"],
                "rejected_count": ledger["rejected_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
