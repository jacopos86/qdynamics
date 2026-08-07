#!/usr/bin/env python3
"""Prewarm and verify both exact-scope HH caches for scaling jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.exact_bench.table_i_canonical_cases import (  # noqa: E402
    TABLE_I_PAPER_I_SCALING_MATRIX_PROFILE,
    table_i_executable_specs,
)


DEFAULT_BATCH_ID = "paper_i_scaling_matrix_parent_powell200_20260710_v1"
DEFAULT_CACHE_DIR = ROOT / "chtc" / "phase3_optuna" / "input" / DEFAULT_BATCH_ID / "hh_pool_cache_v1"
DEFAULT_GENERATOR_REGISTRY_CACHE_DIR = (
    ROOT
    / "chtc"
    / "phase3_optuna"
    / "input"
    / DEFAULT_BATCH_ID
    / "hh_generator_registry_cache_v1"
)
DEFAULT_MANIFEST = DEFAULT_CACHE_DIR.parent / "hh_cache_prewarm_manifest.json"
DEFAULT_SEED_CACHE_DIR = ROOT / "raw_outputs" / "cache" / "hh_pool_cache_v1"
DEFAULT_SEED_GENERATOR_REGISTRY_CACHE_DIR = (
    ROOT / "raw_outputs" / "cache" / "hh_generator_registry_cache_v1"
)
POOL_TERM_CAP = 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mark_batch_ready(manifest_path: Path) -> None:
    batch_dir = manifest_path.parent
    for name in ("paper_i_scaling_matrix_manifest.json", "submission_audit.json"):
        path = batch_dir / name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object at {path}")
        payload["hh_cache_prewarm_status"] = "pass"
        payload["hh_pool_cache_prewarm_status"] = "pass"
        for field in ("hh_pool_cache", "hh_generator_registry_cache"):
            cache = payload.get(field)
            if isinstance(cache, dict):
                cache["status"] = "pass"
        payload["status"] = "ready_for_preflight"
        _write_json(path, payload)


def _cache_events(result: Any) -> list[dict[str, Any]]:
    return [dict(event) for event in result.pool_cache_events if isinstance(event, Mapping)]


def _event_names(events: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(event.get("event") or "") for event in events]


def _cache_path_for_event(
    events: Sequence[Mapping[str, Any]],
    *,
    accepted_names: set[str],
    expected_dir: Path,
) -> Path:
    matches = [event for event in events if str(event.get("event") or "") in accepted_names]
    if not matches:
        raise RuntimeError(f"Required cache event missing; accepted={sorted(accepted_names)} events={_event_names(events)}")
    raw_path = str(matches[-1].get("cache_path") or "").strip()
    if not raw_path:
        raise RuntimeError(f"Required cache event has no cache_path: {matches[-1]}")
    path = Path(raw_path).expanduser().resolve()
    expected = expected_dir.expanduser().resolve()
    if path.parent != expected or path.suffix != ".pickle":
        raise RuntimeError(f"Cache event escaped staged directory: {path} expected_parent={expected}")
    if not path.is_file():
        raise RuntimeError(f"Cache event points to a missing file: {path}")
    return path


def _reset_pickle_cache_dir(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for path in cache_dir.iterdir():
        if path.is_file() and (path.suffix == ".pickle" or path.name.startswith(".")):
            path.unlink()


def _prune_cache_dir(cache_dir: Path, keep_paths: set[Path]) -> list[Path]:
    expected = cache_dir.expanduser().resolve()
    normalized = {path.expanduser().resolve() for path in keep_paths}
    for path in expected.iterdir():
        if path.is_file() and path.suffix == ".pickle" and path.resolve() not in normalized:
            path.unlink()
    remaining = sorted(path.resolve() for path in expected.iterdir() if path.is_file() and path.suffix == ".pickle")
    if set(remaining) != normalized:
        raise RuntimeError(
            f"Pruned cache does not match requested exact files: remaining={len(remaining)} expected={len(normalized)}"
        )
    return remaining


def _seed_batch_cache(cache_dir: Path, seed_cache_dir: Path | None) -> dict[str, Any]:
    if seed_cache_dir is None:
        return {"requested": False, "status": "disabled", "copied_file_count": 0}
    source = Path(seed_cache_dir).expanduser().resolve()
    if not source.is_dir():
        return {
            "requested": True,
            "status": "missing_fallback_to_cold_build",
            "seed_cache_dir": _repo_path(source),
            "copied_file_count": 0,
        }
    source_files = sorted(path for path in source.iterdir() if path.is_file() and path.suffix == ".pickle")
    provenance_rows: list[dict[str, Any]] = []
    for path in source_files:
        target = cache_dir / path.name
        shutil.copy2(path, target)
        provenance_rows.append(
            {
                "name": path.name,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    provenance_digest = hashlib.sha256(
        json.dumps(provenance_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "requested": True,
        "status": "copied",
        "seed_cache_dir": _repo_path(source),
        "seed_cache_file_count": len(source_files),
        "seed_cache_manifest_sha256": provenance_digest,
        "copied_file_count": len(provenance_rows),
    }


def prewarm(
    *,
    cache_dir: Path,
    generator_registry_cache_dir: Path = DEFAULT_GENERATOR_REGISTRY_CACHE_DIR,
    manifest_path: Path,
    seed_cache_dir: Path | None = DEFAULT_SEED_CACHE_DIR,
    seed_generator_registry_cache_dir: Path | None = DEFAULT_SEED_GENERATOR_REGISTRY_CACHE_DIR,
) -> dict[str, Any]:
    cache_dir = Path(cache_dir).expanduser().resolve()
    generator_registry_cache_dir = Path(generator_registry_cache_dir).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    _reset_pickle_cache_dir(cache_dir)
    _reset_pickle_cache_dir(generator_registry_cache_dir)
    pool_seed_provenance = _seed_batch_cache(cache_dir, seed_cache_dir)
    registry_seed_provenance = _seed_batch_cache(
        generator_registry_cache_dir,
        seed_generator_registry_cache_dir,
    )
    os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "disk"
    os.environ["STATIC_ADAPT_HH_POOL_CACHE_SCOPE"] = "exact"
    os.environ["STATIC_ADAPT_HH_POOL_CACHE_DIR"] = str(cache_dir)
    os.environ["STATIC_ADAPT_HH_GENERATOR_REGISTRY_CACHE"] = "disk"
    os.environ["STATIC_ADAPT_HH_GENERATOR_REGISTRY_CACHE_DIR"] = str(generator_registry_cache_dir)
    os.environ["GENERIC_STATIC_TABLE_RESOURCE_QUBIT_CAP"] = "16"
    os.environ["GENERIC_STATIC_TABLE_RESOURCE_POOL_TERM_CAP"] = str(POOL_TERM_CAP)

    from pipelines.exact_bench import generic_static_adapt_variants as variants
    from pipelines.scaffold.hh_continuation_generators import clear_pool_generator_registry_cache_memory
    from pipelines.static_adapt.builders.hh_pool_presets import clear_hh_pool_cache_memory

    clear_hh_pool_cache_memory()
    clear_pool_generator_registry_cache_memory()

    specs = [
        spec
        for spec in table_i_executable_specs(TABLE_I_PAPER_I_SCALING_MATRIX_PROFILE)
        if str(spec.family) == "hh"
    ]
    if len(specs) != 12:
        raise ValueError(f"Expected 12 HH scaling specs, got {len(specs)}")

    build_rows: list[dict[str, Any]] = []
    used_pool_paths: set[Path] = set()
    used_registry_paths: set[Path] = set()
    for spec in specs:
        context = variants._resolve_context_from_spec(spec)
        result = variants._build_full_meta_candidate_pool_with_meta(
            context,
            max_terms=POOL_TERM_CAP,
            hh_full_meta_class_filter_json=None,
        )
        events = _cache_events(result)
        pool_path = _cache_path_for_event(
            events,
            accepted_names={"hardcoded_adapt_pool_cache_hit", "hardcoded_adapt_pool_cache_stored"},
            expected_dir=cache_dir,
        )
        registry_path = _cache_path_for_event(
            events,
            accepted_names={
                "hardcoded_adapt_generator_registry_cache_hit",
                "hardcoded_adapt_generator_registry_cache_stored",
            },
            expected_dir=generator_registry_cache_dir,
        )
        used_pool_paths.add(pool_path)
        used_registry_paths.add(registry_path)
        build_rows.append(
            {
                "case_id": str(spec.benchmark_id),
                "pool_term_count": len(result.candidates),
                "pool_key": result.pool_key,
                "cache_events": _event_names(events),
                "pool_cache_path": _repo_path(pool_path),
                "generator_registry_cache_path": _repo_path(registry_path),
            }
        )

    pool_cache_files = _prune_cache_dir(cache_dir, used_pool_paths)
    registry_cache_files = _prune_cache_dir(generator_registry_cache_dir, used_registry_paths)
    if len(pool_cache_files) != len(specs) or len(registry_cache_files) != len(specs):
        raise RuntimeError(
            "Expected one exact cache file per HH case: "
            f"pool={len(pool_cache_files)} registry={len(registry_cache_files)} cases={len(specs)}"
        )

    clear_hh_pool_cache_memory()
    clear_pool_generator_registry_cache_memory()
    verify_rows: list[dict[str, Any]] = []
    for spec in specs:
        context = variants._resolve_context_from_spec(spec)
        result = variants._build_full_meta_candidate_pool_with_meta(
            context,
            max_terms=POOL_TERM_CAP,
            hh_full_meta_class_filter_json=None,
        )
        events = _cache_events(result)
        by_name = {str(event.get("event") or ""): event for event in events}
        pool_hit = by_name.get("hardcoded_adapt_pool_cache_hit")
        registry_hit = by_name.get("hardcoded_adapt_generator_registry_cache_hit")
        if not isinstance(pool_hit, Mapping) or str(pool_hit.get("cache_level") or "") != "disk":
            raise RuntimeError(
                f"HH pool-cache disk-hit verification failed for {spec.benchmark_id}: {_event_names(events)}"
            )
        if not isinstance(registry_hit, Mapping) or str(registry_hit.get("cache_level") or "") != "disk":
            raise RuntimeError(
                "HH generator-registry-cache disk-hit verification failed for "
                f"{spec.benchmark_id}: {_event_names(events)}"
            )
        pool_path = _cache_path_for_event(
            events,
            accepted_names={"hardcoded_adapt_pool_cache_hit"},
            expected_dir=cache_dir,
        )
        registry_path = _cache_path_for_event(
            events,
            accepted_names={"hardcoded_adapt_generator_registry_cache_hit"},
            expected_dir=generator_registry_cache_dir,
        )
        verify_rows.append(
            {
                "case_id": str(spec.benchmark_id),
                "pool_term_count": len(result.candidates),
                "pool_key": result.pool_key,
                "cache_events": _event_names(events),
                "pool_cache_path": _repo_path(pool_path),
                "generator_registry_cache_path": _repo_path(registry_path),
                "pool_cache_disk_hit_verified": True,
                "generator_registry_cache_disk_hit_verified": True,
                "disk_hit_verified": True,
            }
        )

    def file_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
        return [
            {
                "path": _repo_path(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in paths
        ]

    pool_file_rows = file_rows(pool_cache_files)
    registry_file_rows = file_rows(registry_cache_files)
    payload = {
        "schema": "paper_i_scaling_matrix_hh_dual_cache_prewarm_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "suite_profile": TABLE_I_PAPER_I_SCALING_MATRIX_PROFILE,
        "case_count": len(specs),
        "pool_term_cap": POOL_TERM_CAP,
        "builds": build_rows,
        "disk_hit_verification": verify_rows,
        "pool_cache": {
            "mode": "disk",
            "scope": "exact",
            "cache_dir": _repo_path(cache_dir),
            "seed_cache": pool_seed_provenance,
            "file_count": len(pool_file_rows),
            "files": pool_file_rows,
        },
        "generator_registry_cache": {
            "mode": "disk",
            "cache_dir": _repo_path(generator_registry_cache_dir),
            "seed_cache": registry_seed_provenance,
            "file_count": len(registry_file_rows),
            "files": registry_file_rows,
        },
        "total_size_bytes": sum(int(row["size_bytes"]) for row in pool_file_rows + registry_file_rows),
    }
    _write_json(manifest_path, payload)
    _mark_batch_ready(manifest_path)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--generator-registry-cache-dir",
        type=Path,
        default=DEFAULT_GENERATOR_REGISTRY_CACHE_DIR,
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed-cache-dir", type=Path, default=DEFAULT_SEED_CACHE_DIR)
    parser.add_argument(
        "--seed-generator-registry-cache-dir",
        type=Path,
        default=DEFAULT_SEED_GENERATOR_REGISTRY_CACHE_DIR,
    )
    parser.add_argument("--no-seed-cache", action="store_true", default=False)
    args = parser.parse_args(argv)
    payload = prewarm(
        cache_dir=Path(args.cache_dir),
        generator_registry_cache_dir=Path(args.generator_registry_cache_dir),
        manifest_path=Path(args.manifest),
        seed_cache_dir=None if args.no_seed_cache else Path(args.seed_cache_dir),
        seed_generator_registry_cache_dir=(
            None
            if args.no_seed_cache
            else Path(args.seed_generator_registry_cache_dir)
        ),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
