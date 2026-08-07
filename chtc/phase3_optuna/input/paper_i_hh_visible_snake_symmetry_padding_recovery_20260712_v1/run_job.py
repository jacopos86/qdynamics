#!/usr/bin/env python3
"""Execute one source-locked Paper-I recovery job without route mutation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Sequence


EXPECTED_SCHEMA = "paper_i_hh_visible_snake_recovery_job_manifest_v1"
PADDING_POLICY = "exact_projected_grouped_v1"
EXPECTED_CACHE_MODES = {
    "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "disk",
    "STATIC_ADAPT_HH_POOL_CACHE": "disk",
    "STATIC_ADAPT_HH_GENERATOR_REGISTRY_CACHE": "disk",
}
BUNDLE_ID = "paper_i_hh_visible_snake_symmetry_padding_recovery_20260712_v1"


def _expected_environment(regime_slug: str) -> dict[str, str]:
    cache_root = Path("tmp") / BUNDLE_ID / regime_slug / "cache"
    return {
        **EXPECTED_CACHE_MODES,
        "STATIC_ADAPT_CANDIDATE_RECORD_CACHE_DIR": (
            cache_root / "candidate_records"
        ).as_posix(),
        "STATIC_ADAPT_HH_POOL_CACHE_DIR": (cache_root / "hh_pool").as_posix(),
        "STATIC_ADAPT_HH_POOL_CACHE_SCOPE": "exact",
        "STATIC_ADAPT_HH_GENERATOR_REGISTRY_CACHE_DIR": (
            cache_root / "generator_registry"
        ).as_posix(),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _option_map(argv: Sequence[str]) -> dict[str, Any]:
    if list(argv[:3]) != ["python3", "-m", "pipelines.static_adapt.adapt_pipeline"]:
        raise ValueError("command prefix drift")
    result: dict[str, Any] = {}
    index = 3
    while index < len(argv):
        option = str(argv[index])
        if not option.startswith("--") or option in result:
            raise ValueError(f"invalid or duplicate option: {option!r}")
        if index + 1 < len(argv) and not str(argv[index + 1]).startswith("--"):
            result[option] = str(argv[index + 1])
            index += 2
        else:
            result[option] = True
            index += 1
    return result


def _validate_exact_diff(manifest: dict[str, Any]) -> None:
    command = manifest["command"]
    old = _option_map(command["historical_argv"])
    new = _option_map(command["corrected_argv"])
    changed = {option for option in set(old) | set(new) if old.get(option) != new.get(option)}
    expected = {
        "--output-json",
        "--adapt-current-json",
        "--adapt-estimator-call-ledger-json",
        "--phase3-runtime-split-child-padding-policy",
    }
    if changed != expected:
        raise ValueError(f"argv allowlist drift: changed={sorted(changed)!r}")
    paths = manifest["paths"]
    if new["--output-json"] != paths["output_json"]:
        raise ValueError("output-json path drift")
    if new["--adapt-current-json"] != paths["current_json"]:
        raise ValueError("adapt-current-json path drift")
    if new["--adapt-estimator-call-ledger-json"] != paths["estimator_call_ledger_json"]:
        raise ValueError("estimator ledger path drift")
    if new["--phase3-runtime-split-child-padding-policy"] != PADDING_POLICY:
        raise ValueError("padding policy drift")
    if old.get("--phase3-runtime-split-child-set-symmetry-policy") != "hard_guard":
        raise ValueError("historical hard-guard setting absent")
    if new.get("--phase2-no-batching") is not True or new.get("--phase3-no-batching") is not True:
        raise ValueError("historical batching-disabled contract drift")
    if new.get("--phase3-runtime-split-max-subset-size") != "1":
        raise ValueError("historical singleton subset contract drift")


def _validate_locked_inputs(manifest: dict[str, Any]) -> None:
    source_lock = manifest["source_lock"]
    commands_path = Path(source_lock["historical_commands_json"])
    if _sha256_file(commands_path) != source_lock["historical_commands_sha256"]:
        raise ValueError("historical commands.json hash mismatch")
    rows = json.loads(commands_path.read_text(encoding="utf-8"))
    matches = [row for row in rows if str(row.get("regime")) == str(manifest["regime"])]
    if len(matches) != 1:
        raise ValueError("historical regime row is missing or duplicated")
    historical_argv = [str(token) for token in matches[0]["argv"]]
    if historical_argv != [str(token) for token in manifest["command"]["historical_argv"]]:
        raise ValueError("job manifest historical argv does not match locked commands.json")
    argv_path = Path(manifest["command"]["argv_json"])
    if _sha256_file(argv_path) != manifest["command"]["argv_json_sha256"]:
        raise ValueError("corrected argv sidecar hash mismatch")
    argv_payload = json.loads(argv_path.read_text(encoding="utf-8"))
    if argv_payload.get("schema") != "paper_i_hh_visible_snake_recovery_argv_v1":
        raise ValueError("corrected argv sidecar schema mismatch")
    if [str(token) for token in argv_payload.get("argv", [])] != [
        str(token) for token in manifest["command"]["corrected_argv"]
    ]:
        raise ValueError("corrected argv sidecar does not match job manifest")


def _validate_effective_route_contract(manifest: dict[str, Any]) -> None:
    from pipelines.static_adapt.adapt_pipeline import (
        _resolve_physical_lane_shortlist_budget_contract,
    )

    resolved = _resolve_physical_lane_shortlist_budget_contract(
        static_route_id_key="route_a",
        static_meta_feature_profile="paper_i_production_v1",
        static_lane_route_key="physical_operator_type",
        route_a_funnel_active=False,
        adapt_pool="full_meta",
        adapt_continuation_mode="phase3_v1",
        phase2_enable_batching=False,
        phase3_enable_batching=False,
        phase3_runtime_split_mode="shortlist_pauli_children_v1",
        phase3_runtime_split_selection_mode="archival_child_set_forward_v1",
        phase3_runtime_split_max_subset_size=1,
        phase3_runtime_split_subset_sizes=(1,),
        physical_lane_shortlist_factor=3,
        phase1_shortlist_size_base=24,
        phase2_shortlist_size_base=12,
        phase2_shortlist_fraction_base=0.25,
    )
    expected = manifest["effective_route_lock"]["expected_effective"]
    for key, value in expected.items():
        if isinstance(value, float):
            if abs(float(resolved[key]) - float(value)) > 1e-15:
                raise ValueError(f"effective route float drift for {key}: {resolved[key]!r}")
        elif resolved.get(key) != value:
            raise ValueError(f"effective route drift for {key}: {resolved.get(key)!r}")


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit("usage: run_job.py JOB_MANIFEST.json")
    manifest_path = Path(args[0])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(f"unexpected manifest schema: {manifest.get('schema')!r}")
    _validate_locked_inputs(manifest)
    _validate_exact_diff(manifest)
    _validate_effective_route_contract(manifest)
    expected_environment = _expected_environment(str(manifest["regime_slug"]))
    if manifest.get("environment") != expected_environment:
        raise ValueError("cache environment contract drift")
    cache_dirs = [
        Path(value)
        for key, value in expected_environment.items()
        if key.endswith("_CACHE_DIR")
    ]
    preexisting_cache_dirs = [path.as_posix() for path in cache_dirs if path.exists()]
    if preexisting_cache_dirs:
        raise ValueError(
            "job-local cache directories must start absent; stale cache reuse is forbidden: "
            f"{preexisting_cache_dirs!r}"
        )
    for path in cache_dirs:
        path.mkdir(parents=True, exist_ok=False)
    output_json = Path(manifest["paths"]["output_json"])
    output_root = output_json.parents[1]
    expected_root = Path("raw_outputs/paper_i_hh_visible_snake_symmetry_padding_recovery_20260712")
    if expected_root not in output_root.parents and output_root != expected_root:
        raise ValueError(f"output path escaped isolated root: {output_root}")
    runtime_manifest_path = Path(manifest["paths"]["normalized_run_manifest_json"])
    execution = {
        "schema": "paper_i_hh_visible_snake_recovery_execution_manifest_v1",
        "job_id": manifest["job_id"],
        "regime": manifest["regime"],
        "job_manifest_path": manifest_path.as_posix(),
        "job_manifest_sha256": _sha256_file(manifest_path),
        "source_archive": manifest["source_archive"],
        "command_argv": manifest["command"]["corrected_argv"],
        "environment": expected_environment,
        "cache_initial_state": "empty_directories_created_by_run_job",
        "transferred_cache_artifacts": False,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "exit_code": None,
    }
    _write_manifest(runtime_manifest_path, execution)
    env = os.environ.copy()
    env.update(expected_environment)
    try:
        completed = subprocess.run(manifest["command"]["corrected_argv"], check=False, env=env)
        execution["exit_code"] = int(completed.returncode)
        execution["status"] = "completed" if completed.returncode == 0 else "failed"
        return int(completed.returncode)
    finally:
        execution["finished_utc"] = datetime.now(timezone.utc).isoformat()
        execution["output_artifacts"] = {
            key: {
                "path": value,
                "exists": Path(value).is_file(),
                "sha256": _sha256_file(Path(value)) if Path(value).is_file() else None,
            }
            for key, value in {
                "result_json": manifest["paths"]["output_json"],
                "current_json": manifest["paths"]["current_json"],
                "estimator_call_ledger_json": manifest["paths"]["estimator_call_ledger_json"],
            }.items()
        }
        _write_manifest(runtime_manifest_path, execution)


if __name__ == "__main__":
    raise SystemExit(main())
