#!/usr/bin/env python3
"""Execute one source-locked SR-SNAKE no-ordinary-novelty CHTC row."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


EXPECTED_SCHEMA = "paper_i_hh_sr_no_ordinary_novelty_chtc_job_v1"
BUNDLE_ID = "paper_i_hh_sr_snake_no_ordinary_novelty_all_six_20260715_v1_chtc"
METHOD_DIFF_FLAGS = {
    "--phase2-gram-novelty-policy",
    "--phase3-gram-novelty-policy",
    "--sr-controller-ablation-contract",
}
REGIME_FLAGS = {"--u", "--g-ep", "--n-ph-max"}
PATH_FLAGS = {
    "--adapt-current-json",
    "--adapt-estimator-call-ledger-json",
    "--output-json",
}
ALLOWED_DIFF_FLAGS = METHOD_DIFF_FLAGS | REGIME_FLAGS | PATH_FLAGS

REQUIRED_FIXED = {
    "--problem": "hh",
    "--L": "2",
    "--adapt-pool": "full_meta",
    "--adapt-continuation-mode": "phase3_v1",
    "--static-route-id": "route_a",
    "--static-meta-feature-profile": "paper_i_production_v1",
    "--static-lane-route": "physical_operator_type",
    "--physical-lane-shortlist-aggressiveness": "3",
    "--adapt-inner-optimizer": "POWELL",
    "--adapt-maxiter": "200",
    "--adapt-scipy-maxfev": "0",
    "--adapt-reopt-policy": "windowed",
    "--adapt-full-refit-every": "8",
    "--adapt-final-full-refit": "true",
    "--adapt-final-refit-maxiter": "200",
    "--adapt-max-depth": "30",
    "--phase1-shortlist-size": "24",
    "--phase2-shortlist-size": "12",
    "--phase2-shortlist-fraction": "0.25",
    "--phase3-runtime-split-mode": "shortlist_pauli_children_v1",
    "--phase3-runtime-split-selection-mode": "archival_child_set_forward_v1",
    "--phase3-runtime-split-max-subset-size": "1",
    "--phase3-runtime-split-child-set-symmetry-policy": "hard_guard",
    "--phase3-runtime-split-child-padding-policy": "exact_projected_grouped_v1",
    "--phase1-prune-policy": "recoverability_ladder_v1",
    "--phase1-prune-mode": "both",
    "--adapt-beam-live-branches": "3",
    "--adapt-beam-children-per-parent": "2",
    "--adapt-beam-lambda": "0.005",
    "--historical-singleton-coordinate-solve-policy": (
        "supported_metric_whitened_eigh_v1"
    ),
    "--historical-singleton-trust-region-update-policy": (
        "displacement_calibrated_unbounded_v2"
    ),
    "--adapt-accepted-refit-scope": "full_ansatz_v1",
    "--adapt-accepted-refit-coordinate-chart": "supported_fs_whitened_fixed_v1",
    "--adapt-accepted-refit-base-chart-policy": (
        "expanded_runtime_projected_logical_v1"
    ),
}
REQUIRED_TRUE = (
    "--phase0-no-pilot",
    "--phase2-no-batching",
    "--phase3-no-batching",
    "--allow-archival-phase3-runtime-split",
    "--phase1-prune-enabled",
    "--skip-pdf",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def options(argv: Sequence[str]) -> dict[str, Any]:
    if list(argv[:3]) != ["python3", "-m", "pipelines.static_adapt.adapt_pipeline"]:
        raise ValueError("unexpected command prefix")
    result: dict[str, Any] = {}
    index = 3
    while index < len(argv):
        flag = str(argv[index])
        if not flag.startswith("--") or flag in result:
            raise ValueError(f"invalid or duplicate option: {flag!r}")
        if index + 1 < len(argv) and not str(argv[index + 1]).startswith("--"):
            result[flag] = str(argv[index + 1])
            index += 2
        else:
            result[flag] = True
            index += 1
    return result


def validate(manifest_path: Path, manifest: dict[str, Any]) -> list[str]:
    if manifest.get("schema") != EXPECTED_SCHEMA:
        raise ValueError("unexpected job manifest schema")
    if manifest.get("bundle_id") != BUNDLE_ID:
        raise ValueError("unexpected bundle id")
    slug = str(manifest.get("regime_slug", ""))
    if not slug or "/" in slug or ".." in slug:
        raise ValueError("unsafe regime slug")

    command = manifest["command"]
    anchor_argv = [str(token) for token in command["anchor_argv"]]
    execution_argv = [str(token) for token in command["execution_argv"]]
    anchor = options(anchor_argv)
    execution = options(execution_argv)
    changed = {
        key
        for key in set(anchor) | set(execution)
        if anchor.get(key) != execution.get(key)
    }
    if changed != set(command["changed_flags"]):
        raise ValueError("recorded changed flags do not match executable argv")
    if changed - ALLOWED_DIFF_FLAGS:
        raise ValueError(f"unexpected execution argv drift: {sorted(changed)}")
    if not METHOD_DIFF_FLAGS.issubset(changed):
        raise ValueError("one or more required novelty-ablation differences are absent")
    if command.get("unexpected_differences"):
        raise ValueError("job manifest records unexpected settings differences")

    mismatches: dict[str, Any] = {}
    for flag, expected in REQUIRED_FIXED.items():
        if execution.get(flag) != expected:
            mismatches[flag] = {"expected": expected, "actual": execution.get(flag)}
    for flag in REQUIRED_TRUE:
        if execution.get(flag) is not True:
            mismatches[flag] = {"expected": True, "actual": execution.get(flag)}
    expected_ablation = {
        "--phase2-gram-novelty-policy": "fallback_only_v1",
        "--phase3-gram-novelty-policy": "fallback_only_v1",
        "--sr-controller-ablation-contract": "novelty_prune_controls_v1",
    }
    for flag, expected in expected_ablation.items():
        if execution.get(flag) != expected:
            mismatches[flag] = {"expected": expected, "actual": execution.get(flag)}
    if "--phase3-novelty-ablation-mode" in execution:
        mismatches["--phase3-novelty-ablation-mode"] = {
            "expected": "absent/off",
            "actual": execution["--phase3-novelty-ablation-mode"],
        }
    if "--sr-escape-mode" in execution:
        mismatches["--sr-escape-mode"] = {
            "expected": "absent, frozen default disabled",
            "actual": execution["--sr-escape-mode"],
        }
    physics = manifest["physics"]
    expected_regime = {
        "--u": str(physics["u_over_t"]),
        "--g-ep": str(physics["g_ep"]),
        "--n-ph-max": str(physics["n_ph_work"]),
    }
    for flag, expected in expected_regime.items():
        actual = execution.get(flag)
        if actual is None or float(actual) != float(expected):
            mismatches[flag] = {"expected": expected, "actual": actual}
    if physics["n_ph_work"] != physics["n_ph_reference"]:
        mismatches["same_cutoff_reference"] = {
            "expected": physics["n_ph_work"],
            "actual": physics["n_ph_reference"],
        }
    if mismatches:
        raise ValueError(f"scientific contract mismatch: {mismatches}")

    source_lock = manifest["source_lock"]
    source_files = {
        "anchor_command": "anchor_command_sha256",
        "anchor_settings_diff": "anchor_settings_diff_sha256",
        "source_manifest": "source_manifest_sha256",
    }
    for path_key, hash_key in source_files.items():
        path = Path(source_lock[path_key])
        if sha256(path) != source_lock[hash_key]:
            raise ValueError(f"transferred source record hash mismatch: {path_key}")
    anchor_payload = json.loads(
        Path(source_lock["anchor_command"]).read_text(encoding="utf-8")
    )
    if [str(token) for token in anchor_payload.get("argv", [])] != anchor_argv:
        raise ValueError("anchor command argv does not match job manifest")
    archive = Path(source_lock["source_archive"])
    if sha256(archive) != source_lock["source_archive_sha256"]:
        raise ValueError("source archive hash mismatch in execute sandbox")

    environment = {str(key): str(value) for key, value in manifest["environment"].items()}
    environment_audit = manifest.get("environment_audit", {})
    if environment_audit.get("execution_environment") != manifest.get("environment"):
        raise ValueError("execution environment audit drift")
    if environment_audit.get("scientific_settings_changed") is not False:
        raise ValueError("cache environment classified as scientific drift")
    if environment_audit.get("cache_state_contract") != (
        "empty_job_local_no_cross_route_reuse"
    ):
        raise ValueError("cache isolation contract drift")
    transfer_contract = manifest.get("transfer_contract", {})
    if transfer_contract.get("mode") != "compressed_output_bundle_v1":
        raise ValueError("compressed transfer contract missing")
    if transfer_contract.get("source_directory") != manifest["paths"]["output_root"]:
        raise ValueError("compressed transfer source directory drift")
    if transfer_contract.get("when_to_transfer_output") != "ON_EXIT_OR_EVICT":
        raise ValueError("checkpoint transfer timing drift")

    paths = {key: Path(value) for key, value in manifest["paths"].items()}
    expected_root = Path("raw_outputs") / BUNDLE_ID / slug
    if paths["output_root"] != expected_root:
        raise ValueError("output root drift")
    for key, path in paths.items():
        if key != "output_root" and expected_root not in path.parents:
            raise ValueError(f"output path escaped isolated root: {key}")
    expected_paths = {
        "--adapt-current-json": paths["current_json"].as_posix(),
        "--adapt-estimator-call-ledger-json": paths[
            "estimator_call_ledger_json"
        ].as_posix(),
        "--output-json": paths["result_json"].as_posix(),
    }
    for flag, expected in expected_paths.items():
        if execution.get(flag) != expected:
            raise ValueError(f"execution output path drift: {flag}")
    return execution_argv


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit("usage: run_job.py JOB_MANIFEST.json")
    manifest_path = Path(args[0])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("job manifest must be a JSON object")
    execution_argv = validate(manifest_path, manifest)
    paths = {key: Path(value) for key, value in manifest["paths"].items()}
    output_root = paths["output_root"]
    for path in (
        paths["result_json"],
        paths["current_json"],
        paths["estimator_call_ledger_json"],
        paths["execution_manifest_json"],
        paths["normalized_run_manifest_json"],
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    environment = {str(key): str(value) for key, value in manifest["environment"].items()}
    for key, value in environment.items():
        if key.endswith("_CACHE_DIR"):
            cache_path = Path(value)
            if cache_path.exists():
                raise ValueError(f"job-local cache must begin absent: {cache_path}")
            cache_path.mkdir(parents=True, exist_ok=False)
    normalized = {
        "schema": "paper_i_hh_sr_no_ordinary_novelty_runtime_manifest_v1",
        "job_manifest": manifest_path.as_posix(),
        "job_manifest_sha256": sha256(manifest_path),
        "command_argv": execution_argv,
        "environment": environment,
        "route_identity": manifest["route_identity"],
        "physics": manifest["physics"],
        "scientific_contract": manifest["scientific_contract"],
        "settings_difference": manifest["settings_difference"],
        "source_lock": manifest["source_lock"],
        "started_utc": utc_now(),
    }
    write_json(paths["normalized_run_manifest_json"], normalized)
    execution_record = {
        **normalized,
        "schema": "paper_i_hh_sr_no_ordinary_novelty_execution_v1",
        "status": "running",
        "exit_code": None,
    }
    write_json(paths["execution_manifest_json"], execution_record)
    env = os.environ.copy()
    env.update(environment)
    try:
        completed = subprocess.run(execution_argv, env=env, check=False)
        execution_record["exit_code"] = int(completed.returncode)
        execution_record["status"] = (
            "completed" if completed.returncode == 0 else "failed"
        )
        return int(completed.returncode)
    finally:
        execution_record["finished_utc"] = utc_now()
        execution_record["artifacts"] = {
            key: {
                "path": path.as_posix(),
                "exists": path.is_file(),
                "sha256": sha256(path) if path.is_file() else None,
                "size_bytes": path.stat().st_size if path.is_file() else None,
            }
            for key, path in {
                "result_json": paths["result_json"],
                "current_json": paths["current_json"],
                "estimator_call_ledger_json": paths[
                    "estimator_call_ledger_json"
                ],
            }.items()
        }
        execution_record["output_root"] = output_root.as_posix()
        write_json(paths["execution_manifest_json"], execution_record)


if __name__ == "__main__":
    raise SystemExit(main())
