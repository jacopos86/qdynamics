#!/usr/bin/env python3
"""Validate or execute one source-locked six-regime novelty-on SR-SNAKE row."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


EXPECTED_SCHEMA = "paper_i_hh_sr_noprune_nobeam_novelty_on_chtc_job_v1"
BUNDLE_ID = (
    "paper_i_hh_sr_snake_noprune_nobeam_ordinary_novelty_"
    "all_six_20260715_v1_chtc"
)
REGIME_FLAGS = {"--u", "--g-ep", "--n-ph-max"}
STUDY_FLAGS = {
    "--phase2-gram-novelty-policy",
    "--phase3-gram-novelty-policy",
}
HORIZON_FLAGS = {"--adapt-max-depth"}
PATH_FLAGS = {
    "--adapt-current-json",
    "--adapt-estimator-call-ledger-json",
    "--output-json",
}
ALLOWED_DIFF_FLAGS = REGIME_FLAGS | STUDY_FLAGS | HORIZON_FLAGS | PATH_FLAGS

REQUIRED_FIXED = {
    "--problem": "hh",
    "--L": "2",
    "--ordering": "blocked",
    "--boundary": "open",
    "--t": "1.0",
    "--dv": "0.0",
    "--omega0": "1.0",
    "--boson-encoding": "binary",
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
    "--adapt-beam-live-branches": "1",
    "--adapt-beam-children-per-parent": "1",
    "--adapt-beam-lambda": "0.005",
    "--phase3-backend-cost-mode": "marrakesh_graph_span_v1",
    "--phase3-selector-policy": "algebraic_nested_v1",
    "--phase3-selector-geometry-mode": "reduced",
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
    "--phase2-gram-novelty-policy": "ordinary_multiplier_v1",
    "--phase3-gram-novelty-policy": "ordinary_multiplier_v1",
    "--sr-controller-ablation-contract": "novelty_prune_beam_controls_v1",
}
REQUIRED_TRUE = (
    "--phase0-no-pilot",
    "--phase2-no-batching",
    "--phase3-no-batching",
    "--allow-archival-phase3-runtime-split",
    "--phase1-no-prune",
    "--skip-pdf",
)
FORBIDDEN = (
    "--phase1-prune-enabled",
    "--phase3-novelty-ablation-mode",
    "--sr-escape-mode",
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
    prefix = ["python3", "-m", "pipelines.static_adapt.adapt_pipeline"]
    if list(argv[:3]) != prefix:
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
    if manifest.get("run_class") != "candidate_source_locked_ablation":
        raise ValueError("job must remain the source-locked Study-B ablation")
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
        raise ValueError("recorded changed flags differ from executable argv")
    if changed - ALLOWED_DIFF_FLAGS:
        raise ValueError(f"non-regime route drift: {sorted(changed)}")
    if command.get("unexpected_differences"):
        raise ValueError("job manifest records unexpected settings differences")

    matched_baseline_argv = [
        str(token) for token in command["matched_baseline_argv"]
    ]
    matched_baseline = options(matched_baseline_argv)
    changed_vs_matched_baseline = {
        key
        for key in set(matched_baseline) | set(execution)
        if matched_baseline.get(key) != execution.get(key)
    }
    if changed_vs_matched_baseline != STUDY_FLAGS:
        raise ValueError(
            "Study-B changed fields must be exactly the two ordinary novelty "
            f"policies: {sorted(changed_vs_matched_baseline)}"
        )
    if changed_vs_matched_baseline != set(
        command["changed_flags_vs_matched_baseline"]
    ):
        raise ValueError("recorded Study-B diff does not match executable argv")
    for flag in STUDY_FLAGS:
        if matched_baseline.get(flag) != "fallback_only_v1":
            raise ValueError(f"matched baseline novelty policy drift: {flag}")

    mismatches: dict[str, Any] = {}
    for flag, expected in REQUIRED_FIXED.items():
        if execution.get(flag) != expected:
            mismatches[flag] = {"expected": expected, "actual": execution.get(flag)}
    for flag in REQUIRED_TRUE:
        if execution.get(flag) is not True:
            mismatches[flag] = {"expected": True, "actual": execution.get(flag)}
    for flag in FORBIDDEN:
        if flag in execution:
            mismatches[flag] = {"expected": "absent", "actual": execution[flag]}

    physics = manifest["physics"]
    target_depth = int(manifest["scientific_contract"]["controller_round_target"])
    expected_depth = 30 if slug in {"weak_weak", "intermediate_weak"} else 50
    if target_depth != expected_depth:
        mismatches["controller_round_target"] = {
            "expected": expected_depth,
            "actual": target_depth,
        }
    if execution.get("--adapt-max-depth") != str(expected_depth):
        mismatches["--adapt-max-depth"] = {
            "expected": str(expected_depth),
            "actual": execution.get("--adapt-max-depth"),
        }
    expected_regime = {
        "--u": physics["u_over_t"],
        "--g-ep": physics["g_ep"],
        "--n-ph-max": physics["n_ph_work"],
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
    source_keys = (
        "anchor_command",
        "anchor_result_summary",
        "anchor_execution",
        "anchor_settings_diff",
        "source_revision_manifest",
        "base_source_manifest",
    )
    for key in source_keys:
        source = Path(source_lock[key])
        if sha256(source) != source_lock[f"{key}_sha256"]:
            raise ValueError(f"transferred source record hash mismatch: {key}")
    anchor_payload = json.loads(
        Path(source_lock["anchor_command"]).read_text(encoding="utf-8")
    )
    if [str(token) for token in anchor_payload.get("argv", [])] != anchor_argv:
        raise ValueError("anchor argv differs from locked anchor command")
    result_summary = json.loads(
        Path(source_lock["anchor_result_summary"]).read_text(encoding="utf-8")
    )
    if result_summary.get("source_result_copied_into_bundle") is not False:
        raise ValueError("full anchor result must not be copied into the bundle")
    if result_summary.get("source_result_sha256") != (
        "68fde0ab9de5ae69cee27ac0f54cb52f9e377882969daa0a1630d14f520ffdaa"
    ):
        raise ValueError("anchor result summary source hash drift")
    if result_summary.get("adapt_success") is not True:
        raise ValueError("anchor result summary is not successful")
    archive = Path(source_lock["source_archive"])
    if sha256(archive) != source_lock["source_archive_sha256"]:
        raise ValueError("source archive hash mismatch")

    environment = {
        str(key): str(value) for key, value in manifest["environment"].items()
    }
    audit = manifest["environment_audit"]
    if audit.get("execution_environment") != manifest["environment"]:
        raise ValueError("execution environment audit drift")
    if audit.get("cache_state_contract") != "empty_job_local_no_cross_route_reuse":
        raise ValueError("cache isolation contract drift")
    if environment.get("STATIC_ADAPT_HH_POOL_CACHE_SCOPE") != (
        "accepted_refit_logical_r22_v1"
    ):
        raise ValueError("HH pool cache scope drift from successful anchor")

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
            raise ValueError(f"output path drift: {flag}")
    return execution_argv


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    validate_only = False
    if args and args[0] == "--validate-only":
        validate_only = True
        args = args[1:]
    if len(args) != 1:
        raise SystemExit("usage: run_job.py [--validate-only] JOB_MANIFEST.json")
    manifest_path = Path(args[0])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("job manifest must be a JSON object")
    execution_argv = validate(manifest_path, manifest)
    if validate_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "validation_only": True,
                    "job_manifest": manifest_path.as_posix(),
                    "job_manifest_sha256": sha256(manifest_path),
                },
                sort_keys=True,
            )
        )
        return 0

    paths = {key: Path(value) for key, value in manifest["paths"].items()}
    for path in (
        paths["result_json"],
        paths["current_json"],
        paths["estimator_call_ledger_json"],
        paths["execution_manifest_json"],
        paths["normalized_run_manifest_json"],
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    environment = {
        str(key): str(value) for key, value in manifest["environment"].items()
    }
    for key, value in environment.items():
        if key.endswith("_CACHE_DIR"):
            cache_path = Path(value)
            if cache_path.exists():
                raise ValueError(f"job-local cache must begin absent: {cache_path}")
            cache_path.mkdir(parents=True, exist_ok=False)

    normalized = {
        "schema": "paper_i_hh_sr_noprune_nobeam_novelty_on_runtime_manifest_v1",
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
    execution = {
        **normalized,
        "schema": "paper_i_hh_sr_noprune_nobeam_novelty_on_execution_v1",
        "status": "running",
        "exit_code": None,
    }
    write_json(paths["execution_manifest_json"], execution)
    env = os.environ.copy()
    env.update(environment)
    try:
        completed = subprocess.run(execution_argv, env=env, check=False)
        execution["exit_code"] = int(completed.returncode)
        execution["status"] = "completed" if completed.returncode == 0 else "failed"
        return int(completed.returncode)
    finally:
        execution["finished_utc"] = utc_now()
        execution["artifacts"] = {
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
        write_json(paths["execution_manifest_json"], execution)


if __name__ == "__main__":
    raise SystemExit(main())
