#!/usr/bin/env python3
"""Build the six-regime source-locked SR-SNAKE novelty-on bundle.

The successful local weak--weak run is the executable anchor.  This builder
restores the ordinary Phase-II/III novelty multipliers and applies the settled
30/50-round horizon by regime.  All remaining scientific settings are copied
from the frozen undamped/no-prune/no-beam baseline.
It does not submit jobs or execute a scientific calculation.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


BUNDLE_ID = (
    "paper_i_hh_sr_snake_noprune_nobeam_ordinary_novelty_"
    "all_six_20260715_v1_chtc"
)
BATCH_NAME = "paper-i-hh-sr-noprune-nobeam-novelty-on-six-20260715-v1"
BUNDLE_DIR = Path(__file__).resolve().parent
REPO = BUNDLE_DIR.parents[3]

ANCHOR_ROOT = Path(
    "raw_outputs/"
    "paper_i_hh_sr_snake_weak_weak_undamped_no_prune_no_beam_"
    "no_ordinary_novelty_fallback_on_20260715"
)
ANCHOR_COMMAND = ANCHOR_ROOT / "command.json"
ANCHOR_RESULT = ANCHOR_ROOT / "json/result.json"
ANCHOR_EXECUTION = ANCHOR_ROOT / "execution.json"
ANCHOR_SETTINGS_DIFF = ANCHOR_ROOT / "source_lock_and_settings_diff.json"
SOURCE_REVISION_MANIFEST = ANCHOR_ROOT / "source_lock/source_revision_manifest.json"
SOURCE_ARCHIVE = ANCHOR_ROOT / "source_lock/source_tree_no_beam_ablation_v1.tar.gz"
BASE_SOURCE_MANIFEST = Path(
    "chtc/phase3_optuna/input/"
    "paper_i_hh_sr_snake_no_ordinary_novelty_all_six_20260715_v1_chtc/"
    "source_lock/source_manifest.json"
)
LOCKED_ARCHIVE = BUNDLE_DIR / "source_locked.tar.gz"
LOCAL_SMOKE_VALIDATION = BUNDLE_DIR / "local_smoke_validation.json"

EXPECTED_HASHES = {
    ANCHOR_COMMAND: "87bff4c130d17c994e25dab49ac72d012c9beae97f6d89cd2edb857fc1806014",
    ANCHOR_RESULT: "68fde0ab9de5ae69cee27ac0f54cb52f9e377882969daa0a1630d14f520ffdaa",
    ANCHOR_EXECUTION: "c62770cd1d828a7ee5c4bfb52a24510e3aacb6ba29b28b76f63645081204e6fc",
    ANCHOR_SETTINGS_DIFF: "a9f381968cb503bc9f931ece8711e2574e4b359238134375fa908019ad02c6f6",
    SOURCE_REVISION_MANIFEST: "dc6f68c6f78a737a773b7672846485e6d591f8ffd4cd2ec06aa7491339a33817",
    SOURCE_ARCHIVE: "94c2df6df22c6d277aefdd6559273d943e3724d476ecab6648c6dd11e1fd78c6",
    BASE_SOURCE_MANIFEST: "7e0b8d4b72bcabb5de842ae89afd8d34ce861ebf0e2b9d16fe12f883c73aa416",
}
SOURCE_ARCHIVE_SHA256 = EXPECTED_HASHES[SOURCE_ARCHIVE]
IMAGE_SHA256 = "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"

REGIMES = (
    {
        "slug": "weak_weak",
        "u": "0.25",
        "lambda": 0.25,
        "g_ep": "0.353553390593",
        "n_ph_max": "2",
        "max_depth": "30",
        "memory_mb": 32768,
        "disk_mb": 61440,
    },
    {
        "slug": "intermediate_weak",
        "u": "1.25",
        "lambda": 0.25,
        "g_ep": "0.353553390593",
        "n_ph_max": "2",
        "max_depth": "30",
        "memory_mb": 32768,
        "disk_mb": 61440,
    },
    {
        "slug": "strong_weak_u8",
        "u": "8.0",
        "lambda": 0.25,
        "g_ep": "0.353553390593",
        "n_ph_max": "2",
        "max_depth": "50",
        "memory_mb": 32768,
        "disk_mb": 61440,
    },
    {
        "slug": "weak_strong",
        "u": "0.25",
        "lambda": 1.25,
        "g_ep": "0.790569415042",
        "n_ph_max": "4",
        "max_depth": "50",
        "memory_mb": 40960,
        "disk_mb": 61440,
    },
    {
        "slug": "intermediate_strong",
        "u": "1.25",
        "lambda": 1.25,
        "g_ep": "0.790569415042",
        "n_ph_max": "4",
        "max_depth": "50",
        "memory_mb": 40960,
        "disk_mb": 61440,
    },
    {
        "slug": "strong_strong_u8",
        "u": "8.0",
        "lambda": 1.25,
        "g_ep": "0.790569415042",
        "n_ph_max": "4",
        "max_depth": "50",
        "memory_mb": 40960,
        "disk_mb": 61440,
    },
)

ROUTE_FIXED_OPTIONS = {
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
    "--phase2-gram-novelty-policy": "fallback_only_v1",
    "--phase3-gram-novelty-policy": "fallback_only_v1",
    "--sr-controller-ablation-contract": "novelty_prune_beam_controls_v1",
}

ROUTE_TRUE_FLAGS = (
    "--phase0-no-pilot",
    "--phase2-no-batching",
    "--phase3-no-batching",
    "--allow-archival-phase3-runtime-split",
    "--phase1-no-prune",
    "--skip-pdf",
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
FORBIDDEN_ACTIVE_FLAGS = {
    "--phase1-prune-enabled",
    "--phase3-novelty-ablation-mode",
    "--sr-escape-mode",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


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


def set_option(argv: list[str], flag: str, value: str) -> None:
    if flag in argv:
        position = argv.index(flag)
        if position + 1 >= len(argv) or str(argv[position + 1]).startswith("--"):
            raise ValueError(f"cannot replace boolean option {flag}")
        argv[position + 1] = value
        return
    insertion = argv.index("--adapt-current-json")
    argv[insertion:insertion] = [flag, value]


def route_signature(argv: Sequence[str]) -> dict[str, Any]:
    observed = options(argv)
    return {
        key: value
        for key, value in observed.items()
        if key not in REGIME_FLAGS | STUDY_FLAGS | HORIZON_FLAGS | PATH_FLAGS
    }


def validate_anchor(argv: list[str]) -> None:
    observed = options(argv)
    mismatches: dict[str, Any] = {}
    for flag, expected in ROUTE_FIXED_OPTIONS.items():
        if observed.get(flag) != expected:
            mismatches[flag] = {"expected": expected, "actual": observed.get(flag)}
    for flag in ROUTE_TRUE_FLAGS:
        if observed.get(flag) is not True:
            mismatches[flag] = {"expected": True, "actual": observed.get(flag)}
    for flag in FORBIDDEN_ACTIVE_FLAGS:
        if flag in observed:
            mismatches[flag] = {"expected": "absent", "actual": observed[flag]}
    if mismatches:
        raise ValueError(f"successful local anchor route mismatch: {mismatches}")


def validate_archive(path: Path, base_manifest: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "pipelines/static_adapt/adapt_pipeline.py": (
            "4884830d93e489e9d9f0daef2cdd3dc819d899eef08e5ca33d964b8515d93301"
        ),
        "pipelines/static_adapt/cli_config.py": (
            "d3b46ac2c28eaa43af3d9a14a961761ba247d5d8d9b9781116db3ac28a4a70b9"
        ),
        "pipelines/static_adapt/sr_snake_escape_controller.py": (
            "323183c2819c0bab1e3ee19da24cd2dfea24a269a39baf0e6f3965dd2bf7b1f4"
        ),
    }
    unchanged = (
        "pipelines/static_adapt/sr_snake_route_profile.py",
        "pipelines/static_adapt/accepted_refit.py",
        "pipelines/static_adapt/joint_linear_solve.py",
        "pipelines/scaffold/hh_continuation_scoring.py",
    )
    base_hashes = base_manifest.get("source_hashes", {})
    expected.update({name: str(base_hashes[name]) for name in unchanged})
    observed: dict[str, str] = {}
    file_count = 0
    with tarfile.open(path, "r:gz") as handle:
        for member in handle.getmembers():
            normalized = PurePosixPath(member.name.lstrip("./"))
            if normalized.is_absolute() or ".." in normalized.parts:
                raise ValueError(f"unsafe source archive member: {member.name}")
            if not member.isfile() and not member.isdir():
                raise ValueError(f"special source archive member: {member.name}")
            if member.isfile():
                file_count += 1
                name = normalized.as_posix()
                if name in expected:
                    stream = handle.extractfile(member)
                    if stream is None:
                        raise ValueError(f"unreadable source member: {member.name}")
                    observed[name] = hashlib.sha256(stream.read()).hexdigest()
    mismatches = {
        name: {"expected": digest, "actual": observed.get(name)}
        for name, digest in expected.items()
        if observed.get(name) != digest
    }
    if mismatches:
        raise ValueError(f"critical frozen-source mismatch: {mismatches}")
    return {
        "schema": "paper_i_hh_sr_noprune_nobeam_source_archive_inventory_v1",
        "archive_path": LOCKED_ARCHIVE.relative_to(REPO).as_posix(),
        "archive_sha256": sha256(path),
        "archive_size_bytes": path.stat().st_size,
        "file_count": file_count,
        "critical_source_hashes": expected,
        "source_revision": "no_beam_ablation_v1",
    }


def build_job(
    anchor_argv: list[str],
    regime: dict[str, Any],
    source_records: dict[str, dict[str, str]],
) -> dict[str, Any]:
    slug = str(regime["slug"])
    output_root = Path("raw_outputs") / BUNDLE_ID / slug
    execution = list(anchor_argv)
    set_option(execution, "--u", str(regime["u"]))
    set_option(execution, "--g-ep", str(regime["g_ep"]))
    set_option(execution, "--n-ph-max", str(regime["n_ph_max"]))
    set_option(execution, "--adapt-max-depth", str(regime["max_depth"]))
    set_option(
        execution,
        "--phase2-gram-novelty-policy",
        "ordinary_multiplier_v1",
    )
    set_option(
        execution,
        "--phase3-gram-novelty-policy",
        "ordinary_multiplier_v1",
    )
    paths = {
        "output_root": output_root.as_posix(),
        "result_json": (output_root / "json/result.json").as_posix(),
        "current_json": (output_root / "json/current.json").as_posix(),
        "estimator_call_ledger_json": (
            output_root / "json/estimator_call_ledger.json"
        ).as_posix(),
        "execution_manifest_json": (output_root / "execution.json").as_posix(),
        "normalized_run_manifest_json": (
            output_root / "normalized_run_manifest.json"
        ).as_posix(),
    }
    set_option(execution, "--adapt-current-json", paths["current_json"])
    set_option(
        execution,
        "--adapt-estimator-call-ledger-json",
        paths["estimator_call_ledger_json"],
    )
    set_option(execution, "--output-json", paths["result_json"])

    anchor_options = options(anchor_argv)
    execution_options = options(execution)
    changed = sorted(
        key
        for key in set(anchor_options) | set(execution_options)
        if anchor_options.get(key) != execution_options.get(key)
    )
    unexpected = sorted(
        set(changed) - (REGIME_FLAGS | STUDY_FLAGS | HORIZON_FLAGS | PATH_FLAGS)
    )
    if unexpected:
        raise ValueError(f"{slug} non-regime route drift: {unexpected}")
    if route_signature(execution) != route_signature(anchor_argv):
        raise ValueError(f"{slug} route signature differs from successful anchor")

    environment = {
        "PYTHONPATH": "/work",
        "PYTHONUNBUFFERED": "1",
        "MPLCONFIGDIR": (output_root / "cache/matplotlib").as_posix(),
        "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "disk",
        "STATIC_ADAPT_CANDIDATE_RECORD_CACHE_DIR": (
            output_root / "cache/candidate_records"
        ).as_posix(),
        "STATIC_ADAPT_HH_GENERATOR_REGISTRY_CACHE_DIR": (
            output_root / "cache/hh_generator_registry"
        ).as_posix(),
        "STATIC_ADAPT_HH_POOL_CACHE": "disk",
        "STATIC_ADAPT_HH_POOL_CACHE_DIR": (
            output_root / "cache/hh_pool"
        ).as_posix(),
        "STATIC_ADAPT_HH_POOL_CACHE_SCOPE": "accepted_refit_logical_r22_v1",
    }
    matched_baseline_argv = list(execution)
    set_option(
        matched_baseline_argv,
        "--phase2-gram-novelty-policy",
        "fallback_only_v1",
    )
    set_option(
        matched_baseline_argv,
        "--phase3-gram-novelty-policy",
        "fallback_only_v1",
    )
    matched_baseline_options = options(matched_baseline_argv)
    changed_vs_matched_baseline = sorted(
        key
        for key in set(matched_baseline_options) | set(execution_options)
        if matched_baseline_options.get(key) != execution_options.get(key)
    )
    if changed_vs_matched_baseline != sorted(STUDY_FLAGS):
        raise ValueError(
            f"{slug} Study-B isolation failure: {changed_vs_matched_baseline}"
        )

    return {
        "schema": "paper_i_hh_sr_noprune_nobeam_novelty_on_chtc_job_v1",
        "bundle_id": BUNDLE_ID,
        "batch_name": BATCH_NAME,
        "regime_slug": slug,
        "created_utc": utc_now(),
        "run_class": "candidate_source_locked_ablation",
        "route_identity": {
            "family": "singleton_response_snake",
            "route_label": (
                "undamped_no_prune_no_beam_ordinary_novelty_on_v1"
            ),
            "canonical_identity_changed": False,
            "status": "explicit_candidate_ablation",
        },
        "physics": {
            "problem": "hh",
            "L": 2,
            "u_over_t": float(regime["u"]),
            "lambda": float(regime["lambda"]),
            "g_ep": float(regime["g_ep"]),
            "n_ph_work": int(regime["n_ph_max"]),
            "n_ph_reference": int(regime["n_ph_max"]),
            "same_cutoff_reference": True,
        },
        "command": {
            "anchor_argv": anchor_argv,
            "matched_baseline_argv": matched_baseline_argv,
            "execution_argv": execution,
            "changed_flags": changed,
            "changed_flags_vs_matched_baseline": changed_vs_matched_baseline,
            "allowed_changed_flags": sorted(
                REGIME_FLAGS | STUDY_FLAGS | HORIZON_FLAGS | PATH_FLAGS
            ),
            "unexpected_differences": unexpected,
            "all_nonapproved_fields_identical_to_matched_baseline": True,
        },
        "scientific_contract": {
            "regular_energy_response_model": "undamped",
            "controller_round_target": int(regime["max_depth"]),
            "phase0_enabled": False,
            "phase2_batching_enabled": False,
            "phase3_batching_enabled": False,
            "singleton_admission": True,
            "pruning_enabled": False,
            "beam_enabled": False,
            "beam_live_branches": 1,
            "beam_children_per_parent": 1,
            "phase2_gram_novelty_policy": "ordinary_multiplier_v1",
            "phase3_gram_novelty_policy": "ordinary_multiplier_v1",
            "ordinary_novelty_multipliers_enabled": True,
            "all_energy_models_infeasible_novelty_fallback_retained": True,
            "negative_curvature_escape": "disabled_by_frozen_default",
            "coordinate_solve_scope": "phase3_only_v1",
            "phase3_coordinate_solve": "supported_metric_whitened_eigh_v1",
            "adaptive_trust": "displacement_calibrated_unbounded_v2",
            "accepted_refit_scope": "full_ansatz_v1",
            "accepted_refit_coordinate_chart": "supported_fs_whitened_fixed_v1",
            "accepted_refit_base_chart": "expanded_runtime_projected_logical_v1",
            "optimizer": "POWELL",
            "optimizer_maxiter": 200,
            "optimizer_maxfev": 0,
            "final_full_refit": True,
            "final_refit_maxiter": 200,
        },
        "settings_difference": {
            "scientific_axis": "phase2_phase3_ordinary_novelty_multiplier_restore",
            "regime_fields": {
                "--u": str(regime["u"]),
                "--g-ep": str(regime["g_ep"]),
                "--n-ph-max": str(regime["n_ph_max"]),
            },
            "operational_path_fields": sorted(PATH_FLAGS),
            "settled_horizon_field": {
                "--adapt-max-depth": str(regime["max_depth"]),
            },
            "changed_fields_vs_matched_regime_horizon_baseline": sorted(
                STUDY_FLAGS
            ),
            "all_other_executable_fields_identical_to_matched_baseline": True,
        },
        "source_lock": {
            "source_archive": LOCKED_ARCHIVE.relative_to(REPO).as_posix(),
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            **{
                key: value
                for name, record in source_records.items()
                for key, value in (
                    (name, record["path"]),
                    (f"{name}_sha256", record["sha256"]),
                )
            },
        },
        "paths": paths,
        "environment": environment,
        "environment_audit": {
            "cache_state_contract": "empty_job_local_no_cross_route_reuse",
            "environmental_differences_from_anchor": (
                "remote_work_root_and_isolated_output_cache_paths_only"
            ),
            "hh_pool_cache_scope_equal_anchor": True,
            "execution_environment": environment,
        },
        "transfer_contract": {
            "mode": "compressed_output_bundle_v1",
            "source_directory": output_root.as_posix(),
            "archive": (
                Path("raw_outputs") / BUNDLE_ID / f"{slug}_transfer.tar.gz"
            ).as_posix(),
            "when_to_transfer_output": "ON_EXIT_OR_EVICT",
        },
    }


def frozen_parse_check(job_paths: list[Path]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="sr_novelty_on_six_parse_") as temporary:
        runtime = Path(temporary)
        with tarfile.open(LOCKED_ARCHIVE, "r:gz") as handle:
            handle.extractall(runtime, filter="data")
        parser_probe = (
            "import json,sys; "
            "from pipelines.static_adapt.adapt_pipeline import parse_args; "
            "d=json.load(open(sys.argv[1])); "
            "a=parse_args(d['command']['execution_argv'][3:]); "
            "v=vars(a); "
            "keys=['adapt_max_depth','phase1_prune_enabled',"
            "'adapt_beam_live_branches','adapt_beam_children_per_parent',"
            "'phase2_gram_novelty_policy','phase3_gram_novelty_policy',"
            "'sr_controller_ablation_contract','sr_escape_mode']; "
            "print('__PARSE_JSON__'+json.dumps({k:v.get(k) for k in keys},sort_keys=True))"
        )
        for job_path in job_paths:
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(runtime)
            completed = subprocess.run(
                [sys.executable, "-c", parser_probe, str(job_path)],
                cwd=runtime,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            marker = "__PARSE_JSON__"
            lines = [line for line in completed.stdout.splitlines() if marker in line]
            if completed.returncode != 0 or not lines:
                raise RuntimeError(
                    f"frozen CLI parse failed for {job_path.name}: "
                    f"rc={completed.returncode} stderr={completed.stderr[-2000:]}"
                )
            resolved = json.loads(lines[-1].split(marker, 1)[1])
            manifest = load_json(job_path)
            expected_depth = int(
                manifest["scientific_contract"]["controller_round_target"]
            )
            expected = {
                "adapt_max_depth": expected_depth,
                "phase1_prune_enabled": False,
                "adapt_beam_live_branches": 1,
                "adapt_beam_children_per_parent": 1,
                "phase2_gram_novelty_policy": "ordinary_multiplier_v1",
                "phase3_gram_novelty_policy": "ordinary_multiplier_v1",
                "sr_controller_ablation_contract": (
                    "novelty_prune_beam_controls_v1"
                ),
                "sr_escape_mode": "disabled",
            }
            if resolved != expected:
                raise ValueError(
                    f"frozen parse policy mismatch for {job_path.name}: "
                    f"{resolved} != {expected}"
                )
            records.append(
                {
                    "job_manifest": job_path.relative_to(REPO).as_posix(),
                    "job_manifest_sha256": sha256(job_path),
                    "frozen_cli_parse": "pass",
                    "resolved_fields": resolved,
                }
            )
    return {
        "schema": "paper_i_hh_sr_novelty_on_six_local_parse_validation_v1",
        "created_utc": utc_now(),
        "status": "pass",
        "scientific_execution_performed": False,
        "validation_kind": "frozen_cli_parse_and_job_manifest_validation_only",
        "jobs": records,
    }


def submit_text() -> str:
    base = f"chtc/phase3_optuna/input/{BUNDLE_ID}"
    return f"""universe = vanilla
executable = {base}/execute_source_locked_job.sh
arguments = $(job_manifest) {base}/source_locked.tar.gz {SOURCE_ARCHIVE_SHA256} chtc/phase3_optuna/image.sif {IMAGE_SHA256} $(regime_slug)
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_executable = True
preserve_relative_paths = True
transfer_input_files = {base}/run_job.py, {base}/source_archive_manifest.json, {base}/source_lock_and_settings_diff.json, {base}/bundle_manifest.json, {base}/preflight.json, {base}/local_parse_validation.json, {base}/local_smoke_validation.json, {base}/source_lock/anchor_command.json, {base}/source_lock/anchor_result_summary.json, {base}/source_lock/anchor_execution.json, {base}/source_lock/anchor_settings_diff.json, {base}/source_lock/source_revision_manifest.json, {base}/source_lock/base_source_manifest.json, $(job_manifest), {base}/source_locked.tar.gz, chtc/phase3_optuna/image.sif
transfer_output_files = raw_outputs/{BUNDLE_ID}/$(regime_slug)_transfer.tar.gz
stream_output = False
stream_error = False
log = logs/{BUNDLE_ID}.$(Cluster).$(Process).log
output = logs/{BUNDLE_ID}.$(Cluster).$(Process).out
error = logs/{BUNDLE_ID}.$(Cluster).$(Process).err
requirements = TARGET.HasSIF
request_cpus = 4
request_memory = $(memory_mb)MB
request_disk = $(disk_mb)MB
+MaxRuntime = 259200
+JobBatchName = \"{BATCH_NAME}\"
notification = Never
queue regime_slug, job_manifest, memory_mb, disk_mb from {base}/queue.tsv
"""


def main() -> int:
    if "local_repos" not in REPO.parts or "Documents" in REPO.parts:
        raise RuntimeError(f"non-iCloud checkout guard failed: {REPO}")
    for relative, expected in EXPECTED_HASHES.items():
        path = REPO / relative
        actual = sha256(path)
        if actual != expected:
            raise ValueError(
                f"source-lock hash mismatch: {relative}: {actual} != {expected}"
            )

    anchor_payload = load_json(REPO / ANCHOR_COMMAND)
    anchor_argv = [str(token) for token in anchor_payload.get("argv", [])]
    validate_anchor(anchor_argv)
    anchor_result = load_json(REPO / ANCHOR_RESULT)
    energy = float(anchor_result["adapt_vqe"]["energy"])
    exact = float(anchor_result["ground_state"]["exact_energy"])
    if not bool(anchor_result["adapt_vqe"].get("success")):
        raise ValueError("successful local anchor result is not successful")
    smoke_validation = load_json(LOCAL_SMOKE_VALIDATION)
    expected_smoke = {
        "status": "pass",
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
    }
    for key, expected in expected_smoke.items():
        if smoke_validation.get(key) != expected:
            raise ValueError(
                f"local frozen-source smoke receipt mismatch: {key}: "
                f"{smoke_validation.get(key)!r} != {expected!r}"
            )
    observed_smoke = smoke_validation.get("observed", {})
    if observed_smoke != {
        "pipeline_returncode": 0,
        "adapt_success": True,
        "ansatz_depth": 0,
        "phase1_prune_enabled": False,
        "phase2_gram_novelty_policy": "ordinary_multiplier_v1",
        "phase3_gram_novelty_policy": "ordinary_multiplier_v1",
        "sr_escape_mode": "disabled",
        "same_cutoff_n_ph": True,
    }:
        raise ValueError("local frozen-source smoke observed fields drift")

    shutil.copy2(REPO / SOURCE_ARCHIVE, LOCKED_ARCHIVE)
    if sha256(LOCKED_ARCHIVE) != SOURCE_ARCHIVE_SHA256:
        raise ValueError("copied revised source archive hash mismatch")
    base_source_manifest = load_json(REPO / BASE_SOURCE_MANIFEST)
    archive_inventory = validate_archive(LOCKED_ARCHIVE, base_source_manifest)
    json_dump(BUNDLE_DIR / "source_archive_manifest.json", archive_inventory)

    source_lock_dir = BUNDLE_DIR / "source_lock"
    if source_lock_dir.exists():
        shutil.rmtree(source_lock_dir)
    source_lock_dir.mkdir(parents=True, exist_ok=True)
    source_files = {
        "anchor_command": ANCHOR_COMMAND,
        "anchor_execution": ANCHOR_EXECUTION,
        "anchor_settings_diff": ANCHOR_SETTINGS_DIFF,
        "source_revision_manifest": SOURCE_REVISION_MANIFEST,
        "base_source_manifest": BASE_SOURCE_MANIFEST,
    }
    source_records: dict[str, dict[str, str]] = {}
    for name, relative in source_files.items():
        destination = source_lock_dir / f"{name}.json"
        shutil.copy2(REPO / relative, destination)
        source_records[name] = {
            "path": destination.relative_to(REPO).as_posix(),
            "sha256": sha256(destination),
        }
    anchor_result_summary_path = source_lock_dir / "anchor_result_summary.json"
    json_dump(
        anchor_result_summary_path,
        {
            "schema": "paper_i_hh_sr_successful_anchor_result_summary_v1",
            "source_result_path": ANCHOR_RESULT.as_posix(),
            "source_result_sha256": EXPECTED_HASHES[ANCHOR_RESULT],
            "source_result_copied_into_bundle": False,
            "adapt_success": True,
            "saved_energy": energy,
            "same_cutoff_exact_energy": exact,
            "same_cutoff_absolute_error": abs(energy - exact),
            "n_ph_work": 2,
            "n_ph_reference": 2,
            "ansatz_depth": int(anchor_result["adapt_vqe"]["ansatz_depth"]),
            "controller_round_target": 30,
            "route_checks": {
                "regular_energy_response_model": "undamped",
                "pruning_enabled": False,
                "beam_enabled": False,
                "beam_live_branches": 1,
                "phase2_gram_novelty_policy": "fallback_only_v1",
                "phase3_gram_novelty_policy": "fallback_only_v1",
                "infeasible_model_novelty_fallback_retained": True,
                "negative_curvature_escape": "disabled",
                "accepted_refit_base_chart": (
                    "expanded_runtime_projected_logical_v1"
                ),
            },
        },
    )
    source_records["anchor_result_summary"] = {
        "path": anchor_result_summary_path.relative_to(REPO).as_posix(),
        "sha256": sha256(anchor_result_summary_path),
    }

    jobs: list[dict[str, Any]] = []
    job_paths: list[Path] = []
    queue_lines: list[str] = []
    for regime in REGIMES:
        job = build_job(anchor_argv, regime, source_records)
        slug = str(regime["slug"])
        job_path = BUNDLE_DIR / "jobs" / f"{slug}.json"
        json_dump(job_path, job)
        job_paths.append(job_path)
        jobs.append(
            {
                "regime_slug": slug,
                "job_manifest": job_path.relative_to(REPO).as_posix(),
                "job_manifest_sha256": sha256(job_path),
                "physics": job["physics"],
                "changed_flags_vs_successful_weak_weak_anchor": job["command"][
                    "changed_flags"
                ],
                "changed_flags_vs_matched_baseline": job["command"][
                    "changed_flags_vs_matched_baseline"
                ],
                "controller_round_target": job["scientific_contract"][
                    "controller_round_target"
                ],
            }
        )
        queue_lines.append(
            "\t".join(
                (
                    slug,
                    job_path.relative_to(REPO).as_posix(),
                    str(regime["memory_mb"]),
                    str(regime["disk_mb"]),
                )
            )
        )
    (BUNDLE_DIR / "queue.tsv").write_text(
        "\n".join(queue_lines) + "\n", encoding="utf-8"
    )
    (BUNDLE_DIR / "submit.sub").write_text(submit_text(), encoding="utf-8")

    settings_diff = {
        "schema": "source_locked_sensitivity_audit_v1",
        "bundle_id": BUNDLE_ID,
        "created_utc": utc_now(),
        "status": "pass",
        "source": {
            **source_records,
            "method": "SR-SNAKE",
            "route_family": "singleton_response_snake",
            "route_profile": (
                "undamped_no_prune_no_beam_no_ordinary_novelty_fallback_on_v1"
            ),
            "runner_mode": "direct_frozen_source_archive",
            "validated_same_cutoff_error": abs(energy - exact),
            "validated_energy": energy,
            "same_cutoff_exact_energy": exact,
            "n_ph_work": 2,
            "n_ph_reference": 2,
            "source_variable_value": {
                "phase2_gram_novelty_policy": "fallback_only_v1",
                "phase3_gram_novelty_policy": "fallback_only_v1",
            },
        },
        "source_archive": archive_inventory,
        "sweep": {
            "run_class": "candidate_source_locked_ablation",
            "variable": "ordinary_phase2_phase3_novelty_multipliers",
            "source_value": "fallback_only_v1",
            "target_value": "ordinary_multiplier_v1",
            "settings_changed": sorted(STUDY_FLAGS),
            "runner_mode": "direct_frozen_source_archive",
            "wrapper_used": False,
            "baseline_materialization_status": "complete",
            "unresolved_source_fields": [],
            "fields_added_by_current_defaults": [],
        },
        "anchor": {
            "value": "fallback_only_v1",
            "anchor_result_json": ANCHOR_RESULT.as_posix(),
            "anchor_result_sha256": EXPECTED_HASHES[ANCHOR_RESULT],
            "anchor_reproduces_source": True,
            "metric_abs_diff": 0.0,
            "non_swept_settings_diff": [],
        },
        "horizon_contract": {
            "weak_weak": 30,
            "intermediate_weak": 30,
            "strong_weak_u8": 50,
            "weak_strong": 50,
            "intermediate_strong": 50,
            "strong_strong_u8": 50,
            "comparison_rule": (
                "compare against the baseline at the same regime and horizon"
            ),
        },
        "route_signature_equal_all_rows": True,
        "route_signature": route_signature(anchor_argv),
        "planned_rows": [
            {
                "regime_slug": job["regime_slug"],
                "physics": job["physics"],
                "controller_round_target": job["controller_round_target"],
                "changed_flags_vs_anchor": job[
                    "changed_flags_vs_successful_weak_weak_anchor"
                ],
                "changed_fields_vs_matched_baseline": job[
                    "changed_flags_vs_matched_baseline"
                ],
                "non_swept_settings_diff": [],
            }
            for job in jobs
        ],
        "unexpected_differences": [],
        "canonical_identity_changed": False,
    }
    json_dump(BUNDLE_DIR / "source_lock_and_settings_diff.json", settings_diff)

    bundle_manifest = {
        "schema": "paper_i_hh_sr_noprune_nobeam_novelty_on_six_bundle_v1",
        "bundle_id": BUNDLE_ID,
        "batch_name": BATCH_NAME,
        "created_utc": utc_now(),
        "run_class": "candidate_source_locked_ablation",
        "submission_status": "staged_not_submitted",
        "job_count": len(jobs),
        "jobs": jobs,
        "successful_local_weak_weak_anchor": {
            "result_summary": source_records["anchor_result_summary"],
            "full_result_source": {
                "path": ANCHOR_RESULT.as_posix(),
                "sha256": EXPECTED_HASHES[ANCHOR_RESULT],
                "copied_into_bundle": False,
            },
            "final_same_cutoff_error": abs(energy - exact),
            "controller_rounds": 30,
        },
        "source_archive": archive_inventory,
        "source_records": source_records,
        "local_smoke_validation": {
            "path": LOCAL_SMOKE_VALIDATION.relative_to(REPO).as_posix(),
            "sha256": sha256(LOCAL_SMOKE_VALIDATION),
            "status": "pass",
        },
        "settings_diff": {
            "path": (BUNDLE_DIR / "source_lock_and_settings_diff.json")
            .relative_to(REPO)
            .as_posix(),
            "sha256": sha256(BUNDLE_DIR / "source_lock_and_settings_diff.json"),
        },
        "execution_image": {
            "path": "chtc/phase3_optuna/image.sif",
            "sha256": IMAGE_SHA256,
            "local_presence": False,
            "remote_hash_check_required_before_submission": True,
        },
    }
    json_dump(BUNDLE_DIR / "bundle_manifest.json", bundle_manifest)

    parse_validation = frozen_parse_check(job_paths)
    runner = BUNDLE_DIR / "run_job.py"
    for job_path in job_paths:
        subprocess.run(
            [sys.executable, str(runner), "--validate-only", str(job_path)],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        )
    parse_validation["runner_manifest_validation_all_rows"] = True
    json_dump(BUNDLE_DIR / "local_parse_validation.json", parse_validation)

    preflight = {
        "schema": "paper_i_hh_sr_noprune_nobeam_novelty_on_six_preflight_v1",
        "created_utc": utc_now(),
        "status": "pass",
        "scientific_execution_performed": True,
        "production_scientific_execution_performed": False,
        "checks": {
            "non_icloud_checkout": True,
            "source_lock_hashes": True,
            "revised_archive_hash": True,
            "critical_frozen_source_hashes": True,
            "successful_local_anchor_locked": True,
            "six_regimes_present": len(jobs) == 6,
            "settled_30_50_round_horizons": {
                job["regime_slug"]: job["controller_round_target"]
                for job in jobs
            }
            == {
                "weak_weak": 30,
                "intermediate_weak": 30,
                "strong_weak_u8": 50,
                "weak_strong": 50,
                "intermediate_strong": 50,
                "strong_strong_u8": 50,
            },
            "weak_holstein_same_cutoff_2": all(
                job["physics"]["n_ph_work"] == 2
                for job in jobs
                if job["physics"]["lambda"] == 0.25
            ),
            "strong_holstein_same_cutoff_4": all(
                job["physics"]["n_ph_work"] == 4
                for job in jobs
                if job["physics"]["lambda"] == 1.25
            ),
            "route_signature_equal_all_rows": True,
            "pruning_disabled": True,
            "beam_disabled_1x1": True,
            "ordinary_novelty_multipliers_enabled_phase2_phase3": True,
            "study_b_only_changed_fields_vs_matched_baseline": all(
                job["changed_flags_vs_matched_baseline"] == sorted(STUDY_FLAGS)
                for job in jobs
            ),
            "infeasible_model_novelty_fallback_retained": True,
            "phase0_and_batching_disabled": True,
            "negative_curvature_route_disabled": True,
            "expanded_projected_logical_chart": True,
            "phase3_whitening_and_adaptive_trust": True,
            "frozen_cli_parse_all_rows": True,
            "frozen_depth_zero_route_resolution_smoke": True,
            "runner_manifest_validation_all_rows": True,
            "submit_description_parse": True,
            "remote_image_hash_pending_remote_check": True,
        },
        "blockers_before_submission": [
            "verify remote chtc/phase3_optuna/image.sif SHA-256 equals "
            f"{IMAGE_SHA256}"
        ],
    }
    json_dump(BUNDLE_DIR / "preflight.json", preflight)

    upload_paths = [
        BUNDLE_DIR / "execute_source_locked_job.sh",
        BUNDLE_DIR / "run_job.py",
        BUNDLE_DIR / "submit.sub",
        BUNDLE_DIR / "queue.tsv",
        BUNDLE_DIR / "bundle_manifest.json",
        BUNDLE_DIR / "preflight.json",
        BUNDLE_DIR / "source_archive_manifest.json",
        BUNDLE_DIR / "source_lock_and_settings_diff.json",
        BUNDLE_DIR / "local_parse_validation.json",
        LOCAL_SMOKE_VALIDATION,
        LOCKED_ARCHIVE,
        *sorted(source_lock_dir.glob("*.json")),
        *job_paths,
    ]
    upload_lines = [path.relative_to(REPO).as_posix() for path in upload_paths]
    (BUNDLE_DIR / "upload_artifact_list.txt").write_text(
        "\n".join(upload_lines) + "\n", encoding="utf-8"
    )
    artifact_hashes = {
        "schema": "paper_i_hh_sr_novelty_on_six_submission_artifact_hashes_v1",
        "created_utc": utc_now(),
        "artifacts": {
            path.relative_to(REPO).as_posix(): {
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in upload_paths
        },
        "required_remote_dependency": {
            "path": "chtc/phase3_optuna/image.sif",
            "sha256": IMAGE_SHA256,
        },
    }
    json_dump(BUNDLE_DIR / "submission_artifact_hashes.json", artifact_hashes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
