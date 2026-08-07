#!/usr/bin/env python3
"""Build the six-regime source-locked SR-SNAKE novelty ablation bundle."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


BUNDLE_ID = "paper_i_hh_sr_snake_no_ordinary_novelty_all_six_20260715_v1_chtc"
BATCH_NAME = "paper-i-hh-sr-no-ordinary-novelty-six-20260715-v1"
BUNDLE_DIR = Path(__file__).resolve().parent
REPO = BUNDLE_DIR.parents[3]

ANCHOR_ROOT = Path(
    "raw_outputs/"
    "paper_i_hh_sr_snake_weak_weak_full_accepted_refit_whitened_20260715"
)
ANCHOR_COMMAND = (
    ANCHOR_ROOT / "expanded_runtime_projected_logical_v1_r30/command.json"
)
ANCHOR_SETTINGS_DIFF = (
    ANCHOR_ROOT
    / "expanded_runtime_projected_logical_v1_r30/source_lock_and_settings_diff.json"
)
SOURCE_MANIFEST = ANCHOR_ROOT / "source_lock/source_manifest.json"
SOURCE_ARCHIVE = ANCHOR_ROOT / "source_lock/source_tree.tar.gz"
LOCKED_ARCHIVE = BUNDLE_DIR / "source_locked.tar.gz"

ANCHOR_COMMAND_SHA256 = (
    "7823df8a3cb4c900a0d9a21366c18e354cd982804bdc8ad52199541ec20bd800"
)
ANCHOR_SETTINGS_DIFF_SHA256 = (
    "de0f9cd5fefc88ea27557480b84eeb9d5bc4186e36792e0e5f18a0b4a117501f"
)
SOURCE_MANIFEST_SHA256 = (
    "7e0b8d4b72bcabb5de842ae89afd8d34ce861ebf0e2b9d16fe12f883c73aa416"
)
SOURCE_ARCHIVE_SHA256 = (
    "f0ced05fb7c4ab242ef10323c13ac0e3e3d5be2c15b255c931b73aa8a980cbae"
)
IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)

PARITY_COMMANDS = {
    "intermediate_weak": (
        Path(
            "raw_outputs/"
            "paper_i_hh_sr_snake_intermediate_weak_full_accepted_refit_whitened_20260715/"
            "expanded_runtime_projected_logical_v1_r30/command.json"
        ),
        "576e402f29117726f1dcaea82b2a0654a6089e2034d836a8d1a2f18ed63adfdd",
    ),
    "strong_weak_u8": (
        Path(
            "raw_outputs/"
            "paper_i_hh_sr_snake_strong_weak_u8_full_accepted_refit_whitened_20260715/"
            "expanded_runtime_projected_logical_v1_r30/command.json"
        ),
        "3340e423bf654ba562655ec820023eb76afbb23150941066de120a444b4bc3e1",
    ),
}

REGIMES = (
    {
        "slug": "weak_weak",
        "u": "0.25",
        "lambda": 0.25,
        "g_ep": "0.353553390593",
        "n_ph_max": "2",
        "memory_mb": 24576,
        "disk_mb": 40960,
    },
    {
        "slug": "intermediate_weak",
        "u": "1.25",
        "lambda": 0.25,
        "g_ep": "0.353553390593",
        "n_ph_max": "2",
        "memory_mb": 24576,
        "disk_mb": 40960,
    },
    {
        "slug": "strong_weak_u8",
        "u": "8.0",
        "lambda": 0.25,
        "g_ep": "0.353553390593",
        "n_ph_max": "2",
        "memory_mb": 24576,
        "disk_mb": 40960,
    },
    {
        "slug": "weak_strong",
        "u": "0.25",
        "lambda": 1.25,
        "g_ep": "0.790569415042",
        "n_ph_max": "4",
        "memory_mb": 32768,
        "disk_mb": 61440,
    },
    {
        "slug": "intermediate_strong",
        "u": "1.25",
        "lambda": 1.25,
        "g_ep": "0.790569415042",
        "n_ph_max": "4",
        "memory_mb": 32768,
        "disk_mb": 61440,
    },
    {
        "slug": "strong_strong_u8",
        "u": "8.0",
        "lambda": 1.25,
        "g_ep": "0.790569415042",
        "n_ph_max": "4",
        "memory_mb": 32768,
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
    "--phase1-prune-policy": "recoverability_ladder_v1",
    "--phase1-prune-mode": "both",
    "--adapt-beam-live-branches": "3",
    "--adapt-beam-children-per-parent": "2",
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
    "--phase3-runtime-split-child-padding-policy": "exact_projected_grouped_v1",
}

ROUTE_TRUE_FLAGS = (
    "--phase2-no-batching",
    "--phase3-no-batching",
    "--allow-archival-phase3-runtime-split",
    "--phase1-prune-enabled",
    "--phase0-no-pilot",
    "--skip-pdf",
)

ALLOWED_METHOD_DIFFS = {
    "--phase2-gram-novelty-policy": {
        "from": "ordinary_multiplier_v1",
        "to": "fallback_only_v1",
    },
    "--phase3-gram-novelty-policy": {
        "from": "ordinary_multiplier_v1",
        "to": "fallback_only_v1",
    },
    "--sr-controller-ablation-contract": {
        "from": "off",
        "to": "novelty_prune_controls_v1",
    },
}

OPERATIONAL_PATH_FLAGS = {
    "--adapt-current-json",
    "--adapt-estimator-call-ledger-json",
    "--output-json",
}
REGIME_FLAGS = {"--u", "--g-ep", "--n-ph-max"}


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


def set_option(argv: list[str], flag: str, value: str) -> None:
    if flag in argv:
        position = argv.index(flag)
        if position + 1 >= len(argv) or str(argv[position + 1]).startswith("--"):
            raise ValueError(f"cannot replace boolean option {flag}")
        argv[position + 1] = value
        return
    insertion = argv.index("--adapt-current-json")
    argv[insertion:insertion] = [flag, value]


def validate_anchor(argv: list[str]) -> None:
    observed = options(argv)
    mismatches = {
        flag: {"expected": expected, "actual": observed.get(flag)}
        for flag, expected in ROUTE_FIXED_OPTIONS.items()
        if observed.get(flag) != expected
    }
    for flag in ROUTE_TRUE_FLAGS:
        if observed.get(flag) is not True:
            mismatches[flag] = {"expected": True, "actual": observed.get(flag)}
    for flag in ALLOWED_METHOD_DIFFS:
        if flag in observed:
            mismatches[flag] = {"expected": "absent on anchor", "actual": observed[flag]}
    if "--sr-escape-mode" in observed:
        mismatches["--sr-escape-mode"] = {
            "expected": "absent, frozen default disabled",
            "actual": observed["--sr-escape-mode"],
        }
    if mismatches:
        raise ValueError(f"anchor route mismatch: {mismatches}")


def validate_preserved_weak_regime_parity(anchor: list[str]) -> dict[str, Any]:
    anchor_options = options(anchor)
    ignored = REGIME_FLAGS | OPERATIONAL_PATH_FLAGS
    anchor_route = {key: value for key, value in anchor_options.items() if key not in ignored}
    records: dict[str, Any] = {}
    for slug, (relative, expected_hash) in PARITY_COMMANDS.items():
        path = REPO / relative
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"preserved {slug} command hash mismatch")
        payload = load_json(path)
        candidate = [str(token) for token in payload.get("argv", [])]
        candidate_options = options(candidate)
        candidate_route = {
            key: value for key, value in candidate_options.items() if key not in ignored
        }
        if candidate_route != anchor_route:
            changed = sorted(
                key
                for key in set(anchor_route) | set(candidate_route)
                if anchor_route.get(key) != candidate_route.get(key)
            )
            raise ValueError(f"preserved {slug} route drift: {changed}")
        records[slug] = {
            "path": relative.as_posix(),
            "sha256": actual_hash,
            "route_settings_equal_anchor": True,
        }
    return records


def validate_archive(path: Path, source_manifest: dict[str, Any]) -> dict[str, Any]:
    source_hashes = {
        str(key): str(value)
        for key, value in source_manifest.get("source_hashes", {}).items()
    }
    critical = {
        name: source_hashes[name]
        for name in (
            "pipelines/static_adapt/adapt_pipeline.py",
            "pipelines/static_adapt/cli_config.py",
            "pipelines/static_adapt/sr_snake_route_profile.py",
            "pipelines/static_adapt/sr_snake_escape_controller.py",
            "pipelines/static_adapt/accepted_refit.py",
            "pipelines/static_adapt/joint_linear_solve.py",
            "pipelines/scaffold/hh_continuation_scoring.py",
        )
    }
    observed: dict[str, str] = {}
    file_count = 0
    with tarfile.open(path, "r:gz") as handle:
        for member in handle.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"unsafe archive member: {member.name}")
            if not member.isfile() and not member.isdir():
                raise ValueError(f"special archive member forbidden: {member.name}")
            if member.isfile():
                file_count += 1
                if name.as_posix() in critical:
                    extracted = handle.extractfile(member)
                    if extracted is None:
                        raise ValueError(f"unreadable archive member: {member.name}")
                    observed[name.as_posix()] = hashlib.sha256(extracted.read()).hexdigest()
    mismatches = {
        name: {"expected": expected, "actual": observed.get(name)}
        for name, expected in critical.items()
        if observed.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"critical archive source mismatch: {mismatches}")
    return {
        "schema": "paper_i_hh_sr_source_archive_inventory_v1",
        "archive_path": LOCKED_ARCHIVE.relative_to(REPO).as_posix(),
        "archive_sha256": sha256(path),
        "archive_size_bytes": path.stat().st_size,
        "file_count": file_count,
        "critical_source_hashes": critical,
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
    set_option(execution, "--phase2-gram-novelty-policy", "fallback_only_v1")
    set_option(execution, "--phase3-gram-novelty-policy", "fallback_only_v1")
    set_option(
        execution,
        "--sr-controller-ablation-contract",
        "novelty_prune_controls_v1",
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
    allowed = sorted(OPERATIONAL_PATH_FLAGS | REGIME_FLAGS | set(ALLOWED_METHOD_DIFFS))
    unexpected = sorted(set(changed) - set(allowed))
    missing_method_diffs = sorted(set(ALLOWED_METHOD_DIFFS) - set(changed))
    if unexpected or missing_method_diffs:
        raise ValueError(
            f"{slug} settings drift: unexpected={unexpected}, "
            f"missing_method_diffs={missing_method_diffs}"
        )
    for flag in ALLOWED_METHOD_DIFFS:
        if execution_options.get(flag) != ALLOWED_METHOD_DIFFS[flag]["to"]:
            raise ValueError(f"{slug} failed to materialize {flag}")
    if "--phase3-novelty-ablation-mode" in execution_options:
        raise ValueError("legacy broad novelty ablation must remain absent/off")
    if "--sr-escape-mode" in execution_options:
        raise ValueError("negative-curvature escape must remain absent/frozen-disabled")

    environment = {
        "PYTHONPATH": "/work",
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
        "STATIC_ADAPT_HH_POOL_CACHE_SCOPE": (
            f"sr_no_ordinary_novelty_{slug}_v1"
        ),
    }
    return {
        "schema": "paper_i_hh_sr_no_ordinary_novelty_chtc_job_v1",
        "bundle_id": BUNDLE_ID,
        "batch_name": BATCH_NAME,
        "regime_slug": slug,
        "created_utc": utc_now(),
        "run_class": "diagnostic_ablation_unpromoted",
        "route_identity": {
            "family": "singleton_response_snake",
            "anchor_profile": "supported_whitened_adaptive_trust_v1",
            "anchor_powell_chart": "expanded_runtime_projected_logical_v1",
            "ablation_label": "no_ordinary_phase2_or_phase3_gram_novelty_v1",
            "canonical_identity_changed": False,
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
            "execution_argv": execution,
            "changed_flags": changed,
            "allowed_changed_flags": allowed,
            "unexpected_differences": unexpected,
        },
        "scientific_contract": {
            "controller_round_target": 30,
            "phase0_enabled": False,
            "phase2_batching_enabled": False,
            "phase3_batching_enabled": False,
            "singleton_admission": True,
            "phase2_gram_novelty_policy": "fallback_only_v1",
            "phase3_gram_novelty_policy": "fallback_only_v1",
            "ordinary_novelty_multiplier_enabled": False,
            "novelty_telemetry_retained": True,
            "all_energy_models_infeasible_novelty_fallback_retained": True,
            "phase3_novelty_ablation_mode": "off",
            "negative_curvature_escape": "disabled_by_frozen_default",
            "pruning": "recoverability_ladder_v1",
            "beam_live_branches": 3,
            "accepted_refit_scope": "full_ansatz_v1",
            "accepted_refit_coordinate_chart": "supported_fs_whitened_fixed_v1",
            "accepted_refit_base_chart": "expanded_runtime_projected_logical_v1",
        },
        "settings_difference": {
            "method_differences": ALLOWED_METHOD_DIFFS,
            "regime_axis": {
                "--u": str(regime["u"]),
                "--g-ep": str(regime["g_ep"]),
                "--n-ph-max": str(regime["n_ph_max"]),
            },
            "operational_path_fields": sorted(OPERATIONAL_PATH_FLAGS),
            "all_other_executable_fields_identical_to_anchor": True,
        },
        "source_lock": {
            "source_archive": (BUNDLE_DIR / "source_locked.tar.gz")
            .relative_to(REPO)
            .as_posix(),
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "anchor_command": source_records["anchor_command"]["path"],
            "anchor_command_sha256": source_records["anchor_command"]["sha256"],
            "anchor_settings_diff": source_records["anchor_settings_diff"]["path"],
            "anchor_settings_diff_sha256": source_records["anchor_settings_diff"][
                "sha256"
            ],
            "source_manifest": source_records["source_manifest"]["path"],
            "source_manifest_sha256": source_records["source_manifest"]["sha256"],
        },
        "paths": paths,
        "environment": environment,
        "environment_audit": {
            "cache_state_contract": "empty_job_local_no_cross_route_reuse",
            "scientific_settings_changed": False,
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


def main() -> int:
    if "local_repos" not in REPO.parts or "Documents" in REPO.parts:
        raise RuntimeError(f"non-iCloud checkout guard failed: {REPO}")
    expected = {
        REPO / ANCHOR_COMMAND: ANCHOR_COMMAND_SHA256,
        REPO / ANCHOR_SETTINGS_DIFF: ANCHOR_SETTINGS_DIFF_SHA256,
        REPO / SOURCE_MANIFEST: SOURCE_MANIFEST_SHA256,
        REPO / SOURCE_ARCHIVE: SOURCE_ARCHIVE_SHA256,
    }
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise ValueError(f"source-lock hash mismatch: {path}")

    anchor_payload = load_json(REPO / ANCHOR_COMMAND)
    anchor_argv = [str(token) for token in anchor_payload.get("argv", [])]
    validate_anchor(anchor_argv)
    preserved_parity = validate_preserved_weak_regime_parity(anchor_argv)
    source_manifest_payload = load_json(REPO / SOURCE_MANIFEST)

    shutil.copy2(REPO / SOURCE_ARCHIVE, LOCKED_ARCHIVE)
    if sha256(LOCKED_ARCHIVE) != SOURCE_ARCHIVE_SHA256:
        raise ValueError("copied source archive hash mismatch")
    archive_inventory = validate_archive(LOCKED_ARCHIVE, source_manifest_payload)
    json_dump(BUNDLE_DIR / "source_archive_manifest.json", archive_inventory)

    source_lock_dir = BUNDLE_DIR / "source_lock"
    source_lock_dir.mkdir(parents=True, exist_ok=True)
    copied = {
        "anchor_command": (ANCHOR_COMMAND, source_lock_dir / "anchor_command.json"),
        "anchor_settings_diff": (
            ANCHOR_SETTINGS_DIFF,
            source_lock_dir / "anchor_settings_diff.json",
        ),
        "source_manifest": (SOURCE_MANIFEST, source_lock_dir / "source_manifest.json"),
    }
    source_records: dict[str, dict[str, str]] = {}
    for key, (source_relative, destination) in copied.items():
        shutil.copy2(REPO / source_relative, destination)
        source_records[key] = {
            "path": destination.relative_to(REPO).as_posix(),
            "sha256": sha256(destination),
        }

    jobs: list[dict[str, Any]] = []
    queue_lines: list[str] = []
    for regime in REGIMES:
        job = build_job(anchor_argv, regime, source_records)
        slug = str(regime["slug"])
        job_path = BUNDLE_DIR / "jobs" / f"{slug}.json"
        json_dump(job_path, job)
        jobs.append(
            {
                "regime_slug": slug,
                "job_manifest": job_path.relative_to(REPO).as_posix(),
                "job_manifest_sha256": sha256(job_path),
                "physics": job["physics"],
                "changed_flags": job["command"]["changed_flags"],
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

    settings_audit = {
        "schema": "paper_i_hh_sr_no_ordinary_novelty_settings_diff_v1",
        "bundle_id": BUNDLE_ID,
        "created_utc": utc_now(),
        "anchor": source_records,
        "preserved_weak_regime_route_parity": preserved_parity,
        "method_differences": ALLOWED_METHOD_DIFFS,
        "regime_axes": [job["physics"] for job in jobs],
        "unchanged_contract": {
            "route_settings": ROUTE_FIXED_OPTIONS,
            "true_flags": list(ROUTE_TRUE_FLAGS),
            "phase3_novelty_ablation_mode": "off_by_absence",
            "negative_curvature_escape": "disabled_by_frozen_default",
            "prune_policy": "recoverability_ladder_v1",
            "beam_live_branches": 3,
        },
        "isolation_verdict": "pass",
    }
    json_dump(BUNDLE_DIR / "source_lock_and_settings_diff.json", settings_audit)

    bundle_manifest = {
        "schema": "paper_i_hh_sr_no_ordinary_novelty_six_job_bundle_v1",
        "bundle_id": BUNDLE_ID,
        "batch_name": BATCH_NAME,
        "created_utc": utc_now(),
        "job_count": len(jobs),
        "jobs": jobs,
        "source_archive": archive_inventory,
        "source_records": source_records,
        "settings_diff": {
            "path": (
                BUNDLE_DIR / "source_lock_and_settings_diff.json"
            ).relative_to(REPO).as_posix(),
            "sha256": sha256(BUNDLE_DIR / "source_lock_and_settings_diff.json"),
        },
        "execution_image": {
            "path": "chtc/phase3_optuna/image.sif",
            "sha256": IMAGE_SHA256,
        },
        "submission_status": "staged_not_submitted",
    }
    json_dump(BUNDLE_DIR / "bundle_manifest.json", bundle_manifest)

    preflight = {
        "schema": "paper_i_hh_sr_no_ordinary_novelty_preflight_v1",
        "created_utc": utc_now(),
        "status": "pass",
        "checks": {
            "non_icloud_checkout": True,
            "source_archive_hash": True,
            "critical_source_hashes": True,
            "preserved_weak_route_parity": True,
            "six_regimes_present": len(jobs) == 6,
            "same_cutoff_reference_all_rows": all(
                job["physics"]["n_ph_work"] == job["physics"]["n_ph_reference"]
                for job in jobs
            ),
            "novelty_policy_isolation": True,
            "batching_disabled": True,
            "phase0_disabled": True,
            "negative_curvature_escape_disabled": True,
            "prune_and_beam_unchanged": True,
            "remote_image_hash_pending_remote_check": True,
        },
    }
    json_dump(BUNDLE_DIR / "preflight.json", preflight)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
