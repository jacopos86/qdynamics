#!/usr/bin/env python3
"""Immutable contract helpers for the weak-strong SR outer-reuse pair."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


BUNDLE_ID = (
    "paper_i_hh_sr_outer_information_v5r2_weak_strong_nph7_r50_"
    "20260719_v1_chtc"
)
PAIR_ID = "weak_strong_nph7_r50_control_then_outer_reuse_v1"
SOURCE_RUNTIME_REVISION = "v5r2"
CONTROL_MODE = "control"
REUSE_MODE = "reuse"
MODES = (CONTROL_MODE, REUSE_MODE)

AUTHORITATIVE_JOB_REL = (
    "chtc/phase3_optuna/input/"
    "paper_i_hh_sr_snake_full_response_symmetric_cost_noprune_nobeam_"
    "no_ordinary_novelty_all_six_20260717_v1_chtc/jobs/weak_strong.json"
)
AUTHORITATIVE_JOB_SHA256 = (
    "33e22e62eb23f756d1a20d2597529a1085bbc695b8832e65212b2ae031bc36f7"
)
AUTHORITATIVE_EVIDENCE_VALIDATION_REL = (
    "chtc/phase3_optuna/input/"
    "paper_i_hh_sr_snake_full_response_symmetric_cost_noprune_nobeam_"
    "no_ordinary_novelty_all_six_20260717_v1_chtc/evidence_validation.py"
)
SOURCE_ARCHIVE_REL = (
    "raw_outputs/paper_i_hh_sr_outer_information_source_locked_weak_weak_"
    "20260719/source_lock/"
    "runtime_source_predictive_prephase3_metric_hessian_reuse_"
    "v5r2_suppressed_eps_grad_fallback_20260719.tar.gz"
)
SOURCE_ARCHIVE_SHA256 = (
    "79c1f4b058d6f2ea7fb963975b63ee3e98358f9f463d128115b55fd99b4352d1"
)
SOURCE_AUDIT_REL = (
    "raw_outputs/paper_i_hh_sr_outer_information_source_locked_weak_weak_"
    "20260719/source_lock/"
    "source_lock_audit_predictive_prephase3_metric_hessian_reuse_"
    "v5r2_suppressed_eps_grad_fallback.json"
)
SOURCE_AUDIT_SHA256 = (
    "ed436bae8e36da693af28bb0c066e128f6fd697c9854f534c22d042aa85387e5"
)
PACKAGED_SOURCE_ARCHIVE = "source_locked_v5r2.tar.gz"
PACKAGED_SOURCE_AUDIT = "source_lock_audit_v5r2.json"
RUNTIME_ROOT = (
    "extracted_runtime_predictive_prephase3_metric_hessian_reuse_v5_20260719"
)
IMAGE_REL = "chtc/phase3_optuna/image.sif"
IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)

SR_PROFILE_REQUEST = "sr_snake_no_prune_symmetric_cost_v1"
SR_PROFILE_RESOLVED = (
    "supported_whitened_adaptive_trust_full_response_symmetric_cost_"
    "no_prune_v1"
)
SR_CONTRACT_SHA256 = (
    "69dcf6d5711a05b54cc143562bc6f437511d781164319b2705857b22deda1538"
)
OUTER_PROFILE = "sr_no_prune_symmetric_cost_outer_information_active_v1"
EXPECTED_EXACT_ENERGY = -1.1387206380749124
EXACT_ENERGY_TOLERANCE = 1.0e-12
EXPECTED_N_PH = 7
EXPECTED_TARGET_ROUND = 50
EXPECTED_RESOURCES = {
    "cpus": 4,
    "memory_mb": 49152,
    "disk_mb": 81920,
    "max_runtime_s": 259200,
}

CHECKPOINT_EXIT_CODE = 85
CHECKPOINT_SCHEMA = "paper_i_sr_outer_information_condor_checkpoint_v1"
CHECKPOINT_MANIFEST_NAME = "checkpoint_manifest.json"
CHECKPOINT_CURRENT_NAME = "current.json"

KEY_SOURCE_SHA256 = {
    "pipelines/static_adapt/adapt_pipeline.py": (
        "60587c784564307ec10ea89a7d8f2375a1acf82fab14d485023ec1c1ef69189a"
    ),
    "pipelines/static_adapt/formal_manifold_outer_information.py": (
        "b1b000fba8b3a6b615d820a3dfca5d00bb65be2fa40380d3e124a987771360be"
    ),
    "pipelines/static_adapt/formal_manifold_sr_v3_outer_bridge.py": (
        "fb8f18d159e19ce3b46fdabf7bcbab3a76611dadf65fbf027837ab7e551c2c5d"
    ),
    "pipelines/scaffold/hh_continuation_scoring.py": (
        "dce6fce48c555fa536e08873a1c5e5d58756a0e5a47bd04731cd90dc15e94010"
    ),
}

HISTORICAL_REPAIRED_FAILURE = {
    "id": "sr_outer_reuse_suppressed_gradient_singleton_record_gap_v1",
    "run_id": "20260719T125934Z-c91f6f34",
    "observed_regime": "strong_weak_u8_nph3",
    "observed_controller_round": 30,
    "error": (
        "Source-locked SR outer reuse requires exactly one authoritative "
        "singleton admission."
    ),
    "repair_id": "canonical_sr_suppressed_eps_grad_singleton_exact_metric_v1",
    "repaired_in_source_runtime_revision": SOURCE_RUNTIME_REVISION,
}

REQUIRED_REPAIR_IDS = {
    "canonical_sr_novelty_fallback_exact_metric_v1",
    "canonical_sr_suppressed_eps_grad_singleton_exact_metric_v1",
}
REQUIRED_REGRESSIONS = {
    "test_sr_outer_growth_cache_absence_exact_fallback_is_narrow",
    "test_sr_active_missing_fallback_phase3_growth_uses_exact_powell_and_continues",
    "test_sr_outer_suppressed_eps_grad_singleton_exact_fallback_is_narrow",
}
REQUIRED_QUERY_STATE_CONTRACT = {
    "uncertified_transition_uses_external_predicted_metric": False,
    "uncertified_transition_exact_metric_is_charged": True,
    "candidate_geometry_fabricated": False,
    "accepted_sr_structure_preserved": True,
    "transport_state_invalidated_without_structural_rollback": True,
    "next_round_reanchor_required": True,
    "suppressed_gradient_phase1_singleton_preserved": True,
    "formal_query_accounting_complete_in_regression": True,
    "nfev_reconciled_in_regression": True,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def bundle_dir() -> Path:
    return Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_jsonable(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def checkpoint_archive_path(root: Path, mode: str) -> Path:
    if mode not in MODES:
        raise ValueError(f"invalid checkpoint mode: {mode}")
    return (
        root
        / "raw_outputs"
        / BUNDLE_ID
        / f"{mode}_checkpoint.tar.gz"
    )


def checkpoint_resume_dir(root: Path, mode: str) -> Path:
    if mode not in MODES:
        raise ValueError(f"invalid checkpoint mode: {mode}")
    return root / "raw_outputs" / BUNDLE_ID / "resume_input" / mode


def copy_exact(source: Path, destination: Path, expected_sha256: str) -> None:
    if not source.is_file() or sha256(source) != expected_sha256:
        raise ValueError(f"source missing or hash drifted: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    if sha256(temporary) != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"copied file hash mismatch: {destination}")
    temporary.replace(destination)


def is_appledouble(name: PurePosixPath) -> bool:
    return any(part == "__MACOSX" or part.startswith("._") for part in name.parts)


def inspect_source_archive(archive: Path) -> dict[str, Any]:
    if sha256(archive) != SOURCE_ARCHIVE_SHA256:
        raise ValueError("source archive SHA-256 mismatch")
    files: dict[str, dict[str, Any]] = {}
    directory_count = 0
    appledouble_count = 0
    roots: set[str] = set()
    seen: set[str] = set()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"unsafe archive path: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"archive links are forbidden: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"archive special member is forbidden: {member.name}")
            if is_appledouble(name):
                appledouble_count += 1
                continue
            if not name.parts:
                continue
            roots.add(name.parts[0])
            normalized = str(name)
            if normalized in seen:
                raise ValueError(f"duplicate archive member: {normalized}")
            seen.add(normalized)
            if member.isdir():
                directory_count += 1
                continue
            extracted = handle.extractfile(member)
            if extracted is None:
                raise ValueError(f"regular member is unreadable: {normalized}")
            digest = hashlib.sha256()
            size = 0
            for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
            if size != int(member.size):
                raise ValueError(f"archive member size mismatch: {normalized}")
            relative = PurePosixPath(*name.parts[1:])
            files[str(relative)] = {
                "sha256": digest.hexdigest(),
                "size_bytes": size,
                "archive_member": normalized,
            }
    if roots != {RUNTIME_ROOT}:
        raise ValueError(f"unexpected non-AppleDouble archive roots: {sorted(roots)}")
    for relative, expected in KEY_SOURCE_SHA256.items():
        record = files.get(relative)
        if not isinstance(record, Mapping) or record.get("sha256") != expected:
            raise ValueError(f"key source hash mismatch: {relative}")
    return {
        "schema": "paper_i_sr_outer_information_source_archive_inventory_v1",
        "status": "pass",
        "archive": PACKAGED_SOURCE_ARCHIVE,
        "archive_sha256": SOURCE_ARCHIVE_SHA256,
        "runtime_root": RUNTIME_ROOT,
        "file_count": len(files),
        "directory_count": directory_count,
        "appledouble_ignored_count": appledouble_count,
        "key_sources": {key: files[key] for key in sorted(KEY_SOURCE_SHA256)},
        "files": files,
    }


def safe_extract_source(archive: Path, destination: Path) -> Path:
    """Extract regular source files while ignoring only AppleDouble entries."""

    inventory = inspect_source_archive(archive)
    destination.mkdir(parents=True, exist_ok=True)
    root_resolved = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            name = PurePosixPath(member.name)
            if is_appledouble(name) or member.isdir():
                continue
            target = destination.joinpath(*name.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if root_resolved not in target.resolve().parents:
                raise ValueError(f"archive extraction escaped destination: {member.name}")
            source = handle.extractfile(member)
            if source is None:
                raise ValueError(f"regular member is unreadable: {member.name}")
            temporary = target.with_suffix(target.suffix + ".tmp")
            with temporary.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            os.chmod(temporary, int(member.mode) & 0o777)
            temporary.replace(target)
    runtime = destination / str(inventory["runtime_root"])
    if not runtime.is_dir():
        raise ValueError("source archive did not create its fixed runtime root")
    return runtime


def option_map(argv: Iterable[str]) -> dict[str, Any]:
    tokens = list(argv)
    if tokens[:3] != ["python3", "-m", "pipelines.static_adapt.adapt_pipeline"]:
        raise ValueError("unexpected command prefix")
    parsed: dict[str, Any] = {}
    index = 3
    while index < len(tokens):
        option = str(tokens[index])
        if not option.startswith("--"):
            raise ValueError(f"unexpected positional command token: {option}")
        if option in parsed:
            raise ValueError(f"duplicate command option: {option}")
        if index + 1 < len(tokens) and not str(tokens[index + 1]).startswith("--"):
            parsed[option] = str(tokens[index + 1])
            index += 2
        else:
            parsed[option] = True
            index += 1
    return parsed


def scientific_command_view(job: Mapping[str, Any]) -> dict[str, Any]:
    options = option_map(job["command"]["argv"])
    for key in (
        "--adapt-current-json",
        "--adapt-estimator-call-ledger-json",
        "--output-json",
    ):
        options[key] = "<operational-path>"
    options["--adapt-segment-id"] = "<operational-segment-id>"
    options.setdefault("--adapt-formal-manifold-route-profile", "off")
    return {
        "options": options,
        "physics": job["physics"],
        "segment": {
            key: job["segment"][key]
            for key in (
                "source_controller_round",
                "source_depth",
                "target_controller_round",
                "target_depth",
                "max_new_admissions",
            )
        },
        "sr_profile_request": job["route_identity"]["profile_request"],
        "sr_profile_resolved": job["route_identity"]["profile_resolved"],
        "sr_contract_sha256": job["route_identity"]["profile_contract_sha256"],
        "resource_request": job["resource_request"],
        "environment_policy": {
            "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": job["environment"][
                "STATIC_ADAPT_CANDIDATE_RECORD_CACHE"
            ],
            "STATIC_ADAPT_HH_POOL_CACHE": job["environment"][
                "STATIC_ADAPT_HH_POOL_CACHE"
            ],
            "STATIC_ADAPT_HH_POOL_CACHE_SCOPE": job["environment"][
                "STATIC_ADAPT_HH_POOL_CACHE_SCOPE"
            ],
            "PYTHONNOUSERSITE": job["environment"]["PYTHONNOUSERSITE"],
            "PYTHONDONTWRITEBYTECODE": job["environment"][
                "PYTHONDONTWRITEBYTECODE"
            ],
        },
    }


def recursive_diff(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(path)
            else:
                paths.extend(recursive_diff(left[key], right[key], path))
        return paths
    return [] if left == right else [prefix]


def validate_pair_diff(control: Mapping[str, Any], reuse: Mapping[str, Any]) -> list[str]:
    differences = recursive_diff(
        scientific_command_view(control), scientific_command_view(reuse)
    )
    expected = ["options.--adapt-formal-manifold-route-profile"]
    if differences != expected:
        raise ValueError(
            "control/reuse scientific settings drift; "
            f"expected {expected}, got {differences}"
        )
    return ["adapt_formal_manifold_route_profile"]


def validate_job(
    job: Mapping[str, Any],
    *,
    expected_mode: str | None = None,
    require_anchor: bool = False,
    work_root: Path | None = None,
) -> None:
    mode = str(job.get("pair_contract", {}).get("mode", ""))
    if expected_mode is not None and mode != expected_mode:
        raise ValueError(f"job mode mismatch: {mode} != {expected_mode}")
    if mode not in MODES or job.get("bundle_id") != BUNDLE_ID:
        raise ValueError("unexpected bundle or pair mode")
    if job.get("pair_contract", {}).get("pair_id") != PAIR_ID:
        raise ValueError("pair identity drift")
    if job.get("source_job_lock", {}).get("sha256") != AUTHORITATIVE_JOB_SHA256:
        raise ValueError("authoritative weak-strong job-lock hash drift")
    physics = job.get("physics", {})
    expected_physics = {
        "u_over_t": 0.25,
        "lambda": 1.25,
        "g_ep": 0.790569415042,
        "n_ph_work": EXPECTED_N_PH,
        "n_ph_reference": EXPECTED_N_PH,
        "expected_exact_energy": EXPECTED_EXACT_ENERGY,
        "same_cutoff_reference": True,
    }
    for key, expected in expected_physics.items():
        if physics.get(key) != expected:
            raise ValueError(f"weak-strong NPH7 physics drift: {key}")
    if float(physics.get("exact_energy_tolerance", 0.0)) != EXACT_ENERGY_TOLERANCE:
        raise ValueError("exact-energy tolerance drift")
    route = job.get("route_identity", {})
    if (
        route.get("family") != "singleton_response_snake"
        or route.get("profile_request") != SR_PROFILE_REQUEST
        or route.get("profile_resolved") != SR_PROFILE_RESOLVED
        or route.get("profile_contract_sha256") != SR_CONTRACT_SHA256
    ):
        raise ValueError("source-locked SR route identity drift")
    contract = route.get("profile_contract", {})
    execution = contract.get("execution_settings", {})
    semantics = contract.get("semantic_invariants", {})
    required_execution = {
        "adapt_beam_live_branches": 1,
        "adapt_beam_children_per_parent": 1,
        "phase2_enable_batching": False,
        "phase3_enable_batching": False,
        "phase1_prune_enabled": False,
        "phase2_gram_novelty_policy": "fallback_only_v1",
        "phase3_gram_novelty_policy": "fallback_only_v1",
        "adapt_inner_optimizer": "POWELL",
        "adapt_maxiter": 200,
        "adapt_seed": 7,
        "phase3_response_coordinate_scope": "full_active_plus_singleton_v1",
    }
    for key, expected in required_execution.items():
        if execution.get(key) != expected:
            raise ValueError(f"SR execution setting drift: {key}")
    required_semantics = {
        "admission_rollback_supported": False,
        "pruning_active": False,
        "ordinary_phase2_novelty_multiplier_active": False,
        "ordinary_phase3_novelty_multiplier_active": False,
        "all_energy_models_infeasible_novelty_fallback_active": True,
        "admission_cardinality": 1,
    }
    for key, expected in required_semantics.items():
        if semantics.get(key) != expected:
            raise ValueError(f"SR semantic invariant drift: {key}")
    segment = job.get("segment", {})
    for key, expected in {
        "source_controller_round": 0,
        "source_depth": 0,
        "target_controller_round": EXPECTED_TARGET_ROUND,
        "target_depth": EXPECTED_TARGET_ROUND,
        "max_new_admissions": EXPECTED_TARGET_ROUND,
    }.items():
        if int(segment.get(key, -1)) != expected:
            raise ValueError(f"depth-50 segment drift: {key}")
    if job.get("resource_request") != EXPECTED_RESOURCES:
        raise ValueError("resource request drift")
    options = option_map(job.get("command", {}).get("argv", []))
    if options.get("--n-ph-max") != str(EXPECTED_N_PH):
        raise ValueError("command cutoff drift")
    if options.get("--adapt-max-depth") != str(EXPECTED_TARGET_ROUND):
        raise ValueError("command horizon drift")
    if options.get("--sr-route-profile") != SR_PROFILE_REQUEST:
        raise ValueError("command SR profile drift")
    if options.get("--adapt-disable-hh-seed") is not True:
        raise ValueError("disabled HH preseed drift")
    outer_value = options.get("--adapt-formal-manifold-route-profile", "off")
    expected_outer = "off" if mode == CONTROL_MODE else OUTER_PROFILE
    if outer_value != expected_outer:
        raise ValueError("outer-information profile drift")
    overlay = route.get("outer_information_overlay", {})
    if (
        overlay.get("mode") != mode
        or overlay.get("profile") != expected_outer
        or overlay.get("selector_owner") != "source_locked_sr"
        or overlay.get("accepted_reoptimizer_owner")
        != "source_locked_sr_supported_fs_powell_v1"
        or overlay.get("structural_rollback") is not False
    ):
        raise ValueError("outer-information ownership or rollback drift")
    if any(
        flag in options
        for flag in (
            "--phase2-enable-batching",
            "--phase3-enable-batching",
            "--phase1-prune-enabled",
            "--adapt-beam-live-branches",
            "--phase2-gram-novelty-policy",
            "--phase3-gram-novelty-policy",
        )
    ):
        raise ValueError("profile-owned setting was repeated on the CLI")
    source = job.get("source_lock", {})
    if (
        source.get("archive_sha256") != SOURCE_ARCHIVE_SHA256
        or source.get("audit_sha256") != SOURCE_AUDIT_SHA256
        or source.get("runtime_root") != RUNTIME_ROOT
    ):
        raise ValueError("frozen source-lock identity drift")
    cache_paths = [
        str(value)
        for key, value in job.get("environment", {}).items()
        if key.endswith("_CACHE_DIR")
    ]
    if not cache_paths or any(f"/{mode}/cache/" not in value for value in cache_paths):
        raise ValueError("job cache is not cold and mode-local")
    if require_anchor and mode == REUSE_MODE:
        root = repo_root() if work_root is None else work_root
        gate_path = root / str(job["pair_contract"]["control_gate_path"])
        if not gate_path.is_file():
            raise ValueError("validated current-runtime control gate is missing")
        gate = load_json(gate_path)
        if (
            gate.get("schema") != "paper_i_sr_outer_information_control_gate_v1"
            or gate.get("status") != "pass"
            or gate.get("pair_id") != PAIR_ID
            or gate.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256
            or gate.get("control_job_manifest_sha256")
            != job["pair_contract"]["expected_control_job_manifest_sha256"]
        ):
            raise ValueError("current-runtime control gate is stale or incompatible")


def _resolved_job_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def validate_resume_current(
    job: Mapping[str, Any],
    current_path: Path,
    *,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    """Validate one completed-round structural checkpoint against its node lock."""

    mode = str(job.get("pair_contract", {}).get("mode", ""))
    validate_job(job, expected_mode=mode)
    current = load_json(current_path)
    if (
        current.get("schema_version") != "static_adapt_current_checkpoint_v1"
        or current.get("no_credentials_serialized") is not True
    ):
        raise ValueError("resume current checkpoint schema or credential seal drift")
    checkpoint = current.get("checkpoint")
    adapt = current.get("adapt_vqe")
    settings = current.get("settings")
    if not all(isinstance(value, Mapping) for value in (checkpoint, adapt, settings)):
        raise ValueError("resume current checkpoint blocks are missing")
    checkpoint = dict(checkpoint)
    adapt = dict(adapt)
    settings = dict(settings)
    source_round = int(checkpoint.get("depth", -1))
    source_ansatz_depth = int(checkpoint.get("ansatz_depth", -1))
    if not (0 < source_round <= EXPECTED_TARGET_ROUND):
        raise ValueError("resume checkpoint has no completed outer round")
    if (
        checkpoint.get("reason") != "iteration_done"
        or checkpoint.get("complete") is not False
        or checkpoint.get("branch_id") is not None
        or checkpoint.get("parent_branch_id") is not None
        or checkpoint.get("stop_reason") is not None
        or adapt.get("checkpoint_reason") != "iteration_done"
        or adapt.get("partial_checkpoint") is not True
        or adapt.get("adapt_beam_enabled") is not False
        or adapt.get("branch_id") is not None
        or adapt.get("parent_branch_id") is not None
        or adapt.get("stop_reason") is not None
        or adapt.get("history_checkpoint_complete") is not True
        or int(adapt.get("history_count", -1)) != source_round
        or int(adapt.get("history_tail_count", -1)) != source_round
        or len(adapt.get("history", [])) != source_round
        or len(adapt.get("history_tail", [])) != source_round
        or int(adapt.get("ansatz_depth", -1)) != source_ansatz_depth
        or source_ansatz_depth != source_round
    ):
        raise ValueError("resume checkpoint is not a finalized singleton outer round")
    for expected_round, row in enumerate(adapt.get("history", []), start=1):
        if not isinstance(row, Mapping) or int(row.get("depth", -1)) != expected_round:
            raise ValueError("resume history is not a consecutive cumulative prefix")

    expected_scalars = {
        "problem": "hh",
        "L": 2,
        "n_ph_max": EXPECTED_N_PH,
        "route_family": "singleton_response_snake",
        "sr_route_profile_resolved": SR_PROFILE_RESOLVED,
        "sr_route_profile_contract_sha256": SR_CONTRACT_SHA256,
    }
    for key, expected in expected_scalars.items():
        if settings.get(key) != expected:
            raise ValueError(f"resume settings drift: {key}")
    for key, expected in {
        "u": 0.25,
        "g_ep": 0.790569415042,
        "t": 1.0,
        "omega0": 1.0,
    }.items():
        if abs(float(settings.get(key, float("nan"))) - expected) > 1.0e-12:
            raise ValueError(f"resume physics drift: {key}")
    if settings.get("sr_route_profile_request") not in {
        SR_PROFILE_REQUEST,
        SR_PROFILE_RESOLVED,
    }:
        raise ValueError("resume SR profile request drift")
    expected_outer = "off" if mode == CONTROL_MODE else OUTER_PROFILE
    if str(settings.get("formal_manifold_route_profile", "off")) != expected_outer:
        raise ValueError("resume outer-information profile drift")
    composition = settings.get("formal_manifold_route_composition")
    if isinstance(composition, Mapping):
        required_composition = {
            "route_family": "singleton_response_snake",
            "phase1_prune_enabled": False,
            "phase2_enable_batching": False,
            "phase3_enable_batching": False,
            "structural_rollback_enabled": False,
            "sr_controller_contract_sha256": SR_CONTRACT_SHA256,
        }
        for key, expected in required_composition.items():
            if composition.get(key) != expected:
                raise ValueError(f"resume route-composition drift: {key}")
    elif mode == REUSE_MODE:
        raise ValueError("reuse checkpoint lacks its formal route composition")

    pointer = adapt.get("estimator_call_ledger_checkpoint")
    checkpoint_pointer = checkpoint.get("estimator_call_ledger_checkpoint")
    if not isinstance(pointer, Mapping) or dict(pointer) != checkpoint_pointer:
        raise ValueError("resume estimator-ledger checkpoint pointer disagrees")
    pointer = dict(pointer)
    ledger_name = str(pointer.get("path", ""))
    if (
        pointer.get("schema")
        != "paper_i_estimator_call_ledger_checkpoint_pointer_v1"
        or pointer.get("enabled") is not True
        or pointer.get("status") != "complete"
        or pointer.get("current_round_finalized") is not True
        or int(pointer.get("checkpoint_depth", -1)) != source_round
        or pointer.get("checkpoint_reason") != "iteration_done"
        or not ledger_name
        or Path(ledger_name).name != ledger_name
    ):
        raise ValueError("resume estimator-ledger pointer is incomplete")
    resolved_ledger = current_path.with_name(ledger_name) if ledger_path is None else ledger_path
    if resolved_ledger.name != ledger_name or not resolved_ledger.is_file():
        raise ValueError("resume estimator-ledger checkpoint sidecar is missing")
    observed_ledger_sha = sha256(resolved_ledger)
    if observed_ledger_sha != str(pointer.get("sha256", "")):
        raise ValueError("resume estimator-ledger checkpoint hash mismatch")
    ledger = load_json(resolved_ledger)
    ledger_checkpoint = ledger.get("checkpoint")
    ledger_payload = ledger.get("ledger")
    if (
        ledger.get("schema")
        != "paper_i_estimator_call_ledger_checkpoint_sidecar_v1"
        or ledger.get("no_credentials_serialized") is not True
        or not isinstance(ledger_checkpoint, Mapping)
        or not isinstance(ledger_payload, Mapping)
        or int(ledger_checkpoint.get("depth", -1)) != source_round
        or ledger_checkpoint.get("reason") != "iteration_done"
        or ledger_checkpoint.get("current_round_finalized") is not True
        or ledger_payload.get("schema") != pointer.get("ledger_schema")
        or ledger.get("ledger_fingerprint") != pointer.get("ledger_fingerprint")
        or int(ledger.get("unique_primitive_count", -1))
        != int(pointer.get("unique_primitive_count", -2))
        or int(ledger.get("raw_occurrence_count", -1))
        != int(pointer.get("raw_occurrence_count", -2))
        or int(ledger.get("S_alg", -1)) != int(pointer.get("S_alg", -2))
    ):
        raise ValueError("resume estimator-ledger checkpoint metadata drift")
    if mode == REUSE_MODE:
        outer_checkpoint = adapt.get("sr_outer_information_checkpoint")
        formal_checkpoint = adapt.get("formal_manifold_outer_information_checkpoint")
        if (
            not isinstance(outer_checkpoint, Mapping)
            or dict(outer_checkpoint) != formal_checkpoint
            or adapt.get("sr_outer_information_resume_policy")
            != "validated_outer_geometry_transport_restore_v1"
            or adapt.get("formal_manifold_outer_information_resume_policy")
            != "validated_outer_geometry_transport_restore_v1"
        ):
            raise ValueError("reuse checkpoint lacks matching outer-geometry state")
        outer_values = dict(outer_checkpoint)
        supplied_outer_sha = str(outer_values.pop("checkpoint_sha256", ""))
        encoded_outer = json.dumps(
            outer_values,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        ).encode("utf-8")
        stage = str(outer_values.get("stage", ""))
        pending = outer_values.get("pending")
        pending_source = outer_values.get("pending_source")
        failed_closure = outer_values.get("failed_closure")
        stage_shape_valid = {
            "idle": (
                pending is None
                and pending_source is None
                and failed_closure is None
            ),
            "same_ray_growth": (
                isinstance(pending, Mapping)
                and isinstance(pending_source, Mapping)
                and failed_closure is None
            ),
            "refit_prediction": (
                isinstance(pending, Mapping)
                and isinstance(pending_source, Mapping)
                and failed_closure is None
            ),
            "reanchor_required": (
                isinstance(pending, Mapping)
                and isinstance(pending_source, Mapping)
                and isinstance(failed_closure, Mapping)
            ),
            "closure_passed": (
                isinstance(pending, Mapping)
                and (
                    pending_source is None
                    or isinstance(pending_source, Mapping)
                )
                and failed_closure is None
            ),
        }.get(stage, False)
        if (
            outer_values.get("schema")
            != "formal_manifold_outer_information_checkpoint_v2"
            or not stage_shape_valid
            or supplied_outer_sha
            != hashlib.sha256(encoded_outer).hexdigest()
        ):
            raise ValueError(
                "reuse outer-geometry checkpoint is not hash-valid or stage-consistent"
            )
        immutable_ids = list(outer_values.get("immutable_primitive_ids", []))
        if (
            not immutable_ids
            or immutable_ids != sorted(set(str(value) for value in immutable_ids))
        ):
            raise ValueError("reuse outer-geometry immutable primitive IDs drift")
        ledger_primitive_ids = {
            str(entry.get("primitive_id", ""))
            for entry in ledger_payload.get("entries", [])
            if isinstance(entry, Mapping)
        }
        if not set(immutable_ids).issubset(ledger_primitive_ids):
            raise ValueError("reuse outer-geometry IDs do not close against the ledger")
    elif any(
        isinstance(adapt.get(key), Mapping)
        for key in (
            "sr_outer_information_checkpoint",
            "formal_manifold_outer_information_checkpoint",
        )
    ):
        raise ValueError("control checkpoint unexpectedly contains outer-geometry state")
    return {
        "schema": "paper_i_sr_outer_information_resume_current_validation_v1",
        "status": "pass",
        "mode": mode,
        "source_controller_round": source_round,
        "source_ansatz_depth": source_ansatz_depth,
        "target_controller_round": EXPECTED_TARGET_ROUND,
        "current_sha256": sha256(current_path),
        "ledger_name": ledger_name,
        "ledger_sha256": observed_ledger_sha,
        "ledger_fingerprint": str(pointer.get("ledger_fingerprint", "")),
        "S_alg_prefix": int(pointer.get("S_alg", -1)),
        "outer_information_profile": expected_outer,
    }


def pack_resume_checkpoint(
    mode: str,
    job_manifest_path: Path,
    *,
    work_root: Path | None = None,
    archive_path: Path | None = None,
) -> dict[str, Any]:
    """Atomically package the latest completed outer round for HTCondor."""

    root = repo_root() if work_root is None else work_root.resolve()
    expected_archive = checkpoint_archive_path(root, mode).resolve()
    archive = expected_archive if archive_path is None else archive_path.resolve()
    if archive != expected_archive:
        raise ValueError("checkpoint archive path is not mode-local and canonical")
    job = load_json(job_manifest_path)
    validate_job(job, expected_mode=mode)
    output_current_path = _resolved_job_path(root, job["paths"]["current_json"])
    restored_current_path = checkpoint_resume_dir(root, mode) / CHECKPOINT_CURRENT_NAME
    envelope_base = {
        "schema": CHECKPOINT_SCHEMA,
        "status": "complete",
        "bundle_id": BUNDLE_ID,
        "pair_id": PAIR_ID,
        "mode": mode,
        "checkpoint_exit_code": CHECKPOINT_EXIT_CODE,
        "job_manifest_sha256": sha256(job_manifest_path),
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_audit_sha256": SOURCE_AUDIT_SHA256,
        "source_runtime_revision": SOURCE_RUNTIME_REVISION,
        "scientific_command_digest": digest_jsonable(scientific_command_view(job)),
        "target_controller_round": EXPECTED_TARGET_ROUND,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "no_credentials_serialized": True,
    }
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for candidate in (output_current_path, restored_current_path):
        if not candidate.is_file():
            continue
        candidate_payload = load_json(candidate)
        candidate_pointer = candidate_payload.get("adapt_vqe", {}).get(
            "estimator_call_ledger_checkpoint", {}
        )
        candidate_ledger = candidate.with_name(str(candidate_pointer.get("path", "")))
        candidates.append(
            (
                candidate,
                validate_resume_current(
                    job,
                    candidate,
                    ledger_path=candidate_ledger,
                ),
            )
        )
    if not candidates:
        envelope = {
            **envelope_base,
            "resume_available": False,
            "cold_start_reason": "no_completed_outer_round_checkpoint",
            "source_controller_round": 0,
            "source_ansatz_depth": 0,
        }
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{mode}_checkpoint_", dir=archive.parent
        ) as td:
            staging = Path(td)
            envelope_path = staging / CHECKPOINT_MANIFEST_NAME
            dump_json(envelope_path, envelope)
            temporary_archive = staging / archive.name
            with tarfile.open(temporary_archive, "w:gz") as handle:
                handle.add(
                    envelope_path,
                    arcname=CHECKPOINT_MANIFEST_NAME,
                    recursive=False,
                )
            os.replace(temporary_archive, archive)
        return {
            **envelope,
            "archive": str(archive),
            "archive_sha256": sha256(archive),
        }
    candidates.sort(key=lambda item: int(item[1]["source_controller_round"]))
    if len(candidates) == 2 and int(candidates[0][1]["source_controller_round"]) == int(
        candidates[1][1]["source_controller_round"]
    ):
        if (
            candidates[0][1]["current_sha256"]
            != candidates[1][1]["current_sha256"]
            or candidates[0][1]["ledger_sha256"]
            != candidates[1][1]["ledger_sha256"]
        ):
            raise ValueError("same-round output and restored checkpoints conflict")
    current_path, validation = candidates[-1]
    current = load_json(current_path)
    pointer = current.get("adapt_vqe", {}).get("estimator_call_ledger_checkpoint", {})
    ledger_name = str(pointer.get("path", ""))
    ledger_path = current_path.with_name(ledger_name)
    envelope = {
        **envelope_base,
        "resume_available": True,
        "current_member": CHECKPOINT_CURRENT_NAME,
        "current_sha256": validation["current_sha256"],
        "ledger_member": validation["ledger_name"],
        "ledger_sha256": validation["ledger_sha256"],
        "source_controller_round": validation["source_controller_round"],
        "source_ansatz_depth": validation["source_ansatz_depth"],
        "outer_information_profile": validation["outer_information_profile"],
    }
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{mode}_checkpoint_", dir=archive.parent) as td:
        staging = Path(td)
        envelope_path = staging / CHECKPOINT_MANIFEST_NAME
        dump_json(envelope_path, envelope)
        temporary_archive = staging / archive.name
        with tarfile.open(temporary_archive, "w:gz") as handle:
            handle.add(envelope_path, arcname=CHECKPOINT_MANIFEST_NAME, recursive=False)
            handle.add(current_path, arcname=CHECKPOINT_CURRENT_NAME, recursive=False)
            handle.add(ledger_path, arcname=validation["ledger_name"], recursive=False)
        if (
            sha256(current_path) != validation["current_sha256"]
            or sha256(ledger_path) != validation["ledger_sha256"]
        ):
            raise ValueError("checkpoint inputs changed during atomic packaging")
        os.replace(temporary_archive, archive)
    return {**envelope, "archive": str(archive), "archive_sha256": sha256(archive)}


def restore_resume_checkpoint(
    mode: str,
    job_manifest_path: Path,
    *,
    work_root: Path | None = None,
    archive_path: Path | None = None,
    destination: Path | None = None,
) -> dict[str, Any]:
    """Fail closed, clear node-local ephemera, and restore a mode-bound checkpoint."""

    root = repo_root() if work_root is None else work_root.resolve()
    expected_archive = checkpoint_archive_path(root, mode).resolve()
    archive = expected_archive if archive_path is None else archive_path.resolve()
    if archive != expected_archive or not archive.is_file():
        raise ValueError("canonical mode-local checkpoint archive is missing")
    expected_destination = checkpoint_resume_dir(root, mode).resolve()
    restore_dir = expected_destination if destination is None else destination.resolve()
    if restore_dir != expected_destination:
        raise ValueError("checkpoint restore directory is not mode-local and canonical")
    job = load_json(job_manifest_path)
    validate_job(job, expected_mode=mode)
    output_root = _resolved_job_path(root, job["paths"]["output_root"]).resolve()
    expected_output_root = (root / "raw_outputs" / BUNDLE_ID / mode).resolve()
    if output_root != expected_output_root:
        raise ValueError("job output root is not mode-local and canonical")
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        if any(not member.isfile() for member in members):
            raise ValueError("checkpoint archive may contain regular files only")
        names = [PurePosixPath(member.name) for member in members]
        if any(name.is_absolute() or ".." in name.parts or len(name.parts) != 1 for name in names):
            raise ValueError("checkpoint archive contains an unsafe path")
        manifest_members = [
            member for member in members if member.name == CHECKPOINT_MANIFEST_NAME
        ]
        if len(manifest_members) != 1:
            raise ValueError("checkpoint manifest is missing or duplicated")
        stream = handle.extractfile(manifest_members[0])
        if stream is None:
            raise ValueError("checkpoint manifest is unreadable")
        envelope_bytes = stream.read()
        envelope = json.loads(envelope_bytes.decode("utf-8"))
        if not isinstance(envelope, dict):
            raise ValueError("checkpoint manifest is not a JSON object")
        expected_envelope = {
            "schema": CHECKPOINT_SCHEMA,
            "status": "complete",
            "bundle_id": BUNDLE_ID,
            "pair_id": PAIR_ID,
            "mode": mode,
            "checkpoint_exit_code": CHECKPOINT_EXIT_CODE,
            "job_manifest_sha256": sha256(job_manifest_path),
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "source_audit_sha256": SOURCE_AUDIT_SHA256,
            "source_runtime_revision": SOURCE_RUNTIME_REVISION,
            "scientific_command_digest": digest_jsonable(scientific_command_view(job)),
            "target_controller_round": EXPECTED_TARGET_ROUND,
            "no_credentials_serialized": True,
        }
        for key, expected in expected_envelope.items():
            if envelope.get(key) != expected:
                raise ValueError(f"checkpoint envelope drift: {key}")
        if envelope.get("resume_available") is False:
            if (
                len(members) != 1
                or set(str(name) for name in names) != {CHECKPOINT_MANIFEST_NAME}
                or envelope.get("cold_start_reason")
                != "no_completed_outer_round_checkpoint"
                or int(envelope.get("source_controller_round", -1)) != 0
                or int(envelope.get("source_ansatz_depth", -1)) != 0
            ):
                raise ValueError("cold-start checkpoint sentinel drift")
            restore_dir.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=f".{mode}_restore_", dir=restore_dir.parent
            ) as td:
                staging = Path(td) / "payload"
                staging.mkdir()
                (staging / CHECKPOINT_MANIFEST_NAME).write_bytes(envelope_bytes)
                if output_root.exists():
                    shutil.rmtree(output_root)
                if restore_dir.exists():
                    shutil.rmtree(restore_dir)
                os.replace(staging, restore_dir)
            return {
                "schema": "paper_i_sr_outer_information_resume_current_validation_v1",
                "status": "pass",
                "mode": mode,
                "resume_available": False,
                "source_controller_round": 0,
                "source_ansatz_depth": 0,
                "target_controller_round": EXPECTED_TARGET_ROUND,
                "archive": str(archive),
                "archive_sha256": sha256(archive),
                "resume_current_json": None,
                "resume_ledger_json": None,
                "checkpoint_manifest_json": str(
                    restore_dir / CHECKPOINT_MANIFEST_NAME
                ),
            }
        if envelope.get("resume_available") is not True:
            raise ValueError("checkpoint resume availability is missing")
        ledger_member = str(envelope.get("ledger_member", ""))
        expected_members = {
            CHECKPOINT_MANIFEST_NAME,
            CHECKPOINT_CURRENT_NAME,
            ledger_member,
        }
        if set(str(name) for name in names) != expected_members or len(members) != 3:
            raise ValueError("checkpoint archive member set drift")
        if (
            Path(ledger_member).name != ledger_member
            or not ledger_member.startswith("current.estimator_call_ledger_checkpoint.")
            or not ledger_member.endswith(".json")
            or envelope.get("current_member") != CHECKPOINT_CURRENT_NAME
        ):
            raise ValueError("checkpoint ledger/current member name is invalid")
        restore_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{mode}_restore_", dir=restore_dir.parent
        ) as td:
            staging = Path(td) / "payload"
            staging.mkdir()
            for member in members:
                source = handle.extractfile(member)
                if source is None:
                    raise ValueError(f"checkpoint member is unreadable: {member.name}")
                with (staging / member.name).open("wb") as sink:
                    shutil.copyfileobj(source, sink)
            current_path = staging / CHECKPOINT_CURRENT_NAME
            ledger_path = staging / ledger_member
            if (
                sha256(current_path) != envelope.get("current_sha256")
                or sha256(ledger_path) != envelope.get("ledger_sha256")
            ):
                raise ValueError("checkpoint archive payload hash mismatch")
            validation = validate_resume_current(
                job,
                current_path,
                ledger_path=ledger_path,
            )
            if (
                int(envelope.get("source_controller_round", -1))
                != int(validation["source_controller_round"])
                or int(envelope.get("source_ansatz_depth", -1))
                != int(validation["source_ansatz_depth"])
                or envelope.get("outer_information_profile")
                != validation["outer_information_profile"]
            ):
                raise ValueError("checkpoint envelope/current provenance mismatch")
            if output_root.exists():
                shutil.rmtree(output_root)
            if restore_dir.exists():
                shutil.rmtree(restore_dir)
            os.replace(staging, restore_dir)
    return {
        **validation,
        "resume_available": True,
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "resume_current_json": str(restore_dir / CHECKPOINT_CURRENT_NAME),
        "resume_ledger_json": str(restore_dir / ledger_member),
        "checkpoint_manifest_json": str(restore_dir / CHECKPOINT_MANIFEST_NAME),
    }


def validate_source_lock(bundle: Path | None = None) -> dict[str, Any]:
    base = bundle_dir() if bundle is None else bundle
    archive = base / PACKAGED_SOURCE_ARCHIVE
    audit_path = base / PACKAGED_SOURCE_AUDIT
    if not audit_path.is_file() or sha256(audit_path) != SOURCE_AUDIT_SHA256:
        raise ValueError("v5r2 source audit missing or hash drifted")
    audit = load_json(audit_path)
    if (
        audit.get("status") != "passed"
        or audit.get("runtime", {}).get("archive_sha256") != SOURCE_ARCHIVE_SHA256
        or audit.get("source_identity", {}).get("route_family")
        != "singleton_response_snake"
        or audit.get("source_identity", {}).get("selector_profile")
        != SR_PROFILE_RESOLVED
        or audit.get("source_identity", {}).get("sr_controller_contract_sha256")
        != SR_CONTRACT_SHA256
    ):
        raise ValueError("v5r2 source audit contract drift")
    repair_ids = {
        str(record.get("repair_id"))
        for record in audit.get("repairs", [])
        if isinstance(record, Mapping)
    }
    if not REQUIRED_REPAIR_IDS.issubset(repair_ids):
        raise ValueError("v5r2 required exact-fallback repair is absent")
    regressions = set(audit.get("tests", {}).get("new_regressions", []))
    if (
        int(audit.get("tests", {}).get("failed", -1)) != 0
        or int(audit.get("tests", {}).get("passed", 0)) < 80
        or not REQUIRED_REGRESSIONS.issubset(regressions)
    ):
        raise ValueError("v5r2 regression evidence drift")
    query_state = audit.get("query_and_state_contract", {})
    for key, expected in REQUIRED_QUERY_STATE_CONTRACT.items():
        if query_state.get(key) is not expected:
            raise ValueError(f"v5r2 query/state contract drift: {key}")
    inventory = inspect_source_archive(archive)
    for relative, expected in KEY_SOURCE_SHA256.items():
        if audit.get("runtime", {}).get("key_source_sha256", {}).get(relative) != expected:
            raise ValueError(f"source audit key hash drift: {relative}")
    return inventory


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("inspect", "extract", "checkpoint-pack", "checkpoint-restore"),
    )
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    if args.command == "inspect":
        if len(args.arguments) != 1:
            parser.error("inspect requires ARCHIVE")
        print(json.dumps(inspect_source_archive(Path(args.arguments[0])), sort_keys=True))
    elif args.command == "extract":
        if len(args.arguments) != 2:
            parser.error("extract requires ARCHIVE DESTINATION")
        print(safe_extract_source(Path(args.arguments[0]), Path(args.arguments[1])))
    elif args.command == "checkpoint-pack":
        if len(args.arguments) != 2:
            parser.error("checkpoint-pack requires MODE JOB_MANIFEST")
        print(
            json.dumps(
                pack_resume_checkpoint(
                    args.arguments[0],
                    Path(args.arguments[1]),
                ),
                sort_keys=True,
            )
        )
    else:
        if len(args.arguments) != 2:
            parser.error("checkpoint-restore requires MODE JOB_MANIFEST")
        print(
            json.dumps(
                restore_resume_checkpoint(
                    args.arguments[0],
                    Path(args.arguments[1]),
                ),
                sort_keys=True,
            )
        )
