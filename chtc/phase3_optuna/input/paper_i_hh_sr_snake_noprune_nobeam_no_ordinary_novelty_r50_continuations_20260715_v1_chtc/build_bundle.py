#!/usr/bin/env python3
"""Build the source-locked round-30 -> round-50 SR-SNAKE continuations.

The builder never imports the live scientific tree into the execution archive.
It extracts the preserved 94c2... archive, verifies and applies the separately
reviewed no-beam checkpoint-resume patch, and then repacks a deterministic new
archive.  By default all four round-30 checkpoints are required.  ``--ready-only``
materializes only rows whose complete fetched evidence is present and records
every missing row as blocked; it never invents a checkpoint hash.

This file stages inputs only.  It does not submit or execute a scientific run.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


BUNDLE_ID = (
    "paper_i_hh_sr_snake_noprune_nobeam_no_ordinary_novelty_"
    "r50_continuations_20260715_v1_chtc"
)
BATCH_NAME = "paper-i-hh-sr-noprune-nobeam-nonovelty-r50-cont-20260715-v1"
BUNDLE_DIR = Path(__file__).resolve().parent
REPO = BUNDLE_DIR.parents[3]

PRIOR_BUNDLE_ID = (
    "paper_i_hh_sr_snake_noprune_nobeam_no_ordinary_novelty_"
    "five_20260715_v1_chtc"
)
PRIOR_BUNDLE = Path("chtc/phase3_optuna/input") / PRIOR_BUNDLE_ID
PRIOR_OUTPUT = Path("raw_outputs") / PRIOR_BUNDLE_ID
BASE_ARCHIVE = Path(
    "raw_outputs/"
    "paper_i_hh_sr_snake_weak_weak_undamped_no_prune_no_beam_"
    "no_ordinary_novelty_fallback_on_20260715/"
    "source_lock/source_tree_no_beam_ablation_v1.tar.gz"
)
BASE_ARCHIVE_SHA256 = (
    "94c2df6df22c6d277aefdd6559273d943e3724d476ecab6648c6dd11e1fd78c6"
)
PATCH_MANIFEST = BUNDLE_DIR / "source_lock/no_beam_resume_patch_manifest.json"
PATCH_FILE = BUNDLE_DIR / "source_lock/no_beam_verified_resume.patch"
LOCKED_ARCHIVE = BUNDLE_DIR / "source_locked.tar.gz"
IMAGE_PATH = Path("chtc/phase3_optuna/image.sif")
IMAGE_SHA256 = "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"

SOURCE_DEPTH = 30
TARGET_DEPTH = 50
MAX_NEW_ADMISSIONS = TARGET_DEPTH - SOURCE_DEPTH
ENERGY_REPLAY_TOLERANCE = 1.0e-8
STATE_REPLAY_TOLERANCE = 1.0e-12
MAX_RUNTIME_S = 259200
SIGNED_PREFIX_SCHEMA = "static_adapt_signed_active_prefix_resume_sidecar_v1"
SIGNED_PREFIX_CHECKPOINT_SCHEMA = "paper_i_signed_active_prefix_checkpoint_v1"
SIGNED_PREFIX_CANONICAL_NAME = "signed_active_prefix_checkpoint.json"

REGIMES: tuple[dict[str, Any], ...] = (
    {
        "slug": "strong_weak_u8",
        "u": 8.0,
        "lambda": 0.25,
        "g_ep": 0.353553390593,
        "n_ph_work": 2,
        "memory_mb": 32768,
        "disk_mb": 61440,
    },
    {
        "slug": "weak_strong",
        "u": 0.25,
        "lambda": 1.25,
        "g_ep": 0.790569415042,
        "n_ph_work": 4,
        "memory_mb": 40960,
        "disk_mb": 61440,
    },
    {
        "slug": "intermediate_strong",
        "u": 1.25,
        "lambda": 1.25,
        "g_ep": 0.790569415042,
        "n_ph_work": 4,
        "memory_mb": 40960,
        "disk_mb": 61440,
    },
    {
        "slug": "strong_strong_u8",
        "u": 8.0,
        "lambda": 1.25,
        "g_ep": 0.790569415042,
        "n_ph_work": 4,
        "memory_mb": 40960,
        "disk_mb": 61440,
    },
)

OUTPUT_PATH_FLAGS = {
    "--adapt-current-json",
    "--adapt-estimator-call-ledger-json",
    "--output-json",
}
CONTINUATION_FLAGS = {
    "--adapt-resume-scaffold-json",
    "--adapt-resume-mode",
    "--adapt-resume-boundary-refit-policy",
    "--adapt-segment-id",
    "--adapt-segment-target-depth",
    "--adapt-segment-target-controller-round",
    "--adapt-segment-max-new-admissions",
    "--adapt-resume-compile-smoke",
    "--adapt-resume-smoke-backend",
}
HORIZON_FLAGS = {"--adapt-max-depth"}
ALLOWED_EXECUTABLE_DIFF_FLAGS = (
    OUTPUT_PATH_FLAGS | CONTINUATION_FLAGS | HORIZON_FLAGS
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def options(argv: Sequence[str]) -> dict[str, Any]:
    prefix = ["python3", "-m", "pipelines.static_adapt.adapt_pipeline"]
    if list(argv[:3]) != prefix:
        raise ValueError("unexpected prior command prefix")
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
            raise ValueError(f"cannot replace boolean option: {flag}")
        argv[position + 1] = str(value)
        return
    insertion = argv.index("--adapt-current-json")
    argv[insertion:insertion] = [flag, str(value)]


def _safe_member_name(name: str) -> str:
    normalized = PurePosixPath(name.lstrip("./"))
    if normalized.is_absolute() or ".." in normalized.parts or not normalized.parts:
        raise ValueError(f"unsafe archive member: {name}")
    return normalized.as_posix()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _validate_patch_contract() -> dict[str, Any]:
    if not PATCH_MANIFEST.is_file() or not PATCH_FILE.is_file():
        raise FileNotFoundError(
            "verified no-beam resume patch is not frozen; required files are "
            f"{PATCH_MANIFEST.relative_to(REPO)} and {PATCH_FILE.relative_to(REPO)}"
        )
    manifest = json_load(PATCH_MANIFEST)
    if manifest.get("schema") != "paper_i_hh_sr_no_beam_resume_patch_v3":
        raise ValueError("unexpected no-beam resume patch schema")
    if manifest.get("status") != "verified":
        raise ValueError("no-beam resume patch is not marked verified")
    base_source = manifest.get("base_source_archive")
    if not isinstance(base_source, Mapping) or base_source.get("sha256") != BASE_ARCHIVE_SHA256:
        raise ValueError("no-beam resume patch targets a different base archive")
    patch_record = manifest.get("patch")
    if not isinstance(patch_record, Mapping) or patch_record.get("sha256") != sha256(PATCH_FILE):
        raise ValueError("no-beam resume patch SHA-256 mismatch")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("no-beam resume patch manifest has no file records")
    allowed = []
    for row in files:
        if not isinstance(row, Mapping):
            raise TypeError("patch file record must be an object")
        path = _safe_member_name(str(row.get("path", "")))
        if not path.startswith("pipelines/static_adapt/"):
            raise ValueError(f"patch escapes static-ADAPT source scope: {path}")
        for field in ("base_sha256", "patched_sha256"):
            value = str(row.get(field, ""))
            if len(value) != 64:
                raise ValueError(f"missing {field} for {path}")
        allowed.append(path)
    if len(set(allowed)) != len(allowed):
        raise ValueError("duplicate path in no-beam resume patch manifest")
    tests = manifest.get("verification", {})
    if not isinstance(tests, Mapping) or tests.get("status") != "pass":
        raise ValueError("no-beam resume patch lacks passing verification record")
    return manifest


def _extract_base_archive(root: Path) -> None:
    base = REPO / BASE_ARCHIVE
    if sha256(base) != BASE_ARCHIVE_SHA256:
        raise ValueError("base source archive SHA-256 mismatch")
    with tarfile.open(base, "r:gz") as archive:
        for member in archive.getmembers():
            if member.name in {".", "./"}:
                if not member.isdir():
                    raise ValueError("source archive root member is not a directory")
                continue
            _safe_member_name(member.name)
            if not member.isfile() and not member.isdir():
                raise ValueError(f"special source archive member: {member.name}")
        archive.extractall(root, filter="data")


def _apply_verified_patch(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    before = _tree_hashes(root)
    expected_files = {
        str(row["path"]): {
            "base_sha256": str(row["base_sha256"]),
            "patched_sha256": str(row["patched_sha256"]),
        }
        for row in manifest["files"]
    }
    base_mismatches = {
        path: {"expected": row["base_sha256"], "actual": before.get(path)}
        for path, row in expected_files.items()
        if before.get(path) != row["base_sha256"]
    }
    if base_mismatches:
        raise ValueError(f"patch base-file mismatch: {base_mismatches}")
    completed = subprocess.run(
        ["patch", "-p1", "--batch", "--forward", "-i", str(PATCH_FILE)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "verified no-beam resume patch did not apply cleanly: "
            f"{completed.stdout[-2000:]} {completed.stderr[-2000:]}"
        )
    after = _tree_hashes(root)
    changed = sorted(
        path
        for path in set(before) | set(after)
        if before.get(path) != after.get(path)
    )
    if changed != sorted(expected_files):
        raise ValueError(
            "no-beam patch changed an unexpected source file: "
            f"expected={sorted(expected_files)} actual={changed}"
        )
    patched_mismatches = {
        path: {"expected": row["patched_sha256"], "actual": after.get(path)}
        for path, row in expected_files.items()
        if after.get(path) != row["patched_sha256"]
    }
    if patched_mismatches:
        raise ValueError(f"patched source-file mismatch: {patched_mismatches}")
    return {
        "changed_files": changed,
        "file_hashes_before": {path: before[path] for path in changed},
        "file_hashes_after": {path: after[path] for path in changed},
        "patch_stdout": completed.stdout,
    }


def _write_deterministic_archive(root: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
                    relative = path.relative_to(root).as_posix()
                    if path.is_symlink() or (not path.is_file() and not path.is_dir()):
                        raise ValueError(f"unsupported source tree entry: {relative}")
                    info = tarfile.TarInfo(relative + ("/" if path.is_dir() else ""))
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mode = 0o755 if path.is_dir() or os.access(path, os.X_OK) else 0o644
                    if path.is_dir():
                        info.type = tarfile.DIRTYPE
                        archive.addfile(info)
                    else:
                        data = path.read_bytes()
                        info.size = len(data)
                        archive.addfile(info, io.BytesIO(data))
    temporary.replace(destination)


def build_source_archive() -> tuple[dict[str, Any], Path]:
    patch_manifest = _validate_patch_contract()
    with tempfile.TemporaryDirectory(prefix="sr_r50_source_") as temporary:
        source_root = Path(temporary) / "source"
        source_root.mkdir()
        _extract_base_archive(source_root)
        patch_audit = _apply_verified_patch(source_root, patch_manifest)
        _write_deterministic_archive(source_root, LOCKED_ARCHIVE)
        member_hashes = _tree_hashes(source_root)
    inventory = {
        "schema": "paper_i_hh_sr_r50_patched_source_archive_v1",
        "created_utc": utc_now(),
        "archive_path": LOCKED_ARCHIVE.relative_to(REPO).as_posix(),
        "archive_sha256": sha256(LOCKED_ARCHIVE),
        "archive_size_bytes": LOCKED_ARCHIVE.stat().st_size,
        "base_archive": {
            "path": BASE_ARCHIVE.as_posix(),
            "sha256": BASE_ARCHIVE_SHA256,
            "preserved_unchanged": True,
        },
        "patch": {
            "path": PATCH_FILE.relative_to(REPO).as_posix(),
            "sha256": sha256(PATCH_FILE),
            "manifest": PATCH_MANIFEST.relative_to(REPO).as_posix(),
            "manifest_sha256": sha256(PATCH_MANIFEST),
            **patch_audit,
        },
        "member_count": len(member_hashes),
        "patched_member_hashes": {
            path: member_hashes[path] for path in patch_audit["changed_files"]
        },
        "live_scientific_tree_imported": False,
    }
    json_dump(BUNDLE_DIR / "source_archive_manifest.json", inventory)
    return inventory, LOCKED_ARCHIVE


def source_paths(slug: str) -> dict[str, Path]:
    root = REPO / PRIOR_OUTPUT / slug
    return {
        "prior_job_manifest": REPO / PRIOR_BUNDLE / "jobs" / f"{slug}.json",
        "checkpoint": root / "json/current.json",
        "source_ledger": root / "json/estimator_call_ledger.json",
        "result": root / "json/result.json",
        "execution": root / "execution.json",
        "normalized_manifest": root / "normalized_run_manifest.json",
    }


def row_readiness(regime: Mapping[str, Any]) -> tuple[bool, list[str]]:
    paths = source_paths(str(regime["slug"]))
    missing = [
        path.relative_to(REPO).as_posix()
        for path in paths.values()
        if not path.is_file()
    ]
    return not missing, missing


def _checkpoint_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    adapt = payload.get("adapt_vqe")
    checkpoint = payload.get("checkpoint")
    if not isinstance(adapt, Mapping) or not isinstance(checkpoint, Mapping):
        raise ValueError("resume checkpoint lacks adapt_vqe/checkpoint objects")
    history = adapt.get("history")
    if not isinstance(history, list):
        raise ValueError("resume checkpoint lacks complete history array")
    final_refit = adapt.get("final_full_refit")
    if not isinstance(final_refit, Mapping):
        raise ValueError("resume checkpoint lacks final-refit telemetry")
    metadata = {
        "checkpoint_reason": checkpoint.get("reason"),
        "checkpoint_complete": checkpoint.get("complete"),
        "checkpoint_beam_enabled": checkpoint.get("beam_enabled"),
        "checkpoint_branch_policy": checkpoint.get("checkpoint_branch_policy"),
        "ansatz_depth": int(adapt.get("ansatz_depth", -1)),
        "history_count": len(history),
        "history_count_recorded": int(adapt.get("history_count", -1)),
        "history_checkpoint_complete": adapt.get("history_checkpoint_complete"),
        "adapt_beam_enabled": adapt.get("adapt_beam_enabled"),
        "partial_checkpoint": adapt.get("partial_checkpoint"),
        "saved_energy": float(adapt["energy"]),
        "final_full_refit_attempted": final_refit.get("attempted"),
        "final_full_refit_executed": final_refit.get("executed"),
        "final_full_refit_skipped_reason": final_refit.get("skipped_reason"),
    }
    expected = {
        "checkpoint_reason": "iteration_done",
        "checkpoint_complete": False,
        "checkpoint_beam_enabled": False,
        "ansatz_depth": SOURCE_DEPTH,
        "history_count": SOURCE_DEPTH,
        "history_count_recorded": SOURCE_DEPTH,
        "history_checkpoint_complete": True,
        "adapt_beam_enabled": False,
        "partial_checkpoint": True,
        "final_full_refit_executed": False,
        "final_full_refit_attempted": False,
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"round-30 pre-terminal checkpoint mismatch: {mismatches}")
    return metadata


def _frozen_checkpoint_validation(
    source_tree: Path,
    checkpoint: Path,
    source_ledger: Path,
    signed_prefix_sidecar: Path,
    expected_options: Mapping[str, Any],
) -> dict[str, Any]:
    probe = r'''
import json, os, sys
import numpy as np
from pathlib import Path
from pipelines.static_adapt.resume_scaffold import (
    _load_verified_active_prefix_sidecar,
    extract_verified_singleton_resume_checkpoint,
    load_static_resume_source,
    run_resume_compile_smoke,
    validate_static_hh_resume_source,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import compile_polynomial_action, energy_via_one_apply

checkpoint = Path(sys.argv[1])
source = load_static_resume_source(checkpoint)
validation = validate_static_hh_resume_source(source, continuation_mode="phase3_v1")
verified = extract_verified_singleton_resume_checkpoint(source)
signed_prefix = _load_verified_active_prefix_sidecar(source)
runtime = source.runtime_input
layout = runtime.base_layout
executor = CompiledAnsatzExecutor(
    list(runtime.selected_terms),
    coefficient_tolerance=float(layout.coefficient_tolerance),
    ignore_identity=bool(layout.ignore_identity),
    sort_terms=str(layout.term_order).lower() == "sorted",
    parameterization_mode="logical_shared",
    parameterization_layout=layout,
)
state = executor.prepare_state(runtime.theta_logical, runtime.psi_ref)
energy, _ = energy_via_one_apply(
    state,
    compile_polynomial_action(runtime.resolved_problem.hamiltonian),
)
signed_executor = CompiledAnsatzExecutor(
    list(signed_prefix["terms"]),
    coefficient_tolerance=float(layout.coefficient_tolerance),
    ignore_identity=bool(layout.ignore_identity),
    sort_terms=str(layout.term_order).lower() == "sorted",
    parameterization_mode="logical_shared",
    parameterization_layout=layout,
)
signed_state = signed_executor.prepare_state(runtime.theta_logical, runtime.psi_ref)
signed_energy, _ = energy_via_one_apply(
    signed_state,
    compile_polynomial_action(runtime.resolved_problem.hamiltonian),
)
saved = float(source.payload["adapt_vqe"]["energy"])
smoke = run_resume_compile_smoke(
    source,
    mode="required",
    backend_name="FakeMarrakesh",
    seed_transpiler=7,
    optimization_level=1,
)
payload = {
    "saved_energy": saved,
    "replayed_energy": float(energy),
    "energy_abs_discrepancy": abs(float(energy) - saved),
    "state_l2_discrepancy": float(np.linalg.norm(state - runtime.psi_initial)),
    "signed_prefix_replayed_energy": float(signed_energy),
    "signed_prefix_energy_abs_discrepancy": abs(float(signed_energy) - saved),
    "signed_prefix_state_l2_discrepancy": float(
        np.linalg.norm(signed_state - runtime.psi_initial)
    ),
    "legacy_vs_signed_prefix_state_l2_discrepancy": float(
        np.linalg.norm(state - signed_state)
    ),
    "loader_validation": validation,
    "verified_checkpoint": {
        "source_ansatz_depth": int(verified.ansatz_depth),
        "source_controller_round": int(verified.controller_round),
        "history_count": len(verified.history),
        "restored_S_alg": int(
            verified.estimator_call_ledger_provenance["restored_prefix_S_alg"]
        ),
        "restored_occurrence_count": int(
            verified.estimator_call_ledger_provenance[
                "restored_prefix_occurrence_count"
            ]
        ),
    },
    "signed_active_prefix": signed_prefix["provenance"],
    "compile_smoke": smoke.to_payload(),
}
print("__SR_R50_VALIDATION__" + json.dumps(payload, sort_keys=True))
'''
    with tempfile.TemporaryDirectory(prefix="sr_r50_checkpoint_") as temporary:
        copied = Path(temporary) / "current.json"
        shutil.copy2(checkpoint, copied)
        shutil.copy2(source_ledger, copied.with_name("estimator_call_ledger.json"))
        shutil.copy2(
            signed_prefix_sidecar,
            copied.with_name("signed_active_prefix_checkpoint.json"),
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(source_tree)
        completed = subprocess.run(
            [sys.executable, "-c", probe, str(copied)],
            cwd=source_tree,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
    marker = "__SR_R50_VALIDATION__"
    lines = [line for line in completed.stdout.splitlines() if marker in line]
    if completed.returncode != 0 or not lines:
        raise RuntimeError(
            "frozen checkpoint parity/compile-smoke validation failed: "
            f"rc={completed.returncode} stdout={completed.stdout[-3000:]} "
            f"stderr={completed.stderr[-3000:]}"
        )
    payload = json.loads(lines[-1].split(marker, 1)[1])
    if float(payload["energy_abs_discrepancy"]) > ENERGY_REPLAY_TOLERANCE:
        raise ValueError("checkpoint energy replay discrepancy exceeds tolerance")
    if float(payload["state_l2_discrepancy"]) > STATE_REPLAY_TOLERANCE:
        raise ValueError("checkpoint state replay discrepancy exceeds tolerance")
    if float(payload["signed_prefix_energy_abs_discrepancy"]) > ENERGY_REPLAY_TOLERANCE:
        raise ValueError("signed-prefix energy replay discrepancy exceeds tolerance")
    if float(payload["signed_prefix_state_l2_discrepancy"]) > STATE_REPLAY_TOLERANCE:
        raise ValueError("signed-prefix state replay discrepancy exceeds tolerance")
    verified = payload["verified_checkpoint"]
    if int(verified["source_ansatz_depth"]) != SOURCE_DEPTH:
        raise ValueError("verified resume source depth is not 30")
    if int(verified["source_controller_round"]) != SOURCE_DEPTH:
        raise ValueError("verified resume source controller round is not 30")
    if not bool(payload["compile_smoke"].get("success")):
        raise ValueError("required FakeMarrakesh resume compile smoke did not pass")
    signed_prefix = payload.get("signed_active_prefix", {})
    if (
        not isinstance(signed_prefix, Mapping)
        or signed_prefix.get("schema")
        != "verified_resume_signed_active_prefix_sidecar_v1"
        or int(signed_prefix.get("outer_iteration", -1)) != SOURCE_DEPTH
        or int(signed_prefix.get("operator_count", -1)) != SOURCE_DEPTH
    ):
        raise ValueError("signed active-prefix sidecar verification did not pass")
    payload["status"] = "pass"
    payload["energy_tolerance"] = ENERGY_REPLAY_TOLERANCE
    payload["state_tolerance"] = STATE_REPLAY_TOLERANCE
    payload["expected_source_options"] = {
        key: expected_options[key]
        for key in ("--u", "--g-ep", "--n-ph-max", "--adapt-max-depth")
    }
    return payload


def _unpack_patched_archive() -> tempfile.TemporaryDirectory[str]:
    temporary = tempfile.TemporaryDirectory(prefix="sr_r50_patched_")
    root = Path(temporary.name)
    with tarfile.open(LOCKED_ARCHIVE, "r:gz") as archive:
        archive.extractall(root, filter="data")
    setattr(temporary, "source_root", root)
    return temporary


def _deterministic_gzip_copy(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with source.open("rb") as source_handle, temporary.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            compresslevel=9,
            mtime=0,
        ) as zipped:
            shutil.copyfileobj(source_handle, zipped, length=1024 * 1024)
    temporary.replace(destination)
    with gzip.open(destination, "rb") as zipped:
        digest = hashlib.sha256()
        size = 0
        for chunk in iter(lambda: zipped.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    if digest.hexdigest() != sha256(source) or size != source.stat().st_size:
        raise ValueError(f"compressed input round-trip mismatch: {source}")
    return destination


def _copy_source_checkpoint(slug: str, source: Path) -> Path:
    destination = (
        BUNDLE_DIR / "resume_inputs" / f"{slug}.round30.current.json.gz"
    )
    return _deterministic_gzip_copy(source, destination)


def _copy_source_ledger(slug: str, source: Path) -> Path:
    destination = (
        BUNDLE_DIR
        / "resume_inputs"
        / f"{slug}.round30.estimator_call_ledger.json.gz"
    )
    return _deterministic_gzip_copy(source, destination)


def _signed_prefix_checkpoint_sha256(checkpoint: Mapping[str, Any]) -> str:
    unsigned = dict(checkpoint)
    embedded = unsigned.pop("checkpoint_sha256", None)
    if not isinstance(embedded, str) or len(embedded) != 64:
        raise ValueError("signed-prefix checkpoint lacks embedded SHA-256")
    computed = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if computed != embedded:
        raise ValueError(
            "signed-prefix embedded checkpoint SHA-256 mismatch: "
            f"expected={embedded} actual={computed}"
        )
    return computed


def _jsonable_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _extract_signed_prefix_sidecar(slug: str, result_path: Path) -> tuple[Path, dict[str, Any]]:
    """Extract the exact round-30 post-admission/prune checkpoint.

    The full result remains an authenticated local source artifact and is never
    transferred to CHTC.  ``jq`` is used only to select the typed object; the
    embedded canonical JSON digest proves the object was not altered.
    """

    filter_expression = (
        '[.adapt_vqe.active_prefix_checkpoints[] '
        f'| select(.schema == "{SIGNED_PREFIX_CHECKPOINT_SCHEMA}" '
        f'and .checkpoint_kind == "post_admission_prune" '
        f'and .outer_iteration == {SOURCE_DEPTH} '
        f'and .active_ansatz_depth == {SOURCE_DEPTH})] '
        '| if length == 1 then .[0] '
        'else error("expected exactly one round-30 signed-prefix checkpoint") end'
    )
    completed = subprocess.run(
        ["/usr/bin/jq", "-c", filter_expression, str(result_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{slug} signed-prefix extraction failed: {completed.stderr[-2000:]}"
        )
    checkpoint = json.loads(completed.stdout)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{slug} signed-prefix checkpoint is not an object")
    embedded_sha256 = _signed_prefix_checkpoint_sha256(checkpoint)
    snapshot_filter = r'''
{
  history_row: (
    .adapt_vqe.history[-1]
    | {
        depth,
        drop_policy_enabled,
        drop_plateau_hits,
        stage_name,
        stage_transition_reason
      }
  ),
  controller_snapshots: [
    .adapt_vqe.history[-1].selected_feature_rows[]?
    | .controller_snapshot
    | select(type == "object")
  ],
  selection_evidence: {
    pool_size: .adapt_vqe.pool_size,
    rows: [
      .adapt_vqe.history[]
      | {
          selected_feature_row_count: (.selected_feature_rows | length),
          candidate_pool_index: .selected_feature_rows[0].candidate_pool_index
        }
    ]
  }
}
'''
    snapshot_completed = subprocess.run(
        ["/usr/bin/jq", "-c", snapshot_filter, str(result_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if snapshot_completed.returncode != 0:
        raise RuntimeError(
            f"{slug} controller-snapshot extraction failed: "
            f"{snapshot_completed.stderr[-2000:]}"
        )
    snapshot_evidence = json.loads(snapshot_completed.stdout)
    if not isinstance(snapshot_evidence, dict):
        raise ValueError(f"{slug} controller-snapshot evidence is not an object")
    snapshot_rows = snapshot_evidence.get("controller_snapshots")
    history_evidence_raw = snapshot_evidence.get("history_row")
    selection_evidence_raw = snapshot_evidence.get("selection_evidence")
    if (
        not isinstance(snapshot_rows, list)
        or len(snapshot_rows) != 1
        or not isinstance(history_evidence_raw, dict)
        or not isinstance(selection_evidence_raw, dict)
    ):
        raise ValueError(f"{slug} has no round-30 controller snapshot")
    unique_snapshots = {
        _jsonable_sha256(row): row
        for row in snapshot_rows
        if isinstance(row, dict)
    }
    if len(unique_snapshots) != 1:
        raise ValueError(
            f"{slug} requires exactly one unique round-30 controller snapshot; "
            f"found {len(unique_snapshots)}"
        )
    controller_snapshot_sha256, controller_snapshot = next(
        iter(unique_snapshots.items())
    )
    if (
        controller_snapshot.get("snapshot_version")
        != "phase123_controller_maturity_v2"
        or int(controller_snapshot.get("step_index", -1)) != SOURCE_DEPTH - 1
    ):
        raise ValueError(f"{slug} controller snapshot is not the round-30 maturity-v2 state")
    source_history_row_evidence = {
        "depth": int(history_evidence_raw.get("depth", -1)),
        "drop_policy_enabled": history_evidence_raw.get("drop_policy_enabled"),
        "drop_plateau_hits": int(history_evidence_raw.get("drop_plateau_hits", -1)),
        "stage_name": str(history_evidence_raw.get("stage_name", "")),
        "stage_transition_reason": str(
            history_evidence_raw.get("stage_transition_reason", "")
        ),
        "controller_snapshot_count": len(snapshot_rows),
        "selected_feature_row_index": 0,
    }
    expected_history_evidence = {
        "depth": SOURCE_DEPTH,
        "drop_policy_enabled": False,
        "drop_plateau_hits": 0,
        "stage_name": "core",
        "stage_transition_reason": "stay_core",
        "controller_snapshot_count": 1,
        "selected_feature_row_index": 0,
    }
    if source_history_row_evidence != expected_history_evidence:
        raise ValueError(
            f"{slug} controller-history evidence drift: "
            f"{source_history_row_evidence}"
        )
    source_history_row_evidence_sha256 = _jsonable_sha256(
        source_history_row_evidence
    )
    controller_state = {
        "schema": "static_adapt_singleton_controller_resume_state_v1",
        "controller_round": SOURCE_DEPTH,
        "source_max_depth": SOURCE_DEPTH,
        "phase1_residual_opened": False,
        "phase1_stage_name": "core",
        "source_history_row_evidence": source_history_row_evidence,
        "source_history_row_evidence_sha256": (
            source_history_row_evidence_sha256
        ),
    }
    selection_rows = selection_evidence_raw.get("rows")
    pool_size = int(selection_evidence_raw.get("pool_size", -1))
    if (
        not isinstance(selection_rows, list)
        or len(selection_rows) != SOURCE_DEPTH
        or pool_size <= 0
        or any(not isinstance(row, dict) for row in selection_rows)
    ):
        raise ValueError(f"{slug} selection-count evidence is incomplete")
    selected_feature_row_count_per_round = [
        int(row.get("selected_feature_row_count", -1)) for row in selection_rows
    ]
    ordered_parent_pool_indices = [
        int(row.get("candidate_pool_index", -1)) for row in selection_rows
    ]
    if (
        selected_feature_row_count_per_round != [1] * SOURCE_DEPTH
        or any(index < 0 or index >= pool_size for index in ordered_parent_pool_indices)
    ):
        raise ValueError(f"{slug} selection-count/index evidence drift")
    ordered_logical_candidate_indices: list[int] = []
    selection_state = {
        "schema": "static_adapt_singleton_selection_count_resume_state_v1",
        "controller_round": SOURCE_DEPTH,
        "pool_size": pool_size,
        "seq2p_logical_mode": False,
        "ordered_parent_pool_indices": ordered_parent_pool_indices,
        "ordered_parent_pool_indices_sha256": _jsonable_sha256(
            ordered_parent_pool_indices
        ),
        "selected_feature_row_count_per_round": (
            selected_feature_row_count_per_round
        ),
        "ordered_logical_candidate_indices": ordered_logical_candidate_indices,
        "ordered_logical_candidate_indices_sha256": _jsonable_sha256(
            ordered_logical_candidate_indices
        ),
    }
    sidecar = {
        "schema": SIGNED_PREFIX_SCHEMA,
        "source_result_json": result_path.relative_to(REPO).as_posix(),
        "source_result_sha256": sha256(result_path),
        "checkpoint": checkpoint,
        "controller_snapshot": controller_snapshot,
        "controller_snapshot_sha256": controller_snapshot_sha256,
        "controller_state": controller_state,
        "selection_state": selection_state,
    }
    destination = (
        BUNDLE_DIR
        / "resume_inputs"
        / f"{slug}.round30.{SIGNED_PREFIX_CANONICAL_NAME}"
    )
    json_dump(destination, sidecar)
    audit = {
        "schema": SIGNED_PREFIX_SCHEMA,
        "source_result_json": sidecar["source_result_json"],
        "source_result_sha256": sidecar["source_result_sha256"],
        "sidecar_path": destination.relative_to(REPO).as_posix(),
        "sidecar_sha256": sha256(destination),
        "checkpoint_schema": checkpoint["schema"],
        "checkpoint_kind": checkpoint["checkpoint_kind"],
        "outer_iteration": int(checkpoint["outer_iteration"]),
        "active_ansatz_depth": int(checkpoint["active_ansatz_depth"]),
        "embedded_checkpoint_sha256": embedded_sha256,
        "parameterization_layout_sha256": _jsonable_sha256(
            checkpoint["parameterization"]
        ),
        "controller_snapshot_version": controller_snapshot["snapshot_version"],
        "controller_snapshot_step_index": int(controller_snapshot["step_index"]),
        "controller_snapshot_sha256": controller_snapshot_sha256,
        "controller_snapshot_unique_source_count": 1,
        "controller_state_sha256": _jsonable_sha256(controller_state),
        "source_history_row_evidence_sha256": (
            source_history_row_evidence_sha256
        ),
        "selection_state_sha256": _jsonable_sha256(selection_state),
        "ordered_parent_pool_indices_sha256": selection_state[
            "ordered_parent_pool_indices_sha256"
        ],
        "full_result_transferred": False,
    }
    return destination, audit


def _build_execution_argv(source_argv: list[str], slug: str) -> tuple[list[str], dict[str, str]]:
    output_root = Path("raw_outputs") / BUNDLE_ID / slug
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
        "resume_input_json": (
            output_root / "resume_input/round30_current.json"
        ).as_posix(),
        "resume_input_ledger_json": (
            output_root / "resume_input/estimator_call_ledger.json"
        ).as_posix(),
        "resume_input_signed_prefix_json": (
            output_root / f"resume_input/{SIGNED_PREFIX_CANONICAL_NAME}"
        ).as_posix(),
        "source_ledger_record_json": (
            output_root / "resume_input/source_round30_records.json"
        ).as_posix(),
    }
    execution = list(source_argv)
    set_option(execution, "--adapt-max-depth", str(TARGET_DEPTH))
    set_option(execution, "--adapt-current-json", paths["current_json"])
    set_option(
        execution,
        "--adapt-estimator-call-ledger-json",
        paths["estimator_call_ledger_json"],
    )
    set_option(execution, "--output-json", paths["result_json"])
    additions = {
        "--adapt-resume-scaffold-json": paths["resume_input_json"],
        "--adapt-resume-mode": "scaffold_v1",
        "--adapt-resume-boundary-refit-policy": "verified_checkpoint_no_refit_v1",
        "--adapt-segment-id": f"{slug}-r30-to-r50-v1",
        "--adapt-segment-target-depth": str(TARGET_DEPTH),
        "--adapt-segment-target-controller-round": str(TARGET_DEPTH),
        "--adapt-segment-max-new-admissions": str(MAX_NEW_ADMISSIONS),
        "--adapt-resume-compile-smoke": "required",
        "--adapt-resume-smoke-backend": "FakeMarrakesh",
    }
    for flag, value in additions.items():
        set_option(execution, flag, value)
    return execution, paths


def _build_environment(source_environment: Mapping[str, Any], slug: str) -> tuple[dict[str, str], list[dict[str, str | None]]]:
    output_root = Path("raw_outputs") / BUNDLE_ID / slug
    environment = {str(key): str(value) for key, value in source_environment.items()}
    replacements = {
        "MPLCONFIGDIR": (output_root / "cache/matplotlib").as_posix(),
        "STATIC_ADAPT_CANDIDATE_RECORD_CACHE_DIR": (
            output_root / "cache/candidate_records"
        ).as_posix(),
        "STATIC_ADAPT_HH_GENERATOR_REGISTRY_CACHE_DIR": (
            output_root / "cache/hh_generator_registry"
        ).as_posix(),
        "STATIC_ADAPT_HH_POOL_CACHE_DIR": (output_root / "cache/hh_pool").as_posix(),
    }
    for key, value in replacements.items():
        if key not in environment:
            raise ValueError(f"source environment lacks required cache field: {key}")
        environment[key] = value
    allowed = set(replacements)
    differences = [
        {
            "field": key,
            "source": (
                None if source_environment.get(key) is None else str(source_environment[key])
            ),
            "target": environment.get(key),
            "classification": "isolated_operational_output_or_cache_path",
        }
        for key in sorted(set(source_environment) | set(environment))
        if (None if source_environment.get(key) is None else str(source_environment[key]))
        != environment.get(key)
    ]
    if {str(row["field"]) for row in differences} != allowed:
        raise ValueError(f"unexpected environment drift: {differences}")
    return environment, differences


def _build_job(
    regime: Mapping[str, Any],
    archive_inventory: Mapping[str, Any],
    source_tree: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    slug = str(regime["slug"])
    paths = source_paths(slug)
    prior_job = json_load(paths["prior_job_manifest"])
    source_argv = [str(token) for token in prior_job["command"]["execution_argv"]]
    source_options = options(source_argv)
    expected_regime = {
        "--u": float(regime["u"]),
        "--g-ep": float(regime["g_ep"]),
        "--n-ph-max": int(regime["n_ph_work"]),
        "--adapt-max-depth": SOURCE_DEPTH,
    }
    mismatches = {
        flag: {"expected": expected, "actual": source_options.get(flag)}
        for flag, expected in expected_regime.items()
        if source_options.get(flag) is None or float(source_options[flag]) != float(expected)
    }
    if mismatches:
        raise ValueError(f"{slug} prior execution argv mismatch: {mismatches}")

    execution_record = json_load(paths["execution"])
    normalized = json_load(paths["normalized_manifest"])
    if execution_record.get("status") != "completed" or execution_record.get("exit_code") != 0:
        raise ValueError(f"{slug} source execution is not completed with exit code 0")
    if [str(token) for token in execution_record.get("command_argv", [])] != source_argv:
        raise ValueError(f"{slug} execution.json argv differs from prior job argv")
    if [str(token) for token in normalized.get("command_argv", [])] != source_argv:
        raise ValueError(f"{slug} normalized manifest argv differs from prior job argv")
    if normalized.get("job_manifest_sha256") != sha256(paths["prior_job_manifest"]):
        raise ValueError(f"{slug} normalized manifest source-job hash mismatch")
    if sha256(paths["checkpoint"]) != execution_record["artifacts"]["current_json"]["sha256"]:
        raise ValueError(f"{slug} checkpoint differs from completed execution record")
    if sha256(paths["source_ledger"]) != execution_record["artifacts"]["estimator_call_ledger_json"]["sha256"]:
        raise ValueError(f"{slug} source ledger differs from completed execution record")
    if sha256(paths["result"]) != execution_record["artifacts"]["result_json"]["sha256"]:
        raise ValueError(f"{slug} result differs from completed execution record")

    checkpoint_payload = json_load(paths["checkpoint"])
    checkpoint_meta = _checkpoint_metadata(checkpoint_payload)
    signed_prefix_copy, signed_prefix_audit = _extract_signed_prefix_sidecar(
        slug,
        paths["result"],
    )
    parity = _frozen_checkpoint_validation(
        source_tree,
        paths["checkpoint"],
        paths["source_ledger"],
        signed_prefix_copy,
        source_options,
    )
    checkpoint_copy = _copy_source_checkpoint(slug, paths["checkpoint"])
    source_ledger_copy = _copy_source_ledger(slug, paths["source_ledger"])

    execution_argv, output_paths = _build_execution_argv(source_argv, slug)
    target_options = options(execution_argv)
    changed = sorted(
        key
        for key in set(source_options) | set(target_options)
        if source_options.get(key) != target_options.get(key)
    )
    unexpected = sorted(set(changed) - ALLOWED_EXECUTABLE_DIFF_FLAGS)
    if unexpected:
        raise ValueError(f"{slug} non-approved executable drift: {unexpected}")
    non_horizon_signature_source = {
        key: value
        for key, value in source_options.items()
        if key not in ALLOWED_EXECUTABLE_DIFF_FLAGS
    }
    non_horizon_signature_target = {
        key: value
        for key, value in target_options.items()
        if key not in ALLOWED_EXECUTABLE_DIFF_FLAGS
    }
    if non_horizon_signature_source != non_horizon_signature_target:
        raise ValueError(f"{slug} non-horizon route signature drift")

    environment, environment_diff = _build_environment(
        prior_job["environment"], slug
    )
    source_record = {
        "schema": "paper_i_hh_sr_r30_continuation_source_record_v1",
        "regime_slug": slug,
        "created_utc": utc_now(),
        "prior_job_manifest": paths["prior_job_manifest"].relative_to(REPO).as_posix(),
        "prior_job_manifest_sha256": sha256(paths["prior_job_manifest"]),
        "prior_execution": paths["execution"].relative_to(REPO).as_posix(),
        "prior_execution_sha256": sha256(paths["execution"]),
        "prior_normalized_manifest": paths["normalized_manifest"].relative_to(REPO).as_posix(),
        "prior_normalized_manifest_sha256": sha256(paths["normalized_manifest"]),
        "prior_result": paths["result"].relative_to(REPO).as_posix(),
        "prior_result_sha256": sha256(paths["result"]),
        "source_checkpoint": paths["checkpoint"].relative_to(REPO).as_posix(),
        "source_checkpoint_sha256": sha256(paths["checkpoint"]),
        "source_checkpoint_size_bytes": paths["checkpoint"].stat().st_size,
        "source_estimator_ledger": paths["source_ledger"].relative_to(REPO).as_posix(),
        "source_estimator_ledger_sha256": sha256(paths["source_ledger"]),
        "source_estimator_ledger_size_bytes": paths["source_ledger"].stat().st_size,
        "signed_active_prefix_sidecar": signed_prefix_audit,
        "checkpoint_metadata": checkpoint_meta,
        "checkpoint_parity_and_compile_smoke": parity,
    }
    source_record_path = BUNDLE_DIR / "source_records" / f"{slug}.json"
    json_dump(source_record_path, source_record)

    job = {
        "schema": "paper_i_hh_sr_r30_to_r50_continuation_job_v1",
        "bundle_id": BUNDLE_ID,
        "batch_name": BATCH_NAME,
        "regime_slug": slug,
        "created_utc": utc_now(),
        "run_class": "candidate_source_locked_continuation",
        "route_identity": prior_job["route_identity"],
        "physics": prior_job["physics"],
        "source_lock": {
            "base_source_archive": BASE_ARCHIVE.as_posix(),
            "base_source_archive_sha256": BASE_ARCHIVE_SHA256,
            "patched_source_archive": LOCKED_ARCHIVE.relative_to(REPO).as_posix(),
            "patched_source_archive_sha256": archive_inventory["archive_sha256"],
            "no_beam_resume_patch_manifest": PATCH_MANIFEST.relative_to(REPO).as_posix(),
            "no_beam_resume_patch_manifest_sha256": sha256(PATCH_MANIFEST),
            "no_beam_resume_patch": PATCH_FILE.relative_to(REPO).as_posix(),
            "no_beam_resume_patch_sha256": sha256(PATCH_FILE),
            "source_record": source_record_path.relative_to(REPO).as_posix(),
            "source_record_sha256": sha256(source_record_path),
            "transferred_checkpoint": checkpoint_copy.relative_to(REPO).as_posix(),
            "transferred_checkpoint_sha256": sha256(checkpoint_copy),
            "transferred_checkpoint_uncompressed_sha256": sha256(paths["checkpoint"]),
            "transferred_checkpoint_uncompressed_size_bytes": paths["checkpoint"].stat().st_size,
            "transferred_checkpoint_compression": "deterministic_gzip_mtime0_level9_v1",
            "transferred_source_ledger": source_ledger_copy.relative_to(REPO).as_posix(),
            "transferred_source_ledger_sha256": sha256(source_ledger_copy),
            "transferred_source_ledger_uncompressed_sha256": sha256(paths["source_ledger"]),
            "transferred_source_ledger_uncompressed_size_bytes": paths["source_ledger"].stat().st_size,
            "transferred_source_ledger_compression": "deterministic_gzip_mtime0_level9_v1",
            "transferred_signed_prefix_sidecar": signed_prefix_copy.relative_to(REPO).as_posix(),
            "transferred_signed_prefix_sidecar_sha256": sha256(signed_prefix_copy),
        },
        "command": {
            "source_argv": source_argv,
            "execution_argv": execution_argv,
            "source_options": source_options,
            "execution_options": target_options,
            "changed_flags": changed,
            "allowed_changed_flags": sorted(ALLOWED_EXECUTABLE_DIFF_FLAGS),
            "unexpected_differences": unexpected,
            "non_horizon_non_path_non_resume_settings_diff": [],
        },
        "approved_change": {
            "scientific_axis": "cumulative_controller_horizon_and_ansatz_depth",
            "source_depth": SOURCE_DEPTH,
            "target_depth": TARGET_DEPTH,
            "source_controller_round": SOURCE_DEPTH,
            "target_controller_round": TARGET_DEPTH,
            "maximum_new_singleton_admissions": MAX_NEW_ADMISSIONS,
            "resume_source": "preserved_round30_pre_terminal_current_json",
            "boundary_refit": "skipped_only_after_verified_checkpoint_parity",
            "compile_smoke": "required_FakeMarrakesh",
        },
        "scientific_contract": {
            **prior_job["scientific_contract"],
            "controller_round_target": TARGET_DEPTH,
            "ansatz_depth_target": TARGET_DEPTH,
            "source_controller_round": SOURCE_DEPTH,
            "source_ansatz_depth": SOURCE_DEPTH,
            "segment_max_new_admissions": MAX_NEW_ADMISSIONS,
            "resume_mode": "scaffold_v1",
            "resume_boundary_refit_policy": "verified_checkpoint_no_refit_v1",
            "resume_compile_smoke": "required",
            "resume_smoke_backend": "FakeMarrakesh",
        },
        "settings_difference": {
            "changed_flags": changed,
            "scientific_changed_fields": ["--adapt-max-depth"],
            "cumulative_segment_control_fields": sorted(CONTINUATION_FLAGS),
            "operational_output_path_fields": sorted(OUTPUT_PATH_FLAGS),
            "unexpected_executable_fields": [],
            "all_other_executable_fields_identical": True,
            "environment_differences": environment_diff,
        },
        "paths": output_paths,
        "environment": environment,
        "resources": {
            "request_cpus": 4,
            "request_memory_mb": int(regime["memory_mb"]),
            "request_disk_mb": int(regime["disk_mb"]),
            "max_runtime_s": MAX_RUNTIME_S,
        },
        "transfer_contract": {
            "mode": "compressed_output_bundle_v1",
            "archive": (
                Path("raw_outputs") / BUNDLE_ID / f"{slug}_transfer.tar.gz"
            ).as_posix(),
        },
    }
    validation = {
        "regime_slug": slug,
        "status": "pass",
        "checkpoint_source_sha256": source_record["source_checkpoint_sha256"],
        "checkpoint_transferred_compressed_sha256": sha256(checkpoint_copy),
        "checkpoint_transferred_uncompressed_sha256": sha256(paths["checkpoint"]),
        "checkpoint_parity": parity,
        "source_depth": SOURCE_DEPTH,
        "target_depth": TARGET_DEPTH,
        "non_approved_executable_diff": [],
        "environment_diff": environment_diff,
        "compressed_resume_input_roundtrip": {
            "status": "pass",
            "checkpoint": {
                "compressed_sha256": sha256(checkpoint_copy),
                "uncompressed_sha256": sha256(paths["checkpoint"]),
                "uncompressed_size_bytes": paths["checkpoint"].stat().st_size,
            },
            "estimator_ledger": {
                "compressed_sha256": sha256(source_ledger_copy),
                "uncompressed_sha256": sha256(paths["source_ledger"]),
                "uncompressed_size_bytes": paths["source_ledger"].stat().st_size,
            },
        },
    }
    return job, validation


def _run_zero_admission_boundary_smoke(
    *,
    source_tree: Path,
    job: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute the actual frozen resume boundary without a new admission."""

    if str(job["regime_slug"]) != "strong_weak_u8":
        raise ValueError("boundary smoke is locked to strong_weak_u8")
    source_record = json_load(REPO / Path(job["source_lock"]["source_record"]))
    signed_sidecar = REPO / Path(
        job["source_lock"]["transferred_signed_prefix_sidecar"]
    )
    source_files = source_paths("strong_weak_u8")
    with tempfile.TemporaryDirectory(prefix="sr_r50_boundary_smoke_") as temporary:
        root = Path(temporary)
        resume_root = root / "resume_input"
        output_root = root / "output"
        cache_root = root / "cache"
        resume_root.mkdir(parents=True)
        output_root.mkdir(parents=True)
        checkpoint = resume_root / "current.json"
        ledger = resume_root / "estimator_call_ledger.json"
        sidecar = resume_root / SIGNED_PREFIX_CANONICAL_NAME
        shutil.copy2(source_files["checkpoint"], checkpoint)
        shutil.copy2(source_files["source_ledger"], ledger)
        shutil.copy2(signed_sidecar, sidecar)
        if sha256(checkpoint) != source_record["source_checkpoint_sha256"]:
            raise ValueError("boundary-smoke checkpoint hash mismatch")
        if sha256(ledger) != source_record["source_estimator_ledger_sha256"]:
            raise ValueError("boundary-smoke ledger hash mismatch")
        if sha256(sidecar) != job["source_lock"][
            "transferred_signed_prefix_sidecar_sha256"
        ]:
            raise ValueError("boundary-smoke signed-prefix hash mismatch")

        result_json = output_root / "result.json"
        current_json = output_root / "current.json"
        output_ledger = output_root / "estimator_call_ledger.json"
        execution = [str(token) for token in job["command"]["source_argv"]]
        set_option(execution, "--adapt-max-depth", str(SOURCE_DEPTH))
        set_option(execution, "--adapt-final-full-refit", "false")
        set_option(execution, "--adapt-current-json", current_json.as_posix())
        set_option(
            execution,
            "--adapt-estimator-call-ledger-json",
            output_ledger.as_posix(),
        )
        set_option(execution, "--output-json", result_json.as_posix())
        smoke_additions = {
            "--adapt-resume-scaffold-json": checkpoint.as_posix(),
            "--adapt-resume-mode": "scaffold_v1",
            "--adapt-resume-boundary-refit-policy": "verified_checkpoint_no_refit_v1",
            "--adapt-segment-id": "strong_weak_u8-r30-boundary-smoke-v1",
            "--adapt-segment-target-depth": str(SOURCE_DEPTH),
            "--adapt-segment-target-controller-round": str(SOURCE_DEPTH),
            "--adapt-segment-max-new-admissions": "0",
            "--adapt-resume-compile-smoke": "required",
            "--adapt-resume-smoke-backend": "FakeMarrakesh",
        }
        for flag, value in smoke_additions.items():
            set_option(execution, flag, value)
        environment = os.environ.copy()
        environment.update(
            {
                str(key): str(value)
                for key, value in job["environment"].items()
            }
        )
        environment["PYTHONPATH"] = source_tree.as_posix()
        environment["MPLCONFIGDIR"] = (cache_root / "matplotlib").as_posix()
        environment["STATIC_ADAPT_CANDIDATE_RECORD_CACHE_DIR"] = (
            cache_root / "candidate_records"
        ).as_posix()
        environment["STATIC_ADAPT_HH_GENERATOR_REGISTRY_CACHE_DIR"] = (
            cache_root / "hh_generator_registry"
        ).as_posix()
        environment["STATIC_ADAPT_HH_POOL_CACHE_DIR"] = (
            cache_root / "hh_pool"
        ).as_posix()
        for key in (
            "MPLCONFIGDIR",
            "STATIC_ADAPT_CANDIDATE_RECORD_CACHE_DIR",
            "STATIC_ADAPT_HH_GENERATOR_REGISTRY_CACHE_DIR",
            "STATIC_ADAPT_HH_POOL_CACHE_DIR",
        ):
            Path(environment[key]).mkdir(parents=True, exist_ok=False)
        completed = subprocess.run(
            execution,
            cwd=source_tree,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=900,
        )
        if completed.returncode != 0 or not result_json.is_file() or not output_ledger.is_file():
            raise RuntimeError(
                "zero-admission boundary smoke failed: "
                f"rc={completed.returncode} stdout={completed.stdout[-3000:]} "
                f"stderr={completed.stderr[-3000:]}"
            )
        result_filter = r'''
{
  energy: .adapt_vqe.energy,
  ansatz_depth: .adapt_vqe.ansatz_depth,
  operators: .adapt_vqe.operators,
  adapt_final_full_refit: .adapt_vqe.adapt_final_full_refit,
  final_full_refit: .adapt_vqe.final_full_refit,
  resume_boundary_refit: .adapt_vqe.resume_boundary_refit,
  strict_replay: .adapt_vqe.strict_replay,
  active_generator_sector_contract: .adapt_vqe.active_generator_sector_contract,
  state_sector_contract: .adapt_vqe.state_sector_contract,
  segment: .adapt_segment
}
'''
        result_probe = subprocess.run(
            ["/usr/bin/jq", "-c", result_filter, str(result_json)],
            capture_output=True,
            text=True,
            check=True,
        )
        result = json.loads(result_probe.stdout)
        ledger_filter = r'''
{
  schema: .schema,
  adapt_success: .adapt_success,
  accounting_complete: .accounting.complete,
  accounting_status: .accounting.status,
  S_alg: .accounting.winning_lineage.S_alg,
  N_H_outer: .accounting.winning_lineage.N_H_outer,
  N_H_refit: .accounting.winning_lineage.N_H_refit,
  N_grad: .accounting.winning_lineage.N_grad,
  N_metric: .accounting.winning_lineage.N_metric,
  total_call_occurrences: .ledger.occurrence_summary.total_call_occurrences,
  unique_primitive_count: .ledger.occurrence_summary.unique_primitive_count
}
'''
        ledger_probe = subprocess.run(
            ["/usr/bin/jq", "-c", ledger_filter, str(output_ledger)],
            capture_output=True,
            text=True,
            check=True,
        )
        ledger_summary = json.loads(ledger_probe.stdout)

    expected_energy = float(source_record["checkpoint_metadata"]["saved_energy"])
    expected_labels = json_load(source_files["checkpoint"])["adapt_vqe"]["operators"]
    boundary = result["resume_boundary_refit"]
    segment = result["segment"]
    active_sector = result["active_generator_sector_contract"]
    strict_replay = result["strict_replay"]
    final_refit = result["final_full_refit"]
    sidecar_checkpoint = json_load(signed_sidecar)["checkpoint"]
    checks = {
        "process_exit_zero": completed.returncode == 0,
        "boundary_refit_not_attempted": boundary.get("attempted") is False,
        "boundary_refit_not_executed": boundary.get("executed") is False,
        "boundary_refit_verified_skip": boundary.get("skipped_reason")
        == "verified_checkpoint_no_refit_v1",
        "source_controller_round_30": int(segment.get("source_controller_round", -1))
        == SOURCE_DEPTH,
        "final_controller_round_30": int(segment.get("final_controller_round", -1))
        == SOURCE_DEPTH,
        "final_depth_30": int(segment.get("final_depth", -1)) == SOURCE_DEPTH,
        "zero_new_admissions": int(segment.get("new_admission_records", -1)) == 0,
        "energy_parity": abs(float(result["energy"]) - expected_energy)
        <= ENERGY_REPLAY_TOLERANCE,
        "ansatz_depth_30": int(result["ansatz_depth"]) == SOURCE_DEPTH,
        "ordered_prefix_identical": list(result["operators"]) == list(expected_labels),
        "strict_replay_pass": strict_replay.get("passed") is True,
        "active_sector_pass": active_sector.get("passed_with_parameterization") is True,
        "fixed_guarded_occurrence_count_22": int(
            active_sector.get("fixed_sector_guarded_generator_count", -1)
        ) == 22,
        "state_sector_pass": result["state_sector_contract"].get("passed") is True,
        "binary_padding_pass": float(
            sidecar_checkpoint.get("boson_illegal_codeword_probability", 1.0)
        ) <= 1.0e-12,
        "final_full_refit_disabled": result.get("adapt_final_full_refit") is False,
        "final_full_refit_not_executed": final_refit.get("executed") is False,
        "ledger_adapt_success": ledger_summary.get("adapt_success") is True,
        "ledger_accounting_complete": ledger_summary.get("accounting_complete") is True,
        "ledger_S_alg_restored": int(ledger_summary.get("S_alg", -1))
        == int(source_record["checkpoint_parity_and_compile_smoke"][
            "verified_checkpoint"
        ]["restored_S_alg"]),
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"zero-admission boundary smoke gates failed: {failed}")
    return {
        "schema": "paper_i_hh_sr_r30_boundary_execution_smoke_v1",
        "created_utc": utc_now(),
        "status": "pass",
        "regime_slug": "strong_weak_u8",
        "source_archive_sha256": sha256(LOCKED_ARCHIVE),
        "source_checkpoint_sha256": source_record["source_checkpoint_sha256"],
        "signed_prefix_sidecar_sha256": job["source_lock"][
            "transferred_signed_prefix_sidecar_sha256"
        ],
        "energy": float(result["energy"]),
        "source_energy": expected_energy,
        "energy_abs_discrepancy": abs(float(result["energy"]) - expected_energy),
        "segment": segment,
        "resume_boundary_refit": boundary,
        "ledger_summary": ledger_summary,
        "checks": checks,
        "scientific_admissions_executed": 0,
        "full_result_preserved": False,
    }


def submit_text(archive_sha256: str) -> str:
    base = f"chtc/phase3_optuna/input/{BUNDLE_ID}"
    return f"""universe = vanilla
executable = {base}/execute_source_locked_job.sh
arguments = $(job_manifest) {base}/source_locked.tar.gz {archive_sha256} chtc/phase3_optuna/image.sif {IMAGE_SHA256} $(regime_slug)
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_executable = True
preserve_relative_paths = True
transfer_input_files = {base}/run_job.py, {base}/source_archive_manifest.json, {base}/source_lock_and_settings_diff.json, {base}/bundle_manifest.json, {base}/preflight.json, {base}/source_lock/no_beam_resume_patch_manifest.json, {base}/source_lock/no_beam_verified_resume.patch, $(source_record), $(resume_checkpoint), $(resume_ledger), $(resume_signed_prefix), $(job_manifest), {base}/source_locked.tar.gz, chtc/phase3_optuna/image.sif
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
+MaxRuntime = {MAX_RUNTIME_S}
+JobBatchName = \"{BATCH_NAME}\"
notification = Never
queue regime_slug, job_manifest, source_record, resume_checkpoint, resume_ledger, resume_signed_prefix, memory_mb, disk_mb from {base}/queue.tsv
"""


def _validate_submit_text(text: str, ready_count: int) -> dict[str, Any]:
    required_fragments = {
        "queue_source": "queue regime_slug, job_manifest, source_record, resume_checkpoint, resume_ledger, resume_signed_prefix, memory_mb, disk_mb",
        "request_cpus": "request_cpus = 4",
        "max_runtime": f"+MaxRuntime = {MAX_RUNTIME_S}",
        "source_archive": "source_locked.tar.gz",
        "resume_checkpoint_transfer": "$(resume_checkpoint)",
        "resume_ledger_transfer": "$(resume_ledger)",
        "resume_signed_prefix_transfer": "$(resume_signed_prefix)",
        "on_exit_or_evict": "when_to_transfer_output = ON_EXIT_OR_EVICT",
    }
    missing = [name for name, fragment in required_fragments.items() if fragment not in text]
    if missing:
        raise ValueError(f"submit description validation failed: {missing}")
    if text.count("+JobBatchName =") != 1:
        raise ValueError("submit description must contain exactly one JobBatchName")
    return {
        "status": "pass",
        "ready_queue_rows": ready_count,
        "checks": {name: True for name in required_fragments},
    }


def _artifact_inventory(paths: Iterable[Path]) -> dict[str, Any]:
    return {
        path.relative_to(REPO).as_posix(): {
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
        if path.is_file()
    }


def _negative_runner_guard_tests(job_path: Path) -> dict[str, Any]:
    """Prove sidecar bytes and claimed result provenance both fail closed."""

    runner = BUNDLE_DIR / "run_job.py"
    original_job = json_load(job_path)
    original_sidecar = REPO / Path(
        original_job["source_lock"]["transferred_signed_prefix_sidecar"]
    )
    original_payload = json_load(original_sidecar)
    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="sr_r50_negative_") as temporary:
        root = Path(temporary)

        for case_name, source_lock_field in (
            ("compressed_checkpoint_byte_tamper", "transferred_checkpoint"),
            ("compressed_ledger_byte_tamper", "transferred_source_ledger"),
        ):
            original_compressed = REPO / Path(
                original_job["source_lock"][source_lock_field]
            )
            tampered_compressed = root / f"{case_name}.json.gz"
            tampered_compressed.write_bytes(
                original_compressed.read_bytes() + b"tamper"
            )
            tampered_compressed_job = json.loads(json.dumps(original_job))
            tampered_compressed_job["source_lock"][source_lock_field] = (
                tampered_compressed.as_posix()
            )
            tampered_compressed_job_path = root / f"{case_name}.job.json"
            json_dump(tampered_compressed_job_path, tampered_compressed_job)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(runner),
                    "--validate-only",
                    str(tampered_compressed_job_path),
                ],
                cwd=REPO,
                capture_output=True,
                text=True,
                check=False,
            )
            output = completed.stdout + completed.stderr
            if completed.returncode == 0 or "source-lock hash mismatch" not in output:
                raise RuntimeError(f"{case_name} negative guard did not fail closed")
            cases.append(
                {
                    "case": case_name,
                    "status": "pass_rejected",
                    "expected_failure": "source-lock hash mismatch",
                }
            )

        tampered_sidecar = root / "tampered_sidecar.json"
        tampered_payload = json.loads(json.dumps(original_payload))
        tampered_payload["checkpoint"]["active_ansatz_depth"] = SOURCE_DEPTH - 1
        json_dump(tampered_sidecar, tampered_payload)
        tampered_job = json.loads(json.dumps(original_job))
        tampered_job["source_lock"]["transferred_signed_prefix_sidecar"] = (
            tampered_sidecar.as_posix()
        )
        tampered_job_path = root / "tampered_job.json"
        json_dump(tampered_job_path, tampered_job)
        completed = subprocess.run(
            [sys.executable, str(runner), "--validate-only", str(tampered_job_path)],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        output = completed.stdout + completed.stderr
        if completed.returncode == 0 or "source-lock hash mismatch" not in output:
            raise RuntimeError("sidecar byte-tamper negative guard did not fail closed")
        cases.append(
            {
                "case": "sidecar_byte_tamper",
                "status": "pass_rejected",
                "expected_failure": "source-lock hash mismatch",
            }
        )

        claimed_sidecar = root / "claimed_source_hash_sidecar.json"
        claimed_payload = json.loads(json.dumps(original_payload))
        claimed_payload["source_result_sha256"] = "0" * 64
        json_dump(claimed_sidecar, claimed_payload)
        claimed_job = json.loads(json.dumps(original_job))
        claimed_job["source_lock"]["transferred_signed_prefix_sidecar"] = (
            claimed_sidecar.as_posix()
        )
        claimed_job["source_lock"][
            "transferred_signed_prefix_sidecar_sha256"
        ] = sha256(claimed_sidecar)
        claimed_job_path = root / "claimed_source_hash_job.json"
        json_dump(claimed_job_path, claimed_job)
        completed = subprocess.run(
            [sys.executable, str(runner), "--validate-only", str(claimed_job_path)],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        output = completed.stdout + completed.stderr
        if (
            completed.returncode == 0
            or "signed-prefix source-result provenance mismatch" not in output
        ):
            raise RuntimeError(
                "signed-prefix claimed source-result hash negative guard did not fail closed"
            )
        cases.append(
            {
                "case": "claimed_source_result_sha256_mutation",
                "status": "pass_rejected",
                "expected_failure": "signed-prefix source-result provenance mismatch",
            }
        )
    return {"status": "pass", "cases": cases}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ready-only",
        action="store_true",
        help="Stage only rows with complete fetched round-30 evidence.",
    )
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if "local_repos" not in REPO.parts or "Documents" in REPO.parts:
        raise RuntimeError(f"non-iCloud checkout guard failed: {REPO}")

    readiness = []
    for regime in REGIMES:
        ready, missing = row_readiness(regime)
        readiness.append(
            {
                "regime_slug": regime["slug"],
                "ready": ready,
                "missing_files": missing,
                "resources": {
                    "request_cpus": 4,
                    "request_memory_mb": regime["memory_mb"],
                    "request_disk_mb": regime["disk_mb"],
                    "max_runtime_s": MAX_RUNTIME_S,
                },
            }
        )
    missing_rows = [row for row in readiness if not row["ready"]]
    if missing_rows and not args.ready_only:
        missing_text = "; ".join(
            f"{row['regime_slug']}: {', '.join(row['missing_files'])}"
            for row in missing_rows
        )
        raise FileNotFoundError(
            "full four-row bundle is blocked by missing completed round-30 evidence; "
            f"rerun with --ready-only to stage available rows: {missing_text}"
        )

    archive_inventory, _archive = build_source_archive()
    for directory in (BUNDLE_DIR / "jobs", BUNDLE_DIR / "source_records", BUNDLE_DIR / "resume_inputs", BUNDLE_DIR / "checkpoint_validation"):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, Any]] = []
    built_job_payloads: dict[str, dict[str, Any]] = {}
    validations: list[dict[str, Any]] = []
    queue_lines: list[str] = []
    boundary_smoke: dict[str, Any] | None = None
    patched_tree = _unpack_patched_archive()
    try:
        source_tree = getattr(patched_tree, "source_root")
        for regime, ready_row in zip(REGIMES, readiness):
            if not ready_row["ready"]:
                continue
            job, validation = _build_job(regime, archive_inventory, source_tree)
            slug = str(regime["slug"])
            job_path = BUNDLE_DIR / "jobs" / f"{slug}.json"
            validation_path = BUNDLE_DIR / "checkpoint_validation" / f"{slug}.json"
            json_dump(job_path, job)
            json_dump(validation_path, validation)
            subprocess.run(
                [sys.executable, str(BUNDLE_DIR / "run_job.py"), "--validate-only", str(job_path)],
                cwd=REPO,
                check=True,
                capture_output=True,
                text=True,
            )
            validation["negative_runner_guard_tests"] = (
                _negative_runner_guard_tests(job_path)
            )
            json_dump(validation_path, validation)
            jobs.append(
                {
                    "regime_slug": slug,
                    "job_manifest": job_path.relative_to(REPO).as_posix(),
                    "job_manifest_sha256": sha256(job_path),
                    "checkpoint_validation": validation_path.relative_to(REPO).as_posix(),
                    "checkpoint_validation_sha256": sha256(validation_path),
                    "source_checkpoint_sha256": job["source_lock"][
                        "transferred_checkpoint_uncompressed_sha256"
                    ],
                    "source_estimator_ledger_sha256": job["source_lock"][
                        "transferred_source_ledger_uncompressed_sha256"
                    ],
                    "transferred_checkpoint_sha256": job["source_lock"][
                        "transferred_checkpoint_sha256"
                    ],
                    "transferred_source_ledger_sha256": job["source_lock"][
                        "transferred_source_ledger_sha256"
                    ],
                    "transferred_signed_prefix_sidecar_sha256": job[
                        "source_lock"
                    ]["transferred_signed_prefix_sidecar_sha256"],
                    "resources": job["resources"],
                }
            )
            built_job_payloads[slug] = job
            queue_lines.append(
                "\t".join(
                    (
                        slug,
                        job_path.relative_to(REPO).as_posix(),
                        job["source_lock"]["source_record"],
                        job["source_lock"]["transferred_checkpoint"],
                        job["source_lock"]["transferred_source_ledger"],
                        job["source_lock"]["transferred_signed_prefix_sidecar"],
                        str(regime["memory_mb"]),
                        str(regime["disk_mb"]),
                    )
                )
            )
            validations.append(validation)
        if "strong_weak_u8" not in built_job_payloads:
            raise RuntimeError("strong_weak_u8 is required for the boundary smoke")
        boundary_smoke = _run_zero_admission_boundary_smoke(
            source_tree=source_tree,
            job=built_job_payloads["strong_weak_u8"],
        )
        json_dump(BUNDLE_DIR / "boundary_execution_smoke.json", boundary_smoke)
    finally:
        patched_tree.cleanup()

    if not jobs:
        raise RuntimeError("ready-only generation found no complete checkpoint row")
    if not isinstance(boundary_smoke, Mapping) or boundary_smoke.get("status") != "pass":
        raise RuntimeError("zero-admission boundary execution smoke did not pass")
    (BUNDLE_DIR / "queue.tsv").write_text("\n".join(queue_lines) + "\n", encoding="utf-8")
    submit = submit_text(str(archive_inventory["archive_sha256"]))
    (BUNDLE_DIR / "submit.sub").write_text(submit, encoding="utf-8")
    submit_validation = _validate_submit_text(submit, len(jobs))

    settings_diff = {
        "schema": "source_locked_sensitivity_audit_v1",
        "created_utc": utc_now(),
        "bundle_id": BUNDLE_ID,
        "status": "pass" if not missing_rows else "pass_ready_rows_only_full_matrix_blocked",
        "run_class": "candidate_source_locked_continuation",
        "source": {
            "prior_bundle": PRIOR_BUNDLE.as_posix(),
            "prior_source_archive": BASE_ARCHIVE.as_posix(),
            "prior_source_archive_sha256": BASE_ARCHIVE_SHA256,
            "patched_source_archive": archive_inventory,
        },
        "sweep": {
            "variable": "cumulative_controller_horizon_and_ansatz_depth",
            "source_value": SOURCE_DEPTH,
            "target_value": TARGET_DEPTH,
            "wrapper_used": False,
            "baseline_materialization_status": "complete_for_ready_rows",
            "unresolved_source_fields": [],
            "fields_added_by_current_defaults": [],
            "settings_changed": ["--adapt-max-depth"],
            "continuation_control_fields": sorted(CONTINUATION_FLAGS),
            "operational_path_fields": sorted(OUTPUT_PATH_FLAGS),
        },
        "planned_rows": readiness,
        "ready_rows": jobs,
        "blocked_rows": missing_rows,
        "non_swept_settings_diff": [],
        "unexpected_differences": [],
        "checkpoint_anchor": {
            "kind": "per_regime_round30_pre_terminal_checkpoint_parity",
            "all_ready_rows_passed": all(row["status"] == "pass" for row in validations),
            "validations": validations,
        },
    }
    json_dump(BUNDLE_DIR / "source_lock_and_settings_diff.json", settings_diff)

    preflight = {
        "schema": "paper_i_hh_sr_r30_to_r50_continuation_preflight_v1",
        "created_utc": utc_now(),
        "status": "pass" if not missing_rows else "pass_ready_rows_only_full_matrix_blocked",
        "scientific_execution_performed": False,
        "submission_performed": False,
        "ready_only_mode": bool(args.ready_only),
        "ready_row_count": len(jobs),
        "planned_row_count": len(REGIMES),
        "checks": {
            "non_icloud_checkout": True,
            "base_source_archive_hash": True,
            "verified_patch_only": True,
            "live_scientific_tree_not_imported": True,
            "checkpoint_pre_terminal_round30": True,
            "checkpoint_energy_and_state_parity": True,
            "verified_no_refit_resume_seam": True,
            "required_compile_smoke": True,
            "source_execution_argv_exact": True,
            "strict_normalized_settings_diff": True,
            "isolated_output_and_cache_paths": True,
            "job_manifest_validation": True,
            "sidecar_tamper_and_claimed_hash_guards": all(
                row.get("negative_runner_guard_tests", {}).get("status") == "pass"
                for row in validations
            ),
            "zero_admission_boundary_execution_smoke": boundary_smoke,
            "submit_description_validation": submit_validation,
        },
        "blockers_before_full_four_row_submission": [
            f"missing completed round-30 evidence for {row['regime_slug']}"
            for row in missing_rows
        ],
        "remote_checks_before_any_submission": [
            f"verify {IMAGE_PATH.as_posix()} SHA-256 equals {IMAGE_SHA256}",
            "run Condor preflight against the exact ready queue.tsv",
        ],
    }
    json_dump(BUNDLE_DIR / "preflight.json", preflight)

    bundle_manifest = {
        "schema": "paper_i_hh_sr_r30_to_r50_continuation_bundle_v1",
        "bundle_id": BUNDLE_ID,
        "batch_name": BATCH_NAME,
        "created_utc": utc_now(),
        "run_class": "candidate_source_locked_continuation",
        "submission_status": "staged_not_submitted",
        "ready_only_mode": bool(args.ready_only),
        "planned_row_count": len(REGIMES),
        "ready_row_count": len(jobs),
        "jobs": jobs,
        "planned_rows": readiness,
        "blocked_rows": missing_rows,
        "source_archive": archive_inventory,
        "source_lock_and_settings_diff": {
            "path": (BUNDLE_DIR / "source_lock_and_settings_diff.json").relative_to(REPO).as_posix(),
            "sha256": sha256(BUNDLE_DIR / "source_lock_and_settings_diff.json"),
        },
        "preflight": {
            "path": (BUNDLE_DIR / "preflight.json").relative_to(REPO).as_posix(),
            "sha256": sha256(BUNDLE_DIR / "preflight.json"),
        },
        "execution_image": {
            "path": IMAGE_PATH.as_posix(),
            "sha256": IMAGE_SHA256,
            "remote_hash_check_required_before_submission": True,
        },
        "resources": {
            "request_cpus_all_rows": 4,
            "request_disk_mb_all_rows": 61440,
            "request_memory_mb_by_regime": {
                str(row["slug"]): int(row["memory_mb"]) for row in REGIMES
            },
            "max_runtime_s": MAX_RUNTIME_S,
        },
    }
    json_dump(BUNDLE_DIR / "bundle_manifest.json", bundle_manifest)

    upload_paths = [
        BUNDLE_DIR / "execute_source_locked_job.sh",
        BUNDLE_DIR / "run_job.py",
        BUNDLE_DIR / "submit.sub",
        BUNDLE_DIR / "queue.tsv",
        BUNDLE_DIR / "bundle_manifest.json",
        BUNDLE_DIR / "preflight.json",
        BUNDLE_DIR / "source_archive_manifest.json",
        BUNDLE_DIR / "source_lock_and_settings_diff.json",
        BUNDLE_DIR / "boundary_execution_smoke.json",
        PATCH_MANIFEST,
        PATCH_FILE,
        LOCKED_ARCHIVE,
        *sorted((BUNDLE_DIR / "jobs").glob("*.json")),
        *sorted((BUNDLE_DIR / "source_records").glob("*.json")),
        *sorted(path for path in (BUNDLE_DIR / "resume_inputs").glob("*") if path.is_file()),
        *sorted((BUNDLE_DIR / "checkpoint_validation").glob("*.json")),
    ]
    (BUNDLE_DIR / "upload_artifact_list.txt").write_text(
        "\n".join(path.relative_to(REPO).as_posix() for path in upload_paths) + "\n",
        encoding="utf-8",
    )
    json_dump(
        BUNDLE_DIR / "submission_artifact_hashes.json",
        {
            "schema": "paper_i_hh_sr_r50_submission_artifact_hashes_v1",
            "created_utc": utc_now(),
            "artifacts": _artifact_inventory(upload_paths),
            "required_remote_dependency": {
                "path": IMAGE_PATH.as_posix(),
                "sha256": IMAGE_SHA256,
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
