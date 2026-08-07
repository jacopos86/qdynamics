#!/usr/bin/env python3
"""Build the immutable two-row macro-only post-run-fidelity successor.

This builder changes no scientific setting.  It derives a source archive from
the submitted macro-only parent by replacing only the reporting-only fidelity
auditor and adding its focused regression test.  The successor queues only the
two held strong-Holstein rows from cluster 8890778.
"""

from __future__ import annotations

import copy
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
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "chtc/phase3_optuna/input"
PARENT_ID = (
    "paper_i_hh_sr_snake_macro_only_physical_lanes_all_six_r50_"
    "20260719_v1_chtc"
)
PARENT = INPUT / PARENT_ID
TARGET_ID = (
    "paper_i_hh_sr_snake_macro_only_physical_lanes_fidelity_repair_"
    "remaining2_r50_20260719_v2_chtc"
)
TARGET = INPUT / TARGET_ID
PARENT_BATCH = "paper-i-hh-sr-macro-only-physical-lanes-six-r50-20260719-v1"
TARGET_BATCH = (
    "paper-i-hh-sr-macro-only-physical-lanes-fidelity-repair-"
    "remaining2-r50-20260719-v2"
)
PARENT_CLUSTER = 8890778
ROUTE_DIGEST = "d14d582e532ee41500cd7d3ebaa21b83da91bb3fcf014be53ab8d1049d1452fa"
ROUTE_PROFILE = (
    "supported_whitened_adaptive_trust_full_response_symmetric_cost_"
    "no_prune_macro_only_physical_lanes_v1"
)
ROUTE_REQUEST = "sr_snake_macro_only_physical_lanes_v1"
PARENT_SOURCE_SHA256 = (
    "3a5ed36ebdf260357aa86b3a5ab3c7d8372072329a8fec2e1043e90b6f7c34c7"
)
OLD_FIDELITY_SHA256 = (
    "5534333b6ad14a440a8b5f4e1104d388a11048c1a27b90e7f8466f048cbe1a42"
)
IMAGE_SHA256 = "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
FIDELITY_SOURCE = (
    "agent_guidance/skills/paper-i-results/scripts/"
    "compute_paper_i_main_fidelities.py"
)
FIDELITY_TEST = "test/test_paper_i_main_fidelity_audit.py"
OVERLAYS = (FIDELITY_SOURCE, FIDELITY_TEST)
ROWS = (
    ("intermediate_strong", 4, 49152, 81920),
    ("strong_strong_u8", 5, 57344, 81920),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def replace_tree(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            key: replace_tree(item, replacements) for key, item in value.items()
        }
    if isinstance(value, list):
        return [replace_tree(item, replacements) for item in value]
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
    return value


def deterministic_archive(source: Path, destination: Path) -> None:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(source.rglob("*"), key=lambda p: p.relative_to(source).as_posix()):
            relative = path.relative_to(source).as_posix()
            if path.is_symlink():
                raise ValueError(f"source archive cannot contain symlink: {relative}")
            info = tarfile.TarInfo(relative + ("/" if path.is_dir() else ""))
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            info.mtime = 0
            info.mode = 0o755 if path.is_dir() or os.access(path, os.X_OK) else 0o644
            if path.is_dir():
                info.type = tarfile.DIRTYPE
                archive.addfile(info)
            elif path.is_file():
                info.size = path.stat().st_size
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
            else:
                raise ValueError(f"unsupported archive member: {relative}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as zipped:
            zipped.write(raw.getvalue())


def inventory(source: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(source).as_posix(): {
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(source.rglob("*"))
        if path.is_file()
    }


def build_source_archive(temp: Path) -> tuple[Path, dict[str, dict[str, Any]], dict[str, Any]]:
    parent_archive = PARENT / "source_locked.tar.gz"
    if sha256(parent_archive) != PARENT_SOURCE_SHA256:
        raise ValueError("parent macro-only source archive hash drift")
    source = temp / "source"
    with tarfile.open(parent_archive, "r:gz") as archive:
        archive.extractall(source, filter="data")
    before = inventory(source)
    for relative in OVERLAYS:
        live = ROOT / relative
        if not live.is_file():
            raise FileNotFoundError(live)
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(live, target)
    after = inventory(source)
    changed = sorted(
        relative
        for relative in set(before) | set(after)
        if before.get(relative) != after.get(relative)
    )
    if changed != sorted(OVERLAYS):
        raise ValueError(f"unexpected source archive drift: {changed!r}")
    if before[FIDELITY_SOURCE]["sha256"] != OLD_FIDELITY_SHA256:
        raise ValueError("parent fidelity source hash drift")
    repair = {
        "schema": "paper_i_hh_sr_macro_fidelity_reporting_repair_v1",
        "classification": "post_run_reporting_only_no_scientific_change_v1",
        "parent_source_archive_sha256": PARENT_SOURCE_SHA256,
        "changed_or_added_files": {
            relative: {
                "before": before.get(relative),
                "after": after[relative],
                "role": (
                    "post_run_projector_fidelity_replay"
                    if relative == FIDELITY_SOURCE
                    else "focused_reporting_regression_test"
                ),
            }
            for relative in changed
        },
        "scientific_source_files_changed": [],
        "scientific_settings_changed": False,
    }
    destination = temp / "source_locked.tar.gz"
    deterministic_archive(source, destination)
    return destination, after, repair


def _science_projection(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(payload.get(key))
        for key in (
            "route_identity",
            "physics",
            "segment",
            "cache_policy",
            "evidence_requirements",
            "resource_request",
            "run_class",
        )
    }


def _normalized_command(payload: dict[str, Any]) -> list[str]:
    command = payload.get("command", payload.get("command_argv"))
    if isinstance(command, dict):
        command = command.get("argv")
    if not isinstance(command, list):
        raise ValueError("manifest command must be a list")
    return [str(value).replace(TARGET_ID, PARENT_ID) for value in command]


def _write_runtime_files(
    *, source_sha: str, fidelity_sha: str
) -> None:
    run_job = (PARENT / "run_job.py").read_text(encoding="utf-8")
    if run_job.count(OLD_FIDELITY_SHA256) != 1:
        raise ValueError("parent run_job fidelity hash anchor drift")
    run_job = run_job.replace(PARENT_ID, TARGET_ID).replace(
        OLD_FIDELITY_SHA256, fidelity_sha
    )
    (TARGET / "run_job.py").write_text(run_job, encoding="utf-8")

    validator = (PARENT / "validate_fetched.py").read_text(encoding="utf-8")
    if validator.count(OLD_FIDELITY_SHA256) != 1:
        raise ValueError("parent fetched validator fidelity hash anchor drift")
    validator = validator.replace(OLD_FIDELITY_SHA256, fidelity_sha)
    (TARGET / "validate_fetched.py").write_text(validator, encoding="utf-8")

    shutil.copy2(PARENT / "evidence_validation.py", TARGET / "evidence_validation.py")
    wrapper = (PARENT / "execute_source_locked_job.sh").read_text(encoding="utf-8")
    if wrapper.count(PARENT_ID) != 1:
        raise ValueError("parent execution wrapper bundle anchor drift")
    wrapper = wrapper.replace(PARENT_ID, TARGET_ID)
    (TARGET / "execute_source_locked_job.sh").write_text(wrapper, encoding="utf-8")
    os.chmod(TARGET / "execute_source_locked_job.sh", 0o755)
    shutil.copy2(
        PARENT / "physics_and_exact_reference_lock.json",
        TARGET / "physics_and_exact_reference_lock.json",
    )


def build() -> Path:
    if not PARENT.is_dir():
        raise FileNotFoundError(PARENT)
    if TARGET.exists():
        raise FileExistsError(f"immutable successor already exists: {TARGET}")
    TARGET.mkdir(parents=True)
    try:
        with tempfile.TemporaryDirectory(prefix="macro_fidelity_remaining2_") as tmp_raw:
            archive, source_files, repair = build_source_archive(Path(tmp_raw))
            shutil.copy2(archive, TARGET / "source_locked.tar.gz")
        source_sha = sha256(TARGET / "source_locked.tar.gz")
        fidelity_sha = source_files[FIDELITY_SOURCE]["sha256"]
        test_sha = source_files[FIDELITY_TEST]["sha256"]
        if fidelity_sha != sha256(ROOT / FIDELITY_SOURCE):
            raise ValueError("repaired fidelity source was not frozen byte-identically")
        if test_sha != sha256(ROOT / FIDELITY_TEST):
            raise ValueError("fidelity regression test was not frozen byte-identically")
        _write_runtime_files(source_sha=source_sha, fidelity_sha=fidelity_sha)

        common_replacements = {
            PARENT_ID: TARGET_ID,
            PARENT_BATCH: TARGET_BATCH,
            PARENT_SOURCE_SHA256: source_sha,
            OLD_FIDELITY_SHA256: fidelity_sha,
        }
        archive_manifest = replace_tree(
            load(PARENT / "source_archive_manifest.json"), common_replacements
        )
        archive_manifest.update(
            {
                "archive": f"chtc/phase3_optuna/input/{TARGET_ID}/source_locked.tar.gz",
                "archive_sha256": source_sha,
                "archive_size_bytes": (TARGET / "source_locked.tar.gz").stat().st_size,
                "file_count": len(source_files),
                "files": source_files,
                "derived_from_archive": {
                    "path": f"chtc/phase3_optuna/input/{PARENT_ID}/source_locked.tar.gz",
                    "sha256": PARENT_SOURCE_SHA256,
                },
                "post_run_fidelity_reporting_repair": repair,
            }
        )
        dump(TARGET / "source_archive_manifest.json", archive_manifest)

        revision = replace_tree(
            load(PARENT / "source_revision_manifest.json"), common_replacements
        )
        revision["post_run_fidelity_reporting_repair"] = repair
        dump(TARGET / "source_revision_manifest.json", revision)

        source_revision_sha = sha256(TARGET / "source_revision_manifest.json")
        source_archive_manifest_sha = sha256(TARGET / "source_archive_manifest.json")
        physics_sha = sha256(TARGET / "physics_and_exact_reference_lock.json")
        jobs: list[str] = []
        parity_rows: list[dict[str, Any]] = []
        for slug, parent_proc, _memory, _disk in ROWS:
            parent_job = load(PARENT / "jobs" / f"{slug}.json")
            job = replace_tree(parent_job, common_replacements)
            job["bundle_id"] = TARGET_ID
            job["batch_name"] = TARGET_BATCH
            source_lock = job["source_lock"]
            source_lock["source_archive_sha256"] = source_sha
            source_lock["source_revision_manifest_sha256"] = source_revision_sha
            source_lock["source_archive_manifest_sha256"] = source_archive_manifest_sha
            source_lock["physics_reference_lock_sha256"] = physics_sha
            dump(TARGET / "jobs" / f"{slug}.json", job)

            parent_normalized = load(PARENT / "normalized_manifests" / f"{slug}.json")
            normalized = replace_tree(parent_normalized, common_replacements)
            normalized["bundle_id"] = TARGET_ID
            normalized_lock = normalized["source_lock"]
            normalized_lock["source_archive_sha256"] = source_sha
            normalized_lock["source_revision_manifest_sha256"] = source_revision_sha
            normalized_lock["source_archive_manifest_sha256"] = source_archive_manifest_sha
            normalized_lock["physics_reference_lock_sha256"] = physics_sha
            dump(TARGET / "normalized_manifests" / f"{slug}.json", normalized)

            if _science_projection(job) != _science_projection(parent_job):
                raise ValueError(f"scientific job-manifest drift: {slug}")
            if _science_projection(normalized) != _science_projection(parent_normalized):
                raise ValueError(f"scientific normalized-manifest drift: {slug}")
            if _normalized_command(job) != _normalized_command(parent_job):
                raise ValueError(f"scientific command drift: {slug}")
            if _normalized_command(normalized) != _normalized_command(parent_normalized):
                raise ValueError(f"normalized command drift: {slug}")
            if job["route_identity"]["profile_contract_sha256"] != ROUTE_DIGEST:
                raise ValueError(f"route digest drift: {slug}")
            parity_rows.append(
                {
                    "regime_slug": slug,
                    "parent_proc": parent_proc,
                    "science_projection_sha256": canonical_sha256(
                        _science_projection(job)
                    ),
                    "command_argv_sha256_after_path_normalization": canonical_sha256(
                        _normalized_command(job)
                    ),
                    "scientific_settings_changed": False,
                }
            )
            jobs.append(f"chtc/phase3_optuna/input/{TARGET_ID}/jobs/{slug}.json")

        route_parity = {
            "schema": "paper_i_hh_sr_macro_fidelity_remaining2_route_parity_v1",
            "status": "pass",
            "parent_bundle_id": PARENT_ID,
            "parent_cluster": PARENT_CLUSTER,
            "successor_bundle_id": TARGET_ID,
            "profile_request": ROUTE_REQUEST,
            "profile_resolved": ROUTE_PROFILE,
            "profile_contract_sha256": ROUTE_DIGEST,
            "rows": parity_rows,
            "only_changes": [
                "reporting_only_fidelity_source",
                "focused_reporting_regression_test",
                "bundle_output_and_source_lock_paths",
                "source_archive_and_manifest_hashes",
                "queue_scope_parent_procs_4_5_only",
            ],
            "scientific_settings_changed": False,
            "unexpected_differences": [],
        }
        dump(TARGET / "route_parity.json", route_parity)
        dump(
            TARGET / "scientific_settings_audit.json",
            {
                "schema": "paper_i_hh_sr_macro_fidelity_remaining2_science_audit_v1",
                "status": "pass",
                "profile_contract_sha256": ROUTE_DIGEST,
                "rows": parity_rows,
                "same_cutoff_reference": True,
                "target_controller_round": 50,
                "scientific_settings_changed": False,
                "unexpected_executable_differences": [],
            },
        )
        receipt = {
            "schema": "paper_i_hh_sr_macro_fidelity_remaining2_successor_v1",
            "classification": "post_run_reporting_repair_no_scientific_change_v1",
            "parent_bundle_id": PARENT_ID,
            "parent_cluster": PARENT_CLUSTER,
            "parent_source_archive_sha256": PARENT_SOURCE_SHA256,
            "successor_bundle_id": TARGET_ID,
            "successor_batch_name": TARGET_BATCH,
            "successor_source_archive_sha256": source_sha,
            "route_contract_sha256": ROUTE_DIGEST,
            "route_contract_unchanged": True,
            "source_archive_repair": repair,
            "superseded_parent_rows": [
                {
                    "parent_cluster": PARENT_CLUSTER,
                    "parent_proc": parent_proc,
                    "regime_slug": slug,
                    "scheduler_state": "held",
                    "superseded_by_successor": True,
                }
                for slug, parent_proc, _memory, _disk in ROWS
            ],
            "preserved_parent_rows": {
                "completed_for_local_post_run_salvage": [0, 1, 2],
                "not_duplicated_active_or_completed": [3],
            },
            "scientific_settings_changed": False,
            "submission_performed": False,
        }
        dump(TARGET / "operational_successor_receipt.json", receipt)

        bundle = {
            "schema": "paper_i_hh_sr_macro_fidelity_remaining2_bundle_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "bundle_id": TARGET_ID,
            "batch_name": TARGET_BATCH,
            "jobs": jobs,
            "route_identity": {
                "profile_request": ROUTE_REQUEST,
                "profile_resolved": ROUTE_PROFILE,
                "profile_contract_sha256": ROUTE_DIGEST,
            },
            "run_class": "macro_only_pool_ablation_reporting_repair_remaining2",
            "submission_scope": "exactly_parent_cluster_8890778_procs_4_5",
            "submission_status": "built_not_submitted",
            "source_archive_sha256": source_sha,
            "predecessor": {
                "cluster_id": PARENT_CLUSTER,
                "procs": [4, 5],
                "state": "held",
            },
        }
        dump(TARGET / "bundle_manifest.json", bundle)

        queue_lines = []
        for slug, _proc, memory, disk in ROWS:
            queue_lines.append(
                "\t".join(
                    [
                        slug,
                        f"chtc/phase3_optuna/input/{TARGET_ID}/jobs/{slug}.json",
                        f"chtc/phase3_optuna/input/{TARGET_ID}/normalized_manifests/{slug}.json",
                        str(memory),
                        str(disk),
                    ]
                )
            )
        (TARGET / "queue.tsv").write_text("\n".join(queue_lines) + "\n", encoding="utf-8")

        transfer = ", ".join(
            [
                f"chtc/phase3_optuna/input/{TARGET_ID}/{name}"
                for name in (
                    "run_job.py",
                    "evidence_validation.py",
                    "validate_fetched.py",
                    "source_archive_manifest.json",
                    "source_revision_manifest.json",
                    "physics_and_exact_reference_lock.json",
                    "bundle_manifest.json",
                    "preflight.json",
                    "route_parity.json",
                    "scientific_settings_audit.json",
                )
            ]
            + [
                "$(job_manifest)",
                "$(normalized_manifest)",
                f"chtc/phase3_optuna/input/{TARGET_ID}/source_locked.tar.gz",
                "chtc/phase3_optuna/image.sif",
            ]
        )
        submit = f'''universe = vanilla
# Two-row post-run-fidelity reporting repair; scientific route/settings unchanged.
executable = chtc/phase3_optuna/input/{TARGET_ID}/execute_source_locked_job.sh
arguments = $(job_manifest) chtc/phase3_optuna/input/{TARGET_ID}/source_locked.tar.gz {source_sha} chtc/phase3_optuna/image.sif {IMAGE_SHA256} $(regime_slug)
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_executable = True
preserve_relative_paths = True
transfer_input_files = {transfer}
transfer_output_files = raw_outputs/{TARGET_ID}/$(regime_slug)_transfer.tar.gz
transfer_output_remaps = "raw_outputs/{TARGET_ID}/$(regime_slug)_transfer.tar.gz = $(Cluster).$(Process)__$(regime_slug)_transfer.tar.gz"
stream_output = False
stream_error = False
log = logs/{TARGET_ID}.$(Cluster).$(Process).log
output = logs/{TARGET_ID}.$(Cluster).$(Process).out
error = logs/{TARGET_ID}.$(Cluster).$(Process).err
requirements = TARGET.HasSIF
request_cpus = 4
request_memory = $(memory_mb)MB
request_disk = $(disk_mb)MB
+MaxRuntime = 259200
+JobBatchName = "{TARGET_BATCH}"
notification = Never
queue regime_slug, job_manifest, normalized_manifest, memory_mb, disk_mb from chtc/phase3_optuna/input/{TARGET_ID}/queue.tsv
'''
        (TARGET / "submit.sub").write_text(submit, encoding="utf-8")
        (TARGET / "README.md").write_text(
            f"""# {TARGET_BATCH}

Immutable two-row successor for held cluster `{PARENT_CLUSTER}` procs `4` and `5`.

- Exact parent macro-only/physical-lanes route digest: `{ROUTE_DIGEST}`.
- Exact parent scientific manifests, resource requests, and 50-round horizon.
- Only source changes: repaired post-run fidelity replay plus its focused test.
- Source archive SHA-256: `{source_sha}`.
- Submission status: built, not submitted.
""",
            encoding="utf-8",
        )
        (TARGET / "build_bundle.py").write_text(
            f'''#!/usr/bin/env python3
import importlib.util
from pathlib import Path
SCRIPT = Path(__file__).resolve().parents[2] / "build_paper_i_hh_macro_only_fidelity_successor_20260719.py"
spec = importlib.util.spec_from_file_location("macro_fidelity_successor_builder", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)
if __name__ == "__main__":
    module.verify(Path(__file__).resolve().parent, run_archive_preflight=False)
    print("macro fidelity remaining2 successor verification passed")
''',
            encoding="utf-8",
        )
        (TARGET / "test_bundle.py").write_text(
            '''#!/usr/bin/env python3
import unittest
import build_bundle
class BundleTest(unittest.TestCase):
    def test_immutable_successor(self):
        build_bundle.module.verify(
            build_bundle.Path(__file__).resolve().parent,
            run_archive_preflight=False,
        )
if __name__ == "__main__": unittest.main()
''',
            encoding="utf-8",
        )
        os.chmod(TARGET / "build_bundle.py", 0o755)
        os.chmod(TARGET / "test_bundle.py", 0o755)

        dump(
            TARGET / "preflight.json",
            {
                "schema": "paper_i_hh_sr_macro_fidelity_remaining2_preflight_v1",
                "status": "pending_archive_only_validation",
                "job_count": 2,
                "exact_parent_rows": [4, 5],
                "route_contract_sha256": ROUTE_DIGEST,
                "source_archive_sha256": source_sha,
                "submission_performed": False,
            },
        )
        archive_report = verify(TARGET, run_archive_preflight=True)
        dump(TARGET / "archive_only_preflight.json", archive_report)
        preflight = load(TARGET / "preflight.json")
        preflight.update(
            {
                "status": "pass_built_not_submitted",
                "archive_only_validate_rows_passed": 2,
                "focused_fidelity_test_passed": True,
            }
        )
        dump(TARGET / "preflight.json", preflight)
        write_artifact_hashes(TARGET)
        return TARGET
    except Exception:
        shutil.rmtree(TARGET, ignore_errors=True)
        raise


def _archive_preflight(target: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="macro_remaining2_archive_validate_") as tmp_raw:
        source = Path(tmp_raw) / "source"
        with tarfile.open(target / "source_locked.tar.gz", "r:gz") as archive:
            archive.extractall(source, filter="data")
        copied_bundle = source / "chtc/phase3_optuna/input" / target.name
        shutil.copytree(
            target,
            copied_bundle,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONPATH": str(source),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        # The source-locked job manifest itself enforces PYTHONNOUSERSITE=1 for
        # the CHTC image.  Local archive preflight retains this machine's
        # installed validation dependencies while importing repo modules only
        # from the extracted archive via PYTHONPATH.
        environment.pop("PYTHONNOUSERSITE", None)
        rows = []
        for slug, _proc, _memory, _disk in ROWS:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(copied_bundle / "run_job.py"),
                    "--validate-only",
                    str(copied_bundle / "jobs" / f"{slug}.json"),
                ],
                cwd=source,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            rows.append(
                {
                    "regime_slug": slug,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"archive-only validate failed for {slug}: {completed.stderr}"
                )
        focused = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                FIDELITY_TEST,
                "-k",
                "signed_checkpoint_fidelity_replay_repairs_only_execution_order",
            ],
            cwd=source,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if focused.returncode != 0:
            raise RuntimeError(f"archive fidelity regression failed: {focused.stderr}")
        return {
            "schema": "paper_i_hh_sr_macro_fidelity_remaining2_archive_preflight_v1",
            "status": "pass",
            "source_archive_sha256": sha256(target / "source_locked.tar.gz"),
            "validate_only_rows": rows,
            "focused_fidelity_test": {
                "returncode": focused.returncode,
                "stdout": focused.stdout,
                "stderr": focused.stderr,
            },
        }


def verify(target: Path, *, run_archive_preflight: bool) -> dict[str, Any]:
    if target.name != TARGET_ID:
        raise ValueError(f"unexpected successor bundle id: {target.name}")
    bundle = load(target / "bundle_manifest.json")
    if bundle.get("bundle_id") != TARGET_ID:
        raise ValueError("bundle id drift")
    if bundle.get("submission_status") != "built_not_submitted":
        raise ValueError("successor submission status drift")
    if bundle.get("route_identity", {}).get("profile_contract_sha256") != ROUTE_DIGEST:
        raise ValueError("successor route digest drift")
    archive_manifest = load(target / "source_archive_manifest.json")
    source_sha = sha256(target / "source_locked.tar.gz")
    if source_sha != bundle.get("source_archive_sha256"):
        raise ValueError("bundle/source archive hash drift")
    if source_sha != archive_manifest.get("archive_sha256"):
        raise ValueError("archive manifest digest drift")
    repair = archive_manifest.get("post_run_fidelity_reporting_repair", {})
    changed = repair.get("changed_or_added_files", {})
    if set(changed) != set(OVERLAYS):
        raise ValueError("reporting-only archive diff scope drift")
    if repair.get("scientific_settings_changed") is not False:
        raise ValueError("repair claims a scientific change")
    queue = [line for line in (target / "queue.tsv").read_text().splitlines() if line]
    if len(queue) != 2 or [line.split("\t", 1)[0] for line in queue] != [
        row[0] for row in ROWS
    ]:
        raise ValueError("successor queue must contain exactly the two held rows")
    parity = load(target / "route_parity.json")
    if parity.get("status") != "pass" or parity.get("scientific_settings_changed") is not False:
        raise ValueError("successor route parity failed")
    receipt = load(target / "operational_successor_receipt.json")
    if [row.get("parent_proc") for row in receipt.get("superseded_parent_rows", [])] != [4, 5]:
        raise ValueError("held predecessor row lineage drift")
    for slug, _proc, memory, disk in ROWS:
        job = load(target / "jobs" / f"{slug}.json")
        normalized = load(target / "normalized_manifests" / f"{slug}.json")
        if job.get("route_identity") != normalized.get("route_identity"):
            raise ValueError(f"job/normalized route identity drift: {slug}")
        if job["route_identity"]["profile_contract_sha256"] != ROUTE_DIGEST:
            raise ValueError(f"route digest drift: {slug}")
        if job["resource_request"] != {"cpus": 4, "disk_mb": disk, "max_runtime_s": 259200, "memory_mb": memory}:
            raise ValueError(f"resource request drift: {slug}")
        parent_job = load(PARENT / "jobs" / f"{slug}.json")
        if _science_projection(job) != _science_projection(parent_job):
            raise ValueError(f"scientific job parity drift: {slug}")
        if _normalized_command(job) != _normalized_command(parent_job):
            raise ValueError(f"scientific command parity drift: {slug}")
    if run_archive_preflight:
        return _archive_preflight(target)
    return {
        "schema": "paper_i_hh_sr_macro_fidelity_remaining2_verify_v1",
        "status": "pass",
        "source_archive_sha256": source_sha,
        "job_count": 2,
    }


def write_artifact_hashes(target: Path) -> None:
    artifacts: dict[str, Any] = {}
    for path in sorted(target.rglob("*")):
        if (
            not path.is_file()
            or path.name == "submission_artifact_hashes.json"
            or "__pycache__" in path.parts
        ):
            continue
        relative = path.relative_to(ROOT).as_posix()
        artifacts[relative] = {
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
    dump(
        target / "submission_artifact_hashes.json",
        {
            "schema": "paper_i_hh_sr_macro_fidelity_remaining2_artifact_hashes_v1",
            "artifacts": artifacts,
        },
    )
    essentials = [
        "execute_source_locked_job.sh",
        "run_job.py",
        "evidence_validation.py",
        "validate_fetched.py",
        "source_locked.tar.gz",
        "source_archive_manifest.json",
        "source_revision_manifest.json",
        "physics_and_exact_reference_lock.json",
        "bundle_manifest.json",
        "preflight.json",
        "archive_only_preflight.json",
        "route_parity.json",
        "scientific_settings_audit.json",
        "operational_successor_receipt.json",
        "queue.tsv",
        "submit.sub",
    ]
    essentials += [f"jobs/{row[0]}.json" for row in ROWS]
    essentials += [f"normalized_manifests/{row[0]}.json" for row in ROWS]
    (target / "upload_artifact_list.txt").write_text(
        "\n".join(
            f"chtc/phase3_optuna/input/{TARGET_ID}/{relative}"
            for relative in essentials
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    target = build()
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
