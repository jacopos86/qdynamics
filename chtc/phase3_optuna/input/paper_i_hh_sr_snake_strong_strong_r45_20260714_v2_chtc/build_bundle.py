#!/usr/bin/env python3
"""Build the one-row source-locked SR-SNAKE strong-strong CHTC bundle."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


BUNDLE_ID = "paper_i_hh_sr_snake_strong_strong_r45_20260714_v2_chtc"
BUNDLE_DIR = Path(__file__).resolve().parent
REPO = BUNDLE_DIR.parents[3]
SOURCE_ROOT = Path(
    "raw_outputs/paper_i_hh_route4_round45_best_branch_recovery_20260713"
)
SOURCE_COMMAND = SOURCE_ROOT / "strong_strong_fresh/full/command.json"
SOURCE_MANIFEST = SOURCE_ROOT / "strong_strong_fresh/full/normalized_manifest.json"
SOURCE_ARCHIVE = (
    SOURCE_ROOT
    / "source_lock/archive/paper_i_hh_route4_round45_source_archive_20260713.tar.gz"
)
LOCKED_ARCHIVE = BUNDLE_DIR / "source_locked.tar.gz"
OUTPUT_ROOT = Path("raw_outputs") / BUNDLE_ID / "strong_strong"
TRANSFER_ARCHIVE = Path("raw_outputs") / BUNDLE_ID / "strong_strong_transfer.tar.gz"
JOB_PATH = BUNDLE_DIR / "jobs/strong_strong.json"
IMAGE = Path("chtc/phase3_optuna/image.sif")
IMAGE_SHA256 = "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
SOURCE_ARCHIVE_SHA256 = "f591f4a5296ac2ac8e3b255e41d1ffd8267c905feb0be9992ed5d0895c1c367b"
SOURCE_COMMAND_SHA256 = "3bb011ad5a24c92c5b4f935b707453f53c45eede4e7e5146b41c0b2e88db8e9e"
SOURCE_MANIFEST_SHA256 = "e1f839913cff95a2326c4dff66941c919c85fa38cec9389fc9efc9461949c69f"
CRITICAL_SOURCE_HASHES = {
    "docs/reports/pdf_utils.py": "884d5b19dcdd01f34b6af8f2b2b9523140da5175ca7b2af8efe76c657800e72c",
    "docs/reports/report_pages.py": "e0bd6a7f0cc7431698a334ff2d73eea572669c37c0b11b4e69eae7be662461b4",
    "pipelines/static_adapt/adapt_pipeline.py": "fa4173e13bbe74dfee24bcbde185ff7c2a4f249a0194e64ae69da0feb79f6703",
    "pipelines/static_adapt/cli_config.py": "8916a18ac1b7ce06ef86a69851f2d70b50791438ace463bae9f9acc3b88013b7",
    "pipelines/static_adapt/resume_scaffold.py": "9e0c17d66730b4cb255c9ba8cd0a8eaadcb1ac1513ba49a74aa7d4648e1d0580",
    "pipelines/static_adapt/run_control.py": "3faad1ee1033f043bc4db91c8f5ef973c627d692749691908c4971ad1e803c05",
    "pipelines/static_adapt/output_artifacts.py": "f51cc9d23a41bd34df522c15422ec73d44ff18932f88043d4fdbbea002246e1b",
    "pipelines/reporting/build_paper_i_selected_prefix_qiskit_sidecar.py": "bcb745d13dc9fabb663993948e867741e05cb0eb75b98b879067c9e0a5498cd6",
}
REQUIRED_ARCHIVE_ADDITIONS = {
    "docs/reports/pdf_utils.py": CRITICAL_SOURCE_HASHES["docs/reports/pdf_utils.py"],
    "docs/reports/report_pages.py": CRITICAL_SOURCE_HASHES["docs/reports/report_pages.py"],
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


def text_dump(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


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


def replace(argv: list[str], flag: str, value: str) -> None:
    position = argv.index(flag)
    argv[position + 1] = value


def archive_inventory(path: Path) -> dict[str, Any]:
    members: list[dict[str, Any]] = []
    member_hashes: dict[str, str] = {}
    with tarfile.open(path, "r:gz") as handle:
        for member in handle.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"unsafe archive member: {member.name}")
            record: dict[str, Any] = {
                "path": name.as_posix(),
                "type": "file" if member.isfile() else "directory" if member.isdir() else "other",
                "size_bytes": int(member.size),
            }
            if member.isfile():
                extracted = handle.extractfile(member)
                if extracted is None:
                    raise ValueError(f"unreadable archive member: {member.name}")
                digest = hashlib.sha256(extracted.read()).hexdigest()
                record["sha256"] = digest
                member_hashes[name.as_posix()] = digest
            elif not member.isdir():
                raise ValueError(f"special archive member is forbidden: {member.name}")
            members.append(record)
    mismatches = {
        member: {"expected": expected, "actual": member_hashes.get(member)}
        for member, expected in CRITICAL_SOURCE_HASHES.items()
        if member_hashes.get(member) != expected
    }
    if mismatches:
        raise ValueError(f"critical source lock mismatch: {mismatches}")
    return {
        "schema": "paper_i_hh_sr_snake_source_archive_manifest_v1",
        "archive_path": LOCKED_ARCHIVE.relative_to(REPO).as_posix(),
        "archive_sha256": sha256(path),
        "archive_size_bytes": path.stat().st_size,
        "member_count": len(members),
        "critical_source_hashes": CRITICAL_SOURCE_HASHES,
        "members": members,
    }


def build_locked_archive(source_path: Path) -> str:
    """Copy the scientific source lock and add only required import plumbing."""

    existing: set[str] = set()
    with tarfile.open(source_path, "r:gz") as source, tarfile.open(
        LOCKED_ARCHIVE,
        "w:gz",
        format=tarfile.PAX_FORMAT,
    ) as target:
        for member in source.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"unsafe archive member: {member.name}")
            if not member.isfile() and not member.isdir():
                raise ValueError(f"special archive member is forbidden: {member.name}")
            member.pax_headers = {}
            existing.add(name.as_posix())
            extracted = source.extractfile(member) if member.isfile() else None
            target.addfile(member, extracted)

        for relative, expected in REQUIRED_ARCHIVE_ADDITIONS.items():
            if relative in existing:
                raise ValueError(f"archive addition unexpectedly already present: {relative}")
            path = REPO / relative
            data = path.read_bytes()
            observed = hashlib.sha256(data).hexdigest()
            if observed != expected:
                raise ValueError(
                    f"archive addition hash mismatch for {relative}: {observed} != {expected}"
                )
            member = tarfile.TarInfo(relative)
            member.size = len(data)
            member.mode = 0o644
            member.mtime = 0
            target.addfile(member, io.BytesIO(data))
    return sha256(LOCKED_ARCHIVE)


def required_contract(option_map: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "--problem": "hh",
        "--L": "2",
        "--u": "8.0",
        "--g-ep": "0.790569415042",
        "--n-ph-max": "4",
        "--adapt-pool": "full_meta",
        "--adapt-inner-optimizer": "POWELL",
        "--adapt-maxiter": "200",
        "--adapt-max-depth": "45",
        "--static-lane-route": "physical_operator_type",
        "--physical-lane-shortlist-aggressiveness": "3",
        "--phase3-runtime-split-mode": "shortlist_pauli_children_v1",
        "--phase3-runtime-split-selection-mode": "archival_child_set_forward_v1",
        "--phase3-runtime-split-max-subset-size": "1",
        "--phase3-runtime-split-child-set-symmetry-policy": "hard_guard",
        "--phase3-runtime-split-child-padding-policy": "exact_projected_grouped_v1",
        "--phase1-prune-policy": "recoverability_ladder_v1",
        "--historical-singleton-coordinate-solve-policy": "supported_metric_whitened_eigh_v1",
        "--historical-singleton-trust-region-update-policy": "displacement_calibrated_unbounded_v2",
    }
    mismatches = {
        flag: {"expected": value, "actual": option_map.get(flag)}
        for flag, value in expected.items()
        if option_map.get(flag) != value
    }
    required_true = (
        "--phase0-no-pilot",
        "--phase2-no-batching",
        "--phase3-no-batching",
        "--phase1-prune-enabled",
        "--allow-archival-phase3-runtime-split",
    )
    for flag in required_true:
        if option_map.get(flag) is not True:
            mismatches[flag] = {"expected": True, "actual": option_map.get(flag)}
    forbidden = ("--phase2-enable-batching", "--phase3-enable-batching")
    for flag in forbidden:
        if flag in option_map:
            mismatches[flag] = {"expected": "absent", "actual": option_map[flag]}
    if mismatches:
        raise ValueError(f"scientific contract mismatch: {mismatches}")
    return {
        "hamiltonian": {"problem": "hh", "L": 2, "u_over_t": 8.0, "lambda": 1.25},
        "same_cutoff": {"n_ph_work": 4, "exact_reference_cutoff": 4},
        "controller_round_target": 45,
        "method_family": "singleton_response_snake",
        "profile": "supported_whitened_adaptive_trust_v1",
        "phase0_enabled": False,
        "batching_enabled": False,
        "singleton_child_policy": True,
        "symmetry_policy": "hard_guard",
        "padding_policy": "exact_projected_grouped_v1",
        "prune_policy": "recoverability_ladder_v1",
    }


def main() -> int:
    if "local_repos" not in REPO.parts or "Documents" in REPO.parts:
        raise RuntimeError(f"non-iCloud checkout guard failed: {REPO}")
    source_command_path = REPO / SOURCE_COMMAND
    source_manifest_path = REPO / SOURCE_MANIFEST
    source_archive_path = REPO / SOURCE_ARCHIVE
    if sha256(source_command_path) != SOURCE_COMMAND_SHA256:
        raise ValueError("source command SHA-256 mismatch")
    if sha256(source_manifest_path) != SOURCE_MANIFEST_SHA256:
        raise ValueError("source normalized manifest SHA-256 mismatch")
    if sha256(source_archive_path) != SOURCE_ARCHIVE_SHA256:
        raise ValueError("source archive SHA-256 mismatch")
    locked_archive_sha256 = build_locked_archive(source_archive_path)
    (BUNDLE_DIR / "source_lock").mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_command_path, BUNDLE_DIR / "source_lock/source_command.json")
    shutil.copy2(source_manifest_path, BUNDLE_DIR / "source_lock/source_normalized_manifest.json")

    source_command = load_json(source_command_path)
    source_manifest = load_json(source_manifest_path)
    if source_command.get("schema") != "paper_i_hh_route4_round45_command_v1":
        raise ValueError("unexpected source command schema")
    if source_manifest.get("schema") != "paper_i_hh_route4_round45_normalized_manifest_v1":
        raise ValueError("unexpected source manifest schema")
    baseline = [str(token) for token in source_command["argv"]]
    execution = list(baseline)
    output_json = (OUTPUT_ROOT / "json/result.json").as_posix()
    current_json = (OUTPUT_ROOT / "json/current.json").as_posix()
    ledger_json = (OUTPUT_ROOT / "json/estimator_call_ledger.json").as_posix()
    replace(execution, "--output-json", output_json)
    replace(execution, "--adapt-current-json", current_json)
    replace(execution, "--adapt-estimator-call-ledger-json", ledger_json)
    baseline_options = options(baseline)
    execution_options = options(execution)
    differences = [
        {"field": field, "source": baseline_options.get(field), "target": execution_options.get(field)}
        for field in sorted(set(baseline_options) | set(execution_options))
        if baseline_options.get(field) != execution_options.get(field)
    ]
    allowed = {
        "--output-json",
        "--adapt-current-json",
        "--adapt-estimator-call-ledger-json",
    }
    unexpected = [row for row in differences if row["field"] not in allowed]
    if unexpected or {row["field"] for row in differences} != allowed:
        raise ValueError(f"unexpected CHTC settings diff: {differences}")
    scientific_contract = required_contract(execution_options)
    archive_manifest = archive_inventory(LOCKED_ARCHIVE)
    json_dump(BUNDLE_DIR / "source_archive_manifest.json", archive_manifest)

    cache_root = Path("tmp") / BUNDLE_ID / "strong_strong/cache"
    environment = {
        "PYTHONPATH": ".",
        "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "disk",
        "STATIC_ADAPT_CANDIDATE_RECORD_CACHE_DIR": (cache_root / "candidate_records").as_posix(),
        "STATIC_ADAPT_HH_POOL_CACHE": "disk",
        "STATIC_ADAPT_HH_POOL_CACHE_DIR": (cache_root / "hh_pool").as_posix(),
        "STATIC_ADAPT_HH_POOL_CACHE_SCOPE": "exact",
        "STATIC_ADAPT_HH_GENERATOR_REGISTRY_CACHE": "disk",
        "STATIC_ADAPT_HH_GENERATOR_REGISTRY_CACHE_DIR": (cache_root / "generator_registry").as_posix(),
    }
    source_environment = {
        str(key): str(value) for key, value in source_command.get("environment", {}).items()
    }
    environment_differences = [
        {
            "field": key,
            "source": source_environment.get(key),
            "target": environment.get(key),
            "classification": "isolated_operational_cache_plumbing",
        }
        for key in sorted(set(source_environment) | set(environment))
        if source_environment.get(key) != environment.get(key)
    ]
    job = {
        "schema": "paper_i_hh_sr_snake_strong_strong_r45_chtc_job_v1",
        "bundle_id": BUNDLE_ID,
        "job_id": f"{BUNDLE_ID}__strong_strong",
        "regime": "strong_strong",
        "run_class": "candidate",
        "lineage_scope": "fresh",
        "source_lock": {
            "source_command": (BUNDLE_DIR / "source_lock/source_command.json").relative_to(REPO).as_posix(),
            "source_command_sha256": sha256(BUNDLE_DIR / "source_lock/source_command.json"),
            "source_normalized_manifest": (BUNDLE_DIR / "source_lock/source_normalized_manifest.json").relative_to(REPO).as_posix(),
            "source_normalized_manifest_sha256": sha256(BUNDLE_DIR / "source_lock/source_normalized_manifest.json"),
            "source_archive": LOCKED_ARCHIVE.relative_to(REPO).as_posix(),
            "source_archive_sha256": locked_archive_sha256,
            "critical_source_hashes": CRITICAL_SOURCE_HASHES,
            "base_source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "archive_additions": REQUIRED_ARCHIVE_ADDITIONS,
        },
        "command": {
            "source_argv": baseline,
            "execution_argv": execution,
            "source_options": baseline_options,
            "execution_options": execution_options,
            "allowlisted_differences": differences,
            "unexpected_differences": unexpected,
        },
        "environment": environment,
        "environment_audit": {
            "source_environment": source_environment,
            "execution_environment": environment,
            "differences": environment_differences,
            "scientific_settings_changed": False,
            "cache_state_contract": "empty_job_local_no_cross_route_reuse",
        },
        "scientific_contract": scientific_contract,
        "paths": {
            "output_root": OUTPUT_ROOT.as_posix(),
            "result_json": output_json,
            "current_json": current_json,
            "estimator_call_ledger_json": ledger_json,
            "execution_manifest_json": (OUTPUT_ROOT / "execution_manifest.json").as_posix(),
            "normalized_run_manifest_json": (OUTPUT_ROOT / "normalized_run_manifest.json").as_posix(),
        },
        "transfer_contract": {
            "mode": "compressed_output_bundle_v1",
            "source_directory": OUTPUT_ROOT.as_posix(),
            "transfer_archive": TRANSFER_ARCHIVE.as_posix(),
            "archive_created_by": (BUNDLE_DIR / "execute_source_locked_job.sh").relative_to(REPO).as_posix(),
            "when_to_transfer_output": "ON_EXIT_OR_EVICT",
            "submit_host_quota_protection": True,
        },
        "execution_image": {"path": IMAGE.as_posix(), "sha256": IMAGE_SHA256},
        "repair_lineage": {
            "supersedes_cluster_id": 8778362,
            "superseded_exit_code": 1,
            "failure_class": "source_archive_import_plumbing",
            "failure": "ModuleNotFoundError: No module named 'docs'",
            "scientific_settings_changed": False,
        },
        "generated_utc": utc_now(),
    }
    json_dump(JOB_PATH, job)
    text_dump(
        BUNDLE_DIR / "queue.tsv",
        f"strong_strong\t{JOB_PATH.relative_to(REPO).as_posix()}\t32768\t61440\n",
    )
    archive_rel = LOCKED_ARCHIVE.relative_to(REPO).as_posix()
    bundle_rel = BUNDLE_DIR.relative_to(REPO)
    transfer_inputs = [
        (bundle_rel / "run_job.py").as_posix(),
        (bundle_rel / "source_archive_manifest.json").as_posix(),
        (bundle_rel / "source_lock/source_command.json").as_posix(),
        (bundle_rel / "source_lock/source_normalized_manifest.json").as_posix(),
        JOB_PATH.relative_to(REPO).as_posix(),
        archive_rel,
        IMAGE.as_posix(),
    ]
    submit = f"""universe = vanilla
executable = {(bundle_rel / 'execute_source_locked_job.sh').as_posix()}
arguments = $(job_manifest) {archive_rel} {locked_archive_sha256} {IMAGE.as_posix()} {IMAGE_SHA256}
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_executable = True
preserve_relative_paths = True
transfer_input_files = {', '.join(transfer_inputs)}
transfer_output_files = {TRANSFER_ARCHIVE.as_posix()}
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
+JobBatchName = "paper-i-hh-sr-ss-r45-20260714-v2"
notification = Never
queue regime_slug, job_manifest, memory_mb, disk_mb from {(bundle_rel / 'queue.tsv').as_posix()}
"""
    text_dump(BUNDLE_DIR / "submit.sub", submit)

    generated = [
        BUNDLE_DIR / "build_bundle.py",
        BUNDLE_DIR / "execute_source_locked_job.sh",
        BUNDLE_DIR / "run_job.py",
        BUNDLE_DIR / "source_archive_manifest.json",
        BUNDLE_DIR / "source_lock/source_command.json",
        BUNDLE_DIR / "source_lock/source_normalized_manifest.json",
        JOB_PATH,
        BUNDLE_DIR / "queue.tsv",
        BUNDLE_DIR / "submit.sub",
        LOCKED_ARCHIVE,
    ]
    bundle_manifest = {
        "schema": "paper_i_hh_sr_snake_strong_strong_r45_chtc_bundle_v1",
        "bundle_id": BUNDLE_ID,
        "generated_utc": utc_now(),
        "output_root": OUTPUT_ROOT.as_posix(),
        "files": [
            {
                "path": path.relative_to(REPO).as_posix(),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in generated
        ],
    }
    json_dump(BUNDLE_DIR / "bundle_manifest.json", bundle_manifest)
    preflight = {
        "schema": "paper_i_hh_sr_snake_strong_strong_r45_chtc_preflight_v1",
        "status": "ready_for_remote_dry_run",
        "generated_utc": utc_now(),
        "checks": {
            "non_icloud_checkout": True,
            "source_archive_hash_match": True,
            "base_source_archive_hash_match": True,
            "required_import_plumbing_added": True,
            "critical_source_hashes_match": True,
            "scientific_contract_match": True,
            "argv_settings_diff_is_output_paths_only": True,
            "cache_environment_diff_recorded_separately": True,
            "single_queue_row": True,
            "isolated_output_root": not (REPO / OUTPUT_ROOT).exists(),
            "compressed_transfer_contract": True,
            "image_hash_requires_remote_verification": True,
        },
        "blockers": [],
        "bundle_manifest": {
            "path": (bundle_rel / "bundle_manifest.json").as_posix(),
            "sha256": sha256(BUNDLE_DIR / "bundle_manifest.json"),
        },
        "required_remote_checks": [
            f"sha256sum {IMAGE.as_posix()}",
            "container import smoke for pipelines.static_adapt.adapt_pipeline",
            f"condor_submit -dry-run /tmp/{BUNDLE_ID}.dryrun {(bundle_rel / 'submit.sub').as_posix()}",
            "one-shot duplicate-batch condor_q check",
        ],
    }
    json_dump(BUNDLE_DIR / "preflight_report.json", preflight)
    print(json.dumps({"bundle": BUNDLE_ID, "status": preflight["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
