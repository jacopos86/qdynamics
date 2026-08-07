#!/usr/bin/env python3
"""Build the source-locked weak-weak FM-SNAKE/Append+FM CHTC pair."""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BUNDLE_ID = "paper_i_hh_fm_vs_append_fm_first_hit_weak_weak_20260720_v3_chtc"
SCHEMA = "paper_i_hh_fm_vs_append_fm_first_hit_weak_weak_chtc_bundle_v1"
JOB_ID = "weak_weak_fm_vs_append_fm_first_hit"
BASE_BUNDLE_ID = "paper_i_hh_sr_snake_no_overlap_trust_all_six_r50_20260720_v3_chtc"
BASE_ARCHIVE_SHA256 = "589c85f3bd33ea1fbc7115d3eb8273bb88ccb6e36f57f763d025095dbb43ffc1"
EXPECTED_IMAGE_SHA256 = "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
IMAGE_PATH = "chtc/phase3_optuna/image.sif"
CAMPAIGN_MODULE = "pipelines.exact_bench.paper_i_hh_fm_vs_append_fm_first_hit"

# The immutable SR archive provides the broad dependency closure already used on
# CHTC.  These current-checkout overlays are the complete executable surface for
# this new comparison plus exact-benchmark modules imported by the generic row.
EXPLICIT_OVERLAYS = (
    "pipelines/reporting/build_paper_i_selected_prefix_qiskit_sidecar.py",
    "pipelines/static_adapt/adapt_pipeline.py",
    "pipelines/static_adapt/paper_i_runner.py",
    "pipelines/static_adapt/formal_manifold_warm_start.py",
    "pipelines/static_adapt/joint_linear_solve.py",
    "pipelines/static_adapt/builders/shared_pauli_pool_contract.py",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _deterministic_archive(source_root: Path, output: Path) -> None:
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for path in sorted(source_root.rglob("*")):
                    if not path.is_file():
                        continue
                    relative = path.relative_to(source_root).as_posix()
                    info = tar.gettarinfo(str(path), arcname=relative)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)


def _overlay_paths(repo: Path) -> tuple[Path, ...]:
    exact_bench = tuple(sorted((repo / "pipelines" / "exact_bench").rglob("*.py")))
    explicit = tuple(repo / relative for relative in EXPLICIT_OVERLAYS)
    paths = exact_bench + explicit
    missing = [str(path.relative_to(repo)) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError("missing source overlays: " + ", ".join(missing))
    return tuple(dict.fromkeys(paths))


def _freeze_source(repo: Path, bundle: Path) -> dict[str, Any]:
    base = repo / "chtc" / "phase3_optuna" / "input" / BASE_BUNDLE_ID
    base_archive = base / "source_locked.tar.gz"
    if not base_archive.is_file() or _sha256(base_archive) != BASE_ARCHIVE_SHA256:
        raise RuntimeError("immutable base source archive is absent or changed")
    with tempfile.TemporaryDirectory(prefix="fm-append-pair-source-") as tmp_name:
        root = Path(tmp_name) / "source"
        root.mkdir(parents=True)
        with tarfile.open(base_archive, "r:gz") as archive:
            archive.extractall(root, filter="data")
        overlays: dict[str, dict[str, Any]] = {}
        for source in _overlay_paths(repo):
            relative = source.relative_to(repo)
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            overlays[relative.as_posix()] = {
                "sha256": _sha256(source),
                "size_bytes": source.stat().st_size,
            }
        archive_path = bundle / "source_locked.tar.gz"
        _deterministic_archive(root, archive_path)
        files = {
            path.relative_to(root).as_posix(): {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }
    payload = {
        "schema": "paper_i_hh_fm_vs_append_fm_source_archive_v1",
        "generated_utc": _utc_now(),
        "archive": archive_path.name,
        "archive_sha256": _sha256(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "file_count": len(files),
        "files": files,
        "immutable_parent_archive": {
            "bundle_id": BASE_BUNDLE_ID,
            "sha256": BASE_ARCHIVE_SHA256,
        },
        "overlays": overlays,
        "worker_pythonpath": "source_locked",
        "executable_source_authority": "archive inventory and per-file SHA-256",
        "git_role": "ancestry metadata only",
    }
    _write_json(bundle / "source_archive_manifest.json", payload)
    return payload


def _source_revision(repo: Path, source: dict[str, Any]) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        commit = git("rev-parse", "HEAD")
        tree = git("rev-parse", "HEAD^{tree}")
    except (OSError, subprocess.CalledProcessError):
        commit = None
        tree = None
    return {
        "schema": "paper_i_hh_fm_vs_append_fm_source_revision_v1",
        "generated_utc": _utc_now(),
        "git_commit": commit,
        "git_tree": tree,
        "source_archive_sha256": source["archive_sha256"],
        "source_archive_is_executable_authority": True,
    }


def _campaign_contract(repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if str(repo) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(repo))
    campaign = importlib.import_module(CAMPAIGN_MODULE)
    contract = campaign.matched_contract(("weak-weak",))
    shared_config = campaign.shared_formal_manifold_config()
    return contract, shared_config


def _job_manifest(
    *, source: dict[str, Any], contract: dict[str, Any], shared_config: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema": "paper_i_hh_fm_vs_append_fm_first_hit_chtc_job_v1",
        "bundle_id": BUNDLE_ID,
        "job_id": JOB_ID,
        "run_class": "diagnostic",
        "regime": "weak-weak",
        "physics": {
            "family": "Hubbard-Holstein",
            "L": 2,
            "U_over_t": 0.25,
            "lambda": 0.25,
            "n_ph_work": 3,
            "n_ph_reference": 3,
            "same_cutoff": True,
        },
        "comparison": {
            "initial_ansatz": "empty_hf_reference_v1",
            "automatic_hh_seed_disabled": True,
            "routes": ["fm_snake", "projected_singleton_append_fm"],
            "sequential_within_one_proc": True,
            "target_abs_delta_e": 2.0e-4,
            "max_controller_rounds": 30,
            "optimizer_maxiter": 200,
            "line_search_max_steps": 15,
            "qbroyd_qbang_enabled": False,
            "reported_query_coordinate": "winning_lineage_S_alg_only",
            "discarded_branch_work_reported": False,
            "qiskit": {
                "optimization_level": 0,
                "seed_transpiler": 7,
                "compile_convention": "table_i_basis_gate_transpile_v1",
            },
        },
        "matched_contract_sha256": _payload_sha256(contract),
        "shared_formal_manifold_config_sha256": _payload_sha256(shared_config),
        "source_lock": {
            "archive": "source_locked.tar.gz",
            "archive_sha256": source["archive_sha256"],
            "campaign_module": CAMPAIGN_MODULE,
            "campaign_module_sha256": source["files"][
                "pipelines/exact_bench/paper_i_hh_fm_vs_append_fm_first_hit.py"
            ]["sha256"],
        },
        "command": [
            "python3",
            "-m",
            CAMPAIGN_MODULE,
            "run-pair",
            "--campaign-dir",
            "CAMPAIGN_DIR",
            "--regime",
            "weak-weak",
        ],
        "resources": {
            "request_cpus": 4,
            "request_memory_mb": 24576,
            "request_disk_mb": 40960,
            "max_runtime_seconds": 259200,
        },
        "output_contract": {
            "transfer_mode": "single_compressed_narrow_archive",
            "stream_output": False,
            "stream_error": False,
            "full_worker_checkpoint_transfer": False,
            "terminal_results_and_provenance_only": True,
            "failure_recovery_snapshot": [
                "fm_snake/current.json",
                "projected_singleton_append_fm/partial_result.json",
                "projected_singleton_append_fm/adapt_iteration_progress.jsonl",
            ],
        },
    }


def _write_submit(bundle: Path, source_sha256: str) -> None:
    relative = f"chtc/phase3_optuna/input/{BUNDLE_ID}"
    lines = [
        "universe = vanilla",
        "batch_name = paper-i-hh-fm-vs-append-fm-first-hit-weak-weak-20260720-v3",
        f"executable = {relative}/execute_source_locked_job.sh",
        (
            f"arguments = $(job_manifest) {relative}/source_locked.tar.gz "
            f"{source_sha256} {IMAGE_PATH} {EXPECTED_IMAGE_SHA256} $(job_id)"
        ),
        "should_transfer_files = YES",
        "when_to_transfer_output = ON_EXIT_OR_EVICT",
        "transfer_executable = True",
        "preserve_relative_paths = True",
        (
            f"transfer_input_files = {relative}/run_job.py, $(job_manifest), "
            f"$(normalized_manifest), {relative}/source_archive_manifest.json, "
            f"{relative}/source_revision_manifest.json, {relative}/bundle_manifest.json, "
            f"{relative}/submission_artifact_hashes.json, {relative}/source_locked.tar.gz, "
            f"{IMAGE_PATH}"
        ),
        f"transfer_output_files = raw_outputs/{BUNDLE_ID}/$(job_id)_transfer.tar.gz",
        (
            f'transfer_output_remaps = "raw_outputs/{BUNDLE_ID}/$(job_id)_transfer.tar.gz = '
            f'$(Cluster).$(Process)__$(job_id)_transfer.tar.gz"'
        ),
        "stream_output = False",
        "stream_error = False",
        f"log = logs/{BUNDLE_ID}.$(Cluster).$(Process).log",
        f"output = logs/{BUNDLE_ID}.$(Cluster).$(Process).out",
        f"error = logs/{BUNDLE_ID}.$(Cluster).$(Process).err",
        "requirements = TARGET.HasSIF",
        "request_cpus = 4",
        "request_memory = 24576MB",
        "request_disk = 40960MB",
        "+WantFlocking = true",
        "+MaxRuntime = 259200",
        '+JobBatchName = "paper-i-hh-fm-vs-append-fm-first-hit-weak-weak-v3"',
        "notification = Never",
        (
            "queue job_id, job_manifest, normalized_manifest from "
            f"{relative}/queue.tsv"
        ),
        "",
    ]
    (bundle / "submit.sub").write_text("\n".join(lines), encoding="utf-8")


def _smoke_source_archive(bundle: Path, source: dict[str, Any]) -> dict[str, Any]:
    archive_path = bundle / "source_locked.tar.gz"
    if _sha256(archive_path) != source["archive_sha256"]:
        raise RuntimeError("source archive changed before smoke validation")
    with tempfile.TemporaryDirectory(prefix="fm-append-pair-smoke-") as tmp_name:
        root = Path(tmp_name) / "source"
        root.mkdir(parents=True)
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(root, filter="data")
        env = dict(__import__("os").environ)
        env.update(
            {
                "PYTHONPATH": str(root),
                "PYTHONDONTWRITEBYTECODE": "1",
                "TABLE_I_STATIC_SUITE_PROFILE": (
                    "paper_i_hh_completion_samecutoff_nph3_nph7_20260718_v1"
                ),
                "GENERIC_STATIC_TABLE_RESOURCE_QUBIT_CAP": "12",
                "GENERIC_STATIC_TABLE_RESOURCE_POOL_TERM_CAP": "16384",
                "GENERIC_STATIC_TABLE_EXACT_FIDELITY_MAX_QUBITS": "12",
                "STATIC_ADAPT_HH_POOL_CACHE": "disk",
                "STATIC_ADAPT_HH_POOL_CACHE_SCOPE": "exact",
            }
        )
        help_run = subprocess.run(
            [sys.executable, "-m", CAMPAIGN_MODULE, "--help"],
            cwd=root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if help_run.returncode != 0 or "run-pair" not in help_run.stdout:
            raise RuntimeError("archive-only campaign CLI import smoke failed")
        smoke_campaign = Path(tmp_name) / "campaign"
        plan_run = subprocess.run(
            [
                sys.executable,
                "-m",
                CAMPAIGN_MODULE,
                "plan",
                "--campaign-dir",
                str(smoke_campaign),
                "--regime",
                "weak-weak",
            ],
            cwd=root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=900,
            check=False,
        )
        if plan_run.returncode != 0:
            raise RuntimeError(
                "archive-only weak-weak plan smoke failed: "
                + plan_run.stderr[-2000:]
            )
        planned = json.loads((smoke_campaign / "campaign_manifest.json").read_text())
        source_lock = json.loads(
            (smoke_campaign / "source_lock" / "source_lock.json").read_text()
        )
        if planned.get("status") != "planned" or len(planned.get("rows", [])) != 1:
            raise RuntimeError("archive-only plan did not produce one planned row")
        if source_lock.get("current_source_complete") is not True:
            raise RuntimeError("archive-only campaign source lock is incomplete")
    return {
        "schema": "paper_i_hh_fm_vs_append_fm_first_hit_bundle_preflight_v1",
        "generated_utc": _utc_now(),
        "status": "pass",
        "checks": {
            "source_archive_sha256": "pass",
            "archive_only_cli_import_and_help": "pass",
            "archive_only_weak_weak_plan": "pass",
            "archive_only_source_lock_complete": "pass",
            "one_sequential_pair": "pass",
            "scientific_execution_performed": False,
            "chtc_submission_performed": False,
        },
        "source_archive_sha256": source["archive_sha256"],
        "scientific_blockers": [],
        "operational_blockers": [],
    }


def build() -> dict[str, Any]:
    repo = _repo_root()
    bundle = Path(__file__).resolve().parent
    source = _freeze_source(repo, bundle)
    revision = _source_revision(repo, source)
    _write_json(bundle / "source_revision_manifest.json", revision)
    contract, shared_config = _campaign_contract(repo)
    job = _job_manifest(source=source, contract=contract, shared_config=shared_config)
    job_path = bundle / "jobs" / f"{JOB_ID}.json"
    normalized_path = bundle / "normalized_manifests" / f"{JOB_ID}.json"
    _write_json(job_path, job)
    _write_json(normalized_path, job)
    repo_relative_job = job_path.relative_to(repo).as_posix()
    repo_relative_normalized = normalized_path.relative_to(repo).as_posix()
    with (bundle / "queue.tsv").open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter="\t").writerow(
            [JOB_ID, repo_relative_job, repo_relative_normalized]
        )
    manifest = {
        "schema": SCHEMA,
        "bundle_id": BUNDLE_ID,
        "generated_utc": _utc_now(),
        "status": "prepared_not_submitted",
        "submission_performed": False,
        "job_count": 1,
        "condor_proc_count": 1,
        "sequential_route_count_within_proc": 2,
        "source_archive": {
            "path": str((bundle / "source_locked.tar.gz").relative_to(repo)),
            "sha256": source["archive_sha256"],
        },
        "expected_image_sha256": EXPECTED_IMAGE_SHA256,
        "resources": job["resources"],
        "matched_contract": contract,
        "matched_contract_sha256": job["matched_contract_sha256"],
        "shared_formal_manifold_config": shared_config,
        "shared_formal_manifold_config_sha256": job[
            "shared_formal_manifold_config_sha256"
        ],
        "reporting_scope": {
            "query_coordinate": "winning_lineage_S_alg_only",
            "discarded_branch_work_reported": False,
        },
        "scientific_blockers": [],
        "operational_blockers": [],
    }
    _write_json(bundle / "bundle_manifest.json", manifest)
    _write_submit(bundle, source["archive_sha256"])
    _write_json(bundle / "preflight.json", _smoke_source_archive(bundle, source))
    hashes: dict[str, str] = {}
    for path in sorted(bundle.rglob("*")):
        if (
            path.is_file()
            and path.name != "submission_artifact_hashes.json"
            and "__pycache__" not in path.parts
            and ".pytest_cache" not in path.parts
            and path.suffix != ".pyc"
        ):
            hashes[path.relative_to(bundle).as_posix()] = _sha256(path)
    _write_json(
        bundle / "submission_artifact_hashes.json",
        {
            "schema": "paper_i_hh_fm_vs_append_fm_submission_hashes_v1",
            "generated_utc": _utc_now(),
            "files": hashes,
        },
    )
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
