#!/usr/bin/env python3
"""Build the source-locked six-regime SR-SNAKE-v4 parent bundle.

This builder performs no scientific calculation and never submits CHTC jobs.
It archives the exact committed source tree, resolves the v4 profile from that
source, verifies the six same-cutoff physics anchors, and writes the six fresh
round-0 -> round-30 job records plus a fail-closed submission preflight.
"""

from __future__ import annotations

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
from typing import Any


BUNDLE_ID = "paper_i_hh_sr_snake_v4_candidate_all_six_20260716_v1_chtc"
BATCH_NAME = "paper-i-hh-sr-v4-candidate-six-r30-20260716-v1"
BUNDLE_DIR = Path(__file__).resolve().parent
REPO = BUNDLE_DIR.parents[3]
EXPECTED_HEAD = "dfe8d8cad94167ebb1be6f919eeab3a64bb904d2"
EXPECTED_TREE = "e49f80b371ed0236875b7fa317ce475adb8d5b50"
PROFILE_REQUEST = "sr_snake_v4"
PROFILE_RESOLVED = (
    "supported_whitened_adaptive_trust_full_response_symmetric_cost_fs_prune_v4"
)
PROFILE_CONTRACT_SHA256 = (
    "447b8fe3f4fef340fbb1cd5d221a0234826ba80c7e4e405937004e4ab25bec93"
)
REMOTE_IMAGE_PATH = Path("chtc/phase3_optuna/image.sif")
REMOTE_IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)
REMOTE_QISKIT_VERSION = "2.3.1"
REMOTE_FAKE_BACKEND_RESOLVED = "fake_marrakesh"
REMOTE_FAKE_BACKEND_QUBITS = 156
ARCHIVE_PATH = BUNDLE_DIR / "source_locked.tar.gz"
SOURCE_LOCK_STATE = "frozen_repaired_head_preflight_passed_not_submitted"
SUBMISSION_ENABLED = True

HISTORICAL_MANIFEST_ROOT = Path(
    "chtc/phase3_optuna/input/"
    "paper_i_hh_sr_snake_noprune_nobeam_ordinary_novelty_all_six_"
    "20260715_v1_chtc/jobs"
)
HISTORICAL_RESULT_ROOT = Path(
    "raw_outputs/paper_i_hh_sr_snake_noprune_nobeam_no_ordinary_novelty_"
    "five_20260715_v1_chtc"
)
WEAK_WEAK_RESULT = Path(
    "raw_outputs/paper_i_hh_sr_snake_weak_weak_undamped_no_prune_no_beam_"
    "no_ordinary_novelty_fallback_on_20260715/json/result.json"
)
SMOKE8_ROOT = Path(
    "raw_outputs/paper_i_hh_sr_snake_v4_weak_weak_smoke_20260716/"
    "diagnostic_admissions8_nonbeam_overlay_trust_repair_v2"
)

CRITICAL_SOURCE_HASHES = {
    "pipelines/static_adapt/adapt_pipeline.py": (
        "efea1f3628918d312eba8ec148036c4dbaccc50b83bcdf3347a367d537f98c6e"
    ),
    "pipelines/static_adapt/cli_config.py": (
        "7fc8fa85d598163043759422ea1038e16f3dc9261419671d18d55f19eedebb55"
    ),
    "pipelines/static_adapt/resume_scaffold.py": (
        "9006345e4042b305d5342a4b01fb77c7cf8962cfa3242e4684dc9d3eaf6fd628"
    ),
    "pipelines/static_adapt/sr_snake_route_profile.py": (
        "8f1bf34f760d5047a995ee682bbb9a37ceb4fc2db3744db39d047daa32c9dcb0"
    ),
    "pipelines/scaffold/hh_continuation_scoring.py": (
        "562b5a84494eaa16c50ec8897f1cede392ac7cfef4c5d9b40be058d7993235b2"
    ),
    "pipelines/scaffold/hh_continuation_pruning.py": (
        "3b8be9adce5e52d7beab8fc66bcb4e2252327821c50eb4a2e82c5a1aee0f7ada"
    ),
    "pipelines/exact_bench/paper_i_hh_recovery_prefix_qiskit_sidecar.py": (
        "5486c285deffcb47fd0f5ef0314a9e3ab2fd1c83ebb7e0bb72d629d6a81dd044"
    ),
    "pipelines/exact_bench/table_i_qiskit_resource_compile.py": (
        "cdc182772288593de6087049470a8b6bb47a00c254cd6176276eda63320d19cd"
    ),
    "pipelines/hardcoded/adapt_circuit_execution.py": (
        "1b569d31a45f98522b615fba0bb5645a6fba8af63ecc338f1059f14623364a0e"
    ),
    "pipelines/qiskit_backend_tools.py": (
        "46fcfcce70479b5cad5346b456b689531d4f28fbc1200fe5ef22b5c68494c05b"
    ),
    "test/test_static_adapt_sr_trust_prune.py": (
        "38c5787ac4362d02fff6413e2a1675aabc7206a974aff2906c4eb619b96c0616"
    ),
    "agent_guidance/shared/run-guide.md": (
        "5b69a0655af49e449b8ac9f6dc4c12a8ac027cda700ca8a3551eaea83cba4ff9"
    ),
}

REGIMES: tuple[dict[str, Any], ...] = (
    {
        "slug": "weak_weak", "u": "0.25", "lambda": 0.25,
        "g_ep": "0.353553390593", "n_ph": 2,
        "exact_energy": -0.9183531194991743,
        "manifest_sha256": "24b8c50be54acc6eda506b38e9cd0583bd0ef88b1db6dac47e850b88040dc0b0",
        "result_sha256": "68fde0ab9de5ae69cee27ac0f54cb52f9e377882969daa0a1630d14f520ffdaa",
        "memory_mb": 32768, "disk_mb": 61440,
    },
    {
        "slug": "intermediate_weak", "u": "1.25", "lambda": 0.25,
        "g_ep": "0.353553390593", "n_ph": 2,
        "exact_energy": -0.4949956391086595,
        "manifest_sha256": "7ee0d9b6aea0f12418426a232e42a6962e2ef7cfa47f6dc3ba71061f139b1573",
        "result_sha256": "9e2479fd8308f111cd311e7843ccf1978962dc9d66876bc78f5c69566054a2ed",
        "memory_mb": 32768, "disk_mb": 61440,
    },
    {
        "slug": "strong_weak_u8", "u": "8.0", "lambda": 0.25,
        "g_ep": "0.353553390593", "n_ph": 2,
        "exact_energy": 0.5264587007998427,
        "manifest_sha256": "67ff7a01a5cc1a33b34982e0a3511d889e4cc7aa7f93b7ea042d80bcf3ce5c0e",
        "result_sha256": "b62f89ef9271a2ff42eab2057e9183f48900355967adcabc1fbc9491d22a21f6",
        "memory_mb": 40960, "disk_mb": 61440,
    },
    {
        "slug": "weak_strong", "u": "0.25", "lambda": 1.25,
        "g_ep": "0.790569415042", "n_ph": 4,
        "exact_energy": -1.1385792003592516,
        "manifest_sha256": "6862cab52ebc8b49e15cdeada67c873b066918e5260b0e578244b36e52549c56",
        "result_sha256": "aaf2102a7829ac7a2b4c0f13ef55ef96fe81ed3b70f7d110e2d7c001b6d9cf3e",
        "memory_mb": 49152, "disk_mb": 81920,
    },
    {
        "slug": "intermediate_strong", "u": "1.25", "lambda": 1.25,
        "g_ep": "0.790569415042", "n_ph": 4,
        "exact_energy": -0.6239104048313423,
        "manifest_sha256": "ec5d436919af666fdfe1c28e8f243d44163637348451f46a368a73fb4eefd021",
        "result_sha256": "d00c8ab411fd87429f63095e5ab7cbea2c3b6d535228fbbaf3b5f60bf22499b0",
        "memory_mb": 49152, "disk_mb": 81920,
    },
    {
        "slug": "strong_strong_u8", "u": "8.0", "lambda": 1.25,
        "g_ep": "0.790569415042", "n_ph": 4,
        "exact_energy": 0.5205762777107107,
        "manifest_sha256": "097bd59aff835fbfa39d5b603f384503b3372d0e3df2d480cb94d338399a902d",
        "result_sha256": "c0211bcfad1a7518857d17736ce3f7eccc9da9a2f993a8a7770208ac071b4a88",
        "memory_mb": 49152, "disk_mb": 81920,
    },
)

ARCHIVE_PATHS = (
    "src",
    "pipelines",
    "docs/reports",
    "test/test_static_adapt_sr_v4_runtime.py",
    "test/test_static_adapt_sr_v4_serialization.py",
    "test/test_static_adapt_sr_trust_prune.py",
    "agent_guidance/static-adapt/route-identities.md",
    "agent_guidance/shared/run-guide.md",
    "MATH/paper_facing/paper_I_static_scaffold/"
    "paper_i_sr_snake_v4_candidate_runtime_settings_20260716.md",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump_json(path: Path, payload: Any) -> None:
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


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def git_blob(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{EXPECTED_HEAD}:{path}"], cwd=REPO)


def source_result_path(slug: str) -> Path:
    if slug == "weak_weak":
        return WEAK_WEAK_RESULT
    return HISTORICAL_RESULT_ROOT / slug / "json/result.json"


def verify_source_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    head = git_output("rev-parse", "HEAD")
    tree = git_output("rev-parse", "HEAD^{tree}")
    if head != EXPECTED_HEAD or tree != EXPECTED_TREE:
        raise ValueError(
            f"source revision drift: expected {EXPECTED_HEAD}/{EXPECTED_TREE}, "
            f"got {head}/{tree}"
        )
    critical = {}
    for path, expected in CRITICAL_SOURCE_HASHES.items():
        actual = bytes_sha256(git_blob(path))
        if actual != expected:
            raise ValueError(f"critical committed source drift: {path}: {actual}")
        critical[path] = actual

    physics_rows = []
    for row in REGIMES:
        slug = str(row["slug"])
        manifest_path = HISTORICAL_MANIFEST_ROOT / f"{slug}.json"
        result_path = source_result_path(slug)
        if sha256(REPO / manifest_path) != row["manifest_sha256"]:
            raise ValueError(f"historical manifest hash drift: {slug}")
        if sha256(REPO / result_path) != row["result_sha256"]:
            raise ValueError(f"historical result hash drift: {slug}")
        manifest = load_json(REPO / manifest_path)
        result = load_json(REPO / result_path)
        physics = manifest["physics"]
        exact = float(result["ground_state"]["exact_energy"])
        if int(physics["n_ph_work"]) != int(physics["n_ph_reference"]):
            raise ValueError(f"same-cutoff mismatch in historical manifest: {slug}")
        if int(physics["n_ph_work"]) != int(row["n_ph"]):
            raise ValueError(f"working cutoff drift: {slug}")
        if abs(exact - float(row["exact_energy"])) > 1.0e-12:
            raise ValueError(f"exact reference drift: {slug}: {exact}")
        physics_rows.append({
            "regime_slug": slug,
            "u_over_t": float(row["u"]),
            "lambda": float(row["lambda"]),
            "g_ep": float(row["g_ep"]),
            "n_ph_work": int(row["n_ph"]),
            "n_ph_reference": int(row["n_ph"]),
            "same_cutoff_reference": True,
            "expected_exact_energy": float(row["exact_energy"]),
            "exact_energy_tolerance": 1.0e-12,
            "historical_manifest": manifest_path.as_posix(),
            "historical_manifest_sha256": str(row["manifest_sha256"]),
            "historical_result": result_path.as_posix(),
            "historical_result_sha256": str(row["result_sha256"]),
        })

    sys.path.insert(0, str(REPO))
    from pipelines.static_adapt.sr_snake_route_profile import (  # noqa: PLC0415
        canonical_sr_snake_contract,
        canonical_sr_snake_contract_sha256,
        normalize_sr_route_profile_request,
    )
    resolved = normalize_sr_route_profile_request(PROFILE_REQUEST)
    contract = canonical_sr_snake_contract(PROFILE_REQUEST)
    contract_digest = canonical_sr_snake_contract_sha256(PROFILE_REQUEST)
    if resolved != PROFILE_RESOLVED or contract_digest != PROFILE_CONTRACT_SHA256:
        raise ValueError("v4 profile resolution/digest drift")
    if contract["execution_settings"]["adapt_max_depth"] != 30:
        raise ValueError("v4 adapt_max_depth must remain 30")

    revision = {
        "schema": "paper_i_hh_sr_snake_v4_source_revision_v1",
        "git_commit": head,
        "git_tree": tree,
        "profile_request": PROFILE_REQUEST,
        "profile_resolved": resolved,
        "profile_contract_sha256": contract_digest,
        "critical_source_sha256": critical,
    }
    physics_lock = {
        "schema": "paper_i_hh_sr_snake_v4_physics_exact_reference_lock_v1",
        "same_cutoff_required": True,
        "manual_exact_energy_override_forbidden": True,
        "runtime_exact_energy_recomputed": True,
        "runtime_exact_energy_tolerance": 1.0e-12,
        "rows": physics_rows,
    }
    return revision, {"contract": contract, "physics_lock": physics_lock}


def build_source_archive() -> dict[str, Any]:
    completed = subprocess.run(
        ["git", "archive", "--format=tar", EXPECTED_HEAD, "--", *ARCHIVE_PATHS],
        cwd=REPO,
        check=True,
        stdout=subprocess.PIPE,
    )
    compressed = gzip.compress(completed.stdout, compresslevel=9, mtime=0)
    ARCHIVE_PATH.write_bytes(compressed)
    members: dict[str, dict[str, Any]] = {}
    with tarfile.open(fileobj=io.BytesIO(compressed), mode="r:gz") as handle:
        for member in handle.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"unsafe archive member: {member.name}")
            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                raise ValueError(f"archive contains link/special member: {member.name}")
            if any(
                part in {".DS_Store", "__MACOSX"} or part.startswith("._")
                for part in name.parts
            ):
                raise ValueError(f"archive contains macOS metadata: {member.name}")
            if member.isfile():
                stream = handle.extractfile(member)
                if stream is None:
                    raise ValueError(f"unreadable archive member: {member.name}")
                data = stream.read()
                members[name.as_posix()] = {
                    "sha256": bytes_sha256(data), "size_bytes": len(data)
                }
    for path, expected in CRITICAL_SOURCE_HASHES.items():
        if members.get(path, {}).get("sha256") != expected:
            raise ValueError(f"critical file missing/drifted in archive: {path}")
    return {
        "schema": "paper_i_hh_sr_snake_v4_source_archive_manifest_v1",
        "archive": ARCHIVE_PATH.relative_to(REPO).as_posix(),
        "archive_sha256": sha256(ARCHIVE_PATH),
        "archive_size_bytes": ARCHIVE_PATH.stat().st_size,
        "git_commit": EXPECTED_HEAD,
        "git_tree": EXPECTED_TREE,
        "worker_source_mode": "exact_git_archive_only_v1",
        "worker_pythonpath": "/work",
        "file_count": len(members),
        "files": members,
    }


def archive_only_preflight(
    *, archive: dict[str, Any], job_paths: list[Path]
) -> dict[str, Any]:
    """Prove that validation imports the extracted archive, not the live tree."""

    base_relative = BUNDLE_DIR.relative_to(REPO)
    stage_names = (
        "run_job.py",
        "evidence_validation.py",
        "source_locked.tar.gz",
        "source_archive_manifest.json",
        "source_revision_manifest.json",
        "physics_and_exact_reference_lock.json",
    )
    with tempfile.TemporaryDirectory(prefix="sr_v4_archive_preflight_") as tmp:
        root = Path(tmp)
        with tarfile.open(ARCHIVE_PATH, "r:gz") as handle:
            for member in handle.getmembers():
                name = PurePosixPath(member.name)
                if (
                    name.is_absolute()
                    or ".." in name.parts
                    or member.issym()
                    or member.islnk()
                    or not (member.isfile() or member.isdir())
                    or any(
                        part in {".DS_Store", "__MACOSX"} or part.startswith("._")
                        for part in name.parts
                    )
                ):
                    raise ValueError(f"unsafe isolated-preflight member: {member.name}")
            handle.extractall(root, filter="data")
        staged_bundle = root / base_relative
        staged_bundle.mkdir(parents=True, exist_ok=True)
        for name in stage_names:
            source = BUNDLE_DIR / name
            destination = staged_bundle / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        for job_path in job_paths:
            destination = root / job_path.relative_to(REPO)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(job_path, destination)

        env = os.environ.copy()
        env.update({
            "HOME": str(root / "home"),
            "PYTHONPATH": str(root),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        })
        (root / "home").mkdir()
        dependency_env = os.environ.copy()
        dependency_env.update({
            "PYTHONPATH": str(root),
            "PYTHONDONTWRITEBYTECODE": "1",
        })
        probe_code = (
            "import hashlib,json,pathlib; "
            "import pipelines.static_adapt.sr_snake_route_profile as m; "
            "p=pathlib.Path(m.__file__).resolve(); r=pathlib.Path.cwd().resolve(); "
            "rel=p.relative_to(r).as_posix(); "
            "h=hashlib.sha256(p.read_bytes()).hexdigest(); "
            "print(json.dumps({'module':rel,'sha256':h,"
            "'resolved':m.normalize_sr_route_profile_request('sr_snake_v4'),"
            "'digest':m.canonical_sr_snake_contract_sha256('sr_snake_v4')},sort_keys=True))"
        )
        probe = subprocess.run(
            [sys.executable, "-c", probe_code],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        probe_payload: dict[str, Any] = {}
        if probe.returncode == 0:
            probe_payload = json.loads(probe.stdout)
        expected_route_sha = archive["files"][
            "pipelines/static_adapt/sr_snake_route_profile.py"
        ]["sha256"]
        import_pass = bool(
            probe.returncode == 0
            and probe_payload.get("module")
            == "pipelines/static_adapt/sr_snake_route_profile.py"
            and probe_payload.get("sha256") == expected_route_sha
            and probe_payload.get("resolved") == PROFILE_RESOLVED
            and probe_payload.get("digest") == PROFILE_CONTRACT_SHA256
        )

        parse_rows = []
        for job_path in job_paths:
            staged_job = root / job_path.relative_to(REPO)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(staged_bundle / "run_job.py"),
                    "--validate-only",
                    str(staged_job),
                ],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            parsed_stdout = None
            if completed.returncode == 0:
                parsed_stdout = json.loads(completed.stdout)
            parse_rows.append({
                "job": f"jobs/{job_path.name}",
                "returncode": int(completed.returncode),
                "stdout": parsed_stdout,
                "stderr_empty": not bool(completed.stderr.strip()),
            })

        helper_relative = Path(
            "pipelines/exact_bench/paper_i_hh_recovery_prefix_qiskit_sidecar.py"
        )
        helper = root / helper_relative
        helper_present = helper.is_file()
        helper_help_returncode = None
        if helper_present:
            helper_help = subprocess.run(
                [sys.executable, str(helper), "--help"],
                cwd=root,
                env=dependency_env,
                capture_output=True,
                text=True,
                check=False,
            )
            helper_help_returncode = int(helper_help.returncode)
        focused_test_paths = [
            "test/test_static_adapt_sr_trust_prune.py",
            "test/test_static_adapt_sr_v4_runtime.py",
            "test/test_static_adapt_sr_v4_serialization.py",
        ]
        focused_tests = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *focused_test_paths],
            cwd=root,
            env=dependency_env,
            capture_output=True,
            text=True,
            check=False,
        )
        focused_44_pass = bool(
            focused_tests.returncode == 0
            and "44 passed" in focused_tests.stdout
        )
        all_parse = all(row["returncode"] == 0 for row in parse_rows)
        return {
            "schema": "paper_i_hh_sr_snake_v4_archive_only_preflight_v1",
            "status": (
                "pass" if import_pass and all_parse and helper_present
                and helper_help_returncode == 0 and focused_44_pass else "blocked"
            ),
            "archive_sha256": archive["archive_sha256"],
            "source_import": {
                "status": "pass" if import_pass else "fail",
                "module": probe_payload.get("module"),
                "sha256": probe_payload.get("sha256"),
                "profile_resolved": probe_payload.get("resolved"),
                "profile_contract_sha256": probe_payload.get("digest"),
            },
            "six_validate_only_parses": parse_rows,
            "all_six_validate_only_pass": all_parse,
            "qiskit_helper": {
                "path": helper_relative.as_posix(),
                "present": helper_present,
                "help_returncode": helper_help_returncode,
                "help_pass": helper_help_returncode == 0,
            },
            "focused_source_locked_regressions": {
                "paths": focused_test_paths,
                "expected_pass_count": 44,
                "returncode": int(focused_tests.returncode),
                "pass": focused_44_pass,
                "stderr_empty": not bool(focused_tests.stderr.strip()),
                "dependency_environment": (
                    "local_python_packages_with_extracted_archive_source_only"
                ),
            },
            "live_repo_import_excluded": import_pass,
        }


def job_command(row: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    slug = str(row["slug"])
    root = Path("raw_outputs") / BUNDLE_ID / slug
    paths = {
        "output_root": root.as_posix(),
        "result_json": (root / "json/result.json").as_posix(),
        "current_json": (root / "json/current.json").as_posix(),
        "ledger_json": (root / "json/estimator_call_ledger.json").as_posix(),
        "execution_json": (root / "execution.json").as_posix(),
        "normalized_runtime_manifest_json": (
            root / "normalized_run_manifest.json"
        ).as_posix(),
        "validation_json": (root / "validation.json").as_posix(),
        "qiskit_cost_sidecar_json": (
            root / "qiskit_cost_sidecar.json"
        ).as_posix(),
        "repaired_terminal_checkpoint_json": (
            root / "terminal_checkpoint.execution_order_repaired.json"
        ).as_posix(),
    }
    argv = [
        "python3", "-m", "pipelines.static_adapt.adapt_pipeline",
        "--problem", "hh", "--L", "2", "--ordering", "blocked",
        "--boundary", "open", "--t", "1.0", "--dv", "0.0",
        "--omega0", "1.0", "--boson-encoding", "binary",
        "--u", str(row["u"]), "--g-ep", str(row["g_ep"]),
        "--n-ph-max", str(row["n_ph"]),
        "--sr-route-profile", PROFILE_REQUEST,
        "--adapt-segment-id", f"{slug}-sr-v4-r0-r30-20260716-v1",
        "--adapt-segment-target-controller-round", "30",
        "--adapt-segment-target-depth", "30",
        "--adapt-segment-max-new-admissions", "30",
        "--adapt-current-json-every-depth", "1",
        "--adapt-current-json", paths["current_json"],
        "--adapt-estimator-call-ledger-json", paths["ledger_json"],
        "--output-json", paths["result_json"],
        "--skip-pdf",
    ]
    return argv, paths


def build_job(
    row: dict[str, Any], contract: dict[str, Any], archive: dict[str, Any]
) -> dict[str, Any]:
    argv, paths = job_command(row)
    slug = str(row["slug"])
    environment = {
        "PYTHONPATH": "/work",
        "PYTHONUNBUFFERED": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLCONFIGDIR": f"{paths['output_root']}/cache/matplotlib",
        "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "disk",
        "STATIC_ADAPT_CANDIDATE_RECORD_CACHE_DIR": (
            f"{paths['output_root']}/cache/candidate_records"
        ),
        "STATIC_ADAPT_HH_GENERATOR_REGISTRY_CACHE_DIR": (
            f"{paths['output_root']}/cache/hh_generator_registry"
        ),
        "STATIC_ADAPT_HH_POOL_CACHE": "disk",
        "STATIC_ADAPT_HH_POOL_CACHE_DIR": f"{paths['output_root']}/cache/hh_pool",
        "STATIC_ADAPT_HH_POOL_CACHE_SCOPE": "exact",
    }
    return {
        "schema": "paper_i_hh_sr_snake_v4_candidate_parent_job_v1",
        "bundle_id": BUNDLE_ID,
        "batch_name": BATCH_NAME,
        "run_class": "candidate",
        "regime_slug": slug,
        "route_identity": {
            "family": "singleton_response_snake",
            "profile_request": PROFILE_REQUEST,
            "profile_resolved": PROFILE_RESOLVED,
            "profile_contract_sha256": PROFILE_CONTRACT_SHA256,
            "profile_contract": contract,
        },
        "physics": {
            "problem": "hh", "L": 2, "ordering": "blocked",
            "boundary": "open", "t": 1.0, "dv": 0.0, "omega0": 1.0,
            "u_over_t": float(row["u"]), "lambda": float(row["lambda"]),
            "g_ep": float(row["g_ep"]), "n_ph_work": int(row["n_ph"]),
            "n_ph_reference": int(row["n_ph"]),
            "same_cutoff_reference": True,
            "expected_exact_energy": float(row["exact_energy"]),
            "exact_energy_tolerance": 1.0e-12,
        },
        "segment": {
            "source_controller_round": 0, "source_depth": 0,
            "target_controller_round": 30, "target_depth": 30,
            "max_new_admissions": 30,
            "future_continuation_required_after_validation": slug not in {
                "weak_weak", "intermediate_weak"
            },
            "future_continuation_target": 50 if slug not in {
                "weak_weak", "intermediate_weak"
            } else None,
            "terminal_qiskit_sidecar_outer_iteration": 30,
            "terminal_qiskit_sidecar_required": True,
            "terminal_qiskit_checkpoint_order_policy": (
                "repair_permutation_only_execution_order_fail_closed_v1"
            ),
        },
        "command": {
            "argv": argv,
            "method_configuration_surface": "sr_route_profile_only",
            "explicit_method_overrides": [],
            "manual_exact_reference_override": False,
        },
        "environment": environment,
        "paths": paths,
        "source_lock": {
            "git_commit": EXPECTED_HEAD, "git_tree": EXPECTED_TREE,
            "source_archive": archive["archive"],
            "source_archive_sha256": archive["archive_sha256"],
            "physics_reference_lock": (
                BUNDLE_DIR.relative_to(REPO) / "physics_and_exact_reference_lock.json"
            ).as_posix(),
            "physics_reference_lock_sha256": sha256(
                BUNDLE_DIR / "physics_and_exact_reference_lock.json"
            ),
            "source_revision_manifest": (
                BUNDLE_DIR.relative_to(REPO) / "source_revision_manifest.json"
            ).as_posix(),
            "source_revision_manifest_sha256": sha256(
                BUNDLE_DIR / "source_revision_manifest.json"
            ),
            "source_archive_manifest": (
                BUNDLE_DIR.relative_to(REPO) / "source_archive_manifest.json"
            ).as_posix(),
            "source_archive_manifest_sha256": sha256(
                BUNDLE_DIR / "source_archive_manifest.json"
            ),
            "historical_manifest": (
                HISTORICAL_MANIFEST_ROOT / f"{slug}.json"
            ).as_posix(),
            "historical_manifest_sha256": str(row["manifest_sha256"]),
            "historical_result": source_result_path(slug).as_posix(),
            "historical_result_sha256": str(row["result_sha256"]),
        },
        "resource_request": {
            "cpus": 4, "memory_mb": int(row["memory_mb"]),
            "disk_mb": int(row["disk_mb"]), "max_runtime_s": 259200,
        },
    }


def submit_text(archive_sha: str) -> str:
    base = (BUNDLE_DIR.relative_to(REPO)).as_posix()
    requirements = "TARGET.HasSIF" if SUBMISSION_ENABLED else "False"
    return f"""universe = vanilla
# Source lock, local/archive-only gates, and remote image checks are frozen.
# The user authorized submission; the authenticated main agent submits this file.
executable = {base}/execute_source_locked_job.sh
arguments = $(job_manifest) {base}/source_locked.tar.gz {archive_sha} chtc/phase3_optuna/image.sif {REMOTE_IMAGE_SHA256} $(regime_slug)
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_executable = True
preserve_relative_paths = True
transfer_input_files = {base}/run_job.py, {base}/evidence_validation.py, {base}/validate_fetched.py, {base}/source_archive_manifest.json, {base}/source_revision_manifest.json, {base}/physics_and_exact_reference_lock.json, {base}/bundle_manifest.json, {base}/preflight.json, {base}/route_parity.json, $(job_manifest), $(normalized_manifest), {base}/source_locked.tar.gz, chtc/phase3_optuna/image.sif
transfer_output_files = raw_outputs/{BUNDLE_ID}/$(regime_slug)_transfer.tar.gz
transfer_output_remaps = "raw_outputs/{BUNDLE_ID}/$(regime_slug)_transfer.tar.gz = $(regime_slug)_transfer.tar.gz"
stream_output = False
stream_error = False
log = logs/{BUNDLE_ID}.$(Cluster).$(Process).log
output = logs/{BUNDLE_ID}.$(Cluster).$(Process).out
error = logs/{BUNDLE_ID}.$(Cluster).$(Process).err
requirements = {requirements}
request_cpus = 4
request_memory = $(memory_mb)MB
request_disk = $(disk_mb)MB
+MaxRuntime = 259200
+JobBatchName = \"{BATCH_NAME}\"
notification = Never
queue regime_slug, job_manifest, normalized_manifest, memory_mb, disk_mb from {base}/queue.tsv
"""


def smoke_summary() -> dict[str, Any]:
    expected_hashes = {
        "json/result.json": (
            "89eb332a23ba25fb2dbafb66438b97cf40b3c79a31b168b71c2537cc4958da22"
        ),
        "json/current.json": (
            "cd42ff02b99973b1ff6dbb2334d41bd129f2a6c511ab3ad5c821403a05d85192"
        ),
        "json/estimator_call_ledger.json": (
            "e0cec128682dc65557ea821c4f3acd2856f72bd9009795826c4411271778cc7e"
        ),
        "json/qiskit_cost_sidecar.json": (
            "700597f75bc398299908d5771bfba8c824120105c19c81b2ec96b66bbe08917e"
        ),
        "json/qiskit_compile_repaired_checkpoint.json": (
            "669383efe1f87f4a07a89898740f2c27c2bed2c5b08d86eb0ea121b5a489e612"
        ),
        "source_lock/anchor_command.json": (
            "85c9a424649715c218983bed53cab08554811f360f58d48275f0a9c9b03423f5"
        ),
        "source_lock/validation_summary.json": (
            "e0a2b662824717ddd91a10656c898e6ea84472bcbdaf232ea52bdb4c3b74a9a7"
        ),
    }
    evidence_hashes: dict[str, str] = {}
    for relative, expected in expected_hashes.items():
        path = REPO / SMOKE8_ROOT / relative
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"depth-8 repaired smoke hash drift: {relative}: {actual}")
        evidence_hashes[relative] = actual

    result = load_json(REPO / SMOKE8_ROOT / "json/result.json")
    current = load_json(REPO / SMOKE8_ROOT / "json/current.json")
    ledger = load_json(REPO / SMOKE8_ROOT / "json/estimator_call_ledger.json")
    validation = load_json(
        REPO / SMOKE8_ROOT / "source_lock/validation_summary.json"
    )
    from evidence_validation import validate_parent_evidence  # noqa: PLC0415

    scientific_validation = validate_parent_evidence(
        result=result,
        current=current,
        ledger_sidecar=ledger,
        profile=PROFILE_RESOLVED,
        digest=PROFILE_CONTRACT_SHA256,
        target_round=8,
        target_new_admissions=8,
        require_supported_rank=True,
    )
    if validation.get("status") != "pass":
        raise ValueError("depth-8 repaired smoke validation summary did not pass")
    required_smoke_gates = {
        "route_contract_matches",
        "full_active_plus_singleton_response_every_round",
        "response_supported_rank_nonnull_every_round",
        "accepted_refit_full_ansatz_every_round",
        "adaptive_trust_updated_exactly_once_per_round",
        "state_keyed_estimator_ledger_complete",
        "terminal_refit_executed",
        "terminal_prune_executed",
        "qiskit_fixed_prefix_replay",
        "qiskit_permutation_only_execution_order_repair",
    }
    gates = validation.get("gates", {})
    for gate in required_smoke_gates:
        expected = False if gate in {
            "terminal_refit_executed", "terminal_prune_executed"
        } else True
        if gates.get(gate) is not expected:
            raise ValueError(f"depth-8 repaired smoke gate failed: {gate}")
    if gates.get("shadow_damping_status") != "explicit_unresolved_zero_query_noop":
        raise ValueError("depth-8 repaired smoke shadow-damping receipt drift")

    return {
        "schema": "paper_i_hh_sr_snake_v4_local_smoke_evidence_v1",
        "status": "pass_for_bundle_construction_not_a_production_result",
        "records": [{
            "label": "eight_admission_nonbeam_overlay_trust_repair",
            "root": SMOKE8_ROOT.as_posix(),
            "evidence_sha256": evidence_hashes,
            "exit_success": bool(result.get("adapt_vqe", {}).get("success")),
            "admissions": len(result.get("adapt_vqe", {}).get("history", [])),
            "profile_resolved": result.get("settings", {}).get(
                "sr_route_profile_resolved"
            ),
            "profile_contract_sha256": result.get("settings", {}).get(
                "sr_route_profile_contract_sha256"
            ),
            "same_cutoff_exact_energy": result.get("ground_state", {}).get(
                "exact_energy"
            ),
            "scientific_evidence_validation": scientific_validation,
        }],
        "production_composition_prune_gate": {
            "kind": "focused_source_locked_regression",
            "test_file": "test/test_static_adapt_sr_trust_prune.py",
            "test_file_sha256": CRITICAL_SOURCE_HASHES[
                "test/test_static_adapt_sr_trust_prune.py"
            ],
            "one_nominee": True,
            "one_measured_delete_refit_trial": True,
            "measured_energy_is_acceptance_authority": True,
            "conservative_rho_mu_update": True,
            "added_quantum_query_count": 0,
            "live_only_no_terminal_pruning": True,
            "archive_only_execution_record": "archive_only_preflight.json",
        },
        "passed_gates": [
            "route_profile_and_contract_digest",
            "full_phase3_pre_support_coordinate_count",
            "phase3_response_supported_rank_recorded",
            "full_accepted_refit_coordinate_count",
            "adaptive_trust_one_update_per_accepted_refit",
            "symmetry_and_padding_leakage",
            "checkpoint_roundtrip_and_fixed_prefix_replay",
            "estimator_ledger_closure",
            "shadow_damping_explicit_diagnostic_noop",
            "safe_all_infeasible_prune_skip",
            "production_composition_one_nominee_one_measured_prune_trial",
        ],
        "exact_blockers": [],
        "note": (
            "Diagnostic energies are not production comparisons; these records "
            "establish executable route/profile, checkpoint, trust, and prune-path "
            "health. The prune transaction gate is a source-locked production-"
            "composition regression, not a threshold-modified scientific smoke."
        ),
    }


def remote_preflight_and_cleanup_receipt() -> dict[str, Any]:
    preserved = [
        Path(
            "raw_outputs/chtc_fetch_paper_i_hh_sr_20260716/"
            "paper_i_hh_sr_snake_noprune_nobeam_no_ordinary_novelty_"
            "r50_continuations_20260715_v1_chtc"
        ),
        Path(
            "raw_outputs/chtc_fetch_paper_i_hh_sr_20260716/"
            "paper_i_hh_sr_snake_noprune_nobeam_ordinary_novelty_all_six_"
            "20260715_v1_chtc"
        ),
    ]
    validation_records = {
        Path(
            "raw_outputs/chtc_fetch_paper_i_hh_sr_20260716/"
            "validation_cluster_8811168_20260716/"
            "baseline_continuation_validation.json"
        ): "d92244a9ea7902a5bbabd8e6b3131f0729c2c19ce30b4ee5ee122bbcd1b19175",
        Path(
            "raw_outputs/chtc_fetch_paper_i_hh_sr_20260716/"
            "validation_study_b_8811165/study_b_validation_summary.json"
        ): "326d0d66aa3702618d9aaf58ff7602fa0780cb21017d236c97919998d15e4adc",
    }
    for path in preserved:
        if not (REPO / path).is_dir():
            raise ValueError(f"locally preserved fetched evidence missing: {path}")
    for path, expected in validation_records.items():
        if sha256(REPO / path) != expected:
            raise ValueError(f"local validation-record hash drift: {path}")
    return {
        "schema": "paper_i_hh_sr_snake_v4_remote_preflight_cleanup_receipt_v1",
        "status": "pass",
        "remote_execution_preflight": {
            "image_path": REMOTE_IMAGE_PATH.as_posix(),
            "image_sha256": REMOTE_IMAGE_SHA256,
            "qiskit_import_passed": True,
            "qiskit_version": REMOTE_QISKIT_VERSION,
            "fake_backend_instantiation_passed": True,
            "fake_backend_resolved": REMOTE_FAKE_BACKEND_RESOLVED,
            "fake_backend_qubits": REMOTE_FAKE_BACKEND_QUBITS,
        },
        "storage_cleanup": {
            "scope": "two completed SR output directories already fetched_and_validated",
            "remote_removed_paths": [
                (
                    "/home/jsstrobel/Holstein_phase3_optuna_chtc/raw_outputs/"
                    "paper_i_hh_sr_snake_noprune_nobeam_no_ordinary_novelty_"
                    "r50_continuations_20260715_v1_chtc"
                ),
                (
                    "/home/jsstrobel/Holstein_phase3_optuna_chtc/raw_outputs/"
                    "paper_i_hh_sr_snake_noprune_nobeam_ordinary_novelty_"
                    "all_six_20260715_v1_chtc"
                ),
            ],
            "remote_absence_check": "CLEANUP_OK",
            "post_cleanup_quota": {
                "filesystem": "/dev/md9",
                "space_used": "38917M",
                "quota": "40960M",
                "limit": "51200M",
                "files": 42540,
            },
            "unrelated_remote_paths_modified": False,
            "local_preserved_roots": [path.as_posix() for path in preserved],
            "local_preservation_verified": True,
            "local_validation_records": [
                {"path": path.as_posix(), "sha256": digest}
                for path, digest in validation_records.items()
            ],
        },
        "submission_performed": False,
    }


def main() -> int:
    revision, verified = verify_source_lock()
    contract = verified["contract"]
    dump_json(BUNDLE_DIR / "source_revision_manifest.json", revision)
    dump_json(
        BUNDLE_DIR / "physics_and_exact_reference_lock.json",
        verified["physics_lock"],
    )
    archive = build_source_archive()
    dump_json(BUNDLE_DIR / "source_archive_manifest.json", archive)
    smoke = smoke_summary()
    dump_json(BUNDLE_DIR / "source_lock/local_smoke_evidence.json", smoke)
    remote_receipt = remote_preflight_and_cleanup_receipt()
    dump_json(
        BUNDLE_DIR / "remote_preflight_and_cleanup_receipt.json",
        remote_receipt,
    )

    queue_lines = []
    job_paths = []
    for row in REGIMES:
        job = build_job(row, contract, archive)
        slug = str(row["slug"])
        job_path = BUNDLE_DIR / "jobs" / f"{slug}.json"
        normalized_path = BUNDLE_DIR / "normalized_manifests" / f"{slug}.json"
        dump_json(job_path, job)
        dump_json(normalized_path, {
            "schema": "paper_i_hh_sr_snake_v4_normalized_parent_manifest_v1",
            "bundle_id": BUNDLE_ID,
            "regime_slug": slug,
            "route_identity": job["route_identity"],
            "physics": job["physics"],
            "segment": job["segment"],
            "command_argv": job["command"]["argv"],
            "environment": job["environment"],
            "source_lock": job["source_lock"],
            "resource_request": job["resource_request"],
        })
        job_paths.append(job_path)
        queue_lines.append("\t".join((
            slug,
            job_path.relative_to(REPO).as_posix(),
            normalized_path.relative_to(REPO).as_posix(),
            str(row["memory_mb"]), str(row["disk_mb"]),
        )))
    (BUNDLE_DIR / "queue.tsv").write_text("\n".join(queue_lines) + "\n")
    (BUNDLE_DIR / "submit.sub").write_text(
        submit_text(str(archive["archive_sha256"])), encoding="utf-8"
    )

    parse_rows = []
    for job_path in job_paths:
        completed = subprocess.run(
            [sys.executable, str(BUNDLE_DIR / "run_job.py"), "--validate-only", str(job_path)],
            cwd=REPO, capture_output=True, text=True, check=False,
        )
        parsed_stdout = None
        if completed.returncode == 0:
            parsed_stdout = json.loads(completed.stdout)
        parse_rows.append({
            "job": job_path.relative_to(REPO).as_posix(),
            "returncode": completed.returncode,
            "stdout": parsed_stdout,
            "stderr_empty": not bool(completed.stderr.strip()),
        })
        if completed.returncode != 0:
            raise ValueError(f"job validation failed: {job_path}: {completed.stderr}")

    isolated = archive_only_preflight(archive=archive, job_paths=job_paths)
    dump_json(BUNDLE_DIR / "archive_only_preflight.json", isolated)
    if isolated.get("status") != "pass":
        raise ValueError("archive-only import/parse/helper/regression preflight failed")

    continuation_template = {
        "schema": "paper_i_hh_sr_snake_v4_round30_to_round50_continuation_template_v1",
        "status": "template_only_not_executable",
        "materialization_gate": (
            "materialize one row only after its round-30 parent transfer archive "
            "passes validate_fetched.py; replace every null authenticated lock "
            "with the fetched artifact SHA-256"
        ),
        "source_bundle_id": BUNDLE_ID,
        "source_archive": archive["archive"],
        "source_archive_sha256": archive["archive_sha256"],
        "profile_request": PROFILE_REQUEST,
        "profile_resolved": PROFILE_RESOLVED,
        "profile_contract_sha256": PROFILE_CONTRACT_SHA256,
        "eligible_regimes": [
            "strong_weak_u8", "weak_strong", "intermediate_strong",
            "strong_strong_u8",
        ],
        "segment_contract": {
            "source_controller_round": 30,
            "source_depth": "from_authenticated_round30_signed_checkpoint",
            "target_controller_round": 50,
            "target_depth_cap": 50,
            "max_new_admissions": 20,
            "resume_mode": "scaffold_v1",
            "boundary_policy": "verified_checkpoint_no_refit_v1",
        },
        "compile_smoke": {
            "required": True,
            "backend": "FakeMarrakesh",
            "must_precede_scientific_continuation": True,
        },
        "required_authenticated_locks_per_row": {
            "source_result_sha256": None,
            "source_signed_checkpoint_sha256": None,
            "source_estimator_ledger_sha256": None,
            "source_checkpoint_outer_iteration": 30,
            "source_checkpoint_kind": "post_admission_prune",
            "source_route_profile_contract_sha256": PROFILE_CONTRACT_SHA256,
        },
        "rows": [
            {
                "regime_slug": slug,
                "source_result_sha256": None,
                "source_signed_checkpoint_sha256": None,
                "source_estimator_ledger_sha256": None,
                "materialized": False,
            }
            for slug in (
                "strong_weak_u8", "weak_strong", "intermediate_strong",
                "strong_strong_u8",
            )
        ],
    }
    dump_json(BUNDLE_DIR / "future_round30_to_round50_continuation_template.json", continuation_template)

    image_local = REPO / REMOTE_IMAGE_PATH
    image_local_present = image_local.is_file()
    image_local_match = image_local_present and sha256(image_local) == REMOTE_IMAGE_SHA256
    qiskit_helper = "pipelines/exact_bench/paper_i_hh_recovery_prefix_qiskit_sidecar.py"
    qiskit_helper_archived = qiskit_helper in archive["files"]
    parity = {
        "schema": "paper_i_hh_sr_snake_v4_route_parity_v1",
        "status": "pass",
        "profile_request": PROFILE_REQUEST,
        "profile_resolved": PROFILE_RESOLVED,
        "profile_contract_sha256": PROFILE_CONTRACT_SHA256,
        "all_six_commands_parse": all(row["returncode"] == 0 for row in parse_rows),
        "all_targets_round_30": True,
        "all_max_new_admissions_30": True,
        "no_profile_method_flag_repetition": True,
        "same_cutoff_lock_pass": True,
        "parse_rows": parse_rows,
    }
    dump_json(BUNDLE_DIR / "route_parity.json", parity)
    bundle_manifest = {
        "schema": "paper_i_hh_sr_snake_v4_candidate_bundle_v1",
        "bundle_id": BUNDLE_ID,
        "batch_name": BATCH_NAME,
        "created_utc": utc_now(),
        "run_class": "candidate",
        "parent_stage": "fresh_round0_to_round30_all_six",
        "deeper_continuation_stage": (
            "build only after these parents are fetched and validated; "
            "SW/WS/IS/SS continue authenticated round30 prefixes to round50"
        ),
        "source_revision": revision,
        "source_archive": archive,
        "route_parity": parity,
        "archive_only_preflight": (
            BUNDLE_DIR.relative_to(REPO) / "archive_only_preflight.json"
        ).as_posix(),
        "future_round30_to_round50_continuation_template": (
            BUNDLE_DIR.relative_to(REPO)
            / "future_round30_to_round50_continuation_template.json"
        ).as_posix(),
        "physics_reference_lock": (
            BUNDLE_DIR.relative_to(REPO) / "physics_and_exact_reference_lock.json"
        ).as_posix(),
        "remote_preflight_and_cleanup_receipt": (
            BUNDLE_DIR.relative_to(REPO)
            / "remote_preflight_and_cleanup_receipt.json"
        ).as_posix(),
        "remote_preflight_and_cleanup_receipt_sha256": sha256(
            BUNDLE_DIR / "remote_preflight_and_cleanup_receipt.json"
        ),
        "jobs": [path.relative_to(REPO).as_posix() for path in job_paths],
        "source_lock_state": SOURCE_LOCK_STATE,
        "submission_status": "submission_ready_not_yet_submitted",
    }
    dump_json(BUNDLE_DIR / "bundle_manifest.json", bundle_manifest)
    preflight = {
        "schema": "paper_i_hh_sr_snake_v4_candidate_preflight_v1",
        "created_utc": utc_now(),
        "status": "pass_submission_ready_not_yet_submitted",
        "checks": {
            "exact_git_revision": True,
            "critical_source_hashes": True,
            "source_archive_safe_and_closed": True,
            "six_job_manifests": len(job_paths) == 6,
            "all_job_validations": all(row["returncode"] == 0 for row in parse_rows),
            "same_cutoff_reference_lock": True,
            "v4_profile_and_digest": True,
            "fresh_round0_to_round30_only": True,
            "no_adapt_max_depth_override": True,
            "no_repeated_method_flags": True,
            "worker_pythonpath_archive_only": True,
            "isolated_archive_source_import": (
                isolated["source_import"]["status"] == "pass"
            ),
            "isolated_archive_all_six_validate_only": bool(
                isolated["all_six_validate_only_pass"]
            ),
            "isolated_archive_qiskit_helper_help": bool(
                isolated["qiskit_helper"]["help_pass"]
            ),
            "isolated_archive_focused_44_regressions": bool(
                isolated["focused_source_locked_regressions"]["pass"]
            ),
            "future_round30_to_round50_template_only": True,
            "submission_enabled": SUBMISSION_ENABLED,
            "terminal_qiskit_sidecar_required": True,
            "qiskit_sidecar_helper_in_source_archive": qiskit_helper_archived,
            "qiskit_backend_availability_remote_check": True,
            "remote_image_sha256_rechecked": True,
            "remote_qiskit_import_passed": True,
            "remote_fake_marrakesh_instantiation_passed": True,
            "local_image_present": image_local_present,
            "local_image_hash_matches_prior_remote_digest": image_local_match,
            "phase3_response_supported_rank_recorded": True,
            "shadow_damping_scientific_application_expected": False,
            "shadow_damping_diagnostic_noop_receipt_recorded": True,
            "production_composition_delete_refit_prune_regression_passed": True,
        },
        "remote_image": {
            "path": REMOTE_IMAGE_PATH.as_posix(),
            "verified_remote_sha256": REMOTE_IMAGE_SHA256,
            "qiskit_version": REMOTE_QISKIT_VERSION,
            "fake_backend_resolved": REMOTE_FAKE_BACKEND_RESOLVED,
            "fake_backend_qubits": REMOTE_FAKE_BACKEND_QUBITS,
            "local_copy_present": image_local_present,
            "remote_recheck_passed": True,
        },
        "scientific_blockers": [],
        "submission_blockers": [],
        "submission_authorized": True,
        "submission_performed": False,
    }
    dump_json(BUNDLE_DIR / "preflight.json", preflight)

    upload = [
        (BUNDLE_DIR / name).relative_to(REPO).as_posix()
        for name in (
            "execute_source_locked_job.sh", "run_job.py", "evidence_validation.py",
            "validate_fetched.py",
            "source_locked.tar.gz", "source_archive_manifest.json",
            "source_revision_manifest.json", "physics_and_exact_reference_lock.json",
            "bundle_manifest.json", "preflight.json", "route_parity.json",
            "archive_only_preflight.json",
            "future_round30_to_round50_continuation_template.json",
            "SUBMISSION_READY_NOT_YET_SUBMITTED.md",
            "remote_preflight_and_cleanup_receipt.json",
            "queue.tsv", "submit.sub",
            "source_lock/local_smoke_evidence.json",
        )
    ] + [path.relative_to(REPO).as_posix() for path in job_paths] + [
        (BUNDLE_DIR / "normalized_manifests" / path.name).relative_to(REPO).as_posix()
        for path in job_paths
    ]
    (BUNDLE_DIR / "upload_artifact_list.txt").write_text(
        "\n".join(upload) + "\n", encoding="utf-8"
    )

    inventory: dict[str, Any] = {}
    for path in sorted(BUNDLE_DIR.rglob("*")):
        if (
            path.is_file()
            and path.name != "submission_artifact_hashes.json"
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        ):
            inventory[path.relative_to(REPO).as_posix()] = {
                "sha256": sha256(path), "size_bytes": path.stat().st_size
            }
    dump_json(BUNDLE_DIR / "submission_artifact_hashes.json", {
        "schema": "paper_i_hh_sr_snake_v4_submission_artifact_hashes_v1",
        "artifacts": inventory,
    })
    print(json.dumps({
        "status": preflight["status"],
        "bundle": BUNDLE_DIR.relative_to(REPO).as_posix(),
        "source_archive_sha256": archive["archive_sha256"],
        "jobs": len(job_paths),
        "scientific_blockers": preflight["scientific_blockers"],
        "submission_blockers": preflight["submission_blockers"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
