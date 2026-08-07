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
import site
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


BUNDLE_ID = "paper_i_hh_sr_snake_v4_candidate_all_six_20260716_v6_chtc"
BATCH_NAME = "paper-i-hh-sr-v4-candidate-six-r30-20260716-v6"
V2_BUNDLE_ID = "paper_i_hh_sr_snake_v4_candidate_all_six_20260716_v2_chtc"
BUNDLE_DIR = Path(__file__).resolve().parent
REPO = BUNDLE_DIR.parents[3]
V2_BUNDLE_DIR = BUNDLE_DIR.parent / V2_BUNDLE_ID
# Frozen tested source for the first-order/fail-closed Phase-I/II route.
# The worktree may contain unrelated later edits; all archived source and
# critical hashes are read from this exact commit rather than from live files.
EXPECTED_HEAD = "92cf00bb1e7c5c58cc2328c29cdcae9d772adfc0"
EXPECTED_TREE = "5608d20f6b77d200fa90cfdc0ec5e86feb89a71c"
PROFILE_REQUEST = "sr_snake_v4"
PROFILE_RESOLVED = (
    "supported_whitened_adaptive_trust_full_response_symmetric_cost_fs_prune_v4"
)
PROFILE_CONTRACT_SHA256 = (
    "b6331521fb55f4165e177466536b4e2a5834ff09205ab5532ea70de893f156bc"
)
REMOTE_IMAGE_PATH = Path("chtc/phase3_optuna/image.sif")
REMOTE_IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)
REMOTE_QISKIT_VERSION = "2.3.1"
REMOTE_FAKE_BACKEND_RESOLVED = "fake_marrakesh"
REMOTE_FAKE_BACKEND_QUBITS = 156
ARCHIVE_PATH = BUNDLE_DIR / "source_locked.tar.gz"
SOURCE_LOCK_STATE = "frozen_geometry_fallback_trust_repair_commit"
# Keep Condor matchmaking impossible until the main agent has frozen the source
# and confirmed every local, archive-only, smoke, and remote execution gate.
SUBMISSION_ENABLED = True

PHASE1_ENERGY_MODEL = "first_order_fs_trust_v1"
PHASE2_CURVATURE_POLICY = "measured_required_fail_closed_v1"
PHASE2_CHEAP_CURVATURE_PROXY_POLICY = "off"
ALLOWED_V2_TO_V3_CONTRACT_DIFF_PATHS = {
    "execution_settings.adapt_disable_hh_seed",
    "execution_settings.phase1_energy_model",
    "execution_settings.phase1_score_mode",
    "execution_settings.phase2_cheap_curvature_proxy_policy",
    "execution_settings.phase2_curvature_policy",
    "semantic_invariants.phase1_energy_model",
    "semantic_invariants.hh_preseed_policy",
    "semantic_invariants.phase1_fs_metric_role",
    "semantic_invariants.phase1_phase2_lambda_f_proxy_active",
    "semantic_invariants.phase2_cheap_curvature_proxy_policy",
    "semantic_invariants.phase2_curvature_failure_policy",
    "semantic_invariants.phase2_curvature_policy",
}

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
    "raw_outputs/paper_i_hh_sr_snake_v4_phase12_no_lambda_f_nph3_smoke_20260716/"
    "weak_weak_8_admissions_cache_off_seedless_v5"
)

NONSCIENTIFIC_ARCHIVE_OVERLAYS = {
    "pipelines/hardcoded/adapt_pipeline.py": (
        "93c0e91cd01981f5bfa1e9d1434b74296395ca0837f2901ca66ad18ac63dd42f"
    ),
    "pipelines/hardcoded/hh_continuation_scoring.py": (
        "f25b2ae3f4037758c5f1942e6e3e0c75df04f9c5c7008d8b8dacfa9f150aa492"
    ),
    "pipelines/hardcoded/hh_continuation_generators.py": (
        "8c6292c5c71f67312bdc32afc8d3908a83cb550b8f2d8871ed7f7183824e6570"
    ),
    "pipelines/hardcoded/hh_continuation_symmetry.py": (
        "5f61b9c43c253fb81bc354aace4e015f0c4f06a1e8aa8a48b24a43a11b341e01"
    ),
    "pipelines/hardcoded/hh_continuation_types.py": (
        "f24b1a670179ec17c05132b3b65f9541db54ffd888429951f7e17d6aaaf41f4c"
    ),
}
NONSCIENTIFIC_ARCHIVE_OVERLAY_SIZES = {
    "pipelines/hardcoded/adapt_pipeline.py": 1807,
    "pipelines/hardcoded/hh_continuation_scoring.py": 658,
    "pipelines/hardcoded/hh_continuation_generators.py": 664,
    "pipelines/hardcoded/hh_continuation_symmetry.py": 668,
    "pipelines/hardcoded/hh_continuation_types.py": 654,
}

CRITICAL_SOURCE_PATHS = (
    "pipelines/static_adapt/adapt_pipeline.py",
    "pipelines/static_adapt/cli_config.py",
    "pipelines/static_adapt/output_artifacts.py",
    "pipelines/static_adapt/resume_scaffold.py",
    "pipelines/static_adapt/sr_snake_phase12_policy.py",
    "pipelines/static_adapt/sr_snake_route_profile.py",
    "pipelines/static_adapt/adapt_candidate_record_cache.py",
    "pipelines/scaffold/hh_continuation_scoring.py",
    "pipelines/scaffold/hh_continuation_types.py",
    "pipelines/scaffold/hh_continuation_pruning.py",
    "pipelines/exact_bench/paper_i_hh_recovery_prefix_qiskit_sidecar.py",
    "pipelines/exact_bench/table_i_qiskit_resource_compile.py",
    "pipelines/hardcoded/adapt_circuit_execution.py",
    "pipelines/qiskit_backend_tools.py",
    "test/test_hh_continuation_scoring.py",
    "test/test_static_adapt_sr_route_profile.py",
    "test/test_static_adapt_historical_singleton_overlays.py",
    "test/test_static_adapt_sr_v4_runtime.py",
    "test/test_static_adapt_sr_v4_serialization.py",
    "test/test_static_adapt_sr_trust_prune.py",
    "agent_guidance/static-adapt/route-identities.md",
    "agent_guidance/shared/run-guide.md",
    "MATH/paper_facing/paper_I_static_scaffold/"
    "paper_i_sr_snake_v4_candidate_runtime_settings_20260716.md",
)

REGIMES: tuple[dict[str, Any], ...] = (
    {
        "slug": "weak_weak", "u": "0.25", "lambda": 0.25,
        "g_ep": "0.353553390593", "n_ph": 3,
        "exact_energy": -0.918380919994822,
        "manifest_sha256": "24b8c50be54acc6eda506b38e9cd0583bd0ef88b1db6dac47e850b88040dc0b0",
        "result_sha256": "68fde0ab9de5ae69cee27ac0f54cb52f9e377882969daa0a1630d14f520ffdaa",
        "memory_mb": 32768, "disk_mb": 61440,
    },
    {
        "slug": "intermediate_weak", "u": "1.25", "lambda": 0.25,
        "g_ep": "0.353553390593", "n_ph": 3,
        "exact_energy": -0.4950053491813613,
        "manifest_sha256": "7ee0d9b6aea0f12418426a232e42a6962e2ef7cfa47f6dc3ba71061f139b1573",
        "result_sha256": "9e2479fd8308f111cd311e7843ccf1978962dc9d66876bc78f5c69566054a2ed",
        "memory_mb": 32768, "disk_mb": 61440,
    },
    {
        "slug": "strong_weak_u8", "u": "8.0", "lambda": 0.25,
        "g_ep": "0.353553390593", "n_ph": 3,
        "exact_energy": 0.5264586847939736,
        "manifest_sha256": "67ff7a01a5cc1a33b34982e0a3511d889e4cc7aa7f93b7ea042d80bcf3ce5c0e",
        "result_sha256": "b62f89ef9271a2ff42eab2057e9183f48900355967adcabc1fbc9491d22a21f6",
        "memory_mb": 40960, "disk_mb": 61440,
    },
    {
        "slug": "weak_strong", "u": "0.25", "lambda": 1.25,
        "g_ep": "0.790569415042", "n_ph": 7,
        "exact_energy": -1.1387206380749124,
        "manifest_sha256": "6862cab52ebc8b49e15cdeada67c873b066918e5260b0e578244b36e52549c56",
        "result_sha256": "aaf2102a7829ac7a2b4c0f13ef55ef96fe81ed3b70f7d110e2d7c001b6d9cf3e",
        "memory_mb": 49152, "disk_mb": 81920,
    },
    {
        "slug": "intermediate_strong", "u": "1.25", "lambda": 1.25,
        "g_ep": "0.790569415042", "n_ph": 7,
        "exact_energy": -0.6239396137518493,
        "manifest_sha256": "ec5d436919af666fdfe1c28e8f243d44163637348451f46a368a73fb4eefd021",
        "result_sha256": "d00c8ab411fd87429f63095e5ab7cbea2c3b6d535228fbbaf3b5f60bf22499b0",
        "memory_mb": 49152, "disk_mb": 81920,
    },
    {
        "slug": "strong_strong_u8", "u": "8.0", "lambda": 1.25,
        "g_ep": "0.790569415042", "n_ph": 7,
        "exact_energy": 0.5205762765682517,
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
    "test/test_static_adapt_sr_route_profile.py",
    "test/test_static_adapt_historical_singleton_overlays.py",
    "test/test_static_adapt_resume_scaffold.py",
    "test/test_adapt_candidate_record_cache.py",
    "test/test_hh_continuation_scoring.py",
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


def recursive_diff(left: Any, right: Any, prefix: str = "") -> list[dict[str, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        rows: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(
                recursive_diff(
                    left.get(key, "<MISSING>"),
                    right.get(key, "<MISSING>"),
                    path,
                )
            )
        return rows
    if left == right:
        return []
    return [{"path": prefix, "v2": left, "v3": right}]


def normalize_bundle_strings(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalize_bundle_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_bundle_strings(item) for item in value]
    if isinstance(value, str):
        return (
            value.replace(V2_BUNDLE_ID, "<BUNDLE_ID>")
            .replace(BUNDLE_ID, "<BUNDLE_ID>")
            .replace("20260716-v2", "<REVISION>")
            .replace("20260716-v6", "<REVISION>")
        )
    return value


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def git_blob(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{EXPECTED_HEAD}:{path}"], cwd=REPO)


def verify_overlay_sources() -> dict[str, str]:
    """Verify the two explicit non-scientific compatibility overlays."""

    verified: dict[str, str] = {}
    for relative, expected in NONSCIENTIFIC_ARCHIVE_OVERLAYS.items():
        path = REPO / relative
        if not path.is_file():
            raise ValueError(f"required compatibility overlay is missing: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(
                f"compatibility overlay hash drift: {relative}: "
                f"expected {expected}, got {actual}"
            )
        expected_size = NONSCIENTIFIC_ARCHIVE_OVERLAY_SIZES[relative]
        if path.stat().st_size != expected_size:
            raise ValueError(
                f"compatibility overlay size drift: {relative}: "
                f"expected {expected_size}, got {path.stat().st_size}"
            )
        tracked = subprocess.run(
            ["git", "cat-file", "-e", f"{EXPECTED_HEAD}:{relative}"],
            cwd=REPO,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if tracked.returncode == 0:
            raise ValueError(
                f"compatibility overlay unexpectedly exists in frozen commit: {relative}"
            )
        verified[relative] = actual
    return verified


def exact_source_tar_bytes() -> bytes:
    return subprocess.check_output(
        ["git", "archive", "--format=tar", EXPECTED_HEAD, "--", *ARCHIVE_PATHS],
        cwd=REPO,
    )


def extract_exact_source(destination: Path) -> None:
    """Extract the exact commit and only the two hash-locked overlays."""

    raw = exact_source_tar_bytes()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as handle:
        for member in handle.getmembers():
            name = PurePosixPath(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or member.issym()
                or member.islnk()
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"unsafe exact-source member: {member.name}")
        handle.extractall(destination, filter="data")
    verified = verify_overlay_sources()
    for relative in verified:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, target)


def exact_source_contract_probe() -> dict[str, Any]:
    """Resolve the v4 profile from isolated frozen source, never the live tree."""

    with tempfile.TemporaryDirectory(prefix="sr_v4_exact_source_probe_") as tmp:
        root = Path(tmp)
        extract_exact_source(root)
        code = (
            "import json; "
            "from pipelines.static_adapt.sr_snake_route_profile import "
            "canonical_sr_snake_contract,canonical_sr_snake_contract_sha256,"
            "normalize_sr_route_profile_request; "
            "print(json.dumps({'resolved':normalize_sr_route_profile_request('sr_snake_v4'),"
            "'digest':canonical_sr_snake_contract_sha256('sr_snake_v4'),"
            "'contract':canonical_sr_snake_contract('sr_snake_v4')},"
            "sort_keys=True,allow_nan=False))"
        )
        env = os.environ.copy()
        env.update({
            "HOME": str(root / "home"),
            "PYTHONPATH": os.pathsep.join((
                str(root),
                str(site.getusersitepackages()),
            )),
            "PYTHONDONTWRITEBYTECODE": "1",
        })
        (root / "home").mkdir()
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise ValueError(
                "isolated frozen-source route-profile probe failed: "
                + completed.stderr.strip()
            )
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict):
            raise TypeError("isolated route-profile probe returned a non-object")
        return payload


def source_result_path(slug: str) -> Path:
    if slug == "weak_weak":
        return WEAK_WEAK_RESULT
    return HISTORICAL_RESULT_ROOT / slug / "json/result.json"


def verify_source_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    if len(EXPECTED_HEAD) != 40 or len(EXPECTED_TREE) != 40:
        raise ValueError(
            "v3 source authority is not frozen: replace EXPECTED_HEAD and "
            "EXPECTED_TREE only after the corrected commit is pushed"
        )
    head = git_output("rev-parse", f"{EXPECTED_HEAD}^{{commit}}")
    tree = git_output("rev-parse", f"{EXPECTED_HEAD}^{{tree}}")
    if head != EXPECTED_HEAD or tree != EXPECTED_TREE:
        raise ValueError(
            f"source revision drift: expected {EXPECTED_HEAD}/{EXPECTED_TREE}, "
            f"got {head}/{tree}"
        )
    overlays = verify_overlay_sources()
    critical = {}
    for path in CRITICAL_SOURCE_PATHS:
        actual = bytes_sha256(git_blob(path))
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
        baseline_n_ph = int(physics["n_ph_work"])
        baseline_exact = float(exact)
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
            "exact_reference_key_hash": {
                "weak_weak": "a10820b35d82ea3bd29599b5",
                "intermediate_weak": "8c5f49d0f545a12f898be7ba",
                "strong_weak_u8": "2218571998ef766037aa4d0f",
                "weak_strong": "42872c0f1988ea8bdbd99b79",
                "intermediate_strong": "99397703afad40a7bd87403c",
                "strong_strong_u8": "b941d7eae8f318acfc831c86",
            }[slug],
            "exact_reference_source": (
                "pipelines.exact_bench.static_reference_metrics.exact_energy_for_spec"
            ),
            "baseline_n_ph_work": baseline_n_ph,
            "baseline_exact_energy": baseline_exact,
            "historical_manifest": manifest_path.as_posix(),
            "historical_manifest_sha256": str(row["manifest_sha256"]),
            "historical_result": result_path.as_posix(),
            "historical_result_sha256": str(row["result_sha256"]),
        })

    probe = exact_source_contract_probe()
    resolved = str(probe.get("resolved") or "")
    contract = probe.get("contract")
    contract_digest = str(probe.get("digest") or "")
    if not isinstance(contract, dict):
        raise TypeError("isolated frozen-source route profile returned no contract")
    if resolved != PROFILE_RESOLVED or contract_digest != PROFILE_CONTRACT_SHA256:
        raise ValueError("v4 profile resolution/digest drift")
    if contract["execution_settings"]["adapt_max_depth"] != 30:
        raise ValueError("v4 adapt_max_depth must remain 30")
    execution_settings = contract["execution_settings"]
    semantic_invariants = contract["semantic_invariants"]
    if execution_settings.get("adapt_finite_angle_fallback") is not False:
        raise ValueError("v4 finite-angle fallback must be disabled")
    if semantic_invariants.get("finite_angle_fallback_active") is not False:
        raise ValueError("v4 finite-angle semantic invariant must be false")
    if execution_settings.get("phase3_enable_rescue") is not False:
        raise ValueError("v4 Phase-III rescue must be disabled")
    expected_phase12 = {
        "phase1_energy_model": PHASE1_ENERGY_MODEL,
        "phase2_curvature_policy": PHASE2_CURVATURE_POLICY,
        "phase2_cheap_curvature_proxy_policy": (
            PHASE2_CHEAP_CURVATURE_PROXY_POLICY
        ),
    }
    for key, expected in expected_phase12.items():
        if execution_settings.get(key) != expected:
            raise ValueError(f"v4 {key} drift: {execution_settings.get(key)!r}")
        if semantic_invariants.get(key) != expected:
            raise ValueError(f"v4 semantic invariant {key} drift")
    if semantic_invariants.get("phase1_phase2_lambda_f_proxy_active") is not False:
        raise ValueError("v4 lambda-F proxy semantic invariant must be false")
    if semantic_invariants.get("phase2_curvature_failure_policy") != "abort_run_v1":
        raise ValueError("v4 Phase-II curvature failure policy must abort the run")

    revision = {
        "schema": "paper_i_hh_sr_snake_v4_source_revision_v3",
        "git_commit": head,
        "git_tree": tree,
        "profile_request": PROFILE_REQUEST,
        "profile_resolved": resolved,
        "profile_contract_sha256": contract_digest,
        "phase1_energy_model": PHASE1_ENERGY_MODEL,
        "phase2_curvature_policy": PHASE2_CURVATURE_POLICY,
        "phase2_cheap_curvature_proxy_policy": (
            PHASE2_CHEAP_CURVATURE_PROXY_POLICY
        ),
        "critical_source_sha256": critical,
        "non_scientific_archive_overlays": {
            relative: {
                "sha256": digest,
                "size_bytes": NONSCIENTIFIC_ARCHIVE_OVERLAY_SIZES[relative],
                "mode": "0644",
                "classification": "compatibility_import_shim_only",
                "tracked_in_frozen_commit": False,
            }
            for relative, digest in overlays.items()
        },
    }
    physics_lock = {
        "schema": "paper_i_hh_sr_snake_v4_physics_exact_reference_lock_v3",
        "same_cutoff_required": True,
        "manual_exact_energy_override_forbidden": True,
        "runtime_exact_energy_recomputed": True,
        "runtime_exact_energy_tolerance": 1.0e-12,
        "rows": physics_rows,
    }
    return revision, {"contract": contract, "physics_lock": physics_lock}


def build_source_archive() -> dict[str, Any]:
    overlays = verify_overlay_sources()
    with tempfile.NamedTemporaryFile(prefix="sr_v4_source_", suffix=".tar") as raw:
        raw.write(exact_source_tar_bytes())
        raw.flush()
        with tarfile.open(raw.name, mode="a") as handle:
            existing = {PurePosixPath(member.name).as_posix() for member in handle}
            for relative, expected in sorted(overlays.items()):
                if relative in existing:
                    raise ValueError(
                        f"compatibility overlay collides with frozen source: {relative}"
                    )
                data = (REPO / relative).read_bytes()
                if bytes_sha256(data) != expected:
                    raise ValueError(f"compatibility overlay changed while packaging: {relative}")
                info = tarfile.TarInfo(relative)
                info.size = len(data)
                info.mode = 0o644
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                handle.addfile(info, io.BytesIO(data))
        raw.seek(0)
        compressed = gzip.compress(raw.read(), compresslevel=9, mtime=0)
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
    for path in CRITICAL_SOURCE_PATHS:
        expected = bytes_sha256(git_blob(path))
        if members.get(path, {}).get("sha256") != expected:
            raise ValueError(f"critical file missing/drifted in archive: {path}")
    for path, expected in overlays.items():
        if members.get(path, {}).get("sha256") != expected:
            raise ValueError(f"compatibility overlay missing/drifted in archive: {path}")
    return {
        "schema": "paper_i_hh_sr_snake_v4_source_archive_manifest_v3",
        "archive": ARCHIVE_PATH.relative_to(REPO).as_posix(),
        "archive_sha256": sha256(ARCHIVE_PATH),
        "archive_size_bytes": ARCHIVE_PATH.stat().st_size,
        "git_commit": EXPECTED_HEAD,
        "git_tree": EXPECTED_TREE,
        "worker_source_mode": (
            "exact_git_archive_plus_hashed_nonscientific_overlays_v1"
        ),
        "worker_pythonpath": "/work",
        "non_scientific_archive_overlays": {
            relative: {
                "sha256": digest,
                "size_bytes": NONSCIENTIFIC_ARCHIVE_OVERLAY_SIZES[relative],
                "mode": "0644",
                "classification": "compatibility_import_shim_only",
                "tracked_in_frozen_commit": False,
            }
            for relative, digest in overlays.items()
        },
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
            "PYTHONPATH": os.pathsep.join((
                str(root),
                str(site.getusersitepackages()),
            )),
            "PYTHONDONTWRITEBYTECODE": "1",
        })
        (root / "home").mkdir()
        dependency_env = env.copy()
        live_repo = str(REPO.resolve())
        probe_code = f"""
import hashlib
import json
import pathlib
import sys
import pipelines.static_adapt.sr_snake_route_profile as route
import pipelines.static_adapt.adapt_pipeline as adapt_target
import pipelines.hardcoded.adapt_pipeline as adapt_alias
import pipelines.scaffold.hh_continuation_scoring as scoring_target
import pipelines.hardcoded.hh_continuation_scoring as scoring_alias

root = pathlib.Path.cwd().resolve()
live_repo = pathlib.Path({live_repo!r}).resolve()
route_path = pathlib.Path(route.__file__).resolve()
outside = []
for name, module in sorted(sys.modules.items()):
    if not (name == "pipelines" or name.startswith("pipelines.") or name == "src" or name.startswith("src.")):
        continue
    value = getattr(module, "__file__", None)
    if not value:
        continue
    path = pathlib.Path(value).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        outside.append(name)
live_sys_path = []
for value in sys.path:
    if not value:
        continue
    path = pathlib.Path(value).resolve()
    if path == live_repo or live_repo in path.parents:
        live_sys_path.append(value)
overlay_hashes = {{}}
for relative in {tuple(sorted(NONSCIENTIFIC_ARCHIVE_OVERLAYS))!r}:
    path = root / relative
    overlay_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
print(json.dumps({{
    "module": route_path.relative_to(root).as_posix(),
    "sha256": hashlib.sha256(route_path.read_bytes()).hexdigest(),
    "resolved": route.normalize_sr_route_profile_request("sr_snake_v4"),
    "digest": route.canonical_sr_snake_contract_sha256("sr_snake_v4"),
    "adapt_alias_is_target": adapt_alias is adapt_target,
    "scoring_alias_is_target": scoring_alias is scoring_target,
    "overlay_hashes": overlay_hashes,
    "project_modules_outside_archive": outside,
    "live_repo_sys_path_entries": len(live_sys_path),
}}, sort_keys=True))
"""
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
            and probe_payload.get("adapt_alias_is_target") is True
            and probe_payload.get("scoring_alias_is_target") is True
            and probe_payload.get("overlay_hashes")
            == NONSCIENTIFIC_ARCHIVE_OVERLAYS
            and probe_payload.get("project_modules_outside_archive") == []
            and probe_payload.get("live_repo_sys_path_entries") == 0
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
            "test/test_static_adapt_sr_route_profile.py",
            "test/test_static_adapt_historical_singleton_overlays.py",
            "test/test_static_adapt_resume_scaffold.py",
            "test/test_adapt_candidate_record_cache.py",
            "test/test_hh_continuation_scoring.py",
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
        focused_regressions_pass = focused_tests.returncode == 0
        all_parse = all(row["returncode"] == 0 for row in parse_rows)
        return {
            "schema": "paper_i_hh_sr_snake_v4_archive_only_preflight_v3",
            "status": (
                "pass" if import_pass and all_parse and helper_present
                and helper_help_returncode == 0
                and focused_regressions_pass else "blocked"
            ),
            "archive_sha256": archive["archive_sha256"],
            "source_import": {
                "status": "pass" if import_pass else "fail",
                "module": probe_payload.get("module"),
                "sha256": probe_payload.get("sha256"),
                "profile_resolved": probe_payload.get("resolved"),
                "profile_contract_sha256": probe_payload.get("digest"),
                "adapt_compatibility_alias_is_archived_target": (
                    probe_payload.get("adapt_alias_is_target")
                ),
                "scoring_compatibility_alias_is_archived_target": (
                    probe_payload.get("scoring_alias_is_target")
                ),
                "overlay_hashes": probe_payload.get("overlay_hashes"),
                "project_modules_outside_archive": probe_payload.get(
                    "project_modules_outside_archive"
                ),
                "live_repo_sys_path_entries": probe_payload.get(
                    "live_repo_sys_path_entries"
                ),
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
                "returncode": int(focused_tests.returncode),
                "pass": focused_regressions_pass,
                "pytest_stdout_tail": focused_tests.stdout.strip().splitlines()[-1:],
                "stderr_empty": not bool(focused_tests.stderr.strip()),
                "dependency_environment": (
                    "local_python_packages_with_extracted_archive_source_only"
                ),
            },
            "live_repo_import_excluded": bool(
                probe_payload.get("project_modules_outside_archive") == []
                and probe_payload.get("live_repo_sys_path_entries") == 0
            ),
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
        "--adapt-disable-hh-seed",
        "--adapt-segment-id", f"{slug}-sr-v4-r0-r30-20260716-v6",
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
        "schema": "paper_i_hh_sr_snake_v4_candidate_parent_job_v3",
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
            "phase12_energy_model_contract": {
                "phase1_energy_model": PHASE1_ENERGY_MODEL,
                "phase2_curvature_policy": PHASE2_CURVATURE_POLICY,
                "phase2_cheap_curvature_proxy_policy": (
                    PHASE2_CHEAP_CURVATURE_PROXY_POLICY
                ),
                "lambda_f_proxy_flags_forbidden": True,
                "missing_curvature_failure_policy": "abort_run_v1",
            },
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
            "worker_source_mode": archive["worker_source_mode"],
            "non_scientific_archive_overlays": archive[
                "non_scientific_archive_overlays"
            ],
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


def validate_v2_to_v3_settings_diff(
    *,
    contract: dict[str, Any],
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_contract = load_json(V2_BUNDLE_DIR / "jobs/weak_weak.json")[
        "route_identity"
    ]["profile_contract"]
    contract_diff = recursive_diff(baseline_contract, contract)
    observed_contract_paths = {str(row["path"]) for row in contract_diff}
    if observed_contract_paths != ALLOWED_V2_TO_V3_CONTRACT_DIFF_PATHS:
        raise ValueError(
            "unexpected v2->v3 route-contract drift: "
            f"observed={sorted(observed_contract_paths)}"
        )

    row_audits: list[dict[str, Any]] = []
    for job in jobs:
        slug = str(job["regime_slug"])
        baseline = load_json(V2_BUNDLE_DIR / "jobs" / f"{slug}.json")
        physics_diff = recursive_diff(baseline["physics"], job["physics"])
        physics_diff_paths = {str(row["path"]) for row in physics_diff}
        approved_physics_paths = {
            "expected_exact_energy", "n_ph_reference", "n_ph_work"
        }

        def normalized_cutoff_argv(payload: Any) -> Any:
            normalized = normalize_bundle_strings(payload)
            if not isinstance(normalized, list):
                return normalized
            tokens = list(normalized)
            if "--n-ph-max" in tokens:
                index = tokens.index("--n-ph-max")
                tokens[index + 1] = "<N_PH_MAX>"
            if "--adapt-disable-hh-seed" in tokens:
                tokens.remove("--adapt-disable-hh-seed")
            return tokens

        checks = {
            "physics_diff_exactly_approved": (
                physics_diff_paths == approved_physics_paths
            ),
            "segment_identical": baseline["segment"] == job["segment"],
            "resource_request_identical": (
                baseline["resource_request"] == job["resource_request"]
            ),
            "command_identical_after_revision_path_normalization": (
                {
                    **normalize_bundle_strings(baseline["command"]),
                    "argv": normalized_cutoff_argv(
                        baseline["command"]["argv"]
                    ),
                }
                == {
                    **normalize_bundle_strings(job["command"]),
                    "argv": normalized_cutoff_argv(job["command"]["argv"]),
                }
            ),
            "environment_identical_after_bundle_path_normalization": (
                normalize_bundle_strings(baseline["environment"])
                == normalize_bundle_strings(job["environment"])
            ),
        }
        if not all(checks.values()):
            raise ValueError(f"non-approved v2->v3 job drift for {slug}: {checks}")
        row_audits.append({
            "regime_slug": slug,
            "checks": checks,
            "approved_physics_diff": physics_diff,
        })
    return {
        "schema": "paper_i_hh_sr_snake_v4_v2_to_v3_scientific_settings_diff_v1",
        "baseline_bundle": V2_BUNDLE_DIR.relative_to(REPO).as_posix(),
        "candidate_bundle": BUNDLE_DIR.relative_to(REPO).as_posix(),
        "status": "pass",
        "approved_contract_diff_paths": sorted(
            ALLOWED_V2_TO_V3_CONTRACT_DIFF_PATHS
        ),
        "observed_contract_diff": contract_diff,
        "regime_checks": row_audits,
        "unexpected_executable_differences": [],
    }
    return {
        "schema": "paper_i_hh_sr_snake_v4_candidate_parent_job_v3",
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
            "phase12_energy_model_contract": {
                "phase1_energy_model": PHASE1_ENERGY_MODEL,
                "phase2_curvature_policy": PHASE2_CURVATURE_POLICY,
                "phase2_cheap_curvature_proxy_policy": (
                    PHASE2_CHEAP_CURVATURE_PROXY_POLICY
                ),
                "lambda_f_proxy_flags_forbidden": True,
                "missing_curvature_failure_policy": "abort_run_v1",
            },
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
            "worker_source_mode": archive["worker_source_mode"],
            "non_scientific_archive_overlays": archive[
                "non_scientific_archive_overlays"
            ],
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
# Generated from the frozen v3 source lock.  Matchmaking remains disabled until
# SUBMISSION_ENABLED is explicitly set after every local/archive/remote gate.
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
    smoke_records = (
        "json/result.json",
        "json/current.json",
        "json/estimator_call_ledger.json",
    )
    evidence_hashes = {
        relative: sha256(REPO / SMOKE8_ROOT / relative)
        for relative in smoke_records
    }

    result = load_json(REPO / SMOKE8_ROOT / "json/result.json")
    current = load_json(REPO / SMOKE8_ROOT / "json/current.json")
    ledger = load_json(REPO / SMOKE8_ROOT / "json/estimator_call_ledger.json")
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
    telemetry = result.get("adapt_vqe", {}).get(
        "phase12_energy_model_telemetry", {}
    )
    full_candidates = int(telemetry.get("phase2_full_candidate_occurrences", -1))
    validated_receipts = int(
        telemetry.get("validated_phase2_curvature_receipt_occurrences", -2)
    )
    expected_telemetry = {
        "phase1_energy_model": PHASE1_ENERGY_MODEL,
        "phase2_curvature_policy": PHASE2_CURVATURE_POLICY,
        "phase2_cheap_curvature_proxy_policy": (
            PHASE2_CHEAP_CURVATURE_PROXY_POLICY
        ),
        "phase1_lambda_f_proxy_occurrences": 0,
        "phase2_lambda_f_proxy_occurrences": 0,
        "phase2_missing_curvature_fallback_occurrences": 0,
    }
    for key, expected in expected_telemetry.items():
        if telemetry.get(key) != expected:
            raise ValueError(f"depth-8 Phase-I/II smoke telemetry drift: {key}")
    if full_candidates <= 0 or validated_receipts != full_candidates:
        raise ValueError("depth-8 Phase-II curvature-receipt accounting is open")
    if full_candidates != 354:
        raise ValueError(
            f"authoritative depth-8 smoke receipt count drift: {full_candidates}"
        )
    s_alg = int(
        result.get("adapt_vqe", {})
        .get("estimator_call_accounting", {})
        .get("all_branch_search_work", {})
        .get("S_alg", -1)
    )
    if s_alg != 6555:
        raise ValueError(f"authoritative depth-8 smoke S_alg drift: {s_alg}")

    return {
        "schema": "paper_i_hh_sr_snake_v4_local_smoke_evidence_v3",
        "status": "pass_for_bundle_construction_not_a_production_result",
        "records": [{
            "label": "eight_admission_cache_off_no_lambda_f_proxy_v4",
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
            "phase12_energy_model_telemetry": telemetry,
            "S_alg_all_branch_search_work": s_alg,
        }],
        "production_composition_prune_gate": {
            "kind": "focused_source_locked_regression",
            "test_file": "test/test_static_adapt_sr_trust_prune.py",
            "test_file_sha256": bytes_sha256(
                git_blob("test/test_static_adapt_sr_trust_prune.py")
            ),
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
            "phase1_first_order_fs_trust_only",
            "phase2_measured_curvature_required",
            "phase2_curvature_receipt_count_closure",
            "zero_phase1_phase2_lambda_f_proxy_occurrences",
            "zero_missing_curvature_fallback_occurrences",
            "finite_angle_fallback_disabled_no_guard_probes",
            "phase3_rescue_disabled",
            "phase3_oracle_gradient_mode_off",
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
    gate_path = BUNDLE_DIR / "remote_execution_gate.json"
    gate = load_json(gate_path)
    if gate.get("schema") != "paper_i_hh_sr_snake_v4_remote_execution_gate_v3":
        raise ValueError("unexpected remote execution gate schema")
    remote = gate.get("remote_execution_preflight", {})
    remote_pass = bool(
        gate.get("status") == "pass"
        and remote.get("image_path") == REMOTE_IMAGE_PATH.as_posix()
        and remote.get("image_sha256") == REMOTE_IMAGE_SHA256
        and remote.get("qiskit_import_passed") is True
        and remote.get("qiskit_version") == REMOTE_QISKIT_VERSION
        and remote.get("fake_backend_instantiation_passed") is True
        and remote.get("fake_backend_resolved") == REMOTE_FAKE_BACKEND_RESOLVED
        and int(remote.get("fake_backend_qubits", -1))
        == REMOTE_FAKE_BACKEND_QUBITS
    )
    if SUBMISSION_ENABLED and not remote_pass:
        raise ValueError(
            "submission cannot be enabled before remote_execution_gate.json passes"
        )
    return {
        "schema": "paper_i_hh_sr_snake_v4_remote_preflight_cleanup_receipt_v3",
        "status": "pass" if remote_pass else "blocked_pending_remote_preflight",
        "remote_execution_preflight": remote,
        "remote_execution_gate": gate_path.relative_to(REPO).as_posix(),
        "remote_execution_gate_sha256": sha256(gate_path),
        "storage_cleanup": {
            "scope": "no_cleanup_authorized_or_required_by_bundle_builder",
            "remote_removed_paths": [],
            "unrelated_remote_paths_modified": False,
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
    job_payloads = []
    for row in REGIMES:
        job = build_job(row, contract, archive)
        job_payloads.append(job)
        slug = str(row["slug"])
        job_path = BUNDLE_DIR / "jobs" / f"{slug}.json"
        normalized_path = BUNDLE_DIR / "normalized_manifests" / f"{slug}.json"
        dump_json(job_path, job)
        dump_json(normalized_path, {
            "schema": "paper_i_hh_sr_snake_v4_normalized_parent_manifest_v3",
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
    settings_diff = validate_v2_to_v3_settings_diff(
        contract=contract,
        jobs=job_payloads,
    )
    dump_json(BUNDLE_DIR / "v2_to_v3_scientific_settings_diff.json", settings_diff)
    (BUNDLE_DIR / "queue.tsv").write_text("\n".join(queue_lines) + "\n")
    (BUNDLE_DIR / "submit.sub").write_text(
        submit_text(str(archive["archive_sha256"])), encoding="utf-8"
    )

    isolated = archive_only_preflight(archive=archive, job_paths=job_paths)
    dump_json(BUNDLE_DIR / "archive_only_preflight.json", isolated)
    if isolated.get("status") != "pass":
        raise ValueError("archive-only import/parse/helper/regression preflight failed")
    parse_rows = list(isolated["six_validate_only_parses"])

    continuation_template = {
        "schema": "paper_i_hh_sr_snake_v4_round30_to_round50_continuation_template_v3",
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
        "schema": "paper_i_hh_sr_snake_v4_route_parity_v3",
        "status": "pass",
        "profile_request": PROFILE_REQUEST,
        "profile_resolved": PROFILE_RESOLVED,
        "profile_contract_sha256": PROFILE_CONTRACT_SHA256,
        "all_six_commands_parse": all(row["returncode"] == 0 for row in parse_rows),
        "all_targets_round_30": True,
        "all_max_new_admissions_30": True,
        "no_profile_method_flag_repetition": True,
        "same_cutoff_lock_pass": True,
        "phase1_energy_model": PHASE1_ENERGY_MODEL,
        "phase2_curvature_policy": PHASE2_CURVATURE_POLICY,
        "phase2_cheap_curvature_proxy_policy": (
            PHASE2_CHEAP_CURVATURE_PROXY_POLICY
        ),
        "lambda_f_proxy_flags_forbidden": True,
        "worker_source_mode": archive["worker_source_mode"],
        "non_scientific_archive_overlays": archive[
            "non_scientific_archive_overlays"
        ],
        "parse_rows": parse_rows,
    }
    dump_json(BUNDLE_DIR / "route_parity.json", parity)
    bundle_manifest = {
        "schema": "paper_i_hh_sr_snake_v4_candidate_bundle_v3",
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
        "v2_to_v3_scientific_settings_diff": settings_diff,
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
        "submission_status": (
            "submission_ready_not_yet_submitted"
            if SUBMISSION_ENABLED
            else "submission_blocked_pending_main_agent_all_gate_confirmation"
        ),
    }
    dump_json(BUNDLE_DIR / "bundle_manifest.json", bundle_manifest)
    remote_pass = remote_receipt.get("status") == "pass"
    submission_ready = bool(SUBMISSION_ENABLED and remote_pass)
    submission_blockers = [] if submission_ready else [
        "submission_gate_disabled_in_builder"
        if not SUBMISSION_ENABLED
        else "remote_execution_preflight_not_passed"
    ]
    preflight = {
        "schema": "paper_i_hh_sr_snake_v4_candidate_preflight_v3",
        "created_utc": utc_now(),
        "status": (
            "pass_submission_ready_not_yet_submitted"
            if submission_ready
            else "pass_bundle_built_submission_blocked"
        ),
        "checks": {
            "exact_git_revision": True,
            "critical_source_hashes": True,
            "source_archive_safe_and_closed": True,
            "exact_commit_plus_hashed_nonscientific_overlays": (
                archive["worker_source_mode"]
                == "exact_git_archive_plus_hashed_nonscientific_overlays_v1"
                and set(archive["non_scientific_archive_overlays"])
                == set(NONSCIENTIFIC_ARCHIVE_OVERLAYS)
            ),
            "compatibility_overlay_archive_hashes_closed": all(
                archive["files"].get(relative, {}).get("sha256") == digest
                for relative, digest in NONSCIENTIFIC_ARCHIVE_OVERLAYS.items()
            ),
            "six_job_manifests": len(job_paths) == 6,
            "all_job_validations": all(row["returncode"] == 0 for row in parse_rows),
            "same_cutoff_reference_lock": True,
            "v4_profile_and_digest": True,
            "fresh_round0_to_round30_only": True,
            "no_adapt_max_depth_override": True,
            "no_repeated_method_flags": True,
            "v2_to_v3_settings_diff_exactly_approved": (
                settings_diff["status"] == "pass"
                and not settings_diff["unexpected_executable_differences"]
            ),
            "worker_pythonpath_archive_only": True,
            "isolated_archive_source_import": (
                isolated["source_import"]["status"] == "pass"
            ),
            "isolated_archive_compatibility_aliases": bool(
                isolated["source_import"][
                    "adapt_compatibility_alias_is_archived_target"
                ]
                and isolated["source_import"][
                    "scoring_compatibility_alias_is_archived_target"
                ]
            ),
            "isolated_archive_live_repo_import_excluded": bool(
                isolated["live_repo_import_excluded"]
            ),
            "isolated_archive_all_six_validate_only": bool(
                isolated["all_six_validate_only_pass"]
            ),
            "isolated_archive_qiskit_helper_help": bool(
                isolated["qiskit_helper"]["help_pass"]
            ),
            "isolated_archive_focused_regressions": bool(
                isolated["focused_source_locked_regressions"]["pass"]
            ),
            "future_round30_to_round50_template_only": True,
            "submission_enabled": SUBMISSION_ENABLED,
            "terminal_qiskit_sidecar_required": True,
            "qiskit_sidecar_helper_in_source_archive": qiskit_helper_archived,
            "qiskit_backend_availability_remote_check": True,
            "remote_image_sha256_rechecked": remote_pass,
            "remote_qiskit_import_passed": remote_pass,
            "remote_fake_marrakesh_instantiation_passed": remote_pass,
            "local_image_present": image_local_present,
            "local_image_hash_matches_prior_remote_digest": image_local_match,
            "phase3_response_supported_rank_recorded": True,
            "shadow_damping_scientific_application_expected": False,
            "shadow_damping_diagnostic_noop_receipt_recorded": True,
            "production_composition_delete_refit_prune_regression_passed": True,
            "finite_angle_fallback_disabled": True,
            "phase3_rescue_disabled": True,
            "phase3_oracle_gradient_mode_off": True,
            "phase1_first_order_fs_trust_policy": True,
            "phase2_measured_curvature_required_fail_closed_policy": True,
            "phase2_cheap_curvature_proxy_off": True,
            "phase1_phase2_lambda_f_proxy_inactive": True,
            "smoke_phase2_curvature_receipt_count_closure": True,
            "smoke_lambda_f_proxy_occurrences_zero": True,
            "smoke_missing_curvature_fallback_occurrences_zero": True,
        },
        "remote_image": {
            "path": REMOTE_IMAGE_PATH.as_posix(),
            "verified_remote_sha256": REMOTE_IMAGE_SHA256,
            "qiskit_version": REMOTE_QISKIT_VERSION,
            "fake_backend_resolved": REMOTE_FAKE_BACKEND_RESOLVED,
            "fake_backend_qubits": REMOTE_FAKE_BACKEND_QUBITS,
            "local_copy_present": image_local_present,
            "remote_recheck_passed": remote_pass,
        },
        "scientific_blockers": [],
        "submission_blockers": submission_blockers,
        "submission_authorized": submission_ready,
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
            "v2_to_v3_scientific_settings_diff.json",
            "archive_only_preflight.json",
            "future_round30_to_round50_continuation_template.json",
            "remote_execution_gate.json",
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
            and ".pytest_cache" not in path.parts
            and path.suffix != ".pyc"
        ):
            inventory[path.relative_to(REPO).as_posix()] = {
                "sha256": sha256(path), "size_bytes": path.stat().st_size
            }
    dump_json(BUNDLE_DIR / "submission_artifact_hashes.json", {
        "schema": "paper_i_hh_sr_snake_v4_submission_artifact_hashes_v3",
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
