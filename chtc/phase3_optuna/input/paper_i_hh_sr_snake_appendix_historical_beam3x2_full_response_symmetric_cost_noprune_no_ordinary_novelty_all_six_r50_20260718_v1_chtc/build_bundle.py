#!/usr/bin/env python3
"""Build the source-locked six-regime exact historical-beam appendix bundle.

This builder performs no scientific calculation and never submits CHTC jobs.
It derives byte-for-byte from the frozen main SR archive, applies one
hash-locked conditional legacy-beam overlay, resolves the explicit appendix
3x2 beam profile from that isolated archive, verifies all six same-cutoff
physics anchors, and writes six fresh round-0 to round-50 rows plus a
fail-closed submission preflight.  The main/no-beam archive is never modified.
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


BUNDLE_ID = (
    "paper_i_hh_sr_snake_appendix_historical_beam3x2_full_response_"
    "symmetric_cost_noprune_no_ordinary_novelty_all_six_r50_20260718_v1_chtc"
)
BATCH_NAME = (
    "paper-i-hh-sr-appendix-historical-beam3x2-fullresp-symcost-"
    "noprune-nonovelty-six-r50-20260718-v1"
)
BUNDLE_DIR = Path(__file__).resolve().parent
REPO = BUNDLE_DIR.parents[3]
# Git commit/tree identify ancestry only. The immutable parent archive plus its
# complete per-file inventory and the narrow derived-file overlay are the
# executable authority; worker validation never imports scientific source from
# the current live tree.
EXPECTED_HEAD = "8a746d244a15e2cb16099a732e78e1110a8e59f2"
EXPECTED_TREE = "6cb596ab953386a9c9a3b0698e7b1489e3b0f02e"
PROFILE_REQUEST = "sr_snake_no_prune_symmetric_cost_beam_v1"
PROFILE_RESOLVED = (
    "supported_whitened_adaptive_trust_full_response_symmetric_cost_"
    "no_prune_beam_v1"
)
PROFILE_CONTRACT_SHA256 = (
    "f932974ad3cdbd3b1b38239794cc9e7ab96a94502b53238bcdf5c5760f814a80"
)
PARENT_PROFILE_CONTRACT_SHA256 = (
    "69dcf6d5711a05b54cc143562bc6f437511d781164319b2705857b22deda1538"
)
FINAL_JOB_SCHEMA = "paper_i_hh_sr_appendix_historical_beam3x2_all_six_r50_job_v1"
REMOTE_IMAGE_PATH = Path("chtc/phase3_optuna/image.sif")
REMOTE_IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)
REMOTE_QISKIT_VERSION = "2.3.1"
REMOTE_FAKE_BACKEND_RESOLVED = "fake_marrakesh"
REMOTE_FAKE_BACKEND_QUBITS = 156
ARCHIVE_PATH = BUNDLE_DIR / "source_locked.tar.gz"
PARENT_BUNDLE_DIR = (
    BUNDLE_DIR.parent
    / "paper_i_hh_sr_snake_main_full_response_symmetric_cost_noprune_nobeam_"
    "no_ordinary_novelty_all_six_r50_20260718_v1_chtc"
)
PARENT_ARCHIVE_PATH = PARENT_BUNDLE_DIR / "source_locked.tar.gz"
PARENT_ARCHIVE_MANIFEST_PATH = PARENT_BUNDLE_DIR / "source_archive_manifest.json"
PARENT_ARCHIVE_SHA256 = (
    "fa9014b9608ccb8a301df0429268482abf6dd10c91eb336111a15d00be256d35"
)
BEAM_OVERLAY_PATH = BUNDLE_DIR / "historical_beam_exact_semantics_overlay.patch"
BEAM_OVERLAY_SHA256 = (
    "d2dea3a20a662a426001946b6c69dd1bf4a5afbf9bf45e1f29b9bed578b3409e"
)
BEAM_OVERLAY_FILE_HASHES = {
    "pipelines/static_adapt/adapt_pipeline.py": (
        "be706bd3aa4285e60e12202d8b45df9c1446e491cca7daa11671865a196ce2ab"
    ),
    "pipelines/static_adapt/sr_snake_route_profile.py": (
        "d3ea6d31f39a49792e389f954ebde777fed720a3a5b0460421b6584a85b1352d"
    ),
    "agent_guidance/static-adapt/route-identities.md": (
        "fb5343646f06419b81a91f63b746b92ad87237d0c5f801f5956e9fbd21cab479"
    ),
    "MATH/paper_facing/paper_I_static_scaffold/"
    "paper_i_sr_snake_no_prune_symmetric_cost_candidate_runtime_settings_"
    "20260717.md": (
        "deef3619e8453f2a5b90fd487c67eecadb0a467074274ff8cb8ccdb76295321b"
    ),
    "test/test_static_adapt_historical_beam_profile.py": (
        "ba2bd611eab28bb15b226f519128906f47b42a21f004b1e89863d192f7c1a5ef"
    ),
}
SOURCE_LOCK_STATE = "frozen_main_archive_plus_exact_historical_beam_overlay_v1"
# Keep Condor matchmaking impossible until the main agent has frozen the source
# and confirmed every local, archive-only, smoke, and remote execution gate.
# This skeleton is deliberately impossible to submit until the main agent
# freezes the final tested source and replaces the source-lock sentinels below.
SOURCE_FREEZE_COMPLETE = True
SUBMISSION_ENABLED = True
SUBMISSION_REGIMES = frozenset({
    "weak_weak", "intermediate_weak", "strong_weak_u8",
    "weak_strong", "intermediate_strong", "strong_strong_u8",
})

PHASE1_ENERGY_MODEL = "first_order_fs_trust_v1"
PHASE2_CURVATURE_POLICY = "measured_required_fail_closed_v1"
PHASE2_CHEAP_CURVATURE_PROXY_POLICY = "off"
RUN_CACHE_POLICY = {
    "candidate_record_cache": "disk",
    "hh_pool_cache": "disk",
    "hh_pool_cache_scope": "exact",
    "cache_namespace": "job_local_empty_on_start_v1",
    "cache_semantics": "performance_only_no_scientific_fallback_v1",
}
FIDELITY_POLICY = {
    "schema": "paper_i_hh_sr_post_run_projector_fidelity_policy_v1",
    "reference": "same_cutoff_physical_sector_ground_space_projector",
    "usage_scope": "post_run_reporting_only",
    "controller_decision_eligible": False,
    "optimizer_input_eligible": False,
    "stopping_input_eligible": False,
    "s_alg_charged": False,
    "persistence_required": True,
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
    "raw_outputs/"
    "paper_i_hh_sr_snake_no_prune_symmetric_cost_weak_weak_smoke_20260717/"
    "weak_weak_8_admissions_cache_off_v3"
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

REQUIRED_UNTRACKED_SOURCE_MODULES = {
    "pipelines/static_adapt/formal_manifold_outer_information.py": (
        "d0fbd924aba5b1630fce05c5701c75d2f20397ec08356d84a9d41e7794b2df91"
    ),
    "pipelines/static_adapt/formal_manifold_sr_v3_outer_bridge.py": (
        "fb8f18d159e19ce3b46fdabf7bcbab3a76611dadf65fbf027837ab7e551c2c5d"
    ),
}

# These reporting-only fidelity surfaces are part of the executable/archive
# closure even while some of them are not yet in the Git index.  The snapshot
# builder adds only the paths absent from the tracked ARCHIVE_PATHS selection;
# the set merge makes the transition to tracked files duplicate-free.
REQUIRED_HASH_LOCKED_FIDELITY_FILES = {
    "pipelines/scaffold/ground_space_fidelity.py": (
        "b6a7cba65995f536faa1d9bdb7210aea69918c2cb84babd6abe34c35f7c66ae3"
    ),
    "agent_guidance/skills/paper-i-results/scripts/"
    "compute_paper_i_main_fidelities.py": (
        "5534333b6ad14a440a8b5f4e1104d388a11048c1a27b90e7f8466f048cbe1a42"
    ),
    "test/test_ground_space_fidelity.py": (
        "55ff094aca73f59362886cb5b951c9ab4b70eff2733f9e4061970be88836a8bf"
    ),
}

# The broad cross-method fidelity audit imports comparator implementations that
# postdate the immutable SR parent archive.  Keep its exact live test identity
# as external evidence, but never overlay comparator scientific source into the
# SR worker archive merely to execute that cross-method audit there.
EXTERNAL_CROSS_METHOD_FIDELITY_AUDIT_EVIDENCE = {
    "path": "test/test_paper_i_main_fidelity_audit.py",
    "sha256": (
        "4963c6bd71d8706b4d9816da6c504e134635751293fbd44e5bdce30c74ab3271"
    ),
    "classification": "external_preflight_evidence_not_worker_source_v1",
}

CRITICAL_SOURCE_PATHS = (
    "pipelines/static_adapt/adapt_pipeline.py",
    "pipelines/static_adapt/cli_config.py",
    "pipelines/static_adapt/estimator_call_ledger.py",
    "pipelines/static_adapt/output_artifacts.py",
    "pipelines/static_adapt/resume_scaffold.py",
    "pipelines/static_adapt/sr_snake_phase12_policy.py",
    "pipelines/static_adapt/sr_snake_route_profile.py",
    "pipelines/static_adapt/accepted_refit.py",
    "pipelines/static_adapt/adapt_candidate_record_cache.py",
    "pipelines/scaffold/hh_continuation_scoring.py",
    "pipelines/scaffold/hh_continuation_types.py",
    "pipelines/scaffold/hh_continuation_pruning.py",
    "pipelines/scaffold/ground_space_fidelity.py",
    "pipelines/exact_bench/paper_i_hh_recovery_prefix_qiskit_sidecar.py",
    "pipelines/exact_bench/table_i_qiskit_resource_compile.py",
    "pipelines/hardcoded/adapt_circuit_execution.py",
    "pipelines/qiskit_backend_tools.py",
    "test/test_hh_continuation_scoring.py",
    "test/test_static_adapt_sr_route_profile.py",
    "test/test_static_adapt_historical_beam_profile.py",
    "test/test_static_adapt_historical_singleton_overlays.py",
    "test/test_static_adapt_sr_v4_runtime.py",
    "test/test_static_adapt_sr_v4_serialization.py",
    "test/test_static_adapt_accepted_refit.py",
    "test/test_static_adapt_estimator_call_ledger.py",
    "test/test_ground_space_fidelity.py",
    "agent_guidance/skills/paper-i-results/scripts/"
    "compute_paper_i_main_fidelities.py",
    "agent_guidance/static-adapt/route-identities.md",
    "agent_guidance/shared/run-guide.md",
    "MATH/paper_facing/paper_I_static_scaffold/"
    "paper_i_sr_snake_no_prune_symmetric_cost_candidate_runtime_settings_"
    "20260717.md",
)

CRITICAL_SOURCE_SHA256 = {
    "pipelines/static_adapt/adapt_pipeline.py": BEAM_OVERLAY_FILE_HASHES[
        "pipelines/static_adapt/adapt_pipeline.py"
    ],
    "pipelines/static_adapt/cli_config.py": "5f7f0bf1b08879eb72a6a88c138338ec8a272f76d2e999d8358777b58018829d",
    "pipelines/static_adapt/estimator_call_ledger.py": (
        "cce95198fc5d504cd55f496b3c41a89a4bbb06490311f6d69eb999d5eb5907ce"
    ),
    "pipelines/static_adapt/output_artifacts.py": "fbaa7b9d0a1534842dc40328ab66996680ed4bfeb8a655ce386adfc3fb7d0b20",
    "pipelines/static_adapt/resume_scaffold.py": "ce80b5ee8503366655ef09cbef0c7b9f19c41e217d6daed2b0fd1bd732199fd1",
    "pipelines/static_adapt/sr_snake_phase12_policy.py": "424bc48736e1a01f3c5897f589d7c00e91930adf00381e761fddaf7d685558cb",
    "pipelines/static_adapt/sr_snake_route_profile.py": BEAM_OVERLAY_FILE_HASHES[
        "pipelines/static_adapt/sr_snake_route_profile.py"
    ],
    "pipelines/static_adapt/accepted_refit.py": "a93e830343e9a9abfc93d499ad1882d7dc31ed5dd85f52b1c2f4ec53fc278975",
    "pipelines/static_adapt/adapt_candidate_record_cache.py": "5bcb3e89fbc5f340a6c4e423894755c526c32af95cc69a912ce215222f6669bb",
    "pipelines/scaffold/hh_continuation_scoring.py": "a66d2423d7312b5efa100818506c288e0a9283372097c59f293968a6a3a059a8",
    "pipelines/scaffold/hh_continuation_types.py": "8e46df8d859d54695f98e4b8f9157d38878c8c805c326cd88e42fc7c0ab851fb",
    "pipelines/scaffold/hh_continuation_pruning.py": "3b8be9adce5e52d7beab8fc66bcb4e2252327821c50eb4a2e82c5a1aee0f7ada",
    "pipelines/scaffold/ground_space_fidelity.py": (
        "b6a7cba65995f536faa1d9bdb7210aea69918c2cb84babd6abe34c35f7c66ae3"
    ),
    "pipelines/exact_bench/paper_i_hh_recovery_prefix_qiskit_sidecar.py": "5486c285deffcb47fd0f5ef0314a9e3ab2fd1c83ebb7e0bb72d629d6a81dd044",
    "pipelines/exact_bench/table_i_qiskit_resource_compile.py": "0578f2494273f731152d0194c68724b5bb12967abd5fba6e0708492a5e1723e3",
    "pipelines/hardcoded/adapt_circuit_execution.py": "1b569d31a45f98522b615fba0bb5645a6fba8af63ecc338f1059f14623364a0e",
    "pipelines/qiskit_backend_tools.py": "46fcfcce70479b5cad5346b456b689531d4f28fbc1200fe5ef22b5c68494c05b",
    "test/test_hh_continuation_scoring.py": "0d608a10b3ced205e7da3813410bf248c5da60b9f160afa52e90485fa5446b58",
    "test/test_static_adapt_sr_route_profile.py": "a16626014ee63de580d4137922fd19c9f98aefba09885903930e845c055159ed",
    "test/test_static_adapt_historical_beam_profile.py": BEAM_OVERLAY_FILE_HASHES[
        "test/test_static_adapt_historical_beam_profile.py"
    ],
    "test/test_static_adapt_historical_singleton_overlays.py": "5d01230030d36ec86276284ee5f5974052229bfd008797a6dddc7ee70f25ac2f",
    "test/test_static_adapt_sr_v4_runtime.py": "d35d7c9f99d7d1d0925b21d5d1559c4a099d6d6d0356974a2752d317264569a1",
    "test/test_static_adapt_sr_v4_serialization.py": "5d673eed47b3063920d9f2f075ccaa79459c991d60345a38472562cf3712c9d8",
    "test/test_static_adapt_accepted_refit.py": (
        "ee4874747d3dbe11335f7544b84544a6d7294c070e81b72883566f20573849e4"
    ),
    "test/test_static_adapt_estimator_call_ledger.py": (
        "9bfab183abd825439e902c87aa2abc09ef563da63f480a9db8293131b3dae73b"
    ),
    "test/test_ground_space_fidelity.py": (
        "55ff094aca73f59362886cb5b951c9ab4b70eff2733f9e4061970be88836a8bf"
    ),
    "agent_guidance/skills/paper-i-results/scripts/"
    "compute_paper_i_main_fidelities.py": (
        "5534333b6ad14a440a8b5f4e1104d388a11048c1a27b90e7f8466f048cbe1a42"
    ),
    "agent_guidance/static-adapt/route-identities.md": BEAM_OVERLAY_FILE_HASHES[
        "agent_guidance/static-adapt/route-identities.md"
    ],
    "agent_guidance/shared/run-guide.md": "5b69a0655af49e449b8ac9f6dc4c12a8ac027cda700ca8a3551eaea83cba4ff9",
    "MATH/paper_facing/paper_I_static_scaffold/paper_i_sr_snake_no_prune_symmetric_cost_candidate_runtime_settings_20260717.md": BEAM_OVERLAY_FILE_HASHES[
        "MATH/paper_facing/paper_I_static_scaffold/"
        "paper_i_sr_snake_no_prune_symmetric_cost_candidate_runtime_settings_"
        "20260717.md"
    ],
}

REGIMES: tuple[dict[str, Any], ...] = (
    {
        "slug": "weak_weak", "u": "0.25", "lambda": 0.25,
        "g_ep": "0.353553390593", "n_ph": 3,
        "exact_energy_decimal": "-0.918380919994822",
        "exact_energy": -0.918380919994822,
        "manifest_sha256": "24b8c50be54acc6eda506b38e9cd0583bd0ef88b1db6dac47e850b88040dc0b0",
        "result_sha256": "68fde0ab9de5ae69cee27ac0f54cb52f9e377882969daa0a1630d14f520ffdaa",
        "memory_mb": 32768, "disk_mb": 61440, "target_round": 50,
    },
    {
        "slug": "intermediate_weak", "u": "1.25", "lambda": 0.25,
        "g_ep": "0.353553390593", "n_ph": 3,
        "exact_energy_decimal": "-0.4950053491813613",
        "exact_energy": -0.4950053491813613,
        "manifest_sha256": "7ee0d9b6aea0f12418426a232e42a6962e2ef7cfa47f6dc3ba71061f139b1573",
        "result_sha256": "9e2479fd8308f111cd311e7843ccf1978962dc9d66876bc78f5c69566054a2ed",
        "memory_mb": 32768, "disk_mb": 61440, "target_round": 50,
    },
    {
        "slug": "strong_weak_u8", "u": "8.0", "lambda": 0.25,
        "g_ep": "0.353553390593", "n_ph": 3,
        "exact_energy_decimal": "0.5264586847939736",
        "exact_energy": 0.5264586847939736,
        "manifest_sha256": "67ff7a01a5cc1a33b34982e0a3511d889e4cc7aa7f93b7ea042d80bcf3ce5c0e",
        "result_sha256": "b62f89ef9271a2ff42eab2057e9183f48900355967adcabc1fbc9491d22a21f6",
        "memory_mb": 40960, "disk_mb": 61440, "target_round": 50,
    },
    {
        "slug": "weak_strong", "u": "0.25", "lambda": 1.25,
        "g_ep": "0.790569415042", "n_ph": 7,
        "exact_energy_decimal": "-1.1387206380749124",
        "exact_energy": -1.1387206380749124,
        "manifest_sha256": "6862cab52ebc8b49e15cdeada67c873b066918e5260b0e578244b36e52549c56",
        "result_sha256": "aaf2102a7829ac7a2b4c0f13ef55ef96fe81ed3b70f7d110e2d7c001b6d9cf3e",
        "memory_mb": 49152, "disk_mb": 81920, "target_round": 50,
    },
    {
        "slug": "intermediate_strong", "u": "1.25", "lambda": 1.25,
        "g_ep": "0.790569415042", "n_ph": 7,
        "exact_energy_decimal": "-0.6239396137518493",
        "exact_energy": -0.6239396137518493,
        "manifest_sha256": "ec5d436919af666fdfe1c28e8f243d44163637348451f46a368a73fb4eefd021",
        "result_sha256": "d00c8ab411fd87429f63095e5ab7cbea2c3b6d535228fbbaf3b5f60bf22499b0",
        "memory_mb": 49152, "disk_mb": 81920, "target_round": 50,
    },
    {
        "slug": "strong_strong_u8", "u": "8.0", "lambda": 1.25,
        "g_ep": "0.790569415042", "n_ph": 7,
        "exact_energy_decimal": "0.5205762765682517",
        "exact_energy": 0.5205762765682517,
        "manifest_sha256": "097bd59aff835fbfa39d5b603f384503b3372d0e3df2d480cb94d338399a902d",
        "result_sha256": "c0211bcfad1a7518857d17736ce3f7eccc9da9a2f993a8a7770208ac071b4a88",
        "memory_mb": 49152, "disk_mb": 81920, "target_round": 50,
    },
)

ARCHIVE_PATHS = (
    "src",
    "pipelines",
    "docs/reports",
    "test/test_static_adapt_sr_v4_runtime.py",
    "test/test_static_adapt_sr_v4_serialization.py",
    "test/test_static_adapt_accepted_refit.py",
    "test/test_static_adapt_estimator_call_ledger.py",
    "test/test_static_adapt_sr_route_profile.py",
    "test/test_static_adapt_historical_beam_profile.py",
    "test/test_static_adapt_historical_singleton_overlays.py",
    "test/test_static_adapt_resume_scaffold.py",
    "test/test_adapt_candidate_record_cache.py",
    "test/test_hh_continuation_scoring.py",
    "test/test_ground_space_fidelity.py",
    "agent_guidance/skills/paper-i-results/scripts/"
    "compute_paper_i_main_fidelities.py",
    "agent_guidance/static-adapt/route-identities.md",
    "agent_guidance/shared/run-guide.md",
    "MATH/paper_facing/paper_I_static_scaffold/"
    "paper_i_sr_snake_no_prune_symmetric_cost_candidate_runtime_settings_"
    "20260717.md",
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
            value.replace(BUNDLE_ID, "<BUNDLE_ID>")
            .replace("20260716-v2", "<REVISION>")
        )
    return value


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def git_blob(path: str) -> bytes:
    return (REPO / path).read_bytes()


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


def verify_required_untracked_source_modules() -> dict[str, str]:
    """Lock required executable modules that are absent from the base commit."""

    verified: dict[str, str] = {}
    for relative, expected in REQUIRED_UNTRACKED_SOURCE_MODULES.items():
        path = REPO / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"required untracked source module missing: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(
                f"required untracked source module drift: {relative}: "
                f"expected {expected}, got {actual}"
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
                f"required live-source module unexpectedly exists in base commit: "
                f"{relative}"
            )
        verified[relative] = actual
    return verified


def verify_required_hash_locked_fidelity_files() -> dict[str, str]:
    """Fail closed on every fidelity source/test required by the archive."""

    verified: dict[str, str] = {}
    for relative, expected in REQUIRED_HASH_LOCKED_FIDELITY_FILES.items():
        path = REPO / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"required fidelity source is missing: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(
                f"required fidelity source drift: {relative}: "
                f"expected {expected}, got {actual}"
            )
        verified[relative] = actual
    return verified


def required_hash_locked_overlay_paths(
    tracked_archive_paths: set[str],
) -> tuple[str, ...]:
    """Return only required fidelity files absent from the tracked surface."""

    return tuple(
        sorted(set(REQUIRED_HASH_LOCKED_FIDELITY_FILES) - tracked_archive_paths)
    )


def tracked_snapshot_file_paths() -> set[str]:
    """Return Git-indexed files selected by the declared archive surface."""

    raw = subprocess.check_output(
        ["git", "ls-files", "-z", "--", *ARCHIVE_PATHS], cwd=REPO
    )
    return {token.decode("utf-8") for token in raw.split(b"\0") if token}


def snapshot_file_paths() -> tuple[str, ...]:
    """Return the tracked archive surface plus explicit untracked locks/shims."""

    selected = tracked_snapshot_file_paths()
    selected.update(NONSCIENTIFIC_ARCHIVE_OVERLAYS)
    selected.update(REQUIRED_UNTRACKED_SOURCE_MODULES)
    selected.update(required_hash_locked_overlay_paths(selected))
    selected.add(
        "MATH/paper_facing/paper_I_static_scaffold/"
        "paper_i_sr_snake_no_prune_symmetric_cost_candidate_runtime_settings_"
        "20260717.md"
    )
    for relative in selected:
        path = REPO / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"snapshot source is missing or not regular: {relative}")
    return tuple(sorted(selected))


def _safe_parent_archive_files() -> dict[str, tuple[bytes, int]]:
    """Load and fully authenticate the immutable parent archive."""

    if not PARENT_ARCHIVE_PATH.is_file():
        raise ValueError(f"immutable parent archive missing: {PARENT_ARCHIVE_PATH}")
    if sha256(PARENT_ARCHIVE_PATH) != PARENT_ARCHIVE_SHA256:
        raise ValueError("immutable parent archive SHA-256 drift")
    manifest = load_json(PARENT_ARCHIVE_MANIFEST_PATH)
    if (
        manifest.get("archive_sha256") != PARENT_ARCHIVE_SHA256
        or manifest.get("git_commit") != EXPECTED_HEAD
        or manifest.get("git_tree") != EXPECTED_TREE
    ):
        raise ValueError("immutable parent archive manifest authority drift")
    expected_files = manifest.get("files")
    if not isinstance(expected_files, dict):
        raise ValueError("immutable parent archive lacks a complete file inventory")
    files: dict[str, tuple[bytes, int]] = {}
    with tarfile.open(PARENT_ARCHIVE_PATH, "r:gz") as handle:
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
                raise ValueError(f"unsafe immutable parent member: {member.name}")
            if not member.isfile():
                continue
            stream = handle.extractfile(member)
            if stream is None:
                raise ValueError(f"unreadable immutable parent member: {member.name}")
            data = stream.read()
            relative = name.as_posix()
            record = expected_files.get(relative)
            if not isinstance(record, dict):
                raise ValueError(f"parent member absent from inventory: {relative}")
            if (
                record.get("sha256") != bytes_sha256(data)
                or int(record.get("size_bytes", -1)) != len(data)
            ):
                raise ValueError(f"parent member hash/size drift: {relative}")
            files[relative] = (data, int(member.mode) & 0o777)
    if set(files) != set(expected_files) or len(files) != int(
        manifest.get("file_count", -1)
    ):
        raise ValueError("immutable parent archive inventory is not closed")
    return files


def derived_source_files() -> dict[str, tuple[bytes, int]]:
    """Derive the exact appendix worker source from the frozen main archive."""

    files = _safe_parent_archive_files()
    if sha256(BEAM_OVERLAY_PATH) != BEAM_OVERLAY_SHA256:
        raise ValueError("historical beam overlay hash drift")
    with tempfile.TemporaryDirectory(prefix="sr_historical_beam_overlay_") as tmp:
        root = Path(tmp)
        for relative, (data, mode) in files.items():
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            destination.chmod(mode)
        completed = subprocess.run(
            ["/usr/bin/patch", "-p1", "--fuzz=0", "--batch", "--forward"],
            cwd=root,
            input=BEAM_OVERLAY_PATH.read_text(encoding="utf-8"),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise ValueError(
                "historical beam overlay failed to apply exactly: "
                + completed.stdout.strip()
                + " "
                + completed.stderr.strip()
            )
        for relative, expected in BEAM_OVERLAY_FILE_HASHES.items():
            destination = root / relative
            if not destination.is_file() or destination.is_symlink():
                raise ValueError(f"historical beam overlay omitted {relative}")
            data = destination.read_bytes()
            if bytes_sha256(data) != expected:
                raise ValueError(f"historical beam overlay result drift: {relative}")
            prior_mode = files.get(relative, (b"", 0o644))[1]
            files[relative] = (data, int(prior_mode or 0o644))
    return files


def exact_source_tar_bytes() -> bytes:
    """Materialize a deterministic tar from parent plus narrow overlays."""

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as handle:
        for relative, (data, mode) in sorted(derived_source_files().items()):
            info = tarfile.TarInfo(relative)
            info.size = len(data)
            info.mode = int(mode)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            handle.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def extract_exact_source(destination: Path) -> None:
    """Extract only the hash-locked tested worktree snapshot."""

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


def exact_source_contract_probe() -> dict[str, Any]:
    """Resolve the candidate profile from isolated source, never the live tree."""

    with tempfile.TemporaryDirectory(prefix="sr_symcost_exact_source_probe_") as tmp:
        root = Path(tmp)
        extract_exact_source(root)
        code = (
            "import json; "
            "from pipelines.static_adapt.sr_snake_route_profile import "
            "canonical_sr_snake_contract,canonical_sr_snake_contract_sha256,"
            "normalize_sr_route_profile_request; "
            f"print(json.dumps({{'resolved':normalize_sr_route_profile_request({PROFILE_REQUEST!r}),"
            f"'digest':canonical_sr_snake_contract_sha256({PROFILE_REQUEST!r}),"
            f"'contract':canonical_sr_snake_contract({PROFILE_REQUEST!r})}},"
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
        raise ValueError("immutable parent ancestry metadata is malformed")
    parent_manifest = load_json(PARENT_ARCHIVE_MANIFEST_PATH)
    if (
        parent_manifest.get("archive_sha256") != PARENT_ARCHIVE_SHA256
        or parent_manifest.get("git_commit") != EXPECTED_HEAD
        or parent_manifest.get("git_tree") != EXPECTED_TREE
    ):
        raise ValueError("immutable parent source authority drift")
    derived = derived_source_files()
    external_audit_path = REPO / str(
        EXTERNAL_CROSS_METHOD_FIDELITY_AUDIT_EVIDENCE["path"]
    )
    if (
        not external_audit_path.is_file()
        or sha256(external_audit_path)
        != EXTERNAL_CROSS_METHOD_FIDELITY_AUDIT_EVIDENCE["sha256"]
    ):
        raise ValueError("external cross-method fidelity audit evidence drift")
    required_fidelity = {
        relative: bytes_sha256(derived[relative][0])
        for relative in REQUIRED_HASH_LOCKED_FIDELITY_FILES
    }
    critical = {}
    if set(CRITICAL_SOURCE_PATHS) != set(CRITICAL_SOURCE_SHA256):
        raise ValueError("critical source path/hash inventory mismatch")
    for path, expected in CRITICAL_SOURCE_SHA256.items():
        if path not in derived:
            raise ValueError(f"critical source absent from derived archive: {path}")
        actual = bytes_sha256(derived[path][0])
        if actual != expected:
            raise ValueError(
                f"critical derived-source drift: {path}: expected {expected}, got {actual}"
            )
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
        raise ValueError("candidate profile resolution/digest drift")
    execution_settings = contract["execution_settings"]
    semantic_invariants = contract["semantic_invariants"]
    if "adapt_max_depth" in execution_settings:
        raise ValueError("candidate horizon must come only from the per-regime lock")
    if execution_settings.get("adapt_finite_angle_fallback") is not False:
        raise ValueError("candidate finite-angle fallback must be disabled")
    if semantic_invariants.get("finite_angle_fallback_active") is not False:
        raise ValueError("candidate finite-angle semantic invariant must be false")
    if execution_settings.get("phase3_enable_rescue") is not False:
        raise ValueError("candidate Phase-III rescue must be disabled")
    expected_phase12 = {
        "phase1_energy_model": PHASE1_ENERGY_MODEL,
        "phase2_curvature_policy": PHASE2_CURVATURE_POLICY,
        "phase2_cheap_curvature_proxy_policy": (
            PHASE2_CHEAP_CURVATURE_PROXY_POLICY
        ),
    }
    for key, expected in expected_phase12.items():
        if execution_settings.get(key) != expected:
            raise ValueError(f"candidate {key} drift: {execution_settings.get(key)!r}")
        if semantic_invariants.get(key) != expected:
            raise ValueError(f"candidate semantic invariant {key} drift")
    if semantic_invariants.get("phase1_phase2_lambda_f_proxy_active") is not False:
        raise ValueError("candidate lambda-F proxy semantic invariant must be false")
    if semantic_invariants.get("phase2_curvature_failure_policy") != "abort_run_v1":
        raise ValueError("candidate Phase-II curvature failure policy must abort")
    required_execution = {
        "phase2_gram_novelty_policy": "fallback_only_v1",
        "phase3_gram_novelty_policy": "fallback_only_v1",
        "phase1_prune_enabled": False,
        "adapt_beam_live_branches": 3,
        "adapt_beam_children_per_parent": 2,
        "adapt_beam_terminated_keep": 3,
        "adapt_beam_terminal_archive_mode": "legacy",
        "adapt_beam_lambda": 0.005,
        "phase2_enable_batching": False,
        "phase3_enable_batching": False,
        "phase3_shadow_damping_policy": "off",
        "historical_singleton_coordinate_solve_scope": "phase3_only_v1",
        "historical_singleton_coordinate_solve_policy": (
            "supported_metric_whitened_eigh_v1"
        ),
        "historical_singleton_trust_region_update_policy": (
            "displacement_calibrated_unbounded_v2"
        ),
        "phase3_response_coordinate_scope": "full_active_plus_singleton_v1",
        "adapt_accepted_refit_scope": "full_ansatz_v1",
        "adapt_accepted_refit_coordinate_chart": "supported_fs_whitened_fixed_v1",
        "adapt_accepted_refit_base_chart_policy": (
            "expanded_runtime_projected_logical_v1"
        ),
        "adapt_full_refit_every": 0,
        "adapt_final_full_refit": "false",
        "phase3_hardware_cost_normalization_mode": (
            "family_robust_symmetric_arctan_v1"
        ),
        "adapt_disable_hh_seed": True,
        "adapt_seed": 7,
    }
    for key, expected in required_execution.items():
        if execution_settings.get(key) != expected:
            raise ValueError(f"candidate executable setting drift: {key}")
    required_semantics = {
        "ordinary_phase2_novelty_multiplier_active": False,
        "ordinary_phase3_novelty_multiplier_active": False,
        "all_energy_models_infeasible_novelty_fallback_active": True,
        "all_energy_models_infeasible_novelty_fallback_telemetry_required": True,
        "pruning_active": False,
        "phase2_supported_whitening_active": False,
        "phase3_supported_whitening_active": True,
        "negative_curvature_escape_active": False,
        "periodic_full_refit_active": False,
        "terminal_full_refit_active": False,
        "terminal_prune_active": False,
        "controller_horizon_source": "per_regime_source_lock",
        "appendix_one_factor_ablation": True,
        "beam_shape": "historical_3x2_v1",
        "beam_live_branch_cap": 3,
        "beam_children_per_parent": 2,
        "beam_expanded_child_cap_per_round": 6,
        "beam_terminated_keep": 3,
        "beam_terminal_archive_mode": "legacy",
        "beam_parent_stop_terminal_also_materialized": True,
        "beam_structural_mode": "stop_or_single_admission",
        "beam_terminal_archive_accumulation": "cumulative_across_rounds",
        "beam_terminal_archive_cap": 3,
    }
    for key, expected in required_semantics.items():
        if semantic_invariants.get(key) != expected:
            raise ValueError(f"candidate semantic invariant drift: {key}")

    revision = {
        "schema": "paper_i_hh_sr_historical_beam3x2_source_revision_v1",
        "git_commit": EXPECTED_HEAD,
        "git_tree": EXPECTED_TREE,
        "git_role": "base_ancestry_metadata_only",
        "executable_source_authority": (
            "frozen_main_archive_plus_exact_historical_beam_overlay_inventory_v1"
        ),
        "dirty_live_source_lock": False,
        "profile_request": PROFILE_REQUEST,
        "profile_resolved": resolved,
        "profile_contract_sha256": contract_digest,
        "phase1_energy_model": PHASE1_ENERGY_MODEL,
        "phase2_curvature_policy": PHASE2_CURVATURE_POLICY,
        "phase2_cheap_curvature_proxy_policy": (
            PHASE2_CHEAP_CURVATURE_PROXY_POLICY
        ),
        "critical_source_sha256": critical,
        "immutable_parent_archive": {
            "path": PARENT_ARCHIVE_PATH.relative_to(REPO).as_posix(),
            "sha256": PARENT_ARCHIVE_SHA256,
            "file_count": int(parent_manifest["file_count"]),
        },
        "historical_beam_overlay": {
            "classification": "appendix_one_factor_scientific_overlay_v1",
            "path": BEAM_OVERLAY_PATH.relative_to(REPO).as_posix(),
            "overlay_sha256": BEAM_OVERLAY_SHA256,
            "derived_file_sha256": dict(BEAM_OVERLAY_FILE_HASHES),
            "parent_route_contract_sha256": PARENT_PROFILE_CONTRACT_SHA256,
            "exact_historical_source_commit": (
                "1f1d93c1a0060f0db70da6736cae4ec5ffffc79b^"
            ),
            "exact_semantics": {
                "beam_shape": "3_live_x_2_children",
                "max_admission_children_per_round": 6,
                "parent_stop_terminal_also_materialized": True,
                "structural_mode": "stop_or_single_admission",
                "terminal_archive_accumulation": "cumulative_across_rounds",
                "terminal_archive_cap": 3,
                "beam_lambda": 0.005,
            },
        },
        "required_untracked_source_modules": parent_manifest[
            "required_untracked_source_modules"
        ],
        "required_hash_locked_fidelity_files": {
            relative: {
                "sha256": digest,
                "classification": "reporting_only_fidelity_source_or_test",
                "tracked_in_parent_archive": relative in parent_manifest["files"],
            }
            for relative, digest in required_fidelity.items()
        },
        "required_untracked_hash_overlays": {
            relative: {
                "sha256": required_fidelity[relative],
                "classification": "reporting_only_fidelity_archive_overlay",
                "tracked_in_parent_archive": False,
            }
            for relative in sorted(required_fidelity)
            if relative not in parent_manifest["files"]
        },
        "external_cross_method_fidelity_audit_evidence": {
            **EXTERNAL_CROSS_METHOD_FIDELITY_AUDIT_EVIDENCE,
            "included_in_worker_archive": False,
        },
        "non_scientific_archive_overlays": parent_manifest[
            "non_scientific_archive_overlays"
        ],
    }
    physics_lock = {
        "schema": "paper_i_hh_sr_historical_beam3x2_physics_exact_reference_lock_v1",
        "same_cutoff_required": True,
        "manual_exact_energy_override_forbidden": True,
        "runtime_exact_energy_recomputed": True,
        "runtime_exact_energy_tolerance": 1.0e-12,
        "rows": physics_rows,
    }
    return revision, {"contract": contract, "physics_lock": physics_lock}


def build_source_archive() -> dict[str, Any]:
    parent_manifest = load_json(PARENT_ARCHIVE_MANIFEST_PATH)
    required_fidelity = dict(REQUIRED_HASH_LOCKED_FIDELITY_FILES)
    raw = exact_source_tar_bytes()
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
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
    for path, expected in CRITICAL_SOURCE_SHA256.items():
        if members.get(path, {}).get("sha256") != expected:
            raise ValueError(f"critical file missing/drifted in archive: {path}")
    for path, expected in required_fidelity.items():
        if members.get(path, {}).get("sha256") != expected:
            raise ValueError(
                f"required fidelity source missing/drifted in archive: {path}"
            )
    return {
        "schema": "paper_i_hh_sr_historical_beam3x2_source_archive_manifest_v1",
        "archive": ARCHIVE_PATH.relative_to(REPO).as_posix(),
        "archive_sha256": sha256(ARCHIVE_PATH),
        "archive_size_bytes": ARCHIVE_PATH.stat().st_size,
        "git_commit": EXPECTED_HEAD,
        "git_tree": EXPECTED_TREE,
        "git_role": "base_ancestry_metadata_only",
        "executable_source_authority": (
            "derived_archive_sha256_plus_complete_per_file_sha256_inventory_v1"
        ),
        "worker_source_mode": SOURCE_LOCK_STATE,
        "worker_pythonpath": "/work",
        "immutable_parent_archive": {
            "path": PARENT_ARCHIVE_PATH.relative_to(REPO).as_posix(),
            "sha256": PARENT_ARCHIVE_SHA256,
            "file_count": int(parent_manifest["file_count"]),
        },
        "historical_beam_overlay": {
            "classification": "appendix_one_factor_scientific_overlay_v1",
            "path": BEAM_OVERLAY_PATH.relative_to(REPO).as_posix(),
            "overlay_sha256": BEAM_OVERLAY_SHA256,
            "derived_file_sha256": dict(BEAM_OVERLAY_FILE_HASHES),
            "parent_route_contract_sha256": PARENT_PROFILE_CONTRACT_SHA256,
        },
        "non_scientific_archive_overlays": parent_manifest[
            "non_scientific_archive_overlays"
        ],
        "required_untracked_source_modules": parent_manifest[
            "required_untracked_source_modules"
        ],
        "required_hash_locked_fidelity_files": {
            relative: {
                "sha256": digest,
                "classification": "reporting_only_fidelity_source_or_test",
                "tracked_in_parent_archive": relative in parent_manifest["files"],
            }
            for relative, digest in required_fidelity.items()
        },
        "required_untracked_hash_overlays": {
            relative: {
                "sha256": required_fidelity[relative],
                "classification": "reporting_only_fidelity_archive_overlay",
                "tracked_in_parent_archive": False,
            }
            for relative in sorted(required_fidelity)
            if relative not in parent_manifest["files"]
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
    with tempfile.TemporaryDirectory(prefix="sr_symcost_archive_preflight_") as tmp:
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
    "resolved": route.normalize_sr_route_profile_request({PROFILE_REQUEST!r}),
    "digest": route.canonical_sr_snake_contract_sha256({PROFILE_REQUEST!r}),
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
            "test/test_static_adapt_historical_beam_profile.py",
            "test/test_static_adapt_historical_singleton_overlays.py",
            "test/test_static_adapt_resume_scaffold.py",
            "test/test_adapt_candidate_record_cache.py",
            "test/test_hh_continuation_scoring.py",
            "test/test_static_adapt_accepted_refit.py",
            "test/test_static_adapt_estimator_call_ledger.py",
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
            "schema": "paper_i_hh_sr_historical_beam3x2_archive_only_preflight_v1",
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
    target_round = int(row["target_round"])
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
        "ground_space_fidelity_json": (
            root / "ground_space_projector_fidelity.json"
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
        "--adapt-max-depth", str(target_round),
        "--adapt-segment-id", (
            f"{slug}-sr-appendix-historical-beam3x2-r0-r{target_round}-20260718-v1"
        ),
        "--adapt-segment-target-controller-round", str(target_round),
        "--adapt-segment-target-depth", str(target_round),
        "--adapt-segment-max-new-admissions", str(target_round),
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
        "schema": FINAL_JOB_SCHEMA,
        "bundle_id": BUNDLE_ID,
        "batch_name": BATCH_NAME,
        "run_class": "appendix_historical_beam_ablation_matrix",
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
            "g_ep": float(row["g_ep"]),
            "g_ep_decimal_12": str(row["g_ep"]),
            "n_ph_work": int(row["n_ph"]),
            "n_ph_reference": int(row["n_ph"]),
            "same_cutoff_reference": True,
            "expected_exact_energy": float(row["exact_energy"]),
            "expected_exact_energy_decimal": str(row["exact_energy_decimal"]),
            "exact_energy_tolerance": 1.0e-12,
        },
        "segment": {
            "source_controller_round": 0, "source_depth": 0,
            "target_controller_round": int(row["target_round"]),
            "target_depth": int(row["target_round"]),
            "max_new_admissions": int(row["target_round"]),
            "future_continuation_required_after_validation": False,
            "future_continuation_target": None,
            "terminal_qiskit_sidecar_outer_iteration": int(row["target_round"]),
            "terminal_qiskit_sidecar_required": True,
            "post_run_projector_fidelity_required": True,
            "post_run_projector_fidelity_policy": FIDELITY_POLICY,
            "terminal_qiskit_checkpoint_order_policy": (
                "repair_permutation_only_execution_order_fail_closed_v1"
            ),
        },
        "command": {
            "argv": argv,
            "method_configuration_surface": (
                "sr_route_profile_plus_source_locked_regime_horizon"
            ),
            "explicit_method_overrides": ["adapt_max_depth"],
            "manual_exact_reference_override": False,
        },
        "environment": environment,
        "cache_policy": RUN_CACHE_POLICY,
        "evidence_requirements": {
            "exact_s_alg_ledger_closure_required": True,
            "active_prefix_estimator_receipt_each_round_required": True,
            "terminal_estimator_closure_receipt_required": True,
            "fallback_telemetry_required": True,
            "full_active_plus_singleton_response_each_round_required": True,
            "full_accepted_refit_each_round_required": True,
            "symmetry_and_padding_leakage_gate_required": True,
            "exact_round_50_horizon_required": True,
            "post_run_projector_fidelity": FIDELITY_POLICY,
        },
        "paths": paths,
        "source_lock": {
            "git_commit": EXPECTED_HEAD, "git_tree": EXPECTED_TREE,
            "source_archive": archive["archive"],
            "source_archive_sha256": archive["archive_sha256"],
            "worker_source_mode": archive["worker_source_mode"],
            "non_scientific_archive_overlays": archive[
                "non_scientific_archive_overlays"
            ],
            "required_untracked_source_modules": archive[
                "required_untracked_source_modules"
            ],
            "required_hash_locked_fidelity_files": archive[
                "required_hash_locked_fidelity_files"
            ],
            "required_untracked_hash_overlays": archive[
                "required_untracked_hash_overlays"
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


def validate_candidate_bundle_contract(
    *, contract: dict[str, Any], jobs: list[dict[str, Any]]
) -> dict[str, Any]:
    execution = contract["execution_settings"]
    semantics = contract["semantic_invariants"]
    required_execution = {
        "phase1_energy_model": PHASE1_ENERGY_MODEL,
        "phase2_curvature_policy": PHASE2_CURVATURE_POLICY,
        "phase2_cheap_curvature_proxy_policy": PHASE2_CHEAP_CURVATURE_PROXY_POLICY,
        "phase2_gram_novelty_policy": "fallback_only_v1",
        "phase3_gram_novelty_policy": "fallback_only_v1",
        "phase1_prune_enabled": False,
        "adapt_beam_live_branches": 3,
        "adapt_beam_children_per_parent": 2,
        "adapt_beam_terminated_keep": 3,
        "adapt_beam_terminal_archive_mode": "legacy",
        "adapt_beam_lambda": 0.005,
        "phase2_enable_batching": False,
        "phase3_enable_batching": False,
        "phase3_shadow_damping_policy": "off",
        "adapt_finite_angle_fallback": False,
        "phase3_enable_rescue": False,
        "adapt_full_refit_every": 0,
        "adapt_final_full_refit": "false",
        "phase3_hardware_cost_normalization_mode": (
            "family_robust_symmetric_arctan_v1"
        ),
        "adapt_seed": 7,
        "adapt_disable_hh_seed": True,
    }
    required_semantics = {
        "ordinary_phase2_novelty_multiplier_active": False,
        "ordinary_phase3_novelty_multiplier_active": False,
        "all_energy_models_infeasible_novelty_fallback_active": True,
        "all_energy_models_infeasible_novelty_fallback_telemetry_required": True,
        "pruning_active": False,
        "appendix_one_factor_ablation": True,
        "beam_shape": "historical_3x2_v1",
        "beam_live_branch_cap": 3,
        "beam_children_per_parent": 2,
        "beam_expanded_child_cap_per_round": 6,
        "beam_terminated_keep": 3,
        "beam_terminal_archive_mode": "legacy",
        "beam_parent_stop_terminal_also_materialized": True,
        "beam_structural_mode": "stop_or_single_admission",
        "beam_terminal_archive_accumulation": "cumulative_across_rounds",
        "beam_terminal_archive_cap": 3,
        "phase2_supported_whitening_active": False,
        "phase3_supported_whitening_active": True,
        "negative_curvature_escape_active": False,
        "periodic_full_refit_active": False,
        "terminal_full_refit_active": False,
        "terminal_prune_active": False,
        "controller_horizon_source": "per_regime_source_lock",
    }
    for key, expected in required_execution.items():
        if execution.get(key) != expected:
            raise ValueError(f"candidate executable contract drift: {key}")
    for key, expected in required_semantics.items():
        if semantics.get(key) != expected:
            raise ValueError(f"candidate semantic contract drift: {key}")
    parent_contract = load_json(PARENT_BUNDLE_DIR / "jobs/weak_weak.json")[
        "route_identity"
    ]["profile_contract"]
    execution_diff = recursive_diff(
        parent_contract["execution_settings"], execution
    )
    expected_execution_diff = {
        "adapt_beam_children_per_parent": (1, 2),
        "adapt_beam_live_branches": (1, 3),
        "adapt_beam_terminal_archive_mode": ("disabled", "legacy"),
        "adapt_beam_terminated_keep": (0, 3),
    }
    observed_execution_diff = {
        str(row["path"]): (row["v2"], row["v3"])
        for row in execution_diff
    }
    if observed_execution_diff != expected_execution_diff:
        raise ValueError(
            "appendix route is not the exact one-factor historical beam "
            f"variant: {observed_execution_diff}"
        )
    rows: list[dict[str, Any]] = []
    expected_rows = {str(row["slug"]): row for row in REGIMES}
    if {str(job["regime_slug"]) for job in jobs} != set(expected_rows):
        raise ValueError("candidate job matrix is not exactly the six locked regimes")
    for job in jobs:
        slug = str(job["regime_slug"])
        source = expected_rows[slug]
        target = int(source["target_round"])
        segment = job["segment"]
        argv = list(job["command"]["argv"])
        checks = {
            "profile_digest_exact": (
                job["route_identity"]["profile_contract_sha256"]
                == PROFILE_CONTRACT_SHA256
            ),
            "same_cutoff": (
                int(job["physics"]["n_ph_work"])
                == int(job["physics"]["n_ph_reference"])
            ),
            "g_ep_12_digit_lock": (
                str(job["physics"]["g_ep_decimal_12"])
                == str(source["g_ep"])
                and len(str(source["g_ep"]).split(".")[-1]) == 12
            ),
            "exact_reference_decimal_lock": (
                str(job["physics"]["expected_exact_energy_decimal"])
                == str(source["exact_energy_decimal"])
            ),
            "cache_policy_explicit": job.get("cache_policy") == RUN_CACHE_POLICY,
            "post_run_fidelity_reporting_only": (
                job.get("evidence_requirements", {})
                .get("post_run_projector_fidelity") == FIDELITY_POLICY
                and job["segment"].get("post_run_projector_fidelity_required")
                is True
            ),
            "fresh_round_zero": (
                int(segment["source_controller_round"]) == 0
                and int(segment["source_depth"]) == 0
            ),
            "exact_horizon": (
                int(segment["target_controller_round"]) == target
                and int(segment["target_depth"]) == target
                and int(segment["max_new_admissions"]) == target
            ),
            "horizon_explicit_in_argv": (
                "--adapt-max-depth" in argv
                and argv[argv.index("--adapt-max-depth") + 1] == str(target)
            ),
            "no_manual_exact_override": (
                "--adapt-exact-gs-override" not in argv
                and "--adapt-exact-gs-reference-json" not in argv
            ),
            "profile_only_plus_horizon": (
                job["command"]["explicit_method_overrides"]
                == ["adapt_max_depth"]
            ),
        }
        if not all(checks.values()):
            raise ValueError(f"candidate job contract failed for {slug}: {checks}")
        rows.append({"regime_slug": slug, "target_round": target, "checks": checks})
    return {
        "schema": "paper_i_hh_sr_appendix_historical_beam3x2_settings_audit_v1",
        "status": "pass",
        "profile_request": PROFILE_REQUEST,
        "profile_resolved": PROFILE_RESOLVED,
        "profile_contract_sha256": PROFILE_CONTRACT_SHA256,
        "response_model_damping": {
            "scientific_h_plus_mu_g_damping": "off",
            "historical_numerical_schur_ridge": 1.0e-6,
            "supported_solve_numerical_tolerance": 1.0e-9,
            "classification": (
                "numerical_stability_tolerances_are_not_scientific_damping"
            ),
        },
        "approved_contract": {
            "required_execution_settings": required_execution,
            "required_semantic_invariants": required_semantics,
            "per_regime_horizons": {
                slug: int(row["target_round"])
                for slug, row in expected_rows.items()
            },
        },
        "immutable_parent": {
            "bundle": PARENT_BUNDLE_DIR.relative_to(REPO).as_posix(),
            "source_archive_sha256": PARENT_ARCHIVE_SHA256,
            "route_contract_sha256": PARENT_PROFILE_CONTRACT_SHA256,
        },
        "one_factor_execution_diff": {
            key: {"parent": values[0], "appendix_beam": values[1]}
            for key, values in sorted(observed_execution_diff.items())
        },
        "rows": rows,
        "unexpected_executable_differences": [],
    }


def submit_text(archive_sha: str) -> str:
    base = (BUNDLE_DIR.relative_to(REPO)).as_posix()
    requirements = "TARGET.HasSIF" if SUBMISSION_ENABLED else "False"
    return f"""universe = vanilla
# Generated from the hash-locked locally tested source snapshot.
# SUBMISSION_ENABLED is explicitly set after every local/archive/remote gate.
executable = {base}/execute_source_locked_job.sh
arguments = $(job_manifest) {base}/source_locked.tar.gz {archive_sha} chtc/phase3_optuna/image.sif {REMOTE_IMAGE_SHA256} $(regime_slug)
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_executable = True
preserve_relative_paths = True
transfer_input_files = {base}/run_job.py, {base}/evidence_validation.py, {base}/validate_fetched.py, {base}/source_archive_manifest.json, {base}/source_revision_manifest.json, {base}/physics_and_exact_reference_lock.json, {base}/bundle_manifest.json, {base}/preflight.json, {base}/route_parity.json, {base}/scientific_settings_audit.json, $(job_manifest), $(normalized_manifest), {base}/source_locked.tar.gz, chtc/phase3_optuna/image.sif
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
    scientific_validation = {
        "status": "immutable_parent_smoke_reference_only",
        "note": (
            "This immutable depth-8 smoke validates inherited non-beam science "
            "settings only. Exact historical beam behavior is proved by the "
            "archive-only route and source regressions; no appendix science ran."
        ),
        "result_sha256": evidence_hashes["json/result.json"],
        "current_sha256": evidence_hashes["json/current.json"],
        "ledger_sha256": evidence_hashes["json/estimator_call_ledger.json"],
    }
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
    if full_candidates != 404:
        raise ValueError(
            f"authoritative depth-8 smoke receipt count drift: {full_candidates}"
        )
    s_alg = int(
        result.get("adapt_vqe", {})
        .get("estimator_call_accounting", {})
        .get("all_branch_search_work", {})
        .get("S_alg", -1)
    )
    if s_alg != 7312:
        raise ValueError(f"authoritative depth-8 smoke S_alg drift: {s_alg}")

    return {
        "schema": "paper_i_hh_sr_symcost_noprune_local_smoke_evidence_v1",
        "status": "pass_for_bundle_construction_not_a_production_result",
        "records": [{
            "label": "eight_admission_cache_off_symcost_noprune_candidate_v1",
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
            "shadow_damping_off",
            "historical_beam_exact_semantics_archive_regression",
            "prune_disabled_zero_trials",
            "ordinary_novelty_multipliers_off",
            "infeasible_model_fallback_explicit_telemetry",
        ],
        "exact_blockers": [],
        "note": (
            "Diagnostic energies are not production comparisons; these records "
            "establish executable route/profile, checkpoint, trust, fallback, "
            "no-prune and estimator-accounting health inherited from the parent; "
            "this smoke is not evidence from the historical-beam variant."
        ),
    }


def remote_preflight_and_cleanup_receipt() -> dict[str, Any]:
    gate_path = BUNDLE_DIR / "remote_execution_gate.json"
    if not gate_path.is_file():
        return {
            "schema": "paper_i_hh_sr_symcost_noprune_remote_preflight_receipt_v1",
            "status": "blocked_pending_remote_preflight",
            "remote_execution_preflight": {},
            "remote_execution_gate": gate_path.relative_to(REPO).as_posix(),
            "remote_execution_gate_sha256": None,
            "storage_cleanup": {
                "scope": "no_cleanup_authorized_or_required_by_bundle_builder",
                "remote_removed_paths": [],
                "unrelated_remote_paths_modified": False,
            },
            "submission_performed": False,
        }
    gate = load_json(gate_path)
    if gate.get("schema") != "paper_i_hh_sr_symcost_noprune_remote_execution_gate_v1":
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
        "schema": "paper_i_hh_sr_symcost_noprune_remote_preflight_receipt_v1",
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


def preflight_readiness(
    checks: dict[str, bool],
) -> tuple[bool, tuple[str, ...]]:
    """Return readiness only when every named submission check is true."""

    non_boolean = sorted(
        name for name, value in checks.items() if not isinstance(value, bool)
    )
    if non_boolean:
        raise TypeError(f"preflight checks must be boolean: {non_boolean}")
    failed = tuple(sorted(name for name, value in checks.items() if not value))
    return not failed, failed


def main() -> int:
    if not SOURCE_FREEZE_COMPLETE:
        raise SystemExit(
            "source freeze incomplete: replace every TODO source hash, set "
            "SOURCE_FREEZE_COMPLETE=True only after final tests, and keep "
            "SUBMISSION_ENABLED=False until archive/remote gates pass"
        )
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
            "schema": (
                "paper_i_hh_sr_appendix_historical_beam3x2_all_six_r50_"
                "normalized_manifest_v1"
            ),
            "bundle_id": BUNDLE_ID,
            "regime_slug": slug,
            "route_identity": job["route_identity"],
            "physics": job["physics"],
            "segment": job["segment"],
            "command_argv": job["command"]["argv"],
            "environment": job["environment"],
            "cache_policy": job["cache_policy"],
            "evidence_requirements": job["evidence_requirements"],
            "source_lock": job["source_lock"],
            "resource_request": job["resource_request"],
        })
        job_paths.append(job_path)
        if slug in SUBMISSION_REGIMES:
            queue_lines.append("\t".join((
                slug,
                job_path.relative_to(REPO).as_posix(),
                normalized_path.relative_to(REPO).as_posix(),
                str(row["memory_mb"]), str(row["disk_mb"]),
            )))
    if (
        len(queue_lines) != 6
        or {line.split("\t", 1)[0] for line in queue_lines}
        != set(SUBMISSION_REGIMES)
    ):
        raise ValueError(
            "appendix historical-beam bundle must queue exactly all six locked regimes"
        )
    settings_audit = validate_candidate_bundle_contract(
        contract=contract,
        jobs=job_payloads,
    )
    dump_json(BUNDLE_DIR / "scientific_settings_audit.json", settings_audit)
    (BUNDLE_DIR / "queue.tsv").write_text("\n".join(queue_lines) + "\n")
    (BUNDLE_DIR / "submit.sub").write_text(
        submit_text(str(archive["archive_sha256"])), encoding="utf-8"
    )

    isolated = archive_only_preflight(archive=archive, job_paths=job_paths)
    dump_json(BUNDLE_DIR / "archive_only_preflight.json", isolated)
    if isolated.get("status") != "pass":
        raise ValueError("archive-only import/parse/helper/regression preflight failed")
    parse_rows = list(isolated["six_validate_only_parses"])

    image_local = REPO / REMOTE_IMAGE_PATH
    image_local_present = image_local.is_file()
    image_local_match = image_local_present and sha256(image_local) == REMOTE_IMAGE_SHA256
    qiskit_helper = "pipelines/exact_bench/paper_i_hh_recovery_prefix_qiskit_sidecar.py"
    qiskit_helper_archived = qiskit_helper in archive["files"]
    expected_horizons = {
        str(row["slug"]): int(row["target_round"]) for row in REGIMES
    }
    parity = {
        "schema": "paper_i_hh_sr_appendix_historical_beam3x2_route_parity_v1",
        "status": "pass",
        "profile_request": PROFILE_REQUEST,
        "profile_resolved": PROFILE_RESOLVED,
        "profile_contract_sha256": PROFILE_CONTRACT_SHA256,
        "all_six_commands_parse": all(row["returncode"] == 0 for row in parse_rows),
        "per_regime_horizons": expected_horizons,
        "all_horizons_explicit_in_argv": all(
            "--adapt-max-depth" in job["command"]["argv"]
            and job["command"]["argv"][
                job["command"]["argv"].index("--adapt-max-depth") + 1
            ] == str(expected_horizons[str(job["regime_slug"])])
            for job in job_payloads
        ),
        "no_profile_method_flag_repetition": True,
        "same_cutoff_lock_pass": True,
        "phase1_energy_model": PHASE1_ENERGY_MODEL,
        "phase2_curvature_policy": PHASE2_CURVATURE_POLICY,
        "phase2_cheap_curvature_proxy_policy": (
            PHASE2_CHEAP_CURVATURE_PROXY_POLICY
        ),
        "lambda_f_proxy_flags_forbidden": True,
        "pruning_active": False,
        "beam_shape": "historical_3x2_v1",
        "beam_parent_stop_terminal_also_materialized": True,
        "beam_structural_mode": "stop_or_single_admission",
        "beam_terminal_archive_accumulation": "cumulative_across_rounds",
        "beam_terminal_archive_cap": 3,
        "phase2_ordinary_novelty_multiplier_active": False,
        "phase3_ordinary_novelty_multiplier_active": False,
        "infeasible_model_novelty_fallback_active": True,
        "infeasible_model_novelty_fallback_telemetry_required": True,
        "scientific_settings_audit": settings_audit,
        "worker_source_mode": archive["worker_source_mode"],
        "non_scientific_archive_overlays": archive[
            "non_scientific_archive_overlays"
        ],
        "parse_rows": parse_rows,
    }
    dump_json(BUNDLE_DIR / "route_parity.json", parity)
    bundle_manifest = {
        "schema": "paper_i_hh_sr_appendix_historical_beam3x2_bundle_v1",
        "bundle_id": BUNDLE_ID,
        "batch_name": BATCH_NAME,
        "created_utc": utc_now(),
        "run_class": "appendix_historical_beam_ablation_matrix",
        "parent_stage": "frozen_main_route_exact_one_factor_beam_variant_v1",
        "submission_scope": {
            "regimes": sorted(SUBMISSION_REGIMES),
            "job_count": len(queue_lines),
            "fresh_jobs": True,
            "per_regime_horizons": expected_horizons,
            "scientific_contract": (
                "full_response_symmetric_cost_no_prune_historical_beam3x2_"
                "no_ordinary_novelty_v1"
            ),
        },
        "source_revision": revision,
        "source_archive": archive,
        "route_parity": parity,
        "scientific_settings_audit": settings_audit,
        "archive_only_preflight": (
            BUNDLE_DIR.relative_to(REPO) / "archive_only_preflight.json"
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
        "submission_jobs": [
            path.relative_to(REPO).as_posix()
            for path in job_paths
            if path.stem in SUBMISSION_REGIMES
        ],
        "source_lock_state": SOURCE_LOCK_STATE,
        "submission_status": "pending_preflight_evaluation",
    }
    remote_pass = remote_receipt.get("status") == "pass"
    preflight_checks = {
            "base_git_ancestry_verified": True,
            "immutable_parent_derived_archive_is_executable_authority": True,
            "critical_source_hashes": True,
            "source_archive_safe_and_closed": True,
            "immutable_parent_plus_hash_locked_overlay_inventory": (
                archive["worker_source_mode"] == SOURCE_LOCK_STATE
                and archive["immutable_parent_archive"]["sha256"]
                == PARENT_ARCHIVE_SHA256
                and archive["historical_beam_overlay"]["overlay_sha256"]
                == BEAM_OVERLAY_SHA256
                and archive["historical_beam_overlay"]["derived_file_sha256"]
                == BEAM_OVERLAY_FILE_HASHES
                and set(archive["non_scientific_archive_overlays"])
                == set(NONSCIENTIFIC_ARCHIVE_OVERLAYS)
            ),
            "historical_beam_overlay_result_hashes_closed": all(
                archive["files"].get(relative, {}).get("sha256") == digest
                for relative, digest in BEAM_OVERLAY_FILE_HASHES.items()
            ),
            "compatibility_overlay_archive_hashes_closed": all(
                archive["files"].get(relative, {}).get("sha256") == digest
                for relative, digest in NONSCIENTIFIC_ARCHIVE_OVERLAYS.items()
            ),
            "required_untracked_source_modules_hash_closed": all(
                archive["files"].get(relative, {}).get("sha256") == digest
                for relative, digest in REQUIRED_UNTRACKED_SOURCE_MODULES.items()
            ),
            "required_hash_locked_fidelity_files_hash_closed": all(
                archive["files"].get(relative, {}).get("sha256") == digest
                for relative, digest in REQUIRED_HASH_LOCKED_FIDELITY_FILES.items()
            ),
            "required_untracked_fidelity_overlays_hash_closed": all(
                archive["files"].get(relative, {}).get("sha256")
                == record.get("sha256")
                and record.get("tracked_in_parent_archive") is False
                for relative, record in archive[
                    "required_untracked_hash_overlays"
                ].items()
            ),
            "six_job_manifests": len(job_paths) == 6,
            "all_six_fresh_round50_submission_queue": (
                len(queue_lines) == 6
                and {line.split("\t", 1)[0] for line in queue_lines}
                == set(SUBMISSION_REGIMES)
            ),
            "all_job_validations": all(row["returncode"] == 0 for row in parse_rows),
            "same_cutoff_reference_lock": True,
            "candidate_profile_and_digest": True,
            "fresh_round0_exact_per_regime_horizon": True,
            "adapt_max_depth_explicit_and_source_locked": True,
            "no_repeated_method_flags": True,
            "scientific_settings_audit_passed": (
                settings_audit["status"] == "pass"
                and not settings_audit["unexpected_executable_differences"]
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
            "submission_enabled": SUBMISSION_ENABLED,
            "terminal_qiskit_sidecar_required": True,
            "qiskit_sidecar_helper_in_source_archive": qiskit_helper_archived,
            "qiskit_backend_availability_remote_check": remote_pass,
            "remote_image_sha256_rechecked": remote_pass,
            "remote_qiskit_import_passed": remote_pass,
            "remote_fake_marrakesh_instantiation_passed": remote_pass,
            "phase3_response_supported_rank_recorded": True,
            "shadow_damping_scientific_application_disabled": True,
            "pruning_disabled": True,
            "historical_beam_exact_3x2_enabled": True,
            "historical_beam_parent_stop_terminal_materialized": True,
            "historical_beam_terminal_archive_cumulative_cap3": True,
            "ordinary_phase2_phase3_novelty_multipliers_disabled": True,
            "infeasible_model_novelty_fallback_enabled": True,
            "infeasible_model_novelty_fallback_telemetry_required": True,
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
    }
    submission_ready, failed_preflight_checks = preflight_readiness(
        preflight_checks
    )
    submission_blockers = [
        f"preflight_check_failed:{name}" for name in failed_preflight_checks
    ]
    bundle_manifest["submission_status"] = (
        "submission_ready_not_yet_submitted"
        if submission_ready
        else "submission_blocked_failed_preflight_checks"
    )
    bundle_manifest["failed_preflight_checks"] = list(failed_preflight_checks)
    dump_json(BUNDLE_DIR / "bundle_manifest.json", bundle_manifest)
    preflight = {
        "schema": "paper_i_hh_sr_appendix_historical_beam3x2_preflight_v1",
        "created_utc": utc_now(),
        "status": (
            "pass_submission_ready_not_yet_submitted"
            if submission_ready
            else "blocked_failed_preflight_checks"
        ),
        "checks": preflight_checks,
        "remote_image": {
            "path": REMOTE_IMAGE_PATH.as_posix(),
            "verified_remote_sha256": REMOTE_IMAGE_SHA256,
            "qiskit_version": REMOTE_QISKIT_VERSION,
            "fake_backend_resolved": REMOTE_FAKE_BACKEND_RESOLVED,
            "fake_backend_qubits": REMOTE_FAKE_BACKEND_QUBITS,
            "local_copy_present": image_local_present,
            "remote_recheck_passed": remote_pass,
        },
        "diagnostics": {
            "local_image_present": image_local_present,
            "local_image_hash_matches_prior_remote_digest": image_local_match,
            "local_image_fields_affect_submission_readiness": False,
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
            "scientific_settings_audit.json",
            "archive_only_preflight.json",
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
        "schema": "paper_i_hh_sr_symcost_noprune_submission_artifact_hashes_v1",
        "artifacts": inventory,
    })
    print(json.dumps({
        "status": preflight["status"],
        "bundle": BUNDLE_DIR.relative_to(REPO).as_posix(),
        "source_archive_sha256": archive["archive_sha256"],
        "jobs": len(job_paths),
        "queued_jobs": len(queue_lines),
        "scientific_blockers": preflight["scientific_blockers"],
        "submission_blockers": preflight["submission_blockers"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
