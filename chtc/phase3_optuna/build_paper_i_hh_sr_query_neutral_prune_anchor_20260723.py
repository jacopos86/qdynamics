#!/usr/bin/env python3
"""Build the immutable first-hit anchor for query-neutral Paper-I pruning."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "chtc/phase3_optuna/input"
HELPER_PATH = (
    ROOT / "chtc/phase3_optuna/build_paper_i_hh_sr_material_window_anchor_20260721.py"
)
_SPEC = importlib.util.spec_from_file_location("material_anchor_helper", HELPER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot load material-window anchor helpers")
HELPER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(HELPER)

BASE_ID = (
    "paper_i_hh_sr_snake_no_overlap_trust_material_window_parent_anchor_"
    "weak_weak_r50_20260721_v4_chtc"
)
BASE_BATCH = "paper-i-hh-sr-material-window-parent-anchor-ww-r50-20260721-v4"
BASE = INPUT / BASE_ID
ANCHOR_ID = (
    "paper_i_hh_sr_snake_no_overlap_trust_full_geometry_query_neutral_prune_"
    "parent_anchor_weak_weak_target_20260723_v3_chtc"
)
ANCHOR_BATCH = "paper-i-hh-sr-query-neutral-prune-parent-anchor-ww-target-20260723-v3"
ANCHOR = INPUT / ANCHOR_ID

BASE_SOURCE_SHA256 = (
    "ced6b10d6bfbe4ae6a54495ff2ef4747a90036fa2027b0386555d016d5869a05"
)
EXPECTED_SOURCE_SHA256 = (
    "5747f73be5b6f4a050c5c33c12c87099db7e2edb57ca2eb9a41d9e4a783207e4"
)
PARENT_ALIAS = "sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1"
PARENT_ROUTE = (
    "supported_projected_generalized_source_metric_no_overlap_trust_"
    "full_response_symmetric_cost_no_prune_v1"
)
PARENT_DIGEST = "fd5ec3fa2c98b2a9d1cbcc304241d723f57dbd6210f4ea2daf30753603a146c2"
CANDIDATE_ALIAS = (
    "sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_full_geometry_"
    "query_neutral_prune_v1"
)
CANDIDATE_ROUTE = (
    "supported_projected_generalized_source_metric_no_overlap_trust_"
    "full_response_symmetric_cost_query_neutral_fs_prune_v1"
)
CANDIDATE_DIGEST = (
    "326ae05091b24fcb580d33f86f25add4c1252bcdd64316b82ae14c14c6bb3372"
)
IMAGE_SHA256 = "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
TARGET_ABS_DELTA_E = 2.0e-4
SOURCE_TRANSFER = ROOT / (
    "raw_outputs/chtc_fetch_paper_i_hh_sr_20260721/priority_20260721T0037Z/"
    "8958273.0__weak_weak_transfer.tar.gz"
)
SOURCE_RESULT_MEMBER = (
    "raw_outputs/paper_i_hh_sr_snake_no_overlap_trust_all_six_r50_"
    "20260720_v4_chtc/weak_weak/json/result.json"
)
RECOVERED_PARENT_VALIDATOR = ROOT / (
    "raw_outputs/chtc_fetch_paper_i_hh_sr_20260721/"
    "no_overlap_8958273_weak_cutoff_reporting_recovery/source/"
    "chtc/phase3_optuna/input/"
    "paper_i_hh_sr_snake_no_overlap_trust_all_six_r50_20260720_v4_chtc/"
    "evidence_validation.py"
)
QUERY_VALIDATOR = (
    ROOT / "chtc/phase3_optuna/query_neutral_prune_evidence_validation.py"
)

ADAPT_HUNKS = (
    (863, "98c7135744599a841772262da213a8945b8964726150b74514b62a9b2b31e79d"),
    (871, "6210b8ec24d87bb412a9aea4ca9a0061dfbcd80588a5b375852d84a722f0bee0"),
    (3511, "c236329aba17e0b3ebed53939cb7817d216f1d52672f0adf69de8d421c4e77d1"),
    (6532, "4d0ad03b4412210c112b90367303fbef9511d1103e98e8344beaaae8a51fcb5a"),
    (6569, "ac07229dcc97592a1c98992ccb265d1955a095a995c5f842767f942f44b2c606"),
    (10225, "d263133a71cee397b3e57798e2be37bfee96a346be6e446549098090ab9d0240"),
    (10324, "e0f907452e64abd162c361c325f055dc33862266907f4c5ab07682e0fcb0195a"),
    (10354, "cee7f96649f85d859bfbc3510b636edc449c3b7f3aed9cebc52d6664126bf741"),
    (10413, "b5168f39ee6511f241222fb3178751fe49889fd8fc35d423ba4c99fe6d25d6fe"),
    (16996, "3fd8850154943b3b4db27d485e4fd8fdf4042d314f46bdf9f516911f394c4211"),
    (45807, "254acbf8bcfa34412564ea58d6ec44960e6d545b0cf0541fb3a36efa7e5f2451"),
    (46507, "91679df226699f8397aded97fdfaec9cef4d8353ab7a9aa437ea5976df934ede"),
    (48031, "ce681ebedc58eb1a7bd4296280e67f7ba85ecc60fb138a91367887c770588e54"),
    (48329, "1e7d8760178696b349608c7e31b8ea71f46249148cd838c6205c8dfe684430cf"),
    (48350, "93b0ec8f6c87d061709ec076d34e322cdd7a18db97671cfaa658be50336bf4a1"),
    (49718, "be481fff5d23f2111d5051e973822d3dac4c51f5856a13e88a3c761573a7fa2f"),
)
ROUTE_HUNKS = (
    (80, "3edf9fabccbfda420f6fad3986c2f0c34212f10f60b5111de70fcb6da4c61739"),
    (134, "8e7250b696466bcd8ebd5557173e604ff6c3b25512a77e0a740c7820c98195d3"),
    (200, "4dbebe713070f9d96e1f67ab43604082c4ef20a8d3555756fe085b9d63435af4"),
    (672, "8b93dc9bbe43f7002806245feee5b74721233e82785f13c7024e6d01d9184103"),
    (1612, "ff6f6a38626e5545e1f27cb70aa80346a9f2f70469d675244cb037621e7a55d5"),
    (2203, "73e70c411b31ad6221ea876e26cb4fc68d4ea5924850593cd73c715435401cef"),
    (2283, "954b8a2524029877eb63f2b53e4e80fc80c48f6289dfecb4c0aea3f8bc9aa228"),
    (2386, "2a6089fb40fc8960544247fc444e6c8cd0ac7a183cd9e4a0f7db6f74a0575b7a"),
    (2488, "2f776fdd7d125e09063d21ac7e045326caa1e1feb42f597e23cfc76bba20be82"),
    (2556, "7f6195d5fb41784c479b0a4abb0e90213819b0c34cb85156f6196a00e46a0319"),
    (2593, "b69d9c7f0dab31061495805a8b6d4ec5bc4af681b81a9c09c9773e5879661f93"),
    (2737, "cd3483e8805c851cbff70931822bbde84ae1a8edf492764d66300cd7f9132cce"),
    (2764, "f48c20a6ebde0de09167b070913f58399f652979e9cfab44718c7449d445abdc"),
    (2798, "b9f3808999289173f5b10504d5c85cf5c01f502de0b740b22973d8845bb9c632"),
    (2831, "5c00b045a94bfc6c8ca788890154b0e5e47340f44137745f225217a543c00410"),
    (2874, "1e5fc06d7bd6b4788398d4aea57d022df8901c4a93ccd9e85725a02665fba931"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _selected_hunks(
    *,
    base_text: str,
    live_text: str,
    allowlist: tuple[tuple[int, str], ...],
) -> list[dict[str, Any]]:
    all_hunks = HELPER.unified_hunks(base_text, live_text)
    selected = [
        hunk
        for hunk in all_hunks
        if any(
            marker in "".join(hunk["lines"])
            for marker in (
                "query_neutral",
                "QUERY_NEUTRAL",
                "FULL_GEOMETRY_QUERY_NEUTRAL",
            )
        )
    ]
    actual = tuple((int(h["old_start"]), str(h["sha256"])) for h in selected)
    if actual != allowlist:
        raise ValueError(f"reviewed query-neutral hunk allowlist drift: {actual!r}")
    return selected


def build_source_archive(temp: Path) -> tuple[Path, dict[str, Any]]:
    base_archive = BASE / "source_locked.tar.gz"
    if _sha256(base_archive) != BASE_SOURCE_SHA256:
        raise ValueError("clean reproduced parent archive drift")
    source = temp / "source"
    HELPER.extract_archive(base_archive, source)
    overlays: dict[str, Any] = {}
    for relative, allowlist in (
        ("pipelines/static_adapt/adapt_pipeline.py", ADAPT_HUNKS),
        ("pipelines/static_adapt/sr_snake_route_profile.py", ROUTE_HUNKS),
    ):
        target = source / relative
        parent_text = target.read_text(encoding="utf-8")
        live_text = (ROOT / relative).read_text(encoding="utf-8")
        selected = _selected_hunks(
            base_text=parent_text,
            live_text=live_text,
            allowlist=allowlist,
        )
        reconstructed = HELPER.apply_unified_hunks(parent_text, selected)
        target.write_text(reconstructed, encoding="utf-8")
        overlays[relative] = {
            "parent_sha256": HELPER.sha256_bytes(parent_text.encode()),
            "overlay_sha256": _sha256(target),
            "reviewed_hunks": [
                {
                    "old_start": int(h["old_start"]),
                    "sha256": str(h["sha256"]),
                }
                for h in selected
            ],
        }
    for relative in (
        "pipelines/static_adapt/query_neutral_full_geometry_prune.py",
        "test/test_static_adapt_query_neutral_full_geometry_prune.py",
    ):
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
        overlays[relative] = {
            "parent_sha256": None,
            "overlay_sha256": _sha256(destination),
            "reviewed_hunks": [],
        }
    stale_test = source / "test/test_static_adapt_phase3_material_window_route_profile.py"
    if stale_test.exists():
        stale_test.unlink()
    HELPER.strip_bytecode(source)
    output = temp / "source_locked.tar.gz"
    HELPER.deterministic_archive(source, output)
    if _sha256(output) != EXPECTED_SOURCE_SHA256:
        raise ValueError("deterministic query-neutral source archive drift")
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "test/test_static_adapt_query_neutral_full_geometry_prune.py",
            "test/test_static_adapt_sr_route_profile.py",
        ],
        cwd=source,
        env=env,
        check=True,
    )
    command = (
        "from pipelines.static_adapt.sr_snake_route_profile import "
        "canonical_sr_snake_contract_sha256 as h;"
        f"assert h({PARENT_ALIAS!r})=={PARENT_DIGEST!r};"
        f"assert h({CANDIDATE_ALIAS!r})=={CANDIDATE_DIGEST!r};"
        "print('route-contracts-pass')"
    )
    subprocess.run(
        [sys.executable, "-c", command],
        cwd=source,
        env=env,
        check=True,
    )
    return output, overlays


def _patch_run_job(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    argv_anchor = (
        '        "--adapt-current-json-every-depth", "1",\n'
        '        "--adapt-current-json", str(paths["current_json"]),\n'
    )
    argv_replacement = (
        '        "--adapt-current-json-every-depth", "1",\n'
        '        "--adapt-benchmark-target-abs-delta-e", "0.0002",\n'
        '        "--adapt-current-json", str(paths["current_json"]),\n'
    )
    if text.count(argv_anchor) != 1:
        raise ValueError("run-job expected-argv anchor drift")
    text = text.replace(argv_anchor, argv_replacement, 1)
    import_anchor = (
        "from evidence_validation import (\n"
        "    checkpoint_sha256,\n"
        "    validate_parent_evidence,\n"
        "    validate_projected_generalized_phase3_evidence,\n"
        "    validate_no_overlap_trust_evidence,\n"
        ")\n"
    )
    import_replacement = (
        "from evidence_validation import checkpoint_sha256\n"
        "from query_neutral_prune_evidence_validation import (\n"
        "    validate_parent_first_hit_evidence,\n"
        ")\n"
    )
    if text.count(import_anchor) != 1:
        raise ValueError("run-job validation import anchor drift")
    text = text.replace(import_anchor, import_replacement, 1)
    validation_anchor = (
        '    target_round = int(manifest["segment"]["target_controller_round"])\n'
        '    target_admissions = int(manifest["segment"]["max_new_admissions"])\n'
        "    evidence = validate_parent_evidence(\n"
        "        result=result,\n"
        "        current=current,\n"
        "        ledger_sidecar=ledger,\n"
        "        profile=PROFILE,\n"
        "        digest=DIGEST,\n"
        "        target_round=target_round,\n"
        "        target_new_admissions=target_admissions,\n"
        "        require_supported_rank=True,\n"
        "    )\n"
        "    projected_evidence = validate_projected_generalized_phase3_evidence(\n"
        "        result=result, target_round=target_round\n"
        "    )\n"
        "    no_overlap_evidence = validate_no_overlap_trust_evidence(\n"
        "        result=result, target_round=target_round\n"
        "    )\n"
    )
    validation_replacement = (
        '    target_round = int(result["adapt_segment"]["final_controller_round"])\n'
        "    evidence = validate_parent_first_hit_evidence(\n"
        "        result=result,\n"
        "        current=current,\n"
        "        ledger_sidecar=ledger,\n"
        "        safety_cap=int(manifest[\"segment\"][\"target_controller_round\"]),\n"
        "    )\n"
        '    projected_evidence = evidence["projected_phase3"]\n'
        '    no_overlap_evidence = evidence["no_overlap_trust"]\n'
    )
    if text.count(validation_anchor) != 1:
        raise ValueError("run-job parent validation block drift")
    text = text.replace(validation_anchor, validation_replacement, 1)
    path.write_text(text, encoding="utf-8")


def build_anchor() -> dict[str, Any]:
    if ANCHOR.exists():
        raise FileExistsError(f"immutable anchor exists: {ANCHOR}")
    for required in (
        BASE,
        SOURCE_TRANSFER,
        RECOVERED_PARENT_VALIDATOR,
        QUERY_VALIDATOR,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    with tempfile.TemporaryDirectory(prefix="query-neutral-anchor-") as raw:
        archive, overlays = build_source_archive(Path(raw))
        shutil.copytree(
            BASE,
            ANCHOR,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        HELPER.clean_inherited_bundle_state(ANCHOR)
        shutil.copy2(archive, ANCHOR / "source_locked.tar.gz")
    old_archive_sha = _sha256(BASE / "source_locked.tar.gz")
    replacements = {
        BASE_ID: ANCHOR_ID,
        BASE_BATCH: ANCHOR_BATCH,
        old_archive_sha: EXPECTED_SOURCE_SHA256,
    }
    for relative in (
        "run_job.py",
        "validate_fetched.py",
        "execute_source_locked_job.sh",
    ):
        HELPER.patch_text(ANCHOR / relative, replacements)

    with tempfile.TemporaryDirectory(prefix="query-neutral-inventory-source-") as raw:
        source = Path(raw)
        HELPER.extract_archive(ANCHOR / "source_locked.tar.gz", source)
        files = HELPER.inventory(source)
    overlay_receipt = {
        "schema": "paper_i_query_neutral_prune_exact_hunk_overlay_v1",
        "parent_source_archive_sha256": BASE_SOURCE_SHA256,
        "source_archive_sha256": EXPECTED_SOURCE_SHA256,
        "parent_route_contract_sha256": PARENT_DIGEST,
        "candidate_route_contract_sha256": CANDIDATE_DIGEST,
        "overlay_files": overlays,
        "dormant_material_prune_plumbing_inherited": True,
        "dormant_material_prune_route_executed": False,
    }
    manifest = HELPER.replace_tree(
        _load(BASE / "source_archive_manifest.json"),
        replacements,
    )
    manifest.update(
        {
            "archive": (
                f"chtc/phase3_optuna/input/{ANCHOR_ID}/source_locked.tar.gz"
            ),
            "archive_sha256": EXPECTED_SOURCE_SHA256,
            "archive_size_bytes": (ANCHOR / "source_locked.tar.gz").stat().st_size,
            "file_count": len(files),
            "files": files,
            "query_neutral_prune_source_overlay": overlay_receipt,
        }
    )
    _dump(ANCHOR / "source_archive_manifest.json", manifest)
    manifest_sha = _sha256(ANCHOR / "source_archive_manifest.json")
    revision = HELPER.replace_tree(
        _load(BASE / "source_revision_manifest.json"),
        replacements,
    )
    revision["query_neutral_prune_source_overlay"] = overlay_receipt
    _dump(ANCHOR / "source_revision_manifest.json", revision)
    revision_sha = _sha256(ANCHOR / "source_revision_manifest.json")
    physics = HELPER.replace_tree(
        _load(BASE / "physics_and_exact_reference_lock.json"),
        replacements,
    )
    _dump(ANCHOR / "physics_and_exact_reference_lock.json", physics)
    physics_sha = _sha256(ANCHOR / "physics_and_exact_reference_lock.json")

    job = HELPER.replace_tree(
        copy.deepcopy(_load(BASE / "jobs/weak_weak.json")),
        replacements,
    )
    job["bundle_id"] = ANCHOR_ID
    job["batch_name"] = ANCHOR_BATCH
    command = list(job["command"]["argv"])
    if "--adapt-benchmark-target-abs-delta-e" in command:
        raise ValueError("base anchor unexpectedly has an early-stop override")
    insertion = command.index("--adapt-current-json")
    command[insertion:insertion] = [
        "--adapt-benchmark-target-abs-delta-e",
        str(TARGET_ABS_DELTA_E),
    ]
    job["command"]["argv"] = command
    job["command"]["explicit_method_overrides"] = sorted(
        set(job["command"]["explicit_method_overrides"])
        | {"adapt_benchmark_target_abs_delta_e"}
    )
    job["source_lock"].update(
        {
            "source_archive": (
                f"chtc/phase3_optuna/input/{ANCHOR_ID}/source_locked.tar.gz"
            ),
            "source_archive_sha256": EXPECTED_SOURCE_SHA256,
            "source_archive_manifest": (
                f"chtc/phase3_optuna/input/{ANCHOR_ID}/source_archive_manifest.json"
            ),
            "source_archive_manifest_sha256": manifest_sha,
            "source_revision_manifest": (
                f"chtc/phase3_optuna/input/{ANCHOR_ID}/source_revision_manifest.json"
            ),
            "source_revision_manifest_sha256": revision_sha,
            "physics_reference_lock": (
                f"chtc/phase3_optuna/input/{ANCHOR_ID}/physics_and_exact_reference_lock.json"
            ),
            "physics_reference_lock_sha256": physics_sha,
            "query_neutral_prune_source_overlay": overlay_receipt,
        }
    )
    job["source_value_anchor"] = {
        "schema": "source_locked_sensitivity_anchor_plan_v1",
        "source_transfer_archive": str(SOURCE_TRANSFER.relative_to(ROOT)),
        "source_transfer_archive_sha256": _sha256(SOURCE_TRANSFER),
        "source_result_archive_member": SOURCE_RESULT_MEMBER,
        "source_route_profile": PARENT_ROUTE,
        "source_route_contract_sha256": PARENT_DIGEST,
        "candidate_route_profile": CANDIDATE_ROUTE,
        "candidate_route_contract_sha256": CANDIDATE_DIGEST,
        "source_prefix_comparison_scope": "through_source_first_target_hit_v1",
        "source_first_hit_round": 21,
        "source_first_hit_energy": -0.9183544101009906,
        "source_first_hit_abs_error": 2.650989383079505e-5,
        "benchmark_target_abs_delta_e": TARGET_ABS_DELTA_E,
        "candidate_not_executed": True,
        "fanout_allowed_before_anchor_pass": False,
    }
    _dump(ANCHOR / "jobs/weak_weak.json", job)
    normalized = HELPER.replace_tree(
        _load(BASE / "normalized_manifests/weak_weak.json"),
        replacements,
    )
    normalized.update(
        {
            "bundle_id": ANCHOR_ID,
            "batch_name": ANCHOR_BATCH,
            "command_argv": list(job["command"]["argv"]),
            "source_lock": copy.deepcopy(job["source_lock"]),
            "source_value_anchor": copy.deepcopy(job["source_value_anchor"]),
        }
    )
    _dump(ANCHOR / "normalized_manifests/weak_weak.json", normalized)
    for folder in (ANCHOR / "jobs", ANCHOR / "normalized_manifests"):
        for path in folder.glob("*.json"):
            if path.name != "weak_weak.json":
                path.unlink()

    shutil.copy2(
        RECOVERED_PARENT_VALIDATOR,
        ANCHOR / "evidence_validation.py",
    )
    shutil.copy2(
        RECOVERED_PARENT_VALIDATOR,
        ANCHOR / "evidence_validation_parent.py",
    )
    shutil.copy2(
        QUERY_VALIDATOR,
        ANCHOR / "query_neutral_prune_evidence_validation.py",
    )
    HELPER.patch_text(ANCHOR / "evidence_validation.py", replacements)
    HELPER.patch_text(ANCHOR / "evidence_validation_parent.py", replacements)
    _patch_run_job(ANCHOR / "run_job.py")
    queue_rel = f"chtc/phase3_optuna/input/{ANCHOR_ID}/queue.tsv"
    (ANCHOR / "queue.tsv").write_text(
        f"weak_weak\tchtc/phase3_optuna/input/{ANCHOR_ID}/jobs/weak_weak.json\t"
        f"chtc/phase3_optuna/input/{ANCHOR_ID}/normalized_manifests/weak_weak.json"
        "\t40960\t61440\n",
        encoding="utf-8",
    )
    (ANCHOR / "submit.sub").write_text(
        HELPER.submit_text(
            ANCHOR_ID,
            ANCHOR_BATCH,
            EXPECTED_SOURCE_SHA256,
            queue_rel,
        ),
        encoding="utf-8",
    )
    audit = {
        "schema": "source_locked_sensitivity_audit_v1",
        "status": "anchor_pending",
        "source": {
            "method": "Paper-I no-overlap SR-SNAKE full geometry",
            "regime_or_case": "weak_weak",
            "route_or_profile_id": PARENT_ROUTE,
            "route_contract_sha256": PARENT_DIGEST,
            "source_transfer_archive": str(SOURCE_TRANSFER.relative_to(ROOT)),
            "source_transfer_archive_sha256": _sha256(SOURCE_TRANSFER),
            "source_result_archive_member": SOURCE_RESULT_MEMBER,
            "first_hit_round": 21,
            "first_hit_energy": -0.9183544101009906,
        },
        "sweep": {
            "variable": "phase1_prune_enabled",
            "grid": [False, True],
            "candidate_route": CANDIDATE_ROUTE,
            "candidate_route_contract_sha256": CANDIDATE_DIGEST,
            "explicit_user_approved_operational_change": (
                "stop_at_first_E_T_2e-4_hit_with_50_round_safety_cap"
            ),
        },
        "anchor": {
            "anchor_result_json": None,
            "anchor_reproduces_source_prefix_through_first_hit": False,
            "operator_sequence_match": None,
            "controller_energy_history_exact_match": None,
            "checkpoint_sequence_match": None,
            "ledger_prefix_match": None,
        },
        "fanout_authorized": False,
    }
    _dump(ANCHOR / "source_locked_sensitivity_audit.json", audit)
    receipt = {
        "schema": "paper_i_query_neutral_prune_parent_anchor_bundle_v1",
        "bundle_id": ANCHOR_ID,
        "batch_name": ANCHOR_BATCH,
        "source_archive_sha256": EXPECTED_SOURCE_SHA256,
        "source_archive_manifest_sha256": manifest_sha,
        "source_revision_manifest_sha256": revision_sha,
        "physics_reference_lock_sha256": physics_sha,
        "parent_route_profile": PARENT_ROUTE,
        "parent_route_contract_sha256": PARENT_DIGEST,
        "candidate_route_profile": CANDIDATE_ROUTE,
        "candidate_route_contract_sha256": CANDIDATE_DIGEST,
        "benchmark_target_abs_delta_e": TARGET_ABS_DELTA_E,
        "max_round_safety_cap": 50,
        "job_count": 1,
        "candidate_not_executed": True,
        "fanout_authorized": False,
        "submission_performed": False,
    }
    for filename in ("anchor_bundle_receipt.json", "bundle_manifest.json"):
        _dump(ANCHOR / filename, receipt)
    _dump(
        ANCHOR / "preflight.json",
        {
            "schema": "paper_i_query_neutral_prune_anchor_preflight_v1",
            "status": "pending",
            "checks": {
                "one_parent_anchor_record": True,
                "candidate_not_executed": True,
                "first_hit_stop_enabled": True,
                "fifty_round_safety_cap": True,
                "archive_only_worker_validation": False,
                "archive_focused_tests": False,
            },
        },
    )
    _dump(
        ANCHOR / "remote_execution_gate.json",
        {
            "schema": "paper_i_query_neutral_prune_anchor_remote_gate_v1",
            "status": "pending_authenticated_remote_preflight",
            "image_sha256": IMAGE_SHA256,
            "source_archive_sha256": EXPECTED_SOURCE_SHA256,
            "submission_performed": False,
        },
    )
    (ANCHOR / "README.md").write_text(
        "# Query-neutral prune source-value anchor\n\n"
        "One weak--weak parent replay, stopped at the first Paper-I target hit. "
        "The prune candidate is present but is not executed.\n",
        encoding="utf-8",
    )
    return receipt


def archive_preflight() -> None:
    with tempfile.TemporaryDirectory(prefix="query-neutral-anchor-preflight-") as raw:
        root = Path(raw)
        with tarfile.open(ANCHOR / "source_locked.tar.gz", "r:gz") as archive:
            archive.extractall(root, filter="data")
        target = root / "chtc/phase3_optuna/input" / ANCHOR_ID
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ANCHOR, target)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        subprocess.run(
            [
                sys.executable,
                str(target / "run_job.py"),
                "--validate-only",
                str(target / "jobs/weak_weak.json"),
            ],
            cwd=root,
            env=env,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "test/test_static_adapt_query_neutral_full_geometry_prune.py",
                "test/test_static_adapt_sr_route_profile.py",
            ],
            cwd=root,
            env=env,
            check=True,
        )


def main() -> int:
    receipt = build_anchor()
    archive_preflight()
    preflight = _load(ANCHOR / "preflight.json")
    preflight["status"] = "pass"
    preflight["checks"]["archive_only_worker_validation"] = True
    preflight["checks"]["archive_focused_tests"] = True
    _dump(ANCHOR / "preflight.json", preflight)
    HELPER.strip_bytecode(ANCHOR)
    _dump(
        ANCHOR / "submission_artifact_hashes.json",
        {
            "schema": "paper_i_query_neutral_prune_anchor_artifacts_v1",
            "files": {
                path.relative_to(ANCHOR).as_posix(): {
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(ANCHOR.rglob("*"))
                if path.is_file()
                and path.name != "submission_artifact_hashes.json"
            },
        },
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
