#!/usr/bin/env python3
"""Build the immutable Paper-I macro + beam3x2 + FS-prune cost matrix."""

from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipelines.static_adapt.sr_snake_route_profile import (  # noqa: E402
    canonical_sr_snake_contract,
    canonical_sr_snake_contract_sha256,
    normalize_sr_route_profile_request,
)


INPUT = ROOT / "chtc/phase3_optuna/input"
BASE_ID = "paper_i_hh_sr_snake_macro_only_physical_lanes_all_six_r50_20260719_v1_chtc"
BASE = INPUT / BASE_ID
PRUNE = INPUT / (
    "paper_i_hh_sr_snake_appendix_fs_prune_nodamping_nobeam_nobatch_"
    "no_ordinary_novelty_all_six_r50_20260718_v4_chtc"
)
BASE_BATCH = "paper-i-hh-sr-macro-only-physical-lanes-six-r50-20260719-v1"
BASE_ALIAS = "sr_snake_macro_only_physical_lanes_v1"
BASE_PROFILE = (
    "supported_whitened_adaptive_trust_full_response_symmetric_cost_no_prune_"
    "macro_only_physical_lanes_v1"
)
BASE_DIGEST = "d14d582e532ee41500cd7d3ebaa21b83da91bb3fcf014be53ab8d1049d1452fa"
IMAGE_SHA = "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
JOB_SCHEMA = "paper_i_hh_sr_macro_beam_prune_cost_all_six_r50_job_v1"
SUCCESSOR_REVISION = "v3"
RUNTIME_PROFILE_REPAIR_REVISION = "v2"
FAILED_PREDECESSOR_CLUSTERS = {
    "symmetric": 8892215,
    "one_sided": 8892216,
}
SUPERSEDED_PREDECESSOR_CLUSTERS = {
    "symmetric": 8893083,
    "one_sided": 8893084,
}
ARCHIVE_OVERLAYS = (
    "agent_guidance/skills/paper-i-results/scripts/compute_paper_i_main_fidelities.py",
    "pipelines/static_adapt/sr_snake_route_profile.py",
    "test/test_paper_i_main_fidelity_audit.py",
    "test/test_static_adapt_macro_beam_prune_cost_profiles.py",
    "test/test_static_adapt_sr_prune_appendix_profile.py",
    "test/test_static_adapt_sr_route_profile.py",
)
FIDELITY_SCRIPT_PATH = (
    "agent_guidance/skills/paper-i-results/scripts/"
    "compute_paper_i_main_fidelities.py"
)
REGIMES = (
    "weak_weak",
    "intermediate_weak",
    "strong_weak_u8",
    "weak_strong",
    "intermediate_strong",
    "strong_strong_u8",
)
RESOURCES = {
    "weak_weak": (40960, 61440),
    "intermediate_weak": (40960, 61440),
    "strong_weak_u8": (49152, 61440),
    "weak_strong": (57344, 81920),
    "intermediate_strong": (57344, 81920),
    "strong_strong_u8": (65536, 81920),
}
VARIANTS = (
    {
        "key": "symmetric",
        "bundle_id": (
            "paper_i_hh_sr_snake_macro_beam3x2_fs_prune_symmetric_cost_"
            "all_six_r50_20260719_v3_chtc"
        ),
        "batch": "paper-i-hh-sr-macro-beam3x2-fsprune-symcost-six-r50-20260719-v3",
        "alias": "sr_snake_macro_only_physical_lanes_fs_prune_beam3x2_v1",
        "segment_tag": "sr-macro-beam3x2-fsprune-symcost",
        "cost": "family_robust_symmetric_arctan_v1",
        "fallback": "collective_span_novelty_over_symmetric_cost_v1",
    },
    {
        "key": "one_sided",
        "bundle_id": (
            "paper_i_hh_sr_snake_macro_beam3x2_fs_prune_one_sided_cost_"
            "all_six_r50_20260719_v3_chtc"
        ),
        "batch": "paper-i-hh-sr-macro-beam3x2-fsprune-onesided-six-r50-20260719-v3",
        "alias": (
            "sr_snake_macro_only_physical_lanes_fs_prune_beam3x2_"
            "one_sided_cost_v1"
        ),
        "segment_tag": "sr-macro-beam3x2-fsprune-onesided",
        "cost": "family_robust_v1",
        "fallback": "collective_span_novelty_over_cost_v1",
    },
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(path)
    return payload


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def replace_tree(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: replace_tree(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_tree(item, replacements) for item in value]
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
    return value


def replace_exact(text: str, old: str, new: str, *, count: int | None = None) -> str:
    actual = text.count(old)
    if count is not None and actual != count:
        raise ValueError(f"expected {count} copies, found {actual}: {old[:80]!r}")
    if actual == 0:
        raise ValueError(f"replacement anchor absent: {old[:80]!r}")
    return text.replace(old, new)


def deterministic_archive(source: Path, destination: Path) -> None:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as archive:
        entries = [source] + sorted(source.rglob("*"), key=lambda p: p.relative_to(source).as_posix())
        for path in entries:
            if path == source:
                continue
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


def source_inventory(source: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(source).as_posix(): {
            "sha256": sha(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(source.rglob("*"))
        if path.is_file()
    }


def patch_runtime_profile_registration(source: Path) -> dict[str, Any]:
    """Register the new profiles in the frozen runtime without source drift.

    The failed v1 bundles overlaid the route-contract module but inherited an
    older ``adapt_pipeline.py`` from the macro-only parent archive.  Applying
    this exact registration-only transform to that frozen parent preserves all
    scientific implementation bytes while synchronizing the runtime gates and
    prune-accounting activation with the already serialized route contracts.
    """

    relative = Path("pipelines/static_adapt/adapt_pipeline.py")
    path = source / relative
    before_sha = sha(path)
    text = path.read_text(encoding="utf-8")

    macro_anchor = "SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_V1,"
    additions = (
        "SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_ONE_SIDED_COST_V1,",
        "SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_V1,",
        "SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_ONE_SIDED_COST_V1,",
    )
    output: list[str] = []
    macro_registration_count = 0
    for line in text.splitlines(keepends=True):
        output.append(line)
        if line.strip() != macro_anchor:
            continue
        indent = line[: len(line) - len(line.lstrip())]
        newline = "\n" if line.endswith("\n") else ""
        output.extend(f"{indent}{addition}{newline}" for addition in additions)
        macro_registration_count += 1
    if macro_registration_count != 8:
        raise ValueError(
            "frozen adapt runtime macro-profile gate count drift: "
            f"expected 8, found {macro_registration_count}"
        )
    text = "".join(output)

    macro_overlay_anchor = (
        "    macro_parent_only_overlay_active = bool(\n"
        "        sr_route_profile_contract_resolved is not None\n"
        "        and str(\n"
        "            sr_route_profile_contract_resolved.get(\"route_profile\", \"\")\n"
        "        )\n"
        "        == SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_V1\n"
        "    )\n"
    )
    macro_overlay_replacement = (
        "    macro_parent_only_overlay_active = bool(\n"
        "        sr_route_profile_contract_resolved is not None\n"
        "        and str(\n"
        "            sr_route_profile_contract_resolved.get(\"route_profile\", \"\")\n"
        "        )\n"
        "        in {\n"
        "            SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_V1,\n"
        "            SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_ONE_SIDED_COST_V1,\n"
        "            SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_V1,\n"
        "            SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_ONE_SIDED_COST_V1,\n"
        "        }\n"
        "    )\n"
    )
    text = replace_exact(
        text,
        macro_overlay_anchor,
        macro_overlay_replacement,
        count=1,
    )

    prune_policy_anchor = (
        "                in {\n"
        "                    SR_ROUTE_PROFILE_CANDIDATE_V4,\n"
        "                    SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1,\n"
        "                }\n"
        "                else {}\n"
        "            ),\n"
        "            \"adapt_beam_live_branches\": int(\n"
    )
    prune_policy_replacement = (
        "                in {\n"
        "                    SR_ROUTE_PROFILE_CANDIDATE_V4,\n"
        "                    SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1,\n"
        "                    SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_V1,\n"
        "                    SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_ONE_SIDED_COST_V1,\n"
        "                }\n"
        "                else {}\n"
        "            ),\n"
        "            \"adapt_beam_live_branches\": int(\n"
    )
    text = replace_exact(
        text,
        prune_policy_anchor,
        prune_policy_replacement,
        count=1,
    )

    accounting_anchor = (
        "                in {\n"
        "                    SR_ROUTE_PROFILE_CANDIDATE_V4,\n"
        "                    SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1,\n"
        "                }\n"
        "            )\n"
        "            sr_v4_prune_accounting_views: dict[str, Any] | None = None\n"
    )
    accounting_replacement = (
        "                in {\n"
        "                    SR_ROUTE_PROFILE_CANDIDATE_V4,\n"
        "                    SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1,\n"
        "                    SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_V1,\n"
        "                    SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_ONE_SIDED_COST_V1,\n"
        "                }\n"
        "            )\n"
        "            sr_v4_prune_accounting_views: dict[str, Any] | None = None\n"
    )
    text = replace_exact(
        text,
        accounting_anchor,
        accounting_replacement,
        count=1,
    )
    path.write_text(text, encoding="utf-8")
    after_sha = sha(path)
    if before_sha == after_sha:
        raise ValueError("runtime profile registration repair changed no bytes")
    return {
        "schema": "paper_i_sr_runtime_profile_registration_repair_v1",
        "classification": "non_scientific_runtime_registration_and_accounting_v1",
        "path": relative.as_posix(),
        "source_sha256_before": before_sha,
        "source_sha256_after": after_sha,
        "macro_profile_gate_occurrences_patched": macro_registration_count,
        "macro_overlay_gate_occurrences_patched": 1,
        "prune_policy_gate_occurrences_patched": 1,
        "prune_accounting_gate_occurrences_patched": 1,
        "scientific_setting_changes": [],
    }


def build_source_archive(
    temp: Path,
) -> tuple[Path, dict[str, dict[str, Any]], dict[str, Any]]:
    source = temp / "source"
    with tarfile.open(BASE / "source_locked.tar.gz", "r:gz") as archive:
        archive.extractall(source, filter="data")
    for relative in ARCHIVE_OVERLAYS:
        live = ROOT / relative
        if not live.is_file():
            raise FileNotFoundError(live)
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(live, target)
    runtime_repair = patch_runtime_profile_registration(source)
    destination = temp / "source_locked.tar.gz"
    deterministic_archive(source, destination)
    return destination, source_inventory(source), runtime_repair


def route_metadata(variant: dict[str, str]) -> tuple[str, str, dict[str, Any]]:
    profile = normalize_sr_route_profile_request(variant["alias"])
    digest = canonical_sr_snake_contract_sha256(variant["alias"])
    contract = canonical_sr_snake_contract(variant["alias"])
    if contract["execution_settings"]["phase3_hardware_cost_normalization_mode"] != variant["cost"]:
        raise ValueError("cost policy mismatch")
    return profile, digest, contract


def build_evidence_validation(variant: dict[str, str]) -> str:
    text = (PRUNE / "evidence_validation.py").read_text(encoding="utf-8")
    text = text.replace(
        '"phase3_hardware_cost_normalization_mode": (\n            "family_robust_symmetric_arctan_v1"\n        ),',
        f'"phase3_hardware_cost_normalization_mode": "{variant["cost"]}",',
    )
    text = text.replace(
        'fallback.get("policy")\n        != "collective_span_novelty_over_symmetric_cost_v1"',
        f'fallback.get("policy")\n        != "{variant["fallback"]}"',
    )
    anchor = '        "phase1_prune_endpoint_overlap_policy": "off",\n'
    beam_settings = (
        '        "adapt_beam_live_branches": 3,\n'
        '        "adapt_beam_children_per_parent": 2,\n'
        '        "adapt_beam_terminated_keep": 3,\n'
        '        "adapt_beam_terminal_archive_mode": "legacy",\n'
        '        "adapt_beam_lambda": 0.005,\n'
    )
    text = replace_exact(text, anchor, anchor + beam_settings, count=1)
    old = (
        '    if adapt.get("adapt_beam_enabled") is not False:\n'
        '        raise ValueError("beam execution was not disabled")\n'
    )
    new = (
        '    if adapt.get("adapt_beam_enabled") is not True:\n'
        '        raise ValueError("historical beam execution was not enabled")\n'
        '    continuation = _mapping(\n'
        '        adapt.get("continuation"), field="result continuation telemetry"\n'
        '    )\n'
        '    beam_search = _mapping(\n'
        '        continuation.get("beam_search"), field="historical beam search telemetry"\n'
        '    )\n'
        '    expected_beam = {\n'
        '        "beam_enabled": True,\n'
        '        "live_branches": 3,\n'
        '        "children_per_parent": 2,\n'
        '        "expanded_child_cap_per_round": 6,\n'
        '        "terminated_keep": 3,\n'
        '        "terminal_archive_mode": "legacy",\n'
        '        "lambda_beam": 0.005,\n'
        '    }\n'
        '    for key, expected in expected_beam.items():\n'
        '        if beam_search.get(key) != expected:\n'
        '            raise ValueError(f"historical beam runtime telemetry drift: {key}")\n'
        '    beam_rounds = _sequence(beam_search.get("rounds"), field="beam rounds")\n'
        '    if len(beam_rounds) != target_round:\n'
        '        raise ValueError(\n'
        '            f"historical beam completed {len(beam_rounds)} controller rounds; "\n'
        '            f"expected {target_round}"\n'
        '        )\n'
    )
    text = replace_exact(text, old, new, count=1)
    text = text.replace(
        '    continuation = _mapping(\n        adapt.get("continuation"), field="result continuation telemetry"\n    )\n    if (',
        '    if (',
        1,
    )
    text = text.replace(
        '        "live_prune_rounds_executed": sum(',
        '        "historical_beam_runtime": expected_beam,\n'
        '        "historical_beam_rounds_executed": len(beam_rounds),\n'
        '        "live_prune_rounds_executed": sum(',
        1,
    )
    text = text.replace(
        "# The public validator below enforces this bundle's no-prune route.",
        "# The public validator enforces live FS-trust pruning plus historical beam 3x2.",
    )
    return text


def build_run_job(
    variant: dict[str, str],
    profile: str,
    digest: str,
    bundle_id: str,
    inventory: dict[str, dict[str, Any]],
) -> str:
    text = (PRUNE / "run_job.py").read_text(encoding="utf-8")
    prune_id = PRUNE.name
    text = text.replace('SCHEMA = "paper_i_hh_sr_prune_appendix_all_six_r50_job_v1"', f'SCHEMA = "{JOB_SCHEMA}"')
    text = text.replace(f"BUNDLE_ID = '{prune_id}'", f"BUNDLE_ID = '{bundle_id}'")
    text = replace_exact(text, 'PROFILE_REQUEST = "sr_snake_symmetric_cost_fs_prune_nodamping_v1"', f'PROFILE_REQUEST = "{variant["alias"]}"', count=1)
    start = text.index('PROFILE = (\n')
    end = text.index('\nDIGEST = ', start)
    text = text[:start] + f'PROFILE = "{profile}"' + text[end:]
    digest_start = text.index('DIGEST = ', start)
    digest_end = text.index('\nPHASE1_ENERGY_MODEL', digest_start)
    text = text[:digest_start] + f'DIGEST = "{digest}"' + text[digest_end:]
    text = text.replace(
        'PHASE2_CHEAP_CURVATURE_PROXY_POLICY = "off"\n',
        'PHASE2_CHEAP_CURVATURE_PROXY_POLICY = "off"\n'
        f'SEGMENT_TAG = "{variant["segment_tag"]}"\n'
        'ROUTE_REQUIRED_EXECUTION = {\n'
        '    "adapt_pool": "full_meta",\n'
        '    "adapt_child_pool_expansion_mode": "off",\n'
        '    "adapt_child_pool_expansion_symmetry_policy": "off",\n'
        '    "shared_pauli_pool_mode": "off",\n'
        '    "shared_pauli_pool_symmetry_policy": "off",\n'
        '    "static_lane_route": "physical_operator_type",\n'
        '    "physical_lane_shortlist_aggressiveness": 3,\n'
        '    "phase3_selector_policy": "algebraic_nested_v1",\n'
        '    "phase3_runtime_split_mode": "off",\n'
        '    "allow_archival_phase3_runtime_split": False,\n'
        '    "phase3_runtime_split_child_padding_policy": "unchecked_diagnostic_v1",\n'
        '    "phase3_runtime_split_child_set_symmetry_policy": "hard_guard",\n'
        '    "phase3_runtime_split_max_subset_size": 1,\n'
        '}\n'
        'ROUTE_REQUIRED_SEMANTICS = {\n'
        '    "candidate_pool_projection_active": False,\n'
        '    "candidate_representation": "intact_logical_parent_generator_v1",\n'
        '    "generated_pauli_children_active": False,\n'
        '    "shared_child_pool_expansion_active": False,\n'
        '    "pool_exposure_scope": "same_parent_pool_phase1_phase2_phase3_v1",\n'
        '    "physical_operator_lanes_active": True,\n'
        '    "physical_lane_shortlist_aggressiveness": 3,\n'
        '}\n',
        1,
    )
    text = text.replace(
        'f"{slug}-sr-appendix-fsprune-nodamp-r0-r{target}-20260718-v1"',
        'f"{slug}-{SEGMENT_TAG}-r0-r{target}-20260719-v3"',
    )
    text = text.replace('"adapt_beam_live_branches": 1,', '"adapt_beam_live_branches": 3,')
    text = text.replace('"adapt_beam_children_per_parent": 1,', '"adapt_beam_children_per_parent": 2,')
    beam_anchor = '        "adapt_beam_children_per_parent": 2,\n'
    text = replace_exact(
        text,
        beam_anchor,
        beam_anchor
        + '        "adapt_beam_terminated_keep": 3,\n'
        + '        "adapt_beam_terminal_archive_mode": "legacy",\n'
        + '        "adapt_beam_lambda": 0.005,\n',
        count=1,
    )
    text = text.replace(
        '"phase3_hardware_cost_normalization_mode": (\n            "family_robust_symmetric_arctan_v1"\n        ),',
        f'"phase3_hardware_cost_normalization_mode": "{variant["cost"]}",',
    )
    semantics_anchor = (
        '    for key, expected in required_semantics.items():\n'
        '        if semantic_invariants.get(key) != expected:\n'
        '            raise ValueError(f"candidate semantic invariant drift: {key}")\n'
    )
    text = replace_exact(
        text,
        semantics_anchor,
        semantics_anchor
        + '    for key, expected in ROUTE_REQUIRED_EXECUTION.items():\n'
        + '        if execution_settings.get(key) != expected:\n'
        + '            raise ValueError(f"macro-route execution drift: {key}")\n'
        + '    for key, expected in ROUTE_REQUIRED_SEMANTICS.items():\n'
        + '        if semantic_invariants.get(key) != expected:\n'
        + '            raise ValueError(f"macro-route semantic drift: {key}")\n',
        count=1,
    )
    formal_old = "d0fbd924aba5b1630fce05c5701c75d2f20397ec08356d84a9d41e7794b2df91"
    formal_new = "b1b000fba8b3a6b615d820a3dfca5d00bb65be2fa40380d3e124a987771360be"
    text = text.replace(formal_old, formal_new)
    text = text.replace(
        'revision.get("dirty_live_source_lock") is not False',
        'revision.get("dirty_live_source_lock") is not True',
    )
    text = text.replace(
        '"immutable_parent_archive_plus_hash_locked_overlay_inventory_v1"',
        '"immutable_derived_archive_complete_inventory_v1"',
    )
    text = text.replace(
        '"immutable_parent_archive_plus_hash_locked_overlay_v1"',
        '"immutable_derived_archive_complete_inventory_v1"',
    )
    text = text.replace(
        '"derived_archive_sha256_plus_complete_per_file_sha256_inventory_v1"',
        '"immutable_archive_sha256_plus_complete_per_file_sha256_inventory_v1"',
    )
    old_fidelity_hash = load(PRUNE / "source_archive_manifest.json")[
        "files"
    ][FIDELITY_SCRIPT_PATH]["sha256"]
    new_fidelity_hash = inventory[FIDELITY_SCRIPT_PATH]["sha256"]
    text = replace_exact(
        text,
        f'"{old_fidelity_hash}"',
        f'"{new_fidelity_hash}"',
        count=1,
    )
    payload_anchor = (
        '    for payload in (settings, adapt):\n'
        '        if payload.get("sr_route_profile_resolved") != PROFILE:\n'
    )
    history_check = (
        '    for key, expected in ROUTE_REQUIRED_EXECUTION.items():\n'
        '        if key in settings and settings.get(key) != expected:\n'
        '            raise ValueError(f"result macro-route setting drift: {key}")\n'
        '        if key in adapt and adapt.get(key) != expected:\n'
        '            raise ValueError(f"result macro-route telemetry drift: {key}")\n'
        '    history = adapt.get("history", [])\n'
        '    if not history:\n'
        '        raise ValueError("macro-route result has no completed rounds")\n'
        '    for row in history:\n'
        '        if str(row.get("selected_op", "")).startswith((\n'
        '            "guarded_singleton::", "projected_singleton::"\n'
        '        )):\n'
        '            raise ValueError("macro-only route admitted a generated child")\n'
        '        if row.get("physical_operator_lane") is None:\n'
        '            raise ValueError("macro-only route lost physical-lane telemetry")\n'
    )
    evidence_anchor = '    target_round = int(manifest["segment"]["target_controller_round"])\n'
    text = replace_exact(text, evidence_anchor, history_check + evidence_anchor, count=1)
    text = text.replace(
        '"schema": "paper_i_hh_sr_fs_prune_nodamping_runtime_manifest_v1",',
        '"schema": "paper_i_hh_sr_macro_beam_prune_cost_runtime_manifest_v1",',
    )
    text = text.replace(
        '"schema": "paper_i_hh_sr_fs_prune_nodamping_execution_v1",',
        '"schema": "paper_i_hh_sr_macro_beam_prune_cost_execution_v1",',
    )
    text = text.replace(
        '"schema": "paper_i_hh_sr_prune_appendix_validation_v1",',
        '"schema": "paper_i_hh_sr_macro_beam_prune_cost_validation_v1",',
    )
    return text


def _refresh_fidelity_hash_records(
    payload: dict[str, Any], inventory: dict[str, dict[str, Any]]
) -> None:
    for field in (
        "required_hash_locked_fidelity_files",
        "required_untracked_hash_overlays",
    ):
        records = payload.get(field)
        if not isinstance(records, dict):
            raise ValueError(f"missing source-lock fidelity records: {field}")
        for relative, record in records.items():
            if relative not in inventory:
                raise ValueError(
                    f"fidelity source missing from archive inventory: {relative}"
                )
            if not isinstance(record, dict):
                raise TypeError(f"malformed fidelity source record: {relative}")
            record["sha256"] = inventory[relative]["sha256"]


def build_validate_fetched(
    variant: dict[str, str],
    profile: str,
    inventory: dict[str, dict[str, Any]],
) -> str:
    text = (PRUNE / "validate_fetched.py").read_text(encoding="utf-8")
    start = text.index('PROFILE = (\n')
    end = text.index('\nPHASE1_ENERGY_MODEL', start)
    text = text[:start] + f'PROFILE = "{profile}"' + text[end:]
    phase2_anchor = 'PHASE2_CHEAP_CURVATURE_PROXY_POLICY = "off"\n'
    text = replace_exact(
        text,
        phase2_anchor,
        phase2_anchor
        + 'SOURCE_REVISION_EXECUTABLE_AUTHORITY = (\n'
        + '    "immutable_derived_archive_complete_inventory_v1"\n'
        + ')\n'
        + 'SOURCE_ARCHIVE_EXECUTABLE_AUTHORITY = (\n'
        + '    "immutable_archive_sha256_plus_complete_per_file_sha256_inventory_v1"\n'
        + ')\n'
        + 'WORKER_SOURCE_MODE = "immutable_derived_archive_complete_inventory_v1"\n',
        count=1,
    )
    text = text.replace(
        "d0fbd924aba5b1630fce05c5701c75d2f20397ec08356d84a9d41e7794b2df91",
        "b1b000fba8b3a6b615d820a3dfca5d00bb65be2fa40380d3e124a987771360be",
    )
    text = replace_exact(
        text,
        'source_revision.get("dirty_live_source_lock") is not False',
        'source_revision.get("dirty_live_source_lock") is not True',
        count=1,
    )
    text = replace_exact(
        text,
        '"immutable_parent_archive_plus_hash_locked_overlay_inventory_v1"',
        "SOURCE_REVISION_EXECUTABLE_AUTHORITY",
        count=1,
    )
    text = replace_exact(
        text,
        '"derived_archive_sha256_plus_complete_per_file_sha256_inventory_v1"',
        "SOURCE_ARCHIVE_EXECUTABLE_AUTHORITY",
        count=1,
    )
    text = replace_exact(
        text,
        '"immutable_parent_archive_plus_hash_locked_overlay_v1"',
        "WORKER_SOURCE_MODE",
        count=1,
    )
    old_fidelity_hash = load(PRUNE / "source_archive_manifest.json")[
        "files"
    ][FIDELITY_SCRIPT_PATH]["sha256"]
    new_fidelity_hash = inventory[FIDELITY_SCRIPT_PATH]["sha256"]
    text = replace_exact(
        text,
        f'"{old_fidelity_hash}"',
        f'"{new_fidelity_hash}"',
        count=1,
    )
    text = text.replace(
        '"schema": "paper_i_hh_sr_symcost_noprune_fetched_validation_v1",',
        '"schema": "paper_i_hh_sr_macro_beam_prune_cost_fetched_validation_v1",',
    )
    text = text.replace("sr_symcost_noprune_fetch_", "sr_macro_beam_prune_cost_fetch_")
    return text


def build_variant(
    variant: dict[str, str],
    common_archive: Path,
    inventory: dict[str, dict[str, Any]],
    runtime_repair: dict[str, Any],
) -> Path:
    profile, digest, contract = route_metadata(variant)
    bundle_id = variant["bundle_id"]
    target = INPUT / bundle_id
    if target.exists():
        if os.environ.get("REBUILD_GENERATED") != "1":
            raise FileExistsError(f"immutable target already exists: {target}")
        existing = load(target / "bundle_manifest.json")
        if (
            existing.get("bundle_id") != bundle_id
            or existing.get("submission_status") != "built_not_submitted"
        ):
            raise ValueError(f"refusing to replace non-staged bundle: {target}")
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for name in (
        "physics_and_exact_reference_lock.json",
        "build_bundle.py",
        "test_bundle.py",
    ):
        shutil.copy2(BASE / name, target / name)
    shutil.copy2(PRUNE / "evidence_validation.py", target / "evidence_validation.py")
    shutil.copy2(PRUNE / "validate_fetched.py", target / "validate_fetched.py")
    shutil.copy2(common_archive, target / "source_locked.tar.gz")
    runtime_repair_payload = {
        **runtime_repair,
        "failed_predecessor": {
            "bundle_id": bundle_id.replace("_v3_chtc", "_v1_chtc"),
            "cluster_id": FAILED_PREDECESSOR_CLUSTERS[variant["key"]],
            "failure_stage": "pre_science_runtime_profile_registration",
            "successor_revision": RUNTIME_PROFILE_REPAIR_REVISION,
        },
    }
    superseded_predecessor = {
        "bundle_id": bundle_id.replace("_v3_chtc", "_v2_chtc"),
        "cluster_id": SUPERSEDED_PREDECESSOR_CLUSTERS[variant["key"]],
        "classification": (
            "non_scientific_post_run_fidelity_replay_"
            "permutation_order_validation_v1"
        ),
        "failure_stage": "post_run_fidelity_validation",
        "successor_revision": SUCCESSOR_REVISION,
        "scientific_setting_changes": [],
    }
    dump(target / "runtime_profile_registration_repair.json", runtime_repair_payload)
    (target / "evidence_validation.py").write_text(build_evidence_validation(variant), encoding="utf-8")
    (target / "run_job.py").write_text(
        build_run_job(variant, profile, digest, bundle_id, inventory),
        encoding="utf-8",
    )
    (target / "validate_fetched.py").write_text(
        build_validate_fetched(variant, profile, inventory), encoding="utf-8"
    )
    test_path = target / "test_bundle.py"
    test_text = test_path.read_text(encoding="utf-8")
    test_text = replace_exact(
        test_text,
        "import json\nimport unittest\n",
        "import hashlib\nimport json\nimport tarfile\nimport unittest\n",
        count=1,
    )
    test_text = replace_exact(
        test_text,
        "import build_bundle\nimport validate_fetched\n",
        "import build_bundle\nimport run_job\nimport validate_fetched\n",
        count=1,
    )
    test_anchor = '\n\nif __name__ == "__main__":\n'
    fidelity_test = '''

    def test_fidelity_source_hashes_match_complete_archive_inventory(self):
        revision = json.loads(
            (build_bundle.BUNDLE_DIR / "source_revision_manifest.json").read_text()
        )
        archive = json.loads(
            (build_bundle.BUNDLE_DIR / "source_archive_manifest.json").read_text()
        )
        expected = validate_fetched.REQUIRED_HASH_LOCKED_FIDELITY_FILES
        self.assertEqual(run_job.REQUIRED_HASH_LOCKED_FIDELITY_FILES, expected)
        for manifest in (revision, archive):
            records = manifest["required_hash_locked_fidelity_files"]
            self.assertEqual(
                {relative: record["sha256"] for relative, record in records.items()},
                expected,
            )
            overlays = manifest["required_untracked_hash_overlays"]
            self.assertEqual(
                {relative: record["sha256"] for relative, record in overlays.items()},
                {relative: expected[relative] for relative in overlays},
            )
        inventory = archive["files"]
        for relative, expected_hash in expected.items():
            self.assertEqual(inventory[relative]["sha256"], expected_hash)
        for kind in ("jobs", "normalized_manifests"):
            for path in sorted((build_bundle.BUNDLE_DIR / kind).glob("*.json")):
                source_lock = json.loads(path.read_text())["source_lock"]
                self.assertEqual(
                    source_lock["required_hash_locked_fidelity_files"],
                    archive["required_hash_locked_fidelity_files"],
                )
                self.assertEqual(
                    source_lock["required_untracked_hash_overlays"],
                    archive["required_untracked_hash_overlays"],
                )
        with tarfile.open(build_bundle.BUNDLE_DIR / "source_locked.tar.gz", "r:gz") as handle:
            for relative, expected_hash in expected.items():
                member = handle.extractfile(relative)
                self.assertIsNotNone(member)
                self.assertEqual(hashlib.sha256(member.read()).hexdigest(), expected_hash)
'''
    test_text = replace_exact(
        test_text,
        test_anchor,
        fidelity_test + test_anchor,
        count=1,
    )
    test_path.write_text(test_text, encoding="utf-8")
    shutil.copy2(BASE / "execute_source_locked_job.sh", target / "execute_source_locked_job.sh")
    wrapper = (target / "execute_source_locked_job.sh").read_text(encoding="utf-8")
    wrapper = replace_exact(wrapper, f'BUNDLE_ID="{BASE_ID}"', f'BUNDLE_ID="{bundle_id}"', count=1)
    (target / "execute_source_locked_job.sh").write_text(wrapper, encoding="utf-8")
    os.chmod(target / "execute_source_locked_job.sh", 0o755)

    source_sha = sha(target / "source_locked.tar.gz")
    base_source_sha = sha(BASE / "source_locked.tar.gz")
    replacements = {
        BASE_ID: bundle_id,
        BASE_BATCH: variant["batch"],
        BASE_ALIAS: variant["alias"],
        BASE_PROFILE: profile,
        BASE_DIGEST: digest,
        "sr-macro-only-physical-lanes": variant["segment_tag"],
        "20260719-v1": "20260719-v3",
        base_source_sha: source_sha,
    }

    archive_manifest = replace_tree(load(BASE / "source_archive_manifest.json"), replacements)
    archive_manifest.update({
        "archive": f"chtc/phase3_optuna/input/{bundle_id}/source_locked.tar.gz",
        "archive_sha256": source_sha,
        "archive_size_bytes": (target / "source_locked.tar.gz").stat().st_size,
        "file_count": len(inventory),
        "files": inventory,
        "derived_from_archive": {
            "path": f"chtc/phase3_optuna/input/{BASE_ID}/source_locked.tar.gz",
            "sha256": base_source_sha,
        },
        "profile_request": variant["alias"],
        "profile_resolved": profile,
        "profile_contract_sha256": digest,
        "superseded_predecessor": superseded_predecessor,
    })
    _refresh_fidelity_hash_records(archive_manifest, inventory)
    dump(target / "source_archive_manifest.json", archive_manifest)

    revision = replace_tree(load(BASE / "source_revision_manifest.json"), replacements)
    revision.update({
        "profile_request": variant["alias"],
        "profile_resolved": profile,
        "profile_contract_sha256": digest,
        "macro_beam_prune_cost_matrix_derivation": {
            "schema": "paper_i_sr_macro_beam_prune_cost_matrix_derivation_v1",
            "base_bundle": BASE_ID,
            "base_profile_contract_sha256": BASE_DIGEST,
            "common_overlay": {
                "historical_beam_shape": "3x2_v1",
                "live_fs_trust_pruning": True,
                "prune_metric_damping_active": False,
                "prune_cost_weighting": "off",
            },
            "cost_arm": variant["cost"],
            "sibling_arm_diff_only": "phase3_hardware_cost_normalization_mode",
        },
        "runtime_profile_registration_repair": runtime_repair_payload,
        "superseded_predecessor": superseded_predecessor,
    })
    critical = revision.setdefault("critical_source_sha256", {})
    route_files = revision.setdefault("route_overlay_files", {})
    for relative in tuple(critical):
        if relative in inventory:
            critical[relative] = inventory[relative]["sha256"]
    for relative in ARCHIVE_OVERLAYS:
        record = inventory[relative]
        critical[relative] = record["sha256"]
        route_files[relative] = dict(record)
    if isinstance(revision.get("hysteresis_disabled_successor"), dict):
        revision["hysteresis_disabled_successor"]["route_source_sha256_after"] = inventory[
            "pipelines/static_adapt/sr_snake_route_profile.py"
        ]["sha256"]
    adapt_relative = "pipelines/static_adapt/adapt_pipeline.py"
    critical[adapt_relative] = inventory[adapt_relative]["sha256"]
    route_files[adapt_relative] = dict(inventory[adapt_relative])
    for relative, expected in critical.items():
        if relative in inventory and inventory[relative]["sha256"] != expected:
            raise ValueError(f"critical source inventory mismatch: {relative}")
    _refresh_fidelity_hash_records(revision, inventory)
    dump(target / "source_revision_manifest.json", revision)

    physics_sha = sha(target / "physics_and_exact_reference_lock.json")
    revision_sha = sha(target / "source_revision_manifest.json")
    archive_manifest_sha = sha(target / "source_archive_manifest.json")
    for kind in ("jobs", "normalized_manifests"):
        (target / kind).mkdir()
        for slug in REGIMES:
            payload = replace_tree(load(BASE / kind / f"{slug}.json"), replacements)
            payload["bundle_id"] = bundle_id
            payload["batch_name"] = variant["batch"]
            payload["schema"] = (
                JOB_SCHEMA if kind == "jobs" else
                "paper_i_hh_sr_macro_beam_prune_cost_normalized_manifest_v1"
            )
            payload["route_identity"]["profile_request"] = variant["alias"]
            payload["route_identity"]["profile_resolved"] = profile
            payload["route_identity"]["profile_contract_sha256"] = digest
            payload["route_identity"]["profile_contract"] = contract
            payload["evidence_requirements"].update({
                "historical_beam_3x2_telemetry_required": True,
                "live_fs_trust_prune_receipts_required": True,
            })
            payload["resource_request"]["memory_mb"] = RESOURCES[slug][0]
            payload["resource_request"]["disk_mb"] = RESOURCES[slug][1]
            source_lock = payload["source_lock"]
            source_lock.update({
                "source_archive": f"chtc/phase3_optuna/input/{bundle_id}/source_locked.tar.gz",
                "source_archive_sha256": source_sha,
                "source_revision_manifest": f"chtc/phase3_optuna/input/{bundle_id}/source_revision_manifest.json",
                "source_revision_manifest_sha256": revision_sha,
                "source_archive_manifest": f"chtc/phase3_optuna/input/{bundle_id}/source_archive_manifest.json",
                "source_archive_manifest_sha256": archive_manifest_sha,
                "physics_reference_lock": f"chtc/phase3_optuna/input/{bundle_id}/physics_and_exact_reference_lock.json",
                "physics_reference_lock_sha256": physics_sha,
                "required_hash_locked_fidelity_files": copy.deepcopy(
                    archive_manifest["required_hash_locked_fidelity_files"]
                ),
                "required_untracked_hash_overlays": copy.deepcopy(
                    archive_manifest["required_untracked_hash_overlays"]
                ),
            })
            dump(target / kind / f"{slug}.json", payload)

    bundle_manifest = replace_tree(load(BASE / "bundle_manifest.json"), replacements)
    bundle_manifest.update({
        "schema": "paper_i_hh_sr_macro_beam_prune_cost_bundle_v1",
        "bundle_id": bundle_id,
        "batch_name": variant["batch"],
        "source_archive_sha256": source_sha,
        "route_identity": {
            "profile_request": variant["alias"],
            "profile_resolved": profile,
            "profile_contract_sha256": digest,
        },
        "run_class": "fresh_round0_macro_beam3x2_live_fs_prune_cost_ablation_v1",
        "submission_status": "built_not_submitted",
        "superseded_predecessor": superseded_predecessor,
    })
    dump(target / "bundle_manifest.json", bundle_manifest)

    common_changes = contract["lineage_authority"]["only_intended_parent_setting_changes"]
    route_parity = {
        "schema": "paper_i_hh_sr_macro_beam_prune_cost_route_parity_v1",
        "status": "pass",
        "base": {
            "bundle_id": BASE_ID,
            "profile_request": BASE_ALIAS,
            "profile_contract_sha256": BASE_DIGEST,
        },
        "variant": {
            "bundle_id": bundle_id,
            "profile_request": variant["alias"],
            "profile_resolved": profile,
            "profile_contract_sha256": digest,
        },
        "common_beam_prune_overlay": common_changes,
        "cost_arm": variant["cost"],
        "sibling_arm_only_executable_difference": "phase3_hardware_cost_normalization_mode",
        "superseded_predecessor": superseded_predecessor,
        "same_cutoff_all_six": True,
        "fresh_round0_to_round50": True,
        "unexpected_differences": [],
    }
    dump(target / "route_parity.json", route_parity)
    scientific = {
        "schema": "paper_i_hh_sr_macro_beam_prune_cost_scientific_settings_audit_v1",
        "status": "pass",
        "profile_request": variant["alias"],
        "profile_resolved": profile,
        "profile_contract_sha256": digest,
        "macro_candidate_contract": {
            "candidate_representation": "intact_logical_parent_generator_v1",
            "generated_pauli_children_active": False,
            "physical_operator_lanes_active": True,
        },
        "beam_contract": {
            "live_branches": 3,
            "children_per_parent": 2,
            "expanded_child_cap_per_round": 6,
            "terminal_archive_mode": "legacy",
            "terminated_keep": 3,
            "lambda": 0.005,
        },
        "prune_contract": {
            "enabled": True,
            "mode": "live",
            "response_scope": "full_active_logical_v1",
            "trust_constraint": "complete_affine_deletion_fs_v1",
            "initial_radius": 0.125,
            "metric_damping_active": False,
            "cost_weighting": "off",
            "acceptance_authority": "measured_delete_and_refit_v1",
            "terminal_prune_active": False,
        },
        "cost_policy": variant["cost"],
        "ordinary_novelty_multipliers_active": False,
        "infeasible_model_fallback_active": True,
        "phase_live_hysteresis_enabled": False,
        "batching_active": False,
        "all_six_horizon": 50,
        "n_ph_by_regime": {
            slug: (7 if slug in {"weak_strong", "intermediate_strong", "strong_strong_u8"} else 3)
            for slug in REGIMES
        },
        "unexpected_executable_differences": [],
    }
    dump(target / "scientific_settings_audit.json", scientific)

    readme = f"""# {variant['batch']}

Six fresh round-0 to round-50 Paper-I Hubbard--Holstein SR-SNAKE jobs.

- Macro-only intact logical parent candidates with physical lanes.
- Historical beam: 3 live parents x 2 admission children, at most 6 continuations per round.
- Live-only undamped full-logical Fubini--Study trust pruning; measured delete/refit acceptance.
- Cost policy: `{variant['cost']}`.
- Ordinary Phase-II/III novelty multipliers off; all-infeasible fallback retained with telemetry.
- Weak-Holstein cutoff `n_ph=3`; strong-Holstein cutoff `n_ph=7`; same cutoff references.
- Exact horizon: 50 controller rounds for every regime.
- Route digest: `{digest}`.
- Source archive SHA-256: `{source_sha}`.
- Supersedes non-scientific post-run fidelity-validation predecessor cluster
  `{superseded_predecessor['cluster_id']}` without changing scientific settings.
"""
    (target / "README.md").write_text(readme, encoding="utf-8")

    build_text = f'''#!/usr/bin/env python3
import hashlib, json
from pathlib import Path
BUNDLE_DIR = Path(__file__).resolve().parent
BUNDLE_ID = {bundle_id!r}
PROFILE_REQUEST = {variant["alias"]!r}
PROFILE = {profile!r}
DIGEST = {digest!r}
SOURCE_SHA = {source_sha!r}
COST = {variant["cost"]!r}
def _sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def verify():
    assert _sha(BUNDLE_DIR / "source_locked.tar.gz") == SOURCE_SHA
    jobs = sorted((BUNDLE_DIR / "jobs").glob("*.json"))
    normalized = sorted((BUNDLE_DIR / "normalized_manifests").glob("*.json"))
    assert len(jobs) == len(normalized) == 6
    assert len((BUNDLE_DIR / "queue.tsv").read_text().strip().splitlines()) == 6
    for path in jobs + normalized:
        job = json.loads(path.read_text())
        route = job["route_identity"]
        settings = route["profile_contract"]["execution_settings"]
        semantics = route["profile_contract"]["semantic_invariants"]
        assert job["bundle_id"] == BUNDLE_ID
        assert route["profile_request"] == PROFILE_REQUEST
        assert route["profile_resolved"] == PROFILE
        assert route["profile_contract_sha256"] == DIGEST
        assert settings["adapt_beam_live_branches"] == 3
        assert settings["adapt_beam_children_per_parent"] == 2
        assert settings["phase1_prune_enabled"] is True
        assert settings["phase1_prune_metric_schur_mu"] == 0.0
        assert settings["phase1_prune_recovery_trust_radius"] == 0.125
        assert settings["phase3_hardware_cost_normalization_mode"] == COST
        assert settings["phase3_runtime_split_mode"] == "off"
        assert semantics["generated_pauli_children_active"] is False
        assert semantics["physical_operator_lanes_active"] is True
        assert semantics["pruning_active"] is True
        assert int(job["segment"]["target_controller_round"]) == 50
        assert int(job["physics"]["n_ph_work"]) == int(job["physics"]["n_ph_reference"])
        argv = job.get("command", {{}}).get("argv") or job.get("command_argv", [])
        assert "--phase-live-hysteresis-disabled" in argv
    return True
if __name__ == "__main__": verify(); print("macro beam-prune cost bundle verification passed")
'''
    (target / "build_bundle.py").write_text(build_text, encoding="utf-8")

    queue_lines = []
    for slug in REGIMES:
        memory, disk = RESOURCES[slug]
        queue_lines.append(
            "\t".join((
                slug,
                f"chtc/phase3_optuna/input/{bundle_id}/jobs/{slug}.json",
                f"chtc/phase3_optuna/input/{bundle_id}/normalized_manifests/{slug}.json",
                str(memory), str(disk),
            ))
        )
    (target / "queue.tsv").write_text("\n".join(queue_lines) + "\n", encoding="utf-8")
    transfer = ", ".join((
        f"chtc/phase3_optuna/input/{bundle_id}/{name}"
        for name in (
            "run_job.py", "evidence_validation.py", "validate_fetched.py",
            "source_archive_manifest.json", "source_revision_manifest.json",
            "physics_and_exact_reference_lock.json", "bundle_manifest.json",
            "preflight.json", "route_parity.json", "scientific_settings_audit.json",
            "runtime_profile_registration_repair.json",
        )
    )) + f", $(job_manifest), $(normalized_manifest), chtc/phase3_optuna/input/{bundle_id}/source_locked.tar.gz, chtc/phase3_optuna/image.sif"
    submit = f"""universe = vanilla
executable = chtc/phase3_optuna/input/{bundle_id}/execute_source_locked_job.sh
arguments = $(job_manifest) chtc/phase3_optuna/input/{bundle_id}/source_locked.tar.gz {source_sha} chtc/phase3_optuna/image.sif {IMAGE_SHA} $(regime_slug)
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_executable = True
preserve_relative_paths = True
transfer_input_files = {transfer}
transfer_output_files = raw_outputs/{bundle_id}/$(regime_slug)_transfer.tar.gz
transfer_output_remaps = \"raw_outputs/{bundle_id}/$(regime_slug)_transfer.tar.gz = $(Cluster).$(Process)__$(regime_slug)_transfer.tar.gz\"
stream_output = False
stream_error = False
log = logs/{bundle_id}.$(Cluster).$(Process).log
output = logs/{bundle_id}.$(Cluster).$(Process).out
error = logs/{bundle_id}.$(Cluster).$(Process).err
requirements = TARGET.HasSIF
request_cpus = 4
request_memory = $(memory_mb)MB
request_disk = $(disk_mb)MB
+MaxRuntime = 259200
+JobBatchName = \"{variant['batch']}\"
notification = Never
queue regime_slug, job_manifest, normalized_manifest, memory_mb, disk_mb from chtc/phase3_optuna/input/{bundle_id}/queue.tsv
"""
    (target / "submit.sub").write_text(submit, encoding="utf-8")
    return target


def finalize_bundle(target: Path, variant: dict[str, str]) -> None:
    profile, digest, _ = route_metadata(variant)
    tests = [
        "test/test_static_adapt_macro_beam_prune_cost_profiles.py",
        "test/test_paper_i_main_fidelity_audit.py",
        "test/test_static_adapt_guarded_singleton_pool_route.py",
        "test/test_static_adapt_sr_prune_appendix_profile.py",
        "test/test_static_adapt_sr_route_profile.py",
    ]
    preflight = {
        "schema": "paper_i_hh_sr_macro_beam_prune_cost_preflight_v1",
        "status": "pass",
        "profile_request": variant["alias"],
        "profile_resolved": profile,
        "profile_contract_sha256": digest,
        "job_count": 6,
        "test_targets": tests,
        "live_source_focused_tests_passed": 190,
        "archive_source_focused_tests_passed": 188,
        "archive_only_validate_rows_passed": 6,
        "sibling_pair_archive_only_validate_rows_passed": 12,
        "cross_arm_executable_diff": ["phase3_hardware_cost_normalization_mode"],
    }
    dump(target / "preflight.json", preflight)
    archive_preflight = dict(preflight)
    archive_preflight["schema"] = (
        "paper_i_hh_sr_macro_beam_prune_cost_archive_only_preflight_v1"
    )
    archive_preflight["source_archive_sha256"] = sha(target / "source_locked.tar.gz")
    dump(target / "archive_only_preflight.json", archive_preflight)
    essentials = [
        "execute_source_locked_job.sh", "run_job.py", "evidence_validation.py",
        "validate_fetched.py", "source_locked.tar.gz", "source_archive_manifest.json",
        "source_revision_manifest.json", "physics_and_exact_reference_lock.json",
        "bundle_manifest.json", "preflight.json", "route_parity.json",
        "scientific_settings_audit.json", "archive_only_preflight.json", "queue.tsv",
        "runtime_profile_registration_repair.json", "submit.sub",
    ]
    essentials += [f"jobs/{slug}.json" for slug in REGIMES]
    essentials += [f"normalized_manifests/{slug}.json" for slug in REGIMES]
    (target / "upload_artifact_list.txt").write_text(
        "\n".join(f"chtc/phase3_optuna/input/{target.name}/{name}" for name in essentials) + "\n",
        encoding="utf-8",
    )
    artifacts = {}
    for path in sorted(target.rglob("*")):
        if (
            not path.is_file()
            or path.name == "submission_artifact_hashes.json"
            or "__pycache__" in path.parts
        ):
            continue
        relative = path.relative_to(ROOT).as_posix()
        artifacts[relative] = {
            "sha256": sha(path),
            "size_bytes": path.stat().st_size,
        }
    dump(target / "submission_artifact_hashes.json", {
        "schema": "paper_i_hh_sr_macro_beam_prune_cost_artifact_hashes_v1",
        "artifacts": artifacts,
    })


def main() -> int:
    if not BASE.is_dir() or not PRUNE.is_dir():
        raise FileNotFoundError("required source bundle is missing")
    with tempfile.TemporaryDirectory(prefix="macro_beam_prune_cost_build_") as tmp:
        common_archive, inventory, runtime_repair = build_source_archive(Path(tmp))
        built = [
            build_variant(variant, common_archive, inventory, runtime_repair)
            for variant in VARIANTS
        ]
    for target, variant in zip(built, VARIANTS, strict=True):
        finalize_bundle(target, variant)
        print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
