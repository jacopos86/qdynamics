#!/usr/bin/env python3
"""Build and fail-closed preflight the six-row Paper-I recovery bundle.

The July-8 ``commands.json`` argv arrays are the sole scientific-settings
baseline.  This builder permits only output/current paths plus the approved
runtime-child padding policy and estimator-ledger telemetry sidecar.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import tarfile
from typing import Any, Iterable, Sequence


BUNDLE_ID = "paper_i_hh_visible_snake_symmetry_padding_recovery_20260712_v1"
BUNDLE_REL = Path("chtc/phase3_optuna/input") / BUNDLE_ID
OUTPUT_ROOT = Path("raw_outputs/paper_i_hh_visible_snake_symmetry_padding_recovery_20260712")
BASELINE_COMMANDS = Path(
    "raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/commands.json"
)
SOURCE_ARCHIVE = BUNDLE_REL / "src_sanitized.tar.gz"
IMAGE = Path("chtc/phase3_optuna/image.sif")
IMAGE_SHA256 = "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
HISTORICAL_SRC_ARCHIVE_JULY8 = Path(
    "chtc/phase3_optuna/input/paper_i_hh_weak_weak_snake_mechanism_ablation_20260708_v1/src_sanitized.tar.gz"
)
HISTORICAL_SRC_ARCHIVE_JULY8_SHA256 = (
    "898b122a4873459c52eb1161c314bb7aef324226cc3189e2a5791ff21a0f6164"
)
HISTORICAL_SRC_ARCHIVE_JULY9 = Path(
    "chtc/phase3_optuna/input/paper_i_hh_all_regime_snake_mechanism_ablation_20260709_v1/src_sanitized.tar.gz"
)
HISTORICAL_SRC_ARCHIVE_JULY9_SHA256 = (
    "701d68c1c1654d6f3fae6ff7c37a97b9fe5a1fd4bf6d0b37e8f36443fc173797"
)
PRE_REPAIR_REMOTE_SNAPSHOT = (
    BUNDLE_REL
    / "historical_source_evidence"
    / "paper_i_hh_visible_snake_prerepair_remote_snapshot_20260712.tar.gz"
)
PRE_REPAIR_REMOTE_SNAPSHOT_SHA256 = (
    "1c7b8e76566813d1fc571276373b8284f0c08f6a7a07b4d96390901b346a4f63"
)
PRE_REPAIR_CORRECTED_DIFF = (
    BUNDLE_REL / "historical_source_evidence" / "pre_repair_vs_corrected_source_diff.json"
)
LOCKED_COMMANDS_COPY = (
    BUNDLE_REL / "historical_source_evidence" / "commands_20260708.json"
)
ROUTE_PARITY_GATE = (
    BUNDLE_REL
    / "preflight_evidence"
    / "weak_weak_depth1_prerepair_current_route_parity_20260712.json"
)
ENFORCEMENT_SMOKE_GATE = (
    BUNDLE_REL
    / "preflight_evidence"
    / "weak_weak_depth1_corrected_enforcement_smoke_20260712.json"
)
SOURCE_ROOTS = (Path("src"), Path("pipelines"), Path("docs"))
EXPECTED_REGIMES = (
    "weak-weak",
    "intermediate-weak",
    "strong-weak",
    "weak-strong",
    "intermediate-strong",
    "strong-strong",
)
PADDING_POLICY = "exact_projected_grouped_v1"
CACHE_MODE_ENVIRONMENT = {
    "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "disk",
    "STATIC_ADAPT_HH_POOL_CACHE": "disk",
    "STATIC_ADAPT_HH_GENERATOR_REGISTRY_CACHE": "disk",
}
CACHE_EVIDENCE_LOG = Path(
    "raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/logs/weak_weak.stdout.log"
)
CACHE_EVIDENCE_LOG_SHA256 = (
    "9617ce2980d06fb1269867b81ea74700ececcb3926b5d8ccd2043618a518a0dc"
)
MEMORY_MB = 32768
DISK_MB = 61440
CPUS = 4
MAX_RUNTIME_SECONDS = 172800


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _slug(regime: str) -> str:
    return regime.replace("-", "_")


def _cache_environment(regime_slug: str) -> dict[str, str]:
    cache_root = Path("tmp") / BUNDLE_ID / regime_slug / "cache"
    return {
        **CACHE_MODE_ENVIRONMENT,
        "STATIC_ADAPT_CANDIDATE_RECORD_CACHE_DIR": (
            cache_root / "candidate_records"
        ).as_posix(),
        "STATIC_ADAPT_HH_POOL_CACHE_DIR": (cache_root / "hh_pool").as_posix(),
        "STATIC_ADAPT_HH_POOL_CACHE_SCOPE": "exact",
        "STATIC_ADAPT_HH_GENERATOR_REGISTRY_CACHE_DIR": (
            cache_root / "generator_registry"
        ).as_posix(),
    }


def _normalized_options(argv: Sequence[str]) -> list[dict[str, Any]]:
    if list(argv[:3]) != ["python3", "-m", "pipelines.static_adapt.adapt_pipeline"]:
        raise ValueError(f"unexpected command prefix: {argv[:3]!r}")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    index = 3
    while index < len(argv):
        option = str(argv[index])
        if not option.startswith("--"):
            raise ValueError(f"unexpected positional token at index {index}: {option!r}")
        if option in seen:
            raise ValueError(f"duplicate option: {option}")
        seen.add(option)
        if index + 1 < len(argv) and not str(argv[index + 1]).startswith("--"):
            value: Any = str(argv[index + 1])
            index += 2
        else:
            value = True
            index += 1
        records.append({"option": option, "value": value})
    return records


def _option_map(argv: Sequence[str]) -> dict[str, Any]:
    return {str(row["option"]): row["value"] for row in _normalized_options(argv)}


def _derive_corrected_argv(*, baseline: Sequence[str], regime: str) -> tuple[list[str], dict[str, str]]:
    slug = _slug(regime)
    json_root = OUTPUT_ROOT / slug / "json"
    paths = {
        "output_json": (json_root / "result.json").as_posix(),
        "current_json": (json_root / "current.json").as_posix(),
        "estimator_call_ledger_json": (json_root / "estimator_call_ledger.json").as_posix(),
        "normalized_run_manifest_json": (
            OUTPUT_ROOT / slug / "normalized_run_manifest.json"
        ).as_posix(),
    }
    corrected = list(baseline)
    try:
        output_index = corrected.index("--output-json")
    except ValueError as exc:
        raise ValueError(f"{regime}: baseline has no --output-json") from exc
    if output_index + 1 >= len(corrected):
        raise ValueError(f"{regime}: --output-json has no value")
    corrected[output_index + 1] = paths["output_json"]
    corrected[output_index:output_index] = [
        "--phase3-runtime-split-child-padding-policy",
        PADDING_POLICY,
        "--adapt-current-json",
        paths["current_json"],
        "--adapt-estimator-call-ledger-json",
        paths["estimator_call_ledger_json"],
    ]
    return corrected, paths


def _setting_diff(baseline: Sequence[str], corrected: Sequence[str]) -> list[dict[str, Any]]:
    old = _option_map(baseline)
    new = _option_map(corrected)
    rows: list[dict[str, Any]] = []
    for option in sorted(set(old) | set(new)):
        if old.get(option) != new.get(option) or (option in old) != (option in new):
            rows.append(
                {
                    "option": option,
                    "change": (
                        "added" if option not in old else "removed" if option not in new else "replaced"
                    ),
                    "historical": old.get(option),
                    "corrected": new.get(option),
                }
            )
    return rows


def _validate_allowed_diff(diff: Sequence[dict[str, Any]], *, paths: dict[str, str]) -> None:
    expected = {
        "--output-json": ("replaced", paths["output_json"]),
        "--adapt-current-json": ("added", paths["current_json"]),
        "--adapt-estimator-call-ledger-json": (
            "added",
            paths["estimator_call_ledger_json"],
        ),
        "--phase3-runtime-split-child-padding-policy": ("added", PADDING_POLICY),
    }
    actual = {str(row["option"]): (str(row["change"]), row["corrected"]) for row in diff}
    if actual != expected:
        raise ValueError(f"source-lock diff is not allowlisted: actual={actual!r}, expected={expected!r}")


def _archive_manifest(repo: Path, archive: Path) -> dict[str, Any]:
    if not archive.is_file():
        raise FileNotFoundError(
            f"fresh source archive is required before bundle generation: {archive.as_posix()}"
        )
    members: list[dict[str, Any]] = []
    archived_files: dict[str, str] = {}
    with tarfile.open(archive, mode="r:gz") as handle:
        for member in handle.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"unsafe archive member: {member.name!r}")
            if not name.parts or name.parts[0] not in {root.as_posix() for root in SOURCE_ROOTS}:
                raise ValueError(f"archive member outside source roots: {member.name!r}")
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise ValueError(f"non-regular archive member is forbidden: {member.name!r}")
            record: dict[str, Any] = {
                "path": name.as_posix().rstrip("/"),
                "type": "directory" if member.isdir() else "file" if member.isfile() else "other",
                "size_bytes": int(member.size),
            }
            if member.isfile():
                extracted = handle.extractfile(member)
                if extracted is None:
                    raise ValueError(f"cannot read archive member: {member.name!r}")
                digest = hashlib.sha256()
                for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
                    digest.update(chunk)
                record["sha256"] = digest.hexdigest()
                archived_files[name.as_posix()] = digest.hexdigest()
            members.append(record)

    local_files: dict[str, str] = {}
    excluded_names = {".DS_Store"}
    for root in SOURCE_ROOTS:
        if not root.is_dir():
            raise FileNotFoundError(f"missing source root: {root}")
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"source symlink is forbidden: {path.as_posix()}")
            if not path.is_file():
                continue
            if "__pycache__" in path.parts or path.name.endswith(".pyc") or path.name in excluded_names:
                continue
            local_files[path.as_posix()] = _sha256_file(path)
    if archived_files != local_files:
        missing = sorted(set(local_files) - set(archived_files))
        extra = sorted(set(archived_files) - set(local_files))
        mismatched = sorted(
            path for path in set(local_files) & set(archived_files) if local_files[path] != archived_files[path]
        )
        raise ValueError(
            "archive is not an exact fresh snapshot of source roots: "
            f"missing={missing[:10]!r}, extra={extra[:10]!r}, mismatched={mismatched[:10]!r}"
        )
    member_inventory_sha = hashlib.sha256(_json_bytes(members)).hexdigest()
    return {
        "schema": "paper_i_hh_recovery_source_archive_manifest_v1",
        "archive_path": archive.as_posix(),
        "archive_sha256": _sha256_file(archive),
        "archive_size_bytes": archive.stat().st_size,
        "source_roots": [root.as_posix() for root in SOURCE_ROOTS],
        "regular_file_count": len(archived_files),
        "member_count": len(members),
        "member_inventory_sha256": member_inventory_sha,
        "members": members,
        "exact_local_snapshot": True,
        "symlink_or_special_member_count": 0,
    }


def _historical_code_evidence() -> dict[str, Any]:
    archives = (
        (HISTORICAL_SRC_ARCHIVE_JULY8, HISTORICAL_SRC_ARCHIVE_JULY8_SHA256, "2026-07-08"),
        (HISTORICAL_SRC_ARCHIVE_JULY9, HISTORICAL_SRC_ARCHIVE_JULY9_SHA256, "2026-07-09"),
    )
    records: list[dict[str, Any]] = []
    inventories: list[list[dict[str, Any]]] = []
    for path, expected_sha256, date in archives:
        if not path.is_file():
            raise FileNotFoundError(f"missing historical src archive: {path}")
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"historical src archive hash mismatch for {path}: "
                f"expected={expected_sha256} actual={actual_sha256}"
            )
        inventory: list[dict[str, Any]] = []
        with tarfile.open(path, mode="r:gz") as handle:
            for member in handle.getmembers():
                if not member.isfile():
                    continue
                extracted = handle.extractfile(member)
                if extracted is None:
                    raise ValueError(f"cannot read historical archive member: {member.name}")
                digest = hashlib.sha256()
                for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
                    digest.update(chunk)
                inventory.append(
                    {
                        "path": PurePosixPath(member.name).as_posix(),
                        "size_bytes": int(member.size),
                        "sha256": digest.hexdigest(),
                    }
                )
        inventory_sha256 = hashlib.sha256(_json_bytes(inventory)).hexdigest()
        records.append(
            {
                "date": date,
                "path": path.as_posix(),
                "compressed_sha256": actual_sha256,
                "regular_file_count": len(inventory),
                "decompressed_regular_file_inventory_sha256": inventory_sha256,
                "top_level_roots": sorted({PurePosixPath(row["path"]).parts[0] for row in inventory}),
            }
        )
        inventories.append(inventory)
    return {
        "historical_src_archives": records,
        "july8_july9_decompressed_regular_files_identical": inventories[0] == inventories[1],
        "preserved_historical_src_scope": "src/ only",
        "preserved_historical_pipelines_snapshot": False,
        "full_historical_code_byte_identity_proof": "unresolved",
        "exact_blocker": (
            "The July-8/July-9 source archive preserves src/ but not the historical pipelines/ "
            "snapshot that implemented the controller; therefore a byte-for-byte full historical-to-corrected "
            "code diff cannot be reconstructed."
        ),
        "claims_not_made": [
            "No claim that current pipelines/ is byte-identical to the July-8 execution tree.",
            "No claim that the approved source semantic changes are the only byte-level changes since July 8.",
        ],
        "validated_scope": (
            "Exact July-8 argv equality for all unchanged options, explicit corrected enforcement/telemetry "
            "allowlist, and exact hashes for every file in the corrected source archive."
        ),
    }


def _pre_repair_corrected_diff(
    *, corrected_members: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    if not PRE_REPAIR_REMOTE_SNAPSHOT.is_file():
        raise FileNotFoundError(f"missing pre-repair remote snapshot: {PRE_REPAIR_REMOTE_SNAPSHOT}")
    actual_snapshot_sha256 = _sha256_file(PRE_REPAIR_REMOTE_SNAPSHOT)
    if actual_snapshot_sha256 != PRE_REPAIR_REMOTE_SNAPSHOT_SHA256:
        raise ValueError(
            "pre-repair remote snapshot hash mismatch: "
            f"expected={PRE_REPAIR_REMOTE_SNAPSHOT_SHA256} actual={actual_snapshot_sha256}"
        )
    normalized_snapshot: dict[str, dict[str, Any]] = {}
    raw_member_count = 0
    raw_regular_file_count = 0
    ignored_counts: dict[str, int] = {}
    duplicate_normalized_paths: list[str] = []
    with tarfile.open(PRE_REPAIR_REMOTE_SNAPSHOT, mode="r:gz") as handle:
        for member in handle.getmembers():
            raw_member_count += 1
            if not member.isfile():
                continue
            raw_regular_file_count += 1
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe pre-repair snapshot member: {member.name!r}")
            reason: str | None = None
            if not path.parts or path.parts[0] not in {root.as_posix() for root in SOURCE_ROOTS}:
                reason = "outside_source_roots"
            elif "__pycache__" in path.parts or path.name.endswith(".pyc"):
                reason = "python_cache"
            elif path.name.startswith("._"):
                reason = "appledouble"
            elif path.name == ".DS_Store":
                reason = "ds_store"
            if reason is not None:
                ignored_counts[reason] = ignored_counts.get(reason, 0) + 1
                continue
            extracted = handle.extractfile(member)
            if extracted is None:
                raise ValueError(f"cannot read pre-repair snapshot member: {member.name}")
            digest = hashlib.sha256()
            for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
                digest.update(chunk)
            key = path.as_posix()
            if key in normalized_snapshot:
                duplicate_normalized_paths.append(key)
            normalized_snapshot[key] = {
                "sha256": digest.hexdigest(),
                "size_bytes": int(member.size),
                "mtime_epoch": int(member.mtime),
            }
    if duplicate_normalized_paths:
        raise ValueError(
            f"duplicate normalized paths in pre-repair snapshot: {duplicate_normalized_paths[:10]!r}"
        )

    normalized_corrected = {
        str(member["path"]): {
            "sha256": str(member["sha256"]),
            "size_bytes": int(member["size_bytes"]),
        }
        for member in corrected_members
        if member.get("type") == "file" and member.get("sha256") is not None
    }
    inventory: list[dict[str, Any]] = []
    classification_counts: dict[str, int] = {
        "unchanged": 0,
        "modified": 0,
        "added_in_corrected": 0,
        "absent_from_corrected": 0,
    }
    for path in sorted(set(normalized_snapshot) | set(normalized_corrected)):
        before = normalized_snapshot.get(path)
        after = normalized_corrected.get(path)
        if before is None:
            classification = "added_in_corrected"
        elif after is None:
            classification = "absent_from_corrected"
        elif before["sha256"] == after["sha256"]:
            classification = "unchanged"
        else:
            classification = "modified"
        classification_counts[classification] += 1
        inventory.append(
            {
                "path": path,
                "classification": classification,
                "pre_repair_sha256": None if before is None else before["sha256"],
                "corrected_sha256": None if after is None else after["sha256"],
                "pre_repair_size_bytes": None if before is None else before["size_bytes"],
                "corrected_size_bytes": None if after is None else after["size_bytes"],
                "pre_repair_mtime_epoch": None if before is None else before["mtime_epoch"],
            }
        )
    return {
        "schema": "paper_i_hh_pre_repair_vs_corrected_source_diff_v1",
        "pre_repair_snapshot": {
            "path": PRE_REPAIR_REMOTE_SNAPSHOT.as_posix(),
            "sha256": actual_snapshot_sha256,
            "raw_tar_member_count": raw_member_count,
            "raw_regular_file_count": raw_regular_file_count,
            "normalized_regular_file_count": len(normalized_snapshot),
            "ignored_counts": ignored_counts,
            "capture_scope": "remote repo snapshot fetched before the 2026-07-12 repair upload",
            "historical_mtime_note": "key snapshot files carried July-10 mtimes",
        },
        "corrected_snapshot": {
            "path": SOURCE_ARCHIVE.as_posix(),
            "sha256": _sha256_file(SOURCE_ARCHIVE),
            "normalized_regular_file_count": len(normalized_corrected),
        },
        "classification_counts": classification_counts,
        "inventory_row_count": len(inventory),
        "inventory": inventory,
        "scope_caveat": (
            "This closes the pre-repair remote snapshot versus corrected-source diff, but the remote "
            "snapshot is not proven to be the exact July-8 pipelines execution snapshot."
        ),
    }


def _dataless_audit() -> dict[str, Any]:
    command = ["find", *[root.as_posix() for root in SOURCE_ROOTS], "-type", "f", "-exec", "ls", "-lO", "{}", "+"]
    proc = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    affected = [line for line in proc.stdout.splitlines() if "dataless" in line or "compressed,dataless" in line]
    return {
        "command": "find <source-roots> -type f -exec ls -lO {} +",
        "returncode": proc.returncode,
        "affected_count": len(affected),
        "affected": affected,
        "passed": proc.returncode == 0 and not affected,
        "stderr": proc.stderr.strip(),
    }


def _cli_parse_audit(commands: Iterable[Sequence[str]]) -> dict[str, Any]:
    try:
        from pipelines.static_adapt.adapt_pipeline import parse_args
    except Exception as exc:  # pragma: no cover - fail-closed preflight surface
        return {"passed": False, "error": f"import failed: {type(exc).__name__}: {exc}"}
    failures: list[dict[str, str]] = []
    for argv in commands:
        try:
            parse_args(list(argv)[3:])
        except BaseException as exc:  # argparse exits via SystemExit
            failures.append(
                {
                    "regime_output": str(_option_map(argv).get("--output-json")),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {"passed": not failures, "failures": failures, "command_count": len(list(commands)) if isinstance(commands, list) else None}


def _effective_route_audit() -> dict[str, Any]:
    try:
        from pipelines.static_adapt.adapt_pipeline import (
            _resolve_physical_lane_shortlist_budget_contract,
        )

        resolved = _resolve_physical_lane_shortlist_budget_contract(
            static_route_id_key="route_a",
            static_meta_feature_profile="paper_i_production_v1",
            static_lane_route_key="physical_operator_type",
            route_a_funnel_active=False,
            adapt_pool="full_meta",
            adapt_continuation_mode="phase3_v1",
            phase2_enable_batching=False,
            phase3_enable_batching=False,
            phase3_runtime_split_mode="shortlist_pauli_children_v1",
            phase3_runtime_split_selection_mode="archival_child_set_forward_v1",
            phase3_runtime_split_max_subset_size=1,
            phase3_runtime_split_subset_sizes=(1,),
            physical_lane_shortlist_factor=3,
            phase1_shortlist_size_base=24,
            phase2_shortlist_size_base=12,
            phase2_shortlist_fraction_base=0.25,
        )
    except Exception as exc:
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    expected = {
        "policy": "paper_i_july8_physical_lane_factor_division_v1",
        "historical_paper_i_contract_active": True,
        "historical_route_compatibility_id": "paper_i_july8_physical_singleton_route_v1",
        "physical_lane_shortlist_factor": 3,
        "phase1_shortlist_size_base": 24,
        "phase1_shortlist_size_effective": 8,
        "phase2_shortlist_size_base": 12,
        "phase2_shortlist_size_effective": 4,
        "phase2_shortlist_fraction_base": 0.25,
        "phase2_shortlist_fraction_effective": 1.0 / 12.0,
    }
    passed = all(
        abs(float(resolved[key]) - float(value)) <= 1e-15
        if isinstance(value, float)
        else resolved.get(key) == value
        for key, value in expected.items()
    )
    return {
        "passed": bool(passed),
        "historical_requested": {
            "phase1_shortlist_size": 24,
            "phase2_shortlist_size": 12,
            "phase2_shortlist_fraction": 0.25,
            "physical_lane_shortlist_aggressiveness": 3,
        },
        "expected_effective": expected,
        "resolved_current_source": resolved,
        "source_function": (
            "pipelines.static_adapt.adapt_pipeline."
            "_resolve_physical_lane_shortlist_budget_contract"
        ),
    }


def _scan_machine_local_dependencies(paths: Iterable[Path]) -> dict[str, Any]:
    forbidden = ("/Users/", "file://", "~/", "/Volumes/")
    hits: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(token in line for token in forbidden):
                hits.append({"path": path.as_posix(), "line": line_number, "text": line.strip()})
    return {"passed": not hits, "hit_count": len(hits), "hits": hits}


def _runtime_import_closure() -> dict[str, Any]:
    module_paths: dict[str, Path] = {}
    package_modules: set[str] = set()
    for root in SOURCE_ROOTS:
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            parts = list(path.with_suffix("").parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
                package_modules.add(".".join(parts))
            module_paths[".".join(parts)] = path
    entrypoint = "pipelines.static_adapt.adapt_pipeline"
    pending = [entrypoint]
    visited: set[str] = set()
    parse_failures: list[dict[str, str]] = []
    dynamic_literal_imports: list[dict[str, str]] = []
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        path = module_paths.get(module)
        if path is None:
            continue
        visited.add(module)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            parse_failures.append(
                {"module": module, "path": path.as_posix(), "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        package = module if module in package_modules else module.rsplit(".", 1)[0]
        candidates: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                candidates.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package_parts = package.split(".") if package else []
                    if node.level > 1:
                        package_parts = package_parts[: -(node.level - 1)]
                    base_parts = package_parts + (
                        str(node.module).split(".") if node.module else []
                    )
                    base = ".".join(base_parts)
                else:
                    base = str(node.module or "")
                if base:
                    candidates.add(base)
                for alias in node.names:
                    if alias.name != "*":
                        candidates.add(".".join(part for part in (base, alias.name) if part))
            elif (
                isinstance(node, ast.Call)
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and (
                    (isinstance(node.func, ast.Name) and node.func.id == "import_module")
                    or (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr == "import_module"
                    )
                )
            ):
                literal = str(node.args[0].value)
                candidates.add(literal)
                dynamic_literal_imports.append(
                    {"source_module": module, "imported_module": literal}
                )
        for candidate in sorted(candidates):
            if candidate in module_paths and candidate not in visited:
                pending.append(candidate)
    closure_files = sorted(module_paths[module].as_posix() for module in visited)
    return {
        "entrypoint": entrypoint,
        "method": "recursive AST import/from-import plus string-literal import_module closure",
        "module_count": len(visited),
        "files": closure_files,
        "file_inventory_sha256": hashlib.sha256(_json_bytes(closure_files)).hexdigest(),
        "dynamic_literal_imports": dynamic_literal_imports,
        "parse_failures": parse_failures,
        "passed": not parse_failures and entrypoint in visited,
    }


def _archive_machine_local_dependency_audit() -> dict[str, Any]:
    forbidden = ("/Users/", "/Volumes/", "file://", "~/")
    closure = _runtime_import_closure()
    closure_files = set(closure["files"])
    hits: list[dict[str, Any]] = []
    with tarfile.open(SOURCE_ARCHIVE, mode="r:gz") as handle:
        for member in handle.getmembers():
            if not member.isfile():
                continue
            extracted = handle.extractfile(member)
            if extracted is None:
                continue
            data = extracted.read()
            if b"\x00" in data:
                continue
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                matched = [token for token in forbidden if token in line]
                if matched:
                    hits.append(
                        {
                            "path": PurePosixPath(member.name).as_posix(),
                            "line": line_number,
                            "tokens": matched,
                            "runtime_import_reachable": (
                                PurePosixPath(member.name).as_posix() in closure_files
                            ),
                        }
                    )
    reachable_hits = [row for row in hits if row["runtime_import_reachable"]]
    dormant_hit_files = sorted(
        {str(row["path"]) for row in hits if not row["runtime_import_reachable"]}
    )
    return {
        "archive_path": SOURCE_ARCHIVE.as_posix(),
        "archive_sha256": _sha256_file(SOURCE_ARCHIVE),
        "forbidden_machine_local_tokens": list(forbidden),
        "hit_count": len(hits),
        "hit_file_count": len({str(row["path"]) for row in hits}),
        "hits": hits,
        "runtime_import_closure": closure,
        "runtime_reachable_hit_count": len(reachable_hits),
        "runtime_reachable_hits": reachable_hits,
        "dormant_unreachable_hit_files": dormant_hit_files,
        "classification": (
            "dormant_strings_only" if hits and not reachable_hits else "no_hits" if not hits else "runtime_dependency_hit"
        ),
        "scope_note": (
            "The full exact source snapshot intentionally retains unrelated docs/reporting/exact-bench "
            "files containing machine-local example paths. The execute argv and static runtime import "
            "closure contain no such dependency."
        ),
        "passed": bool(closure["passed"] and not reachable_hits),
    }


def _output_isolation_audit() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for regime in EXPECTED_REGIMES:
        path = OUTPUT_ROOT / _slug(regime)
        existing_files = (
            sorted(child.as_posix() for child in path.rglob("*") if child.is_file())
            if path.exists()
            else []
        )
        rows.append(
            {
                "regime": regime,
                "path": path.as_posix(),
                "exists": path.exists(),
                "existing_file_count": len(existing_files),
                "existing_files": existing_files,
                "isolated_empty_target": not existing_files,
            }
        )
    return {
        "output_root": OUTPUT_ROOT.as_posix(),
        "smoke_subdirectories_are_outside_production_regime_targets": True,
        "rows": rows,
        "passed": all(bool(row["isolated_empty_target"]) for row in rows),
    }


def _gate_evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required preflight gate: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    referenced: list[dict[str, Any]] = []

    def _walk(value: Any) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
                artifact_path = Path(value["path"])
                actual = _sha256_file(artifact_path) if artifact_path.is_file() else None
                referenced.append(
                    {
                        "path": artifact_path.as_posix(),
                        "expected_sha256": value["sha256"],
                        "actual_sha256": actual,
                        "matches": actual == value["sha256"],
                    }
                )
            for child in value.values():
                _walk(child)
        elif isinstance(value, list):
            for child in value:
                _walk(child)

    _walk(payload.get("input_artifacts", {}))
    referenced_passed = bool(referenced and all(row["matches"] for row in referenced))
    passed = bool(
        payload.get("passed") is True
        and str(payload.get("status")).lower() == "passed"
        and referenced_passed
    )
    return {
        "path": path.as_posix(),
        "sha256": _sha256_file(path),
        "schema": payload.get("schema"),
        "gate_reported_passed": payload.get("passed"),
        "referenced_artifact_count": len(referenced),
        "referenced_artifacts": referenced,
        "referenced_artifact_hashes_passed": referenced_passed,
        "passed": passed,
        "execute_node_input": False,
    }


def _submit_text(
    *,
    archive_sha256: str,
    job_manifest_paths: Sequence[Path],
    argv_paths: Sequence[Path],
) -> str:
    transfer_files = [
        (BUNDLE_REL / "run_job.py").as_posix(),
        (BUNDLE_REL / "source_lock_comparison.json").as_posix(),
        (BUNDLE_REL / "source_archive_manifest.json").as_posix(),
        LOCKED_COMMANDS_COPY.as_posix(),
        SOURCE_ARCHIVE.as_posix(),
        *[path.as_posix() for path in job_manifest_paths],
        *[path.as_posix() for path in argv_paths],
        IMAGE.as_posix(),
    ]
    return f"""universe = vanilla
executable = {(BUNDLE_REL / 'execute_source_locked_job.sh').as_posix()}
arguments = $(job_manifest) {SOURCE_ARCHIVE.as_posix()} {archive_sha256} {IMAGE.as_posix()} {IMAGE_SHA256}
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_executable = True
preserve_relative_paths = True
transfer_input_files = {', '.join(transfer_files)}
transfer_output_files = {OUTPUT_ROOT.as_posix()}/$(regime_slug)
stream_output = False
stream_error = False
log = logs/{BUNDLE_ID}.$(Cluster).$(Process).log
output = logs/{BUNDLE_ID}.$(Cluster).$(Process).out
error = logs/{BUNDLE_ID}.$(Cluster).$(Process).err
requirements = TARGET.HasSIF
request_cpus = {CPUS}
request_memory = $(memory_mb)MB
request_disk = $(disk_mb)MB
+MaxRuntime = {MAX_RUNTIME_SECONDS}
+JobBatchName = \"paper-i-hh-visible-snake-recovery-20260712-v1\"
notification = Never
queue regime_slug, job_manifest, memory_mb, disk_mb from {(BUNDLE_REL / 'queue.tsv').as_posix()}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo.resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    bundle = BUNDLE_REL
    jobs_dir = bundle / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    checkout_guard = {
        "checkout_basename": repo.name,
        "contains_local_repos_component": "local_repos" in repo.parts,
        "contains_icloud_documents_component": "Documents" in repo.parts,
    }
    checkout_guard["passed"] = bool(
        checkout_guard["contains_local_repos_component"]
        and not checkout_guard["contains_icloud_documents_component"]
    )
    if not checkout_guard["passed"]:
        raise RuntimeError(f"non-iCloud checkout guard failed: {checkout_guard!r}")
    if not CACHE_EVIDENCE_LOG.is_file():
        raise FileNotFoundError(f"missing historical cache evidence log: {CACHE_EVIDENCE_LOG}")
    if _sha256_file(CACHE_EVIDENCE_LOG) != CACHE_EVIDENCE_LOG_SHA256:
        raise ValueError("historical cache evidence log hash mismatch")

    source_archive_manifest = _archive_manifest(repo, SOURCE_ARCHIVE)
    _write_json(bundle / "source_archive_manifest.json", source_archive_manifest)
    historical_code_evidence = _historical_code_evidence()
    corrected_member_hashes = {
        str(member["path"]): str(member["sha256"])
        for member in source_archive_manifest["members"]
        if member.get("type") == "file" and member.get("sha256") is not None
    }
    critical_corrected_files = (
        "pipelines/scaffold/hh_continuation_generators.py",
        "pipelines/static_adapt/adapt_pipeline.py",
        "pipelines/static_adapt/cli_config.py",
        "pipelines/static_adapt/adapt_candidate_record_cache.py",
        "pipelines/static_adapt/builders/hh_pool_presets.py",
        "pipelines/static_adapt/estimator_call_ledger.py",
        "pipelines/static_adapt/lane_routes.py",
        "pipelines/static_adapt/route_a_child_padding.py",
        "pipelines/static_adapt/runtime_split.py",
        "pipelines/static_adapt/selector_measurement_proxy.py",
        "pipelines/exact_bench/paper_i_hh_recovery_prefix_qiskit_sidecar.py",
    )
    missing_critical = [path for path in critical_corrected_files if path not in corrected_member_hashes]
    if missing_critical:
        raise ValueError(f"corrected source archive lacks critical files: {missing_critical!r}")
    critical_corrected_hashes = {
        path: corrected_member_hashes[path] for path in critical_corrected_files
    }
    pre_repair_corrected_diff = _pre_repair_corrected_diff(
        corrected_members=source_archive_manifest["members"]
    )
    _write_json(PRE_REPAIR_CORRECTED_DIFF, pre_repair_corrected_diff)
    preflight_gates = {
        "route_parity": _gate_evidence(ROUTE_PARITY_GATE),
        "corrected_enforcement_smoke": _gate_evidence(ENFORCEMENT_SMOKE_GATE),
    }
    commands_sha = _sha256_file(BASELINE_COMMANDS)
    LOCKED_COMMANDS_COPY.parent.mkdir(parents=True, exist_ok=True)
    LOCKED_COMMANDS_COPY.write_bytes(BASELINE_COMMANDS.read_bytes())
    if _sha256_file(LOCKED_COMMANDS_COPY) != commands_sha:
        raise ValueError("locked commands copy hash mismatch")
    baseline_rows = json.loads(BASELINE_COMMANDS.read_text(encoding="utf-8"))
    if not isinstance(baseline_rows, list):
        raise TypeError("baseline commands.json must contain a list")
    by_regime = {str(row["regime"]): row for row in baseline_rows}
    if tuple(by_regime) != EXPECTED_REGIMES or len(by_regime) != 6:
        raise ValueError(f"unexpected regime order/content: {tuple(by_regime)!r}")
    effective_route_audit = _effective_route_audit()

    source_rows: list[dict[str, Any]] = []
    job_paths: list[Path] = []
    argv_paths: list[Path] = []
    corrected_commands: list[list[str]] = []
    for regime in EXPECTED_REGIMES:
        row = by_regime[regime]
        baseline = [str(token) for token in row["argv"]]
        corrected, paths = _derive_corrected_argv(baseline=baseline, regime=regime)
        diff = _setting_diff(baseline, corrected)
        _validate_allowed_diff(diff, paths=paths)
        slug = _slug(regime)
        source_result = Path(str(row["output_json"]))
        if not source_result.is_file():
            raise FileNotFoundError(f"missing historical source result: {source_result}")
        argv_path = jobs_dir / f"{slug}.args.json"
        _write_json(
            argv_path,
            {
                "schema": "paper_i_hh_visible_snake_recovery_argv_v1",
                "bundle_id": BUNDLE_ID,
                "regime": regime,
                "argv": corrected,
            },
        )
        job_path = jobs_dir / f"{slug}.json"
        cache_environment = _cache_environment(slug)
        job_manifest = {
            "schema": "paper_i_hh_visible_snake_recovery_job_manifest_v1",
            "bundle_id": BUNDLE_ID,
            "job_id": f"{BUNDLE_ID}__{slug}",
            "regime": regime,
            "regime_slug": slug,
            "source_lock": {
                "historical_commands_json": LOCKED_COMMANDS_COPY.as_posix(),
                "historical_commands_sha256": commands_sha,
                "historical_commands_source_path": BASELINE_COMMANDS.as_posix(),
                "historical_result_json": source_result.as_posix(),
                "historical_result_sha256": _sha256_file(source_result),
                "comparison_json": (bundle / "source_lock_comparison.json").as_posix(),
                "historical_code_identity_status": "unresolved_full_pipelines_snapshot_missing",
                "corrected_critical_source_sha256": critical_corrected_hashes,
            },
            "command": {
                "historical_argv": baseline,
                "corrected_argv": corrected,
                "corrected_shell_display_only": shlex.join(corrected),
                "historical_normalized_options": _normalized_options(baseline),
                "corrected_normalized_options": _normalized_options(corrected),
                "allowlisted_differences": diff,
                "argv_json": argv_path.as_posix(),
                "argv_json_sha256": _sha256_file(argv_path),
            },
            "effective_route_lock": effective_route_audit,
            "approved_source_semantic_changes": {
                "fixed_sector_hard_guard": {
                    "argv_change": False,
                    "existing_option": "--phase3-runtime-split-child-set-symmetry-policy=hard_guard",
                    "source_behavior_change": "hard_guard now requires and executes fixed N_up=1,N_down=1 sector checks",
                },
                "runtime_child_binary_padding": {
                    "argv_change": True,
                    "option": "--phase3-runtime-split-child-padding-policy",
                    "value": PADDING_POLICY,
                },
            },
            "telemetry_additions": {
                "current_checkpoint": paths["current_json"],
                "estimator_call_ledger": paths["estimator_call_ledger_json"],
                "signed_active_prefix_checkpoints": "embedded in current/result JSON telemetry",
            },
            "paths": paths,
            "environment": cache_environment,
            "environment_contract": (
                "historical disk cache modes retained; cache directories are clean, isolated per job, "
                "not transferred in, and not output artifacts"
            ),
            "source_archive": {
                "path": SOURCE_ARCHIVE.as_posix(),
                "sha256": source_archive_manifest["archive_sha256"],
                "manifest": (bundle / "source_archive_manifest.json").as_posix(),
            },
            "execution_image": {
                "path": IMAGE.as_posix(),
                "sha256": IMAGE_SHA256,
                "lock_scope": "verified on CHTC submit host before bundle finalization",
            },
            "resources": {
                "request_cpus": CPUS,
                "request_memory_mb": MEMORY_MB,
                "request_disk_mb": DISK_MB,
                "max_runtime_seconds": MAX_RUNTIME_SECONDS,
                "scale_reference": "chtc/phase3_optuna/submit_paper_i_hh_all_regime_snake_mechanism_ablation_20260709_v1.sub and its queue",
            },
        }
        _write_json(job_path, job_manifest)
        job_paths.append(job_path)
        argv_paths.append(argv_path)
        corrected_commands.append(corrected)
        source_rows.append(
            {
                "regime": regime,
                "regime_slug": slug,
                "historical_argv": baseline,
                "corrected_argv": corrected,
                "historical_normalized_options": _normalized_options(baseline),
                "corrected_normalized_options": _normalized_options(corrected),
                "differences": diff,
                "allowlist_passed": True,
                "historical_result_json": source_result.as_posix(),
                "historical_result_sha256": _sha256_file(source_result),
                "corrected_paths": paths,
                "job_manifest": job_path.as_posix(),
                "job_manifest_sha256": _sha256_file(job_path),
                "argv_json": argv_path.as_posix(),
                "argv_json_sha256": _sha256_file(argv_path),
            }
        )

    source_lock = {
        "schema": "paper_i_hh_visible_snake_symmetry_padding_source_lock_v1",
        "bundle_id": BUNDLE_ID,
        "historical_commands_json": LOCKED_COMMANDS_COPY.as_posix(),
        "historical_commands_source_path": BASELINE_COMMANDS.as_posix(),
        "historical_commands_sha256": commands_sha,
        "corrected_output_root": OUTPUT_ROOT.as_posix(),
        "row_count": 6,
        "regime_order": list(EXPECTED_REGIMES),
        "approved_difference_contract": {
            "replaced_options": ["--output-json"],
            "added_output_or_telemetry_options": [
                "--adapt-current-json",
                "--adapt-estimator-call-ledger-json",
            ],
            "added_enforcement_options": ["--phase3-runtime-split-child-padding-policy"],
            "existing_enforcement_option_whose_source_behavior_is_repaired": [
                "--phase3-runtime-split-child-set-symmetry-policy=hard_guard"
            ],
            "all_other_argv_settings_identical": True,
            "scope_note": (
                "This proves argv/effective-route invariants, not byte-for-byte historical code identity; "
                "the July-8 pipelines snapshot is not preserved."
            ),
        },
        "effective_route_invariants": effective_route_audit,
        "preflight_gates": preflight_gates,
        "code_level_source_lock": {
            "historical": historical_code_evidence,
            "pre_repair_remote_snapshot": {
                "path": PRE_REPAIR_REMOTE_SNAPSHOT.as_posix(),
                "sha256": PRE_REPAIR_REMOTE_SNAPSHOT_SHA256,
                "current_vs_snapshot_diff": PRE_REPAIR_CORRECTED_DIFF.as_posix(),
                "current_vs_snapshot_diff_sha256": _sha256_file(PRE_REPAIR_CORRECTED_DIFF),
                "classification_counts": pre_repair_corrected_diff[
                    "classification_counts"
                ],
                "scope_caveat": pre_repair_corrected_diff["scope_caveat"],
                "execute_node_input": False,
            },
            "corrected_archive_manifest": (bundle / "source_archive_manifest.json").as_posix(),
            "corrected_archive_sha256": source_archive_manifest["archive_sha256"],
            "corrected_regular_file_count": source_archive_manifest["regular_file_count"],
            "corrected_member_inventory_sha256": source_archive_manifest[
                "member_inventory_sha256"
            ],
            "corrected_critical_source_sha256": critical_corrected_hashes,
            "corrected_all_per_file_hashes_location": (
                f"{(bundle / 'source_archive_manifest.json').as_posix()}#members"
            ),
            "classification": "argv_source_locked_but_full_historical_code_diff_unresolved",
        },
        "cache_contract": {
            "historical_argv_explicitly_enabled_cache": False,
            "historical_resolved_defaults": CACHE_MODE_ENVIRONMENT,
            "historical_execution_evidence": {
                "path": CACHE_EVIDENCE_LOG.as_posix(),
                "sha256": CACHE_EVIDENCE_LOG_SHA256,
                "line_evidence": {
                    "4": "hardcoded_adapt_pool_cache_hit cache_level=disk cache_scope=exact",
                    "6": "hardcoded_adapt_generator_registry_cache_hit cache_level=disk",
                    "22": "candidate_record_cache_mode=disk with disk hits",
                },
            },
            "corrected_environment_by_regime": {
                _slug(regime): _cache_environment(_slug(regime))
                for regime in EXPECTED_REGIMES
            },
            "initial_cache_state": "empty_job_local_directories",
            "stale_cache_transfer": False,
            "corrected_cache_location_classification": "performance_plumbing_only_under_current_keyed_cache_contract",
            "historical_code_cache_provenance": "unresolved",
            "known_snapshot_vs_july8_depth1": {
                "resolved_caps_match": True,
                "selected_admission_match": True,
                "energy_match": True,
                "shortlisted_label_overlap": "6/8",
                "july8_candidate_cache": "46 disk hits",
                "july8_runtime_child_set_scores": 38,
                "pre_repair_snapshot_clean_cache": "52 misses",
                "pre_repair_snapshot_runtime_child_set_scores": 44,
                "classification": "unresolved_historical_code_or_cache_provenance",
            },
            "reason": (
                "A valid keyed cache hit returns the same serialized pool/registry/candidate record; "
                "a clean-cache miss recomputes and stores that record. Cache location and initial warmth "
                "are performance plumbing under the corrected current code. The preserved evidence does "
                "not isolate whether the July-8 shortlist mismatch arose from unpreserved pipelines code, "
                "the historical warm cache, or both, so exact historical cache/code parity remains unresolved."
            ),
        },
        "source_archive": {
            "path": SOURCE_ARCHIVE.as_posix(),
            "sha256": source_archive_manifest["archive_sha256"],
            "manifest": (bundle / "source_archive_manifest.json").as_posix(),
        },
        "execution_image": {
            "path": IMAGE.as_posix(),
            "sha256": IMAGE_SHA256,
            "lock_scope": "verified on CHTC submit host before bundle finalization",
        },
        "rows": source_rows,
        "global_allowlist_passed": all(bool(row["allowlist_passed"]) for row in source_rows),
    }
    _write_json(bundle / "source_lock_comparison.json", source_lock)

    queue_lines = [
        "\t".join(
            [
                _slug(regime),
                (jobs_dir / f"{_slug(regime)}.json").as_posix(),
                str(MEMORY_MB),
                str(DISK_MB),
            ]
        )
        for regime in EXPECTED_REGIMES
    ]
    _write_text(bundle / "queue.tsv", "\n".join(queue_lines) + "\n")
    _write_text(
        bundle / "submit.sub",
        _submit_text(
            archive_sha256=str(source_archive_manifest["archive_sha256"]),
            job_manifest_paths=job_paths,
            argv_paths=argv_paths,
        ),
    )

    generated_runtime_paths = [
        bundle / "execute_source_locked_job.sh",
        bundle / "run_job.py",
        bundle / "submit.sub",
        bundle / "queue.tsv",
        bundle / "source_lock_comparison.json",
        bundle / "source_archive_manifest.json",
        LOCKED_COMMANDS_COPY,
        *job_paths,
        *argv_paths,
    ]
    runtime_bundle_machine_local_audit = _scan_machine_local_dependencies(
        generated_runtime_paths
    )
    source_archive_machine_local_audit = _archive_machine_local_dependency_audit()
    output_isolation_audit = _output_isolation_audit()
    dataless_audit = _dataless_audit()
    cli_audit = _cli_parse_audit(corrected_commands)
    image_audit = {
        "path": IMAGE.as_posix(),
        "present_in_local_checkout": IMAGE.is_file(),
        "expected_sha256": IMAGE_SHA256,
        "local_sha256": _sha256_file(IMAGE) if IMAGE.is_file() else None,
        "local_hash_matches": _sha256_file(IMAGE) == IMAGE_SHA256 if IMAGE.is_file() else None,
        "remote_submit_host_verified": True,
        "remote_verified_size": "207M",
        "remote_path_contract": "repo-relative under /home/jsstrobel/Holstein_phase3_optuna_chtc",
        "required_remote_pre_submit_check": "rehash immediately before condor_submit and require exact match",
    }
    submit_text = (bundle / "submit.sub").read_text(encoding="utf-8")
    submit_audit = {
        "relative_paths_only": not any(token in submit_text for token in ("/Users/", "file://", "~/", "/Volumes/")),
        "exact_archive_path_present": SOURCE_ARCHIVE.as_posix() in submit_text,
        "exact_image_path_present": IMAGE.as_posix() in submit_text,
        "exact_image_hash_present": IMAGE_SHA256 in submit_text,
        "archive_glob_absent": "*src_sanitized" not in submit_text and "ls " not in submit_text,
        "resource_scale": {
            "cpus": CPUS,
            "memory_mb": MEMORY_MB,
            "disk_mb": DISK_MB,
            "max_runtime_seconds": MAX_RUNTIME_SECONDS,
        },
        "queue_rows": len(queue_lines),
    }
    bundle_manifest_paths = [
        SOURCE_ARCHIVE,
        bundle / "source_archive_manifest.json",
        bundle / "source_lock_comparison.json",
        bundle / "submit.sub",
        bundle / "queue.tsv",
        bundle / "execute_source_locked_job.sh",
        bundle / "run_job.py",
        bundle / "build_bundle.py",
        bundle / "build_route_parity_evidence.py",
        bundle / "build_enforcement_smoke_evidence.py",
        PRE_REPAIR_REMOTE_SNAPSHOT,
        PRE_REPAIR_CORRECTED_DIFF,
        CACHE_EVIDENCE_LOG,
        HISTORICAL_SRC_ARCHIVE_JULY8,
        HISTORICAL_SRC_ARCHIVE_JULY9,
        LOCKED_COMMANDS_COPY,
        *sorted(
            path
            for path in (bundle / "preflight_evidence").glob("*")
            if path.is_file()
        ),
        *job_paths,
        *argv_paths,
    ]
    bundle_manifest = {
        "schema": "paper_i_hh_visible_snake_recovery_bundle_manifest_v1",
        "bundle_id": BUNDLE_ID,
        "output_root": OUTPUT_ROOT.as_posix(),
        "files": [
            {
                "path": path.as_posix(),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in bundle_manifest_paths
        ],
    }
    _write_json(bundle / "bundle_manifest.json", bundle_manifest)

    blockers: list[str] = []
    if not dataless_audit["passed"]:
        blockers.append("source tree contains dataless files or the dataless audit failed")
    if not cli_audit["passed"]:
        blockers.append("one or more corrected argv arrays fail current CLI parsing")
    if not effective_route_audit["passed"]:
        blockers.append(
            "current source does not resolve the historical July-8 physical-lane shortlist budgets"
        )
    if not all(bool(gate["passed"]) for gate in preflight_gates.values()):
        blockers.append("one or more required depth-1 preflight evidence gates failed")
    if not runtime_bundle_machine_local_audit["passed"]:
        blockers.append("runtime bundle contains a machine-local dependency string")
    if not source_archive_machine_local_audit["passed"]:
        blockers.append(
            "source archive contains a machine-local string in the static adapt_pipeline runtime import closure"
        )
    if not output_isolation_audit["passed"]:
        blockers.append("one or more production regime output targets already contains files")
    if image_audit["present_in_local_checkout"] and not image_audit["local_hash_matches"]:
        blockers.append("local execution image exists but does not match the locked CHTC image SHA-256")
    if (
        not submit_audit["relative_paths_only"]
        or not submit_audit["exact_archive_path_present"]
        or not submit_audit["exact_image_path_present"]
        or not submit_audit["exact_image_hash_present"]
        or not submit_audit["archive_glob_absent"]
    ):
        blockers.append("Condor submit/source-archive path audit failed")
    preflight = {
        "schema": "paper_i_hh_visible_snake_recovery_preflight_v1",
        "bundle_id": BUNDLE_ID,
        "status": "blocked" if blockers else "ready_for_authorized_submission",
        "submission_performed": False,
        "blockers": blockers,
        "checkout_guard": checkout_guard,
        "source_archive": source_archive_manifest,
        "dataless_audit": dataless_audit,
        "runtime_bundle_machine_local_dependency_audit": runtime_bundle_machine_local_audit,
        "source_archive_machine_local_dependency_audit": source_archive_machine_local_audit,
        "output_isolation_audit": output_isolation_audit,
        "cli_parse_audit": cli_audit,
        "effective_route_audit": effective_route_audit,
        "preflight_gates": preflight_gates,
        "submit_audit": submit_audit,
        "image_audit": image_audit,
        "source_lock_global_allowlist_passed": source_lock["global_allowlist_passed"],
        "code_level_source_lock": source_lock["code_level_source_lock"],
        "bundle_manifest": {
            "path": (bundle / "bundle_manifest.json").as_posix(),
            "sha256": _sha256_file(bundle / "bundle_manifest.json"),
        },
        "required_remote_pre_submit_checks": [
            f"require sha256sum {IMAGE.as_posix()} == {IMAGE_SHA256}",
            f"condor_submit -dry-run /tmp/{BUNDLE_ID}.dryrun.log {(bundle / 'submit.sub').as_posix()}",
            "confirm queue row count is six and no unrelated jobs are modified or cancelled",
        ],
    }
    _write_json(bundle / "preflight_report.json", preflight)
    print(json.dumps({"status": preflight["status"], "blockers": blockers, "bundle": bundle.as_posix()}, indent=2))
    return 0 if not blockers else 3


if __name__ == "__main__":
    raise SystemExit(main())
