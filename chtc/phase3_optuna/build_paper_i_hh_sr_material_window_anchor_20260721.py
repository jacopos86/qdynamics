#!/usr/bin/env python3
"""Build the immutable source-value anchor for the Phase-III material window.

The one-row anchor executes the validated full-geometry/no-overlap parent route
under the exact source archive that will later execute the material-window
candidate.  The candidate route is present in the archive but is not executed.
The six-row fanout therefore stays fail closed until an external fetched anchor
is shown to reproduce the locked parent trajectory exactly.
"""

from __future__ import annotations

import copy
import ast
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "chtc/phase3_optuna/input"
BASE_ID = "paper_i_hh_sr_snake_no_overlap_trust_all_six_r50_20260720_v4_chtc"
BASE_BATCH = "paper-i-hh-sr-no-overlap-trust-six-r50-20260720-v4"
BASE = INPUT / BASE_ID
ANCHOR_ID = (
    "paper_i_hh_sr_snake_no_overlap_trust_material_window_parent_anchor_"
    "weak_weak_r50_20260721_v4_chtc"
)
ANCHOR_BATCH = "paper-i-hh-sr-material-window-parent-anchor-ww-r50-20260721-v4"
ANCHOR = INPUT / ANCHOR_ID

BASE_SOURCE_SHA256 = "8f5f88aa18a529906bbf6861d8a2411a3e3c3f0a0e9f7091fbe0a0d8afb443cc"
IMMUTABLE_OVERLAY_ID = (
    "paper_i_hh_sr_snake_no_overlap_trust_material_window_parent_anchor_"
    "weak_weak_r50_20260721_v2_chtc"
)
IMMUTABLE_OVERLAY_ARCHIVE = INPUT / IMMUTABLE_OVERLAY_ID / "source_locked.tar.gz"
IMMUTABLE_OVERLAY_SHA256 = (
    "7b4526fcd26655512dfd219dc53d3ab8ba064a79a1385985eb24dd263639458d"
)
EXPECTED_RECONSTRUCTED_SOURCE_SHA256 = (
    "ced6b10d6bfbe4ae6a54495ff2ef4747a90036fa2027b0386555d016d5869a05"
)

PARENT_ALIAS = "sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1"
PARENT_DIGEST = "fd5ec3fa2c98b2a9d1cbcc304241d723f57dbd6210f4ea2daf30753603a146c2"
CHILD_ALIAS = (
    "sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_"
    "material_window_v1"
)
CHILD_DIGEST = "9d6cfae3fda84eb6a24232358c45a8f25b42c6147bfa4250905910eff687a417"
PARENT_SCOPE = "full_active_plus_singleton_v1"
CHILD_SCOPE = "candidate_material_coupling_window_v1"
IMAGE_SHA256 = "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"

SOURCE_TRANSFER = ROOT / (
    "raw_outputs/chtc_fetch_paper_i_hh_sr_20260721/priority_20260721T0037Z/"
    "8958273.0__weak_weak_transfer.tar.gz"
)
SOURCE_VALIDATION = ROOT / (
    "raw_outputs/chtc_fetch_paper_i_hh_sr_20260721/priority_20260721T0037Z/"
    "8958273.0__weak_weak_validation.json"
)
SOURCE_RESULT_MEMBER = (
    "raw_outputs/paper_i_hh_sr_snake_no_overlap_trust_all_six_r50_"
    "20260720_v4_chtc/weak_weak/json/result.json"
)
RECOVERED_VALIDATOR = ROOT / (
    "raw_outputs/chtc_fetch_paper_i_hh_sr_20260721/"
    "no_overlap_8958273_weak_cutoff_reporting_recovery/source/"
    "chtc/phase3_optuna/input/"
    "paper_i_hh_sr_snake_no_overlap_trust_all_six_r50_20260720_v4_chtc/"
    "evidence_validation.py"
)

THRESHOLD_SOURCE_ARCHIVES = (
    "raw_outputs/chtc_fetch_paper_i_hh_sr_20260721/priority_20260721T0037Z/"
    "8958273.0__weak_weak_transfer.tar.gz",
    "raw_outputs/chtc_fetch_paper_i_hh_sr_20260721/priority_20260721T0037Z/"
    "8958273.1__intermediate_weak_transfer.tar.gz",
    "raw_outputs/chtc_fetch_paper_i_hh_sr_20260721/priority_20260721T0037Z/"
    "8958273.2__strong_weak_u8_transfer.tar.gz",
    "raw_outputs/chtc_fetch_paper_i_hh_sr_20260721/status_20260721T0732Z/"
    "8958273.3__weak_strong_transfer.tar.gz",
    "raw_outputs/chtc_fetch_paper_i_hh_sr_20260721/status_20260721T0502Z/"
    "8958273.4__intermediate_strong_transfer.tar.gz",
    "raw_outputs/chtc_fetch_paper_i_hh_sr_20260721/status_20260721T0149Z/"
    "8958273.5__strong_strong_u8_transfer.tar.gz",
)

PRODUCTION_OVERLAYS = (
    "pipelines/static_adapt/phase3_material_window.py",
    "pipelines/scaffold/hh_continuation_scoring.py",
    "pipelines/static_adapt/sr_snake_route_profile.py",
    "pipelines/static_adapt/selector_candidate_metadata.py",
    "pipelines/static_adapt/adapt_pipeline.py",
    "pipelines/static_adapt/route_a_trust_region.py",
)
FOCUSED_TEST_OVERLAYS = (
    "test/test_static_adapt_phase3_material_window.py",
    "test/test_static_adapt_phase3_material_window_scoring.py",
    "test/test_static_adapt_phase3_material_window_route_profile.py",
    "test/test_static_adapt_phase3_material_window_runtime_contract.py",
    "test/test_static_adapt_route_a_trust_region.py",
)
OVERLAY_FILES = PRODUCTION_OVERLAYS + FOCUSED_TEST_OVERLAYS

# The v2 source archive accidentally copied the whole live adapt pipeline.  The
# entries below are the reviewed Test-1 hunks from the diff between the exact
# no-overlap parent archive and a v2 overlay with the unrelated helper and the
# Test-2 prune-verification lines removed.  Both old-line and full hunk hash are
# required so an unrelated live/source change fails closed.
ADAPT_TEST1_HUNK_ALLOWLIST = (
    (192, "241a29b9b9ecf3ae24f739e2841d331ea1d7284fa554d1dc3a162aef0db7406e"),
    (227, "6b9966ce1b9c3c868bc083c059e411d05da9020c782b09e869c230f6e42115db"),
    (843, "8c5fa99b1b99ee1793a72e5140ad2ae839a4792025ea4887d870ee92029b379d"),
    (859, "9f412f4a432943073963b1435912049ce2a873961a06d4d02f083bcbe4f281ad"),
    (867, "c4ac7707a4b3494bcf2736d13c0687959a1ad7e5ad63252e2afec6058a6f5735"),
    (1977, "70db49a0df20a1f557cbc63f4f768098f6c19e52a2d05e242aed3be1e015a6e4"),
    (3180, "6942a2733fd2562d30de791be8cbff520bcc79d65e3fe3c3e84f560eeee672cd"),
    (4092, "1f8a482a1884626ca4fdc1cdc1a83f1742a65e8de58e94b47afc823bc27c6cac"),
    (4115, "d41b4a9182456eb6de43e2055cb0fdc8342bbb6e0615c043b3b81daa622a75f7"),
    (5746, "755f68f1be0b620686e893b5f755501b5025c421c3c310d4d054096a660723d7"),
    (6157, "8155efc917f48d292ebf223871c30f994a44a58c4028bc5e371285490da21ffc"),
    (6193, "f2d663cf88e7b114f9b657fae6de669c1b6996b9a4159dfaf5eaab094e326013"),
    (11094, "f51bf78532ca114f7352c580087b435c70907e9ea4a715351dd945928412267a"),
    (13134, "ca4ec85134df344eb6c98bb3886132d8b0ce01eba2b6523973da357738206a11"),
    (43273, "236ad5cc659e9447b58b236a1d128e5568af452bb0168ab18f50136eb2979a03"),
    (43405, "8710e54b1914b4ec7c1d67890b4dcc7caeb02efdf373029b27933defcdd762f6"),
    (43642, "66a55e7f717631faec8d8ca39fae44698e0fbd617fb618a7d3f777f95a9421aa"),
    (43659, "286b0d3eb3b6ab1216e85348b59a5fb3fa63caa2e48a3fa1adaaf16c9c768e06"),
    (43685, "2f3171ef2bb08e7cc947bfd184afb21ed8ec88137ace0d251576ef1d0aa50959"),
    (43696, "d979762f9dddb107abc1ef18498d7ddddf5e9fd9b20e42769b6a4e0ec6ef32fc"),
    (43719, "6a7bd6842a378db474cadf32037eea14c8c38d2addd391e78150837c8425d14f"),
    (44244, "fa53eec8830851284759afbc44aacac9077ae20e8b6c5cf140ed6f62f4edbfca"),
    (44258, "9251c65391974a4e74d09983bf1e72e2c3a5ee3aeb130b4724e4d810d3fe2de3"),
    (47350, "6fdadcb0547f7af04ab3d0f03411182ea05c5f7724de167988f7c1a3c10fc6f8"),
    (47387, "36eba8f5740c24d3e359dd1027d62f409933c71562caa7b2b8c2e182b296cd10"),
    (47432, "8b94b61c109e27438e851e8d3b151db06060b086fe17f54badd67950420eafc0"),
    (47478, "394cd240fbe92bbf29cb173062d6cd679950f6239c69ab26f8f8735317f267db"),
    (47511, "caec80aa8f3699b8d83506ff37054a4e53ddae5e748eb39727331ff395867cad"),
)

FORBIDDEN_ADAPT_PARENT_DRIFT_MARKERS = (
    "def _resolve_parent_sector_filter_policy(",
    "complete authoritative Phase-II population",
    "def _phase2_geometry_payload_in_anchor_order(",
    "MATERIAL_WINDOW_FS_PRUNE_VERIFY",
)

PROTECTED_PARENT_FILES = (
    "pipelines/static_adapt/estimator_call_ledger.py",
    "pipelines/static_adapt/formal_manifold_warm_start.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def replace_tree(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: replace_tree(item, replacements) for key, item in value.items()}
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


def extract_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(destination, filter="data")


def _top_level_symbol_ranges(
    text: str,
    *,
    predicate,
) -> list[tuple[int, int]]:
    tree = ast.parse(text)
    ranges: list[tuple[int, int]] = []
    for node in tree.body:
        names: list[str] = []
        if isinstance(node, ast.Assign):
            names.extend(
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(node.name)
        if any(predicate(name) for name in names):
            if node.end_lineno is None:
                raise ValueError(f"missing AST end line for {names}")
            ranges.append((int(node.lineno), int(node.end_lineno)))
    return ranges


def _delete_line_ranges(text: str, ranges: list[tuple[int, int]]) -> str:
    lines = text.splitlines(keepends=True)
    for start, end in sorted(ranges, reverse=True):
        del lines[start - 1 : end]
    return "".join(lines)


def sanitize_adapt_test1_overlay(text: str) -> str:
    """Remove non-Test-1 additions before the reviewed hunk selection."""

    text = _delete_line_ranges(
        text,
        _top_level_symbol_ranges(
            text,
            predicate=lambda name: name == "_phase2_geometry_payload_in_anchor_order",
        ),
    )
    lines = [
        line
        for line in text.splitlines(keepends=True)
        if "MATERIAL_WINDOW_FS_PRUNE_VERIFY" not in line
    ]
    sanitized = "".join(lines)
    ast.parse(sanitized)
    return sanitized


def unified_hunks(base_text: str, overlay_text: str) -> list[dict[str, Any]]:
    # Use the platform unified-diff implementation that produced the reviewed
    # hunk hashes.  Python's SequenceMatcher enables an autojunk heuristic on
    # this very large/repetitive module and merges several independent hunks.
    with tempfile.TemporaryDirectory(prefix="material-window-reviewed-diff-") as raw:
        root = Path(raw)
        parent = root / "parent.py"
        overlay = root / "overlay.py"
        parent.write_text(base_text, encoding="utf-8")
        overlay.write_text(overlay_text, encoding="utf-8")
        completed = subprocess.run(
            [
                "diff",
                "-u",
                "--label",
                "parent",
                "--label",
                "overlay",
                str(parent),
                str(overlay),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    if completed.returncode not in {0, 1}:
        raise RuntimeError(f"unified diff failed: {completed.stderr}")
    lines = completed.stdout.splitlines(keepends=True)
    hunks: list[list[str]] = []
    current: list[str] = []
    for line in lines[2:]:
        if line.startswith("@@"):
            if current:
                hunks.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        hunks.append(current)
    parsed: list[dict[str, Any]] = []
    pattern = re.compile(
        r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@"
    )
    for hunk in hunks:
        match = pattern.match(hunk[0])
        if match is None:
            raise ValueError(f"malformed unified hunk header: {hunk[0]!r}")
        payload = "".join(hunk).encode("utf-8")
        parsed.append(
            {
                "old_start": int(match.group(1)),
                "old_count": int(match.group(2) or 1),
                "new_start": int(match.group(3)),
                "new_count": int(match.group(4) or 1),
                "sha256": sha256_bytes(payload),
                "lines": hunk,
            }
        )
    return parsed


def apply_unified_hunks(base_text: str, hunks: list[dict[str, Any]]) -> str:
    base_lines = base_text.splitlines(keepends=True)
    output: list[str] = []
    cursor = 0
    for hunk in sorted(hunks, key=lambda item: int(item["old_start"])):
        start = int(hunk["old_start"]) - 1
        if start < cursor:
            raise ValueError("selected unified hunks overlap or are out of order")
        output.extend(base_lines[cursor:start])
        index = start
        for line in list(hunk["lines"])[1:]:
            prefix = line[:1]
            payload = line[1:]
            if prefix == " ":
                if index >= len(base_lines) or base_lines[index] != payload:
                    raise ValueError("selected hunk context differs from parent source")
                output.append(payload)
                index += 1
            elif prefix == "-":
                if index >= len(base_lines) or base_lines[index] != payload:
                    raise ValueError("selected hunk removal differs from parent source")
                index += 1
            elif prefix == "+":
                output.append(payload)
            elif prefix == "\\":
                continue
            else:
                raise ValueError(f"unexpected unified hunk line: {line!r}")
        cursor = index
    output.extend(base_lines[cursor:])
    result = "".join(output)
    ast.parse(result)
    return result


def reconstruct_adapt_test1(parent_path: Path, overlay_path: Path) -> dict[str, Any]:
    parent = parent_path.read_text(encoding="utf-8")
    overlay = sanitize_adapt_test1_overlay(
        overlay_path.read_text(encoding="utf-8")
    )
    available = unified_hunks(parent, overlay)
    by_identity = {
        (int(hunk["old_start"]), str(hunk["sha256"])): hunk
        for hunk in available
    }
    required = set(ADAPT_TEST1_HUNK_ALLOWLIST)
    missing = sorted(required.difference(by_identity))
    if missing:
        raise ValueError(f"reviewed Test-1 adapt hunks are missing: {missing}")
    selected = [by_identity[identity] for identity in ADAPT_TEST1_HUNK_ALLOWLIST]
    reconstructed = apply_unified_hunks(parent, selected)
    for marker in FORBIDDEN_ADAPT_PARENT_DRIFT_MARKERS:
        if marker in reconstructed:
            raise ValueError(f"forbidden parent/Test-2 drift entered adapt source: {marker}")
    parent_path.write_text(reconstructed, encoding="utf-8")
    return {
        "selection_policy": "exact_old_line_and_full_unified_hunk_sha256_allowlist_v1",
        "available_hunk_count_after_sanitization": len(available),
        "selected_hunk_count": len(selected),
        "selected_hunks": [
            {
                "old_start": int(hunk["old_start"]),
                "sha256": str(hunk["sha256"]),
            }
            for hunk in selected
        ],
    }


def sanitize_scoring_test1(text: str) -> str:
    """Keep acquisition telemetry absent on the exact parent path."""

    old = '        "acquisition_mode": str(acquisition_mode_key),\n'
    new = (
        '        **({"acquisition_mode": str(acquisition_mode_key)} '
        "if coupling_screen else {}),\n"
    )
    if text.count(old) != 1:
        raise ValueError("exact-insertion acquisition telemetry patch anchor drifted")
    result = text.replace(old, new, 1)
    ast.parse(result)
    return result


def sanitize_route_profile_test1(text: str) -> str:
    """Remove dormant Test-2 route definitions from the Test-1 archive."""

    result = _delete_line_ranges(
        text,
        _top_level_symbol_ranges(
            text,
            predicate=lambda name: "FS_PRUNE_VERIFY" in name.upper(),
        ),
    )
    snippets = (
        "    SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_ALIAS_V1,\n"
        "    SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1,\n",
        "    if requested == (\n"
        "        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1\n"
        "    ):\n"
        "        return canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_material_window_fs_prune_verify_v1_contract()\n",
        "        \"sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_material_window_fs_prune_verify_v1\": (\n"
        "            SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1\n"
        "        ),\n"
        "        \"symmetric_cost_projected_phase3_no_overlap_trust_material_window_fs_prune_verify_v1\": (\n"
        "            SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1\n"
        "        ),\n"
        "        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1: (\n"
        "            SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1\n"
        "        ),\n",
        "    elif requested == (\n"
        "        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1\n"
        "    ):\n"
        "        execution_settings = (\n"
        "            CANONICAL_SR_SNAKE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1_EXECUTION_SETTINGS\n"
        "        )\n",
        '    "CANONICAL_SR_SNAKE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1_EXECUTION_SETTINGS",\n',
        '    "SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_ALIAS_V1",\n'
        '    "SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1",\n',
        '    "canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_material_window_fs_prune_verify_v1_contract",\n'
        '    "canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_material_window_fs_prune_verify_v1_contract_sha256",\n',
    )
    for snippet in snippets:
        if result.count(snippet) != 1:
            raise ValueError(f"Test-2 route cleanup anchor drifted: {snippet[:80]!r}")
        result = result.replace(snippet, "", 1)
    standalone = (
        "        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_"
        "MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1,\n"
    )
    if result.count(standalone) != 4:
        raise ValueError("unexpected Test-2 standalone route reference count")
    result = result.replace(standalone, "")
    if "FS_PRUNE_VERIFY" in result or "fs_prune_verify" in result:
        raise ValueError("Test-2 route source remains after cleanup")
    ast.parse(result)
    return result


def route_profile_test1_source() -> str:
    return '''from __future__ import annotations

from pathlib import Path
import inspect
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.scaffold import hh_continuation_scoring as scoring
from pipelines.static_adapt import sr_snake_route_profile as profiles


def test_parent_and_test1_route_contracts_are_exact() -> None:
    assert profiles.canonical_sr_snake_contract_sha256(
        "sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1"
    ) == "fd5ec3fa2c98b2a9d1cbcc304241d723f57dbd6210f4ea2daf30753603a146c2"
    assert profiles.canonical_sr_snake_contract_sha256(
        "sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_material_window_v1"
    ) == "9d6cfae3fda84eb6a24232358c45a8f25b42c6147bfa4250905910eff687a417"


def test_test1_changes_only_phase3_response_scope() -> None:
    parent = profiles.canonical_sr_snake_contract(
        "sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1"
    )["execution_settings"]
    child = profiles.canonical_sr_snake_contract(
        "sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_material_window_v1"
    )["execution_settings"]
    changed = {key for key, value in child.items() if parent.get(key) != value}
    assert changed == {"phase3_response_coordinate_scope"}
    assert child["phase1_prune_enabled"] is False
    assert child["adapt_beam_live_branches"] == 1
    assert child["adapt_beam_children_per_parent"] == 1


def test_test2_route_is_absent_from_test1_archive() -> None:
    source = inspect.getsource(profiles)
    assert "FS_PRUNE_VERIFY" not in source
    assert "fs_prune_verify" not in source


def test_default_parent_geometry_payload_omits_new_acquisition_telemetry() -> None:
    source = inspect.getsource(scoring._exact_insertion_joint_geometry_payload)
    assert "if coupling_screen else {}" in source
    assert source.count('"acquisition_mode"') == 1
'''


def strip_bytecode(source: Path) -> None:
    for path in sorted(source.rglob("*.pyc")):
        path.unlink()
    for path in sorted(source.rglob("__pycache__"), reverse=True):
        if path.is_dir():
            shutil.rmtree(path)


def _slice_between(text: str, start: str, end: str) -> str:
    if text.count(start) != 1 or text.count(end) < 1:
        raise ValueError(f"protected parent slice anchors drifted: {start!r}, {end!r}")
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[begin:finish]


def assert_parent_compatibility(source: Path, parent: Path) -> dict[str, Any]:
    reconstructed = (source / "pipelines/static_adapt/adapt_pipeline.py").read_text(
        encoding="utf-8"
    )
    parent_text = (parent / "pipelines/static_adapt/adapt_pipeline.py").read_text(
        encoding="utf-8"
    )
    for marker in FORBIDDEN_ADAPT_PARENT_DRIFT_MARKERS:
        if marker in reconstructed:
            raise ValueError(f"forbidden parent drift marker present: {marker}")
    protected_slices = {
        "formal_selector_admission_population": (
            "                if formal_manifold_selector_enabled:\n"
            "                    # Pure FM consumes the last nonempty query-closed surface.",
            "                ordinary_admission_source_records_local = [",
        ),
    }
    slice_receipts: dict[str, Any] = {}
    for name, (start, end) in protected_slices.items():
        expected = _slice_between(parent_text, start, end)
        actual = _slice_between(reconstructed, start, end)
        if actual != expected:
            raise ValueError(f"protected parent block changed: {name}")
        slice_receipts[name] = {
            "sha256": sha256_bytes(actual.encode("utf-8")),
            "byte_identical_to_parent": True,
        }
    protected_files: dict[str, Any] = {}
    for relative in PROTECTED_PARENT_FILES:
        expected = sha256(parent / relative)
        actual = sha256(source / relative)
        if actual != expected:
            raise ValueError(f"protected parent file changed: {relative}")
        protected_files[relative] = {
            "sha256": actual,
            "byte_identical_to_parent": True,
        }
    return {
        "protected_slices": slice_receipts,
        "protected_files": protected_files,
        "forbidden_markers_absent": list(FORBIDDEN_ADAPT_PARENT_DRIFT_MARKERS),
    }


def read_json_member(archive_path: Path, member: str) -> tuple[dict[str, Any], str]:
    with tarfile.open(archive_path, "r:gz") as archive:
        extracted = archive.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"archive member missing: {member}")
        payload = extracted.read()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise TypeError(f"source result is not an object: {member}")
    return value, sha256_bytes(payload)


def result_signature(result: Mapping[str, Any]) -> dict[str, Any]:
    adapt = result.get("adapt_vqe", {})
    history = list(adapt.get("history", []))
    checkpoints = list(adapt.get("active_prefix_checkpoints", []))
    operators = list(adapt.get("operators", []))
    energies = [row.get("energy_after_opt") for row in history]
    checkpoint_hashes = [row.get("checkpoint_sha256") for row in checkpoints]
    if not (len(history) == len(checkpoints) == len(operators) == 50):
        raise ValueError("locked source does not contain exactly 50 complete rounds")
    return {
        "controller_rounds": 50,
        "operators": operators,
        "operator_sequence_sha256": sha256_bytes(
            json.dumps(operators, separators=(",", ":"), allow_nan=False).encode()
        ),
        "controller_energies": energies,
        "controller_energies_sha256": sha256_bytes(
            json.dumps(energies, separators=(",", ":"), allow_nan=False).encode()
        ),
        "checkpoint_sha256_sequence": checkpoint_hashes,
        "checkpoint_sequence_sha256": sha256_bytes(
            json.dumps(checkpoint_hashes, separators=(",", ":"), allow_nan=False).encode()
        ),
        "terminal_energy": adapt.get("energy"),
        "terminal_abs_delta_e": adapt.get("abs_delta_e"),
        "settings": result.get("settings", {}),
        "settings_sha256": sha256_bytes(
            json.dumps(
                result.get("settings", {}), sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ),
    }


def threshold_source_audit() -> dict[str, Any]:
    rows = []
    for relative in THRESHOLD_SOURCE_ARCHIVES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({
            "archive": relative,
            "archive_sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        })
    policy_source = ROOT / "pipelines/static_adapt/phase3_material_window.py"
    return {
        "schema": "paper_i_sr_material_window_threshold_source_audit_v1",
        "status": "source_locked_offline_replay_complete",
        "source_route_profile": PARENT_ALIAS,
        "source_route_contract_sha256": PARENT_DIGEST,
        "source_archives": rows,
        "policy_source": str(policy_source.relative_to(ROOT)),
        "policy_source_sha256": sha256(policy_source),
        "thresholds": {
            "gram_entry_threshold": 4.0e-3,
            "hessian_entry_threshold": 2.0e-22,
            "gram_omitted_l2_tolerance": 1.0,
            "hessian_omitted_l2_tolerance": 1.0,
            "gram_cross_block_tolerance": 1.0e-1,
            "hessian_cross_block_tolerance": 1.0e-1,
            "epsilon": 1.0e-12,
        },
        "offline_replay": {
            "feasibility_labels_compared": 506,
            "feasibility_labels_preserved": 506,
            "within_round_winner_order_comparisons": 300,
            "within_round_winner_order_comparisons_preserved": 300,
            "retained_coordinate_or_pair_fraction_before_closure": 0.9941,
            "closure_refresh_count": 28,
            "estimated_old_old_pair_savings": 221,
        },
        "interpretation": (
            "Conservative source-derived starting thresholds; closure and rank gates "
            "remain authoritative at runtime."
        ),
    }


def isolated_contracts(source: Path) -> dict[str, dict[str, Any]]:
    code = (
        "import json\n"
        "from pipelines.static_adapt.sr_snake_route_profile import "
        "canonical_sr_snake_contract, canonical_sr_snake_contract_sha256\n"
        f"aliases={[PARENT_ALIAS, CHILD_ALIAS]!r}\n"
        "print(json.dumps({a:{'digest':canonical_sr_snake_contract_sha256(a),"
        "'contract':canonical_sr_snake_contract(a)} for a in aliases},"
        "sort_keys=True))\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONNOUSERSITE", None)
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=source, env=env,
        check=True, capture_output=True, text=True,
    )
    value = json.loads(completed.stdout)
    expected = {PARENT_ALIAS: PARENT_DIGEST, CHILD_ALIAS: CHILD_DIGEST}
    actual = {alias: record["digest"] for alias, record in value.items()}
    if actual != expected:
        raise ValueError(f"isolated route digest drift: {actual}")
    return value


def build_source_archive(
    temp: Path,
) -> tuple[
    Path,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    base_archive = BASE / "source_locked.tar.gz"
    if sha256(base_archive) != BASE_SOURCE_SHA256:
        raise ValueError("exact no-overlap parent source archive hash drifted")
    if sha256(IMMUTABLE_OVERLAY_ARCHIVE) != IMMUTABLE_OVERLAY_SHA256:
        raise ValueError("immutable v2 material-window overlay archive hash drifted")
    source = temp / "source"
    overlay_source = temp / "immutable_v2_overlay"
    extract_archive(base_archive, source)
    extract_archive(IMMUTABLE_OVERLAY_ARCHIVE, overlay_source)
    overlays: dict[str, Any] = {}

    for relative in PRODUCTION_OVERLAYS:
        immutable = overlay_source / relative
        if not immutable.is_file():
            raise FileNotFoundError(immutable)
        target = source / relative
        before = sha256(target) if target.is_file() else None
        target.parent.mkdir(parents=True, exist_ok=True)
        reconstruction: dict[str, Any] = {
            "source": "immutable_v2_overlay_archive",
        }
        if relative == "pipelines/static_adapt/adapt_pipeline.py":
            reconstruction.update(reconstruct_adapt_test1(target, immutable))
        elif relative == "pipelines/scaffold/hh_continuation_scoring.py":
            target.write_text(
                sanitize_scoring_test1(immutable.read_text(encoding="utf-8")),
                encoding="utf-8",
            )
            reconstruction["parent_default_acquisition_telemetry"] = "omitted"
        elif relative == "pipelines/static_adapt/sr_snake_route_profile.py":
            target.write_text(
                sanitize_route_profile_test1(
                    immutable.read_text(encoding="utf-8")
                ),
                encoding="utf-8",
            )
            reconstruction["dormant_test2_route_definitions"] = "absent"
        else:
            shutil.copy2(immutable, target)
        overlays[relative] = {
            "classification": "material_window_test1_production_source",
            "parent_sha256": before,
            "overlay_sha256": sha256(target),
            "size_bytes": target.stat().st_size,
            "reconstruction": reconstruction,
        }

    for relative in FOCUSED_TEST_OVERLAYS:
        immutable = overlay_source / relative
        target = source / relative
        before = sha256(target) if target.is_file() else None
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative.endswith("phase3_material_window_route_profile.py"):
            target.write_text(route_profile_test1_source(), encoding="utf-8")
            test_source = "generated_fail_closed_test1_only_route_test"
        else:
            if not immutable.is_file():
                raise FileNotFoundError(immutable)
            shutil.copy2(immutable, target)
            test_source = "immutable_v2_overlay_archive"
        overlays[relative] = {
            "classification": "material_window_test1_focused_regression_test",
            "parent_sha256": before,
            "overlay_sha256": sha256(target),
            "size_bytes": target.stat().st_size,
            "reconstruction": {"source": test_source},
        }

    strip_bytecode(source)
    # assert_parent_compatibility needs the pristine parent tree, not the
    # modified output.  Re-extract that small frozen archive independently.
    pristine_parent = temp / "pristine_parent"
    extract_archive(base_archive, pristine_parent)
    strip_bytecode(pristine_parent)
    compatibility = assert_parent_compatibility(source, pristine_parent)

    allowed = set(OVERLAY_FILES)
    actual_changed: set[str] = set()
    all_files = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file()
    }.union(
        {
            path.relative_to(pristine_parent).as_posix()
            for path in pristine_parent.rglob("*")
            if path.is_file()
        }
    )
    for relative in all_files:
        actual = source / relative
        expected = pristine_parent / relative
        if not actual.is_file() or not expected.is_file() or sha256(actual) != sha256(expected):
            actual_changed.add(relative)
    if actual_changed != allowed:
        raise ValueError(
            "Test-1 source changed paths differ from the reviewed six-module/test "
            f"surface: extra={sorted(actual_changed - allowed)}, "
            f"missing={sorted(allowed - actual_changed)}"
        )
    compatibility["changed_paths"] = sorted(actual_changed)
    compatibility["production_changed_paths"] = sorted(PRODUCTION_OVERLAYS)
    compatibility["production_changed_path_count"] = len(PRODUCTION_OVERLAYS)

    contracts = isolated_contracts(source)
    strip_bytecode(source)
    output = temp / "source_locked.tar.gz"
    deterministic_archive(source, output)
    if sha256(output) != EXPECTED_RECONSTRUCTED_SOURCE_SHA256:
        raise ValueError("deterministic Test-1 reconstructed source hash drifted")
    return output, inventory(source), overlays, contracts, compatibility


def clean_inherited_bundle_state(bundle: Path) -> None:
    for name in (
        "archive_only_preflight.json", "preflight.json", "remote_execution_gate.json",
        "submission_artifact_hashes.json", "submission_gate.json",
        "upload_artifact_list.txt", "route_parity.json", "scientific_settings_audit.json",
        "source_locked_sensitivity_audit.json", "material_window_threshold_source_audit.json",
        "anchor_bundle_receipt.json", "fanout_bundle_receipt.json",
        "route_registration_repair.json", "submission_receipt.json",
        "remote_preflight_provenance_inventory.json",
        "post_submission_provenance_inventory.json",
    ):
        (bundle / name).unlink(missing_ok=True)
    for cache in bundle.rglob("__pycache__"):
        shutil.rmtree(cache)
    for bytecode in bundle.rglob("*.pyc"):
        bytecode.unlink()


def patch_text(path: Path, replacements: Mapping[str, str]) -> None:
    text = path.read_text(encoding="utf-8")
    for old, new in replacements.items():
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def submit_text(bundle_id: str, batch: str, archive_sha: str, queue: str) -> str:
    files = [
        "run_job.py", "evidence_validation.py", "validate_fetched.py",
        "source_archive_manifest.json", "source_revision_manifest.json",
        "physics_and_exact_reference_lock.json", "bundle_manifest.json",
        "source_locked_sensitivity_audit.json", "material_window_threshold_source_audit.json",
        "source_locked.tar.gz",
    ]
    inputs = [f"chtc/phase3_optuna/input/{bundle_id}/{name}" for name in files]
    inputs += ["$(job_manifest)", "$(normalized_manifest)", "chtc/phase3_optuna/image.sif"]
    return f'''universe = vanilla
executable = chtc/phase3_optuna/input/{bundle_id}/execute_source_locked_job.sh
arguments = $(job_manifest) chtc/phase3_optuna/input/{bundle_id}/source_locked.tar.gz {archive_sha} chtc/phase3_optuna/image.sif {IMAGE_SHA256} $(regime_slug)
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_executable = True
preserve_relative_paths = True
transfer_input_files = {", ".join(inputs)}
transfer_output_files = raw_outputs/{bundle_id}/$(regime_slug)_transfer.tar.gz
transfer_output_remaps = "raw_outputs/{bundle_id}/$(regime_slug)_transfer.tar.gz = $(Cluster).$(Process)__$(regime_slug)_transfer.tar.gz"
log = logs/{bundle_id}.$(Cluster).$(Process).log
output = logs/{bundle_id}.$(Cluster).$(Process).out
error = logs/{bundle_id}.$(Cluster).$(Process).err
requirements = TARGET.HasSIF
request_cpus = 4
request_memory = $(memory_mb)MB
request_disk = $(disk_mb)MB
+MaxRuntime = 259200
+JobBatchName = "{batch}"
notification = Never
queue regime_slug, job_manifest, normalized_manifest, memory_mb, disk_mb from {queue}
'''


def build_anchor() -> dict[str, Any]:
    if ANCHOR.exists():
        raise FileExistsError(f"immutable anchor already exists: {ANCHOR}")
    for required in (BASE, SOURCE_TRANSFER, SOURCE_VALIDATION, RECOVERED_VALIDATOR):
        if not required.exists():
            raise FileNotFoundError(required)
    validation = load(SOURCE_VALIDATION)
    if validation.get("result_sha256") is None:
        raise ValueError("source validation does not identify its result")
    source_result, source_result_sha = read_json_member(SOURCE_TRANSFER, SOURCE_RESULT_MEMBER)
    if source_result_sha != validation["result_sha256"]:
        raise ValueError("source transfer/result/validation hash mismatch")
    signature = result_signature(source_result)
    old_archive_sha = sha256(BASE / "source_locked.tar.gz")

    with tempfile.TemporaryDirectory(prefix="paper-i-material-window-anchor-") as raw:
        temp = Path(raw)
        archive, files, overlays, contracts, compatibility = build_source_archive(temp)
        archive_sha = sha256(archive)
        shutil.copytree(BASE, ANCHOR, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        clean_inherited_bundle_state(ANCHOR)
        shutil.copy2(archive, ANCHOR / "source_locked.tar.gz")

    replacements: dict[str, str] = {
        BASE_ID: ANCHOR_ID,
        BASE_BATCH: ANCHOR_BATCH,
        old_archive_sha: archive_sha,
    }
    for record in overlays.values():
        if record["parent_sha256"]:
            replacements[record["parent_sha256"]] = record["overlay_sha256"]

    overlay_receipt = {
        "schema": "paper_i_sr_material_window_source_overlay_v2",
        "parent_bundle": BASE_ID,
        "parent_source_archive_sha256": old_archive_sha,
        "immutable_reviewed_overlay_bundle": IMMUTABLE_OVERLAY_ID,
        "immutable_reviewed_overlay_source_archive_sha256": (
            IMMUTABLE_OVERLAY_SHA256
        ),
        "source_archive_sha256": archive_sha,
        "parent_route_contract_sha256": PARENT_DIGEST,
        "candidate_route_contract_sha256": CHILD_DIGEST,
        "overlay_files": overlays,
        "parent_compatibility": compatibility,
    }
    archive_manifest = replace_tree(load(BASE / "source_archive_manifest.json"), replacements)
    archive_manifest.update({
        "archive": f"chtc/phase3_optuna/input/{ANCHOR_ID}/source_locked.tar.gz",
        "archive_sha256": archive_sha,
        "archive_size_bytes": (ANCHOR / "source_locked.tar.gz").stat().st_size,
        "file_count": len(files),
        "files": files,
        "material_window_source_overlay": overlay_receipt,
    })
    dump(ANCHOR / "source_archive_manifest.json", archive_manifest)
    archive_manifest_sha = sha256(ANCHOR / "source_archive_manifest.json")
    revision = replace_tree(load(BASE / "source_revision_manifest.json"), replacements)
    revision["material_window_source_overlay"] = overlay_receipt
    dump(ANCHOR / "source_revision_manifest.json", revision)
    revision_sha = sha256(ANCHOR / "source_revision_manifest.json")
    physics = replace_tree(load(BASE / "physics_and_exact_reference_lock.json"), replacements)
    dump(ANCHOR / "physics_and_exact_reference_lock.json", physics)
    physics_sha = sha256(ANCHOR / "physics_and_exact_reference_lock.json")

    base_job = load(BASE / "jobs/weak_weak.json")
    job = replace_tree(copy.deepcopy(base_job), replacements)
    job["bundle_id"] = ANCHOR_ID
    job["batch_name"] = ANCHOR_BATCH
    job["source_lock"].update({
        "source_archive": f"chtc/phase3_optuna/input/{ANCHOR_ID}/source_locked.tar.gz",
        "source_archive_sha256": archive_sha,
        "source_archive_manifest": f"chtc/phase3_optuna/input/{ANCHOR_ID}/source_archive_manifest.json",
        "source_archive_manifest_sha256": archive_manifest_sha,
        "source_revision_manifest": f"chtc/phase3_optuna/input/{ANCHOR_ID}/source_revision_manifest.json",
        "source_revision_manifest_sha256": revision_sha,
        "physics_reference_lock": f"chtc/phase3_optuna/input/{ANCHOR_ID}/physics_and_exact_reference_lock.json",
        "physics_reference_lock_sha256": physics_sha,
        "material_window_source_overlay": overlay_receipt,
    })
    job["source_value_anchor"] = {
        "schema": "source_locked_sensitivity_anchor_plan_v1",
        "source_transfer_archive": str(SOURCE_TRANSFER.relative_to(ROOT)),
        "source_transfer_archive_sha256": sha256(SOURCE_TRANSFER),
        "source_result_archive_member": SOURCE_RESULT_MEMBER,
        "source_result_sha256": source_result_sha,
        "source_validation_receipt": str(SOURCE_VALIDATION.relative_to(ROOT)),
        "source_validation_receipt_sha256": sha256(SOURCE_VALIDATION),
        "swept_field": "phase3_response_coordinate_scope",
        "source_value": PARENT_SCOPE,
        "candidate_value": CHILD_SCOPE,
        "source_signature": signature,
        "fanout_allowed_before_anchor_pass": False,
    }
    dump(ANCHOR / "jobs/weak_weak.json", job)
    normalized = replace_tree(load(BASE / "normalized_manifests/weak_weak.json"), replacements)
    normalized.update({
        "bundle_id": ANCHOR_ID,
        "batch_name": ANCHOR_BATCH,
        "source_lock": copy.deepcopy(job["source_lock"]),
        "source_value_anchor": copy.deepcopy(job["source_value_anchor"]),
    })
    dump(ANCHOR / "normalized_manifests/weak_weak.json", normalized)
    for folder in (ANCHOR / "jobs", ANCHOR / "normalized_manifests"):
        for path in folder.glob("*.json"):
            if path.name != "weak_weak.json":
                path.unlink()

    shutil.copy2(RECOVERED_VALIDATOR, ANCHOR / "evidence_validation.py")
    for relative in ("run_job.py", "evidence_validation.py", "validate_fetched.py", "execute_source_locked_job.sh"):
        patch_text(ANCHOR / relative, replacements)
    queue_rel = f"chtc/phase3_optuna/input/{ANCHOR_ID}/queue.tsv"
    (ANCHOR / "queue.tsv").write_text(
        f"weak_weak\tchtc/phase3_optuna/input/{ANCHOR_ID}/jobs/weak_weak.json\t"
        f"chtc/phase3_optuna/input/{ANCHOR_ID}/normalized_manifests/weak_weak.json\t40960\t61440\n",
        encoding="utf-8",
    )
    (ANCHOR / "submit.sub").write_text(
        submit_text(ANCHOR_ID, ANCHOR_BATCH, archive_sha, queue_rel), encoding="utf-8"
    )
    threshold_audit = threshold_source_audit()
    dump(ANCHOR / "material_window_threshold_source_audit.json", threshold_audit)
    audit = {
        "schema": "source_locked_sensitivity_audit_v1",
        "source": {
            "method": "SR-SNAKE no-overlap full geometry",
            "regime_or_case": "weak_weak",
            "source_transfer_archive": str(SOURCE_TRANSFER.relative_to(ROOT)),
            "source_transfer_archive_sha256": sha256(SOURCE_TRANSFER),
            "source_result_archive_member": SOURCE_RESULT_MEMBER,
            "source_result_sha256": source_result_sha,
            "source_validation_receipt": str(SOURCE_VALIDATION.relative_to(ROOT)),
            "source_validation_receipt_sha256": sha256(SOURCE_VALIDATION),
            "route_or_profile_id": PARENT_ALIAS,
            "route_contract_sha256": PARENT_DIGEST,
            "source_variable_value": PARENT_SCOPE,
            "source_signature": signature,
        },
        "sweep": {
            "run_class": "candidate",
            "variable": "phase3_response_coordinate_scope",
            "grid": [PARENT_SCOPE, CHILD_SCOPE],
            "runner_mode": "direct_source_locked_replay",
            "baseline_materialization_status": "complete",
            "unresolved_source_fields": [],
            "fields_added_by_current_defaults": [],
        },
        "planned_rows": [{
            "value": PARENT_SCOPE,
            "changed_fields_vs_source": [],
            "non_swept_settings_diff": [],
            "bundle": ANCHOR_ID,
        }],
        "anchor": {
            "value": PARENT_SCOPE,
            "anchor_result_json": None,
            "anchor_reproduces_source": False,
            "operator_sequence_match": None,
            "controller_energy_history_exact_match": None,
            "checkpoint_sequence_match": None,
            "terminal_metric_exact_match": None,
            "settings_exact_match": None,
            "non_swept_settings_diff": [],
        },
        "fanout_authorized": False,
        "status": "anchor_pending",
    }
    dump(ANCHOR / "source_locked_sensitivity_audit.json", audit)
    receipt = {
        "schema": "paper_i_sr_material_window_parent_anchor_bundle_v1",
        "bundle_id": ANCHOR_ID,
        "batch_name": ANCHOR_BATCH,
        "source_archive_sha256": archive_sha,
        "source_archive_manifest_sha256": archive_manifest_sha,
        "source_revision_manifest_sha256": revision_sha,
        "physics_reference_lock_sha256": physics_sha,
        "parent_route_profile": contracts[PARENT_ALIAS]["contract"]["route_profile"],
        "parent_route_contract_sha256": PARENT_DIGEST,
        "candidate_route_profile": contracts[CHILD_ALIAS]["contract"]["route_profile"],
        "candidate_route_contract_sha256": CHILD_DIGEST,
        "job_count": 1,
        "fanout_authorized": False,
        "candidate_not_executed": True,
        "parent_compatibility": compatibility,
        "submission_performed": False,
    }
    dump(ANCHOR / "anchor_bundle_receipt.json", receipt)
    dump(ANCHOR / "bundle_manifest.json", receipt)
    dump(ANCHOR / "route_parity.json", {
        "schema": "paper_i_sr_material_window_anchor_route_parity_v1",
        "status": "pass",
        "anchor_changed_scientific_fields_vs_source": [],
        "candidate_swept_field": "phase3_response_coordinate_scope",
        "parent_route_contract_sha256": PARENT_DIGEST,
        "candidate_route_contract_sha256": CHILD_DIGEST,
    })
    dump(ANCHOR / "scientific_settings_audit.json", {
        "schema": "paper_i_sr_material_window_anchor_scientific_settings_audit_v1",
        "status": "pass",
        "anchor_changed_scientific_fields_vs_source": [],
        "candidate_not_executed_in_anchor": True,
    })
    preflight = {
        "schema": "paper_i_sr_material_window_anchor_preflight_v1",
        "status": "local_archive_preflight_pending",
        "checks": {
            "one_parent_anchor_record": True,
            "same_cutoff_n_ph_3": True,
            "exact_round_50_horizon": True,
            "candidate_not_executed": True,
            "threshold_sources_hash_locked": True,
            "archive_only_worker_validation": False,
            "archive_focused_tests": False,
        },
    }
    dump(ANCHOR / "preflight.json", preflight)
    dump(ANCHOR / "archive_only_preflight.json", preflight)
    dump(ANCHOR / "remote_execution_gate.json", {
        "schema": "paper_i_sr_material_window_anchor_remote_gate_v1",
        "status": "pending_authenticated_remote_preflight",
        "image_sha256": IMAGE_SHA256,
        "source_archive_sha256": archive_sha,
        "submission_performed": False,
    })
    (ANCHOR / "README.md").write_text(
        "# Phase-III material-window source-value anchor\n\n"
        "One weak-weak full-geometry parent replay. The material-window fanout "
        "is forbidden until exact trajectory reproduction is recorded.\n",
        encoding="utf-8",
    )
    verifier = f'''#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
 r=json.loads((B/"anchor_bundle_receipt.json").read_text())
 assert h(B/"source_locked.tar.gz")==r["source_archive_sha256"]
 jobs=list((B/"jobs").glob("*.json")); assert [p.name for p in jobs]==["weak_weak.json"]
 j=json.loads(jobs[0].read_text())
 assert j["route_identity"]["profile_request"]=={PARENT_ALIAS!r}
 assert j["route_identity"]["profile_contract_sha256"]=={PARENT_DIGEST!r}
 assert j["route_identity"]["profile_contract"]["execution_settings"]["phase3_response_coordinate_scope"]=={PARENT_SCOPE!r}
 assert int(j["segment"]["target_controller_round"])==50
 assert j["physics"]["n_ph_work"]==j["physics"]["n_ph_reference"]==3
 assert json.loads((B/"source_locked_sensitivity_audit.json").read_text())["fanout_authorized"] is False
 assert "requirements = False" not in (B/"submit.sub").read_text()
 assert not (B/"route_registration_repair.json").exists()
 inventory=B/"submission_artifact_hashes.json"
 if inventory.is_file():
  files=json.loads(inventory.read_text())["files"]
  actual={{p.relative_to(B).as_posix() for p in B.rglob("*") if p.is_file() and p.name!="submission_artifact_hashes.json"}}
  assert set(files)==actual
  for relative,metadata in files.items():
   path=B/relative
   assert h(path)==metadata["sha256"]
   assert path.stat().st_size==metadata["size_bytes"]
 return True
if __name__=="__main__": verify(); print("material-window parent anchor verified")
'''
    (ANCHOR / "build_bundle.py").write_text(verifier, encoding="utf-8")
    (ANCHOR / "test_bundle.py").write_text(
        "import build_bundle\ndef test_bundle(): assert build_bundle.verify()\n",
        encoding="utf-8",
    )
    return receipt


def archive_preflight() -> None:
    with tempfile.TemporaryDirectory(prefix="paper-i-material-window-anchor-preflight-") as raw:
        root = Path(raw)
        with tarfile.open(ANCHOR / "source_locked.tar.gz", "r:gz") as archive:
            archive.extractall(root, filter="data")
        target = root / "chtc/phase3_optuna/input" / ANCHOR_ID
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ANCHOR, target)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)
        env.pop("PYTHONNOUSERSITE", None)
        subprocess.run(
            [sys.executable, str(target / "run_job.py"), "--validate-only", str(target / "jobs/weak_weak.json")],
            cwd=root, env=env, check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *FOCUSED_TEST_OVERLAYS],
            cwd=root, env=env, check=True,
        )


def main() -> int:
    receipt = build_anchor()
    subprocess.run([sys.executable, str(ANCHOR / "build_bundle.py")], check=True)
    subprocess.run([sys.executable, "-m", "pytest", "-q", str(ANCHOR / "test_bundle.py")], check=True)
    archive_preflight()
    preflight = load(ANCHOR / "preflight.json")
    preflight["status"] = "pass"
    preflight["checks"]["archive_only_worker_validation"] = True
    preflight["checks"]["archive_focused_tests"] = True
    dump(ANCHOR / "preflight.json", preflight)
    dump(ANCHOR / "archive_only_preflight.json", preflight)
    for cache in ANCHOR.rglob("__pycache__"):
        shutil.rmtree(cache)
    for bytecode in ANCHOR.rglob("*.pyc"):
        bytecode.unlink()
    dump(ANCHOR / "submission_artifact_hashes.json", {
        "schema": "paper_i_sr_material_window_anchor_submission_artifacts_v1",
        "files": {
            path.relative_to(ANCHOR).as_posix(): {
                "sha256": sha256(path), "size_bytes": path.stat().st_size,
            }
            for path in sorted(ANCHOR.rglob("*"))
            if path.is_file() and path.name != "submission_artifact_hashes.json"
        },
    })
    subprocess.run([sys.executable, str(ANCHOR / "build_bundle.py")], check=True)
    final_test_env = os.environ.copy()
    final_test_env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(ANCHOR / "test_bundle.py")],
        check=True, env=final_test_env,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
