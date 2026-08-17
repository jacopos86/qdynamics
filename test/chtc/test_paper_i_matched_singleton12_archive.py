from __future__ import annotations

import gzip
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
from types import ModuleType
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
UTILITY_PATH = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_matched_singleton12_archive_20260815.py"
)
EXECUTION_ID = "matched_singleton12_test_cell"
SOURCE_PREFIX = f"runs/{EXECUTION_ID}"
AUTHORITY = {
    "campaign_id": "paper_i_matched_singleton12_test",
    "execution_authorized": True,
    "package_sha256": "a" * 64,
}
CELL = {
    "comparator_policy": "always_commutation_reduced",
    "execution_id": EXECUTION_ID,
    "nph": 7,
    "regime_id": "strong_strong_u8",
}
ROTATION_AUTHORITY = {
    "archive_rotation_authorized": True,
    "execution_authorized": True,
    "authorization_sha256": "b" * 64,
}


def _load_utility() -> ModuleType:
    name = "paper_i_matched_singleton12_archive_test"
    spec = importlib.util.spec_from_file_location(name, UTILITY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
        sys.dont_write_bytecode = previous
    return module


@pytest.fixture(scope="module")
def utility() -> ModuleType:
    return _load_utility()


@pytest.fixture
def limits(utility: ModuleType) -> Any:
    return utility.ArchiveLimits(
        max_member_payload_bytes=2 * 1024 * 1024,
        max_total_payload_bytes=8 * 1024 * 1024,
        max_decompressed_bytes=16 * 1024 * 1024,
        max_compressed_bytes=8 * 1024 * 1024,
        min_free_disk_bytes=0,
    )


def _prepare_cell(
    tmp_path: Path, utility: ModuleType, *, name: str = "runtime"
) -> tuple[Any, dict[str, Path]]:
    runtime = tmp_path / name
    runtime.mkdir()
    paths = utility.CellArchivePaths(runtime, EXECUTION_ID)
    source = paths.source_root
    (source / "payload").mkdir(parents=True)
    (source / "empty_directory").mkdir()
    (source / "result.json").write_text(
        '{"energy":-1.25,"status":"passed"}\n', encoding="utf-8"
    )
    (source / "payload/checkpoint.json").write_bytes(
        b'{"controller_rounds_completed":50}\n'
    )
    executable = source / "payload/replay.sh"
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    external_root = tmp_path / f"{name}_external"
    external_root.mkdir()
    worker = external_root / "worker_receipt.json"
    guard = external_root / "guard_receipt.json"
    worker.write_text('{"status":"passed","worker":1}\n', encoding="utf-8")
    guard.write_text('{"status":"passed","guard":1}\n', encoding="utf-8")
    return paths, {
        "receipts/guard_receipt.json": guard,
        "receipts/worker_receipt.json": worker,
    }


def _build(
    utility: ModuleType,
    paths: Any,
    external: dict[str, Path],
    limits: Any,
) -> dict[str, Any]:
    return utility.build_cell_archive(
        paths=paths,
        source_member_prefix=SOURCE_PREFIX,
        external_members=external,
        authority_metadata=AUTHORITY,
        cell_metadata=CELL,
        limits=limits,
    )


def _close_and_intend(
    utility: ModuleType,
    paths: Any,
    limits: Any,
) -> None:
    utility.publish_archive_closure(
        paths=paths,
        source_member_prefix=SOURCE_PREFIX,
        authority_metadata=AUTHORITY,
        cell_metadata=CELL,
        limits=limits,
        created_at_utc="2026-08-15T12:00:00Z",
    )
    utility.publish_rotation_intent(
        paths=paths,
        source_member_prefix=SOURCE_PREFIX,
        authority_metadata=AUTHORITY,
        cell_metadata=CELL,
        rotation_authority=ROTATION_AUTHORITY,
        limits=limits,
        created_at_utc="2026-08-15T12:01:00Z",
    )


def test_capacity_contract_distinguishes_initial_and_per_regime_floors(
    utility: ModuleType,
) -> None:
    initial = utility.campaign_capacity_floor()
    assert initial["campaign_minimum_free_bytes"] == 31 * 1024**3
    assert initial["largest_observed_cell_raw_kib"] == 17_500_000
    assert initial["largest_observed_cell_raw_bytes"] == 17_920_000_000
    assert initial["working_space_safety_factor"] == {
        "numerator": 5,
        "denominator": 4,
    }
    assert initial["archive_start_free_floor_bytes"] == 10 * 1024**3
    assert initial["largest_regime_exact_formula_floor_bytes"] == (
        (5 * 17_920_000_000 + 3) // 4 + 10 * 1024**3
    )
    assert initial["capacity_evidence"] == {
        "source_path": (
            "chtc/paper_i_ra_adapt_repair_20260727/"
            "run_local_page12_insertion_comparators_20260812.py"
        ),
        "source_mapping": "PRIOR_RESOURCE_EVIDENCE",
        "cluster_proc": "9605157.5",
    }

    expected = {
        ("strong_strong_u8", 7): (17_500_000, 30_862, 33_137_820_173),
        ("intermediate_strong", 7): (12_500_000, 24_902, 26_737_418_240),
        ("weak_strong", 7): (10_000_000, 21_921, 23_537_494_524),
        ("strong_weak_u8", 3): (3_750_000, 14_471, 15_537_418_240),
        ("intermediate_weak", 3): (3_750_000, 14_471, 15_537_418_240),
        ("weak_weak", 3): (3_500_000, 14_173, 15_217_418_240),
    }
    for (regime_id, nph), (raw_kib, ceil_milligib, enforced_bytes) in expected.items():
        row = utility.regime_launch_capacity_floor(
            regime_id=regime_id, nph=nph
        )
        raw_bytes = raw_kib * 1024
        exact_formula = (5 * raw_bytes + 3) // 4 + 10 * 1024**3
        assert row["observed_working_disk_kib"] == raw_kib
        assert row["observed_working_disk_bytes"] == raw_bytes
        assert row["working_space_safety_factor"] == {
            "numerator": 5,
            "denominator": 4,
        }
        assert row["archive_start_free_floor_bytes"] == 10 * 1024**3
        assert row["exact_formula_floor_bytes"] == exact_formula
        assert row["minimum_free_bytes"] == enforced_bytes
        assert row["minimum_free_milligib"] == ceil_milligib
        assert row["exact_formula_floor_milligib_ceil"] == (
            exact_formula * 1000 + 1024**3 - 1
        ) // 1024**3
        assert row["minimum_free_milligib"] == max(
            row["prior_display_floor_milligib"],
            row["exact_formula_floor_milligib_ceil"],
        )
        assert row["minimum_free_bytes"] >= exact_formula
        assert row["capacity_evidence"]["source_mapping"] == (
            "PRIOR_RESOURCE_EVIDENCE"
        )
    with pytest.raises(utility.Singleton12ArchiveError):
        utility.regime_launch_capacity_floor(
            regime_id="strong_strong_u8", nph=3
        )

    archive_limits = utility.campaign_default_archive_limits()
    assert archive_limits.max_member_payload_bytes == 32 * 1024**3
    assert archive_limits.max_total_payload_bytes == 32 * 1024**3
    assert archive_limits.max_decompressed_bytes == 33 * 1024**3
    assert archive_limits.max_compressed_bytes == 4 * 1024**3
    assert archive_limits.min_free_disk_bytes == 6 * 1024**3
    assert archive_limits.archive_start_free_floor_bytes == 10 * 1024**3


def test_archive_is_deterministic_and_full_stream_validated(
    tmp_path: Path, utility: ModuleType, limits: Any
) -> None:
    first, first_external = _prepare_cell(tmp_path, utility, name="first")
    second, second_external = _prepare_cell(tmp_path, utility, name="second")

    first_validation = _build(utility, first, first_external, limits)
    second_validation = _build(utility, second, second_external, limits)

    assert first.archive_path.read_bytes() == second.archive_path.read_bytes()
    assert first_validation == second_validation
    assert first_validation["status"] == (
        "passed_full_bounded_streaming_validation"
    )
    assert first_validation["authority_metadata"] == AUTHORITY
    assert first_validation["cell_metadata"] == CELL
    assert first_validation["member_validation"]["single_gzip_member_only"]
    assert first.archive_manifest_path.read_bytes().endswith(b"\n")
    assert utility.inspect_rotation_state(first)["state"] == (
        "manifest_published_pending_closure"
    )


def test_build_restart_is_idempotent_but_never_replaces_archive(
    tmp_path: Path, utility: ModuleType, limits: Any
) -> None:
    paths, external = _prepare_cell(tmp_path, utility)
    first = _build(utility, paths, external, limits)
    inode = paths.archive_path.stat().st_ino
    second = _build(utility, paths, external, limits)

    assert second == first
    assert paths.archive_path.stat().st_ino == inode

    target = tmp_path / "receipt.json"
    utility.write_json_atomic_noreplace(target, {"value": 1})
    with pytest.raises(FileExistsError):
        utility.write_json_atomic_noreplace(target, {"value": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 1}


def _tar_info(
    name: str, payload: bytes, *, member_type: bytes = tarfile.REGTYPE
) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = member_type
    info.size = len(payload) if member_type == tarfile.REGTYPE else 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def _write_malicious_archive(
    path: Path, rows: list[tuple[str, bytes, bytes]]
) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w|") as archive:
                for name, payload, member_type in rows:
                    info = _tar_info(name, payload, member_type=member_type)
                    archive.addfile(
                        info,
                        io.BytesIO(payload)
                        if member_type == tarfile.REGTYPE
                        else None,
                    )


@pytest.mark.parametrize(
    "rows",
    [
        [
            ("duplicate", b"one", tarfile.REGTYPE),
            ("duplicate", b"two", tarfile.REGTYPE),
        ],
        [("unsafe_link", b"", tarfile.SYMTYPE)],
        [("../escape", b"bad", tarfile.REGTYPE)],
    ],
)
def test_validator_rejects_duplicate_nonregular_and_traversal_members(
    tmp_path: Path,
    utility: ModuleType,
    limits: Any,
    rows: list[tuple[str, bytes, bytes]],
) -> None:
    archive = tmp_path / "malicious.tar.gz"
    _write_malicious_archive(archive, rows)
    with pytest.raises(utility.Singleton12ArchiveError):
        utility.validate_cell_archive(
            archive,
            expected_execution_id=EXECUTION_ID,
            expected_source_member_prefix=SOURCE_PREFIX,
            expected_authority_metadata=AUTHORITY,
            expected_cell_metadata=CELL,
            limits=limits,
        )


def test_validator_rejects_metadata_drift_and_concatenated_gzip(
    tmp_path: Path, utility: ModuleType, limits: Any
) -> None:
    paths, external = _prepare_cell(tmp_path, utility)
    _build(utility, paths, external, limits)

    with pytest.raises(utility.Singleton12ArchiveError):
        utility.validate_cell_archive(
            paths.archive_path,
            expected_execution_id=EXECUTION_ID,
            expected_source_member_prefix=SOURCE_PREFIX,
            expected_authority_metadata={**AUTHORITY, "campaign_id": "drift"},
            expected_cell_metadata=CELL,
            limits=limits,
        )

    concatenated = tmp_path / "concatenated.tar.gz"
    payload = paths.archive_path.read_bytes()
    concatenated.write_bytes(payload + payload)
    with pytest.raises(utility.Singleton12ArchiveError):
        utility.validate_cell_archive(
            concatenated,
            expected_execution_id=EXECUTION_ID,
            expected_source_member_prefix=SOURCE_PREFIX,
            expected_authority_metadata=AUTHORITY,
            expected_cell_metadata=CELL,
            limits=limits,
        )


def test_validation_failure_preserves_direct_science_tree(
    tmp_path: Path, utility: ModuleType, limits: Any
) -> None:
    paths, external = _prepare_cell(tmp_path, utility)
    symlink = paths.source_root / "payload/unsafe_link"
    symlink.symlink_to(paths.source_root / "result.json")

    with pytest.raises(utility.Singleton12ArchiveError):
        _build(utility, paths, external, limits)

    assert paths.source_root.is_dir()
    assert (paths.source_root / "result.json").is_file()
    assert not paths.archive_path.exists()
    assert not paths.archive_closure_path.exists()
    assert not paths.rotation_intent_path.exists()
    assert not paths.cleanup_receipt_path.exists()


def test_full_rotation_publishes_terminal_archive_backed_closure(
    tmp_path: Path, utility: ModuleType, limits: Any
) -> None:
    paths, external = _prepare_cell(tmp_path, utility)
    _build(utility, paths, external, limits)
    _close_and_intend(utility, paths, limits)
    assert utility.inspect_rotation_state(paths)["state"] == (
        "intent_published_pending_rename"
    )

    cleanup = utility.complete_safe_tree_rotation(
        paths=paths,
        source_member_prefix=SOURCE_PREFIX,
        authority_metadata=AUTHORITY,
        cell_metadata=CELL,
        rotation_authority=ROTATION_AUTHORITY,
        limits=limits,
        completed_at_utc="2026-08-15T12:02:00Z",
    )

    assert cleanup["status"] == (
        "passed_exact_safe_tree_removed_archive_retained"
    )
    assert not paths.source_root.exists()
    assert not paths.retiring_root.exists()
    assert paths.archive_path.is_file()
    assert utility.inspect_rotation_state(paths)["state"] == "archived_closed"
    terminal = utility.validate_archive_backed_closure(
        paths=paths,
        source_member_prefix=SOURCE_PREFIX,
        expected_authority_metadata=AUTHORITY,
        expected_cell_metadata=CELL,
        expected_rotation_authority=ROTATION_AUTHORITY,
        limits=limits,
    )
    assert terminal["status"] == "passed_archive_backed_terminal_closure"
    assert terminal["direct_source_absent"] is True
    assert terminal["retiring_source_absent"] is True
    assert terminal["cleanup_receipt"]["canonical_sha256"] == cleanup["sha256"]


@pytest.mark.parametrize("crash_after_delete", [False, True])
def test_rotation_resumes_after_rename_or_delete_before_cleanup(
    tmp_path: Path,
    utility: ModuleType,
    limits: Any,
    crash_after_delete: bool,
) -> None:
    paths, external = _prepare_cell(tmp_path, utility)
    _build(utility, paths, external, limits)
    _close_and_intend(utility, paths, limits)
    paths.retiring_root.parent.mkdir(exist_ok=True)
    os.rename(paths.source_root, paths.retiring_root)
    if crash_after_delete:
        shutil.rmtree(paths.retiring_root)
        assert utility.inspect_rotation_state(paths)["state"] == (
            "cleanup_receipt_pending"
        )
    else:
        assert utility.inspect_rotation_state(paths)["state"] == (
            "retiring_pending_removal"
        )

    utility.complete_safe_tree_rotation(
        paths=paths,
        source_member_prefix=SOURCE_PREFIX,
        authority_metadata=AUTHORITY,
        cell_metadata=CELL,
        rotation_authority=ROTATION_AUTHORITY,
        limits=limits,
        completed_at_utc="2026-08-15T12:03:00Z",
    )
    assert utility.inspect_rotation_state(paths)["state"] == "archived_closed"


def test_rotation_resumes_after_crash_partway_through_safe_removal(
    tmp_path: Path,
    utility: ModuleType,
    limits: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, external = _prepare_cell(tmp_path, utility)
    _build(utility, paths, external, limits)
    _close_and_intend(utility, paths, limits)
    original_rmtree = utility.shutil.rmtree

    def crashing_rmtree(root: Path) -> None:
        (Path(root) / "result.json").unlink()
        raise OSError("simulated crash during safe removal")

    crashing_rmtree.avoids_symlink_attacks = True
    monkeypatch.setattr(utility.shutil, "rmtree", crashing_rmtree)
    with pytest.raises(OSError, match="simulated crash"):
        utility.complete_safe_tree_rotation(
            paths=paths,
            source_member_prefix=SOURCE_PREFIX,
            authority_metadata=AUTHORITY,
            cell_metadata=CELL,
            rotation_authority=ROTATION_AUTHORITY,
            limits=limits,
            completed_at_utc="2026-08-15T12:03:00Z",
        )
    monkeypatch.setattr(utility.shutil, "rmtree", original_rmtree)

    assert utility.inspect_rotation_state(paths)["state"] == (
        "retiring_pending_removal"
    )
    assert not (paths.retiring_root / "result.json").exists()
    utility.complete_safe_tree_rotation(
        paths=paths,
        source_member_prefix=SOURCE_PREFIX,
        authority_metadata=AUTHORITY,
        cell_metadata=CELL,
        rotation_authority=ROTATION_AUTHORITY,
        limits=limits,
        completed_at_utc="2026-08-15T12:03:01Z",
    )
    assert utility.inspect_rotation_state(paths)["state"] == "archived_closed"


@pytest.mark.parametrize(
    "removal_shape",
    ["one_file", "whole_subtree", "all_children"],
)
def test_rotation_accepts_byte_valid_remaining_manifest_subset(
    tmp_path: Path,
    utility: ModuleType,
    limits: Any,
    removal_shape: str,
) -> None:
    paths, external = _prepare_cell(tmp_path, utility)
    _build(utility, paths, external, limits)
    _close_and_intend(utility, paths, limits)
    paths.retiring_root.parent.mkdir(exist_ok=True)
    os.rename(paths.source_root, paths.retiring_root)

    if removal_shape == "one_file":
        (paths.retiring_root / "result.json").unlink()
    elif removal_shape == "whole_subtree":
        shutil.rmtree(paths.retiring_root / "payload")
    else:
        for child in list(paths.retiring_root.iterdir()):
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    utility.complete_safe_tree_rotation(
        paths=paths,
        source_member_prefix=SOURCE_PREFIX,
        authority_metadata=AUTHORITY,
        cell_metadata=CELL,
        rotation_authority=ROTATION_AUTHORITY,
        limits=limits,
        completed_at_utc="2026-08-15T12:03:02Z",
    )
    assert utility.inspect_rotation_state(paths)["state"] == "archived_closed"


@pytest.mark.parametrize(
    "tamper_kind",
    ["content", "extra", "symlink", "wrong_type"],
)
def test_partial_retiring_tree_rejects_tamper_extra_symlink_and_wrong_type(
    tmp_path: Path,
    utility: ModuleType,
    limits: Any,
    tamper_kind: str,
) -> None:
    paths, external = _prepare_cell(tmp_path, utility)
    _build(utility, paths, external, limits)
    _close_and_intend(utility, paths, limits)
    paths.retiring_root.parent.mkdir(exist_ok=True)
    os.rename(paths.source_root, paths.retiring_root)
    (paths.retiring_root / "payload/checkpoint.json").unlink()
    target = paths.retiring_root / "result.json"

    if tamper_kind == "content":
        original = target.read_bytes()
        target.write_bytes(bytes([original[0] ^ 0x01]) + original[1:])
    elif tamper_kind == "extra":
        (paths.retiring_root / "unexpected.json").write_text(
            '{"unexpected":true}\n', encoding="utf-8"
        )
    elif tamper_kind == "symlink":
        target.unlink()
        target.symlink_to(paths.retiring_root / "payload/replay.sh")
    else:
        target.unlink()
        target.mkdir()

    with pytest.raises(utility.Singleton12ArchiveError):
        utility.complete_safe_tree_rotation(
            paths=paths,
            source_member_prefix=SOURCE_PREFIX,
            authority_metadata=AUTHORITY,
            cell_metadata=CELL,
            rotation_authority=ROTATION_AUTHORITY,
            limits=limits,
            completed_at_utc="2026-08-15T12:03:03Z",
        )

    assert utility.inspect_rotation_state(paths)["state"] == (
        "retiring_pending_removal"
    )
    assert not paths.cleanup_receipt_path.exists()


def test_archive_tamper_invalidates_archived_terminal_closure(
    tmp_path: Path, utility: ModuleType, limits: Any
) -> None:
    paths, external = _prepare_cell(tmp_path, utility)
    _build(utility, paths, external, limits)
    _close_and_intend(utility, paths, limits)
    utility.complete_safe_tree_rotation(
        paths=paths,
        source_member_prefix=SOURCE_PREFIX,
        authority_metadata=AUTHORITY,
        cell_metadata=CELL,
        rotation_authority=ROTATION_AUTHORITY,
        limits=limits,
        completed_at_utc="2026-08-15T12:04:00Z",
    )
    with paths.archive_path.open("r+b") as stream:
        stream.seek(max(1, paths.archive_path.stat().st_size // 2))
        byte = stream.read(1)
        stream.seek(-1, os.SEEK_CUR)
        stream.write(bytes([byte[0] ^ 0x01]))

    with pytest.raises(utility.Singleton12ArchiveError):
        utility.validate_archive_backed_closure(
            paths=paths,
            source_member_prefix=SOURCE_PREFIX,
            expected_authority_metadata=AUTHORITY,
            expected_cell_metadata=CELL,
            expected_rotation_authority=ROTATION_AUTHORITY,
            limits=limits,
        )
