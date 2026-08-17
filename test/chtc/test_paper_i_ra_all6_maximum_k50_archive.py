from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTER_PATH = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_all6_maximum_k50_archive_20260817.py"
)
CAMPAIGN_ID = "paper_i_ra_all6_maximum_k50_archive_test"
EXECUTION_ID = "maximum_k50_test_cell"
CELL_METADATA = {
    "block": "append",
    "execution_id": EXECUTION_ID,
    "horizon": 50,
    "insertion_policy": "append_only",
    "nph": 3,
    "ordinal": 1,
    "regime_id": "weak_weak",
    "route_variant": "test_natural_terminal_v2",
}
CELL_IDENTITY = {
    "execution_id": EXECUTION_ID,
    "cell_ordinal": 1,
    "block": "append",
    "regime_id": "weak_weak",
    "nph": 3,
    "insertion_policy": "append_only",
}


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digested(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["sha256"] = hashlib.sha256(_canonical_bytes(result)).hexdigest()
    return result


def _load_adapter() -> ModuleType:
    name = "paper_i_ra_all6_maximum_k50_archive_test"
    spec = importlib.util.spec_from_file_location(name, ADAPTER_PATH)
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
def adapter() -> ModuleType:
    return _load_adapter()


def _prepare_evidence_files(tmp_path: Path) -> dict[str, Any]:
    runtime = tmp_path / "runtime"
    run_root = runtime / "runs" / EXECUTION_ID
    (run_root / "checkpoint").mkdir(parents=True)
    (run_root / "ledger").mkdir()
    (run_root / "result").mkdir()
    checkpoint = run_root / "checkpoint/checkpoint.json"
    ledger = run_root / "ledger/estimator_ledger.json"
    result = run_root / "result/result.json"
    checkpoint.write_text('{"accepted_controller_rounds":0}\n', encoding="utf-8")
    checkpoint.chmod(0o600)
    ledger.write_text('{"entries":[]}\n', encoding="utf-8")
    result.write_text('{"status":"natural_terminal"}\n', encoding="utf-8")

    (runtime / "worker_receipts").mkdir()
    (runtime / "guard_receipts").mkdir()
    (runtime / "cell_logs").mkdir()
    worker = runtime / "worker_receipts" / f"{EXECUTION_ID}.json"
    guard = runtime / "guard_receipts" / f"{EXECUTION_ID}.json"
    log = runtime / "cell_logs" / f"{EXECUTION_ID}.log"
    worker.write_text('{"status":"passed"}\n', encoding="utf-8")
    guard.write_text('{"status":"passed"}\n', encoding="utf-8")
    log.write_text("cell reached a natural terminal\n", encoding="utf-8")
    return {
        "runtime": runtime,
        "run_root": run_root,
        "worker": worker,
        "guard": guard,
        "log": log,
        "source_artifacts": {
            "checkpoint": checkpoint,
            "estimator_ledger": ledger,
            "result": result,
        },
    }


def _accepted_row(controller_round: int) -> dict[str, Any]:
    return {
        **CELL_IDENTITY,
        "controller_round": controller_round,
        "energy": -1.0 - controller_round / 10,
        "absolute_delta_e": 1.0 / controller_round,
        "placement_state": "append_only",
        "phase0_population_count": 8,
        "phase0_retained_count": 6,
        "phase_i_input_count": 6,
        "phase_i_retained_count": 4,
        "phase_ii_input_count": 4,
        "phase_ii_retained_count": 3,
        "phase_iii_input_count": 3,
        "phase_iii_adaptive_retained_count": 2,
        "phase_iii_final_singleton_count": 1,
        "phase_iii_final_record_id": f"record::{controller_round}",
        "selected_generator": f"generator::{controller_round}",
        "selected_operator": f"operator::{controller_round}",
        "selected_position": controller_round - 1,
        "s_alg": 100 * controller_round,
        "n2q": 10 * controller_round,
        "d2q": 7 * controller_round,
        "dc": 12 * controller_round,
        "checkpoint_sha256": f"{controller_round:02x}" * 32,
    }


def _terminal_attempt(attempted_round: int) -> dict[str, Any]:
    return _digested(
        {
            **CELL_IDENTITY,
            "attempted_controller_round": attempted_round,
            "terminal_controller_outcome": (
                "phase_iii_no_positive_feasible_candidate_v1"
            ),
            "terminal_phase3_selection_receipt_sha256": "b" * 64,
            "terminal_active_prefix_checkpoint_sha256": "c" * 64,
            "placement_state": "append_only",
            "phase0_population_count": 8,
            "phase0_retained_count": 6,
            "phase_i_input_count": 6,
            "phase_i_retained_count": 4,
            "phase_ii_input_count": 4,
            "phase_ii_retained_count": 3,
            "phase_iii_input_count": 3,
            "phase_iii_adaptive_retained_count": 0,
            "phase_iii_final_singleton_count": 0,
        }
    )


def _round_zero_payloads(files: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    checkpoint_sha256 = hashlib.sha256(
        files["source_artifacts"]["checkpoint"].read_bytes()
    ).hexdigest()
    completion = _digested(
        {
            "schema": (
                "paper_i_ra_all6_adaptive_maximum_k50_cell_completion_v1"
            ),
            "campaign_id": CAMPAIGN_ID,
            "execution_id": EXECUTION_ID,
            "completion_kind": (
                "authenticated_phase3_no_positive_natural_terminal_v1"
            ),
            "maximum_controller_rounds": 50,
            "accepted_controller_rounds": 0,
            "terminal_attempted_controller_round": 1,
            "terminal_controller_outcome": (
                "phase_iii_no_positive_feasible_candidate_v1"
            ),
            "terminal_phase3_selection_receipt_sha256": "b" * 64,
            "terminal_active_prefix_checkpoint_sha256": "c" * 64,
            "summary_artifact_status": "not_applicable_round_zero",
            "checkpoint_file_sha256": checkpoint_sha256,
        }
    )
    terminal_attempt = _terminal_attempt(1)
    cell_outcome = _digested(
        {
            "schema": "paper_i_ra_all6_maximum_k50_cell_outcome_v1",
            "execution_id": EXECUTION_ID,
            "completion_kind": (
                "authenticated_phase3_no_positive_natural_terminal_v1"
            ),
            "accepted_controller_rounds": 0,
        }
    )
    return completion, terminal_attempt, cell_outcome


def test_round_zero_compact_payload_binds_terminal_and_omits_summary(
    tmp_path: Path, adapter: ModuleType
) -> None:
    files = _prepare_evidence_files(tmp_path)
    completion, terminal_attempt, cell_outcome = _round_zero_payloads(files)

    compact = adapter.build_compact_payload(
        runtime_root=files["runtime"],
        campaign_id=CAMPAIGN_ID,
        execution_id=EXECUTION_ID,
        cell_metadata=CELL_METADATA,
        cell_completion=completion,
        accepted_rows=[],
        terminal_attempt=terminal_attempt,
        cell_outcome=cell_outcome,
        worker_receipt_path=files["worker"],
        guard_receipt_path=files["guard"],
        log_path=files["log"],
        source_artifact_paths=files["source_artifacts"],
    )

    assert compact["schema"] == (
        "paper_i_ra_all6_maximum_k50_compact_cell_evidence_v1"
    )
    assert compact["accepted_controller_rounds"] == 0
    assert compact["accepted_rows"] == []
    assert compact["accepted_rows_sha256"] == hashlib.sha256(b"[]").hexdigest()
    assert compact["cell_completion_sha256"] == completion["sha256"]
    assert compact["terminal_attempt_sha256"] == terminal_attempt["sha256"]
    assert compact["cell_outcome_sha256"] == cell_outcome["sha256"]
    assert set(compact["source_artifact_bindings"]) == {
        "checkpoint",
        "estimator_ledger",
        "result",
    }
    assert compact["summary_artifact_status"] == "not_applicable_round_zero"
    assert compact["sha256"] == hashlib.sha256(
        _canonical_bytes(
            {
                key: value
                for key, value in compact.items()
                if key != "sha256"
            }
        )
    ).hexdigest()


def test_compact_validator_accepts_ragged_prefix_and_rejects_tamper(
    tmp_path: Path, adapter: ModuleType
) -> None:
    files = _prepare_evidence_files(tmp_path)
    summary = files["run_root"] / "summary/summary.json"
    summary.parent.mkdir()
    summary.write_text('{"accepted_controller_rounds":2}\n', encoding="utf-8")
    files["source_artifacts"]["summary"] = summary
    checkpoint_sha256 = hashlib.sha256(
        files["source_artifacts"]["checkpoint"].read_bytes()
    ).hexdigest()
    summary_sha256 = hashlib.sha256(summary.read_bytes()).hexdigest()
    completion = _digested(
        {
            "schema": (
                "paper_i_ra_all6_adaptive_maximum_k50_cell_completion_v1"
            ),
            "campaign_id": CAMPAIGN_ID,
            "execution_id": EXECUTION_ID,
            "completion_kind": (
                "authenticated_phase3_no_positive_natural_terminal_v1"
            ),
            "maximum_controller_rounds": 50,
            "accepted_controller_rounds": 2,
            "terminal_attempted_controller_round": 3,
            "terminal_controller_outcome": (
                "phase_iii_no_positive_feasible_candidate_v1"
            ),
            "terminal_phase3_selection_receipt_sha256": "b" * 64,
            "terminal_active_prefix_checkpoint_sha256": "c" * 64,
            "summary_artifact_status": "present",
            "checkpoint_file_sha256": checkpoint_sha256,
            "paper_i_summary_sha256": summary_sha256,
        }
    )
    terminal_attempt = _terminal_attempt(3)
    cell_outcome = _digested(
        {
            "schema": "paper_i_ra_all6_maximum_k50_cell_outcome_v1",
            "execution_id": EXECUTION_ID,
            "completion_kind": (
                "authenticated_phase3_no_positive_natural_terminal_v1"
            ),
            "accepted_controller_rounds": 2,
        }
    )
    rows = [_accepted_row(1), _accepted_row(2)]
    compact = adapter.build_compact_payload(
        runtime_root=files["runtime"],
        campaign_id=CAMPAIGN_ID,
        execution_id=EXECUTION_ID,
        cell_metadata=CELL_METADATA,
        cell_completion=completion,
        accepted_rows=rows,
        terminal_attempt=terminal_attempt,
        cell_outcome=cell_outcome,
        worker_receipt_path=files["worker"],
        guard_receipt_path=files["guard"],
        log_path=files["log"],
        source_artifact_paths=files["source_artifacts"],
    )

    assert adapter.validate_compact_payload(
        compact,
        runtime_root=files["runtime"],
        campaign_id=CAMPAIGN_ID,
        execution_id=EXECUTION_ID,
        cell_metadata=CELL_METADATA,
        worker_receipt_path=files["worker"],
        guard_receipt_path=files["guard"],
        log_path=files["log"],
        require_live_source_artifacts=True,
    ) == compact

    tampered = json.loads(json.dumps(compact))
    tampered["accepted_rows"][1]["energy"] = -9.0
    with pytest.raises(adapter.MaximumK50ArchiveError):
        adapter.validate_compact_payload(
            tampered,
            runtime_root=files["runtime"],
            campaign_id=CAMPAIGN_ID,
            execution_id=EXECUTION_ID,
            cell_metadata=CELL_METADATA,
            worker_receipt_path=files["worker"],
            guard_receipt_path=files["guard"],
            log_path=files["log"],
            require_live_source_artifacts=True,
        )

    with pytest.raises(adapter.MaximumK50ArchiveError):
        adapter.build_compact_payload(
            runtime_root=files["runtime"],
            campaign_id=CAMPAIGN_ID,
            execution_id=EXECUTION_ID,
            cell_metadata=CELL_METADATA,
            cell_completion=completion,
            accepted_rows=[
                {"execution_id": EXECUTION_ID, "controller_round": 1},
                {"execution_id": EXECUTION_ID, "controller_round": 2},
            ],
            terminal_attempt=terminal_attempt,
            cell_outcome=cell_outcome,
            worker_receipt_path=files["worker"],
            guard_receipt_path=files["guard"],
            log_path=files["log"],
            source_artifact_paths=files["source_artifacts"],
        )


def test_compact_builder_rejects_foreign_completion_before_archive(
    tmp_path: Path, adapter: ModuleType
) -> None:
    files = _prepare_evidence_files(tmp_path)
    completion, terminal_attempt, cell_outcome = _round_zero_payloads(files)
    foreign_completion = {
        key: value for key, value in completion.items() if key != "sha256"
    }
    foreign_completion["execution_id"] = "foreign_cell"
    foreign_completion = _digested(foreign_completion)

    with pytest.raises(adapter.MaximumK50ArchiveError):
        adapter.build_compact_payload(
            runtime_root=files["runtime"],
            campaign_id=CAMPAIGN_ID,
            execution_id=EXECUTION_ID,
            cell_metadata=CELL_METADATA,
            cell_completion=foreign_completion,
            accepted_rows=[],
            terminal_attempt=terminal_attempt,
            cell_outcome=cell_outcome,
            worker_receipt_path=files["worker"],
            guard_receipt_path=files["guard"],
            log_path=files["log"],
            source_artifact_paths=files["source_artifacts"],
        )


def test_prepare_archive_closes_bytes_but_never_rotates_direct_tree(
    tmp_path: Path, adapter: ModuleType
) -> None:
    files = _prepare_evidence_files(tmp_path)
    completion, terminal_attempt, cell_outcome = _round_zero_payloads(files)
    compact = adapter.build_compact_payload(
        runtime_root=files["runtime"],
        campaign_id=CAMPAIGN_ID,
        execution_id=EXECUTION_ID,
        cell_metadata=CELL_METADATA,
        cell_completion=completion,
        accepted_rows=[],
        terminal_attempt=terminal_attempt,
        cell_outcome=cell_outcome,
        worker_receipt_path=files["worker"],
        guard_receipt_path=files["guard"],
        log_path=files["log"],
        source_artifact_paths=files["source_artifacts"],
    )
    compact_path = (
        files["runtime"] / "compact_cell_receipts" / f"{EXECUTION_ID}.json"
    )
    compact_path.parent.mkdir()
    compact_path.write_bytes(_canonical_bytes(compact) + b"\n")
    authority = _digested(
        {
            "schema": "paper_i_ra_all6_adaptive_maximum_k50_authorization_v1",
            "campaign_id": CAMPAIGN_ID,
            "authorization_basis": (
                "explicit_current_user_maximum_k50_natural_terminal_request"
            ),
            "execution_authorized": True,
            "archive_rotation_authorized": True,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    limits = adapter.ArchiveLimits(
        max_member_payload_bytes=1024 * 1024,
        max_total_payload_bytes=8 * 1024 * 1024,
        max_decompressed_bytes=16 * 1024 * 1024,
        max_compressed_bytes=8 * 1024 * 1024,
        min_free_disk_bytes=0,
    )
    paths = adapter.CellArchivePaths(files["runtime"], EXECUTION_ID)
    paths.archive_path.parent.mkdir(parents=True, exist_ok=True)
    stale_temporary = paths.archive_path.with_name(
        f".{paths.archive_path.name}.tmp.{'0' * 32}"
    )
    stale_temporary.write_bytes(b"interrupted archive build")

    prepared = adapter.prepare_archive(
        runtime_root=files["runtime"],
        campaign_id=CAMPAIGN_ID,
        execution_id=EXECUTION_ID,
        cell_metadata=CELL_METADATA,
        authority_metadata=authority,
        compact_path=compact_path,
        worker_receipt_path=files["worker"],
        guard_receipt_path=files["guard"],
        log_path=files["log"],
        limits=limits,
        created_at_utc="2026-08-17T12:00:00Z",
    )

    state = adapter.strict_archive.inspect_rotation_state(paths)
    assert prepared["status"] == "passed_archive_preparation_no_rotation"
    assert state["state"] == "closure_published_pending_intent"
    assert paths.source_root.is_dir()
    assert not paths.retiring_root.exists()
    assert not paths.rotation_intent_path.exists()
    assert paths.archive_path.is_file()
    assert not stale_temporary.exists()
    assert prepared["compact_payload_sha256"] == compact["sha256"]


def _prepare_round_zero_archive(
    tmp_path: Path, adapter: ModuleType
) -> dict[str, Any]:
    files = _prepare_evidence_files(tmp_path)
    completion, terminal_attempt, cell_outcome = _round_zero_payloads(files)
    compact = adapter.build_compact_payload(
        runtime_root=files["runtime"],
        campaign_id=CAMPAIGN_ID,
        execution_id=EXECUTION_ID,
        cell_metadata=CELL_METADATA,
        cell_completion=completion,
        accepted_rows=[],
        terminal_attempt=terminal_attempt,
        cell_outcome=cell_outcome,
        worker_receipt_path=files["worker"],
        guard_receipt_path=files["guard"],
        log_path=files["log"],
        source_artifact_paths=files["source_artifacts"],
    )
    compact_path = (
        files["runtime"] / "compact_cell_receipts" / f"{EXECUTION_ID}.json"
    )
    compact_path.parent.mkdir()
    compact_path.write_bytes(_canonical_bytes(compact) + b"\n")
    authority = _digested(
        {
            "schema": "paper_i_ra_all6_adaptive_maximum_k50_authorization_v1",
            "campaign_id": CAMPAIGN_ID,
            "authorization_basis": (
                "explicit_current_user_maximum_k50_natural_terminal_request"
            ),
            "execution_authorized": True,
            "archive_rotation_authorized": True,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    rotation_authority = dict(authority)
    limits = adapter.ArchiveLimits(
        max_member_payload_bytes=1024 * 1024,
        max_total_payload_bytes=8 * 1024 * 1024,
        max_decompressed_bytes=16 * 1024 * 1024,
        max_compressed_bytes=8 * 1024 * 1024,
        min_free_disk_bytes=0,
    )
    adapter.prepare_archive(
        runtime_root=files["runtime"],
        campaign_id=CAMPAIGN_ID,
        execution_id=EXECUTION_ID,
        cell_metadata=CELL_METADATA,
        authority_metadata=authority,
        compact_path=compact_path,
        worker_receipt_path=files["worker"],
        guard_receipt_path=files["guard"],
        log_path=files["log"],
        limits=limits,
        created_at_utc="2026-08-17T12:00:00Z",
    )
    return {
        **files,
        "compact": compact,
        "compact_path": compact_path,
        "authority": authority,
        "rotation_authority": rotation_authority,
        "limits": limits,
    }


def test_execute_rotation_rejects_wrong_exact_flag_before_intent_or_rename(
    tmp_path: Path, adapter: ModuleType
) -> None:
    evidence = _prepare_round_zero_archive(tmp_path, adapter)
    paths = adapter.CellArchivePaths(evidence["runtime"], EXECUTION_ID)

    with pytest.raises(adapter.MaximumK50ArchiveError, match="exact authorization"):
        adapter.execute_rotation(
            runtime_root=evidence["runtime"],
            campaign_id=CAMPAIGN_ID,
            execution_id=EXECUTION_ID,
            cell_metadata=CELL_METADATA,
            authority_metadata=evidence["authority"],
            rotation_authority=evidence["rotation_authority"],
            exact_authorization_flag="wrong",
            compact_path=evidence["compact_path"],
            worker_receipt_path=evidence["worker"],
            guard_receipt_path=evidence["guard"],
            log_path=evidence["log"],
            limits=evidence["limits"],
        )

    state = adapter.strict_archive.inspect_rotation_state(paths)
    assert state["state"] == "closure_published_pending_intent"
    assert paths.source_root.is_dir()
    assert not paths.retiring_root.exists()
    assert not paths.rotation_intent_path.exists()


def test_execute_rotation_rejects_unsigned_fabricated_authority(
    tmp_path: Path, adapter: ModuleType
) -> None:
    evidence = _prepare_round_zero_archive(tmp_path, adapter)
    paths = adapter.CellArchivePaths(evidence["runtime"], EXECUTION_ID)
    fabricated = {
        "campaign_id": CAMPAIGN_ID,
        "execution_authorized": True,
        "archive_rotation_authorized": True,
    }

    with pytest.raises(adapter.MaximumK50ArchiveError):
        adapter.execute_rotation(
            runtime_root=evidence["runtime"],
            campaign_id=CAMPAIGN_ID,
            execution_id=EXECUTION_ID,
            cell_metadata=CELL_METADATA,
            authority_metadata=evidence["authority"],
            rotation_authority=fabricated,
            exact_authorization_flag=adapter.EXACT_ROTATION_AUTHORIZATION_FLAG,
            compact_path=evidence["compact_path"],
            worker_receipt_path=evidence["worker"],
            guard_receipt_path=evidence["guard"],
            log_path=evidence["log"],
            limits=evidence["limits"],
        )

    assert adapter.strict_archive.inspect_rotation_state(paths)["state"] == (
        "closure_published_pending_intent"
    )
    assert paths.source_root.is_dir()
    assert not paths.rotation_intent_path.exists()


def test_execute_rotation_with_exact_flag_reaches_archive_closed(
    tmp_path: Path, adapter: ModuleType
) -> None:
    evidence = _prepare_round_zero_archive(tmp_path, adapter)
    paths = adapter.CellArchivePaths(evidence["runtime"], EXECUTION_ID)

    loaded = adapter.execute_rotation(
        runtime_root=evidence["runtime"],
        campaign_id=CAMPAIGN_ID,
        execution_id=EXECUTION_ID,
        cell_metadata=CELL_METADATA,
        authority_metadata=evidence["authority"],
        rotation_authority=evidence["rotation_authority"],
        exact_authorization_flag=adapter.EXACT_ROTATION_AUTHORIZATION_FLAG,
        compact_path=evidence["compact_path"],
        worker_receipt_path=evidence["worker"],
        guard_receipt_path=evidence["guard"],
        log_path=evidence["log"],
        limits=evidence["limits"],
        created_at_utc="2026-08-17T12:01:00Z",
        completed_at_utc="2026-08-17T12:02:00Z",
    )

    state = adapter.strict_archive.inspect_rotation_state(paths)
    assert state["state"] == "archived_closed"
    assert not paths.source_root.exists()
    assert not paths.retiring_root.exists()
    assert paths.archive_path.is_file()
    assert loaded["schema"] == (
        "paper_i_ra_all6_maximum_k50_archive_backed_cell_v1"
    )
    assert loaded["status"] == "passed_archive_backed_maximum_k50_cell"
    assert loaded["accepted_controller_rounds"] == 0
    assert loaded["accepted_rows"] == []
    assert loaded["terminal_attempt_sha256"] == evidence["compact"][
        "terminal_attempt_sha256"
    ]


def _filesystem_snapshot(root: Path) -> list[tuple[str, str, str | None]]:
    rows: list[tuple[str, str, str | None]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_file() and not path.is_symlink():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append((relative, "file", digest))
        elif path.is_dir() and not path.is_symlink():
            rows.append((relative, "directory", None))
        else:
            rows.append((relative, "unsafe", None))
    return rows


def test_archive_backed_loader_is_read_only_and_never_extracts_round_zero(
    tmp_path: Path, adapter: ModuleType
) -> None:
    evidence = _prepare_round_zero_archive(tmp_path, adapter)
    adapter.execute_rotation(
        runtime_root=evidence["runtime"],
        campaign_id=CAMPAIGN_ID,
        execution_id=EXECUTION_ID,
        cell_metadata=CELL_METADATA,
        authority_metadata=evidence["authority"],
        rotation_authority=evidence["rotation_authority"],
        exact_authorization_flag=adapter.EXACT_ROTATION_AUTHORIZATION_FLAG,
        compact_path=evidence["compact_path"],
        worker_receipt_path=evidence["worker"],
        guard_receipt_path=evidence["guard"],
        log_path=evidence["log"],
        limits=evidence["limits"],
        created_at_utc="2026-08-17T12:01:00Z",
        completed_at_utc="2026-08-17T12:02:00Z",
    )
    before = _filesystem_snapshot(evidence["runtime"])

    loaded = adapter.load_archive_backed_cell(
        runtime_root=evidence["runtime"],
        campaign_id=CAMPAIGN_ID,
        execution_id=EXECUTION_ID,
        cell_metadata=CELL_METADATA,
        authority_metadata=evidence["authority"],
        rotation_authority=evidence["rotation_authority"],
        compact_path=evidence["compact_path"],
        worker_receipt_path=evidence["worker"],
        guard_receipt_path=evidence["guard"],
        log_path=evidence["log"],
        limits=evidence["limits"],
    )

    after = _filesystem_snapshot(evidence["runtime"])
    assert after == before
    assert loaded["accepted_rows"] == []
    assert loaded["summary_artifact_status"] == "not_applicable_round_zero"
    assert "summary" not in evidence["compact"]["source_artifact_bindings"]
    assert not (
        evidence["runtime"] / "runs" / EXECUTION_ID
    ).exists(), "loader must not extract the archived run tree"


@pytest.mark.parametrize(
    ("state", "without_authority", "with_authority"),
    [
        ("empty", "await_completed_cell", "await_completed_cell"),
        ("direct_unarchived", "prepare_archive", "prepare_archive"),
        (
            "archive_published_pending_manifest",
            "prepare_archive",
            "prepare_archive",
        ),
        (
            "manifest_published_pending_closure",
            "prepare_archive",
            "prepare_archive",
        ),
        (
            "closure_published_pending_intent",
            "blocked_missing_exact_rotation_authority",
            "execute_rotation",
        ),
        (
            "intent_published_pending_rename",
            "blocked_missing_exact_rotation_authority",
            "execute_rotation",
        ),
        (
            "retiring_pending_removal",
            "blocked_missing_exact_rotation_authority",
            "execute_rotation",
        ),
        (
            "cleanup_receipt_pending",
            "blocked_missing_exact_rotation_authority",
            "execute_rotation",
        ),
        (
            "archived_closed",
            "load_archive_backed_cell",
            "load_archive_backed_cell",
        ),
    ],
)
def test_restart_action_covers_every_legal_strict_state(
    adapter: ModuleType,
    state: str,
    without_authority: str,
    with_authority: str,
) -> None:
    observed = {"state": state, "stale_archive_temporaries": []}
    assert adapter.archive_restart_action(
        observed, exact_authorization_flag=None
    ) == without_authority
    assert adapter.archive_restart_action(
        observed,
        exact_authorization_flag=adapter.EXACT_ROTATION_AUTHORIZATION_FLAG,
    ) == with_authority


def test_restart_action_rejects_unknown_or_unsafe_stale_state(
    adapter: ModuleType,
) -> None:
    with pytest.raises(adapter.MaximumK50ArchiveError):
        adapter.archive_restart_action(
            {"state": "impossible", "stale_archive_temporaries": []},
            exact_authorization_flag=None,
        )
    assert adapter.archive_restart_action(
        {"state": "direct_unarchived", "stale_archive_temporaries": ["x"]},
        exact_authorization_flag=None,
    ) == "prepare_archive"
    with pytest.raises(adapter.MaximumK50ArchiveError):
        adapter.archive_restart_action(
            {
                "state": "closure_published_pending_intent",
                "stale_archive_temporaries": ["x"],
            },
            exact_authorization_flag=adapter.EXACT_ROTATION_AUTHORIZATION_FLAG,
        )


def test_execute_rotation_resumes_after_intent_publication_crash(
    tmp_path: Path, adapter: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _prepare_round_zero_archive(tmp_path, adapter)
    paths = adapter.CellArchivePaths(evidence["runtime"], EXECUTION_ID)
    original_complete = adapter.strict_archive.complete_safe_tree_rotation

    def crash_before_rename(**_kwargs: Any) -> dict[str, Any]:
        raise OSError("simulated crash after durable intent")

    monkeypatch.setattr(
        adapter.strict_archive, "complete_safe_tree_rotation", crash_before_rename
    )
    with pytest.raises(OSError, match="simulated crash"):
        adapter.execute_rotation(
            runtime_root=evidence["runtime"],
            campaign_id=CAMPAIGN_ID,
            execution_id=EXECUTION_ID,
            cell_metadata=CELL_METADATA,
            authority_metadata=evidence["authority"],
            rotation_authority=evidence["rotation_authority"],
            exact_authorization_flag=adapter.EXACT_ROTATION_AUTHORIZATION_FLAG,
            compact_path=evidence["compact_path"],
            worker_receipt_path=evidence["worker"],
            guard_receipt_path=evidence["guard"],
            log_path=evidence["log"],
            limits=evidence["limits"],
            created_at_utc="2026-08-17T12:01:00Z",
        )
    assert adapter.strict_archive.inspect_rotation_state(paths)["state"] == (
        "intent_published_pending_rename"
    )

    monkeypatch.setattr(
        adapter.strict_archive, "complete_safe_tree_rotation", original_complete
    )
    loaded = adapter.execute_rotation(
        runtime_root=evidence["runtime"],
        campaign_id=CAMPAIGN_ID,
        execution_id=EXECUTION_ID,
        cell_metadata=CELL_METADATA,
        authority_metadata=evidence["authority"],
        rotation_authority=evidence["rotation_authority"],
        exact_authorization_flag=adapter.EXACT_ROTATION_AUTHORIZATION_FLAG,
        compact_path=evidence["compact_path"],
        worker_receipt_path=evidence["worker"],
        guard_receipt_path=evidence["guard"],
        log_path=evidence["log"],
        limits=evidence["limits"],
        completed_at_utc="2026-08-17T12:02:00Z",
    )
    assert adapter.strict_archive.inspect_rotation_state(paths)["state"] == (
        "archived_closed"
    )
    assert loaded["accepted_controller_rounds"] == 0


def test_retiring_restart_rejects_external_tamper_before_removal(
    tmp_path: Path, adapter: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _prepare_round_zero_archive(tmp_path, adapter)
    paths = adapter.CellArchivePaths(evidence["runtime"], EXECUTION_ID)
    original_rmtree = adapter.strict_archive.shutil.rmtree

    def crash_during_removal(_root: Path) -> None:
        raise OSError("simulated crash with retiring tree intact")

    crash_during_removal.avoids_symlink_attacks = True
    monkeypatch.setattr(adapter.strict_archive.shutil, "rmtree", crash_during_removal)
    with pytest.raises(OSError, match="retiring tree intact"):
        adapter.execute_rotation(
            runtime_root=evidence["runtime"],
            campaign_id=CAMPAIGN_ID,
            execution_id=EXECUTION_ID,
            cell_metadata=CELL_METADATA,
            authority_metadata=evidence["authority"],
            rotation_authority=evidence["rotation_authority"],
            exact_authorization_flag=adapter.EXACT_ROTATION_AUTHORIZATION_FLAG,
            compact_path=evidence["compact_path"],
            worker_receipt_path=evidence["worker"],
            guard_receipt_path=evidence["guard"],
            log_path=evidence["log"],
            limits=evidence["limits"],
            created_at_utc="2026-08-17T12:01:00Z",
        )
    assert adapter.strict_archive.inspect_rotation_state(paths)["state"] == (
        "retiring_pending_removal"
    )
    monkeypatch.setattr(adapter.strict_archive.shutil, "rmtree", original_rmtree)
    evidence["worker"].write_text('{"status":"tampered"}\n', encoding="utf-8")

    with pytest.raises(adapter.MaximumK50ArchiveError):
        adapter.execute_rotation(
            runtime_root=evidence["runtime"],
            campaign_id=CAMPAIGN_ID,
            execution_id=EXECUTION_ID,
            cell_metadata=CELL_METADATA,
            authority_metadata=evidence["authority"],
            rotation_authority=evidence["rotation_authority"],
            exact_authorization_flag=adapter.EXACT_ROTATION_AUTHORIZATION_FLAG,
            compact_path=evidence["compact_path"],
            worker_receipt_path=evidence["worker"],
            guard_receipt_path=evidence["guard"],
            log_path=evidence["log"],
            limits=evidence["limits"],
        )
    assert adapter.strict_archive.inspect_rotation_state(paths)["state"] == (
        "retiring_pending_removal"
    )
    assert paths.retiring_root.is_dir()
    assert not paths.cleanup_receipt_path.exists()


def test_loader_rejects_archive_byte_tamper(
    tmp_path: Path, adapter: ModuleType
) -> None:
    evidence = _prepare_round_zero_archive(tmp_path, adapter)
    adapter.execute_rotation(
        runtime_root=evidence["runtime"],
        campaign_id=CAMPAIGN_ID,
        execution_id=EXECUTION_ID,
        cell_metadata=CELL_METADATA,
        authority_metadata=evidence["authority"],
        rotation_authority=evidence["rotation_authority"],
        exact_authorization_flag=adapter.EXACT_ROTATION_AUTHORIZATION_FLAG,
        compact_path=evidence["compact_path"],
        worker_receipt_path=evidence["worker"],
        guard_receipt_path=evidence["guard"],
        log_path=evidence["log"],
        limits=evidence["limits"],
        created_at_utc="2026-08-17T12:01:00Z",
        completed_at_utc="2026-08-17T12:02:00Z",
    )
    archive = adapter.CellArchivePaths(
        evidence["runtime"], EXECUTION_ID
    ).archive_path
    with archive.open("r+b") as stream:
        stream.seek(max(1, archive.stat().st_size // 2))
        original = stream.read(1)
        stream.seek(-1, 1)
        stream.write(bytes([original[0] ^ 0x01]))

    with pytest.raises(adapter.MaximumK50ArchiveError):
        adapter.load_archive_backed_cell(
            runtime_root=evidence["runtime"],
            campaign_id=CAMPAIGN_ID,
            execution_id=EXECUTION_ID,
            cell_metadata=CELL_METADATA,
            authority_metadata=evidence["authority"],
            rotation_authority=evidence["rotation_authority"],
            compact_path=evidence["compact_path"],
            worker_receipt_path=evidence["worker"],
            guard_receipt_path=evidence["guard"],
            log_path=evidence["log"],
            limits=evidence["limits"],
        )


def test_compact_payload_supports_exact_maximum_without_terminal_attempt(
    tmp_path: Path, adapter: ModuleType
) -> None:
    files = _prepare_evidence_files(tmp_path)
    summary = files["run_root"] / "summary/summary.json"
    summary.parent.mkdir()
    summary.write_text('{"accepted_controller_rounds":50}\n', encoding="utf-8")
    files["source_artifacts"]["summary"] = summary
    completion = _digested(
        {
            "schema": (
                "paper_i_ra_all6_adaptive_maximum_k50_cell_completion_v1"
            ),
            "campaign_id": CAMPAIGN_ID,
            "execution_id": EXECUTION_ID,
            "completion_kind": "reached_maximum_controller_rounds_v1",
            "maximum_controller_rounds": 50,
            "accepted_controller_rounds": 50,
            "terminal_attempted_controller_round": None,
            "terminal_controller_outcome": None,
            "summary_artifact_status": "present",
            "checkpoint_file_sha256": hashlib.sha256(
                files["source_artifacts"]["checkpoint"].read_bytes()
            ).hexdigest(),
            "paper_i_summary_sha256": hashlib.sha256(
                summary.read_bytes()
            ).hexdigest(),
        }
    )
    outcome = _digested(
        {
            "schema": "paper_i_ra_all6_maximum_k50_cell_outcome_v1",
            "execution_id": EXECUTION_ID,
            "completion_kind": "reached_maximum_controller_rounds_v1",
            "accepted_controller_rounds": 50,
        }
    )
    compact = adapter.build_compact_payload(
        runtime_root=files["runtime"],
        campaign_id=CAMPAIGN_ID,
        execution_id=EXECUTION_ID,
        cell_metadata=CELL_METADATA,
        cell_completion=completion,
        accepted_rows=[_accepted_row(round_number) for round_number in range(1, 51)],
        terminal_attempt=None,
        cell_outcome=outcome,
        worker_receipt_path=files["worker"],
        guard_receipt_path=files["guard"],
        log_path=files["log"],
        source_artifact_paths=files["source_artifacts"],
    )
    assert compact["accepted_controller_rounds"] == 50
    assert len(compact["accepted_rows"]) == 50
    assert compact["terminal_attempt"] is None
    assert compact["terminal_attempt_sha256"] is None
