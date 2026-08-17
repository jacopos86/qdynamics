from __future__ import annotations

import fcntl
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "handoff_local_page12_strong5_to_matched_singleton12_20260815.py"
)
TEST_TARGET_RUNNER_SOURCE = '''#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


RUNTIME_DIR = Path(__file__).resolve().parent / "output/local_runs/matched12"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _load_digested(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("sha256") != hashlib.sha256(
        _canonical_json_bytes(unsigned)
    ).hexdigest():
        raise RuntimeError(f"deep receipt self digest drifted: {path}")
    return value


def validate_completed_terminal_read_only() -> dict[str, Any]:
    terminal = _load_digested(RUNTIME_DIR / "terminal_receipt.json")
    closure = _load_digested(RUNTIME_DIR / "archive_backed_closure.json")
    if (
        closure.get("status") != "passed_archive_backed_terminal_closure"
        or terminal.get("archive_backed_closure_sha256")
        != closure.get("sha256")
    ):
        raise RuntimeError("deep archive-backed closure drifted")
    return terminal
'''


@pytest.fixture()
def gate() -> Any:
    name = "paper_i_matched_singleton12_handoff_gate_test"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_digested(gate: Any, path: Path, value: dict[str, Any]) -> dict[str, Any]:
    payload = gate._digested(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gate._canonical_json_bytes(payload) + b"\n")
    return payload


def _source_prerequisite(gate: Any) -> dict[str, Any]:
    return {
        "runner_path": gate.SOURCE_RUNNER.as_posix(),
        "runner_sha256": gate.SOURCE_RUNNER_SHA256,
        "activation_dir": gate.SOURCE_ACTIVATION_DIR.as_posix(),
        "activation_manifest_sha256": gate.SOURCE_ACTIVATION_SHA256,
        "runtime_dir": gate.SOURCE_RUNTIME_DIR.as_posix(),
        "runtime_manifest_sha256": gate.SOURCE_RUNTIME_SHA256,
        "terminal_schema": gate.SOURCE_TERMINAL_SCHEMA,
        "terminal_status": gate.SOURCE_TERMINAL_STATUS,
        "final_status": gate.SOURCE_FINAL_STATUS,
        "execution_ids": list(gate.SOURCE_EXECUTION_IDS),
    }


def _target_cells() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    regimes = (
        ("weak_weak", 3),
        ("intermediate_weak", 3),
        ("strong_weak_u8", 3),
        ("weak_strong", 7),
        ("intermediate_strong", 7),
        ("strong_strong_u8", 7),
    )
    for regime, n_ph in regimes:
        for method in ("ra_singleton_plateau", "append_singleton"):
            cells.append(
                {
                    "execution_id": f"matched12__{regime}__nph{n_ph}__{method}",
                    "method": method,
                    "regime": regime,
                    "n_ph": n_ph,
                }
            )
    return cells


@pytest.fixture()
def target_contract(
    gate: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    repo_root = tmp_path / "repo"
    activation_dir = repo_root / "activation"
    runtime_dir = repo_root / "output/local_runs/matched12"
    runner = repo_root / "run_matched12.py"
    activation_dir.mkdir(parents=True)
    runner.write_text(TEST_TARGET_RUNNER_SOURCE, encoding="utf-8")
    monkeypatch.setattr(gate, "REPO_ROOT", repo_root)

    planning = _write_digested(
        gate,
        activation_dir / "planning_manifest.json",
        {
            "schema": "test_target_planning_v1",
            "status": "passed_target_plan_not_authorized",
            "execution_authorized": False,
            "submission_authorized": False,
        },
    )
    plan = _write_digested(
        gate,
        activation_dir / "execution_plan.json",
        {"schema": "test_target_plan_v1", "status": "passed"},
    )
    authorization = _write_digested(
        gate,
        activation_dir / "execution_authorization.json",
        {
            "schema": "test_target_authorization_v1",
            "status": "passed",
            "execution_authorized": True,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        },
    )
    activation = _write_digested(
        gate,
        activation_dir / "activation_manifest.json",
        {
            "schema": "test_target_activation_v1",
            "status": "passed_local_activation_authorized",
            "execution_authorized": True,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
            "execution_authorization_sha256": authorization["sha256"],
        },
    )
    parity = _write_digested(
        gate,
        activation_dir / "scientific_parity_canary.json",
        {
            "schema": "test_target_parity_v1",
            "status": "passed_exact_scientific_parity",
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        },
    )
    fingerprint = gate._live_runtime_fingerprint()
    (activation_dir / "runtime_fingerprint.json").write_bytes(
        gate._canonical_json_bytes(fingerprint) + b"\n"
    )

    authority_paths = {
        "planning_manifest": activation_dir / "planning_manifest.json",
        "execution_plan": activation_dir / "execution_plan.json",
        "execution_authorization": activation_dir / "execution_authorization.json",
        "activation_manifest": activation_dir / "activation_manifest.json",
        "scientific_parity_canary": activation_dir / "scientific_parity_canary.json",
        "runtime_fingerprint": activation_dir / "runtime_fingerprint.json",
    }
    authority_bindings = {
        name: gate._binding(path, canonical=True)
        for name, path in authority_paths.items()
    }
    target = {
        "repo_root": repo_root.as_posix(),
        "runner_path": runner.as_posix(),
        "runner_sha256": gate._sha256_file(runner),
        "activation_dir": activation_dir.as_posix(),
        "runtime_dir": runtime_dir.as_posix(),
        "maximum_concurrency": 1,
        "command": [
            gate.PYTHON_EXECUTABLE.as_posix(),
            "-B",
            runner.as_posix(),
            "--run-campaign",
        ],
        "environment": dict(gate.REQUIRED_NUMERICAL_ENVIRONMENT),
        "handoff_receipt_environment_variable": gate.HANDOFF_RECEIPT_ENV,
        "handoff_token_environment_variable": gate.HANDOFF_TOKEN_ENV,
        "handoff_lock_fd_environment_variable": gate.HANDOFF_LOCK_FD_ENV,
        "cells": _target_cells(),
        "authority_bindings": authority_bindings,
        "expected_parity_status": "passed_exact_scientific_parity",
        "minimum_free_disk_bytes": 31 * 1024**3,
        "minimum_available_memory_bytes": gate.HARD_MINIMUM_AVAILABLE_MEMORY_BYTES,
        "scientific_overlap_markers": [
            [runner.as_posix(), "--run-campaign"]
        ],
        "expected_terminal": {
            "path": (runtime_dir / "terminal_receipt.json").as_posix(),
            "schema": "test_matched12_terminal_v1",
            "status": "passed_all_twelve_cells_immutable_closure",
        },
    }
    contract_path = tmp_path / "target_contract.json"
    contract = _write_digested(
        gate,
        contract_path,
        {
            "schema": gate.TARGET_CONTRACT_SCHEMA,
            "status": gate.TARGET_CONTRACT_STATUS,
            "created_at_utc": "2026-08-15T00:00:00Z",
            "gate_script_path": gate.GATE_PATH.as_posix(),
            "gate_script_sha256": gate._sha256_file(gate.GATE_PATH),
            "source_prerequisite": _source_prerequisite(gate),
            "target": target,
            "execution_authorized": True,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        },
    )
    return {
        "path": contract_path,
        "contract": contract,
        "target": target,
        "runner": runner,
        "runtime_dir": runtime_dir,
        "activation": activation,
        "planning": planning,
        "plan": plan,
        "parity": parity,
    }


def _passing_capacity(gate: Any) -> dict[str, Any]:
    return gate._digested(
        {
            "schema": "paper_i_matched_singleton12_handoff_capacity_v1",
            "observed_at_utc": "2026-08-15T00:00:00Z",
            "probe_path": "/tmp",
            "available_memory_bytes": 16 * 1024**3,
            "free_disk_bytes": 128 * 1024**3,
        }
    )


def test_contract_closes_exact_matched_six_plus_six_matrix(
    gate: Any, target_contract: dict[str, Any]
) -> None:
    context = gate._load_target_contract(
        target_contract["path"],
        claim_exists=False,
        capacity_fn=lambda _path: _passing_capacity(gate),
        overlap_fn=lambda _markers: [],
    )

    assert context["contract"]["sha256"] == target_contract["contract"]["sha256"]
    assert context["runtime_state"] == {"state": "absent"}
    assert context["paper_adoption_authorized"] is False
    assert context["paper_evidence_adoption_authorized"] is False
    assert len(context["target"]["cells"]) == 12


def test_contract_rejects_runner_byte_drift(
    gate: Any, target_contract: dict[str, Any]
) -> None:
    target_contract["runner"].write_text("# drift\n", encoding="utf-8")

    with pytest.raises(gate.HandoffGateError, match="runner"):
        gate._load_target_contract(
            target_contract["path"],
            claim_exists=False,
            capacity_fn=lambda _path: _passing_capacity(gate),
            overlap_fn=lambda _markers: [],
        )


@pytest.mark.parametrize("paper_adoption", ["missing", True])
def test_contract_rejects_missing_or_true_paper_adoption_authority(
    gate: Any,
    target_contract: dict[str, Any],
    paper_adoption: str | bool,
) -> None:
    unsigned = {
        key: value
        for key, value in target_contract["contract"].items()
        if key != "sha256"
    }
    if paper_adoption == "missing":
        unsigned.pop("paper_adoption_authorized")
    else:
        unsigned["paper_adoption_authorized"] = paper_adoption
    _write_digested(gate, target_contract["path"], unsigned)

    with pytest.raises(gate.HandoffGateError, match="authority drifted"):
        gate._load_target_contract(
            target_contract["path"],
            claim_exists=False,
            capacity_fn=lambda _path: _passing_capacity(gate),
            overlap_fn=lambda _markers: [],
        )


def test_contract_rejects_initial_claim_below_bound_disk_floor(
    gate: Any, target_contract: dict[str, Any]
) -> None:
    low = _passing_capacity(gate)
    low = gate._digested(
        {
            **{key: value for key, value in low.items() if key != "sha256"},
            "free_disk_bytes": (
                target_contract["target"]["minimum_free_disk_bytes"] - 1
            ),
        }
    )

    with pytest.raises(gate.HandoffGateError, match="capacity"):
        gate._load_target_contract(
            target_contract["path"],
            claim_exists=False,
            capacity_fn=lambda _path: low,
            overlap_fn=lambda _markers: [],
        )


def test_contract_replayable_after_claim_skips_initial_disk_floor(
    gate: Any, target_contract: dict[str, Any]
) -> None:
    target_contract["runtime_dir"].mkdir(parents=True)
    low_disk = gate._digested(
        {
            **{
                key: value
                for key, value in _passing_capacity(gate).items()
                if key != "sha256"
            },
            "free_disk_bytes": 1,
        }
    )

    context = gate._load_target_contract(
        target_contract["path"],
        claim_exists=True,
        capacity_fn=lambda _path: low_disk,
        overlap_fn=lambda _markers: [],
    )

    assert context["runtime_state"] == {"state": "replayable_after_claim"}
    assert context["capacity"]["free_disk_bytes"] == 1


def test_contract_replayable_after_claim_still_rejects_low_memory(
    gate: Any, target_contract: dict[str, Any]
) -> None:
    target_contract["runtime_dir"].mkdir(parents=True)
    low_memory = gate._digested(
        {
            **{
                key: value
                for key, value in _passing_capacity(gate).items()
                if key != "sha256"
            },
            "free_disk_bytes": 1,
            "available_memory_bytes": (
                target_contract["target"]["minimum_available_memory_bytes"] - 1
            ),
        }
    )

    with pytest.raises(gate.HandoffGateError, match="memory capacity"):
        gate._load_target_contract(
            target_contract["path"],
            claim_exists=True,
            capacity_fn=lambda _path: low_memory,
            overlap_fn=lambda _markers: [],
        )


def test_contract_replayable_after_claim_still_rejects_overlap(
    gate: Any, target_contract: dict[str, Any]
) -> None:
    target_contract["runtime_dir"].mkdir(parents=True)
    low_disk = gate._digested(
        {
            **{
                key: value
                for key, value in _passing_capacity(gate).items()
                if key != "sha256"
            },
            "free_disk_bytes": 1,
        }
    )

    with pytest.raises(gate.HandoffGateError, match="overlaps the target"):
        gate._load_target_contract(
            target_contract["path"],
            claim_exists=True,
            capacity_fn=lambda _path: low_disk,
            overlap_fn=lambda _markers: ["123 overlapping scientific command"],
        )


def test_source_terminal_uses_native_closure_validator_and_exact_final_status(
    gate: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_dir = tmp_path / "source_runtime"
    activation_dir = tmp_path / "source_activation"
    (runtime_dir / "status").mkdir(parents=True)
    activation_dir.mkdir()
    terminal = _write_digested(
        gate,
        runtime_dir / "terminal_receipt.json",
        {
            "schema": gate.SOURCE_TERMINAL_SCHEMA,
            "status": gate.SOURCE_TERMINAL_STATUS,
            "execution_ids": list(gate.SOURCE_EXECUTION_IDS),
            "completed_execution_ids": list(gate.SOURCE_EXECUTION_IDS),
        },
    )
    runtime = {"sha256": gate.SOURCE_RUNTIME_SHA256}
    status = _source_status_payload(
        gate,
        runtime=runtime,
        status=gate.SOURCE_FINAL_STATUS,
        completed=list(gate.SOURCE_EXECUTION_IDS),
        current_execution_id=None,
        terminal_receipt_sha256=terminal["sha256"],
    )
    (runtime_dir / "status/campaign.json").write_bytes(
        gate._canonical_json_bytes(status) + b"\n"
    )
    calls: list[str] = []
    source = SimpleNamespace(
        TARGET_EXECUTION_IDS=gate.SOURCE_EXECUTION_IDS,
        _load_worker=lambda: calls.append("load_worker") or object(),
        _closed_inputs=lambda _worker: (calls.append("closed_inputs") or ({}, [])),
        _validate_activation=lambda *_args, **_kwargs: (
            calls.append("validate_activation")
            or ({"sha256": gate.SOURCE_ACTIVATION_SHA256}, {}, {}, {})
        ),
        _ensure_runtime=lambda *_args, **_kwargs: (
            calls.append("ensure_runtime")
            or runtime
        ),
        _validate_terminal_receipt=lambda **_kwargs: (
            calls.append("validate_terminal") or terminal
        ),
        _load_digested=lambda path, label: gate._load_digested(path, label=label),
        _status_payload=lambda **kwargs: _source_status_payload(gate, **kwargs),
    )
    monkeypatch.setattr(gate, "SOURCE_RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(gate, "SOURCE_ACTIVATION_DIR", activation_dir)

    closure = gate._validate_source_terminal(source)

    assert calls == [
        "load_worker",
        "closed_inputs",
        "validate_activation",
        "ensure_runtime",
        "validate_terminal",
    ]
    assert closure["terminal"] == terminal
    assert closure["status"] == status
    assert closure["status_repair"]["repair_performed"] is False


def _source_status_payload(
    gate: Any,
    *,
    runtime: dict[str, Any],
    status: str,
    completed: list[str],
    current_execution_id: str | None,
    child_pid: int | None = None,
    metrics: dict[str, Any] | None = None,
    failure: dict[str, Any] | None = None,
    terminal_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": "test_source_status_v1",
        "status": status,
        "updated_at_utc": "2026-08-15T00:00:00Z",
        "runtime_manifest_sha256": runtime["sha256"],
        "execution_ids": list(gate.SOURCE_EXECUTION_IDS),
        "completed_execution_ids": list(completed),
        "current_execution_id": current_execution_id,
        "child_pid": child_pid,
        "maximum_concurrency": 1,
    }
    if metrics is not None:
        value["metrics"] = dict(metrics)
    if failure is not None:
        value["failure"] = dict(failure)
    if terminal_receipt_sha256 is not None:
        value["terminal_receipt_sha256"] = terminal_receipt_sha256
    return gate._digested(value)


def _source_probe_fixture(
    gate: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    status_kind: str,
    deep_terminal_error: bool = False,
) -> tuple[Path, dict[str, Any], dict[str, Any], list[bool]]:
    runtime_dir = tmp_path / "source_runtime"
    activation_dir = tmp_path / "source_activation"
    (runtime_dir / "status").mkdir(parents=True)
    activation_dir.mkdir()
    (runtime_dir / "campaign.lock").touch()
    runner = tmp_path / "source_runner.py"
    runner.write_text("# pinned source runner fixture\n", encoding="utf-8")
    runtime = {"sha256": gate.SOURCE_RUNTIME_SHA256}
    terminal = _write_digested(
        gate,
        runtime_dir / "terminal_receipt.json",
        {
            "schema": gate.SOURCE_TERMINAL_SCHEMA,
            "status": gate.SOURCE_TERMINAL_STATUS,
            "execution_ids": list(gate.SOURCE_EXECUTION_IDS),
            "completed_execution_ids": list(gate.SOURCE_EXECUTION_IDS),
        },
    )
    if status_kind == "penultimate":
        status = _source_status_payload(
            gate,
            runtime=runtime,
            status="cell_passed_pending_remaining",
            completed=list(gate.SOURCE_EXECUTION_IDS),
            current_execution_id=None,
        )
    elif status_kind == "incomplete":
        status = _source_status_payload(
            gate,
            runtime=runtime,
            status="cell_passed_pending_remaining",
            completed=list(gate.SOURCE_EXECUTION_IDS[:-1]),
            current_execution_id=None,
        )
    elif status_kind == "tampered":
        status = _source_status_payload(
            gate,
            runtime=runtime,
            status="cell_passed_pending_remaining",
            completed=list(gate.SOURCE_EXECUTION_IDS),
            current_execution_id=None,
            failure={"error_type": "InjectedTamper"},
        )
    elif status_kind == "final":
        status = _source_status_payload(
            gate,
            runtime=runtime,
            status=gate.SOURCE_FINAL_STATUS,
            completed=list(gate.SOURCE_EXECUTION_IDS),
            current_execution_id=None,
            terminal_receipt_sha256=terminal["sha256"],
        )
    else:
        raise AssertionError(status_kind)
    (runtime_dir / "status/campaign.json").write_bytes(
        gate._canonical_json_bytes(status) + b"\n"
    )
    writes_under_lock: list[bool] = []

    def validate_terminal(**_kwargs: Any) -> dict[str, Any]:
        if deep_terminal_error:
            raise gate.HandoffGateError("injected deep terminal tamper")
        return terminal

    def write_status(path: Path, value: dict[str, Any]) -> None:
        contender = (runtime_dir / "campaign.lock").open("r", encoding="utf-8")
        try:
            try:
                fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                writes_under_lock.append(True)
            else:
                writes_under_lock.append(False)
                fcntl.flock(contender.fileno(), fcntl.LOCK_UN)
        finally:
            contender.close()
        gate._write_json_atomic(path, value)

    source = SimpleNamespace(
        TARGET_EXECUTION_IDS=gate.SOURCE_EXECUTION_IDS,
        _load_worker=lambda: object(),
        _closed_inputs=lambda _worker: ({}, []),
        _validate_activation=lambda *_args, **_kwargs: (
            {"sha256": gate.SOURCE_ACTIVATION_SHA256},
            {},
            {},
            {},
        ),
        _ensure_runtime=lambda *_args, **_kwargs: runtime,
        _validate_terminal_receipt=validate_terminal,
        _load_digested=lambda path, label: gate._load_digested(path, label=label),
        _status_payload=lambda **kwargs: _source_status_payload(gate, **kwargs),
        _write_json_atomic=write_status,
    )
    monkeypatch.setattr(gate, "SOURCE_RUNNER", runner)
    monkeypatch.setattr(gate, "SOURCE_RUNNER_SHA256", gate._sha256_file(runner))
    monkeypatch.setattr(gate, "SOURCE_RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(gate, "SOURCE_ACTIVATION_DIR", activation_dir)
    monkeypatch.setattr(gate, "_load_source_runner", lambda: source)
    return runtime_dir, terminal, status, writes_under_lock


def _assert_source_lock_released(runtime_dir: Path) -> None:
    with (runtime_dir / "campaign.lock").open("r", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def test_run_gate_repairs_exact_penultimate_status_under_campaign_lock(
    gate: Any,
    target_contract: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = gate._load_target_contract(
        target_contract["path"],
        claim_exists=False,
        capacity_fn=lambda _path: _passing_capacity(gate),
        overlap_fn=lambda _markers: [],
    )
    runtime_dir, terminal, penultimate, writes_under_lock = _source_probe_fixture(
        gate,
        tmp_path,
        monkeypatch,
        status_kind="penultimate",
    )
    state_dir = tmp_path / "gate_state"

    with pytest.raises(gate.HandoffGateError, match="exec returned"):
        gate.run_gate(
            contract_path=target_contract["path"],
            state_dir=state_dir,
            contract_loader=lambda _path, claim_exists: context,
            exec_fn=lambda *_args: None,
            sleep_fn=lambda _seconds: None,
        )

    repaired = gate._load_digested(
        runtime_dir / "status/campaign.json", label="repaired source final status"
    )
    claim = gate._load_digested(
        state_dir / "handoff_receipt.json", label="repaired source handoff claim"
    )
    claim_bytes = (state_dir / "handoff_receipt.json").read_bytes()
    with pytest.raises(gate.HandoffGateError, match="exec returned"):
        gate.run_gate(
            contract_path=target_contract["path"],
            state_dir=state_dir,
            contract_loader=lambda _path, claim_exists: context,
            exec_fn=lambda *_args: None,
            sleep_fn=lambda _seconds: None,
        )
    gate_status = gate._load_digested(
        state_dir / "status.json", label="repaired source gate status"
    )
    repair = claim["source_final_status_repair"]
    assert repaired["status"] == gate.SOURCE_FINAL_STATUS
    assert repaired["terminal_receipt_sha256"] == terminal["sha256"]
    assert repair["repair_performed"] is True
    assert repair["before_status_binding"]["canonical_sha256"] == penultimate[
        "sha256"
    ]
    assert repair["after_status_binding"]["canonical_sha256"] == repaired[
        "sha256"
    ]
    assert gate_status["source_final_status_repair"] == repair
    assert writes_under_lock == [True]
    assert (state_dir / "handoff_receipt.json").read_bytes() == claim_bytes
    _assert_source_lock_released(runtime_dir)


def test_preflight_observes_penultimate_terminal_without_repairing_status(
    gate: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_dir, _terminal, penultimate, writes_under_lock = _source_probe_fixture(
        gate,
        tmp_path,
        monkeypatch,
        status_kind="penultimate",
    )
    status_path = runtime_dir / "status/campaign.json"
    before_bytes = status_path.read_bytes()
    before_sha256 = gate._sha256_file(status_path)
    before_stat = status_path.stat()

    result = gate.preflight(contract_path=tmp_path / "absent_contract.json")

    after_stat = status_path.stat()
    assert result["status"] == "passed_inert_preflight"
    assert result["source_state"] == "finalizing"
    assert status_path.read_bytes() == before_bytes
    assert gate._sha256_file(status_path) == before_sha256
    assert gate._load_digested(status_path, label="inert penultimate status") == (
        penultimate
    )
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns
    assert after_stat.st_size == before_stat.st_size
    assert writes_under_lock == []
    _assert_source_lock_released(runtime_dir)


@pytest.mark.parametrize("status_kind", ["incomplete", "tampered"])
def test_source_probe_rejects_nonexact_penultimate_without_repair(
    gate: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status_kind: str,
) -> None:
    runtime_dir, _terminal, status, writes_under_lock = _source_probe_fixture(
        gate,
        tmp_path,
        monkeypatch,
        status_kind=status_kind,
    )

    with pytest.raises(gate.HandoffGateError, match="terminal closure drifted"):
        gate._probe_source()

    assert gate._load_digested(
        runtime_dir / "status/campaign.json", label="unrepaired source status"
    ) == status
    assert writes_under_lock == []
    _assert_source_lock_released(runtime_dir)


def test_source_probe_rejects_deep_terminal_tamper_without_repair(
    gate: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_dir, _terminal, status, writes_under_lock = _source_probe_fixture(
        gate,
        tmp_path,
        monkeypatch,
        status_kind="penultimate",
        deep_terminal_error=True,
    )

    with pytest.raises(gate.HandoffGateError, match="deep terminal tamper"):
        gate._probe_source()

    assert gate._load_digested(
        runtime_dir / "status/campaign.json", label="unrepaired source status"
    ) == status
    assert writes_under_lock == []
    _assert_source_lock_released(runtime_dir)


def test_source_probe_accepts_already_final_status_without_repair(
    gate: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_dir, _terminal, final, writes_under_lock = _source_probe_fixture(
        gate,
        tmp_path,
        monkeypatch,
        status_kind="final",
    )

    closure = gate._probe_source()

    repair = closure["status_repair"]
    assert closure["status"] == final
    assert repair["repair_performed"] is False
    assert repair["before_status_binding"] == repair["after_status_binding"]
    assert writes_under_lock == []
    _assert_source_lock_released(runtime_dir)


def test_source_probe_fails_if_supervisor_lock_is_free_without_terminal(
    gate: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = tmp_path / "source_runner.py"
    runner.write_text("# source\n", encoding="utf-8")
    runtime = tmp_path / "runtime"
    (runtime / "status").mkdir(parents=True)
    (runtime / "campaign.lock").touch()
    _write_digested(
        gate,
        runtime / "status/campaign.json",
        {"schema": "test", "status": "running_serial_cell"},
    )
    monkeypatch.setattr(gate, "SOURCE_RUNNER", runner)
    monkeypatch.setattr(gate, "SOURCE_RUNNER_SHA256", gate._sha256_file(runner))
    monkeypatch.setattr(gate, "SOURCE_RUNTIME_DIR", runtime)

    with pytest.raises(gate.HandoffGateError, match="inactive without"):
        gate._probe_source()


def _closed_source(gate: Any, tmp_path: Path) -> dict[str, Any]:
    terminal = _write_digested(
        gate,
        tmp_path / "source_terminal.json",
        {
            "schema": gate.SOURCE_TERMINAL_SCHEMA,
            "status": gate.SOURCE_TERMINAL_STATUS,
        },
    )
    status = _write_digested(
        gate,
        tmp_path / "source_status.json",
        {
            "schema": "test_source_status_v1",
            "status": gate.SOURCE_FINAL_STATUS,
            "terminal_receipt_sha256": terminal["sha256"],
        },
    )
    terminal_binding = gate._binding(
        tmp_path / "source_terminal.json", canonical=True
    )
    status_binding = gate._binding(
        tmp_path / "source_status.json", canonical=True
    )
    status_repair = gate._digested(
        {
            "schema": (
                "paper_i_matched_singleton12_source_final_status_repair_v1"
            ),
            "repair_performed": False,
            "before_status_binding": status_binding,
            "after_status_binding": status_binding,
        }
    )
    return {
        "state": "complete",
        "terminal": terminal,
        "status": status,
        "terminal_binding": terminal_binding,
        "status_binding": status_binding,
        "status_repair": status_repair,
        "final_status_pending": False,
    }


def _publish_initial_claim(
    gate: Any, target_contract: dict[str, Any], tmp_path: Path
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    state_dir = tmp_path / "gate_state"
    source = _closed_source(gate, tmp_path)
    context = gate._load_target_contract(
        target_contract["path"],
        claim_exists=False,
        capacity_fn=lambda _path: _passing_capacity(gate),
        overlap_fn=lambda _markers: [],
    )
    with pytest.raises(gate.HandoffGateError, match="exec returned"):
        gate.run_gate(
            contract_path=target_contract["path"],
            state_dir=state_dir,
            source_probe=lambda: source,
            contract_loader=lambda _path, claim_exists: context,
            exec_fn=lambda *_args: None,
            sleep_fn=lambda _seconds: None,
        )
    claim = gate._load_digested(
        state_dir / "handoff_receipt.json", label="completed handoff receipt"
    )
    return state_dir, source, context, claim


def _write_target_terminal(
    gate: Any,
    target_contract: dict[str, Any],
    context: dict[str, Any],
    claim: dict[str, Any],
    *,
    bind_archive_closure: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime_dir = target_contract["runtime_dir"]
    runtime_dir.mkdir(parents=True, exist_ok=True)
    closure = _write_digested(
        gate,
        runtime_dir / "archive_backed_closure.json",
        {
            "schema": "test_archive_backed_closure_v1",
            "status": "passed_archive_backed_terminal_closure",
            "cell_count": 12,
        },
    )
    terminal_fields: dict[str, Any] = {
        "schema": target_contract["target"]["expected_terminal"]["schema"],
        "status": target_contract["target"]["expected_terminal"]["status"],
        "execution_ids": [
            cell["execution_id"] for cell in target_contract["target"]["cells"]
        ],
        "completed_execution_ids": [
            cell["execution_id"] for cell in target_contract["target"]["cells"]
        ],
        "activation_manifest_sha256": context["authority"][
            "activation_manifest"
        ]["sha256"],
        "handoff_receipt_sha256": claim["sha256"],
        "execution_authorized": True,
        "submission_authorized": False,
        "paper_adoption_authorized": False,
        "paper_evidence_adoption_authorized": False,
    }
    if bind_archive_closure:
        terminal_fields["archive_backed_closure_sha256"] = closure["sha256"]
    terminal = _write_digested(
        gate,
        runtime_dir / "terminal_receipt.json",
        terminal_fields,
    )
    return terminal, closure


def _completed_target_loader(
    gate: Any, target_contract: dict[str, Any]
) -> Any:
    low_disk = gate._digested(
        {
            **{
                key: value
                for key, value in _passing_capacity(gate).items()
                if key != "sha256"
            },
            "free_disk_bytes": 1,
        }
    )

    def load(path: Path, *, claim_exists: bool) -> dict[str, Any]:
        return gate._load_target_contract(
            path,
            claim_exists=claim_exists,
            capacity_fn=lambda _path: low_disk,
            overlap_fn=lambda _markers: ["ignored after completed validation"],
        )

    return load


@pytest.mark.parametrize("paper_adoption", ["missing", True])
def test_run_rejects_target_context_without_explicit_paper_adoption_denial(
    gate: Any,
    target_contract: dict[str, Any],
    tmp_path: Path,
    paper_adoption: str | bool,
) -> None:
    source = _closed_source(gate, tmp_path)
    context = gate._load_target_contract(
        target_contract["path"],
        claim_exists=False,
        capacity_fn=lambda _path: _passing_capacity(gate),
        overlap_fn=lambda _markers: [],
    )
    context = dict(context)
    if paper_adoption == "missing":
        context.pop("paper_adoption_authorized")
    else:
        context["paper_adoption_authorized"] = paper_adoption

    with pytest.raises(gate.HandoffGateError, match="Target context authority"):
        gate.run_gate(
            contract_path=target_contract["path"],
            state_dir=tmp_path / "gate_state",
            source_probe=lambda: source,
            contract_loader=lambda _path, claim_exists: context,
            exec_fn=lambda *_args: None,
            sleep_fn=lambda _seconds: None,
        )


def test_run_publishes_one_claim_then_execs_fixed_target_with_inherited_lock(
    gate: Any, target_contract: dict[str, Any], tmp_path: Path
) -> None:
    state_dir = tmp_path / "gate_state"
    source = _closed_source(gate, tmp_path)
    context = gate._load_target_contract(
        target_contract["path"],
        claim_exists=False,
        capacity_fn=lambda _path: _passing_capacity(gate),
        overlap_fn=lambda _markers: [],
    )
    observed: dict[str, Any] = {}

    def fake_exec(executable: str, argv: list[str], environment: dict[str, str]) -> None:
        observed.update(executable=executable, argv=argv, environment=environment)

    with pytest.raises(gate.HandoffGateError, match="exec returned"):
        gate.run_gate(
            contract_path=target_contract["path"],
            state_dir=state_dir,
            source_probe=lambda: source,
            contract_loader=lambda _path, claim_exists: context,
            exec_fn=fake_exec,
            sleep_fn=lambda _seconds: None,
        )

    claim = gate._load_digested(
        state_dir / "handoff_receipt.json", label="test handoff receipt"
    )
    status = gate._load_digested(state_dir / "status.json", label="test gate status")
    assert claim["status"] == gate.HANDOFF_RECEIPT_STATUS
    assert claim["source_terminal_sha256"] == source["terminal"]["sha256"]
    assert claim["submission_authorized"] is False
    assert claim["paper_adoption_authorized"] is False
    assert claim["paper_evidence_adoption_authorized"] is False
    assert claim["source_final_status_repair"]["repair_performed"] is False
    assert observed["executable"] == target_contract["target"]["command"][0]
    assert observed["argv"] == target_contract["target"]["command"]
    assert observed["environment"][gate.HANDOFF_RECEIPT_ENV] == (
        state_dir / "handoff_receipt.json"
    ).as_posix()
    assert len(observed["environment"][gate.HANDOFF_TOKEN_ENV]) == 64
    assert int(observed["environment"][gate.HANDOFF_LOCK_FD_ENV]) > 2
    assert status["status"] == "blocked_exec_returned_unexpectedly"
    assert status["source_final_status_repair"] == claim[
        "source_final_status_repair"
    ]


def test_second_gate_cannot_pass_dedicated_handoff_lock(
    gate: Any, target_contract: dict[str, Any], tmp_path: Path
) -> None:
    state_dir = tmp_path / "gate_state"
    state_dir.mkdir()
    stream = (state_dir / "handoff.lock").open("a+", encoding="utf-8")
    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(gate.HandoffGateError, match="owns the lock"):
            gate.run_gate(
                contract_path=target_contract["path"],
                state_dir=state_dir,
                source_probe=lambda: _closed_source(gate, tmp_path),
                exec_fn=lambda *_args: None,
            )
    finally:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def test_existing_claim_is_replayed_without_republication_after_preexec_crash(
    gate: Any, target_contract: dict[str, Any], tmp_path: Path
) -> None:
    state_dir = tmp_path / "gate_state"
    source = _closed_source(gate, tmp_path)
    context = gate._load_target_contract(
        target_contract["path"],
        claim_exists=False,
        capacity_fn=lambda _path: _passing_capacity(gate),
        overlap_fn=lambda _markers: [],
    )
    exec_count = 0

    def returning_exec(*_args: Any) -> None:
        nonlocal exec_count
        exec_count += 1

    for _attempt in range(2):
        with pytest.raises(gate.HandoffGateError, match="exec returned"):
            gate.run_gate(
                contract_path=target_contract["path"],
                state_dir=state_dir,
                source_probe=lambda: source,
                contract_loader=lambda _path, claim_exists: context,
                exec_fn=returning_exec,
                sleep_fn=lambda _seconds: None,
            )

    assert exec_count == 2
    assert gate._load_digested(
        state_dir / "handoff_receipt.json", label="replayed handoff receipt"
    )["status"] == gate.HANDOFF_RECEIPT_STATUS


@pytest.mark.parametrize("paper_adoption", ["missing", True])
def test_replay_rejects_claim_without_explicit_paper_adoption_denial(
    gate: Any,
    target_contract: dict[str, Any],
    tmp_path: Path,
    paper_adoption: str | bool,
) -> None:
    state_dir, source, context, claim = _publish_initial_claim(
        gate, target_contract, tmp_path
    )
    unsigned = {key: value for key, value in claim.items() if key != "sha256"}
    if paper_adoption == "missing":
        unsigned.pop("paper_adoption_authorized")
    else:
        unsigned["paper_adoption_authorized"] = paper_adoption
    _write_digested(gate, state_dir / "handoff_receipt.json", unsigned)
    exec_called = False

    def forbidden_exec(*_args: Any) -> None:
        nonlocal exec_called
        exec_called = True

    with pytest.raises(
        gate.HandoffGateError, match="Existing immutable handoff receipt drifted"
    ):
        gate.run_gate(
            contract_path=target_contract["path"],
            state_dir=state_dir,
            source_probe=lambda: source,
            contract_loader=lambda _path, claim_exists: context,
            exec_fn=forbidden_exec,
            sleep_fn=lambda _seconds: None,
        )

    assert exec_called is False


def test_valid_immutable_claim_replays_below_initial_disk_floor(
    gate: Any, target_contract: dict[str, Any], tmp_path: Path
) -> None:
    state_dir = tmp_path / "gate_state"
    source = _closed_source(gate, tmp_path)
    exec_count = 0

    def returning_exec(*_args: Any) -> None:
        nonlocal exec_count
        exec_count += 1

    def initial_loader(path: Path, *, claim_exists: bool) -> dict[str, Any]:
        return gate._load_target_contract(
            path,
            claim_exists=claim_exists,
            capacity_fn=lambda _path: _passing_capacity(gate),
            overlap_fn=lambda _markers: [],
        )

    with pytest.raises(gate.HandoffGateError, match="exec returned"):
        gate.run_gate(
            contract_path=target_contract["path"],
            state_dir=state_dir,
            source_probe=lambda: source,
            contract_loader=initial_loader,
            exec_fn=returning_exec,
            sleep_fn=lambda _seconds: None,
        )
    claim_path = state_dir / "handoff_receipt.json"
    immutable_claim_bytes = claim_path.read_bytes()

    target_contract["runtime_dir"].mkdir(parents=True)
    replay_capacity = gate._digested(
        {
            **{
                key: value
                for key, value in _passing_capacity(gate).items()
                if key != "sha256"
            },
            "free_disk_bytes": 1,
        }
    )

    def replay_loader(path: Path, *, claim_exists: bool) -> dict[str, Any]:
        assert claim_exists is True
        return gate._load_target_contract(
            path,
            claim_exists=claim_exists,
            capacity_fn=lambda _path: replay_capacity,
            overlap_fn=lambda _markers: [],
        )

    with pytest.raises(gate.HandoffGateError, match="exec returned"):
        gate.run_gate(
            contract_path=target_contract["path"],
            state_dir=state_dir,
            source_probe=lambda: source,
            contract_loader=replay_loader,
            exec_fn=returning_exec,
            sleep_fn=lambda _seconds: None,
        )

    assert exec_count == 2
    assert claim_path.read_bytes() == immutable_claim_bytes


def test_valid_target_terminal_bound_to_claim_is_idempotent_noop(
    gate: Any, target_contract: dict[str, Any], tmp_path: Path
) -> None:
    state_dir, source, context, claim = _publish_initial_claim(
        gate, target_contract, tmp_path
    )
    terminal, _closure = _write_target_terminal(
        gate, target_contract, context, claim
    )

    exec_called = False

    def forbidden_exec(*_args: Any) -> None:
        nonlocal exec_called
        exec_called = True

    result = gate.run_gate(
        contract_path=target_contract["path"],
        state_dir=state_dir,
        source_probe=lambda: source,
        contract_loader=_completed_target_loader(gate, target_contract),
        exec_fn=forbidden_exec,
    )

    assert result["status"] == "target_already_complete"
    assert result["target_terminal_sha256"] == terminal["sha256"]
    assert terminal["paper_adoption_authorized"] is False
    assert result["target_terminal_binding"]["canonical_sha256"] == terminal[
        "sha256"
    ]
    assert exec_called is False


@pytest.mark.parametrize("paper_adoption", ["missing", True])
def test_completed_terminal_requires_explicit_paper_adoption_denial(
    gate: Any,
    target_contract: dict[str, Any],
    tmp_path: Path,
    paper_adoption: str | bool,
) -> None:
    state_dir, source, context, claim = _publish_initial_claim(
        gate, target_contract, tmp_path
    )
    terminal, _closure = _write_target_terminal(
        gate, target_contract, context, claim
    )
    unsigned = {
        key: value for key, value in terminal.items() if key != "sha256"
    }
    if paper_adoption == "missing":
        unsigned.pop("paper_adoption_authorized")
    else:
        unsigned["paper_adoption_authorized"] = paper_adoption
    _write_digested(
        gate,
        target_contract["runtime_dir"] / "terminal_receipt.json",
        unsigned,
    )
    exec_called = False

    def forbidden_exec(*_args: Any) -> None:
        nonlocal exec_called
        exec_called = True

    with pytest.raises(
        gate.HandoffGateError,
        match="Completed target terminal does not bind the authorized handoff",
    ):
        gate.run_gate(
            contract_path=target_contract["path"],
            state_dir=state_dir,
            source_probe=lambda: source,
            contract_loader=_completed_target_loader(gate, target_contract),
            exec_fn=forbidden_exec,
        )

    assert exec_called is False


def test_redigested_incomplete_terminal_is_not_declared_complete(
    gate: Any, target_contract: dict[str, Any], tmp_path: Path
) -> None:
    state_dir, source, context, claim = _publish_initial_claim(
        gate, target_contract, tmp_path
    )
    terminal, _closure = _write_target_terminal(
        gate,
        target_contract,
        context,
        claim,
        bind_archive_closure=False,
    )
    assert gate._load_digested(
        target_contract["runtime_dir"] / "terminal_receipt.json",
        label="re-digested incomplete terminal",
    ) == terminal
    exec_called = False

    def forbidden_exec(*_args: Any) -> None:
        nonlocal exec_called
        exec_called = True

    with pytest.raises(
        gate.HandoffGateError, match="deep terminal validation failed"
    ):
        gate.run_gate(
            contract_path=target_contract["path"],
            state_dir=state_dir,
            source_probe=lambda: source,
            contract_loader=_completed_target_loader(gate, target_contract),
            exec_fn=forbidden_exec,
        )

    assert exec_called is False


def test_redigested_tampered_archive_closure_is_not_declared_complete(
    gate: Any, target_contract: dict[str, Any], tmp_path: Path
) -> None:
    state_dir, source, context, claim = _publish_initial_claim(
        gate, target_contract, tmp_path
    )
    _terminal, original_closure = _write_target_terminal(
        gate, target_contract, context, claim
    )
    tampered_closure = _write_digested(
        gate,
        target_contract["runtime_dir"] / "archive_backed_closure.json",
        {
            "schema": "test_archive_backed_closure_v1",
            "status": "tampered_archive_backed_terminal_closure",
            "cell_count": 12,
        },
    )
    assert tampered_closure["sha256"] != original_closure["sha256"]
    exec_called = False

    def forbidden_exec(*_args: Any) -> None:
        nonlocal exec_called
        exec_called = True

    with pytest.raises(
        gate.HandoffGateError, match="deep terminal validation failed"
    ):
        gate.run_gate(
            contract_path=target_contract["path"],
            state_dir=state_dir,
            source_probe=lambda: source,
            contract_loader=_completed_target_loader(gate, target_contract),
            exec_fn=forbidden_exec,
        )

    assert exec_called is False


def test_preflight_with_absent_target_contract_is_inert(
    gate: Any, tmp_path: Path
) -> None:
    missing = tmp_path / "not_materialized.json"
    payload = gate.preflight(
        contract_path=missing,
        source_probe=lambda: {"state": "running"},
    )

    assert payload["status"] == "passed_inert_preflight"
    assert payload["target_contract_state"] == "absent"
    assert payload["run_ready"] is False
    assert list(tmp_path.iterdir()) == []


def test_no_replace_claim_refuses_existing_bytes(
    gate: Any, tmp_path: Path
) -> None:
    path = tmp_path / "claim.json"
    first = gate._digested({"schema": "test", "value": 1})
    second = gate._digested({"schema": "test", "value": 2})
    gate._write_json_atomic_noreplace(path, first)

    with pytest.raises(FileExistsError):
        gate._write_json_atomic_noreplace(path, second)

    assert json.loads(path.read_text(encoding="utf-8")) == first
