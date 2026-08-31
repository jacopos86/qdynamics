"""Focused reload/fail-closed tests for the APM generic intent bridge."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

import pipelines.time_dynamics.prospective_apm_exchange_intent_bridge as bridge_module
from pipelines.time_dynamics.acquisition_supervisor import (
    SUBMISSION_INTENT_SCHEMA_V1,
)
from pipelines.time_dynamics.prospective_apm_exchange_chtc_package import (
    PACKAGE_MANIFEST_SCHEMA_V1,
    PACKAGE_PREFLIGHT_SCHEMA_V1,
    SUBMISSION_INTENT_TEMPLATE_V1,
    apm_exchange_submission_unit_id,
)
from pipelines.time_dynamics.prospective_apm_exchange_intent_bridge import (
    APMIntentBridgeError,
    BRIDGE_SCHEMA_V1,
    _digest,
    _file_sha,
    load_apm_exchange_intent_bridge,
    seal_apm_exchange_intent,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    EXEC_PROSPECTIVE_APM,
    ExecutionRequest,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _binding(root: Path, path: Path, *, include_bytes: bool = False) -> dict:
    row = {"path": path.relative_to(root).as_posix(), "sha256": _file_sha(path)}
    if include_bytes:
        row["bytes"] = path.stat().st_size
    return row


def _bridge(root: Path) -> tuple[Path, Path]:
    lane = "paper_ii_hh_product_v1"
    unit_id = "prospective-apm-exchange-0123456789abcdef"
    action = "prospective::test::exchange"
    request_sha = "1" * 64
    evidence = root / "evidence"
    names = (
        "execution_request", "work_order", "score_receipt", "planner_receipt",
        "action_registry", "package_preflight", "package_manifest", "package",
        "submit_descriptor", "intent_template",
    )
    files = {name: evidence / f"{name}.json" for name in names}
    for name, path in files.items():
        _write(
            path,
            {"execution_request_sha256": request_sha}
            if name == "execution_request" else {"name": name},
        )
    payload = {
        "package_kind": "apm_exchange",
        **{
            name: _binding(root, path, include_bytes=(name == "package"))
            for name, path in files.items()
        },
        "job_batch_name": "paper-ii-apm-test",
        "first_reconciliation_outcome": "indeterminate",
        "submission_permitted": False,
        "qiskit_compilation": False,
        "production_submission_enabled": False,
    }
    unit = {"unit_id": unit_id, "action_ids": [action], "payload": payload}
    core = {"schema": SUBMISSION_INTENT_SCHEMA_V1, "lane_id": lane, "unit": unit}
    intent = {
        **core,
        "intent_sha256": _digest(core),
        "recorded_at": "2026-08-30T02:30:00+00:00",
    }
    state = root / "output" / lane / "acquisition_supervisor"
    intent_path = state / "submission_intents" / f"{unit_id}.json"
    _write(intent_path, intent)
    generic = {
        "job_batch_name": payload["job_batch_name"],
        "acquisition_unit_id": unit_id,
        "paper_ii_action_id": action,
        "paper_ii_execution_request_sha256": request_sha,
        "intent_sha256": intent["intent_sha256"],
        "recorded_at": intent["recorded_at"],
    }
    bridge_core = {
        "schema": BRIDGE_SCHEMA_V1,
        "source_submission_intent": {
            **_binding(root, intent_path),
            "intent_sha256": intent["intent_sha256"],
        },
        "generic_reconciliation_intent": generic,
        "first_reconciliation_outcome": "indeterminate",
        "submission_permitted": False,
        "scheduler_contacted": False,
        "qiskit_compilation_requested": False,
        "production_submission_enabled": False,
    }
    bridge = {**bridge_core, "bridge_sha256": _digest(bridge_core)}
    bridge_path = state / "intent_bridges" / f"{unit_id}.json"
    _write(bridge_path, bridge)
    return bridge_path, files["package"]


def test_bridge_reloads_exact_six_fields_and_fails_on_bound_file_tamper(
    tmp_path: Path,
) -> None:
    bridge_path, package_path = _bridge(tmp_path)
    loaded = load_apm_exchange_intent_bridge(
        bridge_path, repository_root=tmp_path
    )
    assert set(loaded["generic_reconciliation_intent"]) == {
        "job_batch_name",
        "acquisition_unit_id",
        "paper_ii_action_id",
        "paper_ii_execution_request_sha256",
        "intent_sha256",
        "recorded_at",
    }
    assert loaded["submission_permitted"] is False
    package_path.write_bytes(package_path.read_bytes() + b"tamper")
    with pytest.raises(APMIntentBridgeError, match="digest changed"):
        load_apm_exchange_intent_bridge(bridge_path, repository_root=tmp_path)


def _sealable_campaign(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Path | str]:
    action = "prospective::test::exchange"
    request = ExecutionRequest(
        action_id=action,
        lane_id="paper_ii_hh_product_v1",
        kind="run",
        binding_id=EXEC_PROSPECTIVE_APM,
        handler=(
            "pipelines.time_dynamics.prospective_apm_exchange_worker:"
            "materialization_request"
        ),
        inputs={"arm": "exchange", "qiskit_compilation": False},
        declared_outputs=("run_receipt",),
        exact_reference_blind=True,
        execution_id=f"{action}::attempt0::mem8g",
        attempt=0,
        planner_sequence=1,
        requested_memory_gib=8,
        arm="exchange",
    )
    request_path = root / "evidence/execution_request.json"
    _write(request_path, request.to_json_dict())
    paths = {
        name: root / f"evidence/{name}.json"
        for name in (
            "work_order", "score_receipt", "planner_receipt",
            "action_registry", "package", "package_manifest",
            "submit_descriptor", "intent_template", "package_preflight",
        )
    }
    for name in ("work_order", "score_receipt", "action_registry", "package"):
        _write(paths[name], {"name": name})
    plan_payload = {"schema": "test_score_derived_apm_plan"}
    _write(paths["planner_receipt"], plan_payload)
    fake_bindings = SimpleNamespace(
        source_epoch_identity_sha256="2" * 64,
        to_json_dict=lambda: {"accuracy_policy": {}},
    )
    fake_source = SimpleNamespace(source_epoch=SimpleNamespace())
    fake_order = SimpleNamespace(
        request=request,
        work_order_sha256="3" * 64,
        bindings=fake_bindings,
        seed=SimpleNamespace(),
        reference=SimpleNamespace(identity_sha256="4" * 64),
    )
    fake_plan = SimpleNamespace(
        next_actions=(SimpleNamespace(),),
        to_json_dict=lambda: plan_payload,
    )
    fake_score = SimpleNamespace(
        to_planner_evidence=lambda _root, _bindings: SimpleNamespace()
    )
    monkeypatch.setattr(
        bridge_module,
        "load_work_order",
        lambda *_args, **_kwargs: (fake_order, fake_source, SimpleNamespace()),
    )
    monkeypatch.setattr(
        bridge_module,
        "ScoreQualificationReceipt",
        SimpleNamespace(load=lambda *_args, **_kwargs: fake_score),
    )
    monkeypatch.setattr(
        bridge_module,
        "AccuracyPolicy",
        SimpleNamespace(from_json_dict=lambda _value: SimpleNamespace()),
    )
    monkeypatch.setattr(
        bridge_module, "plan_prospective_cell", lambda *_args, **_kwargs: fake_plan
    )
    monkeypatch.setattr(
        bridge_module, "resolve_scientific_action", lambda *_args: request
    )
    monkeypatch.setattr(
        bridge_module,
        "load_action_registry_entry",
        lambda *_args, **_kwargs: SimpleNamespace(request=request),
    )
    unit_id = apm_exchange_submission_unit_id(fake_order)
    batch = "paper-ii-apm-test-idempotent"
    manifest_core = {
        "schema": PACKAGE_MANIFEST_SCHEMA_V1,
        "acquisition_unit_id": unit_id,
        "job_batch_name": batch,
        "execution_request_sha256": request.digest,
    }
    _write(
        paths["package_manifest"],
        {**manifest_core, "package_manifest_sha256": _digest(manifest_core)},
    )
    template_core = {
        "schema": SUBMISSION_INTENT_TEMPLATE_V1,
        "acquisition_unit_id": unit_id,
        "lane_id": request.lane_id,
        "job_batch_name": batch,
        "first_reconciliation_outcome": "indeterminate",
        "scheduler_submission_authorized": False,
        "production_submission_enabled": False,
    }
    _write(
        paths["intent_template"],
        {
            **template_core,
            "submission_intent_template_sha256": _digest(template_core),
        },
    )
    paths["submit_descriptor"].write_text(
        "\n".join(
            (
                f'+AcquisitionUnitId = "{unit_id}"',
                f'+AcquisitionLaneId = "{request.lane_id}"',
                f'+PaperIIActionId = "{request.action_id}"',
                f'+PaperIIExecutionRequestSha256 = "{request.digest}"',
                f'+JobBatchName = "{batch}"',
                "queue 1",
                "",
            )
        )
    )
    preflight_core = {
        "schema": PACKAGE_PREFLIGHT_SCHEMA_V1,
        "action_id": request.action_id,
        "execution_request_sha256": request.digest,
        "work_order_sha256": fake_order.work_order_sha256,
        "acquisition_unit_id": unit_id,
        "job_batch_name": batch,
        "scheduler_contacted": False,
        "scheduler_submission_authorized": False,
        "production_submission_enabled": False,
        "qiskit_compilation_requested": False,
        "package": _binding(root, paths["package"]),
        "package_manifest": _binding(root, paths["package_manifest"]),
        "submit_descriptor": _binding(root, paths["submit_descriptor"]),
        "intent_template": _binding(root, paths["intent_template"]),
    }
    _write(
        paths["package_preflight"],
        {**preflight_core, "package_preflight_sha256": _digest(preflight_core)},
    )
    return {"request": request_path, **paths}


def _seal_campaign(root: Path, paths: dict[str, Path | str]) -> dict:
    return seal_apm_exchange_intent(
        repository_root=root,
        request_path=paths["request"],
        work_order_path=paths["work_order"],
        score_receipt_path=paths["score_receipt"],
        planner_receipt_path=paths["planner_receipt"],
        action_registry_path=paths["action_registry"],
        package_preflight_path=paths["package_preflight"],
        remote_root="Holstein_test/apm",
    )


def test_seal_twice_without_timestamp_reuses_durable_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _sealable_campaign(tmp_path, monkeypatch)
    first = _seal_campaign(tmp_path, paths)
    intent = tmp_path / str(first["intent_path"])
    first_bytes = intent.read_bytes()
    time.sleep(0.002)
    second = _seal_campaign(tmp_path, paths)
    assert second == first
    assert intent.read_bytes() == first_bytes


def test_intent_present_bridge_missing_recovers_from_durable_timestamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _sealable_campaign(tmp_path, monkeypatch)
    first = _seal_campaign(tmp_path, paths)
    intent = tmp_path / str(first["intent_path"])
    first_bytes = intent.read_bytes()
    bridge = tmp_path / str(first["bridge_path"])
    bridge.unlink()
    time.sleep(0.002)
    recovered = _seal_campaign(tmp_path, paths)
    assert bridge.is_file()
    assert intent.read_bytes() == first_bytes
    assert recovered["generic_reconciliation_intent"]["recorded_at"] == (
        first["generic_reconciliation_intent"]["recorded_at"]
    )
