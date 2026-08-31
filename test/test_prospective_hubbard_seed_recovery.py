"""Exact later-snapshot recovery contracts for the Hubbard-L4 seed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

import pipelines.time_dynamics.prospective_hubbard_seed_recovery as recovery
from pipelines.time_dynamics.acquisition_supervisor import (
    SUBMISSION_INTENT_SCHEMA_V1,
    SUBMISSION_RECEIPT_SCHEMA_V1,
)
from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    BootstrapEvidence,
    LANE_HUBBARD,
    SourceEpochIdentity,
    build_identity_registry,
    plan_bootstrap_actions,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    SOURCE_EPOCH_RECEIPT_V1,
    SOURCE_TREE_MANIFEST_V1,
    SourceEpochReceipt,
    executor_bindings,
    resolve_bootstrap_action,
)
from pipelines.time_dynamics.prospective_hubbard_seed_recovery import (
    BOOTSTRAP_PACKAGE_MANIFEST_V1,
    COMBINED_SCOPED_SNAPSHOT_V1,
    HUBBARD_LANE_ID,
    load_hubbard_durable_submission_intent,
    recover_hubbard_submission_receipt,
)
from pipelines.time_dynamics.prospective_submission_recovery import (
    PROVISIONAL_SUBMISSION_RESULT_V1,
    SubmissionRecoveryError,
)


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _seal(value: dict, key: str) -> dict:
    result = dict(value)
    result[key] = _canonical_sha(result)
    return result


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _source_receipt(root: Path) -> SourceEpochReceipt:
    epoch = SourceEpochIdentity(
        lane_id=LANE_HUBBARD,
        source_epoch_id="paper-ii-hubbard-recovery-test",
        source_revision="a" * 40,
        source_archive_sha256="c" * 64,
    )
    manifest = root / "source/source_tree_manifest.json"
    _write(
        manifest,
        _seal(
            {
                "schema": SOURCE_TREE_MANIFEST_V1,
                "lane_id": epoch.lane_id,
                "source_epoch_identity_sha256": epoch.identity_sha256,
                "source_revision": epoch.source_revision,
                "source_archive_sha256": epoch.source_archive_sha256,
                "source_tree_sha256": "d" * 64,
                "provenance_kind": "verified_source_archive",
            },
            "source_tree_manifest_sha256",
        ),
    )
    receipt = root / "source/source_epoch_receipt.json"
    _write(
        receipt,
        _seal(
            {
                "schema": SOURCE_EPOCH_RECEIPT_V1,
                "source_epoch": epoch.to_json_dict(),
                "source_tree_manifest": {
                    "path": _relative(root, manifest),
                    "sha256": _file_sha(manifest),
                },
                "source_tree_sha256": "d" * 64,
            },
            "source_epoch_receipt_sha256",
        ),
    )
    return SourceEpochReceipt.load(
        receipt, repository_root=root, expected_epoch=epoch
    )


def _request(source: SourceEpochReceipt):
    plan = plan_bootstrap_actions(
        build_identity_registry(),
        evidence=BootstrapEvidence(
            source_epochs={LANE_HUBBARD: source.source_epoch}
        ),
        available_executors=tuple(row.binding_id for row in executor_bindings()),
    )
    action = next(
        row
        for row in plan.ready_actions
        if row.lane_id == LANE_HUBBARD and row.kind == "build_seed"
    )
    return resolve_bootstrap_action(action)


def _snapshot(
    *,
    captured_at: str,
    cluster_id: int,
    batch: str,
    queue_rows: list[dict],
    history_rows: list[dict],
    reconciliation: dict | None = None,
    intent_binding: dict | None = None,
) -> dict:
    condition = f'ClusterId == {cluster_id} || JobBatchName == "{batch}"'
    reconciled = {
        key: dict(value) for key, value in (reconciliation or {}).items()
    }
    context: dict | None = None
    if intent_binding is not None:
        identity = _canonical_sha(intent_binding)
        prior_sha = hashlib.sha256(b"prior-hubbard-intent-snapshot").hexdigest()
        reconciled[batch]["intent_identity_sha256"] = identity
        reconciled[batch]["absence_proof"] = {
            "schema": "paper_ii_intent_first_absence_proof_v1",
            "intent_identity_sha256": identity,
            "prior_snapshot_sha256": prior_sha,
            "prior_snapshot_status": "indeterminate",
            "prior_captured_at": "2026-08-29T01:03:00-05:00",
            "prior_captured_at_unix_ns": 1,
        }
        context = {
            "schema": "paper_ii_intent_first_reconciliation_context_v1",
            "intents": {
                batch: {
                    **intent_binding,
                    "intent_identity_sha256": identity,
                }
            },
            "prior_snapshot": {
                "snapshot_sha256": prior_sha,
                "captured_at": "2026-08-29T01:03:00-05:00",
                "captured_at_unix_ns": 1,
                "intent_outcomes": {
                    batch: {
                        "status": "indeterminate",
                        "intent_identity_sha256": identity,
                    }
                },
            },
        }
    core = {
        "schema": COMBINED_SCOPED_SNAPSHOT_V1,
        "captured_at": captured_at,
        "captured_at_unix_ns": 2,
        "single_combined_snapshot": True,
        "polling_performed": False,
        "queries": {
            "queue": {
                "argv": ["condor_q", "-constraint", condition, "-json"],
                "returncode": 0,
                "rows": queue_rows,
                "stderr": "",
                "stdout_sha256": "0" * 64,
            },
            "history": {
                "argv": ["condor_history", "-constraint", condition, "-json"],
                "returncode": 0,
                "rows": history_rows,
                "stderr": "",
                "stdout_sha256": "0" * 64,
            },
        },
        "reconciliation": reconciled,
        **({"reconciliation_context": context} if context is not None else {}),
        "scope": {
            "condition": condition,
            "prospective_cluster_ids": [cluster_id],
            "prospective_job_batch_names": [batch],
        },
    }
    return _seal(core, "snapshot_sha256")


def _campaign(root: Path) -> dict:
    source = _source_receipt(root)
    request = _request(source)
    token = "l4__u4__half_filled"
    action_root = (
        root
        / "output"
        / LANE_HUBBARD
        / "bootstrap/activation_test/actions"
        / token
    )
    package = action_root / "seed_action_package.tar.gz"
    descriptor = action_root / "submit.sub"
    runtime_source = (
        root
        / "output"
        / LANE_HUBBARD
        / "bootstrap/activation_test/shared/source_runtime.tar.gz"
    )
    package.parent.mkdir(parents=True, exist_ok=True)
    runtime_source.parent.mkdir(parents=True, exist_ok=True)
    package.write_bytes(b"hubbard seed action package")
    runtime_source.write_bytes(b"exact source archive")
    descriptor.write_text("queue 1\n", encoding="utf-8")
    package_binding = {
        "path": _relative(root, package),
        "sha256": _file_sha(package),
        "bytes": package.stat().st_size,
    }
    source_binding = {
        "path": _relative(root, runtime_source),
        "sha256": _file_sha(runtime_source),
        "bytes": runtime_source.stat().st_size,
    }
    batch = f"paper-ii-seed-{LANE_HUBBARD.replace('_', '-')}-{request.digest[:16]}"
    unit_id = f"hubbard-seed-{request.digest[:16]}"
    package_manifest_path = action_root / "package_manifest.json"
    package_manifest = _seal(
        {
            "schema": BOOTSTRAP_PACKAGE_MANIFEST_V1,
            "status": "prepared_local_only",
            "lane_id": LANE_HUBBARD,
            "action_id": request.action_id,
            "execution_id": request.execution_id,
            "execution_request_sha256": request.digest,
            "execution_request": request.to_json_dict(),
            "job_batch_name": batch,
            "package": {
                "path": package_binding["path"],
                "sha256": package_binding["sha256"],
            },
            "package_bytes": package_binding["bytes"],
            "source_epoch_identity_sha256": source.source_epoch.identity_sha256,
            "source_epoch_receipt": {
                "path": source.receipt_path,
                "sha256": source.receipt_sha256,
            },
            "qiskit_compilation": False,
            "production_submission_enabled": False,
        },
        "bootstrap_package_manifest_sha256",
    )
    _write(package_manifest_path, package_manifest)
    unit = {
        "unit_id": unit_id,
        "action_ids": [request.action_id],
        "payload": {
            "execution_request_sha256": request.digest,
            "expected_process_count": 1,
            "job_batch_name": batch,
            "package": package_binding,
            "package_manifest": {
                "path": _relative(root, package_manifest_path),
                "sha256": _file_sha(package_manifest_path),
            },
            "remote_raw_evidence_policy": "preserve",
            "runtime_source_archive": source_binding,
            "source_epoch_identity_sha256": source.source_epoch.identity_sha256,
            "submission_scope": {
                "action_ids": [request.action_id],
                "process_ids": [0],
            },
            "submit_descriptor": _relative(root, descriptor),
        },
    }
    intent_core = {
        "schema": SUBMISSION_INTENT_SCHEMA_V1,
        "lane_id": LANE_HUBBARD,
        "unit": unit,
    }
    intent = {
        **intent_core,
        "intent_sha256": _canonical_sha(intent_core),
        "recorded_at": "2026-08-29T01:00:00-05:00",
    }
    intent_path = (
        root
        / "output"
        / LANE_HUBBARD
        / "acquisition_supervisor/submission_intents"
        / f"{unit_id}.json"
    )
    _write(intent_path, intent)
    cluster_id = 9714040
    pre = _snapshot(
        captured_at="2026-08-29T01:04:00-05:00",
        cluster_id=cluster_id,
        batch=batch,
        queue_rows=[],
        history_rows=[],
        reconciliation={
            batch: {"status": "verified_absent", "match_count": 0, "matches": []}
        },
        intent_binding={
            "job_batch_name": batch,
            "acquisition_unit_id": unit_id,
            "paper_ii_action_id": request.action_id,
            "paper_ii_execution_request_sha256": request.digest,
            "intent_sha256": intent["intent_sha256"],
            "recorded_at": intent["recorded_at"],
        },
    )
    pre_path = root / "output/time_dynamics_acquisition/pre/combined_snapshot.json"
    _write(pre_path, pre)
    descriptor_binding = {
        "path": _relative(root, descriptor),
        "sha256": _file_sha(descriptor),
    }
    result_core = {
        "schema": PROVISIONAL_SUBMISSION_RESULT_V1,
        "status": "submitted",
        "lane_id": LANE_HUBBARD,
        "unit_id": unit_id,
        "action_ids": [request.action_id],
        "cluster_id": cluster_id,
        "expected_process_ids": [0],
        "job_batch_name": batch,
        "intent_path": _relative(root, intent_path),
        "intent_sha256": intent["intent_sha256"],
        "package": package_binding,
        "runtime_source_archive": source_binding,
        "runtime_image": {
            "path": "chtc/time_dynamics_optuna/image.sif",
            "sha256": "f" * 64,
            "bytes": 216371200,
        },
        "submit_descriptor": descriptor_binding,
        "lifecycle_validator_status": "pass",
        "dry_run_status": "pass",
        "dry_run_ad_sha256": "e" * 64,
        "condor_submit_returncode": 0,
        "condor_submit_stdout": (
            f"Submitting job(s).\n1 job(s) submitted to cluster {cluster_id}.\n"
        ),
        "condor_submit_stderr": "",
        "acceptance_evidence": "condor_submit_zero_exit_and_cluster_id_stdout",
        "scheduler_acceptance_query_performed": False,
        "pre_submit_snapshot_path": _relative(root, pre_path),
        "pre_submit_snapshot_sha256": pre["snapshot_sha256"],
        "remote_raw_evidence_policy": "preserve",
        "submitted_at": "2026-08-29T01:07:00-05:00",
    }
    result = _seal(result_core, "submission_result_sha256")
    result_path = (
        root
        / "output"
        / LANE_HUBBARD
        / "acquisition_supervisor/submission_results/activation_test"
        / f"{unit_id}.json"
    )
    _write(result_path, result)
    row = {
        "ClusterId": cluster_id,
        "ProcId": 0,
        "JobStatus": 1,
        "JobBatchName": batch,
        "AcquisitionUnitId": unit_id,
        "AcquisitionLaneId": LANE_HUBBARD,
        "PaperIIActionId": request.action_id,
        "PaperIIExecutionRequestSha256": request.digest,
        "PaperIISourceEpochId": source.source_epoch.source_epoch_id,
    }
    later = _snapshot(
        captured_at="2026-08-29T01:20:00-05:00",
        cluster_id=cluster_id,
        batch=batch,
        queue_rows=[row],
        history_rows=[],
    )
    later_path = root / "output/time_dynamics_acquisition/later/combined_snapshot.json"
    _write(later_path, later)
    return {
        "source": source,
        "request": request,
        "intent_path": intent_path,
        "result_path": result_path,
        "later_path": later_path,
        "later": later,
        "row": row,
        "cluster_id": cluster_id,
        "unit_id": unit_id,
        "batch": batch,
        "package_manifest_path": package_manifest_path,
    }


def _reseal_snapshot(path: Path, mutate) -> None:
    value = json.loads(path.read_text())
    value.pop("snapshot_sha256")
    mutate(value)
    _write(path, _seal(value, "snapshot_sha256"))


def _reseal_result(path: Path, mutate) -> None:
    value = json.loads(path.read_text())
    value.pop("submission_result_sha256")
    mutate(value)
    _write(path, _seal(value, "submission_result_sha256"))


def test_recovers_exact_hubbard_receipt_and_is_idempotent(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    loaded = load_hubbard_durable_submission_intent(
        campaign["intent_path"], repository_root=tmp_path
    )
    recovered = recover_hubbard_submission_receipt(
        repository_root=tmp_path,
        intent_path=campaign["intent_path"],
        provisional_result_path=campaign["result_path"],
        later_snapshot_path=campaign["later_path"],
    )
    repeated = recover_hubbard_submission_receipt(
        repository_root=tmp_path,
        intent_path=campaign["intent_path"],
        provisional_result_path=campaign["result_path"],
        later_snapshot_path=campaign["later_path"],
    )
    receipt = json.loads((tmp_path / recovered.path).read_text())

    assert repeated == recovered
    assert loaded.request == campaign["request"]
    assert receipt["schema"] == SUBMISSION_RECEIPT_SCHEMA_V1
    assert receipt["lane_id"] == HUBBARD_LANE_ID
    assert receipt["unit_id"] == campaign["unit_id"]
    assert receipt["remote"]["cluster_id"] == campaign["cluster_id"]
    assert (
        receipt["remote"]["source_epoch_id"]
        == campaign["source"].source_epoch.source_epoch_id
    )
    assert (
        receipt["remote"]["source_epoch_identity_sha256"]
        == campaign["source"].source_epoch.identity_sha256
    )
    assert receipt["remote"]["recovery_evidence"]["scheduler_confirmation"] is True
    assert (
        "legacy_submission_preflight_warning"
        not in receipt["remote"]["recovery_evidence"]
    )


def test_exact_9714040_legacy_fallback_is_byte_bound_and_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = Path(__file__).resolve().parents[1]
    intent_path = (
        "output/paper_ii_hubbard_l4_scale_v1/acquisition_supervisor/"
        "submission_intents/hubbard-seed-ee58df1fb13e9669.json"
    )
    provisional_path = (
        "output/paper_ii_hubbard_l4_scale_v1/acquisition_supervisor/"
        "submission_results/activation_5eedf0dc/"
        "hubbard-seed-ee58df1fb13e9669.json"
    )
    pre_submit_path = (
        "output/time_dynamics_acquisition/prospective_combined_snapshots/"
        "cycle_20260829_0104/combined_snapshot.json"
    )
    initial_path = (
        "output/paper_ii_hubbard_l4_scale_v1/acquisition_supervisor/"
        "submission_reconciliations/hubbard-seed-ee58df1fb13e9669/"
        "000_initial_indeterminate.json"
    )
    intent = load_hubbard_durable_submission_intent(
        intent_path, repository_root=repository
    )
    for relative in (provisional_path, pre_submit_path, initial_path):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / relative, target)

    # Binary package/source bindings are independently checked by the ordinary
    # loader.  This focused historical-fixture test exercises the exact JSON
    # byte bindings without copying ignored multi-megabyte archives into tmp.
    monkeypatch.setattr(
        recovery, "_verify_local_binding", lambda *args, **kwargs: None
    )
    result, warning = recovery._load_exact_legacy_9714040_result(
        provisional_path, root=tmp_path, intent=intent
    )

    assert result.cluster_id == 9714040
    assert result.process_ids == (0,)
    assert warning["schema"] == (
        "paper_ii_hubbard_9714040_legacy_preflight_warning_v1"
    )
    assert warning["status"] == "historical_submission_recovery_only"
    assert warning["strict_intent_first_gate_preserved_for_new_submissions"] is True
    assert warning["resubmission_authorized"] is False
    assert warning["exact_identity"]["execution_request_sha256"] == (
        intent.request.digest
    )

    with (tmp_path / provisional_path).open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(
        SubmissionRecoveryError, match="provisional bytes changed"
    ):
        recovery._load_exact_legacy_9714040_result(
            provisional_path, root=tmp_path, intent=intent
        )


def test_nonhistorical_first_snapshot_absence_remains_blocked(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path)
    result = json.loads(campaign["result_path"].read_text())
    pre_path = tmp_path / result["pre_submit_snapshot_path"]

    _reseal_snapshot(
        pre_path,
        lambda value: value.pop("reconciliation_context"),
    )
    new_snapshot_sha = json.loads(pre_path.read_text())["snapshot_sha256"]
    _reseal_result(
        campaign["result_path"],
        lambda value: value.__setitem__(
            "pre_submit_snapshot_sha256", new_snapshot_sha
        ),
    )

    with pytest.raises(
        SubmissionRecoveryError, match="pre-submit snapshot is not canonical"
    ):
        recover_hubbard_submission_receipt(
            repository_root=tmp_path,
            intent_path=campaign["intent_path"],
            provisional_result_path=campaign["result_path"],
            later_snapshot_path=campaign["later_path"],
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("AcquisitionUnitId", "wrong-unit"),
        ("AcquisitionLaneId", "wrong-lane"),
        ("PaperIIExecutionRequestSha256", "0" * 64),
        ("PaperIISourceEpochId", "wrong-source-epoch"),
    ],
)
def test_recovery_requires_exact_scheduler_bindings(
    tmp_path: Path, field: str, replacement: str
) -> None:
    campaign = _campaign(tmp_path)
    _reseal_snapshot(
        campaign["later_path"],
        lambda value: value["queries"]["queue"]["rows"][0].__setitem__(
            field, replacement
        ),
    )
    with pytest.raises(SubmissionRecoveryError, match="batch/unit/lane/action/request/source"):
        recover_hubbard_submission_receipt(
            repository_root=tmp_path,
            intent_path=campaign["intent_path"],
            provisional_result_path=campaign["result_path"],
            later_snapshot_path=campaign["later_path"],
        )


def test_recovery_rejects_nonlater_snapshot_and_manifest_tampering(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path)
    _reseal_snapshot(
        campaign["later_path"],
        lambda value: value.__setitem__(
            "captured_at", "2026-08-29T01:06:00-05:00"
        ),
    )
    with pytest.raises(SubmissionRecoveryError, match="must be later"):
        recover_hubbard_submission_receipt(
            repository_root=tmp_path,
            intent_path=campaign["intent_path"],
            provisional_result_path=campaign["result_path"],
            later_snapshot_path=campaign["later_path"],
        )

    other = _campaign(tmp_path / "other")
    manifest = json.loads(other["package_manifest_path"].read_text())
    manifest["source_epoch_identity_sha256"] = "0" * 64
    _write(other["package_manifest_path"], manifest)
    with pytest.raises(SubmissionRecoveryError, match="manifest digest changed"):
        load_hubbard_durable_submission_intent(
            other["intent_path"], repository_root=tmp_path / "other"
        )


def test_recovery_rejects_changed_remote_raw_preservation_policy(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path)
    _reseal_result(
        campaign["result_path"],
        lambda value: value.__setitem__("remote_raw_evidence_policy", "delete"),
    )
    with pytest.raises(SubmissionRecoveryError, match="preserve remote raw"):
        recover_hubbard_submission_receipt(
            repository_root=tmp_path,
            intent_path=campaign["intent_path"],
            provisional_result_path=campaign["result_path"],
            later_snapshot_path=campaign["later_path"],
        )
