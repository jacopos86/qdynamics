from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipelines.time_dynamics.acquisition_supervisor import (
    SCOPE_AUTHORITY_SCHEMA_V1,
    SUBMISSION_INTENT_SCHEMA_V1,
    SUBMISSION_RECEIPT_SCHEMA_V1,
    SchedulerScope,
)
from pipelines.time_dynamics.prospective_submission_recovery import (
    BOOTSTRAP_SCOPE_MANIFEST_DIGEST_KEY,
    COMBINED_SCOPED_SNAPSHOT_V1,
    EXPECTED_HH_UNIT_COUNT,
    HH_LANE_ID,
    PROVISIONAL_SUBMISSION_RESULT_V1,
    SubmissionRecoveryError,
    build_bootstrap_scope_manifest,
    recover_submission_receipt,
)


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _seal(value: dict, key: str) -> dict:
    return {**value, key: _canonical_sha(value)}


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _snapshot(
    *,
    captured_at: str,
    cluster_ids: list[int],
    batches: list[str],
    queue_rows: list[dict],
    history_rows: list[dict],
    reconciliation: dict | None = None,
    intent_bindings: dict[str, dict] | None = None,
) -> dict:
    condition_parts = [f"ClusterId == {cluster}" for cluster in cluster_ids]
    condition_parts.extend(f'JobBatchName == "{batch}"' for batch in batches)
    reconciled = {
        batch: dict(outcome)
        for batch, outcome in (reconciliation or {}).items()
    }
    normalized_intents: dict[str, dict] = {}
    prior_outcomes: dict[str, dict] = {}
    prior_sha = hashlib.sha256(b"prior-intent-first-snapshot").hexdigest()
    for batch, binding in (intent_bindings or {}).items():
        identity = _canonical_sha(binding)
        normalized_intents[batch] = {
            **binding,
            "intent_identity_sha256": identity,
        }
        prior_outcomes[batch] = {
            "status": "indeterminate",
            "intent_identity_sha256": identity,
        }
        outcome = reconciled[batch]
        outcome["intent_identity_sha256"] = identity
        outcome["absence_proof"] = {
            "schema": "paper_ii_intent_first_absence_proof_v1",
            "intent_identity_sha256": identity,
            "prior_snapshot_sha256": prior_sha,
            "prior_snapshot_status": "indeterminate",
            "prior_captured_at": "2026-08-29T00:05:00-05:00",
            "prior_captured_at_unix_ns": 1,
        }
    core = {
        "schema": COMBINED_SCOPED_SNAPSHOT_V1,
        "captured_at": captured_at,
        "captured_at_unix_ns": 2,
        "polling_performed": False,
        "queries": {
            "queue": {
                "argv": ["condor_q", "-constraint", " || ".join(condition_parts), "-json"],
                "returncode": 0,
                "rows": queue_rows,
                "stderr": "",
                "stdout_sha256": "0" * 64,
            },
            "history": {
                "argv": [
                    "condor_history",
                    "-constraint",
                    " || ".join(condition_parts),
                    "-json",
                ],
                "returncode": 0,
                "rows": history_rows,
                "stderr": "",
                "stdout_sha256": "0" * 64,
            },
        },
        "reconciliation": reconciled,
        **(
            {
                "reconciliation_context": {
                    "schema": "paper_ii_intent_first_reconciliation_context_v1",
                    "intents": normalized_intents,
                    "prior_snapshot": {
                        "snapshot_sha256": prior_sha,
                        "captured_at": "2026-08-29T00:05:00-05:00",
                        "captured_at_unix_ns": 1,
                        "intent_outcomes": prior_outcomes,
                    },
                }
            }
            if intent_bindings
            else {}
        ),
        "scope": {
            "condition": " || ".join(condition_parts),
            "prospective_cluster_ids": cluster_ids,
            "prospective_job_batch_names": batches,
        },
        "single_combined_snapshot": True,
    }
    return _seal(core, "snapshot_sha256")


def _campaign(root: Path, *, count: int = EXPECTED_HH_UNIT_COUNT) -> dict:
    source = (
        root
        / "output"
        / HH_LANE_ID
        / "bootstrap/activation_f99b702f/shared/source_f99b702f_runtime.tar.gz"
    )
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"exact revision-derived compact source")
    source_binding = {
        "path": _relative(root, source),
        "sha256": _file_sha(source),
        "bytes": source.stat().st_size,
    }

    intents: list[Path] = []
    results: list[Path] = []
    units: list[dict] = []
    seed_slugs = (
        "nph1__crossover",
        "nph1__mott_dominated",
        "nph1__phonon_dominated",
        "nph3__crossover",
        "nph3__mott_dominated",
        "nph3__phonon_dominated",
    )
    for index in range(count):
        unit_id = f"hh-seed-{index + 1:016x}"
        slug = seed_slugs[index]
        action_id = (
            f"build_seed::{HH_LANE_ID}::seed::{slug}::{index + 101:016x}"
        )
        request_sha = hashlib.sha256(f"request-{index}".encode()).hexdigest()
        batch = f"paper-ii-seed-{HH_LANE_ID}-{index + 1:016x}"
        action_root = (
            root
            / "output"
            / HH_LANE_ID
            / "bootstrap"
            / "activation_f99b702f"
            / "actions"
            / slug
        )
        action_root.mkdir(parents=True, exist_ok=True)
        package = action_root / "seed_action_package.tar.gz"
        package.write_bytes(f"package-{index}".encode())
        descriptor = action_root / "submit.sub"
        descriptor.write_text("queue 1\n")
        package_binding = {
            "path": _relative(root, package),
            "sha256": _file_sha(package),
            "bytes": package.stat().st_size,
        }
        descriptor_binding = {
            "path": _relative(root, descriptor),
            "sha256": _file_sha(descriptor),
        }
        unit = {
            "unit_id": unit_id,
            "action_ids": [action_id],
            "payload": {
                "execution_request_sha256": request_sha,
                "expected_process_count": 1,
                "job_batch_name": batch,
                "package": package_binding,
                "runtime_source_archive": source_binding,
                "submission_scope": {
                    "action_ids": [action_id],
                    "process_ids": [0],
                },
                "submit_descriptor": descriptor_binding["path"],
            },
        }
        intent_core = {
            "schema": SUBMISSION_INTENT_SCHEMA_V1,
            "lane_id": HH_LANE_ID,
            "unit": unit,
        }
        intent = {
            **intent_core,
            "intent_sha256": _canonical_sha(intent_core),
            "recorded_at": f"2026-08-29T00:00:{index:02d}-05:00",
        }
        intent_path = (
            root
            / "output"
            / HH_LANE_ID
            / "acquisition_supervisor"
            / "submission_intents"
            / f"{unit_id}.json"
        )
        _write(intent_path, intent)
        intents.append(intent_path)
        units.append(
            {
                "unit_id": unit_id,
                "action_id": action_id,
                "request_sha": request_sha,
                "batch": batch,
                "cluster_id": 9000 + index,
                "package": package_binding,
                "descriptor": descriptor_binding,
                "intent": intent,
                "intent_path": intent_path,
            }
        )

    batches = [unit["batch"] for unit in units]
    pre_snapshot = _snapshot(
        captured_at="2026-08-29T00:10:00-05:00",
        cluster_ids=[7777],
        batches=batches,
        queue_rows=[],
        history_rows=[],
        reconciliation={
            batch: {"status": "verified_absent", "match_count": 0, "matches": []}
            for batch in batches
        },
        intent_bindings={
            unit["batch"]: {
                "job_batch_name": unit["batch"],
                "acquisition_unit_id": unit["unit_id"],
                "paper_ii_action_id": unit["action_id"],
                "paper_ii_execution_request_sha256": unit["request_sha"],
                "intent_sha256": unit["intent"]["intent_sha256"],
                "recorded_at": unit["intent"]["recorded_at"],
            }
            for unit in units
        },
    )
    pre_path = (
        root
        / "output"
        / HH_LANE_ID
        / "acquisition_supervisor"
        / "scheduler_snapshots"
        / "pre_submit_f99b702f"
        / "combined_snapshot.json"
    )
    _write(pre_path, pre_snapshot)

    for index, unit in enumerate(units):
        result_core = {
            "schema": PROVISIONAL_SUBMISSION_RESULT_V1,
            "status": "submitted",
            "lane_id": HH_LANE_ID,
            "unit_id": unit["unit_id"],
            "action_ids": [unit["action_id"]],
            "cluster_id": unit["cluster_id"],
            "expected_process_ids": [0],
            "job_batch_name": unit["batch"],
            "intent_path": _relative(root, unit["intent_path"]),
            "intent_sha256": unit["intent"]["intent_sha256"],
            "package": unit["package"],
            "runtime_source_archive": source_binding,
            "runtime_image": {
                "path": "chtc/time_dynamics_optuna/image.sif",
                "sha256": "f" * 64,
                "bytes": 216371200,
            },
            "submit_descriptor": unit["descriptor"],
            "lifecycle_validator_status": "pass",
            "dry_run_status": "pass",
            "dry_run_ad_sha256": "d" * 64,
            "condor_submit_returncode": 0,
            "condor_submit_stdout": (
                "Submitting job(s).\n"
                f"1 job(s) submitted to cluster {unit['cluster_id']}.\n"
            ),
            "condor_submit_stderr": "",
            "acceptance_evidence": "condor_submit_zero_exit_and_cluster_id_stdout",
            "scheduler_acceptance_query_performed": False,
            "pre_submit_snapshot_path": _relative(root, pre_path),
            "pre_submit_snapshot_sha256": pre_snapshot["snapshot_sha256"],
            "remote_raw_evidence_policy": "preserve",
            "submitted_at": f"2026-08-29T00:20:{index:02d}-05:00",
        }
        result = _seal(result_core, "submission_result_sha256")
        result_path = (
            root
            / "output"
            / HH_LANE_ID
            / "acquisition_supervisor"
            / "submission_results"
            / "activation_f99b702f"
            / f"{unit['unit_id']}.json"
        )
        _write(result_path, result)
        results.append(result_path)
        unit["result_path"] = result_path

    later_rows = [
        {
            "ClusterId": unit["cluster_id"],
            "ProcId": 0,
            "JobStatus": 1,
            "JobBatchName": unit["batch"],
            "AcquisitionUnitId": unit["unit_id"],
            "AcquisitionLaneId": HH_LANE_ID,
            "PaperIIActionId": unit["action_id"],
            "PaperIIExecutionRequestSha256": unit["request_sha"],
        }
        for unit in units
    ]
    later_snapshot = _snapshot(
        captured_at="2026-08-29T01:00:00-05:00",
        cluster_ids=[unit["cluster_id"] for unit in units],
        batches=batches,
        queue_rows=later_rows,
        history_rows=[],
    )
    later_path = (
        root
        / "output"
        / HH_LANE_ID
        / "acquisition_supervisor"
        / "scheduler_snapshots"
        / "post_submit_f99b702f"
        / "combined_snapshot.json"
    )
    _write(later_path, later_snapshot)
    return {
        "intents": intents,
        "results": results,
        "units": units,
        "pre_path": pre_path,
        "later_path": later_path,
        "later_snapshot": later_snapshot,
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


def test_recovery_emits_only_canonical_found_receipt_and_is_idempotent(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path)
    unit = campaign["units"][0]

    recovered = recover_submission_receipt(
        repository_root=tmp_path,
        intent_path=unit["intent_path"],
        provisional_result_path=unit["result_path"],
        later_snapshot_path=campaign["later_path"],
    )
    recovered_again = recover_submission_receipt(
        repository_root=tmp_path,
        intent_path=unit["intent_path"],
        provisional_result_path=unit["result_path"],
        later_snapshot_path=campaign["later_path"],
    )

    assert recovered == recovered_again
    receipt = json.loads((tmp_path / recovered.path).read_text())
    assert receipt["schema"] == SUBMISSION_RECEIPT_SCHEMA_V1
    assert receipt["lane_id"] == HH_LANE_ID
    assert receipt["unit_id"] == unit["unit_id"]
    assert receipt["action_ids"] == [unit["action_id"]]
    assert receipt["intent_sha256"] == unit["intent"]["intent_sha256"]
    assert receipt["recovered_from_scheduler"] is True
    assert receipt["remote"]["cluster_id"] == unit["cluster_id"]
    assert receipt["remote"]["process_ids"] == [0]
    assert receipt["remote"]["scheduler_identity"] == f"{unit['cluster_id']}.0"
    assert receipt["remote"]["recovery_evidence"]["scheduler_confirmation"] is True
    assert receipt["recorded_at"] == "2026-08-29T01:00:00-05:00"
    assert recovered.path == (
        f"output/{HH_LANE_ID}/acquisition_supervisor/submission_receipts/"
        f"{unit['unit_id']}.json"
    )


def test_recovery_rejects_first_cycle_indeterminate_absence(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    unit = campaign["units"][0]

    def make_first_cycle(snapshot: dict) -> None:
        snapshot["reconciliation_context"]["prior_snapshot"] = None
        for outcome in snapshot["reconciliation"].values():
            outcome["status"] = "indeterminate"
            outcome.pop("absence_proof", None)
            outcome["indeterminate_reason"] = (
                "first_snapshot_for_exact_intent_identity"
            )

    _reseal_snapshot(campaign["pre_path"], make_first_cycle)
    new_snapshot_sha = json.loads(campaign["pre_path"].read_text())[
        "snapshot_sha256"
    ]
    _reseal_result(
        unit["result_path"],
        lambda value: value.__setitem__(
            "pre_submit_snapshot_sha256", new_snapshot_sha
        ),
    )

    with pytest.raises(
        SubmissionRecoveryError, match="does not prove verified-absent identity"
    ):
        recover_submission_receipt(
            repository_root=tmp_path,
            intent_path=unit["intent_path"],
            provisional_result_path=unit["result_path"],
            later_snapshot_path=campaign["later_path"],
        )


def test_recovery_rejects_intent_and_provisional_digest_tampering(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path)
    unit = campaign["units"][0]
    intent = json.loads(unit["intent_path"].read_text())
    intent["unit"]["payload"]["expected_process_count"] = 2
    _write(unit["intent_path"], intent)

    with pytest.raises(SubmissionRecoveryError, match="intent self digest"):
        recover_submission_receipt(
            repository_root=tmp_path,
            intent_path=unit["intent_path"],
            provisional_result_path=unit["result_path"],
            later_snapshot_path=campaign["later_path"],
        )

    campaign = _campaign(tmp_path / "other")
    unit = campaign["units"][0]
    result = json.loads(unit["result_path"].read_text())
    result["cluster_id"] += 1
    _write(unit["result_path"], result)
    with pytest.raises(SubmissionRecoveryError, match="result self digest"):
        recover_submission_receipt(
            repository_root=tmp_path / "other",
            intent_path=unit["intent_path"],
            provisional_result_path=unit["result_path"],
            later_snapshot_path=campaign["later_path"],
        )


def test_recovery_rejects_unsealed_or_failed_combined_query(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    unit = campaign["units"][0]
    snapshot = json.loads(campaign["later_path"].read_text())
    snapshot["queries"]["queue"]["returncode"] = 1
    _write(campaign["later_path"], snapshot)
    with pytest.raises(SubmissionRecoveryError, match="snapshot self digest"):
        recover_submission_receipt(
            repository_root=tmp_path,
            intent_path=unit["intent_path"],
            provisional_result_path=unit["result_path"],
            later_snapshot_path=campaign["later_path"],
        )

    campaign = _campaign(tmp_path / "failed")
    unit = campaign["units"][0]
    _reseal_snapshot(
        campaign["later_path"],
        lambda value: value["queries"]["queue"].__setitem__("returncode", 1),
    )
    with pytest.raises(SubmissionRecoveryError, match="queue query did not succeed"):
        recover_submission_receipt(
            repository_root=tmp_path / "failed",
            intent_path=unit["intent_path"],
            provisional_result_path=unit["result_path"],
            later_snapshot_path=campaign["later_path"],
        )


def test_recovery_requires_exactly_one_matching_cluster_process(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path)
    unit = campaign["units"][0]
    _reseal_snapshot(
        campaign["later_path"],
        lambda value: value["queries"]["queue"]["rows"].pop(0),
    )
    with pytest.raises(SubmissionRecoveryError, match="exactly one matching"):
        recover_submission_receipt(
            repository_root=tmp_path,
            intent_path=unit["intent_path"],
            provisional_result_path=unit["result_path"],
            later_snapshot_path=campaign["later_path"],
        )

    campaign = _campaign(tmp_path / "duplicate")
    unit = campaign["units"][0]

    def duplicate(value: dict) -> None:
        value["queries"]["history"]["rows"].append(
            dict(value["queries"]["queue"]["rows"][0])
        )

    _reseal_snapshot(campaign["later_path"], duplicate)
    with pytest.raises(SubmissionRecoveryError, match="repeats scheduler identity"):
        recover_submission_receipt(
            repository_root=tmp_path / "duplicate",
            intent_path=unit["intent_path"],
            provisional_result_path=unit["result_path"],
            later_snapshot_path=campaign["later_path"],
        )


def test_recovery_rejects_wrong_scheduler_action_or_request(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    unit = campaign["units"][0]

    def wrong_action(value: dict) -> None:
        value["queries"]["queue"]["rows"][0]["PaperIIActionId"] = "wrong"

    _reseal_snapshot(campaign["later_path"], wrong_action)
    with pytest.raises(SubmissionRecoveryError, match="exact unit/action/request"):
        recover_submission_receipt(
            repository_root=tmp_path,
            intent_path=unit["intent_path"],
            provisional_result_path=unit["result_path"],
            later_snapshot_path=campaign["later_path"],
        )


def test_bootstrap_scope_manifest_is_six_unit_unconfirmed_authority(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path)
    output = (
        tmp_path
        / "output/time_dynamics_acquisition/bootstrap_hh_seed_scopes_f99b702f.json"
    )

    manifest = build_bootstrap_scope_manifest(
        repository_root=tmp_path,
        registry_id="paper-ii-parallel-benchmark-acquisition-v1",
        provisional_result_paths=campaign["results"],
        output_path=output,
    )
    same = build_bootstrap_scope_manifest(
        repository_root=tmp_path,
        registry_id="paper-ii-parallel-benchmark-acquisition-v1",
        provisional_result_paths=list(reversed(campaign["results"])),
        output_path=output,
    )

    assert manifest == same
    payload = json.loads(output.read_text())
    declared = payload.pop(BOOTSTRAP_SCOPE_MANIFEST_DIGEST_KEY)
    assert declared == _canonical_sha(payload)
    assert payload["schema"] == SCOPE_AUTHORITY_SCHEMA_V1
    assert payload["status"] == "provisional_submission_scope_only"
    assert payload["unit_count"] == EXPECTED_HH_UNIT_COUNT
    assert payload["scheduler_confirmation"] is False
    assert payload["scheduler_confirmation_snapshot"] is None
    assert payload["scope_purpose"] == "monitoring_and_recovery_only"
    assert payload["production_submission_enabled"] is False
    assert payload["submission_authorized"] is False
    assert all(unit["scheduler_confirmation"] is False for unit in payload["units"])
    scopes = tuple(SchedulerScope.from_mapping(row) for row in payload["scopes"])
    assert scopes == manifest.scopes
    assert {scope.cluster_id for scope in scopes} == set(range(9000, 9006))


def test_bootstrap_scope_manifest_rejects_partial_or_repeated_scope(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path)
    with pytest.raises(SubmissionRecoveryError, match="exactly 6 units"):
        build_bootstrap_scope_manifest(
            repository_root=tmp_path,
            registry_id="registry-v1",
            provisional_result_paths=campaign["results"][:-1],
            output_path="output/time_dynamics_acquisition/scopes.json",
        )

    second = campaign["results"][1]
    first_cluster = campaign["units"][0]["cluster_id"]

    def repeat_cluster(value: dict) -> None:
        value["cluster_id"] = first_cluster
        value["condor_submit_stdout"] = (
            "Submitting job(s).\n"
            f"1 job(s) submitted to cluster {first_cluster}.\n"
        )

    _reseal_result(second, repeat_cluster)
    with pytest.raises(SubmissionRecoveryError, match="repeats a unit/action/cluster"):
        build_bootstrap_scope_manifest(
            repository_root=tmp_path,
            registry_id="registry-v1",
            provisional_result_paths=campaign["results"],
            output_path="output/time_dynamics_acquisition/scopes.json",
        )


def test_provisional_acceptance_query_claim_is_rejected(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    unit = campaign["units"][0]
    _reseal_result(
        unit["result_path"],
        lambda value: value.__setitem__("scheduler_acceptance_query_performed", True),
    )
    with pytest.raises(SubmissionRecoveryError, match="contract diverged"):
        recover_submission_receipt(
            repository_root=tmp_path,
            intent_path=unit["intent_path"],
            provisional_result_path=unit["result_path"],
            later_snapshot_path=campaign["later_path"],
        )
