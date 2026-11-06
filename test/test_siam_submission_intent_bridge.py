from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipelines.time_dynamics.combined_scheduler_snapshot import (
    CAPTURE_SPEC_V1,
    HH_ACQUISITION_LANE_ID,
    HUBBARD_ACQUISITION_LANE_ID,
    build_combined_snapshot,
)
from pipelines.time_dynamics.siam_chtc_package import (
    SIAM_NB4_CHTC_ACTION_SCHEMA_V4,
    SIAM_NB4_CHTC_MANIFEST_SCHEMA_V4,
    build_siam_source_epoch_contract,
)
from pipelines.time_dynamics.siam_submission_intent_bridge import (
    BridgeError,
    build_post_submit_receipt,
    build_bridge_receipt,
    intent_bound_capture_spec,
    later_capture_plan,
    validate_post_submit_receipt,
    verified_absent_reconciliation_receipt,
)


ATTRIBUTES = (
    "ClusterId,ProcId,JobStatus,ExitCode,HoldReason,JobBatchName,"
    "AcquisitionUnitId,AcquisitionLaneId,PaperIIActionId,"
    "PaperIIExecutionRequestSha256,PaperIISourceEpochId,"
    "SIAMSourceEpochId,SIAMScientificActionId,EnteredCurrentStatus,"
    "CompletionDate,QDate"
)
ACTION_ID = "siam-nb4-avqds-screen-r02-cut1em03"
SOURCE_EPOCH_ID = "siam-nb4-source-446b75f3edc235404bf9"
BATCH = "siam-nb4-screen-t0-t1-101-avqds-rung2"
CONDITION = (
    f'(SIAMSourceEpochId == "{SOURCE_EPOCH_ID}" && '
    f'SIAMScientificActionId == "{ACTION_ID}")'
)


def _bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else _bytes(value))


def _binding(root: Path, path: Path, *, include_bytes: bool = True) -> dict:
    row = {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha(path),
    }
    if include_bytes:
        row["bytes"] = path.stat().st_size
    return row


def _package(root: Path, *, batch: str = BATCH) -> Path:
    package_id = "siam_nb4_avqds_screen_r02_exchangefix_source_test"
    package = root / "chtc" / package_id
    action_path = package / "input/action.json"
    seed_path = package / "input/seed.json"
    worker_path = package / "run_action.py"
    wrapper_path = package / "run_action_apptainer.sh"
    submit_path = package / "submit.sub"
    manifest_path = package / "input/manifest.json"
    upload_path = package / "remote_upload_receipt.json"
    intent_path = package / "submission_intent.json"
    revision = "36c40d522a4c5fb4daa57234fbc85e82b692e8a3"
    tree_sha = "2acc890278b388ace3f09ca989ea73f5a286b0c47ad062f563f53beb48946d66"
    source_epoch = build_siam_source_epoch_contract(
        source_revision=revision,
        source_tree_files_sha256=tree_sha,
    )
    assert source_epoch["source_epoch_id"] == SOURCE_EPOCH_ID
    identity = {
        "scope": "source_epoch_id/action_id",
        "source_epoch_id": SOURCE_EPOCH_ID,
        "action_id": ACTION_ID,
    }
    output = f"raw_outputs/{package_id}/{ACTION_ID}/run.json"
    action = {
        "schema": SIAM_NB4_CHTC_ACTION_SCHEMA_V4,
        "package_id": package_id,
        "action_id": ACTION_ID,
        "source_revision": revision,
        "source_tree": {"files_sha256": tree_sha},
        "source_epoch": source_epoch,
        "scientific_identity": identity,
        "intentional_source_rebaseline": None,
        "arm": "avqds",
        "segment": "screen_t0_t1_101",
        "activation_cut_ladder_index": 2,
        "trajectory_json": output,
        "compile_terminal_cost": False,
        "qiskit_compilation_requested": False,
    }
    _write(action_path, action)
    _write(seed_path, b"{}\n")
    _write(worker_path, b"#!/usr/bin/env python3\n")
    _write(wrapper_path, b"#!/usr/bin/env bash\n")
    submit = (
        f"executable = {package.relative_to(root).as_posix()}/run_action_apptainer.sh\n"
        f"arguments = {ACTION_ID}\n"
        "transfer_input_files = pipelines, src, "
        f"{package.relative_to(root).as_posix()}\n"
        f'+SIAMSourceEpochId = "{SOURCE_EPOCH_ID}"\n'
        f'+SIAMScientificActionId = "{ACTION_ID}"\n'
        "+SIAMIntentionalSourceRebaseline = False\n"
        f'+JobBatchName = "{batch}"\n'
        "queue 1\n"
    ).encode()
    _write(submit_path, submit)
    files = {
        "action": _binding(root, action_path),
        "seed": _binding(root, seed_path),
        "worker": _binding(root, worker_path),
        "wrapper": _binding(root, wrapper_path),
        "submit_descriptor": _binding(root, submit_path),
    }
    manifest = {
        "schema": SIAM_NB4_CHTC_MANIFEST_SCHEMA_V4,
        "status": "PREPARED_NOT_SUBMITTED",
        "package_id": package_id,
        "action_id": ACTION_ID,
        "action_json_sha256": files["action"]["sha256"],
        "seed_sha256": files["seed"]["sha256"],
        "worker_sha256": files["worker"]["sha256"],
        "wrapper_sha256": files["wrapper"]["sha256"],
        "submit_descriptor_sha256": files["submit_descriptor"]["sha256"],
        "source_revision": revision,
        "source_epoch": source_epoch,
        "source_tree_files_sha256": tree_sha,
        "scientific_identity": identity,
        "intentional_source_rebaseline": None,
        "expected_output": output,
        "qiskit_compilation_requested": False,
        "checks": {
            "qiskit_compilation_requested": False,
            "resume_input_present": False,
            "source_rebaseline_evidence_present": False,
        },
        "required_remote_preflight": [
            "verify_action_id_absent_from_prior_packages_and_scheduler_history"
        ],
        "files": files,
    }
    _write(manifest_path, manifest)
    bindings = {
        "action": _binding(root, action_path),
        "manifest": _binding(root, manifest_path),
        "submit": _binding(root, submit_path),
    }
    remote_workspace = "/home/test/Holstein_siam_nb4_source"
    recorded_at = "2026-08-29T06:22:33Z"
    upload = {
        "schema": "siam_nb4_remote_upload_receipt_v1",
        "status": "REMOTE_PACKAGE_VERIFIED_AWAITING_RECONCILIATION_NOT_SUBMITTED",
        "recorded_at_utc": recorded_at,
        "remote_workspace": remote_workspace,
        "remote_raw_evidence_preserved": True,
        "qiskit_compilation_requested": False,
        "source_revision": revision,
        "scientific_identity": {
            "source_epoch_id": SOURCE_EPOCH_ID,
            "action_id": ACTION_ID,
        },
        "bindings": bindings,
        "checks": {
            "condor_dry_run_passed": True,
            "local_package_manifest_verified": True,
            "local_source_epoch_control_plane_verified": True,
            "local_source_tree_contract_verified": True,
            "local_submit_lifecycle_verified": True,
            "quota_checked": True,
            "remote_archive_digest_verified": True,
            "remote_image_digest_verified": True,
            "remote_output_path_absent": True,
            "remote_package_file_digests_verified": True,
            "remote_source_archive_digest_verified": True,
            "remote_source_tree_contract_verified": True,
            "submission_autopilot_absent": True,
            "qiskit_compilation_requested": False,
        },
        "scheduler_reconciliation": {
            "required_next_state": "verified_absent",
            "state": "indeterminate",
            "submission_permitted": False,
        },
    }
    _write(upload_path, upload)
    intent = {
        "schema": "siam_nb4_submission_intent_v1",
        "created_at_utc": recorded_at,
        "state": "indeterminate",
        "submission_permitted": False,
        "required_next_reconciliation": "verified_absent",
        "qiskit_compilation_requested": False,
        "source_revision": revision,
        "remote_workspace": remote_workspace,
        "remote_output_path": output,
        "scientific_identity": {
            "source_epoch_id": SOURCE_EPOCH_ID,
            "action_id": ACTION_ID,
        },
        "bindings": {
            **bindings,
            "remote_upload_receipt": _binding(
                root, upload_path, include_bytes=False
            ),
        },
    }
    _write(intent_path, intent)
    return intent_path


def _capture_spec(*, captured_at: str, captured_at_ns: int) -> dict:
    queries = {}
    for name, command in (("queue", "condor_q"), ("history", "condor_history")):
        queries[name] = {
            "argv": [
                command,
                "-constraint",
                CONDITION,
                "-json",
                "-attributes",
                ATTRIBUTES,
            ],
            "returncode": 0,
            "stderr": "",
        }
    return {
        "schema": CAPTURE_SPEC_V1,
        "captured_at": captured_at,
        "captured_at_unix_ns": captured_at_ns,
        "single_combined_snapshot": True,
        "polling_performed": False,
        "capture_counts": {"queue": 1, "history": 1},
        "reconciliation_intents": {},
        "prior_snapshot": None,
        "scope": {
            "condition": CONDITION,
            "lanes": {
                "hh": {
                    "cluster_ids": [],
                    "job_batch_names": [],
                    "acquisition_lane_ids": [HH_ACQUISITION_LANE_ID],
                },
                "hubbard": {
                    "cluster_ids": [],
                    "job_batch_names": [],
                    "acquisition_lane_ids": [HUBBARD_ACQUISITION_LANE_ID],
                },
                "siam": {
                    "cluster_ids": [],
                    "job_batch_names": [BATCH],
                    "acquisition_lane_ids": [],
                },
            },
        },
        "queries": queries,
    }


def test_bridge_derives_generic_identity_from_digest_bound_siam_evidence(
    tmp_path: Path,
) -> None:
    intent_path = _package(tmp_path)

    receipt = build_bridge_receipt(intent_path, repository_root=tmp_path)

    generic = receipt["generic_reconciliation_intent"]
    assert generic["job_batch_name"] == BATCH
    assert generic["acquisition_unit_id"] == intent_path.parent.name
    assert generic["paper_ii_action_id"] == ACTION_ID
    assert generic["intent_sha256"] == _sha(intent_path)
    assert generic["recorded_at"] == "2026-08-29T06:22:33Z"
    assert generic["paper_ii_execution_request_sha256"] == receipt[
        "execution_request_sha256"
    ]
    assert generic["paper_ii_execution_request_sha256"] != receipt[
        "evidence_bindings"
    ]["action"]["sha256"]
    assert receipt["derivations"]["acquisition_unit_id"] == (
        "package_manifest.package_id"
    )


def test_intent_bound_first_and_later_snapshots_enforce_two_capture_absence(
    tmp_path: Path,
) -> None:
    intent_path = _package(tmp_path)
    bridge = build_bridge_receipt(intent_path, repository_root=tmp_path)
    queue = tmp_path / "queue.json"
    history = tmp_path / "history.json"
    queue.write_bytes(b"")
    history.write_bytes(b"[]")
    first_spec = intent_bound_capture_spec(
        _capture_spec(
            captured_at="2026-08-29T10:20:07-05:00",
            captured_at_ns=1788016807000000000,
        ),
        bridge,
    )
    first = build_combined_snapshot(
        queue_stdout=queue,
        history_stdout=history,
        capture_spec=first_spec,
    )
    first_outcome = first["reconciliation"][BATCH]
    assert first_outcome["status"] == "indeterminate"
    assert first_outcome["indeterminate_reason"] == (
        "first_snapshot_for_exact_intent_identity"
    )

    later_spec = intent_bound_capture_spec(
        _capture_spec(
            captured_at="2026-08-29T10:30:07-05:00",
            captured_at_ns=1788017407000000000,
        ),
        bridge,
        prior_snapshot=first,
    )
    later = build_combined_snapshot(
        queue_stdout=queue,
        history_stdout=history,
        capture_spec=later_spec,
    )
    later_outcome = later["reconciliation"][BATCH]
    assert later_outcome["status"] == "verified_absent"
    assert later_outcome["absence_proof"]["prior_snapshot_sha256"] == first[
        "snapshot_sha256"
    ]
    plan = later_capture_plan(
        first_snapshot=first,
        bridge_receipt=bridge,
        first_snapshot_path="output/first.json",
    )
    assert plan["remote_submit"]["argv"] == [
        "condor_submit",
        f"chtc/{intent_path.parent.name}/submit.sub",
    ]
    bridge_path = tmp_path / "output/bridge.json"
    first_path = tmp_path / "output/first.json"
    later_path = tmp_path / "output/later.json"
    _write(bridge_path, bridge)
    _write(first_path, first)
    _write(later_path, later)
    receipt = verified_absent_reconciliation_receipt(
        repository_root=tmp_path,
        bridge_path=bridge_path,
        first_snapshot_path=first_path,
        verified_snapshot_path=later_path,
        queue_stdout_path=queue,
        history_stdout_path=history,
    )
    assert receipt["state"] == "verified_absent"
    assert receipt["submission_permitted"] is True
    assert receipt["verified_absent_snapshot"]["query_evidence"]["queue"][
        "sha256"
    ] == hashlib.sha256(b"").hexdigest()
    reconciliation_path = tmp_path / "chtc/package/reconciliation.json"
    _write(reconciliation_path, receipt)
    submitted = build_post_submit_receipt(
        repository_root=tmp_path,
        reconciliation_receipt_path=reconciliation_path,
        cluster_id=9714484,
        condor_submit_summary="1 job(s) submitted to cluster 9714484.",
        recorded_at="2026-08-29T16:00:00Z",
        remote_output_absence_recheck_passed=True,
    )
    submission_path = tmp_path / "chtc/package/submission_receipt.json"
    _write(submission_path, submitted)
    assert submitted["scheduler_observation_after_submit"] == {
        "query_performed": False,
        "state": "not_queried_by_contract",
    }
    assert validate_post_submit_receipt(
        submission_path,
        repository_root=tmp_path,
    ) == submitted["submission_receipt_sha256"]


def test_bridge_rejects_a_digest_bound_but_noncanonical_batch(tmp_path: Path) -> None:
    intent_path = _package(tmp_path, batch="siam-r02-wrong")

    with pytest.raises(BridgeError, match="canonical batch"):
        build_bridge_receipt(intent_path, repository_root=tmp_path)


def test_bridge_rejects_tampering_of_a_bound_package_file(tmp_path: Path) -> None:
    intent_path = _package(tmp_path)
    submit = intent_path.parent / "submit.sub"
    submit.write_bytes(submit.read_bytes() + b"\n")

    with pytest.raises(BridgeError, match="submit.*digest"):
        build_bridge_receipt(intent_path, repository_root=tmp_path)


def test_batch_names_keep_historical_form_and_admit_the_compiled_q_arm() -> None:
    from pipelines.time_dynamics.siam_submission_intent_bridge import (
        _accepted_batch_names,
    )

    historical = _accepted_batch_names(
        segment="screen_t0_t1_101", arm="full_apm", ladder_index=0
    )
    assert historical == {"siam-nb4-screen-t0-t1-101-full-apm-rung0"}

    avqds = _accepted_batch_names(
        segment="continue_t1_t5_401", arm="avqds", ladder_index=7
    )
    assert avqds == {"siam-nb4-continue-t1-t5-401-avqds-rung7"}

    compiled = _accepted_batch_names(
        segment="screen_t0_t1_101", arm="exchange_q_compiled", ladder_index=0
    )
    assert (
        "siam-nb4-screen-t0-t1-101-exchange_q_compiled-rung0" in compiled
    )
    assert "siam-nb4-screen-t0-t1-101-exchange-q-compiled-rung0" in compiled
