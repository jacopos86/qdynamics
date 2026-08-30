from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from pipelines.time_dynamics.combined_scheduler_snapshot import (
    CAPTURE_SPEC_V1,
    COMBINED_SCOPED_SNAPSHOT_V1,
    HH_ACQUISITION_LANE_ID,
    HUBBARD_ACQUISITION_LANE_ID,
    REQUIRED_PROJECTED_ATTRIBUTES,
    CombinedSnapshotError,
    build_combined_snapshot,
    main,
    validate_combined_snapshot,
    write_immutable_snapshot,
)
ATTRIBUTES = (
    "ClusterId,ProcId,JobStatus,ExitCode,HoldReason,JobBatchName,"
    "AcquisitionUnitId,AcquisitionLaneId,PaperIIActionId,"
    "PaperIIExecutionRequestSha256,PaperIISourceEpochId,"
    "EnteredCurrentStatus,CompletionDate,QDate"
)
CONDITION = (
    '(ClusterId == 101 || ClusterId == 202 || ClusterId == 303) || '
    '(JobBatchName == "hubbard-next")'
)


def _write_json(path: Path, value: object, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        path.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")
    else:
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _rows() -> tuple[list[dict], list[dict]]:
    queue = [
        {
            "ClusterId": 101,
            "ProcId": 0,
            "JobStatus": 2,
            "JobBatchName": "hh-seed",
            "AcquisitionUnitId": "hh-unit",
            "AcquisitionLaneId": HH_ACQUISITION_LANE_ID,
            "PaperIIActionId": "build-seed-hh",
            "PaperIIExecutionRequestSha256": "a" * 64,
            "PaperIISourceEpochId": "hh-source-epoch",
            "EnteredCurrentStatus": 10,
            "QDate": 9,
        },
        {
            "ClusterId": 202,
            "ProcId": 0,
            "JobStatus": 1,
            "JobBatchName": "hubbard-seed",
            "AcquisitionUnitId": "hubbard-unit",
            "AcquisitionLaneId": HUBBARD_ACQUISITION_LANE_ID,
            "PaperIIActionId": "build-seed-hubbard",
            "PaperIIExecutionRequestSha256": "b" * 64,
            "PaperIISourceEpochId": "hubbard-source-epoch",
            "EnteredCurrentStatus": 11,
            "QDate": 8,
        },
    ]
    history = [
        {
            "ClusterId": 303,
            "ProcId": 0,
            "JobStatus": 4,
            "ExitCode": 0,
            "JobBatchName": "siam-r02",
            "PaperIISourceEpochId": "siam-source-epoch",
            "EnteredCurrentStatus": 12,
            "CompletionDate": 12,
            "QDate": 7,
        }
    ]
    return queue, history


def _spec() -> dict:
    return {
        "schema": CAPTURE_SPEC_V1,
        "captured_at": "2026-08-29T01:30:00-05:00",
        "captured_at_unix_ns": 1787985000000000000,
        "single_combined_snapshot": True,
        "polling_performed": False,
        "capture_counts": {"queue": 1, "history": 1},
        "reconciliation_intents": {
            "hubbard-next": {
                "job_batch_name": "hubbard-next",
                "acquisition_unit_id": "hubbard-next-unit",
                "paper_ii_action_id": "build-seed-hubbard-next",
                "paper_ii_execution_request_sha256": "c" * 64,
                "intent_sha256": "d" * 64,
                "recorded_at": "2026-08-29T01:00:00-05:00",
            }
        },
        "prior_snapshot": None,
        "scope": {
            "condition": CONDITION,
            "lanes": {
                "hh": {
                    "cluster_ids": [101],
                    "job_batch_names": ["hh-seed"],
                    "acquisition_lane_ids": [HH_ACQUISITION_LANE_ID],
                },
                "hubbard": {
                    "cluster_ids": [202],
                    "job_batch_names": ["hubbard-next", "hubbard-seed"],
                    "acquisition_lane_ids": [HUBBARD_ACQUISITION_LANE_ID],
                },
                "siam": {
                    "cluster_ids": [303],
                    "job_batch_names": ["siam-r02"],
                    "acquisition_lane_ids": [],
                },
            },
        },
        "queries": {
            "queue": {
                "argv": [
                    "condor_q",
                    "jsstrobel",
                    "-constraint",
                    CONDITION,
                    "-json",
                    "-attributes",
                    ATTRIBUTES,
                ],
                "returncode": 0,
                "stderr": "",
            },
            "history": {
                "argv": [
                    "condor_history",
                    "jsstrobel",
                    "-constraint",
                    CONDITION,
                    "-json",
                    "-attributes",
                    ATTRIBUTES,
                ],
                "returncode": 0,
                "stderr": "",
            },
        },
    }


def _inputs(tmp_path: Path) -> tuple[Path, Path, dict]:
    queue, history = _rows()
    queue_path = tmp_path / "condor_q.json"
    history_path = tmp_path / "condor_history.json"
    _write_json(queue_path, queue)
    _write_json(history_path, history, compact=True)
    return queue_path, history_path, _spec()


def test_builds_classified_self_digested_snapshot_from_preserved_stdout(
    tmp_path: Path,
) -> None:
    queue_path, history_path, spec = _inputs(tmp_path)

    snapshot = build_combined_snapshot(
        queue_stdout=queue_path,
        history_stdout=history_path,
        capture_spec=spec,
    )

    assert snapshot["schema"] == COMBINED_SCOPED_SNAPSHOT_V1
    assert snapshot["single_combined_snapshot"] is True
    assert snapshot["polling_performed"] is False
    assert snapshot["capture_counts"] == {"queue": 1, "history": 1}
    assert [row["ClusterId"] for row in snapshot["hh_rows"]] == [101]
    assert [row["ClusterId"] for row in snapshot["hubbard_rows"]] == [202]
    assert [row["ClusterId"] for row in snapshot["siam_rows"]] == [303]
    assert snapshot["hh_rows"][0]["PaperIISourceEpochId"] == "hh-source-epoch"
    assert set(REQUIRED_PROJECTED_ATTRIBUTES).issubset(
        snapshot["scope"]["projected_attributes"]
    )
    assert snapshot["queries"]["queue"]["argv"] == spec["queries"]["queue"][
        "argv"
    ]
    assert snapshot["queries"]["queue"]["stdout_sha256"] == hashlib.sha256(
        queue_path.read_bytes()
    ).hexdigest()
    assert snapshot["queries"]["history"]["stdout_sha256"] == hashlib.sha256(
        history_path.read_bytes()
    ).hexdigest()
    first_absence = snapshot["reconciliation"]["hubbard-next"]
    assert first_absence == {
        "status": "indeterminate",
        "match_count": 0,
        "matches": [],
        "intent_identity_sha256": first_absence["intent_identity_sha256"],
        "indeterminate_reason": "first_snapshot_for_exact_intent_identity",
    }
    assert snapshot["reconciliation"]["siam-r02"]["status"] == "found"
    assert snapshot["reconciliation"]["siam-r02"]["matches"][0][
        "paper_ii_source_epoch_id"
    ] == "siam-source-epoch"
    assert validate_combined_snapshot(
        snapshot,
        queue_stdout=queue_path,
        history_stdout=history_path,
    ) == snapshot["snapshot_sha256"]


def test_zero_byte_condor_json_stdout_is_an_authenticated_empty_row_set(
    tmp_path: Path,
) -> None:
    queue, history = _rows()
    queue_path = tmp_path / "condor_q.json"
    history_path = tmp_path / "condor_history.json"
    queue_path.write_bytes(b"")
    _write_json(history_path, [*queue, *history])

    snapshot = build_combined_snapshot(
        queue_stdout=queue_path,
        history_stdout=history_path,
        capture_spec=_spec(),
    )

    assert snapshot["queries"]["queue"]["rows"] == []
    assert snapshot["queries"]["queue"]["stdout_sha256"] == hashlib.sha256(
        b""
    ).hexdigest()
    assert [row["ClusterId"] for row in snapshot["hh_rows"]] == [101]
    assert [row["ClusterId"] for row in snapshot["hubbard_rows"]] == [202]
    assert [row["ClusterId"] for row in snapshot["siam_rows"]] == [303]
    assert validate_combined_snapshot(
        snapshot,
        queue_stdout=queue_path,
        history_stdout=history_path,
    ) == snapshot["snapshot_sha256"]


def test_whitespace_only_condor_stdout_remains_invalid(tmp_path: Path) -> None:
    queue_path, history_path, spec = _inputs(tmp_path)
    queue_path.write_bytes(b"\n")

    with pytest.raises(CombinedSnapshotError, match="not readable JSON"):
        build_combined_snapshot(
            queue_stdout=queue_path,
            history_stdout=history_path,
            capture_spec=spec,
        )


def test_only_later_same_intent_snapshot_proves_verified_absent(
    tmp_path: Path,
) -> None:
    queue_path, history_path, first_spec = _inputs(tmp_path)
    first = build_combined_snapshot(
        queue_stdout=queue_path,
        history_stdout=history_path,
        capture_spec=first_spec,
    )
    batch = "hubbard-next"
    assert first["reconciliation"][batch]["status"] == "indeterminate"

    later_spec = copy.deepcopy(first_spec)
    later_spec["captured_at"] = "2026-08-29T01:31:00-05:00"
    later_spec["captured_at_unix_ns"] += 60_000_000_000
    later_spec["prior_snapshot"] = first
    later = build_combined_snapshot(
        queue_stdout=queue_path,
        history_stdout=history_path,
        capture_spec=later_spec,
    )
    outcome = later["reconciliation"][batch]
    assert outcome["status"] == "verified_absent"
    assert outcome["absence_proof"]["prior_snapshot_sha256"] == (
        first["snapshot_sha256"]
    )
    assert outcome["absence_proof"]["prior_snapshot_status"] == "indeterminate"
    assert validate_combined_snapshot(later) == later["snapshot_sha256"]

    changed_intent_spec = copy.deepcopy(later_spec)
    changed_intent_spec["reconciliation_intents"][batch][
        "paper_ii_action_id"
    ] = "different-action"
    changed = build_combined_snapshot(
        queue_stdout=queue_path,
        history_stdout=history_path,
        capture_spec=changed_intent_spec,
    )
    assert changed["reconciliation"][batch]["status"] == "indeterminate"


def test_immutable_writer_accepts_identical_replay_and_rejects_divergence(
    tmp_path: Path,
) -> None:
    queue_path, history_path, spec = _inputs(tmp_path)
    snapshot = build_combined_snapshot(
        queue_stdout=queue_path,
        history_stdout=history_path,
        capture_spec=spec,
    )
    output = tmp_path / "snapshot.json"

    assert write_immutable_snapshot(output, snapshot) == "created"
    first = output.read_bytes()
    assert write_immutable_snapshot(output, snapshot) == "unchanged"
    assert output.read_bytes() == first

    changed_spec = copy.deepcopy(spec)
    changed_spec["captured_at_unix_ns"] += 1
    changed = build_combined_snapshot(
        queue_stdout=queue_path,
        history_stdout=history_path,
        capture_spec=changed_spec,
    )
    with pytest.raises(CombinedSnapshotError, match="refusing to overwrite"):
        write_immutable_snapshot(output, changed)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("single_combined_snapshot", False, "one combined"),
        ("polling_performed", True, "polling is forbidden"),
        ("capture_counts", {"queue": 2, "history": 1}, "capture_counts"),
    ],
)
def test_rejects_polling_or_more_than_one_capture(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    queue_path, history_path, spec = _inputs(tmp_path)
    spec[field] = value

    with pytest.raises(CombinedSnapshotError, match=message):
        build_combined_snapshot(
            queue_stdout=queue_path,
            history_stdout=history_path,
            capture_spec=spec,
        )


def test_requires_source_epoch_in_the_exact_shared_projection(tmp_path: Path) -> None:
    queue_path, history_path, spec = _inputs(tmp_path)
    for query in spec["queries"].values():
        index = query["argv"].index("-attributes") + 1
        query["argv"][index] = ATTRIBUTES.replace(
            ",PaperIISourceEpochId", ""
        )

    with pytest.raises(CombinedSnapshotError, match="PaperIISourceEpochId"):
        build_combined_snapshot(
            queue_stdout=queue_path,
            history_stdout=history_path,
            capture_spec=spec,
        )


def test_rejects_different_query_scope_or_projection(tmp_path: Path) -> None:
    queue_path, history_path, spec = _inputs(tmp_path)
    history_argv = spec["queries"]["history"]["argv"]
    history_argv[history_argv.index("-constraint") + 1] = "ClusterId == 303"

    with pytest.raises(CombinedSnapshotError, match="constraint differs"):
        build_combined_snapshot(
            queue_stdout=queue_path,
            history_stdout=history_path,
            capture_spec=spec,
        )

    spec = _spec()
    history_argv = spec["queries"]["history"]["argv"]
    history_argv[history_argv.index("-attributes") + 1] += ",RemoteHost"
    with pytest.raises(CombinedSnapshotError, match="same ordered"):
        build_combined_snapshot(
            queue_stdout=queue_path,
            history_stdout=history_path,
            capture_spec=spec,
        )


def test_rejects_unclassified_ambiguous_or_duplicate_rows(tmp_path: Path) -> None:
    queue_path, history_path, spec = _inputs(tmp_path)
    queue, _ = _rows()
    queue[0]["ClusterId"] = 999
    queue[0].pop("JobBatchName")
    queue[0].pop("AcquisitionLaneId")
    _write_json(queue_path, queue)
    with pytest.raises(CombinedSnapshotError, match="exactly one lane"):
        build_combined_snapshot(
            queue_stdout=queue_path,
            history_stdout=history_path,
            capture_spec=spec,
        )

    queue, _ = _rows()
    queue[0]["JobBatchName"] = "hubbard-seed"
    _write_json(queue_path, queue)
    with pytest.raises(CombinedSnapshotError, match="exactly one lane"):
        build_combined_snapshot(
            queue_stdout=queue_path,
            history_stdout=history_path,
            capture_spec=spec,
        )

    queue, history = _rows()
    history[0]["ClusterId"] = queue[0]["ClusterId"]
    history[0]["ProcId"] = queue[0]["ProcId"]
    history[0]["JobBatchName"] = "hh-seed"
    _write_json(queue_path, queue)
    _write_json(history_path, history)
    with pytest.raises(CombinedSnapshotError, match="appears more than once"):
        build_combined_snapshot(
            queue_stdout=queue_path,
            history_stdout=history_path,
            capture_spec=spec,
        )


def test_validator_rejects_tamper_and_raw_stdout_mismatch(tmp_path: Path) -> None:
    queue_path, history_path, spec = _inputs(tmp_path)
    snapshot = build_combined_snapshot(
        queue_stdout=queue_path,
        history_stdout=history_path,
        capture_spec=spec,
    )

    tampered = copy.deepcopy(snapshot)
    tampered["hh_rows"][0]["JobStatus"] = 3
    with pytest.raises(CombinedSnapshotError, match="self digest mismatch"):
        validate_combined_snapshot(tampered)

    queue_path.write_bytes(queue_path.read_bytes() + b"\n")
    with pytest.raises(CombinedSnapshotError, match="queue stdout digest mismatch"):
        validate_combined_snapshot(
            snapshot,
            queue_stdout=queue_path,
            history_stdout=history_path,
        )


def test_cli_build_and_check_are_idempotent(tmp_path: Path, capsys) -> None:
    queue_path, history_path, spec = _inputs(tmp_path)
    spec_path = tmp_path / "capture_spec.json"
    output = tmp_path / "combined_snapshot.json"
    _write_json(spec_path, spec)
    build_args = [
        "build",
        "--queue-json",
        str(queue_path),
        "--history-json",
        str(history_path),
        "--capture-spec",
        str(spec_path),
        "--output",
        str(output),
    ]

    assert main(build_args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "created"
    assert main(build_args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "unchanged"
    assert main(
        [
            "check",
            "--snapshot",
            str(output),
            "--queue-json",
            str(queue_path),
            "--history-json",
            str(history_path),
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "valid"
