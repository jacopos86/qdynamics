"""Build deterministic Paper-II snapshots from preserved Condor JSON output.

This module is intentionally offline.  It accepts one already-captured
``condor_q -json`` stdout payload and one already-captured
``condor_history -json`` stdout payload, authenticates both against an explicit
capture specification, and writes the immutable combined snapshot consumed by
the prospective submission-recovery code.  It never invokes Condor, SSH, or
any other external process.

The capture specification makes the operational discipline machine-checkable:

* exactly one queue capture and one history capture belong to the snapshot;
* polling was not performed;
* both commands used the same literal scope and projected attributes;
* ``PaperIISourceEpochId`` was included in that projection; and
* every returned scheduler row belongs unambiguously to the HH, Hubbard, or
  SIAM lane scope.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


CAPTURE_SPEC_V1 = "paper_ii_combined_scheduler_capture_spec_v1"
COMBINED_SCOPED_SNAPSHOT_V1 = "paper_ii_combined_scoped_scheduler_snapshot_v1"
RECONCILIATION_CONTEXT_V1 = "paper_ii_intent_first_reconciliation_context_v1"
ABSENCE_PROOF_V1 = "paper_ii_intent_first_absence_proof_v1"

LANE_HH = "hh"
LANE_HUBBARD = "hubbard"
LANE_SIAM = "siam"
LANES = (LANE_HH, LANE_HUBBARD, LANE_SIAM)

HH_ACQUISITION_LANE_ID = "paper_ii_hh_product_v1"
HUBBARD_ACQUISITION_LANE_ID = "paper_ii_hubbard_l4_scale_v1"

REQUIRED_PROJECTED_ATTRIBUTES = frozenset(
    {
        "ClusterId",
        "ProcId",
        "JobStatus",
        "ExitCode",
        "HoldReason",
        "JobBatchName",
        "AcquisitionUnitId",
        "AcquisitionLaneId",
        "PaperIIActionId",
        "PaperIIExecutionRequestSha256",
        "PaperIISourceEpochId",
        "EnteredCurrentStatus",
        "CompletionDate",
        "QDate",
    }
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_INTENT_FIELDS = (
    "job_batch_name",
    "acquisition_unit_id",
    "paper_ii_action_id",
    "paper_ii_execution_request_sha256",
    "intent_sha256",
    "recorded_at",
)


class CombinedSnapshotError(ValueError):
    """The capture is ambiguous, incomplete, stale, or noncanonical."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _pretty_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CombinedSnapshotError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CombinedSnapshotError(f"{label} must be a JSON object")
    return value


def _read_rows(path: Path, *, label: str) -> tuple[bytes, list[dict[str, Any]]]:
    try:
        raw = path.read_bytes()
        value = [] if raw == b"" else json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CombinedSnapshotError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise CombinedSnapshotError(f"{label} must be a JSON array of objects")
    return raw, [dict(row) for row in value]


def _aware_timestamp(value: Any) -> str:
    text = str(value or "")
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CombinedSnapshotError(
            "captured_at must be an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CombinedSnapshotError("captured_at must include an explicit UTC offset")
    return text


def _parsed_timestamp(value: Any, *, label: str) -> dt.datetime:
    text = str(value or "")
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CombinedSnapshotError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CombinedSnapshotError(f"{label} must include an explicit UTC offset")
    return parsed


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CombinedSnapshotError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CombinedSnapshotError(f"{label} must be an integer >= {minimum}")
    return value


def _string_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list):
        raise CombinedSnapshotError(f"{label} must be a list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise CombinedSnapshotError(f"{label} entries must be nonempty strings")
        result.append(item)
    if len(result) != len(set(result)):
        raise CombinedSnapshotError(f"{label} contains duplicates")
    return sorted(result)


def _integer_list(value: Any, *, label: str) -> list[int]:
    if not isinstance(value, list):
        raise CombinedSnapshotError(f"{label} must be a list")
    result = [
        _integer(item, label=f"{label} entry", minimum=0)
        for item in value
    ]
    if len(result) != len(set(result)):
        raise CombinedSnapshotError(f"{label} contains duplicates")
    return sorted(result)


def _argument_after(argv: Sequence[str], flag: str, *, query_name: str) -> str:
    positions = [index for index, value in enumerate(argv) if value == flag]
    if len(positions) != 1:
        raise CombinedSnapshotError(
            f"{query_name} argv must contain exactly one {flag} option"
        )
    index = positions[0]
    if index + 1 >= len(argv) or not argv[index + 1]:
        raise CombinedSnapshotError(f"{query_name} {flag} has no value")
    return argv[index + 1]


def _query_contract(
    raw: Any,
    *,
    query_name: str,
    expected_command: str,
    condition: str,
) -> tuple[list[str], list[str], int, str]:
    if not isinstance(raw, Mapping):
        raise CombinedSnapshotError(f"{query_name} query metadata must be an object")
    argv_raw = raw.get("argv")
    if (
        not isinstance(argv_raw, list)
        or not argv_raw
        or not all(isinstance(value, str) and value for value in argv_raw)
    ):
        raise CombinedSnapshotError(f"{query_name} argv must be a string list")
    argv = list(argv_raw)
    if Path(argv[0]).name != expected_command:
        raise CombinedSnapshotError(
            f"{query_name} argv must invoke {expected_command}, not {argv[0]}"
        )
    if argv.count("-json") != 1:
        raise CombinedSnapshotError(
            f"{query_name} argv must contain exactly one -json option"
        )
    observed_condition = _argument_after(
        argv, "-constraint", query_name=query_name
    )
    if observed_condition != condition:
        raise CombinedSnapshotError(
            f"{query_name} constraint differs from the declared combined scope"
        )
    attributes_text = _argument_after(
        argv, "-attributes", query_name=query_name
    )
    attributes = [value.strip() for value in attributes_text.split(",")]
    if any(not value for value in attributes) or len(attributes) != len(
        set(attributes)
    ):
        raise CombinedSnapshotError(
            f"{query_name} projected attributes are empty or duplicated"
        )
    missing = sorted(REQUIRED_PROJECTED_ATTRIBUTES.difference(attributes))
    if missing:
        raise CombinedSnapshotError(
            f"{query_name} projection is missing required attributes: {missing}"
        )
    returncode = _integer(
        raw.get("returncode"), label=f"{query_name} returncode", minimum=0
    )
    if returncode != 0:
        raise CombinedSnapshotError(f"{query_name} capture did not succeed")
    stderr = raw.get("stderr", "")
    if not isinstance(stderr, str):
        raise CombinedSnapshotError(f"{query_name} stderr must be a string")
    return argv, attributes, returncode, stderr


def _lane_scopes(raw: Any) -> dict[str, dict[str, list[Any]]]:
    if not isinstance(raw, Mapping) or set(raw) != set(LANES):
        raise CombinedSnapshotError(
            "scope.lanes must contain exactly hh, hubbard, and siam"
        )
    result: dict[str, dict[str, list[Any]]] = {}
    used_clusters: dict[int, str] = {}
    used_batches: dict[str, str] = {}
    used_lane_ids: dict[str, str] = {}
    for lane in LANES:
        value = raw[lane]
        if not isinstance(value, Mapping):
            raise CombinedSnapshotError(f"scope.lanes.{lane} must be an object")
        unexpected = set(value).difference(
            {"cluster_ids", "job_batch_names", "acquisition_lane_ids"}
        )
        if unexpected:
            raise CombinedSnapshotError(
                f"scope.lanes.{lane} has unsupported fields: {sorted(unexpected)}"
            )
        clusters = _integer_list(
            value.get("cluster_ids", []),
            label=f"scope.lanes.{lane}.cluster_ids",
        )
        batches = _string_list(
            value.get("job_batch_names", []),
            label=f"scope.lanes.{lane}.job_batch_names",
        )
        lane_ids = _string_list(
            value.get("acquisition_lane_ids", []),
            label=f"scope.lanes.{lane}.acquisition_lane_ids",
        )
        if lane == LANE_HH and HH_ACQUISITION_LANE_ID not in lane_ids:
            raise CombinedSnapshotError(
                f"HH scope must include acquisition lane {HH_ACQUISITION_LANE_ID}"
            )
        if (
            lane == LANE_HUBBARD
            and HUBBARD_ACQUISITION_LANE_ID not in lane_ids
        ):
            raise CombinedSnapshotError(
                "Hubbard scope must include acquisition lane "
                f"{HUBBARD_ACQUISITION_LANE_ID}"
            )
        for item, owner in ((cluster, lane) for cluster in clusters):
            previous = used_clusters.setdefault(item, owner)
            if previous != lane:
                raise CombinedSnapshotError(
                    f"cluster {item} is assigned to both {previous} and {lane}"
                )
        for item, owner in ((batch, lane) for batch in batches):
            previous = used_batches.setdefault(item, owner)
            if previous != lane:
                raise CombinedSnapshotError(
                    f"job batch {item} is assigned to both {previous} and {lane}"
                )
        for item, owner in ((lane_id, lane) for lane_id in lane_ids):
            previous = used_lane_ids.setdefault(item, owner)
            if previous != lane:
                raise CombinedSnapshotError(
                    f"acquisition lane {item} maps to both {previous} and {lane}"
                )
        result[lane] = {
            "cluster_ids": clusters,
            "job_batch_names": batches,
            "acquisition_lane_ids": lane_ids,
        }
    return result


def _normalize_scope(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise CombinedSnapshotError("scope must be an object")
    if set(raw) != {"condition", "lanes"}:
        raise CombinedSnapshotError("scope must contain exactly condition and lanes")
    condition = raw.get("condition")
    if not isinstance(condition, str) or not condition:
        raise CombinedSnapshotError("scope.condition must be a nonempty string")
    return {"condition": condition, "lanes": _lane_scopes(raw.get("lanes"))}


def _normalize_reconciliation_intents(
    raw: Any,
    *,
    scope: Mapping[str, Any],
    captured_at: str,
) -> dict[str, dict[str, Any]]:
    """Bind durable intent identity independently of scheduler observations."""

    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise CombinedSnapshotError("reconciliation_intents must be an object")
    scoped_batches = {
        batch
        for lane in LANES
        for batch in scope["lanes"][lane]["job_batch_names"]
    }
    capture_time = _parsed_timestamp(captured_at, label="captured_at")
    result: dict[str, dict[str, Any]] = {}
    for batch, value in raw.items():
        if not isinstance(batch, str) or not batch:
            raise CombinedSnapshotError(
                "reconciliation_intents keys must be nonempty job batch names"
            )
        if batch not in scoped_batches:
            raise CombinedSnapshotError(
                f"reconciliation intent batch is outside scheduler scope: {batch}"
            )
        if not isinstance(value, Mapping) or set(value) != set(_INTENT_FIELDS):
            raise CombinedSnapshotError(
                f"reconciliation intent {batch!r} must contain exactly "
                f"{list(_INTENT_FIELDS)}"
            )
        core = {field: value.get(field) for field in _INTENT_FIELDS}
        if core["job_batch_name"] != batch:
            raise CombinedSnapshotError(
                f"reconciliation intent {batch!r} job_batch_name changed"
            )
        for field in (
            "job_batch_name",
            "acquisition_unit_id",
            "paper_ii_action_id",
        ):
            if not isinstance(core[field], str) or not core[field]:
                raise CombinedSnapshotError(
                    f"reconciliation intent {batch!r} {field} must be nonempty"
                )
        _sha256(
            core["paper_ii_execution_request_sha256"],
            label=f"reconciliation intent {batch!r} request digest",
        )
        _sha256(
            core["intent_sha256"],
            label=f"reconciliation intent {batch!r} self digest",
        )
        recorded_time = _parsed_timestamp(
            core["recorded_at"],
            label=f"reconciliation intent {batch!r} recorded_at",
        )
        if recorded_time > capture_time:
            raise CombinedSnapshotError(
                f"reconciliation intent {batch!r} was recorded after capture"
            )
        result[batch] = {
            **core,
            "intent_identity_sha256": _digest(core),
        }
    return {batch: result[batch] for batch in sorted(result)}


def _normalize_prior_snapshot_binding(
    raw: Any,
    *,
    intents: Mapping[str, Mapping[str, Any]],
    captured_at: str,
    captured_at_unix_ns: int,
    validated_snapshot: bool,
) -> dict[str, Any] | None:
    """Return the prior immutable capture evidence relevant to current intents."""

    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise CombinedSnapshotError("prior_snapshot must be an object or null")
    if validated_snapshot:
        validate_combined_snapshot(raw)
        prior_context = raw.get("reconciliation_context")
        prior_reconciliation = raw.get("reconciliation")
        if not isinstance(prior_context, Mapping) or not isinstance(
            prior_reconciliation, Mapping
        ):
            raise CombinedSnapshotError(
                "prior snapshot lacks intent-first reconciliation context"
            )
        prior_intents = prior_context.get("intents")
        if not isinstance(prior_intents, Mapping):
            raise CombinedSnapshotError("prior snapshot intent context is invalid")
        intent_outcomes: dict[str, Any] = {}
        for batch, intent in intents.items():
            prior_intent = prior_intents.get(batch)
            prior_outcome = prior_reconciliation.get(batch)
            if (
                isinstance(prior_intent, Mapping)
                and isinstance(prior_outcome, Mapping)
                and prior_intent.get("intent_identity_sha256")
                == intent["intent_identity_sha256"]
                and prior_outcome.get("intent_identity_sha256")
                == intent["intent_identity_sha256"]
            ):
                intent_outcomes[batch] = {
                    "status": prior_outcome.get("status"),
                    "intent_identity_sha256": intent["intent_identity_sha256"],
                }
        binding = {
            "snapshot_sha256": raw.get("snapshot_sha256"),
            "captured_at": raw.get("captured_at"),
            "captured_at_unix_ns": raw.get("captured_at_unix_ns"),
            "intent_outcomes": intent_outcomes,
        }
    else:
        binding = dict(raw)
        if set(binding) != {
            "snapshot_sha256",
            "captured_at",
            "captured_at_unix_ns",
            "intent_outcomes",
        }:
            raise CombinedSnapshotError("prior snapshot binding fields are invalid")
    _sha256(binding.get("snapshot_sha256"), label="prior snapshot digest")
    prior_at = _aware_timestamp(binding.get("captured_at"))
    prior_time = _parsed_timestamp(prior_at, label="prior snapshot captured_at")
    current_time = _parsed_timestamp(captured_at, label="captured_at")
    prior_ns = _integer(
        binding.get("captured_at_unix_ns"),
        label="prior snapshot captured_at_unix_ns",
        minimum=1,
    )
    if prior_time >= current_time or prior_ns >= captured_at_unix_ns:
        raise CombinedSnapshotError(
            "prior snapshot must demonstrably precede the current capture"
        )
    outcomes = binding.get("intent_outcomes")
    if not isinstance(outcomes, Mapping):
        raise CombinedSnapshotError("prior snapshot intent outcomes are invalid")
    normalized_outcomes: dict[str, Any] = {}
    for batch, outcome in outcomes.items():
        if batch not in intents or not isinstance(outcome, Mapping):
            raise CombinedSnapshotError(
                "prior snapshot outcome is not bound to a current intent"
            )
        if set(outcome) != {"status", "intent_identity_sha256"}:
            raise CombinedSnapshotError("prior snapshot outcome fields are invalid")
        status = outcome.get("status")
        if status not in {"found", "verified_absent", "indeterminate"}:
            raise CombinedSnapshotError("prior snapshot outcome status is invalid")
        if outcome.get("intent_identity_sha256") != intents[batch][
            "intent_identity_sha256"
        ]:
            raise CombinedSnapshotError("prior snapshot intent identity changed")
        normalized_outcomes[batch] = dict(outcome)
    return {
        "snapshot_sha256": binding["snapshot_sha256"],
        "captured_at": prior_at,
        "captured_at_unix_ns": prior_ns,
        "intent_outcomes": {
            batch: normalized_outcomes[batch]
            for batch in sorted(normalized_outcomes)
        },
    }


def _scheduler_identity(row: Mapping[str, Any], *, label: str) -> tuple[int, int]:
    cluster = _integer(row.get("ClusterId"), label=f"{label} ClusterId")
    process = _integer(row.get("ProcId"), label=f"{label} ProcId")
    _integer(row.get("JobStatus"), label=f"{label} JobStatus", minimum=1)
    source_epoch = row.get("PaperIISourceEpochId")
    if source_epoch is not None and (
        not isinstance(source_epoch, str) or not source_epoch
    ):
        raise CombinedSnapshotError(
            f"{label} PaperIISourceEpochId must be a nonempty string when defined"
        )
    return cluster, process


def _classify_row(
    row: Mapping[str, Any], lane_scopes: Mapping[str, Mapping[str, Sequence[Any]]]
) -> str:
    cluster = int(row["ClusterId"])
    batch = row.get("JobBatchName")
    lane_id = row.get("AcquisitionLaneId")
    candidates: set[str] = set()
    for lane, scope in lane_scopes.items():
        if cluster in scope["cluster_ids"]:
            candidates.add(lane)
        if isinstance(batch, str) and batch in scope["job_batch_names"]:
            candidates.add(lane)
        if isinstance(lane_id, str) and lane_id in scope["acquisition_lane_ids"]:
            candidates.add(lane)
    if len(candidates) != 1:
        raise CombinedSnapshotError(
            "scheduler row must resolve to exactly one lane: "
            f"{cluster}.{row['ProcId']} -> {sorted(candidates)}"
        )
    return next(iter(candidates))


def _matching_rows(
    *,
    batch: str,
    rows_by_query: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for query_name in ("queue", "history"):
        for row in rows_by_query[query_name]:
            if row.get("JobBatchName") == batch:
                result.append(
                    {
                        "query": query_name,
                        "cluster_id": row["ClusterId"],
                        "proc_id": row["ProcId"],
                        "job_status": row["JobStatus"],
                        "acquisition_lane_id": row.get("AcquisitionLaneId"),
                        "acquisition_unit_id": row.get("AcquisitionUnitId"),
                        "paper_ii_action_id": row.get("PaperIIActionId"),
                        "paper_ii_execution_request_sha256": row.get(
                            "PaperIIExecutionRequestSha256"
                        ),
                        "paper_ii_source_epoch_id": row.get(
                            "PaperIISourceEpochId"
                        ),
                    }
                )
    return result


def _assemble_snapshot(
    capture_spec: Mapping[str, Any],
    *,
    rows: Mapping[str, list[dict[str, Any]]],
    stdout_sha256: Mapping[str, str],
    reconciliation_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if capture_spec.get("schema") != CAPTURE_SPEC_V1:
        raise CombinedSnapshotError("capture specification schema is not supported")
    if capture_spec.get("single_combined_snapshot") is not True:
        raise CombinedSnapshotError("capture must be one combined scheduler snapshot")
    if capture_spec.get("polling_performed") is not False:
        raise CombinedSnapshotError("polling is forbidden for a combined snapshot")
    counts = capture_spec.get("capture_counts")
    if not isinstance(counts, Mapping) or dict(counts) != {
        "queue": 1,
        "history": 1,
    }:
        raise CombinedSnapshotError(
            "capture_counts must prove one queue and one history capture"
        )
    captured_at = _aware_timestamp(capture_spec.get("captured_at"))
    captured_at_unix_ns = _integer(
        capture_spec.get("captured_at_unix_ns"),
        label="captured_at_unix_ns",
        minimum=1,
    )
    scope = _normalize_scope(capture_spec.get("scope"))
    if reconciliation_context is None:
        intents = _normalize_reconciliation_intents(
            capture_spec.get("reconciliation_intents"),
            scope=scope,
            captured_at=captured_at,
        )
        prior_binding = _normalize_prior_snapshot_binding(
            capture_spec.get("prior_snapshot"),
            intents=intents,
            captured_at=captured_at,
            captured_at_unix_ns=captured_at_unix_ns,
            validated_snapshot=True,
        )
    else:
        if (
            reconciliation_context.get("schema") != RECONCILIATION_CONTEXT_V1
            or set(reconciliation_context)
            != {"schema", "intents", "prior_snapshot"}
        ):
            raise CombinedSnapshotError(
                "snapshot reconciliation context is not canonical"
            )
        raw_intents = reconciliation_context.get("intents")
        if not isinstance(raw_intents, Mapping):
            raise CombinedSnapshotError("snapshot reconciliation intents are invalid")
        unsealed_intents: dict[str, Any] = {}
        for batch, intent in raw_intents.items():
            if not isinstance(intent, Mapping):
                raise CombinedSnapshotError(
                    "snapshot reconciliation intent must be an object"
                )
            unsealed_intents[batch] = {
                field: intent.get(field) for field in _INTENT_FIELDS
            }
        intents = _normalize_reconciliation_intents(
            unsealed_intents,
            scope=scope,
            captured_at=captured_at,
        )
        if dict(raw_intents) != intents:
            raise CombinedSnapshotError(
                "snapshot reconciliation intent identity is invalid"
            )
        prior_binding = _normalize_prior_snapshot_binding(
            reconciliation_context.get("prior_snapshot"),
            intents=intents,
            captured_at=captured_at,
            captured_at_unix_ns=captured_at_unix_ns,
            validated_snapshot=False,
        )
    normalized_reconciliation_context = {
        "schema": RECONCILIATION_CONTEXT_V1,
        "intents": intents,
        "prior_snapshot": prior_binding,
    }
    queries = capture_spec.get("queries")
    if not isinstance(queries, Mapping) or set(queries) != {"queue", "history"}:
        raise CombinedSnapshotError(
            "capture specification must contain exactly queue and history queries"
        )

    query_contracts: dict[str, tuple[list[str], list[str], int, str]] = {}
    for name, command in (("queue", "condor_q"), ("history", "condor_history")):
        query_contracts[name] = _query_contract(
            queries[name],
            query_name=name,
            expected_command=command,
            condition=scope["condition"],
        )
    if query_contracts["queue"][1] != query_contracts["history"][1]:
        raise CombinedSnapshotError(
            "queue and history must use the same ordered attribute projection"
        )
    projected_attributes = query_contracts["queue"][1]

    for name in ("queue", "history"):
        if not isinstance(rows.get(name), list) or not all(
            isinstance(row, dict) for row in rows[name]
        ):
            raise CombinedSnapshotError(f"{name} rows must be a list of objects")
        digest = stdout_sha256.get(name)
        if not isinstance(digest, str) or len(digest) != 64:
            raise CombinedSnapshotError(f"{name} stdout digest is invalid")

    classified: dict[str, list[dict[str, Any]]] = {lane: [] for lane in LANES}
    seen: set[tuple[int, int]] = set()
    for query_name in ("queue", "history"):
        for index, row in enumerate(rows[query_name]):
            identity = _scheduler_identity(
                row, label=f"{query_name} row {index}"
            )
            if identity in seen:
                raise CombinedSnapshotError(
                    "scheduler identity appears more than once in the combined "
                    f"row set: {identity[0]}.{identity[1]}"
                )
            seen.add(identity)
            lane = _classify_row(row, scope["lanes"])
            classified[lane].append(dict(row))

    snapshot_scope = {
        "condition": scope["condition"],
        "projected_attributes": projected_attributes,
        "lanes": scope["lanes"],
        # Compatibility fields consumed by the existing recovery bridge.
        "active_siam_cluster_ids": scope["lanes"][LANE_SIAM]["cluster_ids"],
        "prospective_cluster_ids": sorted(
            scope["lanes"][LANE_HH]["cluster_ids"]
            + scope["lanes"][LANE_HUBBARD]["cluster_ids"]
        ),
        "prospective_job_batch_names": sorted(
            scope["lanes"][LANE_HH]["job_batch_names"]
            + scope["lanes"][LANE_HUBBARD]["job_batch_names"]
        ),
    }
    query_rows = {
        name: {
            "argv": query_contracts[name][0],
            "returncode": query_contracts[name][2],
            "rows": rows[name],
            "stderr": query_contracts[name][3],
            "stdout_sha256": stdout_sha256[name],
        }
        for name in ("queue", "history")
    }
    reconciliation: dict[str, Any] = {}
    for lane in LANES:
        for batch in scope["lanes"][lane]["job_batch_names"]:
            matches = _matching_rows(batch=batch, rows_by_query=rows)
            intent = intents.get(batch)
            outcome: dict[str, Any] = {
                "status": "found" if matches else "indeterminate",
                "match_count": len(matches),
                "matches": matches,
            }
            if intent is not None:
                outcome["intent_identity_sha256"] = intent[
                    "intent_identity_sha256"
                ]
            if not matches and intent is None:
                outcome["indeterminate_reason"] = (
                    "no_durable_intent_identity_bound_to_capture"
                )
            elif not matches:
                prior_outcome = (
                    prior_binding["intent_outcomes"].get(batch)
                    if prior_binding is not None
                    else None
                )
                if (
                    isinstance(prior_outcome, Mapping)
                    and prior_outcome.get("intent_identity_sha256")
                    == intent["intent_identity_sha256"]
                    and prior_outcome.get("status")
                    in {"indeterminate", "verified_absent"}
                ):
                    outcome["status"] = "verified_absent"
                    outcome["absence_proof"] = {
                        "schema": ABSENCE_PROOF_V1,
                        "intent_identity_sha256": intent[
                            "intent_identity_sha256"
                        ],
                        "prior_snapshot_sha256": prior_binding[
                            "snapshot_sha256"
                        ],
                        "prior_snapshot_status": prior_outcome["status"],
                        "prior_captured_at": prior_binding["captured_at"],
                        "prior_captured_at_unix_ns": prior_binding[
                            "captured_at_unix_ns"
                        ],
                    }
                else:
                    outcome["indeterminate_reason"] = (
                        "first_snapshot_for_exact_intent_identity"
                        if prior_outcome is None
                        else "prior_snapshot_does_not_prove_safe_absence"
                    )
            reconciliation[batch] = outcome

    core: dict[str, Any] = {
        "schema": COMBINED_SCOPED_SNAPSHOT_V1,
        "captured_at": captured_at,
        "captured_at_unix_ns": captured_at_unix_ns,
        "single_combined_snapshot": True,
        "polling_performed": False,
        "capture_counts": {"queue": 1, "history": 1},
        "scope": snapshot_scope,
        "queries": query_rows,
        "reconciliation_context": normalized_reconciliation_context,
        "reconciliation": reconciliation,
        "hh_rows": classified[LANE_HH],
        "hubbard_rows": classified[LANE_HUBBARD],
        "siam_rows": classified[LANE_SIAM],
    }
    return {**core, "snapshot_sha256": _digest(core)}


def build_combined_snapshot(
    *,
    queue_stdout: str | Path,
    history_stdout: str | Path,
    capture_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one self-digested snapshot without contacting the scheduler."""

    rows: dict[str, list[dict[str, Any]]] = {}
    stdout_sha256: dict[str, str] = {}
    for name, path in (
        ("queue", Path(queue_stdout)),
        ("history", Path(history_stdout)),
    ):
        raw, rows[name] = _read_rows(path, label=f"{name} captured stdout")
        stdout_sha256[name] = _bytes_sha256(raw)
    return _assemble_snapshot(
        capture_spec,
        rows=rows,
        stdout_sha256=stdout_sha256,
    )


def validate_combined_snapshot(
    snapshot: Mapping[str, Any],
    *,
    queue_stdout: str | Path | None = None,
    history_stdout: str | Path | None = None,
) -> str:
    """Validate a snapshot and optionally rebind it to preserved stdout files."""

    if snapshot.get("schema") != COMBINED_SCOPED_SNAPSHOT_V1:
        raise CombinedSnapshotError("snapshot schema is not supported")
    declared = snapshot.get("snapshot_sha256")
    if not isinstance(declared, str) or len(declared) != 64:
        raise CombinedSnapshotError("snapshot self digest is invalid")
    core = dict(snapshot)
    core.pop("snapshot_sha256", None)
    if declared != _digest(core):
        raise CombinedSnapshotError("snapshot self digest mismatch")
    scope = snapshot.get("scope")
    if not isinstance(scope, Mapping):
        raise CombinedSnapshotError("snapshot scope is missing")
    queries = snapshot.get("queries")
    if not isinstance(queries, Mapping) or set(queries) != {"queue", "history"}:
        raise CombinedSnapshotError("snapshot query set is incomplete")
    rows_by_query: dict[str, list[dict[str, Any]]] = {}
    stdout_sha256: dict[str, str] = {}
    query_spec: dict[str, Any] = {}
    for name in ("queue", "history"):
        query = queries[name]
        if not isinstance(query, Mapping):
            raise CombinedSnapshotError(f"{name} query must be an object")
        digest = query.get("stdout_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise CombinedSnapshotError(f"{name} stdout digest is invalid")
        rows = query.get("rows")
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise CombinedSnapshotError(f"{name} rows must be a list of objects")
        rows_by_query[name] = [dict(row) for row in rows]
        stdout_sha256[name] = digest
        query_spec[name] = {
            "argv": query.get("argv"),
            "returncode": query.get("returncode"),
            "stderr": query.get("stderr"),
        }

    reconstructed = _assemble_snapshot(
        {
            "schema": CAPTURE_SPEC_V1,
            "captured_at": snapshot.get("captured_at"),
            "captured_at_unix_ns": snapshot.get("captured_at_unix_ns"),
            "single_combined_snapshot": snapshot.get("single_combined_snapshot"),
            "polling_performed": snapshot.get("polling_performed"),
            "capture_counts": snapshot.get("capture_counts"),
            "scope": {
                "condition": scope.get("condition"),
                "lanes": scope.get("lanes"),
            },
            "queries": query_spec,
        },
        rows=rows_by_query,
        stdout_sha256=stdout_sha256,
        reconciliation_context=snapshot.get("reconciliation_context"),
    )
    if dict(snapshot) != reconstructed:
        raise CombinedSnapshotError(
            "snapshot fields diverged from the deterministic reconstruction"
        )

    raw_paths = {"queue": queue_stdout, "history": history_stdout}
    if (queue_stdout is None) != (history_stdout is None):
        raise CombinedSnapshotError(
            "queue and history stdout must both be supplied for raw verification"
        )
    if queue_stdout is not None:
        for name in ("queue", "history"):
            raw_path = raw_paths[name]
            assert raw_path is not None
            raw, rows = _read_rows(
                Path(raw_path), label=f"{name} captured stdout"
            )
            if _bytes_sha256(raw) != queries[name]["stdout_sha256"]:
                raise CombinedSnapshotError(f"{name} stdout digest mismatch")
            if rows != rows_by_query[name]:
                raise CombinedSnapshotError(f"{name} stdout rows changed")
    return declared


def write_immutable_snapshot(path: str | Path, snapshot: Mapping[str, Any]) -> str:
    """Atomically write a snapshot, accepting only byte-identical replay."""

    validate_combined_snapshot(snapshot)
    target = Path(path)
    content = _pretty_bytes(snapshot)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink() or not target.is_file():
            raise CombinedSnapshotError(
                f"snapshot output is not a regular file: {target}"
            )
        if target.read_bytes() != content:
            raise CombinedSnapshotError(
                f"refusing to overwrite immutable snapshot: {target}"
            )
        return "unchanged"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = ""
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return "created"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or validate an offline Paper-II combined snapshot."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--queue-json", type=Path, required=True)
    build.add_argument("--history-json", type=Path, required=True)
    build.add_argument("--capture-spec", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    check = subparsers.add_parser("check")
    check.add_argument("--snapshot", type=Path, required=True)
    check.add_argument("--queue-json", type=Path)
    check.add_argument("--history-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            spec = _read_object(args.capture_spec, label="capture specification")
            snapshot = build_combined_snapshot(
                queue_stdout=args.queue_json,
                history_stdout=args.history_json,
                capture_spec=spec,
            )
            status = write_immutable_snapshot(args.output, snapshot)
            result = {
                "output": str(args.output),
                "snapshot_sha256": snapshot["snapshot_sha256"],
                "status": status,
            }
        else:
            snapshot = _read_object(args.snapshot, label="combined snapshot")
            digest = validate_combined_snapshot(
                snapshot,
                queue_stdout=args.queue_json,
                history_stdout=args.history_json,
            )
            result = {
                "snapshot": str(args.snapshot),
                "snapshot_sha256": digest,
                "status": "valid",
            }
    except CombinedSnapshotError as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through ``main``
    raise SystemExit(main())
