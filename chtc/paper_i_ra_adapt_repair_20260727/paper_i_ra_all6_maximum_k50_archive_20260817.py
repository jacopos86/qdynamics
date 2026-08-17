#!/usr/bin/env python3
"""Strict archive adapter for ragged maximum-k50 Paper-I cell evidence.

The adapter keeps reporting evidence small and independently authenticated
while delegating archive construction and tree rotation to the established
singleton-12 strict archive state machine.  Building compact evidence and
preparing an archive never removes or renames a scientific run tree.  The
only public operation permitted to do that is :func:`execute_rotation`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence

from chtc.paper_i_ra_adapt_repair_20260727 import (
    paper_i_matched_singleton12_archive_20260815 as strict_archive,
)
from chtc.paper_i_ra_adapt_repair_20260727 import (
    paper_i_ra_all6_maximum_k50_reporting_20260817 as ragged_reporting,
)
from pipelines.static_adapt.adaptive_phase_contracts import (
    ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1,
)


COMPACT_SCHEMA = "paper_i_ra_all6_maximum_k50_compact_cell_evidence_v1"
CELL_COMPLETION_SCHEMA = (
    "paper_i_ra_all6_adaptive_maximum_k50_cell_completion_v1"
)
AUTHORIZATION_SCHEMA = (
    "paper_i_ra_all6_adaptive_maximum_k50_authorization_v1"
)
AUTHORIZATION_BASIS = (
    "explicit_current_user_maximum_k50_natural_terminal_request"
)
MAXIMUM_CONTROLLER_ROUNDS = 50
NATURAL_TERMINAL_KIND = (
    "authenticated_phase3_no_positive_natural_terminal_v1"
)
MAXIMUM_ROUNDS_KIND = "reached_maximum_controller_rounds_v1"
NATURAL_TERMINAL_OUTCOME = ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
ARCHIVE_CELL_METADATA_SCHEMA = (
    "paper_i_ra_all6_maximum_k50_archive_cell_metadata_v1"
)
PREPARATION_SCHEMA = "paper_i_ra_all6_maximum_k50_archive_preparation_v1"
EXACT_ROTATION_AUTHORIZATION_FLAG = (
    "execute_authenticated_maximum_k50_exact_tree_rotation_v1"
)
ARCHIVE_BACKED_CELL_SCHEMA = (
    "paper_i_ra_all6_maximum_k50_archive_backed_cell_v1"
)
COMPACT_KEYS = {
    "schema",
    "status",
    "campaign_id",
    "execution_id",
    "cell_metadata",
    "maximum_controller_rounds",
    "completion_kind",
    "accepted_controller_rounds",
    "summary_artifact_status",
    "cell_completion",
    "cell_completion_sha256",
    "accepted_rows",
    "accepted_rows_sha256",
    "terminal_attempt",
    "terminal_attempt_sha256",
    "cell_outcome",
    "cell_outcome_sha256",
    "worker_receipt_binding",
    "guard_receipt_binding",
    "log_file_binding",
    "source_artifact_bindings",
    "source_artifact_bindings_sha256",
    "submission_authorized",
    "paper_adoption_authorized",
    "paper_evidence_adoption_authorized",
    "sha256",
}
IDENTITY_FIELDS = (
    "execution_id",
    "cell_ordinal",
    "block",
    "regime_id",
    "nph",
    "insertion_policy",
)
ACCEPTED_ROW_FIELDS = tuple(ragged_reporting.ACCEPTED_ROW_FIELDS)
TERMINAL_ATTEMPT_FIELDS = tuple(ragged_reporting.TERMINAL_ATTEMPT_FIELDS)


class MaximumK50ArchiveError(ValueError):
    """Raised when compact or archive-backed cell evidence drifts."""


def canonical_json_bytes(payload: Any) -> bytes:
    """Return the compact canonical JSON representation used for bindings."""

    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise MaximumK50ArchiveError(
            "Evidence is not finite canonical JSON."
        ) from exc


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    if "sha256" in result:
        raise MaximumK50ArchiveError("Digest input already contains sha256.")
    result["sha256"] = canonical_sha256(result)
    return result


def _normalized_mapping(
    value: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MaximumK50ArchiveError(f"{label} is not a mapping.")
    try:
        result = json.loads(canonical_json_bytes(dict(value)))
    except json.JSONDecodeError as exc:  # pragma: no cover - canonical encoder
        raise MaximumK50ArchiveError(f"{label} is malformed.") from exc
    if not isinstance(result, dict):
        raise MaximumK50ArchiveError(f"{label} is not an object.")
    return result


def _require_digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise MaximumK50ArchiveError(f"{label} is not a lowercase SHA-256.")
    return value


def _validate_digested_mapping(
    value: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    result = _normalized_mapping(value, label=label)
    observed = result.pop("sha256", None)
    if observed != canonical_sha256(result):
        raise MaximumK50ArchiveError(f"{label} self digest drifted.")
    result["sha256"] = _require_digest(observed, label=f"{label} digest")
    return result


def _require_nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MaximumK50ArchiveError(f"{label} is not a nonnegative integer.")
    return value


def _reporting_identity(
    cell_metadata: Mapping[str, Any], *, execution_id: str
) -> dict[str, Any]:
    ordinal = cell_metadata.get(
        "cell_ordinal", cell_metadata.get("ordinal")
    )
    identity = {
        "execution_id": execution_id,
        "cell_ordinal": ordinal,
        "block": cell_metadata.get("block"),
        "regime_id": cell_metadata.get("regime_id"),
        "nph": cell_metadata.get("nph"),
        "insertion_policy": cell_metadata.get("insertion_policy"),
    }
    if (
        isinstance(ordinal, bool)
        or not isinstance(ordinal, int)
        or ordinal < 1
        or identity["block"] not in {"append", "plateau"}
        or not isinstance(identity["regime_id"], str)
        or not identity["regime_id"]
        or isinstance(identity["nph"], bool)
        or not isinstance(identity["nph"], int)
        or identity["nph"] < 1
        or identity["insertion_policy"]
        != (
            "append_only"
            if identity["block"] == "append"
            else "plateau_commutation"
        )
    ):
        raise MaximumK50ArchiveError("Reporting cell identity drifted.")
    return identity


def _validate_count_chain(row: Mapping[str, Any], *, label: str) -> None:
    names = (
        "phase0_population_count",
        "phase0_retained_count",
        "phase_i_input_count",
        "phase_i_retained_count",
        "phase_ii_input_count",
        "phase_ii_retained_count",
        "phase_iii_input_count",
        "phase_iii_adaptive_retained_count",
        "phase_iii_final_singleton_count",
    )
    counts = {
        name: _require_nonnegative_int(row.get(name), label=f"{label}.{name}")
        for name in names
    }
    if not (
        counts["phase0_retained_count"] <= counts["phase0_population_count"]
        and counts["phase_i_input_count"] == counts["phase0_retained_count"]
        and counts["phase_i_retained_count"] <= counts["phase_i_input_count"]
        and counts["phase_ii_input_count"] == counts["phase_i_retained_count"]
        and counts["phase_ii_retained_count"] <= counts["phase_ii_input_count"]
        and counts["phase_iii_input_count"] == counts["phase_ii_retained_count"]
        and counts["phase_iii_adaptive_retained_count"]
        <= counts["phase_iii_input_count"]
        and counts["phase_iii_final_singleton_count"]
        <= counts["phase_iii_adaptive_retained_count"]
    ):
        raise MaximumK50ArchiveError(f"{label} phase counts drifted.")


def _validate_placement(
    value: Any, *, identity: Mapping[str, Any], label: str
) -> None:
    if identity["block"] == "append":
        valid = value == "append_only"
    else:
        valid = value in {"open", "closed"}
    if not valid:
        raise MaximumK50ArchiveError(f"{label} placement state drifted.")


def _validate_accepted_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    normalized = [dict(row) for row in rows]
    for controller_round, row in enumerate(normalized, start=1):
        if (
            set(row) != set(ACCEPTED_ROW_FIELDS)
            or {field: row.get(field) for field in IDENTITY_FIELDS}
            != dict(identity)
            or row.get("controller_round") != controller_round
        ):
            raise MaximumK50ArchiveError("Accepted reporting row drifted.")
        for name in ("energy", "absolute_delta_e"):
            value = row.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise MaximumK50ArchiveError(
                    f"Accepted reporting row {name} drifted."
                )
        _validate_placement(
            row.get("placement_state"),
            identity=identity,
            label="accepted reporting row",
        )
        _validate_count_chain(row, label="accepted reporting row")
        if row.get("phase_iii_final_singleton_count") != 1:
            raise MaximumK50ArchiveError(
                "Accepted reporting row lacks its Phase-III singleton."
            )
        for name in (
            "phase_iii_final_record_id",
            "selected_generator",
            "selected_operator",
        ):
            if not isinstance(row.get(name), str) or not row[name]:
                raise MaximumK50ArchiveError(
                    f"Accepted reporting row {name} is absent."
                )
        for name in ("selected_position", "s_alg", "n2q", "d2q", "dc"):
            _require_nonnegative_int(
                row.get(name), label=f"accepted reporting row {name}"
            )
        _require_digest(
            row.get("checkpoint_sha256"),
            label="accepted reporting checkpoint",
        )
    return normalized


def _validate_terminal_attempt(
    terminal: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    accepted_controller_rounds: int,
    completion: Mapping[str, Any],
) -> dict[str, Any]:
    row = dict(terminal)
    if (
        set(row) != {*TERMINAL_ATTEMPT_FIELDS, "sha256"}
        or {field: row.get(field) for field in IDENTITY_FIELDS}
        != dict(identity)
        or row.get("attempted_controller_round")
        != accepted_controller_rounds + 1
        or row.get("terminal_controller_outcome")
        != NATURAL_TERMINAL_OUTCOME
    ):
        raise MaximumK50ArchiveError("Terminal reporting attempt drifted.")
    _validate_placement(
        row.get("placement_state"),
        identity=identity,
        label="terminal reporting attempt",
    )
    _validate_count_chain(row, label="terminal reporting attempt")
    if row.get("phase_iii_final_singleton_count") != 0:
        raise MaximumK50ArchiveError(
            "Terminal reporting attempt retained a singleton."
        )
    for name in (
        "terminal_phase3_selection_receipt_sha256",
        "terminal_active_prefix_checkpoint_sha256",
    ):
        _require_digest(row.get(name), label=f"terminal attempt {name}")
        if row.get(name) != completion.get(name):
            raise MaximumK50ArchiveError(
                f"Terminal attempt {name} detached from completion."
            )
    return row


def _plain_file_binding(path: Path, *, root: Path, label: str) -> dict[str, Any]:
    path = Path(path).absolute()
    root = Path(root).absolute()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise MaximumK50ArchiveError(f"{label} escaped its evidence root.") from exc
    if not relative or relative.startswith("../"):
        raise MaximumK50ArchiveError(f"{label} has an unsafe relative path.")
    try:
        observed = path.lstat()
    except FileNotFoundError as exc:
        raise MaximumK50ArchiveError(f"{label} is absent: {path}") from exc
    if not stat.S_ISREG(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
        raise MaximumK50ArchiveError(f"{label} is not a plain file: {path}")
    digest = hashlib.sha256()
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != observed.st_dev
            or opened.st_ino != observed.st_ino
            or opened.st_size != observed.st_size
        ):
            raise MaximumK50ArchiveError(f"{label} changed while opening.")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while block := stream.read(8 * 1024 * 1024):
                digest.update(block)
        final = os.fstat(descriptor)
        if (
            final.st_dev != opened.st_dev
            or final.st_ino != opened.st_ino
            or final.st_size != opened.st_size
            or final.st_mtime_ns != opened.st_mtime_ns
        ):
            raise MaximumK50ArchiveError(f"{label} changed while hashing.")
    finally:
        os.close(descriptor)
    return {
        "mode": 0o755 if opened.st_mode & 0o111 else 0o644,
        "path": relative,
        "sha256": digest.hexdigest(),
        "size_bytes": opened.st_size,
    }


def build_compact_payload(
    *,
    runtime_root: Path,
    campaign_id: str,
    execution_id: str,
    cell_metadata: Mapping[str, Any],
    cell_completion: Mapping[str, Any],
    accepted_rows: Sequence[Mapping[str, Any]],
    terminal_attempt: Mapping[str, Any] | None,
    cell_outcome: Mapping[str, Any],
    worker_receipt_path: Path,
    guard_receipt_path: Path,
    log_path: Path,
    source_artifact_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Build a deterministic compact projection for one completed cell."""

    runtime = Path(runtime_root).absolute()
    run_root = runtime / "runs" / execution_id
    if not isinstance(campaign_id, str) or not campaign_id:
        raise MaximumK50ArchiveError("Campaign ID is absent.")
    if (
        not isinstance(execution_id, str)
        or not execution_id
        or "/" in execution_id
        or "\\" in execution_id
    ):
        raise MaximumK50ArchiveError("Execution ID is unsafe.")
    cell = _normalized_mapping(cell_metadata, label="cell metadata")
    if cell.get("execution_id") != execution_id:
        raise MaximumK50ArchiveError("Cell metadata execution ID drifted.")
    identity = _reporting_identity(cell, execution_id=execution_id)
    completion = _validate_digested_mapping(
        cell_completion, label="cell completion"
    )
    if (
        completion.get("schema") != CELL_COMPLETION_SCHEMA
        or completion.get("campaign_id") != campaign_id
        or completion.get("execution_id") != execution_id
    ):
        raise MaximumK50ArchiveError("Cell completion identity drifted.")
    outcome = _validate_digested_mapping(cell_outcome, label="cell outcome")
    rows = [
        _normalized_mapping(row, label=f"accepted row {index}")
        for index, row in enumerate(accepted_rows, start=1)
    ]
    accepted = _require_nonnegative_int(
        completion.get("accepted_controller_rounds"),
        label="accepted controller rounds",
    )
    rows = _validate_accepted_rows(rows, identity=identity)
    maximum = _require_nonnegative_int(
        completion.get("maximum_controller_rounds"),
        label="maximum controller rounds",
    )
    if maximum != MAXIMUM_CONTROLLER_ROUNDS or accepted > maximum:
        raise MaximumK50ArchiveError("Cell completion horizon drifted.")
    if [row.get("controller_round") for row in rows] != list(
        range(1, accepted + 1)
    ):
        raise MaximumK50ArchiveError(
            "Accepted rows are not the exact ordered accepted prefix."
        )
    if any(
        row.get("execution_id") not in {None, execution_id} for row in rows
    ):
        raise MaximumK50ArchiveError("Accepted row execution ID drifted.")

    kind = completion.get("completion_kind")
    terminal: dict[str, Any] | None
    if kind == NATURAL_TERMINAL_KIND:
        if accepted >= MAXIMUM_CONTROLLER_ROUNDS or terminal_attempt is None:
            raise MaximumK50ArchiveError("Natural terminal cardinality drifted.")
        terminal = _validate_digested_mapping(
            terminal_attempt, label="terminal attempt"
        )
        terminal = _validate_terminal_attempt(
            terminal,
            identity=identity,
            accepted_controller_rounds=accepted,
            completion=completion,
        )
        if (
            completion.get("terminal_attempted_controller_round") != accepted + 1
            or completion.get("terminal_controller_outcome")
            != NATURAL_TERMINAL_OUTCOME
            or terminal.get("attempted_controller_round") != accepted + 1
            or terminal.get("terminal_controller_outcome")
            != NATURAL_TERMINAL_OUTCOME
            or terminal.get("execution_id") != execution_id
        ):
            raise MaximumK50ArchiveError("Natural terminal binding drifted.")
    elif kind == MAXIMUM_ROUNDS_KIND:
        if (
            accepted != MAXIMUM_CONTROLLER_ROUNDS
            or terminal_attempt is not None
            or completion.get("terminal_attempted_controller_round") is not None
            or completion.get("terminal_controller_outcome") is not None
        ):
            raise MaximumK50ArchiveError("Maximum-round completion drifted.")
        terminal = None
    else:
        raise MaximumK50ArchiveError("Unknown maximum-k50 completion kind.")

    if (
        outcome.get("execution_id") != execution_id
        or outcome.get("completion_kind") != kind
        or outcome.get("accepted_controller_rounds") != accepted
    ):
        raise MaximumK50ArchiveError("Cell-outcome binding drifted.")

    summary_status = completion.get("summary_artifact_status")
    expected_roles = {"checkpoint", "estimator_ledger", "result"}
    if accepted == 0:
        if summary_status != "not_applicable_round_zero":
            raise MaximumK50ArchiveError("Round-zero summary status drifted.")
    else:
        expected_roles.add("summary")
        if summary_status != "present":
            raise MaximumK50ArchiveError("Accepted cell is missing summary status.")
    if set(source_artifact_paths) != expected_roles:
        raise MaximumK50ArchiveError("Source artifact roles drifted.")
    sources = {
        role: _plain_file_binding(
            Path(source_artifact_paths[role]),
            root=run_root,
            label=f"{role} source artifact",
        )
        for role in sorted(expected_roles)
    }
    checkpoint_sha256 = completion.get("checkpoint_file_sha256")
    if checkpoint_sha256 != sources["checkpoint"]["sha256"]:
        raise MaximumK50ArchiveError("Completion checkpoint binding drifted.")
    if accepted == 0 and completion.get("paper_i_summary_sha256") is not None:
        raise MaximumK50ArchiveError("Round-zero completion binds a summary.")
    if (
        accepted > 0
        and completion.get("paper_i_summary_sha256")
        != sources["summary"]["sha256"]
    ):
        raise MaximumK50ArchiveError("Completion summary binding drifted.")

    payload = {
        "schema": COMPACT_SCHEMA,
        "status": "passed_ragged_maximum_k50_reporting_projection",
        "campaign_id": campaign_id,
        "execution_id": execution_id,
        "cell_metadata": cell,
        "maximum_controller_rounds": MAXIMUM_CONTROLLER_ROUNDS,
        "completion_kind": kind,
        "accepted_controller_rounds": accepted,
        "summary_artifact_status": summary_status,
        "cell_completion": completion,
        "cell_completion_sha256": completion["sha256"],
        "accepted_rows": rows,
        "accepted_rows_sha256": canonical_sha256(rows),
        "terminal_attempt": terminal,
        "terminal_attempt_sha256": terminal["sha256"] if terminal else None,
        "cell_outcome": outcome,
        "cell_outcome_sha256": outcome["sha256"],
        "worker_receipt_binding": _plain_file_binding(
            worker_receipt_path, root=runtime, label="worker receipt"
        ),
        "guard_receipt_binding": _plain_file_binding(
            guard_receipt_path, root=runtime, label="guard receipt"
        ),
        "log_file_binding": _plain_file_binding(
            log_path, root=runtime, label="cell log"
        ),
        "source_artifact_bindings": sources,
        "source_artifact_bindings_sha256": canonical_sha256(sources),
        "submission_authorized": False,
        "paper_adoption_authorized": False,
        "paper_evidence_adoption_authorized": False,
    }
    return digested(payload)


def _source_paths_from_compact(
    payload: Mapping[str, Any], *, run_root: Path
) -> dict[str, Path]:
    bindings = payload.get("source_artifact_bindings")
    if not isinstance(bindings, Mapping):
        raise MaximumK50ArchiveError("Source artifact bindings are absent.")
    result: dict[str, Path] = {}
    absolute_root = Path(run_root).absolute()
    for role, raw in bindings.items():
        if not isinstance(role, str) or not isinstance(raw, Mapping):
            raise MaximumK50ArchiveError("Source artifact binding is malformed.")
        relative = raw.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or "\\" in relative
        ):
            raise MaximumK50ArchiveError("Source artifact path is unsafe.")
        resolved = (absolute_root / relative).absolute()
        try:
            resolved.relative_to(absolute_root)
        except ValueError as exc:
            raise MaximumK50ArchiveError(
                "Source artifact path escaped the run tree."
            ) from exc
        result[role] = resolved
    return result


def validate_compact_payload(
    payload: Mapping[str, Any],
    *,
    runtime_root: Path,
    campaign_id: str,
    execution_id: str,
    cell_metadata: Mapping[str, Any],
    worker_receipt_path: Path,
    guard_receipt_path: Path,
    log_path: Path,
    require_live_source_artifacts: bool,
) -> dict[str, Any]:
    """Validate a compact projection and, when requested, its live files."""

    observed = _normalized_mapping(payload, label="compact payload")
    if require_live_source_artifacts is not True:
        raise MaximumK50ArchiveError(
            "Archive-backed compact validation requires an archive manifest."
        )
    completion = observed.get("cell_completion")
    rows = observed.get("accepted_rows")
    outcome = observed.get("cell_outcome")
    terminal = observed.get("terminal_attempt")
    if (
        not isinstance(completion, Mapping)
        or not isinstance(rows, list)
        or any(not isinstance(row, Mapping) for row in rows)
        or not isinstance(outcome, Mapping)
        or (terminal is not None and not isinstance(terminal, Mapping))
    ):
        raise MaximumK50ArchiveError("Compact projection content is malformed.")
    rebuilt = build_compact_payload(
        runtime_root=runtime_root,
        campaign_id=campaign_id,
        execution_id=execution_id,
        cell_metadata=cell_metadata,
        cell_completion=completion,
        accepted_rows=rows,
        terminal_attempt=terminal,
        cell_outcome=outcome,
        worker_receipt_path=worker_receipt_path,
        guard_receipt_path=guard_receipt_path,
        log_path=log_path,
        source_artifact_paths=_source_paths_from_compact(
            observed,
            run_root=Path(runtime_root) / "runs" / execution_id,
        ),
    )
    if rebuilt != observed:
        raise MaximumK50ArchiveError("Compact payload binding drifted.")
    return observed


def _load_compact_file(path: Path) -> dict[str, Any]:
    binding = _plain_file_binding(path, root=Path(path).parent, label="compact payload")
    if binding["size_bytes"] > 64 * 1024 * 1024:
        raise MaximumK50ArchiveError("Compact payload exceeds its fixed limit.")
    raw = Path(path).read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaximumK50ArchiveError("Compact payload JSON is malformed.") from exc
    if not isinstance(value, dict):
        raise MaximumK50ArchiveError("Compact payload is not a JSON object.")
    if raw != canonical_json_bytes(value) + b"\n":
        raise MaximumK50ArchiveError("Compact payload is not canonical JSON.")
    return value


def _archive_cell_metadata(compact: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact metadata authenticated inside the strict archive."""

    return {
        "schema": ARCHIVE_CELL_METADATA_SCHEMA,
        "campaign_id": compact["campaign_id"],
        "execution_id": compact["execution_id"],
        "cell_metadata": compact["cell_metadata"],
        "maximum_controller_rounds": compact["maximum_controller_rounds"],
        "completion_kind": compact["completion_kind"],
        "accepted_controller_rounds": compact["accepted_controller_rounds"],
        "summary_artifact_status": compact["summary_artifact_status"],
        "cell_completion_sha256": compact["cell_completion_sha256"],
        "accepted_rows_sha256": compact["accepted_rows_sha256"],
        "terminal_attempt_sha256": compact["terminal_attempt_sha256"],
        "cell_outcome_sha256": compact["cell_outcome_sha256"],
        "worker_receipt_binding": compact["worker_receipt_binding"],
        "guard_receipt_binding": compact["guard_receipt_binding"],
        "log_file_binding": compact["log_file_binding"],
        "source_artifact_bindings": compact["source_artifact_bindings"],
        "source_artifact_bindings_sha256": compact[
            "source_artifact_bindings_sha256"
        ],
        "compact_payload_sha256": compact["sha256"],
    }


def _validate_authority(
    authority_metadata: Mapping[str, Any], *, campaign_id: str
) -> dict[str, Any]:
    authority = _validate_digested_mapping(
        authority_metadata, label="archive authority metadata"
    )
    if (
        authority.get("schema") != AUTHORIZATION_SCHEMA
        or authority.get("campaign_id") != campaign_id
        or authority.get("authorization_basis") != AUTHORIZATION_BASIS
        or authority.get("execution_authorized") is not True
        or authority.get("submission_authorized") is not False
        or authority.get("paper_adoption_authorized") is not False
        or authority.get("paper_evidence_adoption_authorized") is not False
    ):
        raise MaximumK50ArchiveError("Archive authority binding drifted.")
    return authority


def _validate_rotation_authority(
    rotation_authority: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
    campaign_id: str,
) -> dict[str, Any]:
    rotation = _validate_authority(
        rotation_authority, campaign_id=campaign_id
    )
    if (
        rotation != dict(authority)
        or rotation.get("archive_rotation_authorized") is not True
    ):
        raise MaximumK50ArchiveError("Rotation authority binding drifted.")
    return rotation


def _external_members(
    *,
    compact_path: Path,
    worker_receipt_path: Path,
    guard_receipt_path: Path,
    log_path: Path,
) -> dict[str, Path]:
    return {
        "evidence/cell.log": Path(log_path),
        "evidence/compact_cell_evidence.json": Path(compact_path),
        "evidence/guard_receipt.json": Path(guard_receipt_path),
        "evidence/worker_receipt.json": Path(worker_receipt_path),
    }


def prepare_archive(
    *,
    runtime_root: Path,
    campaign_id: str,
    execution_id: str,
    cell_metadata: Mapping[str, Any],
    authority_metadata: Mapping[str, Any],
    compact_path: Path,
    worker_receipt_path: Path,
    guard_receipt_path: Path,
    log_path: Path,
    limits: strict_archive.ArchiveLimits,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build and byte-close an archive without authorizing tree rotation."""

    compact = validate_compact_payload(
        _load_compact_file(compact_path),
        runtime_root=runtime_root,
        campaign_id=campaign_id,
        execution_id=execution_id,
        cell_metadata=cell_metadata,
        worker_receipt_path=worker_receipt_path,
        guard_receipt_path=guard_receipt_path,
        log_path=log_path,
        require_live_source_artifacts=True,
    )
    authority = _validate_authority(
        authority_metadata, campaign_id=campaign_id
    )
    paths = strict_archive.CellArchivePaths(runtime_root, execution_id)
    prefix = f"runs/{execution_id}"
    metadata = _archive_cell_metadata(compact)
    try:
        state = strict_archive.inspect_rotation_state(paths)
        if state["stale_archive_temporaries"]:
            if state["state"] not in {
                "direct_unarchived",
                "archive_published_pending_manifest",
                "manifest_published_pending_closure",
            }:
                raise MaximumK50ArchiveError(
                    "Archive preparation found unsafe archive temporaries."
                )
            removed = strict_archive.discard_stale_archive_temporaries(paths)
            if removed != state["stale_archive_temporaries"]:
                raise MaximumK50ArchiveError(
                    "Archive temporary disposal did not match inspection."
                )
            state = strict_archive.inspect_rotation_state(paths)
            if state["stale_archive_temporaries"]:
                raise MaximumK50ArchiveError(
                    "Archive preparation left unresolved temporaries."
                )
        if state["state"] not in {
            "direct_unarchived",
            "archive_published_pending_manifest",
            "manifest_published_pending_closure",
        }:
            raise MaximumK50ArchiveError(
                "Archive preparation requires a direct completed tree."
            )
        validation = strict_archive.build_cell_archive(
            paths=paths,
            source_member_prefix=prefix,
            external_members=_external_members(
                compact_path=compact_path,
                worker_receipt_path=worker_receipt_path,
                guard_receipt_path=guard_receipt_path,
                log_path=log_path,
            ),
            authority_metadata=authority,
            cell_metadata=metadata,
            limits=limits,
        )
        closure = strict_archive.publish_archive_closure(
            paths=paths,
            source_member_prefix=prefix,
            authority_metadata=authority,
            cell_metadata=metadata,
            limits=limits,
            created_at_utc=created_at_utc,
        )
        _validate_manifest_crossbindings(
            manifest=_load_archive_manifest(paths.archive_manifest_path),
            compact=compact,
            execution_id=execution_id,
            compact_path=compact_path,
            worker_receipt_path=worker_receipt_path,
            guard_receipt_path=guard_receipt_path,
            log_path=log_path,
        )
        final_state = strict_archive.inspect_rotation_state(paths)
    except strict_archive.Singleton12ArchiveError as exc:
        raise MaximumK50ArchiveError(
            "Strict archive preparation failed."
        ) from exc
    if final_state["state"] != "closure_published_pending_intent":
        raise MaximumK50ArchiveError("Archive preparation state drifted.")
    return digested(
        {
            "schema": PREPARATION_SCHEMA,
            "status": "passed_archive_preparation_no_rotation",
            "campaign_id": campaign_id,
            "execution_id": execution_id,
            "compact_payload_sha256": compact["sha256"],
            "archive_validation_sha256": validation["sha256"],
            "archive_closure_sha256": closure["sha256"],
            "strict_state": final_state["state"],
            "direct_source_present": final_state["source_present"],
            "rotation_intent_absent": not final_state[
                "rotation_intent_present"
            ],
        }
    )


def archive_restart_action(
    observed_state: Mapping[str, Any],
    *,
    exact_authorization_flag: str | None,
) -> str:
    """Map every legal strict state to the adapter's sole safe next action."""

    if not isinstance(observed_state, Mapping):
        raise MaximumK50ArchiveError("Strict archive state is malformed.")
    stale = observed_state.get("stale_archive_temporaries")
    if (
        not isinstance(stale, list)
        or any(not isinstance(name, str) for name in stale)
    ):
        raise MaximumK50ArchiveError("Strict archive state is malformed.")
    state = observed_state.get("state")
    actions = {
        "empty": "await_completed_cell",
        "direct_unarchived": "prepare_archive",
        "archive_published_pending_manifest": "prepare_archive",
        "manifest_published_pending_closure": "prepare_archive",
        "closure_published_pending_intent": "execute_rotation",
        "intent_published_pending_rename": "execute_rotation",
        "retiring_pending_removal": "execute_rotation",
        "cleanup_receipt_pending": "execute_rotation",
        "archived_closed": "load_archive_backed_cell",
    }
    if not isinstance(state, str) or state not in actions:
        raise MaximumK50ArchiveError("Unknown strict archive restart state.")
    if stale and state not in {
        "direct_unarchived",
        "archive_published_pending_manifest",
        "manifest_published_pending_closure",
    }:
        raise MaximumK50ArchiveError(
            "Strict archive state has unsafe unresolved temporaries."
        )
    action = actions[state]
    if (
        action == "execute_rotation"
        and exact_authorization_flag != EXACT_ROTATION_AUTHORIZATION_FLAG
    ):
        return "blocked_missing_exact_rotation_authority"
    return action


def execute_rotation(
    *,
    runtime_root: Path,
    campaign_id: str,
    execution_id: str,
    cell_metadata: Mapping[str, Any],
    authority_metadata: Mapping[str, Any],
    rotation_authority: Mapping[str, Any],
    exact_authorization_flag: str,
    compact_path: Path,
    worker_receipt_path: Path,
    guard_receipt_path: Path,
    log_path: Path,
    limits: strict_archive.ArchiveLimits,
    created_at_utc: str | None = None,
    completed_at_utc: str | None = None,
) -> dict[str, Any]:
    """Execute the sole destructive operation after an exact literal opt-in."""

    if exact_authorization_flag != EXACT_ROTATION_AUTHORIZATION_FLAG:
        raise MaximumK50ArchiveError(
            "Tree rotation lacks the exact authorization flag."
        )
    paths = strict_archive.CellArchivePaths(runtime_root, execution_id)
    state = strict_archive.inspect_rotation_state(paths)
    if state["stale_archive_temporaries"]:
        raise MaximumK50ArchiveError(
            "Authorized rotation found unresolved archive temporaries."
        )
    compact_raw = _load_compact_file(compact_path)
    if state["source_present"]:
        compact = validate_compact_payload(
            compact_raw,
            runtime_root=runtime_root,
            campaign_id=campaign_id,
            execution_id=execution_id,
            cell_metadata=cell_metadata,
            worker_receipt_path=worker_receipt_path,
            guard_receipt_path=guard_receipt_path,
            log_path=log_path,
            require_live_source_artifacts=True,
        )
    else:
        compact = _validate_compact_structure(
            compact_raw,
            campaign_id=campaign_id,
            execution_id=execution_id,
            cell_metadata=cell_metadata,
        )
    _validate_persistent_bindings(
        compact=compact,
        runtime_root=runtime_root,
        worker_receipt_path=worker_receipt_path,
        guard_receipt_path=guard_receipt_path,
        log_path=log_path,
    )
    authority = _validate_authority(
        authority_metadata, campaign_id=campaign_id
    )
    rotation = _validate_rotation_authority(
        rotation_authority,
        authority=authority,
        campaign_id=campaign_id,
    )
    prefix = f"runs/{execution_id}"
    metadata = _archive_cell_metadata(compact)
    _validate_manifest_crossbindings(
        manifest=_load_archive_manifest(paths.archive_manifest_path),
        compact=compact,
        execution_id=execution_id,
        compact_path=compact_path,
        worker_receipt_path=worker_receipt_path,
        guard_receipt_path=guard_receipt_path,
        log_path=log_path,
    )
    try:
        state_name = state["state"]
        if state_name == "closure_published_pending_intent":
            strict_archive.publish_rotation_intent(
                paths=paths,
                source_member_prefix=prefix,
                authority_metadata=authority,
                cell_metadata=metadata,
                rotation_authority=rotation,
                limits=limits,
                created_at_utc=created_at_utc,
            )
            state_name = "intent_published_pending_rename"
        if state_name not in {
            "intent_published_pending_rename",
            "retiring_pending_removal",
            "cleanup_receipt_pending",
            "archived_closed",
        }:
            raise MaximumK50ArchiveError(
                "Authorized rotation requires a prepared archive closure."
            )
        if state_name != "archived_closed":
            strict_archive.complete_safe_tree_rotation(
                paths=paths,
                source_member_prefix=prefix,
                authority_metadata=authority,
                cell_metadata=metadata,
                rotation_authority=rotation,
                limits=limits,
                completed_at_utc=completed_at_utc,
            )
    except strict_archive.Singleton12ArchiveError as exc:
        raise MaximumK50ArchiveError(
            "Strict authorized tree rotation failed."
        ) from exc
    return load_archive_backed_cell(
        runtime_root=runtime_root,
        campaign_id=campaign_id,
        execution_id=execution_id,
        cell_metadata=cell_metadata,
        authority_metadata=authority,
        rotation_authority=rotation,
        compact_path=compact_path,
        worker_receipt_path=worker_receipt_path,
        guard_receipt_path=guard_receipt_path,
        log_path=log_path,
        limits=limits,
    )


def _archive_backed_payload(
    *, compact: Mapping[str, Any], closure: Mapping[str, Any]
) -> dict[str, Any]:
    return digested(
        {
            "schema": ARCHIVE_BACKED_CELL_SCHEMA,
            "status": "passed_archive_backed_maximum_k50_cell",
            "campaign_id": compact["campaign_id"],
            "execution_id": compact["execution_id"],
            "cell_metadata": compact["cell_metadata"],
            "completion_kind": compact["completion_kind"],
            "accepted_controller_rounds": compact[
                "accepted_controller_rounds"
            ],
            "summary_artifact_status": compact["summary_artifact_status"],
            "cell_completion": compact["cell_completion"],
            "cell_completion_sha256": compact["cell_completion_sha256"],
            "accepted_rows": compact["accepted_rows"],
            "accepted_rows_sha256": compact["accepted_rows_sha256"],
            "terminal_attempt": compact["terminal_attempt"],
            "terminal_attempt_sha256": compact["terminal_attempt_sha256"],
            "cell_outcome": compact["cell_outcome"],
            "cell_outcome_sha256": compact["cell_outcome_sha256"],
            "compact_payload_sha256": compact["sha256"],
            "archive_backed_closure_sha256": closure["sha256"],
            "archive": closure["archive"],
            "archive_manifest": closure["archive_manifest"],
            "archive_closure": closure["archive_closure"],
            "rotation_intent": closure["rotation_intent"],
            "cleanup_receipt": closure["cleanup_receipt"],
            "direct_run_tree_absent": closure["direct_source_absent"],
            "retiring_tree_absent": closure["retiring_source_absent"],
        }
    )


def _validate_persistent_bindings(
    *,
    compact: Mapping[str, Any],
    runtime_root: Path,
    worker_receipt_path: Path,
    guard_receipt_path: Path,
    log_path: Path,
) -> None:
    runtime = Path(runtime_root).absolute()
    expected = {
        "worker_receipt_binding": _plain_file_binding(
            worker_receipt_path, root=runtime, label="worker receipt"
        ),
        "guard_receipt_binding": _plain_file_binding(
            guard_receipt_path, root=runtime, label="guard receipt"
        ),
        "log_file_binding": _plain_file_binding(
            log_path, root=runtime, label="cell log"
        ),
    }
    if any(compact.get(key) != value for key, value in expected.items()):
        raise MaximumK50ArchiveError("Persistent cell evidence drifted.")


def _validate_binding_shape(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MaximumK50ArchiveError(f"{label} is not a mapping.")
    binding = _normalized_mapping(value, label=label)
    if set(binding) != {"mode", "path", "sha256", "size_bytes"}:
        raise MaximumK50ArchiveError(f"{label} fields drifted.")
    relative = binding.get("path")
    mode = binding.get("mode")
    size = binding.get("size_bytes")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or "\\" in relative
        or isinstance(mode, bool)
        or not isinstance(mode, int)
        or mode < 0
        or mode > 0o7777
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
    ):
        raise MaximumK50ArchiveError(f"{label} is malformed.")
    _require_digest(binding.get("sha256"), label=f"{label} digest")
    return binding


def _validate_compact_structure(
    payload: Mapping[str, Any],
    *,
    campaign_id: str,
    execution_id: str,
    cell_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    compact = _normalized_mapping(payload, label="compact payload")
    if set(compact) != COMPACT_KEYS:
        raise MaximumK50ArchiveError("Compact payload fields drifted.")
    unsigned = dict(compact)
    observed_sha256 = unsigned.pop("sha256", None)
    if observed_sha256 != canonical_sha256(unsigned):
        raise MaximumK50ArchiveError("Compact payload self digest drifted.")
    expected_cell = _normalized_mapping(cell_metadata, label="cell metadata")
    identity = _reporting_identity(expected_cell, execution_id=execution_id)
    if (
        compact.get("schema") != COMPACT_SCHEMA
        or compact.get("status")
        != "passed_ragged_maximum_k50_reporting_projection"
        or compact.get("campaign_id") != campaign_id
        or compact.get("execution_id") != execution_id
        or compact.get("cell_metadata") != expected_cell
        or compact.get("maximum_controller_rounds")
        != MAXIMUM_CONTROLLER_ROUNDS
        or compact.get("submission_authorized") is not False
        or compact.get("paper_adoption_authorized") is not False
        or compact.get("paper_evidence_adoption_authorized") is not False
    ):
        raise MaximumK50ArchiveError("Compact payload identity drifted.")
    completion_raw = compact.get("cell_completion")
    outcome_raw = compact.get("cell_outcome")
    rows_raw = compact.get("accepted_rows")
    terminal_raw = compact.get("terminal_attempt")
    if (
        not isinstance(completion_raw, Mapping)
        or not isinstance(outcome_raw, Mapping)
        or not isinstance(rows_raw, list)
        or any(not isinstance(row, Mapping) for row in rows_raw)
        or (terminal_raw is not None and not isinstance(terminal_raw, Mapping))
    ):
        raise MaximumK50ArchiveError("Compact scientific projection is malformed.")
    completion = _validate_digested_mapping(
        completion_raw, label="cell completion"
    )
    outcome = _validate_digested_mapping(outcome_raw, label="cell outcome")
    rows = [
        _normalized_mapping(row, label=f"accepted row {index}")
        for index, row in enumerate(rows_raw, start=1)
    ]
    rows = _validate_accepted_rows(rows, identity=identity)
    accepted = _require_nonnegative_int(
        compact.get("accepted_controller_rounds"),
        label="accepted controller rounds",
    )
    kind = compact.get("completion_kind")
    if (
        completion.get("schema") != CELL_COMPLETION_SCHEMA
        or completion.get("campaign_id") != campaign_id
        or completion.get("execution_id") != execution_id
        or completion.get("maximum_controller_rounds")
        != MAXIMUM_CONTROLLER_ROUNDS
        or completion.get("accepted_controller_rounds") != accepted
        or completion.get("completion_kind") != kind
        or compact.get("cell_completion_sha256") != completion["sha256"]
        or compact.get("accepted_rows_sha256") != canonical_sha256(rows)
        or [row.get("controller_round") for row in rows]
        != list(range(1, accepted + 1))
        or any(
            row.get("execution_id") not in {None, execution_id} for row in rows
        )
        or outcome.get("execution_id") != execution_id
        or outcome.get("completion_kind") != kind
        or outcome.get("accepted_controller_rounds") != accepted
        or compact.get("cell_outcome_sha256") != outcome["sha256"]
    ):
        raise MaximumK50ArchiveError("Compact scientific binding drifted.")
    if kind == NATURAL_TERMINAL_KIND:
        if not isinstance(terminal_raw, Mapping):
            raise MaximumK50ArchiveError("Natural terminal attempt is absent.")
        terminal = _validate_digested_mapping(
            terminal_raw, label="terminal attempt"
        )
        terminal = _validate_terminal_attempt(
            terminal,
            identity=identity,
            accepted_controller_rounds=accepted,
            completion=completion,
        )
        if (
            accepted >= MAXIMUM_CONTROLLER_ROUNDS
            or completion.get("terminal_attempted_controller_round")
            != accepted + 1
            or completion.get("terminal_controller_outcome")
            != NATURAL_TERMINAL_OUTCOME
            or terminal.get("execution_id") != execution_id
            or terminal.get("attempted_controller_round") != accepted + 1
            or terminal.get("terminal_controller_outcome")
            != NATURAL_TERMINAL_OUTCOME
            or compact.get("terminal_attempt_sha256") != terminal["sha256"]
        ):
            raise MaximumK50ArchiveError("Compact natural-terminal binding drifted.")
    elif kind == MAXIMUM_ROUNDS_KIND:
        if (
            accepted != MAXIMUM_CONTROLLER_ROUNDS
            or terminal_raw is not None
            or compact.get("terminal_attempt_sha256") is not None
            or completion.get("terminal_attempted_controller_round") is not None
            or completion.get("terminal_controller_outcome") is not None
        ):
            raise MaximumK50ArchiveError("Compact maximum-round binding drifted.")
    else:
        raise MaximumK50ArchiveError("Compact completion kind drifted.")

    summary_status = compact.get("summary_artifact_status")
    if summary_status != completion.get("summary_artifact_status"):
        raise MaximumK50ArchiveError("Compact summary status drifted.")
    sources_raw = compact.get("source_artifact_bindings")
    if not isinstance(sources_raw, Mapping):
        raise MaximumK50ArchiveError("Compact source bindings are absent.")
    sources = {
        str(role): _validate_binding_shape(binding, label=f"{role} binding")
        for role, binding in sources_raw.items()
    }
    expected_roles = {"checkpoint", "estimator_ledger", "result"}
    if accepted == 0:
        if summary_status != "not_applicable_round_zero":
            raise MaximumK50ArchiveError("Round-zero summary status drifted.")
        if completion.get("paper_i_summary_sha256") is not None:
            raise MaximumK50ArchiveError("Round-zero summary digest is present.")
    else:
        expected_roles.add("summary")
        if summary_status != "present":
            raise MaximumK50ArchiveError("Accepted prefix lacks a summary.")
    if (
        set(sources) != expected_roles
        or compact.get("source_artifact_bindings_sha256")
        != canonical_sha256(sources)
        or completion.get("checkpoint_file_sha256")
        != sources["checkpoint"]["sha256"]
        or (
            accepted > 0
            and completion.get("paper_i_summary_sha256")
            != sources["summary"]["sha256"]
        )
    ):
        raise MaximumK50ArchiveError("Compact source artifact binding drifted.")
    for key, label in (
        ("worker_receipt_binding", "worker receipt binding"),
        ("guard_receipt_binding", "guard receipt binding"),
        ("log_file_binding", "log file binding"),
    ):
        _validate_binding_shape(compact.get(key), label=label)
    return compact


def _archive_member_row(name: str, path: Path) -> dict[str, Any]:
    local = _plain_file_binding(path, root=Path(path).parent, label=name)
    return {**local, "path": name}


def _load_archive_manifest(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaximumK50ArchiveError("Archive manifest is malformed.") from exc
    if not isinstance(parsed, dict) or raw != canonical_json_bytes(parsed) + b"\n":
        raise MaximumK50ArchiveError("Archive manifest is not canonical JSON.")
    return _validate_digested_mapping(parsed, label="archive manifest")


def _validate_manifest_crossbindings(
    *,
    manifest: Mapping[str, Any],
    compact: Mapping[str, Any],
    execution_id: str,
    compact_path: Path,
    worker_receipt_path: Path,
    guard_receipt_path: Path,
    log_path: Path,
) -> None:
    expected_external = sorted(
        (
            _archive_member_row("evidence/cell.log", log_path),
            _archive_member_row(
                "evidence/compact_cell_evidence.json", compact_path
            ),
            _archive_member_row(
                "evidence/guard_receipt.json", guard_receipt_path
            ),
            _archive_member_row(
                "evidence/worker_receipt.json", worker_receipt_path
            ),
        ),
        key=lambda row: row["path"],
    )
    if manifest.get("external_members") != expected_external:
        raise MaximumK50ArchiveError("Archived external evidence drifted.")
    source_tree = manifest.get("source_tree")
    if not isinstance(source_tree, Mapping) or not isinstance(
        source_tree.get("files"), list
    ):
        raise MaximumK50ArchiveError("Archived source inventory is absent.")
    rows: dict[str, dict[str, Any]] = {}
    for raw in source_tree["files"]:
        if not isinstance(raw, Mapping):
            raise MaximumK50ArchiveError("Archived source row is malformed.")
        relative = raw.get("path")
        if not isinstance(relative, str) or relative in rows:
            raise MaximumK50ArchiveError("Archived source paths are not unique.")
        rows[relative] = dict(raw)
    sources = compact["source_artifact_bindings"]
    for role, binding in sources.items():
        relative = binding["path"]
        expected = {
            **binding,
            "archive_path": f"runs/{execution_id}/{relative}",
        }
        if rows.get(relative) != expected:
            raise MaximumK50ArchiveError(
                f"Archived {role} source binding drifted."
            )


def load_archive_backed_cell(
    *,
    runtime_root: Path,
    campaign_id: str,
    execution_id: str,
    cell_metadata: Mapping[str, Any],
    authority_metadata: Mapping[str, Any],
    rotation_authority: Mapping[str, Any],
    compact_path: Path,
    worker_receipt_path: Path,
    guard_receipt_path: Path,
    log_path: Path,
    limits: strict_archive.ArchiveLimits,
) -> dict[str, Any]:
    """Validate and load an archived cell without extracting or writing."""

    compact = _validate_compact_structure(
        _load_compact_file(compact_path),
        campaign_id=campaign_id,
        execution_id=execution_id,
        cell_metadata=cell_metadata,
    )
    runtime = Path(runtime_root).absolute()
    _validate_persistent_bindings(
        compact=compact,
        runtime_root=runtime,
        worker_receipt_path=worker_receipt_path,
        guard_receipt_path=guard_receipt_path,
        log_path=log_path,
    )
    authority = _validate_authority(
        authority_metadata, campaign_id=campaign_id
    )
    rotation = _validate_rotation_authority(
        rotation_authority,
        authority=authority,
        campaign_id=campaign_id,
    )
    paths = strict_archive.CellArchivePaths(runtime, execution_id)
    if strict_archive.inspect_rotation_state(paths)["state"] != "archived_closed":
        raise MaximumK50ArchiveError("Cell is not archive closed.")
    try:
        closure = strict_archive.validate_archive_backed_closure(
            paths=paths,
            source_member_prefix=f"runs/{execution_id}",
            expected_authority_metadata=authority,
            expected_cell_metadata=_archive_cell_metadata(compact),
            limits=limits,
            expected_rotation_authority=rotation,
            require_cleanup=True,
        )
    except strict_archive.Singleton12ArchiveError as exc:
        raise MaximumK50ArchiveError(
            "Archive-backed strict closure validation failed."
        ) from exc
    manifest = _load_archive_manifest(paths.archive_manifest_path)
    _validate_manifest_crossbindings(
        manifest=manifest,
        compact=compact,
        execution_id=execution_id,
        compact_path=compact_path,
        worker_receipt_path=worker_receipt_path,
        guard_receipt_path=guard_receipt_path,
        log_path=log_path,
    )
    return _archive_backed_payload(compact=compact, closure=closure)


# Re-export the strict helper's bounded public types for runner integration.
ArchiveLimits = strict_archive.ArchiveLimits
CellArchivePaths = strict_archive.CellArchivePaths
