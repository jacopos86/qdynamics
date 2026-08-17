"""Neutral accepted-prefix current-checkpoint publication.

This module owns the durable checkpoint envelope and its authenticated
estimator-ledger, singleton-resume, greedy-batch, and combinatorial-batch
sidecars.  The private compatibility module temporarily re-exports the
publisher while remaining callers migrate to this neutral seam.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipelines.static_adapt.adaptive_phase_contracts import (
    ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1,
)


_CONTENT_ADDRESSED_SIDECAR_FIELDS = {
    "estimator_call_ledger_checkpoint": "estimator_call_ledger_checkpoint",
    "verified_singleton_resume_sidecar": "verified_singleton_resume",
    "greedy_batch_checkpoint_sidecar": "greedy_batch_checkpoint",
    "combinatorial_batch_checkpoint_sidecar": (
        "combinatorial_batch_checkpoint"
    ),
}
_LOGGER = logging.getLogger(__name__)


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"SR-SNAKE compatibility projection is missing {name}.")
    return value


def _require_sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RuntimeError(f"SR-SNAKE compatibility projection is missing {name}.")
    return value


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Durably publish one file without exposing a partial destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _warn_retention_failure(message: str) -> None:
    _LOGGER.warning(
        "Current-checkpoint sidecar retention skipped: %s",
        message,
    )


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_temporary(
    path: Path,
    payload: Mapping[str, Any],
) -> Path:
    """Stream one deterministic pretty JSON document into its target directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            json.dump(
                payload,
                handle,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _publish_temporary(path: Path, temporary_path: Path) -> None:
    try:
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary_path = _write_json_temporary(path, payload)
    _publish_temporary(path, temporary_path)


def _atomic_write_content_addressed_json(
    current_path: Path,
    *,
    filename_role: str,
    payload: Mapping[str, Any],
) -> tuple[Path, str]:
    temporary_path = _write_json_temporary(current_path, payload)
    try:
        sha256 = _sha256_file(temporary_path)
        destination = current_path.with_name(
            f"{current_path.stem}.{filename_role}.{sha256[:16]}.json"
        )
        _publish_temporary(destination, temporary_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination, sha256


def _content_addressed_sidecar_reference(
    pointer: Any,
    *,
    pointer_field: str,
    filename_role: str,
    current_path: Path,
) -> tuple[Path, str] | None:
    if not isinstance(pointer, Mapping):
        _warn_retention_failure(
            f"checkpoint pointer {pointer_field!r} is not a mapping"
        )
        return None
    filename = str(pointer.get("path", ""))
    sha256 = str(pointer.get("sha256", "")).lower()
    prefix = f"{current_path.stem}.{filename_role}."
    digest_prefix = (
        filename[len(prefix) : -len(".json")]
        if filename.startswith(prefix) and filename.endswith(".json")
        else ""
    )
    if (
        not filename
        or Path(filename).name != filename
        or len(digest_prefix) != 16
        or any(
            character not in "0123456789abcdef"
            for character in digest_prefix
        )
        or not _is_sha256(sha256)
        or not sha256.startswith(digest_prefix)
    ):
        _warn_retention_failure(
            f"checkpoint pointer {pointer_field!r} is not a valid "
            "content-addressed sidecar reference"
        )
        return None
    return current_path.with_name(filename), sha256


def _content_addressed_sidecar_references(
    payload: Mapping[str, Any],
    *,
    current_path: Path,
) -> dict[Path, str]:
    adapt = payload.get("adapt_vqe")
    if not isinstance(adapt, Mapping):
        return {}
    references: dict[Path, str] = {}
    ledger_field = "estimator_call_ledger_checkpoint"
    checkpoint = payload.get("checkpoint")
    checkpoint_pointer = (
        checkpoint.get(ledger_field)
        if isinstance(checkpoint, Mapping)
        else None
    )
    adapt_pointer = adapt.get(ledger_field)
    if checkpoint_pointer is not None or adapt_pointer is not None:
        if checkpoint_pointer is None or adapt_pointer is None:
            _warn_retention_failure(
                "estimator-ledger pointers are missing from one checkpoint "
                "owner"
            )
        else:
            checkpoint_reference = _content_addressed_sidecar_reference(
                checkpoint_pointer,
                pointer_field=f"checkpoint.{ledger_field}",
                filename_role=_CONTENT_ADDRESSED_SIDECAR_FIELDS[ledger_field],
                current_path=current_path,
            )
            adapt_reference = _content_addressed_sidecar_reference(
                adapt_pointer,
                pointer_field=f"adapt_vqe.{ledger_field}",
                filename_role=_CONTENT_ADDRESSED_SIDECAR_FIELDS[ledger_field],
                current_path=current_path,
            )
            if (
                checkpoint_reference is not None
                and adapt_reference is not None
            ):
                if checkpoint_reference == adapt_reference:
                    references[adapt_reference[0]] = adapt_reference[1]
                else:
                    _warn_retention_failure(
                        "estimator-ledger checkpoint owners disagree"
                    )
    for pointer_field, filename_role in (
        _CONTENT_ADDRESSED_SIDECAR_FIELDS.items()
    ):
        if pointer_field == ledger_field:
            continue
        pointer = adapt.get(pointer_field)
        if pointer is None:
            continue
        reference = _content_addressed_sidecar_reference(
            pointer,
            pointer_field=f"adapt_vqe.{pointer_field}",
            filename_role=filename_role,
            current_path=current_path,
        )
        if reference is not None:
            references[reference[0]] = reference[1]
    return references


def _load_predecessor_sidecar_references(
    current_path: Path,
) -> dict[Path, str]:
    try:
        predecessor_bytes = current_path.read_bytes()
    except FileNotFoundError:
        return {}
    except OSError as error:
        _warn_retention_failure(
            f"could not read predecessor {current_path.name!r}: {error}"
        )
        return {}
    try:
        predecessor = json.loads(predecessor_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _warn_retention_failure(
            f"could not decode predecessor {current_path.name!r}: {error}"
        )
        return {}
    if not isinstance(predecessor, Mapping):
        _warn_retention_failure(
            f"predecessor {current_path.name!r} is not a JSON object"
        )
        return {}
    return _content_addressed_sidecar_references(
        predecessor,
        current_path=current_path,
    )


def _retire_unreferenced_predecessor_sidecars(
    predecessor_references: Mapping[Path, str],
    *,
    current_payload: Mapping[str, Any],
    current_path: Path,
) -> None:
    current_references = _content_addressed_sidecar_references(
        current_payload,
        current_path=current_path,
    )
    retired_any = False
    for predecessor_path, expected_sha256 in predecessor_references.items():
        if predecessor_path in current_references:
            continue
        try:
            actual_sha256 = _sha256_file(predecessor_path)
        except FileNotFoundError:
            continue
        except OSError as error:
            _warn_retention_failure(
                f"could not read {predecessor_path.name!r}: {error}"
            )
            continue
        if actual_sha256 != expected_sha256:
            _warn_retention_failure(
                f"digest mismatch for {predecessor_path.name!r}"
            )
            continue
        try:
            predecessor_path.unlink()
        except OSError as error:
            _warn_retention_failure(
                f"could not retire {predecessor_path.name!r}: {error}"
            )
            continue
        retired_any = True
    if not retired_any:
        return
    try:
        directory_descriptor = os.open(current_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as error:
        _warn_retention_failure(
            f"could not sync checkpoint directory after retirement: {error}"
        )


def _strip_embedded_full_ledgers(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_embedded_full_ledgers(child)
            for key, child in value.items()
            if str(key) != "full_ledger"
        }
    if isinstance(value, list):
        return [_strip_embedded_full_ledgers(child) for child in value]
    if isinstance(value, tuple):
        return [_strip_embedded_full_ledgers(child) for child in value]
    return copy.deepcopy(value)


_ACTIVE_CHECKPOINT_ONLY_ADAPT_FIELDS = frozenset(
    {
        "boson_subspace_diagnostics",
        "branch_id",
        "controller_measurement_work_summary",
        "parent_branch_id",
        "pool_size",
        "route_a_trust_region_state",
        "strict_replay",
    }
)


def _compatibility_consumer_projection(
    full_projection: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep live-resume fields out of the completed compatibility result."""

    projection = _strip_embedded_full_ledgers(full_projection)
    if not isinstance(projection, dict):
        raise AssertionError("Compatibility projection must remain a dictionary.")
    for field in _ACTIVE_CHECKPOINT_ONLY_ADAPT_FIELDS:
        projection.pop(field, None)
    return projection


def _stable_json_digest(value: Any) -> str:
    digest = hashlib.sha256()

    class _DigestWriter:
        def write(self, text: str) -> int:
            digest.update(text.encode("utf-8"))
            return len(text)

    json.dump(
        value,
        _DigestWriter(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return digest.hexdigest()


def _build_verified_singleton_resume_sidecar(
    adapt_current: Mapping[str, Any],
    *,
    source_path: Path,
    source_sha256: str,
) -> dict[str, Any]:
    """Build the unchanged verified-singleton reader's authenticated state."""

    history = _require_sequence(
        adapt_current.get("history"),
        name="verified-resume history",
    )
    depth = int(adapt_current.get("history_count", -1))
    pool_size = int(adapt_current.get("pool_size", -1))
    if depth < 1 or len(history) != depth or pool_size < 1:
        raise RuntimeError(
            "Verified-singleton resume sidecar requires complete history and "
            "a positive pool size."
        )
    ordered_parent_indices: list[int] = []
    selected_feature_counts: list[int] = []
    controller_snapshot: dict[str, Any] | None = None
    for index, row_raw in enumerate(history):
        row = _require_mapping(
            row_raw,
            name=f"verified-resume history[{index}]",
        )
        ordered_parent_indices.append(int(row["pool_index"]))
        feature_rows = _require_sequence(
            row.get("selected_feature_rows"),
            name=f"verified-resume history[{index}] selected features",
        )
        selected_feature_counts.append(len(feature_rows))
        if len(feature_rows) != 1:
            raise RuntimeError(
                "Verified-singleton resume requires one selected feature row "
                "per accepted round."
            )
        feature = _require_mapping(
            feature_rows[0],
            name=f"verified-resume history[{index}] selected feature",
        )
        snapshot = _require_mapping(
            feature.get("controller_snapshot"),
            name=f"verified-resume history[{index}] controller snapshot",
        )
        if index == depth - 1:
            controller_snapshot = copy.deepcopy(dict(snapshot))
    if controller_snapshot is None:
        raise AssertionError("Verified-resume controller snapshot is missing.")
    controller_evidence = {
        "depth": depth,
        "drop_policy_enabled": False,
        "drop_plateau_hits": 0,
        "stage_name": "core",
        "stage_transition_reason": "stay_core",
        "controller_snapshot_count": 1,
        "selected_feature_row_index": 0,
    }
    return {
        "schema": "static_adapt_signed_active_prefix_resume_sidecar_v2",
        "source_result_json": str(source_path.resolve(strict=False)),
        "source_result_sha256": str(source_sha256),
        "source_result_digest_scope": (
            "static_adapt_verified_singleton_resume_source_projection_v1"
        ),
        "controller_snapshot": controller_snapshot,
        "controller_snapshot_sha256": _stable_json_digest(
            controller_snapshot
        ),
        "controller_state": {
            "schema": "static_adapt_singleton_controller_resume_state_v1",
            "controller_round": depth,
            "source_max_depth": depth,
            "phase1_residual_opened": False,
            "phase1_stage_name": "core",
            "source_history_row_evidence": controller_evidence,
            "source_history_row_evidence_sha256": _stable_json_digest(
                controller_evidence
            ),
        },
        "selection_state": {
            "schema": (
                "static_adapt_singleton_selection_count_resume_state_v1"
            ),
            "controller_round": depth,
            "pool_size": pool_size,
            "seq2p_logical_mode": False,
            "ordered_parent_pool_indices": ordered_parent_indices,
            "ordered_parent_pool_indices_sha256": _stable_json_digest(
                ordered_parent_indices
            ),
            "selected_feature_row_count_per_round": selected_feature_counts,
            "ordered_logical_candidate_indices": [],
            "ordered_logical_candidate_indices_sha256": _stable_json_digest(
                []
            ),
        },
        "no_credentials_serialized": True,
    }


def _build_greedy_batch_checkpoint_sidecar(
    adapt_current: Mapping[str, Any],
    *,
    source_path: Path,
    source_sha256: str,
) -> dict[str, Any]:
    """Authenticate ordered batch history without authorizing reconstruction."""

    if str(adapt_current.get("route_family", "")) != (
        "greedy_batch_response_snake"
    ):
        raise RuntimeError(
            "Greedy batch checkpoint sidecar requires the greedy route."
        )
    history = _require_sequence(
        adapt_current.get("history"),
        name="greedy batch checkpoint history",
    )
    depth = int(adapt_current.get("history_count", -1))
    if depth < 1 or len(history) != depth:
        raise RuntimeError(
            "Greedy batch checkpoint sidecar requires complete history."
        )
    round_projections: list[dict[str, Any]] = []
    for index, row_raw in enumerate(history, start=1):
        row = _require_mapping(
            row_raw,
            name=f"greedy batch checkpoint history[{index - 1}]",
        )
        admission = _require_mapping(
            row.get("greedy_batch_admission"),
            name=(
                f"greedy batch checkpoint history[{index - 1}] "
                "admission"
            ),
        )
        feature_rows = _require_sequence(
            row.get("selected_feature_rows"),
            name=(
                f"greedy batch checkpoint history[{index - 1}] "
                "selected features"
            ),
        )
        cardinality = int(row.get("selected_logical_size", -1))
        record_ids = tuple(
            str(value)
            for value in _require_sequence(
                admission.get("selected_record_ids"),
                name=(
                    f"greedy batch checkpoint history[{index - 1}] "
                    "record ids"
                ),
            )
        )
        generator_ids = tuple(
            str(value)
            for value in _require_sequence(
                admission.get("selected_generator_ids"),
                name=(
                    f"greedy batch checkpoint history[{index - 1}] "
                    "generator ids"
                ),
            )
        )
        operator_labels = tuple(
            str(value)
            for value in _require_sequence(
                row.get("selected_batch_labels"),
                name=(
                    f"greedy batch checkpoint history[{index - 1}] "
                    "operator labels"
                ),
            )
        )
        pool_indices = tuple(
            int(value)
            for value in _require_sequence(
                row.get("selected_pool_indices"),
                name=(
                    f"greedy batch checkpoint history[{index - 1}] "
                    "pool indices"
                ),
            )
        )
        original_positions = tuple(
            int(value)
            for value in _require_sequence(
                admission.get("selected_original_positions"),
                name=(
                    f"greedy batch checkpoint history[{index - 1}] "
                    "original positions"
                ),
            )
        )
        effective_positions = tuple(
            int(value)
            for value in _require_sequence(
                admission.get("selected_effective_positions"),
                name=(
                    f"greedy batch checkpoint history[{index - 1}] "
                    "effective positions"
                ),
            )
        )
        member_fields = (
            record_ids,
            generator_ids,
            operator_labels,
            pool_indices,
            original_positions,
            effective_positions,
            tuple(feature_rows),
        )
        if (
            not 1 <= cardinality <= 5
            or any(len(values) != cardinality for values in member_fields)
            or len(set(record_ids)) != cardinality
            or len(set(generator_ids)) != cardinality
        ):
            raise RuntimeError(
                "Greedy batch checkpoint member cardinalities disagree."
            )
        members = [
            {
                "selected_domain_record_id": record_ids[member_index],
                "generator_id": generator_ids[member_index],
                "selected_operator": operator_labels[member_index],
                "pool_index": pool_indices[member_index],
                "original_insertion_position": (
                    original_positions[member_index]
                ),
                "effective_insertion_position": (
                    effective_positions[member_index]
                ),
                "selected_feature": copy.deepcopy(
                    dict(
                        _require_mapping(
                            feature_rows[member_index],
                            name=(
                                "greedy batch checkpoint selected "
                                f"feature[{member_index}]"
                            ),
                        )
                    )
                ),
            }
            for member_index in range(cardinality)
        ]
        round_projection = {
            "controller_round": index,
            "selected_cardinality": cardinality,
            "composition_identity": str(
                admission.get("composition_identity", "")
            ),
            "members": members,
            "admission": copy.deepcopy(dict(admission)),
            "active_prefix_checkpoint": copy.deepcopy(
                dict(
                    _require_mapping(
                        row.get("active_prefix_checkpoint"),
                        name=(
                            "greedy batch checkpoint active-prefix "
                            f"receipt[{index - 1}]"
                        ),
                    )
                )
            ),
        }
        if (
            not round_projection["composition_identity"]
            or int(row.get("depth", -1)) != index
        ):
            raise RuntimeError(
                "Greedy batch checkpoint round identity is incomplete."
            )
        round_projections.append(round_projection)
    terminal_checkpoint = copy.deepcopy(
        dict(
            _require_mapping(
                adapt_current.get("terminal_active_prefix_checkpoint"),
                name="greedy batch terminal active-prefix checkpoint",
            )
        )
    )
    return {
        "schema": "static_adapt_signed_greedy_batch_checkpoint_sidecar_v1",
        "source_result_json": str(source_path.resolve(strict=False)),
        "source_result_sha256": str(source_sha256),
        "source_result_digest_scope": (
            "static_adapt_greedy_batch_checkpoint_source_projection_v1"
        ),
        "route_family": "greedy_batch_response_snake",
        "route_profile": str(adapt_current.get("route_profile", "")),
        "route_contract_sha256": str(
            adapt_current.get("sr_route_profile_contract_sha256", "")
        ),
        "controller_round": depth,
        "rounds": round_projections,
        "rounds_sha256": _stable_json_digest(round_projections),
        "terminal_active_prefix_checkpoint": terminal_checkpoint,
        "resume_authorization": {
            "enabled": False,
            "status": "not_authorized_until_issue_19",
            "reader_contract": (
                "greedy_batch_checkpoint_projection_only_v1"
            ),
        },
        "no_credentials_serialized": True,
    }


def _build_combinatorial_batch_checkpoint_sidecar(
    adapt_current: Mapping[str, Any],
    *,
    source_path: Path,
    source_sha256: str,
) -> dict[str, Any]:
    """Authenticate exhaustive-subset history without authorizing resume."""

    if str(adapt_current.get("route_family", "")) != (
        "combinatorial_batch_response_snake"
    ):
        raise RuntimeError(
            "Combinatorial checkpoint sidecar requires its batch route."
        )
    projection = copy.deepcopy(dict(adapt_current))
    projection["route_family"] = "greedy_batch_response_snake"
    history = _require_sequence(
        projection.get("history"),
        name="combinatorial batch checkpoint history",
    )
    original_admissions: list[dict[str, Any]] = []
    for index, row_raw in enumerate(history):
        row = _require_mapping(
            row_raw,
            name=f"combinatorial checkpoint history[{index}]",
        )
        if not isinstance(row, dict):
            raise RuntimeError(
                "Combinatorial checkpoint history must be mutable copies."
            )
        admission = copy.deepcopy(
            dict(
                _require_mapping(
                    row.get("combinatorial_batch_admission"),
                    name=(
                        f"combinatorial checkpoint history[{index}] "
                        "admission"
                    ),
                )
            )
        )
        original_admissions.append(admission)
        row["greedy_batch_admission"] = admission
    sidecar = _build_greedy_batch_checkpoint_sidecar(
        projection,
        source_path=source_path,
        source_sha256=source_sha256,
    )
    sidecar.update(
        {
            "schema": (
                "static_adapt_signed_combinatorial_batch_"
                "checkpoint_sidecar_v1"
            ),
            "source_result_digest_scope": (
                "static_adapt_combinatorial_batch_checkpoint_"
                "source_projection_v1"
            ),
            "route_family": "combinatorial_batch_response_snake",
            "resume_authorization": {
                "enabled": False,
                "status": "not_authorized_until_issue_19",
                "reader_contract": (
                    "combinatorial_batch_checkpoint_projection_only_v1"
                ),
            },
        }
    )
    rounds = _require_sequence(
        sidecar.get("rounds"),
        name="combinatorial checkpoint rounds",
    )
    for index, (round_raw, admission) in enumerate(
        zip(rounds, original_admissions, strict=True)
    ):
        if not isinstance(round_raw, dict):
            raise RuntimeError(
                "Combinatorial checkpoint round must be mutable."
            )
        round_raw["proposal"] = {
            key: copy.deepcopy(admission.get(key))
            for key in (
                "identity",
                "maximum_size",
                "search_window_size",
                "ranked_population_count",
                "ranked_window_count",
                "selected_record_ids",
                "score",
                "modeled_energy_decrease",
                "predictive_cost_excess",
                "denominator",
                "geometry_identity",
                "evaluated_subset_count",
                "subset_counts_considered",
                "subset_counts_evaluated",
                "subset_counts_feasible",
            )
        }
        if int(round_raw.get("controller_round", -1)) != index + 1:
            raise RuntimeError(
                "Combinatorial checkpoint rounds are not contiguous."
            )
    sidecar["rounds_sha256"] = _stable_json_digest(rounds)
    return sidecar


def _publish_active_cli_current_checkpoint(
    output_payload: Mapping[str, Any],
    *,
    ledger_payload: Mapping[str, Any],
    path: Path,
    keep_history_tail: int,
) -> None:
    """Publish one resumable accepted-prefix envelope and its private sidecar."""

    predecessor_sidecar_references = _load_predecessor_sidecar_references(path)
    current = dict(output_payload)
    adapt = _require_mapping(
        current.get("adapt_vqe"),
        name="current ADAPT block",
    )
    full_projection = dict(adapt)
    history = _require_sequence(
        full_projection.get("history"),
        name="current accepted history",
    )
    depth = int(full_projection.get("history_count", len(history)))
    if depth != len(history):
        raise RuntimeError(
            "Current-checkpoint full projection history count is inconsistent."
        )
    tail_limit = int(keep_history_tail)
    if tail_limit < 0:
        raise ValueError("keep_history_tail must be nonnegative.")
    resume_history: list[dict[str, Any]] = []
    for index, row_raw in enumerate(history):
        row = dict(
            _require_mapping(
                row_raw,
                name=f"current accepted history[{index}]",
            )
        )
        selected_batch = _require_sequence(
            row.get("selected_batch_labels", row.get("selected_ops")),
            name=f"current accepted history[{index}] selected batch",
        )
        batch_size = len(selected_batch)
        if batch_size < 1:
            raise RuntimeError(
                "Current-checkpoint accepted history contains an empty batch."
            )
        recorded_batch_size = row.get("batch_size")
        if (
            recorded_batch_size is not None
            and int(recorded_batch_size) != batch_size
        ):
            raise RuntimeError(
                "Current-checkpoint accepted history batch size disagrees "
                "with its selected members."
            )
        row["batch_size"] = batch_size
        row.setdefault("branch_id", None)
        row.setdefault("parent_branch_id", None)
        resume_history.append(row)
    requested_history_tail = (
        []
        if tail_limit == 0
        else list(resume_history[-tail_limit:])
    )
    history_tail = requested_history_tail
    adapt_current = _strip_embedded_full_ledgers(
        {
            key: value
            for key, value in full_projection.items()
            if key not in {"history", "history_tail"}
        }
    )
    if not isinstance(adapt_current, dict):
        raise AssertionError("Current ADAPT projection must remain a dictionary.")
    adapt_current["history"] = resume_history
    adapt_current["history_tail"] = history_tail
    beam_enabled = adapt_current.get("adapt_beam_enabled") is True
    branch_id = adapt_current.get("branch_id")
    parent_branch_id = adapt_current.get("parent_branch_id")
    checkpoint_branch_policy = (
        "canonical_terminal_winning_lineage"
        if beam_enabled
        else None
    )
    ledger_scope = (
        "all_executed_branches" if beam_enabled else "single_route"
    )
    if beam_enabled:
        diagnostics = _require_mapping(
            adapt_current.get("beam_search_diagnostics"),
            name="canonical beam search diagnostics",
        )
        winning_branch_ids = tuple(
            str(value)
            for value in _require_sequence(
                diagnostics.get("winning_branch_ids"),
                name="canonical beam winning branch ids",
            )
        )
        history_branch_ids = tuple(
            str(row["branch_id"])
            for row in resume_history
            if row.get("branch_id") not in {None, ""}
        )
        if (
            not winning_branch_ids
            or len(set(winning_branch_ids)) != len(winning_branch_ids)
            or winning_branch_ids != history_branch_ids
            or str(branch_id) != winning_branch_ids[-1]
            or (
                parent_branch_id
                != (
                    None
                    if len(winning_branch_ids) < 2
                    else winning_branch_ids[-2]
                )
            )
        ):
            raise RuntimeError(
                "Canonical beam checkpoint winner metadata disagrees with "
                "its accepted history."
            )
    elif branch_id is not None or parent_branch_id is not None:
        raise RuntimeError(
            "A single-route checkpoint cannot carry beam branch metadata."
        )
    greedy_batch_checkpoint = str(
        adapt_current.get("route_family", "")
    ) == "greedy_batch_response_snake"
    combinatorial_batch_checkpoint = str(
        adapt_current.get("route_family", "")
    ) == "combinatorial_batch_response_snake"
    adapt_current["history_tail_retention"] = {
        "schema": "static_adapt_verified_resume_history_retention_v2",
        "requested_limit": tail_limit,
        "requested_window_count": len(requested_history_tail),
        "serialized_complete_history_count": len(resume_history),
        "serialized_tail_count": len(history_tail),
        "normalized_for_verified_singleton_resume": bool(
            not (
                greedy_batch_checkpoint
                or combinatorial_batch_checkpoint
            )
        ),
        "no_credentials_serialized": True,
    }
    if greedy_batch_checkpoint:
        adapt_current["history_tail_retention"]["schema"] = (
            "static_adapt_greedy_batch_checkpoint_history_retention_v1"
        )
        adapt_current["history_tail_retention"][
            "checkpoint_projection_only"
        ] = True
    elif combinatorial_batch_checkpoint:
        adapt_current["history_tail_retention"]["schema"] = (
            "static_adapt_combinatorial_batch_checkpoint_"
            "history_retention_v1"
        )
        adapt_current["history_tail_retention"][
            "checkpoint_projection_only"
        ] = True

    ledger = dict(ledger_payload)
    ledger_summary = _require_mapping(
        ledger.get("summary"),
        name="checkpoint ledger summary",
    )
    occurrence_summary = _require_mapping(
        ledger.get("occurrence_summary"),
        name="checkpoint ledger occurrence summary",
    )
    ledger_fingerprint = str(ledger.get("ledger_fingerprint", ""))
    if not ledger_fingerprint:
        raise RuntimeError(
            "Current-checkpoint projection lacks a ledger fingerprint."
        )
    raw_occurrences = int(
        occurrence_summary.get("total_call_occurrences", -1)
    )
    unique_primitives = int(
        ledger_summary.get("unique_primitive_count", -1)
    )
    s_unique = int(ledger_summary.get("S_unique", -1))
    projected_accounting = _require_mapping(
        full_projection.get("estimator_call_accounting"),
        name="projected estimator accounting",
    )
    if (
        raw_occurrences != int(projected_accounting.get("S_alg", -1))
        or unique_primitives < 0
        or s_unique != int(projected_accounting.get("S_unique", -1))
    ):
        raise RuntimeError(
            "Current-checkpoint ledger counts disagree with projected accounting."
        )

    sidecar = {
        "schema": "paper_i_estimator_call_ledger_checkpoint_sidecar_v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "reason": "iteration_done",
            "depth": int(depth),
            "beam_enabled": beam_enabled,
            "checkpoint_branch_policy": checkpoint_branch_policy,
            "branch_id": branch_id,
            "parent_branch_id": parent_branch_id,
            "ledger_scope": ledger_scope,
            "current_round_finalized": True,
        },
        "ledger_scope": ledger_scope,
        "ledger": ledger,
        "ledger_fingerprint": ledger_fingerprint,
        "unique_primitive_count": unique_primitives,
        "raw_occurrence_count": raw_occurrences,
        "S_alg": raw_occurrences,
        "S_unique": s_unique,
        "consumer_complete_projection": {
            "schema": "static_adapt_consumer_projection_reference_v1",
            "source_projection_sha256": _stable_json_digest(full_projection),
            "source_projection_digest_scope": (
                "static_adapt_full_projection_v1"
            ),
            "materialized_in": "current_checkpoint.adapt_vqe",
            "embedded_full_ledgers_omitted": True,
        },
        "no_credentials_serialized": True,
    }
    sidecar_path, sidecar_sha256 = _atomic_write_content_addressed_json(
        path,
        filename_role="estimator_call_ledger_checkpoint",
        payload=sidecar,
    )

    pointer = {
        "schema": "paper_i_estimator_call_ledger_checkpoint_pointer_v2",
        "enabled": True,
        "status": "complete",
        "path": sidecar_path.name,
        "sha256": sidecar_sha256,
        "ledger_schema": str(ledger.get("schema", "")),
        "checkpoint_reason": "iteration_done",
        "ledger_fingerprint": ledger_fingerprint,
        "unique_primitive_count": unique_primitives,
        "raw_occurrence_count": raw_occurrences,
        "S_alg": raw_occurrences,
        "S_unique": s_unique,
        "checkpoint_depth": int(depth),
        "beam_enabled": beam_enabled,
        "checkpoint_branch_policy": checkpoint_branch_policy,
        "branch_id": branch_id,
        "parent_branch_id": parent_branch_id,
        "ledger_scope": ledger_scope,
        "current_round_finalized": True,
    }
    adapt_current.update(
        {
            "partial_checkpoint": True,
            "checkpoint_reason": "iteration_done",
            "success": False,
            "stop_reason": None,
            "adapt_beam_enabled": beam_enabled,
            "branch_id": branch_id,
            "parent_branch_id": parent_branch_id,
            "history_count": int(depth),
            "history_tail_count": len(history_tail),
            "history_checkpoint_complete": bool(
                len(adapt_current["history"]) == int(depth)
                and adapt_current["history_tail"]
                == (
                    []
                    if len(history_tail) == 0
                    else adapt_current["history"][-len(history_tail) :]
                )
            ),
            "beam_replay_telemetry": None,
            "formal_manifold_runtime_checkpoint": None,
            "formal_manifold_warm_state_checkpoint": None,
            "formal_manifold_query_closure_checkpoint": None,
            "final_full_refit": {
                "schema_version": "adapt_final_full_refit_v1",
                "attempted": False,
                "executed": False,
                "nfev": 0,
                "skipped_reason": "checkpoint_before_final_refit",
            },
            "estimator_call_ledger_checkpoint": pointer,
        }
    )
    current.update(
        {
            "schema_version": "static_adapt_current_checkpoint_v1",
            "no_credentials_serialized": True,
            "checkpoint": {
                "complete": False,
                "reason": "iteration_done",
                "beam_enabled": beam_enabled,
                "checkpoint_branch_policy": checkpoint_branch_policy,
                "branch_id": branch_id,
                "parent_branch_id": parent_branch_id,
                "ledger_scope": ledger_scope,
                "depth": int(depth),
                "ansatz_depth": int(
                    adapt_current.get(
                        "ansatz_depth",
                        len(adapt_current.get("operators", ())),
                    )
                ),
                "stop_reason": adapt_current.get("stop_reason"),
                "sr_route_profile_contract_sha256": adapt_current.get(
                    "sr_route_profile_contract_sha256"
                ),
                "phase3_response_coordinate_scope": adapt_current.get(
                    "phase3_response_coordinate_scope"
                ),
                "estimator_call_ledger_checkpoint": pointer,
                "path": str(path),
            },
            "adapt_vqe": adapt_current,
        }
    )
    source_projection_sha256 = _stable_json_digest(current)
    if greedy_batch_checkpoint:
        batch_sidecar = _build_greedy_batch_checkpoint_sidecar(
            adapt_current,
            source_path=path,
            source_sha256=source_projection_sha256,
        )
        batch_sidecar_path, batch_sidecar_sha256 = (
            _atomic_write_content_addressed_json(
                path,
                filename_role="greedy_batch_checkpoint",
                payload=batch_sidecar,
            )
        )
        adapt_current["greedy_batch_checkpoint_sidecar"] = {
            "schema": (
                "static_adapt_greedy_batch_checkpoint_sidecar_pointer_v1"
            ),
            "enabled": True,
            "status": "complete",
            "path": batch_sidecar_path.name,
            "sha256": batch_sidecar_sha256,
            "sidecar_schema": (
                "static_adapt_signed_greedy_batch_checkpoint_sidecar_v1"
            ),
            "source_projection_schema": (
                "static_adapt_greedy_batch_checkpoint_source_projection_v1"
            ),
            "source_projection_sha256": source_projection_sha256,
            "resume_enabled": False,
            "resume_status": "not_authorized_until_issue_19",
            "no_credentials_serialized": True,
        }
        _atomic_write_json(path, current)
        _retire_unreferenced_predecessor_sidecars(
            predecessor_sidecar_references,
            current_payload=current,
            current_path=path,
        )
        return
    if combinatorial_batch_checkpoint:
        batch_sidecar = _build_combinatorial_batch_checkpoint_sidecar(
            adapt_current,
            source_path=path,
            source_sha256=source_projection_sha256,
        )
        batch_sidecar_path, batch_sidecar_sha256 = (
            _atomic_write_content_addressed_json(
                path,
                filename_role="combinatorial_batch_checkpoint",
                payload=batch_sidecar,
            )
        )
        adapt_current["combinatorial_batch_checkpoint_sidecar"] = {
            "schema": (
                "static_adapt_combinatorial_batch_checkpoint_"
                "sidecar_pointer_v1"
            ),
            "enabled": True,
            "status": "complete",
            "path": batch_sidecar_path.name,
            "sha256": batch_sidecar_sha256,
            "sidecar_schema": (
                "static_adapt_signed_combinatorial_batch_"
                "checkpoint_sidecar_v1"
            ),
            "source_projection_schema": (
                "static_adapt_combinatorial_batch_checkpoint_"
                "source_projection_v1"
            ),
            "source_projection_sha256": source_projection_sha256,
            "resume_enabled": False,
            "resume_status": "not_authorized_until_issue_19",
            "no_credentials_serialized": True,
        }
        _atomic_write_json(path, current)
        _retire_unreferenced_predecessor_sidecars(
            predecessor_sidecar_references,
            current_payload=current,
            current_path=path,
        )
        return
    round_zero_phase3_natural_terminal = bool(
        int(depth) == 0
        and not adapt_current.get("history")
        and adapt_current.get("terminal_controller_outcome")
        == ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1
        and _require_mapping(
            adapt_current.get("terminal_active_prefix_checkpoint"),
            name="round-zero terminal active-prefix checkpoint",
        ).get("checkpoint_kind")
        == "terminal_phase3_no_positive"
    )
    if round_zero_phase3_natural_terminal:
        _atomic_write_json(path, current)
        _retire_unreferenced_predecessor_sidecars(
            predecessor_sidecar_references,
            current_payload=current,
            current_path=path,
        )
        return
    resume_sidecar = _build_verified_singleton_resume_sidecar(
        adapt_current,
        source_path=path,
        source_sha256=source_projection_sha256,
    )
    resume_sidecar_path, resume_sidecar_sha256 = (
        _atomic_write_content_addressed_json(
            path,
            filename_role="verified_singleton_resume",
            payload=resume_sidecar,
        )
    )
    adapt_current["verified_singleton_resume_sidecar"] = {
        "schema": (
            "static_adapt_verified_singleton_resume_sidecar_pointer_v1"
        ),
        "enabled": True,
        "status": "complete",
        "path": resume_sidecar_path.name,
        "sha256": resume_sidecar_sha256,
        "sidecar_schema": (
            "static_adapt_signed_active_prefix_resume_sidecar_v2"
        ),
        "source_projection_schema": (
            "static_adapt_verified_singleton_resume_source_projection_v1"
        ),
        "source_projection_sha256": source_projection_sha256,
        "no_credentials_serialized": True,
    }
    _atomic_write_json(path, current)
    _retire_unreferenced_predecessor_sidecars(
        predecessor_sidecar_references,
        current_payload=current,
        current_path=path,
    )

__all__ = [
    "_publish_active_cli_current_checkpoint",
    "_stable_json_digest",
]
