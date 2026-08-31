"""Durable supervisor for streamed time-dynamics result acquisition.

The scientific planners deliberately remain elsewhere:

* :meth:`StagedCampaign.next_actions` owns Paper-II rung selection;
* :func:`plan_siam_nb4_ladder` owns the SIAM comparator-first ladder; and
* :mod:`pipelines.artifacts.ingestion` owns one completed-process ingestion
  transaction.

This module owns the missing outer state machine.  It turns one combined
scheduler snapshot into durable, idempotent work: reconcile receipt coverage,
ingest successful deltas, record terminal failures, refresh author-facing
status, and submit only planner-produced units.  A submission intent is written
*before* contacting the scheduler; after a crash the supervisor reconciles the
intent by its stable unit identity before it may submit again.

The supervisor is transport- and campaign-agnostic.  Lane adapters supply the
side effects through :class:`LaneHandlers`; tests use fakes, while CHTC and the
two scientific planners stay behind their existing interfaces.  This keeps the
high-impact scientific state machines out of the orchestration layer.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import fcntl
import hashlib
import importlib
import inspect
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from pipelines.artifacts.ingestion import (
    INGESTION_ENGINE_V1,
    INGESTION_RECEIPT_SCHEMA_V3,
)
from pipelines.time_dynamics.legacy_coverage import (
    LegacyCoverageError,
    validate_legacy_coverage_index,
)

REGISTRY_SCHEMA_V1 = "time_dynamics_acquisition_registry_v1"
LANE_SCIENTIFIC_AUTHORITY_SCHEMA_V1 = (
    "time_dynamics_lane_scientific_authority_v1"
)
SNAPSHOT_SCHEMA_V1 = "time_dynamics_scheduler_snapshot_v1"
STATE_SCHEMA_V1 = "time_dynamics_acquisition_state_v1"
CYCLE_RECEIPT_SCHEMA_V1 = "time_dynamics_acquisition_cycle_v1"
FAILURE_DISPOSITION_SCHEMA_V1 = "time_dynamics_terminal_failure_disposition_v1"
SUBMISSION_INTENT_SCHEMA_V1 = "time_dynamics_submission_intent_v1"
SUBMISSION_RECEIPT_SCHEMA_V1 = "time_dynamics_submission_receipt_v1"
RAW_SCHEDULER_EVIDENCE_SCHEMA_V1 = "time_dynamics_raw_scheduler_evidence_v1"
SCOPE_AUTHORITY_SCHEMA_V1 = "time_dynamics_scheduler_scope_authority_v1"
RECONCILIATION_FOUND = "found"
RECONCILIATION_VERIFIED_ABSENT = "verified_absent"
RECONCILIATION_INDETERMINATE = "indeterminate"
RECONCILIATION_STATES = frozenset(
    {
        RECONCILIATION_FOUND,
        RECONCILIATION_VERIFIED_ABSENT,
        RECONCILIATION_INDETERMINATE,
    }
)

ACTIVE_STATES = frozenset({"idle", "running", "held"})
TERMINAL_STATES = frozenset({"succeeded", "failed", "removed"})
VALID_STATES = ACTIVE_STATES | TERMINAL_STATES
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True, order=True)
class SchedulerScope:
    """One explicit lane/cluster/process scope for a combined snapshot."""

    lane_id: str
    cluster_id: int
    process_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.lane_id or not _SAFE_ID.fullmatch(self.lane_id):
            raise ValueError(f"invalid scheduler-scope lane_id {self.lane_id!r}")
        if self.cluster_id < 0:
            raise ValueError("scheduler-scope cluster_id must be nonnegative")
        if not self.process_ids:
            raise ValueError("scheduler scope must name at least one process")
        if (
            any(process_id < 0 for process_id in self.process_ids)
            or tuple(sorted(set(self.process_ids))) != self.process_ids
        ):
            raise ValueError(
                "scheduler-scope process_ids must be sorted, unique, and nonnegative"
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SchedulerScope":
        process_ids = payload.get("process_ids")
        if not isinstance(process_ids, list):
            raise ValueError("scheduler scope has no process_ids list")
        return cls(
            lane_id=str(payload.get("lane_id") or ""),
            cluster_id=int(payload["cluster_id"]),
            process_ids=tuple(int(value) for value in process_ids),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "cluster_id": self.cluster_id,
            "process_ids": list(self.process_ids),
        }


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace one JSON object without exposing a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        temporary = ""
    except BaseException:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        raise


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object: {path}")
    return dict(value)


def _stable_filename(value: str) -> str:
    return value if _SAFE_ID.fullmatch(value) else hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class ProductionOperationsBinding:
    """Registry authority for one production adapter and scope bootstrap."""

    adapter_id: str
    factory: str
    source_path: Path
    source_sha256: str
    bootstrap_scopes_path: Path
    bootstrap_scopes_sha256: str

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], *, repository_root: Path
    ) -> "ProductionOperationsBinding":
        adapter_id = str(payload.get("adapter_id") or "")
        factory = str(payload.get("factory") or "")
        if not adapter_id or not _SAFE_ID.fullmatch(adapter_id):
            raise ValueError("production operations binding has no valid adapter_id")
        module, separator, function = factory.partition(":")
        if not separator or not module or not function:
            raise ValueError("production operations factory must be module:function")

        def artifact(name: str) -> tuple[Path, str]:
            raw = payload.get(name)
            if not isinstance(raw, Mapping):
                raise ValueError(f"production operations binding has no {name}")
            path = Path(str(raw.get("path") or ""))
            digest = str(raw.get("sha256") or "")
            if (
                not path.parts
                or path.is_absolute()
                or ".." in path.parts
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
            ):
                raise ValueError(
                    f"production operations {name} needs a repository-relative "
                    "path and lowercase SHA-256"
                )
            return repository_root / path, digest

        source_path, source_sha256 = artifact("source")
        bootstrap_path, bootstrap_sha256 = artifact("bootstrap_scopes")
        return cls(
            adapter_id=adapter_id,
            factory=factory,
            source_path=source_path,
            source_sha256=source_sha256,
            bootstrap_scopes_path=bootstrap_path,
            bootstrap_scopes_sha256=bootstrap_sha256,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "factory": self.factory,
            "source": {
                "path": str(self.source_path),
                "sha256": self.source_sha256,
            },
            "bootstrap_scopes": {
                "path": str(self.bootstrap_scopes_path),
                "sha256": self.bootstrap_scopes_sha256,
            },
        }


@dataclass(frozen=True)
class BlockedProductionOperationsBinding:
    """Machine-readable reason why a concrete adapter is not activated."""

    adapter_id: str
    factory: str
    source_path: Path
    source_sha256: str
    blocked_by: tuple[str, ...]

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], *, repository_root: Path
    ) -> "BlockedProductionOperationsBinding":
        adapter_id = str(payload.get("adapter_id") or "")
        factory = str(payload.get("factory") or "")
        status = str(payload.get("status") or "")
        blocked_by = tuple(str(value) for value in payload.get("blocked_by") or ())
        source = payload.get("source")
        if (
            status != "blocked"
            or not adapter_id
            or not _SAFE_ID.fullmatch(adapter_id)
            or ":" not in factory
            or not blocked_by
            or any(not value or not _SAFE_ID.fullmatch(value) for value in blocked_by)
            or not isinstance(source, Mapping)
        ):
            raise ValueError("invalid blocked production operations binding")
        path = Path(str(source.get("path") or ""))
        digest = str(source.get("sha256") or "")
        if (
            not path.parts
            or path.is_absolute()
            or ".." in path.parts
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ValueError(
                "blocked production operations source needs a repository-relative "
                "path and lowercase SHA-256"
            )
        return cls(
            adapter_id=adapter_id,
            factory=factory,
            source_path=repository_root / path,
            source_sha256=digest,
            blocked_by=blocked_by,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "status": "blocked",
            "adapter_id": self.adapter_id,
            "factory": self.factory,
            "source": {
                "path": str(self.source_path),
                "sha256": self.source_sha256,
            },
            "blocked_by": list(self.blocked_by),
        }


@dataclass(frozen=True)
class LaneRegistration:
    """One independently planned scientific lane in the acquisition registry."""

    lane_id: str
    kind: str
    monitor_enabled: bool
    submission_enabled: bool
    attempts_root: Path
    state_root: Path
    scientific_state: Path | None = None
    report_contract: Path | None = None
    planner_authority: str = ""
    blocked_by: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], *, repository_root: Path
    ) -> "LaneRegistration":
        lane_id = str(payload.get("lane_id") or "")
        kind = str(payload.get("kind") or "")
        if not lane_id or not _SAFE_ID.fullmatch(lane_id):
            raise ValueError(f"invalid acquisition lane_id {lane_id!r}")
        if not kind:
            raise ValueError(f"lane {lane_id!r} has no kind")

        def resolve(name: str, *, required: bool) -> Path | None:
            raw = payload.get(name)
            if raw in (None, ""):
                if required:
                    raise ValueError(f"lane {lane_id!r} has no {name}")
                return None
            path = Path(str(raw))
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(
                    f"lane {lane_id!r} {name} must be repository-relative: {path}"
                )
            return repository_root / path

        monitor = bool(payload.get("monitor_enabled", False))
        submit = bool(payload.get("submission_enabled", False))
        blockers = tuple(str(value) for value in payload.get("blocked_by") or ())
        if submit and not monitor:
            raise ValueError(
                f"lane {lane_id!r} cannot submit while monitoring is disabled"
            )
        if submit and blockers:
            raise ValueError(
                f"lane {lane_id!r} cannot submit while blocked_by is nonempty"
            )
        return cls(
            lane_id=lane_id,
            kind=kind,
            monitor_enabled=monitor,
            submission_enabled=submit,
            attempts_root=resolve("attempts_root", required=True),  # type: ignore[arg-type]
            state_root=resolve("state_root", required=True),  # type: ignore[arg-type]
            scientific_state=resolve("scientific_state", required=False),
            report_contract=resolve("report_contract", required=False),
            planner_authority=str(payload.get("planner_authority") or ""),
            blocked_by=blockers,
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class LaneScientificAuthority:
    """Stable scientific paths and digests for one registered lane.

    This projection deliberately excludes scheduler enablement, blockers, and
    the production adapter. Those fields may change as a lane is activated;
    the scientific state/report paths and frozen contract digests may not.
    """

    registry_id: str
    lane_id: str
    planner_authority: str
    scientific_state_path: str
    report_contract_path: str
    planned_report_contract_path: str
    planned_report_contract_file_sha256: str
    planned_report_contract_sha256: str
    planned_report_tuple_structure_sha256: str
    matrix_raw_sha256: str
    matrix_payload_sha256: str
    matrix_expanded_semantic_sha256: str
    matrix_accuracy_policy_sha256: str

    def __post_init__(self) -> None:
        for label, value in (
            ("registry_id", self.registry_id),
            ("lane_id", self.lane_id),
            ("planner_authority", self.planner_authority),
        ):
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"lane scientific authority has invalid {label}")
        if not _SAFE_ID.fullmatch(self.lane_id):
            raise ValueError("lane scientific authority has invalid lane_id")
        for label, value in (
            ("scientific_state_path", self.scientific_state_path),
            ("report_contract_path", self.report_contract_path),
            ("planned_report_contract_path", self.planned_report_contract_path),
        ):
            if not isinstance(value, str):
                raise ValueError(
                    f"lane scientific authority has invalid {label}"
                )
            path = Path(value)
            if (
                not path.parts
                or path.is_absolute()
                or ".." in path.parts
                or value != path.as_posix()
            ):
                raise ValueError(
                    f"lane scientific authority has invalid {label}"
                )
        for label, value in (
            (
                "planned_report_contract_file_sha256",
                self.planned_report_contract_file_sha256,
            ),
            (
                "planned_report_contract_sha256",
                self.planned_report_contract_sha256,
            ),
            (
                "planned_report_tuple_structure_sha256",
                self.planned_report_tuple_structure_sha256,
            ),
            ("matrix_raw_sha256", self.matrix_raw_sha256),
            ("matrix_payload_sha256", self.matrix_payload_sha256),
            (
                "matrix_expanded_semantic_sha256",
                self.matrix_expanded_semantic_sha256,
            ),
            (
                "matrix_accuracy_policy_sha256",
                self.matrix_accuracy_policy_sha256,
            ),
        ):
            if not isinstance(value, str) or not re.fullmatch(
                r"[0-9a-f]{64}", value
            ):
                raise ValueError(
                    f"lane scientific authority has invalid {label}"
                )

    def _unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema": LANE_SCIENTIFIC_AUTHORITY_SCHEMA_V1,
            "registry_id": self.registry_id,
            "lane_id": self.lane_id,
            "planner_authority": self.planner_authority,
            "scientific_state_path": self.scientific_state_path,
            "report_contract_path": self.report_contract_path,
            "planned_report_contract_path": self.planned_report_contract_path,
            "planned_report_contract_file_sha256": (
                self.planned_report_contract_file_sha256
            ),
            "planned_report_contract_sha256": (
                self.planned_report_contract_sha256
            ),
            "planned_report_tuple_structure_sha256": (
                self.planned_report_tuple_structure_sha256
            ),
            "matrix_raw_sha256": self.matrix_raw_sha256,
            "matrix_payload_sha256": self.matrix_payload_sha256,
            "matrix_expanded_semantic_sha256": (
                self.matrix_expanded_semantic_sha256
            ),
            "matrix_accuracy_policy_sha256": (
                self.matrix_accuracy_policy_sha256
            ),
        }

    @property
    def authority_sha256(self) -> str:
        return _digest(self._unsigned_payload())

    @classmethod
    def from_json_dict(
        cls, payload: Mapping[str, Any]
    ) -> "LaneScientificAuthority":
        """Reconstruct and authenticate an embedded authority projection."""

        if not isinstance(payload, Mapping):
            raise ValueError("lane scientific authority must be an object")
        required = {
            "schema",
            "registry_id",
            "lane_id",
            "planner_authority",
            "scientific_state_path",
            "report_contract_path",
            "planned_report_contract_path",
            "planned_report_contract_file_sha256",
            "planned_report_contract_sha256",
            "planned_report_tuple_structure_sha256",
            "matrix_raw_sha256",
            "matrix_payload_sha256",
            "matrix_expanded_semantic_sha256",
            "matrix_accuracy_policy_sha256",
            "authority_sha256",
        }
        if (
            set(payload) != required
            or payload.get("schema")
            != LANE_SCIENTIFIC_AUTHORITY_SCHEMA_V1
        ):
            raise ValueError("unsupported lane scientific authority contract")
        result = cls(
            registry_id=payload.get("registry_id"),
            lane_id=payload.get("lane_id"),
            planner_authority=payload.get("planner_authority"),
            scientific_state_path=payload.get("scientific_state_path"),
            report_contract_path=payload.get("report_contract_path"),
            planned_report_contract_path=payload.get(
                "planned_report_contract_path"
            ),
            planned_report_contract_file_sha256=payload.get(
                "planned_report_contract_file_sha256"
            ),
            planned_report_contract_sha256=payload.get(
                "planned_report_contract_sha256"
            ),
            planned_report_tuple_structure_sha256=payload.get(
                "planned_report_tuple_structure_sha256"
            ),
            matrix_raw_sha256=payload.get("matrix_raw_sha256"),
            matrix_payload_sha256=payload.get("matrix_payload_sha256"),
            matrix_expanded_semantic_sha256=payload.get(
                "matrix_expanded_semantic_sha256"
            ),
            matrix_accuracy_policy_sha256=payload.get(
                "matrix_accuracy_policy_sha256"
            ),
        )
        if payload.get("authority_sha256") != result.authority_sha256:
            raise ValueError("lane scientific authority digest mismatch")
        return result

    def to_json_dict(self) -> dict[str, Any]:
        return {
            **self._unsigned_payload(),
            "authority_sha256": self.authority_sha256,
        }


@dataclass(frozen=True)
class CampaignRegistry:
    registry_id: str
    repository_root: Path
    supervisor_state_root: Path
    lanes: tuple[LaneRegistration, ...]
    scientific_contracts: Mapping[str, Any] = field(default_factory=dict)
    production_operations: ProductionOperationsBinding | None = None
    production_operations_blocked: BlockedProductionOperationsBinding | None = None
    registry_path: Path | None = None

    @classmethod
    def read(
        cls, path: str | Path, *, repository_root: str | Path | None = None
    ) -> "CampaignRegistry":
        registry_path = Path(path)
        payload = _read_object(registry_path, label="acquisition registry")
        if payload.get("schema") != REGISTRY_SCHEMA_V1:
            raise ValueError("unsupported time-dynamics acquisition registry schema")
        root = (
            Path(repository_root).resolve()
            if repository_root is not None
            else registry_path.resolve().parents[3]
        )
        state_raw = Path(str(payload.get("supervisor_state_root") or ""))
        if not state_raw.parts or state_raw.is_absolute() or ".." in state_raw.parts:
            raise ValueError("supervisor_state_root must be repository-relative")
        rows = payload.get("lanes")
        if not isinstance(rows, list) or not rows:
            raise ValueError("acquisition registry must contain at least one lane")
        lanes = tuple(
            LaneRegistration.from_mapping(row, repository_root=root)
            for row in rows
            if isinstance(row, Mapping)
        )
        if len(lanes) != len(rows):
            raise ValueError("every acquisition lane must be a JSON object")
        ids = [lane.lane_id for lane in lanes]
        if len(set(ids)) != len(ids):
            raise ValueError("acquisition registry repeats a lane_id")
        if not str(payload.get("registry_id") or ""):
            raise ValueError("acquisition registry has no registry_id")
        attempts_roots = [lane.attempts_root.resolve() for lane in lanes]
        state_roots = [lane.state_root.resolve() for lane in lanes]
        if len(set(attempts_roots)) != len(attempts_roots):
            raise ValueError("acquisition lanes must not share an attempts_root")
        if len(set(state_roots)) != len(state_roots):
            raise ValueError("acquisition lanes must not share a state_root")
        for lane in lanes:
            if lane.submission_enabled and (
                not lane.planner_authority
                or lane.scientific_state is None
                or lane.report_contract is None
            ):
                raise ValueError(
                    f"submission-enabled lane {lane.lane_id!r} lacks its "
                    "planner/scientific/report binding"
                )
        operations_raw = payload.get("production_operations")
        if operations_raw is None:
            production_operations = None
        elif isinstance(operations_raw, Mapping):
            production_operations = ProductionOperationsBinding.from_mapping(
                operations_raw, repository_root=root
            )
        else:
            raise ValueError("production_operations must be a JSON object")
        blocked_raw = payload.get("production_operations_blocked")
        if blocked_raw is None:
            production_operations_blocked = None
        elif isinstance(blocked_raw, Mapping):
            production_operations_blocked = (
                BlockedProductionOperationsBinding.from_mapping(
                    blocked_raw, repository_root=root
                )
            )
        else:
            raise ValueError(
                "production_operations_blocked must be a JSON object"
            )
        if (
            production_operations is not None
            and production_operations_blocked is not None
        ):
            raise ValueError(
                "production operations cannot be both active and blocked"
            )
        scientific_contracts_raw = payload.get("scientific_contracts", {})
        if not isinstance(scientific_contracts_raw, Mapping):
            raise ValueError("scientific_contracts must be a JSON object")
        return cls(
            registry_id=str(payload.get("registry_id") or ""),
            repository_root=root,
            supervisor_state_root=root / state_raw,
            lanes=lanes,
            scientific_contracts=dict(scientific_contracts_raw),
            production_operations=production_operations,
            production_operations_blocked=production_operations_blocked,
            registry_path=registry_path.resolve(),
        )

    def lane(self, lane_id: str) -> LaneRegistration:
        try:
            return next(lane for lane in self.lanes if lane.lane_id == lane_id)
        except StopIteration as exc:
            raise KeyError(f"unregistered acquisition lane {lane_id!r}") from exc

    def lane_scientific_authority(
        self, lane_id: str
    ) -> LaneScientificAuthority:
        """Project one lane's immutable scientific configuration.

        Only the parsed registry is consulted. In particular, this method
        does not infer paths from naming conventions or inspect mutable output
        files on disk.
        """

        if not isinstance(lane_id, str) or not lane_id:
            raise ValueError("lane scientific authority requires a lane_id")
        try:
            lane = self.lane(lane_id)
        except KeyError as exc:
            raise ValueError(
                f"lane scientific authority names an unregistered lane: {lane_id}"
            ) from exc
        if not self.registry_id or not lane.planner_authority:
            raise ValueError(
                "lane scientific authority lacks registry/planner identity"
            )

        root = self.repository_root.resolve()

        def lane_relative(path: Path | None, *, label: str) -> str:
            if path is None or not path.is_absolute():
                raise ValueError(
                    f"lane scientific authority lacks an absolute {label}"
                )
            try:
                relative = path.relative_to(root)
                path.resolve(strict=False).relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"lane scientific authority {label} is outside repository_root"
                ) from exc
            if not relative.parts or ".." in relative.parts:
                raise ValueError(
                    f"lane scientific authority has invalid {label}"
                )
            return relative.as_posix()

        def contract_relative(value: Any, *, label: str) -> str:
            if not isinstance(value, str):
                raise ValueError(
                    f"lane scientific authority has malformed {label}"
                )
            path = Path(value)
            if (
                not path.parts
                or path.is_absolute()
                or ".." in path.parts
                or value != path.as_posix()
            ):
                raise ValueError(
                    f"lane scientific authority has invalid {label}"
                )
            try:
                (root / path).resolve(strict=False).relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"lane scientific authority {label} is outside repository_root"
                ) from exc
            return path.as_posix()

        contracts = self.scientific_contracts
        matrix = contracts.get("matrix")
        planned_all = contracts.get("planned_report_contracts")
        if not isinstance(matrix, Mapping) or not isinstance(
            planned_all, Mapping
        ):
            raise ValueError(
                "lane scientific authority lacks matrix/planned contracts"
            )
        planned = planned_all.get(lane_id)
        if not isinstance(planned, Mapping):
            raise ValueError(
                "lane scientific authority lacks its planned report contract"
            )
        declared_lane = planned.get("lane_id")
        if declared_lane not in (None, lane_id):
            raise ValueError(
                "lane scientific authority planned contract crosses lanes"
            )

        planned_path = contract_relative(
            planned.get("path"), label="planned report contract path"
        )
        for other_lane_id, other in planned_all.items():
            if (
                other_lane_id != lane_id
                and isinstance(other, Mapping)
                and other.get("path") == planned_path
            ):
                raise ValueError(
                    "lane scientific authority planned path crosses lanes"
                )

        def sha(mapping: Mapping[str, Any], key: str, *, label: str) -> str:
            value = mapping.get(key)
            if not isinstance(value, str) or not re.fullmatch(
                r"[0-9a-f]{64}", value
            ):
                raise ValueError(
                    f"lane scientific authority has malformed {label}"
                )
            return value

        planned_file_sha = sha(
            planned,
            "file_sha256",
            label="planned report contract file digest",
        )
        planned_self_sha = sha(
            planned,
            "report_contract_sha256",
            label="planned report contract self digest",
        )
        planned_tuple_sha = sha(
            planned,
            "tuple_structure_sha256",
            label="planned report tuple digest",
        )
        matrix_raw_sha = sha(matrix, "raw_sha256", label="matrix raw digest")
        matrix_payload_sha = sha(
            matrix, "payload_sha256", label="matrix payload digest"
        )
        matrix_expanded_sha = sha(
            matrix,
            "expanded_semantic_sha256",
            label="matrix expanded semantic digest",
        )
        matrix_policy_sha = sha(
            matrix,
            "accuracy_policy_spec_sha256",
            label="matrix accuracy-policy digest",
        )

        metadata_assertions = {
            "planned_report_contract_file_sha256": planned_file_sha,
            "planned_report_contract_sha256": planned_self_sha,
            "matrix_raw_sha256": matrix_raw_sha,
            "matrix_payload_sha256": matrix_payload_sha,
            "matrix_expanded_semantic_sha256": matrix_expanded_sha,
            "matrix_accuracy_policy_sha256": matrix_policy_sha,
        }
        for key, expected in metadata_assertions.items():
            if key in lane.metadata and lane.metadata.get(key) != expected:
                raise ValueError(
                    f"lane scientific authority {key} crosses contracts"
                )

        return LaneScientificAuthority(
            registry_id=self.registry_id,
            lane_id=lane.lane_id,
            planner_authority=lane.planner_authority,
            scientific_state_path=lane_relative(
                lane.scientific_state, label="scientific state path"
            ),
            report_contract_path=lane_relative(
                lane.report_contract, label="report contract path"
            ),
            planned_report_contract_path=planned_path,
            planned_report_contract_file_sha256=planned_file_sha,
            planned_report_contract_sha256=planned_self_sha,
            planned_report_tuple_structure_sha256=planned_tuple_sha,
            matrix_raw_sha256=matrix_raw_sha,
            matrix_payload_sha256=matrix_payload_sha,
            matrix_expanded_semantic_sha256=matrix_expanded_sha,
            matrix_accuracy_policy_sha256=matrix_policy_sha,
        )

    @property
    def digest(self) -> str:
        """Semantic registry digest used as an explicit configuration epoch."""

        root = self.repository_root.resolve()

        def relative(path: Path | None) -> str | None:
            if path is None:
                return None
            try:
                return path.resolve().relative_to(root).as_posix()
            except ValueError as exc:
                raise ValueError(
                    f"registry-bound path is outside repository_root: {path}"
                ) from exc

        active = self.production_operations
        blocked = self.production_operations_blocked
        return _digest(
            {
                "schema": REGISTRY_SCHEMA_V1,
                "registry_id": self.registry_id,
                "supervisor_state_root": relative(self.supervisor_state_root),
                "lanes": [
                    {
                        "lane_id": lane.lane_id,
                        "kind": lane.kind,
                        "monitor_enabled": lane.monitor_enabled,
                        "submission_enabled": lane.submission_enabled,
                        "attempts_root": relative(lane.attempts_root),
                        "state_root": relative(lane.state_root),
                        "scientific_state": relative(lane.scientific_state),
                        "report_contract": relative(lane.report_contract),
                        "planner_authority": lane.planner_authority,
                        "blocked_by": list(lane.blocked_by),
                        "metadata": dict(lane.metadata),
                    }
                    for lane in self.lanes
                ],
                "scientific_contracts": dict(self.scientific_contracts),
                "production_operations": (
                    None
                    if active is None
                    else {
                        "adapter_id": active.adapter_id,
                        "factory": active.factory,
                        "source": {
                            "path": relative(active.source_path),
                            "sha256": active.source_sha256,
                        },
                        "bootstrap_scopes": {
                            "path": relative(active.bootstrap_scopes_path),
                            "sha256": active.bootstrap_scopes_sha256,
                        },
                    }
                ),
                "production_operations_blocked": (
                    None
                    if blocked is None
                    else {
                        "status": "blocked",
                        "adapter_id": blocked.adapter_id,
                        "factory": blocked.factory,
                        "source": {
                            "path": relative(blocked.source_path),
                            "sha256": blocked.source_sha256,
                        },
                        "blocked_by": list(blocked.blocked_by),
                    }
                ),
            }
        )


@dataclass(frozen=True)
class SchedulerJob:
    lane_id: str
    cluster_id: int
    process_id: int
    state: str
    action_id: str | None = None
    exit_code: int | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SchedulerJob":
        state = str(payload.get("state") or "").lower()
        if state not in VALID_STATES:
            raise ValueError(
                f"scheduler state {state!r} is not one of {sorted(VALID_STATES)}"
            )
        lane_id = str(payload.get("lane_id") or "")
        if not lane_id:
            raise ValueError("scheduler job has no lane_id")
        action_id = (
            None
            if payload.get("action_id") in (None, "")
            else str(payload.get("action_id"))
        )
        if action_id is None:
            raise ValueError("scheduler job has no action_id")
        exit_raw = payload.get("exit_code")
        exit_code = None if exit_raw is None else int(exit_raw)
        if state == "succeeded" and exit_code not in (None, 0):
            raise ValueError("succeeded scheduler job has a nonzero exit_code")
        if state == "failed" and exit_code == 0:
            raise ValueError("failed scheduler job has exit_code zero")
        return cls(
            lane_id=lane_id,
            cluster_id=int(payload["cluster_id"]),
            process_id=int(payload["process_id"]),
            state=state,
            action_id=action_id,
            exit_code=exit_code,
            detail=dict(payload.get("detail") or {}),
        )

    @property
    def key(self) -> tuple[str, int, int]:
        return self.lane_id, self.cluster_id, self.process_id

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "cluster_id": self.cluster_id,
            "process_id": self.process_id,
            "state": self.state,
            "action_id": self.action_id,
            "exit_code": self.exit_code,
            "detail": dict(self.detail),
        }

    @property
    def fingerprint(self) -> str:
        return _digest(self.to_json_dict())


@dataclass(frozen=True)
class SchedulerSnapshot:
    captured_at: str
    jobs: tuple[SchedulerJob, ...]
    source: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SchedulerSnapshot":
        if payload.get("schema") != SNAPSHOT_SCHEMA_V1:
            raise ValueError("unsupported time-dynamics scheduler snapshot schema")
        rows = payload.get("jobs")
        if not isinstance(rows, list):
            raise ValueError("scheduler snapshot jobs must be a list")
        jobs = tuple(
            sorted(
                (SchedulerJob.from_mapping(row) for row in rows),
                key=lambda job: job.key,
            )
        )
        keys = [job.key for job in jobs]
        if len(set(keys)) != len(keys):
            raise ValueError("scheduler snapshot repeats a lane/cluster/process key")
        captured_at = str(payload.get("captured_at") or "")
        try:
            captured = _dt.datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("scheduler snapshot captured_at is not ISO-8601") from exc
        if captured.tzinfo is None:
            raise ValueError("scheduler snapshot captured_at must carry a timezone")
        source = dict(payload.get("source") or {})
        for check in ("queue_query_succeeded", "history_query_succeeded", "scope_complete"):
            if source.get(check) is not True:
                raise ValueError(f"scheduler snapshot source does not prove {check}")
        scopes = source.get("expected_processes")
        if not isinstance(scopes, list):
            raise ValueError("scheduler snapshot source has no expected_processes list")
        normalized_scopes: list[SchedulerScope] = []
        expected: set[tuple[str, int, int]] = set()
        for scope in scopes:
            if not isinstance(scope, Mapping):
                raise ValueError("expected_processes entries must be objects")
            normalized = SchedulerScope.from_mapping(scope)
            normalized_scopes.append(normalized)
            for process_id in normalized.process_ids:
                key = (normalized.lane_id, normalized.cluster_id, process_id)
                if key in expected:
                    raise ValueError("expected_processes repeats a scheduler key")
                expected.add(key)
        scope_keys = [(scope.lane_id, scope.cluster_id) for scope in normalized_scopes]
        if len(set(scope_keys)) != len(scope_keys):
            raise ValueError("expected_processes repeats a lane/cluster scope")
        if expected != set(keys):
            missing = sorted(expected - set(keys))
            extra = sorted(set(keys) - expected)
            raise ValueError(
                "scheduler snapshot is not scope-complete: "
                f"missing={missing}, extra={extra}"
            )
        return cls(captured_at=captured_at, jobs=jobs, source=source)

    @classmethod
    def read(cls, path: str | Path) -> "SchedulerSnapshot":
        return cls.from_mapping(_read_object(Path(path), label="scheduler snapshot"))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": SNAPSHOT_SCHEMA_V1,
            "captured_at": self.captured_at,
            "source": dict(self.source),
            "jobs": [job.to_json_dict() for job in self.jobs],
        }

    @property
    def digest(self) -> str:
        return _digest(self.to_json_dict())

    def jobs_for(self, lane_id: str) -> tuple[SchedulerJob, ...]:
        return tuple(job for job in self.jobs if job.lane_id == lane_id)

    @property
    def expected_scopes(self) -> tuple[SchedulerScope, ...]:
        rows = self.source.get("expected_processes")
        if not isinstance(rows, list):
            raise ValueError("scheduler snapshot source has no expected_processes list")
        if any(not isinstance(row, Mapping) for row in rows):
            raise ValueError("expected_processes entries must be objects")
        return tuple(
            sorted(SchedulerScope.from_mapping(row) for row in rows)
        )


@dataclass(frozen=True)
class SubmissionUnit:
    """One scheduler submission package, possibly containing several actions."""

    unit_id: str
    action_ids: tuple[str, ...]
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.unit_id or len(set(self.action_ids)) != len(self.action_ids):
            raise ValueError("submission unit needs an ID and unique action IDs")
        if not self.action_ids or any(not value for value in self.action_ids):
            raise ValueError("submission unit action_ids must be nonempty strings")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "action_ids": list(self.action_ids),
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class ReconciliationResult:
    """Proof result for one stable submission identity in one snapshot.

    Submission is permitted only when the result is ``verified_absent``.  An
    inconclusive query must be represented as ``indeterminate`` rather than
    overloaded onto a null value.
    """

    status: str
    remote: Mapping[str, Any] = field(default_factory=dict)
    detail: str = ""

    def __post_init__(self) -> None:
        if self.status not in RECONCILIATION_STATES:
            raise ValueError(f"unsupported reconciliation status {self.status!r}")
        if self.status == RECONCILIATION_FOUND and not self.remote:
            raise ValueError("found reconciliation requires remote identity")
        if self.status != RECONCILIATION_FOUND and self.remote:
            raise ValueError(
                f"{self.status} reconciliation cannot carry remote identity"
            )

    @classmethod
    def found(cls, remote: Mapping[str, Any]) -> "ReconciliationResult":
        return cls(status=RECONCILIATION_FOUND, remote=dict(remote))

    @classmethod
    def verified_absent(cls, detail: str = "") -> "ReconciliationResult":
        return cls(status=RECONCILIATION_VERIFIED_ABSENT, detail=str(detail))

    @classmethod
    def indeterminate(cls, detail: str = "") -> "ReconciliationResult":
        return cls(status=RECONCILIATION_INDETERMINATE, detail=str(detail))


@dataclass(frozen=True)
class ArtifactBinding:
    """One file path and digest returned by a scientific/report adapter."""

    path: Path
    sha256: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ArtifactBinding":
        path = Path(str(payload.get("path") or ""))
        digest = str(payload.get("sha256") or "")
        if not path.parts or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("artifact binding needs a path and lowercase SHA-256")
        return cls(path=path, sha256=digest)

    def to_json_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class SubmissionPlan:
    """Planner output bound to its registered authority and input artifacts."""

    units: tuple[SubmissionUnit, ...]
    planner_authority: str
    scientific_state: ArtifactBinding
    report_contract: ArtifactBinding

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "planner_authority": self.planner_authority,
            "scientific_state": self.scientific_state.to_json_dict(),
            "report_contract": self.report_contract.to_json_dict(),
            "units": [unit.to_json_dict() for unit in self.units],
        }


IngestDelta = Callable[
    [LaneRegistration, int, tuple[int, ...], tuple[SchedulerJob, ...]],
    Mapping[str, Any],
]
RefreshStatus = Callable[
    [LaneRegistration, SchedulerSnapshot, Mapping[str, Any]], Mapping[str, Any]
]
PlanSubmissions = Callable[
    [LaneRegistration, SchedulerSnapshot, Mapping[str, Any]], SubmissionPlan
]
ReconcileSubmission = Callable[
    [LaneRegistration, SubmissionUnit, SchedulerSnapshot], ReconciliationResult
]
SubmitUnit = Callable[[LaneRegistration, SubmissionUnit], Mapping[str, Any]]


@dataclass(frozen=True)
class LaneHandlers:
    ingest_delta: IngestDelta | None = None
    refresh_status: RefreshStatus | None = None
    plan_submissions: PlanSubmissions | None = None
    reconcile_submission: ReconcileSubmission | None = None
    submit_unit: SubmitUnit | None = None


class AcquisitionOperations(Protocol):
    """Production operations adapter at the scheduler/transport seam.

    The explicitly named factory may configure authentication and fixed remote
    roots, but the resulting adapter must expose its identity, independently
    capture exactly one combined snapshot for the independently registered
    durable scopes, and return the lane handlers that own the corresponding
    mutations.
    """

    adapter_id: str

    def capture_snapshot(
        self,
        registry: CampaignRegistry,
        *,
        expected_scopes: tuple[SchedulerScope, ...],
    ) -> SchedulerSnapshot: ...

    def handlers_for(
        self, registry: CampaignRegistry
    ) -> Mapping[str, LaneHandlers]: ...


class CycleGuardFailed(RuntimeError):
    """A retryable, durably reported supervisor guard."""

    def __init__(self, guard: str, detail: str = ""):
        super().__init__(f"{guard}" + (f": {detail}" if detail else ""))
        self.guard = str(guard)
        self.detail = str(detail)


class RemoteAccessPending(CycleGuardFailed):
    def __init__(self, detail: str = ""):
        super().__init__("remote_access_pending", detail)


class SupervisorLocked(RuntimeError):
    pass


class SupervisorLock:
    """Crash-releasing advisory lock for the complete multi-lane cycle."""

    def __init__(self, path: Path):
        self.path = path
        self._handle: Any = None

    def __enter__(self) -> "SupervisorLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "a+", encoding="utf-8")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._handle.close()
            self._handle = None
            raise SupervisorLocked(f"acquisition cycle is already running: {self.path}") from exc
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(json.dumps({"pid": os.getpid(), "acquired_at": _utc_now()}))
        self._handle.flush()
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


@dataclass(frozen=True)
class CycleResult:
    status: str
    snapshot_digest: str
    state_path: Path
    receipt_path: Path
    lane_results: Mapping[str, Any]


def _load_state(path: Path, *, registry_id: str) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema": STATE_SCHEMA_V1,
            "registry_id": registry_id,
            "lanes": {},
        }
    state = _read_object(path, label="acquisition supervisor state")
    if state.get("schema") != STATE_SCHEMA_V1:
        raise ValueError("unsupported acquisition supervisor state schema")
    if state.get("registry_id") != registry_id:
        raise ValueError("acquisition state belongs to a different registry")
    if not isinstance(state.get("lanes"), dict):
        raise ValueError("acquisition state lanes must be an object")
    return state


def _resolve_repository_file(
    repository_root: Path, raw_path: str | Path, *, label: str
) -> Path:
    """Resolve one receipt-owned artifact without permitting repository escape."""

    path = Path(raw_path)
    candidate = path if path.is_absolute() else repository_root / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(repository_root.resolve())
    except ValueError as exc:
        raise CycleGuardFailed(
            "artifact_path_escape", f"{label}: {raw_path}"
        ) from exc
    if not resolved.is_file():
        raise CycleGuardFailed("artifact_missing", f"{label}: {resolved}")
    return resolved


def _validate_artifact_binding(
    *,
    repository_root: Path,
    expected_path: Path | None,
    binding: ArtifactBinding,
    label: str,
) -> dict[str, str]:
    if expected_path is None:
        raise CycleGuardFailed("registry_binding_missing", label)
    actual = _resolve_repository_file(
        repository_root, binding.path, label=f"{label} binding"
    )
    if actual != expected_path.resolve():
        raise CycleGuardFailed(
            "artifact_binding_path_mismatch",
            f"{label}: expected {expected_path.resolve()}, got {actual}",
        )
    digest = _sha256_file(actual)
    if digest != binding.sha256:
        raise CycleGuardFailed(
            "artifact_binding_digest_mismatch",
            f"{label}: expected {binding.sha256}, got {digest}",
        )
    return {"path": str(actual), "sha256": digest}


def _receipt_paths(attempts_root: Path) -> tuple[Path, ...]:
    pattern = "cluster_*/increments/*/evidence/increment_receipt.json"
    return tuple(sorted(attempts_root.glob(pattern)))


def _load_receipt(path: Path) -> dict[str, Any]:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CycleGuardFailed(
            "increment_receipt_unreadable", f"{path}: {exc}"
        ) from exc
    if not isinstance(receipt, dict):
        raise CycleGuardFailed(
            "increment_receipt_unreadable", f"{path}: not a JSON object"
        )
    return dict(receipt)


def _receipt_claims_process(receipt: Mapping[str, Any], job: SchedulerJob) -> bool:
    try:
        if int(receipt.get("cluster_id")) != job.cluster_id:
            return False
    except (TypeError, ValueError):
        return False
    processes = receipt.get("process_ids")
    if isinstance(processes, (list, tuple)):
        return job.process_id in {int(value) for value in processes}
    process = receipt.get("process_id")
    return process is not None and int(process) == job.process_id


def _validate_receipt_claim(
    *,
    repository_root: Path,
    receipt_path: Path,
    receipt: Mapping[str, Any],
    job: SchedulerJob,
    scientific_state: Path | None,
) -> dict[str, str]:
    """Validate one Paper-II or SIAM coverage claim down to the run digest."""

    engine_v3 = receipt.get("schema") == INGESTION_RECEIPT_SCHEMA_V3
    processes = receipt.get("process_ids")
    if isinstance(processes, (list, tuple)):
        if engine_v3:
            if (
                receipt.get("engine") != INGESTION_ENGINE_V1
                or receipt.get("status") != "INGESTED"
            ):
                raise CycleGuardFailed(
                    "receipt_coverage_unverifiable",
                    f"{receipt_path}: invalid v3 engine/status binding",
                )
            ledger_append = receipt.get("ledger_append")
            if not isinstance(ledger_append, Mapping):
                raise CycleGuardFailed(
                    "receipt_coverage_unverifiable",
                    f"{receipt_path}: v3 receipt has no ledger_append payload",
                )
            dispositions = ledger_append.get("action_dispositions")
            state_raw = ledger_append.get("scientific_state_after")
            if not isinstance(state_raw, Mapping):
                raise CycleGuardFailed(
                    "receipt_scientific_state_binding_missing", str(receipt_path)
                )
            try:
                state_binding = ArtifactBinding.from_mapping(state_raw)
            except ValueError as exc:
                raise CycleGuardFailed(
                    "receipt_scientific_state_binding_unverifiable",
                    f"{receipt_path}: {exc}",
                ) from exc
            if scientific_state is None:
                raise CycleGuardFailed(
                    "registry_binding_missing",
                    f"{job.lane_id} receipt scientific state",
                )
            # This digest records the ledger immediately after this increment.
            # Later successful increments advance that same canonical file, so
            # replay validates the registered path and the receipt's internal
            # digest agreement rather than comparing historical bytes with the
            # current ledger generation.
            bound_state = _resolve_repository_file(
                repository_root,
                state_binding.path,
                label=f"{job.lane_id} receipt scientific state binding",
            )
            if bound_state != scientific_state.resolve():
                raise CycleGuardFailed(
                    "artifact_binding_path_mismatch",
                    (
                        f"{job.lane_id} receipt scientific state: expected "
                        f"{scientific_state.resolve()}, got {bound_state}"
                    ),
                )
            if ledger_append.get("ledger_after_sha256") != state_binding.sha256:
                raise CycleGuardFailed(
                    "receipt_scientific_state_digest_mismatch",
                    f"{receipt_path}: ledger_after_sha256 differs from its binding",
                )
        else:
            dispositions = receipt.get("action_dispositions")
        if not isinstance(dispositions, list):
            raise CycleGuardFailed(
                "receipt_coverage_unverifiable",
                f"{receipt_path}: process_ids has no action_dispositions",
            )
        rows = [
            row
            for row in dispositions
            if isinstance(row, Mapping)
            and int(row.get("process_id", -1)) == job.process_id
        ]
        if len(rows) != 1:
            raise CycleGuardFailed(
                "receipt_coverage_unverifiable",
                f"{receipt_path}: expected one disposition for proc {job.process_id}",
            )
        row = rows[0]
        action_id = str(row.get("action_id") or "")
        artifact_raw = row.get("run_path")
        artifact_sha = str(row.get("run_sha256") or "")
    else:
        action_id = str(receipt.get("action_id") or "")
        artifacts = [
            value
            for value in (receipt.get("run"), receipt.get("raw_continuation"))
            if isinstance(value, Mapping)
            and value.get("path")
            and value.get("sha256")
        ]
        if len(artifacts) != 1:
            raise CycleGuardFailed(
                "receipt_coverage_unverifiable",
                f"{receipt_path}: expected one verified SIAM run artifact",
            )
        artifact_raw = artifacts[0].get("path")
        artifact_sha = str(artifacts[0].get("sha256") or "")

    if action_id != job.action_id:
        raise CycleGuardFailed(
            "receipt_action_identity_mismatch",
            f"{receipt_path}: scheduler={job.action_id}, receipt={action_id}",
        )
    if not re.fullmatch(r"[0-9a-f]{64}", artifact_sha):
        raise CycleGuardFailed(
            "receipt_coverage_unverifiable",
            f"{receipt_path}: invalid run SHA-256",
        )
    artifact = _resolve_repository_file(
        repository_root,
        str(artifact_raw or ""),
        label=f"covered run {job.cluster_id}.{job.process_id}",
    )
    actual_sha = _sha256_file(artifact)
    if actual_sha != artifact_sha:
        raise CycleGuardFailed(
            "receipt_run_digest_mismatch",
            f"{receipt_path}: expected {artifact_sha}, got {actual_sha}",
        )
    extraction_root = (
        receipt_path.parent / "extracted"
        if engine_v3
        else receipt_path.parents[1] / "extracted"
    ).resolve()
    if not artifact.is_relative_to(extraction_root):
        raise CycleGuardFailed(
            "receipt_run_outside_increment",
            f"{artifact} is not beneath {extraction_root}",
        )
    if artifact.stat().st_size <= 0:
        raise CycleGuardFailed("receipt_run_empty", str(artifact))
    try:
        run_payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CycleGuardFailed("receipt_run_unreadable", f"{artifact}: {exc}") from exc
    if not isinstance(run_payload, Mapping) or artifact.parent.name != action_id:
        raise CycleGuardFailed(
            "receipt_run_action_path_mismatch", f"{artifact}: {action_id}"
        )
    source_revision = str(receipt.get("source_revision") or "")
    run_revision = str(
        (run_payload.get("run_lock") or {}).get("code_revision")
        if isinstance(run_payload.get("run_lock"), Mapping)
        else ""
    )
    if source_revision and run_revision != source_revision:
        raise CycleGuardFailed(
            "receipt_run_source_revision_mismatch",
            f"{artifact}: receipt={source_revision}, run={run_revision}",
        )
    return {
        "receipt": str(receipt_path),
        "action_id": action_id,
        "run_path": str(artifact),
        "run_sha256": actual_sha,
    }


def _validated_coverage_for_jobs(
    *,
    repository_root: Path,
    lane: LaneRegistration,
    jobs: Sequence[SchedulerJob],
) -> dict[tuple[str, int, int], dict[str, str]]:
    receipt_rows = tuple(
        (path, _load_receipt(path)) for path in _receipt_paths(lane.attempts_root)
    )
    normalized: dict[tuple[str, int, int], dict[str, str]] = {}
    index_raw = lane.metadata.get("legacy_coverage_index")
    if index_raw:
        if lane.scientific_state is None:
            raise CycleGuardFailed(
                "legacy_coverage_unverifiable",
                f"{lane.lane_id}: no registered scientific state",
            )
        index_path = _resolve_repository_file(
            repository_root,
            str(index_raw),
            label=f"{lane.lane_id} legacy coverage index",
        )
        expected_index_sha = str(
            lane.metadata.get("legacy_coverage_index_sha256") or ""
        )
        if expected_index_sha and _sha256_file(index_path) != expected_index_sha:
            raise CycleGuardFailed(
                "legacy_coverage_unverifiable",
                f"{index_path}: registry digest binding changed",
            )
        index = _load_receipt(index_path)
        decision_digest = str(lane.metadata.get("decision_handoff_sha256") or "")
        try:
            legacy = validate_legacy_coverage_index(
                index,
                repository_root=repository_root,
                lane_id=lane.lane_id,
                scientific_state=lane.scientific_state,
                decision_handoff_sha256=decision_digest,
            )
        except LegacyCoverageError as exc:
            raise CycleGuardFailed(
                "legacy_coverage_unverifiable", f"{index_path}: {exc}"
            ) from exc
        normalized = {
            (lane.lane_id, cluster_id, process_id): row
            for (cluster_id, process_id), row in legacy.items()
        }
    coverage: dict[tuple[str, int, int], dict[str, str]] = {}
    for job in jobs:
        claims = [
            (path, receipt)
            for path, receipt in receipt_rows
            if _receipt_claims_process(receipt, job)
        ]
        normalized_row = normalized.get(job.key)
        # A verified normalization row replaces only weak pre-v3 claims.  New
        # engine-v3 and singular SIAM receipts remain independently validated,
        # and any disagreement between the two authorities fails closed.
        strong_claims = [
            (path, receipt)
            for path, receipt in claims
            if receipt.get("schema") == INGESTION_RECEIPT_SCHEMA_V3
            or not isinstance(receipt.get("process_ids"), (list, tuple))
        ]
        selected_claims = strong_claims if normalized_row is not None else claims
        if not selected_claims and normalized_row is None:
            continue
        validated = [
            _validate_receipt_claim(
                repository_root=repository_root,
                receipt_path=path,
                receipt=receipt,
                job=job,
                scientific_state=lane.scientific_state,
            )
            for path, receipt in selected_claims
        ]
        if normalized_row is not None:
            if normalized_row["action_id"] != job.action_id:
                raise CycleGuardFailed(
                    "receipt_action_identity_mismatch",
                    (
                        f"legacy index {job.cluster_id}.{job.process_id}: "
                        f"scheduler={job.action_id}, index={normalized_row['action_id']}"
                    ),
                )
            validated.append(normalized_row)
        identities = {
            (row["action_id"], row["run_path"], row["run_sha256"])
            for row in validated
        }
        if len(identities) != 1:
            raise CycleGuardFailed(
                "duplicate_receipt_coverage_conflict",
                f"{job.cluster_id}.{job.process_id} has inconsistent receipts",
            )
        coverage[job.key] = validated[0]
    return coverage


def _snapshot_transition_guard(
    state: Mapping[str, Any], snapshot: SchedulerSnapshot, registry: CampaignRegistry
) -> tuple[str, str] | None:
    prior_registry = state.get("registry_sha256")
    if prior_registry not in (None, registry.digest):
        return (
            "registry_transition_requires_migration",
            f"stored={prior_registry}, current={registry.digest}",
        )
    prior_time_raw = state.get("last_snapshot_captured_at")
    prior_jobs_raw = state.get("last_snapshot_jobs")
    if prior_time_raw is None and prior_jobs_raw is None:
        return None
    if not isinstance(prior_time_raw, str) or not isinstance(prior_jobs_raw, list):
        return "supervisor_state_unverifiable", "missing prior snapshot watermark"
    prior_time = _dt.datetime.fromisoformat(prior_time_raw.replace("Z", "+00:00"))
    current_time = _dt.datetime.fromisoformat(snapshot.captured_at.replace("Z", "+00:00"))
    if current_time < prior_time:
        return "scheduler_snapshot_regressed", f"{snapshot.captured_at} < {prior_time_raw}"
    prior_jobs = {
        SchedulerJob.from_mapping(row).key: SchedulerJob.from_mapping(row)
        for row in prior_jobs_raw
        if isinstance(row, Mapping)
    }
    current_jobs = {job.key: job for job in snapshot.jobs}
    if current_time == prior_time and _digest(prior_jobs_raw) != _digest(
        [job.to_json_dict() for job in snapshot.jobs]
    ):
        return "scheduler_snapshot_timestamp_collision", snapshot.captured_at
    vanished_active = sorted(
        key
        for key, job in prior_jobs.items()
        if job.state in ACTIVE_STATES and key not in current_jobs
    )
    if vanished_active:
        return "active_scheduler_jobs_disappeared", repr(vanished_active)
    regressions = sorted(
        key
        for key, old in prior_jobs.items()
        if old.state in TERMINAL_STATES
        and key in current_jobs
        and current_jobs[key].state != old.state
    )
    if regressions:
        return "terminal_scheduler_state_regressed", repr(regressions)
    return None


def _job_state_digest(jobs: Iterable[SchedulerJob]) -> str:
    """Digest material transitions, excluding snapshot capture time."""

    return _digest([job.to_json_dict() for job in sorted(jobs, key=lambda row: row.key)])


def _record_failure_disposition(
    lane: LaneRegistration, job: SchedulerJob
) -> tuple[Path, bool]:
    root = (
        lane.state_root
        / "scheduler_dispositions"
        / f"cluster_{job.cluster_id}"
    )
    path = root / f"proc{job.process_id}__{job.fingerprint[:16]}.json"
    payload = {
        "schema": FAILURE_DISPOSITION_SCHEMA_V1,
        "lane_id": lane.lane_id,
        "cluster_id": job.cluster_id,
        "process_id": job.process_id,
        "action_id": job.action_id,
        "scheduler_state": job.state,
        "exit_code": job.exit_code,
        "detail": dict(job.detail),
        "job_fingerprint": job.fingerprint,
        "recorded_at": _utc_now(),
    }
    if path.exists():
        existing = _read_object(path, label="terminal failure disposition")
        comparable = dict(existing)
        comparable.pop("recorded_at", None)
        expected = dict(payload)
        expected.pop("recorded_at", None)
        if comparable != expected:
            raise ValueError(f"failure disposition identity collision: {path}")
        return path, False
    _atomic_write_json(path, payload)
    return path, True


def _intent_paths(lane: LaneRegistration, unit: SubmissionUnit) -> tuple[Path, Path]:
    name = _stable_filename(unit.unit_id)
    return (
        lane.state_root / "submission_intents" / f"{name}.json",
        lane.state_root / "submission_receipts" / f"{name}.json",
    )


def _ensure_submission_intent(
    lane: LaneRegistration, unit: SubmissionUnit
) -> tuple[Path, dict[str, Any]]:
    path, _ = _intent_paths(lane, unit)
    core = {
        "schema": SUBMISSION_INTENT_SCHEMA_V1,
        "lane_id": lane.lane_id,
        "unit": unit.to_json_dict(),
    }
    intent = {**core, "intent_sha256": _digest(core), "recorded_at": _utc_now()}
    if path.exists():
        existing = _read_object(path, label="submission intent")
        if existing.get("intent_sha256") != intent["intent_sha256"]:
            raise ValueError(
                f"submission unit {unit.unit_id!r} changed after its intent was recorded"
            )
        return path, existing
    _atomic_write_json(path, intent)
    return path, intent


def _unit_from_intent(
    lane: LaneRegistration, path: Path
) -> tuple[SubmissionUnit, dict[str, Any]]:
    intent = _read_object(path, label="submission intent")
    unit_raw = intent.get("unit")
    if (
        intent.get("schema") != SUBMISSION_INTENT_SCHEMA_V1
        or intent.get("lane_id") != lane.lane_id
        or not isinstance(unit_raw, Mapping)
    ):
        raise CycleGuardFailed("submission_intent_unverifiable", str(path))
    try:
        unit = SubmissionUnit(
            unit_id=str(unit_raw.get("unit_id") or ""),
            action_ids=tuple(str(value) for value in unit_raw.get("action_ids") or ()),
            payload=dict(unit_raw.get("payload") or {}),
        )
    except (TypeError, ValueError) as exc:
        raise CycleGuardFailed(
            "submission_intent_unverifiable", f"{path}: {exc}"
        ) from exc
    core = {
        "schema": SUBMISSION_INTENT_SCHEMA_V1,
        "lane_id": lane.lane_id,
        "unit": unit.to_json_dict(),
    }
    if intent.get("intent_sha256") != _digest(core):
        raise CycleGuardFailed("submission_intent_digest_mismatch", str(path))
    return unit, intent


def _validate_submission_receipt(
    lane: LaneRegistration,
    unit: SubmissionUnit,
    *,
    intent: Mapping[str, Any],
    path: Path,
) -> dict[str, Any]:
    receipt = _read_object(path, label="submission receipt")
    expected_actions = list(unit.action_ids)
    if (
        receipt.get("schema") != SUBMISSION_RECEIPT_SCHEMA_V1
        or receipt.get("lane_id") != lane.lane_id
        or receipt.get("unit_id") != unit.unit_id
        or receipt.get("action_ids") != expected_actions
        or receipt.get("intent_sha256") != intent.get("intent_sha256")
        or not isinstance(receipt.get("remote"), Mapping)
    ):
        raise CycleGuardFailed(
            "submission_receipt_unverifiable", str(path)
        )
    _validated_remote_identity(unit, receipt["remote"])
    return receipt


def _validated_remote_identity(
    unit: SubmissionUnit, remote: Mapping[str, Any]
) -> dict[str, Any]:
    row = dict(remote)
    if row.get("action_ids") != list(unit.action_ids) or not any(
        row.get(field) not in (None, "")
        for field in ("cluster_id", "scheduler_identity")
    ):
        raise CycleGuardFailed(
            "submitted_identity_missing",
            f"unit {unit.unit_id!r} lacks exact action IDs/scheduler identity",
        )
    return row


def _record_submission_receipt(
    lane: LaneRegistration,
    unit: SubmissionUnit,
    *,
    intent: Mapping[str, Any],
    remote: Mapping[str, Any],
    recovered: bool,
) -> Path:
    _, path = _intent_paths(lane, unit)
    payload = {
        "schema": SUBMISSION_RECEIPT_SCHEMA_V1,
        "lane_id": lane.lane_id,
        "unit_id": unit.unit_id,
        "action_ids": list(unit.action_ids),
        "intent_sha256": str(intent["intent_sha256"]),
        "recovered_from_scheduler": bool(recovered),
        "remote": dict(remote),
        "recorded_at": _utc_now(),
    }
    if path.exists():
        _validate_submission_receipt(
            lane, unit, intent=intent, path=path
        )
        return path
    _atomic_write_json(path, payload)
    return path


def _blocked(guard: str, detail: str = "") -> dict[str, Any]:
    return {"status": "blocked", "guard": guard, "detail": detail}


def _verified_refresh_outcome(
    *,
    registry: CampaignRegistry,
    lane: LaneRegistration,
    outcome: Mapping[str, Any],
) -> dict[str, Any]:
    row = dict(outcome)
    if row.get("status") not in {"refreshed", "unchanged"}:
        return row
    binding_raw = row.get("report_contract")
    if not isinstance(binding_raw, Mapping):
        raise CycleGuardFailed(
            "report_contract_binding_missing", lane.lane_id
        )
    binding = ArtifactBinding.from_mapping(binding_raw)
    row["report_contract"] = _validate_artifact_binding(
        repository_root=registry.repository_root,
        expected_path=lane.report_contract,
        binding=binding,
        label=f"{lane.lane_id} report contract",
    )
    return row


def _verified_submission_plan(
    *,
    registry: CampaignRegistry,
    lane: LaneRegistration,
    plan: SubmissionPlan,
) -> tuple[SubmissionUnit, ...]:
    if plan.planner_authority != lane.planner_authority:
        raise CycleGuardFailed(
            "planner_authority_mismatch",
            f"expected {lane.planner_authority}, got {plan.planner_authority}",
        )
    _validate_artifact_binding(
        repository_root=registry.repository_root,
        expected_path=lane.scientific_state,
        binding=plan.scientific_state,
        label=f"{lane.lane_id} scientific state",
    )
    _validate_artifact_binding(
        repository_root=registry.repository_root,
        expected_path=lane.report_contract,
        binding=plan.report_contract,
        label=f"{lane.lane_id} planner report contract",
    )
    return tuple(plan.units)


def run_acquisition_cycle(
    *,
    registry: CampaignRegistry,
    snapshot: SchedulerSnapshot,
    handlers: Mapping[str, LaneHandlers],
    allow_submission: bool = True,
) -> CycleResult:
    """Execute or resume one combined snapshot-to-report cycle.

    The supplied snapshot must contain only registered lanes.  Disabled lanes
    are retained in the cycle receipt but never ingested, planned, or launched.
    Enabled lanes fail closed when a required handler is absent.  Setting
    ``allow_submission=False`` is the replay safety interlock: ingestion and
    reporting may run, but no planner, reconciler, or submitter is invoked.
    """

    unknown_lanes = sorted(
        {job.lane_id for job in snapshot.jobs}
        - {lane.lane_id for lane in registry.lanes}
    )
    if unknown_lanes:
        raise ValueError(f"snapshot names unregistered lanes: {unknown_lanes}")

    state_root = registry.supervisor_state_root
    state_path = state_root / "state.json"
    snapshot_path = state_root / "snapshots" / f"{snapshot.digest}.json"
    cycle_dir = state_root / "cycles" / snapshot.digest

    def finish(
        *,
        state: dict[str, Any],
        lane_results: Mapping[str, Any],
        any_work: bool,
        any_blocked: bool,
        accept_snapshot: bool,
    ) -> CycleResult:
        status = "blocked" if any_blocked else ("complete" if any_work else "noop")
        if accept_snapshot:
            state.update(
                {
                    "registry_sha256": registry.digest,
                    "last_snapshot_sha256": snapshot.digest,
                    "last_snapshot_captured_at": snapshot.captured_at,
                    "last_snapshot_jobs": [
                        job.to_json_dict() for job in snapshot.jobs
                    ],
                }
            )
        else:
            state["last_blocked_snapshot_sha256"] = snapshot.digest
        state.update(
            {
                "schema": STATE_SCHEMA_V1,
                "registry_id": registry.registry_id,
                "last_cycle_at": _utc_now(),
            }
        )
        _atomic_write_json(state_path, state)
        cycle_receipt = {
            "schema": CYCLE_RECEIPT_SCHEMA_V1,
            "registry_id": registry.registry_id,
            "registry_sha256": registry.digest,
            "snapshot": str(snapshot_path),
            "snapshot_sha256": snapshot.digest,
            "status": status,
            "lanes": dict(lane_results),
            "recorded_at": _utc_now(),
        }
        cycle_dir.mkdir(parents=True, exist_ok=True)
        attempt_number = len(tuple(cycle_dir.glob("attempt_*.json"))) + 1
        receipt_path = cycle_dir / f"attempt_{attempt_number:04d}.json"
        _atomic_write_json(receipt_path, cycle_receipt)
        return CycleResult(
            status=status,
            snapshot_digest=snapshot.digest,
            state_path=state_path,
            receipt_path=receipt_path,
            lane_results=dict(lane_results),
        )

    with SupervisorLock(state_root / "acquisition.lock"):
        if not snapshot_path.exists():
            _atomic_write_json(snapshot_path, snapshot.to_json_dict())
        elif (
            _read_object(snapshot_path, label="stored scheduler snapshot")
            != snapshot.to_json_dict()
        ):
            raise ValueError(f"scheduler snapshot digest collision: {snapshot_path}")

        state = _load_state(state_path, registry_id=registry.registry_id)
        snapshot_guard = _snapshot_transition_guard(state, snapshot, registry)
        if snapshot_guard is not None:
            guard, detail = snapshot_guard
            blocked_results = {
                lane.lane_id: {
                    "kind": lane.kind,
                    "status": (
                        "blocked" if lane.monitor_enabled else "launch_disabled"
                    ),
                    "snapshot_validation": _blocked(guard, detail),
                }
                for lane in registry.lanes
            }
            state["last_blocked_guard"] = {"guard": guard, "detail": detail}
            return finish(
                state=state,
                lane_results=blocked_results,
                any_work=True,
                any_blocked=any(lane.monitor_enabled for lane in registry.lanes),
                accept_snapshot=False,
            )

        lane_state = dict(state.get("lanes") or {})
        lane_results: dict[str, Any] = {}
        any_work = False
        any_blocked = False

        for lane in registry.lanes:
            jobs = snapshot.jobs_for(lane.lane_id)
            jobs_digest = _job_state_digest(jobs)
            previous = dict(lane_state.get(lane.lane_id) or {})
            material_transition = previous.get("scheduler_jobs_sha256") != jobs_digest
            submission_execution_enabled = bool(
                lane.submission_enabled and allow_submission
            )
            result: dict[str, Any] = {
                "kind": lane.kind,
                "monitor_enabled": lane.monitor_enabled,
                "submission_enabled": lane.submission_enabled,
                "submission_execution_enabled": submission_execution_enabled,
                "scheduler_jobs_sha256": jobs_digest,
                "material_scheduler_transition": material_transition,
                "job_counts": {
                    state_name: sum(job.state == state_name for job in jobs)
                    for state_name in sorted(VALID_STATES)
                },
                "durable_coverage": [],
                "ingestion": [],
                "failure_dispositions": [],
                "submissions": [],
                "blocked_by": list(lane.blocked_by),
            }
            if not lane.monitor_enabled:
                if jobs:
                    result["status"] = "blocked"
                    result["monitoring_guard"] = _blocked(
                        "unmonitored_lane_has_scheduler_rows",
                        repr([job.key for job in jobs]),
                    )
                    any_blocked = True
                    any_work = True
                else:
                    result["status"] = "launch_disabled"
                lane_results[lane.lane_id] = result
                lane_state[lane.lane_id] = {
                    "scheduler_jobs_sha256": jobs_digest,
                    "last_snapshot_sha256": snapshot.digest,
                    "status": result["status"],
                }
                continue

            lane_handlers = handlers.get(lane.lane_id, LaneHandlers())
            lane_blocked = False
            new_failure_disposition = False
            successful = tuple(job for job in jobs if job.state == "succeeded")
            try:
                coverage = _validated_coverage_for_jobs(
                    repository_root=registry.repository_root,
                    lane=lane,
                    jobs=successful,
                )
            except CycleGuardFailed as exc:
                coverage = {}
                result["ingestion"].append(_blocked(exc.guard, exc.detail))
                lane_blocked = True
            else:
                result["durable_coverage"] = [
                    coverage[key] for key in sorted(coverage)
                ]

            by_cluster: dict[int, list[SchedulerJob]] = {}
            if not lane_blocked:
                for job in successful:
                    if job.key not in coverage:
                        by_cluster.setdefault(job.cluster_id, []).append(job)

            for cluster_id, fresh_jobs in sorted(by_cluster.items()):
                fresh = tuple(sorted(job.process_id for job in fresh_jobs))
                selected = tuple(sorted(fresh_jobs, key=lambda job: job.process_id))
                any_work = True
                if lane_handlers.ingest_delta is None:
                    outcome = _blocked(
                        "missing_ingestion_handler",
                        f"cluster {cluster_id} processes {list(fresh)}",
                    )
                else:
                    try:
                        outcome = dict(
                            lane_handlers.ingest_delta(
                                lane, cluster_id, fresh, selected
                            )
                        )
                    except CycleGuardFailed as exc:
                        outcome = _blocked(exc.guard, exc.detail)
                    except Exception as exc:
                        outcome = _blocked("ingestion_handler_error", str(exc))
                if outcome.get("status") in {"ingested", "already_ingested"}:
                    try:
                        post_coverage = _validated_coverage_for_jobs(
                            repository_root=registry.repository_root,
                            lane=lane,
                            jobs=selected,
                        )
                        missing = [job.key for job in selected if job.key not in post_coverage]
                        if missing:
                            raise CycleGuardFailed(
                                "ingestion_receipt_missing_or_mismatched",
                                repr(missing),
                            )
                    except CycleGuardFailed as exc:
                        outcome = _blocked(exc.guard, exc.detail)
                    else:
                        outcome["verified_coverage"] = [
                            post_coverage[job.key] for job in selected
                        ]
                outcome.update({"cluster_id": cluster_id, "process_ids": list(fresh)})
                result["ingestion"].append(outcome)
                if outcome.get("status") not in {"ingested", "already_ingested"}:
                    lane_blocked = True

            try:
                final_coverage = _validated_coverage_for_jobs(
                    repository_root=registry.repository_root,
                    lane=lane,
                    jobs=successful,
                )
            except CycleGuardFailed as exc:
                final_coverage = coverage
                result["ingestion"].append(_blocked(exc.guard, exc.detail))
                lane_blocked = True
            result["durable_coverage"] = [
                final_coverage[key] for key in sorted(final_coverage)
            ]
            coverage_digest = _digest(
                [
                    {"job_key": list(key), **final_coverage[key]}
                    for key in sorted(final_coverage)
                ]
            )
            coverage_digest_changed = (
                previous.get("durable_coverage_sha256") != coverage_digest
            )
            result["durable_coverage_sha256"] = coverage_digest
            result["durable_coverage_changed"] = coverage_digest_changed

            for job in jobs:
                if job.state not in {"failed", "removed"}:
                    continue
                disposition, created = _record_failure_disposition(lane, job)
                result["failure_dispositions"].append(str(disposition))
                new_failure_disposition = new_failure_disposition or created
                if created:
                    any_work = True

            needs_refresh = (
                material_transition
                or coverage_digest_changed
                or bool(result["ingestion"])
                or new_failure_disposition
                or bool(previous.get("report_refresh_required", False))
            )
            refresh_context = {
                "phase": "evidence_refresh",
                "snapshot_sha256": snapshot.digest,
                "material_scheduler_transition": material_transition,
                "ingestion": list(result["ingestion"]),
                "failure_dispositions": list(result["failure_dispositions"]),
                "gate_pending": lane_blocked,
            }
            if needs_refresh:
                any_work = True
                if lane_handlers.refresh_status is None:
                    refresh = _blocked("missing_report_refresh_handler")
                else:
                    try:
                        refresh = _verified_refresh_outcome(
                            registry=registry,
                            lane=lane,
                            outcome=lane_handlers.refresh_status(
                                lane, snapshot, refresh_context
                            ),
                        )
                    except CycleGuardFailed as exc:
                        refresh = _blocked(exc.guard, exc.detail)
                    except Exception as exc:
                        refresh = _blocked("report_refresh_handler_error", str(exc))
                result["report_refresh"] = refresh
                if refresh.get("status") not in {"refreshed", "unchanged"}:
                    lane_blocked = True

            unresolved_intents: dict[str, tuple[SubmissionUnit, dict[str, Any]]] = {}
            reconciled_intents: dict[str, ReconciliationResult] = {}
            durable_action_owners: dict[str, SubmissionUnit] = {}
            if submission_execution_enabled and not lane_blocked:
                intent_root = lane.state_root / "submission_intents"
                for intent_path in sorted(intent_root.glob("*.json")):
                    try:
                        unit, intent = _unit_from_intent(lane, intent_path)
                        for action_id in unit.action_ids:
                            owner = durable_action_owners.get(action_id)
                            if owner is not None and owner.to_json_dict() != unit.to_json_dict():
                                raise CycleGuardFailed(
                                    "durable_action_ownership_conflict",
                                    (
                                        f"action {action_id!r} is owned by both "
                                        f"{owner.unit_id!r} and {unit.unit_id!r}"
                                    ),
                                )
                            durable_action_owners[action_id] = unit
                        expected_intent, receipt_path = _intent_paths(lane, unit)
                        if expected_intent != intent_path:
                            raise CycleGuardFailed(
                                "submission_intent_filename_mismatch", str(intent_path)
                            )
                        if receipt_path.exists():
                            _validate_submission_receipt(
                                lane, unit, intent=intent, path=receipt_path
                            )
                            continue
                        if lane_handlers.reconcile_submission is None:
                            raise CycleGuardFailed("missing_submission_reconciler")
                        reconciliation = lane_handlers.reconcile_submission(
                            lane, unit, snapshot
                        )
                        if not isinstance(reconciliation, ReconciliationResult):
                            raise CycleGuardFailed(
                                "submission_reconciliation_untyped",
                                f"unit {unit.unit_id!r}",
                            )
                        reconciled_intents[unit.unit_id] = reconciliation
                        if reconciliation.status == RECONCILIATION_INDETERMINATE:
                            unresolved_intents[unit.unit_id] = (unit, intent)
                            continue
                        if reconciliation.status == RECONCILIATION_VERIFIED_ABSENT:
                            continue
                        remote = _validated_remote_identity(
                            unit, reconciliation.remote
                        )
                        receipt = _record_submission_receipt(
                            lane,
                            unit,
                            intent=intent,
                            remote=remote,
                            recovered=True,
                        )
                        result["submissions"].append(
                            {
                                "status": "recovered",
                                "unit_id": unit.unit_id,
                                "intent": str(intent_path),
                                "receipt": str(receipt),
                            }
                        )
                        any_work = True
                    except CycleGuardFailed as exc:
                        result["submissions"].append(_blocked(exc.guard, exc.detail))
                        lane_blocked = True
                        break
                    except Exception as exc:
                        result["submissions"].append(
                            _blocked("submission_reconcile_error", str(exc))
                        )
                        lane_blocked = True
                        break

            units: tuple[SubmissionUnit, ...] = ()
            if submission_execution_enabled and not lane_blocked:
                if lane_handlers.plan_submissions is None:
                    result["planning"] = _blocked("missing_submission_planner")
                    lane_blocked = True
                else:
                    plan_context = {
                        "snapshot_sha256": snapshot.digest,
                        "ingestion": list(result["ingestion"]),
                        "failure_dispositions": list(result["failure_dispositions"]),
                        "recovered_submissions": list(result["submissions"]),
                    }
                    try:
                        plan = lane_handlers.plan_submissions(
                            lane, snapshot, plan_context
                        )
                        if not isinstance(plan, SubmissionPlan):
                            raise CycleGuardFailed(
                                "submission_plan_unbound",
                                "planner did not return SubmissionPlan",
                            )
                        units = _verified_submission_plan(
                            registry=registry, lane=lane, plan=plan
                        )
                    except CycleGuardFailed as exc:
                        result["planning"] = _blocked(exc.guard, exc.detail)
                        lane_blocked = True
                    except Exception as exc:
                        result["planning"] = _blocked(
                            "submission_planner_error", str(exc)
                        )
                        lane_blocked = True

            if submission_execution_enabled and not lane_blocked:
                unit_ids = [unit.unit_id for unit in units]
                action_ids = [action for unit in units for action in unit.action_ids]
                if (
                    len(set(unit_ids)) != len(unit_ids)
                    or len(set(action_ids)) != len(action_ids)
                ):
                    result["planning"] = _blocked(
                        "duplicate_planned_identity",
                        "planner repeated a unit or action ID",
                    )
                    lane_blocked = True
                else:
                    planned = {unit.unit_id: unit for unit in units}
                    ownership_conflicts = [
                        {
                            "action_id": action_id,
                            "owner_unit_id": durable_action_owners[action_id].unit_id,
                            "planned_unit_id": unit.unit_id,
                        }
                        for unit in units
                        for action_id in unit.action_ids
                        if action_id in durable_action_owners
                        and durable_action_owners[action_id].to_json_dict()
                        != unit.to_json_dict()
                    ]
                    if ownership_conflicts:
                        result["planning"] = _blocked(
                            "action_already_owned_by_other_unit",
                            repr(ownership_conflicts),
                        )
                        lane_blocked = True
                        planned = {}
                    stale = [
                        unit_id
                        for unit_id, (unit, _intent) in unresolved_intents.items()
                        if unit_id not in planned
                        or planned[unit_id].to_json_dict() != unit.to_json_dict()
                    ]
                    if stale and not lane_blocked:
                        result["planning"] = _blocked(
                            "unresolved_submission_intent_not_replanned",
                            repr(stale),
                        )
                        lane_blocked = True
                    elif not lane_blocked:
                        result["planning"] = {
                            "status": "planned",
                            "planner_authority": lane.planner_authority,
                            "unit_ids": unit_ids,
                            "action_ids": action_ids,
                        }

            if submission_execution_enabled and not lane_blocked:
                for unit in units:
                    try:
                        intent_path, intent = _ensure_submission_intent(lane, unit)
                        _, submission_path = _intent_paths(lane, unit)
                        if submission_path.exists():
                            _validate_submission_receipt(
                                lane, unit, intent=intent, path=submission_path
                            )
                            outcome = {
                                "status": "already_submitted",
                                "receipt": str(submission_path),
                            }
                        else:
                            any_work = True
                            if unit.unit_id in unresolved_intents:
                                reconciliation = reconciled_intents[unit.unit_id]
                                raise CycleGuardFailed(
                                    "submission_identity_unresolved",
                                    (
                                        f"pre-existing intent {unit.unit_id!r} has "
                                        "no receipt and reconciliation is "
                                        f"{reconciliation.status}: "
                                        f"{reconciliation.detail or 'no detail'}"
                                    ),
                                )
                            elif lane_handlers.reconcile_submission is None:
                                raise CycleGuardFailed("missing_submission_reconciler")
                            elif unit.unit_id in reconciled_intents:
                                reconciliation = reconciled_intents[unit.unit_id]
                            else:
                                reconciliation = lane_handlers.reconcile_submission(
                                    lane, unit, snapshot
                                )
                            if not isinstance(reconciliation, ReconciliationResult):
                                raise CycleGuardFailed(
                                    "submission_reconciliation_untyped",
                                    f"unit {unit.unit_id!r}",
                                )
                            if reconciliation.status == RECONCILIATION_INDETERMINATE:
                                raise CycleGuardFailed(
                                    "submission_identity_unresolved",
                                    (
                                        f"unit {unit.unit_id!r} reconciliation is "
                                        f"{reconciliation.status}: "
                                        f"{reconciliation.detail or 'no detail'}"
                                    ),
                                )
                            if reconciliation.status == RECONCILIATION_FOUND:
                                remote = _validated_remote_identity(
                                    unit, reconciliation.remote
                                )
                                receipt = _record_submission_receipt(
                                    lane,
                                    unit,
                                    intent=intent,
                                    remote=remote,
                                    recovered=True,
                                )
                                outcome = {
                                    "status": "recovered",
                                    "receipt": str(receipt),
                                }
                            elif lane_handlers.submit_unit is None:
                                raise CycleGuardFailed("missing_submission_handler")
                            else:
                                remote = _validated_remote_identity(
                                    unit, lane_handlers.submit_unit(lane, unit)
                                )
                                receipt = _record_submission_receipt(
                                    lane,
                                    unit,
                                    intent=intent,
                                    remote=remote,
                                    recovered=False,
                                )
                                outcome = {
                                    "status": "submitted",
                                    "receipt": str(receipt),
                                }
                    except CycleGuardFailed as exc:
                        outcome = _blocked(exc.guard, exc.detail)
                    except Exception as exc:
                        outcome = _blocked("submission_handler_error", str(exc))
                    outcome.update(
                        {"unit_id": unit.unit_id, "intent": str(_intent_paths(lane, unit)[0])}
                    )
                    result["submissions"].append(outcome)
                    if outcome.get("status") not in {
                        "submitted",
                        "recovered",
                        "already_submitted",
                    }:
                        lane_blocked = True
                        break

            submission_changes = [
                row
                for row in result["submissions"]
                if row.get("status") in {"submitted", "recovered"}
            ]
            if submission_changes:
                post_context = {
                    "phase": "post_submission_refresh",
                    "snapshot_sha256": snapshot.digest,
                    "submissions": submission_changes,
                    "gate_pending": lane_blocked,
                }
                if lane_handlers.refresh_status is None:
                    post_refresh = _blocked("missing_report_refresh_handler")
                else:
                    try:
                        post_refresh = _verified_refresh_outcome(
                            registry=registry,
                            lane=lane,
                            outcome=lane_handlers.refresh_status(
                                lane, snapshot, post_context
                            ),
                        )
                    except CycleGuardFailed as exc:
                        post_refresh = _blocked(exc.guard, exc.detail)
                    except Exception as exc:
                        post_refresh = _blocked(
                            "report_refresh_handler_error", str(exc)
                        )
                result["post_submission_report_refresh"] = post_refresh
                if post_refresh.get("status") not in {"refreshed", "unchanged"}:
                    lane_blocked = True

            result["status"] = "blocked" if lane_blocked else "complete"
            any_blocked = any_blocked or lane_blocked
            lane_results[lane.lane_id] = result
            report_rows = [
                row
                for row in (
                    result.get("report_refresh"),
                    result.get("post_submission_report_refresh"),
                )
                if isinstance(row, Mapping)
            ]
            lane_state[lane.lane_id] = {
                "scheduler_jobs_sha256": jobs_digest,
                "durable_coverage_sha256": coverage_digest,
                "last_snapshot_sha256": snapshot.digest,
                "terminal_failure_fingerprints": sorted(
                    job.fingerprint
                    for job in jobs
                    if job.state in {"failed", "removed"}
                ),
                "report_refresh_required": any(
                    row.get("status") not in {"refreshed", "unchanged"}
                    for row in report_rows
                ),
                "status": result["status"],
            }

        state["lanes"] = lane_state
        return finish(
            state=state,
            lane_results=lane_results,
            any_work=any_work,
            any_blocked=any_blocked,
            accept_snapshot=True,
        )


def _validate_handler_mapping(
    handlers: Any, *, label: str
) -> Mapping[str, LaneHandlers]:
    if not isinstance(handlers, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, LaneHandlers)
        for key, value in handlers.items()
    ):
        raise TypeError(f"{label} must return a mapping of LaneHandlers")
    return handlers


def _normalize_expected_scopes(
    raw_scopes: Any, *, registry: CampaignRegistry
) -> tuple[SchedulerScope, ...]:
    if raw_scopes is None or isinstance(raw_scopes, (str, bytes, Mapping)):
        raise CycleGuardFailed(
            "expected_scheduler_scopes_missing",
            "operations adapter did not return a SchedulerScope sequence",
        )
    try:
        scopes = tuple(raw_scopes)
    except TypeError as exc:
        raise CycleGuardFailed(
            "expected_scheduler_scopes_missing",
            "operations adapter did not return an iterable",
        ) from exc
    if any(not isinstance(scope, SchedulerScope) for scope in scopes):
        raise CycleGuardFailed(
            "expected_scheduler_scopes_untyped",
            "every production scope must be a SchedulerScope",
        )
    normalized = tuple(sorted(scopes))
    scope_keys = [(scope.lane_id, scope.cluster_id) for scope in normalized]
    if len(set(scope_keys)) != len(scope_keys):
        raise CycleGuardFailed(
            "expected_scheduler_scopes_duplicate",
            repr(scope_keys),
        )
    monitored = {
        lane.lane_id for lane in registry.lanes if lane.monitor_enabled
    }
    unknown = sorted({scope.lane_id for scope in normalized} - monitored)
    if unknown:
        raise CycleGuardFailed(
            "expected_scheduler_scope_not_monitored",
            repr(unknown),
        )
    if any(lane.monitor_enabled for lane in registry.lanes) and not normalized:
        raise CycleGuardFailed(
            "expected_scheduler_scopes_empty",
            "a registry with monitored lanes requires durable scheduler scopes",
        )
    return normalized


def _verified_registry_file(
    *,
    repository_root: Path,
    path: Path,
    expected_sha256: str,
    label: str,
) -> Path:
    resolved = _resolve_repository_file(repository_root, path, label=label)
    actual_sha256 = _sha256_file(resolved)
    if actual_sha256 != expected_sha256:
        raise CycleGuardFailed(
            "registry_artifact_digest_mismatch",
            f"{label}: expected {expected_sha256}, got {actual_sha256}",
        )
    return resolved


def _registered_expected_scopes(
    registry: CampaignRegistry,
) -> tuple[SchedulerScope, ...]:
    """Load scheduler scopes from registry-bound durable authority."""

    binding = registry.production_operations
    if binding is None:
        raise CycleGuardFailed("production_operations_binding_missing")
    authority_path = _verified_registry_file(
        repository_root=registry.repository_root,
        path=binding.bootstrap_scopes_path,
        expected_sha256=binding.bootstrap_scopes_sha256,
        label="production bootstrap scheduler scopes",
    )
    authority = _read_object(authority_path, label="scheduler scope authority")
    if (
        authority.get("schema") != SCOPE_AUTHORITY_SCHEMA_V1
        or authority.get("registry_id") != registry.registry_id
    ):
        raise CycleGuardFailed(
            "scheduler_scope_authority_unverifiable", str(authority_path)
        )
    scope_rows = authority.get("scopes")
    if not isinstance(scope_rows, list) or any(
        not isinstance(row, Mapping) for row in scope_rows
    ):
        raise CycleGuardFailed(
            "scheduler_scope_authority_unverifiable",
            f"{authority_path}: scopes must be a list of objects",
        )
    try:
        bootstrap_scopes = tuple(
            SchedulerScope.from_mapping(row) for row in scope_rows
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CycleGuardFailed(
            "scheduler_scope_authority_unverifiable",
            f"{authority_path}: {exc}",
        ) from exc
    processes_by_cluster: dict[tuple[str, int], set[int]] = {}
    for scope in bootstrap_scopes:
        processes_by_cluster.setdefault(
            (scope.lane_id, scope.cluster_id), set()
        ).update(scope.process_ids)

    # Scopes created after bootstrap are derived from the supervisor's own
    # digest-bound intent/receipt pairs, never from the capture adapter.
    for lane in registry.lanes:
        receipt_root = lane.state_root / "submission_receipts"
        intent_root = lane.state_root / "submission_intents"
        for receipt_path in sorted(receipt_root.glob("*.json")):
            intent_path = intent_root / receipt_path.name
            if not intent_path.is_file():
                raise CycleGuardFailed(
                    "submission_scope_unverifiable",
                    f"{receipt_path}: matching intent is missing",
                )
            unit, intent = _unit_from_intent(lane, intent_path)
            receipt = _validate_submission_receipt(
                lane, unit, intent=intent, path=receipt_path
            )
            remote = dict(receipt["remote"])
            try:
                cluster_id = int(remote["cluster_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise CycleGuardFailed(
                    "submission_scope_unverifiable",
                    f"{receipt_path}: no integer cluster_id",
                ) from exc
            process_ids_raw = remote.get("process_ids")
            if process_ids_raw is None:
                process_ids = tuple(range(len(unit.action_ids)))
            elif isinstance(process_ids_raw, (list, tuple)):
                process_ids = tuple(int(value) for value in process_ids_raw)
            else:
                raise CycleGuardFailed(
                    "submission_scope_unverifiable",
                    f"{receipt_path}: process_ids is not a sequence",
                )
            if (
                cluster_id < 0
                or len(process_ids) != len(unit.action_ids)
                or tuple(sorted(set(process_ids))) != process_ids
            ):
                raise CycleGuardFailed(
                    "submission_scope_unverifiable",
                    f"{receipt_path}: action/process scope mismatch",
                )
            processes_by_cluster.setdefault((lane.lane_id, cluster_id), set()).update(
                process_ids
            )

    scopes = tuple(
        SchedulerScope(
            lane_id=lane_id,
            cluster_id=cluster_id,
            process_ids=tuple(sorted(process_ids)),
        )
        for (lane_id, cluster_id), process_ids in sorted(processes_by_cluster.items())
    )
    return _normalize_expected_scopes(scopes, registry=registry)


def _validate_production_snapshot(
    *,
    repository_root: Path,
    snapshot: SchedulerSnapshot,
    adapter_id: str,
    expected_scopes: tuple[SchedulerScope, ...],
) -> None:
    """Bind one production snapshot to its adapter, scopes, and raw evidence."""

    if not adapter_id or not _SAFE_ID.fullmatch(adapter_id):
        raise CycleGuardFailed(
            "operations_adapter_identity_missing",
            repr(adapter_id),
        )
    source_adapter = str(snapshot.source.get("adapter_id") or "")
    if source_adapter != adapter_id:
        raise CycleGuardFailed(
            "scheduler_adapter_identity_mismatch",
            f"expected {adapter_id!r}, got {source_adapter!r}",
        )
    if snapshot.expected_scopes != expected_scopes:
        raise CycleGuardFailed(
            "scheduler_scope_binding_mismatch",
            (
                f"expected {[scope.to_json_dict() for scope in expected_scopes]}, "
                f"got {[scope.to_json_dict() for scope in snapshot.expected_scopes]}"
            ),
        )
    evidence = snapshot.source.get("raw_scheduler_evidence")
    if not isinstance(evidence, Mapping):
        raise CycleGuardFailed(
            "raw_scheduler_evidence_missing",
            "snapshot source has no raw_scheduler_evidence object",
        )
    if evidence.get("schema") != RAW_SCHEDULER_EVIDENCE_SCHEMA_V1:
        raise CycleGuardFailed(
            "raw_scheduler_evidence_unverifiable",
            "unsupported or missing raw scheduler evidence schema",
        )
    for query in ("queue", "history"):
        binding = evidence.get(query)
        if not isinstance(binding, Mapping):
            raise CycleGuardFailed(
                "raw_scheduler_evidence_unverifiable",
                f"missing {query} query evidence",
            )
        reference = str(binding.get("reference") or "")
        sha256 = str(binding.get("sha256") or "")
        if not reference or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise CycleGuardFailed(
                "raw_scheduler_evidence_unverifiable",
                f"{query} evidence needs an immutable reference and SHA-256",
            )
        try:
            evidence_path = _resolve_repository_file(
                repository_root,
                Path(reference),
                label=f"raw {query} scheduler evidence",
            )
        except CycleGuardFailed as exc:
            raise CycleGuardFailed(
                "raw_scheduler_evidence_unverifiable",
                f"{query}: {exc}",
            ) from exc
        actual_sha256 = _sha256_file(evidence_path)
        if actual_sha256 != sha256:
            raise CycleGuardFailed(
                "raw_scheduler_evidence_unverifiable",
                f"{query}: expected {sha256}, got {actual_sha256}",
            )


def load_handler_factory(
    reference: str, registry: CampaignRegistry
) -> Mapping[str, LaneHandlers]:
    """Load ``module:function`` returning the lane-handler mapping.

    Handler construction is explicit rather than encoded in the data registry:
    the registry freezes scientific enablement and paths, while the operations
    host supplies authenticated CHTC transport and report builders.
    """

    module_name, separator, function_name = reference.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("handler factory must be written as module:function")
    factory = getattr(importlib.import_module(module_name), function_name)
    return _validate_handler_mapping(
        factory(registry), label="handler factory"
    )


def load_operations_factory(
    reference: str, registry: CampaignRegistry
) -> AcquisitionOperations:
    """Load one explicitly named production operations adapter factory."""

    _validate_registry_for_production(registry)
    binding = registry.production_operations
    if binding is None:
        raise CycleGuardFailed("production_operations_binding_missing")
    if reference != binding.factory:
        raise CycleGuardFailed(
            "operations_factory_binding_mismatch",
            f"registry={binding.factory!r}, requested={reference!r}",
        )
    source_path = _verified_registry_file(
        repository_root=registry.repository_root,
        path=binding.source_path,
        expected_sha256=binding.source_sha256,
        label="production operations adapter source",
    )
    module_name, separator, function_name = reference.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("operations factory must be written as module:function")
    module = importlib.import_module(module_name)
    module_path_raw = getattr(module, "__file__", None)
    if module_path_raw is None or Path(module_path_raw).resolve() != source_path:
        raise CycleGuardFailed(
            "operations_factory_source_mismatch",
            f"registry={source_path}, imported={module_path_raw}",
        )
    factory = getattr(module, function_name)
    factory_source = inspect.getsourcefile(factory)
    if factory_source is None or Path(factory_source).resolve() != source_path:
        raise CycleGuardFailed(
            "operations_factory_source_mismatch",
            f"registry={source_path}, factory={factory_source}",
        )
    operations = factory(registry)
    if not isinstance(getattr(operations, "adapter_id", None), str):
        raise TypeError("operations factory returned no string adapter_id")
    if operations.adapter_id != binding.adapter_id:
        raise CycleGuardFailed(
            "operations_adapter_identity_mismatch",
            f"registry={binding.adapter_id!r}, factory={operations.adapter_id!r}",
        )
    for method in ("capture_snapshot", "handlers_for"):
        if not callable(getattr(operations, method, None)):
            raise TypeError(f"operations factory returned no callable {method}")
    return operations


def run_production_cycle(
    *,
    registry: CampaignRegistry,
    operations: AcquisitionOperations,
) -> CycleResult:
    """Capture and execute exactly one adapter-bound production cycle."""

    _validate_registry_for_production(registry)
    if registry.production_operations is None:
        raise CycleGuardFailed("production_operations_binding_missing")
    adapter_id = str(getattr(operations, "adapter_id", "") or "")
    binding = registry.production_operations
    if not adapter_id or not _SAFE_ID.fullmatch(adapter_id):
        raise CycleGuardFailed("operations_adapter_identity_missing", repr(adapter_id))
    if adapter_id != binding.adapter_id:
        raise CycleGuardFailed(
            "operations_adapter_identity_mismatch",
            f"registry={binding.adapter_id!r}, adapter={adapter_id!r}",
        )
    _verified_registry_file(
        repository_root=registry.repository_root,
        path=binding.source_path,
        expected_sha256=binding.source_sha256,
        label="production operations adapter source",
    )
    expected_scopes = _registered_expected_scopes(registry)
    snapshot = operations.capture_snapshot(
        registry, expected_scopes=expected_scopes
    )
    if not isinstance(snapshot, SchedulerSnapshot):
        raise TypeError("operations adapter did not return a SchedulerSnapshot")
    _validate_production_snapshot(
        repository_root=registry.repository_root,
        snapshot=snapshot,
        adapter_id=adapter_id,
        expected_scopes=expected_scopes,
    )
    handlers = _validate_handler_mapping(
        operations.handlers_for(registry), label="operations adapter"
    )
    return run_acquisition_cycle(
        registry=registry,
        snapshot=snapshot,
        handlers=handlers,
        allow_submission=True,
    )


def _validate_registry_for_production(registry: CampaignRegistry) -> None:
    """Invoke the strict canonical contract at every production boundary."""

    # Local import avoids a module cycle: the registry-contract builder uses
    # CampaignRegistry as its final structural parser.
    from pipelines.time_dynamics.acquisition_registry_contract import (
        AcquisitionRegistryContractError,
        validate_registry_for_activation,
    )

    try:
        validate_registry_for_activation(registry)
    except AcquisitionRegistryContractError as exc:
        raise CycleGuardFailed(
            "acquisition_registry_contract_invalid", str(exc)
        ) from exc


def run_replay_cycle(
    *,
    registry: CampaignRegistry,
    snapshot: SchedulerSnapshot,
    handlers: Mapping[str, LaneHandlers],
) -> CycleResult:
    """Replay a supplied snapshot with scheduler submission disabled."""

    return run_acquisition_cycle(
        registry=registry,
        snapshot=snapshot,
        handlers=handlers,
        allow_submission=False,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one inert replay cycle or one adapter-bound production cycle."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--snapshot-json", type=Path)
    mode.add_argument("--operations-factory")
    parser.add_argument("--handler-factory")
    parser.add_argument("--repository-root", type=Path)
    args = parser.parse_args(argv)
    registry = CampaignRegistry.read(
        args.registry, repository_root=args.repository_root
    )
    if args.snapshot_json is not None:
        if not args.handler_factory:
            parser.error("--snapshot-json requires --handler-factory")
        result = run_replay_cycle(
            registry=registry,
            snapshot=SchedulerSnapshot.read(args.snapshot_json),
            handlers=load_handler_factory(args.handler_factory, registry),
        )
        execution_mode = "replay_submission_inert"
    else:
        if args.handler_factory:
            parser.error(
                "--operations-factory owns handlers; do not pass --handler-factory"
            )
        operations = load_operations_factory(args.operations_factory, registry)
        result = run_production_cycle(registry=registry, operations=operations)
        execution_mode = "production_operations_adapter"
    print(
        json.dumps(
            {
                "execution_mode": execution_mode,
                "status": result.status,
                "snapshot_sha256": result.snapshot_digest,
                "state": str(result.state_path),
                "receipt": str(result.receipt_path),
                "lanes": result.lane_results,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 2 if result.status == "blocked" else 0


__all__ = [
    "ACTIVE_STATES",
    "AcquisitionOperations",
    "ArtifactBinding",
    "BlockedProductionOperationsBinding",
    "CYCLE_RECEIPT_SCHEMA_V1",
    "CampaignRegistry",
    "CycleGuardFailed",
    "CycleResult",
    "FAILURE_DISPOSITION_SCHEMA_V1",
    "LANE_SCIENTIFIC_AUTHORITY_SCHEMA_V1",
    "LaneHandlers",
    "LaneRegistration",
    "LaneScientificAuthority",
    "load_handler_factory",
    "load_operations_factory",
    "main",
    "RAW_SCHEDULER_EVIDENCE_SCHEMA_V1",
    "REGISTRY_SCHEMA_V1",
    "RemoteAccessPending",
    "SNAPSHOT_SCHEMA_V1",
    "STATE_SCHEMA_V1",
    "SchedulerJob",
    "SchedulerScope",
    "SchedulerSnapshot",
    "SubmissionPlan",
    "SubmissionUnit",
    "SupervisorLock",
    "SupervisorLocked",
    "TERMINAL_STATES",
    "VALID_STATES",
    "run_acquisition_cycle",
    "run_production_cycle",
    "run_replay_cycle",
]


if __name__ == "__main__":
    raise SystemExit(main())
