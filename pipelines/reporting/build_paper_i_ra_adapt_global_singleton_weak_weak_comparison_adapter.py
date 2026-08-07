#!/usr/bin/env python3
"""Build the authenticated weak--weak global-singleton comparison adapter.

The two source archives are intentionally not extracted.  Each archive is
read once as a stream.  The compressed archive bytes and every regular tar
member are SHA-256 checked, while ``worker_outputs/result.json`` is projected
with :mod:`ijson` so the large result is never materialized as one Python
object.

This output is a diagnostic comparison input, not a paper-evidence adoption
receipt.  It covers exactly the completed weak--weak append-commutation and
plateau-commutation arms.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
from types import ModuleType
from typing import Any, BinaryIO, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CAMPAIGN_ROOT = REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727"
RETRIEVED_ROOT = CAMPAIGN_ROOT / "retrieved_chtc_20260730"
SCIENTIFIC_PACKAGE = CAMPAIGN_ROOT / (
    "paper_i_ra_adapt_global_singleton_insertion12_r50_"
    "20260730_v1_chtc"
)
V1_ACTIVATION = CAMPAIGN_ROOT / (
    "paper_i_ra_adapt_global_singleton_insertion12_r50_"
    "20260730_v1_chtc_activation"
)
V2_PACKAGE = CAMPAIGN_ROOT / (
    "paper_i_ra_adapt_global_singleton_insertion12_r50_"
    "20260730_v2_chtc"
)
V2_RELEASE_ACTIVATION = CAMPAIGN_ROOT / (
    "paper_i_ra_adapt_global_singleton_insertion12_r50_"
    "20260730_v2_chtc_activation_release_v2"
)
PRESERVATION_SNAPSHOT = CAMPAIGN_ROOT / (
    "paper_i_ra_adapt_completed_v1_preservation_"
    "9395481_9395482_20260730_v1.json"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "raw_outputs/"
    "paper_i_ra_adapt_global_singleton_weak_weak_comparison_20260730/"
    "diagnostic_adapter.json"
)

ADAPTER_SCHEMA = (
    "paper_i_ra_adapt_global_singleton_weak_weak_"
    "comparison_diagnostic_v1"
)
ARM_SCHEMA = (
    "paper_i_ra_adapt_global_singleton_weak_weak_"
    "comparison_arm_v1"
)
ATTEMPT_SCHEMA = (
    "paper_i_ra_global_singleton_insertion12_worker_attempt_v1"
)
AUTHORIZATION_SCHEMA = (
    "paper_i_ra_global_singleton_insertion12_"
    "execution_authorization_v1"
)
RESULT_SCHEMA = "paper_i_ra_adapt_result_v1"
CAMPAIGN_ID = (
    "paper_i_ra_adapt_global_singleton_insertion_comparison_v1"
)
HORIZON = 50
REGIME_ID = "weak_weak"
NPH = 3
RESULT_MEMBER = "worker_outputs/result.json"
ATTEMPT_RECEIPT_MEMBER = "worker_attempt_receipt.json"
JOB_MEMBER = "authority/job.json"
AUTHORIZATION_MEMBER = "authority/execution_authorization.json"
ACTIVATION_MEMBER = "authority/activation_manifest.json"
IDENTITY_MEMBER = "worker_outputs/attempt_identity.tsv"
EXIT_MEMBER = "worker_outputs/worker_exit_status.txt"


class ComparisonAdapterError(ValueError):
    """Raised when any source or scientific closure is not exact."""


@dataclass(frozen=True)
class ArmSpec:
    key: str
    execution_id: str
    route_id: str
    insertion_policy: str
    insertion_runtime_mode: str
    archive_name: str
    cluster_id: int
    proc_id: int
    activation_root: Path
    operational_package: Path
    preservation_required: bool


ARM_SPECS = (
    ArmSpec(
        key="append",
        execution_id=(
            "global_singleton__weak_weak__nph3__"
            "ra_global_singleton_append_commutation_reduced"
        ),
        route_id="ra_global_singleton_append_commutation_reduced",
        insertion_policy="append_commutation_reduced",
        insertion_runtime_mode="append_commutation_reduced",
        archive_name=(
            "global_singleton__weak_weak__nph3__"
            "ra_global_singleton_append_commutation_reduced"
            "__cluster_9395482__proc_0.tar.gz"
        ),
        cluster_id=9395482,
        proc_id=0,
        activation_root=V1_ACTIVATION,
        operational_package=SCIENTIFIC_PACKAGE,
        preservation_required=True,
    ),
    ArmSpec(
        key="plateau",
        execution_id=(
            "global_singleton__weak_weak__nph3__"
            "ra_global_singleton_plateau_commutation"
        ),
        route_id="ra_global_singleton_plateau_commutation",
        insertion_policy="plateau_commutation",
        insertion_runtime_mode="insertion_commutation_plateau_v1",
        archive_name=(
            "global_singleton__weak_weak__nph3__"
            "ra_global_singleton_plateau_commutation"
            "__cluster_9397760__proc_0.tar.gz"
        ),
        cluster_id=9397760,
        proc_id=0,
        activation_root=V2_RELEASE_ACTIVATION,
        operational_package=V2_PACKAGE,
        preservation_required=False,
    ),
)


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(dict(payload))).hexdigest()


def _digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(payload))
    if "sha256" in result:
        raise ComparisonAdapterError(
            "Cannot self-digest a payload that already has sha256."
        )
    result["sha256"] = _canonical_sha256(result)
    return result


def _verify_self_digest(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> str:
    expected = payload.get("sha256")
    unsigned = dict(payload)
    unsigned.pop("sha256", None)
    observed = _canonical_sha256(unsigned)
    if expected != observed:
        raise ComparisonAdapterError(f"{label} self-digest drifted.")
    return observed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise ComparisonAdapterError(
            f"Path escaped the active repository: {path}"
        ) from exc


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComparisonAdapterError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise ComparisonAdapterError(f"{label} must be a JSON object.")
    return payload


def _verified_json(path: Path, *, label: str) -> dict[str, Any]:
    payload = _load_json(path, label=label)
    _verify_self_digest(payload, label=label)
    return payload


def _file_binding(
    path: Path,
    *,
    canonical_sha256: str | None = None,
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ComparisonAdapterError(f"Unsafe bound file: {path}")
    row: dict[str, Any] = {
        "path": _relative(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if canonical_sha256 is not None:
        row["canonical_sha256"] = canonical_sha256
    return row


def _safe_member_name(name: str) -> str:
    pure = PurePosixPath(name)
    canonical = pure.as_posix()
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or canonical != name
    ):
        raise ComparisonAdapterError(f"Unsafe archive member name: {name}")
    return canonical


class _HashingReader:
    """Minimal read-only wrapper that hashes every byte consumed."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._digest = hashlib.sha256()
        self.size_bytes = 0

    def read(self, size: int = -1) -> bytes:
        block = self._stream.read(size)
        if block:
            self._digest.update(block)
            self.size_bytes += len(block)
        return block

    def readinto(self, buffer: bytearray | memoryview) -> int:
        block = self.read(len(buffer))
        size = len(block)
        buffer[:size] = block
        return size

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()


def _hash_stream(stream: BinaryIO) -> tuple[str, int]:
    reader = _HashingReader(stream)
    for _block in iter(lambda: reader.read(4 * 1024 * 1024), b""):
        pass
    return reader.sha256, reader.size_bytes


def _read_small_member(
    stream: BinaryIO,
    *,
    declared_size: int,
    label: str,
) -> tuple[bytes, str, int]:
    if declared_size > 4 * 1024 * 1024:
        raise ComparisonAdapterError(
            f"{label} exceeds the bounded in-memory member limit."
        )
    reader = _HashingReader(stream)
    raw = reader.read()
    if reader.size_bytes != declared_size:
        raise ComparisonAdapterError(f"{label} size drifted while reading.")
    return raw, reader.sha256, reader.size_bytes


def _json_from_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComparisonAdapterError(f"{label} is invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise ComparisonAdapterError(f"{label} must be a JSON object.")
    return payload


def _integer(value: Any, *, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise ComparisonAdapterError(f"{label} must be an integer.")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ComparisonAdapterError(f"{label} must be an integer.") from exc
    if converted != value or (
        minimum is not None and converted < minimum
    ):
        raise ComparisonAdapterError(f"{label} is outside its contract.")
    return converted


def _finite(value: Any, *, label: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ComparisonAdapterError(f"{label} must be finite.") from exc
    if not math.isfinite(converted):
        raise ComparisonAdapterError(f"{label} must be finite.")
    return converted


class _ResultProjection:
    """Incremental result projection populated by ``ijson`` events."""

    _SINGLE_ROOTS = {
        "protocol": "protocol",
        "run.canonical_reporting.reference_state": "reference_state",
        "run.paper_i_summary.canonical_all_work": "canonical_all_work",
        "run.paper_i_summary.effective_plateau": "effective_plateau",
        "run.problem": "problem",
        "run.route": "route",
        "run.stop": "stop",
    }
    _REPEATED_ROOTS = {
        "run.accepted_trajectory.item": "trajectory",
        "run.accepted_transitions.item": "transitions",
        "run.canonical_reporting.accepted_prefix_work.item": "prefix_work",
        "run.paper_i_summary.accepted_error_trace.item": "error_trace",
        "run.scientific_replay.item": "scientific_replay",
    }

    def __init__(self) -> None:
        self.schema: str | None = None
        self.available_rounds: int | None = None
        self.exact_energy: float | None = None
        self.objects: dict[str, Any] = {}
        self.trajectory: list[dict[str, Any]] = []
        self.transitions: list[dict[str, Any]] = []
        self.prefix_work: list[dict[str, Any]] = []
        self.error_trace: list[dict[str, Any]] = []
        self.selected_replay: dict[int, dict[str, Any]] = {}
        self._active: tuple[str, str, Any] | None = None

    def _finish_repeated(self, kind: str, value: Any) -> None:
        if not isinstance(value, dict):
            raise ComparisonAdapterError(
                f"Projected result {kind} item is not a mapping."
            )
        if kind == "trajectory":
            self.trajectory.append(value)
        elif kind == "transitions":
            self.transitions.append(value)
        elif kind == "prefix_work":
            self.prefix_work.append(value)
        elif kind == "error_trace":
            self.error_trace.append(value)
        elif kind == "scientific_replay":
            plateau = self.objects.get("effective_plateau")
            if not isinstance(plateau, Mapping):
                raise ComparisonAdapterError(
                    "Effective plateau must precede scientific replay "
                    "in the canonical result."
                )
            checkpoint = value.get("checkpoint")
            if not isinstance(checkpoint, Mapping):
                raise ComparisonAdapterError(
                    "Scientific replay row lacks its checkpoint."
                )
            round_index = _integer(
                checkpoint.get("outer_iteration"),
                label="scientific replay outer iteration",
                minimum=1,
            )
            plateau_round = _integer(
                plateau.get("controller_round"),
                label="effective plateau round",
                minimum=1,
            )
            if round_index in {plateau_round, HORIZON}:
                if round_index in self.selected_replay:
                    raise ComparisonAdapterError(
                        "Selected scientific replay round is duplicated."
                    )
                self.selected_replay[round_index] = value
        else:
            raise AssertionError(f"Unknown repeated result kind: {kind}")

    def feed(self, prefix: str, event: str, value: Any, ijson: Any) -> None:
        if self._active is not None:
            root, kind, builder = self._active
            if prefix != root and not prefix.startswith(root + "."):
                raise ComparisonAdapterError(
                    f"Incremental result capture escaped {root!r}."
                )
            builder.event(event, value)
            if prefix == root and event in {"end_map", "end_array"}:
                projected = builder.value
                self._active = None
                if root in self._SINGLE_ROOTS:
                    if kind in self.objects:
                        raise ComparisonAdapterError(
                            f"Projected result object {kind} is duplicated."
                        )
                    self.objects[kind] = projected
                else:
                    self._finish_repeated(kind, projected)
            return

        roots = {**self._SINGLE_ROOTS, **self._REPEATED_ROOTS}
        if prefix in roots and event in {"start_map", "start_array"}:
            builder = ijson.common.ObjectBuilder()
            builder.event(event, value)
            self._active = (prefix, roots[prefix], builder)
            return

        if event not in {"string", "number", "boolean", "null"}:
            return
        if prefix == "schema":
            self.schema = str(value)
        elif prefix == "run.paper_i_summary.available_controller_rounds":
            self.available_rounds = _integer(
                value,
                label="summary available rounds",
                minimum=1,
            )
        elif prefix == (
            "run.paper_i_summary.provenance.exact_same_cutoff_energy"
        ):
            self.exact_energy = _finite(
                value,
                label="summary exact same-cutoff energy",
            )

    def finish(self) -> dict[str, Any]:
        if self._active is not None:
            raise ComparisonAdapterError(
                "Incremental result ended inside a selected object."
            )
        required = {
            "protocol",
            "reference_state",
            "canonical_all_work",
            "effective_plateau",
            "problem",
            "route",
            "stop",
        }
        if (
            self.schema != RESULT_SCHEMA
            or self.available_rounds != HORIZON
            or self.exact_energy is None
            or set(self.objects) != required
            or len(self.trajectory) != HORIZON
            or len(self.transitions) != HORIZON
            or len(self.prefix_work) != HORIZON
            or len(self.error_trace) != HORIZON
        ):
            raise ComparisonAdapterError(
                "Incremental result projection is incomplete."
            )
        plateau_round = _integer(
            self.objects["effective_plateau"].get("controller_round"),
            label="effective plateau round",
            minimum=1,
        )
        if (
            plateau_round > HORIZON
            or set(self.selected_replay) != {plateau_round, HORIZON}
        ):
            raise ComparisonAdapterError(
                "Selected replay closure is incomplete."
            )
        return {
            "schema": self.schema,
            "available_rounds": self.available_rounds,
            "exact_energy": self.exact_energy,
            **self.objects,
            "trajectory": self.trajectory,
            "transitions": self.transitions,
            "prefix_work": self.prefix_work,
            "error_trace": self.error_trace,
            "selected_replay": self.selected_replay,
        }


def _stream_result(
    stream: BinaryIO,
    *,
    declared_size: int,
) -> tuple[dict[str, Any], str, int]:
    try:
        import ijson
    except ModuleNotFoundError as exc:
        raise ComparisonAdapterError(
            "ijson is required to stream the large result member."
        ) from exc

    reader = _HashingReader(stream)
    projection = _ResultProjection()
    try:
        for prefix, event, value in ijson.parse(reader, use_float=True):
            projection.feed(prefix, event, value, ijson)
    except ComparisonAdapterError:
        raise
    except (ijson.JSONError, OverflowError, ValueError) as exc:
        raise ComparisonAdapterError(
            f"Large result JSON stream is invalid: {exc}"
        ) from exc
    for _block in iter(lambda: reader.read(1024 * 1024), b""):
        pass
    if reader.size_bytes != declared_size:
        raise ComparisonAdapterError(
            "Large result member size drifted while streaming."
        )
    return projection.finish(), reader.sha256, reader.size_bytes


def _scan_archive(
    archive_path: Path,
) -> dict[str, Any]:
    if not archive_path.is_file() or archive_path.is_symlink():
        raise ComparisonAdapterError(f"Unsafe attempt archive: {archive_path}")
    small_names = {
        ATTEMPT_RECEIPT_MEMBER,
        JOB_MEMBER,
        AUTHORIZATION_MEMBER,
        ACTIVATION_MEMBER,
        IDENTITY_MEMBER,
        EXIT_MEMBER,
    }
    observed: dict[str, dict[str, Any]] = {}
    small_raw: dict[str, bytes] = {}
    result_projection: dict[str, Any] | None = None

    try:
        with archive_path.open("rb") as raw_archive:
            compressed = _HashingReader(raw_archive)
            with tarfile.open(fileobj=compressed, mode="r|gz") as archive:
                for member in archive:
                    name = _safe_member_name(member.name)
                    if name in observed:
                        raise ComparisonAdapterError(
                            f"Duplicate archive member: {name}"
                        )
                    if (
                        not member.isfile()
                        or member.issym()
                        or member.islnk()
                        or member.size < 0
                    ):
                        raise ComparisonAdapterError(
                            f"Archive member is not a safe regular file: {name}"
                        )
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ComparisonAdapterError(
                            f"Archive member is unreadable: {name}"
                        )
                    if name == RESULT_MEMBER:
                        (
                            result_projection,
                            digest,
                            size_bytes,
                        ) = _stream_result(
                            stream,
                            declared_size=member.size,
                        )
                    elif name in small_names:
                        raw, digest, size_bytes = _read_small_member(
                            stream,
                            declared_size=member.size,
                            label=name,
                        )
                        small_raw[name] = raw
                    else:
                        digest, size_bytes = _hash_stream(stream)
                    if size_bytes != member.size:
                        raise ComparisonAdapterError(
                            f"Archive member size drifted: {name}"
                        )
                    observed[name] = {
                        "sha256": digest,
                        "size_bytes": size_bytes,
                    }
            for _block in iter(
                lambda: compressed.read(4 * 1024 * 1024), b""
            ):
                pass
            archive_sha256 = compressed.sha256
            archive_size = compressed.size_bytes
    except (OSError, tarfile.TarError) as exc:
        raise ComparisonAdapterError(
            f"Attempt archive is unreadable: {archive_path}"
        ) from exc

    if (
        result_projection is None
        or set(small_raw) != small_names
        or archive_size != archive_path.stat().st_size
    ):
        raise ComparisonAdapterError(
            "Attempt archive selected-member closure failed."
        )
    receipt = _json_from_bytes(
        small_raw[ATTEMPT_RECEIPT_MEMBER],
        label="worker attempt receipt",
    )
    job = _json_from_bytes(small_raw[JOB_MEMBER], label="archived job")
    authorization = _json_from_bytes(
        small_raw[AUTHORIZATION_MEMBER],
        label="archived execution authorization",
    )
    activation = _json_from_bytes(
        small_raw[ACTIVATION_MEMBER],
        label="archived activation manifest",
    )
    for label, payload in (
        ("worker attempt receipt", receipt),
        ("archived job", job),
        ("archived execution authorization", authorization),
        ("archived activation manifest", activation),
    ):
        _verify_self_digest(payload, label=label)

    raw_worker_rows = receipt.get("worker_files")
    if not isinstance(raw_worker_rows, list):
        raise ComparisonAdapterError(
            "Worker attempt receipt has no worker member list."
        )
    declared_workers: dict[str, Mapping[str, Any]] = {}
    for raw_row in raw_worker_rows:
        if not isinstance(raw_row, Mapping):
            raise ComparisonAdapterError(
                "Worker attempt receipt row is malformed."
            )
        relative = _safe_member_name(str(raw_row.get("path", "")))
        member_name = f"worker_outputs/{relative}"
        if member_name in declared_workers:
            raise ComparisonAdapterError(
                f"Worker attempt receipt duplicates {member_name}."
            )
        declared_workers[member_name] = raw_row
    expected_members = set(declared_workers) | {
        JOB_MEMBER,
        AUTHORIZATION_MEMBER,
        ACTIVATION_MEMBER,
        ATTEMPT_RECEIPT_MEMBER,
    }
    if set(observed) != expected_members:
        raise ComparisonAdapterError(
            "Archive regular-member set differs from its worker receipt."
        )
    for name, row in declared_workers.items():
        if (
            observed[name]["sha256"] != row.get("sha256")
            or observed[name]["size_bytes"]
            != _integer(
                row.get("size_bytes"),
                label=f"{name} receipt size",
                minimum=0,
            )
        ):
            raise ComparisonAdapterError(
                f"Archive member binding drifted: {name}"
            )
    if (
        receipt.get("schema") != ATTEMPT_SCHEMA
        or receipt.get("job_file_sha256") != observed[JOB_MEMBER]["sha256"]
        or receipt.get("authorization_file_sha256")
        != observed[AUTHORIZATION_MEMBER]["sha256"]
        or receipt.get("activation_manifest_file_sha256")
        != observed[ACTIVATION_MEMBER]["sha256"]
        or receipt.get("worker_exit_status") != 0
        or small_raw[EXIT_MEMBER] != b"0\n"
    ):
        raise ComparisonAdapterError(
            "Worker attempt authority or exit-zero binding failed."
        )
    return {
        "archive": {
            "path": _relative(archive_path),
            "sha256": archive_sha256,
            "size_bytes": archive_size,
            "member_count": len(observed),
        },
        "observed_members": observed,
        "receipt": receipt,
        "job": job,
        "authorization": authorization,
        "activation": activation,
        "attempt_identity": small_raw[IDENTITY_MEMBER],
        "result": result_projection,
    }


def _load_package_contract() -> ModuleType:
    path = SCIENTIFIC_PACKAGE / "package_contract.py"
    spec = importlib.util.spec_from_file_location(
        "paper_i_global_singleton_comparison_package_contract",
        path,
    )
    if spec is None or spec.loader is None:
        raise ComparisonAdapterError(
            "Global-singleton package contract is unavailable."
        )
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _validated_package_authority() -> tuple[dict[str, Any], dict[str, Any]]:
    contract = _load_package_contract()
    try:
        authority = contract.validate_materialization_authority(REPO_ROOT)
    except Exception as exc:
        raise ComparisonAdapterError(
            f"Global-singleton package validation failed: {exc}"
        ) from exc
    if not isinstance(authority, dict):
        raise ComparisonAdapterError(
            "Global-singleton package authority is malformed."
        )
    audit = authority.get("equality_audit")
    delta = authority.get("source_lock_delta")
    final = authority.get("final")
    if (
        not isinstance(audit, Mapping)
        or not isinstance(delta, Mapping)
        or not isinstance(final, Mapping)
        or audit.get("status") != "passed"
        or audit.get("allowed_axis") != "insertion_policy"
        or audit.get("regime_pair_count") != 6
        or audit.get("variant_count") != 12
        or delta.get("status") != "passed"
        or delta.get("all_archive_bindings_preserved") is not True
        or delta.get("all_member_bindings_preserved") is not True
        or delta.get("all_global_source_bindings_preserved") is not True
        or final.get("status") != "passed"
        or final.get("run_class") != "diagnostic"
        or final.get("cell_count") != 12
    ):
        raise ComparisonAdapterError(
            "Package cross-arm or source-lock audit did not pass exactly."
        )
    _verify_self_digest(audit, label="package cross-arm equality audit")
    _verify_self_digest(delta, label="package source-lock delta")
    weak_weak_rows = [
        row
        for row in audit.get("rows", ())
        if isinstance(row, Mapping) and row.get("regime_id") == REGIME_ID
    ]
    if len(weak_weak_rows) != 1 or weak_weak_rows[0].get(
        "status"
    ) != "passed":
        raise ComparisonAdapterError(
            "Package audit lacks the exact weak--weak arm pair."
        )
    projection = {
        "status": "passed",
        "schema": audit["schema"],
        "allowed_axis": "insertion_policy",
        "regime_pair_count": 6,
        "variant_count": 12,
        "canonical_sha256": audit["sha256"],
        "weak_weak_normalized_common_sha256": weak_weak_rows[0][
            "normalized_common_sha256"
        ],
        "source_lock_delta": {
            "schema": delta["schema"],
            "canonical_sha256": delta["sha256"],
            "source_cell_count": delta["source_cell_count"],
            "derived_cell_count": delta["derived_cell_count"],
            "all_archive_bindings_preserved": True,
            "all_member_bindings_preserved": True,
            "all_global_source_bindings_preserved": True,
        },
    }
    return authority, projection


def _verify_local_binding(
    path: Path,
    expected: Mapping[str, Any],
    *,
    label: str,
    canonical_sha256: str | None = None,
) -> dict[str, Any]:
    binding = _file_binding(path, canonical_sha256=canonical_sha256)
    if (
        binding["sha256"] != expected.get("sha256")
        or binding["size_bytes"]
        != _integer(
            expected.get("size_bytes"),
            label=f"{label} expected size",
            minimum=0,
        )
        or (
            canonical_sha256 is not None
            and expected.get("canonical_sha256") != canonical_sha256
        )
    ):
        raise ComparisonAdapterError(f"{label} local binding drifted.")
    return binding


def _protocol_for_arm(
    spec: ArmSpec,
    authority: Mapping[str, Any],
    job: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    protocol_bindings = authority.get("protocol_bindings")
    if not isinstance(protocol_bindings, Mapping):
        raise ComparisonAdapterError(
            "Package protocol-binding map is unavailable."
        )
    expected = protocol_bindings.get(spec.execution_id)
    if not isinstance(expected, Mapping) or job.get("protocol") != expected:
        raise ComparisonAdapterError(
            f"{spec.execution_id}: job/package protocol binding drifted."
        )
    pure = PurePosixPath(str(expected.get("path", "")))
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != str(expected.get("path", ""))
    ):
        raise ComparisonAdapterError(
            f"{spec.execution_id}: unsafe protocol path."
        )
    path = REPO_ROOT.joinpath(*pure.parts)
    protocol = _verified_json(
        path,
        label=f"{spec.execution_id} protocol",
    )
    binding = _verify_local_binding(
        path,
        expected,
        label=f"{spec.execution_id} protocol",
        canonical_sha256=str(protocol["sha256"]),
    )
    if protocol["sha256"] != expected.get("canonical_sha256"):
        raise ComparisonAdapterError(
            f"{spec.execution_id}: protocol canonical digest drifted."
        )
    return protocol, path, binding


def _source_lock_row(
    spec: ArmSpec,
    authority: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    delta = authority.get("source_lock_delta")
    rows = delta.get("rows") if isinstance(delta, Mapping) else None
    if not isinstance(rows, list):
        raise ComparisonAdapterError(
            "Package source-lock rows are unavailable."
        )
    selected = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("cell_id") == spec.execution_id
    ]
    source_locks = protocol.get("source_locks")
    if len(selected) != 1 or not isinstance(source_locks, Mapping):
        raise ComparisonAdapterError(
            f"{spec.execution_id}: source-lock identity is unavailable."
        )
    row = selected[0]
    if (
        row.get("source_lock_id") != source_locks.get(
            "cell_source_lock_id"
        )
        or row.get("source_lock_id")
        != (
            f"weak_weak__nph3__{spec.route_id}"
        ).replace("ra_global_singleton_", "ra_global_singleton_")
        or row.get("target_insertion_policy") != spec.insertion_policy
        or row.get("target_route_id") != spec.route_id
        or row.get("archive_binding_preserved") is not True
        or row.get("member_binding_preserved") is not True
        or row.get("scientific_result_anchor_claimed") is not False
        or not isinstance(
            source_locks.get("cell_source_lock_sha256"), str
        )
        or not isinstance(
            source_locks.get("source_locks_manifest_sha256"), str
        )
    ):
        raise ComparisonAdapterError(
            f"{spec.execution_id}: source-lock identity drifted."
        )
    return {
        "source_lock_id": source_locks["cell_source_lock_id"],
        "cell_source_lock_sha256": source_locks[
            "cell_source_lock_sha256"
        ],
        "source_locks_manifest_sha256": source_locks[
            "source_locks_manifest_sha256"
        ],
        "implementation_source_inventory_sha256": source_locks[
            "implementation_source_inventory_sha256"
        ],
        "predecessor_source_lock_id": row[
            "predecessor_source_lock_id"
        ],
        "archive_binding_preserved": True,
        "member_binding_preserved": True,
    }


def _activation_execution_row(
    activation: Mapping[str, Any],
    execution_id: str,
) -> Mapping[str, Any]:
    rows = activation.get("executions")
    if not isinstance(rows, list):
        raise ComparisonAdapterError(
            "Activation manifest execution rows are unavailable."
        )
    selected = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("execution_id") == execution_id
    ]
    if len(selected) != 1:
        raise ComparisonAdapterError(
            f"{execution_id}: activation execution row is not unique."
        )
    return selected[0]


def _verified_package_manifest(path: Path, *, label: str) -> dict[str, Any]:
    manifest = _verified_json(path, label=label)
    source = manifest.get("source_archive")
    if (
        manifest.get("status") != "passed"
        or not isinstance(source, Mapping)
    ):
        raise ComparisonAdapterError(f"{label} source binding is unavailable.")
    return manifest


def _validate_preservation(
    spec: ArmSpec,
    archive_binding: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not spec.preservation_required:
        return None
    snapshot = _verified_json(
        PRESERVATION_SNAPSHOT,
        label="completed-v1 preservation snapshot",
    )
    rows = snapshot.get("rows")
    selected = [
        row
        for row in rows
        if isinstance(rows, list)
        and isinstance(row, Mapping)
        and row.get("execution_id") == spec.execution_id
    ]
    if len(selected) != 1:
        raise ComparisonAdapterError(
            f"{spec.execution_id}: preservation row is not unique."
        )
    row = selected[0]
    _verify_self_digest(
        row,
        label=f"{spec.execution_id} preservation row",
    )
    verification = row.get("local_verification")
    if (
        snapshot.get("status") != "passed"
        or snapshot.get(
            "all_archive_size_sha256_gzip_tar_authority_checks_passed"
        )
        is not True
        or row.get("status") != "passed"
        or row.get("archive") != {
            key: archive_binding[key]
            for key in ("path", "sha256", "size_bytes")
        }
        or not isinstance(verification, Mapping)
        or verification.get("status") != "passed"
        or verification.get("gzip_and_full_tar_scan_passed") is not True
        or verification.get("regular_member_closure_passed") is not True
        or verification.get("authority_bindings_passed") is not True
        or verification.get("worker_exit_status") != 0
    ):
        raise ComparisonAdapterError(
            f"{spec.execution_id}: preservation binding failed."
        )
    return {
        "snapshot": _file_binding(
            PRESERVATION_SNAPSHOT,
            canonical_sha256=str(snapshot["sha256"]),
        ),
        "row_canonical_sha256": row["sha256"],
        "scientific_result_disposition": row[
            "scientific_result_disposition"
        ],
    }


def _validate_authority_and_protocol(
    spec: ArmSpec,
    scanned: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    job = scanned["job"]
    authorization = scanned["authorization"]
    activation = scanned["activation"]
    receipt = scanned["receipt"]
    if not all(
        isinstance(value, Mapping)
        for value in (job, authorization, activation, receipt)
    ):
        raise ComparisonAdapterError(
            f"{spec.execution_id}: archived authority payload is malformed."
        )

    local_job_path = (
        SCIENTIFIC_PACKAGE / "jobs" / f"{spec.execution_id}.json"
    )
    local_job = _verified_json(
        local_job_path,
        label=f"{spec.execution_id} local scientific job",
    )
    if local_job != job:
        raise ComparisonAdapterError(
            f"{spec.execution_id}: archived scientific job bytes drifted."
        )
    local_authorization_path = (
        spec.activation_root
        / "authorizations"
        / f"{spec.execution_id}.json"
    )
    local_activation_path = spec.activation_root / "activation_manifest.json"
    local_authorization = _verified_json(
        local_authorization_path,
        label=f"{spec.execution_id} local authorization",
    )
    local_activation = _verified_json(
        local_activation_path,
        label=f"{spec.execution_id} local activation",
    )
    if local_authorization != authorization or local_activation != activation:
        raise ComparisonAdapterError(
            f"{spec.execution_id}: archived authority differs from local seal."
        )
    observed = scanned["observed_members"]
    if not isinstance(observed, Mapping):
        raise ComparisonAdapterError(
            f"{spec.execution_id}: observed member map is malformed."
        )
    job_binding = _verify_local_binding(
        local_job_path,
        {
            **observed[JOB_MEMBER],
            "canonical_sha256": job["sha256"],
        },
        label=f"{spec.execution_id} job",
        canonical_sha256=str(job["sha256"]),
    )
    authorization_binding = _verify_local_binding(
        local_authorization_path,
        {
            **observed[AUTHORIZATION_MEMBER],
            "canonical_sha256": authorization["sha256"],
        },
        label=f"{spec.execution_id} authorization",
        canonical_sha256=str(authorization["sha256"]),
    )
    activation_binding = _verify_local_binding(
        local_activation_path,
        {
            **observed[ACTIVATION_MEMBER],
            "canonical_sha256": activation["sha256"],
        },
        label=f"{spec.execution_id} activation",
        canonical_sha256=str(activation["sha256"]),
    )

    expected_identity = (
        f"{spec.execution_id}\t{spec.cluster_id}\t"
        f"{spec.proc_id}\t1\n"
    ).encode("ascii")
    row = _activation_execution_row(activation, spec.execution_id)
    row_job = row.get("job")
    row_authorization = row.get("authorization")
    if (
        scanned["attempt_identity"] != expected_identity
        or receipt.get("execution_id") != spec.execution_id
        or receipt.get("cluster_id") != spec.cluster_id
        or receipt.get("proc_id") != spec.proc_id
        or receipt.get("attempt_ordinal") != 1
        or receipt.get("worker_exit_status") != 0
        or receipt.get("job_file_sha256") != job_binding["sha256"]
        or receipt.get("authorization_file_sha256")
        != authorization_binding["sha256"]
        or receipt.get("activation_manifest_file_sha256")
        != activation_binding["sha256"]
        or not isinstance(row_job, Mapping)
        or not isinstance(row_authorization, Mapping)
        or row_job.get("canonical_sha256") != job["sha256"]
        or row_job.get("sha256") != job_binding["sha256"]
        or row_job.get("size_bytes") != job_binding["size_bytes"]
        or row_authorization.get("canonical_sha256")
        != authorization["sha256"]
        or row_authorization.get("sha256")
        != authorization_binding["sha256"]
        or row_authorization.get("size_bytes")
        != authorization_binding["size_bytes"]
    ):
        raise ComparisonAdapterError(
            f"{spec.execution_id}: attempt/activation binding failed."
        )

    if (
        job.get("schema")
        != "paper_i_ra_global_singleton_insertion_job_v1"
        or job.get("campaign_id") != CAMPAIGN_ID
        or job.get("execution_id") != spec.execution_id
        or job.get("cell_id") != spec.execution_id
        or job.get("route_id") != spec.route_id
        or job.get("insertion_policy") != spec.insertion_policy
        or job.get("insertion_runtime_mode")
        != spec.insertion_runtime_mode
        or job.get("regime_id") != REGIME_ID
        or job.get("nph") != NPH
        or job.get("horizon") != HORIZON
        or job.get("active_gradient_policy")
        != "stationary_source_response_v1"
        or job.get("resource_weighting_scope")
        != "all_phase_resource_weighting_v1"
        or job.get("phase1_cost_term") != "enabled"
        or authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("execution_id") != spec.execution_id
        or authorization.get("job_sha256") != job["sha256"]
        or authorization.get("job_file_sha256") != job_binding["sha256"]
        or authorization.get("execution_authorized") is not True
        or authorization.get("submission_authorized") is not True
        or activation.get("campaign_id") != CAMPAIGN_ID
        or activation.get("execution_authorized") is not True
        or activation.get("submission_authorized") is not True
    ):
        raise ComparisonAdapterError(
            f"{spec.execution_id}: job/authorization scientific axes drifted."
        )

    protocol, _protocol_path, protocol_binding = _protocol_for_arm(
        spec,
        authority,
        job,
    )
    embedded_protocol = scanned["result"].get("protocol")
    route_contract = protocol.get("route_contract")
    request = protocol.get("request")
    method = request.get("method") if isinstance(request, Mapping) else None
    insertion = (
        method.get("insertion") if isinstance(method, Mapping) else None
    )
    adapter = request.get("adapter") if isinstance(request, Mapping) else None
    if (
        embedded_protocol != protocol
        or protocol.get("horizon") != HORIZON
        or protocol.get("active_gradient_policy")
        != "stationary_source_response_v1"
        or protocol.get("resource_weighting_scope")
        != "all_phase_resource_weighting_v1"
        or not isinstance(route_contract, Mapping)
        or not isinstance(insertion, Mapping)
        or insertion.get("kind") != spec.insertion_policy
        or not isinstance(adapter, Mapping)
        or adapter.get("adapter_id")
        != (
            "paper_i_ra_adapt_global_single_pauli_word_"
            "candidate_adapter_v1"
        )
        or route_contract.get("execution_settings", {}).get(
            "adapt_insertion_mode"
        )
        != spec.insertion_runtime_mode
    ):
        raise ComparisonAdapterError(
            f"{spec.execution_id}: result/protocol identity drifted."
        )
    source_lock = _source_lock_row(spec, authority, protocol)

    scientific_manifest = _verified_package_manifest(
        SCIENTIFIC_PACKAGE / "package_manifest.json",
        label="scientific package manifest",
    )
    operational_manifest = _verified_package_manifest(
        spec.operational_package / "package_manifest.json",
        label=f"{spec.key} operational package manifest",
    )
    scientific_source = scientific_manifest["source_archive"]
    operational_source = operational_manifest["source_archive"]
    scientific_source_path = SCIENTIFIC_PACKAGE / str(
        scientific_source["path"]
    )
    operational_source_path = spec.operational_package / str(
        operational_source["path"]
    )
    scientific_source_binding = _verify_local_binding(
        scientific_source_path,
        scientific_source,
        label="scientific source archive",
    )
    operational_source_binding = _verify_local_binding(
        operational_source_path,
        operational_source,
        label=f"{spec.key} operational source archive",
    )
    if (
        job.get("source_archive", {}).get("sha256")
        != scientific_source_binding["sha256"]
        or authorization.get("source_archive_sha256")
        != operational_source_binding["sha256"]
        or receipt.get("source_archive_sha256")
        != operational_source_binding["sha256"]
    ):
        raise ComparisonAdapterError(
            f"{spec.execution_id}: source archive identity drifted."
        )
    if spec.key == "plateau":
        if (
            operational_manifest.get("scientific_parent_package_id")
            != job.get("package_id")
            or operational_manifest.get("scientific_settings_changed") != []
            or operational_manifest.get("protocol_files_byte_identical")
            is not True
            or operational_manifest.get(
                "non_checkpoint_source_members_byte_identical"
            )
            is not True
            or authorization.get("scientific_parent_job_bytes_identical")
            is not True
            or authorization.get("scientific_settings_changed") != []
            or activation.get("scientific_job_bytes_identical") is not True
            or activation.get("scientific_protocol_bytes_identical")
            is not True
        ):
            raise ComparisonAdapterError(
                "Plateau checkpoint-retention supersession drifted "
                "scientifically."
            )
    elif operational_source_binding != scientific_source_binding:
        raise ComparisonAdapterError(
            "Append operational/scientific source binding drifted."
        )

    preservation = _validate_preservation(spec, scanned["archive"])
    return {
        "job": job,
        "protocol": protocol,
        "source": {
            "archive": {
                **scanned["archive"],
                "full_regular_member_scan": True,
                "worker_receipt_member_closure": "passed",
            },
            "result": {
                "member": RESULT_MEMBER,
                **observed[RESULT_MEMBER],
                "parser": "ijson_stream_projection_v1",
            },
            "worker_attempt_receipt": {
                "member": ATTEMPT_RECEIPT_MEMBER,
                **observed[ATTEMPT_RECEIPT_MEMBER],
                "canonical_sha256": receipt["sha256"],
            },
            "job": job_binding,
            "authorization": authorization_binding,
            "activation_manifest": activation_binding,
            "protocol": protocol_binding,
            "scientific_source_archive": scientific_source_binding,
            "operational_source_archive": operational_source_binding,
            "source_lock": source_lock,
            "preservation": preservation,
        },
    }


def _minimal_result_for_prefix(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    plateau = result["effective_plateau"]
    plateau_round = _integer(
        plateau.get("controller_round"),
        label="effective plateau round",
        minimum=1,
    )
    replay_rows: list[dict[str, Any]] = [{} for _ in range(HORIZON)]
    for round_index in {plateau_round, HORIZON}:
        replay_rows[round_index - 1] = result["selected_replay"][
            round_index
        ]
    return {
        "run": {
            "accepted_trajectory": result["trajectory"],
            "scientific_replay": replay_rows,
            "canonical_reporting": {
                "accepted_prefix_work": result["prefix_work"],
                "reference_state": result["reference_state"],
            },
            "route": result["route"],
            "problem": result["problem"],
        }
    }


def _compiled_cost(
    *,
    prefix: Any,
    k: int,
    energy: float,
    error: float,
) -> dict[str, Any]:
    from pipelines.reporting import (
        build_paper_i_ra_adapt_stationary_core_master_pdf as master,
    )

    try:
        resources, checkpoint_sha256, payload = (
            master._compile_prefix_qiskit(prefix, compiler=None)
        )
    except Exception as exc:
        raise ComparisonAdapterError(
            f"Qiskit prefix compilation failed at k={k}: {exc}"
        ) from exc
    return {
        "k": k,
        "energy": energy,
        "error": error,
        "S_alg": int(prefix.algorithmic_work.s_alg),
        "N2q": resources["N2q"],
        "D2q": resources["D2q"],
        "Dc": resources["Dc"],
        "W1q": resources["W1q"],
        "B1q": resources["B1q"],
        "compile_convention": "table_i_basis_gate_transpile_v1",
        "qiskit_basis_work_status": resources[
            "qiskit_basis_work_status"
        ],
        "qiskit_basis_work_schema": resources[
            "qiskit_basis_work_schema"
        ],
        "checkpoint_sha256": checkpoint_sha256,
        "qiskit_version": payload.get("qiskit_version"),
    }


def _project_science(
    spec: ArmSpec,
    result: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("available_rounds") != HORIZON
        or result.get("protocol") != protocol
    ):
        raise ComparisonAdapterError(
            f"{spec.execution_id}: result projection identity drifted."
        )
    exact_energy = _finite(
        result.get("exact_energy"),
        label=f"{spec.execution_id} exact energy",
    )
    transitions = result["transitions"]
    trace = result["error_trace"]
    trajectory = result["trajectory"]
    work_rows = result["prefix_work"]
    points: list[dict[str, Any]] = []
    insertion_rows: list[tuple[int, int, str]] = []
    initial_energy: float | None = None
    for index in range(HORIZON):
        round_index = index + 1
        transition = transitions[index]
        trace_row = trace[index]
        state = trajectory[index]
        work = work_rows[index]
        observed_rounds = (
            _integer(
                transition.get("controller_round"),
                label="transition round",
                minimum=1,
            ),
            _integer(
                trace_row.get("controller_round"),
                label="trace round",
                minimum=1,
            ),
            _integer(
                state.get("controller_round"),
                label="trajectory round",
                minimum=1,
            ),
        )
        if observed_rounds != (round_index, round_index, round_index):
            raise ComparisonAdapterError(
                f"{spec.execution_id}: result round ordering drifted."
            )
        energy_before = _finite(
            transition.get("energy_before"),
            label="transition energy before",
        )
        energy_after = _finite(
            transition.get("energy_after"),
            label="transition energy after",
        )
        accepted_energy = _finite(
            trace_row.get("accepted_energy"),
            label="trace accepted energy",
        )
        error = _finite(
            trace_row.get("absolute_energy_error"),
            label="trace absolute energy error",
        )
        if (
            not math.isclose(
                energy_after,
                accepted_energy,
                abs_tol=1.0e-12,
                rel_tol=1.0e-11,
            )
            or not math.isclose(
                error,
                abs(accepted_energy - exact_energy),
                abs_tol=1.0e-12,
                rel_tol=1.0e-11,
            )
            or (
                index > 0
                and not math.isclose(
                    energy_before,
                    _finite(
                        transitions[index - 1].get("energy_after"),
                        label="previous transition energy",
                    ),
                    abs_tol=1.0e-12,
                    rel_tol=1.0e-11,
                )
            )
        ):
            raise ComparisonAdapterError(
                f"{spec.execution_id}: accepted trace arithmetic drifted."
            )
        if index == 0:
            initial_energy = energy_before
        position = _integer(
            transition.get("insertion_position"),
            label="accepted insertion position",
            minimum=0,
        )
        append_position = index
        if position > append_position:
            raise ComparisonAdapterError(
                f"{spec.execution_id}: insertion position exceeds append."
            )
        position_class = (
            "append" if position == append_position else "interior"
        )
        insertion_rows.append((round_index, position, position_class))
        cumulative_s_alg = _integer(
            transition.get("cumulative_s_alg"),
            label="transition cumulative S_alg",
            minimum=0,
        )
        work_s_alg = _integer(
            work.get("s_alg"),
            label="accepted prefix S_alg",
            minimum=0,
        )
        components = work.get("components")
        if (
            cumulative_s_alg != work_s_alg
            or not isinstance(components, Mapping)
            or sum(
                _integer(
                    components.get(key),
                    label=f"work component {key}",
                    minimum=0,
                )
                for key in (
                    "n_h_outer",
                    "n_h_refit",
                    "n_grad",
                    "n_metric",
                )
            )
            != work_s_alg
        ):
            raise ComparisonAdapterError(
                f"{spec.execution_id}: accepted-prefix work drifted."
            )
        points.append({"k": round_index, "error": error})
    if initial_energy is None:
        raise AssertionError("The fixed 50-round trace cannot be empty.")
    points.insert(
        0,
        {"k": 0, "error": abs(initial_energy - exact_energy)},
    )

    from pipelines.reporting.paper_i_run_summary import (
        PaperIErrorTracePoint,
        select_paper_i_effective_plateau,
    )

    selected = select_paper_i_effective_plateau(
        tuple(
            PaperIErrorTracePoint(
                controller_round=row["k"],
                absolute_energy_error=row["error"],
            )
            for row in points[1:]
        )
    )
    plateau = result["effective_plateau"]
    plateau_round = _integer(
        plateau.get("controller_round"),
        label="serialized effective plateau round",
        minimum=1,
    )
    plateau_error = _finite(
        plateau.get("absolute_energy_error"),
        label="serialized effective plateau error",
    )
    if (
        selected.controller_round != plateau_round
        or not math.isclose(
            selected.absolute_energy_error,
            plateau_error,
            abs_tol=1.0e-14,
            rel_tol=1.0e-12,
        )
        or plateau.get("policy") != "paper_i_effective_plateau_v1"
        or plateau.get("status") != "available"
    ):
        raise ComparisonAdapterError(
            f"{spec.execution_id}: effective plateau selection drifted."
        )
    canonical_all_work = result["canonical_all_work"]
    terminal_s_alg = _integer(
        canonical_all_work.get("s_alg"),
        label="canonical terminal S_alg",
        minimum=0,
    )
    if terminal_s_alg != work_rows[-1]["s_alg"]:
        raise ComparisonAdapterError(
            f"{spec.execution_id}: canonical all-work terminal drifted."
        )
    stop = result["stop"]
    if (
        stop.get("completed_controller_rounds") != HORIZON
        or stop.get("accepted_operator_count") != HORIZON
        or stop.get("primary_reason") != "maximum_controller_rounds"
    ):
        raise ComparisonAdapterError(
            f"{spec.execution_id}: completed stop receipt drifted."
        )
    route = result["route"]
    problem = result["problem"]
    protocol_route = protocol.get("route_contract")
    if (
        not isinstance(protocol_route, Mapping)
        or route.get("contract_sha256") != protocol_route.get("sha256")
        or problem != protocol.get("problem")
    ):
        raise ComparisonAdapterError(
            f"{spec.execution_id}: run route/problem binding drifted."
        )

    from pipelines.reporting import (
        build_paper_i_ra_adapt_stationary_core_master_pdf as master,
    )

    minimal = _minimal_result_for_prefix(result)
    try:
        terminal_prefix = master._ra_prefix(
            minimal,
            controller_round=HORIZON,
            expected_s_alg=terminal_s_alg,
        )
        plateau_prefix = master._ra_prefix(
            minimal,
            controller_round=plateau_round,
            expected_s_alg=int(
                plateau.get("algorithmic_work", {}).get("s_alg", -1)
            ),
        )
    except Exception as exc:
        raise ComparisonAdapterError(
            f"{spec.execution_id}: authenticated prefix reconstruction "
            f"failed: {exc}"
        ) from exc
    serialized_prefix = plateau.get("prefix")
    if not isinstance(serialized_prefix, Mapping):
        raise ComparisonAdapterError(
            f"{spec.execution_id}: plateau prefix serialization is absent."
        )
    terminal_cost = _compiled_cost(
        prefix=terminal_prefix,
        k=HORIZON,
        energy=_finite(
            trace[-1].get("accepted_energy"),
            label="terminal accepted energy",
        ),
        error=float(points[-1]["error"]),
    )
    plateau_cost = _compiled_cost(
        prefix=plateau_prefix,
        k=plateau_round,
        energy=_finite(
            trace[plateau_round - 1].get("accepted_energy"),
            label="effective plateau accepted energy",
        ),
        error=plateau_error,
    )
    serialized_resources = plateau.get("resources")
    if (
        not isinstance(serialized_resources, Mapping)
        or serialized_resources.get("compile_convention")
        != plateau_cost["compile_convention"]
        or serialized_resources.get("compiled_two_qubit_count")
        != plateau_cost["N2q"]
        or serialized_resources.get("compiled_two_qubit_depth")
        != plateau_cost["D2q"]
        or serialized_resources.get("compiled_total_depth")
        != plateau_cost["Dc"]
    ):
        raise ComparisonAdapterError(
            f"{spec.execution_id}: serialized/recompiled plateau cost drifted."
        )

    interior_rows = [
        row for row in insertion_rows if row[2] == "interior"
    ]
    append_rows = [row for row in insertion_rows if row[2] == "append"]
    if spec.key == "append" and interior_rows:
        raise ComparisonAdapterError(
            "Append-commutation arm selected an interior position."
        )
    if spec.key == "plateau" and not interior_rows:
        raise ComparisonAdapterError(
            "Plateau-commutation arm never exercised an interior position."
        )
    return {
        "points": points,
        "terminal": terminal_cost,
        "effective_plateau": plateau_cost,
        "insertion_counts": {
            "round_count": HORIZON,
            "append_count": len(append_rows),
            "interior_count": len(interior_rows),
            "first_interior_round": (
                None if not interior_rows else interior_rows[0][0]
            ),
        },
        "qualification": {
            "status": "passed",
            "result_schema": RESULT_SCHEMA,
            "full_controller_rounds": HORIZON,
            "same_cutoff_trace_math": "passed",
            "canonical_work_closure": "passed",
            "authenticated_prefix_reconstruction": "passed",
            "serialized_plateau_qiskit_cross_check": "passed",
            "route_domain_status": (
                "append_only_passed"
                if spec.key == "append"
                else "plateau_interior_exercised"
            ),
            "exact_same_cutoff_energy": exact_energy,
            "route_profile": route.get("profile"),
            "route_contract_sha256": route.get("contract_sha256"),
            "problem_request_sha256": problem.get(
                "problem_request_sha256"
            ),
        },
    }


def _arm_row(
    spec: ArmSpec,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    scanned = _scan_archive(RETRIEVED_ROOT / spec.archive_name)
    validated = _validate_authority_and_protocol(
        spec,
        scanned,
        authority,
    )
    science = _project_science(
        spec,
        scanned["result"],
        validated["protocol"],
    )
    return _digested(
        {
            "schema": ARM_SCHEMA,
            "execution_id": spec.execution_id,
            "route_id": spec.route_id,
            "insertion_policy": spec.insertion_policy,
            "points": science["points"],
            "terminal": science["terminal"],
            "effective_plateau": science["effective_plateau"],
            "insertion_counts": science["insertion_counts"],
            "source": validated["source"],
            "qualification": science["qualification"],
        }
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        raise ComparisonAdapterError("Comparison ratio is not finite.")
    if denominator == 0:
        raise ComparisonAdapterError("Comparison ratio denominator is zero.")
    return numerator / denominator


def _comparison(arms: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(arms) != 2:
        raise ComparisonAdapterError("Comparison requires exactly two arms.")
    by_policy = {
        str(row.get("insertion_policy")): row for row in arms
    }
    append = by_policy.get("append_commutation_reduced")
    plateau = by_policy.get("plateau_commutation")
    if not isinstance(append, Mapping) or not isinstance(plateau, Mapping):
        raise ComparisonAdapterError(
            "Comparison arms do not cover the exact insertion pair."
        )
    terminal_append = append["terminal"]
    terminal_plateau = plateau["terminal"]
    effective_append = append["effective_plateau"]
    effective_plateau = plateau["effective_plateau"]
    append_exact = append["qualification"]["exact_same_cutoff_energy"]
    plateau_exact = plateau["qualification"]["exact_same_cutoff_energy"]
    if append_exact != plateau_exact:
        raise ComparisonAdapterError(
            "Comparison arms disagree on same-cutoff exact energy."
        )

    def differences(
        left: Mapping[str, Any],
        right: Mapping[str, Any],
    ) -> dict[str, Any]:
        keys = ("error", "S_alg", "N2q", "D2q", "Dc", "W1q", "B1q")
        return {
            "plateau_minus_append": {
                key: right[key] - left[key] for key in keys
            },
            "plateau_over_append": {
                key: _safe_ratio(float(right[key]), float(left[key]))
                for key in keys
            },
        }

    return {
        "comparison_order": [
            "append_commutation_reduced",
            "plateau_commutation",
        ],
        "same_cutoff_exact_energy": append_exact,
        "terminal": differences(terminal_append, terminal_plateau),
        "effective_plateau": differences(
            effective_append,
            effective_plateau,
        ),
        "effective_plateau_round_delta": (
            effective_plateau["k"] - effective_append["k"]
        ),
        "interior_insertion_count_delta": (
            plateau["insertion_counts"]["interior_count"]
            - append["insertion_counts"]["interior_count"]
        ),
    }


def build_adapter(output: Path) -> dict[str, Any]:
    authority, cross_arm_audit = _validated_package_authority()
    arms = [_arm_row(spec, authority) for spec in ARM_SPECS]
    if (
        [row["insertion_policy"] for row in arms]
        != ["append_commutation_reduced", "plateau_commutation"]
        or any(len(row["points"]) != HORIZON + 1 for row in arms)
        or any(
            [point["k"] for point in row["points"]]
            != list(range(HORIZON + 1))
            for row in arms
        )
    ):
        raise ComparisonAdapterError(
            "Final comparison arm ordering or trace closure drifted."
        )
    adapter = _digested(
        {
            "schema": ADAPTER_SCHEMA,
            "status": "passed",
            "diagnostic_only": True,
            "paper_evidence_adopted": False,
            "campaign_id": CAMPAIGN_ID,
            "regime_id": REGIME_ID,
            "nph": NPH,
            "horizon": HORIZON,
            "cross_arm_audit": cross_arm_audit,
            "arms": arms,
            "comparison": _comparison(arms),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ComparisonAdapterError(
            f"Refusing stale output temporary: {temporary}"
        )
    raw = _canonical_json_bytes(adapter) + b"\n"
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return adapter


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        output.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ComparisonAdapterError(
            "Comparison adapter output must remain in the active repository."
        ) from exc
    adapter = build_adapter(output)
    print(
        json.dumps(
            {
                "status": adapter["status"],
                "output": _relative(output),
                "sha256": adapter["sha256"],
                "arm_count": len(adapter["arms"]),
                "insertion_policies": [
                    row["insertion_policy"] for row in adapter["arms"]
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
