"""Fail-closed CHTC operations adapter for the acquisition supervisor.

This module accepts only an already-authenticated OpenSSH ControlMaster socket.
It verifies the socket in batch mode and never falls back to a new login or a
Duo action. A caller supplies scientific lane handlers; this module owns the
transport, one combined queue/history capture, immutable raw evidence,
intent-aware reconciliation, and guarded Condor submission.

The first cycle that discovers a planner unit records the supervisor intent but
cannot submit it because its identity was not in the captured query. The next
cycle queries that durable intent and can prove "found" or "verified_absent"
without a second reconciliation query.
"""

from __future__ import annotations

import datetime as _dt
import base64
import binascii
import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
import tarfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol, Sequence

from pipelines.time_dynamics.acquisition_supervisor import (
    RAW_SCHEDULER_EVIDENCE_SCHEMA_V1,
    SNAPSHOT_SCHEMA_V1,
    SUBMISSION_INTENT_SCHEMA_V1,
    CampaignRegistry,
    CycleGuardFailed,
    LaneHandlers,
    LaneRegistration,
    ReconciliationResult,
    RemoteAccessPending,
    SchedulerScope,
    SchedulerSnapshot,
    SubmissionUnit,
)
from pipelines.time_dynamics.guarded_archive_transfer import (
    ArchiveTransferSpec,
    TransferRecoveryError,
    enforce_local_storage_floor,
    read_finalization_receipt,
    stream_archive_transfer,
    verify_final_archive,
    verified_file_sha256,
)

ADAPTER_ID = "chtc_authenticated_controlmaster_v1"
REMOTE_CAPTURE_SCHEMA_V1 = "chtc_combined_condor_capture_v1"
RECONCILIATION_SCHEMA_V1 = "chtc_submission_reconciliation_v1"
TRANSPORT_STATE_SCHEMA_V1 = "chtc_acquisition_transport_state_v1"

_SAFE_REMOTE_RELATIVE = re.compile(r"[A-Za-z0-9_.\-/]+")
_SAFE_AD_VALUE = re.compile(r"[A-Za-z0-9_.:-]+")
_CONDOR_ACTIVE = {1: "idle", 2: "running", 5: "held", 6: "running", 7: "held"}


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
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_immutable_text(path: Path, value: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != value:
            raise CycleGuardFailed("scheduler_evidence_identity_collision", str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            if path.read_text(encoding="utf-8") != value:
                raise CycleGuardFailed(
                    "scheduler_evidence_identity_collision", str(path)
                )
        else:
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            temporary = ""
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class CommandRunner(Protocol):
    """Internal test seam; argv is always executed without a local shell."""

    def __call__(
        self, argv: tuple[str, ...], *, timeout_seconds: int
    ) -> CommandResult: ...


def subprocess_command_runner(
    argv: tuple[str, ...], *, timeout_seconds: int
) -> CommandResult:
    completed = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return CommandResult(
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


@dataclass(frozen=True)
class AuthenticatedControlMaster:
    """An existing authenticated transport, never a login mechanism."""

    socket_path: Path
    endpoint: str
    runner: CommandRunner = subprocess_command_runner
    ssh_binary: str = "/usr/bin/ssh"
    timeout_seconds: int = 120

    def __post_init__(self) -> None:
        if not self.socket_path.is_absolute():
            raise ValueError("ControlMaster socket path must be absolute")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+", self.endpoint):
            raise ValueError("CHTC endpoint must be a fixed user@host value")
        if self.timeout_seconds <= 0:
            raise ValueError("transport timeout must be positive")

    def check(self) -> None:
        try:
            result = self.runner(
                (
                    self.ssh_binary,
                    "-F",
                    "/dev/null",
                    "-S",
                    str(self.socket_path),
                    "-o",
                    "BatchMode=yes",
                    "-O",
                    "check",
                    self.endpoint,
                ),
                timeout_seconds=min(self.timeout_seconds, 30),
            )
        except UnicodeError as exc:
            raise RemoteAccessPending(
                "remote transport returned non-UTF-8 output"
            ) from exc
        except (OSError, subprocess.TimeoutExpired, TimeoutError) as exc:
            raise RemoteAccessPending(str(exc)[:500]) from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "socket check failed").strip()
            raise RemoteAccessPending(detail[:500])

    def run_remote_python(self, source: str, arguments: Sequence[str]) -> CommandResult:
        self.check()
        remote_command = shlex.join(
            ("python3", "-c", source, *(str(value) for value in arguments))
        )
        try:
            result = self.runner(
                (
                    self.ssh_binary,
                    "-F",
                    "/dev/null",
                    "-S",
                    str(self.socket_path),
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ControlMaster=no",
                    "-o",
                    f"ControlPath={self.socket_path}",
                    "-o",
                    "ControlPersist=no",
                    "-o",
                    "NumberOfPasswordPrompts=0",
                    "-o",
                    "PreferredAuthentications=none",
                    "-o",
                    "PasswordAuthentication=no",
                    "-o",
                    "KbdInteractiveAuthentication=no",
                    "-o",
                    "PubkeyAuthentication=no",
                    "-o",
                    "IdentitiesOnly=yes",
                    "-o",
                    "IdentityAgent=none",
                    "-o",
                    "HostbasedAuthentication=no",
                    "-o",
                    "GSSAPIAuthentication=no",
                    "-o",
                    "ProxyCommand=/usr/bin/false",
                    "-o",
                    "ProxyJump=none",
                    "-o",
                    "PermitLocalCommand=no",
                    "-o",
                    "ClearAllForwardings=yes",
                    "-o",
                    "ForwardAgent=no",
                    "-o",
                    "RequestTTY=no",
                    self.endpoint,
                    remote_command,
                ),
                timeout_seconds=self.timeout_seconds,
            )
        except UnicodeError as exc:
            raise RemoteAccessPending(
                "remote transport returned non-UTF-8 output"
            ) from exc
        except (OSError, subprocess.TimeoutExpired, TimeoutError) as exc:
            raise RemoteAccessPending(str(exc)[:500]) from exc
        if result.returncode == 255:
            detail = (result.stderr or result.stdout or "mux transport lost").strip()
            raise RemoteAccessPending(detail[:500])
        return result

    @staticmethod
    def _remote_json(result: CommandResult, *, label: str) -> Mapping[str, Any]:
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or f"{label} failed").strip()
            raise CycleGuardFailed(f"remote_{label}_failed", detail[:1000])
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise CycleGuardFailed(f"remote_{label}_unreadable", str(exc)) from exc
        if not isinstance(payload, Mapping) or payload.get("ok") is not True:
            raise CycleGuardFailed(f"remote_{label}_unverifiable")
        return payload

    def archive_identity(self, remote_path: str) -> Mapping[str, Any]:
        result = self.run_remote_python(_REMOTE_ARCHIVE_IDENTITY, (remote_path,))
        payload = self._remote_json(result, label="archive_identity")
        try:
            size = int(payload["bytes"])
            member_count = int(payload["member_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CycleGuardFailed("remote_archive_identity_unverifiable") from exc
        digest = str(payload.get("sha256") or "")
        if (
            size <= 0
            or member_count <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or payload.get("gzip_integrity") is not True
            or payload.get("tar_integrity") is not True
            or payload.get("path") != remote_path
        ):
            raise CycleGuardFailed("remote_archive_identity_unverifiable")
        return {
            "path": remote_path,
            "bytes": size,
            "sha256": digest,
            "member_count": member_count,
            "gzip_integrity": True,
            "tar_integrity": True,
        }

    def archive_prefix_sha256(self, remote_path: str, length: int) -> str:
        result = self.run_remote_python(
            _REMOTE_ARCHIVE_PREFIX,
            (remote_path, str(int(length))),
        )
        payload = self._remote_json(result, label="archive_prefix")
        digest = str(payload.get("sha256") or "")
        try:
            observed_length = int(payload.get("length", -1))
        except (TypeError, ValueError) as exc:
            raise CycleGuardFailed("remote_archive_prefix_unverifiable") from exc
        if (
            payload.get("path") != remote_path
            or observed_length != int(length)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise CycleGuardFailed("remote_archive_prefix_unverifiable")
        return digest

    def archive_chunk(self, remote_path: str, offset: int, length: int) -> bytes:
        result = self.run_remote_python(
            _REMOTE_ARCHIVE_CHUNK,
            (remote_path, str(int(offset)), str(int(length))),
        )
        payload = self._remote_json(result, label="archive_chunk")
        try:
            observed_offset = int(payload.get("offset", -1))
            observed_length = int(payload.get("length", -1))
        except (TypeError, ValueError) as exc:
            raise CycleGuardFailed("remote_archive_chunk_unverifiable") from exc
        if (
            payload.get("path") != remote_path
            or observed_offset != int(offset)
            or observed_length != int(length)
        ):
            raise CycleGuardFailed("remote_archive_chunk_unverifiable")
        try:
            decoded = base64.b64decode(
                str(payload.get("base64") or ""), validate=True
            )
        except (ValueError, TypeError, binascii.Error) as exc:
            raise CycleGuardFailed("remote_archive_chunk_unverifiable") from exc
        digest = str(payload.get("sha256") or "")
        if len(decoded) != int(length) or hashlib.sha256(decoded).hexdigest() != digest:
            raise CycleGuardFailed("remote_archive_chunk_unverifiable")
        return decoded


@dataclass(frozen=True)
class CHTCOperationsConfig:
    repository_root: Path
    transport: AuthenticatedControlMaster
    owner: str = "jsstrobel"
    evidence_root: Path = Path("output/time_dynamics_acquisition/scheduler_evidence")
    bootstrap_action_ids: Mapping[tuple[str, int, int], str] = field(
        default_factory=dict
    )
    remote_workdirs: Mapping[str, str] = field(default_factory=dict)
    archive_transfers: Mapping[
        tuple[str, int, tuple[int, ...]], ArchiveTransferSpec
    ] = field(default_factory=dict)
    history_since_epoch: int = 0

    def __post_init__(self) -> None:
        if not self.repository_root.is_absolute():
            raise ValueError("repository_root must be absolute")
        if self.evidence_root.is_absolute() or ".." in self.evidence_root.parts:
            raise ValueError("evidence_root must be repository-relative")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.owner):
            raise ValueError("invalid Condor owner")
        if self.history_since_epoch < 0:
            raise ValueError("history_since_epoch must be nonnegative")
        for key, action_id in self.bootstrap_action_ids.items():
            if len(key) != 3 or not action_id:
                raise ValueError("bootstrap action bindings must be scheduler keys")
        for lane_id, workdir in self.remote_workdirs.items():
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", lane_id):
                raise ValueError("invalid lane remote-workdir key")
            path = PurePosixPath(workdir)
            if not path.is_absolute() or ".." in path.parts:
                raise ValueError("remote workdirs must be absolute canonical paths")
        for key, spec in self.archive_transfers.items():
            if (
                len(key) != 3
                or not str(key[0])
                or int(key[1]) < 0
                or not isinstance(key[2], tuple)
                or not key[2]
                or tuple(sorted(set(int(value) for value in key[2]))) != key[2]
            ):
                raise ValueError("archive-transfer keys must bind lane/cluster/processes")
            if not isinstance(spec, ArchiveTransferSpec):
                raise TypeError("archive-transfer bindings must be ArchiveTransferSpec")
            self._validate_archive_spec_paths(spec)

    def _validate_archive_spec_paths(self, spec: ArchiveTransferSpec) -> None:
        """Confine every mutable path to the registered attempt directory."""

        root_lexical = self.repository_root.absolute()
        root = root_lexical.resolve()
        final = spec.local_final_path
        if ".." in final.parts:
            raise ValueError("archive-transfer paths must be canonical")
        try:
            final.absolute().relative_to(root_lexical)
            final_resolved = final.resolve(strict=False)
            final_resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                "archive-transfer local paths must remain inside repository_root"
            ) from exc
        attempt_root_lexical = final.absolute().parent
        attempt_root = final_resolved.parent
        candidates = (
            final,
            spec.part_path,
            spec.effective_quarantine_root,
            spec.effective_recovery_receipt_path,
            spec.effective_finalization_intent_path,
            spec.effective_finalization_receipt_path,
        )
        for candidate in candidates:
            if ".." in candidate.parts:
                raise ValueError("archive-transfer paths must be canonical")
            try:
                lexical = candidate.absolute()
                lexical.relative_to(root_lexical)
                lexical.relative_to(attempt_root_lexical)
                resolved = candidate.resolve(strict=False)
                resolved.relative_to(root)
                resolved.relative_to(attempt_root)
            except ValueError as exc:
                raise ValueError(
                    "archive-transfer auxiliary paths must remain inside the "
                    "registered attempt directory"
                ) from exc
            current = root_lexical
            for part in lexical.relative_to(root_lexical).parts:
                current = current / part
                if current.is_symlink():
                    raise ValueError(
                        "archive-transfer paths may not traverse symlinks"
                    )


_REMOTE_CAPTURE = r"""
import json
import subprocess
import sys
import time

owner = sys.argv[1]
scope_rows = json.loads(sys.argv[2])
intent_rows = json.loads(sys.argv[3])
history_since = int(sys.argv[4])
scope_terms = []
for scope in scope_rows:
    processes = ",".join(str(int(value)) for value in scope["process_ids"])
    scope_terms.append(
        "(ClusterId==%d && member(ProcId,{%s}))"
        % (int(scope["cluster_id"]), processes)
    )
identity_terms = []
for row in intent_rows:
    unit = row["unit_id"].replace('"', "")
    batch = row.get("job_batch_name", "").replace('"', "")
    identity_terms.append('(AcquisitionUnitId=="%s")' % unit)
    if batch:
        identity_terms.append('(JobBatchName=="%s")' % batch)
constraint = " || ".join(scope_terms + identity_terms) or "False"
attributes = (
    "ClusterId,ProcId,JobStatus,ExitCode,HoldReason,JobBatchName,"
    "AcquisitionLaneId,AcquisitionActionId,AcquisitionUnitId"
)

def invoke(argv):
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    return {
        "argv": argv,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }

queue = invoke([
    "condor_q", owner, "-constraint", constraint, "-json", "-attributes", attributes
])
history_constraint = "(%s) && EnteredCurrentStatus >= %d" % (
    constraint, history_since
)
history = invoke([
    "condor_history", owner, "-constraint", history_constraint,
    "-json", "-attributes", attributes
])
print(json.dumps({
    "schema": "chtc_combined_condor_capture_v1",
    "captured_at_unix_ns": time.time_ns(),
    "scope_rows": scope_rows,
    "intent_rows": intent_rows,
    "queue": queue,
    "history": history,
}, sort_keys=True))
"""


_REMOTE_ARCHIVE_IDENTITY = r"""
import gzip
import hashlib
import json
import os
import stat
import sys
import tarfile

path = sys.argv[1]
st = os.stat(path, follow_symlinks=False)
if not os.path.isabs(path) or os.path.normpath(path) != path or not stat.S_ISREG(st.st_mode):
    raise SystemExit(40)
digest = hashlib.sha256()
with open(path, "rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
with gzip.open(path, "rb") as handle:
    while handle.read(1024 * 1024):
        pass
with tarfile.open(path, "r:gz") as archive:
    members = sum(1 for _ in archive)
print(json.dumps({
    "ok": True,
    "path": path,
    "bytes": st.st_size,
    "sha256": digest.hexdigest(),
    "member_count": members,
    "gzip_integrity": True,
    "tar_integrity": True,
}, sort_keys=True))
"""


_REMOTE_ARCHIVE_PREFIX = r"""
import hashlib
import json
import os
import stat
import sys

path = sys.argv[1]
length = int(sys.argv[2])
st = os.stat(path, follow_symlinks=False)
if (
    not os.path.isabs(path)
    or os.path.normpath(path) != path
    or not stat.S_ISREG(st.st_mode)
    or length < 0
    or length > st.st_size
):
    raise SystemExit(41)
digest = hashlib.sha256()
remaining = length
with open(path, "rb") as handle:
    while remaining:
        chunk = handle.read(min(1024 * 1024, remaining))
        if not chunk:
            raise SystemExit(42)
        digest.update(chunk)
        remaining -= len(chunk)
print(json.dumps({
    "ok": True,
    "path": path,
    "length": length,
    "sha256": digest.hexdigest(),
}, sort_keys=True))
"""


_REMOTE_ARCHIVE_CHUNK = r"""
import base64
import hashlib
import json
import os
import stat
import sys

path = sys.argv[1]
offset = int(sys.argv[2])
length = int(sys.argv[3])
st = os.stat(path, follow_symlinks=False)
if (
    not os.path.isabs(path)
    or os.path.normpath(path) != path
    or not stat.S_ISREG(st.st_mode)
    or offset < 0
    or length <= 0
    or length > 8 * 1024 * 1024
    or offset + length > st.st_size
):
    raise SystemExit(43)
with open(path, "rb") as handle:
    handle.seek(offset)
    data = handle.read(length)
if len(data) != length:
    raise SystemExit(44)
print(json.dumps({
    "ok": True,
    "path": path,
    "offset": offset,
    "length": length,
    "sha256": hashlib.sha256(data).hexdigest(),
    "base64": base64.b64encode(data).decode("ascii"),
}, sort_keys=True))
"""


_REMOTE_SUBMIT = r"""
import json
import os
import re
import subprocess
import sys
import tempfile

workdir, descriptor, unit_id, lane_id, batch_name = sys.argv[1:6]
expected_count = int(sys.argv[6])

def invoke(argv):
    result = subprocess.run(
        argv, cwd=workdir, capture_output=True, text=True, check=False
    )
    return {
        "argv": argv,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }

lifecycle = invoke([
    "python3", "chtc/validate_condor_submit_lifecycle.py", descriptor
])
if lifecycle["returncode"] != 0:
    print(json.dumps({"status": "blocked", "phase": "lifecycle", "result": lifecycle}))
    raise SystemExit(20)
with tempfile.NamedTemporaryFile(
    prefix="acquisition-dryrun-", suffix=".ad", dir=workdir, delete=False
) as handle:
    dryrun_path = handle.name
try:
    dryrun = invoke(["condor_submit", "-dry-run", dryrun_path, descriptor])
    if dryrun["returncode"] != 0:
        print(json.dumps({"status": "blocked", "phase": "dryrun", "result": dryrun}))
        raise SystemExit(21)
    with open(dryrun_path, encoding="utf-8", errors="replace") as handle:
        dryrun_text = handle.read()
    if batch_name not in dryrun_text:
        print(json.dumps({"status": "blocked", "phase": "batch_identity"}))
        raise SystemExit(22)
finally:
    try:
        os.unlink(dryrun_path)
    except FileNotFoundError:
        pass
submit = invoke([
    "condor_submit",
    "-append", '+AcquisitionUnitId = "%s"' % unit_id,
    "-append", '+AcquisitionLaneId = "%s"' % lane_id,
    descriptor,
])
if submit["returncode"] != 0:
    print(json.dumps({"status": "blocked", "phase": "submit", "result": submit}))
    raise SystemExit(23)
match = re.search(r"cluster\s+(\d+)", submit["stdout"], re.IGNORECASE)
if match is None:
    print(json.dumps({"status": "blocked", "phase": "cluster_parse", "result": submit}))
    raise SystemExit(24)
cluster_id = int(match.group(1))
accepted = invoke([
    "condor_q", str(cluster_id), "-json",
    "-attributes", "ClusterId,ProcId,JobStatus"
])
if accepted["returncode"] != 0:
    print(json.dumps({"status": "blocked", "phase": "acceptance", "result": accepted}))
    raise SystemExit(25)
rows = json.loads(accepted["stdout"] or "[]")
process_ids = sorted(int(row["ProcId"]) for row in rows)
if process_ids != list(range(expected_count)):
    print(json.dumps({
        "status": "blocked",
        "phase": "acceptance_scope",
        "cluster_id": cluster_id,
        "process_ids": process_ids,
    }))
    raise SystemExit(26)
print(json.dumps({
    "status": "submitted",
    "cluster_id": cluster_id,
    "process_ids": process_ids,
    "job_batch_name": batch_name,
    "unit_id": unit_id,
    "initial_queue_count": len(process_ids),
}, sort_keys=True))
"""


class CHTCAcquisitionOperations:
    """Concrete CHTC adapter at the supervisor production seam."""

    adapter_id = ADAPTER_ID

    def __init__(
        self,
        config: CHTCOperationsConfig,
        lane_handlers: Mapping[str, LaneHandlers],
    ) -> None:
        self._config = config
        self._base_handlers = dict(lane_handlers)
        self._snapshot: SchedulerSnapshot | None = None
        self._expected_scopes: tuple[SchedulerScope, ...] = ()
        self._verified_absent_units: set[str] = set()

    @property
    def _transport_state_path(self) -> Path:
        return (
            self._config.repository_root
            / "output/time_dynamics_acquisition/production_transport_state.json"
        )

    def _record_transport_state(self, status: str, **detail: Any) -> None:
        _atomic_write_json(
            self._transport_state_path,
            {
                "schema": TRANSPORT_STATE_SCHEMA_V1,
                "adapter_id": self.adapter_id,
                "status": status,
                "updated_at": _utc_now(),
                **detail,
            },
        )

    def _pending_intents(
        self, registry: CampaignRegistry
    ) -> tuple[dict[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for lane in registry.lanes:
            root = lane.state_root / "submission_intents"
            for path in sorted(root.glob("*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise CycleGuardFailed(
                        "submission_intent_unverifiable", f"{path}: {exc}"
                    ) from exc
                unit = payload.get("unit") if isinstance(payload, Mapping) else None
                if (
                    payload.get("schema") != SUBMISSION_INTENT_SCHEMA_V1
                    or payload.get("lane_id") != lane.lane_id
                    or not isinstance(unit, Mapping)
                ):
                    raise CycleGuardFailed("submission_intent_unverifiable", str(path))
                unit_id = str(unit.get("unit_id") or "")
                action_ids = tuple(str(value) for value in unit.get("action_ids") or ())
                unit_payload = unit.get("payload")
                batch_name = (
                    str(unit_payload.get("job_batch_name") or "")
                    if isinstance(unit_payload, Mapping)
                    else ""
                )
                if (
                    not unit_id
                    or unit_id in seen
                    or not action_ids
                    or not _SAFE_AD_VALUE.fullmatch(unit_id)
                    or (batch_name and not _SAFE_AD_VALUE.fullmatch(batch_name))
                ):
                    raise CycleGuardFailed("submission_intent_unverifiable", str(path))
                seen.add(unit_id)
                rows.append(
                    {
                        "lane_id": lane.lane_id,
                        "unit_id": unit_id,
                        "action_ids": list(action_ids),
                        "action_ids_sha256": _digest(list(action_ids)),
                        "job_batch_name": batch_name,
                    }
                )
        return tuple(rows)

    def _known_action_ids(
        self, registry: CampaignRegistry
    ) -> dict[tuple[str, int, int], str]:
        known = dict(self._config.bootstrap_action_ids)
        for lane in registry.lanes:
            receipts = lane.state_root / "submission_receipts"
            intents = lane.state_root / "submission_intents"
            for receipt_path in sorted(receipts.glob("*.json")):
                intent_path = intents / receipt_path.name
                if not intent_path.is_file():
                    raise CycleGuardFailed(
                        "submission_scope_unverifiable",
                        f"{receipt_path}: matching intent missing",
                    )
                try:
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    intent = json.loads(intent_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise CycleGuardFailed(
                        "submission_scope_unverifiable", f"{receipt_path}: {exc}"
                    ) from exc
                unit = intent.get("unit") if isinstance(intent, Mapping) else None
                remote = receipt.get("remote") if isinstance(receipt, Mapping) else None
                if not isinstance(unit, Mapping) or not isinstance(remote, Mapping):
                    raise CycleGuardFailed(
                        "submission_scope_unverifiable", str(receipt_path)
                    )
                actions = tuple(str(value) for value in unit.get("action_ids") or ())
                try:
                    cluster_id = int(remote["cluster_id"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise CycleGuardFailed(
                        "submission_scope_unverifiable", str(receipt_path)
                    ) from exc
                raw_processes = remote.get("process_ids")
                processes = (
                    tuple(range(len(actions)))
                    if raw_processes is None
                    else tuple(int(value) for value in raw_processes)
                )
                if len(actions) != len(processes):
                    raise CycleGuardFailed(
                        "submission_scope_unverifiable", str(receipt_path)
                    )
                for process_id, action_id in zip(processes, actions):
                    key = (lane.lane_id, cluster_id, process_id)
                    if key in known and known[key] != action_id:
                        raise CycleGuardFailed(
                            "scheduler_action_binding_conflict", repr(key)
                        )
                    known[key] = action_id
        return known

    def capture_snapshot(
        self,
        registry: CampaignRegistry,
        *,
        expected_scopes: tuple[SchedulerScope, ...],
    ) -> SchedulerSnapshot:
        intents = self._pending_intents(registry)
        scope_payload = [scope.to_json_dict() for scope in expected_scopes]
        try:
            result = self._config.transport.run_remote_python(
                _REMOTE_CAPTURE,
                (
                    self._config.owner,
                    json.dumps(scope_payload, sort_keys=True, separators=(",", ":")),
                    json.dumps(intents, sort_keys=True, separators=(",", ":")),
                    str(self._config.history_since_epoch),
                ),
            )
        except RemoteAccessPending as exc:
            self._record_transport_state("remote_access_pending", detail=exc.detail)
            raise
        if result.returncode != 0:
            detail = (
                result.stderr or result.stdout or "capture command failed"
            ).strip()
            self._record_transport_state(
                "scheduler_capture_failed", detail=detail[:1000]
            )
            raise CycleGuardFailed("scheduler_capture_failed", detail[:1000])
        try:
            envelope = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            self._record_transport_state("scheduler_capture_unreadable")
            raise CycleGuardFailed("scheduler_capture_unreadable", str(exc)) from exc
        snapshot = self._snapshot_from_envelope(
            registry=registry,
            envelope=envelope,
            expected_scopes=expected_scopes,
            intents=intents,
        )
        self._snapshot = snapshot
        self._expected_scopes = expected_scopes
        self._verified_absent_units.clear()
        self._record_transport_state(
            "snapshot_captured",
            captured_at=snapshot.captured_at,
            snapshot_sha256=snapshot.digest,
            expected_scope_sha256=_digest(scope_payload),
        )
        return snapshot

    def _snapshot_from_envelope(
        self,
        *,
        registry: CampaignRegistry,
        envelope: Mapping[str, Any],
        expected_scopes: tuple[SchedulerScope, ...],
        intents: tuple[dict[str, Any], ...],
    ) -> SchedulerSnapshot:
        expected_scope_rows = [
            scope.to_json_dict() for scope in expected_scopes
        ]
        if envelope.get("schema") != REMOTE_CAPTURE_SCHEMA_V1:
            raise CycleGuardFailed("scheduler_capture_unverifiable", "wrong schema")
        if envelope.get("scope_rows") != expected_scope_rows:
            raise CycleGuardFailed("scheduler_capture_scope_mismatch")
        if envelope.get("intent_rows") != list(intents):
            raise CycleGuardFailed("scheduler_capture_intent_mismatch")
        queries: dict[str, tuple[str, list[dict[str, Any]]]] = {}
        for name in ("queue", "history"):
            row = envelope.get(name)
            if not isinstance(row, Mapping) or int(row.get("returncode", -1)) != 0:
                detail = (
                    str(row.get("stderr") or "")[:1000]
                    if isinstance(row, Mapping)
                    else "missing query result"
                )
                raise CycleGuardFailed(f"scheduler_{name}_query_failed", detail)
            raw = str(row.get("stdout") or "")
            try:
                parsed = json.loads(raw or "[]")
            except json.JSONDecodeError as exc:
                raise CycleGuardFailed(
                    "scheduler_capture_unreadable", f"{name}: {exc}"
                ) from exc
            if not isinstance(parsed, list) or any(
                not isinstance(item, dict) for item in parsed
            ):
                raise CycleGuardFailed(
                    "scheduler_capture_unreadable",
                    f"{name}: expected a JSON object array",
                )
            queries[name] = (raw, parsed)

        evidence_key = _digest(
            {
                "scope_rows": envelope.get("scope_rows"),
                "intent_rows": envelope.get("intent_rows"),
                "captured_at_unix_ns": envelope.get("captured_at_unix_ns"),
                "queue": queries["queue"][0],
                "history": queries["history"][0],
            }
        )
        evidence_dir = (
            self._config.repository_root / self._config.evidence_root / evidence_key
        )
        raw_bindings: dict[str, dict[str, str]] = {}
        for name, (raw, _) in queries.items():
            path = evidence_dir / f"{name}.raw.json"
            _write_immutable_text(path, raw)
            raw_bindings[name] = {
                "reference": str(path.relative_to(self._config.repository_root)),
                "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            }

        expected_keys = {
            (scope.lane_id, scope.cluster_id, process_id)
            for scope in expected_scopes
            for process_id in scope.process_ids
        }
        lane_by_cluster: dict[int, str] = {}
        for scope in expected_scopes:
            prior = lane_by_cluster.setdefault(scope.cluster_id, scope.lane_id)
            if prior != scope.lane_id:
                raise CycleGuardFailed(
                    "scheduler_cluster_lane_ambiguous", str(scope.cluster_id)
                )
        known_actions = self._known_action_ids(registry)
        by_query: dict[str, dict[tuple[str, int, int], dict[str, Any]]] = {
            "queue": {},
            "history": {},
        }
        all_rows = [*queries["queue"][1], *queries["history"][1]]
        for name in ("queue", "history"):
            for raw in queries[name][1]:
                try:
                    cluster_id = int(raw["ClusterId"])
                    process_id = int(raw["ProcId"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise CycleGuardFailed(
                        "scheduler_capture_unreadable",
                        f"{name}: scheduler identity",
                    ) from exc
                lane_id = lane_by_cluster.get(cluster_id)
                if lane_id is None:
                    continue
                key = (lane_id, cluster_id, process_id)
                if key not in expected_keys:
                    continue
                if key in by_query[name]:
                    raise CycleGuardFailed(
                        "scheduler_capture_duplicate_job", repr(key)
                    )
                by_query[name][key] = raw

        jobs: list[dict[str, Any]] = []
        for key in sorted(expected_keys):
            raw = by_query["queue"].get(key) or by_query["history"].get(key)
            if raw is None:
                raise CycleGuardFailed("scheduler_scope_incomplete", repr(key))
            remote_action = str(raw.get("AcquisitionActionId") or "")
            action_id = remote_action or known_actions.get(key, "")
            if not action_id:
                raise CycleGuardFailed(
                    "scheduler_action_binding_missing", repr(key)
                )
            if (
                remote_action
                and key in known_actions
                and known_actions[key] != remote_action
            ):
                raise CycleGuardFailed(
                    "scheduler_action_binding_conflict", repr(key)
                )
            state, exit_code = self._scheduler_state(raw)
            jobs.append(
                {
                    "lane_id": key[0],
                    "cluster_id": key[1],
                    "process_id": key[2],
                    "state": state,
                    "exit_code": exit_code,
                    "action_id": action_id,
                    "detail": {
                        "job_batch_name": str(raw.get("JobBatchName") or ""),
                        "acquisition_unit_id": str(
                            raw.get("AcquisitionUnitId") or ""
                        ),
                        "hold_reason": str(raw.get("HoldReason") or ""),
                    },
                }
            )

        reconciliation_units: dict[str, dict[str, Any]] = {}
        for intent in intents:
            unit_id = intent["unit_id"]
            batch_name = intent["job_batch_name"]
            matches = [
                raw
                for raw in all_rows
                if str(raw.get("AcquisitionUnitId") or "") == unit_id
                or (
                    batch_name
                    and str(raw.get("JobBatchName") or "") == batch_name
                )
            ]
            if not matches:
                reconciliation_units[unit_id] = {
                    "status": "verified_absent",
                    "action_ids_sha256": intent["action_ids_sha256"],
                    "job_batch_name": batch_name,
                }
                continue
            try:
                clusters = {int(raw["ClusterId"]) for raw in matches}
                processes = sorted({int(raw["ProcId"]) for raw in matches})
            except (KeyError, TypeError, ValueError):
                clusters = set()
                processes = []
            if (
                len(clusters) != 1
                or processes != list(range(len(intent["action_ids"])))
            ):
                reconciliation_units[unit_id] = {
                    "status": "indeterminate",
                    "action_ids_sha256": intent["action_ids_sha256"],
                    "detail": "remote identity has conflicting cluster/process scope",
                }
            else:
                reconciliation_units[unit_id] = {
                    "status": "found",
                    "action_ids_sha256": intent["action_ids_sha256"],
                    "remote": {
                        "cluster_id": next(iter(clusters)),
                        "process_ids": processes,
                        "unit_id": unit_id,
                        "job_batch_name": batch_name,
                        "action_ids": list(intent["action_ids"]),
                    },
                }

        captured_at_ns = int(envelope.get("captured_at_unix_ns", 0))
        if captured_at_ns <= 0:
            raise CycleGuardFailed(
                "scheduler_capture_unverifiable", "invalid timestamp"
            )
        captured_at = _dt.datetime.fromtimestamp(
            captured_at_ns / 1_000_000_000,
            tz=_dt.timezone.utc,
        ).astimezone().isoformat(timespec="microseconds")
        return SchedulerSnapshot.from_mapping(
            {
                "schema": SNAPSHOT_SCHEMA_V1,
                "captured_at": captured_at,
                "source": {
                    "queue_query_succeeded": True,
                    "history_query_succeeded": True,
                    "scope_complete": True,
                    "adapter_id": self.adapter_id,
                    "expected_processes": expected_scope_rows,
                    "raw_scheduler_evidence": {
                        "schema": RAW_SCHEDULER_EVIDENCE_SCHEMA_V1,
                        **raw_bindings,
                    },
                    "submission_reconciliation": {
                        "schema": RECONCILIATION_SCHEMA_V1,
                        "units": reconciliation_units,
                    },
                },
                "jobs": jobs,
            }
        )

    @staticmethod
    def _scheduler_state(raw: Mapping[str, Any]) -> tuple[str, int | None]:
        try:
            status = int(raw["JobStatus"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CycleGuardFailed(
                "scheduler_capture_unreadable", "JobStatus"
            ) from exc
        exit_raw = raw.get("ExitCode")
        exit_code = None if exit_raw in (None, "") else int(exit_raw)
        if status in _CONDOR_ACTIVE:
            return _CONDOR_ACTIVE[status], exit_code
        if status == 3:
            return "removed", exit_code
        if status == 4:
            return (
                "succeeded" if exit_code in (None, 0) else "failed",
                exit_code,
            )
        raise CycleGuardFailed("scheduler_status_unsupported", str(status))

    def _reconcile_submission(
        self,
        lane: LaneRegistration,
        unit: SubmissionUnit,
        snapshot: SchedulerSnapshot,
    ) -> ReconciliationResult:
        del lane
        if snapshot is not self._snapshot:
            return ReconciliationResult.indeterminate(
                "not the adapter current snapshot"
            )
        root = snapshot.source.get("submission_reconciliation")
        units = root.get("units") if isinstance(root, Mapping) else None
        row = units.get(unit.unit_id) if isinstance(units, Mapping) else None
        if not isinstance(row, Mapping):
            return ReconciliationResult.indeterminate(
                "unit identity was not included in the captured scheduler query"
            )
        if row.get("action_ids_sha256") != _digest(list(unit.action_ids)):
            return ReconciliationResult.indeterminate(
                "unit action identity changed"
            )
        status = str(row.get("status") or "")
        if status == "found" and isinstance(row.get("remote"), Mapping):
            return ReconciliationResult.found(row["remote"])
        if status == "verified_absent":
            self._verified_absent_units.add(unit.unit_id)
            return ReconciliationResult.verified_absent(
                "exact unit and batch identity absent from captured queue/history"
            )
        return ReconciliationResult.indeterminate(
            str(row.get("detail") or "")
        )

    def _submit_unit(
        self, lane: LaneRegistration, unit: SubmissionUnit
    ) -> Mapping[str, Any]:
        if (
            self._snapshot is None
            or unit.unit_id not in self._verified_absent_units
        ):
            raise CycleGuardFailed(
                "submission_scope_not_verified",
                "unit lacks current verified-absent scheduler proof",
            )
        if self._snapshot.expected_scopes != self._expected_scopes:
            raise CycleGuardFailed(
                "submission_scope_not_verified", "scope digest changed"
            )
        payload = dict(unit.payload)
        batch_name = str(payload.get("job_batch_name") or "")
        descriptor = str(payload.get("submit_descriptor") or "")
        expected_count = int(payload.get("expected_process_count", -1))
        scope = payload.get("submission_scope")
        workdir = self._config.remote_workdirs.get(lane.lane_id, "")
        if (
            not _SAFE_AD_VALUE.fullmatch(unit.unit_id)
            or not batch_name
            or not _SAFE_AD_VALUE.fullmatch(batch_name)
            or not descriptor
            or not _SAFE_REMOTE_RELATIVE.fullmatch(descriptor)
            or PurePosixPath(descriptor).is_absolute()
            or ".." in PurePosixPath(descriptor).parts
            or expected_count != len(unit.action_ids)
            or not isinstance(scope, Mapping)
            or scope.get("action_ids") != list(unit.action_ids)
            or scope.get("process_ids") != list(range(expected_count))
            or not workdir
        ):
            raise CycleGuardFailed(
                "submission_unit_contract_unverifiable", unit.unit_id
            )
        try:
            result = self._config.transport.run_remote_python(
                _REMOTE_SUBMIT,
                (
                    workdir,
                    descriptor,
                    unit.unit_id,
                    lane.lane_id,
                    batch_name,
                    str(expected_count),
                ),
            )
        except RemoteAccessPending as exc:
            self._record_transport_state(
                "remote_access_pending", detail=exc.detail
            )
            raise
        if result.returncode != 0:
            detail = (
                result.stdout or result.stderr or "submit command failed"
            ).strip()
            self._record_transport_state(
                "submission_blocked",
                unit_id=unit.unit_id,
                detail=detail[:2000],
            )
            raise CycleGuardFailed(
                "chtc_submission_blocked", detail[:2000]
            )
        try:
            remote = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise CycleGuardFailed(
                "chtc_submission_receipt_unreadable", str(exc)
            ) from exc
        if (
            not isinstance(remote, Mapping)
            or remote.get("status") != "submitted"
            or remote.get("unit_id") != unit.unit_id
            or remote.get("job_batch_name") != batch_name
            or remote.get("process_ids") != list(range(expected_count))
        ):
            raise CycleGuardFailed(
                "chtc_submission_receipt_unverifiable", unit.unit_id
            )
        normalized = {
            "cluster_id": int(remote["cluster_id"]),
            "process_ids": list(range(expected_count)),
            "unit_id": unit.unit_id,
            "job_batch_name": batch_name,
            "action_ids": list(unit.action_ids),
            "initial_queue_count": int(
                remote.get("initial_queue_count", -1)
            ),
        }
        if normalized["initial_queue_count"] != expected_count:
            raise CycleGuardFailed(
                "chtc_submission_scope_mismatch", unit.unit_id
            )
        self._record_transport_state(
            "unit_submitted", unit_id=unit.unit_id, remote=normalized
        )
        return normalized

    def fetch_verified_archive(
        self,
        lane_id: str,
        cluster_id: int,
        process_ids: tuple[int, ...],
        *,
        chunk_bytes: int = 1 << 20,
    ) -> Mapping[str, Any]:
        """Fetch one registered immutable archive through its exact ``.part``.

        The remote object is never renamed or deleted.  Authentication is
        restricted to the configured existing ControlMaster; loss of that
        transport records ``remote_access_pending`` and fails closed.
        """

        key = (str(lane_id), int(cluster_id), tuple(int(v) for v in process_ids))
        spec = self._config.archive_transfers.get(key)
        if spec is None:
            raise CycleGuardFailed("archive_transfer_binding_missing", repr(key))
        self._config._validate_archive_spec_paths(spec)
        try:
            identity = self._config.transport.archive_identity(spec.remote_path)
            if (
                int(identity["bytes"]) != spec.expected_bytes
                or identity["sha256"] != spec.expected_sha256
                or int(identity["member_count"]) != spec.expected_member_count
            ):
                raise CycleGuardFailed(
                    "remote_archive_binding_mismatch", spec.remote_path
                )
            receipt = stream_archive_transfer(
                spec,
                remote_prefix_sha256=lambda length: (
                    self._config.transport.archive_prefix_sha256(
                        spec.remote_path, length
                    )
                ),
                remote_chunk=lambda offset, length: (
                    self._config.transport.archive_chunk(
                        spec.remote_path, offset, length
                    )
                ),
                chunk_bytes=chunk_bytes,
                extraction_reserve_bytes=spec.extraction_reserve_bytes,
            )
            final_identity = self._config.transport.archive_identity(
                spec.remote_path
            )
            if final_identity != identity:
                raise CycleGuardFailed(
                    "remote_archive_changed_during_transfer", spec.remote_path
                )
        except RemoteAccessPending as exc:
            self._record_transport_state(
                "remote_access_pending",
                phase="archive_transfer",
                remote_path=spec.remote_path,
                detail=exc.detail,
            )
            raise
        except (
            CycleGuardFailed,
            TransferRecoveryError,
            OSError,
            ValueError,
            TimeoutError,
            subprocess.TimeoutExpired,
            zlib.error,
        ) as exc:
            self._record_transport_state(
                "archive_transfer_blocked",
                lane_id=key[0],
                cluster_id=key[1],
                process_ids=list(key[2]),
                remote_path=spec.remote_path,
                exception_type=type(exc).__name__,
                detail=str(exc)[:1000],
            )
            raise
        try:
            archive_binding = self._verified_archive_binding(key, spec, receipt)
        except (CycleGuardFailed, OSError, ValueError, tarfile.TarError) as exc:
            self._record_transport_state(
                "archive_transfer_blocked",
                lane_id=key[0],
                cluster_id=key[1],
                process_ids=list(key[2]),
                remote_path=spec.remote_path,
                exception_type=type(exc).__name__,
                detail=str(exc)[:1000],
            )
            if isinstance(exc, CycleGuardFailed):
                raise
            raise CycleGuardFailed(
                "archive_finalization_unverifiable", str(exc)[:1000]
            ) from exc
        self._record_transport_state(
            "archive_fetched",
            remote_path=spec.remote_path,
            local_final_path=str(spec.local_final_path),
            archive_bytes=spec.expected_bytes,
            archive_sha256=spec.expected_sha256,
            archive_member_count=spec.expected_member_count,
            remote_preserved=True,
        )
        return {
            **dict(receipt),
            "remote_identity": dict(final_identity),
            "remote_preserved": True,
            "verified_archive": archive_binding,
        }

    def _verified_archive_binding(
        self,
        key: tuple[str, int, tuple[int, ...]],
        spec: ArchiveTransferSpec,
        receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Reverify and bind the exact final artifact passed to ingestion."""

        self._config._validate_archive_spec_paths(spec)
        receipt_path = spec.effective_finalization_receipt_path
        final = spec.local_final_path
        try:
            verified = dict(verify_final_archive(spec))
        except TransferRecoveryError as exc:
            raise CycleGuardFailed(
                "archive_finalization_integrity_failed", str(exc)[:1000]
            ) from exc
        if (
            verified.get("archive_bytes") != spec.expected_bytes
            or verified.get("archive_sha256") != spec.expected_sha256
            or verified.get("tar_member_count")
            != spec.expected_member_count
        ):
            raise CycleGuardFailed("archive_finalization_binding_mismatch")
        try:
            disk_receipt = dict(read_finalization_receipt(spec))
            receipt_file_sha256 = verified_file_sha256(spec, receipt_path)
        except TransferRecoveryError as exc:
            raise CycleGuardFailed(
                "archive_finalization_receipt_unverifiable", str(exc)[:1000]
            ) from exc
        try:
            receipt_matches = (
                disk_receipt.get("schema")
                == "guarded_archive_transfer_finalization_v1"
                and disk_receipt.get("status") == "verified_and_finalized"
                and disk_receipt.get("archive_sha256") == spec.expected_sha256
                and int(disk_receipt.get("archive_bytes", -1))
                == spec.expected_bytes
                and int(disk_receipt.get("tar_member_count", -1))
                == spec.expected_member_count
                and disk_receipt.get("gzip_integrity") is True
                and disk_receipt.get("tar_integrity") is True
                and disk_receipt.get("remote_preservation_required") is True
                and re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(disk_receipt.get("receipt_sha256") or ""),
                )
                is not None
            )
        except (TypeError, ValueError):
            receipt_matches = False
        if not receipt_matches:
            raise CycleGuardFailed("archive_finalization_receipt_unverifiable")
        try:
            relative_final = final.absolute().relative_to(
                self._config.repository_root.absolute()
            ).as_posix()
            relative_receipt = receipt_path.absolute().relative_to(
                self._config.repository_root.absolute()
            ).as_posix()
        except ValueError as exc:
            raise CycleGuardFailed("archive_transfer_path_escape") from exc
        archive_binding = {
            "lane_id": key[0],
            "cluster_id": key[1],
            "process_ids": list(key[2]),
            "final_path": relative_final,
            "archive_bytes": spec.expected_bytes,
            "archive_sha256": spec.expected_sha256,
            "archive_member_count": spec.expected_member_count,
            "extraction_reserve_bytes": int(spec.extraction_reserve_bytes),
            "finalization_receipt": {
                "path": relative_receipt,
                "sha256": receipt_file_sha256,
                "payload_sha256": str(
                    disk_receipt.get("receipt_sha256") or ""
                ),
            },
            "remote_path": spec.remote_path,
            "remote_preserved": True,
        }
        archive_binding["verified_archive_binding_sha256"] = _digest(
            archive_binding
        )
        return archive_binding

    def _wrapped_ingestion_handler(
        self,
        base: LaneHandlers,
    ) -> Any:
        if base.ingest_delta is None:
            return None

        def ingest(
            lane: LaneRegistration,
            cluster_id: int,
            process_ids: tuple[int, ...],
            selected_jobs: tuple[Any, ...],
        ) -> Mapping[str, Any]:
            transfer = self.fetch_verified_archive(
                lane.lane_id, cluster_id, process_ids
            )
            verified_archive = dict(transfer["verified_archive"])
            key = (lane.lane_id, int(cluster_id), tuple(process_ids))
            spec = self._config.archive_transfers.get(key)
            if spec is None:
                raise CycleGuardFailed("archive_transfer_binding_missing", repr(key))
            before_ingest = self._verified_archive_binding(
                key, spec, transfer
            )
            if before_ingest != verified_archive:
                raise CycleGuardFailed("verified_archive_binding_changed")
            try:
                enforce_local_storage_floor(
                    spec,
                    required_write_bytes=0,
                    extraction_reserve_bytes=spec.extraction_reserve_bytes,
                )
            except TransferRecoveryError as exc:
                self._record_transport_state(
                    "archive_ingestion_blocked",
                    phase="pre_ingestion_storage_guard",
                    lane_id=key[0],
                    cluster_id=key[1],
                    process_ids=list(key[2]),
                    remote_path=spec.remote_path,
                    detail=str(exc)[:1000],
                )
                raise CycleGuardFailed(
                    "archive_ingestion_storage_guard_failed",
                    str(exc)[:1000],
                ) from exc
            # The scientific handler must explicitly accept and consume the
            # exact verified artifact.  It may not rediscover a different
            # archive from the filesystem after this boundary.
            outcome = dict(
                base.ingest_delta(  # type: ignore[call-arg]
                    lane,
                    cluster_id,
                    process_ids,
                    selected_jobs,
                    verified_archive=verified_archive,
                )
            )
            if outcome.get("consumed_verified_archive_sha256") != (
                verified_archive["verified_archive_binding_sha256"]
            ):
                raise CycleGuardFailed(
                    "ingestion_verified_archive_not_consumed",
                    repr((lane.lane_id, cluster_id, process_ids)),
                )
            consumed = outcome.get("consumed_verified_archive")
            expected_consumed = {
                "verified_archive_binding_sha256": verified_archive[
                    "verified_archive_binding_sha256"
                ],
                "final_path": verified_archive["final_path"],
                "archive_sha256": verified_archive["archive_sha256"],
                "finalization_receipt_sha256": verified_archive[
                    "finalization_receipt"
                ]["sha256"],
            }
            if consumed != expected_consumed:
                raise CycleGuardFailed(
                    "ingestion_verified_archive_consumption_unverifiable",
                    repr((lane.lane_id, cluster_id, process_ids)),
                )
            after_ingest_receipt = stream_archive_transfer(
                spec,
                remote_prefix_sha256=lambda _length: (_ for _ in ()).throw(
                    CycleGuardFailed("unexpected_remote_prefix_recheck")
                ),
                remote_chunk=lambda _offset, _length: (_ for _ in ()).throw(
                    CycleGuardFailed("unexpected_remote_chunk_recheck")
                ),
                extraction_reserve_bytes=spec.extraction_reserve_bytes,
            )
            after_ingest = self._verified_archive_binding(
                key, spec, after_ingest_receipt
            )
            if after_ingest != verified_archive:
                raise CycleGuardFailed("verified_archive_changed_during_ingestion")
            outcome["verified_archive_transfer"] = dict(transfer)
            return outcome

        return ingest

    def handlers_for(
        self, registry: CampaignRegistry
    ) -> Mapping[str, LaneHandlers]:
        handlers: dict[str, LaneHandlers] = {}
        for lane in registry.lanes:
            base = self._base_handlers.get(lane.lane_id, LaneHandlers())
            handlers[lane.lane_id] = LaneHandlers(
                ingest_delta=self._wrapped_ingestion_handler(base),
                refresh_status=base.refresh_status,
                plan_submissions=base.plan_submissions,
                reconcile_submission=self._reconcile_submission,
                submit_unit=self._submit_unit,
            )
        return handlers


def build_operations(
    registry: CampaignRegistry,
) -> CHTCAcquisitionOperations:
    """Fail closed until a reviewed immutable adapter configuration is bound."""

    from pipelines.time_dynamics.acquisition_registry_contract import (
        validate_registry_for_activation,
    )

    validate_registry_for_activation(registry)
    raise CycleGuardFailed(
        "production_operations_configuration_blocked",
        "no scientifically complete CHTC adapter configuration is bound",
    )


__all__ = [
    "ADAPTER_ID",
    "AuthenticatedControlMaster",
    "CHTCAcquisitionOperations",
    "CHTCOperationsConfig",
    "CommandResult",
    "CommandRunner",
    "build_operations",
    "subprocess_command_runner",
]
