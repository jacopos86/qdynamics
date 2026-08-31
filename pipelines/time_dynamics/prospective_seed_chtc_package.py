"""Deterministic, local-only CHTC packages for prospective seed actions.

The materializer is deliberately not a submission adapter.  It binds one
canonical seed work order to an exact ``git archive`` source tar, an immutable
source-epoch receipt, and one verified Apptainer image.  Its Condor descriptor
is an ordinary ``queue 1`` job and the emitted intent artifact is only an inert
template: no function in this module contacts a scheduler or authorizes a
submission.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from chtc.validate_condor_submit_lifecycle import validate_submit_lifecycle
from pipelines.artifacts.storage_guard import (
    DEFAULT_FREE_FLOOR_BYTES,
    FetchStorageAssessment,
    assess_fetch_storage,
)
from pipelines.time_dynamics.prospective_bootstrap_provenance import (
    BOOTSTRAP_INTENT_FIRST_REQUIREMENTS,
    BOOTSTRAP_PACKAGE_MANIFEST_V1,
    BootstrapPackageManifest,
)
from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    LANE_HH,
    LANE_HUBBARD,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    ExecutionRequest,
    SourceEpochReceipt,
    VerifiedFile,
)
from pipelines.time_dynamics.prospective_hh_seed_worker import (
    HHSeedWorkOrder,
    load_work_order as load_hh_work_order,
)
from pipelines.time_dynamics.prospective_hubbard_seed_worker import (
    HubbardSeedWorkOrder,
    load_work_order as load_hubbard_work_order,
)


RUNTIME_IMAGE_IDENTITY_V1 = "paper_ii_runtime_image_identity_v1"
SEED_PACKAGE_PAYLOAD_MANIFEST_V1 = "paper_ii_seed_package_payload_manifest_v1"
SEED_SUBMISSION_INTENT_TEMPLATE_V1 = (
    "paper_ii_seed_submission_intent_template_v1"
)
SEED_PACKAGE_PREFLIGHT_V1 = "paper_ii_seed_package_local_preflight_v1"

_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9_.-]+")
_SAFE_TRANSFER_PATH = re.compile(r"[A-Za-z0-9_./-]+")
_SAFE_CLASSAD_VALUE = re.compile(r"[A-Za-z0-9_.:-]+")
_PACKAGE_NAME = "seed_action_package.tar.gz"
_MANIFEST_NAME = "package_manifest.json"
_DESCRIPTOR_NAME = "submit.sub"
_LAUNCHER_NAME = "launch.sh"
_INTENT_NAME = "submission_intent_template.json"
_PREFLIGHT_NAME = "local_preflight_receipt.json"
_INNER_MANIFEST = "input/package_payload_manifest.json"
_REQUEST_MEMBER = "input/execution_request.json"
_WORK_ORDER_MEMBER = "input/seed_work_order.json"
_SOURCE_RECEIPT_MEMBER = "input/source_epoch_receipt.json"
_SOURCE_TREE_MEMBER = "input/source_tree_manifest.json"
_IMAGE_IDENTITY_MEMBER = "input/runtime_image_identity.json"
_WORKER_MEMBER = "run_seed_action.sh"


@dataclass(frozen=True)
class _SeedLanePackageBinding:
    lane_id: str
    runner_module: str
    required_source_members: frozenset[str]
    work_order_loader: Callable[
        ...,
        tuple[HHSeedWorkOrder | HubbardSeedWorkOrder, SourceEpochReceipt],
    ]

    def load_work_order(
        self,
        path: str | Path,
        *,
        repository_root: str | Path,
    ) -> tuple[HHSeedWorkOrder | HubbardSeedWorkOrder, SourceEpochReceipt]:
        return self.work_order_loader(path, repository_root=repository_root)


_COMMON_REQUIRED_SOURCE_MEMBERS = frozenset(
    {"source/pipelines/time_dynamics/prospective_execution_bindings.py"}
)
_SEED_LANE_PACKAGE_BINDINGS = {
    LANE_HH: _SeedLanePackageBinding(
        lane_id=LANE_HH,
        runner_module="pipelines.time_dynamics.runners.build_prospective_hh_seed",
        required_source_members=_COMMON_REQUIRED_SOURCE_MEMBERS
        | {
            "source/pipelines/time_dynamics/prospective_hh_seed_worker.py",
            "source/pipelines/time_dynamics/runners/build_prospective_hh_seed.py",
        },
        work_order_loader=load_hh_work_order,
    ),
    LANE_HUBBARD: _SeedLanePackageBinding(
        lane_id=LANE_HUBBARD,
        runner_module=(
            "pipelines.time_dynamics.runners.build_prospective_hubbard_seed"
        ),
        required_source_members=_COMMON_REQUIRED_SOURCE_MEMBERS
        | {
            "source/pipelines/time_dynamics/prospective_hubbard_seed_worker.py",
            "source/pipelines/time_dynamics/pure_hubbard_fixed_vqe.py",
            "source/pipelines/time_dynamics/runners/build_prospective_hubbard_seed.py",
        },
        work_order_loader=load_hubbard_work_order,
    ),
}


class SeedPackageError(ValueError):
    """The seed package is stale, unsafe, noncanonical, or authorizing."""


def _seed_lane_binding(lane_id: str) -> _SeedLanePackageBinding:
    try:
        return _SEED_LANE_PACKAGE_BINDINGS[lane_id]
    except KeyError as exc:
        raise SeedPackageError(
            f"unsupported prospective seed lane: {lane_id}"
        ) from exc


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


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


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stream_sha(handle: Any) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: handle.read(1 << 20), b""):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _repository_path(
    root: Path,
    value: str | Path,
    *,
    label: str,
    must_exist: bool,
) -> Path:
    candidate = Path(value)
    target = candidate if candidate.is_absolute() else root / candidate
    target = target.resolve(strict=must_exist)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise SeedPackageError(f"{label} is outside repository root: {value}") from exc
    return target


def _relative(root: Path, path: Path, *, label: str) -> str:
    try:
        result = path.resolve(strict=False).relative_to(root).as_posix()
    except ValueError as exc:
        raise SeedPackageError(f"{label} is outside repository root: {path}") from exc
    if not result or not _SAFE_TRANSFER_PATH.fullmatch(result):
        raise SeedPackageError(f"{label} is not a transfer-safe relative path")
    return result


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise SeedPackageError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SeedPackageError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise SeedPackageError(f"{label} must be a JSON object")
    return value


def _seal(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = dict(payload)
    if key in result:
        raise SeedPackageError(f"payload already contains {key}")
    result[key] = _canonical_sha(result)
    return result


def _verify_seal(payload: Mapping[str, Any], key: str, *, label: str) -> str:
    core = dict(payload)
    declared = core.pop(key, None)
    actual = _canonical_sha(core)
    if declared != actual:
        raise SeedPackageError(f"{label} self digest mismatch")
    return actual


def _write_bytes(path: Path, content: bytes, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise SeedPackageError(f"refusing to overwrite immutable artifact: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o755 if executable else 0o644)
        os.replace(temporary, path)
        temporary = ""
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _verified_file(value: Any, *, label: str) -> VerifiedFile:
    if not isinstance(value, Mapping):
        raise SeedPackageError(f"{label} binding is missing")
    try:
        return VerifiedFile(
            path=str(value.get("path") or ""),
            sha256=str(value.get("sha256") or ""),
        )
    except (TypeError, ValueError) as exc:
        raise SeedPackageError(f"{label} binding is invalid") from exc


@dataclass(frozen=True)
class RuntimeImageIdentity:
    """Verified local bytes for one transferred Apptainer SIF image."""

    path: str
    sha256: str
    bytes: int
    identity_path: str
    identity_sha256: str
    identity_self_sha256: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": RUNTIME_IMAGE_IDENTITY_V1,
            "runtime_kind": "apptainer_sif",
            "image": {
                "path": self.path,
                "sha256": self.sha256,
                "bytes": self.bytes,
            },
            "requirements_expression": "TARGET.HasSIF",
            "qiskit_compilation_authorized": False,
            "runtime_image_identity_sha256": self.identity_self_sha256,
        }


def write_runtime_image_identity(
    path: str | Path,
    *,
    image_path: str | Path,
    repository_root: str | Path,
) -> RuntimeImageIdentity:
    """Freeze one existing local SIF path by byte count and SHA-256."""

    root = Path(repository_root).resolve(strict=True)
    image = _repository_path(
        root, image_path, label="runtime image", must_exist=True
    )
    if not image.is_file() or image.stat().st_size <= 0:
        raise SeedPackageError("runtime image is missing or empty")
    if image.suffix.casefold() != ".sif":
        raise SeedPackageError("runtime image must use the .sif artifact boundary")
    target = _repository_path(
        root, path, label="runtime image identity", must_exist=False
    )
    core = {
        "schema": RUNTIME_IMAGE_IDENTITY_V1,
        "runtime_kind": "apptainer_sif",
        "image": {
            "path": _relative(root, image, label="runtime image"),
            "sha256": _file_sha(image),
            "bytes": image.stat().st_size,
        },
        "requirements_expression": "TARGET.HasSIF",
        "qiskit_compilation_authorized": False,
    }
    _write_bytes(
        target,
        _pretty_bytes(_seal(core, "runtime_image_identity_sha256")),
    )
    return load_runtime_image_identity(target, repository_root=root)


def load_runtime_image_identity(
    path: str | Path, *, repository_root: str | Path
) -> RuntimeImageIdentity:
    root = Path(repository_root).resolve(strict=True)
    source = _repository_path(
        root, path, label="runtime image identity", must_exist=True
    )
    payload = _read_object(source, label="runtime image identity")
    self_sha = _verify_seal(
        payload,
        "runtime_image_identity_sha256",
        label="runtime image identity",
    )
    image_row = payload.get("image")
    if (
        payload.get("schema") != RUNTIME_IMAGE_IDENTITY_V1
        or payload.get("runtime_kind") != "apptainer_sif"
        or payload.get("requirements_expression") != "TARGET.HasSIF"
        or payload.get("qiskit_compilation_authorized") is not False
        or not isinstance(image_row, Mapping)
    ):
        raise SeedPackageError("runtime image identity is not canonical")
    image_file = _verified_file(image_row, label="runtime image")
    try:
        image_file.verify(root)
    except ValueError as exc:
        raise SeedPackageError(
            "runtime image digest/byte identity changed"
        ) from exc
    image = root / image_file.path
    try:
        image_bytes = int(image_row.get("bytes"))
    except (TypeError, ValueError) as exc:
        raise SeedPackageError("runtime image byte count is invalid") from exc
    if image_bytes <= 0 or image.stat().st_size != image_bytes:
        raise SeedPackageError("runtime image digest/byte identity changed")
    return RuntimeImageIdentity(
        path=image_file.path,
        sha256=image_file.sha256,
        bytes=image_bytes,
        identity_path=_relative(root, source, label="runtime image identity"),
        identity_sha256=_file_sha(source),
        identity_self_sha256=self_sha,
    )


@dataclass(frozen=True)
class SeedCHTCResources:
    """Operational resources recorded into one ordinary Condor descriptor."""

    request_cpus: int
    request_memory_gib: int
    request_disk_gib: int
    max_runtime_seconds: int

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise SeedPackageError(f"{name} must be a positive integer")

    def to_json_dict(self) -> dict[str, int]:
        return dict(asdict(self))

    @classmethod
    def from_mapping(cls, value: Any) -> "SeedCHTCResources":
        if not isinstance(value, Mapping):
            raise SeedPackageError("seed CHTC resources are missing")
        try:
            return cls(
                request_cpus=int(value["request_cpus"]),
                request_memory_gib=int(value["request_memory_gib"]),
                request_disk_gib=int(value["request_disk_gib"]),
                max_runtime_seconds=int(value["max_runtime_seconds"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SeedPackageError("seed CHTC resources are invalid") from exc


def _load_request(path: Path) -> ExecutionRequest:
    payload = _read_object(path, label="execution request")
    try:
        request = ExecutionRequest.from_mapping(payload)
    except (TypeError, ValueError) as exc:
        raise SeedPackageError("execution request is invalid") from exc
    if (
        request.kind != "build_seed"
        or request.to_json_dict().get("qiskit_compilation_requested") is not False
    ):
        raise SeedPackageError("request is not a no-Qiskit prospective seed action")
    _seed_lane_binding(request.lane_id)
    return request


def _reload_source_epoch(
    receipt: SourceEpochReceipt, *, root: Path
) -> SourceEpochReceipt:
    current = SourceEpochReceipt.load(
        root / receipt.receipt_path,
        repository_root=root,
        expected_epoch=receipt.source_epoch,
    )
    if current != receipt:
        raise SeedPackageError("source epoch receipt changed after validation")
    if (
        receipt.source_epoch.source_revision is None
        or not re.fullmatch(
            r"[0-9a-f]{40}(?:[0-9a-f]{24})?",
            receipt.source_epoch.source_revision,
        )
        or receipt.source_epoch.source_archive_sha256 is None
    ):
        raise SeedPackageError("source epoch lacks an exact commit/archive identity")
    return current


def _reload_image(identity: RuntimeImageIdentity, *, root: Path) -> RuntimeImageIdentity:
    current = load_runtime_image_identity(
        root / identity.identity_path, repository_root=root
    )
    if current != identity:
        raise SeedPackageError("runtime image digest/byte identity changed")
    return current


def _source_archive_members(
    path: Path, *, binding: _SeedLanePackageBinding
) -> tuple[str, ...]:
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            members = archive.getmembers()
    except (OSError, tarfile.TarError) as exc:
        raise SeedPackageError("runtime source archive is not readable tar/gzip") from exc
    files: list[str] = []
    seen: set[str] = set()
    for member in members:
        name = Path(member.name).as_posix()
        candidate = Path(name)
        if (
            not name
            or candidate.is_absolute()
            or ".." in candidate.parts
            or name in seen
            or not (member.isfile() or member.isdir())
        ):
            raise SeedPackageError(f"source archive has an unsafe member: {name!r}")
        seen.add(name)
        if member.isfile():
            files.append(name)
    missing = sorted(binding.required_source_members.difference(files))
    if missing:
        raise SeedPackageError(
            "source archive lacks the prospective seed worker seam: "
            + ", ".join(missing)
        )
    return tuple(files)


def _validate_source_archive(
    path: Path,
    *,
    receipt: SourceEpochReceipt,
    root: Path,
    binding: _SeedLanePackageBinding,
) -> tuple[str, ...]:
    revision = receipt.source_epoch.source_revision
    assert revision is not None
    try:
        process = subprocess.Popen(
            (
                "git",
                "archive",
                "--format=tar",
                "--prefix=source/",
                revision,
                "--",
                "pipelines",
                "src",
                "requirements.txt",
            ),
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise SeedPackageError("cannot reproduce exact runtime source archive") from exc
    assert process.stdout is not None
    expected_uncompressed_sha, _ = _stream_sha(process.stdout)
    stderr = b"" if process.stderr is None else process.stderr.read()
    if process.wait() != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise SeedPackageError(
            f"cannot reproduce exact runtime source archive: {detail[:500]}"
        )
    try:
        with gzip.open(path, mode="rb") as source:
            observed_uncompressed_sha, _ = _stream_sha(source)
    except (OSError, EOFError) as exc:
        raise SeedPackageError("runtime source archive gzip stream is invalid") from exc
    if observed_uncompressed_sha != expected_uncompressed_sha:
        raise SeedPackageError(
            "runtime source archive does not reproduce the exact committed subset"
        )
    files = _source_archive_members(path, binding=binding)
    completed = subprocess.run(
        (
            "git",
            "ls-tree",
            "-r",
            "-z",
            "--name-only",
            revision,
            "--",
            "pipelines",
            "src",
            "requirements.txt",
        ),
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise SeedPackageError("cannot inventory exact runtime source subset")
    expected_files = tuple(
        f"source/{raw.decode('utf-8')}"
        for raw in completed.stdout.split(b"\0")
        if raw
    )
    if tuple(files) != expected_files:
        raise SeedPackageError(
            "runtime source archive inventory diverges from allowlisted exact commit"
        )
    return files


def _file_row(path: str, content: bytes | Path, *, mode: int) -> dict[str, Any]:
    if isinstance(content, Path):
        sha256 = _file_sha(content)
        size = content.stat().st_size
    else:
        sha256 = hashlib.sha256(content).hexdigest()
        size = len(content)
    return {"path": path, "sha256": sha256, "bytes": size, "mode": mode}


def _write_deterministic_package(
    path: Path, members: Mapping[str, tuple[bytes | Path, int]]
) -> None:
    ordered = tuple(sorted(members))
    with path.open("xb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0
        ) as compressed:
            with tarfile.open(
                fileobj=compressed, mode="w", format=tarfile.GNU_FORMAT
            ) as archive:
                for name in ordered:
                    content, mode = members[name]
                    info = tarfile.TarInfo(name)
                    info.mode = mode
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.type = tarfile.REGTYPE
                    info.size = (
                        content.stat().st_size
                        if isinstance(content, Path)
                        else len(content)
                    )
                    if isinstance(content, Path):
                        with content.open("rb") as source:
                            archive.addfile(info, source)
                    else:
                        archive.addfile(info, io.BytesIO(content))
        raw.flush()
        os.fsync(raw.fileno())


def _package_inventory(path: Path) -> dict[str, dict[str, Any]]:
    try:
        archive = tarfile.open(path, mode="r:gz")
    except (OSError, tarfile.TarError) as exc:
        raise SeedPackageError("seed action package is not readable tar/gzip") from exc
    result: dict[str, dict[str, Any]] = {}
    with archive:
        for member in archive.getmembers():
            name = Path(member.name).as_posix()
            if (
                not name
                or Path(name).is_absolute()
                or ".." in Path(name).parts
                or name in result
                or not member.isfile()
                or member.mtime != 0
                or member.uid != 0
                or member.gid != 0
                or member.mode not in {0o644, 0o755}
            ):
                raise SeedPackageError(
                    f"seed action package has a noncanonical member: {name!r}"
                )
            extracted = archive.extractfile(member)
            if extracted is None:
                raise SeedPackageError(f"cannot read package member: {name}")
            sha256, size = _stream_sha(extracted)
            result[name] = {
                "path": name,
                "sha256": sha256,
                "bytes": size,
                "mode": member.mode,
            }
    return result


def _package_member_bytes(path: Path, member_name: str) -> bytes:
    with tarfile.open(path, mode="r:gz") as archive:
        try:
            member = archive.getmember(member_name)
        except KeyError as exc:
            raise SeedPackageError(f"package member is missing: {member_name}") from exc
        extracted = archive.extractfile(member)
        if extracted is None:
            raise SeedPackageError(f"package member is unreadable: {member_name}")
        return extracted.read()


def _batch_name(request: ExecutionRequest) -> str:
    lane = request.lane_id.replace("_", "-")
    result = f"paper-ii-seed-{lane}-{request.digest[:16]}"
    if not _SAFE_CLASSAD_VALUE.fullmatch(result):
        raise SeedPackageError("derived job batch name is not ClassAd-safe")
    return result


def _action_token(request: ExecutionRequest) -> str:
    return f"{request.lane_id}-{request.digest[:20]}".replace("_", "-")


def _render_submit_descriptor(
    *,
    output_relative: str,
    request: ExecutionRequest,
    source: SourceEpochReceipt,
    runtime_source_archive: VerifiedFile,
    runtime_image: RuntimeImageIdentity,
    resources: SeedCHTCResources,
) -> bytes:
    package_rel = (Path(output_relative) / _PACKAGE_NAME).as_posix()
    launcher_rel = (Path(output_relative) / _LAUNCHER_NAME).as_posix()
    log_root = (Path(output_relative) / "logs").as_posix()
    image_rel = runtime_image.path
    source_rel = runtime_source_archive.path
    for value in (package_rel, launcher_rel, log_root, image_rel, source_rel):
        if not _SAFE_TRANSFER_PATH.fullmatch(value):
            raise SeedPackageError("descriptor contains a non-transfer-safe path")
    batch = _batch_name(request)
    return (
        "universe = vanilla\n"
        f"executable = {launcher_rel}\n"
        "should_transfer_files = YES\n"
        "when_to_transfer_output = ON_EXIT_OR_EVICT\n"
        "transfer_executable = True\n"
        "preserve_relative_paths = False\n"
        f"transfer_input_files = {package_rel}, {source_rel}, {image_rel}\n"
        "transfer_output_files = raw_outputs\n"
        f"log = {log_root}/{batch}.$(Cluster).$(Process).log\n"
        f"output = {log_root}/{batch}.$(Cluster).$(Process).out\n"
        f"error = {log_root}/{batch}.$(Cluster).$(Process).err\n"
        "stream_output = False\n"
        "stream_error = False\n"
        "requirements = TARGET.HasSIF\n"
        f"request_cpus = {resources.request_cpus}\n"
        f"request_memory = {resources.request_memory_gib}GB\n"
        f"request_disk = {resources.request_disk_gib}GB\n"
        "leave_in_queue = False\n"
        f'+PaperIILaneId = "{request.lane_id}"\n'
        f'+PaperIIActionId = "{request.action_id}"\n'
        f'+PaperIIExecutionRequestSha256 = "{request.digest}"\n'
        "+PaperIIRemoteRawPreservation = True\n"
        f'+PaperIISourceEpochId = "{source.source_epoch.source_epoch_id}"\n'
        "+PaperIIQiskitCompilationRequested = False\n"
        f"+MaxRuntime = {resources.max_runtime_seconds}\n"
        f'+JobBatchName = "{batch}"\n'
        "queue 1\n"
    ).encode("utf-8")


def _render_worker(
    *,
    request: ExecutionRequest,
    order: HHSeedWorkOrder | HubbardSeedWorkOrder,
    source: SourceEpochReceipt,
    runtime_source_archive: VerifiedFile,
    runtime_source_archive_bytes: int,
    runtime_image: RuntimeImageIdentity,
    packaged_files: Mapping[str, dict[str, Any]],
    binding: _SeedLanePackageBinding,
) -> bytes:
    output_token = _action_token(request)
    artifact = Path(order.seed.artifact_path).as_posix()
    receipt = Path(artifact).with_name("seed_qualification_receipt.json").as_posix()
    staged_order = (Path("_paper_ii_bootstrap") / output_token / "seed_work_order.json").as_posix()
    checks: list[str] = []
    for name in sorted(packaged_files):
        row = packaged_files[name]
        checks.extend(
            [
                f"verify_file {shlex.quote(name)} {row['bytes']} {row['sha256']}",
            ]
        )
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "umask 077\n"
        'PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'SCRATCH_ROOT="$PWD"\n'
        'verify_file() {\n'
        '  local relative="$1" expected_bytes="$2" expected_sha="$3"\n'
        '  local target="$PACKAGE_ROOT/$relative"\n'
        '  [[ -f "$target" ]] || { echo "missing package input: $relative" >&2; exit 2; }\n'
        '  [[ "$(wc -c < "$target" | tr -d " ")" == "$expected_bytes" ]] || { echo "package input byte mismatch: $relative" >&2; exit 2; }\n'
        '  [[ "$(sha256sum "$target" | awk \'{print $1}\')" == "$expected_sha" ]] || { echo "package input SHA-256 mismatch: $relative" >&2; exit 2; }\n'
        '}\n'
        + "\n".join(checks)
        + "\n"
        f'IMAGE="$SCRATCH_ROOT/{Path(runtime_image.path).name}"\n'
        f'[[ -f "$IMAGE" ]] || {{ echo "missing runtime image" >&2; exit 2; }}\n'
        f'[[ "$(wc -c < "$IMAGE" | tr -d " ")" == "{runtime_image.bytes}" ]] || {{ echo "runtime image byte mismatch" >&2; exit 2; }}\n'
        f'[[ "$(sha256sum "$IMAGE" | awk \'{{print $1}}\')" == "{runtime_image.sha256}" ]] || {{ echo "runtime image SHA-256 mismatch" >&2; exit 2; }}\n'
        'if command -v apptainer >/dev/null 2>&1; then APPTAINER_BIN="$(command -v apptainer)";\n'
        'elif command -v singularity >/dev/null 2>&1; then APPTAINER_BIN="$(command -v singularity)";\n'
        'else echo "No apptainer/singularity on this execute node." >&2; exit 127; fi\n'
        f'RUNTIME_SOURCE="$SCRATCH_ROOT/{Path(runtime_source_archive.path).name}"\n'
        f'[[ -f "$RUNTIME_SOURCE" ]] || {{ echo "missing runtime source archive" >&2; exit 2; }}\n'
        f'[[ "$(wc -c < "$RUNTIME_SOURCE" | tr -d " ")" == "{runtime_source_archive_bytes}" ]] || {{ echo "runtime source archive byte mismatch" >&2; exit 2; }}\n'
        f'[[ "$(sha256sum "$RUNTIME_SOURCE" | awk \'{{print $1}}\')" == "{runtime_source_archive.sha256}" ]] || {{ echo "runtime source archive SHA-256 mismatch" >&2; exit 2; }}\n'
        'SOURCE_ROOT="$SCRATCH_ROOT/source"\n'
        f'RAW_ROOT="$SCRATCH_ROOT/raw_outputs/paper_ii_seed/{output_token}"\n'
        'mkdir -p "$RAW_ROOT"\n'
        'tar -xzf "$RUNTIME_SOURCE" -C "$SCRATCH_ROOT"\n'
        '[[ -d "$SOURCE_ROOT" ]] || { echo "runtime source root missing" >&2; exit 2; }\n'
        f'mkdir -p "$SOURCE_ROOT/{Path(source.receipt_path).parent.as_posix()}"\n'
        f'cp "$PACKAGE_ROOT/{_SOURCE_RECEIPT_MEMBER}" "$SOURCE_ROOT/{source.receipt_path}"\n'
        f'mkdir -p "$SOURCE_ROOT/{Path(source.source_tree_manifest.path).parent.as_posix()}"\n'
        f'cp "$PACKAGE_ROOT/{_SOURCE_TREE_MEMBER}" "$SOURCE_ROOT/{source.source_tree_manifest.path}"\n'
        f'mkdir -p "$SOURCE_ROOT/{Path(staged_order).parent.as_posix()}"\n'
        f'cp "$PACKAGE_ROOT/{_WORK_ORDER_MEMBER}" "$SOURCE_ROOT/{staged_order}"\n'
        'set +e\n'
        '"$APPTAINER_BIN" exec --cleanenv --bind "$SOURCE_ROOT:/work" "$IMAGE" \\\n'
        "  bash -lc 'cd /work && PYTHONPATH=. PYTHONUNBUFFERED=1 python3 -m "
        f"{binding.runner_module} run "
        f"--repo-root . --work-order {shlex.quote(staged_order)}' \\\n"
        '  >"$RAW_ROOT/worker_stdout.json" 2>"$RAW_ROOT/worker_stderr.log"\n'
        'worker_status=$?\n'
        'set -e\n'
        'if [[ "$worker_status" -ne 0 ]]; then exit "$worker_status"; fi\n'
        f'[[ -f "$SOURCE_ROOT/{artifact}" ]] || {{ echo "seed artifact missing" >&2; exit 3; }}\n'
        f'[[ -f "$SOURCE_ROOT/{receipt}" ]] || {{ echo "seed receipt missing" >&2; exit 3; }}\n'
        f'cp "$SOURCE_ROOT/{artifact}" "$RAW_ROOT/seed.json"\n'
        f'cp "$SOURCE_ROOT/{receipt}" "$RAW_ROOT/seed_qualification_receipt.json"\n'
        'printf "%s\\n" "raw evidence preserved; no remote cleanup authorized" > "$RAW_ROOT/REMOTE_RAW_PRESERVED"\n'
    ).encode("utf-8")


def _render_launcher(*, package_sha256: str, package_bytes: int) -> bytes:
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "umask 077\n"
        f'PACKAGE="$PWD/{_PACKAGE_NAME}"\n'
        f'[[ -f "$PACKAGE" ]] || {{ echo "missing {_PACKAGE_NAME}" >&2; exit 2; }}\n'
        f'[[ "$(wc -c < "$PACKAGE" | tr -d " ")" == "{package_bytes}" ]] || {{ echo "package byte mismatch" >&2; exit 2; }}\n'
        f'[[ "$(sha256sum "$PACKAGE" | awk \'{{print $1}}\')" == "{package_sha256}" ]] || {{ echo "package SHA-256 mismatch" >&2; exit 2; }}\n'
        'mkdir action_package\n'
        'tar -xzf "$PACKAGE" -C action_package\n'
        f'exec bash action_package/{_WORKER_MEMBER}\n'
    ).encode("utf-8")


def _validate_package_payload(path: Path) -> dict[str, dict[str, Any]]:
    inventory = _package_inventory(path)
    manifest_bytes = _package_member_bytes(path, _INNER_MANIFEST)
    try:
        payload = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SeedPackageError("package payload manifest is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise SeedPackageError("package payload manifest must be an object")
    _verify_seal(payload, "package_payload_manifest_sha256", label="package payload manifest")
    rows = payload.get("files")
    expected = {name: row for name, row in inventory.items() if name != _INNER_MANIFEST}
    if (
        payload.get("schema") != SEED_PACKAGE_PAYLOAD_MANIFEST_V1
        or payload.get("action_count") != 1
        or payload.get("qiskit_compilation_requested") is not False
        or payload.get("remote_raw_evidence_policy") != "preserve"
        or payload.get("production_submission_enabled") is not False
        or rows != [expected[name] for name in sorted(expected)]
    ):
        raise SeedPackageError("package payload manifest does not bind its members")
    return inventory


def _manifest_payload(
    *,
    root: Path,
    request: ExecutionRequest,
    source: SourceEpochReceipt,
    package_path: Path,
    package_relative: str,
    batch_name: str,
) -> dict[str, Any]:
    inventory = _package_inventory(package_path)
    ordered = list(inventory)
    core = {
        "schema": BOOTSTRAP_PACKAGE_MANIFEST_V1,
        "status": "prepared_local_only",
        "lane_id": request.lane_id,
        "action_id": request.action_id,
        "execution_id": request.execution_id,
        "execution_request_sha256": request.digest,
        "execution_request": request.to_json_dict(),
        "source_epoch_receipt": {
            "path": source.receipt_path,
            "sha256": source.receipt_sha256,
        },
        "source_epoch_identity_sha256": source.source_epoch.identity_sha256,
        "package": {
            "path": package_relative,
            "sha256": _file_sha(package_path),
        },
        "package_bytes": package_path.stat().st_size,
        "archive_format": "tar.gz",
        "member_count": len(ordered),
        "ordered_members": ordered,
        "ordered_members_sha256": _canonical_sha(ordered),
        "submit_descriptor": _DESCRIPTOR_NAME,
        "job_batch_name": batch_name,
        "qiskit_compilation": False,
        "production_submission_enabled": False,
    }
    return _seal(core, "bootstrap_package_manifest_sha256")


def _intent_payload(
    *,
    request: ExecutionRequest,
    source: SourceEpochReceipt,
    runtime_source_archive: VerifiedFile,
    runtime_source_archive_bytes: int,
    package: VerifiedFile,
    package_bytes: int,
    package_manifest: VerifiedFile,
    descriptor: VerifiedFile,
    launcher: VerifiedFile,
    runtime_image: RuntimeImageIdentity,
) -> dict[str, Any]:
    core = {
        "schema": SEED_SUBMISSION_INTENT_TEMPLATE_V1,
        "status": "template_not_recorded",
        "lane_id": request.lane_id,
        "action_id": request.action_id,
        "execution_id": request.execution_id,
        "execution_request_sha256": request.digest,
        "source_epoch_identity_sha256": source.source_epoch.identity_sha256,
        "source_epoch_full_archive_sha256": (
            source.source_epoch.source_archive_sha256
        ),
        "runtime_source_archive": {
            **runtime_source_archive.to_json_dict(),
            "bytes": runtime_source_archive_bytes,
            "subset": ["pipelines", "src", "requirements.txt"],
            "archive_prefix": "source/",
        },
        "package": {**package.to_json_dict(), "bytes": package_bytes},
        "package_manifest": package_manifest.to_json_dict(),
        "submit_descriptor": descriptor.to_json_dict(),
        "launcher": launcher.to_json_dict(),
        "runtime_image": {
            "path": runtime_image.path,
            "sha256": runtime_image.sha256,
            "bytes": runtime_image.bytes,
        },
        "one_action_per_submission": True,
        "intent_first": dict(BOOTSTRAP_INTENT_FIRST_REQUIREMENTS),
        "submission_intent_recorded": False,
        "scheduler_contact_permitted": False,
        "scheduler_submission_authorized": False,
        "production_submission_enabled": False,
        "remote_raw_evidence_policy": "preserve",
        "materialization_instructions": {
            "write_a_new_receipt_before_any_scheduler_contact": True,
            "set_recorded_at_utc": True,
            "preserve_this_template_unchanged": True,
            "first_reconciliation_outcome": "indeterminate",
            "do_not_submit_until_later_verified_absent": True,
        },
    }
    return _seal(core, "intent_template_sha256")


def _assessment_from_mapping(value: Any) -> FetchStorageAssessment:
    if not isinstance(value, Mapping):
        raise SeedPackageError("storage assessment is missing")
    try:
        result = FetchStorageAssessment(
            allowed=bool(value["allowed"]),
            free_bytes=int(value["free_bytes"]),
            free_floor_bytes=int(value["free_floor_bytes"]),
            archive_bytes=int(value["archive_bytes"]),
            expected_extracted_bytes=int(value["expected_extracted_bytes"]),
            estimated_peak_bytes=int(value["estimated_peak_bytes"]),
            headroom_after_peak_bytes=int(value["headroom_after_peak_bytes"]),
            reason=str(value["reason"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SeedPackageError("storage assessment is invalid") from exc
    reproduced = assess_fetch_storage(
        free_bytes=result.free_bytes,
        archive_bytes=result.archive_bytes,
        expected_extracted_bytes=result.expected_extracted_bytes,
        free_floor_bytes=result.free_floor_bytes,
    )
    if result != reproduced:
        raise SeedPackageError("storage assessment does not reproduce")
    return result


@dataclass(frozen=True)
class PreparedSeedActionPackage:
    """Validated local package; never a scheduler authorization."""

    status: str
    action_count: int
    request: VerifiedFile
    request_sha256: str
    work_order: VerifiedFile
    work_order_sha256: str
    source_epoch: SourceEpochReceipt
    source_archive: VerifiedFile
    runtime_image: RuntimeImageIdentity
    resources: SeedCHTCResources
    package: VerifiedFile
    package_bytes: int
    package_manifest: VerifiedFile
    submit_descriptor: VerifiedFile
    launcher: VerifiedFile
    intent_template: VerifiedFile
    preflight_receipt: VerifiedFile
    storage_assessment: FetchStorageAssessment
    remote_raw_evidence_policy: str
    qiskit_compilation_requested: bool
    production_submission_enabled: bool
    scheduler_submission_authorized: bool
    condor_dry_run_executed: bool

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        repository_root: str | Path,
    ) -> "PreparedSeedActionPackage":
        root = Path(repository_root).resolve(strict=True)
        source = _repository_path(
            root, path, label="seed package preflight", must_exist=True
        )
        payload = _read_object(source, label="seed package preflight")
        _verify_seal(
            payload,
            "seed_package_preflight_sha256",
            label="seed package preflight",
        )
        if (
            payload.get("schema") != SEED_PACKAGE_PREFLIGHT_V1
            or payload.get("status") != "passed_local_only_dry_run_pending"
            or payload.get("action_count") != 1
            or payload.get("ordinary_queue_count") != 1
            or payload.get("remote_raw_evidence_policy") != "preserve"
            or payload.get("qiskit_compilation_requested") is not False
            or payload.get("production_submission_enabled") is not False
            or payload.get("scheduler_contacted") is not False
            or payload.get("scheduler_submission_authorized") is not False
            or payload.get("condor_dry_run_executed") is not False
            or payload.get("intent_first")
            != dict(BOOTSTRAP_INTENT_FIRST_REQUIREMENTS)
        ):
            raise SeedPackageError("seed package preflight is not inert/local-only")

        request_file = _verified_file(payload.get("request"), label="request")
        work_file = _verified_file(payload.get("work_order"), label="work order")
        source_archive = _verified_file(
            payload.get("runtime_source_archive"), label="runtime source archive"
        )
        package_file = _verified_file(payload.get("package"), label="package")
        manifest_file = _verified_file(
            payload.get("package_manifest"), label="package manifest"
        )
        descriptor_file = _verified_file(
            payload.get("submit_descriptor"), label="submit descriptor"
        )
        launcher_file = _verified_file(payload.get("launcher"), label="launcher")
        intent_file = _verified_file(
            payload.get("intent_template"), label="intent template"
        )
        image_identity_file = _verified_file(
            payload.get("runtime_image_identity"), label="runtime image identity"
        )
        for file in (
            request_file,
            work_file,
            source_archive,
            package_file,
            manifest_file,
            descriptor_file,
            launcher_file,
            intent_file,
            image_identity_file,
        ):
            file.verify(root)
        request = _load_request(root / request_file.path)
        binding = _seed_lane_binding(request.lane_id)
        order, order_source = binding.load_work_order(
            root / work_file.path, repository_root=root
        )
        source_receipt = SourceEpochReceipt.load(
            root / str(payload.get("source_epoch_receipt_path") or ""),
            repository_root=root,
        )
        runtime_image = load_runtime_image_identity(
            root / image_identity_file.path, repository_root=root
        )
        resources = SeedCHTCResources.from_mapping(payload.get("resources"))
        if (
            order.request != request
            or order_source != source_receipt
            or payload.get("execution_request_sha256") != request.digest
            or payload.get("work_order_sha256") != order.work_order_sha256
            or payload.get("source_epoch_receipt_sha256")
            != source_receipt.receipt_sha256
            or payload.get("source_epoch_identity_sha256")
            != source_receipt.source_epoch.identity_sha256
            or payload.get("source_epoch_full_archive_sha256")
            != source_receipt.source_epoch.source_archive_sha256
            or payload.get("runtime_image")
            != {
                "path": runtime_image.path,
                "sha256": runtime_image.sha256,
                "bytes": runtime_image.bytes,
            }
            or payload.get("runtime_source_archive_bytes")
            != (root / source_archive.path).stat().st_size
            or payload.get("runtime_source_subset")
            != ["pipelines", "src", "requirements.txt"]
            or payload.get("runtime_source_archive_prefix") != "source/"
        ):
            raise SeedPackageError("seed package provenance bindings diverged")
        _validate_source_archive(
            root / source_archive.path,
            receipt=source_receipt,
            root=root,
            binding=binding,
        )
        inventory = _validate_package_payload(root / package_file.path)
        outer = BootstrapPackageManifest.load(
            root / manifest_file.path, repository_root=root
        )
        try:
            package_bytes = int(payload.get("package_bytes"))
        except (TypeError, ValueError) as exc:
            raise SeedPackageError("package byte count is invalid") from exc
        if (
            outer.request != request
            or outer.package != package_file
            or outer.source_epoch_receipt
            != VerifiedFile(
                source_receipt.receipt_path, source_receipt.receipt_sha256
            )
            or outer.package_bytes != package_bytes
            or package_bytes != (root / package_file.path).stat().st_size
            or inventory[_DESCRIPTOR_NAME]["sha256"] != descriptor_file.sha256
            or _package_member_bytes(root / package_file.path, _DESCRIPTOR_NAME)
            != (root / descriptor_file.path).read_bytes()
        ):
            raise SeedPackageError("outer package manifest/descriptor diverged")
        output_relative = Path(descriptor_file.path).parent.as_posix()
        expected_descriptor = _render_submit_descriptor(
            output_relative=output_relative,
            request=request,
            source=source_receipt,
            runtime_source_archive=source_archive,
            runtime_image=runtime_image,
            resources=resources,
        )
        descriptor_bytes = (root / descriptor_file.path).read_bytes()
        if descriptor_bytes != expected_descriptor:
            raise SeedPackageError("submit descriptor is not canonical")
        validate_submit_lifecycle(descriptor_bytes.decode("utf-8"))
        if descriptor_bytes.decode("utf-8").splitlines().count("queue 1") != 1:
            raise SeedPackageError("submit descriptor is not one ordinary queue-1 job")
        expected_launcher = _render_launcher(
            package_sha256=package_file.sha256,
            package_bytes=package_bytes,
        )
        if (root / launcher_file.path).read_bytes() != expected_launcher:
            raise SeedPackageError("launcher does not bind exact package bytes")

        intent = _read_object(root / intent_file.path, label="intent template")
        _verify_seal(intent, "intent_template_sha256", label="intent template")
        if (
            intent.get("schema") != SEED_SUBMISSION_INTENT_TEMPLATE_V1
            or intent.get("status") != "template_not_recorded"
            or intent.get("submission_intent_recorded") is not False
            or intent.get("scheduler_contact_permitted") is not False
            or intent.get("scheduler_submission_authorized") is not False
            or intent.get("production_submission_enabled") is not False
            or intent.get("one_action_per_submission") is not True
            or intent.get("remote_raw_evidence_policy") != "preserve"
            or intent.get("intent_first")
            != dict(BOOTSTRAP_INTENT_FIRST_REQUIREMENTS)
            or intent.get("package")
            != {**package_file.to_json_dict(), "bytes": package_bytes}
            or intent.get("source_epoch_full_archive_sha256")
            != source_receipt.source_epoch.source_archive_sha256
            or intent.get("runtime_source_archive")
            != {
                **source_archive.to_json_dict(),
                "bytes": (root / source_archive.path).stat().st_size,
                "subset": ["pipelines", "src", "requirements.txt"],
                "archive_prefix": "source/",
            }
            or intent.get("package_manifest") != manifest_file.to_json_dict()
            or intent.get("submit_descriptor") != descriptor_file.to_json_dict()
            or intent.get("launcher") != launcher_file.to_json_dict()
        ):
            raise SeedPackageError("intent template is active, stale, or unbound")
        assessment = _assessment_from_mapping(payload.get("storage_assessment"))
        if (
            not assessment.allowed
            or assessment.free_floor_bytes != DEFAULT_FREE_FLOOR_BYTES
            or assessment.archive_bytes != package_bytes
            or assessment.expected_extracted_bytes != 0
        ):
            raise SeedPackageError("package did not preserve the 10-GiB storage guard")
        expected_dry_run = [
            "condor_submit",
            "-dry-run",
            (Path(output_relative) / "dryrun.ad").as_posix(),
            descriptor_file.path,
        ]
        if (
            payload.get("lifecycle_validator_executed") is not True
            or payload.get("lifecycle_validator_status") != "pass"
            or payload.get("condor_dry_run_argv") != expected_dry_run
        ):
            raise SeedPackageError("local dry-run preparation contract diverged")
        preflight_file = VerifiedFile(
            _relative(root, source, label="seed package preflight"),
            _file_sha(source),
        )
        return cls(
            status="passed_local_only_dry_run_pending",
            action_count=1,
            request=request_file,
            request_sha256=request.digest,
            work_order=work_file,
            work_order_sha256=order.work_order_sha256,
            source_epoch=source_receipt,
            source_archive=source_archive,
            runtime_image=runtime_image,
            resources=resources,
            package=package_file,
            package_bytes=package_bytes,
            package_manifest=manifest_file,
            submit_descriptor=descriptor_file,
            launcher=launcher_file,
            intent_template=intent_file,
            preflight_receipt=preflight_file,
            storage_assessment=assessment,
            remote_raw_evidence_policy="preserve",
            qiskit_compilation_requested=False,
            production_submission_enabled=False,
            scheduler_submission_authorized=False,
            condor_dry_run_executed=False,
        )


def materialize_seed_action_package(
    *,
    repository_root: str | Path,
    output_dir: str | Path,
    request_path: str | Path,
    work_order_path: str | Path,
    source_epoch_receipt: SourceEpochReceipt,
    source_archive_path: str | Path,
    runtime_image_identity: RuntimeImageIdentity,
    resources: SeedCHTCResources,
    free_bytes: int | None = None,
) -> PreparedSeedActionPackage:
    """Materialize one deterministic package without scheduler authority."""

    root = Path(repository_root).resolve(strict=True)
    output = _repository_path(
        root, output_dir, label="seed package output", must_exist=False
    )
    if not _SAFE_COMPONENT.fullmatch(output.name):
        raise SeedPackageError("seed package output name is not CHTC-safe")
    output_relative = _relative(root, output, label="seed package output")
    request_source = _repository_path(
        root, request_path, label="execution request", must_exist=True
    )
    work_source = _repository_path(
        root, work_order_path, label="seed work order", must_exist=True
    )
    archive_source = _repository_path(
        root, source_archive_path, label="source archive", must_exist=True
    )
    request = _load_request(request_source)
    binding = _seed_lane_binding(request.lane_id)
    order, order_source = binding.load_work_order(
        work_source, repository_root=root
    )
    source = _reload_source_epoch(source_epoch_receipt, root=root)
    image = _reload_image(runtime_image_identity, root=root)
    if (
        order.request != request
        or order_source != source
        or request.inputs.get("source_epoch") != source.source_epoch.to_json_dict()
        or bool(request.inputs.get("qiskit_compilation"))
    ):
        raise SeedPackageError("request/work-order/source binding diverged")
    _validate_source_archive(
        archive_source,
        receipt=source,
        root=root,
        binding=binding,
    )
    if output.exists():
        prepared = PreparedSeedActionPackage.load(
            output / _PREFLIGHT_NAME, repository_root=root
        )
        if (
            prepared.request_sha256 != request.digest
            or prepared.work_order_sha256 != order.work_order_sha256
            or prepared.source_epoch != source
            or prepared.source_archive
            != VerifiedFile(
                _relative(root, archive_source, label="source archive"),
                _file_sha(archive_source),
            )
            or prepared.runtime_image != image
            or prepared.resources != resources
        ):
            raise SeedPackageError("existing seed package has another identity")
        return prepared

    free = (
        shutil.disk_usage(root).free
        if free_bytes is None
        else int(free_bytes)
    )
    estimated_package_bytes = (
        request_source.stat().st_size
        + work_source.stat().st_size
        + (root / source.receipt_path).stat().st_size
        + (root / source.source_tree_manifest.path).stat().st_size
        + (root / image.identity_path).stat().st_size
        + 8 * 1024**2
    )
    pre_assessment = assess_fetch_storage(
        free_bytes=free,
        archive_bytes=estimated_package_bytes,
        expected_extracted_bytes=0,
        free_floor_bytes=DEFAULT_FREE_FLOOR_BYTES,
    )
    if not pre_assessment.allowed:
        raise SeedPackageError("10-GiB free-space floor blocks package materialization")

    descriptor_content = _render_submit_descriptor(
        output_relative=output_relative,
        request=request,
        source=source,
        runtime_source_archive=VerifiedFile(
            _relative(root, archive_source, label="runtime source archive"),
            _file_sha(archive_source),
        ),
        runtime_image=image,
        resources=resources,
    )
    validate_submit_lifecycle(descriptor_content.decode("utf-8"))
    if descriptor_content.decode("utf-8").splitlines().count("queue 1") != 1:
        raise SeedPackageError("descriptor must contain exactly one queue 1")

    source_receipt_source = root / source.receipt_path
    source_tree_source = root / source.source_tree_manifest.path
    initial_members: dict[str, tuple[bytes | Path, int]] = {
        _REQUEST_MEMBER: (request_source.read_bytes(), 0o644),
        _WORK_ORDER_MEMBER: (work_source.read_bytes(), 0o644),
        _SOURCE_RECEIPT_MEMBER: (source_receipt_source.read_bytes(), 0o644),
        _SOURCE_TREE_MEMBER: (source_tree_source.read_bytes(), 0o644),
        _IMAGE_IDENTITY_MEMBER: ((root / image.identity_path).read_bytes(), 0o644),
        _DESCRIPTOR_NAME: (descriptor_content, 0o644),
    }
    file_rows = {
        name: _file_row(name, content, mode=mode)
        for name, (content, mode) in initial_members.items()
    }
    worker_content = _render_worker(
        request=request,
        order=order,
        source=source,
        runtime_source_archive=VerifiedFile(
            _relative(root, archive_source, label="runtime source archive"),
            _file_sha(archive_source),
        ),
        runtime_source_archive_bytes=archive_source.stat().st_size,
        runtime_image=image,
        packaged_files=file_rows,
        binding=binding,
    )
    initial_members[_WORKER_MEMBER] = (worker_content, 0o755)
    file_rows[_WORKER_MEMBER] = _file_row(
        _WORKER_MEMBER, worker_content, mode=0o755
    )
    payload_manifest = _seal(
        {
            "schema": SEED_PACKAGE_PAYLOAD_MANIFEST_V1,
            "lane_id": request.lane_id,
            "action_id": request.action_id,
            "execution_request_sha256": request.digest,
            "work_order_sha256": order.work_order_sha256,
            "source_epoch_identity_sha256": source.source_epoch.identity_sha256,
            "action_count": 1,
            "files": [file_rows[name] for name in sorted(file_rows)],
            "qiskit_compilation_requested": False,
            "remote_raw_evidence_policy": "preserve",
            "production_submission_enabled": False,
        },
        "package_payload_manifest_sha256",
    )
    initial_members[_INNER_MANIFEST] = (_pretty_bytes(payload_manifest), 0o644)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        package_temp = temporary / _PACKAGE_NAME
        _write_deterministic_package(package_temp, initial_members)
        _validate_package_payload(package_temp)
        package_sha = _file_sha(package_temp)
        package_bytes = package_temp.stat().st_size
        launcher_content = _render_launcher(
            package_sha256=package_sha,
            package_bytes=package_bytes,
        )
        _write_bytes(temporary / _LAUNCHER_NAME, launcher_content, executable=True)
        _write_bytes(temporary / _DESCRIPTOR_NAME, descriptor_content)
        (temporary / "logs").mkdir()

        manifest_payload = _manifest_payload(
            root=root,
            request=request,
            source=source,
            package_path=package_temp,
            package_relative=(Path(output_relative) / _PACKAGE_NAME).as_posix(),
            batch_name=_batch_name(request),
        )
        _write_bytes(temporary / _MANIFEST_NAME, _pretty_bytes(manifest_payload))

        package_file = VerifiedFile(
            (Path(output_relative) / _PACKAGE_NAME).as_posix(), package_sha
        )
        manifest_file = VerifiedFile(
            (Path(output_relative) / _MANIFEST_NAME).as_posix(),
            _file_sha(temporary / _MANIFEST_NAME),
        )
        descriptor_file = VerifiedFile(
            (Path(output_relative) / _DESCRIPTOR_NAME).as_posix(),
            _file_sha(temporary / _DESCRIPTOR_NAME),
        )
        launcher_file = VerifiedFile(
            (Path(output_relative) / _LAUNCHER_NAME).as_posix(),
            _file_sha(temporary / _LAUNCHER_NAME),
        )
        intent_payload = _intent_payload(
            request=request,
            source=source,
            runtime_source_archive=VerifiedFile(
                _relative(root, archive_source, label="runtime source archive"),
                _file_sha(archive_source),
            ),
            runtime_source_archive_bytes=archive_source.stat().st_size,
            package=package_file,
            package_bytes=package_bytes,
            package_manifest=manifest_file,
            descriptor=descriptor_file,
            launcher=launcher_file,
            runtime_image=image,
        )
        _write_bytes(temporary / _INTENT_NAME, _pretty_bytes(intent_payload))
        intent_file = VerifiedFile(
            (Path(output_relative) / _INTENT_NAME).as_posix(),
            _file_sha(temporary / _INTENT_NAME),
        )
        final_assessment = assess_fetch_storage(
            free_bytes=free,
            archive_bytes=package_bytes,
            expected_extracted_bytes=0,
            free_floor_bytes=DEFAULT_FREE_FLOOR_BYTES,
        )
        if not final_assessment.allowed:
            raise SeedPackageError(
                "10-GiB free-space floor rejects completed package size"
            )
        preflight_core = {
            "schema": SEED_PACKAGE_PREFLIGHT_V1,
            "status": "passed_local_only_dry_run_pending",
            "lane_id": request.lane_id,
            "action_id": request.action_id,
            "execution_id": request.execution_id,
            "execution_request_sha256": request.digest,
            "request": {
                "path": _relative(root, request_source, label="request"),
                "sha256": _file_sha(request_source),
            },
            "work_order": {
                "path": _relative(root, work_source, label="work order"),
                "sha256": _file_sha(work_source),
            },
            "work_order_sha256": order.work_order_sha256,
            "source_epoch_receipt_path": source.receipt_path,
            "source_epoch_receipt_sha256": source.receipt_sha256,
            "source_epoch_identity_sha256": source.source_epoch.identity_sha256,
            "source_epoch_full_archive_sha256": (
                source.source_epoch.source_archive_sha256
            ),
            "runtime_source_archive": {
                "path": _relative(root, archive_source, label="source archive"),
                "sha256": _file_sha(archive_source),
            },
            "runtime_source_archive_bytes": archive_source.stat().st_size,
            "runtime_source_subset": ["pipelines", "src", "requirements.txt"],
            "runtime_source_archive_prefix": "source/",
            "runtime_image_identity": {
                "path": image.identity_path,
                "sha256": image.identity_sha256,
            },
            "runtime_image": {
                "path": image.path,
                "sha256": image.sha256,
                "bytes": image.bytes,
            },
            "resources": resources.to_json_dict(),
            "package": package_file.to_json_dict(),
            "package_bytes": package_bytes,
            "package_manifest": manifest_file.to_json_dict(),
            "submit_descriptor": descriptor_file.to_json_dict(),
            "launcher": launcher_file.to_json_dict(),
            "intent_template": intent_file.to_json_dict(),
            "storage_assessment": asdict(final_assessment),
            "storage_free_bytes_source": (
                "filesystem_probe" if free_bytes is None else "caller_supplied_probe"
            ),
            "action_count": 1,
            "ordinary_queue_count": 1,
            "one_action_per_submission": True,
            "remote_raw_evidence_policy": "preserve",
            "qiskit_compilation_requested": False,
            "intent_first": dict(BOOTSTRAP_INTENT_FIRST_REQUIREMENTS),
            "lifecycle_validator": "chtc/validate_condor_submit_lifecycle.py",
            "lifecycle_validator_executed": True,
            "lifecycle_validator_status": "pass",
            "condor_dry_run_argv": [
                "condor_submit",
                "-dry-run",
                (Path(output_relative) / "dryrun.ad").as_posix(),
                descriptor_file.path,
            ],
            "condor_dry_run_executed": False,
            "scheduler_contacted": False,
            "scheduler_submission_authorized": False,
            "production_submission_enabled": False,
            "required_remote_preflight": [
                "verify_package_manifest_bytes_and_sha256",
                "verify_runtime_image_bytes_and_sha256",
                "verify_source_revision_archive_and_work_order_digests",
                "verify_remote_workdirectory_isolated_and_output_absent",
                "write_submission_intent_before_scheduler_contact",
                "record_first_reconciliation_snapshot_as_indeterminate",
                "require_later_verified_absent_before_submission",
                "run_lifecycle_validator_on_exact_descriptor",
                "run_condor_submit_dry_run_on_exact_descriptor",
                "preserve_remote_raw_evidence",
            ],
        }
        preflight_payload = _seal(
            preflight_core, "seed_package_preflight_sha256"
        )
        _write_bytes(temporary / _PREFLIGHT_NAME, _pretty_bytes(preflight_payload))
        os.rename(temporary, output)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)

    return PreparedSeedActionPackage.load(
        output / _PREFLIGHT_NAME, repository_root=root
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    image = commands.add_parser("runtime-image-identity")
    image.add_argument("--repo-root", default=".")
    image.add_argument("--image", required=True)
    image.add_argument("--output", required=True)
    materialize = commands.add_parser("materialize")
    materialize.add_argument("--repo-root", default=".")
    materialize.add_argument("--output-dir", required=True)
    materialize.add_argument("--request", required=True)
    materialize.add_argument("--work-order", required=True)
    materialize.add_argument("--source-epoch-receipt", required=True)
    materialize.add_argument("--source-archive", required=True)
    materialize.add_argument("--runtime-image-identity", required=True)
    materialize.add_argument("--request-cpus", type=int, required=True)
    materialize.add_argument("--request-memory-gib", type=int, required=True)
    materialize.add_argument("--request-disk-gib", type=int, required=True)
    materialize.add_argument("--max-runtime-seconds", type=int, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--repo-root", default=".")
    validate.add_argument("--preflight-receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.repo_root).resolve(strict=True)
    if args.command == "runtime-image-identity":
        identity = write_runtime_image_identity(
            args.output, image_path=args.image, repository_root=root
        )
        summary = {
            "status": "runtime_image_identity_frozen",
            "identity": identity.identity_path,
            "identity_sha256": identity.identity_sha256,
            "image": identity.path,
            "image_sha256": identity.sha256,
            "image_bytes": identity.bytes,
            "scheduler_submission_authorized": False,
        }
    elif args.command == "materialize":
        source = SourceEpochReceipt.load(
            _repository_path(
                root,
                args.source_epoch_receipt,
                label="source epoch receipt",
                must_exist=True,
            ),
            repository_root=root,
        )
        identity = load_runtime_image_identity(
            args.runtime_image_identity, repository_root=root
        )
        prepared = materialize_seed_action_package(
            repository_root=root,
            output_dir=args.output_dir,
            request_path=args.request,
            work_order_path=args.work_order,
            source_epoch_receipt=source,
            source_archive_path=args.source_archive,
            runtime_image_identity=identity,
            resources=SeedCHTCResources(
                args.request_cpus,
                args.request_memory_gib,
                args.request_disk_gib,
                args.max_runtime_seconds,
            ),
        )
        summary = {
            "status": prepared.status,
            "package": prepared.package.path,
            "package_sha256": prepared.package.sha256,
            "package_bytes": prepared.package_bytes,
            "manifest": prepared.package_manifest.path,
            "submit_descriptor": prepared.submit_descriptor.path,
            "intent_template": prepared.intent_template.path,
            "preflight_receipt": prepared.preflight_receipt.path,
            "condor_dry_run_argv": [
                "condor_submit",
                "-dry-run",
                str(Path(prepared.submit_descriptor.path).with_name("dryrun.ad")),
                prepared.submit_descriptor.path,
            ],
            "scheduler_submission_authorized": False,
        }
    elif args.command == "validate":
        prepared = PreparedSeedActionPackage.load(
            args.preflight_receipt, repository_root=root
        )
        summary = {
            "status": prepared.status,
            "package": prepared.package.path,
            "package_sha256": prepared.package.sha256,
            "execution_request_sha256": prepared.request_sha256,
            "action_count": prepared.action_count,
            "remote_raw_evidence_policy": prepared.remote_raw_evidence_policy,
            "condor_dry_run_executed": prepared.condor_dry_run_executed,
            "scheduler_submission_authorized": False,
        }
    else:  # pragma: no cover
        raise AssertionError(f"unknown command: {args.command}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


__all__ = [
    "PreparedSeedActionPackage",
    "RUNTIME_IMAGE_IDENTITY_V1",
    "RuntimeImageIdentity",
    "SEED_PACKAGE_PAYLOAD_MANIFEST_V1",
    "SEED_PACKAGE_PREFLIGHT_V1",
    "SEED_SUBMISSION_INTENT_TEMPLATE_V1",
    "SeedCHTCResources",
    "SeedPackageError",
    "load_runtime_image_identity",
    "main",
    "materialize_seed_action_package",
    "write_runtime_image_identity",
]


if __name__ == "__main__":
    raise SystemExit(main())
