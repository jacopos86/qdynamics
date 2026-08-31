"""Deterministic inert CHTC package for one scientific AVQDS action."""

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
from typing import Any, Mapping, Sequence

from chtc.validate_condor_submit_lifecycle import validate_submit_lifecycle
from pipelines.artifacts.storage_guard import DEFAULT_FREE_FLOOR_BYTES, assess_fetch_storage
from pipelines.time_dynamics.prospective_avqds_trajectory_worker import (
    AVQDSTrajectoryWorkOrder,
    EXECUTOR_OVERLAY_PATHS,
    VERIFIED_PACKAGED_SOURCE_ENV,
    _assert_exact_blind_runtime_payload,
    load_work_order,
)
from pipelines.time_dynamics.prospective_avqds_runtime_source import (
    TRAJECTORY_PROFILE,
    AVQDSRuntimeSourceError,
    AVQDSRuntimeSourceInventory,
    materialize_avqds_runtime_source_archive,
    smoke_avqds_runtime_work_order,
    validate_avqds_runtime_source_archive,
)
from pipelines.time_dynamics.prospective_bootstrap_provenance import (
    BOOTSTRAP_INTENT_FIRST_REQUIREMENTS,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    VerifiedFile,
)
from pipelines.time_dynamics.prospective_seed_chtc_package import (
    RuntimeImageIdentity,
    load_runtime_image_identity,
)


PACKAGE_PREFLIGHT_SCHEMA_V1 = "paper_ii_avqds_trajectory_package_preflight_v1"
PACKAGE_PAYLOAD_MANIFEST_V1 = "paper_ii_avqds_trajectory_package_payload_manifest_v1"
PACKAGE_MANIFEST_SCHEMA_V1 = "paper_ii_avqds_trajectory_package_manifest_v1"
SUBMISSION_INTENT_TEMPLATE_V1 = (
    "paper_ii_avqds_trajectory_submission_intent_template_v1"
)
_PACKAGE_NAME = "avqds_trajectory_action.tar.gz"
_LAUNCHER_NAME = "launch.sh"
_DESCRIPTOR_NAME = "submit.sub"
_MANIFEST_NAME = "package_manifest.json"
_PREFLIGHT_NAME = "local_preflight_receipt.json"
_INTENT_NAME = "submission_intent_template.json"
_WORKER_MEMBER = "run_avqds_trajectory.sh"
_PAYLOAD_MANIFEST_MEMBER = "input/payload_manifest.json"
_ORDER_MEMBER = "input/work_order.json"
_SAFE_PATH = re.compile(r"[A-Za-z0-9_./-]+")
_SAFE_CLASSAD = re.compile(r"[A-Za-z0-9_.:-]+")
_REQUIRED_BASE_MEMBERS = frozenset(
    {
        "source/pipelines/scaffold/runtime_loader.py",
        "source/pipelines/time_dynamics/paper_ii_runs.py",
        "source/pipelines/time_dynamics/prospective_execution_bindings.py",
        "source/pipelines/time_dynamics/runners/ap_append_from_adapt_artifact.py",
    }
)


class AVQDSTrajectoryPackageError(ValueError):
    """The local scientific AVQDS package is stale, unsafe, or non-inert."""


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
    return (json.dumps(dict(value), indent=2, sort_keys=True) + "\n").encode("utf-8")


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


def _seal(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = dict(value)
    result[key] = _canonical_sha(result)
    return result


def _repository_path(
    root: Path, value: str | Path, *, label: str, must_exist: bool
) -> Path:
    candidate = Path(value)
    target = candidate if candidate.is_absolute() else root / candidate
    target = target.resolve(strict=must_exist)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise AVQDSTrajectoryPackageError(f"{label} is outside repository") from exc
    return target


def _relative(root: Path, path: Path, *, label: str) -> str:
    try:
        value = path.resolve(strict=False).relative_to(root).as_posix()
    except ValueError as exc:
        raise AVQDSTrajectoryPackageError(f"{label} is outside repository") from exc
    if not value or not _SAFE_PATH.fullmatch(value):
        raise AVQDSTrajectoryPackageError(f"{label} is not transfer-safe")
    return value


def _write_bytes(path: Path, content: bytes, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise AVQDSTrajectoryPackageError(
                f"refusing to overwrite package evidence: {path}"
            )
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


@dataclass(frozen=True)
class AVQDSTrajectoryCHTCResources:
    request_cpus: int = 1
    request_memory_gib: int = 16
    request_disk_gib: int = 8
    max_runtime_seconds: int = 43_200

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or int(value) <= 0
            for value in asdict(self).values()
        ):
            raise AVQDSTrajectoryPackageError(
                "CHTC resources must be positive integers"
            )

    def to_json_dict(self) -> dict[str, int]:
        return {key: int(value) for key, value in asdict(self).items()}


_FULL_ARCHIVE_SHA_CACHE: dict[tuple[str, str], str] = {}


def _reproduce_full_archive_sha(root: Path, revision: str) -> str:
    """Digest the whole committed tree, memoized per repository/revision.

    A full commit ID is content-addressed, so its ``git archive`` stream is
    immutable.  The Paper-II base tree is ~1.5 GiB and this digest is
    reproduced several times per package build, so recomputing it dominated
    packaging time without adding any tamper evidence.
    """

    key = (str(Path(root).resolve()), str(revision))
    cached = _FULL_ARCHIVE_SHA_CACHE.get(key)
    if cached is not None:
        return cached
    process = subprocess.Popen(
        ("git", "archive", "--format=tar", revision),
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    digest, _ = _stream_sha(process.stdout)
    stderr = b"" if process.stderr is None else process.stderr.read()
    if process.wait() != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise AVQDSTrajectoryPackageError(
            f"cannot reproduce base source archive: {detail[:500]}"
        )
    _FULL_ARCHIVE_SHA_CACHE[key] = digest
    return digest


def _validate_source_archive(
    path: Path,
    *,
    root: Path,
    order: AVQDSTrajectoryWorkOrder,
    source_receipt: Any,
) -> AVQDSRuntimeSourceInventory:
    epoch = source_receipt.source_epoch
    revision = str(epoch.source_revision or "")
    full_sha = str(epoch.source_archive_sha256 or "")
    if not re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", revision) or not re.fullmatch(
        r"[0-9a-f]{64}", full_sha
    ):
        raise AVQDSTrajectoryPackageError(
            "source epoch lacks a full commit/archive identity"
        )
    if _reproduce_full_archive_sha(root, revision) != full_sha:
        raise AVQDSTrajectoryPackageError("base source archive digest changed")
    try:
        return validate_avqds_runtime_source_archive(
            path,
            repository_root=root,
            profile=TRAJECTORY_PROFILE,
            base_revision=revision,
            base_full_archive_sha256=full_sha,
            overlay_revision=order.executor_overlay.overlay_revision,
            overlay_files=order.executor_overlay.files,
        )
    except AVQDSRuntimeSourceError as exc:
        raise AVQDSTrajectoryPackageError(str(exc)) from exc


def _file_row(name: str, value: bytes, *, mode: int) -> dict[str, Any]:
    return {
        "path": name,
        "sha256": hashlib.sha256(value).hexdigest(),
        "bytes": len(value),
        "mode": mode,
    }


def _write_deterministic_tar(
    path: Path, members: Mapping[str, tuple[bytes, int]]
) -> None:
    with path.open("xb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0
        ) as compressed:
            with tarfile.open(
                fileobj=compressed, mode="w", format=tarfile.GNU_FORMAT
            ) as archive:
                for name in sorted(members):
                    content, mode = members[name]
                    info = tarfile.TarInfo(name)
                    info.size = len(content)
                    info.mode = mode
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    archive.addfile(info, io.BytesIO(content))
        raw.flush()
        os.fsync(raw.fileno())


def _batch_name(order: AVQDSTrajectoryWorkOrder) -> str:
    value = (
        f"paper-ii-avqds-{order.request.lane_id}-{order.request.digest[:16]}"
    ).replace("_", "-")
    if not _SAFE_CLASSAD.fullmatch(value):
        raise AVQDSTrajectoryPackageError("derived batch name is unsafe")
    return value


def _descriptor(
    *,
    output_relative: str,
    order: AVQDSTrajectoryWorkOrder,
    source_epoch_id: str,
    source_archive_relative: str,
    image_relative: str,
    resources: AVQDSTrajectoryCHTCResources,
) -> bytes:
    launcher = (Path(output_relative) / _LAUNCHER_NAME).as_posix()
    package = (Path(output_relative) / _PACKAGE_NAME).as_posix()
    logs = (Path(output_relative) / "logs").as_posix()
    if not all(
        _SAFE_PATH.fullmatch(value)
        for value in (
            launcher,
            package,
            logs,
            source_archive_relative,
            image_relative,
        )
    ):
        raise AVQDSTrajectoryPackageError("descriptor contains unsafe paths")
    if not _SAFE_CLASSAD.fullmatch(source_epoch_id):
        raise AVQDSTrajectoryPackageError("source epoch ID is not ClassAd-safe")
    transfer_names = {
        Path(package).name,
        Path(source_archive_relative).name,
        Path(image_relative).name,
    }
    if len(transfer_names) != 3:
        raise AVQDSTrajectoryPackageError("descriptor transfer basenames collide")
    content = (
        "universe = vanilla\n"
        f"executable = {launcher}\n"
        "should_transfer_files = YES\n"
        "when_to_transfer_output = ON_EXIT_OR_EVICT\n"
        "transfer_executable = True\n"
        "preserve_relative_paths = False\n"
        f"transfer_input_files = {package}, {source_archive_relative}, {image_relative}\n"
        "transfer_output_files = raw_outputs\n"
        f"log = {logs}/{_batch_name(order)}.$(Cluster).$(Process).log\n"
        f"output = {logs}/{_batch_name(order)}.$(Cluster).$(Process).out\n"
        f"error = {logs}/{_batch_name(order)}.$(Cluster).$(Process).err\n"
        "stream_output = False\n"
        "stream_error = False\n"
        "requirements = TARGET.HasSIF\n"
        f"request_cpus = {resources.request_cpus}\n"
        f"request_memory = {resources.request_memory_gib}GB\n"
        f"request_disk = {resources.request_disk_gib}GB\n"
        "leave_in_queue = False\n"
        f'+PaperIILaneId = "{order.request.lane_id}"\n'
        f'+PaperIICellId = "{order.bindings.cell_id}"\n'
        f'+PaperIIActionId = "{order.request.action_id}"\n'
        f'+PaperIIExecutionId = "{order.request.execution_id}"\n'
        f'+PaperIIExecutionRequestSha256 = "{order.request.digest}"\n'
        f'+PaperIIBindingsSha256 = "{order.bindings.digest}"\n'
        f'+PaperIISourceEpochId = "{source_epoch_id}"\n'
        f'+PaperIISourceEpochIdentitySha256 = "{order.bindings.source_epoch_identity_sha256}"\n'
        f'+PaperIIAVQDSExecutorOverlayRevision = "{order.executor_overlay.overlay_revision}"\n'
        f'+PaperIIAVQDSExecutorOverlaySha256 = "{order.executor_overlay.identity_sha256}"\n'
        f'+PaperIICompositeSourceSha256 = "{order.executor_overlay.composite_source_identity_sha256}"\n'
        f'+PaperIIHorizon = "{order.request.inputs["horizon"]}"\n'
        f'+PaperIILadderIndex = {int(order.request.inputs["ladder_index"])}\n'
        "+PaperIIExactReferenceBlind = True\n"
        "+PaperIIRemoteRawPreservation = True\n"
        "+PaperIIProductionSubmissionEnabled = False\n"
        "+PaperIIQiskitCompilation = False\n"
        f"+MaxRuntime = {resources.max_runtime_seconds}\n"
        "queue 1\n"
    )
    return content.encode("utf-8")


def _dependency_files(
    order: AVQDSTrajectoryWorkOrder, *, root: Path
) -> dict[str, bytes]:
    """Collect only source, seed, and optional raw-resume runtime inputs."""

    paths: set[str] = {
        order.source_epoch_receipt.path,
        order.seed_receipt.path,
    }
    seed_payload = json.loads((root / order.seed_receipt.path).read_text())
    paths.add(str(seed_payload["artifact"]["path"]))
    if order.resume_run is not None:
        paths.add(order.resume_run.path)
    result: dict[str, bytes] = {}
    for path in sorted(paths):
        if not _SAFE_PATH.fullmatch(path) or Path(path).is_absolute() or ".." in Path(path).parts:
            raise AVQDSTrajectoryPackageError(
                f"unsafe work-order dependency path: {path!r}"
            )
        target = root / path
        if not target.is_file():
            raise AVQDSTrajectoryPackageError(
                f"work-order dependency is missing: {path}"
            )
        result[f"repository/{path}"] = target.read_bytes()
    _assert_exact_blind_runtime_payload(order.to_json_dict())
    return result


def _validate_exact_blind_json_members(contents: Mapping[str, bytes]) -> None:
    """Prove that no packaged JSON dependency carries readable exact inputs."""

    for name, content in contents.items():
        if not name.endswith(".json"):
            continue
        try:
            payload = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AVQDSTrajectoryPackageError(
                f"packaged JSON dependency is unreadable: {name}"
            ) from exc
        try:
            _assert_exact_blind_runtime_payload(payload, path=name)
        except ValueError as exc:
            raise AVQDSTrajectoryPackageError(
                f"package exposes exact-reference runtime material: {name}"
            ) from exc
        if isinstance(payload, Mapping) and str(payload.get("schema") or "").startswith(
            "paper_ii_exact_reference_"
        ):
            raise AVQDSTrajectoryPackageError(
                f"package contains exact-reference evidence: {name}"
            )


def _worker(
    *,
    order: AVQDSTrajectoryWorkOrder,
    source_archive_name: str,
    source_archive_sha256: str,
    image_name: str,
    image_sha256: str,
    rows: Mapping[str, Mapping[str, Any]],
) -> bytes:
    token = order.request.digest[:20]
    checks = "\n".join(
        f'[[ "$(sha256sum "$PACKAGE_ROOT/{name}" | awk \'{{print $1}}\')" == "{row["sha256"]}" ]] || exit 12'
        for name, row in sorted(rows.items())
    )
    copies = []
    for name in sorted(rows):
        if not name.startswith("repository/"):
            continue
        destination = name[len("repository/") :]
        copies.append(
            f'mkdir -p "$SOURCE_ROOT/{Path(destination).parent.as_posix()}"\n'
            f'cp "$PACKAGE_ROOT/{name}" "$SOURCE_ROOT/{destination}"'
        )
    overlay_checks = []
    overlay_copies = []
    for path in EXECUTOR_OVERLAY_PATHS:
        member = f"executor_overlay/{path}"
        base = order.executor_overlay.files[path]["base_sha256"]
        if base is None:
            overlay_checks.append(f'[[ ! -e "$SOURCE_ROOT/{path}" ]] || exit 21')
        else:
            overlay_checks.append(
                f'[[ "$(sha256sum "$SOURCE_ROOT/{path}" | awk \'{{print $1}}\')" == "{base}" ]] || exit 22'
            )
        overlay_copies.append(
            f'mkdir -p "$SOURCE_ROOT/{Path(path).parent.as_posix()}"\n'
            f'cp "$PACKAGE_ROOT/{member}" "$SOURCE_ROOT/{path}"'
        )
    return (
        "#!/usr/bin/env bash\nset -euo pipefail\numask 077\n"
        'PACKAGE_ROOT="$PWD/action_package"\n'
        f'RUNTIME_SOURCE="$PWD/{source_archive_name}"\n'
        f'IMAGE="$PWD/{image_name}"\n'
        f'[[ "$(sha256sum "$RUNTIME_SOURCE" | awk \'{{print $1}}\')" == "{source_archive_sha256}" ]] || exit 3\n'
        f'[[ "$(sha256sum "$IMAGE" | awk \'{{print $1}}\')" == "{image_sha256}" ]] || exit 4\n'
        + checks
        + "\n"
        'SCRATCH_ROOT="$(mktemp -d "$PWD/avqds-scientific.XXXXXX")"\n'
        'trap \'rm -rf "$SCRATCH_ROOT"\' EXIT\n'
        'tar -xzf "$RUNTIME_SOURCE" -C "$SCRATCH_ROOT"\n'
        'SOURCE_ROOT="$SCRATCH_ROOT/source"\n'
        f'mkdir -p "$SOURCE_ROOT/{Path(_ORDER_MEMBER).parent.as_posix()}"\n'
        f'cp "$PACKAGE_ROOT/{_ORDER_MEMBER}" "$SOURCE_ROOT/{_ORDER_MEMBER}"\n'
        + "\n".join(copies)
        + "\n"
        + "\n".join(overlay_checks)
        + "\n"
        + "\n".join(overlay_copies)
        + "\n"
        f'RAW_ROOT="$PWD/raw_outputs/paper_ii_avqds_trajectory/{token}"\n'
        'mkdir -p "$RAW_ROOT"\n'
        'if command -v apptainer >/dev/null 2>&1; then APPTAINER_BIN="$(command -v apptainer)"; '
        'elif command -v singularity >/dev/null 2>&1; then APPTAINER_BIN="$(command -v singularity)"; else exit 127; fi\n'
        'set +e\n'
        '"$APPTAINER_BIN" exec --cleanenv --bind "$SOURCE_ROOT:/work" '
        '"$IMAGE" bash -lc '
        f"'cd /work && PYTHONPATH=. PYTHONUNBUFFERED=1 {VERIFIED_PACKAGED_SOURCE_ENV}={order.executor_overlay.composite_source_identity_sha256} python3 -m pipelines.time_dynamics.runners.run_prospective_avqds_trajectory run --repo-root . --work-order {shlex.quote(_ORDER_MEMBER)}' "
        '>"$RAW_ROOT/worker_stdout.json" 2>"$RAW_ROOT/worker_stderr.log"\n'
        'status=$?\nset -e\n[[ "$status" -eq 0 ]] || exit "$status"\n'
        f'cp "$SOURCE_ROOT/{order.run_path}" "$RAW_ROOT/run.json"\n'
        f'cp "$SOURCE_ROOT/{order.completion_receipt_path}" "$RAW_ROOT/completion_receipt.json"\n'
        'printf "%s\\n" "raw evidence preserved; no remote cleanup authorized" > "$RAW_ROOT/REMOTE_RAW_PRESERVED"\n'
    ).encode("utf-8")


def _launcher(*, package_sha256: str, package_bytes: int) -> bytes:
    return (
        "#!/usr/bin/env bash\nset -euo pipefail\numask 077\n"
        f'PACKAGE="$PWD/{_PACKAGE_NAME}"\n'
        f'[[ -f "$PACKAGE" ]] && [[ "$(wc -c < "$PACKAGE" | tr -d " ")" == "{package_bytes}" ]] && '
        f'[[ "$(sha256sum "$PACKAGE" | awk \'{{print $1}}\')" == "{package_sha256}" ]] || exit 2\n'
        'mkdir action_package\ntar -xzf "$PACKAGE" -C action_package\n'
        f'exec bash action_package/{_WORKER_MEMBER}\n'
    ).encode("utf-8")


@dataclass(frozen=True)
class PreparedAVQDSTrajectoryPackage:
    preflight: VerifiedFile
    package: VerifiedFile
    package_manifest: VerifiedFile
    descriptor: VerifiedFile
    launcher: VerifiedFile
    intent_template: VerifiedFile
    execution_request_sha256: str
    work_order_sha256: str
    scheduler_contacted: bool = False
    production_submission_enabled: bool = False


def validate_avqds_trajectory_package(
    preflight_receipt_path: str | Path,
    *,
    repository_root: str | Path,
) -> PreparedAVQDSTrajectoryPackage:
    root = Path(repository_root).resolve(strict=True)
    source = _repository_path(
        root,
        preflight_receipt_path,
        label="package preflight receipt",
        must_exist=True,
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise AVQDSTrajectoryPackageError("package preflight is not an object")
    core = dict(payload)
    declared = core.pop("package_preflight_sha256", None)
    if declared != _canonical_sha(core):
        raise AVQDSTrajectoryPackageError("package preflight self digest mismatch")
    if (
        payload.get("schema") != PACKAGE_PREFLIGHT_SCHEMA_V1
        or payload.get("scheduler_contacted") is not False
        or payload.get("scheduler_submission_authorized") is not False
        or payload.get("production_submission_enabled") is not False
        or payload.get("qiskit_compilation_requested") is not False
        or payload.get("remote_raw_evidence_policy") != "preserve"
        or payload.get("exact_reference_blind") is not True
        or payload.get("exact_reference_binding_mode") != "opaque_identity_only"
        or payload.get("exact_reference_runtime_artifacts") != []
        or int(payload.get("ordinary_queue_count", -1)) != 1
    ):
        raise AVQDSTrajectoryPackageError("package preflight is not inert")

    def verified(name: str) -> VerifiedFile:
        row = payload.get(name)
        if not isinstance(row, Mapping):
            raise AVQDSTrajectoryPackageError(f"package preflight lacks {name}")
        result = VerifiedFile(str(row.get("path") or ""), str(row.get("sha256") or ""))
        result.verify(root)
        return result

    package = verified("package")
    manifest_file = verified("package_manifest")
    descriptor = verified("submit_descriptor")
    launcher = verified("launcher")
    intent = verified("intent_template")
    manifest = json.loads((root / manifest_file.path).read_text(encoding="utf-8"))
    intent_payload = json.loads((root / intent.path).read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or not isinstance(intent_payload, Mapping):
        raise AVQDSTrajectoryPackageError("package manifest/intent is malformed")
    _manifest_core = dict(manifest)
    _manifest_declared = _manifest_core.pop("package_manifest_sha256", None)
    if _manifest_declared != _canonical_sha(_manifest_core):
        raise AVQDSTrajectoryPackageError("package manifest self digest mismatch")
    validate_submit_lifecycle((root / descriptor.path).read_text(encoding="utf-8"))
    with tarfile.open(root / package.path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or any(
            Path(name).is_absolute() or ".." in Path(name).parts for name in names
        ):
            raise AVQDSTrajectoryPackageError("package archive has unsafe members")
        contents = {
            member.name: archive.extractfile(member).read()
            for member in members
            if member.isfile()
        }
    _validate_exact_blind_json_members(contents)
    try:
        payload_manifest = json.loads(contents[_PAYLOAD_MANIFEST_MEMBER])
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AVQDSTrajectoryPackageError("package lacks its payload manifest") from exc
    if (
        manifest.get("schema") != PACKAGE_MANIFEST_SCHEMA_V1
        or manifest.get("package") != package.to_json_dict()
        or manifest.get("execution_request_sha256")
        != payload.get("execution_request_sha256")
        or manifest.get("work_order_sha256") != payload.get("work_order_sha256")
        or intent_payload.get("scheduler_contacted") is not False
        or intent_payload.get("scheduler_submission_authorized") is not False
        or intent_payload.get("production_submission_enabled") is not False
        or intent_payload.get("first_reconciliation_outcome") != "indeterminate"
        or manifest.get("exact_reference_blind") is not True
        or manifest.get("exact_reference_binding_mode") != "opaque_identity_only"
        or manifest.get("exact_reference_runtime_artifacts") != []
        or not isinstance(payload_manifest, Mapping)
        or payload_manifest.get("exact_reference_blind") is not True
        or payload_manifest.get("exact_reference_binding_mode")
        != "opaque_identity_only"
        or payload_manifest.get("exact_reference_runtime_artifacts") != []
    ):
        raise AVQDSTrajectoryPackageError("package manifest/intent binding diverged")
    return PreparedAVQDSTrajectoryPackage(
        preflight=VerifiedFile(source.relative_to(root).as_posix(), _file_sha(source)),
        package=package,
        package_manifest=manifest_file,
        descriptor=descriptor,
        launcher=launcher,
        intent_template=intent,
        execution_request_sha256=str(payload.get("execution_request_sha256") or ""),
        work_order_sha256=str(payload.get("work_order_sha256") or ""),
    )


def materialize_avqds_trajectory_package(
    *,
    work_order_path: str | Path,
    runtime_source_archive_path: str | Path,
    runtime_image_identity_path: str | Path,
    output_dir: str | Path,
    repository_root: str | Path,
    resources: AVQDSTrajectoryCHTCResources = AVQDSTrajectoryCHTCResources(),
    free_bytes: int | None = None,
) -> PreparedAVQDSTrajectoryPackage:
    """Build one queue-one package without contacting a scheduler or network."""

    root = Path(repository_root).resolve(strict=True)
    order_path = _repository_path(root, work_order_path, label="work order", must_exist=True)
    source_archive = _repository_path(
        root, runtime_source_archive_path, label="runtime source", must_exist=False
    )
    output = _repository_path(root, output_dir, label="package output", must_exist=False)
    order, source_receipt, seed_receipt = load_work_order(
        order_path, repository_root=root
    )
    image = load_runtime_image_identity(
        runtime_image_identity_path, repository_root=root
    )
    if source_archive.exists() and not source_archive.is_file():
        raise AVQDSTrajectoryPackageError("runtime source is not a regular file")
    if not source_archive.exists():
        epoch = source_receipt.source_epoch
        try:
            materialize_avqds_runtime_source_archive(
                source_archive,
                repository_root=root,
                profile=TRAJECTORY_PROFILE,
                base_revision=str(epoch.source_revision or ""),
                base_full_archive_sha256=str(epoch.source_archive_sha256 or ""),
                overlay_revision=order.executor_overlay.overlay_revision,
                overlay_files=order.executor_overlay.files,
            )
        except AVQDSRuntimeSourceError as exc:
            raise AVQDSTrajectoryPackageError(str(exc)) from exc
    source_inventory = _validate_source_archive(
        source_archive,
        root=root,
        order=order,
        source_receipt=source_receipt,
    )
    smoke_dependencies = [
        order.source_epoch_receipt.path,
        order.seed_receipt.path,
        seed_receipt.artifact.path,
    ]
    if order.resume_run is not None:
        smoke_dependencies.append(order.resume_run.path)
    smoke_avqds_runtime_work_order(
        source_archive,
        repository_root=root,
        profile=TRAJECTORY_PROFILE,
        overlay_revision=order.executor_overlay.overlay_revision,
        overlay_files=order.executor_overlay.files,
        work_order_path=order_path,
        dependency_paths=tuple(smoke_dependencies),
        composite_source_identity_sha256=(
            order.executor_overlay.composite_source_identity_sha256
        ),
    )
    if output.exists():
        raise AVQDSTrajectoryPackageError(
            "package output already exists; validate immutable evidence instead"
        )
    free = shutil.disk_usage(root).free if free_bytes is None else int(free_bytes)
    overlay_bytes = sum((root / path).stat().st_size for path in EXECUTOR_OVERLAY_PATHS)
    estimate = source_archive.stat().st_size + image.bytes + overlay_bytes + 16 * 1024**2
    assessment = assess_fetch_storage(
        free_bytes=free,
        archive_bytes=estimate,
        expected_extracted_bytes=0,
        free_floor_bytes=DEFAULT_FREE_FLOOR_BYTES,
    )
    if not assessment.allowed:
        raise AVQDSTrajectoryPackageError(
            "10-GiB free-space floor blocks package materialization"
        )
    output_relative = _relative(root, output, label="package output")
    source_relative = _relative(root, source_archive, label="runtime source")
    descriptor_bytes = _descriptor(
        output_relative=output_relative,
        order=order,
        source_epoch_id=source_receipt.source_epoch.source_epoch_id,
        source_archive_relative=source_relative,
        image_relative=image.path,
        resources=resources,
    )
    inputs = {
        _ORDER_MEMBER: order_path.read_bytes(),
        **_dependency_files(order, root=root),
    }
    for path in EXECUTOR_OVERLAY_PATHS:
        target = root / path
        expected = order.executor_overlay.files[path]["overlay_sha256"]
        if not target.is_file() or _file_sha(target) != expected:
            raise AVQDSTrajectoryPackageError(
                f"executor overlay changed after work-order creation: {path}"
            )
        inputs[f"executor_overlay/{path}"] = target.read_bytes()
    rows = {name: _file_row(name, value, mode=0o644) for name, value in inputs.items()}
    worker = _worker(
        order=order,
        source_archive_name=source_archive.name,
        source_archive_sha256=_file_sha(source_archive),
        image_name=Path(image.path).name,
        image_sha256=_file_sha(root / image.path),
        rows=rows,
    )
    inputs[_WORKER_MEMBER] = worker
    rows[_WORKER_MEMBER] = _file_row(_WORKER_MEMBER, worker, mode=0o755)
    payload_manifest = _seal(
        {
            "schema": PACKAGE_PAYLOAD_MANIFEST_V1,
            "action_id": order.request.action_id,
            "execution_request_sha256": order.request.digest,
            "work_order_sha256": order.work_order_sha256,
            "files": [rows[name] for name in sorted(rows)],
            "reference_identity_sha256": order.bindings.reference_identity_sha256,
            "exact_reference_blind": True,
            "exact_reference_binding_mode": "opaque_identity_only",
            "exact_reference_runtime_artifacts": [],
            "remote_raw_evidence_policy": "preserve",
            "scientific_qualification": None,
            "qiskit_compilation_requested": False,
            "production_submission_enabled": False,
        },
        "package_payload_manifest_sha256",
    )
    members = {name: (value, 0o644) for name, value in inputs.items()}
    members[_WORKER_MEMBER] = (worker, 0o755)
    members[_PAYLOAD_MANIFEST_MEMBER] = (_pretty_bytes(payload_manifest), 0o644)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        package_path = temporary / _PACKAGE_NAME
        _write_deterministic_tar(package_path, members)
        launcher = _launcher(
            package_sha256=_file_sha(package_path),
            package_bytes=package_path.stat().st_size,
        )
        _write_bytes(temporary / _LAUNCHER_NAME, launcher, executable=True)
        _write_bytes(temporary / _DESCRIPTOR_NAME, descriptor_bytes)
        (temporary / "logs").mkdir()
        intent = _seal(
            {
                "schema": SUBMISSION_INTENT_TEMPLATE_V1,
                "status": "template_not_submitted",
                "action_id": order.request.action_id,
                "execution_request_sha256": order.request.digest,
                "work_order_sha256": order.work_order_sha256,
                "intent_first": dict(BOOTSTRAP_INTENT_FIRST_REQUIREMENTS),
                "first_reconciliation_outcome": "indeterminate",
                "scheduler_contacted": False,
                "scheduler_submission_authorized": False,
                "production_submission_enabled": False,
            },
            "submission_intent_template_sha256",
        )
        _write_bytes(temporary / _INTENT_NAME, _pretty_bytes(intent))
        os.rename(temporary, output)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)

    def verified(name: str) -> VerifiedFile:
        path = output / name
        return VerifiedFile((Path(output_relative) / name).as_posix(), _file_sha(path))

    package_file = verified(_PACKAGE_NAME)
    descriptor_file = verified(_DESCRIPTOR_NAME)
    launcher_file = verified(_LAUNCHER_NAME)
    intent_file = verified(_INTENT_NAME)
    with tarfile.open(output / _PACKAGE_NAME, "r:gz") as archive:
        names = tuple(sorted(member.name for member in archive.getmembers()))
    manifest = _seal(
        {
            "schema": PACKAGE_MANIFEST_SCHEMA_V1,
            "status": "prepared_local_only",
            "lane_id": order.request.lane_id,
            "cell_id": order.bindings.cell_id,
            "action_id": order.request.action_id,
            "execution_id": order.request.execution_id,
            "execution_request_sha256": order.request.digest,
            "work_order_sha256": order.work_order_sha256,
            "package": package_file.to_json_dict(),
            "package_bytes": (output / _PACKAGE_NAME).stat().st_size,
            "member_count": len(names),
            "ordered_members": list(names),
            "ordered_members_sha256": _canonical_sha(names),
            "job_batch_name": _batch_name(order),
            "exact_reference_blind": True,
            "reference_identity_sha256": order.bindings.reference_identity_sha256,
            "exact_reference_binding_mode": "opaque_identity_only",
            "exact_reference_runtime_artifacts": [],
            "remote_raw_evidence_policy": "preserve",
            "qiskit_compilation_requested": False,
            "production_submission_enabled": False,
        },
        "package_manifest_sha256",
    )
    _write_bytes(output / _MANIFEST_NAME, _pretty_bytes(manifest))
    manifest_file = verified(_MANIFEST_NAME)
    preflight = _seal(
        {
            "schema": PACKAGE_PREFLIGHT_SCHEMA_V1,
            "status": "passed_local_only_dry_run_pending",
            "action_id": order.request.action_id,
            "execution_request_sha256": order.request.digest,
            "work_order_sha256": order.work_order_sha256,
            "bindings_sha256": order.bindings.digest,
            "source_epoch_identity_sha256": source_receipt.source_epoch.identity_sha256,
            "executor_overlay_identity_sha256": order.executor_overlay.identity_sha256,
            "composite_source_identity_sha256": (
                order.executor_overlay.composite_source_identity_sha256
            ),
            "seed_identity_sha256": order.seed.identity_sha256,
            "seed_artifact_sha256": seed_receipt.artifact.sha256,
            "reference_identity_sha256": order.bindings.reference_identity_sha256,
            "exact_reference_blind": True,
            "exact_reference_binding_mode": "opaque_identity_only",
            "exact_reference_runtime_artifacts": [],
            "runtime_source_archive": {
                "path": source_relative,
                "sha256": _file_sha(source_archive),
            },
            "runtime_source_inventory": source_inventory.to_json_dict(),
            "runtime_image": {
                "path": image.path,
                "sha256": image.sha256,
                "bytes": image.bytes,
            },
            "resources": resources.to_json_dict(),
            "package": package_file.to_json_dict(),
            "package_manifest": manifest_file.to_json_dict(),
            "submit_descriptor": descriptor_file.to_json_dict(),
            "launcher": launcher_file.to_json_dict(),
            "intent_template": intent_file.to_json_dict(),
            "storage_assessment": asdict(assessment),
            "ordinary_queue_count": 1,
            "condor_dry_run_argv": [
                "condor_submit",
                "-dry-run",
                (Path(output_relative) / "dryrun.ad").as_posix(),
                descriptor_file.path,
            ],
            "condor_dry_run_executed": False,
            "scheduler_contacted": False,
            "scheduler_submission_authorized": False,
            "remote_raw_evidence_policy": "preserve",
            "scientific_qualification": None,
            "scoring_required_before_qualification": True,
            "qiskit_compilation_requested": False,
            "production_submission_enabled": False,
        },
        "package_preflight_sha256",
    )
    _write_bytes(output / _PREFLIGHT_NAME, _pretty_bytes(preflight))
    return validate_avqds_trajectory_package(
        output / _PREFLIGHT_NAME, repository_root=root
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--repo-root", default=".")
    materialize.add_argument("--output-dir", required=True)
    materialize.add_argument("--work-order", required=True)
    materialize.add_argument("--source-archive", required=True)
    materialize.add_argument("--runtime-image-identity", required=True)
    materialize.add_argument("--request-cpus", type=int, required=True)
    materialize.add_argument("--request-memory-gib", type=int, required=True)
    materialize.add_argument("--request-disk-gib", type=int, required=True)
    materialize.add_argument("--max-runtime-seconds", type=int, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--repo-root", default=".")
    validate.add_argument("--preflight", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        prepared = validate_avqds_trajectory_package(
            args.preflight, repository_root=args.repo_root
        )
    else:
        prepared = materialize_avqds_trajectory_package(
            work_order_path=args.work_order,
            runtime_source_archive_path=args.source_archive,
            runtime_image_identity_path=args.runtime_image_identity,
            output_dir=args.output_dir,
            repository_root=args.repo_root,
            resources=AVQDSTrajectoryCHTCResources(
                request_cpus=args.request_cpus,
                request_memory_gib=args.request_memory_gib,
                request_disk_gib=args.request_disk_gib,
                max_runtime_seconds=args.max_runtime_seconds,
            ),
        )
    print(
        json.dumps(
            {
                "status": "validated_local_only" if args.command == "validate" else "prepared_local_only",
                "package": prepared.package.path,
                "package_sha256": prepared.package.sha256,
                "package_manifest": prepared.package_manifest.path,
                "submit_descriptor": prepared.descriptor.path,
                "intent_template": prepared.intent_template.path,
                "preflight_receipt": prepared.preflight.path,
                "execution_request_sha256": prepared.execution_request_sha256,
                "work_order_sha256": prepared.work_order_sha256,
                "scheduler_contacted": False,
                "scheduler_submission_authorized": False,
                "production_submission_enabled": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "AVQDSTrajectoryCHTCResources",
    "AVQDSTrajectoryPackageError",
    "PACKAGE_PREFLIGHT_SCHEMA_V1",
    "PreparedAVQDSTrajectoryPackage",
    "main",
    "materialize_avqds_trajectory_package",
    "validate_avqds_trajectory_package",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
