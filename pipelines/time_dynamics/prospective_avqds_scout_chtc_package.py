"""Deterministic local-only CHTC package for one AVQDS prefix action.

The package is inert: it writes an ordinary ``queue 1`` descriptor, an intent
template, and a dry-run command, but never contacts Condor or SSH.  Runtime
source, seed, image, request, and work-order bytes are all digest bound before
the package is materialized.  Remote raw output is explicitly retained.
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
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from chtc.validate_condor_submit_lifecycle import validate_submit_lifecycle
from pipelines.artifacts.storage_guard import DEFAULT_FREE_FLOOR_BYTES, assess_fetch_storage
from pipelines.time_dynamics.prospective_avqds_scout_worker import (
    AVQDSPrefixWorkOrder,
    EXECUTOR_OVERLAY_PATHS,
    ExecutorOverlayIdentity,
    load_work_order,
)
from pipelines.time_dynamics.prospective_avqds_runtime_source import (
    SCOUT_PROFILE,
    AVQDSRuntimeSourceError,
    AVQDSRuntimeSourceInventory,
    materialize_avqds_runtime_source_archive,
    smoke_avqds_runtime_work_order,
    validate_frozen_avqds_runtime_source_archive,
    validate_avqds_runtime_source_archive,
)
from pipelines.time_dynamics.prospective_bootstrap_provenance import (
    BOOTSTRAP_INTENT_FIRST_REQUIREMENTS,
    BootstrapPackageManifest,
    write_bootstrap_package_manifest,
)
from pipelines.time_dynamics.prospective_execution_bindings import VerifiedFile
from pipelines.time_dynamics.prospective_seed_chtc_package import (
    RuntimeImageIdentity,
    load_runtime_image_identity,
)


PACKAGE_PREFLIGHT_SCHEMA_V1 = "paper_ii_avqds_prefix_package_preflight_v1"
PACKAGE_PAYLOAD_MANIFEST_V1 = "paper_ii_avqds_prefix_package_payload_manifest_v1"
SUBMISSION_INTENT_TEMPLATE_V1 = "paper_ii_avqds_prefix_submission_intent_template_v1"
_PACKAGE_NAME = "avqds_prefix_action.tar.gz"
_LAUNCHER_NAME = "launch.sh"
_DESCRIPTOR_NAME = "submit.sub"
_BOOTSTRAP_MANIFEST_NAME = "package_manifest.json"
_PREFLIGHT_NAME = "local_preflight_receipt.json"
_INTENT_NAME = "submission_intent_template.json"
_WORKER_MEMBER = "run_avqds_prefix.sh"
_PAYLOAD_MANIFEST_MEMBER = "input/payload_manifest.json"
_REQUEST_MEMBER = "input/execution_request.json"
_ORDER_MEMBER = "input/work_order.json"
_SOURCE_RECEIPT_MEMBER = "input/source_epoch_receipt.json"
_SEED_RECEIPT_MEMBER = "input/seed_qualification_receipt.json"
_SEED_ARTIFACT_MEMBER = "input/seed.json"
_IMAGE_IDENTITY_MEMBER = "input/runtime_image_identity.json"
_OVERLAY_MANIFEST_MEMBER = "input/executor_overlay_identity.json"
_SAFE_PATH = re.compile(r"[A-Za-z0-9_./-]+")
_SAFE_CLASSAD = re.compile(r"[A-Za-z0-9_.:-]+")
_REQUIRED_SOURCE_MEMBERS = frozenset(
    {
        "source/pipelines/scaffold/runtime_loader.py",
        "source/pipelines/time_dynamics/paper_ii_runs.py",
        "source/pipelines/time_dynamics/prospective_execution_bindings.py",
        "source/pipelines/time_dynamics/runners/ap_append_from_adapt_artifact.py",
    }
)


def _overlay_member(path: str) -> str:
    return (Path("executor_overlay") / path).as_posix()


class AVQDSPrefixPackageError(ValueError):
    """The local package is stale, unsafe, or would authorize submission."""


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
        raise AVQDSPrefixPackageError(f"{label} is outside repository") from exc
    return target


def _relative(root: Path, path: Path, *, label: str) -> str:
    try:
        value = path.resolve(strict=False).relative_to(root).as_posix()
    except ValueError as exc:
        raise AVQDSPrefixPackageError(f"{label} is outside repository") from exc
    if not value or not _SAFE_PATH.fullmatch(value):
        raise AVQDSPrefixPackageError(f"{label} is not transfer-safe")
    return value


def _write_bytes(path: Path, content: bytes, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise AVQDSPrefixPackageError(f"refusing to overwrite package evidence: {path}")
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
class AVQDSCHTCResources:
    request_cpus: int = 1
    request_memory_gib: int = 16
    request_disk_gib: int = 8
    max_runtime_seconds: int = 43_200

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) or int(value) <= 0 for value in asdict(self).values()):
            raise AVQDSPrefixPackageError("CHTC resources must be positive integers")

    def to_json_dict(self) -> dict[str, int]:
        return {key: int(value) for key, value in asdict(self).items()}


def _operational_generation(value: int) -> int:
    generation = int(value)
    if generation not in {1, 2}:
        raise AVQDSPrefixPackageError(
            "AVQDS-prefix operational generation must be one or two"
        )
    return generation


def avqds_scout_submission_unit_id(
    request: Any, *, operational_generation: int
) -> str:
    generation = _operational_generation(operational_generation)
    base = f"bootstrap-avqds-scout-{request.digest[:16]}"
    value = base if generation == 1 else f"{base}--g{generation}"
    if _SAFE_CLASSAD.fullmatch(value) is None:
        raise AVQDSPrefixPackageError(
            "derived AVQDS-prefix submission unit ID is unsafe"
        )
    return value


_FULL_ARCHIVE_SHA_CACHE: dict[tuple[str, str], str] = {}


def _reproduce_full_git_archive_sha(root: Path, revision: str) -> str:
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
        raise AVQDSPrefixPackageError(f"cannot reproduce source archive: {detail[:500]}")
    _FULL_ARCHIVE_SHA_CACHE[key] = digest
    return digest


def _validate_source_archive(
    path: Path, *, root: Path, order: AVQDSPrefixWorkOrder
) -> AVQDSRuntimeSourceInventory:
    epoch = order.request.inputs["source_epoch"]
    revision = str(epoch.get("source_revision") or "")
    full_sha = str(epoch.get("source_archive_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", revision) or not re.fullmatch(
        r"[0-9a-f]{64}", full_sha
    ):
        raise AVQDSPrefixPackageError("source epoch lacks a full commit/archive identity")
    if _reproduce_full_git_archive_sha(root, revision) != full_sha:
        raise AVQDSPrefixPackageError("source epoch full archive digest changed")

    try:
        return validate_avqds_runtime_source_archive(
            path,
            repository_root=root,
            profile=SCOUT_PROFILE,
            base_revision=revision,
            base_full_archive_sha256=full_sha,
            overlay_revision=order.executor_overlay.overlay_revision,
            overlay_files=order.executor_overlay.files,
        )
    except AVQDSRuntimeSourceError as exc:
        raise AVQDSPrefixPackageError(str(exc)) from exc


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
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.GNU_FORMAT) as archive:
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


def _batch_name_from_request(
    request: Any, *, operational_generation: int = 1
) -> str:
    generation = _operational_generation(operational_generation)
    suffix = "" if generation == 1 else f"-g{generation}"
    value = (
        f"paper-ii-avqds-{request.lane_id}-"
        f"{request.digest[:16]}{suffix}"
    ).replace("_", "-")
    if not _SAFE_CLASSAD.fullmatch(value):
        raise AVQDSPrefixPackageError("derived batch name is unsafe")
    return value


def _batch_name(
    order: AVQDSPrefixWorkOrder, *, operational_generation: int = 1
) -> str:
    return _batch_name_from_request(
        order.request, operational_generation=operational_generation
    )


def _descriptor(
    *,
    output_relative: str,
    order: AVQDSPrefixWorkOrder,
    source_archive: VerifiedFile,
    image: RuntimeImageIdentity,
    resources: AVQDSCHTCResources,
    operational_generation: int,
) -> bytes:
    generation = _operational_generation(operational_generation)
    launcher = (Path(output_relative) / _LAUNCHER_NAME).as_posix()
    package = (Path(output_relative) / _PACKAGE_NAME).as_posix()
    logs = (Path(output_relative) / "logs").as_posix()
    for value in (launcher, package, logs, source_archive.path, image.path):
        if not _SAFE_PATH.fullmatch(value):
            raise AVQDSPrefixPackageError("descriptor contains an unsafe transfer path")
    batch = _batch_name(order, operational_generation=generation)
    recovery_classads = (
        ""
        if generation == 1
        else (
            f'+AcquisitionUnitId = "{avqds_scout_submission_unit_id(order.request, operational_generation=generation)}"\n'
            f'+AcquisitionLaneId = "{order.request.lane_id}"\n'
            f"+PaperIIOperationalGeneration = {generation}\n"
        )
    )
    content = (
        "universe = vanilla\n"
        f"executable = {launcher}\n"
        "should_transfer_files = YES\n"
        "when_to_transfer_output = ON_EXIT_OR_EVICT\n"
        "transfer_executable = True\n"
        "preserve_relative_paths = False\n"
        f"transfer_input_files = {package}, {source_archive.path}, {image.path}\n"
        "transfer_output_files = raw_outputs\n"
        f"log = {logs}/{batch}.$(Cluster).$(Process).log\n"
        f"output = {logs}/{batch}.$(Cluster).$(Process).out\n"
        f"error = {logs}/{batch}.$(Cluster).$(Process).err\n"
        "stream_output = False\n"
        "stream_error = False\n"
        "requirements = TARGET.HasSIF\n"
        f"request_cpus = {resources.request_cpus}\n"
        f"request_memory = {resources.request_memory_gib}GB\n"
        f"request_disk = {resources.request_disk_gib}GB\n"
        "leave_in_queue = False\n"
        f'+PaperIILaneId = "{order.request.lane_id}"\n'
        f'+PaperIIActionId = "{order.request.action_id}"\n'
        f'+PaperIIExecutionRequestSha256 = "{order.request.digest}"\n'
        f'+PaperIISourceEpochId = "{order.request.inputs["source_epoch"]["source_epoch_id"]}"\n'
        f'+PaperIIExecutorOverlayRevision = "{order.executor_overlay.overlay_revision}"\n'
        f'+PaperIIExecutorOverlaySha256 = "{order.executor_overlay.identity_sha256}"\n'
        f'+PaperIICompositeSourceSha256 = "{order.executor_overlay.composite_source_identity_sha256}"\n'
        "+PaperIIRemoteRawPreservation = True\n"
        "+PaperIIExactReferenceBlind = True\n"
        "+PaperIIQiskitCompilationRequested = False\n"
        + recovery_classads
        + f"+MaxRuntime = {resources.max_runtime_seconds}\n"
        f'+JobBatchName = "{batch}"\n'
        "queue 1\n"
    ).encode("utf-8")
    validate_submit_lifecycle(content.decode("utf-8"))
    return content


def _worker(
    *,
    order: AVQDSPrefixWorkOrder,
    seed_receipt: Any,
    source_archive: VerifiedFile,
    source_archive_bytes: int,
    image: RuntimeImageIdentity,
    input_rows: Mapping[str, Mapping[str, Any]],
) -> bytes:
    token = f"{order.request.lane_id}-{order.request.digest[:20]}".replace("_", "-")
    staged_order = (Path("_paper_ii_bootstrap") / token / "avqds_work_order.json").as_posix()
    checks = "\n".join(
        f"verify_file {shlex.quote(name)} {row['bytes']} {row['sha256']}"
        for name, row in sorted(input_rows.items())
    )
    overlay_steps: list[str] = []
    for path in EXECUTOR_OVERLAY_PATHS:
        row = order.executor_overlay.files[path]
        target = f'$SOURCE_ROOT/{path}'
        packaged = f'$PACKAGE_ROOT/{_overlay_member(path)}'
        if row["base_present"]:
            overlay_steps.append(
                f'[[ -f "{target}" ]] && [[ "$(sha256sum "{target}" | awk \'{{print $1}}\')" == "{row["base_sha256"]}" ]] || '
                f'{{ echo "pristine base digest mismatch: {path}" >&2; exit 2; }}'
            )
        else:
            overlay_steps.append(
                f'[[ ! -e "{target}" ]] || {{ echo "unexpected pristine base path: {path}" >&2; exit 2; }}'
            )
        overlay_steps.extend(
            (
                f'[[ "$(sha256sum "{packaged}" | awk \'{{print $1}}\')" == "{row["overlay_sha256"]}" ]] || '
                f'{{ echo "executor overlay digest mismatch: {path}" >&2; exit 2; }}',
                f'mkdir -p "$(dirname "{target}")"',
                f'cp "{packaged}" "{target}"',
                f'[[ "$(sha256sum "{target}" | awk \'{{print $1}}\')" == "{row["overlay_sha256"]}" ]] || '
                f'{{ echo "installed executor overlay mismatch: {path}" >&2; exit 2; }}',
            )
        )
    overlay_script = "\n".join(overlay_steps)
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "umask 077\n"
        'PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'SCRATCH_ROOT="$PWD"\n'
        'verify_file() { local relative="$1" expected_bytes="$2" expected_sha="$3"; local target="$PACKAGE_ROOT/$relative"; '
        '[[ -f "$target" ]] && [[ "$(wc -c < "$target" | tr -d " ")" == "$expected_bytes" ]] && '
        '[[ "$(sha256sum "$target" | awk \'{print $1}\')" == "$expected_sha" ]] || exit 2; }\n'
        + checks
        + "\n"
        f'RUNTIME_SOURCE="$SCRATCH_ROOT/{Path(source_archive.path).name}"\n'
        f'[[ -f "$RUNTIME_SOURCE" ]] && [[ "$(wc -c < "$RUNTIME_SOURCE" | tr -d " ")" == "{source_archive_bytes}" ]] && '
        f'[[ "$(sha256sum "$RUNTIME_SOURCE" | awk \'{{print $1}}\')" == "{source_archive.sha256}" ]] || exit 2\n'
        f'IMAGE="$SCRATCH_ROOT/{Path(image.path).name}"\n'
        f'[[ -f "$IMAGE" ]] && [[ "$(wc -c < "$IMAGE" | tr -d " ")" == "{image.bytes}" ]] && '
        f'[[ "$(sha256sum "$IMAGE" | awk \'{{print $1}}\')" == "{image.sha256}" ]] || exit 2\n'
        'if command -v apptainer >/dev/null 2>&1; then APPTAINER_BIN="$(command -v apptainer)"; '
        'elif command -v singularity >/dev/null 2>&1; then APPTAINER_BIN="$(command -v singularity)"; else exit 127; fi\n'
        'tar -xzf "$RUNTIME_SOURCE" -C "$SCRATCH_ROOT"\n'
        'SOURCE_ROOT="$SCRATCH_ROOT/source"\n'
        f'RAW_ROOT="$SCRATCH_ROOT/raw_outputs/paper_ii_avqds/{token}"\n'
        'mkdir -p "$RAW_ROOT"\n'
        f'mkdir -p "$SOURCE_ROOT/{Path(order.source_epoch_receipt.path).parent.as_posix()}"\n'
        f'cp "$PACKAGE_ROOT/{_SOURCE_RECEIPT_MEMBER}" "$SOURCE_ROOT/{order.source_epoch_receipt.path}"\n'
        f'mkdir -p "$SOURCE_ROOT/{Path(order.seed_receipt.path).parent.as_posix()}"\n'
        f'cp "$PACKAGE_ROOT/{_SEED_RECEIPT_MEMBER}" "$SOURCE_ROOT/{order.seed_receipt.path}"\n'
        f'mkdir -p "$SOURCE_ROOT/{Path(seed_receipt.artifact.path).parent.as_posix()}"\n'
        f'cp "$PACKAGE_ROOT/{_SEED_ARTIFACT_MEMBER}" "$SOURCE_ROOT/{seed_receipt.artifact.path}"\n'
        f'mkdir -p "$SOURCE_ROOT/{Path(staged_order).parent.as_posix()}"\n'
        f'cp "$PACKAGE_ROOT/{_ORDER_MEMBER}" "$SOURCE_ROOT/{staged_order}"\n'
        + overlay_script
        + "\n"
        f'cp "$PACKAGE_ROOT/{_OVERLAY_MANIFEST_MEMBER}" "$RAW_ROOT/executor_overlay_identity.json"\n'
        'printf "%s\\n" "pristine base source epoch verified before executor overlay" > "$RAW_ROOT/BASE_SOURCE_VERIFIED_BEFORE_OVERLAY"\n'
        'set +e\n'
        '"$APPTAINER_BIN" exec --cleanenv --bind "$SOURCE_ROOT:/work" "$IMAGE" bash -lc '
        f"'cd /work && PYTHONPATH=. PYTHONUNBUFFERED=1 PAPER_II_VERIFIED_PACKAGED_SOURCE_SHA256={order.executor_overlay.composite_source_identity_sha256} python3 -m pipelines.time_dynamics.runners.run_prospective_avqds_scout run --repo-root . --work-order {shlex.quote(staged_order)}' "
        '>"$RAW_ROOT/worker_stdout.json" 2>"$RAW_ROOT/worker_stderr.log"\n'
        'status=$?\nset -e\n'
        f'if [[ -f "$SOURCE_ROOT/{order.run_part_path}" ]]; then cp "$SOURCE_ROOT/{order.run_part_path}" "$RAW_ROOT/run.json.part"; fi\n'
        'printf "%s\\n" "raw evidence preserved; no remote cleanup authorized" > "$RAW_ROOT/REMOTE_RAW_PRESERVED"\n'
        '[[ "$status" -eq 0 ]] || exit "$status"\n'
        f'cp "$SOURCE_ROOT/{order.run_path}" "$RAW_ROOT/run.json"\n'
        f'cp "$SOURCE_ROOT/{order.completion_receipt_path}" "$RAW_ROOT/completion_receipt.json"\n'
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
class PreparedAVQDSPrefixPackage:
    preflight: VerifiedFile
    package: VerifiedFile
    package_manifest: VerifiedFile
    descriptor: VerifiedFile
    launcher: VerifiedFile
    intent_template: VerifiedFile
    runtime_source_archive: VerifiedFile
    runtime_source_archive_bytes: int
    runtime_source_inventory: AVQDSRuntimeSourceInventory
    execution_request_sha256: str
    work_order_sha256: str
    resources: AVQDSCHTCResources
    operational_generation: int
    scheduler_contacted: bool = False
    production_submission_enabled: bool = False


_IMMUTABLE_RUNTIME_VALIDATION_CACHE: dict[
    tuple[str, str, str, str, str], AVQDSRuntimeSourceInventory
] = {}


def _verify_frozen_packaged_overlay(
    package_path: Path, overlay: ExecutorOverlayIdentity
) -> None:
    """Match packaged overlay bytes to the immutable declared Git revision."""

    try:
        with tarfile.open(package_path, mode="r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                raise AVQDSPrefixPackageError(
                    "frozen AVQDS package repeats an archive member"
                )
            by_name = {member.name: member for member in members}
            required = {
                _OVERLAY_MANIFEST_MEMBER,
                *(_overlay_member(path) for path in EXECUTOR_OVERLAY_PATHS),
            }
            if not required.issubset(by_name):
                raise AVQDSPrefixPackageError(
                    "frozen AVQDS package lacks its executor overlay"
                )
            contents: dict[str, bytes] = {}
            for name in required:
                member = by_name[name]
                pure = PurePosixPath(name)
                if pure.is_absolute() or ".." in pure.parts or not member.isfile():
                    raise AVQDSPrefixPackageError(
                        "frozen AVQDS package overlay member is unsafe"
                    )
                source = archive.extractfile(member)
                if source is None:
                    raise AVQDSPrefixPackageError(
                        "frozen AVQDS package overlay member is unreadable"
                    )
                content = source.read()
                if len(content) != member.size:
                    raise AVQDSPrefixPackageError(
                        "frozen AVQDS package overlay member is truncated"
                    )
                contents[name] = content
    except (OSError, tarfile.TarError) as exc:
        if isinstance(exc, AVQDSPrefixPackageError):
            raise
        raise AVQDSPrefixPackageError(
            "frozen AVQDS package archive is invalid"
        ) from exc
    try:
        packaged_overlay = ExecutorOverlayIdentity.from_mapping(
            json.loads(contents[_OVERLAY_MANIFEST_MEMBER].decode("utf-8"))
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AVQDSPrefixPackageError(
            "frozen AVQDS package overlay identity is invalid"
        ) from exc
    if packaged_overlay != overlay:
        raise AVQDSPrefixPackageError(
            "frozen AVQDS package overlay identity diverged"
        )
    for path in EXECUTOR_OVERLAY_PATHS:
        if (
            hashlib.sha256(contents[_overlay_member(path)]).hexdigest()
            != overlay.files[path]["overlay_sha256"]
        ):
            raise AVQDSPrefixPackageError(
                f"frozen AVQDS package overlay digest diverged: {path}"
            )


def _validate_avqds_prefix_package(
    preflight_receipt_path: str | Path,
    *,
    repository_root: str | Path,
    frozen_submitted: bool,
) -> PreparedAVQDSPrefixPackage:
    """Reload and digest-check one inert package without contacting Condor."""

    root = Path(repository_root).resolve(strict=True)
    source = _repository_path(
        root,
        preflight_receipt_path,
        label="package preflight receipt",
        must_exist=True,
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise AVQDSPrefixPackageError("package preflight receipt is not an object")
    core = dict(payload)
    declared = core.pop("package_preflight_sha256", None)
    if declared != _canonical_sha(core):
        raise AVQDSPrefixPackageError("package preflight self digest mismatch")
    if (
        payload.get("schema") != PACKAGE_PREFLIGHT_SCHEMA_V1
        or payload.get("scheduler_contacted") is not False
        or payload.get("scheduler_submission_authorized") is not False
        or payload.get("production_submission_enabled") is not False
        or payload.get("qiskit_compilation_requested") is not False
        or payload.get("remote_raw_evidence_policy") != "preserve"
        or int(payload.get("ordinary_queue_count", -1)) != 1
    ):
        raise AVQDSPrefixPackageError("package preflight is not inert")
    generation = _operational_generation(
        int(payload.get("operational_generation", 1))
    )
    resource_row = payload.get("resources")
    if not isinstance(resource_row, Mapping):
        raise AVQDSPrefixPackageError("package preflight lacks resources")
    try:
        resources = AVQDSCHTCResources(**dict(resource_row))
    except (TypeError, ValueError) as exc:
        raise AVQDSPrefixPackageError(
            "package preflight resources are invalid"
        ) from exc

    def verified(name: str) -> VerifiedFile:
        row = payload.get(name)
        if not isinstance(row, Mapping):
            raise AVQDSPrefixPackageError(f"package preflight lacks {name}")
        result = VerifiedFile(
            str(row.get("path") or ""), str(row.get("sha256") or "")
        )
        result.verify(root)
        return result

    package = verified("package")
    package_manifest = verified("package_manifest")
    descriptor = verified("submit_descriptor")
    launcher = verified("launcher")
    intent = verified("intent_template")
    runtime_row = payload.get("runtime_source_archive")
    inventory_row = payload.get("runtime_source_inventory")
    if (
        not isinstance(runtime_row, Mapping)
        or set(runtime_row) != {"path", "sha256", "bytes"}
        or not isinstance(inventory_row, Mapping)
    ):
        raise AVQDSPrefixPackageError(
            "package preflight runtime source binding is incomplete"
        )
    runtime_source = verified("runtime_source_archive")
    runtime_bytes_raw = runtime_row.get("bytes")
    if (
        isinstance(runtime_bytes_raw, bool)
        or not isinstance(runtime_bytes_raw, int)
        or runtime_bytes_raw <= 0
        or (root / runtime_source.path).stat().st_size != runtime_bytes_raw
    ):
        raise AVQDSPrefixPackageError(
            "package preflight runtime source byte identity diverged"
        )
    try:
        overlay = ExecutorOverlayIdentity.from_mapping(payload.get("executor_overlay"))
        cache_key = (
            str(root),
            str(declared),
            package.sha256,
            runtime_source.sha256,
            overlay.identity_sha256,
        )
        if frozen_submitted:
            if not (root / ".git").exists():
                raise AVQDSPrefixPackageError(
                    "frozen AVQDS package requires its Git revision store"
                )
            overlay.verify_committed_files(root)
            _verify_frozen_packaged_overlay(root / package.path, overlay)
            runtime_inventory = _IMMUTABLE_RUNTIME_VALIDATION_CACHE.get(cache_key)
            if runtime_inventory is None:
                runtime_inventory = validate_frozen_avqds_runtime_source_archive(
                    root / runtime_source.path,
                    repository_root=root,
                    profile=SCOUT_PROFILE,
                    base_revision=str(inventory_row.get("base_revision") or ""),
                    base_full_archive_sha256=str(
                        inventory_row.get("base_full_archive_sha256") or ""
                    ),
                    overlay_revision=overlay.overlay_revision,
                    overlay_files=overlay.files,
                    run_import_smoke=False,
                )
                _IMMUTABLE_RUNTIME_VALIDATION_CACHE[cache_key] = runtime_inventory
        else:
            overlay.verify_files(root, require_git_revision=(root / ".git").exists())
            runtime_inventory = _IMMUTABLE_RUNTIME_VALIDATION_CACHE.get(cache_key)
            if runtime_inventory is None:
                runtime_inventory = validate_avqds_runtime_source_archive(
                    root / runtime_source.path,
                    repository_root=root,
                    profile=SCOUT_PROFILE,
                    base_revision=str(inventory_row.get("base_revision") or ""),
                    base_full_archive_sha256=str(
                        inventory_row.get("base_full_archive_sha256") or ""
                    ),
                    overlay_revision=overlay.overlay_revision,
                    overlay_files=overlay.files,
                    run_import_smoke=False,
                )
                _IMMUTABLE_RUNTIME_VALIDATION_CACHE[cache_key] = runtime_inventory
    except (AVQDSRuntimeSourceError, ValueError) as exc:
        raise AVQDSPrefixPackageError(
            "package preflight runtime source archive is invalid"
        ) from exc
    if (
        dict(inventory_row) != runtime_inventory.to_json_dict()
        or runtime_inventory.profile != SCOUT_PROFILE
        or runtime_inventory.overlay_revision != overlay.overlay_revision
        or runtime_inventory.base_revision != overlay.base_source_revision
        or runtime_inventory.archive_sha256 != runtime_source.sha256
        or runtime_inventory.archive_bytes != runtime_bytes_raw
        or payload.get("executor_overlay_identity_sha256")
        != overlay.identity_sha256
        or payload.get("executor_overlay_revision") != overlay.overlay_revision
        or payload.get("executor_overlay_files_sha256") != overlay.files_sha256
        or payload.get("composite_source_identity_sha256")
        != overlay.composite_source_identity_sha256
        or payload.get("source_epoch_identity_sha256")
        != overlay.base_source_epoch_identity_sha256
    ):
        raise AVQDSPrefixPackageError(
            "package preflight runtime source inventory diverged"
        )
    manifest = BootstrapPackageManifest.load(
        root / package_manifest.path,
        repository_root=root,
    )
    descriptor_text = (root / descriptor.path).read_text(encoding="utf-8")
    validate_submit_lifecycle(descriptor_text)
    intent_payload = json.loads((root / intent.path).read_text(encoding="utf-8"))
    if (
        manifest.package != package
        or manifest.request.digest
        != str(payload.get("execution_request_sha256") or "")
        or intent_payload.get("scheduler_contacted") is not False
        or intent_payload.get("scheduler_submission_authorized") is not False
        or intent_payload.get("production_submission_enabled") is not False
        or intent_payload.get("first_reconciliation_outcome") != "indeterminate"
        or int(intent_payload.get("operational_generation", 1)) != generation
        or manifest.job_batch_name
        != _batch_name_from_request(
            manifest.request, operational_generation=generation
        )
    ):
        raise AVQDSPrefixPackageError("package manifest/intent binding diverged")
    if generation == 2:
        unit_id = avqds_scout_submission_unit_id(
            manifest.request, operational_generation=generation
        )
        if (
            f'+AcquisitionUnitId = "{unit_id}"' not in descriptor_text
            or f'+AcquisitionLaneId = "{manifest.request.lane_id}"'
            not in descriptor_text
            or f"+PaperIIOperationalGeneration = {generation}"
            not in descriptor_text
        ):
            raise AVQDSPrefixPackageError(
                "generation-two descriptor lacks its operational identity"
            )
    relative = source.relative_to(root).as_posix()
    preflight = VerifiedFile(relative, _file_sha(source))
    return PreparedAVQDSPrefixPackage(
        preflight=preflight,
        package=package,
        package_manifest=package_manifest,
        descriptor=descriptor,
        launcher=launcher,
        intent_template=intent,
        runtime_source_archive=runtime_source,
        runtime_source_archive_bytes=runtime_bytes_raw,
        runtime_source_inventory=runtime_inventory,
        execution_request_sha256=manifest.request.digest,
        work_order_sha256=str(payload.get("work_order_sha256") or ""),
        resources=resources,
        operational_generation=generation,
    )


def validate_avqds_prefix_package(
    preflight_receipt_path: str | Path,
    *,
    repository_root: str | Path,
) -> PreparedAVQDSPrefixPackage:
    """Validate a package against the current worktree before activation."""

    return _validate_avqds_prefix_package(
        preflight_receipt_path,
        repository_root=repository_root,
        frozen_submitted=False,
    )


def validate_frozen_submitted_avqds_prefix_package(
    preflight_receipt_path: str | Path,
    *,
    repository_root: str | Path,
) -> PreparedAVQDSPrefixPackage:
    """Authenticate an already-submitted package at its frozen Git revisions."""

    return _validate_avqds_prefix_package(
        preflight_receipt_path,
        repository_root=repository_root,
        frozen_submitted=True,
    )


def materialize_avqds_prefix_package(
    *,
    work_order_path: str | Path,
    runtime_source_archive_path: str | Path,
    runtime_image_identity_path: str | Path,
    output_dir: str | Path,
    repository_root: str | Path,
    resources: AVQDSCHTCResources = AVQDSCHTCResources(),
    free_bytes: int | None = None,
    operational_generation: int = 1,
) -> PreparedAVQDSPrefixPackage:
    """Build one deterministic package without scheduler or network access."""

    root = Path(repository_root).resolve(strict=True)
    generation = _operational_generation(operational_generation)
    order_path = _repository_path(root, work_order_path, label="work order", must_exist=True)
    source_archive = _repository_path(
        root, runtime_source_archive_path, label="runtime source archive", must_exist=False
    )
    output = _repository_path(root, output_dir, label="package output", must_exist=False)
    order, source_receipt, seed_receipt = load_work_order(
        order_path, repository_root=root
    )
    image = load_runtime_image_identity(
        runtime_image_identity_path, repository_root=root
    )
    if source_archive.exists() and not source_archive.is_file():
        raise AVQDSPrefixPackageError("runtime source archive is not a regular file")
    if not source_archive.exists():
        epoch = order.request.inputs["source_epoch"]
        try:
            materialize_avqds_runtime_source_archive(
                source_archive,
                repository_root=root,
                profile=SCOUT_PROFILE,
                base_revision=str(epoch["source_revision"]),
                base_full_archive_sha256=str(epoch["source_archive_sha256"]),
                overlay_revision=order.executor_overlay.overlay_revision,
                overlay_files=order.executor_overlay.files,
            )
        except AVQDSRuntimeSourceError as exc:
            raise AVQDSPrefixPackageError(str(exc)) from exc
    source_inventory = _validate_source_archive(source_archive, root=root, order=order)
    smoke_avqds_runtime_work_order(
        source_archive,
        repository_root=root,
        profile=SCOUT_PROFILE,
        overlay_revision=order.executor_overlay.overlay_revision,
        overlay_files=order.executor_overlay.files,
        work_order_path=order_path,
        dependency_paths=(
            order.source_epoch_receipt.path,
            order.seed_receipt.path,
            seed_receipt.artifact.path,
        ),
        composite_source_identity_sha256=(
            order.executor_overlay.composite_source_identity_sha256
        ),
    )
    source_file = VerifiedFile(
        _relative(root, source_archive, label="source archive"), _file_sha(source_archive)
    )
    output_relative = _relative(root, output, label="package output")
    if output.exists():
        raise AVQDSPrefixPackageError("package output already exists; immutable replay requires its loader")
    free = shutil.disk_usage(root).free if free_bytes is None else int(free_bytes)
    overlay_bytes = sum((root / path).stat().st_size for path in EXECUTOR_OVERLAY_PATHS)
    estimate = (
        order_path.stat().st_size
        + source_archive.stat().st_size
        + image.bytes
        + overlay_bytes
        + 8 * 1024**2
    )
    assessment = assess_fetch_storage(
        free_bytes=free,
        archive_bytes=estimate,
        expected_extracted_bytes=0,
        free_floor_bytes=DEFAULT_FREE_FLOOR_BYTES,
    )
    if not assessment.allowed:
        raise AVQDSPrefixPackageError("10-GiB free-space floor blocks package materialization")

    descriptor_bytes = _descriptor(
        output_relative=output_relative,
        order=order,
        source_archive=source_file,
        image=image,
        resources=resources,
        operational_generation=generation,
    )
    inputs = {
        _REQUEST_MEMBER: _pretty_bytes(order.request.to_json_dict()),
        _ORDER_MEMBER: order_path.read_bytes(),
        _SOURCE_RECEIPT_MEMBER: (root / source_receipt.receipt_path).read_bytes(),
        _SEED_RECEIPT_MEMBER: (root / seed_receipt.receipt_path).read_bytes(),
        _SEED_ARTIFACT_MEMBER: (root / seed_receipt.artifact.path).read_bytes(),
        _IMAGE_IDENTITY_MEMBER: (root / image.identity_path).read_bytes(),
        _OVERLAY_MANIFEST_MEMBER: _pretty_bytes(order.executor_overlay.to_json_dict()),
        _DESCRIPTOR_NAME: descriptor_bytes,
    }
    for path in EXECUTOR_OVERLAY_PATHS:
        overlay_source = root / path
        expected_sha = order.executor_overlay.files[path]["overlay_sha256"]
        if not overlay_source.is_file() or _file_sha(overlay_source) != expected_sha:
            raise AVQDSPrefixPackageError(
                f"AVQDS executor overlay changed after work-order creation: {path}"
            )
        inputs[_overlay_member(path)] = overlay_source.read_bytes()
    rows = {name: _file_row(name, value, mode=0o644) for name, value in inputs.items()}
    worker_bytes = _worker(
        order=order,
        seed_receipt=seed_receipt,
        source_archive=source_file,
        source_archive_bytes=source_archive.stat().st_size,
        image=image,
        input_rows=rows,
    )
    inputs[_WORKER_MEMBER] = worker_bytes
    rows[_WORKER_MEMBER] = _file_row(_WORKER_MEMBER, worker_bytes, mode=0o755)
    payload_manifest = _seal(
        {
            "schema": PACKAGE_PAYLOAD_MANIFEST_V1,
            "action_id": order.request.action_id,
            "execution_request_sha256": order.request.digest,
            "work_order_sha256": order.work_order_sha256,
            "operational_generation": generation,
            "base_source_epoch_identity_sha256": (
                order.executor_overlay.base_source_epoch_identity_sha256
            ),
            "executor_overlay_revision": order.executor_overlay.overlay_revision,
            "executor_overlay_identity_sha256": (
                order.executor_overlay.identity_sha256
            ),
            "executor_overlay_files_sha256": order.executor_overlay.files_sha256,
            "composite_source_identity_sha256": (
                order.executor_overlay.composite_source_identity_sha256
            ),
            "files": [rows[name] for name in sorted(rows)],
            "remote_raw_evidence_policy": "preserve",
            "scientific_qualification": None,
            "qiskit_compilation_requested": False,
            "production_submission_enabled": False,
        },
        "package_payload_manifest_sha256",
    )
    members = {name: (value, 0o644) for name, value in inputs.items()}
    members[_WORKER_MEMBER] = (worker_bytes, 0o755)
    members[_PAYLOAD_MANIFEST_MEMBER] = (_pretty_bytes(payload_manifest), 0o644)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        assert temporary is not None
        package_path = temporary / _PACKAGE_NAME
        _write_deterministic_tar(package_path, members)
        launcher_bytes = _launcher(
            package_sha256=_file_sha(package_path),
            package_bytes=package_path.stat().st_size,
        )
        _write_bytes(temporary / _LAUNCHER_NAME, launcher_bytes, executable=True)
        _write_bytes(temporary / _DESCRIPTOR_NAME, descriptor_bytes)
        (temporary / "logs").mkdir()
        intent = _seal(
            {
                "schema": SUBMISSION_INTENT_TEMPLATE_V1,
                "status": "template_not_submitted",
                "action_id": order.request.action_id,
                "execution_request_sha256": order.request.digest,
                "work_order_sha256": order.work_order_sha256,
                "operational_generation": generation,
                "base_source_epoch_identity_sha256": (
                    order.executor_overlay.base_source_epoch_identity_sha256
                ),
                "executor_overlay_revision": (
                    order.executor_overlay.overlay_revision
                ),
                "executor_overlay_identity_sha256": (
                    order.executor_overlay.identity_sha256
                ),
                "composite_source_identity_sha256": (
                    order.executor_overlay.composite_source_identity_sha256
                ),
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

    package_manifest = write_bootstrap_package_manifest(
        output / _BOOTSTRAP_MANIFEST_NAME,
        request=order.request,
        source_epoch_receipt=source_receipt,
        package_path=output / _PACKAGE_NAME,
        submit_descriptor=_DESCRIPTOR_NAME,
        job_batch_name=_batch_name(
            order, operational_generation=generation
        ),
        repository_root=root,
    )
    package_file = VerifiedFile(
        (Path(output_relative) / _PACKAGE_NAME).as_posix(), _file_sha(output / _PACKAGE_NAME)
    )
    descriptor_file = VerifiedFile(
        (Path(output_relative) / _DESCRIPTOR_NAME).as_posix(), _file_sha(output / _DESCRIPTOR_NAME)
    )
    launcher_file = VerifiedFile(
        (Path(output_relative) / _LAUNCHER_NAME).as_posix(), _file_sha(output / _LAUNCHER_NAME)
    )
    intent_file = VerifiedFile(
        (Path(output_relative) / _INTENT_NAME).as_posix(), _file_sha(output / _INTENT_NAME)
    )
    manifest_file = VerifiedFile(package_manifest.manifest_path, package_manifest.manifest_sha256)
    preflight_core = {
        "schema": PACKAGE_PREFLIGHT_SCHEMA_V1,
        "status": "passed_local_only_dry_run_pending",
        "action_id": order.request.action_id,
        "execution_request_sha256": order.request.digest,
        "work_order_sha256": order.work_order_sha256,
        "operational_generation": generation,
        "source_epoch_identity_sha256": source_receipt.source_epoch.identity_sha256,
        "executor_overlay": order.executor_overlay.to_json_dict(),
        "executor_overlay_identity_sha256": order.executor_overlay.identity_sha256,
        "executor_overlay_revision": order.executor_overlay.overlay_revision,
        "executor_overlay_files_sha256": order.executor_overlay.files_sha256,
        "composite_source_identity_sha256": (
            order.executor_overlay.composite_source_identity_sha256
        ),
        "source_composition": (
            "verified_pristine_seed_epoch_plus_committed_executor_overlay_v1"
        ),
        "seed_identity_sha256": order.seed.identity_sha256,
        "seed_artifact_sha256": seed_receipt.artifact.sha256,
        "runtime_source_archive": {
            **source_file.to_json_dict(),
            "bytes": source_archive.stat().st_size,
        },
        "runtime_source_inventory": source_inventory.to_json_dict(),
        "runtime_image_identity": {
            "path": image.identity_path,
            "sha256": image.identity_sha256,
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
    }
    preflight_payload = _seal(preflight_core, "package_preflight_sha256")
    _write_bytes(output / _PREFLIGHT_NAME, _pretty_bytes(preflight_payload))
    preflight_file = VerifiedFile(
        (Path(output_relative) / _PREFLIGHT_NAME).as_posix(), _file_sha(output / _PREFLIGHT_NAME)
    )
    return PreparedAVQDSPrefixPackage(
        preflight=preflight_file,
        package=package_file,
        package_manifest=manifest_file,
        descriptor=descriptor_file,
        launcher=launcher_file,
        intent_template=intent_file,
        runtime_source_archive=source_file,
        runtime_source_archive_bytes=source_archive.stat().st_size,
        runtime_source_inventory=source_inventory,
        execution_request_sha256=order.request.digest,
        work_order_sha256=order.work_order_sha256,
        resources=resources,
        operational_generation=generation,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    materialize = commands.add_parser("materialize")
    materialize.add_argument("--repo-root", default=".")
    materialize.add_argument("--work-order", required=True)
    materialize.add_argument("--source-archive", required=True)
    materialize.add_argument("--runtime-image-identity", required=True)
    materialize.add_argument("--output-dir", required=True)
    materialize.add_argument("--request-cpus", type=int, default=1)
    materialize.add_argument("--request-memory-gib", type=int, default=16)
    materialize.add_argument("--request-disk-gib", type=int, default=8)
    materialize.add_argument("--max-runtime-seconds", type=int, default=43_200)
    materialize.add_argument("--operational-generation", type=int, default=1)
    validate = commands.add_parser("validate")
    validate.add_argument("--repo-root", default=".")
    validate.add_argument("--preflight-receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "materialize":
        prepared = materialize_avqds_prefix_package(
            work_order_path=args.work_order,
            runtime_source_archive_path=args.source_archive,
            runtime_image_identity_path=args.runtime_image_identity,
            output_dir=args.output_dir,
            repository_root=args.repo_root,
            resources=AVQDSCHTCResources(
                request_cpus=args.request_cpus,
                request_memory_gib=args.request_memory_gib,
                request_disk_gib=args.request_disk_gib,
                max_runtime_seconds=args.max_runtime_seconds,
            ),
            operational_generation=args.operational_generation,
        )
    else:
        prepared = validate_avqds_prefix_package(
            args.preflight_receipt,
            repository_root=args.repo_root,
        )
    print(
        json.dumps(
            {
                "status": "prepared_local_only",
                "preflight_receipt": prepared.preflight.path,
                "package": prepared.package.path,
                "package_sha256": prepared.package.sha256,
                "execution_request_sha256": (
                    prepared.execution_request_sha256
                ),
                "operational_generation": prepared.operational_generation,
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
    "AVQDSCHTCResources",
    "AVQDSPrefixPackageError",
    "PACKAGE_PREFLIGHT_SCHEMA_V1",
    "PreparedAVQDSPrefixPackage",
    "avqds_scout_submission_unit_id",
    "main",
    "materialize_avqds_prefix_package",
    "validate_avqds_prefix_package",
    "validate_frozen_submitted_avqds_prefix_package",
]


if __name__ == "__main__":
    raise SystemExit(main())
