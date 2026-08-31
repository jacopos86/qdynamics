"""Deterministic local-only CHTC package for one exact-reference action.

This is packaging plumbing, not a submission adapter.  It emits one ordinary
``queue 1`` descriptor plus an inert intent template.  The package binds the
exact source archive, runtime image, planner request, source/seed receipts, and
seed bytes.  Its worker preserves a complete raw output directory and never
removes remote evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import tempfile
from typing import Any, Mapping, Sequence

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
from pipelines.time_dynamics.prospective_exact_reference_worker import (
    ExactReferenceWorkOrder,
    load_work_order,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    ExecutionRequest,
    SourceEpochReceipt,
    VerifiedFile,
)
from pipelines.time_dynamics.prospective_seed_chtc_package import (
    RuntimeImageIdentity,
    SeedCHTCResources,
    _SeedLanePackageBinding,
    _assessment_from_mapping,
    _canonical_sha,
    _file_row,
    _file_sha,
    _package_inventory,
    _package_member_bytes,
    _pretty_bytes,
    _relative,
    _reload_image,
    _reload_source_epoch,
    _repository_path,
    _seal,
    _validate_source_archive,
    _verified_file,
    _verify_seal,
    _write_bytes,
    _write_deterministic_package,
    load_runtime_image_identity,
)


EXACT_REFERENCE_PACKAGE_PAYLOAD_MANIFEST_V1 = (
    "paper_ii_exact_reference_package_payload_manifest_v1"
)
EXACT_REFERENCE_PACKAGE_PREFLIGHT_V1 = (
    "paper_ii_exact_reference_package_local_preflight_v1"
)
EXACT_REFERENCE_SUBMISSION_INTENT_TEMPLATE_V1 = (
    "paper_ii_exact_reference_submission_intent_template_v1"
)

_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9_.-]+")
_SAFE_TRANSFER_PATH = re.compile(r"[A-Za-z0-9_./-]+")
_SAFE_CLASSAD_VALUE = re.compile(r"[A-Za-z0-9_.:-]+")

_PACKAGE_NAME = "exact_reference_action_package.tar.gz"
_MANIFEST_NAME = "package_manifest.json"
_DESCRIPTOR_NAME = "submit.sub"
_LAUNCHER_NAME = "launch.sh"
_INTENT_NAME = "submission_intent_template.json"
_PREFLIGHT_NAME = "local_preflight_receipt.json"
_INNER_MANIFEST = "input/package_payload_manifest.json"
_REQUEST_MEMBER = "input/execution_request.json"
_WORK_ORDER_MEMBER = "input/exact_reference_work_order.json"
_SOURCE_RECEIPT_MEMBER = "input/source_epoch_receipt.json"
_SOURCE_TREE_MEMBER = "input/source_tree_manifest.json"
_SEED_RECEIPT_MEMBER = "input/seed_qualification_receipt.json"
_SEED_ARTIFACT_MEMBER = "input/seed.json"
_IMAGE_IDENTITY_MEMBER = "input/runtime_image_identity.json"
_WORKER_SOURCE_MEMBER = (
    "executor_overlay/pipelines/time_dynamics/prospective_exact_reference_worker.py"
)
_RUNNER_SOURCE_MEMBER = (
    "executor_overlay/pipelines/time_dynamics/runners/"
    "build_prospective_exact_reference.py"
)
_EXECUTION_BINDINGS_SOURCE_RELATIVE = (
    "pipelines/time_dynamics/prospective_execution_bindings.py"
)
_EXECUTION_BINDINGS_SOURCE_MEMBER = (
    f"executor_overlay/{_EXECUTION_BINDINGS_SOURCE_RELATIVE}"
)
_WORKER_MEMBER = "run_exact_reference_action.sh"

_SOURCE_BINDING = _SeedLanePackageBinding(
    lane_id="prospective_exact_reference",
    runner_module="pipelines.time_dynamics.runners.build_prospective_exact_reference",
    required_source_members=frozenset(
        {
            "source/pipelines/time_dynamics/prospective_execution_bindings.py",
        }
    ),
    work_order_loader=load_work_order,
)


class ExactReferencePackageError(ValueError):
    """An exact-reference package is stale, unsafe, or authorizing."""


def _request(path: Path) -> ExecutionRequest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExactReferencePackageError(
            "exact-reference execution request is unreadable"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ExactReferencePackageError(
            "exact-reference execution request must be an object"
        )
    try:
        result = ExecutionRequest.from_mapping(payload)
    except (TypeError, ValueError) as exc:
        raise ExactReferencePackageError(
            "exact-reference execution request is invalid"
        ) from exc
    if (
        result.kind != "build_exact_reference"
        or result.exact_reference_blind
        or result.inputs.get("reporting_only") is not True
        or result.to_json_dict().get("qiskit_compilation_requested") is not False
    ):
        raise ExactReferencePackageError(
            "request is not a reporting-only exact-reference action"
        )
    return result


def _operational_generation(value: int) -> int:
    generation = int(value)
    if generation not in {1, 2}:
        raise ExactReferencePackageError(
            "exact-reference operational generation must be one or two"
        )
    return generation


def exact_reference_submission_unit_id(
    request: ExecutionRequest, *, operational_generation: int
) -> str:
    generation = _operational_generation(operational_generation)
    base = f"bootstrap-exact-reference-{request.digest[:16]}"
    value = base if generation == 1 else f"{base}--g{generation}"
    if _SAFE_COMPONENT.fullmatch(value) is None:
        raise ExactReferencePackageError(
            "derived exact-reference submission unit ID is unsafe"
        )
    return value


def _batch_name(
    request: ExecutionRequest, *, operational_generation: int = 1
) -> str:
    generation = _operational_generation(operational_generation)
    lane = request.lane_id.replace("_", "-")
    suffix = "" if generation == 1 else f"-g{generation}"
    value = f"paper-ii-reference-{lane}-{request.digest[:16]}{suffix}"
    if not _SAFE_CLASSAD_VALUE.fullmatch(value):
        raise ExactReferencePackageError("derived batch name is not ClassAd-safe")
    return value


def _action_token(request: ExecutionRequest) -> str:
    return f"{request.lane_id}-{request.digest[:20]}".replace("_", "-")


def _render_descriptor(
    *,
    output_relative: str,
    request: ExecutionRequest,
    source: SourceEpochReceipt,
    source_archive: VerifiedFile,
    runtime_image: RuntimeImageIdentity,
    resources: SeedCHTCResources,
    operational_generation: int = 1,
) -> bytes:
    generation = _operational_generation(operational_generation)
    package_rel = (Path(output_relative) / _PACKAGE_NAME).as_posix()
    launcher_rel = (Path(output_relative) / _LAUNCHER_NAME).as_posix()
    log_root = (Path(output_relative) / "logs").as_posix()
    for value in (
        package_rel,
        launcher_rel,
        log_root,
        source_archive.path,
        runtime_image.path,
    ):
        if not _SAFE_TRANSFER_PATH.fullmatch(value):
            raise ExactReferencePackageError(
                "descriptor contains a non-transfer-safe path"
            )
    recovery_classads = (
        ""
        if generation == 1
        else (
            f'+AcquisitionUnitId = "{exact_reference_submission_unit_id(request, operational_generation=generation)}"\n'
            f'+AcquisitionLaneId = "{request.lane_id}"\n'
            f"+PaperIIOperationalGeneration = {generation}\n"
        )
    )
    batch_name = _batch_name(request, operational_generation=generation)
    return (
        "universe = vanilla\n"
        f"executable = {launcher_rel}\n"
        "should_transfer_files = YES\n"
        "when_to_transfer_output = ON_EXIT_OR_EVICT\n"
        "transfer_executable = True\n"
        "preserve_relative_paths = False\n"
        f"transfer_input_files = {package_rel}, {source_archive.path}, {runtime_image.path}\n"
        "transfer_output_files = raw_outputs\n"
        f"log = {log_root}/{batch_name}.$(Cluster).$(Process).log\n"
        f"output = {log_root}/{batch_name}.$(Cluster).$(Process).out\n"
        f"error = {log_root}/{batch_name}.$(Cluster).$(Process).err\n"
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
        f'+PaperIIExactReferenceId = "{request.inputs["reference_id"]}"\n'
        "+PaperIIRemoteRawPreservation = True\n"
        f'+PaperIISourceEpochId = "{source.source_epoch.source_epoch_id}"\n'
        "+PaperIIQiskitCompilationRequested = False\n"
        + recovery_classads
        + f"+MaxRuntime = {resources.max_runtime_seconds}\n"
        f'+JobBatchName = "{batch_name}"\n'
        "queue 1\n"
    ).encode("utf-8")


def _render_worker(
    *,
    request: ExecutionRequest,
    order: ExactReferenceWorkOrder,
    source: SourceEpochReceipt,
    source_archive: VerifiedFile,
    source_archive_bytes: int,
    runtime_image: RuntimeImageIdentity,
    packaged_files: Mapping[str, Mapping[str, Any]],
    operational_generation: int,
) -> bytes:
    generation = _operational_generation(operational_generation)
    token = _action_token(request)
    manifest = Path(str(request.inputs["artifact_path"])).as_posix()
    data = Path(manifest).with_name("reference_data.json").as_posix()
    receipt = Path(manifest).with_name("exact_reference_receipt.json").as_posix()
    staged_order = (
        Path("_paper_ii_bootstrap") / token / "exact_reference_work_order.json"
    ).as_posix()
    executor_sources = dict(order.propagation.get("executor_source_files") or {})
    worker_source_relative = (
        "pipelines/time_dynamics/prospective_exact_reference_worker.py"
    )
    runner_source_relative = (
        "pipelines/time_dynamics/runners/build_prospective_exact_reference.py"
    )
    if set(executor_sources) != {worker_source_relative, runner_source_relative}:
        raise ExactReferencePackageError(
            "exact-reference work order has no canonical executor overlay"
        )
    overlay_sources = [worker_source_relative, runner_source_relative]
    binding_rule = ""
    if generation == 2:
        overlay_sources.append(_EXECUTION_BINDINGS_SOURCE_RELATIVE)
        binding_rule = (
            f'  if [[ "$overlay" == {shlex.quote(_EXECUTION_BINDINGS_SOURCE_RELATIVE)} ]]; then '
            f'expected={shlex.quote(str(packaged_files[_EXECUTION_BINDINGS_SOURCE_MEMBER]["sha256"]))}; '
            "replace_source_epoch_binding=1; fi\n"
        )
    overlay_arguments = " ".join(shlex.quote(value) for value in overlay_sources)
    checks = "\n".join(
        f"verify_file {shlex.quote(name)} {row['bytes']} {row['sha256']}"
        for name, row in sorted(packaged_files.items())
    )
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "umask 077\n"
        'PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'SCRATCH_ROOT="$PWD"\n'
        "verify_file() {\n"
        '  local relative="$1" expected_bytes="$2" expected_sha="$3"\n'
        '  local target="$PACKAGE_ROOT/$relative"\n'
        '  [[ -f "$target" ]] || { echo "missing package input: $relative" >&2; exit 2; }\n'
        '  [[ "$(wc -c < "$target" | tr -d " ")" == "$expected_bytes" ]] || { echo "package input byte mismatch: $relative" >&2; exit 2; }\n'
        '  [[ "$(sha256sum "$target" | awk \'{print $1}\')" == "$expected_sha" ]] || { echo "package input SHA-256 mismatch: $relative" >&2; exit 2; }\n'
        "}\n"
        + checks
        + "\n"
        f'IMAGE="$SCRATCH_ROOT/{Path(runtime_image.path).name}"\n'
        f'[[ "$(wc -c < "$IMAGE" | tr -d " ")" == "{runtime_image.bytes}" ]] || {{ echo "runtime image byte mismatch" >&2; exit 2; }}\n'
        f'[[ "$(sha256sum "$IMAGE" | awk \'{{print $1}}\')" == "{runtime_image.sha256}" ]] || {{ echo "runtime image SHA-256 mismatch" >&2; exit 2; }}\n'
        "if command -v apptainer >/dev/null 2>&1; then APPTAINER_BIN=\"$(command -v apptainer)\";\n"
        "elif command -v singularity >/dev/null 2>&1; then APPTAINER_BIN=\"$(command -v singularity)\";\n"
        "else echo \"No apptainer/singularity on this execute node.\" >&2; exit 127; fi\n"
        f'RUNTIME_SOURCE="$SCRATCH_ROOT/{Path(source_archive.path).name}"\n'
        f'[[ "$(wc -c < "$RUNTIME_SOURCE" | tr -d " ")" == "{source_archive_bytes}" ]] || {{ echo "runtime source archive byte mismatch" >&2; exit 2; }}\n'
        f'[[ "$(sha256sum "$RUNTIME_SOURCE" | awk \'{{print $1}}\')" == "{source_archive.sha256}" ]] || {{ echo "runtime source archive SHA-256 mismatch" >&2; exit 2; }}\n'
        'SOURCE_ROOT="$SCRATCH_ROOT/source"\n'
        f'RAW_ROOT="$SCRATCH_ROOT/raw_outputs/paper_ii_exact_reference/{token}"\n'
        'mkdir -p "$RAW_ROOT"\n'
        'tar -xzf "$RUNTIME_SOURCE" -C "$SCRATCH_ROOT"\n'
        '[[ -d "$SOURCE_ROOT" ]] || { echo "runtime source root missing" >&2; exit 2; }\n'
        f"for overlay in {overlay_arguments}; do\n"
        '  target="$SOURCE_ROOT/$overlay"\n'
        '  packaged="$PACKAGE_ROOT/executor_overlay/$overlay"\n'
        '  expected=""\n'
        '  replace_source_epoch_binding=0\n'
        f'  if [[ "$overlay" == {shlex.quote(worker_source_relative)} ]]; then expected={shlex.quote(str(executor_sources[worker_source_relative]))}; fi\n'
        f'  if [[ "$overlay" == {shlex.quote(runner_source_relative)} ]]; then expected={shlex.quote(str(executor_sources[runner_source_relative]))}; fi\n'
        + binding_rule
        + '  [[ -n "$expected" ]] || { echo "unknown executor overlay" >&2; exit 2; }\n'
        '  [[ "$(sha256sum "$packaged" | awk \'{print $1}\')" == "$expected" ]] || { echo "executor overlay package digest mismatch" >&2; exit 2; }\n'
        '  if [[ "$replace_source_epoch_binding" -eq 1 ]] && [[ ! -f "$target" ]]; then echo "source epoch execution binding is missing" >&2; exit 2; fi\n'
        '  if [[ "$replace_source_epoch_binding" -ne 1 ]] && [[ -f "$target" ]] && [[ "$(sha256sum "$target" | awk \'{print $1}\')" != "$expected" ]]; then echo "executor overlay conflicts with source epoch" >&2; exit 2; fi\n'
        '  mkdir -p "$(dirname "$target")"\n'
        '  cp "$packaged" "$target"\n'
        'done\n'
        f'mkdir -p "$SOURCE_ROOT/{Path(source.receipt_path).parent.as_posix()}"\n'
        f'cp "$PACKAGE_ROOT/{_SOURCE_RECEIPT_MEMBER}" "$SOURCE_ROOT/{source.receipt_path}"\n'
        f'mkdir -p "$SOURCE_ROOT/{Path(source.source_tree_manifest.path).parent.as_posix()}"\n'
        f'cp "$PACKAGE_ROOT/{_SOURCE_TREE_MEMBER}" "$SOURCE_ROOT/{source.source_tree_manifest.path}"\n'
        f'mkdir -p "$SOURCE_ROOT/{Path(order.seed_receipt_path).parent.as_posix()}"\n'
        f'cp "$PACKAGE_ROOT/{_SEED_RECEIPT_MEMBER}" "$SOURCE_ROOT/{order.seed_receipt_path}"\n'
        f'mkdir -p "$SOURCE_ROOT/{Path(order.seed_artifact_path).parent.as_posix()}"\n'
        f'cp "$PACKAGE_ROOT/{_SEED_ARTIFACT_MEMBER}" "$SOURCE_ROOT/{order.seed_artifact_path}"\n'
        f'mkdir -p "$SOURCE_ROOT/{Path(staged_order).parent.as_posix()}"\n'
        f'cp "$PACKAGE_ROOT/{_WORK_ORDER_MEMBER}" "$SOURCE_ROOT/{staged_order}"\n'
        "set +e\n"
        '"$APPTAINER_BIN" exec --cleanenv --bind "$SOURCE_ROOT:/work" "$IMAGE" \\\n'
        "  bash -lc 'cd /work && PYTHONPATH=. PYTHONUNBUFFERED=1 python3 -m "
        "pipelines.time_dynamics.runners.build_prospective_exact_reference run "
        f"--repo-root . --work-order {shlex.quote(staged_order)}' \\\n"
        '  >"$RAW_ROOT/worker_stdout.json" 2>"$RAW_ROOT/worker_stderr.log"\n'
        "worker_status=$?\n"
        "set -e\n"
        'if [[ "$worker_status" -ne 0 ]]; then exit "$worker_status"; fi\n'
        f'[[ -f "$SOURCE_ROOT/{manifest}" ]] || {{ echo "exact manifest missing" >&2; exit 3; }}\n'
        f'[[ -f "$SOURCE_ROOT/{data}" ]] || {{ echo "exact data missing" >&2; exit 3; }}\n'
        f'[[ -f "$SOURCE_ROOT/{receipt}" ]] || {{ echo "exact receipt missing" >&2; exit 3; }}\n'
        f'cp "$SOURCE_ROOT/{manifest}" "$RAW_ROOT/manifest.json"\n'
        f'cp "$SOURCE_ROOT/{data}" "$RAW_ROOT/reference_data.json"\n'
        f'cp "$SOURCE_ROOT/{receipt}" "$RAW_ROOT/exact_reference_receipt.json"\n'
        f'cp "$PACKAGE_ROOT/{_REQUEST_MEMBER}" "$RAW_ROOT/execution_request.json"\n'
        f'cp "$PACKAGE_ROOT/{_WORK_ORDER_MEMBER}" "$RAW_ROOT/exact_reference_work_order.json"\n'
        f'cp "$PACKAGE_ROOT/{_SOURCE_RECEIPT_MEMBER}" "$RAW_ROOT/source_epoch_receipt.json"\n'
        f'cp "$PACKAGE_ROOT/{_SEED_RECEIPT_MEMBER}" "$RAW_ROOT/seed_qualification_receipt.json"\n'
        f'cp "$PACKAGE_ROOT/{_SEED_ARTIFACT_MEMBER}" "$RAW_ROOT/seed.json"\n'
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
        "mkdir action_package\n"
        "tar -xzf \"$PACKAGE\" -C action_package\n"
        f"exec bash action_package/{_WORKER_MEMBER}\n"
    ).encode("utf-8")


def _validate_package(path: Path) -> dict[str, dict[str, Any]]:
    inventory = _package_inventory(path)
    try:
        payload = json.loads(
            _package_member_bytes(path, _INNER_MANIFEST).decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExactReferencePackageError(
            "exact-reference package manifest is unreadable"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ExactReferencePackageError(
            "exact-reference package manifest must be an object"
        )
    _verify_seal(
        payload,
        "package_payload_manifest_sha256",
        label="exact-reference package payload manifest",
    )
    expected = {
        name: row for name, row in inventory.items() if name != _INNER_MANIFEST
    }
    try:
        _operational_generation(int(payload.get("operational_generation", 1)))
    except (TypeError, ValueError) as exc:
        raise ExactReferencePackageError(
            "exact-reference package operational generation is invalid"
        ) from exc
    if (
        payload.get("schema") != EXACT_REFERENCE_PACKAGE_PAYLOAD_MANIFEST_V1
        or payload.get("action_count") != 1
        or payload.get("qiskit_compilation_requested") is not False
        or payload.get("remote_raw_evidence_policy") != "preserve"
        or payload.get("production_submission_enabled") is not False
        or payload.get("files") != [expected[name] for name in sorted(expected)]
    ):
        raise ExactReferencePackageError(
            "exact-reference package payload does not bind its members"
        )
    return inventory


def _manifest_payload(
    *,
    request: ExecutionRequest,
    source: SourceEpochReceipt,
    package_path: Path,
    package_relative: str,
    operational_generation: int,
) -> dict[str, Any]:
    generation = _operational_generation(operational_generation)
    inventory = _package_inventory(package_path)
    ordered = list(inventory)
    return _seal(
        {
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
            "job_batch_name": _batch_name(
                request, operational_generation=generation
            ),
            "operational_generation": generation,
            "qiskit_compilation": False,
            "production_submission_enabled": False,
        },
        "bootstrap_package_manifest_sha256",
    )


def _intent_payload(
    *,
    request: ExecutionRequest,
    source: SourceEpochReceipt,
    source_archive: VerifiedFile,
    source_archive_bytes: int,
    runtime_image: RuntimeImageIdentity,
    package: VerifiedFile,
    package_bytes: int,
    manifest: VerifiedFile,
    descriptor: VerifiedFile,
    launcher: VerifiedFile,
    operational_generation: int,
) -> dict[str, Any]:
    generation = _operational_generation(operational_generation)
    return _seal(
        {
            "schema": EXACT_REFERENCE_SUBMISSION_INTENT_TEMPLATE_V1,
            "status": "template_not_recorded",
            "lane_id": request.lane_id,
            "action_id": request.action_id,
            "execution_id": request.execution_id,
            "execution_request_sha256": request.digest,
            "reference_id": request.inputs["reference_id"],
            "source_epoch_identity_sha256": source.source_epoch.identity_sha256,
            "source_epoch_full_archive_sha256": source.source_epoch.source_archive_sha256,
            "runtime_source_archive": {
                **source_archive.to_json_dict(),
                "bytes": source_archive_bytes,
                "subset": ["pipelines", "src", "requirements.txt"],
                "archive_prefix": "source/",
            },
            "runtime_image": {
                "path": runtime_image.path,
                "sha256": runtime_image.sha256,
                "bytes": runtime_image.bytes,
            },
            "package": {**package.to_json_dict(), "bytes": package_bytes},
            "package_manifest": manifest.to_json_dict(),
            "submit_descriptor": descriptor.to_json_dict(),
            "launcher": launcher.to_json_dict(),
            "one_action_per_submission": True,
            "intent_first": dict(BOOTSTRAP_INTENT_FIRST_REQUIREMENTS),
            "submission_intent_recorded": False,
            "scheduler_contact_permitted": False,
            "scheduler_submission_authorized": False,
            "production_submission_enabled": False,
            "remote_raw_evidence_policy": "preserve",
            "operational_generation": generation,
            "materialization_instructions": {
                "write_a_new_receipt_before_any_scheduler_contact": True,
                "set_recorded_at_utc": True,
                "preserve_this_template_unchanged": True,
                "first_reconciliation_outcome": "indeterminate",
                "do_not_submit_until_later_verified_absent": True,
            },
        },
        "intent_template_sha256",
    )


@dataclass(frozen=True)
class PreparedExactReferenceActionPackage:
    """Validated inert package; never a scheduler authorization."""

    status: str
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
    operational_generation: int
    remote_raw_evidence_policy: str = "preserve"
    production_submission_enabled: bool = False
    scheduler_submission_authorized: bool = False
    condor_dry_run_executed: bool = False

    @classmethod
    def load(
        cls, path: str | Path, *, repository_root: str | Path
    ) -> "PreparedExactReferenceActionPackage":
        root = Path(repository_root).resolve(strict=True)
        source = _repository_path(
            root, path, label="exact-reference package preflight", must_exist=True
        )
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExactReferencePackageError(
                "exact-reference package preflight is unreadable"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ExactReferencePackageError(
                "exact-reference package preflight must be an object"
            )
        _verify_seal(
            payload,
            "exact_reference_package_preflight_sha256",
            label="exact-reference package preflight",
        )
        if (
            payload.get("schema") != EXACT_REFERENCE_PACKAGE_PREFLIGHT_V1
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
            raise ExactReferencePackageError(
                "exact-reference package preflight is active or noncanonical"
            )
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
        image_file = _verified_file(
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
            image_file,
        ):
            file.verify(root)
        request = _request(root / request_file.path)
        order, order_source, _seed = load_work_order(
            root / work_file.path, repository_root=root
        )
        source_receipt = SourceEpochReceipt.load(
            root / str(payload.get("source_epoch_receipt_path") or ""),
            repository_root=root,
        )
        image = load_runtime_image_identity(
            root / image_file.path, repository_root=root
        )
        resources = SeedCHTCResources.from_mapping(payload.get("resources"))
        generation = _operational_generation(
            int(payload.get("operational_generation", 1))
        )
        if (
            order.request != request
            or order_source != source_receipt
            or payload.get("execution_request_sha256") != request.digest
            or payload.get("work_order_sha256") != order.work_order_sha256
            or payload.get("source_epoch_receipt_sha256")
            != source_receipt.receipt_sha256
            or payload.get("source_epoch_identity_sha256")
            != source_receipt.source_epoch.identity_sha256
            or payload.get("runtime_source_archive_bytes")
            != (root / source_archive.path).stat().st_size
            or payload.get("runtime_image")
            != {
                "path": image.path,
                "sha256": image.sha256,
                "bytes": image.bytes,
            }
        ):
            raise ExactReferencePackageError(
                "exact-reference preflight provenance diverged"
            )
        _validate_source_archive(
            root / source_archive.path,
            receipt=source_receipt,
            root=root,
            binding=_SOURCE_BINDING,
        )
        inventory = _validate_package(root / package_file.path)
        inner_payload = json.loads(
            _package_member_bytes(
                root / package_file.path, _INNER_MANIFEST
            ).decode("utf-8")
        )
        if (
            int(inner_payload.get("operational_generation", 1)) != generation
            or (
                generation == 2
                and _EXECUTION_BINDINGS_SOURCE_MEMBER not in inventory
            )
        ):
            raise ExactReferencePackageError(
                "successor exact-reference package generation/repair diverged"
            )
        outer = BootstrapPackageManifest.load(
            root / manifest_file.path, repository_root=root
        )
        outer_payload = json.loads(
            (root / manifest_file.path).read_text(encoding="utf-8")
        )
        package_bytes = int(payload.get("package_bytes"))
        if (
            outer.request != request
            or outer.package != package_file
            or int(outer_payload.get("operational_generation", 1)) != generation
            or outer.job_batch_name
            != _batch_name(request, operational_generation=generation)
            or package_bytes != (root / package_file.path).stat().st_size
            or inventory[_DESCRIPTOR_NAME]["sha256"] != descriptor_file.sha256
            or _package_member_bytes(root / package_file.path, _DESCRIPTOR_NAME)
            != (root / descriptor_file.path).read_bytes()
        ):
            raise ExactReferencePackageError(
                "exact-reference package outer bindings diverged"
            )
        output_relative = Path(descriptor_file.path).parent.as_posix()
        expected_descriptor = _render_descriptor(
            output_relative=output_relative,
            request=request,
            source=source_receipt,
            source_archive=source_archive,
            runtime_image=image,
            resources=resources,
            operational_generation=generation,
        )
        if (root / descriptor_file.path).read_bytes() != expected_descriptor:
            raise ExactReferencePackageError(
                "exact-reference submit descriptor is not canonical"
            )
        validate_submit_lifecycle(expected_descriptor.decode("utf-8"))
        if expected_descriptor.decode("utf-8").splitlines().count("queue 1") != 1:
            raise ExactReferencePackageError(
                "exact-reference descriptor is not one queue-1 job"
            )
        if (root / launcher_file.path).read_bytes() != _render_launcher(
            package_sha256=package_file.sha256, package_bytes=package_bytes
        ):
            raise ExactReferencePackageError("exact-reference launcher diverged")
        intent = json.loads((root / intent_file.path).read_text(encoding="utf-8"))
        _verify_seal(intent, "intent_template_sha256", label="intent template")
        if (
            intent.get("schema")
            != EXACT_REFERENCE_SUBMISSION_INTENT_TEMPLATE_V1
            or intent.get("status") != "template_not_recorded"
            or intent.get("submission_intent_recorded") is not False
            or intent.get("scheduler_contact_permitted") is not False
            or intent.get("scheduler_submission_authorized") is not False
            or intent.get("production_submission_enabled") is not False
            or intent.get("remote_raw_evidence_policy") != "preserve"
            or int(intent.get("operational_generation", 1)) != generation
            or intent.get("intent_first")
            != dict(BOOTSTRAP_INTENT_FIRST_REQUIREMENTS)
        ):
            raise ExactReferencePackageError(
                "exact-reference intent template is active or stale"
            )
        assessment = _assessment_from_mapping(payload.get("storage_assessment"))
        if (
            not assessment.allowed
            or assessment.free_floor_bytes != DEFAULT_FREE_FLOOR_BYTES
            or assessment.archive_bytes != package_bytes
            or assessment.expected_extracted_bytes != 0
        ):
            raise ExactReferencePackageError(
                "exact-reference package did not preserve the storage floor"
            )
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
            raise ExactReferencePackageError(
                "exact-reference dry-run preparation diverged"
            )
        return cls(
            status="passed_local_only_dry_run_pending",
            request=request_file,
            request_sha256=request.digest,
            work_order=work_file,
            work_order_sha256=order.work_order_sha256,
            source_epoch=source_receipt,
            source_archive=source_archive,
            runtime_image=image,
            resources=resources,
            package=package_file,
            package_bytes=package_bytes,
            package_manifest=manifest_file,
            submit_descriptor=descriptor_file,
            launcher=launcher_file,
            intent_template=intent_file,
            preflight_receipt=VerifiedFile(
                _relative(root, source, label="exact-reference preflight"),
                _file_sha(source),
            ),
            storage_assessment=assessment,
            operational_generation=generation,
        )


def materialize_exact_reference_action_package(
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
    operational_generation: int = 1,
) -> PreparedExactReferenceActionPackage:
    """Materialize one deterministic exact-reference package, without contact."""

    root = Path(repository_root).resolve(strict=True)
    generation = _operational_generation(operational_generation)
    output = _repository_path(
        root, output_dir, label="exact-reference package output", must_exist=False
    )
    if not _SAFE_COMPONENT.fullmatch(output.name):
        raise ExactReferencePackageError(
            "exact-reference package output name is not CHTC-safe"
        )
    output_relative = _relative(root, output, label="exact-reference package output")
    request_source = _repository_path(
        root, request_path, label="execution request", must_exist=True
    )
    work_source = _repository_path(
        root, work_order_path, label="exact-reference work order", must_exist=True
    )
    archive_source = _repository_path(
        root, source_archive_path, label="runtime source archive", must_exist=True
    )
    request = _request(request_source)
    order, order_source, seed = load_work_order(
        work_source, repository_root=root
    )
    source = _reload_source_epoch(source_epoch_receipt, root=root)
    image = _reload_image(runtime_image_identity, root=root)
    if (
        order.request != request
        or order_source != source
        or request.inputs.get("source_epoch") != source.source_epoch.to_json_dict()
        or request.inputs.get("seed_artifact_sha256") != seed.artifact.sha256
    ):
        raise ExactReferencePackageError(
            "exact-reference request/work-order/source/seed binding diverged"
        )
    try:
        _validate_source_archive(
            archive_source, receipt=source, root=root, binding=_SOURCE_BINDING
        )
    except ValueError as exc:
        raise ExactReferencePackageError(str(exc)) from exc
    if output.exists():
        prepared = PreparedExactReferenceActionPackage.load(
            output / _PREFLIGHT_NAME, repository_root=root
        )
        if (
            prepared.request_sha256 != request.digest
            or prepared.work_order_sha256 != order.work_order_sha256
            or prepared.source_epoch != source
            or prepared.source_archive
            != VerifiedFile(
                _relative(root, archive_source, label="runtime source archive"),
                _file_sha(archive_source),
            )
            or prepared.runtime_image != image
            or prepared.resources != resources
            or prepared.operational_generation != generation
        ):
            raise ExactReferencePackageError(
                "existing exact-reference package has another identity"
            )
        return prepared

    free = shutil.disk_usage(root).free if free_bytes is None else int(free_bytes)
    estimated = (
        request_source.stat().st_size
        + work_source.stat().st_size
        + (root / source.receipt_path).stat().st_size
        + (root / source.source_tree_manifest.path).stat().st_size
        + (root / seed.receipt_path).stat().st_size
        + (root / seed.artifact.path).stat().st_size
        + (root / image.identity_path).stat().st_size
        + 8 * 1024**2
    )
    if not assess_fetch_storage(
        free_bytes=free,
        archive_bytes=estimated,
        expected_extracted_bytes=0,
        free_floor_bytes=DEFAULT_FREE_FLOOR_BYTES,
    ).allowed:
        raise ExactReferencePackageError(
            "10-GiB free-space floor blocks exact-reference package materialization"
        )
    source_archive = VerifiedFile(
        _relative(root, archive_source, label="runtime source archive"),
        _file_sha(archive_source),
    )
    descriptor = _render_descriptor(
        output_relative=output_relative,
        request=request,
        source=source,
        source_archive=source_archive,
        runtime_image=image,
        resources=resources,
        operational_generation=generation,
    )
    validate_submit_lifecycle(descriptor.decode("utf-8"))
    if descriptor.decode("utf-8").splitlines().count("queue 1") != 1:
        raise ExactReferencePackageError(
            "exact-reference descriptor must contain exactly one queue 1"
        )
    members: dict[str, tuple[bytes | Path, int]] = {
        _REQUEST_MEMBER: (request_source.read_bytes(), 0o644),
        _WORK_ORDER_MEMBER: (work_source.read_bytes(), 0o644),
        _SOURCE_RECEIPT_MEMBER: ((root / source.receipt_path).read_bytes(), 0o644),
        _SOURCE_TREE_MEMBER: (
            (root / source.source_tree_manifest.path).read_bytes(),
            0o644,
        ),
        _SEED_RECEIPT_MEMBER: ((root / seed.receipt_path).read_bytes(), 0o644),
        _SEED_ARTIFACT_MEMBER: (root / seed.artifact.path, 0o644),
        _IMAGE_IDENTITY_MEMBER: ((root / image.identity_path).read_bytes(), 0o644),
        _DESCRIPTOR_NAME: (descriptor, 0o644),
    }
    executor_sources = dict(order.propagation.get("executor_source_files") or {})
    worker_relative = "pipelines/time_dynamics/prospective_exact_reference_worker.py"
    runner_relative = (
        "pipelines/time_dynamics/runners/build_prospective_exact_reference.py"
    )
    if set(executor_sources) != {worker_relative, runner_relative}:
        raise ExactReferencePackageError(
            "exact-reference work order executor overlay is incomplete"
        )
    worker_source = _repository_path(
        root, worker_relative, label="exact-reference worker source", must_exist=True
    )
    runner_source = _repository_path(
        root, runner_relative, label="exact-reference runner source", must_exist=True
    )
    if (
        _file_sha(worker_source) != executor_sources[worker_relative]
        or _file_sha(runner_source) != executor_sources[runner_relative]
    ):
        raise ExactReferencePackageError(
            "exact-reference executor overlay changed after work-order creation"
        )
    members[_WORKER_SOURCE_MEMBER] = (worker_source, 0o644)
    members[_RUNNER_SOURCE_MEMBER] = (runner_source, 0o644)
    if generation == 2:
        bindings_source = _repository_path(
            root,
            _EXECUTION_BINDINGS_SOURCE_RELATIVE,
            label="exact-reference execution-binding repair source",
            must_exist=True,
        )
        members[_EXECUTION_BINDINGS_SOURCE_MEMBER] = (bindings_source, 0o644)
    rows = {
        name: _file_row(name, content, mode=mode)
        for name, (content, mode) in members.items()
    }
    worker = _render_worker(
        request=request,
        order=order,
        source=source,
        source_archive=source_archive,
        source_archive_bytes=archive_source.stat().st_size,
        runtime_image=image,
        packaged_files=rows,
        operational_generation=generation,
    )
    members[_WORKER_MEMBER] = (worker, 0o755)
    rows[_WORKER_MEMBER] = _file_row(_WORKER_MEMBER, worker, mode=0o755)
    inner = _seal(
        {
            "schema": EXACT_REFERENCE_PACKAGE_PAYLOAD_MANIFEST_V1,
            "lane_id": request.lane_id,
            "action_id": request.action_id,
            "execution_request_sha256": request.digest,
            "work_order_sha256": order.work_order_sha256,
            "source_epoch_identity_sha256": source.source_epoch.identity_sha256,
            "seed_receipt_sha256": seed.receipt_sha256,
            "seed_artifact_sha256": seed.artifact.sha256,
            "operational_generation": generation,
            "action_count": 1,
            "files": [rows[name] for name in sorted(rows)],
            "qiskit_compilation_requested": False,
            "remote_raw_evidence_policy": "preserve",
            "production_submission_enabled": False,
        },
        "package_payload_manifest_sha256",
    )
    members[_INNER_MANIFEST] = (_pretty_bytes(inner), 0o644)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        assert temporary is not None
        package_temp = temporary / _PACKAGE_NAME
        _write_deterministic_package(package_temp, members)
        _validate_package(package_temp)
        package_sha = _file_sha(package_temp)
        package_bytes = package_temp.stat().st_size
        launcher = _render_launcher(
            package_sha256=package_sha, package_bytes=package_bytes
        )
        _write_bytes(temporary / _LAUNCHER_NAME, launcher, executable=True)
        _write_bytes(temporary / _DESCRIPTOR_NAME, descriptor)
        (temporary / "logs").mkdir()
        package_file = VerifiedFile(
            (Path(output_relative) / _PACKAGE_NAME).as_posix(), package_sha
        )
        manifest_payload = _manifest_payload(
            request=request,
            source=source,
            package_path=package_temp,
            package_relative=package_file.path,
            operational_generation=generation,
        )
        _write_bytes(temporary / _MANIFEST_NAME, _pretty_bytes(manifest_payload))
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
        intent = _intent_payload(
            request=request,
            source=source,
            source_archive=source_archive,
            source_archive_bytes=archive_source.stat().st_size,
            runtime_image=image,
            package=package_file,
            package_bytes=package_bytes,
            manifest=manifest_file,
            descriptor=descriptor_file,
            launcher=launcher_file,
            operational_generation=generation,
        )
        _write_bytes(temporary / _INTENT_NAME, _pretty_bytes(intent))
        intent_file = VerifiedFile(
            (Path(output_relative) / _INTENT_NAME).as_posix(),
            _file_sha(temporary / _INTENT_NAME),
        )
        assessment = assess_fetch_storage(
            free_bytes=free,
            archive_bytes=package_bytes,
            expected_extracted_bytes=0,
            free_floor_bytes=DEFAULT_FREE_FLOOR_BYTES,
        )
        if not assessment.allowed:
            raise ExactReferencePackageError(
                "10-GiB floor rejects completed exact-reference package"
            )
        preflight = _seal(
            {
                "schema": EXACT_REFERENCE_PACKAGE_PREFLIGHT_V1,
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
                "source_epoch_full_archive_sha256": source.source_epoch.source_archive_sha256,
                "runtime_source_archive": source_archive.to_json_dict(),
                "runtime_source_archive_bytes": archive_source.stat().st_size,
                "runtime_image_identity": {
                    "path": image.identity_path,
                    "sha256": image.identity_sha256,
                },
                "runtime_image": {
                    "path": image.path,
                    "sha256": image.sha256,
                    "bytes": image.bytes,
                },
                "seed_receipt": {
                    "path": seed.receipt_path,
                    "sha256": seed.receipt_sha256,
                },
                "seed_artifact": seed.artifact.to_json_dict(),
                "resources": resources.to_json_dict(),
                "operational_generation": generation,
                "package": package_file.to_json_dict(),
                "package_bytes": package_bytes,
                "package_manifest": manifest_file.to_json_dict(),
                "submit_descriptor": descriptor_file.to_json_dict(),
                "launcher": launcher_file.to_json_dict(),
                "intent_template": intent_file.to_json_dict(),
                "storage_assessment": asdict(assessment),
                "storage_free_bytes_source": (
                    "filesystem_probe" if free_bytes is None else "caller_supplied_probe"
                ),
                "action_count": 1,
                "ordinary_queue_count": 1,
                "remote_raw_evidence_policy": "preserve",
                "qiskit_compilation_requested": False,
                "production_submission_enabled": False,
                "scheduler_contacted": False,
                "scheduler_submission_authorized": False,
                "intent_first": dict(BOOTSTRAP_INTENT_FIRST_REQUIREMENTS),
                "lifecycle_validator_executed": True,
                "lifecycle_validator_status": "pass",
                "condor_dry_run_argv": [
                    "condor_submit",
                    "-dry-run",
                    (Path(output_relative) / "dryrun.ad").as_posix(),
                    descriptor_file.path,
                ],
                "condor_dry_run_executed": False,
            },
            "exact_reference_package_preflight_sha256",
        )
        _write_bytes(temporary / _PREFLIGHT_NAME, _pretty_bytes(preflight))
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
    return PreparedExactReferenceActionPackage.load(
        output / _PREFLIGHT_NAME, repository_root=root
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
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
    materialize.add_argument(
        "--operational-generation", type=int, choices=(1, 2), default=1
    )
    validate = commands.add_parser("validate")
    validate.add_argument("--repo-root", default=".")
    validate.add_argument("--preflight-receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.repo_root).resolve(strict=True)
    if args.command == "materialize":
        source = SourceEpochReceipt.load(
            _repository_path(
                root,
                args.source_epoch_receipt,
                label="source epoch receipt",
                must_exist=True,
            ),
            repository_root=root,
        )
        image = load_runtime_image_identity(
            args.runtime_image_identity, repository_root=root
        )
        prepared = materialize_exact_reference_action_package(
            repository_root=root,
            output_dir=args.output_dir,
            request_path=args.request,
            work_order_path=args.work_order,
            source_epoch_receipt=source,
            source_archive_path=args.source_archive,
            runtime_image_identity=image,
            resources=SeedCHTCResources(
                args.request_cpus,
                args.request_memory_gib,
                args.request_disk_gib,
                args.max_runtime_seconds,
            ),
            operational_generation=args.operational_generation,
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
            "operational_generation": prepared.operational_generation,
            "condor_dry_run_argv": [
                "condor_submit",
                "-dry-run",
                str(Path(prepared.submit_descriptor.path).with_name("dryrun.ad")),
                prepared.submit_descriptor.path,
            ],
            "scheduler_submission_authorized": False,
        }
    else:
        prepared = PreparedExactReferenceActionPackage.load(
            args.preflight_receipt, repository_root=root
        )
        summary = {
            "status": prepared.status,
            "package": prepared.package.path,
            "package_sha256": prepared.package.sha256,
            "execution_request_sha256": prepared.request_sha256,
            "operational_generation": prepared.operational_generation,
            "remote_raw_evidence_policy": prepared.remote_raw_evidence_policy,
            "condor_dry_run_executed": prepared.condor_dry_run_executed,
            "scheduler_submission_authorized": False,
        }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


__all__ = [
    "EXACT_REFERENCE_PACKAGE_PAYLOAD_MANIFEST_V1",
    "EXACT_REFERENCE_PACKAGE_PREFLIGHT_V1",
    "EXACT_REFERENCE_SUBMISSION_INTENT_TEMPLATE_V1",
    "ExactReferencePackageError",
    "PreparedExactReferenceActionPackage",
    "exact_reference_submission_unit_id",
    "main",
    "materialize_exact_reference_action_package",
]


if __name__ == "__main__":
    raise SystemExit(main())
