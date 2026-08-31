"""Seal one APM package into the generic intent-first scheduler protocol."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Mapping, Sequence

from pipelines.time_dynamics.acquisition_supervisor import (
    SUBMISSION_INTENT_SCHEMA_V1,
    SubmissionUnit,
)
from pipelines.time_dynamics.prospective_apm_exchange_chtc_package import (
    PACKAGE_MANIFEST_SCHEMA_V1,
    PACKAGE_PREFLIGHT_SCHEMA_V1,
    SUBMISSION_INTENT_TEMPLATE_V1,
    apm_exchange_submission_unit_id,
)
from pipelines.time_dynamics.prospective_apm_exchange_worker import load_work_order
from pipelines.time_dynamics.prospective_execution_bindings import (
    ExecutionRequest,
    ScoreQualificationReceipt,
    VerifiedFile,
    load_action_registry_entry,
    resolve_scientific_action,
)
from pipelines.time_dynamics.prospective_scientific_planner import (
    ProspectiveEvidence,
    plan_prospective_cell,
)
from pipelines.time_dynamics.staged_campaign import AccuracyPolicy


BRIDGE_SCHEMA_V1 = "paper_ii_apm_exchange_generic_intent_bridge_v1"
_SHA = re.compile(r"[0-9a-f]{64}")
_SAFE = re.compile(r"[A-Za-z0-9_.-]+")


class APMIntentBridgeError(ValueError):
    """The APM action/package/intent identity is incomplete or stale."""


def _bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise APMIntentBridgeError(f"{label} must be a regular nonlink file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise APMIntentBridgeError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise APMIntentBridgeError(f"{label} must be a JSON object")
    return value


def _path(root: Path, value: str | Path, label: str) -> Path:
    candidate = Path(value)
    target = candidate if candidate.is_absolute() else root / candidate
    try:
        target = target.resolve(strict=True)
        target.relative_to(root)
    except (OSError, ValueError) as exc:
        raise APMIntentBridgeError(f"{label} is outside the repository") from exc
    if not target.is_file() or target.is_symlink():
        raise APMIntentBridgeError(f"{label} must be a regular nonlink file")
    return target


def _relative(root: Path, path: Path) -> str:
    return path.resolve(strict=False).relative_to(root).as_posix()


def _binding(root: Path, path: Path, *, include_bytes: bool = False) -> dict[str, Any]:
    row: dict[str, Any] = {"path": _relative(root, path), "sha256": _file_sha(path)}
    if include_bytes:
        row["bytes"] = path.stat().st_size
    return row


def _bound(root: Path, raw: Any, label: str, *, include_bytes: bool = False) -> Path:
    if not isinstance(raw, Mapping):
        raise APMIntentBridgeError(f"{label} binding is missing")
    path = _path(root, str(raw.get("path") or ""), label)
    if raw.get("sha256") != _file_sha(path):
        raise APMIntentBridgeError(f"{label} digest changed")
    if include_bytes and raw.get("bytes") != path.stat().st_size:
        raise APMIntentBridgeError(f"{label} byte count changed")
    return path


def _seal(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = dict(payload)
    result[key] = _digest(result)
    return result


def _verify_seal(payload: Mapping[str, Any], key: str, label: str) -> str:
    core = dict(payload)
    declared = core.pop(key, None)
    if not isinstance(declared, str) or _SHA.fullmatch(declared) is None:
        raise APMIntentBridgeError(f"{label} digest is invalid")
    if declared != _digest(core):
        raise APMIntentBridgeError(f"{label} self digest mismatch")
    return declared


def _write_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(dict(payload), indent=2, sort_keys=True).encode() + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise APMIntentBridgeError(f"immutable intent collision: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = ""
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _timestamp(value: str | None) -> str:
    text = value or dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise APMIntentBridgeError("recorded_at is not ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise APMIntentBridgeError("recorded_at lacks a timezone")
    return text


def _remote(value: str, label: str) -> str:
    path = PurePosixPath(str(value).strip())
    if path.is_absolute() or ".." in path.parts or not path.parts or any(
        _SAFE.fullmatch(part) is None for part in path.parts
    ):
        raise APMIntentBridgeError(f"{label} is not a safe home-relative path")
    return path.as_posix()


def seal_apm_exchange_intent(
    *, repository_root: str | Path, request_path: str | Path,
    work_order_path: str | Path, score_receipt_path: str | Path,
    planner_receipt_path: str | Path, action_registry_path: str | Path,
    package_preflight_path: str | Path, remote_root: str,
    recorded_at: str | None = None,
) -> dict[str, Any]:
    """Authenticate and seal one non-submittable, first-snapshot intent."""

    root = Path(repository_root).resolve(strict=True)
    request_file = _path(root, request_path, "execution request")
    request = ExecutionRequest.from_mapping(_read(request_file, "execution request"))
    order_file = _path(root, work_order_path, "APM work order")
    order, source, _seed = load_work_order(order_file, repository_root=root)
    if request != order.request:
        raise APMIntentBridgeError("execution request/work order mismatch")
    score_file = _path(root, score_receipt_path, "score receipt")
    score = ScoreQualificationReceipt.load(score_file, repository_root=root)
    planner_file = _path(root, planner_receipt_path, "planner receipt")
    policy = AccuracyPolicy.from_json_dict(order.bindings.to_json_dict()["accuracy_policy"])
    plan = plan_prospective_cell(
        order.bindings, policy,
        evidence=ProspectiveEvidence(runs=(score.to_planner_evidence(root, order.bindings),)),
    )
    if _read(planner_file, "planner receipt") != plan.to_json_dict():
        raise APMIntentBridgeError("planner receipt is not the score-derived plan")
    if len(plan.next_actions) != 1 or resolve_scientific_action(
        order.bindings, plan.next_actions[0]
    ) != request:
        raise APMIntentBridgeError("planner does not select this exact APM action")
    registry_file = _path(root, action_registry_path, "action registry")
    entry = load_action_registry_entry(
        VerifiedFile(_relative(root, registry_file), _file_sha(registry_file)),
        repository_root=root, bindings=order.bindings,
        expected_seed=order.seed, expected_reference=order.reference,
        expected_source_epoch=source.source_epoch,
    )
    if entry.request != request:
        raise APMIntentBridgeError("action registry request mismatch")

    preflight_file = _path(root, package_preflight_path, "package preflight")
    preflight = _read(preflight_file, "package preflight")
    _verify_seal(preflight, "package_preflight_sha256", "package preflight")
    unit_id = apm_exchange_submission_unit_id(order)
    batch = str(preflight.get("job_batch_name") or "")
    if (
        preflight.get("schema") != PACKAGE_PREFLIGHT_SCHEMA_V1
        or preflight.get("action_id") != request.action_id
        or preflight.get("execution_request_sha256") != request.digest
        or preflight.get("work_order_sha256") != order.work_order_sha256
        or preflight.get("acquisition_unit_id") != unit_id
        or not batch
        or preflight.get("scheduler_contacted") is not False
        or preflight.get("scheduler_submission_authorized") is not False
        or preflight.get("production_submission_enabled") is not False
        or preflight.get("qiskit_compilation_requested") is not False
    ):
        raise APMIntentBridgeError("package preflight is not inert/canonical")
    package = _bound(root, preflight.get("package"), "package")
    manifest_file = _bound(root, preflight.get("package_manifest"), "package manifest")
    descriptor_file = _bound(root, preflight.get("submit_descriptor"), "submit descriptor")
    template_file = _bound(root, preflight.get("intent_template"), "intent template")
    manifest = _read(manifest_file, "package manifest")
    template = _read(template_file, "intent template")
    _verify_seal(manifest, "package_manifest_sha256", "package manifest")
    _verify_seal(template, "submission_intent_template_sha256", "intent template")
    if (
        manifest.get("schema") != PACKAGE_MANIFEST_SCHEMA_V1
        or manifest.get("acquisition_unit_id") != unit_id
        or manifest.get("job_batch_name") != batch
        or manifest.get("execution_request_sha256") != request.digest
        or template.get("schema") != SUBMISSION_INTENT_TEMPLATE_V1
        or template.get("acquisition_unit_id") != unit_id
        or template.get("lane_id") != request.lane_id
        or template.get("job_batch_name") != batch
        or template.get("first_reconciliation_outcome") != "indeterminate"
        or template.get("scheduler_submission_authorized") is not False
        or template.get("production_submission_enabled") is not False
    ):
        raise APMIntentBridgeError("manifest/intent template identity diverged")
    descriptor = descriptor_file.read_text(encoding="utf-8")
    required = (
        f'+AcquisitionUnitId = "{unit_id}"',
        f'+AcquisitionLaneId = "{request.lane_id}"',
        f'+PaperIIActionId = "{request.action_id}"',
        f'+PaperIIExecutionRequestSha256 = "{request.digest}"',
        f'+JobBatchName = "{batch}"',
    )
    if any(descriptor.splitlines().count(line) != 1 for line in required):
        raise APMIntentBridgeError("submit descriptor generic identity is incomplete")

    remote_base = _remote(remote_root, "remote root")
    remote_workdir = _remote(
        (PurePosixPath(remote_base) / request.lane_id /
         order.reference.identity_sha256[:16] / "apm_exchange_r00").as_posix(),
        "remote workdir",
    )
    unit = SubmissionUnit(
        unit_id=unit_id, action_ids=(request.action_id,),
        payload={
            "package_kind": "apm_exchange",
            "execution_request": _binding(root, request_file),
            "work_order": _binding(root, order_file),
            "score_receipt": _binding(root, score_file),
            "planner_receipt": _binding(root, planner_file),
            "action_registry": _binding(root, registry_file),
            "package_preflight": _binding(root, preflight_file),
            "package_manifest": _binding(root, manifest_file),
            "package": _binding(root, package, include_bytes=True),
            "submit_descriptor": _binding(root, descriptor_file),
            "intent_template": _binding(root, template_file),
            "job_batch_name": batch,
            "source_epoch_identity_sha256": order.bindings.source_epoch_identity_sha256,
            "remote_workdir": remote_workdir,
            "remote_package_path": (PurePosixPath(remote_workdir) / package.name).as_posix(),
            "first_reconciliation_outcome": "indeterminate",
            "submission_permitted": False,
            "qiskit_compilation": False,
            "production_submission_enabled": False,
        },
    )
    core = {"schema": SUBMISSION_INTENT_SCHEMA_V1, "lane_id": request.lane_id,
            "unit": unit.to_json_dict()}
    state = root / "output" / request.lane_id / "acquisition_supervisor"
    intent_path = state / "submission_intents" / f"{unit_id}.json"
    if intent_path.exists():
        intent = _read(intent_path, "existing APM submission intent")
        stored_core = {
            "schema": intent.get("schema"),
            "lane_id": intent.get("lane_id"),
            "unit": intent.get("unit"),
        }
        stored_time = _timestamp(str(intent.get("recorded_at") or ""))
        if (
            stored_core != core
            or intent.get("intent_sha256") != _digest(core)
            or (
                recorded_at is not None
                and _timestamp(recorded_at) != stored_time
            )
        ):
            raise APMIntentBridgeError(
                "existing APM submission intent differs from replay"
            )
        recorded = stored_time
    else:
        recorded = _timestamp(recorded_at)
        intent = {
            **core,
            "intent_sha256": _digest(core),
            "recorded_at": recorded,
        }
        _write_immutable(intent_path, intent)
    generic = {
        "job_batch_name": batch,
        "acquisition_unit_id": unit_id,
        "paper_ii_action_id": request.action_id,
        "paper_ii_execution_request_sha256": request.digest,
        "intent_sha256": intent["intent_sha256"],
        "recorded_at": recorded,
    }
    bridge = _seal(
        {
            "schema": BRIDGE_SCHEMA_V1,
            "source_submission_intent": {
                **_binding(root, intent_path), "intent_sha256": intent["intent_sha256"]
            },
            "generic_reconciliation_intent": generic,
            "first_reconciliation_outcome": "indeterminate",
            "submission_permitted": False,
            "scheduler_contacted": False,
            "qiskit_compilation_requested": False,
            "production_submission_enabled": False,
        },
        "bridge_sha256",
    )
    bridge_path = state / "intent_bridges" / f"{unit_id}.json"
    _write_immutable(bridge_path, bridge)
    load_apm_exchange_intent_bridge(bridge_path, repository_root=root)
    return {
        "intent_path": _relative(root, intent_path),
        "intent_file_sha256": _file_sha(intent_path),
        "intent_sha256": intent["intent_sha256"],
        "bridge_path": _relative(root, bridge_path),
        "bridge_file_sha256": _file_sha(bridge_path),
        "bridge_sha256": bridge["bridge_sha256"],
        "generic_reconciliation_intent": generic,
        "remote_workdir": remote_workdir,
        "remote_package_path": unit.payload["remote_package_path"],
        "submit_descriptor": _relative(root, descriptor_file),
    }


def load_apm_exchange_intent_bridge(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    """Reload a bridge and every byte binding needed for reconciliation."""

    root = Path(repository_root).resolve(strict=True)
    bridge_path = _path(root, path, "APM intent bridge")
    bridge = _read(bridge_path, "APM intent bridge")
    _verify_seal(bridge, "bridge_sha256", "APM intent bridge")
    intent_row = bridge.get("source_submission_intent")
    generic = bridge.get("generic_reconciliation_intent")
    if (
        bridge.get("schema") != BRIDGE_SCHEMA_V1
        or not isinstance(intent_row, Mapping)
        or not isinstance(generic, Mapping)
        or bridge.get("first_reconciliation_outcome") != "indeterminate"
        or bridge.get("submission_permitted") is not False
        or bridge.get("scheduler_contacted") is not False
        or bridge.get("qiskit_compilation_requested") is not False
        or bridge.get("production_submission_enabled") is not False
    ):
        raise APMIntentBridgeError("APM intent bridge disposition is invalid")
    intent_path = _bound(root, intent_row, "source submission intent")
    intent = _read(intent_path, "source submission intent")
    unit = intent.get("unit")
    if not isinstance(unit, Mapping) or not isinstance(unit.get("payload"), Mapping):
        raise APMIntentBridgeError("source submission unit is incomplete")
    core = {
        "schema": intent.get("schema"), "lane_id": intent.get("lane_id"),
        "unit": unit,
    }
    intent_sha = intent.get("intent_sha256")
    if (
        intent.get("schema") != SUBMISSION_INTENT_SCHEMA_V1
        or intent_sha != _digest(core)
        or intent_row.get("intent_sha256") != intent_sha
    ):
        raise APMIntentBridgeError("source submission intent digest changed")
    unit_id = str(unit.get("unit_id") or "")
    action_ids = unit.get("action_ids")
    payload = unit["payload"]
    if (
        not unit_id
        or not isinstance(action_ids, list)
        or len(action_ids) != 1
        or payload.get("package_kind") != "apm_exchange"
        or payload.get("first_reconciliation_outcome") != "indeterminate"
        or payload.get("submission_permitted") is not False
        or payload.get("qiskit_compilation") is not False
        or payload.get("production_submission_enabled") is not False
    ):
        raise APMIntentBridgeError("source submission intent is not fail-closed")
    for name in (
        "execution_request", "work_order", "score_receipt", "planner_receipt",
        "action_registry", "package_preflight", "package_manifest",
        "submit_descriptor", "intent_template",
    ):
        _bound(root, payload.get(name), f"intent {name}")
    _bound(root, payload.get("package"), "intent package", include_bytes=True)
    expected = {
        "job_batch_name": payload.get("job_batch_name"),
        "acquisition_unit_id": unit_id,
        "paper_ii_action_id": action_ids[0],
        "paper_ii_execution_request_sha256": _read(
            root / payload["execution_request"]["path"], "bound execution request"
        ).get("request_sha256"),
        "intent_sha256": intent_sha,
        "recorded_at": intent.get("recorded_at"),
    }
    # ExecutionRequest uses ``request_sha256`` as its self-digest key.
    if not isinstance(expected["paper_ii_execution_request_sha256"], str):
        expected["paper_ii_execution_request_sha256"] = _read(
            root / payload["execution_request"]["path"], "bound execution request"
        ).get("execution_request_sha256")
    if dict(generic) != expected or set(generic) != set(expected):
        raise APMIntentBridgeError("generic reconciliation identity diverged")
    if any(
        not isinstance(expected[name], str) or not expected[name]
        for name in expected
    ):
        raise APMIntentBridgeError("generic reconciliation identity is incomplete")
    if _SHA.fullmatch(str(expected["paper_ii_execution_request_sha256"])) is None:
        raise APMIntentBridgeError("generic request digest is invalid")
    recorded_path = (
        root / "output" / str(intent["lane_id"]) / "acquisition_supervisor"
        / "submission_intents" / f"{unit_id}.json"
    ).resolve(strict=True)
    if intent_path != recorded_path:
        raise APMIntentBridgeError("source submission intent path is noncanonical")
    return bridge


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=".")
    for name in ("request", "work-order", "score-receipt", "planner-receipt",
                 "action-registry", "package-preflight"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--recorded-at")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = seal_apm_exchange_intent(
        repository_root=args.repository_root, request_path=args.request,
        work_order_path=args.work_order, score_receipt_path=args.score_receipt,
        planner_receipt_path=args.planner_receipt,
        action_registry_path=args.action_registry,
        package_preflight_path=args.package_preflight,
        remote_root=args.remote_root, recorded_at=args.recorded_at,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
