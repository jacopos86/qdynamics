#!/usr/bin/env python3
"""Validate and execute one authorization-bound Study-1 direct cell.

The worker consumes an immutable bundle protocol through the validated bundle
loader, reconstructs the typed problem request, and calls only the public RA or
Append facade.  It never edits a protocol, template, source lock, or bundle
manifest.  The Paper-I summary is serialized directly from the facade result's
typed canonical summary seam.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import traceback
from dataclasses import is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from package_contract import (  # noqa: E402
    EXECUTION_PLAN_SCHEMA,
    EXPECTED_ARTIFACT_ROLES,
    JOB_SPEC_SCHEMA,
    PACKAGE_ID,
    PACKAGE_MANIFEST_SCHEMA,
    REMOTE_IMAGE_SHA256,
    V7_RELATIVE_ROOT,
    WORKER_RECEIPT_SCHEMA,
    PackageContractError,
    atomic_write_json,
    canonical_sha256,
    digested,
    direct_execution_ids,
    load_json_object,
    objective_gate_diagnostic_contract,
    package_control_plane_receipt,
    safe_relative_path,
    sha256_file,
    validate_authorization_receipt,
    validate_v7_authority,
    verify_exact_key_set,
    verify_self_digest,
)
from objective_gates import (  # noqa: E402
    build_g11_replay_diagnostic,
    validate_cell_objective_gates,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write_json_value(path: Path, value: Any) -> None:
    """Atomically stream canonical JSON without duplicating a large result."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise PackageContractError(f"Refusing to overwrite run artifact: {path}")
    fd, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(raw_temporary)
    encoder = json.JSONEncoder(
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            for chunk in encoder.iterencode(value):
                stream.write(chunk)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _package_file_map(
    package_manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    files = package_manifest.get("files")
    if not isinstance(files, list):
        raise PackageContractError("Package manifest has no file inventory.")
    mapped: dict[str, Mapping[str, Any]] = {}
    for row in files:
        if not isinstance(row, Mapping):
            raise PackageContractError("Package manifest file row is invalid.")
        relative = safe_relative_path(
            row.get("path"), label="package manifest path"
        ).as_posix()
        if relative in mapped:
            raise PackageContractError(
                f"Duplicate package manifest path: {relative}"
            )
        mapped[relative] = row
    return mapped


def _verify_bound_package_file(
    *,
    package_dir: Path,
    file_map: Mapping[str, Mapping[str, Any]],
    path: Path,
    label: str,
) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(package_dir.resolve()).as_posix()
    except ValueError as exc:
        raise PackageContractError(
            f"{label} is outside the package-only tree: {resolved}"
        ) from exc
    row = file_map.get(relative)
    if not isinstance(row, Mapping):
        raise PackageContractError(
            f"{label} is not package-manifest-bound: {relative}"
        )
    actual = sha256_file(resolved)
    if (
        row.get("sha256") != actual
        or int(row.get("size_bytes", -1)) != resolved.stat().st_size
    ):
        raise PackageContractError(f"{label} package hash drifted: {relative}")
    return actual


def _validate_control_plane(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve()
    package_manifest_path = args.package_manifest.resolve()
    package_dir = package_manifest_path.parent
    if not source_root.is_dir() or source_root.is_symlink():
        raise PackageContractError(
            f"Source-locked root is unavailable or unsafe: {source_root}"
        )
    try:
        package_manifest_path.relative_to(source_root)
        package_dir.relative_to(source_root)
    except ValueError as exc:
        raise PackageContractError(
            "The package control plane must be contained by source-root."
        ) from exc
    # Objective-authority validation dynamically imports the authenticated
    # pipeline implementation.  In the clean CHTC container, /work is not on
    # sys.path merely because this worker lives in a nested package directory.
    # Install the already source-archive-authenticated root before that
    # validation, then let validate_v7_authority rehash every trusted member.
    source_root_text = str(source_root)
    if source_root_text not in sys.path:
        sys.path.insert(0, source_root_text)
    package_manifest = load_json_object(
        package_manifest_path, label="package manifest"
    )
    verify_self_digest(package_manifest, label="package manifest")
    if (
        package_manifest.get("schema") != PACKAGE_MANIFEST_SCHEMA
        or package_manifest.get("package_id") != PACKAGE_ID
        or int(package_manifest.get("logical_cell_count", -1)) != 20
        or int(package_manifest.get("direct_execution_count", -1)) != 18
        or package_manifest.get("submission_state") != "not_submitted"
    ):
        raise PackageContractError("Package-manifest state drifted.")
    file_map = _package_file_map(package_manifest)
    for path, label in (
        (args.job_spec, "job spec"),
        (args.execution_plan, "execution plan"),
        (args.authorization_receipt, "authorization receipt"),
        (args.v7_final_receipt, "v7 final receipt"),
        (args.objective_gate_authority, "objective-gate authority"),
        (Path(__file__), "worker implementation"),
        (PACKAGE_DIR / "package_contract.py", "package contract"),
        (PACKAGE_DIR / "objective_gates.py", "objective-gate validator"),
    ):
        _verify_bound_package_file(
            package_dir=package_dir,
            file_map=file_map,
            path=path,
            label=label,
        )

    plan = load_json_object(args.execution_plan, label="execution plan")
    verify_self_digest(plan, label="execution plan")
    if (
        plan.get("schema") != EXECUTION_PLAN_SCHEMA
        or plan.get("package_id") != PACKAGE_ID
        or plan.get("sha256")
        != package_manifest.get("execution_plan_sha256")
        or int(plan.get("logical_cell_count", -1)) != 20
        or int(plan.get("direct_execution_count", -1)) != 18
        or int(plan.get("shared_reference_count", -1)) != 2
        or plan.get("execution_authorized") is not True
        or plan.get("submission_authorized") is not True
        or plan.get("submission_state") != "not_submitted"
    ):
        raise PackageContractError("Execution-plan state drifted.")

    v7_authority = validate_v7_authority(
        source_root,
        v7_root=source_root / V7_RELATIVE_ROOT,
        final_receipt_path=args.v7_final_receipt,
        objective_gate_authority_path=args.objective_gate_authority,
    )
    final_binding = v7_authority["final_receipt_binding"]
    if (
        final_binding["canonical_sha256"]
        != plan["v7_final_receipt"]["canonical_sha256"]
        or final_binding["file_sha256"]
        != plan["v7_final_receipt"]["file_sha256"]
        or final_binding["canonical_sha256"]
        != package_manifest["v7_final_receipt_sha256"]
    ):
        raise PackageContractError("External v7 final receipt binding drifted.")
    objective_binding = v7_authority["objective_gate_authority_binding"]
    if (
        objective_binding["canonical_sha256"]
        != plan["study1_objective_gate_authority"]["canonical_sha256"]
        or objective_binding["file_sha256"]
        != plan["study1_objective_gate_authority"]["file_sha256"]
        or objective_binding["canonical_sha256"]
        != package_manifest[
            "study1_objective_gate_authority_sha256"
        ]
    ):
        raise PackageContractError(
            "External objective-gate authority binding drifted."
        )

    control_plane = package_control_plane_receipt(package_dir)
    if (
        package_manifest.get("package_control_plane_sha256")
        != control_plane["sha256"]
        or plan.get("package_control_plane") != control_plane
    ):
        raise PackageContractError(
            "Authorization-bound package control plane drifted."
        )
    authorization = load_json_object(
        args.authorization_receipt, label="authorization receipt"
    )
    validate_authorization_receipt(
        authorization,
        v7_authority=v7_authority,
        package_control_plane_sha256=control_plane["sha256"],
    )
    authorization_file_sha256 = sha256_file(args.authorization_receipt)
    if (
        authorization["sha256"] != plan["authorization"]["canonical_sha256"]
        or authorization_file_sha256
        != plan["authorization"]["file_sha256"]
        or authorization["sha256"]
        != package_manifest["authorization_sha256"]
    ):
        raise PackageContractError("External authorization binding drifted.")

    job = load_json_object(args.job_spec, label="job spec")
    verify_self_digest(job, label="job spec")
    if (
        job.get("schema") != JOB_SPEC_SCHEMA
        or job.get("package_id") != PACKAGE_ID
        or job.get("execution_plan_sha256") != plan["sha256"]
        or job.get("execution_authorized") is not True
        or job.get("submission_authorized") is not True
        or job.get("authorization") != plan["authorization"]
        or job.get("v7_final_receipt") != plan["v7_final_receipt"]
        or job.get("study1_objective_gate_authority")
        != plan["study1_objective_gate_authority"]
        or job.get("study1_dedupe_sha256")
        != v7_authority["dedupe_sha256"]
        or job.get("package_control_plane") != control_plane
        or job.get("source_inventory_sha256")
        != v7_authority["source_inventory"]["sha256"]
        or job.get("remote_image", {}).get("sha256")
        != REMOTE_IMAGE_SHA256
    ):
        raise PackageContractError("Job-spec authority drifted.")
    execution_id = str(job.get("execution_id", ""))
    if execution_id not in direct_execution_ids():
        raise PackageContractError(
            f"Job is not one of the exact 18 direct executions: {execution_id}"
        )
    if execution_id not in authorization["authorized_direct_execution_ids"]:
        raise PackageContractError(
            f"Job is absent from authorization receipt: {execution_id}"
        )
    direct_rows = plan.get("direct_executions")
    if (
        not isinstance(direct_rows, list)
        or any(not isinstance(row, Mapping) for row in direct_rows)
        or [row.get("execution_id") for row in direct_rows]
        != list(direct_execution_ids())
    ):
        raise PackageContractError("Execution plan has no direct rows.")
    plan_rows = [
        row
        for row in direct_rows
        if isinstance(row, Mapping)
        and row.get("execution_id") == execution_id
    ]
    if len(plan_rows) != 1:
        raise PackageContractError(
            f"Execution plan has no unique direct row: {execution_id}"
        )
    plan_row = plan_rows[0]
    for field in (
        "bundle_id",
        "cell_id",
        "regime_id",
        "route_id",
        "execution_entrypoint",
        "protocol",
        "execution_template",
        "execution_fulfillment",
        "artifact_paths",
        "objective_gate_diagnostics",
        "resources",
    ):
        if job.get(field) != plan_row.get(field):
            raise PackageContractError(
                f"Job/plan binding drifted at {execution_id}.{field}."
            )
    if job.get("objective_gate_diagnostics") != (
        objective_gate_diagnostic_contract(
            bundle_id=str(job["bundle_id"]),
            regime_id=str(job["regime_id"]),
            route_id=str(job["route_id"]),
        )
    ):
        raise PackageContractError(
            "Job objective-gate diagnostic selection drifted."
        )
    if (
        job["execution_fulfillment"].get("fulfillment_kind")
        == "shared_result_reference_v1"
    ):
        raise PackageContractError(
            "A direct worker cannot consume a reference-only logical cell."
        )
    if (
        args.source_archive_sha256 is not None
        and args.source_archive_sha256 != job["source_archive"]["sha256"]
    ):
        raise PackageContractError(
            "Wrapper-reported source archive hash drifted."
        )
    if args.mode == "execute":
        if args.verified_image_sha256 != REMOTE_IMAGE_SHA256:
            raise PackageContractError(
                "Wrapper did not authenticate the exact execution image."
            )
    elif (
        args.verified_image_sha256 is not None
        and args.verified_image_sha256 != REMOTE_IMAGE_SHA256
    ):
        raise PackageContractError("Reported image hash drifted.")

    protocol_path = source_root / job["protocol"]["path"]
    template_path = source_root / job["execution_template"]["path"]
    for path, binding, label in (
        (protocol_path, job["protocol"], "protocol"),
        (template_path, job["execution_template"], "execution template"),
    ):
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != binding["file_sha256"]
        ):
            raise PackageContractError(
                f"Source-locked {label} file hash drifted: {path}"
            )
        payload = load_json_object(path, label=label)
        verify_self_digest(payload, label=label)
        if payload["sha256"] != binding["canonical_sha256"]:
            raise PackageContractError(
                f"Source-locked {label} canonical hash drifted."
            )

    artifact_paths = {
        role: source_root
        / safe_relative_path(
            job["artifact_paths"][role],
            label=f"{execution_id} {role} path",
        )
        for role in EXPECTED_ARTIFACT_ROLES
    }
    verify_exact_key_set(
        artifact_paths,
        EXPECTED_ARTIFACT_ROLES,
        label="job artifact roles",
    )
    worker_receipt_path = source_root / safe_relative_path(
        job["worker_receipt_path"], label="worker receipt path"
    )
    return {
        "source_root": source_root,
        "package_manifest": package_manifest,
        "plan": plan,
        "authorization": authorization,
        "v7_authority": v7_authority,
        "package_control_plane": control_plane,
        "job": job,
        "protocol_path": protocol_path,
        "template_path": template_path,
        "artifact_paths": artifact_paths,
        "worker_receipt_path": worker_receipt_path,
    }


def _immutable_bundle_hashes(
    state: Mapping[str, Any],
) -> dict[str, str]:
    source_root = state["source_root"]
    bundle = state["v7_authority"]["bundle_bindings"][
        state["job"]["bundle_id"]
    ]
    bundle_root = source_root / bundle["bundle_root"]
    hashes: dict[str, str] = {}
    for path in sorted(bundle_root.rglob("*")):
        relative = path.relative_to(bundle_root)
        if "runs" in relative.parts:
            continue
        if path.is_symlink():
            raise PackageContractError(
                f"Immutable bundle contains a symlink: {path}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise PackageContractError(
                f"Immutable bundle contains a non-file: {path}"
            )
        hashes[relative.as_posix()] = sha256_file(path)
    if not hashes:
        raise PackageContractError("Immutable bundle snapshot is empty.")
    return hashes


def _load_problem_and_protocol(state: Mapping[str, Any]) -> tuple[Any, Any]:
    source_root = str(state["source_root"])
    if source_root in sys.path:
        sys.path.remove(source_root)
    sys.path.insert(0, source_root)

    from pipelines.contracts.problem import ProblemRequest
    from pipelines.static_adapt.builders.problem_registry import (
        resolve_problem_context,
    )
    from pipelines.static_adapt.ra_adapt.bundles import (
        load_validated_bundle_protocol,
    )
    from pipelines.static_adapt.sr_snake.contracts import (
        ResolvedProblemReceipt,
    )

    protocol = load_validated_bundle_protocol(state["protocol_path"])
    receipt = protocol.problem
    request = ProblemRequest(
        problem_key=str(receipt.problem_key),
        num_sites=int(receipt.num_sites),
        t=float(receipt.t),
        u=float(receipt.u),
        dv=float(receipt.dv),
        omega0=float(receipt.omega0),
        g_ep=float(receipt.g_ep),
        n_ph_max=int(receipt.n_ph_max),
        boson_encoding=str(receipt.boson_encoding),
        ordering=str(receipt.ordering),
        boundary=str(receipt.boundary),
        include_zero_point=bool(receipt.include_zero_point),
        v_nn=float(receipt.v_nn),
        t_prime=float(receipt.t_prime),
        n_fermions=(
            None
            if receipt.n_fermions is None
            else int(receipt.n_fermions)
        ),
    )
    problem = resolve_problem_context(request)
    observed_receipt = ResolvedProblemReceipt.from_problem(problem)
    if observed_receipt.to_dict() != receipt.to_dict():
        raise PackageContractError(
            "Reconstructed ProblemRequest does not reproduce protocol.problem."
        )
    return problem, protocol


def _typed_summary(result: Any, *, execution_entrypoint: str) -> Mapping[str, Any]:
    if execution_entrypoint == "run_append_adapt":
        summary = getattr(result, "paper_i_summary", None)
        expected_schema = "paper_i_append_run_summary_v1"
    elif execution_entrypoint == "run_ra_adapt":
        run = getattr(result, "run", None)
        summary = getattr(run, "paper_i_summary", None)
        expected_schema = "paper_i_run_summary_v1"
    else:
        raise PackageContractError(
            f"Unknown execution entrypoint: {execution_entrypoint}"
        )
    if (
        summary is None
        or not is_dataclass(summary)
        or getattr(summary, "schema", None) != expected_schema
        or not callable(getattr(summary, "to_dict", None))
    ):
        raise PackageContractError(
            "Facade result lacks its required typed canonical Paper-I summary."
        )
    payload = summary.to_dict()
    if not isinstance(payload, Mapping):
        raise PackageContractError("Typed Paper-I summary is not a mapping.")
    return payload


def _invoke_facade(
    *,
    problem: Any,
    protocol: Any,
    execution_entrypoint: str,
    operational_controls: Any = None,
) -> Any:
    if execution_entrypoint == "run_append_adapt":
        if operational_controls is not None:
            raise PackageContractError(
                "Append-ADAPT has no public continuation controls."
            )
        from pipelines.static_adapt.ra_adapt import run_append_adapt

        return run_append_adapt(problem, protocol)
    if execution_entrypoint == "run_ra_adapt":
        from pipelines.static_adapt.ra_adapt import run_ra_adapt

        return run_ra_adapt(
            problem,
            protocol,
            operational_controls=operational_controls,
        )
    raise PackageContractError(
        f"Unsupported facade: {execution_entrypoint}"
    )


def _run_g11_diagnostics(
    *,
    job: Mapping[str, Any],
    problem: Any,
    protocol: Any,
    primary_result_payload: Mapping[str, Any],
) -> dict[str, Any]:
    contract = job.get("objective_gate_diagnostics")
    if not isinstance(contract, Mapping):
        raise PackageContractError(
            "Job lacks its objective-gate diagnostic contract."
        )
    if contract.get("selected") is not True:
        return build_g11_replay_diagnostic(
            job=job,
            primary_result=primary_result_payload,
            secondary_result=None,
            resumed_result=None,
            resume_checkpoint_file_sha256=None,
        )

    original_cwd = Path.cwd()
    secondary_payload: Mapping[str, Any]
    resumed_payload: Mapping[str, Any] | None = None
    checkpoint_sha256: str | None = None
    if job["execution_entrypoint"] == "run_ra_adapt":
        from pipelines.static_adapt.ra_adapt import (
            RAAdaptOperationalControls,
        )
        from pipelines.static_adapt.sr_snake import (
            AcceptedStateResume,
            CheckpointObservation,
            EstimatorLedgerObservation,
            SRObservationPolicy,
        )

        first_leg_rounds = int(
            contract.get("ra_fresh_leg_maximum_controller_rounds", 0)
        )
        resumed_rounds = int(
            contract.get("ra_resumed_maximum_controller_rounds", 0)
        )
        if first_leg_rounds != 2 or resumed_rounds != 3:
            raise PackageContractError(
                "Selected RA G11 diagnostic horizons drifted."
            )
        with tempfile.TemporaryDirectory(
            prefix=f"{job['execution_id']}__bounded_resume__"
        ) as diagnostic_raw:
            diagnostic_root = Path(diagnostic_raw)
            first_checkpoint = diagnostic_root / "first_leg.current.json"
            first_ledger = diagnostic_root / "first_leg.ledger.json"
            resumed_checkpoint = diagnostic_root / "resumed.current.json"
            resumed_ledger = diagnostic_root / "resumed.ledger.json"
            try:
                os.chdir(diagnostic_root)
                secondary = _invoke_facade(
                    problem=problem,
                    protocol=protocol,
                    execution_entrypoint="run_ra_adapt",
                    operational_controls=RAAdaptOperationalControls(
                        maximum_controller_rounds=first_leg_rounds,
                        observation=SRObservationPolicy(
                            checkpoint=CheckpointObservation(
                                path=first_checkpoint,
                                every_controller_rounds=1,
                                keep_history_tail=100,
                            ),
                            estimator_ledger=EstimatorLedgerObservation(
                                path=first_ledger
                            ),
                        ),
                    ),
                )
                secondary_payload = secondary.to_dict()
                if (
                    not isinstance(secondary_payload, Mapping)
                    or not first_checkpoint.is_file()
                ):
                    raise PackageContractError(
                        "Bounded fresh leg did not produce its authenticated "
                        "checkpoint."
                    )
                checkpoint_sha256 = sha256_file(first_checkpoint)
                resumed = _invoke_facade(
                    problem=problem,
                    protocol=protocol,
                    execution_entrypoint="run_ra_adapt",
                    operational_controls=RAAdaptOperationalControls(
                        maximum_controller_rounds=resumed_rounds,
                        resume=AcceptedStateResume(
                            checkpoint_path=first_checkpoint,
                            checkpoint_sha256=checkpoint_sha256,
                        ),
                        observation=SRObservationPolicy(
                            checkpoint=CheckpointObservation(
                                path=resumed_checkpoint,
                                every_controller_rounds=1,
                                keep_history_tail=100,
                            ),
                            estimator_ledger=EstimatorLedgerObservation(
                                path=resumed_ledger
                            ),
                        ),
                    ),
                )
            finally:
                os.chdir(original_cwd)
            resumed_payload = resumed.to_dict()
            if not isinstance(resumed_payload, Mapping):
                raise PackageContractError(
                    "Authenticated continuation result is not serializable."
                )
    else:
        with tempfile.TemporaryDirectory(
            prefix=f"{job['execution_id']}__independent_replay__"
        ) as secondary_raw:
            try:
                os.chdir(secondary_raw)
                secondary = _invoke_facade(
                    problem=problem,
                    protocol=protocol,
                    execution_entrypoint=str(job["execution_entrypoint"]),
                )
            finally:
                os.chdir(original_cwd)
            secondary_payload = secondary.to_dict()
            if not isinstance(secondary_payload, Mapping):
                raise PackageContractError(
                    "Independent replay result is not serializable."
                )

    return build_g11_replay_diagnostic(
        job=job,
        primary_result=primary_result_payload,
        secondary_result=secondary_payload,
        resumed_result=resumed_payload,
        resume_checkpoint_file_sha256=checkpoint_sha256,
    )


def _execute(state: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    job = state["job"]
    execution_id = job["execution_id"]
    artifact_paths = state["artifact_paths"]
    collisions = [str(path) for path in artifact_paths.values() if path.exists()]
    if state["worker_receipt_path"].exists():
        collisions.append(str(state["worker_receipt_path"]))
    if collisions:
        raise PackageContractError(
            "Refusing to overwrite existing execution state: "
            + ", ".join(collisions)
        )
    immutable_before = _immutable_bundle_hashes(state)
    started_at = _utc_now()
    bundle_root = state["protocol_path"].parent.parent
    original_cwd = Path.cwd()
    try:
        problem, protocol = _load_problem_and_protocol(state)
        os.chdir(bundle_root)
        result = _invoke_facade(
            problem=problem,
            protocol=protocol,
            execution_entrypoint=str(job["execution_entrypoint"]),
        )
    finally:
        os.chdir(original_cwd)

    summary_payload = dict(
        _typed_summary(
            result,
            execution_entrypoint=job["execution_entrypoint"],
        )
    )
    if job["execution_entrypoint"] == "run_append_adapt":
        scientific_receipts = getattr(
            result, "scientific_receipts", None
        )
        if (
            not isinstance(scientific_receipts, Mapping)
            or scientific_receipts.get(
                "paper_i_append_run_summary"
            )
            != summary_payload
            or scientific_receipts.get(
                "paper_i_append_run_summary_sha256"
            )
            != canonical_sha256(summary_payload)
        ):
            raise PackageContractError(
                "Append result did not authenticate its typed canonical "
                "Paper-I summary."
            )
    result_payload = result.to_dict()
    if not isinstance(result_payload, Mapping):
        raise PackageContractError("Facade result is not canonically serializable.")

    checkpoint_path = artifact_paths["checkpoint"]
    ledger_path = artifact_paths["estimator_ledger"]
    loaded_observations: dict[str, Mapping[str, Any]] = {}
    for role, path, label in (
        ("checkpoint", checkpoint_path, "checkpoint"),
        ("estimator_ledger", ledger_path, "estimator ledger"),
    ):
        if not path.is_file() or path.is_symlink():
            raise PackageContractError(
                f"Facade did not atomically produce required {label}: {path}"
            )
        loaded_observations[role] = load_json_object(path, label=label)

    g11_replay_diagnostic = _run_g11_diagnostics(
        job=job,
        problem=problem,
        protocol=protocol,
        primary_result_payload=result_payload,
    )
    objective_gate_receipt = validate_cell_objective_gates(
        job=job,
        protocol=result_payload["protocol"],
        checkpoint=loaded_observations["checkpoint"],
        checkpoint_file_sha256=sha256_file(checkpoint_path),
        checkpoint_size_bytes=checkpoint_path.stat().st_size,
        ledger=loaded_observations["estimator_ledger"],
        ledger_file_sha256=sha256_file(ledger_path),
        ledger_size_bytes=ledger_path.stat().st_size,
        result=result_payload,
        summary=summary_payload,
        objective_authority=state["v7_authority"][
            "objective_gate_authority"
        ],
        replay_diagnostic=g11_replay_diagnostic,
    )

    _atomic_write_json_value(artifact_paths["result"], result_payload)
    _atomic_write_json_value(artifact_paths["summary"], summary_payload)
    output_bindings = {
        role: {
            "path": job["artifact_paths"][role],
            "sha256": sha256_file(artifact_paths[role]),
            "size_bytes": artifact_paths[role].stat().st_size,
            "status": "produced",
        }
        for role in (
            "checkpoint",
            "estimator_ledger",
            "result",
            "summary",
        )
    }
    template = load_json_object(
        state["template_path"], label="execution template"
    )
    template_sha256 = template.pop("sha256")
    template_outputs = template.get("output_artifacts")
    if not isinstance(template_outputs, Mapping) or set(template_outputs) != {
        "checkpoint",
        "estimator_ledger",
        "result",
        "summary",
    }:
        raise PackageContractError(
            "Execution template output-artifact contract drifted."
        )
    manifest_output_bindings = {
        role: {
            **output_bindings[role],
            "path": template_outputs[role]["path"],
        }
        for role in (
            "checkpoint",
            "estimator_ledger",
            "result",
            "summary",
        )
    }
    finished_at = _utc_now()
    execution_manifest = digested(
        {
            **template,
            "schema": "paper_i_ra_adapt_study1_execution_manifest_v1",
            "source_execution_template_sha256": template_sha256,
            "execution_state": "completed",
            "execution_authorized": True,
            "submission_state": (
                "submitted_to_authorized_target"
                if args.mode == "execute"
                else "local_packaged_smoke_not_submitted"
            ),
            "submitted": args.mode == "execute",
            "runtime_mode": args.mode,
            "command_argv": list(sys.argv),
            "command_argv_status": "recorded",
            "cwd": str(bundle_root.relative_to(state["source_root"])),
            "cwd_status": "source_locked_bundle_root",
            "dirty_working_tree": None,
            "dirty_working_tree_status": "not_applicable_to_source_archive",
            "environment_fingerprint": {
                "python": sys.version,
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "source_archive_sha256": job["source_archive"]["sha256"],
                "remote_image_sha256": (
                    args.verified_image_sha256
                    if args.mode == "execute"
                    else None
                ),
            },
            "environment_fingerprint_status": "recorded",
            "git_commit": None,
            "git_commit_status": "source_archive_hash_is_authority",
            "exit_status": 0,
            "exit_status_status": "recorded",
            "timestamps": {
                "started_at": started_at,
                "finished_at": finished_at,
            },
            "timestamps_status": "recorded",
            "package_authority": {
                "package_id": PACKAGE_ID,
                "package_manifest_sha256": state["package_manifest"]["sha256"],
                "execution_plan_sha256": state["plan"]["sha256"],
                "job_spec_sha256": job["sha256"],
                "authorization_sha256": state["authorization"]["sha256"],
                "v7_final_receipt_sha256": (
                    state["v7_authority"]["final_receipt"]["sha256"]
                ),
                "study1_objective_gate_authority_sha256": state[
                    "v7_authority"
                ]["objective_gate_authority"]["sha256"],
                "study1_dedupe_sha256": job["study1_dedupe_sha256"],
                "package_control_plane_sha256": state[
                    "package_control_plane"
                ]["sha256"],
            },
            "output_artifacts": manifest_output_bindings,
        }
    )
    atomic_write_json(
        artifact_paths["execution_manifest"], execution_manifest
    )
    output_bindings["execution_manifest"] = {
        "path": job["artifact_paths"]["execution_manifest"],
        "sha256": sha256_file(artifact_paths["execution_manifest"]),
        "size_bytes": artifact_paths["execution_manifest"].stat().st_size,
        "canonical_sha256": execution_manifest["sha256"],
        "status": "produced",
    }
    immutable_after = _immutable_bundle_hashes(state)
    if immutable_after != immutable_before:
        raise PackageContractError(
            "Protocol/template or immutable bundle authority mutated during run."
        )
    worker_receipt = digested(
        {
            "schema": WORKER_RECEIPT_SCHEMA,
            "package_id": PACKAGE_ID,
            "execution_id": execution_id,
            "status": "completed",
            "mode": args.mode,
            "started_utc": started_at,
            "finished_utc": finished_at,
            "execution_plan_sha256": state["plan"]["sha256"],
            "job_spec_sha256": job["sha256"],
            "authorization_sha256": state["authorization"]["sha256"],
            "package_control_plane_sha256": state[
                "package_control_plane"
            ]["sha256"],
            "source_archive_sha256": job["source_archive"]["sha256"],
            "remote_image_sha256": (
                args.verified_image_sha256
                if args.mode == "execute"
                else None
            ),
            "immutable_bundle_hashes_before": immutable_before,
            "immutable_bundle_hashes_after": immutable_after,
            "artifacts": output_bindings,
            "g11_replay_diagnostic": g11_replay_diagnostic,
            "objective_gates": objective_gate_receipt,
        }
    )
    atomic_write_json(state["worker_receipt_path"], worker_receipt)
    return worker_receipt


def _failure_receipt(
    state: Mapping[str, Any] | None,
    *,
    args: argparse.Namespace,
    exc: BaseException,
) -> None:
    if state is None or args.mode == "validate-only":
        return
    path = state["worker_receipt_path"]
    if path.exists():
        return
    payload = digested(
        {
            "schema": WORKER_RECEIPT_SCHEMA,
            "package_id": PACKAGE_ID,
            "execution_id": state["job"]["execution_id"],
            "status": "failed",
            "mode": args.mode,
            "failed_utc": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "execution_plan_sha256": state["plan"]["sha256"],
            "job_spec_sha256": state["job"]["sha256"],
            "authorization_sha256": state["authorization"]["sha256"],
            "package_control_plane_sha256": state[
                "package_control_plane"
            ]["sha256"],
        }
    )
    atomic_write_json(path, payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("validate-only", "execute", "local-packaged-smoke"),
        default="validate-only",
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--job-spec", type=Path, required=True)
    parser.add_argument("--package-manifest", type=Path, required=True)
    parser.add_argument("--authorization-receipt", type=Path, required=True)
    parser.add_argument("--v7-final-receipt", type=Path, required=True)
    parser.add_argument(
        "--objective-gate-authority", type=Path, required=True
    )
    parser.add_argument("--execution-plan", type=Path, required=True)
    parser.add_argument("--source-archive-sha256")
    parser.add_argument("--verified-image-sha256")
    return parser


def main() -> int:
    args = _parser().parse_args()
    state: dict[str, Any] | None = None
    try:
        state = _validate_control_plane(args)
        if args.mode == "validate-only":
            payload = {
                "status": "passed",
                "mode": "validate-only",
                "package_id": PACKAGE_ID,
                "execution_id": state["job"]["execution_id"],
                "execution_plan_sha256": state["plan"]["sha256"],
                "authorization_sha256": state["authorization"]["sha256"],
                "writes_performed": False,
            }
        else:
            payload = _execute(state, args)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        try:
            _failure_receipt(state, args=args, exc=exc)
        except Exception as receipt_exc:
            print(
                f"ERROR: failure receipt could not be written: {receipt_exc}",
                file=sys.stderr,
            )
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
