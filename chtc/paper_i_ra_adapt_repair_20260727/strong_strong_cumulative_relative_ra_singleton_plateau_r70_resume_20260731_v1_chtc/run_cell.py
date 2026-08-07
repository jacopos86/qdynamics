#!/usr/bin/env python3
"""Execute or deeply preflight the sealed cumulative-relative r70 row."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
from types import ModuleType
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "off"
os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from package_contract import (  # noqa: E402
    ACTIVE_GRADIENT_POLICY,
    ALLOWED_OPERATIONAL_SOURCE_DELTAS,
    BUNDLED_SOURCE_PROTOCOL,
    CAMPAIGN_ID,
    CANDIDATE_REPRESENTATION,
    DERIVED_PROTOCOL_CHANGED_PATHS,
    EXECUTION_ID,
    PACKAGE_ID,
    PLATEAU_COMPARISON,
    PLATEAU_RATIO_THRESHOLD,
    PLATEAU_TRIGGER,
    RESOURCE_WEIGHTING_SCOPE,
    ROUTE_CONTRACT_SHA256,
    SOURCE_CELL_ID,
    SOURCE_HORIZON,
    SOURCE_PROTOCOL_CANONICAL_SHA256,
    TARGET_HORIZON,
    PackageContractError,
    canonical_json_bytes,
    digested,
    load_json,
    safe_relative_path,
    scalar_differences,
    sha256_file,
    verify_self_digest,
)


PACKAGE_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_cumulative_relative_r70_package_manifest_v1"
)
JOB_SCHEMA = "paper_i_ra_adapt_cumulative_relative_r70_job_v1"
AUTHORIZATION_SCHEMA = (
    "paper_i_ra_adapt_cumulative_relative_r70_execution_authorization_v1"
)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PackageContractError(f"{label} must be a mapping.")
    return value


def _sequence(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PackageContractError(f"{label} must be a list.")
    return value


def _package_path(value: Any, *, label: str) -> Path:
    relative = safe_relative_path(value, label=label)
    path = PACKAGE_DIR / relative
    try:
        path.resolve().relative_to(PACKAGE_DIR.resolve())
    except ValueError as exc:
        raise PackageContractError(f"{label} escaped the package.") from exc
    return path


def _verify_binding(
    binding: Mapping[str, Any],
    *,
    label: str,
    canonical: bool = False,
) -> tuple[Path, dict[str, Any] | None]:
    path = _package_path(binding.get("path"), label=f"{label} path")
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(binding.get("size_bytes", -1))
        or sha256_file(path) != binding.get("sha256")
    ):
        raise PackageContractError(f"{label} byte binding drifted.")
    if not canonical:
        return path, None
    payload = load_json(path, label=label)
    observed = verify_self_digest(payload, label=label)
    if observed != binding.get("canonical_sha256"):
        raise PackageContractError(f"{label} canonical binding drifted.")
    return path, payload


def _load_job(job_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = PACKAGE_DIR / "package_manifest.json"
    manifest = load_json(manifest_path, label="package manifest")
    verify_self_digest(manifest, label="package manifest")
    if (
        manifest.get("schema") != PACKAGE_MANIFEST_SCHEMA
        or manifest.get("status") != "passed_inert_one_row"
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("execution_id") != EXECUTION_ID
        or manifest.get("row_count") != 1
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_authorized") is not False
        or manifest.get("submission_ready") is not False
        or manifest.get("submit_descriptor_present") is not False
        or manifest.get("submitted") is not False
    ):
        raise PackageContractError("Inert package manifest drifted.")
    binding = _mapping(manifest.get("job"), label="job binding")
    expected = (PACKAGE_DIR / "job.json").resolve()
    path, payload = _verify_binding(binding, label="job", canonical=True)
    assert payload is not None
    job = payload
    if (
        job_path.resolve() != expected
        or path.resolve() != expected
        or job.get("schema") != JOB_SCHEMA
        or job.get("package_id") != PACKAGE_ID
        or job.get("campaign_id") != CAMPAIGN_ID
        or job.get("execution_id") != EXECUTION_ID
        or job.get("source_cell_id") != SOURCE_CELL_ID
        or job.get("execution_mode")
        != "authenticated_resume_50_to_70"
        or job.get("source_horizon") != SOURCE_HORIZON
        or job.get("target_horizon") != TARGET_HORIZON
        or job.get("source_protocol_sha256")
        != SOURCE_PROTOCOL_CANONICAL_SHA256
        or job.get("source_protocol") != manifest.get("source_protocol")
        or job.get("active_gradient_policy") != ACTIVE_GRADIENT_POLICY
        or job.get("resource_weighting_scope")
        != RESOURCE_WEIGHTING_SCOPE
        or job.get("candidate_representation")
        != CANDIDATE_REPRESENTATION
        or job.get("plateau_cumulative_decrease_ratio_threshold")
        != PLATEAU_RATIO_THRESHOLD
        or job.get("plateau_threshold_comparison")
        != PLATEAU_COMPARISON
        or job.get("plateau_trigger_source") != PLATEAU_TRIGGER
        or job.get("only_scientific_change")
        != "maximum_controller_rounds_50_to_70"
        or job.get("non_swept_settings_diff") != []
        or job.get("execution_authorized") is not False
        or job.get("submission_authorized") is not False
        or job.get("submitted") is not False
    ):
        raise PackageContractError("One-row job contract drifted.")
    for key, canonical in (
        ("source_protocol", True),
        ("derived_protocol", True),
        ("source_archive", False),
        ("source_archive_manifest", True),
        ("source_delta_receipt", True),
    ):
        _verify_binding(
            _mapping(job.get(key), label=f"{key} binding"),
            label=key,
            canonical=canonical,
        )
    delta_path, delta = _verify_binding(
        _mapping(job["source_delta_receipt"], label="source delta"),
        label="source delta",
        canonical=True,
    )
    assert delta is not None
    changed = _sequence(delta.get("changed_members"), label="source deltas")
    if (
        delta_path != PACKAGE_DIR / "source/source_delta_receipt.json"
        or delta.get("status") != "passed_operational_only"
        or {row.get("path") for row in changed}
        != set(ALLOWED_OPERATIONAL_SOURCE_DELTAS)
        or delta.get("scientific_settings_changed") != []
        or delta.get("route_contract_changed") is not False
        or any(
            row.get("scientific_protocol_change") is not False
            or row.get("controller_semantics_change") is not False
            or row.get("accepted_state_change") is not False
            for row in changed
        )
    ):
        raise PackageContractError("Operational-only source delta drifted.")
    return job, manifest


def _validate_authorization(
    path: Path,
    *,
    job: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    authorization = load_json(path, label="execution authorization")
    verify_self_digest(authorization, label="execution authorization")
    if (
        authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("package_id") != PACKAGE_ID
        or authorization.get("campaign_id") != CAMPAIGN_ID
        or authorization.get("execution_id") != EXECUTION_ID
        or authorization.get("job_spec_sha256") != job.get("sha256")
        or authorization.get("package_manifest_sha256")
        != manifest.get("sha256")
        or authorization.get("derived_protocol_sha256")
        != job.get("derived_protocol", {}).get("canonical_sha256")
        or authorization.get("source_archive_sha256")
        != job.get("source_archive", {}).get("sha256")
        or authorization.get("resume_archive_sha256")
        != job.get("resume_input", {}).get("archive", {}).get("sha256")
        or authorization.get("scope") != "single_cell_execution_only"
        or authorization.get("authorization_kind")
        != "explicit_user_execution_authority"
        or authorization.get("execution_authorized") is not True
        or authorization.get("submission_authorized") is not False
        or authorization.get("paper_evidence_adoption_authorized") is not False
    ):
        raise PackageContractError(
            "Execution authorization is absent, stale, or overbroad."
        )
    return authorization


def _extract_bound_tar(
    *,
    archive_path: Path,
    members: Mapping[str, Mapping[str, Any]],
    destination: Path,
    label: str,
) -> None:
    observed: set[str] = set()
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            binding = members.get(member.name)
            relative = safe_relative_path(
                member.name, label=f"{label} member"
            )
            if (
                binding is None
                or member.name in observed
                or not member.isfile()
                or member.issym()
                or member.islnk()
                or member.size != int(binding.get("size_bytes", -1))
            ):
                raise PackageContractError(
                    f"Unexpected {label} member: {member.name}"
                )
            source = archive.extractfile(member)
            if source is None:
                raise PackageContractError(
                    f"Unreadable {label} member: {member.name}"
                )
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            size = 0
            with target.open("xb") as output:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(block)
                    digest.update(block)
                    size += len(block)
                output.flush()
                os.fsync(output.fileno())
            if (
                size != member.size
                or digest.hexdigest() != binding.get("sha256")
            ):
                raise PackageContractError(
                    f"Extracted {label} member drifted: {member.name}"
                )
            observed.add(member.name)
    if observed != set(members):
        raise PackageContractError(f"{label} member closure drifted.")


def _extract_source(job: Mapping[str, Any], destination: Path) -> None:
    archive_path, _ = _verify_binding(
        _mapping(job["source_archive"], label="source archive"),
        label="source archive",
    )
    _manifest_path, manifest = _verify_binding(
        _mapping(job["source_archive_manifest"], label="source manifest"),
        label="source manifest",
        canonical=True,
    )
    assert manifest is not None
    rows = _sequence(manifest.get("members"), label="source members")
    members = {
        str(row["path"]): row for row in rows if isinstance(row, Mapping)
    }
    if (
        manifest.get("archive") != job.get("source_archive")
        or len(members) != len(rows)
        or len(rows) != int(manifest.get("member_count", -1))
    ):
        raise PackageContractError("Source archive manifest drifted.")
    _extract_bound_tar(
        archive_path=archive_path,
        members=members,
        destination=destination,
        label="source archive",
    )


def _extract_resume(job: Mapping[str, Any], destination: Path) -> Path:
    resume = _mapping(job.get("resume_input"), label="resume input")
    archive_path, _ = _verify_binding(
        _mapping(resume.get("archive"), label="resume archive"),
        label="resume archive",
    )
    rows = _sequence(resume.get("members"), label="resume members")
    members = {
        str(row["path"]): row for row in rows if isinstance(row, Mapping)
    }
    checkpoint_relative = safe_relative_path(
        resume.get("checkpoint_path"), label="resume checkpoint"
    )
    if (
        len(members) != len(rows)
        or len(rows) != int(resume.get("member_count", -1))
        or len(members) != 3
        or resume.get("pointer_closed") is not True
        or checkpoint_relative.as_posix() not in members
        or members[checkpoint_relative.as_posix()].get("sha256")
        != resume.get("checkpoint_sha256")
    ):
        raise PackageContractError("Resume archive contract drifted.")
    _extract_bound_tar(
        archive_path=archive_path,
        members=members,
        destination=destination,
        label="resume archive",
    )
    checkpoint = destination / checkpoint_relative
    if sha256_file(checkpoint) != resume["checkpoint_sha256"]:
        raise PackageContractError("Resume checkpoint drifted after extraction.")
    return checkpoint


def _module_is_under(module: ModuleType, root: Path) -> bool:
    origin = getattr(module, "__file__", None)
    if origin is not None:
        try:
            Path(origin).resolve().relative_to(root)
            return True
        except ValueError:
            return False
    locations = getattr(module, "__path__", None)
    if locations is None:
        return True
    for location in locations:
        try:
            Path(location).resolve().relative_to(root)
        except ValueError:
            return False
    return True


def _activate_source_root(source_root: Path) -> None:
    root = source_root.resolve()
    drifted: list[str] = []
    for name, module in tuple(sys.modules.items()):
        if not (
            name == "pipelines"
            or name.startswith("pipelines.")
            or name == "src"
            or name.startswith("src.")
        ):
            continue
        if not _module_is_under(module, root):
            drifted.append(name)
    if drifted:
        raise PackageContractError(
            "Project modules were imported before source-lock activation: "
            + ", ".join(sorted(drifted))
        )
    sys.path.insert(0, str(root))


def _load_protocols(
    *, job: Mapping[str, Any], source_root: Path
) -> tuple[Any, Any, Any]:
    from chtc.paper_i_ra_adapt_repair_20260727 import (
        run_local_cumulative_plateau_pair_20260731 as base,
    )
    from pipelines.static_adapt.ra_adapt.contracts import (
        _attach_validated_bundle_protocol_authority,
        _mint_bundle_protocol_materialization_authority,
        resolved_ra_adapt_protocol_from_mapping,
    )

    if Path(base.REPO_ROOT).resolve() != source_root.resolve():
        raise PackageContractError("Source diagnostic helper escaped archive.")
    source_path, _ = _verify_binding(
        _mapping(job["source_protocol"], label="source protocol"),
        label="source protocol",
        canonical=True,
    )
    if source_path != PACKAGE_DIR / BUNDLED_SOURCE_PROTOCOL:
        raise PackageContractError("Bundled source protocol path drifted.")
    source_payload = load_json(source_path, label="source protocol")
    source = resolved_ra_adapt_protocol_from_mapping(source_payload)
    derived_path, _ = _verify_binding(
        _mapping(job["derived_protocol"], label="derived protocol"),
        label="derived protocol",
        canonical=True,
    )
    derived_payload = load_json(derived_path, label="derived protocol")
    derived = resolved_ra_adapt_protocol_from_mapping(derived_payload)
    if source.bundle_materialization is None:
        raise PackageContractError("Source protocol lost materialization.")
    authority = _mint_bundle_protocol_materialization_authority(
        derived.bundle_materialization,
        source_lock_refs=derived.source_locks,
        protocol_sha256=derived.sha256,
    )
    derived = _attach_validated_bundle_protocol_authority(derived, authority)
    changed = sorted(
        ".".join(str(component) for component in path)
        for path, _before, _after in scalar_differences(
            source.to_dict(), derived.to_dict()
        )
    )
    invariants = derived.route_contract["semantic_invariants"]
    if (
        source.sha256 != SOURCE_PROTOCOL_CANONICAL_SHA256
        or changed != list(DERIVED_PROTOCOL_CHANGED_PATHS)
        or int(source.horizon) != SOURCE_HORIZON
        or int(derived.horizon) != TARGET_HORIZON
        or derived.active_gradient_policy != ACTIVE_GRADIENT_POLICY
        or derived.resource_weighting_scope != RESOURCE_WEIGHTING_SCOPE
        or derived.candidate_representation != CANDIDATE_REPRESENTATION
        or derived.route_contract != source.route_contract
        or derived.route_contract["sha256"] != ROUTE_CONTRACT_SHA256
        or invariants["plateau_cumulative_decrease_ratio_threshold"]
        != PLATEAU_RATIO_THRESHOLD
        or invariants["plateau_threshold_comparison"]
        != PLATEAU_COMPARISON
        or invariants["plateau_trigger_source"] != PLATEAU_TRIGGER
    ):
        raise PackageContractError(
            f"Source-to-r70 protocol delta drifted: {changed}"
        )
    problem = base._problem_from_receipt(derived.problem)
    return source, derived, problem


def _validate_resume(
    *, checkpoint: Path, derived: Any, problem: Any, expected_round: int
) -> Any:
    from pipelines.static_adapt.sr_snake import AcceptedStateResume
    from pipelines.static_adapt.sr_snake._resume import (
        load_canonical_accepted_state_resume,
    )

    hydration = load_canonical_accepted_state_resume(
        AcceptedStateResume(
            checkpoint_path=checkpoint,
            checkpoint_sha256=sha256_file(checkpoint),
        ),
        expected_problem=problem,
        expected_route_profile=str(derived.route_contract["route_profile"]),
        expected_route_contract_sha256=str(
            derived.route_contract["sha256"]
        ),
    )
    if int(hydration.controller_round) != int(expected_round):
        raise PackageContractError("Resume hydrated at the wrong round.")
    return hydration


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        encoder = json.JSONEncoder(
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        for chunk in encoder.iterencode(dict(payload)):
            stream.write(chunk.encode("ascii"))
        stream.write(b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _prefix_rows(job: Mapping[str, Any]) -> list[Any]:
    lineage = _mapping(job.get("lineage"), label="lineage")
    binding = _mapping(lineage.get("r50_result"), label="r50 result")
    path, _ = _verify_binding(binding, label="r50 result")
    payload = load_json(path, label="r50 result")
    run = _mapping(payload.get("run"), label="r50 run")
    rows = _sequence(run.get("accepted_trajectory"), label="r50 trajectory")
    if len(rows) != SOURCE_HORIZON:
        raise PackageContractError("Bound r50 trajectory length drifted.")
    return rows


def preflight(job_path: Path) -> dict[str, Any]:
    job, manifest = _load_job(job_path)
    with tempfile.TemporaryDirectory(
        prefix="paper-i-cumulative-r70-preflight."
    ) as raw:
        root = Path(raw)
        source_root = root / "source"
        resume_root = root / "resume"
        _extract_source(job, source_root)
        checkpoint = _extract_resume(job, resume_root)
        _activate_source_root(source_root)
        source, derived, problem = _load_protocols(
            job=job, source_root=source_root
        )
        hydration = _validate_resume(
            checkpoint=checkpoint,
            derived=derived,
            problem=problem,
            expected_round=SOURCE_HORIZON,
        )
    return digested(
        {
            "schema": (
                "paper_i_ra_adapt_cumulative_relative_r70_"
                "worker_preflight_v1"
            ),
            "status": "passed",
            "package_manifest_sha256": manifest["sha256"],
            "job_spec_sha256": job["sha256"],
            "execution_id": EXECUTION_ID,
            "source_protocol_sha256": source.sha256,
            "derived_protocol_sha256": derived.sha256,
            "resume_checkpoint_sha256": job["resume_input"][
                "checkpoint_sha256"
            ],
            "resume_controller_round": int(hydration.controller_round),
            "source_archive_import_isolated": True,
            "only_scientific_change": (
                "maximum_controller_rounds_50_to_70"
            ),
        }
    )


def run_cell(
    *,
    job_path: Path,
    authorization_path: Path,
    output_dir: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    job, manifest = _load_job(job_path)
    authorization = _validate_authorization(
        authorization_path, job=job, manifest=manifest
    )
    if (
        output_dir.exists()
        or output_dir.is_symlink()
        or receipt_path.exists()
        or receipt_path.is_symlink()
    ):
        raise PackageContractError("Worker destination already exists.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{EXECUTION_ID}.", dir=output_dir.parent
    ) as raw:
        temporary = Path(raw)
        source_root = temporary / "source"
        resume_root = temporary / "resume"
        staging = temporary / "artifacts"
        staging.mkdir()
        _extract_source(job, source_root)
        checkpoint = _extract_resume(job, resume_root)
        _activate_source_root(source_root)
        source, derived, problem = _load_protocols(
            job=job, source_root=source_root
        )
        from pipelines.static_adapt.ra_adapt import (
            RAAdaptOperationalControls,
            run_ra_adapt,
        )
        from pipelines.static_adapt.sr_snake import (
            AcceptedStateResume,
            CheckpointObservation,
            EstimatorLedgerObservation,
            SRObservationPolicy,
        )

        controls = RAAdaptOperationalControls(
            maximum_controller_rounds=TARGET_HORIZON,
            resume=AcceptedStateResume(
                checkpoint_path=checkpoint,
                checkpoint_sha256=job["resume_input"]["checkpoint_sha256"],
            ),
            observation=SRObservationPolicy(
                checkpoint=CheckpointObservation(
                    path=staging / "checkpoint.json",
                    every_controller_rounds=1,
                    keep_history_tail=100,
                ),
                estimator_ledger=EstimatorLedgerObservation(
                    path=staging / "estimator_ledger.json"
                ),
                resource_rounds=(TARGET_HORIZON,),
            ),
        )
        result = run_ra_adapt(problem, derived, operational_controls=controls)
        result_payload = result.to_dict()
        resumed_run = _mapping(result_payload.get("run"), label="result run")
        rows = _sequence(
            resumed_run.get("accepted_trajectory"),
            label="accepted trajectory",
        )
        if (
            not SOURCE_HORIZON <= len(rows) <= TARGET_HORIZON
            or rows[:SOURCE_HORIZON] != _prefix_rows(job)
            or result.protocol.sha256 != derived.sha256
        ):
            raise PackageContractError(
                "Continuation trajectory or protocol closure failed."
            )
        _write_json(staging / "result.json", result_payload)
        if result.run.paper_i_summary is None:
            raise PackageContractError("Continuation returned no summary.")
        _write_json(
            staging / "paper_i_summary.json",
            result.run.paper_i_summary.to_dict(),
        )
        output_checkpoint = staging / "checkpoint.json"
        output_ledger = staging / "estimator_ledger.json"
        if not output_checkpoint.is_file() or not output_ledger.is_file():
            raise PackageContractError("Final observation artifacts are absent.")
        final_hydration = _validate_resume(
            checkpoint=output_checkpoint,
            derived=derived,
            problem=problem,
            expected_round=len(rows),
        )
        if int(final_hydration.controller_round) != len(rows):
            raise PackageContractError(
                "Final checkpoint does not match the result horizon."
            )
        execution_manifest = digested(
            {
                "schema": (
                    "paper_i_ra_adapt_cumulative_relative_r70_"
                    "execution_manifest_v1"
                ),
                "status": "passed",
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "execution_id": EXECUTION_ID,
                "job_spec_sha256": job["sha256"],
                "authorization_sha256": authorization["sha256"],
                "source_protocol_sha256": source.sha256,
                "derived_protocol_sha256": derived.sha256,
                "source_horizon": SOURCE_HORIZON,
                "target_horizon": TARGET_HORIZON,
                "controller_rounds_available": len(rows),
                "prefix_preserved": True,
                "only_scientific_change": (
                    "maximum_controller_rounds_50_to_70"
                ),
                "changed_protocol_paths": list(
                    DERIVED_PROTOCOL_CHANGED_PATHS
                ),
                "scientific_settings_changed_by_source_delta": [],
                "final_checkpoint_canonical_resume_validation": "passed",
                "output_payloads": {
                    path.name: {
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for path in sorted(staging.iterdir())
                    if path.is_file()
                },
            }
        )
        _write_json(
            staging / "execution_manifest.json", execution_manifest
        )
        os.rename(staging, output_dir)
    receipt = digested(
        {
            "schema": (
                "paper_i_ra_adapt_cumulative_relative_r70_"
                "worker_receipt_v1"
            ),
            "status": "passed",
            "package_id": PACKAGE_ID,
            "execution_id": EXECUTION_ID,
            "job_spec_sha256": job["sha256"],
            "authorization_sha256": authorization["sha256"],
            "execution_manifest_sha256": execution_manifest["sha256"],
            "output_dir": output_dir.as_posix(),
            "output_files": [
                {
                    "path": path.name,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(output_dir.iterdir())
                if path.is_file()
            ],
        }
    )
    _write_json(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--execution-authorization", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    try:
        if args.preflight:
            if any(
                value is not None
                for value in (
                    args.execution_authorization,
                    args.output_dir,
                    args.receipt,
                )
            ):
                raise PackageContractError(
                    "Preflight does not accept execution destinations."
                )
            payload = preflight(args.job.resolve())
        else:
            if any(
                value is None
                for value in (
                    args.execution_authorization,
                    args.output_dir,
                    args.receipt,
                )
            ):
                raise PackageContractError(
                    "Execution requires authorization, output, and receipt."
                )
            payload = run_cell(
                job_path=args.job.resolve(),
                authorization_path=args.execution_authorization.resolve(),
                output_dir=args.output_dir.resolve(),
                receipt_path=args.receipt.resolve(),
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(payload).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
