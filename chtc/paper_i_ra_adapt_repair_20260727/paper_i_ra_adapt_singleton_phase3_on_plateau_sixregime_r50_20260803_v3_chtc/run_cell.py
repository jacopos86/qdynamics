#!/usr/bin/env python3
"""Preflight or execute one sealed Phase-III-on-plateau RA r50 row."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import tempfile
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
    ALGORITHM_ID,
    AUTHORIZATION_SCHEMA,
    CAMPAIGN_ID,
    CANDIDATE_REPRESENTATION,
    JOB_SCHEMA,
    PACKAGE_ID,
    PACKAGE_MANIFEST_SCHEMA,
    PLATEAU_CALIBRATION,
    PLATEAU_COMPARISON,
    PLATEAU_PRIOR_MEAN_RATIO_THRESHOLD,
    PLATEAU_TRIGGER,
    PARENT_ROUTE_CONTRACT_SHA256,
    RESOURCE_WEIGHTING_SCOPE,
    ROUTE_CONTRACT_SHA256,
    ROUTE_PROFILE,
    TARGET_HORIZON,
    PackageContractError,
    canonical_json_bytes,
    digested,
    expected_execution_ids,
    load_json,
    safe_relative_path,
    sha256_file,
    verify_self_digest,
)


_ESTIMATOR_CHECKPOINT_SIDECAR = re.compile(
    r"^checkpoint\.estimator_call_ledger_checkpoint\.([0-9a-f]{16})\.json$"
)
_SINGLETON_RESUME_SIDECAR = re.compile(
    r"^checkpoint\.verified_singleton_resume\.([0-9a-f]{16})\.json$"
)
_REQUIRED_SCIENCE_PAYLOADS = frozenset(
    {
        "checkpoint.json",
        "estimator_ledger.json",
        "paper_i_summary.json",
        "result.json",
    }
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
    path = PACKAGE_DIR / safe_relative_path(value, label=label)
    try:
        path.resolve().relative_to(PACKAGE_DIR.resolve())
    except ValueError as exc:
        raise PackageContractError(f"{label} escaped the package.") from exc
    return path


def _verify_binding(
    raw: Any,
    *,
    label: str,
    canonical: bool = False,
) -> tuple[Path, dict[str, Any] | None]:
    row = _mapping(raw, label=f"{label} binding")
    path = _package_path(row.get("path"), label=f"{label} path")
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(row.get("size_bytes", -1))
        or sha256_file(path) != row.get("sha256")
    ):
        raise PackageContractError(f"{label} byte binding drifted.")
    if not canonical:
        return path, None
    payload = load_json(path, label=label)
    observed = verify_self_digest(payload, label=label)
    if observed != row.get("canonical_sha256"):
        raise PackageContractError(f"{label} canonical binding drifted.")
    return path, payload


def _load_closed_job(
    job_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = load_json(PACKAGE_DIR / "package_manifest.json", label="package manifest")
    verify_self_digest(manifest, label="package manifest")
    expected_ids = list(expected_execution_ids())
    if (
        manifest.get("schema") != PACKAGE_MANIFEST_SCHEMA
        or manifest.get("status") != "passed_inert_six_rows"
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("row_count") != 6
        or manifest.get("execution_ids") != expected_ids
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_authorized") is not False
        or manifest.get("submission_ready") is not False
        or manifest.get("submit_descriptor_present") is not False
        or manifest.get("submitted") is not False
    ):
        raise PackageContractError("Inert six-row package manifest drifted.")

    resolved_job = job_path.resolve()
    job_rows = _sequence(manifest.get("jobs"), label="job bindings")
    matching = [
        row
        for row in job_rows
        if isinstance(row, Mapping)
        and _package_path(row.get("path"), label="job path").resolve()
        == resolved_job
    ]
    if len(matching) != 1:
        raise PackageContractError("Requested job is outside the six-row closure.")
    bound_job_path, job_payload = _verify_binding(
        matching[0], label="job", canonical=True
    )
    assert job_payload is not None
    job = job_payload
    execution_id = str(job.get("execution_id", ""))
    if (
        bound_job_path.resolve() != resolved_job
        or execution_id not in expected_ids
        or job.get("schema") != JOB_SCHEMA
        or job.get("package_id") != PACKAGE_ID
        or job.get("campaign_id") != CAMPAIGN_ID
        or job.get("execution_mode") != "fresh_0_to_50"
        or job.get("target_horizon") != TARGET_HORIZON
        or job.get("active_gradient_policy") != ACTIVE_GRADIENT_POLICY
        or job.get("resource_weighting_scope") != RESOURCE_WEIGHTING_SCOPE
        or job.get("candidate_representation") != CANDIDATE_REPRESENTATION
        or job.get("insertion_policy") != "plateau_commutation"
        or job.get("plateau_prior_mean_decrease_ratio_threshold")
        != PLATEAU_PRIOR_MEAN_RATIO_THRESHOLD
        or job.get("plateau_threshold_comparison") != PLATEAU_COMPARISON
        or job.get("plateau_trigger_source") != PLATEAU_TRIGGER
        or job.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
        or job.get("fresh_start_contract")
        != {"kind": "fresh_start", "resume_archive": None, "source_checkpoint": None}
        or job.get("execution_authorized") is not False
        or job.get("submission_authorized") is not False
        or job.get("submitted") is not False
    ):
        raise PackageContractError("Fresh r50 job contract drifted.")

    protocol_rows = _sequence(manifest.get("protocols"), label="protocol bindings")
    protocol_matches = [
        row
        for row in protocol_rows
        if isinstance(row, Mapping) and row.get("execution_id") == execution_id
    ]
    if len(protocol_matches) != 1:
        raise PackageContractError("Job protocol binding is not unique.")
    protocol_path, protocol_payload = _verify_binding(
        protocol_matches[0], label="protocol", canonical=True
    )
    assert protocol_payload is not None
    if (
        protocol_path != _package_path(job.get("protocol_path"), label="protocol path")
        or protocol_matches[0].get("canonical_sha256") != job.get("protocol_sha256")
        or protocol_matches[0].get("sha256") != job.get("protocol_file_sha256")
    ):
        raise PackageContractError("Job-to-protocol binding drifted.")

    _bundle_path, protocol_bundle = _verify_binding(
        manifest.get("protocol_bundle_manifest"),
        label="protocol bundle manifest",
        canonical=True,
    )
    _locks_path, source_locks = _verify_binding(
        manifest.get("source_locks_snapshot"),
        label="source locks snapshot",
        canonical=True,
    )
    assert protocol_bundle is not None and source_locks is not None
    if (
        protocol_bundle.get("sha256")
        != job.get("protocol_bundle_manifest_sha256")
        or source_locks.get("sha256") != job.get("source_locks_snapshot_sha256")
        or source_locks.get("implementation_sources", {}).get("sha256")
        != job.get("implementation_source_inventory_sha256")
    ):
        raise PackageContractError("Job source authority drifted.")
    return job, manifest, protocol_payload, source_locks


def _validate_authorization(
    path: Path,
    *,
    job: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    authority = load_json(path, label="execution authorization")
    verify_self_digest(authority, label="execution authorization")
    source_binding = _mapping(manifest.get("source_archive"), label="source archive")
    if (
        authority.get("schema") != AUTHORIZATION_SCHEMA
        or authority.get("package_id") != PACKAGE_ID
        or authority.get("campaign_id") != CAMPAIGN_ID
        or authority.get("execution_id") != job.get("execution_id")
        or authority.get("job_spec_sha256") != job.get("sha256")
        or authority.get("package_manifest_sha256") != manifest.get("sha256")
        or authority.get("protocol_sha256") != job.get("protocol_sha256")
        or authority.get("protocol_file_sha256") != job.get("protocol_file_sha256")
        or authority.get("source_archive_sha256") != source_binding.get("sha256")
        or authority.get("scope") != "single_cell_chtc_execution_only"
        or authority.get("authorization_kind")
        != "explicit_user_execution_and_submission_authority"
        or authority.get("execution_authorized") is not True
        or authority.get("submission_authorized") is not True
        or authority.get("paper_evidence_adoption_authorized") is not False
    ):
        raise PackageContractError("Execution authorization is stale or overbroad.")
    return authority


def _extract_source(
    *, manifest: Mapping[str, Any], source_locks: Mapping[str, Any], destination: Path
) -> None:
    archive_path, _ = _verify_binding(
        manifest.get("source_archive"), label="source archive"
    )
    _source_manifest_path, source_manifest = _verify_binding(
        manifest.get("source_archive_manifest"),
        label="source archive manifest",
        canonical=True,
    )
    assert source_manifest is not None
    archive_binding = _mapping(source_manifest.get("archive"), label="archive")
    if (
        archive_binding != manifest.get("source_archive")
        or source_manifest.get("status") != "passed"
        or source_manifest.get("no_ambient_repo_imports") is not True
    ):
        raise PackageContractError("Source archive authority drifted.")
    rows = _sequence(source_manifest.get("members"), label="source members")
    members = {
        safe_relative_path(row.get("path"), label="source member").as_posix(): row
        for row in rows
        if isinstance(row, Mapping)
    }
    globals_raw = _mapping(source_locks.get("global_sources"), label="global sources")
    expected_globals = sorted(str(row["path"]) for row in globals_raw.values())
    if (
        len(members) != len(rows)
        or len(rows) != int(source_manifest.get("member_count", -1))
        or source_manifest.get("global_source_paths") != expected_globals
        or source_manifest.get("runtime_path_dependencies") != ["requirements.txt"]
        or any(path not in members for path in (*expected_globals, "requirements.txt"))
    ):
        raise PackageContractError("Source archive member closure drifted.")

    destination.mkdir(parents=True, exist_ok=False)
    observed: set[str] = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            relative = safe_relative_path(member.name, label="tar member").as_posix()
            row = members.get(relative)
            if (
                row is None
                or relative in observed
                or not member.isfile()
                or member.issym()
                or member.islnk()
                or member.size != int(row.get("size_bytes", -1))
            ):
                raise PackageContractError(f"Unsafe source member: {relative}")
            source = archive.extractfile(member)
            if source is None:
                raise PackageContractError(f"Unreadable source member: {relative}")
            target = destination / Path(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            size = 0
            with target.open("xb") as output:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(block)
                    digest.update(block)
                    size += len(block)
            if size != member.size or digest.hexdigest() != row.get("sha256"):
                raise PackageContractError(f"Extracted source drifted: {relative}")
            observed.add(relative)
    if observed != set(members):
        raise PackageContractError("Extracted source member closure drifted.")


def _activate_source_root(source_root: Path) -> None:
    root = source_root.resolve()
    for name in list(sys.modules):
        if name == "pipelines" or name.startswith("pipelines.") or name == "src" or name.startswith("src."):
            del sys.modules[name]
    sys.path[:] = [
        item
        for item in sys.path
        if not (
            (Path(item or ".").resolve() / "pipelines").exists()
            or (Path(item or ".").resolve() / "src").exists()
        )
    ]
    sys.path.insert(0, root.as_posix())
    importlib.invalidate_caches()
    module = importlib.import_module("pipelines.static_adapt.ra_adapt")
    try:
        Path(str(module.__file__)).resolve().relative_to(root)
    except ValueError as exc:
        raise PackageContractError("Runtime implementation escaped the archive.") from exc


def _reassert_source_root(source_root: Path) -> None:
    """Restore locked-source precedence without duplicating module identities."""

    root = source_root.resolve()
    retained: list[str] = []
    for item in sys.path:
        resolved = Path(item or ".").resolve()
        if resolved == root:
            continue
        if (resolved / "pipelines").exists() or (resolved / "src").exists():
            continue
        retained.append(item)
    sys.path[:] = [root.as_posix(), *retained]
    importlib.invalidate_caches()

    for name, module in list(sys.modules.items()):
        if not (
            name == "pipelines"
            or name.startswith("pipelines.")
            or name == "src"
            or name.startswith("src.")
        ):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        try:
            Path(str(module_file)).resolve().relative_to(root)
        except ValueError as exc:
            raise PackageContractError(
                f"Loaded runtime module escaped the archive: {name}"
            ) from exc

    reporting_spec = importlib.util.find_spec(
        "pipelines.reporting.paper_i_run_summary"
    )
    if reporting_spec is None or reporting_spec.origin is None:
        raise PackageContractError("Locked reporting module is unavailable.")
    try:
        Path(reporting_spec.origin).resolve().relative_to(root)
    except ValueError as exc:
        raise PackageContractError(
            "Lazy Paper-I reporting import escaped the archive."
        ) from exc


def _problem_from_protocol(protocol: Any) -> Any:
    from pipelines.contracts.problem import ProblemRequest
    from pipelines.static_adapt.builders.problem_registry import resolve_problem_context

    receipt = protocol.problem
    return resolve_problem_context(
        ProblemRequest(
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
            n_fermions=(None if receipt.n_fermions is None else int(receipt.n_fermions)),
        )
    )


def _load_protocol(
    *, job: Mapping[str, Any], payload: Mapping[str, Any], source_locks: Mapping[str, Any]
) -> tuple[Any, Any]:
    from pipelines.static_adapt.ra_adapt.contracts import (
        _attach_validated_bundle_protocol_authority,
        _mint_bundle_protocol_materialization_authority,
        resolved_ra_adapt_protocol_from_mapping,
    )

    protocol = resolved_ra_adapt_protocol_from_mapping(payload)
    receipt = protocol.bundle_materialization
    invariants = protocol.route_contract["semantic_invariants"]
    route_execution = protocol.route_contract["execution_settings"]
    lineage = protocol.route_contract["lineage_authority"]
    if receipt is None:
        raise PackageContractError("Protocol lost bundle materialization.")
    if (
        protocol.sha256 != job.get("protocol_sha256")
        or receipt.bundle_id != PACKAGE_ID
        or receipt.bundle_manifest_sha256
        != job.get("protocol_bundle_manifest_sha256")
        or receipt.source_locks_sha256 != source_locks.get("sha256")
        or receipt.cell_id != job.get("execution_id")
        or receipt.source_lock_id != job.get("source_lock_id")
        or int(protocol.horizon) != TARGET_HORIZON
        or protocol.request.execution.stop.maximum_controller_rounds
        != TARGET_HORIZON
        or protocol.request.execution.resume.kind != "fresh_start"
        or protocol.request.method.admission.kind != "singleton"
        or protocol.request.method.insertion.kind != "plateau_commutation"
        or protocol.request.method.pruning.kind != "off"
        or protocol.request.method.beam.kind != "off"
        or protocol.algorithm_id != ALGORITHM_ID
        or protocol.route_contract.get("sha256") != ROUTE_CONTRACT_SHA256
        or protocol.route_contract.get("route_profile") != ROUTE_PROFILE
        or lineage.get("parent_contract_sha256")
        != PARENT_ROUTE_CONTRACT_SHA256
        or route_execution.get("adapt_insertion_mode")
        != "insertion_commutation_plateau_v2"
        or invariants.get("experimental_insertion_policy")
        != "insertion_commutation_plateau_v2"
        or invariants.get("plateau_prior_mean_decrease_ratio_threshold")
        != PLATEAU_PRIOR_MEAN_RATIO_THRESHOLD
        or invariants.get("plateau_threshold_comparison") != PLATEAU_COMPARISON
        or invariants.get("plateau_trigger_source") != PLATEAU_TRIGGER
        or invariants.get("plateau_threshold_calibration_status") != PLATEAU_CALIBRATION
        or route_execution.get("ra_phase3_population_activation_policy")
        != "same_round_insertion_plateau_predicate_v1"
        or route_execution.get("ra_phase3_preplateau_materialization_policy")
        != "phase2_winner_only_refit_geometry_v1"
        or invariants.get("phase3_competitive_population_activation")
        != "same_round_insertion_plateau_predicate_v1"
        or invariants.get("phase3_activation_source")
        != "same_round_authenticated_insertion_plateau_domain_open_v1"
        or invariants.get("phase3_preplateau_materialization_policy")
        != "phase2_winner_only_refit_geometry_v1"
        or invariants.get("phase3_activation_independent_latch") is not False
        or invariants.get("phase3_activation_hysteresis_active") is not False
        or protocol.active_gradient_policy != ACTIVE_GRADIENT_POLICY
        or protocol.resource_weighting_scope != RESOURCE_WEIGHTING_SCOPE
        or protocol.candidate_representation != CANDIDATE_REPRESENTATION
    ):
        raise PackageContractError("Typed Phase-III-on-plateau protocol drifted.")
    authority = _mint_bundle_protocol_materialization_authority(
        receipt,
        source_lock_refs=protocol.source_locks,
        protocol_sha256=protocol.sha256,
    )
    protocol = _attach_validated_bundle_protocol_authority(protocol, authority)
    return protocol, _problem_from_protocol(protocol)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _prepare(job_path: Path) -> tuple[dict[str, Any], dict[str, Any], Any, Any, tempfile.TemporaryDirectory[str]]:
    job, manifest, payload, source_locks = _load_closed_job(job_path)
    temporary = tempfile.TemporaryDirectory(
        prefix=f"paper-i-ra-phase3-plateau-r50-{job['execution_id']}."
    )
    try:
        source_root = Path(temporary.name) / "source"
        _extract_source(
            manifest=manifest, source_locks=source_locks, destination=source_root
        )
        original = Path.cwd()
        os.chdir(source_root)
        try:
            _activate_source_root(source_root)
            protocol, problem = _load_protocol(
                job=job, payload=payload, source_locks=source_locks
            )
            _reassert_source_root(source_root)
        finally:
            os.chdir(original)
    except BaseException:
        temporary.cleanup()
        raise
    return job, manifest, protocol, problem, temporary


def preflight(job_path: Path) -> dict[str, Any]:
    job, manifest, protocol, _problem, temporary = _prepare(job_path)
    temporary.cleanup()
    return digested(
        {
            "schema": "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_r50_worker_preflight_v1",
            "status": "passed",
            "execution_id": job["execution_id"],
            "job_spec_sha256": job["sha256"],
            "package_manifest_sha256": manifest["sha256"],
            "protocol_sha256": protocol.sha256,
            "source_archive_import_isolated": True,
            "fresh_start": True,
            "target_horizon": TARGET_HORIZON,
        }
    )


def _execute(
    *, protocol: Any, problem: Any, staging: Path, maximum_rounds: int
) -> tuple[Any, int]:
    from pipelines.static_adapt.ra_adapt import RAAdaptOperationalControls, run_ra_adapt
    from pipelines.static_adapt.sr_snake import (
        CheckpointObservation,
        EstimatorLedgerObservation,
        FreshStart,
        SRObservationPolicy,
    )

    controls = RAAdaptOperationalControls(
        maximum_controller_rounds=maximum_rounds,
        resume=FreshStart(),
        observation=SRObservationPolicy(
            checkpoint=CheckpointObservation(
                path=staging / "checkpoint.json",
                every_controller_rounds=1,
                keep_history_tail=100,
            ),
            estimator_ledger=EstimatorLedgerObservation(
                path=staging / "estimator_ledger.json"
            ),
            resource_rounds=((1,) if maximum_rounds == 1 else (50,)),
        ),
    )
    result = run_ra_adapt(problem, protocol, operational_controls=controls)
    rounds = len(result.run.accepted_trajectory)
    result_payload = result.to_dict()
    accounting = result_payload.get("run", {}).get("estimator_accounting", {})
    summary = result.run.paper_i_summary
    summary_payload = None if summary is None else summary.to_dict()
    requested_round_rows = (
        None if summary_payload is None else summary_payload.get("requested_rounds")
    )
    all_work = accounting.get("all_work", {})
    components = all_work.get("components", {})
    expected_component_names = {
        "n_h_outer",
        "n_h_refit",
        "n_grad",
        "n_metric",
    }
    component_values_are_closed = (
        set(components) == expected_component_names
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in components.values()
        )
    )
    s_alg = all_work.get("s_alg")
    if (
        result.protocol.sha256 != protocol.sha256
        or rounds != maximum_rounds
        or not (staging / "checkpoint.json").is_file()
        or not (staging / "estimator_ledger.json").is_file()
        or accounting.get("complete") is not True
        or accounting.get("prefix_closure_passed") is not True
        or accounting.get("prefix_closure_status") != "complete"
        or accounting.get("exact_blockers") != []
        or not component_values_are_closed
        or not isinstance(s_alg, int)
        or isinstance(s_alg, bool)
        or s_alg != sum(components.values())
        or accounting.get("raw_occurrences") != components
        or accounting.get("raw_occurrence_total") != s_alg
        or accounting.get("winning_lineage") != all_work
        or summary_payload is None
        or summary_payload.get("canonical_all_work") != all_work
        or summary_payload.get("available_controller_rounds") != rounds
        or not isinstance(requested_round_rows, (list, tuple))
        or len(requested_round_rows) != 1
        or requested_round_rows[0].get("controller_round") != maximum_rounds
        or requested_round_rows[0].get("status") != "available"
        or requested_round_rows[0].get("algorithmic_work") != all_work
    ):
        raise PackageContractError("Fresh execution observation closure failed.")
    return result, rounds


def smoke_one_round(job_path: Path) -> dict[str, Any]:
    job, manifest, protocol, problem, temporary = _prepare(job_path)
    try:
        source_root = Path(temporary.name) / "source"
        staging = Path(temporary.name) / "smoke"
        staging.mkdir()
        original = Path.cwd()
        os.chdir(source_root)
        try:
            result, rounds = _execute(
                protocol=protocol, problem=problem, staging=staging, maximum_rounds=1
            )
        finally:
            os.chdir(original)
        if rounds != 1:
            raise PackageContractError("One-round smoke did not complete one round.")
        return digested(
            {
                "schema": "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_r50_worker_smoke_v1",
                "status": "passed_real_run_ra_adapt_one_round",
                "execution_id": job["execution_id"],
                "job_spec_sha256": job["sha256"],
                "package_manifest_sha256": manifest["sha256"],
                "protocol_sha256": result.protocol.sha256,
                "controller_rounds_completed": rounds,
                "fresh_start": True,
                "source_archive_import_isolated": True,
            }
        )
    finally:
        temporary.cleanup()


def _json_sidecar_pointers(value: Any) -> set[tuple[str, str]]:
    pointers: set[tuple[str, str]] = set()
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            path = current.get("path")
            sha256 = current.get("sha256")
            if isinstance(path, str) and isinstance(sha256, str):
                pointers.add((path, sha256))
            stack.extend(current.values())
        elif isinstance(current, (list, tuple)):
            stack.extend(current)
    return pointers


def _science_payload_bindings(staging: Path) -> dict[str, dict[str, Any]]:
    """Close required results plus the two checkpoint-bound dynamic sidecars."""

    if not staging.is_dir() or staging.is_symlink():
        raise PackageContractError("Worker science staging is unavailable.")
    children = sorted(staging.iterdir(), key=lambda path: path.name)
    if any(
        path.is_symlink() or not path.is_file() or path.name in {".", ".."}
        for path in children
    ):
        raise PackageContractError("Worker science payload is unsafe.")
    by_name = {path.name: path for path in children}
    if len(by_name) != len(children):
        raise PackageContractError("Worker science payload names collide.")
    missing = sorted(_REQUIRED_SCIENCE_PAYLOADS - set(by_name))
    if missing:
        raise PackageContractError(
            f"Successful execution payload closure is incomplete: {missing}"
        )

    dynamic = set(by_name) - _REQUIRED_SCIENCE_PAYLOADS
    estimator_names = {
        name for name in dynamic if _ESTIMATOR_CHECKPOINT_SIDECAR.fullmatch(name)
    }
    resume_names = {
        name for name in dynamic if _SINGLETON_RESUME_SIDECAR.fullmatch(name)
    }
    if (
        dynamic != estimator_names | resume_names
        or len(estimator_names) != 1
        or len(resume_names) != 1
    ):
        raise PackageContractError(
            "Successful execution has missing or unpermitted checkpoint sidecars."
        )

    try:
        checkpoint = json.loads(
            by_name["checkpoint.json"].read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackageContractError("Checkpoint JSON is unreadable.") from exc
    if not isinstance(checkpoint, Mapping):
        raise PackageContractError("Checkpoint JSON must be an object.")
    pointers = _json_sidecar_pointers(checkpoint)

    expected_schemas = {
        **{
            name: "paper_i_estimator_call_ledger_checkpoint_sidecar_v2"
            for name in estimator_names
        },
        **{
            name: "static_adapt_signed_active_prefix_resume_sidecar_v2"
            for name in resume_names
        },
    }
    for name, schema in expected_schemas.items():
        path = by_name[name]
        digest = sha256_file(path)
        match = (
            _ESTIMATOR_CHECKPOINT_SIDECAR.fullmatch(name)
            or _SINGLETON_RESUME_SIDECAR.fullmatch(name)
        )
        assert match is not None
        if match.group(1) != digest[:16] or (name, digest) not in pointers:
            raise PackageContractError(
                f"Checkpoint sidecar binding drifted: {name}"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PackageContractError(
                f"Checkpoint sidecar is unreadable: {name}"
            ) from exc
        if not isinstance(payload, Mapping) or payload.get("schema") != schema:
            raise PackageContractError(
                f"Checkpoint sidecar schema drifted: {name}"
            )

    dynamic_pointer_rows = {
        (path, digest)
        for path, digest in pointers
        if _ESTIMATOR_CHECKPOINT_SIDECAR.fullmatch(path)
        or _SINGLETON_RESUME_SIDECAR.fullmatch(path)
    }
    expected_pointer_rows = {
        (name, sha256_file(by_name[name])) for name in dynamic
    }
    if dynamic_pointer_rows != expected_pointer_rows:
        raise PackageContractError("Checkpoint sidecar pointer closure drifted.")
    return {
        name: {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in sorted(by_name.items())
    }


def _publish_staging_directory(staging: Path, output_dir: Path) -> None:
    """Atomically publish a validated sibling on the worker filesystem."""
    if output_dir.exists() or output_dir.is_symlink():
        raise PackageContractError("Worker output destination already exists.")
    if (
        not staging.is_dir()
        or staging.is_symlink()
        or staging.parent.resolve() != output_dir.parent.resolve()
    ):
        raise PackageContractError(
            "Worker success publication must use a same-filesystem sibling."
        )
    os.rename(staging, output_dir)


def run_cell(
    *, job_path: Path, authorization_path: Path, output_dir: Path, receipt_path: Path
) -> dict[str, Any]:
    job, manifest, protocol, problem, temporary = _prepare(job_path)
    try:
        authorization = _validate_authorization(
            authorization_path, job=job, manifest=manifest
        )
        progress_dir = output_dir.with_name(f"{output_dir.name}.in_progress")
        if (
            output_dir.parent.resolve() != receipt_path.parent.resolve()
            or output_dir.exists()
            or output_dir.is_symlink()
            or progress_dir.exists()
            or progress_dir.is_symlink()
            or receipt_path.exists()
            or receipt_path.is_symlink()
        ):
            raise PackageContractError("Worker destination already exists.")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        progress_dir.mkdir(exist_ok=False)
        source_root = Path(temporary.name) / "source"
        original = Path.cwd()
        os.chdir(source_root)
        try:
            result, rounds = _execute(
                protocol=protocol,
                problem=problem,
                staging=progress_dir,
                maximum_rounds=TARGET_HORIZON,
            )
        finally:
            os.chdir(original)
        _write_json(progress_dir / "result.json", result.to_dict())
        _write_json(
            progress_dir / "paper_i_summary.json",
            result.run.paper_i_summary.to_dict(),
        )
        preliminary = _science_payload_bindings(progress_dir)
        execution_manifest = digested(
            {
                "schema": "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_r50_execution_manifest_v2",
                "status": "passed",
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "execution_id": job["execution_id"],
                "job_spec_sha256": job["sha256"],
                "authorization_sha256": authorization["sha256"],
                "protocol_sha256": protocol.sha256,
                "target_horizon": TARGET_HORIZON,
                "controller_rounds_completed": rounds,
                "fresh_start": True,
                "source_checkpoint_consumed": False,
                "worker_owned_live_progress": True,
                "same_filesystem_atomic_success_publication": True,
                "output_payloads": preliminary,
            }
        )
        _write_json(
            progress_dir / "execution_manifest.json", execution_manifest
        )
        if set(path.name for path in progress_dir.iterdir()) != (
            set(preliminary) | {"execution_manifest.json"}
        ):
            raise PackageContractError(
                "Execution manifest publication closure drifted."
            )
        _publish_staging_directory(progress_dir, output_dir)
        receipt = digested(
            {
                "schema": "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_r50_worker_receipt_v2",
                "status": "passed",
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "execution_id": job["execution_id"],
                "job_spec_sha256": job["sha256"],
                "authorization_sha256": authorization["sha256"],
                "execution_manifest_sha256": execution_manifest["sha256"],
                "controller_rounds_completed": rounds,
                "fresh_start": True,
                "artifacts": [
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
    finally:
        temporary.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--smoke-one-round", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--execution-authorization", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    try:
        if args.preflight or args.smoke_one_round:
            if any(
                value is not None
                for value in (args.execution_authorization, args.output_dir, args.receipt)
            ):
                raise PackageContractError("Preflight/smoke accepts no execution destinations.")
            payload = (
                preflight(args.job.resolve())
                if args.preflight
                else smoke_one_round(args.job.resolve())
            )
        else:
            if any(
                value is None
                for value in (args.execution_authorization, args.output_dir, args.receipt)
            ):
                raise PackageContractError("Execution requires authorization and destinations.")
            payload = run_cell(
                job_path=args.job.resolve(),
                authorization_path=args.execution_authorization.resolve(),
                output_dir=args.output_dir.resolve(),
                receipt_path=args.receipt.resolve(),
            )
    except (OSError, PackageContractError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(payload).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
