#!/usr/bin/env python3
"""Preflight or execute one pure-Hubbard Page-12 full-noise cell."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "off"
os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"

from package_contract import (  # noqa: E402
    ACTIVE_GRADIENT_POLICY,
    ACTIVATION_REQUEST_SCHEMA,
    ALGORITHM_ID,
    APPLICATION_SOURCE_LOCK_KEY,
    AUTHORIZATION_SCHEMA,
    BACKEND_COMPILE_SCOPE,
    BUNDLE_ID,
    CAMPAIGN_ID,
    CANDIDATE_ADAPTER_ID,
    CANDIDATE_REPRESENTATION,
    EXPECTED_CANDIDATE_FUNNEL,
    JOB_SCHEMA,
    NOISE_TUPLE_ORDER,
    PACKAGE_ID,
    PACKAGE_MANIFEST_SCHEMA,
    PHASE0_POLICY,
    PHASE0_SHORTLIST_SIZE,
    PHASE0_VARIANT,
    PLATEAU_THRESHOLD,
    RESOURCE_WEIGHTING_SCOPE,
    REMOTE_IMAGE_PATH,
    REMOTE_IMAGE_SHA256,
    SELECTOR_COMPILE_COST_PHASE_REUSE,
    SELECTOR_COMPILE_COST_POLICY,
    TARGET_HORIZON,
    TARGET_ROUTE_PROFILE,
    PackageContractError,
    canonical_json_bytes,
    digested,
    expected_execution_ids,
    load_json,
    safe_relative_path,
    sha256_file,
    validate_control_file_bindings,
    verify_self_digest,
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
    result = PACKAGE_DIR / safe_relative_path(value, label=label)
    try:
        result.resolve().relative_to(PACKAGE_DIR.resolve())
    except ValueError as exc:
        raise PackageContractError(f"{label} escaped the package.") from exc
    return result


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
    if verify_self_digest(payload, label=label) != row.get(
        "canonical_sha256"
    ):
        raise PackageContractError(f"{label} canonical binding drifted.")
    return path, payload


def _expected_artifact_paths(job: Mapping[str, Any]) -> dict[str, str]:
    execution_id = str(job.get("execution_id", ""))
    root = PurePosixPath("runs") / execution_id
    suffixes = {
        "execution_manifest": PurePosixPath("execution_manifest.json"),
        "checkpoint": PurePosixPath("checkpoints/current.json"),
        "estimator_ledger": PurePosixPath("result/estimator_ledger.json"),
        "result": PurePosixPath("result/result.json"),
        "summary": PurePosixPath("summary/summary.json"),
    }
    raw = _mapping(
        job.get("expected_run_artifacts"), label="expected run artifacts"
    )
    if set(raw) != set(suffixes):
        raise PackageContractError("Expected-artifact role closure drifted.")
    paths: dict[str, str] = {}
    for role, suffix in suffixes.items():
        row = _mapping(raw.get(role), label=f"expected {role}")
        path = PurePosixPath(str(row.get("path", "")))
        if (
            path != root / suffix
            or row.get("required") is not True
            or row.get("direct_file_required") is not True
            or row.get("reference_receipt_required") is not False
            or row.get("fulfillment_kind") != "direct_execution_v1"
        ):
            raise PackageContractError(
                f"Expected-artifact contract drifted for {role}."
            )
        paths[role] = suffix.as_posix()
    return paths


def _load_closed_job(
    job_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = load_json(
        PACKAGE_DIR / "package_manifest.json",
        label="package manifest",
    )
    verify_self_digest(manifest, label="package manifest")
    expected_ids = list(expected_execution_ids())
    if (
        manifest.get("schema") != PACKAGE_MANIFEST_SCHEMA
        or manifest.get("status") != "passed_inert_six_cells"
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("bundle_id") != BUNDLE_ID
        or manifest.get("row_count") != 6
        or manifest.get("execution_ids") != expected_ids
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_authorized") is not False
        or manifest.get("submission_ready") is not False
        or manifest.get("submit_descriptor_present") is not False
        or manifest.get("submitted") is not False
    ):
        raise PackageContractError("Inert six-cell package manifest drifted.")
    validate_control_file_bindings(PACKAGE_DIR, manifest)

    resolved_job = job_path.resolve()
    job_rows = _sequence(manifest.get("jobs"), label="job bindings")
    matches = [
        row
        for row in job_rows
        if isinstance(row, Mapping)
        and _package_path(row.get("path"), label="job path").resolve()
        == resolved_job
    ]
    if len(matches) != 1:
        raise PackageContractError("Requested job is outside the package.")
    bound_job, job_payload = _verify_binding(
        matches[0], label="job", canonical=True
    )
    assert job_payload is not None
    job = job_payload
    execution_id = str(job.get("execution_id", ""))
    if (
        bound_job.resolve() != resolved_job
        or execution_id not in expected_ids
        or job.get("schema") != JOB_SCHEMA
        or job.get("package_id") != PACKAGE_ID
        or job.get("campaign_id") != CAMPAIGN_ID
        or job.get("bundle_id") != BUNDLE_ID
        or job.get("algorithm_id") != ALGORITHM_ID
        or job.get("active_gradient_policy") != ACTIVE_GRADIENT_POLICY
        or job.get("resource_weighting_scope") != RESOURCE_WEIGHTING_SCOPE
        or job.get("candidate_representation") != CANDIDATE_REPRESENTATION
        or job.get("candidate_adapter_id") != CANDIDATE_ADAPTER_ID
        or job.get("selector_compile_cost_scope") != BACKEND_COMPILE_SCOPE
        or job.get("application_source_contract_sha256")
        != manifest.get("application_source_contract_sha256s", {}).get(
            execution_id
        )
        or job.get("num_sites") != 2
        or job.get("nph") != 0
        or float(job.get("u_over_t", -1.0)) not in {1.5, 8.0}
        or job.get("noise_level_id") not in {"low", "high", "extreme"}
        or not isinstance(job.get("source_lock_id"), str)
        or int(job.get("target_horizon", -1)) != TARGET_HORIZON
        or job.get("fresh_start_contract")
        != {
            "kind": "fresh_start",
            "resume_archive": None,
            "source_checkpoint": None,
        }
        or job.get("output_archive_contract")
        != {
            "schema": (
                "paper_i_pure_hubbard_page12_noise_terminal_archive_v1"
            ),
            "path_template": (
                "transfer/{execution_id}__{cluster_id}__{proc_id}.tar.gz"
            ),
            "transfer_policy": "on_exit_only_v1",
            "scheduler_attempt_identity_inside_archive": True,
        }
        or job.get("execution_authorized") is not False
        or job.get("submission_authorized") is not False
        or job.get("submitted") is not False
    ):
        raise PackageContractError("Sealed gradient-Phase-0 job drifted.")

    protocol_rows = _sequence(
        manifest.get("protocols"), label="protocol bindings"
    )
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
        protocol_path
        != _package_path(job.get("protocol_path"), label="protocol path")
        or protocol_matches[0].get("canonical_sha256")
        != job.get("protocol_sha256")
        or protocol_matches[0].get("sha256")
        != job.get("protocol_file_sha256")
    ):
        raise PackageContractError("Job-to-protocol binding drifted.")

    _bundle_path, bundle = _verify_binding(
        manifest.get("bundle_manifest"),
        label="bundle manifest",
        canonical=True,
    )
    _expected_path, expected_artifacts = _verify_binding(
        manifest.get("bundle_expected_artifacts"),
        label="bundle expected artifacts",
        canonical=True,
    )
    _locks_path, locks = _verify_binding(
        manifest.get("bundle_source_locks"),
        label="bundle source locks",
        canonical=True,
    )
    assert bundle is not None and expected_artifacts is not None
    assert locks is not None
    artifact_cells = expected_artifacts.get("cells")
    expected_cell = (
        artifact_cells.get(execution_id)
        if isinstance(artifact_cells, Mapping)
        else None
    )
    expected_run_artifacts = (
        expected_cell.get("expected_run_artifacts")
        if isinstance(expected_cell, Mapping)
        else None
    )
    if (
        bundle.get("sha256") != job.get("bundle_manifest_sha256")
        or expected_artifacts.get("sha256")
        != job.get("expected_artifacts_manifest_sha256")
        or expected_run_artifacts != job.get("expected_run_artifacts")
        or locks.get("sha256") != job.get("source_locks_sha256")
        or locks.get("implementation_sources", {}).get("sha256")
        != job.get("implementation_source_inventory_sha256")
        or not isinstance(locks.get("cell_locks", {}).get(job.get("source_lock_id")), Mapping)
        or locks.get("cell_locks", {})[job.get("source_lock_id")].get(
            "application_source_contract_sha256"
        )
        != job.get("application_source_contract_sha256")
        or manifest.get("route_contract_sha256_by_noise_level", {}).get(
            job.get("noise_level_id")
        )
        != job.get("route_contract_sha256")
    ):
        raise PackageContractError("Job source authority drifted.")
    _expected_artifact_paths(job)
    return job, manifest, protocol_payload, locks


def _validate_authorization(
    path: Path,
    *,
    job: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    control_files = validate_control_file_bindings(PACKAGE_DIR, manifest)
    authority = load_json(path, label="execution authorization")
    verify_self_digest(authority, label="execution authorization")
    source_archive = _mapping(
        manifest.get("source_archive"), label="source archive"
    )
    if path.parent.name != "authorizations":
        raise PackageContractError(
            "Execution authorization must be inside an activation directory."
        )
    activation_root = path.parent.parent

    def activation_binding(
        raw: Any,
        *,
        label: str,
        expected_path: str,
    ) -> dict[str, Any]:
        row = _mapping(raw, label=f"{label} binding")
        if row.get("path") != expected_path:
            raise PackageContractError(f"{label} path drifted.")
        target = activation_root / safe_relative_path(
            row.get("path"), label=f"{label} path"
        )
        if (
            not target.is_file()
            or target.is_symlink()
            or target.stat().st_size != int(row.get("size_bytes", -1))
            or sha256_file(target) != row.get("sha256")
        ):
            raise PackageContractError(f"{label} byte binding drifted.")
        payload = load_json(target, label=label)
        if verify_self_digest(payload, label=label) != row.get(
            "canonical_sha256"
        ):
            raise PackageContractError(f"{label} canonical binding drifted.")
        return payload

    request = activation_binding(
        authority.get("activation_request"),
        label="activation request",
        expected_path="activation_request.json",
    )
    image_probe = activation_binding(
        authority.get("image_runtime_probe"),
        label="image runtime probe",
        expected_path="image_runtime_probe.json",
    )
    probe = image_probe.get("pinned_image_runtime_probe")
    backend_probe = (
        probe.get("probe") if isinstance(probe, Mapping) else None
    )
    pinned_p4 = (
        probe.get("p4_numerical_witness")
        if isinstance(probe, Mapping)
        else None
    )
    if (
        authority.get("schema") != AUTHORIZATION_SCHEMA
        or authority.get("package_id") != PACKAGE_ID
        or authority.get("campaign_id") != CAMPAIGN_ID
        or authority.get("execution_id") != job.get("execution_id")
        or authority.get("job_spec_sha256") != job.get("sha256")
        or authority.get("package_manifest_sha256") != manifest.get("sha256")
        or authority.get("package_manifest_file_sha256")
        != sha256_file(PACKAGE_DIR / "package_manifest.json")
        or authority.get("control_file_sha256s")
        != {
            name: row["sha256"]
            for name, row in control_files.items()
        }
        or authority.get("protocol_sha256") != job.get("protocol_sha256")
        or authority.get("source_archive_sha256")
        != source_archive.get("sha256")
        or authority.get("scope") != "single_cell_chtc_execution_only"
        or authority.get("authorization_kind")
        != "explicit_user_execution_and_submission_authority"
        or authority.get("pinned_image_path") != REMOTE_IMAGE_PATH
        or authority.get("pinned_image_sha256") != REMOTE_IMAGE_SHA256
        or authority.get("execution_authorized") is not True
        or authority.get("submission_authorized") is not True
        or authority.get("paper_evidence_adoption_authorized") is not False
        or authority.get("submitted") is not False
        or request.get("schema") != ACTIVATION_REQUEST_SCHEMA
        or request.get("package_id") != PACKAGE_ID
        or request.get("campaign_id") != CAMPAIGN_ID
        or request.get("bundle_id") != BUNDLE_ID
        or request.get("package_manifest_sha256") != manifest.get("sha256")
        or request.get("requested_execution_ids")
        != list(expected_execution_ids())
        or request.get("scope")
        != "prepare_six_cell_chtc_execution_and_submission_v1"
        or request.get("authorization_kind")
        != "explicit_user_execution_and_submission_authority"
        or request.get("explicit_user_authority_recorded") is not True
        or request.get("execution_authorized") is not True
        or request.get("submission_authorized") is not True
        or request.get("paper_evidence_adoption_authorized") is not False
        or request.get("submitted") is not False
        or image_probe.get("status") != "passed_inert_package"
        or image_probe.get("package_manifest_sha256") != manifest.get("sha256")
        or image_probe.get("package_manifest_file_sha256")
        != sha256_file(PACKAGE_DIR / "package_manifest.json")
        or image_probe.get("control_file_sha256s")
        != {
            name: row["sha256"]
            for name, row in control_files.items()
        }
        or image_probe.get("launch_ready") is not True
        or image_probe.get("deep_pinned_numerical_p4_passed") is not True
        or image_probe.get("execution_authorized") is not False
        or image_probe.get("submission_authorized") is not False
        or not isinstance(probe, Mapping)
        or probe.get("status") != "passed"
        or probe.get("image_sha256") != REMOTE_IMAGE_SHA256
        or not isinstance(backend_probe, Mapping)
        or backend_probe.get("resolved_backend_name")
        != "FakeMarrakesh"
        or backend_probe.get("backend_resolution_kind")
        != "fake_exact"
        or not isinstance(pinned_p4, Mapping)
        or pinned_p4.get("schema")
        != "paper_i_pure_hubbard_page12_noise_p4_receipt_v1"
        or pinned_p4.get("status") != "passed"
        or pinned_p4.get("scientific_execution_performed") is not True
        or pinned_p4.get("source_locked_archive_validated") is not True
        or pinned_p4.get("real_noisy_gradient_probe_passed") is not True
        or pinned_p4.get("real_noisy_powell_probe_passed") is not True
    ):
        raise PackageContractError(
            "Execution authorization is stale or overbroad."
        )
    return authority


def _extract_source(
    *,
    manifest: Mapping[str, Any],
    source_locks: Mapping[str, Any],
    destination: Path,
) -> None:
    archive_path, _ = _verify_binding(
        manifest.get("source_archive"), label="source archive"
    )
    _source_path, source_manifest = _verify_binding(
        manifest.get("source_archive_manifest"),
        label="source archive manifest",
        canonical=True,
    )
    assert source_manifest is not None
    if (
        source_manifest.get("archive") != manifest.get("source_archive")
        or source_manifest.get("status") != "passed"
        or source_manifest.get("no_ambient_repo_imports") is not True
        or source_manifest.get("implementation_source_inventory_sha256")
        != source_locks.get("implementation_sources", {}).get("sha256")
    ):
        raise PackageContractError("Runtime source archive authority drifted.")
    rows = _sequence(source_manifest.get("members"), label="source members")
    members = {
        safe_relative_path(row.get("path"), label="source member").as_posix(): row
        for row in rows
        if isinstance(row, Mapping)
    }
    if len(members) != len(rows) or len(rows) != int(
        source_manifest.get("member_count", -1)
    ):
        raise PackageContractError("Runtime source member closure drifted.")
    destination.mkdir(parents=True, exist_ok=False)
    observed: set[str] = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            relative = safe_relative_path(
                member.name, label="tar member"
            ).as_posix()
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
        raise PackageContractError("Runtime source extraction is incomplete.")


def _activate_source_root(source_root: Path) -> None:
    root = source_root.resolve()
    for name in list(sys.modules):
        if (
            name == "pipelines"
            or name.startswith("pipelines.")
            or name == "src"
            or name.startswith("src.")
        ):
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
        raise PackageContractError(
            "Runtime implementation escaped the source archive."
        ) from exc


def _problem_from_protocol(protocol: Any) -> Any:
    from pipelines.contracts.problem import ProblemRequest
    from pipelines.static_adapt.builders.problem_registry import (
        resolve_problem_context,
    )

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
            n_fermions=(
                None
                if receipt.n_fermions is None
                else int(receipt.n_fermions)
            ),
        )
    )


def _load_protocol(
    *,
    job: Mapping[str, Any],
    payload: Mapping[str, Any],
    source_locks: Mapping[str, Any],
) -> tuple[Any, Any]:
    from pipelines.static_adapt.ra_adapt.contracts import (
        _attach_validated_bundle_protocol_authority,
        _mint_bundle_protocol_materialization_authority,
        resolved_ra_adapt_protocol_from_mapping,
    )

    protocol = resolved_ra_adapt_protocol_from_mapping(payload)
    receipt = protocol.bundle_materialization
    route = protocol.route_contract
    execution = route.get("execution_settings")
    invariants = route.get("semantic_invariants")
    lineage = route.get("lineage_authority")
    noise = (
        execution.get("ra_controller_noise_contract")
        if isinstance(execution, Mapping)
        else None
    )
    if (
        receipt is None
        or receipt.bundle_id != BUNDLE_ID
        or receipt.bundle_manifest_sha256
        != job.get("bundle_manifest_sha256")
        or receipt.source_locks_sha256 != source_locks.get("sha256")
        or receipt.cell_id != job.get("execution_id")
        or receipt.source_lock_id != job.get("source_lock_id")
        or protocol.sha256 != job.get("protocol_sha256")
        or protocol.algorithm_id != ALGORITHM_ID
        or protocol.route_contract.get("sha256")
        != job.get("route_contract_sha256")
        or protocol.route_contract.get("route_profile") != TARGET_ROUTE_PROFILE
        or int(protocol.horizon) != int(job.get("target_horizon", -1))
        or protocol.request.execution.stop.maximum_controller_rounds
        != int(job.get("target_horizon", -1))
        or protocol.request.execution.resume.kind != "fresh_start"
        or protocol.request.method.admission.kind != "singleton"
        or protocol.request.method.insertion.kind != "plateau_commutation"
        or protocol.request.method.pruning.kind != "off"
        or protocol.request.method.beam.kind != "off"
        or protocol.request.adapter.adapter_id != CANDIDATE_ADAPTER_ID
        or protocol.active_gradient_policy != ACTIVE_GRADIENT_POLICY
        or protocol.resource_weighting_scope != RESOURCE_WEIGHTING_SCOPE
        or protocol.candidate_representation != CANDIDATE_REPRESENTATION
        or protocol.problem.problem_key != "hubbard"
        or protocol.problem.num_sites != 2
        or protocol.problem.n_ph_max != 0
        or protocol.problem.n_fermions != 2
        or float(protocol.problem.u) != float(job.get("u_over_t", -1.0))
        or protocol.source_locks.get(APPLICATION_SOURCE_LOCK_KEY)
        != job.get("application_source_contract_sha256")
        or not isinstance(execution, Mapping)
        or execution.get("phase3_backend_cost_scope") != BACKEND_COMPILE_SCOPE
        or execution.get("phase3_hardware_cost_normalization_mode")
        != "zero_centered_signed_arctan_v1"
        or execution.get("static_lane_route") != "global_single_population"
        or "physical_lane_shortlist_aggressiveness" in execution
        or execution.get("ra_phase0_gradient_shortlist_policy")
        != PHASE0_POLICY
        or execution.get("ra_phase0_gradient_shortlist_size")
        != PHASE0_SHORTLIST_SIZE
        or execution.get("adapt_parallel_gradient_workers") != 1
        or not isinstance(noise, Mapping)
        or noise.get("noise_level_id") != job.get("noise_level_id")
        or tuple(noise.get("noise_tuple_order", ())) != NOISE_TUPLE_ORDER
        or list(noise.get("noise_tuple", ())) != job.get("noise_tuple")
        or noise.get("optimizer_evaluation_order") != "serial_v1"
        or not isinstance(invariants, Mapping)
        or invariants.get("selector_compile_cost_policy")
        != SELECTOR_COMPILE_COST_POLICY
        or invariants.get("selector_compile_cost_phase_reuse")
        != SELECTOR_COMPILE_COST_PHASE_REUSE
        or invariants.get("phase_i_compile_cost_source")
        != "structural_proxy_v1"
        or invariants.get("phase_ii_compile_cost_source")
        != "backend_transpile_v1"
        or invariants.get("phase_iii_compile_cost_source")
        != "backend_transpile_v1"
        or invariants.get(
            "phase_ii_phase_iii_qiskit_negative_delta_reward_enabled"
        )
        is not True
        or invariants.get(
            "phase_ii_phase_iii_qiskit_population_normalization_policy"
        )
        != "zero_centered_signed_arctan_v1"
        or invariants.get("physical_operator_lanes_active") is not False
        or invariants.get("shortlist_population_policy")
        != "single_global_population_v1"
        or "plateau_prior_mean_decrease_ratio_threshold" in invariants
        or invariants.get("plateau_cumulative_decrease_ratio_threshold")
        != PLATEAU_THRESHOLD
        or invariants.get("plateau_threshold_comparison")
        != "marginal_to_prior_cumulative_strictly_below_v1"
        or invariants.get("plateau_trigger_source")
        != (
            "immediately_preceding_marginal_over_prior_cumulative_"
            "accepted_post_full_refit_energy_decrease_v1"
        )
        or invariants.get("plateau_threshold_calibration_status")
        != "source_locked_completed_trajectory_replay_v1"
        or invariants.get("plateau_energy_source")
        != "persisted_noisy_controller_energy_before_after_v1"
        or invariants.get("application_lane")
        != "paper_i_pure_hubbard_page12_full_noise_v1"
        or invariants.get("controller_noise_active") is not True
        or invariants.get("controller_noise_candidate_gradient_scoring")
        != "noisy"
        or invariants.get("controller_noise_powell_refit_objective")
        != "noisy"
        or invariants.get("controller_noise_geometry_and_gram") != "exact"
        or invariants.get("phase0_active") is not True
        or invariants.get("phase0_score")
        != "standard_adapt_absolute_gradient_v1"
        or invariants.get("phase0_fubini_metric_active") is not False
        or invariants.get("phase0_resource_cost_active") is not False
        or invariants.get("phase0_compile_cost_active") is not False
        or invariants.get("phase0_shortlist_size") != PHASE0_SHORTLIST_SIZE
        or invariants.get("phase0_estimator_components") != ["N_grad"]
        or invariants.get("candidate_funnel_order")
        != EXPECTED_CANDIDATE_FUNNEL
        or (
            PHASE0_VARIANT == "macro"
            and invariants.get("post_exposure_singleton_phase_i_policy")
            != (
                "phase_i_on_guarded_singletons_from_retained_"
                "macro_shortlist_v1"
            )
        )
        or (
            PHASE0_VARIANT == "global_singleton"
            and (
                invariants.get("phase_i_candidate_supply")
                != "global_guarded_singleton_pool_v1"
                or invariants.get("phase_i_candidate_visibility")
                != "all_executable_candidates_v1"
                or invariants.get("phase_ii_candidate_exposure")
                != "identity_on_retained_singletons_v1"
            )
        )
        or PHASE0_VARIANT not in {"macro", "global_singleton"}
        or not isinstance(lineage, Mapping)
    ):
        raise PackageContractError("Typed gradient-Phase-0 protocol drifted.")
    authority = _mint_bundle_protocol_materialization_authority(
        receipt,
        source_lock_refs=protocol.source_locks,
        protocol_sha256=protocol.sha256,
    )
    protocol = _attach_validated_bundle_protocol_authority(protocol, authority)
    return protocol, _problem_from_protocol(protocol)


def _prepare(
    job_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Any,
    Any,
    tempfile.TemporaryDirectory[str],
]:
    job, manifest, payload, source_locks = _load_closed_job(job_path)
    temporary = tempfile.TemporaryDirectory(
        prefix=f"paper-i-phase3-qiskit-{job['execution_id']}."
    )
    try:
        source_root = Path(temporary.name) / "source"
        _extract_source(
            manifest=manifest,
            source_locks=source_locks,
            destination=source_root,
        )
        original = Path.cwd()
        os.chdir(source_root)
        try:
            _activate_source_root(source_root)
            protocol, problem = _load_protocol(
                job=job,
                payload=payload,
                source_locks=source_locks,
            )
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
            "schema": (
                "paper_i_pure_hubbard_page12_fullnoise_"
                "worker_preflight_v1"
            ),
            "status": "passed",
            "execution_id": job["execution_id"],
            "job_spec_sha256": job["sha256"],
            "package_manifest_sha256": manifest["sha256"],
            "protocol_sha256": protocol.sha256,
            "route_contract_sha256": protocol.route_contract["sha256"],
            "source_archive_import_isolated": True,
            "fresh_start": True,
            "target_horizon": job["target_horizon"],
            "scientific_execution_performed": False,
        }
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _execute(
    *,
    protocol: Any,
    problem: Any,
    staging: Path,
    maximum_rounds: int,
) -> tuple[Any, int]:
    from pipelines.static_adapt.ra_adapt import (
        RAAdaptOperationalControls,
        run_ra_adapt,
    )
    from pipelines.static_adapt.sr_snake import (
        CheckpointObservation,
        EstimatorLedgerObservation,
        FreshStart,
        SRObservationPolicy,
    )

    (staging / "checkpoints").mkdir(parents=True, exist_ok=False)
    (staging / "result").mkdir(parents=True, exist_ok=False)
    (staging / "summary").mkdir(parents=True, exist_ok=False)
    controls = RAAdaptOperationalControls(
        maximum_controller_rounds=maximum_rounds,
        resume=FreshStart(),
        observation=SRObservationPolicy(
            checkpoint=CheckpointObservation(
                path=staging / "checkpoints/current.json",
                every_controller_rounds=1,
                keep_history_tail=100,
            ),
            estimator_ledger=EstimatorLedgerObservation(
                path=staging / "result/estimator_ledger.json"
            ),
            resource_rounds=(maximum_rounds,),
        ),
    )
    result = run_ra_adapt(
        problem,
        protocol,
        operational_controls=controls,
    )
    rounds = len(result.run.accepted_trajectory)
    if (
        result.protocol.sha256 != protocol.sha256
        or not 1 <= rounds <= maximum_rounds
        or not (staging / "checkpoints/current.json").is_file()
        or not (staging / "result/estimator_ledger.json").is_file()
    ):
        raise PackageContractError("Scientific execution closure failed.")
    return result, rounds


def _publish_staging(staging: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise PackageContractError("Worker output destination already exists.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sibling = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.publish-",
            dir=destination.parent,
        )
    )
    try:
        shutil.copytree(staging, sibling, dirs_exist_ok=True)
        os.rename(sibling, destination)
    except BaseException:
        shutil.rmtree(sibling, ignore_errors=True)
        raise


def _scheduler_identity(
    *,
    execution_id: str,
    cluster_id: int,
    proc_id: int,
    attempt_ordinal: int,
) -> dict[str, Any]:
    if (
        isinstance(cluster_id, bool)
        or isinstance(proc_id, bool)
        or isinstance(attempt_ordinal, bool)
        or cluster_id < 1
        or proc_id < 0
        or attempt_ordinal < 1
    ):
        raise PackageContractError("Scheduler attempt identity is invalid.")
    return {
        "schema": "paper_i_chtc_scheduler_attempt_identity_v1",
        "cluster_id": cluster_id,
        "proc_id": proc_id,
        "attempt_ordinal": attempt_ordinal,
        "terminal_output_archive": (
            f"transfer/{execution_id}__{cluster_id}__{proc_id}.tar.gz"
        ),
        "transfer_policy": "on_exit_only_v1",
    }


def run_cell(
    *,
    job_path: Path,
    authorization_path: Path,
    output_dir: Path,
    receipt_path: Path,
    scheduler_cluster_id: int,
    scheduler_proc_id: int,
    scheduler_attempt_ordinal: int,
) -> dict[str, Any]:
    job, manifest, protocol, problem, temporary = _prepare(job_path)
    try:
        scheduler = _scheduler_identity(
            execution_id=str(job["execution_id"]),
            cluster_id=scheduler_cluster_id,
            proc_id=scheduler_proc_id,
            attempt_ordinal=scheduler_attempt_ordinal,
        )
        authority = _validate_authorization(
            authorization_path,
            job=job,
            manifest=manifest,
        )
        if (
            output_dir.exists()
            or output_dir.is_symlink()
            or receipt_path.exists()
            or receipt_path.is_symlink()
            or output_dir.name != job["execution_id"]
            or output_dir.parent.name != "runs"
        ):
            raise PackageContractError("Worker destination already exists.")
        source_root = Path(temporary.name) / "source"
        staging = Path(temporary.name) / "cell_output"
        staging.mkdir()
        original = Path.cwd()
        os.chdir(source_root)
        try:
            result, rounds = _execute(
                protocol=protocol,
                problem=problem,
                staging=staging,
                maximum_rounds=int(job["target_horizon"]),
            )
        finally:
            os.chdir(original)
        _write_json(staging / "result/result.json", result.to_dict())
        if result.run.paper_i_summary is None:
            raise PackageContractError(
                "Paper-I summary is required by the expected-artifact contract."
            )
        _write_json(
            staging / "summary/summary.json",
            result.run.paper_i_summary.to_dict(),
        )
        expected_paths = _expected_artifact_paths(job)
        preliminary = {
            role: {
                "path": str(job["expected_run_artifacts"][role]["path"]),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for role, relative in expected_paths.items()
            if role != "execution_manifest"
            for path in (staging / relative,)
            if path.is_file()
        }
        if set(preliminary) != set(expected_paths).difference(
            {"execution_manifest"}
        ):
            raise PackageContractError(
                "Required scientific output artifact is absent."
            )
        execution_manifest = digested(
            {
                "schema": (
                    "paper_i_pure_hubbard_page12_fullnoise_"
                    "execution_manifest_v1"
                ),
                "status": "passed",
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "execution_id": job["execution_id"],
                "job_spec_sha256": job["sha256"],
                "authorization_sha256": authority["sha256"],
                "protocol_sha256": protocol.sha256,
                "route_contract_sha256": protocol.route_contract["sha256"],
                "target_horizon": job["target_horizon"],
                "controller_rounds_completed": rounds,
                "fresh_start": True,
                "source_checkpoint_consumed": False,
                "scheduler_attempt": scheduler,
                "output_payloads": preliminary,
            }
        )
        _write_json(
            staging / expected_paths["execution_manifest"],
            execution_manifest,
        )
        if any(
            not (staging / relative).is_file()
            for relative in expected_paths.values()
        ):
            raise PackageContractError(
                "Expected-artifact fulfillment is incomplete."
            )
        _publish_staging(staging, output_dir)
        receipt = digested(
            {
                "schema": (
                    "paper_i_pure_hubbard_page12_fullnoise_"
                    "worker_receipt_v1"
                ),
                "status": "passed",
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "execution_id": job["execution_id"],
                "job_spec_sha256": job["sha256"],
                "authorization_sha256": authority["sha256"],
                "execution_manifest_sha256": execution_manifest["sha256"],
                "controller_rounds_completed": rounds,
                "fresh_start": True,
                "scheduler_attempt": scheduler,
                "artifacts": [
                    {
                        "path": (
                            PurePosixPath("runs")
                            / job["execution_id"]
                            / path.relative_to(output_dir)
                        ).as_posix(),
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for path in sorted(output_dir.rglob("*"))
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
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--execution-authorization", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--scheduler-cluster-id", type=int)
    parser.add_argument("--scheduler-proc-id", type=int)
    parser.add_argument("--scheduler-attempt-ordinal", type=int)
    args = parser.parse_args()
    try:
        if args.preflight:
            if any(
                value is not None
                for value in (
                    args.execution_authorization,
                    args.output_dir,
                    args.receipt,
                    args.scheduler_cluster_id,
                    args.scheduler_proc_id,
                    args.scheduler_attempt_ordinal,
                )
            ):
                raise PackageContractError(
                    "Preflight accepts no execution destinations."
                )
            payload = preflight(args.job.resolve())
        else:
            if any(
                value is None
                for value in (
                    args.execution_authorization,
                    args.output_dir,
                    args.receipt,
                    args.scheduler_cluster_id,
                    args.scheduler_proc_id,
                    args.scheduler_attempt_ordinal,
                )
            ):
                raise PackageContractError(
                    "Execution requires authorization and destinations."
                )
            payload = run_cell(
                job_path=args.job.resolve(),
                authorization_path=args.execution_authorization.resolve(),
                output_dir=args.output_dir.resolve(),
                receipt_path=args.receipt.resolve(),
                scheduler_cluster_id=args.scheduler_cluster_id,
                scheduler_proc_id=args.scheduler_proc_id,
                scheduler_attempt_ordinal=args.scheduler_attempt_ordinal,
            )
    except (
        OSError,
        PackageContractError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(payload).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
