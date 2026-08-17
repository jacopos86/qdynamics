#!/usr/bin/env python3
"""Execute the real P3 and source-extracted P4 numerical witnesses."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
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
    ALGORITHM_ID,
    APPLICATION_SOURCE_LOCK_KEY,
    BUNDLE_ID,
    CANDIDATE_REPRESENTATION,
    CELL_ROWS,
    PACKAGE_ID,
    P3_RECEIPT_SCHEMA,
    P4_RECEIPT_SCHEMA,
    RESOURCE_WEIGHTING_SCOPE,
    TARGET_HORIZON,
    PackageContractError,
    canonical_json_bytes,
    canonical_sha256,
    digested,
    execution_id,
    load_json,
    repo_root_from_script,
    safe_relative_path,
    sha256_file,
    verify_self_digest,
)


REPO_ROOT = repo_root_from_script(__file__)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _numerical_evidence(
    result: Any,
    *,
    witness_execution_id: str,
) -> dict[str, Any]:
    expected_witness = execution_id(1.5, "low")
    if witness_execution_id != expected_witness:
        raise PackageContractError(
            "Numerical evidence counts are scoped only to the U=1.5 low-noise "
            "one-round witness."
        )
    trajectory = list(result.run.accepted_trajectory)
    replay = list(result.run.scientific_replay)
    receipts = result.scientific_receipts
    noise_receipt = (
        receipts.get("controller_noise")
        if isinstance(receipts, Mapping)
        else None
    )
    accepted_rounds = (
        receipts.get("accepted_round_receipts")
        if isinstance(receipts, Mapping)
        else None
    )
    if (
        len(trajectory) != 1
        or len(replay) != 1
        or int(result.run.stop.completed_controller_rounds) != 1
        or not isinstance(noise_receipt, Mapping)
        or not isinstance(accepted_rounds, list)
        or len(accepted_rounds) != 1
        or not isinstance(accepted_rounds[0], Mapping)
    ):
        raise PackageContractError("Numerical witness did not accept one round.")
    state = trajectory[0]
    replay_row = replay[0]
    round_row = dict(accepted_rounds[0])
    controller = round_row.get("controller_noise")
    runtime_delta = (
        controller.get("runtime_delta")
        if isinstance(controller, Mapping)
        else None
    )
    accepted_refit = getattr(replay_row, "accepted_refit", None)
    phase3 = getattr(getattr(replay_row, "phase", None), "phase3", None)
    if (
        not isinstance(controller, Mapping)
        or not isinstance(runtime_delta, Mapping)
        or noise_receipt.get("schema")
        != "paper_i_pure_hubbard_controller_noise_receipt_v1"
        or noise_receipt.get("candidate_gradient_scoring") != "noisy"
        or noise_receipt.get("powell_refit_objective") != "noisy"
        or noise_receipt.get("geometry_and_gram") != "exact"
        or noise_receipt.get("reported_energy") != "exact_diagnostic"
        or noise_receipt.get("same_circuit_incumbent") is not True
        or noise_receipt.get("optimizer_evaluation_order") != "serial_v1"
        or noise_receipt.get("candidate_record_cache")
        != "off_fail_closed_v1"
        or int(noise_receipt.get("accepted_round_count", -1)) != 1
        or runtime_delta.get("schema")
        != "paper_i_pure_hubbard_controller_noise_transition_delta_v1"
        or int(getattr(state, "controller_round", -1)) != 1
        or int(getattr(replay_row, "controller_round", -1)) != 1
        or getattr(accepted_refit, "policy", None) in {None, ""}
        or getattr(accepted_refit, "full_ansatz", None) is not True
        or int(getattr(accepted_refit, "supported_rank", 0)) <= 0
        or getattr(phase3, "coordinate_scope", None) in {None, ""}
        or int(getattr(phase3, "supported_rank", 0)) <= 0
    ):
        raise PackageContractError(
            "Numerical witness lacks the noisy-gradient/Powell dual-energy receipt."
        )
    exact_energy = float(getattr(state, "energy"))
    controller_energy = float(noise_receipt["final_controller_energy"])
    if (
        float(controller["exact_diagnostic_energy_after"]) != exact_energy
        or float(noise_receipt["final_exact_diagnostic_energy"]) != exact_energy
        or controller_energy != float(controller["controller_energy_after"])
        or controller_energy == exact_energy
    ):
        raise PackageContractError(
            "Numerical witness collapsed controller and exact diagnostic energy."
        )

    raw_trace_rows = noise_receipt.get("evaluation_records")
    delta_rows = runtime_delta.get("evaluation_records_delta")
    value_noise = noise_receipt.get("value_noise")
    compiled_receipts = noise_receipt.get("compiled_noise_receipts")
    delta_compiled_receipts = runtime_delta.get(
        "compiled_noise_receipts_delta"
    )
    rng_state_after = runtime_delta.get("rng_state_after")
    if (
        not isinstance(raw_trace_rows, list)
        or not raw_trace_rows
        or raw_trace_rows != delta_rows
        or int(noise_receipt.get("evaluation_count", -1))
        != len(raw_trace_rows)
        or int(runtime_delta.get("evaluation_count_before", -1)) != 0
        or int(runtime_delta.get("evaluation_count_after", -1))
        != len(raw_trace_rows)
        or noise_receipt.get("evaluation_records_sha256")
        != canonical_sha256(raw_trace_rows)
        or runtime_delta.get("evaluation_records_delta_sha256")
        != canonical_sha256(raw_trace_rows)
        or runtime_delta.get("cumulative_evaluation_records_sha256")
        != canonical_sha256(raw_trace_rows)
        or not isinstance(value_noise, Mapping)
        or value_noise.get("model") != "gaussian_iid_v1"
        or int(value_noise.get("draw_count", -1)) != len(raw_trace_rows)
        or not isinstance(rng_state_after, Mapping)
        or value_noise.get("rng_state") != rng_state_after
        or runtime_delta.get("rng_state_after_sha256")
        != canonical_sha256(rng_state_after)
        or not isinstance(compiled_receipts, Mapping)
        or not compiled_receipts
        or compiled_receipts != delta_compiled_receipts
        or int(
            runtime_delta.get("compiled_noise_receipt_count_before", -1)
        )
        != 0
        or int(
            runtime_delta.get("compiled_noise_receipt_count_after", -1)
        )
        != len(compiled_receipts)
        or noise_receipt.get("compiled_noise_receipts_sha256")
        != canonical_sha256(compiled_receipts)
        or runtime_delta.get("compiled_noise_receipts_delta_sha256")
        != canonical_sha256(compiled_receipts)
        or runtime_delta.get("cumulative_compiled_noise_receipts_sha256")
        != canonical_sha256(compiled_receipts)
        or runtime_delta.get("noise_contract_sha256")
        != noise_receipt.get("noise_contract_sha256")
        or any(
            not isinstance(value, Mapping)
            or value.get("schema")
            != "paper_i_pure_hubbard_controller_noise_evaluation_v1"
            or value.get("evaluation_ordinal") != index
            or not isinstance(value.get("consumer_scope"), str)
            or not isinstance(value.get("stage"), str)
            or not isinstance(value.get("value_noise"), Mapping)
            or value["value_noise"].get("model") != "gaussian_iid_v1"
            or value["value_noise"].get("draw_index_start") != index - 1
            or value["value_noise"].get("draw_index_stop") != index
            or value["value_noise"].get("n_draws") != 1
            or value.get("parameterized_plan_digest") in {None, ""}
            or value.get("parameterized_plan_digest") not in compiled_receipts
            or value.get("compiled_noise_receipt_sha256") in {None, ""}
            or value.get("compiled_noise_receipt_sha256")
            != compiled_receipts[value["parameterized_plan_digest"]].get(
                "sha256"
            )
            for index, value in enumerate(raw_trace_rows, start=1)
        )
        or any(
            not isinstance(receipt, Mapping)
            or receipt.get("schema")
            != "paper_i_pure_hubbard_compiled_noise_plan_receipt_v1"
            or receipt.get("parameterized_plan_digest") != digest
            or int(receipt.get("synthetic_coherent", {}).get("inserted_count", 0))
            <= 0
            or receipt.get("compile_signature", {}).get(
                "synthetic_coherent_inserted_after_transpile"
            )
            is not True
            for digest, receipt in compiled_receipts.items()
        )
    ):
        raise PackageContractError(
            "Numerical witness lacks ordered controller-noise evaluation records."
        )
    trace_rows = [dict(value) for value in raw_trace_rows]
    gradient_rows = [
        dict(value)
        for value in trace_rows
        if "gradient" in str(value["consumer_scope"]).lower()
        and value.get("probe_sign") in {"plus", "minus"}
        and isinstance(value.get("candidate_label"), str)
    ]
    depth_opt_rows = [
        dict(value)
        for value in trace_rows
        if value.get("probe_sign") is None
        and value.get("stage") == "depth_opt"
        and value.get("consumer_scope") == "depth_opt"
    ]
    incumbent_rows = [
        dict(value)
        for value in trace_rows
        if value.get("probe_sign") is None
        and value.get("stage") == "accepted_refit_same_circuit_incumbent"
        and value.get("consumer_scope")
        == "accepted_refit_same_circuit_incumbent"
    ]
    if not gradient_rows or not depth_opt_rows or not incumbent_rows:
        raise PackageContractError(
            "Per-evaluation controller-noise traces do not cover gradient and Powell."
        )
    gradient_pairs: dict[tuple[str, str, int], set[str]] = {}
    for value in gradient_rows:
        scope = str(value["consumer_scope"])
        sign = str(value["probe_sign"])
        suffix = f":{sign}"
        if not scope.endswith(suffix):
            raise PackageContractError(
                "Noisy finite-difference trace has an invalid signed scope."
            )
        key = (
            str(value["stage"]),
            str(value["candidate_label"]),
            int(value["actual_insertion_position"]),
        )
        gradient_pairs.setdefault(key, set()).add(sign)
    phase0_pairs = {
        key for key, signs in gradient_pairs.items()
        if key[0] == "phase0_global_singleton_gradient_surface"
        and signs == {"plus", "minus"}
    }
    phase2_pairs = {
        key for key, signs in gradient_pairs.items()
        if key[0] == "phase2" and signs == {"plus", "minus"}
    }
    scored = round_row.get("scored_insertion_position_population")
    phases = scored.get("phases") if isinstance(scored, Mapping) else None
    phase2_population = (
        phases[1].get("records")
        if isinstance(phases, list)
        and len(phases) == 3
        and isinstance(phases[1], Mapping)
        else None
    )
    phase3_population = (
        phases[2].get("records")
        if isinstance(phases, list)
        and len(phases) == 3
        and isinstance(phases[2], Mapping)
        else None
    )
    phase3_reused_pairs = {
        ("phase2", str(value["pool_label"]), int(value["insertion_position"]))
        for value in phase3_population or ()
        if isinstance(value, Mapping)
    }
    stage_counts = {
        stage: sum(value["stage"] == stage for value in trace_rows)
        for stage in sorted({str(value["stage"]) for value in trace_rows})
    }
    if (
        len(trace_rows) != 144
        or len(compiled_receipts) != 22
        or stage_counts
        != {
            "accepted_refit_same_circuit_incumbent": 1,
            "depth_opt": 55,
            "phase0_global_singleton_gradient_surface": 44,
            "phase2": 44,
        }
        or len(phase0_pairs) != 22
        or len(phase2_pairs) != 22
        or not isinstance(phase2_population, list)
        or not isinstance(phase3_population, list)
        or not phase3_population
        or not phase3_reused_pairs.issubset(phase2_pairs)
        or any(value["stage"] == "phase3" for value in trace_rows)
    ):
        raise PackageContractError(
            "Noisy Phase-0/II gradients do not close through Phase-III reuse."
        )
    return {
        "evidence_count_scope": (
            "u1p5_low_noise_one_controller_round_witness_only_v1"
        ),
        "evidence_count_execution_id": witness_execution_id,
        "controller_energy_after": controller_energy,
        "exact_diagnostic_energy_after": exact_energy,
        "value_noise_draw_count": int(value_noise["draw_count"]),
        "controller_noise_evaluation_count": len(trace_rows),
        "per_evaluation_trace_count": len(trace_rows),
        "noisy_gradient_trace_count": len(gradient_rows),
        "noisy_phase0_gradient_pair_count": len(phase0_pairs),
        "noisy_phase2_gradient_pair_count": len(phase2_pairs),
        "phase3_reused_noisy_phase2_gradient_count": len(
            phase3_reused_pairs
        ),
        "noisy_powell_depth_opt_trace_count": len(depth_opt_rows),
        "noisy_powell_incumbent_trace_count": len(incumbent_rows),
        "controller_noise_stage_counts": stage_counts,
        "compiled_noise_plan_count": len(compiled_receipts),
        "phase3_no_redundant_noise_oracle_call": True,
        "real_noisy_gradient_probe_passed": True,
        "real_noisy_powell_probe_passed": True,
    }


def _controls(output_root: Path) -> Any:
    from pipelines.static_adapt.ra_adapt import RAAdaptOperationalControls
    from pipelines.static_adapt.sr_snake import (
        CheckpointObservation,
        EstimatorLedgerObservation,
        FreshStart,
        SRObservationPolicy,
    )

    return RAAdaptOperationalControls(
        maximum_controller_rounds=1,
        resume=FreshStart(),
        observation=SRObservationPolicy(
            checkpoint=CheckpointObservation(
                path=output_root / "current.json",
                every_controller_rounds=1,
                keep_history_tail=4,
            ),
            estimator_ledger=EstimatorLedgerObservation(
                path=output_root / "estimator_ledger.json"
            ),
        ),
    )


def _application_sources() -> dict[str, str]:
    from pipelines.static_adapt.ra_adapt.pure_hubbard_noise_page12 import (
        build_paper_i_pure_hubbard_noise_page12_problem,
        build_paper_i_pure_hubbard_noise_page12_request,
        paper_i_pure_hubbard_noise_page12_application_source_contract,
    )

    result: dict[str, str] = {}
    for u_value, noise_level, _noise_tuple in CELL_ROWS:
        problem = build_paper_i_pure_hubbard_noise_page12_problem(u=u_value)
        request = build_paper_i_pure_hubbard_noise_page12_request(
            noise_level=noise_level,
            maximum_controller_rounds=TARGET_HORIZON,
        )
        source = paper_i_pure_hubbard_noise_page12_application_source_contract(
            problem,
            request,
        )
        result[execution_id(u_value, noise_level)] = str(source["sha256"])
    return {key: result[key] for key in sorted(result)}


def run_p3(*, output: Path) -> dict[str, Any]:
    from pipelines.static_adapt.ra_adapt.bundles import (
        _implementation_source_inventory,
    )
    from pipelines.static_adapt.ra_adapt.contracts import (
        RA_ADAPT_PROTOCOL_SCHEMA_V1,
        RA_STAGED_SELECTOR_ID,
        _attach_validated_bundle_protocol_authority,
        _mint_bundle_protocol_materialization_authority,
        bundle_protocol_materialization_receipt,
    )
    from pipelines.static_adapt.ra_adapt.engine import (
        build_resolved_ra_protocol,
        run_ra_adapt,
    )
    from pipelines.static_adapt.ra_adapt.pure_hubbard_noise_page12 import (
        build_paper_i_pure_hubbard_noise_page12_problem,
        build_paper_i_pure_hubbard_noise_page12_request,
        paper_i_pure_hubbard_noise_page12_application_source_contract,
    )

    application_sources = _application_sources()
    representative_id = execution_id(1.5, "low")
    problem = build_paper_i_pure_hubbard_noise_page12_problem(u=1.5)
    request = build_paper_i_pure_hubbard_noise_page12_request(
        noise_level="low",
        maximum_controller_rounds=TARGET_HORIZON,
    )
    source = paper_i_pure_hubbard_noise_page12_application_source_contract(
        problem,
        request,
    )
    refs = {
        "source_locks_manifest_sha256": "1" * 64,
        "implementation_source_inventory_sha256": "2" * 64,
        "cell_source_lock_id": "p3_pure_hubbard_noise_witness",
        "cell_source_lock_sha256": "3" * 64,
        "ed_cutoff_reference_sha256": canonical_sha256(
            source["same_cutoff_exact_reference"]
        ),
        APPLICATION_SOURCE_LOCK_KEY: source["sha256"],
    }
    materialization = bundle_protocol_materialization_receipt(
        bundle_id="p3_pure_hubbard_noise_witness",
        bundle_manifest_sha256="4" * 64,
        source_locks_sha256="1" * 64,
        source_lock_refs=refs,
        cell_id=representative_id,
        source_lock_id="p3_pure_hubbard_noise_witness",
        protocol_schema=RA_ADAPT_PROTOCOL_SCHEMA_V1,
        algorithm_id=ALGORITHM_ID,
        candidate_representation=CANDIDATE_REPRESENTATION,
        selector_identity=RA_STAGED_SELECTOR_ID,
        active_gradient_policy=ACTIVE_GRADIENT_POLICY,
        resource_weighting_scope=RESOURCE_WEIGHTING_SCOPE,
    )
    authority = _mint_bundle_protocol_materialization_authority(
        materialization,
        source_lock_refs=refs,
    )
    protocol = build_resolved_ra_protocol(
        problem,
        request,
        materialization_authority=authority,
    )
    protocol = _attach_validated_bundle_protocol_authority(
        protocol,
        _mint_bundle_protocol_materialization_authority(
            materialization,
            source_lock_refs=refs,
            protocol_sha256=protocol.sha256,
        ),
    )
    with tempfile.TemporaryDirectory(prefix="paper-i-pure-hubbard-noise-p3-") as raw:
        result = run_ra_adapt(
            problem,
            protocol,
            operational_controls=_controls(Path(raw)),
        )
        evidence = _numerical_evidence(
            result,
            witness_execution_id=representative_id,
        )
    implementation = _implementation_source_inventory(REPO_ROOT)
    receipt = digested(
        {
            "schema": P3_RECEIPT_SCHEMA,
            "status": "passed",
            "package_id": PACKAGE_ID,
            "execution_mode": "real_host_numerical_witness_v1",
            "representative_execution_id": representative_id,
            "representative_protocol_sha256": protocol.sha256,
            "implementation_source_inventory_sha256": implementation["sha256"],
            "application_source_contract_sha256s": application_sources,
            "completed_controller_rounds": 1,
            "scientific_execution_performed": True,
            **evidence,
            "paper_facing_result_allowed": False,
            "execution_authorized": False,
            "submission_authorized": False,
        }
    )
    _write_json(output, receipt)
    return receipt


def _extract_source(destination: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source_manifest = load_json(
        PACKAGE_DIR / "source/source_archive_manifest.json",
        label="source archive manifest",
    )
    verify_self_digest(source_manifest, label="source archive manifest")
    archive_binding = source_manifest.get("archive")
    rows = source_manifest.get("members")
    if (
        source_manifest.get("status") != "passed"
        or not isinstance(archive_binding, Mapping)
        or not isinstance(rows, list)
    ):
        raise PackageContractError("Source archive manifest is incomplete.")
    archive_path = PACKAGE_DIR / safe_relative_path(
        archive_binding.get("path"),
        label="source archive path",
    )
    if (
        not archive_path.is_file()
        or archive_path.is_symlink()
        or sha256_file(archive_path) != archive_binding.get("sha256")
        or archive_path.stat().st_size != int(archive_binding.get("size_bytes", -1))
    ):
        raise PackageContractError("Source archive binding drifted.")
    declared = {
        safe_relative_path(row.get("path"), label="source member").as_posix(): row
        for row in rows
        if isinstance(row, Mapping)
    }
    if len(declared) != len(rows) or len(rows) != source_manifest.get("member_count"):
        raise PackageContractError("Source member closure drifted.")
    destination.mkdir(parents=True, exist_ok=False)
    observed: set[str] = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            relative = safe_relative_path(member.name, label="tar member").as_posix()
            row = declared.get(relative)
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
            target = destination / relative
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
    if observed != set(declared):
        raise PackageContractError("Source extraction is incomplete.")
    return source_manifest, dict(archive_binding)


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
        raise PackageContractError("P4 import escaped the source archive.") from exc


def run_p4(*, job_path: Path, output: Path) -> dict[str, Any]:
    job = load_json(job_path, label="P4 job")
    verify_self_digest(job, label="P4 job")
    if (
        job.get("package_id") != PACKAGE_ID
        or job.get("algorithm_id") != ALGORITHM_ID
        or int(job.get("target_horizon", -1)) != TARGET_HORIZON
    ):
        raise PackageContractError("P4 source job drifted.")
    protocol_path = PACKAGE_DIR / safe_relative_path(
        job.get("protocol_path"),
        label="P4 protocol path",
    )
    if (
        not protocol_path.is_file()
        or protocol_path.is_symlink()
        or sha256_file(protocol_path) != job.get("protocol_file_sha256")
    ):
        raise PackageContractError("P4 protocol binding drifted.")
    protocol_payload = load_json(protocol_path, label="P4 protocol")
    verify_self_digest(protocol_payload, label="P4 protocol")

    with tempfile.TemporaryDirectory(prefix="paper-i-pure-hubbard-noise-p4-") as raw:
        root = Path(raw)
        source_root = root / "source"
        source_manifest, archive_binding = _extract_source(source_root)
        original = Path.cwd()
        os.chdir(source_root)
        try:
            _activate_source_root(source_root)
            from pipelines.static_adapt.ra_adapt.contracts import (
                _attach_validated_bundle_protocol_authority,
                _mint_bundle_protocol_materialization_authority,
                resolved_ra_adapt_protocol_from_mapping,
            )
            from pipelines.static_adapt.ra_adapt.engine import run_ra_adapt
            from pipelines.static_adapt.ra_adapt.pure_hubbard_noise_page12 import (
                build_paper_i_pure_hubbard_noise_page12_problem,
            )

            protocol = resolved_ra_adapt_protocol_from_mapping(protocol_payload)
            materialization = protocol.bundle_materialization
            if materialization is None:
                raise PackageContractError("P4 protocol lacks materialization.")
            protocol = _attach_validated_bundle_protocol_authority(
                protocol,
                _mint_bundle_protocol_materialization_authority(
                    materialization,
                    source_lock_refs=protocol.source_locks,
                    protocol_sha256=protocol.sha256,
                ),
            )
            problem = build_paper_i_pure_hubbard_noise_page12_problem(
                u=float(job["u_over_t"])
            )
            (root / "outputs").mkdir()
            result = run_ra_adapt(
                problem,
                protocol,
                operational_controls=_controls(root / "outputs"),
            )
            evidence = _numerical_evidence(
                result,
                witness_execution_id=str(job["execution_id"]),
            )
            active_module = importlib.import_module(
                "pipelines.static_adapt.ra_adapt.pure_hubbard_noise_page12"
            )
            Path(str(active_module.__file__)).resolve().relative_to(
                source_root.resolve()
            )
        finally:
            os.chdir(original)
    receipt = digested(
        {
            "schema": P4_RECEIPT_SCHEMA,
            "status": "passed",
            "package_id": PACKAGE_ID,
            "bundle_id": BUNDLE_ID,
            "execution_mode": "source_extracted_packaged_numerical_witness_v1",
            "execution_id": job["execution_id"],
            "job_spec_sha256": job["sha256"],
            "protocol_sha256": job["protocol_sha256"],
            "source_archive_sha256": archive_binding["sha256"],
            "source_archive_manifest_sha256": source_manifest["sha256"],
            "completed_controller_rounds": 1,
            "source_locked_archive_validated": True,
            "source_locked_import_isolated": True,
            "scientific_execution_performed": True,
            **evidence,
            "paper_facing_result_allowed": False,
            "execution_authorized": False,
            "submission_authorized": False,
        }
    )
    _write_json(output, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("p3", "p4"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "p3":
            if args.job is not None:
                raise PackageContractError("P3 accepts no packaged job.")
            receipt = run_p3(output=args.output.resolve())
        else:
            if args.job is None:
                raise PackageContractError("P4 requires --job.")
            receipt = run_p4(
                job_path=args.job.resolve(),
                output=args.output.resolve(),
            )
    except (OSError, PackageContractError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(receipt).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
