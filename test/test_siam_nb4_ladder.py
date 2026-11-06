"""Transition tests for the pure SIAM Nb4 comparator-first ladder."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math

import pytest

from pipelines.time_dynamics import siam_nb4_ladder as ladder
from pipelines.time_dynamics.diagnostics.siam_offline_scoring import (
    SIAM_ACCURACY_POLICY_V2,
    SIAM_OFFLINE_SCORE_V3,
)
from pipelines.time_dynamics.diagnostics.siam_exact_reference_store import (
    SIAM_EXACT_REFERENCE_STORE_V2,
)
from pipelines.time_dynamics.diagnostics.siam_segment_stitching import (
    SIAM_NB4_STITCH_V1,
    SIAM_NB4_TIMES,
)
from pipelines.time_dynamics.paper_ii_runs import (
    SIAM_NB4_CONTINUATION_SEGMENT,
    SIAM_NB4_SCREEN_SEGMENT,
    build_siam_nb4_continuation_run,
    build_siam_nb4_screen_run,
)
from pipelines.time_dynamics.resume import (
    CONTROLLER_RESUME_SCHEMA_V1,
    RESUME_SCHEMA_V2,
)
from pipelines.time_dynamics.siam_activation_ladder import (
    SIAM_NB4_CANONICAL_LADDER_SHA256,
    SIAM_NB4_LEGACY_ACTIVATION_CUT_LADDER,
    SIAM_NB4_LEGACY_LADDER_SHA256,
    ladder_payload,
)
from pipelines.time_dynamics.siam_chtc_package import (
    SIAM_NB4_ACTIVATION_CUT_LADDER,
    SIAM_NB4_CHTC_ACTION_SCHEMA_V3,
    SIAM_NB4_CHTC_MANIFEST_SCHEMA_V3,
    SIAM_NB4_CHTC_RECEIPT_SCHEMA_V3,
    SIAM_NB4_RESUME_TERMINAL_PROOF_V1,
)
from src.quantum.siam_observables import SIAM_OBSERVABLES_V2
from src.quantum.siam_reference import SIAM_REFERENCE_V2


REVISION = "a" * 40
SEED_SHA256 = "b" * 64
INITIAL_CUT = SIAM_NB4_ACTIVATION_CUT_LADDER[0]
NEXT_CUT = SIAM_NB4_ACTIVATION_CUT_LADDER[1]
LEGACY_CUT = 1.0e-3


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _pretty_sha256(value: object) -> str:
    return hashlib.sha256(
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    ).hexdigest()


def _argv_sha256(argv: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(argv, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _action_id(
    arm: str, segment: str, cut: float, route_id: str = ladder.ROUTE_HISTORICAL
) -> str:
    return ladder.siam_nb4_action_id(
        arm=arm, segment=segment, activation_cut=cut, route_id=route_id
    )


def _score(
    arm: str,
    *,
    point_count: int,
    run_sha256: str,
    response: float = 0.02,
    infidelity: float = 2.0e-5,
    energy: float = 5.0e-4,
    primitive_responses: dict[str, float] | None = None,
) -> dict[str, object]:
    primitive_names = ("m_d", "D_d", "C_1", "C_2", "C_3", "C_4")
    responses = {
        name: float((primitive_responses or {}).get(name, response))
        for name in primitive_names
    }
    max_response = max(responses.values())
    broad = bool(
        max_response <= 0.15 and infidelity <= 1.0e-4 and energy <= 1.0e-3
    )
    individual = bool(
        max_response <= 0.05 and infidelity <= 1.0e-4 and energy <= 1.0e-3
    )
    normalized = {name: value / 0.05 for name, value in responses.items()}
    normalized["fidelity"] = infidelity / 1.0e-4
    correlation_mean = sum(normalized[f"C_{index}"] ** 2 for index in range(1, 5)) / 4.0
    burden = math.sqrt(
        0.25
        * (
            normalized["m_d"] ** 2
            + normalized["D_d"] ** 2
            + correlation_mean
            + normalized["fidelity"] ** 2
        )
    )
    pair_normalized = {**normalized, "energy": energy / 1.0e-3}
    times = [0.01 * index for index in range(point_count)]
    core: dict[str, object] = {
        "schema": SIAM_OFFLINE_SCORE_V3,
        "reporting_only": True,
        "accuracy_policy": {
            "schema": SIAM_ACCURACY_POLICY_V2,
            "energy_guard": "sector_spectral_width_normalized_max_drift",
            "energy_in_aggregate_burden": False,
            "energy_in_componentwise_matching": True,
        },
        "observable_contract": {
            "schema": SIAM_OBSERVABLES_V2,
            "reporting_role": "offline_derived_from_checkpoint_statevectors",
            "impurity_number": "N_d=n_d_up+n_d_down",
            "impurity_spin_square": "S_d2=3/4*(N_d-2*D_d)",
            "integrated_screening": "Sigma_r=S_d2+sum_{j<=r}C_j",
            "primitive_gate_set_unchanged": True,
        },
        "method": {
            "structural_policy": "avqds" if arm == ladder.ARM_AVQDS else "exchange",
            "comparison_fingerprint": "6" * 64,
            "physics_fingerprint": "7" * 64,
            "run_lock_schema": "paper_ii_run_lock_v4",
            "code_revision": REVISION,
        },
        "source": {
            "run_path": "/not-reopened/run.json",
            "seed_sha256": SEED_SHA256,
            "run_sha256": run_sha256,
            "seed_path": "/not-reopened/seed.json",
            "post_quench_hamiltonian_sha256": "2" * 64,
        },
        "reference": {
            "key": "shared-nb4-t5-reference",
            "manifest_sha256": "3" * 64,
            "data_sha256": "4" * 64,
            "store_schema": SIAM_EXACT_REFERENCE_STORE_V2,
            "reference_schema": SIAM_REFERENCE_V2,
            "observable_schema": SIAM_OBSERVABLES_V2,
            "propagator_schema": "siam_exact_sector_eigh_v1",
        },
        "grid": {
            "times": times,
            "ordered_values_sha256": _canonical_sha256(times),
            "point_count": point_count,
        },
        "thresholds": {
            "response_floor": 1.0e-12,
            "individual_response_target": 0.05,
            "broad_response_guard": 0.15,
            "mean_infidelity_ceiling": 1.0e-4,
            "relative_energy_drift_ceiling": 1.0e-3,
            "initial_state_infidelity_ceiling": 1.0e-12,
            "initial_energy_error_ceiling": 1.0e-10,
            "aggregate_burden_ratio_ceiling": 2.0,
            "componentwise_ratio_deadband": 0.10,
            "componentwise_ratio_ceiling": 2.0,
        },
        "primitive_response_metrics": {
            name: {
                "mean_absolute_error": value * 0.1,
                "response_scale": 0.1,
                "response_normalized_mean_absolute_error": value,
            }
            for name, value in responses.items()
        },
        "derived_screening_metrics": {
            f"Sigma_{radius}": {
                "mean_absolute_error": response * 0.1,
                "response_scale": 0.1,
                "response_normalized_mean_absolute_error": response,
            }
            for radius in range(1, 5)
        },
        "series": {
            "time": times,
            "observed": {},
            "exact": {},
            "absolute_error": {},
        },
        "diagnostics": {
            "energy": {
                "sector_spectral_width": 2.0,
                "max_absolute_drift": 2.0 * energy,
                "g_E": energy,
                "relative_spectral_drift_ceiling": 1.0e-3,
                "pass": energy <= 1.0e-3,
            },
            "initial_boundary": {
                "boundary_time": 0.0,
                "boundary_role": "immutable_seed_boundary",
                "exact_seed_equality_required": True,
                "state_infidelity": 0.0,
                "absolute_energy_error": 0.0,
                "pass": True,
            },
        },
        "summary": {
            "mean_infidelity": infidelity,
            "max_primitive_response_normalized_error": max_response,
            "aggregate_accuracy_burden": burden,
            "normalized_accuracy_components": normalized,
            "pair_normalized_accuracy_components": pair_normalized,
            "individual_target_pass": individual,
            "broad_response_guard_pass": broad,
            "fidelity_pass": infidelity <= 1.0e-4,
            "energy_conservation_pass": energy <= 1.0e-3,
            "classification": (
                "qualified"
                if individual
                else "candidate_broad_guard"
                if broad
                else "candidate_outside_broad_guard"
            ),
        },
    }
    constant_series = {
        "m_d": [1.0] * point_count,
        "D_d": [0.0] * point_count,
        "N_d": [1.0] * point_count,
        "S_d2": [0.75] * point_count,
        "I_d": [0.0] * point_count,
        **{f"C_{site}": [0.0] * point_count for site in range(1, 5)},
        **{f"Sigma_{site}": [0.75] * point_count for site in range(1, 5)},
    }
    core["series"]["observed"] = constant_series
    core["series"]["exact"] = json.loads(json.dumps(constant_series))
    core["series"]["absolute_error"] = {
        name: [0.0] * point_count for name in constant_series
    }
    return {**core, "score_sha256": _canonical_sha256(core)}


def _redigested_score_without_primitive(
    score: dict[str, object], name: str
) -> dict[str, object]:
    """Model an adversarial self-consistent score with one omitted gate."""

    payload = json.loads(json.dumps(score))
    payload.pop("score_sha256")
    payload["primitive_response_metrics"].pop(name)
    normalized = payload["summary"]["normalized_accuracy_components"]
    normalized.pop(name)
    payload["summary"]["pair_normalized_accuracy_components"].pop(name)
    primitive_values = [
        float(metric["response_normalized_mean_absolute_error"])
        for metric in payload["primitive_response_metrics"].values()
    ]
    payload["summary"]["max_primitive_response_normalized_error"] = max(
        primitive_values
    )
    correlations = [
        float(value) ** 2
        for component, value in normalized.items()
        if component.startswith("C_")
    ]
    payload["summary"]["aggregate_accuracy_burden"] = math.sqrt(
        0.25
        * (
            float(normalized["m_d"]) ** 2
            + float(normalized["D_d"]) ** 2
            + sum(correlations) / len(correlations)
            + float(normalized["fidelity"]) ** 2
        )
    )
    return {**payload, "score_sha256": _canonical_sha256(payload)}


def _raw_receipt(
    arm: str,
    segment: str,
    cut: float,
    *,
    package_tag: str = "",
    legacy_r00: bool = False,
    route_id: str = ladder.ROUTE_HISTORICAL,
) -> dict[str, object]:
    if legacy_r00:
        assert cut == LEGACY_CUT
        segment_token = "screen" if segment == ladder.SEGMENT_SCREEN else "continue"
        action_id = f"siam-nb4-{arm}-{segment_token}-r00-cut1em03"
        package_ladder = SIAM_NB4_LEGACY_ACTIVATION_CUT_LADDER
        package_index = 0
        package_ladder_sha256 = SIAM_NB4_LEGACY_LADDER_SHA256
    else:
        action_id = _action_id(arm, segment, cut, route_id)
        package_ladder = SIAM_NB4_ACTIVATION_CUT_LADDER
        package_index = package_ladder.index(cut)
        package_ladder_sha256 = SIAM_NB4_CANONICAL_LADDER_SHA256
    policy = "avqds" if arm == ladder.ARM_AVQDS else "exchange"
    package_id = f"pkg-{action_id}{package_tag}"
    seed_path = f"{package_id}/input/seed.json"
    trajectory_json = f"raw_outputs/{package_id}/{action_id}/run.json"
    if segment == ladder.SEGMENT_SCREEN:
        command = build_siam_nb4_screen_run(
            seed_path=seed_path,
            arm=policy,
            activation_cut=cut,
            output_json=trajectory_json,
            code_revision=REVISION,
        )
    else:
        command = build_siam_nb4_continuation_run(
            seed_path=seed_path,
            arm=policy,
            resume_from_run_json=f"{package_id}/input/resume/screen_run.json",
            activation_cut=cut,
            output_json=trajectory_json,
            code_revision=REVISION,
        )
    argv = list(command.argv())
    argv_sha256 = _argv_sha256(argv)
    source_rows = [
        {
            "path": "pipelines/time_dynamics/fixture.py",
            "kind": "file",
            "mode": 0o644,
            "bytes": 1,
            "sha256": "1" * 64,
            "link_target": None,
        },
        {
            "path": "src/quantum/fixture.py",
            "kind": "file",
            "mode": 0o644,
            "bytes": 1,
            "sha256": "2" * 64,
            "link_target": None,
        },
    ]
    source_tree = {
        "schema": "siam_source_tree_contract_v1",
        "source_revision": REVISION,
        "file_count": len(source_rows),
        "files_sha256": _canonical_sha256(source_rows),
        "files": source_rows,
    }
    action_contract = {
        "schema": SIAM_NB4_CHTC_ACTION_SCHEMA_V3,
        "package_id": package_id,
        "action_id": action_id,
        "source_revision": REVISION,
        "source_tree": source_tree,
        "arm": policy,
        "segment": segment,
        "activation_cut": cut,
        "activation_cut_ladder_index": package_index,
        "activation_cut_ladder_sha256": package_ladder_sha256,
        "seed_path": seed_path,
        "seed_sha256": SEED_SHA256,
        "trajectory_json": trajectory_json,
        "expected_run_id": command.run_id,
        "expected_runner_argv_sha256": argv_sha256,
        "expected_execution_contract": {
            "support_patch_config": {"fixture": "coherent"},
            "solve_repair_config": {"fixture": "coherent"},
            "candidate_pool_contract": {"fixture": "coherent"},
        },
        "exact_reference_blind": True,
        "compile_terminal_cost": False,
        "qiskit_compilation_requested": False,
    }
    action_sha256 = _pretty_sha256(action_contract)
    manifest_contract = {
        "schema": SIAM_NB4_CHTC_MANIFEST_SCHEMA_V3,
        "package_id": package_id,
        "action_id": action_id,
        "source_revision": REVISION,
        "action_json_sha256": action_sha256,
        "runner_argv_sha256": argv_sha256,
        "expected_run_id": command.run_id,
        "expected_output": trajectory_json,
        "source_tree_contract": "siam_source_tree_contract_v1",
        "source_tree_files_sha256": source_tree["files_sha256"],
        "source_tree_file_count": source_tree["file_count"],
        "activation_cut_ladder": list(ladder_payload(package_ladder)),
        "activation_cut_ladder_sha256": package_ladder_sha256,
        "files": {"action": {"sha256": action_sha256}},
    }
    segment_contract = (
        SIAM_NB4_SCREEN_SEGMENT
        if segment == ladder.SEGMENT_SCREEN
        else SIAM_NB4_CONTINUATION_SEGMENT
    )
    terminal_state_sha256 = "9" * 64
    resume_terminal_proof = {
        "schema": SIAM_NB4_RESUME_TERMINAL_PROOF_V1,
        "resume_schema": RESUME_SCHEMA_V2,
        "controller_schema": CONTROLLER_RESUME_SCHEMA_V1,
        "terminal_time": float(segment_contract.times[-1]),
        "terminal_checkpoint_index": len(segment_contract.times) - 1,
        "logical_coordinate_count": 0,
        "runtime_pauli_coordinate_count": 0,
        "previous_accepted_velocity_count": 0,
        "support_contract_sha256": "5" * 64,
        "theta_contract_sha256": "6" * 64,
        "controller_state_sha256": "7" * 64,
        "reconstructed_state_semantic_sha256": terminal_state_sha256,
        "terminal_state_semantic_sha256": terminal_state_sha256,
        "reconstructed_state_infidelity": 0.0,
    }
    return {
        "schema": SIAM_NB4_CHTC_RECEIPT_SCHEMA_V3,
        "status": "COMPLETED",
        "package_id": package_id,
        "action_id": action_id,
        "run_id": command.run_id,
        "arm": policy,
        "segment": segment,
        "activation_cut": cut,
        "source_revision": REVISION,
        "seed_path": seed_path,
        "seed_sha256": SEED_SHA256,
        "action_json_sha256": action_sha256,
        "package_manifest_sha256": _pretty_sha256(manifest_contract),
        "source_tree_files_sha256": source_tree["files_sha256"],
        "source_tree_file_count": source_tree["file_count"],
        "exact_reference_blind": True,
        "qiskit_compilation_requested": False,
        "runner_argv": argv,
        "runner_argv_sha256": argv_sha256,
        "trajectory_json": trajectory_json,
        "trajectory_sha256": ("d" if segment == ladder.SEGMENT_SCREEN else "e")
        * 64,
        "trajectory_bytes": 123 if segment == ladder.SEGMENT_SCREEN else 456,
        "action_contract": action_contract,
        "package_manifest_contract": manifest_contract,
        "resume_terminal_proof": resume_terminal_proof,
        "checks": {
            "source_revision": True,
            "uncured_local_subdivisions": True,
            "resume_schema": True,
            "trajectory_points": True,
            "ordered_time_grid": True,
            "initial_time": True,
            "terminal_time": True,
            "configuration_lock": True,
            "segment_initial_state": True,
            "segment_initial_energy": True,
            **(
                {"initial_seed_state": True, "initial_seed_energy": True}
                if segment == ladder.SEGMENT_SCREEN
                else {"resume_boundary_state": True, "resume_boundary_energy": True}
            ),
            "state_norm_sector": True,
            "resume_reconstruction": True,
            "resume_support_parameters": True,
            "resume_terminal_statevector": True,
            "controller_resume_reconstruction": True,
            "controller_resume_terminal_state": True,
        },
    }


def _stitch_proof(
    arm: str,
    cut: float,
    *,
    screen_receipt: dict[str, object],
    continuation_receipt: dict[str, object],
    stitched_sha256: str,
) -> dict[str, object]:
    policy = "avqds" if arm == ladder.ARM_AVQDS else "exchange"
    grid = {
        "t_initial": 0.0,
        "t_final": 5.0,
        "point_count": 501,
        "ordered_values_sha256": _canonical_sha256(list(SIAM_NB4_TIMES)),
    }
    protocol = {
        "seed_sha256": SEED_SHA256,
        "structural_policy": policy,
        "code_revision": REVISION,
    }
    stitch_sources = [
        {
            "ordinal": 0,
            "run_path": "/not-reopened/screen.json",
            "run_sha256": str(screen_receipt["trajectory_sha256"]),
            "resume_payload_sha256": "8" * 64,
            "t_initial": float(SIAM_NB4_SCREEN_SEGMENT.times[0]),
            "t_final": float(SIAM_NB4_SCREEN_SEGMENT.times[-1]),
            "point_count": len(SIAM_NB4_SCREEN_SEGMENT.times),
            "canonical_start_index": 0,
            "canonical_stop_index": 100,
            "activation_l2_cut": cut,
            "code_revision": REVISION,
        },
        {
            "ordinal": 1,
            "run_path": "/not-reopened/continuation.json",
            "run_sha256": str(continuation_receipt["trajectory_sha256"]),
            "resume_payload_sha256": "9" * 64,
            "t_initial": float(SIAM_NB4_CONTINUATION_SEGMENT.times[0]),
            "t_final": float(SIAM_NB4_CONTINUATION_SEGMENT.times[-1]),
            "point_count": len(SIAM_NB4_CONTINUATION_SEGMENT.times),
            "canonical_start_index": 100,
            "canonical_stop_index": 500,
            "activation_l2_cut": cut,
            "code_revision": REVISION,
        },
    ]
    stitch_core = {
        "schema": SIAM_NB4_STITCH_V1,
        "seed_path": "/not-reopened/seed.json",
        "seed_sha256": SEED_SHA256,
        "source_segments": stitch_sources,
        "protocol": protocol,
        "protocol_sha256": _canonical_sha256(protocol),
        "canonical_grid": grid,
    }
    stitch = {**stitch_core, "stitch_sha256": _canonical_sha256(stitch_core)}
    proof_sources = []
    for ordinal, (source, segment, receipt) in enumerate(
        (
            (stitch_sources[0], ladder.SEGMENT_SCREEN, screen_receipt),
            (stitch_sources[1], ladder.SEGMENT_CONTINUATION, continuation_receipt),
        )
    ):
        receipt_utf8 = json.dumps(
            receipt, indent=2, sort_keys=True, allow_nan=False
        ) + "\n"
        proof_sources.append(
            {
                "ordinal": ordinal,
                "segment": segment,
                "trajectory_sha256": source["run_sha256"],
                "trajectory_bytes": int(receipt["trajectory_bytes"]),
                "t_initial": source["t_initial"],
                "t_final": source["t_final"],
                "point_count": source["point_count"],
                "canonical_start_index": source["canonical_start_index"],
                "canonical_stop_index": source["canonical_stop_index"],
                "activation_l2_cut": source["activation_l2_cut"],
                "source_revision": source["code_revision"],
                "completion_receipt_path": f"/not-reopened/{segment}.json",
                "completion_receipt_file_bytes": len(receipt_utf8.encode("utf-8")),
                "completion_receipt_file_sha256": hashlib.sha256(
                    receipt_utf8.encode("utf-8")
                ).hexdigest(),
                "completion_receipt_file_utf8": receipt_utf8,
                "completion_receipt_document_sha256": _canonical_sha256(receipt),
                "completion_receipt": receipt,
            }
        )
    canonical_document_sha256 = "a" * 64
    core = {
        "schema": ladder.SIAM_NB4_STITCH_COMPLETION_V1,
        "status": "COMPLETED",
        "problem_family": "siam",
        "arm": policy,
        "stitched_trajectory": {
            "path": "/not-reopened/stitched.json",
            "sha256": stitched_sha256,
            "bytes": 999,
            "canonical_document_sha256": canonical_document_sha256,
            "source_rederived_document_sha256": canonical_document_sha256,
        },
        "seed": {"path": stitch["seed_path"], "sha256": SEED_SHA256},
        "canonical_grid": grid,
        "stitch": stitch,
        "source_segments": proof_sources,
        "checks": {
            "stitched_trajectory_digest_bound": True,
            "stitched_payload_rederived_from_sources": True,
            "embedded_stitch_digest_verified": True,
            "canonical_grid_verified": True,
            "seed_binding_verified": True,
            "arm_binding_verified": True,
            "all_source_segment_digests_bound": True,
            "all_source_completion_receipts_positive_v3": True,
        },
    }
    return {**core, "completion_sha256": _canonical_sha256(core)}


def _record(
    arm: str,
    segment: str,
    cut: float,
    *,
    state: str = ladder.STATE_COMPLETED,
    response: float = 0.02,
    infidelity: float = 2.0e-5,
    energy: float = 5.0e-4,
    primitive_responses: dict[str, float] | None = None,
    scored: bool = True,
    failure_reason: str = "runner failed",
    legacy_r00: bool = False,
    route_id: str = ladder.ROUTE_HISTORICAL,
    exchange_route_bundle: str | None = None,
) -> ladder.SIAMNb4ActionRecord:
    if legacy_r00:
        assert cut == LEGACY_CUT
        token = "screen" if segment == ladder.SEGMENT_SCREEN else "continue"
        action_id = f"siam-nb4-{arm}-{token}-r00-cut1em03"
        screen_id = f"siam-nb4-{arm}-screen-r00-cut1em03"
    else:
        action_id = _action_id(arm, segment, cut, route_id)
        screen_id = _action_id(arm, ladder.SEGMENT_SCREEN, cut, route_id)
    if state != ladder.STATE_COMPLETED:
        return ladder.SIAMNb4ActionRecord(
            action_id=action_id,
            arm=arm,
            segment=segment,
            activation_cut=cut,
            state=state,
            route_id=route_id,
            resume_from_action_id=(
                screen_id if segment == ladder.SEGMENT_CONTINUATION else None
            ),
            failure_reason=(failure_reason if state == ladder.STATE_FAILED else ""),
        )

    receipt = _raw_receipt(
        arm, segment, cut, legacy_r00=legacy_r00, route_id=route_id
    )
    score = None
    proof = None
    if scored:
        run_sha256 = (
            str(receipt["trajectory_sha256"])
            if segment == ladder.SEGMENT_SCREEN
            else "f" * 64
        )
        score = _score(
            arm,
            point_count=(101 if segment == ladder.SEGMENT_SCREEN else 501),
            run_sha256=run_sha256,
            response=response,
            infidelity=infidelity,
            energy=energy,
            primitive_responses=primitive_responses,
        )
        if exchange_route_bundle is not None:
            score["method"]["exchange_route_bundle"] = exchange_route_bundle
            score["method"]["exchange_route_contract_sha256"] = "5" * 64
            score.pop("score_sha256")
            score = {**score, "score_sha256": _canonical_sha256(score)}
        if segment == ladder.SEGMENT_CONTINUATION:
            screen_receipt = _raw_receipt(
                arm,
                ladder.SEGMENT_SCREEN,
                cut,
                legacy_r00=legacy_r00,
                route_id=route_id,
            )
            proof = _stitch_proof(
                arm,
                cut,
                screen_receipt=screen_receipt,
                continuation_receipt=receipt,
                stitched_sha256=run_sha256,
            )
    return ladder.SIAMNb4ActionRecord(
        action_id=action_id,
        arm=arm,
        segment=segment,
        activation_cut=cut,
        state=state,
        route_id=route_id,
        resume_from_action_id=(
            screen_id if segment == ladder.SEGMENT_CONTINUATION else None
        ),
        completion_receipt=receipt,
        score=score,
        score_completion_proof=proof,
    )


def _screen(arm: str, cut: float, **kwargs) -> ladder.SIAMNb4ActionRecord:
    return _record(arm, ladder.SEGMENT_SCREEN, cut, **kwargs)


def _continuation(arm: str, cut: float, **kwargs) -> ladder.SIAMNb4ActionRecord:
    return _record(arm, ladder.SEGMENT_CONTINUATION, cut, **kwargs)


def _pair_history(
    *,
    avqds_cut: float = INITIAL_CUT,
    apm_cut: float = INITIAL_CUT,
    avqds_response: float = 0.04,
    apm_response: float = 0.01,
    avqds_infidelity: float = 2.0e-5,
    apm_infidelity: float = 2.0e-5,
    avqds_primitives: dict[str, float] | None = None,
    apm_primitives: dict[str, float] | None = None,
    pair_admissible: bool = False,
) -> list[ladder.SIAMNb4ActionRecord]:
    if pair_admissible:
        avqds_response = apm_response = 0.02
        avqds_infidelity = apm_infidelity = 2.0e-5
        avqds_primitives = apm_primitives = None
    records: list[ladder.SIAMNb4ActionRecord] = []
    for arm, cut, response, infidelity, primitive_responses in (
        (
            ladder.ARM_AVQDS,
            avqds_cut,
            avqds_response,
            avqds_infidelity,
            avqds_primitives,
        ),
        (
            ladder.ARM_FULL_APM,
            apm_cut,
            apm_response,
            apm_infidelity,
            apm_primitives,
        ),
    ):
        max_index = SIAM_NB4_ACTIVATION_CUT_LADDER.index(cut)
        for index in range(max_index + 1):
            rung = SIAM_NB4_ACTIVATION_CUT_LADDER[index]
            records.append(
                _screen(
                    arm,
                    rung,
                    # Every earlier rung carries the measured broad-guard miss
                    # that causally opened the next exact rung.
                    response=(0.16 if index < max_index else 0.02),
                )
            )
        records.append(
            _continuation(
                arm,
                cut,
                response=response,
                infidelity=infidelity,
                primitive_responses=primitive_responses,
            )
        )
    return records


def _run_actions(plan: ladder.SIAMNb4LadderPlan):
    return [row for row in plan.next_actions if row.kind == ladder.KIND_RUN]


def test_initial_frontier_is_only_avqds_at_common_cut() -> None:
    plan = ladder.plan_siam_nb4_ladder([])
    assert len(plan.next_actions) == 1
    action = plan.next_actions[0]
    assert (action.arm, action.segment, action.activation_cut) == (
        ladder.ARM_AVQDS,
        ladder.SEGMENT_SCREEN,
        INITIAL_CUT,
    )
    assert action.exact_reference_blind is True
    assert action.qiskit_compilation_requested is False
    assert plan.phase == "avqds_screening"


def test_legacy_r00_cut1e3_is_retained_at_canonical_index_two() -> None:
    legacy = _screen(
        ladder.ARM_AVQDS,
        LEGACY_CUT,
        response=0.08,
        legacy_r00=True,
    )
    assert legacy.action_id == "siam-nb4-avqds-screen-r00-cut1em03"
    assert legacy.ladder_index == 2
    assert legacy.uses_legacy_r00_identity is True

    plan = ladder.plan_siam_nb4_ladder([legacy])
    emitted = [
        (row.action_id, row.activation_cut, row.ladder_index)
        for row in plan.next_actions
    ]
    assert emitted == [
        ("siam-nb4-avqds-screen-r00-cut1em02", INITIAL_CUT, 0)
    ]
    disposition = next(
        row for row in plan.dispositions if row.subject_id == legacy.action_id
    )
    assert disposition.disposition == "legacy_r00_mapped_to_canonical_index_2"
    assert legacy.action_id in {
        row.subject_id for row in plan.dispositions
    }


def test_legacy_screen_and_continuation_do_not_emit_canonical_cut_duplicate() -> None:
    history = [
        _screen(
            ladder.ARM_AVQDS,
            LEGACY_CUT,
            response=0.08,
            legacy_r00=True,
        ),
        _continuation(
            ladder.ARM_AVQDS,
            LEGACY_CUT,
            response=0.08,
            legacy_r00=True,
        ),
    ]
    plan = ladder.plan_siam_nb4_ladder(history)
    assert all(action.activation_cut != LEGACY_CUT for action in plan.next_actions)
    assert plan.candidate_action_ids == (
        "siam-nb4-avqds-continue-r00-cut1em03",
    )
    assert plan.next_actions[0].activation_cut == INITIAL_CUT


def test_failed_coarsest_backfill_deterministically_opens_second_rung() -> None:
    history = [
        _screen(ladder.ARM_AVQDS, INITIAL_CUT, response=0.16),
        _screen(
            ladder.ARM_AVQDS,
            LEGACY_CUT,
            response=0.08,
            legacy_r00=True,
        ),
    ]
    plan = ladder.plan_siam_nb4_ladder(history)
    assert [(action.activation_cut, action.ladder_index) for action in plan.next_actions] == [
        (NEXT_CUT, 1)
    ]


@pytest.mark.parametrize("state", [ladder.STATE_SUBMITTED, ladder.STATE_IN_FLIGHT])
def test_submitted_or_inflight_action_excludes_exact_duplicate(state: str) -> None:
    record = _screen(ladder.ARM_AVQDS, INITIAL_CUT, state=state)
    plan = ladder.plan_siam_nb4_ladder([record])
    assert _run_actions(plan) == []
    row = next(item for item in plan.dispositions if item.subject_id == record.action_id)
    assert row.disposition == "awaiting_completion"


def test_failed_action_blocks_automatic_duplicate_or_rung_advance() -> None:
    record = _screen(ladder.ARM_AVQDS, INITIAL_CUT, state=ladder.STATE_FAILED)
    plan = ladder.plan_siam_nb4_ladder([record])
    assert plan.next_actions == ()
    assert plan.blocked is True
    assert any(row.disposition == "failed_action_blocks_progress" for row in plan.dispositions)


def test_completed_action_requires_positive_raw_receipt() -> None:
    with pytest.raises(ladder.SIAMNb4LadderError, match="positive raw receipt"):
        ladder.SIAMNb4ActionRecord(
            action_id=_action_id(ladder.ARM_AVQDS, ladder.SEGMENT_SCREEN, INITIAL_CUT),
            arm=ladder.ARM_AVQDS,
            segment=ladder.SEGMENT_SCREEN,
            activation_cut=INITIAL_CUT,
            state=ladder.STATE_COMPLETED,
        )


def test_raw_receipt_must_match_action_identity() -> None:
    record = _screen(ladder.ARM_AVQDS, INITIAL_CUT)
    receipt = dict(record.completion_receipt or {})
    receipt["action_id"] = "different"
    tampered = ladder.SIAMNb4ActionRecord(
        action_id=record.action_id,
        arm=record.arm,
        segment=record.segment,
        activation_cut=record.activation_cut,
        state=record.state,
        completion_receipt=receipt,
        score=record.score,
    )
    with pytest.raises(ladder.SIAMNb4LadderError, match="invalid raw receipt"):
        ladder.plan_siam_nb4_ladder([tampered])


def test_real_completion_validator_rejects_qiskit_tamper() -> None:
    record = _screen(ladder.ARM_AVQDS, INITIAL_CUT)
    receipt = dict(record.completion_receipt or {})
    receipt["qiskit_compilation_requested"] = True
    tampered = ladder.SIAMNb4ActionRecord(
        action_id=record.action_id,
        arm=record.arm,
        segment=record.segment,
        activation_cut=record.activation_cut,
        state=record.state,
        completion_receipt=receipt,
        score=record.score,
    )
    with pytest.raises(ladder.SIAMNb4LadderError, match="invalid raw receipt"):
        ladder.plan_siam_nb4_ladder([tampered])


def test_continuation_stitch_must_bind_recorded_screen_receipt() -> None:
    screen = _screen(ladder.ARM_AVQDS, INITIAL_CUT)
    alternate_receipt = _raw_receipt(
        ladder.ARM_AVQDS,
        ladder.SEGMENT_SCREEN,
        INITIAL_CUT,
        package_tag="-alternate",
    )
    alternate_screen = ladder.SIAMNb4ActionRecord(
        action_id=screen.action_id,
        arm=screen.arm,
        segment=screen.segment,
        activation_cut=screen.activation_cut,
        state=screen.state,
        completion_receipt=alternate_receipt,
        score=screen.score,
    )
    continuation = _continuation(ladder.ARM_AVQDS, INITIAL_CUT)
    with pytest.raises(ladder.SIAMNb4LadderError, match="recorded same-cut screen"):
        ladder.plan_siam_nb4_ladder([alternate_screen, continuation])


def test_history_cannot_mix_seed_contracts_across_rungs() -> None:
    first = _screen(ladder.ARM_AVQDS, INITIAL_CUT, response=0.16)
    second = _screen(ladder.ARM_AVQDS, NEXT_CUT, response=0.16)
    new_seed_sha256 = "f" * 64
    receipt = json.loads(json.dumps(second.completion_receipt))
    receipt["seed_sha256"] = new_seed_sha256
    receipt["action_contract"]["seed_sha256"] = new_seed_sha256
    action_sha256 = _pretty_sha256(receipt["action_contract"])
    receipt["action_json_sha256"] = action_sha256
    receipt["package_manifest_contract"]["action_json_sha256"] = action_sha256
    receipt["package_manifest_contract"]["files"]["action"][
        "sha256"
    ] = action_sha256
    receipt["package_manifest_sha256"] = _pretty_sha256(
        receipt["package_manifest_contract"]
    )
    score_core = dict(second.score or {})
    score_core.pop("score_sha256")
    score_core["source"] = {
        **dict(score_core["source"]),
        "seed_sha256": new_seed_sha256,
    }
    score = {**score_core, "score_sha256": _canonical_sha256(score_core)}
    rebound = ladder.SIAMNb4ActionRecord(
        action_id=second.action_id,
        arm=second.arm,
        segment=second.segment,
        activation_cut=second.activation_cut,
        state=second.state,
        completion_receipt=receipt,
        score=score,
    )
    with pytest.raises(ladder.SIAMNb4LadderError, match="mixes source revisions or seed"):
        ladder.plan_siam_nb4_ladder([first, rebound])


def test_completed_raw_screen_waits_for_offline_score() -> None:
    record = _screen(ladder.ARM_AVQDS, INITIAL_CUT, scored=False)
    plan = ladder.plan_siam_nb4_ladder([record])
    assert plan.next_actions == ()
    assert any(row.disposition == "awaiting_offline_score" for row in plan.dispositions)


def test_ladder_rejects_redigested_screen_missing_nb4_primitive() -> None:
    record = _screen(ladder.ARM_AVQDS, INITIAL_CUT)
    assert record.score is not None
    tampered = replace(
        record,
        score=_redigested_score_without_primitive(dict(record.score), "C_4"),
    )

    with pytest.raises(
        ladder.SIAMNb4LadderError,
        match=r"primitive response metrics must contain exactly .*C_4",
    ):
        ladder.plan_siam_nb4_ladder([tampered])


def test_outside_guard_screen_advances_exactly_one_tighter_rung() -> None:
    history = [_screen(ladder.ARM_AVQDS, INITIAL_CUT, response=0.16)]
    plan = ladder.plan_siam_nb4_ladder(history)
    assert len(_run_actions(plan)) == 1
    action = _run_actions(plan)[0]
    assert action.arm == ladder.ARM_AVQDS
    assert action.activation_cut == NEXT_CUT
    assert "no threshold monotonicity" in action.reason


def test_ladder_rejects_rung_skipping() -> None:
    history = [
        _screen(ladder.ARM_AVQDS, INITIAL_CUT, response=0.16),
        _screen(ladder.ARM_AVQDS, 1.0e-4, response=0.16),
    ]
    with pytest.raises(ladder.SIAMNb4LadderError, match="skips"):
        ladder.plan_siam_nb4_ladder(history)


def test_tighter_rung_requires_measured_screen_miss_or_unmatched_pair() -> None:
    history = [
        _screen(ladder.ARM_AVQDS, INITIAL_CUT, response=0.02),
        _screen(ladder.ARM_AVQDS, NEXT_CUT, response=0.02),
    ]
    with pytest.raises(ladder.SIAMNb4LadderError, match="unmatched-pair decision"):
        ladder.plan_siam_nb4_ladder(history)


def test_independently_tighter_counterpart_can_select_coarser_arm() -> None:
    history = [
        _screen(ladder.ARM_AVQDS, INITIAL_CUT, response=0.02),
        _continuation(ladder.ARM_AVQDS, INITIAL_CUT, response=0.04),
        _screen(ladder.ARM_FULL_APM, INITIAL_CUT, response=0.16),
        _screen(ladder.ARM_FULL_APM, NEXT_CUT, response=0.16),
        _screen(ladder.ARM_FULL_APM, SIAM_NB4_ACTIVATION_CUT_LADDER[2], response=0.02),
        _continuation(ladder.ARM_FULL_APM, SIAM_NB4_ACTIVATION_CUT_LADDER[2], response=0.01),
        _screen(ladder.ARM_AVQDS, NEXT_CUT, response=0.02),
    ]
    plan = ladder.plan_siam_nb4_ladder(history)
    assert any(
        action.arm == ladder.ARM_AVQDS
        and action.segment == ladder.SEGMENT_CONTINUATION
        and action.activation_cut == NEXT_CUT
        for action in plan.next_actions
    )


def test_outside_guard_at_floor_emits_safeguard() -> None:
    history = [
        _screen(ladder.ARM_AVQDS, cut, response=0.16)
        for cut in SIAM_NB4_ACTIVATION_CUT_LADDER
    ]
    plan = ladder.plan_siam_nb4_ladder(history)
    assert len(plan.next_actions) == 1
    assert plan.next_actions[0].kind == ladder.KIND_RECORD_SAFEGUARD
    assert plan.next_actions[0].activation_cut == SIAM_NB4_ACTIVATION_CUT_LADDER[-1]
    assert plan.phase == "safeguard"


def test_recorded_floor_safeguard_is_terminal_and_idempotent() -> None:
    history = [
        _screen(ladder.ARM_AVQDS, cut, response=0.16)
        for cut in SIAM_NB4_ACTIVATION_CUT_LADDER
    ]
    first = ladder.plan_siam_nb4_ladder(history)
    emitted = first.next_actions[0]
    safeguard = ladder.SIAMNb4SafeguardRecord(
        action_id=emitted.action_id,
        arm=emitted.arm,
        activation_cut=emitted.activation_cut,
        reason=emitted.reason,
    )
    replay = ladder.plan_siam_nb4_ladder([*history, safeguard])
    assert replay.next_actions == ()
    assert replay.recorded_safeguard_ids == (safeguard.action_id,)
    assert replay.blocked is True
    assert replay.phase == "safeguard"


def test_safeguard_without_a_proven_floor_miss_is_rejected() -> None:
    floor = SIAM_NB4_ACTIVATION_CUT_LADDER[-1]
    safeguard = ladder.SIAMNb4SafeguardRecord(
        action_id="siam-nb4-avqds-ladder-floor-safeguard",
        arm=ladder.ARM_AVQDS,
        activation_cut=floor,
        reason="fixture",
    )
    with pytest.raises(ladder.SIAMNb4LadderError, match="floor screen"):
        ladder.plan_siam_nb4_ladder([safeguard])


def test_floor_safeguard_suppresses_speculative_other_lane_work() -> None:
    history = [_screen(ladder.ARM_AVQDS, INITIAL_CUT, response=0.02)]
    history.extend(
        _screen(ladder.ARM_FULL_APM, cut, response=0.16)
        for cut in SIAM_NB4_ACTIVATION_CUT_LADDER
    )
    plan = ladder.plan_siam_nb4_ladder(history)
    assert len(plan.next_actions) == 1
    assert plan.next_actions[0].kind == ladder.KIND_RECORD_SAFEGUARD
    assert plan.next_actions[0].arm == ladder.ARM_FULL_APM


def test_broad_safe_avqds_opens_apm_and_same_cut_continuation() -> None:
    history = [_screen(ladder.ARM_AVQDS, INITIAL_CUT, response=0.08)]
    plan = ladder.plan_siam_nb4_ladder(history)
    actions = _run_actions(plan)
    assert {(row.arm, row.segment, row.activation_cut) for row in actions} == {
        (ladder.ARM_AVQDS, ladder.SEGMENT_CONTINUATION, INITIAL_CUT),
        (ladder.ARM_FULL_APM, ladder.SEGMENT_SCREEN, INITIAL_CUT),
    }
    continuation = next(row for row in actions if row.arm == ladder.ARM_AVQDS)
    assert continuation.resume_from_action_id == history[0].action_id


def test_apm_scout_start_allows_contiguous_tighter_rung_replay() -> None:
    scout_index = 2
    tighter_index = scout_index + 1
    history = [
        _screen(ladder.ARM_AVQDS, SIAM_NB4_ACTIVATION_CUT_LADDER[0], response=0.16),
        _screen(ladder.ARM_AVQDS, SIAM_NB4_ACTIVATION_CUT_LADDER[1], response=0.16),
        _screen(
            ladder.ARM_AVQDS,
            SIAM_NB4_ACTIVATION_CUT_LADDER[scout_index],
            response=0.02,
        ),
        _screen(
            ladder.ARM_FULL_APM,
            SIAM_NB4_ACTIVATION_CUT_LADDER[scout_index],
            response=0.16,
        ),
        ladder.SIAMNb4ActionRecord(
            action_id=_action_id(
                ladder.ARM_FULL_APM,
                ladder.SEGMENT_SCREEN,
                SIAM_NB4_ACTIVATION_CUT_LADDER[tighter_index],
            ),
            arm=ladder.ARM_FULL_APM,
            segment=ladder.SEGMENT_SCREEN,
            activation_cut=SIAM_NB4_ACTIVATION_CUT_LADDER[tighter_index],
            state=ladder.STATE_SUBMITTED,
        ),
    ]

    plan = ladder.plan_siam_nb4_ladder(history)

    assert all(
        action.action_id != history[-1].action_id for action in plan.next_actions
    )
    assert any(
        disposition.subject_id == history[-1].action_id
        and disposition.disposition == "awaiting_completion"
        for disposition in plan.dispositions
    )


def test_apm_cannot_open_before_broad_safe_comparator() -> None:
    history = [_screen(ladder.ARM_FULL_APM, INITIAL_CUT)]
    with pytest.raises(ladder.SIAMNb4LadderError, match="Full APM cannot open"):
        ladder.plan_siam_nb4_ladder(history)


def test_continuation_requires_same_cut_broad_safe_screen() -> None:
    history = [
        _screen(ladder.ARM_AVQDS, INITIAL_CUT, response=0.16),
        _continuation(ladder.ARM_AVQDS, INITIAL_CUT),
    ]
    with pytest.raises(ladder.SIAMNb4LadderError, match="broad-safe"):
        ladder.plan_siam_nb4_ladder(history)


def test_scored_continuation_requires_stitch_proof() -> None:
    receipt = _raw_receipt(
        ladder.ARM_AVQDS, ladder.SEGMENT_CONTINUATION, INITIAL_CUT
    )
    with pytest.raises(ladder.SIAMNb4LadderError, match="stitch proof"):
        ladder.SIAMNb4ActionRecord(
            action_id=str(receipt["action_id"]),
            arm=ladder.ARM_AVQDS,
            segment=ladder.SEGMENT_CONTINUATION,
            activation_cut=INITIAL_CUT,
            state=ladder.STATE_COMPLETED,
            resume_from_action_id=_action_id(
                ladder.ARM_AVQDS, ladder.SEGMENT_SCREEN, INITIAL_CUT
            ),
            completion_receipt=receipt,
            score=_score(
                ladder.ARM_AVQDS,
                point_count=501,
                run_sha256="f" * 64,
            ),
        )


def test_matched_full_horizon_pair_stops_ladder() -> None:
    plan = ladder.plan_siam_nb4_ladder(
        _pair_history(pair_admissible=True)
    )
    assert plan.next_actions == ()
    assert plan.complete is True
    assert plan.cost_comparison_admissible is True
    assert plan.phase == "complete"
    assert set(plan.candidate_action_ids) == {
        _action_id(ladder.ARM_AVQDS, ladder.SEGMENT_CONTINUATION, INITIAL_CUT),
        _action_id(ladder.ARM_FULL_APM, ladder.SEGMENT_CONTINUATION, INITIAL_CUT),
    }


def test_ladder_rejects_redigested_pair_with_both_scores_missing_c4() -> None:
    history = _pair_history(pair_admissible=True)
    tampered: list[ladder.SIAMNb4ActionRecord] = []
    changed_scores = 0
    for record in history:
        if record.segment == ladder.SEGMENT_CONTINUATION:
            assert record.score is not None
            record = replace(
                record,
                score=_redigested_score_without_primitive(
                    dict(record.score), "C_4"
                ),
            )
            changed_scores += 1
        tampered.append(record)
    assert changed_scores == 2

    with pytest.raises(
        ladder.SIAMNb4LadderError,
        match=r"primitive response metrics must contain exactly .*C_4",
    ):
        ladder.plan_siam_nb4_ladder(tampered)


def test_unmatched_pair_tightens_only_higher_burden_arm() -> None:
    plan = ladder.plan_siam_nb4_ladder(_pair_history())
    actions = _run_actions(plan)
    assert len(actions) == 1
    assert actions[0].arm == ladder.ARM_AVQDS
    assert actions[0].segment == ladder.SEGMENT_SCREEN
    assert actions[0].activation_cut == NEXT_CUT
    assert len(plan.candidate_action_ids) == 2
    assert plan.cost_comparison_admissible is False


def test_unmatched_exact_burden_tie_deterministically_tightens_full_apm() -> None:
    plan = ladder.plan_siam_nb4_ladder(
        _pair_history(
            avqds_response=0.01,
            apm_response=0.01,
            avqds_primitives={"C_1": 0.04},
            apm_primitives={"C_2": 0.04},
        )
    )
    actions = _run_actions(plan)
    assert len(actions) == 1
    assert actions[0].arm == ladder.ARM_FULL_APM


def test_single_guard_offender_wins_even_with_lower_burden() -> None:
    plan = ladder.plan_siam_nb4_ladder(
        _pair_history(
            avqds_response=0.001,
            apm_response=0.14,
            avqds_infidelity=1.1e-4,
        )
    )
    actions = _run_actions(plan)
    assert len(actions) == 1
    assert actions[0].arm == ladder.ARM_AVQDS


def test_pair_mismatch_at_floor_emits_accuracy_confounded_safeguard() -> None:
    floor = SIAM_NB4_ACTIVATION_CUT_LADDER[-1]
    plan = ladder.plan_siam_nb4_ladder(
        _pair_history(
            apm_cut=floor,
            avqds_response=0.01,
            apm_response=0.04,
        )
    )
    assert len(plan.next_actions) == 1
    assert plan.next_actions[0].kind == ladder.KIND_RECORD_SAFEGUARD
    assert plan.next_actions[0].arm == ladder.ARM_FULL_APM
    assert "accuracy-confounded" in plan.next_actions[0].reason


def test_duplicate_action_ids_are_rejected() -> None:
    record = _screen(ladder.ARM_AVQDS, INITIAL_CUT)
    with pytest.raises(ladder.SIAMNb4LadderError, match="repeats action ID"):
        ladder.plan_siam_nb4_ladder([record, record])


def test_plan_json_is_executor_ready_and_has_no_submission_side_effect() -> None:
    plan = ladder.plan_siam_nb4_ladder([])
    payload = plan.to_json_dict()
    assert payload["schema"] == ladder.SIAM_NB4_LADDER_PLAN_V1
    assert payload["next_actions"][0]["kind"] == ladder.KIND_RUN
    assert payload["next_actions"][0]["qiskit_compilation_requested"] is False
    assert payload["next_actions"][0]["exact_reference_blind"] is True


FROZEN_HISTORICAL_ACTION_IDS = {
    (ladder.ARM_FULL_APM, ladder.SEGMENT_SCREEN, SIAM_NB4_ACTIVATION_CUT_LADDER[0]):
        "siam-nb4-full_apm-screen-r00-cut1em02",
    (ladder.ARM_FULL_APM, ladder.SEGMENT_CONTINUATION, SIAM_NB4_ACTIVATION_CUT_LADDER[7]):
        "siam-nb4-full_apm-continue-r07-cut3em06",
    (ladder.ARM_AVQDS, ladder.SEGMENT_SCREEN, SIAM_NB4_ACTIVATION_CUT_LADDER[0]):
        "siam-nb4-avqds-screen-r00-cut1em02",
}


def test_historical_action_ids_are_byte_identical_and_qc_ids_are_new() -> None:
    for (arm, segment, cut), frozen in FROZEN_HISTORICAL_ACTION_IDS.items():
        assert ladder.siam_nb4_action_id(
            arm=arm, segment=segment, activation_cut=cut
        ) == frozen
        assert ladder.siam_nb4_action_id(
            arm=arm,
            segment=segment,
            activation_cut=cut,
            route_id=ladder.ROUTE_HISTORICAL,
        ) == frozen
        compiled = ladder.siam_nb4_action_id(
            arm=arm,
            segment=segment,
            activation_cut=cut,
            route_id=ladder.ROUTE_Q_COMPILED,
        )
        assert compiled == frozen.replace("siam-nb4-", "siam-nb4-qc-", 1)
        assert compiled != frozen

    assert ladder.ROUTE_TOKENS[ladder.ROUTE_HISTORICAL] is None
    assert ladder.ROUTE_TOKENS[ladder.ROUTE_Q_COMPILED] == "qc"


def test_compiled_q_ladder_opens_at_r00_in_the_avqds_lane() -> None:
    plan = ladder.plan_siam_nb4_ladder([], route_id=ladder.ROUTE_Q_COMPILED)
    assert plan.route_id == ladder.ROUTE_Q_COMPILED
    assert plan.phase == "avqds_screening"
    assert len(plan.next_actions) == 1
    action = plan.next_actions[0]
    assert (action.arm, action.segment, action.activation_cut, action.ladder_index) == (
        ladder.ARM_AVQDS,
        ladder.SEGMENT_SCREEN,
        INITIAL_CUT,
        0,
    )
    assert action.action_id == "siam-nb4-qc-avqds-screen-r00-cut1em02"
    assert action.route_id == ladder.ROUTE_Q_COMPILED
    assert plan.to_json_dict()["route_id"] == ladder.ROUTE_Q_COMPILED

    historical = ladder.plan_siam_nb4_ladder([])
    assert historical.next_actions[0].action_id == (
        "siam-nb4-avqds-screen-r00-cut1em02"
    )
    assert historical.route_id == ladder.ROUTE_HISTORICAL


def test_plan_refuses_a_record_from_another_route() -> None:
    record = _screen(ladder.ARM_AVQDS, INITIAL_CUT)
    with pytest.raises(ladder.SIAMNb4LadderError, match="belongs to route"):
        ladder.plan_siam_nb4_ladder([record], route_id=ladder.ROUTE_Q_COMPILED)


def _compiled_q_screen(arm: str, cut: float, *, route_bundle: str):
    """Build one completed compiled-Q screen record with a declared route."""

    return _record(
        arm,
        ladder.SEGMENT_SCREEN,
        cut,
        route_id=ladder.ROUTE_Q_COMPILED,
        exchange_route_bundle=route_bundle,
    )


def test_compiled_q_ladder_refuses_a_score_from_the_historical_route() -> None:
    off_route = _compiled_q_screen(
        ladder.ARM_FULL_APM,
        INITIAL_CUT,
        route_bundle="paper_ii_generalized_exchange_v2",
    )
    with pytest.raises(ladder.SIAMNb4LadderError, match="is not on the"):
        ladder.plan_siam_nb4_ladder([off_route], route_id=ladder.ROUTE_Q_COMPILED)

    on_route = _compiled_q_screen(
        ladder.ARM_FULL_APM,
        INITIAL_CUT,
        route_bundle=ladder.ROUTE_Q_COMPILED,
    )
    # The exchange lane still needs its comparator screen, so the planner blocks
    # rather than accepting the arm; the route itself is no longer the reason.
    with pytest.raises(
        ladder.SIAMNb4LadderError, match="before a valid broad-safe AVQDS screen"
    ):
        ladder.plan_siam_nb4_ladder([on_route], route_id=ladder.ROUTE_Q_COMPILED)


def test_compiled_q_avqds_lane_is_its_own_and_advances_to_full_apm() -> None:
    screen = _record(
        ladder.ARM_AVQDS,
        ladder.SEGMENT_SCREEN,
        INITIAL_CUT,
        route_id=ladder.ROUTE_Q_COMPILED,
    )
    assert screen.action_id == "siam-nb4-qc-avqds-screen-r00-cut1em02"
    plan = ladder.plan_siam_nb4_ladder([screen], route_id=ladder.ROUTE_Q_COMPILED)
    emitted = {row.action_id for row in plan.next_actions}
    assert "siam-nb4-qc-avqds-continue-r00-cut1em02" in emitted
    assert "siam-nb4-qc-full_apm-screen-r00-cut1em02" in emitted
    assert all(row.route_id == ladder.ROUTE_Q_COMPILED for row in plan.next_actions)
    assert plan.phase == "two_arm_screening"


def test_historical_ladder_accepts_a_pre_route_and_a_labelled_historical_score() -> None:
    record = _screen(ladder.ARM_FULL_APM, INITIAL_CUT)
    assert "exchange_route_bundle" not in record.score["method"]

    labelled = _record(
        ladder.ARM_FULL_APM,
        ladder.SEGMENT_SCREEN,
        INITIAL_CUT,
        exchange_route_bundle="paper_ii_generalized_exchange_v2",
    )
    for candidate in (record, labelled):
        with pytest.raises(
            ladder.SIAMNb4LadderError,
            match="before a valid broad-safe AVQDS screen",
        ):
            ladder.plan_siam_nb4_ladder([candidate])
