"""Digest-bound SIAM raw-segment lineage and canonical Nb4 stitching."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pipelines.time_dynamics.diagnostics.siam_segment_stitching import (
    SIAM_NB4_STITCH_COMPLETION_V1,
    SIAM_NB4_STITCH_V1,
    SIAM_NB4_TIMES,
    build_siam_nb4_stitch_completion_receipt,
    finalize_siam_resume_lineage,
    prepare_siam_resume_lineage,
    stitch_siam_nb4_segments,
    validate_siam_nb4_stitch_completion_receipt,
    write_siam_nb4_stitch_completion_receipt,
)
from pipelines.time_dynamics.paper_ii_runs import (
    build_siam_nb4_continuation_run,
    build_siam_nb4_screen_run,
)
from pipelines.time_dynamics.resume import (
    CONTROLLER_RESUME_SCHEMA_V1,
    RESUME_SCHEMA_V2,
)
from pipelines.time_dynamics.run_lock import build_run_lock
from pipelines.time_dynamics.siam_activation_ladder import (
    SIAM_NB4_ACTIVATION_CUT_LADDER,
    SIAM_NB4_CANONICAL_LADDER_SHA256,
    ladder_payload,
)
from pipelines.time_dynamics.siam_chtc_package import (
    SIAM_NB4_RESUME_TERMINAL_PROOF_V1,
)
from pipelines.time_dynamics.siam_screening import PILOT_NB4
from pipelines.time_dynamics.siam_seed import (
    portable_statevector_sha256,
    write_siam_seed_manifest,
)


SOURCE_REVISION = "a" * 40


def _canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _pretty_sha256(payload: dict) -> str:
    return hashlib.sha256(
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(time: float, index: int, *, interval_cost: int) -> dict:
    state = [float(np.cos(time)), float(np.sin(time))]
    return {
        "index": int(index),
        "time": float(time),
        "energy_expectation": -1.0,
        "psi_real": state,
        "psi_imag": [0.0, 0.0],
        "integration_local_substep_count": int(interval_cost),
        "integration_rhs_evaluation_count": int(interval_cost + (1 if interval_cost else 0)),
        "integration_attempted_substep_count": int(interval_cost + (1 if interval_cost else 0)),
        "integration_rejected_substep_count": int(1 if interval_cost else 0),
        "integration_local_subdivision_reason": None,
        "parameter_control_substep_count": None,
        "parameter_control_rhs_evaluation_count": None,
        "parameter_control_attempted_substep_count": None,
        "parameter_control_rejected_substep_count": None,
        "patch_commit_count": int(1 if interval_cost else 0),
        "patch_append_inserted_coordinate_count": int(1 if interval_cost else 0),
        "patch_delete_deleted_coordinate_count": 0,
        "patch_exchange_inserted_coordinate_count": 0,
        "patch_exchange_deleted_coordinate_count": 0,
    }


def _route_guards(route_bundle_id: str | None) -> dict:
    """Return the run-lock guards for one exchange route (or the pre-route set)."""

    guards: dict = {"max_joint_patch_evaluations": 50000}
    if route_bundle_id is None:
        return guards
    contract = {
        "bundle_id": route_bundle_id,
        "objective_id": "realized_q",
        "cost_model_id": "compiled_population",
        "kernel_id": "shifted_paper_i",
        "compiler": None,
    }
    guards["exchange_route_contract"] = contract
    guards["exchange_route_contract_sha256"] = hashlib.sha256(
        json.dumps(
            contract, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    return guards


def _run(
    seed: Path,
    times: tuple[float, ...],
    *,
    cut: float,
    interval_cost: int,
    arm: str = "exchange",
    route_bundle_id: str | None = None,
) -> dict:
    start = SIAM_NB4_TIMES.index(times[0])
    rows = [
        _row(
            time,
            start + offset,
            interval_cost=(interval_cost if offset + 1 < len(times) else 0),
        )
        for offset, time in enumerate(times)
    ]
    lock = build_run_lock(
        seed_artifact_json=str(seed),
        verified_seed_sha256=None,
        family_key="siam",
        n_ph_max=None,
        times=times,
        drive_profile=None,
        integrator_method="euler",
        inverse_policy={"pinv_rcond": 1.0e-10, "ridge_lambda": 1.0e-6, "solve_damping": 0.0},
        solve_repair={"profile_id": "state_motion", "state_motion_l2_step_max": 5.0e-5},
        structural_policy=arm,
        guards=_route_guards(route_bundle_id),
        time_step_controller={"state_motion_l2_step_max": 5.0e-5},
        candidate_pool={
            "schema": "paper_ii_effective_candidate_pool_v1",
            "ordered_atom_contract_sha256": "a" * 64,
            "append_occurrence_policy": "layer_reuse",
        },
        exact_reference_json=None,
        code_revision=SOURCE_REVISION,
    )
    return {
        "schema": "ap_mclachlan_append_from_adapt_artifact_v1",
        "run_lock": lock,
        "resume_state": {
            "schema": RESUME_SCHEMA_V2,
            "problem_family": "siam",
            "parameterization_mode": "logical_shared",
            "time": float(times[-1]),
            "coordinates": [],
            "theta_runtime": [],
            "theta_logical": [],
            "controller_state": {
                "schema": CONTROLLER_RESUME_SCHEMA_V1,
                "checkpoint_index": int(rows[-1]["index"]),
                "prune_runtime_state": {},
                "repair_hold_state": None,
                "previous_accepted_theta_dot": None,
            },
        },
        "initial_state": {"synthetic": True},
        "final_state": {"synthetic": True},
        "trajectory": {
            "metadata": {"parameterization_mode": "logical_shared"},
            "points": [{"index": row["index"], "time": row["time"]} for row in rows],
        },
        "plot_rows": rows,
        "summary": {
            "support_patch_config": {
                "dynamics_policy": arm,
                "insertion_l2_cut": float(cut),
                "avqds_l2_cut": float(cut),
            },
            "active_parameter_count_initial": 0,
            "runtime_parameter_count_initial": 0,
            "runtime_pauli_parameter_count_initial": 0,
            "logical_parameter_count_initial": 0,
            "active_parameter_count_final": 2,
            "runtime_parameter_count_final": 2,
            "runtime_pauli_parameter_count_final": 4,
            "logical_parameter_count_final": 2,
            "uncured_local_subdivision_count": 0,
        },
        "decision_data_flow": {"uses_exact_reference_for_decision": False},
    }


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_action_receipt(
    path: Path,
    *,
    seed: Path,
    trajectory: Path,
    arm: str,
    segment: str,
    activation_cut: float,
    status: str = "COMPLETED",
) -> None:
    trajectory_payload = json.loads(trajectory.read_text(encoding="utf-8"))
    resume = trajectory_payload["resume_state"]
    controller = resume["controller_state"]
    terminal_row = trajectory_payload["plot_rows"][-1]
    terminal_state = np.asarray(
        terminal_row["psi_real"], dtype=float
    ) + 1.0j * np.asarray(terminal_row["psi_imag"], dtype=float)
    terminal_state_sha256 = portable_statevector_sha256(terminal_state)
    support_contract = {
        "coordinates": list(resume["coordinates"]),
        "runtime_coordinate_labels": [],
        "runtime_pauli_coordinate_labels": [],
        "selected_term_labels": [],
    }
    theta_contract = {
        "theta_runtime": [float(value) for value in resume["theta_runtime"]],
        "theta_logical": [float(value) for value in resume["theta_logical"]],
    }
    previous_velocity = controller["previous_accepted_theta_dot"]
    resume_terminal_proof = {
        "schema": SIAM_NB4_RESUME_TERMINAL_PROOF_V1,
        "resume_schema": RESUME_SCHEMA_V2,
        "controller_schema": CONTROLLER_RESUME_SCHEMA_V1,
        "terminal_time": float(resume["time"]),
        "terminal_checkpoint_index": int(controller["checkpoint_index"]),
        "logical_coordinate_count": len(resume["theta_logical"]),
        "runtime_pauli_coordinate_count": len(resume["theta_runtime"]),
        "previous_accepted_velocity_count": (
            0 if previous_velocity is None else len(previous_velocity)
        ),
        "support_contract_sha256": _canonical_sha256(support_contract),
        "theta_contract_sha256": _canonical_sha256(theta_contract),
        "controller_state_sha256": _canonical_sha256(controller),
        "reconstructed_state_semantic_sha256": terminal_state_sha256,
        "terminal_state_semantic_sha256": terminal_state_sha256,
        "reconstructed_state_infidelity": 0.0,
    }
    seed_token = "inputs/seed.json"
    trajectory_token = f"raw_outputs/{trajectory.parent.name}/run.json"
    if segment == "screen":
        command = build_siam_nb4_screen_run(
            seed_path=seed_token,
            arm=arm,
            activation_cut=activation_cut,
            output_json=trajectory_token,
            code_revision=SOURCE_REVISION,
        )
        segment_id = "screen_t0_t1_101"
    else:
        command = build_siam_nb4_continuation_run(
            seed_path=seed_token,
            arm=arm,
            resume_from_run_json="input/resume/source_run.json",
            activation_cut=activation_cut,
            output_json=trajectory_token,
            code_revision=SOURCE_REVISION,
        )
        segment_id = "continue_t1_t5_401"
    argv = list(command.argv())
    argv_sha256 = hashlib.sha256(
        json.dumps(argv, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()
    package_id = f"package-{segment}-{arm}"
    action_id = f"action_{segment}_{arm}"
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
        "source_revision": SOURCE_REVISION,
        "file_count": len(source_rows),
        "files_sha256": hashlib.sha256(
            json.dumps(
                source_rows,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        "files": source_rows,
    }
    action_contract = {
        "schema": "siam_nb4_chtc_action_v3",
        "package_id": package_id,
        "action_id": action_id,
        "source_revision": SOURCE_REVISION,
        "source_tree": source_tree,
        "arm": arm,
        "segment": segment_id,
        "activation_cut": float(activation_cut),
        "activation_cut_ladder_index": SIAM_NB4_ACTIVATION_CUT_LADDER.index(
            float(activation_cut)
        ),
        "activation_cut_ladder_sha256": SIAM_NB4_CANONICAL_LADDER_SHA256,
        "seed_path": seed_token,
        "seed_sha256": _file_sha256(seed),
        "trajectory_json": trajectory_token,
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
        "schema": "siam_nb4_chtc_package_manifest_v3",
        "package_id": package_id,
        "action_id": action_id,
        "source_revision": SOURCE_REVISION,
        "action_json_sha256": action_sha256,
        "runner_argv_sha256": argv_sha256,
        "expected_run_id": command.run_id,
        "expected_output": trajectory_token,
        "source_tree_contract": "siam_source_tree_contract_v1",
        "source_tree_files_sha256": source_tree["files_sha256"],
        "source_tree_file_count": source_tree["file_count"],
        "activation_cut_ladder": list(
            ladder_payload(SIAM_NB4_ACTIVATION_CUT_LADDER)
        ),
        "activation_cut_ladder_sha256": SIAM_NB4_CANONICAL_LADDER_SHA256,
        "files": {"action": {"sha256": action_sha256}},
    }
    payload = {
        "schema": "siam_nb4_chtc_action_receipt_v3",
        "status": status,
        "package_id": package_id,
        "action_id": action_id,
        "run_id": command.run_id,
        "source_revision": SOURCE_REVISION,
        "action_json_sha256": action_sha256,
        "package_manifest_sha256": _pretty_sha256(manifest_contract),
        "source_tree_files_sha256": source_tree["files_sha256"],
        "source_tree_file_count": source_tree["file_count"],
        "seed_path": seed_token,
        "seed_sha256": _file_sha256(seed),
        "arm": arm,
        "activation_cut": float(activation_cut),
        "segment": segment_id,
        "exact_reference_blind": True,
        "qiskit_compilation_requested": False,
        "runner_argv": argv,
        "runner_argv_sha256": argv_sha256,
        "trajectory_json": trajectory_token,
        "trajectory_sha256": _file_sha256(trajectory),
        "trajectory_bytes": trajectory.stat().st_size,
        "action_contract": action_contract,
        "package_manifest_contract": manifest_contract,
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
                {
                    "initial_seed_state": True,
                    "initial_seed_energy": True,
                }
                if segment == "screen"
                else {
                    "resume_boundary_state": True,
                    "resume_boundary_energy": True,
                }
            ),
            "state_norm_sector": True,
            "resume_reconstruction": True,
            "resume_support_parameters": True,
            "resume_terminal_statevector": True,
            "controller_resume_reconstruction": True,
            "controller_resume_terminal_state": True,
        },
        "resume_terminal_proof": resume_terminal_proof,
    }
    _write(path, payload)


def _pair(
    tmp_path: Path,
    *,
    arm: str = "exchange",
    first_route: str | None = None,
    second_route: str | None = None,
) -> tuple[Path, Path, Path, dict, dict]:
    seed = tmp_path / "seed.json"
    write_siam_seed_manifest(seed, PILOT_NB4)
    first_path = tmp_path / "segment_0_1.json"
    second_path = tmp_path / "segment_1_5.json"
    first = _run(
        seed,
        SIAM_NB4_TIMES[:101],
        cut=1.0e-3,
        interval_cost=2,
        arm=arm,
        route_bundle_id=first_route,
    )
    # A terminal segment row must not contribute work after its endpoint.  Put
    # deliberately bogus values here to prove the stitcher retains only the
    # resumed boundary row for cost aggregation.
    first["plot_rows"][-1]["integration_local_substep_count"] = 777
    first["plot_rows"][-1]["patch_commit_count"] = 777
    _write(first_path, first)

    second = _run(
        seed,
        SIAM_NB4_TIMES[100:],
        cut=1.0e-4,
        interval_cost=3,
        arm=arm,
        route_bundle_id=second_route,
    )
    prepared = prepare_siam_resume_lineage(first_path, first)
    second["segment_lineage"] = finalize_siam_resume_lineage(
        prepared,
        second,
        controller_input=first["resume_state"]["controller_state"],
    )
    _write(second_path, second)
    return seed, first_path, second_path, first, second


def _completed_stitch_inputs(
    tmp_path: Path, *, arm: str = "avqds", route_bundle_id: str | None = None
) -> tuple[Path, tuple[Path, Path], Path]:
    seed, first_path, second_path, _first, _second = _pair(
        tmp_path,
        arm=arm,
        first_route=route_bundle_id,
        second_route=route_bundle_id,
    )
    stitched_path = tmp_path / "stitched.json"
    stitch_siam_nb4_segments(
        (first_path, second_path), seed_path=seed, output_path=stitched_path
    )
    first_receipt = tmp_path / "screen_action_receipt.json"
    second_receipt = tmp_path / "continuation_action_receipt.json"
    _write_action_receipt(
        first_receipt,
        seed=seed,
        trajectory=first_path,
        arm=arm,
        segment="screen",
        activation_cut=1.0e-3,
    )
    _write_action_receipt(
        second_receipt,
        seed=seed,
        trajectory=second_path,
        arm=arm,
        segment="continuation",
        activation_cut=1.0e-4,
    )
    return stitched_path, (first_receipt, second_receipt), seed


def test_stitch_preserves_lineage_cut_schedule_and_row_costs(tmp_path: Path) -> None:
    seed, first_path, second_path, _first, second = _pair(tmp_path)
    output = tmp_path / "stitched.json"

    stitched = stitch_siam_nb4_segments(
        (first_path, second_path),
        seed_path=seed,
        output_path=output,
    )

    assert stitched["segment_stitching"]["schema"] == SIAM_NB4_STITCH_V1
    assert len(stitched["plot_rows"]) == 501
    assert [row["time"] for row in stitched["plot_rows"]].count(1.0) == 1
    assert stitched["plot_rows"][100] == second["plot_rows"][0]
    assert stitched["summary"]["accepted_internal_substep_count"] == 1400
    assert stitched["summary"]["accepted_patch_count"] == 500
    assert stitched["segment_stitching"]["row_costs"] == {
        key: stitched["summary"][key]
        for key in stitched["segment_stitching"]["row_costs"]
    }
    assert [
        row["activation_l2_cut"]
        for row in stitched["segment_stitching"]["activation_cut_schedule"]
    ] == [1.0e-3, 1.0e-4]
    assert len(stitched["segment_stitching"]["source_segments"]) == 2
    assert stitched["run_lock"]["physics"]["time_grid"]["point_count"] == 501
    assert stitched["run_lock"]["physics"]["time_grid"]["t_final"] == 5.0
    assert json.loads(output.read_text(encoding="utf-8")) == stitched
    # Immutable same-content replay is a no-op.
    assert stitch_siam_nb4_segments(
        (first_path, second_path), seed_path=seed, output_path=output
    ) == stitched


def test_continuation_rejects_protocol_drift_but_allows_cut_change(
    tmp_path: Path,
) -> None:
    seed = tmp_path / "seed.json"
    write_siam_seed_manifest(seed, PILOT_NB4)
    first_path = tmp_path / "first.json"
    first = _run(seed, SIAM_NB4_TIMES[:101], cut=1.0e-3, interval_cost=1)
    _write(first_path, first)
    prepared = prepare_siam_resume_lineage(first_path, first)

    changed_cut = _run(seed, SIAM_NB4_TIMES[100:], cut=1.0e-4, interval_cost=1)
    lineage = finalize_siam_resume_lineage(
        prepared,
        changed_cut,
        controller_input=first["resume_state"]["controller_state"],
    )
    assert lineage["activation_cut_changed"] is True

    changed_code = deepcopy(changed_cut)
    changed_code["run_lock"]["code_revision"] = "different-revision"
    with pytest.raises(ValueError, match="changed seed, family, structural method"):
        finalize_siam_resume_lineage(
            prepared,
            changed_code,
            controller_input=first["resume_state"]["controller_state"],
        )

    changed_controller = deepcopy(first["resume_state"]["controller_state"])
    changed_controller["checkpoint_index"] += 1
    with pytest.raises(ValueError, match="controller input differs"):
        finalize_siam_resume_lineage(
            prepared,
            changed_cut,
            controller_input=changed_controller,
        )


def test_stitch_rejects_source_bytes_changed_after_resume(tmp_path: Path) -> None:
    seed, first_path, second_path, first, _second = _pair(tmp_path)
    first["post_resume_tamper"] = True
    _write(first_path, first)

    with pytest.raises(ValueError, match="source-run digest mismatch"):
        stitch_siam_nb4_segments(
            (first_path, second_path),
            seed_path=seed,
        )


def test_stitched_completion_receipt_is_self_contained_and_digest_bound(
    tmp_path: Path,
) -> None:
    stitched_path, source_receipts, seed = _completed_stitch_inputs(tmp_path)

    receipt = build_siam_nb4_stitch_completion_receipt(
        stitched_path,
        source_action_receipt_paths=source_receipts,
    )
    verified = validate_siam_nb4_stitch_completion_receipt(
        receipt,
        expected_trajectory_sha256=_file_sha256(stitched_path),
        expected_seed_sha256=_file_sha256(seed),
        expected_arm="avqds",
    )

    assert verified["schema"] == SIAM_NB4_STITCH_COMPLETION_V1
    assert verified["status"] == "COMPLETED"
    assert verified["stitched_trajectory"]["sha256"] == _file_sha256(
        stitched_path
    )
    assert verified["stitched_trajectory"]["bytes"] == stitched_path.stat().st_size
    assert verified["stitch"]["stitch_sha256"] == json.loads(
        stitched_path.read_text(encoding="utf-8")
    )["segment_stitching"]["stitch_sha256"]
    assert verified["canonical_grid"] == {
        "t_initial": 0.0,
        "t_final": 5.0,
        "point_count": 501,
        "ordered_values_sha256": _canonical_sha256(list(SIAM_NB4_TIMES)),
    }
    assert [row["segment"] for row in verified["source_segments"]] == [
        "screen_t0_t1_101",
        "continue_t1_t5_401",
    ]
    for row, path in zip(verified["source_segments"], source_receipts):
        document = json.loads(path.read_text(encoding="utf-8"))
        assert row["completion_receipt_file_sha256"] == _file_sha256(path)
        assert row["completion_receipt_document_sha256"] == _canonical_sha256(
            document
        )
        assert row["completion_receipt"] == document
        assert json.loads(row["completion_receipt_file_utf8"]) == document

    # Validation uses only the proof document.  Removing all underlying inputs
    # cannot change a positive validation result.
    source_runs = tuple(
        Path(row["run_path"]) for row in receipt["stitch"]["source_segments"]
    )
    for path in (*source_receipts, *source_runs, stitched_path, seed):
        path.unlink()
    assert validate_siam_nb4_stitch_completion_receipt(receipt) == verified


@pytest.mark.parametrize(
    "mutation",
    ["plot_row", "trajectory_point", "summary", "run_lock"],
)
def test_stitched_completion_builder_rejects_post_stitch_payload_mutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    stitched_path, source_receipts, _seed = _completed_stitch_inputs(tmp_path)
    stitched = json.loads(stitched_path.read_text(encoding="utf-8"))
    if mutation == "plot_row":
        stitched["plot_rows"][250]["energy_expectation"] = 123456.0
    elif mutation == "trajectory_point":
        stitched["trajectory"]["points"][250]["forged"] = True
    elif mutation == "summary":
        stitched["summary"]["accepted_patch_count"] += 1
    elif mutation == "run_lock":
        stitched["run_lock"]["policy"]["forged"] = True
    else:  # pragma: no cover - parametrization exhausts the cases.
        raise AssertionError(mutation)
    _write(stitched_path, stitched)

    with pytest.raises(ValueError, match="source-derived payload"):
        build_siam_nb4_stitch_completion_receipt(
            stitched_path,
            source_action_receipt_paths=source_receipts,
        )


def test_stitched_completion_accepts_the_full_apm_exchange_arm(
    tmp_path: Path,
) -> None:
    stitched_path, source_receipts, _seed = _completed_stitch_inputs(
        tmp_path, arm="exchange"
    )

    receipt = build_siam_nb4_stitch_completion_receipt(
        stitched_path,
        source_action_receipt_paths=source_receipts,
    )

    assert receipt["arm"] == "exchange"
    assert validate_siam_nb4_stitch_completion_receipt(
        receipt, expected_arm="full_apm"
    ) == receipt


def test_stitched_completion_writer_is_immutable_and_idempotent(
    tmp_path: Path,
) -> None:
    stitched_path, source_receipts, _seed = _completed_stitch_inputs(tmp_path)
    receipt = build_siam_nb4_stitch_completion_receipt(
        stitched_path,
        source_action_receipt_paths=source_receipts,
    )
    destination = tmp_path / "evidence" / "stitch_completion_receipt.json"

    assert write_siam_nb4_stitch_completion_receipt(destination, receipt) == receipt
    first_bytes = destination.read_bytes()
    assert write_siam_nb4_stitch_completion_receipt(destination, receipt) == receipt
    assert destination.read_bytes() == first_bytes

    changed = deepcopy(receipt)
    changed["stitched_trajectory"]["path"] = "different.json"
    core = dict(changed)
    core.pop("completion_sha256")
    changed["completion_sha256"] = _canonical_sha256(core)
    with pytest.raises(FileExistsError, match="immutable SIAM stitch"):
        write_siam_nb4_stitch_completion_receipt(destination, changed)


def test_stitched_completion_builder_rejects_negative_source_receipt(
    tmp_path: Path,
) -> None:
    stitched_path, source_receipts, _seed = _completed_stitch_inputs(tmp_path)
    negative = json.loads(source_receipts[1].read_text(encoding="utf-8"))
    negative["status"] = "FAILED"
    _write(source_receipts[1], negative)

    with pytest.raises(ValueError, match="status COMPLETED"):
        build_siam_nb4_stitch_completion_receipt(
            stitched_path,
            source_action_receipt_paths=source_receipts,
        )


def test_stitched_completion_builder_rejects_reordered_source_receipts(
    tmp_path: Path,
) -> None:
    stitched_path, source_receipts, _seed = _completed_stitch_inputs(tmp_path)

    with pytest.raises(ValueError, match="mismatch"):
        build_siam_nb4_stitch_completion_receipt(
            stitched_path,
            source_action_receipt_paths=tuple(reversed(source_receipts)),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("outer_digest", "completion-receipt digest mismatch"),
        ("stitch_digest", "embedded stitch digest mismatch"),
        ("canonical_grid", "canonical grid mismatch"),
        ("source_trajectory", "trajectory SHA-256 mismatch"),
        ("source_document_digest", "document digest mismatch"),
        ("source_file_digest", "file SHA-256 mismatch"),
        ("negative_source", "status COMPLETED"),
    ],
)
def test_stitched_completion_validator_rejects_self_contained_tampering(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    stitched_path, source_receipts, _seed = _completed_stitch_inputs(tmp_path)
    receipt = build_siam_nb4_stitch_completion_receipt(
        stitched_path,
        source_action_receipt_paths=source_receipts,
    )
    tampered = deepcopy(receipt)

    if mutation == "outer_digest":
        tampered["arm"] = "exchange"
    elif mutation == "stitch_digest":
        tampered["stitch"]["seed_sha256"] = "b" * 64
    elif mutation == "canonical_grid":
        tampered["canonical_grid"]["point_count"] = 500
    elif mutation == "source_trajectory":
        tampered["source_segments"][0]["trajectory_sha256"] = "b" * 64
    elif mutation == "source_document_digest":
        tampered["source_segments"][0][
            "completion_receipt_document_sha256"
        ] = "b" * 64
    elif mutation == "source_file_digest":
        tampered["source_segments"][0][
            "completion_receipt_file_sha256"
        ] = "b" * 64
    elif mutation == "negative_source":
        row = tampered["source_segments"][0]
        row["completion_receipt"]["status"] = "FAILED"
        row["completion_receipt_document_sha256"] = _canonical_sha256(
            row["completion_receipt"]
        )
        receipt_file = (
            json.dumps(row["completion_receipt"], indent=2, sort_keys=True) + "\n"
        )
        row["completion_receipt_file_utf8"] = receipt_file
        row["completion_receipt_file_bytes"] = len(receipt_file.encode("utf-8"))
        row["completion_receipt_file_sha256"] = hashlib.sha256(
            receipt_file.encode("utf-8")
        ).hexdigest()
    else:  # pragma: no cover - parametrization exhausts the cases.
        raise AssertionError(mutation)

    if mutation != "outer_digest":
        core = dict(tampered)
        core.pop("completion_sha256")
        tampered["completion_sha256"] = _canonical_sha256(core)

    with pytest.raises(ValueError, match=message):
        validate_siam_nb4_stitch_completion_receipt(tampered)


Q_COMPILED_ROUTE = "paper_ii_exchange_q_compiled_v1"


def test_stitch_refuses_segments_from_different_exchange_routes(
    tmp_path: Path,
) -> None:
    seed, first_path, second_path, _first, _second = _pair(
        tmp_path,
        first_route=None,
        second_route=Q_COMPILED_ROUTE,
    )

    with pytest.raises(ValueError, match="different exchange routes"):
        stitch_siam_nb4_segments((first_path, second_path), seed_path=seed)


def test_stitched_completion_records_the_compiled_q_route(tmp_path: Path) -> None:
    stitched_path, source_receipts, seed = _completed_stitch_inputs(
        tmp_path, arm="exchange", route_bundle_id=Q_COMPILED_ROUTE
    )

    receipt = build_siam_nb4_stitch_completion_receipt(
        stitched_path,
        source_action_receipt_paths=source_receipts,
    )
    verified = validate_siam_nb4_stitch_completion_receipt(
        receipt,
        expected_trajectory_sha256=_file_sha256(stitched_path),
        expected_seed_sha256=_file_sha256(seed),
        expected_arm="exchange_q_compiled",
    )
    assert verified["arm"] == "exchange"
    assert verified["route_bundle_id"] == Q_COMPILED_ROUTE
    assert len(verified["route_contract_sha256"]) == 64


def test_historical_stitch_pair_is_unchanged_and_records_no_route(
    tmp_path: Path,
) -> None:
    stitched_path, source_receipts, seed = _completed_stitch_inputs(
        tmp_path, arm="exchange"
    )

    receipt = build_siam_nb4_stitch_completion_receipt(
        stitched_path,
        source_action_receipt_paths=source_receipts,
    )
    verified = validate_siam_nb4_stitch_completion_receipt(
        receipt,
        expected_trajectory_sha256=_file_sha256(stitched_path),
        expected_seed_sha256=_file_sha256(seed),
        expected_arm="exchange",
    )
    assert verified["arm"] == "exchange"
    assert verified["route_bundle_id"] is None
    assert verified["route_contract_sha256"] is None


def test_assert_siam_resume_code_revision_rejects_mismatch_and_missing():
    from pipelines.time_dynamics.diagnostics.siam_segment_stitching import (
        assert_siam_resume_code_revision,
    )

    prepared = {"source_protocol": {"code_revision": "3e1691c4" + "0" * 32}}
    assert_siam_resume_code_revision(prepared, "3e1691c4" + "0" * 32)
    with pytest.raises(ValueError, match="differs from the source segment"):
        assert_siam_resume_code_revision(prepared, "2ef146d6" + "0" * 32)
    with pytest.raises(ValueError, match="requires --code-revision"):
        assert_siam_resume_code_revision(prepared, None)
