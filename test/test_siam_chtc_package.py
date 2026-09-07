"""Focused contracts for reusable one-action SIAM CHTC packages."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from unittest import mock

import numpy as np
import pytest

import pipelines.time_dynamics.siam_chtc_package as siam_chtc_package_module
from pipelines.time_dynamics.paper_ii_runs import (
    SIAM_NB4_CONTINUATION_SEGMENT,
    SIAM_NB4_PILOT_EXCHANGE_SEARCH,
    SIAM_NB4_PILOT_SEARCH,
    SIAM_NB4_SCREEN_SEGMENT,
    SearchProfile,
    build_siam_nb4_screen_run,
)
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
    _build_parser,
)
from pipelines.time_dynamics.ap_mclachlan.state import (
    AP_PARAMETERIZATION_LOGICAL_SHARED,
    state_from_scaffold_runtime_input,
    state_with_appended_runtime_coordinates,
)
from pipelines.time_dynamics.resume import (
    ControllerResumeState,
    build_resume_state,
)
from pipelines.time_dynamics.siam_activation_ladder import (
    SIAM_NB4_CANONICAL_LADDER_SHA256,
    ladder_payload,
)
from pipelines.time_dynamics.siam_chtc_package import (
    SIAM_NB4_ACTIVATION_CUT_LADDER,
    SIAM_NB4_CHTC_ACTION_SCHEMA_V3,
    SIAM_NB4_CHTC_ACTION_SCHEMA_V4,
    SIAM_NB4_CHTC_ACTION_SCHEMA_V5,
    SIAM_NB4_CHTC_MANIFEST_SCHEMA_V3,
    SIAM_NB4_CHTC_MANIFEST_SCHEMA_V4,
    SIAM_NB4_CHTC_MANIFEST_SCHEMA_V5,
    SIAMSourceEpochRebaseline,
    build_siam_source_epoch_closure,
    build_siam_source_epoch_contract,
    prepare_siam_nb4_chtc_action,
    run_prepared_siam_action,
    validate_siam_nb4_completion_receipt,
)
from pipelines.time_dynamics.siam_screening import PILOT_NB4
from pipelines.time_dynamics.siam_seed import (
    SIAM_SEED_SCHEMA_V1,
    load_siam_runtime_input,
    load_siam_seed_manifest,
    write_siam_seed_manifest,
)
from src.quantum.siam_model import build_initial_state, build_siam_hamiltonian
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix


REVISION = "a" * 40
RESUME_REVISION = "b" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _argv_sha256(argv: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(
            argv, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _pretty_json_sha256(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    ).hexdigest()


@pytest.fixture(scope="module")
def package_repo(tmp_path_factory) -> tuple[Path, Path, Path]:
    root = tmp_path_factory.mktemp("siam_chtc_repo")
    marker = root / "pipelines" / "test_source_marker.py"
    marker.parent.mkdir(parents=True)
    marker.write_text("SOURCE_LOCKED = True\n", encoding="utf-8")
    image = root / "chtc" / "runtime" / "image.sif"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"test-only-siam-runtime-image\n")
    seed = root / "inputs" / "siam_nb4_seed.json"
    write_siam_seed_manifest(seed, PILOT_NB4)
    return root, seed, image


def _test_source_tree_contract(root: Path, revision: str) -> dict[str, object]:
    marker = root / "pipelines" / "test_source_marker.py"
    rows = [
        {
            "path": "pipelines/test_source_marker.py",
            "kind": "file",
            "mode": marker.stat().st_mode & 0o777,
            "bytes": marker.stat().st_size,
            "sha256": _sha256(marker),
            "link_target": None,
        },
        {
            "path": "src/test_source_marker.py",
            "kind": "file",
            "mode": 0o644,
            "bytes": len(b"SOURCE_LOCKED = True\n"),
            "sha256": hashlib.sha256(b"SOURCE_LOCKED = True\n").hexdigest(),
            "link_target": None,
        },
    ]
    source_marker = root / "src" / "test_source_marker.py"
    source_marker.parent.mkdir(parents=True, exist_ok=True)
    source_marker.write_text("SOURCE_LOCKED = True\n", encoding="utf-8")
    rows[1]["mode"] = source_marker.stat().st_mode & 0o777
    encoded = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "schema": "siam_source_tree_contract_v1",
        "source_revision": revision,
        "file_count": len(rows),
        "files_sha256": hashlib.sha256(encoded).hexdigest(),
        "files": rows,
    }


def _prepare(
    package_repo,
    *,
    package_id: str,
    action_id: str,
    arm: str = "avqds",
    segment: str = "screen",
    cut: float = 1.0e-3,
    resume: Path | None = None,
    resume_receipt: Path | None = None,
    source_rebaseline: SIAMSourceEpochRebaseline | None = None,
    search_profile=SIAM_NB4_PILOT_SEARCH,
):
    root, seed, image = package_repo
    with mock.patch.object(
        siam_chtc_package_module, "_verify_source_revision"
    ), mock.patch.object(
        siam_chtc_package_module,
        "_build_source_tree_contract",
        return_value=_test_source_tree_contract(root, REVISION),
    ):
        return prepare_siam_nb4_chtc_action(
            repo_root=root,
            package_dir=Path("chtc") / package_id,
            seed_path=seed,
            image_path=image,
            source_revision=REVISION,
            action_id=action_id,
            activation_cut=cut,
            arm=arm,
            segment=segment,
            search_profile=search_profile,
            resume_from_run_json=resume,
            resume_action_receipt_json=resume_receipt,
            prepared_at="2026-08-27T00:00:00+00:00",
            intentional_source_rebaseline=source_rebaseline,
        )


def test_screen_package_binds_action_control_plane_and_raw_contract(
    package_repo,
) -> None:
    prepared = _prepare(
        package_repo,
        package_id="siam_screen_contract",
        action_id="screen_avqds_rung0",
    )
    action = json.loads(prepared.action.read_text(encoding="utf-8"))
    manifest = json.loads(prepared.manifest.read_text(encoding="utf-8"))

    assert action["schema"] == SIAM_NB4_CHTC_ACTION_SCHEMA_V4
    assert manifest["schema"] == SIAM_NB4_CHTC_MANIFEST_SCHEMA_V4
    assert manifest["status"] == "PREPARED_NOT_SUBMITTED"
    assert manifest["action_count"] == 1
    assert manifest["action_json_sha256"] == _sha256(prepared.action)
    assert prepared.action_sha256 == manifest["action_json_sha256"]
    assert prepared.manifest_sha256 == _sha256(prepared.manifest)
    assert action["activation_cut_ladder_index"] == 2
    assert action["activation_cut"] == SIAM_NB4_ACTIVATION_CUT_LADDER[2]
    assert action["exact_reference_blind"] is True
    assert action["compile_terminal_cost"] is False
    assert action["qiskit_compilation_requested"] is False
    assert action["record_statevectors"] is True
    assert action["require_complete_candidate_pool"] is True
    assert action["resume"] is None
    assert action["intentional_source_rebaseline"] is None
    assert action["scientific_identity"] == {
        "scope": "source_epoch_id/action_id",
        "source_epoch_id": action["source_epoch"]["source_epoch_id"],
        "action_id": "screen_avqds_rung0",
    }
    assert (
        "verify_action_id_absent_from_prior_packages_and_scheduler_history"
        in manifest["required_remote_preflight"]
    )
    assert not any(
        "composite_source_epoch_action_id" in value
        for value in manifest["required_remote_preflight"]
    )
    assert action["trajectory_json"] == (
        "raw_outputs/siam_screen_contract/screen_avqds_rung0/run.json"
    )

    root, _seed, _image = package_repo
    for row in manifest["files"].values():
        path = root / row["path"]
        assert path.stat().st_size == row["bytes"]
        assert _sha256(path) == row["sha256"]
    assert manifest["worker_sha256"] == _sha256(prepared.worker)
    assert manifest["wrapper_sha256"] == _sha256(prepared.wrapper)
    assert manifest["submit_descriptor_sha256"] == _sha256(prepared.submit)
    assert action["expected_runner_argv_sha256"] == manifest[
        "runner_argv_sha256"
    ]

    worker = prepared.worker.read_text(encoding="utf-8")
    assert prepared.action_sha256 in worker
    assert "run_prepared_siam_action" in worker
    wrapper = prepared.wrapper.read_text(encoding="utf-8")
    assert manifest["expected_image"]["sha256"] in wrapper
    assert str(manifest["expected_image"]["bytes"]) in wrapper
    assert manifest["worker_sha256"] in wrapper
    assert "mkdir -p raw_outputs" in wrapper
    submit = prepared.submit.read_text(encoding="utf-8")
    assert "queue 1" in submit
    assert "when_to_transfer_output = ON_EXIT" in submit
    assert "ON_EXIT_OR_EVICT" not in submit
    assert "transfer_output_files = raw_outputs" in submit
    assert "stream_output = False" in submit
    assert "stream_error = False" in submit
    assert f'+SIAMSourceEpochId = "{action["source_epoch"]["source_epoch_id"]}"' in submit
    assert '+SIAMScientificActionId = "screen_avqds_rung0"' in submit
    assert "+SIAMIntentionalSourceRebaseline = False" in submit
    assert "qiskit" not in worker.lower()
    assert "qiskit" not in wrapper.lower()
    assert "qiskit" not in submit.lower()

    command = build_siam_nb4_screen_run(
        seed_path=action["seed_path"],
        arm=action["arm"],
        activation_cut=action["activation_cut"],
        output_json=action["trajectory_json"],
        code_revision=REVISION,
    )
    parsed = _build_parser().parse_args(list(command.argv()))
    assert parsed.artifact_json is None
    assert parsed.reference_energy_json is None
    assert parsed.seed_reference_energy_json is None
    assert parsed.record_statevector is True
    assert parsed.require_complete_candidate_pool is True
    assert parsed.enable_drive is False


def _source_rebaseline_evidence(
    root: Path,
    *,
    new_source_epoch_id: str,
    action_id: str,
    predecessor_revision: str = RESUME_REVISION,
    closure_status: str = "CLOSED_SUPERSEDED",
) -> SIAMSourceEpochRebaseline:
    predecessor = root / "chtc" / f"predecessor_{action_id}"
    predecessor_manifest = predecessor / "input" / "manifest.json"
    predecessor_submission = predecessor / "submission_receipt.json"
    predecessor_closure = predecessor / "source_epoch_closure.json"
    predecessor_manifest.parent.mkdir(parents=True, exist_ok=True)
    predecessor_epoch_id = f"siam-nb4-old-{action_id}"
    predecessor_tree_sha = "c" * 64
    predecessor_manifest_payload = {
        "schema": SIAM_NB4_CHTC_MANIFEST_SCHEMA_V3,
        "status": "PREPARED_NOT_SUBMITTED",
        "package_id": predecessor.name,
        "action_id": action_id,
        "source_revision": predecessor_revision,
        "source_tree_files_sha256": predecessor_tree_sha,
        "expected_output": (
            f"raw_outputs/{predecessor.name}/{action_id}/run.json"
        ),
        "checks": {"qiskit_compilation_requested": False},
    }
    predecessor_manifest.write_text(
        json.dumps(predecessor_manifest_payload, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    manifest_sha = _sha256(predecessor_manifest)
    predecessor_submission_payload = {
        "schema": "siam_nb4_chtc_submission_receipt_v4",
        "status": "SUBMITTED",
        "cluster_id": 123456,
        "package_id": predecessor.name,
        "action_id": action_id,
        "source_revision": predecessor_revision,
        "source_tree_files_sha256": predecessor_tree_sha,
        "package_manifest_sha256": manifest_sha,
        # Older same-action ancestors may exist; this names the immediate one.
        "rebaseline_of": {"source_revision": "d" * 40},
    }
    predecessor_submission.write_text(
        json.dumps(predecessor_submission_payload, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    closure = build_siam_source_epoch_closure(
        source_epoch_id=predecessor_epoch_id,
        superseded_by_source_epoch_id=new_source_epoch_id,
        package_id=predecessor.name,
        action_id=action_id,
        source_revision=predecessor_revision,
        source_tree_files_sha256=predecessor_tree_sha,
        submission_receipt_sha256=_sha256(predecessor_submission),
        package_manifest_sha256=manifest_sha,
        closed_at="2026-08-28T00:00:00+00:00",
    )
    closure["status"] = closure_status
    if closure_status != "CLOSED_SUPERSEDED":
        closure_core = dict(closure)
        closure_core.pop("closure_receipt_sha256")
        closure["closure_receipt_sha256"] = hashlib.sha256(
            json.dumps(
                closure_core,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    predecessor_closure.write_text(
        json.dumps(closure, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return SIAMSourceEpochRebaseline(
        predecessor_source_epoch_id=predecessor_epoch_id,
        predecessor_submission_receipt=predecessor_submission,
        predecessor_package_manifest=predecessor_manifest,
        predecessor_epoch_closure_receipt=predecessor_closure,
    )


def test_explicit_source_epoch_rebaseline_binds_closed_predecessor(
    package_repo,
) -> None:
    root, _seed, _image = package_repo
    source_tree = _test_source_tree_contract(root, REVISION)
    new_epoch = build_siam_source_epoch_contract(
        source_revision=REVISION,
        source_tree_files_sha256=str(source_tree["files_sha256"]),
    )
    action_id = "screen_avqds_source_rebaseline"
    evidence = _source_rebaseline_evidence(
        root,
        new_source_epoch_id=new_epoch["source_epoch_id"],
        action_id=action_id,
    )
    prepared = _prepare(
        package_repo,
        package_id="siam_source_epoch_rebaseline_contract",
        action_id=action_id,
        source_rebaseline=evidence,
    )
    action = json.loads(prepared.action.read_text(encoding="utf-8"))
    manifest = json.loads(prepared.manifest.read_text(encoding="utf-8"))
    rebaseline = action["intentional_source_rebaseline"]

    assert action["source_epoch"] == new_epoch
    assert manifest["source_epoch"] == new_epoch
    assert manifest["intentional_source_rebaseline"] == rebaseline
    assert rebaseline["mode"] == "intentional_source_rebaseline"
    assert rebaseline["scientific_action_id"] == action_id
    assert all(rebaseline["checks"].values())
    assert set(rebaseline["artifacts"]) == {
        "predecessor_submission_receipt",
        "predecessor_package_manifest",
        "predecessor_epoch_closure_receipt",
    }
    assert (
        "verify_action_id_absent_from_prior_packages_and_scheduler_history"
        not in manifest["required_remote_preflight"]
    )
    assert (
        "verify_composite_source_epoch_action_id_absent_from_prior_packages_and_scheduler_history"
        in manifest["required_remote_preflight"]
    )
    assert (
        "verify_historical_same_action_ids_use_distinct_source_epochs_and_bind_immediate_predecessor"
        in manifest["required_remote_preflight"]
    )
    assert action["qiskit_compilation_requested"] is False
    assert manifest["checks"]["qiskit_compilation_requested"] is False
    submit = prepared.submit.read_text(encoding="utf-8")
    assert "+SIAMIntentionalSourceRebaseline = True" in submit
    siam_chtc_package_module._verify_manifest_files(manifest, repo_root=root)
    siam_chtc_package_module._verify_source_epoch_control_plane(
        action=action, manifest=manifest, repo_root=root
    )


def test_source_rebaseline_rejects_unclosed_predecessor(package_repo) -> None:
    root, _seed, _image = package_repo
    source_tree = _test_source_tree_contract(root, REVISION)
    new_epoch = build_siam_source_epoch_contract(
        source_revision=REVISION,
        source_tree_files_sha256=str(source_tree["files_sha256"]),
    )
    evidence = _source_rebaseline_evidence(
        root,
        new_source_epoch_id=new_epoch["source_epoch_id"],
        action_id="screen_avqds_unclosed_predecessor",
        closure_status="SUBMITTED",
    )
    with pytest.raises(ValueError, match="not closed/superseded"):
        _prepare(
            package_repo,
            package_id="siam_unclosed_source_epoch_rebaseline",
            action_id="screen_avqds_unclosed_predecessor",
            source_rebaseline=evidence,
        )


def test_source_rebaseline_rejects_same_source_revision(package_repo) -> None:
    root, _seed, _image = package_repo
    source_tree = _test_source_tree_contract(root, REVISION)
    new_epoch = build_siam_source_epoch_contract(
        source_revision=REVISION,
        source_tree_files_sha256=str(source_tree["files_sha256"]),
    )
    evidence = _source_rebaseline_evidence(
        root,
        new_source_epoch_id=new_epoch["source_epoch_id"],
        action_id="screen_avqds_same_source",
        predecessor_revision=REVISION,
    )
    with pytest.raises(ValueError, match="new source revision"):
        _prepare(
            package_repo,
            package_id="siam_same_source_epoch_rebaseline",
            action_id="screen_avqds_same_source",
            source_rebaseline=evidence,
        )


def test_apm_alias_packages_one_exact_noninitial_rung(package_repo) -> None:
    prepared = _prepare(
        package_repo,
        package_id="siam_apm_screen_contract",
        action_id="screen_full_apm_rung1",
        arm="apm",
        cut=SIAM_NB4_ACTIVATION_CUT_LADDER[1],
    )
    action = json.loads(prepared.action.read_text(encoding="utf-8"))

    assert action["arm"] == "exchange"
    assert action["method_label"] == "full_apm"
    assert action["activation_cut_ladder_index"] == 1
    assert action["expected_run_id"].endswith("exchange_cut0.00333333333333")


def _screen_payload(
    *, seed: Path, arm: str, source_revision: str
) -> dict[str, object]:
    seed_manifest = load_siam_seed_manifest(seed)
    block_count = int(seed_manifest.pool["block_count"])
    config = seed_manifest.physical_config()
    state = np.asarray(
        build_initial_state(
            int(config.n_bath), hopping=float(config.hopping)
        ),
        dtype=complex,
    )
    psi_real = [float(value) for value in state.real]
    psi_imag = [float(value) for value in state.imag]
    h_post = np.asarray(
        hamiltonian_matrix(
            build_siam_hamiltonian(
                int(config.n_bath),
                u=float(config.u),
                v=float(config.v_final),
                hopping=float(config.hopping),
            )
        ),
        dtype=complex,
    )
    initial_energy = float(np.real(np.vdot(state, h_post @ state)))
    times = SIAM_NB4_SCREEN_SEGMENT.times
    runtime_input = load_siam_runtime_input(seed)
    runtime_state = state_from_scaffold_runtime_input(
        runtime_input,
        parameterization_mode=AP_PARAMETERIZATION_LOGICAL_SHARED,
    )
    controller_state = ControllerResumeState(
        checkpoint_index=len(times) - 1,
        prune_runtime_state={
            "support_identity_hash": (
                siam_chtc_package_module._support_identity_hash(runtime_state)
            ),
            "loss_history": {},
            "conditioning_history": {},
            "cooldown_until_index": {},
            "accepted_commit_count": 0,
            "accumulated_drift": 0.0,
            "accumulated_drift_time": times[-2],
            "last_support_transition_metadata": {},
        },
        repair_hold_state=None,
        previous_accepted_theta_dot=(),
    ).to_json_dict()
    resume_state = build_resume_state(
        runtime_state,
        theta_runtime=runtime_state.theta_runtime,
        time=times[-1],
        seed_artifact_json=str(seed),
        controller_state=controller_state,
    )
    command = build_siam_nb4_screen_run(
        seed_path=str(seed),
        arm=arm,
        activation_cut=1.0e-3,
        output_json="raw_outputs/test/run.json",
        code_revision=source_revision,
    )
    execution_contract = siam_chtc_package_module._execution_contract_from_argv(
        command.argv(),
        seed_path=seed,
        resume_payload=None,
    )
    numerics = {
        "integrator_method": "euler",
        "inverse_policy": {
            "pinv_rcond": 1.0e-10,
            "ridge_lambda": 1.0e-6,
            "solve_damping": 0.0,
        },
        "solve_repair": execution_contract["solve_repair_config"],
    }
    return {
        "schema": "ap_mclachlan_append_from_adapt_artifact_v1",
        "run_lock": {
            "schema": "paper_ii_run_lock_v4",
            "code_revision": source_revision,
            "numerics": numerics,
            "physics": {
                "seed_sha256": _sha256(seed),
                "family_key": "siam",
                "n_ph_max": None,
                "drive_profile": None,
                "exact_reference_json": None,
                "exact_reference_sha256": None,
                "time_grid": {
                    "t_initial": times[0],
                    "t_final": times[-1],
                    "point_count": len(times),
                    "ordered_values_sha256": hashlib.sha256(
                        json.dumps(
                            [float(value) for value in times],
                            separators=(",", ":"),
                            allow_nan=False,
                        ).encode("utf-8")
                    ).hexdigest(),
                },
            },
            "policy": {
                "structural_policy": arm,
                "guards": {
                    "max_structural_pool_size": 64,
                    "max_joint_patch_evaluations": 50_000,
                    "max_certification_attempts_per_level": 12,
                    "max_certification_attempts_per_deletion_branch": 2,
                    "max_insertion_batch_size": 1,
                    "residual_ratio_threshold": 0.02,
                },
            },
            "shared_protocol": {
                "numerics": numerics,
                "time_step_controller": {
                    "composed_parameter_step_max": None,
                    "local_subdivision_enabled": True,
                    "max_local_subdivisions": 11,
                    "parameter_displacement_max": None,
                    "state_motion_l2_step_max": 5.0e-5,
                },
                "candidate_pool": execution_contract[
                    "candidate_pool_contract"
                ],
            },
        },
        "decision_data_flow": {
            "uses_exact_reference_for_decision": False,
            "uses_future_exact_forecast_for_decision": False,
            "uses_reference_for_decision": False,
        },
        "trajectory": {
            "point_count": len(times),
            "points": [
                {
                    "index": index,
                    "time": time,
                    "theta_runtime": [],
                    "fixed_step": {"theta_dot": []},
                    "integration_to_next": (
                        None
                        if index == len(times) - 1
                        else {
                            "repair_summary": {
                                "effective_ridge_lambdas": [1.0e-6],
                                "effective_pinv_rconds": [1.0e-10],
                                "effective_solve_dampings": [0.0],
                            }
                        }
                    ),
                }
                for index, time in enumerate(times)
            ],
            "controller_resume_state": controller_state,
            "metadata": {
                "reference_energy_json": None,
                "seed_reference_energy_json": None,
                "uses_exact_reference_for_decision": False,
                "uses_future_exact_forecast_for_decision": False,
                "uses_reference_for_decision": False,
            },
        },
        "plot_rows": [
            {
                "index": index,
                "time": time,
                "psi_real": psi_real,
                "psi_imag": psi_imag,
                "energy_expectation": initial_energy,
            }
            for index, time in enumerate(times)
        ],
        "summary": {
            "candidate_pool_complete": True,
            "candidate_pool_term_count": block_count,
            "uncured_local_subdivision_count": 0,
            "parameterization_mode": "logical_shared",
            "support_patch_config": execution_contract[
                "support_patch_config"
            ],
        },
        "initial_state": {
            "candidate_pool_complete": True,
            "candidate_pool_term_count": block_count,
        },
        "final_state": runtime_state.to_json_dict(),
        "resume_state": resume_state,
    }


def _write_resume(path: Path, *, seed: Path, arm: str) -> None:
    payload = _screen_payload(
        seed=seed,
        arm=arm,
        source_revision=RESUME_REVISION,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _set_nonseed_terminal_resume(
    payload: dict[str, object],
    *,
    seed: Path,
    theta_value: float = 0.2,
) -> dict[str, object]:
    """Give a screen fixture one valid non-seed terminal coordinate/state."""

    runtime_input = load_siam_runtime_input(seed)
    initial = state_from_scaffold_runtime_input(
        runtime_input,
        parameterization_mode=AP_PARAMETERIZATION_LOGICAL_SHARED,
    )
    terminal, _theta_zero = state_with_appended_runtime_coordinates(
        initial,
        [runtime_input.candidate_pool_terms[0]],
        coordinate_labels=["fixture_terminal::logical::generator"],
    )
    theta = np.asarray([float(theta_value)], dtype=float)
    statevector = terminal.prepare_state(theta)
    points = payload["trajectory"]["points"]
    terminal_index = int(points[-1]["index"])
    controller = ControllerResumeState(
        checkpoint_index=terminal_index,
        prune_runtime_state={
            "support_identity_hash": (
                siam_chtc_package_module._support_identity_hash(terminal)
            ),
            "loss_history": {},
            "conditioning_history": {},
            "cooldown_until_index": {},
            "accepted_commit_count": 1,
            "accumulated_drift": 0.0,
            "accumulated_drift_time": float(points[-2]["time"]),
            "last_support_transition_metadata": {
                "prune_history_transition": "append_cleared_geometry_history",
            },
        },
        repair_hold_state=None,
        previous_accepted_theta_dot=(0.0,),
    ).to_json_dict()
    points[-2]["fixed_step"]["theta_dot"] = [0.0]
    points[-1]["theta_runtime"] = theta.tolist()
    payload["trajectory"]["controller_resume_state"] = controller
    terminal_row = payload["plot_rows"][-1]
    terminal_row["psi_real"] = [float(value) for value in statevector.real]
    terminal_row["psi_imag"] = [float(value) for value in statevector.imag]
    config = load_siam_seed_manifest(seed).physical_config()
    h_post = np.asarray(
        hamiltonian_matrix(
            build_siam_hamiltonian(
                int(config.n_bath),
                u=float(config.u),
                v=float(config.v_final),
                hopping=float(config.hopping),
            )
        ),
        dtype=complex,
    )
    terminal_row["energy_expectation"] = float(
        np.real(np.vdot(statevector, h_post @ statevector))
    )
    payload["final_state"] = terminal.to_json_dict()
    payload["resume_state"] = build_resume_state(
        terminal,
        theta_runtime=theta,
        time=float(terminal_row["time"]),
        seed_artifact_json=str(seed),
        controller_state=controller,
    )
    return payload


def _write_completion_receipt(
    path: Path,
    *,
    trajectory: Path,
    seed: Path,
    arm: str,
    activation_cut: float = 1.0e-3,
    status: str = "COMPLETED",
    search_profile_id: str | None = None,
) -> None:
    root = seed.parents[1]
    seed_rel = seed.relative_to(root).as_posix()
    trajectory_rel = trajectory.relative_to(root).as_posix()
    command = build_siam_nb4_screen_run(
        seed_path=seed_rel,
        arm=arm,
        activation_cut=activation_cut,
        output_json=trajectory_rel,
        code_revision=RESUME_REVISION,
        search_profile_id=search_profile_id,
    )
    runner_argv = list(command.argv())
    runner_argv_sha256 = _argv_sha256(runner_argv)
    source_tree = _test_source_tree_contract(root, RESUME_REVISION)
    package_id = f"source-screen-{arm}"
    action_id = f"source_screen_{arm}"
    action_contract = {
        "schema": SIAM_NB4_CHTC_ACTION_SCHEMA_V3,
        "package_id": package_id,
        "action_id": action_id,
        "source_revision": RESUME_REVISION,
        "source_tree": source_tree,
        "arm": arm,
        "segment": "screen_t0_t1_101",
        "activation_cut": activation_cut,
        "activation_cut_ladder_index": SIAM_NB4_ACTIVATION_CUT_LADDER.index(
            activation_cut
        ),
        "activation_cut_ladder_sha256": SIAM_NB4_CANONICAL_LADDER_SHA256,
        "seed_path": seed_rel,
        "seed_sha256": _sha256(seed),
        "trajectory_json": trajectory_rel,
        "expected_run_id": command.run_id,
        "expected_runner_argv_sha256": runner_argv_sha256,
        "expected_execution_contract": {
            "support_patch_config": {"fixture": "coherent"},
            "solve_repair_config": {"fixture": "coherent"},
            "candidate_pool_contract": {"fixture": "coherent"},
        },
        "exact_reference_blind": True,
        "compile_terminal_cost": False,
        "qiskit_compilation_requested": False,
    }
    if search_profile_id is not None:
        action_contract["search_profile_id"] = search_profile_id
    action_json_sha256 = _pretty_json_sha256(action_contract)
    package_manifest_contract = {
        "schema": SIAM_NB4_CHTC_MANIFEST_SCHEMA_V3,
        "package_id": package_id,
        "action_id": action_id,
        "source_revision": RESUME_REVISION,
        "action_json_sha256": action_json_sha256,
        "runner_argv_sha256": runner_argv_sha256,
        "expected_run_id": command.run_id,
        "expected_output": trajectory_rel,
        "source_tree_contract": "siam_source_tree_contract_v1",
        "source_tree_files_sha256": source_tree["files_sha256"],
        "source_tree_file_count": source_tree["file_count"],
        "activation_cut_ladder": list(
            ladder_payload(SIAM_NB4_ACTIVATION_CUT_LADDER)
        ),
        "activation_cut_ladder_sha256": SIAM_NB4_CANONICAL_LADDER_SHA256,
        "files": {"action": {"sha256": action_json_sha256}},
    }
    if search_profile_id is not None:
        package_manifest_contract["search_profile_id"] = search_profile_id
    trajectory_payload = json.loads(trajectory.read_text(encoding="utf-8"))
    resume_checks, resume_terminal_proof = (
        siam_chtc_package_module._assert_completed_terminal_resume(
            trajectory_payload,
            seed_runtime_input=load_siam_runtime_input(seed),
        )
    )
    payload = {
        "schema": "siam_nb4_chtc_action_receipt_v3",
        "status": status,
        "package_id": package_id,
        "action_id": action_id,
        "run_id": command.run_id,
        "arm": arm,
        "activation_cut": activation_cut,
        "seed_path": seed_rel,
        "seed_sha256": _sha256(seed),
        "source_revision": RESUME_REVISION,
        "action_json_sha256": action_json_sha256,
        "package_manifest_sha256": _pretty_json_sha256(
            package_manifest_contract
        ),
        "source_tree_files_sha256": source_tree["files_sha256"],
        "source_tree_file_count": source_tree["file_count"],
        "segment": "screen_t0_t1_101",
        "exact_reference_blind": True,
        "qiskit_compilation_requested": False,
        "runner_argv": runner_argv,
        "runner_argv_sha256": runner_argv_sha256,
        "trajectory_json": trajectory_rel,
        "trajectory_sha256": _sha256(trajectory),
        "trajectory_bytes": trajectory.stat().st_size,
        "resume_terminal_proof": resume_terminal_proof,
        "action_contract": action_contract,
        "package_manifest_contract": package_manifest_contract,
        **(
            {}
            if search_profile_id is None
            else {"search_profile_id": search_profile_id}
        ),
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
            "initial_seed_state": True,
            "initial_seed_energy": True,
            "state_norm_sector": True,
            **resume_checks,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_public_completion_receipt_validator_requires_full_registered_command(
    package_repo,
) -> None:
    root, seed, _image = package_repo
    trajectory = root / "raw_outputs" / "validator" / "run.json"
    _write_resume(trajectory, seed=seed, arm="avqds")
    receipt_path = trajectory.with_name("action_receipt.json")
    _write_completion_receipt(
        receipt_path,
        trajectory=trajectory,
        seed=seed,
        arm="avqds",
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    verified = validate_siam_nb4_completion_receipt(
        receipt,
        expected_source_revision=RESUME_REVISION,
        expected_seed_sha256=_sha256(seed),
        expected_trajectory_sha256=_sha256(trajectory),
        expected_arm="avqds",
        expected_segment="screen",
    )

    assert verified["action_id"] == "source_screen_avqds"
    assert verified["segment"] == SIAM_NB4_SCREEN_SEGMENT.segment_id
    assert verified["trajectory_bytes"] == trajectory.stat().st_size
    assert verified["resume_terminal_proof"]["schema"] == (
        "siam_nb4_resume_terminal_proof_v1"
    )


@pytest.mark.parametrize(
    "field",
    [
        "package_id",
        "run_id",
        "activation_cut",
        "segment",
        "runner_argv",
        "runner_argv_sha256",
        "trajectory_bytes",
        "action_json_sha256",
        "package_manifest_sha256",
        "source_tree_files_sha256",
        "source_tree_file_count",
        "action_contract",
        "package_manifest_contract",
        "resume_terminal_proof",
    ],
)
def test_public_completion_receipt_validator_rejects_missing_v3_fields(
    package_repo, field: str
) -> None:
    root, seed, _image = package_repo
    trajectory = root / "raw_outputs" / f"missing_{field}" / "run.json"
    _write_resume(trajectory, seed=seed, arm="avqds")
    receipt_path = trajectory.with_name("action_receipt.json")
    _write_completion_receipt(
        receipt_path,
        trajectory=trajectory,
        seed=seed,
        arm="avqds",
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop(field)

    with pytest.raises((ValueError, SystemExit)):
        validate_siam_nb4_completion_receipt(receipt)


def test_continuation_copies_and_binds_verified_screen_resume(
    package_repo,
) -> None:
    root, seed, _image = package_repo
    resume = root / "inputs" / "exchange_screen_run.json"
    _write_resume(resume, seed=seed, arm="exchange")
    source_receipt = resume.with_name("action_receipt.json")
    _write_completion_receipt(
        source_receipt,
        trajectory=resume,
        seed=seed,
        arm="exchange",
        activation_cut=1.0e-4,
    )
    prepared = _prepare(
        package_repo,
        package_id="siam_apm_continuation_contract",
        action_id="continue_full_apm_rung2",
        arm="full_apm",
        segment="continuation",
        cut=1.0e-4,
        resume=resume,
        resume_receipt=source_receipt,
    )
    action = json.loads(prepared.action.read_text(encoding="utf-8"))
    manifest = json.loads(prepared.manifest.read_text(encoding="utf-8"))

    assert prepared.resume is not None
    assert prepared.resume_action_receipt is not None
    assert prepared.resume_action_receipt.name == "action_receipt.json"
    assert prepared.resume.read_bytes() == resume.read_bytes()
    assert prepared.resume_action_receipt.read_bytes() == source_receipt.read_bytes()
    assert action["resume"]["sha256"] == _sha256(prepared.resume)
    assert action["resume"]["source_action_receipt_sha256"] == _sha256(
        prepared.resume_action_receipt
    )
    assert action["resume"]["source_action_id"] == "source_screen_exchange"
    assert action["resume"]["source_revision"] == RESUME_REVISION
    assert action["resume"]["packaged_action_receipt_path"].endswith(
        "/input/resume/action_receipt.json"
    )
    assert action["resume"]["point_count"] == 101
    assert action["segment"] == "continue_t1_t5_401"
    assert action["expected_point_count"] == 401
    assert action["expected_time_initial"] == 1.0
    assert action["expected_time_final"] == 5.0
    assert manifest["checks"]["resume_input_present"] is True
    assert manifest["files"]["resume"]["sha256"] == _sha256(prepared.resume)
    assert manifest["files"]["resume_action_receipt"]["sha256"] == _sha256(
        prepared.resume_action_receipt
    )
    assert manifest["files"]["resume_action_receipt"]["path"].endswith(
        "/input/resume/action_receipt.json"
    )


def test_runtime_accepts_packaged_continuation_completion_receipt(
    package_repo, monkeypatch
) -> None:
    root, seed, _image = package_repo
    resume = root / "inputs" / "runtime_continue" / "screen_run.json"
    _write_resume(resume, seed=seed, arm="avqds")
    source_receipt = resume.with_name("action_receipt.json")
    _write_completion_receipt(
        source_receipt,
        trajectory=resume,
        seed=seed,
        arm="avqds",
    )
    prepared = _prepare(
        package_repo,
        package_id="runtime_continuation_contract",
        action_id="runtime_continuation_action",
        segment="continuation",
        resume=resume,
        resume_receipt=source_receipt,
    )

    def runner_reached(_argv, *, check):
        assert check is True
        raise RuntimeError("runner reached")

    monkeypatch.setattr(
        siam_chtc_package_module.subprocess, "run", runner_reached
    )
    monkeypatch.chdir(root)
    with pytest.raises(RuntimeError, match="runner reached"):
        run_prepared_siam_action(
            package_dir=prepared.package_dir.relative_to(root),
            expected_action_id="runtime_continuation_action",
            expected_action_sha256=prepared.action_sha256,
            expected_source_revision=REVISION,
        )


@pytest.mark.parametrize("status", ["FAILED", "RUNNING", "UNKNOWN"])
def test_continuation_rejects_noncompleted_source_action_receipts(
    package_repo, status: str
) -> None:
    root, seed, _image = package_repo
    token = status.lower()
    resume = root / "inputs" / token / "screen_run.json"
    _write_resume(resume, seed=seed, arm="avqds")
    source_receipt = resume.with_name("action_receipt.json")
    _write_completion_receipt(
        source_receipt,
        trajectory=resume,
        seed=seed,
        arm="avqds",
        status=status,
    )
    with pytest.raises(ValueError, match="status COMPLETED"):
        _prepare(
            package_repo,
            package_id=f"reject_{token}_receipt",
            action_id=f"reject_{token}_receipt",
            segment="continuation",
            resume=resume,
            resume_receipt=source_receipt,
        )


def test_continuation_rejects_missing_source_action_receipt(
    package_repo,
) -> None:
    root, seed, _image = package_repo
    resume = root / "inputs" / "missing_receipt" / "screen_run.json"
    _write_resume(resume, seed=seed, arm="avqds")
    with pytest.raises(ValueError, match="explicit completed.*action_receipt"):
        _prepare(
            package_repo,
            package_id="reject_missing_receipt",
            action_id="reject_missing_receipt",
            segment="continuation",
            resume=resume,
        )


def test_continuation_rejects_legacy_incomplete_receipt_schema(
    package_repo,
) -> None:
    root, seed, _image = package_repo
    resume = root / "inputs" / "legacy_receipt" / "screen_run.json"
    _write_resume(resume, seed=seed, arm="avqds")
    source_receipt = resume.with_name("action_receipt.json")
    _write_completion_receipt(
        source_receipt,
        trajectory=resume,
        seed=seed,
        arm="avqds",
    )
    receipt = json.loads(source_receipt.read_text(encoding="utf-8"))
    receipt["schema"] = "siam_nb4_chtc_action_receipt_v2"
    source_receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match="requires a v3, v4, or v5 screen action receipt"
    ):
        _prepare(
            package_repo,
            package_id="reject_legacy_receipt",
            action_id="reject_legacy_receipt",
            segment="continuation",
            resume=resume,
            resume_receipt=source_receipt,
        )


def test_continuation_rejects_incomplete_checkpoint_statevectors(
    package_repo,
) -> None:
    root, seed, _image = package_repo
    resume = root / "inputs" / "short_statevector" / "screen_run.json"
    payload = _screen_payload(
        seed=seed,
        arm="avqds",
        source_revision=RESUME_REVISION,
    )
    payload["plot_rows"][0]["psi_real"] = [1.0, 0.0]
    payload["plot_rows"][0]["psi_imag"] = [0.0, 0.0]
    resume.parent.mkdir(parents=True, exist_ok=True)
    resume.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_receipt = resume.with_name("action_receipt.json")
    _write_completion_receipt(
        source_receipt,
        trajectory=resume,
        seed=seed,
        arm="avqds",
    )
    with pytest.raises(ValueError, match="complete recorded statevector"):
        _prepare(
            package_repo,
            package_id="reject_short_statevector",
            action_id="reject_short_statevector",
            segment="continuation",
            resume=resume,
            resume_receipt=source_receipt,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("trajectory_sha256", "trajectory SHA-256 mismatch"),
        ("ordered_time_grid", "affirmative source/time-grid checks"),
        ("runner_argv", "action contract differs from its receipt"),
    ],
)
def test_continuation_rejects_unbound_source_completion_receipt(
    package_repo, mutation: str, message: str
) -> None:
    root, seed, _image = package_repo
    resume = root / "inputs" / f"bad_receipt_{mutation}" / "screen_run.json"
    _write_resume(resume, seed=seed, arm="avqds")
    source_receipt = resume.with_name("action_receipt.json")
    _write_completion_receipt(
        source_receipt,
        trajectory=resume,
        seed=seed,
        arm="avqds",
    )
    receipt = json.loads(source_receipt.read_text(encoding="utf-8"))
    if mutation == "trajectory_sha256":
        receipt[mutation] = "0" * 64
    elif mutation == "runner_argv":
        argv = receipt["runner_argv"]
        for option in ("--insertion-l2-cut", "--avqds-l2-cut"):
            argv[argv.index(option) + 1] = "0.0003"
        receipt["runner_argv_sha256"] = _argv_sha256(argv)
    else:
        receipt["checks"][mutation] = False
    source_receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=message):
        _prepare(
            package_repo,
            package_id=f"reject_bad_receipt_{mutation}",
            action_id=f"reject_bad_receipt_{mutation}",
            segment="continuation",
            resume=resume,
            resume_receipt=source_receipt,
        )


def test_preparer_rejects_nonportable_seed(package_repo) -> None:
    root, _seed, image = package_repo
    legacy = root / "inputs" / "legacy_seed.json"
    write_siam_seed_manifest(legacy, PILOT_NB4, schema=SIAM_SEED_SCHEMA_V1)
    with mock.patch.object(
        siam_chtc_package_module, "_verify_source_revision"
    ), mock.patch.object(
        siam_chtc_package_module,
        "_build_source_tree_contract",
        return_value=_test_source_tree_contract(root, REVISION),
    ):
        with pytest.raises(ValueError, match="portable seed schema v2"):
            prepare_siam_nb4_chtc_action(
                repo_root=root,
                package_dir="chtc/legacy_seed_rejected",
                seed_path=legacy,
                image_path=image,
                source_revision=REVISION,
                action_id="legacy_seed_rejected",
                activation_cut=1.0e-3,
                arm="avqds",
                segment="screen",
            )


def test_preparer_rejects_an_unversioned_source_tree(package_repo) -> None:
    root, seed, image = package_repo
    with pytest.raises(ValueError, match="requires a Git worktree"):
        prepare_siam_nb4_chtc_action(
            repo_root=root,
            package_dir="chtc/unversioned_source_rejected",
            seed_path=seed,
            image_path=image,
            source_revision=REVISION,
            action_id="unversioned_source_rejected",
            activation_cut=1.0e-3,
            arm="avqds",
            segment="screen",
        )


@pytest.mark.parametrize("cut", [2.0e-4, 1.0e-8, 0.0, float("nan")])
def test_preparer_rejects_unregistered_activation_cuts(
    package_repo, cut: float
) -> None:
    with pytest.raises(ValueError, match="activation cut"):
        _prepare(
            package_repo,
            package_id=f"bad_cut_{str(cut).replace('.', '_')}",
            action_id="bad_cut",
            cut=cut,
        )


def test_preparer_never_replaces_package_or_raw_output(package_repo) -> None:
    prepared = _prepare(
        package_repo,
        package_id="immutable_package_contract",
        action_id="immutable_action",
    )
    with pytest.raises(FileExistsError, match="replace SIAM CHTC package"):
        _prepare(
            package_repo,
            package_id="immutable_package_contract",
            action_id="immutable_action",
        )

    root, _seed, _image = package_repo
    output = root / "raw_outputs" / "occupied_output_contract" / "occupied" / "run.json"
    output.parent.mkdir(parents=True)
    output.write_text("do not overwrite\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite SIAM raw output"):
        _prepare(
            package_repo,
            package_id="occupied_output_contract",
            action_id="occupied",
        )
    assert prepared.package_dir.is_dir()


def test_runtime_accepts_actual_mapping_trajectory_points_schema(
    package_repo, monkeypatch
) -> None:
    prepared = _prepare(
        package_repo,
        package_id="mapping_trajectory_contract",
        action_id="mapping_trajectory_action",
    )
    root, _seed, _image = package_repo
    action = json.loads(prepared.action.read_text(encoding="utf-8"))
    payload = _screen_payload(
        seed=prepared.seed,
        arm="avqds",
        source_revision=REVISION,
    )

    def fake_runner(argv, *, check):
        assert check is True
        assert "--output-json" in argv
        output = root / action["trajectory_json"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        siam_chtc_package_module.subprocess, "run", fake_runner
    )
    monkeypatch.chdir(root)
    assert run_prepared_siam_action(
        package_dir=prepared.package_dir.relative_to(root),
        expected_action_id="mapping_trajectory_action",
        expected_action_sha256=prepared.action_sha256,
        expected_source_revision=REVISION,
    ) == 0
    receipt = json.loads(
        (
            root
            / "raw_outputs"
            / "mapping_trajectory_contract"
            / "mapping_trajectory_action"
            / "action_receipt.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["status"] == "COMPLETED"
    assert receipt["checks"]["trajectory_points"] is True
    assert receipt["checks"]["ordered_time_grid"] is True
    assert receipt["checks"]["configuration_lock"] is True
    assert receipt["checks"]["segment_initial_state"] is True
    assert receipt["checks"]["segment_initial_energy"] is True
    assert receipt["checks"]["initial_seed_state"] is True
    assert receipt["checks"]["initial_seed_energy"] is True
    assert receipt["checks"]["state_norm_sector"] is True
    assert receipt["checks"]["resume_reconstruction"] is True
    assert receipt["checks"]["resume_support_parameters"] is True
    assert receipt["checks"]["resume_terminal_statevector"] is True
    assert receipt["checks"]["controller_resume_reconstruction"] is True
    assert receipt["checks"]["controller_resume_terminal_state"] is True
    assert receipt["resume_terminal_proof"]["schema"] == (
        "siam_nb4_resume_terminal_proof_v1"
    )
    assert receipt["resume_terminal_proof"]["logical_coordinate_count"] == 0
    assert receipt["action_json_sha256"] == prepared.action_sha256
    assert receipt["source_tree_file_count"] == 2


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("activation_cut", "frozen support policy"),
        ("controller", "time-step controller"),
        ("ridge", "inverse policy"),
    ),
)
def test_completed_run_binds_cut_controller_and_numerics(
    package_repo, mutation: str, message: str
) -> None:
    prepared = _prepare(
        package_repo,
        package_id=f"completed_config_{mutation}",
        action_id=f"completed_config_{mutation}",
    )
    action = json.loads(prepared.action.read_text(encoding="utf-8"))
    payload = _screen_payload(
        seed=prepared.seed,
        arm="avqds",
        source_revision=REVISION,
    )
    if mutation == "activation_cut":
        payload["summary"]["support_patch_config"]["avqds_l2_cut"] = 0.1
    elif mutation == "controller":
        payload["run_lock"]["shared_protocol"]["time_step_controller"][
            "state_motion_l2_step_max"
        ] = 1.0e-3
    elif mutation == "ridge":
        payload["run_lock"]["numerics"]["inverse_policy"][
            "ridge_lambda"
        ] = 1.0e-5
        payload["run_lock"]["shared_protocol"]["numerics"] = payload[
            "run_lock"
        ]["numerics"]
    else:  # pragma: no cover
        raise AssertionError(mutation)
    with pytest.raises(ValueError, match=message):
        siam_chtc_package_module._validate_completed_run(
            payload,
            action=action,
            seed_manifest=load_siam_seed_manifest(prepared.seed),
            seed_runtime_input=load_siam_runtime_input(prepared.seed),
        )


@pytest.mark.parametrize(
    ("field", "mutated"),
    (
        ("ridge_ladder", [1.0e-7]),
        ("pinv_rcond_ladder", [1.0e-11]),
        ("solve_damping_ladder", [1.0e-8]),
    ),
)
def test_completed_run_requires_canonical_configured_inverse_ladders(
    package_repo, field: str, mutated: list[float]
) -> None:
    prepared = _prepare(
        package_repo,
        package_id=f"completed_configured_ladder_{field}",
        action_id=f"completed_configured_ladder_{field}",
    )
    action = json.loads(prepared.action.read_text(encoding="utf-8"))
    payload = _screen_payload(
        seed=prepared.seed,
        arm="avqds",
        source_revision=REVISION,
    )
    action["expected_execution_contract"]["solve_repair_config"][field] = mutated
    payload["run_lock"]["numerics"]["solve_repair"][field] = mutated

    with pytest.raises(ValueError, match="canonical inverse ladder"):
        siam_chtc_package_module._validate_completed_run(
            payload,
            action=action,
            seed_manifest=load_siam_seed_manifest(prepared.seed),
            seed_runtime_input=load_siam_runtime_input(prepared.seed),
        )


@pytest.mark.parametrize(
    ("field", "mutated"),
    (
        ("effective_ridge_lambdas", [1.0e-7]),
        ("effective_pinv_rconds", [1.0e-11]),
        ("effective_solve_dampings", [1.0e-8]),
    ),
)
def test_completed_run_requires_pinned_effective_policy_on_every_interval(
    package_repo, field: str, mutated: list[float]
) -> None:
    prepared = _prepare(
        package_repo,
        package_id=f"completed_effective_policy_{field}",
        action_id=f"completed_effective_policy_{field}",
    )
    action = json.loads(prepared.action.read_text(encoding="utf-8"))
    payload = _screen_payload(
        seed=prepared.seed,
        arm="avqds",
        source_revision=REVISION,
    )
    penultimate = payload["trajectory"]["points"][-2]
    penultimate["integration_to_next"]["repair_summary"][field] = mutated

    with pytest.raises(ValueError, match="effective inverse policy"):
        siam_chtc_package_module._validate_completed_run(
            payload,
            action=action,
            seed_manifest=load_siam_seed_manifest(prepared.seed),
            seed_runtime_input=load_siam_runtime_input(prepared.seed),
        )


def test_completed_run_rejects_malformed_terminal_controller_resume(
    package_repo,
) -> None:
    prepared = _prepare(
        package_repo,
        package_id="completed_bad_terminal_controller",
        action_id="completed_bad_terminal_controller",
    )
    action = json.loads(prepared.action.read_text(encoding="utf-8"))
    payload = _screen_payload(
        seed=prepared.seed,
        arm="avqds",
        source_revision=REVISION,
    )
    payload["resume_state"]["controller_state"] = {
        "schema": "paper_ii_controller_resume_state_v1",
        "checkpoint_index": len(SIAM_NB4_SCREEN_SEGMENT.times) - 1,
        "prune_runtime_state": {},
        "repair_hold_state": {"malformed": True},
        "previous_accepted_theta_dot": [],
    }

    with pytest.raises(
        ValueError, match="controller resume cannot be reconstructed"
    ):
        siam_chtc_package_module._validate_completed_run(
            payload,
            action=action,
            seed_manifest=load_siam_seed_manifest(prepared.seed),
            seed_runtime_input=load_siam_runtime_input(prepared.seed),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("coordinate", "does not reproduce the terminal statevector"),
        ("theta", "angles differ from the terminal point"),
    ),
)
def test_completed_run_rejects_well_shaped_false_terminal_resume(
    package_repo, mutation: str, message: str
) -> None:
    prepared = _prepare(
        package_repo,
        package_id=f"completed_false_terminal_{mutation}",
        action_id=f"completed_false_terminal_{mutation}",
    )
    action = json.loads(prepared.action.read_text(encoding="utf-8"))
    payload = _screen_payload(
        seed=prepared.seed,
        arm="avqds",
        source_revision=REVISION,
    )
    _set_nonseed_terminal_resume(payload, seed=prepared.seed)
    if mutation == "coordinate":
        rotation = payload["resume_state"]["coordinates"][0]["rotations"][0]
        rotation["coeff_real"] = -float(rotation["coeff_real"])
    elif mutation == "theta":
        payload["resume_state"]["theta_runtime"][0] += 0.125
        payload["resume_state"]["theta_logical"][0] += 0.125
    else:  # pragma: no cover
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match=message):
        siam_chtc_package_module._validate_completed_run(
            payload,
            action=action,
            seed_manifest=load_siam_seed_manifest(prepared.seed),
            seed_runtime_input=load_siam_runtime_input(prepared.seed),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("state", "prepared seed"),
        ("norm", "norm/sector guards"),
        ("energy", "initial energy differs"),
    ),
)
def test_completed_run_binds_seed_state_energy_norm_and_sector(
    package_repo, mutation: str, message: str
) -> None:
    prepared = _prepare(
        package_repo,
        package_id=f"completed_boundary_{mutation}",
        action_id=f"completed_boundary_{mutation}",
    )
    action = json.loads(prepared.action.read_text(encoding="utf-8"))
    payload = _screen_payload(
        seed=prepared.seed,
        arm="avqds",
        source_revision=REVISION,
    )
    row = payload["plot_rows"][0]
    if mutation == "state":
        real = np.asarray(row["psi_real"], dtype=float)
        nonzero = np.flatnonzero(np.abs(real) > 1.0e-12)
        assert nonzero.size >= 2
        real[nonzero[0]] *= -1.0
        row["psi_real"] = real.tolist()
    elif mutation == "norm":
        row["psi_real"] = (2.0 * np.asarray(row["psi_real"])).tolist()
        row["psi_imag"] = (2.0 * np.asarray(row["psi_imag"])).tolist()
    elif mutation == "energy":
        row["energy_expectation"] += 1.0e-4
    else:  # pragma: no cover
        raise AssertionError(mutation)
    with pytest.raises(ValueError, match=message):
        siam_chtc_package_module._validate_completed_run(
            payload,
            action=action,
            seed_manifest=load_siam_seed_manifest(prepared.seed),
            seed_runtime_input=load_siam_runtime_input(prepared.seed),
        )


def test_completed_continuation_binds_resume_terminal_not_original_seed(
    package_repo,
) -> None:
    root, seed, _image = package_repo
    resume = root / "inputs" / "continuation_boundary" / "screen_run.json"
    resume_payload = _screen_payload(
        seed=seed,
        arm="avqds",
        source_revision=RESUME_REVISION,
    )
    _set_nonseed_terminal_resume(resume_payload, seed=seed)
    boundary = resume_payload["plot_rows"][-1]
    resume.parent.mkdir(parents=True, exist_ok=True)
    resume.write_text(
        json.dumps(resume_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_receipt = resume.with_name("action_receipt.json")
    _write_completion_receipt(
        source_receipt,
        trajectory=resume,
        seed=seed,
        arm="avqds",
    )
    prepared = _prepare(
        package_repo,
        package_id="completed_continuation_boundary",
        action_id="completed_continuation_boundary",
        segment="continuation",
        resume=resume,
        resume_receipt=source_receipt,
    )
    action = json.loads(prepared.action.read_text(encoding="utf-8"))

    payload = _screen_payload(
        seed=prepared.seed,
        arm="avqds",
        source_revision=REVISION,
    )
    times = SIAM_NB4_CONTINUATION_SEGMENT.times
    payload["run_lock"]["physics"]["time_grid"] = {
        "t_initial": times[0],
        "t_final": times[-1],
        "point_count": len(times),
        "ordered_values_sha256": hashlib.sha256(
            json.dumps(
                [float(value) for value in times],
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    }
    payload["trajectory"]["point_count"] = len(times)
    index_offset = int(
        resume_payload["resume_state"]["controller_state"]["checkpoint_index"]
    )
    terminal_theta = list(resume_payload["resume_state"]["theta_runtime"])
    payload["trajectory"]["points"] = [
        {
            "index": index_offset + index,
            "time": time,
            "theta_runtime": list(terminal_theta),
            "fixed_step": {"theta_dot": [0.0]},
            "integration_to_next": (
                None
                if index == len(times) - 1
                else {
                    "repair_summary": {
                        "effective_ridge_lambdas": [1.0e-6],
                        "effective_pinv_rconds": [1.0e-10],
                        "effective_solve_dampings": [0.0],
                    }
                }
            ),
        }
        for index, time in enumerate(times)
    ]
    payload["plot_rows"] = [
        {
            "index": index_offset + index,
            "time": time,
            "psi_real": list(boundary["psi_real"]),
            "psi_imag": list(boundary["psi_imag"]),
            "energy_expectation": boundary["energy_expectation"],
        }
        for index, time in enumerate(times)
    ]
    terminal_controller = json.loads(
        json.dumps(resume_payload["resume_state"]["controller_state"])
    )
    terminal_controller["checkpoint_index"] = index_offset + len(times) - 1
    terminal_controller["prune_runtime_state"]["accumulated_drift_time"] = (
        times[-2]
    )
    payload["trajectory"]["controller_resume_state"] = terminal_controller
    payload["resume_state"] = json.loads(
        json.dumps(resume_payload["resume_state"])
    )
    payload["resume_state"]["time"] = 5.0
    payload["resume_state"]["controller_state"] = terminal_controller
    payload["final_state"] = json.loads(
        json.dumps(resume_payload["final_state"])
    )

    checks, proof = siam_chtc_package_module._validate_completed_run(
        payload,
        action=action,
        seed_manifest=load_siam_seed_manifest(prepared.seed),
        seed_runtime_input=load_siam_runtime_input(prepared.seed),
        resume_payload=resume_payload,
    )
    assert checks["resume_boundary_state"] is True
    assert checks["resume_boundary_energy"] is True
    assert checks["resume_reconstruction"] is True
    assert proof["logical_coordinate_count"] == 1
    assert "initial_seed_state" not in checks

    original_seed_state = _screen_payload(
        seed=prepared.seed,
        arm="avqds",
        source_revision=REVISION,
    )["plot_rows"][0]
    payload["plot_rows"][0]["psi_real"] = original_seed_state["psi_real"]
    payload["plot_rows"][0]["psi_imag"] = original_seed_state["psi_imag"]
    with pytest.raises(ValueError, match="resume boundary"):
        siam_chtc_package_module._validate_completed_run(
            payload,
            action=action,
            seed_manifest=load_siam_seed_manifest(prepared.seed),
            seed_runtime_input=load_siam_runtime_input(prepared.seed),
            resume_payload=resume_payload,
        )


def test_runtime_rejects_source_tree_mutation_before_launch(
    package_repo, monkeypatch
) -> None:
    prepared = _prepare(
        package_repo,
        package_id="runtime_source_tree_tamper",
        action_id="runtime_source_tree_tamper",
    )
    root, _seed, _image = package_repo
    (root / "pipelines" / "test_source_marker.py").write_text(
        "SOURCE_LOCKED = False\n", encoding="utf-8"
    )
    monkeypatch.chdir(root)
    try:
        with pytest.raises(ValueError, match="source-tree bytes differ"):
            run_prepared_siam_action(
                package_dir=prepared.package_dir.relative_to(root),
                expected_action_id="runtime_source_tree_tamper",
                expected_action_sha256=prepared.action_sha256,
                expected_source_revision=REVISION,
            )
    finally:
        (root / "pipelines" / "test_source_marker.py").write_text(
            "SOURCE_LOCKED = True\n", encoding="utf-8"
        )


def test_runtime_manifest_preflight_detects_worker_tampering_before_launch(
    package_repo, monkeypatch
) -> None:
    prepared = _prepare(
        package_repo,
        package_id="runtime_tamper_contract",
        action_id="runtime_tamper_action",
    )
    prepared.worker.write_text(
        prepared.worker.read_text(encoding="utf-8") + "# tampered\n",
        encoding="utf-8",
    )
    root, _seed, _image = package_repo
    monkeypatch.chdir(root)
    with pytest.raises(ValueError, match="worker.*(wrong size|fails SHA-256)"):
        run_prepared_siam_action(
            package_dir=prepared.package_dir.relative_to(root),
            expected_action_id="runtime_tamper_action",
            expected_action_sha256=prepared.action_sha256,
            expected_source_revision=REVISION,
        )


def test_wrapper_detects_worker_tampering_before_container_launch(
    package_repo,
) -> None:
    prepared = _prepare(
        package_repo,
        package_id="wrapper_tamper_contract",
        action_id="wrapper_tamper_action",
    )
    prepared.worker.write_text(
        prepared.worker.read_text(encoding="utf-8") + "# tampered\n",
        encoding="utf-8",
    )
    root, _seed, image = package_repo
    completed = subprocess.run(
        [str(prepared.wrapper), "wrapper_tamper_action"],
        cwd=root,
        env={**os.environ, "PROJECT_IMAGE": str(image)},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "SIAM worker SHA-256 mismatch" in completed.stderr


def test_runtime_rejects_incomplete_manifest_digest_membership(
    package_repo, monkeypatch
) -> None:
    prepared = _prepare(
        package_repo,
        package_id="missing_manifest_row_contract",
        action_id="missing_manifest_row_action",
    )
    manifest = json.loads(prepared.manifest.read_text(encoding="utf-8"))
    del manifest["files"]["worker"]
    prepared.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    root, _seed, _image = package_repo
    monkeypatch.chdir(root)
    with pytest.raises(ValueError, match="wrong exact membership"):
        run_prepared_siam_action(
            package_dir=prepared.package_dir.relative_to(root),
            expected_action_id="missing_manifest_row_action",
            expected_action_sha256=prepared.action_sha256,
            expected_source_revision=REVISION,
        )


def test_runtime_rejects_top_level_manifest_digest_disagreement(
    package_repo, monkeypatch
) -> None:
    prepared = _prepare(
        package_repo,
        package_id="manifest_digest_disagreement_contract",
        action_id="manifest_digest_disagreement_action",
    )
    manifest = json.loads(prepared.manifest.read_text(encoding="utf-8"))
    manifest["worker_sha256"] = "0" * 64
    prepared.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    root, _seed, _image = package_repo
    monkeypatch.chdir(root)
    with pytest.raises(ValueError, match="worker_sha256 disagrees"):
        run_prepared_siam_action(
            package_dir=prepared.package_dir.relative_to(root),
            expected_action_id="manifest_digest_disagreement_action",
            expected_action_sha256=prepared.action_sha256,
            expected_source_revision=REVISION,
        )


# ---------------------------------------------------------------------------
# Compiled-Q exchange route (v5 lineage)
# ---------------------------------------------------------------------------

ROUTE_Q_COMPILED = "paper_ii_exchange_q_compiled_v1"
Q_COMPILED_CONTRACT_SHA256 = (
    "7883d93767e00523539d509abf92d126b0effd255eeefc8184a11f2636c356b1"
)
# Frozen from the pre-change packager on the identical fixture: a historical
# action must keep producing these bytes forever, because the r00--r07 lineage
# was submitted under them.
FROZEN_HISTORICAL_AVQDS_SCREEN = {
    "action_sha256": (
        "cbb2aae79d9654309cf406a09be1c76464767d7bfa405ce6cddf5a4e01e5ab88"
    ),
    "manifest_sha256": (
        "df6d589065f795a6e518561258bbd4f0035be6f06b9457a4e182064e33e2898e"
    ),
    "worker_sha256": (
        "171b0d58fb1c585c2f78a23d1cee3e5df99179b637950e3000dbedca6f6f6053"
    ),
    "wrapper_sha256": (
        "35bd15209cc37e0df8d1127343ac6b2c9e2e705b5784c2d42b7910777a480f7c"
    ),
    "submit_sha256": (
        "1c26e3224016b2ca150509043b00794dd520888666ce909249b65ed979ea701f"
    ),
}
FROZEN_HISTORICAL_EXCHANGE_SCREEN = {
    "action_sha256": (
        "04373e4c519e37fa1de73b58b7599cfa2f76aeee67e681358593f66ede947536"
    ),
    "manifest_sha256": (
        "55febf7ea80b6097ffbb7d851d94177d22533e4c5337f1d9ba3bd3cea8227cf9"
    ),
    "submit_sha256": (
        "36ee67db5e44586dc55d5167b6ffab2bf6550e57e8b45eab7cb97922da732f4c"
    ),
}


def _pinned_source_tree_contract(root: Path, revision: str) -> dict[str, object]:
    """The fixture source tree with pinned modes, so digests are umask-free."""

    marker = root / "pipelines" / "test_source_marker.py"
    source_marker = root / "src" / "test_source_marker.py"
    source_marker.parent.mkdir(parents=True, exist_ok=True)
    source_marker.write_text("SOURCE_LOCKED = True\n", encoding="utf-8")
    body = b"SOURCE_LOCKED = True\n"
    rows = [
        {
            "path": "pipelines/test_source_marker.py",
            "kind": "file",
            "mode": 0o644,
            "bytes": marker.stat().st_size,
            "sha256": hashlib.sha256(body).hexdigest(),
            "link_target": None,
        },
        {
            "path": "src/test_source_marker.py",
            "kind": "file",
            "mode": 0o644,
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
            "link_target": None,
        },
    ]
    encoded = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "schema": "siam_source_tree_contract_v1",
        "source_revision": revision,
        "file_count": len(rows),
        "files_sha256": hashlib.sha256(encoded).hexdigest(),
        "files": rows,
    }


def _prepare_pinned(
    package_repo,
    *,
    package_id: str,
    action_id: str,
    arm: str,
    route_id: str = "historical",
    cut: float = 1.0e-3,
    search_profile=SIAM_NB4_PILOT_SEARCH,
):
    root, seed, image = package_repo
    with mock.patch.object(
        siam_chtc_package_module, "_verify_source_revision"
    ), mock.patch.object(
        siam_chtc_package_module,
        "_build_source_tree_contract",
        return_value=_pinned_source_tree_contract(root, REVISION),
    ):
        return prepare_siam_nb4_chtc_action(
            repo_root=root,
            package_dir=Path("chtc") / package_id,
            seed_path=seed,
            image_path=image,
            source_revision=REVISION,
            action_id=action_id,
            activation_cut=cut,
            arm=arm,
            segment="screen",
            route_id=route_id,
            search_profile=search_profile,
            prepared_at="2026-08-27T00:00:00+00:00",
        )


def _batch_name(submit: Path) -> str:
    return [
        line.split("=", 1)[1].strip().strip('"')
        for line in submit.read_text(encoding="utf-8").splitlines()
        if line.startswith("+JobBatchName")
    ][0]


@pytest.mark.parametrize(
    "alias",
    ("exchange_q_compiled", "full_apm_q_compiled", "Full-APM-Q-Compiled"),
)
def test_compiled_q_aliases_resolve_to_the_registered_route_arm(
    alias: str,
) -> None:
    assert (
        siam_chtc_package_module._canonical_arm(alias) == "exchange_q_compiled"
    )


def test_canonical_arm_error_names_the_compiled_q_arm() -> None:
    with pytest.raises(ValueError, match="exchange_q_compiled"):
        siam_chtc_package_module._canonical_arm("gamma_compiled")


def test_compiled_q_arm_reports_exchange_policy_and_its_own_method_label() -> None:
    assert siam_chtc_package_module._arm_dynamics_policy(
        "exchange_q_compiled"
    ) == "exchange"
    assert siam_chtc_package_module._arm_method_label(
        "exchange_q_compiled"
    ) == "full_apm_q_compiled"
    assert siam_chtc_package_module._arm_method_label("exchange") == "full_apm"
    assert siam_chtc_package_module._arm_method_label("avqds") == "avqds"
    assert siam_chtc_package_module._arm_route_bundle_id(
        "exchange_q_compiled"
    ) == ROUTE_Q_COMPILED
    assert siam_chtc_package_module._route_contract_sha256(
        ROUTE_Q_COMPILED
    ) == Q_COMPILED_CONTRACT_SHA256


def test_compiled_q_package_carries_v5_route_identity(package_repo) -> None:
    from pipelines.time_dynamics.siam_nb4_ladder import siam_nb4_action_id

    action_id = siam_nb4_action_id(
        arm="full_apm",
        segment="screen",
        activation_cut=SIAM_NB4_ACTIVATION_CUT_LADDER[0],
        route_id=ROUTE_Q_COMPILED,
    )
    assert action_id == "siam-nb4-qc-full_apm-screen-r00-cut1em02"
    prepared = _prepare_pinned(
        package_repo,
        package_id="siam_nb4_qc_full_apm_r00",
        action_id=action_id,
        arm="exchange_q_compiled",
        route_id=ROUTE_Q_COMPILED,
        cut=SIAM_NB4_ACTIVATION_CUT_LADDER[0],
    )
    action = json.loads(prepared.action.read_text(encoding="utf-8"))
    manifest = json.loads(prepared.manifest.read_text(encoding="utf-8"))

    assert action["schema"] == SIAM_NB4_CHTC_ACTION_SCHEMA_V5
    assert manifest["schema"] == SIAM_NB4_CHTC_MANIFEST_SCHEMA_V5
    assert action["arm"] == "exchange_q_compiled"
    assert action["method_label"] == "full_apm_q_compiled"
    assert action["route_id"] == ROUTE_Q_COMPILED
    assert action["route_bundle_id"] == ROUTE_Q_COMPILED
    assert action["route_contract_sha256"] == Q_COMPILED_CONTRACT_SHA256
    for field in ("route_id", "route_bundle_id", "route_contract_sha256"):
        assert manifest[field] == action[field]
    assert action["activation_cut_ladder_index"] == 0
    assert "--exchange-route-bundle" in " ".join(
        json.dumps(action["expected_execution_contract"])
    ) or action["expected_execution_contract"][
        "support_patch_config"
    ]["dynamics_policy"] == "exchange"


def test_compiled_q_batch_name_matches_the_submission_intent_bridge(
    package_repo,
) -> None:
    from pipelines.time_dynamics.siam_nb4_ladder import siam_nb4_action_id
    from pipelines.time_dynamics.siam_submission_intent_bridge import (
        _accepted_batch_names,
    )

    prepared = _prepare_pinned(
        package_repo,
        package_id="siam_nb4_qc_batch_name",
        action_id=siam_nb4_action_id(
            arm="full_apm",
            segment="screen",
            activation_cut=SIAM_NB4_ACTIVATION_CUT_LADDER[0],
            route_id=ROUTE_Q_COMPILED,
        ),
        arm="exchange_q_compiled",
        route_id=ROUTE_Q_COMPILED,
        cut=SIAM_NB4_ACTIVATION_CUT_LADDER[0],
    )
    batch = _batch_name(prepared.submit)
    assert batch == (
        "siam-nb4-screen-t0-t1-101-exchange_q_compiled-rung0"
    )
    assert batch in _accepted_batch_names(
        segment=SIAM_NB4_SCREEN_SEGMENT.segment_id,
        arm="exchange_q_compiled",
        ladder_index=0,
    )


def test_avqds_on_the_compiled_q_route_is_route_free_but_v5(
    package_repo,
) -> None:
    from pipelines.time_dynamics.siam_nb4_ladder import siam_nb4_action_id
    from pipelines.time_dynamics.siam_submission_intent_bridge import (
        _accepted_batch_names,
    )

    action_id = siam_nb4_action_id(
        arm="avqds",
        segment="screen",
        activation_cut=SIAM_NB4_ACTIVATION_CUT_LADDER[0],
        route_id=ROUTE_Q_COMPILED,
    )
    assert action_id == "siam-nb4-qc-avqds-screen-r00-cut1em02"
    prepared = _prepare_pinned(
        package_repo,
        package_id="siam_nb4_qc_avqds_r00",
        action_id=action_id,
        arm="avqds",
        route_id=ROUTE_Q_COMPILED,
        cut=SIAM_NB4_ACTIVATION_CUT_LADDER[0],
    )
    action = json.loads(prepared.action.read_text(encoding="utf-8"))
    assert action["schema"] == SIAM_NB4_CHTC_ACTION_SCHEMA_V5
    assert action["route_id"] == ROUTE_Q_COMPILED
    assert action["route_bundle_id"] == "not_applicable"
    assert action["route_contract_sha256"] is None
    assert action["method_label"] == "avqds"
    batch = _batch_name(prepared.submit)
    assert batch == "siam-nb4-screen-t0-t1-101-avqds-rung0"
    assert batch in _accepted_batch_names(
        segment=SIAM_NB4_SCREEN_SEGMENT.segment_id,
        arm="avqds",
        ladder_index=0,
    )


def test_compiled_q_route_rejects_a_nonladder_action_id(package_repo) -> None:
    with pytest.raises(ValueError, match="must be the ladder identity"):
        _prepare_pinned(
            package_repo,
            package_id="siam_nb4_qc_bad_action_id",
            action_id="siam-nb4-full_apm-screen-r00-cut1em02",
            arm="exchange_q_compiled",
            route_id=ROUTE_Q_COMPILED,
            cut=SIAM_NB4_ACTIVATION_CUT_LADDER[0],
        )


def test_historical_arm_is_refused_on_the_compiled_q_route(
    package_repo,
) -> None:
    from pipelines.time_dynamics.siam_nb4_ladder import siam_nb4_action_id

    with pytest.raises(ValueError, match="not the packaged ladder route"):
        _prepare_pinned(
            package_repo,
            package_id="siam_nb4_qc_wrong_arm",
            action_id=siam_nb4_action_id(
                arm="full_apm",
                segment="screen",
                activation_cut=SIAM_NB4_ACTIVATION_CUT_LADDER[0],
                route_id=ROUTE_Q_COMPILED,
            ),
            arm="exchange",
            route_id=ROUTE_Q_COMPILED,
            cut=SIAM_NB4_ACTIVATION_CUT_LADDER[0],
        )


def test_unregistered_route_id_is_refused(package_repo) -> None:
    with pytest.raises(ValueError, match="package route must be one of"):
        _prepare_pinned(
            package_repo,
            package_id="siam_nb4_unknown_route",
            action_id="siam-nb4-avqds-screen-r00-cut1em02",
            arm="avqds",
            route_id="paper_ii_exchange_gamma_compiled_v1",
            cut=SIAM_NB4_ACTIVATION_CUT_LADDER[0],
        )


def _compiled_q_screen_payload(seed: Path) -> dict[str, object]:
    payload = _screen_payload(
        seed=seed,
        arm="exchange_q_compiled",
        source_revision=REVISION,
    )
    policy = payload["run_lock"]["policy"]
    policy["structural_policy"] = "exchange"
    policy["guards"]["exchange_route_contract"] = {
        "bundle_id": ROUTE_Q_COMPILED,
        "objective_id": "realized_q",
        "cost_model_id": "compiled_population",
        "kernel_id": "shifted_paper_i",
        "compiler": None,
    }
    policy["guards"]["exchange_route_contract_sha256"] = (
        Q_COMPILED_CONTRACT_SHA256
    )
    return payload


def test_completed_configuration_accepts_a_compiled_q_run_lock(
    package_repo,
) -> None:
    from pipelines.time_dynamics.siam_nb4_ladder import siam_nb4_action_id

    prepared = _prepare_pinned(
        package_repo,
        package_id="siam_nb4_qc_config_ok",
        action_id=siam_nb4_action_id(
            arm="full_apm",
            segment="screen",
            activation_cut=1.0e-3,
            route_id=ROUTE_Q_COMPILED,
        ),
        arm="exchange_q_compiled",
        route_id=ROUTE_Q_COMPILED,
        cut=1.0e-3,
    )
    action = json.loads(prepared.action.read_text(encoding="utf-8"))
    siam_chtc_package_module._assert_completed_configuration(
        _compiled_q_screen_payload(prepared.seed), action=action
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("bundle", "different exchange route"),
        ("digest", "changed the exchange route contract"),
        ("absent", "records no exchange route contract"),
        ("policy", "resolved the wrong arm"),
    ),
)
def test_completed_configuration_rejects_a_mismatched_compiled_q_route(
    package_repo, mutation: str, message: str
) -> None:
    from pipelines.time_dynamics.siam_nb4_ladder import siam_nb4_action_id

    prepared = _prepare_pinned(
        package_repo,
        package_id=f"siam_nb4_qc_config_{mutation}",
        action_id=siam_nb4_action_id(
            arm="full_apm",
            segment="screen",
            activation_cut=1.0e-3,
            route_id=ROUTE_Q_COMPILED,
        ),
        arm="exchange_q_compiled",
        route_id=ROUTE_Q_COMPILED,
        cut=1.0e-3,
    )
    action = json.loads(prepared.action.read_text(encoding="utf-8"))
    payload = _compiled_q_screen_payload(prepared.seed)
    guards = payload["run_lock"]["policy"]["guards"]
    if mutation == "bundle":
        guards["exchange_route_contract"]["bundle_id"] = (
            "paper_ii_generalized_exchange_v2"
        )
    elif mutation == "digest":
        guards["exchange_route_contract_sha256"] = "5" * 64
    elif mutation == "absent":
        guards.pop("exchange_route_contract")
    elif mutation == "policy":
        payload["run_lock"]["policy"]["structural_policy"] = "avqds"
    else:  # pragma: no cover
        raise AssertionError(mutation)
    with pytest.raises(ValueError, match=message):
        siam_chtc_package_module._assert_completed_configuration(
            payload, action=action
        )


@pytest.mark.parametrize(
    ("arm", "action_id", "package_id", "frozen"),
    (
        (
            "avqds",
            "siam-nb4-avqds-screen-r00-cut1em03",
            "siam_nb4_frozen_avqds",
            FROZEN_HISTORICAL_AVQDS_SCREEN,
        ),
        (
            "exchange",
            "siam-nb4-full_apm-screen-r00-cut1em03",
            "siam_nb4_frozen_exchange",
            FROZEN_HISTORICAL_EXCHANGE_SCREEN,
        ),
    ),
)
def test_historical_packaging_output_is_byte_frozen(
    package_repo,
    arm: str,
    action_id: str,
    package_id: str,
    frozen: dict[str, str],
) -> None:
    """The v4 lineage must never move: r00--r07 were submitted under it."""

    prepared = _prepare_pinned(
        package_repo,
        package_id=package_id,
        action_id=action_id,
        arm=arm,
    )
    action = json.loads(prepared.action.read_text(encoding="utf-8"))
    manifest = json.loads(prepared.manifest.read_text(encoding="utf-8"))
    assert action["schema"] == SIAM_NB4_CHTC_ACTION_SCHEMA_V4
    assert manifest["schema"] == SIAM_NB4_CHTC_MANIFEST_SCHEMA_V4
    for field in ("route_id", "route_bundle_id", "route_contract_sha256"):
        assert field not in action
        assert field not in manifest
    assert prepared.action_sha256 == frozen["action_sha256"]
    assert prepared.manifest_sha256 == frozen["manifest_sha256"]
    assert _sha256(prepared.submit) == frozen["submit_sha256"]
    if "worker_sha256" in frozen:
        assert _sha256(prepared.worker) == frozen["worker_sha256"]
        assert _sha256(prepared.wrapper) == frozen["wrapper_sha256"]


# ---------------------------------------------------------------------------
# Declared search budget: the exchange-insertion frontier
# ---------------------------------------------------------------------------

EXCHANGE_PROFILE_ID = SIAM_NB4_PILOT_EXCHANGE_SEARCH.profile_id


def test_exchange_frontier_profile_matches_the_pilot_budget_exactly() -> None:
    """Only the exchange face differs; the work limits are the same numbers."""

    assert SIAM_NB4_PILOT_EXCHANGE_SEARCH.exchange_insertion_frontier == 8
    assert SIAM_NB4_PILOT_SEARCH.exchange_insertion_frontier is None
    for field in (
        "pool_size",
        "joint_patch_evaluations",
        "certification_attempts_per_level",
        "certification_attempts_per_deletion_branch",
        "rounds_per_checkpoint",
        "insertion_cardinality",
    ):
        assert getattr(SIAM_NB4_PILOT_EXCHANGE_SEARCH, field) == getattr(
            SIAM_NB4_PILOT_SEARCH, field
        )
    assert SIAM_NB4_PILOT_EXCHANGE_SEARCH.profile_id != (
        SIAM_NB4_PILOT_SEARCH.profile_id
    )


def test_default_search_profile_emits_no_frontier_flag(package_repo) -> None:
    prepared = _prepare(
        package_repo,
        package_id="siam_screen_default_budget",
        action_id="screen_default_budget",
        arm="exchange",
    )
    action = json.loads(prepared.action.read_text(encoding="utf-8"))
    manifest = json.loads(prepared.manifest.read_text(encoding="utf-8"))

    assert "search_profile_id" not in action
    assert "search_profile_id" not in manifest
    support = action["expected_execution_contract"]["support_patch_config"]
    assert "exchange_insertion_frontier" not in support
    expected = build_siam_nb4_screen_run(
        seed_path=action["seed_path"],
        arm="exchange",
        activation_cut=action["activation_cut"],
        output_json=action["trajectory_json"],
        code_revision=REVISION,
    )
    assert "--exchange-insertion-frontier" not in expected.argv()
    assert action["expected_runner_argv_sha256"] == _argv_sha256(
        list(expected.argv())
    )


def test_frontier_search_profile_emits_the_flag_and_records_its_id(
    package_repo,
) -> None:
    prepared = _prepare(
        package_repo,
        package_id="siam_screen_frontier_budget",
        action_id="screen_frontier_budget",
        arm="exchange",
        search_profile=SIAM_NB4_PILOT_EXCHANGE_SEARCH,
    )
    action = json.loads(prepared.action.read_text(encoding="utf-8"))
    manifest = json.loads(prepared.manifest.read_text(encoding="utf-8"))

    assert action["search_profile_id"] == EXCHANGE_PROFILE_ID
    assert manifest["search_profile_id"] == EXCHANGE_PROFILE_ID
    support = action["expected_execution_contract"]["support_patch_config"]
    assert support["exchange_insertion_frontier"] == 8
    expected = build_siam_nb4_screen_run(
        seed_path=action["seed_path"],
        arm="exchange",
        activation_cut=action["activation_cut"],
        output_json=action["trajectory_json"],
        code_revision=REVISION,
        search_profile_id=EXCHANGE_PROFILE_ID,
    )
    argv = list(expected.argv())
    assert "--exchange-insertion-frontier" in argv
    assert argv[argv.index("--exchange-insertion-frontier") + 1] == "8"
    assert action["expected_runner_argv_sha256"] == _argv_sha256(argv)


def test_frontier_profile_separates_the_packaged_identity(package_repo) -> None:
    """The same rung under the two budgets is not the same packaged action."""

    default = _prepare(
        package_repo,
        package_id="siam_screen_budget_default",
        action_id="screen_budget_default",
        arm="exchange",
    )
    frontier = _prepare(
        package_repo,
        package_id="siam_screen_budget_frontier",
        action_id="screen_budget_frontier",
        arm="exchange",
        search_profile=SIAM_NB4_PILOT_EXCHANGE_SEARCH,
    )
    default_action = json.loads(default.action.read_text(encoding="utf-8"))
    frontier_action = json.loads(frontier.action.read_text(encoding="utf-8"))

    assert (
        default_action["expected_runner_argv_sha256"]
        != frontier_action["expected_runner_argv_sha256"]
    )
    assert default.action_sha256 != frontier.action_sha256


def test_compiled_q_frontier_package_carries_both_identities(
    package_repo,
) -> None:
    from pipelines.time_dynamics.siam_nb4_ladder import siam_nb4_action_id

    action_id = siam_nb4_action_id(
        arm="full_apm",
        segment="screen",
        activation_cut=SIAM_NB4_ACTIVATION_CUT_LADDER[0],
        route_id=ROUTE_Q_COMPILED,
    )
    prepared = _prepare_pinned(
        package_repo,
        package_id="siam_nb4_qc_full_apm_frontier_r00",
        action_id=action_id,
        arm="exchange_q_compiled",
        route_id=ROUTE_Q_COMPILED,
        cut=SIAM_NB4_ACTIVATION_CUT_LADDER[0],
        search_profile=SIAM_NB4_PILOT_EXCHANGE_SEARCH,
    )
    action = json.loads(prepared.action.read_text(encoding="utf-8"))
    manifest = json.loads(prepared.manifest.read_text(encoding="utf-8"))

    assert action["schema"] == SIAM_NB4_CHTC_ACTION_SCHEMA_V5
    assert action["route_id"] == ROUTE_Q_COMPILED
    assert action["search_profile_id"] == EXCHANGE_PROFILE_ID
    assert manifest["search_profile_id"] == EXCHANGE_PROFILE_ID
    assert action["expected_execution_contract"]["support_patch_config"][
        "exchange_insertion_frontier"
    ] == 8


def test_packager_refuses_an_unregistered_search_profile(package_repo) -> None:
    stranger = SearchProfile(
        profile_id="siam_nb4_unregistered_v1",
        pool_size=64,
        joint_patch_evaluations=50_000,
        certification_attempts_per_level=12,
        certification_attempts_per_deletion_branch=2,
        rounds_per_checkpoint=12,
        exchange_insertion_frontier=4,
    )
    with pytest.raises(ValueError, match="registered search profile"):
        _prepare(
            package_repo,
            package_id="siam_screen_unregistered_budget",
            action_id="screen_unregistered_budget",
            arm="exchange",
            search_profile=stranger,
        )


def test_completion_receipt_validator_reports_the_declared_profile(
    package_repo,
) -> None:
    root, seed, _image = package_repo
    trajectory = root / "raw_outputs" / "frontier_validator" / "run.json"
    _write_resume(trajectory, seed=seed, arm="exchange")
    receipt_path = trajectory.with_name("action_receipt.json")
    _write_completion_receipt(
        receipt_path,
        trajectory=trajectory,
        seed=seed,
        arm="exchange",
        search_profile_id=EXCHANGE_PROFILE_ID,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    verified = validate_siam_nb4_completion_receipt(receipt)
    assert verified["search_profile_id"] == EXCHANGE_PROFILE_ID

    default_trajectory = root / "raw_outputs" / "default_validator" / "run.json"
    _write_resume(default_trajectory, seed=seed, arm="exchange")
    default_receipt_path = default_trajectory.with_name("action_receipt.json")
    _write_completion_receipt(
        default_receipt_path,
        trajectory=default_trajectory,
        seed=seed,
        arm="exchange",
    )
    default_receipt = json.loads(
        default_receipt_path.read_text(encoding="utf-8")
    )
    assert "search_profile_id" not in default_receipt
    assert validate_siam_nb4_completion_receipt(default_receipt)[
        "search_profile_id"
    ] == SIAM_NB4_PILOT_SEARCH.profile_id


def test_completion_receipt_validator_refuses_a_half_declared_profile(
    package_repo,
) -> None:
    root, seed, _image = package_repo
    trajectory = root / "raw_outputs" / "half_declared" / "run.json"
    _write_resume(trajectory, seed=seed, arm="exchange")
    receipt_path = trajectory.with_name("action_receipt.json")
    _write_completion_receipt(
        receipt_path,
        trajectory=trajectory,
        seed=seed,
        arm="exchange",
        search_profile_id=EXCHANGE_PROFILE_ID,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop("search_profile_id")

    with pytest.raises(ValueError, match="search-profile identity"):
        validate_siam_nb4_completion_receipt(receipt)


def test_declared_profile_gate_refuses_unregistered_and_redundant_ids() -> None:
    """Only a registered non-default budget may be written into a document."""

    assert (
        siam_chtc_package_module._declared_search_profile_id(None) is None
    )
    assert siam_chtc_package_module._declared_search_profile_id(
        EXCHANGE_PROFILE_ID
    ) == EXCHANGE_PROFILE_ID
    with pytest.raises(ValueError, match="not registered"):
        siam_chtc_package_module._declared_search_profile_id(
            "siam_nb4_not_registered_v1"
        )
    with pytest.raises(ValueError, match="stay implicit"):
        siam_chtc_package_module._declared_search_profile_id(
            SIAM_NB4_PILOT_SEARCH.profile_id
        )
    assert siam_chtc_package_module._document_search_profile(
        {}
    ) is SIAM_NB4_PILOT_SEARCH
    assert siam_chtc_package_module._document_search_profile(
        {"search_profile_id": EXCHANGE_PROFILE_ID}
    ) is SIAM_NB4_PILOT_EXCHANGE_SEARCH


def test_run_command_refuses_an_unregistered_search_profile_id() -> None:
    with pytest.raises(ValueError, match="registered search profiles"):
        build_siam_nb4_screen_run(
            seed_path="input/seed.json",
            arm="exchange",
            output_json="raw_outputs/x/run.json",
            code_revision=REVISION,
            search_profile_id="siam_nb4_not_registered_v1",
        )


def test_continuation_refuses_a_cross_profile_screen_receipt(
    package_repo,
) -> None:
    root, seed, _image = package_repo
    resume = root / "inputs" / "cross_profile_screen_run.json"
    _write_resume(resume, seed=seed, arm="exchange")
    source_receipt = resume.with_name("action_receipt.json")
    _write_completion_receipt(
        source_receipt,
        trajectory=resume,
        seed=seed,
        arm="exchange",
        activation_cut=1.0e-4,
    )

    with pytest.raises(ValueError, match="different search profiles"):
        _prepare(
            package_repo,
            package_id="siam_apm_cross_profile_continuation",
            action_id="continue_cross_profile_rung2",
            arm="full_apm",
            segment="continuation",
            cut=1.0e-4,
            resume=resume,
            resume_receipt=source_receipt,
            search_profile=SIAM_NB4_PILOT_EXCHANGE_SEARCH,
        )
