"""Focused source-composition contracts for prospective full APM/exchange."""

from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest

import pipelines.time_dynamics.prospective_apm_exchange_chtc_package as apm_package
import pipelines.time_dynamics.prospective_apm_exchange_worker as apm_worker
from pipelines.time_dynamics.parallel_benchmark_matrix import (
    ParallelBenchmarkMatrix,
)
from pipelines.time_dynamics.prospective_apm_exchange_worker import (
    COMPLETED_UNSCORED,
    EXECUTOR_OVERLAY_PATHS,
    build_work_order,
    load_work_order,
    materialize_apm_exchange,
    validate_raw_run,
    write_work_order,
)
from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    LANE_HUBBARD,
    SourceEpochIdentity,
    build_identity_registry,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    SEED_ARTIFACT_V1,
    SEED_QUALIFICATION_RECEIPT_V1,
    SOURCE_EPOCH_RECEIPT_V1,
    SOURCE_TREE_MANIFEST_V1,
    SeedQualificationReceipt,
    SourceEpochReceipt,
    VerifiedFile,
    executor_bindings,
    resolve_scientific_action,
)
from pipelines.time_dynamics.prospective_scientific_planner import (
    ARM_APM,
    ARM_AVQDS,
    HH_HUBBARD_ACTIVATION_CUT_LADDER,
    HORIZON_FULL,
    HORIZON_PREFIX,
    KIND_RUN,
    ProspectiveNextAction,
    ProspectiveScientificBindings,
    policy_sha256,
    prospective_run_action_id,
)
from pipelines.time_dynamics.prospective_seed_chtc_package import (
    write_runtime_image_identity,
)
from pipelines.time_dynamics.run_lock import (
    RUN_LOCK_SCHEMA_V4,
    comparison_fingerprint,
    physics_fingerprint,
)
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
    RUNNER_SCHEMA_V1,
)
from pipelines.time_dynamics.staged_campaign import AccuracyPolicy


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _seal(payload: dict[str, object], key: str) -> dict[str, object]:
    result = dict(payload)
    result[key] = _canonical_sha(result)
    return result


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments), cwd=root, text=True
    ).strip()


def _policy() -> AccuracyPolicy:
    return AccuracyPolicy.response_normalized_matched_pair(
        response_scales={
            "weakslow": {
                horizon: {
                    "energy_rms_excursion": 0.2,
                    "doublon_rms_excursion": 0.1,
                    "max_site_occupation_rms_excursion": 0.1,
                }
                for horizon in (HORIZON_PREFIX, HORIZON_FULL)
            }
        },
        response_relative_target=0.05,
        response_relative_guard=0.15,
        matched_accuracy_factor=2.0,
        matched_component_deadband=0.10,
        infidelity_ceiling=1.0e-4,
        absolute_energy_error_ceiling=1.0e-2,
        absolute_doublon_error_ceiling=5.0e-3,
        absolute_site_occupation_error_ceiling=5.0e-3,
    )


def _receipts_and_request(root: Path):
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test Author")
    base_files = {
        "pipelines/scaffold/runtime_loader.py",
        "pipelines/time_dynamics/paper_ii_runs.py",
        "pipelines/time_dynamics/prospective_execution_bindings.py",
        "pipelines/time_dynamics/runners/ap_append_from_adapt_artifact.py",
        "src/base_runtime.py",
        "requirements.txt",
        *(
            path
            for path in EXECUTOR_OVERLAY_PATHS
            if "prospective_apm_exchange" not in path
            and not path.endswith("/patch_enumeration.py")
        ),
    }
    for relative in sorted(base_files):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            f"# pristine qualified-seed source: {relative}\n",
            encoding="utf-8",
        )
    _git(root, "add", "pipelines", "src", "requirements.txt")
    _git(root, "commit", "-qm", "pristine qualified-seed epoch")
    base_revision = _git(root, "rev-parse", "HEAD")
    base_archive = subprocess.run(
        ("git", "archive", "--format=tar", base_revision),
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout

    project_root = Path(__file__).resolve().parents[1]
    for relative in EXECUTOR_OVERLAY_PATHS:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project_root / relative, target)
    _git(root, "add", *EXECUTOR_OVERLAY_PATHS)
    _git(root, "commit", "-qm", "committed APM executor overlay")

    registry = build_identity_registry()
    reference = next(
        row
        for row in registry.references
        if row.cell_id == "hubbard__l4__u4__weakslow"
    )
    seed = next(row for row in registry.seeds if row.seed_id == reference.seed_id)
    epoch = SourceEpochIdentity(
        lane_id=LANE_HUBBARD,
        source_epoch_id="paper-ii-hubbard-apm-test-source",
        source_revision=base_revision,
        source_archive_sha256=hashlib.sha256(base_archive).hexdigest(),
    )
    tree_sha = "d" * 64
    tree_path = root / "source/tree_manifest.json"
    _write(
        tree_path,
        _seal(
            {
                "schema": SOURCE_TREE_MANIFEST_V1,
                "lane_id": epoch.lane_id,
                "source_epoch_identity_sha256": epoch.identity_sha256,
                "source_revision": epoch.source_revision,
                "source_archive_sha256": epoch.source_archive_sha256,
                "source_tree_sha256": tree_sha,
                "provenance_kind": "git_archive",
            },
            "source_tree_manifest_sha256",
        ),
    )
    source_path = root / "source/source_epoch_receipt.json"
    _write(
        source_path,
        _seal(
            {
                "schema": SOURCE_EPOCH_RECEIPT_V1,
                "source_epoch": epoch.to_json_dict(),
                "source_tree_manifest": {
                    "path": tree_path.relative_to(root).as_posix(),
                    "sha256": _file_sha(tree_path),
                },
                "source_tree_sha256": tree_sha,
            },
            "source_epoch_receipt_sha256",
        ),
    )
    source_receipt = SourceEpochReceipt.load(
        source_path, repository_root=root, expected_epoch=epoch
    )

    exact_energy = -1.0
    variational_energy = exact_energy + 5.0e-7
    artifact_path = root / seed.artifact_path
    _write(
        artifact_path,
        _seal(
            {
                "schema": SEED_ARTIFACT_V1,
                "seed_id": seed.seed_id,
                "seed_identity_sha256": seed.identity_sha256,
                "lane_id": seed.lane_id,
                "model": seed.model,
                "physical": dict(seed.physical),
                "seed_policy": seed.policy,
                "initializer": seed.initializer,
                "restarts": seed.restarts,
                "optimizer_seed": seed.optimizer_seed,
                "optimizer_execution_id": "test-hubbard-vqe4",
                "source_epoch_identity_sha256": epoch.identity_sha256,
                "variational_energy": variational_energy,
                "exact_energy": exact_energy,
                "exact_state_infidelity": 1.0e-8,
                "adapt_vqe": {
                    "parameterization_mode": "logical_shared",
                    "parameterization_execution_mode": "logical_shared",
                },
                "qiskit_compilation_used": False,
            },
            "seed_artifact_sha256",
        ),
    )
    seed_receipt_path = artifact_path.with_name(
        "seed_qualification_receipt.json"
    )
    _write(
        seed_receipt_path,
        _seal(
            {
                "schema": SEED_QUALIFICATION_RECEIPT_V1,
                "seed_id": seed.seed_id,
                "seed_identity_sha256": seed.identity_sha256,
                "artifact": {
                    "path": artifact_path.relative_to(root).as_posix(),
                    "sha256": _file_sha(artifact_path),
                },
                "energy_error": abs(variational_energy - exact_energy),
                "exact_state_infidelity": 1.0e-8,
                "initializer": seed.initializer,
                "restarts": seed.restarts,
                "optimizer_seed": seed.optimizer_seed,
                "optimizer_execution_id": "test-hubbard-vqe4",
                "variational_energy": variational_energy,
                "exact_energy": exact_energy,
                "source_epoch_identity_sha256": epoch.identity_sha256,
            },
            "seed_receipt_sha256",
        ),
    )
    seed_receipt = SeedQualificationReceipt.load(
        seed_receipt_path,
        repository_root=root,
        expected_seed=seed,
        expected_source_epoch=epoch,
    )
    matrix = ParallelBenchmarkMatrix.read()
    policy = _policy()
    bindings = ProspectiveScientificBindings(
        lane_id=reference.lane_id,
        cell_id=reference.cell_id,
        policy_regime="hubbard_l4_u4",
        drive=str(reference.drive["drive_id"]),
        controller="state_motion_5e-5",
        cell_identity_sha256=reference.identity_sha256,
        seed_id=seed.seed_id,
        seed_identity_sha256=seed.identity_sha256,
        seed_artifact_sha256=seed_receipt.artifact.sha256,
        seed_receipt_sha256=seed_receipt.receipt_sha256,
        reference_id=reference.reference_id,
        reference_identity_sha256=reference.identity_sha256,
        reference_manifest_sha256="4" * 64,
        reference_data_sha256="5" * 64,
        reference_receipt_sha256="6" * 64,
        source_epoch_identity_sha256=epoch.identity_sha256,
        source_revision=epoch.source_revision,
        source_tree_sha256=tree_sha,
        matrix_id=matrix.matrix_id,
        matrix_raw_sha256=matrix.raw_sha256,
        matrix_payload_sha256=matrix.payload_sha256,
        matrix_expanded_semantic_sha256=matrix.expanded_semantic_sha256,
        matrix_accuracy_policy_id="response_normalized_matched_pair_v2",
        matrix_accuracy_policy_sha256=matrix.accuracy_policy_spec_sha256,
        accuracy_policy_mode=policy.mode,
        accuracy_policy=policy.to_json_dict(),
        accuracy_policy_sha256=policy_sha256(policy),
    )
    cut = HH_HUBBARD_ACTIVATION_CUT_LADDER[0]
    action_id = prospective_run_action_id(
        bindings,
        arm=ARM_APM,
        horizon=HORIZON_PREFIX,
        activation_cut=cut,
    )
    scout_id = prospective_run_action_id(
        bindings,
        arm=ARM_AVQDS,
        horizon=HORIZON_PREFIX,
        activation_cut=cut,
    )
    action = ProspectiveNextAction(
        kind=KIND_RUN,
        action_id=action_id,
        execution_id=f"{action_id}::attempt0::mem8g",
        arm=ARM_APM,
        horizon=HORIZON_PREFIX,
        activation_cut=cut,
        ladder_index=0,
        attempt=0,
        planner_sequence=1,
        requested_memory_gib=8,
        reason="test canonical APM prefix",
        inputs={
            **bindings.to_json_dict(),
            "arm": ARM_APM,
            "horizon": HORIZON_PREFIX,
            "activation_cut": cut,
            "ladder_index": 0,
            "record_statevector": True,
            "exact_reference_blind": True,
            "resource_only_retry": False,
            "scientific_action_id": action_id,
            "seeded_from_scout_action_id": scout_id,
            "resume_from_action_id": None,
            "resume_run_receipt_sha256": None,
            "resume_run_sha256": None,
        },
        exact_reference_blind=True,
    )
    return source_receipt, seed_receipt, resolve_scientific_action(bindings, action)


def _runtime_contract(order) -> dict[str, object]:
    args = apm_worker._build_apm_parser().parse_args(list(order.runner_argv))
    support = apm_worker._support_patch_config_from_args(args)
    solve = apm_worker._solve_repair_config_from_args(args)
    candidate_pool = {
        "schema": "paper_ii_effective_candidate_pool_v1",
        "ordered_atom_contract_sha256": "c" * 64,
        "candidate_pool_raw": 8,
        "candidate_pool_deduplicated": 8,
        "candidate_pool_order": "frozen_pool_order",
        "append_occurrence_policy": "layer_reuse",
    }
    return {
        "parameterization_mode": "logical_shared",
        "family_key": "hubbard",
        "n_ph_max": None,
        "drive_profile": {
            "A": 0.3,
            "omega": 1.0,
            "tbar": 1.0,
            "phi": 0.0,
            "pattern": "staggered",
            "custom_weights": None,
            "include_identity": False,
            "time_sampling": "midpoint",
            "t0": 0.0,
            "family_key": "hubbard",
            "operator_label": "spinful_onsite_density",
            "spatial_weights": [1.0, -1.0, 1.0, -1.0],
            "reference_method_name": "exponential_midpoint_magnus2_order2",
        },
        "inverse_policy": {
            "pinv_rcond": float(args.pinv_rcond),
            "ridge_lambda": float(args.ridge_lambda),
            "solve_damping": float(args.solve_damping),
        },
        "solve_repair": solve.to_json_dict(),
        "time_step_controller": {
            "parameter_displacement_max": support.avqds_delta_theta_max,
            "state_motion_l2_step_max": solve.state_motion_l2_step_max,
            "composed_parameter_step_max": solve.parameter_step_max,
            "local_subdivision_enabled": bool(solve.local_subdivision_enabled),
            "max_local_subdivisions": int(solve.max_local_subdivisions),
        },
        "candidate_pool": candidate_pool,
        "support_patch_config": support.to_json_dict(),
        "guards": {
            key: getattr(support, key, None)
            for key in (
                "max_joint_patch_evaluations",
                "max_certification_attempts_per_level",
                "max_certification_attempts_per_deletion_branch",
                "max_insertion_batch_size",
                "residual_ratio_threshold",
            )
        },
    }


def _fake_raw(order, seed_receipt, runtime: dict[str, object]):
    times = apm_worker._expected_times(
        order.reference, horizon=HORIZON_PREFIX
    )
    physics = {
        "seed_artifact_json": seed_receipt.artifact.path,
        "seed_sha256": seed_receipt.artifact.sha256,
        "family_key": runtime["family_key"],
        "n_ph_max": runtime["n_ph_max"],
        "time_grid": {
            "t_initial": times[0],
            "t_final": times[-1],
            "point_count": len(times),
            "ordered_values_sha256": _canonical_sha(times),
        },
        "drive_profile": deepcopy(runtime["drive_profile"]),
        "exact_reference_json": None,
        "exact_reference_sha256": None,
    }
    numerics = {
        "integrator_method": "euler",
        "inverse_policy": deepcopy(runtime["inverse_policy"]),
        "solve_repair": deepcopy(runtime["solve_repair"]),
    }
    shared = {
        "physics_fingerprint": physics_fingerprint(physics),
        "numerics": numerics,
        "time_step_controller": deepcopy(runtime["time_step_controller"]),
        "candidate_pool": deepcopy(runtime["candidate_pool"]),
    }
    lock = {
        "schema": RUN_LOCK_SCHEMA_V4,
        "physics": physics,
        "physics_fingerprint": physics_fingerprint(physics),
        "numerics": numerics,
        "shared_protocol": shared,
        "comparison_fingerprint": comparison_fingerprint(shared),
        "policy": {
            "structural_policy": "exchange",
            "guards": deepcopy(runtime["guards"]),
        },
        "code_revision": order.executor_overlay.overlay_revision,
    }
    dimension = 1 << 8
    real = [1.0] + [0.0] * (dimension - 1)
    imag = [0.0] * dimension
    return {
        "schema": RUNNER_SCHEMA_V1,
        "run_lock": lock,
        "initial_state": {"parameterization_mode": "logical_shared"},
        "final_state": {"parameterization_mode": "logical_shared"},
        "plot_rows": [
            {"time": value, "psi_real": real, "psi_imag": imag}
            for value in times
        ],
        "summary": {
            "structural_selection_policy": (
                apm_worker.GENERALIZED_EXCHANGE_SELECTION_POLICY_V2
            ),
            "support_patch_config": deepcopy(runtime["support_patch_config"]),
            "parameterization_mode": "logical_shared",
            "candidate_pool_complete": True,
            "candidate_pool_term_count": 8,
        },
        "decision_data_flow": {
            "uses_reference_for_decision": False,
            "uses_exact_reference_for_decision": False,
            "uses_future_exact_forecast_for_decision": False,
            "uses_statevector_as_ideal_observable_estimator": True,
        },
    }


@pytest.mark.parametrize("worker_name", (
    "prospective_apm_exchange_worker",
    "prospective_avqds_scout_worker",
    "prospective_avqds_trajectory_worker",
))
def test_executor_overlay_contains_current_pool_and_enumeration_dependencies(
    worker_name: str,
) -> None:
    import importlib

    worker = importlib.import_module(f"pipelines.time_dynamics.{worker_name}")
    paths = worker.EXECUTOR_OVERLAY_PATHS
    assert len(paths) == len(set(paths))
    required_modules = (
        "paper_ii_runs", "runners.ap_append_from_adapt_artifact",
        "ap_mclachlan.adaptive_trajectory", "ap_mclachlan.exchange_config",
        "ap_mclachlan.support_atoms", "ap_mclachlan.exchange_integration",
        "ap_mclachlan.exchange_structural", "ap_mclachlan.exchange_selector",
        "ap_mclachlan.deletion_family", "ap_mclachlan.patch_enumeration",
        "ap_mclachlan.patch_evaluation",
        "ap_mclachlan.insertion_words", "generalized_exchange",
    )
    for module_name in required_modules:
        assert f"pipelines/time_dynamics/{module_name.replace('.', '/')}.py" in paths


def test_hubbard_work_order_binds_fixed_policy_and_loader_overlay(
    tmp_path: Path,
) -> None:
    source, seed, request = _receipts_and_request(tmp_path)
    order = build_work_order(
        request,
        source_epoch_receipt=source,
        seed_receipt=seed,
        repository_root=tmp_path,
    )
    argv = list(order.runner_argv)
    assert argv[argv.index("--replay-candidate-pool-mode") + 1] == (
        "replay_family_pool"
    )
    assert argv[argv.index("--append-cost-normalization-mode") + 1] == (
        "family_robust_v1"
    )
    assert "--record-statevector" in argv
    assert "--require-complete-candidate-pool" in argv
    assert not any("qiskit" in token.lower() for token in argv)
    assert order.policy["fs_radius_experiment_status"] == "superseded_deferred"
    assert order.policy["complete_ranked_family_preserved_after_rejection"] is True
    loader = "pipelines/scaffold/runtime_loader.py"
    assert loader in order.executor_overlay.files
    assert order.executor_overlay.files[loader]["base_sha256"] != (
        order.executor_overlay.files[loader]["overlay_sha256"]
    )
    enumeration = "pipelines/time_dynamics/ap_mclachlan/patch_enumeration.py"
    assert order.executor_overlay.files[enumeration]["base_present"] is False
    assert order.executor_overlay.files[enumeration]["base_sha256"] is None
    assert "--max-structural-pool-size" not in argv
    assert order.executor_overlay.base_source_revision == (
        source.source_epoch.source_revision
    )
    assert order.executor_overlay.overlay_revision != (
        source.source_epoch.source_revision
    )

    path = tmp_path / "orders/apm.json"
    write_work_order(path, order)
    loaded, loaded_source, loaded_seed = load_work_order(
        path, repository_root=tmp_path
    )
    assert loaded == order
    assert loaded_source == source
    assert loaded_seed == seed


def test_apm_binding_resolves_to_inert_operational_materialization_adapter(
    tmp_path: Path,
) -> None:
    source, seed, request = _receipts_and_request(tmp_path)
    binding = next(
        row
        for row in executor_bindings()
        if row.binding_id == apm_worker.EXEC_PROSPECTIVE_APM
    )
    assert binding.handler == (
        "pipelines.time_dynamics.prospective_apm_exchange_worker:"
        "materialization_request"
    )
    adapter = binding.resolve()(request)
    assert adapter["status"] == "operational_materialization_ready"
    assert adapter["execution_request_sha256"] == request.digest
    assert adapter["seed_identity_sha256"] == seed.seed_identity_sha256
    assert adapter["source_epoch_identity_sha256"] == (
        source.source_epoch.identity_sha256
    )
    assert adapter["runner_module"] == (
        "pipelines.time_dynamics.runners.run_prospective_apm_exchange"
    )
    assert adapter["required_receipts"] == [
        "source_epoch_receipt",
        "seed_qualification_receipt",
    ]
    assert adapter["fixed_exchange_policy"][
        "fs_radius_experiment_status"
    ] == "superseded_deferred"
    assert adapter["scheduler_contacted"] is False
    assert adapter["scheduler_submission_authorized"] is False
    assert adapter["production_submission_enabled"] is False


def test_raw_validator_is_completed_unscored_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, seed, request = _receipts_and_request(tmp_path)
    order = build_work_order(
        request,
        source_epoch_receipt=source,
        seed_receipt=seed,
        repository_root=tmp_path,
    )
    runtime = _runtime_contract(order)
    monkeypatch.setattr(
        apm_worker,
        "_expected_runtime_contract",
        lambda *_args, **_kwargs: runtime,
    )

    def execute(actual_order, root: Path) -> None:
        _write(root / actual_order.run_part_path, _fake_raw(actual_order, seed, runtime))

    receipt = materialize_apm_exchange(
        order, repository_root=tmp_path, execute=execute
    )
    assert receipt.status == COMPLETED_UNSCORED
    final_payload = json.loads(
        (tmp_path / receipt.run.path).read_text(encoding="utf-8")
    )
    composition = final_payload["prospective_source_composition"]
    assert composition["base_seed_source_epoch"]["source_epoch_id"] == (
        source.source_epoch.source_epoch_id
    )
    assert composition["executed_executor_overlay"]["overlay_revision"] == (
        order.executor_overlay.overlay_revision
    )
    assert composition["seed_epoch_rerun"] is False
    assert composition["seed_epoch_relabelled"] is False

    mutations = []
    bad_dimension = _fake_raw(order, seed, runtime)
    bad_dimension["plot_rows"][0]["psi_real"] = [1.0]
    bad_dimension["plot_rows"][0]["psi_imag"] = [0.0]
    mutations.append(bad_dimension)
    bad_drive = _fake_raw(order, seed, runtime)
    bad_drive["run_lock"]["physics"]["drive_profile"]["omega"] = 9.0
    mutations.append(bad_drive)
    bad_pool = _fake_raw(order, seed, runtime)
    bad_pool["run_lock"]["shared_protocol"]["candidate_pool"][
        "ordered_atom_contract_sha256"
    ] = "f" * 64
    mutations.append(bad_pool)
    bad_parameterization = _fake_raw(order, seed, runtime)
    bad_parameterization["summary"]["parameterization_mode"] = (
        "per_pauli_term"
    )
    mutations.append(bad_parameterization)
    bad_cut = _fake_raw(order, seed, runtime)
    bad_cut["summary"]["support_patch_config"]["insertion_l2_cut"] = 0.5
    mutations.append(bad_cut)

    for index, payload in enumerate(mutations):
        target = tmp_path / f"tampered/raw_{index}.json"
        _write(target, payload)
        apm_worker.annotate_raw_run(
            target,
            order=order,
            source_epoch=source.source_epoch,
        )
        with pytest.raises(ValueError):
            validate_raw_run(
                target,
                order=order,
                seed_receipt=seed,
                source_epoch=source.source_epoch,
                repository_root=tmp_path,
            )


def test_full_horizon_resumes_only_the_same_cut_apm_prefix(
    tmp_path: Path,
) -> None:
    source, seed, prefix_request = _receipts_and_request(tmp_path)
    bindings = ProspectiveScientificBindings.from_json_dict(
        prefix_request.inputs
    )
    cut = float(prefix_request.inputs["activation_cut"])
    prefix_id = prospective_run_action_id(
        bindings,
        arm=ARM_APM,
        horizon=HORIZON_PREFIX,
        activation_cut=cut,
    )
    resume_run_path = tmp_path / "resume/apm_prefix_run.json"
    resume_receipt_path = tmp_path / "resume/apm_prefix_receipt.json"
    _write(resume_run_path, {"resume_state": {"schema": "test"}})
    _write(resume_receipt_path, {"status": COMPLETED_UNSCORED})
    resume_run = VerifiedFile(
        resume_run_path.relative_to(tmp_path).as_posix(),
        _file_sha(resume_run_path),
    )
    resume_receipt = VerifiedFile(
        resume_receipt_path.relative_to(tmp_path).as_posix(),
        _file_sha(resume_receipt_path),
    )
    action_id = prospective_run_action_id(
        bindings,
        arm=ARM_APM,
        horizon=HORIZON_FULL,
        activation_cut=cut,
    )
    action = ProspectiveNextAction(
        kind=KIND_RUN,
        action_id=action_id,
        execution_id=f"{action_id}::attempt0::mem8g",
        arm=ARM_APM,
        horizon=HORIZON_FULL,
        activation_cut=cut,
        ladder_index=0,
        attempt=0,
        planner_sequence=2,
        requested_memory_gib=8,
        reason="test same-cut full-horizon recheck",
        inputs={
            **bindings.to_json_dict(),
            "arm": ARM_APM,
            "horizon": HORIZON_FULL,
            "activation_cut": cut,
            "ladder_index": 0,
            "record_statevector": True,
            "exact_reference_blind": True,
            "resource_only_retry": False,
            "scientific_action_id": action_id,
            "seeded_from_scout_action_id": None,
            "resume_from_action_id": prefix_id,
            "resume_run_receipt_sha256": resume_receipt.sha256,
            "resume_run_sha256": resume_run.sha256,
        },
        exact_reference_blind=True,
    )
    request = resolve_scientific_action(bindings, action)
    order = build_work_order(
        request,
        source_epoch_receipt=source,
        seed_receipt=seed,
        repository_root=tmp_path,
        resume_run=resume_run,
        resume_receipt=resume_receipt,
    )
    argv = list(order.runner_argv)
    assert argv[argv.index("--resume-from-run-json") + 1] == resume_run.path
    assert argv[argv.index("--t-initial") + 1] == "5.0"
    num_time_values = [
        argv[index + 1]
        for index, token in enumerate(argv)
        if token == "--num-times"
    ]
    assert num_time_values[-1] == "376"
    assert order.resume_run == resume_run
    assert order.resume_receipt == resume_receipt
    with pytest.raises(ValueError, match="requires digest-bound prefix"):
        build_work_order(
            request,
            source_epoch_receipt=source,
            seed_receipt=seed,
            repository_root=tmp_path,
        )


def test_hubbard_package_is_inert_and_binds_source_epoch_and_loader(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, seed, request = _receipts_and_request(tmp_path)
    order = build_work_order(
        request,
        source_epoch_receipt=source,
        seed_receipt=seed,
        repository_root=tmp_path,
    )
    order_path = tmp_path / "orders/apm.json"
    write_work_order(order_path, order)
    source_archive = tmp_path / "runtime_source.tar.gz"
    source_tar = subprocess.run(
        (
            "git",
            "archive",
            "--format=tar",
            "--prefix=source/",
            source.source_epoch.source_revision,
            "--",
            "pipelines",
            "src",
            "requirements.txt",
        ),
        cwd=tmp_path,
        check=True,
        capture_output=True,
    ).stdout
    compressed = io.BytesIO()
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=compressed, compresslevel=9, mtime=0
    ) as handle:
        handle.write(source_tar)
    source_archive.write_bytes(compressed.getvalue())
    image_path = tmp_path / "runtime/test.sif"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"test-apptainer-image")
    identity_path = tmp_path / "runtime/image_identity.json"
    write_runtime_image_identity(
        identity_path, image_path=image_path, repository_root=tmp_path
    )

    prepared = apm_package.materialize_apm_exchange_package(
        work_order_path=order_path,
        runtime_source_archive_path=source_archive,
        runtime_image_identity_path=identity_path,
        output_dir="packages/apm",
        repository_root=tmp_path,
        free_bytes=100 * 1024**3,
    )
    descriptor = (tmp_path / prepared.descriptor.path).read_text(
        encoding="utf-8"
    )
    assert descriptor.splitlines().count("queue 1") == 1
    unit_id = apm_package.apm_exchange_submission_unit_id(order)
    assert f'+AcquisitionUnitId = "{unit_id}"' in descriptor
    assert f'+AcquisitionLaneId = "{order.request.lane_id}"' in descriptor
    assert (
        f'+PaperIISourceEpochId = "{source.source_epoch.source_epoch_id}"'
        in descriptor
    )
    assert "+PaperIIExactReferenceBlind = True" in descriptor
    assert "+PaperIIQiskitCompilationRequested = False" in descriptor
    preflight = json.loads(
        (tmp_path / prepared.preflight.path).read_text(encoding="utf-8")
    )
    assert preflight["scheduler_contacted"] is False
    assert preflight["scheduler_submission_authorized"] is False
    assert preflight["production_submission_enabled"] is False
    assert preflight["acquisition_unit_id"] == unit_id
    assert preflight["job_batch_name"] in descriptor
    assert preflight["executor_overlay_identity_sha256"] == (
        order.executor_overlay.identity_sha256
    )
    manifest = json.loads(
        (tmp_path / prepared.package_manifest.path).read_text(encoding="utf-8")
    )
    assert manifest["execution_request"]["kind"] == KIND_RUN
    assert manifest["acquisition_unit_id"] == unit_id
    assert manifest["production_submission_enabled"] is False
    with tarfile.open(tmp_path / prepared.package.path, "r:gz") as archive:
        names = set(archive.getnames())
        worker = archive.extractfile("run_apm_exchange.sh")
        assert worker is not None
        worker_text = worker.read().decode("utf-8")
    loader_member = (
        "executor_overlay/pipelines/scaffold/runtime_loader.py"
    )
    assert loader_member in names
    assert all(f"executor_overlay/{path}" in names for path in EXECUTOR_OVERLAY_PATHS)
    assert "input/executor_overlay_identity.json" in names
    assert "BASE_SOURCE_VERIFIED_BEFORE_OVERLAY" in worker_text
    assert "pristine base digest mismatch" in worker_text
    assert "run_prospective_apm_exchange run" in worker_text
    assert "REMOTE_RAW_PRESERVED" in worker_text
    syntax = subprocess.run(
        ("bash", "-n"), input=worker_text, text=True, capture_output=True
    )
    assert syntax.returncode == 0, syntax.stderr

    assert apm_package.main(
        [
            "materialize",
            "--repo-root",
            str(tmp_path),
            "--output-dir",
            "packages/apm_cli",
            "--work-order",
            order_path.relative_to(tmp_path).as_posix(),
            "--source-archive",
            source_archive.relative_to(tmp_path).as_posix(),
            "--runtime-image-identity",
            identity_path.relative_to(tmp_path).as_posix(),
            "--request-cpus",
            "1",
            "--request-memory-gib",
            "16",
            "--request-disk-gib",
            "8",
            "--max-runtime-seconds",
            "43200",
        ]
    ) == 0
    cli = json.loads(capsys.readouterr().out)
    assert cli["status"] == "prepared_local_only"
    assert cli["package"] == "packages/apm_cli/apm_exchange_action.tar.gz"
    assert cli["scheduler_contacted"] is False
    assert cli["scheduler_submission_authorized"] is False
    assert cli["production_submission_enabled"] is False
