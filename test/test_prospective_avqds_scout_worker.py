"""Focused contracts for the prospective exact-blind AVQDS prefix worker."""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import gzip
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

import pipelines.time_dynamics.prospective_avqds_scout_chtc_package as scout_package
import pipelines.time_dynamics.prospective_avqds_runtime_source as runtime_source
import pipelines.time_dynamics.prospective_avqds_scout_worker as scout_worker_module
import pipelines.time_dynamics.prospective_avqds_trajectory_worker as trajectory_worker_module
import pipelines.time_dynamics.runners.run_prospective_avqds_scout as scout_runner_module
import pipelines.scaffold.runtime_loader as runtime_loader_module
import pipelines.time_dynamics.prospective_execution_bindings as execution_bindings_module
import pipelines.time_dynamics.fixed_vqe_conditioning as fixed_vqe_module
import pipelines.time_dynamics.prospective_hh_seed_worker as hh_seed_worker_module
from pipelines.time_dynamics.prospective_avqds_scout_worker import (
    EXECUTOR_OVERLAY_PATHS,
    SCOUT_STATUS,
    build_work_order,
    load_frozen_submitted_work_order,
    load_work_order,
    materialize_scout,
    write_work_order,
)
from pipelines.time_dynamics.prospective_avqds_runtime_source import (
    SCOUT_PROFILE,
    TRAJECTORY_PROFILE,
    materialize_avqds_runtime_source_archive,
)
from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    BootstrapEvidence,
    LANE_HH,
    LANE_HUBBARD,
    SourceEpochIdentity,
    build_identity_registry,
    plan_bootstrap_actions,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    SEED_ARTIFACT_V1,
    SEED_QUALIFICATION_RECEIPT_V1,
    SOURCE_EPOCH_RECEIPT_V1,
    SOURCE_TREE_MANIFEST_V1,
    ProspectiveExecutionError,
    SeedQualificationReceipt,
    SourceEpochReceipt,
    executor_bindings,
    resolve_bootstrap_action,
)
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
    RUNNER_SCHEMA_V1,
)
from pipelines.time_dynamics.prospective_seed_chtc_package import (
    write_runtime_image_identity,
)
from pipelines.time_dynamics.pure_hubbard_fixed_vqe import (
    PureHubbardFixedVQEConfig,
    build_pure_hubbard_fixed_vqe_problem,
)
from src.quantum.ansatz_parameterization import (
    expand_legacy_logical_theta,
    serialize_layout,
)
from src.quantum.vqe_latex_python_pairs import PauliPolynomial, PauliTerm


LOCKED_HUBBARD_POOL_COUNT = 113
LOCKED_HUBBARD_POOL_SHA256 = (
    "a01e952a7496faa48ce10bcba64a1c90b5cced229f7c3ae24c87dfd2d1fbf1bb"
)
LOCKED_EFFECTIVE_POOL_SHA256 = (
    "74ef9d7f50bbe6e163b572b4ea3a7b083886c906b2b2b865f3663bf57b654539"
)


def _state_payload(state: np.ndarray, *, kind: str) -> dict[str, object]:
    vector = np.asarray(state, dtype=complex).reshape(-1)
    nq = int(np.log2(vector.size))
    return {
        "source": "test",
        "handoff_state_kind": kind,
        "nq_total": nq,
        "amplitudes_qn_to_q0": {
            format(index, f"0{nq}b"): {
                "re": float(np.real(amplitude)),
                "im": float(np.imag(amplitude)),
            }
            for index, amplitude in enumerate(vector)
            if abs(complex(amplitude)) > 1.0e-14
        },
    }


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments), cwd=root, text=True
    ).strip()


def _receipts_and_request(root: Path):
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test Author")
    repository_source = Path(__file__).resolve().parents[1]
    for top_level in ("docs", "pipelines", "src"):
        for source_file in (repository_source / top_level).rglob("*.py"):
            relative = source_file.relative_to(repository_source)
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file, target)
    for pattern in (
        "pipelines/time_dynamics/campaigns/*.json",
        "src/quantum/chemistry/*.json",
    ):
        for source_file in repository_source.glob(pattern):
            relative = source_file.relative_to(repository_source)
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file, target)
    for relative in (
        *EXECUTOR_OVERLAY_PATHS,
        "pipelines/time_dynamics/prospective_exact_reference_worker.py",
        "pipelines/time_dynamics/runners/build_prospective_exact_reference.py",
        "requirements.txt",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"# pristine seed-epoch source: {relative}\n", encoding="utf-8")
    _git(root, "add", "docs", "pipelines", "src", "requirements.txt")
    _git(root, "commit", "-qm", "pristine qualified-seed epoch")
    base_revision = _git(root, "rev-parse", "HEAD")
    base_archive = subprocess.run(
        ("git", "archive", "--format=tar", base_revision),
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    overlay_sources = {
        "pipelines/scaffold/runtime_loader.py": Path(runtime_loader_module.__file__),
        "pipelines/time_dynamics/prospective_execution_bindings.py": Path(
            execution_bindings_module.__file__
        ),
        "pipelines/time_dynamics/prospective_avqds_scout_worker.py": Path(
            scout_worker_module.__file__
        ),
        "pipelines/time_dynamics/runners/run_prospective_avqds_scout.py": Path(
            scout_runner_module.__file__
        ),
    }
    for relative, source_file in overlay_sources.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, target)
    _git(root, "add", *overlay_sources)
    _git(root, "commit", "-qm", "committed AVQDS executor overlay")

    registry = build_identity_registry()
    seed = next(row for row in registry.seeds if row.lane_id == LANE_HUBBARD)
    epoch = SourceEpochIdentity(
        lane_id=LANE_HUBBARD,
        source_epoch_id="paper-ii-avqds-scout-test",
        source_revision=base_revision,
        source_archive_sha256=hashlib.sha256(base_archive).hexdigest(),
    )
    tree_path = root / "source/tree_manifest.json"
    tree_sha = "d" * 64
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

    config = PureHubbardFixedVQEConfig(
        num_sites=4,
        n_up=2,
        n_down=2,
        reps=2,
        restarts=1,
        maxiter=1,
    )
    problem = build_pure_hubbard_fixed_vqe_problem(config)
    theta_logical = np.zeros(problem.ansatz.num_parameters, dtype=float)
    theta_runtime = expand_legacy_logical_theta(
        theta_logical, problem.ansatz.layout
    )
    settings = config.to_json_dict()
    settings.update(
        {
            "adapt_pool": "uccsd_qeb_hva_blocks",
            "dv": 0.0,
            "sector_n_dn": int(config.n_down),
            "include_zero_point": False,
        }
    )
    exact_energy = float(problem.exact_energy)
    variational_energy = exact_energy + 5.0e-7
    artifact_path = root / seed.artifact_path
    artifact = _seal(
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
            "settings": settings,
            "ansatz_input_state": _state_payload(
                problem.reference_state, kind="reference_state"
            ),
            "initial_state": _state_payload(
                problem.reference_state, kind="prepared_state"
            ),
            "adapt_vqe": {
                "pool_type": "uccsd_qeb_hva_blocks",
                "structure_locked": True,
                "parameterization": serialize_layout(problem.ansatz.layout),
                "parameterization_mode": "logical_shared",
                "parameterization_execution_mode": "logical_shared",
                "operators": [
                    str(term.label) for term in problem.generator_terms
                ],
                "optimal_point": [float(value) for value in theta_runtime],
                "logical_optimal_point": [
                    float(value) for value in theta_logical
                ],
                "fixed_scaffold_metadata": {
                    "generator_filter": (
                        "fixed_spin_sector_logical_generators_v1"
                    ),
                    "term_order_basis": (
                        "repository_deduplicated_pool_order_v1"
                    ),
                    "generator_execution": (
                        "grouped_exact_logical_shared_v1"
                    ),
                    "generator_pool_sha256": LOCKED_HUBBARD_POOL_SHA256,
                    "operator_count": len(problem.generator_terms),
                },
            },
            "architecture_binding": {
                "policy": {
                    "generator_pool": "uccsd_qeb_hva_blocks",
                    "generator_filter": (
                        "fixed_spin_sector_logical_generators_v1"
                    ),
                    "generator_order": (
                        "repository_deduplicated_pool_order_v1"
                    ),
                    "generator_execution": (
                        "grouped_exact_logical_shared_v1"
                    ),
                    "expected_unique_generator_count": (
                        LOCKED_HUBBARD_POOL_COUNT
                    ),
                },
                "generator_pool_sha256": LOCKED_HUBBARD_POOL_SHA256,
                "unique_generator_count": LOCKED_HUBBARD_POOL_COUNT,
                "sector_dimension": 36,
            },
            "qiskit_compilation_used": False,
        },
        "seed_artifact_sha256",
    )
    _write(artifact_path, artifact)
    seed_receipt_path = artifact_path.with_name("seed_qualification_receipt.json")
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
    plan = plan_bootstrap_actions(
        registry,
        evidence=BootstrapEvidence(
            seed_artifact_sha256={seed.seed_id: seed_receipt.artifact.sha256},
            source_epochs={LANE_HUBBARD: epoch},
        ),
        available_executors=tuple(row.binding_id for row in executor_bindings()),
    )
    action = next(
        row
        for row in plan.ready_actions
        if row.kind == "screen_avqds" and row.lane_id == LANE_HUBBARD
    )
    return source_receipt, seed_receipt, resolve_bootstrap_action(action)


def _fake_raw(order, seed_receipt, source_epoch) -> dict[str, object]:
    grid = order.reference.grid
    times = [
        float(grid["start"]) + index * float(grid["step"])
        for index in range(int(grid["prefix_points"]))
    ]
    contract = order.execution_contract
    hilbert_dimension = int(contract["physical_hilbert_dimension"])
    zero_state_real = [0.0] * hilbert_dimension
    zero_state_real[0] = 1.0
    points = [
        {
            "time": value,
            "integration_to_next": (
                None
                if index + 1 == len(times)
                else {"local_subdivision_reason": None}
            ),
        }
        for index, value in enumerate(times)
    ]
    return {
        "schema": RUNNER_SCHEMA_V1,
        "run_lock": {
            "physics": {
                "seed_sha256": seed_receipt.artifact.sha256,
                "time_grid": {
                    "point_count": len(times),
                    "ordered_values_sha256": grid["ordered_prefix_sha256"],
                },
                "drive_profile": copy.deepcopy(contract["drive_profile"]),
                "exact_reference_json": None,
                "exact_reference_sha256": None,
            },
            "numerics": {
                "integrator_method": contract["integrator"],
                "inverse_policy": copy.deepcopy(contract["inverse_policy"]),
                "solve_repair": copy.deepcopy(
                    contract["solve_repair_config"]
                ),
            },
            "policy": {
                "structural_policy": contract["support_patch_config"][
                    "dynamics_policy"
                ],
                "guards": copy.deepcopy(contract["paper_declared_guards"]),
            },
            "shared_protocol": {
                "candidate_pool": copy.deepcopy(
                    contract["candidate_pool_contract"]
                ),
                "time_step_controller": copy.deepcopy(
                    contract["time_step_controller"]
                ),
            },
            "code_revision": order.executor_overlay.overlay_revision,
        },
        "initial_state": {
            **copy.deepcopy(contract["initial_state_contract"]),
            "candidate_pool_source": copy.deepcopy(
                contract["candidate_pool_source"]
            ),
        },
        "hamiltonian": copy.deepcopy(contract["hamiltonian"]),
        "drive_aligned_ansatz": copy.deepcopy(
            contract["drive_aligned_ansatz"]
        ),
        "trajectory": {
            "point_count": len(times),
            "controller_config": copy.deepcopy(contract["controller_config"]),
            "support_patch_config": copy.deepcopy(
                contract["support_patch_config"]
            ),
            "solve_repair_config": copy.deepcopy(
                contract["solve_repair_config"]
            ),
            "metadata": {
                "parameterization_mode": contract["parameterization_mode"],
                "replay_candidate_pool_mode": "replay_family_pool",
                "solve_repair_requested": contract["solve_repair_config"][
                    "enabled"
                ],
                "solve_repair_config": copy.deepcopy(
                    contract["solve_repair_config"]
                ),
                "drive_aligned_ansatz_requested": True,
            },
            "points": points,
        },
        "plot_rows": [
            {
                "time": value,
                "psi_real": list(zero_state_real),
                "psi_imag": [0.0] * hilbert_dimension,
            }
            for value in times
        ],
        "decision_data_flow": {
            "uses_reference_for_decision": False,
            "uses_exact_reference_for_decision": False,
            "uses_future_exact_forecast_for_decision": False,
        },
        "paper_ii_execution_attestation": (
            scout_worker_module._execution_attestation(order)
        ),
    }


def test_work_order_locks_complete_pool_and_exact_blind_runner(tmp_path: Path) -> None:
    source, seed, request = _receipts_and_request(tmp_path)
    order = build_work_order(
        request,
        source_epoch_receipt=source,
        seed_receipt=seed,
        repository_root=tmp_path,
    )
    argv = list(order.runner_argv)
    assert argv[argv.index("--loader-mode") + 1] == "replay_family"
    assert argv[argv.index("--replay-candidate-pool-mode") + 1] == "replay_family_pool"
    assert argv[argv.index("--parameterization-mode") + 1] == "logical_shared"
    assert argv[argv.index("--code-revision") + 1] == order.executor_overlay.overlay_revision
    assert order.executor_overlay.overlay_revision != source.source_epoch.source_revision
    assert "--require-complete-candidate-pool" in argv
    assert "--record-statevector" in argv
    assert "--reference-energy-json" not in argv
    assert not any("qiskit" in token.lower() for token in argv)
    assert order.to_json_dict()["scoring_required_before_qualification"] is True
    assert order.to_json_dict()["production_submission_enabled"] is False
    serialized_order = json.dumps(order.to_json_dict(), sort_keys=True)
    assert all(
        token not in serialized_order
        for token in (
            '"artifact_root"',
            '"external_manifest_path"',
            '"external_manifest_sha256"',
            '"external_data_path"',
            '"external_data_sha256"',
            '"materialized_key"',
            '"response_scales"',
        )
    )
    contract = order.execution_contract
    assert contract["activation_cut"] == 1.0e-2
    assert contract["activation_ladder_index"] == 0
    assert contract["reporting_grid"]["final"] == 5.0
    assert contract["reporting_grid"]["point_count"] == 126
    assert contract["fail_on_unsupported_steps"] is True
    assert contract["solve_repair_config"]["state_motion_l2_step_max"] == 5.0e-5
    assert contract["solve_repair_config"]["max_local_subdivisions"] == 11
    assert contract["paper_declared_guards"] == {
        "max_joint_patch_evaluations": 50_000,
        "max_certification_attempts_per_level": 12,
        "max_certification_attempts_per_deletion_branch": 2,
        "max_structural_pool_size": 128,
        "max_insertion_batch_size": 1,
        "residual_ratio_threshold": 0.02,
    }
    assert contract["parameterization_mode"] == "logical_shared"
    assert contract["physical_hilbert_dimension"] == 256
    assert contract["drive_profile"] == {
        "A": 2.4,
        "custom_weights": None,
        "family_key": "hubbard",
        "include_identity": False,
        "omega": 3.0,
        "operator_label": "spinful_onsite_density",
        "pattern": "staggered",
        "phi": 0.0,
        "reference_method_name": "exponential_midpoint_magnus2_order2",
        "spatial_weights": [1.0, -1.0, 1.0, -1.0],
        "t0": 0.0,
        "tbar": 1.0,
        "time_sampling": "midpoint",
    }
    assert contract["candidate_pool_contract"] == {
        "schema": "paper_ii_effective_candidate_pool_v1",
        "ordered_atom_contract_sha256": LOCKED_EFFECTIVE_POOL_SHA256,
        "candidate_pool_raw": LOCKED_HUBBARD_POOL_COUNT,
        "candidate_pool_deduplicated": LOCKED_HUBBARD_POOL_COUNT,
        "candidate_pool_order": "frozen_pool_order",
        "append_occurrence_policy": "layer_reuse",
    }
    declared_pool = contract["candidate_pool_source"]["filter_payload"][
        "explicit_complete_pool_override"
    ]
    assert declared_pool["declared_generator_pool_sha256"] == (
        LOCKED_HUBBARD_POOL_SHA256
    )
    assert declared_pool["declared_generator_pool_count"] == (
        LOCKED_HUBBARD_POOL_COUNT
    )
    assert declared_pool["declared_generator_pool_order"] == (
        "repository_deduplicated_pool_order_v1"
    )
    assert declared_pool["prospective_seed_declarations_verified"] is True

    path = tmp_path / "orders/scout.json"
    write_work_order(path, order)
    loaded, loaded_source, loaded_seed = load_work_order(
        path, repository_root=tmp_path
    )
    assert loaded.to_json_dict() == order.to_json_dict()
    assert loaded_source.receipt_sha256 == source.receipt_sha256
    assert loaded_seed.receipt_sha256 == seed.receipt_sha256


def test_scout_seed_loader_modes_are_lane_typed_and_fail_closed() -> None:
    assert scout_worker_module._SEED_LOADER_MODE_BY_LANE == {
        LANE_HH: "fixed_scaffold",
        LANE_HUBBARD: "replay_family",
    }
    with pytest.raises(ProspectiveExecutionError, match="unsupported AVQDS seed-loader lane"):
        scout_worker_module._runner_argv(
            SimpleNamespace(lane_id="unknown_paper_ii_lane"),
            seed_receipt=None,
            executor_overlay=None,
            output_path="unused.json",
            repository_root=Path("."),
        )


def test_declared_hh_candidate_pool_is_rebuilt_and_digest_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = fixed_vqe_module.FixedVQEModelConfig(
        num_sites=1,
        t=1.0,
        u=1.0,
        dv=0.0,
        omega0=1.0,
        g_ep=0.5,
        n_ph_max=1,
        sector_n_up=1,
        sector_n_dn=0,
    )
    settings = model.to_settings_payload()
    settings["adapt_pool"] = "fixed_scaffold_locked"
    declared_digest = "a" * 64
    pool_contract = {
        "pool_key": "full_meta",
        "parent_term_count": 1,
        "atom_count": 1,
        "pauli_child_count": 1,
        "polyterm_child_count": 0,
        "ordered_atom_contract_sha256": declared_digest,
        "meta": {
            "atom_order": "parent_pool_order_then_parent_native_child_order",
            "symmetry_guard": "mandatory_fixed_count_sector_and_binary_padding",
        },
    }
    atom = SimpleNamespace(
        atom_label="declared_pool_atom",
        build_polynomial=lambda: PauliPolynomial(
            "JW", [PauliTerm(3, ps="eex", pc=1.0)]
        ),
    )
    rebuilt_pool = SimpleNamespace(
        pool_key="full_meta",
        atoms=(atom,),
        meta=dict(pool_contract["meta"]),
        ordered_atom_contract_sha256=declared_digest,
        to_json_dict=lambda: copy.deepcopy(pool_contract),
    )
    monkeypatch.setattr(
        fixed_vqe_module,
        "build_fixed_vqe_generator_pool",
        lambda **_kwargs: rebuilt_pool,
    )
    payload = {
        "schema": "paper_ii_vqe_seed_artifact_v1",
        "lane_id": LANE_HH,
        "model": "hubbard_holstein",
        "seed_id": f"{LANE_HH}::seed::test",
        "settings": settings,
        "replay_candidate_pool_mode": "replay_family_pool",
        "architecture_binding": {
            "policy": hh_seed_worker_module.architecture_policy(),
            "generator_pool": pool_contract,
            "generator_pool_ordered_atom_contract_sha256": declared_digest,
        },
    }
    loaded = SimpleNamespace(
        payload=payload,
        cfg=SimpleNamespace(
            L=1,
            t=1.0,
            u=1.0,
            dv=0.0,
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
            sector_n_up=1,
            sector_n_dn=0,
            paop_r=1,
            paop_split_paulis=False,
            paop_prune_eps=0.0,
            paop_normalization="none",
        ),
        replay_context=SimpleNamespace(
            cfg=object(),
            h_poly=object(),
            replay_terms=(object(), object()),
        ),
    )

    rebuilt = runtime_loader_module._declared_prospective_hh_candidate_pool(
        loaded,
        loader_mode="fixed_scaffold",
    )
    assert rebuilt is not None
    terms, source = rebuilt
    assert [term.label for term in terms] == ["declared_pool_atom"]
    assert source.completeness == "complete"
    override = source.filter_payload["explicit_complete_pool_override"]
    assert override["declared_generator_pool_sha256"] == declared_digest
    assert override["prospective_seed_declarations_verified"] is True

    tampered = copy.deepcopy(payload)
    tampered["architecture_binding"][
        "generator_pool_ordered_atom_contract_sha256"
    ] = "b" * 64
    with pytest.raises(ValueError, match="digest or metadata diverged"):
        runtime_loader_module._declared_prospective_hh_candidate_pool(
            SimpleNamespace(
                payload=tampered,
                replay_context=loaded.replay_context,
            ),
            loader_mode="fixed_scaffold",
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "rung_cut",
        "state_motion",
        "unsupported_mode",
        "paper_guard",
        "drive",
        "parameterization",
        "pool_digest",
        "pool_count",
        "pool_order",
        "hilbert_dimension",
        "missing_attestation",
        "uncured_step",
    ),
)
def test_raw_validation_rejects_every_locked_execution_contract_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    source, seed, request = _receipts_and_request(tmp_path)
    order = build_work_order(
        request,
        source_epoch_receipt=source,
        seed_receipt=seed,
        repository_root=tmp_path,
    )
    raw = _fake_raw(order, seed, source.source_epoch)
    if mutation == "rung_cut":
        raw["trajectory"]["support_patch_config"]["avqds_l2_cut"] = 1.0e-3
    elif mutation == "state_motion":
        raw["trajectory"]["solve_repair_config"][
            "state_motion_l2_step_max"
        ] = 1.0e-4
    elif mutation == "unsupported_mode":
        raw["paper_ii_execution_attestation"][
            "fail_on_unsupported_steps"
        ] = False
    elif mutation == "paper_guard":
        raw["run_lock"]["policy"]["guards"][
            "max_joint_patch_evaluations"
        ] = 2_000
    elif mutation == "drive":
        raw["hamiltonian"]["drive_profile"]["omega"] = 6.0
    elif mutation == "parameterization":
        raw["initial_state"]["parameterization_mode"] = "per_pauli_term"
    elif mutation == "pool_digest":
        raw["run_lock"]["shared_protocol"]["candidate_pool"][
            "ordered_atom_contract_sha256"
        ] = "0" * 64
    elif mutation == "pool_count":
        raw["run_lock"]["shared_protocol"]["candidate_pool"][
            "candidate_pool_deduplicated"
        ] = LOCKED_HUBBARD_POOL_COUNT - 1
    elif mutation == "pool_order":
        raw["run_lock"]["shared_protocol"]["candidate_pool"][
            "candidate_pool_order"
        ] = "reverse_order_v9"
    elif mutation == "hilbert_dimension":
        raw["plot_rows"][0]["psi_real"].pop()
        raw["plot_rows"][0]["psi_imag"].pop()
    elif mutation == "missing_attestation":
        raw.pop("paper_ii_execution_attestation")
    elif mutation == "uncured_step":
        raw["trajectory"]["points"][0]["integration_to_next"][
            "local_subdivision_reason"
        ] = "max_depth_not_cured"
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(mutation)
    raw_path = tmp_path / "tampered.json"
    _write(raw_path, raw)

    with pytest.raises(ValueError):
        scout_worker_module.validate_raw_run(
            raw_path,
            order=order,
            seed_receipt=seed,
            source_epoch=source.source_epoch,
        )


def test_raw_validation_accepts_distinct_declared_and_effective_pool_counts(
    tmp_path: Path,
) -> None:
    source, seed, request = _receipts_and_request(tmp_path)
    order = build_work_order(
        request,
        source_epoch_receipt=source,
        seed_receipt=seed,
        repository_root=tmp_path,
    )
    contract = copy.deepcopy(order.execution_contract)
    contract["initial_state_contract"]["candidate_pool_term_count"] = 1020
    contract["candidate_pool_source"]["filter_payload"] = {
        "explicit_complete_pool_override": {
            "declared_generator_pool_count": 1020,
            "declared_generator_pool_sha256": "1" * 64,
        }
    }
    contract["candidate_pool_contract"].update(
        {
            "candidate_pool_raw": 1800,
            "candidate_pool_deduplicated": 125,
            "ordered_atom_contract_sha256": "2" * 64,
        }
    )
    contract["execution_contract_sha256"] = _canonical_sha(
        {
            key: value
            for key, value in contract.items()
            if key != "execution_contract_sha256"
        }
    )
    order = replace(
        order,
        execution_contract=contract,
        work_order_sha256="3" * 64,
    )
    raw_path = tmp_path / "valid-distinct-pool-coordinates.json"
    _write(raw_path, _fake_raw(order, seed, source.source_epoch))

    scout_worker_module.validate_raw_run(
        raw_path,
        order=order,
        seed_receipt=seed,
        source_epoch=source.source_epoch,
    )


def test_direct_default_execution_requires_verified_packaged_source_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, seed, request = _receipts_and_request(tmp_path)
    order = build_work_order(
        request,
        source_epoch_receipt=source,
        seed_receipt=seed,
        repository_root=tmp_path,
    )
    monkeypatch.delenv(
        scout_worker_module.VERIFIED_PACKAGED_SOURCE_ENV,
        raising=False,
    )

    with pytest.raises(ValueError, match="verified packaged-source identity"):
        materialize_scout(order, repository_root=tmp_path)

    assert not (tmp_path / order.run_part_path).exists()


def test_materialization_is_unscored_immutable_and_crash_replay_safe(
    tmp_path: Path,
) -> None:
    source, seed, request = _receipts_and_request(tmp_path)
    order = build_work_order(
        request,
        source_epoch_receipt=source,
        seed_receipt=seed,
        repository_root=tmp_path,
    )
    calls = 0

    def execute(actual_order, root: Path) -> None:
        nonlocal calls
        calls += 1
        _write(
            root / actual_order.run_part_path,
            _fake_raw(actual_order, seed, source.source_epoch),
        )

    first = materialize_scout(
        order, repository_root=tmp_path, execute=execute
    )
    assert calls == 1
    assert first.status == SCOUT_STATUS
    assert first.to_bootstrap_evidence().action_id == request.action_id
    receipt_payload = json.loads(
        (tmp_path / first.receipt_path).read_text(encoding="utf-8")
    )
    assert receipt_payload["scientific_qualification"] is None
    assert receipt_payload["score_receipt"] is None
    assert receipt_payload["remote_raw_evidence_policy"] == "preserve"
    assert receipt_payload["remote_cleanup_authorized"] is False
    assert receipt_payload["qiskit_compilation_used"] is False
    assert (
        receipt_payload["executor_overlay_identity_sha256"]
        == order.executor_overlay.identity_sha256
    )
    assert (
        receipt_payload["composite_source_identity_sha256"]
        == order.executor_overlay.composite_source_identity_sha256
    )

    second = materialize_scout(
        order, repository_root=tmp_path, execute=execute
    )
    assert calls == 1
    assert second.receipt_sha256 == first.receipt_sha256

    receipt_path = tmp_path / order.completion_receipt_path
    receipt_path.unlink()
    recovered = materialize_scout(
        order, repository_root=tmp_path, execute=execute
    )
    assert calls == 1
    assert recovered.receipt_sha256 == first.receipt_sha256


def test_materialization_recovers_valid_interrupted_part_without_rerun(
    tmp_path: Path,
) -> None:
    source, seed, request = _receipts_and_request(tmp_path)
    order = build_work_order(
        request,
        source_epoch_receipt=source,
        seed_receipt=seed,
        repository_root=tmp_path,
    )
    _write(
        tmp_path / order.run_part_path,
        _fake_raw(order, seed, source.source_epoch),
    )

    def forbidden_execute(_order, _root: Path) -> None:
        raise AssertionError("valid interrupted output must not rerun")

    receipt = materialize_scout(
        order, repository_root=tmp_path, execute=forbidden_execute
    )
    assert (tmp_path / order.run_path).is_file()
    assert not (tmp_path / order.run_part_path).exists()
    assert receipt.status == SCOUT_STATUS


def test_local_package_is_inert_queue_one_and_preserves_raw_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, seed, request = _receipts_and_request(tmp_path)
    order = build_work_order(
        request,
        source_epoch_receipt=source,
        seed_receipt=seed,
        repository_root=tmp_path,
    )
    order_path = tmp_path / "orders/scout.json"
    write_work_order(order_path, order)
    source_archive = tmp_path / "runtime_source.tar.gz"
    materialize_avqds_runtime_source_archive(
        source_archive,
        repository_root=tmp_path,
        profile=SCOUT_PROFILE,
        base_revision=str(source.source_epoch.source_revision),
        base_full_archive_sha256=str(source.source_epoch.source_archive_sha256),
        overlay_revision=order.executor_overlay.overlay_revision,
        overlay_files=order.executor_overlay.files,
    )
    image_path = tmp_path / "runtime/test.sif"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"test-apptainer-image")
    identity_path = tmp_path / "runtime/image_identity.json"
    write_runtime_image_identity(
        identity_path, image_path=image_path, repository_root=tmp_path
    )
    observed_usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(
        scout_package.shutil,
        "disk_usage",
        lambda _path: observed_usage._replace(free=100 * 1024**3),
    )
    assert scout_package.main(
        [
            "materialize",
            "--repo-root",
            str(tmp_path),
            "--work-order",
            str(order_path),
            "--source-archive",
            str(source_archive),
            "--runtime-image-identity",
            str(identity_path),
            "--output-dir",
            "packages/avqds",
        ]
    ) == 0
    materialized_cli = json.loads(capsys.readouterr().out)
    assert materialized_cli["status"] == "prepared_local_only"
    assert materialized_cli["scheduler_contacted"] is False
    prepared = scout_package.validate_avqds_prefix_package(
        materialized_cli["preflight_receipt"],
        repository_root=tmp_path,
    )

    descriptor = (tmp_path / prepared.descriptor.path).read_text(encoding="utf-8")
    assert descriptor.splitlines().count("queue 1") == 1
    assert "transfer_output_files = raw_outputs" in descriptor
    assert "+PaperIIRemoteRawPreservation = True" in descriptor
    assert "+PaperIIExactReferenceBlind = True" in descriptor
    assert "+PaperIIQiskitCompilationRequested = False" in descriptor
    assert (
        f'+PaperIISourceEpochId = "{source.source_epoch.source_epoch_id}"'
        in descriptor
    )
    assert order.executor_overlay.base_source_revision == source.source_epoch.source_revision
    assert order.executor_overlay.overlay_revision != source.source_epoch.source_revision
    assert (
        order.executor_overlay.files["pipelines/scaffold/runtime_loader.py"][
            "base_sha256"
        ]
        != order.executor_overlay.files["pipelines/scaffold/runtime_loader.py"][
            "overlay_sha256"
        ]
    )
    preflight = json.loads((tmp_path / prepared.preflight.path).read_text(encoding="utf-8"))
    assert preflight["scheduler_contacted"] is False
    assert preflight["scheduler_submission_authorized"] is False
    assert preflight["condor_dry_run_executed"] is False
    assert preflight["scientific_qualification"] is None
    assert preflight["scoring_required_before_qualification"] is True
    assert (
        preflight["source_epoch_identity_sha256"]
        == source.source_epoch.identity_sha256
    )
    assert (
        preflight["executor_overlay_identity_sha256"]
        == order.executor_overlay.identity_sha256
    )
    assert (
        preflight["composite_source_identity_sha256"]
        == order.executor_overlay.composite_source_identity_sha256
    )
    with tarfile.open(tmp_path / prepared.package.path, "r:gz") as archive:
        names = set(archive.getnames())
        worker = archive.extractfile("run_avqds_prefix.sh")
        assert worker is not None
        worker_text = worker.read().decode("utf-8")
    syntax = subprocess.run(
        ("bash", "-n"), input=worker_text, text=True, capture_output=True
    )
    assert syntax.returncode == 0, syntax.stderr
    assert "input/seed.json" in names
    assert "input/work_order.json" in names
    assert "input/executor_overlay_identity.json" in names
    assert (
        "executor_overlay/pipelines/scaffold/runtime_loader.py" in names
    )
    assert (
        "executor_overlay/pipelines/time_dynamics/"
        "prospective_execution_bindings.py" in names
    )
    assert "BASE_SOURCE_VERIFIED_BEFORE_OVERLAY" in worker_text
    assert "pristine base digest mismatch" in worker_text
    assert "REMOTE_RAW_PRESERVED" in worker_text
    part_copy = (
        f'if [[ -f "$SOURCE_ROOT/{order.run_part_path}" ]]; then cp '
        f'"$SOURCE_ROOT/{order.run_part_path}" "$RAW_ROOT/run.json.part"; fi'
    )
    assert part_copy in worker_text
    assert worker_text.index(part_copy) < worker_text.index(
        '[[ "$status" -eq 0 ]] || exit "$status"'
    )
    assert "run_prospective_avqds_scout run" in worker_text
    assert (
        "PAPER_II_VERIFIED_PACKAGED_SOURCE_SHA256="
        f"{order.executor_overlay.composite_source_identity_sha256}"
        in worker_text
    )
    validated = scout_package.validate_avqds_prefix_package(
        prepared.preflight.path,
        repository_root=tmp_path,
    )
    assert validated.package == prepared.package
    assert validated.runtime_source_archive.path == source_archive.relative_to(
        tmp_path
    ).as_posix()
    assert validated.runtime_source_archive.sha256 == _file_sha(source_archive)
    assert validated.runtime_source_archive_bytes == source_archive.stat().st_size
    assert validated.runtime_source_inventory.profile == SCOUT_PROFILE
    assert (
        validated.runtime_source_inventory.archive_sha256
        == validated.runtime_source_archive.sha256
    )
    with tarfile.open(source_archive, "r:gz") as archive:
        source_names = tuple(archive.getnames())
    assert source_names
    assert all(name.endswith(".py") for name in source_names)
    assert not any("campaigns/" in name for name in source_names)
    assert not any("exact_response_scales" in name for name in source_names)
    assert not any("/output/" in name for name in source_names)

    successor = scout_package.materialize_avqds_prefix_package(
        work_order_path=order_path,
        runtime_source_archive_path=source_archive,
        runtime_image_identity_path=identity_path,
        output_dir="packages/avqds_g2",
        repository_root=tmp_path,
        operational_generation=2,
        free_bytes=100 * 1024**3,
    )
    assert successor.operational_generation == 2
    assert successor.resources == scout_package.AVQDSCHTCResources()
    successor_descriptor = (tmp_path / successor.descriptor.path).read_text(
        encoding="utf-8"
    )
    expected_unit = scout_package.avqds_scout_submission_unit_id(
        order.request, operational_generation=2
    )
    assert f'+AcquisitionUnitId = "{expected_unit}"' in successor_descriptor
    assert "+PaperIIOperationalGeneration = 2" in successor_descriptor
    assert (
        f'+JobBatchName = "{scout_package._batch_name(order, operational_generation=2)}"'
        in successor_descriptor
    )
    reloaded_successor = scout_package.validate_avqds_prefix_package(
        successor.preflight.path,
        repository_root=tmp_path,
    )
    assert reloaded_successor == successor

    unsafe_source = tmp_path / "unsafe_exact_source.tar.gz"
    with tarfile.open(source_archive, "r:gz") as archive:
        source_members = {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile()
        }
    source_members[
        "source/pipelines/time_dynamics/campaigns/"
        "paper_ii_exact_response_scales_v1.json"
    ] = b'{"schema":"paper_ii_exact_response_scales_v1"}\n'
    with tarfile.open(unsafe_source, "w:gz") as archive:
        for name, content in sorted(source_members.items()):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    with pytest.raises(ValueError, match="forbidden path"):
        scout_package._validate_source_archive(
            unsafe_source, root=tmp_path, order=order
        )

    trajectory_overlay = trajectory_worker_module.discover_executor_overlay(
        tmp_path, base_source_epoch=source.source_epoch
    )
    trajectory_source = tmp_path / "trajectory_exact_blind_source.tar.gz"
    trajectory_inventory = materialize_avqds_runtime_source_archive(
        trajectory_source,
        repository_root=tmp_path,
        profile=TRAJECTORY_PROFILE,
        base_revision=str(source.source_epoch.source_revision),
        base_full_archive_sha256=str(source.source_epoch.source_archive_sha256),
        overlay_revision=trajectory_overlay.overlay_revision,
        overlay_files=trajectory_overlay.files,
    )
    assert trajectory_inventory.profile == TRAJECTORY_PROFILE
    assert trajectory_inventory.archive_sha256 != _file_sha(source_archive)
    with tarfile.open(trajectory_source, "r:gz") as archive:
        trajectory_names = tuple(archive.getnames())
    assert trajectory_names
    assert all(name.endswith(".py") for name in trajectory_names)
    assert not any("campaigns/" in name for name in trajectory_names)
    assert not any("exact_response_scales" in name for name in trajectory_names)
    assert scout_package.main(
        [
            "validate",
            "--repo-root",
            str(tmp_path),
            "--preflight-receipt",
            prepared.preflight.path,
        ]
    ) == 0
    cli_payload = json.loads(capsys.readouterr().out)
    assert cli_payload["status"] == "prepared_local_only"
    assert cli_payload["scheduler_contacted"] is False
    assert cli_payload["scheduler_submission_authorized"] is False


def test_work_order_replay_is_matrix_file_blind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, seed, request = _receipts_and_request(tmp_path)
    order = build_work_order(
        request,
        source_epoch_receipt=source,
        seed_receipt=seed,
        repository_root=tmp_path,
    )
    path = tmp_path / "orders/matrix_blind.json"
    write_work_order(path, order)

    def forbidden_registry():
        raise AssertionError("remote work-order replay reopened campaign matrix")

    monkeypatch.setattr(scout_worker_module, "build_identity_registry", forbidden_registry)
    loaded, _source, _seed = load_work_order(path, repository_root=tmp_path)
    assert loaded.to_json_dict() == order.to_json_dict()


def test_frozen_submitted_work_order_uses_committed_overlay_and_rejects_resealed_runner_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, seed, request = _receipts_and_request(tmp_path)
    order = build_work_order(
        request,
        source_epoch_receipt=source,
        seed_receipt=seed,
        repository_root=tmp_path,
    )
    path = tmp_path / "orders/frozen_submitted.json"
    write_work_order(path, order)

    overlay_path = tmp_path / EXECUTOR_OVERLAY_PATHS[2]
    overlay_path.write_text(
        overlay_path.read_text(encoding="utf-8")
        + "\n# later committed executor revision\n",
        encoding="utf-8",
    )
    _git(tmp_path, "add", EXECUTOR_OVERLAY_PATHS[2])
    _git(tmp_path, "commit", "-qm", "later executor overlay")

    with pytest.raises(ProspectiveExecutionError, match="overlay bytes changed"):
        load_work_order(path, repository_root=tmp_path)

    def forbidden_rebuild(*_args, **_kwargs):
        raise AssertionError("frozen submitted work order used the live builder")

    monkeypatch.setattr(scout_worker_module, "build_work_order", forbidden_rebuild)
    loaded, loaded_source, loaded_seed = load_frozen_submitted_work_order(
        path, repository_root=tmp_path
    )
    assert loaded.to_json_dict() == order.to_json_dict()
    assert loaded_source.receipt_sha256 == source.receipt_sha256
    assert loaded_seed.receipt_sha256 == seed.receipt_sha256

    payload = json.loads(path.read_text(encoding="utf-8"))
    argv = payload["runner"]["argv"]
    argv[argv.index("--output-json") + 1] = "output/resealed-but-wrong.json.part"
    payload["runner"]["argv_sha256"] = _canonical_sha(argv)
    contract = payload["execution_contract"]
    contract["runner_argv_sha256"] = payload["runner"]["argv_sha256"]
    contract.pop("execution_contract_sha256")
    contract["execution_contract_sha256"] = _canonical_sha(contract)
    payload.pop("work_order_sha256")
    payload["work_order_sha256"] = _canonical_sha(payload)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ProspectiveExecutionError, match="runner binding diverged"):
        load_frozen_submitted_work_order(path, repository_root=tmp_path)


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("response_scales", {"t5": {"energy_rms_excursion": 1.0}}),
        ("exact_reference_receipt", {"path": "exact/receipt.json"}),
        ("unexpected_runtime_input", "not-authorized"),
    ),
)
def test_work_order_replay_rejects_reintroduced_exact_or_extra_inputs(
    tmp_path: Path, key: str, value: object
) -> None:
    source, seed, request = _receipts_and_request(tmp_path)
    order = build_work_order(
        request,
        source_epoch_receipt=source,
        seed_receipt=seed,
        repository_root=tmp_path,
    )
    path = tmp_path / "orders/reintroduced_exact.json"
    write_work_order(path, order)
    payload = json.loads(path.read_text(encoding="utf-8"))
    request_payload = payload["execution_request"]
    request_payload["inputs"][key] = value
    request_payload.pop("execution_request_sha256")
    request_payload["execution_request_sha256"] = _canonical_sha(request_payload)
    payload.pop("work_order_sha256")
    payload["work_order_sha256"] = _canonical_sha(payload)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(
        ProspectiveExecutionError,
        match="forbidden exact inputs|unauthorized inputs",
    ):
        load_work_order(path, repository_root=tmp_path)


@pytest.mark.parametrize(
    "source",
    (
        "from importlib import import_module as load\nload(target)\n",
        "import importlib\nimportlib.resources.files(package)\n",
        "import importlib.resources as resources\nresources.files(package)\n",
    ),
)
def test_runtime_closure_rejects_unmapped_aliased_or_nested_dynamic_import(
    source: str,
) -> None:
    path = "pipelines/unmapped_runtime_module.py"
    with pytest.raises(ValueError, match="unmapped dynamic runtime dependency"):
        runtime_source._import_targets(
            path,
            source.encode("utf-8"),
            frozenset({path}),
        )


def test_runtime_archive_rejects_noncanonical_member_spelling(
    tmp_path: Path,
) -> None:
    path = tmp_path / "double-slash.tar.gz"
    raw = io.BytesIO()
    with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
        with tarfile.open(
            fileobj=compressed, mode="w", format=tarfile.GNU_FORMAT
        ) as archive:
            content = b"# runtime\n"
            info = tarfile.TarInfo("source//pipelines/runtime.py")
            info.size = len(content)
            info.mode = 0o644
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(content))
    path.write_bytes(raw.getvalue())
    with pytest.raises(ValueError, match="unsafe member"):
        runtime_source._read_archive(path)


def test_package_rejects_tampered_committed_executor_overlay(tmp_path: Path) -> None:
    source, seed, request = _receipts_and_request(tmp_path)
    order = build_work_order(
        request,
        source_epoch_receipt=source,
        seed_receipt=seed,
        repository_root=tmp_path,
    )
    order_path = tmp_path / "orders/scout.json"
    write_work_order(order_path, order)
    source_archive = tmp_path / "runtime_source.tar.gz"
    materialize_avqds_runtime_source_archive(
        source_archive,
        repository_root=tmp_path,
        profile=SCOUT_PROFILE,
        base_revision=str(source.source_epoch.source_revision),
        base_full_archive_sha256=str(source.source_epoch.source_archive_sha256),
        overlay_revision=order.executor_overlay.overlay_revision,
        overlay_files=order.executor_overlay.files,
    )
    loader = tmp_path / "pipelines/scaffold/runtime_loader.py"
    loader.write_text(loader.read_text(encoding="utf-8") + "# tampered\n", encoding="utf-8")
    image = tmp_path / "runtime/test.sif"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"test image")
    identity = tmp_path / "runtime/image_identity.json"
    write_runtime_image_identity(identity, image_path=image, repository_root=tmp_path)

    with pytest.raises(ValueError, match="overlay bytes changed"):
        scout_package.materialize_avqds_prefix_package(
            work_order_path=order_path,
            runtime_source_archive_path=source_archive,
            runtime_image_identity_path=identity,
            output_dir="packages/tampered",
            repository_root=tmp_path,
            free_bytes=100 * 1024**3,
        )
