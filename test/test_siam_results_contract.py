"""Results-first SIAM promotion never rediscovers raw trajectories."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from pipelines.time_dynamics.diagnostics import (
    siam_results_contract as results_module,
)
from pipelines.resource_estimation.pauli_graph_span import (
    PAPER_I_MARRAKESH_GRAPH_SPAN_PROFILE_V1,
    marrakesh_logical_embedding_v1,
)
from pipelines.time_dynamics.diagnostics.siam_offline_scoring import (
    SIAM_ACCURACY_POLICY_V2,
    SIAM_OFFLINE_SCORE_V2,
    SIAM_OFFLINE_SCORE_V3,
    SIAM_OFFLINE_SCORE_V4,
    SIAM_ROUTE_HISTORICAL_UNRECORDED,
)
from pipelines.time_dynamics.diagnostics.siam_exact_reference_store import (
    SIAM_EXACT_REFERENCE_STORE_V2,
    SIAM_EXACT_SLICE_VIEW_V2,
    SIAMExactReferenceStore,
)
from pipelines.time_dynamics.diagnostics.siam_results_contract import (
    SIAM_RESULTS_DIAGNOSTIC_ROLE,
    SIAM_RESULTS_MANUSCRIPT_ROLE,
    SIAM_RESULTS_PDF_RECEIPT_V2,
    SIAM_RESULTS_PROMOTION_V2,
    SIAMResultsContractError,
    build_siam_results_pdf_receipt,
    build_siam_results_promotion,
    render_siam_results_pdf,
    write_siam_results_pdf_artifact,
    write_siam_results_promotion,
)
from pipelines.time_dynamics.diagnostics.siam_segment_stitching import (
    SIAM_NB4_STITCH_COMPLETION_V1,
)
from pipelines.time_dynamics.diagnostics.siam_terminal_resources import (
    FAKE_MARRAKESH_BACKEND_NAME,
    FAKE_MARRAKESH_GRAPH_SNAPSHOT_V1,
    FAKE_MARRAKESH_NUM_QUBITS,
    FAKE_MARRAKESH_UNDIRECTED_EDGES_SHA256_V1,
    FAKE_MARRAKESH_UNDIRECTED_EDGES_V1,
    SIAM_TERMINAL_RESOURCE_V1,
)
from pipelines.time_dynamics.siam_chtc_package import (
    SIAM_NB4_CHTC_RECEIPT_SCHEMA_V3,
)
from pipelines.time_dynamics.siam_activation_ladder import (
    SIAM_NB4_ACTIVATION_CUT_LADDER,
    SIAM_NB4_CANONICAL_LADDER_SHA256,
    canonical_index,
    ladder_payload,
)
from pipelines.time_dynamics.paper_ii_runs import build_siam_nb4_screen_run
from pipelines.time_dynamics.paper_ii_runs import (
    SIAM_NB4_CONTINUATION_TIMES,
    SIAM_NB4_FULL_TIMES,
    SIAM_NB4_SCREEN_TIMES,
)
from pipelines.time_dynamics.siam_screening import PILOT_NB4
from pipelines.time_dynamics.siam_seed import write_siam_seed_manifest
from src.quantum.siam_observables import SIAM_OBSERVABLES_V2
from src.quantum.siam_reference import SIAM_REFERENCE_V2


TEST_REVISION = "a" * 40


def _document_sha256(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pretty_sha256(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    ).hexdigest()


def _time_grid_sha256(times: tuple[float, ...] | list[float]) -> str:
    return hashlib.sha256(
        json.dumps(
            [float(value) for value in times],
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _score(
    *,
    policy: str,
    run_sha256: str,
    reference: dict[str, object],
    response_error: float = 0.02,
    mean_infidelity: float = 2.0e-5,
    g_e: float = 5.0e-4,
    grid_times: tuple[float, ...] = SIAM_NB4_SCREEN_TIMES,
    seed_sha256: str = "1" * 64,
    code_revision: str = TEST_REVISION,
    schema: str = SIAM_OFFLINE_SCORE_V3,
    exchange_route_bundle: str | None = None,
) -> dict[str, object]:
    primitive_names = ("m_d", "D_d", "C_1", "C_2", "C_3", "C_4")
    response_target = 0.05
    fidelity_ceiling = 1.0e-4
    energy_ceiling = 1.0e-3
    normalized = {
        name: response_error / response_target for name in primitive_names
    }
    normalized["fidelity"] = mean_infidelity / fidelity_ceiling
    correlation_mean = 0.25 * (
        normalized["C_1"] ** 2
        + normalized["C_2"] ** 2
        + normalized["C_3"] ** 2
        + normalized["C_4"] ** 2
    )
    burden = math.sqrt(
        0.25
        * (
            normalized["m_d"] ** 2
            + normalized["D_d"] ** 2
            + correlation_mean
            + normalized["fidelity"] ** 2
        )
    )
    pair_normalized = {**normalized, "energy": g_e / energy_ceiling}
    observable_individual_pass = response_error <= response_target
    observable_broad_pass = response_error <= 0.15
    fidelity_pass = mean_infidelity <= fidelity_ceiling
    energy_pass = g_e <= energy_ceiling
    individual_pass = bool(
        observable_individual_pass and fidelity_pass and energy_pass
    )
    broad_pass = bool(observable_broad_pass and fidelity_pass and energy_pass)
    grid_sha256 = _time_grid_sha256(grid_times)
    core: dict[str, object] = {
        "schema": schema,
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
        "source": {
            "run_path": "/deliberately/not/read/run.json",
            "run_sha256": run_sha256,
            "seed_path": "/deliberately/not/read/seed.json",
            "seed_sha256": seed_sha256,
            "post_quench_hamiltonian_sha256": "2" * 64,
        },
        "reference": reference,
        "grid": {
            "times": [float(value) for value in grid_times],
            "ordered_values_sha256": grid_sha256,
            "point_count": len(grid_times),
        },
        "method": {
            "structural_policy": policy,
            "comparison_fingerprint": "6" * 64,
            "physics_fingerprint": "7" * 64,
            "code_revision": code_revision,
            "run_lock_schema": "paper_ii_run_lock_v4",
        },
        "thresholds": {
            "response_floor": 1.0e-12,
            "individual_response_target": response_target,
            "broad_response_guard": 0.15,
            "mean_infidelity_ceiling": fidelity_ceiling,
            "relative_energy_drift_ceiling": energy_ceiling,
            "initial_state_infidelity_ceiling": 1.0e-12,
            "initial_energy_error_ceiling": 1.0e-10,
            "aggregate_burden_ratio_ceiling": 2.0,
            "componentwise_ratio_deadband": 0.10,
            "componentwise_ratio_ceiling": 2.0,
        },
        "primitive_response_metrics": {
            name: {
                "mean_absolute_error": response_error * 0.1,
                "response_scale": 0.1,
                "response_normalized_mean_absolute_error": response_error,
            }
            for name in primitive_names
        },
        "derived_screening_metrics": {
            f"Sigma_{radius}": {
                "mean_absolute_error": response_error * 0.1,
                "response_scale": 0.1,
                "response_normalized_mean_absolute_error": response_error,
            }
            for radius in range(1, 5)
        },
        "series": {
            "time": [float(value) for value in grid_times],
            "observed": {},
            "exact": {},
            "absolute_error": {},
        },
        "diagnostics": {
            "energy": {
                "sector_spectral_width": 2.0,
                "max_absolute_drift": 2.0 * g_e,
                "g_E": g_e,
                "relative_spectral_drift_ceiling": energy_ceiling,
                "pass": energy_pass,
            },
            "initial_boundary": {
                "boundary_time": float(grid_times[0]),
                "boundary_role": (
                    "immutable_seed_boundary"
                    if float(grid_times[0]) == 0.0
                    else "approximate_resume_boundary"
                ),
                "exact_seed_equality_required": float(grid_times[0]) == 0.0,
                "state_infidelity": 0.0,
                "absolute_energy_error": 0.0,
                "pass": True if float(grid_times[0]) == 0.0 else None,
            },
        },
        "summary": {
            "mean_infidelity": mean_infidelity,
            "max_primitive_response_normalized_error": response_error,
            "aggregate_accuracy_burden": burden,
            "normalized_accuracy_components": normalized,
            "pair_normalized_accuracy_components": pair_normalized,
            "individual_target_pass": individual_pass,
            "broad_response_guard_pass": broad_pass,
            "fidelity_pass": fidelity_pass,
            "energy_conservation_pass": energy_pass,
            "classification": (
                "qualified"
                if individual_pass
                else "candidate_broad_guard"
                if broad_pass
                else "candidate_outside_broad_guard"
            ),
        },
    }
    point_count = len(grid_times)
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
    if exchange_route_bundle is not None:
        core["method"]["exchange_route_bundle"] = exchange_route_bundle
        core["method"]["exchange_route_contract_sha256"] = "5" * 64
    return {**core, "score_sha256": _document_sha256(core)}


def _resource(
    *,
    run_sha256: str,
    accepted_steps: int,
    n_theta: int,
    seed_sha256: str = "1" * 64,
    terminal_time: float = 1.0,
) -> dict[str, object]:
    snapshot = {
        "schema": FAKE_MARRAKESH_GRAPH_SNAPSHOT_V1,
        "backend_name": FAKE_MARRAKESH_BACKEND_NAME,
        "num_qubits": FAKE_MARRAKESH_NUM_QUBITS,
        "undirected_coupling_edge_count": len(
            FAKE_MARRAKESH_UNDIRECTED_EDGES_V1
        ),
        "undirected_coupling_edges_sha256": (
            FAKE_MARRAKESH_UNDIRECTED_EDGES_SHA256_V1
        ),
        "source": "frozen_fake_marrakesh_conf_marrakesh_v1",
    }

    def absolute_record(
        *, theta: int, c2q: float, depth: float, c1q: float, support: str
    ) -> dict[str, object]:
        return {
            "schema": "pauli_graph_span_absolute_cost_v1",
            "profile_id": PAPER_I_MARRAKESH_GRAPH_SPAN_PROFILE_V1,
            "backend_name": FAKE_MARRAKESH_BACKEND_NAME,
            "no_transpile": True,
            "routing_seed_used": False,
            "num_qubits": 10,
            "embedding_logical_to_physical": list(
                marrakesh_logical_embedding_v1(10)
            ),
            "coupling_edge_count": len(FAKE_MARRAKESH_UNDIRECTED_EDGES_V1),
            "graph_snapshot": snapshot,
            "coefficient_tolerance": 1.0e-12,
            "ordered_generator_occurrence_count": theta,
            "nonidentity_pauli_rotation_count": theta * 5,
            "independent_parameter_count": theta,
            "absolute_c_hat_2q": c2q,
            "absolute_c_hat_d": depth,
            "absolute_c_hat_1q": c1q,
            "absolute_theta_count": float(theta),
            "ordered_support_sha256": support,
        }

    initial = absolute_record(
        theta=0, c2q=0.0, depth=0.0, c1q=0.0, support="8" * 64
    )
    terminal = absolute_record(
        theta=n_theta,
        c2q=float(100 + n_theta),
        depth=float(200 + n_theta),
        c1q=float(300 + n_theta),
        support="9" * 64,
    )
    initial_tuple = {
        "C_hat_2q": 0.0,
        "C_hat_d": 0.0,
        "C_hat_1q": 0.0,
        "N_theta": 0,
    }
    terminal_tuple = {
        "C_hat_2q": float(100 + n_theta),
        "C_hat_d": float(200 + n_theta),
        "C_hat_1q": float(300 + n_theta),
        "N_theta": n_theta,
    }
    delta = {
        "absolute_c_hat_2q": terminal_tuple["C_hat_2q"],
        "absolute_c_hat_d": terminal_tuple["C_hat_d"],
        "absolute_c_hat_1q": terminal_tuple["C_hat_1q"],
        "absolute_theta_count": float(n_theta),
    }
    core: dict[str, object] = {
        "schema": SIAM_TERMINAL_RESOURCE_V1,
        "trajectory_json": "/deliberately/not/read/run.json",
        "trajectory_json_sha256": run_sha256,
        "seed_json": "/deliberately/not/read/seed.json",
        "seed_json_sha256": seed_sha256,
        "problem_family": "siam",
        "parameterization_mode": "logical_shared",
        "terminal_time": terminal_time,
        "accepted_integration_steps": accepted_steps,
        "terminal_logical_macro_count": n_theta,
        "terminal_pauli_rotation_count": n_theta * 5,
        "canonical_tuple_order": [
            "C_hat_2q",
            "C_hat_d",
            "C_hat_1q",
            "N_theta",
        ],
        "initial_resource_tuple": initial_tuple,
        "terminal_resource_tuple": terminal_tuple,
        "terminal_minus_initial_resource_tuple": terminal_tuple,
        "graph_span_comparison": {
            "schema": "pauli_graph_span_trajectory_comparison_v1",
            "profile_id": PAPER_I_MARRAKESH_GRAPH_SPAN_PROFILE_V1,
            "reporting_role": "routing_independent_structural_cost_primary_v1",
            "no_transpile": True,
            "same_declared_graph_for_initial_and_terminal": True,
            "initial": initial,
            "terminal": terminal,
            "terminal_minus_initial": delta,
        },
        "qiskit_compilation": {
            "required": False,
            "performed": False,
            "role": "optional_diagnostic_sidecar_only",
        },
    }
    return {**core, "resource_sha256": _document_sha256(core)}


def _completion(
    *,
    policy: str,
    run_sha256: str,
    seed_sha256: str = "1" * 64,
    code_revision: str = TEST_REVISION,
    first_source_file_sha256: str = "2" * 64,
) -> dict[str, object]:
    package_id = f"package-{policy}"
    action_id = f"action-{policy}"
    seed_path = f"{package_id}/input/seed.json"
    output_path = f"raw_outputs/{package_id}/{action_id}/run.json"
    command = build_siam_nb4_screen_run(
        seed_path=seed_path,
        arm=policy,
        activation_cut=1.0e-3,
        output_json=output_path,
        code_revision=code_revision,
    )
    runner_argv = list(command.argv())
    runner_argv_sha256 = hashlib.sha256(
        json.dumps(
            runner_argv, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    source_rows = [
        {
            "path": "pipelines/time_dynamics/fixture.py",
            "kind": "file",
            "mode": 0o644,
            "bytes": 1,
            "sha256": first_source_file_sha256,
            "link_target": None,
        },
        {
            "path": "src/quantum/fixture.py",
            "kind": "file",
            "mode": 0o644,
            "bytes": 1,
            "sha256": "3" * 64,
            "link_target": None,
        },
    ]
    source_tree = {
        "schema": "siam_source_tree_contract_v1",
        "source_revision": code_revision,
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
        "source_revision": code_revision,
        "source_tree": source_tree,
        "arm": policy,
        "segment": "screen_t0_t1_101",
        "activation_cut": 1.0e-3,
        "activation_cut_ladder_index": canonical_index(1.0e-3),
        "activation_cut_ladder_sha256": SIAM_NB4_CANONICAL_LADDER_SHA256,
        "seed_path": seed_path,
        "seed_sha256": seed_sha256,
        "trajectory_json": output_path,
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
    action_sha256 = _pretty_sha256(action_contract)
    manifest_contract = {
        "schema": "siam_nb4_chtc_package_manifest_v3",
        "package_id": package_id,
        "action_id": action_id,
        "source_revision": code_revision,
        "activation_cut_ladder": list(
            ladder_payload(SIAM_NB4_ACTIVATION_CUT_LADDER)
        ),
        "activation_cut_ladder_sha256": SIAM_NB4_CANONICAL_LADDER_SHA256,
        "action_json_sha256": action_sha256,
        "runner_argv_sha256": runner_argv_sha256,
        "expected_run_id": command.run_id,
        "expected_output": output_path,
        "source_tree_contract": "siam_source_tree_contract_v1",
        "source_tree_files_sha256": source_tree["files_sha256"],
        "source_tree_file_count": source_tree["file_count"],
        "files": {"action": {"sha256": action_sha256}},
    }
    return {
        "schema": SIAM_NB4_CHTC_RECEIPT_SCHEMA_V3,
        "status": "COMPLETED",
        "package_id": package_id,
        "action_id": action_id,
        "run_id": command.run_id,
        "source_revision": code_revision,
        "seed_path": seed_path,
        "seed_sha256": seed_sha256,
        "arm": policy,
        "activation_cut": 1.0e-3,
        "segment": "screen_t0_t1_101",
        "exact_reference_blind": True,
        "qiskit_compilation_requested": False,
        "runner_argv": runner_argv,
        "runner_argv_sha256": runner_argv_sha256,
        "trajectory_json": output_path,
        "trajectory_sha256": run_sha256,
        "trajectory_bytes": 123,
        "resume_terminal_proof": {
            "schema": "siam_nb4_resume_terminal_proof_v1",
            "resume_schema": "paper_ii_resume_state_v2",
            "controller_schema": "paper_ii_controller_resume_state_v1",
            "terminal_time": 1.0,
            "terminal_checkpoint_index": 100,
            "logical_coordinate_count": 1,
            "runtime_pauli_coordinate_count": 1,
            "previous_accepted_velocity_count": 1,
            "support_contract_sha256": "4" * 64,
            "theta_contract_sha256": "5" * 64,
            "controller_state_sha256": "6" * 64,
            "reconstructed_state_semantic_sha256": "7" * 64,
            "terminal_state_semantic_sha256": "7" * 64,
            "reconstructed_state_infidelity": 0.0,
        },
        "action_json_sha256": action_sha256,
        "package_manifest_sha256": _pretty_sha256(manifest_contract),
        "source_tree_files_sha256": source_tree["files_sha256"],
        "source_tree_file_count": source_tree["file_count"],
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
            "initial_seed_state": True,
            "initial_seed_energy": True,
            "state_norm_sector": True,
            "resume_reconstruction": True,
            "resume_support_parameters": True,
            "resume_terminal_statevector": True,
            "controller_resume_reconstruction": True,
            "controller_resume_terminal_state": True,
        },
    }


def _genuine_exact_view(
    tmp_path: Path,
    *,
    view_kind: str = "prefix",
) -> tuple[dict[str, object], str]:
    shared = tmp_path.parent / "_siam_results_exact_fixture"
    seed_path = shared / "seed.json"
    write_siam_seed_manifest(seed_path, PILOT_NB4)
    seed_sha256 = hashlib.sha256(seed_path.read_bytes()).hexdigest()
    store = SIAMExactReferenceStore(shared / "exact")
    full = store.get_or_build(
        seed_path=seed_path,
        times=SIAM_NB4_FULL_TIMES,
    )
    if view_kind == "prefix":
        receipt_path = tmp_path / "exact_prefix_receipt.json"
        view = store.prefix_view(
            full,
            times=SIAM_NB4_SCREEN_TIMES,
            receipt_path=receipt_path,
        )
        field = "prefix_view"
    elif view_kind == "slice":
        receipt_path = tmp_path / "exact_slice_receipt.json"
        view = store.slice_view(
            full,
            times=SIAM_NB4_CONTINUATION_TIMES,
            receipt_path=receipt_path,
        )
        field = "slice_view"
    else:  # pragma: no cover - fixture misuse guard
        raise AssertionError(f"unsupported exact-view fixture {view_kind!r}")
    reference = {
        "key": full.key,
        "manifest_path": str(full.manifest_path.resolve()),
        "manifest_sha256": full.manifest_sha256,
        "data_path": str(full.data_path.resolve()),
        "data_sha256": full.data_sha256,
        "store_schema": SIAM_EXACT_REFERENCE_STORE_V2,
        "reference_schema": SIAM_REFERENCE_V2,
        "observable_schema": SIAM_OBSERVABLES_V2,
        "propagator_schema": str(full.manifest["propagator_schema"]),
        field: {
            "schema": str(view.receipt["schema"]),
            "receipt_path": str(receipt_path.resolve()),
            "receipt_sha256": str(view.receipt["receipt_sha256"]),
            "ordered_values_sha256": str(
                view.receipt[view_kind]["ordered_values_sha256"]
            ),
            "point_count": int(view.receipt[view_kind]["point_count"]),
        },
    }
    return reference, seed_sha256


def _fixture(
    tmp_path: Path,
    *,
    avqds_response_error: float = 0.02,
    apm_response_error: float = 0.02,
    apm_g_e: float = 5.0e-4,
    include_apm: bool = True,
    avqds_revision: str = TEST_REVISION,
    apm_revision: str = TEST_REVISION,
    apm_first_source_file_sha256: str = "2" * 64,
    apm_score_schema: str = SIAM_OFFLINE_SCORE_V3,
    apm_exchange_route_bundle: str | None = None,
) -> dict[str, Path | None]:
    reference, seed_sha256 = _genuine_exact_view(tmp_path)
    paths: dict[str, Path | None] = {
        "avqds_score": tmp_path / "avqds" / "score.json",
        "avqds_resource": tmp_path / "avqds" / "resources.json",
        "avqds_completion": tmp_path / "avqds" / "completion.json",
        "apm_score": None,
        "apm_resource": None,
        "apm_completion": None,
    }
    avqds_run = "a" * 64
    _write_json(
        paths["avqds_score"],
        _score(
            policy="avqds",
            run_sha256=avqds_run,
            response_error=avqds_response_error,
            reference=reference,
            seed_sha256=seed_sha256,
            code_revision=avqds_revision,
        ),
    )
    _write_json(
        paths["avqds_resource"],
        _resource(
            run_sha256=avqds_run,
            accepted_steps=120,
            n_theta=11,
            seed_sha256=seed_sha256,
        ),
    )
    _write_json(
        paths["avqds_completion"],
        _completion(
            policy="avqds",
            run_sha256=avqds_run,
            seed_sha256=seed_sha256,
            code_revision=avqds_revision,
        ),
    )
    if include_apm:
        paths["apm_score"] = tmp_path / "full_apm" / "score.json"
        paths["apm_resource"] = tmp_path / "full_apm" / "resources.json"
        paths["apm_completion"] = tmp_path / "full_apm" / "completion.json"
        apm_run = "b" * 64
        _write_json(
            paths["apm_score"],
            _score(
                policy="exchange",
                run_sha256=apm_run,
                response_error=apm_response_error,
                g_e=apm_g_e,
                reference=reference,
                seed_sha256=seed_sha256,
                code_revision=apm_revision,
                schema=apm_score_schema,
                exchange_route_bundle=apm_exchange_route_bundle,
            ),
        )
        _write_json(
            paths["apm_resource"],
            _resource(
                run_sha256=apm_run,
                accepted_steps=234,
                n_theta=9,
                seed_sha256=seed_sha256,
            ),
        )
        _write_json(
            paths["apm_completion"],
            _completion(
                policy="exchange",
                run_sha256=apm_run,
                seed_sha256=seed_sha256,
                code_revision=apm_revision,
                first_source_file_sha256=apm_first_source_file_sha256,
            ),
        )
    return paths


def _build(
    paths: dict[str, Path | None],
    *,
    promotion_role: str = SIAM_RESULTS_DIAGNOSTIC_ROLE,
):
    return build_siam_results_promotion(
        avqds_score_path=paths["avqds_score"],
        avqds_resource_path=paths["avqds_resource"],
        avqds_completion_receipt_path=paths["avqds_completion"],
        full_apm_score_path=paths["apm_score"],
        full_apm_resource_path=paths["apm_resource"],
        full_apm_completion_receipt_path=paths["apm_completion"],
        promotion_role=promotion_role,
    )


def test_contract_uses_scores_and_qiskit_free_resources_without_raw_runs(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)

    receipt, latex = _build(paths)

    assert receipt["schema"] == SIAM_RESULTS_PROMOTION_V2
    assert receipt["observable_contract"]["schema"] == SIAM_OBSERVABLES_V2
    assert receipt["promotion_role"] == SIAM_RESULTS_DIAGNOSTIC_ROLE
    assert receipt["counts"] == {"qualified": 2, "candidate": 0, "pending": 0}
    assert receipt["exact"]["state"] == "reference"
    assert receipt["slots"]["avqds"]["state"] == "qualified"
    assert receipt["slots"]["full_apm"]["state"] == "qualified"
    assert receipt["slots"]["avqds"]["individual_gate_pass"] is True
    assert receipt["slots"]["avqds"]["qualification_basis"] == (
        "individual_target"
    )
    assert receipt["slots"]["avqds"]["accepted_integration_steps"] == 120
    assert receipt["slots"]["avqds"]["completion_proof_role"] == (
        "diagnostic_screen"
    )
    assert receipt["slots"]["avqds"]["completion_segment"] == (
        "screen_t0_t1_101"
    )
    assert receipt["slots"]["avqds"]["terminal_time"] == 1.0
    assert receipt["slots"]["avqds"]["manuscript_eligible"] is False
    exact_view_input = receipt["slots"]["avqds"]["inputs"]["score"][
        "exact_view_receipt"
    ]
    exact_view_path = Path(exact_view_input["path"])
    assert exact_view_input["schema"] == (
        "siam_exact_reference_prefix_view_v2"
    )
    assert exact_view_input["file_sha256"] == hashlib.sha256(
        exact_view_path.read_bytes()
    ).hexdigest()
    assert exact_view_input["receipt_sha256"] == json.loads(
        exact_view_path.read_text(encoding="utf-8")
    )["receipt_sha256"]
    assert receipt["slots"]["full_apm"]["terminal_n_theta"] == 9
    assert receipt["slots"]["full_apm"]["terminal_graph_proxy"] == [
        109.0,
        209.0,
        309.0,
    ]
    assert receipt["canonical_resource_accounting"][
        "qiskit_compilation_required"
    ] is False
    assert receipt["pair_comparison"]["cost_comparison_admissible"] is True
    assert receipt["pair_comparison"]["reader_verdict"] == (
        "individually qualified + matched"
    )
    assert "Exact & reference" in latex
    assert "AVQDS & qualified" in latex
    assert "Full APM & qualified" in latex
    assert "Pair verdict: individually qualified + matched." in latex
    assert "diagnostic screening evidence; not manuscript promotion" in latex
    assert "Qiskit" not in latex
    assert all(
        line.endswith(r"\\")
        for line in latex.splitlines()
        if line.startswith(("Exact", "AVQDS", "Full APM"))
    )


def test_results_promotion_rejects_a_redigested_sigma_v1_score(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    score_path = paths["apm_score"]
    assert score_path is not None
    score = json.loads(score_path.read_text(encoding="utf-8"))
    score.pop("score_sha256")
    score["schema"] = SIAM_OFFLINE_SCORE_V2
    score.pop("observable_contract")
    for field in ("store_schema", "reference_schema", "observable_schema"):
        score["reference"].pop(field)
    score["score_sha256"] = _document_sha256(score)
    _write_json(score_path, score)

    with pytest.raises(SIAMResultsContractError, match="Sigma-v2"):
        _build(paths)


def test_revision_only_mismatch_keeps_both_numerical_candidates(
    tmp_path: Path,
) -> None:
    other_revision = "b" * 40
    paths = _fixture(tmp_path, apm_revision=other_revision)
    avqds_score = json.loads(
        paths["avqds_score"].read_text(encoding="utf-8")
    )
    apm_score = json.loads(paths["apm_score"].read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="method.code_revision"):
        results_module.classify_siam_matched_pair(apm_score, avqds_score)

    receipt, latex = _build(paths)

    assert receipt["counts"] == {"qualified": 0, "candidate": 2, "pending": 0}
    assert receipt["slots"]["avqds"]["state"] == "candidate"
    assert receipt["slots"]["full_apm"]["state"] == "candidate"
    assert receipt["slots"]["avqds"]["mean_infidelity"] == 2.0e-5
    assert receipt["slots"]["full_apm"]["terminal_n_theta"] == 9
    pair = receipt["pair_comparison"]
    assert pair["state"] == "pending_incomparable"
    assert pair["classification"] == "pending_incomparable_code_revision"
    assert pair["cost_comparison_admissible"] is False
    assert pair["pending_reason"] == "method_code_revision_mismatch"
    assert pair["reader_verdict"] == "pending - incomparable revisions"
    assert pair["required_resolution"] == (
        "Final stitched continuations must share one code revision before "
        "matched-cost claims."
    )
    assert len(pair["shared_source_tree_files_sha256"]) == 64
    assert "AVQDS & candidate" in latex
    assert "Full APM & candidate" in latex
    assert pair["required_resolution"] in latex


def test_revision_mismatch_with_different_source_tree_fails_closed(
    tmp_path: Path,
) -> None:
    paths = _fixture(
        tmp_path,
        apm_revision="b" * 40,
        apm_first_source_file_sha256="8" * 64,
    )

    with pytest.raises(
        SIAMResultsContractError,
        match="different source-tree digests",
    ):
        _build(paths)


def test_candidate_numbers_remain_visible_and_are_not_changed_to_pending(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, apm_response_error=0.08)

    receipt, latex = _build(paths)
    apm = receipt["slots"]["full_apm"]

    assert receipt["counts"] == {"qualified": 0, "candidate": 2, "pending": 0}
    assert receipt["slots"]["avqds"]["state"] == "candidate"
    assert receipt["slots"]["avqds"]["qualification_basis"] is None
    assert apm["state"] == "candidate"
    assert apm["response_normalized_primitive_errors"]["m_d"] == 0.08
    assert apm["mean_infidelity"] == 2.0e-5
    assert apm["accepted_integration_steps"] == 234
    assert apm["terminal_n_theta"] == 9
    assert apm["individual_gate_pass"] is False
    assert apm["individual_failed_gates"] == [
        "individual_response_target_pass"
    ]
    assert receipt["pair_comparison"]["reader_verdict"] == (
        "accuracy-confounded"
    )
    assert "AVQDS & candidate" in latex
    assert "Full APM & candidate" in latex
    assert "8.000e-02" in latex
    assert "Full APM & pending" not in latex


def test_missing_completed_arm_is_a_pending_placeholder(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, include_apm=False)

    receipt, latex = _build(paths)

    assert receipt["counts"] == {"qualified": 1, "candidate": 0, "pending": 1}
    assert receipt["slots"]["full_apm"]["state"] == "pending"
    assert receipt["slots"]["full_apm"]["mean_infidelity"] is None
    assert receipt["pair_comparison"]["state"] == "pending"
    assert "Full APM & pending & --" in latex
    assert "Pair verdict: pending." in latex


def test_matched_candidate_pair_upgrades_both_slots_to_qualified(
    tmp_path: Path,
) -> None:
    paths = _fixture(
        tmp_path,
        avqds_response_error=0.08,
        apm_response_error=0.08,
    )

    receipt, latex = _build(paths)

    assert receipt["pair_comparison"]["cost_comparison_admissible"] is True
    assert receipt["pair_comparison"]["reader_verdict"] == "matched"
    assert receipt["counts"] == {"qualified": 2, "candidate": 0, "pending": 0}
    for method in ("avqds", "full_apm"):
        slot = receipt["slots"][method]
        assert slot["state"] == "qualified"
        assert slot["qualification_basis"] == "matched_pair"
        assert slot["individual_gate_pass"] is False
        assert slot["individual_failed_gates"] == [
            "individual_response_target_pass"
        ]
    assert "AVQDS & qualified" in latex
    assert "Full APM & qualified" in latex
    assert "Pair verdict: matched." in latex


def test_failed_energy_guard_is_a_numerical_candidate(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, apm_g_e=2.0e-3)

    receipt, latex = _build(paths)
    apm = receipt["slots"]["full_apm"]

    assert apm["state"] == "candidate"
    assert apm["response_normalized_primitive_errors"]["m_d"] == 0.02
    assert apm["energy_guard"] == {
        "g_E": 2.0e-3,
        "ceiling": 1.0e-3,
        "pass": False,
    }
    assert apm["failed_gates"] == ["energy_guard_pass"]
    assert "Full APM & candidate" in latex


def test_failed_fidelity_guard_with_passing_observables_is_a_candidate(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    score_path = paths["apm_score"]
    assert score_path is not None
    prior_score = json.loads(score_path.read_text(encoding="utf-8"))
    score = _score(
        policy="exchange",
        run_sha256="b" * 64,
        response_error=0.02,
        mean_infidelity=2.0e-4,
        reference=prior_score["reference"],
        seed_sha256=prior_score["source"]["seed_sha256"],
    )
    _write_json(score_path, score)

    receipt, latex = _build(paths)
    apm = receipt["slots"]["full_apm"]

    assert apm["state"] == "candidate"
    assert apm["gates"]["individual_response_target_pass"] is True
    assert apm["gates"]["fidelity_ceiling_pass"] is False
    assert apm["failed_gates"] == ["fidelity_ceiling_pass"]
    assert "Full APM & candidate" in latex


def test_promotion_is_deterministic_idempotent_and_digest_bound(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    first, first_latex = _build(paths)
    second, second_latex = _build(paths)
    assert first == second
    assert first_latex == second_latex

    receipt_path = tmp_path / "promotion.json"
    latex_path = tmp_path / "table.tex"
    write_siam_results_promotion(
        receipt_path=receipt_path,
        latex_path=latex_path,
        receipt=first,
        latex=first_latex,
    )
    original_receipt = receipt_path.read_bytes()
    original_latex = latex_path.read_bytes()
    write_siam_results_promotion(
        receipt_path=receipt_path,
        latex_path=latex_path,
        receipt=first,
        latex=first_latex,
    )
    assert receipt_path.read_bytes() == original_receipt
    assert latex_path.read_bytes() == original_latex
    with pytest.raises(SIAMResultsContractError, match="LaTeX fragment digest"):
        write_siam_results_promotion(
            receipt_path=tmp_path / "other.json",
            latex_path=tmp_path / "other.tex",
            receipt=first,
            latex=first_latex + "% changed\n",
        )


def test_results_pdf_is_deterministic_digest_bound_and_results_first(
    tmp_path: Path,
) -> None:
    from pypdf import PdfReader

    receipt, _latex = _build(
        _fixture(tmp_path / "inputs", apm_response_error=0.08)
    )
    first = render_siam_results_pdf(receipt)
    second = render_siam_results_pdf(receipt)
    assert first == second
    assert first.startswith(b"%PDF-")

    pdf_receipt = build_siam_results_pdf_receipt(receipt, first)
    assert pdf_receipt["schema"] == SIAM_RESULTS_PDF_RECEIPT_V2
    assert pdf_receipt["promotion_sha256"] == receipt["promotion_sha256"]
    assert pdf_receipt["pdf"]["sha256"] == hashlib.sha256(first).hexdigest()
    assert pdf_receipt["qiskit_compilation_used"] is False

    pdf_path = tmp_path / "siam_results.pdf"
    pdf_receipt_path = tmp_path / "siam_results_pdf_receipt.json"
    write_siam_results_pdf_artifact(
        pdf_path=pdf_path,
        receipt_path=pdf_receipt_path,
        pdf_bytes=first,
        receipt=pdf_receipt,
    )
    write_siam_results_pdf_artifact(
        pdf_path=pdf_path,
        receipt_path=pdf_receipt_path,
        pdf_bytes=first,
        receipt=pdf_receipt,
    )
    reader = PdfReader(pdf_path)
    assert len(reader.pages) == 1
    text = reader.pages[0].extract_text()
    assert "SIAM screening benchmark results" in text
    assert "Pair verdict: accuracy-confounded" in text
    assert "AVQDS" in text and "Full APM" in text
    assert "candidate" in text
    assert "no Qiskit compilation is used" in text
    assert "declared-grid" in text
    assert "full-grid" not in text

    altered = bytearray(first)
    altered[-20] ^= 1
    with pytest.raises(SIAMResultsContractError, match="receipt mismatch"):
        write_siam_results_pdf_artifact(
            pdf_path=tmp_path / "tampered.pdf",
            receipt_path=tmp_path / "tampered.json",
            pdf_bytes=bytes(altered),
            receipt=pdf_receipt,
        )


def test_tampered_score_and_cross_run_resource_fail_closed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    score = json.loads(paths["avqds_score"].read_text(encoding="utf-8"))
    score["summary"]["mean_infidelity"] = 9.0e-5
    _write_json(paths["avqds_score"], score)
    with pytest.raises(SIAMResultsContractError, match="fails verification"):
        _build(paths)

    paths = _fixture(tmp_path / "cross_run")
    resource = json.loads(paths["avqds_resource"].read_text(encoding="utf-8"))
    resource.pop("resource_sha256")
    resource["trajectory_json_sha256"] = "c" * 64
    resource["resource_sha256"] = _document_sha256(resource)
    _write_json(paths["avqds_resource"], resource)
    with pytest.raises(SIAMResultsContractError, match="different trajectories"):
        _build(paths)


def test_missing_exact_view_receipt_is_a_hard_rejection(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, include_apm=False)
    score_path = paths["avqds_score"]
    assert score_path is not None
    score = json.loads(score_path.read_text(encoding="utf-8"))
    Path(score["reference"]["prefix_view"]["receipt_path"]).unlink()

    with pytest.raises(
        SIAMResultsContractError,
        match="exact-reference view receipt",
    ):
        _build(paths)


def test_redigested_tampered_exact_view_receipt_is_rejected(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, include_apm=False)
    score_path = paths["avqds_score"]
    assert score_path is not None
    score = json.loads(score_path.read_text(encoding="utf-8"))
    receipt_path = Path(score["reference"]["prefix_view"]["receipt_path"])
    view_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    view_receipt.pop("receipt_sha256")
    view_receipt["prefix"]["t_final"] = 0.99
    view_receipt["receipt_sha256"] = _document_sha256(view_receipt)
    _write_json(receipt_path, view_receipt)

    score.pop("score_sha256")
    score["reference"]["prefix_view"]["receipt_sha256"] = view_receipt[
        "receipt_sha256"
    ]
    score["score_sha256"] = _document_sha256(score)
    _write_json(score_path, score)

    with pytest.raises(
        SIAMResultsContractError,
        match="exact-reference-store verification",
    ):
        _build(paths)


def test_genuine_slice_receipt_is_reopened_and_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path, include_apm=False)
    score_path = paths["avqds_score"]
    resource_path = paths["avqds_resource"]
    assert score_path is not None
    assert resource_path is not None
    reference, seed_sha256 = _genuine_exact_view(
        tmp_path / "continuation",
        view_kind="slice",
    )
    run_sha256 = "a" * 64
    _write_json(
        score_path,
        _score(
            policy="avqds",
            run_sha256=run_sha256,
            reference=reference,
            grid_times=SIAM_NB4_CONTINUATION_TIMES,
            seed_sha256=seed_sha256,
        ),
    )
    _write_json(
        resource_path,
        _resource(
            run_sha256=run_sha256,
            accepted_steps=120,
            n_theta=11,
            seed_sha256=seed_sha256,
            terminal_time=5.0,
        ),
    )

    def validate_continuation(_receipt, **expected):
        return {
            "package_id": "package-avqds",
            "action_id": "action-avqds",
            "trajectory_sha256": expected["expected_trajectory_sha256"],
            "segment": results_module.SIAM_NB4_CONTINUATION_SEGMENT.segment_id,
            "source_tree_files_sha256": "d" * 64,
        }

    monkeypatch.setattr(
        results_module,
        "validate_siam_nb4_completion_receipt",
        validate_continuation,
    )

    promotion, _latex = _build(paths)
    slot = promotion["slots"]["avqds"]
    view_input = slot["inputs"]["score"]["exact_view_receipt"]
    assert slot["completion_proof_role"] == "diagnostic_continuation"
    assert slot["terminal_time"] == 5.0
    assert view_input["schema"] == SIAM_EXACT_SLICE_VIEW_V2
    assert view_input["view_kind"] == "slice"
    assert view_input["point_count"] == 401
    assert view_input["start_index"] == 100
    assert view_input["stop_index_exclusive"] == 501


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    (
        ("thresholds", "individual_response_target", 0.06, "exact policy-v2 thresholds"),
        ("thresholds", "componentwise_ratio_ceiling", 3.0, "exact policy-v2 thresholds"),
        ("accuracy_policy", "energy_in_aggregate_burden", True, "exact policy-v2 identity"),
        ("accuracy_policy", "extra_policy_switch", False, "exact policy-v2 identity"),
    ),
)
def test_policy_v2_constants_are_pinned_during_promotion(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    paths = _fixture(tmp_path, include_apm=False)
    score_path = paths["avqds_score"]
    assert score_path is not None
    score = json.loads(score_path.read_text(encoding="utf-8"))
    score.pop("score_sha256")
    score[section][field] = value
    score["score_sha256"] = _document_sha256(score)
    _write_json(score_path, score)

    with pytest.raises(SIAMResultsContractError, match=message):
        _build(paths)


@pytest.mark.parametrize(
    ("section", "missing_name", "message"),
    (
        ("primitive_response_metrics", "C_4", "must contain exactly"),
        (
            "derived_screening_metrics",
            "Sigma_4",
            "Sigma-v2 derived-screening metric schema mismatch",
        ),
    ),
)
def test_nb4_promotion_requires_every_primitive_and_screening_radius(
    tmp_path: Path, section: str, missing_name: str, message: str
) -> None:
    paths = _fixture(tmp_path, include_apm=False)
    score_path = paths["avqds_score"]
    assert score_path is not None
    score = json.loads(score_path.read_text(encoding="utf-8"))
    score.pop("score_sha256")
    score[section].pop(missing_name)
    if section == "primitive_response_metrics":
        score["summary"]["normalized_accuracy_components"].pop(missing_name)
        score["summary"]["pair_normalized_accuracy_components"].pop(missing_name)
        remaining = score["summary"]["normalized_accuracy_components"]
        correlation_names = sorted(
            name for name in remaining if name.startswith("C_")
        )
        correlation_mean = sum(
            remaining[name] ** 2 for name in correlation_names
        ) / len(correlation_names)
        score["summary"]["aggregate_accuracy_burden"] = math.sqrt(
            0.25
            * (
                remaining["m_d"] ** 2
                + remaining["D_d"] ** 2
                + correlation_mean
                + remaining["fidelity"] ** 2
            )
        )
    score["score_sha256"] = _document_sha256(score)
    _write_json(score_path, score)

    with pytest.raises(SIAMResultsContractError, match=message):
        _build(paths)


def test_nb4_promotion_rejects_inconsistent_sigma_normalization(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, include_apm=False)
    score_path = paths["avqds_score"]
    assert score_path is not None
    score = json.loads(score_path.read_text(encoding="utf-8"))
    score.pop("score_sha256")
    score["derived_screening_metrics"]["Sigma_4"][
        "response_normalized_mean_absolute_error"
    ] = 0.03
    score["score_sha256"] = _document_sha256(score)
    _write_json(score_path, score)

    with pytest.raises(SIAMResultsContractError, match="normalization is inconsistent"):
        _build(paths)


def test_incomparable_exact_references_and_qiskit_requirement_fail_closed(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    score = json.loads(paths["apm_score"].read_text(encoding="utf-8"))
    score.pop("score_sha256")
    score["reference"]["key"] = "other-reference"
    score["score_sha256"] = _document_sha256(score)
    _write_json(paths["apm_score"], score)
    with pytest.raises(SIAMResultsContractError, match="full reference identity"):
        _build(paths)

    paths = _fixture(tmp_path / "qiskit")
    resource = json.loads(paths["apm_resource"].read_text(encoding="utf-8"))
    resource.pop("resource_sha256")
    resource["qiskit_compilation"]["required"] = True
    resource["resource_sha256"] = _document_sha256(resource)
    _write_json(paths["apm_resource"], resource)
    with pytest.raises(SIAMResultsContractError, match="strictly Qiskit-free"):
        _build(paths)


def test_graph_proxy_tuple_and_frozen_provenance_fail_closed(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path / "tuple")
    resource_path = paths["apm_resource"]
    assert resource_path is not None
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    resource.pop("resource_sha256")
    resource["terminal_resource_tuple"]["C_hat_2q"] += 1.0
    resource["resource_sha256"] = _document_sha256(resource)
    _write_json(resource_path, resource)
    with pytest.raises(SIAMResultsContractError, match="graph-span derivation"):
        _build(paths)

    paths = _fixture(tmp_path / "snapshot")
    resource_path = paths["apm_resource"]
    assert resource_path is not None
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    resource.pop("resource_sha256")
    resource["graph_span_comparison"]["terminal"]["graph_snapshot"][
        "undirected_coupling_edges_sha256"
    ] = "0" * 64
    resource["resource_sha256"] = _document_sha256(resource)
    _write_json(resource_path, resource)
    with pytest.raises(SIAMResultsContractError, match="graph_snapshot"):
        _build(paths)

    paths = _fixture(tmp_path / "profile")
    resource_path = paths["apm_resource"]
    assert resource_path is not None
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    resource.pop("resource_sha256")
    resource["graph_span_comparison"]["profile_id"] = "other-profile"
    resource["resource_sha256"] = _document_sha256(resource)
    _write_json(resource_path, resource)
    with pytest.raises(SIAMResultsContractError, match="profile_id"):
        _build(paths)


def test_all_pending_and_half_bound_slots_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(SIAMResultsContractError, match="At least one verified"):
        build_siam_results_promotion()

    score_path = tmp_path / "score.json"
    reference, seed_sha256 = _genuine_exact_view(tmp_path)
    _write_json(
        score_path,
        _score(
            policy="avqds",
            run_sha256="a" * 64,
            reference=reference,
            seed_sha256=seed_sha256,
        ),
    )
    with pytest.raises(SIAMResultsContractError, match="requires score"):
        build_siam_results_promotion(avqds_score_path=score_path)


def test_negative_completion_receipt_is_a_hard_rejection(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, include_apm=False)
    completion_path = paths["avqds_completion"]
    assert completion_path is not None
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["status"] = "FAILED"
    _write_json(completion_path, completion)

    with pytest.raises(
        SIAMResultsContractError, match="requires status COMPLETED"
    ):
        _build(paths)


@pytest.mark.parametrize(
    "field",
    (
        "run_id",
        "activation_cut",
        "segment",
        "runner_argv",
        "runner_argv_sha256",
        "trajectory_bytes",
    ),
)
def test_incomplete_v3_completion_receipt_is_a_hard_rejection(
    tmp_path: Path, field: str
) -> None:
    paths = _fixture(tmp_path, include_apm=False)
    completion_path = paths["avqds_completion"]
    assert completion_path is not None
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion.pop(field)
    _write_json(completion_path, completion)

    with pytest.raises(SIAMResultsContractError, match="canonical validation"):
        _build(paths)


def test_t1_screen_is_diagnostic_and_cannot_enter_manuscript_role(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, include_apm=False)

    receipt, latex = _build(paths, promotion_role=SIAM_RESULTS_DIAGNOSTIC_ROLE)
    assert receipt["manuscript_contract"] == {
        "validated_stitched_t5_required": False,
        "all_numerical_slots_validated_stitched_t5": False,
        "screen_t1_role": "diagnostic_only",
    }
    assert receipt["slots"]["avqds"]["completion_proof_role"] == (
        "diagnostic_screen"
    )
    assert "not manuscript promotion" in latex

    with pytest.raises(
        SIAMResultsContractError,
        match="manuscript promotion requires a validated stitched T=5",
    ):
        _build(paths, promotion_role=SIAM_RESULTS_MANUSCRIPT_ROLE)


def test_manuscript_role_accepts_only_validated_stitched_t5_proofs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    full_times = list(SIAM_NB4_FULL_TIMES)
    for prefix in ("avqds", "apm"):
        score_path = paths[f"{prefix}_score"]
        resource_path = paths[f"{prefix}_resource"]
        completion_path = paths[f"{prefix}_completion"]
        assert score_path is not None
        assert resource_path is not None
        assert completion_path is not None

        score = json.loads(score_path.read_text(encoding="utf-8"))
        score.pop("score_sha256")
        score["reference"].pop("prefix_view")
        score["grid"] = {
            "times": full_times,
            "ordered_values_sha256": _time_grid_sha256(full_times),
            "point_count": 501,
        }
        score["score_sha256"] = _document_sha256(score)
        _write_json(score_path, score)

        resource = json.loads(resource_path.read_text(encoding="utf-8"))
        resource.pop("resource_sha256")
        resource["terminal_time"] = 5.0
        resource["resource_sha256"] = _document_sha256(resource)
        _write_json(resource_path, resource)
        _write_json(completion_path, {"schema": SIAM_NB4_STITCH_COMPLETION_V1})

    def validate_stitched(receipt, **expected):
        assert receipt["schema"] == SIAM_NB4_STITCH_COMPLETION_V1
        return {
            "schema": SIAM_NB4_STITCH_COMPLETION_V1,
            "completion_sha256": "c" * 64,
            "stitched_trajectory": {
                "sha256": expected["expected_trajectory_sha256"]
            },
            "source_segments": [
                {
                    "source_revision": TEST_REVISION,
                    "completion_receipt": {
                        "source_tree_files_sha256": "d" * 64
                    },
                },
                {
                    "source_revision": TEST_REVISION,
                    "completion_receipt": {
                        "source_tree_files_sha256": "d" * 64
                    },
                },
            ],
        }

    monkeypatch.setattr(
        results_module,
        "validate_siam_nb4_stitch_completion_receipt",
        validate_stitched,
    )

    receipt, latex = _build(
        paths, promotion_role=SIAM_RESULTS_MANUSCRIPT_ROLE
    )
    assert receipt["promotion_role"] == SIAM_RESULTS_MANUSCRIPT_ROLE
    assert receipt["manuscript_contract"][
        "all_numerical_slots_validated_stitched_t5"
    ] is True
    for slot in ("avqds", "full_apm"):
        assert receipt["slots"][slot]["completion_proof_role"] == (
            "full_horizon_stitched"
        )
        assert receipt["slots"][slot]["completion_segment"] == (
            "stitched_t0_t5_501"
        )
        assert receipt["slots"][slot]["terminal_time"] == 5.0
        assert receipt["slots"][slot]["manuscript_eligible"] is True
        assert receipt["slots"][slot]["inputs"]["completion_receipt"][
            "source_tree_files_sha256"
        ] == "d" * 64
    assert "validated stitched $T=5$ manuscript evidence" in latex


Q_COMPILED_ROUTE = "paper_ii_exchange_q_compiled_v1"


def test_pre_route_promotion_declares_the_historical_unrecorded_route(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    receipt, latex = _build(paths)

    assert receipt["route"] == SIAM_ROUTE_HISTORICAL_UNRECORDED
    assert (
        f"Exchange route: {SIAM_ROUTE_HISTORICAL_UNRECORDED}." in latex
    )
    assert set(receipt["slots"]) == {"avqds", "full_apm"}


def test_compiled_q_promotion_declares_its_route_without_new_columns(
    tmp_path: Path,
) -> None:
    paths = _fixture(
        tmp_path,
        apm_score_schema=SIAM_OFFLINE_SCORE_V4,
        apm_exchange_route_bundle=Q_COMPILED_ROUTE,
    )
    receipt, latex = _build(paths)

    assert receipt["route"] == Q_COMPILED_ROUTE
    assert f"Exchange route: {Q_COMPILED_ROUTE}." in latex
    # The route is a header field only: the two promotion slots and the table's
    # column count are unchanged.
    assert set(receipt["slots"]) == {"avqds", "full_apm"}
    assert r"\begin{tabular}{llcccccc}" in latex

    pdf = render_siam_results_pdf(receipt)
    assert pdf.startswith(b"%PDF")
