from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.noise_oracle_runtime import OracleConfig
from pipelines.hardcoded.hh_fixed_manifold_mclachlan import (
    FixedManifoldRunSpec,
    load_run_context,
)
from pipelines.hardcoded.hh_fixed_manifold_measured import (
    FixedManifoldAugmentationConfig,
    FixedManifoldDriveConfig,
    FixedManifoldMeasuredConfig,
    _augment_loaded_context_with_drive_generator,
    assemble_measured_geometry,
    run_fixed_manifold_measured,
)
from pipelines.hardcoded.hh_fixed_manifold_observables import (
    build_checkpoint_observable_plan,
    build_checkpoint_observable_plan_from_layout,
    build_heisenberg_generator,
    flatten_runtime_rotations,
)
from pipelines.hardcoded.hh_vqe_from_adapt_family import ReplayScaffoldContext
from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import apply_compiled_polynomial, compile_polynomial_action
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, PauliPolynomial, PauliTerm, expval_pauli_polynomial


def _basis(nq: int, idx: int) -> np.ndarray:
    out = np.zeros(1 << int(nq), dtype=complex)
    out[int(idx)] = 1.0
    return out


def _toy_runtime_context(
    theta: np.ndarray | None = None,
) -> tuple[ReplayScaffoldContext, CompiledAnsatzExecutor, Any]:
    x_term = AnsatzTerm(
        label="op_x",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="x", pc=1.0)]),
    )
    y_term = AnsatzTerm(
        label="op_y",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="y", pc=1.0)]),
    )
    h_poly = PauliPolynomial("JW", [PauliTerm(1, ps="z", pc=1.0)])
    layout = build_parameter_layout(
        [x_term, y_term],
        ignore_identity=True,
        coefficient_tolerance=1.0e-12,
        sort_terms=True,
    )
    executor = CompiledAnsatzExecutor(
        [x_term, y_term],
        coefficient_tolerance=1.0e-12,
        parameterization_mode="per_pauli_term",
        parameterization_layout=layout,
    )
    theta_runtime = (
        np.asarray([0.2, -0.3], dtype=float)
        if theta is None
        else np.asarray(theta, dtype=float).reshape(-1)
    )
    psi_ref = _basis(1, 0)
    replay_context = ReplayScaffoldContext(
        cfg=SimpleNamespace(reps=1),
        h_poly=h_poly,
        psi_ref=psi_ref,
        payload_in={"ansatz_input_state": {"source": "hf"}},
        family_info={"resolved": "toy_fixed"},
        family_pool=(x_term, y_term),
        pool_meta={"candidate_pool_complete": True, "fixed_manifold_locked": True},
        replay_terms=(x_term, y_term),
        base_layout=layout,
        adapt_theta_runtime=np.array(theta_runtime, copy=True),
        adapt_theta_logical=np.array(theta_runtime, copy=True),
        adapt_depth=2,
        handoff_state_kind="prepared_state",
        provenance_source="explicit",
        family_terms_count=2,
    )
    return replay_context, executor, h_poly


def _toy_fixed_scaffold_payload() -> dict:
    return {
        "pipeline": "toy_fixed_scaffold_export",
        "settings": {
            "L": 1,
            "problem": "hh",
            "t": 1.0,
            "u": 4.0,
            "dv": 0.0,
            "omega0": 1.0,
            "g_ep": 0.5,
            "n_ph_max": 1,
            "boson_encoding": "binary",
            "ordering": "blocked",
            "boundary": "open",
            "adapt_pool": "fixed_scaffold_locked",
        },
        "adapt_vqe": {
            "success": True,
            "pool_type": "fixed_scaffold_locked",
            "method": "toy_fixed_scaffold",
            "num_particles": {"n_up": 1, "n_dn": 0},
            "operators": ["toy_x"],
            "optimal_point": [0.2],
            "logical_optimal_point": [0.2],
            "parameterization": {
                "mode": "per_pauli_term_v1",
                "term_order": "native",
                "ignore_identity": True,
                "coefficient_tolerance": 1.0e-12,
                "logical_operator_count": 1,
                "runtime_parameter_count": 1,
                "blocks": [
                    {
                        "candidate_label": "toy_x",
                        "logical_index": 0,
                        "runtime_start": 0,
                        "runtime_count": 1,
                        "runtime_terms_exyz": [
                            {
                                "pauli_exyz": "eex",
                                "coeff_re": 1.0,
                                "coeff_im": 0.0,
                                "nq": 3,
                            }
                        ],
                    }
                ],
            },
            "structure_locked": True,
            "fixed_scaffold_kind": "toy_locked_v1",
            "fixed_scaffold_metadata": {
                "route_family": "locked_imported_scaffold_v1",
                "subject_kind": "toy_locked_v1",
                "source_artifact_json": "toy_source.json",
            },
        },
        "ansatz_input_state": {
            "source": "hf",
            "nq_total": 3,
            "amplitudes_qn_to_q0": {"001": {"re": 1.0, "im": 0.0}},
            "handoff_state_kind": "reference_state",
        },
        "initial_state": {
            "source": "fixed_scaffold_vqe",
            "nq_total": 3,
            "amplitudes_qn_to_q0": {
                "001": {"re": np.cos(0.2), "im": 0.0},
                "000": {"re": 0.0, "im": -np.sin(0.2)},
            },
            "handoff_state_kind": "prepared_state",
        },
    }


def test_heisenberg_generator_matches_exact_centered_runtime_tangent() -> None:
    replay_context, executor, _h_poly = _toy_runtime_context()
    theta = np.asarray(replay_context.adapt_theta_runtime, dtype=float)
    layout = replay_context.base_layout
    psi = executor.prepare_state(theta, replay_context.psi_ref)
    _psi_full, raw_tangents = executor.prepare_state_with_runtime_tangents(
        theta,
        replay_context.psi_ref,
        runtime_indices=(0,),
    )
    raw_tangent = np.asarray(raw_tangents[0], dtype=complex).reshape(-1)
    exact_centered = raw_tangent - complex(np.vdot(psi, raw_tangent)) * psi

    rotations = flatten_runtime_rotations(layout)
    A0 = build_heisenberg_generator(
        rotations,
        theta,
        0,
        drop_abs_tol=1.0e-12,
        hermiticity_tol=1.0e-10,
    )
    Apsi = apply_compiled_polynomial(psi, compile_polynomial_action(A0, tol=1.0e-12))
    meanA = float(np.real(np.vdot(psi, Apsi)))
    reconstructed = -1.0j * float(rotations[0].coeff_real) * (Apsi - meanA * psi)

    assert np.linalg.norm(exact_centered - reconstructed) < 1.0e-10


def test_checkpoint_observable_plan_wrapper_matches_layout_builder_on_toy_case() -> None:
    replay_context, _executor, h_poly = _toy_runtime_context()
    theta = np.asarray(replay_context.adapt_theta_runtime, dtype=float)
    wrapped = build_checkpoint_observable_plan(
        replay_context,
        theta,
        h_poly=h_poly,
        drop_abs_tol=1.0e-12,
        hermiticity_tol=1.0e-10,
        max_observable_terms=128,
    )
    direct = build_checkpoint_observable_plan_from_layout(
        replay_context.base_layout,
        theta,
        psi_ref=replay_context.psi_ref,
        h_poly=h_poly,
        drop_abs_tol=1.0e-12,
        hermiticity_tol=1.0e-10,
        max_observable_terms=128,
    )

    assert wrapped.stats == direct.stats
    assert wrapped.energy.term_count == direct.energy.term_count
    assert wrapped.variance_h2.term_count == direct.variance_h2.term_count
    assert [spec.term_count for spec in wrapped.generator_means] == [spec.term_count for spec in direct.generator_means]
    assert {
        tuple(pair): int(spec.term_count) for pair, spec in wrapped.pair_anticommutators.items()
    } == {
        tuple(pair): int(spec.term_count) for pair, spec in direct.pair_anticommutators.items()
    }
    assert [spec.term_count for spec in wrapped.force_anticommutators] == [spec.term_count for spec in direct.force_anticommutators]


def test_measured_geometry_matches_exact_baseline_on_toy_case() -> None:
    replay_context, executor, h_poly = _toy_runtime_context()
    theta = np.asarray(replay_context.adapt_theta_runtime, dtype=float)
    plan = build_checkpoint_observable_plan(
        replay_context,
        theta,
        h_poly=h_poly,
        drop_abs_tol=1.0e-12,
        hermiticity_tol=1.0e-10,
        max_observable_terms=128,
    )
    psi = executor.prepare_state(theta, replay_context.psi_ref)
    energy = float(expval_pauli_polynomial(psi, plan.energy.poly))
    h2 = float(expval_pauli_polynomial(psi, plan.variance_h2.poly))
    generator_means = [float(expval_pauli_polynomial(psi, spec.poly)) for spec in plan.generator_means]
    pair_expectations = {
        pair: (0.0 if spec.is_zero else float(expval_pauli_polynomial(psi, spec.poly)))
        for pair, spec in plan.pair_anticommutators.items()
    }
    force_expectations = [
        (0.0 if spec.is_zero else float(expval_pauli_polynomial(psi, spec.poly)))
        for spec in plan.force_anticommutators
    ]
    geom = assemble_measured_geometry(
        plan=plan,
        energy=energy,
        h2=h2,
        generator_means=generator_means,
        pair_expectations=pair_expectations,
        force_expectations=force_expectations,
        geom_cfg=FixedManifoldMeasuredConfig(),
    )

    _psi_full, raw_tangents = executor.prepare_state_with_runtime_tangents(
        theta,
        replay_context.psi_ref,
        runtime_indices=tuple(range(int(plan.runtime_rotations.__len__()))),
    )
    centered_cols = []
    for idx in range(len(plan.runtime_rotations)):
        tangent = np.asarray(raw_tangents[idx], dtype=complex).reshape(-1)
        centered_cols.append(tangent - complex(np.vdot(psi, tangent)) * psi)
    T = np.column_stack(centered_cols)
    hpsi = apply_compiled_polynomial(psi, compile_polynomial_action(h_poly, tol=1.0e-12))
    b_bar = -1.0j * (hpsi - energy * psi)
    G_exact = np.asarray(np.real(T.conj().T @ T), dtype=float)
    f_exact = np.asarray(np.real(T.conj().T @ b_bar), dtype=float).reshape(-1)
    K_exact = np.asarray(
        G_exact + 1.0e-8 * np.eye(int(G_exact.shape[0]), dtype=float),
        dtype=float,
    )
    theta_dot_exact = np.asarray(np.linalg.pinv(K_exact, rcond=1.0e-10) @ f_exact, dtype=float)
    variance_exact = float(max(0.0, np.real(np.vdot(hpsi, hpsi)) - energy * energy))
    eps_proj_exact = float(
        max(0.0, variance_exact - float(f_exact @ (np.linalg.pinv(G_exact, rcond=1.0e-10) @ f_exact)))
    )
    rho_exact = float(eps_proj_exact / max(variance_exact, 1.0e-14))

    assert np.max(np.abs(np.asarray(geom["G"]) - G_exact)) < 1.0e-10
    assert np.max(np.abs(np.asarray(geom["f"]) - f_exact)) < 1.0e-10
    assert np.max(np.abs(np.asarray(geom["theta_dot_step"]) - theta_dot_exact)) < 1.0e-10
    assert abs(float(geom["rho_miss"]) - rho_exact) < 1.0e-10


def test_drive_changes_force_but_not_metric_on_toy_case() -> None:
    replay_context, executor, h_poly_static = _toy_runtime_context()
    theta = np.asarray(replay_context.adapt_theta_runtime, dtype=float)
    psi = executor.prepare_state(theta, replay_context.psi_ref)
    h_poly_driven = PauliPolynomial(
        "JW",
        [
            PauliTerm(1, ps="z", pc=1.0),
            PauliTerm(1, ps="x", pc=0.35),
        ],
    )

    def _measured(plan):
        energy = float(expval_pauli_polynomial(psi, plan.energy.poly))
        h2 = float(expval_pauli_polynomial(psi, plan.variance_h2.poly))
        generator_means = [float(expval_pauli_polynomial(psi, spec.poly)) for spec in plan.generator_means]
        pair_expectations = {
            pair: (0.0 if spec.is_zero else float(expval_pauli_polynomial(psi, spec.poly)))
            for pair, spec in plan.pair_anticommutators.items()
        }
        force_expectations = [
            (0.0 if spec.is_zero else float(expval_pauli_polynomial(psi, spec.poly)))
            for spec in plan.force_anticommutators
        ]
        return assemble_measured_geometry(
            plan=plan,
            energy=energy,
            h2=h2,
            generator_means=generator_means,
            pair_expectations=pair_expectations,
            force_expectations=force_expectations,
            geom_cfg=FixedManifoldMeasuredConfig(),
        )

    static_plan = build_checkpoint_observable_plan(
        replay_context,
        theta,
        h_poly=h_poly_static,
        drop_abs_tol=1.0e-12,
        hermiticity_tol=1.0e-10,
        max_observable_terms=128,
    )
    driven_plan = build_checkpoint_observable_plan(
        replay_context,
        theta,
        h_poly=h_poly_driven,
        drop_abs_tol=1.0e-12,
        hermiticity_tol=1.0e-10,
        max_observable_terms=128,
    )
    geom_static = _measured(static_plan)
    geom_driven = _measured(driven_plan)

    assert static_plan.energy.term_count == 1
    assert driven_plan.energy.term_count == 2
    assert [spec.term_count for spec in static_plan.generator_means] == [spec.term_count for spec in driven_plan.generator_means]
    assert {
        tuple(pair): int(spec.term_count) for pair, spec in static_plan.pair_anticommutators.items()
    } == {
        tuple(pair): int(spec.term_count) for pair, spec in driven_plan.pair_anticommutators.items()
    }
    assert np.allclose(np.asarray(geom_static["G"], dtype=float), np.asarray(geom_driven["G"], dtype=float), atol=1.0e-10)
    assert not np.allclose(np.asarray(geom_static["f"], dtype=float), np.asarray(geom_driven["f"], dtype=float), atol=1.0e-8)


def test_pareto_lean_l2_checkpoint0_observable_plan_within_cap() -> None:
    loaded = load_run_context(
        FixedManifoldRunSpec(
            name="pareto_lean_l2",
            artifact_json=Path(
                "artifacts/json/adapt_hh_L2_ecut1_pareto_lean_l2_phase3_powell_rerun_with_ansatz_input_20260321T214822Z.json"
            ),
            loader_mode="replay_family",
            generator_family="match_adapt",
            fallback_family="full_meta",
        ),
        tag="pytest_pareto_observables",
    )
    plan = build_checkpoint_observable_plan(
        loaded.replay_context,
        loaded.replay_context.adapt_theta_runtime,
        drop_abs_tol=1.0e-12,
        hermiticity_tol=1.0e-10,
        max_observable_terms=512,
    )

    assert int(plan.stats["runtime_parameter_count"]) == 25
    assert int(plan.stats["max_observable_terms_any"]) <= 512
    assert int(plan.stats["generator_mean_count"]) == 25
    assert int(plan.stats["pair_anticommutator_count"]) == 300
    assert int(plan.stats["force_anticommutator_count"]) == 25


def test_locked_7term_checkpoint0_observable_plan_within_cap() -> None:
    loaded = load_run_context(
        FixedManifoldRunSpec(
            name="locked_7term",
            artifact_json=Path("artifacts/json/hh_prune_nighthawk_gate_pruned_7term.json"),
            loader_mode="fixed_scaffold",
            generator_family="fixed_scaffold_locked",
            fallback_family="full_meta",
        ),
        tag="pytest_7term_observables",
    )
    plan = build_checkpoint_observable_plan(
        loaded.replay_context,
        loaded.replay_context.adapt_theta_runtime,
        drop_abs_tol=1.0e-12,
        hermiticity_tol=1.0e-10,
        max_observable_terms=4096,
    )

    assert int(plan.stats["runtime_parameter_count"]) == 7
    assert int(plan.stats["max_observable_terms_any"]) <= 4096
    assert int(plan.stats["generator_mean_count"]) == 7
    assert int(plan.stats["pair_anticommutator_count"]) == 21
    assert int(plan.stats["force_anticommutator_count"]) == 7


def test_drive_generator_augmentation_preserves_prepared_state_and_expands_layout() -> None:
    loaded = load_run_context(
        FixedManifoldRunSpec(
            name="locked_7term",
            artifact_json=Path("artifacts/json/hh_prune_nighthawk_gate_pruned_7term.json"),
            loader_mode="fixed_scaffold",
            generator_family="fixed_scaffold_locked",
            fallback_family="full_meta",
        ),
        tag="pytest_drive_aug",
    )
    augmented = _augment_loaded_context_with_drive_generator(
        loaded,
        drive_cfg=FixedManifoldDriveConfig(
            enable_drive=True,
            drive_A=0.6,
            drive_omega=1.0,
            drive_tbar=1.0,
            drive_phi=0.0,
            drive_pattern="staggered",
            drive_custom_s=None,
            drive_include_identity=False,
            drive_time_sampling="midpoint",
            drive_t0=0.0,
            exact_steps_multiplier=2,
        ),
        aug_cfg=FixedManifoldAugmentationConfig(drive_generator_mode="aligned_density"),
    )

    assert int(augmented.replay_context.base_layout.logical_parameter_count) == 6
    assert int(augmented.replay_context.base_layout.runtime_parameter_count) == 11
    assert tuple(augmented.replay_context.base_layout.blocks[: len(loaded.replay_context.base_layout.blocks)]) == tuple(
        loaded.replay_context.base_layout.blocks
    )
    assert str(augmented.loader_summary.get("drive_generator_mode")) == "aligned_density"
    assert str(augmented.loader_summary.get("family_pool_origin")) == "replay_terms_plus_drive_augmented"
    assert int(augmented.loader_summary.get("augmentation_runtime_parameter_delta")) == 4
    assert np.linalg.norm(np.asarray(augmented.psi_initial) - np.asarray(loaded.psi_initial)) < 1.0e-12


def test_measured_runner_rejects_nonideal_or_repeat_configs(tmp_path: Path) -> None:
    payload = _toy_fixed_scaffold_payload()
    artifact_json = tmp_path / "toy_config_gate.json"
    artifact_json.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="noise_mode='ideal'"):
        run_fixed_manifold_measured(
            FixedManifoldRunSpec(
                name="toy_locked",
                artifact_json=artifact_json,
                loader_mode="fixed_scaffold",
            ),
            tag="pytest_nonideal",
            output_json=tmp_path / "nonideal.json",
            t_final=0.1,
            num_times=2,
            oracle_cfg=OracleConfig(
                noise_mode="shots",
                shots=1024,
                seed=7,
                oracle_repeats=1,
                oracle_aggregate="mean",
            ),
            geom_cfg=FixedManifoldMeasuredConfig(),
        )

    with pytest.raises(ValueError, match="oracle_repeats=1"):
        run_fixed_manifold_measured(
            FixedManifoldRunSpec(
                name="toy_locked",
                artifact_json=artifact_json,
                loader_mode="fixed_scaffold",
            ),
            tag="pytest_repeat",
            output_json=tmp_path / "repeat.json",
            t_final=0.1,
            num_times=2,
            oracle_cfg=OracleConfig(
                noise_mode="ideal",
                shots=1024,
                seed=7,
                oracle_repeats=2,
                oracle_aggregate="mean",
            ),
            geom_cfg=FixedManifoldMeasuredConfig(),
        )


def test_measured_runner_rejects_non_basis_reference_state(tmp_path: Path) -> None:
    payload = _toy_fixed_scaffold_payload()
    payload["ansatz_input_state"]["amplitudes_qn_to_q0"] = {
        "001": {"re": 1.0 / np.sqrt(2.0), "im": 0.0},
        "000": {"re": 1.0 / np.sqrt(2.0), "im": 0.0},
    }
    phase = np.exp(-1.0j * 0.2)
    payload["initial_state"]["amplitudes_qn_to_q0"] = {
        "001": {"re": float(np.real(phase / np.sqrt(2.0))), "im": float(np.imag(phase / np.sqrt(2.0)))},
        "000": {"re": float(np.real(phase / np.sqrt(2.0))), "im": float(np.imag(phase / np.sqrt(2.0)))},
    }
    artifact_json = tmp_path / "toy_nonbasis.json"
    artifact_json.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="one-hot computational-basis HF reference state"):
        run_fixed_manifold_measured(
            FixedManifoldRunSpec(
                name="toy_locked",
                artifact_json=artifact_json,
                loader_mode="fixed_scaffold",
            ),
            tag="pytest_nonbasis",
            output_json=tmp_path / "out.json",
            t_final=0.1,
            num_times=2,
            oracle_cfg=OracleConfig(
                noise_mode="ideal",
                shots=1024,
                seed=7,
                oracle_repeats=1,
                oracle_aggregate="mean",
            ),
            geom_cfg=FixedManifoldMeasuredConfig(),
        )


def test_measured_runner_augmented_payload_reports_augmented_pool(tmp_path: Path) -> None:
    payload = _toy_fixed_scaffold_payload()
    artifact_json = tmp_path / "toy_fixed_aug.json"
    artifact_json.write_text(json.dumps(payload), encoding="utf-8")

    result = run_fixed_manifold_measured(
        FixedManifoldRunSpec(
            name="toy_locked",
            artifact_json=artifact_json,
            loader_mode="fixed_scaffold",
        ),
        tag="pytest_augmented_payload",
        output_json=tmp_path / "toy_augmented.json",
        t_final=0.1,
        num_times=2,
        oracle_cfg=OracleConfig(
            noise_mode="ideal",
            shots=1024,
            seed=7,
            oracle_repeats=1,
            oracle_aggregate="mean",
        ),
        geom_cfg=FixedManifoldMeasuredConfig(observable_max_terms=128),
        drive_cfg=FixedManifoldDriveConfig(
            enable_drive=True,
            drive_A=0.6,
            drive_omega=1.0,
            drive_tbar=1.0,
            drive_phi=0.0,
            drive_pattern="staggered",
            drive_custom_s=None,
            drive_include_identity=False,
            drive_time_sampling="midpoint",
            drive_t0=0.0,
            exact_steps_multiplier=1,
        ),
        aug_cfg=FixedManifoldAugmentationConfig(drive_generator_mode="aligned_density"),
    )

    assert result["run_name"] == "toy_locked_aligned_density"
    assert result["loader"]["family_pool_origin"] == "replay_terms_plus_drive_augmented"
    assert result["loader"]["drive_generator_mode"] == "aligned_density"
    assert result["manifest"]["effective_pool_kind"] == "replay_terms_plus_drive_augmented"
    assert result["manifest"]["drive_generator_mode"] == "aligned_density"


def test_measured_runner_replay_family_real_artifact_smoke(tmp_path: Path) -> None:
    result = run_fixed_manifold_measured(
        FixedManifoldRunSpec(
            name="pareto_lean_l2",
            artifact_json=Path(
                "artifacts/json/adapt_hh_L2_ecut1_pareto_lean_l2_phase3_powell_rerun_with_ansatz_input_20260321T214822Z.json"
            ),
            loader_mode="replay_family",
            generator_family="match_adapt",
            fallback_family="full_meta",
        ),
        tag="pytest_pareto_smoke",
        output_json=tmp_path / "pareto_smoke.json",
        t_final=0.1,
        num_times=2,
        oracle_cfg=OracleConfig(
            noise_mode="ideal",
            shots=1024,
            seed=7,
            oracle_repeats=1,
            oracle_aggregate="mean",
        ),
        geom_cfg=FixedManifoldMeasuredConfig(observable_max_terms=512),
    )

    assert result["run_name"] == "pareto_lean_l2"
    assert result["loader"]["loader_mode"] == "replay_family"
    assert result["loader"]["family_pool_origin"] == "replay_terms_only"
    assert result["summary"]["runtime_parameter_count"] == 25
    assert len(result["trajectory"]) == 2


def test_measured_runner_drive_a0_matches_static_on_toy_case(tmp_path: Path) -> None:
    payload = _toy_fixed_scaffold_payload()
    artifact_json = tmp_path / "toy_fixed_a0.json"
    artifact_json.write_text(json.dumps(payload), encoding="utf-8")

    static_result = run_fixed_manifold_measured(
        FixedManifoldRunSpec(
            name="toy_locked",
            artifact_json=artifact_json,
            loader_mode="fixed_scaffold",
        ),
        tag="pytest_static_a0",
        output_json=tmp_path / "static.json",
        t_final=0.2,
        num_times=3,
        oracle_cfg=OracleConfig(
            noise_mode="ideal",
            shots=1024,
            seed=7,
            oracle_repeats=1,
            oracle_aggregate="mean",
        ),
        geom_cfg=FixedManifoldMeasuredConfig(observable_max_terms=128),
    )
    drive_result = run_fixed_manifold_measured(
        FixedManifoldRunSpec(
            name="toy_locked",
            artifact_json=artifact_json,
            loader_mode="fixed_scaffold",
        ),
        tag="pytest_drive_a0",
        output_json=tmp_path / "drive_a0.json",
        t_final=0.2,
        num_times=3,
        oracle_cfg=OracleConfig(
            noise_mode="ideal",
            shots=1024,
            seed=7,
            oracle_repeats=1,
            oracle_aggregate="mean",
        ),
        geom_cfg=FixedManifoldMeasuredConfig(observable_max_terms=128),
        drive_cfg=FixedManifoldDriveConfig(
            enable_drive=True,
            drive_A=0.0,
            drive_omega=1.1,
            drive_tbar=0.8,
            drive_phi=0.2,
            exact_steps_multiplier=2,
        ),
    )

    for row_static, row_drive in zip(static_result["trajectory"], drive_result["trajectory"]):
        assert abs(float(row_static["geometry"]["energy"]) - float(row_drive["geometry"]["energy"])) < 1.0e-10
        assert abs(float(row_static["geometry"]["rho_miss"]) - float(row_drive["geometry"]["rho_miss"])) < 1.0e-10
        assert abs(float(row_static["geometry"]["condition_number"]) - float(row_drive["geometry"]["condition_number"])) < 1.0e-10
        assert abs(float(row_static["geometry"]["theta_dot_l2"]) - float(row_drive["geometry"]["theta_dot_l2"])) < 1.0e-10
        assert abs(float(row_static["audit"]["fidelity_exact_audit"]) - float(row_drive["audit"]["fidelity_exact_audit"])) < 1.0e-10
        assert abs(float(row_static["audit"]["abs_energy_total_error_exact_audit"]) - float(row_drive["audit"]["abs_energy_total_error_exact_audit"])) < 1.0e-10


def test_measured_runner_toy_drive_schema(tmp_path: Path) -> None:
    payload = _toy_fixed_scaffold_payload()
    artifact_json = tmp_path / "toy_fixed_drive.json"
    artifact_json.write_text(json.dumps(payload), encoding="utf-8")
    out_json = tmp_path / "measured_drive.json"

    result = run_fixed_manifold_measured(
        FixedManifoldRunSpec(
            name="toy_locked",
            artifact_json=artifact_json,
            loader_mode="fixed_scaffold",
        ),
        tag="pytest_measured_drive",
        output_json=out_json,
        t_final=0.2,
        num_times=3,
        oracle_cfg=OracleConfig(
            noise_mode="ideal",
            shots=1024,
            seed=7,
            oracle_repeats=1,
            oracle_aggregate="mean",
        ),
        geom_cfg=FixedManifoldMeasuredConfig(observable_max_terms=128),
        drive_cfg=FixedManifoldDriveConfig(
            enable_drive=True,
            drive_A=0.25,
            drive_omega=1.1,
            drive_tbar=0.8,
            drive_phi=0.5,
            drive_t0=0.3,
            exact_steps_multiplier=2,
        ),
    )

    assert result["manifest"]["drive_enabled"] is True
    assert result["manifest"]["reference_audit_backend"] == "dense_piecewise_constant_reference"
    assert result["manifest"]["reference_audit_method"] == "exponential_midpoint_magnus2_order2"
    assert result["reference_config"]["method"] == "exponential_midpoint_magnus2_order2"
    assert result["reference_config"]["reference_steps_multiplier"] == 2
    assert result["reference_config"]["reference_steps"] == 4
    assert result["reference_config"]["geometry_sample_time_policy"] == "interval_midpoint_plus_t0_with_final_endpoint_fallback"
    assert result["projection_config"]["integrator"] == "explicit_euler"
    assert result["projection_config"]["time_sampling"] == "midpoint"
    assert result["projection_config"]["geometry_sample_time_policy"] == "interval_midpoint_plus_t0_with_final_endpoint_fallback"
    assert result["drive_profile"]["A"] == pytest.approx(0.25)
    assert result["trajectory"][0]["physical_time"] == pytest.approx(0.35)
    assert result["trajectory"][1]["physical_time"] == pytest.approx(0.45)
    assert int(result["trajectory"][0]["geometry"]["drive_term_count"]) > 0


def test_measured_runner_toy_end_to_end_schema(tmp_path: Path) -> None:
    payload = _toy_fixed_scaffold_payload()
    artifact_json = tmp_path / "toy_fixed.json"
    artifact_json.write_text(json.dumps(payload), encoding="utf-8")
    out_json = tmp_path / "measured.json"

    result = run_fixed_manifold_measured(
        FixedManifoldRunSpec(
            name="toy_locked",
            artifact_json=artifact_json,
            loader_mode="fixed_scaffold",
        ),
        tag="pytest_measured",
        output_json=out_json,
        t_final=0.2,
        num_times=3,
        oracle_cfg=OracleConfig(
            noise_mode="ideal",
            shots=1024,
            seed=7,
            oracle_repeats=1,
            oracle_aggregate="mean",
        ),
        geom_cfg=FixedManifoldMeasuredConfig(observable_max_terms=128),
    )

    assert result["pipeline"] == "hh_fixed_manifold_measured_mclachlan_v1"
    assert result["manifest"]["geometry_backend"] == "oracle"
    assert result["manifest"]["structure_policy"] == "fixed_manifold_locked_pool"
    assert len(result["trajectory"]) == 3
    assert "decision_backend" not in result["summary"]
    assert "min_fidelity_exact_audit" in result["summary"]
    assert "max_abs_energy_total_error_exact_audit" in result["summary"]
    assert "max_rho_miss" in result["summary"]
    assert "max_condition_number" in result["summary"]
    assert "max_theta_dot_l2" in result["summary"]
    assert result["summary"]["max_rho_miss"] >= result["summary"]["final_rho_miss"]
