from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.time_dynamics.legacy.experiments.checkpoint_local_adapt import (
    CheckpointLocalAdaptConfig,
    _fidelity_gradient_components,
    _objective_payload,
    adapt_checkpoint_snapshot,
    adapt_checkpoint_snapshot_with_state,
    available_candidate_terms,
    resolve_candidate_pool_terms,
    run_checkpoint_local_adapt_from_args,
)
from pipelines.time_dynamics.fixed_manifold.exact_fit import FrozenScaffoldExactFitConfig
from pipelines.time_dynamics.legacy.checkpoint_controller import (
    _build_candidate_carrier,
    _layout_from_carriers,
    _site_resolved_number_observables,
)
from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, PauliPolynomial, PauliTerm, hamiltonian_matrix


def _basis(idx: int) -> np.ndarray:
    out = np.zeros(2, dtype=complex)
    out[int(idx)] = 1.0
    return out


def _toy_terms() -> tuple[AnsatzTerm, AnsatzTerm]:
    x_term = AnsatzTerm(
        label="op_x",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="x", pc=1.0)]),
    )
    y_term = AnsatzTerm(
        label="op_y",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="y", pc=1.0)]),
    )
    return x_term, y_term


def _toy_snapshot_and_bundle() -> tuple[dict[str, object], dict[str, object]]:
    x_term, y_term = _toy_terms()
    h_poly = PauliPolynomial("JW", [PauliTerm(1, ps="z", pc=1.0)])
    hmat = np.asarray(hamiltonian_matrix(h_poly), dtype=complex)
    psi_ref = _basis(0)

    base_layout = build_parameter_layout([x_term], ignore_identity=True, coefficient_tolerance=1.0e-12, sort_terms=True)
    x_carrier = _build_candidate_carrier(
        x_term,
        logical_index=0,
        unique_label="op_x__r0",
        template_layout=base_layout,
        candidate_pool_index=0,
    )
    layout = _layout_from_carriers([x_carrier], template=base_layout)
    executor = CompiledAnsatzExecutor(
        [AnsatzTerm(label=str(x_carrier.label), polynomial=x_carrier.polynomial)],
        coefficient_tolerance=float(layout.coefficient_tolerance),
        ignore_identity=bool(layout.ignore_identity),
        sort_terms=(str(layout.term_order).strip().lower() == "sorted"),
        parameterization_mode="per_pauli_term",
        parameterization_layout=layout,
    )
    theta_current = np.asarray([0.2], dtype=float)
    psi_current = np.asarray(executor.prepare_state(theta_current, psi_ref), dtype=complex).reshape(-1)

    xy_layout = build_parameter_layout([x_term, y_term], ignore_identity=True, coefficient_tolerance=1.0e-12, sort_terms=True)
    xy_executor = CompiledAnsatzExecutor(
        [x_term, y_term],
        coefficient_tolerance=float(xy_layout.coefficient_tolerance),
        ignore_identity=bool(xy_layout.ignore_identity),
        sort_terms=(str(xy_layout.term_order).strip().lower() == "sorted"),
        parameterization_mode="per_pauli_term",
        parameterization_layout=xy_layout,
    )
    psi_exact = np.asarray(xy_executor.prepare_state(np.asarray([0.2, 0.35], dtype=float), psi_ref), dtype=complex).reshape(-1)
    exact_raw = _site_resolved_number_observables(psi_exact, num_sites=1, ordering="blocked")
    current_raw = _site_resolved_number_observables(psi_current, num_sites=1, ordering="blocked")

    snapshot = {
        "checkpoint_index": 7,
        "time": 1.4,
        "time_stop": 1.5,
        "physical_time": 1.45,
        "hmat_step": hmat,
        "drive_term_count": 0,
        "terms": [x_carrier],
        "layout": layout,
        "executor": executor,
        "theta_runtime": theta_current,
        "psi_ref": psi_ref,
        "psi_current": psi_current,
        "psi_exact": psi_exact,
        "current_observables": {
            "site_occupations": [float(x) for x in np.asarray(current_raw.n_site, dtype=float).tolist()],
            "doublon": float(current_raw.doublon),
            "staggered": float(current_raw.staggered),
        },
        "exact_observables": {
            "site_occupations": [float(x) for x in np.asarray(exact_raw.n_site, dtype=float).tolist()],
            "doublon": float(exact_raw.doublon),
            "staggered": float(exact_raw.staggered),
        },
        "energy_exact": float(np.real(np.vdot(psi_exact, hmat @ psi_exact))),
        "scaffold_labels": [str(x_carrier.label)],
        "logical_block_count": 1,
        "runtime_parameter_count": 1,
        "num_sites": 1,
        "ordering": "blocked",
        "reference_energy_total_span_full_run": 0.5,
    }
    replay_cfg = SimpleNamespace(
        L=1,
        t=1.0,
        u=0.0,
        dv=0.0,
        omega0=1.0,
        g_ep=0.0,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        paop_r=1,
        paop_split_paulis=False,
        paop_prune_eps=1.0e-12,
        paop_normalization="none",
        sector_n_up=1,
        sector_n_dn=0,
    )
    replay_context = SimpleNamespace(
        family_pool=(x_term, y_term),
        pool_meta={"candidate_pool_complete": True},
        family_info={"resolved": "toy_family"},
        h_poly=h_poly,
        cfg=replay_cfg,
    )
    bundle = {"loaded": SimpleNamespace(replay_context=replay_context)}
    return snapshot, bundle


def test_fidelity_gradient_components_detect_helpful_appended_direction() -> None:
    x_term, y_term = _toy_terms()
    psi_ref = _basis(0)
    base_layout = build_parameter_layout([x_term], ignore_identity=True, coefficient_tolerance=1.0e-12, sort_terms=True)
    executor = CompiledAnsatzExecutor(
        [x_term, y_term],
        coefficient_tolerance=float(base_layout.coefficient_tolerance),
        ignore_identity=bool(base_layout.ignore_identity),
        sort_terms=True,
        parameterization_mode="per_pauli_term",
        parameterization_layout=build_parameter_layout([x_term, y_term], ignore_identity=True, coefficient_tolerance=1.0e-12, sort_terms=True),
    )
    psi_exact = np.asarray(executor.prepare_state(np.asarray([0.2, 0.35], dtype=float), psi_ref), dtype=complex).reshape(-1)

    grads = _fidelity_gradient_components(
        aug_executor=executor,
        theta_aug=np.asarray([0.2, 0.0], dtype=float),
        psi_ref=psi_ref,
        psi_exact=psi_exact,
        runtime_indices=(1,),
    )

    assert len(grads) == 1
    assert abs(float(grads[0])) > 1.0e-6


def test_available_candidate_terms_drops_current_signature() -> None:
    snapshot, bundle = _toy_snapshot_and_bundle()

    pool_terms, _meta = resolve_candidate_pool_terms(bundle, pool_mode="family_pool")
    available = available_candidate_terms(snapshot["terms"], pool_terms)

    assert [term.label for _, term in available] == ["op_y"]


def test_adapt_checkpoint_snapshot_adds_operator_and_recovers_high_fidelity() -> None:
    snapshot, bundle = _toy_snapshot_and_bundle()
    adapt_cfg = CheckpointLocalAdaptConfig(
        objective="fidelity_first",
        pool_mode="family_pool",
        target_fidelity=0.999,
        max_steps=2,
        gradient_threshold=1.0e-8,
        probe_scale=0.2,
        min_fidelity_gain=1.0e-6,
        plateau_patience=1,
        candidate_rank_limit=4,
    )
    fit_cfg = FrozenScaffoldExactFitConfig(
        objectives=("fidelity_first",),
        method="Powell",
        maxiter=120,
        restarts=3,
        seed=5,
        initial_sigma=0.05,
    )

    payload = adapt_checkpoint_snapshot(snapshot, bundle=bundle, adapt_cfg=adapt_cfg, fit_cfg=fit_cfg)

    assert payload["operators_added"] >= 1
    assert payload["history"][0]["selected_label"] == "op_y"
    assert float(payload["final_metrics"]["fidelity_exact"]) > 0.999
    assert int(payload["final_logical_block_count"]) == 2


def test_adapt_checkpoint_snapshot_with_state_preserves_runtime_objects() -> None:
    snapshot, bundle = _toy_snapshot_and_bundle()
    adapt_cfg = CheckpointLocalAdaptConfig(
        objective="fidelity_first",
        pool_mode="family_pool",
        target_fidelity=0.999,
        max_steps=2,
        gradient_threshold=1.0e-8,
        probe_scale=0.2,
        min_fidelity_gain=1.0e-6,
        plateau_patience=1,
        candidate_rank_limit=4,
    )
    fit_cfg = FrozenScaffoldExactFitConfig(
        objectives=("fidelity_first",),
        method="Powell",
        maxiter=120,
        restarts=3,
        seed=5,
        initial_sigma=0.05,
    )

    result = adapt_checkpoint_snapshot_with_state(snapshot, bundle=bundle, adapt_cfg=adapt_cfg, fit_cfg=fit_cfg)

    assert result.payload["final_logical_block_count"] == result.state.layout.logical_parameter_count
    assert result.payload["final_runtime_parameter_count"] == result.state.layout.runtime_parameter_count
    assert len(result.state.terms) == int(result.state.layout.logical_parameter_count)
    assert result.state.theta_runtime.shape == (int(result.state.layout.runtime_parameter_count),)
    assert list(result.state.scaffold_labels) == result.payload["final_scaffold_labels"]
    psi_fit = result.state.executor.prepare_state(result.state.theta_runtime, snapshot["psi_ref"])
    assert float(abs(np.vdot(snapshot["psi_exact"], psi_fit)) ** 2) > 0.999


def test_resolve_candidate_pool_terms_full_meta_uses_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot, bundle = _toy_snapshot_and_bundle()
    _ = snapshot
    x_term, y_term = _toy_terms()

    def _fake_builder(**_kwargs: object) -> tuple[list[AnsatzTerm], str, dict[str, object] | None, dict[str, object] | None]:
        return [x_term, y_term], "fake_full_meta", {"classes": 2}, None

    monkeypatch.setattr(
        "pipelines.time_dynamics.legacy.experiments.checkpoint_local_adapt.build_hh_pool_by_key",
        _fake_builder,
    )

    pool_terms, meta = resolve_candidate_pool_terms(bundle, pool_mode="full_meta")

    assert [term.label for term in pool_terms] == ["op_x", "op_y"]
    assert meta["pool_method"] == "fake_full_meta"


def test_adapt_checkpoint_snapshot_phase3_joint_rescue_prefers_best_joint_gain_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, bundle = _toy_snapshot_and_bundle()

    def _fake_fit_checkpoint_snapshot(snapshot_in, *, cfg):
        labels = [str(x) for x in snapshot_in.get("scaffold_labels", [])]
        theta_runtime = [0.0] * int(snapshot_in.get("runtime_parameter_count", 1))
        if len(labels) == 1:
            metrics = {
                "theta_runtime": theta_runtime,
                "fidelity_exact": 0.70,
                "abs_energy_total_error": 0.20,
                "site_occupations_abs_error_max": 0.10,
            }
        elif labels and str(labels[0]).startswith("op_y"):
            metrics = {
                "theta_runtime": theta_runtime,
                "fidelity_exact": 0.74,
                "abs_energy_total_error": 0.08,
                "site_occupations_abs_error_max": 0.01,
            }
        else:
            metrics = {
                "theta_runtime": theta_runtime,
                "fidelity_exact": 0.81,
                "abs_energy_total_error": 0.19,
                "site_occupations_abs_error_max": 0.09,
            }
        return {
            "current_metrics": dict(metrics),
            "objectives": [
                {
                    "objective": str(cfg.objectives[0]),
                    "best_metrics": dict(metrics),
                }
            ],
        }

    monkeypatch.setattr(
        "pipelines.time_dynamics.legacy.experiments.checkpoint_local_adapt.fit_checkpoint_snapshot",
        _fake_fit_checkpoint_snapshot,
    )

    adapt_cfg = CheckpointLocalAdaptConfig(
        strategy="phase3_joint_rescue_v1",
        objective="fidelity_first",
        pool_mode="family_pool",
        target_fidelity=0.99,
        max_steps=1,
        gradient_threshold=1.0e-8,
        probe_scale=0.0,
        candidate_rank_limit=8,
        joint_site_weight=2.0,
        joint_energy_weight=3.0,
        joint_min_gain=0.0,
        joint_opt_mode="fidelity_fit_joint_rank",
    )
    fit_cfg = FrozenScaffoldExactFitConfig(
        objectives=("fidelity_first",),
        method="Powell",
        maxiter=20,
        restarts=1,
        seed=5,
        initial_sigma=0.05,
    )

    payload = adapt_checkpoint_snapshot(snapshot, bundle=bundle, adapt_cfg=adapt_cfg, fit_cfg=fit_cfg)

    assert payload["strategy"] == "phase3_joint_rescue_v1"
    assert payload["history"][0]["selected_position_id"] == 0
    assert payload["history"][0]["selected_fit_objective"] == "fidelity_first"
    assert payload["history"][0]["joint_ranking_top"][0]["position_id"] == 0
    assert payload["joint_rescue"]["positions_considered"] == "all_insertions"


def test_adapt_checkpoint_snapshot_phase3_joint_rescue_joint_fit_mode_uses_balanced_cfg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, bundle = _toy_snapshot_and_bundle()
    seen_cfgs: list[FrozenScaffoldExactFitConfig] = []

    def _fake_fit_checkpoint_snapshot(snapshot_in, *, cfg):
        del snapshot_in
        seen_cfgs.append(cfg)
        metrics = {
            "theta_runtime": [0.0],
            "fidelity_exact": 0.75,
            "abs_energy_total_error": 0.15,
            "site_occupations_abs_error_max": 0.05,
        }
        return {
            "current_metrics": dict(metrics),
            "objectives": [
                {
                    "objective": str(cfg.objectives[0]),
                    "best_metrics": dict(metrics),
                }
            ],
        }

    monkeypatch.setattr(
        "pipelines.time_dynamics.legacy.experiments.checkpoint_local_adapt.fit_checkpoint_snapshot",
        _fake_fit_checkpoint_snapshot,
    )

    adapt_cfg = CheckpointLocalAdaptConfig(
        strategy="phase3_joint_rescue_v1",
        objective="fidelity_first",
        pool_mode="family_pool",
        target_fidelity=0.5,
        max_steps=1,
        joint_site_weight=2.5,
        joint_energy_weight=3.5,
        joint_opt_mode="joint_fit_joint_rank",
    )
    fit_cfg = FrozenScaffoldExactFitConfig(
        objectives=("fidelity_first",),
        method="Powell",
        maxiter=20,
        restarts=1,
        seed=5,
        initial_sigma=0.05,
    )

    payload = adapt_checkpoint_snapshot(snapshot, bundle=bundle, adapt_cfg=adapt_cfg, fit_cfg=fit_cfg)

    assert seen_cfgs
    assert seen_cfgs[0].objectives == ("balanced",)
    assert seen_cfgs[0].balanced_energy_weight == pytest.approx(7.0)
    assert seen_cfgs[0].balanced_site_weight == pytest.approx(2.5)
    assert payload["fit_objective_used"] == "balanced"


def test_run_checkpoint_local_adapt_from_args_reuses_one_exact_reference_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    build_cache_ids: list[int] = []
    capture_cache_ids: list[int] = []

    def _fake_build_controller_bundle_from_args(args, *, exact_reference_cache=None):
        del args
        build_cache_ids.append(id(exact_reference_cache))
        return {"cfg": {"mode": "exact_v1"}, "drive_config": None, "oracle_config": None}

    def _fake_capture_checkpoint_snapshot_from_args(
        args,
        *,
        checkpoint_index: int,
        force_stay_checkpoints,
        exact_reference_cache=None,
    ):
        del args, force_stay_checkpoints
        capture_cache_ids.append(id(exact_reference_cache))
        return {"checkpoint_index": int(checkpoint_index)}, {"loaded": SimpleNamespace(replay_context=SimpleNamespace())}

    def _fake_adapt_checkpoint_snapshot(snapshot, *, bundle, adapt_cfg, fit_cfg):
        del bundle, adapt_cfg, fit_cfg
        return {"checkpoint_index": int(snapshot["checkpoint_index"]), "final_metrics": {"fidelity_exact": 1.0}}

    monkeypatch.setattr(
        "pipelines.time_dynamics.legacy.experiments.checkpoint_local_adapt.build_controller_bundle_from_args",
        _fake_build_controller_bundle_from_args,
    )
    monkeypatch.setattr(
        "pipelines.time_dynamics.legacy.experiments.checkpoint_local_adapt.capture_checkpoint_snapshot_from_args",
        _fake_capture_checkpoint_snapshot_from_args,
    )
    monkeypatch.setattr(
        "pipelines.time_dynamics.legacy.experiments.checkpoint_local_adapt.adapt_checkpoint_snapshot",
        _fake_adapt_checkpoint_snapshot,
    )
    args = SimpleNamespace(
        checkpoint_adapt_checkpoints="2,5",
        force_stay_checkpoints="",
        checkpoint_adapt_objective="balanced",
        checkpoint_adapt_pool_mode="full_meta",
        checkpoint_adapt_target_fidelity=0.99,
        checkpoint_adapt_max_steps=4,
        checkpoint_adapt_gradient_threshold=1.0e-6,
        checkpoint_adapt_probe_scale=0.15,
        checkpoint_adapt_min_fidelity_gain=1.0e-4,
        checkpoint_adapt_plateau_patience=2,
        checkpoint_adapt_candidate_rank_limit=8,
        fit_method="Powell",
        fit_maxiter=10,
        fit_restarts=2,
        fit_seed=7,
        fit_initial_sigma=0.25,
        fit_balanced_energy_weight=3.0,
        fit_balanced_site_weight=1.0,
        output_json=str(tmp_path / "adapt.json"),
        artifact_json=str(tmp_path / "artifact.json"),
        run_tag="adapt_cache",
        loader_mode="replay_family",
    )

    payload = run_checkpoint_local_adapt_from_args(args)

    assert payload["checkpoint_adapt_checkpoints"] == [2, 5]
    assert len(build_cache_ids) == 1
    assert len(capture_cache_ids) == 2
    assert len(set(build_cache_ids + capture_cache_ids)) == 1
