from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.ansatz_parameterization import build_parameter_layout, serialize_layout
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, PauliPolynomial, PauliTerm

import pipelines.hardcoded.hh_staged_workflow as wf
from pipelines.hardcoded.hh_staged_noiseless import parse_args
from pipelines.hardcoded.hh_realtime_checkpoint_types import ScaffoldAcceptanceResult
from pipelines.hardcoded.hh_staged_workflow import resolve_staged_hh_config


def _basis(dim: int, idx: int) -> np.ndarray:
    out = np.zeros(dim, dtype=complex)
    out[int(idx)] = 1.0
    return out


def _amplitudes_qn_to_q0(psi: np.ndarray) -> dict[str, dict[str, float]]:
    nq = int(round(np.log2(int(np.asarray(psi).size))))
    out: dict[str, dict[str, float]] = {}
    for idx, amp in enumerate(np.asarray(psi, dtype=complex).reshape(-1)):
        if abs(amp) <= 1e-14:
            continue
        out[format(idx, f"0{nq}b")] = {"re": float(np.real(amp)), "im": float(np.imag(amp))}
    return out


def test_resolve_staged_defaults_from_run_guide_formulae() -> None:
    args = parse_args(["--L", "3", "--skip-pdf"])
    cfg = resolve_staged_hh_config(args)

    assert int(cfg.warm_start.reps) == 3
    assert int(cfg.warm_start.restarts) == 5
    assert int(cfg.warm_start.maxiter) == 4000
    assert int(cfg.adapt.max_depth) == 120
    assert int(cfg.adapt.maxiter) == 5000
    assert float(cfg.adapt.eps_grad) == pytest.approx(5e-7)
    assert float(cfg.adapt.eps_energy) == pytest.approx(1e-9)
    assert str(cfg.adapt.continuation_mode) == "phase3_v1"
    assert int(cfg.replay.reps) == 3
    assert int(cfg.replay.restarts) == 5
    assert int(cfg.replay.maxiter) == 4000
    assert str(cfg.replay.continuation_mode) == "phase3_v1"
    assert int(cfg.dynamics.t_final) == 15
    assert int(cfg.dynamics.trotter_steps) == 192
    assert int(cfg.dynamics.num_times) == 201
    assert int(cfg.dynamics.exact_steps_multiplier) == 2
    assert float(cfg.gates.ecut_1) == pytest.approx(1e-1)
    assert float(cfg.gates.ecut_2) == pytest.approx(1e-4)
    assert bool(cfg.dynamics.enable_drive) is False
    assert cfg.default_provenance["warm_reps"] == "run_guide.ws_reps(L)=L"
    assert cfg.default_provenance["replay_continuation_mode"] == "workflow.replay_mode := adapt_continuation_mode"


def test_nondefault_sector_override_rejected_cleanly() -> None:
    args = parse_args(["--L", "2", "--sector-n-up", "2", "--skip-pdf"])
    with pytest.raises(ValueError, match="half-filled sector"):
        resolve_staged_hh_config(args)


def test_handoff_continuation_meta_keeps_curated_details_and_drops_large_rows() -> None:
    meta = wf._handoff_continuation_meta(
        {
            "continuation_mode": "phase3_v1",
            "scaffold_fingerprint_lite": {"num_parameters": 3},
            "continuation": {
                "mode": "phase3_v1",
                "optimizer_memory": {"cached": True},
                "runtime_split_summary": {"selected_child_count": 2},
                "score_version": "phase3_reduced_rerank_v1",
                "phase1_feature_rows": [{"drop": True}],
                "phase2_shortlist_rows": [{"drop": True}],
            },
        }
    )
    assert meta["continuation_mode"] == "phase3_v1"
    assert meta["continuation_scaffold"] == {"num_parameters": 3}
    assert meta["continuation_details"]["runtime_split_summary"]["selected_child_count"] == 2
    assert meta["continuation_details"]["score_version"] == "phase3_reduced_rerank_v1"
    assert "phase1_feature_rows" not in meta["continuation_details"]
    assert "phase2_shortlist_rows" not in meta["continuation_details"]


def test_resolve_preserves_explicit_pareto_lean_pool() -> None:
    args = parse_args(["--L", "2", "--adapt-pool", "pareto_lean", "--skip-pdf"])
    cfg = resolve_staged_hh_config(args)
    assert str(cfg.adapt.pool) == "pareto_lean"


def test_underparameterized_override_rejected_without_smoke_flag() -> None:
    args = parse_args([
        "--L",
        "2",
        "--warm-reps",
        "1",
        "--skip-pdf",
    ])
    with pytest.raises(ValueError, match="Under-parameterized staged HH run rejected"):
        resolve_staged_hh_config(args)

    args = parse_args([
        "--L",
        "2",
        "--warm-reps",
        "1",
        "--skip-pdf",
        "--smoke-test-intentionally-weak",
    ])
    cfg = resolve_staged_hh_config(args)
    assert int(cfg.warm_start.reps) == 1


def test_resolve_staged_builds_phase3_raw_oracle_config() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--skip-pdf",
            "--adapt-continuation-mode",
            "phase3_v1",
            "--phase3-oracle-gradient-mode",
            "runtime",
            "--phase3-oracle-backend-name",
            "ibm_marrakesh",
        ]
    )
    cfg = resolve_staged_hh_config(args)
    oracle_cfg = cfg.adapt.phase3_oracle_gradient_config
    cfg_payload = asdict(cfg)

    assert oracle_cfg is not None
    assert str(oracle_cfg.noise_mode) == "runtime"
    assert str(oracle_cfg.execution_surface) == "raw_measurement_v1"
    assert str(oracle_cfg.execution_surface_requested) == "auto"
    assert str(oracle_cfg.raw_transport) == "auto"
    assert float(oracle_cfg.gradient_step) == pytest.approx(float(cfg.adapt.finite_angle))
    assert cfg_payload["adapt"]["phase3_oracle_gradient_config"]["execution_surface"] == "raw_measurement_v1"


def test_run_staged_hh_noiseless_rejects_phase3_oracle_config(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--skip-pdf",
            "--output-json",
            str(tmp_path / "hh_staged.json"),
            "--output-pdf",
            str(tmp_path / "hh_staged.pdf"),
            "--adapt-continuation-mode",
            "phase3_v1",
            "--phase3-oracle-gradient-mode",
            "backend_scheduled",
            "--phase3-oracle-use-fake-backend",
            "--phase3-oracle-backend-name",
            "FakeNighthawk",
        ]
    )
    cfg = resolve_staged_hh_config(args)
    with pytest.raises(ValueError, match="staged noise workflow"):
        wf.run_staged_hh_noiseless(cfg, run_command="pytest")


def test_workflow_runs_matched_family_replay_and_static_plus_drive_profiles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dim = 1 << int(wf._hh_nq_total(2, 1, "binary"))
    psi_hf = _basis(dim, 0)
    psi_warm = _basis(dim, 1)
    psi_adapt = _basis(dim, 2)
    psi_final = _basis(dim, 3)
    runtime_layout = serialize_layout(
        build_parameter_layout(
            [
                AnsatzTerm(
                    label="op_1",
                    polynomial=PauliPolynomial(
                        "JW",
                        [PauliTerm(6, ps="xeeeee", pc=1.0), PauliTerm(6, ps="zeeeee", pc=0.5)],
                    ),
                ),
                AnsatzTerm(
                    label="op_2",
                    polynomial=PauliPolynomial("JW", [PauliTerm(6, ps="eyeeee", pc=1.0)]),
                ),
            ]
        )
    )
    calls: dict[str, object] = {}

    monkeypatch.setattr(wf, "build_hubbard_holstein_hamiltonian", lambda **kwargs: object())
    monkeypatch.setattr(wf, "hubbard_holstein_reference_state", lambda **kwargs: np.array(psi_hf, copy=True))
    monkeypatch.setattr(wf.hc_pipeline, "_collect_hardcoded_terms_exyz", lambda h: (["eeeeee"], {"eeeeee": 1.0 + 0.0j}))
    monkeypatch.setattr(wf.hc_pipeline, "_build_hamiltonian_matrix", lambda coeff: np.eye(dim, dtype=complex))

    def _fake_run_hardcoded_vqe(**kwargs):
        calls["warm_kwargs"] = kwargs
        return {
            "success": True,
            "ansatz": "hh_hva_ptw",
            "optimizer_method": str(kwargs["method"]),
            "energy": -1.00,
            "exact_filtered_energy": -1.02,
            "message": "warm_ok",
        }, np.array(psi_warm, copy=True)

    def _fake_run_adapt(**kwargs):
        calls["adapt_kwargs"] = kwargs
        return {
            "success": True,
            "energy": -1.03,
            "exact_gs_energy": -1.04,
            "abs_delta_e": 0.01,
            "ansatz_depth": 2,
            "num_parameters": 3,
            "logical_num_parameters": 2,
            "pool_type": "phase3_v1",
            "continuation_mode": str(kwargs["adapt_continuation_mode"]),
            "stop_reason": "eps_grad",
            "operators": ["op_1", "op_2"],
            "optimal_point": [0.1, 0.15, 0.2],
            "logical_optimal_point": [0.125, 0.2],
            "parameterization": runtime_layout,
            "measurement_cache_summary": {
                "groups_known": 3.0,
                "plan_version": "phase1_qwc_basis_cover_reuse",
            },
            "compile_cost_proxy_summary": {
                "version": "phase3_v1_proxy",
                "components": ["gate_proxy_total", "cx_proxy_total"],
            },
            "continuation": {
                "optimizer_memory": {"cached": True},
                "selected_generator_metadata": [{"generator_id": "g1"}],
                "runtime_split_summary": {
                    "mode": "shortlist_pauli_children_v1",
                    "selected_child_count": 1,
                },
            },
            "history": [
                {
                    "depth": 1,
                    "depth_cumulative": 1,
                    "batch_size": 1,
                    "candidate_family": "phase3_v1",
                    "selection_mode": "append",
                    "energy_after_opt": -1.03,
                    "delta_abs_current": 0.01,
                    "delta_abs_drop_from_prev": 0.01,
                    "measurement_cache_stats": {
                        "groups_new": 1,
                        "shots_new": 1000.0,
                        "reuse_count_cost": 1.0,
                    },
                    "compile_cost_proxy": {
                        "gate_proxy_total": 4.0,
                        "cx_proxy_total": 2.0,
                        "sq_proxy_total": 4.0,
                    },
                }
            ],
        }, np.array(psi_adapt, copy=True)

    def _fake_write_handoff_state_bundle(**kwargs):
        calls["handoff_kwargs"] = kwargs

    def _fake_replay_run(cfg):
        calls["replay_cfg"] = cfg
        return {
            "generator_family": {
                "requested": "match_adapt",
                "resolved": "paop_lf_std",
                "resolution_source": "adapt_vqe.pool_type",
            },
            "seed_baseline": {"theta_policy": "auto", "abs_delta_e": 0.005},
            "exact": {"E_exact_sector": -1.05},
            "vqe": {"energy": -1.049, "stop_reason": "converged"},
            "replay_contract": {"continuation_mode": str(cfg.replay_continuation_mode)},
            "best_state": {"amplitudes_qn_to_q0": _amplitudes_qn_to_q0(psi_final)},
        }

    def _fake_simulate_trajectory(**kwargs):
        calls.setdefault("propagators", []).append(str(kwargs["propagator"]))
        rows = [
            {
                "time": 0.0,
                "fidelity": 1.0,
                "energy_total_trotter": -1.049,
                "energy_total_exact": -1.049,
                "doublon_trotter": 0.1,
                "doublon_exact": 0.1,
            },
            {
                "time": 1.0,
                "fidelity": 0.99,
                "energy_total_trotter": -1.045,
                "energy_total_exact": -1.049,
                "doublon_trotter": 0.11,
                "doublon_exact": 0.10,
            },
        ]
        return rows, []

    class _FakeDriveTemplate:
        def labels_exyz(self, include_identity: bool = False):
            return ["zeeeee"]

    class _FakeDrive:
        def __init__(self):
            self.include_identity = False
            self.template = _FakeDriveTemplate()
            self.coeff_map_exyz = lambda _t: {"zeeeee": 0.1 + 0.0j}

    monkeypatch.setattr(wf.hc_pipeline, "_run_hardcoded_vqe", _fake_run_hardcoded_vqe)
    monkeypatch.setattr(wf.adapt_mod, "_run_hardcoded_adapt_vqe", _fake_run_adapt)
    monkeypatch.setattr(wf, "write_handoff_state_bundle", _fake_write_handoff_state_bundle)
    monkeypatch.setattr(wf.replay_mod, "run", _fake_replay_run)
    monkeypatch.setattr(wf.hc_pipeline, "_simulate_trajectory", _fake_simulate_trajectory)
    monkeypatch.setattr(wf, "build_gaussian_sinusoid_density_drive", lambda **kwargs: _FakeDrive())

    args = parse_args(
        [
            "--L",
            "2",
            "--adapt-pool",
            "pareto_lean",
            "--skip-pdf",
            "--enable-drive",
            "--output-json",
            str(tmp_path / "hh_staged.json"),
            "--output-pdf",
            str(tmp_path / "hh_staged.pdf"),
        ]
    )
    cfg = resolve_staged_hh_config(args)
    payload = wf.run_staged_hh_noiseless(cfg, run_command="python pipelines/hardcoded/hh_staged_noiseless.py --L 2")

    warm_kwargs = calls["warm_kwargs"]
    adapt_kwargs = calls["adapt_kwargs"]
    replay_cfg = calls["replay_cfg"]
    handoff_kwargs = calls["handoff_kwargs"]

    assert warm_kwargs["ansatz_name"] == "hh_hva_ptw"
    assert str(adapt_kwargs["adapt_pool"]) == "pareto_lean"
    assert np.allclose(adapt_kwargs["psi_ref_override"], psi_warm)
    assert handoff_kwargs["handoff_state_kind"] == "prepared_state"
    assert np.allclose(handoff_kwargs["ansatz_input_state"], psi_warm)
    assert handoff_kwargs["ansatz_input_state_source"] == "warm_start_hva"
    assert handoff_kwargs["ansatz_input_state_handoff_state_kind"] == "prepared_state"
    assert handoff_kwargs["adapt_optimal_point"] == [0.1, 0.15, 0.2]
    assert handoff_kwargs["adapt_logical_optimal_point"] == [0.125, 0.2]
    assert handoff_kwargs["adapt_logical_num_parameters"] == 2
    assert handoff_kwargs["adapt_parameterization"]["mode"] == "per_pauli_term_v1"
    assert replay_cfg.generator_family == "match_adapt"
    assert replay_cfg.replay_continuation_mode == "phase3_v1"
    assert payload["stage_pipeline"]["conventional_replay"]["generator_family"]["requested"] == "match_adapt"
    assert payload["stage_pipeline"]["adapt_vqe"]["measurement_cache_summary"]["groups_known"] == pytest.approx(3.0)
    assert payload["stage_pipeline"]["adapt_vqe"]["num_parameters"] == 3
    assert payload["stage_pipeline"]["adapt_vqe"]["logical_num_parameters"] == 2
    assert payload["stage_pipeline"]["adapt_vqe"]["compile_cost_proxy_summary"]["version"] == "phase3_v1_proxy"
    assert payload["stage_pipeline"]["adapt_vqe"]["runtime_split_summary"]["selected_child_count"] == 1
    assert set(payload["dynamics_noiseless"]["profiles"].keys()) == {"static", "drive"}
    static_profile = payload["dynamics_noiseless"]["profiles"]["static"]
    static_rows = static_profile["methods"]["suzuki2"]["trajectory"]
    assert static_profile["ground_state_reference"]["energy"] == pytest.approx(-1.05)
    assert abs(static_rows[0]["energy_total_trotter"] - static_rows[0]["energy_total_exact"]) == pytest.approx(0.0)
    assert static_rows[0]["abs_energy_error_vs_ground_state"] == pytest.approx(1e-3)
    assert payload["comparisons"]["noiseless_vs_ground_state"]["static"]["suzuki2"]["final_abs_energy_error"] == pytest.approx(5e-3)
    assert payload["comparisons"]["noiseless_vs_reference"]["static"]["suzuki2"]["final_fidelity"] == pytest.approx(0.99)
    assert "noiseless_vs_exact" not in payload["comparisons"]
    assert payload["workflow_contract"]["noiseless_energy_metric"].startswith("|E_method(t) - E_exact_sector_replay|")
    assert payload["pareto_tracking"]["current_run"]["frontier_count"] == 1
    assert payload["pareto_tracking"]["rolling"]["ledger_row_count"] == 3
    assert Path(payload["artifacts"]["pareto"]["run_rows_json"]).exists()
    assert Path(payload["artifacts"]["pareto"]["rolling_frontier_json"]).exists()
    assert calls["propagators"] == ["suzuki2", "cfqm4", "suzuki2", "cfqm4"]
    assert Path(cfg.artifacts.output_json).exists()


def test_run_stage_pipeline_forwards_phase3_oracle_config_and_stage_summary_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dim = 1 << int(wf._hh_nq_total(2, 1, "binary"))
    psi_hf = _basis(dim, 0)
    psi_warm = _basis(dim, 1)
    psi_adapt = _basis(dim, 2)
    psi_final = _basis(dim, 3)
    calls: dict[str, object] = {}

    monkeypatch.setattr(wf, "build_hubbard_holstein_hamiltonian", lambda **kwargs: object())
    monkeypatch.setattr(wf, "hubbard_holstein_reference_state", lambda **kwargs: np.array(psi_hf, copy=True))
    monkeypatch.setattr(wf.hc_pipeline, "_collect_hardcoded_terms_exyz", lambda h: (["eeeeee"], {"eeeeee": 1.0 + 0.0j}))
    monkeypatch.setattr(wf.hc_pipeline, "_build_hamiltonian_matrix", lambda coeff: np.eye(dim, dtype=complex))
    monkeypatch.setattr(
        wf.hc_pipeline,
        "_run_hardcoded_vqe",
        lambda **kwargs: ({"energy": -1.0, "exact_filtered_energy": -1.0, "ansatz": "hh_hva_ptw"}, np.array(psi_warm, copy=True)),
    )

    def _fake_run_adapt(**kwargs):
        calls["adapt_kwargs"] = kwargs
        return {
            "energy": -1.03,
            "exact_gs_energy": -1.04,
            "ansatz_depth": 2,
            "num_parameters": 3,
            "logical_num_parameters": 2,
            "pool_type": "phase3_v1",
            "continuation_mode": str(kwargs["adapt_continuation_mode"]),
            "continuation": {
                "gradient_uncertainty_source": "oracle_fd_stderr_v1",
                "oracle_gradient_scope": "selection_only",
                "oracle_gradient_config": {
                    "execution_surface_requested": "auto",
                    "execution_surface": "raw_measurement_v1",
                    "raw_transport": "auto",
                    "raw_store_memory": True,
                    "raw_artifact_path": str(tmp_path / "phase3.ndjson.gz"),
                },
                "oracle_execution_surface": "raw_measurement_v1",
                "oracle_backend_info": {"backend_name": "FakeNighthawk"},
                "oracle_raw_transport": "sampler_v2",
                "oracle_gradient_raw_records_total": 8,
                "oracle_gradient_raw_artifact_path": str(tmp_path / "phase3.ndjson.gz"),
                "reoptimization_backend": "exact_statevector",
            },
        }, np.array(psi_adapt, copy=True)

    monkeypatch.setattr(wf.adapt_mod, "_run_hardcoded_adapt_vqe", _fake_run_adapt)
    monkeypatch.setattr(wf, "write_handoff_state_bundle", lambda **kwargs: None)
    monkeypatch.setattr(
        wf.replay_mod,
        "run",
        lambda cfg: {
            "generator_family": {"requested": "match_adapt", "resolved": "paop_lf_std"},
            "seed_baseline": {"theta_policy": "auto"},
            "exact": {"E_exact_sector": -1.05},
            "vqe": {"energy": -1.049, "stop_reason": "converged"},
            "replay_contract": {"continuation_mode": str(cfg.replay_continuation_mode)},
            "best_state": {"amplitudes_qn_to_q0": _amplitudes_qn_to_q0(psi_final)},
        },
    )

    args = parse_args(
        [
            "--L",
            "2",
            "--skip-pdf",
            "--output-json",
            str(tmp_path / "hh_staged.json"),
            "--output-pdf",
            str(tmp_path / "hh_staged.pdf"),
            "--adapt-continuation-mode",
            "phase3_v1",
            "--phase3-oracle-gradient-mode",
            "runtime",
            "--phase3-oracle-backend-name",
            "ibm_marrakesh",
            "--phase3-oracle-raw-store-memory",
            "--phase3-oracle-raw-artifact-path",
            str(tmp_path / "phase3.ndjson.gz"),
        ]
    )
    cfg = resolve_staged_hh_config(args)
    stage_result = wf.run_stage_pipeline(cfg)
    stage_summary = wf._stage_summary(stage_result, cfg)
    adapt_kwargs = calls["adapt_kwargs"]
    oracle_cfg = adapt_kwargs["phase3_oracle_gradient_config"]
    adapt_stage = stage_summary["adapt_vqe"]

    assert oracle_cfg is not None
    assert str(oracle_cfg.execution_surface) == "raw_measurement_v1"
    assert bool(oracle_cfg.raw_store_memory) is True
    assert str(oracle_cfg.raw_artifact_path) == str(tmp_path / "phase3.ndjson.gz")
    assert adapt_stage["gradient_uncertainty_source"] == "oracle_fd_stderr_v1"
    assert adapt_stage["oracle_gradient_scope"] == "selection_only"
    assert adapt_stage["oracle_execution_surface"] == "raw_measurement_v1"
    assert adapt_stage["oracle_raw_transport"] == "sampler_v2"
    assert adapt_stage["oracle_gradient_raw_records_total"] == 8
    assert adapt_stage["oracle_gradient_raw_artifact_path"] == str(tmp_path / "phase3.ndjson.gz")
    assert adapt_stage["reoptimization_backend"] == "exact_statevector"
    assert adapt_stage["oracle_gradient_config"]["raw_store_memory"] is True


def test_run_staged_hh_noiseless_adds_checkpoint_controller_block_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--skip-pdf",
            "--checkpoint-controller-mode",
            "exact_v1",
            "--output-json",
            str(tmp_path / "hh_staged.json"),
            "--output-pdf",
            str(tmp_path / "hh_staged.pdf"),
        ]
    )
    cfg = resolve_staged_hh_config(args)
    stage_result = wf.StageExecutionResult(
        h_poly=object(),
        hmat=np.eye(4, dtype=complex),
        ordered_labels_exyz=["ee"],
        coeff_map_exyz={"ee": 1.0 + 0.0j},
        nq_total=2,
        psi_hf=_basis(4, 0),
        psi_warm=_basis(4, 1),
        psi_adapt=_basis(4, 2),
        psi_final=_basis(4, 3),
        warm_payload={"energy": -1.0, "exact_filtered_energy": -1.1},
        adapt_payload={"energy": -1.05, "exact_gs_energy": -1.1, "stop_reason": "eps_grad"},
        replay_payload={
            "vqe": {"energy": -1.09},
            "exact": {"E_exact_sector": -1.1},
            "best_state": {"best_theta": [0.1]},
        },
    )

    fake_context = SimpleNamespace(payload_in={"adapt_vqe": {"pool_type": "phase3_v1"}})
    monkeypatch.setattr(wf, "run_stage_pipeline", lambda _cfg: stage_result)
    monkeypatch.setattr(wf.replay_mod, "build_replay_scaffold_context", lambda *_args, **_kwargs: fake_context)
    monkeypatch.setattr(wf, "run_noiseless_profiles", lambda _stage, _cfg: {"profiles": {"static": {"methods": {}}}})
    monkeypatch.setattr(
        wf,
        "run_adaptive_realtime_checkpoint_profile",
        lambda _stage, _cfg: {"mode": "exact_v1", "status": "completed", "summary": {"append_count": 1, "stay_count": 1}},
    )

    payload = wf.run_staged_hh_noiseless(cfg, run_command="python staged.py --checkpoint-controller-mode exact_v1")

    assert payload["workflow_contract"]["adaptive_realtime_checkpoint_mode"] == "exact_v1"
    assert payload["adaptive_realtime_checkpoint"]["mode"] == "exact_v1"
    assert "profiles" in payload["dynamics_noiseless"]


def test_run_staged_hh_noiseless_rejects_oracle_v1(
    tmp_path: Path,
) -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--skip-pdf",
            "--checkpoint-controller-mode",
            "oracle_v1",
            "--output-json",
            str(tmp_path / "hh_staged.json"),
            "--output-pdf",
            str(tmp_path / "hh_staged.pdf"),
        ]
    )
    cfg = resolve_staged_hh_config(args)

    with pytest.raises(ValueError, match="oracle_v1"):
        wf.run_staged_hh_noiseless(cfg)


def test_run_adaptive_realtime_checkpoint_profile_requires_best_theta_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--skip-pdf",
            "--checkpoint-controller-mode",
            "exact_v1",
            "--output-json",
            str(tmp_path / "hh_staged.json"),
            "--output-pdf",
            str(tmp_path / "hh_staged.pdf"),
        ]
    )
    cfg = resolve_staged_hh_config(args)
    stage_result = wf.StageExecutionResult(
        h_poly=object(),
        hmat=np.eye(2, dtype=complex),
        ordered_labels_exyz=["e"],
        coeff_map_exyz={"e": 1.0 + 0.0j},
        nq_total=1,
        psi_hf=np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
        psi_warm=np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
        psi_adapt=np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
        psi_final=np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
        warm_payload={"energy": -1.0, "exact_filtered_energy": -1.0},
        adapt_payload={"energy": -1.0, "exact_gs_energy": -1.0, "stop_reason": "eps_grad"},
        replay_payload={
            "vqe": {"energy": -1.0},
            "exact": {"E_exact_sector": -1.0},
            "best_state": {"amplitudes_qn_to_q0": {"0": {"re": 1.0, "im": 0.0}}},
        },
    )
    fake_context = SimpleNamespace(payload_in={"adapt_vqe": {"pool_type": "phase3_v1"}})
    monkeypatch.setattr(wf.replay_mod, "build_replay_scaffold_context", lambda *_args, **_kwargs: fake_context)

    with pytest.raises(ValueError, match="best_state.best_theta"):
        wf.run_adaptive_realtime_checkpoint_profile(stage_result, cfg)


def test_run_adaptive_realtime_checkpoint_profile_allows_drive_and_forwards_drive_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--skip-pdf",
            "--enable-drive",
            "--drive-A",
            "1.5",
            "--drive-omega",
            "2.0",
            "--drive-tbar",
            "2.5",
            "--drive-t0",
            "5.0",
            "--drive-pattern",
            "staggered",
            "--checkpoint-controller-mode",
            "exact_v1",
            "--checkpoint-controller-exact-forecast-baseline-step-refine-rounds",
            "2",
            "--checkpoint-controller-exact-forecast-baseline-blend-weights",
            "0,0.25,0.5,1.0",
            "--checkpoint-controller-exact-forecast-baseline-gain-scales",
            "1.0,1.1,1.25",
            "--checkpoint-controller-exact-forecast-horizon-steps",
            "3",
            "--checkpoint-controller-exact-forecast-horizon-weights",
            "3,2,1",
            "--checkpoint-controller-exact-forecast-energy-slope-weight",
            "500",
            "--checkpoint-controller-exact-forecast-energy-curvature-weight",
            "25",
            "--checkpoint-controller-exact-forecast-energy-excursion-under-weight",
            "300",
            "--checkpoint-controller-exact-forecast-energy-excursion-over-weight",
            "120",
            "--checkpoint-controller-exact-forecast-energy-excursion-rel-tolerance",
            "0.05",
            "--output-json",
            str(tmp_path / "hh_staged.json"),
            "--output-pdf",
            str(tmp_path / "hh_staged.pdf"),
        ]
    )
    cfg = resolve_staged_hh_config(args)
    stage_result = wf.StageExecutionResult(
        h_poly=object(),
        hmat=np.eye(4, dtype=complex),
        ordered_labels_exyz=["ee"],
        coeff_map_exyz={"ee": 1.0 + 0.0j},
        nq_total=2,
        psi_hf=_basis(4, 0),
        psi_warm=_basis(4, 1),
        psi_adapt=_basis(4, 2),
        psi_final=_basis(4, 3),
        warm_payload={"energy": -1.0, "exact_filtered_energy": -1.1},
        adapt_payload={"energy": -1.05, "exact_gs_energy": -1.1, "stop_reason": "eps_grad"},
        replay_payload={
            "vqe": {"energy": -1.09},
            "exact": {"E_exact_sector": -1.1},
            "best_state": {"best_theta": [0.1]},
        },
    )
    fake_context = SimpleNamespace(payload_in={"adapt_vqe": {"pool_type": "phase3_v1"}})
    fake_acceptance = ScaffoldAcceptanceResult(
        accepted=True,
        reason="ok",
        structure_locked=False,
        source_kind="pytest",
    )
    captured: dict[str, object] = {}

    class _FakeController:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            return SimpleNamespace(
                reference={"kind": "driven_piecewise_constant_reference_from_replay_seed"},
                trajectory=[],
                ledger=[],
                summary={"append_count": 0, "stay_count": 1},
            )

    monkeypatch.setattr(wf.replay_mod, "build_replay_scaffold_context", lambda *_args, **_kwargs: fake_context)
    monkeypatch.setattr(wf, "validate_scaffold_acceptance", lambda payload_in: fake_acceptance)
    monkeypatch.setattr(wf, "RealtimeCheckpointController", _FakeController)

    payload = wf.run_adaptive_realtime_checkpoint_profile(stage_result, cfg)

    assert payload is not None
    drive_cfg = captured["drive_config"]
    assert drive_cfg is not None
    assert bool(drive_cfg.enabled) is True
    assert float(drive_cfg.drive_A) == pytest.approx(1.5)
    assert float(drive_cfg.drive_t0) == pytest.approx(5.0)
    assert str(drive_cfg.drive_pattern) == "staggered"
    controller_cfg = captured["cfg"]
    assert int(controller_cfg.exact_forecast_baseline_step_refine_rounds) == 2
    assert tuple(controller_cfg.exact_forecast_baseline_blend_weights) == pytest.approx((0.0, 0.25, 0.5, 1.0))
    assert tuple(controller_cfg.exact_forecast_baseline_gain_scales) == pytest.approx((1.0, 1.1, 1.25))
    assert int(controller_cfg.exact_forecast_tracking_horizon_steps) == 3
    assert tuple(controller_cfg.exact_forecast_tracking_horizon_weights) == pytest.approx((3.0, 2.0, 1.0))
    assert float(controller_cfg.exact_forecast_energy_slope_weight) == pytest.approx(500.0)
    assert float(controller_cfg.exact_forecast_energy_curvature_weight) == pytest.approx(25.0)
    assert float(controller_cfg.exact_forecast_energy_excursion_under_weight) == pytest.approx(300.0)
    assert float(controller_cfg.exact_forecast_energy_excursion_over_weight) == pytest.approx(120.0)
    assert float(controller_cfg.exact_forecast_energy_excursion_rel_tolerance) == pytest.approx(0.05)
    assert str(payload["reference"]["kind"]) == "driven_piecewise_constant_reference_from_replay_seed"
