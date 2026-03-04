from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.cfqm_vs_suzuki_efficiency_suite import (  # noqa: E402
    EfficiencyConfig,
    _calibration_summary_from_samples,
    _build_exact_tie_tables,
    _chunk_rows,
    _compute_expm_multiply_calls,
    _expand_drive_cases,
    _expand_scenarios,
    _fit_loglog_slope,
    _maybe_calibrate_transpile,
    _select_fake_backend_spec,
    _to_config,
    parse_args,
    run_efficiency_suite,
)


def test_compute_expm_multiply_calls() -> None:
    m, r, t = _compute_expm_multiply_calls(
        method="cfqm4", stage_mode="exact_sparse", n_steps=10, n_ref_steps=80
    )
    assert (m, r, t) == (20, 80, 100)

    m, r, t = _compute_expm_multiply_calls(
        method="cfqm6", stage_mode="exact_dense", n_steps=7, n_ref_steps=56
    )
    assert (m, r, t) == (35, 56, 91)

    m, r, t = _compute_expm_multiply_calls(
        method="cfqm6", stage_mode="pauli_suzuki2", n_steps=7, n_ref_steps=56
    )
    assert (m, r, t) == (0, 56, 56)

    m, r, t = _compute_expm_multiply_calls(
        method="suzuki2", stage_mode="exact_sparse", n_steps=12, n_ref_steps=96
    )
    assert (m, r, t) == (0, 96, 96)


def test_build_exact_tie_tables() -> None:
    rows = [
        {
            "scenario_id": "A",
            "drive_case_id": "D",
            "track": "T",
            "method": "suzuki2",
            "cx_proxy_total": 100,
            "pauli_rot_count_total": 30,
            "expm_multiply_calls_total": 80,
            "wall_time_s": 1.0,
            "max_energy_abs_err": 1.0e-2,
            "stage_mode": "exact_sparse",
            "trotter_steps": 16,
            "max_doublon_abs_err": 1.0e-2,
            "final_infidelity": 1.0e-2,
        },
        {
            "scenario_id": "A",
            "drive_case_id": "D",
            "track": "T",
            "method": "cfqm4",
            "cx_proxy_total": 100,
            "pauli_rot_count_total": 35,
            "expm_multiply_calls_total": 96,
            "wall_time_s": 1.2,
            "max_energy_abs_err": 5.0e-3,
            "stage_mode": "exact_sparse",
            "trotter_steps": 8,
            "max_doublon_abs_err": 5.0e-3,
            "final_infidelity": 5.0e-3,
        },
    ]
    ties = _build_exact_tie_tables(rows, axis="cx_proxy")
    assert len(ties) == 1
    assert ties[0]["target_cost"] == 100.0
    methods = {r["method"] for r in ties[0]["rows"]}
    assert methods == {"suzuki2", "cfqm4"}


def test_fit_loglog_slope_order_four() -> None:
    dts = [0.4, 0.2, 0.1]
    errs = [dt**4 for dt in dts]
    slope = _fit_loglog_slope(dts=dts, errors=errs)
    assert slope is not None
    assert abs(float(slope) - 4.0) < 1.0e-6


def test_chunk_rows_bounds() -> None:
    rows = [[str(i)] for i in range(53)]
    chunks = list(_chunk_rows(rows, chunk_size=20))
    assert len(chunks) == 3
    assert len(chunks[0]) == 20
    assert len(chunks[1]) == 20
    assert len(chunks[2]) == 13


def test_expand_new_hh_scenarios() -> None:
    scenarios = _expand_scenarios(["hh_L2_nb1", "hh_L3_nb1"])
    ids = [s.scenario_id for s in scenarios]
    assert ids == ["hh_L2_nb1", "hh_L3_nb1"]
    assert scenarios[0].L == 2 and scenarios[0].n_ph_max == 1
    assert scenarios[1].L == 3 and scenarios[1].n_ph_max == 1


def test_parse_sinusoid_omegas_defaults_preserved() -> None:
    cfg = _to_config(parse_args([]))
    assert cfg.sinusoid_omegas == (0.5, 2.0, 8.0)


def test_parse_gaussian_tbars_defaults_preserved() -> None:
    cfg = _to_config(parse_args([]))
    assert cfg.gaussian_tbars == (0.25, 0.5)


def test_parse_calibration_defaults_preserved() -> None:
    cfg = _to_config(parse_args([]))
    assert cfg.calibration_backend == "auto"
    assert cfg.calibration_strict is False


def test_select_backend_auto_wide_enough() -> None:
    specs = [
        ("FakeLimaV2", object, 5),
        ("FakeJakartaV2", object, 7),
        ("FakeCairoV2", object, 27),
    ]
    chosen, reason = _select_fake_backend_spec(specs=specs, required_nq=9, requested="auto")
    assert reason is None
    assert chosen is not None
    assert chosen[0] == "FakeCairoV2"


def test_calibration_summary_counts() -> None:
    summary = _calibration_summary_from_samples(
        [
            {"status": "ok"},
            {"status": "ok"},
            {"status": "skipped"},
            {"status": "failed"},
        ],
        targets_total=6,
    )
    assert summary == {"targets_total": 6, "ok": 2, "skipped": 1, "failed": 1}


def test_calibration_failure_nonfatal_when_not_strict() -> None:
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_ibm_runtime")
    out = _maybe_calibrate_transpile(
        rows_internal=[
            {
                "run_id": "r_bad",
                "scenario_id": "S",
                "method": "cfqm4",
                "stage_mode": "exact_sparse",
                "trotter_steps": 8,
                "cx_proxy_total": 1,
                "ordered_labels": ["z"],
                "static_coeff_map": {"z": "not_a_number"},
            }
        ],
        active_coeff_tol=1e-14,
        t_final=1.0,
        enabled=True,
        backend_request="auto",
        strict=False,
    )
    assert out["enabled"] is True
    assert out["summary"]["failed"] == 1
    assert isinstance(out["samples"], list) and out["samples"]
    assert out["samples"][0]["status"] == "failed"


def test_calibration_failure_strict_raises() -> None:
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_ibm_runtime")
    with pytest.raises(RuntimeError):
        _maybe_calibrate_transpile(
            rows_internal=[
                {
                    "run_id": "r_bad",
                    "scenario_id": "S",
                    "method": "cfqm4",
                    "stage_mode": "exact_sparse",
                    "trotter_steps": 8,
                    "cx_proxy_total": 1,
                    "ordered_labels": ["z"],
                    "static_coeff_map": {"z": "not_a_number"},
                }
            ],
            active_coeff_tol=1e-14,
            t_final=1.0,
            enabled=True,
            backend_request="auto",
            strict=True,
        )


def test_expand_drive_cases_with_singleton_filters() -> None:
    cfg = _to_config(
        parse_args(
            [
                "--problem-grid",
                "hh_L2_nb1",
                "--drive-grid",
                "sinusoid,gaussian_sharp",
                "--sinusoid-omegas",
                "2.0",
                "--gaussian-tbars",
                "0.25",
                "--output-dir",
                "artifacts/cfqm_efficiency_benchmark/test_singleton_filters",
            ]
        )
    )
    cases = _expand_drive_cases(cfg)
    ids = {c["drive_case_id"] for c in cases}
    assert ids == {"sin_om2.0_tb5.0", "gauss_om2.0_tb0.25"}
    assert len(cases) == 2


def test_invalid_drive_filter_values_raise() -> None:
    with pytest.raises(ValueError):
        _to_config(
            parse_args(
                [
                    "--sinusoid-omegas",
                    "0.0",
                ]
            )
        )
    with pytest.raises(ValueError):
        _to_config(
            parse_args(
                [
                    "--gaussian-tbars",
                    "-1.0",
                ]
            )
        )


def test_run_efficiency_suite_smoke_with_mock_runner(tmp_path: Path) -> None:
    cfg = EfficiencyConfig(
        problem_grid=("hubbard_L4",),
        drive_grid=("sinusoid",),
        methods=("suzuki2", "cfqm4"),
        stage_mode_grid=("exact_sparse", "pauli_suzuki2"),
        steps_grid_override=(8, 16),
        reference_steps_multiplier=4,
        error_metrics=("final_infidelity", "max_doublon_abs_err", "max_energy_abs_err"),
        cost_metrics=("expm_calls", "pauli_rot_count", "cx_proxy", "depth_proxy", "wall_time"),
        equal_cost_axis=("cx_proxy", "pauli_rot_count", "expm_calls", "wall_time"),
        equal_cost_policy="exact_tie_only",
        calibrate_transpile=False,
        calibration_backend="auto",
        calibration_strict=False,
        output_dir=tmp_path,
        include_fallback_appendix=True,
        boundary="periodic",
        ordering="blocked",
        t_final=1.0,
        num_times=5,
        drive_A=0.2,
        drive_phi=0.0,
        drive_include_identity=False,
        drive_time_sampling="midpoint",
        sinusoid_omegas=(0.5, 2.0, 8.0),
        gaussian_tbars=(0.25, 0.5),
        initial_state_source="exact",
        adapt_input_json=None,
        adapt_strict_match=True,
        vqe_ansatz="uccsd",
        vqe_reps=1,
        vqe_restarts=1,
        vqe_maxiter=1,
        vqe_method="COBYLA",
        adapt_pool="uccsd",
        adapt_max_depth=1,
        adapt_maxiter=1,
    )

    def fake_runner(cmd: list[str], _cwd: Path) -> dict[str, object]:
        method = "suzuki2"
        steps = 8
        t_final = 1.0
        for i, tok in enumerate(cmd):
            if tok == "--propagator":
                method = str(cmd[i + 1])
            if tok == "--trotter-steps":
                steps = int(cmd[i + 1])
            if tok == "--t-final":
                t_final = float(cmd[i + 1])

        if method == "piecewise_exact":
            err = 0.0
        elif method == "suzuki2":
            err = 1.0 / float(steps**2)
        else:
            err = 1.0 / float(steps**4)

        traj = []
        for k in range(cfg.num_times):
            frac = 0.0 if cfg.num_times <= 1 else float(k) / float(cfg.num_times - 1)
            t = frac * t_final
            traj.append(
                {
                    "time": t,
                    "fidelity": 1.0 - err,
                    "energy_total_trotter": t + err,
                    "doublon_trotter": 0.5 * t + err,
                }
            )

        payload = {
            "settings": {
                "L": 4,
                "problem": "hubbard",
                "ordering": "blocked",
                "t_final": float(t_final),
                "drive": {
                    "enabled": True,
                    "A": 0.2,
                    "omega": 2.0,
                    "tbar": 5.0,
                    "phi": 0.0,
                    "t0": 0.0,
                    "pattern": "staggered",
                    "custom_s": None,
                    "include_identity": False,
                    "time_sampling": "midpoint",
                },
            },
            "hamiltonian": {
                "coefficients_exyz": [
                    {"label_exyz": "zzzzzzzz", "coeff": {"re": 1.0, "im": 0.0}},
                    {"label_exyz": "xxxxxxxx", "coeff": {"re": 0.5, "im": 0.0}},
                ]
            },
            "trajectory": traj,
            "_run_runtime_s": 0.01,
        }
        return payload

    out = run_efficiency_suite(cfg, run_pipeline=fake_runner)
    payload = out["payload"]

    assert out["runs_json"].exists()
    assert out["runs_csv"].exists()
    assert out["summary_json"].exists()
    assert out["pareto_json"].exists()
    assert out["slope_json"].exists()
    assert out["pdf"].exists()

    assert payload["schema"] == "cfqm_efficiency_suite_v1"
    assert isinstance(payload["runs"], list) and payload["runs"]
    assert isinstance(payload["slope_fits"], list) and payload["slope_fits"]
    assert "equal_cost_exact_ties" in payload
    assert "transpile_calibration" in payload

    parsed = json.loads(out["runs_json"].read_text(encoding="utf-8"))
    assert parsed["schema"] == "cfqm_efficiency_suite_v1"


def test_run_efficiency_suite_adapt_json_pass_through(tmp_path: Path) -> None:
    adapt_json = tmp_path / "adapt_seed.json"
    adapt_json.write_text("{}", encoding="utf-8")

    cfg = EfficiencyConfig(
        problem_grid=("hh_L2_nb1",),
        drive_grid=("sinusoid",),
        methods=("suzuki2",),
        stage_mode_grid=("exact_sparse",),
        steps_grid_override=(8, 16),
        reference_steps_multiplier=2,
        error_metrics=("final_infidelity", "max_doublon_abs_err", "max_energy_abs_err"),
        cost_metrics=("expm_calls", "pauli_rot_count", "cx_proxy", "depth_proxy", "wall_time"),
        equal_cost_axis=("cx_proxy",),
        equal_cost_policy="exact_tie_only",
        calibrate_transpile=False,
        calibration_backend="auto",
        calibration_strict=False,
        output_dir=tmp_path / "out",
        include_fallback_appendix=True,
        boundary="periodic",
        ordering="blocked",
        t_final=1.0,
        num_times=3,
        drive_A=0.2,
        drive_phi=0.0,
        drive_include_identity=False,
        drive_time_sampling="midpoint",
        sinusoid_omegas=(0.5, 2.0, 8.0),
        gaussian_tbars=(0.25, 0.5),
        initial_state_source="adapt_json",
        adapt_input_json=adapt_json,
        adapt_strict_match=False,
        vqe_ansatz="uccsd",
        vqe_reps=1,
        vqe_restarts=1,
        vqe_maxiter=1,
        vqe_method="COBYLA",
        adapt_pool="uccsd",
        adapt_max_depth=1,
        adapt_maxiter=1,
    )

    seen_adapt_flag = {"value": False}
    seen_no_strict_flag = {"value": False}

    def fake_runner(cmd: list[str], _cwd: Path) -> dict[str, object]:
        seen_adapt_flag["value"] = seen_adapt_flag["value"] or ("--adapt-input-json" in cmd)
        seen_no_strict_flag["value"] = seen_no_strict_flag["value"] or ("--no-adapt-strict-match" in cmd)
        method = "suzuki2"
        if "--propagator" in cmd:
            method = str(cmd[cmd.index("--propagator") + 1])
        steps = int(cmd[cmd.index("--trotter-steps") + 1])
        t_final = float(cmd[cmd.index("--t-final") + 1])
        err = 0.0 if method == "piecewise_exact" else 1.0 / max(1, steps**2)
        traj = [
            {"time": 0.0, "fidelity": 1.0 - err, "energy_total_trotter": err, "doublon_trotter": err},
            {"time": t_final, "fidelity": 1.0 - err, "energy_total_trotter": err, "doublon_trotter": err},
        ]
        return {
            "settings": {
                "L": 2,
                "problem": "hh",
                "ordering": "blocked",
                "t_final": t_final,
                "drive": {
                    "enabled": True,
                    "A": 0.2,
                    "omega": 0.5,
                    "tbar": 5.0,
                    "phi": 0.0,
                    "t0": 0.0,
                    "pattern": "staggered",
                    "custom_s": None,
                    "include_identity": False,
                    "time_sampling": "midpoint",
                },
            },
            "hamiltonian": {
                "coefficients_exyz": [
                    {"label_exyz": "zzzz", "coeff": {"re": 1.0, "im": 0.0}},
                ]
            },
            "trajectory": traj,
            "_run_runtime_s": 0.01,
        }

    run_efficiency_suite(cfg, run_pipeline=fake_runner)
    assert seen_adapt_flag["value"] is True
    assert seen_no_strict_flag["value"] is True


def test_run_efficiency_suite_auto_adapt_pool_hh(tmp_path: Path) -> None:
    cfg = EfficiencyConfig(
        problem_grid=("hh_L2_nb1",),
        drive_grid=("sinusoid",),
        methods=("suzuki2",),
        stage_mode_grid=("exact_sparse",),
        steps_grid_override=(8, 16),
        reference_steps_multiplier=2,
        error_metrics=("final_infidelity", "max_doublon_abs_err", "max_energy_abs_err"),
        cost_metrics=("expm_calls", "pauli_rot_count", "cx_proxy", "depth_proxy", "wall_time"),
        equal_cost_axis=("cx_proxy",),
        equal_cost_policy="exact_tie_only",
        calibrate_transpile=False,
        calibration_backend="auto",
        calibration_strict=False,
        output_dir=tmp_path / "out_auto_pool",
        include_fallback_appendix=True,
        boundary="periodic",
        ordering="blocked",
        t_final=1.0,
        num_times=3,
        drive_A=0.2,
        drive_phi=0.0,
        drive_include_identity=False,
        drive_time_sampling="midpoint",
        sinusoid_omegas=(0.5, 2.0, 8.0),
        gaussian_tbars=(0.25, 0.5),
        initial_state_source="exact",
        adapt_input_json=None,
        adapt_strict_match=True,
        vqe_ansatz="hh_hva",
        vqe_reps=1,
        vqe_restarts=1,
        vqe_maxiter=1,
        vqe_method="COBYLA",
        adapt_pool="auto",
        adapt_max_depth=1,
        adapt_maxiter=1,
    )

    seen_pool = {"value": None}

    def fake_runner(cmd: list[str], _cwd: Path) -> dict[str, object]:
        if "--adapt-pool" in cmd:
            seen_pool["value"] = str(cmd[cmd.index("--adapt-pool") + 1])
        method = "suzuki2"
        if "--propagator" in cmd:
            method = str(cmd[cmd.index("--propagator") + 1])
        steps = int(cmd[cmd.index("--trotter-steps") + 1])
        t_final = float(cmd[cmd.index("--t-final") + 1])
        err = 0.0 if method == "piecewise_exact" else 1.0 / max(1, steps**2)
        traj = [
            {"time": 0.0, "fidelity": 1.0 - err, "energy_total_trotter": err, "doublon_trotter": err},
            {"time": t_final, "fidelity": 1.0 - err, "energy_total_trotter": err, "doublon_trotter": err},
        ]
        return {
            "settings": {
                "L": 2,
                "problem": "hh",
                "ordering": "blocked",
                "t_final": t_final,
                "drive": {
                    "enabled": True,
                    "A": 0.2,
                    "omega": 0.5,
                    "tbar": 5.0,
                    "phi": 0.0,
                    "t0": 0.0,
                    "pattern": "staggered",
                    "custom_s": None,
                    "include_identity": False,
                    "time_sampling": "midpoint",
                },
            },
            "hamiltonian": {
                "coefficients_exyz": [
                    {"label_exyz": "zzzz", "coeff": {"re": 1.0, "im": 0.0}},
                ]
            },
            "trajectory": traj,
            "_run_runtime_s": 0.01,
        }

    run_efficiency_suite(cfg, run_pipeline=fake_runner)
    assert seen_pool["value"] == "paop_std"
