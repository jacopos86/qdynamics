from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipelines.hardcoded.hh_staged_controller_dynamics_report import (
    load_workflow_report_entry,
    write_staged_controller_dynamics_pdf,
)


"payload(label, status) = minimal staged HH driven workflow"
def _write_workflow_artifact(
    path: Path,
    *,
    controller_mode: str,
    controller_status: str,
    include_controller_trajectory: bool,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    logs_dir = path.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    if controller_status == "env_blocked":
        stdout_lines = [
            'AI_LOG {"event":"noise_oracle_backend_scheduled_compile_start","circuit_name":"circuit-a","ts_utc":"2026-03-28T15:00:00+00:00"}\n',
            'AI_LOG {"event":"noise_oracle_backend_scheduled_compile_done","circuit_name":"circuit-a","compiled_two_qubit_count":118,"compiled_depth":318,"ts_utc":"2026-03-28T15:00:01+00:00"}\n',
            'AI_LOG {"event":"noise_oracle_backend_scheduled_compile_cache_hit","circuit_name":"circuit-a","compiled_two_qubit_count":118,"compiled_depth":318,"ts_utc":"2026-03-28T15:00:05+00:00"}\n',
        ]
        (logs_dir / "stdout.log").write_text("".join(stdout_lines), encoding="utf-8")
    else:
        (logs_dir / "stdout.log").write_text("", encoding="utf-8")
    (logs_dir / "stderr.log").write_text("", encoding="utf-8")

    controller_trajectory = []
    controller_ledger = []
    if include_controller_trajectory:
        controller_trajectory = [
            {
                "time": 0.0,
                "physical_time": 0.0,
                "fidelity_exact": 1.0,
                "abs_energy_total_error": 2.0e-2,
                "rho_miss": 0.9,
                "motion_kink_score": 0.1,
                "logical_block_count": 1,
                "runtime_parameter_count": 2,
                "selected_noisy_energy_mean": 0.82,
                "stay_noisy_energy_mean": 0.83,
                "oracle_budget_scale": 1.0,
                "oracle_confirm_limit": 2,
                "action_kind": "append",
            },
            {
                "time": 0.1,
                "physical_time": 0.1,
                "fidelity_exact": 0.99,
                "abs_energy_total_error": 1.5e-2,
                "rho_miss": 0.4,
                "motion_kink_score": 0.3,
                "logical_block_count": 2,
                "runtime_parameter_count": 4,
                "selected_noisy_energy_mean": 0.81,
                "stay_noisy_energy_mean": 0.815,
                "oracle_budget_scale": 2.0,
                "oracle_confirm_limit": 3,
                "action_kind": "stay",
            },
        ]
        controller_ledger = [
            {
                "oracle_cache_hits": 1,
                "oracle_cache_misses": 2,
                "exact_cache_hits": 3,
                "exact_cache_misses": 1,
                "geometry_memo_hits": 4,
                "geometry_memo_misses": 1,
                "raw_group_cache_hits": 5,
                "raw_group_cache_misses": 2,
            },
            {
                "oracle_cache_hits": 2,
                "oracle_cache_misses": 1,
                "exact_cache_hits": 2,
                "exact_cache_misses": 0,
                "geometry_memo_hits": 3,
                "geometry_memo_misses": 1,
                "raw_group_cache_hits": 4,
                "raw_group_cache_misses": 1,
            },
        ]

    payload = {
        "pipeline": "hh_staged_noise",
        "command": "pytest staged-controller-report",
        "settings": {
            "physics": {
                "L": 2,
                "t": 1.0,
                "u": 2.0,
                "dv": 0.0,
                "omega0": 1.0,
                "g_ep": 1.0,
                "n_ph_max": 1,
                "boundary": "periodic",
                "ordering": "blocked",
            },
            "dynamics": {
                "enable_drive": True,
                "drive_A": 0.6,
                "drive_omega": 1.0,
                "drive_tbar": 1.0,
                "drive_phi": 0.0,
                "drive_pattern": "staggered",
                "drive_time_sampling": "midpoint",
                "drive_t0": 0.0,
                "methods": ["cfqm4"],
                "t_final": 0.1,
                "num_times": 3,
                "trotter_steps": 2,
                "exact_steps_multiplier": 1,
            },
            "noise": {
                "modes": ["shots"],
                "backend_name": "FakeMarrakesh",
                "controller_noise_mode": "backend_scheduled",
            },
            "realtime_checkpoint": {
                "mode": controller_mode,
            },
        },
        "dynamics_noiseless": {
            "profiles": {
                "drive": {
                    "drive_enabled": True,
                    "drive_profile": {
                        "A": 0.6,
                        "omega": 1.0,
                        "tbar": 1.0,
                        "phi": 0.0,
                        "pattern": "staggered",
                        "custom_weights": None,
                        "time_sampling": "midpoint",
                        "t0": 0.0,
                    },
                    "methods": {
                        "cfqm4": {
                            "trajectory": [
                                {
                                    "time": 0.0,
                                    "fidelity": 1.0,
                                    "energy_total_exact": 0.80,
                                    "energy_total_trotter": 0.80,
                                    "staggered_exact": 0.00,
                                    "doublon_exact": 0.25,
                                },
                                {
                                    "time": 0.05,
                                    "fidelity": 0.995,
                                    "energy_total_exact": 0.82,
                                    "energy_total_trotter": 0.821,
                                    "staggered_exact": 0.02,
                                    "doublon_exact": 0.255,
                                },
                                {
                                    "time": 0.10,
                                    "fidelity": 0.990,
                                    "energy_total_exact": 0.84,
                                    "energy_total_trotter": 0.842,
                                    "staggered_exact": 0.03,
                                    "doublon_exact": 0.26,
                                },
                            ]
                        }
                    },
                }
            }
        },
        "dynamics_noisy": {
            "profiles": {
                "drive": {
                    "drive_enabled": True,
                    "drive_profile": {
                        "A": 0.6,
                        "omega": 1.0,
                        "tbar": 1.0,
                        "phi": 0.0,
                        "pattern": "staggered",
                        "custom_weights": None,
                        "time_sampling": "midpoint",
                        "t0": 0.0,
                    },
                    "methods": {
                        "cfqm4": {
                            "modes": {
                                "shots": {
                                    "trajectory": [
                                        {
                                            "time": 0.0,
                                            "energy_total_ideal": 0.80,
                                            "energy_total_noisy": 0.75,
                                            "energy_total_delta_noisy_minus_ideal": -0.05,
                                            "staggered_ideal": 0.00,
                                            "staggered_noisy": -0.01,
                                            "doublon_ideal": 0.25,
                                            "doublon_noisy": 0.24,
                                        },
                                        {
                                            "time": 0.05,
                                            "energy_total_ideal": 0.82,
                                            "energy_total_noisy": 0.78,
                                            "energy_total_delta_noisy_minus_ideal": -0.04,
                                            "staggered_ideal": 0.02,
                                            "staggered_noisy": 0.015,
                                            "doublon_ideal": 0.255,
                                            "doublon_noisy": 0.245,
                                        },
                                        {
                                            "time": 0.10,
                                            "energy_total_ideal": 0.84,
                                            "energy_total_noisy": 0.81,
                                            "energy_total_delta_noisy_minus_ideal": -0.03,
                                            "staggered_ideal": 0.03,
                                            "staggered_noisy": 0.025,
                                            "doublon_ideal": 0.26,
                                            "doublon_noisy": 0.255,
                                        },
                                    ]
                                }
                            }
                        }
                    },
                }
            }
        },
        "adaptive_realtime_checkpoint": {
            "mode": controller_mode,
            "status": controller_status,
            "reason": ("timeout_after_1200s" if controller_status == "env_blocked" else None),
            "reference": {
                "kind": "driven_piecewise_constant_exact_reference_from_replay_seed",
                "times": [0.0, 0.05, 0.10],
            },
            "trajectory": controller_trajectory,
            "ledger": controller_ledger,
            "summary": {
                "append_count": (1 if include_controller_trajectory else 0),
                "stay_count": (1 if include_controller_trajectory else 0),
                "oracle_decision_checkpoints": (2 if include_controller_trajectory else 0),
                "exact_decision_checkpoints": 0,
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


"entry = load_workflow_report_entry(path)"
def test_load_workflow_report_entry_round_trip(tmp_path: Path) -> None:
    workflow_path = _write_workflow_artifact(
        tmp_path / "run_on" / "workflow.json",
        controller_mode="oracle_v1",
        controller_status="completed",
        include_controller_trajectory=True,
    )

    entry = load_workflow_report_entry(workflow_path)

    assert entry.label == "oracle_v1 (completed)"
    assert entry.baseline.times.tolist() == [0.0, 0.05, 0.1]
    assert entry.noisy.energy_total_delta.tolist() == [-0.05, -0.04, -0.03]
    assert entry.controller.append_count == 1
    assert entry.controller.oracle_cache_hits == 3
    assert entry.controller.geometry_memo_hits == 7


"env_blocked entry should still expose compile/cache progress"
def test_load_workflow_report_entry_env_blocked_uses_compile_log_summary(tmp_path: Path) -> None:
    workflow_path = _write_workflow_artifact(
        tmp_path / "run_env_blocked" / "workflow.json",
        controller_mode="oracle_v1",
        controller_status="env_blocked",
        include_controller_trajectory=False,
    )

    entry = load_workflow_report_entry(workflow_path)

    assert entry.controller.status == "env_blocked"
    assert entry.controller.time_axis.size == 0
    assert entry.controller.compile_summary.compile_start_count == 1
    assert entry.controller.compile_summary.compile_done_count == 1
    assert entry.controller.compile_summary.compile_cache_hit_count == 1
    assert entry.controller.compile_summary.mean_two_qubit_count == pytest.approx(118.0)


"pdf = write_staged_controller_dynamics_pdf(entries)"
def test_write_staged_controller_dynamics_pdf(tmp_path: Path) -> None:
    on_path = _write_workflow_artifact(
        tmp_path / "run_on" / "workflow.json",
        controller_mode="oracle_v1",
        controller_status="completed",
        include_controller_trajectory=True,
    )
    off_path = _write_workflow_artifact(
        tmp_path / "run_off" / "workflow.json",
        controller_mode="off",
        controller_status="disabled",
        include_controller_trajectory=False,
    )
    out_pdf = tmp_path / "driven_controller_report.pdf"

    result = write_staged_controller_dynamics_pdf(
        input_jsons=[on_path, off_path],
        output_pdf=out_pdf,
        run_command="pytest staged-controller-report",
    )

    assert result == out_pdf.resolve()
    assert out_pdf.exists()
    assert out_pdf.stat().st_size > 0
