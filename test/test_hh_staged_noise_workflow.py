from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipelines.hardcoded.hh_staged_noise as noise_cli
import pipelines.hardcoded.hh_staged_noise_workflow as noise_wf
import pipelines.hardcoded.hh_staged_workflow as base_wf
from pipelines.hardcoded.hh_staged_noise import parse_args


def _basis(dim: int, idx: int) -> np.ndarray:
    out = np.zeros(dim, dtype=complex)
    out[int(idx)] = 1.0
    return out


def _stage_result(dim: int = 8) -> base_wf.StageExecutionResult:
    psi0 = _basis(dim, 0)
    psi1 = _basis(dim, 1)
    psi2 = _basis(dim, 2)
    psi3 = _basis(dim, 3)
    return base_wf.StageExecutionResult(
        h_poly=object(),
        hmat=np.eye(dim, dtype=complex),
        ordered_labels_exyz=["eee"],
        coeff_map_exyz={"eee": 1.0 + 0.0j},
        nq_total=int(round(np.log2(dim))),
        psi_hf=np.array(psi0, copy=True),
        psi_warm=np.array(psi1, copy=True),
        psi_adapt=np.array(psi2, copy=True),
        psi_final=np.array(psi3, copy=True),
        warm_payload={"energy": -1.0, "exact_filtered_energy": -1.1},
        adapt_payload={"energy": -1.05, "exact_gs_energy": -1.1, "stop_reason": "eps_grad"},
        replay_payload={"vqe": {"energy": -1.09}, "exact": {"E_exact_sector": -1.1}},
    )


def test_resolve_noise_defaults_and_retagged_artifacts() -> None:
    cfg = noise_wf.resolve_staged_hh_noise_config(parse_args(["--L", "2", "--skip-pdf"]))

    assert cfg.noise.methods == ("cfqm4", "suzuki2")
    assert cfg.noise.modes == ("ideal", "shots", "aer_noise")
    assert int(cfg.noise.shots) == 2048
    assert int(cfg.noise.oracle_repeats) == 4
    assert str(cfg.noise.oracle_aggregate) == "mean"
    assert cfg.noise.mitigation_config == {"mode": "none", "zne_scales": [], "dd_sequence": None}
    assert cfg.noise.symmetry_mitigation_config == {
        "mode": "off",
        "num_sites": 2,
        "ordering": "blocked",
        "sector_n_up": 1,
        "sector_n_dn": 1,
    }
    assert bool(cfg.noise.include_final_audit) is False
    assert str(cfg.staged.artifacts.tag).startswith("hh_staged_noise_")
    assert Path(cfg.staged.artifacts.output_json).name == f"{cfg.staged.artifacts.tag}.json"
    assert Path(cfg.staged.artifacts.replay_output_json).name == f"{cfg.staged.artifacts.tag}_replay.json"


def test_explicit_noise_tag_is_preserved() -> None:
    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(["--L", "2", "--skip-pdf", "--tag", "custom_noise_tag"])
    )

    assert str(cfg.staged.artifacts.tag) == "custom_noise_tag"
    assert Path(cfg.staged.artifacts.output_json).name == "custom_noise_tag.json"


def test_run_noisy_profiles_uses_final_state_and_optional_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mode_calls: list[dict[str, object]] = []
    audit_calls: list[dict[str, object]] = []

    def _fake_run_noisy_mode_isolated(*, kwargs: dict[str, object], timeout_s: int) -> dict[str, object]:
        mode_calls.append({"kwargs": kwargs, "timeout_s": timeout_s})
        return {
            "success": True,
            "trajectory": [
                {
                    "time": 0.0,
                    "energy_total_noisy": -1.0,
                    "doublon_noisy": 0.1,
                    "energy_total_delta_noisy_minus_ideal": 0.0,
                    "doublon_delta_noisy_minus_ideal": 0.0,
                }
            ],
            "delta_uncertainty": {
                "energy_total": {
                    "max_abs_delta": 0.0,
                    "max_abs_delta_over_stderr": 0.0,
                    "mean_abs_delta_over_stderr": 0.0,
                }
            },
            "benchmark_cost": {
                "term_exp_count_total": 1,
                "pauli_rot_count_total": 1,
                "cx_proxy_total": 0,
                "sq_proxy_total": 0,
                "depth_proxy_total": 1,
            },
            "benchmark_runtime": {
                "wall_total_s": 0.1,
                "oracle_eval_s_total": 0.05,
                "oracle_calls_total": 1,
            },
        }

    def _fake_run_noisy_audit_mode_isolated(*, kwargs: dict[str, object], timeout_s: int) -> dict[str, object]:
        audit_calls.append({"kwargs": kwargs, "timeout_s": timeout_s})
        return {
            "success": True,
            "final_observables": {
                "energy_total": {"noisy_mean": -1.0, "delta_mean": 0.0, "delta_stderr": 0.0},
                "doublon": {"noisy_mean": 0.1, "delta_mean": 0.0, "delta_stderr": 0.0},
            },
        }

    monkeypatch.setattr(noise_wf.noise_report, "_run_noisy_mode_isolated", _fake_run_noisy_mode_isolated)
    monkeypatch.setattr(noise_wf.noise_report, "_run_noisy_audit_mode_isolated", _fake_run_noisy_audit_mode_isolated)
    monkeypatch.setattr(
        noise_wf.noise_report,
        "_collect_noisy_benchmark_rows",
        lambda dynamics_noisy: [{"profile": "static", "method": "cfqm4", "mode": "ideal"}],
    )

    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--enable-drive",
                "--include-final-audit",
                "--noise-modes",
                "ideal,shots",
                "--noisy-methods",
                "cfqm4",
            ]
        )
    )
    stage_result = _stage_result()
    dynamics_noisy, noisy_final_audit, dynamics_benchmarks = noise_wf.run_noisy_profiles(stage_result, cfg)

    assert set(dynamics_noisy["profiles"].keys()) == {"static", "drive"}
    assert set(noisy_final_audit["profiles"].keys()) == {"static", "drive"}
    assert dynamics_benchmarks["rows"] == [{"profile": "static", "method": "cfqm4", "mode": "ideal"}]
    assert len(mode_calls) == 4
    assert len(audit_calls) == 4
    assert all(np.allclose(rec["kwargs"]["psi_seed"], stage_result.psi_final) for rec in mode_calls)
    assert all(np.allclose(rec["kwargs"]["psi_seed"], stage_result.psi_final) for rec in audit_calls)
    assert {str(rec["kwargs"]["method"]) for rec in mode_calls} == {"cfqm4"}
    assert {str(rec["kwargs"]["noise_mode"]) for rec in mode_calls} == {"ideal", "shots"}


def test_run_staged_hh_noise_merges_base_payload_and_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--noise-modes",
                "ideal",
                "--noisy-methods",
                "cfqm4",
                "--output-json",
                str(tmp_path / "hh_staged_noise.json"),
                "--output-pdf",
                str(tmp_path / "hh_staged_noise.pdf"),
            ]
        )
    )
    stage_result = _stage_result()
    calls: dict[str, object] = {}

    def _fake_run_stage_pipeline(staged_cfg: base_wf.StagedHHConfig) -> base_wf.StageExecutionResult:
        calls["stage_cfg"] = staged_cfg
        return stage_result

    monkeypatch.setattr(noise_wf.base_wf, "run_stage_pipeline", _fake_run_stage_pipeline)
    monkeypatch.setattr(
        noise_wf.base_wf,
        "run_noiseless_profiles",
        lambda stage_result_arg, staged_cfg: {
            "profiles": {
                "static": {
                    "methods": {
                        "cfqm4": {
                            "trajectory": [
                                {
                                    "time": 0.0,
                                    "energy_total_trotter": -1.0,
                                    "doublon_trotter": 0.1,
                                    "fidelity": 1.0,
                                }
                            ]
                        }
                    }
                }
            }
        },
    )
    monkeypatch.setattr(
        noise_wf,
        "run_noisy_profiles",
        lambda stage_result_arg, cfg_arg: (
            {
                "profiles": {
                    "static": {
                        "methods": {
                            "cfqm4": {
                                "modes": {
                                    "ideal": {
                                        "success": True,
                                        "trajectory": [
                                            {
                                                "time": 0.0,
                                                "energy_total_noisy": -1.0,
                                                "doublon_noisy": 0.1,
                                                "energy_total_delta_noisy_minus_ideal": 0.0,
                                                "doublon_delta_noisy_minus_ideal": 0.0,
                                            }
                                        ],
                                    }
                                }
                            }
                        }
                    }
                }
            },
            {"profiles": {}},
            {"rows": [{"profile": "static", "method": "cfqm4", "mode": "ideal"}]},
        ),
    )
    monkeypatch.setattr(
        noise_wf.base_wf,
        "assemble_payload",
        lambda **kwargs: {
            "pipeline": "hh_staged_noiseless",
            "workflow_contract": {},
            "settings": {},
            "artifacts": {
                "workflow": {
                    "output_json": str(cfg.staged.artifacts.output_json),
                    "output_pdf": str(cfg.staged.artifacts.output_pdf),
                },
                "intermediate": {
                    "adapt_handoff_json": str(cfg.staged.artifacts.handoff_json),
                    "replay_output_json": str(cfg.staged.artifacts.replay_output_json),
                },
            },
            "stage_pipeline": {
                "warm_start": {"delta_abs": 1e-1},
                "adapt_vqe": {"delta_abs": 1e-2},
                "conventional_replay": {"delta_abs": 1e-3},
            },
            "dynamics_noiseless": kwargs["dynamics_noiseless"],
        },
    )
    monkeypatch.setattr(noise_wf.base_wf, "_compute_comparisons", lambda payload: {"base_compare": True})
    monkeypatch.setattr(noise_wf.noise_report, "_compute_comparisons", lambda payload: {"noise_compare": True})

    writes: dict[str, object] = {}

    def _fake_write_json(path: Path, payload: dict[str, object]) -> None:
        writes["json_path"] = Path(path)
        writes["payload"] = payload

    monkeypatch.setattr(noise_wf.base_wf, "_write_json", _fake_write_json)
    monkeypatch.setattr(
        noise_wf,
        "write_staged_hh_noise_pdf",
        lambda payload, cfg_arg, run_command: writes.setdefault("pdf_called", True),
    )

    payload = noise_wf.run_staged_hh_noise(cfg, run_command="python pipelines/hardcoded/hh_staged_noise.py --L 2")

    assert calls["stage_cfg"] is cfg.staged
    assert payload["pipeline"] == "hh_staged_noise"
    assert payload["workflow_contract"]["noise_extension"] == "final_only_noisy_dynamics"
    assert payload["settings"]["noise"]["methods"] == ["cfqm4"]
    assert payload["dynamics_benchmarks"]["rows"] == [{"profile": "static", "method": "cfqm4", "mode": "ideal"}]
    assert payload["comparisons"] == {"base_compare": True, "noise_compare": True}
    assert writes["json_path"] == cfg.staged.artifacts.output_json
    assert bool(writes["pdf_called"]) is True


def test_noise_cli_main_print_contract(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    fake_cfg = SimpleNamespace(staged=SimpleNamespace(artifacts=SimpleNamespace(skip_pdf=True)))
    fake_payload = {
        "artifacts": {
            "workflow": {"output_json": "artifacts/json/hh_staged_noise.json", "output_pdf": "artifacts/pdf/hh_staged_noise.pdf"},
            "intermediate": {
                "adapt_handoff_json": "artifacts/json/hh_staged_noise_adapt_handoff.json",
                "replay_output_json": "artifacts/json/hh_staged_noise_replay.json",
            },
        }
    }

    monkeypatch.setattr(noise_cli, "resolve_staged_hh_noise_config", lambda args: fake_cfg)
    monkeypatch.setattr(noise_cli, "run_staged_hh_noise", lambda cfg: fake_payload)

    noise_cli.main(["--skip-pdf"])
    lines = capsys.readouterr().out.strip().splitlines()

    assert lines == [
        "workflow_json=artifacts/json/hh_staged_noise.json",
        "adapt_handoff_json=artifacts/json/hh_staged_noise_adapt_handoff.json",
        "replay_json=artifacts/json/hh_staged_noise_replay.json",
    ]
